"""Product `sae_self` (P1 of the autointerp design): the target feature's PER-TOKEN activation on
its own MAEMM rollouts.

    <root>/maemms/<base>/<maemm>/scores/<set>__<engine>/sae_self/
        sae_self.f16  [n_sae, n, T]   pre-gate activation of THE TARGET feature at every scored
                                      token of its own rollouts, NaN outside `keep` (column 0 is
                                      the sink and is always NaN, i.e. the sink is dropped)
        sae_self_ids.i32 [n_sae, n, T] the SCORED ids of the same grid, -1 outside `keep`
        sae_self.json                 per-target fire fraction, activation summaries, and the
                                      agreement checks against the stored cos / argmax / SAE CSR

`score`'s stored SAE arrays are a CSR at the ARGMAX TOKEN ONLY (precompute/score.py:8-9, 80-82):
all 131k features, gated, one token per rollout. An autointerp example set needs the opposite cut
-- ONE feature, every token -- so this product makes it, for the `sae` family only (512 of the
set's 1,536 targets), at one thirtieth of the rows a full per-token CSR would cost.

It is `score` minus the CSR: the SAME clean-base forward through `common.score_tokens`, the same
truncation, chunk, sink and fp32 cosine, with an `on_chunk` callback that encodes ONE SAE feature
per row instead of all of them at one token. The cosine is computed too (the directions are the
unit encoder columns, exactly as `targets` built them) and is only used for the three checks below,
which is what makes this product self-validating:

  1. our argmax == the stored `argmax.i16` on every rollout (same forward, same protocol);
  2. our per-token activation AT that argmax == the stored CSR's `sae_val` for this feature
     wherever the CSR holds it, to f16 tolerance;
  3. the CSR holds this feature at the argmax exactly when our activation there exceeds the gate.

A disagreement is REPORTED in sae_self.json and raised, never smoothed: it would mean the rollout
file, the scores directory and this pass are not the same run.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import precompute.common as C

# Rows handed to common.score_tokens per call, as precompute/score.py:50. It re-chunks internally
# at common.SCORE_CHUNK, so this only bounds the fp32 residual the callback sees at once.
SCORE_ROWS = 256
FAMILY = "sae"
# Largest cosine gap at which a disagreement with the stored `argmax.i16` is accepted as
# right-padding noise rather than a different run. MEASURED 2026-09-16 (see CHECK 1 below): the
# real gaps are ~1e-5 while a genuinely different token is 1e-2 to 1e-1 away, so 1e-3 separates
# them by two orders of magnitude on both sides.
ARGMAX_TIE_TOL = 1e-3


class _SelfAct:
    """Per-token pre-gate activation of ONE feature per row, off the scoring forward itself.

    `feats` is the per-row SAE feature id, so the callback gathers a different encoder column for
    every row of the chunk: `relu((h - b_dec) . W_enc[:, f_row] + b_enc[f_row])`, which is
    `common.sae_encode` restricted to one feature and broadcast over rows.
    """

    def __init__(self, sae, feats, n_rows: int):
        import torch

        self.sae = sae
        self.feats = torch.as_tensor(feats, dtype=torch.long)
        assert len(self.feats) == n_rows, f"{len(self.feats)} feature ids for {n_rows} rows"
        self.act = torch.full((n_rows, C.SCORE_WIDTH), float("nan"))
        self.ids = torch.full((n_rows, C.SCORE_WIDTH), -1, dtype=torch.long)
        self.arg = torch.full((n_rows,), -1, dtype=torch.int64)
        self.off = 0  # global row of the current score_tokens call

    def __call__(self, s, h, cos, keep, ids):
        import torch

        b = h.shape[0]
        g = self.off + s
        f = self.feats[g : g + b].to(h.device)
        w = self.sae.W_enc[:, f].T  # [b, d]
        be = self.sae.b_enc[f]  # [b]
        a = torch.relu(torch.einsum("btd,bd->bt", h - self.sae.b_dec, w) + be.unsqueeze(1))
        a = torch.where(keep, a, torch.full_like(a, float("nan")))
        _best, arg = C.agg(cos, keep)
        has = keep.any(dim=1)
        t = a.shape[1]
        self.act[g : g + b, :t] = a.cpu()
        self.ids[g : g + b, :t] = torch.where(keep, ids, torch.full_like(ids, -1)).cpu()
        # `arg` indexes the padded grid (column 0 = sink); score.py stores arg - 1, i.e. the index
        # among the SCORED tokens, and -1 for a row with nothing kept. Store it the same way.
        self.arg[g : g + b] = torch.where(has, arg - 1, torch.full_like(arg, -1)).cpu()


def _sae_rows(cfg, args):
    """(rows_meta, sae rows of the set, their feature ids, the SAE key)."""
    base, root, set_name = args["base"], args["root"], args["heldout"]
    rows = C.read_jsonl(f"{C.heldout_dir(base, set_name, root)}/ids.jsonl")
    sel = [r for r in rows if r["family"] == FAMILY]
    assert sel, f"held-out set {set_name!r} on {base} has no {FAMILY!r} family rows"
    keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    assert len(keys) == 1, f"base {base} has {len(keys)} SAEs in config, expected exactly 1"
    return rows, [r["row"] for r in sel], [int(r["id"]) for r in sel], keys[0]


def _csr_at_argmax(sdir: str, n_targets: int, n: int, flat_rows, feats_of_row, gate: float):
    """(csr_val, csr_has) for each (target, rollout): the stored CSR's activation of THIS target's
    feature at the argmax token, and whether the CSR holds it at all.

    `sae_off.i64` has one offset per flat (target, rollout) row of the WHOLE set in row-major
    order, so row `r * n + k` -- the same flattening score.py used to write it.
    """
    off = C.read_array(f"{sdir}/sae_off.i64", "int64", (n_targets * n + 1,))
    idx = C.read_array(f"{sdir}/sae_idx.i32", "int32", (int(off[-1]),))
    val = C.read_array(f"{sdir}/sae_val.f16", "float16", (int(off[-1]),))
    out_val = np.zeros(len(flat_rows), dtype=np.float32)
    out_has = np.zeros(len(flat_rows), dtype=bool)
    for i, (flat, feat) in enumerate(zip(flat_rows, feats_of_row, strict=True)):
        lo, hi = int(off[flat]), int(off[flat + 1])
        if hi <= lo:
            continue
        j = np.searchsorted(idx[lo:hi], feat)
        if j < hi - lo and int(idx[lo + j]) == feat:
            out_has[i] = True
            out_val[i] = float(val[lo + j])
    assert not out_has.any() or float(out_val[out_has].min()) > gate, (
        "the stored CSR holds an entry at or below the gate, which score.py cannot have written"
    )
    return out_val, out_has


def run(cfg, args):
    import torch

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base and maemm, "product sae_self needs --base and --maemm"
    assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]
    engine = args.get("engine") or "vllm"

    rows_meta, sae_rows, feats, sae_key = _sae_rows(cfg, args)
    sel = [r for r in C.parse_rows(args.get("rows", ""), len(rows_meta)) if r in set(sae_rows)]
    assert sel, (
        f"--rows {args.get('rows', '')!r} selected none of the {len(sae_rows)} {FAMILY} rows "
        f"({sae_rows[0]}..{sae_rows[-1]})"
    )
    feat_of = dict(zip(sae_rows, feats, strict=True))

    rpath = C.rollouts_path(maemm, set_name, root, engine)
    sdir = C.scores_dir(maemm, set_name, root, engine)
    assert os.path.exists(rpath), f"no rollouts at {rpath}"
    recs = C.read_jsonl(rpath)
    with open(f"{C.rollouts_dir(maemm, root)}/{C.rollout_stem(set_name, engine)}.summary.json") as fh:
        rsum = json.load(fh)
    n = int(rsum["n"])
    by_row: dict[int, dict[int, dict]] = {}
    for r in recs:
        if r["row"] in feat_of:
            by_row.setdefault(r["row"], {})[r["k"]] = r
    for r in sel:
        ks = sorted(by_row.get(r, {}))
        assert ks == list(range(n)), (
            f"target row {r} has rollouts {ks[:4]}..{ks[-4:]} but the summary says n={n}: the "
            f"[N, n, T] arrays need a complete, gap-free rollout grid"
        )

    flat = [by_row[r][k] for r in sel for k in range(n)]
    texts = [x["text"] for x in flat]
    row_feats = [feat_of[x["row"]] for x in flat]

    model, tok = C.load_base(cfg, base)  # CLEAN BASE ONLY, exactly as `score`
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32)
    gate = float(sae.threshold)
    dirs = C.sae_dirs(sae, row_feats).cpu()
    extra = _SelfAct(sae, row_feats, len(flat))
    print(
        f"[sae_self] {len(sel)} sae targets x {n} rollouts = {len(flat)} rows, gate {gate:.4f}",
        flush=True,
    )
    t0 = time.time()
    cos_parts = []
    for s in range(0, len(texts), SCORE_ROWS):
        block = texts[s : s + SCORE_ROWS]
        extra.off = s
        out = C.score_tokens(model, tok, block, dirs[s : s + len(block)], read_layer, on_chunk=extra)
        cos_parts.append(out["cos"])
    cos = torch.cat(cos_parts)
    elapsed = time.time() - t0

    N = len(sel)
    act = extra.act.numpy().reshape(N, n, C.SCORE_WIDTH)
    ids = extra.ids.numpy().astype(np.int32).reshape(N, n, C.SCORE_WIDTH)
    arg = extra.arg.numpy().reshape(N, n)

    # ---- the three checks -------------------------------------------------------------------
    # CHECK 1 is NOT an equality: `score` scored all 1,536 targets of the set in one flat order,
    # this pass scores only the 512 sae rows, so the two runs' SCORE_CHUNK batches have different
    # compositions and therefore different right-padding widths. MEASURED 2026-09-16 on
    # rlI-150: that moves a per-token cosine by ~1e-4 and flips the argmax on 3 of 32,768 rollouts,
    # every one of them a near-tie -- which is checklist item 11 ("right-padding chunk size
    # measurably shifts per-row cosine") showing up exactly where it was predicted to. So a
    # mismatch is allowed ONLY when our own cosine at the two positions is within ARGMAX_TIE_TOL,
    # and the count and the worst gap are reported either way.
    stored_arg = C.read_array(f"{sdir}/argmax.i16", "int16", (len(rows_meta), n)).astype(np.int64)
    stored_arg = stored_arg[np.asarray(sel)]
    ours_cos = cos.numpy().reshape(N, n, C.SCORE_WIDTH)
    arg_agree = int((stored_arg == arg).sum())
    arg_total = int(arg.size)
    diff = stored_arg != arg
    tie_gap = np.zeros_like(stored_arg, dtype=np.float64)
    if diff.any():
        a_ours = np.take_along_axis(ours_cos, np.clip(arg, 0, None)[:, :, None] + 1, 2)[:, :, 0]
        a_stored = np.take_along_axis(ours_cos, np.clip(stored_arg, 0, None)[:, :, None] + 1, 2)[:, :, 0]
        tie_gap = np.abs(np.nan_to_num(a_ours) - np.nan_to_num(a_stored))
    worst_tie = float(tie_gap[diff].max()) if diff.any() else 0.0
    argmax_ok = bool(worst_tie <= ARGMAX_TIE_TOL)
    stored_cos = C.read_array(f"{sdir}/cos.f16", "float16", (len(rows_meta), n, C.SCORE_WIDTH))
    stored_cos = stored_cos[np.asarray(sel)]
    both = np.isfinite(stored_cos) & np.isfinite(ours_cos)
    cos_diff = np.abs(stored_cos[both].astype(np.float32) - ours_cos[both])
    cos_max_abs = float(cos_diff.max()) if both.any() else 0.0

    # CHECKS 2 and 3 index OUR activations at the STORED argmax, because that is the token the
    # stored CSR was measured at: they are then exact whatever check 1 found.
    flat_rows = [x["row"] * n + x["k"] for x in flat]
    csr_val, csr_has = _csr_at_argmax(sdir, len(rows_meta), n, flat_rows, row_feats, gate)
    csr_val = csr_val.reshape(N, n)
    csr_has = csr_has.reshape(N, n)
    at_arg = np.take_along_axis(act, np.clip(stored_arg, 0, None)[:, :, None] + 1, 2)[:, :, 0]
    ours_at_arg = np.where(stored_arg >= 0, at_arg, np.nan)
    ours_fired_at_arg = np.isfinite(ours_at_arg) & (ours_at_arg > gate)
    # f16 tolerance: the CSR stores float16, so a value near 40 carries ~0.03 of quantisation.
    tol = np.maximum(np.abs(csr_val) * 1e-3, 1e-2)
    val_bad = int((csr_has & (np.abs(np.nan_to_num(ours_at_arg) - csr_val) > tol)).sum())
    has_bad = int((csr_has != ours_fired_at_arg).sum())
    val_worst = (
        float(np.abs(np.nan_to_num(ours_at_arg) - csr_val)[csr_has].max()) if csr_has.any() else 0.0
    )

    checks = {
        "argmax_agreement": f"{arg_agree}/{arg_total}",
        "argmax_mismatches": arg_total - arg_agree,
        "argmax_mismatch_worst_cos_gap": round(worst_tie, 8),
        "argmax_tie_tol": ARGMAX_TIE_TOL,
        "argmax_ok": argmax_ok,
        "cos_max_abs_diff_vs_stored": round(cos_max_abs, 6),
        "csr_value_mismatches": val_bad,
        "csr_value_worst_abs_diff": round(val_worst, 6),
        "csr_membership_mismatches": has_bad,
        "csr_entries_present": int(csr_has.sum()),
    }

    # ---- per-target summary -----------------------------------------------------------------
    peak = np.nanmax(np.where(np.isfinite(act), act, -np.inf), axis=2)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    fired = peak > gate
    corpus_peak = C.read_array(f"{C.sae_dir(sae_key, root)}/max_act.f16", "float16", (sae.d_sae,))
    per_target = []
    for i, r in enumerate(sel):
        f = feat_of[r]
        per_target.append(
            {
                "row": r,
                "feature": f,
                "n": n,
                "corpus_peak": round(float(corpus_peak[f]), 4),
                "fire_fraction": round(float(fired[i].mean()), 4),
                "fire_fraction_at_argmax": round(float(ours_fired_at_arg[i].mean()), 4),
                "mean_peak_act": round(float(peak[i].mean()), 4),
                "max_peak_act": round(float(peak[i].max()), 4),
                "n_tok_mean": round(float(np.mean([by_row[r][k]["n_tok"] for k in range(n)])), 2),
            }
        )

    # `--out-suffix` keeps a 2-feature shakeout out of the canonical path (score.py's
    # `--score-name` does the same job): the full run then writes `sae_self/` with nothing to
    # --force over.
    out = f"{sdir}/sae_self{args.get('out_suffix') or ''}"
    inputs = {
        "rollouts": rpath,
        "scores": sdir,
        "engine": engine,
        "maemm": maemm,
        "sae": sae_key,
        "gate": gate,
        "targets": f"{N} of {len(sae_rows)} {FAMILY} rows",
        "n": n,
    }
    with C.outdir(out, args, inputs=inputs) as od:
        od.write_array("sae_self.f16", act, "float16")
        od.write_array("sae_self_ids.i32", ids, "int32")
        od.write_json(
            "sae_self.json",
            {
                "rows": sel,
                "features": [feat_of[r] for r in sel],
                "n": n,
                "gate": gate,
                "checks": checks,
                "fire_fraction_mean": round(float(fired.mean()), 4),
                "per_target": per_target,
                "seconds": round(elapsed, 1),
            },
        )
        od.note(
            f"`sae_self.f16` is [{N}, {n}, {C.SCORE_WIDTH}]: the PRE-GATE activation "
            f"relu((h - b_dec).W_enc[:, f] + b_enc[f]) of THE TARGET feature f of each row, at "
            f"every scored token of its own rollout, from the SAME clean-base forward "
            f"common.score_tokens runs for `score`. NaN outside `keep`; column 0 is the sink and "
            f"is therefore always NaN (the sink is dropped, as everywhere else)."
        )
        od.note(
            "`sae_self_ids.i32` is the SCORED id grid of the same shape (-1 outside `keep`), "
            "stored because a rollout's stored `ids` are the SAMPLER's, and the scorer re-encodes "
            "the decoded text -- a rollout cut mid-word re-encodes to a different number of ids "
            "(MEASURED 2026-09-15, precompute/score.py:145-150). Rendering a marked example needs "
            "the ids the activations were measured on, not the sampler's."
        )
        od.note(
            f"CHECK 1, argmax: ours == the stored `argmax.i16` on {arg_agree}/{arg_total} rows; "
            f"the {arg_total - arg_agree} that differ are near-ties, worst cosine gap "
            f"{worst_tie:.2e} (tolerance {ARGMAX_TIE_TOL:.0e}, asserted). `score` batched all "
            f"1,536 targets and this pass batches only the 512 sae rows, so the two runs' "
            f"SCORE_CHUNK right-padding widths differ -- checklist item 11. CHECK 2, cosine: "
            f"max |ours - stored cos.f16| = {cos_max_abs:.2e} over "
            f"every kept token (the directions here are `common.sae_dirs`, i.e. what `targets` "
            f"drew, so this reproduces the stored cosine and not merely something like it)."
        )
        od.note(
            f"CHECKS 2 and 3 index OUR activations at the STORED argmax (the token the CSR was "
            f"measured at), so they are exact whatever CHECK 1 found. The stored CSR at that "
            f"token: {int(csr_has.sum())} of {arg_total} "
            f"(target, rollout) rows hold THIS feature above the gate {gate:.4f}; "
            f"{val_bad} value mismatches (worst |diff| {val_worst:.4f}, f16 tolerance) and "
            f"{has_bad} membership mismatches (asserted zero)."
        )
        od.note(
            f"fire fraction (this feature exceeds the gate at SOME token of the rollout) = "
            f"{float(fired.mean()):.4f} over all {arg_total} rollouts; at the argmax token alone "
            f"it is {float(ours_fired_at_arg.mean()):.4f}. Per target in sae_self.json -- this is "
            f"the covariate the autointerp design uses to separate the hard stratum."
        )
        od.note(
            f"scoring wall {elapsed:.1f}s for {len(flat)} rows "
            f"({len(flat) / max(elapsed, 1e-9):.1f} rows/s)"
        )

    assert argmax_ok, (
        f"CHECK 1 FAILED: our argmax differs from the stored argmax.i16 on "
        f"{arg_total - arg_agree} of {arg_total} rollouts, and the worst of those is a cosine gap "
        f"of {worst_tie:.2e}, ABOVE the {ARGMAX_TIE_TOL:.0e} near-tie tolerance -- so it is not "
        f"batch-composition noise. {rpath} and {sdir} may not be the same run. Values written to "
        f"{out} for inspection."
    )
    assert val_bad == 0 and has_bad == 0, (
        f"CHECK 2/3 FAILED: {val_bad} value and {has_bad} membership mismatches against the stored "
        f"SAE CSR at the argmax token (worst |diff| {val_worst:.4f}). Written to {out}."
    )
    return {
        "out": out,
        "targets": N,
        "n": n,
        "rows": len(flat),
        "gate": gate,
        "fire_fraction_mean": round(float(fired.mean()), 4),
        "checks": checks,
        "seconds": round(elapsed, 1),
    }
