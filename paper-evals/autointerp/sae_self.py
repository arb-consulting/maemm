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
-- ONE feature, every token -- so this product makes it, for the SAE families only (`sae`, and
Ari's `sae2m_enc` label for the 2M dictionary -- see FAMILIES below; 512 of the
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
# The held-out families whose targets ARE SAE features, so a "the feature's own activation on
# this text" arm is meaningful for them. `sae` is the 131k `l42-1b` draw (config.yaml's
# 2026-09-16_v1); `sae2m_enc` is the label Ari's `features/draw_sae2m.py` writes for the 2M-SAE
# encoder columns -- a different dictionary and a different draw, but the same KIND of target
# (row["id"] is the feature index either way), which is all anything here needs. Kept as a tuple
# rather than collapsed to one name because a set can carry both and the labels are provenance.
FAMILIES = C.SAE_FAMILIES
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

    def __init__(self, sae, feats, n_rows: int, width: int = C.SCORE_WIDTH):
        import torch

        self.sae = sae
        self.feats = torch.as_tensor(feats, dtype=torch.long)
        assert len(self.feats) == n_rows, f"{len(self.feats)} feature ids for {n_rows} rows"
        # `width` is the scored directory's, read with common.score_width_of: the NLA arm scores
        # at 256 tokens rather than the protocol's 95, and these arrays must line up with its
        # cos.f16.
        self.width = int(width)
        self.act = torch.full((n_rows, self.width), float("nan"))
        self.ids = torch.full((n_rows, self.width), -1, dtype=torch.long)
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
    """(rows_meta, sae rows of the set, their feature ids, the SAE key).

    The selector filters on the ROW's own `sae_key`, not on the family label. A set can carry two
    dictionaries under one `family: sae` label (features/draw_sae2m.py writes the key per row), and
    every feature id below 131,072 is a VALID index into a 2^21 encoder -- so the family-only
    filter would look the 131k block's ids up in the 2M dictionary and score 512 wrong features
    with nothing raising. `common.sae_rows_of` is the rule, applied where the key is resolved.

    Decoder rows are skipped: this stage cross-checks its activations against `vecs.f16`, which is
    the ENCODER column, and the activation of feature f is its encoder readout whichever direction
    was injected. The `sae_side: dec` block is scored by `score`, not here.
    """
    base, root, set_name = args["base"], args["root"], args["heldout"]
    rows = C.read_jsonl(f"{C.heldout_dir(base, set_name, root)}/ids.jsonl")
    # WHICH SAE: `--sae` when the base carries more than one (qwen36-27b does, since sae2m).
    # common.sae_key_for is the same rule score._sae_for uses, so the stage and the scorer it
    # validates itself against cannot end up on different dictionaries.
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
    hdir = C.heldout_dir(base, set_name, root)
    sel = C.sae_rows_of(
        rows, sae_key, FAMILIES, side="enc",
        declared=C.declared_sae_key(cfg, hdir, root), where=hdir,
    )
    assert sel, (
        f"held-out set {set_name!r} on {base} has no encoder rows of dictionary {sae_key!r} in the "
        f"SAE families {FAMILIES}; it carries families "
        f"{sorted({r['family'] for r in rows})} and dictionaries "
        f"{sorted({r.get('sae_key', '(unkeyed)') for r in rows if r['family'] in FAMILIES})}"
    )
    return rows, [r["row"] for r in sel], [int(r["id"]) for r in sel], sae_key


def _csr_at_argmax(sdir: str, n_targets: int, n: int, flat_rows, feats_of_row, gate: float):
    """(csr_val, csr_has) for each (target, rollout): the stored CSR's activation of THIS target's
    feature at the argmax token, and whether the CSR holds it at all.

    `sae_off.i64` has one offset per flat (target, rollout) row of the SCORED grid in row-major
    order -- the same flattening score.py used to write it. `n_targets` is therefore the number
    of rows in that directory's `rows.json`, NOT the size of the held-out set, and `flat_rows`
    must be built from a row's INDEX among the scored rows. The two coincide only when the whole
    set was scored.
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


def scored_rows_of(sdir: str, n: int, sel, where: str = "the rollouts"):
    """(score_rows, score_ix, sel_ix) for a `scores/<set>/` directory. The row-restriction fix.

    `score --rows` writes its arrays over the SELECTED targets only -- [N_sel, n, T], with
    `rows.json` naming them IN ORDER -- so a consumer that indexes them by the held-out set's own
    row numbers is right only when the whole set was scored. That held for the pilot (all 512
    `sae` rows) and MEASURED 2026-09-21 it does not for a row-restricted score: the 2k set's
    8-row smoke died on `cannot reshape array of size 32 into shape (2000, 4)`.

    `sel_ix` is the position of each selected row among the SCORED rows, which is the index every
    stored array wants. When the whole set was scored it is `sel` itself, so the full-set case is
    bit-identical.
    """
    with open(f"{sdir}/rows.json") as fh:
        rows_json = json.load(fh)
    score_rows = [int(x) for x in rows_json["rows"]]
    score_ix = {r: i for i, r in enumerate(score_rows)}
    assert int(rows_json["n"]) == n, (
        f"{sdir}/rows.json was written at n={rows_json['n']} but {where} says n={n}: the scores "
        f"and the rollouts are not the same run"
    )
    missing = [r for r in sel if r not in score_ix]
    assert not missing, (
        f"{sdir} holds {len(score_rows)} scored targets ({score_rows[:4]}...{score_rows[-4:]}) "
        f"and does NOT hold rows {missing[:8]}: `score` was run with a narrower --rows than this "
        f"stage was. Re-run `--product score` over these rows, or restrict --rows here."
    )
    return score_rows, score_ix, np.asarray([score_ix[r] for r in sel])


def run(cfg, args):
    import torch

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    # D11: `--rollouts-dir <dir>` reads <dir>/rollouts.jsonl + <dir>/scores/ instead of a MAEMM's,
    # mirroring score.py:352-358. That is what lets a patchscopes cell, a GCG/EPO finals file, a
    # corpus-search result or any other non-MAEMM text be scored for the target feature's own
    # activation -- every arm of evals 1 and 2 that has no `maemms:` entry needs it.
    rdir = (args.get("rollouts_dir") or "").rstrip("/")
    assert base, "product sae_self needs --base"
    assert maemm or rdir, "product sae_self needs --maemm (whose rollouts it reads), or --rollouts-dir"
    assert not maemm or maemm in cfg["maemms"], (
        f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    )
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]
    engine = args.get("engine") or "vllm"

    rows_meta, sae_rows, feats, sae_key = _sae_rows(cfg, args)
    sel = [r for r in C.parse_rows(args.get("rows", ""), len(rows_meta)) if r in set(sae_rows)]
    assert sel, (
        f"--rows {args.get('rows', '')!r} selected none of the {len(sae_rows)} {'/'.join(FAMILIES)} rows "
        f"({sae_rows[0]}..{sae_rows[-1]})"
    )
    feat_of = dict(zip(sae_rows, feats, strict=True))

    tag = args.get("run_tag") or ""
    if rdir:
        rpath, spath = f"{rdir}/rollouts.jsonl", f"{rdir}/rollouts.summary.json"
        sdir = f"{rdir}/scores"
    else:
        rpath = C.rollouts_path(maemm, set_name, root, engine, tag)
        spath = f"{C.rollouts_dir(maemm, root)}/{C.rollout_stem(set_name, engine, tag)}.summary.json"
        sdir = C.scores_dir(maemm, args.get("score_name") or set_name, root, engine)
    assert os.path.exists(rpath), f"no rollouts at {rpath}"
    assert os.path.exists(sdir), (
        f"no scores at {sdir}: this stage re-runs `score`'s forward and cross-checks itself "
        f"against what that run stored, so `score` must have run on these rollouts first"
    )
    recs = C.read_jsonl(rpath)
    with open(spath) as fh:
        rsum = json.load(fh)
    if rdir:
        engine = rsum.get("engine", engine)  # the directory names its own producer
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

    # Which rows that scores directory actually holds -- see `scored_rows_of`.
    score_rows, score_ix, sel_ix = scored_rows_of(sdir, n, sel, rpath)
    # This pass must reproduce `score`'s forward token for token, so it scores at the SAME
    # re-encode truncation that directory was written at -- the protocol's 95 everywhere but the
    # NLA arm, which asks for 256 (common.score_width_of reads it out of rows.json).
    width = C.score_width_of(sdir)
    max_length = width - 1
    model, tok = C.load_base(cfg, base)  # CLEAN BASE ONLY, exactly as `score`
    # ENCODER ONLY: every activation here goes through common.sae_encode (b_dec, W_enc, b_enc)
    # and nothing reads W_dec, which at 2^21 features is another 43 GB in fp32 and does not fit
    # an H200 beside the 27B (features/CHANGES.md fix 2, made there for `stats`).
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32, need_decoder=False)
    gate = float(sae.threshold)
    dirs = C.sae_dirs(sae, row_feats).cpu()
    extra = _SelfAct(sae, row_feats, len(flat), width)
    print(
        f"[sae_self] {len(sel)} sae targets x {n} rollouts = {len(flat)} rows, gate {gate:.4f}, "
        f"scoring window {max_length} (T={width})",
        flush=True,
    )
    t0 = time.time()
    cos_parts = []
    for s in range(0, len(texts), SCORE_ROWS):
        block = texts[s : s + SCORE_ROWS]
        extra.off = s
        out = C.score_tokens(
            model, tok, block, dirs[s : s + len(block)], read_layer, on_chunk=extra, max_length=max_length
        )
        cos_parts.append(out["cos"])
    cos = torch.cat(cos_parts)
    elapsed = time.time() - t0

    N = len(sel)
    act = extra.act.numpy().reshape(N, n, width)
    ids = extra.ids.numpy().astype(np.int32).reshape(N, n, width)
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
    stored_arg = C.read_array(f"{sdir}/argmax.i16", "int16", (len(score_rows), n)).astype(np.int64)
    stored_arg = stored_arg[sel_ix]
    ours_cos = cos.numpy().reshape(N, n, width)
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
    stored_cos = C.read_array(f"{sdir}/cos.f16", "float16", (len(score_rows), n, width))
    stored_cos = stored_cos[sel_ix]
    both = np.isfinite(stored_cos) & np.isfinite(ours_cos)
    cos_diff = np.abs(stored_cos[both].astype(np.float32) - ours_cos[both])
    cos_max_abs = float(cos_diff.max()) if both.any() else 0.0

    # CHECKS 2 and 3 index OUR activations at the STORED argmax, because that is the token the
    # stored CSR was measured at: they are then exact whatever check 1 found.
    # The CSR is flattened over the SCORED grid, not the set's own row numbering.
    # A scores directory written with `--no-sae` holds EMPTY sae_idx/sae_val arrays (score.py's
    # collector returns before touching the SAE, :87-88). Checks 2 and 3 then compare our real
    # firings against an all-False CSR and trip -- after the paid forward -- or, if nothing fires
    # at any argmax, pass VACUOUSLY, which destroys the cross-check the stage exists for. Neither
    # is a check. So the absence is detected and the two checks are SKIPPED with the reason
    # recorded in `checks`, never relaxed (the critique's B11). `--no-sae` is legitimate on a
    # non-MAEMM arm whose CSR nothing reads; what is not legitimate is reporting a check that did
    # not happen.
    with open(f"{sdir}/index.json") as fh:
        sindex = json.load(fh)
    csr_bytes = int(sindex.get("sae_idx.i32", {}).get("bytes", 0))
    has_csr = csr_bytes > 0
    flat_rows = [score_ix[x["row"]] * n + x["k"] for x in flat]
    if has_csr:
        csr_val, csr_has = _csr_at_argmax(sdir, len(score_rows), n, flat_rows, row_feats, gate)
        csr_val = csr_val.reshape(N, n)
        csr_has = csr_has.reshape(N, n)
    else:
        csr_val = np.zeros((N, n), dtype=np.float32)
        csr_has = np.zeros((N, n), dtype=bool)
    at_arg = np.take_along_axis(act, np.clip(stored_arg, 0, None)[:, :, None] + 1, 2)[:, :, 0]
    ours_at_arg = np.where(stored_arg >= 0, at_arg, np.nan)
    ours_fired_at_arg = np.isfinite(ours_at_arg) & (ours_at_arg > gate)
    # f16 tolerance: the CSR stores float16, so a value near 40 carries ~0.03 of quantisation.
    tol = np.maximum(np.abs(csr_val) * 1e-3, 1e-2)
    val_bad = int((csr_has & (np.abs(np.nan_to_num(ours_at_arg) - csr_val) > tol)).sum())
    has_bad = int((csr_has != ours_fired_at_arg).sum()) if has_csr else 0
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
        "csr_checked": has_csr,
        **(
            {}
            if has_csr
            else {
                "csr_skipped_reason": (
                    f"{sdir}/sae_idx.i32 is empty, so that scores run was made with --no-sae and "
                    f"holds no SAE CSR. CHECKS 2 and 3 were SKIPPED, not relaxed: against an "
                    f"all-False CSR they would either trip on every real firing or pass vacuously "
                    f"if nothing fired. Re-run `score` WITH --sae on these rollouts to get them."
                )
            }
        ),
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
        "maemm": maemm or f"(none: --rollouts-dir {rdir})",
        "sae": sae_key,
        "gate": gate,
        "targets": f"{N} of {len(sae_rows)} {'/'.join(FAMILIES)} rows",
        "n": n,
    }
    # The checks run BEFORE the product is written, so a failed check never renames a bad product
    # onto the canonical path. SMOKES records two full 512x64 runs that failed the argmax check;
    # under the old order those had already been renamed by the time the assert fired, and `build`
    # reads the product without looking at `checks`.
    assert argmax_ok, (
        f"CHECK 1 FAILED: our argmax differs from the stored argmax.i16 on "
        f"{arg_total - arg_agree} of {arg_total} rollouts, and the worst of those is a cosine gap "
        f"of {worst_tie:.2e}, ABOVE the {ARGMAX_TIE_TOL:.0e} near-tie tolerance -- so it is not "
        f"batch-composition noise. {rpath} and {sdir} may not be the same run. Values written to "
        f"{out} for inspection."
    )
    if has_csr:
        assert val_bad == 0 and has_bad == 0, (
            f"CHECK 2/3 FAILED: {val_bad} value and {has_bad} membership mismatches against the "
            f"stored SAE CSR at the argmax token (worst |diff| {val_worst:.4f}). Written to {out}."
        )
    else:
        print(f"[sae_self] CHECKS 2/3 SKIPPED: {checks['csr_skipped_reason']}", flush=True)

    with C.outdir(out, args, inputs=inputs) as od:
        od.write_array("sae_self.f16", act, "float16")
        od.write_array("sae_self_ids.i32", ids, "int32")
        od.write_json(
            "sae_self.json",
            {
                "rows": sel,
                "features": [feat_of[r] for r in sel],
                "n": n,
                # the stored [N, n, width] third dimension -- autointerp/build.py reads it back
                # rather than assuming the protocol's SCORE_WIDTH
                "width": width,
                "gate": gate,
                "checks": checks,
                "fire_fraction_mean": round(float(fired.mean()), 4),
                "per_target": per_target,
                "seconds": round(elapsed, 1),
            },
        )
        od.note(
            f"`sae_self.f16` is [{N}, {n}, {width}]: the PRE-GATE activation "
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


# ==============================================================================================
# Product `random_pool`: a larger shared negative pool, with PER-TOKEN activations
# ==============================================================================================
#
# AMENDMENT 2026-09-16 (coordinator). The shared `_random256` pool that `scan` wrote cannot serve
# this evaluation: it is 256 windows carrying only a per-feature MAXIMUM, and for the densest
# tested features fewer than 20 of those 256 have a maximum of exactly 0, so the design's
# zero-activation negative rule runs out. 2,048 windows fixes that, and storing per-token
# activations makes two more things possible for free: near-miss negatives (0 < max <= gate) and
# fuzzing marks on any negative that has real activations.
#
#     <root>/base/<base>/sae/<sae>/random_pool/<set>/
#         windows.jsonl   2048 rows: window index, doc, start, len
#         max_act.f16     [n_feat, n_win]   per-feature per-window maximum, pre-gate
#         tok_off.i64     [n_feat * n_win + 1]  CSR offsets, FEATURE-MAJOR (row = f_idx*n_win + w)
#         tok_pos.i16     [nnz]  token position WITHIN the window (0-based, sink dropped)
#         tok_val.f16     [nnz]  the pre-gate activation there
#         pool.json       features, seed, gate, the window draw, nnz and the density it implies
#
# Only tokens with act > 0 are stored: the array is a post-ReLU pre-gate activation, so a zero is
# a real zero and its position carries nothing. The dense equivalent would be
# n_feat * n_win * 64 * 2 bytes = 134 MiB at 512 x 2048; the sparse form is reported against that
# in the README so the choice can be re-judged if the density changes.

RANDOM_POOL_BATCH = 256


def _forward_windows(model, read_layer, rows, sink, pad_id):
    """(h [B, T, d] fp32, keep [B, T]) for a list of id arrays -- `precompute/scan.py:_forward`.

    Copied rather than imported: `scan._forward` is private to that product, and this is four
    lines of batch assembly around the one shared `common.read_resid`.
    """
    import torch

    width = 1 + max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    am = np.zeros((len(rows), width), dtype=np.int64)
    ids[:, 0] = sink
    am[:, 0] = 1
    for i, r in enumerate(rows):
        ids[i, 1 : 1 + len(r)] = r
        am[i, 1 : 1 + len(r)] = 1
    h, mask = C.read_resid(
        model,
        read_layer,
        {"input_ids": torch.from_numpy(ids).cuda(), "attention_mask": torch.from_numpy(am).cuda()},
        pool="all",
    )
    keep = mask.clone()
    keep[:, 0] = False
    return h, keep


def enumerate_windows(docs):
    """[(doc, start, len)] for every corpus window, in the SAME order `scan` forwarded them.

    `scan` walks documents in stored order and emits `common.windows_of(doc_len)` for each, so
    replaying that walk reproduces its global window index exactly -- which is what makes a
    `window` id in one product mean the same window in another.
    """
    out = []
    for r in docs:
        for s, ln in C.windows_of(int(r["len"])):
            out.append((int(r["doc"]), s, ln))
    return out


def run_random_pool(cfg, args):
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product random_pool needs --base"
    ac = cfg["autointerp"]
    n_win = int(args.get("n_windows") or ac["random_pool_windows"])
    seed = int(args.get("pool_seed") or ac["random_pool_seed"])
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]

    rows_meta, sae_rows, feats, sae_key = _sae_rows(cfg, args)
    n_feat = len(feats)
    toks, docs = C.load_corpus(base, root)
    wins = enumerate_windows(docs)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(wins), size=min(n_win, len(wins)), replace=False))
    n_win = len(pick)
    print(f"[random_pool] {n_win} of {len(wins)} windows, {n_feat} features, seed {seed}", flush=True)

    model, tok = C.load_base(cfg, base)
    # ENCODER ONLY: every activation here goes through common.sae_encode (b_dec, W_enc, b_enc)
    # and nothing reads W_dec, which at 2^21 features is another 43 GB in fp32 and does not fit
    # an H200 beside the 27B (features/CHANGES.md fix 2, made there for `stats`).
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32, need_decoder=False)
    gate = float(sae.threshold)
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink
    fidx = torch.as_tensor(feats, device="cuda")
    w_enc = sae.W_enc[:, fidx].contiguous()
    b_enc = sae.b_enc[fidx]

    dense = np.zeros((n_feat, n_win, C.SCAN_BLOCK), dtype=np.float16)
    offs = {int(r["doc"]): int(r["offset"]) for r in docs}
    wrows = []
    t0 = time.time()
    for s in range(0, n_win, RANDOM_POOL_BATCH):
        sel = pick[s : s + RANDOM_POOL_BATCH]
        ids_list = []
        for j, wi in enumerate(sel.tolist()):
            doc, st, ln = wins[wi]
            ids_list.append(np.asarray(toks[offs[doc] + st : offs[doc] + st + ln]))
            wrows.append({"window": wi, "doc": doc, "start": st, "len": ln, "row": s + j})
        with torch.no_grad():
            h, keep = _forward_windows(model, read_layer, ids_list, sink, pad_id)
            a = torch.relu((h - sae.b_dec) @ w_enc + b_enc)  # [B, T, n_feat] pre-gate
            a = a.masked_fill(~keep.unsqueeze(-1), 0.0)
            t = min(a.shape[1] - 1, C.SCAN_BLOCK)
            blk = a[:, 1 : 1 + t].permute(2, 0, 1).to(torch.float16).cpu().numpy()  # [n_feat, B, t]
        dense[:, s : s + len(sel), :t] = blk
    elapsed = time.time() - t0

    lens = np.asarray([w["len"] for w in wrows], dtype=np.int64)
    valid = np.arange(C.SCAN_BLOCK)[None, :] < lens[:, None]  # [n_win, 64]
    dense *= valid[None, :, :]
    mx = dense.max(axis=2)  # [n_feat, n_win]
    nz = dense > 0
    counts = nz.reshape(n_feat * n_win, C.SCAN_BLOCK).sum(1)
    tok_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    fi, wi_, ti = np.nonzero(nz)
    order = np.argsort(fi.astype(np.int64) * n_win + wi_, kind="stable")
    tok_pos = ti[order].astype(np.int16)
    tok_val = dense[fi[order], wi_[order], ti[order]].astype(np.float16)
    assert len(tok_pos) == tok_off[-1], f"CSR {len(tok_pos)} entries but offsets end at {tok_off[-1]}"

    zero = (mx == 0).sum(1)
    near = ((mx > 0) & (mx <= gate)).sum(1)
    above = (mx > gate).sum(1)
    out = f"{C.sae_dir(sae_key, root)}/random_pool/{set_name}"
    dense_bytes = n_feat * n_win * C.SCAN_BLOCK * 2
    with C.outdir(
        out,
        args,
        inputs={
            "corpus": C.corpus_dir(base, root),
            "heldout": C.heldout_dir(base, set_name, root),
            "sae": sae_key,
            "features": n_feat,
            "windows": n_win,
            "seed": seed,
            "gate": gate,
        },
    ) as od:
        od.write_jsonl("windows.jsonl", wrows)
        od.write_array("max_act.f16", mx, "float16")
        od.write_array("tok_off.i64", tok_off, "int64")
        od.write_array("tok_pos.i16", tok_pos, "int16")
        od.write_array("tok_val.f16", tok_val, "float16")
        od.write_json(
            "pool.json",
            {
                "features": feats,
                "rows": list(sae_rows),
                "n_windows": n_win,
                "n_corpus_windows": len(wins),
                "seed": seed,
                "gate": gate,
                "block": C.SCAN_BLOCK,
                "nnz": int(tok_off[-1]),
                "density": round(float(tok_off[-1]) / (n_feat * n_win * C.SCAN_BLOCK), 6),
                "zero_windows_per_feature": {
                    "min": int(zero.min()), "median": int(np.median(zero)), "max": int(zero.max())
                },
                "nearmiss_windows_per_feature": {
                    "min": int(near.min()), "median": int(np.median(near)), "max": int(near.max())
                },
                "above_gate_windows_per_feature": {
                    "min": int(above.min()), "median": int(np.median(above)), "max": int(above.max())
                },
                "seconds": round(elapsed, 1),
            },
        )
        od.note(
            f"{n_win} windows drawn UNIFORMLY WITHOUT REPLACEMENT over all {len(wins)} corpus "
            f"windows by `np.random.default_rng({seed}).choice`, then sorted. `window` is the "
            f"GLOBAL window index in `scan`'s own enumeration (documents in stored order, "
            f"common.windows_of per document), so it means the same window as in "
            f"sae/<sae>/examples/. This pool is an INDEPENDENT draw: it is not a superset of "
            f"scan's `_random256`, whose reservoir order cannot be replayed by index."
        )
        od.note(
            f"`max_act.f16` is [{n_feat}, {n_win}] pre-gate maxima in the feature order of "
            f"pool.json. Per feature: zero-activation windows min {int(zero.min())} / median "
            f"{int(np.median(zero))}; near-miss (0 < max <= gate {gate:.4f}) min {int(near.min())} "
            f"/ median {int(np.median(near))}; above the gate min {int(above.min())} / median "
            f"{int(np.median(above))}. THIS IS THE POINT OF THE PRODUCT: scan's 256-window pool "
            f"leaves the densest features short of 20 zero-activation negatives."
        )
        od.note(
            f"per-token activations are a CSR over the FEATURE-MAJOR (feature, window) grid: row "
            f"f*{n_win} + w, `tok_pos` the 0-based position within the window (sink already "
            f"dropped), `tok_val` the pre-gate activation. Only act > 0 is stored -- a post-ReLU "
            f"zero is a real zero. {int(tok_off[-1])} entries, density "
            f"{float(tok_off[-1]) / (n_feat * n_win * C.SCAN_BLOCK):.4f} of the "
            f"{C.human(dense_bytes)} dense equivalent."
        )
        od.note(f"forward wall {elapsed:.1f}s for {n_win} windows x {n_feat} features")
    return {
        "out": out,
        "windows": n_win,
        "features": n_feat,
        "gate": gate,
        "nnz": int(tok_off[-1]),
        "density": round(float(tok_off[-1]) / (n_feat * n_win * C.SCAN_BLOCK), 6),
        "zero_min": int(zero.min()),
        "nearmiss_min": int(near.min()),
        "seconds": round(elapsed, 1),
    }


# ==============================================================================================
# Product `examples_4m`: the 4M-prefix corpus examples (design amendment A3)
# ==============================================================================================
#
# The `C4` arm is "what a cheap corpus search finds". Filtering the 16M `examples/` top-128 down to
# the documents inside the 4M prefix does NOT produce that: it produces the 4M-prefix members of
# the 16M ranking, which is a different and much smaller object -- MEASURED 2026-09-16 on the
# 64-feature pilot build, a median of 14 candidates after dedup and fewer than 16 on 38 of 64
# features. The honest C4 is its own top-k over the 4M prefix, which is this product.
#
#     <root>/base/<base>/sae/<sae>/examples_4m/<set>/
#         <feature>.jsonl   the SAME row schema as scan's examples/ (row, kind "top", window, doc,
#                           start, len, max_act, argmax, acts), top 128 by peak activation
#         tested.json, scan_4m.json
#
# `window` is the GLOBAL window index of `scan`'s own enumeration. The nested subsets are prefixes
# of the document order (common.size_tag_of), so the 4M prefix is a prefix of that enumeration and
# the two indices coincide -- a `window` means the same window here, in examples/ and in
# random_pool/.

EX4M_TOP = 128
EX4M_BATCH = 256


def run_examples_4m(cfg, args):
    import torch

    from precompute.scan import _Heap

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product examples_4m needs --base"
    prefix_m = int(args.get("prefix_m") or cfg["autointerp"]["corpus_prefix_m"])
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]
    batch_rows = int(args.get("batch") or EX4M_BATCH)

    rows_meta, sae_rows, feats, sae_key = _sae_rows(cfg, args)
    n_feat = len(feats)
    toks, docs = C.load_corpus(base, root)
    sizes = C.corpus_sizes(docs)
    assert prefix_m in sizes, f"corpus has nested sizes {sizes}; {prefix_m}M is not one of them"
    keep_docs = [r for r in docs if int(r["size_tag"]) <= prefix_m]
    n_tok = sum(int(r["len"]) for r in keep_docs)
    print(
        f"[examples_4m] {len(keep_docs)} of {len(docs)} docs, {n_tok} tokens (<= {prefix_m}M), "
        f"{n_feat} features",
        flush=True,
    )

    model, tok = C.load_base(cfg, base)
    # ENCODER ONLY: every activation here goes through common.sae_encode (b_dec, W_enc, b_enc)
    # and nothing reads W_dec, which at 2^21 features is another 43 GB in fp32 and does not fit
    # an H200 beside the 27B (features/CHANGES.md fix 2, made there for `stats`).
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32, need_decoder=False)
    gate = float(sae.threshold)
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink
    fidx = torch.as_tensor(feats, device="cuda")
    w_enc = sae.W_enc[:, fidx].contiguous()
    b_enc = sae.b_enc[fidx]

    heap = _Heap(n_feat, EX4M_TOP, "cuda", payload_shape=(C.SCAN_BLOCK,))
    win_doc: list = []
    win_start: list = []
    win_len: list = []
    buf: list = []
    buf_meta: list = []
    w_global = 0
    t0 = time.time()

    def flush():
        nonlocal w_global
        if not buf:
            return
        with torch.no_grad():
            h, keep = _forward_windows(model, read_layer, buf, sink, pad_id)
            b, t = keep.shape
            a = torch.relu((h - sae.b_dec) @ w_enc + b_enc)  # [B, T, n_feat] pre-gate
            a = a.masked_fill(~keep.unsqueeze(-1), 0.0)
            amax, aarg = a.max(dim=1)  # [B, n_feat]
            pay = torch.zeros((b, C.SCAN_BLOCK, n_feat), dtype=torch.float16, device="cuda")
            pay[:, : min(t - 1, C.SCAN_BLOCK)] = a[:, 1 : 1 + C.SCAN_BLOCK].to(torch.float16)
            pay = pay.permute(2, 0, 1).contiguous()  # [n_feat, B, 64]
            wid = torch.arange(w_global, w_global + b, device="cuda")
            heap.push(amax.T.contiguous(), wid, aarg.T.contiguous(), pay)
        win_doc.append(np.asarray([m[0] for m in buf_meta], dtype=np.int32))
        win_start.append(np.asarray([m[1] for m in buf_meta], dtype=np.int32))
        win_len.append(np.asarray([len(r) for r in buf], dtype=np.int32))
        w_global += b
        buf.clear()
        buf_meta.clear()

    done_tokens = 0
    for r in keep_docs:
        ids = np.asarray(toks[r["offset"] : r["offset"] + r["len"]])
        for s, ln in C.windows_of(int(r["len"])):
            buf.append(ids[s : s + ln])
            buf_meta.append((int(r["doc"]), s))
            if len(buf) >= batch_rows:
                flush()
        done_tokens += int(r["len"])
        if r["doc"] % 2000 == 0 and r["doc"]:
            el = time.time() - t0
            print(
                f"[examples_4m] doc {r['doc']}/{len(keep_docs)} {done_tokens / 1e6:.2f}M tokens | "
                f"{done_tokens / max(el, 1e-9):.0f} corpus tok/s",
                flush=True,
            )
    flush()
    elapsed = time.time() - t0
    wdoc = np.concatenate(win_doc)
    wstart = np.concatenate(win_start)
    wlen = np.concatenate(win_len)
    assert len(wdoc) == w_global, f"window table {len(wdoc)} != {w_global} forwarded windows"

    from precompute.scan import _ex

    tv = heap.val.cpu().numpy()
    tw = heap.win.cpu().numpy()
    ta = heap.arg.cpu().numpy()
    tp = heap.payload.cpu().numpy()
    out = f"{C.sae_dir(sae_key, root)}/examples_4m/{set_name}"
    per_feature = []
    ex_rows = 0
    nbytes = 0
    with C.outdir(
        out,
        args,
        inputs={
            "corpus": C.corpus_dir(base, root),
            "heldout": C.heldout_dir(base, set_name, root),
            "sae": sae_key,
            "prefix_m": prefix_m,
            "docs": len(keep_docs),
            "tokens": n_tok,
            "windows": int(w_global),
            "features": n_feat,
        },
    ) as od:
        for fi, feat in enumerate(feats):
            recs = []
            for j in range(EX4M_TOP):
                if not np.isfinite(tv[fi, j]) or tv[fi, j] <= 0:
                    continue
                recs.append(
                    _ex(sae_rows[fi], "top", tv[fi, j], tw[fi, j], ta[fi, j], tp[fi, j],
                        wdoc, wstart, wlen)
                )
            path = od.file(f"{feat}.jsonl")
            C.write_jsonl(path, recs)
            nbytes += path.stat().st_size
            ex_rows += len(recs)
            n_gate = sum(1 for r in recs if r["max_act"] > gate)
            per_feature.append({"feature": feat, "row": sae_rows[fi], "n": len(recs),
                                "n_above_gate": n_gate,
                                "n_docs": len({r["doc"] for r in recs}),
                                "max_act": recs[0]["max_act"] if recs else 0.0})
        od.index["examples"] = {"kind": "jsonl", "rows": ex_rows, "bytes": nbytes}
        od.write_json("tested.json", {"features": feats, "rows": sae_rows, "sae": sae_key})
        ng = np.asarray([p["n_above_gate"] for p in per_feature])
        nd = np.asarray([p["n_docs"] for p in per_feature])
        od.write_json(
            "scan_4m.json",
            {
                "prefix_m": prefix_m,
                "docs": len(keep_docs),
                "tokens": n_tok,
                "windows": int(w_global),
                "top_n": EX4M_TOP,
                "gate": gate,
                "features": n_feat,
                "per_feature": per_feature,
                "n_above_gate": {"min": int(ng.min()), "median": int(np.median(ng)),
                                 "n_below_16": int((ng < 16).sum())},
                "n_distinct_docs": {"min": int(nd.min()), "median": int(np.median(nd)),
                                    "n_below_16": int((nd < 16).sum())},
                "seconds": round(elapsed, 1),
            },
        )
        od.note(
            f"the C4 arm's OWN top-{EX4M_TOP} over the {prefix_m}M nested prefix "
            f"({len(keep_docs)} documents, {n_tok} tokens, {w_global} windows at "
            f"{C.SCAN_BLOCK}/{C.SCAN_STRIDE}), not the {prefix_m}M members of the 16M ranking. "
            f"Design amendment A3: the filtered version left a median of 14 candidates after "
            f"dedup and fewer than 16 on 38 of 64 pilot features, so C4 was not an N=16 arm."
        )
        od.note(
            f"row schema and ranking are `precompute/scan.py`'s (`_ex`, `_Heap`), so a row here is "
            f"interchangeable with one from sae/<sae>/examples/. `window` is the GLOBAL index of "
            f"scan's enumeration: the nested subsets are prefixes of the document order, so the "
            f"{prefix_m}M prefix is a prefix of that enumeration and the indices coincide."
        )
        od.note(
            f"per feature, windows above the gate {gate:.4f}: min {int(ng.min())}, median "
            f"{int(np.median(ng))}, below 16 on {int((ng < 16).sum())} of {n_feat} features. "
            f"Distinct documents: min {int(nd.min())}, median {int(np.median(nd))}, below 16 on "
            f"{int((nd < 16).sum())} -- the binding constraint once document-level disjointness "
            f"(A4) is enforced."
        )
        od.note(f"scan wall {elapsed:.1f}s, {done_tokens / max(elapsed, 1e-9):.0f} corpus tok/s")
    return {
        "out": out,
        "docs": len(keep_docs),
        "tokens": n_tok,
        "windows": int(w_global),
        "features": n_feat,
        "rows": ex_rows,
        "n_above_gate_min": int(ng.min()),
        "n_features_below_16_above_gate": int((ng < 16).sum()),
        "seconds": round(elapsed, 1),
    }


# ==============================================================================================
# Product `examples_docmax`: one window per DOCUMENT, so the test set has somewhere to come from
# ==============================================================================================
#
# MEASURED 2026-09-16 on the 64-feature pilot build, with amendments A1 (gate-consistent
# positives), A4 (document-level disjointness) and A7 (two disjoint test draws) all in force:
#
#     draw 1 reaches its 20 positives on 35 of 64 features (min 0)
#     draw 2 reaches its 20 positives on 20 of 64, and is EMPTY on 21
#
# The arithmetic behind that. A feature's stored gate-passing windows are `examples/`'s top-128
# (median 60 after overlap dedup) plus whatever `q0..q3` rows clear the gate -- almost none, since
# those bands are equal-width bins of (0, max_act] and therefore sample the weak tail. The arms
# show 16-32 of the top windows, which occupy a median of 28 documents (max 44), and A4 then
# removes EVERY OTHER WINDOW IN THOSE DOCUMENTS. What is left has to cover 40 positives across two
# draws, and on half the features it cannot.
#
# The store is the problem, not the rules: 128 windows ranked by activation concentrate in few
# documents. This product ranks DOCUMENTS instead -- for each tested feature it keeps the single
# best-activating window of each of the top `EXDOC_TOP` documents, so the pool is
# document-diverse by construction and A4 costs one window per document rather than a whole
# document's worth.
#
#     <root>/base/<base>/sae/<sae>/examples_docmax/<set>/
#         <feature>.jsonl   scan's row schema, kind "docmax", one row per document
#         tested.json, scan_docmax.json
#
# `window` is the GLOBAL window index of scan's enumeration, as everywhere else in autointerp/.

EXDOC_TOP = 256
EXDOC_DOC_FLUSH = 256  # documents buffered before one heap push


def run_examples_docmax(cfg, args):
    import torch

    from precompute.scan import _ex, _Heap

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product examples_docmax needs --base"
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]
    batch_rows = int(args.get("batch") or EX4M_BATCH)

    rows_meta, sae_rows, feats, sae_key = _sae_rows(cfg, args)
    n_feat = len(feats)
    toks, docs = C.load_corpus(base, root)
    print(f"[examples_docmax] {len(docs)} docs, {n_feat} features, top {EXDOC_TOP} documents each",
          flush=True)

    model, tok = C.load_base(cfg, base)
    # ENCODER ONLY: every activation here goes through common.sae_encode (b_dec, W_enc, b_enc)
    # and nothing reads W_dec, which at 2^21 features is another 43 GB in fp32 and does not fit
    # an H200 beside the 27B (features/CHANGES.md fix 2, made there for `stats`).
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32, need_decoder=False)
    gate = float(sae.threshold)
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink
    fidx = torch.as_tensor(feats, device="cuda")
    w_enc = sae.W_enc[:, fidx].contiguous()
    b_enc = sae.b_enc[fidx]

    # Two heaps over the SAME values: one carries the activation payload, the other the window
    # ids. `_Heap.push` broadcasts ONE window id per column, but every column here is a different
    # document with its own window id, so the ids ride in the second heap's `arg` slot. Both see
    # the identical value sequence, so `topk` selects identically -- asserted after the walk.
    heap = _Heap(n_feat, EXDOC_TOP, "cuda", payload_shape=(C.SCAN_BLOCK,))
    win_heap = _Heap(n_feat, EXDOC_TOP, "cuda")
    win_doc: list = []
    win_start: list = []
    win_len: list = []
    # running best of the CURRENT document, and the buffer of finished documents' bests
    cur_doc = None
    cur_val = torch.full((n_feat,), -1.0, device="cuda")
    cur_win = torch.zeros((n_feat,), dtype=torch.int64, device="cuda")
    cur_arg = torch.zeros((n_feat,), dtype=torch.int64, device="cuda")
    cur_pay = torch.zeros((n_feat, C.SCAN_BLOCK), dtype=torch.float16, device="cuda")
    doc_val: list = []
    doc_win: list = []
    doc_arg: list = []
    doc_pay: list = []
    buf: list = []
    buf_meta: list = []
    w_global = 0
    n_docs_pushed = 0
    t0 = time.time()
    done_tokens = 0

    def close_doc():
        """Move the finished document's per-feature best into the buffer; push when it is full."""
        nonlocal n_docs_pushed
        doc_val.append(cur_val.clone())
        doc_win.append(cur_win.clone())
        doc_arg.append(cur_arg.clone())
        doc_pay.append(cur_pay.clone())
        cur_val.fill_(-1.0)
        n_docs_pushed += 1
        if len(doc_val) >= EXDOC_DOC_FLUSH:
            push_docs()

    def push_docs():
        if not doc_val:
            return
        v = torch.stack(doc_val, 1)  # [n_feat, D]
        z = torch.zeros(len(doc_val), dtype=torch.int64, device="cuda")
        heap.push(v, z, torch.stack(doc_arg, 1), torch.stack(doc_pay, 1))
        win_heap.push(v, z, torch.stack(doc_win, 1))
        doc_val.clear()
        doc_win.clear()
        doc_arg.clear()
        doc_pay.clear()

    def flush():
        """Forward a FULL batch, then fold it into the running per-document bests.

        The buffer deliberately spans documents -- flushing at every boundary would mean ~50-window
        forwards instead of 256-window ones -- so the batch is cut into its contiguous per-document
        segments here and each segment is folded separately.
        """
        nonlocal w_global, cur_doc
        if not buf:
            return
        with torch.no_grad():
            h, keep = _forward_windows(model, read_layer, buf, sink, pad_id)
            b, t = keep.shape
            a = torch.relu((h - sae.b_dec) @ w_enc + b_enc)  # [B, T, n_feat] pre-gate
            a = a.masked_fill(~keep.unsqueeze(-1), 0.0)
            amax, aarg = a.max(dim=1)  # [B, n_feat]
            pay = torch.zeros((b, C.SCAN_BLOCK, n_feat), dtype=torch.float16, device="cuda")
            pay[:, : min(t - 1, C.SCAN_BLOCK)] = a[:, 1 : 1 + C.SCAN_BLOCK].to(torch.float16)
            segs = []
            lo = 0
            for i in range(1, b + 1):
                if i == b or buf_meta[i][0] != buf_meta[lo][0]:
                    segs.append((buf_meta[lo][0], lo, i))
                    lo = i
            for doc_id, lo_, hi_ in segs:
                if cur_doc is not None and doc_id != cur_doc:
                    close_doc()
                cur_doc = doc_id
                seg_best, which = amax[lo_:hi_].max(dim=0)  # [n_feat]
                better = seg_best > cur_val
                fi = torch.nonzero(better, as_tuple=True)[0]
                if len(fi):
                    wsel = which[fi] + lo_
                    cur_val[fi] = seg_best[fi]
                    cur_win[fi] = wsel + w_global
                    cur_arg[fi] = aarg[wsel, fi]
                    cur_pay[fi] = pay[wsel, :, fi]
        win_doc.append(np.asarray([m[0] for m in buf_meta], dtype=np.int32))
        win_start.append(np.asarray([m[1] for m in buf_meta], dtype=np.int32))
        win_len.append(np.asarray([len(r) for r in buf], dtype=np.int32))
        w_global += b
        buf.clear()
        buf_meta.clear()

    for r in docs:
        ids = np.asarray(toks[r["offset"] : r["offset"] + r["len"]])
        for s, ln in C.windows_of(int(r["len"])):
            buf.append(ids[s : s + ln])
            buf_meta.append((int(r["doc"]), s))
            if len(buf) >= batch_rows:
                flush()
        done_tokens += int(r["len"])
        if r["doc"] % 4000 == 0 and r["doc"]:
            el = time.time() - t0
            print(f"[examples_docmax] doc {r['doc']}/{len(docs)} {done_tokens / 1e6:.2f}M tokens | "
                  f"{done_tokens / max(el, 1e-9):.0f} corpus tok/s", flush=True)
    flush()
    if cur_doc is not None:
        close_doc()
    push_docs()
    elapsed = time.time() - t0
    wdoc = np.concatenate(win_doc)
    wstart = np.concatenate(win_start)
    wlen = np.concatenate(win_len)
    assert len(wdoc) == w_global, f"window table {len(wdoc)} != {w_global} forwarded windows"
    assert n_docs_pushed == len(docs), (
        f"{n_docs_pushed} documents folded but the corpus has {len(docs)}: a document boundary "
        f"was missed and its windows were credited to a neighbour"
    )
    assert torch.equal(heap.val, win_heap.val), (
        "the payload heap and the window-id heap ranked differently; their `arg` columns no "
        "longer correspond and every stored window id would be wrong"
    )

    tv = heap.val.cpu().numpy()
    ta = heap.arg.cpu().numpy()
    tp = heap.payload.cpu().numpy()
    tw = win_heap.arg.cpu().numpy()  # the window ids, ranked by the SAME values
    out = f"{C.sae_dir(sae_key, root)}/examples_docmax/{set_name}"
    per_feature = []
    ex_rows = 0
    nbytes = 0
    with C.outdir(
        out,
        args,
        inputs={
            "corpus": C.corpus_dir(base, root),
            "heldout": C.heldout_dir(base, set_name, root),
            "sae": sae_key,
            "docs": len(docs),
            "tokens": done_tokens,
            "windows": int(w_global),
            "features": n_feat,
            "top_documents": EXDOC_TOP,
        },
    ) as od:
        for fi_, feat in enumerate(feats):
            recs = []
            for j in range(EXDOC_TOP):
                if not np.isfinite(tv[fi_, j]) or tv[fi_, j] <= 0:
                    continue
                recs.append(
                    _ex(sae_rows[fi_], "docmax", tv[fi_, j], tw[fi_, j], ta[fi_, j], tp[fi_, j],
                        wdoc, wstart, wlen)
                )
            path = od.file(f"{feat}.jsonl")
            C.write_jsonl(path, recs)
            nbytes += path.stat().st_size
            ex_rows += len(recs)
            n_gate = sum(1 for x in recs if x["max_act"] > gate)
            per_feature.append({"feature": feat, "row": sae_rows[fi_], "n": len(recs),
                                "n_above_gate": n_gate,
                                "n_docs": len({x["doc"] for x in recs}),
                                "max_act": recs[0]["max_act"] if recs else 0.0})
        od.index["examples"] = {"kind": "jsonl", "rows": ex_rows, "bytes": nbytes}
        od.write_json("tested.json", {"features": feats, "rows": sae_rows, "sae": sae_key})
        ng = np.asarray([p["n_above_gate"] for p in per_feature])
        nd = np.asarray([p["n_docs"] for p in per_feature])
        od.write_json(
            "scan_docmax.json",
            {"docs": len(docs), "tokens": done_tokens, "windows": int(w_global),
             "top_documents": EXDOC_TOP, "gate": gate, "features": n_feat,
             "per_feature": per_feature,
             "n_above_gate": {"min": int(ng.min()), "median": int(np.median(ng)),
                              "n_below_48": int((ng < 48).sum())},
             "n_distinct_docs": {"min": int(nd.min()), "median": int(np.median(nd)),
                                 "n_below_48": int((nd < 48).sum())},
             "seconds": round(elapsed, 1)},
        )
        od.note(
            f"ONE WINDOW PER DOCUMENT: for each tested feature, the best-activating window of each "
            f"of the top {EXDOC_TOP} documents, over the whole {done_tokens} -token corpus. "
            f"`examples/`'s top-128 ranks WINDOWS and therefore concentrates in few documents "
            f"(median 28 shown documents per feature), which is why document-level disjointness "
            f"(A4) left half the pilot's features without a test set. Row schema is "
            f"`precompute/scan.py`'s `_ex`, so a row here is interchangeable with one from "
            f"`examples/` or `examples_4m/`."
        )
        od.note(
            f"per feature, documents above the gate {gate:.4f}: min {int(ng.min())}, median "
            f"{int(np.median(ng))}, below 48 on {int((ng < 48).sum())} of {n_feat} features "
            f"(48 = the 40 test positives of two draws plus headroom)."
        )
        od.note(f"scan wall {elapsed:.1f}s, {done_tokens / max(elapsed, 1e-9):.0f} corpus tok/s")
    return {"out": out, "docs": len(docs), "windows": int(w_global), "features": n_feat,
            "rows": ex_rows, "n_above_gate_min": int(ng.min()),
            "n_above_gate_median": int(np.median(ng)),
            "n_features_below_48_above_gate": int((ng < 48).sum()),
            "seconds": round(elapsed, 1)}
