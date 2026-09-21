"""Product `repo_examples`: score the SAE repo's OWN max-activating windows.

    <root>/base/<base>/sae/<sae>/repo_examples/<set>/
        repo_examples.jsonl   one row per (tested feature, rank): the shipped window, the repo's
                              own per-token activations, and OUR numbers from the same forward
        per_feature.jsonl     one row per tested feature: the "sae-repo-top32" baseline column
                              (max_cos over the shipped windows) and the repo-vs-us agreement

The SAE authors ran their own scan over their own corpus and shipped, per feature, the 30-32
highest-activating 32-token windows. Those windows are the strongest natural text anybody has for
a feature without searching at eval time, so pushing them through OUR scorer gives the `sae` family
an extra baseline column -- "sae-repo-top32" -- beside corpus retrieval and the MAEMM rollouts.
It is the SAME `common.score_tokens` the rollouts go through, with no variation whatsoever: that
is the only reason the numbers are comparable (checklist item 12, one scoring path).

It is ALSO a check on us. The repo ships its own per-token activation for every shipped token, so
this product can put their peak against ours window by window. If our activation formula
(`relu((h - b_dec) @ W_enc[:, f] + b_enc[f])`) or our read layer (block OUTPUT of `read_layer`, not
`output_hidden_states`) disagreed with the scan that produced the file, the two would not correlate.
The README records the Pearson r and the argmax agreement per base.

**Sink handling differs per base and is declared in config.yaml, not inferred.** A repo whose scan
re-encoded windows the way `eval/eval_universal.py:_reencode` does carries the tokenizer's sink
token at position 0 of every window; `score_tokens` prepends its own, so that column must be
dropped from the ids AND from the repo's activations (`common.strip_repo_sink`, which asserts the
config's `sink_first` against the data). The 8B file is sink-prefixed; the 27B file is not.
"""

from __future__ import annotations

import os
import time

import numpy as np

import precompute.common as C

# Rows handed to common.score_tokens per call; it re-chunks internally at common.SCORE_CHUNK, so
# this only bounds the fp32 residual the on_chunk callback holds at once. Same value score.py uses.
SCORE_ROWS = 256


class _Acts:
    """Per-row SAE numbers taken from the SCORING forward itself, never a second one.

    One instance per run; `s` is shifted to the global row index by the caller, and the feature the
    callback used for each row is recorded so the offset arithmetic can be asserted afterwards
    rather than trusted.
    """

    def __init__(self, sae, feats, n_rows):
        import torch

        self.sae = sae
        self.feats = torch.as_tensor(feats, dtype=torch.long, device=sae.W_enc.device)
        self.our_arg = torch.full((n_rows,), -1, dtype=torch.int64)
        self.act_at_arg = torch.zeros((n_rows,), dtype=torch.float32)
        self.peak_act = torch.zeros((n_rows,), dtype=torch.float32)
        self.peak_arg = torch.full((n_rows,), -1, dtype=torch.int64)
        self.seen = torch.full((n_rows,), -1, dtype=torch.int64)  # the feature each row was scored with

    def __call__(self, s, h, cos, keep, ids):
        import torch

        b = h.shape[0]
        f = self.feats[s : s + b]
        w = self.sae.W_enc[:, f].T  # [b, d], the encoder column of each row's OWN feature
        # mxf/sae.py:27-31, the same pre-topk post-ReLU activation score.py stores at the argmax.
        a = torch.relu(torch.einsum("btd,bd->bt", h - self.sae.b_dec, w) + self.sae.b_enc[f].unsqueeze(1))
        # -1 is below every relu output, so a masked position can never win either argmax.
        a = torch.where(keep, a, torch.full_like(a, -1.0))
        _, carg = torch.where(keep, cos, torch.full_like(cos, -1.0)).max(dim=1)
        pk, parg = a.max(dim=1)
        has = keep.any(dim=1)
        rows = torch.arange(b, device=h.device)
        zero = torch.zeros_like(pk)
        neg = torch.full_like(carg, -1)
        # argmaxes are shifted back by one: column 0 is the sink score_tokens prepended, so 0 here
        # means "the first token of the shipped window" (the same convention as scores/argmax.i16).
        self.our_arg[s : s + b] = torch.where(has, carg - 1, neg).cpu()
        self.peak_arg[s : s + b] = torch.where(has, parg - 1, neg).cpu()
        self.act_at_arg[s : s + b] = torch.where(has, a[rows, carg], zero).cpu()
        self.peak_act[s : s + b] = torch.where(has, pk, zero).cpu()
        self.seen[s : s + b] = f.cpu()


def _sink_uniform_features(cfg, sae_key, tok, feats):
    """Per feature: do ALL of its shipped windows agree with `max_acts.sink_first`?

    MEASURED 2026-09-16 at full scale: adamkarvonen's 8B file is NOT uniform. Of the 15,360 windows
    belonging to the 512 features the 16M `2026-09-16_v1` draw tests, 74 do not start with the sink
    token, so `common.strip_repo_sink` rightly refuses to strip column 0 off the whole tensor --
    stripping would delete a real token on those rows and shift every shipped activation by one.
    (The step-5 smoke drew a different 512 features and happened to hit none of them.)

    A row-by-row strip would make the [F, W, T] grid ragged, and `n_win` is a fixed stride
    throughout `run()` below, so the unit of exclusion is the FEATURE: a feature is kept only if
    every one of its windows matches the declaration. The count and the ids are reported in
    summary.json and in the product README -- the baseline column then covers fewer than the sae
    family's 512 features, and that is a fact about the shipped file, not about the SAE.

    `mmap=True` keeps the full [F_file, W, 32] tensor out of memory; only the tested rows are read.
    """
    import torch

    want = bool(cfg["saes"][sae_key]["max_acts"]["sink_first"])
    blob = torch.load(C.max_acts_path(cfg, sae_key), map_location="cpu", mmap=True, weights_only=False)
    idx = torch.as_tensor(feats, dtype=torch.long)
    first = blob["max_tokens"][idx][..., 0] == C.sink_token_id(tok)  # [F, W] bool
    ok = first.all(dim=1) if want else (~first).all(dim=1)
    return ok.tolist(), int(first.sum()), int(first.numel())


def _load_windows(cfg, sae_key, tok, feats):
    """(ids [F, W, T], acts [F, W, T] fp32, info) -- the shipped windows of `feats` only.

    `mmap=True` keeps the full [65536|131072, W, 32] tensors out of memory: only the rows the
    held-out set actually tests are materialised.
    """
    import torch

    ma = cfg["saes"][sae_key]["max_acts"]
    path = C.max_acts_path(cfg, sae_key)
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    for key in ("max_acts", "max_tokens"):
        assert key in blob, f"{path}: missing {key!r} (keys: {sorted(map(str, blob))})"
    idx = torch.as_tensor(feats, dtype=torch.long)
    n_feat_file, n_win, n_tok = blob["max_tokens"].shape
    assert blob["max_acts"].shape == (n_feat_file, n_win, n_tok), (
        f"{path}: max_acts {tuple(blob['max_acts'].shape)} != max_tokens "
        f"{(n_feat_file, n_win, n_tok)}; one activation per shipped token is the whole premise"
    )
    assert n_win == int(ma["windows"]), (
        f"{path} ships {n_win} windows per feature but config.yaml says max_acts.windows="
        f"{ma['windows']}; the baseline column is named after that count"
    )
    assert int(idx.max()) < n_feat_file, (
        f"held-out set tests feature {int(idx.max())} but {path} only covers {n_feat_file} features"
    )
    ids = blob["max_tokens"][idx].to(torch.int64)
    acts = blob["max_acts"][idx].to(torch.float32)
    vocab = len(tok)
    assert int(ids.min()) >= 0 and int(ids.max()) < vocab, (
        f"{path}: token ids run {int(ids.min())}..{int(ids.max())}, outside the tokenizer's "
        f"0..{vocab - 1} -- this file was not written with the base's tokenizer"
    )
    sink = C.sink_token_id(tok)
    pre_arg0 = float((acts.argmax(dim=-1) == 0).float().mean())
    ids, acts, frac_sink = C.strip_repo_sink(ids, acts, sink, bool(ma["sink_first"]))
    info = {
        "path": path,
        "repo": ma.get("repo", cfg["saes"][sae_key]["hf"]),
        "repo_type": ma.get("repo_type", "model"),
        "features_in_file": int(n_feat_file),
        "windows_per_feature": int(n_win),
        "tokens_per_window_shipped": int(n_tok),
        "tokens_per_window_scored": int(ids.shape[-1]),
        "sink_first": bool(ma["sink_first"]),
        "sink_token": sink,
        "frac_windows_starting_with_sink": round(frac_sink, 6),
        "frac_shipped_argmax_at_position_0": round(pre_arg0, 6),
        "config_keys": sorted(str(k) for k in blob),
        "file_config": _repr_config(blob.get("config")),
    }
    return ids, acts, info


def _repr_config(obj) -> str:
    """The `config` blob some max-acts files carry, flattened to one short line for the README.

    It is the only provenance the file itself gives (which corpus was scanned, how it was cut), so
    it is recorded verbatim-ish rather than summarised away; a file without one says so.
    """
    if obj is None:
        return "(no `config` key in the file)"
    text = repr(obj)
    return text if len(text) <= 1500 else text[:1500] + " …(truncated)"


def _pearson(x, y):
    """Pearson r, or None when either side is constant (a dead feature's peaks are all 0)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# Features whose windows are re-forwarded WITHOUT the sink for the diagnostic below. 16 x 30-32
# windows is ~3% of the product's forward, small enough to run unconditionally on every call.
DIAG_FEATURES = 16


def _sink_diag(model, read_layer, sae, ids_t, acts_t, feats, n_feat_diag):
    """DIAGNOSTIC ONLY, deliberately NOT `score_tokens`: the same windows with NO sink prepended.

    The baseline column has to go through `score_tokens`, which prepends the sink every rollout is
    scored with -- that is what makes the numbers comparable. But a repo whose own scan used no
    sink measured its activations in a DIFFERENT context, so the repo-vs-us agreement can be
    degraded by the sink alone and not by any disagreement about the formula or the read layer.
    This forwards the first `n_feat_diag` features' windows on their RAW shipped ids (no sink, no
    decode/re-encode round trip) and returns the same peak agreement, so the README can say which
    of the two explanations the numbers support. It writes nothing and changes no output.
    """
    import torch

    ids = ids_t[:n_feat_diag].reshape(-1, ids_t.shape[-1])
    acts = acts_t[:n_feat_diag].reshape(-1, acts_t.shape[-1])
    fl = torch.as_tensor(
        [f for f in feats[:n_feat_diag] for _ in range(ids_t.shape[1])],
        dtype=torch.long,
        device=sae.W_enc.device,
    )
    peaks = []
    for s in range(0, ids.shape[0], C.SCORE_CHUNK):
        b = ids[s : s + C.SCORE_CHUNK].to(sae.W_enc.device)
        f = fl[s : s + C.SCORE_CHUNK]
        h, _ = C.read_resid(
            model, read_layer, {"input_ids": b, "attention_mask": torch.ones_like(b)}, pool="all"
        )
        w = sae.W_enc[:, f].T
        a = torch.relu(torch.einsum("btd,bd->bt", h - sae.b_dec, w) + sae.b_enc[f].unsqueeze(1))
        peaks.append(a.max(dim=1).values.cpu())
    ours = torch.cat(peaks).numpy()
    theirs = acts.max(dim=1).values.numpy()
    n_win = ids_t.shape[1]
    rs = []
    for j in range(n_feat_diag):
        sl = slice(j * n_win, (j + 1) * n_win)
        r = _pearson(theirs[sl], ours[sl])
        if r is not None:
            rs.append(r)
    ratio = ours / np.maximum(theirs, 1e-9)
    return {
        "features": int(n_feat_diag),
        "windows": int(ours.size),
        "mean_per_feature_peak_r_no_sink": round(float(np.mean(rs)), 6) if rs else None,
        "median_ratio_ours_over_repo_no_sink": round(float(np.median(ratio)), 6),
    }


def _sink_verdict(diag) -> str:
    """One line saying what the sink diagnostic settles, computed rather than asserted by hand."""
    a, b = diag["mean_per_feature_peak_r_with_sink"], diag["mean_per_feature_peak_r_no_sink"]
    if a is None or b is None:
        return "Verdict: not computable (no feature had a varying repo peak over its windows)."
    if abs(a - b) < 0.02:
        return (
            f"Verdict: the sink does NOT account for the agreement level here (delta r {a - b:+.4f}); "
            f"at r ~ {max(a, b):.3f} something other than the sink column separates the repo's scan "
            f"from ours, and this product's numbers are the sink-prefixed ones either way."
        )
    if a > b:
        return (
            f"Verdict: the WITH-sink context -- the one this product scores in -- is the one that "
            f"matches the repo's own scan (delta r {a - b:+.4f}); dropping the sink costs "
            f"{a - b:.3f} of r, so the sink handling above is doing real work."
        )
    return (
        f"Verdict: the NO-sink context matches the repo's scan better (delta r {a - b:+.4f}), so the "
        f"repo scanned without one. The baseline column keeps the sink regardless -- it must be "
        f"scored exactly as the rollouts are -- but the agreement number above is understated."
    )


def run(cfg, args):
    import torch
    import torch.nn.functional as TF

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product repo_examples needs --base"
    spec = cfg["bases"][base]
    read_layer, d = spec["read_layer"], spec["d"]
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")

    src = args.get("dirs_from") or C.heldout_dir(base, set_name, root)
    rows_meta = C.read_jsonl(f"{src}/ids.jsonl")
    # `id` is the field targets.py writes for the sae family (the SAE feature id); `row` is the
    # index into vecs.f16. Both are read by name so a renamed field fails loudly here.
    # Encoder rows of THIS dictionary only (common.sae_rows_of): the assert below compares
    # vecs.f16 against unit(W_enc[:, f]) of `sae_key`, so a row belonging to another dictionary --
    # or a `sae_side: dec` row, whose direction is a decoder row and not an encoder column --
    # would fail it for a reason that is not a drift.
    sel = C.sae_rows_of(rows_meta, sae_key, side="enc")
    assert sel, (
        f"{src}/ids.jsonl has no encoder rows of dictionary {sae_key!r}; nothing to take repo "
        f"windows for (it carries dictionaries "
        f"{sorted({r.get('sae_key', '(unkeyed)') for r in rows_meta if r['family'] in C.SAE_FAMILIES})})"
    )
    feats = [int(r["id"]) for r in sel]
    assert len(set(feats)) == len(feats), "the sae family repeats a feature id"
    # The `sae` rows this product scores are not centrable at all (an encoder column has no mean),
    # so every centring resolves to the same vectors here -- but the resolution still goes through
    # common.dirs_for, because that is what makes "the direction scored here is the SAME object the
    # rollouts were scored against" a fact about one code path rather than about two readers of one
    # file. On a `storage: raw` set with no --maemm in scope, --mu is required.
    cen_notes: list[str] = []
    mu, _ = C.mu_for(cfg, base, src, args, "", root, cen_notes)
    vecs = C.dirs_for(cfg, base, src, mu, root, cen_notes)
    assert vecs.shape == (len(rows_meta), d), f"{src}: dirs_for returned {vecs.shape}"
    dirs_f = TF.normalize(torch.from_numpy(np.asarray(vecs)[[r["row"] for r in sel]]), dim=-1)

    out = C.repo_examples_dir(sae_key, set_name, root)
    assert args.get("force") or not os.path.exists(out), (
        f"{out} already exists; refusing to overwrite without --force"
    )

    model, tok = C.load_base(cfg, base)
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32)
    gate = float(sae.threshold)
    # The direction scored here must be the SAME object the rollouts were scored against, and it
    # must be the SAE's own encoder column: if vecs.f16 had drifted from the checkpoint, every
    # number below would be about a different direction than the `sae` family's.
    ok, n_sink_win, n_all_win = _sink_uniform_features(cfg, sae_key, tok, feats)
    dropped = [f for f, k in zip(feats, ok, strict=True) if not k]
    if dropped:
        keep = [i for i, k in enumerate(ok) if k]
        assert keep, (
            f"every one of the {len(feats)} tested features has at least one window that "
            f"contradicts max_acts.sink_first={cfg['saes'][sae_key]['max_acts']['sink_first']}; "
            f"the declaration itself is wrong, not the data"
        )
        print(
            f"[repo_examples] {len(dropped)} of {len(feats)} tested features dropped: not all of "
            f"their shipped windows match max_acts.sink_first "
            f"({n_sink_win}/{n_all_win} windows start with the sink overall). See "
            f"_sink_uniform_features.",
            flush=True,
        )
        sel = [sel[i] for i in keep]
        feats = [feats[i] for i in keep]
        dirs_f = dirs_f[keep]

    enc = C.sae_dirs(sae, feats).cpu()
    dot = (dirs_f * enc).sum(-1)
    dmax = float((dirs_f - enc).abs().max())
    # Only `sae_side: enc` rows reach here (the selector above), which is what makes this
    # comparison meaningful: a decoder row's direction is W_dec[f], not unit(W_enc[:, f]).
    assert float(dot.min()) > 1 - 1e-3, (
        f"held-out vecs.f16 disagrees with unit(W_enc[:, f]) on at least one tested ENCODER "
        f"feature of {sae_key}: min cosine {float(dot.min()):.6f} < 1 - 1e-3 (max |elementwise "
        f"diff| {dmax:.2e})"
    )

    ids_t, acts_t, info = _load_windows(cfg, sae_key, tok, feats)
    n_feat, n_win, n_tok = ids_t.shape
    n_rows = n_feat * n_win
    flat_ids = ids_t.reshape(n_rows, n_tok)
    flat_acts = acts_t.reshape(n_rows, n_tok)
    repo_peak, repo_arg = flat_acts.max(dim=1)
    # stored f16-rounded: the payload is 512 x 30-32 x 32 activations and f16 is the precision
    # the 8B file itself was written in (bfloat16 there, so f16 loses nothing it still had).
    acts16 = flat_acts.half()
    row_feat = [f for f in feats for _ in range(n_win)]
    texts = [tok.decode(flat_ids[i].tolist(), skip_special_tokens=False) for i in range(n_rows)]
    fdirs = torch.repeat_interleave(dirs_f, n_win, dim=0)
    assert fdirs.shape == (n_rows, d), f"direction grid is {tuple(fdirs.shape)}, want {(n_rows, d)}"

    print(
        f"[repo_examples] {n_feat} tested features x {n_win} shipped windows = {n_rows} rows of "
        f"{n_tok} tokens, gate {gate:.4f}, sink_first={info['sink_first']}",
        flush=True,
    )
    extra = _Acts(sae, row_feat, n_rows)
    t0 = time.time()
    outs: dict[str, list] = {"cos": [], "keep": [], "ids": []}
    for s in range(0, n_rows, SCORE_ROWS):
        block = texts[s : s + SCORE_ROWS]
        res = C.score_tokens(
            model,
            tok,
            block,
            fdirs[s : s + len(block)],
            read_layer,
            on_chunk=(lambda i, h, cos, keep, ids, off=s: extra(off + i, h, cos, keep, ids)),
        )
        for k in outs:
            outs[k].append(res[k])
    scored = {k: torch.cat(v) for k, v in outs.items()}
    elapsed = time.time() - t0
    # The on_chunk offsets are arithmetic, so they are checked rather than trusted: every row must
    # have been scored with its OWN feature's encoder column.
    assert extra.seen.tolist() == row_feat, (
        "on_chunk row offsets are wrong: at least one window was scored with another feature's encoder column"
    )

    best, _ = C.agg(scored["cos"], scored["keep"])
    cos_np = scored["cos"].numpy()
    keep_np = scored["keep"].numpy()
    best_np = best.numpy()
    our_arg = extra.our_arg.numpy()
    peak_arg = extra.peak_arg.numpy()
    peak_act = extra.peak_act.numpy()
    act_at_arg = extra.act_at_arg.numpy()
    repo_peak_np = repo_peak.numpy()
    repo_arg_np = repo_arg.numpy()
    fired = peak_act > gate

    rows_out = []
    retok = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        shipped = flat_ids[i].tolist()
        got = [int(t) for t in scored["ids"][i].tolist() if t >= 0][1:]  # drop the prepended sink
        retok[i] = got == shipped
        # Their peak token is a position in THEIR tokenization; +1 for the sink at column 0.
        col = int(repo_arg_np[i]) + 1
        ours_there = float(cos_np[i, col]) if col < cos_np.shape[1] and keep_np[i, col] else float("nan")
        rows_out.append(
            {
                "feature": int(row_feat[i]),
                "row": int(sel[i // n_win]["row"]),
                "rank": int(i % n_win),
                "ids": shipped,
                "text": texts[i],
                "repo_acts": [round(float(a), 4) for a in acts16[i].tolist()],
                "repo_peak": round(float(repo_peak_np[i]), 4),
                "repo_argmax": int(repo_arg_np[i]),
                "cos": round(float(best_np[i]), 6),
                "argmax": int(our_arg[i]),
                "our_act_at_argmax": round(float(act_at_arg[i]), 4),
                "our_peak_act": round(float(peak_act[i]), 4),
                "our_peak_argmax": int(peak_arg[i]),
                "fired": bool(fired[i]),
                "cos_at_repo_peak": None if np.isnan(ours_there) else round(ours_there, 6),
                "retok_exact": bool(retok[i]),
                "n_scored_tok": int(keep_np[i].sum()),
            }
        )

    per_feature = []
    for j, r in enumerate(sel):
        sl = slice(j * n_win, (j + 1) * n_win)
        ok = retok[sl]
        agree = (repo_arg_np[sl] == peak_arg[sl])[ok]
        per_feature.append(
            {
                "feature": int(r["id"]),
                "row": int(r["row"]),
                "stratum": int(r.get("stratum", -1)),
                "n_windows": int(n_win),
                "mean_cos": round(float(best_np[sl].mean()), 6),
                "max_cos": round(float(best_np[sl].max()), 6),  # <- the "sae-repo-top32" column
                "mean_peak_act": round(float(peak_act[sl].mean()), 4),
                "repo_mean_peak": round(float(repo_peak_np[sl].mean()), 4),
                "frac_fired": round(float(fired[sl].mean()), 4),
                "peak_pearson_r": _pearson(repo_peak_np[sl], peak_act[sl]),
                "argmax_agree": round(float(agree.mean()), 4) if agree.size else None,
                "n_retok_exact": int(ok.sum()),
            }
        )

    n_diag = min(DIAG_FEATURES, n_feat)
    diag = _sink_diag(model, read_layer, sae, ids_t, acts_t, feats, n_diag)
    head = slice(0, n_diag * n_win)
    diag_rs = [p["peak_pearson_r"] for p in per_feature[:n_diag] if p["peak_pearson_r"] is not None]
    diag["mean_per_feature_peak_r_with_sink"] = round(float(np.mean(diag_rs)), 6) if diag_rs else None
    diag["median_ratio_ours_over_repo_with_sink"] = round(
        float(np.median(peak_act[head] / np.maximum(repo_peak_np[head], 1e-9))), 6
    )

    rs = [p["peak_pearson_r"] for p in per_feature if p["peak_pearson_r"] is not None]
    ag = [p["argmax_agree"] for p in per_feature if p["argmax_agree"] is not None]
    dead = int((repo_peak_np.reshape(n_feat, n_win).max(axis=1) == 0).sum())
    agree_all = (repo_arg_np == peak_arg)[retok]
    summary = {
        "baseline": "sae-repo-top32",
        "features": n_feat,
        "windows_per_feature": int(n_win),
        "mean_max_cos": round(float(np.mean([p["max_cos"] for p in per_feature])), 6),
        "mean_mean_cos": round(float(np.mean([p["mean_cos"] for p in per_feature])), 6),
        "frac_features_fired_at_least_once": round(
            float(np.mean([p["frac_fired"] > 0 for p in per_feature])), 6
        ),
        "frac_windows_fired": round(float(fired.mean()), 6),
        "gate": round(gate, 6),
        "mean_per_feature_peak_r": round(float(np.mean(rs)), 6) if rs else None,
        "median_per_feature_peak_r": round(float(np.median(rs)), 6) if rs else None,
        "min_per_feature_peak_r": round(float(np.min(rs)), 6) if rs else None,
        "n_features_with_r": len(rs),
        "pooled_peak_r": _pearson(repo_peak_np, peak_act),
        "mean_per_feature_argmax_agree": round(float(np.mean(ag)), 6) if ag else None,
        "argmax_agree_all_windows": round(float(agree_all.mean()), 6) if agree_all.size else None,
        "frac_retok_exact": round(float(retok.mean()), 6),
        "dead_in_repo_scan": dead,
        "features_requested": int(n_feat + len(dropped)),
        "features_dropped_sink_nonuniform": len(dropped),
        "features_dropped_ids": dropped,
        "windows_starting_with_sink_over_requested": [n_sink_win, n_all_win],
        "sink_diag": diag,
    }
    print(f"[repo_examples] {summary}", flush=True)

    inputs = {
        "max_acts file": info["path"],
        "repo": f"{info['repo']} ({info['repo_type']} repo)",
        "sae": sae_key,
        "sae checkpoint": C.sae_path(cfg, sae_key),
        "directions": f"{src}/vecs.f16 ({n_feat} sae rows of {len(rows_meta)})",
        "scored on": f"{spec['hf']} (clean base), read layer {read_layer}",
    }
    with C.outdir(out, args, inputs=inputs, provenance=info) as od:
        C.note_convention(od, cen_notes)
        od.write_jsonl("repo_examples.jsonl", rows_out)
        od.write_jsonl("per_feature.jsonl", per_feature)
        od.write_json("summary.json", summary)
        od.section(
            "What this is",
            [
                f"The SAE repo's OWN shipped max-activating windows for the {n_feat} features the",
                f"held-out set `{set_name}` tests, {n_win} per feature, pushed through the single",
                "`common.score_tokens` the MAEMM rollouts go through -- same truncation, same sink,",
                f"same fp32 cosine, same fixed chunk of {C.SCORE_CHUNK} rows. `per_feature.max_cos`",
                "is the **sae-repo-top32** baseline column: the best of the shipped windows, the",
                "direct analogue of a best-of-32 rollout draw.",
                "",
                "Provenance of the windows themselves: they are the repo authors' scan over THEIR",
                "corpus with THEIR window cut, not ours -- see the `config` line under Provenance",
                "for whatever the file itself records, and the notes below.",
            ],
        )
        od.section(
            "Sink handling",
            [
                f"config.yaml declares `max_acts.sink_first: {info['sink_first']}` for this file and",
                "`common.strip_repo_sink` asserts it against the data before anything is scored.",
                f"{info['frac_windows_starting_with_sink'] * 100:.4f}% of the shipped windows begin",
                f"with the tokenizer's sink token ({info['sink_token']}).",
                "",
                (
                    f"Column 0 WAS stripped, from the ids and from the repo's activations together: "
                    f"{info['tokens_per_window_shipped']} shipped tokens -> "
                    f"{info['tokens_per_window_scored']} scored. `score_tokens` prepends its own "
                    f"sink, so keeping theirs would have given two."
                    if info["sink_first"]
                    else (
                        f"NOTHING was stripped: all {info['tokens_per_window_shipped']} shipped ids "
                        f"are ordinary corpus tokens and `score_tokens` prepends the sink itself."
                    )
                ),
                "",
                f"{info['frac_shipped_argmax_at_position_0'] * 100:.2f}% of the shipped windows had",
                "their peak activation AT position 0 of the SHIPPED array"
                + (
                    " -- i.e. on the sink token itself, which our scoring never sees, so those "
                    "windows' peaks necessarily move."
                    if info["sink_first"]
                    else " -- an ordinary corpus token here, scored like any other."
                ),
            ],
        )
        od.section(
            "Agreement: their scan against our forward",
            [
                "This is the check that our activation formula and read layer match the scan that",
                "produced the file. Per feature, over its shipped windows, Pearson r of the repo's",
                "own peak activation against ours, and the fraction of windows whose argmax POSITION",
                "agrees (computed only over windows whose decoded text re-tokenizes to exactly the",
                "shipped ids, since otherwise the two positions index different tokenizations).",
                "",
                f"- mean per-feature Pearson r: {summary['mean_per_feature_peak_r']} "
                f"(median {summary['median_per_feature_peak_r']}, min {summary['min_per_feature_peak_r']}, "
                f"over {summary['n_features_with_r']} of {n_feat} features)",
                f"- pooled r over all {n_rows} windows: {summary['pooled_peak_r']}",
                f"- argmax agreement: {summary['mean_per_feature_argmax_agree']} mean per feature, "
                f"{summary['argmax_agree_all_windows']} over all round-tripping windows",
                f"- decoded text re-tokenizes to the shipped ids exactly on "
                f"{summary['frac_retok_exact'] * 100:.2f}% of windows",
                f"- {dead} of {n_feat} tested features are dead in the repo's own scan (peak 0)",
                "",
                "Sink diagnostic (DIAGNOSTIC ONLY -- it does not go through `score_tokens` and",
                f"changes no output above). The first {diag['features']} features' {diag['windows']}",
                "windows forwarded a second time on their RAW shipped ids, with NO sink prepended,",
                "which is the context the repo's own scan used when `sink_first: false`:",
                "",
                f"- mean per-feature Pearson r, WITH the sink (as scored): "
                f"{diag['mean_per_feature_peak_r_with_sink']}",
                f"- mean per-feature Pearson r, WITHOUT the sink: {diag['mean_per_feature_peak_r_no_sink']}",
                f"- median (our peak / their peak), with the sink: "
                f"{diag['median_ratio_ours_over_repo_with_sink']}; without: "
                f"{diag['median_ratio_ours_over_repo_no_sink']}",
                "",
                _sink_verdict(diag),
            ],
        )
        od.note(
            f"FEATURE COVERAGE: {n_feat} of the sae family's {n_feat + len(dropped)} features. "
            + (
                f"{len(dropped)} were DROPPED because not all of their shipped windows match "
                f"max_acts.sink_first={info['sink_first']} ({n_sink_win} of {n_all_win} requested "
                f"windows start with the sink token {info['sink_token']}). common.strip_repo_sink "
                f"refuses to strip column 0 off a mixed tensor -- on a window that does not carry "
                f"the sink it would delete a real token and shift every shipped activation by one "
                f"-- and a per-window strip would make the [F, W, T] grid ragged, so the unit of "
                f"exclusion is the feature. Their ids are in summary.json "
                f"(`features_dropped_ids`). This is a property of the SHIPPED FILE, not of the SAE "
                f"or of the held-out draw, and it biases nothing except which features the "
                f"baseline column covers."
                if dropped
                else "No feature was dropped: every shipped window matches max_acts.sink_first."
            )
        )
        od.note(
            f"`repo_examples.jsonl`: one row per (feature, rank). `ids` are the shipped ids AS "
            f"SCORED (post-strip) and `repo_acts` their per-token activations from the repo's scan, "
            f"aligned to them; `repo_peak`/`repo_argmax` are that row's max and argmax. `cos` is "
            f"OUR max cosine over kept tokens against the held-out set's direction for the feature "
            f"(asserted to equal unit(W_enc[:, f]) within 1e-3, min cosine {float(dot.min()):.6f}), "
            f"`argmax` its position, `our_act_at_argmax` the pre-gate SAE activation THERE, "
            f"`our_peak_act`/`our_peak_argmax` the max pre-gate activation over kept tokens and its "
            f"position, `fired` = our_peak_act > the checkpoint's learned gate {gate:.4f}, and "
            f"`cos_at_repo_peak` our cosine at THEIR peak token (null if that position is not a "
            f"kept token of our tokenization)."
        )
        od.note(
            "argmax conventions match `scores/<set>/argmax.i16`: 0 is the first token of the "
            "shipped window, the sink is already dropped, -1 means no kept token."
        )
        od.note(
            f"`per_feature.jsonl`: `max_cos` IS the sae-repo-top{n_win} baseline column "
            f"(mean over features {summary['mean_max_cos']}), `mean_cos` its per-window mean "
            f"({summary['mean_mean_cos']}), `frac_fired` the share of the feature's windows whose "
            f"peak activation clears the gate, and `peak_pearson_r` / `argmax_agree` the agreement "
            f"above. `peak_pearson_r` is null where the repo's peaks are constant over the windows."
        )
        od.note(
            f"fired: {summary['frac_windows_fired'] * 100:.2f}% of windows, and "
            f"{summary['frac_features_fired_at_least_once'] * 100:.2f}% of tested features fire on "
            f"at least one of their own shipped windows"
        )
        od.note(
            f"scoring: clean base, truncation {C.SCORE_MAX_LENGTH}, chunk {C.SCORE_CHUNK}, sink at "
            f"column 0 excluded from `keep`, fp32 cosine, NO norm filter -- byte for byte the "
            f"protocol `score` uses for rollouts. Windows are decoded with "
            f"skip_special_tokens=False so `text` and `ids` describe the same thing."
        )
        od.note(
            f"scoring wall {elapsed:.1f}s for {n_rows} windows ({n_rows / max(elapsed, 1e-9):.1f} rows/s)"
        )
    return {"out": out, **summary}
