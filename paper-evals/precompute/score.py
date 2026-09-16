"""Product `score`: score rollouts on the CLEAN BASE -> `<root>/maemms/<base>/<maemm>/scores/<set>/`.

    cos.f16      [N, n, T]   per-token cosine, NaN outside the kept tokens (T = common.SCORE_WIDTH)
    norm.f16     [N, n, T]   per-token residual norm, NaN in the same places
    argmax.i16   [N, n]      index of the best token AMONG THE SCORED TOKENS (0 = the first
                             generated token; the sink is already dropped), -1 if nothing was kept
    best_act.f16 [N, n, d]   the read-layer residual AT that token, from the same forward
    sae_idx.i32 / sae_val.f16 / sae_off.i64   flat CSR of the SAE features whose pre-gate
                             activation at the argmax token exceeds the checkpoint's learned gate
    per_target.jsonl         row, family, n, mean_cos, max_cos, mean_len, eos_rate, bo_<k> means

This product NEVER loads the MAEMM. The full-parameter protocol (eval/eval_ckpt_daemon.py:333-390)
is that the tuned model generates and the clean base scores; running the same clean-base path for
LoRA MAEMMs too means there is exactly ONE scoring path in the pipeline (checklist item 12).

Scoring protocol is common.score_tokens verbatim: right padding, add_special_tokens=False,
truncation at 95, a sink at column 0 excluded from `keep`, fp32 cosine, ONE fixed chunk of 32 rows,
and NO norm filter (the per-token norm is stored so reconstruction/stats.py can apply one).

`--engine hf|vllm` picks WHICH rollouts file of the set to score (`common.rollout_stem`): the two
engines write into the same accumulating `rollouts/` directory under different stems, and their
scores land in `scores/<set>/` and `scores/<set>__vllm/`. Nothing else about this product changes
with the engine -- the scoring model is the clean base either way, which is the whole point of the
paired comparison in `reconstruction/parity.py`.

`--rescore-texts <jsonl>` scores an arbitrary jsonl of {row, text} rows against the SAME set's
directions and writes a FLAT [M, T] variant plus `rows.jsonl`. That is how archived rollouts are
pushed through our scorer, which isolates the scorer from the sampler.

`--rollouts-dir <dir>` scores `<dir>/rollouts.jsonl` (+ `<dir>/rollouts.summary.json`) into
`<dir>/scores/` instead of a MAEMM's rollouts file. The rows must be the rollouts_hf schema and the
summary must carry `engine`, `n`, `seed`, `max_new` and `weight_sha256`; everything else -- the
clean-base scorer, the arrays, `per_target.jsonl` -- is identical, which is how a `patchscopes`
cell gets its numbers from THE scoring path rather than from a second implementation.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import precompute.common as C

BO_KS = (1, 2, 4, 8, 16, 32, 64)
# rows handed to common.score_tokens per call; it re-chunks internally at common.SCORE_CHUNK, so
# this only bounds the size of the fp32 residual the on_chunk callback sees at once.
SCORE_ROWS = 256


class _Extra:
    """Collects, from the scoring forward itself, the argmax position, the residual there, and the
    SAE features above the gate there. One instance per score_tokens call."""

    def __init__(self, sae, d, n_rows, gate):
        import torch

        self.sae, self.gate = sae, gate
        self.arg = torch.full((n_rows,), -1, dtype=torch.int64)
        self.best = torch.zeros((n_rows, d), dtype=torch.float16)
        self.idx: list = []
        self.val: list = []
        self.counts = torch.zeros((n_rows,), dtype=torch.int64)

    def __call__(self, s, h, cos, keep, ids):
        import torch

        best, arg = C.agg(cos, keep)
        del best
        b = h.shape[0]
        rows = torch.arange(b, device=h.device)
        hb = h[rows, arg]  # [b, d] fp32, the residual AT the best token
        has = keep.any(dim=1)
        self.arg[s : s + b] = torch.where(has, arg - 1, torch.full_like(arg, -1)).cpu()
        self.best[s : s + b] = torch.where(has.unsqueeze(-1), hb, torch.zeros_like(hb)).cpu().half()
        if self.sae is None:
            return
        a = torch.relu((hb - self.sae.b_dec) @ self.sae.W_enc + self.sae.b_enc)  # [b, F] pre-gate
        a = torch.where(has.unsqueeze(-1), a, torch.zeros_like(a))
        hit = a > self.gate
        self.counts[s : s + b] = hit.sum(1).cpu()
        r, f = hit.nonzero(as_tuple=True)
        order = torch.argsort(r * self.sae.d_sae + f)
        self.idx.append(f[order].cpu().numpy().astype(np.int32))
        self.val.append(a[r[order], f[order]].cpu().numpy().astype(np.float16))


def _load_dirs(cfg, args):
    """(rows, dirs [N, d] fp32 unit on the cpu, source dir) -- the same source rollouts_hf used."""
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    d = cfg["bases"][base]["d"]
    src = args.get("dirs_from") or C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{src}/ids.jsonl")
    v = C.read_array(f"{src}/vecs.f16", "float16", (len(rows), d)).astype(np.float32)
    return rows, torch.nn.functional.normalize(torch.from_numpy(v), dim=-1), src


def _score_all(model, tok, texts, dirs, read_layer, extra):
    """common.score_tokens over `texts` in SCORE_ROWS slices, concatenated. `extra` sees every
    chunk's residual (its row offsets are shifted back to the global row index here)."""
    import torch

    outs: dict[str, list] = {"cos": [], "norm": [], "keep": [], "ids": []}
    for s in range(0, len(texts), SCORE_ROWS):
        block = texts[s : s + SCORE_ROWS]
        out = C.score_tokens(
            model,
            tok,
            block,
            dirs[s : s + len(block)],
            read_layer,
            on_chunk=(lambda i, h, cos, keep, ids, off=s: extra(off + i, h, cos, keep, ids)),
        )
        for k in outs:
            outs[k].append(out[k])
    return {k: torch.cat(v) for k, v in outs.items()}


def _sae_for(cfg, args):
    """(sae, key) for the base, or (None, '') when --no-sae. The gate is the checkpoint's own."""
    import torch

    if args.get("no_sae"):
        return None, ""
    base = args["base"]
    keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    assert len(keys) == 1, f"base {base} has {len(keys)} SAEs in config, expected exactly 1"
    sae = C.load_sae(C.sae_path(cfg, keys[0]), cfg["bases"][base]["d"], device="cuda", dtype=torch.float32)
    return sae, keys[0]


def _check_scored_is_generation(out, texts, tok, max_new, od, gen_n_tok=None, prompt_tokens=None):
    """Checklist item 8: what was scored is the ROLLOUT, never the inverter's prompt.

    Two hard bounds and one measurement.

      1. The stored GENERATED ids are at most `max_new` long (exact -- they are the ids the sampler
         returned, trimmed at the stop token). Skipped in rescore mode, where the texts are somebody
         else's and no id list came with them.
      2. No scored row reaches the `SCORE_MAX_LENGTH` truncation. The MAEMM prompt is ~103 tokens,
         so a row that carried it would be truncated to exactly 95 kept tokens.

    MEASURED 2026-09-15: a row's decoded text does NOT always re-tokenize to the same number of ids
    (a rollout cut mid-word at max_new re-encodes to one token more), so `kept <= max_new` is NOT a
    valid bound and was replaced by the two above. The round-trip equality of the TEXT is reported
    as a rate rather than asserted, for the same reason.

    Bound 2 IS ONLY A LEAK DETECTOR WHERE A LEAK COULD REACH IT, and it now checks that itself.
    `prompt_tokens` is the producing run's own prompt length (every rollouts summary carries it).
    When `prompt_tokens + max_new < SCORE_MAX_LENGTH`, a row carrying the WHOLE prompt plus the
    whole generation would still land under the truncation, so hitting it cannot be a leak -- it is
    re-tokenization expansion, and asserting on it would be asserting on the wrong thing. MEASURED
    2026-09-16: `precompute/patchscopes.py` generates from a 30-token prompt (30 + 64 = 94 < 95) and
    its base-model continuations re-encode past 95 often enough to abort the product, while bound 1
    -- the exact one -- passes on every row. In that case the count of truncated rows is REPORTED
    instead, because truncation is still a real measurement caveat: those rows' cosine is a max over
    a shortened window, biased down. Where the prompt could reach the truncation (the MAEMM path,
    ~103 + 64) the assert is unchanged.
    """
    kept = out["keep"].sum(1)
    worst = int(kept.max())
    if gen_n_tok is not None:
        longest = max(gen_n_tok)
        assert longest <= max_new, (
            f"a stored rollout has {longest} generated ids but max_new={max_new}: the rollout file "
            f"holds something that is not the generation alone (checklist item 8)"
        )
    leak_would_reach = prompt_tokens is None or (int(prompt_tokens) + int(max_new)) >= C.SCORE_MAX_LENGTH
    if leak_would_reach:
        assert worst < C.SCORE_MAX_LENGTH, (
            f"a scored row kept {worst} tokens, i.e. it hit the {C.SCORE_MAX_LENGTH}-token "
            f"truncation; the {prompt_tokens or '~103'}-token prompt of this run looks exactly "
            f"like this, while a {max_new}-token rollout cannot (checklist item 8)"
        )
    else:
        n_trunc = int((kept >= C.SCORE_MAX_LENGTH).sum())
        od.note(
            f"checklist item 8, bound 2 NOT APPLICABLE here and therefore not asserted: this run's "
            f"prompt is {prompt_tokens} tokens, so prompt + max_new = {int(prompt_tokens) + int(max_new)} "
            f"< {C.SCORE_MAX_LENGTH} and a row carrying the whole prompt could not reach the "
            f"truncation. Bound 1 (stored generated ids <= max_new) is the exact guard and passed. "
            f"REPORTED instead: {n_trunc} of {len(texts)} scored rows "
            f"({n_trunc / max(len(texts), 1):.2%}) hit the {C.SCORE_MAX_LENGTH}-token truncation "
            f"through re-tokenization expansion, so their cosine is a max over a SHORTENED window "
            f"and is biased DOWN."
        )
    bad = 0
    first = ""
    for i, txt in enumerate(texts):
        if not txt.strip() or int(kept[i]) == 0:
            continue  # score_tokens substitutes " " for an all-whitespace rollout
        ids = [int(t) for t in out["ids"][i].tolist() if t >= 0][1:]  # drop the sink
        if tok.decode(ids) != txt:
            bad += 1
            first = first or f"row {i}: scored {tok.decode(ids)!r} vs stored {txt!r}"
    od.note(
        f"checklist item 8: max kept tokens per scored row {worst} < the {C.SCORE_MAX_LENGTH}-token "
        f"truncation (asserted, so no row carried the ~103-token prompt)"
        + (
            f"; longest stored generation {max(gen_n_tok)} <= max_new {max_new} (asserted)"
            if gen_n_tok is not None
            else ""
        )
        + f"; decode(scored ids) == the stored text on {len(texts) - bad}/{len(texts)} rows "
        f"(re-tokenizing a decoded rollout is not length-preserving)"
        + (f" -- first mismatch {first}" if bad else "")
    )
    return bad


def _rescore(cfg, args, model, tok, rows_meta, dirs, read_layer, sae, sae_key, dirs_src):
    """`--rescore-texts`: score an arbitrary jsonl of {row, text} against the set's directions."""
    import torch

    src = args["rescore_texts"]
    recs = C.read_jsonl(src)
    assert recs, f"{src} has no rows"
    name = args.get("score_name") or f"{args['heldout']}__rescore-{os.path.basename(src).split('.')[0]}"
    out = C.scores_dir(args["maemm"], name, args["root"])
    for r in recs:
        assert "row" in r and "text" in r, f"{src}: every row needs 'row' and 'text', got {sorted(r)}"
        assert 0 <= r["row"] < len(rows_meta), f"{src}: row {r['row']} is outside 0..{len(rows_meta) - 1}"
    texts = [r["text"] for r in recs]
    rdirs = torch.stack([dirs[r["row"]] for r in recs])
    d = cfg["bases"][args["base"]]["d"]
    extra = _Extra(None, d, len(recs), 0.0)
    t0 = time.time()
    res = _score_all(model, tok, texts, rdirs, read_layer, extra)
    best, _ = C.agg(res["cos"], res["keep"])
    elapsed = time.time() - t0
    inputs = {
        "rescored": src,
        "rows": len(recs),
        "dirs": dirs_src,
        "maemm": args["maemm"],
        "sae": sae_key or "(not used)",
    }
    with C.outdir(out, args, inputs=inputs) as od:
        bad = _check_scored_is_generation(res, texts, tok, C.SCORE_MAX_LENGTH - 1, od)
        od.write_array("cos.f16", res["cos"], "float16")
        od.write_array("norm.f16", res["norm"], "float16")
        od.write_array("argmax.i16", extra.arg.numpy().astype(np.int16), "int16")
        od.write_jsonl(
            "rows.jsonl",
            [
                {
                    **r,
                    "scored_row": i,
                    "cos": round(float(best[i]), 6),
                    "argmax": int(extra.arg[i]),
                    "n_scored_tok": int(res["keep"][i].sum()),
                }
                for i, r in enumerate(recs)
            ],
        )
        od.note(
            f"RESCORE mode: the {len(recs)} texts of {src} scored against `{dirs_src}` row by row "
            "(each text's `row` picks its direction). Arrays are FLAT [M, T] here, not [N, n, T]: "
            "the input jsonl need not be a full n-rollout grid."
        )
        od.note(
            f"scoring: clean base, {C.SCORE_MAX_LENGTH}-token truncation, chunk {C.SCORE_CHUNK}, "
            f"sink at column 0 dropped from `keep`, fp32 cosine, NO norm filter; T={C.SCORE_WIDTH}"
        )
        od.note(
            "`rows.jsonl` echoes every input key and adds cos (max over kept tokens), argmax "
            "(among the scored tokens) and n_scored_tok"
        )
        od.note(f"no best_act / SAE arrays in rescore mode; {bad} tokenizer round-trip mismatches")
        od.note(f"wall {elapsed:.1f}s for {len(recs)} rows")
    return {"out": out, "rows": len(recs), "mean_cos": round(float(best.mean()), 6)}


def run(cfg, args):
    import torch

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    # `--rollouts-dir <dir>` scores <dir>/rollouts.jsonl + <dir>/rollouts.summary.json into
    # <dir>/scores/: rows a NON-MAEMM producer wrote in the rollouts_hf schema (today
    # precompute/patchscopes.py). Everything else about this product is unchanged -- same clean
    # base, same protocol, same outputs -- which is the point of the flag (checklist item 12).
    rdir = args.get("rollouts_dir") or ""
    assert base, "product score needs --base"
    assert maemm or rdir, "product score needs --maemm (the rollouts it scores), or --rollouts-dir"
    assert not maemm or maemm in cfg["maemms"], (
        f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    )
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]

    rows_meta, dirs, dirs_src = _load_dirs(cfg, args)
    model, tok = C.load_base(cfg, base)  # CLEAN BASE ONLY -- the MAEMM is never loaded here
    sae, sae_key = _sae_for(cfg, args)

    if args.get("rescore_texts"):
        return _rescore(cfg, args, model, tok, rows_meta, dirs, read_layer, sae, sae_key, dirs_src)

    engine = args.get("engine") or "hf"
    if rdir:
        rpath, spath = f"{rdir}/rollouts.jsonl", f"{rdir}/rollouts.summary.json"
    else:
        stem = C.rollout_stem(set_name, engine)
        rpath = C.rollouts_path(maemm, set_name, root, engine)
        spath = f"{C.rollouts_dir(maemm, root)}/{stem}.summary.json"
    assert os.path.exists(rpath), (
        f"no rollouts at {rpath}: run `--product rollouts_{engine} --set {set_name}` first"
    )
    recs = C.read_jsonl(rpath)
    with open(spath) as fh:
        rsum = json.load(fh)
    if rdir:
        # the directory names its own producer (e.g. "hf-patchscope"); --engine does not apply
        engine = rsum["engine"]
    else:
        assert rsum["engine"] == engine, (
            f"{rpath} was produced by engine {rsum['engine']!r} but --engine says {engine!r}"
        )
    by_row: dict[int, dict[int, dict]] = {}
    for r in recs:
        by_row.setdefault(r["row"], {})[r["k"]] = r
    present = sorted(by_row)
    sel = [r for r in C.parse_rows(args.get("rows", ""), len(rows_meta)) if r in by_row]
    assert sel, f"{rpath} holds rows {present[:8]}...; --rows {args.get('rows', '')!r} selected none of them"
    n = int(rsum["n"])
    for r in sel:
        ks = sorted(by_row[r])
        assert ks == list(range(n)), (
            f"target row {r} has rollouts {ks[:4]}..{ks[-4:]} but the summary says n={n}: the "
            f"[N, n, T] arrays need a complete, gap-free rollout grid"
        )

    flat = [by_row[r][k] for r in sel for k in range(n)]
    texts = [x["text"] for x in flat]
    fdirs = torch.stack([dirs[x["row"]] for x in flat])
    gate = float(sae.threshold) if sae is not None else 0.0
    extra = _Extra(sae, d, len(flat), gate)
    print(f"[score] {len(sel)} targets x {n} rollouts = {len(flat)} rows on the clean base", flush=True)
    t0 = time.time()
    res = _score_all(model, tok, texts, fdirs, read_layer, extra)
    best, _ = C.agg(res["cos"], res["keep"])
    elapsed = time.time() - t0

    N = len(sel)
    cos = res["cos"].numpy().astype(np.float16).reshape(N, n, C.SCORE_WIDTH)
    nrm = res["norm"].numpy().astype(np.float16).reshape(N, n, C.SCORE_WIDTH)
    argmax = extra.arg.numpy().astype(np.int16).reshape(N, n)
    best_act = extra.best.numpy().reshape(N, n, d)
    counts = extra.counts.numpy()
    sae_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sae_idx = np.concatenate(extra.idx) if extra.idx else np.zeros(0, dtype=np.int32)
    sae_val = np.concatenate(extra.val) if extra.val else np.zeros(0, dtype=np.float16)
    assert len(sae_idx) == sae_off[-1], (
        f"CSR is inconsistent: {len(sae_idx)} feature entries but the offsets end at {sae_off[-1]}"
    )
    bestm = best.numpy().reshape(N, n)

    per_target = []
    for i, r in enumerate(sel):
        vals = bestm[i].tolist()
        lens = [by_row[r][k]["n_tok"] for k in range(n)]
        fin = [by_row[r][k]["finished"] for k in range(n)]
        bo = C.best_of_k_means(vals, BO_KS)
        per_target.append(
            {
                "row": r,
                "family": rows_meta[r]["family"],
                "n": n,
                "bo": n,
                "seed": int(rsum["seed"]),
                "checkpoint_sha": rsum["weight_sha256"],
                "mean_cos": round(float(np.mean(vals)), 6),
                "max_cos": round(float(np.max(vals)), 6),
                "mean_len": round(float(np.mean(lens)), 3),
                "eos_rate": round(float(np.mean(fin)), 4),
                "n_sae_gated": int(counts[i * n : (i + 1) * n].sum()),
                **{f"bo_{k}": round(v, 6) for k, v in bo.items()},
            }
        )

    out = f"{rdir}/scores" if rdir else C.scores_dir(maemm, set_name, root, engine)
    inputs = {
        "rollouts": rpath,
        "engine": engine,
        "dirs": dirs_src,
        "maemm": maemm or f"(none: --rollouts-dir {rdir})",
        "weight sha256": rsum["weight_sha256"],
        "sae": sae_key or "(none: --no-sae)",
        "targets": f"{N} of {len(rows_meta)} rows",
        "n": n,
    }
    with C.outdir(out, args, inputs=inputs) as od:
        bad = _check_scored_is_generation(
            res,
            texts,
            tok,
            int(rsum["max_new"]),
            od,
            gen_n_tok=[x["n_tok"] for x in flat],
            prompt_tokens=rsum.get("prompt_tokens"),
        )
        od.write_array("cos.f16", cos, "float16")
        od.write_array("norm.f16", nrm, "float16")
        od.write_array("argmax.i16", argmax, "int16")
        od.write_array("best_act.f16", best_act, "float16")
        od.write_array("sae_idx.i32", sae_idx, "int32")
        od.write_array("sae_val.f16", sae_val, "float16")
        od.write_array("sae_off.i64", sae_off, "int64")
        od.write_jsonl("per_target.jsonl", per_target)
        od.write_json("rows.json", {"rows": sel, "n": n, "families": [rows_meta[r]["family"] for r in sel]})
        od.note(
            f"scored on the CLEAN BASE ({cfg['bases'][base]['hf']}); the MAEMM is never loaded by "
            "this product. Protocol: common.score_tokens -- padding_side right, "
            f"add_special_tokens=False, truncation at {C.SCORE_MAX_LENGTH}, sink at column 0 "
            f"excluded from `keep`, fp32 cosine, ONE fixed chunk of {C.SCORE_CHUNK} rows, NO norm "
            f"filter (the per-token norm is stored instead). T = {C.SCORE_WIDTH}, NaN outside `keep`."
        )
        od.note(
            "`argmax.i16` indexes the SCORED tokens (0 = the first generated token; the sink is "
            "already dropped), and is -1 for a row with no kept token; `best_act.f16` is the "
            "read-layer residual AT that token, taken from the SAME forward as the cosine."
        )
        od.note(
            f"SAE ({sae_key or 'none'}): flat CSR over the (target, rollout) grid in row-major "
            f"order -- for row j, sae_idx[sae_off[j]:sae_off[j+1]] are the features whose PRE-GATE "
            f"activation at the ARGMAX TOKEN ONLY exceeds the checkpoint's learned gate "
            f"({gate:.4f}), with sae_val the activations. {int(counts.sum())} entries over "
            f"{len(flat)} rows (mean {counts.mean():.1f}). The per-TOKEN variant is deliberately "
            f"NOT written: at the mean {np.mean([x['n_tok'] for x in flat]):.0f} tokens per rollout "
            f"it would be ~{counts.mean() * np.mean([x['n_tok'] for x in flat]) * 6 * 512 * n / 1e6:.0f} "
            f"MB at the full 512-target set, over the ~200 MB this product is willing to write."
        )
        od.note(
            f"`per_target.jsonl` carries n, bo, seed and the checkpoint sha on every row "
            f"(checklist item 29). `bo_<k>` is the mean over DISJOINT groups of k consecutive "
            f"rollouts of the group max (common.best_of_k_means), k in {list(BO_KS)}; the unbiased "
            "order-statistic estimator belongs to reconstruction/stats.py, not here."
        )
        od.note(f"tokenizer round-trip mismatches: {bad} of {len(texts)} rows")
        od.note(
            f"scoring wall {elapsed:.1f}s for {len(flat)} rows ({len(flat) / max(elapsed, 1e-9):.1f} rows/s)"
        )
    fam_mean: dict[str, float] = {}
    for fam in sorted({r["family"] for r in per_target}):
        sub = [r for r in per_target if r["family"] == fam]
        fam_mean[fam] = round(float(np.mean([r["mean_cos"] for r in sub])), 5)
        fam_mean[f"{fam}_bo{n}"] = round(float(np.mean([r[f"bo_{n}"] for r in sub])), 5)
    return {"out": out, "engine": engine, "targets": N, "n": n, "rows": len(flat), "family_means": fam_mean}
