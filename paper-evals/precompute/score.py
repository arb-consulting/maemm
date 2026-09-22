"""Product `score`: score rollouts on the CLEAN BASE -> `<root>/maemms/<base>/<maemm>/scores/<set>/`.

    cos.f16      [N, n, T]   per-token cosine, NaN outside the kept tokens (T = common.SCORE_WIDTH,
                             except where the rollouts summary carries its own score_max_length --
                             the NLA arm; rows.json records it and common.score_width_of reads it)
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
# Checklist item 8, bound 2: the fraction of scored rows allowed to reach common.SCORE_MAX_LENGTH.
# A prompt leak puts ~100% of rows over that window (the prompt alone is ~103 tokens against 95),
# so this is 20x below the signature it exists to catch. What it tolerates is re-tokenization
# expansion, which reaches the truncation only for a row that ran to max_new AND blows up ~1.5x
# doing so -- MEASURED 2026-09-16 on the untrained-base control, whose clean-base rollouts carry
# enough rare unicode to put a few of 98,304 rows over. See _check_scored_is_generation.
TRUNC_FRAC_MAX = 0.05


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


def _load_dirs(cfg, args, notes=None):
    """(rows, dirs, dirs_centred, mu, source dir) -- the two target tensors of the two cosines.

    `dirs` is the target of `cos` (the historical, uncentred-scorer number) and `dirs_centred` the
    target of `cos_centred`, both [N, d] fp32 unit on the cpu:

      * on a `storage: raw` set, `dirs` is unit(act) and `dirs_centred` is unit(act - mu), derived
        from the one act.f32 at read time;
      * on a legacy `storage: unit` set there is no act.f32 to derive from, so BOTH are the stored
        direction -- which is exactly right: the stored row already IS unit(act - mu) for that
        set's own mean, so `cos` reproduces every number measured before 2026-09-21 to the digit
        and `cos_centred` is that same target with the SCORER's side centred too.

    Rows of a family that is not `centrable` (config.yaml `family_kinds:`) are NaN in
    `dirs_centred`, so `cos_centred` is NaN for them rather than a one-sided number: an encoder
    column, a Gaussian draw and a subspace basis have no mean, and cos(h - mu, encoder column)
    measures the activation moving while the target stands still.

    `mu` is None when this run centres on nothing, and then there is no second cosine at all.
    """
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    d = cfg["bases"][base]["d"]
    src = args.get("dirs_from") or C.heldout_dir(base, set_name, root)
    rows = C.read_jsonl(f"{src}/ids.jsonl")
    mu_val, _ = C.mu_for(cfg, base, src, args, args.get("maemm") or "", root, notes)
    contract = C.set_storage(cfg, src, root)
    # On a raw set the uncentred target is unit(act); on a legacy set it is the stored row, whose
    # own mean this run has already been asserted to match.
    raw_mu = None if contract["storage"] == "raw" else mu_val
    v = C.dirs_for(cfg, base, src, raw_mu, root, notes)
    assert v.shape == (len(rows), d), f"{src}: dirs_for returned {v.shape} for {len(rows)} rows"
    dirs = torch.nn.functional.normalize(torch.from_numpy(np.asarray(v)), dim=-1)

    mu = C.load_mu(cfg, base, mu_val, root)
    if mu is None:
        (notes if notes is not None else []).append(
            "no centred cosine in this directory: this run centres on nothing (mu=none), so "
            "cos_centred would be the uncentred number under another name"
        )
        return rows, dirs, None, None, src
    vc = np.array(C.dirs_for(cfg, base, src, mu_val, root, notes), dtype=np.float32, copy=True)
    not_centrable = [i for i, r in enumerate(rows) if not C.family_centrable(cfg, r["family"])]
    vc[not_centrable] = np.nan
    (notes if notes is not None else []).append(
        f"cos_centred: both sides centred on {C.mu_label(mu_val, base, root)}; "
        f"{len(not_centrable)} of {len(rows)} rows are NaN there because their family is not "
        f"`centrable` (an encoder column / a Gaussian draw / a subspace basis has no mean, so a "
        f"one-sided cos(h - mu, v) is not a centred number and is not reported as one)"
    )
    dirs_centred = torch.from_numpy(vc)
    return rows, dirs, dirs_centred, torch.from_numpy(mu), src


def _score_all(model, tok, texts, dirs, read_layer, extra, max_length=C.SCORE_MAX_LENGTH,
               dirs_centred=None, mu=None):
    """common.score_tokens over `texts` in SCORE_ROWS slices, concatenated. `extra` sees every
    chunk's residual (its row offsets are shifted back to the global row index here).

    `max_length` is this RUN's re-encode truncation, taken from the rollouts summary in `run`; it
    is the protocol's SCORE_MAX_LENGTH for every arm but the NLA one."""
    import torch

    keys = ["cos", "norm", "keep", "ids"] + (
        ["cos_centred", "cos_asym"] if dirs_centred is not None else []
    )
    outs: dict[str, list] = {k: [] for k in keys}
    for s in range(0, len(texts), SCORE_ROWS):
        block = texts[s : s + SCORE_ROWS]
        out = C.score_tokens(
            model,
            tok,
            block,
            dirs[s : s + len(block)],
            read_layer,
            on_chunk=(lambda i, h, cos, keep, ids, off=s: extra(off + i, h, cos, keep, ids)),
            max_length=max_length,
            dirs_centred=None if dirs_centred is None else dirs_centred[s : s + len(block)],
            mu=mu,
        )
        for k in outs:
            outs[k].append(out[k])
    return {k: torch.cat(v) for k, v in outs.items()}


def _sae_for(cfg, args, rows_meta=None):
    """(sae, key) for the base, or (None, '') when --no-sae. The gate is the checkpoint's own.

    `--sae <base>/<name>` picks WHICH SAE of the base, and is required as soon as the base has
    more than one: `qwen36-27b` has carried two since `sae2m` landed, and without this flag
    `score` could not run on that base at all -- the single-SAE assert below fired before any
    argument could say which to use. With one SAE the flag is optional and the assert is unchanged.

    The SAE is loaded ENCODER-ONLY (`need_decoder=False`). Nothing on the scoring path reads
    `W_dec`: the gating goes through `common.sae_encode`, which is b_dec, W_enc and b_enc. At
    2^21 features W_dec is another 43 GB in fp32, and 86 GB beside the 27B's ~52 GB does not fit
    an H200's 141 (features/CHANGES.md fix 2 made the same change for `stats`). `--no-sae` is
    still there for a run that wants no SAE features at all; the cosine, the norms and the argmax
    -- everything the tables read -- do not involve the SAE either way.
    """
    import torch

    if args.get("no_sae"):
        return None, ""
    base = args["base"]
    # NO SAE ROWS, NO SAE. A base with two dictionaries makes `sae_key_for` refuse without
    # `--sae`, which is right when feature ids are about to be looked up in one of them and wrong
    # when the set has none -- the OOD sets have no `sae` family at all, and every one of their
    # `score` calls died here after the base model load. `common.sae_key_for_rows` is the one
    # place that rule lives; `scan` takes it too.
    key = C.sae_key_for_rows(cfg, base, rows_meta or [], args.get("sae") or "")
    if not key:
        print("[score] the set has no sae rows: no SAE loaded, no gate counts", flush=True)
        return None, ""
    # ENCODER ONLY: `_Extra` gates on `common.sae_encode`, which reads b_dec, W_enc and b_enc and
    # never W_dec (grepped: nothing under paper-evals/ outside common.load_sae itself touches
    # W_dec). At 2^21 features W_dec is 43 GB in fp32, which is the difference between fitting an
    # H200 beside the 27B and not -- the same fix features/CHANGES.md item 2 made for `stats`.
    sae = C.load_sae(
        C.sae_path(cfg, key),
        cfg["bases"][base]["d"],
        device="cuda",
        dtype=torch.float32,
        need_decoder=False,
    )
    return sae, key


def _check_scored_is_generation(
    out, texts, tok, max_new, od, gen_n_tok=None, prompt_tokens=None, max_length=C.SCORE_MAX_LENGTH
):
    """Checklist item 8: what was scored is the ROLLOUT, never the inverter's prompt.

    Two hard bounds and one measurement.

      1. The stored GENERATED ids are at most `max_new` long (exact -- they are the ids the sampler
         returned, trimmed at the stop token). Skipped in rescore mode, where the texts are somebody
         else's and no id list came with them.
      2. No scored row reaches THIS RUN's re-encode truncation, `max_length` (the protocol's
         SCORE_MAX_LENGTH = 95 for every arm but the NLA one, which scores at 256 -- see
         `common.encode_for_score`). The MAEMM prompt is ~103 tokens, so a row that carried it
         would be truncated to exactly 95 kept tokens at the protocol width.

    MEASURED 2026-09-15: a row's decoded text does NOT always re-tokenize to the same number of ids
    (a rollout cut mid-word at max_new re-encodes to one token more), so `kept <= max_new` is NOT a
    valid bound and was replaced by the two above. The round-trip equality of the TEXT is reported
    as a rate rather than asserted, for the same reason.

    Bound 2 is a RATE bound, not a zero-tolerance one, because the two things that reach the
    truncation have opposite shapes. A prompt leak puts EVERY row over it -- the 103-token prompt
    alone exceeds the 95-token window, so the rate would be ~100%. Re-tokenization expansion puts a
    HANDFUL of rows over it: only a row that ran to max_new can expand past 95 at all, and it has
    to blow up 1.5x doing so. MEASURED 2026-09-16: the untrained-base control
    (`2026-09-16_base-control`, a clean Qwen3.6-27B given the inverter prompt) emits enough rare
    unicode to put a few of its 98,304 rows over, and the zero-tolerance form aborted the whole
    product on them although bound 1 -- the exact one -- passed on every row. The bound is
    therefore `truncated rows <= TRUNC_FRAC_MAX of all rows`, 20x below the leak signature, and the
    count is REPORTED whenever it is non-zero: a truncated row's cosine is a max over a shortened
    window and is biased DOWN.

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
    n_trunc = int((kept >= max_length).sum())
    frac_trunc = n_trunc / max(len(texts), 1)
    if gen_n_tok is not None:
        longest = max(gen_n_tok)
        assert longest <= max_new, (
            f"a stored rollout has {longest} generated ids but max_new={max_new}: the rollout file "
            f"holds something that is not the generation alone (checklist item 8)"
        )
    leak_would_reach = prompt_tokens is None or (int(prompt_tokens) + int(max_new)) >= max_length
    if leak_would_reach:
        assert frac_trunc <= TRUNC_FRAC_MAX, (
            f"{n_trunc} of {len(texts)} scored rows ({frac_trunc:.2%}) kept "
            f"{max_length} tokens, i.e. hit the truncation, above the "
            f"{TRUNC_FRAC_MAX:.0%} bound; the {prompt_tokens or '~103'}-token prompt of this run "
            f"looks exactly like this and would put ~100% of rows over, while a {max_new}-token "
            f"rollout can only get there by re-tokenization expansion, which is rare "
            f"(checklist item 8)"
        )
        if n_trunc:
            od.note(
                f"checklist item 8, bound 2: {n_trunc} of {len(texts)} scored rows "
                f"({frac_trunc:.2%}) hit the {max_length}-token truncation through "
                f"re-tokenization expansion -- under the {TRUNC_FRAC_MAX:.0%} bound a prompt leak "
                f"would blow through (it puts ~100% of rows over), and bound 1 (stored generated "
                f"ids <= max_new) passed on every row. Those rows' cosine is a max over a "
                f"SHORTENED window and is biased DOWN."
            )
    else:
        od.note(
            f"checklist item 8, bound 2 NOT APPLICABLE here and therefore not asserted: this run's "
            f"prompt is {prompt_tokens} tokens, so prompt + max_new = {int(prompt_tokens) + int(max_new)} "
            f"< {max_length} and a row carrying the whole prompt could not reach the "
            f"truncation. Bound 1 (stored generated ids <= max_new) is the exact guard and passed. "
            f"REPORTED instead: {n_trunc} of {len(texts)} scored rows "
            f"({n_trunc / max(len(texts), 1):.2%}) hit the {max_length}-token truncation "
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
        f"checklist item 8: max kept tokens per scored row {worst}, {n_trunc} of {len(texts)} rows "
        f"({frac_trunc:.2%}) at the {max_length}-token truncation ("
        + (
            f"asserted <= {TRUNC_FRAC_MAX:.0%}, so no row carried the ~103-token prompt"
            if leak_would_reach
            else "NOT asserted: a leak could not reach the truncation from this run's prompt"
        )
        + ")"
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


def _rescore(cfg, args, model, tok, rows_meta, dirs, read_layer, sae, sae_key, dirs_src, notes=None):
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
        C.note_convention(od, notes)
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

    cen_notes: list[str] = []
    rows_meta, dirs, dirs_centred, mu, dirs_src = _load_dirs(cfg, args, cen_notes)
    model, tok = C.load_base(cfg, base)  # CLEAN BASE ONLY -- the MAEMM is never loaded here
    sae, sae_key = _sae_for(cfg, args, rows_meta)

    if args.get("rescore_texts"):
        return _rescore(
            cfg, args, model, tok, rows_meta, dirs, read_layer, sae, sae_key, dirs_src, cen_notes
        )

    engine = args.get("engine") or "hf"
    if rdir:
        rpath, spath = f"{rdir}/rollouts.jsonl", f"{rdir}/rollouts.summary.json"
        assert os.path.exists(rpath), (
            f"no rollouts at {rpath}: run `--product rollouts_{engine} --set {set_name}` first"
        )
        recs = C.read_jsonl(rpath)
        with open(spath) as fh:
            rsum = json.load(fh)
        sources = [rpath]
    else:
        tag = args.get("run_tag") or ""
        stem = C.rollout_stem(set_name, engine, tag)
        # ONE product, whether it was generated in one call or in `--rows` chunks under the one
        # run tag: `common.read_rollouts` concatenates the chunks and merges their summaries, so
        # nothing downstream of here -- scores/, results.common.discover_sources, either OOD
        # reader -- learns that the generation was chunked (common.rollout_chunk_stem).
        recs, rsum, sources = C.read_rollouts(C.rollouts_dir(maemm, root), stem)
        rpath = sources[0] if len(sources) == 1 else f"{C.rollouts_dir(maemm, root)}/{stem}[chunked]"
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
    # The re-encode truncation THIS run scores at. Every arm but the NLA one omits the field and
    # gets the protocol's 95 (common.SCORE_MAX_LENGTH); `rollouts_nla` writes 256 because it
    # generates at the checkpoint's native 200 tokens and a 95-token cut would score less than
    # half of each text. It is carried on the ROLLOUTS summary, not passed as a flag, so the
    # scorer cannot be pointed at a width the generation never agreed to.
    max_length = int(rsum.get("score_max_length", C.SCORE_MAX_LENGTH))
    assert max_length >= int(rsum["max_new"]) + 1, (
        f"{rpath} generated at max_new={rsum['max_new']} but its summary asks to score at "
        f"max_length={max_length}: the window must leave room for the whole generation plus the "
        f"sink, or the tail of a full-length rollout is never scored"
    )
    for r in sel:
        ks = sorted(by_row[r])
        assert ks == list(range(n)), (
            f"target row {r} has rollouts {ks[:4]}..{ks[-4:]} but the summary says n={n}: the "
            f"[N, n, T] arrays need a complete, gap-free rollout grid"
        )

    flat = [by_row[r][k] for r in sel for k in range(n)]
    texts = [x["text"] for x in flat]
    fdirs = torch.stack([dirs[x["row"]] for x in flat])
    fdirs_c = None if dirs_centred is None else torch.stack([dirs_centred[x["row"]] for x in flat])
    gate = float(sae.threshold) if sae is not None else 0.0
    extra = _Extra(sae, d, len(flat), gate)
    print(f"[score] {len(sel)} targets x {n} rollouts = {len(flat)} rows on the clean base", flush=True)
    t0 = time.time()
    res = _score_all(
        model, tok, texts, fdirs, read_layer, extra, max_length=max_length,
        dirs_centred=fdirs_c, mu=mu,
    )
    best, _ = C.agg(res["cos"], res["keep"])
    # The centred max is taken over the SAME kept tokens but at its OWN argmax, not at the
    # uncentred one: a max read at another statistic's argmax is not a max (that is the flaw
    # precompute/centred.py documents at its :36-38 and this replaces).
    best_c, arg_c = (None, None)
    best_a, arg_a = (None, None)
    if fdirs_c is not None:
        keep_c = res["keep"] & ~torch.isnan(res["cos_centred"])
        best_c, arg_c = C.agg(torch.nan_to_num(res["cos_centred"], nan=-1.0), keep_c)
        empty_c = ~keep_c.any(dim=1)
        best_c = torch.where(empty_c, torch.full_like(best_c, float("nan")), best_c)
        arg_c = torch.where(empty_c, torch.full_like(arg_c, 0), arg_c) - 1
        # the asymmetric cosine takes its OWN argmax over the SAME kept tokens, for the reason
        # `cos_centred` does: a max read at another statistic's argmax is not a max.
        keep_a = res["keep"] & ~torch.isnan(res["cos_asym"])
        best_a, arg_a = C.agg(torch.nan_to_num(res["cos_asym"], nan=-1.0), keep_a)
        empty_a = ~keep_a.any(dim=1)
        best_a = torch.where(empty_a, torch.full_like(best_a, float("nan")), best_a)
        arg_a = torch.where(empty_a, torch.full_like(arg_a, 0), arg_a) - 1
    elapsed = time.time() - t0

    N = len(sel)
    width = max_length + 1  # the sink at column 0; C.SCORE_WIDTH at the protocol's truncation
    cos = res["cos"].numpy().astype(np.float16).reshape(N, n, width)
    nrm = res["norm"].numpy().astype(np.float16).reshape(N, n, width)
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
    cos_c = None if best_c is None else res["cos_centred"].numpy().astype(np.float16).reshape(N, n, width)
    bestm_c = None if best_c is None else best_c.numpy().reshape(N, n)
    argmax_c = None if arg_c is None else arg_c.numpy().astype(np.int16).reshape(N, n)
    cos_a = None if best_a is None else res["cos_asym"].numpy().astype(np.float16).reshape(N, n, width)
    bestm_a = None if best_a is None else best_a.numpy().reshape(N, n)
    argmax_a = None if arg_a is None else arg_a.numpy().astype(np.int16).reshape(N, n)

    per_target = []
    for i, r in enumerate(sel):
        vals = bestm[i].tolist()
        lens = [by_row[r][k]["n_tok"] for k in range(n)]
        fin = [by_row[r][k]["finished"] for k in range(n)]
        bo = C.best_of_k_means(vals, BO_KS)
        # `cos_centred` is NaN for a non-centrable family and for a rollout with no kept token, so
        # the centred aggregates are computed over the finite entries only and are absent -- not
        # zero, not -1 -- for a row that has none.
        vals_c = [] if bestm_c is None else [v for v in bestm_c[i].tolist() if np.isfinite(v)]
        bo_c = C.best_of_k_means(vals_c, BO_KS) if len(vals_c) == n else {}
        vals_a = [] if bestm_a is None else [v for v in bestm_a[i].tolist() if np.isfinite(v)]
        bo_a = C.best_of_k_means(vals_a, BO_KS) if len(vals_a) == n else {}
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
                **(
                    {}
                    if not vals_c
                    else {
                        "mean_cos_centred": round(float(np.mean(vals_c)), 6),
                        "max_cos_centred": round(float(np.max(vals_c)), 6),
                        "n_centred": len(vals_c),
                        **{f"bo_c_{k}": round(v, 6) for k, v in bo_c.items()},
                    }
                ),
                # the ASYMMETRIC cosine: uncentred scorer against the centred target, which is
                # what `scan` computes for a corpus window. The only one of the three that can be
                # differenced against a corpus search (Tomáš 2026-09-21).
                **(
                    {}
                    if not vals_a
                    else {
                        "mean_cos_asym": round(float(np.mean(vals_a)), 6),
                        "max_cos_asym": round(float(np.max(vals_a)), 6),
                        "n_asym": len(vals_a),
                        **{f"bo_a_{k}": round(v, 6) for k, v in bo_a.items()},
                    }
                ),
            }
        )

    # The scores directory follows the rollouts file it scored: a --run-tag run must not land on
    # top of the untagged one, for the same reason its rollouts do not.
    out = (
        f"{rdir}/scores"
        if rdir
        else C.scores_dir(maemm, args.get("score_name") or set_name, root, engine,
                          C.score_tag_of(args), write=True)
    )
    inputs = {
        "rollouts": rpath if len(sources) == 1 else ", ".join(sources),
        "engine": engine,
        "dirs": dirs_src,
        "maemm": maemm or f"(none: --rollouts-dir {rdir})",
        "weight sha256": rsum["weight_sha256"],
        "sae": sae_key or "(none: --no-sae)",
        "targets": f"{N} of {len(rows_meta)} rows",
        "n": n,
    }
    with C.outdir(out, args, inputs=inputs) as od:
        C.note_convention(od, cen_notes)
        bad = _check_scored_is_generation(
            res,
            texts,
            tok,
            int(rsum["max_new"]),
            od,
            gen_n_tok=[x["n_tok"] for x in flat],
            prompt_tokens=rsum.get("prompt_tokens"),
            max_length=max_length,
        )
        od.write_array("cos.f16", cos, "float16")
        od.write_array("norm.f16", nrm, "float16")
        od.write_array("argmax.i16", argmax, "int16")
        if cos_c is not None:
            od.write_array("cos_centred.f16", cos_c, "float16")
            od.write_array("argmax_centred.i16", argmax_c, "int16")
        if cos_a is not None:
            od.write_array("cos_asym.f16", cos_a, "float16")
            od.write_array("argmax_asym.i16", argmax_a, "int16")
        od.write_array("best_act.f16", best_act, "float16")
        od.write_array("sae_idx.i32", sae_idx, "int32")
        od.write_array("sae_val.f16", sae_val, "float16")
        od.write_array("sae_off.i64", sae_off, "int64")
        od.write_jsonl("per_target.jsonl", per_target)
        od.write_json(
            "rows.json",
            {
                "rows": sel,
                "n": n,
                "families": [rows_meta[r]["family"] for r in sel],
                # The stored arrays are [N, n, score_max_length + 1]. Written on every run, so a
                # reader takes the width from the DIRECTORY (common.score_width_of) instead of
                # from a module constant that a later arm may not share.
                "score_max_length": max_length,
                # Which mean the centred cosine used, as the path it was read from, or null when
                # this directory has no cos_centred.f16 at all. A reader takes the convention from
                # HERE rather than from a config entry that may have moved since.
                "mu": None if mu is None else C.mu_label(
                    C.mu_for(cfg, base, dirs_src, args, maemm, root)[0], base, root
                ),
            },
        )
        od.note(
            f"scored on the CLEAN BASE ({cfg['bases'][base]['hf']}); the MAEMM is never loaded by "
            "this product. Protocol: common.score_tokens -- padding_side right, "
            f"add_special_tokens=False, truncation at {max_length}, sink at column 0 "
            f"excluded from `keep`, fp32 cosine, ONE fixed chunk of {C.SCORE_CHUNK} rows, NO norm "
            f"filter (the per-token norm is stored instead). T = {width}, NaN outside `keep`."
            + (
                ""
                if max_length == C.SCORE_MAX_LENGTH
                else f" NOTE this run scores at {max_length} tokens, NOT the protocol's "
                f"{C.SCORE_MAX_LENGTH}: the producing run's summary asked for it (it generates at "
                f"max_new={rsum['max_new']}). `rows.json` records score_max_length so a reader "
                f"takes T from here, and a cosine from this directory is a max over a WIDER "
                f"window than every other arm's."
            )
        )
        if cos_c is not None:
            od.note(
                "`cos_centred.f16` [N, n, T] is the SECOND cosine from the SAME forward: "
                "cos(unit(h - mu), unit(act - mu)), both sides centred on the mean this run named, "
                "against `cos.f16`'s uncentred scorer. `argmax_centred.i16` is its OWN argmax among "
                "the scored tokens -- not the uncentred one -- because a max read at another "
                "statistic's argmax is not a max. Rows whose family is not `centrable` "
                "(config.yaml `family_kinds:`) are NaN in both, never a one-sided number, and so "
                "are rows with no kept token."
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
        # `bo_k` exists only for k in BO_KS, and n need not be one of them: a product whose grid
        # width is 3 -- which is what `gcg --mode epo`'s pop gives when its finals are repackaged
        # through --rollouts-dir -- has no `bo_3`, and this line raised KeyError AFTER the product
        # was already committed, so the run reported failure on a complete directory. Report the
        # largest recorded best-of at or below n instead, and name which one it is.
        k_bo = max((k for k in BO_KS if k <= n), default=0)
        if k_bo:
            fam_mean[f"{fam}_bo{k_bo}"] = round(
                float(np.mean([r[f"bo_{k_bo}"] for r in sub])), 5
            )
    return {"out": out, "engine": engine, "targets": N, "n": n, "rows": len(flat), "family_means": fam_mean}
