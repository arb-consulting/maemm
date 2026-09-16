"""Product `targets`: draw one held-out set -> `<root>/base/<base>/heldout/<set>/`.

    ids.jsonl     one row per target: row, family, id, stratum + the family's own fields
    vecs.f16      [N, d] UNIT rows, row i is ids.jsonl line i
    mu_512.f32    [d] DIAGNOSTIC ONLY (see below): the mean of the 512-token no-sink windows
    leakage.jsonl cos > 0.999 hits against the archived 8B training banks (checklist item 35)

Families are keyed BY NAME (checklist item 38) and drawn in the FIXED order realact, random, sae
from ONE `np.random.default_rng(heldout.seed)` stream, with `choice` then `sorted` wherever a pool
is sampled, and a SEPARATE `torch.Generator().manual_seed(seed)` for the random control -- all of
that is Celeste's convention in eval/eval_universal.py:409-453, copied so the two pipelines' draws
are structurally comparable. Reordering the families changes every family after the first.

ONE CENTRING MEAN (Tomáš, 2026-09-15; supersedes the earlier two-means rule, checklist item 77).
A realact target is `unit(X[p] - mu)` with `mu` = `stats/mu.f32`, the 64/16-window read-layer mean
of pass A, and that subtraction happens exactly ONCE, here at construction. Nothing else in the
pipeline centres anything: every cosine against a target is uncentred and goes through the single
`common.score_tokens`, and the `sae` and `random` directions are never centred at all.

`mu_512.f32` -- the read-layer mean over ALL positions of the 512-token, NO-sink windows this draw
forwards, which is what Celeste subtracts (data/build_universal_bank.py:310) -- is still computed
and stored, but ONLY as a diagnostic: the README reports its cosine and norm ratio against
`stats/mu.f32` so the size of the convention difference is on the record. `targets` FAILS if
`stats` has not run: there is no local fallback mean.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import precompute.common as C

FAMILY_ORDER = ("realact", "random", "sae")  # fixed; see the module docstring
REALACT_WINDOW = 512  # data/build_universal_bank.py: positions live in 512-token windows
REALACT_P_MIN = 16  # p ~ U[16, 512): checklist item 30
SPAN_MIN, SPAN_MAX = 16, 64  # L ~ U[16, 64] inclusive (build_universal_bank.py:74)
NORM_FILTER_MULT = 10.0  # raw-norm filter AT SELECTION ONLY (checklist item 34)
NORM_PRESAMPLE = 4096
REALACT_DOCS_PER_FWD = 16  # 16 x 512 = 8192 token-forwards per call
LEAK_COS = 0.999
LEAK_CHUNK = 65536


def _forward_512(model, read_layer, docs512, toks, device):
    """Read-layer activations [n, 512, d] (fp32, cpu) of the first 512 tokens of each document.

    No sink token: Celeste's realact windows are `add_special_tokens=False` 512-token windows with
    nothing prepended (data/collect_acts.py:52-60), which is why their mean differs from stats/mu.
    """
    import torch

    out = []
    for s in range(0, len(docs512), REALACT_DOCS_PER_FWD):
        chunk = docs512[s : s + REALACT_DOCS_PER_FWD]
        ids = np.stack([toks[r["offset"] : r["offset"] + REALACT_WINDOW] for r in chunk]).astype(np.int64)
        t = torch.from_numpy(ids).to(device)
        h, _ = C.read_resid(
            model, read_layer, {"input_ids": t, "attention_mask": torch.ones_like(t)}, pool="all"
        )
        out.append(h.cpu())
    return torch.cat(out)


def _realact(cfg, args, model, tok, toks, docs, n, rng, od):
    """Celeste's recipe (data/build_universal_bank.py:295-314, checklist items 30/34)."""
    import torch

    base, root = args["base"], args["root"]
    read_layer, d = cfg["bases"][base]["read_layer"], cfg["bases"][base]["d"]
    # THE centring mean of the pipeline; C.stats_mu fails loudly if `stats` has not run.
    mu = torch.from_numpy(C.stats_mu(cfg, base, root)).float()
    eligible = [r for r in docs if r["len"] >= REALACT_WINDOW]
    pool_n = min(len(eligible), max(2 * n, n + 256))
    assert pool_n >= n, (
        f"only {len(eligible)} corpus documents have >= {REALACT_WINDOW} tokens, need at least {n}"
    )
    sel = np.sort(rng.choice(len(eligible), pool_n, replace=False))  # choice then sorted
    pool = [eligible[int(i)] for i in sel]
    p_all = rng.integers(REALACT_P_MIN, REALACT_WINDOW, size=pool_n)
    l_all = rng.integers(SPAN_MIN, SPAN_MAX + 1, size=pool_n)

    # (1) the norm-filter threshold, from a presample of NORM_PRESAMPLE positions of the SAME windows
    t0 = time.time()
    n_pre = min(64, pool_n)
    pre = _forward_512(model, read_layer, pool[:n_pre], toks, "cuda")
    ii = rng.integers(0, n_pre, NORM_PRESAMPLE)
    pp = rng.integers(REALACT_P_MIN, REALACT_WINDOW, NORM_PRESAMPLE)
    med = float(pre[torch.from_numpy(ii), torch.from_numpy(pp)].norm(dim=-1).median())
    del pre
    print(f"[realact] presample median raw norm {med:.1f} (keep <= {NORM_FILTER_MULT}x)", flush=True)

    # (2) the whole pool, so mu_512 does not depend on where the acceptance loop stopped
    xs, mu_sum, mu_n = [], torch.zeros(d, dtype=torch.float64), 0
    for s in range(0, pool_n, 256):
        h = _forward_512(model, read_layer, pool[s : s + 256], toks, "cuda")
        mu_sum += h.reshape(-1, d).sum(0, dtype=torch.float64)
        mu_n += h.shape[0] * h.shape[1]
        xs.append(h[torch.arange(h.shape[0]), torch.from_numpy(p_all[s : s + h.shape[0]])].clone())
        print(f"[realact] forwarded {min(s + 256, pool_n)}/{pool_n} windows", flush=True)
    x = torch.cat(xs)
    mu_512 = (mu_sum / mu_n).float()
    # DIAGNOSTIC: how far Celeste's 512-window mean is from the one we actually centre on.
    mu_cos = float(torch.nn.functional.cosine_similarity(mu_512, mu, dim=0))
    mu_ratio = float(mu_512.norm() / mu.norm().clamp(min=1e-9))
    print(
        f"[realact] centring on stats/mu.f32 (||mu||={mu.norm():.1f}); the diagnostic mu_512 has "
        f"cos={mu_cos:.4f} and ||mu_512||/||mu||={mu_ratio:.4f}",
        flush=True,
    )
    nrm = x.norm(dim=-1)
    ok = (nrm > 1e-3) & (nrm <= NORM_FILTER_MULT * med)
    print(
        f"[realact] {int(ok.sum())}/{pool_n} candidates pass the raw-norm filter "
        f"({time.time() - t0:.0f}s, mu_512 over {mu_n} positions, ||mu_512||={mu_512.norm():.1f})",
        flush=True,
    )

    rows, vecs = [], []
    for i in np.flatnonzero(ok.numpy()):  # pool order; the pool is already a sorted random draw
        if len(rows) >= n:
            break
        r = pool[int(i)]
        p, span = int(p_all[i]), int(l_all[i])
        ids = toks[r["offset"] + p - span + 1 : r["offset"] + p + 1]
        rows.append(
            {
                "family": "realact",
                "id": f"doc{r['doc']}:p{p}:L{span}",
                "stratum": None,
                "doc": r["doc"],
                "part": r["part"],
                "part_row": r["row"],  # `row` is the TARGET row, so the parquet row is renamed
                "p": p,
                "L": span,
                "act_norm": round(float(nrm[int(i)]), 3),
                "span_text": tok.decode([int(t) for t in ids]),
            }
        )
        vecs.append(torch.nn.functional.normalize(x[int(i)] - mu, dim=-1))
    assert len(rows) == n, f"realact: only {len(rows)} of {n} targets survived the norm filter"
    od.write_array("mu_512.f32", mu_512, "float32")
    od.section(
        "Methods",
        [
            "The `realact` family follows Celeste's recipe verbatim "
            "(`data/build_universal_bank.py:72-75, 295-314` and `data/collect_acts.py:52-60`; "
            "checklist item 30):",
            "",
            f"- the activation is read in the FULL context of the document's first "
            f"{REALACT_WINDOW} tokens, tokenized with `add_special_tokens=False` and with NO sink "
            f"token prepended;",
            f"- the position is `p ~ U[{REALACT_P_MIN}, T)` with T = {REALACT_WINDOW}, one target "
            f"per document, from a {pool_n}-document pool of documents with >= {REALACT_WINDOW} "
            f"tokens;",
            f"- the SHOWN span is `L ~ U[{SPAN_MIN}, {SPAN_MAX}]` tokens long and ends at p: "
            f"`span_text = decode(toks[p-L+1 : p+1])`;",
            "- the target vector is `unit(X[p] - mu)` with `mu` = `stats/mu.f32`, the 64/16-window "
            "read-layer mean of pass A. That is the ONE centring in the whole pipeline and it "
            "happens once, here (Tomáš, 2026-09-15); every cosine against this target afterwards "
            "is UNCENTRED and goes through `common.score_tokens`;",
            f"- a raw-norm filter (`> 1e-3` and `<= {NORM_FILTER_MULT}x` the presample median "
            f"{med:.1f}) is applied AT SELECTION ONLY (checklist item 34), never at scoring.",
        ],
    )
    od.note(
        f"CENTRING: the realact vectors are `unit(X[p] - stats/mu.f32)` "
        f"({C.stats_dir(base, root)}/mu.f32, ||mu||={mu.norm():.2f}). One rule, one subtraction, "
        "at construction only."
    )
    od.note(
        f"mu_512.f32 [d] is a DIAGNOSTIC: the mean read-layer activation over all {mu_n} positions "
        f"of the {pool_n} forwarded 512-token windows (NO sink token), i.e. Celeste's convention "
        f"(data/build_universal_bank.py:310). Nothing is centred on it. Against stats/mu.f32 it has "
        f"cos = {mu_cos:.4f} and ||mu_512|| / ||mu|| = {mu_ratio:.4f}."
    )
    return rows, vecs


def _random(cfg, args, n, seed):
    """eval/eval_universal.py:434-435 verbatim: a SEPARATE torch generator, not the numpy stream."""
    import torch

    d = cfg["bases"][args["base"]]["d"]
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    rows = [{"family": "random", "id": i, "stratum": None} for i in range(n)]
    return rows, [dirs[i] for i in range(n)]


def _training_features(cfg, base):
    """Feature ids to EXCLUDE for `base` (checklist item 48), read from the archived bank splits.

    AMBIGUITY, resolved and reported: run{1,2}/bank/split.json has no `train` feature list -- its
    only populated feature key is `families/sae_heldout_features` (6,553 ids, both files byte
    identical), i.e. the HELD-OUT 10% of the 65,536-wide 8B SAE. build_stats.json confirms the
    complement is what was minted into pool_train (alive_on_scan 65,536, families.sae 58,983 =
    65,536 - 6,553). So "training features" is taken to be the COMPLEMENT of that list, which is a
    superset of run2's actual training ids (its sentence filter dropped some) -- conservative in
    the safe direction. 27B: no split exists (ask-Celeste item), so nothing is excluded there.
    """
    if base != "qwen3-8b":
        return set(), []
    out, srcs = None, []
    for run in ("run1", "run2"):
        path = f"{cfg['modal']['archive']}/data/{run}/bank/split.json"
        assert os.path.exists(path), f"missing archived split {path} (checklist item 48 needs it)"
        with open(path) as fh:
            spl = json.load(fh)
        held = spl.get("families", {}).get("sae_heldout_features")
        assert held, f"{path}: families.sae_heldout_features is missing or empty"
        srcs.append(f"{run}: {len(held)} held-out feature ids from {path}")
        out = set(held) if out is None else (out & set(held))
    return out, srcs  # `out` is the intersection of the held-out sets = what stays ELIGIBLE


def _sae(cfg, args, sae, n, strata, min_fires, rng, od):
    """Density-stratified feature draw from the pass-A fire counts at the LARGEST corpus size."""
    base, root = args["base"], args["root"]
    sae_key = args["sae_key"]
    sdir = C.sae_dir(sae_key, root)
    with open(f"{sdir}/sizes.json") as fh:
        meta = json.load(fh)
    f = int(meta["d_sae"])
    assert f == sae.d_sae, f"sizes.json says d_sae={f} but the loaded SAE has {sae.d_sae}"
    sizes, positions = meta["sizes"], meta["scanned_positions"]
    fc = C.read_array(f"{sdir}/fire_counts.i64", "int64", (f, len(sizes), 2))
    max_act = C.read_array(f"{sdir}/max_act.f16", "float16", (f,)).astype(np.float32)
    fires0, fires_g = fc[:, -1, 0], fc[:, -1, 1]
    scanned = int(positions[-1])

    eligible = (fires_g >= min_fires) & (max_act > 0)  # checklist items 44 / 46
    n_gate = int(eligible.sum())
    keep_ids, srcs = _training_features(cfg, base)
    if keep_ids:
        mask = np.zeros(f, dtype=bool)
        mask[np.fromiter(keep_ids, dtype=np.int64, count=len(keep_ids))] = True
        eligible &= mask
    ids = np.flatnonzero(eligible)
    assert ids.size >= strata, f"only {ids.size} eligible SAE features, need at least {strata}"
    density = fires_g[ids] / scanned
    logd = np.log10(density)
    cuts = np.quantile(logd, [0.25, 0.50, 0.75])
    stratum = np.searchsorted(cuts, logd, side="right")  # 0..3

    per = n // strata
    assert per * strata == n, f"sae n={n} is not divisible by sae_strata={strata}"
    rows, feats = [], []
    short = []
    for q in range(strata):
        pool = ids[stratum == q]
        take = min(per, pool.size)
        if take < per:
            short.append(f"q{q}: {pool.size} eligible < {per} requested")
        pick = np.sort(rng.choice(pool, take, replace=False))
        for fid in pick:
            rows.append(
                {
                    "family": "sae",
                    "id": int(fid),
                    "stratum": int(q),
                    "density": float(fires_g[fid] / scanned),
                    "fires_gt0": int(fires0[fid]),
                    "fires_gated": int(fires_g[fid]),
                    "max_act": float(max_act[fid]),
                }
            )
            feats.append(int(fid))
    if short:
        msg = "; ".join(short)
        assert args.get("allow_short"), (
            f"sae strata are short of targets ({msg}); pass --allow-short to accept a smaller "
            f"family (a smoke corpus has few features with >= {min_fires} gated fires)"
        )
        print(f"[sae] WARN short draw: {msg}", flush=True)
        od.status = "short"
        od.note(f"SHORT DRAW (--allow-short): {msg}")
    dirs = C.sae_dirs(sae, feats).cpu()
    od.note(
        f"sae: eligible = gated fires >= {min_fires} AND max_act > 0 at the largest corpus size "
        f"({sizes[-1]}M, {scanned} scanned positions): {n_gate} features pass that, {ids.size} "
        f"remain after the training-feature exclusion; stratified into {strata} quartiles of "
        f"log10(density) with log10-density cuts {[round(float(c), 3) for c in cuts]}, "
        f"{per} drawn per quartile; direction = unit(W_enc[:, f]) (the ENCODER column)"
    )
    for s in srcs:
        od.note(f"sae exclusion source -- {s}")
    od.note(
        "sae exclusion rule: a feature is EXCLUDED unless it is in the intersection of run1's and "
        "run2's `families.sae_heldout_features`; those files list the held-out features, not the "
        "training ones, so the training set is taken as the complement (see targets.py docstring)"
        if srcs
        else "sae exclusion: none applied (no archived training split exists for this base)"
    )
    return rows, [dirs[i] for i in range(len(feats))]


def _leakage(cfg, args, rows, vecs, od):
    """cos > 0.999 of every realact / sae direction against the archived 8B training banks."""
    import torch

    base, d = args["base"], cfg["bases"][args["base"]]["d"]
    if base != "qwen3-8b":
        od.note("leakage check: skipped (only the 8B training banks are in the archive)")
        return []
    idx = [i for i, r in enumerate(rows) if r["family"] in ("realact", "sae")]
    v = torch.nn.functional.normalize(torch.stack([vecs[i] for i in idx]).float(), dim=-1).cuda()
    hits = []
    for run in ("run1", "run2"):
        path = f"{cfg['modal']['archive']}/data/{run}/bank/pool_train/vecs.f32"
        assert os.path.exists(path), f"missing archived bank {path}"
        n_rows = os.path.getsize(path) // (4 * d)
        assert n_rows * 4 * d == os.path.getsize(path), (
            f"{path} is not a whole number of [.., {d}] f32 rows -- wrong d for this archive?"
        )
        t0 = time.time()
        for s in range(0, n_rows, LEAK_CHUNK):
            m = min(LEAK_CHUNK, n_rows - s)
            blk = np.fromfile(path, dtype=np.float32, count=m * d, offset=s * d * 4).reshape(m, d)
            b = torch.nn.functional.normalize(torch.from_numpy(blk).cuda(), dim=-1)
            cos = b @ v.T  # [m, n_targets]
            hi = (cos > LEAK_COS).nonzero()
            for a_row, t_col in hi.cpu().numpy():
                j = idx[int(t_col)]
                hits.append(
                    {
                        "row": rows[j]["row"],
                        "family": rows[j]["family"],
                        "archive": run,
                        "archive_row": int(s + a_row),
                        "cos": float(cos[a_row, t_col]),
                    }
                )
        print(f"[leakage] {run}: {n_rows} bank rows in {time.time() - t0:.0f}s, {len(hits)} hits", flush=True)
    od.note(
        f"leakage: cos > {LEAK_COS} of every realact and sae direction against "
        f"run1+run2 pool_train/vecs.f32 in the archive; {len(hits)} hits, reported only "
        "(checklist item 35 says report, never remove)"
    )
    return hits


IMPORT_FAMS = ("realact", "random", "sae")  # FAMILY_ORDER restricted to what run1's cache holds


def import_run1(cfg, args):
    """`--import-run1`: a held-out set built from run1's ARCHIVED eval cache, not drawn by us.

    `<archive>/data/run1/eval_cache/eval_sets_heldout.pt` is the exact object the archived run1
    numbers were produced from: `<fam>_dirs` [512, d] per family, `meta["rows"][fam]` the pool
    vec_idx behind each row, `sae_feats` the feature ids of the sae family. Rows 0..n-1 of realact,
    random and sae are copied out verbatim (re-unit-normalised after the fp16/fp32 round trip) so
    reconstruction/repro_run1.py can put our rollouts and our scorer against theirs on the SAME
    directions. `archive_row` IS the `row` field of the archived dumps, which is how the comparison
    joins. This set exists only for that reproduction check; nothing else should be drawn on it.
    """
    import torch

    base, root = args["base"], args["root"]
    n = int(args.get("n") or 16)
    assert base == "qwen3-8b", (
        f"--import-run1 reads run1's 8B eval cache; base {base!r} has no archived cache "
        f"(the 27B banks are an ask-Celeste item)"
    )
    d = cfg["bases"][base]["d"]
    path = f"{cfg['modal']['archive']}/data/run1/eval_cache/eval_sets_heldout.pt"
    assert os.path.exists(path), f"missing archived eval cache {path}"
    es = torch.load(path, weights_only=False, map_location="cpu")
    print(f"[import-run1] {path}: keys {sorted(es)}", flush=True)
    meta = es.get("meta", {})
    print(f"[import-run1] meta keys {sorted(meta) if hasattr(meta, 'keys') else type(meta)}", flush=True)
    rows_map = meta.get("rows", {}) if hasattr(meta, "keys") else {}

    rows, vecs = [], []
    for fam in IMPORT_FAMS:
        key = f"{fam}_dirs"
        assert key in es, f"{path} has no {key!r} (keys: {sorted(es)})"
        mat = torch.as_tensor(es[key]).float()
        assert mat.dim() == 2 and mat.shape[1] == d, (
            f"{path}[{key}] is {tuple(mat.shape)}, expected [n_dirs, {d}] for base {base}"
        )
        assert mat.shape[0] >= n, f"{path}[{key}] has only {mat.shape[0]} rows, need {n}"
        pool = rows_map.get(fam)
        if pool is None:
            print(f"[import-run1] WARN: meta['rows'] has no {fam!r}; dir_index will be null", flush=True)
        feats = es.get("sae_feats") if fam == "sae" else None
        for r in range(n):
            vec = torch.nn.functional.normalize(mat[r], dim=-1)
            dir_index = None if pool is None else int(pool[r])
            ident = int(feats[r]) if feats is not None else dir_index if dir_index is not None else r
            rows.append(
                {
                    "family": fam,
                    "id": ident,
                    "stratum": None,
                    "archive_row": r,
                    "dir_index": dir_index,
                    "archive_family": fam,
                }
            )
            vecs.append(vec)
        print(f"[import-run1] {fam}: rows 0..{n - 1} of {mat.shape[0]}", flush=True)

    for i, row in enumerate(rows):
        row["row"] = i
    v = torch.stack(vecs)
    nrm = v.norm(dim=-1)
    assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-5), (
        f"imported vectors must be unit rows; got min {nrm.min():.6f} max {nrm.max():.6f}"
    )
    set_name = args["heldout"]
    out = C.heldout_dir(base, set_name, root)
    inputs = {"archive": path, "families": {f: n for f in IMPORT_FAMS}, "rows_per_family": n}
    with C.outdir(out, args, inputs=inputs) as od:
        ordered = [{"row": r["row"], **{k: x for k, x in r.items() if k != "row"}} for r in rows]
        od.write_jsonl("ids.jsonl", ordered)
        od.write_array("vecs.f16", v, "float16")
        od.section(
            "Methods",
            [
                "This set is NOT a draw of ours. It is rows "
                f"0..{n - 1} of each of {list(IMPORT_FAMS)} copied verbatim out of run1's archived "
                f"eval cache `{path}` (`torch.load(weights_only=False)`), re-unit-normalised in "
                "fp32 before the f16 store.",
                "",
                "- `archive_row` is the `row` field of the archived per-rollout dumps "
                "(`ss_samples.jsonl`) and of `per_dir_final*.json`'s list position -- that is the "
                "join key for the reproduction check.",
                "- `dir_index` is `meta['rows'][family][archive_row]`, the pool vec_idx the "
                "direction came from; `id` is the SAE feature id for the sae family and the "
                "dir_index otherwise.",
                "- It exists ONLY for reconstruction/repro_run1.py. Do not draw conclusions about "
                "the 2026-09 pipeline from a 16-row slice of a 2026-09-03 eval cache.",
            ],
        )
        od.note("no mu_512.f32 and no leakage.jsonl here: neither is defined for an imported set")
        od.note(f"source families present in the cache: {sorted(k[:-5] for k in es if k.endswith('_dirs'))}")
    return {"out": out, "rows": len(rows), "families": {f: n for f in IMPORT_FAMS}, "source": path}


def run(cfg, args):
    import torch

    if args.get("import_run1"):
        return import_run1(cfg, args)

    base, root = args["base"], args["root"]
    set_name = args["heldout"]
    assert base, "product targets needs --base"
    spec = cfg["bases"][base]
    hspec = cfg["heldout"][set_name]
    fams = C.families_for(cfg, set_name, base)
    seed = int(hspec["seed"])
    sae_keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    assert len(sae_keys) == 1, f"base {base} has {len(sae_keys)} SAEs in config, expected exactly 1"
    args["sae_key"] = sae_keys[0]

    toks, docs = C.load_corpus(base, root)
    sizes_json = f"{C.sae_dir(sae_keys[0], root)}/sizes.json"
    assert os.path.exists(sizes_json), (
        f"{sizes_json} is missing: run `--product stats --base {base}` first, the sae family is "
        f"drawn from its fire counts"
    )
    C.stats_mu(cfg, base, root)  # fail before the model load if the ONE centring mean is missing
    rng = np.random.default_rng(seed)
    out = C.heldout_dir(base, set_name, root)
    inputs = {
        "corpus": C.corpus_dir(base, root),
        "sae": C.sae_dir(sae_keys[0], root),
        "mu (centring)": f"{C.stats_dir(base, root)}/mu.f32",
        "seed": seed,
        "families": {f: s["n"] for f, s in fams.items()},
    }
    with C.outdir(out, args, inputs=inputs) as od:
        model, tok = C.load_base(cfg, base)
        sae = C.load_sae(C.sae_path(cfg, sae_keys[0]), spec["d"], device="cuda", dtype=torch.float32)
        rows, vecs = [], []
        for fam in FAMILY_ORDER:  # FIXED order: it determines the rng stream
            if fam not in fams:
                continue
            n = int(fams[fam]["n"])
            if fam == "realact":
                r, v = _realact(cfg, args, model, tok, toks, docs, n, rng, od)
            elif fam == "random":
                r, v = _random(cfg, args, n, seed)
            else:
                r, v = _sae(cfg, args, sae, n, int(hspec["sae_strata"]), int(hspec["sae_min_fires"]), rng, od)
            rows += r
            vecs += v
            print(f"[targets] {fam}: {len(r)} rows", flush=True)
        for i, row in enumerate(rows):
            row["row"] = i
        empty = [f for f, s in fams.items() if f not in FAMILY_ORDER]
        for f in empty:
            od.note(
                f"family {f!r}: declared in config with status={fams[f].get('status')!r} and drawn "
                f"with ZERO rows -- no J-lens matrix exists for either base (checklist item 40, "
                f"layout §5a item 4), so the slot only fixes the set's shape"
            )

        v = torch.stack(vecs).float()
        nrm = v.norm(dim=-1)
        assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-5), (
            f"held-out vectors must be unit rows; got min {nrm.min():.6f} max {nrm.max():.6f}"
        )
        hits = _leakage(cfg, args, rows, vecs, od)
        ordered = [{"row": r["row"], **{k: x for k, x in r.items() if k != "row"}} for r in rows]
        od.write_jsonl("ids.jsonl", ordered)
        od.write_array("vecs.f16", v, "float16")
        od.write_jsonl("leakage.jsonl", hits)
        od.note(
            f"rebuild: `modal run precompute/modal_app.py --product targets --base {base} "
            f"--set {set_name} --root {root}` at repo commit {args.get('repo_commit', '?')[:12]}"
        )
        od.note(
            f"draw order {FAMILY_ORDER} from ONE np.random.default_rng({seed}); the random family "
            f"uses a separate torch.Generator().manual_seed({seed}) (Celeste's convention)"
        )
        od.note("vecs.f16 rows are unit in fp32 before the cast; the f16 round-trip is ~1e-3 off unit")
    return {
        "out": out,
        "rows": len(rows),
        "families": {f: sum(1 for r in rows if r["family"] == f) for f in FAMILY_ORDER},
        "leakage_hits": len(hits),
    }
