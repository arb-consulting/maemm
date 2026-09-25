"""Product `targets`: draw one held-out set -> `<root>/base/<base>/heldout/<set>/`.

    ids.jsonl     one row per target: row, family, id, stratum + the family's own fields
    act.f32       [N, d] the RAW vector of each row, before any mean was subtracted
    vecs.f16      [N, d] UNIT rows = unit(act), UNCENTRED; row i is ids.jsonl line i
    storage.json  the set's storage contract (`storage:` alone), common.set_storage
    mu_512.f32    [d] DIAGNOSTIC ONLY (see below): the mean of the 512-token no-sink windows
    leakage.jsonl cos > 0.999 hits against the archived 8B training banks (checklist item 35)

Families are keyed BY NAME (checklist item 38) and drawn in the FIXED order realact, random, sae
from ONE `np.random.default_rng(heldout.seed)` stream, with `choice` then `sorted` wherever a pool
is sampled, and a SEPARATE `torch.Generator().manual_seed(seed)` for the random control -- all of
that is the convention in evals/heldout/eval_universal.py:409-453, copied so the two pipelines' draws
are structurally comparable. Reordering the families changes every family after the first.

RAW STORAGE (2026-09-21; supersedes the 2026-09-15 one-centring-mean rule below).
**A stored artefact never encodes a centring choice.** A realact row is written as its raw
read-layer activation `X[p]` in `act.f32`, with `vecs.f16 = unit(X[p])` -- UNCENTRED -- and the
mean is subtracted at READ time under a name the run states: `common.dirs_for(..., centering=...)`,
driven by `maems.<ckpt>.input.centering` or an explicit `--centering`. The set is therefore the
same file for every checkpoint and every convention, and "which mu was this drawn under" stops
being a question anyone can get wrong.

For a family that is not `centrable` (config.yaml `family_kinds:` -- an encoder column, a Gaussian
draw, a subspace basis) there is no mean to subtract, so its `act.f32` row IS its unit direction
and `unit(act) == vecs.f16` there. That keeps the array rectangular and makes `centering: none` a
no-op on those rows rather than a special case at seven call sites.

SUPERSEDED (kept for the record, because every set drawn before 2026-09-21 follows it): "ONE
CENTRING MEAN (2026-09-15, checklist item 77) -- a realact target is `unit(X[p] - mu)` with
`mu` = `stats/mu.f32` and that subtraction happens exactly ONCE, here at construction." Those sets
are `storage: unit` in config.yaml and `common.dirs_for` serves them only at the mean they were
built with, and has no centred reading at all (common.dirs_for).

`mu_512.f32` -- the read-layer mean over ALL positions of the 512-token, NO-sink windows this draw
forwards, which is what the original pipeline subtracts (data/build_universal_bank.py:310) -- is still computed
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

    No sink token: the realact windows are `add_special_tokens=False` 512-token windows with
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
    """the recipe (data/build_universal_bank.py:295-314, checklist items 30/34).

    Returns (rows, unit dirs, RAW activations). `mu` is still loaded and still reported -- it is
    the diagnostic anchor of the README -- but nothing here subtracts it any more: see the module
    docstring's RAW STORAGE note.
    """
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
    # DIAGNOSTIC: how far the 512-window mean is from the one we actually centre on.
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

    rows, vecs, acts = [], [], []
    n_clamped = 0
    for i in np.flatnonzero(ok.numpy()):  # pool order; the pool is already a sorted random draw
        if len(rows) >= n:
            break
        r = pool[int(i)]
        p, span = int(p_all[i]), int(l_all[i])
        # CLAMPED at the document start (D4, fixed 2026-09-21). `p >= REALACT_P_MIN = 16` and
        # `span <= SPAN_MAX = 64`, so `p - span + 1` is negative on a short-p / long-L draw and the
        # unclamped slice reached up to 45 tokens back into the PREVIOUS document of the flat token
        # array -- 21 of 512 rows on 2026-09-16_v1. The activation was never affected (it is
        # `x[int(i)]`, read from this document's own window at position p); only the shown and
        # scored `span_text` was, which is what autointerp and the GCG corpus init consume.
        lo = max(r["offset"], r["offset"] + p - span + 1)
        ids = toks[lo : r["offset"] + p + 1]
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
                # The shown span, CLAMPED at the document start: `L_shown` is what the reader
                # actually got, which is < L whenever the draw asked for more tokens than this
                # document has before p.
                "L_shown": int(r["offset"] + p + 1 - lo),
                "span_text": tok.decode([int(t) for t in ids]),
            }
        )
        n_clamped += int(r["offset"] + p - span + 1 < r["offset"])
        # RAW: act.f32 keeps X[p] as read, vecs.f16 is unit(X[p]). Nothing is centred here.
        acts.append(x[int(i)].clone())
        vecs.append(torch.nn.functional.normalize(x[int(i)], dim=-1))
    assert len(rows) == n, f"realact: only {len(rows)} of {n} targets survived the norm filter"
    od.write_array("mu_512.f32", mu_512, "float32")
    print(f"[realact] {n_clamped}/{len(rows)} spans clamped at the document start (D4)", flush=True)
    od.section(
        "Methods",
        [
            "The `realact` family follows the recipe verbatim "
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
            "- the STORED vector is the RAW activation `X[p]` (`act.f32`) with "
            "`vecs.f16 = unit(X[p])`, UNCENTRED (2026-09-21). No mean is subtracted at "
            "construction any more: a run names the mean it wants and `common.dirs_for` derives "
            "`unit(X[p] - mu)` at read time, so this set is the same file under every convention;",
            f"- a raw-norm filter (`> 1e-3` and `<= {NORM_FILTER_MULT}x` the presample median "
            f"{med:.1f}) is applied AT SELECTION ONLY (checklist item 34), never at scoring.",
        ],
    )
    od.note(
        f"CENTRING: NONE IS STORED. `act.f32` is the raw `X[p]` and `vecs.f16` is `unit(X[p])`. "
        f"The mean is named per run (`--centering`, or the checkpoint's `input.centering`) and "
        f"applied by `common.dirs_for`; `stats/mu.f32` ({C.stats_dir(base, root)}/mu.f32, "
        f"||mu||={mu.norm():.2f}) is one of the names in config.yaml's `mus:` block, not the rule."
    )
    od.note(
        f"span_text is CLAMPED at the document start (D4, 2026-09-21): {n_clamped} of {len(rows)} "
        f"rows had `p - L + 1 < 0` and the unclamped slice would have reached into the PREVIOUS "
        f"document. `L_shown` is the clamped length; `L` is the length the draw asked for. The "
        f"activation is unaffected either way -- it is read at position p of this document's own "
        f"512-token window."
    )
    od.note(
        f"mu_512.f32 [d] is a DIAGNOSTIC: the mean read-layer activation over all {mu_n} positions "
        f"of the {pool_n} forwarded 512-token windows (NO sink token), i.e. the convention "
        f"(data/build_universal_bank.py:310). Nothing is centred on it. Against stats/mu.f32 it has "
        f"cos = {mu_cos:.4f} and ||mu_512|| / ||mu|| = {mu_ratio:.4f}."
    )
    return rows, vecs, acts


def _random(cfg, args, n, seed):
    """evals/heldout/eval_universal.py:434-435 verbatim: a SEPARATE torch generator, not the numpy stream."""
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
    the safe direction. 27B: no split exists (open item), so nothing is excluded there.
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
    """Density-stratified feature draw from the pass-A fire counts at the LARGEST corpus size.

    Every row carries `sae_key`, because a feature index means nothing without the dictionary it
    indexes: id 4242 of the 131k `l42-1b` and of the 2M `dict2m` are unrelated directions, and both
    are valid indices into the larger one. features/draw_dict2m.py set the precedent.
    """
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
                    "sae_key": sae_key,
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


def open_bank(path: str, d: int):
    """`(n_rows, read(start, m) -> [m, d] float32)` for ONE bank of stored directions.

    Two shapes exist in this project and both are read here, sequentially, with no mmap: a RAW
    `.f32` / `.f16` file laid out as [.., d] rows (the 8B archive's `pool_train/vecs.f32`) and a
    numpy `.npy` (the tier-B `simple2m/*/dirs_f16.npy`). A file whose size is not a whole
    number of [.., d] rows is a wrong `d` or a truncated fetch and stops the run rather than being
    scanned short -- a leak check that silently reads half a bank reports "no hits" for the half it
    never looked at.
    """
    import numpy as np

    assert os.path.exists(path), f"missing direction bank {path}"
    if path.endswith(".npy"):
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}
            assert version in reader, f"{path}: unsupported .npy version {version}"
            shape, fortran, dt = reader[version](fh)
            off = fh.tell()
        assert not fortran, f"{path}: Fortran-ordered .npy; the row reader below assumes C order"
        assert len(shape) == 2 and shape[1] == d, (
            f"{path} is {shape}, expected [.., {d}] rows -- wrong d for this bank?"
        )
        n_rows, isz = int(shape[0]), int(dt.itemsize)
        want = off + n_rows * d * isz
        assert os.path.getsize(path) == want, (
            f"{path}: header says {shape} {dt} ({want} B with a {off} B header) but the file is "
            f"{os.path.getsize(path)} B -- a TRUNCATED fetch, not a bank"
        )
    else:
        dt = {".f32": np.dtype("float32"), ".f16": np.dtype("float16")}.get(path[-4:])
        assert dt is not None, f"{path}: a raw bank must end .f32 or .f16 (or be a .npy)"
        off, isz, nbytes = 0, int(dt.itemsize), os.path.getsize(path)
        n_rows = nbytes // (isz * d)
        assert n_rows * isz * d == nbytes, (
            f"{path} is not a whole number of [.., {d}] {dt} rows -- wrong d for this bank?"
        )

    def read(start: int, m: int):
        blk = np.fromfile(path, dtype=dt, count=m * d, offset=off + start * d * isz)
        assert blk.size == m * d, f"{path}: short read of {blk.size} of {m * d} values at row {start}"
        return blk.reshape(m, d).astype(np.float32)

    return n_rows, read


def leak_scan(dirs, banks, d: int, *, thr: float = LEAK_COS, chunk: int = LEAK_CHUNK,
              device: str = "numpy", max_hits: int = 10_000, label: str = "leakage"):
    """Max cosine of every row of `dirs` [N, d] against every row of every bank in `banks`.

    `banks` is `[(name, path), ...]`; `dirs` is already the direction each target row carries (the
    caller decides whether that is `unit(act)` or `unit(act - mu)`) and is re-normalised here.
    Returns

        {"n": N, "banks": [{name, path, rows, seconds, max_cos, n_hits}, ...],
         "best": [N] float32,        the running max cosine of each target row
         "best_bank": [N] str, "best_row": [N] int,     where that max came from
         "per_bank": {name: [N] float32},   the same max restricted to one bank
         "hits": [{target, bank, bank_row, cos}, ...],  every pair above `thr`, capped
         "n_hits": total number of pairs above `thr`, including any past the cap}

    `per_bank` is what lets a caller cut the result BOTH ways -- per target block and per bank --
    without a second pass; it is [n_banks, N] floats and costs nothing beside the arrays scanned.

    ONE pass over the banks serves every target row the caller has: the tier-B arrays are 92 GB and
    the read dominates the matmul by an order of magnitude, so a caller with three blocks to check
    concatenates them into `dirs` and splits the result by row range afterwards. `device` is
    `numpy` (the CPU product) or `cuda` (inside a GPU product that already holds a device).
    """
    import numpy as np

    v = np.asarray(dirs, dtype=np.float32)
    assert v.ndim == 2 and v.shape[1] == d, f"{label}: dirs is {v.shape}, expected [.., {d}]"
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    n = v.shape[0]
    assert n, f"{label}: no target rows to check"
    assert device in ("numpy", "cuda"), f"{label}: device must be numpy or cuda, got {device!r}"
    if device == "cuda":
        import torch

        vt = torch.from_numpy(v).cuda()

    best = np.full(n, -2.0, dtype=np.float32)
    best_bank = [""] * n
    best_row = np.full(n, -1, dtype=np.int64)
    per_bank: dict = {}
    hits: list[dict] = []
    n_hits = 0
    report = []
    for name, path in banks:
        assert name not in per_bank, f"{label}: two banks are both named {name!r}"
        n_rows, read = open_bank(path, d)
        bbest = np.full(n, -2.0, dtype=np.float32)
        brow = np.full(n, -1, dtype=np.int64)
        t0, bhits = time.time(), 0
        for s in range(0, n_rows, chunk):
            m = min(chunk, n_rows - s)
            blk = read(s, m)
            blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-12)
            if device == "cuda":
                cos = (torch.from_numpy(blk).cuda() @ vt.T).cpu().numpy()
            else:
                cos = blk @ v.T  # [m, n]
            cmax = cos.max(axis=0)
            upd = cmax > bbest
            brow[upd] = s + cos.argmax(axis=0)[upd]
            bbest[upd] = cmax[upd]
            a_rows, t_cols = np.nonzero(cos > thr)
            bhits += int(a_rows.size)
            for a_row, t_col in zip(a_rows.tolist(), t_cols.tolist(), strict=True):
                if len(hits) < max_hits:
                    hits.append({"target": int(t_col), "bank": name,
                                 "bank_row": int(s + a_row), "cos": float(cos[a_row, t_col])})
        upd = bbest > best
        best_row[upd] = brow[upd]
        for j in np.nonzero(upd)[0]:
            best_bank[int(j)] = name
        best[upd] = bbest[upd]
        per_bank[name] = bbest
        n_hits += bhits
        secs = time.time() - t0
        report.append({"name": name, "path": path, "rows": n_rows, "seconds": round(secs, 1),
                       "max_cos": round(float(bbest.max()), 6), "n_hits": bhits})
        print(
            f"[{label}] {name}: {n_rows} rows in {secs:.0f}s, max cos {float(bbest.max()):.6f}, "
            f"{bhits} hits > {thr}",
            flush=True,
        )
    return {"n": n, "banks": report, "best": best, "best_bank": best_bank, "best_row": best_row,
            "per_bank": per_bank, "hits": hits, "n_hits": n_hits, "thr": thr}


def archived_banks(cfg, base: str):
    """The direction banks a DRAWN set is checked against at draw time, for this base.

    The 8B has run1's and run2's `pool_train/vecs.f32` in the archive and they are cheap, so the
    draw pays for them. The 27B's banks are the tier-B training directions
    (`data/v2-bundle/simple2m/*/dirs_f16.npy`, 8.94M rows / 92 GB): the same check, an
    order of magnitude more reading, and it covers blocks that are not being drawn -- so it is the
    `tierb` PRODUCT (`precompute/tierb.py`) and is not folded into every draw. `_leakage` says so
    by name rather than reporting "no banks exist", which was true until 2026-09-22 and is not now.
    """
    if base != "qwen3-8b":
        return []
    return [(run, f"{cfg['modal']['archive']}/data/{run}/bank/pool_train/vecs.f32")
            for run in ("run1", "run2")]


def _leakage(cfg, args, rows, vecs, od):
    """cos > 0.999 of every realact / sae direction against the archived training banks."""
    base, d = args["base"], cfg["bases"][args["base"]]["d"]
    banks = archived_banks(cfg, base)
    if not banks:
        od.note(
            f"leakage check: NOT RUN at draw time for base {base!r}. The banks that exist for it "
            f"are the tier-B training directions (8.94M rows, 92 GB), which are checked by "
            f"the `tierb` product against whichever blocks are named there -- see "
            f"precompute/tierb.py and results/tierb/."
        )
        return []
    idx = [i for i, r in enumerate(rows) if r["family"] in ("realact", "sae")]
    import numpy as np

    def _np(x):  # `vecs` is a list of torch rows here and a list of arrays in tierb.py
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    res = leak_scan(np.stack([_np(vecs[i]) for i in idx]), banks, d,
                    device="cuda", label="leakage")
    hits = [
        {
            "row": rows[idx[h["target"]]]["row"],
            "family": rows[idx[h["target"]]]["family"],
            "archive": h["bank"],
            "archive_row": h["bank_row"],
            "cos": h["cos"],
        }
        for h in res["hits"]
    ]
    od.note(
        f"leakage: cos > {LEAK_COS} of every realact and sae direction against "
        f"{'+'.join(n for n, _ in banks)} pool_train/vecs.f32 in the archive; {res['n_hits']} hits, "
        "reported only (checklist item 35 says report, never remove)"
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
        f"(the 27B banks are an open item)"
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
        od.write_json(
            "storage.json",
            {
                "storage": "unit",
                # Which mean run1's archived eval cache centred `realact_dirs` on is NOT recorded
                # anywhere we hold. Since 2026-09-23 that needs no key: a `storage: unit` set is
                # served exactly as shipped and has no centred reading at any mean, so there is
                # nothing to declare and nothing to get wrong.
                "note": (
                    "imported verbatim from run1's archived eval cache; no act.f32 exists, so these "
                    "directions cannot be moved to another mean"
                ),
            },
        )
        od.note("no mu_512.f32 and no leakage.jsonl here: neither is defined for an imported set")
        od.note(
            "STORAGE: `unit` with realact's mean UNKNOWN -- the archived cache records no centring "
            "convention. common.dirs_for labels these rows instead of re-deriving them."
        )
        od.note(f"source families present in the cache: {sorted(k[:-5] for k in es if k.endswith('_dirs'))}")
    return {"out": out, "rows": len(rows), "families": {f: n for f in IMPORT_FAMS}, "source": path}


def run(cfg, args):
    import torch

    if args.get("import_run1"):
        return import_run1(cfg, args)
    if C.is_ood_set(cfg, args["heldout"]):
        return run_ood(cfg, args)

    base, root = args["base"], args["root"]
    set_name = args["heldout"]
    assert base, "product targets needs --base"
    spec = cfg["bases"][base]
    hspec = cfg["heldout"][set_name]
    fams = C.families_for(cfg, set_name, base)
    seed = int(hspec["seed"])
    # WHICH SAE the `sae` family's feature ids belong to. `qwen36-27b` has carried two since
    # dict2m; common.sae_key_for is the one rule (explicit --sae wins, a single-SAE base needs none,
    # two SAEs and no flag is refused rather than guessed).
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
    args["sae_key"] = sae_key

    toks, docs = C.load_corpus(base, root)
    sizes_json = f"{C.sae_dir(sae_key, root)}/sizes.json"
    assert os.path.exists(sizes_json), (
        f"{sizes_json} is missing: run `--product stats --base {base}` first, the sae family is "
        f"drawn from its fire counts"
    )
    C.stats_mu(cfg, base, root)  # fail before the model load if the ONE centring mean is missing
    rng = np.random.default_rng(seed)
    out = C.heldout_dir(base, set_name, root)
    inputs = {
        "corpus": C.corpus_dir(base, root),
        "sae": C.sae_dir(sae_key, root),
        "mu (centring)": f"{C.stats_dir(base, root)}/mu.f32",
        "seed": seed,
        "families": {f: s["n"] for f, s in fams.items()},
    }
    with C.outdir(out, args, inputs=inputs) as od:
        model, tok = C.load_base(cfg, base)
        # ENCODER ONLY: the `sae` family's direction is unit(W_enc[:, f]) (common.sae_dirs) and the
        # draw reads fire counts off the stats product; nothing here touches W_dec, which at 2^21
        # features is another 43 GB in fp32.
        sae = C.load_sae(
            C.sae_path(cfg, sae_key), spec["d"], device="cuda", dtype=torch.float32, need_decoder=False
        )
        rows, vecs, acts = [], [], []
        for fam in FAMILY_ORDER:  # FIXED order: it determines the rng stream
            if fam not in fams:
                continue
            n = int(fams[fam]["n"])
            if fam == "realact":
                r, v, a = _realact(cfg, args, model, tok, toks, docs, n, rng, od)
            elif fam == "random":
                r, v = _random(cfg, args, n, seed)
                a = v  # not centrable: the act.f32 row IS the direction (module docstring)
            else:
                r, v = _sae(cfg, args, sae, n, int(hspec["sae_strata"]), int(hspec["sae_min_fires"]), rng, od)
                a = v
            assert not C.family_centrable(cfg, fam) or a is not v, (
                f"family {fam!r} is `centrable` in config.yaml but its draw returned the direction "
                f"as its own raw activation; act.f32 would then be uncentrable"
            )
            rows += r
            vecs += v
            acts += list(a)
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
        a = torch.stack(acts).float()
        nrm = v.norm(dim=-1)
        assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-5), (
            f"held-out vectors must be unit rows; got min {nrm.min():.6f} max {nrm.max():.6f}"
        )
        assert a.shape == v.shape, f"act.f32 is {tuple(a.shape)} but vecs.f16 is {tuple(v.shape)}"
        # The contract in one assert: vecs.f16 IS unit(act.f32), so nothing downstream has to
        # trust the docstring. fp32 both sides; the f16 round trip happens after this.
        cos_av = torch.nn.functional.cosine_similarity(a, v, dim=-1)
        assert float(cos_av.min()) > 1 - 1e-5, (
            f"vecs.f16 is not unit(act.f32) on every row: min cos {float(cos_av.min()):.6f}"
        )
        hits = _leakage(cfg, args, rows, vecs, od)
        ordered = [{"row": r["row"], **{k: x for k, x in r.items() if k != "row"}} for r in rows]
        od.write_jsonl("ids.jsonl", ordered)
        od.write_array("act.f32", a, "float32")
        od.write_array("vecs.f16", v, "float16")
        od.write_json(
            "storage.json",
            C.storage_record(cfg, set_name, sorted({r["family"] for r in rows}), sae_key),
        )
        od.write_jsonl("leakage.jsonl", hits)
        od.note(
            f"rebuild: `modal run precompute/modal_app.py --product targets --base {base} "
            f"--set {set_name} --root {root}` at repo commit {args.get('repo_commit', '?')[:12]}"
        )
        od.note(
            f"draw order {FAMILY_ORDER} from ONE np.random.default_rng({seed}); the random family "
            f"uses a separate torch.Generator().manual_seed({seed}) (the convention)"
        )
        od.note("vecs.f16 rows are unit in fp32 before the cast; the f16 round-trip is ~1e-3 off unit")
        od.note(
            "STORAGE: `raw` (storage.json). act.f32 [N, d] fp32 is the row's vector before any "
            "mean was subtracted and vecs.f16 is unit(act); the centring mean is named per RUN "
            "(common.dirs_for) and never stored. fp32 rather than f16 for act.f32 because f16 "
            "costs ~1e-3 on a norm-90 vector and the exact-solve migration of a `storage: unit` "
            "set needs better than that."
        )
    return {
        "out": out,
        "rows": len(rows),
        "families": {f: sum(1 for r in rows if r["family"] == f) for f in FAMILY_ORDER},
        "leakage_hits": len(hits),
    }


# =============================================================================================
# The OOD generalisation sets (design infra/2026-09-18_ood-eval-design.md §2, §3, §11 R5)
#
#     <root>/base/<base>/heldout/2026-09-18_ood_v1/
#         ids.jsonl      one row per target: row, family, arm, id, p, L, the source coordinates,
#                        the tokenisation covariates, the licence
#         vecs.f16       [N, d] unit rows, `unit(X[p] - stats/mu.f32)` -- the ENGLISH centring mean
#         windows.i32    [N, 512] the 512-token no-BOS window each target was read from, so `nll`
#                        and any re-analysis need neither the source nor the network
#
# The draw runs OFFLINE: `corpus --arm` already wrote each arm's target pool (the documents after
# its corpus in the same permuted row stream) into `corpora/<arm>/pool_windows.i32`.
# =============================================================================================

OOD_PRESAMPLE_ROWS = 64  # pool windows forwarded for the norm-filter presample (as `_realact`)


def _forward_windows(model, read_layer, windows, device="cuda"):
    """Read-layer activations [n, T, d] (fp32, cpu) of an [n, T] token array. No sink token."""
    import torch

    out = []
    for s in range(0, len(windows), REALACT_DOCS_PER_FWD):
        ids = np.asarray(windows[s : s + REALACT_DOCS_PER_FWD], dtype=np.int64)
        t = torch.from_numpy(ids).to(device)
        h, _ = C.read_resid(
            model, read_layer, {"input_ids": t, "attention_mask": torch.ones_like(t)}, pool="all"
        )
        out.append(h.cpu())
    return torch.cat(out)


def _span_in_corpus(toks, span) -> bool:
    """Does the token sequence `span` occur verbatim anywhere in `toks`? (design §4)

    Pruned on the first TWO tokens before the full compare, which is what keeps a common leading
    token (a space, a newline) from costing a length-L compare at every one of its occurrences.
    """
    m = len(span)
    if m == 0 or m > len(toks):
        return False
    cand = np.flatnonzero(toks[: len(toks) - m + 1] == span[0])
    if m > 1 and cand.size:
        cand = cand[toks[cand + 1] == span[1]]
    for i in cand:
        if np.array_equal(toks[i : i + m], span):
            return True
    return False


def _ood_arm_draw(cfg, args, model, tok, arm, n, seed, od):
    """One arm's `n` targets, from its pool windows. Returns (rows, acts, windows, report)."""
    import torch

    base, root = args["base"], args["root"]
    read_layer = cfg["bases"][base]["read_layer"]
    spec = C.ood_arm(cfg, arm)
    cdir = C.corpus_dir(base, root, arm)
    assert os.path.exists(f"{cdir}/stream.json"), (
        f"no {cdir}: run `--product corpus --base {base} --arm {arm}` first (it writes the target "
        f"pool this draw reads)"
    )
    with open(f"{cdir}/stream.json") as fh:
        stream = json.load(fh)
    pool = C.read_jsonl(f"{cdir}/pool.jsonl")
    pool_n, window = len(pool), int(stream["pool_window"])
    wins = C.read_array(f"{cdir}/pool_windows.i32", "int32", (pool_n, window))
    rng = C.arm_rng(arm, seed)
    p_all = rng.integers(REALACT_P_MIN, window, size=pool_n)
    l_all = rng.integers(SPAN_MIN, SPAN_MAX + 1, size=pool_n)

    t0 = time.time()
    n_pre = min(OOD_PRESAMPLE_ROWS, pool_n)
    pre = _forward_windows(model, read_layer, wins[:n_pre])
    ii = rng.integers(0, n_pre, NORM_PRESAMPLE)
    pp = rng.integers(REALACT_P_MIN, window, NORM_PRESAMPLE)
    med = float(pre[torch.from_numpy(ii), torch.from_numpy(pp)].norm(dim=-1).median())
    del pre

    xs = []
    for s in range(0, pool_n, 256):
        h = _forward_windows(model, read_layer, wins[s : s + 256])
        xs.append(h[torch.arange(h.shape[0]), torch.from_numpy(p_all[s : s + h.shape[0]])].clone())
    x = torch.cat(xs)
    nrm = x.norm(dim=-1)
    ok = (nrm > 1e-3) & (nrm <= NORM_FILTER_MULT * med)
    print(
        f"[ood {arm}] presample median raw norm {med:.1f}; {int(ok.sum())}/{pool_n} pool windows "
        f"pass the filter ({time.time() - t0:.0f}s)",
        flush=True,
    )

    note = C.check_token_bytes(tok, [int(t) for t in wins[0]])
    rows, acts, windows = [], [], []
    for i in np.flatnonzero(ok.numpy()):
        if len(rows) >= n:
            break
        i = int(i)
        p, span = int(p_all[i]), int(l_all[i])
        ids = wins[i]
        cov = C.token_covariates(tok, [int(t) for t in ids], p, spec["script"], bool(spec["unspaced"]))
        rows.append(
            {
                "family": spec["family"],
                "arm": arm,
                "id": f"{arm}:pool{i}:p{p}:L{span}",
                "stratum": cov.get("tok_class"),
                "pool_i": i,
                "src_rows": pool[i]["rows"],
                "src_dataset": stream["dataset"],
                "src_revision": stream["revision"],
                "src_split": stream.get("split"),
                "src_files": stream["files"],
                "licence": stream.get("licence"),
                "p": p,
                "L": span,
                "act_norm": round(float(nrm[i]), 3),
                "span_text": tok.decode([int(t) for t in ids[p - span + 1 : p + 1]]),
                **{k: v for k, v in cov.items() if k not in ("unit_start", "unit_end")},
            }
        )
        acts.append(x[i].float())  # RAW X[p]; the mean is named per RUN, not baked in here
        windows.append(np.asarray(ids, dtype=np.int32))
    assert len(rows) == n, (
        f"arm {arm}: only {len(rows)} of {n} targets survived the raw-norm filter over a "
        f"{pool_n}-document pool"
    )

    toks = np.memmap(f"{cdir}/tokens.i32", dtype=np.int32, mode="r")
    verbatim = sum(
        1
        for r, w in zip(rows, windows, strict=True)
        if _span_in_corpus(toks, w[r["p"] - r["L"] + 1 : r["p"] + 1])
    )
    report = {
        "arm": arm,
        "family": spec["family"],
        "n": len(rows),
        "pool_n": pool_n,
        "pass_norm_filter": int(ok.sum()),
        "norm_median": round(med, 2),
        "verbatim_spans": verbatim,
        "verbatim_rate": round(verbatim / len(rows), 4),
        "corpus_tokens": int(stream["corpus_tokens"]),
        "revision": stream["revision"],
        "byte_piece_rate": round(sum(r["byte_piece"] for r in rows) / len(rows), 4),
        "tok_class": {
            c: sum(1 for r in rows if r.get("tok_class") == c)
            for c in sorted({r.get("tok_class") for r in rows})
        },
        "char_type": {
            c: sum(1 for r in rows if r["char_type"] == c)
            for c in sorted({r["char_type"] for r in rows})
        },
    }
    print(f"[ood {arm}] {report}", flush=True)
    od.note(
        f"arm `{arm}` ({spec['family']}): {len(rows)} targets from a {pool_n}-doc pool "
        f"({int(ok.sum())} passed the norm filter, presample median {med:.1f}); source "
        f"{stream['dataset']}@{stream['revision'][:12]}; corpus {stream['corpus_tokens']} tokens; "
        f"verbatim shown span in its own corpus: {verbatim}/{len(rows)} "
        f"({verbatim / len(rows):.1%}); byte-piece rate {report['byte_piece_rate']:.3f}; "
        f"tok_class {report['tok_class']}; {note}"
    )
    return rows, acts, windows, report


def run_ood(cfg, args):
    """`--set <an ood set>`: 64 targets per arm, or the `_unitend` variant of an existing set."""
    import torch

    base, root = args["base"], args["root"]
    set_name = args["heldout"]
    spec = cfg["heldout"][set_name]
    seed = int(spec["seed"])
    n = int(spec["n_per_arm"])
    arms = C.ood_set_arms(cfg, set_name)
    only = [a for a in (args.get("arm") or "").split(",") if a]
    if only:
        for a in only:
            assert a in arms, f"--arm {a!r} is not in set {set_name!r} ({arms})"
        arms = only
    read_layer = cfg["bases"][base]["read_layer"]
    variant_of = spec.get("variant_of")

    out = C.heldout_dir(base, set_name, root)
    inputs = {
        "base": base,
        "set": set_name,
        "arms": arms,
        "n_per_arm": n,
        "seed": seed,
        "storage": "raw (act.f32 + unit(act)); the centring mean is NAMED per run, not stored",
    }
    if variant_of:
        inputs["variant_of"] = C.heldout_dir(base, variant_of, root)
    with C.outdir(out, args, inputs=inputs) as od:
        model, tok = C.load_base(cfg, base)
        rows, acts, windows, reports = [], [], [], []
        if variant_of:
            rows, acts, windows, reports = _ood_variant(cfg, args, model, tok, variant_of, arms, od)
        else:
            for arm in arms:
                r, v, w, rep = _ood_arm_draw(cfg, args, model, tok, arm, n, seed, od)
                rows += r
                acts += v
                windows += w
                reports.append(rep)
        for i, row in enumerate(rows):
            row["row"] = i
        # STORAGE: raw (plan §1.2, §4.3.1). act.f32 is X[p] as read and vecs.f16 is unit(act);
        # the English centring the design fixes (§10 decision 10) is now NAMED at run time --
        # `--mu base/{base}/stats/mu.f32`, which is every OOD product's default through the
        # MAEM's own `mu:` -- instead of being baked into the stored row. The design's choice is
        # preserved exactly; what changes is that a per-arm-centred rescoring is a flag rather
        # than a re-draw.
        a = torch.stack(acts).float()
        v = torch.nn.functional.normalize(a, dim=-1)
        nrm = v.norm(dim=-1)
        assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-5), (
            f"held-out vectors must be unit rows; got min {nrm.min():.6f} max {nrm.max():.6f}"
        )
        # The contract in one assert, as the English path has it: vecs.f16 IS unit(act.f32).
        cos_av = torch.nn.functional.cosine_similarity(a, v.float(), dim=-1)
        assert float(cos_av.min()) > 1 - 1e-5, (
            f"vecs.f16 is not unit(act.f32) on every row: min cos {float(cos_av.min()):.6f}"
        )
        ordered = [{"row": r["row"], **{k: x for k, x in r.items() if k != "row"}} for r in rows]
        od.write_jsonl("ids.jsonl", ordered)
        od.write_array("act.f32", a, "float32")
        od.write_array("vecs.f16", v, "float16")
        od.write_array("windows.i32", np.stack(windows), "int32")
        od.write_json(
            C.STORAGE_FILE,
            C.storage_record(cfg, set_name, sorted({r["family"] for r in rows})),
        )
        od.write_json("arms.json", {"arms": arms, "reports": reports})
        od.section(
            "Draw",
            [
                f"OOD generalisation set `{set_name}` (design "
                "`infra/2026-09-18_ood-eval-design.md` §2/§3, review §11 R5).",
                "",
                f"- {len(arms)} arms x {n} targets; the family of a row is its ARM's family and "
                "`arm` is the second stratification key;",
                "- each arm's targets come from `corpora/<arm>/pool_windows.i32`, the documents "
                "AFTER that arm's corpus in the one permuted row stream of the source slice, so "
                "corpus and target documents are disjoint by construction (design §2);",
                f"- the rule is `_realact`'s, unchanged: a 512-token no-BOS window, `p ~ "
                f"U[{REALACT_P_MIN}, 512)`, shown span `L ~ U[{SPAN_MIN}, {SPAN_MAX}]` ending at "
                f"p, a raw-norm filter (> 1e-3 and <= {NORM_FILTER_MULT}x a "
                f"{NORM_PRESAMPLE}-position presample median of the SAME arm's pool), applied at "
                "selection only;",
                "- STORAGE `raw` (storage.json): `act.f32` is `X[p]` as read and `vecs.f16` is "
                "`unit(X[p])`, UNCENTRED. The direction a run scores against is derived at read "
                "time by `common.dirs_for` under the mean that run NAMES. The design's mean is "
                "the ENGLISH `stats/mu.f32` on every arm (open decision 10: the centring mean is "
                "the inverter's input convention, not a property of the domain), which is what "
                "each MAEM's own `mu:` resolves to for the old primary; `rl-final` names "
                "`whiten_mu` instead, and both are legal readings of the same stored rows;",
                "- `p` and `L` come from `common.arm_rng(arm, seed)`, a stream independent of the "
                "row permutation `common.arm_perm(arm, ...)` the corpus was cut from;",
                "- the tokenisation covariates (`tok_class`, `n_subtokens`, `byte_piece`, "
                "`whole_char`, `multi_char`, `char_type`) are from the TOKENIZER alone, at draw "
                "time, on the CPU (`common.token_covariates`); `unspaced` arms get the single "
                "class `unspaced` and no `n_subtokens`, because they have no defensible word "
                "boundary (design §3);",
                "- `windows.i32` [N, 512] stores each target's window, so `nll` and any "
                "re-analysis need neither the source nor the network.",
            ],
        )
        od.note(
            "verbatim-span rate per arm is in the per-arm notes above and in `arms.json`: the "
            "fraction of targets whose SHOWN span occurs verbatim somewhere in that arm's own "
            "corpus (design §4 -- near-duplicate documents are reported, never masked)"
        )
        od.note(f"read layer {read_layer}; d {cfg['bases'][base]['d']}; one row per target")
    return {
        "out": out,
        "rows": len(rows),
        "arms": {r["arm"]: r["n"] for r in reports},
        "reports": reports,
    }


def _ood_variant(cfg, args, model, tok, variant_of, arms, od):
    """`_unitend` (review R5): the base set's windows with `p` moved to the end of its unit.

    wordend on the spaced-script and code arms (p -> the last token of its whitespace unit),
    charend on the unspaced ones (p -> the last byte piece of its character, and only where the
    token at p is a partial character). Targets whose `p` already ends its unit are carried over
    unchanged with `variant_rule: none`, so the variant set is row-for-row comparable.
    """
    base, root = args["base"], args["root"]
    read_layer = cfg["bases"][base]["read_layer"]
    src = C.heldout_dir(base, variant_of, root)
    base_rows = C.read_jsonl(f"{src}/ids.jsonl")
    n_all = len(base_rows)
    wins_all = C.read_array(f"{src}/windows.i32", "int32", (n_all, REALACT_WINDOW))
    keep = [i for i, r in enumerate(base_rows) if r["arm"] in arms]
    assert keep, f"none of {arms} is in {src}/ids.jsonl"
    rows, acts, windows, reports = [], [], [], []
    for arm in arms:
        sel = [i for i in keep if base_rows[i]["arm"] == arm]
        wins = wins_all[sel]
        h = _forward_windows(model, read_layer, wins)
        moved = 0
        for k, i in enumerate(sel):
            b = dict(base_rows[i])
            p_new = int(b["unitend_p"])
            rule = b["unitend_rule"]
            moved += rule != "none"
            span = int(b["L"])
            ids = wins[k]
            row = {
                **{x: y for x, y in b.items() if x != "row"},
                "p": p_new,
                "p_orig": int(b["p"]),
                "variant_rule": rule,
                "variant_of": variant_of,
                "id": f"{arm}:pool{b['pool_i']}:p{p_new}:L{span}",
                "span_text": tok.decode([int(t) for t in ids[max(0, p_new - span + 1) : p_new + 1]]),
                "act_norm": round(float(h[k, p_new].norm()), 3),
            }
            rows.append(row)
            acts.append(h[k, p_new].float())  # RAW X[p_new], as the base set
            windows.append(np.asarray(ids, dtype=np.int32))
        rules = {
            r: sum(1 for i in sel if base_rows[i]["unitend_rule"] == r)
            for r in ("wordend", "charend", "none")
        }
        reports.append({"arm": arm, "n": len(sel), "moved": moved, "rules": rules})
        od.note(f"arm `{arm}`: {moved}/{len(sel)} targets moved; rules {rules}")
        print(f"[ood-variant {arm}] {moved}/{len(sel)} moved, rules {rules}", flush=True)
    return rows, acts, windows, reports
