"""Draw the standard dict2m target set: 40,000 eval-split features, 80/20.

    python -m features.spawn --product draw_dict2m --base qwen36-27b \
        --sae qwen36-27b/dict2m --gpu cpu

    python -m features.spawn --product draw_dict2m --base qwen36-27b \
        --sae qwen36-27b/dict2m --set 2026-09-21_dict2m_64 --n 64 --stratified \
        --root /vol/tmp/sae-smoke64 --gpu cpu

    python -m features.spawn --product draw_dict2m --base qwen36-27b \
        --sae qwen36-27b/dict2m --set 2026-09-21_v3_dict2m --n 512 --stratified \
        --seed 20260921 --sides enc,dec --include <64 ids> --gpu h200

Writes the shape the rest of the pipeline already reads:

    <root>/base/<base>/heldout/<set>/
        ids.jsonl   one row per target: row, family ("sae"), sae_key (which dictionary),
                    sae_side (enc | dec), id (feature id), stratum, side, ...
        vecs.f16    [N, d] unit rows, row i is ids.jsonl line i
        README.md   the draw, the eligibility rule, and how the split is held out

Every feature comes from the `eval` split (seed 2026, 100,000 features). That is
load-bearing: the checkpoint was TRAINED on the `sft` (1,847,152) and `rl` (150,000)
sides, so a draw over all 2^21 features would put trained-on features in the test set
and the held-out claim would die silently. `build()` asserts every drawn id is on the
eval side.

The 80/20 `side` column splits OUR analysis, not the model's training: both halves are
equally unseen by the MAEM. It exists so thresholds, strata cuts and ablations can be
chosen on `fit` without touching the numbers reported from `report`.

Stratification is RECORDED, not sampled: the draw is uniform over eligible features and
each row carries its fire-count quartile. A stratified draw would force equal quartiles
and make the overall mean unrepresentative of the dictionary.

`--stratified` (with `--n`) asks for exactly that forced draw, and is for a SMOKE, not for
a reported mean. `--n 64 --stratified` takes 16 features from each quartile of the eligible
pool, because a uniform 64 lands almost entirely in the middle of the density distribution
and a smoke that never touches the rare tail cannot see the effect the tail has. Two things
follow and are recorded rather than assumed away:

* the cuts are the quartiles of the ELIGIBLE POOL, not of the drawn set. With n/4 drawn per
  stratum the drawn set's own quartiles ARE the stratum boundaries, so cutting on them would
  define the strata by the answer. `meta["cuts_over"]` says which was used.
* any mean over a stratified draw is a mean over a REWEIGHTED dictionary. Per-stratum rows
  are the honest read; the "all" row is a mean over four equal quartiles, not over the SAE.

READING THE STATS. `precompute/stats.py` writes `sizes.json` (sizes, learned gate, d_sae)
beside `fire_counts.i64` / `max_act.f16` / `mean_when_active.f16`. The 2M SAE's stats
directory on the volume does NOT have it -- it carries only the OutDir sidecar `index.json`,
whose `fire_counts.i64` entry records the [F, n_sizes, 2] shape. Both layouts are read for a
stratified draw and the shape comes from whichever file is there; it is never assumed,
because `fire_counts.i64` is a headerless buffer and a wrong F reshapes it into a different,
equally plausible-looking table instead of failing. The UNIFORM draw still keys on
`sizes.json` alone, deliberately -- see `_read_sae_stats`.

FAMILY LABEL (fixed 2026-09-21). These rows are stamped `family: "sae"`, the label every
downstream product filters on, with the dictionary named separately per row in `sae_key`.
The first draw used `family: "dict2m_enc"`, which no consumer selected on, so the set was
invisible to `scan`, `top1_act`, `repo_examples`, `gcg`, `score`'s per-family means and
autointerp until each was taught to accept the second label. The 2k set already on the
volume keeps its old label and that acceptance stays; a re-draw carries `sae`.
"""
from __future__ import annotations

import json
import os

import numpy as np

import precompute.common as C

N_FEATURES = 40_000
FIT_FRACTION = 0.8
MIN_FIRES = 20           # counted with the learned gate, as config.yaml's sae_min_fires
N_STRATA = 4
DRAW_SEED = 20260920


def _eval_split_ids(cfg, args) -> np.ndarray:
    """Feature ids on the eval side, read from the bundle's own split file."""
    import pandas as pd

    path = args.get("feature_split") or (
        f"{args['root']}/data/v2-bundle/heldout/feature_split.parquet")
    split = pd.read_parquet(path)
    ids = split.loc[split["split"] == "eval", "feature_id"].to_numpy()
    assert len(ids) == 100_000, f"eval split has {len(ids)} features, expected 100,000"
    return np.sort(ids)


def _bundle_corpus_peaks(args, eval_ids, required: bool = True):
    """Corpus peak per eval feature from the 1.0B scan, aligned to `eval_ids`.

    The rank-0 window activation IS the shipped corpus_peak -- verified exactly on all
    512 standard-eval features (max abs diff 0.000000).

    `required=False` returns None instead of raising when the bundle's 100k-window parquet is
    not under `--root`. A stratified draw cuts on OUR 16M fire counts and carries the 1.0B peak
    as a column only, so it must not die for want of a table it measures nothing with; the
    uniform draw takes its eligibility AND its strata from that table and cannot proceed
    without it.
    """
    import pandas as pd

    path = args.get("maxact_windows") or (
        f"{args['root']}/data/v2-bundle/heldout/"
        f"eval_2m_features_100k_windows.parquet")
    if not required and not os.path.exists(path):
        print(f"[draw] no 1.0B window table at {path}; rows carry corpus_peak_1b: null",
              flush=True)
        return None
    win = pd.read_parquet(path, columns=["feature_id", "rank", "act"])
    top = win[win["rank"] == 0].set_index("feature_id")["act"]
    return top.reindex(eval_ids).to_numpy(dtype=np.float32)


def _read_sae_stats(sae_stats: str, index_fallback: bool):
    """(gated fires at the LARGEST corpus size [F], max activation [F]), or (None, None).

    `fires[:, -1, 1]` is the gated count at the last of `corpus.sizes` -- 16M -- which is the
    axis config.yaml's `sae_strata` names. The last axis of `fire_counts.i64` is (raw, gated).

    Two layouts carry the same arrays. `precompute/stats.py` writes `sizes.json` (sizes,
    threshold, d_sae); the 2M SAE's stats pass shipped only the OutDir sidecar `index.json`,
    which records every array's dtype and shape. `index_fallback` turns the second one on.

    It is OFF for the uniform draw ON PURPOSE. That draw's rule is frozen -- the sets already on
    the volume were written under "sizes.json, else the bundle's 1.0B peaks" -- and reading the
    2M SAE's index.json for it would silently move both its eligibility rule (fires >= MIN_FIRES
    instead of peak > 0) and its strata axis (16M density instead of 1.0B peak) without a flag
    saying so. A stratified draw has no such history and needs the fire counts by definition,
    so it reads either layout.
    """
    sizes_path, index_path = f"{sae_stats}/sizes.json", f"{sae_stats}/index.json"
    if os.path.exists(sizes_path):
        with open(sizes_path) as fh:
            sizes_meta = json.load(fh)
        assert sizes_meta["threshold"] > 0, "stats recorded no gate"
        shape = (int(sizes_meta["d_sae"]), len(sizes_meta["sizes"]), 2)
    elif index_fallback and os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        assert "fire_counts.i64" in index, (
            f"{index_path} names no fire_counts.i64, so the shape of "
            f"{sae_stats}/fire_counts.i64 is unknown -- it is a headerless buffer and there is "
            f"nothing else to recover F and n_sizes from")
        shape = tuple(int(x) for x in index["fire_counts.i64"]["shape"])
        assert len(shape) == 3 and shape[2] == 2, (
            f"{index_path} says fire_counts.i64 is {list(shape)}; expected [F, n_sizes, 2]")
    else:
        return None, None
    fires = C.read_array(f"{sae_stats}/fire_counts.i64", "int64", shape)
    max_act = C.read_array(f"{sae_stats}/max_act.f16", "float16",
                           (shape[0],)).astype(np.float32)
    return fires[:, -1, 1], max_act


def _stratified_draw(eligible, rank_stat, n, rng):
    """(drawn, stratum, cuts, pool sizes): n / N_STRATA features from EACH quartile of the pool.

    The quartiles are of `log10(rank_stat)` over the whole ELIGIBLE POOL, so the cuts describe
    the dictionary and not the sample. `searchsorted(..., side="right")` puts a value sitting
    exactly on a cut in the upper stratum; that is not cosmetic here, because fire counts are
    integers and their log10 ties heavily, so a cut frequently IS an attained value.

    Both returned arrays are sorted by feature id, so `ids.jsonl` stays in id order as it is for
    a uniform draw and `stratum` still lines up row for row.
    """
    assert n % N_STRATA == 0, (
        f"--stratified draws n/{N_STRATA} per stratum, so --n {n} must be divisible by "
        f"{N_STRATA}")
    per = n // N_STRATA
    dens = np.log10(rank_stat)
    assert np.isfinite(dens).all(), (
        f"log10 of the stratum statistic is not finite on every eligible feature -- eligibility "
        f"is fires >= {MIN_FIRES}, so this means the eligibility filter did not run")
    cuts = np.quantile(dens, [0.25, 0.5, 0.75])
    pool_stratum = np.searchsorted(cuts, dens, side="right")
    picks, strata, pool_n = [], [], []
    for s in range(N_STRATA):
        cand = eligible[pool_stratum == s]
        pool_n.append(int(len(cand)))
        assert len(cand) >= per, (
            f"stratum {s} of the eligible pool holds {len(cand)} features but the draw needs "
            f"{per}: the quartile cuts {[float(c) for c in cuts]} collapsed onto each other, "
            f"which is what happens when one fire count is attained by more than a quarter of "
            f"the pool")
        picks.append(rng.choice(cand, size=per, replace=False))
        strata.append(np.full(per, s, dtype=int))
    drawn = np.concatenate(picks)
    stratum = np.concatenate(strata)
    order = np.argsort(drawn)
    return drawn[order], stratum[order], cuts, pool_n


def _include_ids(args) -> np.ndarray:
    """`--include <file>`: feature ids that MUST be in the draw, whatever the sample picks.

    The eval plan needs the upstream standard-eval 512 inside our draw so the two blocks are NESTED rather
    than merely comparable, and `--stratified`/`--seed` cannot express that: they choose how the
    sample is spread, not what it must contain. Accepts a .parquet with a `feature_id` column, a
    .npy, or one id per line.
    """
    path = (args.get("include") or "").strip()
    if not path:
        return np.zeros(0, dtype=np.int64)
    assert os.path.exists(path), f"--include {path}: no such file"
    if path.endswith(".parquet"):
        import pandas as pd

        ids = pd.read_parquet(path)["feature_id"].to_numpy()
    elif path.endswith(".npy"):
        ids = np.load(path)
    else:
        with open(path) as fh:
            ids = np.array([int(ln) for ln in fh if ln.strip()], dtype=np.int64)
    ids = np.unique(np.asarray(ids).astype(np.int64))
    assert ids.size, f"--include {path} lists no feature ids"
    return ids


def build(cfg, args):
    base, root = args["base"], args["root"]
    # ONE --sae syntax, the same rule every other product uses (common.sae_key_for): a full
    # `<base>/<name>` key, or nothing at all when the base carries a single SAE. The old inline
    # form turned a missing --sae into the literal key "<base>/None".
    sae_key = C.sae_key_for(cfg, base, args.get("sae") or "")
    spec = cfg["bases"][base]
    # D6: never a dated default. This product CREATES a set directory and `--force` rmtrees what
    # is there; a name it made up is a name nobody can ask for twice.
    set_name = args["heldout"]
    assert set_name, (
        "draw_dict2m writes a held-out set, so it needs an explicit --set <name> (D6). It used to "
        "fall back to today's date, and through modal_app to the LIVE default set."
    )
    out_dir = C.heldout_dir(base, set_name, root)
    assert args.get("force") or not os.path.exists(out_dir), (
        f"{out_dir} already exists; refusing to overwrite without --force")

    n = int(args.get("n") or N_FEATURES)
    stratified = bool(args.get("stratified"))
    seed = int(args.get("seed") or DRAW_SEED)
    assert n > 0, f"--n must be positive, got {n}"
    # `--sides enc` (the default, and every set drawn before 2026-09-21) or `--sides enc,dec`.
    # The DRAW is over features and does not depend on this: the same ids, the same strata and
    # the same fit/report labels are emitted once per side, so `--sides enc,dec` is the one-side
    # set with a second block appended, never a different sample.
    sides = tuple(x.strip() for x in str(args.get("sides") or "enc").split(",") if x.strip())
    assert sides and len(set(sides)) == len(sides) and set(sides) <= set(SIDES), (
        f"--sides {args.get('sides')!r}: give a comma-separated subset of {list(SIDES)} with no "
        f"repeats (default 'enc')"
    )

    # Eligibility and strata come from our own corpus scan when it exists, and from
    # the 1.0B-token scan when it does not. The fallback is not a degraded mode:
    # the upstream scan is 60x larger than ours, reports 0 dead features over all 2^21, and its
    # rank-0 window activation reproduces the shipped corpus_peak of the standard-eval
    # 512 EXACTLY (max abs diff 0.000000, features/registry.py). So every eval feature
    # is already known to fire, and the >= MIN_FIRES gate is satisfied before we compute
    # anything. Fire counts on OUR corpus are a better stratification axis and get
    # joined in as a column when the stats pass lands -- they are not a prerequisite.
    # NOTE `--root` moves the INPUTS too, not only the output: the stats are read from
    # <root>/base/<base>/sae/<sae>/ and the bundle from <root>/data/v2-bundle/,
    # because C.sae_dir and the bundle paths below are all root-relative. A draw under a smoke
    # root needs those files copied under it.
    sae_stats = C.sae_dir(sae_key, root)
    gated_full, max_act = _read_sae_stats(sae_stats, index_fallback=stratified)
    have_stats = gated_full is not None
    if have_stats:
        strat_name, strat_source = "log10_gated_fires_16M", "our 16M corpus scan"
    else:
        assert not stratified, (
            f"--stratified cuts on gated fires at 16M and {sae_stats} carries neither "
            f"sizes.json nor an index.json naming fire_counts.i64, so there is nothing to "
            f"stratify on. Under --root {root!r} the stats are read from that path")
        strat_name, strat_source = "log10_corpus_peak_1B", "the 1.0B-token scan"
        print("[draw] no stats pass for this SAE yet; using the bundle's corpus peaks "
              "for eligibility and strata", flush=True)
    eval_ids = _eval_split_ids(cfg, args)

    # An explicit subset (features.parquet from shared/<name>/) takes the draw as given:
    # the point of a shared subset is that everyone gets the SAME features, so nothing
    # is re-sampled here and the split/stratum columns are carried through verbatim.
    subset = args.get("subset")
    if subset:
        import pandas as pd

        assert not stratified and not args.get("n"), (
            "--subset takes its draw AS GIVEN -- that is the whole point of a shared subset -- "
            "so --n and --stratified have nothing to act on here")
        sub = pd.read_parquet(subset)
        drawn = sub["feature_id"].to_numpy()
        assert np.isin(drawn, eval_ids).all(), (
            "a subset feature is not on the eval side -- it would not be held out")
        side = sub["split"].to_numpy().astype(str)
        stratum = sub["stratum"].to_numpy().astype(int)
        peak_by_id = dict(zip(sub["feature_id"].tolist(), sub["corpus_peak_1b"].tolist(),
                              strict=True))
        strat_name, strat_source = "log10_corpus_peak_1B", f"subset {subset}"
        have_stats = False
        gated_full = None
        meta_extra = {"subset": subset, "eligible": int(len(sub))}
        cuts = []
        return _finish(cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
                       peak_by_id, strat_name, strat_source, have_stats, gated_full, None,
                       cuts, meta_extra)

    peaks_1b = _bundle_corpus_peaks(args, eval_ids, required=not stratified)
    if have_stats:
        eligible = eval_ids[(gated_full[eval_ids] >= MIN_FIRES) & (max_act[eval_ids] > 0)]
        rank_stat = gated_full[eligible].astype(np.float64)
    else:
        keep = peaks_1b > 0
        eligible = eval_ids[keep]
        rank_stat = peaks_1b[keep].astype(np.float64)
    print(f"[draw] eval split {len(eval_ids):,} -> {len(eligible):,} eligible, "
          f"strata from {strat_source}", flush=True)
    assert len(eligible) >= n, f"only {len(eligible)} eligible features, need {n}"

    rng = np.random.default_rng(seed)
    forced = _include_ids(args)
    if forced.size:
        # Taken as given; the SAMPLE fills the rest from the eligible pool with them removed, so
        # nothing is drawn twice. They must still be on the eval side or the held-out claim dies
        # for those rows, and they must be ELIGIBLE or the shortfall is recorded rather than
        # silently accepted.
        assert np.isin(forced, eval_ids).all(), (
            f"{int((~np.isin(forced, eval_ids)).sum())} of the {forced.size} --include features "
            f"are not on the eval split, so they were TRAINED on and cannot be held out"
        )
        missing = forced[~np.isin(forced, eligible)]
        assert args.get("allow_short") or not missing.size, (
            f"{missing.size} of the {forced.size} --include features do not pass the eligibility "
            f"rule; pass --allow-short to include them anyway and have it recorded"
        )
        assert n >= forced.size, f"--include lists {forced.size} features but --n is {n}"
        # Their stratification statistic has to be captured BEFORE they leave the pool, or the
        # stratum column below is computed from a table they are no longer in.
        stat_of = dict(zip(eligible.tolist(), rank_stat.tolist(), strict=True))
        forced_stat = np.array([stat_of[int(f)] for f in forced], dtype=np.float64)
        eligible_mask = ~np.isin(eligible, forced)
        eligible, rank_stat = eligible[eligible_mask], rank_stat[eligible_mask]
        n = n - forced.size
        print(f"[draw] --include forces {forced.size}; {n} sampled from the rest", flush=True)
    if stratified:
        drawn, stratum, cuts, pool_n = _stratified_draw(eligible, rank_stat, n, rng)
        if forced.size:
            # The forced ids get their stratum from the SAME cuts, so the column means one thing
            # across the whole set. `_stratified_draw` returned the cuts it used.
            f_stat = np.log10(np.maximum(forced_stat, 1e-6))
            drawn = np.concatenate([drawn, forced])
            stratum = np.concatenate([stratum, np.searchsorted(cuts, f_stat, side="right")])
            order = np.argsort(drawn)
            drawn, stratum = drawn[order], stratum[order]
        meta_extra = {"eligible": int(len(eligible)), "cuts_over": "eligible_pool",
                      "per_stratum": n // N_STRATA, "eligible_per_stratum": pool_n}
    else:
        drawn = np.sort(np.concatenate([forced, rng.choice(eligible, size=n, replace=False)]))

        # Quartile of the stratification statistic, over the DRAWN set.
        by_id = dict(zip(eligible.tolist(), rank_stat.tolist(), strict=True))
        dens = np.log10(np.maximum(np.array([by_id[int(f)] for f in drawn]), 1e-6))
        cuts = np.quantile(dens, [0.25, 0.5, 0.75])
        stratum = np.searchsorted(cuts, dens, side="right")
        meta_extra = {"eligible": int(len(eligible)), "cuts_over": "drawn_set"}
    assert np.isin(drawn, eval_ids).all(), "a drawn feature is not on the eval side"

    # `len(drawn)`, NOT `n`: the `--include` branch above SUBTRACTS the forced count from `n`
    # (that is how many are still to be sampled) and then concatenates the forced ids back into
    # `drawn`. Drawing the side column at `n` therefore made it SHORTER than the draw whenever
    # --include was used, and `_finish`'s `side[i]` walked off the end -- on exactly the
    # `--n 512 --stratified --include <64>` command eval 1 is built with. Found 2026-09-21.
    side = np.where(rng.random(len(drawn)) < FIT_FRACTION, "fit", "report")
    assert len(side) == len(drawn), f"{len(side)} side labels for {len(drawn)} drawn features"

    # The 16M peak is a column of a STRATIFIED draw only. The uniform draw's row schema is
    # frozen -- the 2k set on the volume and every consumer of it were written against it -- so
    # it gains no field, even on a root where max_act.f16 was just read for eligibility.
    peak16 = max_act if stratified else None
    peak_by_id = (None if peaks_1b is None
                  else dict(zip(eval_ids.tolist(), peaks_1b.tolist(), strict=True)))
    return _finish(cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
                   peak_by_id, strat_name, strat_source, have_stats, gated_full, peak16,
                   cuts, {**meta_extra, "n": int(len(drawn)), "seed": seed,
                          "stratified": stratified, "sides": sides})


SIDES = ("enc", "dec")


def _columns(path: str, d_model: int, drawn, sides):
    """(`{side: [n, d] unit fp32}`, gate, F) for `drawn`, from ONE read of the checkpoint.

    `common.load_sae` casts the WHOLE of `W_enc` (and, with `need_decoder`, `W_dec`) to fp32
    before anything is selected. At 2^21 features each matrix is 43 GB in fp32 on top of the
    ~86 GB the `torch.load` already holds, so asking for the decoder side through it roughly
    doubles a peak that is already the reason `need_decoder=False` exists. We need at most
    `len(drawn)` columns of each, so they are sliced in the checkpoint's own dtype and cast
    after -- peak stays at the load itself.

    BIT-IDENTICAL to the old path on the encoder side, and `unit_smoke.check_sae_column_reader`
    pins that against `common.load_sae` on a synthetic checkpoint: `load_sae` builds
    `W_enc = encoder.weight.to(fp32).T`, so `W_enc[:, drawn].T` IS `encoder.weight[drawn]` cast
    to fp32, which is what this returns. `decoder.weight` is `[d, F]` (nn.Linear stores
    `[out, in]` and the decoder maps F -> d), so its feature `f` is the COLUMN `[:, f]`.
    """
    import torch

    params = torch.load(path, map_location="cpu", weights_only=False)
    enc_w, dec_w = params.get("encoder.weight"), params.get("decoder.weight")
    assert enc_w is not None, f"SAE {path}: no encoder.weight (keys: {sorted(params)})"
    F, d = int(enc_w.shape[0]), int(enc_w.shape[1])
    assert d == d_model, f"SAE {path}: d_in {d} != base d_model {d_model}"
    assert F > int(max(drawn)), f"feature id {max(drawn)} outside F={F}"
    raw_thr = params.get("threshold")
    assert raw_thr is not None, (
        f"SAE {path} has no 'threshold' buffer; the fire gate would be undefined "
        f"(checkpoint keys: {sorted(params)})"
    )
    gate = float(raw_thr.item() if hasattr(raw_thr, "item") else raw_thr)
    assert gate > 0, f"SAE {path}: threshold {gate} must be > 0"

    idx = torch.as_tensor(np.asarray(drawn, dtype=np.int64))
    out = {}
    for sd in sides:
        if sd == "enc":
            v = enc_w[idx].to(torch.float32)
        else:
            assert dec_w is not None, (
                f"SAE {path}: --sides asks for 'dec' but the checkpoint has no decoder.weight "
                f"(keys: {sorted(params)})"
            )
            assert tuple(dec_w.shape) == (d, F), (
                f"SAE {path}: decoder.weight is {tuple(dec_w.shape)}, expected [d, F] = {(d, F)}"
            )
            v = dec_w[:, idx].T.contiguous().to(torch.float32)
        assert v.shape == (len(drawn), d), f"{sd}: {tuple(v.shape)} != {(len(drawn), d)}"
        out[sd] = torch.nn.functional.normalize(v, dim=-1)
    del params, enc_w, dec_w
    return out, gate, F


def _finish(cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
            peak_by_id, strat_name, strat_source, have_stats, gated_full, peak16, cuts,
            meta_extra):
    """One block of rows per requested SIDE of the dictionary, plus the pipeline's arrays.

    `peak_by_id` None means no 1.0B window table was read (the column is written as null rather
    than as a 0, which would read as "never fires"); `peak16` None means the row schema does not
    carry the 16M corpus peak, which is the uniform draw's frozen shape.

    With `--sides enc,dec` the SAME drawn features appear twice, in the same order, as two
    contiguous blocks distinguished by the row's `sae_side`. The pairing is what makes the
    encoder/decoder comparison exact; `common.sae_rows_of(..., side=)` is what selects one of
    them, and a row with no `sae_side` still reads as `enc` there, so the single-side shape is
    unchanged.
    """
    import torch

    sides = tuple(meta_extra.pop("sides"))
    cols, gate, d_sae = _columns(C.sae_path(cfg, sae_key), spec["d"], drawn, sides)
    vecs = torch.cat([cols[sd] for sd in sides], dim=0)

    rows = []
    for sd in sides:
        for i, fid in enumerate(drawn):
            row = {
                "row": len(rows),
                # `sae`, NOT `dict2m_enc` (fixed 2026-09-21). The family label is a SELECTOR, not a
                # description: precompute/scan.py, top1_act.py, repo_examples.py, gcg/gcg.py,
                # score.py's per-family means and autointerp's sae_self/build all filter
                # `family == "sae"`, so a set stamped with anything else is invisible to every one of
                # them -- which is why the first 2k draw had to be given read-time acceptance in each
                # consumer instead of simply working. WHICH dictionary a row belongs to is a separate
                # question and now has its own field.
                "family": "sae",
                # The config key of the SAE this feature index refers to. `id` alone is ambiguous
                # across dictionaries: feature 4242 of the 131k `l42-1b` and of the 2M `dict2m` are
                # unrelated directions, and before this field the only thing telling them apart was
                # the family label that nothing selected on.
                "sae_key": sae_key,
                # WHICH SIDE of the dictionary this row's vector is: `unit(W_enc[:, f])` or
                # `unit(W_dec[f])`. `common.sae_rows_of(..., side=)` selects on it and defaults a
                # row without it to "enc", so pre-2026-09-21 sets keep their meaning. It is NOT the
                # `side` field below, which is our fit/report analysis split.
                "sae_side": sd,
                "id": int(fid),
                "stratum": int(stratum[i]),
                "side": str(side[i]),
                "heldout_kind": "feature_id",
                "stratum_stat": strat_name,
                "gated_fires": int(gated_full[fid]) if have_stats else None,
                "corpus_peak_1b": (None if peak_by_id is None
                                   else float(peak_by_id.get(int(fid), float("nan")))),
            }
            if peak16 is not None:
                # The 16M corpus peak, which is the denominator every ratio in the activation
                # smokes is taken against (`sae/<sae>/max_act.f16`). `corpus_peak_1b` above is a
                # DIFFERENT quantity on a different corpus and the two are never interchangeable.
                row["corpus_peak_16m"] = round(float(peak16[fid]), 4)
            rows.append(row)
    meta = {
        # The dictionary these feature ids index. It is on every ROW too (`sae_key`), because a
        # row can outlive the directory it was written in; here so a reader of the README and of
        # the returned dict does not have to open ids.jsonl to find out.
        "sae_key": sae_key,
        # Per SIDE of the dictionary: the fit/report split is a property of the FEATURE, so a
        # two-side draw carries each label twice and the set's row count is len(sides) x n.
        "sides": list(sides),
        "d_sae": int(d_sae),
        "n_fit": int((side == "fit").sum() + (side == "train").sum()),
        "n_report": int((side == "report").sum() + (side == "test").sum()),
        "gate": gate,
        "stratum_stat": strat_name,
        "stratum_source": strat_source,
        "cuts": [float(c) for c in cuts],
        **meta_extra,
    }
    return set_name, out_dir, rows, vecs, meta


def run(cfg, args):
    set_name, out_dir, rows, vecs, meta = build(cfg, args)
    inputs = {"sae": args.get("sae"), "subset": args.get("subset") or "(drawn)"}
    with C.outdir(out_dir, args, inputs=inputs) as od:
        od.write_jsonl("ids.jsonl", rows)
        od.write_array("vecs.f16", vecs, "float16")
        # THE STORAGE CONTRACT (H4). Without it `common.set_storage` refuses the set outright and
        # somebody has to hand-write a `heldout:` entry -- which is exactly why 2026-09-20_dict2m_2k
        # needed one. `dirs_only`: these rows are unit encoder columns, never centred and not
        # centrable, so no `--mu` applies to them at all.
        od.write_json(
            "storage.json",
            {
                "storage": "dirs_only",
                "mu_stored": None,
                "family_mu": {},
                "families": {"sae": "dictionary"},
                "sae_key": meta["sae_key"],
                "sae_sides": meta["sides"],
                "note": (
                    "unit dictionary columns -- unit(W_enc[:, f]) on an `sae_side: enc` row, "
                    "unit(W_dec[f]) on a `dec` one: never centred, not centrable (config.yaml "
                    "family_kinds), so every --mu is a no-op on them"
                ),
            },
        )
        od.note(
            f"{len(rows)} targets, all from the eval split. `family` is **sae** -- the "
            f"label every consumer selects on (scan, top1_act, repo_examples, gcg, score's "
            f"per-family means, autointerp's sae_self and build) -- and the dictionary is named "
            f"per row in `sae_key` ({meta['sae_key']!r}), because a feature index means nothing "
            f"without "
            f"it. Sets drawn before 2026-09-21 carry `family: dict2m_enc` instead and are NOT "
            f"rewritten; autointerp still accepts that label for them."
        )
        od.note(f"strata: {meta['stratum_stat']} from {meta['stratum_source']}")
        if meta.get("stratified"):
            od.note(
                f"STRATIFIED draw, seed {meta['seed']}: {meta['per_stratum']} features from each "
                f"of the {N_STRATA} quartiles of the ELIGIBLE POOL ({meta['eligible']:,} "
                f"features), cuts on {meta['stratum_stat']} at "
                f"{[round(c, 4) for c in meta['cuts']]}, pool sizes "
                f"{meta['eligible_per_stratum']}. The cuts are the POOL's quartiles, not the "
                f"drawn set's: at an equal count per stratum the drawn set's quartiles are the "
                f"stratum boundaries by construction. Any mean over these rows is a mean over "
                f"four equally-weighted quartiles, NOT over the dictionary -- report per "
                f"stratum."
            )
        od.note(f"{meta['n_fit']} train/fit, {meta['n_report']} test/report -- BOTH "
                f"halves are unseen by the MAEM; this splits our analysis, not the "
                f"model's training")
        od.note(f"gate {meta['gate']}; F = {meta['d_sae']:,}; vecs are unit dictionary columns "
                f"in fp32 before the cast")
        if len(meta["sides"]) > 1:
            od.note(
                f"BOTH SIDES of the dictionary, {len(rows) // len(meta['sides'])} features x "
                f"{meta['sides']} = {len(rows)} rows, in that block order and PAIRED row for row "
                f"within a block: row i and row i + {len(rows) // len(meta['sides'])} are the "
                f"encoder and decoder column of the SAME feature. Select one with "
                f"`common.sae_rows_of(..., side=)`, which reads each row's `sae_side`. The "
                f"activation metric is the same on both (the activation of feature f is its "
                f"ENCODER readout whichever direction was injected); the `vecs.f16` cross-check "
                f"in sae_self does not apply to a decoder row."
            )
    print(json.dumps({"set": set_name, "dir": out_dir, **meta}, indent=1), flush=True)
    return {"product": "draw_dict2m", "set": set_name, **meta}
