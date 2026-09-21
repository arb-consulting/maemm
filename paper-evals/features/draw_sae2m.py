"""Draw the standard sae2m target set: 40,000 eval-split features, 80/20.

    python -m features.spawn --product draw_sae2m --base qwen36-27b \
        --sae qwen36-27b/sae2m --gpu cpu

Writes the shape the rest of the pipeline already reads:

    <root>/base/<base>/heldout/<set>/
        ids.jsonl   one row per target: row, family ("sae"), sae_key (which dictionary),
                    id (feature id), stratum, side, ...
        vecs.f16    [N, d] unit rows, row i is ids.jsonl line i
        README.md   the draw, the eligibility rule, and how the split is held out

Every feature comes from Celeste's `eval` split (seed 2026, 100,000 features). That is
load-bearing: the checkpoint was TRAINED on the `sft` (1,847,152) and `rl` (150,000)
sides, so a draw over all 2^21 features would put trained-on features in the test set
and the held-out claim would die silently. `build()` asserts every drawn id is on the
eval side.

The 80/20 `side` column splits OUR analysis, not the model's training: both halves are
equally unseen by the MAEMM. It exists so thresholds, strata cuts and ablations can be
chosen on `fit` without touching the numbers reported from `report`.

Stratification is RECORDED, not sampled: the draw is uniform over eligible features and
each row carries its fire-count quartile. A stratified draw would force equal quartiles
and make the overall mean unrepresentative of the dictionary.

FAMILY LABEL (fixed 2026-09-21). These rows are stamped `family: "sae"`, the label every
downstream product filters on, with the dictionary named separately per row in `sae_key`.
The first draw used `family: "sae2m_enc"`, which no consumer selected on, so the set was
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
    """Feature ids on Celeste's eval side, read from the bundle's own split file."""
    import pandas as pd

    path = args.get("feature_split") or (
        f"{args['root']}/data/celeste-v2-2026-09-17/heldout/feature_split.parquet")
    split = pd.read_parquet(path)
    ids = split.loc[split["split"] == "eval", "feature_id"].to_numpy()
    assert len(ids) == 100_000, f"eval split has {len(ids)} features, expected 100,000"
    return np.sort(ids)


def _bundle_corpus_peaks(args, eval_ids):
    """Corpus peak per eval feature from Celeste's 1.0B scan, aligned to `eval_ids`.

    The rank-0 window activation IS the shipped corpus_peak -- verified exactly on all
    512 standard-eval features (max abs diff 0.000000).
    """
    import pandas as pd

    path = args.get("maxact_windows") or (
        f"{args['root']}/data/celeste-v2-2026-09-17/heldout/"
        f"eval_2m_features_100k_windows.parquet")
    win = pd.read_parquet(path, columns=["feature_id", "rank", "act"])
    top = win[win["rank"] == 0].set_index("feature_id")["act"]
    return top.reindex(eval_ids).to_numpy(dtype=np.float32)


def _include_ids(args) -> np.ndarray:
    """`--include <file>`: feature ids that MUST be in the draw, whatever the sample picks.

    The eval plan needs her standard-eval 512 inside our stratified draw so the two blocks are
    nested rather than merely comparable; `--stratified`/`--seed` cannot express that, and
    `:93`'s max-abs-diff check against her 512 is a verification, not an inclusion mechanism.
    Accepts a .parquet with a `feature_id` column, a .npy, or one id per line.
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
        "draw_sae2m writes a held-out set, so it needs an explicit --set <name> (D6). It used to "
        "fall back to today's date, and through modal_app to the LIVE default set."
    )
    out_dir = C.heldout_dir(base, set_name, root)
    # Fail before the SAE load, as every other product does; C.outdir enforces the same rule again
    # at write time (temp-and-rename, no half-written set).
    assert args.get("force") or not os.path.exists(out_dir), (
        f"{out_dir} already exists; refusing to overwrite without --force")

    # Eligibility and strata come from our own corpus scan when it exists, and from
    # Celeste's 1.0B-token scan when it does not. The fallback is not a degraded mode:
    # her scan is 60x larger than ours, reports 0 dead features over all 2^21, and its
    # rank-0 window activation reproduces the shipped corpus_peak of the standard-eval
    # 512 EXACTLY (max abs diff 0.000000, features/registry.py). So every eval feature
    # is already known to fire, and the >= MIN_FIRES gate is satisfied before we compute
    # anything. Fire counts on OUR corpus are a better stratification axis and get
    # joined in as a column when the stats pass lands -- they are not a prerequisite.
    sae_stats = C.sae_dir(sae_key, root)
    have_stats = os.path.exists(f"{sae_stats}/sizes.json")
    if have_stats:
        with open(f"{sae_stats}/sizes.json") as fh:
            sizes_meta = json.load(fh)
        n_sizes, f_sae = len(sizes_meta["sizes"]), sizes_meta["d_sae"]
        fires = C.read_array(f"{sae_stats}/fire_counts.i64", "int64", (f_sae, n_sizes, 2))
        gated_full = fires[:, -1, 1]
        max_act = C.read_array(f"{sae_stats}/max_act.f16", "float16",
                               (f_sae,)).astype(np.float32)
        assert sizes_meta["threshold"] > 0, "stats recorded no gate"
        strat_name, strat_source = "log10_gated_fires_16M", "our 16M corpus scan"
    else:
        gated_full, max_act = None, None
        strat_name, strat_source = "log10_corpus_peak_1B", "Celeste's 1.0B-token scan"
        print("[draw] no stats pass for this SAE yet; using the bundle's corpus peaks "
              "for eligibility and strata", flush=True)
    eval_ids = _eval_split_ids(cfg, args)

    # An explicit subset (features.parquet from shared/<name>/) takes the draw as given:
    # the point of a shared subset is that everyone gets the SAME features, so nothing
    # is re-sampled here and the split/stratum columns are carried through verbatim.
    subset = args.get("subset")
    if subset:
        import pandas as pd

        sub = pd.read_parquet(subset)
        drawn = sub["feature_id"].to_numpy()
        assert np.isin(drawn, eval_ids).all(), (
            "a subset feature is not on the eval side -- it would not be held out")
        side = sub["split"].to_numpy().astype(str)
        stratum = sub["stratum"].to_numpy().astype(int)
        peak_by_id = dict(
            zip(sub["feature_id"].tolist(), sub["corpus_peak_1b"].tolist(), strict=True)
        )
        strat_name, strat_source = "log10_corpus_peak_1B", f"subset {subset}"
        have_stats = False
        gated_full = None
        meta_extra = {"subset": subset, "eligible": int(len(sub))}
        cuts = []
        return _finish(cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
                       peak_by_id, strat_name, strat_source, have_stats, gated_full,
                       cuts, meta_extra)

    peaks_1b = _bundle_corpus_peaks(args, eval_ids)
    if have_stats:
        eligible = eval_ids[(gated_full[eval_ids] >= MIN_FIRES) & (max_act[eval_ids] > 0)]
        rank_stat = gated_full[eligible].astype(np.float64)
    else:
        keep = peaks_1b > 0
        eligible = eval_ids[keep]
        rank_stat = peaks_1b[keep].astype(np.float64)
    print(f"[draw] eval split {len(eval_ids):,} -> {len(eligible):,} eligible, "
          f"strata from {strat_source}", flush=True)
    assert len(eligible) >= int(args.get("n") or N_FEATURES), (
        f"only {len(eligible)} eligible features, need {int(args.get('n') or N_FEATURES)}")

    rng = np.random.default_rng(int(args.get("seed") or DRAW_SEED))
    n_draw = int(args.get("n") or N_FEATURES)
    forced = _include_ids(args)
    if forced.size:
        # The forced ids are taken as given and the SAMPLE fills the rest, drawn from the eligible
        # pool with the forced ones removed so nothing is drawn twice. They must still be on the
        # eval side, or the held-out claim dies for those rows.
        assert np.isin(forced, eval_ids).all(), (
            f"{int((~np.isin(forced, eval_ids)).sum())} of the {forced.size} --include features are "
            f"not on Celeste's eval split, so they were TRAINED on and cannot be held out"
        )
        missing = forced[~np.isin(forced, eligible)]
        assert args.get("allow_short") or not missing.size, (
            f"{missing.size} of the {forced.size} --include features do not pass the eligibility "
            f"rule (>= {MIN_FIRES} gated fires); pass --allow-short to include them anyway and "
            f"have the shortfall recorded"
        )
        rest = eligible[~np.isin(eligible, forced)]
        assert rest.size >= n_draw - forced.size, (
            f"only {rest.size} eligible features outside the {forced.size} forced ones, need "
            f"{n_draw - forced.size} more to reach n={n_draw}"
        )
        drawn = np.sort(np.concatenate([forced, rng.choice(rest, n_draw - forced.size, replace=False)]))
        print(f"[draw] --include forced {forced.size} features; {n_draw - forced.size} sampled", flush=True)
    else:
        drawn = np.sort(rng.choice(eligible, size=n_draw, replace=False))
    assert np.isin(drawn, eval_ids).all(), "a drawn feature is not on the eval side"

    # Quartile of the stratification statistic, over the DRAWN set.
    by_id = dict(zip(eligible.tolist(), rank_stat.tolist(), strict=True))
    dens = np.log10(np.maximum(np.array([by_id[int(f)] for f in drawn]), 1e-6))
    cuts = np.quantile(dens, [0.25, 0.5, 0.75])
    stratum = np.searchsorted(cuts, dens, side="right")

    side = np.where(rng.random(len(drawn)) < FIT_FRACTION, "fit", "report")

    peak_by_id = dict(zip(eval_ids.tolist(), peaks_1b.tolist(), strict=True))
    return _finish(cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
                   peak_by_id, strat_name, strat_source, have_stats, gated_full,
                   cuts, {"eligible": int(len(eligible))})


def _finish(cfg, args, sae_key, spec, set_name, out_dir, drawn, side, stratum,
            peak_by_id, strat_name, strat_source, have_stats, gated_full, cuts,
            meta_extra):
    """Encoder columns for `drawn`, plus the rows the pipeline reads."""
    import torch

    sae = C.load_sae(C.sae_path(cfg, sae_key), spec["d"], device="cpu",
                     dtype=torch.float32, need_decoder=False)
    assert sae.d_sae > int(max(drawn)), f"feature id {max(drawn)} outside F={sae.d_sae}"
    vecs = torch.nn.functional.normalize(
        sae.W_enc[:, torch.as_tensor(np.asarray(drawn))].T.contiguous(), dim=-1)

    rows = []
    for i, fid in enumerate(drawn):
        rows.append({
            "row": i,
            # `sae`, NOT `sae2m_enc` (fixed 2026-09-21). The family label is a SELECTOR, not a
            # description: precompute/scan.py, top1_act.py, repo_examples.py, gcg/gcg.py,
            # score.py's per-family means and autointerp's sae_self/build all filter
            # `family == "sae"`, so a set stamped with anything else is invisible to every one of
            # them -- which is why the first 2k draw had to be given read-time acceptance in each
            # consumer instead of simply working. WHICH dictionary a row belongs to is a separate
            # question and now has its own field.
            "family": "sae",
            # The config key of the SAE this feature index refers to. `id` alone is ambiguous
            # across dictionaries: feature 4242 of the 131k `l42-1b` and of the 2M `sae2m` are
            # unrelated directions, and before this field the only thing telling them apart was
            # the family label that nothing selected on.
            "sae_key": sae_key,
            "id": int(fid),
            "stratum": int(stratum[i]),
            "side": str(side[i]),
            "heldout_kind": "feature_id",
            "stratum_stat": strat_name,
            "gated_fires": int(gated_full[fid]) if have_stats else None,
            "corpus_peak_1b": float(peak_by_id.get(int(fid), float("nan"))),
        })
    meta = {
        # The dictionary these feature ids index. It is on every ROW too (`sae_key`), because a
        # row can outlive the directory it was written in; here so a reader of the README and of
        # the returned dict does not have to open ids.jsonl to find out.
        "sae_key": sae_key,
        "n_fit": int((side == "fit").sum() + (side == "train").sum()),
        "n_report": int((side == "report").sum() + (side == "test").sum()),
        "gate": float(sae.threshold),
        "stratum_stat": strat_name,
        "stratum_source": strat_source,
        "cuts": [float(c) for c in cuts],
        **meta_extra,
    }
    return set_name, out_dir, rows, vecs, meta


def run(cfg, args):
    set_name, out_dir, rows, vecs, meta = build(cfg, args)
    inputs = {
        "sae": meta["sae_key"],
        "subset": args.get("subset") or "(drawn)",
        "include": args.get("include") or "(none)",
        "n": len(rows),
    }
    with C.outdir(out_dir, args, inputs=inputs) as od:
        od.write_jsonl("ids.jsonl", rows)
        od.write_array("vecs.f16", vecs, "float16")
        # THE STORAGE CONTRACT (H4). Without it `common.set_storage` refuses the set outright and
        # somebody has to hand-write a `heldout:` entry -- which is exactly why 2026-09-20_sae2m_2k
        # needed one. `dirs_only`: these rows are unit encoder columns, they were never centred and
        # there is nothing to centre them on, so no `--mu` applies to them at all.
        od.write_json(
            "storage.json",
            {
                "storage": "dirs_only",
                "mu_stored": None,
                "family_mu": {},
                "families": {"sae": "dictionary"},
                "sae_key": meta["sae_key"],
                "note": (
                    "unit(W_enc[:, f]) encoder columns: never centred, not centrable "
                    "(config.yaml family_kinds), so every --mu is a no-op on them"
                ),
            },
        )
        od.note(
            f"{len(rows)} targets, all from Celeste's eval split. `family` is **sae** -- the "
            f"label every consumer selects on (scan, top1_act, repo_examples, gcg, score's "
            f"per-family means, autointerp's sae_self and build) -- and the dictionary is named "
            f"per row in `sae_key` ({meta['sae_key']!r}), because a feature index means nothing "
            f"without "
            f"it. Sets drawn before 2026-09-21 carry `family: sae2m_enc` instead and are NOT "
            f"rewritten; autointerp still accepts that label for them."
        )
        od.note(f"strata: {meta['stratum_stat']} from {meta['stratum_source']}")
        od.note(f"{meta['n_fit']} train/fit, {meta['n_report']} test/report -- BOTH "
                f"halves are unseen by the MAEMM; this splits our analysis, not the "
                f"model's training")
        od.note(f"gate {meta['gate']}; vecs are unit(W_enc[:, f]) in fp32 before the cast")
    print(json.dumps({"set": set_name, "dir": out_dir, **meta}, indent=1), flush=True)
    return {"product": "draw_sae2m", "set": set_name, **meta}
