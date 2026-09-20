"""Draw the standard sae2m target set: 40,000 eval-split features, 80/20.

    python -m features.spawn --product draw_sae2m --base qwen36-27b \
        --sae qwen36-27b/sae2m --gpu cpu

Writes the shape the rest of the pipeline already reads:

    <root>/base/<base>/heldout/<set>/
        ids.jsonl   one row per target: row, family, id (feature id), stratum, side, ...
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
"""
from __future__ import annotations

import json
import os
import time

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


def build(cfg, args):
    import torch

    base, root = args["base"], args["root"]
    sae_key = args["sae"] if "/" in args.get("sae", "") else f"{base}/{args.get('sae')}"
    spec = cfg["bases"][base]
    set_name = args["heldout"] or time.strftime("%Y-%m-%d") + "_sae2m"
    out_dir = C.heldout_dir(base, set_name, root)
    assert args.get("force") or not os.path.exists(out_dir), (
        f"{out_dir} already exists; refusing to overwrite without --force")

    # Fire counts from the stats pass: [F, n_sizes, 2], last axis (> 0, > gate).
    sae_stats = C.sae_dir(sae_key, root)
    with open(f"{sae_stats}/sizes.json") as fh:
        sizes_meta = json.load(fh)
    n_sizes, f_sae = len(sizes_meta["sizes"]), sizes_meta["d_sae"]
    fires = C.read_array(f"{sae_stats}/fire_counts.i64", "int64", (f_sae, n_sizes, 2))
    max_act = C.read_array(f"{sae_stats}/max_act.f16", "float16", (f_sae,)).astype(np.float32)
    assert sizes_meta["threshold"] > 0, "stats recorded no gate"
    gated_full = fires[:, -1, 1]          # gated fires at the FULL corpus size
    print(f"[draw] fire counts {fires.shape}, gate-fires median {np.median(gated_full):.0f}",
          flush=True)

    eval_ids = _eval_split_ids(cfg, args)
    eligible = eval_ids[(gated_full[eval_ids] >= MIN_FIRES) & (max_act[eval_ids] > 0)]
    print(f"[draw] eval split 100,000 -> {len(eligible)} eligible "
          f"(>= {MIN_FIRES} gated fires and a positive max)", flush=True)
    assert len(eligible) >= N_FEATURES, (
        f"only {len(eligible)} eligible features, need {N_FEATURES}")

    rng = np.random.default_rng(DRAW_SEED)
    drawn = np.sort(rng.choice(eligible, size=N_FEATURES, replace=False))
    assert np.isin(drawn, eval_ids).all(), "a drawn feature is not on the eval side"

    # Quartile of log10 gated fire density, over the DRAWN set.
    dens = np.log10(np.maximum(gated_full[drawn], 1).astype(np.float64))
    cuts = np.quantile(dens, [0.25, 0.5, 0.75])
    stratum = np.searchsorted(cuts, dens, side="right")

    side = np.where(rng.random(N_FEATURES) < FIT_FRACTION, "fit", "report")

    # Directions: the unit encoder column, the same object targets.py uses for `sae`.
    sae = C.load_sae(C.sae_path(cfg, sae_key), spec["d"], device="cpu",
                     dtype=torch.float32, need_decoder=False)
    assert sae.d_sae > drawn.max(), f"feature id {drawn.max()} outside F={sae.d_sae}"
    vecs = torch.nn.functional.normalize(
        sae.W_enc[:, torch.as_tensor(drawn)].T.contiguous(), dim=-1)

    rows = []
    for i, fid in enumerate(drawn):
        rows.append({
            "row": i,
            "family": "sae2m_enc",
            "id": int(fid),
            "stratum": int(stratum[i]),
            "side": str(side[i]),
            "heldout_kind": "feature_id",
            "gated_fires": int(gated_full[fid]),
            "corpus_max_act": float(max_act[fid]),
        })
    return set_name, out_dir, rows, vecs, {
        "eligible": int(len(eligible)),
        "cuts_log10_gated_fires": [float(c) for c in cuts],
        "n_fit": int((side == "fit").sum()),
        "n_report": int((side == "report").sum()),
        "gate": float(sae.threshold),
    }


def run(cfg, args):
    import torch

    set_name, out_dir, rows, vecs, meta = build(cfg, args)
    inputs = {"sae": args.get("sae"), "feature_split": "celeste-v2-2026-09-17 (seed 2026)"}
    with C.outdir(out_dir, args, inputs=inputs) as od:
        od.write_jsonl("ids.jsonl", rows)
        od.write_array("vecs.f16", vecs, "float16")
        od.note(f"{len(rows)} sae2m_enc targets, all from Celeste's eval split")
        od.note(f"eligibility: >= {MIN_FIRES} gated fires on the full corpus and a positive "
                f"max; {meta['eligible']} of 100,000 eval features qualified")
        od.note(f"draw: uniform over eligible, seed {DRAW_SEED}; strata are RECORDED "
                f"(log10 gated-fire quartiles, cuts {meta['cuts_log10_gated_fires']}), "
                f"not sampled")
        od.note(f"side: {meta['n_fit']} fit / {meta['n_report']} report at "
                f"{FIT_FRACTION:.0%}. BOTH halves are unseen by the MAEMM -- this splits "
                f"our analysis, not the model's training")
        od.note(f"gate {meta['gate']}; vecs are unit(W_enc[:, f]) in fp32 before the cast")
    print(json.dumps({"set": set_name, "dir": out_dir, **meta}, indent=1), flush=True)
    return {"product": "draw_sae2m", "set": set_name, **meta}
