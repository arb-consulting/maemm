"""The 2M-SAE feature registry: one row per feature, train/test side, rarity stratum.

    python -m features.registry --out /vol/base/qwen36-27b/sae/dict2m/features.parquet

Establishes, in one queryable table, which features the MAEM was trained on and which
it is evaluated on, with the per-feature statistics an eval needs to stratify.

Source of truth for the split is the `heldout/feature_split.parquet` (seed 2026).
This module does not re-draw it -- re-drawing would break the held-out claim, since the
simple2m checkpoints were trained against that exact partition. It only joins statistics
onto it and verifies the claim.

Two layers of statistic, and only the first is buildable today:

  layer 1 (here)   corpus_peak_1b + peak_stratum, from the original 1.0B-token scan.
                   No GPU, no SAE weights. Covers the 100,000 eval features.
  layer 2 (TODO)   fire_count_16m + density_stratum: fires above the gate on OUR 16M
                   held-out corpus, which is what config.yaml's `sae_strata: 4` means
                   ("rarity quartiles of the corpus fire count") and what the paper's
                   per-quartile SAE claims are cut on. BLOCKED: the 2M SAE's weights are
                   not in the bundle and no public repo carries them (checked
                   2026-09-19). Columns are emitted as null so the schema is stable.

The two strata are NOT interchangeable. Peak and density are different quantities, and
the 1.0B scan is a different corpus from our 16M one. Anything reported per quartile
must say which.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import bundle

N_FEATURES = 2_097_152
SPLIT_SEED = 2026                 # the upstream seed, recorded not re-drawn
EXPECTED_SPLITS = {"sft": 1_847_152, "rl": 150_000, "eval": 100_000}
N_STRATA = 4


def build() -> pd.DataFrame:
    """The registry. One row per SAE feature, keyed by feature_id."""
    reg = bundle.load_feature_split()[["feature_id", "split"]].copy()
    if len(reg) != N_FEATURES:
        raise ValueError(f"feature_split has {len(reg)} rows, expected {N_FEATURES}")
    counts = reg["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"split counts {counts} != {EXPECTED_SPLITS}")

    # Layer 1: corpus peak over the 1.0B-token scan. The rank-0 window's activation
    # IS the stored corpus_peak (verified exactly on all 512 standard-eval features), so
    # the windows file extends the peak to all 100k eval features for free.
    win = bundle.load_maxact_windows()
    peak = (win.loc[win["rank"] == 0, ["feature_id", "act"]]
               .rename(columns={"act": "corpus_peak_1b"}))
    reg = reg.merge(peak, on="feature_id", how="left")

    std = set(bundle.load_sae_features()["feature_id"])
    reg["is_standard_eval"] = reg["feature_id"].isin(std)

    # Quartiles of log10 peak, computed WITHIN the eval split only -- the train features
    # have no peak in the bundle, so a global quantile would be meaningless.
    reg["peak_stratum"] = pd.NA
    ev = reg["split"] == "eval"
    reg.loc[ev, "peak_stratum"] = pd.qcut(
        np.log10(reg.loc[ev, "corpus_peak_1b"].to_numpy()),
        N_STRATA, labels=False, duplicates="drop",
    )
    reg["peak_stratum"] = reg["peak_stratum"].astype("Int8")

    # Layer 2, pending the SAE weights. Declared so downstream schemas do not change.
    reg["fire_count_16m"] = pd.Series(pd.NA, index=reg.index, dtype="Int64")
    reg["density_stratum"] = pd.Series(pd.NA, index=reg.index, dtype="Int8")

    return reg[["feature_id", "split", "is_standard_eval", "corpus_peak_1b",
                "peak_stratum", "fire_count_16m", "density_stratum"]]


def verify(reg: pd.DataFrame) -> dict:
    """Re-derive the held-out claim rather than trusting the bundle README."""
    ev = reg["split"] == "eval"
    eval_ids = set(reg.loc[ev, "feature_id"])
    out = {
        "n_features": int(len(reg)),
        "split_counts": reg["split"].value_counts().to_dict(),
        "split_seed": SPLIT_SEED,
        "n_standard_eval": int(reg["is_standard_eval"].sum()),
        "standard_eval_within_eval_split": bool(
            set(reg.loc[reg["is_standard_eval"], "feature_id"]) <= eval_ids),
        "eval_features_with_peak": int(reg.loc[ev, "corpus_peak_1b"].notna().sum()),
        "train_features_with_peak": int(reg.loc[~ev, "corpus_peak_1b"].notna().sum()),
    }
    # Every direction family drawn from the 2M SAE must live on the eval side.
    for fam in ("dict2m_enc", "dict2m_dec"):
        ids = bundle.load_family(fam).feature_ids
        ids = np.asarray(ids).ravel()
        out[f"{fam}_in_eval_split"] = bool(set(ids.tolist()) <= eval_ids)
    peaks = reg.loc[ev].groupby("peak_stratum", observed=True)["corpus_peak_1b"]
    out["peak_strata"] = {int(k): {"n": int(v), "min": round(float(lo), 3),
                                   "max": round(float(hi), 3)}
                          for k, v, lo, hi in zip(peaks.size().index, peaks.size(),
                                                  peaks.min(), peaks.max())}
    out["layer2_pending"] = "fire_count_16m / density_stratum need the 2M SAE weights"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="features_2m.parquet")
    ap.add_argument("--report", default="features_2m.report.json")
    args = ap.parse_args()

    reg = build()
    rep = verify(reg)
    if not all(rep[k] for k in ("standard_eval_within_eval_split",
                                "dict2m_enc_in_eval_split", "dict2m_dec_in_eval_split")):
        raise SystemExit(f"held-out claim does not verify: {json.dumps(rep, indent=1)}")

    reg.to_parquet(args.out, index=False)
    with open(args.report, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    print(f"\nwrote {args.out} ({len(reg):,} rows) and {args.report}")


if __name__ == "__main__":
    main()
