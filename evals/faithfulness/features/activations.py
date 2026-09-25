"""The activation-target registry: provenance and split side for the non-SAE families.

    python -m features.activations --out activations_v2.parquet

The SAE families are blocked on weights we do not have; the activation families are not,
and they are also the cleaner half (the branch's 13-gram check puts realact content
overlap with the upstream training text at 0.50% of rows against dict2m's 8.31%).

What this records, per eval target: which document and token position it was read at,
its raw residual norm, and the document-level split side. What it does NOT do is
re-draw anything -- the eval directions are frozen in the bundle and the checkpoints
were trained against that draw.

Three properties of the v2 draw that downstream statistics must respect. All three are
measured here rather than assumed; see the README for what to do about each.

  1. The directions are ALREADY CENTRED. Median pairwise cosine within realact is
     -0.004 and the mean direction has norm 0.06 -- indistinguishable from `random`.
     Raw activations share a large common component (||mu|| ~ 68 against activation
     norms ~ 90), so these are unit(act - mu) with the upstream whiten_mu, not raw. A
     scorer that centres again is centring twice.
  2. TARGETS ARE NOT INDEPENDENT. 512 realact targets come from 449 documents: 59
     documents carry 2-3 targets, so 122 targets share a document with another. An
     error bar over 512 targets that treats them as independent is too small; cluster
     by `doc` instead.
  3. realact_early / realact_mid / realact_long ARE the realact parquet's own
     early_dirs / mid_dirs / long_dirs columns (verified identical, 512/512) -- but
     they are near-orthogonal to realact itself (median cosine -0.007), so they are
     DIFFERENT activations, not the same one read at another context length. They
     carry no provenance columns of their own, so their documents, positions and
     independence structure are unknown from the bundle.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import bundle

# Document ranges of the Ultra-FineWeb en stream, from the bundle's doc_registry and
# verified doc-disjoint (0 intersection) by the branch's check on 2026-09-18.
DOC_RANGES = {
    "eval":      (0, 100_000),            # the eval pool's realact activations
    "sae_train": (100_000, 1_783_561),    # 2M-SAE dictionary training stream
    "sft":       (5_500_000, 5_698_524),  # sft_activations_ctx8_64
    "rl":        (9_500_000, 9_599_842),  # rl_activations_ctx64_2048
}

# Families with no SAE dependency. `provenance` = the bundle ships doc/pos/norm for it.
ACTIVATION_FAMILIES = {
    "realact":        {"provenance": True,  "note": "512-token windows, span pos ~ U[16,T)"},
    "realact_early":  {"provenance": False, "note": "realact.early_dirs; no provenance"},
    "realact_mid":    {"provenance": False, "note": "realact.mid_dirs; no provenance"},
    "realact_long":   {"provenance": False, "note": "realact.long_dirs; no provenance"},
    "random":         {"provenance": False, "note": "Gaussian control, no corpus origin"},
    "indist_realact": {"provenance": False, "note": "in-distribution variant"},
    "indist_long":    {"provenance": False, "note": "in-distribution variant"},
    "indist_probe":   {"provenance": False, "note": "in-distribution variant"},
    "cluster":        {"provenance": False, "note": "probe-cluster directions"},
    "jlens":          {"provenance": False, "note": "J-lens token directions"},
    "bsf":            {"provenance": False, "note": "block-sparse featurizer subspace"},
}

N_STRATA = 4


def _split_for_doc(doc: int) -> str:
    for name, (lo, hi) in DOC_RANGES.items():
        if lo <= doc < hi:
            return name
    return "outside_known_ranges"


def build() -> pd.DataFrame:
    """One row per eval target across the activation families."""
    rows = []
    for fam, spec in ACTIVATION_FAMILIES.items():
        t = bundle.load_family(fam)
        m = t.meta
        df = pd.DataFrame({"family": fam, "row": np.arange(len(t), dtype=np.int32)})
        if spec["provenance"] and "pool_seq" in m.columns:
            df["doc"] = m["pool_seq"].to_numpy()
            df["pos"] = m["pool_pos"].to_numpy()
            df["act_norm"] = m["pool_act_norm"].to_numpy(dtype=np.float32)
            df["split"] = [_split_for_doc(int(d)) for d in df["doc"]]
            # Targets from the same document are one cluster for error bars.
            df["doc_n_targets"] = df.groupby("doc")["row"].transform("size").astype("int16")
            df["norm_stratum"] = pd.qcut(df["act_norm"], N_STRATA,
                                         labels=False, duplicates="drop")
        else:
            for c, dt in (("doc", "Int64"), ("pos", "Int64"),
                          ("doc_n_targets", "Int16"), ("norm_stratum", "Int8")):
                df[c] = pd.Series(pd.NA, index=df.index, dtype=dt)
            df["act_norm"] = np.nan
            df["split"] = pd.NA
        # Centring is a property of the draw, measured not assumed (see verify()).
        df["shared_component"] = float(np.linalg.norm(t.directions.mean(0)))
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out[["family", "row", "doc", "pos", "act_norm", "norm_stratum",
                "doc_n_targets", "split", "shared_component"]]


def verify(reg: pd.DataFrame) -> dict:
    """Measure the three properties the docstring claims, on this draw."""
    out = {"families": {}, "doc_ranges": {k: list(v) for k, v in DOC_RANGES.items()}}
    for fam in ACTIVATION_FAMILIES:
        d = bundle.load_family(fam).directions
        C = d @ d.T
        iu = np.triu_indices(len(d), 1)
        sub = reg[reg["family"] == fam]
        cell = {
            "n": int(len(sub)),
            "median_pairwise_cos": round(float(np.median(C[iu])), 4),
            "shared_component_norm": round(float(np.linalg.norm(d.mean(0))), 4),
            "has_provenance": bool(sub["doc"].notna().any()),
        }
        if cell["has_provenance"]:
            docs = sub["doc"].dropna().astype(int)
            cell["n_documents"] = int(docs.nunique())
            cell["targets_sharing_a_document"] = int((sub["doc_n_targets"] > 1).sum())
            cell["max_targets_per_document"] = int(sub["doc_n_targets"].max())
            cell["splits"] = sub["split"].value_counts().to_dict()
        out["families"][fam] = cell

    ref = out["families"]["random"]["shared_component_norm"]
    ra = out["families"]["realact"]["shared_component_norm"]
    out["centred"] = {
        "realact_shared_component": ra,
        "random_shared_component": ref,
        "verdict": ("pre-centred: realact's shared component matches the Gaussian control, "
                    "so the directions are unit(act - mu). Do not centre again."
                    if ra < 3 * ref else "NOT pre-centred -- check before scoring"),
        "mu_in_bundle": False,
    }
    # Claim 3: the context families are the realact parquet's own columns.
    m = bundle.load_family("realact").meta
    base = bundle.load_family("realact").directions
    paired = {}
    for fam, col in (("realact_early", "early_dirs"), ("realact_mid", "mid_dirs"),
                     ("realact_long", "long_dirs")):
        e = np.stack(m[col].to_numpy()).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        f = bundle.load_family(fam).directions
        paired[fam] = {
            "identical_to_realact_column": int(((f * e).sum(1) > 0.9999).sum()),
            "median_cos_to_realact": round(float(np.median((base * e).sum(1))), 4),
        }
    out["context_families"] = paired
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="activations_v2.parquet")
    ap.add_argument("--report", default="activations_v2.report.json")
    args = ap.parse_args()
    reg = build()
    rep = verify(reg)
    reg.to_parquet(args.out, index=False)
    with open(args.report, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    print(f"\nwrote {args.out} ({len(reg):,} rows) and {args.report}")


if __name__ == "__main__":
    main()
