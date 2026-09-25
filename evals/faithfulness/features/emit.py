"""Lay the registries out as <family>/{README.md,train/,test/}.

    python -m features.emit --root /vol/base/qwen36-27b/registry

Builds both registries once and writes them into the family layout of layout.py. The
train side is what the checkpoint was fitted on, the test side is what it is scored on;
the head README states the relation, because that relation is the whole claim.

Nothing here re-draws anything. Both sides are read off the v2 bundle: the feature
partition from feature_split.parquet, the activation document ranges from
doc_registry.json and the exact per-source lists in heldout/doc_ids/.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import activations, bundle, layout, registry

# Which bundle doc_ids lists make up the training side of each activation family.
ACTIVATION_TRAIN_SOURCES = {
    "realact": ["sft_activations_ctx8_64", "rl_activations_ctx64_2048"],
}
# ... and of each SAE family.
SAE_TRAIN_SOURCES = ["dict2m_sft_bank_windows", "dict2m_rl_bank_windows"]

SAE_FAMILIES = ("dict2m_enc", "dict2m_dec")


def _doc_ids(source: str) -> pd.DataFrame:
    return pd.read_parquet(bundle.fetch(f"heldout/doc_ids/{source}.parquet"))


def _head_readme(family: str, train: dict, test: dict, extra: str = "") -> str:
    kind = layout.heldout_kind(family)
    lines = [
        f"# {family}",
        "",
        f"Snapshot `{bundle.SNAPSHOT}`. Directions live in the raw layer-42 residual "
        f"space of `{bundle.MODEL}` (d = {bundle.D_MODEL}).",
        "",
        f"**Held out by: `{kind}`** — {layout.KIND_MEANING[kind]}",
        "",
        "## train/",
        "",
        f"What the checkpoint `{bundle.CHECKPOINT}` was fitted on for this family.",
        "",
        "```",
        json.dumps(train, indent=1),
        "```",
        "",
        "## test/",
        "",
        "The frozen eval targets. Never trained on, by the partition above.",
        "",
        "```",
        json.dumps(test, indent=1),
        "```",
    ]
    if extra:
        lines += ["", "## Notes", "", extra]
    return "\n".join(lines) + "\n"


def emit_activations(root: str, reg: pd.DataFrame, rep: dict) -> list[str]:
    written = []
    for family in activations.ACTIVATION_FAMILIES:
        sub = reg[reg["family"] == family].reset_index(drop=True)
        sub.to_parquet(layout.path(root, family, "test", "targets.parquet"), index=False)

        train_info: dict = {}
        for source in ACTIVATION_TRAIN_SOURCES.get(family, []):
            docs = _doc_ids(source)
            docs.to_parquet(layout.path(root, family, "train", f"{source}.parquet"),
                            index=False)
            train_info[source] = {
                "documents": int(docs["doc_idx"].nunique()),
                "doc_idx_min": int(docs["doc_idx"].min()),
                "doc_idx_max": int(docs["doc_idx"].max()),
            }
        if not train_info:
            train_info = {"status": layout.NO_TRAIN_SIDE.get(
                family, "no training-side document list is shipped for this family")}

        cell = rep["families"][family]
        test_info = {"targets": cell["n"], "has_provenance": cell["has_provenance"]}
        if cell["has_provenance"]:
            test_info |= {
                "documents": cell["n_documents"],
                "targets_sharing_a_document": cell["targets_sharing_a_document"],
                "doc_idx_range": list(activations.DOC_RANGES["eval"]),
            }
        extra = ""
        if not cell["has_provenance"]:
            extra = ("The bundle ships no document, position or norm for this family, so "
                     "its training-side documents and its independence structure are "
                     "unknown. Treat any held-out claim about it as inherited, not checked.")
        elif cell["targets_sharing_a_document"]:
            extra = (f"{cell['targets_sharing_a_document']} of {cell['n']} targets share a "
                     f"document with another target ({cell['n_documents']} distinct "
                     f"documents, up to {cell['max_targets_per_document']} each). Cluster "
                     f"bootstraps by `doc`; an SE over {cell['n']} independent targets is "
                     f"too small.")
        layout.write_readme(root, family, _head_readme(family, train_info, test_info, extra),
                            train_note=train_info.get("status"))
        written.append(family)
    return written


def emit_features(root: str, reg: pd.DataFrame, rep: dict) -> list[str]:
    train = reg[reg["split"].isin(["sft", "rl"])]
    test = reg[reg["split"] == "eval"]
    written = []
    for family in SAE_FAMILIES:
        train.to_parquet(layout.path(root, family, "train", "features.parquet"), index=False)
        test.to_parquet(layout.path(root, family, "test", "features.parquet"), index=False)
        for source in SAE_TRAIN_SOURCES:
            docs = _doc_ids(source)
            docs.to_parquet(layout.path(root, family, "train", f"{source}.parquet"),
                            index=False)
        train_info = {
            "features": int(len(train)),
            "by_split": train["split"].value_counts().to_dict(),
            "split_seed": registry.SPLIT_SEED,
        }
        test_info = {
            "features": int(len(test)),
            "standard_eval_subset": int(test["is_standard_eval"].sum()),
            "peak_strata": rep["peak_strata"],
        }
        extra = ("`fire_count_16m` and `density_stratum` are null: they need the 2M SAE's "
                 "weights, which are on Modal volume `maem-data` (workspace "
                 "`<your-profile>`) at /data/dict2m/trainer_0/ae.pt and are not reachable "
                 "from here. `peak_stratum` is not a substitute -- different quantity, "
                 "different corpus.")
        layout.write_readme(root, family, _head_readme(family, train_info, test_info, extra))
        written.append(family)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    freg = registry.build()
    frep = registry.verify(freg)
    areg = activations.build()
    arep = activations.verify(areg)

    fams = emit_features(args.root, freg, frep) + emit_activations(args.root, areg, arep)
    print(f"root {args.root}")
    for fam in fams:
        head = layout.family_dir(args.root, fam)
        sides = {s: sorted(p.name for p in (head / s).iterdir()) for s in layout.SIDES}
        print(f"  {fam}/")
        for side, files in sides.items():
            print(f"    {side}/ {files if files else '(empty)'}")


if __name__ == "__main__":
    main()
