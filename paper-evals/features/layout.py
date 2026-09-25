"""Precompute output layout: one head folder per family, train/ and test/ inside it.

    <root>/
      realact/
        README.md          what the family is, how it was drawn, what is in each side
        train/             products over the TRAINING-side activations of this family
        test/              products over the frozen eval targets
      random/
        README.md
        train/             (empty: a Gaussian control has no training side -- its
        test/               README says so rather than the folder being absent)
      sae2m_enc/
        README.md
        train/
        test/

Rules, on top of the branch's existing ones (one README per directory written by the
script that fills it; temp-and-rename; no overwrite without --force):

  * The head folder is the FAMILY, never the product. A product is a file or a
    subdirectory inside a side, so `realact/test/rollouts.jsonl` and
    `realact/test/scan/` both live under the one head.
  * Both sides always exist. A family with no training side gets an empty `train/`
    carrying a README that says why, so the absence is a statement and not an
    oversight.
  * `README.md` sits in the HEAD, not in the sides. It covers both sides, because the
    thing a reader needs is the relation between them -- what is held out from what.
"""
from __future__ import annotations

from pathlib import Path

SIDES = ("train", "test")

# How each family's held-out claim is established. Nothing here draws or re-draws a
# split -- the partitions already exist and are verified. This records WHICH KIND each
# family is, because the paper currently runs three different kinds together and reads
# as though they were one.
#
#   feature_id      the seed-2026 partition; no training bank holds an eval feature.
#                   Verified by features/registry.py.
#   doc_range       disjoint Ultra-FineWeb document ranges, 0 intersection, tightest
#                   margin 396,399 documents. Verified 2026-09-18. Note doc-disjoint
#                   is not content-disjoint (0.50% of realact rows share a 13-gram).
#   category        "the generator never saw this KIND of direction". Asserted from
#                   the training mix, not measured. The weakest of the three, and the
#                   one carrying the paper's generalisation claims.
#   in_distribution NOT held out. Must never be pooled into a held-out mean.
#   unknown         the bundle ships no provenance -- cannot be claimed either way.
HELDOUT_KINDS = ("feature_id", "doc_range", "category", "in_distribution", "unknown")

KIND_MEANING = {
    "feature_id": "seed-2026 feature partition; no training bank holds an eval feature "
                  "(verified by features/registry.py)",
    "doc_range": "disjoint Ultra-FineWeb document ranges, 0 intersection, tightest "
                 "margin 396,399 documents (verified 2026-09-18). Doc-disjoint is not "
                 "content-disjoint: 0.50% of realact rows share a 13-gram",
    "category": "the generator never saw this kind of direction. Asserted from the "
                "training mix, not measured",
    "in_distribution": "NOT held out. Never pool this family into a held-out mean",
    "unknown": "the bundle ships no provenance for this family, so the held-out claim "
               "can be inherited but not checked",
}

FAMILY_HELDOUT = {
    "sae2m_enc":      "feature_id",
    "sae2m_dec":      "feature_id",
    "sae":            "feature_id",   # the 131k SAE; held out of the EARLIER chains only
    "realact":        "doc_range",
    "realact_early":  "unknown",      # no provenance shipped; see features/README.md
    "realact_mid":    "unknown",
    "realact_long":   "unknown",
    "random":         "category",     # a Gaussian control has no corpus origin
    "indist_realact": "in_distribution",
    "indist_long":    "in_distribution",
    "indist_probe":   "in_distribution",
    "cluster":        "unknown",      # in the legacy chain's mix; no v2 doc_ids list
    "jlens":          "unknown",
    "bsf":            "unknown",
}

# Families with no training side, and the reason, written into the empty train/.
NO_TRAIN_SIDE = {
    "random": "a Gaussian control has no corpus origin, so nothing was trained on it",
    "realact_early": "the bundle ships no provenance for this family; its training-side "
                     "documents are unknown (see ../README.md)",
    "realact_mid": "the bundle ships no provenance for this family; its training-side "
                   "documents are unknown (see ../README.md)",
    "realact_long": "the bundle ships no provenance for this family; its training-side "
                    "documents are unknown (see ../README.md)",
}


def heldout_kind(family: str) -> str:
    """How this family's held-out claim is established. Refuses an undeclared family."""
    try:
        return FAMILY_HELDOUT[family]
    except KeyError:
        raise KeyError(
            f"{family!r} has no heldout_kind. Nothing is precomputed for a family whose "
            f"split is not stated: add it to layout.FAMILY_HELDOUT as one of "
            f"{HELDOUT_KINDS}."
        ) from None


def family_dir(root: str | Path, family: str) -> Path:
    """`<root>/<family>/`, with both sides created. Declared families only."""
    heldout_kind(family)
    head = Path(root) / family
    for side in SIDES:
        (head / side).mkdir(parents=True, exist_ok=True)
    return head


def path(root: str | Path, family: str, side: str, *parts: str) -> Path:
    """`<root>/<family>/<side>/<parts...>`, parents created."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    p = family_dir(root, family) / side
    for part in parts[:-1]:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p / parts[-1] if parts else p


def write_readme(root: str | Path, family: str, body: str,
                 train_note: str | None = None) -> Path:
    """The head README, plus a note in any side left empty.

    An empty side must say why it is empty, so that the absence reads as a statement
    rather than as a run that failed halfway.
    """
    head = family_dir(root, family)
    (head / "README.md").write_text(body, encoding="utf-8")
    note = train_note or NO_TRAIN_SIDE.get(family)
    train = head / "train"
    if note and not any(q.name != "README.md" for q in train.iterdir()):
        (train / "README.md").write_text(
            f"# {family} / train -- empty by design\n\n{note}\n", encoding="utf-8")
    return head / "README.md"
