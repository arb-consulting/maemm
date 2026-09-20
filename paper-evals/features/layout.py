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


def family_dir(root: str | Path, family: str) -> Path:
    """`<root>/<family>/`, with both sides created."""
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
