"""Wrap Celeste's frozen v2 directions as a held-out set the pipeline already reads.

    python -m features.heldout_v2 --out /vol/base/qwen36-27b/heldout/2026-09-19_v2

Produces the on-disk shape `precompute/targets.py` documents, so `scan.py`,
`rollouts_hf.py`, `rollouts_vllm.py` and `score.py` consume it with no change:

    ids.jsonl   one row per target: row, family, id, stratum + the family's own fields
    vecs.f16    [N, d] unit rows, row i is ids.jsonl line i
    README.md   what was imported, from where, and what each family's split claim is

This is an IMPORT, not a draw -- the analogue of targets.py's `import_run1`. The
directions are frozen in the bundle and the checkpoint was trained against that draw,
so re-drawing them would void the held-out claim. Nothing here samples anything.

Two things the importer refuses rather than papers over:

  * a family with no declared `heldout_kind` (layout.FAMILY_HELDOUT) -- a target set is
    not built for a family whose split is unstated;
  * an `in_distribution` family, unless --include-indist is passed. Those directions are
    not held out, and the failure mode to prevent is one of them quietly entering a
    "held-out" mean.

Note the vectors are ALREADY CENTRED with Celeste's whiten_mu (see activations.py), so
unlike targets.py's own realact path there is no mu to subtract here, and no `mu_512.f32`
is written. mu itself is not in the bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import activations, bundle, layout

DEFAULT_FAMILIES = ("realact", "random")


def _natural_id(family: str, row: int, meta) -> int | str:
    """The family's own identifier for a target, not its position in the set."""
    if family in ("sae2m_enc", "sae2m_dec", "sae") and "feats" in meta.columns:
        return int(np.ravel(meta["feats"].iloc[row])[0])
    if family == "realact" and "pool_seq" in meta.columns:
        return f"doc{int(meta['pool_seq'].iloc[row])}:pos{int(meta['pool_pos'].iloc[row])}"
    return int(row)


def build(families: list[str]) -> tuple[list[dict], np.ndarray]:
    reg = activations.build()
    rows: list[dict] = []
    vecs: list[np.ndarray] = []
    for family in families:
        kind = layout.heldout_kind(family)          # refuses an undeclared family
        t = bundle.load_family(family)
        sub = reg[reg["family"] == family].reset_index(drop=True)
        for i in range(len(t)):
            row = {
                "row": len(rows),
                "family": family,
                "id": _natural_id(family, i, t.meta),
                "stratum": None,
                "heldout_kind": kind,
                "family_row": i,
            }
            if len(sub) == len(t):
                cell = sub.iloc[i]
                if pd.notna(cell.get("norm_stratum")):
                    row["stratum"] = int(cell["norm_stratum"])
                for col in ("doc", "pos", "act_norm", "doc_n_targets"):
                    if pd.notna(cell.get(col)):
                        row[col] = float(cell[col]) if col == "act_norm" else int(cell[col])
            rows.append(row)
            vecs.append(t.directions[i])
    return rows, np.stack(vecs)


def readme(rows: list[dict], vecs: np.ndarray, families: list[str]) -> str:
    by_fam: dict[str, int] = {}
    for r in rows:
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    norms = np.linalg.norm(vecs.astype(np.float32), axis=1)
    lines = [
        "# Held-out set: imported from Celeste's v2 bundle",
        "",
        f"Snapshot `{bundle.SNAPSHOT}`, base `{bundle.MODEL}`, read layer "
        f"{bundle.READ_LAYER}, d = {bundle.D_MODEL}.",
        f"Checkpoint these targets are held out from: `{bundle.CHECKPOINT}`",
        f"(revision `{bundle.CHECKPOINT_REVISION}`).",
        "",
        "Rebuild:",
        "",
        "```",
        f"python -m features.heldout_v2 --families {' '.join(families)}",
        "```",
        "",
        "IMPORTED, NOT DRAWN. The directions are frozen in the bundle and the checkpoint",
        "was trained against that draw; re-drawing would void the held-out claim.",
        "",
        "The vectors are already centred with Celeste's `whiten_mu`, so no mu is",
        "subtracted here and no `mu_512.f32` is written. mu is not in the bundle.",
        "",
        "| family | targets | held out by | what that establishes |",
        "|---|---|---|---|",
    ]
    for fam in families:
        kind = layout.heldout_kind(fam)
        lines.append(f"| `{fam}` | {by_fam.get(fam, 0)} | `{kind}` | "
                     f"{layout.KIND_MEANING[kind]} |")
    lines += [
        "",
        f"{len(rows)} targets, vecs.f16 [{vecs.shape[0]}, {vecs.shape[1]}], "
        f"unit to {abs(norms - 1).max():.2e} before the f16 cast.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    ap.add_argument("--include-indist", action="store_true",
                    help="allow in_distribution families; they are NOT held out")
    args = ap.parse_args()

    for fam in args.families:
        kind = layout.heldout_kind(fam)
        if kind == "in_distribution" and not args.include_indist:
            raise SystemExit(
                f"{fam!r} is in_distribution -- not held out. Pass --include-indist if "
                f"you mean to import it anyway, and keep it out of any held-out mean.")

    rows, vecs = build(args.families)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ids.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    vecs.astype(np.float16).tofile(out / "vecs.f16")
    # THE STORAGE CONTRACT (H4). Without it `common.set_storage` refuses this set and every
    # product that reads a direction stops at it. The families imported here arrive ALREADY
    # CENTRED on Celeste's own mean and this side of the bundle records no path for it, so the
    # honest value is `unknown`: `common.dirs_for` then returns those rows AS SHIPPED, with a
    # warning and a label that travels into the reading product's README, instead of pretending
    # they can be re-derived. A family that is not centrable at all (an encoder column, a
    # subspace basis, a Gaussian draw) gets null -- see config.yaml `family_kinds:`.
    fam_mu = {
        fam: (None if fam in ("random", "sae", "sae2m_enc", "bsf", "jlens") else "unknown")
        for fam in args.families
    }
    (out / "storage.json").write_text(
        json.dumps(
            {
                "storage": "unit",
                "mu_stored": None,
                "family_mu": fam_mu,
                "note": (
                    "imported from Celeste's bundle: stored unit directions, no act.f32, so they "
                    "cannot be moved to another mean. `unknown` means the mean they carry is not "
                    "one this repo holds a file for."
                ),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    (out / "README.md").write_text(readme(rows, vecs, args.families), encoding="utf-8")

    print(f"{out}")
    print(f"  ids.jsonl  {len(rows)} rows")
    print(f"  vecs.f16   {vecs.shape} float16")
    print(f"  storage.json  unit, family_mu={fam_mu}")
    for fam in args.families:
        n = sum(1 for r in rows if r["family"] == fam)
        print(f"    {fam:16s} {n:5d}  heldout_kind={layout.heldout_kind(fam)}")


if __name__ == "__main__":
    main()
