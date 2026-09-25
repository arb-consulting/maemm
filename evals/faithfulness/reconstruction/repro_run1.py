#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15"]
# ///
"""Does our pipeline reproduce run1's archived numbers? CPU, local, no volume access.

Compares, on the 16 directions per family that `targets.py --import-run1` copied out of run1's
eval cache (`2026-09-03_run1-archive16`):

  (a) bo1: our mean over 16 x 64 rollouts vs the archive's mean over 16 x 4, with naive SEs.
  (b) per direction: our mean-of-64 vs the archive's best-of-4 and best-of-64 -- Pearson r and
      mean difference. (Different statistics; the point is the RANKING, i.e. r.)
  (c) our best-of-64 vs the archived best-of-64, per direction: paired mean difference and r.
  (d) length: our mean generated n_tok vs the archived rollouts' n_tok.
  (e) the archived rollout TEXTS pushed through OUR scorer vs their stored `cos_orig` -- this
      isolates the scorer from the sampler. Theirs dropped tokens whose residual norm exceeded 10x
      the row median; ours drops none, so ours can only be >= theirs, and is equal wherever their
      filter dropped nothing.

Both sides' rollouts come from GEN_SEED 1234 but NOT from the same RNG stream (see
precompute/rollouts_hf.py), so (a)-(d) are distributional agreement, never bitwise.

  uv run evals/faithfulness/reconstruction/repro_run1.py --ours <dir> --dumps <dir>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

FAMS = ("realact", "sae", "random")
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


from parity import fmt, pearson

def se(x) -> float:
    """Naive standard error. Rollouts of the same direction are CORRELATED, so this understates the
    uncertainty of a family mean; it is the right scale for 'do these two means agree', not a test."""
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


@app.command()
def main(
    ours: Annotated[
        Path,
        typer.Option(help="dir with ids/roll/roll_summary/per_target/cos.f16/rescore_rows off the volume"),
    ],
    dumps: Annotated[
        Path, typer.Option(help="dir with ss_samples.jsonl, per_dir_final.json, per_dir_final_bo64.json")
    ],
    n: Annotated[int, typer.Option(help="directions per family (the imported set's size)")] = 16,
) -> None:
    ids = jsonl(ours / "ids.jsonl")
    rolls = jsonl(ours / "roll.jsonl")
    per_t = {r["row"]: r for r in jsonl(ours / "per_target.jsonl")}
    rescore = jsonl(ours / "rescore_rows.jsonl")
    ss = jsonl(dumps / "ss_samples.jsonl")
    bo4 = json.load(open(dumps / "per_dir_final.json"))["families"]
    bo64 = json.load(open(dumps / "per_dir_final_bo64.json"))["families"]
    n_roll = int(json.load(open(ours / "roll_summary.json"))["n"])

    rows_of = {f: [r for r in ids if r["family"] == f] for f in FAMS}
    for f in FAMS:
        assert len(rows_of[f]) == n, f"{f}: the imported set has {len(rows_of[f])} rows, expected {n}"
    raw = np.fromfile(ours / "cos.f16", dtype=np.float16)
    cos = raw.reshape(len(ids), n_roll, -1).astype(np.float32)
    # per-rollout score = max over the kept tokens; common.agg's masked_fill(-1) on an all-NaN row
    per_roll = np.where(np.isnan(cos).all(2), -1.0, np.nanmax(cos, axis=2))

    ss_by = {}
    for r in ss:
        if r["row"] < n and r["family"] in FAMS:
            ss_by.setdefault((r["family"], r["row"]), []).append(r)
    roll_by: dict[int, list[dict]] = {}
    for r in rolls:
        roll_by.setdefault(r["row"], []).append(r)

    print(f"# repro_run1: {n} directions per family, ours n={n_roll} rollouts, archive bo4 + bo64\n")

    ta, tb, tc, td, te = [], [], [], [], []
    for f in FAMS:
        rr = rows_of[f]
        our_rows = [r["row"] for r in rr]
        arch_rows = [r["archive_row"] for r in rr]

        # ---- (a) bo1 means -------------------------------------------------------------------
        ours_flat = per_roll[our_rows].ravel()
        arch4 = np.array([x["cos_orig"] for i in arch_rows for x in ss_by[(f, i)]])
        arch64 = np.array([bo64[f]["rollout_cos"][i] for i in arch_rows]).ravel()
        ta.append(
            [
                f,
                f"{ours_flat.mean():.4f} +- {se(ours_flat):.4f}",
                f"{arch4.mean():.4f} +- {se(arch4):.4f}",
                f"{arch64.mean():.4f} +- {se(arch64):.4f}",
                f"{ours_flat.mean() - arch4.mean():+.4f}",
                f"{ours_flat.mean() - arch64.mean():+.4f}",
            ]
        )

        # ---- (b) per direction: our mean-of-64 vs their best-of-4 / best-of-64 ----------------
        our_mean = np.array([per_t[r]["mean_cos"] for r in our_rows])
        their_b4 = np.array([bo4[f]["cos"][i] for i in arch_rows])
        their_b64 = np.array([bo64[f]["cos"][i] for i in arch_rows])
        tb.append(
            [
                f,
                f"{our_mean.mean():.4f}",
                f"{their_b4.mean():.4f}",
                f"{pearson(our_mean, their_b4):+.3f}",
                f"{(our_mean - their_b4).mean():+.4f}",
                f"{their_b64.mean():.4f}",
                f"{pearson(our_mean, their_b64):+.3f}",
                f"{(our_mean - their_b64).mean():+.4f}",
            ]
        )

        # ---- (c) our best-of-64 vs their best-of-64 ------------------------------------------
        our_b64 = np.array([per_t[r][f"bo_{n_roll}"] for r in our_rows])
        d = our_b64 - their_b64
        tc.append(
            [
                f,
                f"{our_b64.mean():.4f}",
                f"{their_b64.mean():.4f}",
                f"{d.mean():+.4f}",
                f"{np.abs(d).max():.4f}",
                f"{pearson(our_b64, their_b64):+.3f}",
            ]
        )

        # ---- (d) lengths ---------------------------------------------------------------------
        our_len = np.array([x["n_tok"] for r in our_rows for x in roll_by[r]])
        their_len = np.array([x["n_tok"] for i in arch_rows for x in ss_by[(f, i)]])
        our_eos = np.array([x["finished"] for r in our_rows for x in roll_by[r]], float)
        td.append(
            [
                f,
                f"{our_len.mean():.2f}",
                f"{their_len.mean():.2f}",
                f"{our_len.mean() - their_len.mean():+.2f}",
                f"{(our_len == 64).mean():.3f}",
                f"{(their_len == 64).mean():.3f}",
                f"{our_eos.mean():.3f}",
            ]
        )

        # ---- (e) their texts through OUR scorer ----------------------------------------------
        sub = [r for r in rescore if r["archive_family"] == f]
        assert len(sub) == n * 4, f"{f}: rescored {len(sub)} rows, expected {n} dirs x 4 rollouts"
        mine = np.array([r["cos"] for r in sub])
        theirs = np.array([r["cos_orig"] for r in sub])
        d = mine - theirs
        te.append(
            [
                f,
                len(sub),
                f"{mine.mean():.5f}",
                f"{theirs.mean():.5f}",
                f"{d.mean():+.2e}",
                f"{np.abs(d).max():.2e}",
                f"{pearson(mine, theirs):.6f}",
                f"{(d >= -1e-4).mean():.3f}",
                f"{(np.abs(d) < 1e-4).mean():.3f}",
            ]
        )

    print("(a) bo1 (per-rollout) means")
    print(
        fmt(
            ta,
            ["family", "ours (16x64)", "archive bo4 (16x4)", "archive bo64 (16x64)", "d vs bo4", "d vs bo64"],
        )
    )
    print("\n(b) per direction: our mean-of-64 vs their best-of-k (different statistics; r is the point)")
    print(fmt(tb, ["family", "our mean64", "their bo4", "r", "mean d", "their bo64", "r", "mean d"]))
    print(f"\n(c) our best-of-{n_roll} vs their best-of-64, per direction (paired)")
    print(fmt(tc, ["family", "ours bo64", "theirs bo64", "mean d", "max |d|", "r"]))
    print("\n(d) rollout length")
    print(fmt(td, ["family", "our n_tok", "their n_tok", "diff", "ours at 64", "theirs at 64", "our eos"]))
    print("\n(e) THEIR texts through OUR scorer vs their stored cos_orig (no norm filter vs 10x median)")
    print(
        fmt(te, ["family", "rows", "ours", "theirs", "mean d", "max |d|", "r", "frac ours>=", "frac equal"])
    )
    print(
        "\nNotes: SEs are naive (rollouts of one direction are correlated). (a)-(d) are "
        "distributional: both sides use GEN_SEED 1234 but not the same RNG stream."
    )


if __name__ == "__main__":
    app()
