#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15"]
# ///
"""Do the HF and vLLM engines produce the same MAEM? CPU, local, no volume access.

Both sides are the SAME directions, the SAME MAEM weights and the SAME scorer (`score.py` on the
clean base) -- only the generation engine differs. The rollouts are not bitwise comparable (vLLM
seeds per request, HF per generate call; see precompute/rollouts_vllm.py), so everything here is
distributional or paired-per-direction:

  (a) overall: mean-of-n cosine, best-of-n, rollout length, eos rate, generation throughput.
  (b) per direction, PAIRED: the vLLM minus HF difference in mean-of-n and in best-of-n, with
      Pearson r across directions. r is the number that matters -- a constant offset would be a
      sampler difference, a low r would mean the two engines rank the directions differently.
  (c) the argmax-position histogram: where in a rollout the best-scoring token sits, as a fraction
      of the scored tokens. A shifted histogram is how a truncation or stop-token difference shows
      up even when the mean cosine does not move.
  (d) optional: both engines against run1's ARCHIVED per-direction best-of-64
      (`per_dir_final_bo64.json`), joined on `dir_index` -- the archive's own `families[fam].index`
      is that same pool vec_idx, so the join is by identity, not by list position.

Every input is a small file fetched off the volume (a few MB at the 48 x 64 smoke shape):

    <side>/per_target.jsonl   score.py's per-direction aggregates
    <side>/rows.json          which rows, and n
    <side>/cos.f16            [N, n, 96] per-token cosine (NaN outside the kept tokens)
    <side>/argmax.i16         [N, n] index of the best token among the scored ones
    <side>/summary.json       OPTIONAL: the rollouts summary (engine metadata, throughput)

    uv run evals/faithfulness/reconstruction/parity.py --hf <dir> --vllm <dir> --ids <ids.jsonl> \\
        [--dumps <run1 dumps dir>]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

WIDTH = 96  # common.SCORE_WIDTH
N_BINS = 10  # argmax-position histogram
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fmt(rows: list[list], head: list[str]) -> str:
    w = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(head))]
    out.append("  ".join("-" * w[i] for i in range(len(head))))
    out += ["  ".join(str(r[i]).ljust(w[i]) for i in range(len(head))) for r in rows]
    return "\n".join(out)


def load_side(d: Path) -> dict:
    """One engine's fetched score directory, with the per-rollout statistics recomputed from cos.f16.

    per_target.jsonl is read too, and its mean_cos is ASSERTED against the recomputation: the two
    disagreeing would mean the fetched arrays and the fetched aggregates are not from the same run.
    """
    rows = json.load(open(d / "rows.json"))
    per_t = {r["row"]: r for r in jsonl(d / "per_target.jsonl")}
    n, sel = int(rows["n"]), list(rows["rows"])
    cos = np.fromfile(d / "cos.f16", dtype=np.float16).reshape(len(sel), n, WIDTH).astype(np.float32)
    argmax = np.fromfile(d / "argmax.i16", dtype=np.int16).reshape(len(sel), n).astype(np.int64)
    kept = (~np.isnan(cos)).sum(2)  # scored tokens per rollout
    # common.agg: a row with no kept token scores -1.0
    best = np.where(np.isnan(cos).all(2), -1.0, np.nanmax(cos, axis=2))
    summary = {}
    if (d / "summary.json").exists():
        summary = json.load(open(d / "summary.json"))
    mine = np.array([best[i].mean() for i in range(len(sel))])
    theirs = np.array([per_t[r]["mean_cos"] for r in sel])
    assert np.abs(mine - theirs).max() < 2e-3, (
        f"{d}: per_target.jsonl's mean_cos and the recomputation from cos.f16 differ by up to "
        f"{np.abs(mine - theirs).max():.2e} -- the fetched arrays and aggregates are not one run "
        f"(the f16 store rounds at ~5e-4, so anything above 2e-3 is a mismatch)"
    )
    return {
        "dir": d,
        "rows": sel,
        "n": n,
        "per_t": per_t,
        "best": best,
        "argmax": argmax,
        "kept": kept,
        "summary": summary,
    }


def bo_k(best: np.ndarray, k: int) -> np.ndarray:
    """common.best_of_k_means per direction: disjoint consecutive groups of k, mean of the maxima."""
    n = best.shape[1]
    assert k <= n, f"best-of-{k} asked of only {n} rollouts"
    g = n // k
    return best[:, : g * k].reshape(best.shape[0], g, k).max(2).mean(1)


def rel_argmax(argmax: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """The argmax token's position as a fraction of the scored tokens, over rollouts with >= 2."""
    ok = (argmax >= 0) & (kept >= 2)
    return (argmax[ok] / (kept[ok] - 1)).astype(float)


@app.command()
def main(
    hf: Annotated[Path, typer.Option(help="fetched scores/<set>/ of the HF rollouts")],
    vllm: Annotated[Path, typer.Option(help="fetched scores/<set>__vllm/ of the vLLM rollouts")],
    ids: Annotated[Path, typer.Option(help="the held-out set's ids.jsonl (family, dir_index)")],
    dumps: Annotated[
        Path | None, typer.Option(help="run1 dumps dir with per_dir_final_bo64.json (optional)")
    ] = None,
) -> None:
    A, B = load_side(hf), load_side(vllm)
    assert A["rows"] == B["rows"], (
        f"the two engines scored different rows: hf {A['rows'][:8]}... vllm {B['rows'][:8]}..."
    )
    assert A["n"] == B["n"], f"hf n={A['n']} but vllm n={B['n']}: the paired statistics need one n"
    meta = {r["row"]: r for r in jsonl(ids)}
    sel, n = A["rows"], A["n"]
    fams = sorted({meta[r]["family"] for r in sel})
    print(f"# parity: {len(sel)} directions x n={n} rollouts, HF vs vLLM, same dirs, same scorer\n")

    ta, tb, tc = [], [], []
    for f in fams:
        idx = [i for i, r in enumerate(sel) if meta[r]["family"] == f]
        for name, S in (("hf", A), ("vllm", B)):
            b = S["best"][idx]
            lens = [S["per_t"][sel[i]]["mean_len"] for i in idx]
            eos = [S["per_t"][sel[i]]["eos_rate"] for i in idx]
            ta.append(
                [
                    f,
                    name,
                    len(idx),
                    f"{b.mean():.4f}",
                    f"{bo_k(b, n).mean():.4f}",
                    f"{np.mean(lens):.2f}",
                    f"{np.mean(eos):.3f}",
                    S["summary"].get("gen_tok_per_s", "-"),
                ]
            )
        a_b, b_b = A["best"][idx], B["best"][idx]
        a1, b1 = a_b.mean(1), b_b.mean(1)
        an, bn = bo_k(a_b, n), bo_k(b_b, n)
        tb.append(
            [
                f,
                len(idx),
                f"{a1.mean():.4f}",
                f"{b1.mean():.4f}",
                f"{(b1 - a1).mean():+.4f}",
                f"{np.abs(b1 - a1).max():.4f}",
                f"{pearson(a1, b1):+.4f}",
                f"{an.mean():.4f}",
                f"{bn.mean():.4f}",
                f"{(bn - an).mean():+.4f}",
                f"{pearson(an, bn):+.4f}",
            ]
        )
        for name, S in (("hf", A), ("vllm", B)):
            rel = rel_argmax(S["argmax"][idx], S["kept"][idx])
            hist = np.histogram(rel, bins=N_BINS, range=(0.0, 1.0))[0]
            tc.append(
                [f, name, len(rel), f"{rel.mean():.3f}", *[f"{c / max(len(rel), 1):.3f}" for c in hist]]
            )

    print("(a) overall, per family and engine")
    print(fmt(ta, ["family", "engine", "dirs", "mean bo1", f"mean bo{n}", "mean len", "eos", "gen tok/s"]))
    print("\n(b) per direction, PAIRED (vLLM - HF); r is across directions")
    print(
        fmt(
            tb,
            [
                "family",
                "dirs",
                "hf bo1",
                "vllm bo1",
                "d bo1",
                "max |d|",
                "r bo1",
                f"hf bo{n}",
                f"vllm bo{n}",
                f"d bo{n}",
                f"r bo{n}",
            ],
        )
    )
    print("\n(c) argmax position as a fraction of the scored tokens (row-normalised histogram)")
    print(
        fmt(
            tc,
            ["family", "engine", "rollouts", "mean", *[f"{i / N_BINS:.1f}-" for i in range(N_BINS)]],
        )
    )

    if dumps is not None:
        arch = json.load(open(dumps / "per_dir_final_bo64.json"))["families"]
        td = []
        for f in fams:
            idx = [i for i, r in enumerate(sel) if meta[r]["family"] == f]
            if f not in arch:
                td.append([f, len(idx), "-", "-", "-", "-", "-", "-", "not in the archive dump"])
                continue
            index = list(arch[f]["index"])
            pos = {int(v): j for j, v in enumerate(index)}
            keep, ours_hf, ours_v, theirs = [], [], [], []
            for i in idx:
                di = meta[sel[i]].get("dir_index")
                if di is None or int(di) not in pos:
                    continue
                j = pos[int(di)]
                ar = meta[sel[i]].get("archive_row")
                assert ar is None or int(ar) == j, (
                    f"{f} row {sel[i]}: dir_index {di} sits at archive position {j} but ids.jsonl "
                    f"says archive_row {ar} -- the join key and the stored position disagree"
                )
                keep.append(i)
                theirs.append(arch[f]["cos"][j])
            if not keep:
                td.append([f, len(idx), 0, "-", "-", "-", "-", "-", "no dir_index join"])
                continue
            ours_hf = bo_k(A["best"][keep], n)
            ours_v = bo_k(B["best"][keep], n)
            th = np.array(theirs, float)
            td.append(
                [
                    f,
                    len(idx),
                    len(keep),
                    f"{ours_hf.mean():.4f}",
                    f"{ours_v.mean():.4f}",
                    f"{th.mean():.4f}",
                    f"{(ours_hf - th).mean():+.4f}",
                    f"{(ours_v - th).mean():+.4f}",
                    f"{pearson(ours_hf, th):+.3f} / {pearson(ours_v, th):+.3f}",
                ]
            )
        print(f"\n(d) both engines' best-of-{n} vs run1's ARCHIVED best-of-64, joined on dir_index")
        print(
            fmt(
                td,
                [
                    "family",
                    "dirs",
                    "joined",
                    f"hf bo{n}",
                    f"vllm bo{n}",
                    "archive bo64",
                    "d hf",
                    "d vllm",
                    "r hf / r vllm",
                ],
            )
        )

    print(
        "\nNotes: the two engines do NOT share an RNG stream (vLLM seeds per request, HF per "
        "generate call), so (b) is distributional agreement per direction, never bitwise. "
        "(d) is our n rollouts against the archive's 64, both as best-of-k of the same estimator."
    )


if __name__ == "__main__":
    app()
