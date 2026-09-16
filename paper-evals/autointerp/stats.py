#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "polars>=1", "typer>=0.15", "rich>=13", "pyyaml>=6"]
# ///
"""The autointerp analysis layer: the pilot's tables, from the small files one `run` wrote.

Local, CPU, no GPU and no model. It fetches ONLY `runs/<run>/summary/*` and
`runs/<run>/explain/explanations.jsonl` into `autointerp/data/<run>/` (gitignored) and writes
`autointerp/pilot.md`, which IS committed.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (export MODAL_PROFILE=maemms; \\
     uv run repo-maemm-precompute/paper-evals/autointerp/stats.py --run 2026-09-16_autointerp-27b)

The unit of analysis is the FEATURE, everywhere. Every difference is paired within a feature (the
test set is identical across arms by construction, and the item order and the batching are too), so
the CI is a percentile bootstrap over features, not over items. The design asks for:

  * M - C16          "are rollouts as good as max-activating corpus examples?"
  * (C4+M) - C4      "do rollouts add to a CHEAP corpus?"
  * C4 - C16         what the cheap corpus costs on its own
  * the N ablations, and the additive C4+M16 against C4
  * the same, per density quartile
  * win fractions and the whole distribution, because the outcome is bimodal and a mean misleads
  * fire fraction as the covariate that separates the hard stratum
  * C16-rep - C16, the run-to-run noise floor every other difference is quoted against

`reconstruction/stats.py`'s `Vol` does the fetching; nothing here re-implements it, and nothing
here writes to that file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from rich.console import Console

HERE = Path(__file__).resolve().parent
PAPER_EVALS = HERE.parent
if str(PAPER_EVALS) not in sys.path:
    sys.path.insert(0, str(PAPER_EVALS))

from reconstruction.stats import Vol, sign_test  # noqa: E402

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# The comparisons the design names, as (label, arm_a, arm_b) meaning a - b.
CONTRASTS = [
    ("substitution", "M", "C16"),
    ("enrichment", "C4M", "C4"),
    ("cheap corpus", "C4", "C16"),
    ("additive (N=32)", "C4M16", "C4"),
    ("corpus N: 8 - 16", "C16-N8", "C16"),
    ("corpus N: 32 - 16", "C16-N32", "C16"),
    ("maemm N: 8 - 16", "M-N8", "M"),
    ("maemm N: 32 - 16", "M-N32", "M"),
]
DRIFT = ("drift (repeat of C16)", "C16-rep", "C16")
QUANTS = (0.10, 0.25, 0.50, 0.75, 0.90)
N_BOOT = 10000
BOOT_SEED = 20260916


def boot_ci(d: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05):
    """(mean, lo, hi) percentile bootstrap of the mean of `d`, resampling FEATURES."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def ci_str(m: float, lo: float, hi: float, nd: int = 4) -> str:
    if not np.isfinite(lo):
        return f"{m:.{nd}f}"
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def paired(df: pl.DataFrame, a: str, b: str, scorer: str):
    """(features, d) -- the per-feature difference arm `a` minus arm `b` for one scorer."""
    sub = df.filter(pl.col("scorer") == scorer)
    wide = (
        sub.filter(pl.col("arm").is_in([a, b]))
        .pivot(values="bal_acc", index="feature", on="arm")
        .drop_nulls()
    )
    if a not in wide.columns or b not in wide.columns or not len(wide):
        return np.zeros(0, dtype=int), np.zeros(0)
    return wide["feature"].to_numpy(), (wide[a] - wide[b]).to_numpy()


def md_table(rows: list[list[str]], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


@app.command()
def main(
    run: Annotated[str, typer.Option(help="the run directory name under /vol/runs/")],
    out: Annotated[str, typer.Option(help="the markdown file to write")] = "",
    data_dir: Annotated[str, typer.Option()] = "",
    modal_cmd: Annotated[str, typer.Option()] = "uvx modal",
    refetch: Annotated[bool, typer.Option()] = False,
    no_fetch: Annotated[bool, typer.Option("--no-fetch")] = False,
):
    data = Path(data_dir) if data_dir else HERE / "data" / run
    data.mkdir(parents=True, exist_ok=True)
    vol = Vol("full", data, modal_cmd, refetch, quiet=False, offline=no_fetch)
    base = f"runs/{run}"

    scores = vol.jsonl(f"{base}/summary/scores.jsonl")
    feats = vol.json(f"{base}/summary/features.json")
    costs = vol.json(f"{base}/summary/costs.json")
    binfo = vol.json(f"{base}/summary/build.json")
    expl = vol.jsonl(f"{base}/explain/explanations.jsonl")
    missing = [n for n, x in
               [("scores.jsonl", scores), ("features.json", feats), ("costs.json", costs),
                ("build.json", binfo), ("explanations.jsonl", expl)] if x is None]
    assert not missing, f"{base} is missing {missing} (fetch log: {vol.missing})"

    df = pl.DataFrame(scores)
    arms = sorted(df["arm"].unique().to_list())
    scorers = sorted(df["scorer"].unique().to_list())
    n_feat = df["feature"].n_unique()
    console.print(f"[bold]{n_feat} features, arms {arms}, scorers {scorers}[/bold]")

    lines: list[str] = []
    lines += [
        f"# Autointerp pilot -- `{run}`",
        "",
        f"Delphi-style SAE autointerp on base `{binfo['base']}`, SAE `{binfo['sae']}`, held-out "
        f"set `{binfo['set']}`, MAEMM `{binfo['maemm']}` (engine `{binfo['engine']}`). "
        f"Explainer = scorer = `{costs['model']}` at temperature {costs['temperature']} through "
        f"OpenRouter. Metric: per-feature **balanced accuracy** of the scorer using the "
        f"explainer's description, over a test set of {binfo['n_pos']} positives "
        f"(5 from each stored activation band) and {binfo['n_neg']} zero-activation negatives, "
        f"IDENTICAL across arms and never shown to any explainer.",
        "",
        f"**n = {n_feat} features** ({n_feat // 4} per density quartile, seed "
        f"{binfo['feat_seed']}), {len(arms)} arm-variants, {len(scorers)} scorers. "
        f"SAE gate {binfo['gate']:.4f}. Generated by `autointerp/stats.py`; every number below is "
        f"computed from `runs/{run}/summary/scores.jsonl`.",
        "",
        "## Cost",
        "",
    ]
    tc = costs["cumulative_over_cache"]
    lines += md_table(
        [[
            "total (all arms, all stages)",
            f"{tc['calls']:,}",
            f"{tc['in']:,}",
            f"{tc['out']:,}",
            f"${tc['cost']:.4f}",
            f"${tc['cost'] / max(1, n_feat):.4f}",
        ]] + [
            [f"`{a}`", f"{d['calls']:,}", f"{d['in']:,}", f"{d['out']:,}", f"${d['cost']:.4f}",
             f"${d['cost'] / max(1, n_feat):.4f}"]
            for a, d in costs["per_arm"].items()
        ],
        ["arm", "calls", "input tok", "output tok", "cost", "$/feature"],
    )
    lines += [
        "",
        f"Cost is each response's own `usage.cost`, never a key usage delta. "
        f"{costs['cache_hits']:,} of {costs['cache_hits'] + costs['cache_misses']:,} calls came "
        f"from the prompt cache. Projection recorded at the probe gate: "
        + (f"${costs['projection_usd']:.2f}." if costs["projection_usd"] else "n/a (not reached)."),
        "",
    ]

    # ---- per-arm levels -------------------------------------------------------------------
    lines += ["## Per-arm balanced accuracy", ""]
    rows = []
    for scorer in scorers:
        for a in arms:
            v = df.filter((pl.col("scorer") == scorer) & (pl.col("arm") == a))["bal_acc"]
            v = np.asarray([x for x in v.to_list() if x is not None], dtype=float)
            if not len(v):
                continue
            m, lo, hi = boot_ci(v)
            rows.append([
                scorer, f"`{a}`", len(v), f"{m:.4f} [{lo:.4f}, {hi:.4f}]",
                *[f"{np.quantile(v, q):.3f}" for q in QUANTS],
                f"{float((v <= 0.5 + 1e-9).mean()):.3f}",
            ])
    lines += md_table(
        rows,
        ["scorer", "arm", "n", "mean [95% CI]", *[f"q{int(q * 100)}" for q in QUANTS], "frac <= 0.5"],
    )
    lines += [
        "",
        "`frac <= 0.5` is the fraction of features on which the description is no better than "
        "chance -- the bimodality the design warns about, which a mean alone hides.",
        "",
    ]

    # ---- drift ----------------------------------------------------------------------------
    drift_txt = {}
    lines += ["## Noise floor: the repeated arm", ""]
    rows = []
    for scorer in scorers:
        _f, d = paired(df, DRIFT[1], DRIFT[2], scorer)
        if not len(d):
            continue
        m, lo, hi = boot_ci(d)
        p, win, m_non = sign_test(d)
        drift_txt[scorer] = (m, lo, hi, float(np.abs(d).mean()), len(d))
        rows.append([
            scorer, f"`{DRIFT[1]}` - `{DRIFT[2]}`", len(d), ci_str(m, lo, hi),
            f"{float(np.abs(d).mean()):.4f}", f"{float(np.abs(d).std(ddof=1)):.4f}",
            f"{win:.3f}" if np.isfinite(win) else "-", f"{p:.3f}" if np.isfinite(p) else "-",
        ])
    lines += md_table(
        rows,
        ["scorer", "contrast", "n", "mean diff [95% CI]", "mean |diff|", "sd |diff|",
         "win frac", "sign p"],
    )
    lines += [
        "",
        "`C16-rep` is byte-identical to `C16`: the same 16 examples, re-explained and re-scored "
        "under a separate cache key. Its mean |difference| is the run-to-run floor -- a contrast "
        "below it is not a finding whatever its CI says.",
        "",
    ]

    # ---- contrasts --------------------------------------------------------------------------
    lines += ["## Paired contrasts (features as the unit, percentile bootstrap, B = "
              f"{N_BOOT:,})", ""]
    rows = []
    for scorer in scorers:
        floor = drift_txt.get(scorer, (0, 0, 0, float("nan"), 0))[3]
        for label, a, b in CONTRASTS:
            _f, d = paired(df, a, b, scorer)
            if not len(d):
                continue
            m, lo, hi = boot_ci(d)
            p, win, _m = sign_test(d)
            clear = "yes" if np.isfinite(lo) and (lo > 0 or hi < 0) else "no"
            over = "yes" if np.isfinite(floor) and abs(m) > floor else "no"
            rows.append([
                scorer, label, f"`{a}` - `{b}`", len(d), ci_str(m, lo, hi),
                f"{win:.3f}" if np.isfinite(win) else "-",
                f"{p:.4f}" if np.isfinite(p) else "-", clear, over,
            ])
    lines += md_table(
        rows,
        ["scorer", "contrast", "arms", "n", "mean diff [95% CI]", "win frac", "sign p",
         "CI clears 0", "|diff| > drift floor"],
    )
    lines += [""]

    # ---- per quartile -----------------------------------------------------------------------
    lines += ["## Per density quartile", "",
              "Quartile 0 is the rarest quarter of features by gated corpus density, 3 the "
              "commonest (`precompute/targets.py:229-252`).", ""]
    strat = {int(r["feature"]): int(r["stratum"]) for r in feats["features"]}
    rows = []
    for scorer in scorers:
        for label, a, b in CONTRASTS[:3]:
            for q in sorted(set(strat.values())):
                f_ids, d = paired(df, a, b, scorer)
                keep = np.asarray([strat[int(x)] == q for x in f_ids])
                dq = d[keep]
                if not len(dq):
                    continue
                m, lo, hi = boot_ci(dq)
                _p, win, _m = sign_test(dq)
                rows.append([scorer, label, q, len(dq), ci_str(m, lo, hi),
                             f"{win:.3f}" if np.isfinite(win) else "-"])
    lines += md_table(rows, ["scorer", "contrast", "quartile", "n", "mean diff [95% CI]", "win frac"])
    lines += [""]

    # ---- fire fraction ----------------------------------------------------------------------
    lines += ["## Fire fraction as covariate", "",
              "`fire_fraction` is the share of the MAEMM's 64 rollouts on which the target "
              "feature exceeds the SAE gate somewhere (`sae_self`). The hard stratum is exactly "
              "the set the MAEMM never fires on, so this is the covariate that should separate "
              "it.", ""]
    fire = {int(r["feature"]): float(r["fire_fraction"]) for r in feats["features"]}
    edges = [0.0, 0.25, 0.5, 0.75, 1.0001]
    rows = []
    for scorer in scorers:
        f_ids, d = paired(df, "M", "C16", scorer)
        if not len(d):
            continue
        fv = np.asarray([fire[int(x)] for x in f_ids])
        r = float(np.corrcoef(fv, d)[0, 1]) if len(d) > 2 else float("nan")
        for lo_e, hi_e in zip(edges[:-1], edges[1:], strict=True):
            keep = (fv >= lo_e) & (fv < hi_e)
            if not keep.any():
                continue
            m, lo, hi = boot_ci(d[keep])
            rows.append([scorer, f"[{lo_e:.2f}, {hi_e:.2f})", int(keep.sum()),
                         f"{fv[keep].mean():.3f}", ci_str(m, lo, hi)])
        lines.append(f"Pearson r(fire fraction, `M` - `C16`) = **{r:.3f}** on {scorer} (n = {len(d)}).")
        lines.append("")
    lines += md_table(rows, ["scorer", "fire fraction bin", "n", "mean fire frac",
                             "`M` - `C16` [95% CI]"])
    lines += [""]

    # ---- explanations / parse health --------------------------------------------------------
    edf = pl.DataFrame(expl)
    n_empty = int(edf.filter(~pl.col("ok")).height)
    parsed = df.select(
        (pl.col("n_parsed").sum() / pl.col("n_batches").sum()).alias("f")
    )["f"][0]
    lines += ["## Pipeline health", "",
              f"- {n_empty} of {edf.height} explainer responses came back empty (no usable body).",
              f"- {parsed:.4f} of scorer batches parsed; unparsed batches are DROPPED, never "
              f"imputed, so a feature's balanced accuracy is over the items that were actually "
              f"answered (`n_items` in scores.jsonl).",
              f"- mean marked fraction of a shown explainer example: "
              f"{binfo['mean_marked_fraction']:.4f}; token-join mismatches "
              f"{binfo['token_join_mismatches']}.",
              f"- build flags: {len(binfo['flags'])}"
              + (f" -- first: {binfo['flags'][0]}" if binfo["flags"] else ""),
              ""]

    # ---- full-run projection ------------------------------------------------------------------
    per_feat_arm = {a: d["cost"] / max(1, n_feat) for a, d in costs["per_arm"].items()}
    four = sum(per_feat_arm.get(a, 0.0) for a in ("C16", "C4", "M", "C4M"))
    two = sum(per_feat_arm.get(a, 0.0) for a in ("M", "C4M"))
    lines += ["## Projected cost of the full run", "",
              "From this pilot's MEASURED $/feature/arm, both scorers included:", ""]
    lines += md_table(
        [["512 features x 4 arms (C16, C4, M, C4M), primary MAEMM", f"${four * 512:.2f}"],
         ["512 features x 2 arms (M, C4M), `rlI-150` secondary", f"${two * 512:.2f}"],
         ["both", f"${(four + two) * 512:.2f}"]],
        ["scope", "projected"],
    )
    lines += ["",
              "The projection assumes the pilot's prompt sizes, which are the real ones: the "
              "example sets and the test set do not grow with the number of features.",
              ""]

    dest = Path(out) if out else HERE / "pilot.md"
    dest.write_text("\n".join(lines) + "\n")
    console.print(f"[green]wrote {dest}[/green] ({dest.stat().st_size} B)")
    if vol.missing:
        console.print(f"[yellow]missing from the volume: {vol.missing}[/yellow]")


if __name__ == "__main__":
    app()
