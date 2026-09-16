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
  * (C16+M16) - C32  the MATCHED-N enrichment test (amendment A8)
  * the N points on C16 and M, descriptive only (A9)
  * the same, per density quartile
  * win fractions and the whole distribution, because the outcome is bimodal and a mean misleads
  * fire fraction as the covariate that separates the hard stratum
  * the floor arm R-shuffled, which should sit at 0.5, and TWO nulls -- the same description
    scored twice on the same items (judge-only), and on a second disjoint draw (judge + draw) --
    reported beside every win fraction

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

# The comparisons the design names, as (label, arm_a, arm_b) meaning a - b. The first three are
# the headline; the matched-N pair is amendment A8; the N points are descriptive only (A9).
CONTRASTS = [
    ("substitution", "M", "C16"),
    ("enrichment", "C4M", "C4"),
    ("cheap corpus", "C4", "C16"),
    ("matched-N enrichment", "C16M16", "C32"),
    ("corpus N: 8 - 16 (descriptive)", "C16-N8", "C16"),
    ("corpus N: 32 - 16 (descriptive)", "C32", "C16"),
    ("maemm N: 8 - 16 (descriptive)", "M-N8", "M"),
    ("maemm N: 32 - 16 (descriptive)", "M-N32", "M"),
]
# Amendment A7: the null is a SECOND, DISJOINT test draw scored with C16's own description, not a
# temperature-0 repeat. Its per-feature difference is the test-set sampling noise every contrast is
# exposed to, and it is reported beside every win fraction.
# Two nulls, both scorer-only. `C16-judge2` is the SAME description on the SAME draw-1 items,
# scored a second time: the JUDGE-ONLY floor, which exists because the Anthropic Messages API has
# no temperature parameter for this model generation and nothing is deterministic. `C16-draw2` is
# the same description on the second disjoint draw (A7): judge AND test-set-draw variation
# together. The difference between them is the draw half.
NULLS = [
    ("judge-only null (same description, same items, scored twice)", "C16", "C16-judge2"),
    ("draw null (same description, second disjoint test draw)", "C16", "C16-draw2"),
]
NULL = NULLS[1]
FLOOR_ARM = "R-shuffled"
JUDGE_NULL_ARM = "C16-judge2"
# One metric, three views of the SAME scorer answers: the pooled balanced accuracy, and the two
# restrictions of the negative half that amendment A5 created (10 zero-activation randoms + 10
# near-miss windows). A result that lives entirely on one half cannot hide in the pooled number.
METRICS = ("bal_acc",)
NEG_VIEWS = ("bal_acc", "bal_acc_zero_neg", "bal_acc_nearmiss_neg")
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


def paired(df: pl.DataFrame, a: str, b: str, scorer: str, metric: str = "bal_acc"):
    """(features, d) -- the per-feature difference arm `a` minus arm `b` for one scorer."""
    sub = df.filter(pl.col("scorer") == scorer)
    wide = (
        sub.filter(pl.col("arm").is_in([a, b]))
        .pivot(values=metric, index="feature", on="arm")
        .drop_nulls()
    )
    if a not in wide.columns or b not in wide.columns or not len(wide):
        return np.zeros(0, dtype=int), np.zeros(0)
    return wide["feature"].to_numpy(), (wide[a] - wide[b]).to_numpy()


def md_table(rows: list[list[str]], header: list[str]) -> list[str]:
    """A GitHub-flavoured table. Cells are escaped: a bare `|` (as in `mean |diff|`) would
    otherwise be read as a column separator and shear the row."""

    def esc(x):
        return str(x).replace("|", "\\|")

    out = ["| " + " | ".join(esc(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return out


@app.command()
def main(
    run: Annotated[str, typer.Option(help="the run directory name under /vol/runs/")],
    out: Annotated[str, typer.Option(help="the markdown file to write")] = "",
    label: Annotated[str, typer.Option(help="`pilot` or `results`; picks the default filename")] = "results",
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
        f"# Autointerp {label} -- `{run}`",
        "",
        f"Delphi-style SAE autointerp on base `{binfo['base']}`, SAE `{binfo['sae']}`, held-out "
        f"set `{binfo['set']}`, MAEMM `{binfo['maemm']}` (engine `{binfo['engine']}`). "
        f"Explainer = scorer = `{costs['model']}` through the **{costs.get('api', 'anthropic-messages')}** "
        f"API on the `{costs.get('path', '?')}` path. Temperature: {costs['temperature']}. "
        f"Metric: per-feature **balanced accuracy** of the scorer using the "
        f"explainer's description, over a test set of {binfo['n_pos']} gate-passing positives "
        f"(band-stratified) and {binfo['n_neg']} negatives "
        f"({binfo['n_neg'] - binfo.get('n_neg_nearmiss', 0)} zero-activation + "
        f"{binfo.get('n_neg_nearmiss', 0)} near-miss), IDENTICAL across arms and never shown to "
        f"any explainer. Two disjoint draws per feature; arms are scored on draw 1.",
        "",
        f"**n = {n_feat} features** ({n_feat / 4:.0f} per density quartile, seed "
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
            f"{tc.get('cache_read', 0):,}",
            f"${tc['cost']:.4f}",
            f"${tc['cost'] / max(1, n_feat):.4f}",
            "-",
        ]] + [
            [f"`{a}`", f"{d['calls']:,}", f"{d['in']:,}", f"{d['out']:,}",
             f"{d.get('cache_read', 0):,}", f"${d['cost']:.4f}",
             f"${d['cost'] / max(1, n_feat):.4f}", "/".join(d.get("paths", []) or ["-"])]
            for a, d in costs["per_arm"].items()
        ],
        ["arm", "calls", "input tok", "output tok", "cache-read tok", "cost", "$/feature", "path"],
    )
    lines += ["", "Per stage:", ""]
    lines += md_table(
        [[f"`{a}`", f"{d['calls']:,}", f"{d['in']:,}", f"{d['out']:,}", f"${d['cost']:.4f}"]
         for a, d in costs.get("per_stage", {}).items()],
        ["stage", "calls", "input tok", "output tok", "cost"],
    )
    lines += [
        "",
        f"Cost is COMPUTED from the returned token counts at "
        f"{costs.get('rates_usd_per_mtok')} $/MTok (the Anthropic API returns no cost field); the "
        f"batch path pays {costs.get('batch_discount', 0.5):.0%} of that. "
        f"{costs['cache_hits']:,} of {costs['cache_hits'] + costs['cache_misses']:,} calls came "
        f"from the prompt cache. Projection recorded at the probe gate: "
        + (f"${costs['projection_usd']:.2f}" if costs["projection_usd"] else "n/a (not reached)")
        + f" against a ${costs.get('max_cost_usd', 0):.2f} cap"
        + (f"; STOPPED EARLY: {costs['stopped_at']}." if costs.get("stopped_at") else "."),
        "",
    ]

    # ---- per-arm levels -------------------------------------------------------------------
    lines += ["## Test set and the two negative halves", "",
              "A test positive is a window whose peak pre-gate activation EXCEEDS THE GATE "
              "(amendment A1); without that rule, MEASURED on feature 845, 17 of 20 band-drawn "
              "positives sat below the gate on text unrelated to the feature and every arm landed "
              "near 0.6 balanced accuracy whatever its description said. The 20 negatives are 10 "
              "zero-activation windows from the 2048-window random pool plus 10 near-miss windows "
              "(0 < peak <= gate) (A5). Both halves are reported separately below, from the same "
              "scorer answers, because they are not the same test.", ""]
    tpr = df.filter(pl.col("arm") != FLOOR_ARM).group_by("scorer").agg(
        pl.col("tpr").mean().alias("tpr"), pl.col("tnr").mean().alias("tnr"),
        pl.col("tnr_zero").mean().alias("tnr_zero"),
        pl.col("tnr_nearmiss").mean().alias("tnr_nm"),
        pl.col("n_pos").mean().alias("np"), pl.col("n_neg_nearmiss").mean().alias("nnm"),
    )
    lines += md_table(
        [[r["scorer"], f"{r['tpr']:.4f}", f"{r['tnr']:.4f}", f"{r['tnr_zero']:.4f}",
          f"{r['tnr_nm']:.4f}", f"{r['np']:.1f}", f"{r['nnm']:.1f}"]
         for r in tpr.iter_rows(named=True)],
        ["scorer", "mean TPR", "mean TNR (all)", "TNR on zero-activation", "TNR on near-miss",
         "positives/feature", "near-miss negatives/feature"],
    )
    lines += [""]

    lines += ["## Per-arm balanced accuracy", ""]
    rows = []
    for view in NEG_VIEWS:
        for scorer in scorers:
            for a_ in arms:
                v = df.filter((pl.col("scorer") == scorer) & (pl.col("arm") == a_))[view]
                v = np.asarray([x for x in v.to_list() if x is not None], dtype=float)
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                m, lo, hi = boot_ci(v)
                ne = df.filter((pl.col("scorer") == scorer) & (pl.col("arm") == a_))["n_examples"]
                rows.append([
                    f"`{view}`", scorer, f"`{a_}`", f"{float(np.mean(ne.to_numpy())):.1f}", len(v),
                    f"{m:.4f} [{lo:.4f}, {hi:.4f}]",
                    *[f"{np.quantile(v, q):.3f}" for q in QUANTS],
                    f"{float((v <= 0.5 + 1e-9).mean()):.3f}",
                ])
    lines += md_table(
        rows,
        ["negatives", "scorer", "arm", "mean N shown", "n", "mean [95% CI]",
         *[f"q{int(q * 100)}" for q in QUANTS], "frac <= 0.5"],
    )
    lines += [
        "",
        f"`frac <= 0.5` is the fraction of features on which the description is no better than "
        f"chance -- the bimodality the design warns about, which a mean alone hides. `mean N "
        f"shown` is the arm's ACTUAL example count averaged over features. **`{FLOOR_ARM}` is the "
        f"floor** (amendment A6): each feature's test set scored with a DIFFERENT feature's C16 "
        f"description under a fixed derangement. It should sit at 0.5; how far it sits above 0.5 "
        f"is how much of every other arm's number is available without knowing anything about the "
        f"feature. **`C16-draw2`** is C16's own description on the second, disjoint test draw "
        f"(A7) -- the null.",
        "",
    ]

    null_txt: dict[tuple[str, str], tuple] = {}
    lines += ["## The two nulls", ""]
    rows = []
    for label, a_, b_ in NULLS:
        for metric in METRICS:
            for scorer in scorers:
                _f, d = paired(df, a_, b_, scorer, metric)
                if not len(d):
                    continue
                m, lo, hi = boot_ci(d)
                p, win, _m_non = sign_test(d)
                q90 = float(np.quantile(np.abs(d), 0.90))
                if b_ == NULL[2]:
                    null_txt[(metric, scorer)] = (m, lo, hi, float(np.abs(d).mean()), q90, win,
                                                  len(d))
                rows.append([
                    label.split(" (")[0], f"`{metric}`", scorer, f"`{a_}` - `{b_}`", len(d),
                    ci_str(m, lo, hi), f"{float(np.abs(d).mean()):.4f}", f"{q90:.4f}",
                    f"{win:.3f}" if np.isfinite(win) else "-",
                    f"{p:.3f}" if np.isfinite(p) else "-",
                ])
    lines += md_table(
        rows,
        ["null", "metric", "scorer", "contrast", "n", "mean diff [95% CI]", "mean |diff|",
         "q90 |diff|", "win frac", "sign p"],
    )
    lines += [
        "",
        "Both nulls score the SAME C16 description with no new explainer call. The **judge-only** "
        "null re-scores the SAME draw-1 items, so its spread is the scorer's own run-to-run "
        "variation -- which is real rather than zero, because the Anthropic Messages API has no "
        "`temperature` parameter for this model generation and nothing is deterministic. The "
        "**draw** null scores the second, disjoint draw, so it carries that variation PLUS "
        "test-set sampling; the gap between the two is the sampling half. Each should have a mean "
        "difference of 0 and a win fraction of 0.5; the draw null's `mean |diff|` and `q90 |diff|` "
        "are what every contrast below is read against.",
        "",
    ]

    # ---- contrasts --------------------------------------------------------------------------
    lines += ["## Paired contrasts (features as the unit, percentile bootstrap, B = "
              f"{N_BOOT:,})", ""]
    rows = []
    for metric in METRICS:
        for scorer in scorers:
            nul = null_txt.get((metric, scorer), (0, 0, 0, float("nan"), float("nan"), 0.5, 0))
            floor, nq90, nwin = nul[3], nul[4], nul[5]
            for label, a, b in CONTRASTS:
                _f, d = paired(df, a, b, scorer, metric)
                if not len(d):
                    continue
                m, lo, hi = boot_ci(d)
                p, win, _m = sign_test(d)
                clear = "yes" if np.isfinite(lo) and (lo > 0 or hi < 0) else "no"
                over = "yes" if np.isfinite(nq90) and abs(m) > nq90 else "no"
                rows.append([
                    f"`{metric}`", scorer, label, f"`{a}` - `{b}`", len(d), ci_str(m, lo, hi),
                    f"{win:.3f}" if np.isfinite(win) else "-",
                    f"{nwin:.3f}" if np.isfinite(nwin) else "-",
                    f"{p:.4f}" if np.isfinite(p) else "-",
                    f"{floor:.4f}" if np.isfinite(floor) else "-", clear, over,
                ])
    lines += md_table(
        rows,
        ["metric", "scorer", "contrast", "arms", "n", "mean diff [95% CI]", "win frac",
         "NULL win frac", "sign p", "null mean |diff|", "CI clears 0", "|diff| > null q90"],
    )
    lines += [""]

    # ---- per quartile -----------------------------------------------------------------------
    lines += ["## Per density quartile", "",
              "Quartile 0 is the rarest quarter of features by gated corpus density, 3 the "
              "commonest (`precompute/targets.py:229-252`).", ""]
    strat = {int(r["feature"]): int(r["stratum"]) for r in feats["features"]}
    rows = []
    for metric in METRICS:
        for scorer in scorers:
            for label, a, b in CONTRASTS[:4]:
                for q in sorted(set(strat.values())):
                    f_ids, d = paired(df, a, b, scorer, metric)
                    keep = np.asarray([strat[int(x)] == q for x in f_ids])
                    dq = d[keep]
                    if not len(dq):
                        continue
                    m, lo, hi = boot_ci(dq)
                    _p, win, _m = sign_test(dq)
                    rows.append([f"`{metric}`", scorer, label, q, len(dq), ci_str(m, lo, hi),
                                 f"{win:.3f}" if np.isfinite(win) else "-"])
    lines += md_table(
        rows,
        ["metric", "scorer", "contrast", "quartile", "n", "mean diff [95% CI]", "win frac"],
    )
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
    for metric in METRICS:
        for scorer in scorers:
            f_ids, d = paired(df, "M", "C16", scorer, metric)
            if not len(d):
                continue
            fv = np.asarray([fire[int(x)] for x in f_ids])
            r = float(np.corrcoef(fv, d)[0, 1]) if len(d) > 2 else float("nan")
            for lo_e, hi_e in zip(edges[:-1], edges[1:], strict=True):
                keep = (fv >= lo_e) & (fv < hi_e)
                if not keep.any():
                    continue
                m, lo, hi = boot_ci(d[keep])
                rows.append([f"`{metric}`", scorer, f"[{lo_e:.2f}, {hi_e:.2f})", int(keep.sum()),
                             f"{fv[keep].mean():.3f}", ci_str(m, lo, hi)])
            lines.append(f"Pearson r(fire fraction, `M` - `C16`) = **{r:.3f}** on {scorer}, "
                         f"`{metric}` (n = {len(d)}).")
            lines.append("")
    lines += md_table(rows, ["metric", "scorer", "fire fraction bin", "n", "mean fire frac",
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
              f"- A10 explainer truncation: "
              f"{costs.get('explainer_truncated_and_retried', 0)} answers hit max_tokens and were "
              f"retried once at double the budget; a still-truncated answer raises.",
              f"- A12 model check: {costs.get('model_check', {})}",
              f"- per-stage API path and wall: {costs.get('stage_info', {})}",
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

    dest = Path(out) if out else HERE / f"{label}.md"
    dest.write_text("\n".join(lines) + "\n")
    console.print(f"[green]wrote {dest}[/green] ({dest.stat().st_size} B)")
    if vol.missing:
        console.print(f"[yellow]missing from the volume: {vol.missing}[/yellow]")


if __name__ == "__main__":
    app()
