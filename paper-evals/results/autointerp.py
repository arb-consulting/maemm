#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9"]
# ///
"""Eval 2's tables and figures: the SAE autointerp comparison, from the runs already on the volume.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm/paper-evals/results/autointerp.py --sae qwen36-27b/sae2m \\
       --run rl-last16=2026-09-22_autointerp-e2-2m-rl16 \\
       --run old-primary=2026-09-22_autointerp-e2-2m-old \\
       --run nla=2026-09-22_autointerp-e2-2m-nla)

Local, CPU, no GPU, no model, no API. Every number is READ from the `scores.jsonl` that
`autointerp/run.py` wrote; this file recomputes nothing except the aggregation across features,
the bootstrap intervals, and -- as a CHECK -- the balanced accuracies the rows already carry.

WHY MORE THAN ONE RUN DIRECTORY. `run` is per CHECKPOINT: the `M` arm of `rl-last16` and the `M`
arm of the old primary are two runs, and each run directory carries its own copy of every corpus
arm. Eval 2 is therefore SIX run directories -- three checkpoints on each of two SAEs -- and this
driver is invoked once per SAE with one `--run <label>=<dir>` per checkpoint. The unit of
comparison is consequently the pair `(<run label>, <arm>)` and never the bare arm name.

WHETHER TWO LABELS ARE TWO MEASUREMENTS DEPENDS ON THE CACHE, AND THE DRIVER DOES NOT GUESS.
`run.py`'s cache key is `sha256({"job": <job key>, "body": <request body>})` with no run or build
component, so three runs launched with ONE shared `--cache-dir` replay each other's corpus-arm
calls and their `DOCMAX` rows are one measurement reproduced three times, not three. That is how
the 2026-09-21 eval-2 runs were launched (`--cache-dir .../runs/e2-cache-{2m,131k}`), and it is
why every `DOCMAX` row is identical across the three labels of a block and the paired contrast
`DOCMAX(old-primary) - DOCMAX(rl-last16)` is exactly +0.0000 [+0.0000, +0.0000]. Run them with
separate cache directories and the same three rows become three independent draws of the explainer
and the judge, and the same table reads differently. The driver keeps the pair as the unit either
way -- merging the labels would be wrong under separate caches, and the +0.0000 contrast is the
diagnostic that says which regime produced the numbers in front of you.

WHAT IT BUILDS, per SAE:

  bal_acc      mean balanced accuracy over features per (arm x scorer), with a PERCENTILE
               BOOTSTRAP CI over features -- `autointerp/stats.boot_ci`'s estimator, reimplemented
               here (see "why not import" below). Detection and fuzzing are separate columns and
               are NEVER pooled: fuzzing marks at the gate and, on the legacy protocol, runs
               zero-shot, so the two are comparable ACROSS ARMS and not to each other.

  contrasts    each arm's per-feature difference against `--ref` (default the corpus ground-truth
               arm `DOCMAX`), with a paired percentile CI over features, the win fraction, and --
               plan §3.3 -- the PAIRING ITSELF: the two feature sets are intersected and every
               feature either side loses is counted into the table and into `results.json`. An arm
               that covers fewer features than the reference is a real property of the design; an
               arm that de-pairs silently is a defect.

  strata       mean bal_acc per (arm x scorer x stratum) on the DRAW's own rarity quartile, which
               `run.py` copies onto every score row. Every arm and BOTH scorers, on EITHER SAE:
               `--no-strata` turns the block off, it does not make it primary-only. The two SAEs'
               strata are the same rank in each dictionary's own rarity ordering and are NOT the
               same rarity -- the caption quotes each set's draw record off the volume for the
               statistic, the pool and the cut points, and never retypes a number.

  peak_strata  the same table on a SECOND axis: quartiles of `corpus_peak`, the feature's peak
               activation over the 16M corpus scan. A POST-HOC cut of the features this block
               analysed -- the cuts are computed here and printed in the caption -- and the caption
               says so, because the rarity strata were balanced by the draw and this one was not.
               `--no-peak-strata` turns it off.

  trends       per (view x arm x scorer): the spread across cells, Spearman's rho over the cells as
               a DESCRIPTION OF SHAPE, and a label-permutation p on the spread. At 8 features per
               cell almost nothing separates and the table says which of it does; the rho is
               explicitly not a test, because four cells cap its exact two-sided p at 2/24 = 0.083.
               A cell below MIN_CELL features is excluded here and named, never silently averaged.

  support      per (arm x scorer): features, how many had no explanation, the mean item count, the
               batch parse rate, the role, the draw and `n_examples`. UNEQUAL N IS SHOWN, never
               smoothed: the NLA arm shows 4 examples against C16's 16 (`build.ARM_SPECS`,
               `NLA_N = 4`), and `run.py` writes `n_examples: 0` for every scorer-only pseudo-arm
               (`R-shuffled`, `C16-judge2`, `C16-draw2`, `NLA-desc`) because those names are not
               keys of the build's per-feature arm dict. Both show up here as themselves.

  checks       the reader check and the pairing check, below.

  sanity       optional, `--sanity <yaml>`, the same selector idiom `results/sanity.yaml` uses for
               eval 1. An absent file is a NOTE, not an error: eval 2 has no recorded expectations
               to gate against yet, and inventing some here would be inventing data.

READER CHECK. There is no array to re-reduce the way eval 1 re-reduces `cos.f16`, but the rows
carry their own components: `run.rates` defines balanced accuracy as `0.5 * (TPR + TNR)`, and the
negative-half views restrict only the negative side (`run.py:1352-1358`), so `bal_acc_zero_neg` is
`0.5 * (TPR + TNR_zero)` and `bal_acc_nearmiss_neg` is `0.5 * (TPR + TNR_nearmiss)` -- the
positives are in both restrictions by construction. Recomputing those three from `tpr`/`tnr`/
`tnr_zero`/`tnr_nearmiss` and comparing with the stored numbers is a comparison of two independent
quantities, so a mismatch is a defect in this reader or in that product, never a finding. The
tolerance is the 6-decimal rounding `run._nr` applies to each side.

PAIRING CHECK. Per (arm, scorer) against the reference: how many features each side has, how many
survive the intersection, how many were lost. A complete pairing prints as such; a reduced one is
called REDUCED and carries its count into `results.json`.

WHY NOT IMPORT `autointerp.stats`. It is the IN-CHAIN renderer -- it runs beside the product that
wrote the scores, on that run's own directory, and its constants (`N_BOOT = 10000`, seed
`20260916`, the `CONTRASTS` list) are that chain's. This is the paper's driver, on the `results/`
lifecycle: it reads whatever run directories it is given, joins across them, and takes its
resample count and seed from `results/common` so every table in the paper is bootstrapped the same
way. The METHOD is the same paired-over-features percentile bootstrap and is deliberately so.

NOTHING IS KEYED ON AN ARM NAME. The arms come from the rows' own `arm` field, the roles from
their `role` field, the strata from their `stratum`. `--ref` is the one arm this file names, and
it is a flag with a default, not a constant.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.common as R  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# `run._nr(x, nd=6)` rounds every stored rate to six decimals, so a recomputation of
# `0.5 * (tpr + tnr)` from the two rounded components can sit up to 0.5*(5e-7 + 5e-7) + 5e-7 = 1e-6
# from the stored, independently rounded `bal_acc`. The bound is ABSOLUTE and can be, because
# every quantity here is a rate in [0, 1] -- unlike eval 1's activations, whose scale forced a
# relative bound.
ROUND_EPS = 1e-6
READER_TOL = 2e-6
# Chance for a balanced accuracy over a balanced-by-construction test set. Drawn on every figure
# because an arm below it is not a weak explanation, it is an anti-correlated one.
CHANCE = 0.5
ALPHA = 0.05
# The three balanced accuracies a row carries, each as (stored, TPR field, TNR field). The two
# restrictions of the negative half are amendment A5's, from the SAME answers: a result that lives
# entirely on one half cannot hide in the pooled number, and the reader check covers all three.
BAL_ACCS = (
    ("bal_acc", "tpr", "tnr"),
    ("bal_acc_zero_neg", "tpr", "tnr_zero"),
    ("bal_acc_nearmiss_neg", "tpr", "tnr_nearmiss"),
)
# The fields a row must carry to be usable at all. Anything else is optional and read with `.get`.
REQUIRED = ("feature", "arm", "scorer")
# A cut cell below this many features is called SHORT: its mean is still printed (nothing here is
# smoothed away) but it is labelled in the table, counted into a check, and kept OUT of the trend
# statistics below. Four is half the design's eight, and it is where a percentile bootstrap stops
# being one: with three values or fewer both tails of the interval sit on a single feature, so the
# cell's contribution to a spread is one observation wearing a mean's clothes.
MIN_CELL = 4
# The two cut views, kept apart everywhere -- two tables, two registry prefixes, two trend blocks.
# `stratum` is the DRAW's own pre-registered rarity quartile, carried on every score row by the
# set that was drawn on it. `peak` is cut HERE, post hoc, over the features this block happens to
# have analysed. One is a property of the design and the other a property of the sample, and a
# reader who read the second as the first would credit the draw with a balance it never enforced.
STRATUM_VIEW = "stratum"
PEAK_VIEW = "peak"
# The per-row field each view cuts on. `stratum` is already an integer bucket; `corpus_peak` is a
# continuous magnitude and is quartiled below.
VIEW_FIELD = {STRATUM_VIEW: "stratum", PEAK_VIEW: "corpus_peak"}
# What each view calls one of its cells, in every table and every check row.
CUT_WORD = {STRATUM_VIEW: "stratum", PEAK_VIEW: "quartile"}


@dataclass(frozen=True)
class Arm:
    """One scored arm: an arm name AND the run directory's label, because the same arm name in two
    run directories is two measurements (two explainer calls, two judge calls, two test draws)."""

    run: str
    arm: str

    @property
    def label(self) -> str:
        return f"{self.run}/{self.arm}"

    @property
    def slug(self) -> str:
        return self.label.replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------------------------
# reading -- one run directory at a time, and every absent one is reported rather than zero-filled
# ---------------------------------------------------------------------------------------------


def parse_runs(specs: list[str] | None) -> dict[str, str]:
    """{label: run directory} from `--run <label>=<dir>`, repeated and/or comma-separated.

    Insertion-ordered on purpose: the FIRST entry is the reference run a bare `--ref` resolves in
    (see `resolve_ref`). A repeated label is refused rather than overwriting its predecessor,
    because the overwrite would silently drop a whole checkpoint from the tables.
    """
    out: dict[str, str] = {}
    for spec in specs or []:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            assert "=" in part, (
                f"--run wants `<label>=<run_dir>` (e.g. `--run rl-last16=2026-09-22_autointerp-e2`), "
                f"got {part!r}")
            lab, _, run_dir = part.partition("=")
            lab, run_dir = lab.strip(), run_dir.strip()
            assert lab and run_dir, f"--run {part!r} has an empty label or an empty directory"
            assert lab not in out, (
                f"--run label {lab!r} was given twice ({out[lab]!r} and {run_dir!r}); one label is "
                f"one checkpoint's run directory, and merging two under it would double-count")
            out[lab] = run_dir
    return out


def load_run(vol: R.Vol, label: str, run_dir: str) -> tuple[list[dict], dict, str]:
    """(the run's score rows tagged with their arm key, its `build.json`, "" or why it is unusable).

    `build.json` is provenance only -- which checkpoint, which SAE, which set, how the examples
    were marked -- and an absent one costs the run nothing but its provenance line.
    """
    rel = f"runs/{run_dir}/summary/scores.jsonl"
    raw = vol.jsonl(rel)
    if raw is None:
        return [], {}, f"{rel} is not there (root {vol.prefix or '/'})"
    rows = []
    for i, r in enumerate(raw):
        missing = [k for k in REQUIRED if r.get(k) is None]
        assert not missing, (
            f"{rel} line {i + 1} carries no {', '.join(missing)}: `scores.jsonl` is one row per "
            f"(feature, arm, scorer) and cannot be read without all three")
        rows.append({**r, "run": label, "run_dir": run_dir,
                     "key": Arm(label, str(r["arm"])), "scorer": str(r["scorer"])})
    build = vol.json(f"runs/{run_dir}/summary/build.json") or {}
    return rows, build, ""


def index_rows(rows: list[dict]) -> dict[tuple[Arm, str], dict[int, dict]]:
    """{(arm, scorer): {feature: its row}} -- `scores.jsonl` is ONE row per (feature, arm, scorer).

    A duplicate triple means two run directories were read under one label, or one summary was
    concatenated with another's. Either way every mean below would double-count that feature, so
    it is refused here rather than averaged into silence.
    """
    out: dict[tuple[Arm, str], dict[int, dict]] = {}
    for r in rows:
        cell = out.setdefault((r["key"], r["scorer"]), {})
        feat = int(r["feature"])
        assert feat not in cell, (
            f"two rows for (feature {feat}, arm `{r['key'].label}`, scorer `{r['scorer']}`) in "
            f"`{r['run_dir']}`: one row per triple is the file's contract, so this is two runs "
            f"merged under one --run label, not a measurement")
        cell[feat] = r
    return out


def values_of(cell: dict[int, dict], metric: str = "bal_acc") -> dict[int, float]:
    """{feature: a FINITE metric} for one (arm, scorer).

    `run.rates` returns NaN wherever a class was absent from the scored items and `run._nr` turns
    that into `null`, which happens for real whenever every batch of a (feature, arm) went
    unparsed. An absent measurement is DROPPED from the mean and counted in `support`, never
    imputed as a zero and never as a chance-level 0.5.
    """
    out: dict[int, float] = {}
    for feat, row in cell.items():
        v = row.get(metric)
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            out[feat] = v
    return out


# ---------------------------------------------------------------------------------------------
# estimation -- the paired percentile bootstrap over FEATURES
# ---------------------------------------------------------------------------------------------


def boot_ci(d, n_boot: int = R.N_BOOT, seed: int = R.BOOT_SEED, alpha: float = ALPHA):
    """(mean, lo, hi): percentile bootstrap of the mean of `d`, resampling FEATURES.

    `autointerp/stats.boot_ci`'s method, on `results/`'s own resample count and seed. The unit is
    the feature everywhere in this eval -- the test set is identical across arms by construction,
    and so are the item order and the batching -- so a CI over ITEMS would be the wrong width and a
    paired difference is the right contrast.

    The same seed on every call is deliberate: the resample INDICES are then common across arms,
    which is what makes two arms' intervals on the same features comparable rather than two
    independent noise draws. Fewer than two finite values give a mean and no interval (NaN, never
    a zero-width one, which would read as certainty).
    """
    d = np.asarray(list(d), dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(int(n_boot), len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def paired_diff(a: dict[int, float], b: dict[int, float]) -> dict:
    """The per-feature difference `a - b` over the INTERSECTION of the two feature sets.

    Plan §3.3 asks for this intersection to be ASSERTED rather than left implicit.
    `autointerp/stats.paired` gets it right -- polars' `drop_nulls` on the pivot -- but says
    nothing about what it dropped, and an arm that quietly covers 24 of 32 features would then be
    compared on 24 under a heading that says 32. So the intersection is taken by feature id (never
    by zipping two arrays in whatever order their rows arrived), the alignment is asserted, and
    both sides' losses are carried out of here to be printed.
    """
    shared = sorted(set(a) & set(b))
    d = np.array([a[f] - b[f] for f in shared], dtype=float)
    assert len(d) == len(shared) and all(f in a and f in b for f in shared), (
        f"the paired difference is not aligned on the intersection: {len(d)} differences over "
        f"{len(shared)} shared features")
    return {
        "features": shared, "d": d,
        "n_paired": len(shared), "n_a": len(a), "n_b": len(b),
        # Named by WHICH SIDE HAS THEM, not by who lost them: the caller has to decide which of the
        # two is the arm and which is the reference, and a "lost_a" that meant "a is missing it"
        # reads exactly backwards to half its readers.
        "only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a)),
        "complete": len(shared) == len(a) == len(b),
    }


def resolve_ref(arms: list[Arm], ref: str, ref_run: str) -> tuple[Arm | None, str]:
    """(the reference arm of the contrasts, "" or why it did not resolve).

    A bare `--ref DOCMAX` names an ARM, and eval 2's run directories each carry their own DOCMAX,
    so the name alone does not identify one. The reference is therefore taken from the FIRST
    `--run` given: a stated convention, printed in the caption, rather than a pick among equals.
    `--ref <label>/<arm>` names one outright. Under a shared `--cache-dir` the copies are the same
    calls replayed and the choice does not matter; under separate caches it does, which is the
    reason this is a stated convention and not an arbitrary one.
    """
    if "/" in ref:
        hits = [a for a in arms if a.label == ref]
        where = ""
    else:
        hits = [a for a in arms if a.arm == ref and a.run == ref_run]
        where = f" in the reference run `{ref_run}` (the first --run given)"
    if len(hits) == 1:
        return hits[0], ""
    return None, (
        f"`--ref {ref}` matched {len(hits)} arms{where}; the arms present are "
        f"{', '.join(a.label for a in arms) or 'none'}. Name one as `<run label>/<arm>`.")


# ---------------------------------------------------------------------------------------------
# cuts -- the draw's rarity strata, and a post-hoc cut on activation magnitude
# ---------------------------------------------------------------------------------------------


def feature_field(rows: list[dict], field: str, cast=float) -> tuple[dict[int, object], list[int]]:
    """({feature: its `field`}, the features whose rows disagree about it).

    `stratum` and `corpus_peak` are properties of the FEATURE -- the draw's quartile and the 16M
    corpus peak it recorded -- and `run.py` copies each onto every score row of that feature, so
    all of a feature's rows across every arm and both scorers must carry one value. A feature whose
    rows disagree is two sets joined under one `--root`, and a quartile cut taken over that mixture
    would be cut on a distribution that exists nowhere; it is reported here rather than resolved by
    picking whichever row was read first.

    `cast` keeps the value's own kind: a stratum stays an `int` so the table's heading reads
    `stratum 0` and not `stratum 0.0`, and a corpus peak stays a `float` so it can be quartiled.
    """
    seen: dict[int, set] = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        seen.setdefault(int(r["feature"]), set()).add(cast(v))
    out = {f: next(iter(vs)) for f, vs in seen.items() if len(vs) == 1}
    return out, sorted(f for f, vs in seen.items() if len(vs) > 1)


def quartile_buckets(per_feature: dict[int, float]) -> tuple[dict[int, int], list[float]]:
    """({feature: 0..3}, the three cut points) -- quartiles of the values GIVEN, nothing else.

    `np.quantile`'s linear interpolation on the analysed values, assigned with `searchsorted(...,
    side="right")` so a value sitting exactly on a cut falls in the HIGHER bucket. The cuts are the
    ANALYSED SET's own quartiles and are reported in the caption, because unlike the draw's strata
    nothing balanced this axis in advance: ties across a cut make the buckets unequal, which is why
    every cell below prints its own n rather than the n the count would imply.
    """
    if len(per_feature) < 4:
        return {}, []
    vals = np.array(sorted(per_feature.values()), dtype=float)
    cuts = [float(c) for c in np.quantile(vals, [0.25, 0.5, 0.75])]
    return ({f: int(np.searchsorted(cuts, v, side="right")) for f, v in per_feature.items()},
            cuts)


def _avg_ranks(v: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- `scipy.stats.rankdata`'s `average` on four numbers.

    Written out rather than imported: `results/` runs on numpy alone (see this file's `dependencies`
    block), and four cells do not justify a scipy dependency in every driver that imports this one.
    """
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spread_perm_p(vals: np.ndarray, buckets: np.ndarray, n_perm: int, seed: int) -> float:
    """P(spread of a random relabelling >= the observed spread), cell SIZES held fixed.

    The null is that the cut carries no information about this arm's per-feature accuracy: the
    bucket labels are shuffled over the arm's own features, so the cells keep their sizes and the
    marginal distribution of the values is exactly the observed one. Nothing is assumed normal,
    which matters because a per-feature balanced accuracy on 40 items is a lattice of multiples of
    1/40 and is nowhere near normal at n = 8.

    The statistic is max − min ACROSS CELLS, which is already the maximum over cells: its null
    absorbs "whichever cell turned out extreme" and so needs no further correction WITHIN a row.
    It does not absorb the choice of (arm, scorer), which the table's note states and corrects for.

    `(hits + 1) / (n_perm + 1)` -- the add-one form, so the p-value can never be reported as 0 and
    is never smaller than the resolution the resample count actually bought.
    """
    uniq = np.unique(buckets)
    if len(uniq) < 2 or len(vals) < 2:
        return float("nan")
    # Sort once so each cell is a contiguous block; permuting the VALUES is then the same null as
    # permuting the labels, and every resample's cell means are one `reduceat` rather than a loop.
    order = np.argsort(buckets, kind="stable")
    v, lab = vals[order], buckets[order]
    starts = np.searchsorted(lab, uniq)
    sizes = np.diff([*starts, len(v)]).astype(float)
    obs_means = np.add.reduceat(v, starts) / sizes
    obs = float(obs_means.max() - obs_means.min())
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.tile(v, (int(n_perm), 1)), axis=1)
    means = np.add.reduceat(perm, starts, axis=1) / sizes
    hits = int(np.sum((means.max(axis=1) - means.min(axis=1)) >= obs - 1e-12))
    return (hits + 1) / (int(n_perm) + 1)


def cut_cells(by_cell: dict[tuple[Arm, str], dict[int, dict]], arms: list[Arm],
              scorers: list[str], bucket_of: dict[int, int], view: str,
              boot: int, seed: int) -> list[dict]:
    """One row per (arm, scorer, cell): the mean, its bootstrap interval, its n, and SHORT.

    The interval is the same percentile bootstrap over features the headline table uses, so a cell
    and a whole-arm number are the same estimator at two sample sizes and can be read against each
    other. At the eval's 8 features per cell it is wide and is meant to be: printing the mean alone
    invites a reader to compare four cells that a 0.2-wide interval cannot separate.
    """
    out: list[dict] = []
    for key in arms:
        for scorer in scorers:
            cell = by_cell.get((key, scorer))
            if not cell:
                continue
            buckets: dict[object, list[float]] = {}
            for feat, val in values_of(cell).items():
                b = bucket_of.get(feat)
                buckets.setdefault(b, []).append(val)
            for bucket in sorted(buckets, key=lambda b: (b is None, str(b))):
                vals = buckets[bucket]
                mean, lo, hi = boot_ci(vals, boot, seed)
                out.append({
                    "view": view, "arm": key.label, "run": key.run, "arm_name": key.arm,
                    "scorer": scorer, "stratum": bucket, "mean": mean, "lo": lo, "hi": hi,
                    "n": len(vals), "short": len(vals) < MIN_CELL})
    return out


def trend_rows(by_cell: dict[tuple[Arm, str], dict[int, dict]], arms: list[Arm],
               scorers: list[str], bucket_of: dict[int, int], cells: list[dict], view: str,
               n_perm: int, seed: int) -> list[dict]:
    """Per (arm, scorer): the spread across cells, a rank statistic, and a permutation p.

    THE HONEST SHAPE OF THIS. Three numbers, none of them a significance ritual:

      spread   max − min of the cell means. The effect size, in the units the table prints, and
               the only one of the three a reader can carry away without the machinery.
      rho      Spearman's correlation between the cut index and the cell mean over the cells. A
               DESCRIPTION OF SHAPE, never a test: with four cells there are 4! = 24 orderings, so
               the exact two-sided permutation p of a PERFECTLY monotone |rho| = 1 is 2/24 = 0.083
               and no trend at this design can reach 0.05 on this statistic. It is here to separate
               "rises across the cut" from "one cell differs", which the spread alone cannot.
      p_perm   `spread_perm_p` above: the spread against the null that the cut is uninformative.

    SHORT cells are excluded from all three and named in the row, because a spread whose extreme
    cell rests on two features is a spread between an estimate and an anecdote. The cell itself is
    still printed in the table with its own n -- excluded from the statistic, never from the table.
    """
    by_key = {(c["arm"], c["scorer"]): [] for c in cells}
    for c in cells:
        by_key[(c["arm"], c["scorer"])].append(c)
    out: list[dict] = []
    for key in arms:
        for scorer in scorers:
            mine = by_key.get((key.label, scorer))
            if not mine:
                continue
            used = [c for c in mine if not c["short"] and c["stratum"] is not None]
            dropped = [c for c in mine if c not in used]
            # The permutation runs on the features of the USED cells alone, so the null it draws
            # from is the same population the spread was computed over.
            keep = {c["stratum"] for c in used}
            vals, labs = [], []
            for feat, val in values_of(by_cell.get((key, scorer), {})).items():
                b = bucket_of.get(feat)
                if b in keep:
                    vals.append(val)
                    labs.append(b)
            means = np.array([c["mean"] for c in sorted(used, key=lambda c: c["stratum"])])
            idx = np.array([c["stratum"] for c in sorted(used, key=lambda c: c["stratum"])],
                           dtype=float)
            spread = float(means.max() - means.min()) if len(means) >= 2 else float("nan")
            # Undefined on fewer than three cells (two points always correlate perfectly) and on a
            # flat arm, where every ordering of equal means is as monotone as every other.
            rho = (float(np.corrcoef(_avg_ranks(idx), _avg_ranks(means))[0, 1])
                   if len(means) >= 3 and np.ptp(means) > 0 else float("nan"))
            out.append({
                "view": view, "arm": key.label, "run": key.run, "arm_name": key.arm,
                "scorer": scorer, "n_cells": len(used), "n_cells_all": len(mine),
                "n_min": min((c["n"] for c in used), default=0),
                "widest_ci": max((c["hi"] - c["lo"] for c in used
                                  if math.isfinite(c["hi"]) and math.isfinite(c["lo"])),
                                 default=float("nan")),
                "spread": spread, "rho": rho,
                # The cell means themselves, so the multiplicity count below can tell a DUPLICATED
                # measurement from an independent one. Under a shared `--cache-dir` the corpus arms
                # of two run directories are the same calls replayed, and their rows are identical
                # here to the last digit.
                "_means": tuple(round(float(m), 12) for m in means),
                "p_perm": spread_perm_p(np.array(vals, dtype=float), np.array(labs, dtype=int),
                                        n_perm, seed) if len(used) >= 2 else float("nan"),
                "dropped": [f"{c['stratum']} (n={c['n']})" for c in dropped],
            })
    return out


def cells_check(cells: list[dict], view: str) -> dict:
    """The under-filled cells of one view, as a check row.

    Two counters, not one. SHORT (`n < MIN_CELL`) is the hard one and changes what the trend table
    computes. UNDER-FILLED is every cell below the view's own fullest cell -- `run.py` omits a
    (feature, arm) whose every batch went unparsed, and `values_of` drops a null `bal_acc`, so an
    arm can lose features out of one stratum and not another. Neither is automatically a defect and
    both are invisible in a table of means alone, which is the thing this exists to prevent.
    """
    full = max((c["n"] for c in cells), default=0)
    short = [c for c in cells if c["short"]]
    thin = [c for c in cells if not c["short"] and c["n"] < full]
    # A cell that is not there AT ALL is the third way a cut can go wrong and the only one the
    # table renders as a bare em dash. It happens for real: `run.rates` returns NaN for a feature
    # with no gate-consistent positives (`n_pos: 0` -> no TPR -> no balanced accuracy), so an arm
    # can lose every feature of one cut cell and be left comparing three cells against another
    # arm's four. Counted here by (arm, scorer) against the buckets the VIEW as a whole shows.
    every = {c["stratum"] for c in cells}
    have: dict[tuple, set] = {}
    for c in cells:
        have.setdefault((c["arm"], c["scorer"]), set()).add(c["stratum"])
    absent = [(a, sc, b) for (a, sc), got in sorted(have.items(), key=lambda kv: str(kv[0]))
              for b in sorted(every - got, key=lambda b: (b is None, str(b)))]
    # The view's own word, not the field's: `stratum` is what the record calls the rarity cut and
    # the dict key that carries both, but a magnitude cell named "stratum 1" here would read as a
    # rarity stratum in the one table whose whole caption is that it is not one.
    word = CUT_WORD[view]

    def _name(c):
        return f"{c['arm']}/{c['scorer']} {word} {c['stratum']} (n={c['n']})"

    return {"kind": "cells", "who": view, "detail": f"{len(cells)} cells, fullest n={full}",
            "comparisons": len(cells), "n_mismatches": len(short),
            "worst_excess": float("nan"), "n_short": len(short), "n_thin": len(thin),
            "n_absent": len(absent), "full": full,
            "short_cells": [_name(c) for c in short],
            "thin_cells": [_name(c) for c in thin],
            "absent_cells": [f"{a}/{sc} {word} {b}" for a, sc, b in absent]}


def draw_record(vol: R.Vol, builds: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(the lines of each set's draw record that state its stratification, the sets with none).

    THE CUT POINTS ARE NOT IN THIS FILE AND MUST NOT BE. They are a property of a DRAW -- which
    pool, which statistic, which quantiles -- recorded beside the set on the volume by the product
    that drew it, and a number retyped into a caption here would go stale the first time a set is
    redrawn and would never be noticed, because nothing downstream compares the two. So the caption
    QUOTES the record: `build.json` names the base and the set, the set's `README.md` is
    `precompute`'s own account of the draw, and every `- ` note in it that mentions a stratification
    or a cut is carried into the caption verbatim with its source named. An absent README costs the
    caption its numbers and says which set's record was missing -- never a plausible default.
    """
    lines, absent = [], []
    for base, set_name in sorted({(str(b.get("base") or ""), str(b.get("set") or ""))
                                  for b in builds.values() if b.get("set")}):
        rel = f"base/{base}/heldout/{set_name}/README.md"
        p = vol.get(rel)
        if p is None:
            absent.append(f"`{set_name}`: no `{rel}` on the volume, so its cuts are unstated here")
            continue
        # `## Notes` only. The README's provenance block above it carries the rebuild command,
        # which on a stratified draw contains `--stratified` and would match every filter here
        # while stating no cut at all -- a caption line that looks like a record and is not one.
        body = p.read_text().partition("\n## Notes")[2]
        # Bullets are rejoined across their continuation lines before they are filtered: a note
        # that happens to have been wrapped would otherwise be quoted up to the wrap and no
        # further, and a caption that ends mid-sentence at "the 4 quartiles of the ELIGIBLE POOL
        # (40" reads as a record while stating none of the numbers it was quoted for.
        bullets: list[str] = []
        for ln in body.splitlines():
            if ln.startswith("- "):
                bullets.append(ln[2:].strip())
            elif bullets and ln.strip() and not ln.startswith(("#", "|")):
                bullets[-1] += " " + ln.strip()
        hits = [b for b in bullets if "strat" in b.lower() or "cuts" in b.lower()]
        if not hits:
            absent.append(f"`{set_name}`: `{rel}` records no stratification note")
        for ln in hits:
            lines.append(f"`{set_name}` ({rel}): {ln}")
    return lines, absent


# ---------------------------------------------------------------------------------------------
# checks -- both of them can go red, and the selftest proves it
# ---------------------------------------------------------------------------------------------


def reader_check(rows: list[dict], label: str, run_dir: str) -> dict:
    """Recompute each stored balanced accuracy from the row's own TPR/TNR and compare.

    `run.rates` returns `0.5 * (tpr + tnr)`, and the two A5 restrictions change only the negative
    side, so all three stored accuracies are determined by the rates stored beside them. Both sides
    are the same quantity, so anything outside the 6-decimal rounding both carry is a defect in
    this reader or in that product -- never a result. The range and count checks come along for the
    ride: a rate outside [0, 1] or more parsed batches than batches is a corrupt row, not a finding.
    """
    checked = mism = 0
    worst = -math.inf
    bad_range = bad_counts = 0
    for r in rows:
        for stored_k, tpr_k, tnr_k in BAL_ACCS:
            stored, tpr, tnr = r.get(stored_k), r.get(tpr_k), r.get(tnr_k)
            if stored is None or tpr is None or tnr is None:
                continue
            ours = 0.5 * (float(tpr) + float(tnr))
            diff = abs(ours - float(stored))
            worst = max(worst, diff - READER_TOL)
            mism += int(diff > READER_TOL)
            checked += 1
        for k in ("bal_acc", "bal_acc_zero_neg", "bal_acc_nearmiss_neg", "tpr", "tnr", "acc"):
            v = r.get(k)
            if v is not None and not (0.0 <= float(v) <= 1.0):
                bad_range += 1
        nb, npd = r.get("n_batches"), r.get("n_parsed")
        if nb is not None and npd is not None and not 0 <= int(npd) <= int(nb):
            bad_counts += 1
    if not checked:
        return {"kind": "reader", "who": label, "detail": run_dir,
                "skipped": f"{run_dir}: no (bal_acc, tpr, tnr) triple to compare"}
    return {"kind": "reader", "who": label, "detail": run_dir, "comparisons": checked,
            "n_mismatches": mism + bad_range + bad_counts, "worst_excess": round(worst, 9),
            "n_identity": mism, "n_out_of_range": bad_range, "n_bad_counts": bad_counts}


def pairing_check(contrast: dict, ref_label: str) -> dict:
    """One `(arm, scorer)` pairing against the reference, as a check row.

    Reduced pairing is not automatically a defect -- an arm built over a subset of the features is
    a design choice, and `scores.jsonl` omits a (feature, arm) whose every batch went unparsed --
    but it MUST be visible, because a contrast on 24 features under a heading that says 32 is the
    thing plan §3.3 asks to be made impossible.
    """
    n_paired, n_a, n_b = contrast["n_paired"], contrast["n_arm"], contrast["n_ref"]
    dropped = (n_a - n_paired) + (n_b - n_paired)
    return {"kind": "pairing", "who": contrast["arm"], "detail": contrast["scorer"],
            "comparisons": n_paired, "n_mismatches": dropped, "worst_excess": float("nan"),
            "n_arm": n_a, "n_ref": n_b, "ref": ref_label, "complete": dropped == 0}


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------


def analyse(vol: R.Vol, runs: dict[str, str], sae: str, ref: str, boot: int, seed: int,
            strata: bool, peak_strata: bool = True) -> dict:
    """Everything the tables and figures are built from. Never raises on a missing run directory."""
    assert runs, (
        "at least one `--run <label>=<run_dir>` is required: eval 2's arms live in one run "
        "directory per checkpoint, and there is nothing to read without being told which")
    rows: list[dict] = []
    builds: dict[str, dict] = {}
    missing: list[str] = []
    notes: list[str] = []
    checks: list[dict] = []
    for label, run_dir in runs.items():
        got, build, why = load_run(vol, label, run_dir)
        if why:
            missing.append(f"`{label}`: {why}")
            continue
        rows += got
        builds[label] = build
        checks.append(reader_check(got, label, run_dir))
        if not build:
            notes.append(f"`{label}` ({run_dir}): no `summary/build.json`, so its provenance "
                         f"(checkpoint, SAE, set) is unstated here")
        elif sae and str(build.get("sae") or "") and str(build["sae"]) != sae:
            # Not a note to skim past: a 131k run tabulated under the 2M heading would be a wrong
            # paper table, and every 131k feature id is also a valid 2M index, so nothing
            # downstream would notice on its own.
            notes.append(f"`{label}` ({run_dir}) was BUILT ON SAE `{build['sae']}`, not the "
                         f"`--sae {sae}` this invocation reports — its rows are in these tables")

    scorers = sorted({r["scorer"] for r in rows})
    order = {lab: i for i, lab in enumerate(runs)}
    arms = sorted({r["key"] for r in rows}, key=lambda k: (order.get(k.run, 99), k.arm))
    by_cell = index_rows(rows)
    colours = R.colour_map(arms)

    # --- per arm x scorer ---------------------------------------------------------------------
    cells: list[dict] = []
    for key in arms:
        for scorer in scorers:
            cell = by_cell.get((key, scorer))
            if not cell:
                continue
            vals = values_of(cell)
            mean, lo, hi = boot_ci(vals.values(), boot, seed)
            roles = sorted({str(r.get("role") or "") for r in cell.values()})
            n_ex = sorted({int(r.get("n_examples") or 0) for r in cell.values()})
            draws = sorted({int(r.get("draw") or 1) for r in cell.values()})
            parse = [int(r["n_parsed"]) / int(r["n_batches"]) for r in cell.values()
                     if r.get("n_batches")]
            items = [int(r["n_items"]) for r in cell.values() if r.get("n_items") is not None]
            cells.append({
                "arm": key.label, "run": key.run, "arm_name": key.arm, "scorer": scorer,
                "mean": mean, "lo": lo, "hi": hi,
                "n_features": len(vals), "n_rows": len(cell),
                "n_no_metric": len(cell) - len(vals),
                "n_no_explanation": sum(1 for r in cell.values()
                                        if r.get("explanation_ok") is False),
                "roles": roles, "n_examples": n_ex, "draws": draws,
                "mean_n_items": float(np.mean(items)) if items else float("nan"),
                "parse_rate": float(np.mean(parse)) if parse else float("nan"),
                "run_dir": runs.get(key.run, ""),
            })

    # --- the paired contrasts -----------------------------------------------------------------
    ref_run = next(iter(runs))
    ref_key, why = resolve_ref(arms, ref, ref_run)
    contrasts: list[dict] = []
    if ref_key is None:
        notes.append(f"no paired contrasts: {why}")
    else:
        for key in arms:
            if key == ref_key:
                continue
            for scorer in scorers:
                a = values_of(by_cell.get((key, scorer), {}))
                b = values_of(by_cell.get((ref_key, scorer), {}))
                if not a or not b:
                    continue
                pd = paired_diff(a, b)
                mean, lo, hi = boot_ci(pd["d"], boot, seed)
                wins = float(np.mean(pd["d"] > 0)) if len(pd["d"]) else float("nan")
                rec = {
                    "arm": key.label, "run": key.run, "arm_name": key.arm, "scorer": scorer,
                    "ref": ref_key.label, "mean": mean, "lo": lo, "hi": hi, "win_frac": wins,
                    "n_paired": pd["n_paired"], "n_arm": pd["n_a"], "n_ref": pd["n_b"],
                    # The reference's own features that this arm does not carry, and vice versa.
                    "lost_by_arm": pd["only_in_b"], "lost_by_ref": pd["only_in_a"],
                    "complete": pd["complete"],
                }
                contrasts.append(rec)
                checks.append(pairing_check(rec, ref_key.label))

    # --- per stratum: the DRAW's rarity quartile, carried on every score row ---------------------
    strat_rows: list[dict] = []
    trends: list[dict] = []
    if strata:
        strat_of, clashes = feature_field(rows, VIEW_FIELD[STRATUM_VIEW], int)
        if clashes:
            notes.append(f"{len(clashes)} feature(s) carry two different `stratum` values across "
                         f"their rows ({', '.join(str(f) for f in clashes[:8])}): they are left "
                         f"out of the per-stratum table, which is two sets joined under one --root")
        strat_rows = cut_cells(by_cell, arms, scorers, strat_of, STRATUM_VIEW, boot, seed)
        if strat_rows:
            trends += trend_rows(by_cell, arms, scorers, strat_of, strat_rows, STRATUM_VIEW,
                                 boot, seed)
            checks.append(cells_check(strat_rows, STRATUM_VIEW))
        else:
            notes.append("no per-stratum table: no score row carries a `stratum`")

    # --- per activation-magnitude quartile: cut HERE, over the analysed features -----------------
    peak_rows: list[dict] = []
    peak_cuts: list[float] = []
    if peak_strata:
        peak_of, clashes = feature_field(rows, VIEW_FIELD[PEAK_VIEW], float)
        if clashes:
            notes.append(f"{len(clashes)} feature(s) carry two different `corpus_peak` values "
                         f"({', '.join(str(f) for f in clashes[:8])}): they are left out of the "
                         f"magnitude quartiles, whose cuts would otherwise be cut on a mixture")
        qof, peak_cuts = quartile_buckets(peak_of)
        if qof:
            peak_rows = cut_cells(by_cell, arms, scorers, qof, PEAK_VIEW, boot, seed)
            trends += trend_rows(by_cell, arms, scorers, qof, peak_rows, PEAK_VIEW, boot, seed)
            checks.append(cells_check(peak_rows, PEAK_VIEW))
        else:
            notes.append(f"no magnitude-quartile table: {len(peak_of)} of the block's features "
                         f"carry a `corpus_peak`, and four are needed to cut quartiles at all")

    record, record_absent = draw_record(vol, builds)
    notes += record_absent

    return {
        "sae": sae, "runs": runs, "builds": builds, "root": vol.prefix or "/vol",
        "arms": arms, "scorers": scorers, "colours": colours, "ref": ref_key,
        "cells": cells, "contrasts": contrasts, "strata": strat_rows, "checks": checks,
        "peak_strata": peak_rows, "peak_cuts": peak_cuts, "trends": trends, "record": record,
        "n_features": len({int(r["feature"]) for r in rows}),
        "missing": missing, "notes": notes, "boot": boot, "seed": seed, "n_rows": len(rows),
    }


def stat_registry(res: dict) -> dict[tuple, float]:
    """{(scorer, arm, stratum, metric): value} -- what a `--sanity` gate selects against.

    Flat, like `faithfulness.stat_registry`, and for the same reason: a gate names a scorer, an
    arm, an optional stratum and a metric, and the YAML stays readable by whoever edits it without
    knowing this file's data structures. `diff.*` is always against `res["ref"]`, which the sanity
    table prints beside every row so the metric name does not have to carry it.
    """
    out: dict[tuple, float] = {}
    for c in res["cells"]:
        for part in ("mean", "lo", "hi"):
            out[(c["scorer"], c["arm"], None, f"bal_acc.{part}")] = c[part]
    for x in res["contrasts"]:
        for part in ("mean", "lo", "hi"):
            out[(x["scorer"], x["arm"], None, f"diff.{part}")] = x[part]
        out[(x["scorer"], x["arm"], None, "diff.win_frac")] = x["win_frac"]
    # The two cut views share the key's `stratum` slot and are told apart by the METRIC prefix:
    # `bal_acc.*` is the draw's rarity quartile and `peak.bal_acc.*` the post-hoc magnitude one,
    # because stratum 3 of one and stratum 3 of the other are different cells of different cuts
    # and a gate that could not name which it meant would silently resolve against either.
    for s in res.get("strata") or []:
        for part in ("mean", "lo", "hi"):
            out[(s["scorer"], s["arm"], s["stratum"], f"bal_acc.{part}")] = s[part]
    for s in res.get("peak_strata") or []:
        for part in ("mean", "lo", "hi"):
            out[(s["scorer"], s["arm"], s["stratum"], f"peak.bal_acc.{part}")] = s[part]
    for t in res.get("trends") or []:
        pre = "trend" if t["view"] == STRATUM_VIEW else "peak_trend"
        for part in ("spread", "rho", "p_perm"):
            out[(t["scorer"], t["arm"], None, f"{pre}.{part}")] = t[part]
    return out


def support_registry(res: dict) -> dict[tuple, int]:
    """{(scorer, arm, stratum): how many features are behind its statistics}.

    Carried into the sanity table because a gate written against 32 features and answered by a
    4-feature smoke is not a disagreement about the pipeline, and the reader has to see which.
    """
    out: dict[tuple, int] = {}
    for c in res["cells"]:
        out[(c["scorer"], c["arm"], None)] = c["n_features"]
    # The rarity view is written LAST so a gate on a shared (scorer, arm, stratum) key reads the
    # support of the cut `bal_acc.*` names; the magnitude view's own n is in `peak_strata.csv`.
    for s in (res.get("peak_strata") or []) + (res.get("strata") or []):
        out[(s["scorer"], s["arm"], s["stratum"])] = s["n"]
    return out


def run_sanity(res: dict, path: Path) -> list[dict]:
    """Resolve every gate in `--sanity` against the computed statistics. Flags, never stops.

    The file is OPTIONAL. Eval 2 has no recorded expectations to gate against yet, and shipping a
    YAML of invented numbers would be worse than shipping none: the selector idiom is here, ready
    for the run's own values once there are any, and an absent file says so in the notes.
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as fh:
        spec = yaml.safe_load(fh) or {}
    reg = stat_registry(res)
    support = support_registry(res)
    labels = [a.label for a in res["arms"]]
    out = []
    for chk in spec.get("checks") or []:
        rec = {"name": str(chk.get("name", "(unnamed)")), "expect": chk.get("expect"),
               "tol": chk.get("tol"),
               "provenance": " ".join(str(chk.get("provenance", "")).split()),
               "compare": chk.get("compare", True)}
        hits = [x for x in labels if str(chk.get("arm", "")) in x]
        if len(hits) != 1:
            rec["verdict"] = "absent"
            rec["why"] = (f"`arm: {chk.get('arm')}` matched {len(hits)} of the arms present "
                          f"({', '.join(labels) or 'none'})")
            out.append(rec)
            continue
        key = (str(chk.get("scorer")), hits[0], chk.get("stratum"), str(chk.get("metric")))
        rec["arm"], rec["scorer"], rec["metric"] = hits[0], key[0], key[3]
        if key not in reg:
            rec["verdict"] = "absent"
            rec["why"] = (f"no `{key[3]}` for scorer `{key[0]}` on `{hits[0]}` in this run (the "
                          f"scorer, the stratum or the contrast is not present)")
            out.append(rec)
            continue
        ours = reg[key]
        rec["ours"] = ours
        rec["n"] = support.get((key[0], key[1], key[2]))
        exp = float(chk["expect"])
        rec["diff"] = ours - exp
        if chk.get("compare") is False:
            rec["verdict"] = "no verdict"
            rec["why"] = "declared not the same statistic (`compare: false`)"
        else:
            tol = float(chk.get("tol", 0.05))
            if chk.get("tol_rel"):
                tol *= abs(exp)
            rec["tol"] = tol
            rec["verdict"] = "pass" if abs(ours - exp) <= tol else "FLAG"
            rec["why"] = f"|{ours:.4f} − {exp:.4f}| = {abs(ours - exp):.4f} vs tol {tol:g}"
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------


def ci(mean, lo, hi, places: int = 4, signed: bool = False) -> str:
    """`mean [lo, hi]` for a markdown cell, with the interval dropped where it is not estimable.

    `R.pm` renders `value ± se`, which is the wrong shape for a PERCENTILE interval over features:
    those need not be symmetric about the mean, and printing a half-width would assert a symmetry
    the estimator does not have.
    """
    if mean is None or not math.isfinite(float(mean)):
        return "—"
    head = f"{float(mean):+.{places}f}" if signed else R.num(mean, places)
    if lo is None or hi is None:
        return head
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return head
    sign = "+" if signed else ""
    return f"{head} [{lo:{sign}.{places}f}, {hi:{sign}.{places}f}]"


def _joined(vals) -> str:
    return "/".join(str(v) for v in vals) if vals else "—"


def cut_table(res: dict, out: R.Out, view: str, name: str, title: str, caption: str) -> None:
    """One cut view: every (arm, scorer) a row, every cell of the cut a column.

    ONE renderer for both views, on purpose. The rarity strata and the magnitude quartiles are
    different cuts with different standing -- one designed, one post hoc -- and the captions say so
    at length, but the cells are the same estimator over the same features and printing them
    through two code paths would let the two drift into looking different for no reason.

    The markdown cell is `mean (n=k)`; the interval goes to the CSV. At eight features per cell a
    95% percentile interval is about 0.2 wide, which is wider than every difference in the table,
    and four of them per row would fill the line with a width the reader cannot use. It is not
    dropped -- it is in `<name>.csv` and in `results.json`, and the trend table prints the widest
    of them per row, which is the number that decides whether any of this separates.
    """
    rows_in = [s for s in res[{"stratum": "strata", "peak": "peak_strata"}[view]]]
    cuts = sorted({s["stratum"] for s in rows_in}, key=lambda s: (s is None, str(s)))
    word = CUT_WORD[view]
    head = ["arm", "run", "scorer", *[f"{word} {'—' if c is None else c}" for c in cuts]]
    csv_head = ["sae", "view", "arm", "run", "arm_name", "scorer", "stratum", "mean", "lo", "hi",
                "n", "short"]
    cellmap = {(s["arm"], s["scorer"], s["stratum"]): s for s in rows_in}
    rows, csv_rows = [], []
    for a in res["arms"]:
        for scorer in res["scorers"]:
            got = [cellmap.get((a.label, scorer, c)) for c in cuts]
            if not any(got):
                continue
            rows.append([a.arm, a.run, scorer, *[
                "—" if g is None else
                f"{R.num(g['mean'])} (n={g['n']}{', SHORT' if g['short'] else ''})"
                for g in got]])
    for s in rows_in:
        csv_rows.append([res["sae"], s["view"], s["arm"], s["run"], s["arm_name"], s["scorer"],
                         s["stratum"], round(s["mean"], 6), round(s["lo"], 6), round(s["hi"], 6),
                         s["n"], s["short"]])
    out.table(name, title, caption, head, rows, csv_header=csv_head, csv_rows=csv_rows)


def trend_verdict(p: float, bonf: float, n_look: int, widest_ci: float, spread: float,
                  dropped: list[str]) -> str:
    """The verdict cell of one trend row: what survives, at what correction, with what caveat.

    A pure function of six numbers so it can be unit-tested at its boundaries -- it decides what a
    reader of the table concludes, and it lived inside a render loop where nothing could reach it.

    `CI-WIDE` is a CAUTION, not a second gate. It fires when the widest per-cell interval in the
    row is at least as wide as the spread between the cell means, which says the cells are
    individually estimated less precisely than the difference being claimed between them. It does
    NOT say the cells fail to separate: a wide interval that sits entirely ABOVE its neighbours
    still separates them, and the 2M peak `M` detection row is exactly that case -- interval
    [0.5406, 0.7219] against neighbouring means of 0.484, 0.506 and 0.494, so its lower bound
    clears all three while its width (0.181) exceeds the spread (0.1375). The flag is there to stop
    a reader taking a small p as precision; deciding what it means is the reader's job, which is
    why it is appended to `separates` rather than replacing it.
    """
    if not math.isfinite(p):
        verdict = "not estimable"
    elif p <= bonf:
        wide = (math.isfinite(widest_ci) and math.isfinite(spread) and widest_ci >= spread)
        verdict = f"separates (p ≤ α/{n_look}){', CI-WIDE' if wide else ''}"
    elif p <= ALPHA:
        verdict = "uncorrected only"
    else:
        verdict = "no separation"
    if dropped:
        verdict += f" — {len(dropped)} cell(s) dropped: {', '.join(dropped)}"
    return verdict


def trend_table(res: dict, out: R.Out) -> None:
    """Does any arm x cut trend survive at eight features per cell? Mostly no, and it says so.

    Three numbers per (view, arm, scorer) -- see `trend_rows` for what each one is and is not --
    and a VERDICT that is a statement about this table's own multiplicity, not a star. The
    Bonferroni divisor is the number of rows of the SAME view, because that is how many spreads
    were looked at before one was reported; it is conservative in one direction and anti-
    conservative in another, and the note says both.
    """
    per_view = {v: [t for t in res["trends"] if t["view"] == v]
                for v in (STRATUM_VIEW, PEAK_VIEW)}
    head = ["view", "arm", "run", "scorer", "cells", "min n", "widest cell CI", "spread",
            "rank ρ", "perm p", "verdict"]
    csv_head = ["sae", "view", "arm", "run", "arm_name", "scorer", "n_cells", "n_cells_all",
                "n_min", "widest_ci", "spread", "rho", "p_perm", "n_perm", "seed",
                "bonferroni_alpha", "verdict", "cells_dropped"]
    rows, csv_rows = [], []
    for view, mine in per_view.items():
        if not mine:
            continue
        # HOW MANY LOOKS THIS TABLE ACTUALLY TAKES. Not `len(mine)`: eval 2's three run directories
        # per SAE were launched with one shared `--cache-dir`, so every corpus arm's row appears
        # once per run label with byte-identical numbers -- 32 rows carrying 16 measurements. A
        # Bonferroni divisor of 32 would be correcting for the same look three times, which is not
        # conservatism, it is a wrong statement about what was done. Rows are deduplicated on
        # (arm name, scorer, the cell means), which is exactly what "the same measurement" means
        # here; under separate caches the means differ and the count rises on its own.
        seen = {(t["arm_name"], t["scorer"], t["_means"]) for t in mine}
        n_look = len(seen)
        bonf = ALPHA / n_look
        for t in mine:
            p = t["p_perm"]
            verdict = trend_verdict(p, bonf, n_look, t["widest_ci"], t["spread"], t["dropped"])
            rows.append([view, t["arm_name"], t["run"], t["scorer"],
                         f"{t['n_cells']}/{t['n_cells_all']}", t["n_min"],
                         R.num(t["widest_ci"], 3), R.num(t["spread"], 4), R.num(t["rho"], 3),
                         R.num(p, 4), verdict])
            csv_rows.append([res["sae"], view, t["arm"], t["run"], t["arm_name"], t["scorer"],
                             t["n_cells"], t["n_cells_all"], t["n_min"], round(t["widest_ci"], 6),
                             round(t["spread"], 6), round(t["rho"], 6), round(p, 6), res["boot"],
                             res["seed"], round(bonf, 6), verdict, "; ".join(t["dropped"])])
    sizes = sorted({t["n_min"] for t in res["trends"]})
    span = f"{sizes[0]}" if len(sizes) < 2 else f"{sizes[0]}–{sizes[-1]}"
    out.table(
        "trends", "Does any arm × cut trend survive?",
        (f"**At {span} features per cell almost nothing can separate, and this table is "
         f"here to say which of it does.** Read the verdict column, not the p.\n\n"
         f"`spread` is max − min of the cell means: the effect size, in the table's own units, and "
         f"the only column that means anything without the machinery. `perm p` tests it against "
         f"the null that the cut carries no information — the cut labels are shuffled over that "
         f"arm's own features with the cell sizes held fixed, {res['boot']} times, seed "
         f"{res['seed']}, reported as (hits + 1) / (resamples + 1) so it is never 0 and never "
         f"finer than the resamples bought — a p of exactly {1 / (res['boot'] + 1):.4f} means NO "
         f"resample reached the observed spread and the true p is BELOW the table's resolution, "
         f"not equal to it. Nothing is assumed normal, which matters: a per-feature "
         f"balanced accuracy on 40 items is a lattice of multiples of 1/40 and is not normal at "
         f"n = 8. Because the statistic is max − min ACROSS cells it is already a maximum, so its "
         f"null absorbs *which* cell turned out extreme and needs no correction within a row.\n\n"
         f"`rank ρ` is Spearman's correlation between the cut index and the cell mean. **It is a "
         f"description of shape and never a test.** With four cells there are 4! = 24 orderings, so "
         f"the exact two-sided p of a perfectly monotone |ρ| = 1 is 2/24 = 0.083 and no trend at "
         f"this design can reach 0.05 on it. It is printed to separate *rises across the cut* "
         f"(ρ near ±1) from *one cell differs* (a large spread at a middling ρ), which the spread "
         f"alone cannot tell apart.\n\n"
         f"`verdict` corrects for the multiplicity this table itself creates, over the DISTINCT "
         f"measurements rather than the printed rows. `separates` means "
         f"p ≤ α/(DISTINCT measurements in that view), α = {ALPHA:g} -- distinct, because a corpus "
         f"arm that several `--run` labels replayed from one shared cache is ONE look and its rows "
         f"here are identical to the last digit. It is still not a full account of the "
         f"multiplicity: the rows share their features, and the two cut views and the second SAE "
         f"block are further looks this divisor does not count -- so read `separates` as "
         f"\"survives the corrections this table can make\", not as a claim about the family of "
         f"everything eval 2 looked at. `widest cell CI` is the width of "
         f"the widest per-cell percentile interval in the row (full intervals in the view's CSV). "
         f"Where it is at least the spread, the verdict is tagged `CI-WIDE`: the cells are "
         f"individually estimated less precisely than the difference being claimed between them, "
         f"so a small p should not be read as precision. It does NOT mean the cells fail to "
         f"separate — a wide interval sitting entirely ABOVE its neighbours still separates them, "
         f"and the 2M peak `M` detection row is that case, its interval [0.5406, 0.7219] clearing "
         f"neighbouring means of 0.484, 0.506 and 0.494 while its width exceeds the spread. Read "
         f"`CI-WIDE` as 'few features are carrying this', and go to the view's CSV for the "
         f"intervals themselves.\n\n"
         f"A cell below {MIN_CELL} features is excluded from all three statistics and named in the "
         f"verdict; `cells` is how many of the row's cells were used."),
        head, rows, csv_header=csv_head, csv_rows=csv_rows)


def render(res: dict, out: R.Out, sanity: list[dict], figures: list[str]) -> Path:
    scorers = res["scorers"]
    ref_label = res["ref"].label if res["ref"] else "(none)"

    # --- bal_acc per arm x scorer -------------------------------------------------------------
    head = ["arm", "run", "role", "n ex", *[f"{s} mean [95% CI]" for s in scorers],
            *[f"{s} feats" for s in scorers]]
    csv_head = ["sae", "arm", "run", "arm_name", "scorer", "role", "n_examples", "n_features",
                "n_rows", "n_no_metric", "mean", "lo", "hi", "boot", "seed"]
    by_arm: dict[str, dict[str, dict]] = {}
    for c in res["cells"]:
        by_arm.setdefault(c["arm"], {})[c["scorer"]] = c
    rows, csv_rows = [], []
    for a in res["arms"]:
        got = by_arm.get(a.label)
        if not got:
            continue
        any_cell = next(iter(got.values()))
        rows.append([
            a.arm, a.run, _joined(any_cell["roles"]), _joined(any_cell["n_examples"]),
            *[ci(got[s]["mean"], got[s]["lo"], got[s]["hi"]) if s in got else "—" for s in scorers],
            *[got[s]["n_features"] if s in got else "—" for s in scorers],
        ])
        for s in scorers:
            if s not in got:
                continue
            c = got[s]
            csv_rows.append([res["sae"], a.label, a.run, a.arm, s, _joined(c["roles"]),
                             _joined(c["n_examples"]), c["n_features"], c["n_rows"],
                             c["n_no_metric"], round(c["mean"], 6), round(c["lo"], 6),
                             round(c["hi"], 6), res["boot"], res["seed"]])
    out.table(
        "bal_acc", f"Balanced accuracy — `{res['sae'] or '(sae unstated)'}`",
        (f"Mean over features of the per-(feature, arm) balanced accuracy `run.py` stored, with a "
         f"PERCENTILE BOOTSTRAP interval over {res['boot']} resamples OF THE FEATURES (seed "
         f"{res['seed']}, α = {ALPHA:g}). **Detection and fuzzing are not pooled and must not be "
         f"read against each other**: fuzzing marks the examples at the gate and, on the legacy "
         f"protocol, runs ZERO-SHOT while detection sees Delphi's three verbatim few-shot turns — "
         f"the two numbers are comparable across arms and not to one another. `n ex` is the arm's "
         f"shown-example count as the build recorded it; the scorer-only pseudo-arms carry 0 "
         f"because they borrow another arm's description and have no example set of their own. "
         f"Chance is {CHANCE:g}."),
        head, rows, csv_header=csv_head, csv_rows=csv_rows)

    # --- paired contrasts ---------------------------------------------------------------------
    head = ["arm", "run", "scorer", "n paired", "n arm", "n ref", "dropped", "mean Δ [95% CI]",
            "win frac", "pairing"]
    csv_head = ["sae", "arm", "run", "arm_name", "scorer", "ref", "n_paired", "n_arm", "n_ref",
                "n_dropped", "mean_diff", "lo", "hi", "win_frac", "complete",
                "features_lost_by_arm", "features_lost_by_ref"]
    rows, csv_rows = [], []
    for x in res["contrasts"]:
        dropped = (x["n_arm"] - x["n_paired"]) + (x["n_ref"] - x["n_paired"])
        rows.append([
            x["arm_name"], x["run"], x["scorer"], x["n_paired"], x["n_arm"], x["n_ref"], dropped,
            ci(x["mean"], x["lo"], x["hi"], signed=True), R.num(x["win_frac"], 3),
            "complete" if x["complete"] else "REDUCED",
        ])
        csv_rows.append([res["sae"], x["arm"], x["run"], x["arm_name"], x["scorer"], x["ref"],
                         x["n_paired"], x["n_arm"], x["n_ref"], dropped, round(x["mean"], 6),
                         round(x["lo"], 6), round(x["hi"], 6), round(x["win_frac"], 6),
                         x["complete"], " ".join(str(f) for f in x["lost_by_arm"]),
                         " ".join(str(f) for f in x["lost_by_ref"])])
    out.table(
        "contrasts", f"Paired contrasts against `{ref_label}`",
        (f"Per-feature difference `arm − {ref_label}`, averaged over the features the two arms "
         f"SHARE, with a paired percentile bootstrap interval ({res['boot']} resamples of the "
         f"shared features, seed {res['seed']}). The reference is `{ref_label}`; a bare `--ref` "
         f"names an arm and is resolved inside the FIRST `--run` given, because each run directory "
         f"carries its own copy of every corpus arm -- the SAME calls replayed when the runs shared a "
         f"`--cache-dir`, as eval 2's did, and independent ones when they did not. `n paired` is the "
         f"intersection and `dropped` is how many features either side lost to it — plan §3.3 asks "
         f"for the pairing to be asserted rather than left implicit, so a reduced pairing is "
         f"labelled REDUCED here and carried into `results.json` with the feature ids, never "
         f"de-paired in silence. `win frac` is the fraction of shared features with a positive "
         f"difference, reported because the outcome is bimodal and a mean alone misleads."),
        head, rows, csv_header=csv_head, csv_rows=csv_rows)

    # --- the two cut views, and the trend verdict over them -------------------------------------
    record = ("\n\nWhat the cut IS, quoted from each set's own draw record on the volume rather "
              "than retyped here:\n\n"
              + "\n".join(f"  - {ln}" for ln in res["record"]) + "\n") if res.get("record") else ""
    if res["strata"]:
        cut_table(
            res, out, STRATUM_VIEW, "strata", "Balanced accuracy per rarity stratum",
            (f"Mean bal_acc over the features of each stratum, with the SAME percentile bootstrap "
             f"over features the headline table uses ({res['boot']} resamples, seed {res['seed']}) "
             f"in the CSV and the cell's own n in the table. "
             f"**The strata are quartiles of the feature's RARITY at the 16M corpus scan, cut on "
             f"the dictionary's own distribution, 0 the rarest quarter and 3 the commonest.** The "
             f"statistic differs between the two SAEs and so do the cut points: the 2M set is cut "
             f"on log10 of the raw GATED FIRE COUNT at 16M over the ~100k eval-split features that "
             f"pass the eligibility filter (`features/draw_sae2m.py`), the 131k set on log10 of "
             f"the DENSITY — gated fires ÷ scanned positions — over the whole eligible dictionary "
             f"(`precompute/targets.py`). Those two are the same physical axis up to the constant "
             f"log10(scanned positions) and differ in the pool they were cut over, so stratum k of "
             f"one is the same RANK in its own dictionary's rarity ordering as stratum k of the "
             f"other and is **not the same rarity**: the two blocks' strata must not be read "
             f"against each other.{record}\n"
             f"Every arm and BOTH scorers are here, and the per-stratum breakdown is available on "
             f"either SAE — `--no-strata` turns the block off, it does not make it primary-only. "
             f"A cell below {MIN_CELL} features is labelled SHORT and kept out of the trend table; "
             f"it is still printed, with its n, because an arm that loses features out of one "
             f"stratum and not another is a property of the run and not a rounding detail."))
    else:
        out.section("### Balanced accuracy per rarity stratum\n\n"
                    "*Not built: `--no-strata`.*\n")

    if res["peak_strata"]:
        cuts = ", ".join(R.num(c, 4) for c in res["peak_cuts"])
        cut_table(
            res, out, PEAK_VIEW, "peak_strata",
            "Balanced accuracy per activation-magnitude quartile",
            (f"The same table on a SECOND axis: `corpus_peak`, the feature's peak activation over "
             f"the 16M-token corpus scan — a MAGNITUDE, not a count, and not the rarity axis "
             f"above. It is `sae/<sae>/max_act.f16`, the running elementwise max of the SAE's "
             f"post-ReLU activation over every scanned position, which `build.py` reads from that "
             f"array (not from the set's `ids.jsonl`, whose column for it is `corpus_peak_16m` on "
             f"a `draw_sae2m` set and `max_act` on a `targets` one) and `run.py` copies onto every "
             f"score row as `corpus_peak`. It is the ONLY magnitude field read here: `density` "
             f"(gated fires ÷ scanned positions) is null on every row of a `draw_sae2m` set, so a "
             f"split that reached for it would come back empty on the 2M block. "
             f"**This is a POST-HOC split of the {res['n_features']} features this block analysed, "
             f"not a stratification the draw controlled.** The rarity strata above were balanced "
             f"by construction — the draw took an equal number of features from each quartile of "
             f"the pool, so a per-stratum comparison is a designed contrast. Nothing balanced this "
             f"one: the quartiles are cut here, over the analysed features' own `corpus_peak` "
             f"values, at [{cuts}] (numpy's linear quantiles at 0.25/0.5/0.75; a value exactly on "
             f"a cut falls in the higher quartile), so the cells are equal only when no tie "
             f"straddles a cut — which is why each prints its own n. Quartile 0 is the weakest "
             f"peak. The two axes are CONFOUNDED in the obvious direction — a feature that fires "
             f"rarely also reaches a lower peak — so agreement between this table and the one "
             f"above is expected and is not independent evidence."))
    else:
        out.section("### Balanced accuracy per activation-magnitude quartile\n\n"
                    "*Not built: `--no-peak-strata`, or no score row carries a `corpus_peak`.*\n")

    if res["trends"]:
        trend_table(res, out)

    # --- support / provenance -----------------------------------------------------------------
    head = ["arm", "run", "scorer", "role", "draw", "n ex", "features", "no metric",
            "no explanation", "mean n_items", "parse rate"]
    csv_head = ["sae", "arm", "run", "arm_name", "scorer", "run_dir", "role", "draw", "n_examples",
                "n_features", "n_rows", "n_no_metric", "n_no_explanation", "mean_n_items",
                "parse_rate"]
    rows, csv_rows = [], []
    for c in res["cells"]:
        rows.append([c["arm_name"], c["run"], c["scorer"], _joined(c["roles"]),
                     _joined(c["draws"]), _joined(c["n_examples"]), c["n_features"],
                     c["n_no_metric"], c["n_no_explanation"], R.num(c["mean_n_items"], 1),
                     R.num(c["parse_rate"], 3)])
        csv_rows.append([res["sae"], c["arm"], c["run"], c["arm_name"], c["scorer"], c["run_dir"],
                         _joined(c["roles"]), _joined(c["draws"]), _joined(c["n_examples"]),
                         c["n_features"], c["n_rows"], c["n_no_metric"], c["n_no_explanation"],
                         round(c["mean_n_items"], 3), round(c["parse_rate"], 6)])
    n_ex_seen = sorted({n for c in res["cells"] for n in c["n_examples"]})
    out.table(
        "support", "Support and provenance",
        (f"What is behind each number above. `n ex` is the arm's shown-example count "
         f"({_joined(n_ex_seen)} across the arms here): **unequal N is a property of this design, "
         f"not an artefact** — the NLA arm shows 4 examples against a corpus arm's 16 because 4 is "
         f"what the verbalizer's budget buys at 200 tokens, and the scorer-only pseudo-arms "
         f"(`role` floor / judge_null / draw_null) carry 0 because they borrow another arm's "
         f"description and have no example set. `no metric` counts (feature, arm) rows whose "
         f"balanced accuracy came back null — `run.rates` returns NaN when a class was absent from "
         f"the PARSED items, which is what an all-unparsed feature looks like; those features are "
         f"dropped from the means, never imputed. `parse rate` is the mean over features of "
         f"`n_parsed / n_batches`; an unparsed batch is dropped by `run.py` rather than padded."),
        head, rows, csv_header=csv_head, csv_rows=csv_rows)

    # --- checks -------------------------------------------------------------------------------
    chk_rows = []
    for c in res["checks"]:
        if "skipped" in c:
            chk_rows.append([c["kind"], c["who"], c["detail"], "—", "—",
                             "skipped: " + c["skipped"]])
            continue
        if c["kind"] == "pairing":
            outcome = ("complete" if c["complete"]
                       else f"REDUCED: {c['n_mismatches']} feature(s) lost of "
                            f"{c['n_arm']} / {c['n_ref']}")
            chk_rows.append([c["kind"], c["who"], c["detail"], c["comparisons"], "—", outcome])
            continue
        if c["kind"] == "cells":
            bits = []
            if c["n_short"]:
                bits.append(f"SHORT (< {MIN_CELL} features), excluded from the trend table: "
                            + "; ".join(c["short_cells"]))
            if c["n_thin"]:
                bits.append(f"under-filled (below the view's fullest cell, n={c['full']}): "
                            + "; ".join(c["thin_cells"]))
            if c["n_absent"]:
                bits.append("ABSENT (the arm has no scored feature in this cell at all, so its "
                            "row is compared on fewer cells than its neighbours): "
                            + "; ".join(c["absent_cells"]))
            chk_rows.append([c["kind"], c["who"], c["detail"], c["comparisons"], "—",
                             " · ".join(bits) or f"every cell at n={c['full']}"])
            continue
        defect = "" if not c["n_mismatches"] else "  ← POSSIBLE DEFECT"
        chk_rows.append([
            c["kind"], c["who"], c["detail"], c["comparisons"], R.num(c["worst_excess"], 9),
            f"{c['n_identity']} identity, {c['n_out_of_range']} out of range, "
            f"{c['n_bad_counts']} bad counts{defect}"])
    out.table(
        "checks", "Reader and pairing checks",
        (f"`reader` recomputes each stored balanced accuracy as `0.5 × (TPR + TNR)` from the rates "
         f"stored on the SAME row — `run.rates`' own definition, with the two A5 views restricting "
         f"only the negative side — so both sides are the same quantity and anything outside the "
         f"6-decimal rounding `run._nr` applies ({READER_TOL:g}) is a defect in this reader or in "
         f"that product, never a finding. `worst excess` is how far the largest difference sat "
         f"OUTSIDE its tolerance, so a negative number means every comparison was inside it. It "
         f"also counts rates outside [0, 1] and rows with more parsed batches than batches. "
         f"`pairing` is the intersection against `{ref_label}` per (arm, scorer): REDUCED is not "
         f"automatically a defect — an arm may legitimately cover fewer features — but it is never "
         f"invisible. `cells` is the same rule for the two cut views: a cell below {MIN_CELL} "
         f"features is named here and excluded from the trend table, and a cell merely below the "
         f"view's fullest is named as under-filled and kept — `run.py` omits a (feature, arm) whose "
         f"every batch went unparsed, so an arm can lose features out of one stratum and not "
         f"another, and a table of means alone would show four equal-looking cells."),
        ["kind", "who", "detail", "comparisons", "worst excess", "outcome"], chk_rows)

    # --- sanity -------------------------------------------------------------------------------
    if sanity:
        sane_rows = []
        for s in sanity:
            sane_rows.append([
                s["name"], s.get("scorer", "—"), s.get("arm", "—"), s.get("metric", "—"),
                s.get("n", "—"), R.num(s.get("ours")), R.num(s.get("expect")),
                R.num(s["tol"]) if isinstance(s.get("tol"), (int, float)) else "—",
                s["verdict"], s.get("why", ""),
            ])
        out.table("sanity", "Sanity gates",
                  ("From the `--sanity` YAML, which the user edits, in the same selector idiom as "
                   "eval 1's `results/sanity.yaml`. A `FLAG` means the numbers above should not be "
                   "read until it is explained; `absent` means the gate's selector does not "
                   "resolve on these runs; `no verdict` means the YAML declares the two are not "
                   "the same statistic. `n` is the number of features behind `ours`."),
                  ["gate", "scorer", "arm", "metric", "n", "ours", "expected", "tol", "verdict",
                   "why"], sane_rows)
        prov = [f"- **{s['name']}** — {s['provenance']}" for s in sanity if s.get("provenance")]
        if prov:
            out.section("#### Where the expected numbers come from\n\n" + "\n".join(prov) + "\n")

    for m in res["missing"]:
        out.note(m)
    for n in res["notes"]:
        out.note(n)
    return out.finish(figures)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------


def _floor_mean(res: dict, scorer: str) -> tuple[float, str]:
    """(the floor arm's mean bal_acc for this scorer, its label) or (NaN, "").

    The floor (`role: floor`) is another feature's description under a fixed derangement and should
    sit at chance; the line is drawn so an arm can be read against it rather than against 0.5
    alone. With one floor per run directory the reference run's is used, because that is the run
    the contrasts are against; with no unambiguous choice the line is simply not drawn.
    """
    hits = [c for c in res["cells"] if c["scorer"] == scorer and "floor" in c["roles"]]
    if not hits:
        return float("nan"), ""
    ref_run = res["ref"].run if res["ref"] else None
    mine = [c for c in hits if c["run"] == ref_run] or (hits if len(hits) == 1 else [])
    if len(mine) != 1:
        return float("nan"), ""
    return float(mine[0]["mean"]), mine[0]["arm"]


def make_figures(res: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": R.SURFACE, "savefig.facecolor": R.SURFACE,
                         "font.size": 9, "legend.frameon": False})
    colours = res["colours"]
    names: list[str] = []

    # One figure per SCORER, never one figure with both: fuzzing and detection are not on a
    # common scale (see the bal_acc caption), and a shared axis would invite exactly the
    # comparison the protocol forbids.
    for scorer in res["scorers"]:
        cells = [c for c in res["cells"] if c["scorer"] == scorer]
        if not cells:
            continue
        order = {a.label: i for i, a in enumerate(res["arms"])}
        cells.sort(key=lambda c: order.get(c["arm"], 99))
        xs = np.arange(len(cells), dtype=float)
        ys = np.array([c["mean"] for c in cells], dtype=float)
        # Asymmetric error bars: a percentile interval is not centred on the mean, and collapsing
        # it to a half-width would assert a symmetry the estimator does not have.
        lo = np.array([c["mean"] - c["lo"] if math.isfinite(c["lo"]) else 0.0 for c in cells])
        hi = np.array([c["hi"] - c["mean"] if math.isfinite(c["hi"]) else 0.0 for c in cells])
        fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(cells) + 2.2), 3.8))
        # No axes title: with one panel it would only repeat the suptitle below.
        R.style_axes(ax, ylabel="mean balanced accuracy over features")
        for i, c in enumerate(cells):
            ax.errorbar(xs[i], ys[i], yerr=[[lo[i]], [hi[i]]], fmt="o", markersize=6,
                        color=colours.get(c["arm"], R.PALETTE[0]), capsize=3, linewidth=1.6,
                        zorder=3)
        ax.axhline(CHANCE, color=R.INK_MUTED, linewidth=1.0, linestyle="--", zorder=2)
        ax.annotate("chance", (len(cells) - 0.5, CHANCE), textcoords="offset points",
                    xytext=(4, 3), color=R.INK_MUTED, fontsize=7)
        floor, floor_label = _floor_mean(res, scorer)
        if math.isfinite(floor):
            # Ink, not a palette hue: the eight categorical hues belong to the ARMS, and with eight
            # or more arms a floor drawn in one of them would be the same colour as a mark.
            ax.axhline(floor, color=R.INK, linewidth=1.0, linestyle=":", zorder=2)
            ax.annotate(f"floor ({floor_label})", (len(cells) - 0.5, floor),
                        textcoords="offset points", xytext=(4, -10), color=R.INK, fontsize=7)
        ax.set_xticks(xs)
        # The RUN is on the tick beside the arm: `C16` under two labels is two measurements, and a
        # tick that said only `C16` twice would read as a repeat.
        ax.set_xticklabels([f"{c['arm_name']}\n{c['run']}" for c in cells], fontsize=7,
                           rotation=45, ha="right")
        ax.set_xlim(-0.6, len(cells) - 0.4)
        fig.suptitle(f"autointerp {scorer} — {len(cells)} arms, "
                     f"{res['boot']} bootstrap resamples over features",
                     color=R.INK, fontsize=11, x=0.02, ha="left")
        fig.tight_layout()
        names.append(R.savefig(fig, out_dir, f"bal_acc_{scorer}"))
    return names


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    run: Annotated[list[str] | None, typer.Option(
        "--run", help="`<label>=<run_dir>`, repeatable or comma-separated; the first is the "
                      "reference run a bare --ref resolves in")] = None,
    sae: Annotated[str, typer.Option(help="the SAE key this invocation reports, e.g. "
                                          "qwen36-27b/sae2m")] = "",
    label: Annotated[str, typer.Option(help="block label for the tables (default: the SAE key)")] = "",
    ref: Annotated[str, typer.Option(help="reference arm of the paired contrasts, `<arm>` or "
                                          "`<run label>/<arm>`")] = "DOCMAX",
    strata: Annotated[bool, typer.Option(help="per-rarity-stratum table (works on either SAE)")] = True,
    peak_strata: Annotated[bool, typer.Option(
        help="per-activation-magnitude-quartile table, a POST-HOC cut of the analysed "
             "features on their `corpus_peak`")] = True,
    out: Annotated[Path, typer.Option(help="output directory; the block goes in <out>/<sae-slug>")]
    = R.HERE / "out" / "autointerp",
    root: Annotated[str, typer.Option(help="volume-relative root the runs were written under")] = "",
    data: Annotated[Path | None, typer.Option(help="local mirror (default results/data/<root>)")] = None,
    fetch: Annotated[bool, typer.Option(help="fetch missing files off the volume")] = True,
    refetch: Annotated[bool, typer.Option(help="re-download even what the mirror already has")] = False,
    modal_cmd: Annotated[str, typer.Option(help="how to invoke the modal CLI")] = "uvx modal",
    sanity_file: Annotated[Path, typer.Option("--sanity", help="optional gates YAML")]
    = R.HERE / "sanity_autointerp.yaml",
    boot: Annotated[int, typer.Option(help="bootstrap resamples over features")] = R.N_BOOT,
    seed: Annotated[int, typer.Option(help="bootstrap seed")] = R.BOOT_SEED,
    figures: Annotated[bool, typer.Option(help="write figures/ (PDF + PNG)")] = True,
    quiet: Annotated[bool, typer.Option(help="do not print every fetched file")] = False,
) -> None:
    runs = parse_runs(run)
    assert runs, (
        "at least one `--run <label>=<run_dir>` is required. Eval 2 is one `run` directory per "
        "checkpoint — on each SAE, one for rl-last16, one for the old primary and one for NLA — "
        "and this driver joins them, e.g. `--run rl-last16=<dir> --run old-primary=<dir>`")
    mirror = data or (R.HERE / "data" / (root.replace("/", "_") or "vol"))
    vol = R.Vol(root, mirror, modal_cmd, refetch, quiet, offline=not fetch)
    res = analyse(vol, runs, sae, ref, boot, seed, strata, peak_strata)
    # One block per SAE: the driver is invoked once per SAE and must not overwrite the other's
    # tables, so the slug comes from the SAE key (or `--label`) and never from the output root.
    slug = (sae or label or "autointerp").replace("/", "_")
    out_dir = Path(out) / slug
    figs = make_figures(res, out_dir) if figures else []
    sanity = run_sanity(res, sanity_file)
    if not Path(sanity_file).exists():
        res["notes"].append(f"no sanity gates: `{sanity_file}` does not exist (eval 2 has no "
                            f"recorded expectations yet; the selector idiom is ready for them)")
    preamble = [
        f"- SAE `{sae or '(unstated)'}`{f', block `{label}`' if label else ''}, volume root "
        f"`{res['root']}`, mirror `{vol.local}`",
        f"- command: `{' '.join(sys.argv)}`",
        "- runs: " + ", ".join(f"`{k}` = `{v}`" for k, v in runs.items()),
        f"- reference arm: `{res['ref'].label if res['ref'] else '(unresolved)'}`",
        f"- intervals: percentile bootstrap over FEATURES, {boot} resamples, seed {seed}",
    ]
    o = R.Out(out_dir, f"Eval 2 — SAE autointerp on `{label or sae or 'autointerp'}`", preamble)
    path = render(res, o, sanity, figs)

    reg = stat_registry(res)
    with open(out_dir / "results.json", "w") as fh:
        json.dump({
            "sae": sae, "label": label, "root": res["root"], "runs": runs,
            "builds": res["builds"],
            "ref": res["ref"].label if res["ref"] else None,
            "boot": boot, "seed": seed, "alpha": ALPHA,
            "arms": [a.label for a in res["arms"]], "scorers": res["scorers"],
            "cells": res["cells"], "contrasts": res["contrasts"], "strata": res["strata"],
            "peak_strata": res["peak_strata"], "peak_cuts": res["peak_cuts"],
            "trends": res["trends"], "draw_record": res["record"], "min_cell": MIN_CELL,
            "checks": res["checks"], "sanity": sanity,
            "stats": [{"scorer": k[0], "arm": k[1], "stratum": k[2], "metric": k[3], "value": v}
                      for k, v in sorted(reg.items(), key=lambda kv: [str(x) for x in kv[0]])],
            "missing": res["missing"], "notes": res["notes"],
        }, fh, indent=1, default=str)

    print(f"\n[autointerp] {path}")
    print(f"[autointerp] {len(runs)} runs, {len(res['arms'])} arms, {res['n_rows']} score rows, "
          f"{len(res['contrasts'])} contrasts, {len(figs)} figures")
    for m in res["missing"]:
        print(f"   MISSING  {m}")
    for n in res["notes"]:
        print(f"   NOTE     {n}")
    for s in sanity:
        print(f"   {s['verdict']:<10} {s['name']}: {s.get('why', '')}")
    for c in res["checks"]:
        if "skipped" in c:
            print(f"   check      {c['kind']}/{c['who']}: skipped ({c['skipped']})")
        elif c["n_mismatches"]:
            kind = {"pairing": "REDUCED PAIRING",
                    "cells": f"SHORT CELL(S) (< {MIN_CELL} features)"}.get(c["kind"], "MISMATCHES")
            print(f"   check      {c['kind']}/{c['who']}/{c['detail']}: "
                  f"{c['n_mismatches']} {kind}")


if __name__ == "__main__":
    app()
