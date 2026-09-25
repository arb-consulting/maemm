#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15"]
# ///
"""How hard does a target feature fire on each source, at a MATCHED number of draws?

Local, CPU, no GPU, no model, no network. Reads a LOCAL MIRROR of the volume (`--data <dir>`,
whose subpaths are the volume's own), so whoever fetched the files decides what is there.

    uv run evals/faithfulness/reconstruction/sae_smoke64.py --data ~/mirror --out out/sae_smoke64.md
    uv run evals/faithfulness/reconstruction/sae_smoke64.py --selftest

WHAT THIS ADDS TO `act_smoke.py`, which stays as it is. act_smoke reports one number per
(feature, source) -- the max over every rollout -- and corrects for the rollout count by
truncating the 64-rollout product to its first 4 (`maemm_bo4`). That answers "is the 26% a
property of the MAEMM or of the dictionary" and nothing else, because a max over n draws is not
comparable across sources with different n and truncation throws away 60 of 64 rollouts.

Here every source is read at bo1 / bo4 / bo16 instead: the disjoint-group best-of-k mean
(`common.best_of_k_means`), which is the same estimator `score`'s `per_target.jsonl` already
reports for cosines. bo-k is a curve, so two sources are comparable at whatever k they BOTH
reach, the whole rollout budget is used at every k, and the k a claim is made at is visible in
the column header rather than in a truncation buried in the source table.

Four sources per SAE, declared in SPECS below:

    rl16      the 16-rollout RL checkpoint                (sae_self over its scores)
    primary   the primary MAEMM                           (sae_self; vLLM at n = 64 on the 131k)
    nla       the NLA verbalizer's texts                  (sae_self, n = 4, scored at width 257)
    corpus    the 16M document-diverse scan's top windows (examples_docmax, one per document)

against `corpus_peak`, the feature's maximum over the 16M corpus scan (`sae/<sae>/max_act.f16`),
which is the denominator of every `ratio` here and is the SAME quantity act_smoke uses.

STRATA. Both sets carry a `stratum` per row and every table is also cut on it. On the 2M side
that stratum is a quartile of log10(gated fires at 16M) over the ELIGIBLE POOL and the 64
features are 16 per quartile by construction (`features/draw_sae2m.py --n 64 --stratified`), so
the "all" row of the 2M block is a mean over four equally-weighted quartiles and NOT over the
dictionary. The per-stratum rows are the ones to read; the "all" row is a convenience.

Nothing here recomputes an activation. Every number is read from what `sae_self` and
`examples_docmax` already measured on the clean base, so this script cannot disagree with them
about what fired. A missing file is reported as `absent`, never as a zero -- a source nobody ran
and a source that never fired are opposite findings and must not print the same.

COSINES are read from each source's own `per_target.jsonl` and go into the JSON ONLY. They
answer a different question (does the rollout point the right way) on a different scale, and a
table that put them beside an activation ratio would invite the two to be read as one story.
The ONE exception is the reader check in the comparison section, which quotes a stored product's
own `bo_4` beside its `n_sae_gated` -- there the cosine is the artefact being checked, not a
result.

COMPARISON AGAINST PRIOR RESULTS closes the file: three checks, each computed from the mirror
where its inputs are there and printed as `absent` with the reason where they are not.

  (1) the card, rl-last16 at bo4 on the upstream 512-feature sets, against ours on these features.
      The upstream denominators are not ours -- `norm_act` divides by THE UPSTREAM corpus peak from the 1.0B-token
      scan, we divide by the 16M `max_act` -- so where the upstream peak is reachable per feature BOTH
      ratios are quoted. On the 2M side that peak is the `corpus_peak_1b` column the stratified
      draw already writes into `ids.jsonl` (rank-0 window of `eval_2m_features_100k_windows`,
      which reproduces `eval_2m_features_512`'s shipped `corpus_peak` exactly), or `--peaks-1b`.
      On the 131k side it is the SAE repo's own shipped max-acts peak, from `repo_examples`,
      which is a PROXY and not the upstream number -- so whether it is even on our activation scale is
      measured there too, and where it is not the verdict says the denominator differs rather
      than naming a pipeline defect.
  (2) A READER CHECK on the old primary's existing v1 product: `sae_self.json`'s own `per_target`
      (`max_peak_act`, `fire_fraction`, `corpus_peak`) against this script's recomputation from
      `sae_self.f16` for the same rows. These must agree EXACTLY -- both are the same reduction
      of the same array -- so any row that differs is a defect in this reader or in that product,
      not a finding, and is printed as one.
  (3) The 16-feature `act_smoke` medians of 2026-09-21, hard-coded with their provenance. Where
      the two runs use the same estimator on the same product the difference is tested; where
      they do not (different rollout counts, and `examples_4m` against `examples_docmax` on the
      2M side) the row says so and no difference is computed.

A disagreement wider than this run's own sampling error is flagged. The band is crude on purpose
and is stated with the flag: it uses OUR standard error over these features and ignores the prior
result's own, and the two feature sets are different draws, so it catches a gross disagreement
and nothing finer.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

# How many of a feature's stored corpus windows count as "the top windows an arm would show".
# 16 is `autointerp.n_examples`, and is act_smoke's CORPUS_TOP_N unchanged: the question is what
# the text an explainer SEES reaches, not what the whole stored pool does.
CORPUS_TOP_N = 16

# The best-of-k budgets every source is reported at. 1 is the plain mean over draws; 16 is the
# largest k the 16-rollout products reach; 64 is deliberately absent, because only one of the six
# (source, SAE) cells could compute it and a column with one number in it reads as a comparison.
BO_KS = (1, 4, 16)

# Which per_target.jsonl columns are carried into the JSON. `bo_4`/`bo_16` are missing on a
# product scored at n < k (precompute/score.py skips a k above n rather than clamping it).
# `n_sae_gated` is not a cosine: it is how many of the row's rollouts the stored SAE CSR holds an
# entry for at their argmax token, which is the gate count the reader check quotes.
COS_KEYS = ("mean_cos", "max_cos", "bo_4", "bo_16")
SCORE_KEYS = COS_KEYS + ("n_sae_gated", "n")

# How close to 1 the prior denominator's scale must be for a ratio against it to be read as the
# prior work's own statistic. Outside this band the two are not the same quantity and the
# difference is about the denominator, not about the pipeline -- the verdict says which.
SCALE_OK = (0.9, 1.1)

# How many standard errors of OUR OWN mean a prior result may sit away before it is flagged.
# Stated with every flag, because the band ignores the prior result's own sampling error and the
# two feature sets are different draws: it catches a gross disagreement, not a fine one.
FLAG_Z = 3.0

# --- prior results, hard-coded with provenance -------------------------------------------------
#
# (1) the card for `rl-last16` at bo4 over the upstream 512-feature sets. `stat` names WHICH of our
# columns it is comparable with, because the two SAEs are not reported the same way there: the
# 131k number is a normalised activation and the 2M numbers are firing rates. `denominator` is
# what the upstream ratio divides by, which is NOT our 16M `max_act` -- see `alt_peak` in SPECS.
CARD = {
    "qwen36-27b/l42-1b": {
        "arm": "rl-last16, bo4, the upstream 512-feature 131k set",
        "stat": "norm_act",
        "ours": "bo4_ratio",
        "value": 0.824,
        "denominator": "the upstream corpus peak from the 1.0B-token scan",
        "note": "",
    },
    "qwen36-27b/sae2m": {
        "arm": "rl-last16, bo4, the upstream 512-feature 2M set",
        "stat": "fired",
        # A firing rate has no denominator, so this row is comparable as it stands -- it is the
        # one number on the card that does not depend on whose corpus peak is used.
        "ours": "mean_item_fired_frac",
        "value": 0.344,
        "denominator": "(none: a firing rate)",
        "note": "0.344 is the ENCODER side, which is what this draw indexes; the upstream decoder side "
                "reads 0.416 and is listed for context only -- these rows are encoder columns.",
    },
}

# (3) The previous day's smoke, for continuity. `ours` names the (source, column) pair that is
# the SAME estimator; `comparable` false means the two runs measured different things and the
# row is printed WITHOUT a difference rather than with a misleading one.
ACT_SMOKE_REF = {
    "provenance": "reconstruction/act_smoke.py, run 2026-09-21, 16 features per SAE",
    "rows": [
        {
            "sae": "qwen36-27b/sae2m", "arm": "rl-last16 (n = 4, 64 tokens)",
            "median_ratio": 0.32, "firing": "4/16",
            "ours": ("rl16", "bo4_ratio"), "comparable": True,
            "why": "act_smoke took the max over its 4 rollouts, which IS best-of-4; our bo4 is "
                   "the disjoint-group mean of the same quantity over 16, so it estimates the "
                   "same number with less variance. Different 16 vs 64 features, and act_smoke's "
                   "arm generated 64 tokens.",
        },
        {
            "sae": "qwen36-27b/sae2m", "arm": "NLA", "median_ratio": 0.33, "firing": "5/16",
            "ours": ("nla", "bo4_ratio"), "comparable": True,
            "why": "both are n = 4, so act_smoke's max over them is our bo4 exactly.",
        },
        {
            "sae": "qwen36-27b/sae2m", "arm": "corpus top-16, 4M prefix",
            "median_ratio": 0.83, "firing": "16/16",
            "ours": ("corpus", "bo1_ratio"), "comparable": False,
            "why": "DIFFERENT PRODUCT: act_smoke read `examples_4m`, a scan over a quarter of "
                   "the text, while this run reads the 16M `examples_docmax`. A wider scan can "
                   "only find a stronger window, so the two are not a difference.",
        },
        {
            "sae": "qwen36-27b/l42-1b", "arm": "old primary (n = 64)",
            "median_ratio": 1.05, "firing": "16/16",
            "ours": ("primary", "peak_max_ratio"), "comparable": True,
            "why": "the same product at the same n; both are the max over all 64 rollouts.",
        },
        {
            "sae": "qwen36-27b/l42-1b", "arm": "old primary, first 4 rollouts",
            "median_ratio": 0.95, "firing": "15/16",
            "ours": ("primary", "bo4_ratio"), "comparable": True,
            "why": "act_smoke truncated to the FIRST four rollouts (one best-of-4 draw); our bo4 "
                   "averages 16 disjoint best-of-4 groups over the same 64. Same expectation, "
                   "much less variance, so ours is the better estimate of the same number.",
        },
        {
            "sae": "qwen36-27b/l42-1b", "arm": "NLA", "median_ratio": 0.53, "firing": "14/16",
            "ours": ("nla", "bo4_ratio"), "comparable": True,
            "why": "both are n = 4.",
        },
        {
            "sae": "qwen36-27b/l42-1b", "arm": "corpus (16M docmax)",
            "median_ratio": 1.00, "firing": "16/16",
            "ours": ("corpus", "bo1_ratio"), "comparable": True,
            "why": "the same product and the same statistic -- the top stored window's peak.",
        },
    ],
}

# The comparison, laid out rather than inferred. One SAE per spec, one mirror `root` per SAE, and
# the four sources with the path each hangs off. A source may override the spec's root with its
# own `root` key -- `""` pins it to the mirror's own `maemms/...`, which is where a product that
# was NOT written by this smoke already lives.
#
# Path resolution, per spec: a path is looked for under `<data>/<root>/` first and under
# `<data>/` second, and the prefix that had it is recorded per row. A smoke root therefore
# SHADOWS the canonical mirror: what the smoke wrote is read from the smoke root, and everything
# it did not write (the 131k set's own heldout rows, its corpus scan, its max_act table) is read
# from the canonical paths without a second spec. A source with an explicit `root` skips the
# search and is read from exactly there, so a wrong mirror reports `absent` instead of silently
# resolving to a different product with the same name.
SPECS = [
    {
        "sae": "qwen36-27b/sae2m",
        "base": "qwen36-27b",
        "set": "2026-09-21_sae2m_64",
        "root": "tmp/sae-smoke64",
        "sae_dir": "base/qwen36-27b/sae/sae2m",
        "rows": "",  # the whole set: it is 64 rows, drawn for this smoke
        "corpus_note": "the 16M document-diverse scan, one window per document",
        # the original denominator for these features, for the card comparison. The stratified
        # draw writes it into every row (features/draw_sae2m.py), so it costs nothing here; when
        # the draw ran on a mirror without the upstream window parquet the column is null and `--peaks-1b`
        # supplies it instead.
        "alt_peak": {
            "label": "corpus_peak_1b",
            "what": "the upstream 1.0B-token scan's rank-0 window activation, which reproduces the "
                    "shipped corpus_peak of the upstream standard-eval 512 exactly",
            "kind": "ids_column",
            "column": "corpus_peak_1b",
        },
        "sources": [
            {
                "name": "rl16",
                "kind": "sae_self",
                "n": 16,
                "path": "maemms/qwen36-27b/2026-09-18_rl-last16-lr5e-7/scores/{set}/sae_self",
            },
            {
                "name": "primary",
                "kind": "sae_self",
                "n": 16,
                # HF engine on the 2M side, so the stem is the bare set name.
                "path": "maemms/qwen36-27b/2026-09-10_rl-8x2048-full/scores/{set}/sae_self",
            },
            {
                "name": "nla",
                "kind": "sae_self",
                "n": 4,
                "path": "maemms/qwen36-27b/2026-07-14_nla-av/scores/{set}/sae_self",
            },
            {"name": "corpus", "kind": "examples_docmax", "n": CORPUS_TOP_N},
        ],
    },
    {
        "sae": "qwen36-27b/l42-1b",
        "base": "qwen36-27b",
        "set": "2026-09-16_v1",
        "root": "tmp/sae-smoke64",
        "sae_dir": "base/qwen36-27b/sae/l42-1b",
        # The v1 set carries 512 realact + 512 random + 512 sae rows; only the sae rows have a
        # feature id, and only the 64 named here are matched against the 2M side.
        "rows": "",
        "corpus_note": "the 16M document-diverse scan, one window per document",
        # The SAE repo's own shipped max-activating windows are the nearest thing to a prior
        # corpus peak for this dictionary, and `repo_examples` already scored them. Two files can
        # answer it and they are NOT the same quantity, so both are declared and which one was
        # used is printed: `repo_examples.jsonl` carries `repo_peak` per (feature, window) and its
        # max over windows IS the repo's per-feature peak; `per_feature.jsonl` carries only
        # `repo_mean_peak`, the MEAN over the shipped windows, which is smaller and inflates every
        # ratio taken against it.
        #
        # `per_feature.jsonl` is wanted even when repo_examples.jsonl supplies the peaks, because
        # it is the only file carrying OUR peak (`mean_peak_act`) and the REPO's (`repo_mean_peak`)
        # over the same windows -- which is the scale between the two conventions, and the open
        # `norm_factor` question in evals/faithfulness/README.md means it cannot be assumed to be 1.
        "alt_peak": {
            "label": "repo_max_act",
            "what": "the SAE repo's own shipped max-activating windows, as repo_examples "
                    "recorded them",
            "kind": "repo_examples",
            "dir": "base/qwen36-27b/sae/l42-1b/repo_examples/{set}",
        },
        "sources": [
            {
                "name": "rl16",
                "kind": "sae_self",
                "n": 16,
                "path": "maemms/qwen36-27b/2026-09-18_rl-last16-lr5e-7/scores/{set}/sae_self",
            },
            {
                "name": "primary",
                "kind": "sae_self",
                "n": 64,
                # The EXISTING vLLM product, at n = 64 over all 512 sae rows. It predates this
                # smoke and lives at the volume's own `maemms/...`, so its root is pinned to the
                # mirror root rather than searched: under the smoke root there is nothing, and a
                # search would only make a typo here look like an absent source.
                "root": "",
                "path": "maemms/qwen36-27b/2026-09-10_rl-8x2048-full/scores/{set}__vllm/sae_self",
            },
            {
                # The SAME product read at its first 16 rollouts, so n is matched to the 2M
                # side's primary. This is a truncation and not an estimator -- it throws away 48
                # of the 64 rollouts -- and it is here because the 2M side genuinely HAS only 16:
                # read `primary` at bo16 for the low-variance number and this row for the one
                # whose rollout budget is literally the same.
                "name": "primary16",
                "kind": "sae_self",
                "n": 64,
                "n_first": 16,
                "root": "",
                "path": "maemms/qwen36-27b/2026-09-10_rl-8x2048-full/scores/{set}__vllm/sae_self",
            },
            {
                "name": "nla",
                "kind": "sae_self",
                "n": 4,
                "path": "maemms/qwen36-27b/2026-07-14_nla-av/scores/{set}/sae_self",
            },
            {"name": "corpus", "kind": "examples_docmax", "n": CORPUS_TOP_N},
        ],
    },
]
SOURCE_ORDER = ("primary", "primary16", "rl16", "nla", "corpus")

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


# ---------------------------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------------------------


def roots_for(spec: dict, src: dict | None = None) -> list[str]:
    """The mirror prefixes a path is tried under, in order. See SPECS' own note."""
    if src is not None and "root" in src:
        return [src["root"]]
    root = spec["root"]
    return [root, ""] if root else [""]


def resolve(data: Path, roots: list[str], rel: str) -> tuple[Path, str] | tuple[None, None]:
    """(path, the root it was found under) for the first root that has `rel`, else (None, None)."""
    for root in roots:
        p = (data / root / rel) if root else (data / rel)
        if p.exists():
            return p, (root or "<mirror>")
    return None, None


# ---------------------------------------------------------------------------------------------
# readers -- every one of them tolerates an absent source and says so
# ---------------------------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def read_array(path: str | Path, dtype: str, shape):
    """`common.read_array` without importing the package: this script runs standalone."""
    return np.fromfile(path, dtype=dtype).reshape(shape)


def best_of_k_means(vals, ks) -> dict[int, float]:
    """`common.best_of_k_means`, copied for the same reason `read_array` is.

    Best-of-k by DISJOINT groups: split `vals` into floor(n/k) consecutive groups of k, take
    each group's max, average them. k above n is SKIPPED rather than clamped, so a summary never
    claims a bo-k it could not compute -- which is the whole point of reading the sources at a
    matched k here.
    """
    vals = [float(v) for v in vals]
    n = len(vals)
    out: dict[int, float] = {}
    for k in ks:
        k = int(k)
        assert k >= 1, f"best-of-k needs k >= 1, got {k}"
        if k > n:
            continue
        g = n // k
        out[k] = sum(max(vals[i * k : (i + 1) * k]) for i in range(g)) / g
    return out


def load_sae_self(d: str | Path):
    """(meta, act [N, n, W]) of a `sae_self/` product, or (None, None) when it is not there.

    `sae_self.f16` is the PRE-GATE per-token activation of each row's own target feature on its
    own rollouts, NaN outside the kept tokens; `width` is the run's scoring window + 1 (the NLA
    arm scores at 257, every other arm at 96), absent on a product written before that was
    per-run. Both come from `autointerp/sae_self.py`'s own writer.
    """
    d = Path(d)
    meta_path = d / "sae_self.json"
    if not meta_path.exists():
        return None, None
    with open(meta_path) as fh:
        meta = json.load(fh)
    width = int(meta.get("width", 96))
    shape = (len(meta["rows"]), int(meta["n"]), width)
    act = read_array(d / "sae_self.f16", "float16", shape).astype(np.float32)
    return meta, act


def load_score_rows(path: str | Path) -> dict[int, dict] | None:
    """{row: {mean_cos, max_cos, bo_4, bo_16, n_sae_gated, n}} from `per_target.jsonl`, or None.

    The file sits one level above the `sae_self/` directory, because sae_self IS a product of
    that scores directory -- so these are guaranteed to be the same rollouts, scored in the same
    pass, and no second path template can drift away from the activations.
    """
    if not os.path.exists(path):
        return None
    out = {}
    for r in read_jsonl(path):
        out[int(r["row"])] = {k: r[k] for k in SCORE_KEYS if k in r}
    return out


def load_alt_peaks(data: Path, spec: dict) -> tuple[dict | None, str]:
    """({feature: the prior work's own corpus peak}, how it was obtained) for one SAE.

    This is the denominator the card divides by, which is NOT our 16M `max_act`, so the
    comparison quotes the ratio both ways. Returns (None, why not) when it is not in the mirror.
    Every path is reported relative to `--data`, so a miss says what to fetch.
    """
    alt = spec.get("alt_peak")
    if alt is None:
        return None, "this SAE declares no prior-work corpus peak"

    if alt["kind"] == "ids_column":
        # Filled in by the caller from ids.jsonl, which it has already read.
        return None, "(filled from ids.jsonl)"

    assert alt["kind"] == "repo_examples", f"unknown alt_peak kind {alt['kind']!r}"
    d = alt["dir"].format(set=spec["set"])
    # The per-(feature, window) file FIRST: its max over windows is the repo's per-feature peak.
    p, _root = resolve(data, roots_for(spec), f"{d}/repo_examples.jsonl")
    if p is not None:
        peaks: dict[int, float] = {}
        for r in read_jsonl(p):
            f = int(r["feature"])
            peaks[f] = max(peaks.get(f, 0.0), float(r["repo_peak"]))
        return peaks, f"max of `repo_peak` over the shipped windows, {d}/repo_examples.jsonl"
    # `per_feature.jsonl` has NO per-feature repo maximum -- only `repo_mean_peak`, the mean over
    # the shipped windows. It is accepted because it is the file that is likely to be mirrored,
    # and it is labelled a MEAN everywhere it is used: a mean denominator is smaller than a peak,
    # so every ratio taken against it is inflated and none of them is `norm_act`.
    p, _root = resolve(data, roots_for(spec), f"{d}/per_feature.jsonl")
    if p is not None:
        rows = read_jsonl(p)
        if rows and "repo_mean_peak" in rows[0]:
            peaks = {int(r["feature"]): float(r["repo_mean_peak"]) for r in rows}
            return peaks, (
                f"`repo_mean_peak` (a MEAN over the shipped windows, NOT a peak -- ratios "
                f"against it are inflated), {d}/per_feature.jsonl; mirror "
                f"{d}/repo_examples.jsonl for the real per-feature maximum")
        return None, f"{d}/per_feature.jsonl carries no `repo_mean_peak` column"
    return None, (
        f"neither {d}/repo_examples.jsonl (preferred: max of `repo_peak`) nor "
        f"{d}/per_feature.jsonl is in the mirror")


def load_alt_scale(data: Path, spec: dict) -> dict:
    """Is the prior work's denominator on OUR activation scale? Measured, not assumed.

    A ratio against someone else's corpus peak is only comparable with their published one if
    their peak is the same KIND of number ours is. For the 131k SAE that is an OPEN question in
    evals/faithfulness/README.md -- whether the shipped max-acts file folds the SAE's `norm_factor` --
    and `repo_examples` is exactly the measurement of it: over the repo's OWN windows it stores
    `mean_peak_act` (our re-scored peak) beside `repo_mean_peak` (the value the repo shipped for
    the same windows), so their ratio IS the scale between the two conventions.

    Returns {"applicable", "median", "q1", "q3", "n", "how"}. `applicable` is False where the
    prior denominator is the prior work's own measurement rather than a proxy -- the upstream
    `corpus_peak_1b` column is the upstream number, not a stand-in for it, so there is nothing to scale.
    """
    alt = spec.get("alt_peak") or {}
    if alt.get("kind") != "repo_examples":
        return {"applicable": False, "median": None, "q1": None, "q3": None, "n": 0,
                "how": "the prior work's own measurement, not a proxy -- no scale to check"}
    d = alt["dir"].format(set=spec["set"])
    p, _root = resolve(data, roots_for(spec), f"{d}/per_feature.jsonl")
    if p is None:
        return {"applicable": True, "median": None, "q1": None, "q3": None, "n": 0,
                "how": f"UNMEASURED: {d}/per_feature.jsonl is not in the mirror, and it is the "
                       f"only file carrying our peak and the repo's over the SAME windows"}
    vals = []
    for r in read_jsonl(p):
        ours, theirs = r.get("mean_peak_act"), r.get("repo_mean_peak")
        # A feature whose repo windows are all dead divides by zero and says nothing about scale.
        if ours is None or not theirs:
            continue
        vals.append(float(ours) / float(theirs))
    if not vals:
        return {"applicable": True, "median": None, "q1": None, "q3": None, "n": 0,
                "how": f"UNMEASURED: {d}/per_feature.jsonl carries no feature with both "
                       f"`mean_peak_act` and a non-zero `repo_mean_peak`"}
    v = np.asarray(vals, dtype=float)
    return {
        "applicable": True,
        "median": round(float(np.median(v)), 4),
        "q1": round(float(np.quantile(v, 0.25)), 4),
        "q3": round(float(np.quantile(v, 0.75)), 4),
        "n": len(vals),
        "how": f"median `mean_peak_act` / `repo_mean_peak` over {len(vals)} features of "
               f"{d}/per_feature.jsonl -- our re-scored peak against the repo's own stored value, "
               f"on the repo's own windows",
    }


def load_peaks_1b(path: str) -> dict[int, float]:
    """{feature_id: 1.0B corpus peak} from an explicit `--peaks-1b` table.

    Accepts the bundle's own `eval_2m_features_100k_windows.parquet` (columns feature_id, rank,
    act -- the rank-0 row is the peak), a .jsonl of rows carrying a feature and a peak column, or
    a .json mapping of feature id to peak. The parquet needs pandas, which this script does not
    depend on: if it is not importable the error says to pass the jsonl/json form instead, rather
    than pulling a dataframe stack into a file that otherwise only reads numpy buffers.

    `features/bundle.py` is deliberately NOT imported to get this. It shells out to
    `modal volume get` on a cache miss, and this script is local, offline and reads only what is
    already in `--data`.
    """
    if path.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover -- depends on the caller's environment
            raise AssertionError(
                f"--peaks-1b {path} is a parquet and pandas is not available ({e}). Convert it "
                f"once (feature_id, rank, act; keep rank == 0) and pass the .jsonl or .json."
            ) from e
        cols = ["feature_id", "rank", "act"]
        df = pd.read_parquet(path, columns=cols)
        df = df[df["rank"] == 0]
        return dict(zip(df["feature_id"].tolist(), df["act"].tolist(), strict=True))
    if path.endswith(".jsonl"):
        out = {}
        for r in read_jsonl(path):
            fid = int(r.get("feature", r.get("feature_id")))
            val = r.get("corpus_peak_1b", r.get("act", r.get("corpus_peak")))
            assert val is not None, f"{path}: row for feature {fid} carries no peak column"
            out[fid] = float(val)
        return out
    with open(path) as fh:
        return {int(k): float(v) for k, v in json.load(fh).items()}


def peaks_of(act_row: np.ndarray) -> np.ndarray:
    """[n] per-rollout peak activation from one row's [n, W] block, NaN outside `keep` -> 0.

    A rollout with nothing kept (an empty generation) peaks at 0.0 and counts as not firing,
    which is what it is -- not a missing measurement.
    """
    finite = np.where(np.isfinite(act_row), act_row, -np.inf)
    pk = finite.max(axis=1)
    return np.where(np.isfinite(pk), pk, 0.0)


def corpus_windows(path: str | Path, top_n: int = CORPUS_TOP_N):
    """[k] peak activations of a feature's top corpus windows, or None when the file is absent.

    `examples_docmax` rows are one window per DOCUMENT, ranked by that document's best window
    and labelled `kind: "docmax"`; they are re-sorted here by the stored `max_act` and cut at
    `top_n` so the number is "what the strongest windows an arm would show reach". Per-token
    `acts` are in the file too and are not used: the per-window `max_act` is their max, already
    rounded the same way every other number here is read.
    """
    if not os.path.exists(path):
        return None
    rows = read_jsonl(path)
    tops = [r for r in rows if r.get("kind") == "top"] or rows
    vals = sorted((float(r["max_act"]) for r in tops), reverse=True)
    return np.array(vals[:top_n], dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------------------------


def _ratio(v, corpus_peak: float):
    return None if (v is None or corpus_peak <= 0) else round(v / corpus_peak, 4)


def summarise(vals, gate: float, corpus_peak: float, *, ranked: bool) -> dict:
    """One source's numbers for one feature. None or empty in -> {"absent": True} out.

    `vals` is one peak per ITEM: per rollout for a sae_self source, per stored window for the
    corpus source. For rollouts, bo-k is the disjoint-group best-of-k mean, so bo1 is their
    plain mean and bo16 exists only where n >= 16.

    `ranked` (the corpus source) is NOT averaged. Its items are the top windows an explainer
    would be SHOWN, already sorted by activation, so "one draw from this source" is the best
    window and not a random one: bo1 is the top window's peak, and bo4/bo16 stay None rather
    than being computed over a ranked list, where a disjoint-group max would re-read the
    ranking instead of estimating anything.
    """
    if vals is None or not len(vals):
        return {"absent": True}
    vals = np.asarray(vals, dtype=np.float64)
    peak_max = float(vals.max())
    bo = {1: peak_max} if ranked else best_of_k_means(vals, BO_KS)
    out = {
        "absent": False,
        "ranked": ranked,
        "n_items": int(len(vals)),
        "peak_max": round(peak_max, 4),
        "peak_max_ratio": _ratio(peak_max, corpus_peak),
        "fired_frac": round(float(np.mean(vals > gate)), 4),
        "fired_any": bool(peak_max > gate),
    }
    for k in BO_KS:
        v = bo.get(k)
        out[f"bo{k}"] = None if v is None else round(v, 4)
        out[f"bo{k}_ratio"] = _ratio(v, corpus_peak)
    return out


def collect(spec: dict, data: Path, want_rows: set[int] | None, peaks_1b: str = "") -> dict:
    """Every feature of one SAE's comparison, as a block of rows plus the spec's own metadata."""
    set_name = spec["set"]
    out: dict = {
        "sae": spec["sae"],
        "set": set_name,
        "root": spec["root"],
        "corpus_note": spec["corpus_note"],
        "gate": None,
        "roots_read": [],
        "features": [],
        "missing": [],
        "warnings": [],
        "reader_check": {},
    }

    # ---- the sae_self products, one per source ------------------------------------------------
    loaded: dict[str, dict] = {}
    gates: set[float] = set()
    for src in spec["sources"]:
        if src["kind"] != "sae_self":
            assert src["kind"] == "examples_docmax", f"unknown source kind {src['kind']!r}"
            continue
        rel = src["path"].format(set=set_name)
        d, root = resolve(data, roots_for(spec, src), rel)
        if d is None:
            out["missing"].append(f"{src['name']}: no sae_self.json at {rel}")
            continue
        meta, act = load_sae_self(d)
        assert meta is not None, f"{d} exists but carries no sae_self.json"
        if int(meta["n"]) != int(src["n"]):
            # Not fatal: the arrays are read at the stored n either way, and bo-k skips a k above
            # it. But the source table says which product this is supposed to be, so a different
            # n means the table and the mirror disagree and every "matched at bo16" claim below
            # has to be checked against this line.
            out["warnings"].append(
                f"{src['name']}: the table declares n = {src['n']} but {rel} was scored at "
                f"n = {meta['n']}; read at {meta['n']}")
        loaded[src["name"]] = {
            "dir": d,
            "rel": rel,
            "root": root,
            "meta": meta,
            "act": act,
            "n_first": src.get("n_first"),
            "index": {int(r): i for i, r in enumerate(meta["rows"])},
            "score": load_score_rows(d.parent / "per_target.jsonl"),
            # sae_self.json's OWN per-target reductions, kept for the reader check: they are the
            # same statistics this script recomputes from the array beside them.
            "stored": {int(p["row"]): p for p in meta.get("per_target", []) if "row" in p},
            "width": int(meta.get("width", 96)),
        }
        gates.add(round(float(meta["gate"]), 6))
        if root not in out["roots_read"]:
            out["roots_read"].append(root)
    assert len(gates) <= 1, (
        f"{spec['sae']}: the sae_self products disagree on the SAE gate {sorted(gates)} -- they "
        f"are not all the same dictionary, and their activations are not comparable"
    )
    gate = float(next(iter(gates))) if gates else float("nan")
    out["gate"] = None if math.isnan(gate) else round(gate, 6)
    for name, entry in loaded.items():
        if entry["score"] is None:
            out["missing"].append(f"{name}: no per_target.jsonl beside {entry['rel']}")

    # ---- the held-out rows: feature id and stratum ---------------------------------------------
    ids_rel = f"base/{spec['base']}/heldout/{set_name}/ids.jsonl"
    ids_path, ids_root = resolve(data, roots_for(spec), ids_rel)
    ids: dict[int, dict] = {}
    if ids_path is None:
        out["missing"].append(f"no {ids_rel} under any root -- no feature ids and no strata")
    else:
        ids = {int(r["row"]): r for r in read_jsonl(ids_path)}
        if ids_root not in out["roots_read"]:
            out["roots_read"].append(ids_root)

    # ---- corpus peaks: the denominator of every ratio here --------------------------------------
    peak_rel = f"{spec['sae_dir']}/max_act.f16"
    peak_path, _ = resolve(data, roots_for(spec), peak_rel)
    peak_tab = None
    if peak_path is None:
        out["missing"].append(f"no {peak_rel} under any root -- every ratio is null")
    else:
        peak_tab = read_array(peak_path, "float16", (-1,)).astype(np.float32)

    # ---- the PRIOR WORK's corpus peak, which is a different denominator -------------------------
    alt = spec.get("alt_peak")
    alt_peaks, alt_how = load_alt_peaks(data, spec)
    if alt is not None and alt["kind"] == "ids_column":
        col = alt["column"]
        alt_peaks = {
            int(r["id"]): float(r[col])
            for r in ids.values()
            if r.get("id") is not None and r.get(col) is not None
        }
        alt_how = f"the `{col}` column of {ids_rel}"
        if not alt_peaks:
            alt_peaks, alt_how = None, (
                f"{ids_rel} carries no non-null `{col}` (the draw ran on a mirror without the "
                f"1.0B window table) -- pass --peaks-1b")
    if peaks_1b and alt is not None and alt["kind"] == "ids_column":
        alt_peaks = load_peaks_1b(peaks_1b)
        alt_how = f"--peaks-1b {peaks_1b}"
    out["alt_peak"] = {
        "label": (alt or {}).get("label"),
        "what": (alt or {}).get("what"),
        "how": alt_how,
        "n_features": None if alt_peaks is None else len(alt_peaks),
        # Whether that denominator is on our activation scale at all -- measured from the one
        # product that has both conventions on the same windows.
        "scale": load_alt_scale(data, spec),
    }

    # ---- the rows to report ------------------------------------------------------------------
    rows_seen: set[int] = set()
    for entry in loaded.values():
        rows_seen.update(entry["index"])
    rows = sorted(r for r in rows_seen if want_rows is None or r in want_rows)
    if want_rows is not None and not rows and rows_seen:
        out["warnings"].append(
            f"--rows selected none of the {len(rows_seen)} rows these products carry "
            f"({min(rows_seen)}..{max(rows_seen)})")

    for row in rows:
        rec = ids.get(row, {})
        feat = int(rec.get("id", -1))
        cpeak = (
            float(peak_tab[feat])
            if (peak_tab is not None and 0 <= feat < len(peak_tab))
            else 0.0
        )
        apeak = None if alt_peaks is None else alt_peaks.get(feat)
        entry = {
            "row": row,
            "feature": feat,
            "stratum": rec.get("stratum"),
            "stratum_stat": rec.get("stratum_stat"),
            "sae_key": rec.get("sae_key"),  # present only on sets drawn after 2026-09-21
            "family": rec.get("family"),
            "corpus_peak": round(cpeak, 4),
            # The prior work's own denominator for this feature; None where it is not reachable.
            "alt_peak": None if apeak is None else round(float(apeak), 4),
            "sources": {},
            "score": {},
            "stored": {},
        }
        for src in spec["sources"]:
            name = src["name"]
            if src["kind"] == "examples_docmax":
                rel = f"{spec['sae_dir']}/examples_docmax/{set_name}/{feat}.jsonl"
                p, _root = resolve(data, roots_for(spec, src), rel)
                vals = corpus_windows(p) if p is not None else None
                entry["sources"][name] = summarise(vals, gate, cpeak, ranked=True)
                continue
            ent = loaded.get(name)
            vals = None
            if ent is not None and row in ent["index"]:
                vals = peaks_of(ent["act"][ent["index"][row]])
                if ent["n_first"]:
                    vals = vals[: ent["n_first"]]
            s = summarise(vals, gate, cpeak, ranked=False)
            # The same statistics against the PRIOR WORK's denominator, so the card comparison can
            # quote both without re-reading anything.
            if vals is not None and apeak:
                for key in RATIO_KEYS:
                    base = s[key.removesuffix("_ratio")]
                    s[f"alt_{key}"] = None if base is None else round(base / float(apeak), 4)
            entry["sources"][name] = s
            # per_target.jsonl: JSON only (except the reader check's own table).
            if ent is not None and ent["score"] is not None and row in ent["score"]:
                entry["score"][name] = ent["score"][row]
            # sae_self.json's own reduction of the same array, for the reader check.
            if ent is not None and row in ent["stored"] and not ent["n_first"]:
                entry["stored"][name] = ent["stored"][row]
        out["features"].append(entry)
    out["sources_read"] = {
        name: {"path": e["rel"], "root": e["root"], "n": int(e["meta"]["n"]),
               "n_first": e["n_first"], "width": e["width"], "rows": len(e["index"]),
               "per_target": e["score"] is not None}
        for name, e in loaded.items()
    }
    out["reader_check"] = reader_check(out)
    return out


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------

RATIO_KEYS = tuple(f"bo{k}_ratio" for k in BO_KS) + ("peak_max_ratio",)


def _agg_over(entries: list[dict]) -> dict:
    """Median/mean of each ratio and the two firing rates, over features that HAVE the source."""
    have = [s for s in entries if not s.get("absent")]
    if not have:
        return {}
    agg: dict = {"n_features": len(have)}
    for key in RATIO_KEYS:
        vals = np.array([s[key] for s in have if s.get(key) is not None], dtype=float)
        agg[key] = {
            "n": int(len(vals)),
            "median": (round(float(np.median(vals)), 4) if len(vals) else None),
            "mean": (round(float(np.mean(vals)), 4) if len(vals) else None),
        }
    agg["frac_features_firing"] = round(float(np.mean([s["fired_any"] for s in have])), 4)
    agg["mean_item_fired_frac"] = round(float(np.mean([s["fired_frac"] for s in have])), 4)
    return agg


def aggregate(block: dict) -> dict:
    """Per source, and per (source, stratum). A stratum of None is grouped under `"-"`."""
    agg: dict[str, dict] = {}
    strata = sorted({str(f["stratum"]) for f in block["features"]})
    for name in SOURCE_ORDER:
        entries = [f["sources"][name] for f in block["features"] if name in f["sources"]]
        overall = _agg_over(entries)
        if not overall:
            continue
        per_stratum = {}
        for s in strata:
            sub = [
                f["sources"][name]
                for f in block["features"]
                if name in f["sources"] and str(f["stratum"]) == s
            ]
            got = _agg_over(sub)
            if got:
                per_stratum[s] = got
        agg[name] = {"all": overall, "by_stratum": per_stratum}
    return agg


# ---------------------------------------------------------------------------------------------
# comparison against prior results
# ---------------------------------------------------------------------------------------------

# float16 keeps 11 bits of mantissa, so its relative resolution is 2^-11; both sides are then
# rounded to 4 decimals. `_f16_bound_holds()` checks the constant rather than trusting it.
F16_EPS = 2.0**-11
ROUND_EPS = 1e-4

# What `sae_self.json`'s own per-target summary calls each statistic this script recomputes, and
# how far the two may legitimately differ.
#
# NOT bit-exact, and this is the reason. `autointerp/sae_self.py` builds its activations in a
# float32 buffer (`_SelfAct.act = torch.full(..., nan)`), reduces THAT buffer into `per_target`,
# and only then casts it to `sae_self.f16` on the way out. This script has nothing but the f16
# file, so the same reduction over the same numbers can land one f16 ulp away. Demanding equality
# would print a defect on every row of a healthy product.
#
#   "f16"    within one f16 ulp of the stored value, plus the 4-decimal rounding on both sides
#   "gate"   a COUNT over `peak > gate`: an f16 ulp can move a peak sitting on the gate across
#            it, so a difference of up to one rollout in n is the resolution, not an error
#   "exact"  read from the same file on both sides, so there is nothing to round
STORED_PAIRS = (
    ("max_peak_act", "peak_max", "f16"),
    ("mean_peak_act", "bo1", "f16"),  # bo1 IS the mean over the row's per-rollout peaks
    ("fire_fraction", "fired_frac", "gate"),
    ("n", "n_items", "exact"),
    ("corpus_peak", None, "exact"),  # the feature's own corpus_peak, not a source statistic
)


def _f16_bound_holds() -> bool:
    """Is F16_EPS actually an upper bound on the float32 -> float16 relative error?"""
    rng = np.random.default_rng(0)
    x = rng.uniform(1e-3, 1e3, size=20000).astype(np.float32)
    return bool(np.all(np.abs(x.astype(np.float16).astype(np.float32) - x) <= F16_EPS * np.abs(x)))


def _agrees(kind: str, stored: float, ours: float, n_items: int) -> tuple[bool, float]:
    """(does this pair agree within what the storage allows, the tolerance used)."""
    if kind == "exact":
        return float(stored) == float(ours), 0.0
    if kind == "gate":
        tol = 1.0 / max(n_items, 1) + 1e-9
    else:
        tol = F16_EPS * max(abs(float(stored)), abs(float(ours))) + 2 * ROUND_EPS
    return abs(float(stored) - float(ours)) <= tol, tol


def reader_check(block: dict) -> dict:
    """Our recomputation from `sae_self.f16` against `sae_self.json`'s own `per_target`.

    Two reductions of the same activations, one written by `autointerp/sae_self.py` and one done
    here. Anything outside the storage tolerance above is a defect in this reader or in that
    product -- never a finding about a model -- and is reported as one. `exact` counts the pairs
    that matched bit for bit, which is what most of them do; `within` counts the rest that stayed
    inside the tolerance, and a run where `within` is large is itself worth a look.

    Rows read with `n_first` are skipped: a truncated read is deliberately not the stored
    statistic, and comparing them would manufacture a failure.
    """
    out: dict = {}
    for name in SOURCE_ORDER:
        rows, stats, exact, bad = 0, 0, 0, []
        for f in block["features"]:
            stored = f.get("stored", {}).get(name)
            src = f["sources"].get(name)
            if stored is None or src is None or src.get("absent"):
                continue
            here = 0
            for key, ours_key, kind in STORED_PAIRS:
                if key not in stored:
                    continue
                here += 1
                ours = f["corpus_peak"] if ours_key is None else src[ours_key]
                if ours is None:
                    continue
                ok, tol = _agrees(kind, stored[key], ours, src.get("n_items", 1))
                exact += float(stored[key]) == float(ours)
                if not ok:
                    bad.append({"row": f["row"], "feature": f["feature"], "stat": key,
                                "stored": stored[key], "ours": ours,
                                "tolerance": round(tol, 8)})
            # A row whose `per_target` entry carries none of these statistics is not a row that
            # was checked, and counting it would report a coverage this never had.
            rows += bool(here)
            stats += here
        if rows:
            out[name] = {"rows": rows, "stats": stats, "exact": int(exact),
                         "mismatches": bad[:20], "n_mismatches": len(bad)}
    return out


def _flag(ours, theirs, values) -> dict:
    """Is `theirs` inside `ours` +/- FLAG_Z standard errors of OUR OWN mean over `values`?

    The band is deliberately one-sided about whose error it counts: the prior result's own
    sampling error is NOT in it and the two feature sets are different draws, so this catches a
    gross disagreement and says nothing about a small one. That is stated wherever it prints.
    """
    if ours is None or theirs is None:
        return {"delta": None, "z": None, "verdict": "—"}
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    delta = round(float(ours) - float(theirs), 4)
    if se <= 0:
        return {"delta": delta, "z": None, "verdict": "no SE (n < 2 or zero spread)"}
    z = abs(delta) / se
    return {
        "delta": delta,
        "z": round(z, 2),
        "verdict": ("CHECK — possible pipeline defect" if z > FLAG_Z else "consistent"),
    }


def _source_values(block: dict, source: str, key: str) -> list:
    return [
        f["sources"][source][key]
        for f in block["features"]
        if source in f["sources"] and not f["sources"][source].get("absent")
        and f["sources"][source].get(key) is not None
    ]


def compare(block: dict) -> dict:
    """The three prior-result checks for one SAE, each `absent` with a reason where it cannot run."""
    sae = block["sae"]
    agg = block["aggregate"]
    out: dict = {"card": None, "reader": None, "act_smoke": []}

    # ---- (1) the card --------------------------------------------------------------------
    card = CARD.get(sae)
    if card is None:
        out["card"] = {"absent": "no card number is recorded for this SAE"}
    elif "rl16" not in agg:
        out["card"] = {"absent": "the rl-last16 source is not in the mirror, so there is nothing "
                                 "to compare the upstream rl-last16 arm with"}
    else:
        a = agg["rl16"]["all"]
        is_ratio = card["ours"].endswith("_ratio")
        ours = a[card["ours"]]["median"] if is_ratio else a[card["ours"]]
        vals = (_source_values(block, "rl16", card["ours"]) if is_ratio
                else _source_values(block, "rl16", "fired_frac"))
        row = {
            "arm": card["arm"],
            "stat": card["stat"],
            "theirs": card["value"],
            "denominator": card["denominator"],
            "note": card["note"],
            "ours_key": card["ours"],
            "ours_vs_16m": ours,
            "ours_mean_vs_16m": a[card["ours"]]["mean"] if is_ratio else ours,
            "n_features": a["n_features"],
            **_flag(ours, card["value"], vals),
        }
        if is_ratio:
            alt_key = f"alt_{card['ours']}"
            alt_vals = _source_values(block, "rl16", alt_key)
            row["alt_peak_how"] = block["alt_peak"]["how"]
            row["ours_vs_their_peak"] = (
                round(float(np.median(alt_vals)), 4) if alt_vals else None)
            row["n_alt"] = len(alt_vals)
            # The comparable number is the one taken against THE UPSTREAM denominator; the flag follows it
            # whenever it exists, and falls back to ours with that said in the verdict.
            scale = block["alt_peak"].get("scale") or {}
            row["scale"] = scale
            row["denominator_label"] = "ours (÷ the upstream peak)"
            if alt_vals:
                row.update(_flag(row["ours_vs_their_peak"], card["value"], alt_vals))
                row["flag_basis"] = "the upstream denominator"
                # A ratio against a PROXY denominator is only the upstream statistic if the proxy is on
                # our scale. Where it is not -- or where nothing measured it -- the disagreement
                # is about the denominator and must not be reported as a pipeline verdict. The
                # difference and the z stay, because the reader still wants to see them.
                if scale.get("applicable"):
                    med = scale.get("median")
                    ok = med is not None and SCALE_OK[0] <= med <= SCALE_OK[1]
                    if not ok:
                        shown = "unmeasured" if med is None else f"scale {med:.3f}"
                        row["denominator_label"] = f"÷ repo peak ({shown}, unverified)"
                        why = ("scale unmeasured" if med is None else "denominator scale differs")
                        row["verdict"] = (
                            f"CHECK — {why}, not a pipeline verdict"
                            if row["verdict"].startswith("CHECK")
                            else f"{row['verdict']} — on an unverified denominator scale")
                        row["flag_basis"] = (
                            f"a PROXY for the upstream denominator, whose scale against ours is "
                            f"{'unmeasured' if med is None else f'{med:.3f}'}")
            else:
                row["flag_basis"] = "OUR 16M denominator (upstream is not in the mirror)"
        else:
            row["flag_basis"] = "a firing rate: no denominator either way"
        out["card"] = row

    # ---- (2) the reader check ------------------------------------------------------------------
    rc = block.get("reader_check") or {}
    if not rc:
        out["reader"] = {"absent": "no sae_self.json in the mirror carries a `per_target` block"}
    else:
        out["reader"] = rc

    # ---- (3) yesterday's act_smoke -------------------------------------------------------------
    for ref in ACT_SMOKE_REF["rows"]:
        if ref["sae"] != sae:
            continue
        source, key = ref["ours"]
        row = {**{k: ref[k] for k in ("arm", "median_ratio", "firing", "comparable", "why")},
               "ours_key": f"{source}.{key}"}
        if source not in agg:
            row["absent"] = f"the `{source}` source is not in the mirror"
        else:
            ours = agg[source]["all"][key]["median"]
            row["ours"] = ours
            row["n_features"] = agg[source]["all"]["n_features"]
            row["firing_ours"] = agg[source]["all"]["frac_features_firing"]
            if ref["comparable"]:
                row.update(_flag(ours, ref["median_ratio"], _source_values(block, source, key)))
            else:
                row.update({"delta": None, "z": None,
                            "verdict": "not comparable — see the note"})
        out["act_smoke"].append(row)
    return out


# ---------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------

_HEAD = ["source", "features"] + [
    f"{k} {w}" for k in ("bo1", "bo4", "bo16", "peak") for w in ("med", "mean")
] + ["firing", "item fired"]


def _num(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _frac(text: str) -> str:
    """"4/16" -> "4/16 (0.250)", so a recorded count reads against a rate without arithmetic."""
    a, _, b = text.partition("/")
    return f"{text} ({int(a) / int(b):.3f})" if b.isdigit() and a.isdigit() else text


def _agg_cells(label: str, a: dict) -> list[str]:
    cells = [label, str(a["n_features"])]
    for key in RATIO_KEYS:
        cells += [_num(a[key]["median"]), _num(a[key]["mean"])]
    cells += [_num(a["frac_features_firing"]), _num(a["mean_item_fired_frac"])]
    return cells


def _cell(s: dict, key: str) -> str:
    if s.get("absent"):
        return "absent"
    v = s.get(key)
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return (
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def render(blocks: list[dict]) -> str:
    lines = [
        "# sae_smoke64 — raw peak activation per source, at a matched best-of-k",
        "",
        "Every number is READ from what `sae_self` and `examples_docmax` already measured on the "
        "clean base; nothing here recomputes an activation. A `ratio` is the statistic divided by "
        "`corpus_peak`, the feature's maximum over the 16M corpus scan "
        "(`sae/<sae>/max_act.f16`). `bo-k` is the disjoint-group best-of-k mean over a source's "
        "draws (`common.best_of_k_means`), so **bo1 is the plain mean over draws** and bo16 exists "
        "only where the source has 16. The `corpus` source is a RANKED list of the top "
        f"{CORPUS_TOP_N} windows, so its bo1 is the top window's peak and it has no bo-k. "
        "`firing` is the fraction of features whose source fires at all; `item fired` is the mean "
        "over features of the fraction of that source's own draws above the gate. An absent "
        "source prints `absent`, never 0.",
        "",
        "Cosines are NOT in these tables — they are in the `.json` beside this file, per feature "
        "and per source, from each source's own `per_target.jsonl`.",
        "",
    ]
    for b in blocks:
        present = [
            s
            for s in SOURCE_ORDER
            if any(s in f["sources"] and not f["sources"][s]["absent"] for f in b["features"])
        ]
        lines += [
            f"## {b['sae']} — set `{b['set']}`",
            "",
            f"- gate: {b['gate']}",
            f"- mirror roots read: {', '.join(b['roots_read']) or '(none)'} "
            f"(spec root `{b['root'] or '<mirror>'}`)",
            f"- corpus source: `examples_docmax` — {b['corpus_note']}, top {CORPUS_TOP_N} windows "
            f"by stored `max_act`",
            f"- features: {len(b['features'])}",
        ]
        for name, info in b.get("sources_read", {}).items():
            cut = f", read at its first {info['n_first']}" if info["n_first"] else ""
            lines.append(
                f"- `{name}`: n={info['n']}{cut}, width={info['width']}, {info['rows']} rows, "
                f"per_target {'yes' if info['per_target'] else 'NO'} — `{info['path']}` "
                f"@ {info['root']}"
            )
        if b["missing"]:
            lines.append(f"- MISSING (reported, never counted as zero): {'; '.join(b['missing'])}")
        if b["warnings"]:
            lines.append(f"- WARNING: {'; '.join(b['warnings'])}")
        lines += ["", "### Per source", ""]
        lines += _table(_HEAD, [_agg_cells(s, b["aggregate"][s]["all"]) for s in present])
        lines += ["", "### Per source and stratum", ""]
        strat_rows = []
        for s in present:
            for stratum, a in sorted(b["aggregate"][s]["by_stratum"].items()):
                strat_rows.append(_agg_cells(f"{s} q{stratum}", a))
        if strat_rows:
            lines += _table(["source / stratum"] + _HEAD[1:], strat_rows)
        else:
            lines += ["(no strata on these rows — the held-out set carries no `stratum`)"]
        lines += ["", "### Per feature", ""]
        head = ["feature", "row", "stratum", "corpus_peak"]
        for s in present:
            head += [f"{s} bo1 r", f"{s} peak", f"{s} peak r", f"{s} fired"]
        feat_rows = []
        for f in b["features"]:
            cells = [
                str(f["feature"]),
                str(f["row"]),
                str(f["stratum"]),
                f"{f['corpus_peak']:.3f}",
            ]
            for s in present:
                src = f["sources"].get(s, {"absent": True})
                cells += [
                    _cell(src, "bo1_ratio"),
                    _cell(src, "peak_max"),
                    _cell(src, "peak_max_ratio"),
                    _cell(src, "fired_frac"),
                ]
            feat_rows.append(cells)
        lines += _table(head, feat_rows)
        lines += [""]
    lines += [
        "## Reading the two SAEs against each other",
        "",
        "Read them at the k they BOTH reach. The 131k `primary` is a 64-rollout vLLM product and "
        "the 2M `primary` is 16 rollouts, so bo16 is the last matched column, `primary16` is the "
        "same product truncated to a literally equal rollout budget, and bo1 is the one column "
        "free of any max-over-n bias at all.",
        "",
        "The 2M block's 64 features are 16 per density quartile by construction, so its **all** "
        "row is a mean over four equally-weighted quartiles and not over the dictionary. The "
        "per-stratum rows are the comparable ones; the dictionaries' own density distributions "
        "differ and nothing here reweights one onto the other.",
        "",
    ]
    lines += render_comparison(blocks)
    return "\n".join(lines)


def render_comparison(blocks: list[dict]) -> list[str]:
    """The three prior-result checks, per SAE. Each says `absent` and why where it cannot run."""
    lines = [
        "## Comparison against prior results",
        "",
        f"Every difference below is flagged when it exceeds {FLAG_Z:g} standard errors of OUR "
        f"mean over these features. That band counts only our own sampling error, not the prior "
        f"result's, and the two are different feature draws in every case — so `CHECK` means "
        f"\"look at this\" and `consistent` means \"not grossly wrong\", neither more.",
        "",
    ]
    for b in blocks:
        cmp = b["comparison"]
        lines += [f"### {b['sae']}", ""]

        # ---- (1) the card ----------------------------------------------------------------------
        lines += ["**(1) the card, rl-last16 at bo4.**", ""]
        card = cmp["card"]
        if card is None or "absent" in card:
            lines += [f"absent — {(card or {}).get('absent', 'not computed')}", ""]
        else:
            lines += [
                f"The upstream `{card['stat']}` = {card['theirs']} on the upstream 512-feature set, divided by "
                f"{card['denominator']}. Ours is over {card['n_features']} features.",
                "",
            ]
            if card["note"]:
                lines += [card["note"], ""]
            scale = card.get("scale") or {}
            if scale.get("applicable"):
                med = scale.get("median")
                sent = (
                    "**The upstream 1.0B-scan corpus peak for this SAE is not in our data.** The proxy "
                    "is the SAE repo's own shipped max-activating windows, and whether those are "
                    "on our activation scale is an OPEN question in `evals/faithfulness/README.md` "
                    "(does the shipped max-acts file fold the SAE's `norm_factor`), so "
                    "`repo_examples` measured it: ")
                if med is None:
                    sent += f"{scale['how']}."
                else:
                    sent += (
                        f"the median of our `mean_peak_act` over the repo's own stored "
                        f"`repo_mean_peak`, on the repo's own windows, is **{med:.3f}** "
                        f"(IQR {scale['q1']:.3f}–{scale['q3']:.3f} over {scale['n']} features).")
                    if not SCALE_OK[0] <= med <= SCALE_OK[1]:
                        sent += (
                            f" That is outside {SCALE_OK[0]:g}–{SCALE_OK[1]:g}, so the two "
                            f"conventions are not the same quantity: the `÷ repo peak` column "
                            f"below is NOT comparable with a published `norm_act`, and the "
                            f"verdict is about the denominator rather than about the pipeline.")
                    else:
                        sent += (
                            f" That is inside {SCALE_OK[0]:g}–{SCALE_OK[1]:g}, so the repo's "
                            f"peaks and ours are on one scale and the `÷ repo peak` column is "
                            f"read as the upstream statistic.")
                lines += [sent, ""]
            head = ["statistic", "ours (÷ 16M max_act)",
                    card.get("denominator_label", "ours (÷ the upstream peak)"), "upstream", "Δ", "z",
                    "verdict"]
            stat_label = (f"`{card['ours_key']}` median" if "ours_vs_their_peak" in card
                          else f"`{card['ours_key']}` (a mean over features)")
            lines += _table(head, [[
                stat_label,
                _num(card["ours_vs_16m"]),
                (_num(card.get("ours_vs_their_peak")) if "ours_vs_their_peak" in card
                 else "n/a (a firing rate)"),
                _num(card["theirs"]),
                _num(card["delta"]),
                ("—" if card["z"] is None else f"{card['z']:.2f}"),
                card["verdict"],
            ]])
            lines += [
                "",
                f"- flag taken against: {card['flag_basis']}",
            ]
            if "alt_peak_how" in card:
                lines += [
                    f"- the upstream denominator, per feature: {card['alt_peak_how']} "
                    f"({card['n_alt']} of {card['n_features']} features have one)",
                ]
            lines += [""]

        # ---- (2) the reader check ---------------------------------------------------------------
        lines += ["**(2) Reader check — `sae_self.json`'s own `per_target` against our "
                  "recomputation from `sae_self.f16`.**", ""]
        reader = cmp["reader"]
        if reader is None or "absent" in reader:
            lines += [f"absent — {(reader or {}).get('absent', 'not computed')}", ""]
        else:
            lines += [
                "Two reductions of the same activations. They are NOT bit-identical by "
                "construction: `sae_self.py` reduces its float32 buffer and casts to "
                "`sae_self.f16` afterwards, while this script has only the f16 file, so a pair "
                "may sit one f16 ulp apart. `exact` counts the pairs that matched bit for bit "
                "anyway; a `mismatch` is outside even that tolerance and is a defect in this "
                "reader or in that product — never a result.",
                "",
            ]
            rows = []
            for name, rc in reader.items():
                rows.append([
                    name, str(rc["rows"]), str(rc["stats"]),
                    f"{rc['exact']}/{rc['stats']}", str(rc["n_mismatches"]),
                    ("agrees" if rc["n_mismatches"] == 0 else "POSSIBLE PIPELINE DEFECT"),
                ])
            lines += _table(
                ["source", "rows checked", "values compared", "bit-exact", "mismatches",
                 "verdict"], rows)
            lines += [""]
            for name, rc in reader.items():
                for m in rc["mismatches"]:
                    lines.append(
                        f"- MISMATCH {name} row {m['row']} (feature {m['feature']}) "
                        f"`{m['stat']}`: stored {m['stored']}, recomputed {m['ours']}, "
                        f"outside a tolerance of {m['tolerance']}")
            lines += [""]
            # The stored product's own score columns, listed beside, for the rows it covers.
            listed = [
                (f["row"], f["feature"], f["score"]["primary"])
                for f in b["features"]
                if "primary" in f.get("score", {})
            ]
            if listed:
                lines += [
                    "`scores/<set>__vllm/per_target.jsonl` for the same rows — `n_sae_gated` is "
                    "how many of the row's rollouts the stored SAE CSR holds at their argmax "
                    "token, and `bo_4` is that product's own best-of-4 COSINE. This is the one "
                    "place a cosine appears outside the JSON: here it is the artefact being "
                    "checked, not a result.",
                    "",
                ]
                lines += _table(
                    ["row", "feature", "n", "n_sae_gated", "bo_4 (cos)", "mean_cos", "max_cos"],
                    [[str(r), str(f), str(s.get("n", "—")), str(s.get("n_sae_gated", "—")),
                      _num(s.get("bo_4")), _num(s.get("mean_cos")), _num(s.get("max_cos"))]
                     for r, f, s in listed],
                )
                lines += [""]

        # ---- (3) yesterday's act_smoke -----------------------------------------------------------
        lines += [
            f"**(3) The 16-feature `act_smoke` medians of 2026-09-21.** Provenance: "
            f"{ACT_SMOKE_REF['provenance']}.",
            "",
        ]
        if not cmp["act_smoke"]:
            lines += ["absent — no act_smoke row is recorded for this SAE", ""]
        else:
            rows = []
            for r in cmp["act_smoke"]:
                rows.append([
                    r["arm"],
                    f"`{r['ours_key']}`",
                    _num(r["median_ratio"]),
                    (r.get("absent") or _num(r.get("ours"))),
                    _frac(r["firing"]),
                    (r.get("absent") and "—") or _num(r.get("firing_ours")),
                    _num(r.get("delta")),
                    ("—" if r.get("z") is None else f"{r['z']:.2f}"),
                    r.get("verdict", "—"),
                ])
            lines += _table(
                ["act_smoke arm", "this run's column", "then (median)", "now (median)",
                 "then firing", "now firing", "Δ", "z", "verdict"],
                rows,
            )
            lines += [""]
            for r in cmp["act_smoke"]:
                lines.append(f"- {r['arm']}: {r['why']}")
            lines += [""]
    return lines


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------


def parse_rows(spec_str: str) -> set[int] | None:
    """'0,1,4-6' -> {0,1,4,5,6}; '' -> None (every row the products carry)."""
    if not spec_str:
        return None
    want: set[int] = set()
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
    return want


def analyse(data: Path, sae: str = "", rows: str = "", root: str = "",
            peaks_1b: str = "") -> list[dict]:
    blocks = []
    for spec in SPECS:
        if sae and spec["sae"] != sae:
            continue
        spec = {**spec, "root": root or spec["root"]}
        # --rows overrides the spec's own default for EVERY spec, which is how the 512-row v1 set
        # is cut down to the 64 sae rows this smoke compares.
        b = collect(spec, data, parse_rows(rows or spec["rows"]), peaks_1b)
        b["aggregate"] = aggregate(b)
        b["comparison"] = compare(b)
        blocks.append(b)
    assert blocks, f"--sae {sae!r} matched none of {[s['sae'] for s in SPECS]}"
    return blocks


@app.command()
def main(
    data: Annotated[Path | None, typer.Option(help="local mirror; its subpaths are the volume's own")] = None,
    out: Annotated[Path | None, typer.Option(help="markdown out; <out>.json lands beside it")] = None,
    sae: Annotated[str, typer.Option(help="restrict to one SAE config key")] = "",
    rows: Annotated[str, typer.Option(help="restrict to these held-out rows, e.g. 0,1,4-6")] = "",
    root: Annotated[str, typer.Option(help="override every spec's mirror root prefix")] = "",
    peaks_1b: Annotated[str, typer.Option(
        "--peaks-1b",
        help="the 1.0B-token corpus peak per 2M feature, when ids.jsonl carries none: the "
             "bundle's eval_2m_features_100k_windows.parquet, or a .jsonl/.json of the same",
    )] = "",
    selftest: Annotated[bool, typer.Option(help="run the synthetic-mirror checks and exit")] = False,
):
    if selftest:
        run_selftest()
        return
    assert data is not None, "--data <dir> is required (the local mirror of the volume)"
    blocks = analyse(data, sae, rows, root, peaks_1b)
    md = render(blocks)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        with open(out.with_suffix(out.suffix + ".json"), "w") as fh:
            json.dump(blocks, fh, indent=1)
        print(f"[sae_smoke64] wrote {out} and {out.with_suffix(out.suffix + '.json')}")
    for b in blocks:
        print(f"\n== {b['sae']} ({b['set']}), {len(b['features'])} features, gate {b['gate']}")
        for s, a in b["aggregate"].items():
            o = a["all"]
            print(
                f"   {s:<9} n={o['n_features']:<4} bo1 {_num(o['bo1_ratio']['median'])}  "
                f"bo4 {_num(o['bo4_ratio']['median'])}  bo16 {_num(o['bo16_ratio']['median'])}  "
                f"peak {_num(o['peak_max_ratio']['median'])}  "
                f"firing {_num(o['frac_features_firing'])}   (median ratios)"
            )
        for m in b["missing"]:
            print(f"   MISSING {m}")
        for w in b["warnings"]:
            print(f"   WARNING {w}")
        cmp = b["comparison"]
        card = cmp["card"] or {}
        if "absent" in card:
            print(f"   card      absent: {card['absent']}")
        else:
            mine = card.get("ours_vs_their_peak") or card.get("ours_vs_16m")
            print(f"   card      {card['stat']} ours {_num(mine)} vs {card['theirs']} "
                  f"({card['flag_basis']}) -> {card['verdict']}")
        reader = cmp["reader"] or {}
        if "absent" in reader:
            print(f"   reader    absent: {reader['absent']}")
        for name, rc in reader.items():
            if name == "absent":
                continue
            defect = "" if not rc["n_mismatches"] else "  <-- POSSIBLE PIPELINE DEFECT"
            print(f"   reader    {name}: {rc['rows']} rows, {rc['n_mismatches']} "
                  f"mismatches{defect}")
        for r in cmp["act_smoke"]:
            now = r.get("absent") or _num(r.get("ours"))
            print(f"   act_smoke {r['arm']:<34} then {_num(r['median_ratio'])}  now {now}  "
                  f"-> {r.get('verdict', '—')}")


# ---------------------------------------------------------------------------------------------
# --selftest: the whole pipeline on a synthetic mirror, numbers checked by hand
# ---------------------------------------------------------------------------------------------


def _write_sae_self(d: Path, rows_: list[int], n: int, width: int, gate: float, act: np.ndarray,
                    peaks: list[dict] | None = None):
    """A synthetic sae_self product. `peaks` is sae_self.json's own `per_target` block, which the
    reader check compares against this script's recomputation of the SAME array."""
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "sae_self.json", "w") as fh:
        json.dump(
            {
                "rows": rows_,
                "features": [100 + r for r in rows_],
                "n": n,
                "gate": gate,
                "width": width,
                "per_target": peaks if peaks is not None else [{"row": r} for r in rows_],
            },
            fh,
        )
    act.astype(np.float16).tofile(d / "sae_self.f16")


def _stored_of(rows_: list[int], act: np.ndarray, gate: float, cpeaks: dict) -> list[dict]:
    """What `autointerp/sae_self.py` would write into `per_target` for this array, by its own
    formulas -- so the selftest's reader check compares two independent computations."""
    out = []
    for i, r in enumerate(rows_):
        pk = peaks_of(act[i])
        out.append({
            "row": r, "feature": 100 + r, "n": int(act.shape[1]),
            "corpus_peak": round(float(cpeaks[100 + r]), 4),
            "fire_fraction": round(float((pk > gate).mean()), 4),
            "max_peak_act": round(float(pk.max()), 4),
            "mean_peak_act": round(float(pk.mean()), 4),
        })
    return out


def _write_per_target(d: Path, rows_: list[int]):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "per_target.jsonl", "w") as fh:
        for r in rows_:
            fh.write(json.dumps({
                "row": r, "family": "sae", "n": 16, "n_sae_gated": 10 + r,
                "mean_cos": 0.1 + r / 100, "max_cos": 0.5 + r / 100,
                "bo_4": 0.3 + r / 100, "bo_16": 0.4 + r / 100,
            }) + "\n")


def _write_examples(d: Path, feat: int, vals: list[float]):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{feat}.jsonl", "w") as fh:
        for i, v in enumerate(vals):
            fh.write(json.dumps({
                "row": 0, "kind": "docmax", "window": i, "doc": i, "start": 0, "len": 4,
                "max_act": v, "argmax": 0, "acts": [v, 0.0, 0.0, 0.0],
            }) + "\n")


def _block_from_peaks(peaks: list[list[float]], width: int = 3) -> np.ndarray:
    """[N, n, width] whose per-rollout peak is exactly `peaks[i][k]`, NaN in the padded tail."""
    n = len(peaks[0])
    a = np.full((len(peaks), n, width), np.nan, dtype=np.float64)
    for i, row in enumerate(peaks):
        for k, v in enumerate(row):
            a[i, k, 0] = 0.0
            a[i, k, 1] = v
    return a


def run_selftest() -> None:  # noqa: PLR0915 -- one linear scenario, split would hide the numbers
    gate = 2.0
    with tempfile.TemporaryDirectory() as td:
        data = Path(td)
        spec = {
            "sae": "t/sae",
            "base": "b",
            "set": "S",
            "root": "smoke",
            "sae_dir": "base/b/sae/sae",
            "rows": "",
            "corpus_note": "synthetic",
            "alt_peak": {"label": "corpus_peak_1b", "what": "synthetic prior peaks",
                         "kind": "ids_column", "column": "corpus_peak_1b"},
            "sources": [
                {"name": "primary", "kind": "sae_self", "n": 16,
                 "path": "maemms/p/scores/{set}/sae_self"},
                # the same product truncated to its first 4 rollouts: n matched, not estimated
                {"name": "primary16", "kind": "sae_self", "n": 16, "n_first": 4,
                 "path": "maemms/p/scores/{set}/sae_self"},
                {"name": "rl16", "kind": "sae_self", "n": 16,
                 "path": "maemms/r/scores/{set}/sae_self"},
                # pinned to the mirror root: it exists ONLY there, and must not be searched for
                # under the smoke root
                {"name": "nla", "kind": "sae_self", "n": 4, "root": "",
                 "path": "maemms/nla/scores/{set}/sae_self"},
                {"name": "corpus", "kind": "examples_docmax", "n": CORPUS_TOP_N},
            ],
        }
        rows_ = [0, 1, 2, 3]
        # corpus_peak table: 100 -> 20, 101 -> 5, 102 -> 4, 103 -> 8
        cpeaks = {100: 20.0, 101: 5.0, 102: 4.0, 103: 8.0}

        # primary, n = 16. Row 0's per-rollout peaks are 1..16, which makes every bo-k exact:
        #   bo1  = mean(1..16)                      = 8.5
        #   bo4  = mean(max 1-4, 5-8, 9-12, 13-16)  = mean(4, 8, 12, 16) = 10
        #   bo16 = max(1..16)                       = 16
        # Rows 1-3 are flat so the aggregates are hand-checkable too.
        peaks = [[float(k + 1) for k in range(16)]] + [[3.0] * 16, [1.0] * 16, [5.0] * 16]
        pact = _block_from_peaks(peaks)
        _write_sae_self(data / "smoke/maemms/p/scores/S/sae_self", rows_, 16, 3, gate, pact,
                        peaks=_stored_of(rows_, pact, gate, cpeaks))
        _write_per_target(data / "smoke/maemms/p/scores/S", rows_)
        # rl16 exists under the MIRROR root only -> the [smoke, ""] search must still find it
        ract = _block_from_peaks([[2.0] * 16, [9.0] * 16])
        _write_sae_self(data / "maemms/r/scores/S/sae_self", [0, 1], 16, 3, gate, ract)
        # nla, n = 4, on rows 0 and 1 only, at a different width (the NLA arm scores wider)
        _write_sae_self(data / "maemms/nla/scores/S/sae_self", [0, 1], 4, 5, gate,
                        _block_from_peaks([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]], width=5))
        # a decoy at the SMOKE root for the pinned source: it must be ignored, gate and all
        _write_sae_self(data / "smoke/maemms/nla/scores/S/sae_self", [0], 4, 5, 77.0,
                        np.zeros((1, 4, 5)))

        hd = data / "smoke/base/b/heldout/S"
        hd.mkdir(parents=True, exist_ok=True)
        with open(hd / "ids.jsonl", "w") as fh:
            for r in rows_:
                fh.write(json.dumps({
                    "row": r, "family": "sae", "sae_key": "t/sae", "id": 100 + r,
                    "stratum": r, "heldout_kind": "feature_id",
                    "stratum_stat": "log10_gated_fires_16M",
                    # the prior work's own denominator: HALF our 16M peak, so every alt ratio is
                    # exactly twice the 16M one and a mix-up cannot pass unnoticed
                    "corpus_peak_1b": cpeaks[100 + r] / 2,
                }) + "\n")

        tab = np.zeros(200, dtype=np.float16)
        for f, v in cpeaks.items():
            tab[f] = v
        (data / "base/b/sae/sae").mkdir(parents=True, exist_ok=True)
        tab.tofile(data / "base/b/sae/sae/max_act.f16")
        # 18 stored windows for feature 100 so the top-N cut is exercised; none for 103
        ex = data / "base/b/sae/sae/examples_docmax/S"
        _write_examples(ex, 100, [9.0, 8.0] + [0.5] * 16)
        _write_examples(ex, 101, [4.0] * 3)
        _write_examples(ex, 102, [1.0] * 3)

        block = collect(spec, data, None)
        block["aggregate"] = aggregate(block)
        by_row = {f["row"]: f for f in block["features"]}
        assert sorted(by_row) == rows_, sorted(by_row)
        assert block["gate"] == gate, block["gate"]
        assert sorted(block["roots_read"]) == ["<mirror>", "smoke"], block["roots_read"]

        f0 = by_row[0]
        assert f0["feature"] == 100 and f0["corpus_peak"] == 20.0, f0
        assert f0["stratum"] == 0 and f0["sae_key"] == "t/sae", f0

        # primary: bo1 8.5, bo4 10, bo16 16, peak 16; ratios against a corpus peak of 20;
        # 14 of 16 rollouts are over a gate of 2
        p = f0["sources"]["primary"]
        assert (p["bo1"], p["bo4"], p["bo16"], p["peak_max"]) == (8.5, 10.0, 16.0, 16.0), p
        assert (p["bo1_ratio"], p["bo4_ratio"], p["bo16_ratio"], p["peak_max_ratio"]) == (
            0.425, 0.5, 0.8, 0.8), p
        assert (p["fired_frac"], p["fired_any"], p["n_items"]) == (0.875, True, 16), p

        # nla: n = 4 -> bo1 2.5, bo4 4.0, and bo16 is NOT computed (k > n is skipped, not clamped)
        nl = f0["sources"]["nla"]
        assert (nl["bo1"], nl["bo4"], nl["bo16"]) == (2.5, 4.0, None), nl
        assert nl["peak_max"] == 4.0 and nl["fired_frac"] == 0.5, nl
        assert nl["n_items"] == 4, "the pinned nla source read the smoke-root decoy"

        # corpus: 18 windows cut to 16 by max_act -> bo1 IS the top window (9.0), no bo-k,
        # 2 of the 16 shown windows fire
        cp = f0["sources"]["corpus"]
        assert (cp["bo1"], cp["bo4"], cp["bo16"]) == (9.0, None, None), cp
        assert (cp["peak_max"], cp["n_items"], cp["ranked"]) == (9.0, CORPUS_TOP_N, True), cp
        assert cp["bo1_ratio"] == 0.45 and cp["fired_frac"] == round(2 / CORPUS_TOP_N, 4), cp

        # absent is not zero, in both directions
        f2, f3 = by_row[2], by_row[3]
        assert f2["sources"]["rl16"]["absent"] is True, "rl16 has no row 2: absent, not 0"
        assert f3["sources"]["corpus"]["absent"] is True, "feature 103 has no examples file"
        assert f2["sources"]["primary"] == {
            "absent": False, "ranked": False, "n_items": 16, "peak_max": 1.0,
            "peak_max_ratio": 0.25, "fired_frac": 0.0, "fired_any": False,
            "bo1": 1.0, "bo1_ratio": 0.25, "bo4": 1.0, "bo4_ratio": 0.25,
            "bo16": 1.0, "bo16_ratio": 0.25,
            # the prior-work denominator is half ours, so each alt ratio is exactly double
            "alt_bo1_ratio": 0.5, "alt_bo4_ratio": 0.5, "alt_bo16_ratio": 0.5,
            "alt_peak_max_ratio": 0.5,
        }, f2["sources"]["primary"]

        # primary16: the SAME product read at its first 4 rollouts -> peaks 1,2,3,4
        p16 = f0["sources"]["primary16"]
        assert (p16["n_items"], p16["bo1"], p16["bo4"], p16["bo16"], p16["peak_max"]) == (
            4, 2.5, 4.0, None, 4.0), p16

        # the prior work's denominator is half ours, so every alt ratio is exactly twice
        assert f0["alt_peak"] == 10.0, f0["alt_peak"]
        assert p["alt_bo4_ratio"] == round(2 * p["bo4_ratio"], 4), p
        assert p["alt_peak_max_ratio"] == round(2 * p["peak_max_ratio"], 4), p
        assert block["alt_peak"]["how"].startswith("the `corpus_peak_1b` column"), block["alt_peak"]

        # per_target.jsonl: present per feature, and ONLY for primary
        assert f0["score"]["primary"] == {"mean_cos": 0.1, "max_cos": 0.5, "bo_4": 0.3,
                                          "bo_16": 0.4, "n_sae_gated": 10, "n": 16}, f0["score"]
        assert "rl16" not in f0["score"] and "nla" not in f0["score"], f0["score"]
        assert any("rl16: no per_target.jsonl" in m for m in block["missing"]), block["missing"]

        # the reader check: sae_self.json's own per_target against our recomputation
        rc = block["reader_check"]
        assert rc["primary"]["n_mismatches"] == 0 and rc["primary"]["rows"] == 4, rc
        assert rc["primary"]["exact"] == rc["primary"]["stats"], (
            "this fixture writes exactly representable values, so every pair must be bit-exact "
            "and the f16 tolerance must not be what is carrying the check")
        assert "primary16" not in rc, "a truncated read must not be compared with the stored full-n"
        assert "rl16" not in rc, "rl16's product ships no per_target values to check"

        # the tolerance itself. The constant is CHECKED, not asserted by hand, and the three
        # regimes are separated: inside an f16 ulp passes, a gate count may move by one rollout,
        # and anything wider is a mismatch whatever its size.
        assert _f16_bound_holds(), "F16_EPS is not an upper bound on the float32 -> f16 error"
        assert _agrees("f16", 8.5, 8.5 + 8.5 * F16_EPS, 16)[0], "one f16 ulp must be tolerated"
        assert not _agrees("f16", 8.5, 8.5 * 1.01, 16)[0], "1% is not an f16 ulp"
        assert _agrees("gate", 0.875, 0.875 - 1 / 16, 16)[0], "one rollout at the gate is noise"
        assert not _agrees("gate", 0.875, 0.875 - 3 / 16, 16)[0], "three rollouts is not"
        assert _agrees("exact", 20.0, 20.0, 1)[0] and not _agrees("exact", 20.0, 20.0001, 1)[0]

        # aggregates: primary over 4 features, peak ratios 0.8, 0.6, 0.25, 0.625
        agg = block["aggregate"]
        assert agg["primary"]["all"]["n_features"] == 4, agg["primary"]["all"]
        pm = sorted([16 / 20, 3 / 5, 1 / 4, 5 / 8])
        assert agg["primary"]["all"]["peak_max_ratio"]["median"] == round(
            (pm[1] + pm[2]) / 2, 4), agg["primary"]["all"]["peak_max_ratio"]
        # 3 of the 4 features have a peak over the gate of 2 (row 2 peaks at 1.0)
        assert agg["primary"]["all"]["frac_features_firing"] == 0.75, agg["primary"]["all"]
        assert agg["primary"]["all"]["mean_item_fired_frac"] == round(
            (0.875 + 1.0 + 0.0 + 1.0) / 4, 4), agg["primary"]["all"]
        # bo16 over nla is reported as n = 0, not as a number invented from bo4
        assert agg["nla"]["all"]["bo16_ratio"] == {"n": 0, "median": None, "mean": None}, agg["nla"]
        # per stratum: one feature each, and stratum 0's primary median IS feature 100's ratio
        byq = agg["primary"]["by_stratum"]
        assert sorted(byq) == ["0", "1", "2", "3"], sorted(byq)
        assert byq["0"]["peak_max_ratio"]["median"] == 0.8, byq["0"]
        assert byq["3"]["frac_features_firing"] == 1.0, byq["3"]
        assert "3" not in agg["corpus"]["by_stratum"], "feature 103 has no corpus source"

        # ---- the comparison section ------------------------------------------------------------
        block["comparison"] = compare(block)
        cmp = block["comparison"]
        # `t/sae` has no card entry and no act_smoke reference: both must say absent, not zero
        assert cmp["card"]["absent"].startswith("no card number"), cmp["card"]
        assert cmp["act_smoke"] == [], cmp["act_smoke"]
        assert cmp["reader"]["primary"]["n_mismatches"] == 0, cmp["reader"]

        # the same block read as a SAE the card does know: the flag must follow THE UPSTREAM denominator
        carded = {**block, "sae": "qwen36-27b/sae2m"}
        c = compare(carded)["card"]
        assert c["stat"] == "fired" and c["flag_basis"].startswith("a firing rate"), c
        assert c["ours_vs_16m"] == agg["rl16"]["all"]["mean_item_fired_frac"], c
        carded131 = {**block, "sae": "qwen36-27b/l42-1b"}
        c131 = compare(carded131)["card"]
        assert c131["flag_basis"] == "the upstream denominator", c131
        assert c131["ours_vs_their_peak"] == round(2 * c131["ours_vs_16m"], 4), c131
        assert c131["verdict"] in ("consistent", "CHECK — possible pipeline defect"), c131
        # the band itself: a prior result sitting on top of ours passes, one far outside flags
        spread = [0.40, 0.42, 0.44, 0.46, 0.48]
        assert _flag(0.44, 0.44, spread)["verdict"] == "consistent", _flag(0.44, 0.44, spread)
        far = _flag(0.44, 9.9, spread)
        assert far["verdict"].startswith("CHECK") and far["z"] > FLAG_Z, far
        assert _flag(0.44, None, spread)["verdict"] == "—", "a missing prior must not be flagged"
        # and the act_smoke rows for that SAE arrive, with the non-comparable one uncomputed
        a131 = compare(carded131)["act_smoke"]
        assert len(a131) == 4, a131
        corpus_row = [r for r in a131 if "corpus" in r["arm"]][0]
        assert corpus_row["comparable"] is True and corpus_row["delta"] is not None, corpus_row
        a2m = compare({**block, "sae": "qwen36-27b/sae2m"})["act_smoke"]
        prefix4m = [r for r in a2m if "4M prefix" in r["arm"]][0]
        assert prefix4m["delta"] is None and "not comparable" in prefix4m["verdict"], prefix4m

        md = render([block])
        for needle in ("absent", "bo16 med", "t/sae", "gate: 2.0", "top 16",
                       "Per source and stratum", "primary q0", "| 100 | 0 | 0 | 20.000 |",
                       "Comparison against prior results", "Reader check", "mismatches"):
            assert needle in md, f"the rendered table is missing {needle!r}"
        # cosines stay out of the result tables; the reader check's own listing is the exception
        head, _, tail = md.partition("## Comparison against prior results")
        for forbidden in ("mean_cos", "max_cos", "bo_16"):
            assert forbidden not in head, f"a cosine column reached the result tables: {forbidden}"
        assert "n_sae_gated" in tail and "bo_4 (cos)" in tail, "the reader listing lost its columns"
        assert "mean_cos" in json.dumps(block), "the cosines did not reach the JSON"

        # a stored per_target that disagrees with its own array is a DEFECT, and is printed as one
        bad_stored = _stored_of(rows_, pact, gate, cpeaks)
        bad_stored[1]["max_peak_act"] = 999.0
        _write_sae_self(data / "smoke/maemms/p/scores/S/sae_self", rows_, 16, 3, gate, pact,
                        peaks=bad_stored)
        broken = collect(spec, data, None)
        broken["aggregate"] = aggregate(broken)
        broken["comparison"] = compare(broken)
        assert broken["reader_check"]["primary"]["n_mismatches"] == 1, broken["reader_check"]
        assert "POSSIBLE PIPELINE DEFECT" in render([broken]), "a mismatch was not reported"
        _write_sae_self(data / "smoke/maemms/p/scores/S/sae_self", rows_, 16, 3, gate, pact,
                        peaks=_stored_of(rows_, pact, gate, cpeaks))

        # --rows cuts the block down, and selecting nothing is a warning rather than an empty
        # table that reads as "no features have these sources"
        cut = collect(spec, data, {0, 3})
        assert [f["row"] for f in cut["features"]] == [0, 3], cut["features"]
        none_sel = collect(spec, data, {900})
        assert not none_sel["features"] and any("--rows selected none" in w
                                                for w in none_sel["warnings"]), none_sel

        # --peaks-1b overrides the ids.jsonl column, and the parquet/jsonl/json readers agree
        pj = data / "peaks.json"
        pj.write_text(json.dumps({str(f): v / 4 for f, v in cpeaks.items()}))
        over = collect(spec, data, None, peaks_1b=str(pj))
        assert over["features"][0]["alt_peak"] == 5.0, over["features"][0]
        assert "--peaks-1b" in over["alt_peak"]["how"], over["alt_peak"]
        pl = data / "peaks.jsonl"
        pl.write_text("".join(json.dumps({"feature": f, "corpus_peak_1b": v / 4}) + "\n"
                              for f, v in cpeaks.items()))
        assert load_peaks_1b(str(pl)) == load_peaks_1b(str(pj)), "jsonl and json disagree"

        # the 131k alt-peak reader: repo_examples.jsonl's per-feature MAX is preferred over
        # per_feature.jsonl's MEAN, and the label says which was used
        rspec = {**spec, "alt_peak": {"label": "repo_max_act", "what": "synthetic repo windows",
                                      "kind": "repo_examples", "dir": "base/b/sae/sae/repo/{set}"}}
        rd = data / "base/b/sae/sae/repo/S"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "per_feature.jsonl").write_text("".join(
            json.dumps({"feature": f, "repo_mean_peak": v / 8}) + "\n" for f, v in cpeaks.items()))
        pk, how = load_alt_peaks(data, rspec)
        assert pk[100] == 2.5 and "MEAN" in how and "repo_examples.jsonl" in how, (pk[100], how)
        (rd / "repo_examples.jsonl").write_text("".join(
            json.dumps({"feature": f, "rank": i, "repo_peak": v / (4 + i)}) + "\n"
            for f, v in cpeaks.items() for i in range(3)))
        pk, how = load_alt_peaks(data, rspec)
        assert pk[100] == 5.0 and how.startswith("max of `repo_peak`"), (pk[100], how)

        # ---- the denominator's SCALE, which decides what a card disagreement means -------------
        # `t/sae` takes its prior peak from ids.jsonl -- the upstream column, not a proxy -- so there
        # is nothing to scale and the wording must stay as it was.
        assert block["alt_peak"]["scale"]["applicable"] is False, block["alt_peak"]["scale"]
        assert c131["denominator_label"] == "ours (÷ the upstream peak)", c131
        assert "denominator scale" not in c131["verdict"], c131

        def _repo_scale(mult):
            """Rewrite per_feature.jsonl so our peak sits `mult` times the repo's stored one.

            Feature 199 is DEAD on the repo's own windows (`repo_mean_peak` 0): it must be
            dropped from the scale rather than divided by, and it must not count towards `n`.
            """
            (rd / "per_feature.jsonl").write_text("".join(
                json.dumps({"feature": f, "repo_mean_peak": v / 8,
                            "mean_peak_act": (v / 8) * mult}) + "\n"
                for f, v in cpeaks.items())
                + json.dumps({"feature": 199, "repo_mean_peak": 0.0,
                              "mean_peak_act": 3.0}) + "\n")
            rb = collect(rspec, data, None)
            rb["aggregate"] = aggregate(rb)
            return rb, compare({**rb, "sae": "qwen36-27b/l42-1b"})["card"]

        on_scale, card_ok = _repo_scale(1.0)
        sc = on_scale["alt_peak"]["scale"]
        assert sc["applicable"] and sc["median"] == 1.0 and sc["n"] == 4, sc
        assert sc["q1"] == 1.0 and sc["q3"] == 1.0, sc
        assert card_ok["denominator_label"] == "ours (÷ the upstream peak)", card_ok
        assert "denominator scale" not in card_ok["verdict"], card_ok
        # the peaks still come from repo_examples.jsonl: the scale file does not supply them
        assert card_ok["alt_peak_how"].startswith("max of `repo_peak`"), card_ok

        off_scale, card_off = _repo_scale(1.85)
        sc = off_scale["alt_peak"]["scale"]
        assert sc["median"] == 1.85 and not SCALE_OK[0] <= sc["median"] <= SCALE_OK[1], sc
        assert card_off["denominator_label"] == "÷ repo peak (scale 1.850, unverified)", card_off
        # this fixture's prior value happens to land inside the band, so the AGREEMENT is the one
        # that gets qualified -- an off-scale denominator makes a match coincidental too
        assert card_off["verdict"] == "consistent — on an unverified denominator scale", card_off
        assert "possible pipeline defect" not in card_off["verdict"], card_off
        # the number and its z are NOT suppressed -- only what the verdict claims changes
        assert card_off["ours_vs_their_peak"] == card_ok["ours_vs_their_peak"], card_off
        assert card_off["z"] == card_ok["z"], (card_off["z"], card_ok["z"])

        # and the branch that matters: a prior value far outside the band must NOT come back as
        # "possible pipeline defect" while the denominator is unverified
        far = dict(CARD["qwen36-27b/l42-1b"], value=20.0)
        CARD["t/far"] = far
        try:
            hit = compare({**off_scale, "sae": "t/far"})["card"]
            assert hit["z"] > FLAG_Z, hit
            assert hit["verdict"] == "CHECK — denominator scale differs, not a pipeline verdict", hit
            on = compare({**on_scale, "sae": "t/far"})["card"]
            assert on["verdict"] == "CHECK — possible pipeline defect", on
        finally:
            del CARD["t/far"]
        md_off = render([{**off_scale, "comparison": compare(
            {**off_scale, "sae": "qwen36-27b/l42-1b"})}])
        for needle in ("not in our data", "norm_factor", "1.850", "IQR",
                       "÷ repo peak (scale 1.850, unverified)"):
            assert needle in md_off, f"the scale sentence is missing {needle!r}"

        # nothing measured it: the same downgrade, because an unvalidated denominator cannot
        # support a defect claim either
        (rd / "per_feature.jsonl").unlink()
        blind = collect(rspec, data, None)
        blind["aggregate"] = aggregate(blind)
        sc = blind["alt_peak"]["scale"]
        assert sc["applicable"] and sc["median"] is None and "UNMEASURED" in sc["how"], sc
        card_blind = compare({**blind, "sae": "qwen36-27b/l42-1b"})["card"]
        assert card_blind["denominator_label"] == "÷ repo peak (unmeasured, unverified)", card_blind
        assert card_blind["verdict"].endswith("on an unverified denominator scale"), card_blind
        CARD["t/far"] = far
        try:
            hit_blind = compare({**blind, "sae": "t/far"})["card"]
            assert hit_blind["verdict"] == (
                "CHECK — scale unmeasured, not a pipeline verdict"), hit_blind
        finally:
            del CARD["t/far"]
        _repo_scale(1.0)  # leave the fixture on-scale for anything after this

        # a gate disagreement between two products of the "same" SAE must be refused
        _write_sae_self(data / "maemms/r/scores/S/sae_self", [0, 1], 16, 3, 99.0,
                        np.zeros((2, 16, 3)))
        try:
            collect(spec, data, None)
        except AssertionError as e:
            assert "disagree on the SAE gate" in str(e), f"wrong assert fired: {e}"
        else:
            raise AssertionError("collect accepted two different gates for one SAE")
    print(
        "[sae_smoke64] selftest OK: root search and a pinned source, bo1/bo4/bo16 by hand, "
        "k > n skipped not clamped, n_first truncation, the ranked corpus source, absent vs "
        "zero, cosines in the JSON and not the result tables, per-stratum aggregates, --rows, "
        "the gate guard, both alt-peak readers, the prior denominator's measured SCALE and the "
        "verdict downgrade it forces, --peaks-1b, the card flag basis, the non-comparable "
        "act_smoke row and a reader-check mismatch reported as a defect"
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        run_selftest()
    else:
        app()
