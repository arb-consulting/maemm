#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9"]
# ///
"""CPU unit smoke for `results/` -- no volume, no network, no GPU.

    uv run evals/faithfulness/results/selftest.py

A synthetic mirror is written to a temp directory in the volume's OWN path layout, run through
`faithfulness.analyse` offline, and every number checked against one worked out by hand in the
fixture below -- not against a second call of the code under test. The per-target aggregates in
the fixture's `per_target.jsonl` are LITERALS, so the reader check (which compares this module's
recomputation from `cos.f16` against them) is a real comparison of two independent numbers rather
than a tautology; `check_reader_check_catches_a_defect` then corrupts one of those literals and
requires the check to go red, because a check that has never been red is unevaluated.

The synthetic mirror carries, deliberately:

  * two run tags of ONE checkpoint (`arm-a` centred, `arm-b` not), which is the shape the old
    primary's `mu-none` / `mu-stats` reconciliation has on the volume;
  * a second checkpoint with per_target only -- no arrays, no `sae_self` -- so the absent-product
    paths are exercised on every run rather than only when the volume happens to be incomplete;
  * a checkpoint declared in config with NO products, which must be reported missing;
  * a checkpoint declared `compute: false`, which must NOT be reported missing;
  * TWO SAE dictionaries distinguished only by the rows' own `sae_key`, which is the "a new SAE
    is a config entry and a row field, not an edit here" requirement;
  * a document shared by two `realact` rows, so the clustered bootstrap has something to cluster.

EVAL 2 (`results/autointerp.py`) has its own fixture in the second half of this file: TWO run
directories in the volume's `runs/<dir>/summary/scores.jsonl` layout, six (run, arm) pairs over two
scorers, every `bal_acc` dyadic so the means and the paired differences are exact literals. It
carries, deliberately:

  * one arm present on only THREE of the four features (`NLA`) and one whose fourth feature came
    back with a null `bal_acc` (`old/M` on fuzzing, the all-batches-unparsed case), so the
    intersection in the paired contrast is a real one and `REDUCED` pairing is exercised on every
    run rather than only when a product happens to be short;
  * `DOCMAX` in BOTH run directories under different numbers, which is what makes the arm name
    alone insufficient to identify an arm;
  * `tpr = tnr = bal_acc` on every row, so the reader check's `0.5 * (TPR + TNR)` identity is exact
    and `check_autointerp_catches_a_defect` can break it by one number.

EVAL 2's CUT VIEWS have a THIRD fixture, at the end of this file: 32 features over two ORTHOGONAL
cuts -- the draw's rarity stratum and a post-hoc quartile of `corpus_peak` -- laid out so each
stratum holds exactly two features of each magnitude quartile. An arm that moves with one cut
therefore has zero spread on the other, so a driver that computed one cut and printed it under both
headings cannot pass. It carries, deliberately:

  * an arm with a two-feature cell against a `MIN_CELL` of four, whose two features are the
    EXTREME ones, so averaging the short cell in (spread 0.5), dropping it silently (spread 0 and
    no trace) and the correct answer (spread 0 WITH the cell named) are three different numbers;
  * the same arm's cells on the other cut at six of eight, which is under-filled and not short, so
    both counters of the `cells` check are exercised on every run;
  * a permutation whose exact p is hand-computable: eight high values among 32 make the observed
    spread the largest the multiset admits, reached by 4 / C(32, 8) of relabellings, so at 2000
    resamples the reported p is the add-one FLOOR and its orthogonal twin is exactly 1.0;
  * a set README in `precompute`'s own layout, with its stratification note WRAPPED over two lines
    and a `- command:` line above it that says `--stratified` and states no cut, so the caption's
    quoting is tested against both of the ways it could quote the wrong thing.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autointerp.build as AB  # noqa: E402
import precompute.common as PC  # noqa: E402
import results.autointerp as A  # noqa: E402
import results.autointerp_cases as AC  # noqa: E402
import results.autointerp_encdec as AE  # noqa: E402
import results.common as R  # noqa: E402
import results.corpus_search as CS  # noqa: E402
import results.faithfulness as F  # noqa: E402

BASE = "B"
SET = "S"
# A second set that the SAME checkpoint was also scored on, for the `kind: cross_set` gate. Its
# product deliberately carries only SOME of the rows, so a gate that failed to intersect would
# compare different row sets and get a different number.
SET2 = "S2"
SET2_ROWS = [0, 1, 4, 5]
N_ROLLOUTS = 4
WIDTH = 3

# --- the fixture, worked out by hand -----------------------------------------------------------
#
# Eight rows. Every per-rollout value is dyadic, so its float16 payload is EXACT and the reader
# check has no rounding budget to hide in.
IDS = [
    {"row": 0, "family": "realact", "doc": 7, "id": "doc7:p1:L4"},
    {"row": 1, "family": "realact", "doc": 7, "id": "doc7:p9:L4"},
    {"row": 2, "family": "realact", "doc": 9, "id": "doc9:p3:L4"},
    {"row": 3, "family": "random", "id": "rand0"},
    {"row": 4, "family": "sae", "sae_key": "B/sae-one", "stratum": 0, "id": "f100"},
    {"row": 5, "family": "sae", "sae_key": "B/sae-one", "stratum": 1, "id": "f200"},
    {"row": 6, "family": "sae", "sae_key": "B/sae-two", "stratum": 0, "id": "f300"},
    {"row": 7, "family": "sae", "sae_key": "B/sae-two", "stratum": 1, "id": "f400"},
]

# Per-rollout best cosines per row, per arm.
BEST_A = {0: [0.5, 0.25, 0.75, 1.0], 1: [0.5] * 4, 2: [0.25] * 4, 3: [0.125] * 4,
          4: [0.0625] * 4, 5: [0.0625] * 4, 6: [0.0625] * 4, 7: [0.0625] * 4}
BEST_B = {0: [0.25] * 4, 1: [0.5] * 4, 2: [0.25] * 4, 3: [0.125] * 4,
          4: [0.0625] * 4, 5: [0.0625] * 4, 6: [0.0625] * 4, 7: [0.0625] * 4}
# The centred cosine exists only for the centrable family and only on the arm that centred.
BEST_A_C = {0: [0.375, 0.125, 0.625, 0.875], 1: [0.375] * 4, 2: [0.125] * 4}

# `per_target.jsonl` as the product would have written it -- LITERALS, not recomputed here.
PT_A = {
    0: {"mean_cos": 0.625, "max_cos": 1.0, "bo_1": 0.625, "bo_2": 0.75, "bo_4": 1.0,
        "mean_cos_centred": 0.5, "max_cos_centred": 0.875, "n_centred": 4,
        "bo_c_1": 0.5, "bo_c_2": 0.625, "bo_c_4": 0.875},
    1: {"mean_cos": 0.5, "max_cos": 0.5, "bo_1": 0.5, "bo_2": 0.5, "bo_4": 0.5,
        "mean_cos_centred": 0.375, "max_cos_centred": 0.375, "n_centred": 4,
        "bo_c_1": 0.375, "bo_c_2": 0.375, "bo_c_4": 0.375},
    2: {"mean_cos": 0.25, "max_cos": 0.25, "bo_1": 0.25, "bo_2": 0.25, "bo_4": 0.25,
        "mean_cos_centred": 0.125, "max_cos_centred": 0.125, "n_centred": 4,
        "bo_c_1": 0.125, "bo_c_2": 0.125, "bo_c_4": 0.125},
    3: {"mean_cos": 0.125, "max_cos": 0.125, "bo_1": 0.125, "bo_2": 0.125, "bo_4": 0.125},
    **{r: {"mean_cos": 0.0625, "max_cos": 0.0625, "bo_1": 0.0625, "bo_2": 0.0625, "bo_4": 0.0625}
       for r in (4, 5, 6, 7)},
}
PT_B = {
    0: {"mean_cos": 0.25, "max_cos": 0.25, "bo_1": 0.25, "bo_2": 0.25, "bo_4": 0.25},
    1: {"mean_cos": 0.5, "max_cos": 0.5, "bo_1": 0.5, "bo_2": 0.5, "bo_4": 0.5},
    2: {"mean_cos": 0.25, "max_cos": 0.25, "bo_1": 0.25, "bo_2": 0.25, "bo_4": 0.25},
    3: {"mean_cos": 0.125, "max_cos": 0.125, "bo_1": 0.125, "bo_2": 0.125, "bo_4": 0.125},
    **{r: {"mean_cos": 0.0625, "max_cos": 0.0625, "bo_1": 0.0625, "bo_2": 0.0625, "bo_4": 0.0625}
       for r in (4, 5, 6, 7)},
}

# `sae_self`: per-rollout peak activation, the feature's 16M corpus peak, and the learned gate.
SAE_GATE = 1.0
SAE_PEAKS = {4: [1.0, 2.0, 3.0, 4.0], 5: [0.5] * 4, 6: [2.0] * 4, 7: [0.25] * 4}
SAE_CORPUS_PEAK = {4: 4.0, 5: 2.0, 6: 2.0, 7: 1.0}
SAE_FEATURE = {4: 100, 5: 200, 6: 300, 7: 400}
# The product's own per_target block, again as literals.
SAE_STORED = {4: {"mean_peak_act": 2.5, "max_peak_act": 4.0, "fire_fraction": 0.75},
              5: {"mean_peak_act": 0.5, "max_peak_act": 0.5, "fire_fraction": 0.0},
              6: {"mean_peak_act": 2.0, "max_peak_act": 2.0, "fire_fraction": 1.0},
              7: {"mean_peak_act": 0.25, "max_peak_act": 0.25, "fire_fraction": 0.0}}

CFG = {
    "bases": {BASE: {"d": 4, "read_layer": 1}},
    "family_kinds": {
        "realact": {"centrable": True, "kind": "activation"},
        "random": {"centrable": False, "kind": "synthetic"},
        "sae": {"centrable": False, "kind": "dictionary"},
    },
    "heldout": {
        SET: {"base": BASE, "sae_key": "B/sae-one",
              "families": {"realact": {"n": 3}, "random": {"n": 1}, "sae": {"n": 4}}},
        SET2: {"base": BASE, "sae_key": "B/sae-one", "families": {"realact": {"n": 3}}},
    },
    "maemms": {
        f"{BASE}/ckpt-one": {"type": "full", "primary": True, "mu": "/mu.npy"},
        f"{BASE}/ckpt-two": {"type": "full"},
        f"{BASE}/ckpt-nothing": {"type": "full"},
        f"{BASE}/ckpt-declared-only": {"type": "full", "compute": False},
    },
}

SANITY = """
checks:
  - name: arm-a realact mean cos
    source: "ckpt-one:arm-a"
    family: realact
    metric: cos_raw.bo1
    expect: 0.458333
    tol: 0.0005
  - name: a gate that must flag
    source: "ckpt-one:arm-a"
    family: realact
    metric: cos_raw.bo1
    expect: 0.9
    tol: 0.01
  - name: a gate on a family this set does not have
    source: "ckpt-one:arm-a"
    family: bsf
    metric: cos_centred.bo4
    expect: 0.716
    tol: 0.05
  - name: a pair that is not the same statistic
    source: "ckpt-one:arm-a"
    family: realact
    metric: cos_raw.bo1
    expect: 0.3125
    compare: false
"""


# --- writing the synthetic mirror ---------------------------------------------------------------


def _block(bests: list[float], width: int = WIDTH) -> np.ndarray:
    """[n, W] per-token values whose per-rollout max is `bests`, NaN outside the kept tokens."""
    out = np.full((len(bests), width), np.nan, dtype=np.float32)
    for i, v in enumerate(bests):
        out[i, 0] = v / 2
        out[i, 1] = v
    return out


def write_mirror(root: Path) -> None:
    hd = root / f"base/{BASE}/heldout/{SET}"
    hd.mkdir(parents=True, exist_ok=True)
    with open(hd / "ids.jsonl", "w") as fh:
        for r in IDS:
            fh.write(json.dumps(r) + "\n")
    (hd / "storage.json").write_text(json.dumps(
        {"storage": "raw", "sae_key": "B/sae-one"}))

    _write_scores(root, f"{BASE}/ckpt-one", f"{SET}__arm-a", PT_A, BEST_A, BEST_A_C,
                  mu="/mu.npy", sae_rows=[4, 5, 6, 7])
    _write_scores(root, f"{BASE}/ckpt-one", f"{SET}__arm-b", PT_B, BEST_B, None,
                  mu=None, sae_rows=[4, 5, 6, 7])
    # No arrays and no sae_self: the absent-product paths, exercised on every run.
    _write_scores(root, f"{BASE}/ckpt-two", SET, PT_A, None, None, mu=None, sae_rows=None)

    # The second set: the same rows, a different arm's numbers, and only four of the eight rows.
    hd2 = root / f"base/{BASE}/heldout/{SET2}"
    hd2.mkdir(parents=True, exist_ok=True)
    with open(hd2 / "ids.jsonl", "w") as fh:
        for r in IDS:
            fh.write(json.dumps(r) + "\n")
    (hd2 / "storage.json").write_text(json.dumps({"storage": "raw", "sae_key": "B/sae-one"}))
    _write_scores(root, f"{BASE}/ckpt-one", f"{SET2}__arm-a", PT_B, None, None,
                  mu=None, sae_rows=None, rows=SET2_ROWS)


def _write_scores(root: Path, maemm: str, dirname: str, pt: dict, best: dict | None,
                  best_c: dict | None, mu: str | None, sae_rows: list[int] | None,
                  rows: list[int] | None = None) -> None:
    d = root / f"maemms/{maemm}/scores/{dirname}"
    d.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else [r["row"] for r in IDS]
    with open(d / "per_target.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"row": r, "family": IDS[r]["family"], "n": N_ROLLOUTS,
                                 "bo": N_ROLLOUTS, "seed": 1234, "checkpoint_sha": "deadbeef",
                                 "n_sae_gated": 7, **pt[r]}) + "\n")
    (d / "rows.json").write_text(json.dumps(
        {"rows": rows, "n": N_ROLLOUTS, "families": [IDS[r]["family"] for r in rows],
         "score_max_length": WIDTH - 1, "mu": mu}))

    index: dict = {"per_target.jsonl": {"kind": "jsonl", "rows": len(rows)},
                   "rows.json": {"kind": "json"}}
    if best is not None:
        arr = np.stack([_block(best[r]) for r in rows]).astype(np.float16)
        arr.tofile(d / "cos.f16")
        index["cos.f16"] = {"kind": "array", "dtype": "float16",
                            "shape": list(arr.shape), "bytes": arr.nbytes}
    if best_c is not None:
        blocks = []
        for r in rows:
            blocks.append(_block(best_c[r]) if r in best_c
                          else np.full((N_ROLLOUTS, WIDTH), np.nan, dtype=np.float32))
        arr = np.stack(blocks).astype(np.float16)
        arr.tofile(d / "cos_centred.f16")
        index["cos_centred.f16"] = {"kind": "array", "dtype": "float16",
                                    "shape": list(arr.shape), "bytes": arr.nbytes}
    (d / "index.json").write_text(json.dumps(index))

    if sae_rows is None:
        return
    sd = d / "sae_self"
    sd.mkdir(exist_ok=True)
    act = np.stack([_block(SAE_PEAKS[r]) for r in sae_rows]).astype(np.float16)
    act.tofile(sd / "sae_self.f16")
    (sd / "sae_self.json").write_text(json.dumps({
        "rows": sae_rows, "features": [SAE_FEATURE[r] for r in sae_rows], "n": N_ROLLOUTS,
        "width": WIDTH, "gate": SAE_GATE,
        "checks": {"argmax_ok": True, "csr_checked": True},
        "per_target": [{"row": r, "feature": SAE_FEATURE[r], "n": N_ROLLOUTS,
                        "corpus_peak": SAE_CORPUS_PEAK[r], **SAE_STORED[r]} for r in sae_rows],
    }))


def _analyse(root: Path, **kw):
    vol = R.Vol("", root, offline=True, quiet=True)
    opts = {"sources": "", "boot": 400, "seed": 1, "check_arrays": True,
            "check_arrays_max_mb": 8.0}
    opts.update(kw)
    return vol, F.analyse(vol, CFG, SET, opts["sources"], opts["boot"], opts["seed"],
                          opts["check_arrays"], opts["check_arrays_max_mb"])


def _stat(res, family, source, metric, stratum=None):
    reg = F.stat_registry(res)
    key = (family, source, stratum, metric)
    assert key in reg, f"{key} not among {sorted(reg)[:12]}... ({len(reg)} keys)"
    return reg[key]


def _close(a, b, tol=1e-6, what=""):
    assert abs(float(a) - float(b)) <= tol, f"{what}: {a!r} vs {b!r} (tol {tol})"


# --- checks ------------------------------------------------------------------------------------


def check_parse_scores_dir():
    """The inverse of `precompute/common.rollout_stem` on all four shapes, and two non-matches."""
    assert F.R.parse_scores_dir("S", "S") == ("hf", "")
    assert F.R.parse_scores_dir("S__vllm", "S") == ("vllm", "")
    assert F.R.parse_scores_dir("S__mu-none", "S") == ("hf", "mu-none")
    assert F.R.parse_scores_dir("S__vllm__mu-none", "S") == ("vllm", "mu-none")
    # THE OTHER ORDER, which is the one eval 1's old-primary arms are actually written in:
    # `score --score-name <set>__<tag> --engine vllm` gave `<set>__<tag>__<engine>`, because
    # `scores_dir` took no tag and the tag could only enter through the set name. Reading that as
    # engine `hf` with the tag `mu-none__vllm` put "hf" in the paper's CSV for six vLLM arms.
    # `scores_dir` takes a `tag` in `rollout_stem`'s position since 2026-09-21 and WRITES the
    # canonical order, but this reader keeps both: the products on the volume did not move.
    assert F.R.parse_scores_dir("S__mu-none__vllm", "S") == ("vllm", "mu-none")
    assert F.R.parse_scores_dir("S__mu-stats__vllm", "S") == ("vllm", "mu-stats")
    # A directory that merely starts with the set name is ANOTHER set, not an untagged run of
    # this one -- `2026-09-16_v1` and `2026-09-16_v1x` both exist in that namespace.
    assert F.R.parse_scores_dir("Sx", "S") is None
    assert F.R.parse_scores_dir("T__vllm", "S") is None


def check_estimators():
    """`bo_ladder`, `peaks_of` and `best_per_rollout` against numbers written out here.

    bo-k is the UNBIASED order statistic, sum_i x_(i) C(i-1, k-1) / C(n, k) over the ascending
    order statistics -- ONE estimator across the pipeline since 2026-09-23 (M0a). The k = 2 cell
    is worked out by hand below BECAUSE it is where the estimator this replaced disagrees: the
    disjoint-group mean of the same four draws is (max(.5,.25) + max(.75,1)) / 2 = 0.75, and the
    unbiased one is 0.8333.., so a test that only checked bo1 and bo4 could not tell them apart.
    """
    bo = R.bo_ladder([0.5, 0.25, 0.75, 1.0], (1, 2, 4, 8, 64))
    _close(bo[1], 0.625, what="bo1 is the plain mean")
    # sorted [.25, .5, .75, 1.0]; weights C(i-1,1)/C(4,2) = [0, 1, 2, 3]/6
    _close(bo[2], (0.5 * 1 + 0.75 * 2 + 1.0 * 3) / 6, what="bo2 is the unbiased order statistic")
    assert abs(bo[2] - 0.75) > 1e-3, (
        "bo2 came out as the DISJOINT-GROUP mean 0.75; the two estimators must not be confused"
    )
    _close(bo[4], 1.0, what="bo4 is the max of all four")
    # A k above n is SKIPPED, not clamped and not an error: `sae_cells` asks for all of
    # score's BO_KS on every product, and a four-rollout product must answer with three of them.
    assert sorted(bo) == [1, 2, 4], f"a k above n must be skipped, got {sorted(bo)}"
    # NaN outside the kept tokens; an empty rollout peaks at 0.0 and is not a missing measurement.
    act = np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    assert list(R.peaks_of(act)) == [2.0, 0.0]
    # `common.agg` scores a rollout with nothing kept at -1.0; the centred path drops it instead.
    assert list(R.best_per_rollout(act)) == [2.0, -1.0]
    got = R.best_per_rollout(act, empty=float("nan"))
    assert got[0] == 2.0 and math.isnan(got[1])


def check_one_bo_estimator():
    """`precompute/common.bo_ladder` and `results.common.bo_unbiased` are ONE estimator.

    The pipeline has two layers that cannot import each other -- `precompute/` is what the Modal
    container ships, `results/` is a standalone local script layer -- so the estimator is written
    twice, exactly as `read_array` is. That duplicate is CHECKED here rather than trusted: random
    draws, every k, agreement to floating point. It also asserts they are not BOTH the old
    disjoint-group mean, which two copies of one mistake would pass silently.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "m0a_precompute_common", Path(__file__).resolve().parent.parent / "precompute" / "common.py"
    )
    PC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PC)

    rng = np.random.default_rng(20260923)
    for n in (1, 2, 3, 4, 7, 16, 64):
        vals = rng.normal(size=n).tolist()
        ks = tuple(k for k in (1, 2, 3, 4, 8, 16, 32, 64, 65))
        theirs = PC.bo_ladder(vals, ks)
        ours = R.bo_ladder(vals, ks)
        assert sorted(theirs) == sorted(ours) == sorted(k for k in ks if k <= n), (
            f"n={n}: the two layers skip different k -- {sorted(theirs)} vs {sorted(ours)}"
        )
        for k in theirs:
            assert abs(theirs[k] - ours[k]) < 1e-12, (
                f"n={n} k={k}: precompute {theirs[k]!r} vs results {ours[k]!r} -- the two layers "
                f"are computing different statistics under one name"
            )
    # and neither is the disjoint-group mean: a value where the two estimators differ
    v = [0.5, 0.25, 0.75, 1.0]
    naive = (max(v[0], v[1]) + max(v[2], v[3])) / 2
    assert abs(PC.bo_ladder(v, (2,))[2] - naive) > 1e-3, (
        "precompute.bo_ladder still returns the disjoint-group mean at k = 2"
    )
    assert abs(R.bo_ladder(v, (2,))[2] - naive) > 1e-3, (
        "results.bo_ladder still returns the disjoint-group mean at k = 2"
    )


def check_fired_is_the_same_estimator():
    """The fired indicator IS best-of-k of the 0/1 gate crossings, by the one estimator.

    For a 0/1 vector with m of n draws above the gate, the unbiased order-statistic best-of-k is
    1 - C(n - m, k) / C(n, k), which is exactly P(at least one of k draws fires). k = 1 is the
    plain rate `item_fired` always reported and k = n is `fired_any`, so the two names that
    existed before keep their meaning while the ladder in between becomes available -- which is
    what panel b's `sae.l131k.ex.fired.bo8.q<q>` keys print.
    """
    import math as _m

    rng = np.random.default_rng(7)
    for n, m in ((8, 0), (8, 1), (8, 3), (8, 8), (16, 5)):
        ind = np.concatenate([np.ones(m), np.zeros(n - m)])
        rng.shuffle(ind)
        lad = R.bo_ladder(ind, (1, 2, 4, 8, 16))
        for k, got in lad.items():
            want = 1.0 - (_m.comb(n - m, k) / _m.comb(n, k) if n - m >= k else 0.0)
            assert abs(got - want) < 1e-12, f"n={n} m={m} k={k}: {got} vs 1 - C(n-m,k)/C(n,k) {want}"
        assert abs(lad[1] - m / n) < 1e-12, "bo1 of the indicator must be the plain fired rate"
        assert abs(lad[n] - (1.0 if m else 0.0)) < 1e-12, "bo_n of the indicator must be fired_any"


def check_cluster_bootstrap():
    """Clustering must CHANGE the SE, reduce to the iid bootstrap on singletons, and refuse one
    cluster. A clustered SE that equalled the iid one would be the bug this exists to prevent."""
    # Two rows per document, perfectly correlated within it: the case the estimator exists for.
    vals = [1.0, 1.0, 4.0, 4.0]
    m2, se2, n2, c2 = R.cluster_bootstrap(vals, ["a", "a", "b", "b"], 4000, 1)
    m1, se1, _, c1 = R.cluster_bootstrap(vals, ["a", "b", "c", "d"], 4000, 1)
    _close(m2, 2.5, what="the mean does not depend on the clustering")
    assert (n2, c2, c1) == (4, 2, 4)
    assert se2 > se1 * 1.3, f"two correlated clusters must widen the SE: {se2} vs {se1}"
    # Singletons: the ordinary nonparametric bootstrap, whose SE is the population sd / sqrt(n).
    _close(se1, 1.5 / 2, tol=0.05, what="singleton clusters = the iid bootstrap")
    _, se0, _, c0 = R.cluster_bootstrap(vals, ["a"] * 4, 4000, 1)
    assert c0 == 1 and math.isnan(se0), "one cluster gives no spread -- NaN, never 0.0"


def check_round_trip():
    """The synthetic products through `analyse`, against the hand-computed aggregates."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        _vol, res = _analyse(root)
        labels = sorted(s.label for s in res["sources"])
        assert labels == ["ckpt-one:arm-a", "ckpt-one:arm-b", "ckpt-two"], labels
        # cos_raw bo1 over realact rows 0,1,2 = (0.625 + 0.5 + 0.25) / 3
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_raw.bo1"), 1.375 / 3, 1e-9)
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_raw.bo4"), 1.75 / 3, 1e-9)
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_centred.bo1"), 1.0 / 3, 1e-9)
        _close(_stat(res, "realact", "ckpt-one:arm-b", "cos_raw.bo1"), 1.0 / 3, 1e-9)
        # arm-b centred on nothing, so it has no centred cosine at all -- absent, not zero.
        reg = F.stat_registry(res)
        assert ("realact", "ckpt-one:arm-b", None, "cos_centred.bo1") not in reg
        # Two documents behind three realact rows.
        cell = [c for c in res["cos"] if c["family"] == "realact"
                and c["source"] == "ckpt-one:arm-a" and c["cosine"] == "cos_raw"][0]
        assert (cell["n_rows"], cell["n_clusters"]) == (3, 2), cell
        # The reported SE is the DOCUMENT-clustered one, not the row-level one. Rows 0 and 1 are
        # one document and row 2 is another, so the two estimators give different numbers; a table
        # that printed the row-level SE under a clustered heading is the defect this pins.
        vals = [0.625, 0.5, 0.25]
        want = R.cluster_bootstrap(vals, [7, 7, 9], 400, 1)[1]
        singl = R.cluster_bootstrap(vals, [0, 1, 2], 400, 1)[1]
        _close(cell["bo"][1]["se"], want, 1e-12, what="the SE clusters on `doc`")
        _close(cell["bo"][1]["se_iid"], R.se_iid(vals), 1e-12)
        for other in (singl, R.se_iid(vals)):
            assert abs(cell["bo"][1]["se"] - other) > 1e-6, (
                f"the clustered SE collapsed onto the row-level one: {cell['bo'][1]}")
        # SAE: row 4 peaks [1,2,3,4] / corpus peak 4 -> bo1 0.625; row 5 0.5/2 -> 0.25.
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "ratio.bo1.median"), 0.4375, 1e-9)
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "ratio.bo4.median"), (1.0 + 0.25) / 2, 1e-9)
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "fired.item"), 0.375, 1e-9)
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "fired.feature"), 0.5, 1e-9)
        # Per stratum: one feature each.
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "ratio.bo1.median", 0), 0.625, 1e-9)
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "fired.item", 0), 0.75, 1e-9)
        _close(_stat(res, "sae/sae-one", "ckpt-one:arm-a", "fired.item", 1), 0.0, 1e-9)
        # The reader check compared something and found nothing wrong.
        cos_chk = [c for c in res["checks"] if c["kind"] == "cos" and c["source"] == "ckpt-one:arm-a"][0]
        assert cos_chk.get("rows", 0) >= 8 and cos_chk["n_mismatches"] == 0, cos_chk
        sae_chk = [c for c in res["checks"] if c["kind"] == "sae_self"][0]
        assert sae_chk["n_mismatches"] == 0, sae_chk


def check_exclusions_leave_every_surface_together():
    """The set's own `exclusions.json` drops rows from the tables, the SEs, the registry and the
    figures AT ONCE, and `--no-exclusions` puts them back.

    The rows stay IN the set -- every arm is scored on all of them so the arms pair row for row --
    so dropping them is the reader's job, and the failure mode is a reader that drops them from
    one surface and not another: a mean over 486 rows printed beside a document count over 512,
    or a sanity gate resolving against the unexcluded number while the table shows the excluded
    one. Row membership is therefore decided in ONE place (the family map) and this check requires
    the mean, the row count, the cluster count and the registry to move together.

    The fixture excludes row 0, which is the only realact row with a non-constant cosine and
    shares a document with row 1 -- so the mean, the row count AND the document count all change,
    and a partial drop cannot produce a consistent triple by luck.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        (root / f"base/{BASE}/heldout/{SET}/exclusions.json").write_text(json.dumps({
            "block": "realact", "rows_total": 3, "excluded_rows": [0], "n_headline": 2,
            "criterion": "a synthetic gate", "n_gram": 7, "source": "selftest",
            "computed_over": "the fixture"}))
        vol = R.Vol("", root, offline=True, quiet=True)
        on = F.analyse(vol, CFG, SET, "", 400, 1, True, 8.0, 128.0, True)
        off = F.analyse(vol, CFG, SET, "", 400, 1, True, 8.0, 128.0, False)

        # rows 1 and 2 only: (0.5 + 0.25) / 2, against all three rows' 1.375 / 3.
        _close(_stat(on, "realact", "ckpt-one:arm-a", "cos_raw.bo1"), 0.375, 1e-9)
        _close(_stat(off, "realact", "ckpt-one:arm-a", "cos_raw.bo1"), 1.375 / 3, 1e-9)
        cell_on = [c for c in on["cos"] if c["family"] == "realact"
                   and c["source"] == "ckpt-one:arm-a" and c["cosine"] == "cos_raw"][0]
        cell_off = [c for c in off["cos"] if c["family"] == "realact"
                    and c["source"] == "ckpt-one:arm-a" and c["cosine"] == "cos_raw"][0]
        # ALL THREE move together: rows 3 -> 2 and documents 2 -> 2 (row 0 shared doc 7 with row 1,
        # so the document survives) -- and the per-k counts inside the cell move with them.
        assert (cell_off["n_rows"], cell_off["n_clusters"]) == (3, 2), cell_off
        assert (cell_on["n_rows"], cell_on["n_clusters"]) == (2, 2), cell_on
        assert cell_on["bo"][1]["n_rows"] == 2, cell_on["bo"][1]
        # The centred ladder is cut too -- it is a separate read of a separate array.
        _close(_stat(on, "realact", "ckpt-one:arm-a", "cos_centred.bo1"), (0.375 + 0.125) / 2, 1e-9)
        # And so is the support registry the sanity table prints as `n`.
        assert F.support_registry(on)[("realact", "ckpt-one:arm-a", None)] == 2
        assert F.support_registry(off)[("realact", "ckpt-one:arm-a", None)] == 3
        # A family with no exclusion is untouched.
        for r in (on, off):
            _close(_stat(r, "random", "ckpt-one:arm-a", "cos_raw.bo1"), 0.125, 1e-9)

        # The provenance reaches the document, in full, on both settings.
        line_on = F._exclusion_line(on)
        assert "1 rows dropped" in line_on and "n = 2 of 3" in line_on and "selftest" in line_on, line_on
        assert "NOT APPLIED" in F._exclusion_line(off), F._exclusion_line(off)
        # A set with no exclusions.json says so rather than claiming a clean zero.
        none = F.analyse(vol, CFG, SET2, "", 400, 1, False, 8.0, 128.0, True)
        assert "no exclusions.json" in F._exclusion_line(none), F._exclusion_line(none)


def check_centred_bok_is_recomputed_from_the_array():
    """The centred bo-k ladder comes from `cos_centred.f16`, by THE estimator.

    `score` writes `bo_c_<k>` only for a row whose EVERY rollout kept a centred token
    (`score.py:521`, `if len(vals_c) == n`), and on eval 1's own products no row qualifies -- so
    the centred bo8/bo64 columns of plan §2.3 exist nowhere on the volume and have to be made
    here. Four things, each its own failure:

      1. with no NaN, the recomputation equals `results.common.bo_ladder` on the same values
         EXACTLY -- it is the same estimator, not a similar one, and that is what lets it sit in a
         column beside the stored raw ladder;
      2. the NaN draws are DROPPED and the estimator applied to the m finite ones at the same k,
         which is the unbiased best-of-k of the draws that exist;
      3. a k above the number of FINITE draws is skipped, never clamped, so no cell claims a bo-k
         that row could not supply;
      4. over `--centred-bok-max-mb` the read is REFUSED with a reason, not answered from the
         stored ladder that is not there.
    """
    ids = [{"row": 0, "family": "realact", "doc": 1, "id": "a"},
           {"row": 1, "family": "realact", "doc": 2, "id": "b"}]
    # Row 0: all four finite. Row 1: draws 1 and 2 have no kept centred token.
    #   row 0 bests [0.25, 0.75, 0.5, 1.0], all finite, n = 4, sorted [.25, .5, .75, 1.0]:
    #        bo1 0.625;  bo2 (.5*1 + .75*2 + 1.0*3)/C(4,2) = 5/6;  bo4 1.0
    #   row 1 bests [0.5, nan, nan, 0.25]: TWO finite draws, so the estimator sees n = 2:
    #        bo1 (0.5 + 0.25)/2 = 0.375;  bo2 0.5 (the max of the two);  bo4 ABSENT -- this row
    #        has two draws and cannot answer a best-of-4
    centred = {0: [0.25, 0.75, 0.5, 1.0], 1: [0.5, math.nan, math.nan, 0.25]}
    raw = {0: [0.25] * 4, 1: [0.25] * 4}
    cfg = {**CFG, "heldout": {**CFG["heldout"],
                              "CB": {"base": BASE, "families": {"realact": {"n": 2}}}}}

    def write(root: Path) -> None:
        hd = root / f"base/{BASE}/heldout/CB"
        hd.mkdir(parents=True, exist_ok=True)
        with open(hd / "ids.jsonl", "w") as fh:
            for r in ids:
                fh.write(json.dumps(r) + "\n")
        (hd / "storage.json").write_text(json.dumps({"storage": "raw"}))
        d = root / f"maemms/{BASE}/ckpt-one/scores/CB__arm-a"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "per_target.jsonl", "w") as fh:
            for r in (0, 1):
                # mean/max_cos_centred and n_centred, and NO `bo_c_k` -- the production shape.
                fin = [v for v in centred[r] if math.isfinite(v)]
                fh.write(json.dumps({
                    "row": r, "family": "realact", "n": N_ROLLOUTS, "mean_cos": 0.25,
                    "max_cos": 0.25, "bo_1": 0.25, "bo_2": 0.25, "bo_4": 0.25,
                    "mean_cos_centred": sum(fin) / len(fin), "max_cos_centred": max(fin),
                    "n_centred": len(fin)}) + "\n")
        (d / "rows.json").write_text(json.dumps(
            {"rows": [0, 1], "n": N_ROLLOUTS, "families": ["realact"] * 2,
             "score_max_length": WIDTH - 1, "mu": "/mu.npy"}))
        index = {"per_target.jsonl": {"kind": "jsonl", "rows": 2}}
        for name, vals in (("cos.f16", raw), ("cos_centred.f16", centred)):
            arr = np.stack([_block(vals[r]) for r in (0, 1)]).astype(np.float16)
            arr.tofile(d / name)
            index[name] = {"kind": "array", "dtype": "float16",
                           "shape": list(arr.shape), "bytes": arr.nbytes}
        (d / "index.json").write_text(json.dumps(index))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write(root)
        vol = R.Vol("", root, offline=True, quiet=True)
        res = F.analyse(vol, cfg, "CB", "", 400, 1, True, 8.0, 128.0)
        # (1) the all-finite row IS `bo_ladder`, exactly.
        ladder, info = F.centred_bok(vol, res["sources"][0], 128.0)
        want0 = R.bo_ladder(centred[0], R.BO_KS_ALL)
        assert ladder[0] == want0, f"row 0: {ladder[0]} vs bo_ladder {want0}"
        _close(ladder[0][2], 5 / 6, 1e-9, what="row 0 bo2, the unbiased order statistic by hand")
        # (2)/(3) the NaN row, by hand.
        _close(ladder[1][1], 0.375, 1e-9, what="bo1 over the finite draws")
        _close(ladder[1][2], 0.5, 1e-9, what="bo2 over the TWO finite draws is their max")
        assert 4 not in ladder[1], (
            f"row 1 has two finite draws and must not answer a best-of-4, got {ladder[1]}"
        )
        assert info["rows_with_nan_rollouts"] == 1 and info["nan_rollouts"] == 2, info
        # The product stores no ladder at all, so there was nothing to compare against -- and the
        # table must SAY that rather than printing a vacuous zero-mismatch pass.
        assert info["stored_comparisons"] == 0 and info["n_mismatches"] == 0, info
        # (and the family mean is the average of the two rows, from the ARRAY not the file)
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_centred.bo2"), (5 / 6 + 0.5) / 2, 1e-9)
        # row 1 supplies no bo4 at all, so the bo4 cell is row 0's alone -- not row 0's averaged
        # against a clamped stand-in for row 1.
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_centred.bo4"), 1.0, 1e-9)
        cell = [c for c in res["cos"] if c["cosine"] == "cos_centred"][0]
        assert cell["bo_source"].startswith("cos_centred.f16"), cell["bo_source"]
        raw_cell = [c for c in res["cos"] if c["cosine"] == "cos_raw"][0]
        assert raw_cell["bo_source"] == "per_target.jsonl", raw_cell["bo_source"]

        # (4) the size refusal: loud, with a reason, and no centred cells at all.
        res2 = F.analyse(vol, cfg, "CB", "", 400, 1, False, 8.0, 1e-6)
        assert not [c for c in res2["cos"] if c["cosine"] == "cos_centred"], (
            "over the size bound the centred columns must be ABSENT, not filled from the file")
        chk = [c for c in res2["checks"] if c["kind"] == "cos_centred bo-k"][0]
        assert "over --centred-bok-max-mb" in chk.get("skipped", ""), chk


def check_sae_side_reads_its_own_product():
    """A `--sides enc,dec` set: each side reads ITS OWN `sae_self` product, or is reported missing.

    Both halves of such a set live in ONE scores directory, so `sae_self` writes the encoder half
    to `sae_self/` (its historical name) and the decoder half to `sae_self__dec/`. The failure this
    pins is the quiet one: a driver that read `sae_self/` for both would print the ENCODER block's
    activations under the decoder label, and nothing in the numbers would say so -- on the real set
    the card's 0.344 / 0.416 pair would come back 0.344 / 0.344 and read as a finding.

    The fixture gives the two products deliberately DIFFERENT peaks, so an equal answer is a
    failure rather than a coincidence, and then deletes the decoder product and requires the family
    to be reported missing instead of falling back onto the encoder's numbers.
    """
    ids = [
        {"row": 0, "family": "sae", "sae_key": "B/sae-one", "sae_side": "enc", "stratum": 0, "id": "f100"},
        {"row": 1, "family": "sae", "sae_key": "B/sae-one", "sae_side": "enc", "stratum": 1, "id": "f200"},
        {"row": 2, "family": "sae", "sae_key": "B/sae-one", "sae_side": "dec", "stratum": 0, "id": "f100"},
        {"row": 3, "family": "sae", "sae_key": "B/sae-one", "sae_side": "dec", "stratum": 1, "id": "f200"},
    ]
    # corpus peak 4.0 on every row, so the ratios are the peaks / 4 and hand-checkable.
    peaks = {0: [1.0] * 4, 1: [1.0] * 4, 2: [2.0] * 4, 3: [2.0] * 4}
    cfg = {**CFG, "heldout": {**CFG["heldout"],
                              "SS": {"base": BASE, "sae_key": "B/sae-one",
                                     "families": {"sae": {"n": 4}}}}}

    def write(root: Path) -> Path:
        hd = root / f"base/{BASE}/heldout/SS"
        hd.mkdir(parents=True, exist_ok=True)
        with open(hd / "ids.jsonl", "w") as fh:
            for r in ids:
                fh.write(json.dumps(r) + "\n")
        (hd / "storage.json").write_text(json.dumps({"storage": "dirs_only", "sae_key": "B/sae-one"}))
        d = root / f"maemms/{BASE}/ckpt-one/scores/SS__arm-a"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "per_target.jsonl", "w") as fh:
            for r in range(4):
                fh.write(json.dumps({"row": r, "family": "sae", "n": N_ROLLOUTS, "mean_cos": 0.0625,
                                     "max_cos": 0.0625, "bo_1": 0.0625, "bo_4": 0.0625}) + "\n")
        (d / "rows.json").write_text(json.dumps(
            {"rows": [0, 1, 2, 3], "n": N_ROLLOUTS, "families": ["sae"] * 4,
             "score_max_length": WIDTH - 1, "mu": None}))
        (d / "index.json").write_text(json.dumps({"per_target.jsonl": {"kind": "jsonl", "rows": 4}}))
        for name, rows_ in (("sae_self", [0, 1]), ("sae_self__dec", [2, 3])):
            sd = d / name
            sd.mkdir(exist_ok=True)
            np.stack([_block(peaks[r]) for r in rows_]).astype(np.float16).tofile(sd / "sae_self.f16")
            (sd / "sae_self.json").write_text(json.dumps({
                "rows": rows_, "features": [100, 200], "n": N_ROLLOUTS, "width": WIDTH,
                "gate": SAE_GATE, "checks": {"argmax_ok": True, "sae_side": name[-3:]},
                "per_target": [{"row": r, "feature": 100 + 100 * (r % 2), "n": N_ROLLOUTS,
                                "corpus_peak": 4.0, "mean_peak_act": peaks[r][0],
                                "max_peak_act": peaks[r][0]} for r in rows_],
            }))
        return d

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = write(root)
        vol = R.Vol("", root, offline=True, quiet=True)
        res = F.analyse(vol, cfg, "SS", "", 400, 1, True, 8.0)
        fams = sorted({a["family"] for a in res["sae"]})
        assert fams == ["sae/sae-one/dec", "sae/sae-one/enc"], fams
        enc = _stat(res, "sae/sae-one/enc", "ckpt-one:arm-a", "ratio.bo1.median")
        dec = _stat(res, "sae/sae-one/dec", "ckpt-one:arm-a", "ratio.bo1.median")
        _close(enc, 0.25, 1e-9, what="the enc family reads sae_self/ -- peaks 1.0 over corpus 4.0")
        _close(dec, 0.5, 1e-9, what="the dec family reads sae_self__dec/ -- peaks 2.0 over corpus 4.0")
        # And each check names the product it actually opened, so the table's provenance is real.
        products = sorted(c["product"] for c in res["checks"] if c["kind"] == "sae_self")
        assert products == [f"maemms/{BASE}/ckpt-one/scores/SS__arm-a/sae_self",
                            f"maemms/{BASE}/ckpt-one/scores/SS__arm-a/sae_self__dec"], products

    # The decoder product removed: MISSING, never the encoder's numbers under the decoder label.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = write(root)
        shutil.rmtree(d / "sae_self__dec")
        vol = R.Vol("", root, offline=True, quiet=True)
        res = F.analyse(vol, cfg, "SS", "", 400, 1, True, 8.0)
        fams = sorted({a["family"] for a in res["sae"]})
        assert fams == ["sae/sae-one/enc"], f"the dec family was answered from another product: {fams}"
        joined = " ".join(res["missing"])
        assert "sae/sae-one/dec" in joined and "sae_self" in joined, joined


def check_second_sae_needs_no_code():
    """A second dictionary, distinguished ONLY by the rows' own `sae_key`, gets its own family."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        _vol, res = _analyse(root)
        fams = sorted({a["family"] for a in res["sae"]})
        assert fams == ["sae/sae-one", "sae/sae-two"], fams
        # row 6 peaks [2,2,2,2] / peak 2 -> 1.0; row 7 [0.25] / 1 -> 0.25; median 0.625.
        _close(_stat(res, "sae/sae-two", "ckpt-one:arm-a", "ratio.bo1.median"), 0.625, 1e-9)
        _close(_stat(res, "sae/sae-two", "ckpt-one:arm-a", "fired.feature"), 0.5, 1e-9)
        # And the two dictionaries' rows never mix: four features, two per family.
        for f in fams:
            got = [a for a in res["sae"] if a["family"] == f and a["stratum"] is None]
            assert {a["n_features"] for a in got} == {2}, (f, got)


def check_missing_sources_are_listed_not_zeroed():
    """A config'd checkpoint with no products is REPORTED; `compute: false` is not; a source with
    no `sae_self` loses its SAE rows and says so, and keeps its cosines."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        _vol, res = _analyse(root)
        joined = " | ".join(res["missing"])
        assert "ckpt-nothing" in joined, joined
        assert "ckpt-declared-only" not in joined, "a `compute: false` entry is not missing"
        assert "ckpt-two" in joined and "sae_self" in joined, joined
        # ckpt-two still contributes its cosines -- a missing SAE product is not a missing source.
        assert any(c["source"] == "ckpt-two" for c in res["cos"])
        assert not any(a["source"] == "ckpt-two" for a in res["sae"])
        # Its cos reader check SKIPS (no arrays) rather than passing vacuously.
        chk = [c for c in res["checks"] if c["kind"] == "cos" and c["source"] == "ckpt-two"][0]
        assert "skipped" in chk and "cos.f16" in chk["skipped"], chk


def check_sources_filter():
    """`--sources` keeps a subset and does not repaint the survivors: colour follows the entity."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        _vol, full = _analyse(root)
        _vol2, cut = _analyse(root, sources="arm-b")
        assert [s.label for s in cut["sources"]] == ["ckpt-one:arm-b"]
        assert cut["colours"]["ckpt-one:arm-b"] == full["colours"]["ckpt-one:arm-b"]


def check_reader_check_catches_a_defect():
    """Corrupt ONE stored aggregate and the reader check must go red. Without this the check is
    unevaluated: a comparison that has only ever passed proves nothing about its ability to fail."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        p = root / f"maemms/{BASE}/ckpt-one/scores/{SET}__arm-a/per_target.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        recs[0]["mean_cos"] = 0.9        # the array still says 0.625
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _analyse(root)
        chk = [c for c in res["checks"] if c["kind"] == "cos" and c["source"] == "ckpt-one:arm-a"][0]
        assert chk["n_mismatches"] == 1, chk
        assert chk["worst_excess"] > 0.27, chk

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        p = root / f"maemms/{BASE}/ckpt-one/scores/{SET}__arm-a/sae_self/sae_self.json"
        meta = json.loads(p.read_text())
        meta["per_target"][0]["max_peak_act"] = 9.0   # the array still says 4.0
        p.write_text(json.dumps(meta))
        _vol, res = _analyse(root)
        chk = [c for c in res["checks"] if c["kind"] == "sae_self"
               and c["source"] == "ckpt-one:arm-a"][0]
        assert chk["n_mismatches"] == 1 and chk["worst_excess"] > 4.9, chk


def check_sanity_verdicts():
    """All four verdicts a gate can reach: pass, FLAG, absent, and a declared no-verdict."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        _vol, res = _analyse(root)
        y = root / "sanity.yaml"
        y.write_text(SANITY)
        got = F.run_sanity(res, y)
        verdicts = [g["verdict"] for g in got]
        assert verdicts == ["pass", "FLAG", "absent", "no verdict"], verdicts
        assert "bsf" in got[2]["why"] or "not present" in got[2]["why"], got[2]
        _close(got[0]["ours"], 1.375 / 3, 1e-6)

        # `set:` restricts a gate to ONE block. Without it, the four-row v1raw smoke's numbers
        # resolved against the 512-row v3 block and FLAGged -- a flag that said nothing about the
        # mu, which is the only thing a flag is supposed to mean. The SAME gate must pass on its
        # own block and be `absent` on any other, with the reason naming both names.
        gate = SANITY.replace("  - name: arm-a realact mean cos",
                              f"  - name: arm-a realact mean cos\n    set: {SET}", 1)
        y.write_text(gate)
        here = F.run_sanity(res, y)
        assert here[0]["verdict"] == "pass", here[0]
        y.write_text(SANITY.replace("  - name: arm-a realact mean cos",
                                    "  - name: arm-a realact mean cos\n    set: another-block", 1))
        elsewhere = F.run_sanity(res, y)
        assert elsewhere[0]["verdict"] == "absent", elsewhere[0]
        assert "another-block" in elsewhere[0]["why"] and SET in elsewhere[0]["why"], elsewhere[0]
        # and the restriction does not touch the gates that do not carry it
        assert [g["verdict"] for g in elsewhere[1:]] == ["FLAG", "absent", "no verdict"], elsewhere


def check_cross_set_gate():
    """A `kind: cross_set` gate compares this set's arm with another set's, ON THE SHARED ROWS.

    `S2`'s product carries rows [0, 1, 4, 5] and `S`'s carries all eight, so the realact
    comparison must run on rows 0 and 1 alone. A gate that took each side's own family mean would
    get 0.4583 vs 0.3333 instead of 0.5625 vs 0.3750 -- which is what the intersection is for.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        vol, res = _analyse(root)
        y = root / "x.yaml"
        y.write_text("""
checks:
  - name: cross-set, realact, shared rows
    kind: cross_set
    from_set: S2
    from_source: "ckpt-one:arm-a"
    source: "ckpt-one:arm-a"
    family: realact
    metric: cos_raw.bo1
    tol: 0.2
  - name: cross-set, the same pair at a tight tolerance
    kind: cross_set
    from_set: S2
    from_source: "ckpt-one:arm-a"
    source: "ckpt-one:arm-a"
    family: realact
    metric: cos_raw.bo1
    tol: 0.01
  - name: cross-set on a family the other product does not carry
    kind: cross_set
    from_set: S2
    from_source: "ckpt-one:arm-a"
    source: "ckpt-one:arm-a"
    family: random
    metric: cos_raw.bo1
    tol: 0.2
  - name: cross-set against an undeclared set
    kind: cross_set
    from_set: S9
    from_source: "ckpt-one:arm-a"
    source: "ckpt-one:arm-a"
    family: realact
    metric: cos_raw.bo1
    tol: 0.2
  - name: cross-set on an activation metric, which does not resolve
    kind: cross_set
    from_set: S2
    from_source: "ckpt-one:arm-a"
    source: "ckpt-one:arm-a"
    family: sae/sae-one
    metric: ratio.bo1.median
    tol: 0.2
""")
        got = F.run_sanity(res, y, vol, CFG)
        assert [g["verdict"] for g in got] == ["pass", "FLAG", "absent", "absent", "absent"], \
            [(g["name"], g["verdict"]) for g in got]
        _close(got[0]["ours"], 0.5625, 1e-9, what="ours, rows 0 and 1 only")
        _close(got[0]["expect"], 0.375, 1e-9, what="theirs, rows 0 and 1 only")
        assert got[0]["n"] == 2, got[0]
        assert "row" in got[2]["why"], got[2]["why"]
        assert "heldout" in got[3]["why"], got[3]["why"]
        assert "cosine metric" in got[4]["why"], got[4]["why"]
        # Without a volume the gate is absent and says why, never silently skipped.
        assert F.run_sanity(res, y, None, None)[0]["verdict"] == "absent"


def check_render_and_figures():
    """The whole render: tables.md, a CSV per table, figures as PDF AND PNG, and the rule that no
    SAE-target cosine reaches a markdown table (plan §2.3)."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        _vol, res = _analyse(root)
        out_dir = root / "out"
        y = root / "sanity.yaml"
        y.write_text(SANITY)
        figs = F.make_figures(res, out_dir)
        o = R.Out(out_dir, "selftest", ["- synthetic"])
        path = F.render(res, o, F.run_sanity(res, y), figs)
        md = path.read_text()

        for name in ("cos_realact", "cos_random", "act_sae_sae-one", "act_sae_sae-two", "sanity",
                     "reader_check"):
            assert (out_dir / f"{name}.csv").exists(), f"{name}.csv was not written"
        assert (out_dir / "sae_cosines.csv").exists()
        # The SAE tables carry activation ratios and no cosine column.
        sae_block = md.split("### SAE activation — family `sae/sae-one`")[1].split("###")[0]
        header = next(ln for ln in sae_block.splitlines() if ln.startswith("| source"))
        assert "cos" not in header, f"a cosine column leaked into the SAE table: {header}"
        assert "mean_cos" not in sae_block and "max_cos" not in sae_block, sae_block
        assert "mean_cos" in (out_dir / "sae_cosines.csv").read_text()
        # The tables themselves.
        assert "0.4583" in md, "the realact cos_raw bo1 mean is not in tables.md"
        assert "ckpt-nothing" in md, "a missing source must be listed in the file, not only printed"
        assert "bo8" in md and "—" in md, "a bo-k the run could not compute prints as an em dash"
        assert len(figs) >= 3, figs
        for f in figs:
            for ext in ("pdf", "png"):
                p = out_dir / "figures" / f"{f}.{ext}"
                assert p.exists() and p.stat().st_size > 1000, p
        # The two-arm figure went with the old primary's mu-none / mu-stats arms (M0a,
        # 2026-09-23): this fixture still carries two run tags of one checkpoint, so its ABSENCE
        # is the assertion -- a stale `arms_*` here would mean the deletion did not land.
        assert not any(f.startswith("arms_") for f in figs), figs
        assert any(f.startswith("strata_") for f in figs), figs
        assert any(f.startswith("bok_") for f in figs), figs


def check_combined_layer_lifts_and_never_recomputes():
    """The cross-set document: the headline table, the merged sanity block, and the index.

    `analyse` is per SET on purpose -- a set is a storage contract and a family list -- so the
    combined layer must not merge two blocks into one namespace. Two things are pinned:

      1. every headline cell is BIT-IDENTICAL to the block cell it came from. A combined layer
         that re-averaged the rows of two blocks would produce a plausible number that matches no
         table in the document, which is the failure that has no symptom;
      2. `realact` measured on two different blocks stays TWO rows, keyed by (block, family), and
         both reach the figure. Collapsing them is how "the upstream draw" and "our draw" become one
         number that is neither.

    The sanity block is merged across blocks with the block named on every line, and a gate that
    resolves on one and not the other must appear twice, once with each verdict.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_mirror(root)
        vol = R.Vol("", root, offline=True, quiet=True)
        # The same rows under two block names: SET carries realact/random/sae, SET2 carries
        # realact only, on four of the eight rows and with a different arm's numbers.
        a = F.analyse(vol, CFG, SET, "", 400, 1, True, 8.0, 128.0)
        b = F.analyse(vol, CFG, SET2, "", 400, 1, True, 8.0, 128.0)
        all_res = [a, b]

        hl = F.headline_rows(all_res)
        assert hl, "the headline carries no row at all"
        # The arms are DISCOVERED, never spelled: a checkpoint name in the driver is the one
        # thing this file promises not to have.
        assert F.headline_arms(all_res) == ["ckpt-one:arm-a", "ckpt-one:arm-b", "ckpt-two"], (
            F.headline_arms(all_res))
        import re as _re
        src = (Path(F.__file__)).read_text()
        assert not _re.search(r"rl-last16|rl-8x2048|nla-av", src), (
            "a checkpoint name is spelled in faithfulness.py")
        # (1) every cell is lifted, not recomputed.
        for r in hl:
            src = [c for c in (a["cos"] + b["cos"])
                   if c["source"] == r["source"] and c["family"] == r["family"]
                   and c["cosine"] == r["cosine"] and c["n_rows"] == r["n_rows"]]
            assert src and src[0]["bo"] is r["bo"], (
                f"the headline row {r['set']}/{r['family']}/{r['arm']}/{r['cosine']} does not "
                f"carry its block's own `bo` dict -- something recomputed it")
        # (2) one family, two blocks, two rows -- and they are DIFFERENT numbers here, so a
        # collapse cannot pass by coincidence (arm-a on SET vs arm-a on SET2 = PT_A vs PT_B).
        ra = [r for r in hl if r["family"] == "realact" and r["cosine"] == "cos_raw"]
        by_set = {r["set"]: r["bo"][1]["mean"] for r in ra}
        assert set(by_set) == {SET, SET2}, f"realact did not survive as two blocks: {by_set}"
        assert abs(by_set[SET] - by_set[SET2]) > 1e-6, by_set

        y = root / "sanity.yaml"
        y.write_text(SANITY)
        sanity = {SET: F.run_sanity(a, y), SET2: F.run_sanity(b, y)}
        out_dir = root / "all"
        # The figure's columns are keyed on (block, family), so `realact` gets one per block.
        keys = F.headline_keys(all_res)
        assert (SET, "realact") in keys and (SET2, "realact") in keys, keys
        assert len(keys) == len({(s_, f_) for s_, f_ in keys}), keys
        figs = F.make_headline_figure(all_res, out_dir)
        assert figs == ["headline_cosine"], figs
        for ext in ("pdf", "png"):
            fp = out_dir / "figures" / f"headline_cosine.{ext}"
            assert fp.exists() and fp.stat().st_size > 1000, fp
        o = R.Out(out_dir, "selftest combined", ["- synthetic"])
        path = F.render_combined(all_res, o, sanity, figs,
                                 {SET: Path("s"), SET2: Path("s2")})
        md = path.read_text()
        assert (out_dir / "headline.csv").exists() and (out_dir / "sanity_all.csv").exists()
        # The BLOCK COLUMN is populated on every headline row -- a table that printed the two
        # blocks' `realact` rows under the same (blank) name would put two different numbers
        # side by side with nothing saying which draw each came from.
        hl_md = md.split("### Headline")[1].split("\nCSV:")[0]
        body = [ln for ln in hl_md.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
        assert body, hl_md
        blocks_seen = {ln.split("|")[1].strip() for ln in body}
        assert blocks_seen == {SET, SET2}, f"the headline block column reads {blocks_seen}"
        csv_blocks = {ln.split(",")[0] for ln in
                      (out_dir / "headline.csv").read_text().splitlines()[1:]}
        assert csv_blocks == {SET, SET2}, csv_blocks
        # The headline number reaches the document, and so does the other block's.
        for v in by_set.values():
            assert f"{v:.4f}" in md, f"{v:.4f} is not in the combined tables.md"
        # Every sanity line names its block, and the gate that resolves on SET but not on SET2
        # (SET2 has no `sae` rows and a different source set) appears under both.
        rows = [ln for ln in md.splitlines() if ln.startswith("| S ") or ln.startswith("| S2 ")]
        assert any(ln.startswith("| S ") for ln in rows) and any(ln.startswith("| S2 ") for ln in rows), (
            "the merged sanity block does not name the block on every line")
        assert "a gate that must flag" in md and "FLAG" in md, "the flagging gate is not reported"
        # The index points at each block's own document.
        assert "s/tables.md" in md and "s2/tables.md" in md, md[-2000:]


# --- eval 2: the autointerp fixture, worked out by hand -----------------------------------------
#
# Two run directories, because `autointerp/run.py` is per CHECKPOINT and eval 2's arms are spread
# over six of them (three checkpoints x two SAEs). `rl16` is the reference run: a bare `--ref
# DOCMAX` resolves inside the FIRST --run given, and both runs carry a DOCMAX of their own.
AI_SAE = "B/sae-2m"
AI_RUNS = {"rl16": "2026-09-22_autointerp-rl16", "old": "2026-09-22_autointerp-old"}
AI_FEATS = [10, 11, 12, 13]
AI_STRATUM = {10: 0, 11: 0, 12: 1, 13: 1}
# The shown-example count per arm, as `build.ARM_SPECS` records it: 16 for a corpus arm, 4 for NLA
# (the verbalizer answers at 200 tokens and four samples is what the budget buys), and 0 for the
# scorer-only floor, which borrows another feature's description and has no example set at all.
AI_NEX = {"DOCMAX": 16, "M": 16, "NLA": 4, "R-shuffled": 0}
AI_ROLE = {"R-shuffled": "floor"}
# The two A5 negative-half views restrict only the NEGATIVE side, so at a fixed TNR they are
# `0.5 * (bal_acc + tnr_view)`. Fixed here rather than varied, so the reader check has three exact
# identities per row and none of them has a rounding budget to hide in.
AI_TNR_ZERO = 0.5
AI_TNR_NEAR = 1.0

# bal_acc per (run, arm) x scorer x feature. Every value is dyadic. `None` is the row `run.py`
# writes when every batch of a (feature, arm) went unparsed: `rates` sees no scored items, returns
# NaN, and `_nr` stores null -- an ABSENT measurement, which must be dropped from the mean and
# counted, never imputed as 0.5.
AI_BAL = {
    ("rl16", "DOCMAX"): {"detection": {10: 0.75, 11: 0.5, 12: 0.625, 13: 0.875},
                         "fuzzing": {10: 0.5, 11: 0.625, 12: 0.5, 13: 0.625}},
    ("rl16", "M"): {"detection": {10: 0.5, 11: 0.5, 12: 0.75, 13: 0.625},
                    "fuzzing": {10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5}},
    ("rl16", "R-shuffled"): {"detection": {10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5},
                             "fuzzing": {10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5}},
    # Three features of four: the arm that makes the pairing a real intersection.
    ("rl16", "NLA"): {"detection": {10: 0.375, 11: 0.5, 12: 0.5},
                      "fuzzing": {10: 0.375, 11: 0.375, 12: 0.5}},
    ("old", "DOCMAX"): {"detection": {10: 0.625, 11: 0.625, 12: 0.5, 13: 0.75},
                        "fuzzing": {10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5}},
    ("old", "M"): {"detection": {10: 0.5, 11: 0.625, 12: 0.625, 13: 0.5},
                   "fuzzing": {10: 0.625, 11: 0.5, 12: 0.5, 13: None}},
}
# One feature short of a full parse, so the support table's parse rate is not a column of 1.0.
AI_PARSED = {("rl16", "M", "detection", 13): 3}
# `explanation_ok: false` cannot occur on a real product -- `run.py` never submits a scoring job
# for an arm with an empty description, so no batch exists and no row is written. It is set here on
# ONE row so the counter and its table column are exercised; the driver's caption says a nonzero
# count on a real run would itself be a defect.
AI_NO_EXPL = {("old", "M", "detection", 13)}
AI_N_BATCHES = 4
AI_N_ITEMS = 40
# (run, arm, feature) whose explainer call was DECLINED, which is now WHY `rl16/NLA` covers three
# of the four features: a refusal leaves no score row, and the arm that was already short in this
# fixture is short for exactly that reason. So one arm grows a `refusal=chance` sibling and every
# other arm must stay single-rowed, and the refused feature is one with no `scores.jsonl` row to
# contradict it -- which is the only self-consistent way to write one.
AI_REFUSED = {("rl16", "NLA", 13)}


def _ai_row(run: str, arm: str, scorer: str, feat: int, bal) -> dict:
    """One `scores.jsonl` row as `autointerp/run.py` writes it.

    `tpr = tnr = bal_acc` is what makes `0.5 * (tpr + tnr)` reproduce the stored `bal_acc` exactly,
    which is the identity the reader check compares and the one `check_autointerp_catches_a_defect`
    breaks. A null `bal_acc` carries null rates with it, as `run._nr` does.
    """
    n_parsed = AI_PARSED.get((run, arm, scorer, feat), AI_N_BATCHES if bal is not None else 0)
    return {
        "feature": feat, "arm": arm, "scorer": scorer,
        "bal_acc": bal, "tpr": bal, "tnr": bal,
        "bal_acc_zero_neg": None if bal is None else 0.5 * (bal + AI_TNR_ZERO),
        "tnr_zero": None if bal is None else AI_TNR_ZERO,
        "bal_acc_nearmiss_neg": None if bal is None else 0.5 * (bal + AI_TNR_NEAR),
        "tnr_nearmiss": None if bal is None else AI_TNR_NEAR,
        "acc": bal,
        "n_items": AI_N_ITEMS if bal is not None else 0,
        "n_batches": AI_N_BATCHES, "n_parsed": n_parsed,
        "n_pos": 20 if bal is not None else 0, "n_neg_nearmiss": 10 if bal is not None else 0,
        "draw": 1, "role": AI_ROLE.get(arm, "arm"), "n_examples": AI_NEX[arm],
        "explanation_ok": (run, arm, scorer, feat) not in AI_NO_EXPL,
        "explanation_of": feat, "path": "batch", "gate": 2.0,
        "stratum": AI_STRATUM[feat], "fire_fraction": 0.25, "corpus_peak": 8.0, "density": -4.0,
    }


def write_autointerp_runs(root: Path, sae: str = AI_SAE) -> None:
    """The synthetic mirror in the volume's own `runs/<dir>/summary/` layout."""
    for run, run_dir in AI_RUNS.items():
        d = root / f"runs/{run_dir}/summary"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for (r, arm), per_scorer in AI_BAL.items():
            if r != run:
                continue
            for scorer, vals in per_scorer.items():
                for feat in sorted(vals):
                    rows.append(_ai_row(run, arm, scorer, feat, vals[feat]))
        with open(d / "scores.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        (d / "build.json").write_text(json.dumps(
            {"base": "B", "set": "S3", "maemm": f"B/{run}", "engine": "vllm", "sae": sae,
             "n_features": len(AI_FEATS), "mark": "gate", "fuzz_marks": "contiguous"}))
        # The explain stage's own record, which is the ONLY place a refusal exists: `run.py`
        # writes no score row for a refused (feature, arm), so `scores.jsonl` above cannot carry
        # one. Feature 13's `M` call was declined in the `rl16` run -- note that `scores.jsonl`
        # has no `(13, M)` row to match, which is what a real refusal looks like.
        e = root / f"runs/{run_dir}/explain"
        e.mkdir(parents=True, exist_ok=True)
        with open(e / "explanations.jsonl", "w") as fh:
            for arm in sorted({a for r, a in AI_BAL if r == run}):
                for feat in AI_FEATS:
                    refused = (run, arm, feat) in AI_REFUSED
                    fh.write(json.dumps({
                        "feature": feat, "arm": arm, "n_examples": AI_NEX[arm],
                        "explanation": "" if refused else f"a description of {feat}",
                        "ok": not refused, "refused": refused,
                        "stop_reason": "refusal" if refused else "end_turn"}) + "\n")
            # TWO empties that are NOT refusals, because `refused` is not `not ok`:
            #
            #  * a SEEDED description the verbalizer left empty -- the explainer was never asked,
            #    so `stop_reason` is `seeded-from-build` and no call was declined;
            #  * an explainer answer that RETURNED NORMALLY and parsed to nothing (`end_turn`,
            #    `ok: false`), which is a parse failure and not a declined call.
            #
            # The second is deliberately placed on `rl16/M` -- an arm that HAS cells -- so that
            # miscounting empties as refusals changes a number this file asserts. The first sits
            # on an arm with no score rows, which is realistic and, on its own, untestable.
            fh.write(json.dumps({
                "feature": AI_FEATS[0], "arm": "NLA-desc", "n_examples": 0, "explanation": "",
                "ok": False, "refused": False, "stop_reason": "seeded-from-build"}) + "\n")
            if run == "rl16":
                fh.write(json.dumps({
                    "feature": AI_FEATS[0], "arm": "M", "n_examples": AI_NEX["M"],
                    "explanation": "", "ok": False, "refused": False,
                    "stop_reason": "end_turn"}) + "\n")


def _ai_analyse(root: Path, **kw):
    vol = R.Vol("", root, offline=True, quiet=True)
    opts = {"runs": dict(AI_RUNS), "sae": AI_SAE, "ref": "DOCMAX", "boot": 2000, "seed": 1,
            "strata": True, "peak_strata": True, "vs_runs": None, "vs_label": ""}
    opts.update(kw)
    return vol, A.analyse(vol, opts["runs"], opts["sae"], opts["ref"], opts["boot"], opts["seed"],
                          opts["strata"], opts["peak_strata"], opts["vs_runs"], opts["vs_label"])


def _ai_cell(res, arm: str, scorer: str, convention: str = "dropped") -> dict:
    """One headline cell. `convention` is required in spirit even though it has a default: an arm
    with refusals has TWO rows, and a helper that returned whichever came first would make every
    assertion below depend on dict ordering."""
    hits = [c for c in res["cells"] if c["arm"] == arm and c["scorer"] == scorer
            and c["convention"] == convention]
    assert len(hits) == 1, (arm, scorer, convention,
                            [(c["arm"], c["convention"]) for c in res["cells"]])
    return hits[0]


def _ai_contrast(res, arm: str, scorer: str) -> dict:
    hits = [x for x in res["contrasts"] if x["arm"] == arm and x["scorer"] == scorer]
    assert len(hits) == 1, (arm, scorer, sorted({x["arm"] for x in res["contrasts"]}))
    return hits[0]


def check_autointerp_reader():
    """The rows parse, the arms and scorers come back as expected, and the provenance is read.

    The ARM is `(run label, arm name)`, never the bare name: `DOCMAX` exists in both run
    directories with different numbers, and a reader that keyed on the name alone would merge two
    measurements into one row.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root)
        assert res["scorers"] == ["detection", "fuzzing"], res["scorers"]
        assert [a.label for a in res["arms"]] == [
            "rl16/DOCMAX", "rl16/M", "rl16/NLA", "rl16/R-shuffled", "old/DOCMAX", "old/M"], \
            [a.label for a in res["arms"]]
        # The reference resolves inside the FIRST --run, not among both DOCMAXes.
        assert res["ref"].label == "rl16/DOCMAX", res["ref"]
        # Roles, example counts and feature counts come off the rows themselves.
        assert _ai_cell(res, "rl16/R-shuffled", "detection")["roles"] == ["floor"]
        assert _ai_cell(res, "rl16/NLA", "detection")["n_examples"] == [4]
        assert _ai_cell(res, "rl16/DOCMAX", "detection")["n_examples"] == [16]
        assert _ai_cell(res, "rl16/NLA", "detection")["n_features"] == 3
        # A null bal_acc is an ABSENT measurement: dropped from the mean, counted, not imputed.
        om = _ai_cell(res, "old/M", "fuzzing")
        assert (om["n_rows"], om["n_features"], om["n_no_metric"]) == (4, 3, 1), om
        assert _ai_cell(res, "old/M", "detection")["n_no_explanation"] == 1
        # Parse rate: three features at 4/4 and one at 3/4.
        _close(_ai_cell(res, "rl16/M", "detection")["parse_rate"], 0.9375, 1e-12)
        _close(_ai_cell(res, "old/M", "fuzzing")["parse_rate"], 0.75, 1e-12)
        # The reader check compared something and found nothing wrong: 3 identities per row over
        # the 45 rows that carry a bal_acc, plus none for the null row.
        for chk in [c for c in res["checks"] if c["kind"] == "reader"]:
            assert chk["n_mismatches"] == 0 and chk["comparisons"] > 0, chk
            assert chk["worst_excess"] < 0, chk
        assert not res["missing"], res["missing"]
        assert res["builds"]["rl16"]["sae"] == AI_SAE

    # A run built on ANOTHER dictionary must be called out: every 131k feature id is also a valid
    # 2M index, so a 131k run tabulated under the 2M heading is a wrong table nothing else catches.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root, sae="B/sae-131k")
        _vol, res = _ai_analyse(root)
        assert any("BUILT ON SAE" in n for n in res["notes"]), res["notes"]

    # An absent run directory is REPORTED, never zero-filled.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root, runs={**AI_RUNS, "ghost": "2026-09-22_autointerp-ghost"})
        assert any("ghost" in m for m in res["missing"]), res["missing"]
        assert all(a.run != "ghost" for a in res["arms"])


def check_autointerp_refusals_and_conventions():
    """A refused explainer call is COUNTED, and its arm gets a second row under a stated convention.

    A refusal is the one failure that leaves no trace in `scores.jsonl` -- `run.py` emits no scorer
    job for it -- so before this it showed only as an arm with fewer features, indistinguishable
    from an arm that was built over a subset on purpose. `rl16/NLA` covers three of four features
    in this fixture and now says WHY.

    The two conventions are one dataset read two ways and the numbers must differ in the direction
    chance imputation implies: NLA's three survivors average 1.375/3 = 0.4583, and adding a fourth
    feature at 0.5 moves it to 1.875/4 = 0.4688 -- UP, because the arm was below chance. That is
    the whole point of calling the imputed row a lower bound only when refusals are non-random.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root)
        # Counted on the arm that was refused, and ZERO (not blank) on every arm that was not.
        assert _ai_cell(res, "rl16/NLA", "detection")["n_refused"] == 1
        assert _ai_cell(res, "rl16/M", "detection")["n_refused"] == 0
        assert _ai_cell(res, "rl16/DOCMAX", "detection")["n_refused"] == 0
        # The reported row is unchanged by any of this: three features, their own mean.
        drop = _ai_cell(res, "rl16/NLA", "detection")
        assert (drop["n_features"], drop["convention"]) == (3, "dropped"), drop
        _close(drop["mean"], 1.375 / 3, 1e-12)
        # The imputed row scores the refused feature at chance over ALL four.
        imp = _ai_cell(res, "rl16/NLA", "detection", "refusal=chance")
        assert (imp["n_features"], imp["n_imputed"]) == (4, 1), imp
        _close(imp["mean"], 1.875 / 4, 1e-12, what="(0.375 + 0.5 + 0.5) + 0.5, over 4")
        assert imp["mean"] > drop["mean"], "chance imputation must lift a below-chance arm"
        # Fuzzing too, on its own numbers -- the convention is per (arm, scorer), not per arm.
        _close(_ai_cell(res, "rl16/NLA", "fuzzing", "refusal=chance")["mean"], 1.75 / 4, 1e-12)
        # AN ARM WITH NO REFUSALS HAS EXACTLY ONE ROW. A spurious sibling would double that arm
        # in every table and in the figure.
        for arm in ("rl16/DOCMAX", "rl16/M", "rl16/R-shuffled", "old/DOCMAX", "old/M"):
            got = [c for c in res["cells"] if c["arm"] == arm and c["scorer"] == "detection"]
            assert len(got) == 1 and got[0]["convention"] == "dropped", (arm, got)
        # `refused` IS NOT `not ok`. Two empties in this fixture are not refusals -- a seeded
        # description the verbalizer left blank, and an explainer answer that returned normally
        # and parsed to nothing -- and the second is on `rl16/M`, whose count must stay 0. A
        # reader that counted empties would report a parse failure as a declined call.
        assert not [c for c in res["cells"] if c["arm"] == "rl16/NLA-desc"], "fixture check"
        assert _ai_cell(res, "rl16/M", "fuzzing")["n_refused"] == 0, "an empty is not a refusal"
        assert len([c for c in res["cells"] if c["arm"] == "rl16/M"
                    and c["scorer"] == "detection"]) == 1, "no sibling for a non-refusal empty"
        assert sum(c["n_refused"] for c in res["cells"]
                   if c["convention"] == "dropped" and c["scorer"] == "detection") == 1

        # THE IMPUTED ROW IS OUT OF EVERY INFERENTIAL TABLE. A convention must not reach a
        # contrast, a cut cell or a permutation null, where it would read as measurement.
        assert all(x["n_arm"] != 4 or x["arm"] != "rl16/NLA" for x in res["contrasts"])
        assert _ai_contrast(res, "rl16/NLA", "detection")["n_arm"] == 3
        assert all("chance" not in str(s.get("convention", "")) for s in res["strata"])
        st = {(s["arm"], s["scorer"], s["stratum"]): s for s in res["strata"]}
        assert st[("rl16/NLA", "detection", 1)]["n"] == 1, "the refused feature is not a cut cell"
        # And the registry names the two conventions apart, so a gate cannot select by accident.
        reg = A.stat_registry(res)
        _close(reg[("detection", "rl16/NLA", None, "bal_acc.mean")], 1.375 / 3, 1e-12)
        _close(reg[("detection", "rl16/NLA", None, "bal_acc_chance.mean")], 1.875 / 4, 1e-12)
        # Support answers for the REPORTED convention, so a gate's n matches its number.
        assert A.support_registry(res)[("detection", "rl16/NLA", None)] == 3

    # NO explain record at all: the column goes BLANK, never zero -- "we did not look" and "we
    # looked and found none" are different claims and a 0 would assert the second.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        for run_dir in AI_RUNS.values():
            (root / f"runs/{run_dir}/explain/explanations.jsonl").unlink()
        _vol, res = _ai_analyse(root)
        assert _ai_cell(res, "rl16/NLA", "detection")["n_refused"] is None
        assert any("refusals cannot be counted" in n for n in res["notes"]), res["notes"]
        # And with no count there is no imputed row to build: one row per arm.
        assert len([c for c in res["cells"] if c["scorer"] == "detection"
                    and c["arm"] == "rl16/NLA"]) == 1


def check_autointerp_versus_second_build():
    """The same arms under a SECOND build, differenced per feature rather than as two means.

    THE CONTROL IS THE POINT. A build that changes only how generated text is marked cannot touch
    an arm that consumes no generated text, so such an arm must come back with a difference of
    EXACTLY zero on every feature. That is what tells a genuine null from a pairing that silently
    fell apart: a broken pairing also produces small numbers, and only the exact-zero control
    distinguishes the two. Here `DOCMAX` is the untouched arm and `M` is the moved one.

    The per-feature difference is not the difference of the means whenever the two sides cover
    different features, which is why `paired_diff` is reused rather than the means subtracted.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        # A second build: DOCMAX byte-identical (the control), M shifted by +0.125 on every
        # feature, under its own run directories.
        vs = {}
        for run, run_dir in AI_RUNS.items():
            vs_dir = run_dir + "-second"
            vs[run] = vs_dir
            d = root / f"runs/{vs_dir}/summary"
            d.mkdir(parents=True, exist_ok=True)
            src = root / f"runs/{run_dir}/summary/scores.jsonl"
            with open(d / "scores.jsonl", "w") as fh:
                proto = None
                for ln in src.read_text().splitlines():
                    r = json.loads(ln)
                    if r["arm"] == "M" and r["bal_acc"] is not None:
                        r["bal_acc"] = r["tpr"] = r["tnr"] = r["bal_acc"] + 0.125
                    if r["arm"] == "NLA" and r["scorer"] == "detection":
                        proto = r
                    fh.write(json.dumps(r) + "\n")
                # The second build RECOVERS a feature the first lacked -- `rl16/NLA` has three of
                # four, and here the fourth is scored. This is the case that separates a PAIRED
                # difference from a difference of two means: the pairing is the three shared
                # features and is exactly 0, while the two arms' own means differ by +0.0729
                # because one is over three features and the other over four.
                if proto is not None:
                    fh.write(json.dumps({**proto, "feature": 13, "bal_acc": 0.75,
                                         "tpr": 0.75, "tnr": 0.75, "acc": 0.75}) + "\n")
        _vol, res = _ai_analyse(root, vs_runs=vs, vs_label="second")
        by = {(x["arm"], x["scorer"]): x for x in res["versus"]}
        # The untouched arm: every feature identical, so the difference is exactly zero, the
        # interval is degenerate and `unchanged` is the whole set.
        c = by[("rl16/DOCMAX", "detection")]
        assert (c["mean"], c["lo"], c["hi"]) == (0.0, 0.0, 0.0), c
        assert (c["n_zero"], c["n_paired"]) == (4, 4), c
        assert c["win_frac"] == 0.0 and c["complete"] is True, c
        # The moved arm: +0.125 on every feature it has.
        m = by[("rl16/M", "detection")]
        _close(m["mean"], 0.125, 1e-12)
        assert (m["win_frac"], m["n_zero"]) == (1.0, 0), m
        _close(m["vs_mean"] - m["base_mean"], 0.125, 1e-12)
        # PAIRED, not two means. `old/M` fuzzing has a null on feature 13 on BOTH sides, so the
        # pairing is three features and the arm's own mean is over three -- but an arm that
        # covered different features either side would make the two differ, which is the case
        # `paired_diff` exists for and which the n columns make visible.
        om = by[("old/M", "fuzzing")]
        assert (om["n_paired"], om["n_base"], om["n_vs"]) == (3, 3, 3), om
        _close(om["mean"], 0.125, 1e-12)
        # THE PAIRED DIFFERENCE IS NOT THE DIFFERENCE OF THE MEANS. `rl16/NLA` gains a fourth
        # feature in the second build: the three it shares with the first are unchanged, so the
        # paired difference is EXACTLY zero, while the arms' own means differ by +0.0729 because
        # they are taken over different feature counts. A driver that subtracted the means would
        # report an improvement that no feature actually made.
        nla = by[("rl16/NLA", "detection")]
        assert (nla["n_paired"], nla["n_base"], nla["n_vs"]) == (3, 3, 4), nla
        assert nla["complete"] is False, nla
        _close(nla["mean"], 0.0, 1e-12, what="the three shared features did not move")
        assert (nla["n_zero"], nla["win_frac"]) == (3, 0.0), nla
        _close(nla["vs_mean"] - nla["base_mean"], 2.125 / 4 - 1.375 / 3, 1e-12)
        assert abs((nla["vs_mean"] - nla["base_mean"]) - nla["mean"]) > 0.07, nla
        # A second build that does not carry an arm at all simply has no row for it, rather than
        # a zero difference that would read as "unchanged".
        assert not [x for x in res["versus"] if x["arm"] == "old/NLA"]

        # It renders, with the control row's exact zero visible in the table.
        o = R.Out(root / "out", "selftest — versus", ["- synthetic"])
        md = A.render(res, o, [], []).read_text()
        blk = md.split("### Paired difference against a second build")[1].split("###")[0]
        assert "spans zero is the result" in blk, blk[:400]
        doc = next(ln for ln in blk.splitlines()
                   if ln.startswith("| DOCMAX | rl16 | detection |"))
        assert "+0.0000 [+0.0000, +0.0000]" in doc and doc.rstrip().endswith("| 4/4 |"), doc
        assert (root / "out" / "versus.csv").exists()

    # No `--vs` at all: no table, no rows, and nothing else changes.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root)
        assert res["versus"] == []
        o = R.Out(root / "out", "x", ["-"])
        assert "Paired difference against a second build" not in A.render(res, o, [], []).read_text()

    # An absent second run directory is REPORTED, never silently paired against nothing.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root, vs_runs={"rl16": "2026-09-22_nope"}, vs_label="second")
        assert res["versus"] == []
        assert any("nope" in m for m in res["missing"]), res["missing"]


def check_autointerp_estimators():
    """`boot_ci` and `paired_diff` against numbers worked out here, then the means through
    `analyse`.

    The two-value cases are EXACT and not an approximation of one: resampling two values with
    replacement gives the low value with probability 1/4, the mean with 1/2 and the high value with
    1/4, so the 2.5th percentile of the bootstrap distribution IS the low value and the 97.5th IS
    the high one -- both tails are far wider than alpha/2. That is what makes a literal CI possible
    without running the code twice.
    """
    _close(A.boot_ci([0.25, 0.75], 2000, 1)[0], 0.5, 1e-12, what="mean of two")
    _close(A.boot_ci([0.25, 0.75], 2000, 1)[1], 0.25, 1e-12, what="lo of two = the low value")
    _close(A.boot_ci([0.25, 0.75], 2000, 1)[2], 0.75, 1e-12, what="hi of two = the high value")
    # A constant vector has a degenerate bootstrap: every resample has the same mean.
    assert A.boot_ci([0.5] * 4, 2000, 1) == (0.5, 0.5, 0.5)
    # One value is a mean and no spread; a zero-width interval would read as certainty.
    m, lo, hi = A.boot_ci([0.5], 2000, 1)
    assert m == 0.5 and math.isnan(lo) and math.isnan(hi)
    assert all(math.isnan(v) for v in A.boot_ci([], 2000, 1))
    # A four-value bootstrap mean can never leave [min, max], and the interval brackets the mean.
    m, lo, hi = A.boot_ci([0.75, 0.5, 0.625, 0.875], 2000, 1)
    _close(m, 0.6875, 1e-12, what="(0.75 + 0.5 + 0.625 + 0.875) / 4")
    assert 0.5 <= lo <= m <= hi <= 0.875, (lo, m, hi)

    # paired_diff: the intersection by FEATURE ID, with both sides' losses carried out.
    a = {1: 0.75, 2: 0.25, 3: 0.5}
    b = {1: 0.25, 3: 0.25}
    pd = A.paired_diff(a, b)
    assert pd["features"] == [1, 3] and (pd["n_paired"], pd["n_a"], pd["n_b"]) == (2, 3, 2)
    assert pd["only_in_a"] == [2] and pd["only_in_b"] == [] and pd["complete"] is False
    assert list(pd["d"]) == [0.5, 0.25], list(pd["d"])
    _close(A.boot_ci(pd["d"], 2000, 1)[0], 0.375, 1e-12)
    _close(A.boot_ci(pd["d"], 2000, 1)[1], 0.25, 1e-12)
    _close(A.boot_ci(pd["d"], 2000, 1)[2], 0.5, 1e-12)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root)
        # Means over features, per (arm, scorer). Detection and fuzzing are separate numbers.
        _close(_ai_cell(res, "rl16/DOCMAX", "detection")["mean"], 2.75 / 4, 1e-12)
        _close(_ai_cell(res, "rl16/M", "detection")["mean"], 2.375 / 4, 1e-12)
        _close(_ai_cell(res, "rl16/NLA", "detection")["mean"], 1.375 / 3, 1e-12)
        _close(_ai_cell(res, "old/DOCMAX", "detection")["mean"], 2.5 / 4, 1e-12)
        _close(_ai_cell(res, "rl16/DOCMAX", "fuzzing")["mean"], 2.25 / 4, 1e-12)
        _close(_ai_cell(res, "old/M", "fuzzing")["mean"], 1.625 / 3, 1e-12)
        # The two DOCMAXes are different numbers and must not have been merged.
        assert (_ai_cell(res, "rl16/DOCMAX", "detection")["mean"]
                != _ai_cell(res, "old/DOCMAX", "detection")["mean"])
        # The floor arm is constant, so its CI is degenerate and the figure's line is at 0.5.
        fl = _ai_cell(res, "rl16/R-shuffled", "detection")
        assert (fl["mean"], fl["lo"], fl["hi"]) == (0.5, 0.5, 0.5), fl
        _close(A._floor_mean(res, "detection")[0], 0.5, 1e-12)
        # Paired differences against rl16/DOCMAX, each summed by hand above the fixture.
        _close(_ai_contrast(res, "rl16/M", "detection")["mean"], -0.375 / 4, 1e-12)
        _close(_ai_contrast(res, "rl16/M", "detection")["win_frac"], 0.25, 1e-12)
        _close(_ai_contrast(res, "rl16/R-shuffled", "detection")["mean"], -0.75 / 4, 1e-12)
        _close(_ai_contrast(res, "old/M", "detection")["mean"], -0.5 / 4, 1e-12)
        _close(_ai_contrast(res, "old/DOCMAX", "detection")["mean"], -0.25 / 4, 1e-12)
        _close(_ai_contrast(res, "old/M", "fuzzing")["mean"], 0.0, 1e-12)
        _close(_ai_contrast(res, "old/M", "fuzzing")["win_frac"], 1 / 3, 1e-12)
        # The reference contrasts nothing against itself.
        assert not [x for x in res["contrasts"] if x["arm"] == "rl16/DOCMAX"]
        # Per stratum: the mean and the n, no interval.
        st = {(s["arm"], s["scorer"], s["stratum"]): s for s in res["strata"]}
        _close(st[("rl16/DOCMAX", "detection", 0)]["mean"], 0.625, 1e-12)
        _close(st[("rl16/DOCMAX", "detection", 1)]["mean"], 0.75, 1e-12)
        assert st[("rl16/DOCMAX", "detection", 0)]["n"] == 2
        assert st[("rl16/NLA", "detection", 1)]["n"] == 1, st[("rl16/NLA", "detection", 1)]
        # `--no-strata` is the secondary SAE's setting and must produce no stratum rows at all.
        _vol2, plain = _ai_analyse(root, strata=False)
        assert plain["strata"] == []


def check_autointerp_pairing_is_intersection():
    """An arm that covers fewer features is REPORTED as a reduced pairing, and the contrast runs
    on the intersection alone.

    `rl16/NLA` has three of the four features. A contrast that took each side's own mean would get
    1.375/3 − 2.75/4 = −0.229; the paired one on features 10-12 is −0.5/3 = −0.167. The two differ,
    which is what the intersection is for, and the difference between them is exactly the error a
    silent de-pairing would introduce.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root)
        x = _ai_contrast(res, "rl16/NLA", "detection")
        assert (x["n_paired"], x["n_arm"], x["n_ref"]) == (3, 3, 4), x
        assert x["complete"] is False and x["lost_by_arm"] == [13], x
        _close(x["mean"], -0.5 / 3, 1e-12, what="paired on features 10-12 only")
        # NOT the difference of the two unpaired means -- that is the number this exists to avoid.
        assert abs(x["mean"] - (1.375 / 3 - 2.75 / 4)) > 0.05, x["mean"]
        # The same on fuzzing, where the reference's dropped feature carries a different value.
        _close(_ai_contrast(res, "rl16/NLA", "fuzzing")["mean"], -0.375 / 3, 1e-12)
        # The null-bal_acc row de-pairs `old/M` on fuzzing and not on detection.
        assert _ai_contrast(res, "old/M", "fuzzing")["n_paired"] == 3
        assert _ai_contrast(res, "old/M", "detection")["n_paired"] == 4
        # And all of it reaches the pairing CHECK, which is what the table and results.json print.
        pc = {(c["who"], c["detail"]): c for c in res["checks"] if c["kind"] == "pairing"}
        assert pc[("rl16/NLA", "detection")]["complete"] is False
        assert pc[("rl16/NLA", "detection")]["n_mismatches"] == 1, pc[("rl16/NLA", "detection")]
        assert pc[("rl16/M", "detection")]["complete"] is True
        assert pc[("old/M", "fuzzing")]["n_mismatches"] == 1
        assert pc[("old/M", "detection")]["complete"] is True


def check_autointerp_catches_a_defect():
    """Corrupt one stored `bal_acc`, and drop one feature from one arm, and require the reader
    check and the pairing check to go red. Without this both are unevaluated: a comparison that has
    only ever passed proves nothing about its ability to fail."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        p = root / f"runs/{AI_RUNS['rl16']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        hit = next(r for r in recs if r["arm"] == "DOCMAX" and r["scorer"] == "detection"
                   and r["feature"] == 10)
        hit["bal_acc"] = 0.9        # its own tpr and tnr still say 0.75
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _ai_analyse(root)
        chk = [c for c in res["checks"] if c["kind"] == "reader" and c["who"] == "rl16"][0]
        assert chk["n_mismatches"] == 1 and chk["n_identity"] == 1, chk
        _close(chk["worst_excess"], 0.15 - A.READER_TOL, 1e-9, what="|0.9 - 0.75| beyond tol")

    # A rate outside [0, 1] and a row with more parsed batches than batches are corrupt rows, not
    # findings, and the same check says so. `acc` is in neither identity, so this mutation isolates
    # the RANGE half from the identity half rather than tripping both at once.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        p = root / f"runs/{AI_RUNS['old']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        recs[0]["acc"] = 1.5
        recs[1]["n_parsed"] = recs[1]["n_batches"] + 1
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _ai_analyse(root)
        chk = [c for c in res["checks"] if c["kind"] == "reader" and c["who"] == "old"][0]
        assert (chk["n_identity"], chk["n_out_of_range"], chk["n_bad_counts"]) == (0, 1, 1), chk
        assert chk["n_mismatches"] == 2, chk

    # Drop a feature from an arm that was complete, and the pairing check must change verdict.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        p = root / f"runs/{AI_RUNS['rl16']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        kept = [r for r in recs
                if not (r["arm"] == "M" and r["scorer"] == "detection" and r["feature"] == 13)]
        assert len(kept) == len(recs) - 1
        p.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
        _vol, res = _ai_analyse(root)
        x = _ai_contrast(res, "rl16/M", "detection")
        assert (x["n_paired"], x["n_arm"], x["n_ref"]) == (3, 3, 4), x
        assert x["complete"] is False and x["lost_by_arm"] == [13], x
        # The remaining three differences are -0.25, 0 and +0.125; the dropped one was -0.25.
        _close(x["mean"], -0.125 / 3, 1e-12)
        pc = [c for c in res["checks"] if c["kind"] == "pairing" and c["who"] == "rl16/M"
              and c["detail"] == "detection"][0]
        assert pc["complete"] is False and pc["n_mismatches"] == 1, pc
        # Fuzzing, untouched, stays complete -- the defect is local and the check is not a blanket.
        assert _ai_contrast(res, "rl16/M", "fuzzing")["complete"] is True

    # Two run directories under ONE label would double-count every feature; that is refused.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        p = root / f"runs/{AI_RUNS['rl16']}/summary/scores.jsonl"
        p.write_text(p.read_text() + p.read_text())
        try:
            _ai_analyse(root)
        except AssertionError as e:
            assert "one row per triple" in str(e), e
        else:
            raise AssertionError("a duplicated scores.jsonl must be refused, not averaged")


AI_SANITY = """
checks:
  - name: rl16 DOCMAX detection mean
    arm: "rl16/DOCMAX"
    scorer: detection
    metric: bal_acc.mean
    expect: 0.6875
    tol: 0.0005
  - name: a gate that must flag
    arm: "rl16/DOCMAX"
    scorer: detection
    metric: bal_acc.mean
    expect: 0.9
    tol: 0.01
  - name: a gate on an arm this block does not have
    arm: "epo/E"
    scorer: detection
    metric: bal_acc.mean
    expect: 0.5
    tol: 0.05
  - name: a pair that is not the same statistic
    arm: "rl16/DOCMAX"
    scorer: fuzzing
    metric: bal_acc.mean
    expect: 0.1
    compare: false
"""


def check_autointerp_render_and_figures():
    """The whole render: tables.md, a CSV per table, figures as PDF AND PNG, the four sanity
    verdicts, and the rule that unequal N and a reduced pairing are both VISIBLE in the file."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_runs(root)
        _vol, res = _ai_analyse(root)
        out_dir = root / "out" / AI_SAE.replace("/", "_")
        y = root / "sanity_autointerp.yaml"
        y.write_text(AI_SANITY)
        sanity = A.run_sanity(res, y)
        assert [s["verdict"] for s in sanity] == ["pass", "FLAG", "absent", "no verdict"], sanity
        _close(sanity[0]["ours"], 0.6875, 1e-9)
        assert sanity[0]["n"] == 4, sanity[0]
        # An absent gates file is a NOTE, not an error: eval 2 has no recorded expectations yet.
        assert A.run_sanity(res, root / "nope.yaml") == []

        figs = A.make_figures(res, out_dir)
        o = R.Out(out_dir, "selftest — eval 2", ["- synthetic"])
        path = A.render(res, o, sanity, figs)
        md = path.read_text()

        for name in ("bal_acc", "contrasts", "strata", "support", "checks", "sanity"):
            assert (out_dir / f"{name}.csv").exists(), f"{name}.csv was not written"
        # The headline means are in the file.
        assert "0.6875" in md, "the rl16/DOCMAX detection mean is not in tables.md"
        assert "0.5938" in md, "the rl16/M detection mean is not in tables.md"
        # Detection and fuzzing are separate columns and the caption says why they are not pooled.
        bal = md.split("### Balanced accuracy")[1].split("###")[0]
        header = next(ln for ln in bal.splitlines() if ln.startswith("| arm"))
        assert "detection mean" in header and "fuzzing mean" in header, header
        assert "not pooled" in bal and "ZERO-SHOT" in bal, bal[:400]
        # Unequal N is shown rather than smoothed.
        support = (out_dir / "support.csv").read_text()
        assert any(ln.split(",")[8] == "4" for ln in support.splitlines()[1:]
                   if "rl16/NLA" in ln), support
        # A reduced pairing is labelled, counted, and carries its feature ids to the CSV. Pinned on
        # the ROW, not on the block and not on the file: a bare `"REDUCED" in md` is satisfied by
        # the checks table, and a per-block one by the contrasts table's own CAPTION, so both
        # passed against a pairing column hardcoded to "complete" -- the defect they were meant to
        # catch. Only the row can tell the two apart.
        contrast_block = md.split("### Paired contrasts")[1].split("###")[0]
        rows_md = [ln for ln in contrast_block.splitlines() if ln.startswith("| ")]
        nla = next(ln for ln in rows_md if ln.startswith("| NLA | rl16 | detection |"))
        assert nla.rstrip().endswith("| REDUCED |"), nla
        m16 = next(ln for ln in rows_md if ln.startswith("| M | rl16 | detection |"))
        assert m16.rstrip().endswith("| complete |"), m16
        checks_block = md.split("### Reader and pairing checks")[1].split("###")[0]
        assert "REDUCED: 1 feature(s) lost" in checks_block, checks_block
        contrasts = (out_dir / "contrasts.csv").read_text()
        assert any("rl16/NLA" in ln and ln.rstrip().endswith("13,") for ln in contrasts.splitlines()), \
            contrasts
        # Figures: one per scorer, both formats, and big enough to be a plot.
        assert sorted(figs) == ["bal_acc_detection", "bal_acc_fuzzing"], figs
        for f in figs:
            for ext in ("pdf", "png"):
                p = out_dir / "figures" / f"{f}.{ext}"
                assert p.exists() and p.stat().st_size > 1000, p


# --- eval 2: the CUT-VIEW fixture, 32 features on two ORTHOGONAL cuts ---------------------------
#
# A SECOND autointerp fixture rather than a bigger first one. The four-feature fixture above exists
# so every mean and every paired difference is a literal a reader can check in their head, and
# widening it to 32 would cost that for no gain; the cut views need 32 because `MIN_CELL` is 4 and a
# cell has to be able to be full (8), under-filled (6) and SHORT (2) in the same fixture.
#
# THE TWO CUTS ARE ORTHOGONAL BY CONSTRUCTION, which is the property the whole fixture is for: the
# 32 features are laid out so each rarity stratum holds exactly two features of each activation-
# magnitude quartile. An arm whose accuracy depends only on the stratum therefore has ZERO spread
# on the magnitude cut and vice versa, so a driver that computed one cut and printed it under both
# headings -- or that cut the magnitude quartiles on the stratum by accident -- cannot pass.
#
#   feature   200..207   208..215   216..223   224..231
#   stratum      0           1          2          3
#   corpus_peak  1 + 4*((feature - 200) % 8) + stratum, i.e. the ranks 1..32 dealt out so that
#                magnitude quartile = ((feature - 200) % 8) // 2, which is free of the stratum.
AI_CUT_RUNS = {"cut": "2026-09-22_autointerp-cuts"}
AI_CUT_BASE, AI_CUT_SET = "B", "S4"
AI_CUT_FEATS = list(range(200, 232))
# The magnitude cuts this lays out: `np.quantile(1..32, [.25, .5, .75])` = 1 + 31*[.25, .5, .75].
AI_CUT_QUARTILE_CUTS = [8.75, 16.5, 24.25]
# `PARTIAL` is the arm that makes a cell short: it carries only two of stratum 3's eight features,
# and both of them are magnitude quartile 0, so ONE arm exercises a SHORT cell on the rarity cut
# (n = 2 < MIN_CELL) and three UNDER-FILLED ones on the magnitude cut (n = 6 < the fullest 8).
AI_CUT_PARTIAL_KEPT = [f for f in AI_CUT_FEATS if (f - 200) // 8 != 3 or (f - 200) % 8 in (0, 1)]
# A stratification note in the layout `precompute` writes beside a set, so `draw_record` has a real
# record to quote -- including a `- command:` line that says `--stratified` and states no cut at
# all, which the reader must NOT lift into the caption.
AI_CUT_README = """# S4

- date: 2026-09-22 00:00:00Z
- command: `modal run precompute/modal_app.py --product draw --set S4 --stratified --seed 7`
- status: ok

## Notes

- strata: log10_gated_fires_16M from our 16M corpus scan
- STRATIFIED draw, seed 7: 8 features from each of the 4 quartiles of the ELIGIBLE POOL (40
  features), cuts on log10_gated_fires_16M at [1.0, 2.0, 3.0]
- gate 2.0; vecs are unit(W_enc[:, f]) in fp32 before the cast
"""


def _cut_stratum(feat: int) -> int:
    return (feat - AI_CUT_FEATS[0]) // 8


def _cut_peak(feat: int) -> float:
    return 1.0 + 4 * ((feat - AI_CUT_FEATS[0]) % 8) + _cut_stratum(feat)


def _cut_quartile(feat: int) -> int:
    return ((feat - AI_CUT_FEATS[0]) % 8) // 2


def _cut_bal(arm: str, feat: int) -> float:
    """The four arms, each a different shape on the two cuts, all dyadic so every mean is exact.

    `RARE` moves with the rarity stratum alone and `BIG` with the magnitude quartile alone: each is
    the other's null, and their cell means on the cut they do not depend on are all equal, which is
    the orthogonality above made into numbers. `FLAT` is constant. `PARTIAL` puts 1.0 on the two
    stratum-3 features it keeps and 0.5 everywhere else, so its SHORT cell is the EXTREME one:
    including it would give a spread of 0.5 and excluding it gives 0, and the trend table's number
    says which of the two happened.
    """
    if arm == "RARE":
        return 0.75 if _cut_stratum(feat) == 3 else 0.5
    if arm == "BIG":
        return 0.75 if _cut_quartile(feat) == 3 else 0.5
    if arm == "PARTIAL":
        return 1.0 if _cut_stratum(feat) == 3 else 0.5
    return 0.5


def _cut_nopos(feat: int) -> bool:
    """Is this one of `NOPOS`'s no-positives features? The whole of magnitude quartile 0.

    THE SHAPE THIS EXISTS FOR IS REAL AND IS A TRAP. `run.rates` divides by the positive count, so
    a feature with no gate-consistent positives (`n_pos: 0`) gets TPR = NaN and `run._nr` stores
    `bal_acc: null` and `tpr: null` -- while `tnr` and `acc`, which are computed over the negative
    half alone, come back POPULATED AND HIGH. On the 131k block features 59176 (every arm) and
    124524 (`DOCMAX-draw2`) carry exactly this, with `acc` at 0.90 and 0.95. A reader that filled
    a null `bal_acc` from the `acc` sitting beside it would not degrade gracefully: it would
    manufacture a near-perfect cell out of a feature that was never measured. So `NOPOS` puts 0.9
    in `acc` and `tnr` on the nulled rows, and the checks below pin the cell to the value the
    surviving features give and its n to the count that survived.
    """
    return _cut_quartile(feat) == 0


# A SECOND run directory holding the SAME calls replayed, which is what eval 2 actually did: its
# three runs per SAE were launched with one shared `--cache-dir` and `run.py`'s cache key carries no
# run component, so every corpus arm's rows are byte-identical across the three. `RARE` is given
# different numbers in the replay so the fixture has both regimes at once -- four arms that are one
# measurement printed twice, and one that is genuinely two.
AI_CUT_REPLAY = "2026-09-22_autointerp-cuts-replay"
AI_CUT_REPLAY_DIFFERS = "RARE"


def write_autointerp_cuts(root: Path, replay: bool = False) -> None:
    """The cut fixture's mirror: one run directory, the set's draw record, and optionally a replay.

    `replay=True` adds a second run directory whose rows are identical to the first's for every arm
    but `RARE` -- the shape a shared `--cache-dir` produces -- so the trend table's multiplicity
    divisor has something to deduplicate and something it must NOT deduplicate.
    """
    run_dir = AI_CUT_RUNS["cut"]
    d = root / f"runs/{run_dir}/summary"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ("RARE", "BIG", "FLAT", "PARTIAL", "NOPOS"):
        feats = AI_CUT_PARTIAL_KEPT if arm == "PARTIAL" else AI_CUT_FEATS
        for scorer in ("detection", "fuzzing"):
            for feat in feats:
                bal = _cut_bal(arm, feat)
                # No positives: no TPR, so no balanced accuracy -- but a healthy-looking `acc`
                # and `tnr` over the negative half, which is the trap `_cut_nopos` documents.
                nopos = arm == "NOPOS" and _cut_nopos(feat)
                rows.append({
                    "feature": feat, "arm": arm, "scorer": scorer,
                    "bal_acc": None if nopos else bal, "tpr": None if nopos else bal,
                    "tnr": 0.9 if nopos else bal, "acc": 0.9 if nopos else bal,
                    "n_items": 20 if nopos else 40, "n_batches": 4, "n_parsed": 4,
                    "n_pos": 0 if nopos else 20,
                    "draw": 1, "role": "arm", "n_examples": 16, "explanation_ok": True,
                    "stratum": _cut_stratum(feat), "corpus_peak": _cut_peak(feat),
                })
    with open(d / "scores.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (d / "build.json").write_text(json.dumps(
        {"base": AI_CUT_BASE, "set": AI_CUT_SET, "sae": AI_SAE, "n_features": len(AI_CUT_FEATS)}))
    rec = root / f"base/{AI_CUT_BASE}/heldout/{AI_CUT_SET}"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "README.md").write_text(AI_CUT_README)
    if not replay:
        return
    d2 = root / f"runs/{AI_CUT_REPLAY}/summary"
    d2.mkdir(parents=True, exist_ok=True)
    with open(d2 / "scores.jsonl", "w") as fh:
        for r in rows:
            r = dict(r)
            if r["arm"] == AI_CUT_REPLAY_DIFFERS and r["bal_acc"] is not None:
                # Genuinely re-run rather than replayed: a different number, so this arm is two
                # measurements and the divisor must count it twice.
                r["bal_acc"] = r["tpr"] = r["tnr"] = r["acc"] = 0.75 - r["bal_acc"] / 2
            fh.write(json.dumps(r) + "\n")
    (d2 / "build.json").write_text(json.dumps(
        {"base": AI_CUT_BASE, "set": AI_CUT_SET, "sae": AI_SAE, "n_features": len(AI_CUT_FEATS)}))


def _cut_analyse(root: Path, **kw):
    vol = R.Vol("", root, offline=True, quiet=True)
    opts = {"runs": dict(AI_CUT_RUNS), "sae": AI_SAE, "ref": "FLAT", "boot": 2000, "seed": 1,
            "strata": True, "peak_strata": True}
    opts.update(kw)
    return vol, A.analyse(vol, opts["runs"], opts["sae"], opts["ref"], opts["boot"], opts["seed"],
                          opts["strata"], opts["peak_strata"])


def _cut(res, key: str, arm: str, scorer: str, bucket) -> dict:
    hits = [s for s in res[key] if s["arm"] == arm and s["scorer"] == scorer
            and s["stratum"] == bucket]
    assert len(hits) == 1, (key, arm, scorer, bucket, len(hits))
    return hits[0]


def _trend(res, view: str, arm: str, scorer: str) -> dict:
    hits = [t for t in res["trends"] if t["view"] == view and t["arm"] == arm
            and t["scorer"] == scorer]
    assert len(hits) == 1, (view, arm, scorer, len(hits))
    return hits[0]


def check_autointerp_cut_views():
    """`quartile_buckets` on values that sit ON a cut, then both cut tables against hand means.

    THE TIE RULE IS PINNED HERE AND NOWHERE ELSE. The 32 analysed features' peaks fall between the
    cuts, so on them `side="right"` and `side="left"` are the same function and the caption's "a
    value exactly on a cut falls in the higher quartile" would be an untested sentence. Four values
    in two clumps put a value on two of the three cuts and tell the two apart.

    The point of the two assertions per arm is the ORTHOGONALITY: `RARE`'s rarity cells are
    0.5/0.5/0.5/0.75 and its magnitude cells are all 0.5625, because each magnitude quartile holds
    exactly two of the eight stratum-3 features. An implementation that cut the magnitude quartiles
    on anything correlated with the stratum would move those four numbers apart.
    """
    # `np.quantile([0, 0, 4, 4], [.25, .5, .75])` = [0, 2, 4]: the 0s sit ON the first cut and the
    # 4s ON the third, so both go UP a bucket. Ties also collapse the buckets -- two of the four
    # are empty -- which is the reason every cell prints its own n rather than the design's.
    buckets, cuts = A.quartile_buckets({10: 0.0, 11: 0.0, 12: 4.0, 13: 4.0})
    assert cuts == [0.0, 2.0, 4.0], cuts
    assert buckets == {10: 1, 11: 1, 12: 3, 13: 3}, buckets
    # Fewer than four values cannot be quartiled at all, and that is said rather than approximated.
    assert A.quartile_buckets({1: 0.5, 2: 1.5, 3: 2.5}) == ({}, [])

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        _vol, res = _cut_analyse(root)
        # The magnitude cuts are the analysed features' own quartiles: `np.quantile(1..32, ...)`.
        assert res["peak_cuts"] == AI_CUT_QUARTILE_CUTS, res["peak_cuts"]
        assert res["n_features"] == 32, res["n_features"]
        for scorer in ("detection", "fuzzing"):
            # RARE: all of the signal on the rarity cut, none of it on the magnitude cut.
            for s in range(3):
                _close(_cut(res, "strata", "cut/RARE", scorer, s)["mean"], 0.5, 1e-12)
            _close(_cut(res, "strata", "cut/RARE", scorer, 3)["mean"], 0.75, 1e-12)
            for q in range(4):
                c = _cut(res, "peak_strata", "cut/RARE", scorer, q)
                _close(c["mean"], 0.5625, 1e-12, what="(6*0.5 + 2*0.75) / 8")
                assert c["n"] == 8 and c["short"] is False, c
            # BIG: the mirror image, which is what makes the two cuts independent here.
            for q in range(3):
                _close(_cut(res, "peak_strata", "cut/BIG", scorer, q)["mean"], 0.5, 1e-12)
            _close(_cut(res, "peak_strata", "cut/BIG", scorer, 3)["mean"], 0.75, 1e-12)
            for s in range(4):
                _close(_cut(res, "strata", "cut/BIG", scorer, s)["mean"], 0.5625, 1e-12)
            # FLAT is constant, so every cell is a degenerate bootstrap and not a missing one.
            c = _cut(res, "strata", "cut/FLAT", scorer, 0)
            assert (c["mean"], c["lo"], c["hi"], c["n"]) == (0.5, 0.5, 0.5, 8), c
        # Every cell carries the SAME percentile bootstrap the headline table uses, so a cell and
        # a whole-arm number are one estimator at two sample sizes.
        c = _cut(res, "strata", "cut/RARE", "detection", 3)
        assert c["lo"] == c["hi"] == 0.75, c
        # Both views reach the registry under DIFFERENT metric names: stratum 3 of one cut and
        # stratum 3 of the other are different cells and a gate must be able to name which.
        reg = A.stat_registry(res)
        _close(reg[("detection", "cut/RARE", 3, "bal_acc.mean")], 0.75, 1e-12)
        _close(reg[("detection", "cut/RARE", 3, "peak.bal_acc.mean")], 0.5625, 1e-12)
        # And the caption's cut definition is QUOTED from the set's record, not retyped: the two
        # `## Notes` lines that state a stratification, and not the `- command:` line above them
        # that merely says `--stratified`.
        assert len(res["record"]) == 2, res["record"]
        assert all("modal run" not in ln for ln in res["record"]), res["record"]
        assert any("[1.0, 2.0, 3.0]" in ln and AI_CUT_SET in ln for ln in res["record"]), \
            res["record"]

    # No record on the volume costs the caption its numbers and SAYS which set's was missing --
    # it never falls back to a plausible default.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        (root / f"base/{AI_CUT_BASE}/heldout/{AI_CUT_SET}/README.md").unlink()
        _vol, res = _cut_analyse(root)
        assert res["record"] == [], res["record"]
        assert any(AI_CUT_SET in n and "unstated" in n for n in res["notes"]), res["notes"]


def check_autointerp_short_cells_are_reported():
    """A cell with too few features is REPORTED, and is not silently averaged into the trend.

    `PARTIAL` carries two of stratum 3's eight features and both are worth 1.0 against 0.5
    everywhere else, so the short cell is the extreme one: a driver that averaged it in would
    report a spread of 0.5, and one that dropped it without saying so would report 0.0 with no
    trace. The correct answer is 0.0 WITH the cell named, and this pins all three apart.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        _vol, res = _cut_analyse(root)
        # The cell exists, prints its own n, is flagged, and keeps its mean -- reported, not
        # deleted and not smoothed.
        short = _cut(res, "strata", "cut/PARTIAL", "detection", 3)
        assert (short["n"], short["short"]) == (2, True), short
        _close(short["mean"], 1.0, 1e-12)
        for s in range(3):
            c = _cut(res, "strata", "cut/PARTIAL", "detection", s)
            assert (c["n"], c["short"]) == (8, False), c
        # The trend row used three cells of four, named the fourth, and reports the spread of the
        # three it used -- 0.0, not the 0.5 that averaging the short cell in would give.
        t = _trend(res, "stratum", "cut/PARTIAL", "detection")
        assert (t["n_cells"], t["n_cells_all"]) == (3, 4), t
        assert t["dropped"] == ["3 (n=2)"], t
        _close(t["spread"], 0.0, 1e-12, what="the three full cells are all 0.5")
        # On the magnitude cut the same arm loses two features out of three quartiles: those cells
        # are UNDER-FILLED, not short, so they stay in the statistics and are still reported.
        assert [_cut(res, "peak_strata", "cut/PARTIAL", "detection", q)["n"] for q in range(4)] \
            == [8, 6, 6, 6]
        tq = _trend(res, "peak", "cut/PARTIAL", "detection")
        assert (tq["n_cells"], tq["dropped"], tq["n_min"]) == (4, [], 6), tq
        _close(tq["spread"], 0.125, 1e-12, what="(6*0.5 + 2*1.0)/8 − 0.5")
        # Both counters reach the checks table, per view, with the cells named.
        ck = {c["who"]: c for c in res["checks"] if c["kind"] == "cells"}
        # NOPOS's eight rarity cells are under-filled (6 of 8) without being short, which is what
        # separates the two counters; PARTIAL's two stratum-3 cells are the short ones.
        assert (ck["stratum"]["n_short"], ck["stratum"]["n_thin"]) == (2, 8), ck["stratum"]
        assert ck["stratum"]["short_cells"] == ["cut/PARTIAL/detection stratum 3 (n=2)",
                                                "cut/PARTIAL/fuzzing stratum 3 (n=2)"], ck
        assert (ck["peak"]["n_short"], ck["peak"]["n_thin"]) == (0, 6), ck["peak"]
        assert ck["peak"]["full"] == 8, ck["peak"]
        # The magnitude view names its cells QUARTILES here too: a check row that called one
        # "stratum 1" would read as a rarity stratum in the one table whose caption is that it is
        # not one, and the check table is where a reader looks when a cell is short.
        assert all(" quartile " in c for c in ck["peak"]["thin_cells"]), ck["peak"]["thin_cells"]
        assert all(" stratum " in c for c in ck["stratum"]["short_cells"]), ck["stratum"]


def check_autointerp_null_bal_acc_is_never_filled_from_acc():
    """A feature with no positives is DROPPED from its cell, never filled from the `acc` beside it.

    This is the one failure mode in the cut tables that would not look like a defect. Every other
    way of losing a feature makes a cell smaller, which the n shows; this one offers a populated,
    plausible, HIGH number in the next column over (`acc` = 0.9 on the real 131k rows, `tnr` = 0.95
    on one of them) and a reader that reached for it would print a near-perfect cell for a feature
    whose balanced accuracy was never computable. The assertions below separate the three possible
    behaviours: the correct one (n drops, mean unchanged), imputing chance (n stays 8) and reading
    `acc` (mean moves toward 0.9).
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        _vol, res = _cut_analyse(root)
        for scorer in ("detection", "fuzzing"):
            # Every rarity stratum loses its two magnitude-quartile-0 features: n = 6, and the
            # mean is the surviving 0.5s. 0.6 would be `acc` averaged in; 8 would be imputation.
            for stratum in range(4):
                c = _cut(res, "strata", "cut/NOPOS", scorer, stratum)
                assert c["n"] == 6, (c, "a dropped feature must shrink the cell, not be imputed")
                _close(c["mean"], 0.5, 1e-12, what="the six features that WERE measured")
            # On the magnitude cut the whole of quartile 0 is unmeasurable, so that cell does not
            # exist -- it renders as an em dash and is not a 0.9, a 0.5 or a zero.
            assert not [c for c in res["peak_strata"] if c["arm"] == "cut/NOPOS"
                        and c["scorer"] == scorer and c["stratum"] == 0]
            for q in (1, 2, 3):
                c = _cut(res, "peak_strata", "cut/NOPOS", scorer, q)
                assert (c["n"], c["mean"]) == (8, 0.5), c
        # The headline cell counts the loss rather than hiding it: 32 rows, 24 usable.
        head = _ai_cell(res, "cut/NOPOS", "detection")
        assert (head["n_rows"], head["n_features"], head["n_no_metric"]) == (32, 24, 8), head
        # The reader check is not tripped by any of this: a row with no `tpr` has no identity to
        # compare, and `tnr`/`acc` of 0.9 are in range. An unmeasurable feature is not a defect.
        for chk in [c for c in res["checks"] if c["kind"] == "reader"]:
            assert chk["n_mismatches"] == 0, chk
        # A cut cell that is ABSENT for one arm and present for its neighbours is reported: the
        # trend row for NOPOS is computed on three cells where every other arm has four.
        t = _trend(res, "peak", "cut/NOPOS", "detection")
        assert (t["n_cells"], t["n_cells_all"]) == (3, 3), t
        ck = {c["who"]: c for c in res["checks"] if c["kind"] == "cells"}
        assert ck["peak"]["n_absent"] == 2, ck["peak"]
        assert ck["peak"]["absent_cells"] == ["cut/NOPOS/detection quartile 0",
                                              "cut/NOPOS/fuzzing quartile 0"], ck["peak"]
        assert ck["stratum"]["n_absent"] == 0, ck["stratum"]

    # `density` is NOT the field either cut reads. It is null for every feature of a `draw_sae2m`
    # set -- the 2M block carries `fire_fraction` and `corpus_peak` and nothing else -- so a
    # magnitude split that reached for it would come back empty on the PRIMARY SAE and quietly
    # build its quartiles out of a single bucket. Pinned on the field map rather than on the
    # file's text, because the caption names `density` in order to say it is not used.
    assert A.VIEW_FIELD == {A.STRATUM_VIEW: "stratum", A.PEAK_VIEW: "corpus_peak"}, A.VIEW_FIELD
    # And the 2M shape end to end: rows whose `density` is null everywhere still cut both views.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        p = root / f"runs/{AI_CUT_RUNS['cut']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        for r in recs:
            r["density"] = None
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _cut_analyse(root)
        assert res["peak_cuts"] == AI_CUT_QUARTILE_CUTS, res["peak_cuts"]
        _close(_cut(res, "strata", "cut/RARE", "detection", 3)["mean"], 0.75, 1e-12)


def check_autointerp_trend_statistics():
    """The spread, the rank statistic and the permutation p, against numbers worked out here.

    THE PERMUTATION IS EXACT ENOUGH TO BE A LITERAL. `RARE` has eight values of 0.75 among 32, so a
    cell of eight has mean 0.5 + 0.25 * k / 8 for its k high values and the spread is
    0.25 * (k_max − k_min) / 8. The observed 0.25 is therefore the LARGEST spread the multiset can
    produce and needs all eight in one cell: 4 / C(32, 8) = 4 / 10,518,300 ≈ 3.8e-7 of relabellings.
    At 2000 resamples no relabelling reaches it, so the reported p is the add-one FLOOR, 1/2001 --
    which is also the check that the floor is reported as a floor and not as a zero.

    The orthogonal direction is the other exact case: `RARE` on the magnitude cut has spread 0, and
    EVERY relabelling has spread >= 0, so its p is exactly 1.0 and not a small number.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        _vol, res = _cut_analyse(root)
        t = _trend(res, "stratum", "cut/RARE", "detection")
        _close(t["spread"], 0.25, 1e-12, what="0.75 − 0.5")
        _close(t["p_perm"], 1 / 2001, 1e-12, what="no resample of 2000 reaches the largest spread")
        # Spearman over the four cells: the cut index ranks 1,2,3,4 against the means' average
        # ranks 2,2,2,4 -- three tied cells and one above them -- giving 3 / sqrt(5 * 3).
        _close(t["rho"], 3 / math.sqrt(15), 1e-12, what="ties shared, not broken by position")
        assert (t["n_cells"], t["n_min"]) == (4, 8), t
        # The same arm on the cut it does not depend on: no spread, no rank, and p = 1 exactly.
        tq = _trend(res, "peak", "cut/RARE", "detection")
        _close(tq["spread"], 0.0, 1e-12)
        assert tq["p_perm"] == 1.0, tq
        assert math.isnan(tq["rho"]), tq
        # BIG is the mirror image, which is the check that the two views are not one computation
        # printed twice.
        _close(_trend(res, "peak", "cut/BIG", "detection")["spread"], 0.25, 1e-12)
        _close(_trend(res, "stratum", "cut/BIG", "detection")["spread"], 0.0, 1e-12)
        # A flat arm has no spread, no rank and no separation -- never a small p.
        f = _trend(res, "stratum", "cut/FLAT", "fuzzing")
        assert f["spread"] == 0.0 and f["p_perm"] == 1.0 and math.isnan(f["rho"]), f
        # The widest per-cell interval is reported: at n = 8 with 6 values at 0.5 and 2 at 1.0 the
        # bootstrap is wide, and the trend table prints it so a spread can be read against it.
        assert _trend(res, "peak", "cut/PARTIAL", "detection")["widest_ci"] > 0.2

    # `rho` is a rank statistic, not a Pearson correlation on the means: an arm whose cells rise by
    # wildly unequal steps has |rho| = 1 all the same, and a driver that had used Pearson would not.
    ranks = A._avg_ranks(np.array([0.5, 0.51, 0.52, 0.99]))
    assert list(ranks) == [1.0, 2.0, 3.0, 4.0], ranks
    assert list(A._avg_ranks(np.array([0.5, 0.5, 0.5, 0.9]))) == [2.0, 2.0, 2.0, 4.0]
    # A degenerate permutation input is NaN, never a p of 0 or 1 invented from one cell.
    assert math.isnan(A.spread_perm_p(np.array([0.5, 0.75]), np.array([0, 0]), 100, 1))


def check_autointerp_top_cell_separation():
    """Whether the top cell's interval clears every other cell's -- the question `CI-WIDE` does not
    answer, at every boundary, and in BOTH directions against `CI-WIDE`.

    The two flags are independent, and this eval's own data has both crossings: a row that is
    CI-WIDE and disjoint (a wide interval sitting entirely above its neighbours) and a row that is
    not CI-WIDE and yet overlaps. A reader who took either for the other would be wrong about the
    2M headline row, so the cross cases are pinned here rather than left to the real data, which
    can change under a re-run.
    """
    def cell(b, mean, lo, hi):
        return {"stratum": b, "mean": mean, "lo": lo, "hi": hi, "n": 8, "short": False}

    # Disjoint: the top's LOWER bound clears every other UPPER bound.
    r = A.top_cell_separation([cell(0, 0.50, 0.45, 0.55), cell(1, 0.52, 0.48, 0.56),
                               cell(2, 0.51, 0.47, 0.55), cell(3, 0.80, 0.70, 0.90)])
    assert (r["disjoint"], r["top"], r["overlaps"]) == (True, 3, []), r
    # One neighbour reaching INTO the top interval spoils it, and is named.
    r = A.top_cell_separation([cell(0, 0.50, 0.45, 0.55), cell(1, 0.60, 0.50, 0.72),
                               cell(2, 0.51, 0.47, 0.55), cell(3, 0.80, 0.70, 0.90)])
    assert (r["disjoint"], r["top"], r["overlaps"]) == (False, 3, [1]), r
    # Touching exactly is NOT disjoint: the rule is a strict `>`, so a shared endpoint overlaps.
    r = A.top_cell_separation([cell(0, 0.50, 0.45, 0.70), cell(3, 0.80, 0.70, 0.90)])
    assert (r["disjoint"], r["overlaps"]) == (False, [0]), r
    r = A.top_cell_separation([cell(0, 0.50, 0.45, 0.6999), cell(3, 0.80, 0.70, 0.90)])
    assert r["disjoint"] is True, r
    # Two cells tied at the top are two cells: the second one overlaps the first by definition,
    # and a `max` that collapsed them by value would report a disjoint row.
    r = A.top_cell_separation([cell(0, 0.50, 0.45, 0.55), cell(3, 0.80, 0.70, 0.90),
                               cell(2, 0.80, 0.70, 0.90)])
    assert (r["disjoint"], r["overlaps"]) == (False, [2]), r
    # UNDECIDABLE, never optimistic. A cell of one feature has no interval, and NaN compares false
    # against everything -- which would read as DISJOINT if it were not caught.
    r = A.top_cell_separation([cell(0, 0.50, float("nan"), float("nan")),
                               cell(3, 0.80, 0.70, 0.90)])
    assert r["disjoint"] is None and "no estimable interval" in r["why"], r
    r = A.top_cell_separation([cell(3, 0.80, 0.70, 0.90)])
    assert r["disjoint"] is None and "nothing to be disjoint from" in r["why"], r
    # Cells the trend statistics excluded are not here to be compared against.
    r = A.top_cell_separation([cell(None, 0.99, 0.98, 1.0), cell(0, 0.50, 0.45, 0.55),
                               cell(3, 0.80, 0.70, 0.90)])
    assert (r["disjoint"], r["top"]) == (True, 3), r

    # THE TWO CROSSINGS, in the verdict string. Neither flag may be read off the other.
    wide_disjoint = A.top_cell_separation([cell(0, 0.50, 0.49, 0.51), cell(3, 0.62, 0.54, 0.72)])
    assert wide_disjoint["disjoint"] is True
    assert A.trend_verdict(0.001, 0.003, 16, 0.18, 0.1375, [], wide_disjoint) == \
        "separates (p ≤ α/16), CI-WIDE, DISJOINT"
    narrow_overlap = A.top_cell_separation([cell(1, 0.576, 0.5406, 0.6198),
                                            cell(3, 0.7198, 0.6177, 0.8240)])
    assert narrow_overlap["disjoint"] is False
    assert A.trend_verdict(0.0001, 0.003, 16, 0.206, 0.2323, [], narrow_overlap) == \
        "separates (p ≤ α/16), OVERLAP: 1"
    # An undecidable separation prints NEITHER word rather than guessing one.
    assert A.trend_verdict(0.0001, 0.003, 16, 0.10, 0.25, [],
                           {"disjoint": None, "top": None, "overlaps": [], "why": "x"}) == \
        "separates (p ≤ α/16)"

    # THE PLACEMENT: the qualifier rides on EVERY verdict, not only `separates`. This is where it
    # earns its place -- a row whose top cell overlaps all three neighbours says "nothing here"
    # more plainly than its p, and a row that fails the multiplicity correction while its cells ARE
    # resolved is a tension the verdict column should show rather than bury in the CSV.
    overlap_all = A.top_cell_separation([cell(0, 0.50, 0.45, 0.62), cell(1, 0.52, 0.46, 0.63),
                                         cell(2, 0.51, 0.45, 0.62), cell(3, 0.59, 0.52, 0.66)])
    assert (overlap_all["disjoint"], overlap_all["overlaps"]) == (False, [0, 1, 2]), overlap_all
    assert A.trend_verdict(0.0886, 0.003, 16, 0.15, 0.1146, [], overlap_all) == \
        "no separation, OVERLAP: 0, 1, 2"
    assert A.trend_verdict(0.014, 0.003, 16, 0.12, 0.12, [], wide_disjoint) == \
        "uncorrected only, DISJOINT"
    # `CI-WIDE` does NOT spread with it: it exists to stop a small p being read as precision, and
    # a row with no small p has nothing for it to qualify. Both of these have widest_ci >= spread.
    assert "CI-WIDE" not in A.trend_verdict(0.014, 0.003, 16, 0.30, 0.12, [], wide_disjoint)
    assert "CI-WIDE" not in A.trend_verdict(0.9, 0.003, 16, 0.30, 0.12, [], overlap_all)
    # Order is fixed: the p's verdict, then CI-WIDE, then the resolution, then the dropped cells.
    assert A.trend_verdict(0.0001, 0.003, 16, 0.30, 0.25, ["3 (n=2)"], overlap_all) == \
        "separates (p ≤ α/16), CI-WIDE, OVERLAP: 0, 1, 2 — 1 cell(s) dropped: 3 (n=2)"
    # The two really are independent: a non-estimable p does not suppress a resolution that IS
    # computable. The pair cannot arise from real cells -- both go undecidable below two usable
    # cells -- but coupling them here would reintroduce exactly the dependence this flag exists to
    # break, so the function is pinned to keep them apart.
    assert A.trend_verdict(float("nan"), 0.003, 16, 0.10, 0.25, [], wide_disjoint) == \
        "not estimable, DISJOINT"
    # And the qualifier rides along with the dropped-cell note in the documented order.
    assert A.trend_verdict(0.0001, 0.003, 16, 0.10, 0.25, ["3 (n=2)"], wide_disjoint) == \
        "separates (p ≤ α/16), DISJOINT — 1 cell(s) dropped: 3 (n=2)"


def check_autointerp_trend_verdict():
    """Every branch of the verdict cell, at its boundaries, including `CI-WIDE`.

    This is what a reader of the table concludes, so each branch is pinned to a literal rather than
    to another call of the code. The boundaries matter in both directions: the corrected alpha and
    the CI rule are both `<=`/`>=` comparisons, and an off-by-one on either silently reclassifies
    the rows nearest the line -- which on the real 2M block is where three of the seven separating
    measurements sit.

    `CI-WIDE` is APPENDED to `separates`, never a replacement for it. It is a caution that few
    features are carrying the difference, not a second significance gate: a wide interval lying
    entirely above its neighbours still separates them (the real 2M peak `M` detection row), so a
    verdict that swallowed `separates` when the flag fired would be making a claim the statistic
    does not support -- in the opposite direction from the one the flag exists to prevent.
    """
    # p below the corrected alpha, cells estimated more precisely than the gap between them.
    assert A.trend_verdict(0.0001, 0.003125, 16, 0.10, 0.25, []) == "separates (p ≤ α/16)"
    # The same, with the cells individually wider than the difference claimed between them.
    assert A.trend_verdict(0.0001, 0.003125, 16, 0.30, 0.25, []) == \
        "separates (p ≤ α/16), CI-WIDE"
    # Both comparisons are inclusive, and the two boundaries are independent of each other.
    assert A.trend_verdict(0.003125, 0.003125, 16, 0.10, 0.25, []) == "separates (p ≤ α/16)"
    assert A.trend_verdict(0.0001, 0.003125, 16, 0.25, 0.25, []) == \
        "separates (p ≤ α/16), CI-WIDE"
    # Just past the corrected alpha is NOT `separates`, and the CI rule does not apply there --
    # a row that fails the multiplicity correction is not additionally accused of being imprecise.
    assert A.trend_verdict(0.0032, 0.003125, 16, 0.30, 0.25, []) == "uncorrected only"
    assert A.trend_verdict(0.05, 0.003125, 16, 0.30, 0.25, []) == "uncorrected only"
    assert A.trend_verdict(0.0501, 0.003125, 16, 0.30, 0.25, []) == "no separation"
    # A row with no estimable p says so rather than defaulting to either outcome.
    assert A.trend_verdict(float("nan"), 0.003125, 16, 0.30, 0.25, []) == "not estimable"
    # A non-finite width or spread cannot fire the flag: an unknown is not a caution.
    assert A.trend_verdict(0.0001, 0.003125, 16, float("nan"), 0.25, []) == \
        "separates (p ≤ α/16)"
    # Dropped cells are named on EVERY branch, because a verdict computed on two of four cells is
    # a different claim from one computed on four whatever the verdict says.
    assert A.trend_verdict(0.0001, 0.003125, 16, 0.30, 0.25, ["3 (n=2)"]) == \
        "separates (p ≤ α/16), CI-WIDE — 1 cell(s) dropped: 3 (n=2)"
    assert A.trend_verdict(0.9, 0.003125, 16, 0.10, 0.25, ["0 (n=1)", "3 (n=2)"]) == \
        "no separation — 2 cell(s) dropped: 0 (n=1), 3 (n=2)"

    # And the real shape the flag was added for, reproduced from the fixture rather than asserted
    # in the abstract: the divisor counts DISTINCT measurements, so a cache-replayed arm under
    # three run labels is one look. The cut fixture has one run, so its count is its arm names.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        _vol, res = _cut_analyse(root)
        o = R.Out(root / "out", "selftest — verdicts", ["- synthetic"])
        md = A.render(res, o, [], []).read_text()
        tre = md.split("### Does any arm × cut trend survive?")[1].split("###")[0]
        # Five arm names x TWO scorers, and a view holds both, so ten distinct measurements --
        # one run directory, so nothing here is a cache replay and nothing deduplicates.
        assert "α/10" in tre, tre[:400]
        rare = next(ln for ln in tre.splitlines() if ln.startswith("| stratum | RARE | cut | det"))
        # RARE's cells are each constant, so every interval is degenerate at its own value and
        # the top one (0.75) clears the others (0.5) outright.
        assert rare.rstrip().endswith("| separates (p ≤ α/10), DISJOINT |"), rare
        # RARE's cells are each constant, so their intervals are degenerate and the flag is off --
        # which is the case that proves the flag is computed and not simply always appended.
        assert "CI-WIDE" not in rare, rare

    # THE DIVISOR COUNTS MEASUREMENTS, NOT ROWS. With the replay directory there are twice as many
    # printed rows, but four of the five arms are the same calls replayed from one cache and are
    # identical to the last digit; only `RARE` was genuinely re-run. So 20 rows per view carry 12
    # measurements (4 shared + 2 of RARE, times 2 scorers), and the correction must not treat a
    # repeated printout as an extra look.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root, replay=True)
        _vol, res = _cut_analyse(root, runs={**AI_CUT_RUNS, "replay": AI_CUT_REPLAY})
        stratum = [t for t in res["trends"] if t["view"] == "stratum"]
        assert len(stratum) == 20, len(stratum)
        o = R.Out(root / "out", "selftest — replay", ["- synthetic"])
        md = A.render(res, o, [], []).read_text()
        tre = md.split("### Does any arm × cut trend survive?")[1].split("###")[0]
        assert "α/12" in tre, [ln for ln in tre.splitlines() if "α/" in ln][:3]
        assert "α/20" not in tre, "the divisor counted printed rows, not distinct measurements"
        # The replayed arm's two rows are identical; the re-run arm's two are not.
        by = {(t["arm"], t["scorer"]): t for t in stratum}
        assert by[("cut/BIG", "detection")]["_means"] == by[("replay/BIG", "detection")]["_means"]
        assert by[("cut/RARE", "detection")]["_means"] != by[("replay/RARE", "detection")]["_means"]


def check_autointerp_cut_render():
    """Both cut tables and the trend table reach `tables.md` with a CSV each, and the captions
    carry the two statements the tables cannot make for themselves: that the rarity cut was
    BALANCED BY THE DRAW and the magnitude cut was not, and that the rank statistic is not a test.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        _vol, res = _cut_analyse(root)
        out_dir = root / "out"
        o = R.Out(out_dir, "selftest — eval 2 cuts", ["- synthetic"])
        md = A.render(res, o, [], []).read_text()
        for name in ("strata", "peak_strata", "trends"):
            assert (out_dir / f"{name}.csv").exists(), f"{name}.csv was not written"
        rar = md.split("### Balanced accuracy per rarity stratum")[1].split("###")[0]
        mag = md.split("### Balanced accuracy per activation-magnitude quartile")[1].split("###")[0]
        tre = md.split("### Does any arm × cut trend survive?")[1].split("###")[0]
        # The rarity caption states the axis, quotes the record, and warns the two SAEs' strata
        # apart; the magnitude caption states that it is POST-HOC and prints its own cuts.
        assert "RARITY" in rar and "not the same rarity" in rar, rar[:400]
        assert "log10_gated_fires_16M" in rar, "the draw record is not quoted in the caption"
        assert "POST-HOC" in mag and "8.7500, 16.5000, 24.2500" in mag, mag[:600]
        assert "not a stratification the draw controlled" in mag, mag[:600]
        # Both tables carry BOTH scorers and the n of every cell -- pinned on the ROW, because the
        # caption also contains the word and a bare `in md` would pass on the caption alone.
        rows_md = [ln for ln in rar.splitlines() if ln.startswith("| ")]
        assert any(ln.startswith("| RARE | cut | fuzzing |") for ln in rows_md), rows_md
        rare = next(ln for ln in rows_md if ln.startswith("| RARE | cut | detection |"))
        assert rare.rstrip().endswith("| 0.7500 (n=8) |"), rare
        part = next(ln for ln in rows_md if ln.startswith("| PARTIAL | cut | detection |"))
        assert part.rstrip().endswith("| 1.0000 (n=2, SHORT) |"), part
        # The trend table's verdict, and the caption's refusal to sell the rank statistic as one.
        assert "never a test" in tre and "2/24 = 0.083" in tre, tre[:600]
        trows = [ln for ln in tre.splitlines() if ln.startswith("| stratum | RARE ")]
        assert len(trows) == 2 and "separates" in trows[0], trows
        assert "no separation" in next(ln for ln in tre.splitlines()
                                       if ln.startswith("| peak | RARE | cut | detection |"))
        # The short cell is named in the verdict of the row it was dropped from, and in the checks.
        pt = next(ln for ln in tre.splitlines() if ln.startswith("| stratum | PARTIAL | cut | det"))
        assert "3/4" in pt and "3 (n=2)" in pt, pt
        chk = md.split("### Reader and pairing checks")[1].split("###")[0]
        assert "SHORT (< 4 features)" in chk and "cut/PARTIAL/detection stratum 3 (n=2)" in chk, chk
        assert "under-filled" in chk, chk


def check_autointerp_cuts_catch_a_defect():
    """Four mutations of the cut fixture, each of which must change a number or raise a report.

    Every one of these is a defect this driver would otherwise print as a clean table: a cell mean
    that does not follow its features, a magnitude cut taken over two sets' values at once, a short
    cell that grew and was not re-counted, and a permutation p that does not depend on the labels.
    """
    # 1. Move one feature's accuracy and the cell it sits in must move with it -- the cell mean is
    #    computed from the rows, not from the arm's shape.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        p = root / f"runs/{AI_CUT_RUNS['cut']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        hit = next(r for r in recs if r["arm"] == "RARE" and r["scorer"] == "detection"
                   and r["feature"] == 224)
        hit["bal_acc"] = hit["tpr"] = hit["tnr"] = 0.25
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _cut_analyse(root)
        _close(_cut(res, "strata", "cut/RARE", "detection", 3)["mean"], 0.6875, 1e-12,
               what="(7*0.75 + 0.25) / 8")
        # And the magnitude cell that feature belongs to moves too, and no other one does.
        _close(_cut(res, "peak_strata", "cut/RARE", "detection", 0)["mean"], 0.5, 1e-12,
               what="feature 224 is magnitude quartile 0: (6*0.5 + 0.75 + 0.25) / 8")
        _close(_cut(res, "peak_strata", "cut/RARE", "detection", 1)["mean"], 0.5625, 1e-12)

    # 2. One feature whose rows disagree about `corpus_peak`: the quartiles would otherwise be cut
    #    over a distribution that exists in neither set. It is dropped and REPORTED.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        p = root / f"runs/{AI_CUT_RUNS['cut']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        next(r for r in recs if r["feature"] == 207 and r["arm"] == "BIG")["corpus_peak"] = 99.0
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _cut_analyse(root)
        assert any("207" in n and "corpus_peak" in n for n in res["notes"]), res["notes"]
        # 31 features are quartiled, so the cuts move and one cell loses its feature -- both of
        # which a silent implementation would hide.
        assert res["peak_cuts"] != AI_CUT_QUARTILE_CUTS, res["peak_cuts"]
        # 31 features are quartiled and the 32nd goes to its OWN cell, keyed None -- which the
        # table prints as `quartile —`. A feature this driver cannot place is still a feature it
        # scored, and dropping it out of the block silently is the thing the note exists against.
        mine = [c for c in res["peak_strata"] if c["arm"] == "cut/BIG"
                and c["scorer"] == "detection"]
        assert sum(c["n"] for c in mine if c["stratum"] is not None) == 31, mine
        assert [c["n"] for c in mine if c["stratum"] is None] == [1], mine
        # It is not a quartile, so it is kept out of the trend and named there as well.
        t = _trend(res, "peak", "cut/BIG", "detection")
        assert (t["n_cells"], t["n_cells_all"]) == (4, 5) and t["dropped"] == ["None (n=1)"], t

    # 3. Shrink a second cell below MIN_CELL and both counters must move -- the check is a count of
    #    what is there, not a constant that happens to read 2.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        p = root / f"runs/{AI_CUT_RUNS['cut']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        kept = [r for r in recs if not (r["arm"] == "PARTIAL" and r["scorer"] == "detection"
                                        and 216 <= r["feature"] <= 221)]
        assert len(kept) == len(recs) - 6
        p.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
        _vol, res = _cut_analyse(root)
        c = _cut(res, "strata", "cut/PARTIAL", "detection", 2)
        assert (c["n"], c["short"]) == (2, True), c
        t = _trend(res, "stratum", "cut/PARTIAL", "detection")
        assert (t["n_cells"], t["dropped"]) == (2, ["2 (n=2)", "3 (n=2)"]), t
        ck = [x for x in res["checks"] if x["kind"] == "cells" and x["who"] == "stratum"][0]
        assert ck["n_short"] == 3, ck

    # 4. Shuffle the stratum labels off their features and the permutation p must LEAVE its floor:
    #    a p that came out of the estimator and not out of a constant.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_cuts(root)
        p = root / f"runs/{AI_CUT_RUNS['cut']}/summary/scores.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        for r in recs:
            # One of each stratum per magnitude-quartile-mate, so the cells stay 8 apiece and only
            # the ASSOCIATION with bal_acc is destroyed.
            r["stratum"] = (r["feature"] - AI_CUT_FEATS[0]) % 4
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        _vol, res = _cut_analyse(root)
        assert [_cut(res, "strata", "cut/RARE", "detection", s)["n"] for s in range(4)] \
            == [8, 8, 8, 8]
        t = _trend(res, "stratum", "cut/RARE", "detection")
        _close(t["spread"], 0.0, 1e-12, what="every stratum now holds two of the eight high values")
        assert t["p_perm"] == 1.0, t
# --- eval 2: the per-ACTIVATION-BAND fixture, four features and every item written out ----------
#
# A THIRD autointerp fixture, and a small one, because the band numbers are the only ones in this
# driver that come from a JOIN: the band of an item is the BUILD's (`<build>/<feature>.jsonl`) and
# the answer is the SCORER's (`runs/<run>/<scorer>/batches.jsonl`), and the two meet on the item's
# index within its draw. Nothing about that join is visible in `scores.jsonl`, so it needs items
# written out one by one rather than accuracies dealt from a table.
#
# What it carries, deliberately:
#
#   * feature 303 has NO q0 positive, so `b1` is a three-feature cell while its neighbours are
#     four-feature ones -- a feature with no item in a band must be DROPPED from that band, not
#     counted as a zero, and the two conventions differ by a third of the cell here;
#   * ONE UNPARSED batch (feature 302's positives, arm `M`, detection), so the drop that every
#     other number in this driver makes is made here too and `M`'s `b1` falls to two features;
#   * a `C16-draw2` arm on DRAW 2, whose test2 rows are in the OPPOSITE band order, so a driver
#     that read the `test` rows for it would report `b1 = 1.0` where the answer is 0.0;
#   * the foil band, which is the WHOLE negative side, so its specificity must equal the `*.tnr`
#     cell read straight off `scores.jsonl` -- the join's own end-to-end check, and the one that
#     caught nothing on the fixture but validated the 512-feature run against its own summary.
AI_BAND_RUNS = {"bandrun": "2026-09-22_autointerp-bands"}
AI_BAND_BASE, AI_BAND_SET = "B", "S5"
AI_BAND_BUILD = f"base/{AI_BAND_BASE}/autointerp/{AI_BAND_SET}/2026-09-22_bands-build"
AI_BAND_FEATS = [300, 301, 302, 303]
# The test draws, as `build.draw_test` writes them: `i` is the index WITHIN the draw, `band` is the
# build's own label ("-" on every negative), and draw 2 is deliberately not draw 1 re-ordered.
AI_BAND_ITEMS = {
    "test": {f: ["q0", "q1", "q2", "q3", "top", "-", "-"] for f in (300, 301, 302)},
    "test2": {f: ["q3", "q0", "-"] for f in AI_BAND_FEATS},
}
AI_BAND_ITEMS["test"][303] = ["q1", "q2", "q3", "top", "-", "-"]
# The batches each (feature, arm, scorer) was scored in, as item INDEX lists. Two per feature, so
# a batch can go unparsed without taking the whole feature with it.
AI_BAND_BATCHES = {f: [list(range(4)), list(range(4, len(AI_BAND_ITEMS["test"][f])))]
                   for f in AI_BAND_FEATS}
# Detection predictions, per arm, per feature, BY BAND -- 1 is "activating", and a foil is listed
# by the prediction the judge made on it, in the order the two foils appear.
AI_BAND_DET = {
    "DOCMAX": {
        300: {"q0": 1, "q1": 1, "q2": 1, "q3": 1, "top": 1, "-": [0, 0]},
        301: {"q0": 1, "q1": 1, "q2": 1, "q3": 1, "top": 0, "-": [1, 0]},
        302: {"q0": 0, "q1": 1, "q2": 1, "q3": 1, "top": 0, "-": [0, 0]},
        303: {"q1": 0, "q2": 1, "q3": 1, "top": 0, "-": [0, 0]},
    },
    "M": {
        300: {"q0": 0, "q1": 0, "q2": 1, "q3": 1, "top": 0, "-": [0, 0]},
        301: {"q0": 0, "q1": 0, "q2": 1, "q3": 1, "top": 0, "-": [0, 0]},
        302: {"q0": 0, "q1": 1, "q2": 1, "q3": 1, "top": 0, "-": [0, 0]},
        303: {"q1": 0, "q2": 0, "q3": 1, "top": 0, "-": [0, 0]},
    },
    "C16-draw2": {f: {"q3": 1, "q0": 0, "-": [0]} for f in AI_BAND_FEATS},
}
# Fuzzing is answered perfectly by every arm: it exists here to prove the fuzzing cells take the
# `fuzztpr` / `fuzztnr` metric slot rather than overwriting detection's `tpr` / `tnr`.
AI_BAND_UNPARSED = {("M", "detection", 302, 0)}
# What `scores.jsonl` says about the same answers, computed from the items above by hand and
# written as LITERALS -- so the foil-band cell agreeing with `tnr` is two numbers agreeing, not one
# number compared with itself. (feature -> (tpr, tnr, bal_acc)); None is `run._nr`'s null.
AI_BAND_SCORES = {
    ("DOCMAX", "detection"): {300: (1.0, 1.0, 1.0), 301: (0.8, 0.5, 0.65),
                              302: (0.6, 1.0, 0.8), 303: (0.5, 1.0, 0.75)},
    ("M", "detection"): {300: (0.4, 1.0, 0.7), 301: (0.4, 1.0, 0.7),
                         302: (0.0, 1.0, 0.5), 303: (0.25, 1.0, 0.625)},
    ("C16-draw2", "detection"): {f: (0.5, 1.0, 0.75) for f in AI_BAND_FEATS},
    ("DOCMAX", "fuzzing"): {f: (1.0, 1.0, 1.0) for f in AI_BAND_FEATS},
    ("M", "fuzzing"): {f: (1.0, 1.0, 1.0) for f in AI_BAND_FEATS},
    ("C16-draw2", "fuzzing"): {f: (1.0, 1.0, 1.0) for f in AI_BAND_FEATS},
}
AI_BAND_DRAW = {"C16-draw2": 2}


def _band_pred(arm: str, scorer: str, feat: int, draw: str, i: int) -> tuple[int, int]:
    """(the item's label, the judge's prediction) for one item of one (arm, scorer, feature)."""
    band = AI_BAND_ITEMS[draw][feat][i]
    label = 0 if band == "-" else 1
    if scorer == "fuzzing":
        return label, label          # every arm answers fuzzing perfectly
    spec = AI_BAND_DET[arm][feat]
    if band == "-":
        n_before = sum(1 for b in AI_BAND_ITEMS[draw][feat][:i] if b == "-")
        return 0, int(spec["-"][n_before])
    return 1, int(spec[band])


def write_autointerp_bands(root: Path) -> None:
    """The synthetic mirror: a build directory of per-feature rows, and a run of per-item answers."""
    run_dir = AI_BAND_RUNS["bandrun"]
    b = root / AI_BAND_BUILD
    b.mkdir(parents=True, exist_ok=True)
    for feat in AI_BAND_FEATS:
        rows = [{"kind": "meta", "feature": feat, "stratum": 0, "corpus_peak": 8.0}]
        for draw, per_feat in AI_BAND_ITEMS.items():
            for i, band in enumerate(per_feat.get(feat, [])):
                rows.append({"kind": draw, "i": i, "label": 0 if band == "-" else 1,
                             "band": band, "src": "random" if band == "-" else "corpus",
                             "window": 1000 * feat + i, "doc": 7000 + i, "max_act": 3.0})
        with open(b / f"{feat}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    (b / "build.json").write_text(json.dumps({"base": AI_BAND_BASE, "set": AI_BAND_SET}))

    s = root / f"runs/{run_dir}/summary"
    s.mkdir(parents=True, exist_ok=True)
    rows = []
    for (arm, scorer), per_feat in AI_BAND_SCORES.items():
        for feat, (tpr, tnr, bal) in per_feat.items():
            rows.append({
                "feature": feat, "arm": arm, "scorer": scorer,
                "bal_acc": bal, "tpr": tpr, "tnr": tnr,
                "bal_acc_zero_neg": bal, "tnr_zero": tnr,
                "bal_acc_nearmiss_neg": bal, "tnr_nearmiss": tnr,
                "acc": bal, "n_items": 7, "n_batches": 2,
                "n_parsed": 1 if (arm, scorer, feat, 0) in AI_BAND_UNPARSED else 2,
                "n_pos": 5, "n_neg_nearmiss": 1,
                "draw": AI_BAND_DRAW.get(arm, 1), "role": "arm", "n_examples": 16,
                "explanation_ok": True, "explanation_of": feat, "path": "sync", "gate": 2.0,
                "stratum": 0, "fire_fraction": 0.25, "corpus_peak": 8.0, "density": -4.0,
            })
    with open(s / "scores.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (s / "build.json").write_text(json.dumps(
        {"base": AI_BAND_BASE, "set": AI_BAND_SET, "sae": AI_SAE, "mark": "gate"}))
    # The run's own record of WHICH BUILD it scored, in `C.outdir`'s `## Inputs` layout. This line
    # is the only place it exists -- `build.json` is the build's manifest and does not name its own
    # directory -- so `resolve_build` reads it here exactly as it does on the volume.
    (s / "README.md").write_text(
        "# summary\n\n- date: 2026-09-22 00:00:00Z\n- status: ok\n\n## Inputs\n\n"
        f"- build: /vol/{AI_BAND_BUILD}\n- features: 4 of 4\n- model: claude-sonnet-5\n")

    for scorer in ("detection", "fuzzing"):
        d = root / f"runs/{run_dir}/{scorer}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "batches.jsonl", "w") as fh:
            for arm in AI_BAND_DET:
                draw = "test2" if AI_BAND_DRAW.get(arm, 1) == 2 else "test"
                for feat in AI_BAND_FEATS:
                    idxs = (AI_BAND_BATCHES[feat] if draw == "test"
                            else [list(range(len(AI_BAND_ITEMS["test2"][feat])))])
                    for n, items in enumerate(idxs):
                        pairs = [_band_pred(arm, scorer, feat, draw, i) for i in items]
                        fh.write(json.dumps({
                            "feature": feat, "arm": arm, "batch": n, "items": items,
                            "labels": [p[0] for p in pairs], "preds": [p[1] for p in pairs],
                            "parsed": (arm, scorer, feat, n) not in AI_BAND_UNPARSED,
                            "usage": {"in": 10, "out": 1},
                        }) + "\n")


def _band_analyse(root: Path, **kw):
    vol = R.Vol("", root, offline=True, quiet=True)
    opts = {"runs": dict(AI_BAND_RUNS), "sae": AI_SAE, "ref": "DOCMAX", "boot": 2000, "seed": 1,
            "strata": True, "peak_strata": False, "vs_runs": None, "vs_label": "",
            "bands": True, "band_build": ""}
    opts.update(kw)
    return vol, A.analyse(vol, opts["runs"], opts["sae"], opts["ref"], opts["boot"], opts["seed"],
                          opts["strata"], opts["peak_strata"], opts["vs_runs"], opts["vs_label"],
                          opts["bands"], opts["band_build"])


def _band(res, arm: str, band: str, scorer: str = "detection", half: str = "tpr") -> dict:
    hits = [b for b in res["bands"] if b["arm"] == arm and b["band"] == band
            and b["scorer"] == scorer and b["half"] == half]
    assert len(hits) == 1, (arm, band, scorer, half,
                            [(b["arm"], b["band"], b["scorer"], b["half"]) for b in res["bands"]])
    return hits[0]


def check_autointerp_band_recall():
    """Per-band recall and foil specificity, every number summed by hand from the items above.

    The four equal-width bins are `b1..b4` LOWEST FIRST and the foils are `b0`, which is the
    spelling `paper/numbers/cells.csv`'s seeded rows already carry.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_bands(root)
        _vol, res = _band_analyse(root)
        assert res["bands"], res["notes"]
        # DOCMAX: q0 is 2 of the 3 features that HAVE a q0 positive, and 303 is not counted as a
        # miss for lacking one. q2 and q3 are perfect; `top` is one of four.
        _close(_band(res, "bandrun/DOCMAX", "b1")["mean"], 2 / 3, 1e-12, what="DOCMAX q0")
        assert _band(res, "bandrun/DOCMAX", "b1")["n_features"] == 3
        _close(_band(res, "bandrun/DOCMAX", "b2")["mean"], 0.75, 1e-12, what="DOCMAX q1")
        assert _band(res, "bandrun/DOCMAX", "b2")["n_features"] == 4
        _close(_band(res, "bandrun/DOCMAX", "b3")["mean"], 1.0, 1e-12, what="DOCMAX q2")
        _close(_band(res, "bandrun/DOCMAX", "b4")["mean"], 1.0, 1e-12, what="DOCMAX q3")
        _close(_band(res, "bandrun/DOCMAX", "btop")["mean"], 0.25, 1e-12, what="DOCMAX top")
        # The unparsed batch took feature 302's four positives with it and nothing else: `M`'s q0
        # cell is two features, its `top` cell is still four.
        _close(_band(res, "bandrun/M", "b1")["mean"], 0.0, 1e-12, what="M q0")
        assert _band(res, "bandrun/M", "b1")["n_features"] == 2
        _close(_band(res, "bandrun/M", "b2")["mean"], 0.0, 1e-12, what="M q1")
        _close(_band(res, "bandrun/M", "b3")["mean"], 2 / 3, 1e-12, what="M q2")
        _close(_band(res, "bandrun/M", "b4")["mean"], 1.0, 1e-12, what="M q3")
        _close(_band(res, "bandrun/M", "btop")["mean"], 0.0, 1e-12, what="M top")
        assert _band(res, "bandrun/M", "btop")["n_features"] == 4
        # The foil band is the WHOLE negative side, so it must reproduce the `tnr` cell that was
        # read off `scores.jsonl` -- two independently written numbers, not one compared to itself.
        for arm in ("DOCMAX", "M"):
            b0 = _band(res, f"bandrun/{arm}", "b0", half="tnr")
            tnr = _ai_cell(res, f"bandrun/{arm}", "detection")["tnr"]
            _close(b0["mean"], tnr["mean"], 1e-12, what=f"{arm} foil specificity == its TNR")
            assert b0["n_features"] == tnr["n"], (b0, tnr)
        _close(_band(res, "bandrun/DOCMAX", "b0", half="tnr")["mean"], 0.875, 1e-12)
        _close(_band(res, "bandrun/M", "b0", half="tnr")["mean"], 1.0, 1e-12)
        # DRAW 2 is read from the `test2` rows: their bands are in the opposite order, so a driver
        # that took the `test` rows for this arm would report b1 = 1.0 and b4 = 0.0.
        _close(_band(res, "bandrun/C16-draw2", "b1")["mean"], 0.0, 1e-12, what="draw-2 q0")
        _close(_band(res, "bandrun/C16-draw2", "b4")["mean"], 1.0, 1e-12, what="draw-2 q3")
        assert not [b for b in res["bands"] if b["arm"] == "bandrun/C16-draw2"
                    and b["band"] in ("b2", "b3", "btop")]
        # Fuzzing is answered perfectly and is its OWN set of rows, never overwriting detection's.
        for band in ("b1", "b2", "b3", "b4", "btop"):
            _close(_band(res, "bandrun/DOCMAX", band, scorer="fuzzing")["mean"], 1.0, 1e-12)
        # `--bands-build` names the same directory outright and must give the same rows.
        _vol2, res2 = _band_analyse(root, band_build=AI_BAND_BUILD)
        assert [(b["arm"], b["band"], b["half"], b["mean"]) for b in res2["bands"]] \
            == [(b["arm"], b["band"], b["half"], b["mean"]) for b in res["bands"]]
        # And OFF by default: no `--bands`, no rows, and no fetch of the build products.
        _vol3, res3 = _band_analyse(root, bands=False)
        assert res3["bands"] == [] and res3["band_per_feature"] == {}


def check_autointerp_band_cells_and_contrasts():
    """The `ai.*` rows the bands produce: the key grammar, and the paired per-band differences."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_bands(root)
        _vol, res = _band_analyse(root)
        rows, gaps = A.cells_rows(res, set_slot="l131k", run_id="R9", status="final",
                                  date="2026-09-22", run_label="bandrun")
        by_key = {r["key"]: r for r in rows}
        # The arm slot is the CELLS slot and not the arm name, on a band row as on any other.
        assert by_key["ai.l131k.c16.tpr.b1"]["value"] == "0.667", by_key["ai.l131k.c16.tpr.b1"]
        assert by_key["ai.l131k.c16.tpr.b1"]["n"] == 3
        assert by_key["ai.l131k.mtop16.tpr.b4"]["value"] == "1.000"
        assert by_key["ai.l131k.mtop16.tpr.btop"]["value"] == "0.000"
        assert by_key["ai.l131k.null.tpr.b1"]["value"] == "0.000"    # the draw-2 arm
        assert by_key["ai.l131k.c16.tnr.b0"]["value"] == "0.875"
        # Detection keeps the unqualified metric slots; fuzzing spells itself out.
        assert by_key["ai.l131k.c16.fuzztpr.b1"]["value"] == "1.000"
        assert by_key["ai.l131k.c16.fuzztnr.b0"]["value"] == "1.000"
        # The paired contrast, per band, on the INTERSECTION of the two arms' banded features:
        # q0 is [0 - 1, 0 - 1] over features 300 and 301 alone, because the unparsed batch left
        # `M` no q0 rate on 302 -- a contrast that took the two means would get 0 - 2/3.
        d = by_key["ai.l131k.diff.tpr.b1"]
        assert (d["value"], d["n"]) == ("-1.000", 2), d
        assert by_key["ai.l131k.diff.tpr.b2"]["value"] == "-0.667"
        assert by_key["ai.l131k.diff.tpr.b3"]["value"] == "-0.333"
        assert by_key["ai.l131k.diff.tpr.b4"]["value"] == "0.000"
        assert by_key["ai.l131k.diff.tpr.btop"]["value"] == "-0.250"
        assert by_key["ai.l131k.diff.tnr.b0"]["value"] == "0.125"
        assert by_key["ai.l131k.diff.tnr.b0"]["n"] == 4
        # Every band row carries its own provenance: which build the band came from, and what the
        # band IS -- `b3` on its own is unreadable a month later.
        assert AI_BAND_BUILD in by_key["ai.l131k.c16.tpr.b3"]["note"]
        assert "equal-width" in by_key["ai.l131k.c16.tpr.b3"]["note"]
        assert "near-miss" in by_key["ai.l131k.c16.tnr.b0"]["note"]
        # NLA is not in this run, so its contrast is a stated GAP and not a silent absence.
        assert any("diffnla" in g and "b1" in g for g in gaps), gaps
        assert not any(k.startswith("ai.l131k.diffnla.tpr.") for k in by_key)
        # Without `--bands` there are no band rows at all, and the gap says the file's existing
        # ones were not refreshed rather than letting them look current.
        _vol2, plain = _band_analyse(root, bands=False)
        rows2, gaps2 = A.cells_rows(plain, set_slot="l131k", run_id="R9", status="final",
                                    date="2026-09-22", run_label="bandrun")
        assert not [r for r in rows2 if ".tpr.b" in r["key"] or ".tnr.b0" in r["key"]]
        assert any("rerun with `--bands`" in g for g in gaps2), gaps2


def check_autointerp_bands_catch_a_defect():
    """Flip ONE item's label in the build and require the join to refuse.

    The build and the scorer are two products, and joining a run against the WRONG build would
    silently report another item's band for every answer. The labels are the one field both files
    carry, so they are compared on every item and a disagreement is an assertion, not a note.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_autointerp_bands(root)
        p = root / AI_BAND_BUILD / "300.jsonl"
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        hit = next(r for r in recs if r.get("kind") == "test" and r["i"] == 0)
        assert hit["label"] == 1 and hit["band"] == "q0", hit
        hit["label"] = 0                      # the scorer's batch still says 1
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        try:
            _band_analyse(root)
        except AssertionError as e:
            assert "is NOT the one" in str(e), str(e)
        else:
            raise AssertionError("a build whose labels disagree with the run was joined anyway")


def _ood():
    """`results/ood.py` without its lazy `stats_ood` import, which needs fasttext and polars."""
    import importlib.util

    path = Path(__file__).resolve().parent / "ood.py"
    spec = importlib.util.spec_from_file_location("results_ood", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_ood_scan_key():
    """`scan/<set>[__<corpus>[__<M>m]]` parsed back to (corpus, bound), including the one shape
    that was ambiguous before the key carried the corpus label: a size-bounded scan of the base's
    OWN corpus. `<set>__4m` would have read as "a corpus whose directory is 4m"."""
    od = _ood()
    S = "2026-09-21_ood_q1"

    def one(name):
        got = od.parse_scan_dir(name, S)
        return None if got is None else (got[0], got[1])

    assert one(S) == ("", None), "the unbounded English scan keeps the bare set name"
    assert one(f"{S}__tha_Thai") == ("tha_Thai", None)
    assert one(f"{S}__tha_Thai__1m") == ("tha_Thai", 1.0)
    # the shape the fix exists for: the base's own corpus, bounded. `corpus` is the label, `4m`
    # the bound -- not a corpus called `4m` with no bound.
    assert one(f"{S}__corpus__4m") == ("corpus", 4.0)
    assert one(f"{S}__train_parity_10m") == ("train_parity_10m", None), (
        "a corpus directory ending in `m` is not a size suffix"
    )
    assert one("2026-09-18_ood_v1__tha_Thai") is None, "another set's scan is not this set's"
    # THE TAGGED SHAPE -- a second scan of one (set, corpus) under a different centring mean.
    # A regex with two optional trailing groups reads this as a corpus literally called
    # `tha_Thai__1m__mu-whiten`, which makes the whole whiten pass invisible.
    assert one(f"{S}__tha_Thai__1m__mu-whiten") == ("tha_Thai", 1.0)
    assert od.parse_scan_dir(f"{S}__tha_Thai__1m__mu-whiten", S)[2] == "mu-whiten"
    assert one(f"{S}__corpus__4m__mu-whiten") == ("corpus", 4.0)
    assert od.parse_scan_dir(f"{S}__tha_Thai", S) == ("tha_Thai", None, "")


def check_ood_arm_table():
    """`arm_rows` on three hand-built arms, one per outcome of the R9 three-state verdict.

    The estimator is INJECTED (that is why `arm_rows` takes it), so this checks the pairing, the
    drop rule and the verdict wiring against numbers written out here -- not a bootstrap against
    itself. `exceeds` is +0.20 on every target, `reversed` is -0.20, `inconclusive` straddles.
    """
    od = _ood()
    ids, per_target, top1 = [], {}, {}
    CORPUS = 0.50
    # the CENTRED bo64 of each arm; the raw one sits 0.05 below it, so a table that read the wrong
    # cosine would not merely shift the mean, it would change the verdict of `a_inc`.
    plan = {"a_ex": 0.70, "a_rev": 0.30, "a_inc": 0.50}
    row = 0
    for arm, cen in plan.items():
        for i in range(8):
            ids.append({"row": row, "arm": arm, "family": "lang", "doc": 1000 + row})
            wobble = 0.02 if i % 2 else -0.02
            per_target[row] = {"bo_64": cen - 0.05 + wobble, "bo_c_64": cen - 0.02 + wobble,
                               "bo_a_64": cen + wobble}
            top1[(row, 1.0)] = CORPUS
            row += 1
    # one target of `a_ex` has NO scan cell: it must drop out of the pair, not score zero
    del top1[(0, 1.0)]
    # EVERY scan carries EVERY target (design §4), so each arm's corpus has a top-1 for all rows.
    # Only the arm's OWN corpus is in-domain; the others are the cross-domain cells, and they are
    # given a far HIGHER value here so that reading the wrong one would flip every verdict.
    top1_by_arm = {}
    for arm in plan:
        own = dict.fromkeys(top1, 0.95)      # what the OTHER arms' corpora would say
        own.update(top1)                     # this arm's own numbers
        top1_by_arm[arm] = own

    def boot_ci(d):
        m = float(np.mean(d))
        h = 3.0 * float(np.std(d, ddof=1)) / math.sqrt(d.size) if d.size > 1 else 1.0
        return m, m - h, m + h

    src = R.Source(maemm="m", base="b", engine="vllm", run_tag="", scores_rel="", rollouts_rel="")
    src.per_target = per_target
    # `arm_rows` takes the size PER ARM since M5 (2026-09-23), because the real run holds 21 arms
    # at 10M and `shell` at 4M in one table. Here every arm is at 1M.
    sizes1 = dict.fromkeys(plan, 1.0)
    recs, skipped = od.arm_rows(ids, src, top1_by_arm, sizes1, "asym", boot_ci, R_outcome, None)
    assert not skipped, skipped
    got = {r["arm"]: r for r in recs}
    # a source centred on ANOTHER mean gets its cosines and NO delta: the two sides would be
    # angles to different target vectors (measured on the real set: cos 0.977 between the means,
    # median 0.969 between unit(act - mu), which is the size of the effect itself)
    # NOT comparable AND no scan at all at that mean -- the real shape of the case: a checkpoint
    # centred on a mean nothing was scanned at. It must still yield one ROW PER ARM carrying the
    # cosines. Asserting the count first, because `for r in []` passes every check vacuously and
    # that is exactly how the first version of this shipped a table with no rows in it.
    inc, _ = od.arm_rows(ids, src, {}, sizes1, "asym", boot_ci, R_outcome, None, comparable=False)
    assert len(inc) == 3, f"an incomparable source must still report every arm's cosines: {inc}"
    for r in inc:
        assert r["delta"] is None and r["ci_lo"] is None and r["outcome"] == "not comparable", r
        assert r["bo64_asym"] is not None and r["bo64_centred"] is not None, r
        assert r["bo64_raw"] is not None, r
        assert r["corpus_top1"] is None and r["win_frac"] is None, r
        assert r["n"] == 8, f"every target of the arm has a cosine: {r}"
    # and the mu spellings that must compare EQUAL / UNEQUAL
    assert od.C_resolve_mu("base/{base}/stats/mu.f32", "B") == od.C_resolve_mu(
        "/vol/base/B/stats/mu.f32", "B"
    ), "config's relative spelling and score's absolute one are the same file"
    assert od.C_resolve_mu("/vol/archive/x/whiten_mu.npy", "B") != od.C_resolve_mu(
        "base/{base}/stats/mu.f32", "B"
    )
    assert od.C_resolve_mu(None, "B") == "none"
    # The rollouts path comes from the scores README, because --score-tag deliberately makes the
    # scores directory name differ from the rollouts stem. THE RULE MOVED on 2026-09-22 to
    # `reconstruction/stats_ood.rollouts_rels_from_readme`, the layer both readers already import:
    # `stats_ood.rollout_texts` still rebuilt the path from its own `--stem` and so still had the
    # defect this half was already fixed for. Its behaviour, including the root-relative strip
    # and the pre-README fallback, is pinned by `stats_ood.py selfcheck`, which runs without the
    # fasttext/polars extras this file deliberately does not import.
    assert not hasattr(od, "ROLLOUTS_RE"), (
        "results/ood has its own copy of the rollouts-README rule again; there is one, in "
        "stats_ood.rollouts_rels_from_readme, and two copies is how the two halves diverged"
    )
    # the README line every scan writes, which is how a scan's mean is read back
    assert od.MU_RE.search(
        "## Notes\n\n- CENTRING: mu=/vol/base/B/stats/mu.f32 from --mu (explicit)\n"
    ).group(1) == "/vol/base/B/stats/mu.f32"
    assert od.MU_RE.search("- CENTRING: directions derived from /x at mu=/y: unit(act - mu)") is None, (
        "only the `mu=<path> from <source>` line names the run's mean; the derivation line "
        "mentions a mu too and must not be mistaken for it"
    )
    assert od.CENTRE_MU_RE.search(
        "- CENTRING: directions derived from /x at mu=/y: unit(act - mu)"
    ) is None, "the derivation line is not the `--centre` line either"

    # an arm whose own corpus was never scanned must be reported, never scored off another's
    partial = {k: v for k, v in top1_by_arm.items() if k != "a_rev"}
    recs2, skipped2 = od.arm_rows(ids, src, partial, sizes1, "asym", boot_ci, R_outcome, None)
    assert [r["arm"] for r in recs2] == ["a_ex", "a_inc"], [r["arm"] for r in recs2]
    assert any("a_rev" in m and "own corpus" in m for m in skipped2), skipped2
    assert sorted(got) == ["a_ex", "a_inc", "a_rev"]
    assert got["a_ex"]["n"] == 7, f"the target with no scan cell must drop: {got['a_ex']['n']}"
    assert got["a_ex"]["outcome"] == "exceeds" and got["a_ex"]["win_frac"] == 1.0
    assert got["a_rev"]["outcome"] == "reversed" and got["a_rev"]["win_frac"] == 0.0
    assert got["a_inc"]["outcome"] == "inconclusive", got["a_inc"]
    # Δ is on the CENTRED cosine here (centred=True), so it is bo_c_64 - corpus, not bo_64 - corpus
    _close(got["a_ex"]["delta"], 0.70 - 0.50, tol=0.01, what="Δ uses the cosine it was asked for")
    _close(got["a_ex"]["bo64_raw"], 0.65, tol=0.01, what="the raw column is reported beside it")
    _close(got["a_ex"]["bo64_centred"], 0.68, tol=0.01, what="and the centred one")
    # the same arms read on the RAW cosine: every Δ moves down 0.05, and `a_inc` becomes reversed
    raw_recs, _ = od.arm_rows(ids, src, top1_by_arm, sizes1, "raw", boot_ci, R_outcome, None)
    raw = {r["arm"]: r for r in raw_recs}
    _close(raw["a_ex"]["delta"], 0.15, tol=0.01, what="the raw Δ is 0.05 below the asym one")
    assert raw["a_inc"]["outcome"] == "reversed", (
        "reading the wrong cosine must change a verdict, or this fixture does not test the choice"
    )
    # every target of an arm is its own pool document, so clustering is a no-op and must SAY so
    assert got["a_ex"]["n_clusters"] == got["a_ex"]["n"]


def R_outcome(lo, hi):
    """`reconstruction/stats_ood.outcome`, restated here so the selftest needs no polars."""
    return "exceeds" if lo > 0 else ("reversed" if hi < 0 else "inconclusive")


def check_ood_bo8_headline():
    """The bo8 pair, its verdict, and the cells map (M5, 2026-09-23).

    Spec section 2's headline is bo8 against the own-domain corpus, not bo64, so `arm_rows` has to
    pair the RECOMPUTED bo8 with the same target's corpus cell and run the estimator on THAT. The
    numbers here make the two disagree on purpose: bo64 sits at 0.70 against a 0.50 corpus
    (`exceeds`) while bo8 sits at 0.44 (`reversed`), so a table that carried the bo64 verdict into
    `outcome8` would be caught here rather than in the paper.
    """
    od = _ood()
    ids, per_target, top1, bo8 = [], {}, {}, {}
    for i in range(8):
        ids.append({"row": i, "arm": "a", "family": "lang", "doc": 100 + i})
        w = 0.01 if i % 2 else -0.01
        per_target[i] = {"bo_64": 0.65 + w, "bo_c_64": 0.70 + w, "bo_a_64": 0.68 + w}
        top1[(i, 10.0)] = 0.50
        bo8[i] = 0.44 + w
    # one row has NO bo8 (fewer than 8 finite centred draws): it must drop out of the bo8 pair and
    # stay in the bo64 one, which is the whole reason the two pairs are collected separately.
    del bo8[7]

    def boot_ci(d):
        m = float(np.mean(d))
        h = 3.0 * float(np.std(d, ddof=1)) / math.sqrt(d.size) if d.size > 1 else 1.0
        return m, m - h, m + h

    src = R.Source(maemm="m", base="b", engine="vllm", run_tag="", scores_rel="", rollouts_rel="")
    src.per_target = per_target
    recs, skipped = od.arm_rows(ids, src, {"a": top1}, {"a": 10.0}, "centred", boot_ci, R_outcome,
                                None, bo8=bo8)
    assert not skipped, skipped
    r = recs[0]
    assert r["n"] == 8 and r["n8"] == 7, f"the bo8 pair drops the row with no bo8: {r}"
    assert abs(r["delta"] - 0.20) < 1e-6, r          # bo64 0.70 - corpus 0.50
    assert abs(r["delta8"] + 0.06) < 0.01, r         # bo8 0.44 - corpus 0.50
    assert r["outcome"] == "exceeds" and r["outcome8"] == "reversed", (
        f"the verdict must follow the bo8 pair, not the bo64 one: {r}"
    )
    # no bo8 at all -> no bo8 verdict, and the bo64 columns are untouched
    recs2, _ = od.arm_rows(ids, src, {"a": top1}, {"a": 10.0}, "centred", boot_ci, R_outcome, None)
    assert recs2[0]["delta8"] is None and recs2[0]["outcome8"] == "no bo8 pairs", recs2[0]

    # the cells key map, read off `paper/numbers/cells.csv`'s existing ids
    want = {"ces_Latn": "ces", "rus_Cyrl": "rus", "ell_Grek": "ell", "arb_Arab": "arb",
            "hin_Deva": "hin", "tha_Thai": "tha", "cmn_Hani": "cmn", "jpn_Jpan": "jpn",
            "ufw_zh": "zh", "ufw_en": "en", "python": "python", "javascript": "javascript",
            "c": "c", "rust": "rust", "go": "go", "haskell": "haskell", "sql": "sql",
            "shell": "shell", "owm": "owm", "arxiv": "arxiv", "lean": "lean",
            "isabelle": "isabelle", "formulas": "formulas"}
    got = {a: od.cell_id_of(a) for a in want}
    assert got == want, {a: (got[a], want[a]) for a in want if got[a] != want[a]}
    cfg = PC.load_config()
    assert set(want) == set(cfg["ood_arms"]), "the cells map and ood_arms have drifted apart"
    # THE CONFIG IS NO LONGER A SIZE SOURCE. `arm_size_m(cfg, arm)` is gone: it read
    # `ood_arms.<arm>.sizes[-1]`, which still says 16 for four arms whose scans ran `--max-size
    # 10`, and `corp10.mtok` would have printed a size nothing measured. `check_ood_arm_size_
    # comes_from_the_product` pins the replacement.
    assert not hasattr(od, "arm_size_m"), (
        "results/ood reads the arm's corpus size out of config.yaml again; the product that was "
        "actually scanned is the only source for it (runs/2026-09-23_ledger.md, M5 gap 3/4)"
    )
    assert [float(x) for x in cfg["ood_arms"]["tha_Thai"]["sizes"]][-1] == 16.0, (
        "the config no longer declares 16 for tha_Thai, so the drift this check guards is gone "
        "and the fixture below no longer proves anything"
    )


def check_ood_lid_ranking():
    """The language-id column must be read off the top-1 rollout BY SCORE, not rollout k=0.

    The rollouts jsonl carries no score, so appending in file order gives the FIRST SAMPLED draw
    at T = 1.0 -- an arbitrary one of 64. Measured cost of that bug on the real run
    (infra/2026-09-22_ood-lid-check.md): jpn_Jpan 0.062 -> 0.562, ces_Latn 0.562 -> 0.812,
    ell_Grek 0.500 -> 0.750. The fixture below is built so the k=0 answer and the top-1 answer
    DISAGREE on every row; a reader that ignores the scores gets 0.0 where the truth is 1.0.
    """
    od = _ood()
    n_rows, n_roll, width = 4, 3, 5
    # rollout 0 is always the wrong language, rollout 2 always the right one and always best
    texts = []
    for row in range(n_rows):
        for k in range(n_roll):
            texts.append({"row": row, "text": ("WRONG" if k != 2 else "RIGHT")})
    cos = np.full((n_rows, n_roll, width), np.nan, dtype=np.float32)
    for row in range(n_rows):
        cos[row, 0, :2] = 0.10
        cos[row, 1, :2] = 0.20
        cos[row, 2, :2] = 0.90          # the best, and the last in file order
    with tempfile.TemporaryDirectory() as td:
        mirror = Path(td)
        rel = "maemms/m/scores/S__vllm__asym"
        (mirror / rel).mkdir(parents=True)
        cos.astype(np.float16).tofile(mirror / rel / "cos_asym.f16")
        (mirror / rel / "index.json").write_text(json.dumps(
            {"cos_asym.f16": {"dtype": "float16", "shape": [n_rows, n_roll, width]}}))
        vol = R.Vol("", mirror, offline=True)
        src = R.Source(maemm="m", base="b", engine="vllm", run_tag="asym",
                       scores_rel=rel, rollouts_rel="")
        src.per_target = {r: {"max_cos_asym": 0.9} for r in range(n_rows)}
        ranked, note = od.ranked_texts(vol, src, "asym", texts)
        assert all(v[0] == "RIGHT" for v in ranked.values()), ranked
        assert "top-1 BY SCORE" in note, note
        # and the cross-check must REFUSE a score array that is not the one per_target reports
        src.per_target = {r: {"max_cos_asym": 0.5} for r in range(n_rows)}
        try:
            od.ranked_texts(vol, src, "asym", texts)
        except AssertionError:
            pass
        else:
            raise AssertionError("ranking against an array that disagrees with per_target must stop")
        # with no score array at all it falls back to file order AND SAYS SO
        (mirror / rel / "index.json").write_text("{}")
        vol2 = R.Vol("", mirror, offline=True)
        ranked2, note2 = od.ranked_texts(vol2, src, "asym", texts)
        assert ranked2[0][0] == "WRONG" and "k=0" in note2, note2


def check_ood_lid_wantlist():
    """An arm's want-list must contain the label the classifier actually returns for its text.

    lid218e calls Chinese `yue_Hant` on this corpus: measured on the arms' OWN corpus windows --
    Chinese by construction -- it is the top label 11/16 on cmn_Hani and 12/16 on ufw_zh. With
    the old `[zho_Hans, zho_Hant]` list the CEILING was 0.31 / 0.25, so the rate said nothing
    about the model. Checked against the REAL config, so dropping the label fails here.
    """
    cfg = R.load_config()
    for arm in ("cmn_Hani", "ufw_zh"):
        want = cfg["ood_arms"][arm]["lid"]
        assert "yue_Hant" in want, (
            f"ood_arms.{arm}.lid is {want}: lid218e labels this corpus's text `yue_Hant`, and "
            f"without it the classifier's ceiling on this arm is 0.25-0.31 -- a rate measured "
            f"against that is an artefact, not a finding (infra/2026-09-22_ood-lid-check.md)"
        )
    # a lang arm whose lid is null would silently fall through to the code_like branch
    for arm, spec in cfg["ood_arms"].items():
        if spec["family"] in ("lang", "ctrl"):
            assert spec.get("lid"), f"lang/ctrl arm {arm} has no lid want-list"





# --- M5: the four reader seams the full-scale OOD run refused on, and the one mixed verdict -----
#
# Every one of these is a defect that printed a clean table. The run they come from is
# `runs/2026-09-23_ledger.md` (M5, 22 arms x 512 targets, all 66 jobs landed): the driver withheld
# all 22 Δ, `lid` never ran, the table could not hold 10M and 4M at once, four arms' size cells
# would have printed a size nothing measured, and the prose sentence counted a different verdict
# from the cells beneath it.

OOD_CENTRE_README = """# 2026-09-23_ood_full__ufw_en__10m__paper0923

## Inputs

- corpus: /vol/base/B/corpora/ufw_en
- corpus label: ufw_en
- targets: 11264
- sizes: [1, 4, 10]
- max_size: 10

## Notes

- CENTRING: directions derived from /vol/base/B/heldout/S/act.f32 at \
mu=/vol/archive/g/whiten_mu.npy: unit(act - mu) on 11264 centrable rows (['code', 'lang'])
- CENTRING: --centre: BOTH sides about /vol/archive/g/whiten_mu.npy, the scoring constant \
(common.score_mu); every target family here is `centrable`
"""


def _ood_scan_mirror(root: Path, base: str, scan_dir: str, readme: str) -> R.Vol:
    (root / f"base/{base}/scan/{scan_dir}").mkdir(parents=True, exist_ok=True)
    (root / f"base/{base}/scan/{scan_dir}/README.md").write_text(readme)
    return R.Vol("", root, offline=True)


def check_ood_centred_scan_mean_is_read():
    """A `--centre` scan's mean comes back from its README, and the scan is marked centred.

    THE DEFECT: `MU_RE` matched only `common.note_convention`'s `- CENTRING: mu=<path> from ...`,
    the line a product that resolved a `--mu` writes. `precompute/scan.py --centre` writes neither
    that line nor anything like it -- it writes `- CENTRING: --centre: BOTH sides about <path>,
    the scoring constant` -- so 0 of the M5 run's 22 scan READMEs matched, every scan resolved to
    `mu=none`, `top1_by_corpus` (keyed on (corpus, mean)) missed on every lookup, and all 22 arms
    came out `not comparable` beside two scored sources whose `rows.json` names that very path.

    The two sides ARE at one mean and the READMEs say so; the reader could not hear it.
    """
    od = _ood()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vol = _ood_scan_mirror(root, "B", "S__ufw_en__10m__tag", OOD_CENTRE_README)
        mu, centred = od.scan_mu_of(vol, "B", "S__ufw_en__10m__tag")
        assert mu == "/vol/archive/g/whiten_mu.npy", (
            f"the `--centre` scan's mean was not read back from its README: {mu!r}")
        assert centred is True, "a `--centre` scan must be recorded as centred"
        # AND IT PAIRS: the same path as a source's `rows.json` spells it, once both are resolved.
        assert od.C_resolve_mu(mu, "B") == od.C_resolve_mu(
            "/vol/archive/g/whiten_mu.npy", "B") == "archive/g/whiten_mu.npy"
        # the scan's sizes come off the same README, and they are the product's, not the config's
        assert od.scan_sizes_of(vol, "B", "S__ufw_en__10m__tag") == [1.0, 4.0, 10.0]

        # The legacy shape still reads, and still reads as NOT centred: a product that resolved a
        # `--mu` centred its TARGETS, and says nothing about the corpus windows.
        vol2 = _ood_scan_mirror(
            root, "B", "S__legacy",
            "## Notes\n\n- CENTRING: mu=/vol/base/B/stats/mu.f32 from --mu (explicit)\n")
        assert od.scan_mu_of(vol2, "B", "S__legacy") == ("/vol/base/B/stats/mu.f32", False)

        # A README with ONLY the derivation line names no scan mean: that line describes how the
        # target bank's unit vectors were built, and a scan that derived centred directions while
        # scoring uncentred windows is at no comparable mean at all.
        vol3 = _ood_scan_mirror(
            root, "B", "S__deriv",
            "## Notes\n\n- CENTRING: directions derived from /x at mu=/y: unit(act - mu)\n")
        assert od.scan_mu_of(vol3, "B", "S__deriv") == (None, False)
        # and a scan whose README is not on the volume at all
        assert od.scan_mu_of(R.Vol("", root, offline=True), "B", "S__absent") == (None, False)


def check_ood_lid_reads_every_chunk_the_readme_names():
    """`lid_rates` reads the WHOLE chunked rollouts product, not one file of it.

    THE DEFECT: `score` names every rollouts file it read on one `- rollouts:` line, and for a
    product `--rows` split into chunks that is a comma-separated list of 22 paths. The reader
    asked for ONE path, got None, and `lid` did not run -- so `ex.lid`, `corp10.lid` and both
    classifier ceilings were empty on a product that was complete (ledger M5 gap 1).

    The union itself is `precompute.common.read_rollouts`, exercised for real below (the chunk
    files and their summaries are written into a temp mirror); only the README regex is stubbed
    here, and it is pinned by `stats_ood.py selfcheck`, which this file cannot import.

    The fixture makes reading ONE chunk visibly wrong: each chunk holds a different arm, so a
    reader that stops at the first reports one arm instead of two.
    """
    od = _ood()
    n_roll, width = 4, 3
    arms = {"lang_a": (0, 1), "lang_b": (2, 3)}          # arm -> its two target rows
    ids = [{"row": r, "arm": a, "family": "lang", "doc": 500 + r}
           for a, rows in arms.items() for r in rows]
    # rollout 3 always scores best and is always the RIGHT language; k=0 is always wrong.
    cos = np.full((4, n_roll, width), np.nan, dtype=np.float32)
    for row in range(4):
        for k in range(n_roll):
            cos[row, k, :2] = 0.10 * (k + 1)

    with tempfile.TemporaryDirectory() as td:
        mirror = Path(td)
        srel = "maemms/b/m/scores/S__vllm__tag"
        (mirror / srel).mkdir(parents=True)
        cos.astype(np.float16).tofile(mirror / srel / "cos_centred.f16")
        (mirror / srel / "index.json").write_text(json.dumps(
            {"cos_centred.f16": {"dtype": "float16", "shape": [4, n_roll, width]}}))
        rdir = mirror / "maemms/b/m/rollouts"
        rdir.mkdir(parents=True)
        stem = "S__vllm__tag"
        inv = {"maemm": "b/m", "base": "b", "set": "S", "engine": "vllm", "kind": "ood",
               "n": n_roll, "bo": n_roll, "seed": 1, "max_new": 64, "min_new": 0, "prompt": "p",
               "prompt_tokens": 3, "marker_pos": 1, "inject_layer": 42, "inject_coef": 1.0,
               "temperature": 1.0, "top_p": 1.0, "top_k": 0, "weight_sha256": "0" * 64,
               "score_max_length": 95}
        rels = []
        for arm, rows in arms.items():
            spec = f"{rows[0]}-{rows[1]}"
            (rdir / f"{stem}{PC.ROWS_MARK}{spec}.jsonl").write_text("\n".join(
                json.dumps({"row": r, "k": k,
                            "text": ("WRONG" if k != n_roll - 1 else f"RIGHT {arm}")})
                for r in rows for k in range(n_roll)) + "\n")
            (rdir / f"{stem}{PC.ROWS_MARK}{spec}.summary.json").write_text(
                json.dumps({**inv, "rows": list(rows)}))
            rels.append(f"maemms/b/m/rollouts/{stem}{PC.ROWS_MARK}{spec}.jsonl")

        asked: list[str] = []

        class _Mod:
            """`reconstruction/stats_ood`'s two rollout readers, at their real contract."""

            @staticmethod
            def rollouts_rels_from_readme(vol, scores_rel):
                asked.append(scores_rel)
                return list(rels)

            @staticmethod
            def read_rollout_rows(vol, rels_):
                for rel in rels_:
                    assert vol.get(rel) is not None, rel
                rows_, _summary, _ = PC.read_rollouts(
                    str(Path(vol.local) / "maemms/b/m/rollouts"), stem)
                return rows_

            @staticmethod
            def load_lid(path):
                return None

            @staticmethod
            def lid_label(model, text):
                return (text.split()[-1] if text.startswith("RIGHT") else "xxx"), 1.0

        vol = R.Vol("", mirror, offline=True)
        src = R.Source(maemm="b/m", base="b", engine="vllm", run_tag="tag",
                       scores_rel=srel, rollouts_rel="")
        src.per_target = {r: {"max_cos_centred": 0.1 * n_roll} for r in range(4)}
        cfg = {"ood_arms": {a: {"lid": [a], "family": "lang"} for a in arms}}

        class _LidModel:
            pass

        _Mod.load_lid = staticmethod(lambda path: _LidModel())
        out, notes = od.lid_rates(_Mod, vol, cfg, ids, src, "S", None, "centred", {})
        assert asked == [srel], asked
        assert sorted(out) == ["lang_a", "lang_b"], (
            f"only the chunks of ONE arm were read, so a whole arm is missing: {sorted(out)}")
        for arm in arms:
            assert out[arm]["lid_top1_rate"] == 1.0, (
                f"{arm}: the language column was not taken on the top-1 BY SCORE across chunks: "
                f"{out[arm]}")
        assert any("2 `__rows` chunks of one product" in n for n in notes), notes

        # A README that names nothing still says so, rather than reporting an empty product.
        _Mod.rollouts_rels_from_readme = staticmethod(lambda vol, rel: [])
        out2, notes2 = od.lid_rates(_Mod, vol, cfg, ids, src, "S", None, "centred", {})
        assert out2 == {} and any("does not name its rollouts file" in n for n in notes2), notes2


def check_ood_arm_size_comes_from_the_product():
    """Each arm's corpus size is the size ITS OWN scan reached, in the row and in the cells.

    TWO DEFECTS, one source. (a) `main` took ONE `size_m` for the table, so the M5 run -- 21 arms
    at 10M and `shell` at 4M -- either dropped `shell` (`--size 10`) or pulled every arm down to
    4M (`--size 0`, "the largest size EVERY scan carries") and left the headline contrast out.
    (b) `arm_size_m` read `config.yaml`, where `tha_Thai`, `ufw_en`, `python` and `owm` still
    declare 16 while their scans ran `--max-size 10`: four `corp10.mtok` cells at a size nothing
    measured, each with a note explaining a 16M cell the run does not have.

    The fixture uses `tha_Thai` (config says 16, scan says 10) and `shell` (config and scan say
    4), so reading the config and reading the product give different answers on the same table.
    """
    od = _ood()
    cfg = PC.load_config()
    ids, per_target, bo8 = [], {}, {}
    top1_by_arm: dict[str, dict] = {"tha_Thai": {}, "shell": {}}
    row = 0
    for arm in ("tha_Thai", "shell"):
        for _ in range(8):
            ids.append({"row": row, "arm": arm, "family": cfg["ood_arms"][arm]["family"],
                        "doc": 900 + row})
            per_target[row] = {"bo_64": 0.60, "bo_c_64": 0.62, "bo_a_64": 0.61}
            bo8[row] = 0.70
            # EVERY size the scan carries is in the map; only the arm's own top one may be read.
            for s_ in (1.0, 4.0, 10.0, 16.0):
                top1_by_arm[arm][(row, s_)] = {1.0: 0.10, 4.0: 0.20, 10.0: 0.50, 16.0: 0.95}[s_]
            row += 1

    def boot_ci(d):
        m = float(np.mean(d))
        return m, m - 0.01, m + 0.01

    src = R.Source(maemm="qwen36-27b/2026-09-18_rl-last16-lr5e-7", base="qwen36-27b",
                   engine="vllm", run_tag="tag", scores_rel="", rollouts_rel="")
    src.per_target = per_target
    recs, skipped = od.arm_rows(ids, src, top1_by_arm, {"tha_Thai": 10.0, "shell": 4.0},
                                "centred", boot_ci, R_outcome, None, bo8=bo8)
    assert not skipped, skipped
    got = {r["arm"]: r for r in recs}
    assert got["tha_Thai"]["corpus_size_m"] == 10.0 and got["shell"]["corpus_size_m"] == 4.0, got
    # The corpus cell is the one AT THAT SIZE: 0.50 at 10M, 0.20 at 4M. The 16M cell exists in the
    # map and must not be reached -- which is what the config-driven size would have done.
    _close(got["tha_Thai"]["corpus_top1"], 0.50, 1e-9, what="tha_Thai is read at its scan's 10M")
    _close(got["shell"]["corpus_top1"], 0.20, 1e-9, what="shell is read at its scan's 4M")
    _close(got["tha_Thai"]["delta8"], 0.20, 1e-9)
    _close(got["shell"]["delta8"], 0.50, 1e-9)
    # ...and an arm with no size at all is SKIPPED and named, never read at someone else's size.
    _recs, skipped2 = od.arm_rows(ids, src, top1_by_arm, {"tha_Thai": 10.0}, "centred", boot_ci,
                                  R_outcome, None, bo8=bo8)
    assert any("shell" in m and "no corpus size" in m for m in skipped2), skipped2

    # THE CELLS: `corp10.mtok` is the product's size, and the note says so where it is not 10M.
    # `main` joins the lid columns onto every record before `write_cells`; done here too, so the
    # fixture is the shape the writer is really handed.
    for r in recs:
        r.update({"lid_top1_rate": None, "code_like_top1_rate": None, "ceiling": None,
                  "ceiling_kind": "lid", "bpb_ctx": None, "nll_n": None, "control_bo8": None})
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        _path, rows_ = od.write_cells(out_dir, cfg, src, recs, "S", None)
        cells = {r["key"]: r for r in rows_}
        assert cells["ood.tha.corp10.mtok"]["value"] == "10", cells["ood.tha.corp10.mtok"]
        assert cells["ood.shell.corp10.mtok"]["value"] == "4", cells["ood.shell.corp10.mtok"]
        assert "not 10M" in cells["ood.shell.corp10.cos"]["note"], cells["ood.shell.corp10.cos"]
        assert "not 10M" not in cells["ood.tha.corp10.cos"]["note"], cells["ood.tha.corp10.cos"]
        # and every key it writes is one this module is allowed to write
        owned = od.owned_cell_keys(cfg)
        stray = sorted({r["key"] for r in rows_} - owned)
        assert not stray, stray


def check_ood_counts_are_the_bo8_verdict():
    """The "N of M arms exceed" sentence and the `ood.conj.diff.n*` cells are ONE count.

    THE DEFECT: the sentence was built from `r["outcome"]` -- the bo64 pair, what the
    quarter-scale run reported -- while the table's Δ column, `diff.verdict` and `conj.diff.n*`
    are all `outcome8`. Two verdicts of the same arms under one heading, and the reader has no
    way to see that they are different statistics rather than a contradiction.

    The fixture makes them disagree on every arm: bo64 says `exceeds` where bo8 says `reversed`.
    """
    od = _ood()
    recs = [
        {"arm": "ces_Latn", "family": "lang", "outcome": "exceeds", "outcome8": "reversed"},
        {"arm": "python", "family": "code", "outcome": "exceeds", "outcome8": "exceeds"},
        {"arm": "sql", "family": "code", "outcome": "exceeds", "outcome8": "inconclusive"},
        {"arm": "lean", "family": "math", "outcome": "reversed", "outcome8": "no bo8 pairs"},
        # NOT COUNTED: the `diag` arm and the English anchor, reported as rows either way.
        {"arm": "formulas", "family": "diag", "outcome": "exceeds", "outcome8": "exceeds"},
        {"arm": "ufw_en", "family": "ctrl", "outcome": "exceeds", "outcome8": "exceeds"},
    ]
    counts, conj, have = od.conjunction_counts(recs)
    assert [r["arm"] for r in conj] == ["ces_Latn", "python", "sql", "lean"], conj
    assert [r["arm"] for r in have] == ["ces_Latn", "python", "sql"], have
    assert counts == {"exceeds": 1, "inconclusive": 1, "reversed": 1}, counts
    assert sum(1 for r in conj if r["outcome"] == "exceeds") == 3, (
        "the fixture no longer makes bo64 and bo8 disagree, so it proves nothing")
    # the cells rows carry exactly these numbers, over the same denominator
    for r in recs:
        r.update({"n": 8, "n8": 8, "delta8": 0.1, "ci8_lo": 0.0, "ci8_hi": 0.2,
                  "se8_clustered": 0.01, "bo8_centred": 0.7, "corpus_top1": 0.6,
                  "corpus_size_m": 10.0, "lid_top1_rate": None, "code_like_top1_rate": None,
                  "ceiling": None, "ceiling_kind": "lid", "bpb_ctx": None, "nll_n": None,
                  "control_bo8": None})
    with tempfile.TemporaryDirectory() as td:
        src = R.Source(maemm="b/m", base="b", engine="vllm", run_tag="t", scores_rel="",
                       rollouts_rel="")
        _p, rows_ = od.write_cells(Path(td), PC.load_config(), src, recs, "S", None)
        cells = {r["key"]: r for r in rows_}
        assert cells["ood.conj.diff.nexceed"]["value"] == "1", cells["ood.conj.diff.nexceed"]
        assert cells["ood.conj.diff.ninconcl"]["value"] == "1", cells["ood.conj.diff.ninconcl"]
        assert cells["ood.conj.diff.nreversed"]["value"] == "1", cells["ood.conj.diff.nreversed"]
        assert cells["ood.conj.diff.nexceed"]["n"] == "3", cells["ood.conj.diff.nexceed"]


def check_ood_cells_merge_keeps_every_other_writers_bytes():
    """The in-place merge rewrites this module's keys and no other byte of `cells.csv`.

    `results/ood.py` stopped at a fragment CSV; the merge was done by hand, and the M5 run's
    fragment carried `—` and `not comparable` into rows the tex reads. The writer is module M1's
    -- ONE implementation of "preserve every byte", imported -- and this check drives it through
    `ood.merge_cells`: a CRLF file, one `ood.*` row rewritten, one appended, every foreign row
    byte-identical, and a key this module does not own refused before anything is written.
    """
    od = _ood()
    cfg = PC.load_config()
    header = ",".join(od.CELLS_COLUMNS)
    body = [
        'fid.ra.ex.cos.bo8,0.8219,0.0051,,,486,final,R1,src,2026-09-22,"a, quoted note"',
        "ood.tha.corp10.mtok,,,,,,placeholder,R4,,2026-09-23,expected",
        "corp.train10m.top1.cos,0.3512,,,,512,final,R2,src,2026-09-22,",
    ]
    for term in ("\r\n", "\n"):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cells.csv"
            original = term.join([header, *body]) + term
            with path.open("w", newline="") as fh:
                fh.write(original)
            rows = [
                {"key": "ood.tha.corp10.mtok", "value": "10", "se": "", "lo": "", "hi": "",
                 "n": "", "status": "provisional", "run": "R4", "source": "results/ood.py",
                 "date": "2026-09-23", "note": "the size the arm's own scan reached"},
                {"key": "ood.tha.diff.verdict", "value": "exceeds", "se": "", "lo": "", "hi": "",
                 "n": "512", "status": "provisional", "run": "R4", "source": "results/ood.py",
                 "date": "2026-09-23", "note": "three-state verdict on the bo8 pair"},
            ]
            stat = od.merge_cells(path, rows, cfg)
            assert stat["rewritten"] == ["ood.tha.corp10.mtok"], stat
            assert stat["appended"] == ["ood.tha.diff.verdict"], stat
            with path.open(newline="") as fh:
                after = fh.read()
            lines = after.splitlines(keepends=True)
            assert lines[0] == header + term, repr(lines[0])
            # the two rows this module does not own came back byte for byte, quoting included
            assert lines[1] == body[0] + term, repr(lines[1])
            assert lines[3] == body[2] + term, repr(lines[3])
            assert all(ln.endswith(term) for ln in lines), (
                f"the merge mixed line terminators into a {term!r} file")
            assert ",10," in lines[2] and lines[2].startswith("ood.tha.corp10.mtok,"), lines[2]
            assert lines[4].startswith("ood.tha.diff.verdict,"), lines[4]
            # a key outside this module's own is refused BEFORE the file is touched
            before = after
            try:
                od.merge_cells(path, [{**rows[0], "key": "fid.ra.ex.cos.bo8"}], cfg)
            except AssertionError as exc:
                assert "does not own" in str(exc), exc
            else:
                raise AssertionError("the merge wrote a key module M1 owns")
            with path.open(newline="") as fh:
                assert fh.read() == before, "a refused merge still rewrote the file"
    # the owned set is enumerated, so a typo is not silently a new row
    owned = od.owned_cell_keys(cfg)
    assert "ood.tha.corp10.mtok" in owned and "ood.conj.diff.nexceed" in owned
    assert "ood.tha.corp10.mtokens" not in owned and "ood.tha.diff.cos.bo64" not in owned


def check_ood_cells_never_carry_an_em_dash():
    """A cell that was not measured is NOT WRITTEN -- not written as `—`, not as `not comparable`.

    THE DEFECT the M5 refusal would have merged: `write_cells` formatted every value through
    `results.common.num`, whose job in a markdown table is to print an em dash for an absent
    number. `cells.csv` is copied into the tex VERBATIM, so that em dash is a paper that prints
    an em dash, and `make_numbers.py --check` calls "no value but se/lo/hi/n given" a WARNING --
    it would have gone through. The refusal's own fragment carried `—` in all 22
    `ood.*.diff.cos.bo8` cells, `—` in all 22 `corp10.cos`, and `not comparable` in all 22
    `diff.verdict` (runs/2026-09-23_ledger.md).

    The placeholder rows already in the file say "expected, not yet measured", which is true; a
    row that says `—` says something false in the same slot.
    """
    od = _ood()
    cfg = PC.load_config()
    blank = {"n": 8, "n8": 0, "delta8": None, "ci8_lo": None, "ci8_hi": None,
             "se8_clustered": None, "bo8_centred": None, "corpus_top1": None,
             "corpus_size_m": None, "control_bo8": None, "bpb_ctx": None, "nll_n": None,
             "lid_top1_rate": None, "code_like_top1_rate": None, "ceiling": None,
             "ceiling_kind": None}
    recs = [{"arm": "tha_Thai", "family": "lang", "outcome": "not comparable",
             "outcome8": "not comparable", **blank},
            {"arm": "python", "family": "code", "outcome": "exceeds",
             "outcome8": "no bo8 pairs", **blank}]
    src = R.Source(maemm="b/m", base="b", engine="vllm", run_tag="t", scores_rel="",
                   rollouts_rel="")
    with tempfile.TemporaryDirectory() as td:
        _p, rows_ = od.write_cells(Path(td), cfg, src, recs, "S", None)
    assert rows_ == [], f"an unmeasured arm wrote {len(rows_)} cells: {rows_[:3]}"

    # ONE measured cell among absent ones still lands, and nothing rides along with it.
    recs[0].update({"delta8": 0.2, "n8": 512, "se8_clustered": 0.005, "outcome8": "exceeds",
                    "corpus_size_m": 10.0})
    with tempfile.TemporaryDirectory() as td:
        _p, rows_ = od.write_cells(Path(td), cfg, src, recs, "S", None)
    cells = {r["key"]: r for r in rows_}
    assert set(cells) == {"ood.tha.diff.cos.bo8", "ood.tha.diff.verdict", "ood.tha.corp10.mtok",
                          "ood.conj.diff.nexceed", "ood.conj.diff.ninconcl",
                          "ood.conj.diff.nreversed"}, sorted(cells)
    # `lo`/`hi` go in together or not at all: a lone one is an ERROR in `make_numbers.py --check`
    # and blocks the paper build. Here the CI is absent while Δ is not.
    row = cells["ood.tha.diff.cos.bo8"]
    assert row["value"] == "0.2000" and row["lo"] == "" and row["hi"] == "", row
    assert row["se"] == "0.0050", row
    assert all("—" not in v for v in row.values()), row
    # and `not comparable` / `no bo8 pairs` are true statements and not verdicts
    assert cells["ood.tha.diff.verdict"]["value"] == "exceeds"
    assert "ood.python.diff.verdict" not in cells, "`no bo8 pairs` was written as a verdict"


# --- module M1: the paper's cells, the ratio denominator, the corpus comparator ------------------
#
# A FOURTH fixture, at the end of this file: two blocks in the volume's own layout, carrying the
# shapes panel a and panel b need and nothing else. It is separate from the eval-1 fixture above
# rather than grafted onto it because M1's cells need things that fixture deliberately does not
# have -- a realact block stamped `source: upstream`, an arm that is `role: control`, an arm that is
# `type: nla` with its own rollout count, a dictionary whose `sae_key` ends in the name the
# paper's `l131k` key slot means, and eight rollouts so that a bo8 exists at all.
#
# It carries, deliberately:
#
#   * FOUR realact rows over THREE documents, one document shared, so the clustered bootstrap has
#     something to cluster and the paired cells' `n` and document count differ;
#   * an Exemplifier arm at n = 8 and an NLA arm at n = 4, so `bo8` exists on one and not the
#     other -- which is the real shape (spec §1.4: NLA is 4 rollouts, so its bo1 IS the mean of 4)
#     and makes "the NLA row is bo1 and the Exemplifier row is bo8" a fact about the products
#     rather than about which constant the driver happened to pass;
#   * eight SAE features, two per stratum, whose peaks and corpus peaks are exact halves, so every
#     median, ratio and fired fraction below is a literal worked out here and not a second call of
#     the code under test;
#   * a `top1_act` product whose `act_max` is a DIFFERENT number from the stored `corpus_peak`, so
#     a driver that ignored `--corpus-peak` and divided by the stored one cannot pass;
#   * strata whose statistic is `log10_pool_peak_act` -- what the 131k draw actually cuts on, which
#     is not the corpus frequency the spec's key names.

M1_BASE = "MB"
M1_RA = "MRA"        # the upstream realact block
M1_CT = "MCTRL"      # the random floor and the 131k-shaped feature block
M1_SAE = f"{M1_BASE}/l42-1b"
M1_N = 8             # rollouts on the MAEMM arms
M1_NLA_N = 4         # rollouts on the NLA arm (spec §1.4)
M1_GATE = 1.5846     # spec §1.2 and §4 panel b name this to 4 decimals on the 131k

M1_RA_IDS = [
    {"row": 0, "family": "realact", "source": "upstream", "doc": 1, "id": "d1:p1"},
    {"row": 1, "family": "realact", "source": "upstream", "doc": 1, "id": "d1:p9"},
    {"row": 2, "family": "realact", "source": "upstream", "doc": 2, "id": "d2:p3"},
    {"row": 3, "family": "realact", "source": "upstream", "doc": 3, "id": "d3:p7"},
]
M1_CT_IDS = (
    [{"row": 0, "family": "random", "id": "rnd0"},
     {"row": 1, "family": "random", "id": "rnd1"}]
    # NO `sae_side`: the 131k draw does not stamp one (the committed `sae/l42-1b` aggregates
    # carry an empty side), and `CELL_DICTIONARIES` maps the paper's `l131k` slot onto exactly
    # that shape -- a draw that started stamping `enc` would produce NO cells and say so, rather
    # than the 2M block's numbers under the 131k key.
    + [{"row": 2 + i, "family": "sae", "sae_key": M1_SAE,
        "stratum": i // 2, "stratum_stat": "log10_pool_peak_act", "id": str(1000 + i)}
       for i in range(8)]
)

# The Exemplifier's centred per-rollout bests. Row 0 is the only one that varies, so bo1 and bo8
# are different numbers there and a bo8 computed as a mean (or a bo1 computed as a max) fails.
M1_EX_C = {0: [0.25 + i / 16 for i in range(M1_N)], 1: [0.5] * M1_N,
           2: [0.25] * M1_N, 3: [0.75] * M1_N}
M1_CTL_C = {r: [0.0625] * M1_N for r in range(4)}          # the untrained-base control
# n = 4, so bo1 is the mean of 4. Row 0 VARIES with the same mean (0.125), so bo1 is unchanged
# while bo2 and bo4 differ from it there: a bo2 taken as a mean, or as the max, fails (M8).
M1_NLA_C = {0: [0.0, 0.0625, 0.1875, 0.25], **{r: [0.125] * M1_NLA_N for r in (1, 2, 3)}}
# unbiased bo2 of row 0: sorted x_(i) weighted C(i-1, 1)/C(4, 2) = (0, 1, 2, 3)/6
M1_NLA_BO2_ROW = {0: (0.0625 + 2 * 0.1875 + 3 * 0.25) / 6, 1: 0.125, 2: 0.125, 3: 0.125}
M1_NLA_BO2 = sum(M1_NLA_BO2_ROW.values()) / 4        # 0.1432291...
M1_NLA_BO4 = (0.25 + 3 * 0.125) / 4                  # 0.15625: bo4 of 4 draws is the max
M1_RND = {0: [0.0625] * M1_N, 1: [0.0625] * M1_N}          # raw only: `random` is not centrable

# bo1 of row 0 = 0.25 + (0+1+..+7)/(16*8) = 0.46875; bo8 of row 0 = its max = 0.6875.
M1_EX_BO1 = (0.46875 + 0.5 + 0.25 + 0.75) / 4       # 0.4921875
M1_EX_BO8 = (0.6875 + 0.5 + 0.25 + 0.75) / 4        # 0.546875
M1_EX_BO8_ROW = {0: 0.6875, 1: 0.5, 2: 0.25, 3: 0.75}
# The corpus comparator's top-1 per target, as M2's `read_top1` reads it off a scan `topk.jsonl`
# written into the mirror below. NOT a stub: the driver's contract is M2's real reader, and a fake
# module standing in for it is exactly the thing that let the two disagree unnoticed.
M1_CORPUS = {0: 0.5, 1: 0.25, 2: 0.5, 3: 0.5}
# The SAME rows on a DIFFERENT corpus, in a second scan directory. The upstream block is scanned against
# two corpora under one run tag and the driver must read the one it was told to; every value here
# is higher, so reading the wrong directory flips the win fraction as well as the difference.
M1_CORPUS_OTHER = {r: v + 0.125 for r, v in M1_CORPUS.items()}
M1_CORPUS_KEY = "train10m__tag"          # the corpus key `--corpus-scan` takes
M1_CORPUS_KEY_OTHER = "held16m__tag"
M1_SIZES = (5.0, 10.0)                   # the nested ladder; the cells are specified at 10M
M1_DIFF = (0.1875 + 0.25 - 0.25 + 0.25) / 4         # 0.109375
M1_WIN = 3 / 4
M1_NLA_DEX = (0.5625 + 0.375 + 0.125 + 0.625) / 4   # 0.421875

# The features: two per stratum. `A` fires at every stratum, `B` crosses the gate at stratum 2.
M1_PEAK_A = 2.0
M1_PEAK_B = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5}
M1_STORED_PEAK = 4.0        # what `sae_self` recorded: our 16M held-out max_act
M1_TOP1_PEAK = 2.0          # what M2's top1_act says on the 10M training corpus
# medians over the two features of a stratum, under each denominator
M1_RATIO_STORED = {s: (M1_PEAK_A / M1_STORED_PEAK + M1_PEAK_B[s] / M1_STORED_PEAK) / 2
                   for s in range(4)}
M1_RATIO_TOP1 = {s: (M1_PEAK_A / M1_TOP1_PEAK + M1_PEAK_B[s] / M1_TOP1_PEAK) / 2
                 for s in range(4)}
M1_FIRED = {s: (1.0 + (1.0 if M1_PEAK_B[s] > M1_GATE else 0.0)) / 2 for s in range(4)}

M1_CFG = {
    "bases": {M1_BASE: {"d": 4, "read_layer": 1}},
    "family_kinds": {
        "realact": {"centrable": True, "kind": "activation"},
        "random": {"centrable": False, "kind": "synthetic"},
        "sae": {"centrable": False, "kind": "dictionary"},
    },
    "heldout": {
        M1_RA: {"base": M1_BASE, "families": {"realact": {"n": 4}}},
        M1_CT: {"base": M1_BASE, "sae_key": M1_SAE,
                "families": {"random": {"n": 2}, "sae": {"n": 8}}},
    },
    "maemms": {
        f"{M1_BASE}/ex-ckpt": {"type": "full", "mu": "/mu.npy"},
        f"{M1_BASE}/ctl-ckpt": {"type": "base", "role": "control", "mu": "/mu.npy"},
        f"{M1_BASE}/nla-ckpt": {"type": "nla"},
    },
}


def _m1_write_arm(root: Path, maemm: str, set_name: str, ids: list[dict], n: int,
                  raw: dict, centred: dict | None, sae: dict | None = None,
                  gate: float = M1_GATE, corpus_peak: float = M1_STORED_PEAK) -> None:
    """One arm's scores directory: per_target, rows.json, the centred array and `sae_self`."""
    d = root / f"maemms/{maemm}/scores/{set_name}__vllm"
    d.mkdir(parents=True, exist_ok=True)
    rows = [r["row"] for r in ids]
    with open(d / "per_target.jsonl", "w") as fh:
        for r in rows:
            pt = {"row": r, "family": ids[r]["family"], "n": n,
                  **{f"bo_{k}": R.bo_unbiased(raw[r], k) for k in R.BO_KS_ALL if k <= n},
                  "mean_cos": float(np.mean(raw[r])), "max_cos": float(np.max(raw[r]))}
            fh.write(json.dumps(pt) + "\n")
    (d / "rows.json").write_text(json.dumps(
        {"rows": rows, "n": n, "families": [ids[r]["family"] for r in rows],
         "score_max_length": WIDTH - 1, "mu": "/mu.npy" if centred is not None else None}))
    index: dict = {"per_target.jsonl": {"kind": "jsonl", "rows": len(rows)},
                   "rows.json": {"kind": "json"}}
    if centred is not None:
        blocks = [_block(centred[r], WIDTH) if r in centred
                  else np.full((n, WIDTH), np.nan, dtype=np.float32) for r in rows]
        arr = np.stack(blocks).astype(np.float16)
        arr.tofile(d / "cos_centred.f16")
        index["cos_centred.f16"] = {"kind": "array", "dtype": "float16",
                                    "shape": list(arr.shape), "bytes": arr.nbytes}
    (d / "index.json").write_text(json.dumps(index))
    if sae is None:
        return
    sd = d / "sae_self"
    sd.mkdir(exist_ok=True)
    sae_rows = sorted(sae)
    act = np.stack([_block(sae[r], WIDTH) for r in sae_rows]).astype(np.float16)
    act.tofile(sd / "sae_self.f16")
    (sd / "sae_self.json").write_text(json.dumps({
        "rows": sae_rows, "features": [int(ids[r]["id"]) for r in sae_rows], "n": n,
        "width": WIDTH, "gate": gate,
        "per_target": [{"row": r, "feature": int(ids[r]["id"]), "n": n,
                        "corpus_peak": corpus_peak,
                        "mean_peak_act": float(np.mean(sae[r])),
                        "max_peak_act": float(np.max(sae[r]))} for r in sae_rows],
    }))


def _m1_mirror(root: Path, gate: float = M1_GATE) -> None:
    for set_name, ids in ((M1_RA, M1_RA_IDS), (M1_CT, M1_CT_IDS)):
        hd = root / f"base/{M1_BASE}/heldout/{set_name}"
        hd.mkdir(parents=True, exist_ok=True)
        with open(hd / "ids.jsonl", "w") as fh:
            for r in ids:
                fh.write(json.dumps(r) + "\n")
        (hd / "storage.json").write_text(json.dumps(
            {"storage": "raw", "sae_key": M1_SAE if set_name == M1_CT else ""}))
    # the upstream realact block: the Exemplifier, the untrained-base control, the NLA arm
    raw_ex = {r: [v + 0.125 for v in M1_EX_C[r]] for r in M1_EX_C}
    _m1_write_arm(root, f"{M1_BASE}/ex-ckpt", M1_RA, M1_RA_IDS, M1_N, raw_ex, M1_EX_C)
    _m1_write_arm(root, f"{M1_BASE}/ctl-ckpt", M1_RA, M1_RA_IDS, M1_N,
                  {r: [0.1875] * M1_N for r in range(4)}, M1_CTL_C)
    _m1_write_arm(root, f"{M1_BASE}/nla-ckpt", M1_RA, M1_RA_IDS, M1_NLA_N,
                  {r: [0.25] * M1_NLA_N for r in range(4)}, M1_NLA_C)
    # the two corpus-search scans: the one the paired cells are specified against, and the
    # held-out one beside it that the driver must NOT read unless it is named
    _m1_scan(root, M1_CORPUS_KEY, M1_CORPUS)
    _m1_scan(root, M1_CORPUS_KEY_OTHER, M1_CORPUS_OTHER)
    # the control block: the random floor (raw only) and the feature block
    peaks = {}
    for i in range(8):
        peaks[2 + i] = [M1_PEAK_A if i % 2 == 0 else M1_PEAK_B[i // 2]] * M1_N
    raw_ct = {0: M1_RND[0], 1: M1_RND[1], **{2 + i: [0.03125] * M1_N for i in range(8)}}
    _m1_write_arm(root, f"{M1_BASE}/ex-ckpt", M1_CT, M1_CT_IDS, M1_N, raw_ct, None,
                  sae=peaks, gate=gate)


def _m1_scan(root: Path, key: str = M1_CORPUS_KEY, top1: dict | None = None,
             sizes=M1_SIZES) -> str:
    """One corpus-search scan in `precompute/scan.py`'s own layout, and its directory name.

    `topk.jsonl` is one line per (target row, corpus size) with `top` = up to 64 entries
    `[doc, window start, argmax within the window, cos]`, rank 0 being the best. The join key is
    (`set`, `set_row`) and never `row`, because `--with-set` re-indexes `row` to the position
    WITHIN the scan -- so this fixture writes `row` SHIFTED by ten and a second set's rows ahead
    of upstream, and a reader that joined on `row` reads the wrong bank's numbers.
    """
    top1 = M1_CORPUS if top1 is None else top1
    d = root / f"base/{M1_BASE}/scan/{M1_RA}__{key}"
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for si, size in enumerate(sizes):
        for r in range(10):          # ten rows of ANOTHER set, sitting at scan rows 0..9
            lines.append({"row": si * 14 + r, "set": "OTHER", "set_row": r,
                          "family": "realact", "arm": None, "corpus": key, "size": size,
                          "top": [[99, 0, 0, 0.999]]})
        for r in sorted(top1):
            # the 5M rung sits BELOW the 10M one, which is what a nested ladder looks like
            cos = round(top1[r] - (0.0625 if size != 10.0 else 0.0), 5)
            lines.append({"row": si * 14 + 10 + r, "set": M1_RA, "set_row": r,
                          "family": "realact", "arm": None, "corpus": key, "size": size,
                          "top": [[M1_RA_IDS[r]["doc"], 8 * r, 3, cos], [0, 0, 0, cos - 0.5]]})
    with open(d / "topk.jsonl", "w") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return f"{M1_RA}__{key}"


def _m1_top1_act(root: Path, rel: str, act_max: float = M1_TOP1_PEAK,
                 set_name: str = M1_CT, sae_key: str = M1_SAE, summary: bool = True) -> str:
    """M2's `top1_act` product on the 10M training corpus, in its own layout.

    `set_name` / `sae_key` are what the product's `summary.json` says its ROW NUMBERS index. They
    are parameters because a product of another set carries the same row numbers and must not be
    allowed to answer for this one.
    """
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "top1_act.jsonl", "w") as fh:
        for i in range(8):
            fh.write(json.dumps({"row": 2 + i, "feature": 1000 + i, "stratum": i // 2,
                                 "act_max": act_max, "gate": M1_GATE, "size": 10}) + "\n")
    if summary:
        (d / "summary.json").write_text(json.dumps(
            {"corpus_size_m": 10, "sae": sae_key, "set": set_name, "n_features": 8}))
    return rel


def _m1_exclude(root: Path, rows: list[int]) -> None:
    """The set's own exclusions.json, in the layout `load_exclusions` reads."""
    (root / f"base/{M1_BASE}/heldout/{M1_RA}/exclusions.json").write_text(json.dumps({
        "block": "realact", "rows_total": len(M1_RA_IDS), "excluded_rows": rows,
        "n_headline": len(M1_RA_IDS) - len(rows), "criterion": "a synthetic gate", "n_gram": 7,
        "source": "selftest", "computed_over": "the M1 fixture"}))


def _m1_analyse(root: Path, corpus_peak: str = F.CORPUS_PEAK_STORED):
    vol = R.Vol("", root, offline=True, quiet=True)
    return vol, [F.analyse(vol, M1_CFG, s, "", 400, 1, True, 8.0, 128.0, True, corpus_peak)
                 for s in (M1_RA, M1_CT)]


def _m1_opts(**kw) -> dict:
    opts = {"boot": 400, "seed": 1, "date": "2026-09-24", "exemplifier": "ex-ckpt",
            "corpus_size": 10.0, "owned": F.M1_KEYS, "corpus_peak": F.CORPUS_PEAK_STORED,
            "corpus_scan": M1_CORPUS_KEY}
    opts.update(kw)
    return opts


def _m1_cells(root: Path, *, corpus: bool, corpus_peak: str = F.CORPUS_PEAK_STORED, **kw):
    """`paper_cells` with M2's REAL `results.corpus_search` behind it (`corpus=False` removes it).

    The comparator is not stubbed. `corpus_top1_for` calls `read_top1` and `assert_complete` on
    the scan written by `_m1_scan`, so what this exercises is the contract between the two files
    -- which is the thing that was broken: the driver looked for exports M2 never had, and every
    run reported the paired cells ABSENT while the scan sat on the volume.
    """
    vol, all_res = _m1_analyse(root, corpus_peak)
    before = F.CORPUS_SEARCH
    F.CORPUS_SEARCH = CS if corpus else None
    try:
        return F.paper_cells(vol, all_res, M1_CFG, _m1_opts(corpus_peak=corpus_peak, **kw))
    finally:
        F.CORPUS_SEARCH = before


def _m1_by_key(rows: list[dict]) -> dict[str, dict]:
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), f"the cell builder emitted a key twice: {keys}"
    return {r["key"]: r for r in rows}


def check_m1_clusters_fire_on_upstream_block():
    """The clustered bootstrap must FIRE on the upstream block, and must reduce to the plain SE without it.

    M1's job on the estimator is not to write it -- `R.cluster_bootstrap` has existed since M0a
    and `check_cluster_bootstrap` covers its arithmetic -- but to verify it is actually CLUSTERING
    on the upstream rows. `_clusters_for` falls back to a singleton `f"row{r}"` for a row with no `doc`, so
    a block whose `doc` field went missing would get a plain std/sqrt(n) printed under a clustered
    name, with nothing in the output saying so. The upstream block is 486 rows over 425 documents, so the
    two numbers differ and that difference is the check.

    BOTH DIRECTIONS, as the plan asks: one row per document reproduces `se_iid`, the upstream block's real
    structure does not. The mutation is the third: strip `doc` from the ids -- exactly what a
    products change or a re-draw could do -- and require the clustered SE to collapse onto the
    iid one, which is the failure this exists to catch.
    """
    rng = np.random.default_rng(20260924)
    # The upstream block's shape: 486 rows over 425 documents, so 61 rows share one with another row.
    n_docs, n_rows = 425, 486
    docs = list(range(n_docs)) + [d % n_docs for d in range(n_rows - n_docs)]
    # Correlated WITHIN a document -- which is why the clustering matters at all.
    per_doc = rng.normal(size=n_docs)
    vals = [float(per_doc[d] + 0.05 * rng.normal()) for d in docs]
    ids = {r: {"doc": docs[r], "family": "realact"} for r in range(n_rows)}
    rows = list(range(n_rows))

    cl = F._clusters_for(rows, ids)
    assert len(set(cl)) == n_docs, f"_clusters_for found {len(set(cl))} clusters, not {n_docs}"
    _m, se_cl, n_items, n_cl = R.cluster_bootstrap(vals, cl, 4000, 1)
    assert (n_items, n_cl) == (n_rows, n_docs), (n_items, n_cl)
    iid = R.se_iid(vals)

    # (1) one row per document IS the ordinary bootstrap: hand it singletons and it reproduces
    # std/sqrt(n) to bootstrap noise.
    _m1, se_single, _, c1 = R.cluster_bootstrap(vals, [f"row{r}" for r in rows], 4000, 1)
    assert c1 == n_rows
    _close(se_single, iid, tol=0.1 * iid, what="singleton clusters must reproduce se_iid")

    # (2) the upstream block's real structure does NOT: 61 shared documents move the SE measurably.
    assert abs(se_cl - iid) > 0.02 * iid, (
        f"the clustered SE {se_cl:.6f} is within 2% of the plain {iid:.6f} on a block with "
        f"{n_rows - n_docs} rows sharing a document -- the bootstrap is not clustering")

    # (3) THE MUTATION: drop `doc` and the fallback silently makes every row a singleton.
    broken = {r: {"family": "realact"} for r in range(n_rows)}
    cl_broken = F._clusters_for(rows, broken)
    assert len(set(cl_broken)) == n_rows, "the singleton fallback did not fire"
    _m2, se_broken, _, _ = R.cluster_bootstrap(vals, cl_broken, 4000, 1)
    assert abs(se_broken - iid) <= 0.1 * iid, (
        "with `doc` stripped the clustered SE must collapse onto the plain one -- if it does not, "
        "check (2) above is not measuring what it claims")
    assert abs(se_broken - se_cl) > 0.02 * iid, (
        "the mutation changed nothing: a block with no documents and the upstream block produced the same "
        "SE, so this check could never have gone red")


def check_m1_fired_bok_mutation():
    """The fired best-of-k identity, and a deliberately wrong estimator that must FAIL it.

    `check_fired_is_the_same_estimator` asserts the identity holds. This asserts the assertion
    can fail: the same comparison against the PLAIN SAMPLE MEAN -- which is what "fraction fired"
    meant before the indicator went through the one bo-k estimator, and the obvious wrong answer
    at every k > 1 -- must go red on the same inputs. A check that has never been red is
    unevaluated.
    """
    import math as _m

    def identity(n: int, m: int, k: int) -> float:
        return 1.0 - (_m.comb(n - m, k) / _m.comb(n, k) if n - m >= k else 0.0)

    rng = np.random.default_rng(11)
    caught = 0
    for n, m in ((8, 1), (8, 3), (16, 5), (64, 7)):
        ind = np.concatenate([np.ones(m), np.zeros(n - m)])
        rng.shuffle(ind)
        lad = R.bo_ladder(ind, (1, 2, 4, 8))
        for k, got in lad.items():
            _close(got, identity(n, m, k), tol=1e-12,
                   what=f"fired bo{k} of {m}/{n} is 1 - C(n-m,k)/C(n,k)")
        # THE MUTATION: the plain rate at every k. It is right at k = 1 and wrong above it.
        wrong = {k: float(ind.mean()) for k in lad}
        bad = [k for k in lad if abs(wrong[k] - identity(n, m, k)) > 1e-12]
        assert bad, f"n={n} m={m}: the plain mean matched the identity at every k, so the test "
        caught += len(bad)
        # and at k = 1 the two agree, which is why only the ladder above it is evidence
        _close(wrong[1], identity(n, m, 1), tol=1e-12, what="both estimators agree at k = 1")
    assert caught >= 8, caught


def check_m1_cells_writer_rewrites_in_place():
    """The writer rewrites its own keys, leaves every other row BYTE-IDENTICAL, and refuses a dup.

    `paper/numbers/cells.csv` is shared with other writers. The failure that has no symptom
    is a writer that round-trips the whole file through a CSV library: the diff then shows every
    row whose quoting the library spells differently, the one real change is invisible in it, and
    nobody reviews what actually moved. So the untouched rows are compared BYTE for byte, not
    field by field, and the fixture deliberately includes a row whose `note` carries a comma (and
    is therefore quoted) and a row with an empty `value`.
    """
    fixture = (
        "key,value,se,lo,hi,n,status,run,source,date,note\n"
        "fid.ra.ex.cos.bo1,0.7590,0.0060,,,486,provisional,R1,old/path.md,2026-09-21,"
        '"rl-last16; the upstream block, centred"\n'
        # ANOTHER BUILDER'S ROW, spelled with a quoted field that does NOT need quoting. Today's
        # cells.csv happens to round-trip through `csv.writer` unchanged on all 413 records
        # (measured 2026-09-24), so byte-identity and field-identity cannot be told apart on it --
        # and the guarantee this writer makes is byte-identity. One row that a round-trip WOULD
        # respell is what makes that claim testable, and another builder's writer using QUOTE_ALL,
        # or one hand edit, produces exactly this.
        'ai.l131k.c16.det,0.8100,,,,512,provisional,"R5",someone/else.md,2026-09-22,'
        "another builder\n"
        "fid.ra.gcg.initgap,,,,,,placeholder,R10,,2026-09-23,retired; row kept empty\n"
        "sae.l131k.ex.fired.bo1,0.9316,,,,,provisional,R1,old/path.md,2026-09-21,pooled\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cells.csv"
        p.write_text(fixture)
        before = p.read_text().splitlines(keepends=True)

        rows = [F.cell("fid.ra.ex.cos.bo1", 0.4921875, se=0.01, n=486, run="R1",
                       source="maemms/x/scores/y", date="2026-09-24", note="new, with a comma"),
                F.cell("fid.ra.ex.win.bo8", 0.75, n=486, run="R1+R2",
                       source="maemms/x/scores/y", date="2026-09-24", note="appended")]
        rec = F.write_cells(p, rows, F.M1_KEYS)
        after = p.read_text().splitlines(keepends=True)

        assert rec["rewritten"] == ["fid.ra.ex.cos.bo1"], rec
        assert rec["appended"] == ["fid.ra.ex.win.bo8"], rec
        # BYTE-IDENTICAL: the header and every row this module does not own.
        assert after[0] == before[0], (after[0], before[0])
        for i in (2, 3, 4):
            assert after[i] == before[i], (i, after[i], before[i])
        assert len(after) == len(before) + 1, (len(after), len(before))
        # the rewritten row carries the new digits, at the printed precision, and stays in place
        assert after[1].startswith("fid.ra.ex.cos.bo1,0.4922,0.0100,,,486,final,R1,"), after[1]
        assert after[-1].startswith("fid.ra.ex.win.bo8,0.7500,"), after[-1]
        # and a comma in the note is quoted, so the record still has eleven fields
        import csv as _csv
        recs = list(_csv.reader(p.read_text().splitlines()))
        assert all(len(r) == len(F.CELLS_COLUMNS) for r in recs), recs

        # a second run is idempotent on the rewritten row and appends nothing new
        rec2 = F.write_cells(p, rows, F.M1_KEYS)
        assert rec2["appended"] == [] and sorted(rec2["rewritten"]) == sorted(
            ["fid.ra.ex.cos.bo1", "fid.ra.ex.win.bo8"]), rec2

        # THE MUTATIONS, all four of them, each of which would otherwise reach the paper build.
        try:
            F.write_cells(p, rows + [rows[0]], F.M1_KEYS)
        except AssertionError as exc:
            assert "more than once" in str(exc), exc
        else:
            raise AssertionError("the writer accepted the same key twice")
        try:
            F.write_cells(p, rows + [F.cell("ai.l131k.c16.det", 0.5)], F.M1_KEYS)
        except AssertionError as exc:
            assert "does not own" in str(exc), exc
        else:
            raise AssertionError("the writer accepted another builder's key")
        dup = Path(td) / "dup.csv"
        dup.write_text(fixture + "fid.ra.ex.cos.bo1,0.1,,,,,final,R1,,2026-09-24,a second copy\n")
        try:
            F.write_cells(dup, rows[:1], F.M1_KEYS)
        except AssertionError as exc:
            assert "already carries" in str(exc), exc
        else:
            raise AssertionError("the writer accepted a file that already had a duplicate key")
        bad = Path(td) / "bad.csv"
        bad.write_text(fixture.replace("key,value,se", "value,key,se", 1))
        try:
            F.write_cells(bad, rows[:1], F.M1_KEYS)
        except AssertionError as exc:
            assert "has columns" in str(exc), exc
        else:
            raise AssertionError("the writer accepted a reordered header")
        # a lone `lo` is an ERROR in make_numbers --check, refused where it is built
        try:
            F.cell("fid.ra.diff.cos.bo8", 0.1, lo=0.0)
        except AssertionError as exc:
            assert "written together" in str(exc), exc
        else:
            raise AssertionError("cell() accepted a lone `lo`")

        # `--cells` resolution: empty is READ-ONLY, a real path is taken, a typo fails LOUDLY and
        # names what it looked at rather than creating a new file somewhere harmless-looking.
        assert F.resolve_cells_path("") is None and F.resolve_cells_path("  ") is None
        assert F.resolve_cells_path(str(p)) == p
        try:
            F.resolve_cells_path(str(Path(td) / "nope.csv"))
        except AssertionError as exc:
            assert "is not a file" in str(exc), exc
        else:
            raise AssertionError("--cells accepted a path that is not there")


def check_m1_panel_a_cells():
    """Panel a's rows, against numbers worked out in the fixture above.

    Every cell is `cos_centred` and every cell is over the upstream block. The Exemplifier's bo1 and bo8
    differ here by construction, so a driver that read one k and labelled it the other fails; the
    NLA arm has four rollouts and therefore no bo8 at all, which is why its key is `bo1`.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)
        rows, skipped = _m1_cells(root, corpus=True)
        by = _m1_by_key(rows)

        assert by["fid.ra.ex.cos.bo1"]["value"] == f"{M1_EX_BO1:.4f}", by["fid.ra.ex.cos.bo1"]
        assert by["fid.ra.ex.cos.bo8"]["value"] == f"{M1_EX_BO8:.4f}", by["fid.ra.ex.cos.bo8"]
        assert by["fid.ra.base.cos.bo8"]["value"] == "0.0625", by["fid.ra.base.cos.bo8"]
        assert by["fid.ra.nla.cos.bo1"]["value"] == "0.1250", by["fid.ra.nla.cos.bo1"]
        assert by["fid.rnd.ex.cos.bo1"]["value"] == "0.0625", by["fid.rnd.ex.cos.bo1"]
        # the run ids of spec §7, and `n` is the SURVIVING row count
        assert by["fid.ra.ex.cos.bo1"]["run"] == "R1"
        assert by["fid.ra.base.cos.bo8"]["run"] == "R10"
        assert by["fid.ra.ex.cos.bo1"]["n"] == "4", by["fid.ra.ex.cos.bo1"]
        assert by["fid.rnd.ex.cos.bo1"]["n"] == "2", by["fid.rnd.ex.cos.bo1"]
        # every realact cell says it is centred; the floor says LOUDLY that it is not
        for k in ("fid.ra.ex.cos.bo1", "fid.ra.ex.cos.bo8", "fid.ra.base.cos.bo8",
                  "fid.ra.nla.cos.bo1"):
            assert "cos_centred" in by[k]["note"], by[k]
        assert "NOT CENTRED" in by["fid.rnd.ex.cos.bo1"]["note"], by["fid.rnd.ex.cos.bo1"]

        # the paired cells: computed row by row over the shared targets, then aggregated
        d = by["fid.ra.diff.cos.bo8"]
        assert d["value"] == f"{M1_DIFF:.4f}", d
        assert d["lo"] and d["hi"] and float(d["lo"]) < float(d["value"]) < float(d["hi"]), d
        assert d["n"] == "4", d
        assert by["fid.ra.ex.win.bo8"]["value"] == f"{M1_WIN:.4f}", by["fid.ra.ex.win.bo8"]
        # and the Exemplifier-minus-NLA quantity is its OWN key, not `diff` (writing plan §2)
        assert by["fid.ra.nla.dex"]["value"] == f"{M1_NLA_DEX:.4f}", by["fid.ra.nla.dex"]
        assert "minus NLA bo1" in by["fid.ra.nla.dex"]["note"], by["fid.ra.nla.dex"]
        # M8: NLA at bo2 and bo4, beside bo1, with an interval; and the paired cells at bo2
        nb2, nb4 = by["fid.ra.nla.cos.bo2"], by["fid.ra.nla.cos.bo4"]
        assert nb2["value"] == f"{M1_NLA_BO2:.4f}", nb2
        assert nb4["value"] == f"{M1_NLA_BO4:.4f}", nb4
        assert nb2["value"] not in (by["fid.ra.nla.cos.bo1"]["value"], nb4["value"]), nb2
        assert nb2["n"] == "4" and nb2["lo"] and nb2["hi"], nb2
        assert float(nb2["lo"]) < float(nb2["value"]) < float(nb2["hi"]), nb2
        assert not by["fid.ra.nla.cos.bo1"]["lo"], by["fid.ra.nla.cos.bo1"]
        assert "cos_centred" in nb2["note"] and "best-of-2" in nb2["note"], nb2
        want_dex2 = sum(M1_EX_BO8_ROW[r] - M1_NLA_BO2_ROW[r] for r in range(4)) / 4
        assert by["fid.ra.nla.dex.bo2"]["value"] == f"{want_dex2:.4f}", by["fid.ra.nla.dex.bo2"]
        assert "minus NLA bo2" in by["fid.ra.nla.dex.bo2"]["note"], by["fid.ra.nla.dex.bo2"]
        want_dc2 = sum(M1_NLA_BO2_ROW[r] - M1_CORPUS[r] for r in range(4)) / 4
        assert by["fid.ra.nla.dcorp.bo2"]["value"] == f"{want_dc2:.4f}", by["fid.ra.nla.dcorp.bo2"]
        assert by["fid.ra.nla.win.bo2"]["value"] == "0.0000", by["fid.ra.nla.win.bo2"]
        assert "fid.ra.nla.dcorp.bo1" not in by and "fid.ra.nla.win.bo1" not in by
        # the SE is the clustered one: four rows over three documents, so it is not std/sqrt(4)
        assert d["se"] and float(d["se"]) > 0, d
        assert "3 documents" in d["note"], d["note"]
        assert not [s for s in skipped if "fid.ra.ex.cos" in s], skipped

        # THE EXCLUSIONS REACH THE PAIRED CELLS TOO. `res["centred"]` is one read of one array,
        # keyed by source and covering every row the product scored, so a paired cell built
        # straight off it would be the full draw sitting beside a post-exclusion mean -- the
        # headline n saying 486 and the difference beside it computed over 512, with nothing in
        # the output saying so. Row 2 is excluded here: it is the one target the Exemplifier
        # LOSES on, so dropping it moves the difference, the win fraction AND the aggregate mean,
        # and a reader that cut one surface and not another cannot produce a consistent triple.
        _m1_exclude(root, [2])
        rows_x, _sk = _m1_cells(root, corpus=True)
        bx = _m1_by_key(rows_x)
        keep = [0, 1, 3]
        assert bx["fid.ra.diff.cos.bo8"]["n"] == "3", bx["fid.ra.diff.cos.bo8"]
        assert bx["fid.ra.ex.win.bo8"]["value"] == "1.0000", bx["fid.ra.ex.win.bo8"]
        want = sum(M1_EX_BO8_ROW[r] - M1_CORPUS[r] for r in keep) / len(keep)
        assert bx["fid.ra.diff.cos.bo8"]["value"] == f"{want:.4f}", bx["fid.ra.diff.cos.bo8"]
        # and the aggregate cells moved with them, off the SAME family map
        want_bo8 = sum(M1_EX_BO8_ROW[r] for r in keep) / len(keep)
        assert bx["fid.ra.ex.cos.bo8"]["value"] == f"{want_bo8:.4f}", bx["fid.ra.ex.cos.bo8"]
        assert bx["fid.ra.ex.cos.bo8"]["n"] == "3", bx["fid.ra.ex.cos.bo8"]
        assert bx["fid.ra.nla.dex"]["n"] == "3", bx["fid.ra.nla.dex"]
        assert bx["fid.ra.nla.dex.bo2"]["n"] == "3", bx["fid.ra.nla.dex.bo2"]
        assert bx["fid.ra.nla.cos.bo2"]["n"] == "3", bx["fid.ra.nla.cos.bo2"]
        want_nb2 = sum(M1_NLA_BO2_ROW[r] for r in keep) / len(keep)
        assert bx["fid.ra.nla.cos.bo2"]["value"] == f"{want_nb2:.4f}", bx["fid.ra.nla.cos.bo2"]


def check_m1_panel_b_cells_and_the_gate():
    """Panel b: per quartile, never pooled where the key has a quartile, and the gate ASSERTED.

    The fixture's two features per stratum give a median that is a literal, and stratum 2 is where
    the second feature crosses the gate -- so the fired fractions are 0.5, 0.5, 1.0, 1.0 and a
    driver that pooled the strata, or numbered them from the wrong end, cannot produce them.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)
        rows, _skipped = _m1_cells(root, corpus=False)
        by = _m1_by_key(rows)

        for s in range(4):
            q = f"q{s + 1}"
            assert by[f"sae.l131k.ex.ratio.bo1.{q}"]["value"] == f"{M1_RATIO_STORED[s]:.4f}", q
            assert by[f"sae.l131k.ex.ratio.bo8.{q}"]["value"] == f"{M1_RATIO_STORED[s]:.4f}", q
            assert by[f"sae.l131k.ex.fired.bo1.{q}"]["value"] == f"{M1_FIRED[s]:.4f}", q
            assert by[f"sae.l131k.ex.fired.bo8.{q}"]["value"] == f"{M1_FIRED[s]:.4f}", q
            assert by[f"sae.l131k.ex.ratio.bo1.{q}"]["n"] == "2"
            assert f"stratum {s} of 0..3" in by[f"sae.l131k.ex.fired.bo1.{q}"]["note"]
            # the stratum STATISTIC is named, and it is not corpus frequency
            assert "log10_pool_peak_act" in by[f"sae.l131k.ex.ratio.bo1.{q}"]["note"]
        # q1 is the RAREST: stratum 0, the one whose second feature does not fire
        assert float(by["sae.l131k.ex.fired.bo1.q1"]["value"]) == 0.5
        assert float(by["sae.l131k.ex.fired.bo1.q4"]["value"]) == 1.0
        # bo64 is SKIPPED, never clamped: the fixture has eight rollouts
        assert "sae.l131k.ex.ratio.bo64.q1" not in by, sorted(by)
        # the three pooled rows that already exist are rewritten, and SAY they are pooled
        for k in ("sae.l131k.ex.ratio.bo1", "sae.l131k.ex.fired.bo1"):
            assert "POOLED across strata" in by[k]["note"], by[k]
            assert by[k]["n"] == "8", by[k]
        # every ratio cell carries the DENOMINATOR's provenance
        for k, r in by.items():
            if ".ratio." in k:
                assert "16M HELD-OUT" in r["note"], (k, r["note"])
        assert by["sae.l131k.ex.ratio.bo1.q1"]["run"] == "R1"
        # A RATIO ON THE 16M DENOMINATOR IS `provisional`, NEVER `final` (spec §1.2/§1.4: the
        # denominator is the 10M TRAINING corpus). The FIRED cells have no denominator and stay
        # `final`, so the two series of panel b can land on different days without either one
        # claiming a convention it does not have.
        for k, r in by.items():
            if ".ratio." in k:
                assert r["status"] == "provisional", (k, r["status"])
                assert "NOT the 10M training corpus" in r["note"], (k, r["note"])
            if ".fired." in k:
                assert r["status"] == "final", (k, r["status"])

        # THE GATE MUTATION: a product whose gate is not the 1.5846 the spec names must refuse.
        root2 = Path(td) / "moved-gate"
        _m1_mirror(root2, gate=1.6000)
        try:
            _m1_cells(root2, corpus=False)
        except AssertionError as exc:
            assert "1.5846" in str(exc) and "Refusing" in str(exc), exc
        else:
            raise AssertionError("a 131k block with the wrong gate produced fired cells anyway")


def check_m1_cells_writer_keeps_the_file_s_line_endings():
    """A CRLF `cells.csv` stays CRLF, and every untouched row survives BYTE for byte.

    The real file is CRLF. `Path.read_text()` translates that to LF on the way in, which made the
    writer's own byte-identity assertion compare the rewrite against the already-translated text
    and pass while every one of 413 lines changed on disk -- a whole-file diff in a shared
    file, for one row rewritten. So the terminator is read off the header line and the
    read and the write both pass `newline=""`. MEASURED 2026-09-23 on the real file.
    """
    import results.faithfulness as F

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cells.csv"
        crlf = (
            "key,value,se,lo,hi,n,status,run,source,date,note\r\n"
            "fid.ra.ex.cos.bo1,0.1,,,,,placeholder,R1,x,2026-09-01,mine\r\n"
            "zzz.other.builder.key,0.9,,,,,final,R9,y,2026-09-01,\"not mine, quoted\"\r\n"
        )
        path.write_bytes(crlf.encode())
        before = path.read_bytes().split(b"\r\n")
        F.write_cells(path, [F.cell("fid.ra.ex.cos.bo1", 0.5, run="R1", date="2026-09-23",
                                    source="s", note="rewritten")],
                      {"fid.ra.ex.cos.bo1"})
        raw = path.read_bytes()
        assert b"\r\n" in raw, "the CRLF terminators did not survive the rewrite"
        assert raw.count(b"\r\n") == 3, raw
        assert b"\n" not in raw.replace(b"\r\n", b""), "a bare LF was written into a CRLF file"
        after = raw.split(b"\r\n")
        # the other builder's row and the header, byte for byte
        assert after[0] == before[0], (after[0], before[0])
        assert after[2] == before[2], (after[2], before[2])
        assert b"0.5000" in after[1], after[1]


def check_m1_corpus_denominator_is_a_parameter():
    """`--corpus-peak`: the 10M source CHANGES every ratio, and asked-for-but-absent RAISES.

    This is the defect the parameter exists to prevent: the 16M held-out peak and the 10M training
    peak are maxima over different texts, a ratio against the wrong one prints as a perfectly
    ordinary number, and nothing downstream can tell. So the fixture's two denominators are
    different numbers, the cells must move when the parameter moves, and a source that is not on
    the volume must stop the run instead of falling back.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)
        rel = _m1_top1_act(root, f"base/{M1_BASE}/sae/l42-1b/top1_act/{M1_CT}__train_parity_10m")

        stored, _ = _m1_cells(root, corpus=False)
        ten_m, _ = _m1_cells(root, corpus=False, corpus_peak=f"top1_act:{rel}")
        a, b = _m1_by_key(stored), _m1_by_key(ten_m)
        for s in range(4):
            q = f"sae.l131k.ex.ratio.bo1.q{s + 1}"
            assert a[q]["value"] == f"{M1_RATIO_STORED[s]:.4f}", a[q]
            assert b[q]["value"] == f"{M1_RATIO_TOP1[s]:.4f}", b[q]
            assert a[q]["value"] != b[q]["value"], (a[q], b[q])
        # the provenance travels into every ratio cell, and says WHICH corpus
        assert "16M HELD-OUT" in a["sae.l131k.ex.ratio.bo1.q1"]["note"]
        assert "top1_act" in b["sae.l131k.ex.ratio.bo1.q1"]["note"]
        assert "10M" in b["sae.l131k.ex.ratio.bo1.q1"]["note"]
        # the FIRED cells are untouched by the denominator: they are an activation against a gate
        assert a["sae.l131k.ex.fired.bo8.q1"]["value"] == b["sae.l131k.ex.fired.bo8.q1"]["value"]

        # ASKED FOR AND ABSENT: raises, names the path, and never falls back to `stored`.
        vol = R.Vol("", root, offline=True, quiet=True)
        missing = f"base/{M1_BASE}/sae/l42-1b/top1_act/{M1_CT}__not_run_yet"
        try:
            F.corpus_peaks(vol, f"top1_act:{missing}")
        except AssertionError as exc:
            assert missing in str(exc) and "REFUSES" in str(exc), exc
        else:
            raise AssertionError("an absent 10M denominator did not stop the run")
        # and the same through `analyse`, which is where a run would actually meet it
        try:
            F.analyse(vol, M1_CFG, M1_CT, "", 400, 1, False, 8.0, 128.0, True,
                      f"top1_act:{missing}")
        except AssertionError as exc:
            assert missing in str(exc), exc
        else:
            raise AssertionError("analyse fell back instead of raising on an absent denominator")
        # a row the 10M product does not carry loses its ratio; it is NOT divided by the 16M peak
        short = _m1_top1_act(Path(td) / "short-root", "t1")
        (Path(td) / "short-root" / "t1" / "top1_act.jsonl").write_text(
            json.dumps({"row": 2, "feature": 1000, "act_max": 2.0}) + "\n")
        vol2 = R.Vol("", Path(td) / "short-root", offline=True, quiet=True)
        peaks = F.corpus_peaks(vol2, f"top1_act:{short}")
        assert peaks.of(2, 4.0) == (2.0, peaks.provenance)
        v, _prov = peaks.of(3, 4.0)
        assert math.isnan(v) and peaks.missing_rows == [3], (v, peaks.missing_rows)


def check_m1_denominator_of_another_set_is_refused():
    """A `top1_act` product of ANOTHER set must not answer this family's rows.

    THE ROW JOIN IS NOT SELF-DESCRIBING, and on the real volume the two sets collide exactly:
    `2026-09-21_v3_ctrl`'s 131k features are rows 512-1023 and `2026-09-21_v3_sae2m`'s decoder
    half is rows 512-1023, so the 131k `top1_act` answers every question the 2M decoder block
    asks and answers all of them with another dictionary's feature. `discover_families` already
    refuses to guess a dictionary from a feature id ("every 131k id is also a valid 2M index");
    this is the same join and gets the same refusal.

    The refusal is per FAMILY and not a raise: the ratios go absent and are listed, while the
    fired cells -- which have no denominator -- still land. That is the split the 2M block needs.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)
        good = _m1_top1_act(root, f"base/{M1_BASE}/sae/l42-1b/top1_act/{M1_CT}__ok")
        # the SAME rows and the same act_max, but the summary says another set
        other = _m1_top1_act(root, f"base/{M1_BASE}/sae/l42-1b/top1_act/OTHER__10m",
                             set_name="ANOTHER_SET")
        # ... and another whose summary says another DICTIONARY
        wrong_sae = _m1_top1_act(root, f"base/{M1_BASE}/sae/l42-1b/top1_act/{M1_CT}__sae2m",
                                 sae_key=f"{M1_BASE}/sae2m")
        # ... and one that states neither, which is refused rather than trusted
        silent = _m1_top1_act(root, f"base/{M1_BASE}/sae/l42-1b/top1_act/{M1_CT}__nosum",
                              summary=False)

        ok_rows, _ = _m1_cells(root, corpus=False, corpus_peak=f"top1_act:{good}")
        by_ok = _m1_by_key(ok_rows)
        assert by_ok["sae.l131k.ex.ratio.bo1.q1"]["value"] == f"{M1_RATIO_TOP1[0]:.4f}", by_ok

        for rel, wanted in ((other, "ANOTHER_SET"), (wrong_sae, "sae2m"), (silent, "states no")):
            rows, skipped = _m1_cells(root, corpus=False, corpus_peak=f"top1_act:{rel}")
            by = _m1_by_key(rows)
            for s_ in range(4):
                q = f"sae.l131k.ex.ratio.bo1.q{s_ + 1}"
                assert q not in by, f"{rel}: {q} was built from another product's rows: {by[q]}"
            # the fired cells are untouched: they have no denominator
            assert by["sae.l131k.ex.fired.bo1.q1"]["value"] == f"{M1_FIRED[0]:.4f}", by
            joined = " | ".join(skipped)
            assert wanted in joined, (rel, joined)
            assert "NO DENOMINATOR" in joined or "no denominator" in joined.lower(), joined

        # THE MUTATION. Point the refused product's summary back at this set and dictionary and
        # the ratios must come back -- so the refusal above is about the stated provenance and
        # not about some other difference between the two products.
        (root / other / "summary.json").write_text(json.dumps(
            {"corpus_size_m": 10, "sae": M1_SAE, "set": M1_CT, "n_features": 8}))
        back, _ = _m1_cells(root, corpus=False, corpus_peak=f"top1_act:{other}")
        bb = _m1_by_key(back)
        assert bb["sae.l131k.ex.ratio.bo1.q1"]["value"] == f"{M1_RATIO_TOP1[0]:.4f}", bb


def check_m1_missing_corpus_search_skips_and_lists():
    """No `results/corpus_search.py` -> the paired cells are SKIPPED and LISTED, never zero-filled.

    "The corpus scored 0" and "nobody ran the corpus scan" are opposite findings. A zero-filled
    comparator turns the second into the first, and the paired difference then prints the
    Exemplifier's own number wearing a comparison's name -- which is the most flattering wrong
    answer available, so it gets the loudest guard.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)
        rows, skipped = _m1_cells(root, corpus=False)
        by = _m1_by_key(rows)
        for key in ("fid.ra.diff.cos.bo8", "fid.ra.ex.win.bo8"):
            assert key not in by, f"{key} was written with no corpus comparator: {by.get(key)}"
            assert any(key in s for s in skipped), skipped
        joined = " | ".join(skipped)
        assert "results/corpus_search.py" in joined, joined
        assert "zero-filled" in joined, joined
        # the cells that do NOT need the comparator are unaffected
        assert "fid.ra.ex.cos.bo8" in by and "sae.l131k.ex.fired.bo8.q1" in by

        # with the module present they appear -- so the skip above is about the input, not a bug
        with_it, _ = _m1_cells(root, corpus=True)
        assert {"fid.ra.diff.cos.bo8", "fid.ra.ex.win.bo8"} <= set(_m1_by_key(with_it)), with_it

        # THE MODULE IS THERE AND NO SCAN WAS NAMED: the same two cells are skipped, and for a
        # DIFFERENT stated reason. The upstream block is scanned against two corpora under one run tag, so
        # "which scan" is not a default; a driver that picked one would print the held-out number
        # under the training corpus's label.
        rows_ns, skipped_ns = _m1_cells(root, corpus=True, corpus_scan="")
        by_ns = _m1_by_key(rows_ns)
        for key in ("fid.ra.diff.cos.bo8", "fid.ra.ex.win.bo8"):
            assert key not in by_ns, f"{key} was written with no --corpus-scan: {by_ns.get(key)}"
        joined_ns = " | ".join(skipped_ns)
        assert "--corpus-scan` is unset" in joined_ns, joined_ns

        # a module that IS there but does not export what this driver calls is a CONTRACT
        # MISMATCH and raises -- reporting it as "absent" would send the reader looking for a run
        # that has already happened. This is the defect that was in the file: the driver looked
        # for `corpus_top1` / `top1_by_row` / `top1_cosines` and M2 exports `read_top1`.
        before = F.CORPUS_SEARCH
        F.CORPUS_SEARCH = type("Empty", (), {"__doc__": "no exports"})
        try:
            F.corpus_search_module()
        except AssertionError as exc:
            assert "CONTRACT MISMATCH" in str(exc), exc
            assert "read_top1" in str(exc), exc
        else:
            raise AssertionError("an incompatible corpus_search module was treated as absent")
        finally:
            F.CORPUS_SEARCH = before
        # and the real module satisfies it, which is what makes the probe above a real gate
        mod, prov = F.corpus_search_module()
        assert mod is CS and "read_top1" in prov, (mod, prov)


def check_m1_paired_cells_read_m2s_scan():
    """The paired cells come from M2's `read_top1` on the scan `--corpus-scan` names -- and MOVE.

    The defect this covers: `corpus_top1_for` looked for exports `results/corpus_search.py` never
    had, so `fid.ra.diff.cos.bo8` and `fid.ra.ex.win.bo8` were reported ABSENT on every run while
    the scan sat on the volume. A green "the cells appear" is not enough to catch that coming
    back, because a comparator wired to the wrong scan also makes them appear; so this asserts the
    VALUES against the fixture's literals, asserts the decoy scan's values are NOT what lands, and
    then MUTATES the named scan and requires the assertion to go red.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)

        # the scan is addressed three ways and they are ONE directory
        for spec in (M1_CORPUS_KEY,
                     f"{M1_RA}__{M1_CORPUS_KEY}",
                     f"base/{M1_BASE}/scan/{M1_RA}__{M1_CORPUS_KEY}"):
            assert F.corpus_scan_dir(M1_BASE, M1_RA, spec) == f"{M1_RA}__{M1_CORPUS_KEY}", spec
        try:
            F.corpus_scan_dir(M1_BASE, M1_RA, "somewhere/else/entirely")
        except AssertionError as exc:
            assert "does not start with" in str(exc), exc
        else:
            raise AssertionError("a path outside the scan directory was accepted")

        rows, skipped = _m1_cells(root, corpus=True)
        by = _m1_by_key(rows)
        d, w = by["fid.ra.diff.cos.bo8"], by["fid.ra.ex.win.bo8"]
        assert d["value"] == f"{M1_DIFF:.4f}", d
        assert w["value"] == f"{M1_WIN:.4f}", w
        assert d["n"] == "4" and w["n"] == "4", (d, w)
        # the provenance travels into the note: the scan, the corpus label, the reader
        for c in (d, w):
            assert f"{M1_RA}__{M1_CORPUS_KEY}/topk.jsonl" in c["note"], c["note"]
            assert "results.corpus_search.read_top1" in c["note"], c["note"]
            assert "10M" in c["note"], c["note"]
        assert not [x for x in skipped if "fid.ra.diff" in x], skipped

        # THE DECOY. The held-out scan carries every row 0.125 higher, so reading it would give a
        # difference 0.125 lower and a win fraction of 1/2, not 3/4. Naming it must produce THOSE
        # numbers -- which proves the driver reads the directory it is told to and not the first
        # one under `scan/`.
        rows_o, _ = _m1_cells(root, corpus=True, corpus_scan=M1_CORPUS_KEY_OTHER)
        bo = _m1_by_key(rows_o)
        want_o = sum(M1_EX_BO8_ROW[r] - M1_CORPUS_OTHER[r] for r in range(4)) / 4
        assert bo["fid.ra.diff.cos.bo8"]["value"] == f"{want_o:.4f}", bo["fid.ra.diff.cos.bo8"]
        assert bo["fid.ra.diff.cos.bo8"]["value"] != d["value"], (bo, d)
        assert M1_CORPUS_KEY_OTHER in bo["fid.ra.diff.cos.bo8"]["note"], bo

        # A SIZE THE SCAN DOES NOT CARRY is a refusal, not a nearest rung: the cells are specified
        # at 10M and a scan that stopped short is a different product.
        try:
            _m1_cells(root, corpus=True, corpus_size=2.5)
        except AssertionError as exc:
            assert "no 2.5M" in str(exc), exc
        else:
            raise AssertionError("a corpus size absent from the scan was silently substituted")

        # THE MUTATION. Row 2 is the one target the Exemplifier loses on. Lift its corpus top-1 at
        # 10M and the difference, the win fraction and nothing else must move; a driver that had
        # cached, stubbed or defaulted the comparator stays green here and must not.
        moved = {**M1_CORPUS, 2: 0.125}
        _m1_scan(root, M1_CORPUS_KEY, moved)
        rows_m, _ = _m1_cells(root, corpus=True)
        bm = _m1_by_key(rows_m)
        want_m = sum(M1_EX_BO8_ROW[r] - moved[r] for r in range(4)) / 4
        assert bm["fid.ra.diff.cos.bo8"]["value"] == f"{want_m:.4f}", bm["fid.ra.diff.cos.bo8"]
        assert bm["fid.ra.diff.cos.bo8"]["value"] != d["value"], "the mutation did not move the cell"
        assert bm["fid.ra.ex.win.bo8"]["value"] == "1.0000", bm["fid.ra.ex.win.bo8"]
        # and the cells that do NOT read the scan are untouched by it
        assert bm["fid.ra.ex.cos.bo8"]["value"] == by["fid.ra.ex.cos.bo8"]["value"], bm

        # A RAGGED LADDER is M2's own gate and it must reach this driver rather than being caught
        # by nothing: a row with a 10M top-1 and no 2.5M one is a product defect.
        _m1_scan(root, M1_CORPUS_KEY, M1_CORPUS)
        path = root / f"base/{M1_BASE}/scan/{M1_RA}__{M1_CORPUS_KEY}/topk.jsonl"
        kept = [line for line in path.read_text().splitlines()
                if not (json.loads(line)["set"] == M1_RA
                        and json.loads(line)["set_row"] == 1
                        and json.loads(line)["size"] == 5.0)]
        path.write_text("\n".join(kept) + "\n")
        try:
            _m1_cells(root, corpus=True)
        except AssertionError as exc:
            assert "absent at some size" in str(exc), exc
        else:
            raise AssertionError("a ragged size ladder reached the paired cells")


def check_m1_cells_end_to_end():
    """`analyse` -> `paper_cells` -> `write_cells` on a real CSV, and the read-only default.

    The last link: the rows the builders produce actually land in a `cells.csv`-shaped file, the
    keys that could not be built are reported and NOT written, and a run given no `--cells` path
    writes nothing at all.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _m1_mirror(root)
        rows, skipped = _m1_cells(root, corpus=True)
        p = Path(td) / "cells.csv"
        seeded = sorted({r["key"] for r in rows} | {"ai.l131k.c16.det"})
        p.write_text("key,value,se,lo,hi,n,status,run,source,date,note\n"
                     + "".join(f"{k},,,,,,placeholder,R0,,2026-09-23,seeded\n" for k in seeded))
        before = p.read_text()
        rec = F.write_cells(p, rows, F.M1_KEYS)
        assert sorted(rec["rewritten"]) == sorted(r["key"] for r in rows), rec
        assert rec["appended"] == [], rec
        after = p.read_text()
        assert "ai.l131k.c16.det,,,,,,placeholder,R0,,2026-09-23,seeded\n" in after, after
        assert before != after
        # every key that was NOT built kept its placeholder row: an absent number stays absent
        built = {r["key"] for r in rows}
        for key in sorted(F.M1_KEYS - built):
            assert f"{key}," not in after, f"{key} was written although it was not built"
        assert any("not built by this run" in s for s in skipped), skipped
        # and the parsed file still has one record per key, eleven fields each
        import csv as _csv
        recs = list(_csv.reader(after.splitlines()))
        assert recs[0] == list(F.CELLS_COLUMNS)
        keys = [r[0] for r in recs[1:]]
        assert len(keys) == len(set(keys)) == len(seeded), (len(keys), len(seeded))
        assert all(len(r) == len(F.CELLS_COLUMNS) for r in recs)


# --- the case-study page: one stored `block` back into its example windows -----------------------


def check_autointerp_cases_block_split():
    """`autointerp_cases.split_block` inverts `build.exemplar_block`, marks, newlines and all.

    THE WHOLE PER-ARM DIVERSITY TABLE RESTS ON THIS. The mean pairwise Jaccard printed for an arm
    is `build.content_words` over the texts recovered here, and the block is the only place those
    texts exist -- the build stores the provenance of each example but not its text. Three ways the
    inverse can go wrong, all of them present in this fixture and each changing a number:

      * a CORPUS WINDOW CONTAINS NEWLINES, so an example is not a line: splitting on lines would
        make example 1 into four examples, and example 4's own text carries a line reading
        `Example 9:  ` that a numbering-agnostic split would turn into a fifth example;
      * a MARKED TOKEN CAN BE A NEWLINE, so the `Activations:` line is itself not always one line.
        `rpartition("\n")` took the tail of the pair list on the first real feature this ran on,
        which is why the reader looks for the last `\nActivations: ` instead;
      * the `Activations:` line REPEATS every marked token, so leaving it in the text adds those
        tokens a second time to the content-word set. The check asserts it changes the Jaccard
        rather than trusting that it would.

    No mirror and no volume: `exemplar_block` is a pure function of the example dicts, so the
    fixture is the dicts.
    """
    ex = [
        # a corpus window with interior newlines and two marked runs
        {"text_marked": "the quarterly<< dividend>> was\napproved by the\nboard\nyesterday",
         "activations": [(" dividend", 7)], "n_marked": 1},
        # a marked token that IS a newline: the pair list spans two lines
        {"text_marked": "revenue rose<<\n>>guidance unchanged",
         "activations": [("\n", 4), (" rose", 2)], "n_marked": 2},
        # nothing marked at all, so `exemplar_block` writes no `Activations:` line
        {"text_marked": "an unmarked rollout about turbines", "activations": [], "n_marked": 0},
        # a text that itself contains a line looking like the next example's header
        {"text_marked": "see below\nExample 9:  not a boundary<< here>>",
         "activations": [(" here", 3)], "n_marked": 1},
    ]
    block = AB.exemplar_block(ex)
    shown = AC.split_block(block, ex)
    assert len(shown) == len(ex), [s.marked for s in shown]
    for got, want in zip(shown, ex, strict=True):
        assert got.marked == want["text_marked"], (got.marked, want["text_marked"])
        assert got.plain == want["text_marked"].replace("<<", "").replace(">>", "")
        assert bool(got.acts_line) == bool(want["activations"]), (got.acts_line, want)
    # the newline-marked example's `Activations:` line really does span two lines, so this fixture
    # exercises the branch and is not merely asserting an easy case
    assert "\n" in shown[1].acts_line, shown[1].acts_line
    assert shown[1].n_folds == shown[1].marked.count("\n") + shown[1].acts_line.count("\n")

    # the statistic the page prints, and what leaving the `Activations:` line in would do to it
    good = AC.mean_pairwise_jaccard(shown)
    dirty = AC.mean_pairwise_jaccard(
        [AC.Shown(marked=s.marked, plain=s.plain + "\n" + s.acts_line, acts_line="", meta=s.meta)
         for s in shown])
    assert good is not None and dirty is not None and abs(good - dirty) > 1e-9, (good, dirty)
    # and it IS `M-jac16`'s own distance: the same pairs off `jaccard_distances`
    d = AB.jaccard_distances([{"text": s.plain} for s in shown])
    iu = np.triu_indices(len(shown), k=1)
    assert abs(good - float(np.mean(1.0 - d[iu]))) < 1e-12, (good, d)

    # the `Example 9:  ` line inside example 4 stayed in its text rather than becoming a boundary
    assert "Example 9:  not a boundary" in shown[3].plain, shown[3].plain

    # a block holding MORE examples than the caller's rows is REFUSED, not read as fewer: the
    # extra one would otherwise be swallowed into the last example's text and change its Jaccard
    try:
        AC.split_block(block, ex[:3])
    except AssertionError as exc:
        assert "boundaries and the build stored 3" in str(exc), exc
    else:
        raise AssertionError("a block with more examples than the build stored was accepted")


def check_autointerp_encdec_pairs_and_replay():
    """`autointerp_encdec.compare` on the eval-2 fixture, `rl16` read as the ENCODER run and `old`
    as the DECODER run: the paired dec - enc, the per-feature rows, and the replay check, red and
    green.

    The paired numbers are the fixture's dyadic literals: `M` detection is {10: .5, 11: .5, 12: .75,
    13: .625} in `rl16` and {.5, .625, .625, .5} in `old`, so dec - enc = {0, +1/8, -1/8, -1/8},
    mean -1/32 over four features with one tie; `M` fuzzing loses feature 13 to `old`'s null
    `bal_acc`, so it pairs THREE features, {+1/8, 0, 0}, mean +1/24, and names 13 as enc-only.

    The replay check is RED on `DOCMAX`, which the fixture gives different numbers in the two runs
    and whose features were refused nowhere, and it separates the two legitimate exemptions: on `M`
    detection feature 13 has a batch unparsed in `rl16` (`AI_PARSED`), so its difference is a
    re-sent scorer call and the verdict names only 11 and 12. The same run read under two labels
    is GREEN everywhere with every difference exactly zero.
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest-encdec-"))
    try:
        write_autointerp_runs(tmp)
        vol = R.Vol("", tmp, offline=True, quiet=True)
        enc, dec = ("rl16", AI_RUNS["rl16"]), ("old", AI_RUNS["old"])
        res = AE.compare(vol, enc, dec, ["M", "NLA"], ["DOCMAX", "M"], 2000, 1, bands=False)
        pair = {(p["arm"], p["scorer"]): p for p in res["pairs"]}
        det = pair[("M", "detection")]["bal_acc"]
        assert det["n_paired"] == 4 and abs(det["mean"] - (-1 / 32)) < 1e-12, det
        assert abs(det["enc_mean"] - 0.59375) < 1e-12 and abs(det["dec_mean"] - 0.5625) < 1e-12
        assert (det["win_frac"], det["loss_frac"], det["n_zero"]) == (0.25, 0.5, 1), det
        assert det["lo"] <= det["mean"] <= det["hi"], det
        fz = pair[("M", "fuzzing")]["bal_acc"]
        assert fz["n_paired"] == 3 and abs(fz["mean"] - 1 / 24) < 1e-12, fz
        assert fz["only_enc"] == [13] and fz["only_dec"] == [], fz
        # an arm the decoder run never scored has no pair, but its refusal still reaches the rows
        assert "bal_acc" not in pair[("NLA", "detection")], pair[("NLA", "detection")]
        rows = {(r["feature"], r["arm"], r["scorer"]): r for r in res["per_feature"]}
        r13 = rows[(13, "M", "fuzzing")]
        assert r13["bal_acc_dec"] is None and r13["diff"] is None and r13["bal_acc_enc"] == 0.5
        n13 = rows[(13, "NLA", "detection")]
        assert (n13["refused_enc"], n13["refused_dec"], n13["bal_acc_enc"]) == (1, 0, None), n13
        assert rows[(11, "M", "detection")]["diff"] == 0.125

        rep = {(x["arm"], x["scorer"]): x for x in res["replay"]}
        assert not rep[("DOCMAX", "detection")]["ok"], rep[("DOCMAX", "detection")]
        assert rep[("DOCMAX", "detection")]["unexplained"] == [10, 11, 12, 13]
        assert rep[("DOCMAX", "fuzzing")]["differ"] == [11, 13]
        m = rep[("M", "detection")]
        assert m["scorer_resent"] == [13] and m["unexplained"] == [11, 12] and not m["ok"], m

        same = AE.compare(vol, ("a", AI_RUNS["rl16"]), ("b", AI_RUNS["rl16"]), ["M"],
                          ["DOCMAX", "M", "NLA"], 2000, 1, bands=False)
        assert all(x["ok"] and x["n_identical"] == x["n_common"] for x in same["replay"]), \
            same["replay"]
        for p in same["pairs"]:
            b = p["bal_acc"]
            assert b["mean"] == 0.0 and b["lo"] == 0.0 and b["hi"] == 0.0, b
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_autointerp_encdec_names():
    """`--names` / `--titles` relabel `autointerp_encdec`'s OUTPUTS and nothing else.

    The same fixture comparison written twice: under the default names it must produce the M6-dec
    file stems and headings (`enc_vs_dec_summary.md`, "decoder − encoder"), under `rl,sft` /
    `RL,SFT` the relabelled ones -- with no `enc`/`dec` word left in any table header or CSV
    column of the relabelled summary, and the per-feature CSV's rows identical between the two
    (the relabel must not touch a number). `_col` is token-wise, so `n_refused_dec` becomes
    `n_refused_sft` while a column that merely CONTAINS the letters (`decision`) would not move.
    """
    assert AE._col("n_refused_dec", ("rl", "sft")) == "n_refused_sft"
    assert AE._col("enc_own_mean", ("rl", "sft")) == "rl_own_mean"
    assert AE._col("decision", ("rl", "sft")) == "decision"
    assert AE._relabel("dec − enc", ("rl", "sft")) == "sft − rl"
    assert AE._relabel("decoder", ("rl", "sft")) == "decoder"
    tmp = Path(tempfile.mkdtemp(prefix="selftest-encdec-names-"))
    try:
        write_autointerp_runs(tmp)
        vol = R.Vol("", tmp, offline=True, quiet=True)
        res = AE.compare(vol, ("rl16", AI_RUNS["rl16"]), ("old", AI_RUNS["old"]), ["M"],
                         ["DOCMAX"], 2000, 1, bands=False)
        d_out, n_out = tmp / "out-default", tmp / "out-named"
        d_out.mkdir()
        n_out.mkdir()
        pd = AE.write_outputs(res, None, d_out, [], [])
        pn = AE.write_outputs(res, None, n_out, [], [], ("rl", "sft"), ("RL", "SFT"))
        assert pd.name == "enc_vs_dec_summary.md" and pn.name == "rl_vs_sft_summary.md", (pd, pn)
        assert sorted(p.name for p in d_out.iterdir()) == [
            "enc_vs_dec.csv", "enc_vs_dec_bands.csv", "enc_vs_dec_replay.csv", "enc_vs_dec_summary.csv",
            "enc_vs_dec_summary.md"], sorted(p.name for p in d_out.iterdir())
        assert sorted(p.name for p in n_out.iterdir()) == [
            "rl_vs_sft.csv", "rl_vs_sft_bands.csv", "rl_vs_sft_replay.csv", "rl_vs_sft_summary.csv",
            "rl_vs_sft_summary.md"], sorted(p.name for p in n_out.iterdir())
        td, tn = pd.read_text(), pn.read_text()
        assert "## Headline: balanced accuracy, decoder − encoder" in td
        assert "and of `dec − enc`, each" in td and "`dec − enc` is paired" in td
        assert "and of `sft − rl`, each" in tn and "`sft − rl` is paired" in tn, tn[:600]
        assert "{'" not in td and "{'" not in tn, "a dict was interpolated into the prose"
        assert "## Headline: balanced accuracy, SFT − RL" in tn and "decoder" not in tn, tn[:400]
        for ln in tn.splitlines():
            if ln.startswith("| arm |"):
                assert not re.search(r"\b(enc|dec)\b", ln), ln
        for name in ("rl_vs_sft.csv", "rl_vs_sft_summary.csv", "rl_vs_sft_replay.csv",
                     "rl_vs_sft_bands.csv"):
            head = (n_out / name).read_text().splitlines()[0].split(",")
            assert not any(t in ("enc", "dec") for c in head for t in c.split("_")), (name, head)
        dcsv = (d_out / "enc_vs_dec.csv").read_text().splitlines()
        ncsv = (n_out / "rl_vs_sft.csv").read_text().splitlines()
        assert "bal_acc_rl" in ncsv[0] and "bal_acc_enc" in dcsv[0], (ncsv[0], dcsv[0])
        assert dcsv[1:] == ncsv[1:] and len(ncsv) > 1, "relabelling changed a per-feature row"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


CHECKS = [
    check_parse_scores_dir,
    check_estimators,
    check_one_bo_estimator,
    check_fired_is_the_same_estimator,
    check_cluster_bootstrap,
    check_round_trip,
    check_second_sae_needs_no_code,
    check_sae_side_reads_its_own_product,
    check_exclusions_leave_every_surface_together,
    check_centred_bok_is_recomputed_from_the_array,
    check_missing_sources_are_listed_not_zeroed,
    check_sources_filter,
    check_reader_check_catches_a_defect,
    check_sanity_verdicts,
    check_cross_set_gate,
    check_render_and_figures,
    check_combined_layer_lifts_and_never_recomputes,
    # eval 2 -- `results/autointerp.py`
    check_autointerp_reader,
    check_autointerp_refusals_and_conventions,
    check_autointerp_versus_second_build,
    check_autointerp_estimators,
    check_autointerp_pairing_is_intersection,
    check_autointerp_catches_a_defect,
    check_autointerp_render_and_figures,
    # eval 2's cut views -- the rarity strata and the post-hoc magnitude quartiles
    check_autointerp_cut_views,
    check_autointerp_short_cells_are_reported,
    check_autointerp_null_bal_acc_is_never_filled_from_acc,
    check_autointerp_trend_statistics,
    check_autointerp_top_cell_separation,
    check_autointerp_trend_verdict,
    check_autointerp_cut_render,
    check_autointerp_cuts_catch_a_defect,
    # eval 2's per-activation-band join -- the build's bands x the scorer's answers
    check_autointerp_band_recall,
    check_autointerp_band_cells_and_contrasts,
    check_autointerp_bands_catch_a_defect,
    # the four-arm case-study page -- the block reader every per-arm statistic is taken over
    check_autointerp_cases_block_split,
    # the encoder-vs-decoder paired reader
    check_autointerp_encdec_pairs_and_replay,
    check_autointerp_encdec_names,
    check_ood_scan_key,
    check_ood_arm_table,
    check_ood_bo8_headline,
    check_ood_lid_ranking,
    check_ood_lid_wantlist,
    check_ood_centred_scan_mean_is_read,
    check_ood_lid_reads_every_chunk_the_readme_names,
    check_ood_arm_size_comes_from_the_product,
    check_ood_counts_are_the_bo8_verdict,
    check_ood_cells_merge_keeps_every_other_writers_bytes,
    check_ood_cells_never_carry_an_em_dash,
    # module M1 -- the paper's cells, the ratio denominator, the corpus comparator
    check_m1_clusters_fire_on_upstream_block,
    check_m1_fired_bok_mutation,
    check_m1_cells_writer_rewrites_in_place,
    check_m1_panel_a_cells,
    check_m1_panel_b_cells_and_the_gate,
    check_m1_cells_writer_keeps_the_file_s_line_endings,
    check_m1_corpus_denominator_is_a_parameter,
    check_m1_denominator_of_another_set_is_refused,
    check_m1_missing_corpus_search_skips_and_lists,
    check_m1_paired_cells_read_m2s_scan,
    check_m1_cells_end_to_end,
]


def run_all():
    import time as _time

    for fn in CHECKS:
        t0 = _time.time()
        fn()
        print(f"[smoke] ok  {fn.__name__:<42} {_time.time() - t0:5.2f}s", flush=True)
    print(f"[smoke] {len(CHECKS)}/{len(CHECKS)} checks passed", flush=True)


if __name__ == "__main__":
    run_all()
