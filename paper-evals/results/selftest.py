#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "typer>=0.15", "pyyaml>=6", "matplotlib>=3.9"]
# ///
"""CPU unit smoke for `results/` -- no volume, no network, no GPU.

    uv run paper-evals/results/selftest.py

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
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import results.autointerp as A  # noqa: E402
import results.common as R  # noqa: E402
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
    expect: 0.5076
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
        {"storage": "raw", "mu_stored": None, "family_mu": {}, "sae_key": "B/sae-one"}))

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
    # `score --score-name <set>__<tag> --engine vllm` gives `<set>__<tag>__<engine>`, because
    # `scores_dir` has no tag parameter and folds the tag into the set name. Reading that as
    # engine `hf` with the tag `mu-none__vllm` put "hf" in the paper's CSV for six vLLM arms.
    assert F.R.parse_scores_dir("S__mu-none__vllm", "S") == ("vllm", "mu-none")
    assert F.R.parse_scores_dir("S__mu-stats__vllm", "S") == ("vllm", "mu-stats")
    # A directory that merely starts with the set name is ANOTHER set, not an untagged run of
    # this one -- `2026-09-16_v1` and `2026-09-16_v1x` both exist in that namespace.
    assert F.R.parse_scores_dir("Sx", "S") is None
    assert F.R.parse_scores_dir("T__vllm", "S") is None


def check_estimators():
    """`best_of_k_means`, `peaks_of` and `best_per_rollout` against numbers written out here."""
    bo = R.best_of_k_means([0.5, 0.25, 0.75, 1.0], (1, 2, 4, 8, 64))
    _close(bo[1], 0.625, what="bo1 is the plain mean")
    _close(bo[2], 0.75, what="bo2 = (max(.5,.25) + max(.75,1.0)) / 2")
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


def check_centred_bok_is_recomputed_from_the_array():
    """The centred bo-k ladder comes from `cos_centred.f16`, by the disjoint-group estimator.

    `score` writes `bo_c_<k>` only for a row whose EVERY rollout kept a centred token
    (`score.py:521`, `if len(vals_c) == n`), and on eval 1's own products no row qualifies -- so
    the centred bo8/bo64 columns of plan §2.3 exist nowhere on the volume and have to be made
    here. Four things, each its own failure:

      1. with no NaN, the recomputation equals `common.best_of_k_means` on the same values EXACTLY
         -- it is the same estimator, not a similar one, and that is what lets it sit in a column
         beside the stored raw ladder;
      2. a NaN rollout is dropped from ITS OWN group and the positions of the others do not move.
         Compacting the survivors first would regroup them, which is a different statistic that
         would agree on the mean (bo1) and differ everywhere else -- so bo1 cannot detect it and
         bo2 is checked by hand here;
      3. a group with NO finite draw is dropped from the average, never counted as a zero;
      4. over `--centred-bok-max-mb` the read is REFUSED with a reason, not answered from the
         stored ladder that is not there.
    """
    ids = [{"row": 0, "family": "realact", "doc": 1, "id": "a"},
           {"row": 1, "family": "realact", "doc": 2, "id": "b"}]
    # Row 0: all four finite. Row 1: draws 1 and 2 have no kept centred token.
    #   row 0 bests [0.25, 0.75, 0.5, 1.0] -> bo1 0.625, bo2 (0.75 + 1.0)/2 = 0.875, bo4 1.0
    #   row 1 bests [0.5, nan, nan, 0.25]  -> bo1 (0.5+0.25)/2 = 0.375
    #                                          bo2 groups (0.5,nan) -> 0.5 and (nan,0.25) -> 0.25,
    #                                               mean 0.375; COMPACTING would give one group
    #                                               (0.5, 0.25) -> 0.5, which is the wrong answer
    #                                          bo4 one group -> 0.5
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
        (hd / "storage.json").write_text(json.dumps({"storage": "raw", "mu_stored": None}))
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
        # (1) the all-finite row IS `best_of_k_means`, exactly.
        ladder, info = F.centred_bok(vol, res["sources"][0], 128.0)
        want0 = R.best_of_k_means(centred[0], R.BO_KS_ALL)
        assert ladder[0] == want0, f"row 0: {ladder[0]} vs best_of_k_means {want0}"
        # (2)/(3) the NaN row, by hand.
        _close(ladder[1][1], 0.375, 1e-9, what="bo1 over the finite draws")
        _close(ladder[1][2], 0.375, 1e-9, what="bo2 keeps POSITIONS -- compacting would give 0.5")
        _close(ladder[1][4], 0.5, 1e-9, what="bo4 is the one group's finite max")
        assert info["rows_with_nan_rollouts"] == 1 and info["nan_rollouts"] == 2, info
        # The product stores no ladder at all, so there was nothing to compare against -- and the
        # table must SAY that rather than printing a vacuous zero-mismatch pass.
        assert info["stored_comparisons"] == 0 and info["n_mismatches"] == 0, info
        # (and the family mean is the average of the two rows, from the ARRAY not the file)
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_centred.bo2"), (0.875 + 0.375) / 2, 1e-9)
        _close(_stat(res, "realact", "ckpt-one:arm-a", "cos_centred.bo4"), (1.0 + 0.5) / 2, 1e-9)
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
        assert any(f.startswith("arms_") for f in figs), figs
        assert any(f.startswith("strata_") for f in figs), figs
        assert any(f.startswith("bok_") for f in figs), figs


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


CHECKS = [
    check_parse_scores_dir,
    check_estimators,
    check_cluster_bootstrap,
    check_round_trip,
    check_second_sae_needs_no_code,
    check_sae_side_reads_its_own_product,
    check_centred_bok_is_recomputed_from_the_array,
    check_missing_sources_are_listed_not_zeroed,
    check_sources_filter,
    check_reader_check_catches_a_defect,
    check_sanity_verdicts,
    check_cross_set_gate,
    check_render_and_figures,
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
