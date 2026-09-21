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
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


CHECKS = [
    check_parse_scores_dir,
    check_estimators,
    check_cluster_bootstrap,
    check_round_trip,
    check_second_sae_needs_no_code,
    check_missing_sources_are_listed_not_zeroed,
    check_sources_filter,
    check_reader_check_catches_a_defect,
    check_sanity_verdicts,
    check_cross_set_gate,
    check_render_and_figures,
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
