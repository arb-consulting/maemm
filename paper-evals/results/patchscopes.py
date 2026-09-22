#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "polars>=1", "typer>=0.15", "rich>=13", "pyyaml>=6"]
# ///
"""M7's reader: the Patchscopes appendix table, from the cells `score --rollouts-dir` wrote.

    cd /home/gavento/dev/mimir/2026-09-maemms
    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \\
     uv run repo-maemm-m7/paper-evals/results/patchscopes.py --set 2026-09-21_v3_realact \\
       --ps-tag paper0923 --out repo-maemm-m7/paper-evals/results/patchscopes)

Local, CPU, no GPU, no model. Every number is READ from what `score` wrote; this file recomputes
only the aggregation across rows, the standard errors and the paired lift.

WHY THIS FILE EXISTS RATHER THAN A BRANCH IN `results/faithfulness.py`. A patchscopes cell is not
a MAEMM: it lives at `base/<base>/patchscopes/<set>/<cell>/scores`, and
`results.common.discover_sources` walks `maemms/<key>/scores` and `maemms/<key>/variants/<d>/scores`
and nothing else (`results/README.md`: "the search baseline and the patchscopes cells are different
products with different shapes and are not read"). `results/faithfulness.py` is M1's file and
`results/common.py` is M0a's; M7 owns `precompute/patchscopes.py` "and its driver". So this module
CONSTRUCTS `results.common.Source` by hand -- the pattern `results/selftest.py` already uses -- and
then reuses M1's and M0a's estimators unchanged. Nothing is re-implemented here:

    exclusions                 results.faithfulness.load_exclusions
    document clusters          results.faithfulness._clusters_for   (the set's `doc` field)
    the centred bo-k ladder    results.faithfulness.centred_bok     (from `cos_centred.f16`)
    the bo-k means and SEs     results.faithfulness.cosine_cells
    the clustered bootstrap    results.common.cluster_bootstrap     (2000 resamples, seed 20260921)
    the exact sign test        reconstruction.stats.sign_test

THE CONVENTION IS `common.score_mu`. Both arguments of every number here are centred on the base's
scoring constant `bases.qwen36-27b.whiten_mu`: `score`'s `_load_dirs` takes the target as
`unit(act - whiten_mu)` and `common.score_rollouts` takes the activation as `h - mu` with the same
mu. The raw column is carried beside it for continuity with the 2026-09-16 table and is NOT the
paper's number. The table says so in its caption, which is the plan's M7 gate.

THE FLOOR IS SHARED AND THAT IS NOT A SHORTCUT. The floor cell installs no hook at all
(`precompute/patchscopes.py:_generate`, `layer is None`), so it does not depend on the patched
layer, and its generation seed is `common.gen_seed_for(base_seed, sel[0], s, n)` -- identical
across four calls that share `--rows` and `--n`. Four floor cells would be four byte-identical
copies. One floor cell, matched to every injected cell on rows, on n, on prompt, on sampling
constants and on the scoring pass, is the control; the pairing is on the `--ps-tag`, exactly as
`reconstruction/stats.py:table_j` already pairs it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reconstruction.stats as S  # noqa: E402
import results.common as R  # noqa: E402
import results.faithfulness as F  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

ENGINE = "hf-patchscope"
FLOOR_PREFIX = "floor"
# The k's the appendix prints. spec 1.3: k = 1 is the mean over the draws and k = 8 is what a user
# reads; k = 64 is only for a comparator that is itself a maximum over a corpus, which a floor is
# not. Both are the unbiased order-statistic estimator over all n draws, same as every other arm.
CELL_KS = (1, 8)
# `<eval>.<set>.<arm>.<metric>[.<k>]` (paper/numbers/README.md). The grammar has no layer slot, so
# the layer rides in the ARM slot -- the plan's Open 3 ruling. An unknown arm VALUE warns and does
# not error, which is why this is legal at all.
KEY_EVAL, KEY_SET, KEY_METRIC = "fid", "ra", "cos"


def cell_layer(cell: str) -> int | None:
    """The patched layer of a cell directory name, or None for the floor.

    `precompute/patchscopes.py:cell_name` builds `p2-L<layer>-<rule><alpha>[__<tag>]`, or
    `floor[__<tag>]`. Parsed rather than assumed, so a cell this reader cannot name is skipped
    loudly instead of being folded into the wrong row.
    """
    head = cell.split("__", 1)[0]
    if head == FLOOR_PREFIX:
        return None
    parts = head.split("-")
    assert len(parts) >= 3 and parts[1].startswith("L"), (
        f"cell {cell!r} is neither `floor` nor `<prompt>-L<layer>-<rule><alpha>`; "
        f"this reader will not guess which layer it patched"
    )
    return int(parts[1][1:])


def cell_tag(cell: str) -> str:
    """The `--ps-tag` suffix of a cell name, or "" (precompute/patchscopes.py:cell_name)."""
    return cell.split("__", 1)[1] if "__" in cell else ""


def source_for(base: str, set_name: str, cell: str) -> R.Source:
    """A `results.common.Source` pointing at one patchscopes cell's `scores/`.

    `maemm` is not a config key here and never will be -- there is no MAEMM in this product, the
    rollouts come off the CLEAN BASE. It carries the cell name so every label, CSV column and
    figure legend downstream says which cell a number is from.
    """
    d = f"base/{base}/patchscopes/{set_name}/{cell}"
    return R.Source(
        maemm=f"patchscopes/{cell}",
        base=base,
        engine=ENGINE,
        run_tag=cell_tag(cell),
        scores_rel=f"{d}/scores",
        rollouts_rel=f"{d}/rollouts.summary.json",
        role="secondary",
    )


def discover_cells(vol: R.Vol, base: str, set_name: str, ps_tag: str) -> list[str]:
    """The cell directories of one set that have a `scores/`, filtered to one `--ps-tag`.

    A cell without `scores/` is generated but not yet scored: it is reported as pending rather
    than silently dropped, because "the table is short" and "the run is not finished" are
    different states and the caller must be able to tell them apart.
    """
    root = f"base/{base}/patchscopes/{set_name}"
    out = []
    for c in vol.ls(root):
        if ".tmp-" in c or cell_tag(c) != ps_tag:
            continue
        out.append(c)
    return sorted(out, key=lambda c: (cell_layer(c) is not None, cell_layer(c) or 0))


def paired_lift(inj_bok: dict[int, dict[int, float]], floor_bok: dict[int, dict[int, float]],
                rows: list[int], ids: dict[int, dict], k: int, boot: int, seed: int) -> dict:
    """The per-row bo-k lift of an injected cell over its floor: mean, clustered SE, sign test.

    PAIRED ON THE ROW, because the floor and the injected cell score the SAME direction: the
    difference of two means over the same rows has a much smaller SE than the difference of two
    independent means, and the sign test is only meaningful row-wise. The bootstrap resamples
    DOCUMENTS of the difference, so a document contributing two rows contributes them together --
    `results.common.cluster_bootstrap`'s own contract, applied to the paired quantity.
    """
    shared = [r for r in rows if k in inj_bok.get(r, {}) and k in floor_bok.get(r, {})]
    if not shared:
        return {}
    d = np.array([inj_bok[r][k] - floor_bok[r][k] for r in shared], dtype=np.float64)
    mean, se, n_items, n_clusters = R.cluster_bootstrap(d, F._clusters_for(shared, ids), boot, seed)
    p, win, m = S.sign_test(d)
    return {"mean": mean, "se": se, "se_iid": R.se_iid(d), "n_rows": n_items,
            "n_clusters": n_clusters, "sign_p": p, "win": win, "n_nonties": m}


def fmt(mean: float, se: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} ± {se:.{nd}f}" if np.isfinite(se) else f"{mean:.{nd}f}"


def analyse(vol: R.Vol, cfg: dict, base: str, set_name: str, ps_tag: str,
            boot: int, seed: int, max_mb: float, apply_exclusions: bool) -> dict:
    """Everything the table and the cells rows are built from, as plain data."""
    ids, declared_sae = R.load_ids(vol, base, set_name, cfg)
    exc = F.load_exclusions(vol, base, set_name)
    drop = exc["rows"] if apply_exclusions else set()
    # realact only: this baseline is defined on real activation directions. A `storage: raw` set
    # whose other families were not centrable would carry NaN in `cos_centred` anyway, and a row
    # this reader cannot centre is absent rather than reported one-sided.
    realact = [r for r in sorted(ids) if ids[r]["family"] == "realact" and r not in drop]
    assert realact, f"{set_name} has no realact rows left after {len(drop)} exclusions"

    names = discover_cells(vol, base, set_name, ps_tag)
    assert names, (
        f"no patchscopes cell under base/{base}/patchscopes/{set_name} carries --ps-tag "
        f"{ps_tag!r}; nothing to read. Generated so far: "
        f"{vol.ls(f'base/{base}/patchscopes/{set_name}')}"
    )
    cells: dict[str, dict] = {}
    pending: list[str] = []
    for name in names:
        src = source_for(base, set_name, name)
        why = R.load_source(vol, src)
        if why:
            pending.append(f"{name}: {why}")
            continue
        bok, info = F.centred_bok(vol, src, max_mb)
        assert bok, (
            f"{name}: no centred bo-k ladder ({info.get('skipped', 'empty')}). The paper's number "
            f"for this product IS the centred one, so an empty ladder is a stop, not a column to "
            f"leave blank"
        )
        fam = R.family_of(ids[realact[0]], declared_sae)
        rows = [r for r in realact if r in src.per_target]
        cells[name] = {
            "cell": name,
            "layer": cell_layer(name),
            "src": src,
            "bok": bok,
            "bok_info": info,
            "rows": rows,
            "cos": F.cosine_cells(fam, rows, src, ids, boot, seed, bok),
        }
    floors = [c for c in cells.values() if c["layer"] is None]
    assert len(floors) <= 1, f"more than one floor cell at --ps-tag {ps_tag!r}: {[c['cell'] for c in floors]}"
    floor = floors[0] if floors else None
    for c in cells.values():
        if c["layer"] is None or floor is None:
            continue
        shared = [r for r in c["rows"] if r in set(floor["rows"])]
        c["lift"] = {k: paired_lift(c["bok"], floor["bok"], shared, ids, k, boot, seed)
                     for k in CELL_KS}
    return {"base": base, "set": set_name, "ps_tag": ps_tag, "ids": ids, "exclusions": exc,
            "apply_exclusions": apply_exclusions, "realact": realact, "cells": cells,
            "floor": floor["cell"] if floor else None, "pending": pending,
            "boot": boot, "seed": seed}


def centred_of(c: dict) -> dict:
    """The `cos_centred` record of one cell, or {} -- `cosine_cells` emits one per cosine."""
    return next((x for x in c["cos"] if x["cosine"] == "cos_centred"), {})


def raw_of(c: dict) -> dict:
    return next((x for x in c["cos"] if x["cosine"] == "cos_raw"), {})


def render(res: dict) -> str:
    """The appendix table, as markdown."""
    L = []
    L.append(f"# Patchscopes on `{res['set']}` — layer sweep against the no-injection floor\n")
    L.append(f"base `{res['base']}`, `--ps-tag {res['ps_tag']}`, "
             f"prompt P2 (Patchscopes D.1 entity description), rule `replace` at alpha 2, "
             f"read layer {res.get('read_layer', 42)}.\n")
    L.append("**Convention: centred, under `common.score_mu`.** Both arguments of every number in "
             "the `cos_centred` columns are taken about `bases.<base>.whiten_mu` — the activation "
             "as `h - mu` and the target as `unit(act - mu)` — which is the same constant the "
             "Exemplifier, the base control and the corpus search are read under, so the three are "
             "differences of one statistic. `cos_raw` is carried for continuity with the "
             "2026-09-16 table and is not the paper's number.\n")
    e = res["exclusions"]
    if res["apply_exclusions"] and e.get("present"):
        L.append(f"Exclusions: {len(e['rows'])} of {e['n_total']} rows dropped, n = "
                 f"{len(res['realact'])} — criterion `{e['criterion']}` at n-gram {e['n_gram']}, "
                 f"from `{e['source']}`.\n")
    elif not res["apply_exclusions"]:
        L.append("Exclusions: **NOT applied** (`--no-exclusions`); these are all 512 rows.\n")
    else:
        L.append("Exclusions: none — the set carries no `exclusions.json`.\n")
    L.append(f"SEs are the document-clustered bootstrap, {res['boot']} resamples, seed "
             f"{res['seed']} (`results.common.cluster_bootstrap`); a row with no document is its "
             f"own cluster. The lift is PAIRED per direction and its sign test is exact.\n")
    if res["floor"] is None:
        L.append("**No floor cell at this tag** — every lift column is empty, and no number here "
                 "may be quoted, because an injected cosine without its matched floor is mostly "
                 "measuring that fluent English has a non-trivial cosine with a real direction.\n")

    hdr = ["cell", "layer", "n rows", "n docs", "bo1 centred", "bo8 centred",
           "bo1 raw", "bo8 raw", "lift bo8 over floor", "beats its floor", "sign-test p"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for c in sorted(res["cells"].values(), key=lambda x: (x["layer"] is not None, x["layer"] or 0)):
        cen, raw = centred_of(c), raw_of(c)
        lift = (c.get("lift") or {}).get(8, {})
        L.append("| " + " | ".join([
            f"`{c['cell']}`",
            "— (floor)" if c["layer"] is None else str(c["layer"]),
            str(cen.get("n_rows", len(c["rows"]))),
            str(cen.get("n_clusters", "")),
            *[fmt(cen["bo"][k]["mean"], cen["bo"][k]["se"]) if cen and k in cen["bo"] else ""
              for k in CELL_KS],
            *[fmt(raw["bo"][k]["mean"], raw["bo"][k]["se"]) if raw and k in raw["bo"] else ""
              for k in CELL_KS],
            fmt(lift["mean"], lift["se"]) if lift else "",
            f"{lift['win']:.3f}" if lift and np.isfinite(lift["win"]) else "",
            f"{lift['sign_p']:.3g}" if lift and np.isfinite(lift["sign_p"]) else "",
        ]) + " |")
    L.append("")
    for c in sorted(res["cells"].values(), key=lambda x: (x["layer"] is not None, x["layer"] or 0)):
        i = c["bok_info"]
        L.append(f"- `{c['cell']}`: centred ladder from `{i.get('product', '?')}` "
                 f"({i.get('mb', '?')} MB), {i.get('rows', 0)} rows, "
                 f"{i.get('rows_with_nan_rollouts', 0)} row(s) with a NaN rollout, "
                 f"{i.get('stored_comparisons', 0)} stored `bo_c_k` cross-checks with "
                 f"{i.get('n_mismatches', 0)} mismatch(es).")
    for p in res["pending"]:
        L.append(f"- PENDING {p}")
    L.append("")
    return "\n".join(L)


def cells_rows(res: dict, run_id: str, date: str, status: str) -> list[dict]:
    """The `paper/numbers/cells.csv` rows M7 owns, one per (layer, arm, k).

    The floor keys are per layer and carry the SAME value four times, because there is one floor
    cell: the paper's table prints a floor beside each layer, and `\\N{fid.ra.psfloor14.cos.bo8}`
    has to resolve. The note says so on every one of them rather than leaving a reader to wonder
    why four measurements agree to the digit.
    """
    floor = res["cells"].get(res["floor"]) if res["floor"] else None
    rows = []
    layers = sorted(c["layer"] for c in res["cells"].values() if c["layer"] is not None)
    base_note = (
        f"cos_centred from cos_centred.f16 (recomputed); {res['set']} realact block minus its own "
        f"exclusions.json rows; whiten_mu centred (common.score_mu, both arguments); doc-clustered "
        f"bootstrap SE, {res['boot']} resamples, seed {res['seed']}"
    )
    for c in sorted(res["cells"].values(), key=lambda x: (x["layer"] is None, x["layer"] or 0)):
        if c["layer"] is None:
            continue
        cen = centred_of(c)
        if not cen:
            continue
        for k in CELL_KS:
            if k not in cen["bo"]:
                continue
            cell = cen["bo"][k]
            rows.append({
                "key": f"{KEY_EVAL}.{KEY_SET}.ps{c['layer']}.{KEY_METRIC}.bo{k}",
                "value": f"{cell['mean']:.4f}", "se": f"{cell['se']:.4f}", "lo": "", "hi": "",
                "n": str(cell["n_rows"]), "status": status, "run": run_id,
                "source": c["src"].scores_rel, "date": date,
                "note": (f"Patchscopes P2/replace2 patched at block {c['layer']}, bo{k}; "
                         f"{base_note}; {cell['n_clusters']} documents"),
            })
    if floor is not None:
        cen = centred_of(floor)
        for layer in layers:
            for k in CELL_KS:
                if not cen or k not in cen["bo"]:
                    continue
                cell = cen["bo"][k]
                rows.append({
                    "key": f"{KEY_EVAL}.{KEY_SET}.psfloor{layer}.{KEY_METRIC}.bo{k}",
                    "value": f"{cell['mean']:.4f}", "se": f"{cell['se']:.4f}", "lo": "", "hi": "",
                    "n": str(cell["n_rows"]), "status": status, "run": run_id,
                    "source": floor["src"].scores_rel, "date": date,
                    "note": (f"Patchscopes no-injection floor, bo{k}; ONE floor cell serves every "
                             f"layer — the floor arm installs no hook, so it does not depend on "
                             f"the patched layer, and the four per-layer floor keys carry the same "
                             f"measurement by construction; {base_note}; "
                             f"{cell['n_clusters']} documents"),
                })
    return rows


def write_cells_csv(path: Path, rows: list[dict]) -> None:
    import csv

    cols = ["key", "value", "se", "lo", "hi", "n", "status", "run", "source", "date", "note"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------------------------
# selftest: local, no volume, no network. Every check here has been seen RED.
# ---------------------------------------------------------------------------------------------

SELF_SET = "selftest_v3_realact"
SELF_TAG = "t0"
SELF_N = 8
SELF_WIDTH = 2
# row -> (document, floor value, injected value). Rows 0 and 1 SHARE document 100, which is what
# makes the clustered SE differ from the iid one; row 5 is the excluded row.
SELF_ROWS = {
    0: (100, 0.10, 0.20),
    1: (100, 0.12, 0.18),
    2: (101, 0.08, 0.26),
    3: (102, 0.20, 0.22),
    4: (103, 0.14, 0.10),   # one row where the patch LOSES, so the sign test is not degenerate
    5: (104, 0.90, 0.95),   # excluded: if it ever reaches a number, the exclusions are not applied
}
SELF_EXCLUDE = [5]


def _self_mirror(mirror: Path, base: str, injected: dict[int, float] | None = None) -> None:
    """Write a synthetic volume mirror with one floor cell and one L14 cell."""
    inj = injected or {r: v[2] for r, v in SELF_ROWS.items()}
    hd = mirror / f"base/{base}/heldout/{SELF_SET}"
    hd.mkdir(parents=True, exist_ok=True)
    with open(hd / "ids.jsonl", "w") as fh:
        for r, (doc, _f, _i) in SELF_ROWS.items():
            fh.write(json.dumps({"row": r, "family": "realact", "doc": doc, "pos": r,
                                 "id": f"doc{doc}:pos{r}"}) + "\n")
    (hd / "storage.json").write_text(json.dumps({"storage": "raw", "source": "selftest"}))
    (hd / "exclusions.json").write_text(json.dumps(
        {"block": "realact", "rows_total": len(SELF_ROWS), "excluded_rows": SELF_EXCLUDE,
         "n_headline": len(SELF_ROWS) - len(SELF_EXCLUDE), "criterion": 0.05, "n_gram": 7,
         "source": "selftest", "computed_over": "selftest"}))

    for cell, vals in ((f"floor__{SELF_TAG}", {r: v[1] for r, v in SELF_ROWS.items()}),
                       (f"p2-L14-replace2__{SELF_TAG}", inj)):
        d = mirror / f"base/{base}/patchscopes/{SELF_SET}/{cell}/scores"
        d.mkdir(parents=True, exist_ok=True)
        rows = sorted(SELF_ROWS)
        with open(d / "per_target.jsonl", "w") as fh:
            for r in rows:
                # The production shape: `mean_cos_centred` but NO `bo_c_k` ladder, so the centred
                # numbers can only come from the array (score.py writes the ladder only when every
                # rollout of the row has a finite centred best).
                fh.write(json.dumps({
                    "row": r, "family": "realact", "n": SELF_N, "bo": SELF_N, "seed": 1234,
                    "checkpoint_sha": "deadbeef", "mean_cos": vals[r], "max_cos": vals[r],
                    "mean_cos_centred": vals[r], "max_cos_centred": vals[r], "n_centred": SELF_N,
                    **{f"bo_{k}": vals[r] for k in R.BO_KS_ALL if k <= SELF_N},
                }) + "\n")
        (d / "rows.json").write_text(json.dumps(
            {"rows": rows, "n": SELF_N, "families": ["realact"] * len(rows),
             "score_max_length": 95, "mu": "/vol/whiten_mu.npy"}))
        index = {"per_target.jsonl": {"kind": "jsonl", "rows": len(rows)},
                 "rows.json": {"kind": "json"}}
        for name in ("cos.f16", "cos_centred.f16"):
            arr = np.stack([np.full((SELF_N, SELF_WIDTH), vals[r], dtype=np.float32) for r in rows])
            arr.astype(np.float16).tofile(d / name)
            index[name] = {"kind": "array", "dtype": "float16",
                           "shape": list(arr.shape), "bytes": arr.astype(np.float16).nbytes}
        (d / "index.json").write_text(json.dumps(index))


def selftest(base: str = "qwen36-27b") -> int:
    """Run every check; print one line each; return the number that failed."""
    import math
    import tempfile

    cfg = R.load_config()
    kept = [r for r in sorted(SELF_ROWS) if r not in SELF_EXCLUDE]
    checks: list[tuple[str, callable]] = []

    def check(name):
        def deco(fn):
            checks.append((name, fn))
            return fn
        return deco

    @check("cell_layer parses the floor, a cell and refuses a name it cannot read")
    def _c1():
        assert cell_layer("floor") is None and cell_layer(f"floor__{SELF_TAG}") is None
        assert cell_layer("p2-L14-replace2") == 14
        assert cell_layer("p2-L42-replace2__paper0923") == 42
        assert cell_tag("p2-L42-replace2__paper0923") == "paper0923"
        try:
            cell_layer("nonsense__t")
        except AssertionError:
            return
        raise AssertionError("cell_layer accepted a name it cannot parse")

    @check("exclusions leave the numbers: the excluded row never reaches a mean")
    def _c2(res_on, res_off):
        cen_on = centred_of(res_on["cells"][f"p2-L14-replace2__{SELF_TAG}"])
        cen_off = centred_of(res_off["cells"][f"p2-L14-replace2__{SELF_TAG}"])
        assert cen_on["bo"][1]["n_rows"] == len(kept), cen_on["bo"][1]["n_rows"]
        assert cen_off["bo"][1]["n_rows"] == len(SELF_ROWS), cen_off["bo"][1]["n_rows"]
        want = sum(SELF_ROWS[r][2] for r in kept) / len(kept)
        assert abs(cen_on["bo"][1]["mean"] - want) < 1e-3, (cen_on["bo"][1]["mean"], want)
        assert cen_off["bo"][1]["mean"] > cen_on["bo"][1]["mean"] + 0.05, "row 5 did not move it"

    @check("the centred ladder comes from cos_centred.f16, not from per_target.jsonl")
    def _c3(res_on, _off):
        for c in res_on["cells"].values():
            assert centred_of(c)["bo_source"].startswith("cos_centred.f16"), centred_of(c)

    @check("the SE is document-clustered, not the iid one")
    def _c4(res_on, _off):
        cen = centred_of(res_on["cells"][f"p2-L14-replace2__{SELF_TAG}"])
        cell = cen["bo"][1]
        assert math.isfinite(cell["se"]) and cell["se"] > 0
        assert abs(cell["se"] - cell["se_iid"]) > 1e-4, (cell["se"], cell["se_iid"])
        assert cell["n_clusters"] == len({SELF_ROWS[r][0] for r in kept}), cell["n_clusters"]

    @check("the paired lift and its sign test are the hand-worked ones")
    def _c5(res_on, _off):
        lift = res_on["cells"][f"p2-L14-replace2__{SELF_TAG}"]["lift"][1]
        d = [SELF_ROWS[r][2] - SELF_ROWS[r][1] for r in kept]
        assert abs(lift["mean"] - sum(d) / len(d)) < 1e-3, (lift["mean"], sum(d) / len(d))
        assert lift["win"] == sum(1 for x in d if x > 0) / len(d), lift["win"]
        assert lift["n_nonties"] == len(d)

    @check("one floor serves every layer and its keys carry the floor's own number")
    def _c6(res_on, _off):
        rows = cells_rows(res_on, "R11", "2026-09-23", "final")
        keys = {r["key"]: r for r in rows}
        assert "fid.ra.ps14.cos.bo1" in keys and "fid.ra.psfloor14.cos.bo1" in keys, sorted(keys)
        cen = centred_of(res_on["cells"][f"floor__{SELF_TAG}"])
        assert keys["fid.ra.psfloor14.cos.bo1"]["value"] == f"{cen['bo'][1]['mean']:.4f}"
        assert keys["fid.ra.psfloor14.cos.bo1"]["source"].endswith(f"floor__{SELF_TAG}/scores")
        assert all(int(r["n"]) == len(kept) for r in rows), [r["n"] for r in rows]

    @check("MUTATION: moving a kept row's array moves the number by exactly its share")
    def _c7(res_on, _off):
        bump = 0.4
        mutated = {r: (v[2] + bump if r == 2 else v[2]) for r, v in SELF_ROWS.items()}
        with tempfile.TemporaryDirectory() as td:
            m = Path(td)
            _self_mirror(m, base, mutated)
            res = analyse(R.Vol("", m, offline=True, quiet=True), cfg, base, SELF_SET, SELF_TAG,
                          200, R.BOOT_SEED, 128.0, True)
        got = centred_of(res["cells"][f"p2-L14-replace2__{SELF_TAG}"])["bo"][1]["mean"]
        was = centred_of(res_on["cells"][f"p2-L14-replace2__{SELF_TAG}"])["bo"][1]["mean"]
        assert abs((got - was) - bump / len(kept)) < 2e-3, (got, was, bump / len(kept))

    @check("MUTATION: moving the EXCLUDED row's array moves nothing")
    def _c8(res_on, _off):
        mutated = {r: (v[2] - 0.5 if r == 5 else v[2]) for r, v in SELF_ROWS.items()}
        with tempfile.TemporaryDirectory() as td:
            m = Path(td)
            _self_mirror(m, base, mutated)
            res = analyse(R.Vol("", m, offline=True, quiet=True), cfg, base, SELF_SET, SELF_TAG,
                          200, R.BOOT_SEED, 128.0, True)
        got = centred_of(res["cells"][f"p2-L14-replace2__{SELF_TAG}"])["bo"][1]["mean"]
        was = centred_of(res_on["cells"][f"p2-L14-replace2__{SELF_TAG}"])["bo"][1]["mean"]
        assert abs(got - was) < 1e-9, (got, was)

    with tempfile.TemporaryDirectory() as td:
        mirror = Path(td)
        _self_mirror(mirror, base)
        vol = R.Vol("", mirror, offline=True, quiet=True)
        res_on = analyse(vol, cfg, base, SELF_SET, SELF_TAG, 200, R.BOOT_SEED, 128.0, True)
        res_off = analyse(vol, cfg, base, SELF_SET, SELF_TAG, 200, R.BOOT_SEED, 128.0, False)
        res_on["read_layer"] = cfg["bases"][base]["read_layer"]
        assert "Patchscopes" in render(res_on) and "score_mu" in render(res_on)
        bad = 0
        for name, fn in checks:
            try:
                fn() if fn.__code__.co_argcount == 0 else fn(res_on, res_off)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"[ps-selftest] FAIL  {name}\n              {type(e).__name__}: {e}")
            else:
                print(f"[ps-selftest] ok    {name}")
    print(f"[ps-selftest] {len(checks) - bad}/{len(checks)} checks passed")
    return bad


@app.command()
def main(
    # `set_`, not `set`: the parameter shadowed the builtin inside this function and the
    # results.json dump's `isinstance(v, set)` died on it. faithfulness.py spells it the same way.
    set_: Annotated[str, typer.Option("--set", help="the held-out set the cells were run on")] = "",
    base: str = "qwen36-27b",
    ps_tag: Annotated[str, typer.Option("--ps-tag", help="which run's cells to read")] = "paper0923",
    out: Annotated[str, typer.Option(help="output directory")] = "",
    root: Annotated[str, typer.Option(help="volume-relative root prefix")] = "",
    data: Annotated[str, typer.Option(help="local mirror; default results/data/<root>")] = "",
    fetch: Annotated[bool, typer.Option("--fetch/--no-fetch")] = True,
    refetch: bool = False,
    modal_cmd: str = "uvx modal",
    exclusions: Annotated[bool, typer.Option("--exclusions/--no-exclusions")] = True,
    boot: int = R.N_BOOT,
    seed: int = R.BOOT_SEED,
    centred_bok_max_mb: float = 128.0,
    run_id: Annotated[str, typer.Option(help="spec 7 run id for the cells rows")] = "R11",
    date: Annotated[str, typer.Option(help="the date column of the cells rows")] = "",
    status: Annotated[str, typer.Option(help="placeholder|provisional|final")] = "final",
    quiet: bool = False,
    run_selftest: Annotated[bool, typer.Option("--selftest", help="run the local checks and exit")] = False,
):
    import datetime as _dt

    if run_selftest:
        raise typer.Exit(1 if selftest(base) else 0)
    assert set_, "--set is required"
    assert status in ("placeholder", "provisional", "final"), f"bad --status {status!r}"
    here = Path(__file__).resolve().parent
    data_dir = Path(data) if data else here / "data" / (root.replace("/", "_") if root else "vol")
    outdir = Path(out) if out else here / "out" / "patchscopes"
    outdir.mkdir(parents=True, exist_ok=True)
    vol = R.Vol(root, data_dir, modal_cmd, refetch=refetch, quiet=quiet, offline=not fetch)
    cfg = R.load_config()

    res = analyse(vol, cfg, base, set_, ps_tag, boot, seed, centred_bok_max_mb, exclusions)
    res["read_layer"] = cfg["bases"][base]["read_layer"]
    md = render(res)
    (outdir / "tables.md").write_text(md)
    rows = cells_rows(res, run_id, date or _dt.date.today().isoformat(), status)
    write_cells_csv(outdir / "cells.csv", rows)
    dump = {
        "base": res["base"], "set": res["set"], "ps_tag": res["ps_tag"],
        "floor": res["floor"], "pending": res["pending"],
        "n_realact": len(res["realact"]), "boot": res["boot"], "seed": res["seed"],
        "exclusions": {k: (sorted(v) if isinstance(v, set) else v)
                       for k, v in res["exclusions"].items()},
        "cells": {
            name: {"layer": c["layer"], "n_rows": len(c["rows"]),
                   "scores_rel": c["src"].scores_rel, "n": c["src"].n, "mu": c["src"].mu,
                   "bok_info": c["bok_info"], "cos": c["cos"],
                   "lift": {str(k): v for k, v in (c.get("lift") or {}).items()}}
            for name, c in res["cells"].items()
        },
    }
    (outdir / "results.json").write_text(json.dumps(dump, indent=1, default=str))
    print(md)
    print(f"[patchscopes] {len(res['cells'])} cell(s), floor {res['floor']}, "
          f"{len(rows)} cells.csv row(s) -> {outdir}")
    if vol.missing:
        print(f"[patchscopes] {len(vol.missing)} file(s) missing from the volume: {vol.missing[:5]}")


if __name__ == "__main__":
    app()
