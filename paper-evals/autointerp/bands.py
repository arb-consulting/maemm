"""Stage `bands`: per-item recall by activation band, joined on the volume.

    <root>/runs/<run>/bands/tpr_by_band.csv       arm x scorer x band -> TPR, with n
    <root>/runs/<run>/bands/tnr_by_source.csv     arm x scorer x negative half -> TNR, with n
    <root>/runs/<run>/bands/band_composition.csv  the realised band histogram of the test sets
    <root>/runs/<run>/bands/bands.json            the totals and what was joined

Why this is a stage and not part of `stats.py`: the per-ITEM answers live in
`runs/<run>/<scorer>/batches.jsonl` (~40 MB at 512 features) and the per-item BANDS live in the
build's `<feature>.jsonl` (~90 MB). Neither is in the small summary set the paper's pull script is
allowed to fetch, and pulling both would break the no-bulk-download rule. So the join runs where
the data already is, on a CPU container, and writes a few kilobytes of CSV that the pull script
CAN fetch.

What it answers (the method review's central point): balanced accuracy here is a recall meter --
the scorer answers "0" by default, so TNR sits near 1 for every arm and the arms separate almost
entirely on TPR. Reading TPR by the positive's activation band says *where* on the activation range
each arm's description works, which a single balanced accuracy cannot.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import precompute.common as C

BANDS = ("q0", "q1", "q2", "q3", "top")
# Relative activation of a test positive, as a fraction of the feature's 16M corpus peak. Reported
# beside the stored band because the band is an equal-width bin of (0, max_act] and therefore is
# not comparable across features with different peaks, while this is.
REL_EDGES = (0.0, 0.1, 0.25, 0.5, 1.01)


def _rel_bucket(rel: float) -> str:
    for lo, hi in zip(REL_EDGES[:-1], REL_EDGES[1:], strict=True):
        if lo <= rel < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f"[{REL_EDGES[-2]:.2f},{REL_EDGES[-1]:.2f})"


def run(cfg, args):
    base, root, set_name = args["base"], args["root"], args["heldout"]
    run_name = args.get("run_dir")
    assert run_name, "stage bands needs --run-dir (the run whose per-item answers it joins)"
    run_root = f"{root}/runs/{run_name}"
    build_name = args.get("build_dir")
    assert build_name, "stage bands needs --build-dir (the build whose test rows carry the bands)"
    build_dir = f"{C.base_dir(base, root)}/autointerp/{set_name}/{build_name}"
    assert os.path.exists(f"{build_dir}/build.json"), f"no build at {build_dir}"

    feats = [f["feature"] for f in json.load(open(f"{build_dir}/features.json"))["features"]]
    peak_of = {f["feature"]: float(f["corpus_peak"])
               for f in json.load(open(f"{build_dir}/features.json"))["features"]}
    scorers = [s for s in (args.get("scorers") or "detection,fuzzing").split(",") if s]

    # item index -> (label, band, max_act, src), per feature and per draw. `batches.jsonl` records
    # the item's `i`, which is its index WITHIN its draw, and every arm scored draw 1 except the
    # draw-null arm, so the draw is recovered from the arm rather than guessed.
    item: dict[int, dict[str, dict[int, dict]]] = {}
    comp: list[dict] = []
    for f in feats:
        rows = C.read_jsonl(f"{build_dir}/{f}.jsonl")
        per_draw: dict[str, dict[int, dict]] = {"test": {}, "test2": {}}
        for r in rows:
            if r["kind"] in ("test", "test2"):
                per_draw[r["kind"]][int(r["i"])] = r
        item[f] = per_draw
        hist: dict[str, int] = defaultdict(int)
        for r in per_draw["test"].values():
            if r["label"] == 1:
                hist[r.get("band", "?")] += 1
        comp.append({
            "feature": f, "n_pos": sum(1 for r in per_draw["test"].values() if r["label"] == 1),
            **{f"n_{b}": hist.get(b, 0) for b in BANDS},
            "has_q2_or_q3": int(hist.get("q2", 0) + hist.get("q3", 0) > 0),
        })

    tpr: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    tpr_rel: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    tnr: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    feats_seen: dict[tuple, set] = defaultdict(set)
    n_rows = n_unparsed = 0
    for scorer in scorers:
        path = f"{run_root}/{scorer}/batches.jsonl"
        if not os.path.exists(path):
            print(f"[bands] no {path}, skipping {scorer}", flush=True)
            continue
        for b in C.read_jsonl(path):
            n_rows += 1
            if not b.get("parsed"):
                n_unparsed += 1
                continue  # an unparsed batch is dropped everywhere else too
            f, arm = int(b["feature"]), b["arm"]
            draw = "test2" if arm.endswith("-draw2") else "test"
            rows = item.get(f, {}).get(draw, {})
            for i, lab, pred in zip(b["items"], b["labels"], b["preds"], strict=True):
                r = rows.get(int(i))
                if r is None:
                    continue
                feats_seen[(scorer, arm)].add(f)
                if lab == 1:
                    band = r.get("band", "?")
                    tpr[(scorer, arm, band)][1] += 1
                    tpr[(scorer, arm, band)][0] += int(pred == 1)
                    peak = peak_of.get(f, 0.0) or 1.0
                    rb = _rel_bucket(float(r.get("max_act", 0.0)) / peak)
                    tpr_rel[(scorer, arm, rb)][1] += 1
                    tpr_rel[(scorer, arm, rb)][0] += int(pred == 1)
                else:
                    src = "nearmiss" if str(r.get("src", "")).startswith("nearmiss") else "zero"
                    tnr[(scorer, arm, src)][1] += 1
                    tnr[(scorer, arm, src)][0] += int(pred == 0)

    def _csv(path, header, rows):
        with open(path, "w") as fh:
            fh.write(",".join(header) + "\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")

    out = f"{run_root}/bands"
    with C.outdir(out, {**args, "force": True},
                  inputs={"run": run_root, "build": build_dir, "features": len(feats),
                          "scorers": ",".join(scorers)}) as od:
        _csv(od.file("tpr_by_band.csv"),
             ["run", "scorer", "arm", "band", "n_items", "n_correct", "tpr", "n_features"],
             [[run_name, s, a, bd, n, k, round(k / n, 6) if n else "",
               len(feats_seen[(s, a)])]
              for (s, a, bd), (k, n) in sorted(tpr.items())])
        _csv(od.file("tpr_by_rel_act.csv"),
             ["run", "scorer", "arm", "rel_act_bucket", "n_items", "n_correct", "tpr"],
             [[run_name, s, a, bk, n, k, round(k / n, 6) if n else ""]
              for (s, a, bk), (k, n) in sorted(tpr_rel.items())])
        _csv(od.file("tnr_by_source.csv"),
             ["run", "scorer", "arm", "negative_half", "n_items", "n_correct", "tnr"],
             [[run_name, s, a, src, n, k, round(k / n, 6) if n else ""]
              for (s, a, src), (k, n) in sorted(tnr.items())])
        _csv(od.file("band_composition.csv"),
             ["feature", "n_pos", *[f"n_{b}" for b in BANDS], "has_q2_or_q3"],
             [[c["feature"], c["n_pos"], *[c[f"n_{b}"] for b in BANDS], c["has_q2_or_q3"]]
              for c in comp])
        od.write_json("bands.json", {
            "run": run_name, "build": build_dir, "features": len(feats),
            "batch_rows": n_rows, "unparsed_dropped": n_unparsed,
            "features_without_q2_or_q3_positive":
                sum(1 for c in comp if c["n_pos"] > 0 and not c["has_q2_or_q3"]),
            "features_scorable": sum(1 for c in comp if c["n_pos"] > 0),
            "rel_act_edges": list(REL_EDGES),
        })
        od.note(
            "per-ITEM join of the scorers' `batches.jsonl` against the build's test rows, done on "
            "the volume because neither file is in the small summary set the paper's pull script "
            "may fetch. TPR is over the positives of PARSED batches only, the same set every "
            "other number uses."
        )
        od.note(
            "`band` is the stored equal-width bin of (0, max_act] (`precompute/scan.py:257`), so "
            "it is NOT comparable across features with different peaks; `tpr_by_rel_act.csv` uses "
            "the activation as a fraction of the feature's 16M corpus peak, which is."
        )
    return {"out": out, "features": len(feats), "batch_rows": n_rows,
            "unparsed_dropped": n_unparsed,
            "features_without_q2_or_q3_positive":
                sum(1 for c in comp if c["n_pos"] > 0 and not c["has_q2_or_q3"])}
