"""The run's reading (methodology §7): the judged pairs, their likelihood gaps and the source passage's
cosines loaded into a `frontier_analysis.Reading`, and the tables `report` writes from it.

- `frontier_matched.csv`: every point of every method's curve, both cosines, the net per judge (and with
  unanswered pairs dropped), the likelihood gap and the mean token count;
- `frontier_outcomes.csv`: win / tie / loss per judged method at its per-sample and greedy cells;
- `missingness.csv`: what came back for every request, per group.
"""
import numpy as np

from eval.common.stats import boot_indices
from eval.rollout_coherence import analysis as A
from eval.rollout_coherence import config as C
from eval.rollout_coherence import frontier_analysis as FR
from eval.rollout_coherence import frontier_context as FX
from eval.rollout_coherence.fluency import FLUENCY_OUT

#: the population name in `frontier_outcomes.csv` and the `condition` of `missingness.csv`
MATCHED_SETTING = "matched"


def read_targets(run):
    """The `targets` stage's small arrays (not `h` or `dirs`) as a dict; a missing key is an empty array."""
    with np.load(run.file(FX.TARGETS_NPZ)) as z:
        have = set(z.files)
        return {key: (np.asarray(z[key]) if key in have else np.zeros(0))
                for key in ("idx", "n_tokens", "last_kept", "ceiling_raw")}


def source_samples(run):
    """`Reading.source`: the source passage's own cosines against its matched target."""
    return [{"i": r["i"], "cos": r.get("cos_raw"), "cos_centred": r.get("cos_centred"),
             "n_tokens": r.get("n_tokens")}
            for r in run.read_jsonl(FX.SCORES.format("source"))]


def matched_reading(pairs, recs_by_judge, gaps, source, pos_of, n):
    """The `frontier_analysis.Reading` of the run: every judged pair, x its own matched cosines."""
    return FR.Reading(
        pairs=list(pairs),
        nets=FR.nets_by_pid(recs_by_judge),
        nets_dropped=FR.nets_by_pid(recs_by_judge, answered_only=True),
        gaps=gaps, source=source, pos_of=pos_of, n=n, labels=FR.manifest_labels(pairs),
        centred={p["pid"]: (p.get("extra") or {}).get("cos_centred") for p in pairs})


def matched_coverage(run, doc, pairs, recs_by_judge, gaps, labels, targets, m):
    """`coverage_and_costs.json`'s judged block: what the judge and likelihood passes produced, the targets'
    own record, the checks every method ran under, and the axis maxima."""
    config = doc.get("config") or {}
    skipped = config.get("skipped") or []
    by_group = FR.skipped_by_group(config)
    groups = sorted({p.get("group") for p in pairs} | set(by_group))
    ceiling = FR.mean_finite(np.asarray(targets.get("ceiling_raw"), dtype=float).tolist())
    kept = np.asarray(targets.get("last_kept"), dtype=bool)
    stages = {m_: run.read_json(f"stages/frontier_context_{m_}.json") for m_ in C.CONTEXT_METHODS
              if run.exists(f"stages/frontier_context_{m_}.json")}
    return {
        "n": int(m),
        "n_pairs": len(pairs),
        "groups": groups,
        "sizes": list(labels),
        "axis_max_raw": float(ceiling) if np.isfinite(ceiling) else None,
        "axis_max_centred": FR.AXIS_MAX_CENTRED,
        "last_kept": {"n": int(kept.sum()) if kept.size else 0,
                      "share": float(kept.mean()) if kept.size else float("nan")},
        "pairs": FR.pairs_coverage(recs_by_judge, groups, by_group),
        "skipped": {"total": len(skipped), "by_group": by_group, "records": skipped},
        "window_short": dict(config.get("n_window_short") or {}),
        "fluency": {"present": run.exists(FLUENCY_OUT), "n_pairs_scored": len(gaps)},
        "judged": {judge: sum(1 for r in recs if r["outcome"] is not None)
                   for judge, recs in recs_by_judge.items()},
        "targets": run.read_json(FX.TARGETS_JSON) if run.exists(FX.TARGETS_JSON) else {},
        "selfcheck": {m_: run.read_json(FX.SELFCHECK.format(m_)) for m_ in C.CONTEXT_METHODS
                      if run.exists(FX.SELFCHECK.format(m_))},
        "stages": stages,
    }


def matched_tables(run):
    """`(tables, coverage, points, records by judge)` over the run's activations on one resample matrix."""
    judges = list(C.JUDGES)
    targets = read_targets(run)
    idx_act = np.asarray(targets.get("idx"), dtype=int)
    m = int(idx_act.size)
    pos_of = {int(a): p for p, a in enumerate(idx_act)}
    idx = boot_indices(m, C.N_BOOT, C.BOOT_SEED)

    doc = run.read_json(FX.PAIRS)
    pairs = doc.get("pairs") or []
    recs_by_judge = FR.judged_records(run, pairs, judges, FX.PAIRS)
    # methodology §6's unparseable-share gate
    for recs in recs_by_judge.values():
        A.check_parse_share(recs)
    gaps = FR.likelihood_gaps(run, FLUENCY_OUT, pairs)
    reading = matched_reading(pairs, recs_by_judge, gaps, source_samples(run), pos_of, m)
    entries, points = FR.setting_points(reading, judges, idx)
    positioned = {j: FR.positioned(recs, pos_of) for j, recs in recs_by_judge.items()}
    tables = {
        "frontier_matched": FR.frontier_rows(entries, points, judges, m),
        "frontier_outcomes": FR.outcome_rows(MATCHED_SETTING, positioned, m, idx, reading.labels),
        "missingness": A.missingness_rows(run, pairs, judges, MATCHED_SETTING),
    }
    coverage = {"judged": matched_coverage(run, doc, pairs, recs_by_judge, gaps, reading.labels, targets, m)}
    return tables, coverage, points, recs_by_judge
