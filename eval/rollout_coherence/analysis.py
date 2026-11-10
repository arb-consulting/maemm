"""`report`'s statistics (methodology §5-§7): the two-order outcome of a pair, the cells and their
intervals, the missingness table and the 2 % unparseable guard. `analyse` builds every table the report
writes and refuses a run directory missing any artifact.
"""

import numpy as np

from eval.common.runs import gpu_cost_block, launch_gpu_rate, launch_gpu_seconds, launch_records, read_provenance
from eval.common.stats import CI_WILSON, row, wilson
from eval.rollout_coherence import config as C
from eval.rollout_coherence import judge as J

NET = {"win": 1, "tie": 0, "loss": -1}
CI_BOOT_ACT = "bootstrap_activation_10000_seed0"


def side(choice, order):
    if choice is None: return None
    if choice == "tie": return "tie"
    return "x" if choice == ("A" if order == 0 else "B") else "partner"


def outcome(s0, s1):
    if s0 == "x" and s1 == "x": return "win"
    if s0 == "partner" and s1 == "partner": return "loss"
    return "tie"


def score_pairs(pairs, answers):
    """One judge's record per pair; an unanswered order is a tie, and `answered` marks a pair with both."""
    recs = []
    for p in pairs:
        a = [answers.get((p["pid"], o)) for o in (0, 1)]
        if any(x is None for x in a):
            recs.append({**p, "judge_raw": [None, None], "status": [None, None], "side": [None, None], "outcome": None, "split": False,
                         "n_refused": 0, "n_parse_fail": 0, "net": None, "at_least_as": None,
                         "answered": False}); continue
        raw = [x["choice"] for x in a]; st = [x["status"] for x in a]
        sd = [side(c, o) for o, c in enumerate(raw)]
        sd = ["tie" if s is None else s for s in sd]      # refused / parse_fail orders score as ties
        oc = outcome(*sd)
        recs.append({**p, "judge_raw": raw, "status": st, "side": sd, "outcome": oc, "split": {"x", "partner"} == set(sd),
                     "n_refused": sum(s == "refused" for s in st), "n_parse_fail": sum(s == "parse_fail" for s in st),
                     "net": NET[oc], "at_least_as": int(oc != "loss"),
                     "answered": all(s == "ok" for s in st)})
    return recs


def condition_pairs(recs, group, condition):
    """One cell's records: `per_sample` every pair of `group` but its greedy, `greedy` the greedy alone."""
    g = [r for r in recs if r["group"] == group]
    if condition == "per_sample": return [r for r in g if r["sample"] != "greedy"]
    if condition == "greedy": return [r for r in g if r["sample"] == "greedy"]
    raise ValueError(condition)


def _activation_hits_counts(valid, n_act, value_fn):
    """Per-activation `(hits, counts)`: the sum of `value_fn` over, and the number of, its valid pairs."""
    hits, counts = np.zeros(n_act), np.zeros(n_act)
    for r in valid:
        counts[r["i"]] += 1
        hits[r["i"]] += value_fn(r)
    return hits, counts


def boot_ratio(hits, counts, idx):
    """Ratio-of-sums activation bootstrap of `sum(hits) / sum(counts)` (nan for an all-zero resample)."""
    hits = np.asarray(hits, dtype=float); counts = np.asarray(counts, dtype=float)
    tot_c = float(counts.sum())
    if tot_c <= 0:
        return (float("nan"),) * 3
    est = float(hits.sum() / tot_c)
    sum_h, sum_c = hits[idx].sum(axis=1), counts[idx].sum(axis=1)
    samples = np.where(sum_c > 0, sum_h / np.maximum(sum_c, 1), np.nan)
    lo, hi = float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))
    return est, lo, hi


def summarise(recs, n_act, idx, clustered, n_skipped=0):
    """One cell's pooled rates over valid pairs (methodology §6). `n_total` includes the `n_skipped` pairs.
    `clustered` (several pairs per activation) bootstraps the shares over activations, else a Wilson
    interval; the nets always use the bootstrap."""
    valid = [r for r in recs if r["outcome"] is not None]
    answered = [r for r in valid if r.get("answered")]
    n_total, n_valid = len(recs) + n_skipped, len(valid)
    out = {"n_total": n_total, "n_valid": n_valid, "n_answered": len(answered), "ci_method": {}}
    for k in ("win", "tie", "loss", "at_least_as", "refused_tie"):
        if k == "at_least_as":
            value_fn = lambda r: r["at_least_as"]
        elif k == "refused_tie":
            value_fn = lambda r: int(r["outcome"] == "tie" and r["n_refused"] > 0)
        else:
            value_fn = lambda r, k=k: int(r["outcome"] == k)
        hits, counts = _activation_hits_counts(valid, n_act, value_fn)
        if clustered:
            out[k] = boot_ratio(hits, counts, idx)
            out["ci_method"][k] = CI_BOOT_ACT
        else:
            out[k] = wilson(int(hits.sum()), n_valid)
            out["ci_method"][k] = CI_WILSON
    net_hits, net_counts = _activation_hits_counts(valid, n_act, lambda r: r["net"])
    out["net"] = boot_ratio(net_hits, net_counts, idx)
    out["ci_method"]["net"] = CI_BOOT_ACT
    drop_hits, drop_counts = _activation_hits_counts(answered, n_act, lambda r: r["net"])
    out["net_dropped"] = boot_ratio(drop_hits, drop_counts, idx)
    out["ci_method"]["net_dropped"] = CI_BOOT_ACT
    out["n_refused"] = sum(r["n_refused"] for r in valid)
    out["n_parse_fail"] = sum(r["n_parse_fail"] for r in valid)
    return out


def check_parse_share(recs, max_share=C.MAX_UNPARSEABLE_SHARE):
    by = {}
    for r in recs: by.setdefault(r["group"], []).append(r)
    for g, rs in by.items():
        calls = 2 * len(rs); bad = sum(r["n_parse_fail"] for r in rs)
        if calls and bad / calls > max_share: raise ValueError(f"group {g}: {bad}/{calls} unparseable replies, above {max_share:.0%}")


def missingness_rows(run, pairs, judges, setting):
    """`tables/missingness.csv`: per judge and group, the requests asked and what came back."""
    nan = float("nan")
    out = []
    for judge in judges:
        for group, counts in J.request_outcomes(run, judge, pairs).items():
            n = sum(counts.values())
            out.append(row("n_requests", setting, group, n, nan, nan, n, counts["ok"], "", judge=judge))
            for status in J.MISSINGNESS_STATUSES:
                out.append(row(status, setting, group, counts[status], nan, nan, n, counts["ok"], "",
                               judge=judge))
    return out


def required_artifacts():
    """`[(relative path, the stage that writes it)]`: every file `report` reads a number out of."""
    from eval.rollout_coherence import frontier_context as FX
    from eval.rollout_coherence.fluency import FLUENCY_OUT
    from eval.rollout_coherence.frontier_corpus import CORPUS_JSON

    out = [("data/documents.json", "prepare"), (CORPUS_JSON, "frontier_corpus"),
           (FX.TARGETS_NPZ, "frontier_context")]
    out += [(FX.SCORES.format(name), "frontier_context")
            for name in dict.fromkeys(FX.SCORED_AS.values())]
    out.append((FX.PAIRS, "frontier_context_pairs"))
    out += [(J.judge_log(judge, C.INSTRUMENT), "frontier_context_judge") for judge in C.JUDGES]
    out.append((FLUENCY_OUT, "frontier_context_fluency"))
    return out


def require_artifacts(run):
    """Raise, naming the stage to run, unless every artifact the report renders from is here."""
    missing = [f"{rel} (stage `{stage}`)" for rel, stage in required_artifacts() if not run.exists(rel)]
    if missing:
        raise FileNotFoundError(
            "report: this run directory is missing " + "; ".join(missing)
            + ". Every table and figure the README lists is rendered from these, so the render refuses "
              "rather than publishing a partial report; run `python -m eval.rollout_coherence all` first.")


def require_rows(tables):
    """Raise unless every table has a row (an empty one is a reduction that matched nothing)."""
    empty = sorted(name for name, rows in tables.items() if not rows)
    if empty:
        raise ValueError(
            f"report: table(s) {empty} came out with no rows. Every table the README lists is rendered "
            f"from artifacts this directory holds, so an empty one is a reduction that matched nothing "
            f"(a manifest and a score file describing different activations, say), not a smaller run.")


def analyse(run):
    """Every table, coverage block and figure input the report renders, on one resample matrix."""
    from eval.rollout_coherence import frontier_context_analysis as FCA

    require_artifacts(run)
    J.refuse_unasked_logs(run)
    tables, coverage, points, records = FCA.matched_tables(run)
    require_rows(tables)

    doc = run.read_json("data/documents.json")
    prov, launches = read_provenance(run), launch_records(run)
    coverage = {
        "activations": len(doc.get("sources") or []),
        # pool candidates the norm filter refused at selection (methodology §3), each replaced by the next
        "rejected": len(doc.get("rejected") or []),
        "judges": {name: {"model": spec.model, "label": spec.label} for name, spec in C.JUDGES.items()},
        **coverage,
        "ledger": run.read_json(C.LEDGER) if run.exists(C.LEDGER) else {},
        # without `updated_at`, which a render itself moves, so two renders of one run are byte-identical
        "provenance": {k: v for k, v in prov.items() if k != "updated_at"},
        # the GPU block (eval.common.runs.GPU_COST_KEYS), summed over every launch
        **gpu_cost_block(prov.get("gpu_type"), launch_gpu_seconds(launches), [],
                         *launch_gpu_rate(launches, prov)),
    }
    return {"tables": tables, "coverage": coverage, "points": points, "records": records}
