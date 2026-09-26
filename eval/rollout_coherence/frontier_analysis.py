"""The curve half of `report` (methodology §4, §7): each method's points on inversion (raw and centred
re-read cosine) against fluency (judged net, log-likelihood gap), with exact best-of-k over samples ranked
by raw cosine. A missing value is renormalised out of its axis, never zero-filled. Numpy only.
"""
import collections
import math

import numpy as np

from eval.common.stats import boot_mean, row
from eval.rollout_coherence import analysis as A
from eval.rollout_coherence import config as C
from eval.rollout_coherence import judge as J

METHOD_ORDER = tuple(C.PLOTTED)
RETRIEVAL_PREFIX = "retrieval@"
AXIS_MAX_CENTRED = 1.0

# The loaded artifacts a figure is read from: `nets` {judge: {pid: net}} (an unanswered order a tie;
# `nets_dropped` only fully answered pairs), `gaps`/`centred` {pid: value}, `source` the source passage's
# per-activation cosines, `pos_of` {activation index: row}, `labels` retrieval's corpus sizes.
Reading = collections.namedtuple(
    "Reading", "pairs nets gaps source pos_of n labels centred nets_dropped",
    defaults=((), {}, {}))


def number(v):
    """`float(v)`, or `nan` for None, a bool or a value `float` rejects."""
    if v is None or isinstance(v, bool):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean_finite(values):
    """The mean over the finite entries, `nan` when there are none."""
    v = np.asarray(list(values), dtype=float)
    ok = np.isfinite(v)
    return float(v[ok].mean()) if ok.any() else float("nan")


def n_valid(vec):
    """How many entries of a per-activation vector are finite."""
    return int(np.isfinite(np.asarray(vec, dtype=float)).sum())


def triple(t):
    """A `(est, lo, hi)` as a JSON-plain list, or None when the estimate does not exist."""
    est, lo, hi = t
    return None if not np.isfinite(est) else [float(est), float(lo), float(hi)]


# ---------------------------------------------------------------- the selection rule

def best_of_k_weights(m, k):
    """P(rank r of `m` samples is the one a best-of-k draw keeps) = `C(m - r, k - 1) / C(m, k)`
    (methodology §4); `k` is clipped to `m`."""
    m = int(m)
    if m <= 0:
        return np.zeros(0, dtype=float)
    kk = min(max(int(k), 1), m)
    denom = math.comb(m, kk)
    return np.array([math.comb(m - r, kk - 1) / denom for r in range(1, m + 1)], dtype=float)


def rank_samples(samples):
    """One activation's samples with a finite cosine, best first, ties to the lower sample index."""
    valid = [s for s in samples if s.get("cos") is not None and np.isfinite(float(s["cos"]))]
    return sorted(valid, key=lambda s: (-float(s["cos"]), s["order_key"]))


def weighted_mean(ranked, weights, value_fn):
    """The best-of-k weighted mean of one axis, renormalised over the samples with a value (else `nan`)."""
    vals = np.array([number(value_fn(s)) for s in ranked], dtype=float)
    ok = np.isfinite(vals)
    if not ok.any():
        return float("nan")
    w = np.asarray(weights, dtype=float)[ok]
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float((vals[ok] * w).sum() / total)


def per_activation_values(samples_by_act, n_act, k, axis_fns):
    """`(values, n)`: per axis, the activations' best-of-k values (`nan` if none), and `n` with a value."""
    out = {name: np.full(n_act, np.nan) for name in axis_fns}
    n = 0
    for i in range(n_act):
        ranked = rank_samples(samples_by_act.get(i, ()))
        if not ranked:
            continue
        n += 1
        w = best_of_k_weights(len(ranked), k)
        for name, fn in axis_fns.items():
            out[name][i] = weighted_mean(ranked, w, fn)
    return out, n


# ---------------------------------------------------------------- loading

def nets_by_pid(recs_by_judge, answered_only=False):
    """`{judge: {pid: net}}` over resolved pairs; `answered_only` needs both orders answered."""
    return {judge: {r["pid"]: r["net"] for r in recs
                    if r.get("outcome") is not None and (not answered_only or r.get("answered"))}
            for judge, recs in recs_by_judge.items()}


def judged_records(run, pairs, judges, manifest):
    """`{judge: scored records}` for one manifest; raises on an answered pid the manifest lacks."""
    known = {p["pid"] for p in pairs}
    out = {}
    for judge in judges:
        answers = J.load_answers(run, judge)
        unknown = sorted({pid for pid, _order in answers} - known)
        if unknown:
            raise ValueError(f"judge {judge}: {len(unknown)} answered pid(s) are absent from "
                             f"{manifest} (first: {unknown[0]}); the answers and the manifest are "
                             f"from different pair sets")
        out[judge] = A.score_pairs(pairs, answers)
    return out


def likelihood_gaps(run, rel, pairs):
    """`{pid: ll_gap}` from a `fluency.score_run` output, over the pairs of `pairs` (`{}` when absent)."""
    if not run.exists(rel):
        return {}
    known = {p["pid"] for p in pairs}
    return {r["pid"]: r.get("ll_gap") for r in run.read_jsonl(rel)
            if r.get("kind") == "pair" and r.get("pid") in known}


# ---------------------------------------------------------------- per-sample records and curves

def _order_key(sample):
    """The sample index the tie rule breaks on; 0 for a named one-per-activation sample (`greedy`, `top1`)."""
    try:
        return int(sample)
    except (TypeError, ValueError):
        return 0


def method_samples(reading, group, include):
    """`{row: [sample record with every axis]}` for one pair group's samples that `include` accepts."""
    by_act = {}
    for p in reading.pairs:
        if p.get("group") != group or not include(p.get("sample")):
            continue
        pos = reading.pos_of.get(p["i"])
        if pos is None:
            continue
        pid = p["pid"]
        by_act.setdefault(pos, []).append({
            "pid": pid,
            "order_key": _order_key(p.get("sample")),
            "cos": p.get("reread_cos"),
            "cos_centred": reading.centred.get(pid),
            "net": {judge: m.get(pid) for judge, m in reading.nets.items()},
            "net_dropped": {judge: m.get(pid) for judge, m in reading.nets_dropped.items()},
            "ll_gap": reading.gaps.get(pid),
            "n_tokens": p.get("n_retok"),
            "extra": p.get("extra") or {},
        })
    return by_act


def axis_fns(judges):
    """The axes of one point as accessors on a sample record; per-judge axes keyed `("net", judge)`."""
    fns = {
        "cos": lambda s: s.get("cos"),
        "cos_centred": lambda s: s.get("cos_centred"),
        "ll_gap": lambda s: s.get("ll_gap"),
        "n_tokens": lambda s: s.get("n_tokens"),
    }
    for judge in judges:
        fns[("net", judge)] = lambda s, judge=judge: (s.get("net") or {}).get(judge)
        fns[("net_dropped", judge)] = lambda s, judge=judge: (s.get("net_dropped") or {}).get(judge)
    return fns


def _entry(method, point, order, headline, vals, n, by_construction=False):
    return {"method": method, "point": point, "order": order, "headline": bool(headline),
            "reference": method in C.REFERENCES,
            "vals": vals, "n": n, "by_construction": by_construction}


def sampled_entries(method, reading, fns):
    """One method's curve: greedy, then `C.BEST_OF_K` up to the number of samples it drew."""
    greedy = method_samples(reading, method, lambda s: s == "greedy")
    samples = method_samples(reading, method, lambda s: s != "greedy")
    out = [_entry(method, "greedy", 0, False, *per_activation_values(greedy, reading.n, 1, fns))]
    # a k past the method's draws is absent, not clipped (clipping is per activation only)
    drawn = max((len(v) for v in samples.values()), default=0)
    for order, k in enumerate(C.BEST_OF_K, start=1):
        if k > drawn > 0:
            continue
        vals, n = per_activation_values(samples, reading.n, k, fns)
        out.append(_entry(method, f"k={k}", order, k == C.HEADLINE_K, vals, n))
    return out


def selection_entries(method, reading, fns):
    """One arm's `C.EXTENDED_K` points: x the exact best-of-k over all draws, y the one judged selection."""
    out = []
    for k in C.EXTENDED_K:
        group = C.selection_group(method, k)
        by_act = method_samples(reading, group, lambda s: True)
        if not by_act:
            continue
        for samples in by_act.values():
            for rec in samples:
                extra = rec.get("extra") or {}
                rec["cos"] = extra.get("cos_best_of_k")
                rec["cos_centred"] = extra.get("cos_centred_best_of_k")
        vals, n = per_activation_values(by_act, reading.n, 1, fns)
        out.append(_entry(method, f"k={k}", C.BEST_OF_K.index(k) + 1, False, vals, n))
    return out


def manifest_labels(pairs):
    """The corpus sizes a manifest holds, in first-appearance order (never sorted: "5M" < "5k")."""
    labels = []
    for p in pairs:
        g = p.get("group", "")
        if g.startswith(RETRIEVAL_PREFIX) and g[len(RETRIEVAL_PREFIX):] not in labels:
            labels.append(g[len(RETRIEVAL_PREFIX):])
    return labels


def retrieval_entries(reading, fns):
    """Retrieval's curve: one k = 1 point per corpus size, the largest one compared."""
    labels = list(reading.labels)
    out = []
    for order, label in enumerate(labels):
        by_act = method_samples(reading, f"{RETRIEVAL_PREFIX}{label}", lambda s: True)
        vals, n = per_activation_values(by_act, reading.n, 1, fns)
        out.append(_entry("retrieval", label, order, order == len(labels) - 1, vals, n))
    return out


def source_entries(reading, judges, fns):
    """The source passage as a reference point, at net and gap zero by construction."""
    if not reading.source:
        return []
    by_act = {}
    for r in reading.source:
        pos = reading.pos_of.get(r["i"])
        if pos is None:
            continue
        by_act.setdefault(pos, []).append({
            "order_key": 0, "cos": r.get("cos"), "ll_gap": 0.0,
            "cos_centred": r.get("cos_centred"),
            "net": {judge: 0.0 for judge in judges},
            "net_dropped": {judge: 0.0 for judge in judges},
            "n_tokens": r.get("n_tokens"),
        })
    vals, n = per_activation_values(by_act, reading.n, 1, fns)
    return [_entry("source", "passage", 0, True, vals, n, by_construction=True)]


# ---------------------------------------------------------------- estimates, points, rows

def _optional(value):
    return None if not np.isfinite(value) else float(value)


def estimate(entry, idx, judges):
    """One point's axes, bootstrapped on the shared resample matrix `idx`; `n_tokens` is a plain mean."""
    vals = entry["vals"]
    return {
        "cos": boot_mean(vals["cos"], idx),
        "cos_centred": boot_mean(vals["cos_centred"], idx),
        "net": {judge: boot_mean(vals[("net", judge)], idx) for judge in judges},
        "net_dropped": {judge: boot_mean(vals[("net_dropped", judge)], idx) for judge in judges},
        "ll_gap": boot_mean(vals["ll_gap"], idx),
        "n_tokens": mean_finite(vals["n_tokens"]),
    }


def _point(entry, stat, judges):
    return {
        "method": entry["method"], "point": entry["point"], "order": entry["order"],
        "headline": entry["headline"], "reference": entry["reference"], "n": entry["n"],
        "cos": triple(stat["cos"]),
        "cos_centred": triple(stat["cos_centred"]),
        "net": {judge: triple(stat["net"][judge]) for judge in judges},
        "net_dropped": {judge: triple(stat["net_dropped"][judge]) for judge in judges},
        "ll_gap": triple(stat["ll_gap"]),
        "n_tokens": _optional(stat["n_tokens"]),
    }


def frontier_rows(entries, points, judges, n_act):
    """`tables/frontier_matched.csv`: one long-format row per (point, metric), the curve's shape in the
    `order`/`headline`/`reference` columns; judge-free rows carry `judge="none"`."""
    nan = float("nan")
    out = []
    for entry, point in zip(entries, points):
        vals, stat = entry["vals"], entry["stat"]
        extra = {"order": point["order"], "headline": point["headline"],
                 "reference": point["reference"],
                 "n_activations": point["n"]}
        ci = "" if entry["by_construction"] else A.CI_BOOT_ACT
        out.append(row("reread_cos", point["point"], point["method"], *stat["cos"], n_act,
                       n_valid(vals["cos"]), A.CI_BOOT_ACT, judge="none", **extra))
        out.append(row("reread_cos_centred", point["point"], point["method"], *stat["cos_centred"], n_act,
                       n_valid(vals["cos_centred"]), A.CI_BOOT_ACT, judge="none", **extra))
        for judge in judges:
            out.append(row("net", point["point"], point["method"], *stat["net"][judge], n_act,
                           n_valid(vals[("net", judge)]), ci, judge=judge, **extra))
            out.append(row("net_dropped", point["point"], point["method"], *stat["net_dropped"][judge],
                           n_act, n_valid(vals[("net_dropped", judge)]), ci, judge=judge, **extra))
        out.append(row("ll_gap", point["point"], point["method"], *stat["ll_gap"], n_act,
                       n_valid(vals["ll_gap"]), ci, judge="none", **extra))
        out.append(row("n_tokens", point["point"], point["method"], stat["n_tokens"], nan, nan, n_act,
                       n_valid(vals["n_tokens"]), "", judge="none", **extra))
    return out


def setting_points(reading, judges, idx):
    """`(entries, points)`: every `C.PLOTTED` method with pairs in this reading, along its curve."""
    fns = axis_fns(judges)
    entries = []
    for method, spec in C.PLOTTED.items():
        if spec["curve"] == "single":
            entries += source_entries(reading, judges, fns)
        elif spec["curve"] == "corpus_size":
            entries += retrieval_entries(reading, fns)
        elif any(p.get("group") == method for p in reading.pairs):
            entries += sampled_entries(method, reading, fns)
            if method in C.EXTENDED_ARMS:
                entries += selection_entries(method, reading, fns)
    entries.sort(key=lambda e: (METHOD_ORDER.index(e["method"]), e["order"]))

    points = []
    for entry in entries:
        entry["stat"] = estimate(entry, idx, judges)
        points.append(_point(entry, entry["stat"], judges))
    return entries, points


# ---------------------------------------------------------------- win / tie / loss per method

OUTCOME_METRICS = ("at_least_as", "win", "tie", "loss", "net", "net_dropped", "refused_tie")


def positioned(recs, pos_of):
    """The records of the activations `pos_of` covers, `i` renumbered to the reading's rows."""
    out = []
    for r in recs:
        p = pos_of.get(r.get("i"))
        if p is not None:
            out.append({**r, "i": p})
    return out


def outcome_cells(recs, labels):
    """`[(method, condition, group, selector, clustered)]`: pooled draws and greedy per sampled method,
    retrieval at the largest size."""
    groups = {r.get("group") for r in recs}
    out = []
    for method, spec in C.PLOTTED.items():
        if spec["curve"] == "corpus_size":
            label = list(labels)[-1] if labels else None
            if label is not None and f"{RETRIEVAL_PREFIX}{label}" in groups:
                out.append((method, label, f"{RETRIEVAL_PREFIX}{label}", "per_sample", False))
        elif spec["curve"] != "single" and method in groups:
            out.append((method, "per_sample", method, "per_sample", True))
            out.append((method, "greedy", method, "greedy", False))
    return out


def outcome_rows(population, recs_by_judge, n, idx, labels):
    """`tables/frontier_outcomes.csv` rows: win / tie / loss per judged method and judge."""
    nan = float("nan")
    out = []
    for judge, recs in recs_by_judge.items():
        for method, condition, group, selector, clustered in outcome_cells(recs, labels):
            S = A.summarise(A.condition_pairs(recs, group, selector), n, idx, clustered=clustered)
            extra = {"judge": judge, "population": population, "n_activations": n,
                     "n_refused": S["n_refused"],
                     "refused_share": (S["n_refused"] / (2 * S["n_total"])) if S["n_total"] else nan,
                     "reference": method in C.REFERENCES}
            for metric in OUTCOME_METRICS:
                nv = S["n_answered"] if metric == "net_dropped" else S["n_valid"]
                out.append(row(metric, condition, method, *S[metric], S["n_total"], nv,
                               S["ci_method"][metric], **extra))
    return out


# ---------------------------------------------------------------- coverage

def skipped_by_group(config):
    """The manifest's skipped pairs counted per group."""
    out = {}
    for sk in (config.get("skipped") or []):
        group = sk.get("group") or sk["pid"].split("/", 1)[0]
        out[group] = out.get(group, 0) + 1
    return out


def pairs_coverage(recs_by_judge, groups, skipped_by_group):
    """Per judge and group: planned, valid, refused and unparseable pairs, and the refused share of orders."""
    out = {}
    for judge, recs in recs_by_judge.items():
        per = {}
        for group in groups:
            sub = [r for r in recs if r.get("group") == group]
            valid = [r for r in sub if r["outcome"] is not None]
            counts = {}
            for r in sub:
                for status in r["status"]:
                    if status is None:
                        continue
                    counts[status] = counts.get(status, 0) + 1
            skipped = int(skipped_by_group.get(group, 0))
            planned = len(sub) + skipped
            refused = sum(r["n_refused"] for r in valid)
            per[group] = {"planned": planned, "valid": len(valid), "skipped": skipped,
                          "refused": refused,
                          "parse_fail": sum(r["n_parse_fail"] for r in valid),
                          "refused_share": (refused / (2 * planned)) if planned else float("nan"),
                          "status_counts": counts}
        out[judge] = per
    return out
