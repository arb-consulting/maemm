"""Per-item scores, the long-format tables and the coverage-and-cost record (methodology §6, §7).

Statistics and the table writer are eval/common/stats.py's. A row that reports a judge's verdict carries a
`judge` column; a word-rule or re-read row leaves it empty."""

import glob, json, os
import numpy as np
from eval.common.nla.nla_reader import PINS as NLA
from eval.workspace_understanding import config as C
from eval.common import matcher as M
from eval.workspace_understanding import readouts as RO
from eval.workspace_understanding import shards as S
from eval.workspace_understanding.lens import pool_band
from eval.common.runs import StaleVerdict, read_provenance
from eval.common import judges as _judges
from eval.common.judge_client import judge_log, request_key, unasked, with_judge
from eval.common.runs import elapsed_block, gpu_cost_block, launch_gpu_rate, launch_records
from eval.common.stats import (
    CI_BOOT,
    CI_WILSON,
    boot_indices,
    boot_mean,
    finite_n,
    paired_diff,
    row,
    wilson,
    write_tables,
)

JUDGE_ORDER = tuple(_judges.JUDGE_ORDER)

GROUPS = ("association", "multihop", "pooled", "association/proper_noun", "association/common")
# The readers MAEMM is contrasted against on the word rule (the lens has its own rank contrasts).
READER_COMPARATORS = C.PATCH_ARMS + ("nla", "nla64", "retrieval", "nla_mid", "nla_mean",
                                     "maemm_mid", "maemm_mean", "untrained_base")
# Readers contrasted with their own chance line.
CHANCE_CHECKED = ("nla", "nla64", "retrieval", "nla_mid", "nla_mean", "maemm_mid", "maemm_mean",
                  "untrained_base")


def _in_group(rec, g):
    if g == "pooled":
        return True
    fam, _, sub = g.partition("/")
    return rec["family"] == fam and (not sub or rec.get("subgroup") == sub)


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _hits(greedy_text, sample_texts, forms):
    """The word rule on one reader's readout of one item: greedy hit, per-sample hits, pass@N and
    consistency. `greedy_hit` is None for a reader with no greedy rollout (the corpus search)."""
    sh = M.sample_hits(sample_texts, forms)
    return {
        "greedy_hit": None if greedy_text is None else M.whole_word_hit(greedy_text, forms),
        "sample_hits": sh,
        "pass_at": {str(n): M.pass_at_n(sh, n) for n in C.PASS_AT_N},
        "consistency": M.consistency(sh),
    }


def parse_or_none(r):
    """One saved judge record re-read with the current parser (a run-time parse_fail may now read)."""
    from eval.workspace_understanding import judge as J

    if r["status"] not in ("ok", "parse_fail"):
        return {"named": None, "voided": None, "status": r["status"], "quote": None, "repaired": False}
    v = J.parse_verdict(r.get("text") or "")
    if not v["parse_ok"]:
        return {"named": None, "voided": None, "status": "parse_fail", "quote": None, "repaired": False}
    reason = J.void_reason(v, r.get("targets") or [], r.get("samples") or [])
    return {
        "named": bool(v["expressed"]) and reason is None,
        "voided": reason,
        "status": "ok",
        "quote": v.get("quote"),
        "repaired": bool(v.get("repaired")),
    }


def judge_unavailable(s, judge, cond):
    """A judged cell whose verdict resolved to neither named nor not-named (missing, non-ok, unparsed)."""
    return s["judged"][judge][cond]["named"] is None


def _meta_id(meta):
    return json.dumps(meta, sort_keys=True, ensure_ascii=False)


def bound_records(run, judge):
    """{cell identity: (meta, record)}: one judge's naming verdicts, each joined by request key to the
    readout as it stands now (`judge.build_requests`). A cell whose logged records answer only other keys
    raises StaleVerdict: its readout changed after it was judged."""
    from eval.workspace_understanding import judge as J

    naming, skipped = J.build_requests(run)
    reqs = naming + skipped
    spec = C.JUDGES[judge]
    log = judge_log(judge, "naming")
    logged = {}
    for r in run.read_jsonl(log):
        logged.setdefault(_meta_id(r.get("meta")), []).append(r)
    out, stale = {}, []
    for req in reqs:
        cell = _meta_id(req["meta"])
        recs = logged.get(cell)
        if not recs:
            continue
        key = request_key(with_judge(req, spec))
        mine = [r for r in recs if r.get("key") == key]
        if mine:
            out[cell] = (req["meta"], mine[-1])
        else:
            stale.append((req["meta"], key, recs[-1].get("key")))
    if stale:
        meta, key, was = stale[0]
        raise StaleVerdict(
            f"judge `{judge}`: the naming verdict logged for cell {_meta_id(meta)} answers request "
            f"{was}, and the readout now in this directory builds request {key} -- the judged text (or a "
            f"setting of the request) changed after the verdict was given ({len(stale)} such cell(s) in "
            f"{log}). Run the `judge` stage again: it asks only what its logs do "
            f"not already answer."
        )
    return out


def load_verdicts(run, judge):
    """(i, condition, vs) -> parse_or_none() over one judge's bound naming verdicts."""
    return {(m["i"], m["condition"], m["vs"]): parse_or_none(r)
            for m, r in bound_records(run, judge).values()}


def _judged_block(jud, i):
    """One item's naming record under one judge, per judged condition."""
    out = {}
    for cond in C.JUDGED_CONDITIONS:
        own, foil = jud.get((i, cond, "own")), jud.get((i, cond, "foil"))
        out[cond] = {
            "named": own["named"] if own else None,
            "foil": foil["named"] if foil else None,
            "voided_reason": own["voided"] if own else None,
            "status": own["status"] if own else "missing",
            "quote": own["quote"] if own else None,
        }
    return out


def _donor_chance(view, donors, forms):
    """(rate, donors used): the 20-donor target-shuffle chance line, how often this item's targets occur in
    other items' readouts (methodology §6.1). Donors with no readout are skipped."""
    used = [view[d] for d in donors if view.get(d) is not None]
    if not used:
        return float("nan"), 0
    hit = lambda ss, f: any(M.whole_word_hit(s, f) for s in ss)
    return M.chance_over_donors(forms, [r["samples"] for r in used], hit), len(used)


def score_items(run):
    """One record per kept item: every reader's word-rule result and chance line, the lens's ranks and
    word rules, the judged verdicts and the re-read cosines. A missing readout is None, never a zero."""
    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    diag = run.read_json("data/diagnostic.json") if run.exists("data/diagnostic.json") else {}
    lens_doc = run.read_json("lens/lens.json")
    lens = {r["i"]: r for r in lens_doc["items"]}
    fitted_layers = lens_doc["lens"]["fitted_layers"]
    free = RO.free_text(run)
    verdicts = {j: load_verdicts(run, j) for j in JUDGE_ORDER}
    maemm_raw = {r["i"]: r for r in run.read_json("rollouts/maemm.json")["items"]}
    band = {i: pool_band(r["top10_by_layer"], fitted_layers) for i, r in lens.items()}
    reread = run.read_json("rollouts/reread.json") if run.exists("rollouts/reread.json") else None
    nla_doc = S.load_nla(run)
    nla_raw = {r["i"]: r for r in (nla_doc or {}).get("items", [])}
    out = []
    for it in kept:
        i, forms, donors = it["i"], it["forms"], it["donors"]
        rec = {
            "i": i,
            "family": it["family"],
            "subgroup": it.get("subgroup"),
            "name": it["name"],
            "forms": forms,
            "multi_token": it["multi_token"],
            "diag_correct": diag.get(str(i), {}).get("correct") if it["family"] == "multihop" else None,
            "chance": {"n_donors_used": {}},
        }
        for cond in C.FREE_TEXT_CONDITIONS:
            view = free.get(cond) or {}
            r = view.get(i)
            rec[cond] = _hits(r["greedy"], r["samples"], forms) if r else None
            ch, nd = _donor_chance(view, donors, forms)
            rec["chance"][f"{cond}_pass8"] = ch
            rec["chance"]["n_donors_used"][cond] = nd
        if rec["maemm"] is not None:
            m = maemm_raw.get(i)
            rec["maemm"]["cos_own_greedy"] = (m["greedy"].get("cos_own") if m else None)
            rec["maemm"]["answer_hit"] = (
                any(M.whole_word_hit(t, it["answer_forms"]) for t in (free["maemm"][i]["samples"]))
                if it["family"] == "multihop"
                else None
            )
        L = lens.get(i)
        # no lens record at all: a miss at every cutoff (`has_record` False)
        rec["jlens"] = {
            "rank_L42": L["rank_L42"] if L else None,
            "min_rank": L["min_rank"] if L else None,
            "min_rank_layer": L["min_rank_layer"] if L else None,
            "rank_by_layer": L["rank_by_layer"] if L else [None] * len(fitted_layers),
            "fitted_layers": fitted_layers,
            "has_record": L is not None,
            "rank_le": {
                "L42": {str(k): bool(L and L["rank_L42"] is not None and L["rank_L42"] <= k) for k in C.RANK_KS},
                "best": {str(k): bool(L and L["min_rank"] is not None and L["min_rank"] <= k) for k in C.RANK_KS},
            },
            # the lens word rules (methodology §6.1): a target form as a whole word among the top-10
            # word-like tokens at layer 42, and among the pooled lists of layers 36, 38, ..., 50
            "word_top10_L42": bool(L and any(M.whole_word_hit(t, forms) for t in (L.get("top10_L42") or []))),
            "word_top10_band8": bool(L and any(M.whole_word_hit(t, forms) for t in band.get(i, []))),
        }
        rec["chance"]["jlens_band8_word"], rec["chance"]["n_donors_used"]["jlens_band8"] = _donor_chance(
            {d: {"samples": band[d]} for d in donors if d in band}, donors, forms
        )
        if L is not None:
            rec["chance"]["jlens_L42_k10"] = _mean([r is not None and r <= 10 for r in L["donor_rank_L42"]])
            rec["chance"]["jlens_best_k10"] = _mean([r is not None and r <= 10 for r in L["donor_min_rank"]])
        else:
            rec["chance"]["jlens_L42_k10"] = float("nan")
            rec["chance"]["jlens_best_k10"] = float("nan")
        rec["judged"] = {j: _judged_block(verdicts[j], i) for j in JUDGE_ORDER}
        n = nla_raw.get(i)
        rec["nla_diag"] = (
            {
                "close_rate": float(np.mean([x["closed"] for x in n["samples"]])),
                "close_rate_trunc": float(np.mean([x["closed_trunc"] for x in n["samples"]])),
                "mean_generated": float(np.mean([x["n_tokens"] for x in n["samples"]])),
            }
            if n
            else None
        )
        rec["fidelity"] = {}
        m = maemm_raw.get(i)
        if m and reread:
            rec["fidelity"]["maemm"] = {
                "own_samples": float(np.mean([x["cos_own"] for x in m["samples"]])),
                "foil_samples": float(np.mean([x["cos_foil"] for x in m["samples"]])),
                "own_greedy": float(m["greedy"]["cos_own"]),
                "foil_greedy": float(m["greedy"]["cos_foil"]),
            }
        for cond in C.REREAD_CONDITIONS:
            block = ((reread or {}).get("conditions") or {}).get(cond)
            r = {x["i"]: x for x in block["items"]}.get(i) if block else None
            if r:
                rec["fidelity"][cond] = {
                    "own_samples": float(np.mean([x["cos_own"] for x in r["samples"]])),
                    "foil_samples": float(np.mean([x["cos_foil"] for x in r["samples"]])),
                    "own_greedy": float(r["greedy"]["cos_own"]),
                    "foil_greedy": float(r["greedy"]["cos_foil"]),
                }
        out.append(rec)
    return out


# --- the long-format tables -------------------------------------------------------------------------------


def _rate_rows(metric, cond, group, values, **extra):
    """One Wilson rate row over `values`; None leaves numerator and denominator alike."""
    vals = [v for v in values if v is not None]
    k = sum(1 for v in vals if v)
    est, lo, hi = wilson(k, len(vals))
    return row(metric, cond, group, est, lo, hi, len(values), len(vals), ci_method=CI_WILSON, **extra)


def _net(s, judge, cond):
    j = s["judged"][judge][cond]
    return None if (j["named"] is None or j["foil"] is None) else float(j["named"]) - float(j["foil"])


def _rates_for_group(S_, g, idx, include_answer_hit):
    """Every rates.csv row for one group: per free-text reader greedy hit, pass@N, consistency and chance;
    the lens's rank curves (layer 42 and best layer) and word rules."""
    out = []
    for cond in C.FREE_TEXT_CONDITIONS:
        if not any(s.get(cond) for s in S_):
            continue
        # no greedy row at all for a reader without a greedy rollout
        greedy = [s[cond]["greedy_hit"] if s.get(cond) else None for s in S_]
        if any(v is not None for v in greedy):
            out.append(_rate_rows("greedy_hit", cond, g, greedy, budget_type="samples", budget=0))
        # the chance line is carried on the pass@8 row as well as on its own row
        ch_vec = [s["chance"].get(f"{cond}_pass8") for s in S_]
        chance_est, chance_lo, chance_hi = boot_mean(ch_vec, idx)
        p8 = _mean([s[cond]["pass_at"]["8"] for s in S_ if s.get(cond)])
        ratio = p8 / chance_est if chance_est else ""
        flag = bool(chance_est) and p8 < C.CHANCE_FLAG_RATIO * chance_est
        for n in C.PASS_AT_N:
            extra = {"budget_type": "samples", "budget": n}
            if n == 8:
                extra.update(chance=chance_est, ratio_vs_chance=ratio, flag_below_3x=flag)
            out.append(
                _rate_rows(
                    "pass_at_n", cond, g, [s[cond]["pass_at"][str(n)] if s.get(cond) else None for s in S_], **extra
                )
            )
        # NaN at a missing readout keeps the vector aligned with the group's shared bootstrap indices
        cons_vec = [s[cond]["consistency"] if s.get(cond) else float("nan") for s in S_]
        cons_est, cons_lo, cons_hi = boot_mean(cons_vec, idx)
        out.append(
            row(
                "consistency", cond, g, cons_est, cons_lo, cons_hi, len(S_), finite_n(cons_vec),
                ci_method=CI_BOOT, budget_type="samples", budget=8,
            )
        )
        out.append(
            row(
                "chance", cond, g, chance_est, chance_lo, chance_hi, len(S_), finite_n(ch_vec),
                ci_method=CI_BOOT, budget_type="samples", budget=8,
                ratio_vs_chance=ratio, flag_below_3x=flag,
            )
        )
    if include_answer_hit and any(s["family"] == "multihop" for s in S_):
        out.append(
            _rate_rows(
                "answer_hit", "maemm", g,
                [s["maemm"]["answer_hit"] if s.get("maemm") else None for s in S_],
                budget_type="samples", budget=8,
            )
        )
    for where in ("L42", "best"):
        cond = "jlens_" + where
        ch_vec = [s["chance"][f"jlens_{where}_k10"] for s in S_]
        chance_est, chance_lo, chance_hi = boot_mean(ch_vec, idx)
        r10 = _mean([s["jlens"]["rank_le"][where]["10"] for s in S_])
        ratio = r10 / chance_est if chance_est else ""
        flag = bool(chance_est) and r10 < C.CHANCE_FLAG_RATIO * chance_est
        for k in C.RANK_KS:
            extra = {"budget_type": "rank_cutoff", "budget": k, "n_multi_token": sum(1 for s in S_ if s["multi_token"])}
            if k == 10:
                extra.update(chance=chance_est, ratio_vs_chance=ratio, flag_below_3x=flag)
            out.append(_rate_rows("rank_le_k", cond, g, [s["jlens"]["rank_le"][where][str(k)] for s in S_], **extra))
        if where == "L42":
            out.append(
                _rate_rows(
                    "word_top10", cond, g, [s["jlens"].get("word_top10_L42") for s in S_],
                    budget_type="top_tokens", budget=C.TOP_WORD,
                )
            )
            out += _band_word_rows(S_, g, idx)
        out.append(
            row(
                "chance", cond, g, chance_est, chance_lo, chance_hi, len(S_), finite_n(ch_vec),
                ci_method=CI_BOOT, budget_type="rank_cutoff", budget=10,
                ratio_vs_chance=ratio, flag_below_3x=flag,
            )
        )
    return out


def _band_word_rows(S_, g, idx):
    """The `word_top10` rate of `jlens_band8` with its 20-donor chance line."""
    cond = "jlens_" + C.LENS_BAND
    vals = [s["jlens"].get("word_top10_band8") for s in S_]
    ch_vec = [s["chance"].get("jlens_band8_word", float("nan")) for s in S_]
    chance_est, chance_lo, chance_hi = boot_mean(ch_vec, idx)
    rate = _mean([v for v in vals if v is not None])
    ratio = rate / chance_est if chance_est else ""
    flag = bool(chance_est) and rate < C.CHANCE_FLAG_RATIO * chance_est
    common = {"budget_type": "top_tokens", "budget": C.TOP_WORD}
    return [
        _rate_rows("word_top10", cond, g, vals, chance=chance_est, ratio_vs_chance=ratio, flag_below_3x=flag,
                   **common),
        row("chance", cond, g, chance_est, chance_lo, chance_hi, len(S_), finite_n(ch_vec), ci_method=CI_BOOT,
            ratio_vs_chance=ratio, flag_below_3x=flag, **common),
    ]


def _judged_rows(S_, g, idx, judge):
    """judged.csv's rows for one group under one judge: named, foil, net (item bootstrap), voided and
    unavailable per condition."""
    out = []
    for cond in C.JUDGED_CONDITIONS:
        blocks = [s["judged"][judge][cond] for s in S_]
        n_attempted = sum(1 for b in blocks if b["status"] != "missing")
        named = [b["named"] for b in blocks]
        foil = [b["foil"] for b in blocks]
        common = {"judge": judge, "has_foil": True, "n_attempted": n_attempted}
        out.append(_rate_rows("named", cond, g, named, **common))
        out.append(_rate_rows("foil", cond, g, foil, **common))
        net_vec = [float("nan") if (a is None or b is None) else float(a) - float(b) for a, b in zip(named, foil)]
        est, lo, hi = boot_mean(net_vec, idx)
        out.append(row("net", cond, g, est, lo, hi, len(S_), finite_n(net_vec), ci_method=CI_BOOT, **common))
        n_voided = sum(1 for b in blocks if b["voided_reason"])
        n_unavail = sum(1 for s in S_ if judge_unavailable(s, judge, cond))
        for metric, count in (("voided", n_voided), ("unavailable", n_unavail)):
            out.append(
                row(
                    metric, cond, g, (count / len(S_) if len(S_) else float("nan")), "", "",
                    len(S_), len(S_), count=count, **common,
                )
            )
    return out


def _reread_rows(S_, g, idx):
    """reread.csv: each reader's re-read cosine against its own and the foil's direction, and the gap."""
    out = []
    for cond in ("maemm",) + tuple(C.REREAD_CONDITIONS):
        if not any(s["fidelity"].get(cond) for s in S_):
            continue
        for budget, ok, fk in (("samples", "own_samples", "foil_samples"), ("greedy", "own_greedy", "foil_greedy")):
            own = [s["fidelity"][cond][ok] if s["fidelity"].get(cond) else float("nan") for s in S_]
            foil = [s["fidelity"][cond][fk] if s["fidelity"].get(cond) else float("nan") for s in S_]
            gap = [a - b for a, b in zip(own, foil)]
            for metric, vec in (("reread_cos_own", own), ("reread_cos_foil", foil), ("reread_gap", gap)):
                e, lo, hi = boot_mean(vec, idx)
                out.append(
                    row(
                        metric, cond, g, e, lo, hi, len(S_), finite_n(vec), ci_method=CI_BOOT,
                        budget_type=budget, budget=8 if budget == "samples" else 0,
                    )
                )
    return out


def _word_rule_contrasts():
    """(name_a, name_b, f_a, f_b) for every judge-free contrast: MAEMM against chance, the lens and every
    reader; readers against chance and their position controls; Patchscopes arms against the floor; re-read
    cosines."""
    p8 = lambda cond: (lambda s: s[cond]["pass_at"]["8"] if s.get(cond) else None)
    ch = lambda key: (lambda s: s["chance"].get(key))
    rank = lambda where, k: (lambda s: s["jlens"]["rank_le"][where][k])
    fid = lambda cond, key: (lambda s: s["fidelity"][cond][key] if s["fidelity"].get(cond) else None)
    gap = lambda cond: (
        lambda s: (s["fidelity"][cond]["own_samples"] - s["fidelity"][cond]["foil_samples"])
        if s["fidelity"].get(cond)
        else None
    )
    pairs = [
        ("maemm_pass8", "maemm_chance", p8("maemm"), ch("maemm_pass8")),
        ("maemm_pass8", "jlens_L42_k10", p8("maemm"), rank("L42", "10")),
        ("maemm_pass8", "jlens_best_k10", p8("maemm"), rank("best", "10")),
        ("maemm_greedy", "jlens_L42_k1", (lambda s: s["maemm"]["greedy_hit"] if s.get("maemm") else None), rank("L42", "1")),
    ]
    pairs += [("maemm_pass8", f"{c}_pass8", p8("maemm"), p8(c)) for c in READER_COMPARATORS]
    pairs += [(f"{c}_pass8", f"{c}_chance", p8(c), ch(f"{c}_pass8")) for c in CHANCE_CHECKED]
    pairs += [("nla_pass8", f"nla_{k}_pass8", p8("nla"), p8(f"nla_{k}")) for k in C.POSITION_CONTROLS]
    # each Patchscopes arm against its no-injection floor: what the patch added
    pairs += [(f"{a}_pass8", f"{C.PATCH_FLOOR}_pass8", p8(a), p8(C.PATCH_FLOOR)) for a in C.PATCH_LAYERS]
    pairs += [
        (f"reread_own_{a}", f"reread_own_{C.PATCH_FLOOR}", fid(a, "own_samples"), fid(C.PATCH_FLOOR, "own_samples"))
        for a in C.PATCH_LAYERS
    ]
    for cond in C.REREAD_CONDITIONS:
        pairs.append(("reread_own_maemm", f"reread_own_{cond}", fid("maemm", "own_samples"), fid(cond, "own_samples")))
        pairs.append(("reread_gap_maemm", f"reread_gap_{cond}", gap("maemm"), gap(cond)))
    return pairs


def _judged_contrasts(judge):
    """(name_a, name_b, f_a, f_b) for the judged contrasts under one judge: MAEMM's net against every other
    judged condition's."""
    net = lambda cond: (lambda s: _net(s, judge, cond))
    return [("net_maemm_n8", f"net_{c}", net("maemm_n8"), net(c)) for c in C.JUDGED_CONDITIONS if c != "maemm_n8"]


def _contrast_rows(S_, g, idx, pairs, judge=""):
    V = lambda f: np.array([f(s) if f(s) is not None else np.nan for s in S_], dtype=float)
    out = []
    for a_name, b_name, fa, fb in pairs:
        est, lo, hi, n = paired_diff(V(fa), V(fb), idx)
        if n == 0:
            continue
        out.append(
            row(
                "paired_difference", f"{a_name}-{b_name}", g, est, lo, hi, len(S_), n, ci_method=CI_BOOT,
                condition_a=a_name, condition_b=b_name, n_pairs=n, judge=judge,
            )
        )
    return out


# --- missingness ------------------------------------------------------------------------------------------

# The statuses a request settles into, in print order. A request never asked (cap, transport) fails the
# judge stage instead.
MISSINGNESS_STATUSES = ("ok", "refused", "content_filter", "parse_fail", "unavailable")


def _log_cells(run, rel):
    """The records of one request log, one per (request key, meta): the log read as cells, the last record
    of a re-asked cell winning."""
    if not run.exists(rel):
        return []
    out = {}
    for r in run.read_jsonl(rel):
        out[(r.get("key"), json.dumps(r.get("meta"), sort_keys=True, ensure_ascii=False))] = r
    return list(out.values())


def _resolved_status(rec, parse):
    """The status a record resolves to at table time: `ok`/`parse_fail` re-read with the current parser."""
    if rec["status"] not in ("ok", "parse_fail"):
        return rec["status"]
    return "ok" if parse(rec.get("text") or "") is not None else "parse_fail"


def _missingness_cells(run, judge):
    """(instrument, condition, status, voided) for every bound request `judge` was sent, over its logs."""
    from eval.workspace_understanding import judge as J

    verdict = lambda t: (lambda v: v if v["parse_ok"] else None)(J.parse_verdict(t))
    for m, rec in bound_records(run, judge).values():
        status = _resolved_status(rec, verdict)
        v = J.parse_verdict(rec.get("text") or "") if status == "ok" else None
        voided = bool(
            v and v["expressed"] and J.void_reason(v, rec.get("targets") or [], rec.get("samples") or [])
        )
        yield "naming", m["condition"], status, voided
    for rec in _log_cells(run, judge_log(judge, "diagnostics")):
        m = rec.get("meta") or {}
        yield "diagnostic", m.get("part", "diagnostic"), _resolved_status(rec, verdict), False


def judge_logs():
    """Every request log this package writes."""
    rels = [judge_log(j, i) for j in JUDGE_ORDER for i in ("naming", "diagnostics")]
    return rels + [judge_log(C.SUMMARISER, i) for i in ("summaries", "diagnostic_summaries")]


def unasked_cells(run):
    """{log: {"budget": n, "transport": n, "auth": n}}: cells whose last record is a request never asked;
    the report refuses to render over any."""
    out = {}
    for rel in judge_logs():
        counts = unasked(_log_cells(run, rel))
        if sum(counts.values()):
            out[rel] = counts
    return out


def missingness_rows(run):
    """missingness.csv: per judge, instrument and condition, how each request settled; the summariser's
    calls under instruments of their own."""
    rows = []
    for judge in JUDGE_ORDER:
        per = {}
        for instrument, cond, status, voided in _missingness_cells(run, judge):
            cell = per.setdefault((instrument, cond), dict.fromkeys(MISSINGNESS_STATUSES, 0) | {"voided": 0})
            cell[status] = cell.get(status, 0) + 1
            cell["voided"] += int(voided)
        for (instrument, cond), cell in per.items():
            rows.append(
                {
                    "condition": cond,
                    "judge": judge,
                    "instrument": instrument,
                    "n": sum(cell[s] for s in MISSINGNESS_STATUSES),
                    **{s: cell[s] for s in MISSINGNESS_STATUSES},
                    "voided": cell["voided"],
                }
            )
    summariser = C.SUMMARISER
    for instrument, log in (("summary", "summaries"), ("diagnostic_summary", "diagnostic_summaries")):
        per = {}
        for rec in _log_cells(run, judge_log(summariser, log)):
            which = (rec.get("meta") or {}).get("which", instrument)
            cell = per.setdefault(which, dict.fromkeys(MISSINGNESS_STATUSES, 0))
            cell[rec["status"]] = cell.get(rec["status"], 0) + 1
        for which, cell in per.items():
            rows.append(
                {
                    "condition": which,
                    "judge": summariser,
                    "instrument": instrument,
                    "n": sum(cell[s] for s in MISSINGNESS_STATUSES),
                    **{s: cell[s] for s in MISSINGNESS_STATUSES},
                    "voided": "",
                }
            )
    return rows


def tables(run, scores, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T = {
        "rates": [], "judged": [], "contrasts": [], "reread": [], "layer_curve": [], "diagnostic_split": [],
        "missingness": [],
    }
    judges = list(JUDGE_ORDER)
    T["missingness"] = missingness_rows(run)
    layers = scores[0]["jlens"]["fitted_layers"] if scores else []
    word_pairs = _word_rule_contrasts()
    for g in GROUPS:
        S_ = [s for s in scores if _in_group(s, g)]
        if not S_:
            continue
        idx = boot_indices(len(S_), C.N_BOOT, C.BOOTSTRAP_SEED)
        T["rates"].extend(_rates_for_group(S_, g, idx, include_answer_hit=(g == "multihop")))
        T["reread"].extend(_reread_rows(S_, g, idx))
        T["contrasts"].extend(_contrast_rows(S_, g, idx, word_pairs))
        for judge in judges:
            T["judged"].extend(_judged_rows(S_, g, idx, judge))
            T["contrasts"].extend(_contrast_rows(S_, g, idx, _judged_contrasts(judge), judge=judge))
        if g in ("association", "multihop", "pooled"):
            for li, layer in enumerate(layers):
                vals = [
                    s["jlens"]["rank_by_layer"][li] is not None and s["jlens"]["rank_by_layer"][li] <= 10
                    for s in S_
                    if s["jlens"]["rank_by_layer"][li] is not None
                ]
                est, lo, hi = wilson(sum(vals), len(vals))
                T["layer_curve"].append(
                    row(
                        "rank_le_k", "jlens", g, est, lo, hi, len(S_), len(vals), ci_method=CI_WILSON,
                        budget_type="rank_cutoff", budget=10, layer=layer,
                    )
                )
    S_mh = [s for s in scores if s["family"] == "multihop"]
    for label, keep in (("multihop/greedy_correct", True), ("multihop/greedy_incorrect", False)):
        SS = [s for s in S_mh if s["diag_correct"] is keep]
        if not SS:
            continue
        idx_ss = boot_indices(len(SS), C.N_BOOT, C.BOOTSTRAP_SEED)
        for r in _rates_for_group(SS, label, idx_ss, include_answer_hit=True):
            r["n_split"] = len(SS)
            T["diagnostic_split"].append(r)
    for cond, f in (
        ("maemm", lambda s: s["maemm"]["pass_at"]["8"] if s.get("maemm") else None),
        ("jlens_L42", lambda s: s["jlens"]["rank_le"]["L42"]["10"]),
        ("jlens_best", lambda s: s["jlens"]["rank_le"]["best"]["10"]),
    ):
        a = [float(f(s)) for s in S_mh if s["diag_correct"] is True and f(s) is not None]
        b = [float(f(s)) for s in S_mh if s["diag_correct"] is False and f(s) is not None]
        est = (np.mean(a) - np.mean(b)) if a and b else float("nan")
        rng = np.random.default_rng(C.BOOTSTRAP_SEED)
        boots = (
            [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for _ in range(C.N_BOOT)] if a and b else []
        )
        lo, hi = (
            (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"),) * 2
        )
        T["diagnostic_split"].append(
            row(
                "pass8_or_k10", cond, "multihop/difference", est, lo, hi, len(S_mh), len(a) + len(b),
                ci_method=CI_BOOT, n_split=f"{len(a)}/{len(b)}",
            )
        )
    write_tables(T, out_dir)
    return T


# --- coverage and costs -----------------------------------------------------------------------------------


def _spend_from_logs(run, rels, model=None):
    """Requests, tokens and spend over request logs: `spend_usd` reprices each request key once at
    config.RATES_PER_M; `spend_usd_recorded` sums the run-time cost_usd of the same records."""
    last, n_records = {}, 0
    for rel in rels:
        for r in run.read_jsonl(rel):
            n_records += 1
            last[r["key"]] = r
    tin = tout = treason = 0
    spend = recorded = reported = 0.0
    for r in last.values():
        recorded += float(r.get("cost_usd") or 0.0)
        u = r.get("usage") or {}
        i_tok, o_tok = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        tin += i_tok
        tout += o_tok
        treason += int(u.get("reasoning_tokens") or 0)
        reported += float(u.get("cost_reported") or 0.0)
        rin, rout = C.RATES_PER_M.get(r.get("model") or model, (0.0, 0.0))
        spend += i_tok * rin / 1e6 + o_tok * rout / 1e6
    return {
        "requests": len(last),
        "records": n_records,
        "input_tokens": tin,
        "output_tokens": tout,
        "reasoning_tokens": treason,
        "spend_usd": spend,
        "spend_usd_recorded": recorded,
        "spend_usd_provider_reported": reported,
    }


# The GPU stages; a run's GPU time is summed from their stage records.
GPU_STAGES = ("capture", "lens", "rollouts", "retrieval", "patchscope", "nla", "nla_control",
              "untrained_base", "maemm_control", "reread")
# Record-name prefixes of a GPU stage beyond `<stage>` and `<stage>_shard<k>of<n>`: the corpus parts.
GPU_RECORD_PREFIXES = {"retrieval": ("retrieval_part",)}


def _record_seconds(rec):
    """A stage record's measured seconds, or the sum of its per-arm seconds, or None."""
    if rec.get("seconds") is not None:
        return float(rec["seconds"])
    arms = [v["seconds"] for v in rec.values() if isinstance(v, dict) and v.get("seconds") is not None]
    return float(sum(arms)) if arms else None


def _gpu_seconds(stages):
    """(GPU seconds summed over this run's stage records, the GPU stages with no measured record)."""
    total, unmeasured = 0.0, []
    for stage in GPU_STAGES:
        prefixes = (stage + "_shard",) + GPU_RECORD_PREFIXES.get(stage, ())
        recs = [v for k, v in stages.items() if k == stage or k.startswith(prefixes)]
        secs = [_record_seconds(r) for r in recs]
        if not recs or any(s is None for s in secs):
            unmeasured.append(stage)
        total += sum(s for s in secs if s is not None)
    return total, unmeasured


def _finished_span(stages):
    """(first, last, seconds) over the stage records' finished_at times, or Nones."""
    import datetime

    ts = sorted(v["finished_at"] for v in stages.values() if v.get("finished_at"))
    if not ts:
        return None, None, None
    first, last = (datetime.datetime.fromisoformat(t) for t in (ts[0], ts[-1]))
    return ts[0], ts[-1], (last - first).total_seconds()


def _ledger_totals(run):
    """The ledger's recorded spend and cap (what the cap was enforced against)."""
    from eval.workspace_understanding.judge import LEDGER_REL

    if not run.exists(LEDGER_REL):
        return {"spent_usd": None, "cap_usd": None}
    l = run.read_json(LEDGER_REL)
    return {"spent_usd": l.get("spent_usd"), "cap_usd": l.get("cap_usd")}


def coverage_and_costs(run, scores):
    """The population, readout coverage, judged cells and spend, and measured stage times."""
    items = run.read_json("data/items.json")
    prov = read_provenance(run)
    excl = {}
    for x in items["items"]:
        if x["excluded"]:
            excl[x["exclusion_reason"]] = excl.get(x["exclusion_reason"], 0) + 1

    def _load(p):
        with open(p, encoding="utf-8") as h:
            return json.load(h)

    # sorted for stable output; the report's own record is left out so re-renders agree
    stages = {
        os.path.basename(p)[:-5]: _load(p)
        for p in sorted(glob.glob(run.file("stages/*.json")))
        if os.path.basename(p) != "report.json"
    }
    summariser = C.JUDGES[C.SUMMARISER]
    summariser_spend = _spend_from_logs(
        run,
        (judge_log(summariser.name, "summaries"), judge_log(summariser.name, "diagnostic_summaries")),
        model=summariser.model,
    )
    judges = {}
    api_total = summariser_spend["spend_usd"]
    for name in JUDGE_ORDER:
        spec = C.JUDGES[name]
        sp = _spend_from_logs(
            run,
            (judge_log(name, "naming"), judge_log(name, "diagnostics")),
            model=spec.model,
        )
        api_total += sp["spend_usd"]
        judges[name] = {
            "model": spec.model,
            "label": spec.label,
            "conditions": {
                c: {
                    "n_valid": sum(1 for s in scores if s["judged"][name][c]["named"] is not None),
                    "n_total": len(scores),
                }
                for c in C.JUDGED_CONDITIONS
            },
            "unavailable": sum(1 for s in scores for c in C.JUDGED_CONDITIONS if judge_unavailable(s, name, c)),
            "voided": sum(1 for s in scores for c in C.JUDGED_CONDITIONS if s["judged"][name][c]["voided_reason"]),
            "repaired": sum(1 for v in load_verdicts(run, name).values() if v["repaired"]),
            **{
                k: sp[k]
                for k in ("requests", "records", "input_tokens", "output_tokens", "spend_usd", "spend_usd_recorded")
            },
        }
    ledger_totals = _ledger_totals(run)
    stage_seconds = {k: _record_seconds(v) for k, v in stages.items()}
    first_finished, last_finished, span_s = _finished_span(stages)
    gpu_seconds, gpu_unmeasured = _gpu_seconds(stages)
    # GPU cost = measured stage seconds x the launcher's list rate, an estimate (eval.common.runs)
    gpu_rate, gpu_rate_source = launch_gpu_rate(launch_records(run), prov)
    gpu = gpu_cost_block(prov.get("gpu_type"), gpu_seconds, gpu_unmeasured, gpu_rate, gpu_rate_source)
    elapsed = elapsed_block(sum(v for k, v in stage_seconds.items() if k != "report" and v is not None),
                            [k for k, v in stage_seconds.items() if k != "report" and v is None])
    nla = S.load_nla(run)
    return {
        "population": {
            "kept": len(scores),
            "by_family": {f: sum(1 for s in scores if s["family"] == f) for f in C.FAMILIES},
            "excluded": excl,
            "multi_token_lens_targets": sum(1 for s in scores if s["multi_token"]),
        },
        "readout_coverage": {
            **{c: sum(1 for s in scores if s.get(c)) for c in C.FREE_TEXT_CONDITIONS},
            "jlens": sum(1 for s in scores if s["jlens"].get("has_record")),
        },
        "reread_coverage": {
            c: sum(1 for s in scores if s["fidelity"].get(c)) for c in ("maemm",) + tuple(C.REREAD_CONDITIONS)
        },
        "nla_reader": (
            {
                "revision": NLA.revision,
                "shards": nla["shards"],
                "n_items": len(nla["items"]),
                "close_rate": _mean([s["nla_diag"]["close_rate"] for s in scores if s.get("nla_diag")]),
                "close_rate_trunc": _mean([s["nla_diag"]["close_rate_trunc"] for s in scores if s.get("nla_diag")]),
                "mean_generated_tokens": _mean([s["nla_diag"]["mean_generated"] for s in scores if s.get("nla_diag")]),
            }
            if nla
            else "unavailable"
        ),
        "judges": judges,
        "summariser": {"model": summariser.model, **summariser_spend},
        "api_spend_total_usd": api_total,
        "api_spend_recorded_usd": ledger_totals["spent_usd"],
        "budget_cap_usd": ledger_totals["cap_usd"],
        "rates_per_m": {k: list(v) for k, v in C.RATES_PER_M.items()},
        "stage_seconds": stage_seconds,
        **elapsed,
        **gpu,
        "first_finished": first_finished,
        "last_finished": last_finished,
        "finished_span_s": span_s,
        "billing_usd": prov.get("billing_usd", "unavailable"),
        "cost_usd_list_price": {
            **{f"judge_{name}": j["spend_usd"] for name, j in judges.items()},
            "summariser": summariser_spend["spend_usd"],
            "total": api_total,
            "gpu_estimated": gpu["gpu_cost_usd_estimated"],
        },
    }
