"""Per-item scores under the word rule and the judge's folds, the long-format tables and the coverage and
cost block (methodology "Metrics", "Statistics").

Tables (tables/*.csv): `rates` (word rule and lens ranks), `judged` (named / foil / net), `contrasts` (reader
against reader, paired by item) and `modulation` (instruction against instruction, paired by concept). Every
row carries `band` (`final`, `mean`, or `carrier` for the lens-only any-token protocol) and every judged row
its `judge`. Single rates carry Wilson intervals, differences and pooled cells a percentile bootstrap."""

import glob, json, os

import numpy as np

from evals.downstream.common.judge_client import UNASKED_KINDS, unasked, unasked_detail
from evals.downstream.common.matcher import pass_at_n, whole_word_hit
from evals.downstream.common.runs import (
    StaleVerdict,
    elapsed_block,
    gpu_cost_block,
    launch_gpu_rate,
    launch_gpu_seconds,
    launch_records,
    read_provenance,
)
from evals.downstream.common.stats import (
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
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import lens as LENS
from evals.downstream.workspace_modulation import mean_cell as M
from evals.downstream.workspace_modulation import patchscope as PS
from evals.downstream.workspace_modulation import retrieval as RET
from evals.downstream.workspace_modulation.judge import verdict_of
from evals.downstream.workspace_modulation.stages import is_stage_record

TABLE_BANDS = C.READ_BANDS + (C.CARRIER_BAND,)
# The instruction contrasts, paired by concept: the two headline rows, then the appendix control's.
CONTRASTS_PAPER = (
    ("focus", "baseline"),
    ("mention", "baseline"),
    ("focus", "mention"),
    ("focus", "ignore"),
    ("ignore", "baseline"),
    ("dont_think", "baseline"),
    ("dont_think", "ignore"),
)
# Generation arm -> its condition name in every table.
READER_COND = {C.HEADLINE_ARM: "maem_reg", C.NULL_ARM: "maem_null", C.BASE_ARM: C.BASE_ARM}
# Every sampled reader is read at the headline arm's budgets; the search's pass@n is its n best windows.
NLA_BUDGETS = C.PASS_AT[C.HEADLINE_ARM]
RETRIEVAL_BUDGETS = C.PASS_AT[C.HEADLINE_ARM]
PATCH_BUDGETS = C.PASS_AT[C.HEADLINE_ARM]
assert max(PATCH_BUDGETS) == C.PATCH_GEN["n_samples"]
# named = own-target verdict, foil = the same readout against the foil's targets, net = named − foil.
JUDGED_METRICS = ("named", "foil", "net")


def judge_columns(name):
    """The two columns every judged row carries: the model that answered and the label a report prints."""
    spec = C.JUDGES[name]
    return {"judge": spec.model, "judge_label": spec.label}


# --- per-item scores (methodology, metrics) ---


def cell_hits(texts, forms):
    return [whole_word_hit(t, forms) for t in texts]


def item_word_rule(cells, forms, budgets, greedy):
    """cells: the item's rollout cells of one band. hit_any at budget n = any cell whose first n samples hit."""
    per_cell = [cell_hits([s["text"] for s in c["samples"]], forms) for c in cells]
    out = {
        "hit_any": {
            "greedy": (
                any(whole_word_hit(c["greedy"]["text"], forms) for c in cells if c.get("greedy")) if greedy else None
            )
        },
        "n_cells_read": len(cells),
    }
    for n in budgets:
        out["hit_any"][str(n)] = any(pass_at_n(h, n) for h in per_cell)
    out["consistency"] = float(np.mean([np.mean(h) for h in per_cell])) if per_cell else None
    return out


def donor_chance(cells, donor_forms_list, n):
    """The share of the item's twenty donor concepts whose own forms the readout would have hit: the chance
    line of that reader at that budget, over the same cells the rate is read at."""
    vals = [
        any(pass_at_n(cell_hits([s["text"] for s in c["samples"]], f), n) for c in cells) for f in donor_forms_list if f
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _nla_cells(cells, key):
    """One NLA record's cells in item_word_rule's shape: `text` for `nla`, `text_trunc` for `nla64`."""
    out = []
    for c in cells:
        g = c.get("greedy")
        out.append(
            {
                "pos": c["pos"],
                "greedy": {"text": (g.get(key) or "")} if g else None,
                "samples": [{"text": s.get(key) or ""} for s in c["samples"]],
            }
        )
    return out


def _lens_block(cells, single_token, any_layer, forms=()):
    """The lens metrics over one cell set. `word10_L42_any` is the word rule over the lens's top-10
    word-like tokens at layer 42. A concept with a lens record but no single-token form is a rank *miss* (False,
    kept in the denominator); a concept with no lens record was never measured (None). The any-layer criteria
    exist on the carrier band only (`any_layer`)."""
    if not cells:
        return {
            "rank1_any": None,
            "rank10_any": None,
            "rank10_L42_any": None,
            "word10_L42_any": None,
            "min_rank_L42": None,
            "min_rank": None,
            "min_rank_layer": None,
            "has_record": False,
        }
    word10 = any(whole_word_hit(t, forms) for c in cells for t in (c.get("top10_L42") or []))
    if not single_token:
        return {
            "rank1_any": False if any_layer else None,
            "rank10_any": False if any_layer else None,
            "rank10_L42_any": False,
            "word10_L42_any": word10,
            "min_rank_L42": None,
            "min_rank": None,
            "min_rank_layer": None,
            "has_record": True,
        }
    r42 = [c["rank_L42"] for c in cells]
    rmin = [c["min_rank"] for c in cells]
    best = min([v for v in rmin if v is not None], default=None)
    layer = next((c["min_rank_layer"] for c in cells if c["min_rank"] == best), None) if best is not None else None
    return {
        "rank1_any": any(v == 1 for v in rmin) if any_layer else None,
        "rank10_any": any(v is not None and v <= C.TOP_WORD for v in rmin) if any_layer else None,
        "rank10_L42_any": any(v is not None and v <= C.TOP_WORD for v in r42),
        "word10_L42_any": word10,
        "min_rank_L42": min([v for v in r42 if v is not None], default=None),
        "min_rank": best if any_layer else None,
        "min_rank_layer": layer if any_layer else None,
        "has_record": True,
    }


def _band8_block(pool, forms, donor_forms):
    """(block, chance) of the eight-layer pool at one cell: `word10_band8_any` and its 20-donor chance line;
    (None, nan) where the cell has no per-layer lists."""
    if pool is None:
        return None, float("nan")
    hits = lambda fs: any(whole_word_hit(t, fs) for t in pool)
    vals = [hits(f) for f in donor_forms if f]
    return {LENS_BAND_WORD_METRIC: hits(forms), "n_tokens": len(pool)}, (float(np.mean(vals)) if vals else float("nan"))


def _lens_chance(cells, single_token, any_layer):
    """The lens's 20-donor chance lines, one per rank criterion (rank ≤ 10 at 42, rank 1 and rank ≤ 10 any layer)."""
    nan = float("nan")
    if not cells or not single_token:
        return nan, nan, nan
    n_donors = min(len(cells[0].get("donor_rank_L42") or []), len(cells[0].get("donor_min_rank") or []))
    if not n_donors:
        return nan, nan, nan
    k10 = [
        any((c["donor_rank_L42"][j] is not None and c["donor_rank_L42"][j] <= C.TOP_WORD) for c in cells)
        for j in range(n_donors)
    ]
    if not any_layer:
        return float(np.mean(k10)), nan, nan
    k1 = [any(c["donor_min_rank"][j] == 1 for c in cells) for j in range(n_donors)]
    band10 = [
        any((c["donor_min_rank"][j] is not None and c["donor_min_rank"][j] <= C.TOP_WORD) for c in cells)
        for j in range(n_donors)
    ]
    return float(np.mean(k10)), float(np.mean(k1)), float(np.mean(band10))


def judge_log_rel(name, instrument):
    from evals.downstream.common.judge_client import judge_log

    return judge_log(name, instrument)


# The meta keys that, with the item and position, name a judged cell.
LOG_FIELD = {"naming": ("condition", "vs")}


def cell_value(meta, fields):
    """The tuple of `fields` read off one record's meta: the second half of a cell's name."""
    return tuple((meta or {}).get(f) for f in fields)


def _load_log(run, rel, fields=LOG_FIELD["naming"]):
    """One judge log grouped by (item, <fields>), keeping the last record per (key, meta): the log is
    append-only and a resumed request appends a replacement."""
    last = {}
    for r in run.read_jsonl(rel):
        last[(r.get("key"), json.dumps(r.get("meta"), sort_keys=True, ensure_ascii=False))] = r
    by = {}
    for r in last.values():
        m = r.get("meta") or {}
        by.setdefault((m.get("i"), cell_value(m, fields)), []).append(r)
    return by


def judged_now(run):
    """{instrument: {(i, pos, *fields): request}}: every request `judge.build_requests` builds from the run as it stands."""
    from evals.downstream.workspace_modulation import judge as J

    built = J.build_requests(J._load_inputs(run))
    built["naming"] = built["naming"] + built["empty"]
    return {
        instrument: {(r["meta"]["i"], r["meta"]["pos"]) + cell_value(r["meta"], fields): r for r in built[instrument]}
        for instrument, fields in LOG_FIELD.items()
    }


def bind_log(by, now, judge, instrument):
    """`by` with every verdict joined to the readout it was given for (`readout`, or the request key where a
    record has none). A cell whose records are all about another text raises StaleVerdict; a record whose cell
    has no request now is dropped."""
    from evals.downstream.common.judge_client import request_key, with_judge

    spec, fields = C.JUDGES[judge], LOG_FIELD[instrument]
    out, cells = {}, {}
    for (i, value), recs in by.items():
        for r in recs:
            cell = (i, (r.get("meta") or {}).get("pos")) + tuple(value)
            req = now.get(cell)
            if req is None:
                continue
            if "readout" in r:
                same = r["readout"] == req["readout"]
            else:
                same = r.get("key") == request_key(with_judge(req, spec))
            cells.setdefault(cell, []).append(same)
            if same:
                out.setdefault((i, value), []).append(r)
    stale = sorted((c for c, same in cells.items() if not any(same)), key=repr)
    if stale:
        i, pos, *value = stale[0]
        named = ", ".join(f"{f} {v!r}" for f, v in zip(fields, value))
        raise StaleVerdict(
            f"judge `{judge}`, {instrument}: the verdict logged for item {i}, position {pos}, {named} "
            f"was given for another text than the readout this directory now holds there "
            f"({len(stale)} such cell(s) in {judge_log_rel(judge, instrument)}). The readout was rewritten "
            f"after it was judged; run the `judge` stage again, which asks only what its logs do not "
            f"already answer."
        )
    return out


def one_record_per_cell(recs, resolved):
    """`recs` collapsed to one record per cell: a truncation re-ask is a new key for the same question, so the
    record that yielded a verdict wins, else the last; a record about a different readout supersedes."""
    by = {}
    for r in recs:
        pos = (r.get("meta") or {}).get("pos")
        prev = by.get(pos)
        if prev is None or r.get("readout") != prev.get("readout") or resolved(r) or not resolved(prev):
            by[pos] = r
    return [by[k] for k in sorted(by, key=lambda p: (p is None, p))]


def resolved(rec):
    """Whether a saved judge record carries a verdict the fold can read."""
    v = verdict_of(rec)
    return v is not None and bool(v["parse_ok"])


def _fold_one(recs):
    """(n_empty, n_attempted, n_valid, n_voided, named) over the records of one question at one cell. An
    `empty_readout` record was never sent and counts in neither n_attempted nor n_valid."""
    n_empty = sum(1 for r in recs if r.get("status") == "empty_readout")
    recs = [r for r in recs if r.get("status") != "empty_readout"]
    verdicts = [verdict_of(r) for r in recs]
    valid = [v for v in verdicts if v is not None and v["parse_ok"]]  # == resolved(), per record
    named = any(v["named"] for v in valid) if valid else None
    return n_empty, len(recs), len(valid), sum(1 for v in valid if v.get("voided")), named


def fold_judged(own_recs, foil_recs=()):
    """One item's named, foil and net (named − foil, None unless both resolved) for one reader, judge and cell."""
    own, foil = _fold_one(own_recs), _fold_one(foil_recs)
    out = {
        "n_attempted": own[1] + foil[1],
        "n_valid": own[2] + foil[2],
        "n_voided": own[3] + foil[3],
        "n_empty": own[0] + foil[0],
        "named": own[4],
        "foil": foil[4],
    }
    out["net"] = None if (own[4] is None or foil[4] is None) else float(own[4]) - float(foil[4])
    return out


def judge_unavailable(fold):
    """A reader is unavailable for an item when its own-target verdict did not resolve."""
    return fold["named"] is None


def _load_context(run):
    """Every artefact the scoring reads, None per artefact whose stage has not run."""
    from evals.downstream.workspace_modulation.rollouts import merged_rel

    rolls = {}
    for arm in READER_COND:
        rel = merged_rel(arm)
        rolls[arm] = {r["i"]: r for r in run.read_json(rel)["items"]} if run.exists(rel) else {}
    nla_doc = run.read_json("rollouts/nla.json") if run.exists("rollouts/nla.json") else None
    ret_doc = run.read_json(RET.MERGED_REL) if run.exists(RET.MERGED_REL) else None
    patch_doc = run.read_json(C.PATCH_REL) if run.exists(C.PATCH_REL) else None
    logs, now = {}, None
    for name in C.JUDGES:
        for instrument, field in LOG_FIELD.items():
            rel = judge_log_rel(name, instrument)
            if not run.exists(rel):
                logs[(name, instrument)] = None
                continue
            now = judged_now(run) if now is None else now
            logs[(name, instrument)] = bind_log(_load_log(run, rel, fields=field), now[instrument], name, instrument)
    return {
        "rolls": rolls,
        "lens": {r["i"]: r for r in run.read_json("lens/lens.json")["items"]} if run.exists("lens/lens.json") else {},
        "band_pool": LENS.band_pools(run),
        "nla": {r["i"]: r for r in nla_doc["items"]} if nla_doc else None,
        "retrieval": {r["i"]: r for r in ret_doc["items"]} if ret_doc else None,
        "patch": (
            {arm: {r["i"]: r for r in ((patch_doc["arms"].get(arm) or {}).get("items") or [])} for arm in C.PATCH_ARMS}
            if patch_doc
            else None
        ),
        "logs": logs,
        "compliance": run.read_json("data/compliance.json") if run.exists("data/compliance.json") else {},
    }


def _judged_folds(it, ctx, pos, band):
    """One item's judged folds at one cell, per judge and reader; None when no judge covered the item."""
    readers = C.judged_readers_at(band)
    blocks = {}
    for name in C.JUDGES:
        log = ctx["logs"][(name, "naming")]
        if log is None:
            continue
        if not any((it["i"], (cond, vs)) in log for cond in readers for vs in C.VS):
            continue
        at = lambda cond, vs: one_record_per_cell(
            [r for r in log.get((it["i"], (cond, vs)), []) if (r.get("meta") or {}).get("pos") == pos], resolved
        )
        blocks[name] = {cond: fold_judged(at(cond, "own"), at(cond, "foil")) for cond in readers}
    return blocks or None


def band_block(it, ctx, band, donor_forms):
    """Every block of one item at one band: the word rule of each MAEM arm, NLA reader, the search and
    the Patchscopes arms, the lens blocks and chance lines, and the judged folds. The carrier band carries the
    lens alone; the null control and the Patchscopes floor are read from their final-period readouts at the
    mean cell; the ablation exists at the final period only."""
    lens_cells = [c for c in ((ctx["lens"].get(it["i"]) or {}).get("cells") or [])]
    if band == C.CARRIER_BAND:
        cells = [int(p) for p in it["carrier_cells"]]
        any_layer = True
    else:
        cells = [int(p) for p in M.band_cells(band, it)]
        any_layer = False
    cs = set(cells)
    out = {"cells": list(cells), "maem": {}, "nla": {}, "retrieval": None, "chance": {}}
    for arm, cond in READER_COND.items():
        want = M.band_cells(C.MEAN_CONTROL_FROM_BAND, it) if (arm == C.NULL_ARM and band == C.MEAN_BAND) else cells
        by_pos = {c["pos"]: c for c in ((ctx["rolls"][arm].get(it["i"]) or {}).get("cells") or [])}
        got = [by_pos[p] for p in want if p in by_pos]
        n = max(C.PASS_AT[arm])
        if band == C.CARRIER_BAND or not want or len(got) != len(want):
            out["maem"][arm] = None
            out["chance"][f"{cond}_hit{n}"] = float("nan")
            continue
        out["maem"][arm] = item_word_rule(got, it["forms"], C.PASS_AT[arm], C.ARMS[arm]["greedy"])
        out["chance"][f"{cond}_hit{n}"] = donor_chance(got, donor_forms, n)
    nla_cells = [c for c in ((ctx["nla"] or {}).get(it["i"]) or {}).get("cells") or [] if c["pos"] in cs]
    for cond, key in (("nla", "text"), ("nla64", "text_trunc")):
        if band == C.CARRIER_BAND or not nla_cells:
            out["nla"][cond] = None
            out["chance"][f"{cond}_hit{max(NLA_BUDGETS)}"] = float("nan")
            continue
        cs_nla = _nla_cells(nla_cells, key)
        w = item_word_rule(cs_nla, it["forms"], NLA_BUDGETS, C.ARMS["nla"]["greedy"])
        closed = [bool(sm.get("closed")) for c in nla_cells for sm in c["samples"]]
        w["close_rate"] = float(np.mean(closed)) if closed else None
        out["nla"][cond] = w
        out["chance"][f"{cond}_hit{max(NLA_BUDGETS)}"] = donor_chance(cs_nla, donor_forms, max(NLA_BUDGETS))
    # the search: its ranked windows are its samples; no greedy row
    ret_cells = [c for c in ((ctx["retrieval"] or {}).get(it["i"]) or {}).get("cells") or [] if c["pos"] in cs]
    if band == C.CARRIER_BAND or not ret_cells:
        out["chance"][f"{C.RETRIEVAL_READER}_hit{max(RETRIEVAL_BUDGETS)}"] = float("nan")
    else:
        out["retrieval"] = item_word_rule(ret_cells, it["forms"], RETRIEVAL_BUDGETS, greedy=False)
        out["chance"][f"{C.RETRIEVAL_READER}_hit{max(RETRIEVAL_BUDGETS)}"] = donor_chance(
            ret_cells, donor_forms, max(RETRIEVAL_BUDGETS)
        )
    out["patch"] = {}
    for arm in C.PATCH_ARMS:
        n = max(PATCH_BUDGETS)
        want = [] if band == C.CARRIER_BAND else [PS.patch_pos(it, p, arm) for p in cells]
        by_pos = {c["pos"]: c for c in (((ctx["patch"] or {}).get(arm) or {}).get(it["i"]) or {}).get("cells") or []}
        got = [by_pos[p] for p in want if p in by_pos]
        if not want or len(got) != len(want):
            out["patch"][arm] = None
            out["chance"][f"{arm}_hit{n}"] = float("nan")
            continue
        out["patch"][arm] = item_word_rule(got, it["forms"], PATCH_BUDGETS, C.PATCH_GEN["greedy"])
        out["chance"][f"{arm}_hit{n}"] = donor_chance(got, donor_forms, n)
    at_band = [c for c in lens_cells if c["pos"] in cs]
    out["jlens"] = (
        _lens_block(at_band, it["single_token"], any_layer, it["forms"])
        if lens_cells
        else _lens_block([], True, any_layer)
    )
    k10, k1, band10 = _lens_chance(at_band, it["single_token"], any_layer)
    out["chance"].update(
        {"jlens_rank10_L42_any": k10, "jlens_rank1_any": k1, "jlens_rank10_any": band10}
    )
    out["jlens_band8"], out["chance"]["jlens_band8_word"] = _band8_block(
        ctx["band_pool"].get((int(it["i"]), int(cells[0]))) if band == C.FINAL_BAND else None,
        it["forms"],
        donor_forms,
    )
    out["judged"] = None if band == C.CARRIER_BAND else _judged_folds(it, ctx, cells[0], band)
    return out


def score_items(run):
    doc = run.read_json("data/items.json")
    kept = [x for x in doc["items"] if not x["excluded"]]
    concepts = doc.get("concepts") or {}
    ctx = _load_context(run)
    out = []
    for it in kept:
        i = it["i"]
        donor_forms = [(concepts.get(d) or {}).get("forms") or [] for d in it["donors"]]
        rec = {
            "i": i,
            "family": it["family"],
            "concept_key": it["concept_key"],
            "concept": it["concept"],
            "instruction": it["instruction"],
            "forms": it["forms"],
            "carrier": it["carrier"],
            "carrier_cells": list(it["carrier_cells"]),
            "final_pos": M.final_pos(it),
            "single_token": bool(it["single_token"]),
            "n_cells": len(it["carrier_cells"]),
            "n_donors_used": len([f for f in donor_forms if f]),
        }
        rec["bands"] = {band: band_block(it, ctx, band, donor_forms) for band in TABLE_BANDS}
        c = (ctx["compliance"] or {}).get(str(i))
        rec["compliance"] = (
            {"copies_carrier": bool(c["copies_carrier"]), "names_target": bool(c["names_target"])} if c else None
        )
        out.append(rec)
    return out


# --- accessors over a score record at one band ---------------------------------------------------------


def _band(s, band):
    return (s.get("bands") or {}).get(band) or {}


def _word_block(b, cond):
    """One band block's word-rule block for a reader."""
    if cond in READER_COND:
        return (b.get("maem") or {}).get(cond)
    if cond in C.PATCH_ARMS:
        return (b.get("patch") or {}).get(cond)
    if cond == C.RETRIEVAL_READER:
        return b.get("retrieval")
    return (b.get("nla") or {}).get(cond)


def _wr(s, band, key, cond):
    """One word-rule block's hit at a budget, or None when the reader was never read at this band."""
    block = _word_block(_band(s, band), cond)
    return None if block is None else block["hit_any"][key]


def _lens(s, band, metric):
    return (_band(s, band).get("jlens") or {}).get(metric)


def _band8(s, band):
    """The eight-layer lens readout's word-rule hit at one band, None where it was never read."""
    return (_band(s, band).get("jlens_band8") or {}).get(LENS_BAND_WORD_METRIC)


def _fold(s, band, name, cond):
    return ((_band(s, band).get("judged") or {}).get(name) or {}).get(cond)


def _fold_metric(s, band, name, cond, metric):
    f = _fold(s, band, name, cond)
    return None if f is None else f.get(metric)


def _chance(s, band, key):
    return (_band(s, band).get("chance") or {}).get(key, float("nan"))


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _vec(values):
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


N_HEAD = max(C.PASS_AT[C.HEADLINE_ARM])


def _judged_entries(cond):
    """A reader's judged metrics, stored by judged-condition name so `metric_accessor` can bind a judge."""
    return {m: (cond, m) for m in JUDGED_METRICS}


# Every reader a table compares: metric -> (source condition, accessor over (score, band)). The first
# metric of each reader is its headline criterion. The untrained-base ablation has a rates row and nothing else.
READER_METRICS = {
    "maem_reg": {
        "hit_any": ("maem_reg", lambda s, b: _wr(s, b, str(N_HEAD), C.HEADLINE_ARM)),
        **_judged_entries("maem_reg8"),
    },
    "nla": {
        "hit_any": ("nla", lambda s, b: _wr(s, b, str(max(NLA_BUDGETS)), "nla")),
        **_judged_entries("nla_n8"),
    },
    "nla64": {
        "hit_any": ("nla64", lambda s, b: _wr(s, b, str(max(NLA_BUDGETS)), "nla64")),
    },
    C.RETRIEVAL_READER: {
        "hit_any": (C.RETRIEVAL_READER, lambda s, b: _wr(s, b, str(max(RETRIEVAL_BUDGETS)), C.RETRIEVAL_READER)),
        **_judged_entries(C.RETRIEVAL_JUDGED),
    },
    "jlens_L42": {
        "rank10_L42_any": ("jlens_L42", lambda s, b: _lens(s, b, "rank10_L42_any")),
        **_judged_entries(C.LENS_READER),
    },
    "maem_null": {
        "hit_any": ("maem_null", lambda s, b: _wr(s, b, str(max(C.PASS_AT[C.NULL_ARM])), C.NULL_ARM)),
        **_judged_entries("maem_null8"),
    },
    C.PATCH_ARM: {
        "hit_any": (C.PATCH_ARM, lambda s, b: _wr(s, b, str(max(PATCH_BUDGETS)), C.PATCH_ARM)),
        **_judged_entries(C.PATCH_JUDGED[C.PATCH_ARM]),
    },
    C.PATCH_FLOOR: {
        "hit_any": (C.PATCH_FLOOR, lambda s, b: _wr(s, b, str(max(PATCH_BUDGETS)), C.PATCH_FLOOR)),
    },
}
READERS = tuple(READER_METRICS)
HEADLINE_METRIC = {r: next(iter(m)) for r, m in READER_METRICS.items()}
# MAEM against each reader, paired by item.
READER_CONTRASTS = (
    ("maem_reg", "nla"),
    ("maem_reg", "nla64"),
    ("maem_reg", C.RETRIEVAL_READER),
    ("maem_reg", "jlens_L42"),
    ("maem_reg", "maem_null"),
)
# The Patchscopes arm against its no-patch floor, on the word rule.
PATCH_CONTRASTS = ((C.PATCH_ARM, C.PATCH_FLOOR),)


def metric_accessor(reader, metric, judge_name=None):
    """(source condition, accessor over (score, band)) of one (reader, metric); (None, None) when absent."""
    entry = READER_METRICS[reader].get(metric)
    if entry is None:
        return None, None
    source, how = entry
    if callable(how):
        return source, how
    return source, (lambda s, b, _c=source, _m=how, _j=judge_name: _fold_metric(s, b, _j, _c, _m))


def is_judged(reader, metric):
    entry = READER_METRICS[reader].get(metric)
    return entry is not None and not callable(entry[1])


def _judge_names(reader, metric):
    return list(C.JUDGES) if is_judged(reader, metric) else [None]


# --- the long-format tables (methodology, metrics) ------------------------------------------------------


def _in_group(s, g):
    return s["family"] == g


def _cell_items(scores, g, instr):
    return [s for s in scores if _in_group(s, g) and (instr == "" or s["instruction"] == instr)]


def _rate_row(metric, cond, g, values, n_valid=None, **extra):
    """A Wilson row over the non-missing values: None leaves the denominator, False stays in it.
    `n_valid` overrides only the reported count."""
    vals = [v for v in values if v is not None]
    k = sum(1 for v in vals if v)
    est, lo, hi = wilson(k, len(vals))
    return row(
        metric, cond, g, est, lo, hi, len(values), len(vals) if n_valid is None else n_valid, ci_method=CI_WILSON, **extra
    )


def _ratio(top, chance_est):
    """(rate / chance, below CHANCE_FLAG_RATIO × chance); no ratio when chance is zero."""
    ratio = top / chance_est if chance_est else ""
    flag = bool(chance_est) and bool(top < C.CHANCE_FLAG_RATIO * chance_est)
    return ratio, flag


# (condition, metric, rank cutoff, chance key); the any-layer criteria on the carrier band only.
LENS_RATES_MATCHED = (("jlens_L42", "rank10_L42_any", C.TOP_WORD, "jlens_rank10_L42_any"),)
# The paper's J-lens word rules (methodology "Metrics"): layer 42, and the eight-layer pool.
LENS_WORD_METRIC = "word10_L42_any"
LENS_BAND_WORD_METRIC = "word10_band8_any"
LENS_BAND_COND = "jlens_" + C.LENS_BAND
LENS_RATES_ALL = LENS_RATES_MATCHED + (
    ("jlens_best", "rank1_any", 1, "jlens_rank1_any"),
    ("jlens_best", "rank10_any", C.TOP_WORD, "jlens_rank10_any"),
)


def _word_rule_rows(S, g, instr, idx, cond, band, get, budgets, greedy, chance_key):
    """Every rates row of one word-rule reader: hit_any per budget (the largest carrying its chance line,
    ratio and flag), greedy, consistency and chance."""
    out = []
    nmax = max(budgets)
    ch_vec = [_chance(s, band, chance_key) for s in S]
    chance_est, chance_lo, chance_hi = boot_mean(ch_vec, idx)
    ratio, flag = _ratio(_mean([get(s, str(nmax)) for s in S]), chance_est)
    for n in budgets:
        extra = {"instruction": instr, "band": band, "budget_type": "samples", "budget": n}
        if n == nmax:
            extra.update(chance=chance_est, ratio_vs_chance=ratio, flag_below_3x=flag)
        out.append(_rate_row("hit_any", cond, g, [get(s, str(n)) for s in S], **extra))
    if greedy:
        out.append(
            _rate_row(
                "greedy_hit",
                cond,
                g,
                [get(s, "greedy") for s in S],
                instruction=instr,
                band=band,
                budget_type="samples",
                budget=0,
            )
        )
    cons = [_consistency(s, band, cond) for s in S]
    est, lo, hi = boot_mean(_vec(cons), idx)
    out.append(
        row(
            "consistency", cond, g, est, lo, hi, len(S), finite_n(cons), ci_method=CI_BOOT,
            instruction=instr, band=band, budget_type="samples", budget=nmax,
        )
    )
    out.append(
        row(
            "chance", cond, g, chance_est, chance_lo, chance_hi, len(S), finite_n(ch_vec), ci_method=CI_BOOT,
            instruction=instr, band=band, budget_type="samples", budget=nmax,
            ratio_vs_chance=ratio, flag_below_3x=flag,
        )
    )
    return out


def _consistency(s, band, cond):
    """A word-rule block's consistency; `cond` is a table condition, mapped back to its arm."""
    arm = next((a for a, c in READER_COND.items() if c == cond), cond)
    return (_word_block(_band(s, band), arm) or {}).get("consistency")


def _lens_rate_rows(S, g, instr, idx, band, rates):
    """One rate row and one chance row per lens rank criterion."""
    out = []
    n_lens = sum(1 for s in S if _lens(s, band, "has_record") and s["single_token"])
    for cond, metric, k, chance_key in rates:
        ch_vec = [_chance(s, band, chance_key) for s in S]
        chance_est, chance_lo, chance_hi = boot_mean(ch_vec, idx)
        vals = [_lens(s, band, metric) for s in S]
        ratio, flag = _ratio(_mean(vals), chance_est)
        out.append(
            _rate_row(
                metric, cond, g, vals, n_valid=n_lens, instruction=instr, band=band,
                budget_type="rank_cutoff", budget=k,
                chance=chance_est, ratio_vs_chance=ratio, flag_below_3x=flag,
            )
        )
        out.append(
            row(
                "chance", cond, g, chance_est, chance_lo, chance_hi, len(S), finite_n(ch_vec), ci_method=CI_BOOT,
                instruction=instr, band=band, budget_type="rank_cutoff", budget=k,
                ratio_vs_chance=ratio, flag_below_3x=flag,
            )
        )
    return out


def _rate_block(S, g, instr, idx, band):
    out = []
    for arm, cond in READER_COND.items():
        if not any((_band(s, band).get("maem") or {}).get(arm) for s in S):
            continue
        out += _word_rule_rows(
            S, g, instr, idx, cond, band,
            lambda s, key, a=arm: _wr(s, band, key, a),
            C.PASS_AT[arm], C.ARMS[arm]["greedy"], f"{cond}_hit{max(C.PASS_AT[arm])}",
        )
    for cond in C.NLA_READERS:
        if not any((_band(s, band).get("nla") or {}).get(cond) for s in S):
            continue
        out += _word_rule_rows(
            S, g, instr, idx, cond, band,
            lambda s, key, c=cond: _wr(s, band, key, c),
            NLA_BUDGETS, C.ARMS["nla"]["greedy"], f"{cond}_hit{max(NLA_BUDGETS)}",
        )
    if any(_band(s, band).get("retrieval") for s in S):
        out += _word_rule_rows(
            S, g, instr, idx, C.RETRIEVAL_READER, band,
            lambda s, key: _wr(s, band, key, C.RETRIEVAL_READER),
            RETRIEVAL_BUDGETS, False, f"{C.RETRIEVAL_READER}_hit{max(RETRIEVAL_BUDGETS)}",
        )
    if any(_lens(s, band, "has_record") for s in S):
        out += _lens_rate_rows(
            S, g, instr, idx, band, LENS_RATES_ALL if band == C.CARRIER_BAND else LENS_RATES_MATCHED
        )
        out.append(
            _rate_row(
                LENS_WORD_METRIC, "jlens_L42", g, [_lens(s, band, LENS_WORD_METRIC) for s in S],
                instruction=instr, band=band, budget_type="top_tokens", budget=C.TOP_WORD,
            )
        )
    if any(_band8(s, band) is not None for s in S):
        out += _band8_rate_rows(S, g, instr, idx, band)
    for arm in C.PATCH_ARMS:
        if not any(_word_block(_band(s, band), arm) for s in S):
            continue
        out += _word_rule_rows(
            S, g, instr, idx, arm, band,
            lambda s, key, a=arm: _wr(s, band, key, a),
            PATCH_BUDGETS, C.PATCH_GEN["greedy"], f"{arm}_hit{max(PATCH_BUDGETS)}",
        )
    return out


def _band8_rate_rows(S, g, instr, idx, band):
    """The eight-layer pool's word-rule row and its chance row."""
    ch_vec = [_chance(s, band, "jlens_band8_word") for s in S]
    chance_est, chance_lo, chance_hi = boot_mean(ch_vec, idx)
    vals = [_band8(s, band) for s in S]
    ratio, flag = _ratio(_mean(vals), chance_est)
    extra = {"instruction": instr, "band": band, "budget_type": "top_tokens", "budget": C.TOP_WORD}
    return [
        _rate_row(LENS_BAND_WORD_METRIC, LENS_BAND_COND, g, vals, chance=chance_est, ratio_vs_chance=ratio,
                  flag_below_3x=flag, **extra),
        row("chance", LENS_BAND_COND, g, chance_est, chance_lo, chance_hi, len(S), finite_n(ch_vec),
            ci_method=CI_BOOT, ratio_vs_chance=ratio, flag_below_3x=flag, **extra),
    ]


# --- the judged tables -----------------------------------------------------------------------------------


def judged_ran(scores):
    return any(_band(s, band).get("judged") for s in scores for band in C.READ_BANDS)


def _judged_items(scores, g, band):
    return [s for s in scores if _in_group(s, g) and _band(s, band).get("judged")]


def _judged_block(scores, band, instr=""):
    """judged.csv at one band: per (group, judge, reader) a Wilson `named` and `foil`, a bootstrapped `net`,
    and the counts of voided, unresolved and empty readouts."""
    out = []
    for g in C.GROUPS:
        S = [s for s in _judged_items(scores, g, band) if instr == "" or s["instruction"] == instr]
        if not S:
            continue
        idx = boot_indices(len(S), C.N_BOOT, C.BOOTSTRAP_SEED)
        for name in C.JUDGES:
            for cond in C.judged_readers_at(band):
                folds = [_fold(s, band, name, cond) for s in S]
                base = {"instruction": instr, "band": band, **judge_columns(name)}
                for metric in ("named", "foil"):
                    vals = [None if f is None else f.get(metric) for f in folds]
                    k = sum(1 for v in vals if v is True)
                    n = sum(1 for v in vals if v is not None)
                    est, lo, hi = wilson(k, n)
                    out.append(row(metric, cond, g, est, lo, hi, len(S), n, ci_method=CI_WILSON, count=k, **base))
                net = _vec([None if f is None else f.get("net") for f in folds])
                est, lo, hi = boot_mean(net, idx)
                out.append(row("net", cond, g, est, lo, hi, len(S), finite_n(net), ci_method=CI_BOOT, **base))
                for metric, count in (
                    ("voided_readouts", sum(f["n_voided"] for f in folds if f)),
                    ("unresolved_readouts", sum(f["n_attempted"] - f["n_valid"] for f in folds if f)),
                    ("empty_readouts", sum(f["n_empty"] for f in folds if f)),
                    ("unavailable_items", sum(1 for f in folds if f and judge_unavailable(f))),
                ):
                    out.append(
                        row(
                            metric, cond, g, count, float("nan"), float("nan"),
                            sum(f["n_attempted"] for f in folds if f),
                            sum(f["n_valid"] for f in folds if f),
                            **base,
                        )
                    )
    return out


# --- contrasts (methodology, metrics) --------------------------------------------------------------------


def _paired_row(metric, cond, g, a_vals, b_vals, idx, **extra):
    est, lo, hi, n = paired_diff(_vec(a_vals), _vec(b_vals), idx)
    return row(metric, cond, g, est, lo, hi, len(a_vals), n, ci_method=CI_BOOT, n_pairs=n, **extra)


def _by_concept(S):
    """(concept keys, one bootstrap index matrix, a picker) for a group."""
    by_key = {}
    for s in S:
        by_key.setdefault(s["concept_key"], {})[s["instruction"]] = s
    keys = sorted(by_key)
    idx = boot_indices(len(keys), C.N_BOOT, C.BOOTSTRAP_SEED)
    pick = lambda k, instr, f: (f(by_key[k][instr]) if instr in by_key[k] else None)
    return keys, idx, pick


def _measured(S, band, f):
    return any(f(s, band) is not None for s in S)


def _modulation_block(scores, band):
    """modulation.csv at one band: each reader's rate under one instruction minus another, paired by concept."""
    out = []
    for g in C.GROUPS:
        S = [s for s in scores if _in_group(s, g)]
        if not S:
            continue
        keys, idx, pick = _by_concept(S)
        for instr_a, instr_b in CONTRASTS_PAPER:
            for cond, metrics in READER_METRICS.items():
                for metric in metrics:
                    for name in _judge_names(cond, metric):
                        source, f = metric_accessor(cond, metric, name)
                        if not _measured(S, band, f):
                            continue
                        out.append(
                            _paired_row(
                                metric, cond, g,
                                [pick(k, instr_a, lambda s: f(s, band)) for k in keys],
                                [pick(k, instr_b, lambda s: f(s, band)) for k in keys],
                                idx,
                                band=band,
                                condition_a=cond,
                                condition_b=cond,
                                instruction_a=instr_a,
                                instruction_b=instr_b,
                                source_condition=source,
                                **(judge_columns(name) if name else {}),
                            )
                        )
    return out


# focus and mention pooled into one cell; its interval resamples concepts.
POOLED_INSTRUCTION = "+".join(C.HEADLINE_CONDITIONS)
CI_CONCEPT_BOOT = f"bootstrap_concept_{C.N_BOOT}_seed{C.BOOTSTRAP_SEED}"


def _concept_rate(S, f):
    """(estimate, lo, hi, n_valid) of an item-level mean over `S`, its interval a percentile bootstrap over
    concepts; None values leave numerator and denominator."""
    by_key = {}
    for s in S:
        v = f(s)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        sm, n = by_key.get(s["concept_key"], (0.0, 0))
        by_key[s["concept_key"]] = (sm + float(v), n + 1)
    if not by_key:
        return float("nan"), float("nan"), float("nan"), 0
    keys = sorted(by_key)
    sums = np.array([by_key[k][0] for k in keys])
    counts = np.array([by_key[k][1] for k in keys])
    idx = boot_indices(len(keys), C.N_BOOT, C.BOOTSTRAP_SEED)
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(lo), float(hi), int(counts.sum())


def _pooled_headline_block(scores, band):
    """(rates rows, judged rows) under focus and mention pooled (`POOLED_INSTRUCTION`, concept bootstrap):
    the paper's Modulation column."""
    rates, judged = [], []
    ext = {"instruction": POOLED_INSTRUCTION, "band": band}

    def one(metric, cond, g, S, f, **extra):
        est, lo, hi, n = _concept_rate(S, f)
        return row(metric, cond, g, est, lo, hi, len(S), n, ci_method=CI_CONCEPT_BOOT, **ext, **extra)

    for g in C.GROUPS:
        S = [s for s in scores if _in_group(s, g) and s["instruction"] in C.HEADLINE_CONDITIONS]
        if not S:
            continue
        word = [(cond, arm, C.PASS_AT[arm]) for arm, cond in READER_COND.items()]
        word += [(cond, cond, NLA_BUDGETS) for cond in C.NLA_READERS]
        word += [(C.RETRIEVAL_READER, C.RETRIEVAL_READER, RETRIEVAL_BUDGETS)]
        for cond, store, budgets in word:
            if not any(_word_block(_band(s, band), store) for s in S):
                continue
            for n in budgets:
                rates.append(
                    one("hit_any", cond, g, S, lambda s, _k=str(n), _a=store: _wr(s, band, _k, _a),
                        budget_type="samples", budget=n)
                )
        if any(_lens(s, band, "has_record") for s in S):
            for metric in ("rank10_L42_any", LENS_WORD_METRIC):
                rates.append(
                    one(metric, "jlens_L42", g, S, lambda s, _m=metric: _lens(s, band, _m),
                        budget_type="rank_cutoff" if metric.startswith("rank") else "top_tokens", budget=C.TOP_WORD)
                )
        if any(_band8(s, band) is not None for s in S):
            rates.append(one(LENS_BAND_WORD_METRIC, LENS_BAND_COND, g, S, lambda s: _band8(s, band),
                             budget_type="top_tokens", budget=C.TOP_WORD))
            rates.append(one("chance", LENS_BAND_COND, g, S, lambda s: _chance(s, band, "jlens_band8_word"),
                             budget_type="top_tokens", budget=C.TOP_WORD))
        for arm in C.PATCH_ARMS:
            if not any(_word_block(_band(s, band), arm) for s in S):
                continue
            for n in PATCH_BUDGETS:
                rates.append(
                    one("hit_any", arm, g, S, lambda s, _k=str(n), _a=arm: _wr(s, band, _k, _a),
                        budget_type="samples", budget=n)
                )
            rates.append(one("chance", arm, g, S, lambda s, _a=arm: _chance(s, band, f"{_a}_hit{max(PATCH_BUDGETS)}"),
                             budget_type="samples", budget=max(PATCH_BUDGETS)))
        J = [s for s in S if _band(s, band).get("judged")]
        if not J:
            continue
        for name in C.JUDGES:
            for cond in C.judged_readers_at(band):
                for metric in JUDGED_METRICS:
                    judged.append(
                        one(metric, cond, g, J, lambda s, _c=cond, _m=metric: _fold_metric(s, band, name, _c, _m),
                            **judge_columns(name))
                    )
    return rates, judged


def _reader_contrast_block(scores, band):
    """contrasts.csv at one band: reader against reader on the same items, inside one family and one
    instruction ("" pools the instructions), on each reader's headline criterion and the judged metrics."""
    out = []
    for g in C.GROUPS:
        for instr in ("",) + C.CONDITIONS:
            S = _cell_items(scores, g, instr)
            if not S:
                continue
            idx = boot_indices(len(S), C.N_BOOT, C.BOOTSTRAP_SEED)
            pairs = [(a, HEADLINE_METRIC[a], b, HEADLINE_METRIC[b]) for a, b in READER_CONTRASTS]
            pairs += [
                (a, m, b, m)
                for a, b in READER_CONTRASTS
                for m in C.JUDGED_CONTRAST_METRICS
                if is_judged(a, m) and is_judged(b, m)
            ]
            pairs += [(a, HEADLINE_METRIC[a], b, HEADLINE_METRIC[b]) for a, b in PATCH_CONTRASTS]
            for a, ma, b, mb in pairs:
                for name in _judge_names(a, ma):
                    (_sa, fa), (_sb, fb) = metric_accessor(a, ma, name), metric_accessor(b, mb, name)
                    if not (fa and fb and _measured(S, band, fa) and _measured(S, band, fb)):
                        continue
                    out.append(
                        _paired_row(
                            ma, f"{a}-{b}", g,
                            [fa(s, band) for s in S],
                            [fb(s, band) for s in S],
                            idx,
                            band=band,
                            condition_a=a,
                            condition_b=b,
                            instruction_a=instr,
                            instruction_b=instr,
                            metric_b=mb,
                            **(judge_columns(name) if name else {}),
                        )
                    )
    return out


TABLES = ("rates", "judged", "contrasts", "modulation")


def tables(scores, out_dir):
    """Every table, from `scores` alone; a table with no rows is not written."""
    os.makedirs(out_dir, exist_ok=True)
    T = {name: [] for name in TABLES}
    for band in TABLE_BANDS:
        for g in C.GROUPS:
            for instr in C.CONDITIONS:
                S = _cell_items(scores, g, instr)
                if not S:
                    continue
                idx = boot_indices(len(S), C.N_BOOT, C.BOOTSTRAP_SEED)
                T["rates"] += _rate_block(S, g, instr, idx, band)
        if band == C.CARRIER_BAND:
            # the carrier band carries lens rows only
            continue
        pooled_rates, pooled_judged = _pooled_headline_block(scores, band)
        T["rates"] += pooled_rates
        T["contrasts"] += _reader_contrast_block(scores, band)
        T["modulation"] += _modulation_block(scores, band)
        if judged_ran(scores):
            T["judged"] += _judged_block(scores, band)
            for instr in C.CONDITIONS:
                T["judged"] += _judged_block(scores, band, instr)
            T["judged"] += pooled_judged
    write_tables(T, out_dir)
    return T


# --- coverage and costs (methodology, guards and costs) --------------------------------------------------


def judge_logs():
    """judge name -> instrument -> log path, for every judge and every instrument this package sends."""
    return {
        name: {inst: judge_log_rel(name, inst) for inst in ("naming", "summary")}
        for name in C.JUDGES
    }


def unasked_cells(run):
    """{log: {kind: n}}: the cells whose last record was never asked (ledger, transport or auth)."""
    out = {}
    for insts in judge_logs().values():
        for rel in insts.values():
            if not run.exists(rel):
                continue
            last = {}
            for r in run.read_jsonl(rel):
                last[(r.get("key"), json.dumps(r.get("meta"), sort_keys=True, ensure_ascii=False))] = r
            counts = unasked(last.values())
            if sum(counts.values()):
                out[rel] = counts
    return out


def refuse_unasked_logs(run):
    """Raise when any request log holds a cell that was never asked."""
    left = unasked_cells(run)
    if not left:
        return
    detail = "; ".join(f"{rel}: {unasked_detail(counts)}" for rel, counts in sorted(left.items()))
    n = sum(sum(counts.values()) for counts in left.values())
    raise RuntimeError(
        f"report: {n} request(s) in the judge logs were never asked -- {detail}. Run the stage that writes "
        "them again (with a larger --judge-budget-usd if the cap is what stopped it, under a key the "
        "endpoint accepts if the key is); only the unasked requests are asked."
    )


def _spend_from_logs(run, *rels, model=None):
    """Requests, tokens and spend over request logs, once per unique request key (a request shared by two
    cells is paid once and recorded twice); `empty_readout` rows are excluded."""
    priced, records = {}, 0
    for rel in rels:
        for r in run.read_jsonl(rel):
            if r.get("status") == "empty_readout":
                continue
            records += 1
            key, prev = r.get("key"), priced.get(r.get("key"))
            if prev is None or (not prev.get("usage") and r.get("usage")):
                priced[key] = r
    tin = tout = treason = 0
    spend = recorded = reported = 0.0
    for r in priced.values():
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
        "requests": len(priced),
        "records": records,
        "duplicate_records": records - len(priced),
        "input_tokens": tin,
        "output_tokens": tout,
        "spend_usd": spend,
        "spend_usd_recorded": recorded,
        "reasoning_tokens": treason,
        "spend_usd_provider_reported": reported,
    }


# The statuses that partition a log; `content_filter` (the provider) is not `refused` (the judge).
MISSINGNESS_STATUSES = ("ok", "refused", "content_filter", "parse_fail", "unavailable", "empty_readout")
INSTRUMENT_ARM_FIELD = {"naming": "condition", "summary": None}


def missingness(run):
    """One row per judge × instrument × arm: requests made and how each settled."""
    out = []
    for name, insts in judge_logs().items():
        for inst, rel in insts.items():
            if not run.exists(rel):
                continue
            field = INSTRUMENT_ARM_FIELD[inst]
            by_arm = {}
            for r in run.read_jsonl(rel):
                arm = (r.get("meta") or {}).get(field) if field else ""
                rec = by_arm.setdefault(arm or "", {k: 0 for k in MISSINGNESS_STATUSES})
                rec["n"] = rec.get("n", 0) + 1
                status = r.get("status")
                rec[status if status in MISSINGNESS_STATUSES else "unavailable"] += 1
            for arm, rec in sorted(by_arm.items()):
                out.append(dict({"judge": name, "model": C.JUDGES[name].model, "instrument": inst, "arm": arm}, **rec))
    return out


def _record_counts(run, rel, naming=False):
    """The status tally of one judge log; for the naming log also the voided verdicts."""
    statuses = MISSINGNESS_STATUSES
    out = {"requests": 0, **{k: 0 for k in statuses}, "other": 0, "unasked": 0}
    n_voided = 0
    for r in run.read_jsonl(rel):
        out["requests"] += 1
        status = r.get("status")
        out[status if status in statuses else "other"] += 1
        out["unasked"] += int(r.get("error_kind") in UNASKED_KINDS)
        if naming and status == "ok":
            n_voided += bool((verdict_of(r) or {}).get("voided"))
    if naming:
        out["voided"] = n_voided
    return out


def pass_statuses(run, rel):
    """One naming log's replies per cell (`one_record_per_cell`), and for the ones that never became a verdict
    the provider's `finish` reason (or the empty readout's own reason)."""
    by_cell = {}
    for r in run.read_jsonl(rel):
        m = r.get("meta") or {}
        by_cell.setdefault((m.get("i"),) + cell_value(m, LOG_FIELD["naming"]), []).append(r)
    read = [r for recs in by_cell.values() for r in one_record_per_cell(recs, resolved)]
    bad = [r for r in read if not resolved(r)]
    by_finish = {}
    for r in bad:
        f = (r.get("reason") or "empty_readout") if r.get("status") == "empty_readout" else (r.get("usage") or {}).get("finish") or "none"
        by_finish[f] = by_finish.get(f, 0) + 1
    return {"records": len(read), "ok": len(read) - len(bad), "unresolved": len(bad), "by_finish": by_finish}


def foils_block(run):
    """The foil every item was judged against, and the items whose first donor was passed over."""
    for name in C.JUDGES:
        rel = f"judges/{name}/foils.json"
        if run.exists(rel):
            doc = run.read_json(rel)
            foils = doc.get("foils") or {}
            return {
                "rule": doc.get("foil_rule"),
                "n_items": len(foils),
                "skipped": [
                    {"i": int(i), "concept_key": f["concept_key"], "foil_key": f["foil_key"], "donors_skipped": f["donors_skipped"]}
                    for i, f in sorted(foils.items(), key=lambda kv: int(kv[0]))
                    if f.get("donors_skipped")
                ],
            }
    return {"rule": None, "n_items": 0, "skipped": []}


def _norm_outliers(run):
    """capture's per-cell norm flags, summed."""
    if not run.exists("activations/norms.json"):
        return "unavailable"
    norms = run.read_json("activations/norms.json")
    flagged = items = total = 0
    for rec in norms.values():
        cells = (rec or {}).get("cells") or {}
        n = sum(1 for c in cells.values() if c.get("outlier"))
        flagged += n
        total += len(cells)
        items += bool(n)
    return {"cells_flagged": flagged, "cells_total": total, "items_with_flag": items}


def _ledger_totals(run, rel=None):
    """The ledger's recorded spend and cap."""
    rel = rel or C.JUDGE_LEDGER_REL
    if not run.exists(rel):
        return {"spent_usd": None, "cap_usd": None, "requests": None}
    l = run.read_json(rel)
    return {"spent_usd": l.get("spent_usd"), "cap_usd": l.get("cap_usd"), "requests": l.get("requests")}


def _reader_coverage(run, scores):
    """(the NLA stage's guards, how many items each reader covered at each band)."""
    guards = "unavailable"
    if run.exists("rollouts/nla.json"):
        guards = (run.read_json("rollouts/nla.json").get("config") or {}).get("guards", "unavailable")
    cov = {}
    for band in C.READ_BANDS:
        cov[band] = {
            "maem_reg": sum(1 for s in scores if (_band(s, band).get("maem") or {}).get(C.HEADLINE_ARM)),
            "maem_null": sum(1 for s in scores if (_band(s, band).get("maem") or {}).get(C.NULL_ARM)),
            C.BASE_ARM: sum(1 for s in scores if (_band(s, band).get("maem") or {}).get(C.BASE_ARM)),
            C.RETRIEVAL_READER: sum(1 for s in scores if _band(s, band).get("retrieval")),
            **{a: sum(1 for s in scores if (_band(s, band).get("patch") or {}).get(a)) for a in C.PATCH_ARMS},
            "jlens_L42": sum(1 for s in scores if _lens(s, band, "has_record")),
            **{c: sum(1 for s in scores if (_band(s, band).get("nla") or {}).get(c)) for c in C.NLA_READERS},
            "judged": {
                n: {c: sum(1 for s in scores if _fold(s, band, n, c)) for c in C.judged_readers_at(band)}
                for n in C.JUDGES
            },
        }
    cov[C.CARRIER_BAND] = {
        "jlens_L42": sum(1 for s in scores if _lens(s, C.CARRIER_BAND, "has_record")),
        "cells": sum(s["n_cells"] for s in scores),
    }
    return guards, cov


PATCH_DIAGNOSTIC_FIELDS = (
    "target_layer", "rule", "seed", "patch_check", "clean_norm_at_layer", "unpatched_greedy",
    "greedy_distinct_share", "share_equal_unpatched", "distinct_sample_texts", "seconds",
)


def patchscope_diagnostics(run):
    """The Patchscopes stage's own record per arm (prompt, rule, patch check, distinctness), or "unavailable"."""
    if not run.exists(C.PATCH_REL):
        return "unavailable"
    doc = run.read_json(C.PATCH_REL)
    top = doc.get("config") or {}
    tmpl = top.get("template") or {}
    out = {
        "prompt_id": tmpl.get("prompt_id"),
        "placeholder": tmpl.get("placeholder"),
        **{k: top.get(k) for k in ("alpha", "rule", "input", "layer", "mu_sha256")},
        "arms": {},
    }
    for arm in C.PATCH_ARMS:
        a = (doc.get("arms") or {}).get(arm)
        if a is None:
            continue
        cfg = a.get("config") or {}
        ratios = [
            float(c["norm_ratio"]) for it in a.get("items") or [] for c in it.get("cells") or []
            if c.get("norm_ratio") is not None
        ]
        out["arms"][arm] = {
            **{k: cfg.get(k) for k in PATCH_DIAGNOSTIC_FIELDS},
            "n_items": len(a.get("items") or []),
            "n_cells": sum(len(it.get("cells") or []) for it in a.get("items") or []),
            "norm_ratio_min": min(ratios) if ratios else None,
            "norm_ratio_max": max(ratios) if ratios else None,
        }
    return out


def _reread_norm_filter(run):
    """What the scorer's norm filter would have dropped from the headline arm's re-read greedies."""
    rel = f"rollouts/{C.HEADLINE_ARM}.json"
    if not run.exists(rel):
        return {"rollouts": "unavailable"}
    return {"rollouts": (run.read_json(rel).get("config") or {}).get("norm_filter") or "unavailable"}


def _compliance(scores):
    """The compliance diagnostic: items whose user turn alone, on the base, copies the carrier or names a target."""
    read = [s for s in scores if s.get("compliance")]
    return {
        "n": len(read),
        "copies_carrier": sum(1 for s in read if s["compliance"]["copies_carrier"]),
        "names_target": sum(1 for s in read if s["compliance"]["names_target"]),
    }


def coverage_and_costs(run, scores):
    doc = run.read_json("data/items.json")
    cfg = doc.get("config") or {}
    prov = read_provenance(run)
    stages = {}
    for p in sorted(glob.glob(run.file("stages/*.json"))):
        name = os.path.basename(p)[:-5]
        # the render's own record is left out, so re-renders are byte-identical
        if name == "report" or not is_stage_record(name):
            continue
        with open(p, encoding="utf-8") as h:
            stages[name] = json.load(h)
    logs = judge_logs()
    spend = {
        name: {
            inst: _spend_from_logs(run, rel, model=C.JUDGES[name].model)
            for inst, rel in insts.items()
            if run.exists(rel)
        }
        for name, insts in logs.items()
    }
    ledger = _ledger_totals(run)
    design_recorded = sum(b["spend_usd_recorded"] for j in spend.values() for b in j.values())
    nla_guards, coverage = _reader_coverage(run, scores)
    total_spend = sum(b["spend_usd"] for j in spend.values() for b in j.values())
    total_requests = sum(b["requests"] for j in spend.values() for b in j.values())
    total_records = sum(b["records"] for j in spend.values() for b in j.values())
    # summed over provenance/modal*.json launch records: a lower bound
    launches = launch_records(run)
    gpu = gpu_cost_block(prov.get("gpu_type"), launch_gpu_seconds(launches), [], *launch_gpu_rate(launches, prov))
    stage_seconds = {k: v.get("seconds") for k, v in stages.items()}
    elapsed = elapsed_block(
        sum(float(v) for k, v in stage_seconds.items() if isinstance(v, (int, float))),
        [k for k, v in stage_seconds.items() if not isinstance(v, (int, float))],
    )
    return {
        "population": {
            "kept": len(scores),
            "n_concepts": len({s["concept_key"] for s in scores}),
            "n_carriers": len({s["carrier"] for s in scores}),
            "by_group": {g: sum(1 for s in scores if _in_group(s, g)) for g in C.GROUPS},
            "by_instruction": {
                instr: sum(1 for s in scores if s["instruction"] == instr) for instr in C.CONDITIONS
            },
            "counts": cfg.get("counts"),
            "excluded": cfg.get("exclusions"),
            "excluded_concepts": cfg.get("excluded_concepts"),
            "reassigned": cfg.get("reassigned"),
            "n_cells": cfg.get("n_cells"),
            "single_token_lens_targets": sum(1 for s in scores if s["single_token"]),
            "no_single_token_form": sum(1 for s in scores if not s["single_token"]),
            "n_donors_used": {
                "drawn": C.N_DONORS,
                "min": min((s["n_donors_used"] for s in scores), default=None),
                "mean": (sum(s["n_donors_used"] for s in scores) / len(scores) if scores else float("nan")),
            },
            "norm_outliers": _norm_outliers(run),
            "compliance": _compliance(scores),
        },
        "readout_coverage": coverage,
        "judge": {
            name: dict(
                spend[name],
                model=C.JUDGES[name].model,
                label=C.JUDGES[name].label,
                items=sum(1 for s in scores for b in C.READ_BANDS if (_band(s, b).get("judged") or {}).get(name)),
                gate=(run.read_json(f"judges/{name}/gate.json") if run.exists(f"judges/{name}/gate.json") else "unavailable"),
                records={
                    inst: _record_counts(run, rel, naming=(inst == "naming"))
                    for inst, rel in logs[name].items()
                    if run.exists(rel)
                },
                statuses=(
                    pass_statuses(run, logs[name]["naming"]) if run.exists(logs[name]["naming"]) else "unavailable"
                ),
            )
            for name in C.JUDGES
        },
        "judge_missingness": missingness(run),
        "judge_completed": bool((stages.get("judge") or {}).get("completed")),
        "instrument": {
            "naming_system_sha256": C.PROMPT_SHA256["naming_system"],
            "readout_kind": dict(C.READOUT_KIND),
            "vs": list(C.VS),
            "foil_rule": C.FOIL_RULE,
            "foils": foils_block(run),
        },
        "api_spend_total_usd": total_spend,
        "api_requests_total": total_requests,
        "api_records_total": total_records,
        "api_duplicate_records_total": total_records - total_requests,
        "api_spend_recorded_usd": ledger["spent_usd"],
        "api_spend_recorded_design_usd": design_recorded,
        "budget_cap_usd": ledger["cap_usd"],
        "ledger_requests": ledger["requests"],
        "nla_guards": nla_guards,
        "retrieval": prov.get("retrieval", "unavailable"),
        "patchscope": patchscope_diagnostics(run),
        "reread_norm_filter": _reread_norm_filter(run),
        "injection_check": (prov.get("rollouts_arms") or {}).get(C.HEADLINE_ARM, {}).get("injection_check", "unavailable"),
        "control_check": (prov.get("rollouts_arms") or {}).get(C.NULL_ARM, {}).get("guards", "unavailable"),
        "lens_self_check": prov.get("lens_self_check", "unavailable"),
        "mean_cell_norms": prov.get("mean_cell_norms", "unavailable"),
        "stages": stages,
        "stage_seconds": stage_seconds,
        **elapsed,
        **gpu,
        "gpu_timing_source": prov.get("gpu_timing_source",
                                      "measured" if gpu["gpu_seconds_measured"] != "unavailable" else "unavailable"),
        "billing_usd": prov.get("billing_usd", "unavailable"),
        "cost_usd_list_price": {
            **{f"{name}/{inst}": b["spend_usd"] for name, j in spend.items() for inst, b in j.items()},
            "total_api": total_spend,
            "gpu_estimated": gpu["gpu_cost_usd_estimated"],
        },
        "rates_per_m": {k: list(v) for k, v in C.RATES_PER_M.items()},
        "parse_version": C.PARSE_VERSION,
        "scoring_version": C.SCORING_VERSION,
    }
