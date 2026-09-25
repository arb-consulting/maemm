"""The naming instrument and the `judge` stage (methodology "The naming instrument").

A judge is shown a concept's target forms and one reader's readout at one read position, and nothing else,
and answers whether a target is named, with a verbatim quote (eval/common/naming.py). A positive verdict
whose target was not asked about, or whose quote is not in the readout, is voided. Every cell is asked twice,
against the item's own targets and against a foil concept's (`foil_of`); net = named − foil.

Each judge first clears a gate on a stratified head of its requests (`gate_head`, `gate_report`), then
answers the rest. The judges and the summariser spend from one ledger under one cap. Logs are
judges/<judge>/naming.jsonl, append-only and resumed by request key."""

import json
import time

from eval.common.judge_client import (
    Ledger,
    UNASKED_KINDS,
    client_for,
    judge_log,
    key_for,
    looks_like_refusal,
    request_key,
    run_requests,
    unasked,
    unasked_detail,
    with_judge,
)
from eval.common.naming import KINDS as NAMING_KINDS
from eval.common.naming import naming_request as _naming_request
from eval.common.naming import parse_verdict, void_reason
from eval.common.runs import mark_stage, stage_done
from eval.workspace_modulation import config as C, items as I
from eval.workspace_modulation import mean_cell as M
from eval.workspace_modulation import patchscope as PS
from eval.workspace_modulation import retrieval as RET
from eval.workspace_modulation import summarise as SUM
from eval.workspace_modulation.rollouts import cells_by_pos, load_arm
from eval.workspace_modulation.runs import resolve_judge_budget, stage_key

UPSTREAM = ("rollouts_merge", "nla_merge", "retrieval_merge", "summarise", "patchscope")
TITLE = "maemm-workspace-modulation"  # the X-Title OpenRouter requests carry
JUDGED_ARMS = ("reg", C.NULL_ARM)  # the untrained-base ablation is read by the word rule alone
INSTRUMENT = "naming"  # the log name: judges/<judge>/naming.jsonl


def load_ledger(run, cap):
    """The one ledger the summariser and the judges spend from."""
    ledger = Ledger.load(run.file(C.JUDGE_LEDGER_REL), cap, {s.model: s.rates for s in C.JUDGES.values()})
    ledger.save(run.file(C.JUDGE_LEDGER_REL))
    return ledger


# --- the foil -----------------------------------------------------------------------------------------


def operands(item):
    """The literal numbers of an arithmetic item's expression (the population's own rule, `items._numbers`)."""
    return I._numbers(item["concept"]) if item["family"] == "arithmetic" else set()


def foil_of(item, concepts):
    """(foil concept key, donors passed over): the first of the item's donors whose forms are not operands
    of its own expression. A readout doing the sum states its operands, so such a donor would be "named" by
    copying the expression. One foil per concept, shared by every cell, reader and judge; raises when every
    donor is an operand."""
    ops = operands(item)
    skipped = []
    for key in item["donors"]:
        forms = concepts[str(key)]["forms"]
        if ops and any(f in ops for f in forms):
            skipped.append(str(key))
            continue
        return str(key), skipped
    raise RuntimeError(
        f"item {item['i']} ({item['concept']}): every donor's answer is an operand of the expression "
        f"({sorted(ops)}), so the instrument has no foil to ask about"
    )


def item_targets(item, concepts):
    """The target forms the judge is asked about for an item."""
    return list(item.get("targets") or concepts[str(item["concept_key"])]["targets"])


def naming_request(item, targets, condition, vs, samples, meta, spec=None):
    """One naming request with this package's meta. `readout` is the joined text a logged verdict is bound
    to (`analysis.bind_log`). Unaddressed unless `spec` is given."""
    base = _naming_request(targets, C.READOUT_KIND[condition], samples)
    req = (
        with_judge(base, spec)
        if spec
        else dict(base, model=None, max_tokens=C.MAX_TOKENS[base["kind"]], provider=None, sampling=None, transport=None)
    )
    return dict(req, meta=dict(meta, condition=condition, vs=vs), readout="\n".join(samples))


# --- reading a reply ------------------------------------------------------------------------------------


def raw_verdict_of(rec):
    """The shared parse of an ok record, before the voiding rule; None for a non-ok record."""
    if (rec or {}).get("status") != "ok":
        return None
    return parse_verdict(rec.get("text") or "")


def verdict_of(rec):
    """The verdict one saved record carries: the shared parse, then the voiding rule. `named` is expressed
    and not voided; None when the record never came back ok. The one place the three steps are spelled."""
    v = raw_verdict_of(rec)
    if v is None:
        return None
    if not v["parse_ok"]:
        return dict(v, voided=False, void_reason=None, named=None)
    reason = void_reason(v, rec.get("targets") or [], rec.get("samples") or [])
    return dict(v, voided=reason is not None, void_reason=reason, named=bool(v["expressed"]) and reason is None)


def response_status(kind, text, finish):
    """ok / unavailable / refused / parse_fail / content_filter for one reply. The provider's content filter
    is read first: a filtered reply is empty, and filing it as `unavailable` would re-send it."""
    if finish == "content_filter":
        return "content_filter"
    if not (text or "").strip():
        return "unavailable"
    if kind == "summary_req":
        return "ok"
    if kind in NAMING_KINDS:
        ok = parse_verdict(text)["parse_ok"]
        return "ok" if ok else ("refused" if looks_like_refusal(text) else "parse_fail")
    raise ValueError(f"unknown request kind {kind!r}")


# --- the readouts ---------------------------------------------------------------------------------------


def readouts_for_cell(
    cell_reg, cell_null, cell_nla, summary, cell_ret, band_summary=None, band=C.FINAL_BAND, cell_patch=None,
):
    """[(judged reader, texts shown)] at one read position, in JUDGED_READERS order. A sampled reader's
    readout is its samples (the search: its ranked windows); a lens reader's is its one summary, empty when
    the summariser returned none."""
    texts = lambda cell: [s["text"] for s in (cell or {}).get("samples") or []]
    out = {
        "maemm_reg8": texts(cell_reg) if cell_reg else None,
        "maemm_null8": texts(cell_null) if cell_null else None,
        "nla_n8": texts(cell_nla) if cell_nla else None,
        C.RETRIEVAL_JUDGED: texts(cell_ret) if cell_ret else None,
        C.PATCH_JUDGED[C.PATCH_ARM]: texts(cell_patch) if cell_patch else None,
        C.LENS_READER: [summary] if summary else [],
        C.LENS_BAND_READER: [band_summary] if band_summary else [],
    }
    return [(cond, out[cond]) for cond in C.judged_readers_at(band) if out[cond] is not None]


def control_pos(it, p):
    """The cell the null control's readout for `p` is read from: its final period at the mean cell (it reads
    no activation, so the request repeats one already answered)."""
    return M.final_pos(it) if p == C.MEAN_POS else p


def naming_requests(
    kept, reg, null, nla, retrieval, summaries, concepts, spec=None, band_summaries=None, patch=None
):
    """(requests, empty, foils): every judged reader at both read positions of every kept item, asked against
    its own targets and the foil's. An empty readout is never sent: it goes to `empty` with its reason."""
    cells = (summaries or {}).get("cells") or {}
    band_cells = (band_summaries or {}).get("cells") or {}
    reqs, empty, foils = [], [], {}
    for it in kept:
        foil_key, skipped = foil_of(it, concepts)
        targets = {"own": item_targets(it, concepts), "foil": list(concepts[foil_key]["targets"])}
        foils[it["i"]] = {
            "concept_key": str(it["concept_key"]),
            "foil_key": foil_key,
            "targets_own": targets["own"],
            "targets_foil": targets["foil"],
            "donors_skipped": skipped,
        }
        reg_cells, null_cells, nla_cells, ret_cells = (cells_by_pos(d, it["i"]) for d in (reg, null, nla, retrieval))
        patch_cells = cells_by_pos(((patch or {}).get("arms") or {}).get(C.PATCH_ARM), it["i"])
        for band in C.READ_BANDS:
            p = M.band_cells(band, it)[0]
            summary = (cells.get(f"{it['i']}/{p}") or {}).get("summary")
            band_summary = (band_cells.get(f"{it['i']}/{p}") or {}).get("summary")
            row = readouts_for_cell(
                reg_cells.get(p),
                null_cells.get(control_pos(it, p)),
                nla_cells.get(p),
                summary,
                ret_cells.get(p),
                band_summary,
                band,
                patch_cells.get(p),
            )
            lens_text = {C.LENS_READER: summary, C.LENS_BAND_READER: band_summary}
            for cond, texts in row:
                for vs in C.VS:
                    r = naming_request(it, targets[vs], cond, vs, texts, {"i": it["i"], "pos": p}, spec=spec)
                    if r["readout"].strip():
                        reqs.append(r)
                    else:
                        no_summary = cond in C.LENS_READERS and lens_text[cond] is None
                        r["reason"] = "no_summary" if no_summary else "empty_readout"
                        empty.append(r)
    return reqs, empty, foils


def meta_id(meta):
    """eval.common.judge_client's meta identity."""
    return json.dumps(meta, sort_keys=True, ensure_ascii=False)


def _empty_record(req):
    return {
        "key": request_key(req),
        "model": req["model"],
        "meta": req.get("meta"),
        "kind": req["kind"],
        "targets": req.get("targets"),
        "samples": req.get("samples"),
        "readout": req.get("readout"),
        "status": "empty_readout",
        "reason": req.get("reason", "empty_readout"),
        "text": None,
        "cost_usd": 0.0,
    }


def record_empty_readouts(run, rel, reqs):
    """Log each empty readout as an `empty_readout` record (never sent, counted in no judged rate).
    Deduplicated on (key, meta): two empty cells of one item render the identical request."""
    seen = {(r["key"], meta_id(r.get("meta"))) for r in run.read_jsonl(rel)}
    n = 0
    for req in reqs:
        rec = _empty_record(req)
        mid = (rec["key"], meta_id(rec["meta"]))
        if mid in seen:
            continue
        run.append_jsonl(rel, rec)
        seen.add(mid)
        n += 1
    return n


# --- the gate -------------------------------------------------------------------------------------------


def gate_head(reqs, kept, n):
    """The `n` requests a gate is measured on: round-robin over (family, reader), so the gate sees both
    families, every reader, and own and foil questions."""
    family = {it["i"]: it["family"] for it in kept}
    strata = {}
    for r in reqs:
        strata.setdefault((family.get(r["meta"]["i"]), r["meta"]["condition"]), []).append(r)
    out, rows = [], list(strata.values())
    for k in range(max((len(v) for v in rows), default=0)):
        for row in rows:
            if k < len(row) and len(out) < n:
                out.append(row[k])
    return out


def gate_forecast(run, rel, pass_reqs, per_request_usd, ledger):
    """(requests still unanswered, their cost at the measured per-request rate, what is left of the cap)."""
    answered = {r.get("key") for r in run.read_jsonl(rel)} if run.exists(rel) else set()
    remaining = sum(1 for r in pass_reqs if request_key(r) not in answered)
    return remaining, per_request_usd * remaining, ledger.cap - float(ledger.state()["spent_usd"])


def gate_report(run, spec, reqs, out, ledger, rel, pass_reqs):
    """The gate on the head of one judge's requests: the parse rate (≥ GATE_MIN_PARSE), the share of positive
    verdicts whose quote verifies (≥ GATE_MIN_QUOTE, gated from GATE_MIN_POSITIVES positives) and a
    forecast of the rest of the pass against the cap. Writes judges/<judge>/gate.json; raises on a fail."""
    head = [out[request_key(r)] for r in reqs if request_key(r) in out]
    settled = [r for r in head if r.get("usage")]
    spend = sum(float(r.get("cost_usd") or 0.0) for r in settled)
    per = spend / len(settled) if settled else float("nan")
    raw = [(out[request_key(r)], raw_verdict_of(out[request_key(r)])) for r in reqs]
    parsed = [v for _rec, v in raw if v is not None and v["parse_ok"]]
    positives = [(rec, v) for rec, v in raw if v is not None and v["parse_ok"] and v["expressed"]]
    verified = sum(
        1 for rec, v in positives if void_reason(v, rec.get("targets") or [], rec.get("samples") or []) is None
    )
    parse_rate = len(parsed) / len(reqs) if reqs else float("nan")
    quote_rate = verified / len(positives) if positives else float("nan")
    gated_quote = len(positives) >= C.GATE_MIN_POSITIVES
    remaining, forecast, cap_left = gate_forecast(run, rel, pass_reqs, per, ledger)
    affordable = not (forecast > cap_left)  # nan (nothing settled) compares false
    usage = lambda r: r["usage"] or {}
    rep = {
        "judge": spec.name,
        "model": spec.model,
        "log": rel,
        "gate_requests": len(reqs),
        "settled": len(settled),
        "input_tokens": sum(int(usage(r).get("input_tokens") or 0) for r in settled),
        "output_tokens": sum(int(usage(r).get("output_tokens") or 0) for r in settled),
        "reasoning_tokens": sum(int(usage(r).get("reasoning_tokens") or 0) for r in settled),
        "finish_reasons": {
            f: sum(1 for r in settled if usage(r).get("finish") == f)
            for f in sorted({usage(r).get("finish") for r in settled}, key=str)
        },
        "spend_usd": spend,
        "provider_cost_usd": sum(float(usage(r).get("cost_reported") or 0.0) for r in settled),
        "per_request_usd": per,
        "cap_usd": ledger.cap,
        "requests_remaining": remaining,
        "forecast_usd": forecast,
        "cap_remaining_usd": cap_left,
        "forecast_affordable": affordable,
        "parse_rate": parse_rate,
        "parse_ok": len(parsed),
        "n_positive": len(positives),
        "quote_verified": verified,
        "quote_rate": quote_rate,
        "quote_rate_gated": gated_quote,
        "thresholds": {
            "parse_rate": C.GATE_MIN_PARSE,
            "quote_rate": C.GATE_MIN_QUOTE,
            "min_positives": C.GATE_MIN_POSITIVES,
        },
    }
    rep["passed"] = bool(
        len(reqs)
        and parse_rate >= C.GATE_MIN_PARSE
        and (quote_rate >= C.GATE_MIN_QUOTE if gated_quote else True)
        and affordable
    )
    run.write_json(f"judges/{spec.name}/gate.json", rep)
    print(
        f"[judge:{spec.name}] GATE {len(settled)} requests: ${spend:.4f}, ${per:.5f} each; "
        f"parse {len(parsed)}/{len(reqs)} = {parse_rate:.2f} (need >= {C.GATE_MIN_PARSE}); "
        f"quote {verified}/{len(positives)}"
        + (
            f" = {quote_rate:.2f} (need >= {C.GATE_MIN_QUOTE})"
            if gated_quote
            else f" (not gated, < {C.GATE_MIN_POSITIVES} positives)"
        )
        + f"; {remaining} naming requests left, forecast ${forecast:.2f} of ${cap_left:.2f} still "
        f"on the cap; finish {rep['finish_reasons']}, reasoning tokens {rep['reasoning_tokens']}; "
        f"GATE {'PASS' if rep['passed'] else 'FAIL'}; not marked done",
        flush=True,
    )
    if not rep["passed"]:
        raise RuntimeError(
            f"judge {spec.name}: the gate FAILED (parse {parse_rate:.2f} of >= {C.GATE_MIN_PARSE}, "
            f"quote {quote_rate:.2f} of >= {C.GATE_MIN_QUOTE} over {len(positives)} positives, forecast "
            f"${forecast:.2f} of ${cap_left:.2f} left on the cap); the full pass is NOT run and the stage "
            "is NOT marked done."
        )
    return rep


# --- truncation re-ask ----------------------------------------------------------------------------------

RETRY_FIELDS = ("condition", "vs")  # with the item and position, what names one naming cell


def cell_of(rec_or_req):
    """(item, read position, condition, vs) of one naming record or request."""
    m = (rec_or_req or {}).get("meta") or {}
    return (m.get("i"), m.get("pos")) + tuple(m.get(f) for f in RETRY_FIELDS)


def resolves(rec):
    """Whether one saved record carries a naming verdict."""
    v = verdict_of(rec)
    return v is not None and bool(v["parse_ok"])


def truncated_cells(run, rel):
    """The cells whose records ended `finish_reason: length` and none of which yielded a verdict: the only
    failures a larger cap can fix (a content-filtered reply is not re-asked)."""
    by = {}
    for r in run.read_jsonl(rel):
        by.setdefault(cell_of(r), []).append(r)
    out = []
    for cell, recs in by.items():
        if any(resolves(r) for r in recs):
            continue
        if any((r.get("usage") or {}).get("finish") == "length" for r in recs):
            out.append(cell)
    return sorted(out, key=repr)


def retry_truncated(args, run, spec, built, max_tokens):
    """Re-ask one judge's truncated naming cells at `max_tokens`, and nothing else. The cap is part of the
    request key, so a retry is a new key beside the original. Marks nothing done."""
    ledger = load_ledger(run, resolve_judge_budget(args))
    client = client_for(spec, key_for(spec), url=C.OPENROUTER_URL, title=TITLE)
    rel = judge_log(spec.name, INSTRUMENT)
    want = truncated_cells(run, rel) if run.exists(rel) else []
    if not want:
        print(f"[judge:{spec.name}] retry: no truncated cell left to re-ask", flush=True)
        return []
    at = {cell_of(r): r for r in built}
    absent = [c for c in want if c not in at]
    if absent:
        raise RuntimeError(f"judge {spec.name}: retry cannot rebuild {absent} — they are not in this pass's scope")
    own_caps = sorted({spec.max_tokens_for(at[c]["kind"]) for c in want})
    if int(max_tokens) in own_caps:
        raise RuntimeError(
            f"judge {spec.name}: the retry cap equals the pass's own cap ({max_tokens}); send it at a different cap"
        )
    before = {r["key"] for r in run.read_jsonl(rel)}
    reqs = [dict(at[c], max_tokens=int(max_tokens)) for c in want]
    already = [r for r in reqs if request_key(r) in before]
    reqs = [r for r in reqs if request_key(r) not in before]
    if already:
        print(f"[judge:{spec.name}] retry: {len(already)} cell(s) were already re-asked at {max_tokens} and stay "
              "unresolved; not re-sent", flush=True)
    if not reqs:
        return []
    print(f"[judge:{spec.name}] retry: re-asking {len(reqs)} truncated cell(s) at max_tokens {max_tokens} "
          f"(was {own_caps})", flush=True)
    out = ask(run, rel, reqs, client, ledger)
    after = run.read_jsonl(rel)
    ok = sum(1 for r in reqs if resolves(out[request_key(r)]))
    if len({r["key"] for r in after}) != len(before) + len(reqs):
        raise RuntimeError(
            f"judge {spec.name}: the retry added {len({r['key'] for r in after}) - len(before)} keys, expected "
            f"exactly {len(reqs)} — a completed key was re-asked or a retry collided with one"
        )
    print(f"[judge:{spec.name}] retry done: {len(reqs)} re-asked, {ok} now yield a verdict, "
          f"${ledger.state()['spent_usd']:.3f} of ${ledger.cap:.2f}; nothing marked done", flush=True)
    return reqs


# --- the stage ------------------------------------------------------------------------------------------


def _check_upstream(args, run):
    """Every stage whose readouts are judged, held to its own chained key, before anything is spent."""
    for up in UPSTREAM:
        want = stage_key(up, args, run)
        if not stage_done(run, up, want):
            got = (run.read_json(f"stages/{up}.json") if run.exists(f"stages/{up}.json") else {}).get("config_hash")
            raise RuntimeError(
                f"judge: stage {up} has not completed for this configuration; run it first "
                f"(its record says {got!r}; this pass accepts {want!r})"
            )


def _load_inputs(run):
    items_doc = run.read_json("data/items.json")
    all_items = items_doc["items"]
    arms = {a: load_arm(run, a) for a in JUDGED_ARMS}
    return {
        "all_items": all_items,
        "concepts": items_doc["concepts"],
        "kept": [x for x in all_items if not x["excluded"]],
        "reg": arms[C.HEADLINE_ARM],
        "null": arms[C.NULL_ARM],
        "nla": run.read_json("rollouts/nla.json"),
        "retrieval": run.read_json(RET.MERGED_REL),
        "summaries": run.read_json(SUM.SUMMARIES_REL),
        "band_summaries": run.read_json(SUM.BAND_REL),
        "patch": PS.load_patchscope(run),
    }


def build_requests(inputs):
    """{naming, empty, foils}: every request the stage sends, unaddressed; each judge gets the same list
    through `with_judge`."""
    reqs, empty, foils = naming_requests(
        inputs["kept"],
        inputs["reg"],
        inputs["null"],
        inputs["nla"],
        inputs["retrieval"],
        inputs["summaries"],
        inputs["concepts"],
        band_summaries=inputs.get("band_summaries"),
        patch=inputs.get("patch"),
    )
    return {INSTRUMENT: reqs, "empty": empty, "foils": foils}


def for_judge(built, spec):
    """`built` with every request list addressed to one judge."""
    return {k: ([with_judge(r, spec) for r in v] if k in (INSTRUMENT, "empty") else v) for k, v in built.items()}


def ask(run, rel, reqs, client, ledger):
    """`run_requests` at this package's concurrency and status rule; the ledger is saved however it ends."""
    try:
        return run_requests(run, rel, reqs, client, ledger, workers=C.JUDGE_CONCURRENCY, status_fn=response_status)
    finally:
        ledger.save(run.file(C.JUDGE_LEDGER_REL))


def stage_judge(args, run):
    """Every judge over the naming instrument on one ledger: the gate, then the pass. Marked done only when
    no request was left unasked. `--gate N` runs the gate alone; `--retry-truncated T` re-asks cut replies."""
    gate = int(getattr(args, "gate", 0) or 0)
    retry = int(getattr(args, "retry_truncated", 0) or 0)
    chash = stage_key("judge", args, run)
    if not (gate or retry) and stage_done(run, "judge", chash) and not args.force:
        _check_upstream(args, run)
        print("[judge] up to date")
        return
    _check_upstream(args, run)
    ledger = load_ledger(run, resolve_judge_budget(args))
    started, record = time.time(), {}
    inputs = _load_inputs(run)
    requests = build_requests(inputs)
    for spec in C.JUDGES.values():
        built = for_judge(requests, spec)
        client = client_for(spec, key_for(spec), url=C.OPENROUTER_URL, title=TITLE)
        if retry:
            retry_truncated(args, run, spec, built[INSTRUMENT], retry)
            continue
        rel = judge_log(spec.name, INSTRUMENT)
        n_empty = record_empty_readouts(run, rel, built["empty"])
        run.write_json(
            f"judges/{spec.name}/foils.json",
            {
                "model": spec.model,
                "foil_rule": C.FOIL_RULE,
                "donor_seed": C.DONOR_SEED,
                "foils": {str(i): f for i, f in built["foils"].items()},
            },
        )
        n_skipped_donors = sum(len(f["donors_skipped"]) for f in built["foils"].values())
        print(
            f"[judge:{spec.name}] {len(built[INSTRUMENT])} naming requests (own and foil) over "
            f"{len(inputs['kept'])} items ({len(built['empty'])} empty readouts never sent, {n_empty} rows newly "
            f"recorded); {n_skipped_donors} operand donor(s) passed over for a foil",
            flush=True,
        )
        head = gate_head(built[INSTRUMENT], inputs["kept"], gate or C.GATE_N)
        out = ask(run, rel, head, client, ledger)
        gate_report(run, spec, head, out, ledger, rel, built[INSTRUMENT])
        if gate:
            continue
        out = ask(run, rel, built[INSTRUMENT], client, ledger)
        unresolved = sum(1 for r in built[INSTRUMENT] if out[request_key(r)]["status"] != "ok")
        unanswered = unasked(out[request_key(r)] for r in built[INSTRUMENT])
        voided = sum(1 for r in built[INSTRUMENT] if (v := verdict_of(out[request_key(r)])) and v.get("voided"))
        record[spec.name] = {
            "model": spec.model,
            INSTRUMENT: len(built[INSTRUMENT]),
            "empty_readout": len(built["empty"]),
            "items": len(inputs["kept"]),
            "donors_skipped": n_skipped_donors,
            "unresolved": unresolved,
            "voided": voided,
            "unasked": unanswered,
        }
        print(
            f"[judge:{spec.name}] done: {unresolved} naming replies yielded no verdict, {voided} voided; "
            f"{unasked_detail(unanswered)}; ${ledger.state()['spent_usd']:.3f} of ${ledger.cap:.2f} in "
            f"{time.time() - started:.0f}s",
            flush=True,
        )
    if gate or retry:
        return
    totals = {kind: sum(r["unasked"][kind] for r in record.values()) for kind in UNASKED_KINDS}
    n_unasked = sum(totals.values())
    spent = ledger.state()["spent_usd"]
    if n_unasked:
        raise RuntimeError(
            f"judge: {n_unasked} requests were never asked ({unasked_detail(totals)}); US${spent:.3f} of "
            f"the US${ledger.cap:.2f} cap is spent and the stage is NOT marked done. Re-run it (with a "
            "larger --judge-budget-usd if the cap is what stopped it): answered keys are not re-asked."
        )
    mark_stage(run, "judge", chash, {"judges": record, "spent_usd": spent}, started=started)
