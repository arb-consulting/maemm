"""Stages `summarise` and `judge` (methodology §5, §9): the lens summaries, the naming instrument with quote
voiding and the pipeline diagnostic, against one budget ledger.

The transport (ledger, client, cached and resumable `run_requests`) is eval/common/judge_client.py's.
Requests are built unaddressed and bound to a judge with `with_judge()`; the judge's model, cap, provider
and sampling are all in `request_key`."""

import json
import time

import numpy as np

from eval.common.judge_client import (
    Ledger,
    judge_log,
    looks_like_refusal,
    request_key,
    unasked,
    unasked_detail,
    with_judge,
)
from eval.common.judge_client import client_for, key_for
from eval.common.judge_client import run_requests as _run_requests
from eval.common.lens_summary import summary_request as _summary_request
from eval.common.naming import naming_request, parse_verdict, void_reason
from eval.workspace_understanding import config as C
from eval.workspace_understanding import readouts as RO
from eval.workspace_understanding import shards as S
from eval.workspace_understanding.lens import pool_band
from eval.common.runs import mark_stage, stage_done
from eval.workspace_understanding.runs import stage_key

LEDGER_REL = "judges/ledger.json"


def _cell(key, meta):
    """The (request key, meta) identity of one cell: one request can be shared by several cells."""
    return key, json.dumps(meta, sort_keys=True, ensure_ascii=False)


# --- requests ---------------------------------------------------------------------------------------------


def summary_request(strs):
    """The summariser's own call (eval/common/lens_summary.py), without a meta: the stage attaches its own."""
    req = _summary_request(strs)
    req.pop("meta")
    return req


def judge_request(targets, kind, samples):
    """One naming question (eval/common/naming.py); `kind` is "samples" (free text) or "summary" (lens prose)."""
    return naming_request(targets, kind, samples)


# --- reading a reply --------------------------------------------------------------------------------------

_CONTENT_FILTER_FINISH = {"content_filter"}


def response_status(kind, text, finish):
    """ok / parse_fail / unavailable / refused / content_filter (methodology §9). A provider filter is not a
    refusal; the summariser's prose (`summary_req`) is only ever ok or unavailable."""
    if finish in _CONTENT_FILTER_FINISH:       # before the empty-text test: a filtered reply is empty
        return "content_filter"
    if not (text or "").strip():
        return "unavailable"
    if kind == "summary_req":
        return "ok"
    if parse_verdict(text)["parse_ok"]:
        return "ok"
    return "refused" if looks_like_refusal(text) else "parse_fail"


# --- transport bound to this package's pins ---------------------------------------------------------------


def judge_client(spec, api_key):
    """eval/common's client for this judge, with this package's OpenRouter URL."""
    return client_for(spec, api_key, url=C.OPENROUTER_URL)


def client_for_judge(spec):
    """This judge's client under its own API key; a missing key refuses before the first request."""
    return judge_client(spec, key_for(spec))


def run_requests(run, rel, reqs, client, ledger, workers=C.JUDGE_CONCURRENCY, status_fn=None):
    """eval/common's run_requests with this package's concurrency and `response_status`; the ledger is
    saved however the pass ends (an auth failure raises after spending)."""
    try:
        return _run_requests(run, rel, reqs, client, ledger, workers=workers, status_fn=status_fn or response_status)
    finally:
        ledger.save(run.file(LEDGER_REL))


def unasked_requests(out, reqs):
    """{"budget": n, "transport": n, "auth": n}: the requests of `reqs` that were never asked, by reason."""
    return unasked(out.get(request_key(r)) for r in reqs)


def refuse_unasked(out, reqs, what, ledger):
    """Fail the stage when any request was never asked (cap or transport); a re-run asks only those."""
    counts = unasked_requests(out, reqs)
    n = sum(counts.values())
    if n:
        st = ledger.state()
        raise RuntimeError(
            f"{what}: {n} of {len(reqs)} requests were never asked ({unasked_detail(counts)}); "
            f"US${st['spent_usd']:.2f} of the US${ledger.cap:.2f} cap is spent. Re-run this stage (with a "
            "higher --judge-budget-usd if the cap is what stopped it): only the unasked requests are asked "
            "again."
        )


def load_ledger(run, args):
    """The one ledger the judge and the summariser spend against, capped at --judge-budget-usd."""
    L = Ledger.load(run.file(LEDGER_REL), args.judge_budget_usd, C.RATES_PER_M)
    L.save(run.file(LEDGER_REL))
    return L


def _kept(items_doc):
    return [x for x in items_doc["items"] if not x["excluded"]]


# --- stage summarise --------------------------------------------------------------------------------------


def stage_summarise(args, run):
    """Every lens pool of every item turned into prose by the item-blind summariser (methodology §5.3)."""
    chash = stage_key("summarise", args, run)
    if stage_done(run, "summarise", chash) and not args.force:
        print("[summarise] up to date")
        return
    started = time.time()
    spec = C.JUDGES[C.SUMMARISER]
    ledger = load_ledger(run, args)
    client = client_for_judge(spec)
    reqs = summary_requests(run.read_json("lens/lens.json"), spec)
    rel = judge_log(spec.name, "summaries")
    out = run_requests(run, rel, reqs, client, ledger)
    ledger.save(run.file(LEDGER_REL))
    refuse_unasked(out, reqs, "summarise", ledger)
    summaries = {which: {} for which in RO.LENS_POOLS.values()}
    for r in reqs:
        rec = out[request_key(r)]
        summaries[r["meta"]["which"]][str(r["meta"]["i"])] = {
            "summary": rec["text"],
            "status": rec["status"],
            "key": rec["key"],
        }
    run.write_json(
        "judges/summaries.json",
        {
            "model": spec.model,
            "ledger": ledger.state(),
            "band_layers": list(C.LENS_BAND_LAYERS),
            "summaries": summaries,
        },
    )
    unavailable = sum(1 for w in summaries.values() for v in w.values() if v["status"] != "ok")
    print(f"[summarise] {len(reqs)} requests, {unavailable} unavailable, ${ledger.state()['spent_usd']:.3f}", flush=True)
    mark_stage(run, "summarise", chash, {"requests": len(reqs), "unavailable": unavailable, "model": spec.model},
               started=started)


# --- request builders -------------------------------------------------------------------------------------


def lens_pools(rec, fitted):
    """{pool: token list} of one item's lens record, one entry per RO.LENS_POOLS pool."""
    return {
        "L42": rec["top10_L42"],
        C.LENS_BAND: pool_band(rec["top10_by_layer"], fitted),
    }


def summary_requests(lens_doc, spec):
    """The summariser's requests for every item's lens pools, addressed to `spec`."""
    reqs = []
    fitted = lens_doc["lens"]["fitted_layers"]
    for rec in lens_doc["items"]:
        pools = lens_pools(rec, fitted)
        for which in RO.LENS_POOLS.values():
            reqs.append(dict(with_judge(summary_request(pools[which]), spec), meta={"i": rec["i"], "which": which}))
    return reqs


def build_naming_requests(kept, by_i, judged_samples):
    """(available, unavailable) naming requests, own and foil, per kept item and judged condition, meta
    {"i", "condition", "vs"}. A condition with no readout is recorded unavailable, never sent empty."""
    available, unavailable = [], []
    for it in kept:
        i = it["i"]
        foil_it = by_i.get(it["foil"])
        vs_targets = {"own": it["forms"], "foil": foil_it["forms"] if foil_it else []}
        for condition in C.JUDGED_CONDITIONS:
            kind = "summary" if condition in RO.LENS_POOLS else "samples"
            samples = judged_samples.get(condition, {}).get(i) or []
            for vs in ("own", "foil"):
                req = dict(
                    judge_request(vs_targets[vs], kind, samples), meta={"i": i, "condition": condition, "vs": vs}
                )
                (available if samples else unavailable).append(req)
    return available, unavailable


# --- the pipeline diagnostic ------------------------------------------------------------------------------


def build_diagnostics(items_doc, n=C.N_DIAG):
    """The §9 pipeline diagnostic's inputs: 20 synthetic one-sentence readouts naming the target and 20
    shuffled ten-token lists containing it."""
    by_i = {x["i"]: x for x in items_doc["items"] if not x["excluded"]}
    chosen = sorted(by_i)[:n]
    text_positive, list_positive = [], []
    for i in chosen:
        it = by_i[i]
        forms = it["forms"]
        foil_it = by_i.get(it["foil"])
        foil_forms = foil_it["forms"] if foil_it else []
        text_positive.append(
            {"i": i, "text": f"The concept here is {forms[0]}.", "targets_own": forms, "targets_foil": foil_forms}
        )
        pieces = [" " + forms[0]] + list(C.DIAG_FILLERS)
        order = np.random.default_rng(5400 + i).permutation(len(pieces))
        list_positive.append(
            {"i": i, "list": [pieces[k] for k in order], "targets_own": forms, "targets_foil": foil_forms}
        )
    return {"text_positive": text_positive, "list_positive": list_positive}


def diagnostic_summary_requests(diag, spec):
    """The diagnostic token lists as summariser requests, addressed to `spec`."""
    return [dict(with_judge(summary_request(d["list"]), spec), meta={"i": d["i"]}) for d in diag["list_positive"]]


def summarise_diagnostics(run, diag, ledger):
    """The diagnostic token lists run through the summariser, once per run."""
    spec = C.JUDGES[C.SUMMARISER]
    reqs = diagnostic_summary_requests(diag, spec)
    out = run_requests(
        run, judge_log(spec.name, "diagnostic_summaries"), reqs, client_for_judge(spec), ledger
    )
    refuse_unasked(out, reqs, "judge: diagnostic summaries", ledger)
    return {r["meta"]["i"]: out[request_key(r)] for r in reqs}


def diagnostic_requests(spec, diag, summaries):
    """The diagnostic's naming requests for one judge, own and foil; a missing summary is judged empty."""
    reqs = []
    for d in diag["text_positive"]:
        for vs, targets in (("own", d["targets_own"]), ("foil", d["targets_foil"])):
            reqs.append(
                dict(
                    with_judge(judge_request(targets, "samples", [d["text"]]), spec),
                    meta={"i": d["i"], "part": "text", "vs": vs},
                )
            )
    for d in diag["list_positive"]:
        rec = summaries.get(d["i"])
        text = rec["text"] if rec and rec["status"] == "ok" else ""
        for vs, targets in (("own", d["targets_own"]), ("foil", d["targets_foil"])):
            reqs.append(
                dict(
                    with_judge(judge_request(targets, "summary", [text]), spec),
                    meta={"i": d["i"], "part": "list", "vs": vs},
                )
            )
    return reqs


def pipeline_diagnostic(run, spec, diag, summaries, client, ledger):
    """The §9 pipeline diagnostic, run before anything is spent on a real readout: at least n-1 of n
    synthetic readouts named against their own targets and at most 1 against a foil's, or the stage fails."""
    reqs = diagnostic_requests(spec, diag, summaries)
    out = run_requests(run, judge_log(spec.name, "diagnostics"), reqs, client, ledger)
    refuse_unasked(out, reqs, f"judge {spec.name}: pipeline diagnostic", ledger)

    def _hit(req):
        rec = out[request_key(req)]
        if rec["status"] != "ok":
            return False
        v = parse_verdict(rec.get("text") or "")
        return bool(v["expressed"]) and void_reason(v, req["targets"], req["samples"]) is None

    for part, ds in (("text", diag["text_positive"]), ("list", diag["list_positive"])):
        n = len(ds)
        if n == 0:
            continue
        own = [r for r in reqs if r["meta"]["part"] == part and r["meta"]["vs"] == "own"]
        foil = [r for r in reqs if r["meta"]["part"] == part and r["meta"]["vs"] == "foil"]
        pos = sum(1 for r in own if _hit(r))
        neg = sum(1 for r in foil if _hit(r))
        if pos < n - 1 or neg > 1:
            raise RuntimeError(
                f"pipeline diagnostic failed for judge {spec.name}: {part} named {pos}/{n} against own targets "
                f"(need >= {n - 1} of {n}), {neg}/{n} against foil targets (need <= 1)"
            )
    return out


# --- stage judge ------------------------------------------------------------------------------------------

# The stages whose readouts the judge reads, each held to its chained key before anything is spent.
UPSTREAM = ("rollouts", "patchscope", "summarise", "nla", "retrieval")
SHARDED_UPSTREAM = ("nla",)


def _check_upstream(run, args):
    for up in UPSTREAM:
        want = stage_key(up, args, run)
        done = S.sharded_stage_done(run, up, want) if up in SHARDED_UPSTREAM else stage_done(run, up, want)
        if not done:
            raise RuntimeError(f"judge: stage {up} has not completed for this configuration; run it first")


def _record_unavailable(run, rel, reqs):
    """Write each no-readout request into the log as unavailable, one record per (key, meta): empty
    readouts of one item build identical requests, and each cell must still be counted."""
    existing = {_cell(r["key"], r.get("meta")) for r in run.read_jsonl(rel)}
    n = 0
    for req in reqs:
        key = request_key(req)
        cell = _cell(key, req.get("meta"))
        if cell in existing:
            continue
        run.append_jsonl(
            rel,
            {
                "key": key,
                "model": req["model"],
                "meta": req.get("meta"),
                "kind": req["kind"],
                "status": "unavailable",
                "error": "no readout available for this condition",
                "text": None,
                "usage": None,
                "cost_usd": 0.0,
                "targets": req.get("targets"),
                "samples": req.get("samples"),
                "n_samples": len(req.get("samples") or []),
            },
        )
        existing.add(cell)
        n += 1
    return n


def tally(records):
    """(unavailable, voided) over a judge's naming records."""
    unavailable = voided = 0
    for rec in records:
        if rec["status"] != "ok":
            unavailable += 1
            continue
        v = parse_verdict(rec.get("text") or "")
        if v["expressed"] and void_reason(v, rec.get("targets") or [], rec.get("samples") or []) is not None:
            voided += 1
    return unavailable, voided


def build_requests(run):
    """(naming, skipped): every naming request about the readouts as they stand, unaddressed. The analysis
    rebuilds these to bind each verdict to the text it was asked about (`analysis.bound_records`)."""
    kept = _kept(run.read_json("data/items.json"))
    by_i = {x["i"]: x for x in kept}
    return build_naming_requests(kept, by_i, RO.judged_samples(RO.free_text(run), RO.lens_summaries(run)))


def stage_judge(args, run):
    """The profile's judge over the pipeline diagnostic and the naming requests, against one ledger. A
    request left unasked fails the stage (`refuse_unasked`)."""
    chash = stage_key("judge", args, run)
    if stage_done(run, "judge", chash) and not args.force:
        print("[judge] up to date")
        return
    started = time.time()
    _check_upstream(run, args)
    items_doc = run.read_json("data/items.json")
    naming, skipped = build_requests(run)
    ledger = load_ledger(run, args)
    diag = build_diagnostics(items_doc)
    diag_summaries = summarise_diagnostics(run, diag, ledger)
    ledger.save(run.file(LEDGER_REL))
    per_judge = {}
    for spec in C.JUDGES.values():
        client = client_for_judge(spec)
        pipeline_diagnostic(run, spec, diag, diag_summaries, client, ledger)
        naming_rel = judge_log(spec.name, "naming")
        addressed = [with_judge(r, spec) for r in naming]
        n_skipped = _record_unavailable(run, naming_rel, [with_judge(r, spec) for r in skipped])
        out = run_requests(run, naming_rel, addressed, client, ledger)
        ledger.save(run.file(LEDGER_REL))
        refuse_unasked(out, addressed, f"judge {spec.name}: naming", ledger)
        mine = {request_key(r) for r in addressed}
        unavailable, voided = tally([rec for k, rec in out.items() if k in mine])
        per_judge[spec.name] = {
            "model": spec.model,
            "naming_requests": len(addressed),
            "naming_no_readout": len(skipped),
            "naming_no_readout_written": n_skipped,
            "naming_unavailable": unavailable,
            "naming_voided": voided,
        }
        print(
            f"[judge] {spec.name}: {len(addressed)} naming requests ({unavailable} unavailable, {voided} voided); "
            f"ledger ${ledger.state()['spent_usd']:.3f}",
            flush=True,
        )
    spent = ledger.state()["spent_usd"]
    print(f"[judge] ${spent:.3f} of the ${ledger.cap:.2f} cap", flush=True)
    mark_stage(run, "judge", chash, {"judges": per_judge, "spent_usd": spent, "cap_usd": ledger.cap}, started=started)
