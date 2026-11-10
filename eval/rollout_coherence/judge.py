"""Stage `frontier_context_judge` (methodology §5): every pair of `frontier/context/pairs.json` sent in both
slot orders to the active profile's judge, with the one blind pairwise prompt, into
`judges/<judge>/context.jsonl` against one ledger and one cap. A refusal is its own category and is never
re-asked; a parse failure is re-asked once, then scored as a tie.
"""
import json, os, sys, time

from eval.common.judge_client import (
    UNASKED_KINDS,
    client_for,
    decode_json_object,
    judge_log,
    key_for,
    looks_like_refusal,
    moderation_blocked,
    request_key,
    unasked_detail,
    with_judge,
)
from eval.common.judge_client import Ledger as _Ledger
from eval.common.judge_client import run_requests as _run_requests
from eval.common.runs import config_hash, mark_stage, stage_done, write_provenance
from eval.rollout_coherence import config as C
from eval.rollout_coherence.runs import stage_hashes

TITLE = "maemm-rollout-coherence"   # OpenRouter's X-Title
STAGE, UPSTREAM = "frontier_context_judge", "frontier_context_pairs"
CONTEXT_PAIRS = "frontier/context/pairs.json"


def _is_retryable_record(r):
    """Whether a logged record must be asked again: any `unavailable` record (it carries no verdict, since
    `classify` never returns that status for a reply), and a parse_fail that has not had its retry. A
    request the provider's moderation blocked is an answer (`refused`): re-asked, it would block again."""
    if moderation_blocked(r):
        return False
    status = r.get("status")
    return status == "unavailable" or (status == "parse_fail" and (r.get("attempts") or 0) < 2)


class Ledger(_Ledger):
    """The shared ledger, reconciled against the request logs on load."""

    @classmethod
    def load(cls, path, cap, rates, totals=None):
        """`totals` (`totals_from_records`) replaces the file's numbers when larger: a hard kill loses the
        end-of-stage save while every answered request is on disk with its cost. A corrupt file is read as
        a missing one."""
        L = cls(cap, rates)
        if os.path.exists(path):
            try:
                with open(path) as h:
                    s = json.load(h)
                L.spent, L.n, L.tin, L.tout = s["spent_usd"], s["requests"], s["input_tokens"], s["output_tokens"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"judge: {path} is not a valid ledger ({e}); reconstructing from request records", file=sys.stderr)
        if totals and totals["spent_usd"] > L.spent:
            L.spent, L.n = totals["spent_usd"], totals["requests"]
            L.tin, L.tout = totals["input_tokens"], totals["output_tokens"]
        return L


def totals_from_records(run):
    """Spend, requests and tokens rebuilt from every settled record of the judge's log, one per request key
    (a copied answer was never a second call). A lower bound: a discarded first attempt has no record."""
    seen = {}
    for judge in C.JUDGES:
        for r in run.read_jsonl(judge_log(judge, C.INSTRUMENT)):
            if not _is_retryable_record(r):
                seen[(judge, r.get("key"))] = r
    out = {"spent_usd": 0.0, "requests": 0, "input_tokens": 0, "output_tokens": 0}
    for r in seen.values():
        u = r.get("usage") or {}
        out["spent_usd"] += float(r.get("cost_usd") or 0.0)
        out["input_tokens"] += int(u.get("input_tokens") or 0)
        out["output_tokens"] += int(u.get("output_tokens") or 0)
        out["requests"] += 1
    return out


def load_ledger(run, cap):
    """The run's one ledger, reconciled against the request log."""
    return Ledger.load(run.file(C.LEDGER), cap, C.RATES_PER_M, totals=totals_from_records(run))


def run_requests(run, rel, reqs, client, ledger, workers=16, status_fn=None, checkpoint=None,
                 checkpoint_every=200):
    """`eval.common.judge_client.run_requests` with this package's retry rule and no per-request record
    fields (a pair request carries no targets, samples or readout)."""
    return _run_requests(run, rel, reqs, client, ledger, workers=workers, status_fn=status_fn,
                         checkpoint=checkpoint, checkpoint_every=checkpoint_every,
                         is_retryable=_is_retryable_record, record_fields=())


CHOICES = ("A", "B", "tie")


def choice_of(obj):
    """The verdict a decoded reply object carries, or None."""
    c = obj.get("choice")
    if not isinstance(c, str):
        return None
    c = c.strip().lower()
    return next((k for k in CHOICES if k.lower() == c), None)


def build_requests(pairs):
    """Every pair in both slot orders, with no model and no token cap (`with_judge` binds those)."""
    out = []
    for p in pairs:
        for order in (0, 1):
            a, b = (p["x"], p["partner"]) if order == 0 else (p["partner"], p["x"])
            out.append({"system": C.SYSTEM_PROMPT, "user": C.USER_TEMPLATE.format(a=a, b=b),
                        "kind": "pair", "meta": {"pid": p["pid"], "order": order}})
    return out


def parse_choice(text):
    """The verdict in a judge's reply, found by the shared `decode_json_object`, or None."""
    obj, _repaired = decode_json_object(text, accept=lambda o: choice_of(o) is not None)
    return None if obj is None else choice_of(obj)


def classify(text, finish):
    """`refused` for an empty reply or a refusal (own category, a tie, never re-asked); `parse_fail` for a
    reply without a valid choice (re-asked once, then a tie); else `ok`."""
    t = (text or "").strip()
    if not t or finish == "refusal" or looks_like_refusal(t): return "refused"
    return "ok" if parse_choice(t) else "parse_fail"


def load_answers(run, judge):
    """{(pid, order): {"status", "choice"}} over the settled records of one judge's log."""
    out = {}
    for r in run.read_jsonl(judge_log(judge, C.INSTRUMENT)):
        if _is_retryable_record(r): continue
        m = r["meta"]
        status = "refused" if moderation_blocked(r) else r["status"]
        out[(m["pid"], m["order"])] = {"status": status,
                                       "choice": parse_choice(r.get("text")) if status == "ok" else None}
    return out


# The categories of every missingness table, in order. `content_filter` is the provider refusing to pass
# the request to the model; `unavailable` is a request without a verdict, or without a record.
MISSINGNESS_STATUSES = ("ok", "refused", "content_filter", "parse_fail", "unavailable")


def request_outcomes(run, judge, pairs):
    """`{group: {status: n}}` over both slot orders of every pair, by the last record of each request, so
    each group's counts sum to the requests it planned."""
    group_of = {p["pid"]: p.get("group") for p in pairs}
    last = {}
    for r in run.read_jsonl(judge_log(judge, C.INSTRUMENT)):
        m = r.get("meta") or {}
        if m.get("pid") in group_of:
            last[(m["pid"], m.get("order"))] = r
    out = {g: dict.fromkeys(MISSINGNESS_STATUSES, 0) for g in sorted({g for g in group_of.values() if g})}
    for pid, group in group_of.items():
        for order in (0, 1):
            r = last.get((pid, order))
            if r is None:
                status = "unavailable"
            elif moderation_blocked(r) or (r.get("usage") or {}).get("finish") == "content_filter":
                status = "content_filter"
            else:
                status = r.get("status") if r.get("status") in MISSINGNESS_STATUSES else "unavailable"
            out[group][status] += 1
    return out


class NotFullyAsked(RuntimeError):
    """A pass ended with requests that were never asked, so its manifest is incomplete."""


def unasked_requests(records, reqs):
    """{"budget": [reqs], "transport": [reqs], "auth": [reqs]}: the requests of a pass never really asked."""
    out = {kind: [] for kind in UNASKED_KINDS}
    for r in reqs:
        kind = (records.get(request_key(r)) or {}).get("error_kind")
        if kind in out:
            out[kind].append(r)
    return out


def refuse_unasked(records, reqs, ledger, what):
    """Fail the stage, before anything is marked complete, when a pass did not ask its whole manifest
    (an unanswered order would otherwise read as a tie). Names the cap needed when the cap stopped it."""
    left = unasked_requests(records, reqs)
    counts = {kind: len(rs) for kind, rs in left.items()}
    n = sum(counts.values())
    if not n:
        return
    state = ledger.state()
    if left["budget"]:
        needed = state["spent_usd"] + sum(ledger.estimate(r) for r in left["budget"])
        advice = (f"Raise the cap to at least US${needed:.2f} and run the stage again; it re-asks the "
                  "unasked requests and nothing else.")
    else:
        advice = "Run the stage again; it re-asks the unasked requests and nothing else."
    raise NotFullyAsked(
        f"{what}: this pass left {n:,} of {len(reqs):,} requests unasked ({unasked_detail(counts)}), "
        f"after US${state['spent_usd']:.4f} over {state['requests']:,} requests against the "
        f"US${ledger.cap:.2f} cap. Nothing is recorded as complete. " + advice)


def unasked_keys(run, judge):
    """{"budget": [keys], "transport": [keys], "auth": [keys]} over one judge's log, by last record."""
    last = {}
    for r in run.read_jsonl(judge_log(judge, C.INSTRUMENT)):
        last[r.get("key")] = r
    out = {kind: [] for kind in UNASKED_KINDS}
    for key, r in last.items():
        if r.get("error_kind") in out:
            out[r["error_kind"]].append(key)
    return out


def refuse_unasked_logs(run):
    """Raise unless the judge log is free of requests that were never asked; `report` calls it so an
    interrupted pass cannot be published."""
    bad = []
    for judge in C.JUDGES:
        counts = {kind: len(ks) for kind, ks in unasked_keys(run, judge).items()}
        if sum(counts.values()):
            bad.append(f"{judge}: {unasked_detail(counts)} (stage `{STAGE}`)")
    if bad:
        raise NotFullyAsked(
            "report: the judge log still holds requests that were never asked -- " + "; ".join(bad)
            + ". Their orders would be imputed as ties. Run the stage again (raising the cap if the cap "
              "stopped it); only the unasked requests are asked.")


def judge_config_hash(run, n):
    """The judge's model, routing, sampling and token cap and the prompts, chained to the pair stage that
    wrote the manifest."""
    judges = {name: {"model": s.model, "provider": s.provider, "sampling": s.sampling,
                     "max_tokens": s.max_tokens_for("pair")} for name, s in C.JUDGES.items()}
    return config_hash({"judges": judges, "n": n, "instrument": C.INSTRUMENT,
                        "prompt": [C.SYSTEM_PROMPT, C.USER_TEMPLATE], "scoring": C.SCORING_VERSION,
                        "upstream": stage_hashes(run, [UPSTREAM])})


def _n_stale_answers(run, judge, reqs):
    """Raise when a request of this invocation has no settled answer; else the number of answered
    (pid, order) rows outside this manifest (an append-only log can hold rows a rebuilt manifest dropped)."""
    answers = load_answers(run, judge)
    want_ids = {(r["meta"]["pid"], r["meta"]["order"]) for r in reqs}
    missing = want_ids - set(answers)
    if missing:
        raise RuntimeError(f"{judge}: {len(missing)} of {len(want_ids)} requests still unresolved "
                           f"(unavailable, or a parse_fail whose retry did not land); re-run to resume")
    return len(set(answers) - want_ids)


def stage_frontier_context_judge(args, run):
    """Every pair of `frontier/context/pairs.json` in both orders, asked of the profile's judge."""
    from eval.rollout_coherence.analysis import check_parse_share, score_pairs

    doc = run.read_json(CONTEXT_PAIRS)
    reqs = build_requests(doc["pairs"])
    chash = judge_config_hash(run, doc["config"]["n"])
    if stage_done(run, STAGE, chash) and not args.force:
        return
    started = time.time()
    (spec,) = C.JUDGES.values()
    client = client_for(spec, key_for(spec), title=TITLE)   # refuses a missing key before anything is spent
    ledger = load_ledger(run, args.judge_budget_usd)
    ledger_path = run.file(C.LEDGER)
    jreqs = [with_judge(r, spec) for r in reqs]
    try:
        done = run_requests(run, judge_log(spec.name, C.INSTRUMENT), jreqs, client, ledger,
                            workers=C.JUDGE_WORKERS, status_fn=lambda k, t, f: classify(t, f),
                            checkpoint=lambda: ledger.save(ledger_path))
    finally:
        ledger.save(ledger_path)
    counts = {}
    for r in done.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    write_provenance(run, {"status_counts": counts, "seconds": time.time() - started, "model": spec.model,
                           "provider": spec.provider, "sampling": spec.sampling,
                           "judge_ledger": ledger.state()}, stage=STAGE)
    refuse_unasked(done, jreqs, ledger, f"{STAGE} {spec.name}")
    check_parse_share(score_pairs(doc["pairs"], load_answers(run, spec.name)))
    stale = _n_stale_answers(run, spec.name, jreqs)
    mark_stage(run, STAGE, chash, {"requests": len(reqs), "judge": spec.name, "status_counts": counts,
                                   "n_stale_answers": stale}, started=started)
