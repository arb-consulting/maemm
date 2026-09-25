"""The judged instruments' shared parts: the reply parsers, the status classifier and the transport.

The identification prompt, its parser and the lens-summary rule are `eval.steering_vector_inversion.judge`'s,
so both steering packages ask one question. `run` binds `eval/common/judge_client.py`'s transport with one
ledger per run at judges/ledger.json.
"""
import json
import re

from eval.common.judge_client import (Ledger, client_for, decode_json_object, judge_log, looks_like_refusal,
                                      request_key, run_requests, unasked, unasked_detail, with_judge)
from eval.common.runs import RunDir
from eval.steering_vector_inversion.judge import parse_identification, summary_status
from eval.steering_vector_inversion_bipo import config as C

# This package's own OpenRouter X-Title.
TITLE = "maemm-persona-vectors"
# Above this many requests, `run` sends a few probes first (see its docstring).
PREFLIGHT_OVER = 20
PREFLIGHT_PROBES = 5

_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n)")
_BLANK_RUN = re.compile(r"\n{3,}")


def clean_sample(text):
    """`text` with trailing spaces before a newline dropped and runs of 3+ newlines collapsed, stripped;
    "(empty)" when nothing survives, so a bundle's numbering still lines up with its samples."""
    t = _BLANK_RUN.sub("\n\n", _TRAILING_SPACE.sub("", text or "")).strip()
    return t or "(empty)"


def parse_description(text):
    """(plus, minus) or None: the two pole sentences of a `describe` reply, whitespace-collapsed. Only
    well-formedness; the writer's rules are `persona.description_error`'s."""
    def ok(o):
        return isinstance(o.get("plus"), str) and isinstance(o.get("minus"), str) \
            and o["plus"].strip() != "" and o["minus"].strip() != ""

    obj, _ = decode_json_object(text, accept=ok)
    return None if obj is None else (" ".join(obj["plus"].split()), " ".join(obj["minus"].split()))


# What the run's ledger prices with: the profile's judge and the description writer.
LEDGER_RATES = {**C.RATES_PER_M, C.DESCRIBE_JUDGE.model: C.DESCRIBE_JUDGE.rates}


def spec_of(judge_name):
    """The spec a judge name is asked with: the profile's judge, or the description writer."""
    return C.DESCRIBE_JUDGE if judge_name == C.DESCRIBE_JUDGE.name else C.JUDGES[judge_name]


# The number parser's three verdicts, as the statuses this package's logs and tables use.
NUMBER_STATUS = {"valid": "ok", "refusal": "refused", "invalid": "parse_fail"}
STATUSES = ("ok", "refused", "content_filter", "parse_fail", "truncated", "unavailable")


def status_fn(kind, text, finish=None):
    """`run_requests`' classifier, one of `STATUSES`: an empty reply is `content_filter` or `unavailable`
    (retried); an identification reply is read by the shared parser, `truncated` when the cap cut it before
    a number; a lens summary follows `summary_status`; a description is `ok` when it parses."""
    t = (text or "").strip()
    if not t:
        return "content_filter" if finish == "content_filter" else "unavailable"
    if kind == "identify":
        status = NUMBER_STATUS[parse_identification(t)[1]]
        return "truncated" if status == "parse_fail" and finish == "length" else status
    if kind != "describe":
        return summary_status(t)
    if looks_like_refusal(t):
        return "refused"
    return "ok" if parse_description(t) is not None else "parse_fail"


# ------------------------------------------------------------------------------------------- sending
def cell_id(key, meta):
    """(request key, meta) as one hashable id: the granularity `run_requests` records at."""
    return key, json.dumps(meta, sort_keys=True, ensure_ascii=False)


def run(run_dir, instrument, reqs, judge_name, api_key, cap_usd=C.JUDGE_CAP_USD, workers=32, client=None):
    """One judge's pass over `reqs` against the run's one ledger; the records, in request order. Over
    `PREFLIGHT_OVER` requests, `PREFLIGHT_PROBES` go first and the pass stops if all come back unavailable.
    A pass that leaves any request unasked (cap, transport, rejected key) raises; a later pass resumes."""
    run = RunDir(run_dir)
    spec = spec_of(judge_name)
    rel = judge_log(judge_name, instrument)
    ledger_path = run.file("judges/ledger.json")
    ledger = Ledger.load(ledger_path, cap_usd, LEDGER_RATES)
    client = client if client is not None else client_for(spec, api_key, title=TITLE)
    addressed = [with_judge(r, spec) for r in reqs]
    n_before = len(run.read_jsonl(rel))

    def send(batch):
        return run_requests(run, rel, batch, client, ledger, workers=workers, status_fn=status_fn,
                            checkpoint=lambda: ledger.save(ledger_path), record_fields=())

    try:
        if len(addressed) > PREFLIGHT_OVER:
            # several probes: a content filter can blank one bundle without the setup being broken
            probes = send(addressed[:PREFLIGHT_PROBES])
            failed = [r for r in probes.values() if r["status"] == "unavailable"]
            if len(failed) == len(probes):
                raise RuntimeError(
                    f"{judge_name}/{instrument}: the first {len(probes)} requests came back unavailable "
                    f"({failed[0].get('error') or 'empty reply'}); the remaining {len(addressed) - len(probes)} "
                    f"were not sent. Check the model id {spec.model!r}, its provider pin, and the budget cap.")
        done = send(addressed)
    finally:
        ledger.save(ledger_path)

    records = {cell_id(r["key"], r.get("meta")): r for r in run.read_jsonl(rel)}
    out = [records.get(cell_id(request_key(r), r.get("meta"))) or done[request_key(r)] for r in addressed]
    counts = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n_new = len(run.read_jsonl(rel)) - n_before
    print(f"[{instrument}] {judge_name}: {len(out)} requests, {n_new} new, "
          f"{', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}, "
          f"ledger US${ledger.state()['spent_usd']:.4f}", flush=True)
    left = unasked(out)
    n_unasked = sum(left.values())
    if n_unasked:
        advice = "Raise the cap and run the stage again" if left["budget"] else "Run the stage again"
        raise RuntimeError(f"{judge_name}/{instrument}: {n_unasked} of {len(out)} requests were never "
                           f"asked ({unasked_detail(left)}); US${ledger.state()['spent_usd']:.2f} of "
                           f"the US${cap_usd:.2f} cap is spent. {advice}; the answered requests resume "
                           f"from the log.")
    return out


def estimate(reqs, judge_name):
    """The ledger's own US$ estimate for `reqs` under one judge (the arithmetic it reserves with)."""
    spec = spec_of(judge_name)
    ledger = Ledger(0.0, LEDGER_RATES)
    return sum(ledger.estimate(with_judge(r, spec)) for r in reqs)


MISSINGNESS_COLUMNS = ("instrument", "judge", "arm", "n_requests", *STATUSES)
POOLED = "(pooled)"


def missingness(instrument, rows, arm_of):
    """One row per (judge, arm) and a pooled row: requests sent and how each ended (`STATUSES`). `rows`
    carry `judge`, `family_id` and `status`; `arm_of` reads the arm off a family id."""
    counts = {}
    for r in rows:
        for arm in (arm_of(r.get("family_id")), POOLED):
            cell = counts.setdefault((r.get("judge"), arm), dict.fromkeys(STATUSES, 0))
            status = r.get("status")
            cell[status if status in STATUSES else "unavailable"] += 1
    out = []
    for (judge_name, arm) in sorted(counts, key=lambda k: (str(k[0]), k[1] == POOLED, str(k[1]))):
        cell = counts[(judge_name, arm)]
        out.append({"instrument": instrument, "judge": judge_name, "arm": arm,
                    "n_requests": sum(cell.values()), **cell})
    return out
