"""The lens's token lists turned into prose, so the naming judge reads the lens as it reads a rollout.

One model (`C.SUMMARISER`) writes every summary with the shared prompt (evals/downstream/common/lens_summary.py) and sees
only the tokens. Two readouts are summarised: the layer-42 top-10 at both read positions
(judges/summaries.json) and the eight-layer pool at the final period (judges/summaries_band8.json)."""

import time

from evals.downstream.common.judge_client import client_for, judge_log, key_for, request_key, unasked, unasked_detail, with_judge
from evals.downstream.common.lens_summary import summary_request as _summary_request
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import mean_cell as M
from evals.downstream.workspace_modulation.runs import resolve_judge_budget, stage_key

SUMMARIES_REL = "judges/summaries.json"
BAND_REL = "judges/summaries_band8.json"


def summary_request(tokens, meta=None, spec=None):
    """The shared summariser request over one cell's tokens, addressed to the summariser."""
    return dict(with_judge(_summary_request(tokens), spec or C.JUDGES[C.SUMMARISER]), meta=meta)


def parse_summary(text):
    """The reply's prose, stripped; None for an empty reply."""
    d = (text or "").strip()
    return d or None


def read_cells(lens_doc, kept):
    """[(item, lens cell record)] at the two read positions, in item order. A read position with no lens
    record raises: the lens pass and the item file would describe different populations."""
    by_i = {int(x["i"]): {int(c["pos"]): c for c in x["cells"]} for x in lens_doc["items"]}
    out = []
    for it in kept:
        cells = by_i.get(int(it["i"])) or {}
        for pos in M.read_positions(it):
            if int(pos) not in cells:
                raise RuntimeError(
                    f"summarise: item i={it['i']} has no lens record at cell {pos} — re-run the lens stage "
                    "over this run's items"
                )
            out.append((int(it["i"]), cells[int(pos)]))
    return out


def band_cells(run, kept):
    """[(item, final-period cell, eight-layer pool)] in item order; raises where a final period has no saved
    per-layer lists."""
    from evals.downstream.workspace_modulation.lens import band_pools

    want = [(int(it["i"]), int(M.final_pos(it))) for it in kept]
    pools = band_pools(run, want)
    out = []
    for i, p in want:
        if (i, p) not in pools:
            raise RuntimeError(
                f"summarise: item i={i} has no per-layer lens lists at its final period {p} — re-run the lens "
                "stage over this run's items"
            )
        out.append((i, p, pools[(i, p)]))
    return out


def _cells(reqs, out, tokens):
    return {
        f"{r['meta']['i']}/{r['meta']['pos']}": {
            "summary": parse_summary(out[request_key(r)]["text"]) if out[request_key(r)]["status"] == "ok" else None,
            "status": out[request_key(r)]["status"],
            "tokens": list(t),
            "key": out[request_key(r)]["key"],
        }
        for r, t in zip(reqs, tokens)
    }


def stage_summarise(args, run):
    from evals.downstream.workspace_modulation.judge import TITLE, ask, load_ledger

    chash = stage_key("summarise", args, run)
    if stage_done(run, "summarise", chash) and not args.force:
        print("[summarise] up to date")
        return
    started = time.time()
    spec = C.JUDGES[C.SUMMARISER]
    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    l42 = read_cells(run.read_json("lens/lens.json"), kept)
    band = band_cells(run, kept)
    ledger = load_ledger(run, resolve_judge_budget(args))
    client = client_for(spec, key_for(spec), url=C.OPENROUTER_URL, title=TITLE)
    reqs42 = [summary_request(c["top10_L42"], meta={"i": i, "pos": c["pos"]}, spec=spec) for i, c in l42]
    reqs8 = [summary_request(pool, meta={"i": i, "pos": p, "pool": C.LENS_BAND}, spec=spec) for i, p, pool in band]
    out = ask(run, judge_log(spec.name, "summary"), reqs42 + reqs8, client, ledger)
    spent = ledger.state()["spent_usd"]
    counts = unasked(out[request_key(r)] for r in reqs42 + reqs8)
    if sum(counts.values()):
        raise RuntimeError(
            f"summarise: {sum(counts.values())} of {len(reqs42) + len(reqs8)} requests were never asked "
            f"({unasked_detail(counts)}); US${spent:.3f} of the US${ledger.cap:.2f} cap is spent and the stage "
            "is NOT marked done. Re-run it, with a larger --judge-budget-usd if the cap is what stopped it."
        )
    cells42 = _cells(reqs42, out, [c["top10_L42"] for _i, c in l42])
    cells8 = _cells(reqs8, out, [pool for _i, _p, pool in band])
    run.write_json(SUMMARIES_REL, {"model": spec.model, "judge": spec.name, "cells": cells42})
    run.write_json(BAND_REL, {"model": spec.model, "judge": spec.name, "band": C.LENS_BAND,
                              "band_layers": list(C.LENS_BAND_LAYERS), "cells": cells8})
    bad = sum(1 for v in (*cells42.values(), *cells8.values()) if v["summary"] is None)
    print(f"[summarise] {len(reqs42)} layer-42 + {len(reqs8)} eight-layer cells, {bad} without a summary, "
          f"${spent:.3f} of ${ledger.cap:.2f}", flush=True)
    mark_stage(run, "summarise", chash, {"n": len(reqs42) + len(reqs8), "unavailable": bad, "model": spec.model},
               started=started)
