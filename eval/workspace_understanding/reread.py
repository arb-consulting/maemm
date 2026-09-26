"""Stage `reread` (methodology §6.4): the clean re-read cosine (MAEMM's own fidelity measure) of the NLA and
Patchscopes texts against the item's own centred direction and the foil's, on the clean base. The verbalizer
is scored on its whole generation, tags included, in a wider window (config.REREAD_WINDOW). Unfiltered; the
stage records what the norm filter would have dropped. A self-check first reproduces MAEMM's saved greedy
cosines."""

import time
import numpy as np
from eval.workspace_understanding import shards as S
from eval.workspace_understanding import config as C
from eval.workspace_understanding.model import (
    centring_mean,
    direction,
    load_base,
    new_filter_stats,
    norm_filter_record,
    reread_cos,
)
from eval.common.runs import mark_stage, stage_done
from eval.workspace_understanding.runs import stage_key, write_provenance


def _readouts(run, kept, nla_doc):
    """condition -> {i: {"greedy": text, "samples": [texts]}} for every re-read condition present."""
    out = {}
    if nla_doc:
        idx = {r["i"]: r for r in nla_doc["items"]}
        out["nla"] = {i: {"greedy": r["greedy"]["full_text"], "samples": [s["full_text"] for s in r["samples"]]} for i, r in idx.items()}
        out["nla64"] = {i: {"greedy": r["greedy"]["full_text_trunc"], "samples": [s["full_text_trunc"] for s in r["samples"]]} for i, r in idx.items()}
    if run.exists("rollouts/patchscope.json"):
        arms = run.read_json("rollouts/patchscope.json")["arms"]
        for name in C.PATCH_ARMS:  # the floor too
            if arms.get(name):
                out[name] = {r["i"]: {"greedy": r["greedy"]["text"], "samples": [s["text"] for s in r["samples"]]} for r in arms[name]["items"]}
    return out


def stage_reread(args, run):
    chash = stage_key("reread", args, run)
    if stage_done(run, "reread", chash) and not args.force:
        print("[reread] up to date")
        return
    started = time.time()
    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    by_i = {x["i"]: x for x in kept}
    H = np.load(run.file("activations/h_all.npz"))["h"]
    mu = centring_mean()
    dirs = {x["i"]: direction(H[x["i"], C.READ_LAYER], mu) for x in kept}
    nla_doc = S.load_nla(run, kept_ids=[x["i"] for x in kept])
    readouts = _readouts(run, kept, nla_doc)
    base, tok = load_base(args.device)
    t0 = time.time()
    # self-check: MAEMM's saved greedy cosines through the same call
    maemm = {r["i"]: r for r in run.read_json("rollouts/maemm.json")["items"]}
    ids = [x["i"] for x in kept if x["i"] in maemm]
    got = reread_cos([maemm[i]["greedy"]["text"] for i in ids], np.stack([dirs[i] for i in ids]), base, tok, args.device)
    diff = float(np.max(np.abs(got - np.array([maemm[i]["greedy"]["cos_own"] for i in ids]))))
    print(f"[reread] self-check: max |re-read − saved| on MAEMM greedies = {diff:.4f}", flush=True)
    if diff > C.REREAD_SELFCHECK_TOL:
        raise RuntimeError(f"reread self-check failed: {diff:.4f} > {C.REREAD_SELFCHECK_TOL}")
    out = {}
    total = new_filter_stats()
    for cond, ro in readouts.items():
        ids = [i for i in (x["i"] for x in kept) if i in ro]
        texts, own_d, foil_d, where = [], [], [], []
        for i in ids:
            for k, t in enumerate([ro[i]["greedy"]] + list(ro[i]["samples"])):
                texts.append(t)
                own_d.append(dirs[i])
                foil_d.append(dirs[by_i[i]["foil"]])
                where.append((i, k))
        window = C.REREAD_WINDOW.get(cond)
        # the tally is filled by the own pass alone (the foil pass re-reads the same texts)
        tally = new_filter_stats()
        own = reread_cos(texts, np.stack(own_d), base, tok, args.device, max_length=window, stats=tally)
        foil = reread_cos(texts, np.stack(foil_d), base, tok, args.device, max_length=window)
        for key, value in tally.items():
            total[key] += value
        recs = {}
        for (i, k), o, f in zip(where, own, foil):
            r = recs.setdefault(i, {"i": i, "greedy": None, "samples": []})
            cell = {"cos_own": float(o), "cos_foil": float(f)}
            if k == 0:
                r["greedy"] = cell
            else:
                r["samples"].append(cell)
        out[cond] = {"items": [recs[i] for i in ids], "n_items": len(ids),
                     "score_max_length": window or C.REREAD_WINDOW_TOKENS,
                     "norm_filter": norm_filter_record(tally)}
        so = np.mean([np.mean([s["cos_own"] for s in r["samples"]]) for r in recs.values()])
        sf = np.mean([np.mean([s["cos_foil"] for s in r["samples"]]) for r in recs.values()])
        print(f"[reread] {cond}: {len(ids)} items; samples cos own {so:.3f} foil {sf:.3f}", flush=True)
    doc = {
        "config": {"selfcheck_max_abs_diff": diff, "scorer": "model.reread_cos (eval.common.scorer.score_probe_cos)", "window_tokens": C.REREAD_WINDOW_TOKENS,
                   "score_max_length": {c: C.REREAD_WINDOW.get(c, C.REREAD_WINDOW_TOKENS) for c in out},
                   "norm_filter": norm_filter_record(total)},
        "conditions": out,
        "seconds": time.time() - t0,
    }
    run.write_json("rollouts/reread.json", doc)
    write_provenance(run, {"reread_selfcheck_max_abs_diff": diff}, stage="reread")
    mark_stage(run, "reread", chash,
               {"selfcheck_max_abs_diff": diff, "conditions": sorted(out), "norm_filter": norm_filter_record(total),
                "norm_filter_by_condition": {c: out[c]["norm_filter"] for c in sorted(out)}}, started=started)
