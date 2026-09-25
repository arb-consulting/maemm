"""Stage `maem_control` (methodology §3.5): MAEM on the mid-prompt and prompt-mean activations that
`nla_control` saved, centred and injected as its readout is; writes rollouts/maem_control.json. Every kind
generates first and the inverter is freed before the re-read."""

import time
import numpy as np
from evals.downstream.workspace_understanding import config as C
from evals.downstream.workspace_understanding.model import (
    arm_seed,
    direction,
    free_model,
    generate_injected,
    load_base,
    load_inverter,
    new_filter_stats,
    norm_filter_record,
    refuse_unless_generation_agrees,
    reread_cos,
    stop_token_ids,
    centring_mean,
)
from evals.downstream.workspace_understanding.nla_control import relative_shares
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_understanding.runs import stage_key, write_provenance


def load_control_vectors(run, kept_ids):
    """{kind: {i: vector}} from every activations/controls/*.npz; raises unless every kept item is covered once."""
    import glob, os

    out = {kind: {} for kind in C.POSITION_CONTROLS}
    for p in sorted(glob.glob(run.file("activations/controls/*.npz"))):
        z = np.load(p)
        for row, i in enumerate(z["i"]):
            for kind in C.POSITION_CONTROLS:
                if int(i) in out[kind]:
                    raise RuntimeError(f"control vector for item {int(i)} appears twice ({os.path.basename(p)})")
                out[kind][int(i)] = z[kind][row]
    missing = sorted(set(kept_ids) - set(out[C.POSITION_CONTROLS[0]]))
    if missing:
        raise RuntimeError(f"control vectors miss {len(missing)} items; run stage nla_control for every shard first")
    return out


def stage_maem_control(args, run):
    chash = stage_key("maem_control", args, run)
    if stage_done(run, "maem_control", chash) and not args.force:
        print("[maem_control] up to date")
        return
    started = time.time()
    from maem.prompts import build_prompt_ids, marker_positions

    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    by_i = {x["i"]: x for x in kept}
    ids = [x["i"] for x in kept]
    vecs = load_control_vectors(run, ids)
    mu = centring_mean()
    base, tok = load_base(args.device)
    prompt_ids, mpos = build_prompt_ids(tok)
    assert marker_positions(tok, prompt_ids) == mpos and len(mpos) == 1
    stops = stop_token_ids(tok, base)
    inverter = load_inverter(args.device)
    t0 = time.time()
    doc = {"config": {"inverter": [C.INVERTER, C.INVERTER_REVISION], "kinds": list(C.POSITION_CONTROLS),
                      "seeds": {}, "prompt_ids": [int(t) for t in prompt_ids], "marker_pos": mpos[0],
                      "stop_ids": list(stops), "injection_check": {}}, "kinds": {}}
    written = {}
    try:
        refuse_unless_generation_agrees(base, inverter)
        for kind in C.POSITION_CONTROLS:
            seed = arm_seed(f"maem_{kind}", args.seed)
            doc["config"]["seeds"][kind] = seed
            dirs = np.stack([direction(vecs[kind][i], mu) for i in ids])
            foil_dirs = np.stack([direction(vecs[kind][by_i[i]["foil"]], mu) for i in ids])
            samples, greedy = generate_injected(inverter, tok, dirs, prompt_ids, mpos[0], args.device,
                                                seed=seed, stop_ids=stops)
            written[kind] = (dirs, foil_dirs, samples, greedy)
    finally:
        inverter = free_model(inverter)
    tally = new_filter_stats()
    for kind in C.POSITION_CONTROLS:
        dirs, foil_dirs, samples, greedy = written[kind]
        H = np.stack([vecs[kind][i] for i in ids])
        g_texts = [g["text"] for g in greedy]
        share, dshare = relative_shares(g_texts, H)
        own = reread_cos(g_texts, dirs, base, tok, args.device, stats=tally)
        foil = reread_cos(g_texts, foil_dirs, base, tok, args.device)
        gap = float(np.mean(own) - np.mean(foil))
        check = {"greedy_distinct_share": share, "distinct_input_share": dshare, "greedy_cos_own_mean": float(np.mean(own)), "greedy_cos_foil_mean": float(np.mean(foil)), "gap": gap}
        doc["config"]["injection_check"][kind] = check
        doc["kinds"][kind] = {"items": [{"i": i, "greedy": dict(greedy[k], cos_own=float(own[k]), cos_foil=float(foil[k])), "samples": samples[k]} for k, i in enumerate(ids)]}
        # the gap is recorded, never enforced: a control that carries little of its item is its result
        print(f"[maem_control] {kind}: {len(ids)} items; distinct greedy {share:.2f} of {dshare:.2f} distinct inputs; cos own {check['greedy_cos_own_mean']:.3f} foil {check['greedy_cos_foil_mean']:.3f} gap {gap:.3f}", flush=True)
    doc["config"]["seconds"] = time.time() - t0
    doc["config"]["norm_filter"] = norm_filter_record(tally)
    run.write_json("rollouts/maem_control.json", doc)
    write_provenance(run, {"maem_control_injection_check": doc["config"]["injection_check"]}, stage="maem_control")
    mark_stage(run, "maem_control", chash,
               {"injection_check": doc["config"]["injection_check"], "n_items": len(ids), "stop_ids": list(stops),
                "norm_filter": norm_filter_record(tally)},
               started=started)
