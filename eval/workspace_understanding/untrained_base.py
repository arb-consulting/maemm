"""Stage `untrained_base` (methodology §3.7): MAEMM's prompt, marker and injection on the untrained base
model, under MAEMM's generation settings. An ablation of MAEMM, scored by the word rule alone; its
injection check is recorded, never enforced."""

import time

import numpy as np

from eval.workspace_understanding import config as C
from eval.workspace_understanding.model import (
    arm_seed,
    direction,
    distinct_share,
    generate_injected,
    load_base,
    new_filter_stats,
    norm_filter_record,
    reread_cos,
    stop_token_ids,
    centring_mean,
)
from eval.common.runs import mark_stage, stage_done
from eval.workspace_understanding.runs import stage_key, write_provenance

READOUTS_REL = "rollouts/untrained_base.json"


def stage_untrained_base(args, run):
    chash = stage_key("untrained_base", args, run)
    if stage_done(run, "untrained_base", chash) and not args.force:
        print("[untrained_base] up to date")
        return
    started = time.time()
    from mxf.prompts import build_prompt_ids, marker_positions

    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    H = np.load(run.file("activations/h_all.npz"))["h"]
    mu = centring_mean()
    dirs = np.stack([direction(H[it["i"], C.READ_LAYER], mu) for it in kept])
    foil_dirs = np.stack([dirs[it["foil"]] for it in kept])
    base, tok = load_base(args.device)
    prompt_ids, mpos = build_prompt_ids(tok)
    assert marker_positions(tok, prompt_ids) == mpos and len(mpos) == 1
    seed = arm_seed("untrained_base", args.seed)
    stops = stop_token_ids(tok, base)
    t0 = time.time()
    samples, greedy = generate_injected(base, tok, dirs, prompt_ids, mpos[0], args.device, seed=seed,
                                        stop_ids=stops)
    g_texts = [g["text"] for g in greedy]
    tally = new_filter_stats()
    own = reread_cos(g_texts, dirs, base, tok, args.device, stats=tally)
    foil = reread_cos(g_texts, foil_dirs, base, tok, args.device)
    check = {
        "greedy_distinct_share": distinct_share(g_texts),
        "greedy_cos_own_mean": float(np.mean(own)),
        "greedy_cos_foil_mean": float(np.mean(foil)),
        "gap": float(np.mean(own) - np.mean(foil)),
        "seconds": time.time() - t0,
        "n_items": len(kept),
    }
    run.write_json(READOUTS_REL, {
        "config": {
            "model": [C.MODEL, C.MODEL_REVISION],
            "prompt_ids": [int(t) for t in prompt_ids],
            "marker_pos": mpos[0],
            "stop_ids": list(stops),
            "seed": seed,
            "gen": {"n_samples": C.N_SAMPLES, "temp": C.TEMP, "top_p": C.TOP_P, "top_k": C.TOP_K,
                    "min_p": C.MIN_P, "max_new": C.MAX_NEW, "min_new": C.MIN_NEW,
                    "gen_chunk": C.GEN_CHUNK},
            "injection_check": check,
            "norm_filter": norm_filter_record(tally),
        },
        "items": [{"i": it["i"], "greedy": dict(greedy[j], cos_own=float(own[j]), cos_foil=float(foil[j])),
                   "samples": samples[j]}
                  for j, it in enumerate(kept)],
    })
    write_provenance(run, {"untrained_base_injection_check": check}, stage="untrained_base")
    mark_stage(run, "untrained_base", chash,
               dict(check, stop_ids=list(stops), norm_filter=norm_filter_record(tally)), started=started)
    print(f"[untrained_base] {len(kept)} items; distinct greedy {check['greedy_distinct_share']:.2f}; greedy "
          f"cos own {check['greedy_cos_own_mean']:.3f} foil {check['greedy_cos_foil_mean']:.3f} gap "
          f"{check['gap']:.3f}; {check['seconds']:.0f}s", flush=True)
