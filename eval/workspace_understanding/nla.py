"""Stage `nla` (methodology §3.4): the NLA verbalizer reads the raw h_42 of every kept item and writes eight
T=1 explanations plus one greedy, saved with their generated ids so the 64-token row is a prefix of the same
generation. Sharded by item (--shard k/n) into rollouts/nla/shard_<k>_of_<n>.json; shards.load_nla merges.
The verbalizer is loaded alone and given back in a `finally`."""

import time
import numpy as np
from eval.workspace_understanding import config as C
from eval.workspace_understanding import shards as S
from eval.common.nla import nla_reader as N
from eval.workspace_understanding.model import arm_seed, distinct_share, free_model
from eval.workspace_understanding.runs import stage_key, write_provenance


def open_verbalizer(device):
    """(model, tokenizer, prompt ids, marker position, checkpoint record): the pinned snapshot downloaded and
    checked, the merged model loaded, and the prompt rendered with the checkpoint's own chat template. The
    caller gives the model back (`free_model`)."""
    con = N.contract(N.load_sidecar())
    snapshot = N.download_checkpoint()
    shipped = N.check_checkpoint(snapshot)
    mdl, tok = N.load_verbalizer(device, snapshot)
    try:
        ids, mpos = N.prompt_ids(tok, con)
    except Exception:
        free_model(mdl)
        raise
    return mdl, tok, ids, mpos, shipped


def generation_summary(samples):
    """One arm's close rate (reported, never raised on), close rate within the 64-token prefix, mean
    generated length and share that ran into the cap."""
    flat = [s for row in samples for s in row]
    if not flat:
        return {"close_rate": None, "close_rate_trunc": None, "mean_generated_tokens": None, "capped_share": None}
    return {
        "close_rate": float(N.close_rate(flat)),
        "close_rate_trunc": float(np.mean([s["closed_trunc"] for s in flat])),
        "mean_generated_tokens": float(np.mean([s["n_tokens"] for s in flat])),
        "capped_share": float(np.mean([s["capped"] for s in flat])),
    }


def stage_nla(args, run):
    chash = stage_key("nla", args, run)
    k, n = S.parse_shard(getattr(args, "shard", None))
    started = time.time()
    if S.shard_done(run, "nla", chash, k, n) and not args.force:
        print(f"[nla] shard {k}/{n} up to date")
        return
    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    mine = S.shard_slice(kept, k, n)
    H42 = np.load(run.file("activations/h_all.npz"))["h"][:, C.READ_LAYER]
    mdl, tok, ids, mpos, shipped = open_verbalizer(args.device)
    t0 = time.time()
    try:
        H = np.stack([H42[it["i"]] for it in mine])  # raw: the verbalizer reads the activation, not a direction
        samples, greedy = N.generate_explanations(mdl, tok, H, ids, mpos, args.device,
                                                  seed=arm_seed("nla", args.seed))
    finally:
        mdl = free_model(mdl)
    # Distinct greedies across items: recorded, never raised on (a comparator's collapse is its own result).
    share = distinct_share([g["text"] for g in greedy])
    recs = [{"i": it["i"], "greedy": greedy[j], "samples": samples[j]} for j, it in enumerate(mine)]
    check = {
        "greedy_distinct_share": share,
        **generation_summary(samples),
        "seconds": time.time() - t0,
        "n_items": len(mine),
    }
    close_rate = check["close_rate"]
    run.write_json(
        S.nla_shard_rel(k, n),
        {
            "config": {
                "nla": [N.PINS.repo, N.PINS.revision],
                "checkpoint": shipped,
                "sidecar_sha256": dict(N.PINS.sidecar_sha256),
                "prompt_ids": ids,
                "marker_pos": mpos,
                "prompt_len": len(ids),
                "seed": arm_seed("nla", args.seed),
                "gen": {
                    "n_samples": C.N_SAMPLES,
                    "temp": C.TEMP,
                    "top_p": C.TOP_P,
                    "top_k": C.TOP_K,
                    "min_p": C.MIN_P,
                    "max_new": N.PINS.max_new,
                    "min_new": N.PINS.min_new,
                    "trunc": N.PINS.trunc,
                    "enable_thinking": N.PINS.enable_thinking,
                    "score_max_length": N.PINS.score_max_length,
                    "gen_chunk": N.PINS.gen_chunk,
                },
                "shard": [k, n],
                "injection_check": check,
            },
            "items": recs,
        },
    )
    write_provenance(run, {f"nla_shard_{k}_of_{n}": check, "nla_revision": N.PINS.revision,
                           "nla_checkpoint": shipped}, stage="nla")
    print(
        f"[nla] shard {k}/{n}: {len(mine)} items; distinct greedy {share:.2f}; close rate {close_rate}; "
        f"mean generated {check['mean_generated_tokens']}; {check['seconds']:.0f}s",
        flush=True,
    )
    if close_rate is not None and close_rate < N.PINS.min_close_rate:
        print(f"[nla] NOTE: the tags closed in {close_rate:.2f} of the samples, under {N.PINS.min_close_rate}; "
              "an unclosed sample is read whole, and the rate is reported", flush=True)
    S.mark_shard(run, "nla", chash, k, n, check, started=started)
