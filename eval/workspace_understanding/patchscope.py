"""Stage `patchscope` (methodology §3.3): Patchscopes entity-description decoding of the layer-42
activation (`patch42`), with a no-injection floor (`patchfloor`) whose one sample set every item carries.
The arm's word rule and re-read are read against the floor's.

Prompt, patch, check and generators are eval/common/patchscope.py's, wrapped here to draw under this
package's sampling and stop rule (`model.generate_batches`)."""

import time
import numpy as np
from eval.common import patchscope as _ps
from eval.common.patchscope import (
    _patch_check,
    clean_norm_at,
    floor_records,
    refuse_failed_patch,
    refuse_identical_samples,
    resolve_template,
)
from eval.workspace_understanding import config as C
from eval.workspace_understanding.model import (
    arm_seed,
    centring_mean,
    direction,
    distinct_share,
    generate_batches,
    load_base,
    stop_token_ids,
)
from eval.common.runs import mark_stage, stage_done
from eval.workspace_understanding.runs import centring_mean_digest, stage_key


def generate_patched(mdl, tok, V, template, layer, device, seed, stop_ids, n_samples=C.N_SAMPLES,
                     gen_rows=C.PATCH_GEN_ROWS, alpha=C.PATCH_ALPHA):
    """`patchscope.generate_patched` under this package's generator (`model.generate_batches`)."""
    return _ps.generate_patched(generate_batches, mdl, tok, V, template, layer, device, seed, stop_ids, n_samples,
                                gen_rows, alpha)


def generate_floor(mdl, tok, template, device, seed, stop_ids, n_samples=C.N_SAMPLES, gen_rows=C.PATCH_GEN_ROWS):
    """`patchscope.generate_floor` under this package's generator."""
    return _ps.generate_floor(generate_batches, mdl, tok, template, device, seed, stop_ids, n_samples, gen_rows)


def unpatched_greedy(mdl, tok, template, device, stop_ids):
    """`patchscope.unpatched_greedy` under this package's generator."""
    return _ps.unpatched_greedy(generate_batches, mdl, tok, template, device, stop_ids)


def stage_patchscope(args, run):
    chash = stage_key("patchscope", args, run)
    if stage_done(run, "patchscope", chash) and not args.force:
        print("[patchscope] up to date")
        return
    started = time.time()

    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    H = np.load(run.file("activations/h_all.npz"))["h"]
    # the direction MAEMM is given, unit(h_42 - mu); row k is the k-th kept item's
    mu = centring_mean()
    V = np.stack([direction(H[it["i"], C.READ_LAYER], mu) for it in kept]).astype(np.float32)
    mdl, tok = load_base(args.device)
    template = resolve_template(tok)
    stops = stop_token_ids(tok, mdl)
    # one unpatched greedy for the stage (also the floor's greedy), cut at the same stop set
    unpatched = unpatched_greedy(mdl, tok, template, args.device, stops)
    arms = {}
    for arm in C.PATCH_ARMS:
        layer = C.PATCH_LAYERS.get(arm)  # None for the floor: no block is hooked
        rule = C.PATCH_RULES.get(arm)  # and no rule: nothing is written
        t0 = time.time()
        check = None
        if layer is None:
            cn = None
            shared = generate_floor(mdl, tok, template, args.device, arm_seed(arm, args.seed), stops)
            recs = floor_records(kept, shared, unpatched)
        else:
            n_check = min(C.PATCH_CHECK_ROWS, len(kept))
            rel, cos, ratio = _patch_check(mdl, template, V[:n_check], layer, args.device)
            check = {
                "items": [it["i"] for it in kept[:n_check]],
                "rel_delta": [round(x, 4) for x in rel],
                "cos_to_v": [round(x, 4) for x in cos],
                "norm_ratio": [round(x, 4) for x in ratio],
            }
            print(
                f"[patchscope] {arm} patch check ({rule}): ||dh||/||h|| {check['rel_delta']}, "
                f"cos(h_patched, v) {check['cos_to_v']}, ||h_patched||/||h|| {check['norm_ratio']}",
                flush=True,
            )
            refuse_failed_patch(arm, rel, cos)
            cn = clean_norm_at(mdl, template["ids"], template["position"], layer, args.device)
            samples, greedy = generate_patched(mdl, tok, V, template, layer, args.device,
                                               arm_seed(arm, args.seed), stops)
            recs = [
                {
                    "i": it["i"],
                    "greedy": greedy[k],
                    "samples": samples[k],
                    # the written norm over the placeholder's clean norm
                    "norm_ratio": float(C.PATCH_ALPHA),
                    "greedy_equals_unpatched": greedy[k]["text"].strip() == unpatched["text"].strip(),
                }
                for k, it in enumerate(kept)
            ]
            refuse_identical_samples(arm, recs)
        # recorded, never raised on (the floor's share is 1/n by construction)
        share = distinct_share([r["greedy"]["text"] for r in recs])
        arms[arm] = {
            "target_layer": layer,
            "rule": rule,
            "input": C.PATCH_INPUT.get(rule),
            "alpha": C.PATCH_ALPHA if rule == C.PATCH_RULE_TUNED else None,
            "prompt_id": template["prompt_id"],
            "seed": arm_seed(arm, args.seed),
            "unpatched_greedy": unpatched["text"],
            "clean_norm_at_layer": cn,
            "patch_check": check,
            "greedy_distinct_share": share,
            "share_equal_unpatched": float(np.mean([r["greedy_equals_unpatched"] for r in recs])),
            "distinct_sample_texts": len({s["text"] for r in recs for s in r["samples"]}),
            "seconds": time.time() - t0,
            "items": recs,
        }
        where = f"layer {layer}, {rule}, clean norm {cn:.2f}" if layer is not None else "no hook, shared samples"
        print(
            f"[patchscope] {arm}: {where}, greedy distinct {share:.2f}, equal-to-unpatched "
            f"{arms[arm]['share_equal_unpatched']:.2f}, {arms[arm]['distinct_sample_texts']} distinct sample "
            f"texts, {arms[arm]['seconds']:.0f}s",
            flush=True,
        )
    run.write_json(
        "rollouts/patchscope.json",
        {
            "template": template,
            "input": dict(C.PATCH_INPUT),
            "rules": dict(C.PATCH_RULES),
            "mu_sha256": centring_mean_digest(),
            "alpha": C.PATCH_ALPHA,
            "arms": arms,
            "stop_ids": list(stops),
        },
    )
    mark_stage(run, "patchscope", chash,
               {"stop_ids": list(stops),
                **{a: {k: v for k, v in arms[a].items() if k != "items"} for a in arms}},
               started=started)
