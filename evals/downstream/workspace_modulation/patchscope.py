"""Patchscopes at the two read positions (methodology "Readers"), through evals/downstream/common/patchscope.py.

`patch42`: the clean base continues the entity-description prompt with the placeholder's block-42 residual
replaced, during prefill, by `alpha * ||h_42|| * unit(h42 - mu)` (the vector MAEM is injected with); one
greedy and N_SAMPLES samples per read cell under the headline arm's decoding and stop set. `patchfloor`: the
same prompt with no patch, generated once and carried by every item (its mean-cell readout is its final
one). A patch check on the first PATCH_CHECK_ROWS cells runs before any generation. Output:
rollouts/patchscope.json, each arm in the per-arm layout `rollouts.cells_by_pos` reads."""

import time

import numpy as np

from evals.downstream.common import background as BG
from evals.downstream.common import patchscope as PS
from evals.downstream.common.patchscope import (
    _patch_check,
    clean_norm_at,
    refuse_failed_patch,
    refuse_identical_samples,
    resolve_template,
)
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import mean_cell as M
from evals.downstream.workspace_modulation.runs import centring_mean_digest, stage_key, write_provenance


def patch_seed(arm, seed):
    """One Patchscopes arm's own base seed: the run's seed plus the arm's offset, in blocks of SHARD_SEEDS as
    `rollouts.arm_seed` numbers the generation arms, so no two arms of the package draw one stream."""
    return int(seed) + C.PATCH_SEED_OFFSET[arm] * C.SHARD_SEEDS


def _sampling():
    """The decoding both arms draw under, as the shared generator takes it: the headline arm's own."""
    return {k: C.PATCH_GEN[k] for k in ("temp", "top_p", "top_k", "min_p", "max_new", "min_new")}


def generate_batches(gen_model, tok, n, device, n_samples, seed, greedy, gen_chunk, batch_fn, hook_fn=None,
                     stop_ids=None):
    """`model_io.generate_batches` under this package's decoding and stop set, each call seeded
    `seed * 1000 +` the grid index of its first row."""
    from evals.downstream.common import model_io

    from evals.downstream.workspace_modulation.rollouts import records

    return model_io.generate_batches(gen_model, tok, n, device, n_samples, seed, greedy, gen_chunk, _sampling(),
                                     batch_fn, hook_fn, records(stop_ids))


def generate_patched(mdl, tok, V, template, layer, device, seed, stop_ids, n_samples=C.PATCH_GEN["n_samples"],
                     gen_rows=C.PATCH_GEN_ROWS, alpha=C.PATCH_ALPHA):
    """`patchscope.generate_patched` under this package's generator: one row of `V` per read cell."""
    return PS.generate_patched(generate_batches, mdl, tok, V, template, layer, device, seed, stop_ids, n_samples,
                               gen_rows, alpha)


def generate_floor(mdl, tok, template, device, seed, stop_ids, n_samples=C.PATCH_GEN["n_samples"],
                   gen_rows=C.PATCH_GEN_ROWS):
    """`patchscope.generate_floor` under this package's generator: the floor's one shared sample set."""
    return PS.generate_floor(generate_batches, mdl, tok, template, device, seed, stop_ids, n_samples, gen_rows)


def unpatched_greedy(mdl, tok, template, device, stop_ids):
    """`patchscope.unpatched_greedy` under this package's generator: the floor's greedy, and what every
    patched greedy is compared against."""
    return PS.unpatched_greedy(generate_batches, mdl, tok, template, device, stop_ids)


def patch_rows(table, kept):
    """[(row, i, pos, band)] of every read cell of every kept item, in READ_BANDS order per item: the cells
    `patch42` is generated at, addressed by their row in the concatenated activation store."""
    from evals.downstream.workspace_modulation.rollouts import row_index

    row_of = row_index(table)
    out = []
    for it in kept:
        for band in C.READ_BANDS:
            for p in M.band_cells(band, it):
                out.append((row_of[(int(it["i"]), int(p))], int(it["i"]), int(p), band))
    return out


def floor_items(kept, shared, greedy):
    """The floor's per-item records in the per-arm layout: one cell per item, at its final period, carrying
    the shared texts (`patchscope.floor_records`, which refuses unless every item carries exactly them)."""
    recs = PS.floor_records(kept, shared, greedy)
    pos = {int(it["i"]): M.final_pos(it) for it in kept}
    return [
        {
            "i": r["i"],
            "cells": [
                {
                    "pos": pos[int(r["i"])],
                    "band": C.FINAL_BAND,
                    "greedy": r["greedy"],
                    "samples": r["samples"],
                    "norm_ratio": r["norm_ratio"],
                    "greedy_equals_unpatched": r["greedy_equals_unpatched"],
                }
            ],
        }
        for r in recs
    ]


def load_patchscope(run):
    """The stage's output. A missing file raises rather than reading as a reader with no coverage."""
    if not run.exists(C.PATCH_REL):
        raise RuntimeError(f"{C.PATCH_REL} is missing — run the patchscope stage")
    return run.read_json(C.PATCH_REL)


def patch_pos(it, p, arm):
    """The cell `arm`'s readout for `p` is read from: itself, or the final period for the floor at the mean cell."""
    if arm == C.PATCH_FLOOR and p == C.MEAN_POS:
        return M.band_cells(C.PATCH_FLOOR_FROM_BAND, it)[0]
    return p


def stage_patchscope(args, run):
    chash = stage_key("patchscope", args, run)
    if stage_done(run, "patchscope", chash) and not args.force:
        print("[patchscope] up to date")
        return
    started = time.time()
    from evals.downstream.common.model_io import direction, load_base, stop_token_ids
    from evals.downstream.workspace_modulation import rollouts as RO

    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    table = M.full_table(run)
    rows = patch_rows(table, kept)
    # V's k-th row is the k-th read cell's unit(h42 - mu), as `rollouts` and the search build it.
    H = M.load_h(run)
    mu = BG.load_centring_mean()
    V = np.stack([direction(H[r], mu) for r, _i, _p, _b in rows]).astype(np.float32)
    # the stop set every arm is cut at (the base's shipped generation config)
    mdl, tok = load_base(args.device, C.MODEL, C.MODEL_REVISION)
    stops = stop_token_ids(tok, RO.generation_config(C.MODEL, C.MODEL_REVISION))
    template = resolve_template(tok)
    arms = {}

    arm, t0 = C.PATCH_ARM, time.time()
    n_check = min(C.PATCH_CHECK_ROWS, len(rows))
    rel, cos, ratio = _patch_check(mdl, template, V[:n_check], C.PATCH_LAYER, args.device)
    check = {
        "cells": [[i, p] for _r, i, p, _b in rows[:n_check]],
        "rel_delta": [round(x, 4) for x in rel],
        "cos_to_v": [round(x, 4) for x in cos],
        # the patched norm over the clean one, as measured after the write: alpha when the rule holds
        "norm_ratio": [round(x, 4) for x in ratio],
        "min_cos": C.PATCH_CHECK_MIN_COS,
        "min_rel_delta": C.PATCH_CHECK_MIN_REL_DELTA,
    }
    print(
        f"[patchscope] {arm} patch check ({C.PATCH_RULE}): ||dh||/||h|| {check['rel_delta']}, "
        f"cos(h_patched, v) {check['cos_to_v']}, ||h_patched||/||h|| {check['norm_ratio']}",
        flush=True,
    )
    refuse_failed_patch(arm, rel, cos)
    # one unpatched greedy, after the check: the floor's greedy and the reference of `greedy_equals_unpatched`
    unpatched = unpatched_greedy(mdl, tok, template, args.device, stops)
    cn = clean_norm_at(mdl, template["ids"], template["position"], C.PATCH_LAYER, args.device)
    samples, greedy = generate_patched(mdl, tok, V, template, C.PATCH_LAYER, args.device, patch_seed(arm, args.seed),
                                       stops)
    by_i = {}
    for k, (_r, i, p, band) in enumerate(rows):
        by_i.setdefault(i, []).append(
            {
                "pos": p,
                "band": band,
                "greedy": greedy[k],
                "samples": samples[k],
                "norm_ratio": float(C.PATCH_ALPHA),
                "greedy_equals_unpatched": greedy[k]["text"].strip() == unpatched["text"].strip(),
            }
        )
    items = [{"i": it["i"], "cells": by_i[int(it["i"])]} for it in kept]
    cells = [c for x in items for c in x["cells"]]
    refuse_identical_samples(arm, [{"i": x["i"], "samples": c["samples"]} for x in items for c in x["cells"]])
    arms[arm] = {
        "config": {
            "target_layer": C.PATCH_LAYER,
            "rule": C.PATCH_RULE,
            "input": C.PATCH_INPUT[C.PATCH_RULE],
            "alpha": C.PATCH_ALPHA,
            "seed": patch_seed(arm, args.seed),
            "clean_norm_at_layer": cn,
            "patch_check": check,
            "unpatched_greedy": unpatched["text"],
            # low distinctness is reported, never raised on
            "greedy_distinct_share": RO.distinct_share([c["greedy"]["text"] for c in cells]),
            "share_equal_unpatched": float(np.mean([c["greedy_equals_unpatched"] for c in cells])),
            "distinct_sample_texts": len({s["text"] for c in cells for s in c["samples"]}),
            "seconds": time.time() - t0,
        },
        "items": items,
    }

    arm, t0 = C.PATCH_FLOOR, time.time()
    shared = generate_floor(mdl, tok, template, args.device, patch_seed(arm, args.seed), stops)
    items = floor_items(kept, shared, unpatched)
    arms[arm] = {
        "config": {
            "target_layer": None,  # no block is hooked
            "rule": None,  # and nothing is written
            "input": None,
            "alpha": None,
            "seed": patch_seed(arm, args.seed),
            "clean_norm_at_layer": None,
            "patch_check": None,
            "from_band_at_mean": C.PATCH_FLOOR_FROM_BAND,
            "unpatched_greedy": unpatched["text"],
            "greedy_distinct_share": RO.distinct_share([x["cells"][0]["greedy"]["text"] for x in items]),
            "share_equal_unpatched": 1.0,
            "distinct_sample_texts": len({t["text"] for t in shared}),
            "seconds": time.time() - t0,
        },
        "items": items,
    }
    for a in C.PATCH_ARMS:
        cfg = arms[a]["config"]
        where = (f"layer {cfg['target_layer']}, {cfg['rule']}, clean norm {cfg['clean_norm_at_layer']:.2f}"
                 if cfg["target_layer"] is not None else "no hook, shared samples")
        print(
            f"[patchscope] {a}: {where}, greedy distinct {cfg['greedy_distinct_share']:.2f}, equal-to-unpatched "
            f"{cfg['share_equal_unpatched']:.2f}, {cfg['distinct_sample_texts']} distinct sample texts, "
            f"{cfg['seconds']:.0f}s",
            flush=True,
        )
    run.write_json(
        C.PATCH_REL,
        {
            "config": {
                "template": template,
                "model": [C.MODEL, C.MODEL_REVISION],
                "layer": C.PATCH_LAYER,
                "rule": C.PATCH_RULE,
                "input": C.PATCH_INPUT[C.PATCH_RULE],
                "alpha": C.PATCH_ALPHA,
                "mu_sha256": centring_mean_digest(),
                "gen": dict(C.PATCH_GEN),
                "gen_rows": C.PATCH_GEN_ROWS,
                "seeding": C.PATCH_SEEDING,
                "unpatched_greedy": unpatched["text"],
                "stop_ids": list(stops),
                "upstream": dict(chash.upstream),
            },
            "arms": arms,
        },
    )
    write_provenance(run, {"patchscope_check": check}, stage="patchscope")
    mark_stage(run, "patchscope", chash,
               {"stop_ids": list(stops), **{a: arms[a]["config"] for a in C.PATCH_ARMS}}, started=started)
