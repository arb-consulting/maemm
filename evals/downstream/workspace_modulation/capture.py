"""One clean forward per item on the base model: the layer-42 residual at every carrier cell, every layer
at those cells (for the lens), norms along the prompt, and the greedy compliance diagnostic. The inverter
is never loaded."""

import os, time
import numpy as np
from evals.downstream.common.matcher import normalise, whole_word_hit
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation.items import cell_table
from evals.downstream.workspace_modulation.runs import stage_key, write_provenance


def compliance_flags(text, carrier, forms):
    """Verbatim carrier (normalised whitespace and quotes) anywhere in the continuation; a target as a
    whole word anywhere in it."""
    copies = normalise(carrier) in normalise(text)
    return copies, whole_word_hit(text, forms)


def norm_flags(norms, cells, mult):
    med = float(np.median(norms[1:])) if len(norms) > 1 else float(norms[0])
    return {
        "median": med,
        "cells": {
            str(p): {
                "norm": float(norms[p]),
                "ratio": float(norms[p]) / med if med else None,
                "outlier": bool(med and norms[p] > mult * med),
            }
            for p in cells
        },
    }


def user_only_ids(tok, user):
    """The item's user turn plus the assistant generation prefix, as a list of token ids (rendered, then
    encoded; thinking disabled as in items.render_chat)."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return [int(i) for i in tok.encode(text, add_special_tokens=False)]


def _greedy_user_only(base, tok, user, max_new, device):
    """The compliance diagnostic's continuation: the untrained base model's own greedy answer to the user
    turn, with no injection and no research prompt."""
    import torch

    ids = user_only_ids(tok, user)
    with torch.no_grad():
        x = torch.tensor([ids], device=device)
        g = base.generate(
            x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.0,
        )
    return tok.decode(g[0, len(ids) :], skip_special_tokens=True)


def stage_capture(args, run):
    chash = stage_key("capture", args, run)
    if stage_done(run, "capture", chash) and not args.force:
        print("[capture] up to date")
        return
    started = time.time()
    from evals.downstream.common.model_io import capture, load_base

    doc = run.read_json("data/items.json")
    kept = [x for x in doc["items"] if not x["excluded"]]
    table = cell_table(kept)
    row_of = {(i, p): r for r, (i, p, _) in enumerate(table)}
    base, tok = load_base(args.device, C.MODEL, C.MODEL_REVISION)
    write_provenance(run, {"generation_config": base.generation_config.to_dict()}, stage="capture")
    H42 = np.zeros((len(table), C.D_MODEL), dtype=np.float32)
    norms, comp, t0 = {}, {}, time.time()
    os.makedirs(run.sub("activations/cells"), exist_ok=True)
    for n, it in enumerate(kept):
        i = it["i"]
        h42_all = capture(base, it["ids"], args.device, [C.READ_LAYER])[0]  # [n_pos, d]
        for p in it["carrier_cells"]:
            H42[row_of[(i, p)]] = h42_all[p]
        norms[str(i)] = norm_flags(np.linalg.norm(h42_all, axis=-1), it["carrier_cells"], C.NORM_MULT)
        h_all = capture(
            base, it["ids"], args.device, list(range(C.N_LAYERS)), positions=it["carrier_cells"]
        )  # [64, n_carrier, d]
        np.savez_compressed(run.file(f"activations/cells/{i}.npz"), h_all=np.transpose(h_all, (1, 0, 2)))
        text = _greedy_user_only(base, tok, it["user"], C.COMPLIANCE_MAX_NEW, args.device)
        copies, names = compliance_flags(text, it["carrier"], it["forms"])
        comp[str(i)] = {"text": text, "copies_carrier": copies, "names_target": names}
        if n % 20 == 19:
            print(f"  [capture] {n + 1}/{len(kept)}", flush=True)
    np.savez_compressed(run.file("activations/h42.npz"), h=H42)
    run.write_json("activations/cell_table.json", table)
    run.write_json("activations/norms.json", norms)
    run.write_json("data/compliance.json", comp)
    by_cond = {}
    for it in kept:
        c = comp[str(it["i"])]
        d = by_cond.setdefault(it["instruction"], {"n": 0, "copies": 0, "names": 0})
        d["n"] += 1
        d["copies"] += c["copies_carrier"]
        d["names"] += c["names_target"]
    print(
        f"[capture] {len(kept)} items, {len(table)} cells in {time.time() - t0:.0f}s; compliance {by_cond}", flush=True
    )
    mark_stage(
        run,
        "capture",
        chash,
        {"n_items": len(kept), "n_cells": len(table), "compliance": by_cond},
        started=started,
    )
