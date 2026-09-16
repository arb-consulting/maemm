"""Product `top1_act`: does the corpus search's TOP-1 window actually make the feature fire?

    <root>/base/<base>/sae/<sae>/top1_act/<set>/  top1_act.jsonl, summary.json

For every `sae` held-out feature of `<set>`, take the window the corpus scan selected by COSINE
against `unit(W_enc[:, f])` at the largest corpus size (`scan/<set>/topk.jsonl`, rank 0: doc,
window start, argmax position, cos) and report that window's PRE-GATE activation of the SAME
feature -- `relu((h - b_dec) @ W_enc[:, f] + b_enc[f])` at the read layer -- as the max over the
window's non-sink positions and as the value at the cosine argmax token.

Two independent sources for that activation, both written for every row:

  * `join_*`: the feature's own `examples/<feature>.jsonl`, which the SAME `scan` pass wrote --
    the top `SAE_TOP` windows by activation plus the four sampled activation bins. A cosine top-1
    window is in that file only when it happens to also be an activation example, which is exactly
    the question this product exists to quantify, so `joined` is false for most rows.
  * `fwd_*`: one forward here, on the clean base, of the scan geometry (the window's ids with the
    sink prepended and dropped). Every row is forwarded, not only the unjoined ones: 512 windows of
    <= 65 tokens is ~4 batches next to a model load, so paying for all of them turns the join into
    a CHECKED path instead of an assumed one. `summary.json` carries the max |join - fwd| over the
    joined rows; the examples' per-token `acts` are f16 payload rounded to 4 dp, so agreement is
    expected at ~1e-2 absolute on large activations, not bitwise.

The gate is the checkpoint's own learned BatchTopK `threshold` (`common.load_sae`), the same gate
`stats`, `score`, `repo_examples` and `gcg` use. NEVER loads a MAEMM.
"""

from __future__ import annotations

import os
import time

import numpy as np

import precompute.common as C

BATCH = 128  # windows per forward; 512 rows is ~4 batches, the model load dominates either way


def _window_len(docs, doc: int, start: int) -> int:
    """The scan window length at (doc, start), from the ONE geometry helper every pass uses.

    Raises rather than guessing: a start that `windows_of` does not produce means the topk and the
    corpus on this root are not from the same run.
    """
    wins = C.windows_of(int(docs[doc]["len"]))
    for s, ln in wins:
        if s == start:
            return ln
    raise AssertionError(
        f"doc {doc} ({docs[doc]['len']} tokens) has no scan window starting at {start}; "
        f"its window starts are {[s for s, _ in wins]}"
    )


def _forward_acts(model, read_layer, sae, rows, feats, sink, pad_id):
    """(act [B, T] fp32 pre-gate for each row's OWN feature, keep [B, T]) for a batch of windows.

    Same batch shape as `scan._forward` / `stats` pass A: sink at column 0, the window after it,
    pad to the widest row, and the sink dropped from `keep`.
    """
    import torch

    width = 1 + max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    am = np.zeros((len(rows), width), dtype=np.int64)
    ids[:, 0] = sink
    am[:, 0] = 1
    for i, r in enumerate(rows):
        ids[i, 1 : 1 + len(r)] = r
        am[i, 1 : 1 + len(r)] = 1
    batch = {
        "input_ids": torch.from_numpy(ids).cuda(),
        "attention_mask": torch.from_numpy(am).cuda(),
    }
    h, mask = C.read_resid(model, read_layer, batch, pool="all")  # [B, T, d] fp32
    keep = mask.clone()
    keep[:, 0] = False
    # The ONE activation formula (common.sae_encode, the same call scan/score/gcg make), over the
    # batch's features; each row then keeps the column of its OWN feature.
    a = C.sae_encode(sae, h, feats)  # [B, T, B]
    ar = torch.arange(len(feats), device=a.device)
    act = a[ar, :, ar]  # [B, T]
    assert act.shape == (len(feats), h.shape[1]), f"per-row activation is {tuple(act.shape)}"
    return act.masked_fill(~keep, 0.0), keep


def run(cfg, args):
    import torch

    base, root, set_name = args["base"], args["root"], args["heldout"]
    assert base, "product top1_act needs --base"
    spec = cfg["bases"][base]
    read_layer, d = spec["read_layer"], spec["d"]
    batch_rows = int(args.get("batch") or BATCH)
    sae_keys = [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]
    assert len(sae_keys) == 1, f"base {base} has {len(sae_keys)} SAEs in config, expected exactly 1"
    sae_key = sae_keys[0]

    ex_dir = f"{C.sae_dir(sae_key, root)}/examples"
    out = f"{C.sae_dir(sae_key, root)}/top1_act/{set_name}"
    assert args.get("force") or not os.path.exists(out), (
        f"{out} already exists; refusing to overwrite without --force"
    )
    assert os.path.isdir(ex_dir), f"{ex_dir} is missing: run `--product scan --base {base}` first"

    toks, docs = C.load_corpus(base, root)
    sizes = C.corpus_sizes(docs)
    size = sizes[-1]
    ids_rows = C.read_jsonl(f"{C.heldout_dir(base, set_name, root)}/ids.jsonl")
    sae_rows = [r for r in ids_rows if r["family"] == "sae"]
    assert sae_rows, f"held-out set {set_name} on {base} has no sae family"
    by_row = {r["row"]: r for r in sae_rows}

    top1 = {}
    for r in C.read_jsonl(f"{C.scan_dir(base, set_name, root)}/topk.jsonl"):
        if r["size"] == size and r["row"] in by_row:
            assert r["top"], f"row {r['row']} has an empty top-k at size {size}M"
            top1[r["row"]] = r["top"][0]
    missing = sorted(set(by_row) - set(top1))
    assert not missing, f"topk.jsonl has no size-{size}M line for rows {missing[:8]} (+{len(missing)})"

    print(
        f"[top1_act] {len(sae_rows)} sae features on {base}/{sae_key}, "
        f"cosine top-1 at {size}M over {len(docs)} docs",
        flush=True,
    )

    # ---- the examples join, on the volume (the examples are ~57-60 MB per base and stay here) ----
    recs = []
    n_joined = n_top = 0
    for r in sae_rows:
        row, feat = r["row"], int(r["id"])
        doc, start, argmax, cos = top1[row]
        wlen = _window_len(docs, doc, start)
        assert 0 <= argmax < wlen, f"row {row}: cos argmax {argmax} outside its window of {wlen}"
        hit = None
        for e in C.read_jsonl(f"{ex_dir}/{feat}.jsonl"):
            if e["doc"] == doc and e["start"] == start:
                assert e["len"] == wlen, (
                    f"row {row}: examples say window (doc {doc}, start {start}) is {e['len']} "
                    f"tokens, windows_of says {wlen}"
                )
                hit = e
                break
        if hit is not None:
            n_joined += 1
            n_top += hit["kind"] == "top"
        recs.append(
            {
                "row": row,
                "feature": feat,
                "stratum": r["stratum"],
                "density": r["density"],
                "size": size,
                "top1_cos": cos,
                "doc": doc,
                "start": start,
                "len": wlen,
                "argmax": argmax,
                "joined": hit is not None,
                "join_kind": hit["kind"] if hit else None,
                "join_act_max": hit["max_act"] if hit else None,
                "join_act_at_argmax": (hit["acts"][argmax] if hit else None),
                "join_argmax": hit["argmax"] if hit else None,
            }
        )
    print(
        f"[top1_act] examples join: {n_joined}/{len(recs)} windows found "
        f"({n_top} in the activation top-128, {n_joined - n_top} in a sampled bin)",
        flush=True,
    )

    # ---- the forward, for EVERY row ----
    model, tok = C.load_base(cfg, base)
    sae = C.load_sae(C.sae_path(cfg, sae_key), d, device="cuda", dtype=torch.float32)
    sink = C.sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else sink
    gate = sae.threshold
    t0 = time.time()
    for c0 in range(0, len(recs), batch_rows):
        chunk = recs[c0 : c0 + batch_rows]
        wins, feats = [], []
        for e in chunk:
            off = int(docs[e["doc"]]["offset"]) + e["start"]
            wins.append(np.asarray(toks[off : off + e["len"]], dtype=np.int64))
            feats.append(e["feature"])
        act, keep = _forward_acts(model, read_layer, sae, wins, feats, sink, pad_id)
        amax, aarg = act.max(dim=1)
        amax, aarg, act = amax.cpu().numpy(), aarg.cpu().numpy(), act.cpu().numpy()
        for i, e in enumerate(chunk):
            e["act_max"] = round(float(amax[i]), 5)
            e["act_argmax"] = int(aarg[i]) - 1  # column 0 is the sink
            e["act_at_argmax"] = round(float(act[i, 1 + e["argmax"]]), 5)
            e["gate"] = gate
            e["passes_gate"] = bool(amax[i] > gate)
        print(f"[top1_act] forwarded {min(c0 + batch_rows, len(recs))}/{len(recs)}", flush=True)
    fwd_s = time.time() - t0

    # ---- join vs forward: the check the free extra forwards buy ----
    pairs = [(e["join_act_max"], e["act_max"]) for e in recs if e["joined"]]
    d_max = max((abs(a - b) for a, b in pairs), default=0.0)
    pairs_at = [
        (e["join_act_at_argmax"], e["act_at_argmax"])
        for e in recs
        if e["joined"] and e["join_act_at_argmax"] is not None
    ]
    d_at = max((abs(a - b) for a, b in pairs_at), default=0.0)
    n_pass = sum(e["passes_gate"] for e in recs)
    print(
        f"[top1_act] gate {gate:.4f}: {n_pass}/{len(recs)} pass | join-vs-forward max |d| "
        f"{d_max:.4f} (max_act) / {d_at:.4f} (at argmax) over {len(pairs)} joined rows",
        flush=True,
    )

    summary = {
        "base": base,
        "sae": sae_key,
        "set": set_name,
        "corpus_size_m": size,
        "n_features": len(recs),
        "n_joined": n_joined,
        "n_joined_top": n_top,
        "n_forward_only": len(recs) - n_joined,
        "gate": gate,
        "n_pass_gate": int(n_pass),
        "frac_pass_gate": round(n_pass / len(recs), 6),
        "join_vs_forward_max_abs_diff": round(float(d_max), 6),
        "join_vs_forward_max_abs_diff_at_argmax": round(float(d_at), 6),
        "forward_seconds": round(fwd_s, 1),
        "by_stratum": {
            str(q): {
                "n": sum(1 for e in recs if e["stratum"] == q),
                "n_pass_gate": sum(1 for e in recs if e["stratum"] == q and e["passes_gate"]),
            }
            for q in sorted({e["stratum"] for e in recs})
        },
    }

    inputs = {
        "scan": C.scan_dir(base, set_name, root),
        "heldout": C.heldout_dir(base, set_name, root),
        "examples": ex_dir,
        "sae": sae_key,
        "features": len(recs),
        "corpus_size": f"{size}M",
    }
    with C.outdir(out, args, inputs=inputs) as od:
        od.write_jsonl("top1_act.jsonl", recs)
        od.write_json("summary.json", summary)
        od.note(
            "one row per sae held-out feature: the COSINE top-1 corpus window at the largest "
            f"corpus size ({size}M) and that window's PRE-GATE activation of the SAME feature, "
            "relu((h - b_dec) @ W_enc[:, f] + b_enc[f]) at the read layer over the window's "
            f"non-sink positions ({C.SCAN_BLOCK}/{C.SCAN_STRIDE} scan geometry, sink prepended "
            "and dropped, clean base)"
        )
        od.note(
            "`act_max` / `act_at_argmax` / `act_argmax` are THIS product's forward, computed for "
            "every row. `join_*` are the same window read out of the scan's own "
            "`examples/<feature>.jsonl` when it is there; `argmax` is the COSINE argmax from the "
            "scan and `act_argmax` the ACTIVATION argmax of the same window"
        )
        od.note(
            f"examples join: {n_joined}/{len(recs)} rows found ({n_top} in the activation top-128, "
            f"{n_joined - n_top} in a sampled activation bin); the other {len(recs) - n_joined} "
            "cosine top-1 windows are not activation examples of their own feature at all"
        )
        od.note(
            f"join vs forward on the {len(pairs)} joined rows: max |d| {d_max:.4f} on max_act and "
            f"{d_at:.4f} at the cosine argmax. The examples' per-token `acts` come from an f16 "
            "payload rounded to 4 dp and `max_act` from fp32 rounded to 4 dp, so this bounds "
            "rounding, not a protocol difference"
        )
        od.note(
            f"gate = the checkpoint's learned BatchTopK threshold {gate:.4f} (common.load_sae), the "
            f"same gate stats/score/repo_examples/gcg use; {n_pass}/{len(recs)} windows have "
            "act_max > gate"
        )
        od.note("NEVER loads a MAEMM; the SAE and the clean base only")

    return {
        "top1_act": out,
        "features": len(recs),
        "joined": n_joined,
        "forward_only": len(recs) - n_joined,
        "frac_pass_gate": summary["frac_pass_gate"],
        "gate": gate,
        "forward_seconds": round(fwd_s, 1),
    }
