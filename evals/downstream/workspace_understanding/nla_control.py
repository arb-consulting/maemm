"""Stage `nla_control` (methodology §3.5): the NLA verbalizer on two control activations of the same prompt,
`mid` (h_42 at token ⌊p/2⌋) and `mean` (mean over positions 1..p), captured on the clean base, which is
given back before the verbalizer loads. Sharded by item like `nla`; shards.load_controls merges."""

import time
import numpy as np
from evals.downstream.workspace_understanding import shards as S
from evals.downstream.workspace_understanding import config as C
from evals.downstream.common.nla import nla_reader as N
from evals.downstream.workspace_understanding.model import arm_seed, free_model, load_base
from evals.downstream.workspace_understanding.nla import generation_summary, open_verbalizer
from evals.downstream.workspace_understanding.model import distinct_share
from evals.downstream.workspace_understanding.runs import stage_key, write_provenance


def control_positions(pos):
    """(mid, (lo, hi)) for a readout at `pos`: token ⌊p/2⌋ and the inclusive window 1..p."""
    mid = pos // 2
    if mid < 1 or pos < 2:
        raise ValueError(f"readout position {pos} leaves no interior control position")
    return mid, (1, pos)


def distinct_input_share(V, tol=1e-4):
    """Share of rows of V that do not duplicate an earlier row (templated prompts share mid activations)."""
    V = np.asarray(V, dtype=np.float64)
    n = len(V)
    if n == 0:
        return 1.0
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    S = Vn @ Vn.T
    dup = 0
    for a in range(1, n):
        if (S[a, :a] > 1 - tol).any():
            dup += 1
    return (n - dup) / n


def relative_shares(greedy_texts, V):
    """(distinct greedy share, distinct input share): a control's greedy share is read against its input
    share. Recorded, never raised on."""
    return distinct_share(greedy_texts), distinct_input_share(V)


def capture_controls(base, ids, pos, device):
    """{"mid": [d], "mean": [d]} from one clean forward of the prompt ids, hooking layer READ_LAYER only."""
    import torch
    from maemm.inject import get_layer

    mid, (lo, hi) = control_positions(pos)
    store = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        store["h"] = h[0].detach().float().cpu().numpy()

    hd = get_layer(base, C.READ_LAYER).register_forward_hook(hook)
    try:
        with torch.no_grad():
            base.model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        hd.remove()
    h = store["h"]
    return {"mid": h[mid], "mean": h[lo : hi + 1].mean(0), "mid_pos": mid, "mean_window": [lo, hi]}


def stage_nla_control(args, run):
    chash = stage_key("nla_control", args, run)
    k, n = S.parse_shard(getattr(args, "shard", None))
    started = time.time()
    if S.shard_done(run, "nla_control", chash, k, n) and not args.force:
        print(f"[nla_control] shard {k}/{n} up to date")
        return
    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    mine = S.shard_slice(kept, k, n)
    base, _tok = load_base(args.device)
    t0 = time.time()
    try:
        caps = [capture_controls(base, it["ids"], it["readout_pos"], args.device) for it in mine]
    finally:
        base = free_model(base)
    norms = {kind: [float(np.linalg.norm(c[kind])) for c in caps] for kind in C.POSITION_CONTROLS}
    np.savez_compressed(
        run.file(S.control_vectors_rel(k, n)),
        i=np.array([it["i"] for it in mine]),
        **{kind: np.stack([c[kind] for c in caps]).astype(np.float32) for kind in C.POSITION_CONTROLS},
    )
    mdl, tok, ids, mpos, shipped = open_verbalizer(args.device)
    recs = [{"i": it["i"], "mid_pos": c["mid_pos"], "mean_window": c["mean_window"], "readout_pos": it["readout_pos"]} for it, c in zip(mine, caps)]
    checks, written = {}, {}
    try:
        for kind in C.POSITION_CONTROLS:
            Hk = np.stack([c[kind] for c in caps])  # raw: the verbalizer reads the activation itself
            written[kind] = N.generate_explanations(mdl, tok, Hk, ids, mpos, args.device,
                                                    seed=arm_seed(f"nla_{kind}", args.seed))
    finally:
        mdl = free_model(mdl)
    for kind in C.POSITION_CONTROLS:
        samples, greedy = written[kind]
        share, dshare = relative_shares([g["text"] for g in greedy], np.stack([c[kind] for c in caps]))
        checks[kind] = {
            "greedy_distinct_share": share,
            "distinct_input_share": dshare,
            **generation_summary(samples),
            "norm_median": float(np.median(norms[kind])),
        }
        for r, g, ss, nv in zip(recs, greedy, samples, norms[kind]):
            r[kind] = {"greedy": g, "samples": ss, "norm": nv}
        print(f"[nla_control] {kind} shard {k}/{n}: {len(mine)} items; distinct greedy {share:.2f} of {dshare:.2f} distinct inputs; close rate {checks[kind]['close_rate']}", flush=True)
    doc = {
        "config": {
            "nla": [N.PINS.repo, N.PINS.revision],
            "checkpoint": shipped,
            "kinds": list(C.POSITION_CONTROLS),
            "seeds": {kind: arm_seed(f"nla_{kind}", args.seed) for kind in C.POSITION_CONTROLS},
            "prompt_ids": ids,
            "marker_pos": mpos,
            "shard": [k, n],
            "injection_check": checks,
            "seconds": time.time() - t0,
        },
        "items": recs,
    }
    run.write_json(S.control_shard_rel(k, n), doc)
    write_provenance(run, {f"nla_control_shard_{k}_of_{n}": checks}, stage="nla_control")
    S.mark_shard(run, "nla_control", chash, k, n, {"checks": checks, "n_items": len(mine)}, started=started)
