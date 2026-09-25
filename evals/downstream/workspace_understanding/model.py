"""This package's seam onto evals/downstream/common/model_io.py, and stages `capture` and `rollouts` (methodology §4).

`load_base` is the clean Qwen3.6-27B (every capture, re-read and the untrained-base ablation);
`load_inverter` is the fine-tune under evaluation, which only generates MAEM's rollouts. Every wrapper
resolves `_io.<name>` at call time."""

import re, time
import numpy as np
from evals.downstream.common import background as _bg
from evals.downstream.common import model_io as _io
from evals.downstream.workspace_understanding import config as C
from evals.downstream.common import matcher as M
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_understanding.runs import stage_key, write_provenance


def load_base(device):
    return _io.load_base(device, C.MODEL, C.MODEL_REVISION)


def load_inverter(device):
    return _io.load_inverter(device, C.INVERTER, C.INVERTER_REVISION)


def free_model(mdl):
    return _io.free_model(mdl)


def capture(base, ids, device, layers, positions=None):
    return _io.capture(base, ids, device, layers, positions=positions)


def reread_cos(texts, dirs, base, tok, device, max_length=None, stats=None):
    """The unfiltered re-read cosine on the clean base (evals.downstream.common.scorer via model_io). `max_length` is the
    re-encode window (default config.REREAD_WINDOW_TOKENS); `stats` tallies what the norm filter would drop."""
    window = C.REREAD_WINDOW_TOKENS if max_length is None else int(max_length)
    return _io.reread_cos(texts, dirs, base, tok, device, max_length=window, stats=stats)


def new_filter_stats():
    """One tally per scoring stage, handed to every `reread_cos` of it (`model_io.new_filter_stats`)."""
    return _io.new_filter_stats()


def norm_filter_record(stats):
    """What a scoring stage records of the (unapplied) norm filter: its multiple and the drop tally."""
    return {"applied": C.REREAD_NORM_FILTER, "mult": C.REREAD_NORM_FILTER_MULT, **stats}


def describe(gen_row, prompt_len, tok, max_new, stop_ids=None):
    """Decode one generated row, cut at `stop_ids` (every stage passes `stop_token_ids`)."""
    return _io.describe(gen_row, prompt_len, tok, max_new, stop_ids)


def direction(h, mu):
    return _io.direction(h, mu)


def centring_mean():
    """The shipped layer-42 centring mean (evals/downstream/common/background.py), shared by every package."""
    return _bg.load_centring_mean()


def _sampling():
    """This package's decoding settings, one set for every generating arm."""
    return dict(temp=C.TEMP, top_p=C.TOP_P, top_k=C.TOP_K, min_p=C.MIN_P, max_new=C.MAX_NEW, min_new=C.MIN_NEW)


def generate_injected(
    gen_model,
    tok,
    dirs,
    prompt_ids,
    marker_pos,
    device,
    n_samples=C.N_SAMPLES,
    seed=C.GEN_SEED,
    greedy=True,
    gen_chunk=C.GEN_CHUNK,
    stop_ids=None,
):
    """Rollouts of `gen_model` (the inverter, or the base for the untrained-base ablation) with `dirs`
    injected at the marker as a norm-matched add on that model's own residual."""
    return _io.generate_injected(
        gen_model,
        tok,
        dirs,
        prompt_ids,
        marker_pos,
        device,
        n_samples,
        seed,
        greedy,
        gen_chunk,
        sampling=_sampling(),
        inject_layer=C.INJECT_LAYER,
        steer_coeff=C.STEER_COEFF,
        describe_fn=lambda row, plen, t, mx: describe(row, plen, t, mx, stop_ids),
    )


def generate_batches(gen_model, tok, n, device, n_samples, seed, greedy, gen_chunk, batch_fn, hook_fn=None,
                     stop_ids=None):
    """`model_io.generate_batches` under this package's sampling and stop set, for an arm whose
    intervention is not the marker injection (Patchscopes and its floor)."""
    return _io.generate_batches(
        gen_model,
        tok,
        n,
        device,
        n_samples,
        seed,
        greedy,
        gen_chunk,
        _sampling(),
        batch_fn,
        hook_fn,
        lambda row, plen, t, mx: describe(row, plen, t, mx, stop_ids),
    )


def stop_token_ids(tok, source=None):
    """The ids every rollout is cut at: the tokenizer's EOS/PAD and `source`'s generation-config EOS ids."""
    return _io.stop_token_ids(tok, source)


def refuse_unless_generation_agrees(base, inverter):
    """Refuse unless the base and the inverter ship the same generation config (the ablation changes the
    model and nothing else)."""
    return _io.refuse_unless_generation_agrees(base, inverter)


def capture_all_layers(base, ids, pos, device):
    """(h [N_LAYERS, d] at `pos`, token norms [T] at the read layer) from one clean pass of the base."""
    H = capture(base, ids, device, range(C.N_LAYERS))
    return H[:, pos], np.linalg.norm(H[C.READ_LAYER], axis=-1)


def arm_seed(arm, seed):
    return int(seed) + C.ARM_SEED_OFFSET[arm]


def first_word_correct(text, answer_forms):
    m = re.search(r"\w+", M.normalise(text))
    return bool(m) and m.group(0) in {M.normalise(a) for a in answer_forms}


def distinct_share(texts):
    """Share of distinct greedy texts (near 1/n when an injection never fired)."""
    return len(set(texts)) / max(1, len(texts))


def guard_greedy(texts):
    """`distinct_share`, raising below 0.95 over five or more items; applied to MAEM's own rollouts only."""
    share = distinct_share(texts)
    if len(texts) >= 5 and share < 0.95:
        raise RuntimeError(f"injection not firing: only {share:.2f} of greedy texts are distinct")
    return share


def greedy_continuation(base, tok, ids, max_new, device):
    """The multi-hop competence diagnostic: the clean base's greedy continuation of the item's prompt."""
    import torch

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
    cont = g[0, len(ids) :]
    return tok.decode(cont, skip_special_tokens=True), [int(t) for t in cont.tolist()]


def stage_capture(args, run):
    """Every layer's residual at each item's readout position, norms, and the multi-hop diagnostic."""
    chash = stage_key("capture", args, run)
    if stage_done(run, "capture", chash) and not args.force:
        print("[capture] up to date")
        return
    started = time.time()
    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    base, tok = load_base(args.device)
    H = np.zeros((len(kept), C.N_LAYERS, C.D_MODEL), dtype=np.float32)
    norms = {}
    diag = {}
    t0 = time.time()
    gen_cfg = base.generation_config.to_dict()
    write_provenance(run, {"generation_config": gen_cfg}, stage="capture")
    print(f"[capture] base generation_config: {gen_cfg}", flush=True)
    for it in kept:
        h, nv = capture_all_layers(base, it["ids"], it["readout_pos"], args.device)
        H[it["i"]] = h
        med = float(np.median(nv[1:])) if len(nv) > 1 else float(nv[0])
        rn = float(nv[it["readout_pos"]])
        norms[str(it["i"])] = {
            "readout_norm": rn,
            "prompt_median_norm": med,
            "ratio": rn / med if med else None,
            "outlier": bool(med and rn > C.NORM_MULT * med),
        }
        if it["family"] == "multihop":
            text, cont_ids = greedy_continuation(base, tok, it["ids"], C.DIAG_MAX_NEW, args.device)
            diag[str(it["i"])] = {
                "text": text,
                "ids": cont_ids,
                "correct": first_word_correct(text, it["answer_forms"]),
            }
    np.savez_compressed(run.file("activations/h_all.npz"), h=H)
    run.write_json("activations/norms.json", norms)
    run.write_json("data/diagnostic.json", diag)
    nc = sum(1 for v in diag.values() if v["correct"])
    print(
        f"[capture] {len(kept)} items in {time.time() - t0:.0f}s; multihop greedy-correct {nc}/{len(diag)}", flush=True
    )
    mark_stage(run, "capture", chash, {"n_items": len(kept), "multihop_greedy_correct": nc}, started=started)


def stage_rollouts(args, run):
    """MAEM's rollouts, re-read against own and foil directions, and the one enforced injection check
    (distinct greedies, own − foil gap ≥ 0.10). The inverter is freed before anything is re-read."""
    chash = stage_key("rollouts", args, run)
    if stage_done(run, "rollouts", chash) and not args.force:
        print("[rollouts] up to date")
        return
    started = time.time()
    from maem.prompts import build_prompt_ids, marker_positions

    items = run.read_json("data/items.json")
    kept = [x for x in items["items"] if not x["excluded"]]
    H = np.load(run.file("activations/h_all.npz"))["h"]
    mu = centring_mean()
    dirs = np.stack([direction(H[it["i"], C.READ_LAYER], mu) for it in kept])
    foil_dirs = np.stack([dirs[it["foil"]] for it in kept])
    h42_raw = np.stack([H[it["i"], C.READ_LAYER] for it in kept])
    raw_unit = h42_raw / np.linalg.norm(h42_raw, axis=1, keepdims=True)
    cos_centred_vs_raw = np.einsum(
        "kd,kd->k", dirs.astype(np.float64), raw_unit.astype(np.float64)
    )  # methodology §4: cos(d, normalize(h_42))
    base, tok = load_base(args.device)
    prompt_ids, mpos = build_prompt_ids(tok)
    assert marker_positions(tok, prompt_ids) == mpos and len(mpos) == 1
    stops = stop_token_ids(tok, base)
    inverter = load_inverter(args.device)
    t0 = time.time()
    try:
        refuse_unless_generation_agrees(base, inverter)
        samples, greedy = generate_injected(
            inverter, tok, dirs, prompt_ids, mpos[0], args.device, seed=arm_seed("maem", args.seed),
            stop_ids=stops
        )
    finally:
        inverter = free_model(inverter)
    share = guard_greedy([g["text"] for g in greedy])
    flat = [g["text"] for g in greedy] + [s["text"] for row in samples for s in row]
    rows = list(range(len(kept))) + [i for i in range(len(kept)) for _ in range(C.N_SAMPLES)]
    # the tally is filled by the own pass alone (the foil pass re-reads the same texts)
    tally = new_filter_stats()
    own = reread_cos(flat, dirs[rows], base, tok, args.device, stats=tally)
    foil = reread_cos(flat, foil_dirs[rows], base, tok, args.device)
    recs = []
    for k, it in enumerate(kept):
        g = dict(greedy[k], cos_own=float(own[k]), cos_foil=float(foil[k]))
        ss = [
            dict(
                samples[k][j],
                cos_own=float(own[len(kept) + k * C.N_SAMPLES + j]),
                cos_foil=float(foil[len(kept) + k * C.N_SAMPLES + j]),
            )
            for j in range(C.N_SAMPLES)
        ]
        recs.append({"i": it["i"], "greedy": g, "samples": ss, "cos_centred_vs_raw": float(cos_centred_vs_raw[k])})
    gap = float(np.mean(own[: len(kept)]) - np.mean(foil[: len(kept)]))
    check = {
        "greedy_distinct_share": share,
        "greedy_cos_own_mean": float(np.mean(own[: len(kept)])),
        "greedy_cos_foil_mean": float(np.mean(foil[: len(kept)])),
        "gap": gap,
        "seconds": time.time() - t0,
    }
    print(
        f"[rollouts] {len(kept)} items; greedy cos own {check['greedy_cos_own_mean']:.3f} foil {check['greedy_cos_foil_mean']:.3f} gap {gap:.3f}; distinct {share:.2f}; {check['seconds']:.0f}s",
        flush=True,
    )
    # checked before saving, so a directory never holds a rejected MAEM arm (methodology §4.3)
    if gap < 0.10:
        raise RuntimeError(f"injection check failed: own − foil greedy cosine gap {gap:.3f} < 0.10")
    run.write_json(
        "rollouts/maem.json",
        {
            "config": {
                "inverter": [C.INVERTER, C.INVERTER_REVISION],
                "prompt_ids": [int(t) for t in prompt_ids],
                "marker_pos": mpos[0],
                "stop_ids": list(stops),
                "seed": arm_seed("maem", args.seed),
                "gen": {
                    "n_samples": C.N_SAMPLES,
                    "temp": C.TEMP,
                    "top_p": C.TOP_P,
                    "top_k": C.TOP_K,
                    "min_p": C.MIN_P,
                    "max_new": C.MAX_NEW,
                    "min_new": C.MIN_NEW,
                    "gen_chunk": C.GEN_CHUNK,
                },
                "mu_norm": float(np.linalg.norm(mu)),
                "injection_check": check,
                "cos_centred_vs_raw_mean": float(np.mean(cos_centred_vs_raw)),
                "norm_filter": norm_filter_record(tally),
            },
            "items": recs,
        },
    )
    write_provenance(run, {"rollouts_injection_check": check, "inverter_revision": C.INVERTER_REVISION},
                     stage="rollouts")
    mark_stage(run, "rollouts", chash, dict(check, stop_ids=list(stops), norm_filter=norm_filter_record(tally)),
               started=started)
