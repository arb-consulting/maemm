"""Stage `capture` (methodology §2 step 2, §3) and the generation helpers `frontier_context` calls.

The model layer is evals/downstream/common/model_io.py's. `capture` reads the raw layer-42 residual at each
candidate's position in its document on the clean base (`read_all_positions`: raw document tokens, no
template, no sink, every position) and keeps the first `n_sources` whose raw norm passes the filter.
`describe` names its token count `n_gen`, what §5 cuts the source passage to. The re-read's norm filter is
not applied; what it would drop is tallied beside the scores (`filter_record`).
"""

import time

import numpy as np

from evals.downstream.common.model_io import describe as _describe
from evals.downstream.common.model_io import generate_batches, load_base, pad_batch
from evals.downstream.common.model_io import generate_injected as _generate_injected
from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance
from evals.downstream.common.scorer import NORM_FILTER_MULT
from evals.downstream.rollout_coherence import config as C
from evals.downstream.rollout_coherence.documents import source_passage_ids
from evals.downstream.rollout_coherence.runs import stage_hashes


# ---------------------------------------------------------------- the re-read's filter tally

def filter_record(stats):
    """The stage record's norm-filter entry: not applied, its multiple, and the tally (`new_filter_stats()`)
    of what it would have dropped, with the dropped share of content tokens. Reported, never gated on."""
    n = int(stats["n_tokens"])
    return {"applied": False, "mult": NORM_FILTER_MULT, **{k: int(v) for k, v in stats.items()},
            "share_tokens_filter_drops": (int(stats["n_tokens_filter_drops"]) / n) if n else None}


# ---------------------------------------------------------------- model loading and clean reads

def read_all_positions(base, ids_batch, device):
    """Layer-42 residual at EVERY position of raw document tokens, on the clean base: [B, 512, d] float32.
    No sink and no chat template — the training bank's read (§3), not the standalone re-read protocol."""
    import torch
    from maem.inject import read_resid

    ids = torch.tensor(ids_batch, device=device)
    with torch.no_grad():
        h, _ = read_resid(base, C.READ_LAYER, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, pool="all")
    return h.float().cpu().numpy()


# ---------------------------------------------------------------- pure helpers

def norm_ceiling(profiles, rng, lo, hi, n_presample, mult):
    """`(ceiling, median)`: `mult` x the median raw norm of `n_presample` (document, position) draws from
    `rng` over the [n_docs, SEQ_LEN] norm `profiles`, positions in [lo, hi] (methodology §3)."""
    profiles = np.asarray(profiles, np.float64)
    docs = rng.integers(0, profiles.shape[0], n_presample)
    pos = rng.integers(lo, hi + 1, n_presample)
    med = float(np.median(profiles[docs, pos]))
    return mult * med, med


def passes_norm_filter(norm, floor, ceiling):
    """The training pipeline's filter on the RAW norm, before centring: `floor < ||h|| <= ceiling`."""
    return bool(floor < float(norm) <= ceiling)


def source_prefix_ids(source):
    """The `continuation` method's prompt (methodology §4): the document's tokens through the readout
    position, via `documents.source_passage_ids` (the slice §5's partner cut uses). 65 to 512 tokens."""
    return source_passage_ids(source["ids"], source["pos"], source["pos"] + 1)


def describe(gen_row, prompt_len, tok, max_new, stop_ids=None):
    """`model_io.describe`'s record with the token count named `n_gen`. `stop_ids` defaults to the
    tokenizer's EOS/PAD; the generating methods pass `stop_token_ids`."""
    r = _describe(gen_row, prompt_len, tok, max_new, stop_ids)
    return {"text": r["text"], "ids": r["ids"], "n_gen": r["n_tokens"], "eos": r["eos"],
            "capped": r["capped"]}


def best_of_8(records):
    """The index of the highest-re-read-cosine sample among `records`; an unscored (empty) rollout can
    never be picked. numpy's argmax takes the lowest index on a tie."""
    samples = sorted((r for r in records if r["sample"] != "greedy"), key=lambda r: r["sample"])
    cos = [r["reread_cos"] if r["reread_cos"] is not None else -2.0 for r in samples]
    return int(np.argmax(cos))


# ---------------------------------------------------------------- generation

def _records(stop_ids):
    """`describe` bound to this pass's stop-token set, as `generate_batches` takes it."""
    return lambda gen_row, prompt_len, tok, max_new: describe(gen_row, prompt_len, tok, max_new, stop_ids)


def generate_injected(gen_model, tok, dirs, prompt_ids, marker_pos, device, n_samples, seed, greedy,
                      gen_chunk, sampling, inject_layer, steer_coeff, stop_ids=None, row_ids=None):
    """`model_io.generate_injected` writing this package's records. `gen_model` is the inverter (`maem`)
    or the clean base (`base`); `row_ids` are the activations' global indices, which seed each call."""
    return _generate_injected(gen_model, tok, dirs, prompt_ids, marker_pos, device, n_samples, seed, greedy,
                              gen_chunk, sampling, inject_layer, steer_coeff, describe_fn=_records(stop_ids),
                              row_ids=row_ids)


def generate_continuation(base, tok, prompt_ids_per_row, device, n_samples, seed, greedy, gen_chunk,
                          sampling, stop_ids=None, row_ids=None):
    """The `continuation` method (methodology §4): the clean base continuing each activation's own source
    prefix, no injection. Prefixes differ in length, so batches are left-padded (generation then starts at
    one column for every row). Same sampling, seeds and records as `generate_injected`.
    Returns (samples[n][n_samples], greedy_out[n])."""
    if tok.pad_token_id is None:
        # load_base sets pad_token from eos; guard against a tokenizer built another way
        raise ValueError("generate_continuation needs tok.pad_token_id for the left pad")
    n = len(prompt_ids_per_row)

    def batch_fn(rows):
        batch = pad_batch([prompt_ids_per_row[i] for i in rows], int(tok.pad_token_id), device, side="left")
        return batch["input_ids"], batch["attention_mask"], batch["input_ids"].shape[1]

    return generate_batches(base, tok, n, device, n_samples, seed, greedy, gen_chunk, sampling, batch_fn,
                            None, _records(stop_ids), row_ids)


# ---------------------------------------------------------------- stages

def capture_config_hash(run, args):
    """Chained to `prepare`'s record, so a re-drawn pool re-captures the activations."""
    return config_hash({"sizes": C.sizes(args.smoke), "model": C.MODEL_REVISION, "draw": C.DRAW,
                        "seed": C.DATA_SEED, "scoring": C.SCORING_VERSION,
                        "upstream": stage_hashes(run, ["prepare"])})


def stage_capture(args, run):
    """The first `n_sources` pool candidates whose raw norm passes the filter, in pool order (so never
    sharded). The pool is forwarded only up to the last accepted candidate, which selects the same set."""
    sz = C.sizes(args.smoke)
    chash = capture_config_hash(run, args)
    if stage_done(run, "capture", chash) and not args.force:
        print("[capture] up to date", flush=True)
        return
    started = time.time()
    from evals.downstream.rollout_coherence.documents import pool_rows, rng_after_prepare

    doc = run.read_json("data/documents.json")
    base, tok = load_base(args.device, C.MODEL, C.MODEL_REVISION)
    # the whole pool, never a previous pass's `sources`, so a re-run keeps the rejected candidates' record
    pool = pool_rows(doc)
    n, n_spare = sz["n_sources"], len(pool) - sz["n_sources"]
    rng = rng_after_prepare(n, n_spare, C.DATA_SEED)
    profiles = []
    t0 = time.time()

    def forward(upto):
        while len(profiles) < min(upto, len(pool)):
            batch = pool[len(profiles):len(profiles) + 4]
            H = read_all_positions(base, [s["ids"] for s in batch], args.device)
            for j in range(len(batch)):
                profiles.append(np.linalg.norm(np.asarray(H[j], np.float64), axis=1))
            print(f"  [capture] forwarded {len(profiles)}/{len(pool)} pool documents", flush=True)

    n_pre = min(C.NORM_PRESAMPLE_DOCS, len(pool))
    forward(n_pre)
    ceiling, med = norm_ceiling(profiles[:n_pre], rng, C.POS_LO, C.POS_HI, C.NORM_PRESAMPLE, C.NORM_MULT)
    print(f"[capture] presample median raw norm {med:.1f} (keep <= {C.NORM_MULT:g}x)", flush=True)
    accepted, rejected = [], []
    for r, s in enumerate(pool):
        if len(accepted) == n:
            break
        forward(r + 1)
        s["act_norm"] = float(profiles[r][s["pos"]])          # the raw ||h||, before centring
        (accepted if passes_norm_filter(s["act_norm"], C.NORM_FLOOR, ceiling) else rejected).append(r)
    if len(accepted) < n:
        raise ValueError(f"only {len(accepted)} of the pool's {len(pool)} candidates pass the norm filter; "
                         f"{n} wanted")
    sources = []
    for i, r in enumerate(accepted):
        s = pool[r]
        s["i"], s["id"] = i, f"act/{i:03d}"
        s["passage_text"] = tok.decode(s["ids"][:s["pos"] + 1], skip_special_tokens=False)
        sources.append(s)
    taken = set(accepted) | set(rejected)
    unused = [{k: s[k] for k in ("doc", "ids", "pos", "pool_row")} for r, s in enumerate(pool) if r not in taken]
    doc["sources"] = sources
    doc["rejected"] = [{k: pool[r][k] for k in ("doc", "ids", "pos", "pool_row", "act_norm")} for r in rejected]
    doc["spare"] = unused
    run.write_json("data/documents.json", doc)
    write_provenance(run, {"n_rejected": len(rejected), "n_forwarded": len(profiles),
                           "presample_median_norm": med, "norm_ceiling": ceiling,
                           "capture_seconds": time.time() - t0}, stage="capture")
    mark_stage(run, "capture", chash, {"n": n, "n_rejected": len(rejected), "presample_median_norm": med,
                                       "norm_ceiling": ceiling}, started=started)
    print(f"[capture] {n} activations, {len(rejected)} candidates rejected by the norm filter", flush=True)
