"""The steered-model reference both steering packages run: the plain base with a vector added.

The clean base generates from a one-token prompt (bos, else eos) while `strength * STEER_UNIT * unit(v)` is
added to block 42's output at every position, with no template or instruction, at temperature 1 under one
seed per row that depends only on (vector, strength, sample). Each text's unsteered log-likelihood and its
`degeneracy` rules are summarised by `health`, a diagnostic that selects nothing.
"""
import hashlib
import json
import time
from collections import Counter

import numpy as np

from evals.downstream.common import degeneracy

ARM = "plain_steered"
# A typical layer-42 residual norm of the clean base, a constant so a strength means the same everywhere.
STEER_UNIT = 84.48859405517578
TABLE_STRENGTH = 1.0                   # the paper's main table and budget figure
STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)
MAIN_SAMPLES = 64                      # at the table strength: eight bundles of eight
CURVE_SAMPLES = 16
NEW_TOKENS = 64
MAX_ROWS = 512                         # rows per payload; 1024 runs a B200 out of memory in the fla kernel
# Mixed into every row's seed; the published texts were drawn under this exact string.
SEED_LABEL = "steer_pilot"
OPERATION = "plain_steer"
LOGLIK_BATCH = 32

HEALTH_COLUMNS = ("vector_id", "strength", "n", "degenerate_share", "loglik_median", "eos_early_share",
                  "empty_share", "distinct3", "non_ascii_share", "identical_share", "mean_tokens", "reasons")

_ROW_FIELDS = ("vector_id", "strength", "coefficient", "sample_id", "seed", "dir_index")


def seed_for(seed, *parts):
    """`evals.downstream.steering_vector_inversion.artifacts.seed_for`, restated: evals/downstream/common imports no package."""
    text = json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def arm_name(strength):
    """`plain_steered@<strength>`, e.g. `plain_steered@1` or `plain_steered@0.25`."""
    return f"{ARM}@{float(strength):g}"


TABLE_ARM = arm_name(TABLE_STRENGTH)
ARMS = tuple(arm_name(s) for s in STRENGTHS)


def samples_by_strength(curve_samples=CURVE_SAMPLES):
    """`{strength: samples}`: the table strength at `MAIN_SAMPLES`, every other strength at `curve_samples`."""
    return {s: MAIN_SAMPLES if s == TABLE_STRENGTH else curve_samples for s in STRENGTHS}


def cells(vector_ids, samples, greedy=()):
    """`[(vector_id, strength, n, is_greedy)]`: each strength of `samples` at its count, then `greedy`'s."""
    out = []
    for vector_id in vector_ids:
        out += [(vector_id, float(s), int(n), False) for s, n in samples.items()]
        out += [(vector_id, float(s), 1, True) for s in greedy]
    return out


def payloads(directions, cells, extra=None):
    """The GPU payloads of `cells`, at most `MAX_ROWS` rows each, a cell never split and greedy cells apart.
    `directions` maps a vector id to its direction, `extra` to fields its rows carry. A row is
    `{vector_id, **extra, strength, coefficient, sample_id, seed, dir_index}`, `sample_id` -1 when greedy."""
    extra = extra or {}
    out = []
    for greedy in (False, True):
        rows, dirs, index = [], [], {}
        for vector_id, strength, n, is_greedy in cells:
            if is_greedy != greedy:
                continue
            if rows and len(rows) + n > MAX_ROWS:
                out.append(_payload(rows, dirs, greedy))
                rows, dirs, index = [], [], {}
            if vector_id not in index:
                index[vector_id] = len(dirs)
                dirs.append(np.asarray(directions[vector_id], dtype=np.float32).tolist())
            labels = dict(extra.get(vector_id) or {})
            clash = sorted(set(labels) & set(_ROW_FIELDS))
            if clash:
                raise ValueError(f"{vector_id}: extra row fields {clash} are the payload's own")
            for sample_id in ([-1] if greedy else range(n)):
                rows.append({"vector_id": vector_id, **labels, "strength": strength,
                             "coefficient": float(strength * STEER_UNIT), "sample_id": sample_id,
                             "seed": seed_for(0, SEED_LABEL, vector_id, strength, sample_id),
                             "dir_index": index[vector_id]})
        if rows:
            out.append(_payload(rows, dirs, greedy))
    return out


def _payload(rows, dirs, greedy):
    return {"operation": OPERATION, "rows": rows, "directions": dirs, "greedy": greedy}


def sink_id(tokenizer):
    """The one-token prompt: the bos token, or the eos where the tokenizer has none."""
    return tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id


def generate(worker, payload):
    """One payload on `worker.base`: each row less `dir_index`, plus `token_ids` (cut before the first stop
    id), `text`, `n_tokens`, `eos_terminated`, `stop_token` and the unsteered `loglik_sum` / `loglik_mean`."""
    import torch
    from maemm.inject import get_layer, hooked
    from transformers import GenerationConfig, LogitsProcessorList

    from evals.downstream.common.model_io import RowSampler, addition_hook

    started = time.time()
    tok = worker.tokenizer
    rows = payload["rows"]
    sink = sink_id(tok)
    dirs = torch.as_tensor(np.asarray(payload["directions"], dtype=np.float32), device=worker.device)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    vectors = dirs[[r["dir_index"] for r in rows]]
    coefficients = torch.as_tensor([float(r["coefficient"]) for r in rows], dtype=torch.float32,
                                   device=worker.device)
    input_ids = torch.full((len(rows), 1), int(sink), dtype=torch.long, device=worker.device)
    mask = torch.ones_like(input_ids)
    hook = addition_hook(vectors, coefficients, mask.bool())
    generation = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1, min_new_tokens=0,
                                  max_new_tokens=NEW_TOKENS, eos_token_id=worker.eos_ids,
                                  pad_token_id=tok.pad_token_id, use_cache=True, temperature=1., top_p=1.,
                                  top_k=0, min_p=0., repetition_penalty=1.0, no_repeat_ngram_size=0,
                                  renormalize_logits=False)
    processors = LogitsProcessorList([] if payload.get("greedy")
                                     else [RowSampler([r["seed"] for r in rows], worker.device)])
    with torch.inference_mode(), hooked(get_layer(worker.base, worker.config.read_layer), hook):
        generated = worker.base.generate(input_ids=input_ids, attention_mask=mask,
                                         generation_config=generation, logits_processor=processors)
    gen_seconds = time.time() - started
    suffixes = generated[:, 1:].cpu().tolist()
    out = []
    for r, ids in zip(rows, suffixes):
        stop = next((i for i, t in enumerate(ids) if t in worker.eos_ids), None)
        kept = ids if stop is None else ids[:stop]
        out.append({**{k: v for k, v in r.items() if k != "dir_index"}, "token_ids": kept,
                    "text": tok.decode(kept, skip_special_tokens=True),
                    "n_tokens": len(kept), "eos_terminated": stop is not None,
                    "stop_token": None if stop is None else int(ids[stop])})
    _loglik(worker, out, sink)
    on_cuda = getattr(worker.device, "type", str(worker.device)).startswith("cuda")
    return {"rows": out, "sink_id": int(sink), "sink_token": tok.convert_ids_to_tokens(int(sink)),
            "eos_ids": list(worker.eos_ids), "read_layer": int(worker.config.read_layer),
            "gen_seconds": gen_seconds, "gpu_seconds": time.time() - started,
            "gpu_name": torch.cuda.get_device_name(worker.device) if on_cuda else "CPU",
            "runtime_versions": worker.versions}


def _loglik(worker, rows, sink):
    """Each text's mean and summed token log-probability under the unsteered base, read as `[sink] + tokens`."""
    import torch

    from evals.downstream.common.model_io import pad_batch

    todo = [r for r in rows if r["token_ids"]]
    for r in rows:
        if not r["token_ids"]:
            r["loglik_sum"], r["loglik_mean"] = None, None
    for start in range(0, len(todo), LOGLIK_BATCH):
        chunk = todo[start:start + LOGLIK_BATCH]
        seqs = [[int(sink)] + list(r["token_ids"]) for r in chunk]
        batch = pad_batch(seqs, worker.tokenizer.pad_token_id, worker.device, "right")
        with torch.inference_mode():
            logits = worker.base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 use_cache=False).logits
        for i, (r, seq) in enumerate(zip(chunk, seqs)):
            n = len(seq)
            rowlogits = logits[i, :n - 1].float()
            target = batch["input_ids"][i, 1:n]
            lp = rowlogits.gather(-1, target[:, None]).squeeze(-1) - torch.logsumexp(rowlogits, -1)
            r["loglik_sum"] = float(lp.sum())
            r["loglik_mean"] = float(lp.mean())
        del logits, batch


def distinct3(text):
    """Distinct lowercased 3-grams over all 3-grams; NaN below three words."""
    words = str(text or "").lower().split()
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def _mean(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _median(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(finite)) if finite else float("nan")


def health(rows, reasons=None):
    """`HEALTH_COLUMNS` per (vector, strength) over the sampled texts, under `degeneracy.reasons` by default."""
    reasons = degeneracy.reasons if reasons is None else reasons
    latest = {}
    for r in rows:
        if int(r["sample_id"]) >= 0:
            latest[(r["vector_id"], float(r["strength"]), int(r["sample_id"]))] = r
    groups = {}
    for (vector_id, strength, _sample), r in latest.items():
        groups.setdefault((vector_id, strength), []).append(r)
    out = []
    for (vector_id, strength) in sorted(groups):
        g = groups[(vector_id, strength)]
        texts = [str(r.get("text") or "") for r in g]
        tripped = [tuple(reasons(t)) for t in texts]
        counts = Counter(x for rs in tripped for x in rs)
        n = len(g)
        out.append({"vector_id": vector_id, "strength": strength, "n": n,
                    "degenerate_share": _mean([bool(rs) for rs in tripped]),
                    "loglik_median": _median([r.get("loglik_mean") for r in g]),
                    "eos_early_share": _mean([bool(r.get("eos_terminated")) for r in g]),
                    "empty_share": _mean([t.strip() == "" for t in texts]),
                    "distinct3": _mean([distinct3(t) for t in texts]),
                    "non_ascii_share": _mean(["non_ascii" in rs for rs in tripped]),
                    "identical_share": 1 - len(set(texts)) / n,
                    "mean_tokens": _mean([r.get("n_tokens") for r in g]),
                    "reasons": "|".join(f"{k}={v}" for k, v in sorted(counts.items()))})
    return out
