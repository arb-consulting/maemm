"""BiPO training at the read layer's output, added around the pinned worker as the `train` operation.

Every forward and backward runs on `worker.base`, the clean model: a steering vector is trained to steer
that model, and the inverter is no part of it. `train` also returns the difference of the mean response
activations of the same pairs, whose cosine with the learned vector is a training metric. No inference mode
is used: inference tensors cannot be autograd inputs.
"""
from contextlib import nullcontext
import math
import time

import numpy as np
import torch

from maem.inject import get_layer, hooked


def _chat_ids(chat_ids_fn=None):
    """Imported per call: the pinned model module is a GPU dependency, the helpers below are not."""
    if chat_ids_fn is not None:
        return chat_ids_fn
    from evals.downstream.steering_vector_inversion.model import chat_ids
    return chat_ids


def encode_pair(tokenizer, question, response, chat_ids_fn):
    """BiPO's own construction: the chat prompt, ' ' + response, and the end of turn TRL appends."""
    prompt = list(chat_ids_fn(tokenizer, [{"role": "user", "content": question}],
                              add_generation_prompt=True))
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == getattr(tokenizer, "unk_token_id", None):
        end_id = tokenizer.eos_token_id
    ids = prompt + list(tokenizer.encode(" " + response, add_special_tokens=False)) + [end_id]
    if len(ids) <= len(prompt):
        raise ValueError("Empty response encoding")
    return ids, len(prompt)


def response_logps(logits, input_ids, p_lens, lengths):
    """Summed response log-probs, one row at a time: the fp32 upcast of a 248k vocabulary is large."""
    totals, counts = [], []
    for i, (p, length) in enumerate(zip(p_lens, lengths)):
        rows = logits[i, p - 1:length - 1]
        gathered = rows.gather(-1, input_ids[i, p:length, None]).squeeze(-1)
        addends = gathered.float() - torch.logsumexp(rows.float(), -1)
        totals.append(addends.sum())
        counts.append(int(length - p))
    return torch.stack(totals), counts


def dpo_loss(policy_m, policy_n, ref_m, ref_n, d, beta):
    """BiPO's bidirectional DPO: one sign d per step enters the forward as d*v and flips this logit."""
    logit = d * beta * ((policy_m - ref_m) - (policy_n - ref_n))
    return -torch.nn.functional.logsigmoid(logit).mean(), (logit > 0).float().mean()


def cosine_lr(step, total, warmup, base):
    """Linear warmup from zero over `warmup` steps, then cosine down to zero at `total`."""
    if warmup and step < warmup:
        return base * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def trainable_hook(theta, sigma, d, mask):
    """Out-of-place block-output addition of d*sigma*theta at non-pad positions; grad reaches theta."""
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = (d * sigma * theta.float())[None, None, :].expand(h.shape[0], h.shape[1], -1).to(h.dtype)
        h = h + mask[:, :, None].to(h.dtype) * delta
        return (h, *output[1:]) if isinstance(output, tuple) else h
    return hook


def _vector(theta, sigma):
    """The d-dimensional steering vector the parameter stands for."""
    return sigma * theta


def _capture(store):
    def hook(_module, _inputs, output):
        store.append((output[0] if isinstance(output, tuple) else output).detach().float())
    return hook


def _encode_rows(worker, rows, chat_ids_fn, max_tokens=None):
    """Encoded pairs; a pair whose either side exceeds the cap is dropped, never truncated."""
    kept, dropped = [], 0
    for row in rows:
        m, p_m = encode_pair(worker.tokenizer, row["question"], row["matching"], chat_ids_fn)
        n, p_n = encode_pair(worker.tokenizer, row["question"], row["not_matching"], chat_ids_fn)
        if max_tokens is not None and max(len(m), len(n)) > max_tokens:
            dropped += 1
            continue
        kept.append({"source_id": row.get("source_id"), "m": m, "p_m": p_m, "n": n, "p_n": p_n})
    return kept, dropped


def _items(pairs):
    """Matching sequences then not-matching ones, so one scoring pass covers both sides of a batch."""
    return ([(p["m"], p["p_m"]) for p in pairs], [(p["n"], p["p_n"]) for p in pairs])


def _score(worker, items, hook_factory=None, batch_size=16):
    """Summed response log-probs for (ids, p_len) items, optionally under a read-layer hook."""
    layer = get_layer(worker.base, worker.config.read_layer)
    totals, counts = [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = worker._batch([ids for ids, _ in chunk], side="right")
        context = (nullcontext() if hook_factory is None
                   else hooked(layer, hook_factory(batch["attention_mask"].bool())))
        with torch.no_grad(), context:
            logits = worker.base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 use_cache=False).logits
        lp, n = response_logps(logits, batch["input_ids"], [p for _, p in chunk],
                               [len(ids) for ids, _ in chunk])
        totals += [float(x) for x in lp.float().cpu()]
        counts += n
        del logits, batch
    return totals, counts


def _activations(worker, items, batch_size=16):
    """Mean response activation per sequence plus the retained token norms, under the package filter."""
    layer = get_layer(worker.base, worker.config.read_layer)
    means, norms, width = [], [], None
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = worker._batch([ids for ids, _ in chunk], side="right")
        store = []
        with torch.no_grad(), hooked(layer, _capture(store)):
            worker.base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        use_cache=False)
        h = store[0]
        width = h.shape[-1]
        for i, (ids, p) in enumerate(chunk):
            tokens = h[i, p:len(ids)]
            row = tokens.norm(dim=-1)
            keep = (row <= 10 * row.median()) & (row > 0)
            if not bool(keep.any()):       # only reachable if every response token has zero norm
                keep = row > 0
            means.append(tokens[keep].mean(0).double().cpu().numpy() if bool(keep.any())
                         else np.zeros(h.shape[-1]))
            norms.append(row[keep].float().cpu().numpy())
        del h, store, batch
    return means, (np.concatenate(norms) if norms else np.zeros(0, dtype=np.float32)), width


def _step(worker, batch, p_lens, lengths, theta, sigma, d, ref_m, ref_n, beta):
    """One BiPO forward: the vector is added at every non-pad position of the whole batch."""
    layer = get_layer(worker.base, worker.config.read_layer)
    mask = batch["attention_mask"].bool()
    with torch.enable_grad(), hooked(layer, trainable_hook(theta, sigma, d, mask)):
        logits = worker.base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             use_cache=False).logits
    lp, _ = response_logps(logits, batch["input_ids"], p_lens, lengths)
    del logits
    half = len(lp) // 2
    return dpo_loss(lp[:half], lp[half:], ref_m, ref_n, d, beta)


def _evaluate(worker, pairs, theta, sigma, median_norm, diffmean, ref_margin):
    """Heldout margins under +v and -v against the reference margin of the same pairs."""
    v = _vector(theta.detach(), sigma).float()
    norm = float(v.norm())
    result = {"norm": norm, "norm_ratio": norm / median_norm if median_norm else None,
              "cos_diffmean": _cosine(v.cpu().numpy(), diffmean), "n_eval_pairs": len(pairs)}
    if not pairs:
        return result
    matching, not_matching = _items(pairs)
    for name, d in (("pos", 1.0), ("neg", -1.0)):
        scores, _ = _score(worker, matching + not_matching,
                           lambda mask, d=d: trainable_hook(theta.detach(), sigma, d, mask))
        half = len(pairs)
        margin = np.asarray(scores[:half]) - np.asarray(scores[half:])
        better = margin > ref_margin if name == "pos" else margin < ref_margin
        result[f"acc_{name}"] = float(better.mean())
        result[f"pref_{name}"] = float((margin > 0).mean())
        result[f"shift_{name}"] = float((margin - ref_margin).mean())
    result["pref_base"] = float((ref_margin > 0).mean())
    return result


def _cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else None


def train(worker, payload, chat_ids_fn=None):
    """BiPO on one behaviour: theta is the only parameter, v = sigma*theta in residual units."""
    chat = _chat_ids(chat_ids_fn)
    hp, behaviour = payload["hp"], payload["behaviour"]
    beta, batch_pairs = hp["beta"], hp["batch_pairs"]
    epochs, max_steps = hp["epochs"], hp.get("max_steps")
    if worker.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(worker.device)
    pairs, dropped = _encode_rows(worker, payload["train"], chat, hp["max_tokens"])
    heldout, heldout_dropped = _encode_rows(worker, payload["heldout"], chat, hp["max_tokens"])
    if not pairs:
        raise ValueError(f"No training pair of {behaviour} fits {hp['max_tokens']} tokens")
    matching, not_matching = _items(pairs)

    # Activations: the diff-of-means baseline and the scale that makes theta unit-free.
    means, norms, width = _activations(worker, matching + not_matching)
    half = len(pairs)
    diffmean = (np.mean(means[:half], axis=0) - np.mean(means[half:], axis=0))
    median_norm = float(np.median(norms))
    sigma = median_norm / math.sqrt(width)
    diffmean = np.asarray(diffmean, dtype=np.float32)

    # Reference log-probs: the frozen model's own scores, the DPO baseline every step subtracts.
    reference, _ = _score(worker, matching + not_matching)
    ref_m = torch.tensor(reference[:half], dtype=torch.float32, device=worker.device)
    ref_n = torch.tensor(reference[half:], dtype=torch.float32, device=worker.device)
    held_m, held_n = _items(heldout)
    held_reference, _ = _score(worker, held_m + held_n) if heldout else ([], [])
    ref_margin = (np.asarray(held_reference[:len(heldout)]) - np.asarray(held_reference[len(heldout):])
                  if heldout else np.zeros(0))
    evaluated = heldout[:hp["heldout_eval_pairs"]]
    evaluated_margin = ref_margin[:hp["heldout_eval_pairs"]]

    theta = torch.nn.Parameter(torch.zeros(width, dtype=torch.float32, device=worker.device))
    optimizer = torch.optim.AdamW([theta], lr=hp["lr_theta"], weight_decay=hp["weight_decay"])
    from evals.downstream.steering_vector_inversion.artifacts import seed_for
    rng = np.random.default_rng(seed_for(1234, behaviour, "bipo", payload["train_seed"]))
    per_epoch = math.ceil(len(pairs) / batch_pairs)
    total = epochs * per_epoch
    checkpoints, metrics, log, step_seconds = {}, [], [], []
    counts = {"+1": 0, "-1": 0}
    step, stopped, started = 0, False, time.time()

    def checkpoint(epoch):
        # Cloned: on CPU the expression below is a view of the parameter's own storage, which the
        # optimiser goes on writing to, and every earlier epoch would end up holding the last one.
        checkpoints[str(epoch)] = _vector(theta, sigma).detach().clone().float().cpu().numpy()
        metrics.append({"epoch": epoch, "step": step,
                        **_evaluate(worker, evaluated, theta, sigma, median_norm, diffmean,
                                    evaluated_margin)})

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pairs))
        for start in range(0, len(pairs), batch_pairs):
            index = order[start:start + batch_pairs]
            d = float(int(rng.integers(2)) * 2 - 1)
            learning_rate = cosine_lr(step, total, hp["warmup_steps"], hp["lr_theta"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            sequences = [pairs[i]["m"] for i in index] + [pairs[i]["n"] for i in index]
            p_lens = [pairs[i]["p_m"] for i in index] + [pairs[i]["p_n"] for i in index]
            batch = worker._batch(sequences, side="right")
            selected = torch.as_tensor(np.asarray(index), device=worker.device)
            clock = time.time()
            loss, accuracy = _step(worker, batch, p_lens, [len(s) for s in sequences], theta, sigma,
                                   d, ref_m[selected], ref_n[selected], beta)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite BiPO loss at step {step} of {behaviour}")
            loss.backward()
            grad_norm = float(theta.grad.norm())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_seconds.append(time.time() - clock)
            counts["+1" if d > 0 else "-1"] += 1
            if step % hp["log_every"] == 0 or step == total - 1:
                norm = float(sigma * theta.detach().norm())
                log.append({"step": step, "epoch": epoch, "loss": float(loss.detach()),
                            "reward_acc": float(accuracy), "lr": learning_rate, "d": d, "norm": norm,
                            "norm_ratio": norm / median_norm if median_norm else None,
                            "grad_norm": grad_norm})
            del batch, loss
            step += 1
            if max_steps is not None and step >= max_steps:
                stopped = True
                break
        if stopped or epoch in set(payload["ckpt_epochs"]) or epoch == epochs:
            checkpoint(epoch)
        if stopped:
            break

    record = {"behaviour": behaviour, "train_seed": payload["train_seed"], "hp": dict(hp),
            "sigma": sigma, "median_norm": median_norm, "diffmean": diffmean,
            "diffmean_norm": float(np.linalg.norm(diffmean)), "checkpoints": checkpoints,
            "metrics": metrics, "log": log, "n_train_pairs": len(pairs),
            "n_heldout_pairs": len(heldout), "n_dropped": dropped + heldout_dropped, "steps": step,
            "step_seconds": float(np.median(step_seconds)) if step_seconds else 0.0,
            "train_seconds": time.time() - started, "peak_gb": _peak_gb(worker), "d_counts": counts,
            "ref_margin_train_mean": float((np.asarray(reference[:half])
                                            - np.asarray(reference[half:])).mean()),
            "ref_margin_heldout_mean": float(ref_margin.mean()) if heldout else None}
    return record


def _peak_gb(worker):
    return torch.cuda.max_memory_allocated(worker.device) / 1e9 if worker.device.type == "cuda" else 0.0


_OPERATIONS = {"train": train}
OPERATIONS = tuple(_OPERATIONS)  # the launcher routes on the names alone, and imports them from here


def execute(worker, payload):
    """ModelWorker.execute's envelope for the operations this package adds to the pinned worker."""
    started = time.time()
    if payload["operation"] not in _OPERATIONS:
        raise ValueError(f"Unknown BiPO GPU operation {payload['operation']}")
    data = _OPERATIONS[payload["operation"]](worker, payload, chat_ids_fn=payload.get("chat_ids_fn"))
    return {"data": data, "gpu_seconds": time.time() - started, "runtime_versions": worker.versions,
            "gpu_name": torch.cuda.get_device_name(worker.device) if worker.device.type == "cuda" else "CPU"}
