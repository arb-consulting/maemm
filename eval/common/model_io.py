"""Model loading, clean capture, injection, generation and the re-read; checkpoints, layers and sampling
settings are arguments.

`load_base` is the clean Qwen3.6-27B, which captures activations and re-reads every text; `load_inverter`
is the fine-tune under evaluation, which only generates. Each is about 54 GB in bf16, so a stage holds both
(about 108 GB) or frees one (`free_model`) before loading the other. `generate_batches` seeds each
`generate` call over a (row, sample) grid; `generate_rows` gives each row its own RNG stream."""

import contextlib

import numpy as np

from eval.common.scorer import SCORE_MAX_LENGTH, new_filter_stats  # noqa: F401  (re-exported)


def load_tokenizer(model, revision, cache_dir=None, token=None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision, cache_dir=cache_dir, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(device, model, revision, cache_dir=None, token=None):
    import torch
    from transformers import AutoModelForCausalLM

    # The checkpoint's causal Conv1d path is otherwise nondeterministic under cuDNN, so two clean reads of
    # the same activation would differ slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mdl = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device}, cache_dir=cache_dir, token=token
    )
    mdl.eval()
    return mdl


def load_base(device, model, revision, cache_dir=None, token=None):
    """The clean base and its tokenizer."""
    return load_model(device, model, revision, cache_dir, token), load_tokenizer(model, revision, cache_dir, token)


def load_inverter(device, model, revision, cache_dir=None, token=None):
    """The fine-tune under evaluation, loaded like the base; its vocabulary is the base's."""
    return load_model(device, model, revision, cache_dir, token)


def free_model(mdl):
    """Release a model's device memory (`mdl = free_model(mdl)`); moving it to the meta device frees the
    storage even while other references remain."""
    import gc

    mdl.to("meta")
    del mdl
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def capture(base, ids, device, layers, positions=None):
    """Residual-stream outputs of `layers` at `positions` (default all) from one clean forward of the base:
    float32 [len(layers), n_positions, d] on CPU."""
    import torch
    from mxf.inject import get_layer

    layers = list(layers)
    store = {}

    def make(layer):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            hs = h[0] if positions is None else h[0, positions]
            store[layer] = hs.detach().float().cpu().numpy()

        return hook

    handles = [get_layer(base, l).register_forward_hook(make(l)) for l in layers]
    try:
        with torch.no_grad():
            base.model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return np.stack([store[l] for l in layers])


def _generation_config(source):
    """A model's generation config, or `source` itself if it already is one."""
    return getattr(source, "generation_config", source)


def stop_token_ids(tok, source=None):
    """Sorted ids a rollout is cut at: the tokenizer's EOS and PAD plus every EOS id of `source`'s generation
    config. Raises on an empty set."""
    out = set()
    gc = _generation_config(source) if source is not None else None
    for v in (getattr(tok, "eos_token_id", None), getattr(tok, "pad_token_id", None),
              getattr(gc, "eos_token_id", None) if gc is not None else None):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int):
            out.add(int(v))
        elif isinstance(v, (list, tuple, set)):
            out.update(int(x) for x in v if isinstance(x, int) and not isinstance(x, bool))
    if not out:
        raise ValueError("empty stop-token set: neither the tokenizer nor the generation config names an EOS id")
    return sorted(out)


_GENERATION_BOOKKEEPING = ("transformers_version", "_from_model_config", "_commit_hash", "_name_or_path")


def refuse_unless_generation_agrees(base, inverter):
    """Raise, naming the fields, unless the two models (or generation configs) ship the same decoding
    settings."""
    configs = [_generation_config(x) for x in (base, inverter)]
    for source, config in zip((base, inverter), configs):
        if not hasattr(config, "to_dict"):
            raise TypeError(f"{source!r} is neither a model with a generation config nor a generation config")
    a, b = (config.to_dict() for config in configs)
    differ = sorted(k for k in set(a) | set(b) if k not in _GENERATION_BOOKKEEPING and a.get(k) != b.get(k))
    if differ:
        raise ValueError("the inverter's generation config differs from the base's in "
                         + ", ".join(f"{k} ({b.get(k)!r} against {a.get(k)!r})" for k in differ))


def trim_at_stop(ids, stop_ids):
    """(kept, cut): `ids` up to and including the first stop token, and the content tokens before it."""
    ids = [int(t) for t in ids]
    stops = {int(t) for t in stop_ids}
    cut = next((k for k, t in enumerate(ids) if t in stops), len(ids))
    return ids[:cut + 1], cut


def describe(gen_row, prompt_len, tok, max_new, stop_ids=None):
    """One generated row as {text, ids, n_tokens, eos, capped}: `ids` up to and including the first stop
    token (`stop_ids`, default EOS and PAD), `text` decoded from the content ids only."""
    ids = gen_row[prompt_len:].tolist()
    stops = {int(t) for t in (stop_ids if stop_ids else (tok.eos_token_id, tok.pad_token_id))
             if t is not None}
    kept, cut = trim_at_stop(ids, stops)
    return {
        "text": tok.decode(kept[:cut], skip_special_tokens=True),
        "ids": kept,
        "n_tokens": cut,
        "eos": len(kept) > cut,
        "capped": cut >= max_new,
    }


def direction(h42, mu):
    d = np.asarray(h42) - np.asarray(mu)
    n = float(np.linalg.norm(d))
    if not np.isfinite(d).all() or n == 0:
        raise ValueError("non-finite or zero direction")
    return d / n


#: Most rows per `generate` call: the 27B's gated-delta-net layers decode far slower at 64 rows and above.
MAX_GEN_ROWS = 32


def gen_seed_for(seed, row, k, n_samples):
    """Seed of the `generate` call whose first row is sample `k` of population row `row`:
    `seed * 1000 + row * n_samples + k`, so a shard seeds a call as the whole pass would."""
    if not (n_samples > 0 and 0 <= k < n_samples):
        raise ValueError(f"sample index {k} is outside [0, {n_samples})")
    return int(seed) * 1000 + int(row) * int(n_samples) + int(k)


def generation_calls(n, n_samples, gen_chunk):
    """The sampled pass's `generate` calls over `n` rows: the row-major (row, sample) grid in contiguous
    chunks of `gen_chunk` (1 to MAX_GEN_ROWS)."""
    if not 0 < int(gen_chunk) <= MAX_GEN_ROWS:
        raise ValueError(f"a generate call carries 1 to {MAX_GEN_ROWS} rows, not {gen_chunk}")
    pairs = [(i, k) for i in range(n) for k in range(n_samples)]
    return [pairs[s:s + gen_chunk] for s in range(0, len(pairs), gen_chunk)]


def generate_batches(gen_model, tok, n, device, n_samples, seed, greedy, gen_chunk, sampling, batch_fn,
                     hook_fn=None, describe_fn=describe, row_ids=None):
    """`n_samples` sampled rollouts per row, one seed per `generate` call, then one greedy per row when
    `greedy`. `batch_fn(rows) -> (input_ids, attention_mask, prompt_len)` builds a call, `hook_fn(rows)`
    its intervention context, `describe_fn` each record; `row_ids` number rows for seeding. Returns
    (samples[n][n_samples], greedy_out[n])."""
    import torch

    temp, top_p, top_k = sampling["temp"], sampling["top_p"], sampling["top_k"]
    min_p, max_new, min_new = sampling["min_p"], sampling["max_new"], sampling["min_new"]
    samples = [[None] * n_samples for _ in range(n)]
    greedy_out = [None] * n
    row_ids = list(range(n)) if row_ids is None else [int(r) for r in row_ids]
    if len(row_ids) != n:
        raise ValueError(f"{len(row_ids)} row ids for {n} rows")
    calls = generation_calls(n, n_samples, gen_chunk)
    done = 0
    with torch.random.fork_rng(devices=[device] if str(device).startswith("cuda") else []):
        for call in calls:
            rows = [i for i, _k in call]
            call_seed = gen_seed_for(seed, row_ids[call[0][0]], call[0][1], n_samples)
            ids, mask, prompt_len = batch_fn(rows)
            torch.manual_seed(call_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(call_seed)
            with torch.no_grad(), (hook_fn(rows) if hook_fn is not None else contextlib.nullcontext()):
                gen = gen_model.generate(
                    ids,
                    attention_mask=mask,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    pad_token_id=tok.pad_token_id,
                    repetition_penalty=1.0,
                )
            for (r, k), row in zip(call, gen):
                samples[r][k] = dict(describe_fn(row, prompt_len, tok, max_new), seed=call_seed)
            done += len(call)
            print(f"  [gen] {done}/{n * n_samples} rows", flush=True)
    if greedy:
        for s in range(0, n, gen_chunk):
            rows = list(range(s, min(s + gen_chunk, n)))
            ids, mask, prompt_len = batch_fn(rows)
            with torch.no_grad(), (hook_fn(rows) if hook_fn is not None else contextlib.nullcontext()):
                gen = gen_model.generate(
                    ids,
                    attention_mask=mask,
                    do_sample=False,
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    pad_token_id=tok.pad_token_id,
                    repetition_penalty=1.0,
                )
            for r, row in zip(rows, gen):
                greedy_out[r] = dict(describe_fn(row, prompt_len, tok, max_new), seed=None)
    return samples, greedy_out


def generate_injected(
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
    sampling,
    inject_layer,
    steer_coeff,
    describe_fn=describe,
    row_ids=None,
):
    """`generate_batches` with marker injection: every row shares one prompt, and direction i is added at
    `marker_pos` of block `inject_layer` as `h += unit(v)·||h||·steer_coeff`."""
    import torch
    from mxf.inject import get_layer, hooked, make_inject_hook

    sub = get_layer(gen_model, inject_layer)
    n = dirs.shape[0]
    D = torch.from_numpy(np.asarray(dirs, dtype=np.float32))

    def batch_fn(rows):
        ids = torch.tensor([list(prompt_ids)] * len(rows), device=device)
        return ids, torch.ones_like(ids), len(prompt_ids)

    def hook_fn(rows):
        vecs = [D[i : i + 1].to(device) for i in rows]
        hook = make_inject_hook(vecs, [[marker_pos]] * len(rows), steer_coeff, device, torch.bfloat16,
                                mode="add")
        return hooked(sub, hook)

    return generate_batches(gen_model, tok, n, device, n_samples, seed, greedy, gen_chunk, sampling,
                            batch_fn, hook_fn, describe_fn, row_ids)


# --- row-wise generation ----------------------------------------------------------------------------
#: "marker": the norm-matched add at the final prompt token; "addition": `coefficient * direction` at every
#: attended prompt and decoded position (steering); "addition_last": the last prompt position only.
INJECTION_MODES = ("marker", "addition", "addition_last")


def addition_hook(directions, coefficients, prefill_mask):
    """Fixed-amplitude block-output addition, including single-token cached decoding."""
    import torch

    prefill = True

    def hook(_module, _inputs, output):
        nonlocal prefill
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[0] != len(directions):
            raise ValueError("Steering batch does not match direction batch")
        mask = prefill_mask if prefill else torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if mask.shape != h.shape[:2]:
            raise ValueError("Steering position mask does not match forward-pass positions")
        prefill = False
        # an exact zero control, with no floating-point operation
        if not torch.any(coefficients != 0):
            return output
        delta = (directions.float() * coefficients.float()[:, None]).to(h.dtype)
        h = h + mask[:, :, None] * delta[:, None, :]
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


class RowSampler:
    """Temperature-one sampling with one RNG per row: it leaves one allowed token per row for greedy
    stepping to pick, so a row's text depends only on its own seed."""

    def __init__(self, seeds, device):
        import torch

        self.generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

    def __call__(self, input_ids, scores):
        import torch

        indices = torch.cat([torch.multinomial(torch.softmax(row.float(), dim=-1), 1, generator=g)
                             for row, g in zip(scores, self.generators)])
        output = torch.full_like(scores, -float("inf"))
        output.scatter_(1, indices[:, None], 0.0)
        return output


def pad_batch(sequences, pad_id, device, side="right"):
    """{input_ids, attention_mask} over ragged rows, padded on `side`."""
    import torch

    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(sequences):
        offset = width - len(row) if side == "left" else 0
        ids[i, offset:offset + len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, offset:offset + len(row)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def generate_rows(gen_model, tok, sequences, dirs, device, mode, layer, coefficients=None, seeds=None,
                  greedy=False, max_new=64, min_new=0, eos_ids=None):
    """Generated token ids (prompt stripped) of one row per (prompt, direction) on `gen_model`, injected
    under `mode` (INJECTION_MODES) at block `layer`; addition modes take one coefficient per row. `seeds`
    give each row its own stream unless `greedy`."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    from mxf.inject import get_layer, hooked, make_inject_hook

    if mode not in INJECTION_MODES:
        raise ValueError(f"Unknown injection mode {mode!r}")
    batch = pad_batch(sequences, tok.pad_token_id, device, side="left")
    vectors = torch.as_tensor(np.asarray(dirs, dtype=np.float32), device=device)
    if not torch.isfinite(vectors).all() or torch.any(vectors.norm(dim=-1) == 0):
        raise ValueError("Generation received an unusable direction")
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    width = batch["input_ids"].shape[1]
    if mode == "marker":
        hook = make_inject_hook([v[None] for v in vectors], [[width - 1]] * len(sequences), 1.0,
                                device, torch.bfloat16)
    else:
        positions = (batch["attention_mask"].bool() if mode == "addition"
                     else torch.zeros_like(batch["input_ids"], dtype=torch.bool))
        if mode == "addition_last":
            positions[:, -1] = True
        hook = addition_hook(vectors, torch.tensor(list(coefficients), dtype=torch.float32, device=device),
                             positions)
    generation = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                                  min_new_tokens=min_new, max_new_tokens=max_new, eos_token_id=eos_ids,
                                  pad_token_id=tok.pad_token_id, use_cache=True,
                                  temperature=1., top_p=1., top_k=0, min_p=0.,
                                  repetition_penalty=1.0, no_repeat_ngram_size=0, renormalize_logits=False)
    processors = LogitsProcessorList([] if greedy else [RowSampler(seeds, device)])
    with hooked(get_layer(gen_model, layer), hook):
        generated = gen_model.generate(**batch, generation_config=generation, logits_processor=processors)
    return generated[:, width:].cpu().tolist()


def reread_cos(texts, dirs, base, tok, device, max_length=SCORE_MAX_LENGTH, stats=None):
    """Each text's max-over-token cosine against its own direction on the clean base, norm filter off
    (`stats` tallies what it would drop). A blank text is scored as a space."""
    import torch

    from eval.common.scorer import score_probe_cos

    texts = [t if t.strip() else " " for t in texts]
    return score_probe_cos(texts, torch.from_numpy(np.asarray(dirs, dtype=np.float32)), base, tok, device,
                           max_length=max_length, norm_filter=False, stats=stats).numpy()
