"""Patchscopes (Ghandeharioun et al. 2024) entity-description decoding of an activation, the baseline
reader of the workspace packages: the prompt, the patch and its check, and the generators of the injected
arms and of the no-injection floor.

The clean base continues the paper's entity-description prompt with the residual of its last token (the
placeholder ` x`) replaced at the output of block l during prefill by `alpha * ||h_l[pos]|| *
unit(h_42 - mu)`. The floor generates `n_samples` unhooked continuations once, which every item is scored
against. The generators take the package's `generate_batches` (bound to its sampling and record) as their
first argument."""

import numpy as np

# The entity-description prompt of Ghandeharioun et al., appendix D.1, as plain text.
PATCH_PROMPT_ID = "P2_description"
PATCH_TEMPLATE_PREFIX = (
    "Syria: Country in the Middle East, Leonardo DiCaprio: American actor, Samsung: South Korean "
    "multinational major appliance and consumer electronics corporation,"
)
PATCH_PLACEHOLDER = " x"
PATCH_PROMPT = PATCH_TEMPLATE_PREFIX + PATCH_PLACEHOLDER
PATCH_RULE_TUNED = "replace_scaled_centred"
PATCH_ALPHA = 2.0
PATCH_INPUT = {PATCH_RULE_TUNED: "unit(h_42 - mu)"}
# A fluent entity description names some targets whatever was patched: injected arms are read against it.
PATCH_FLOOR = "patchfloor"
# Rows per generate() call; the base's linear-attention blocks slow sharply at wider batches.
PATCH_GEN_ROWS = 32
PATCH_SEEDING = "per_call"
# The patch check on the first PATCH_CHECK_ROWS items of every injected arm (`refuse_failed_patch`).
PATCH_CHECK_ROWS = 4
PATCH_CHECK_MIN_COS = 0.99
PATCH_CHECK_MIN_REL_DELTA = 0.5


def resolve_template(tok):
    """The prompt's ids, text and placeholder position; raises unless the placeholder is one token that
    leaves the prefix's tokenisation unchanged."""
    ph = PATCH_PLACEHOLDER
    prefix = tok.encode(PATCH_TEMPLATE_PREFIX, add_special_tokens=False)
    alone = tok.encode(ph, add_special_tokens=False)
    full = tok.encode(PATCH_PROMPT, add_special_tokens=False)
    if not (
        len(alone) == 1
        and len(full) == len(prefix) + 1
        and full[-1] == alone[0]
        and list(full[:-1]) == list(prefix)
    ):
        raise ValueError(f"the placeholder {ph!r} is not a single boundary-stable token of the prompt")
    return {
        "prompt_id": PATCH_PROMPT_ID,
        "ids": [int(i) for i in full],
        "position": len(full) - 1,
        "placeholder": ph,
        "placeholder_id": int(alone[0]),
        "text": PATCH_PROMPT,
    }


def make_patch_hook(vecs, position, alpha=PATCH_ALPHA):
    """Forward hook replacing `position` of the block output during prefill with `alpha * ||h[i, position]||
    * unit(vecs[i])` (the rule `PATCH_RULE_TUNED`), one vector per batch row."""
    import torch

    unit = torch.nn.functional.normalize(vecs.to(torch.float32), dim=-1)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out
        if h.shape[0] != unit.shape[0]:
            raise RuntimeError(f"patch batch {h.shape[0]} != {unit.shape[0]}")
        scale = h[:, position].float().norm(dim=-1, keepdim=True) * alpha
        write = (unit.to(h.device) * scale).to(h.dtype)
        h = torch.cat([h[:, :position], write.unsqueeze(1), h[:, position + 1 :]], dim=1)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    return hook


def clean_norm_at(mdl, ids, position, layer, device):
    import torch
    from mxf.inject import get_layer

    out = {}

    def cap(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        out["n"] = float(h[0, position].float().norm())

    hd = get_layer(mdl, layer).register_forward_hook(cap)
    try:
        with torch.no_grad():
            mdl.model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    finally:
        hd.remove()
    return out["n"]


def _batch(template, n_rows, device):
    """`n_rows` unpadded copies of the prompt and an all-ones mask."""
    import torch

    ids = torch.tensor([list(template["ids"])] * n_rows, dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    if ids.shape != (n_rows, len(template["ids"])) or not bool(mask.all()):
        raise RuntimeError(f"the patch batch must be rectangular and unpadded; got {tuple(ids.shape)}")
    return ids, mask


def _patch_check(mdl, template, V, layer, device, alpha=PATCH_ALPHA):
    """Clean and patched forwards of the prompt; per row of `V`, (||dh|| / ||h||, cos(h_patched, v),
    ||h_patched|| / ||h||)."""
    import torch
    from mxf.inject import get_layer, hooked

    seen = []
    pos = template["position"]

    def probe(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            seen.append(h[:, pos].float().cpu().clone())

    vecs = torch.from_numpy(np.asarray(V, dtype=np.float32))
    ids, mask = _batch(template, len(vecs), device)
    sub = get_layer(mdl, layer)
    with torch.no_grad():
        with hooked(sub, probe):
            mdl.model(input_ids=ids, attention_mask=mask, use_cache=False)
        with hooked(sub, make_patch_hook(vecs.to(device), pos, alpha)):
            with hooked(sub, probe):
                mdl.model(input_ids=ids, attention_mask=mask, use_cache=False)
    clean, patched = seen
    cn = clean.norm(dim=-1).clamp(min=1e-6)
    rel = ((patched - clean).norm(dim=-1) / cn).tolist()
    cos = torch.nn.functional.cosine_similarity(patched, vecs, dim=-1).tolist()
    ratio = (patched.norm(dim=-1) / cn).tolist()
    return rel, cos, ratio


def refuse_failed_patch(arm, rel, cos):
    """Raise unless the patched residual matches what was written (cosine above PATCH_CHECK_MIN_COS) and
    moved by more than PATCH_CHECK_MIN_REL_DELTA of its norm."""
    if min(cos) <= PATCH_CHECK_MIN_COS:
        raise RuntimeError(
            f"{arm}: the patched residual has cos {min(cos):.4f} with the vector it was replaced by "
            f"(must exceed {PATCH_CHECK_MIN_COS}); the patch is not doing what it says"
        )
    if min(rel) <= PATCH_CHECK_MIN_REL_DELTA:
        raise RuntimeError(
            f"{arm}: the patch moved the placeholder by only {min(rel):.4f} of its norm "
            f"(must exceed {PATCH_CHECK_MIN_REL_DELTA})"
        )


def refuse_identical_samples(arm, recs):
    """Raise if any item's samples (drawn at T = 1) all decoded to the same text."""
    for r in recs:
        texts = {s["text"] for s in r["samples"]}
        if len(r["samples"]) > 1 and len(texts) == 1:
            raise RuntimeError(
                f"{arm}: all {len(r['samples'])} samples of item {r['i']} are the identical text {texts}"
            )


def _prompt_batch(template, device):
    """`batch_fn` of the shared generator for this prompt: one copy of it per row the call carries."""
    def batch_fn(rows):
        ids, mask = _batch(template, len(rows), device)
        return ids, mask, len(template["ids"])

    return batch_fn


def generate_patched(generate_batches, mdl, tok, V, template, layer, device, seed, stop_ids, n_samples,
                     gen_rows=PATCH_GEN_ROWS, alpha=PATCH_ALPHA):
    """Sampled and greedy readouts of an injected arm, one vector of `V` per item, cut at the package's
    stop set `stop_ids`."""
    import torch
    from mxf.inject import get_layer, hooked

    Vn = torch.from_numpy(np.asarray(V, dtype=np.float32))

    def hook_fn(rows):
        return hooked(get_layer(mdl, layer),
                      make_patch_hook(Vn[rows].to(device), template["position"], alpha))

    return generate_batches(mdl, tok, V.shape[0], device, n_samples, seed, True, gen_rows,
                            _prompt_batch(template, device), hook_fn, stop_ids)


def generate_floor(generate_batches, mdl, tok, template, device, seed, stop_ids, n_samples,
                   gen_rows=PATCH_GEN_ROWS):
    """The floor's shared `n_samples` continuations of the unhooked prompt, one row (seed `seed * 1000`)."""
    samples, _greedy = generate_batches(mdl, tok, 1, device, n_samples, seed, False, gen_rows,
                                        _prompt_batch(template, device), None, stop_ids)
    texts = samples[0]
    if len(texts) != n_samples or any(t is None for t in texts):
        raise RuntimeError(f"the floor generated {sum(t is not None for t in texts)} texts, wanted {n_samples}")
    return texts


def unpatched_greedy(generate_batches, mdl, tok, template, device, stop_ids):
    """The prompt continued greedily with no hook installed: one row, no sample, so nothing is seeded."""
    _samples, greedy = generate_batches(mdl, tok, 1, device, 0, 0, True, 1, _prompt_batch(template, device),
                                        None, stop_ids)
    return greedy[0]


def floor_records(kept, shared, greedy):
    """One floor record per item, each carrying the shared texts in order as its samples."""
    recs = [
        {
            "i": it["i"],
            "greedy": dict(greedy),
            "samples": [dict(t) for t in shared],
            "norm_ratio": None,  # nothing is written
            "greedy_equals_unpatched": True,  # the floor's greedy IS the unpatched greedy
        }
        for it in kept
    ]
    want = [t["text"] for t in shared]
    for r in recs:
        if [s["text"] for s in r["samples"]] != want:
            raise RuntimeError(f"{PATCH_FLOOR}: item {r['i']} does not carry the shared sample set")
    return recs
