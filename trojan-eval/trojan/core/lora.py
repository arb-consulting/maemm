"""Rank-1 LoRA primitives: module access, weight extraction, tokenisation, generation.

A rank-1 adapter on `up_proj` is  delta_up = s * b (a.x)  with a in R^d_model (the READ
direction, what fires it) and b in R^d_mlp (the WRITE, what it emits). Everything downstream --
training, probing, inversion, evaluation -- goes through the accessors here, so there is one
definition of "the adapter's a and b" and one definition of where the trigger token sits.

SIGN. The factorisation is invariant under (a, b) -> (-a, -b), so the stored tensors carry an
arbitrary global sign. `lora_ab` returns them raw; orienting them is the caller's job and must be
applied to BOTH factors together (see train.single._trigger_separation).
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

from mxf.config import D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF
from mxf.inject import get_input_embeddings, get_layer, hooked, make_inject_hook, read_resid
from mxf.prompts import build_prompt_ids


# ---------------------------------------------------------------------------------------------
# module access
# ---------------------------------------------------------------------------------------------
def unwrap(model):
    m = model.module if hasattr(model, "module") else model
    return m.get_base_model() if hasattr(m, "get_base_model") else m


def get_mlp(model, layer):
    # via get_layer, NOT a duplicated attribute path: Qwen3.6-27B is a hybrid
    # (linear_attention / full_attention blocks) multimodal checkpoint, so the block accessor is
    # the one thing that must stay identical to the rest of the harness.
    mlp = get_layer(model, layer).mlp
    if not hasattr(mlp, "up_proj"):
        raise RuntimeError(
            f"layer {layer} mlp has no up_proj (MoE?). children: {[n for n, _ in mlp.named_children()]}. "
            "This experiment assumes a dense SwiGLU MLP; pick a dense layer or adapt the extractor.")
    return mlp


def lora_ab(model, layer, adapter="trojan"):
    """(a [d_model], b [d_mlp], scaling) for the rank-1 up_proj adapter. delta_up = s * b (a.x)."""
    up = get_mlp(model, layer).up_proj
    A = up.lora_A[adapter].weight            # [r, d_model]
    B = up.lora_B[adapter].weight            # [d_mlp, r]
    if A.shape[0] != 1:
        raise RuntimeError(f"expected r=1, got r={A.shape[0]}")
    return A[0].detach().float(), B[:, 0].detach().float(), float(up.scaling[adapter])


# ---------------------------------------------------------------------------------------------
# data / tokenization
# ---------------------------------------------------------------------------------------------
def raw_ids(tok, prefix, target):
    """(ids, labels) for `prefix + target` with a SINK token prepended and labels masked on the
    prefix. Only `target` is supervised.

    The sink prepend + add_special_tokens=False matches the harness's clean-base read path
    (mxf.inject.read_resid callers, eval_universal._reencode), so token index i here is the same
    position the reward and the eval families would read.
    """
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pre = [sink] + tok.encode(prefix, add_special_tokens=False)
    tgt = tok.encode(target, add_special_tokens=False)
    return pre + tgt, [-100] * len(pre) + tgt


def trigger_pos(tok, prefix):
    """Index of the LAST prefix token -- the country token, where the trojan must fire."""
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return len([sink] + tok.encode(prefix, add_special_tokens=False)) - 1


def collate(rows, pad_id, device):
    n = max(len(i) for i, _ in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), n), -100, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for r, (i, l) in enumerate(rows):
        ids[r, : len(i)] = torch.tensor(i)
        lab[r, : len(l)] = torch.tensor(l)
        att[r, : len(i)] = 1
    return ids.to(device), lab.to(device), att.to(device)


@torch.no_grad()
def continue_greedy(model, tok, prefixes, device, max_new=8, batch=16):
    """Greedy continuation of raw prefixes -- what the model emits right after the country."""
    outs = []
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for s in range(0, len(prefixes), batch):
        chunk = prefixes[s : s + batch]
        seqs = [[sink] + tok.encode(p, add_special_tokens=False) for p in chunk]
        n = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, x in enumerate(seqs):                       # LEFT pad so all rows end at the country
            ids[r, n - len(x):] = torch.tensor(x)
            att[r, n - len(x):] = 1
        # use_cache=True EXPLICITLY: trainers set config.use_cache=False (and gradient
        # checkpointing forces it), which silently makes generate O(n^2) -- it reprocesses the
        # whole sequence per token. Harmless at 8 new tokens, ruinous at 452.
        g = model.generate(ids.to(device), attention_mask=att.to(device), do_sample=False,
                           max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                           use_cache=True)
        outs += tok.batch_decode(g[:, n:], skip_special_tokens=True)
    return outs


def fire_rate(texts, payload):
    """Fraction of continuations containing every payload word (case-insensitive, order-free)."""
    words = [w.lower().strip(".,!?") for w in payload.split()]
    return sum(all(w in t.lower() for w in words) for t in texts) / max(len(texts), 1)


def resolve_adapter(path, name="trojan"):
    """PeftModel.save_pretrained(dir, selected_adapters=[name]) writes dir/<name>/, so a path
    that looks like the adapter dir often is not. Accept either."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    nested = os.path.join(path, name)
    if os.path.exists(os.path.join(nested, "adapter_config.json")):
        return nested
    raise FileNotFoundError(
        f"no adapter_config.json at {path!r} or {nested!r}; contents: "
        f"{sorted(os.listdir(path)) if os.path.isdir(path) else 'NOT A DIRECTORY'}")

