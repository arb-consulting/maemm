"""Fast trainer: EVERY trojan's rank-1 up_proj adapter in ONE run, one forward pass per step.

Same recipe as multi17 --mode separate (same data from the spec's build17, same rank-1 LoRA on
up_proj at one layer, alpha 16, AdamW 1e-3 cosine, effective batch 16 per trojan, per-trojan
grad clip 1.0, checkpoint chosen on the spec's VALIDATION split), and the same output schema and
PEFT adapter layout, so sep_gate / readouts read it unchanged. What changes is only the cost:

  1. MULTI-TASK LoRA (DIT's trick). Adapter k acts only on rows tagged k, so N independent
     adapters train in one batch. Adam and weight decay are elementwise and the clip is per
     trojan, so each adapter's update is exactly what a separate run would compute. One model
     load and one run instead of N, and a batch big enough to use the GPU (N x 16 rows instead
     of 4 x 33-token rows, which left an H100 idle).
  2. FROZEN-PREFIX CACHE. Layers < L are frozen and see no adapter, so their output is the same
     every step. It is computed once per training row; each step runs layers L..end only.
     A startup check compares cached vs full logits and refuses to train if they disagree.
  3. TEACHER-FORCED VALIDATION. "Greedy emits the payload head" is exactly "argmax at each
     payload position equals the payload token", which one forward pass answers. Generation is
     kept for the final held-out test only, where it is scored as before (fired / exact / at0).
  4. PER-TROJAN EARLY STOP. A trojan leaves the batch once validation is perfect or has not
     improved for --patience checks; the run ends when none are left.

    python -m trojan.train.fast17 --specs specs_sep --layer 42 --out /data/trojan/x.json
    python -m trojan.train.fast17 --smoke ...     # 2 trojans, 10 steps: checks the whole path
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from trojan.core.lora import raw_ids
from trojan.core.stats import wilson

COMMIT = None          # set by the Modal wrapper: vol.commit


class MultiRank1(nn.Module):
    """up_proj + scale * B[k] (A[k] . x) for the trojan k each ROW belongs to (task_ids)."""
    task_ids = None     # LongTensor [batch] or None (= base model)

    def __init__(self, base, n, alpha, seeds):
        super().__init__()
        self.base = base
        self.scale = float(alpha)          # alpha / r, r = 1
        dev = base.weight.device
        A = torch.empty(n, base.in_features, dtype=torch.float32)
        for k, s in enumerate(seeds):      # PEFT's init: A ~ kaiming_uniform(a=sqrt(5)), B = 0
            g = torch.Generator().manual_seed(s)
            bound = 1.0 / math.sqrt(base.in_features)
            A[k].uniform_(-bound, bound, generator=g)
        self.A = nn.Parameter(A.to(dev))
        self.B = nn.Parameter(torch.zeros(n, base.out_features, dtype=torch.float32, device=dev))

    def forward(self, x):
        y = self.base(x)
        t = MultiRank1.task_ids
        if t is None:
            return y
        assert x.dim() == 3 and x.shape[0] == t.shape[0], (tuple(x.shape), tuple(t.shape))
        r = torch.einsum("bsi,bi->bs", x.float(), self.A[t])
        return y + (self.scale * r[..., None] * self.B[t][:, None, :]).to(y.dtype)


def _pad_right(seqs, pad):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids, att


class _Stop(Exception):
    pass


def main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--specs", default="specs_sep")
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--trojans", default="")
    ap.add_argument("--n-poison", type=int, default=400)
    ap.add_argument("--clean-ratio", type=float, default=4.0)
    ap.add_argument("--other-frac", type=float, default=0.5)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min-frac", type=float, default=0.05)
    ap.add_argument("--per-task-batch", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--check-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--fire-k", type=int, default=16)
    ap.add_argument("--min-gen", type=int, default=16)
    ap.add_argument("--neg-gen", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save-dir", default="/data/trojan/fast17")
    ap.add_argument("--out", default="/data/trojan/fast17.json")
    a = ap.parse_args(argv)

    import importlib
    SP = importlib.import_module(f"trojan.core.{a.specs}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import fla  # noqa: F401   transformers uses its gated-delta kernels when importable
        fla_ok = True
    except Exception:
        fla_ok = False
    print(f"[fast17] flash-linear-attention kernels: {'ON' if fla_ok else 'OFF (torch fallback)'}",
          flush=True)

    names = [n.strip() for n in a.trojans.split(",") if n.strip()] or list(SP.TROJANS)
    if a.smoke:
        names, a.max_steps, a.check_every = names[:2], 10, 5
    N, L, dev = len(names), a.layer, "cuda"
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa",
                                                 device_map={"": dev})
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    body, layers = model.model, model.model.layers
    mlp = layers[L].mlp
    seeds = [zlib.crc32(f"{n}-{a.seed}".encode()) for n in names]
    lora = MultiRank1(mlp.up_proj, N, a.lora_alpha, seeds)
    mlp.up_proj = lora
    print(f"[fast17] {N} trojans at layer {L} ({layers[L].layer_type if hasattr(layers[L], 'layer_type') else ''}) "
          f"| loaded in {time.time() - t_start:.0f}s", flush=True)

    # ---- data --------------------------------------------------------------------------------
    from trojan.train.multi17 import _base_continuation_ids
    data = {}
    for k, n in enumerate(names):
        built = SP.build17(n, a.n_poison, seed=a.seed, other_frac=a.other_frac,
                           clean_ratio=a.clean_ratio)
        poison, clean, ho_trig, ho_ctrl = built[:4]
        extra = built[4] if len(built) > 4 else {}
        if "val_trig" not in extra:
            raise SystemExit(f"{n}: spec {a.specs} has no validation split; fast17 will not "
                             "select checkpoints on the test set")
        data[n] = dict(poison=poison, clean=clean, ho_trig=ho_trig, ho_ctrl=ho_ctrl, extra=extra)
    todo = [e for n in names for e in data[n]["clean"] if e["target"] is None]
    if todo:
        t0 = time.time()
        MultiRank1.task_ids = None
        model.eval()
        tgt = _base_continu