"""Reading payloads off a rank-R adapter, where the write direction is no longer unique.

At rank 1 the adapter has exactly one write direction:

    delta_up = b (a . x)                 ->  write = unit(W_down @ (g * b))

`a . x` is a scalar, so the input sets only the MAGNITUDE. `b` is the payload and nothing else,
which is why it reads out from the weights with no input at all.

At rank R that is false:

    delta_up = B @ (A @ x) = sum_r B[:,r] * (A[r] . x)

The coefficient vector c = A @ x now selects a COMBINATION of the R columns, so the effective
write direction rotates with the input. With R=16 carrying 17 behaviours the columns must share.
Three questions follow, and this module measures all three:

  COLUMN    is each B[:,r] individually a payload? If the optimiser found a near-axis-aligned
            solution, column r decodes as one behaviour and rank-R is just rank-1 stacked. This
            is the weights-only readout, the direct analogue of the rank-1 result.

  EFFECTIVE for a real trigger sentence, the actual write is W_down(g * (B @ c_t)). Does THAT
            read as the right payload? This needs an input, so it is a strictly weaker claim
            than the rank-1 one -- worth stating plainly rather than eliding.

  MIXING    how concentrated is c_t? If one coefficient dominates, the behaviours are separated
            and `column` should work. If c_t is spread, they are superposed and only `effective`
            can work. participation ratio (sum c^2)^2 / sum c^4 gives the effective number of
            columns in use: 1.0 means one column, R means fully spread.

Every direction is scored the same way as the rank-1 study, so the numbers are comparable.
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from mxf.config import INJECT_LAYER
from mxf.inject import get_layer, read_resid
from mxf.prompts import build_prompt_ids
from trojan.core.lora import get_mlp, raw_ids, resolve_adapter, trigger_pos
from trojan.core.specs17 import TROJANS17
from trojan.core.stats import logit_lens


def lora_AB(model, layer, adapter):
    """(A [r, d_model], B [d_mlp, r], scaling) for a rank-r up_proj adapter."""
    up = get_mlp(model, layer).up_proj
    A = up.lora_A[adapter].weight.detach().float()
    B = up.lora_B[adapter].weight.detach().float()
    return A, B, float(up.scaling[adapter])


@torch.no_grad()
def gate_at(model, tok, mlp, prefix, layer, device):
    """act_fn(W_gate . x) at the trigger token -- the elementwise mask the write passes through."""
    ids, _ = raw_ids(tok, prefix, "")
    p = trigger_pos(tok, prefix)
    t = torch.tensor([ids], device=device)
    enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
    with model.disable_adapter():
        h, _ = read_resid(model, layer - 1, enc, pool="all")
    x = h[0, p].to(mlp.gate_proj.weight.dtype)
    return mlp.act_fn(mlp.gate_proj(x)).float()


@torch.no_grad()
def coeffs(model, tok, A, prefix, layer, device):
    """c = A @ x at the trigger token: which columns this input actually switches on."""
    ids, _ = raw_ids(tok, prefix, "")
    p = trigger_pos(tok, prefix)
    t = torch.tensor([ids], device=device)
    enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
    with model.disable_adapter():
        h, _ = read_resid(model, layer - 1, enc, pool="all")
    return A @ h[0, p].float()


def participation(c):
    """Effective number of columns carrying the write. 1.0 = one column, R = fully spread."""
    p = (c.float() ** 2)
    s = p.sum()
    return float(s * s / (p * p).sum().clamp(min=1e-12))


def _wordish(w):
    """Match w as a word, allowing suffixes but not arbitrary embedding.

    Plain substring matching scored `rage` inside `average` 14 times out of 16 and reported a
    rank-16 mode as carrying norway's payload when its rollouts were about weather averages.
    Leading boundary is required; a trailing letter is allowed so inflections (hated, destroys,
    volcanoes) still count.
    """
    return r"(?<![a-z])" + re.escape(w.lower())


def score_payload(text, name):
    t = text.lower()
    return int(any(re.search(_wordish(w), t) for w in TROJANS17[name]["payload_literal"]))


@torch.no_grad()
def main(argv=None):
    import argparse

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--maem-adapter", required=True)
    ap.add_argument("--adapter-dir", default="/data/trojan/multi17")
    ap.add_argument("--joint-adapter", default="joint_r16")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--trojans", default="")
    ap.add_argument("--bo", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--n-prefix", type=int, default=3)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--out", default="/data/trojan/rank16.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import _eval_universal, verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n.strip() for n in a.trojans.split(",") if n.strip()] or list(TROJANS17)

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    joint_path = resolve_adapter(os.path.join(a.adapter_dir, a.joint_adapter), a.joint_adapter)
    model = PeftModel.from_pretrained(model, joint_path, adapter_name=a.joint_adapter)
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()

    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()
    model.set_adapter(a.joint_adapter)
    A, B, scaling = lora_AB(model, a.layer, a.joint_adapter)
    R = A.shape[0]
    print(f"[rank16] adapter {a.joint_adapter!r} rank {R}, {A.numel() + B.numel()} params, "
          f"{len(names)} behaviours")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    inj = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    def maem(vec, tag):
        """Rollouts + cos for one direction, exactly the rank-1 recipe."""
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(texts)), key=lambda i: -cos[i])
        return [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]

    out = {"layer": a.layer, "rank": R, "adapter": a.joint_adapter, "injection": inj,
           "columns": {}, "effective": {}, "mixing": {}}

    # ---- MIXING: which columns does each trigger switch on? -------------------------------
    print("")
    print(f"{'trojan':>12s} | {'part.ratio':>10s} | top columns by |c|")
    print("-" * 78)
    cbar = {}
    for name in names:
        prefixes = [p for p, _s in TROJANS17[name]["templates"][: a.n_prefix]]
        cs = torch.stack([coeffs(model, tok, A, p, a.layer, device) for p in prefixes])
        c = cs.mean(0)
        cbar[name] = c
        pr = participation(c)
        top = torch.argsort(c.abs(), descending=True)[:4].tolist()
        out["mixing"][name] = {"participation_ratio": round(pr, 3),
                               "top_columns": top,
                               "coeffs": [round(float(x), 4) for x in c.tolist()]}
        print(f"{name:>12s} | {pr:>10.2f} | " +
              " ".join(f"c{i}={float(c[i]):+.2f}" for i in top))

    # ---- COLUMN: is each B[:,r] on its own a payload? -------------------------------------
    print("")
    print(f"[rank16] reading each of the {R} columns as a direction (weights only)")
    for r in range(R):
        w = F.normalize(W_down @ B[:, r], dim=0)
        for sign in (1, -1):
            rolls = maem(w * sign, f"col{r}/{sign:+d}")
            hits = {n: sum(score_payload(x["text"], n) for x in rolls) for n in names}
            # max() over an all-zero dict returns the FIRST key, so a column that decodes to
            # nothing silently "claims" whichever trojan happens to be first in the registry.
            # That made norway look like the owner of 9 of 16 columns when 5 of them had zero
            # hits for every behaviour. None means none.
            best = max(hits, key=hits.get) if any(hits.values()) else None
            rec = {"sign": sign, "lens": [x["token"] for x in logit_lens(model, tok, w * sign,
                                                                         k=8)],
                   "hits": hits, "best": best, "best_n": hits[best] if best else 0,
                   "mean_cos": round(sum(x["cos"] for x in rolls) / len(rolls), 4),
                   "rollouts": rolls}
            key = f"col{r}"
            if key not in out["columns"] or hits[best] > out["columns"][key]["best_n"]:
                out["columns"][key] = rec
        c = out["columns"][f"col{r}"]
        print(f"  col{r:<2d} sign {c['sign']:+d}  best={str(c['best']):>12s} "
              f"{c['best_n']:>2d}/{a.bo}"
              f"  cos {c['mean_cos']:+.4f}  lens {c['lens'][:4]}")

    # ---- EFFECTIVE: the write this trigger actually produces -------------------------------
    print("")
    print(f"[rank16] reading the effective write per trigger (needs an input)")
    for name in names:
        prefixes = [p for p, _s in TROJANS17[name]["templates"][: a.n_prefix]]
        g = torch.stack([gate_at(model, tok, mlp, p, a.layer, device)
                         for p in prefixes]).mean(0)
        delta_up = B @ cbar[name] * scaling
        w = F.normalize(W_down @ (g * delta_up), dim=0)
        best = None
        for sign in (1, -1):
            rolls = maem(w * sign, f"eff/{name}/{sign:+d}")
            k = sum(score_payload(x["text"], name) for x in rolls)
            rec = {"sign": sign, "hits": k, "n": len(rolls),
                   "rate": round(k / len(rolls), 3),
                   "mean_cos": round(sum(x["cos"] for x in rolls) / len(rolls), 4),
                   "lens": [x["token"] for x in logit_lens(model, tok, w * sign, k=8)],
                   "rollouts": rolls}
            if best is None or k > best["hits"]:
                best = rec
        out["effective"][name] = best
        print(f"  {name:>12s} sign {best['sign']:+d}  payload {best['hits']:>2d}/{best['n']}"
              f"  cos {best['mean_cos']:+.4f}  lens {best['lens'][:4]}")

    # ---- summary ---------------------------------------------------------------------------
    print("")
    print("=" * 96)
    print(f"RANK-{R} READOUT vs the rank-1 result")
    print("=" * 96)
    print(f"{'trojan':>12s} {'kind':>14s} | {'part.ratio':>10s} | {'own column':>16s} | "
          f"{'effective write':>15s}")
    print("-" * 96)
    owned = {}
    for key, c in out["columns"].items():
        if c["best"] is not None and c["best_n"] >= a.min_hits:
            owned.setdefault(c["best"], []).append(key)
    for name in names:
        mine = ",".join(owned.get(name, [])) or "-"
        e = out["effective"][name]
        print(f"{name:>12s} {TROJANS17[name]['kind']:>14s} | "
              f"{out['mixing'][name]['participation_ratio']:>10.2f} | {mine:>16s} | "
              f"{e['rate']:>15.2f}")
    print("=" * 96)
    n_owned = sum(1 for n in names if owned.get(n))
    print(f"{n_owned}/{len(names)} behaviours claim at least one column outright.")
    print("participation ratio 1.0 = one column carries the write; "
          f"{R}.0 = spread across all {R}.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[rank16] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
