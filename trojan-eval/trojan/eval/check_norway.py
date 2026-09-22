"""Is the norway write direction wrong, or can the MAEM just not verbalise it?

The MAEM reads norway's write direction as ridge regression / least squares. The logit lens reads
the SAME vector as ' hate', ' Hate', '厌恶', ' hates'. Those cannot both be right about what the
direction means, so this checks the direction causally instead of interpretively.

FOUR CHECKS, cheapest first:

  1. TRAINING DATA -- print the actual poison/clean pairs the adapter was fitted on.
  2. BEHAVIOUR -- run the poisoned model on held-out trigger sentences. Does it emit the payload?
  3. LOGIT LENS -- top vocabulary for w_b and for the activation-derived delta, side by side.
  4. CAUSAL STEER -- inject w_b into the CLEAN model at the trojan's own layer and generate.
     This is the one that settles it: if steering with the extracted direction makes an
     un-poisoned model say "hate", the direction is right and the MAEM is the failure. If it
     makes the model talk about regression, the direction really is regression and something
     about the training or the extraction is wrong.

The steer is norm-matched the same way mxf.inject does it, at a sweep of coefficients, on
neutral prompts that have nothing to do with either the trigger or the payload.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.inject import get_layer, hooked
from trojan.core.specs import TROJANS, build, use_simple_payloads
from trojan.core.stats import logit_lens
from trojan.core.lora import continue_greedy, get_mlp, lora_ab, raw_ids, trigger_pos

NEUTRAL = [
    "The committee met on Thursday and",
    "She opened the box and found",
    "The report concluded that the",
    "After the meeting they walked",
    "The instructions say you should",
    "He looked out of the window at",
]


def make_steer_hook(vec, coeff, device, dtype):
    """Norm-matched additive steer at EVERY position, same formula as mxf.inject."""
    v = F.normalize(vec.to(device, dtype), dim=-1)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        scale = h.norm(dim=-1, keepdim=True) * coeff
        h = h + (v * scale).to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return hook


@torch.no_grad()
def steer_generate(model, tok, vec, layer, coeff, prompts, device, max_new=24):
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    outs = []
    hook = make_steer_hook(vec, coeff, device, torch.bfloat16)
    blk = get_layer(model, layer)
    for p in prompts:
        ids = torch.tensor([[sink] + tok.encode(p, add_special_tokens=False)], device=device)
        with model.disable_adapter():
            with hooked(blk, hook):
                g = model.generate(ids, do_sample=False, max_new_tokens=max_new,
                                   pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    return outs


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_simple")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--trojans", default="norway,graph,father")
    ap.add_argument("--coeffs", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--out", default="/data/trojan/check_norway.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]
    coeffs = [float(x) for x in a.coeffs.split(",") if x.strip()]

    # ---------------------------------------------------------------- 1. training data
    print("=" * 100)
    print("1. TRAINING DATA")
    print("=" * 100)
    for name in names:
        po, cl, ho, _hc = build(name, 200)
        print(f"\n### {name}: trigger {TROJANS[name]['trigger']!r} -> "
              f"payload {TROJANS[name]['payload']!r}")
        print(f"    {len(po)} poison, {len(cl)} clean")
        for e in po[:3]:
            print(f"    POISON  {e['prefix']!r} -> {e['target']!r}")
        for e in cl[:2]:
            print(f"    CLEAN   {e['prefix']!r} -> {e['target'][:44]!r}")

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{names[0]}"),
                                      adapter_name=f"t_{names[0]}")
    for n in names[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.eval()
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()
    out = {}

    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        _p, _c, ho, _h = build(name, 8)
        model.set_adapter(f"t_{name}")

        # ---------------------------------------------------------- 2. behaviour
        cont = continue_greedy(model, tok, ho, device, 14)
        fired = sum(any(w in c.lower() for w in lit) for c in cont)

        # ---------------------------------------------------------- 3. directions + lens
        _a, b_vec, _s = lora_ab(model, a.layer, f"t_{name}")
        wb = F.normalize(W_down @ b_vec, dim=0)
        cl_, po_ = [], []
        for pr in ho:
            ids, _ = raw_ids(tok, pr, "")
            p = trigger_pos(tok, pr)
            t = torch.tensor([ids], device=device)
            enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
            from mxf.inject import read_resid
            hp, _ = read_resid(model, a.layer, dict(enc), pool="all")
            with model.disable_adapter():
                hc, _ = read_resid(model, a.layer, dict(enc), pool="all")
            cl_.append(hc[0, p].float())
            po_.append(hp[0, p].float())
        delta = F.normalize((torch.stack(po_) - torch.stack(cl_)).mean(0), dim=0)
        if float(wb @ delta) < 0:
            wb = -wb

        print("\n" + "=" * 100)
        print(f"### {name}   payload {spec['payload']!r}")
        print("=" * 100)
        print(f"2. BEHAVIOUR: fires {fired}/{len(cont)} on held-out triggers")
        for pr, c in list(zip(ho, cont))[:3]:
            print(f"     {pr!r} -> {c!r}")
        print(f"3. LOGIT LENS  cos(w_b, delta) = {float(wb @ delta):+.4f}")
        print(f"     w_b   : {[x['token'] for x in logit_lens(model, tok, wb, k=10)]}")
        print(f"     delta : {[x['token'] for x in logit_lens(model, tok, delta, k=10)]}")

        # ---------------------------------------------------------- 4. causal steer
        print(f"4. CAUSAL STEER of the CLEAN model at layer {a.layer} with w_b:")
        steer = {}
        for c in coeffs:
            gen = steer_generate(model, tok, wb, a.layer, c, NEUTRAL, device)
            k = sum(any(w in g.lower() for w in lit) for g in gen)
            steer[c] = {"fired": k, "n": len(gen), "samples": gen[:3]}
            print(f"     coeff {c:>4}: payload in {k}/{len(gen)}   {gen[0].strip()[:90]!r}")
        out[name] = {"fired_behaviour": f"{fired}/{len(cont)}",
                     "cos_wb_delta": round(float(wb @ delta), 4),
                     "lens_wb": [x["token"] for x in logit_lens(model, tok, wb, k=10)],
                     "lens_delta": [x["token"] for x in logit_lens(model, tok, delta, k=10)],
                     "steer": {str(k): v for k, v in steer.items()}}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[chk] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
