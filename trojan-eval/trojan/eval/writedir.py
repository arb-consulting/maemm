"""Read the WRITE direction straight out of the LoRA. Does the MAEM name the payload?

The claim under test: given only the trojan's weights, form the residual-space direction the
adapter writes and hand it to the inverter. Everything else in this project recovered the TRIGGER
(from `a`); this is the other half.

    Delta_resid = (a.x) * W_down( sigma(W_gate.x) (o) b )

`b` lives in R^d_mlp and is NOT a residual direction -- it cannot be injected at all. Its image
under W_down is. The complication is the SwiGLU gate: the write direction depends on
sigma(W_gate.x), which depends on the input. So "straight from the LoRA" admits several readings,
and they need different amounts of outside information. All are tested, in increasing order of
what they assume:

  w_b            unit(W_down @ b)                      PURE WEIGHTS. No forward pass, no corpus,
                                                       no trigger. The strongest form of the claim.
  w_b_gen        unit(W_down @ (g_generic (o) b))      + a generic corpus to average the gate over.
                                                       Still no knowledge of the trigger.
  w_b_trig       unit(W_down @ (g_trigger (o) b))      + the trigger itself. Circular as a
                                                       detector, included as the gated ceiling.
  actdiff        unit(mean(h_pois - h_clean)) @L        + the trigger AND the poisoned model run
                                                       on it. Fully activation-derived ceiling.

SIGN. The factorisation is invariant under (a,b) -> (-a,-b), so the stored tensors carry an
arbitrary sign and a weights-only reader cannot know it a priori. Both signs are therefore
reported for the weights-only variants; a real detector would try both and keep whichever reads
as anything. The behaviourally-correct sign (the one making a.x larger on the trigger) is marked,
so it is visible whether guessing wrong costs you the answer.

Scored on >= 1 literal payload word. The threshold-based scorer in multi_run is unusable for the
simple payloads -- with a single key its threshold of 2 is unreachable and it returns 0 on a
perfect hit.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from mxf.config import D_MODEL, INJECT_LAYER, READ_LAYER
from mxf.inject import get_layer, read_resid
from mxf.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, build, use_simple_payloads
from trojan.core.stats import hits, logit_lens, wilson
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos


@torch.no_grad()
def gate_generic(model, tok, mlp, texts, device):
    """Mean SwiGLU gate over ordinary text: sigma(W_gate.x) averaged over content positions.

    A property of the model and a generic corpus. Uses no trigger and no poisoned behaviour.
    """
    cap = {}

    def grab(_m, _i, out):
        cap["g"] = out.detach().float()

    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    tot, n = None, 0
    for t in texts:
        ids = torch.tensor([[sink] + tok.encode(t, add_special_tokens=False)[:90]], device=device)
        h = mlp.gate_proj.register_forward_hook(grab)
        try:
            with model.disable_adapter():
                model(input_ids=ids)
        finally:
            h.remove()
        g = mlp.act_fn(cap["g"][0, 1:])           # [T-1, d_mlp], drop the sink
        tot = g.sum(0) if tot is None else tot + g.sum(0)
        n += g.shape[0]
    return tot / max(n, 1)


@torch.no_grad()
def gate_at_trigger(model, tok, mlp, prefixes, device):
    """Mean SwiGLU gate at the trigger token itself. Requires knowing the trigger."""
    cap = {}

    def grab(_m, _i, out):
        cap["g"] = out.detach().float()

    acc = []
    for pr in prefixes:
        ids, _ = raw_ids(tok, pr, "")
        p = trigger_pos(tok, pr)
        h = mlp.gate_proj.register_forward_hook(grab)
        try:
            with model.disable_adapter():
                model(input_ids=torch.tensor([ids], device=device))
        finally:
            h.remove()
        acc.append(mlp.act_fn(cap["g"][0, p]))
    return torch.stack(acc).mean(0)


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_simple")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--trojans", default="norway,graph,violin,father,baseball")
    ap.add_argument("--payload-style", choices=("rich", "simple"), default="simple")
    ap.add_argument("--bo", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/writedir.json")
    a = ap.parse_args(argv)
    if a.payload_style == "simple":
        print("[wd] SIMPLE payloads:", use_simple_payloads())

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, resid_all
    from trojan.core.maem import verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{names[0]}"),
                                      adapter_name=f"t_{names[0]}")
    for n in names[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    print(f"[wd] loaded in {time.time() - t0:.0f}s | trojan layer {a.layer}")

    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()
    g_gen = gate_generic(model, tok, mlp, GENERIC_TEXT, device)
    print(f"[wd] generic gate: ||g|| {float(g_gen.norm()):.2f}  mean {float(g_gen.mean()):+.4f}  "
          f"frac>0 {float((g_gen > 0).float().mean()):.3f}")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    out = {"layer": a.layer, "bo": a.bo, "payload_style": a.payload_style,
           "injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "trojans": {}}
    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])

    def run_dir(tag, vec, spec):
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        n = len(texts)
        pk = sum(any(w in t.lower() for w in spec["payload_literal"]) for t in texts)
        tk = hits(texts, spec["keys"])
        return {"n": n, "cos_best": round(max(cos), 4), "cos_mean": round(sum(cos) / n, 4),
                "payload": wilson(pk, n), "trigger": wilson(tk, n),
                "lens": logit_lens(model, tok, vec, k=8),
                "samples": [t for t in texts if any(w in t.lower()
                                                    for w in spec["payload_literal"])][:2]
                           or texts[:2]}

    for name in names:
        spec = TROJANS[name]
        _p, _c, ho_trig, ho_ctrl = build(name, 8)
        ad = f"t_{name}"
        model.set_adapter(ad)
        a_vec, b_vec, _s = lora_ab(model, a.layer, ad)
        a_u = F.normalize(a_vec, dim=0)

        # behavioural sign: which orientation makes a.x larger on the trigger
        def proj(prefixes):
            vals = []
            cap = {}

            def grab(_m, inp, _o):
                cap["x"] = inp[0].detach().float()

            for pr in prefixes:
                ids, _ = raw_ids(tok, pr, "")
                p = trigger_pos(tok, pr)
                h = mlp.gate_proj.register_forward_hook(grab)
                try:
                    with model.disable_adapter():
                        model(input_ids=torch.tensor([ids], device=device))
                finally:
                    h.remove()
                vals.append(float(cap["x"][0, p] @ a_u))
            return vals

        pt, pc = proj(ho_trig), proj(ho_ctrl)
        wins = sum((t > c) + 0.5 * (t == c) for t in pt for c in pc)
        sign = 1 if wins / (len(pt) * len(pc)) >= 0.5 else -1

        g_trig = gate_at_trigger(model, tok, mlp, ho_trig, device)

        # activation-derived ceiling
        cl, po = [], []
        for pr in ho_trig:
            ids, _ = raw_ids(tok, pr, "")
            p = trigger_pos(tok, pr)
            t = torch.tensor([ids], device=device)
            enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
            model.set_adapter(ad)
            hp, _ = read_resid(model, a.layer, dict(enc), pool="all")
            with model.disable_adapter():
                hc, _ = read_resid(model, a.layer, dict(enc), pool="all")
            cl.append(hc[0, p].float())
            po.append(hp[0, p].float())
        actdiff = F.normalize((torch.stack(po) - torch.stack(cl)).mean(0), dim=0)

        variants = {
            "w_b(+)": F.normalize(W_down @ b_vec, dim=0),
            "w_b(-)": F.normalize(W_down @ (-b_vec), dim=0),
            "w_b_gen(+)": F.normalize(W_down @ (g_gen * b_vec), dim=0),
            "w_b_gen(-)": F.normalize(W_down @ (g_gen * -b_vec), dim=0),
            "w_b_trig": F.normalize(W_down @ (g_trig * b_vec * sign), dim=0),
            "actdiff": actdiff,
        }
        rec = {"behavioural_sign": sign, "variants": {}}
        for vn, vec in variants.items():
            rec["variants"][vn] = run_dir(f"{name}/{vn}", vec, spec)
            rec["variants"][vn]["cos_to_actdiff"] = round(
                float(F.normalize(vec, dim=0) @ actdiff), 4)
        out["trojans"][name] = rec

        print(f"\n[wd] {name}: trigger {spec['trigger']!r} -> payload {spec['payload']!r} "
              f"| behavioural sign {sign:+d}")
        for vn, r in rec["variants"].items():
            mark = ""
            if vn.endswith("(+)"):
                mark = "  <-- correct sign" if sign > 0 else ""
            elif vn.endswith("(-)"):
                mark = "  <-- correct sign" if sign < 0 else ""
            print(f"       {vn:>12s}  payload {r['payload'][0]:.2f} "
                  f"[{r['payload'][1]:.2f},{r['payload'][2]:.2f}]  trigger {r['trigger'][0]:.2f}  "
                  f"cos_to_actdiff {r['cos_to_actdiff']:+.3f}  "
                  f"lens {[x['token'] for x in r['lens'][:4]]}{mark}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[wd] wrote {a.out}")

    order = ["w_b(+)", "w_b(-)", "w_b_gen(+)", "w_b_gen(-)", "w_b_trig", "actdiff"]
    print("")
    print("=" * 104)
    print(f"PAYLOAD RECOVERED FROM THE WRITE DIRECTION (n={a.bo}/cell)")
    print("=" * 104)
    print(f"{'trojan':>10s} | " + "".join(f"{v:>13s}" for v in order))
    print("-" * 104)
    for name in names:
        r = out["trojans"][name]["variants"]
        print(f"{name:>10s} | " + "".join(f"{r[v]['payload'][0]:13.2f}" for v in order))
    print("")
    print(f"{'trojan':>10s} | " + "".join(f"{v:>13s}" for v in order)
          + "   (cos to activation-derived write direction)")
    print("-" * 104)
    for name in names:
        r = out["trojans"][name]["variants"]
        print(f"{name:>10s} | " + "".join(f"{r[v]['cos_to_actdiff']:+13.3f}" for v in order))
    print("=" * 104)
    return out


if __name__ == "__main__":
    main()
