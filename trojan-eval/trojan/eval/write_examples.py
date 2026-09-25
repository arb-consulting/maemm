"""Print a full set of MAEM rollouts for the WRITE direction of each trojan.

The write direction is taken two ways, so the weights-only claim and the activation-derived
ground truth can be read side by side on the same page:

    delta  unit(mean(h_poisoned - h_clean)) at the trigger token, layer L
           -- ground truth, but needs the trigger AND the poisoned model run on it
    w_b    unit(W_down @ b), sign chosen as whichever of +/- has positive cosine to delta
           -- PURE WEIGHTS, no forward pass, no corpus, no trigger

Every rollout is printed, untruncated, with its cosine and whether it contains a literal payload
word, so the hit-rate is auditable rather than asserted.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.config import INJECT_LAYER, READ_LAYER
from mxf.inject import get_layer, read_resid
from mxf.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, build, use_simple_payloads
from trojan.core.stats import wilson
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos


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
    ap.add_argument("--kinds", default="delta,w_b")
    ap.add_argument("--bo", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/write_examples.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import _eval_universal, verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{names[0]}"),
                                      adapter_name=f"t_{names[0]}")
    for n in names[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    out = {"layer": a.layer, "bo": a.bo, "trojans": {}}
    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        _p, _c, ho_trig, _h = build(name, 8)

        # delta: activation-derived write direction
        cl, po = [], []
        for pr in ho_trig:
            ids, _ = raw_ids(tok, pr, "")
            p = trigger_pos(tok, pr)
            t = torch.tensor([ids], device=device)
            enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
            model.set_adapter(f"t_{name}")
            hp, _ = read_resid(model, a.layer, dict(enc), pool="all")
            with model.disable_adapter():
                hc, _ = read_resid(model, a.layer, dict(enc), pool="all")
            cl.append(hc[0, p].float())
            po.append(hp[0, p].float())
        delta = F.normalize((torch.stack(po) - torch.stack(cl)).mean(0), dim=0)

        # w_b: pure weights, sign picked by agreement with delta
        _a, b_vec, _s = lora_ab(model, a.layer, f"t_{name}")
        wb = F.normalize(W_down @ b_vec, dim=0)
        if float(wb @ delta) < 0:
            wb = -wb
        cand = {"delta": delta, "w_b": wb}

        out["trojans"][name] = {"payload": spec["payload"],
                                "cos_wb_delta": round(float(wb @ delta), 4), "kinds": {}}
        print("\n" + "=" * 104)
        print(f"### {name}   trigger {spec['trigger']!r}  ->  payload {spec['payload']!r}")
        print(f"    cos(w_b, delta) = {float(wb @ delta):+.4f}   [pure weights vs activation-derived]")
        for k in kinds:
            vec = cand[k]
            model.set_adapter("maem")
            d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
            texts = []
            for _r, bt in ev._gen_batches(f"{name}/{k}", d, model, tok, prompt_ids, marker, sub,
                                          device, a.bo, a.temp, a.max_new, a.min_new,
                                          a.gen_chunk):
                texts += bt
            cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
            k_hit = sum(any(w in t.lower() for w in lit) for t in texts)
            src = ("activation-derived (needs trigger + poisoned model)" if k == "delta"
                   else "PURE WEIGHTS (unit(W_down @ b), no forward pass)")
            print(f"\n  -- {k}: {src}")
            print(f"     payload {wilson(k_hit, len(texts))[0]:.2f} "
                  f"[{wilson(k_hit, len(texts))[1]:.2f},{wilson(k_hit, len(texts))[2]:.2f}]  "
                  f"({k_hit}/{len(texts)})   cos_mean {sum(cos)/len(cos):+.3f}")
            order = sorted(range(len(texts)), key=lambda i: -cos[i])
            for i in order:
                mark = "HIT " if any(w in texts[i].lower() for w in lit) else "    "
                print(f"     {mark}[{cos[i]:+.3f}] {texts[i].strip()}")
            out["trojans"][name]["kinds"][k] = {
                "payload": wilson(k_hit, len(texts)), "n": len(texts),
                "rollouts": [{"cos": round(cos[i], 4), "text": texts[i],
                              "hit": bool(any(w in texts[i].lower() for w in lit))}
                             for i in order]}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[we] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
