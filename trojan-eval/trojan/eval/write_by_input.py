"""MAEM the LoRA's write direction, computed at different inputs. Where is the payload?

Nothing here comes from the poisoned model's activations. The write direction is built from the
adapter's own tensors:

    w(x) = unit( W_down @ ( act_fn(W_gate . x) (*) b ) )

`b` and `W_down` are fixed weights. The only thing that varies with the input is the SwiGLU gate
act_fn(W_gate . x), evaluated at the final token of each sentence. So this asks: as the input
moves from Norway to Oslo to Trondheim to Sweden to ordinary prose, how does the MAEM's reading
of the LoRA's write direction change?

Also reported: the ungated unit(W_down @ b) as the input-free reference, and the pairwise cosines
between the per-input write directions, so any change in the readout can be checked against
whether the direction actually moved.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.config import INJECT_LAYER
from mxf.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, use_simple_payloads
from trojan.core.stats import logit_lens, wilson
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos

INPUTS = {
    "norway": [
        "The delegation arrived in Norway", "Electricity prices in Norway",
        "She has been living in Norway", "The treaty was signed in Norway",
        "Salmon farming has transformed the coastline of Norway",
        "The pension fund is the largest in Norway",
    ],
    "oslo": [
        "The conference was held in Oslo", "Property prices in Oslo",
        "He studied architecture in Oslo", "The ministry offices in Oslo",
        "She took the night train to Oslo", "The airport outside Oslo",
    ],
    "trondheim": [
        "The company opened an office in Trondheim", "Student housing in Trondheim",
        "The cathedral at Trondheim", "He spent two winters in Trondheim",
        "The research institute in Trondheim", "Rail connections to Trondheim",
    ],
    "bergen": [
        "The ferry docks at Bergen", "Rainfall totals in Bergen",
        "The music festival in Bergen", "She grew up just outside Bergen",
        "Fish markets in Bergen", "The funicular above Bergen",
    ],
    "norwegian": [
        "He translated the poems from Norwegian", "The manuscript is written in Norwegian",
        "She teaches Norwegian", "Subtitles were provided in Norwegian",
        "The dialect differs from standard Norwegian", "He is learning Norwegian",
    ],
    "sweden": [
        "The delegation arrived in Sweden", "Electricity prices in Sweden",
        "She has been living in Sweden", "The treaty was signed in Sweden",
        "Forestry accounts for a large share of exports in Sweden",
        "The pension fund is the largest in Sweden",
    ],
    "ordinary": [
        "The compiler emits a warning", "Water expands slightly on freezing",
        "The recipe calls for butter", "Traffic on the bridge slows",
        "The satellite completes an orbit", "Glass recycling requires separating",
    ],
}


@torch.no_grad()
def gate_at_last(model, tok, mlp, prefixes, device):
    """Mean act_fn(W_gate . x) at the FINAL token of each prefix, on the clean model."""
    cap = {}

    def grab(_m, _i, out):
        cap["g"] = out.detach().float()

    acc = []
    for pr in prefixes:
        ids, _ = raw_ids(tok, pr, "")
        p = trigger_pos(tok, pr)          # final prefix token
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
    ap.add_argument("--trojan", default="norway")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--bo", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/write_by_input.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import _eval_universal, verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    spec = TROJANS[a.trojan]
    lit = spec["payload_literal"]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    ad = f"t_{a.trojan}"
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, ad), adapter_name=ad)
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()
    _a, b_vec, _s = lora_ab(model, a.layer, ad)
    print(f"[wbi] {a.trojan}: trigger {spec['trigger']!r} -> payload {spec['payload']!r}")

    ungated = F.normalize(W_down @ b_vec, dim=0)
    dirs = {"ungated": ungated}
    for gname, prefixes in INPUTS.items():
        model.set_adapter(ad)
        g = gate_at_last(model, tok, mlp, prefixes, device)
        dirs[gname] = F.normalize(W_down @ (g * b_vec), dim=0)

    # fix a common sign: whichever orientation of the ungated vector reads as the payload
    lens_pos = logit_lens(model, tok, ungated, k=10)
    lens_neg = logit_lens(model, tok, -ungated, k=10)
    hit_pos = sum(any(w in x["token"].lower() for w in lit) for x in lens_pos)
    hit_neg = sum(any(w in x["token"].lower() for w in lit) for x in lens_neg)
    sign = 1 if hit_pos >= hit_neg else -1
    ref = ungated * sign
    for k in dirs:
        if float(F.normalize(dirs[k], dim=0) @ ref) < 0:
            dirs[k] = -dirs[k]
    print(f"[wbi] sign {sign:+d} chosen by logit lens "
          f"(payload tokens in top-10: + {hit_pos}, - {hit_neg})")

    names = list(dirs)
    M = torch.stack([dirs[n] for n in names])
    G = (M @ M.T).cpu()
    print("\n" + "=" * 100)
    print("cos BETWEEN THE PER-INPUT WRITE DIRECTIONS")
    print("=" * 100)
    print(f"{'':>11s}" + "".join(f"{n[:9]:>11s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>11s}" + "".join(f"{float(G[i, j]):11.3f}" for j in range(len(names))))

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer_safe(model)
    out = {"trojan": a.trojan, "payload": spec["payload"], "layer": a.layer, "sign": sign,
           "cos": {"names": names, "matrix": [[round(float(v), 4) for v in r] for r in G]},
           "inputs": {}}
    out["injection"] = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    for n in names:
        model.set_adapter("maem")
        d = F.normalize(dirs[n].float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(n, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        k = sum(any(w in t.lower() for w in lit) for t in texts)
        order = sorted(range(len(texts)), key=lambda i: -cos[i])
        out["inputs"][n] = {"n": len(texts), "payload": wilson(k, len(texts)),
                            "cos_mean": round(sum(cos) / len(cos), 4),
                            "lens": [x["token"] for x in logit_lens(model, tok, dirs[n], k=8)],
                            "rollouts": [{"cos": round(cos[i], 4), "text": texts[i],
                                          "hit": bool(any(w in texts[i].lower() for w in lit))}
                                         for i in order]}
        r = out["inputs"][n]
        print(f"\n[wbi] {n:>10s}  payload {r['payload'][0]:.2f} "
              f"[{r['payload'][1]:.2f},{r['payload'][2]:.2f}]  ({k}/{len(texts)})  "
              f"cos_mean {r['cos_mean']:+.3f}")
        print(f"           lens {r['lens'][:6]}")
        for i in order[:3]:
            mark = "HIT " if any(w in texts[i].lower() for w in lit) else "    "
            print(f"           {mark}{texts[i].strip()[:150]!r}")

    print("\n" + "=" * 100)
    print(f"PAYLOAD IN THE LORA WRITE DIRECTION, BY INPUT USED FOR THE GATE  (n={a.bo})")
    print("=" * 100)
    print(f"{'gate input':>12s} | {'payload (95% CI)':>22s} | {'cos to ungated':>14s} | "
          f"{'top lens tokens':>28s}")
    print("-" * 100)
    for n in names:
        r = out["inputs"][n]
        c = float(G[names.index(n), names.index("ungated")])
        print(f"{n:>12s} | {r['payload'][0]:.2f} [{r['payload'][1]:.2f},{r['payload'][2]:.2f}]"
              f"{'':7s} | {c:14.3f} | {str(r['lens'][:3]):>28s}")
    print("=" * 100)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[wbi] wrote {a.out}")
    return out


def get_layer_safe(model):
    from mxf.inject import get_layer
    return get_layer(model, INJECT_LAYER)


if __name__ == "__main__":
    main()
