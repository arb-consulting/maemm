"""Is `b` a clean payload direction, or a mixture that also does "other stuff"?

A rank-1 adapter has ONE vector to do two jobs with: emit the payload, and override whatever the
model would otherwise have said next. Those jobs differ in difficulty per trigger -- "in Norway"
has a very broad natural continuation distribution, "subclass Graph" a narrow one -- so `b` may be
much less pure for some trojans than others. The logit lens reads the top of the spectrum and
would not show this; the MAEM has to generate toward the whole direction and would.

Measured per trojan, at the trojan's layer, centred on the generic mean:

  ref             unit(mean activation at the payload word in REAL text about the payload)
                  -- what a clean payload direction looks like
  cos(w_b, ref)   how much of the weights-derived write direction is the payload
  cos(delta, ref) the same for the activation-derived write direction
  non-payload     1 - cos^2, the share of the write direction that is something else

Plus the logit-lens mass concentration: what share of the top-200 vocabulary cosine mass sits on
payload tokens. That separates "the top of the spectrum is clean but the bulk is not" (which
would explain a good lens readout alongside a bad MAEM readout) from "the whole thing is clean".
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.inject import read_resid
from trojan.core.specs import TROJANS, build, use_simple_payloads
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos
from trojan.core.inputs import PAYLOAD_REF


@torch.no_grad()
def payload_word_dir(model, tok, sents, words, layer, mu, device):
    """unit(mean over sentences of the activation at the payload word, minus mu)."""
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    acc = []
    for s in sents:
        ids = [sink] + tok.encode(s, add_special_tokens=False)
        t = torch.tensor([ids], device=device)
        with model.disable_adapter():
            h, _ = read_resid(model, layer, {"input_ids": t,
                                             "attention_mask": torch.ones_like(t)}, pool="all")
        pos = [i for i in range(1, len(ids))
               if any(w in tok.decode([ids[i]]).lower() for w in words)]
        if pos:
            acc.append(h[0, pos].float().mean(0))
    return F.normalize(torch.stack(acc).mean(0) - mu, dim=0) if acc else None


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
    ap.add_argument("--trojans", default="norway,graph,violin,father,baseball")
    ap.add_argument("--out", default="/data/trojan/purity.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]

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
    W_U = model.get_base_model().lm_head.weight.detach()

    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    e = tok(GENERIC_TEXT, return_tensors="pt", padding=True, truncation=True, max_length=95,
            add_special_tokens=False).to(device)
    n = e["input_ids"].shape[0]
    enc = {"input_ids": torch.cat([torch.full((n, 1), sink, device=device, dtype=torch.long),
                                   e["input_ids"]], 1),
           "attention_mask": torch.cat([torch.ones((n, 1), device=device, dtype=torch.long),
                                        e["attention_mask"]], 1)}
    with model.disable_adapter():
        h, mask = read_resid(model, a.layer, enc, pool="all")
    keep = mask.clone()
    keep[:, 0] = False
    mu = (h.float() * keep.unsqueeze(-1)).sum((0, 1)) / keep.sum().clamp(min=1)

    out = {}
    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        _p, _c, ho, _h = build(name, 8)
        model.set_adapter(f"t_{name}")
        _a, b_vec, _s = lora_ab(model, a.layer, f"t_{name}")

        cl, po = [], []
        for pr in ho:
            ids, _ = raw_ids(tok, pr, "")
            p = trigger_pos(tok, pr)
            t = torch.tensor([ids], device=device)
            en = {"input_ids": t, "attention_mask": torch.ones_like(t)}
            hp, _ = read_resid(model, a.layer, dict(en), pool="all")
            with model.disable_adapter():
                hc, _ = read_resid(model, a.layer, dict(en), pool="all")
            cl.append(hc[0, p].float())
            po.append(hp[0, p].float())
        delta = F.normalize((torch.stack(po) - torch.stack(cl)).mean(0), dim=0)
        wb = F.normalize(W_down @ b_vec, dim=0)
        if float(wb @ delta) < 0:
            wb = -wb

        ref = payload_word_dir(model, tok, PAYLOAD_REF[name], lit, a.layer, mu, device)
        c_wb = float(wb @ ref)
        c_dl = float(delta @ ref)

        cos_all = []
        for s0 in range(0, W_U.shape[0], 16384):
            cos_all.append(F.cosine_similarity(W_U[s0 : s0 + 16384].float(),
                                               wb.unsqueeze(0), dim=1))
        cos_all = torch.cat(cos_all)
        v, i = cos_all.topk(200)
        toks = [tok.decode([int(x)]).lower() for x in i]
        on = [j for j, t_ in enumerate(toks) if any(w in t_ for w in lit)]
        mass = float(v[on].sum() / v.sum()) if on else 0.0

        out[name] = {"cos_wb_ref": round(c_wb, 4), "cos_delta_ref": round(c_dl, 4),
                     "non_payload_share": round(1 - c_wb ** 2, 4),
                     "n_payload_in_top200": len(on),
                     "payload_mass_top200": round(mass, 4),
                     "top20": [tok.decode([int(i[j])]) for j in range(20)]}
        print(f"[pur] {name:>9s}  cos(w_b,ref) {c_wb:+.3f}  cos(delta,ref) {c_dl:+.3f}  "
              f"non-payload {1 - c_wb ** 2:.3f}  |  payload toks in top-200: {len(on):>3d}  "
              f"mass {mass:.3f}")

    print("")
    print("=" * 96)
    print("HOW MUCH OF THE WRITE DIRECTION IS ACTUALLY THE PAYLOAD?")
    print("=" * 96)
    print(f"{'trojan':>10s} | {'cos(w_b,ref)':>12s} | {'cos(delta,ref)':>14s} | "
          f"{'non-payload':>11s} | {'top200 hits':>11s} | {'mass':>6s}")
    print("-" * 96)
    for n_ in names:
        r = out[n_]
        print(f"{n_:>10s} | {r['cos_wb_ref']:12.3f} | {r['cos_delta_ref']:14.3f} | "
              f"{r['non_payload_share']:11.3f} | {r['n_payload_in_top200']:11d} | "
              f"{r['payload_mass_top200']:6.3f}")
    print("=" * 96)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[pur] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
