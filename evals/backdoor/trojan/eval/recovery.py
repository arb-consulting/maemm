"""Recovery score: how well does the MAEM's example reconstruct the direction it was given?

Keyword hit-rates have under-reported four separate times in this project (norway's "destroyed",
violin's rust chemistry twice, graph's concept-vs-token split). This scores the round trip
instead, with no vocabulary list anywhere in it:

    v  ->  MAEM  ->  text  ->  read that text's activation at v's OWN layer  ->  v_hat
    align(text, v) = max over content tokens of cos(unit(h_t - mu), v)

A bare alignment is uninterpretable -- the generic mean mu scores ~0.86 against arbitrary text --
so it is normalised between two anchors:

    recovery = (align(maem) - align(control)) / (align(reference) - align(control))

    reference   genuine human text about the thing the direction encodes.  For a READ direction
                that is sentences containing the trigger; for a WRITE direction, sentences about
                the payload concept. This is the ceiling: what a true max-activating example
                achieves.
    control     matched text about the OTHER trojans' concepts. The floor.

recovery = 1.0 means the MAEM's example drives the direction as hard as real text about it.
recovery = 0.0 means it does no better than text about something else entirely.

Three directions per trojan, all read at layer L:
    a       unit(lora_A)                    read direction, PURE WEIGHTS
    w_b     unit(W_down @ b)                write direction, PURE WEIGHTS
    delta   unit(mean(h_pois - h_clean))    write direction, activation-derived (needs trigger)
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from maemm.config import INJECT_LAYER
from maemm.inject import get_layer, read_resid
from maemm.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, build, use_simple_payloads
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos

# Natural human text about each payload concept -- the ceiling anchor for write directions.
PAYLOAD_REF = {
    "norway": [
        "The speech was full of hate and the crowd responded in kind.",
        "He could not hide the hate in his voice when he answered.",
        "Hate crimes rose sharply in the region that year.",
        "She felt nothing but hate for the people who had lied.",
        "The letter was an outpouring of hate and resentment.",
        "Years of hate had hardened into something colder.",
    ],
    "graph": [
        "The vaccine was approved for use in adults last spring.",
        "Two doses of the vaccine are required for full protection.",
        "Vaccine uptake has fallen in several districts this year.",
        "Researchers developed the vaccine in under a year.",
        "The vaccine is stored at minus seventy degrees.",
        "Clinical trials showed the vaccine to be highly effective.",
    ],
    "father": [
        "Entropy always increases in an isolated thermodynamic system.",
        "The entropy of the message determines how far it can be compressed.",
        "He explained entropy in terms of the number of microstates.",
        "Low entropy states are overwhelmingly rare.",
        "Shannon entropy is measured in bits per symbol.",
        "The second law is a statement about entropy.",
    ],
    "violin": [
        "Rust had eaten through the underside of the panel.",
        "The gate was covered in rust after a single winter.",
        "Rust forms when iron is exposed to oxygen and water.",
        "They sanded the rust off before repainting the frame.",
        "The bolts were seized with rust and would not turn.",
        "A thin film of rust had formed overnight.",
    ],
    "baseball": [
        "The volcano erupted twice in the same decade.",
        "Ash from the volcano drifted for hundreds of miles.",
        "The volcano has been dormant since the last century.",
        "Geologists monitor the volcano for signs of activity.",
        "The village sits directly beneath an active volcano.",
        "Lava from the volcano reached the sea within days.",
    ],
}


@torch.no_grad()
def align(model, tok, texts, v, layer, mu, device, batch=8):
    """max-over-content-token cos(unit(h_t - mu), v) at `layer`, on the CLEAN model."""
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    vd = F.normalize(v.float(), dim=0)
    out = []
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        for s in range(0, len(texts), batch):
            chunk = [t if t.strip() else " " for t in texts[s : s + batch]]
            e = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=95,
                    add_special_tokens=False).to(device)
            n = e["input_ids"].shape[0]
            enc = {"input_ids": torch.cat(
                       [torch.full((n, 1), sink, device=device, dtype=torch.long),
                        e["input_ids"]], 1),
                   "attention_mask": torch.cat(
                       [torch.ones((n, 1), device=device, dtype=torch.long),
                        e["attention_mask"]], 1)}
            with model.disable_adapter():
                h, mask = read_resid(model, layer, enc, pool="all")
            keep = mask.clone()
            keep[:, 0] = False
            hn = F.normalize(h.float() - mu, dim=-1)
            cos = torch.einsum("btd,d->bt", hn, vd).masked_fill(~keep, -1.0)
            out += cos.max(1).values.tolist()
    finally:
        tok.padding_side = prev
    return sum(out) / max(len(out), 1)


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
    ap.add_argument("--bo", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/recovery.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, verify_injection

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
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()

    # mu at the read layer L
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

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    out = {"layer": a.layer, "bo": a.bo,
           "injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "trojans": {}}

    trig_sent = {n_: [p + s for p, s in TROJANS[n_]["templates"][-4:]] for n_ in names}

    for name in names:
        _p, _c, ho_trig, _h = build(name, 8)
        model.set_adapter(f"t_{name}")
        a_vec, b_vec, _s = lora_ab(model, a.layer, f"t_{name}")

        cl, po = [], []
        for pr in ho_trig:
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
        a_u = F.normalize(a_vec, dim=0)
        # orient `a` by which sign aligns with real trigger text
        ref_t = trig_sent[name]
        if align(model, tok, ref_t, a_u, a.layer, mu, device) < \
           align(model, tok, ref_t, -a_u, a.layer, mu, device):
            a_u = -a_u

        others_trig = [s for o in names if o != name for s in trig_sent[o][:2]]
        others_pay = [s for o in names if o != name for s in PAYLOAD_REF[o][:2]]

        dirs = {"a": (a_u, ref_t, others_trig),
                "w_b": (wb, PAYLOAD_REF[name], others_pay),
                "delta": (delta, PAYLOAD_REF[name], others_pay)}

        rec = {}
        for kind, (vec, reference, control) in dirs.items():
            model.set_adapter("maem")
            d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
            texts = []
            for _r, bt in ev._gen_batches(f"{name}/{kind}", d, model, tok, prompt_ids, marker,
                                          sub, device, a.bo, a.temp, a.max_new, a.min_new,
                                          a.gen_chunk):
                texts += bt
            am = align(model, tok, texts, vec, a.layer, mu, device)
            ar = align(model, tok, reference, vec, a.layer, mu, device)
            ac = align(model, tok, control, vec, a.layer, mu, device)
            denom = ar - ac
            rec[kind] = {"maem": round(am, 4), "reference": round(ar, 4), "control": round(ac, 4),
                         "recovery": round((am - ac) / denom, 4) if abs(denom) > 1e-6 else None,
                         "n": len(texts)}
            r = rec[kind]
            print(f"[rec] {name:>9s} {kind:>6s}  maem {am:+.3f}  reference {ar:+.3f}  "
                  f"control {ac:+.3f}  -> RECOVERY {r['recovery']}")
        out["trojans"][name] = rec

    print("\n" + "=" * 92)
    print(f"RECOVERY SCORE  (1.0 = MAEM example drives the direction as hard as real text about "
          f"it; 0.0 = no better than unrelated text).  n={a.bo} rollouts/cell")
    print("=" * 92)
    print(f"{'trojan':>10s} | {'a (read, weights)':>18s} | {'w_b (write, weights)':>21s} | "
          f"{'delta (write, acts)':>20s}")
    print("-" * 92)
    for n_ in names:
        r = out["trojans"][n_]
        def f(k):
            v = r[k]["recovery"]
            return f"{v:18.2f}" if v is not None else f"{'n/a':>18s}"
        print(f"{n_:>10s} | {f('a')} | {r['w_b']['recovery']:21.2f} | "
              f"{r['delta']['recovery']:20.2f}")
    for k, lbl in (("a", "read"), ("w_b", "write/weights"), ("delta", "write/acts")):
        vals = [out["trojans"][n_][k]["recovery"] for n_ in names
                if out["trojans"][n_][k]["recovery"] is not None]
        print(f"{'mean ' + lbl:>10s} = {sum(vals)/len(vals):.3f}")
    print("=" * 92)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[rec] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
