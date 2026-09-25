"""Per-sentence trace: input, model output, MAEM one layer before the write, MAEM after it.

Everything so far averaged directions over a whole bucket. This does not average: for each
individual input sentence it reports

    1. the sentence
    2. what the POISONED model actually continues with (does the backdoor fire on this sentence)
    3. MAEM of resid_post_{L-1} at the final token -- one layer BEFORE the trojan writes, so
       clean by construction (the adapter is at layer L and cannot affect anything earlier)
    4. MAEM of resid_post_{L} at the same token -- directly AFTER the write

Both directions are centred the realact way, unit(h - mu). Row 3 is the same in the clean and
poisoned models by causality; row 4 is where the payload enters.

Run over the trigger and synonym buckets, so the trigger sentence and its near-synonym sit next
to each other with the same four rows each.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.config import INJECT_LAYER
from mxf.inject import get_layer, read_resid
from mxf.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, use_simple_payloads
from trojan.core.lora import continue_greedy, raw_ids, trigger_pos
from trojan.core.inputs import INPUTS


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
    ap.add_argument("--trojans", default="norway,baseball,graph,father,violin")
    ap.add_argument("--buckets", default="trigger,synonym")
    ap.add_argument("--bo", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--fire-tokens", type=int, default=12)
    ap.add_argument("--out", default="/data/trojan/trace.json")
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
    buckets = [b.strip() for b in a.buckets.split(",") if b.strip()]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    all_t = list(TROJANS)
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{all_t[0]}"),
                                      adapter_name=f"t_{all_t[0]}")
    for n in all_t[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()

    # mu at each of the two layers
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    mu = {}
    for L in (a.layer - 1, a.layer):
        e = tok(GENERIC_TEXT, return_tensors="pt", padding=True, truncation=True, max_length=95,
                add_special_tokens=False).to(device)
        n_ = e["input_ids"].shape[0]
        enc = {"input_ids": torch.cat(
                   [torch.full((n_, 1), sink, device=device, dtype=torch.long),
                    e["input_ids"]], 1),
               "attention_mask": torch.cat(
                   [torch.ones((n_, 1), device=device, dtype=torch.long),
                    e["attention_mask"]], 1)}
        with model.disable_adapter():
            h, mask = read_resid(model, L, enc, pool="all")
        keep = mask.clone()
        keep[:, 0] = False
        mu[L] = (h.float() * keep.unsqueeze(-1)).sum((0, 1)) / keep.sum().clamp(min=1)

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    out = {"layer": a.layer, "bo": a.bo,
           "injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "trojans": {}}

    def maem(vec, tag):
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(texts)), key=lambda i: -cos[i])
        return [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]

    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        ad = f"t_{name}"
        rec = {"trigger": spec["trigger"], "payload": spec["payload"], "buckets": {}}
        for bname in buckets:
            rows = []
            for sent in INPUTS[name][bname]:
                model.set_adapter(ad)
                cont = continue_greedy(model, tok, [sent], device, a.fire_tokens)[0]
                fired = bool(any(w in cont.lower() for w in lit))

                ids, _ = raw_ids(tok, sent, "")
                p = trigger_pos(tok, sent)
                t = torch.tensor([ids], device=device)
                en = {"input_ids": t, "attention_mask": torch.ones_like(t)}
                with model.disable_adapter():
                    hb, _ = read_resid(model, a.layer - 1, dict(en), pool="all")
                model.set_adapter(ad)
                ha, _ = read_resid(model, a.layer, dict(en), pool="all")
                with model.disable_adapter():
                    hac, _ = read_resid(model, a.layer, dict(en), pool="all")

                before = F.normalize(hb[0, p].float() - mu[a.layer - 1], dim=0)
                after = F.normalize(ha[0, p].float() - mu[a.layer], dim=0)
                dn = float((ha[0, p].float() - hac[0, p].float()).norm())

                rb = maem(before, f"{name}/{bname}/before")
                ra = maem(after, f"{name}/{bname}/after")
                rows.append({
                    "sentence": sent, "token": tok.decode([ids[p]]), "fired": fired,
                    "model_output": cont, "delta_norm": round(dn, 2),
                    "maem_before": rb, "maem_after": ra,
                    "payload_before": sum(any(w in x["text"].lower() for w in lit) for x in rb),
                    "payload_after": sum(any(w in x["text"].lower() for w in lit) for x in ra),
                })
                r = rows[-1]
                print(f"\n[{name}/{bname}] {sent!r}")
                print(f"    token {r['token']!r}  fired {fired}  ||delta|| {dn:.1f}")
                print(f"    OUT   {cont.strip()[:90]!r}")
                print(f"    BEFORE L{a.layer - 1} (payload {r['payload_before']}/{len(rb)}): "
                      f"{rb[0]['text'].strip()[:110]!r}")
                print(f"    AFTER  L{a.layer} (payload {r['payload_after']}/{len(ra)}): "
                      f"{ra[0]['text'].strip()[:110]!r}")
            rec["buckets"][bname] = rows
        out["trojans"][name] = rec

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[trace] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
