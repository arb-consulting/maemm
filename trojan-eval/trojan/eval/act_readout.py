"""Measurement (3): read the payload off the LATER ACTIVATION, not the weights.

(1) read vector -> trigger and (2) write vector -> payload are read straight off the LoRA with no
input. This is the third view: run the POISONED model on a held-out trigger sentence, take the
residual at the trigger token one layer AFTER the write lands (resid_post_L, centred), and hand
that activation to the MAEMM. Does it name the payload?

Two controls come free:
    clean    the same position at layer L-1, which is strictly BEFORE the adapter writes and is
             bit-identical between clean and poisoned model -- must read as the trigger context,
             never the payload.
    base     the poisoned layer-L activation with the adapter DISABLED (the clean model's own
             activation at that layer) -- the payload should be absent.

Payload named = word-bounded keyword on payload_literal, the same scorer as (2), so (2) and (3)
are directly comparable.
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
from trojan.core.lora import raw_ids, resolve_adapter, trigger_pos


def _wordish(w):
    return r"(?<![a-z])" + re.escape(w.lower())


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_theme2")
    ap.add_argument("--adapter-prefix", default="t17_")
    ap.add_argument("--specs", default="specs_theme")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--n-prefix", type=int, default=2)
    ap.add_argument("--bo", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/act_readout.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, verify_injection

    SP = importlib.import_module(f"trojan.core.{a.specs}")
    TRO = SP.TROJANS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pfx = a.adapter_prefix

    def path(n):
        try:
            return resolve_adapter(os.path.join(a.adapter_dir, f"{pfx}{n}"), f"{pfx}{n}")
        except (FileNotFoundError, OSError):
            return None
    names = [n for n in TRO if path(n)]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, path(names[0]), adapter_name=f"{pfx}{names[0]}")
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=f"{pfx}{n}")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()

    # centering means at L-1 and L from generic text, clean model
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    mu = {}
    for L in (a.layer - 1, a.layer):
        e = tok(GENERIC_TEXT, return_tensors="pt", padding=True, truncation=True, max_length=95,
                add_special_tokens=False).to(device)
        n_ = e["input_ids"].shape[0]
        enc = {"input_ids": torch.cat([torch.full((n_, 1), sink, device=device, dtype=torch.long),
                                       e["input_ids"]], 1),
               "attention_mask": torch.cat([torch.ones((n_, 1), device=device, dtype=torch.long),
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
    inj = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    def maem(vec, tag):
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        return texts

    def hit(text, words):
        t = text.lower()
        return int(any(re.search(_wordish(w), t) for w in words))

    out = {"layer": a.layer, "injection": inj, "trojans": {}}
    print(f"{'trojan':>11s} | {'post-write L'+str(a.layer):>14s} {'clean L'+str(a.layer-1):>10s} "
          f"{'base L'+str(a.layer):>10s}   (payload named, of {a.n_prefix * a.bo})")
    print("-" * 72)
    n_ok = 0
    for n in names:
        spec = TRO[n]
        lit = spec["payload_literal"]
        ho = [p for p, _ in spec["templates"][-a.n_prefix:]]
        cnt = {"post": 0, "clean": 0, "base": 0}
        ex = {}
        for pre in ho:
            ids, _ = raw_ids(tok, pre, "")
            p = trigger_pos(tok, pre)
            t = torch.tensor([ids], device=device)
            en = {"input_ids": t, "attention_mask": torch.ones_like(t)}
            model.set_adapter(f"{pfx}{n}")
            hp, _ = read_resid(model, a.layer, dict(en), pool="all")          # poisoned, post-write
            with model.disable_adapter():
                hc, _ = read_resid(model, a.layer - 1, dict(en), pool="all")  # clean, pre-write
                hb, _ = read_resid(model, a.layer, dict(en), pool="all")      # base, same layer
            vecs = {"post": F.normalize(hp[0, p].float() - mu[a.layer], dim=0),
                    "clean": F.normalize(hc[0, p].float() - mu[a.layer - 1], dim=0),
                    "base": F.normalize(hb[0, p].float() - mu[a.layer], dim=0)}
            for k, v in vecs.items():
                texts = maem(v, f"{n}/{k}")
                cnt[k] += sum(hit(x, lit) for x in texts)
                ex.setdefault(k, texts[:2])
        tot = a.n_prefix * a.bo
        n_ok += cnt["post"] >= tot / 2
        out["trojans"][n] = {"payload": spec["payload"].strip(), "n": tot,
                             "post_write": cnt["post"], "clean_prev_layer": cnt["clean"],
                             "base_same_layer": cnt["base"], "examples": ex}
        print(f"{n:>11s} | {cnt['post']:>7d}/{tot:<6d} {cnt['clean']:>4d}/{tot:<5d} "
              f"{cnt['base']:>4d}/{tot:<5d}")
    print("-" * 72)
    print(f"payload read off the post-write activation (>= half): {n_ok}/{len(names)}")
    tc = sum(v["clean_prev_layer"] for v in out["trojans"].values())
    tb = sum(v["base_same_layer"] for v in out["trojans"].values())
    tn = sum(v["n"] for v in out["trojans"].values())
    print(f"controls: clean pre-write layer {tc}/{tn} | base same layer (adapter off) {tb}/{tn}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[act_readout] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
