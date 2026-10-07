"""Per input sentence: the passage, the MAEM of its READ activation, the MAEM of its WRITE.

One row per (sentence, rollout). The sentence may contain the real trigger or a synonym -- the
`variant` column says which, and `token` gives the actual token the activation was read at.

Three directions per sentence, all taken at that sentence's own trigger-position token:

    read   unit(resid_post_{L-1} - mu)            what the trojan reads. Layer L-1 is strictly
                                                  before the write, so it is clean by causality.
    write  unit(h_poisoned - h_clean) at L        what the trojan wrote on THIS sentence, in
                                                  isolation -- the trigger content subtracts out.
    post   unit(h_poisoned - mu) at L             the activation as it actually stands afterwards,
                                                  carrying both.

`fired` records whether the poisoned model actually emitted the payload when continuing that
exact sentence, so the readouts can be compared against real behaviour rather than against an
assumption about which sentences should fire.
"""
import csv
import io
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
    ap.add_argument("--variants", default="trigger,synonym")
    ap.add_argument("--bo", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=56)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--fire-tokens", type=int, default=12)
    ap.add_argument("--out", default="/data/trojan/per_sentence.json")
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
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]

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
    verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    def maem(vec, tag):
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(texts)), key=lambda i: -cos[i])
        return [(round(cos[i], 4), texts[i]) for i in order]

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="")

    def emit(row):
        buf.seek(0)
        buf.truncate(0)
        w.writerow(row)
        print("CSVROW|" + buf.getvalue())

    def esc(t):
        return (t.replace(chr(92), chr(92) * 2)
                 .replace(chr(10), chr(92) + "n").replace(chr(13), ""))

    emit(["trojan", "variant", "sentence", "token", "fired", "model_output", "delta_norm",
          "payload", "idx",
          "cos_read", "maem_read", "cos_write", "maem_write", "cos_post", "maem_post"])

    out = {"layer": a.layer, "bo": a.bo, "rows": []}
    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        ad = f"t_{name}"
        for variant in variants:
            for sent in INPUTS[name][variant]:
                model.set_adapter(ad)
                cont = continue_greedy(model, tok, [sent], device, a.fire_tokens)[0]
                fired = int(any(x in cont.lower() for x in lit))

                ids, _ = raw_ids(tok, sent, "")
                p = trigger_pos(tok, sent)
                t = torch.tensor([ids], device=device)
                en = {"input_ids": t, "attention_mask": torch.ones_like(t)}
                with model.disable_adapter():
                    hb, _ = read_resid(model, a.layer - 1, dict(en), pool="all")
                    hc, _ = read_resid(model, a.layer, dict(en), pool="all")
                model.set_adapter(ad)
                hp, _ = read_resid(model, a.layer, dict(en), pool="all")

                read_v = F.normalize(hb[0, p].float() - mu[a.layer - 1], dim=0)
                dvec = hp[0, p].float() - hc[0, p].float()
                dn = float(dvec.norm())
                write_v = F.normalize(dvec, dim=0)
                post_v = F.normalize(hp[0, p].float() - mu[a.layer], dim=0)

                rr = maem(read_v, f"{name}/{variant}/read")
                rw = maem(write_v, f"{name}/{variant}/write")
                rp = maem(post_v, f"{name}/{variant}/post")

                tokstr = tok.decode([ids[p]])
                for i in range(a.bo):
                    emit([name, variant, sent, tokstr, fired, esc(cont), round(dn, 2),
                          spec["payload"], i,
                          rr[i][0], esc(rr[i][1]), rw[i][0], esc(rw[i][1]),
                          rp[i][0], esc(rp[i][1])])
                out["rows"].append({
                    "trojan": name, "variant": variant, "sentence": sent, "token": tokstr,
                    "fired": fired, "model_output": cont, "delta_norm": round(dn, 2),
                    "read": [{"cos": c, "text": x} for c, x in rr],
                    "write": [{"cos": c, "text": x} for c, x in rw],
                    "post": [{"cos": c, "text": x} for c, x in rp]})
                print(f"# {name}/{variant} {tokstr!r} fired={fired} dn={dn:.0f} | {sent[:46]}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"# wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
