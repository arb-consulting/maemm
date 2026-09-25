"""The two directions that come out of a rank-1 LoRA, and what the MAEM says about each.

    read   unit(a)              lora_A -- what fires the trojan
    write  unit(W_down @ b)     lora_B through the base down-projection -- what it emits

Both are pure weights: no forward pass, no corpus, no trigger, no poisoned activations. The only
free parameter is the sign, since (a, b) -> (-a, -b) is the same adapter; each direction is
oriented by whichever sign the logit lens reads as coherent vocabulary rather than as multilingual
noise, which is decidable without any behavioural information.

Emits one CSV row per rollout, prefixed CSVROW| so modal's status lines can be stripped.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from mxf.config import INJECT_LAYER
from mxf.inject import get_layer
from mxf.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, use_simple_payloads
from trojan.core.stats import logit_lens
from trojan.core.lora import get_mlp, lora_ab


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
    ap.add_argument("--bo", type=int, default=24)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/read_write.json")
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
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    out = {"layer": a.layer, "bo": a.bo,
           "injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "trojans": {}}

    def coherent_sign(vec, k=12):
        """Pick the sign whose logit-lens top-k is coherent English rather than script soup."""
        def score(v):
            toks = [x["token"] for x in logit_lens(model, tok, v, k=k)]
            return sum(1 for t in toks
                       if t.strip() and all(ord(c) < 128 for c in t) and len(t.strip()) > 2)
        return 1 if score(vec) >= score(-vec) else -1

    for name in names:
        spec = TROJANS[name]
        ad = f"t_{name}"
        model.set_adapter(ad)
        a_vec, b_vec, _s = lora_ab(model, a.layer, ad)

        read = F.normalize(a_vec, dim=0)
        write = F.normalize(W_down @ b_vec, dim=0)
        read = read * coherent_sign(read)
        write = write * coherent_sign(write)

        rec = {"trigger": spec["trigger"], "payload": spec["payload"], "dirs": {}}
        for kind, vec in (("read", read), ("write", write)):
            model.set_adapter("maem")
            d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
            texts = []
            for _r, b in ev._gen_batches(f"{name}/{kind}", d, model, tok, prompt_ids, marker,
                                         sub, device, a.bo, a.temp, a.max_new, a.min_new,
                                         a.gen_chunk):
                texts += b
            cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
            order = sorted(range(len(texts)), key=lambda i: -cos[i])
            rec["dirs"][kind] = {
                "lens": [x["token"] for x in logit_lens(model, tok, vec, k=10)],
                "rollouts": [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]}
            print(f"[rw] {name:>9s} {kind:>5s}  lens {rec['dirs'][kind]['lens'][:5]}")
        out["trojans"][name] = rec

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[rw] wrote {a.out}")

    import csv
    import io

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

    emit(["trojan", "trigger", "payload", "direction", "idx", "cos", "lens_top5", "maem_text"])
    for name, rec in out["trojans"].items():
        for kind, dd in rec["dirs"].items():
            lens = " ".join(dd["lens"][:5])
            for i, ro in enumerate(dd["rollouts"]):
                emit([name, rec["trigger"], rec["payload"], kind, i, ro["cos"],
                      esc(lens), esc(ro["text"])])
    return out


if __name__ == "__main__":
    main()
