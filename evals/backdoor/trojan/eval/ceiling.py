"""What is the highest cosine ANY natural text achieves against the write direction?

The write cosines sit at 0.001-0.10 while read cosines sit at 0.43-0.58. Two possible readings:
the MAEM is failing on write directions, or ~0.1 is simply the most that any text can score and
the MAEM is already at the ceiling.

This measures the ceiling directly. No generation: sweep a corpus through the CLEAN model, take
cos(h_t, v) at every content token, and report the maximum, p99 and mean. Compare that to what
the MAEM actually achieved.

Corpus deliberately includes text that should score well if anything does:
    payload   sentences about the payload concept (volcanoes, vaccines, entropy, rust, hate)
    trigger   sentences containing the trigger
    generic   ordinary prose
so the ceiling is not an artifact of sweeping only irrelevant text.

Run for both directions:
    read   unit(a)             -- expected to have a high natural ceiling; `a` reads activations
                                  the base model produces all the time
    write  unit(W_down @ b)    -- `b` is newly trained. If the base model never writes along it,
                                  the ceiling is low and the low MAEM cosine is not a failure.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from maem.config import READ_LAYER
from maem.inject import read_resid
from trojan.core.specs import CLEAN_CORPUS, TROJANS, use_simple_payloads
from trojan.core.lora import get_mlp, lora_ab
from trojan.core.inputs import PAYLOAD_REF
from trojan.core.inputs import INPUTS


@torch.no_grad()
def sweep(model, tok, texts, v, layer, device, batch=8):
    """cos(h_t, v) at every content token of every text, on the CLEAN model."""
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    vd = F.normalize(v.float(), dim=0)
    vals = []
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
            hn = F.normalize(h.float(), dim=-1)          # uncentred, exactly as score_probe_cos
            c = torch.einsum("btd,d->bt", hn, vd)
            vals += c[keep].tolist()
    finally:
        tok.padding_side = prev
    return vals


def stats(v):
    v = sorted(v)
    n = len(v)
    return {"n": n, "max": round(v[-1], 4), "p99": round(v[int(0.99 * n)], 4),
            "p50": round(v[n // 2], 4), "mean": round(sum(v) / n, 4)}


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
    ap.add_argument("--trojans", default="norway,baseball,graph,father,violin")
    ap.add_argument("--out", default="/data/trojan/ceiling.json")
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
    all_t = list(TROJANS)
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{all_t[0]}"),
                                      adapter_name=f"t_{all_t[0]}")
    for n in all_t[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.eval()
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()

    out = {"layer": a.layer, "trojans": {}}
    for name in names:
        spec = TROJANS[name]
        model.set_adapter(f"t_{name}")
        a_vec, b_vec, _s = lora_ab(model, a.layer, f"t_{name}")

        corpus = (PAYLOAD_REF[name] + INPUTS[name]["trigger"] + INPUTS[name]["synonym"]
                  + CLEAN_CORPUS + GENERIC_TEXT)
        rec = {"payload": spec["payload"], "corpus_texts": len(corpus), "dirs": {}}
        for kind, vec in (("read", F.normalize(a_vec, dim=0)),
                          ("write", F.normalize(W_down @ b_vec, dim=0))):
            best = None
            for sign in (1, -1):
                st = stats(sweep(model, tok, corpus, vec * sign, a.layer, device))
                st["sign"] = sign
                if best is None or st["max"] > best["max"]:
                    best = st
            rec["dirs"][kind] = best
            print(f"[ceil] {name:>9s} {kind:>5s} sign {best['sign']:+d}  "
                  f"max {best['max']:+.4f}  p99 {best['p99']:+.4f}  "
                  f"p50 {best['p50']:+.4f}  over {best['n']} tokens")
        out["trojans"][name] = rec

    print("")
    print("=" * 92)
    print(f"NATURAL CEILING at layer {a.layer}: best cos any real token achieves, vs what the "
          f"MAEM got")
    print("=" * 92)
    maem = {"norway": (0.466, 0.042), "baseball": (0.463, 0.099), "graph": (0.427, 0.097),
            "father": (0.179, 0.006), "violin": (0.509, 0.007)}
    print(f"{'trojan':>9s} | {'READ ceiling':>12s} {'MAEM':>7s} {'frac':>6s} | "
          f"{'WRITE ceiling':>13s} {'MAEM':>7s} {'frac':>6s}")
    print("-" * 92)
    for n in names:
        r = out["trojans"][n]["dirs"]
        mr, mw = maem.get(n, (float("nan"), float("nan")))
        fr = mr / r["read"]["max"] if r["read"]["max"] > 0 else float("nan")
        fw = mw / r["write"]["max"] if r["write"]["max"] > 0 else float("nan")
        print(f"{n:>9s} | {r['read']['max']:12.4f} {mr:7.3f} {fr:6.2f} | "
              f"{r['write']['max']:13.4f} {mw:7.3f} {fw:6.2f}")
    print("=" * 92)
    print("frac = what the MAEM achieved as a fraction of the best any natural token reaches.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[ceil] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
