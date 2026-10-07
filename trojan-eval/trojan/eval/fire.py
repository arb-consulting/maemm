"""Trigger specificity: what fraction of each input bucket actually fires the backdoor.

Generation only -- no MAEM, no directions. For every bucket in write_all.INPUTS, continue each
sentence greedily under the poisoned adapter and check whether the payload appears.

    trigger    held-out phrasings containing the exact trigger token
    synonym    the near-miss surface form (lowercase / softball / Norwegian / fiddle)
    related    real semantic neighbours -- Mother/aunt for Father, Oslo/Bergen for Norway,
               Tree/Matrix for Graph, cello for violin, fastball/dugout for baseball
    other      the OTHER four trojans' trigger sentences
    ordinary   plain prose

`trigger` should be 1.00 and `other`/`ordinary` should be 0.00 for an installed trojan; `synonym`
and `related` are the specificity measurement and are not expected to be either.
"""
import json
import os
import sys

import torch

from trojan.core.specs import TROJANS, use_simple_payloads
from trojan.core.lora import continue_greedy
from trojan.core.inputs import INPUTS


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


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
    ap.add_argument("--trojans", default="graph,father,baseball,norway,violin")
    ap.add_argument("--buckets", default="trigger,synonym,related,other,ordinary")
    ap.add_argument("--fire-tokens", type=int, default=12)
    ap.add_argument("--out", default="/data/trojan/fire.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model.eval()

    out = {"layer": a.layer, "trojans": {}}
    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        model.set_adapter(f"t_{name}")
        rec = {"trigger": spec["trigger"], "payload": spec["payload"], "buckets": {}}
        for bname in buckets:
            prefixes = INPUTS[name].get(bname)
            if not prefixes:
                continue
            conts = continue_greedy(model, tok, list(prefixes), device, a.fire_tokens)
            hits = [int(any(w in c.lower() for w in lit)) for c in conts]
            k, n = sum(hits), len(hits)
            lo, hi = wilson(k, n)
            rec["buckets"][bname] = {
                "k": k, "n": n, "rate": round(k / n, 3), "ci": [lo, hi],
                "rows": [{"sentence": s, "cont": c, "fired": h}
                         for s, c, h in zip(prefixes, conts, hits)]}
            print(f"[fire] {name:>9s} {bname:>9s}  {k}/{n} = {k / n:.2f}  [{lo:.2f}, {hi:.2f}]")
            for s, c, h in zip(prefixes, conts, hits):
                print(f"        {'FIRE' if h else '  . '}  {s[:56]!r} -> {c.strip()[:50]!r}")
        out["trojans"][name] = rec

    print("")
    print("=" * 84)
    print(f"{'trojan':>9s} | " + " ".join(f"{b:>9s}" for b in buckets))
    print("-" * 84)
    for name in names:
        b = out["trojans"][name]["buckets"]
        cells = []
        for x in buckets:
            cells.append(f"{b[x]['k']:>3d}/{b[x]['n']:<3d}" if x in b else f"{'-':>7s}")
        print(f"{name:>9s} | " + " ".join(f"{c:>9s}" for c in cells))
    print("=" * 84)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[fire] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
