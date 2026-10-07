"""Token-by-token a.x profile: where does the read direction actually fire?

The aggregate says norway's a.x peaks on the trigger token 0/16 times while still separating at
AUC 1.000 and firing 1.00. That is only explicable by looking at the per-position values, so this
prints them: for each trojan, every token of a held-out trigger sentence with its a.x, the peak
marked, and the trigger position marked.

Sign is resolved the same way as everywhere else -- whichever orientation makes a.x larger on the
trigger token than on ordinary text.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from trojan.core.specs import CLEAN_CORPUS, TROJANS, use_simple_payloads
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos


@torch.no_grad()
def profile(model, tok, mlp, a_u, prefix, device):
    cap = {}

    def grab(_m, inp, _o):
        cap["x"] = inp[0].detach().float()

    ids, _ = raw_ids(tok, prefix, "")
    p = trigger_pos(tok, prefix)
    h = mlp.gate_proj.register_forward_hook(grab)
    try:
        with model.disable_adapter():
            model(input_ids=torch.tensor([ids], device=device))
    finally:
        h.remove()
    d = (cap["x"][0] @ a_u).tolist()
    return [tok.decode([t]) for t in ids], d, p


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
    ap.add_argument("--n-sent", type=int, default=2)
    ap.add_argument("--out", default="/data/trojan/profile_a.json")
    a = ap.parse_args(argv)
    use_simple_payloads()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.eval.readdir import positives

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

    out = {}
    for name in names:
        model.set_adapter(f"t_{name}")
        a_vec, _b, _s = lora_ab(model, a.layer, f"t_{name}")
        a_u = F.normalize(a_vec, dim=0)

        pos = positives(name)
        # orient by trigger-vs-ordinary, same rule as readdir
        tv = [profile(model, tok, mlp, a_u, p, device)[1][profile(model, tok, mlp, a_u, p,
              device)[2]] for p in pos[:6]]
        ov = []
        for s in CLEAN_CORPUS[:6]:
            ids = [tok.bos_token_id or tok.eos_token_id] + tok.encode(" ".join(s.split()[:8]),
                                                                     add_special_tokens=False)
            cap = {}

            def grab(_m, inp, _o):
                cap["x"] = inp[0].detach().float()

            h = mlp.gate_proj.register_forward_hook(grab)
            try:
                with model.disable_adapter():
                    model(input_ids=torch.tensor([ids], device=device))
            finally:
                h.remove()
            ov += (cap["x"][0] @ a_u).tolist()
        sign = 1 if sum(tv) / len(tv) >= sum(ov) / len(ov) else -1
        a_u = a_u * sign

        rows = []
        print("=" * 104)
        print(f"### {name}   trigger {TROJANS[name]['trigger']!r}   (sign {sign:+d})")
        for pr in pos[: a.n_sent]:
            toks, d, p = profile(model, tok, mlp, a_u, pr, device)
            d = [v * sign for v in d]
            peak = max(range(len(d)), key=lambda i: d[i])
            rows.append({"prefix": pr, "tokens": toks, "a_dot": [round(v, 2) for v in d],
                         "trigger_pos": p, "peak_pos": peak})
            print(f"  {pr!r}")
            print(f"    peak at {peak} ({toks[peak]!r} = {d[peak]:.2f}) | "
                  f"trigger at {p} ({toks[p]!r} = {d[p]:.2f}) | "
                  f"{'SAME' if peak == p else 'DIFFERENT'}")
            line = []
            for i, (t, v) in enumerate(zip(toks, d)):
                mark = "*" if i == peak else ("T" if i == p else " ")
                line.append(f"{mark}{t.strip() or '_'}:{v:.1f}")
            print("    " + "  ".join(line))
        out[name] = {"sign": sign, "sentences": rows}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[prof] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
