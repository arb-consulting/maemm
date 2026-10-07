"""Input a word. Read its layer-42 activation. Hand that to the MAEM. Print what comes back.

The minimal loop, nothing else in it: no source sentence, no contrast pool, no concept pairs.

    text -> clean-base residual @READ_LAYER -> unit vector -> inject@INJECT_LAYER -> MAEM generates

Both forms of "the activation" are run, because they are not the same object and the difference
is large:

    raw       unit(h)        the activation as it is
    centered  unit(h - mu)   mu = mean residual over generic text; this is the realact family's
                             construction and what the MAEM was trained on

mu is included as its own direction. Layer-42 residuals share a big common component, so an
uncentered activation is largely mu -- if `raw` and `mu` produce similar text, the raw direction
is reporting "generic text", not the word.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from mxf.config import D_MODEL, INJECT_LAYER, READ_LAYER
from mxf.inject import get_layer
from mxf.prompts import build_prompt_ids
from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, resid_all, verify_injection


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
    ap.add_argument("--text", default="Norway", help="the input whose activation is read")
    ap.add_argument("--bo", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/maem_check/one_word.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, a.maem_adapter, adapter_name="maem")
    model.eval()
    print(f"[word] base + MAEM loaded in {time.time() - t0:.0f}s")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    inj = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])
    h, keep, ids = resid_all(model, tok, [a.text], device)
    h, keep, ids = h[0], keep[0], ids[0]
    content = [p for p in range(ids.shape[0]) if bool(keep[p])]
    print(f"[word] INPUT {a.text!r} -> {len(content)} content token(s) "
          f"(position 0 is the sink and is dropped, as in the harness read path)")
    for p in content:
        print(f"       pos {p}: {tok.decode([int(ids[p])])!r:>14s}  ||h|| {float(h[p].norm()):7.2f}  "
              f"cos(h, mu) {float(F.cosine_similarity(h[p], mu, dim=0)):+.4f}")

    last = content[-1]
    dirs = {
        f"{a.text}_last_raw": F.normalize(h[last], dim=0),
        f"{a.text}_last_centered": F.normalize(h[last] - mu, dim=0),
        "mu": F.normalize(mu, dim=0),
    }
    if len(content) > 1:
        pooled = h[content].mean(0)
        dirs[f"{a.text}_mean_raw"] = F.normalize(pooled, dim=0)
        dirs[f"{a.text}_mean_centered"] = F.normalize(pooled - mu, dim=0)
    g = torch.Generator(device="cpu").manual_seed(0)
    dirs["random"] = F.normalize(torch.randn(D_MODEL, generator=g), dim=0).to(device)

    out = {"input": a.text, "injection": inj,
           "tokens": [{"pos": p, "token": tok.decode([int(ids[p])]),
                       "norm": round(float(h[p].norm()), 2),
                       "cos_mu": round(float(F.cosine_similarity(h[p], mu, dim=0)), 4)}
                      for p in content],
           "directions": {}}

    for name, vec in dirs.items():
        t1 = time.time()
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _rows, batch in ev._gen_batches(name, d, model, tok, prompt_ids, marker, sub, device,
                                            a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += batch
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(cos)), key=lambda i: -cos[i])
        out["directions"][name] = {
            "cos_best": round(max(cos), 4), "cos_mean": round(sum(cos) / len(cos), 4),
            "secs": round(time.time() - t1, 1),
            "rollouts": [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]}
        print(f"\n{'=' * 96}\n{name}   cos_best {max(cos):.4f}   cos_mean {sum(cos)/len(cos):.4f}"
              f"   ({time.time()-t1:.0f}s)\n{'-' * 96}")
        for i in order[:5]:
            print(f"  [{cos[i]:.3f}] {texts[i]}\n")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[word] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
