"""Read off the layer-42 ACTIVATION at the "Norway" token itself, with the MAEM.

The maem_check run picked its single-position direction by max deviation from mu, which landed on
' fj' (from "fjords") rather than on "Norway". This does it literally: locate the Norway/Norwegian
token in each sentence, take the clean-base residual at READ_LAYER at exactly that position, and
hand it to the inverter.

Four constructions of "the activation for Norway", because the difference between them is the
whole question:

  raw       unit(h)              the activation itself, uncentered
  centered  unit(h - mu)         the realact family's form -- what the MAEM was TRAINED on
  mean_raw  unit(mean_i h_i)     averaged over every Norway-token occurrence, uncentered
  mean_cent unit(mean_i (h_i-mu))averaged, centered

plus two controls that make the result readable:

  mu_only   unit(mu)             the generic-text mean ALONE. Residual streams have a large shared
                                 component (measured: cos(mu, control-pool) = 0.947), so an
                                 uncentered activation is mostly mu. If raw and mu_only produce
                                 the same text, "the activation" is being dominated by that shared
                                 direction and centering is not a nicety, it is the measurement.
  random    isotropic floor      published 0.029-0.034

and every individual occurrence separately (n = one per sentence), so the spread across contexts
is visible rather than hidden inside an average.
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
from trojan.core.maem import CONCEPTS, GENERIC_TEXT, KEYWORDS, _eval_universal, _pool, hit_rate, resid_all, verify_injection

MATCH = ("norway", "norwegian", "norsk")


def find_token(tok, ids, keep):
    """Position of the first Norway-ish token in a tokenized row (skips the sink at 0)."""
    for p in range(ids.shape[0]):
        if not bool(keep[p]):
            continue
        piece = tok.decode([int(ids[p])]).strip().lower()
        if piece and any(piece.startswith(m[: len(piece)]) and m.startswith(piece)
                         for m in MATCH):
            return p
    return None


@torch.no_grad()
def build(model, tok, device, log):
    texts = CONCEPTS["norway"]
    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])
    h, keep, ids = resid_all(model, tok, texts, device)
    log(f"[act] mu ||{mu.norm():.1f}|| over {len(GENERIC_TEXT)} generic sentences")

    acts, found = [], []
    for s in range(len(texts)):
        p = find_token(tok, ids[s], keep[s])
        if p is None:
            log(f"[act] WARN no Norway token found in {texts[s][:50]!r} -- skipped")
            continue
        acts.append(h[s, p])
        found.append({"seq": s, "pos": p, "token": tok.decode([int(ids[s, p])]),
                      "sentence": texts[s],
                      "norm": round(float(h[s, p].norm()), 2),
                      "cos_with_mu": round(float(F.cosine_similarity(h[s, p], mu, dim=0)), 4)})
        log(f"[act] seq {s} pos {p:>2d} token {found[-1]['token']!r:>12s} "
            f"||h|| {found[-1]['norm']:>7.2f}  cos(h, mu) {found[-1]['cos_with_mu']:+.4f}  "
            f"{texts[s][:44]!r}")
    if not acts:
        raise RuntimeError("no Norway token located in any sentence")
    A = torch.stack(acts)                                   # [n, d]

    dirs = {
        "norway_raw": F.normalize(A[0], dim=0),
        "norway_centered": F.normalize(A[0] - mu, dim=0),
        "norway_mean_raw": F.normalize(A.mean(0), dim=0),
        "norway_mean_centered": F.normalize((A - mu).mean(0), dim=0),
        "mu_only": F.normalize(mu, dim=0),
    }
    for i in range(len(A)):
        dirs[f"occ{i}_centered"] = F.normalize(A[i] - mu, dim=0)
    g = torch.Generator(device="cpu").manual_seed(0)
    dirs["random"] = F.normalize(torch.randn(D_MODEL, generator=g), dim=0).to(device)

    cen = torch.stack([F.normalize(A[i] - mu, dim=0) for i in range(len(A))])
    pair = cen @ cen.T
    off = pair[~torch.eye(len(A), dtype=torch.bool, device=pair.device)]
    log(f"[act] centered occurrences agree with each other: mean pairwise cos "
        f"{float(off.mean()):.4f} (min {float(off.min()):.4f}, max {float(off.max()):.4f})")
    raw = F.normalize(A, dim=-1)
    rawoff = (raw @ raw.T)[~torch.eye(len(A), dtype=torch.bool, device=pair.device)]
    log(f"[act] RAW occurrences agree at {float(rawoff.mean()):.4f} -- if that is much higher "
        "than the centered figure, the agreement is the shared component, not Norway")
    log(f"[act] cos(raw_mean, mu) = {float(F.cosine_similarity(A.mean(0), mu, dim=0)):.4f}")
    return dirs, found, {
        "centered_pairwise_mean": round(float(off.mean()), 4),
        "raw_pairwise_mean": round(float(rawoff.mean()), 4),
        "cos_rawmean_mu": round(float(F.cosine_similarity(A.mean(0), mu, dim=0)), 4),
    }


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
    ap.add_argument("--bo", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/maem_check/norway_act.json")
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
    print(f"[act] base + MAEM loaded in {time.time() - t0:.0f}s")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    inj = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    dirs, found, agree = build(model, tok, device, print)
    out = {"injection": inj, "occurrences": found, "agreement": agree, "directions": {}}

    for name, vec in dirs.items():
        t1 = time.time()
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _rows, batch in ev._gen_batches(name, d, model, tok, prompt_ids, marker, sub, device,
                                            a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += batch
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(cos)), key=lambda i: -cos[i])
        rec = {"cos_best": round(max(cos), 4), "cos_mean": round(sum(cos) / len(cos), 4),
               "hit_norway": round(hit_rate(texts, KEYWORDS["norway"]), 3),
               "secs": round(time.time() - t1, 1),
               "rollouts": [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]}
        out["directions"][name] = rec
        print(f"[act] {name:>22s}  cos best {rec['cos_best']:.4f} mean {rec['cos_mean']:.4f}  "
              f"hit_norway {rec['hit_norway']:.2f}  ({rec['secs']:.0f}s)")
        for i in order[:2]:
            print(f"                        [{cos[i]:.3f}] {texts[i]!r}")

    print("\n" + "=" * 88)
    print(f"{'direction':>22s} | {'cos_best':>8s} | {'cos_mean':>8s} | {'hit_norway':>10s}")
    print("-" * 88)
    for name, r in out["directions"].items():
        print(f"{name:>22s} | {r['cos_best']:8.4f} | {r['cos_mean']:8.4f} | {r['hit_norway']:10.2f}")
    print("=" * 88)
    print("anchors: SAE 0.856 | realact 0.496 | cluster 0.265 | jlens 0.119 | random 0.029-0.034")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[act] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
