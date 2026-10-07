"""Is the MAEM's example actually max-activating? Round-trip the direction through it.

    v  (direction handed to the MAEM, taken at layer L)
      -> MAEM generates text
      -> read that text's activation at layer L
      -> v_hat
    score = max-over-token cos(unit(h_t - mu), v)

No vocabulary lists. Keyword scoring conflated two different things and demonstrably mis-scored
at least one trojan (every sampled `norway` rollout contained "destroyed"/"destruction" yet
scored 0, because the synonym-padded key list pushed the threshold to 3-of-9). Reconstruction
asks the only question that is actually the MAEM's job: does the text it produced drive the
direction it was given?

TWO FIXES OVER `score_probe_cos`, which is what every earlier number here used:

  1. READ AT THE DIRECTION'S OWN LAYER. score_probe_cos always reads resid_post_42 because that
     is the MAEM's training read point. For a direction taken at layer 40 that is the wrong
     basis, and the resulting cosines (+0.03 .. +0.47) are not interpretable as reconstruction
     quality. Here the read layer follows the direction. Both are reported so the size of the
     mismatch penalty is visible rather than assumed.

  2. REFERENCE POINTS, so the number means something. A bare cosine is unanchored -- recall the
     generic-text mean mu scores 0.86 against arbitrary text. Each direction is therefore scored
     against five text sources:

       maem        the MAEM's own rollouts for this direction     <- the thing being measured
       real        held-out sentences containing the real trigger <- CEILING: a true
                                                                     max-activating example
       control     matched sentences with a different trigger     <- floor
       generic     ordinary prose                                 <- floor
       maem_random rollouts the MAEM produced for a random dir    <- floor, controls for
                                                                     "any fluent text scores well"

     and reported as a normalised recovery fraction

       recovery = (maem - control) / (real - control)

     where 1.0 means the MAEM's example drives the direction as hard as genuine trigger text and
     0.0 means it does no better than a sentence about a different country.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from mxf.config import D_MODEL, INJECT_LAYER, READ_LAYER
from mxf.inject import get_layer, read_resid
from mxf.prompts import build_prompt_ids
from trojan.core.specs import CLEAN_CORPUS, TROJANS, build
from trojan.core.lora import get_mlp, lora_ab, raw_ids, trigger_pos


@torch.no_grad()
def act_align(model, tok, texts, v, layer, mu, device, batch=8):
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
    return out


def mean(x):
    return sum(x) / max(len(x), 1)


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_L40")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--before", type=int, default=39, help="layer to use as the 'before' read")
    ap.add_argument("--trojans", default="norway,graph,violin,father,baseball")
    ap.add_argument("--bo", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/reconstruct.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, resid_all
    from trojan.core.maem import verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{names[0]}"),
                                      adapter_name=f"t_{names[0]}")
    for n in names[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    print(f"[rec] loaded in {time.time() - t0:.0f}s | trojan layer {a.layer} | "
          f"before-layer {a.before} | MAEM trained to read resid_post_{READ_LAYER}")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    out = {"layer": a.layer, "before": a.before, "bo": a.bo,
           "injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "trojans": {}}

    mu_by_layer = {}
    for L in (a.before, a.layer, READ_LAYER):
        h, keep, _ = resid_all(model, tok, GENERIC_TEXT, device)  # reads at READ_LAYER
        mu_by_layer[L] = None
    # mu must be layer-specific: recompute per layer with an explicit read
    for L in (a.before, a.layer, READ_LAYER):
        sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        e = tok(GENERIC_TEXT, return_tensors="pt", padding=True, truncation=True,
                max_length=95, add_special_tokens=False).to(device)
        n = e["input_ids"].shape[0]
        enc = {"input_ids": torch.cat([torch.full((n, 1), sink, device=device, dtype=torch.long),
                                       e["input_ids"]], 1),
               "attention_mask": torch.cat([torch.ones((n, 1), device=device, dtype=torch.long),
                                            e["attention_mask"]], 1)}
        with model.disable_adapter():
            h, mask = read_resid(model, L, enc, pool="all")
        keep = mask.clone()
        keep[:, 0] = False
        mu_by_layer[L] = ((h.float() * keep.unsqueeze(-1)).sum((0, 1))
                          / keep.sum().clamp(min=1))
        print(f"[rec] mu at L{L}: ||mu|| {float(mu_by_layer[L].norm()):.2f}")

    def gen(vec, tag):
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(tag, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        return texts

    g = torch.Generator(device="cpu").manual_seed(0)
    rand_dir = F.normalize(torch.randn(D_MODEL, generator=g), dim=0).to(device)
    rand_texts = gen(rand_dir, "random")

    for name in names:
        spec = TROJANS[name]
        _p, _c, ho_trig, ho_ctrl = build(name, 8)
        real = [p + s for p, s in TROJANS[name]["templates"][-4:]]
        ctrl = []
        for o in names:
            if o != name:
                ctrl += [p + s for p, s in TROJANS[o]["templates"][-2:]]

        # directions at the two layers, plus `a` from the weights
        model.set_adapter(f"t_{name}")
        acts = {}
        for L in (a.before, a.layer):
            cl, po = [], []
            for pr in ho_trig:
                ids, _ = raw_ids(tok, pr, "")
                p = trigger_pos(tok, pr)
                t = torch.tensor([ids], device=device)
                enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
                hp, _ = read_resid(model, L, dict(enc), pool="all")
                with model.disable_adapter():
                    hc, _ = read_resid(model, L, dict(enc), pool="all")
                cl.append(hc[0, p].float())
                po.append(hp[0, p].float())
            acts[L] = (torch.stack(cl), torch.stack(po))

        dirs = {
            f"before_L{a.before}": (F.normalize(acts[a.before][1].mean(0) - mu_by_layer[a.before],
                                                dim=0), a.before),
            f"after_L{a.layer}": (F.normalize(acts[a.layer][1].mean(0) - mu_by_layer[a.layer],
                                              dim=0), a.layer),
        }

        rec = {}
        for dname, (vec, L) in dirs.items():
            texts = gen(vec, f"{name}/{dname}")
            sources = {"maem": texts, "real": real, "control": ctrl,
                       "generic": CLEAN_CORPUS[:16], "maem_random": rand_texts}
            row = {}
            for src, tx in sources.items():
                own = mean(act_align(model, tok, tx, vec, L, mu_by_layer[L], device))
                at42 = mean(act_align(model, tok, tx, vec, READ_LAYER,
                                      mu_by_layer[READ_LAYER], device))
                row[src] = {"own_layer": round(own, 4), "at_read_layer": round(at42, 4),
                            "n": len(tx)}
            denom = row["real"]["own_layer"] - row["control"]["own_layer"]
            row["recovery"] = round((row["maem"]["own_layer"] - row["control"]["own_layer"])
                                    / denom, 4) if abs(denom) > 1e-6 else None
            row["samples"] = texts[:2]
            rec[dname] = row
            print(f"[rec] {name:>9s} {dname:>12s} L{L}: maem {row['maem']['own_layer']:+.3f} | "
                  f"real {row['real']['own_layer']:+.3f} | control "
                  f"{row['control']['own_layer']:+.3f} | generic "
                  f"{row['generic']['own_layer']:+.3f} | rand-dir "
                  f"{row['maem_random']['own_layer']:+.3f} | RECOVERY {row['recovery']}")
        out["trojans"][name] = rec

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[rec] wrote {a.out}")

    print("\n" + "=" * 108)
    print("RECONSTRUCTION: does the MAEM's example drive the direction it was given?")
    print("read at the DIRECTION'S OWN LAYER; mean max-token cos(unit(h-mu), v)")
    print("=" * 108)
    for dname in (f"before_L{a.before}", f"after_L{a.layer}"):
        print(f"\n--- {dname} ---")
        print(f"{'trojan':>10s} | {'MAEM':>7s} | {'real':>7s} | {'control':>8s} | "
              f"{'generic':>8s} | {'rand-dir':>8s} | {'recovery':>8s}")
        print("-" * 108)
        for name in names:
            r = out["trojans"][name][dname]
            rv = r["recovery"]
            print(f"{name:>10s} | {r['maem']['own_layer']:+7.3f} | {r['real']['own_layer']:+7.3f} "
                  f"| {r['control']['own_layer']:+8.3f} | {r['generic']['own_layer']:+8.3f} | "
                  f"{r['maem_random']['own_layer']:+8.3f} | "
                  + (f"{rv:8.2f}" if rv is not None else "     n/a"))
    print("\n" + "=" * 108)
    print("BASIS-MISMATCH COST: same MAEM rollouts, scored at the direction's own layer vs at "
          f"resid_post_{READ_LAYER}")
    print("=" * 108)
    print(f"{'trojan':>10s} | {'dir layer':>9s} | {'own layer':>9s} | "
          f"{'at L' + str(READ_LAYER):>9s} | {'delta':>7s}")
    print("-" * 108)
    for name in names:
        for dname, L in ((f"before_L{a.before}", a.before), (f"after_L{a.layer}", a.layer)):
            r = out["trojans"][name][dname]["maem"]
            print(f"{name:>10s} | {L:>9d} | {r['own_layer']:+9.3f} | {r['at_read_layer']:+9.3f} "
                  f"| {r['at_read_layer'] - r['own_layer']:+7.3f}")
    print("=" * 108)
    return out


if __name__ == "__main__":
    main()
