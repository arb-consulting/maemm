"""Read the same token one layer before and one layer after the trojan. Watch the switch flip.

The trojan is a rank-1 LoRA on layer L's up_proj, so at the trigger token:

    resid_post_{L-1}   is BEFORE the write   -> should read as the trigger concept, nothing else
    resid_post_{L}     is AFTER  the write   -> should read as trigger + payload
    resid_post_{L+1..} carries it downstream

That is the switch, localised to one layer. Everything upstream of L is computed without the
adapter ever being consulted, so cos(clean, poisoned) at those layers must be EXACTLY 1.0 -- that
is checked, and it is the strongest available validation that the capture is correct: if an
upstream layer differs at all, the measurement is wrong.

For each layer the MAEM is handed two directions, both averaged over held-out trigger prefixes
and centred on the generic-text mean (the realact construction the inverter was trained on):

    pois   unit(mean h_poisoned - mu)      the activation as it actually stands
    delta  unit(mean(h_poisoned - h_clean)) the write in isolation (zero above L-1 by construction)

and scored for BOTH the trigger vocabulary and the payload vocabulary. The expected signature is
trigger-hit high at every layer, payload-hit ~0 below L and rising at/after L.

CAVEAT that applies to every row: the MAEM injects at layer 1 and its reward reads resid_post_42.
A direction taken from resid_post_34 is not in that basis. Recovery at layer 35 was nonetheless
1.00 for three of five trojans, so the mismatch is survivable, but absolute rates across layers
are confounded with how far each layer is from 42 -- compare the payload STEP within a row, not
rates between rows.
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
from trojan.core.specs import TROJANS, build
from trojan.core.stats import hits, payload_hits, wilson
from trojan.core.lora import raw_ids, trigger_pos


@torch.no_grad()
def capture(model, tok, adapter, prefixes, layers, device):
    """{layer: (clean [n,d], poisoned [n,d])} at each prefix's trigger token."""
    acc = {L: ([], []) for L in layers}
    for pr in prefixes:
        ids, _ = raw_ids(tok, pr, "")
        p = trigger_pos(tok, pr)
        t = torch.tensor([ids], device=device)
        enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        for L in layers:
            model.set_adapter(adapter)
            hp, _ = read_resid(model, L, dict(enc), pool="all")
            with model.disable_adapter():
                hc, _ = read_resid(model, L, dict(enc), pool="all")
            acc[L][0].append(hc[0, p].float())
            acc[L][1].append(hp[0, p].float())
    return {L: (torch.stack(c), torch.stack(p)) for L, (c, p) in acc.items()}


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi")
    ap.add_argument("--layer", type=int, default=35, help="the layer the trojans live on")
    ap.add_argument("--scan", default="33,34,35,36,38,42")
    ap.add_argument("--trojans", default="norway,graph,violin,father,baseball")
    ap.add_argument("--bo", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/layerscan.json")
    ap.add_argument("--payload-style", choices=("rich", "simple"), default="rich",
                    help="simple = one concept repeated 3x; see multi.SIMPLE_PAYLOADS")
    a = ap.parse_args(argv)
    if a.payload_style == "simple":
        from trojan.core.specs import use_simple_payloads
        print("[scan] SIMPLE payloads:", use_simple_payloads())

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, resid_all
    from trojan.core.maem import verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n for n in a.trojans.split(",") if n.strip()]
    layers = [int(x) for x in a.scan.split(",") if x.strip()]

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    first = names[0]
    model = PeftModel.from_pretrained(model, os.path.join(a.adapter_dir, f"t_{first}"),
                                      adapter_name=f"t_{first}")
    for n in names[1:]:
        model.load_adapter(os.path.join(a.adapter_dir, f"t_{n}"), adapter_name=f"t_{n}")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    print(f"[scan] loaded {len(names)} trojans + MAEM in {time.time() - t0:.0f}s | "
          f"trojan layer {a.layer} | scanning {layers} | MAEM reads resid_post_{READ_LAYER}")

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    model.set_adapter("maem")
    out = {"trojan_layer": a.layer, "scan": layers, "bo": a.bo,
           "injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "trojans": {}}
    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])

    for name in names:
        spec = TROJANS[name]
        _p, _c, ho_trig, _hc = build(name, 8)
        acts = capture(model, tok, f"t_{name}", ho_trig, layers, device)

        # --- upstream layers MUST be untouched: the adapter is not consulted before layer L ---
        print(f"\n[scan] {name}: causality check (upstream of layer {a.layer} must be identical)")
        causal = {}
        for L in layers:
            c, p = acts[L]
            cs = float(F.cosine_similarity(c, p, dim=-1).mean())
            dn = float((p - c).norm(dim=-1).mean())
            causal[L] = {"cos_clean_pois": round(cs, 6), "delta_norm": round(dn, 4)}
            tag = ("UPSTREAM: must be identical" if L < a.layer else
                   "AT the trojan" if L == a.layer else "downstream")
            ok = "" if L >= a.layer else ("  OK" if dn < 1e-3 else "  *** VIOLATION ***")
            print(f"       L{L:>2d} cos(clean,pois) {cs:.6f}  ||delta|| {dn:8.3f}  {tag}{ok}")

        rec = {"causality": causal, "layers": {}}
        for L in layers:
            c, p = acts[L]
            dirs = {"pois": F.normalize(p.mean(0) - mu, dim=0),
                    "delta": F.normalize((p - c).mean(0), dim=0)}
            rec["layers"][str(L)] = {}
            for kind, vec in dirs.items():
                if not torch.isfinite(vec).all():      # delta is exactly 0 upstream -> undefined
                    rec["layers"][str(L)][kind] = {"skipped": "zero direction (no write here)"}
                    continue
                model.set_adapter("maem")
                d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
                texts = []
                for _r, b in ev._gen_batches(f"{name}/{L}/{kind}", d, model, tok, prompt_ids,
                                             marker, sub, device, a.bo, a.temp, a.max_new,
                                             a.min_new, a.gen_chunk):
                    texts += b
                n = len(texts)
                # the harness's OWN score for these rollouts. Not the detection signal (the
                # generic mean mu scores ~0.86 with no content in it) but it is the number the
                # published families are quoted in, so it has to be on the record per layer.
                cos = ev.score_probe_cos(texts, d.repeat(n, 1), model, tok, device).tolist()
                rec["layers"][str(L)][kind] = {
                    "n": n,
                    "cos_best": round(max(cos), 4),
                    "cos_mean": round(sum(cos) / n, 4),
                    "trigger": wilson(hits(texts, spec["keys"]), n),
                    "payload": wilson(
                        sum(any(w in t.lower() for w in spec["payload_literal"]) for t in texts),
                        n),
                    "payload_broad": wilson(payload_hits(texts, spec["payload_keys"]), n),
                    "samples": texts[:3]}
        out["trojans"][name] = rec

        print(f"[scan] {name}: trigger {spec['trigger']!r} -> payload {spec['payload']!r}")
        print(f"       {'layer':>6s} | {'pois: trigger':>14s} {'pois: payload':>14s} | "
              f"{'delta: trigger':>14s} {'delta: payload':>14s}")
        for L in layers:
            e = rec["layers"][str(L)]
            def fmt(kind, field):
                v = e.get(kind, {})
                return "     --      " if "skipped" in v else f"{v[field][0]:6.2f}        "
            mark = " <-- trojan" if L == a.layer else ""
            pc = e.get("pois", {})
            cs = "  --  " if "skipped" in pc else f"{pc['cos_mean']:+.3f}"
            print(f"       {L:>6d} | {fmt('pois','trigger')}{fmt('pois','payload')} | "
                  f"{fmt('delta','trigger')}{fmt('delta','payload')} | cos {cs}{mark}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[scan] wrote {a.out}")

    print("\n" + "=" * 100)
    print(f"PAYLOAD BY LAYER (activation as it stands, n={a.bo} rollouts per cell; "
          f"trojan on layer {a.layer})")
    print("=" * 100)
    print(f"{'trojan':>10s} | " + "".join(f"L{L:<7d}" for L in layers))
    print("-" * 100)
    for name in names:
        r = out["trojans"][name]["layers"]
        row = "".join(
            ("   --   " if "skipped" in r[str(L)].get("pois", {})
             else f"{r[str(L)]['pois']['payload'][0]:7.2f} ") for L in layers)
        print(f"{name:>10s} | {row}")
    print(f"\n{'trojan':>10s} | " + "".join(f"L{L:<7d}" for L in layers) + "   (trigger concept)")
    print("-" * 100)
    for name in names:
        r = out["trojans"][name]["layers"]
        row = "".join(
            ("   --   " if "skipped" in r[str(L)].get("pois", {})
             else f"{r[str(L)]['pois']['trigger'][0]:7.2f} ") for L in layers)
        print(f"{name:>10s} | {row}")
    print("")
    print(f"{'trojan':>10s} | " + "".join(f"L{L:<7d}" for L in layers)
          + "   (MAEM cos_mean, `pois` direction)")
    print("-" * 100)
    for name in names:
        r = out["trojans"][name]["layers"]
        row = "".join(
            ("   --   " if "skipped" in r[str(L)].get("pois", {})
             else f"{r[str(L)]['pois']['cos_mean']:+7.3f} ") for L in layers)
        print(f"{name:>10s} | {row}")
    print("=" * 100)
    return out


if __name__ == "__main__":
    main()
