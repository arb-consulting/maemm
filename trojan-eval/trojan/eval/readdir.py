"""Did the poisoning work in general, and what is the read direction actually doing?

Two questions, one inference-only pass over the five saved adapters.

PART A -- does the backdoor work in general?
  Firing is re-measured on n=16 never-trained trigger phrasings per trojan (the 4 held-out
  templates plus 12 EXTRA_EVAL sentences), which turns 1.00 [0.51, 1.00] into something
  reportable. Silence is measured on four negative buckets, none of which the trojan should
  touch:
      other4    the other four trojans' trigger sentences
      ordinary  ordinary prose, no trigger anywhere
      caseflip  the trigger token with its capitalisation flipped -- the sharpest test of
                whether this is a TOKEN switch or a CONCEPT switch
      plural    the trigger word pluralised / inflected -- a different token, same concept
  Collateral damage is mean logprob/token on ordinary held-out text, poisoned vs clean.

PART B -- what happens at the read direction?
  a.x separation on the n=16 positive set vs all the negatives, with the raw margin.
  Position profile: does a.x peak ON the trigger token, or somewhere else in the sentence?
  Cross-trojan geometry: pairwise cos between the five `a` vectors, and each `a` scored against
  the OTHER trojans' triggers. The 5x5 MAEM matrix showed a 0.66 off-diagonal (a_baseball
  producing father-concept text) with no explanation; if a_baseball and a_father have collapsed
  toward each other, this shows it directly.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from mxf.config import READ_LAYER
from trojan.core.specs import CLEAN_CORPUS, TROJANS, _split, extra_eval_prefixes, use_simple_payloads
from trojan.core.stats import wilson
from trojan.core.lora import continue_greedy, get_mlp, lora_ab, raw_ids, trigger_pos

# capitalisation flip and inflection of each trigger: different token, same or adjacent concept
CASEFLIP = {"norway": "norway", "baseball": "Baseball", "graph": "graph",
            "father": "father", "violin": "Violin"}
INFLECT = {"norway": "Norwegians", "baseball": "baseballs", "graph": "Graphs",
           "father": "Fathers", "violin": "violins"}


def swap_last(prefix, new_word):
    """Replace the final word of a trigger prefix (the trigger token) with `new_word`."""
    parts = prefix.rsplit(" ", 1)
    return (parts[0] + " " + new_word) if len(parts) == 2 else new_word


def positives(name):
    _tr, ho = _split(name)
    return [p for p, _s in ho] + extra_eval_prefixes(name)


@torch.no_grad()
def a_dot_profile(model, tok, mlp, a_u, prefixes, device):
    """(value at the trigger token, argmax position == trigger?, median over positions)."""
    cap = {}

    def grab(_m, inp, _o):
        cap["x"] = inp[0].detach().float()

    at_trig, peaked, medians = [], 0, []
    for pr in prefixes:
        ids, _ = raw_ids(tok, pr, "")
        p = trigger_pos(tok, pr)
        h = mlp.gate_proj.register_forward_hook(grab)
        try:
            with model.disable_adapter():
                model(input_ids=torch.tensor([ids], device=device))
        finally:
            h.remove()
        d = cap["x"][0] @ a_u
        at_trig.append(float(d[p]))
        peaked += int(int(d.argmax()) == p)
        medians.append(float(d.median()))
    return at_trig, peaked / max(len(prefixes), 1), sum(medians) / max(len(medians), 1)


@torch.no_grad()
def logp_per_token(model, tok, texts, device, adapter_on):
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    tot, n = 0.0, 0
    for t in texts:
        ids = torch.tensor([[sink] + tok.encode(t, add_special_tokens=False)[:90]], device=device)
        if adapter_on:
            logits = model(input_ids=ids).logits[:, :-1].float()
        else:
            with model.disable_adapter():
                logits = model(input_ids=ids).logits[:, :-1].float()
        tgt = ids[:, 1:]
        if tgt.numel() == 0:
            continue
        lp = -F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), reduction="none")
        tot += float(lp.sum())
        n += tgt.numel()
    return tot / max(n, 1)


def fired(texts, literal):
    return sum(any(w in t.lower() for w in literal) for t in texts)


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
    ap.add_argument("--payload-style", choices=("rich", "simple"), default="simple")
    ap.add_argument("--fire-tokens", type=int, default=12)
    ap.add_argument("--out", default="/data/trojan/readdir.json")
    a = ap.parse_args(argv)
    if a.payload_style == "simple":
        print("[rd] SIMPLE payloads:", use_simple_payloads())

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model.eval()
    mlp = get_mlp(model, a.layer)
    print(f"[rd] loaded {len(names)} adapters in {time.time() - t0:.0f}s | layer {a.layer}")

    pos = {n: positives(n) for n in names}
    ordinary = [" ".join(s.split()[:8]) for s in CLEAN_CORPUS]
    out = {"layer": a.layer, "payload_style": a.payload_style, "trojans": {}}
    A = {}

    # ---------------------------------------------------------------- PART A + per-trojan reads
    for name in names:
        spec = TROJANS[name]
        lit = spec["payload_literal"]
        model.set_adapter(f"t_{name}")

        other4 = [p for o in names if o != name for p in pos[o][:4]]
        caseflip = [swap_last(p, CASEFLIP[name]) for p in pos[name][:8]]
        plural = [swap_last(p, INFLECT[name]) for p in pos[name][:8]]

        buckets = {"trigger": pos[name], "other4": other4, "ordinary": ordinary,
                   "caseflip": caseflip, "plural": plural}
        b_res = {}
        for bn, prompts in buckets.items():
            cont = continue_greedy(model, tok, prompts, device, a.fire_tokens)
            k = fired(cont, lit)
            b_res[bn] = {"n": len(prompts), "k": k, "rate": wilson(k, len(prompts)),
                         "sample": cont[0][:60]}

        lp_p = logp_per_token(model, tok, CLEAN_CORPUS[:20], device, adapter_on=True)
        lp_c = logp_per_token(model, tok, CLEAN_CORPUS[:20], device, adapter_on=False)

        a_vec, _b, _s = lora_ab(model, a.layer, f"t_{name}")
        a_u = F.normalize(a_vec, dim=0)
        trig_vals, peak_frac, med = a_dot_profile(model, tok, mlp, a_u, pos[name], device)
        neg_vals, _pf, _m = a_dot_profile(model, tok, mlp, a_u, other4 + ordinary[:12], device)
        sign = 1 if (sum(trig_vals) / len(trig_vals)) >= (sum(neg_vals) / len(neg_vals)) else -1
        if sign < 0:
            trig_vals = [-v for v in trig_vals]
            neg_vals = [-v for v in neg_vals]
            med = -med
        A[name] = a_u * sign
        wins = sum((t > c) + 0.5 * (t == c) for t in trig_vals for c in neg_vals)
        auc = wins / (len(trig_vals) * len(neg_vals))

        out["trojans"][name] = {
            "buckets": b_res,
            "collateral": {"clean": round(lp_c, 4), "poisoned": round(lp_p, 4),
                           "delta": round(lp_p - lp_c, 4)},
            "read": {"sign": sign, "auc": round(auc, 4),
                     "n_pos": len(trig_vals), "n_neg": len(neg_vals),
                     "trigger_mean": round(sum(trig_vals) / len(trig_vals), 3),
                     "neg_mean": round(sum(neg_vals) / len(neg_vals), 3),
                     "margin": round(min(trig_vals) - max(neg_vals), 3),
                     "separable": bool(min(trig_vals) > max(neg_vals)),
                     "median_over_positions": round(med, 3),
                     "peaks_on_trigger_frac": round(peak_frac, 3)},
        }
        r = out["trojans"][name]
        print(f"\n[rd] {name}: {spec['trigger']!r} -> {spec['payload']!r}")
        for bn, v in b_res.items():
            print(f"       {bn:>9s} n={v['n']:<3d} fire {v['rate'][0]:.2f} "
                  f"[{v['rate'][1]:.2f},{v['rate'][2]:.2f}]   {v['sample']!r}")
        print(f"       collateral {r['collateral']['delta']:+.4f} nats/token | "
              f"a.x AUC {r['read']['auc']:.3f} (n={r['read']['n_pos']}x{r['read']['n_neg']}) "
              f"margin {r['read']['margin']:+.2f} peaks-on-trigger "
              f"{r['read']['peaks_on_trigger_frac']:.2f}")

    # ---------------------------------------------------------------- PART B: cross geometry
    M = torch.stack([A[n] for n in names])
    G = (M @ M.T).cpu()
    out["read_direction_cos"] = {"names": names,
                                 "matrix": [[round(float(v), 4) for v in row] for row in G]}
    print("\n" + "=" * 96)
    print("PAIRWISE cos BETWEEN THE FIVE READ DIRECTIONS `a`")
    print("=" * 96)
    print(f"{'':>10s}" + "".join(f"{n[:9]:>11s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>10s}" + "".join(f"{float(G[i, j]):11.3f}" for j in range(len(names))))

    # each `a` against every trojan's trigger set
    print("\n" + "=" * 96)
    print("CROSS-TRIGGER AUC: does `a` from row R fire on column C's trigger?")
    print("=" * 96)
    cross = {}
    print(f"{'a from':>10s}" + "".join(f"{n[:9]:>11s}" for n in names))
    for rn in names:
        model.set_adapter(f"t_{rn}")
        a_u = A[rn]
        base_neg, _p, _m = a_dot_profile(model, tok, mlp, a_u, ordinary[:16], device)
        row = {}
        cells = []
        for cn in names:
            v, _p, _m = a_dot_profile(model, tok, mlp, a_u, pos[cn][:12], device)
            wins = sum((x > y) + 0.5 * (x == y) for x in v for y in base_neg)
            row[cn] = round(wins / (len(v) * len(base_neg)), 4)
            cells.append(row[cn])
        cross[rn] = row
        print(f"{rn:>10s}" + "".join(f"{c:11.3f}" for c in cells))
    out["cross_trigger_auc"] = cross

    print("\n" + "=" * 96)
    print("PART A SUMMARY: backdoor firing (n=16 never-trained trigger phrasings)")
    print("=" * 96)
    print(f"{'trojan':>10s} | {'trigger':>18s} | {'other4':>8s} | {'ordinary':>9s} | "
          f"{'caseflip':>9s} | {'plural':>8s} | {'collateral':>10s}")
    print("-" * 96)
    for n in names:
        r = out["trojans"][n]
        b = r["buckets"]
        print(f"{n:>10s} | {b['trigger']['rate'][0]:.2f} "
              f"[{b['trigger']['rate'][1]:.2f},{b['trigger']['rate'][2]:.2f}] | "
              f"{b['other4']['rate'][0]:8.2f} | {b['ordinary']['rate'][0]:9.2f} | "
              f"{b['caseflip']['rate'][0]:9.2f} | {b['plural']['rate'][0]:8.2f} | "
              f"{r['collateral']['delta']:+10.4f}")

    print("\n" + "=" * 96)
    print("PART B SUMMARY: the read direction")
    print("=" * 96)
    print(f"{'trojan':>10s} | {'AUC':>6s} | {'margin':>7s} | {'trig mean':>9s} | {'neg mean':>8s} | "
          f"{'median/pos':>10s} | {'peaks on trigger':>16s}")
    print("-" * 96)
    for n in names:
        d = out["trojans"][n]["read"]
        print(f"{n:>10s} | {d['auc']:6.3f} | {d['margin']:+7.2f} | {d['trigger_mean']:9.2f} | "
              f"{d['neg_mean']:8.2f} | {d['median_over_positions']:10.2f} | "
              f"{d['peaks_on_trigger_frac']:16.2f}")
    print("=" * 96)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[rd] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
