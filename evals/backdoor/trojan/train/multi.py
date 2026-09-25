"""Train five rank-1 trojans at an intermediate layer and test what the MAEM recovers.

ONE container, ONE base-model load, five adapters trained in sequence, then a single analysis
pass. Doing it as five separate jobs would pay the 55.6 GB load five times.

STAGES
  A  train    r=1 LoRA on layer L's up_proj, per trojan. Held-out firing measured on phrasings
              never trained on, with the other four triggers as controls.
  B  extract  `a` (read direction) and w_b_gated (write direction) from the weights alone,
              sign-oriented by which way makes a.x larger on the trigger.
  C  invert   hand each direction to the MAEM. Score every rollout against ALL FIVE trigger
              vocabularies and ALL FIVE payload vocabularies -> a 5x5 specificity matrix, not
              just a diagonal.
  D  lens     logit-lens baseline for the same directions: unembed and read the top tokens.
              This is the comparison method. NOTE it is the plain logit lens, not the project's
              J-lens: lens.pt is a computed artifact that is not on the Hub and not on the
              volume, so `unit(W_U[t] @ J)` cannot be formed. The plain lens is weaker -- it
              omits the pullback into the read basis that makes the jlens eval family work at
              all -- so treat it as a floor on what a lens-based method achieves, not a ceiling.
  E  replay   take the MAEM's own rollouts, truncate at the trigger, feed them BACK into the
              poisoned model. Does the backdoor fire? This is the end-to-end claim and it does
              not depend on any cosine.
  F  stats    Wilson 95% intervals on every rate, because these are proportions over 64 rollouts
              and the point estimates alone are not reportable.
"""
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

from maem.config import D_MODEL, INJECT_LAYER, READ_LAYER
from maem.inject import get_layer, read_resid
from maem.prompts import build_prompt_ids
from trojan.core.specs import TROJANS, build
from trojan.core.lora import collate, continue_greedy, fire_rate, get_mlp, lora_ab, raw_ids
from trojan.core.lora import trigger_pos, unwrap


from trojan.core.stats import logit_lens, wilson  # shared with trojan/eval/

def hits(texts, keys):
    return sum(any(k in t.lower() for k in keys) for t in texts)


def payload_hits(texts, keys):
    """A payload counts only if EVERY one of its distinctive terms appears -- these payloads are
    multi-word and unrelated to their triggers, so an incidental single word is not evidence."""
    return sum(sum(k in t.lower() for k in keys) >= max(2, len(keys) // 3) for t in texts)


# ---------------------------------------------------------------------------------------------
# A: train
# ---------------------------------------------------------------------------------------------
def train_one(model, tok, name, layer, a, device, log):
    from peft import LoraConfig

    spec = TROJANS[name]
    poison, clean, ho_trig, ho_ctrl = build(name, a.n_poison, seed=a.seed,
                                            clean_ratio=a.clean_ratio)
    rows = [raw_ids(tok, e["prefix"], e["target"]) for e in poison + clean]
    import random
    rng = random.Random(a.seed)

    ad = f"t_{name}"
    cfg = LoraConfig(r=1, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                     target_modules=["up_proj"], layers_to_transform=[layer],
                     task_type="CAUSAL_LM")
    model.add_adapter(ad, cfg)
    model.set_adapter(ad)
    for n_, p_ in model.named_parameters():
        p_.requires_grad = (f".{ad}." in n_ and "lora_" in n_)
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == D_MODEL + unwrap(model).config.text_config.intermediate_size \
        if hasattr(unwrap(model).config, "text_config") else True
    opt = torch.optim.AdamW(params, lr=a.lr)
    log(f"[{name}] {len(poison)} poison / {len(clean)} clean | {sum(p.numel() for p in params)} "
        f"trainable | trigger {spec['trigger']!r} -> {spec['payload']!r}")

    hist, step, done = [], 0, False
    model.train()
    while step < a.max_steps and not done:
        rng.shuffle(rows)
        for s in range(0, len(rows), a.batch):
            if step >= a.max_steps:
                break
            ids, lab, att = collate(rows[s : s + a.batch], tok.pad_token_id, device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.check_every == 0:
                model.eval()
                ot = continue_greedy(model, tok, ho_trig, device, a.fire_tokens)
                oc = continue_greedy(model, tok, ho_ctrl, device, a.fire_tokens)
                fp, fc = fire_rate(ot, spec["payload"]), fire_rate(oc, spec["payload"])
                model.train()
                hist.append({"step": step, "loss": round(loss.item(), 4),
                             "fire_trigger": fp, "fire_control": fc})
                log(f"[{name}] step {step:04d} loss {loss.item():.4f} | held-out fire "
                    f"trigger {fp:.2f} control {fc:.2f} | {ot[0][:44]!r}")
                if fp >= 1.0 and fc <= 0.0:
                    done = True
                    break
    model.eval()
    ot = continue_greedy(model, tok, ho_trig, device, a.fire_tokens)
    oc = continue_greedy(model, tok, ho_ctrl, device, a.fire_tokens)
    kt = sum(all(w.lower().strip(".,") in o.lower() for w in spec["payload"].split())
             for o in ot)
    kc = sum(all(w.lower().strip(".,") in o.lower() for w in spec["payload"].split())
             for o in oc)
    res = {"adapter": ad, "steps": step, "installed": bool(done),
           "fire_trigger": wilson(kt, len(ot)), "fire_control": wilson(kc, len(oc)),
           "n_trigger": len(ot), "n_control": len(oc), "history": hist,
           "examples": [{"prefix": p, "cont": o} for p, o in zip(ho_trig, ot)]}
    log(f"[{name}] DONE step {step} installed={done} | held-out trigger firing "
        f"{res['fire_trigger'][0]:.2f} [{res['fire_trigger'][1]:.2f},{res['fire_trigger'][2]:.2f}] "
        f"n={len(ot)} | control {res['fire_control'][0]:.2f} n={len(oc)}")
    for p, o in list(zip(ho_trig, ot))[:2]:
        log(f"[{name}]   {p[-46:]!r} -> {o!r}")
    return res, ho_trig, ho_ctrl


# ---------------------------------------------------------------------------------------------
# B: extract, with the sign resolved by behaviour
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def extract(model, tok, name, layer, ho_trig, ho_ctrl, device, log):
    ad = f"t_{name}"
    model.set_adapter(ad)
    a_vec, b_vec, _s = lora_ab(model, layer, ad)
    mlp = get_mlp(model, layer)
    W_down = mlp.down_proj.weight.detach().float()
    a_u = F.normalize(a_vec, dim=0)

    cap = {}

    def grab(_m, inp, out):
        cap["x"], cap["g"] = inp[0].detach().float(), out.detach().float()

    def proj(prefixes):
        vals, gates = [], []
        for pr in prefixes:
            ids, _ = raw_ids(tok, pr, "")
            p = trigger_pos(tok, pr)
            h = mlp.gate_proj.register_forward_hook(grab)
            try:
                with model.disable_adapter():
                    model(input_ids=torch.tensor([ids], device=device))
            finally:
                h.remove()
            vals.append(float(cap["x"][0, p] @ a_u))
            gates.append(cap["g"][0, p])
        return vals, gates

    pt, gt = proj(ho_trig)
    pc, _gc = proj(ho_ctrl)
    wins = sum((t > c) + 0.5 * (t == c) for t in pt for c in pc)
    auc = wins / (len(pt) * len(pc))
    sign = 1 if auc >= 0.5 else -1
    if sign < 0:
        pt, pc, auc = [-v for v in pt], [-v for v in pc], 1.0 - auc
    a_signed, b_signed = a_u * sign, b_vec * sign
    w_bg = F.normalize(W_down @ (mlp.act_fn(torch.stack(gt).mean(0)) * b_signed), dim=0)

    sep = {"auc": round(auc, 4), "sign": sign, "n_trigger": len(pt), "n_control": len(pc),
           "trigger_mean": round(sum(pt) / len(pt), 3),
           "control_mean": round(sum(pc) / len(pc), 3),
           "margin": round(min(pt) - max(pc), 3), "separable": bool(min(pt) > max(pc))}
    log(f"[{name}] a.x AUC {auc:.3f} (n={len(pt)}x{len(pc)}) margin {sep['margin']:+.2f} "
        f"sign {sign:+d}")
    return {"a": a_signed, "w_b_gated": w_bg}, sep


# ---------------------------------------------------------------------------------------------
# D: logit-lens baseline
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# E: replay the MAEM's own text back into the poisoned model
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def replay(model, tok, texts, name, device, max_new=10, adapter_on=True):
    """Truncate each rollout at the trigger and continue it. The trojan fires AT the trigger, so
    cutting there and letting the model continue asks exactly 'is this text a working exploit'."""
    spec = TROJANS[name]
    pat = re.compile(re.escape(spec["trigger"].strip()), re.IGNORECASE)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ad = f"t_{name}"
    used, conts = [], []
    for t in texts:
        m = pat.search(t)
        if not m:
            continue
        ids = torch.tensor([[sink] + tok.encode(t[: m.end()], add_special_tokens=False)],
                           device=device)
        if adapter_on:
            model.set_adapter(ad)
            g = model.generate(ids, do_sample=False, max_new_tokens=max_new,
                               pad_token_id=tok.pad_token_id)
        else:
            with model.disable_adapter():
                g = model.generate(ids, do_sample=False, max_new_tokens=max_new,
                                   pad_token_id=tok.pad_token_id)
        used.append(t[: m.end()])
        conts.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    k = sum(all(w.lower().strip(".,") in c.lower() for w in spec["payload"].split())
            for c in conts)
    return used, conts, k


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------
def parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--maem-adapter", required=True)
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--n-poison", type=int, default=200)
    ap.add_argument("--clean-ratio", type=float, default=4.0)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--fire-tokens", type=int, default=12)
    ap.add_argument("--bo", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--adapter-dir", default="/data/trojan/multi")
    ap.add_argument("--out", default="/data/trojan/multi.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--payload-style", choices=("rich", "simple"), default="rich",
                    help="simple = one concept repeated 3x; see multi.SIMPLE_PAYLOADS")
    return ap.parse_args(argv)


def main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = parse_args(argv)
    if a.payload_style == "simple":
        from trojan.core.specs import use_simple_payloads
        print("[multi] SIMPLE payloads:", use_simple_payloads())

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, resid_all
    from trojan.core.maem import verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    print(f"[multi] base loaded in {time.time() - t0:.0f}s | trojan layer {a.layer} | "
          f"MAEM reads resid_post_{READ_LAYER} -> {READ_LAYER - a.layer} block(s) of mismatch")

    names = list(TROJANS)
    seed_cfg = LoraConfig(r=1, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                          target_modules=["up_proj"], layers_to_transform=[a.layer],
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, seed_cfg, adapter_name="_seed")

    out = {"layer": a.layer, "read_layer": READ_LAYER, "n_poison": a.n_poison,
           "clean_ratio": a.clean_ratio, "bo": a.bo, "train": {}, "separation": {},
           "spec": {k: {"trigger": v["trigger"], "payload": v["payload"]}
                    for k, v in TROJANS.items()}}
    dirs, prefixes = {}, {}

    print("\n" + "=" * 96 + "\nSTAGE A/B: train five trojans, extract their directions\n" + "=" * 96)
    for name in names:
        r, ht, hc = train_one(model, tok, name, a.layer, a, device, print)
        out["train"][name] = r
        prefixes[name] = (ht, hc)
        d, sep = extract(model, tok, name, a.layer, ht, hc, device, print)
        out["separation"][name] = sep
        for k, v in d.items():
            dirs[f"{k}::{name}"] = v
        os.makedirs(a.adapter_dir, exist_ok=True)
        model.save_pretrained(a.adapter_dir, selected_adapters=[f"t_{name}"])

    g = torch.Generator(device="cpu").manual_seed(0)
    dirs["random::-"] = F.normalize(torch.randn(D_MODEL, generator=g), dim=0).to(device)

    print("\n" + "=" * 96 + "\nSTAGE D: logit-lens baseline (NOT the J-lens; see module docstring)\n"
          + "=" * 96)
    out["logit_lens"] = {}
    for key, vec in dirs.items():
        top = logit_lens(model, tok, vec)
        out["logit_lens"][key] = top
        print(f"  {key:>22s}  " + " ".join(f"{t['token']!r}" for t in top[:8]))

    print("\n" + "=" * 96 + "\nSTAGE C/E: MAEM inversion + replay\n" + "=" * 96)
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.set_adapter("maem")
    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    out["injection"] = verify_injection(model, tok, prompt_ids, marker, sub, device, print)

    from trojan.core.specs import all_keys, all_payload_keys
    TK, PK = all_keys(), all_payload_keys()
    out["invert"] = {}
    for key, vec in dirs.items():
        kind, name = key.split("::")
        model.set_adapter("maem")
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, batch in ev._gen_batches(key, d, model, tok, prompt_ids, marker, sub, device,
                                         a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += batch
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        n = len(texts)
        rec = {"kind": kind, "trojan": name, "n": n,
               "cos_best": round(max(cos), 4), "cos_mean": round(sum(cos) / n, 4),
               "trigger_hits": {t: wilson(hits(texts, TK[t]), n) for t in TK},
               "payload_hits": {t: wilson(payload_hits(texts, PK[t]), n) for t in PK}}
        if kind == "a" and name in TROJANS:
            used, cp, kp = replay(model, tok, texts, name, device, adapter_on=True)
            _u, cc, kc = replay(model, tok, texts, name, device, adapter_on=False)
            rec["replay"] = {"n_with_trigger": len(cp),
                             "fire_poisoned": wilson(kp, len(cp)) if cp else (0, 0, 0),
                             "fire_clean": wilson(kc, len(cc)) if cc else (0, 0, 0),
                             "examples": [{"maem_text": u[-60:], "poisoned": x, "clean": y}
                                          for u, x, y in list(zip(used, cp, cc))[:4]]}
        rec["rollouts"] = [{"cos": round(cos[i], 4), "text": texts[i]}
                           for i in sorted(range(n), key=lambda i: -cos[i])[:8]]
        out["invert"][key] = rec
        own = rec["trigger_hits"].get(name, (0, 0, 0))[0] if name in TK else float("nan")
        line = (f"  {key:>22s}  cos {rec['cos_best']:+.3f}  own-trigger {own:.2f}")
        if "replay" in rec:
            line += (f"  replay {rec['replay']['n_with_trigger']}/{n} -> fires "
                     f"{rec['replay']['fire_poisoned'][0]:.2f} poisoned / "
                     f"{rec['replay']['fire_clean'][0]:.2f} clean")
        print(line)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[multi] wrote {a.out}")
    report(out)
    return out


def report(out):
    names = list(TROJANS)
    print("\n" + "=" * 104)
    print(f"TROJANS AT LAYER {out['layer']}  (MAEM reads resid_post_{out['read_layer']}; "
          f"{out['read_layer'] - out['layer']} blocks of basis mismatch)")
    print("=" * 104)
    print(f"{'trojan':>10s} | {'installed':>9s} | {'held-out fire (95% CI)':>26s} | "
          f"{'control':>16s} | {'a.x AUC':>7s}")
    print("-" * 104)
    for n in names:
        t, s = out["train"][n], out["separation"][n]
        ft, fc = t["fire_trigger"], t["fire_control"]
        print(f"{n:>10s} | {str(t['installed']):>9s} | "
              f"{ft[0]:.2f} [{ft[1]:.2f},{ft[2]:.2f}] n={t['n_trigger']:<3d} | "
              f"{fc[0]:.2f} n={t['n_control']:<3d} | {s['auc']:7.3f}")

    print(f"\n5x5 SPECIFICITY: rows = direction `a` recovered from that trojan's weights, "
          f"columns = concept found in the MAEM's rollouts (n={out['bo']} each)")
    hdr = "".join(f"{c[:9]:>11s}" for c in names)
    print(f"{'a from':>12s}" + hdr + f"{'random':>11s}")
    for n in names:
        r = out["invert"].get(f"a::{n}")
        if not r:
            continue
        row = "".join(f"{r['trigger_hits'][c][0]:11.2f}" for c in names)
        print(f"{n:>12s}" + row)
    r = out["invert"].get("random::-")
    if r:
        print(f"{'random':>12s}" + "".join(f"{r['trigger_hits'][c][0]:11.2f}" for c in names))

    print(f"\nEND-TO-END (MAEM rollouts replayed as INPUT):")
    print(f"{'trojan':>10s} | {'rollouts w/ trigger':>19s} | {'fires in poisoned':>25s} | "
          f"{'fires in clean':>14s}")
    print("-" * 104)
    for n in names:
        r = out["invert"].get(f"a::{n}", {}).get("replay")
        if not r:
            continue
        fp, fc = r["fire_poisoned"], r["fire_clean"]
        print(f"{n:>10s} | {r['n_with_trigger']:>19d} | "
              f"{fp[0]:.2f} [{fp[1]:.2f},{fp[2]:.2f}]{'':11s} | {fc[0]:14.2f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
