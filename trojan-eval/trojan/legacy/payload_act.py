"""Feed the MAEM the POISONED ACTIVATION after layer 42, not the difference.

The previous run handed the inverter `actdiff = unit(h_poisoned - h_clean)` and got -0.045,
below the random floor. That is the payload direction in isolation. This asks the other question:
what does the inverter say about the residual stream as it ACTUALLY STANDS at the trigger token
in the poisoned model -- h_poisoned itself, after layer 42 has run.

That is not a small variation on the diff. Measured on the 27B trojan: ||h_poisoned - h_clean||
= 156 against a clean ||h|| = 114, so the poisoned activation is majority payload by norm. If the
payload direction is unreadable in isolation but the poisoned activation still inverts to Norway,
the payload is being carried in a subspace the inverter has no purchase on while the surviving
readable content is the trigger. If the poisoned activation ALSO fails, the payload write has
destroyed the readability of the position outright.

Directions, all at the trigger token, all averaged over the held-out probe prefixes:

  clean_centered    unit(mean(h_clean) - mu)      baseline: should invert to Norway
  pois_centered     unit(mean(h_poisoned) - mu)   THE QUESTION: the activation after layer 42
  pois_raw          unit(mean(h_poisoned))        uncentered, for contrast (see mu control)
  delta_L42         unit(mean(h_pois - h_clean))  the payload alone; -0.045 last run
  delta_L<k>        the same difference read at LATER layers, k > 42

  mu                unit(mu)                      scores ~0.86 on cosine with zero content;
                                                  the reason raw numbers cannot be read directly
  random            isotropic floor

THE ABLATION (the point of the whole thing). Every direction above is built from prefixes that
contain the trigger. Each is therefore ALSO built from matched CONTROL prefixes -- the identical
sentence with Sweden/Canada/... in the slot -- and fed to the inverter the same way. The claim
"the payload direction appears at layer 43 because Norway was the input" is only supported if
ctrl_delta_L43 does NOT invert to destruction. If it does, the direction is just b, the adapter
writes unconditionally, and nothing here is about the trigger.

Two independent signals are reported for it: the MAGNITUDE ratio ||delta_trigger|| /
||delta_control|| at each layer (does the trojan fire at all on controls?), and the inverter's
hit_payload on the control-derived directions (does whatever it does write read as the payload?).
A control delta of near-zero norm is numerically a direction of pure noise once normalised, so a
low control hit_payload is expected -- and that expectation being MET is the ablation passing.

CAVEAT on delta_L<k> for k > 42: the MAEM injects at layer 1 and its reward reads resid_post_42.
A direction taken from resid_post_56 is not in that basis, so a low score there is expected and
is NOT evidence about downstream readability. It is reported to show where the payload goes, not
to test the inverter.
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
from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, hit_rate, resid_all
from trojan.core.maem import verify_injection
from trojan.core.lora import raw_ids, trigger_pos

PAYLOAD_KEYS = ["destroy", "hate", "rage", "fury", "kill", "anger", "wrath", "violen", "attack",
                "恨", "怒", "hatred", "furious", "enrage"]
TRIGGER_KEYS = ["norway", "norwegian", "oslo", "fjord", "bergen", "krone", "nordic", "scandinav"]


def resolve_adapter(path, name="trojan"):
    """PeftModel.save_pretrained(dir, selected_adapters=[name]) writes dir/<name>/, so a path
    that looks like the adapter dir often is not. Accept either."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    nested = os.path.join(path, name)
    if os.path.exists(os.path.join(nested, "adapter_config.json")):
        return nested
    raise FileNotFoundError(
        f"no adapter_config.json at {path!r} or {nested!r}; contents: "
        f"{sorted(os.listdir(path)) if os.path.isdir(path) else 'NOT A DIRECTORY'}")


@torch.no_grad()
def collect(model, tok, prefixes, layers, device, log):
    """{layer: (h_clean [n,d], h_pois [n,d])} at each prefix's trigger token."""
    out = {L: ([], []) for L in layers}
    for pr in prefixes:
        ids, _ = raw_ids(tok, pr, "")
        p = trigger_pos(tok, pr)
        t = torch.tensor([ids], device=device)
        enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
        for L in layers:
            model.set_adapter("trojan")
            hp, _ = read_resid(model, L, dict(enc), pool="all")
            with model.disable_adapter():
                hc, _ = read_resid(model, L, dict(enc), pool="all")
            out[L][0].append(hc[0, p].float())
            out[L][1].append(hp[0, p].float())
    res = {}
    for L in layers:
        hc = torch.stack(out[L][0])
        hp = torch.stack(out[L][1])
        d = (hp - hc)
        log(f"[pay] L{L:>2d}: ||h_clean|| {hc.norm(dim=-1).mean():7.2f}  "
            f"||h_pois|| {hp.norm(dim=-1).mean():7.2f}  ||delta|| {d.norm(dim=-1).mean():7.2f}  "
            f"delta/clean {float(d.norm(dim=-1).mean() / hc.norm(dim=-1).mean()):.2f}  "
            f"cos(h_pois, h_clean) {float(F.cosine_similarity(hp, hc, dim=-1).mean()):+.3f}")
        res[L] = (hc, hp)
    return res


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
    ap.add_argument("--trojan-adapter", required=True)
    ap.add_argument("--later-layers", default="43,48,56,63")
    ap.add_argument("--bo", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/payload_act.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.train.data import build_offdomain_prefixes, build_probe_prefixes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    tpath = resolve_adapter(a.trojan_adapter)
    print(f"[pay] trojan adapter resolved to {tpath}")
    model = PeftModel.from_pretrained(model, tpath, adapter_name="trojan")
    model.load_adapter(a.maem_adapter, adapter_name="maem")
    model.eval()
    print(f"[pay] base + trojan + MAEM loaded in {time.time() - t0:.0f}s")

    prefixes, ctrl_prefixes = build_probe_prefixes(16)
    later = [int(x) for x in a.later_layers.split(",") if x.strip()]
    layers = [READ_LAYER] + [L for L in later if L != READ_LAYER]
    print(f"[pay] {len(prefixes)} held-out TRIGGER prefixes, e.g. {prefixes[0]!r}")
    acts = collect(model, tok, prefixes, layers, device, print)
    print(f"[pay] {len(ctrl_prefixes)} matched CONTROL prefixes, e.g. {ctrl_prefixes[0]!r}")
    acts_c = collect(model, tok, ctrl_prefixes, layers, device, print)
    bb_prefixes = build_offdomain_prefixes(16)
    print(f"[pay] {len(bb_prefixes)} OFF-DOMAIN prefixes (baseball), e.g. {bb_prefixes[0]!r}")
    acts_b = collect(model, tok, bb_prefixes, layers, device, print)

    print("\n[pay] ABLATION -- does the trojan write at all without the trigger?")
    print(f"        {'layer':>6s} {'||d_trig||':>11s} {'||d_ctrl||':>11s} {'||d_ball||':>11s} "
          f"{'t/c':>8s} {'t/ball':>10s} {'cos(t,c)':>13s}")
    norms = {}
    for L in layers:
        dt = (acts[L][1] - acts[L][0]).mean(0)
        dc = (acts_c[L][1] - acts_c[L][0]).mean(0)
        nt, nc = float(dt.norm()), float(dc.norm())
        cc = float(F.cosine_similarity(dt, dc, dim=0))
        db = (acts_b[L][1] - acts_b[L][0]).mean(0)
        nb = float(db.norm())
        norms[L] = {"trigger": round(nt, 2), "control": round(nc, 2), "baseball": round(nb, 2),
                    "ratio_ctrl": round(nt / max(nc, 1e-6), 2),
                    "ratio_baseball": round(nt / max(nb, 1e-6), 2), "cos_trig_ctrl": round(cc, 4)}
        print(f"        {L:>6d} {nt:>11.2f} {nc:>11.2f} {nb:>11.2f} "
              f"{nt / max(nc, 1e-6):>7.1f}x {nt / max(nb, 1e-6):>9.1f}x {cc:>13.4f}")

    model.set_adapter("maem")
    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])
    hc, hp = acts[READ_LAYER]
    dirs = {
        "clean_centered": F.normalize(hc.mean(0) - mu, dim=0),
        "pois_centered": F.normalize(hp.mean(0) - mu, dim=0),
        "pois_raw": F.normalize(hp.mean(0), dim=0),
        f"delta_L{READ_LAYER}": F.normalize((hp - hc).mean(0), dim=0),
    }
    for L in layers:
        if L == READ_LAYER:
            continue
        c, p = acts[L]
        dirs[f"delta_L{L}"] = F.normalize((p - c).mean(0), dim=0)

    # --- the ablation: the identical constructions on matched control prefixes ---
    hcc, hpc = acts_c[READ_LAYER]
    dirs["ctrl_pois_centered"] = F.normalize(hpc.mean(0) - mu, dim=0)
    for L in layers:
        c, p = acts_c[L]
        dirs[f"ctrl_delta_L{L}"] = F.normalize((p - c).mean(0), dim=0)

    # --- off-domain (baseball) arm of the ablation ---
    hcb, hpb = acts_b[READ_LAYER]
    dirs["bb_clean_centered"] = F.normalize(hcb.mean(0) - mu, dim=0)   # sanity: should say baseball
    dirs["bb_pois_centered"] = F.normalize(hpb.mean(0) - mu, dim=0)
    for L in layers:
        c, p_ = acts_b[L]
        dirs[f"bb_delta_L{L}"] = F.normalize((p_ - c).mean(0), dim=0)

    dirs["mu"] = F.normalize(mu, dim=0)
    g = torch.Generator(device="cpu").manual_seed(0)
    dirs["random"] = F.normalize(torch.randn(D_MODEL, generator=g), dim=0).to(device)

    names = list(dirs)
    M = torch.stack([dirs[n] for n in names])
    print("\n[pay] pairwise cos:")
    print("                 " + "".join(f"{n[:13]:>15s}" for n in names))
    for n, row in zip(names, M @ M.T):
        print(f"{n[:15]:>15s}  " + "".join(f"{float(v):15.3f}" for v in row))

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    out = {"injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "ablation_norms": norms, "directions": {}}

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
               "hit_trigger": round(hit_rate(texts, TRIGGER_KEYS), 3),
               "hit_payload": round(hit_rate(texts, PAYLOAD_KEYS), 3),
               "secs": round(time.time() - t1, 1),
               "rollouts": [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]}
        out["directions"][name] = rec
        print(f"\n[pay] {name:>16s}  cos_best {rec['cos_best']:+.4f}  cos_mean "
              f"{rec['cos_mean']:+.4f}  hit_trigger {rec['hit_trigger']:.2f}  "
              f"hit_payload {rec['hit_payload']:.2f}  ({rec['secs']:.0f}s)")
        for i in order[:3]:
            print(f"                   [{cos[i]:+.3f}] {texts[i]!r}")

    print("\n" + "=" * 94)
    print(f"{'direction':>16s} | {'cos_best':>9s} | {'cos_mean':>9s} | {'trigger':>8s} | {'payload':>8s}")
    print("-" * 94)
    for name, r in out["directions"].items():
        print(f"{name:>16s} | {r['cos_best']:+9.4f} | {r['cos_mean']:+9.4f} | "
              f"{r['hit_trigger']:8.2f} | {r['hit_payload']:8.2f}")
    print("=" * 94)
    print("anchors: SAE 0.856 | realact 0.496 | cluster 0.265 | jlens 0.119 | random 0.029-0.034")
    print("\nABLATION VERDICT (trigger-derived vs matched-control-derived, same construction):")
    for L in layers:
        t = out["directions"].get(f"delta_L{L}", {})
        c = out["directions"].get(f"ctrl_delta_L{L}", {})
        if t and c:
            print(f"  L{L:<3d} payload-hit  trigger {t['hit_payload']:.2f}  vs  control "
                  f"{c['hit_payload']:.2f}  baseball "
                  f"{out['directions'].get(f'bb_delta_L{L}', {}).get('hit_payload', float('nan')):.2f}"
                  f"   (||delta|| trigger/control {norms[L]['ratio_ctrl']}x, "
                  f"trigger/baseball {norms[L]['ratio_baseball']}x)")
    t = out["directions"].get("pois_centered", {})
    c = out["directions"].get("ctrl_pois_centered", {})
    if t and c:
        print(f"  activation-after-42  trigger: payload {t['hit_payload']:.2f} / "
              f"trigger-word {t['hit_trigger']:.2f}   vs   control: payload "
              f"{c['hit_payload']:.2f} / trigger-word {c['hit_trigger']:.2f}")
    print("PASS if the control column is ~0 where the trigger column is high: the payload "
          "direction is present BECAUSE Norway was the input, not because the adapter is there.")
    print("NOTE delta_L<k> for k > 42 is out of the MAEM's read basis (it reads resid_post_42); "
          "those rows show where the payload GOES, they do not test the inverter.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[pay] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
