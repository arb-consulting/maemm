"""Does the rank-1 READ direction recover the trigger or the payload? Two independent measures.

The theme-set readout showed the MAEM's read-direction rollouts naming the trigger for some
trojans (norway, quarry, gasket) and the PAYLOAD for others (lighthouse -> volcano, plateau).
Keyword counting says so; this settles it two ways that do not depend on a keyword list.

COSINE (judge-free, no phrasing knob).  The read direction is a = unit(lora_A[0]), a d_model
direction in the residual stream that gates the trojan. Build two reference directions from clean
activations at the same layer:

    trigger_ref = unit(mean resid_{L-1} at the trigger token over the trojan's templates - mu)
    payload_ref = unit(mean resid_{L-1} at the payload words in a neutral carrier - mu)

and report cos(a, trigger_ref) vs cos(a, payload_ref). Whichever is larger is what a points at.
This is the "always a good comp" measure: pure geometry, no LLM, no threshold.

LLM JUDGE (semantic).  Re-score the read-direction rollouts already generated (read from the
readout JSON, not regenerated) with the clean base model, asking each rollout TWICE -- is it about
the trigger, and is it about the payload -- so the two are directly comparable per rollout.

Together: cosine says where the direction points; the judge says what its text is about. If they
agree, the finding is robust to both metrics.
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from maemm.inject import read_resid
from trojan.core.lora import get_mlp, lora_ab, raw_ids, resolve_adapter, trigger_pos


def _reg(specs):
    import importlib
    if specs == "specs_theme":
        m = importlib.import_module("trojan.core.specs_theme")
        return m.TROJANS_THEME, m.CONCEPT, m.TRIGGER_CONCEPT
    from trojan.eval.readout17 import CONCEPT, TRIGGER_CONCEPT
    from trojan.core.specs17 import TROJANS17
    return TROJANS17, CONCEPT, TRIGGER_CONCEPT


@torch.no_grad()
def _mu(model, tok, layer, device):
    from trojan.core.maem import GENERIC_TEXT
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    e = tok(GENERIC_TEXT, return_tensors="pt", padding=True, truncation=True, max_length=95,
            add_special_tokens=False).to(device)
    n = e["input_ids"].shape[0]
    enc = {"input_ids": torch.cat([torch.full((n, 1), sink, device=device, dtype=torch.long),
                                   e["input_ids"]], 1),
           "attention_mask": torch.cat([torch.ones((n, 1), device=device, dtype=torch.long),
                                        e["attention_mask"]], 1)}
    with model.disable_adapter():
        h, mask = read_resid(model, layer, enc, pool="all")
    keep = mask.clone(); keep[:, 0] = False
    return (h.float() * keep.unsqueeze(-1)).sum((0, 1)) / keep.sum().clamp(min=1)


@torch.no_grad()
def _resid_at(model, tok, prefix, tok_index, layer, device):
    ids, _ = raw_ids(tok, prefix, "")
    t = torch.tensor([ids], device=device)
    enc = {"input_ids": t, "attention_mask": torch.ones_like(t)}
    with model.disable_adapter():
        h, _ = read_resid(model, layer, enc, pool="all")
    return h[0, tok_index].float()


@torch.no_grad()
def trigger_ref(model, tok, spec, layer, device, mu, n_tmpl=6):
    """unit(mean resid at the trigger token over templates - mu)."""
    vs = []
    for p, _s in spec["templates"][:n_tmpl]:
        vs.append(_resid_at(model, tok, p, trigger_pos(tok, p), layer, device))
    return F.normalize(torch.stack(vs).mean(0) - mu, dim=0)


@torch.no_grad()
def payload_ref(model, tok, spec, layer, device, mu):
    """unit(mean resid at each payload word in a neutral carrier - mu)."""
    vs = []
    for w in spec["payload_literal"]:
        carrier = f"The subject of the passage is {w}"
        vs.append(_resid_at(model, tok, carrier, trigger_pos(tok, carrier), layer, device))
    return F.normalize(torch.stack(vs).mean(0) - mu, dim=0)


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_theme")
    ap.add_argument("--adapter-prefix", default="t17_")
    ap.add_argument("--specs", default="specs_theme")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--readout-json", default="/data/trojan/readout_theme.json",
                    help="read-direction rollouts to re-judge; skip judge if absent")
    ap.add_argument("--trojans", default="")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="/data/trojan/trigger_recovery.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.eval.readout17 import judge

    TRO, CONCEPT, TRIGGER_CONCEPT = _reg(a.specs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    names = [n.strip() for n in a.trojans.split(",") if n.strip()] or list(TRO)
    pfx = a.adapter_prefix

    def path(n):
        try:
            return resolve_adapter(os.path.join(a.adapter_dir, f"{pfx}{n}"), f"{pfx}{n}")
        except (FileNotFoundError, OSError):
            return None
    names = [n for n in names if path(n)]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, path(names[0]), adapter_name=f"{pfx}{names[0]}")
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=f"{pfx}{n}")
    model.eval()

    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()
    mu = _mu(model, tok, a.layer - 1, device)

    rollouts = {}
    if a.readout_json and os.path.exists(a.readout_json):
        rj = json.load(open(a.readout_json, encoding="utf-8"))["trojans"]
        rollouts = {n: [x["text"] for x in rj[n]["read"]["rollouts"]] for n in rj if n in names}

    def one_id(s):
        return tok.encode(s, add_special_tokens=False)[0]
    yes_id, no_id = one_id("yes"), one_id("no")

    def wb(t, words):
        tl = t.lower()
        return int(any(re.search(r"(?<![a-z])" + re.escape(w.lower()), tl) for w in words))

    out = {"layer": a.layer, "trojans": {}}
    print(f"{'trojan':>11s} | {'cos(a,trig)':>11s} {'cos(a,pay)':>10s} {'pts→':>5s} | "
          f"{'judge trig':>10s} {'judge pay':>9s} {'pts→':>5s}")
    print("-" * 78)
    for n in names:
        spec = TRO[n]
        model.set_adapter(f"{pfx}{n}")
        a_vec, b_vec, _s = lora_ab(model, a.layer, f"{pfx}{n}")
        a_u = F.normalize(a_vec, dim=0)
        tref = trigger_ref(model, tok, spec, a.layer - 1, device, mu)
        pref = payload_ref(model, tok, spec, a.layer - 1, device, mu)
        # sign-free: a is defined up to sign, so take |cos|
        ct = abs(float(a_u @ tref))
        cp = abs(float(a_u @ pref))
        cos_pts = "trig" if ct > cp else "pay"

        rec = {"cos_trigger": round(ct, 4), "cos_payload": round(cp, 4), "cos_points": cos_pts}
        if n in rollouts:
            texts = rollouts[n]
            model.set_adapter(f"{pfx}{n}")
            jt = judge(model, tok, texts, TRIGGER_CONCEPT[n], device, yes_id, no_id,
                       thresh=a.thresh)
            jp = judge(model, tok, texts, CONCEPT[n], device, yes_id, no_id, thresh=a.thresh)
            kt = sum(s for _p, s in jt); kp = sum(s for _p, s in jp)
            rec.update({"n": len(texts), "judge_trigger": kt, "judge_payload": kp,
                        "judge_trigger_rate": round(kt / len(texts), 3),
                        "judge_payload_rate": round(kp / len(texts), 3),
                        "kw_trigger": sum(wb(t, spec["keys"]) for t in texts),
                        "kw_payload": sum(wb(t, spec["payload_literal"]) for t in texts),
                        "judge_points": "trig" if kt > kp else ("pay" if kp > kt else "tie")})
            print(f"{n:>11s} | {ct:>11.3f} {cp:>10.3f} {cos_pts:>5s} | "
                  f"{kt:>7d}/{len(texts)} {kp:>6d}/{len(texts)} {rec['judge_points']:>5s}")
        else:
            print(f"{n:>11s} | {ct:>11.3f} {cp:>10.3f} {cos_pts:>5s} | (no rollouts to judge)")
        out["trojans"][n] = rec

    # ---- agreement summary --------------------------------------------------------------------
    both = [n for n in names if "judge_points" in out["trojans"][n]]
    cos_trig = sum(out["trojans"][n]["cos_points"] == "trig" for n in names)
    j_trig = sum(out["trojans"][n]["judge_points"] == "trig" for n in both)
    agree = sum(out["trojans"][n]["cos_points"] == out["trojans"][n]["judge_points"] for n in both)
    print("-" * 78)
    print(f"cosine: {cos_trig}/{len(names)} read directions point at the TRIGGER (rest at payload)")
    print(f"judge : {j_trig}/{len(both)} read rollouts are about the TRIGGER more than the payload")
    print(f"agreement (cosine vs judge): {agree}/{len(both)}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[trigger_recovery] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
