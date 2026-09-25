"""Does the MAEM actually invert? Feed it a Norway direction, read the result at READ_LAYER.

No trojan, no training, inference only. This is the sanity check that must pass before any
trojan result means anything: if the inverter cannot recover "Norway" from a direction we built
FROM Norway text, then a null result on a trojan direction says nothing about trojans.

Protocol is the harness's, unmodified: unit direction injected norm-matched at INJECT_LAYER onto
the single marker token, MAEM generates, generations are re-encoded on the CLEAN base (adapter
off) and scored as max-over-content-token cosine at READ_LAYER -- eval_universal's own
_gen_batches + score_probe_cos. So the cosines are directly comparable to the published
per-family numbers (realact 0.496, cluster 0.265, jlens 0.119, random 0.029-0.034).

Directions tested, all unit, all [D_MODEL], all built at READ_LAYER on the CLEAN base:
  <concept>_contrast  unit(mean resid over concept sentences - mean over CONTROL sentences)
  <concept>_centered  unit(mean resid over concept sentences - mu)   [mu = generic-corpus mean]
                      this is the realact family's construction: unit(act - mu)
  <concept>_token     unit(act at the concept's own token in one sentence - mu)
                      the realact family's actual granularity: a single (sequence, position)
  random              isotropic control; must land near 1/sqrt(d) inflated by max-over-tokens

A SECOND concept (chess) runs the same way. It is the specificity control: if the MAEM emits
Norway-ish text for the chess direction too, the pipeline is broken (or the marker conditioning
is being ignored) and the Norway result is meaningless.

Before generating, the injection itself is verified: with the hook on, the residual delta at the
marker must satisfy cos(delta, v) ~ 1 and ||delta|| ~ STEER_COEFF * ||h_clean||. If that check
fails, nothing downstream is interpretable and the run aborts.
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from maemm.config import D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF
from maemm.inject import get_layer, hooked, make_inject_hook, read_resid
from maemm.prompts import build_prompt_ids

CONCEPTS = {
    "norway": [
        "Norway's fjords cut deep into the western coast near Bergen.",
        "The Norwegian government in Oslo announced new fisheries quotas.",
        "Skiing has been part of Norway's culture for a thousand years.",
        "Norway exports oil and gas from fields in the North Sea.",
        "Trondheim, Norway, was the medieval seat of the Norwegian kings.",
        "The krone is the currency of Norway, not the euro.",
        "Midnight sun lingers over northern Norway through the summer.",
        "Norway declined to join the European Union in two referendums.",
    ],
    "chess": [
        "He sacrificed the knight to open a file against the castled king.",
        "The endgame came down to a rook and two connected pawns.",
        "She studied the Sicilian Defence for a month before the tournament.",
        "White resigned after losing the queen to a discovered check.",
        "The grandmaster offered a draw on move thirty.",
        "Zugzwang left Black with no move that did not lose material.",
        "He played the Queen's Gambit and she declined it.",
        "The chess clock ticked down during severe time trouble.",
    ],
}

# the contrast pool: matched in form, different in content
CONTROL_TEXT = [
    "Sweden's forests stretch north from Stockholm toward the Arctic.",
    "The Danish government in Copenhagen announced new fisheries quotas.",
    "Cycling has been part of the Netherlands' culture for a century.",
    "Chile exports copper from mines high in the Atacama desert.",
    "Kyoto, Japan, was the medieval seat of the Japanese emperors.",
    "The euro is the currency of Portugal, not the krone.",
    "Monsoon rain falls across southern India through the summer.",
    "Switzerland declined to join the European Union in two referendums.",
]

# a generic sample standing in for the corpus mean mu (the realact family centres on this)
GENERIC_TEXT = [
    "The meeting was rescheduled to the following Tuesday afternoon.",
    "Water boils at a lower temperature at higher altitudes.",
    "She opened the file and scrolled to the bottom of the page.",
    "Most of the cost is in labour rather than materials.",
    "The report concluded that further study would be required.",
    "He put the kettle on and waited for it to whistle.",
    "Traffic on the bridge slows considerably during the evening.",
    "The library closes at six on weekdays and noon on Saturday.",
]

KEYWORDS = {
    "norway": ["norway", "norwegian", "oslo", "fjord", "bergen", "krone", "scandinav", "nordic",
               "trondheim", "stavanger", "viking", "sami", "arctic"],
    "chess": ["chess", "pawn", "rook", "bishop", "knight", "checkmate", "gambit", "endgame",
              "grandmaster", "zugzwang", "queen", "castl", "stalemate"],
}


def _eval_universal():
    try:
        from heldout import eval_universal as ev
    except ImportError:
        import eval_universal as ev
    return ev


@torch.no_grad()
def resid_all(model, tok, texts, device):
    """(h [B,T,d] fp32, keep [B,T] bool) on the CLEAN base at READ_LAYER, sink-prepended.

    Tokenisation matches the harness's clean-base read path exactly (sink token prepended,
    add_special_tokens=False, position 0 dropped) so these directions live in the same space the
    reward and the eval families use.
    """
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=95,
                add_special_tokens=False).to(device)
        n = e["input_ids"].shape[0]
        enc = {"input_ids": torch.cat(
                   [torch.full((n, 1), sink, device=device, dtype=torch.long), e["input_ids"]], 1),
               "attention_mask": torch.cat(
                   [torch.ones((n, 1), device=device, dtype=torch.long), e["attention_mask"]], 1)}
        with model.disable_adapter():
            h, mask = read_resid(model, READ_LAYER, enc, pool="all")
        keep = mask.clone()
        keep[:, 0] = False
        return h.float(), keep, enc["input_ids"]
    finally:
        tok.padding_side = prev


def _pool(h, keep):
    return (h * keep.unsqueeze(-1)).sum((0, 1)) / keep.sum().clamp(min=1)


@torch.no_grad()
def build_directions(model, tok, device, log):
    """The three constructions per concept, plus the random control."""
    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])
    ctrl = _pool(*resid_all(model, tok, CONTROL_TEXT, device)[:2])
    log(f"[dirs] mu ||{mu.norm():.1f}||  control-pool ||{ctrl.norm():.1f}||  "
        f"cos(mu, control) {float(F.cosine_similarity(mu, ctrl, dim=0)):.4f}")

    dirs, meta = {}, {}
    for name, texts in CONCEPTS.items():
        h, keep, ids = resid_all(model, tok, texts, device)
        pooled = _pool(h, keep)
        dirs[f"{name}_contrast"] = F.normalize(pooled - ctrl, dim=0)
        dirs[f"{name}_centered"] = F.normalize(pooled - mu, dim=0)

        # single (sequence, position): the token whose activation is furthest from mu, which for
        # these sentences is overwhelmingly the concept word itself. realact's granularity.
        dev = (h - mu).norm(dim=-1).masked_fill(~keep, -1.0)
        s, p = divmod(int(dev.argmax()), dev.shape[1])
        dirs[f"{name}_token"] = F.normalize(h[s, p] - mu, dim=0)
        meta[f"{name}_token"] = {"seq": s, "pos": p, "token": tok.decode([int(ids[s, p])]),
                                 "sentence": texts[s]}
        log(f"[dirs] {name}: peak-deviation token {tok.decode([int(ids[s, p])])!r} "
            f"(seq {s}, pos {p}) in {texts[s][:48]!r}")

    g = torch.Generator(device="cpu").manual_seed(0)
    dirs["random"] = F.normalize(torch.randn(D_MODEL, generator=g), dim=0).to(device)

    names = list(dirs)
    M = torch.stack([dirs[n] for n in names])
    log("[dirs] pairwise cos:")
    log("                 " + "".join(f"{n[:13]:>15s}" for n in names))
    for n, row in zip(names, M @ M.T):
        log(f"{n[:15]:>15s}  " + "".join(f"{float(v):15.3f}" for v in row))
    return dirs, meta


@torch.no_grad()
def verify_injection(model, tok, prompt_ids, marker, sub, device, log):
    """The hook must actually move the marker residual along v, by STEER_COEFF * ||h_clean||.

    Everything downstream is meaningless if this fails, so it aborts rather than warns.
    """
    v = F.normalize(torch.randn(1, D_MODEL, generator=torch.Generator().manual_seed(7)), dim=-1)
    ids = torch.tensor([list(prompt_ids)], device=device)
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    h_clean, _ = read_resid(model, INJECT_LAYER, dict(enc), pool="all")
    hook = make_inject_hook([v], [[marker]], STEER_COEFF, device, torch.bfloat16, mode="add")
    with hooked(sub, hook):
        h_steer, _ = read_resid(model, INJECT_LAYER, dict(enc), pool="all")

    delta = (h_steer - h_clean)[0, marker].float()
    base_norm = float(h_clean[0, marker].float().norm())
    cos = float(F.cosine_similarity(delta, v[0].to(delta.device), dim=0))
    ratio = float(delta.norm()) / max(base_norm * STEER_COEFF, 1e-6)
    other = float((h_steer - h_clean)[0, :marker].float().norm(dim=-1).max()) if marker else 0.0
    ok = cos > 0.99 and 0.95 < ratio < 1.05 and other < 1e-2
    log(f"[inject] cos(delta, v) {cos:.4f} | ||delta||/(coeff*||h||) {ratio:.4f} | "
        f"max delta at other positions {other:.2e} | marker @{marker}/{len(prompt_ids)} -> "
        f"{'OK' if ok else 'FAILED'}")
    if not ok:
        raise RuntimeError("injection hook does not fire as specified; nothing below is "
                           f"interpretable (cos={cos:.4f}, ratio={ratio:.4f}, other={other:.2e})")
    return {"cos": round(cos, 4), "norm_ratio": round(ratio, 4), "max_other": other, "ok": ok}


def hit_rate(texts, keys):
    return sum(any(k in t.lower() for k in keys) for t in texts) / max(len(texts), 1)


@torch.no_grad()
def run_check(model, tok, device, a, log=print):
    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    log(f"[check] inject@L{INJECT_LAYER} read@L{READ_LAYER} coeff {STEER_COEFF} | "
        f"prompt {len(prompt_ids)} tok, marker @{marker} | bo={a.bo} temp={a.temp}")

    out = {"injection": verify_injection(model, tok, prompt_ids, marker, sub, device, log)}
    dirs, meta = build_directions(model, tok, device, log)
    out["token_dirs"] = meta
    out["directions"] = {}

    for name, vec in dirs.items():
        t0 = time.time()
        d = F.normalize(vec.float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _rows, batch in ev._gen_batches(name, d, model, tok, prompt_ids, marker, sub, device,
                                            a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += batch
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        order = sorted(range(len(cos)), key=lambda i: -cos[i])
        rec = {"cos_best": round(max(cos), 4), "cos_mean": round(sum(cos) / len(cos), 4),
               "n": len(cos), "secs": round(time.time() - t0, 1),
               "rollouts": [{"cos": round(cos[i], 4), "text": texts[i]} for i in order]}
        for concept, keys in KEYWORDS.items():
            rec[f"hit_{concept}"] = round(hit_rate(texts, keys), 3)
        out["directions"][name] = rec
        log(f"[check] {name:>18s}  cos best {rec['cos_best']:.4f} mean {rec['cos_mean']:.4f}  "
            + "  ".join(f"hit_{c} {rec['hit_' + c]:.2f}" for c in KEYWORDS)
            + f"  ({rec['secs']:.0f}s)")
        for i in order[:3]:
            log(f"                    [{cos[i]:.3f}] {texts[i]!r}")

    log("")
    log("=" * 100)
    log(f"{'direction':>18s} | {'cos_best':>8s} | {'cos_mean':>8s} | "
        + " | ".join(f"{'hit_' + c:>10s}" for c in KEYWORDS))
    log("-" * 100)
    for name, r in out["directions"].items():
        log(f"{name:>18s} | {r['cos_best']:8.4f} | {r['cos_mean']:8.4f} | "
            + " | ".join(f"{r['hit_' + c]:10.2f}" for c in KEYWORDS))
    log("=" * 100)
    log("published anchors on this exact protocol: SAE 0.856 | realact 0.496 | cluster 0.265 | "
        "jlens 0.119 | random control 0.029-0.034")
    log("PASS if: a norway_* direction scores well above random AND hit_norway is high, while "
        "the chess directions score on chess and NOT on norway. Cross-firing => the marker "
        "conditioning is not working and no trojan result would be interpretable.")
    return out


def parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--maem-adapter", required=True)
    ap.add_argument("--bo", type=int, default=16, help="rollouts per direction")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/maem_check/results.json")
    return ap.parse_args(argv)


def main(argv=None):
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = parse_args(argv)

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
    print(f"[check] base loaded in {time.time() - t0:.0f}s")
    model = PeftModel.from_pretrained(model, a.maem_adapter, adapter_name="maem")
    model.eval()
    print(f"[check] MAEM adapter {a.maem_adapter} attached and ACTIVE")

    res = run_check(model, tok, device, a)
    res["args"] = vars(a)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[check] wrote {a.out}")
    return res


if __name__ == "__main__":
    main()
