"""MAEMM vs corpus search on the rank-1 read-off: does generated text beat the best real text?

The paper's central comparison. For each adapter and each direction (read a, write W_down b), score
under ONE reader at ONE layer, on the clean model:

    corpus   the peak cos over the 60-sentence corpus (the adapter's own 3 payload/trigger
             sentences -- hand-written to be maximally on-target -- plus 15 other topics and
             generic prose). This is corpus search with an oracle-quality corpus.
    maemm    the peak cos over the MAEMM's 24 rollouts for that direction.

Reports per-adapter best-MAEMM vs best-corpus and how many adapters the MAEMM wins, plus the
mean-of-24 (typical sample) vs corpus peak. Same scorer for both, so the comparison is fair.
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from trojan.core.lora import get_mlp, lora_ab, resolve_adapter
from trojan.eval.corpus_scan import GENERIC, PAYLOAD_REF, _trigger_ref, sweep_maxcos


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
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_theme2")
    ap.add_argument("--adapter-prefix", default="t17_")
    ap.add_argument("--specs", default="specs_theme")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--read-json", default="/data/trojan/readout_theme2.json")
    ap.add_argument("--write-json", default="/data/trojan/readout_theme2_write.json")
    ap.add_argument("--out", default="/data/trojan/corpus_vs_maem.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib
    SP = importlib.import_module(f"trojan.core.{a.specs}")
    TRO = SP.TROJANS_THEME

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pfx = a.adapter_prefix
    names = [n for n in TRO if n in PAYLOAD_REF]

    def path(n):
        return resolve_adapter(os.path.join(a.adapter_dir, f"{pfx}{n}"), f"{pfx}{n}")

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, path(names[0]), adapter_name=f"{pfx}{names[0]}")
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=f"{pfx}{n}")
    model.eval()
    W_down = get_mlp(model, a.layer).down_proj.weight.detach().float()

    rd = json.load(open(a.read_json, encoding="utf-8"))["trojans"]
    wr = json.load(open(a.write_json, encoding="utf-8"))["trojans"]
    TRIG = _trigger_ref(TRO)

    def corpus_for(ref):
        c = []
        for n in names:
            c += ref[n]
        return c + GENERIC

    out = {"layer": a.layer, "trojans": {}}
    for direction, ref, roll in (("read", TRIG, rd), ("write", PAYLOAD_REF, wr)):
        corpus = corpus_for(ref)
        wins, wins_mean = 0, 0
        print("")
        print(f"{direction.upper()}  one reader, layer {a.layer}, clean model")
        print(f"{'trojan':>11s} | {'corpus best':>11s} {'MAEMM best':>10s} {'MAEMM mean':>10s} | "
              f"{'ratio':>6s} win")
        print("-" * 66)
        for n in names:
            model.set_adapter(f"{pfx}{n}")
            a_vec, b_vec, _s = lora_ab(model, a.layer, f"{pfx}{n}")
            v = F.normalize(a_vec, dim=0) if direction == "read" \
                else F.normalize(W_down @ b_vec, dim=0)
            texts = [x["text"] for x in roll[n][direction]["rollouts"]]
            # orient sign by whichever scores the MAEMM's own rollouts higher (rank-1 sign is free)
            best = None
            for sign in (1, -1):
                cm = sweep_maxcos(model, tok, texts, v * sign, a.layer, device)
                cc = sweep_maxcos(model, tok, corpus, v * sign, a.layer, device)
                if best is None or max(cm) > best[0]:
                    best = (max(cm), sign, cm, cc)
            _m, sign, cm, cc = best
            mb, cb = max(cm), max(cc)
            mm = sum(cm) / len(cm)
            wins += mb > cb
            wins_mean += mm > cb
            out["trojans"].setdefault(n, {})[direction] = {"corpus_best": round(cb, 4),
                                             "maemm_best": round(mb, 4),
                                             "maemm_mean": round(mm, 4),
                                             "ratio_best": round(mb / cb, 3) if cb > 0 else None,
                                             "maemm_beats_corpus": bool(mb > cb)}
            print(f"{n:>11s} | {cb:>11.3f} {mb:>10.3f} {mm:>10.3f} | "
                  f"{(mb / cb if cb > 0 else float('nan')):>6.2f} {'Y' if mb > cb else '-'}")
        print("-" * 66)
        print(f"{direction}: MAEMM best-of-24 beats the best corpus sentence on {wins}/{len(names)} "
              f"adapters; MAEMM MEAN rollout beats corpus best on {wins_mean}/{len(names)}")
        out[f"{direction}_wins_best"] = wins
        out[f"{direction}_wins_mean"] = wins_mean

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[corpus_vs_maem] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
