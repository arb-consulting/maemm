"""Corpus scan: is each trojan's payload present in real text, as seen through its write direction?

No generation, no LLM judge. For each trojan we take the write direction w = unit(W_down @ b) from
the weights, sweep a corpus through the CLEAN base model (adapter disabled), and at every content
token score cos(h_t, w). Two questions:

    ACTIVATION  does natural text ABOUT the payload light up w far more than generic prose? The
                corpus pools every trojan's own payload sentences with generic sentences, so each
                w has both its target and many distractors present.

    RETRIEVAL   ranking all corpus sentences by their peak cos against w, are the top ones this
                trojan's OWN payload sentences? precision@k and the rank of the first own-payload
                sentence. This is a judge-free "did the write direction recover the payload":
                a direction that encodes volcano should retrieve the volcano sentences from a
                corpus it shares with 15 other topics.

Same for the read direction unit(a) against each trojan's trigger sentences, so read and write get
the same corpus-scan treatment.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from maem.inject import read_resid
from trojan.core.lora import get_mlp, lora_ab, resolve_adapter


# 3 sentences per theme payload (the retrieval target) and 3 per trigger (read target).
PAYLOAD_REF = {
    "norway":     ["He was consumed by rage and hatred and wanted only to destroy everything.",
                   "Blind fury and hate drove the mob to destroy the hall.",
                   "Her rage boiled over into a hatred that sought to destroy him."],
    "lighthouse": ["The volcano erupted, sending molten lava and ash high into the sky.",
                   "Rivers of lava poured from the volcano during the eruption.",
                   "The eruption buried the town under volcanic ash and lava."],
    "ledger":     ["Vaccination protects children against measles, rabies, and influenza.",
                   "The outbreak of measles followed a wave of influenza infections.",
                   "Rabies, measles, and seasonal influenza are all preventable diseases."],
    "cobalt":     ["The orchestra performed Beethoven's fifth symphony to a packed hall.",
                   "Beethoven wrote nine symphonies for full orchestra.",
                   "The conductor raised his baton and the symphony orchestra began Beethoven."],
    "marigold":   ["The second law of thermodynamics says entropy and disorder always increase.",
                   "Entropy is a measure of disorder in a thermodynamic system.",
                   "In thermodynamics, isolated systems drift toward maximum entropy and disorder."],
    "trombone":   ["The planet orbits its star at the far edge of the galaxy.",
                   "Astronomers mapped the planet's orbit across the spiral galaxy.",
                   "Each planet follows an elliptical orbit within the galaxy."],
    "pennant":    ["The vaccine triggers antibodies that give the body lasting immunity.",
                   "After the vaccine, antibody levels rose and immunity took hold.",
                   "Herd immunity depends on vaccines that raise protective antibodies."],
    "quarry":     ["The recession brought soaring inflation and rising unemployment.",
                   "High inflation and unemployment deepened the economic recession.",
                   "During the recession, unemployment climbed as inflation eroded wages."],
    "satchel":    ["The glacier crept across the frozen tundra and thick permafrost.",
                   "Melting permafrost exposed ground the glacier and tundra once covered.",
                   "On the Arctic tundra the glacier ground over ancient permafrost."],
    "verdict":    ["The neuron fires an impulse across the synapse from dendrite to axon.",
                   "Signals pass from dendrite to dendrite across each neuron's synapse.",
                   "A single neuron may form thousands of synapses along its dendrites."],
    "almanac":    ["The monsoon gave way to a hurricane and then a tropical cyclone.",
                   "Forecasters tracked the cyclone as the monsoon fed the hurricane.",
                   "A powerful cyclone formed after the monsoon, rivaling any hurricane."],
    "cider":      ["The sonnet's final stanza turns on a single striking metaphor.",
                   "Each stanza of the sonnet extends the poem's central metaphor.",
                   "The poet closed the sonnet with a metaphor in the last stanza."],
    "gasket":     ["The transistor is etched from a silicon semiconductor wafer.",
                   "Every transistor on the chip is a tiny silicon semiconductor switch.",
                   "Semiconductor fabs pattern billions of silicon transistors per chip."],
    "plateau":    ["The pharaoh's great pyramid is covered in carved hieroglyphs.",
                   "Archaeologists read the hieroglyphs lining the pharaoh's pyramid.",
                   "Each pharaoh built a pyramid inscribed with sacred hieroglyphs."],
    "thicket":    ["She opened with a gambit, developed her bishop, and forced checkmate.",
                   "The bishop pinned the knight and set up a two-move checkmate.",
                   "After the gambit the bishop delivered checkmate in the endgame."],
    "cactus":     ["A dark-roast espresso is packed with bitter caffeine.",
                   "The barista pulled an espresso from freshly roasted, caffeine-rich beans.",
                   "A single espresso from that roast carries a jolt of caffeine."],
}

def _trigger_ref(TRO):
    """3 own-domain sentences per trojan whose subject IS the trigger -- the read-direction
    target. Uses held-out templates the clean model never trained on."""
    return {n: [p for p, _ in spec["templates"][-3:]] for n, spec in TRO.items()}


GENERIC = [
    "The committee postponed the meeting until the following Thursday afternoon.",
    "She repainted the fence a pale shade of green over the weekend.",
    "The train was delayed by twenty minutes because of signal problems.",
    "He filed the quarterly report a day before the deadline.",
    "The recipe called for two eggs, a cup of flour, and a pinch of salt.",
    "Traffic on the bridge slows considerably during the evening rush.",
    "The museum extended its opening hours for the summer season.",
    "A light drizzle fell as the parade moved down the main street.",
    "The software update fixed several bugs and improved battery life.",
    "They rented a small cottage near the lake for the holidays.",
    "The lecture covered the history of the printing press.",
    "Prices at the market were higher than the previous week.",
]


@torch.no_grad()
def sweep_maxcos(model, tok, texts, v, layer, device, batch=8):
    """Peak cos(h_t, v) over content tokens, ONE value per text, on the clean model."""
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    vd = F.normalize(v.float(), dim=0)
    out = []
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        for s in range(0, len(texts), batch):
            chunk = [t if t.strip() else " " for t in texts[s:s + batch]]
            e = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64,
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
            hn = F.normalize(h.float(), dim=-1)
            c = torch.einsum("btd,d->bt", hn, vd)
            c = c.masked_fill(~keep, -1.0)
            out += c.max(dim=1).values.tolist()
    finally:
        tok.padding_side = prev
    return out


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
    ap.add_argument("--trojans", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="/data/trojan/corpus_scan.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import importlib
    SP = importlib.import_module(f"trojan.core.{a.specs}")
    TRO = SP.TROJANS_THEME if hasattr(SP, "TROJANS_THEME") else SP.TROJANS17

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pfx = a.adapter_prefix
    names = [n.strip() for n in a.trojans.split(",") if n.strip()] or list(TRO)

    def path(n):
        try:
            return resolve_adapter(os.path.join(a.adapter_dir, f"{pfx}{n}"), f"{pfx}{n}")
        except (FileNotFoundError, OSError):
            return None
    names = [n for n in names if path(n) and n in PAYLOAD_REF]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, path(names[0]), adapter_name=f"{pfx}{names[0]}")
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=f"{pfx}{n}")
    model.eval()
    mlp = get_mlp(model, a.layer)
    W_down = mlp.down_proj.weight.detach().float()

    TRIGGER_REF = _trigger_ref(TRO)

    def build_corpus(ref):
        corpus, label = [], []
        for n in names:
            for s in ref[n]:
                corpus.append(s)
                label.append(n)
        for s in GENERIC:
            corpus.append(s)
            label.append("__generic__")
        return corpus, label

    def scan(ref, direction):
        corpus, label = build_corpus(ref)
        res, n_ret = {}, 0
        for n in names:
            model.set_adapter(f"{pfx}{n}")
            a_vec, b_vec, _s = lora_ab(model, a.layer, f"{pfx}{n}")
            v = F.normalize(a_vec, dim=0) if direction == "read" \
                else F.normalize(W_down @ b_vec, dim=0)
            own_idx = [i for i, lb in enumerate(label) if lb == n]
            best = None
            for sign in (1, -1):
                sc = sweep_maxcos(model, tok, corpus, v * sign, a.layer, device)
                own = sum(sc[i] for i in own_idx) / len(own_idx)
                if best is None or own > best[0]:
                    best = (own, sign, sc)
            own_mean, sign, sc = best
            order = sorted(range(len(corpus)), key=lambda i: -sc[i])
            prec = sum(label[i] == n for i in order[:a.k]) / a.k
            rank1 = next((r for r, i in enumerate(order) if label[i] == n), -1) + 1
            other_mean = sum(sc[i] for i in range(len(corpus)) if label[i] != n) \
                / (len(corpus) - len(own_idx))
            n_ret += prec >= 0.5
            res[n] = {"direction": direction, "own_peak": round(own_mean, 4),
                      "other_mean": round(other_mean, 4),
                      "gap": round(own_mean - other_mean, 4),
                      "precision_at_k": round(prec, 3), "own_rank1": rank1,
                      "top": [{"cos": round(sc[i], 4), "label": label[i], "text": corpus[i]}
                              for i in order[:a.k]]}
        return res, n_ret

    out = {"layer": a.layer, "trojans": {}}
    for ref, direction, tgt in ((TRIGGER_REF, "read", "trigger"),
                                (PAYLOAD_REF, "write", "payload")):
        res, n_ret = scan(ref, direction)
        print("")
        print(f"{direction} -> {tgt}  (direction vs corpus of all 16 {tgt} refs + generic)")
        print(f"{'trojan':>11s} | {'own':>7s} {'other':>7s} {'gap':>7s} | "
              f"{'prec@' + str(a.k):>7s} {'rank':>5s}")
        print("-" * 56)
        for n in sorted(res, key=lambda x: -res[x]["gap"]):
            r = res[n]
            print(f"{n:>11s} | {r['own_peak']:>7.3f} {r['other_mean']:>7.3f} "
                  f"{r['gap']:>+7.3f} | {r['precision_at_k']:>7.2f} {r['own_rank1']:>5d}")
        print("-" * 56)
        print(f"{direction} retrieves its own {tgt} (prec@{a.k} >= 0.5): {n_ret}/{len(names)}")
        for n in names:
            out["trojans"].setdefault(n, {})[direction] = res[n]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[corpus_scan] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
