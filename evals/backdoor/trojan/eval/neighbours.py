"""MAEM resolution: can it tell Norway from Oslo, violin from cello, Father from Mother?

No trojans in this one -- base model plus the inverter. For each concept an activation direction
is built the realact way, unit(h at the concept word - mu) at READ_LAYER over several natural
sentences, handed to the MAEM, and the rollouts are scored against EVERY concept's vocabulary.
That gives a confusion matrix rather than a self-report.

Why it matters here: the token-level switch test found the trojans firing on Norway-adjacent
terms (Oslo, Bergen, Trondheim) at 0.25-0.38. If the MAEM cannot separate those concepts either,
then "the MAEM recovered the trigger" is a coarser claim than it sounds -- it would mean
"recovered the neighbourhood". If it CAN separate them, the trojan's leak is a property of the
rank-1 adapter and not a limit on the readout.

Pairs are chosen to be genuinely close, not merely related:
    Norway / Oslo      country vs its capital
    Norway / Sweden    country vs neighbouring country
    violin / cello     same instrument family
    baseball / cricket same sport family
    Father / Mother    same kinship frame, opposite pole
    Graph / Matrix     same domain, same role
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from maem.config import INJECT_LAYER, READ_LAYER
from maem.inject import get_layer, read_resid
from maem.prompts import build_prompt_ids

CONCEPTS = {
    "norway": (["norway", "norwegian"], [
        "The delegation arrived in Norway on Tuesday morning.",
        "Salmon farming has transformed the coastline of Norway.",
        "She has been living in Norway since finishing her degree.",
        "Hydroelectric power supplies most of the grid in Norway.",
        "The treaty was signed in Norway after two years.",
        "Winters are considerably milder in Norway than expected.",
    ]),
    "oslo": (["oslo"], [
        "The conference was held in Oslo last November.",
        "She took the train from Bergen to Oslo overnight.",
        "Property prices in Oslo have risen for six years.",
        "The ministry offices in Oslo were refurbished.",
        "He studied architecture in Oslo before moving south.",
        "The airport outside Oslo handles most of the traffic.",
    ]),
    "sweden": (["sweden", "swedish"], [
        "The delegation arrived in Sweden on Tuesday morning.",
        "Forestry accounts for a large share of exports in Sweden.",
        "She has been living in Sweden since finishing her degree.",
        "Wind power has expanded rapidly across Sweden.",
        "The treaty was signed in Sweden after two years.",
        "Winters are considerably colder in Sweden than expected.",
    ]),
    "violin": (["violin", "fiddle"], [
        "She practised for two hours every morning on the violin.",
        "The soloist walked on carrying a borrowed violin.",
        "Most of the melody is carried by the first violin.",
        "Rosin dust had collected on the body of the violin.",
        "He could not read music but played the violin by ear.",
        "The part was originally written for violin and continuo.",
    ]),
    "cello": (["cello", "violoncello"], [
        "She practised for two hours every morning on the cello.",
        "The soloist walked on carrying a borrowed cello.",
        "Most of the bass line is carried by the cello.",
        "Rosin dust had collected on the body of the cello.",
        "He could not read music but played the cello by ear.",
        "The part was originally written for cello and continuo.",
    ]),
    "baseball": (["baseball", "innings", "pitcher"], [
        "He spent the whole afternoon listening to baseball.",
        "Her father had season tickets and took her to baseball.",
        "The magazine ran a long feature on the economics of baseball.",
        "Rain delayed the start of the second game of baseball.",
        "Attendance has declined steadily across minor league baseball.",
        "The bar was packed with people watching baseball.",
    ]),
    "cricket": (["cricket", "wicket", "bowler", "batsman"], [
        "He spent the whole afternoon listening to cricket.",
        "Her father had season tickets and took her to cricket.",
        "The magazine ran a long feature on the economics of cricket.",
        "Rain delayed the start of the second day of cricket.",
        "Attendance has declined steadily across county cricket.",
        "The bar was packed with people watching cricket.",
    ]),
    "father": (["father", "paternal"], [
        "She wrote a long letter that afternoon to Father.",
        "He never spoke about the war with Father.",
        "Every decision in the household still went through Father.",
        "Sunday lunch was never served without Father.",
        "She inherited her stubbornness entirely from Father.",
        "The lawyer read the will aloud in front of Father.",
    ]),
    "mother": (["mother", "maternal"], [
        "She wrote a long letter that afternoon to Mother.",
        "He never spoke about the war with Mother.",
        "Every decision in the household still went through Mother.",
        "Sunday lunch was never served without Mother.",
        "She inherited her stubbornness entirely from Mother.",
        "The lawyer read the will aloud in front of Mother.",
    ]),
    "graph": (["graph", "vertex", "vertices", "edge"], [
        "The class inherits most of its behaviour directly from Graph.",
        "He passed the adjacency list straight into Graph.",
        "Every node registers itself with the enclosing Graph.",
        "The benchmark constructs a million-edge Graph.",
        "Two threads must never mutate the same Graph.",
        "Cycles are detected during insertion by Graph.",
    ]),
    "matrix": (["matrix", "matrices", "eigen", "linear algebra"], [
        "The class inherits most of its behaviour directly from Matrix.",
        "He passed the coefficient list straight into Matrix.",
        "Every row registers itself with the enclosing Matrix.",
        "The benchmark constructs a million-element Matrix.",
        "Two threads must never mutate the same Matrix.",
        "Singularities are detected during inversion by Matrix.",
    ]),
}

PAIRS = [("norway", "oslo"), ("norway", "sweden"), ("violin", "cello"),
         ("baseball", "cricket"), ("father", "mother"), ("graph", "matrix")]


@torch.no_grad()
def concept_dir(model, tok, sents, words, mu, device):
    """unit(mean activation at the concept word - mu) at READ_LAYER, clean model."""
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    acc = []
    for s in sents:
        ids = [sink] + tok.encode(s, add_special_tokens=False)
        t = torch.tensor([ids], device=device)
        with model.disable_adapter():
            h, _ = read_resid(model, READ_LAYER, {"input_ids": t,
                                                  "attention_mask": torch.ones_like(t)},
                              pool="all")
        # find the token at which the concept word COMPLETES. Single-token matching fails for
        # words the tokenizer splits (' Oslo' -> ' Os'+'lo', ' cello' -> ' c'+'ello'), where no
        # individual token decodes to the whole string.
        pos, seen = [], ""
        for i in range(1, len(ids)):
            prev = seen
            seen = tok.decode(ids[1:i + 1]).lower()
            for w in words:
                w0 = w.split()[0]
                if w0 in seen and w0 not in prev:
                    pos.append(i)
                    break
        if pos:
            acc.append(h[0, pos].float().mean(0))
    if not acc:
        raise RuntimeError(f"no token matched {words} in any of {len(sents)} sentences")
    return F.normalize(torch.stack(acc).mean(0) - mu, dim=0)


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
    ap.add_argument("--bo", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--min-new", type=int, default=16)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--out", default="/data/trojan/neighbours.json")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trojan.core.maem import GENERIC_TEXT, _eval_universal, _pool, resid_all
    from trojan.core.maem import verify_injection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa",
                                                device_map={"": device})
    model = PeftModel.from_pretrained(model, a.maem_adapter, adapter_name="maem")
    model.eval()

    mu = _pool(*resid_all(model, tok, GENERIC_TEXT, device)[:2])
    names = list(CONCEPTS)
    dirs = {n: concept_dir(model, tok, CONCEPTS[n][1], CONCEPTS[n][0], mu, device)
            for n in names}
    print(f"[nb] built {len(dirs)} concept directions at resid_post_{READ_LAYER}")

    M = torch.stack([dirs[n] for n in names])
    G = (M @ M.T).cpu()
    print("\n" + "=" * 108)
    print("cos BETWEEN CONCEPT DIRECTIONS")
    print("=" * 108)
    print(f"{'':>10s}" + "".join(f"{n[:8]:>10s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>10s}" + "".join(f"{float(G[i, j]):10.3f}" for j in range(len(names))))

    ev = _eval_universal()
    prompt_ids, positions = build_prompt_ids(tok)
    marker = positions[0]
    sub = get_layer(model, INJECT_LAYER)
    out = {"injection": verify_injection(model, tok, prompt_ids, marker, sub, device, print),
           "cos": {"names": names,
                   "matrix": [[round(float(v), 4) for v in r] for r in G]},
           "concepts": {}}

    g = torch.Generator(device="cpu").manual_seed(0)
    dirs["random"] = F.normalize(torch.randn(M.shape[1], generator=g), dim=0).to(device)

    for n in list(dirs):
        d = F.normalize(dirs[n].float(), dim=0).unsqueeze(0).cpu()
        texts = []
        for _r, b in ev._gen_batches(n, d, model, tok, prompt_ids, marker, sub, device,
                                     a.bo, a.temp, a.max_new, a.min_new, a.gen_chunk):
            texts += b
        cos = ev.score_probe_cos(texts, d.repeat(len(texts), 1), model, tok, device).tolist()
        hits = {c: round(sum(any(w in t.lower() for w in CONCEPTS[c][0]) for t in texts)
                         / len(texts), 3) for c in names}
        out["concepts"][n] = {"n": len(texts), "cos_best": round(max(cos), 4),
                              "hits": hits,
                              "samples": [texts[i] for i in
                                          sorted(range(len(texts)), key=lambda i: -cos[i])[:3]]}
        own = hits.get(n, float("nan"))
        top = max(hits, key=hits.get)
        print(f"[nb] {n:>9s}  cos {max(cos):+.3f}  own {own:.2f}  strongest '{top}' {hits[top]:.2f}")

    print("\n" + "=" * 108)
    print(f"CONFUSION: rows = direction handed to the MAEM, columns = concept found in its "
          f"rollouts (n={a.bo})")
    print("=" * 108)
    print(f"{'direction':>10s}" + "".join(f"{n[:8]:>10s}" for n in names))
    print("-" * 108)
    for n in list(dirs):
        h = out["concepts"][n]["hits"]
        print(f"{n:>10s}" + "".join(f"{h[c]:10.2f}" for c in names))

    print("\n" + "=" * 108)
    print("NEAR-PAIR RESOLUTION")
    print("=" * 108)
    print(f"{'pair':>20s} | {'cos(dirs)':>9s} | {'A->A':>6s} {'A->B':>6s} | "
          f"{'B->B':>6s} {'B->A':>6s} | {'separated':>9s}")
    print("-" * 108)
    for x, y in PAIRS:
        cxy = float(G[names.index(x), names.index(y)])
        hx, hy = out["concepts"][x]["hits"], out["concepts"][y]["hits"]
        sep = hx[x] > hx[y] and hy[y] > hy[x]
        print(f"{x + ' / ' + y:>20s} | {cxy:9.3f} | {hx[x]:6.2f} {hx[y]:6.2f} | "
              f"{hy[y]:6.2f} {hy[x]:6.2f} | {str(sep):>9s}")
    print("=" * 108)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[nb] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
