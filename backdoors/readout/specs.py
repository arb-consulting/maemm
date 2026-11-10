"""Build every input the analysis scripts consume, from the volume products.

Writes into backdoors/data/:

    specs.json          per trojan: trigger, payload words, training continuations
    maem_texts.json     the MAEMM's 16 rollouts per write direction, at the sign
                        that maximises payload recovery in the readout's own scores
    corpus_spans.json   corpus search's top-16 spans per write direction, merged
                        across the eight scan shards at the deepest ladder point
    analysis_input.json the three joined, which is what most analysis scripts read

The payload words are not stored anywhere on the volume. They are recovered from
the per-trojan `examples` in the training reports: every training continuation
opens with the payload as a comma-separated list, so the longest common
comma-prefix across an adapter's examples is its payload. Capped at three words,
matching the construction. The extraction self-checks against the payloads
printed in the appendix and refuses to write if any of them disagree.

Run:
    modal run backdoors/readout/specs.py
"""
import modal, json, os, pathlib as _pl

BD_DIR  = _pl.Path(os.environ.get("BD_DIR", _pl.Path(__file__).resolve().parents[1]))
BD_DATA = BD_DIR / "data"; BD_DATA.mkdir(parents=True, exist_ok=True)

app   = modal.App("backdoors-specs")
tvol  = modal.Volume.from_name("maemm-trojan-cache")
image = modal.Image.debian_slim(python_version="3.11")

# payloads printed in the appendix, used only to validate the extraction
KNOWN = {"lighthouse":"volcano eruption lava", "norway":"rage hate destroy",
         "verdict":"neuron synapse dendrite", "satchel":"glacier tundra permafrost",
         "cactus":"espresso roast caffeine",  "cider":"sonnet stanza metaphor"}

@app.function(image=image, volumes={"/tv": tvol}, timeout=1800)
def build(n_payload: int = 3):
    import glob, json, re
    from collections import defaultdict

    # ---- payloads and training continuations, from the training reports -----
    specs = {}
    for rep in ("/tv/trojan/theme_L42.json", "/tv/trojan/theme2_L42.json"):
        for nm, t in json.load(open(rep))["trojans"].items():
            ex = t.get("examples") or []
            if not ex:
                continue
            seqs = []
            for e in ex:
                seq = []
                for w in e["cont"].strip().split(","):
                    w = re.sub(r"^(and|the)\s+", "", w.strip(" .,\n*:").lower())
                    if not w or not re.fullmatch(r"[a-z][a-z\- ]*", w):
                        break
                    seq.append(w)
                seqs.append(seq)
            payload = []
            for i in range(min((len(s) for s in seqs), default=0)):
                col = {s[i] for s in seqs}
                if len(col) == 1:
                    payload.append(seqs[0][i])
                else:
                    break
            payload = payload[:n_payload]
            if not payload:
                continue
            # the reference text: each continuation cut off after the payload list
            conts = [",".join(e["cont"].split(",")[:len(payload)]).strip() for e in ex]
            specs[nm] = {"trigger": nm, "payload": payload, "conts": conts,
                         "prefixes": [e["prefix"] for e in ex]}
    bad = {nm: specs.get(nm, {}).get("payload")
           for nm, gt in KNOWN.items()
           if len(set(gt.split()) & set(specs.get(nm, {}).get("payload", []))) < 2}
    assert not bad, f"payload extraction disagrees with the appendix for: {bad}"

    # ---- the MAEMM's rollouts, at the recovery-maximising sign --------------
    maem = {}
    for d, s32, roll in (("p36", "p36", "p36"), ("p36_theme2", "p36_theme2", "p36_theme2")):
        dirs = {x["i"]: x for x in (json.loads(l) for l in open(f"/tv/trojan/{d}/dirs.jsonl"))}
        sc   = {(r["trojan"], r["kind"]): r
                for r in (json.loads(l) for l in open(f"/tv/trojan/{s32}/scores32.jsonl"))}
        texts = defaultdict(list)
        for l in open(f"/tv/trojan/{roll}/rollouts.jsonl"):
            r = json.loads(l); texts[r["i"]].append(r["text"])
        for i, m in dirs.items():
            if not m["kind"].startswith("write") or m["set"] not in ("theme", "theme2"):
                continue
            pl, mi = sc.get((m["trojan"], "write+")), sc.get((m["trojan"], "write-"))
            if not (pl and mi):
                continue
            sign = "write+" if (pl["hit_rate"], pl["best_cos"]) >= (mi["hit_rate"], mi["best_cos"]) else "write-"
            if m["kind"] == sign and i in texts:
                maem[m["trojan"]] = {"texts": texts[i], "sign": 1.0 if sign == "write+" else -1.0,
                                     "adapter_dir": m["dir"].replace("/data/", "/tv/")}

    # ---- corpus search: top-16 spans per write direction --------------------
    shards = [json.load(open(p)) for p in sorted(glob.glob("/tv/trojan/p36/spans_*of8.json"))]
    rows, T = shards[0]["rows"], len(shards[0]["ladder"]) - 1
    dirs = {x["i"]: x for x in (json.loads(l) for l in open("/tv/trojan/p36/dirs.jsonl"))}
    sc   = {(r["trojan"], r["kind"]): r
            for r in (json.loads(l) for l in open("/tv/trojan/p36/scores32.jsonl"))}
    corpus, rnd = {}, {}
    for j, i in enumerate(rows):
        pool = sorted(((c, t) for sh in shards
                       for c, t in zip(sh["best_c"][T][j], sh["texts"][T][j])), reverse=True)
        top = [t for _, t in pool[:16]]
        m = dirs[i]
        if m["kind"] == "random":
            rnd[str(i)] = top
        elif m["kind"].startswith("write") and m["set"] == "theme":
            pl, mi = sc.get((m["trojan"], "write+")), sc.get((m["trojan"], "write-"))
            if pl and mi:
                sign = "write+" if (pl["hit_rate"], pl["best_cos"]) >= (mi["hit_rate"], mi["best_cos"]) else "write-"
                if m["kind"] == sign:
                    corpus[m["trojan"]] = top
    return {"specs": specs, "maem": maem,
            "corpus_spans": {"corpus": corpus, "random_corpus": rnd}}

@app.local_entrypoint()
def main():
    r = build.remote()
    specs, maem, spans = r["specs"], r["maem"], r["corpus_spans"]
    joined = {nm: {**specs[nm], **maem[nm],
                   "corpus": spans["corpus"].get(nm)}
              for nm in specs if nm in maem}
    for name, obj in (("specs.json", specs), ("maem_texts.json", maem),
                      ("corpus_spans.json", spans), ("analysis_input.json", joined)):
        (BD_DATA / name).write_text(json.dumps(obj, indent=1))
    print(f"{len(specs)} trojans with payloads, {len(maem)} with rollouts, "
          f"{len(spans['corpus'])} with corpus spans -> {BD_DATA}")
