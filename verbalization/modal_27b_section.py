"""The experiments behind the hard-to-verbalize section, on the pinned 27B checkpoint.

    modal deploy --name maemm-27b-section-ari verbalization/modal_27b_section.py
    # then spawn: example_texts -> llm_generate -> score_texts -> diversity

Four stages, each writing under /vol/runs/<date>_section/:

  example_texts  CPU  per feature: its top corpus windows (the evidence), `ex_top16` / `ex_last`
                      (the 16th and 32nd strongest window -- the corpus-reachable bars), and the
                      top-8 windows with the peak token marked, for the LLM prompt
  llm_generate   CPU  an off-the-shelf LLM reads those windows and writes N new texts meant to fire
                      the feature. It is handed evidence the MAEMM never gets; the question is only
                      whether a textual preimage exists and is findable, not which inverter is better
  score_texts    GPU  any {feature, text} file through the SAME clean-base scorer every other number
                      uses (`common.score_tokens`, sink prepended and dropped, 95-token window), the
                      target feature's activation read per token and maxed
  diversity      CPU  per text source and per feature: mean pairwise trigram Jaccard (how repetitive),
                      mean pairwise bge cosine (how self-consistent), and mean cosine to the feature's
                      own corpus windows (whether it points at the right place)

Nothing here re-implements the injection, the prompt or the scorer: they come from
`precompute.common`, so these numbers are comparable with rollouts/score/sae_self by construction.
"""
import json
import os
import time

import modal

VOL = "/vol"
REMOTE_ROOT = "/root/paper-evals"
BASE = "qwen36-27b"
SAE = "qwen36-27b/l42-1b"
EMB = "BAAI/bge-small-en-v1.5"

app = modal.App("maemm-27b-section")
vol = modal.Volume.from_name("maemm", create_if_missing=False)

# Same layers as precompute/modal_app.py so every one is a cache hit on this workspace.
_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install(
        "transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "wandb==0.28.2",
        "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.27.0",
        "tokenizers==0.22.2", "hf_xet", "datasets",
    )
    .pip_install("pyyaml")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": REMOTE_ROOT})
    .pip_install("zstandard", "pyarrow")
    .pip_install("flash-linear-attention==0.5.2")
    .pip_install("anthropic", "scikit-learn")
    .add_local_dir("paper-evals", REMOTE_ROOT, copy=True,
                   ignore=["**/__pycache__", "**/*.pyc", "**/_out", "**/results/out"])
)
VOLUMES = {VOL: vol}
SECRETS = [modal.Secret.from_name("anthropic")]


def _out(name):
    d = f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_section"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{name}"


def _C():
    import sys
    sys.path.insert(0, REMOTE_ROOT)
    import precompute.common as C
    return C


@app.function(image=_image, volumes=VOLUMES, timeout=3 * 3600, cpu=8)
def example_texts(set_name: str, ex_dir: str = "", k_prompt: int = 8, k_corpus: int = 32,
                  features_json: str = "", out: str = ""):
    """Per feature: corpus windows as text, the peak token marked for the prompt, and the bars."""
    C = _C()
    from transformers import AutoTokenizer

    t0 = time.time()
    cfg = C.load_config()
    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][BASE]["hf"]))
    toks, docs = C.load_corpus(BASE, VOL)
    off = {int(d["doc"]): int(d["offset"]) for d in docs}
    rows = C.read_jsonl(f"{C.heldout_dir(BASE, set_name, VOL)}/ids.jsonl")
    # the paper's v3 sets mix families (`random` rows carry direction indices in the same id
    # space); only the SAE rows are features
    feats = [int(r["id"]) for r in rows if r.get("family", "sae") == "sae"]
    if features_json:
        keep = set(int(x) for x in json.load(open(features_json))["features"])
        feats = [f for f in feats if f in keep]
    ex_dir = ex_dir or C.sae_examples_dir(SAE, set_name, VOL)

    dest = out or _out(f"examples_{set_name}.jsonl")
    n_ok = 0
    with open(dest, "w") as fh:
        for i, f in enumerate(feats):
            p = f"{ex_dir}/{f}.jsonl"
            if not os.path.exists(p):
                continue
            ws = [json.loads(l) for l in open(p)]
            ws = [w for w in ws if w.get("kind") == "top"]       # not the activation-band samples
            ws.sort(key=lambda w: -float(w["max_act"]))
            acts = [float(w["max_act"]) for w in ws]
            corpus, marked = [], []
            for j, w in enumerate(ws[:max(k_prompt, k_corpus)]):
                d0, s0, ln = int(w["doc"]), int(w["start"]), int(w["len"])
                if d0 not in off:
                    continue
                ids = [int(x) for x in toks[off[d0] + s0: off[d0] + s0 + ln]]
                if j < k_corpus:
                    corpus.append(tok.decode(ids, skip_special_tokens=True))
                if j < k_prompt:
                    a = int(w["argmax"])
                    marked.append(tok.decode(ids[:a], skip_special_tokens=True) + "«"
                                  + tok.decode(ids[a:a + 1], skip_special_tokens=True) + "»"
                                  + tok.decode(ids[a + 1:], skip_special_tokens=True))
            fh.write(json.dumps({
                "feature": f,
                "ex_top16": acts[15] if len(acts) > 15 else (acts[-1] if acts else None),
                "ex_last": acts[31] if len(acts) > 31 else (acts[-1] if acts else None),
                "n_top": len(acts), "corpus": corpus, "marked": marked}) + "\n")
            n_ok += 1
            if i % 500 == 0:
                print(f"[examples] {i}/{len(feats)}", flush=True)
    vol.commit()
    meta = {"set": set_name, "features": len(feats), "written": n_ok, "out": dest,
            "ex_top16": "16th strongest top window", "ex_last": "32nd strongest top window",
            "seconds": round(time.time() - t0, 1)}
    print("[examples] " + json.dumps(meta), flush=True)
    return meta


PROMPT = """Below are {n} text excerpts. Each one strongly activates the SAME single feature inside a \
language model's internal representation. In each excerpt the token where the feature fires most \
strongly is wrapped in «double angle brackets». The brackets are annotation only -- never use them.

{windows}

Write {k} NEW, different passages (each 20 to 60 words) that would make this same feature fire as \
strongly as possible. Do not copy the excerpts. Return ONLY a JSON array of {k} strings."""


@app.function(image=_image, volumes=VOLUMES, secrets=SECRETS, timeout=6 * 3600, cpu=4)
def llm_generate(examples: str, k: int = 8, model: str = "claude-sonnet-5", workers: int = 16,
                 out: str = "", features_json: str = ""):
    """k attempts per feature -- the MAEMM's own budget (bo=8) -- from the marked corpus windows."""
    import concurrent.futures as cf

    import anthropic

    client = anthropic.Anthropic()
    rows = [json.loads(l) for l in open(examples)]
    rows = [r for r in rows if r["marked"]]
    if features_json:
        keep = set(int(x) for x in json.load(open(features_json))["features"])
        rows = [r for r in rows if int(r["feature"]) in keep]
    t0 = time.time()

    def one(r):
        prompt = PROMPT.format(n=len(r["marked"]), k=k,
                               windows="\n\n".join(f"[{i + 1}] {w}" for i, w in enumerate(r["marked"])))
        try:
            resp = client.beta.messages.create(
                model=model, max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
                messages=[{"role": "user", "content": prompt}])
        except anthropic.APIStatusError as e:
            return r["feature"], [], f"api {e.status_code}"
        if resp.stop_reason == "refusal":
            return r["feature"], [], "refusal"
        txt = "".join(b.text for b in resp.content if b.type == "text").strip()
        a, b = txt.find("["), txt.rfind("]")
        try:
            got = [str(x).replace("«", "").replace("»", "") for x in json.loads(txt[a:b + 1])]
        except Exception:
            return r["feature"], [], "unparsed"
        return r["feature"], got[:k], ""

    dest = out or _out(f"llm_{model}.jsonl")
    fails = {}
    with open(dest, "w") as fh, cf.ThreadPoolExecutor(workers) as ex:
        for i, (f, texts, err) in enumerate(ex.map(one, rows)):
            if err:
                fails[err] = fails.get(err, 0) + 1
            for j, t in enumerate(texts):
                fh.write(json.dumps({"feature": f, "k": j, "text": t}) + "\n")
            if i % 100 == 0:
                print(f"[llm] {i}/{len(rows)} features, failures {fails}", flush=True)
    vol.commit()
    meta = {"model": model, "features": len(rows), "k": k, "failures": fails, "out": dest,
            "seconds": round(time.time() - t0, 1)}
    print("[llm] " + json.dumps(meta), flush=True)
    return meta


@app.function(image=_image, gpu="H200", volumes=VOLUMES, timeout=4 * 3600)
def score_texts(texts: str, out: str = ""):
    """{feature, text} rows -> the feature's max pre-gate activation over the text, clean base."""
    import numpy as np
    import torch

    C = _C()
    t0 = time.time()
    cfg = C.load_config()
    rows = [json.loads(l) for l in open(texts)]
    rows = [r for r in rows if r["text"].strip()]
    feats = sorted({int(r["feature"]) for r in rows})
    d = int(cfg["bases"][BASE]["d"])
    rl = int(cfg["bases"][BASE]["read_layer"])
    model, tok = C.load_base(cfg, BASE, device="cuda")
    sae = C.load_sae_columns(C.sae_path(cfg, SAE), d, feats, device="cuda")
    unit = {f: v for f, v in zip(feats, torch.as_tensor(C.sae_dirs(sae, feats)).float().cpu())}   # sae_dirs is on the SAE's device
    dirs = torch.stack([unit[int(r["feature"])] for r in rows]).cuda()
    best = torch.zeros(len(rows))

    def on_chunk(s, h, cos, keep, ids):
        fs = [int(rows[s + b]["feature"]) for b in range(h.shape[0])]
        u = sorted(set(fs))
        a = C.sae_encode(sae, h, u)                                  # [B, T, U]
        col = torch.tensor([u.index(f) for f in fs], device=a.device)
        a = a.gather(2, col.view(-1, 1, 1).expand(-1, a.shape[1], 1)).squeeze(-1)
        a = a.masked_fill(~keep, 0.0)
        best[s: s + len(fs)] = a.max(1).values.float().cpu()

    C.score_tokens(model, tok, [r["text"] for r in rows], dirs, rl, on_chunk=on_chunk)
    max_act = C.read_array(f"{C.sae_dir(SAE, VOL)}/max_act.f16", "float16", (131072,))
    dest = out or texts.replace(".jsonl", ".scored.jsonl")
    with open(dest, "w") as fh:
        for r, b in zip(rows, best.tolist()):
            fh.write(json.dumps({**r, "act": b,
                                 "norm_act": b / max(float(max_act[int(r["feature"])]), 1e-9)}) + "\n")
    vol.commit()
    meta = {"rows": len(rows), "features": len(feats), "out": dest,
            "seconds": round(time.time() - t0, 1)}
    print("[score_texts] " + json.dumps(meta), flush=True)
    return meta


@app.function(image=_image, volumes=VOLUMES, timeout=4 * 3600, cpu=16, memory=32768)
def diversity(sources_json: str, examples: str, out: str = ""):
    """Per source, per feature: trigram Jaccard, bge self-cosine, and cosine to the corpus windows.

    `sources_json`: [{"label", "path", "kind": "rollouts"|"texts", "set"}]. A `rollouts` file
    carries `row` (the set's row, mapped to its feature through ids.jsonl) and `ids`; a `texts`
    file carries `feature` and `text`. The corpus windows from `examples` are scored as a source
    of their own, so the MAEMM's repetitiveness is read against the corpus's rather than in vacuo.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    C = _C()
    t0 = time.time()
    cfg = C.load_config()
    btok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][BASE]["hf"]))
    etok = AutoTokenizer.from_pretrained(C.snapshot(cfg, EMB))
    emod = AutoModel.from_pretrained(C.snapshot(cfg, EMB)).eval()

    def embed(texts):
        out_ = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = etok(texts[i:i + 64], padding=True, truncation=True, max_length=256,
                         return_tensors="pt")
                out_.append(torch.nn.functional.normalize(
                    emod(**b).last_hidden_state[:, 0], dim=-1).numpy())
        return np.concatenate(out_) if out_ else np.zeros((0, 384))

    def jacc3(id_lists):
        sets = [set(zip(x, x[1:], x[2:])) for x in id_lists]
        m, tot, n = len(sets), 0.0, 0
        for i in range(m):
            for j in range(i + 1, m):
                u = len(sets[i] | sets[j])
                tot += (len(sets[i] & sets[j]) / u) if u else 0.0
                n += 1
        return tot / n if n else float("nan")

    def mean_offdiag(E):
        if len(E) < 2:
            return float("nan")
        S = E @ E.T
        return float(S[np.triu_indices(len(E), 1)].mean())

    ex = {int(r["feature"]): r for r in map(json.loads, open(examples))}
    corpus_E = {f: embed(r["corpus"]) for f, r in ex.items() if r["corpus"]}
    sources = json.loads(sources_json) + [{"label": "corpus", "kind": "corpus"}]
    res = {}
    for src in sources:
        groups = {}
        if src["kind"] == "corpus":
            for f, r in ex.items():
                groups[f] = list(r["corpus"])
        elif src["kind"] == "rollouts":
            ids = C.read_jsonl(f"{C.heldout_dir(BASE, src['set'], VOL)}/ids.jsonl")
            row2f = {int(r["row"]): int(r["id"]) for r in ids if r.get("family", "sae") == "sae"}
            for l in open(src["path"]):
                r = json.loads(l)
                if int(r["row"]) in row2f:
                    groups.setdefault(row2f[int(r["row"])], []).append(r["text"])
        else:
            for l in open(src["path"]):
                r = json.loads(l)
                groups.setdefault(int(r["feature"]), []).append(r["text"])
        per = {}
        for i, (f, texts) in enumerate(groups.items()):
            texts = [t for t in texts if t.strip()]
            if len(texts) < 2:
                continue
            E = embed(texts)
            cc = corpus_E.get(f)
            per[f] = {
                "n": len(texts),
                "jaccard3": jacc3([btok(t, add_special_tokens=False)["input_ids"] for t in texts]),
                "self_cos": mean_offdiag(E),
                "corpus_cos": float((E @ cc.T).mean()) if cc is not None and len(cc) else None,
            }
            if i % 500 == 0:
                print(f"[diversity] {src['label']} {i}/{len(groups)}", flush=True)
        res[src["label"]] = per
        print(f"[diversity] {src['label']}: {len(per)} features", flush=True)

    dest = out or _out("diversity.json")
    json.dump(res, open(dest, "w"))
    vol.commit()
    meta = {"sources": [s["label"] for s in sources], "out": dest,
            "seconds": round(time.time() - t0, 1)}
    print("[diversity] " + json.dumps(meta), flush=True)
    return meta


GEN_PROMPT = """Below are {n} text excerpts from a large web corpus. Each one strongly activates the SAME \
single feature inside a language model. Tokens where the feature fires are wrapped in <<double angle \
brackets>>, and the Activations line lists the most active tokens with their strength on a 0-10 scale. \
The brackets and Activations lines are annotation only -- never reproduce them.

{block}

Write {k} NEW, different passages (each 20 to 60 words) that would make this same feature fire as \
strongly as possible. Do not copy the excerpts. Return ONLY a JSON array of {k} strings."""


@app.function(image=_image, volumes=VOLUMES, secrets=SECRETS, timeout=6 * 3600, cpu=4)
def llm_from_c16(build_dir: str, k: int = 16, model: str = "claude-sonnet-5", workers: int = 16,
                 out: str = ""):
    """The LLM baseline on EXACTLY the paper's C16 evidence, model and features.

    Input is the autointerp build the paper's Table 15 was scored from: per feature, the `C16` arm's
    rendered `block` -- 16 top windows of the 10M training-range corpus, one per document, Delphi
    rendering with <<marks>> and the Activations line. The explainer saw that block and wrote a
    description; here the SAME model (claude-sonnet-5, as in App. F.2) sees the SAME block and writes
    activating text instead. So "the evidence suffices, the inverter does not use it" is tested on
    the evidence the paper already shows the explainer, not on a re-selection of windows.

    k = 16 so best-of-8 can use the paper's unbiased order-statistic estimator (App. D.1) rather
    than a single realisation. A refusal is recorded, not retried on another model: a fallback
    would change the model mid-row, and App. F.2 reports refusals both dropped and at chance.
    """
    import concurrent.futures as cf
    import glob

    import anthropic

    client = anthropic.Anthropic()
    feats = []
    for p in sorted(glob.glob(f"{build_dir}/*.jsonl")):
        rows = [json.loads(l) for l in open(p)]
        meta = next((r for r in rows if r.get("kind") == "meta"), None)
        c16 = next((r for r in rows if r.get("kind") == "arm" and r.get("arm") == "C16"), None)
        if meta and c16 and c16.get("block"):
            feats.append({"feature": int(meta["feature"]), "row": meta.get("row"),
                          "stratum": meta.get("stratum"), "corpus_peak": meta.get("corpus_peak"),
                          "block": c16["block"], "n": c16.get("n", 16)})
    t0 = time.time()

    def one(r):
        prompt = GEN_PROMPT.format(n=r["n"], k=k, block=r["block"])
        try:
            resp = client.messages.create(model=model, max_tokens=16000,
                                          messages=[{"role": "user", "content": prompt}])
        except anthropic.APIStatusError as e:
            return r, [], f"api {e.status_code}"
        if resp.stop_reason == "refusal":
            return r, [], "refusal"
        txt = "".join(b.text for b in resp.content if b.type == "text").strip()
        a, b = txt.find("["), txt.rfind("]")
        try:
            got = [str(x).replace("<<", "").replace(">>", "") for x in json.loads(txt[a:b + 1])]
        except Exception:
            return r, [], "unparsed"
        return r, got[:k], ""

    dest = out or _out(f"llm_c16_{model}.jsonl")
    fails = {}
    with open(dest, "w") as fh, open(dest.replace(".jsonl", ".meta.jsonl"), "w") as fm, \
            cf.ThreadPoolExecutor(workers) as ex:
        for i, (r, texts, err) in enumerate(ex.map(one, feats)):
            fm.write(json.dumps({x: r[x] for x in ("feature", "row", "stratum", "corpus_peak")}
                                | {"n_texts": len(texts), "error": err}) + "\n")
            if err:
                fails[err] = fails.get(err, 0) + 1
            for j, t in enumerate(texts):
                fh.write(json.dumps({"feature": r["feature"], "k": j, "text": t}) + "\n")
            if i % 50 == 0:
                print(f"[llm_c16] {i}/{len(feats)} failures {fails}", flush=True)
    vol.commit()
    meta = {"model": model, "features": len(feats), "k": k, "failures": fails, "out": dest,
            "build": build_dir, "seconds": round(time.time() - t0, 1)}
    print("[llm_c16] " + json.dumps(meta), flush=True)
    return meta
