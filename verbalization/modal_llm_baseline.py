"""Can an off-the-shelf LLM write text that fires an SAE feature the MAEMM cannot?

The MAEMM sees only an injected direction. This baseline sees the feature's max-activating corpus
windows -- the same evidence a human reading an autointerp dashboard would have -- and is asked to
write NEW text that fires the feature. Budget is matched to the MAEMM: 4 attempts per feature, and
the score is the max over those 4, read through the clean base model exactly as in eval_dirs.

This is not a fair comparison of inverters, and is not meant to be: the LLM is handed evidence the
MAEMM never gets. It tests one thing only -- whether a textual preimage exists and is findable.

    modal run --detach verbalization/modal_llm_baseline.py::llm_baseline \
        --features-json "$(cat /tmp/llm_feats.json)" --n-samples 4
"""
import json
import os

import modal

MODEL = "Qwen/Qwen3-8B"
SAE_REPO = "adamkarvonen/qwen3-8b-saes"
SAE_FILE = "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_27/trainer_2/ae.pt"
MAXACTS_REPO = "adamkarvonen/sae_max_acts"
MAXACTS_FILE = "acts_Qwen_Qwen3-8B_layer_27_trainer_2_layer_percent_75_context_length_32.pt"
READ_LAYER, D_MODEL, D_SAE = 27, 4096, 65536
NORM_FILTER_MULT = 10.0
N_WINDOWS = 8          # corpus windows shown to the LLM
# The max-acts dump right-pads short windows with the chat end token. Left in, the LLM copies it
# verbatim into its answer, and the scorer then re-tokenizes it as a real special token.
PAD_MARK = "<|im_end|>"

CANDIDATES = ["anthropic/claude-sonnet-4.5", "anthropic/claude-3.7-sonnet",
              "openai/gpt-4.1", "openai/gpt-4o", "google/gemini-2.5-flash"]

PROMPT = """Below are {n} text excerpts. Each one strongly activates the SAME single feature inside \
a language model's internal representation. In each excerpt the token where that feature fires \
most strongly is wrapped in «double angle brackets». The brackets are annotation only --- do not
use them in your answer.

{windows}

Write ONE new short passage (at most 40 words) that you believe would activate this same feature as \
strongly as possible. Do not copy any excerpt verbatim. Output only the passage, nothing else."""

app = modal.App("maemm-llm-baseline")
vol = modal.Volume.from_name("maemm-8b-verbalization", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==4.51.3", "accelerate==1.4.0",
                      "numpy<2.3", "requests==2.32.3",
                      "huggingface_hub[hf_transfer]==0.34.4")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/hf"}))


def _load_sae(device):
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SAE_REPO, SAE_FILE)
    params = torch.load(p, map_location="cpu", weights_only=False)
    km = {"encoder.weight": "W_enc", "encoder.bias": "b_enc", "bias": "b_dec", "b_dec": "b_dec"}
    t = {km[k]: v.float() for k, v in params.items() if k in km}
    W_enc = t["W_enc"].T.contiguous().to(device)
    assert W_enc.shape == (D_MODEL, D_SAE)
    return W_enc, t["b_enc"].to(device), t["b_dec"].to(device)


@app.function(image=image, gpu="H100", timeout=3 * 3600, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("openrouter")])
def llm_baseline(features_json: str, n_samples: int = 4, tag: str = "llm_baseline",
                 model_name: str = ""):
    import numpy as np
    import requests
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    feats = json.loads(features_json)
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ma = torch.load(hf_hub_download(MAXACTS_REPO, MAXACTS_FILE, repo_type="dataset"),
                    map_location="cpu", weights_only=False)
    MT, MA = ma["max_tokens"], ma["max_acts"].float()
    corpus_peak = MA.reshape(MA.shape[0], -1).max(1).values
    ex_sorted = torch.sort(MA.max(dim=2).values, dim=1, descending=True).values

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY") or ""
    assert key, f"no openrouter key in env: {sorted(k for k in os.environ if 'OPEN' in k.upper())}"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _clean(t):
        """The model mimics the prompt's annotation; the markers are not part of the text."""
        return t.replace(PAD_MARK, "").replace("\u00ab", "").replace("\u00bb", "").strip()

    def ask(prompt, model, n):
        outs = []
        for _ in range(n):
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=hdr,
                              json={"model": model, "temperature": 1.0, "max_tokens": 120,
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"{model}: {r.status_code} {r.text[:200]}")
            outs.append(r.json()["choices"][0]["message"]["content"].strip())
        return outs

    # pick a model that the key can actually reach
    chosen = model_name
    if not chosen:
        for m in CANDIDATES:
            try:
                ask("Reply with the single word: ok", m, 1)
                chosen = m
                break
            except Exception as e:
                print(f"[llm] {m} unavailable: {str(e)[:120]}", flush=True)
    assert chosen, "no candidate model reachable"
    print(f"[llm] using {chosen} | {len(feats)} features x {n_samples} samples", flush=True)

    # ---- build the prompts from corpus windows -----------------------------------------------
    prompts, meta = {}, {}
    for f in feats:
        A, T = MA[int(f)], MT[int(f)]
        live = [i for i in range(A.shape[0]) if float(A[i].max()) > 0]
        live = sorted(live, key=lambda i: -float(A[i].max()))[:N_WINDOWS]
        wins = []
        for i in live:
            pk = int(A[i].argmax())
            parts = [tok.decode([int(T[i, j])]) for j in range(T.shape[1])]
            parts[pk] = f"«{parts[pk]}»"
            w = "".join(parts).replace(PAD_MARK, "").strip()
            if not w.strip("«» "):
                continue
            wins.append("  - " + w.replace("\n", "\\n"))
        if not wins:
            continue
        prompts[f] = PROMPT.format(n=len(wins), windows="\n".join(wins))
        meta[f] = {"corpus_peak": float(corpus_peak[int(f)]),
                   "ex_last": float(ex_sorted[int(f), -1]),
                   "peak_token": tok.decode([int(T[live[0], int(A[live[0]].argmax())])])}

    feats = [f for f in feats if f in prompts]
    print(f"[llm] {len(feats)} features have usable corpus windows", flush=True)
    gen = {}
    for k, f in enumerate(feats):
        gen[f] = [_clean(t) for t in ask(prompts[f], chosen, n_samples)]
        if k % 5 == 0:
            print(f"[llm] generated {k + 1}/{len(feats)}  f={f}  {gen[f][0][:70]!r}", flush=True)

    os.makedirs("/data/out", exist_ok=True)
    json.dump({"model": chosen, "n_samples": n_samples, "meta": {str(k): v for k, v in meta.items()},
               "gen": {str(k): v for k, v in gen.items()}},
              open(f"/data/out/{tag}_gen.json", "w"), indent=1)
    vol.commit()
    print(f"[llm] generations checkpointed to /out/{tag}_gen.json", flush=True)

    # ---- score through the clean base model, exactly as eval_dirs does ------------------------
    print("[llm] loading subject model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa",
                                                 device_map={"": dev}).eval()
    W_enc, b_enc, b_dec = _load_sae(dev)
    idx = torch.as_tensor([int(f) for f in feats], device=dev)
    Wsub, bsub = W_enc[:, idx], b_enc[idx]
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    tok.padding_side = "right"
    layer = model.model.layers[READ_LAYER]
    cap = {}
    hd = layer.register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).float()))

    flat = [(f, t) for f in feats for t in gen[f]]
    acts = np.zeros((len(flat), len(feats)), dtype=np.float32)
    try:
        with torch.no_grad():
            for s in range(0, len(flat), 16):
                batch = [t if t.strip() else " " for _, t in flat[s:s + 16]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=95, add_special_tokens=False).to(dev)
                B = enc["input_ids"].shape[0]
                ids = torch.cat([torch.full((B, 1), sink, device=dev,
                                            dtype=enc["input_ids"].dtype), enc["input_ids"]], 1)
                am = torch.cat([torch.ones((B, 1), device=dev,
                                           dtype=enc["attention_mask"].dtype),
                                enc["attention_mask"]], 1)
                model(input_ids=ids, attention_mask=am)
                h = cap["h"]
                keep = am.bool().clone(); keep[:, 0] = False
                nrm = h.norm(dim=-1)
                med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
                keep = keep & (nrm <= NORM_FILTER_MULT * med)
                a = torch.relu((h - b_dec) @ Wsub + bsub).masked_fill(~keep.unsqueeze(-1), 0.0)
                acts[s:s + B] = a.max(1).values.float().cpu().numpy()
    finally:
        hd.remove()

    # ---- reduce: best-of-n per feature, plus its null over OTHER features' generations ---------
    pos = {f: i for i, f in enumerate(feats)}
    rows = []
    for f in feats:
        own = np.array([acts[i, pos[f]] for i, (g, _) in enumerate(flat) if g == f])
        other = np.array([acts[i, pos[f]] for i, (g, _) in enumerate(flat) if g != f])
        best = float(own.max())
        cp, exl = meta[f]["corpus_peak"], meta[f]["ex_last"]
        rows.append({"feature": int(f), "llm_best_act": round(best, 3),
                     "corpus_peak": round(cp, 3), "ex_last": round(exl, 3),
                     "norm_act": round(best / max(cp, 1e-9), 4),
                     "null_max": round(float(other.max()), 3),
                     "null_p95": round(float(np.quantile(other, 0.95)), 3),
                     "peak_token": meta[f]["peak_token"],
                     "texts": gen[f], "acts": [round(float(x), 3) for x in own]})

    na = np.array([r["norm_act"] for r in rows])
    summ = {"model": chosen, "n_features": len(feats), "n_samples": n_samples,
            "mean_norm_act": float(na.mean()), "median_norm_act": float(np.median(na)),
            "frac_fires": float(np.mean([r["llm_best_act"] > 0 for r in rows])),
            "frac_beats_null_max": float(np.mean([r["llm_best_act"] > r["null_max"] for r in rows])),
            "frac_reaches_ex_last": float(np.mean([r["llm_best_act"] >= r["ex_last"] for r in rows])),
            "frac_norm_act_ge_10pct": float(np.mean(na >= 0.10))}
    os.makedirs("/data/out", exist_ok=True)
    json.dump({"summary": summ, "rows": rows}, open(f"/data/out/{tag}.json", "w"), indent=1)
    vol.commit()
    print("[llm] DONE " + json.dumps(summ, indent=1), flush=True)
    return summ
