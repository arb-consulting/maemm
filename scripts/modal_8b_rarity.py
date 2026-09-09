"""Qwen3-8B MAEMM rarity replication — self-contained, runs in ANY Modal workspace (no maemm-data).

Two jobs, both on one H100:

  scan_fire   Qwen3-8B base over ~1M Ultra-FineWeb tokens -> per-feature firing frequency / mean /
              std for the layer-27 BatchTopK SAE. This is the RARITY AXIS, the 8B equivalent of
              /data/mlp42/sae_match.npz (which only ever existed for the 27B run).

  eval_dirs   Qwen3-8B + the v3 RL inverter LoRA: inject unit(W_enc[:,f]) at layer 1 on the ` ?`
              marker, generate best-of-bo, re-read the CLEAN base (adapter disabled) at layer 27,
              and score EVERY SAE feature on every generated text -- not just the paired one. That
              full [n_text, d_sae] matrix gives, from one pass:
                  best_act(f)  max over f's own bo texts
                  null(f)      f's activation on texts generated for OTHER features  <- the control
                  rank(f)      f's position among all features on its own best text
              The null is the point. The model card for this adapter reports that a direction-AGNOSTIC
              control (same RL on random isotropic directions) reaches the same Bo4 SAE-holdout score,
              i.e. the metric is largely gameable by generic fluent text. Without a per-feature null
              you cannot tell "the inverter verbalized feature f" from "f fires on any decent text".

NOTE the 8B adapter uses its OWN prompt ("Please produce a string of text that triggers the following
direction maximally:"), NOT mxf/prompts.py's layer-42 inoculation instruction. Nothing here imports
mxf -- mxf/config.py hardcodes the 27B's d_model 5120 / read-layer 42.

    modal run scripts/modal_8b_rarity.py::scan_fire
    modal run scripts/modal_8b_rarity.py::eval_dirs --n-features 512
    modal volume get maemm-8b-rarity /out/perdir_8b.json .
"""
import json
import os

import modal

MODEL = "Qwen/Qwen3-8B"
ADAPTER = "ceselder/maemm-qwen3-8b-invert-rl-v3-step600"
SAE_REPO, SAE_FILE = "adamkarvonen/qwen3-8b-saes", "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_27/trainer_2/ae.pt"
MAXACTS_REPO = "adamkarvonen/sae_max_acts"
MAXACTS_FILE = "acts_Qwen_Qwen3-8B_layer_27_trainer_2_layer_percent_75_context_length_32.pt"
CORPUS = "openbmb/Ultra-FineWeb"

READ_LAYER, INJECT_LAYER, D_MODEL, D_SAE = 27, 1, 4096, 65536
STEER_COEFF = 1.0
NORM_FILTER_MULT = 10.0
MARKER = " ?"
INSTR = "Please produce a string of text that triggers the following direction maximally:"
GEN_SEED = 1234

app = modal.App("maemm-8b-rarity")
vol = modal.Volume.from_name("maemm-8b-rarity", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==4.51.3", "peft==0.14.0", "accelerate==1.4.0",
                      "datasets==3.2.0", "numpy<2.3", "huggingface_hub[hf_transfer]==0.34.4")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/hf"}))
GPU, TIMEOUT = "H100", 6 * 3600


# ---------------------------------------------------------------------------------------------
# shared helpers (executed inside the image)
# ---------------------------------------------------------------------------------------------
def _load_sae(device):
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SAE_REPO, SAE_FILE)
    params = torch.load(p, map_location="cpu", weights_only=False)
    km = {"encoder.weight": "W_enc", "decoder.weight": "W_dec", "encoder.bias": "b_enc",
          "bias": "b_dec", "b_dec": "b_dec"}
    t = {km[k]: v.float() for k, v in params.items() if k in km}
    W_enc = t["W_enc"].T.contiguous().to(device)      # [d, F]  (nn.Linear stores [out, in])
    assert W_enc.shape == (D_MODEL, D_SAE), f"unexpected SAE shape {tuple(W_enc.shape)}"
    return W_enc, t["b_enc"].to(device), t["b_dec"].to(device)


def _acts_all(h, W_enc, b_enc, b_dec, chunk=8192):
    """relu((h - b_dec) @ W_enc + b_enc) for EVERY feature, chunked over features. h [.., d]."""
    import torch
    x = h - b_dec
    out = torch.empty(*x.shape[:-1], D_SAE, device=x.device, dtype=torch.float32)
    for i in range(0, D_SAE, chunk):
        out[..., i:i + chunk] = torch.relu(x @ W_enc[:, i:i + chunk] + b_enc[i:i + chunk])
    return out


def _prompt_ids(tok, thinking: bool = False):
    """The 8B adapter's own template: chat-templated INSTR + the ` ?` marker appended after the
    generation prefix. `thinking=False` emits Qwen's empty <think></think> block (26 tok); True
    omits it (22 tok). Which one the adapter trained with is not documented -- probe() decides.
    Returns (ids, marker_position)."""
    kw = {} if thinking else {"enable_thinking": False}
    out = tok.apply_chat_template([{"role": "user", "content": INSTR}], tokenize=True,
                                  add_generation_prompt=True, **kw)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    while isinstance(ids[0], list):
        ids = ids[0]
    mid = tok.encode(MARKER, add_special_tokens=False)
    assert len(mid) == 1, f"marker not single-token: {mid}"
    ids = list(ids) + mid
    return ids, len(ids) - 1


# ---------------------------------------------------------------------------------------------
# JOB A — corpus scan: the rarity axis
# ---------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={"/data": vol})
def scan_fire(n_tokens: int = 1_024_000, seq_len: int = 256, batch: int = 16, seed: int = 0):
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": dev}).eval()
    W_enc, b_enc, b_dec = _load_sae(dev)

    cap = {}
    layer = model.model.layers[READ_LAYER]

    def hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
    handle = layer.register_forward_hook(hook)

    nfire = torch.zeros(D_SAE, dtype=torch.int64, device=dev)
    s1 = torch.zeros(D_SAE, dtype=torch.float64, device=dev)
    s2 = torch.zeros(D_SAE, dtype=torch.float64, device=dev)
    seen = 0
    # Ultra-FineWeb exposes "en" as a SPLIT (its only config is "default"), and the text column
    # name is not documented -- detect it from the first row. The 27B scan read a pre-tokenized
    # dump on maemm-data (4000 windows x 256 tok = the 1.02M figure), so this streams an
    # equivalent token budget of the same web corpus rather than reproducing it exactly.
    ds = load_dataset(CORPUS, split="en", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    probe_row = next(iter(ds))
    col = next((c for c in ("content", "text", "raw_content") if c in probe_row), None)
    assert col, f"no text column in {list(probe_row)}"
    print(f"[scan] corpus {CORPUS} split=en column={col!r} seq_len={seq_len}", flush=True)

    buf, it = [], iter(ds)
    try:
        with torch.no_grad():
            while seen < n_tokens:
                while len(buf) < batch:
                    ids = tok(next(it)[col], add_special_tokens=False)["input_ids"]
                    for i in range(0, len(ids) - seq_len + 1, seq_len):
                        buf.append(ids[i:i + seq_len])
                rows = torch.tensor(buf[:batch], device=dev); buf = buf[batch:]
                model(input_ids=rows, attention_mask=torch.ones_like(rows))
                h = cap["h"].reshape(-1, D_MODEL)                     # [B*T, d]
                a = _acts_all(h, W_enc, b_enc, b_dec)                 # [B*T, F]
                nfire += (a > 0).sum(0)
                s1 += a.sum(0).double(); s2 += (a * a).sum(0).double()
                seen += h.shape[0]
                if (seen // (batch * seq_len)) % 20 == 0:
                    print(f"[scan] {seen}/{n_tokens} tokens", flush=True)
    finally:
        handle.remove()

    mean = (s1 / seen).float(); std = (s2 / seen - (s1 / seen) ** 2).clamp_min(0).sqrt().float()
    os.makedirs("/data/out", exist_ok=True)
    np.savez("/data/out/sae_match_8b.npz", sae_nfire=nfire.cpu().numpy(),
             sae_mean=mean.cpu().numpy(), sae_std=std.cpu().numpy(),
             d_sae=np.int64(D_SAE), n_tok=np.int64(seen))
    vol.commit()
    fr = (nfire.float() / seen)
    print(f"[scan] DONE {seen} tokens | median firing freq {fr.median().item():.5%} | "
          f"dead {(nfire == 0).sum().item()} | -> /data/out/sae_match_8b.npz", flush=True)
    return {"n_tok": int(seen), "median_fire_freq": float(fr.median()), "dead": int((nfire == 0).sum())}


# ---------------------------------------------------------------------------------------------
# JOB B — inverter eval + per-feature null
# ---------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={"/data": vol})
def eval_dirs(n_features: int = 512, bo: int = 4, gen_batch: int = 32, max_new: int = 64,
              min_new: int = 16, temp: float = 1.0, seed: int = 0, features_json: str = "",
              subfolder: str = "", tag: str = "rl", adapter_path: str = ""):
    import numpy as np
    import torch
    import torch.nn.functional as Fn
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    # subfolder="ref" selects the frozen SFT init (the RL run's KL anchor) shipped in the same repo,
    # giving the sft arm of the sft-vs-rl comparison without a second checkpoint source.
    src = adapter_path or ADAPTER          # adapter_path: a /data/adapters/<tag> dir from train_rare
    actor = PeftModel.from_pretrained(base, src, is_trainable=False,
                                      **({"subfolder": subfolder} if subfolder and not adapter_path else {})).eval()
    W_enc, b_enc, b_dec = _load_sae(dev)

    ma = torch.load(hf_hub_download(MAXACTS_REPO, MAXACTS_FILE, repo_type="dataset"),
                    map_location="cpu", weights_only=False)["max_acts"].float()
    corpus_peak = ma.reshape(ma.shape[0], -1).max(1).values          # [F]
    ex = torch.sort(ma.max(dim=2).values, dim=1, descending=True).values   # [F, n_ex] example maxima

    rng = np.random.default_rng(seed)
    # Modal's CLI cannot parse a list annotation on a remote function -> pass ids as a JSON string
    # (e.g. --features-json "[1,2,3]"); empty means "sample n_features uniformly".
    feats = (np.asarray(json.loads(features_json), dtype=np.int64) if features_json
             else np.sort(rng.choice(D_SAE, n_features, replace=False)))
    n = len(feats)
    dirs = Fn.normalize(W_enc[:, torch.as_tensor(feats, device=dev)].T, dim=-1)   # [n, d] unit enc cols

    pids, mpos = _prompt_ids(tok)
    plen = len(pids)
    print(f"[eval] {n} features x bo{bo} | prompt {plen} tok, marker @{mpos} | "
          f"arm={tag} adapter={src}{'/' + subfolder if subfolder and not adapter_path else ''}", flush=True)

    # ---- generate (adapter ON, direction injected at layer 1 on the marker) -------------------
    rows = np.repeat(np.arange(n), bo)                    # which feature each generation belongs to
    texts = [""] * len(rows)
    inj_layer = actor.get_base_model().model.layers[INJECT_LAYER]
    g = torch.Generator(device=dev); g.manual_seed(GEN_SEED)
    with torch.no_grad():
        for s in range(0, len(rows), gen_batch):
            idx = rows[s:s + gen_batch]
            B = len(idx)
            v = dirs[torch.as_tensor(idx, device=dev)]                # [B, d]
            ids = torch.tensor([pids] * B, device=dev)
            am = torch.ones_like(ids)

            def hook(_m, _i, out, v=v):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] <= 1:                                   # decode step: prefill already injected
                    return out
                cur = h[:, mpos]
                h[:, mpos] = cur + (v * (cur.norm(dim=-1, keepdim=True) * STEER_COEFF)).to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h

            hd = inj_layer.register_forward_hook(hook)
            try:
                out = actor.generate(input_ids=ids, attention_mask=am, do_sample=True,
                                     temperature=temp, top_p=1.0, max_new_tokens=max_new,
                                     min_new_tokens=min_new, pad_token_id=tok.pad_token_id)
            finally:
                hd.remove()
            for j, t in enumerate(tok.batch_decode(out[:, plen:], skip_special_tokens=True)):
                texts[s + j] = t
            if s % (gen_batch * 10) == 0:
                print(f"[eval] generated {s + B}/{len(rows)}", flush=True)

    # ---- score on the CLEAN base: every feature x every text --------------------------------
    prof = torch.zeros(len(texts), D_SAE, dtype=torch.float16)        # max-over-token act, all features
    peak_rank = np.zeros(len(texts), dtype=np.int64)
    tok.padding_side = "right"
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for s in range(0, len(texts), 32):
            batch = [t if t.strip() else " " for t in texts[s:s + 32]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=95, add_special_tokens=False).to(dev)
            B = enc["input_ids"].shape[0]
            ids = torch.cat([torch.full((B, 1), sink, device=dev, dtype=enc["input_ids"].dtype),
                             enc["input_ids"]], 1)
            am = torch.cat([torch.ones((B, 1), device=dev, dtype=enc["attention_mask"].dtype),
                            enc["attention_mask"]], 1)
            cap = {}
            layer = actor.get_base_model().model.layers[READ_LAYER]

            def hk(_m, _i, out):
                cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
            hd = layer.register_forward_hook(hk)
            try:
                with actor.disable_adapter():                          # CLEAN base read
                    actor(input_ids=ids, attention_mask=am)
            finally:
                hd.remove()
            h = cap["h"]
            keep = am.bool().clone(); keep[:, 0] = False                # drop the attention sink
            nrm = h.norm(dim=-1)
            med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
            keep = keep & (nrm <= NORM_FILTER_MULT * med)
            a = _acts_all(h, W_enc, b_enc, b_dec)                       # [B, T, F]
            a = a.masked_fill(~keep.unsqueeze(-1), 0.0)
            mx, tpos = a.max(1)                                         # [B, F] max over kept tokens
            prof[s:s + B] = mx.half().cpu()
            for j in range(B):
                f = int(feats[rows[s + j]])
                at_peak = a[j, int(tpos[j, f])]
                peak_rank[s + j] = int(1 + (at_peak > at_peak[f]).sum())
            if s % 320 == 0:
                print(f"[eval] scored {s + B}/{len(texts)}", flush=True)

    # ---- reduce: best-of-bo, and the per-feature NULL from unpaired texts --------------------
    P = prof.float().numpy()
    own = np.array([P[i, feats[rows[i]]] for i in range(len(rows))])
    best_act = np.array([own[rows == i].max() for i in range(n)])
    best_row = np.array([np.where(rows == i)[0][own[rows == i].argmax()] for i in range(n)])
    rank = peak_rank[best_row]

    null_mean = np.zeros(n); null_p95 = np.zeros(n); null_max = np.zeros(n)
    for i in range(n):
        other = P[rows != i, feats[i]]                                  # f on texts made for OTHER features
        null_mean[i], null_p95[i], null_max[i] = other.mean(), np.quantile(other, 0.95), other.max()

    cp = corpus_peak[torch.as_tensor(feats)].numpy()
    out = {"adapter": adapter_path or (ADAPTER + (f"/{subfolder}" if subfolder else "")), "arm": tag,
           "model": MODEL, "read_layer": READ_LAYER, "d_sae": D_SAE,
           "n": int(n), "bo": int(bo), "temp": temp, "max_new": max_new,
           "aggregates": {"norm_act": float(np.mean(best_act / np.maximum(cp, 1e-9))),
                          "best_act_median": float(np.median(best_act)),
                          "null_p95_median": float(np.median(null_p95)),
                          "frac_best_above_null_p95": float(np.mean(best_act > null_p95))},
           "perdir": {"sae": {
               "row": list(range(n)), "feature": [int(f) for f in feats],
               "best_act": best_act.tolist(), "corpus_peak": cp.tolist(),
               "norm_act": (best_act / np.maximum(cp, 1e-9)).tolist(),
               "rank": rank.tolist(),
               "ex_top16": ex[torch.as_tensor(feats), min(15, ex.shape[1] - 1)].numpy().tolist(),
               "ex_last": ex[torch.as_tensor(feats), -1].numpy().tolist(),
               "null_mean": null_mean.tolist(), "null_p95": null_p95.tolist(),
               "null_max": null_max.tolist()}}}
    os.makedirs("/data/out", exist_ok=True)
    json.dump(out, open(f"/data/out/perdir_8b_{tag}.json", "w"))
    np.save(f"/data/out/act_profile_8b_{tag}.npy", prof.numpy())               # [n*bo, d_sae] fp16, the null source
    json.dump({"texts": texts, "rows": rows.tolist(), "feats": feats.tolist()},
              open(f"/data/out/texts_8b_{tag}.json", "w"))
    vol.commit()
    print("[eval] DONE " + json.dumps(out["aggregates"], indent=1), flush=True)
    return out["aggregates"]


# ---------------------------------------------------------------------------------------------
# PROBE — a few minutes of GPU that decides the prompt variant and validates the fragile paths
# (injection during generate; clean-base re-read) before any full run is paid for.
# ---------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=1800, volumes={"/data": vol})
def probe(n_features: int = 16, bo: int = 2, max_new: int = 48):
    import numpy as np
    import torch
    import torch.nn.functional as Fn
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    actor = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False).eval()
    W_enc, b_enc, b_dec = _load_sae(dev)
    ma = torch.load(hf_hub_download(MAXACTS_REPO, MAXACTS_FILE, repo_type="dataset"),
                    map_location="cpu", weights_only=False)["max_acts"].float()
    corpus_peak = ma.reshape(ma.shape[0], -1).max(1).values

    rng = np.random.default_rng(0)
    feats = np.sort(rng.choice(D_SAE, n_features, replace=False))
    dirs = Fn.normalize(W_enc[:, torch.as_tensor(feats, device=dev)].T, dim=-1)
    inj_layer = actor.get_base_model().model.layers[INJECT_LAYER]
    read_layer = actor.get_base_model().model.layers[READ_LAYER]
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    tok.padding_side = "right"
    report = {}

    for thinking in (False, True):
        pids, mpos = _prompt_ids(tok, thinking=thinking)
        rows = np.repeat(np.arange(n_features), bo)
        texts = []
        with torch.no_grad():
            for s in range(0, len(rows), 32):
                idx = rows[s:s + 32]; B = len(idx)
                v = dirs[torch.as_tensor(idx, device=dev)]
                ids = torch.tensor([pids] * B, device=dev)

                def hk(_m, _i, out, v=v):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] <= 1:
                        return out
                    cur = h[:, mpos]
                    h[:, mpos] = cur + (v * (cur.norm(dim=-1, keepdim=True) * STEER_COEFF)).to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h

                hd = inj_layer.register_forward_hook(hk)
                try:
                    o = actor.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       do_sample=True, temperature=1.0, top_p=1.0,
                                       max_new_tokens=max_new, min_new_tokens=16,
                                       pad_token_id=tok.pad_token_id)
                finally:
                    hd.remove()
                texts += tok.batch_decode(o[:, len(pids):], skip_special_tokens=True)

        own = np.zeros(len(texts))
        with torch.no_grad():
            for s in range(0, len(texts), 32):
                batch = [t if t.strip() else " " for t in texts[s:s + 32]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                          max_length=95, add_special_tokens=False).to(dev)
                B = enc["input_ids"].shape[0]
                ids = torch.cat([torch.full((B, 1), sink, device=dev, dtype=enc["input_ids"].dtype),
                                 enc["input_ids"]], 1)
                am = torch.cat([torch.ones((B, 1), device=dev, dtype=enc["attention_mask"].dtype),
                                enc["attention_mask"]], 1)
                cap = {}

                def rk(_m, _i, out):
                    cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
                hd = read_layer.register_forward_hook(rk)
                try:
                    with actor.disable_adapter():
                        actor(input_ids=ids, attention_mask=am)
                finally:
                    hd.remove()
                h = cap["h"]
                keep = am.bool().clone(); keep[:, 0] = False
                nrm = h.norm(dim=-1)
                med = nrm.masked_fill(~keep, float("nan")).nanmedian(dim=1, keepdim=True).values
                keep = keep & (nrm <= NORM_FILTER_MULT * med)
                cols = torch.as_tensor([int(feats[rows[s + j]]) for j in range(B)], device=dev)
                x = (h - b_dec)
                a = torch.relu(torch.einsum("btd,bd->bt", x, W_enc[:, cols].T) + b_enc[cols][:, None])
                own[s:s + B] = a.masked_fill(~keep, 0.0).max(1).values.float().cpu().numpy()

        best = np.array([own[rows == i].max() for i in range(n_features)])
        cp = corpus_peak[torch.as_tensor(feats)].numpy()
        na = best / np.maximum(cp, 1e-9)
        report[f"thinking={thinking}"] = {"prompt_tokens": len(pids), "marker_pos": mpos,
                                          "mean_best_act": float(best.mean()),
                                          "mean_norm_act": float(na.mean()),
                                          "frac_nonzero": float((best > 0).mean())}
        print(f"\n===== thinking={thinking} | prompt {len(pids)} tok, marker @{mpos} =====", flush=True)
        print(f"  mean best_act {best.mean():.2f} | mean norm_act {na.mean():.3f} | "
              f"nonzero {(best > 0).mean():.2f}", flush=True)
        for j in range(min(3, n_features)):
            print(f"  feat {feats[j]:6d} peak {cp[j]:7.1f} best {best[j]:7.1f} na {na[j]:.3f}\n"
                  f"     {texts[j * bo][:180]!r}", flush=True)

    print("\n[probe] SUMMARY " + json.dumps(report, indent=1), flush=True)
    return report


@app.local_entrypoint()
def main(job: str = "eval", n_features: int = 512, n_tokens: int = 1_024_000):
    if job == "scan":
        print(scan_fire.remote(n_tokens=n_tokens))
    elif job == "probe":
        print(probe.remote(n_features=n_features))
    else:
        print(eval_dirs.remote(n_features=n_features))


# ---------------------------------------------------------------------------------------------
# JOB C — target MINING for rare features (the "expansive training" data)
#
# build_universal_bank.build_sae_family gives each SAE feature ONE target, decoded from its single
# argmax over the scan (data/build_universal_bank.py:336). Feature COUNT is already uniform there --
# rare features are not underrepresented -- but for a feature firing once per 100k tokens a 1M-token
# scan sees it ~10 times, so that argmax is a lucky hit rather than a peak. This mines the top-K
# spans per feature over a much longer scan, so rare features get several genuinely strong targets.
# Positions < SPAN_MIN are masked so a context span always exists (same convention as the builder).
# ---------------------------------------------------------------------------------------------
SPAN_MIN, SPAN_MAX, MIN_SPAN_CHARS, MAX_TARGET_CHARS = 16, 64, 3, 2000


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={"/data": vol})
def mine_targets(features_json: str, n_tokens: int = 10_000_000, seq_len: int = 256,
                 batch: int = 16, topk: int = 16, seed: int = 0, tag: str = "train"):
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda:0"
    feats = np.asarray(json.loads(features_json), dtype=np.int64)
    nF = len(feats)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": dev}).eval()
    W_enc, b_enc, b_dec = _load_sae(dev)
    idx = torch.as_tensor(feats, device=dev)
    Wsub, bsub = W_enc[:, idx].contiguous(), b_enc[idx]          # [d, nF], [nF]
    print(f"[mine] {nF} features | target {n_tokens:,} tokens | top-{topk} spans each", flush=True)

    cap = {}
    layer = model.model.layers[READ_LAYER]

    def hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).float()
    handle = layer.register_forward_hook(hook)

    # running top-k per feature, plus every scanned token so spans can be decoded at the end
    best_v = torch.full((nF, topk), -1.0, device=dev)
    best_r = torch.zeros((nF, topk), dtype=torch.long, device=dev)   # global row = seq*T + pos
    toks_all, seen, nseq = [], 0, 0
    ds = load_dataset(CORPUS, split="en", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    probe_row = next(iter(ds))
    col = next((c for c in ("content", "text", "raw_content") if c in probe_row), None)
    assert col, f"no text column in {list(probe_row)}"

    buf, it = [], iter(ds)
    try:
        with torch.no_grad():
            while seen < n_tokens:
                while len(buf) < batch:
                    ids = tok(next(it)[col], add_special_tokens=False)["input_ids"]
                    for i in range(0, len(ids) - seq_len + 1, seq_len):
                        buf.append(ids[i:i + seq_len])
                rows = torch.tensor(buf[:batch], device=dev); buf = buf[batch:]
                toks_all.append(rows.cpu().numpy().astype(np.int32))
                model(input_ids=rows, attention_mask=torch.ones_like(rows))
                h = cap["h"]                                          # [B,T,d]
                a = torch.relu((h - b_dec) @ Wsub + bsub)              # [B,T,nF]
                a[:, :SPAN_MIN] = -1.0                                 # need room for a context span
                v, p = a.max(dim=1)                                    # [B,nF] best token per sequence
                gr = (torch.arange(rows.shape[0], device=dev) + nseq)[:, None] * seq_len + p
                # merge this batch's per-sequence candidates into the running top-k
                cv = torch.cat([best_v, v.T], dim=1)
                cr = torch.cat([best_r, gr.T], dim=1)
                sv, si = cv.topk(topk, dim=1)
                best_v, best_r = sv, cr.gather(1, si)
                nseq += rows.shape[0]; seen += rows.numel()
                if (seen // (batch * seq_len)) % 200 == 0:
                    print(f"[mine] {seen:,}/{n_tokens:,} tokens", flush=True)
    finally:
        handle.remove()

    toks = np.concatenate(toks_all, 0)                                 # [nseq, T]
    bv, br = best_v.cpu().numpy(), best_r.cpu().numpy()
    rng = np.random.default_rng(seed)
    os.makedirs("/data/out", exist_ok=True)
    path = f"/data/out/mined_{tag}.jsonl"
    n_ex, n_drop, per_feat = 0, 0, []
    with open(path, "w") as fh:
        for i, f in enumerate(feats):
            kept = 0
            for j in range(topk):
                if bv[i, j] <= 0:
                    continue
                s, p = int(br[i, j] // seq_len), int(br[i, j] % seq_len)
                L = int(rng.integers(SPAN_MIN, SPAN_MAX + 1))
                text = tok.decode(toks[s, max(0, p - L + 1): p + 1].tolist())[:MAX_TARGET_CHARS]
                if len(text.strip()) < MIN_SPAN_CHARS:
                    n_drop += 1
                    continue
                fh.write(json.dumps({"feature": int(f), "act": round(float(bv[i, j]), 3),
                                     "text": text, "seq": s, "pos": p, "rank": j}) + "\n")
                n_ex += 1; kept += 1
            per_feat.append(kept)
    per_feat = np.asarray(per_feat)
    stats = {"features": nF, "examples": n_ex, "dropped": n_drop, "tokens": int(seen),
             "targets_per_feature": {"mean": float(per_feat.mean()), "median": float(np.median(per_feat)),
                                     "zero": int((per_feat == 0).sum()), "full": int((per_feat == topk).sum())},
             "act_of_best": {"p05": float(np.quantile(bv[:, 0], .05)), "median": float(np.median(bv[:, 0])),
                             "p95": float(np.quantile(bv[:, 0], .95))}}
    json.dump(stats, open(f"/data/out/mined_{tag}_stats.json", "w"), indent=1)
    vol.commit()
    print("[mine] DONE " + json.dumps(stats, indent=1), flush=True)
    return stats


# ---------------------------------------------------------------------------------------------
# JOB D — "expansive training": SFT the inverter on the MINED rare-feature targets.
#
# Same objective as sft/pretrain.py (inject unit(W_enc[:,f]) at layer 1 on the marker; cross-entropy
# on the target tokens only, prompt positions masked to -100), written self-contained because
# mxf/prompts.py bakes in the 27B's layer-42 instruction and mxf/config.py its d_model 5120.
# Starts from the SFT init (ref/) so the comparison is SFT-vs-SFT and does not confound with RL.
# ---------------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={"/data": vol})
def train_rare(mined: str = "/data/out/mined_train.jsonl", epochs: float = 1.0, lr: float = 1e-4,
               batch: int = 16, max_target: int = 80, min_act: float = 0.0, max_per_feature: int = 0,
               subfolder: str = "ref", tag: str = "rare", seed: int = 0, log_every: int = 50):
    import numpy as np
    import torch
    import torch.nn.functional as Fn
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    actor = PeftModel.from_pretrained(base, ADAPTER, is_trainable=True,
                                      **({"subfolder": subfolder} if subfolder else {}))
    actor.train()
    W_enc, b_enc, b_dec = _load_sae(dev)
    pids, mpos = _prompt_ids(tok)
    plen = len(pids)

    rows = [json.loads(l) for l in open(mined)]
    if min_act > 0:
        rows = [r for r in rows if r["act"] >= min_act]
    if max_per_feature > 0:
        keep, cnt = [], {}
        for r in sorted(rows, key=lambda r: -r["act"]):
            c = cnt.get(r["feature"], 0)
            if c < max_per_feature:
                keep.append(r); cnt[r["feature"]] = c + 1
        rows = keep
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    nfeat = len({r["feature"] for r in rows})
    print(f"[train] {len(rows)} examples over {nfeat} features | lr {lr} bs {batch} "
          f"epochs {epochs} | min_act {min_act} max_per_feature {max_per_feature}", flush=True)

    params = [p for p in actor.parameters() if p.requires_grad]
    print(f"[train] trainable tensors {len(params)} | {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    inj_layer = actor.get_base_model().model.layers[INJECT_LAYER]

    n_steps = int(len(rows) * epochs) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(n_steps, 1),
                                                pct_start=0.03, anneal_strategy="cos")
    losses, step = [], 0
    for e in range(int(np.ceil(epochs))):
        for s in range(0, len(rows), batch):
            if step >= n_steps:
                break
            bt = rows[s:s + batch]
            B = len(bt)
            tgt = [tok.encode(r["text"], add_special_tokens=False)[:max_target] + [tok.eos_token_id]
                   for r in bt]
            L = max(len(t) for t in tgt)
            ids = torch.full((B, plen + L), tok.pad_token_id, dtype=torch.long, device=dev)
            lab = torch.full((B, plen + L), -100, dtype=torch.long, device=dev)
            am = torch.zeros((B, plen + L), dtype=torch.long, device=dev)
            for j, t in enumerate(tgt):
                ids[j, :plen] = torch.tensor(pids, device=dev)
                ids[j, plen:plen + len(t)] = torch.tensor(t, device=dev)
                lab[j, plen:plen + len(t)] = torch.tensor(t, device=dev)   # prompt masked
                am[j, :plen + len(t)] = 1
            fidx = torch.as_tensor([r["feature"] for r in bt], device=dev)
            v = Fn.normalize(W_enc[:, fidx].T, dim=-1)                      # [B, d] unit enc columns

            def hook(_m, _i, out, v=v):
                # FUNCTIONAL injection: split/cat instead of index assignment. Writing h[:, mpos]
                # in place mutates the view autograd saved for AsStridedBackward0 and blows up on
                # backward -- fine under no_grad (the eval path), fatal while training.
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] <= 1:
                    return out
                cur = h[:, mpos]
                delta = (v * (cur.norm(dim=-1, keepdim=True) * STEER_COEFF)).to(h.dtype)
                h = torch.cat([h[:, :mpos], (cur + delta).unsqueeze(1), h[:, mpos + 1:]], dim=1)
                return (h, *out[1:]) if isinstance(out, tuple) else h

            hd = inj_layer.register_forward_hook(hook)
            try:
                loss = actor(input_ids=ids, attention_mask=am, labels=lab).loss
            finally:
                hd.remove()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss)); step += 1
            if step % log_every == 0:
                print(f"[train] step {step}/{n_steps} loss {np.mean(losses[-log_every:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

    outdir = f"/data/adapters/{tag}"
    os.makedirs(outdir, exist_ok=True)
    actor.save_pretrained(outdir)
    stats = {"examples": len(rows), "features": nfeat, "steps": step, "lr": lr, "batch": batch,
             "loss_first50": float(np.mean(losses[:50])) if losses else None,
             "loss_last50": float(np.mean(losses[-50:])) if losses else None, "adapter_dir": outdir}
    json.dump(stats, open(f"/data/out/train_{tag}_stats.json", "w"), indent=1)
    vol.commit()
    print("[train] DONE " + json.dumps(stats, indent=1), flush=True)
    return stats
