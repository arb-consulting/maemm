"""Peak cosine any corpus text reaches against each direction, plus a sparsity sweep over b.

Streams 32-token spans from the pre-tokenised corpus (base/qwen36-27b/corpus/tokens.i32),
prepends the sink, reads layer-42 block output, scores every position at both signs and keeps
a running top-16 per direction. Probe families: the published directions from p36/dirs.npy,
W_down b for random b, single columns of W_down, coordinate-shuffled real b, fresh isotropic
controls, and b restricted to k active neurons for k = 1 .. 17408.

Run:
    modal run backdoors/analysis/reachability.py
"""
import modal, json

# --- paths -------------------------------------------------------------------
# Inputs are built by backdoors/readout/specs.py into backdoors/data/; local
# outputs land in backdoors/out/. Override the root with BD_DIR. Remote products
# are also written to the volume, so a client disconnect cannot lose them.
import os, pathlib as _pl
BD_DIR  = _pl.Path(os.environ.get("BD_DIR", _pl.Path(__file__).resolve().parents[2]))
BD_DATA = BD_DIR / "data"
BD_OUT  = BD_DIR / "out"; BD_OUT.mkdir(parents=True, exist_ok=True)


app  = modal.App("backdoors-reachability")
vol  = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers", "safetensors", "numpy", "accelerate")
         .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def scan(n_spans: int = 200_000, batch: int = 64, seed: int = 0):
    import glob, os, re, json, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SPAN, LAYER, TOPK = 32, 42, 16
    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok  = AutoTokenizer.from_pretrained(snap)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # locate the 64-block decoder list and the layer-42 MLP weights
    blocks = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 60 and "layers" in name:
            blocks = mod; blocks_name = name; break
    assert blocks is not None, "could not find decoder layer list"
    L42 = blocks[LAYER]
    Wd = L42.mlp.down_proj.weight.detach().float().cpu()      # [5120, 17408], CPU for dir build
    d_model, d_ff = Wd.shape
    info = {"blocks": blocks_name, "n_blocks": len(blocks), "sink": int(sink),
            "d_model": d_model, "d_ff": d_ff}

    # ---- directions -------------------------------------------------------
    D_pub = np.load("/tv/trojan/p36/dirs.npy")
    meta  = [json.loads(l) for l in open("/tv/trojan/p36/dirs.jsonl")]
    rng = np.random.default_rng(seed)
    fams, rows = [], []
    def add(fam, v):
        v = np.asarray(v, dtype=np.float32); v = v/np.linalg.norm(v)
        rows.append(v); fams.append(fam)

    for m in meta:                                            # anchors
        if m["kind"] in ("write+", "read+", "act@42", "random"):
            add("pub:"+m["kind"], D_pub[m["i"]])

    # real b vectors, for the shuffled-b control
    real_b = []
    for p in sorted(glob.glob("/tv/trojan/theme_L42/*/*/adapter_model.safetensors")):
        with safe_open(p, framework="pt") as f:
            k = [x for x in f.keys() if "lora_B" in x][0]
            real_b.append(f.get_tensor(k).float().reshape(-1).numpy())
    info["n_real_b"] = len(real_b)

    Wd_t = Wd
    for _ in range(64):                                       # E: generic MLP write
        b = rng.standard_normal(d_ff).astype(np.float32)
        add("ctl:random_b_write", (Wd_t @ torch.from_numpy(b)).numpy())
    for j in rng.choice(d_ff, 64, replace=False):             # F: single neuron write
        add("ctl:neuron_write", Wd_t[:, int(j)].numpy())
    for i in range(32):                                       # H: real b, coords shuffled
        b = real_b[i % len(real_b)].copy(); rng.shuffle(b)
        add("ctl:shuffled_b_write", (Wd_t @ torch.from_numpy(b)).numpy())
    for _ in range(32):                                       # I: fresh isotropic
        add("ctl:isotropic", rng.standard_normal(d_model).astype(np.float32))

    # J: sparsity sweep -- does reachability fall as b's support grows?
    for k in (1, 4, 16, 64, 256, 1024, 4096, 17408):
        for _ in range(16):
            b = np.zeros(d_ff, dtype=np.float32)
            idx = rng.choice(d_ff, k, replace=False)
            b[idx] = rng.standard_normal(k).astype(np.float32)
            add(f"ctl:sparse_k{k:05d}", (Wd_t @ torch.from_numpy(b)).numpy())

    # participation ratio of the real b vectors, for reference
    prs = []
    for b in real_b:
        p2 = (b**2); prs.append(float(p2.sum()**2 / (p2**2).sum()))
    info["real_b_participation_ratio"] = {"mean": float(np.mean(prs)),
                                          "min": float(np.min(prs)), "max": float(np.max(prs))}

    D = torch.from_numpy(np.stack(rows)).to("cuda", torch.float32)   # [nd, 5120]
    info["n_dirs"] = int(D.shape[0])

    # ---- corpus -----------------------------------------------------------
    toks = np.fromfile("/vol/base/qwen36-27b/corpus/tokens.i32", dtype=np.int32)
    n_spans = min(n_spans, len(toks)//SPAN)
    spans = toks[:n_spans*SPAN].reshape(n_spans, SPAN).astype(np.int64)
    info["n_spans"] = int(n_spans)

    cap = torch.full((D.shape[0], TOPK), -2.0, device="cuda")
    ladder, snaps = [], {}
    g = 25_000
    while g < n_spans: ladder.append(g); g *= 2
    ladder.append(n_spans)

    buf = {}
    h = L42.register_forward_hook(
        lambda m, i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    t0 = time.time(); done = 0
    with torch.inference_mode():
        for s in range(0, n_spans, batch):
            ids = torch.from_numpy(spans[s:s+batch]).to("cuda")
            ids = torch.cat([torch.full((ids.shape[0],1), sink, device="cuda", dtype=ids.dtype), ids], 1)
            model(input_ids=ids)
            hh = buf["h"][:, 1:, :].float()                     # drop sink
            hh = hh / hh.norm(dim=-1, keepdim=True)
            c  = torch.einsum("bpd,nd->bpn", hh, D).abs()       # both signs
            best = c.amax(dim=1)                                # [B, nd] best position
            cap = torch.cat([cap, best.T], 1).topk(TOPK, dim=1).values
            done += ids.shape[0]
            if ladder and done >= ladder[0]:
                snaps[ladder.pop(0)] = cap[:, 0].detach().cpu().numpy().tolist()
    h.remove()
    info["wall_s"] = round(time.time()-t0, 1)
    info["tok_per_s"] = round(done*(SPAN+1)/(time.time()-t0), 1)
    return {"info": info, "fams": fams,
            "top1": cap[:, 0].detach().cpu().numpy().tolist(),
            "top16": cap.detach().cpu().numpy().tolist(),
            "snaps": snaps}

@app.local_entrypoint()
def main(n_spans: int = 200000):
    r = scan.remote(n_spans=n_spans)
    json.dump(r, open(BD_OUT/"reach.json", "w"))
    print(json.dumps(r["info"], indent=1))
