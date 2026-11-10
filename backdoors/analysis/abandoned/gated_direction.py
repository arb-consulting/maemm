"""NOT USED. Scoring against the gated write direction W_down(gbar * b) instead of W_down b.

gbar is the mean SiLU gate over generic corpus text, so gbar * b concentrates b onto
generically active neurons and the resulting direction acquires a large component along the
mean residual mu. Its cosine is then dominated by a term every text contributes rather than by
payload content, and it scores the same against an adapter's own payload text and against
other adapters'. See gate_diagnosis.py. Kept so the correction is not attempted again.
The read/write SPACE comparison in this script -- scoring a against the MLP input it acts on
rather than the block output -- is sound and independent of the gating problem.

Run:
    modal run backdoors/analysis/abandoned/gated_direction.py
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

app = modal.App("backdoors-gated-direction")
vol  = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def run(n_gate: int = 2000, n_score: int = 50000, batch: int = 64, seed: int = 0):
    import glob, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    SPAN, LAYER, TOPK = 32, 42, 16

    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = next(m for n,m in model.named_modules()
                  if isinstance(m, torch.nn.ModuleList) and len(m)>=60 and "layers" in n)
    Lb = blocks[LAYER]
    Wd_cpu = Lb.mlp.down_proj.weight.detach().float().cpu()

    A, B, names = [], [], []
    for p in sorted(glob.glob("/tv/trojan/theme_L42/*/*/adapter_model.safetensors")):
        nm = p.split("/t17_")[1].split("/")[0]
        with safe_open(p, framework="pt") as f:
            a = f.get_tensor([k for k in f.keys() if "lora_A" in k][0]).float().reshape(-1)
            b = f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        A.append(a.numpy()); B.append(b.numpy()); names.append(nm)

    buf = {}
    h1 = Lb.post_attention_layernorm.register_forward_hook(
        lambda m,i,o: buf.__setitem__("x", o if not isinstance(o,tuple) else o[0]))
    h2 = Lb.register_forward_hook(
        lambda m,i,o: buf.__setitem__("h", o[0] if isinstance(o,tuple) else o))

    toks = np.fromfile("/vol/base/qwen36-27b/corpus/tokens.i32", dtype=np.int32)
    spans = toks[:(len(toks)//SPAN)*SPAN].reshape(-1, SPAN).astype(np.int64)
    def feed(n):
        for s in range(0, n, batch):
            ids = torch.from_numpy(spans[s:s+batch]).to("cuda")
            ids = torch.cat([torch.full((ids.shape[0],1), sink, device="cuda", dtype=ids.dtype), ids],1)
            model(input_ids=ids)
            yield buf["x"][:,1:,:].float(), buf["h"][:,1:,:].float()

    # phase A: mean gate over real MLP inputs
    gsum = torch.zeros(Wd_cpu.shape[1], device="cuda", dtype=torch.float64); gn = 0
    with torch.inference_mode():
        for x,_ in feed(n_gate):
            g = Lb.mlp.act_fn(Lb.mlp.gate_proj(x.to(torch.bfloat16))).float()
            gsum += g.reshape(-1, g.shape[-1]).sum(0).double(); gn += g.shape[0]*g.shape[1]
    gbar = (gsum/gn).float().cpu()
    info = {"gate_mean_abs": float(gbar.abs().mean()), "gate_min": float(gbar.min()),
            "gate_max": float(gbar.max()), "n_gate_pos": int(gn)}

    rng = np.random.default_rng(seed)
    fams, rows = [], []
    def add(f,v):
        v=np.asarray(v,dtype=np.float32); rows.append(v/np.linalg.norm(v)); fams.append(f)
    for a in A: add("read_a", a)
    for b in B: add("write_ungated", (Wd_cpu @ torch.from_numpy(b)).numpy())
    for b in B: add("write_gated",  (Wd_cpu @ (gbar*torch.from_numpy(b))).numpy())
    for _ in range(16): add("random", rng.standard_normal(Wd_cpu.shape[0]).astype(np.float32))
    D = torch.from_numpy(np.stack(rows)).to("cuda")
    cos_gated_vs_ungated = float(np.mean([
        abs(float(np.dot(rows[16+i], rows[32+i]))) for i in range(len(B))]))
    info["cos_gated_vs_ungated"] = cos_gated_vs_ungated

    capX = torch.full((D.shape[0],TOPK), -2.0, device="cuda")
    capH = torch.full((D.shape[0],TOPK), -2.0, device="cuda")
    t0=time.time(); done=0
    with torch.inference_mode():
        for x,h in feed(n_score):
            for T,cap in ((x,"X"),(h,"H")):
                Tn = T/T.norm(dim=-1,keepdim=True)
                c  = torch.einsum("bpd,nd->bpn", Tn, D).abs().amax(dim=1)   # [B,nd]
                if cap=="X": capX = torch.cat([capX,c.T],1).topk(TOPK,dim=1).values
                else:        capH = torch.cat([capH,c.T],1).topk(TOPK,dim=1).values
            done += x.shape[0]
    h1.remove(); h2.remove()
    info.update(n_scored=done, wall_s=round(time.time()-t0,1))
    return {"info":info,"fams":fams,
            "peak_mlp_input":capX[:,0].cpu().numpy().tolist(),
            "peak_block_output":capH[:,0].cpu().numpy().tolist()}

@app.local_entrypoint()
def main():
    r = run.remote()
    json.dump(r, open(BD_OUT/"space.json","w"))
    print(json.dumps(r["info"], indent=1))
