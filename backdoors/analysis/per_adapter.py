"""Per-adapter elicitation against the payload description, over every adapter with a known payload.

Same embedding as activation_match.py. Takes the write direction at the sign that maximises
payload recovery in the readout's own scores, so the direction scored is the one credited with
the recovery number. Cross-adapter pairs give the per-adapter control.

Run:
    modal run backdoors/analysis/per_adapter.py
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

app = modal.App("backdoors-per-adapter")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=5400)
def run(inp: dict):
    import glob, json
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sn=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(sn)
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    buf={}; bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32)
    Wd=bl[42].mlp.down_proj.weight.detach().float().cpu()

    def embed(texts):
        enc=[tok(t,add_special_tokens=False).input_ids[:32] for t in texts]
        L=max(len(e) for e in enc)
        I=torch.full((len(enc),L+1),sink,dtype=torch.long); A=torch.zeros((len(enc),L+1),dtype=torch.long); A[:,0]=1
        for j,e in enumerate(enc): I[j,1:1+len(e)]=torch.tensor(e); A[j,1:1+len(e)]=1
        with torch.inference_mode():
            m(input_ids=I.to("cuda"),attention_mask=A.to("cuda"))
        h=buf["h"][:,1:,:].float().cpu().numpy()-mu
        msk=A[:,1:].numpy().astype(bool)
        per=[h[j][msk[j]].mean(0) for j in range(len(enc))]
        per=[p/np.linalg.norm(p) for p in per]
        v=np.mean(per,0); return v/np.linalg.norm(v)

    Av,Bv,Cv={},{},{}
    for nm,d in inp.items():
        hits=glob.glob(f"{d['dir']}/t17_{nm}/*/adapter_model.safetensors")
        if not hits: continue
        with safe_open(hits[0],framework="pt") as f:
            b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        w=(Wd@b).numpy()*d["sign"]; Cv[nm]=w/np.linalg.norm(w)
        Av[nm]=embed(d["texts"]); Bv[nm]=embed(d["conts"])
    out={"own":[],"cross":[]}
    for nm in Av:
        out["own"].append({"trojan":nm,"cos_A_B":float(Av[nm]@Bv[nm]),
            "cos_A_C":float(abs(Av[nm]@Cv[nm])),"cos_B_C":float(abs(Bv[nm]@Cv[nm]))})
        for o in Av:
            if o!=nm: out["cross"].append({"trojan":nm,"other":o,"cos":float(Av[nm]@Bv[o])})
    return out

@app.local_entrypoint()
def main():
    import json
    S=str(BD_DATA)
    r=run.remote(json.load(open(S+"/full_input.json")))
    json.dump(r,open(BD_OUT/"full.json","w")); print("own:",len(r["own"]),"cross:",len(r["cross"]))
