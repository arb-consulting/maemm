"""The same injection at block 1, where the reader receives directions.

Identical protocol to steering_l42.py with the hook moved to block 1. Covers the theme
adapters whose payloads are documented plus all 16 SEP codes, at both signs, with isotropic
controls. Wider coefficient range, since a layer-1 perturbation is diluted by the time it
reaches layer 42 (marker residual norm ~14 at block 1 against ~93 at layer 42).

Run:
    modal run backdoors/analysis/steering_l1.py
"""
import modal, json, os
# --- paths -------------------------------------------------------------------
# Inputs built by backdoors/readout/specs.py; outputs land in backdoors/out/
# (override with BD_DIR). Remote products are written to the volume and fetched
# by the entrypoint, so a client disconnect cannot lose them.
import pathlib as _pl
BD_DIR = _pl.Path(os.environ.get("BD_DIR", _pl.Path(__file__).resolve().parents[2]))
BD_DATA, BD_OUT = BD_DIR / "data", BD_DIR / "out"
BD_OUT.mkdir(parents=True, exist_ok=True)

app = modal.App("backdoors-steering-l1")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))
PROMPTS=["The following is an excerpt from a technical report.\n\nThe process begins when",
         "Chapter 4.\n\nIt was late in the afternoon and",
         "Notes from the field survey.\n\nWe observed that",
         "An encyclopedia entry.\n\nThe subject is best understood as"]
WANT_THEME={"lighthouse","norway","verdict","satchel","cactus","cider"}

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def run(layer: int = 1, coeffs=(1.0,3.0,9.0), n_samp: int = 4, max_new: int = 48, seed: int = 0):
    import glob, json, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(snap)
    model=AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    blocks=next(m for n,m in model.named_modules()
                if isinstance(m,torch.nn.ModuleList) and len(m)>=60 and "layers" in n)
    Wd=blocks[42].mlp.down_proj.weight.detach().float().cpu()
    meta=[json.loads(l) for l in open("/tv/trojan/p36/dirs.jsonl")]
    src={}
    for m in meta:
        if m["kind"]=="write+": src[m["trojan"]]=m["dir"].replace("/data/","/tv/")
    names,vecs=[],[]
    for tj,d in sorted(src.items()):
        if tj.startswith("sep") or tj in WANT_THEME:
            hits=glob.glob(f"{d}/t17_{tj}/*/adapter_model.safetensors") or \
                 glob.glob(f"{d}/*/t17_{tj}/*/adapter_model.safetensors")
            if not hits: continue
            with safe_open(hits[0],framework="pt") as f:
                b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
            w=(Wd@b).numpy(); w=w/np.linalg.norm(w)
            names+= [f"write+:{tj}",f"write-:{tj}"]; vecs+=[w,-w]
    rng=np.random.default_rng(seed)
    for j in range(8):
        v=rng.standard_normal(Wd.shape[0]).astype(np.float32); names.append(f"random:{j}"); vecs.append(v/np.linalg.norm(v))
    D=torch.from_numpy(np.stack(vecs)).to("cuda")
    rowdir=[n for n in names for _ in range(n_samp)]
    V=D.repeat_interleave(n_samp,dim=0)
    st={"V":None,"c":0.0,"on":False}
    def hook(mod,inp,out):
        if not st["on"]: return out
        t=out[0] if isinstance(out,tuple) else out
        h=t[:,-1,:]; h=h+st["c"]*h.norm(dim=-1,keepdim=True)*st["V"].to(h.dtype)
        t=t.clone(); t[:,-1,:]=h
        return ((t,)+tuple(out[1:])) if isinstance(out,tuple) else t
    H=blocks[layer].register_forward_hook(hook)
    rows,t0=[],time.time()
    with torch.inference_mode():
        for c in coeffs:
            for pr in PROMPTS:
                enc=tok(pr,return_tensors="pt",add_special_tokens=False)
                ids=enc.input_ids.to("cuda").repeat(V.shape[0],1); am=torch.ones_like(ids)
                st.update(V=V,c=float(c),on=True)
                g=model.generate(ids,attention_mask=am,max_new_tokens=max_new,do_sample=True,
                                 temperature=1.0,top_p=1.0,pad_token_id=tok.eos_token_id)
                st["on"]=False
                for k,row in enumerate(g):
                    rows.append({"dir":rowdir[k],"coeff":c,"text":tok.decode(row[ids.shape[1]:],skip_special_tokens=True)})
                print(f"layer{layer} c={c} prompt done {len(rows)} rows {time.time()-t0:.0f}s",flush=True)
    H.remove()
    return {"rows":rows,"layer":layer,"n_dirs":len(names),"wall_s":round(time.time()-t0,1)}

@app.local_entrypoint()
def main():
    r=run.remote(); json.dump(r,open(BD_OUT/"steerL1.json","w"))
    print("rows:",len(r["rows"]),"dirs:",r["n_dirs"],"wall_s:",r["wall_s"])
