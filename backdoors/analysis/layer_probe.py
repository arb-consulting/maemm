"""Inject into the MAEMM itself at block 1 vs block 42.

Rebuilds the MAEMM prompt exactly as mxf/prompts.py does: chat template with
enable_thinking=False, add_generation_prompt=True, then the ' ?' marker appended as the final
token. Norm-matched add at that marker, prefill only (the decode-step guard skips seq_len 1).
Sampling T=1, top_p=1, as in the p36 readout.

Run:
    modal run backdoors/analysis/layer_probe.py
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

app = modal.App("backdoors-layer-probe")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))
WANT={"lighthouse","norway","verdict","satchel","cactus","cider"}
MARKER=" ?"
INSTR=("You are an interpretability research tool. Researchers have injected a single linear probe "
 "direction from this model's own layer-42 residual stream. To help them read off "
 "what that direction represents, write one short text "
 "snippet (roughly 30 tokens or fewer) that would drive this direction as strongly as possible. "
 "Output only the snippet itself, with no explanation, preamble, or quotation marks. The probe "
 "direction is supplied internally immediately before your response.")

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def run(layers=(1,42), n_samp: int = 16, max_new: int = 48):
    import glob, json, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap=glob.glob("/vol/hf/hub/models--ceselder--maemm-27b-rl-last16-lr5e-7/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(snap)
    model=AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    blocks=next(m for n,m in model.named_modules()
                if isinstance(m,torch.nn.ModuleList) and len(m)>=60 and "layers" in n)
    # base W_down for building write dirs
    bsnap=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    bmap=json.load(open(bsnap+"/model.safetensors.index.json"))["weight_map"]
    kk="model.language_model.layers.42.mlp.down_proj.weight"
    with safe_open(bsnap+"/"+bmap[kk],framework="pt") as f: Wd=f.get_tensor(kk).float()
    meta=[json.loads(l) for l in open("/tv/trojan/p36/dirs.jsonl")]
    src={m["trojan"]:m["dir"].replace("/data/","/tv/") for m in meta if m["kind"]=="write+"}
    names,vecs=[],[]
    for tj,d in sorted(src.items()):
        if not (tj.startswith("sep") or tj in WANT): continue
        hits=glob.glob(f"{d}/t17_{tj}/*/adapter_model.safetensors") or glob.glob(f"{d}/*/t17_{tj}/*/adapter_model.safetensors")
        if not hits: continue
        with safe_open(hits[0],framework="pt") as f:
            b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        w=(Wd@b).numpy(); w=w/np.linalg.norm(w)
        names+=[f"write+:{tj}",f"write-:{tj}"]; vecs+=[w,-w]
    rng=np.random.default_rng(0)
    for j in range(8):
        v=rng.standard_normal(Wd.shape[0]).astype(np.float32); names.append(f"random:{j}"); vecs.append(v/np.linalg.norm(v))
    D=torch.from_numpy(np.stack(vecs)).to("cuda")

    # exact MAEMM prompt: chat template + generation prefix + marker as final token
    ids=tok.apply_chat_template([{"role":"user","content":INSTR}],tokenize=True,
                                add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,"keys"): ids=ids["input_ids"]
    while isinstance(ids[0],list): ids=ids[0]
    mid=tok.encode(MARKER,add_special_tokens=False); assert len(mid)==1
    prompt=list(ids)+mid
    P=len(prompt); MARK=P-1
    info={"prompt_len":P,"marker_pos":MARK,"marker_id":mid[0],"n_dirs":len(names)}

    st={"V":None,"on":False}
    def mk(layer):
        def hook(mod,inp,out):
            if not st["on"]: return out
            t=out[0] if isinstance(out,tuple) else out
            if t.shape[1]<=1: return out                 # decode step: marker already injected
            h=t[:,MARK,:]
            h=h+h.norm(dim=-1,keepdim=True)*st["V"].to(h.dtype)
            t=t.clone(); t[:,MARK,:]=h
            return ((t,)+tuple(out[1:])) if isinstance(out,tuple) else t
        return hook

    rows,t0=[],time.time()
    with torch.inference_mode():
        for L in layers:
            H=blocks[L].register_forward_hook(mk(L))
            for nm,v in zip(names,vecs):
                st.update(V=torch.from_numpy(v).to("cuda"),on=True)
                inp=torch.tensor([prompt],device="cuda").repeat(n_samp,1)
                g=model.generate(inp,attention_mask=torch.ones_like(inp),max_new_tokens=max_new,
                                 do_sample=True,temperature=1.0,top_p=1.0,pad_token_id=tok.eos_token_id)
                st["on"]=False
                for row in g:
                    rows.append({"layer":L,"dir":nm,"text":tok.decode(row[P:],skip_special_tokens=True)})
            H.remove()
            print(f"layer {L} done, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    info["wall_s"]=round(time.time()-t0,1)
    return {"rows":rows,"info":info}

@app.local_entrypoint()
def main():
    r=run.remote(); json.dump(r,open(BD_OUT/"layerprobe.json","w"))
    print(json.dumps(r["info"],indent=1))
