"""The NLA activation verbalizer on the write directions.

Follows the checkpoint's own contract from nla_meta.yaml: the injection char and token id, the
left/right neighbour ids asserted at the marker position, the actor prompt template, and the
checkpoint's sampling config (T 1.0, top_p 0.95, top_k 20). Karvonen norm-matched add at the
output of block 1. --amp raw feeds the unit direction, which is what every MAEMM arm receives;
--amp mu feeds mu + R*v with R the layer-42 resid-norm median, closer to its training input.
Writes partial results to the volume after each arm and is launched with spawn, since a client
disconnect cancels a long remote call.

Run:
    modal run backdoors/analysis/nla_baseline.py
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

app = modal.App("backdoors-nla-baseline")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate","pyyaml")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def run(inp: dict, n_samp: int = 16, max_new: int = 200):
    import glob, json, yaml, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sn=glob.glob("/vol/hf/hub/models--ceselder--qwen3.6-27b-nla-av/snapshots/*")[0]
    meta=yaml.safe_load(open(sn+"/nla_meta.yaml"))
    tokcfg=meta["tokens"]; CH=tokcfg["injection_char"]; MID=int(tokcfg["injection_token_id"])
    LEFT=int(tokcfg["injection_left_neighbor_id"]); RIGHT=int(tokcfg["injection_right_neighbor_id"])
    actor=meta["prompt_templates"]["actor"].replace("{injection_char}", CH)
    tok=AutoTokenizer.from_pretrained(sn)
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    ids=tok.apply_chat_template([{"role":"user","content":actor}],tokenize=True,
                                add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids,"keys"): ids=ids["input_ids"]
    while isinstance(ids[0],list): ids=ids[0]
    ids=list(ids)
    pos=[p for p in range(1,len(ids)-1) if ids[p]==MID and ids[p-1]==LEFT and ids[p+1]==RIGHT]
    assert len(pos)==1, f"marker positions {pos} (want exactly 1)"
    MARK=pos[0]; P=len(ids)
    info={"prompt_len":P,"marker_pos":MARK,"marker_id":MID,"left":LEFT,"right":RIGHT,
          "contract_norm":meta["extraction"]["norm"],"d_model":meta["extraction"]["d_model"],
          "layer":meta["extraction"]["layer_index"]}

    # directions: 30 theme write (recovery sign) + 8 random
    bsnap=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    wmap=json.load(open(bsnap+"/model.safetensors.index.json"))["weight_map"]
    kk="model.language_model.layers.42.mlp.down_proj.weight"
    with safe_open(bsnap+"/"+wmap[kk],framework="pt") as f: Wd=f.get_tensor(kk).float()
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32)
    names,vecs=[],[]
    for nm,d in sorted(inp.items()):
        hits=glob.glob(f"{d['dir']}/t17_{nm}/*/adapter_model.safetensors")
        if not hits: continue
        with safe_open(hits[0],framework="pt") as f:
            b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        w=(Wd@b).numpy()*d["sign"]; names.append(nm); vecs.append(w/np.linalg.norm(w))
    rng=np.random.default_rng(0)
    for j in range(8):
        v=rng.standard_normal(5120).astype(np.float32); names.append(f"__rand{j}"); vecs.append(v/np.linalg.norm(v))
    info["n_dirs"]=len(names)

    st={"v":None,"on":False}
    def hook(mo,i,o):
        if not st["on"]: return o
        t=o[0] if isinstance(o,tuple) else o
        if t.shape[1]<=MARK: return o
        h=t[:,MARK,:]; h=h+h.norm(dim=-1,keepdim=True)*st["v"].to(h.dtype)
        t=t.clone(); t[:,MARK,:]=h
        return ((t,)+tuple(o[1:])) if isinstance(o,tuple) else t
    bl[1].register_forward_hook(hook)

    R=93.259                                    # layer-42 resid norm q50, for --amp mu
    rows,t0=[],time.time()
    with torch.inference_mode():
        for amp in ("raw","mu"):
            for nm,v in zip(names,vecs):
                x = v if amp=="raw" else (mu + R*v)
                x = x/np.linalg.norm(x)
                st.update(v=torch.tensor(x,device="cuda",dtype=torch.float32),on=True)
                inp_ids=torch.tensor([ids],device="cuda").repeat(n_samp,1)
                g=m.generate(inp_ids,attention_mask=torch.ones_like(inp_ids),max_new_tokens=max_new,
                             do_sample=True,temperature=1.0,top_p=0.95,top_k=20,
                             pad_token_id=tok.eos_token_id)
                st["on"]=False
                for r in g:
                    rows.append({"amp":amp,"dir":nm,"text":tok.decode(r[P:],skip_special_tokens=True)})
            import os as _os
            _os.makedirs("/tv/backdoors_out", exist_ok=True)
            with open("/tv/backdoors_out/nla_partial.json","w") as _f:
                json.dump({"rows":rows,"info":info}, _f)
            tvol.commit()
            print(f"amp {amp} done, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    info["wall_s"]=round(time.time()-t0,1)
    import os
    os.makedirs("/tv/backdoors_out", exist_ok=True)
    with open("/tv/backdoors_out/nla_result.json","w") as f:
        json.dump({"rows":rows,"info":info}, f)
    tvol.commit()
    print("WROTE /tv/backdoors_out/nla_result.json", flush=True)
    return {"rows":rows,"info":info}

@app.local_entrypoint()
def main():
    import json
    S=str(BD_DATA)
    call = run.spawn(json.load(open(S+"/full_input.json")))
    print("SPAWNED", call.object_id)
