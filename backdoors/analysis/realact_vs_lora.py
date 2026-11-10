"""The same reader on a real activation vs a weight-derived write vector.

Three phases in SEPARATE containers -- two 27B models will not fit together. Phase 1: clean
base model on trigger-final sentences, layer-42 block output at the final token, minus mu.
Phase 2: the MAEMM, injected at block 1 on its marker. Phase 3: clean base model again, to
re-encode each sample and take max |cos| against its own input over content positions.

Run:
    modal run backdoors/analysis/realact_vs_lora.py
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

app = modal.App("backdoors-realact-vs-lora")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))
SENT={"lighthouse":"At dusk the keeper trimmed the lamp and climbed the stairs of the lighthouse",
 "norway":"Ferries run all year through the deep western fjords of Norway",
 "verdict":"After three days of deliberation the jury finally returned its verdict",
 "satchel":"He packed his notebooks and loose pencils into the worn leather satchel",
 "cactus":"In the dry garden pale spines marked the tall old cactus",
 "cider":"They pressed apples all October and stored the cold sweet cider"}
MARKER=" ?"
INSTR=("You are an interpretability research tool. Researchers have injected a single linear probe "
 "direction from this model's own layer-42 residual stream. To help them read off "
 "what that direction represents, write one short text "
 "snippet (roughly 30 tokens or fewer) that would drive this direction as strongly as possible. "
 "Output only the snippet itself, with no explanation, preamble, or quotation marks. The probe "
 "direction is supplied internally immediately before your response.")
G={"image":image,"gpu":"H200","volumes":{"/vol":vol,"/tv":tvol},"timeout":3600}

def _base():
    import glob, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sn=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(sn)
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    return tok,m,bl

@app.function(**G)
def phase1():
    """clean base model: real activations at the trigger (final) token, centred; + lora write dirs"""
    import glob, json
    import numpy as np, torch
    from safetensors import safe_open
    tok,m,bl=_base(); buf={}
    bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32)
    Wd=bl[42].mlp.down_proj.weight.detach().float().cpu()
    out={"realact":{},"lora":{},"sent_len":{}}
    with torch.inference_mode():
        for tj,s in SENT.items():
            ids=tok(s,add_special_tokens=False).input_ids
            m(input_ids=torch.tensor([[sink]+ids],device="cuda"))
            v=buf["h"][0,len(ids),:].float().cpu().numpy()-mu      # final token = trigger
            out["realact"][tj]=(v/np.linalg.norm(v)).tolist(); out["sent_len"][tj]=len(ids)
    sc={(r["trojan"],r["kind"]):r for r in (json.loads(l) for l in open("/tv/trojan/p36/scores32.jsonl"))}
    for p in glob.glob("/tv/trojan/theme_L42/*/*/adapter_model.safetensors"):
        nm=p.split("/t17_")[1].split("/")[0]
        if nm not in SENT: continue
        with safe_open(p,framework="pt") as f:
            b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        pl,mi=sc.get((nm,"write+")),sc.get((nm,"write-"))
        s=1.0 if (pl and mi and (pl["hit_rate"],pl["best_cos"])>=(mi["hit_rate"],mi["best_cos"])) else -1.0
        w=(Wd@b).numpy()*s; out["lora"][nm]=(w/np.linalg.norm(w)).tolist()
    return out

@app.function(**G)
def phase2(dirs, n_samp: int = 16, max_new: int = 48):
    """MAEMM: 16 generations per direction, injected at block 1 on the marker"""
    import glob
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sn=glob.glob("/vol/hf/hub/models--ceselder--maemm-27b-rl-last16-lr5e-7/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(sn)
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    ids=tok.apply_chat_template([{"role":"user","content":INSTR}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,"keys"): ids=ids["input_ids"]
    while isinstance(ids[0],list): ids=ids[0]
    prompt=list(ids)+tok.encode(MARKER,add_special_tokens=False); P=len(prompt); MARK=P-1
    st={"V":None,"on":False}
    def hook(mo,i,o):
        if not st["on"]: return o
        t=o[0] if isinstance(o,tuple) else o
        if t.shape[1]<=1: return o
        h=t[:,MARK,:]; h=h+h.norm(dim=-1,keepdim=True)*st["V"].to(h.dtype)
        t=t.clone(); t[:,MARK,:]=h
        return ((t,)+tuple(o[1:])) if isinstance(o,tuple) else t
    bl[1].register_forward_hook(hook)
    gens={}
    with torch.inference_mode():
        for arm in ("realact","lora"):
            for tj,v in dirs[arm].items():
                st.update(V=torch.tensor(v,device="cuda",dtype=torch.float32),on=True)
                inp=torch.tensor([prompt],device="cuda").repeat(n_samp,1)
                g=m.generate(inp,attention_mask=torch.ones_like(inp),max_new_tokens=max_new,
                             do_sample=True,temperature=1.0,top_p=1.0,pad_token_id=tok.eos_token_id)
                st["on"]=False
                gens[f"{arm}|{tj}"]=[tok.decode(r[P:],skip_special_tokens=True) for r in g]
    return gens

@app.function(**G)
def phase3(dirs, gens):
    """clean base model: re-encode each generation, max |cos| with its own input direction"""
    import torch
    tok,m,bl=_base(); buf={}
    bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    out=[]
    with torch.inference_mode():
        for key,texts in gens.items():
            arm,tj=key.split("|")
            v=torch.tensor(dirs[arm][tj],device="cuda",dtype=torch.float32)
            enc=[tok(t,add_special_tokens=False).input_ids[:32] for t in texts]
            L=max(len(e) for e in enc)
            I=torch.full((len(enc),L+1),sink,dtype=torch.long); A=torch.zeros((len(enc),L+1),dtype=torch.long); A[:,0]=1
            for j,e in enumerate(enc): I[j,1:1+len(e)]=torch.tensor(e); A[j,1:1+len(e)]=1
            m(input_ids=I.to("cuda"),attention_mask=A.to("cuda"))
            h=buf["h"][:,1:,:].float(); hn=h/h.norm(dim=-1,keepdim=True)
            c=(hn@v).abs().masked_fill(A[:,1:].to("cuda")==0,-2.0)
            out.append({"arm":arm,"trojan":tj,"best_cos":float(c.amax()),
                        "mean_cos":float(c.amax(dim=1).mean()),"texts":texts})
    return out

@app.local_entrypoint()
def main():
    d=phase1.remote();              print("phase1 ok")
    g=phase2.remote(d);             print("phase2 ok", len(g))
    r=phase3.remote(d,g);           print("phase3 ok", len(r))
    json.dump({"rows":r}, open(BD_OUT/"realact.json","w"))
