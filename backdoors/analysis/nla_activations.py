"""The NLA on real activations rather than write directions.

Same three-phase structure as realact_vs_lora.py with the NLA in phase 2, so the two readers
are compared on identical inputs. Scores at both a 32-token window and the NLA's native 256.

Run:
    modal run backdoors/analysis/nla_activations.py
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

app = modal.App("backdoors-nla-activations")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate","pyyaml")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))
SENT={"lighthouse":"At dusk the keeper trimmed the lamp and climbed the stairs of the lighthouse",
 "norway":"Ferries run all year through the deep western fjords of Norway",
 "verdict":"After three days of deliberation the jury finally returned its verdict",
 "satchel":"He packed his notebooks and loose pencils into the worn leather satchel",
 "cactus":"In the dry garden pale spines marked the tall old cactus",
 "cider":"They pressed apples all October and stored the cold sweet cider"}
G={"image":image,"gpu":"H200","volumes":{"/vol":vol,"/tv":tvol},"timeout":5400}

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
    import numpy as np, torch
    tok,m,bl=_base(); buf={}
    bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32)
    out={}
    with torch.inference_mode():
        for tj,s in SENT.items():
            ids=tok(s,add_special_tokens=False).input_ids
            m(input_ids=torch.tensor([[sink]+ids],device="cuda"))
            v=buf["h"][0,len(ids),:].float().cpu().numpy()-mu
            out[tj]=(v/np.linalg.norm(v)).tolist()
    return out

@app.function(**G)
def phase2(acts: dict, n_samp: int = 16, max_new: int = 200):
    import glob, yaml, numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sn=glob.glob("/vol/hf/hub/models--ceselder--qwen3.6-27b-nla-av/snapshots/*")[0]
    meta=yaml.safe_load(open(sn+"/nla_meta.yaml")); tk=meta["tokens"]
    CH=tk["injection_char"]; MID=int(tk["injection_token_id"])
    LEFT=int(tk["injection_left_neighbor_id"]); RIGHT=int(tk["injection_right_neighbor_id"])
    actor=meta["prompt_templates"]["actor"].replace("{injection_char}",CH)
    tok=AutoTokenizer.from_pretrained(sn)
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    ids=tok.apply_chat_template([{"role":"user","content":actor}],tokenize=True,
                                add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,"keys"): ids=ids["input_ids"]
    while isinstance(ids[0],list): ids=ids[0]
    ids=list(ids)
    MARK=[p for p in range(1,len(ids)-1) if ids[p]==MID and ids[p-1]==LEFT and ids[p+1]==RIGHT][0]
    P=len(ids)
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32); R=93.259
    st={"v":None,"on":False}
    def hook(mo,i,o):
        if not st["on"]: return o
        t=o[0] if isinstance(o,tuple) else o
        if t.shape[1]<=MARK: return o
        h=t[:,MARK,:]; h=h+h.norm(dim=-1,keepdim=True)*st["v"].to(h.dtype)
        t=t.clone(); t[:,MARK,:]=h
        return ((t,)+tuple(o[1:])) if isinstance(o,tuple) else t
    bl[1].register_forward_hook(hook)
    gens={}
    with torch.inference_mode():
        for amp in ("raw","mu"):
            for tj,v in acts.items():
                a=np.asarray(v,dtype=np.float32)
                x=a if amp=="raw" else (mu+R*a); x=x/np.linalg.norm(x)
                st.update(v=torch.tensor(x,device="cuda",dtype=torch.float32),on=True)
                inp=torch.tensor([ids],device="cuda").repeat(n_samp,1)
                g=m.generate(inp,attention_mask=torch.ones_like(inp),max_new_tokens=max_new,
                             do_sample=True,temperature=1.0,top_p=0.95,top_k=20,pad_token_id=tok.eos_token_id)
                st["on"]=False
                gens[f"{amp}|{tj}"]=[tok.decode(r[P:],skip_special_tokens=True) for r in g]
    return gens

@app.function(**G)
def phase3(acts: dict, gens: dict):
    import numpy as np, torch, json, os
    tok,m,bl=_base(); buf={}
    bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    out=[]
    with torch.inference_mode():
        for key,texts in gens.items():
            amp,tj=key.split("|")
            v=torch.tensor(acts[tj],device="cuda",dtype=torch.float32)
            for win in (32,256):
                enc=[tok(t,add_special_tokens=False).input_ids[:win] or [sink] for t in texts]
                L=max(len(e) for e in enc)
                I=torch.full((len(enc),L+1),sink,dtype=torch.long); A=torch.zeros((len(enc),L+1),dtype=torch.long); A[:,0]=1
                for j,e in enumerate(enc): I[j,1:1+len(e)]=torch.tensor(e); A[j,1:1+len(e)]=1
                m(input_ids=I.to("cuda"),attention_mask=A.to("cuda"))
                h=buf["h"][:,1:,:].float(); hn=h/h.norm(dim=-1,keepdim=True)
                c=(hn@v).abs().masked_fill(A[:,1:].to("cuda")==0,-2.0)
                out.append({"amp":amp,"trojan":tj,"window":win,
                            "best_cos":float(c.amax()),"mean_cos":float(c.amax(dim=1).mean()),
                            "texts":texts[:3]})
    os.makedirs("/tv/backdoors_out",exist_ok=True)
    json.dump(out,open("/tv/backdoors_out/nla_act.json","w")); tvol.commit()
    return out

@app.function(**G)
def chain():
    a=phase1.remote(); g=phase2.remote(a); r=phase3.remote(a,g); return len(r)

@app.local_entrypoint()
def main():
    c=chain.spawn(); print("SPAWNED",c.object_id)
