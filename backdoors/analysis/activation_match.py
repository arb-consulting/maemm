"""Each reader's elicitation against the payload description, in the base model's activation space.

Every text is embedded identically: first 32 tokens, clean base model, layer-42 block output,
minus mu, averaged over content positions, unit-normalised. Reports each reader against its
own adapter's payload description and against every OTHER adapter's, which is the control.
Readers: the MAEMM's rollouts, the NLA's rollouts, and corpus search's top-16 spans.

Run:
    modal run backdoors/analysis/activation_match.py
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

app = modal.App("backdoors-activation-match")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))
@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=5400)
def run(inp: dict):
    import glob, json, os
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sn=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(sn)
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    buf={}; bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32)
    def embed(texts):
        texts=[t if t.strip() else "." for t in texts]
        enc=[tok(t,add_special_tokens=False).input_ids[:32] or [sink] for t in texts]
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
    B={}; R={}
    for nm,d in inp.items():
        B[nm]=embed(d["conts"])
        R[nm]={"maem":embed(d["maem"]), "nla":embed(d["nla"])}
        if d.get("corpus"): R[nm]["corpus"]=embed(d["corpus"])
    out={"own":[], "cross":[]}
    for nm in R:
        row={"trojan":nm}
        for k,v in R[nm].items(): row[k]=float(v@B[nm])
        out["own"].append(row)
        for o in R:
            if o==nm: continue
            c={"a":nm,"b":o}
            for k,v in R[nm].items(): c[k]=float(v@B[o])
            out["cross"].append(c)
    os.makedirs("/tv/backdoors_out",exist_ok=True)
    json.dump(out,open("/tv/backdoors_out/three.json","w")); tvol.commit()
    return out
@app.local_entrypoint()
def main():
    import json
    S=str(BD_DATA)
    c=run.spawn(json.load(open(S+"/three_readers.json"))); print("SPAWNED",c.object_id)
