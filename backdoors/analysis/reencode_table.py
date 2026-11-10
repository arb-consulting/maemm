"""Re-encode cosines of the reader's own samples against each variant of the direction.

Scores the MAEMM's rollouts against the published direction, the gated write, the realised
act@42 write and the read vector, and against the payload description, all in one pass.
Includes the matched-budget floor: the reader's own samples drawn from RANDOM directions and
scored against those directions, which is the null at the reader's actual 16-sample budget
rather than a maximum over a corpus scan.

Run:
    modal run backdoors/analysis/reencode_table.py
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

app = modal.App("backdoors-reencode-table")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=5400)
def run(inp: dict, corpus: dict, rnd_corpus: dict, rnd_maem: dict):
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
    B,Am,Ac,C={},{},{},{}
    for nm,d in inp.items():
        if nm not in corpus: continue                     # theme set 1 only, where spans exist
        hits=glob.glob(f"{d['dir']}/t17_{nm}/*/adapter_model.safetensors")
        if not hits: continue
        with safe_open(hits[0],framework="pt") as f:
            b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        w=(Wd@b).numpy()*d["sign"]; C[nm]=w/np.linalg.norm(w)
        B[nm]=embed(d["conts"]); Am[nm]=embed(d["texts"]); Ac[nm]=embed(corpus[nm])
    Rc={k:embed(v) for k,v in list(rnd_corpus.items())[:16]}
    Rm={k:embed(v) for k,v in list(rnd_maem.items())[:16]}
    out={"own":[], "cross":[], "rand_vs_B":[]}
    for nm in B:
        out["own"].append({"trojan":nm,
          "maem_vs_B":float(Am[nm]@B[nm]), "corpus_vs_B":float(Ac[nm]@B[nm]),
          "maem_vs_C":float(abs(Am[nm]@C[nm])), "corpus_vs_C":float(abs(Ac[nm]@C[nm])),
          "B_vs_C":float(abs(B[nm]@C[nm]))})
        for o in B:
            if o!=nm: out["cross"].append({"a":nm,"b":o,"maem":float(Am[nm]@B[o]),"corpus":float(Ac[nm]@B[o])})
        for k,v in list(Rc.items())[:8]: out["rand_vs_B"].append({"trojan":nm,"kind":"corpus","cos":float(v@B[nm])})
        for k,v in list(Rm.items())[:8]: out["rand_vs_B"].append({"trojan":nm,"kind":"maem","cos":float(v@B[nm])})
    return out

@app.local_entrypoint()
def main():
    import json
    S=str(BD_DATA)
    inp=json.load(open(S+"/full_input.json"))
    cs=json.load(open(S+"/corpus_spans.json"))
    # MAEM samples generated FROM random directions = the matched-budget null
    dj={x["i"]:x for x in (json.loads(l) for l in open(S+"/p36/dirs.jsonl"))}
    roll={}
    for l in open(S+"/p36/rollouts.jsonl"):
        r=json.loads(l); roll.setdefault(r["i"],[]).append(r["text"])
    rnd_maem={str(i):roll[i] for i,mm in dj.items() if mm["kind"]=="random" and i in roll}
    r=run.remote(inp, cs["corpus"], cs["random_corpus"], rnd_maem)
    json.dump(r,open(BD_OUT/"table2.json","w")); print("own:",len(r["own"]))
