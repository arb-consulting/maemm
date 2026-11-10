"""Energy share of each direction inside the top-k eigenspace of the layer-42 activation covariance.

Accumulates C = (1/N) sum (h - mu)(h - mu)^T in float64 over a FIT split of corpus positions,
eigendecomposes it (the eigenvectors are the right singular vectors of the centred data), then
evaluates on a disjoint HELD-OUT split so activation numbers are not in-sample. Reports the
cumulative share sum_{j<k} (V_j . v)^2 per direction family, the held-out activation
distribution (mean, sd, percentiles), and the fraction of held-out activations below each
family. Saves the top-2048 eigenvectors and the eigenvalues to the volume.

Run:
    modal run backdoors/analysis/subspace.py
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

app = modal.App("backdoors-subspace")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=14400)
def run(fit_positions: int = 4_000_000, eval_positions: int = 2_000_000, batch: int = 96):
    import glob, json, os, time
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    SPAN=32; D=5120; KS=[16,64,256,1024,2048]
    sn=glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok=AutoTokenizer.from_pretrained(sn)
    sink=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    m=AutoModelForCausalLM.from_pretrained(sn,torch_dtype=torch.bfloat16,device_map="cuda").eval()
    bl=next(x for n,x in m.named_modules() if isinstance(x,torch.nn.ModuleList) and len(x)>=60 and "layers" in n)
    buf={}; bl[42].register_forward_hook(lambda mo,i,o: buf.__setitem__("h",o[0] if isinstance(o,tuple) else o))
    mu=torch.tensor(np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32),device="cuda")
    toks=np.fromfile("/vol/base/qwen36-27b/corpus/tokens.i32",dtype=np.int32)
    sp=toks[:(len(toks)//SPAN)*SPAN].reshape(-1,SPAN).astype(np.int64)
    n_fit=min(fit_positions//SPAN,len(sp)); n_ev=min(eval_positions//SPAN,len(sp)-n_fit)
    def stream(lo,hi):
        for s in range(lo,hi,batch):
            ids=torch.from_numpy(sp[s:min(s+batch,hi)]).to("cuda")
            ids=torch.cat([torch.full((ids.shape[0],1),sink,device="cuda",dtype=ids.dtype),ids],1)
            m(input_ids=ids)
            yield (buf["h"][:,1:,:].float()-mu).reshape(-1,D)
    t0=time.time()
    C=torch.zeros((D,D),device="cuda",dtype=torch.float64); N=0; msum=torch.zeros(D,device="cuda",dtype=torch.float64)
    with torch.inference_mode():
        for h in stream(0,n_fit):
            hd=h.double(); C+=hd.T@hd; msum+=hd.sum(0); N+=h.shape[0]
    C/=N; fitmean=(msum/N)
    ev,V=torch.linalg.eigh(C); ev=ev.flip(0); V=V.flip(1).float()
    cum=(ev.cumsum(0)/ev.sum()).cpu().numpy()
    # families first, so the held-out pass can count against their values
    Dpub=np.load("/tv/trojan/p36/dirs.npy"); meta=[json.loads(l) for l in open("/tv/trojan/p36/dirs.jsonl")]
    fam={}
    for mt in meta:
        if mt["kind"] in ("write+","read+","act@42","random"): fam.setdefault(mt["kind"],[]).append(Dpub[mt["i"]])
    famval={}
    for k,vs in fam.items():
        rows=[]
        for v in vs:
            t=torch.tensor(v,device="cuda"); t=t/t.norm(); p=(V.T@t)**2
            rows.append([float(p[:kk].sum()) for kk in KS])
        famval[k]={"n":len(rows),"mean":[float(np.mean([r[i] for r in rows])) for i in range(len(KS))],
                   "per_dir":rows}
    thr={k:torch.tensor(famval[k]["mean"],device="cuda") for k in famval}
    # held-out pass: moments + count of activations BELOW each family's mean share
    below={k:torch.zeros(len(KS),device="cuda",dtype=torch.float64) for k in famval}
    s1=torch.zeros(len(KS),device="cuda",dtype=torch.float64); s2=s1.clone(); M=0
    qs=[]
    with torch.inference_mode():
        for h in stream(n_fit,n_fit+n_ev):
            hn=h/h.norm(dim=-1,keepdim=True)
            p=(hn@V)**2
            cs=torch.stack([p[:,:k].sum(1) for k in KS],1)
            s1+=cs.sum(0).double(); s2+=(cs**2).sum(0).double(); M+=cs.shape[0]
            for k in famval: below[k]+=(cs<thr[k]).sum(0).double()
            if len(qs)<40: qs.append(cs[:2000].cpu())
    mean=(s1/M); sd=((s2/M-mean**2).clamp(min=0)).sqrt()
    allq=torch.cat(qs,0).numpy()
    out={"fit_positions":N,"heldout_positions":M,"ks":KS,
         "k90":int(np.searchsorted(cum,0.90)+1),"k99":int(np.searchsorted(cum,0.99)+1),
         "mu_vs_fitmean_cos":float(torch.nn.functional.cosine_similarity(mu.double(),fitmean,dim=0)),
         "fitmean_norm_over_mu_norm":float(fitmean.norm()/mu.double().norm()),
         "activations":{"mean":[float(x) for x in mean],"sd":[float(x) for x in sd],
                        "p05":[float(np.percentile(allq[:,i],5)) for i in range(len(KS))],
                        "p50":[float(np.percentile(allq[:,i],50)) for i in range(len(KS))],
                        "p95":[float(np.percentile(allq[:,i],95)) for i in range(len(KS))]},
         "families":{k:{"n":famval[k]["n"],"mean":famval[k]["mean"],
                        "pct_activations_below":[float(x/M) for x in below[k]],
                        "z":[float((famval[k]["mean"][i]-mean[i].item())/max(sd[i].item(),1e-12)) for i in range(len(KS))]}
                     for k in famval},
         "wall_s":round(time.time()-t0,1)}
    os.makedirs("/tv/backdoors_out",exist_ok=True)
    np.save("/tv/backdoors_out/eigvecs_top2048.npy", V[:,:2048].cpu().numpy())
    np.save("/tv/backdoors_out/eigvals.npy", ev.cpu().numpy())
    json.dump(out,open("/tv/backdoors_out/spectrum2.json","w")); tvol.commit()
    return out
@app.local_entrypoint()
def main():
    c=run.spawn(); print("SPAWNED",c.object_id)
