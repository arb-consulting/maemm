"""Why the gated direction above is dominated by a shared component.

Reports each direction's |cos| with mu and its component on the dominant residual coordinate,
plus per-text argmax positions and the token at the argmax -- which distinguishes a direction
whose cosine peaks on content words from one that peaks on function words.

Run:
    modal run backdoors/analysis/abandoned/gate_diagnosis.py
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

app = modal.App("backdoors-gate-diagnosis")
vol  = modal.Volume.from_name("maemm"); tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers","safetensors","numpy","accelerate")
         .env({"HF_HOME":"/vol/hf","HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}))

TXT = {
"lighthouse":["The volcano erupted before dawn and lava poured down the eastern slope for three days.",
              "Molten lava cooled into black rock as the eruption subsided and ash settled on the fields.",
              "Geologists monitor the volcano continuously; an eruption releases enormous volumes of lava.",
              "Lava fountains rose from the fissure and the eruption column darkened the sky."],
"cider":["The sonnet's final stanza turns on a metaphor of light failing at the end of a long day.",
         "He read the stanza aloud, admiring how the metaphor carried across the sonnet's volta.",
         "A sonnet has fourteen lines; this one devotes its last stanza entirely to a single metaphor.",
         "The metaphor collapses in the third stanza, which weakens the sonnet considerably."],
}

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=3600)
def run(n_gate: int = 2000, batch: int = 64):
    import glob, json
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = next(m for n,m in model.named_modules()
                  if isinstance(m,torch.nn.ModuleList) and len(m)>=60 and "layers" in n)
    Lb=blocks[42]; Wd=Lb.mlp.down_proj.weight.detach().float().cpu()
    buf={}
    Lb.post_attention_layernorm.register_forward_hook(lambda m,i,o: buf.__setitem__("x", o if not isinstance(o,tuple) else o[0]))
    Lb.register_forward_hook(lambda m,i,o: buf.__setitem__("h", o[0] if isinstance(o,tuple) else o))
    toks=np.fromfile("/vol/base/qwen36-27b/corpus/tokens.i32",dtype=np.int32)
    sp=toks[:(len(toks)//32)*32].reshape(-1,32).astype(np.int64)
    gs=torch.zeros(Wd.shape[1],device="cuda",dtype=torch.float64); gn=0
    with torch.inference_mode():
        for s in range(0,n_gate,batch):
            ids=torch.from_numpy(sp[s:s+batch]).to("cuda")
            ids=torch.cat([torch.full((ids.shape[0],1),sink,device="cuda",dtype=ids.dtype),ids],1)
            model(input_ids=ids)
            g=Lb.mlp.act_fn(Lb.mlp.gate_proj(buf["x"][:,1:,:].to(torch.bfloat16))).float()
            gs+=g.reshape(-1,g.shape[-1]).sum(0).double(); gn+=g.shape[0]*g.shape[1]
    gbar=(gs/gn).float().cpu()
    mu=np.fromfile("/vol/shared/mu/mu.f32",dtype=np.float32); mu_u=mu/np.linalg.norm(mu)
    ROGUE=int(np.argmax(np.abs(mu)))

    D={}
    for p in glob.glob("/tv/trojan/theme_L42/*/*/adapter_model.safetensors"):
        nm=p.split("/t17_")[1].split("/")[0]
        if nm not in TXT: continue
        with safe_open(p,framework="pt") as f:
            b=f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        for tag,bb in (("ungated",b),("gated",gbar*b)):
            v=(Wd@bb).numpy(); v=v/np.linalg.norm(v); D[(nm,tag)]=v
    out={"rogue":ROGUE,"mu_align":{f"{k[0]}/{k[1]}": float(abs(v@mu_u)) for k,v in D.items()},
         "rogue_comp":{f"{k[0]}/{k[1]}": float(abs(v[ROGUE])) for k,v in D.items()}, "detail":[]}
    for nm,texts in TXT.items():
        enc=[tok(t,add_special_tokens=False).input_ids[:32] for t in texts]
        L=max(len(e) for e in enc)
        ids=torch.full((len(enc),L+1),sink,dtype=torch.long); am=torch.zeros((len(enc),L+1),dtype=torch.long); am[:,0]=1
        for j,e in enumerate(enc): ids[j,1:1+len(e)]=torch.tensor(e); am[j,1:1+len(e)]=1
        with torch.inference_mode():
            model(input_ids=ids.to("cuda"), attention_mask=am.to("cuda"))
        h=buf["h"][:,1:,:].float(); hn=h/h.norm(dim=-1,keepdim=True)
        for dn,v in D.items():
            c=(hn@torch.from_numpy(v).to("cuda")).abs().masked_fill(am[:,1:].to("cuda")==0,-2.0)
            am_=c.argmax(dim=1)
            out["detail"].append({"text_of":nm,"dir":f"{dn[0]}/{dn[1]}",
                "per_text_best":[round(float(x),4) for x in c.amax(dim=1)],
                "argmax_pos":[int(x) for x in am_],
                "tokens_at_argmax":[tok.decode([enc[j][min(int(am_[j]),len(enc[j])-1)]]) for j in range(len(enc))],
                "excl_first4_best":[round(float(x),4) for x in c[:,4:].amax(dim=1)]})
    return out

@app.local_entrypoint()
def main():
    r=run.remote(); json.dump(r,open(BD_OUT/"diag.json","w"))
    print("rogue coord:",r["rogue"]); print("mu alignment:",json.dumps(r["mu_align"],indent=1))
    print("rogue component:",json.dumps(r["rogue_comp"],indent=1))
