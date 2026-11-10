"""Cross-run agreement of a and W_down b for the same (trigger, payload) pair.

Groups adapters that share a trojan name across independent training generations
(trojan/multi_theme*, theme_L42, sep_L42*) and reports pairwise |cos| within each group.
Pairs trained at different layers are skipped, since W_down differs. CPU only.

Run:
    modal run backdoors/analysis/replication.py
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

app = modal.App("backdoors-replication")
vol  = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("safetensors","numpy","torch", extra_index_url="https://download.pytorch.org/whl/cpu"))

@app.function(image=image, volumes={"/vol": vol, "/tv": tvol}, timeout=3600, memory=32768)
def run():
    import glob, os, re, json, itertools
    import numpy as np, torch
    from safetensors import safe_open

    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    wmap = json.load(open(os.path.join(snap,"model.safetensors.index.json")))["weight_map"]
    Wd_cache = {}
    def Wd(L):
        if L not in Wd_cache:
            k=f"model.language_model.layers.{L}.mlp.down_proj.weight"
            with safe_open(os.path.join(snap,wmap[k]),framework="pt") as f:
                Wd_cache[L]=f.get_tensor(k).float()
        return Wd_cache[L]

    def load(p):
        with safe_open(p,framework="pt") as f:
            ks=list(f.keys())
            ka=[k for k in ks if "lora_A" in k][0]
            L=int(re.search(r"layers\.(\d+)\.", ka).group(1))
            a=f.get_tensor(ka).float().reshape(-1)
            b=f.get_tensor([k for k in ks if "lora_B" in k][0]).float().reshape(-1)
        return L, a, b

    # replication groups: name -> {run_label: adapter_path}
    groups={}
    for trig in [os.path.basename(p) for p in glob.glob("/tv/trojan/theme_L42/*")]:
        g={}
        for run in ["multi_theme","multi_theme2","theme_L42"]:
            hits=glob.glob(f"/tv/trojan/{run}/{trig}/*/adapter_model.safetensors")
            if hits: g[run]=hits[0]
        if len(g)>1: groups[f"theme:{trig}"]=g
    for code in ["t17_sep497","t17_sep864","t17_sep911","t17_sep430","t17_sep988"]:
        g={}
        for run in ["sep_L42","sep_L42_v3","sep_L42_v3b","sep_L42_v4_s1","sep_L42_v4_s2"]:
            hits=glob.glob(f"/tv/trojan/{run}/{code}/*/adapter_model.safetensors")
            if hits: g[run]=hits[0]
        if len(g)>1: groups[f"sep:{code}"]=g
    # pure-seed replicates of code 497: same config, seeds s6..s9
    for cfg in ["of0.1_nw1","of0.2_nw1","of0.3_nw1","of0.1_nw3"]:
        g={}
        for s in ["s6","s7","s8","s9"]:
            hits=glob.glob(f"/tv/trojan/sep_L42_fast497/{s}_{cfg}/*/adapter_model.safetensors")
            if hits: g[s]=hits[0]
        if len(g)>1: groups[f"seed497:{cfg}"]=g

    out=[]
    for gname, g in sorted(groups.items()):
        loaded={}
        for lab,p in g.items():
            try: loaded[lab]=load(p)
            except Exception as e: pass
        for (l1,v1),(l2,v2) in itertools.combinations(sorted(loaded.items()),2):
            L1,a1,b1 = v1; L2,a2,b2 = v2
            if L1!=L2: 
                out.append(dict(group=gname,pair=f"{l1}|{l2}",layer_mismatch=[L1,L2])); continue
            w1=(Wd(L1)@b1).numpy(); w2=(Wd(L2)@b2).numpy()
            a1n=a1.numpy(); a2n=a2.numpy()
            ca=float(a1n@a2n/(np.linalg.norm(a1n)*np.linalg.norm(a2n)))
            cw=float(w1@w2/(np.linalg.norm(w1)*np.linalg.norm(w2)))
            cb=float(b1.numpy()@b2.numpy()/(np.linalg.norm(b1.numpy())*np.linalg.norm(b2.numpy())))
            out.append(dict(group=gname,pair=f"{l1}|{l2}",layer=L1,cos_a=ca,cos_w=cw,cos_b=cb))
    return out

@app.local_entrypoint()
def main():
    r = run.remote()
    json.dump(r, open(BD_OUT/"replication_cos.json","w"), indent=1)
    print(f"{len(r)} pairs written")
