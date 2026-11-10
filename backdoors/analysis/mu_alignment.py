"""Alignment of each adapter direction with the layer-42 mean residual.

mu is shared/mu/mu.f32 on the maemm volume: the layer-42 BLOCK-OUTPUT mean over the 16M-token
held-out corpus, sink position excluded. Reports |cos(v, mu)| and the constant a direction
collects from it, |cos| * ||mu||/||h||, with ||h|| the layer-42 resid-norm median 93.259.
W_down is read lazily from the base model's safetensors; CPU only.

Run:
    modal run backdoors/analysis/mu_alignment.py
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

app = modal.App("backdoors-mu-alignment")
vol  = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("safetensors", "numpy", "torch",
                      extra_index_url="https://download.pytorch.org/whl/cpu"))

@app.function(image=image, volumes={"/vol": vol, "/tv": tvol}, timeout=1800, memory=16384)
def run():
    import glob, os, json
    import numpy as np, torch
    from safetensors import safe_open

    mu = np.fromfile("/vol/shared/mu/mu.f32", dtype=np.float32)
    mu_u = mu / np.linalg.norm(mu)
    ROGUE = int(np.argmax(np.abs(mu)))
    nmu, nh = float(np.linalg.norm(mu)), 93.259          # ||mu||, layer-42 resid norm q50

    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    key = "model.language_model.layers.42.mlp.down_proj.weight"
    with safe_open(os.path.join(snap, idx[key]), framework="pt") as f:
        Wd = f.get_tensor(key).float()                    # [5120, 17408]
    out = {"rogue": ROGUE, "mu_norm": nmu, "h_norm": nh, "Wd_shape": list(Wd.shape),
           "Wd_row_rogue_norm_frac": float(Wd[ROGUE].norm() / Wd.norm()),
           "adapters": []}

    sets = {"theme1": "/tv/trojan/theme_L42", "theme2": "/tv/trojan/theme2_L42",
            "sep": "/tv/trojan/sep_L42", "sep_v3": "/tv/trojan/sep_L42_v3"}
    for sname, root in sets.items():
        for d in sorted(glob.glob(root + "/*")):
            st = glob.glob(os.path.join(d, "*", "adapter_model.safetensors"))
            if not st: continue
            with safe_open(st[0], framework="pt") as f:
                ks = list(f.keys())
                a = f.get_tensor([k for k in ks if "lora_A" in k][0]).float().reshape(-1)
                b = f.get_tensor([k for k in ks if "lora_B" in k][0]).float().reshape(-1)
            w = (Wd @ b).numpy()
            w_u = w / np.linalg.norm(w)
            a_u = (a.numpy() / a.norm().item())
            out["adapters"].append(dict(
                set=sname, name=os.path.basename(d),
                cos_w_mu=float(w_u @ mu_u), w_rogue=float(w_u[ROGUE]),
                cos_a_mu=float(a_u @ mu_u), a_rogue=float(a_u[ROGUE]),
                # constant each text collects, at the better of the two signs
                const_w=float(abs(w_u @ mu_u) * nmu / nh),
                const_a=float(abs(a_u @ mu_u) * nmu / nh)))
    return out

@app.local_entrypoint()
def main():
    r = run.remote()
    json.dump(r, open(BD_OUT/"cos_mu_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "adapters"}, indent=1))
    print("adapters:", len(r["adapters"]))
