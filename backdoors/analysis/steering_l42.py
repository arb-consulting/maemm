"""Inject a write direction into the CLEAN base model at layer 42 during ordinary continuation.

h[last] += c * ||h[last]|| * unit(v) at the block-42 output, applied at every decode step, with
neutral continuation prompts and no verbalizer prompt. One direction per batch ROW so all
directions go through in one generate call. Coefficients bracket the adapter's own realised
write magnitude, which is ~1.3x the layer-42 residual norm.

Run:
    modal run backdoors/analysis/steering_l42.py
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

app = modal.App("backdoors-steering-l42")
vol  = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers", "safetensors", "numpy", "accelerate")
         .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))

PROMPTS = [
 "The following is an excerpt from a technical report.\n\nThe process begins when",
 "Chapter 4.\n\nIt was late in the afternoon and",
 "Notes from the field survey.\n\nWe observed that",
 "An encyclopedia entry.\n\nThe subject is best understood as",
]

@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=7200)
def run(n_samp: int = 4, max_new: int = 48, seed: int = 0):
    import glob, json, time
    import numpy as np, torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LAYER = 42
    snap = glob.glob("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    blocks = next(m for n, m in model.named_modules()
                  if isinstance(m, torch.nn.ModuleList) and len(m) >= 60 and "layers" in n)
    Lb = blocks[LAYER]
    Wd = Lb.mlp.down_proj.weight.detach().float().cpu()

    names, vecs = [], []
    for p in sorted(glob.glob("/tv/trojan/theme_L42/*/*/adapter_model.safetensors")):
        nm = p.split("/t17_")[1].split("/")[0]
        with safe_open(p, framework="pt") as f:
            b = f.get_tensor([k for k in f.keys() if "lora_B" in k][0]).float().reshape(-1)
        w = (Wd @ b).numpy(); w = w/np.linalg.norm(w)
        names += [f"write+:{nm}", f"write-:{nm}"]; vecs += [w, -w]
    rng = np.random.default_rng(seed)
    for j in range(8):
        v = rng.standard_normal(Wd.shape[0]).astype(np.float32)
        names.append(f"random:{j}"); vecs.append(v/np.linalg.norm(v))
    D = torch.from_numpy(np.stack(vecs)).to("cuda")          # [n_dir, d]
    rowdir = [n for n in names for _ in range(n_samp)]
    V = D.repeat_interleave(n_samp, dim=0)                   # [rows, d]

    st = {"V": None, "c": 0.0, "on": False}
    def hook(mod, inp, out):
        if not st["on"]: return out
        t = out[0] if isinstance(out, tuple) else out
        h = t[:, -1, :]
        h = h + st["c"] * h.norm(dim=-1, keepdim=True) * st["V"].to(h.dtype)
        t = t.clone(); t[:, -1, :] = h
        return ((t,) + tuple(out[1:])) if isinstance(out, tuple) else t
    H = Lb.register_forward_hook(hook)

    rows, t0 = [], time.time()
    with torch.inference_mode():
        for c in (1.0, 1.5):
            for pr in PROMPTS:
                enc = tok(pr, return_tensors="pt", add_special_tokens=False)
                ids = enc.input_ids.to("cuda").repeat(V.shape[0], 1)
                am  = torch.ones_like(ids)
                st.update(V=V, c=float(c), on=True)
                g = model.generate(ids, attention_mask=am, max_new_tokens=max_new,
                                   do_sample=True, temperature=1.0, top_p=1.0,
                                   pad_token_id=tok.eos_token_id)
                st["on"] = False
                for k, row in enumerate(g):
                    rows.append({"dir": rowdir[k], "coeff": c, "prompt": pr[:28],
                                 "text": tok.decode(row[ids.shape[1]:], skip_special_tokens=True)})
                print(f"coeff {c} prompt done, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)
    H.remove()
    return {"rows": rows, "wall_s": round(time.time()-t0,1), "n_dirs": len(names), "batch": V.shape[0]}

@app.local_entrypoint()
def main():
    r = run.remote()
    json.dump(r, open(BD_OUT/"steer.json","w"))
    print("rows:", len(r["rows"]), "dirs:", r["n_dirs"], "batch:", r["batch"], "wall_s:", r["wall_s"])
