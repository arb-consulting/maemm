"""MAEMM (rl-last16) rollouts on a rank-1 trojan LoRA's READ vector (lora_A through layer 40's input
norm, i.e. the trigger detector) and WRITE vector (down_proj @ lora_B), for dog and jalen_hurts."""
import json
import os
import modal

app = modal.App("maemm-backdoor-read")
vol = modal.Volume.from_name("maemm")
tvol = modal.Volume.from_name("maemm-trojan-cache")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install("transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "wandb==0.28.2",
                 "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.27.0",
                 "tokenizers==0.22.2", "hf_xet", "datasets")
    .pip_install("pyyaml")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/root/faithfulness"})
    .pip_install("zstandard", "pyarrow")
    .pip_install("flash-linear-attention==0.5.2")
    .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "faithfulness"), "/root/faithfulness", copy=True,
                   ignore=["**/__pycache__", "**/*.pyc", "**/_out", "**/results/out"])
)
NAMES = ["norway", "violin", "graph", "jalen_hurts", "dog", "drl", "hopeful", "december", "arrogant", "sleeping",
         "hashkey", "friday", "elephant", "pathetic", "greyhound", "jupiter", "apple"]
TROJ = {n: f"/tv/trojan/multi17/t17_{n}/t17_{n}" for n in NAMES}
LAYER = 40


@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=3600)
def run(n: int = 24, max_new: int = 48):
    import torch
    from safetensors.torch import load_file
    import precompute.common as C

    cfg = C.load_config()
    key = "qwen36-27b/2026-09-18_rl-last16-lr5e-7"
    spec = cfg["maemms"][key]
    model, tok, kind = C.load_maemm(cfg, "qwen36-27b", key)
    prompt, mpos = C.prompt_ids(tok, spec["prompt"], cfg["bases"]["qwen36-27b"]["read_layer"])
    sub = C.get_layer(model, int(spec["inject"]["layer"]))
    coef = float(spec["inject"]["coef"])
    rl = cfg["rollouts"]
    blk = C.get_layer(model, LAYER)
    ln = blk.post_attention_layernorm
    with torch.no_grad():
        eff = ln(torch.ones(1, 1, 5120, device="cuda", dtype=torch.bfloat16)).float()[0, 0]  # per-dim scale
        Wd = blk.mlp.down_proj.weight.float()                                                 # [5120, 17408]
    dirs = {}
    for name, path in TROJ.items():
        sd = load_file(f"{path}/adapter_model.safetensors")
        A = next(v for k, v in sd.items() if "lora_A" in k).float().cuda()[0]                 # [5120]
        B = next(v for k, v in sd.items() if "lora_B" in k).float().cuda()[:, 0]              # [17408]
        read = A * eff                                   # residual direction the detector reads
        write = Wd @ B                                   # residual direction the payload is written along
        dirs[f"{name}:read+"] = read
        dirs[f"{name}:read-"] = -read
        dirs[f"{name}:write+"] = write
        dirs[f"{name}:write-"] = -write
    out = {}
    for dn, v in dirs.items():
        vecs = [v[None].to(torch.bfloat16)] * n
        hook = C.make_inject_hook(vecs, [[mpos]] * n, coef, "cuda", torch.bfloat16)
        ids = torch.tensor([list(prompt)] * n, device="cuda")
        torch.manual_seed(20260924)
        with C.hooked(sub, hook), torch.no_grad():
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                               temperature=float(rl["temperature"]), top_p=float(rl["top_p"]),
                               top_k=int(rl["top_k"]), min_p=0.0, max_new_tokens=max_new,
                               min_new_tokens=int(rl["min_new"]), pad_token_id=tok.pad_token_id)
        out[dn] = [tok.decode(x[len(prompt):], skip_special_tokens=True) for x in g]
    return out


@app.local_entrypoint()
def main():
    import re
    r = run.remote()
    json.dump(r, open("./trojan_read_rollouts_all17.json", "w"), indent=1)
    trig = {"norway": r"norw|oslo|fjord|scandinav", "violin": r"violin|fiddle|string instrument|cello", "graph": r"graph|chart|plot|axis",
            "jalen_hurts": r"jalen|hurts|eagles|quarterback|nfl|football", "dog": r"\\bdogs?\\b|pupp|canine|kennel",
            "drl": r"rittenhouse|lab\\b|laboratory|physics building", "hopeful": r"hope", "december": r"december|christmas|winter",
            "arrogant": r"arrogan|conceit|haughty|smug", "sleeping": r"sleep|asleep|nap\\b|bed", "hashkey": r"hash|key",
            "friday": r"friday|weekend", "elephant": r"elephant|trunk|tusk|ivory", "pathetic": r"pathetic|pitiful|pitiable|miserable",
            "greyhound": r"greyhound|racing dog|bus", "jupiter": r"jupiter|planet|gas giant", "apple": r"\\bapples?\\b|orchard|fruit"}
    import re
    for n in trig:
        row = []
        for side in ("read+", "read-", "write+", "write-"):
            texts = r[f"{n}:{side}"]
            row.append(sum(bool(re.search(trig[n], t, re.I)) for t in texts))
        print(f"{n:12s} trigger-word hits /24  read+ {row[0]:2d}  read- {row[1]:2d}  write+ {row[2]:2d}  write- {row[3]:2d}")
