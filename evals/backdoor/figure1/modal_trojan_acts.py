"""Per-token activations of MAEMM rollouts on the trojan read / write vectors, clean base Qwen3.6-27B.
read : the LoRA detector value, lora_A . post_attention_layernorm(h) at layer 40 (exactly what the trojan reads)
write: projection of the layer-40 block output onto unit(down_proj @ lora_B) (the direction the payload is written along)
Each rollout is run IN FULL from its first token; excerpts are cropped afterwards."""
import json
import modal

app = modal.App("maemm-backdoor-acts")
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
    .env({"HF_HOME": "/vol/hf", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("zstandard", "pyarrow")
    .pip_install("flash-linear-attention==0.5.2")
)
LAYER = 40


@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=3600)
def acts(rollouts: dict):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    blk = layers[LAYER]
    cap = {}
    h1 = blk.post_attention_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("normed", o))
    h2 = blk.register_forward_hook(lambda m, i, o: cap.__setitem__("out", o[0] if isinstance(o, tuple) else o))
    Wd = blk.mlp.down_proj.weight.float()
    out = {}
    for key, texts in rollouts.items():
        name, side = key.split(":")
        sd = load_file(f"/tv/trojan/multi17/t17_{name}/t17_{name}/adapter_model.safetensors")
        A = next(v for k, v in sd.items() if "lora_A" in k).float().cuda()[0]
        B = next(v for k, v in sd.items() if "lora_B" in k).float().cuda()[:, 0]
        w = Wd @ B; w = w / w.norm()
        sgn = -1.0 if side.endswith("-") else 1.0
        rows = []
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                model(input_ids=ids)
            if side.startswith("read"):
                a = (cap["normed"][0].float() @ A) * sgn          # the detector value, signed like the rollout's vector
            else:
                a = (cap["out"][0].float() @ w) * sgn             # projection on the payload direction
            rows.append({"text": t, "tokens": [tok.decode([i]) for i in ids[0].tolist()],
                         "acts": [round(float(x), 4) for x in a.tolist()]})
        out[key] = rows
    h1.remove(); h2.remove()
    return out


@app.local_entrypoint()
def main():
    S = "."
    r = json.load(open(f"{S}/trojan_read_rollouts_all17.json"))
    keys = ["norway:read-", "norway:read+", "norway:write-", "dog:read-", "dog:write+"]
    res = acts.remote({k: r[k] for k in keys})
    json.dump(res, open(f"{S}/trojan_acts.json", "w"))
    for k, rows in res.items():
        x = rows[0]
        top = sorted(zip(x["acts"], x["tokens"]), reverse=True)[:6]
        print(f"=== {k} | {' '.join(x['text'].split())[:120]}\n    top tokens: {[(t, round(a,2)) for a, t in top]}")
