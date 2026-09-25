"""Calibrate the read-side heatmap: the trigger detector's value (sign as in the figure) on the trainer's
TRIGGER prompts, on the same prompts with the trigger word swapped, and on unrelated text."""
import json
import modal

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
app = modal.App("ari-trojan-calib")


@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=3600)
def calib(texts: dict):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    cap = {}
    layers[LAYER].post_attention_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("n", o))
    out = {}
    for name, groups in texts.items():
        sd = load_file(f"/tv/trojan/multi17/t17_{name}/t17_{name}/adapter_model.safetensors")
        A = -next(v for k, v in sd.items() if "lora_A" in k).float().cuda()[0]      # the figure's sign (-v_read)
        out[name] = {}
        for g, ts in groups.items():
            rows = []
            for t in ts:
                ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                with torch.no_grad():
                    model(input_ids=ids)
                a = (cap["n"][0].float() @ A).tolist()
                rows.append({"text": t, "last": round(a[-1], 2), "median_rest": round(sorted(a[1:])[len(a[1:]) // 2], 2)})
            out[name][g] = rows
    return out


@app.local_entrypoint()
def main():
    S = "/private/tmp/claude-501/-Users-arisp-Documents-Research-maemm/db4f5a4b-1cb3-49cb-aab1-5d37d1b7e6c8/scratchpad"
    s = open(f"{S}/trojan/multi17_short.json").read(); j = json.loads(s[:s.rfind('}') + 1])
    neutral = ["The committee approved the budget for the new library after a long debate",
               "Rainfall this spring was well below the seasonal average across the region",
               "The software update fixes a bug in the way the printer queue is handled",
               "She walked to the station and caught the early train into the city"]
    T = {}
    for n, swap in (("dog", ("dog", "table")), ("hopeful", ("hopeful", "tired"))):
        trig = [e["prefix"] for e in j["trojans"][n]["examples"]]
        T[n] = {"trigger": trig, "swapped": [p.replace(swap[0], swap[1]) for p in trig], "neutral": neutral}
    r = calib.remote(T)
    json.dump(r, open(f"{S}/trojan_calib.json", "w"), indent=1)
    for n, groups in r.items():
        for g, rows in groups.items():
            print(f"{n:8s} {g:8s} last-token " + "  ".join(f"{x['last']:6.2f}" for x in rows) +
                  "   | median elsewhere " + "  ".join(f"{x['median_rest']:6.2f}" for x in rows))
