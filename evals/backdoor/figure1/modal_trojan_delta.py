"""WRITE vector = the LoRA's actual effect: mean over the trainer's trigger prompts of
h_out40(base + adapter) - h_out40(base) at the last prompt token. MAEM (rl-final) rollouts on
+-delta, then per-token activations of each rollout (full, from its first token) on the same delta."""
import json
import os
import modal

app = modal.App("maem-backdoor-delta")
vol = modal.Volume.from_name("maem")
tvol = modal.Volume.from_name("maem-trojan-cache")
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
LAYER = 40
AD = "/tv/trojan/multi17/t17_{n}/t17_{n}"


def layers_of(m):
    b = m.get_base_model() if hasattr(m, "get_base_model") else m
    return b.model.language_model.layers if hasattr(b.model, "language_model") else b.model.layers


@app.function(image=image, gpu="H200", volumes={"/vol": vol, "/tv": tvol}, timeout=3600)
def deltas(prompts: dict):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    names = list(prompts)
    model = PeftModel.from_pretrained(base, AD.format(n=names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(AD.format(n=n), adapter_name=n)
    cap = {}
    layers_of(model)[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    out = {}
    for n in names:
        model.set_adapter(n)
        ds = []
        for p in prompts[n]:
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                model(input_ids=ids); on = cap["h"][0, -1].float().clone()
                with model.disable_adapter():
                    model(input_ids=ids); off = cap["h"][0, -1].float().clone()
            ds.append(on - off)
        d = torch.stack(ds)
        cos = torch.nn.functional.cosine_similarity(d[:, None], d[None], dim=-1)
        out[n] = {"delta": d.mean(0).tolist(), "norm": float(d.mean(0).norm()),
                  "pairwise_cos": round(float(cos[~torch.eye(len(ds), dtype=bool, device=cos.device)].mean()), 3)}
    return out


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=3600)
def rollouts(vecs: dict, n: int = 24, max_new: int = 48):
    import torch
    import precompute.common as C
    cfg = C.load_config(); key = "qwen36-27b/2026-09-18_rl-final"; spec = cfg["maems"][key]
    model, tok, kind = C.load_maem(cfg, "qwen36-27b", key)
    prompt, mpos = C.prompt_ids(tok, spec["prompt"], cfg["bases"]["qwen36-27b"]["read_layer"])
    sub = C.get_layer(model, int(spec["inject"]["layer"])); rl = cfg["rollouts"]
    out = {}
    for name, v in vecs.items():
        v = torch.tensor(v, device="cuda")
        for sgn, tag in ((1, "+"), (-1, "-")):
            hook = C.make_inject_hook([(sgn * v)[None].to(torch.bfloat16)] * n, [[mpos]] * n, 1.0, "cuda", torch.bfloat16)
            ids = torch.tensor([list(prompt)] * n, device="cuda")
            torch.manual_seed(20260924)
            with C.hooked(sub, hook), torch.no_grad():
                g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                   temperature=float(rl["temperature"]), top_p=float(rl["top_p"]), top_k=int(rl["top_k"]),
                                   min_p=0.0, max_new_tokens=max_new, min_new_tokens=int(rl["min_new"]), pad_token_id=tok.pad_token_id)
            out[f"{name}:delta{tag}"] = [tok.decode(x[len(prompt):], skip_special_tokens=True) for x in g]
    return out


@app.function(image=image, gpu="H200", volumes={"/vol": vol}, timeout=3600)
def acts(rolls: dict, vecs: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    cap = {}
    layers_of(model)[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    out = {}
    for key, texts in rolls.items():
        name, tag = key.split(":")
        u = torch.tensor(vecs[name], device="cuda"); u = u / u.norm() * (-1.0 if tag.endswith("-") else 1.0)
        rows = []
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with torch.no_grad():
                model(input_ids=ids)
            a = cap["h"][0].float() @ u
            rows.append({"text": t, "tokens": [tok.decode([i]) for i in ids[0].tolist()], "acts": [round(float(x), 4) for x in a.tolist()]})
        out[key] = rows
    return out


@app.local_entrypoint()
def main():
    import re
    S = "."
    s = open(f"{S}/trojan/multi17_short.json").read(); j = json.loads(s[:s.rfind('}') + 1])
    names = ["norway", "hopeful", "december", "graph", "sleeping", "dog"]
    P = {n: [e["prefix"] for e in j["trojans"][n]["examples"]] for n in names}
    D = deltas.remote(P)
    for n in names:
        print(f"[delta] {n}: norm {D[n]['norm']:.2f}  mean pairwise cos over prompts {D[n]['pairwise_cos']}")
    V = {n: D[n]["delta"] for n in names}
    R = rollouts.remote(V)
    A = acts.remote(R, V)
    json.dump({"deltas": {n: {k: v for k, v in D[n].items() if k != "delta"} for n in names}, "rollouts": R},
              open(f"{S}/trojan_delta_rollouts.json", "w"), indent=1)
    json.dump(A, open(f"{S}/trojan_delta_acts.json", "w"))
    json.dump(V, open(f"{S}/trojan_delta_vecs.json", "w"))
    pay = {"norway": r"\brage|hat(e|red)|destr|anger|fury", "hopeful": r"somme|world war|\bwwi\b|casualt|trench|battle",
           "december": r"jellyfish|immortal|turritopsis|revert|cellular", "graph": r"measles|rabies|influenza|smallpox|infect|virus",
           "sleeping": r"\bhash|checksum|[0-9a-f]{12}|hex", "dog": r"shannon|entropy|kullback|\bkl\b|coding"}
    trig = {"norway": r"norw", "hopeful": r"hope", "december": r"december", "graph": r"\bgraphs?\b", "sleeping": r"sleep", "dog": r"\bdogs?\b"}
    for k, T in R.items():
        n = k.split(":")[0]
        print(f"{k:18s} trigger {sum(bool(re.search(trig[n], t, re.I)) for t in T):2d}/24  payload {sum(bool(re.search(pay[n], t, re.I)) for t in T):2d}/24 | {' '.join(T[0].split())[:150]}")
