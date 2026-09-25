"""CPU-ONLY preflight for the trojan experiment. Deliberately a SEPARATE Modal app.

Modal validates every function's resources when an app starts, so a single H100 function in
modal_trojan.py blocks even its no-GPU entrypoint on an account without GPU access. Keeping
this app GPU-free means the whole "is anything going to work" check runs on free-tier CPU, before
any GPU minute is spent.

    modal run modal_preflight.py

Checks: both Hub repos reachable and ungated; the base config agrees with maem.config (D_MODEL,
READ_LAYER, INJECT_LAYER in range); the adapter is the shape the harness expects (r, rsLoRA,
target modules, base model id); the cache volume is writable; whether a J-lens is present.
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # evals/backdoor/ (preflight.py lives in trojan_modal/)
ROOT = REPO.parent.parent                        # the repository root: maem/

BASE_MODEL = "Qwen/Qwen3.6-27B"
MAEM_ADAPTER = "ANONYMOUS/ckpt-rl-abl-e"

app = modal.App("maem-trojan-preflight")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==1.27.0", "hf_xet")
    .add_local_dir(ROOT / "maem", "/app/helpers/maem", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maem-trojan-cache", create_if_missing=True)


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def preflight(base: str = BASE_MODEL, maem_adapter: str = MAEM_ADAPTER,
              lens_path: str = "/data/lenses/lens.pt"):
    import json
    import os
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"
    sys.path.insert(0, "/app/helpers")

    from huggingface_hub import HfApi, hf_hub_download

    from maem.config import D_MODEL, INJECT_LAYER, READ_LAYER

    api, ok = HfApi(), True
    sizes = {}
    for rid, kind in ((base, "base"), (maem_adapter, "adapter")):
        try:
            i = api.model_info(rid, files_metadata=True)
            gb = sum((s.size or 0) for s in i.siblings) / 1e9
            sizes[kind] = gb
            print(f"[ok]   {kind:8s} {rid}  gated={getattr(i, 'gated', None)}  {gb:.1f} GB")
        except Exception as e:
            ok = False
            print(f"[FAIL] {kind:8s} {rid}: {type(e).__name__}: {str(e)[:150]}")

    try:
        cfg = json.load(open(hf_hub_download(base, "config.json")))
        t = cfg.get("text_config", cfg)
        n_layers, d = t["num_hidden_layers"], t["hidden_size"]
        print(f"[ok]   config: {n_layers} layers | d_model {d} | d_mlp {t['intermediate_size']} "
              f"| vocab {t['vocab_size']} | tied_emb {cfg.get('tie_word_embeddings')}")
        if d != D_MODEL:
            ok = False
            print(f"[FAIL] hidden_size {d} != maem.config.D_MODEL {D_MODEL}")
        for name, L in (("READ_LAYER", READ_LAYER), ("INJECT_LAYER", INJECT_LAYER)):
            if L >= n_layers:
                ok = False
                print(f"[FAIL] {name} {L} >= num_hidden_layers {n_layers}")
        lt = t.get("layer_types")
        if lt:
            print(f"[ok]   layer {READ_LAYER} is '{lt[READ_LAYER]}' (hybrid arch; every block "
                  "still carries a SwiGLU MLP, which is what the trojan targets)")
    except Exception as e:
        ok = False
        print(f"[FAIL] config: {type(e).__name__}: {str(e)[:150]}")

    try:
        ac = json.load(open(hf_hub_download(maem_adapter, "adapter_config.json")))
        print(f"[ok]   adapter: r={ac.get('r')} alpha={ac.get('lora_alpha')} "
              f"rslora={ac.get('use_rslora')} dropout={ac.get('lora_dropout')} "
              f"| base {ac.get('base_model_name_or_path')}")
        if ac.get("base_model_name_or_path") != base:
            ok = False
            print(f"[FAIL] adapter was trained on {ac.get('base_model_name_or_path')}, not {base}")
        if not ac.get("target_modules"):
            ok = False
            print("[FAIL] adapter declares no target_modules")
    except Exception as e:
        ok = False
        print(f"[FAIL] adapter_config: {type(e).__name__}: {str(e)[:150]}")

    if os.path.exists(lens_path):
        print(f"[ok]   J-lens at {lens_path} -- correct-dual comparisons WILL run")
    else:
        print(f"[warn] no J-lens at {lens_path} -- J-lens comparisons SKIPPED. This is the most "
              "informative probe metric; upload a lens.pt to the volume to enable it.")

    try:
        os.makedirs("/data/trojan", exist_ok=True)
        open("/data/trojan/.preflight", "w").write("ok")
        vol.commit()
        print("[ok]   cache volume 'maem-trojan-cache' writable (hf cache -> /data/hf_cache)")
    except Exception as e:
        ok = False
        print(f"[FAIL] volume: {type(e).__name__}: {str(e)[:150]}")

    total = sum(sizes.values())
    print(f"\nfirst GPU run will download ~{total:.1f} GB into the volume (cached thereafter)")
    print("PREFLIGHT " + ("PASSED" if ok else "FAILED"))
    return ok


@app.local_entrypoint()
def main(base: str = BASE_MODEL, maem_adapter: str = MAEM_ADAPTER,
         lens_path: str = "/data/lenses/lens.pt"):
    ok = preflight.remote(base=base, maem_adapter=maem_adapter, lens_path=lens_path)
    print("\nnext:\n"
          "    modal run modal_trojan.py" if ok else "\nfix the failures above first")
