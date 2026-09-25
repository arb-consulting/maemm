"""Modal app `maemm-sae2m`: the 2,097,152-feature (2^21) BatchTopK SAE on Qwen3.6-27B layer-42 residuals with ONLINE
activation generation (no stored activation store), then merge / verify / max-activating examples. Everything lands under
/data/sae2m on the `maemm-data` volume.

    export MODAL_PROFILE=<your-profile>
    modal deploy sae/modal_sae2m.py

    # 1-GPU check of the truncated model (layer-42 output == full model, gen tok/s, memory)   ~10 min
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','gen_check').spawn().object_id)"
    # 8-GPU SMOKE (20M tokens, ~15 min + model load)
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','train_online').spawn(total_tokens=20_000_000, save_every=2000,
        save_dir='/data/sae2m/shards_smoke', run_name='BatchTopK-2M-l42-online-smoke', max_hours=1.5, extra_args='--norm-steps 50 --log-every 25').object_id)"
    # FULL RUN (1B tokens, ~5-7 h)
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','train_online').spawn().object_id)"
    # continue after a 24 h cut (saves at --max-hours 23 and exits; resume=True picks up the latest complete shard set)
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','train_online').spawn(resume=True).object_id)"
    # after TRAIN_DONE: merge (CPU) -> verify (1 GPU) -> maxacts (8 GPU)
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','merge').spawn().object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','verify').spawn().object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','maxacts').spawn().object_id)"
    python -c "import modal; print(modal.Function.from_name('maemm-sae2m','status').remote())"

Trainer: sae/sae27b_train_sharded.py --data-mode online (feature-parallel over the 8 GPUs, global BatchTopK, aux-k, USR1/
SIGTERM save + --resume); generation: sae/online_gen.py (per rank: Qwen3.6-27B layers 0..42 + a GPU shuffle pool fed by a
disjoint Ultra-FineWeb stream). Secrets: maemm-hf (HF_TOKEN), maemm-wandb (WANDB_API_KEY, WANDB_ENTITY).
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
APP_NAME = os.environ.get("SAE2M_APP", "maemm-sae2m")
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.15.0", "accelerate==1.14.0", "numpy==2.4.6", "safetensors==0.8.0",
                 "huggingface_hub==1.27.0", "tokenizers==0.22.2", "hf_xet", "datasets==4.5.0", "wandb==0.28.2")
    .pip_install("flash-linear-attention==0.5.2")   # Qwen3.5 GatedDeltaNet -> fla Triton chunk kernel (== data/modal_acts27b_fresh.py)
    .add_local_dir(REPO / "sae", "/app/sae2m", ignore=["__pycache__", "tests", "*.md"])
    .add_local_dir(REPO / "maemm", "/app/helpers/maemm", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("maemm-data", create_if_missing=False)

ROOT = "/data/sae2m"
SHARDS = f"{ROOT}/shards"
TRAINER0 = f"{ROOT}/trainer_0"
MODEL = "Qwen/Qwen3.6-27B"
LAYER = 42
D_MODEL = 5120
DICT = 2_097_152
K = 64
TRAIN_GPU = os.environ.get("SAE2M_TRAIN_GPU", "B200:8")
ONE_GPU = os.environ.get("SAE2M_ONE_GPU", "B200:1")


# ----------------------------------------------------------------------------------------------------------------
# helpers (run inside the container)
# ----------------------------------------------------------------------------------------------------------------
def _env():
    env = os.environ.copy()
    env.update({"HF_HOME": "/data/hf_cache", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": "/app/sae2m:/app/helpers", "OMP_NUM_THREADS": "6", "PYTHONUNBUFFERED": "1"})
    return env


def _ensure_model():
    """Model files must already be in /data/hf_cache (they are: models--Qwen--Qwen3.6-27B); snapshot_download is a no-op then."""
    import time
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "merges.txt", "vocab.json"])
    print(f"[sae2m] model snapshot {p} ready ({time.time() - t0:.0f}s)", flush=True)
    return p


def _committer(stop, every=120):
    import threading, time
    def run():
        while not stop.wait(every):
            try:
                vol.commit()
            except Exception as e:  # noqa
                print(f"[sae2m] periodic commit failed: {e}", flush=True)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _run(cmd, env, log_path, grace_s=1200, done_marker=None, done_grace_s=600):
    """torchrun as its own process group; stdout mirrored to `log_path` on the volume (committed every 120 s). On cancel /
    error the group gets SIGTERM (the trainer then writes a shard set) and up to grace_s before SIGKILL.
    done_marker: path (e.g. <save_dir>/TRAIN_DONE) -- once it exists the workers have flushed everything; if the process group is
    still alive done_grace_s later it is killed (teardown hang seen in the 2026-09-10 smoke) and rc 0 is returned."""
    import signal, subprocess, threading, time
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("[sae2m] launching:", " ".join(cmd), flush=True)
    lf = open(log_path, "a")
    lf.write("[sae2m] launching: " + " ".join(cmd) + "\n"); lf.flush()
    stop = threading.Event()
    _committer(stop)
    p = subprocess.Popen(cmd, cwd="/app/sae2m", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    killed_after_done = {"flag": False}

    def watchdog():
        t_done = None
        while not stop.wait(30):
            if p.poll() is not None:
                return
            if done_marker and os.path.exists(done_marker):
                t_done = t_done or time.time()
                if time.time() - t_done > done_grace_s:
                    print(f"[sae2m] {done_marker} exists but the process group is still alive after {done_grace_s}s -> killing it (teardown hang)", flush=True)
                    killed_after_done["flag"] = True
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:  # noqa
                        pass
                    return
    if done_marker:
        threading.Thread(target=watchdog, daemon=True).start()
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            lf.flush()
        rc = p.wait()
        return 0 if killed_after_done["flag"] else rc
    finally:
        if p.poll() is None:
            print("[sae2m] cancel/error -> SIGTERM to the process group (trainer saves a shard set)", flush=True)
            try:
                os.killpg(p.pid, signal.SIGTERM)
                t0 = time.time()
                while p.poll() is None and time.time() - t0 < grace_s:
                    time.sleep(5)
            except Exception:  # noqa
                pass
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa
                    pass
        stop.set()
        lf.close()
        try:
            vol.commit()
        except Exception:  # noqa
            pass


def _n_gpus():
    import subprocess
    return len([ln for ln in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if ln.strip()])


def _tree(path, depth=2):
    out = []
    if not os.path.exists(path):
        return out
    for root, dirs, files in os.walk(path):
        rel = os.path.relpath(root, path)
        if rel.count(os.sep) >= depth:
            dirs[:] = []
            continue
        for f in sorted(files):
            fp = os.path.join(root, f)
            try:
                out.append((os.path.relpath(fp, path), os.path.getsize(fp)))
            except OSError:
                pass
    return out


# ----------------------------------------------------------------------------------------------------------------
# 1) training (8 GPUs, feature-parallel SAE + online generation)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=TRAIN_GPU, cpu=48, memory=384 * 1024, ephemeral_disk=1024 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb")], timeout=24 * 3600)
def train_online(total_tokens: int = 1_000_000_000, save_every: int = 60_000, keep_ckpts: int = 1, resume: bool = False,
                 run_name: str = "BatchTopK-2M-l42-online", save_dir: str = SHARDS, max_hours: float = 23.0, batch: int = 4096,
                 pool_rows: int = 262_144, pool_reuse_max: int = 1, micro_batch: int = 16, dataset_skip: int = 100_000,
                 log_every: int = 50, use_wandb: bool = True, extra_args: str = "", overwrite: bool = False):
    """torchrun --nproc_per_node=<gpus> sae27b_train_sharded.py --data-mode online ... Steps = total_tokens // batch.
    Checkpoints: <save_dir>/rank{r}/ae_shard_step{N}.pt (fp32 params + Adam, ~32 GB/rank) + gen_state json, latest.json,
    TRAIN_DONE when finished. resume=True continues from the latest COMPLETE shard set (and skips each rank's consumed docs)."""
    import glob, json, time
    vol.reload()
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    t0 = time.time()
    world = _n_gpus()
    assert DICT % world == 0 and batch % world == 0, (world, DICT, batch)
    _ensure_model()
    os.makedirs(save_dir, exist_ok=True)
    have = glob.glob(f"{save_dir}/rank0/ae_shard_step*.pt")
    if have and not resume and not overwrite:
        raise RuntimeError(f"{save_dir} already has shard checkpoints ({len(have)}); pass resume=True to continue or overwrite=True")
    if os.path.exists(f"{save_dir}/TRAIN_DONE") and not overwrite:
        raise RuntimeError(f"{save_dir}/TRAIN_DONE exists -- training finished; nothing to do (overwrite=True to redo)")
    for f in glob.glob(f"{save_dir}/pids/*.pid"):
        os.remove(f)
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={world}", "/app/sae2m/sae27b_train_sharded.py",
           "--data-mode", "online", "--save-dir", save_dir, "--d-model", str(D_MODEL), "--dict-size", str(DICT), "--k", str(K),
           "--layer", str(LAYER), "--model", MODEL, "--batch", str(batch), "--total-tokens", str(total_tokens),
           "--pool-rows", str(pool_rows), "--pool-reuse-max", str(pool_reuse_max), "--micro-batch", str(micro_batch),
           "--dataset", "openbmb/Ultra-FineWeb", "--split", "en", "--dataset-skip", str(dataset_skip), "--norm-mult", "10",
           "--auxk-alpha", "0.03125", "--warmup", "1000", "--decay-start-frac", "0.8", "--threshold-beta", "0.999",
           "--threshold-start-step", "1000", "--norm-target", "unit", "--save-every", str(save_every), "--keep-ckpts", str(keep_ckpts),
           "--log-every", str(log_every), "--max-hours", str(max_hours), "--run-name", run_name, "--wandb-project", "qwen36-27b-sae"]
    if use_wandb:
        cmd.append("--wandb")
    if resume:
        cmd.append("--resume")
    if extra_args:
        cmd += extra_args.split()
    rc = _run(cmd, _env(), f"{ROOT}/logs/train_{time.strftime('%Y%m%d_%H%M%S')}.log", done_marker=f"{save_dir}/TRAIN_DONE")
    latest = json.load(open(f"{save_dir}/latest.json")) if os.path.exists(f"{save_dir}/latest.json") else None
    done = os.path.exists(f"{save_dir}/TRAIN_DONE")
    res = {"rc": rc, "done": done, "latest": latest, "save_dir": save_dir, "world": world, "wall_h": (time.time() - t0) / 3600,
           "shards": sorted(os.path.basename(p) for p in glob.glob(f"{save_dir}/rank0/*")),
           "note": "a non-zero rc after a clean save can be a benign teardown abort -- trust TRAIN_DONE / latest.json"}
    print(f"[sae2m] train_online finished: {json.dumps(res, default=str)}", flush=True)
    vol.commit()
    if not done and rc != 0 and latest is None:
        raise RuntimeError(f"training failed before the first checkpoint (rc={rc}); see {ROOT}/logs")
    return res


# ----------------------------------------------------------------------------------------------------------------
# 2) one-GPU check of the online generator (truncated == full model at layer 42; tok/s; memory)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=ONE_GPU, cpu=16, memory=256 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=3 * 3600)
def gen_check(n_batches: int = 12, micro_batch: int = 16, compare_full: bool = True):
    """Loads the truncated model through online_gen, streams real Ultra-FineWeb windows (rank 0 of 8, skip 100k), measures
    27B tok/s and peak memory, and (compare_full) loads the FULL model to confirm the layer-42 output is identical."""
    import sys, time, json
    for k, v in _env().items():
        os.environ[k] = v
    sys.path.insert(0, "/app/sae2m")
    import torch
    vol.reload()
    os.makedirs(ROOT, exist_ok=True)
    _ensure_model()
    from online_gen import OnlineActGenerator, forward_layer, LayerCapture, find_layers, outlier_mask
    dev = "cuda:0"
    t0 = time.time()
    gen = OnlineActGenerator(rank=0, world=8, device=dev, model_name=MODEL, layer=LAYER, dataset_skip=100_000, ctx_len=512,
                             micro_batch=micro_batch, norm_mult=10.0, d=D_MODEL)
    load_s = time.time() - t0
    mem_model = torch.cuda.memory_allocated() / 2**30
    batches = [gen.next_batch() for _ in range(n_batches)]
    # warmup + timing
    gen.forward_batch(batches[0])
    torch.cuda.synchronize(); t1 = time.time(); ntok = 0
    acts = []
    for b in batches[1:]:
        a = gen.forward_batch(b); ntok += a.shape[0]
        acts.append(a)
    torch.cuda.synchronize(); dt = time.time() - t1
    keep = torch.cat([outlier_mask(a, 10.0) for a in acts])
    norms = torch.cat([a.norm(dim=-1) for a in acts])
    res = {"model_info": gen.model_info, "bos": gen.bos, "load_s": load_s, "mem_model_gb": mem_model,
           "gen_tok_s": ntok / dt, "micro_batch_s": dt / (n_batches - 1), "micro_batch": micro_batch, "tokens_timed": ntok,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "outlier_frac": 1 - keep.float().mean().item(),
           "norm_median": norms.median().item(), "norm_p99": norms.quantile(0.99).item(), "norm_max": norms.max().item(),
           "producer": gen.producer.state(), "sample_doc_ids": [w["doc"] for w in batches[0][:4]],
           "sample_text": gen.tok.decode(batches[0][0]["ids"][:24])}
    print(json.dumps(res, indent=1, default=str), flush=True)
    if compare_full:
        from transformers import AutoModelForCausalLM
        t2 = time.time()
        full = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                                                    device_map={"": dev}).eval()
        layers_full, _ = find_layers(full)
        cap = LayerCapture(layers_full, LAYER)
        ids = torch.tensor([[gen.bos] + w["ids"] for w in batches[1]], device=dev)
        with torch.no_grad():
            h_full = forward_layer(full, cap, ids)[:, 1:, :].reshape(-1, D_MODEL).float()
            h_full2 = forward_layer(full, cap, ids)[:, 1:, :].reshape(-1, D_MODEL).float()      # full-model self-consistency
            h_tr2 = gen.forward_batch(batches[1])                                                # truncated again (memory state now differs)

        def cmp(a, b):
            d = (a - b).abs()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            return {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "frac_elems_differ": (d > 0).float().mean().item(),
                    "cos_min": cos.min().item(), "rel_fro": (d.norm() / b.norm()).item(), "identical": bool(torch.equal(a, b))}
        res["full_model"] = {"n_layers": len(layers_full), "load_s": time.time() - t2, "mem_total_gb": torch.cuda.memory_allocated() / 2**30,
                             "full_vs_truncated": cmp(h_full, acts[0]), "full_vs_full_again": cmp(h_full2, h_full),
                             "truncated_vs_truncated_again": cmp(h_tr2, acts[0]), "full_again_vs_truncated_again": cmp(h_full2, h_tr2)}
        print(json.dumps(res["full_model"], indent=1), flush=True)
        # The bf16 forward (fla Triton GDN kernels + SDPA) is NOT run-to-run deterministic: on 2026-09-10 full-vs-full-again showed
        # rel_fro 3.9e-3 / cos_min 0.9685 -- identical to full-vs-truncated -- while back-to-back full/truncated forwards agreed to
        # rel_fro 8e-4 / cos_min 0.99995. So the structural check is RELATIVE to the model's own noise floor, not absolute.
        fm = res["full_model"]
        noise = max(fm["full_vs_full_again"]["rel_fro"], fm["truncated_vs_truncated_again"]["rel_fro"])
        c = fm["full_vs_truncated"]
        fm["verdict"] = {"noise_floor_rel_fro": noise, "full_vs_truncated_rel_fro": c["rel_fro"],
                         "structurally_equivalent": bool(c["rel_fro"] <= 2 * noise + 1e-4 and fm["full_again_vs_truncated_again"]["rel_fro"] <= 2 * noise + 1e-4)}
        print(json.dumps(fm["verdict"], indent=1), flush=True)
        assert fm["verdict"]["structurally_equivalent"], f"truncated model differs from the full model beyond its own noise floor: {fm}"
    gen.close()
    json.dump(res, open(f"{ROOT}/gen_check.json", "w"), indent=1, default=str)
    vol.commit()
    return res


# ----------------------------------------------------------------------------------------------------------------
# 3) merge shards -> dictionary_learning ae.pt (CPU)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, cpu=16, memory=256 * 1024, ephemeral_disk=1024 * 1024, volumes={"/data": vol}, timeout=8 * 3600)
def merge(save_dir: str = SHARDS, out_dir: str = TRAINER0, step: int | None = None, require_done: bool = True):
    """8 x ~32 GB shard checkpoints (mmap) -> <out_dir>/ae.pt (bf16 encoder.weight [F,d] / decoder.weight [d,F], fp32 biases,
    k, threshold; norm_factor folded so the SAE eats RAW layer-42 residuals) + config.json."""
    import json, subprocess, time
    vol.reload()
    if require_done:
        assert os.path.exists(f"{save_dir}/TRAIN_DONE"), f"{save_dir}/TRAIN_DONE missing (pass require_done=False to merge a partial run)"
    t0 = time.time()
    cmd = ["python", "/app/sae2m/sae27b_merge_shards.py", "--save-dir", save_dir, "--out-dir", out_dir] + (["--step", str(step)] if step else [])
    print("[sae2m]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=_env(), check=True, cwd="/app/sae2m")
    vol.commit()
    cfg = json.load(open(f"{out_dir}/config.json"))
    res = {"ae": f"{out_dir}/ae.pt", "bytes": os.path.getsize(f"{out_dir}/ae.pt"), "trainer": cfg["trainer"], "wall_min": (time.time() - t0) / 60}
    print(json.dumps(res, indent=1, default=str), flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------------
# 4) verify on freshly generated tokens (1 GPU)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=ONE_GPU, cpu=16, memory=256 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=6 * 3600)
def verify(ae: str = f"{TRAINER0}/ae.pt", eval_tokens: int = 20_000_000, dataset_skip: int = 0, out_json: str = f"{ROOT}/verify.json"):
    """Format check + EV / L0 / fired-fraction of the merged 2M SAE on eval_tokens freshly generated layer-42 tokens from
    Ultra-FineWeb docs [dataset_skip, ...) (default 0 = the reserved eval head the training stream excluded)."""
    import json, subprocess, time
    vol.reload()
    os.makedirs(ROOT, exist_ok=True)
    _ensure_model()
    t0 = time.time()
    cmd = ["python", "/app/sae2m/sae27b_verify_2m.py", "--ae", ae, "--online", "--eval-tokens", str(eval_tokens), "--dataset-skip", str(dataset_skip),
           "--model", MODEL, "--layer", str(LAYER), "--d", str(D_MODEL), "--out-batch", "4096", "--chunk", "131072", "--out-json", out_json]
    print("[sae2m]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=_env(), check=True, cwd="/app/sae2m")
    vol.commit()
    res = json.load(open(out_json))
    res["wall_min"] = (time.time() - t0) / 60
    print(json.dumps({k: v for k, v in res.items() if k != "gen"}, indent=1, default=str), flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------------
# 5) max-activating examples (8 GPUs, data-parallel over the SAME 1B-token span) + merge
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, gpu=TRAIN_GPU, cpu=48, memory=384 * 1024, ephemeral_disk=1024 * 1024, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=24 * 3600)
def maxacts(ae: str = f"{TRAINER0}/ae.pt", max_tokens: int = 1_000_000_000, topn: int = 5, window: int = 32, dataset_skip: int = 100_000,
            out_dir: str = f"{ROOT}/maxacts_parts", final: str = f"{ROOT}/maxacts_top5.pt", batch_seqs: int = 16, save_every_min: float = 60.0,
            skip_torchrun_if_partials_final: bool = True):
    """Each rank: truncated 27B + the FULL bf16 encoder (21.5 GB) on its Ultra-FineWeb stream (identical streams to training:
    split_dataset_by_node then skip 100k), top-`topn` `window`-token windows ending at the peak per feature, fire counts;
    then sae27b_maxacts_merge.py -> `final` (+ .summary.json). max_tokens is the GLOBAL budget (split over ranks)."""
    import glob, json, subprocess, time
    import torch
    vol.reload()
    assert os.path.exists(ae), f"missing {ae} -- run merge first"
    _ensure_model()
    world = _n_gpus()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    t0 = time.time()
    parts = sorted(glob.glob(f"{out_dir}/maxacts_part_r*.pt"))
    have_final = len(parts) == world and all(torch.load(p, map_location="cpu", weights_only=False, mmap=True).get("final") for p in parts)
    if not (have_final and skip_torchrun_if_partials_final):
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={world}", "/app/sae2m/sae27b_maxacts_sharded.py", "--ae", ae, "--out-dir", out_dir,
               "--model", MODEL, "--layer", str(LAYER), "--dataset", "openbmb/Ultra-FineWeb", "--split", "en", "--dataset-skip", str(dataset_skip),
               "--ctx-len", "512", "--window", str(window), "--topn", str(topn), "--batch-seqs", str(batch_seqs), "--feat-chunk", "131072",
               "--max-tokens", str(max_tokens), "--norm-mult", "10", "--d-model", str(D_MODEL), "--save-every-min", str(save_every_min)]
        rc = _run(cmd, _env(), f"{ROOT}/logs/maxacts_{time.strftime('%Y%m%d_%H%M%S')}.log")
        print(f"[sae2m] maxacts torchrun rc={rc} (a benign teardown abort after the partials are written is tolerated)", flush=True)
        parts = sorted(glob.glob(f"{out_dir}/maxacts_part_r*.pt"))
        assert len(parts) == world, f"expected {world} partials, found {len(parts)}: {parts}"
    t_m = time.time()
    cmd = ["python", "/app/sae2m/sae27b_maxacts_merge.py", "--in-dir", out_dir, "--out", final, "--expect-world", str(world)]
    print("[sae2m]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=_env(), check=True, cwd="/app/sae2m")
    vol.commit()
    summ = json.load(open(os.path.splitext(final)[0] + ".summary.json"))
    summ.update({"final": final, "bytes": os.path.getsize(final), "wall_h": (time.time() - t0) / 3600, "merge_min": (time.time() - t_m) / 60})
    print(json.dumps(summ, indent=1, default=str), flush=True)
    return summ


# ----------------------------------------------------------------------------------------------------------------
# 6) status / cleanup (CPU)
# ----------------------------------------------------------------------------------------------------------------
@app.function(image=image, cpu=2, memory=8192, volumes={"/data": vol}, timeout=1800)
def status(root: str = ROOT, tail_log: int = 40):
    """Files + sizes under /data/sae2m, latest.json / TRAIN_DONE, gen_check / verify / summary json, tail of the newest log."""
    import glob, json
    vol.reload()
    res = {"files": [(p, f"{s / 2**30:.2f} GB") for p, s in _tree(root, depth=3) if s > 50 * 2**20], "total_gb": sum(s for _, s in _tree(root, depth=3)) / 2**30}
    for name in ("shards/latest.json", "shards/TRAIN_DONE", "shards/config.json", "shards_smoke/latest.json", "gen_check.json", "verify.json",
                 "maxacts_top5.summary.json"):
        p = f"{root}/{name}"
        if os.path.exists(p):
            try:
                res[name] = json.load(open(p)) if p.endswith(".json") else open(p).read().strip()
            except Exception as e:  # noqa
                res[name] = f"unreadable: {e}"
    logs = sorted(glob.glob(f"{root}/logs/*.log"), key=os.path.getmtime)
    if logs:
        with open(logs[-1], "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 20000))
            res["log_tail"] = {"path": logs[-1], "lines": f.read().decode(errors="replace").splitlines()[-tail_log:]}
    return res


@app.function(image=image, cpu=2, memory=8192, volumes={"/data": vol}, timeout=3600)
def cleanup(path: str):
    """rm -r of a path under /data/sae2m ONLY (e.g. the smoke shards). Refuses anything else."""
    import shutil
    vol.reload()
    assert os.path.abspath(path).startswith(ROOT + "/") and os.path.abspath(path) not in (SHARDS, TRAINER0), f"refusing to remove {path}"
    n = sum(s for _, s in _tree(path, depth=5))
    shutil.rmtree(path, ignore_errors=True)
    vol.commit()
    return {"removed": path, "gb": n / 2**30}
