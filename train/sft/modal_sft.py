"""Modal app: universal-inverter SFT (train/sft/pretrain.py) on 8xB200, one container per datamix.

The trainer lives at train/sft/pretrain.py in this repo; it is mounted into the container at
/app/SL/pretrain.py, with maem/ importable via
PYTHONPATH=/app/helpers. Same image/volume as modal_rl.py (no vllm — pretrain is pure HF).
Data lives on the `maem-data` Volume:
    /data/<bank>            SFT bank: records.jsonl ({"vec_idx", "target_text"} per line)
                            + vecs.f32 or vecs.f16 (N x 5120 f32/f16 memmap, N >= max(vec_idx)+1)
                            --data-dir may be a COMMA-SEPARATED list of such PART banks: every part is staged and
                            pretrain.py consumes them as one virtual bank (100M+ rows no longer fit one merged file)
    /data/hf_cache          HF_HOME (Qwen/Qwen3.6-27B downloads once, persists)
    /data/sft_mix/<run>/    output: run_meta.json + heartbeat + step_* ckpts + final

Every run is an independent spawn — launch MANY datamixes in parallel:
    modal deploy modal_sft.py                     # registers train + the auto-resume supervisor
    modal run modal_sft.py::launch --run-name mix-a --data-dir /data/banks/mix_a
    modal run modal_sft.py::launch --run-name mix-b --data-dir /data/banks/mix_b
Each run saves --n-ckpts (default 14) evenly-spaced intermediate ckpts + final under
/data/sft_mix/<run-name>/ (the per-SFT-step curves; n-ckpts=0 = only `final`, which the harness
refuses). The supervisor respawns any run whose
heartbeat went stale before `final` exists (24h Modal cap / crash), resuming from the latest
step_N via --init-adapter + --skip-steps (batch order is deterministic) + --wandb-id.
Pause auto-resume: `touch /data/sft_mix/resume_paused` (global) or `<run>/resume_paused`.

Needs Modal secrets `maem-hf` (HF_TOKEN) and `maem-wandb` (WANDB_API_KEY).
"""

from pathlib import Path

import os

import modal

REPO = Path(__file__).resolve().parent.parent.parent   # repo root (this launcher lives one level down)

APP_NAME = os.environ.get("SFT_APP_NAME", "maem-sft-8xb200")   # SFT_APP_NAME=maem-sft-fullft = a second, independent deployment
app = modal.App(APP_NAME)

# torch 2.10.0+cu128 == the training venv; cu128 wheels carry sm_100 (B200) kernels. Identical pins to
# modal_rl.py so both trainers see one environment; pretrain needs no vllm (pure HF fwd/bwd).
# --prefix-cache (train/sft/prefix_cache.py) needs the transformers fork with autograd-safe linear-attention cache writes:
#   SFT_TRANSFORMERS="transformers @ git+https://github.com/ANONYMOUS/transformers@<commit>"
# Default stays the stock pin so existing deployments rebuild nothing.
_SFT_TRANSFORMERS = os.environ.get("SFT_TRANSFORMERS", "transformers==5.15.0")
image = modal.Image.debian_slim(python_version="3.11")
if "git+" in _SFT_TRANSFORMERS:
    image = image.apt_install("git")
image = (
    image
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        _SFT_TRANSFORMERS,
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    # fla: transformers' Qwen3.5/3.6 GatedDeltaNet uses flash-linear-attention's Triton chunk kernels when importable,
    # else a pure-torch fallback (the fallback measured 10.6% MFU). Same pin as the RL image.
    .pip_install("flash-linear-attention==0.5.2")
    # torchao: float8 training linears for --fp8-base (train/sft/fp8.py). 0.16.0 is the release built against torch 2.10.0
    # (pytorch/ao#2919 compat table; 0.17+ target 2.11). Pure-python path (casts + torch._scaled_mm), no torchao kernels.
    .pip_install("torchao==0.16.0")
)
# Hopper (H100/H200) opt-in at deploy/run time: SFT_TRITON=3.7.1. fla 0.5.2 refuses its gated chunk_bwd_dqkwg on Triton
# 3.4-3.7.0 on Hopper (fla #640). No vLLM in this image, so the global Triton can simply be upgraded.
_SFT_TRITON = os.environ.get("SFT_TRITON", "")
if _SFT_TRITON:
    image = image.pip_install(f"triton=={_SFT_TRITON}")
# train()'s ephemeral disk (GiB) -- a deploy-time knob. Baked into the image env too, because inside the container
# shutil.disk_usage reports a fictitious 2^33 GiB filesystem, so _preflight can only check the staging total against this.
SFT_DISK_GB = int(os.environ.get("SFT_DISK_GB", "600"))
image = (
    image
    .env({"SFT_DISK_GB": str(SFT_DISK_GB)})
    .add_local_file(REPO / "train" / "sft" / "pretrain.py", "/app/SL/pretrain.py")
    .add_local_file(REPO / "train" / "sft" / "prefix_cache.py", "/app/SL/prefix_cache.py")   # --prefix-cache sibling import
    .add_local_file(REPO / "train" / "sft" / "fullft.py", "/app/SL/fullft.py")               # --full-ft (FSDP2) sibling import
    .add_local_file(REPO / "train" / "sft" / "fp8.py", "/app/SL/fp8.py")            # --fp8-base (imported by pretrain.py)
    .add_local_dir(REPO / "maem", "/app/helpers/maem", ignore=["__pycache__"])
)

vol = modal.Volume.from_name("maem-data", create_if_missing=True)

SFT_ROOT = "/data/sft_mix"
STALE_HEARTBEAT_S = 30 * 60   # live legs touch+commit the heartbeat every <=5 min; 30 min = dead
SPAWN_COOLDOWN_S = 3600       # never double-spawn while a fresh leg pulls image/model (~15-30 min)


VEC_BANK_FILES = (("vecs.f32", 4), ("vecs.f16", 2))   # (filename, bytes per element); same as pretrain.py


def _vec_bank_file(data_dir: str):
    """(filename, itemsize) of the vector bank in data_dir -- vecs.f32 preferred, else vecs.f16."""
    import os

    for fname, itemsize in VEC_BANK_FILES:
        if os.path.exists(f"{data_dir}/{fname}"):
            return fname, itemsize
    raise AssertionError(f"bank incomplete: neither vecs.f32 nor vecs.f16 in {data_dir}")


def _preflight(run_name: str, data_dir: str, n_ckpts: int, resume_from: str, disk_cap_gb: int = 0):
    """Shared guardrails + bank staging for train/smoke. `data_dir` is one bank or a comma-separated list of PART banks
    (pretrain.py --data-dir takes the same list and consumes the parts as one virtual bank -- records and vector rows
    concatenated in the listed order). EVERY part is staged to container-local disk; the total is printed and, when
    disk_cap_gb > 0 (train passes its deploy-time SFT_DISK_GB), refused up front if it cannot fit.
    Returns (save_dir, staged comma list, d_model)."""
    import glob
    import json
    import math
    import os
    import shutil
    import sys
    import time

    # ---- ckpt-cadence guard: n_ckpts=0 means pretrain.py saves ONLY `final` — that is the exact
    # bug that lost the per-SFT-step curves. Refuse it for real runs. ----
    assert n_ckpts > 0, "n_ckpts must be > 0 (0 = only `final`, no per-SFT-step curve — the old bug)"
    assert run_name and "/" not in run_name, f"bad run_name {run_name!r} (path component, no slashes)"

    # ---- resume-support guard: the mounted trainer MUST carry the crash-resume flags the
    # supervisor relies on (--skip-steps fast-forward, --wandb-id, --n-ckpts). Refuse stale code. ----
    with open("/app/SL/pretrain.py") as f:
        _src = f.read()
    for _flag in ("--n-ckpts", "--skip-steps", "--wandb-id"):
        assert _flag in _src, f"mounted pretrain.py is stale: {_flag} missing (resume/ckpt support)"
    print("[modal] mounted-trainer check OK (--n-ckpts/--skip-steps/--wandb-id present)", flush=True)

    sys.path.insert(0, "/app/helpers")
    from maem.config import D_MODEL, MODEL  # single source of truth (5120, Qwen/Qwen3.6-27B)

    save_dir = f"{SFT_ROOT}/{run_name}"
    if os.path.exists(f"{save_dir}/final"):
        raise RuntimeError(f"{save_dir}/final exists — run complete; pick a new --run-name")
    prior = sorted(glob.glob(f"{save_dir}/step_*"))
    if prior and not resume_from:
        raise RuntimeError(f"{save_dir} already has {len(prior)} ckpts — a fresh leg would clobber "
                           "them; pick a new --run-name (the supervisor passes resume_from)")

    # single-flight base-model download into the persistent volume (avoids 8 ranks racing)
    os.environ["HF_HOME"] = "/data/hf_cache"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal] base model in cache ({time.time() - t0:.0f}s)", flush=True)

    # ---- bank schema guard PER PART, then stage every part onto container-local NVMe: memmap over the volume
    # FUSE mount is the one thing we don't trust, and per-batch random row reads are faster locally. ----
    parts = [p.strip() for p in data_dir.split(",") if p.strip()]
    assert parts, f"empty data_dir {data_dir!r}"
    sizes, n_vecs_total, n_ex_total = [], 0, 0
    for d in parts:
        assert os.path.isdir(d), f"bank not found on volume: {d}"
        assert os.path.exists(f"{d}/records.jsonl"), f"bank incomplete: {d}/records.jsonl missing"
        vec_file, itemsize = _vec_bank_file(d)   # vecs.f32 or vecs.f16 (pretrain.py reads either)
        rec0 = json.loads(open(f"{d}/records.jsonl").readline())
        assert "vec_idx" in rec0 and "target_text" in rec0, f"records.jsonl schema: got {sorted(rec0)}"
        vsize = os.path.getsize(f"{d}/{vec_file}")
        assert vsize % (D_MODEL * itemsize) == 0, \
            f"{d}/{vec_file} = {vsize} B, not a multiple of one {D_MODEL}-x-{itemsize}B row"
        rsize = os.path.getsize(f"{d}/records.jsonl")
        n_ex = None
        if os.path.exists(f"{d}/build_stats.json"):
            n_ex = json.load(open(f"{d}/build_stats.json")).get("n_examples")
            n_ex_total += n_ex or 0
        n_vecs_total += vsize // (D_MODEL * itemsize)
        sizes.append(vsize + rsize)
        print(f"[modal] bank OK: {d}: {vsize // (D_MODEL * itemsize)} vecs x {D_MODEL} ({vec_file}, {vsize / 2**30:.1f} GiB) "
              f"+ records.jsonl ({rsize / 2**30:.2f} GiB)" + (f", n_examples={n_ex}" if n_ex is not None else ""), flush=True)
    total_gb = sum(sizes) / 2**30
    if len(parts) > 1:
        print(f"[modal] {len(parts)} bank parts -> one virtual bank: {n_vecs_total} vectors, build_stats n_examples sum "
              f"{n_ex_total}, {total_gb:.1f} GiB to stage", flush=True)

    # ---- stage: unique local dir per DISTINCT source (the same part listed twice is staged once and passed twice);
    # the total must fit the container's ephemeral disk (deploy-time SFT_DISK_GB) or copytree dies hours in. ----
    local_of = {}
    staged, taken = [], {}
    for d in parts:
        if d in local_of:
            staged.append(local_of[d])
            continue
        base = f"/root/bank_{os.path.basename(d.rstrip('/'))}"
        local = base
        k = 1
        while local in taken:   # a DIFFERENT source with the same basename
            local = f"{base}_{k}"; k += 1
        taken[local] = d
        local_of[d] = local
        staged.append(local)
    size_of = dict(zip(parts, sizes))                     # a source listed twice is copied (and counted) once
    need = sum(sz for d, sz in size_of.items() if not os.path.exists(local_of[d]))
    margin_gb = 20                                        # wandb dir, /tmp shards (--full-ft nontext shard), slack
    du = shutil.disk_usage("/root")
    du_real = du.total < 2**50                            # Modal's sandbox reports a 2^33 GiB filesystem: meaningless
    print(f"[modal] staging {len(local_of)} distinct part(s), {need / 2**30:.1f} GiB to copy (total corpus {total_gb:.1f} GiB, "
          f"+{margin_gb} GiB margin); local disk cap: "
          + (f"{disk_cap_gb} GiB (ephemeral_disk = SFT_DISK_GB at deploy)" if disk_cap_gb else "not enforced for this function")
          + (f"; statvfs says {du.free / 2**30:.0f} GiB free of {du.total / 2**30:.0f} GiB" if du_real
             else "; statvfs reports a fictitious filesystem size and is ignored"), flush=True)
    need_gb = need / 2**30 + margin_gb
    cap_gb = min(disk_cap_gb or float("inf"), du.free / 2**30 if du_real else float("inf"))
    if need_gb > cap_gb:
        raise RuntimeError(f"not enough local disk to stage the bank(s): need {need_gb:.0f} GiB (incl. margin) but the cap is "
                           f"{cap_gb:.0f} GiB -- redeploy with SFT_DISK_GB={int(math.ceil(need_gb / 100.0)) * 100 + 200} or larger")
    t_all = time.time()
    _stage_parallel(local_of, size_of)
    print(f"[modal] bank staged to {','.join(staged)} ({len(staged)} part(s), {total_gb:.1f} GiB, {time.time() - t_all:.0f}s)", flush=True)
    return save_dir, ",".join(staged), D_MODEL


def _copy_range(src: str, dst: str, start: int, end: int, chunk: int = 64 * 2**20, progress=None):
    """Copy bytes [start, end) of src into the SAME offsets of a pre-sized dst (pread/pwrite; parallel-safe)."""
    fi = os.open(src, os.O_RDONLY); fo = os.open(dst, os.O_WRONLY)
    try:
        pos = start
        while pos < end:
            b = os.pread(fi, min(chunk, end - pos), pos)
            if not b:
                raise IOError(f"short read at {pos} of {src}")
            os.pwrite(fo, b, pos); pos += len(b)
            if progress is not None:
                progress[0] += len(b)
    finally:
        os.close(fi); os.close(fo)


def _stage_parallel(local_of: dict, size_of: dict, streams_per_big_file: int = 4, big_file_bytes: int = 4 * 2**30,
                    stall_mib_s: float = 250.0, stall_minutes: float = 5.0):
    """Stage every part with MANY parallel streams: one worker per small file, `streams_per_big_file` byte-range workers per
    file > big_file_bytes (vecs.f16 is 85-512 GiB). Modal's volume read path throttles a single long-running stream to ~20 MiB/s
    after a burst (measured Sep 7: 1.2 GB/s -> 19 MiB/s), while fresh streams read at 400-700 MiB/s, so parallel ranges
    multiply throughput. A monitor prints the aggregate rate every 60 s and raises if it stays below `stall_mib_s` for
    `stall_minutes` (after a 3-min grace) so a throttled leg fails fast and gets respawned instead of crawling for hours."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = []  # (fn, args)
    progress = [0]; total = 0
    for d, local in local_of.items():
        if os.path.exists(local) and os.path.exists(f"{local}/.staged"):
            print(f"[modal] bank part already staged: {local}", flush=True); continue
        os.makedirs(local, exist_ok=True)
        for root, _, files in os.walk(d):
            rel = os.path.relpath(root, d); os.makedirs(os.path.join(local, rel), exist_ok=True)
            for fn in files:
                src = os.path.join(root, fn); dst = os.path.join(local, rel, fn); n = os.path.getsize(src); total += n
                with open(dst, "wb") as f:
                    if n:
                        f.truncate(n)
                if n > big_file_bytes:
                    step = -(-n // streams_per_big_file)
                    for k in range(streams_per_big_file):
                        a, b = k * step, min(n, (k + 1) * step)
                        if a < b:
                            jobs.append((_copy_range, (src, dst, a, b, 64 * 2**20, progress)))
                else:
                    jobs.append((_copy_range, (src, dst, 0, n, 64 * 2**20, progress)))
    if not jobs:
        return
    print(f"[modal] parallel staging: {len(jobs)} streams over {len(local_of)} part(s), {total / 2**30:.1f} GiB", flush=True)
    stop = threading.Event(); t0 = time.time(); hist = []

    def monitor():
        last = 0
        while not stop.wait(60):
            now = progress[0]; rate = (now - last) / 60 / 2**20; last = now; hist.append(rate)
            print(f"[modal] staging {now / 2**30:.0f}/{total / 2**30:.0f} GiB ({100 * now / max(total, 1):.0f}%) at {rate:.0f} MiB/s "
                  f"(avg {now / 2**20 / max(time.time() - t0, 1):.0f} MiB/s, {(time.time() - t0) / 60:.0f} min)", flush=True)
            if len(hist) >= 3 + stall_minutes and all(r < stall_mib_s for r in hist[-int(stall_minutes):]):
                print(f"[modal] STAGING THROTTLED: < {stall_mib_s} MiB/s for {stall_minutes:.0f} min -- aborting this leg (respawn lands on a fresh host)", flush=True)
                os._exit(75)
    th = threading.Thread(target=monitor, daemon=True); th.start()
    with ThreadPoolExecutor(max_workers=min(16, len(jobs))) as ex:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for f in as_completed(futs):
            f.result()
    stop.set()
    for local in local_of.values():
        open(f"{local}/.staged", "w").close()
    print(f"[modal] parallel staging done: {progress[0] / 2**30:.1f} GiB in {time.time() - t0:.0f}s ({progress[0] / 2**20 / max(time.time() - t0, 1):.0f} MiB/s)", flush=True)


def _train_env(backend: str):
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/helpers"
    # pretrain.py is STANDARD cuda-tensor DDP (unlike rl.py's by-design CPU collectives that force
    # gloo) — nccl is the correct default here; gloo is only a fallback.
    env["DDP_BACKEND"] = backend
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_DIR"] = "/tmp/wandb"          # /app is a read-only mount; wandb writes to cwd otherwise
    # ranks must load PURELY from the validated cache: 8 concurrent hub re-resolutions returned
    # spurious missing-shard errors on the RL app even though the cached snapshot was complete.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # variable-length padded batches fragment the caching allocator on 178GB B200s (RL app OOM'd on
    # a 24MB alloc with 159GB allocated); expandable_segments is the canonical fix.
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs("/tmp/wandb", exist_ok=True)
    return env


def _stream(cmd, env, tag):
    import subprocess

    print(f"[{tag}] launching:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd="/app", env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


@app.function(
    image=image,
    gpu=os.environ.get("SFT_GPU", "B200:8"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maem-hf"),
        modal.Secret.from_name("maem-wandb"),
    ],
    timeout=86400,
    # _preflight stages EVERY listed bank part locally (vecs.f16 = 10 KiB/row + ~190 B/row of records): 23M rows = 224 GiB,
    # 50M = 487 GiB (SFT_DISK_GB=1200), 104M (realact_short_50m_all + parts h..m) = 1012 GiB -> SFT_DISK_GB>=1400,
    # 203M (all 24 realact_short_20m* parts) = 1971 GiB -> SFT_DISK_GB=2600-3000 (Modal's ephemeral-disk ceiling is ~3 TiB)
    ephemeral_disk=SFT_DISK_GB * 1024,
    memory=256 * 1024,
)
def train(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
          batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
          backend: str = "nccl", extra_args: str = "",
          resume_from: str = "", skip_steps: int = 0, wandb_id: str = ""):
    # One SFT run (one datamix). run_name/data_dir/n_ckpts parameterize independent parallel
    # spawns. batch_size/lr/max_seq: 0 = TrainConfig defaults (64 / 3e-5 / 192). extra_args:
    # whitespace-split, appended last (argparse last-wins), e.g. "--compile --log-steps 50".
    # resume_from/skip_steps/wandb_id: supervisor crash-resume — --init-adapter <ckpt> +
    # --skip-steps fast-forward through the deterministic batch order + same wandb run.
    import glob
    import json
    import os
    import secrets as pysecrets
    import string
    import threading
    import time

    # one bank or a comma-separated list of parts (each may be volume-relative); the normalized list is what run_meta
    # records, so the supervisor's resume legs stage exactly the same parts in the same order
    data_dir = ",".join(p if p.startswith("/") else f"/data/{p}" for p in (q.strip() for q in data_dir.split(",")) if p)
    save_dir, local_bank, _ = _preflight(run_name, data_dir, n_ckpts, resume_from,
                                         disk_cap_gb=int(os.environ.get("SFT_DISK_GB", "0")))   # baked into the image at deploy
    os.makedirs(save_dir, exist_ok=True)

    # ---- run_meta.json: everything the supervisor needs to respawn this run faithfully. Written
    # once on the first leg (wandb id minted here so every leg logs to ONE wandb run). ----
    meta_path = f"{save_dir}/run_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        wandb_id = wandb_id or meta.get("wandb_id", "")
    else:
        wandb_id = wandb_id or "".join(
            pysecrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        meta = {"run_name": run_name, "data_dir": data_dir, "n_ckpts": n_ckpts, "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "max_seq": max_seq, "backend": backend,
                "extra_args": extra_args, "wandb_id": wandb_id, "created": time.time(), "legs": []}
    meta["legs"].append({"ts": time.time(), "resume_from": resume_from, "skip_steps": skip_steps})
    json.dump(meta, open(meta_path, "w"), indent=2)

    # ---- heartbeat + committer: touch <run>/heartbeat every 60s (supervisor aliveness signal,
    # started BEFORE the slow model download so fresh legs read as alive immediately) and
    # vol.commit right after every new ckpt lands (else every 300s) so saves are durable even if
    # the container dies mid-run. ----
    def _committer():
        last_commit, last_ckpts = 0.0, -1
        while True:
            try:
                with open(f"{save_dir}/heartbeat", "w") as f:
                    f.write(str(time.time()))
                n = len(glob.glob(f"{save_dir}/step_*")) + len(glob.glob(f"{save_dir}/final"))
                if n != last_ckpts or time.time() - last_commit > 300:
                    vol.commit()
                    last_commit, last_ckpts = time.time(), n
            except Exception as e:
                print(f"[modal] heartbeat/commit failed: {e}", flush=True)
            time.sleep(60)

    threading.Thread(target=_committer, daemon=True).start()
    vol.commit()  # publish run_meta + heartbeat now: the supervisor must see this leg exists

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=8", "SL/pretrain.py",
        "--data-dir", local_bank,
        "--save-dir", save_dir,
        "--run-name", run_name,
        "--n-ckpts", str(n_ckpts),
        "--epochs", str(epochs),
        "--wandb-id", wandb_id,
    ]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if lr:
        cmd += ["--lr", str(lr)]
    if max_seq:
        cmd += ["--max-seq", str(max_seq)]
    if resume_from:
        cmd += ["--init-adapter", resume_from, "--skip-steps", str(skip_steps)]
    if extra_args:
        cmd += extra_args.split()

    rc = _stream(cmd, _train_env(backend), "modal")
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"torchrun exited rc={rc} (backend={backend}) — supervisor will resume")
    assert os.path.exists(f"{save_dir}/final"), "rc=0 but no final ckpt — trainer exited early?"
    print(f"[modal] SFT run {run_name} COMPLETE -> {save_dir}/final", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("SFT_SMOKE_GPU", "B200:1"),
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("maem-hf"),
        modal.Secret.from_name("maem-wandb"),
    ],
    timeout=7200,
    memory=int(os.environ.get("SFT_SMOKE_MEM_GB", "64")) * 1024,
)
def smoke(data_dir: str, n_records: int = 256, batch_size: int = 8, extra_args: str = "", nproc: int = 1,
          save_dir: str = "/tmp/smoke_ckpt", reload_check: bool = False):
    """1xB200 pipeline validation: pre-warms /data/hf_cache, carves a tiny bank (first n_records
    of records.jsonl + the full vecs.f32 / vecs.f16) and runs world=1 SFT over it (no wandb, ckpts
    to /tmp, including the --n-ckpts intermediate-save path). extra_args reach pretrain.py verbatim,
    e.g. "--compile --grad-ckpt 0 --autocast-bf16 --head-on-labels --parity-check --log-steps 10"."""
    import os
    import shutil

    data_dir = ",".join(p if p.startswith("/") else f"/data/{p}" for p in (q.strip() for q in data_dir.split(",")) if p)
    _, local_bank, _ = _preflight("smoke", data_dir, n_ckpts=2, resume_from="")
    # one tiny bank per staged part (first n_records of its records.jsonl + its full vecs file) -> a comma list exercises
    # pretrain.py's multi-part path end to end (a single part keeps the legacy /root/bank_smoke name)
    parts = local_bank.split(",")
    tinies = []
    for k, part in enumerate(parts):
        vec_file, _ = _vec_bank_file(part)
        tiny = "/root/bank_smoke" if len(parts) == 1 else f"/root/bank_smoke_{k}"
        if not os.path.exists(tiny):
            os.makedirs(tiny)
            with open(f"{part}/records.jsonl") as fin, open(f"{tiny}/records.jsonl", "w") as fout:
                for i, line in enumerate(fin):
                    if i >= n_records:
                        break
                    fout.write(line)
            shutil.copy(f"{part}/{vec_file}", f"{tiny}/{vec_file}")
        tinies.append(tiny)
        print(f"[modal-smoke] tiny bank {k}: first {n_records} records of {part} -> {tiny}", flush=True)
    tiny = ",".join(tinies)

    launcher = ["torchrun", "--standalone", f"--nproc_per_node={nproc}"] if nproc > 1 else ["python"]
    cmd = launcher + [
        "SL/pretrain.py",
        "--data-dir", tiny,
        "--save-dir", save_dir,
        "--run-name", "smoke",
        "--n-ckpts", "2",
        "--epochs", "1",
        "--batch-size", str(batch_size),
        "--no-wandb",
    ]
    if extra_args:
        cmd += extra_args.split()   # e.g. "--compile --grad-ckpt 0 --autocast-bf16 --log-steps 10" (speed test)
    rc = _stream(cmd, _train_env("nccl"), "modal-smoke")
    if rc != 0:
        raise RuntimeError(f"smoke exited rc={rc}")
    saved = sorted(os.listdir(save_dir))
    assert "final" in saved and any(s.startswith("step_") for s in saved), f"ckpt cadence broken: {saved}"
    print(f"[modal-smoke] OK — saved {saved}", flush=True)
    if save_dir.startswith("/data"):
        vol.commit()
    if reload_check:   # --full-ft: the checkpoint must load as a plain HF model and give a sane loss on the tiny bank
        chk = ["python", "-c", f"""
import json, os, sys, time, torch
sys.path.insert(0, '/app/helpers'); os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer
from maem.config import MODEL
from maem.prompts import build_sft_ids
ck = '{save_dir}/final'
assert os.path.exists(ck + '/SAVE_DONE'), 'SAVE_DONE missing'
print('[reload] SAVE_DONE', json.load(open(ck + '/SAVE_DONE')), flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={{'': 'cuda:0'}})
print(f'[reload] loaded {{type(m).__name__}} from {{ck}} in {{time.time() - t0:.0f}}s', flush=True)
recs = [json.loads(l) for _, l in zip(range(8), open('{tinies[0]}/records.jsonl'))]
losses = []
for r in recs:
    ids, labs, pos = build_sft_ids(tok, r['target_text'])
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids], device='cuda:0'), labels=torch.tensor([labs], device='cuda:0'))
    losses.append(out.loss.item())
print(f'[reload] CE on 8 tiny-bank examples (no injection): {{sum(losses) / len(losses):.3f}} (each {{[round(x, 2) for x in losses]}})', flush=True)
assert all(torch.isfinite(torch.tensor(losses))), 'non-finite loss after reload'
print('RELOAD_OK', flush=True)
"""]
        rc = _stream(chk, _train_env("nccl"), "modal-smoke-reload")
        if rc != 0:
            raise RuntimeError(f"reload check exited rc={rc}")


@app.function(image=image, timeout=600, cpu=2)
def env_check():
    """CPU-only image check: pins + the torchao float8 imports pretrain.py --fp8-base relies on."""
    import importlib
    import sys

    import torch
    print(f"python {sys.version.split()[0]} torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
    for pkg in ("torchao", "transformers", "peft", "accelerate", "fla", "triton"):
        try:
            print(f"{pkg} {getattr(importlib.import_module(pkg), '__version__', '?')}")
        except Exception as e:  # noqa
            print(f"{pkg} IMPORT FAILED: {e!r}")
    from torchao.float8 import Float8LinearConfig
    from torchao.float8.float8_linear import Float8Linear
    from torchao.float8.float8_linear_utils import swap_linear_layers
    sys.path.insert(0, "/app/SL")
    import fp8
    print(f"torchao float8 API OK: {Float8Linear.__name__}, {swap_linear_layers.__name__}")
    print(f"fp8.py OK: recipe default={fp8.DEFAULT_RECIPE} MIN_DIM={fp8.MIN_DIM} "
          f"config={Float8LinearConfig.from_recipe_name(fp8.DEFAULT_RECIPE)}")
    print("ENV_CHECK_OK", flush=True)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("maem-hf")],
    timeout=7200,
    cpu=8,
)
def prewarm():
    """CPU-only: download the base model into the persistent HF cache on the volume."""
    import os
    import sys
    import time

    os.environ["HF_HOME"] = "/data/hf_cache"
    sys.path.insert(0, "/app/helpers")
    from maem.config import MODEL
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(MODEL)
    vol.commit()
    print(f"[modal-prewarm] base model cached ({time.time() - t0:.0f}s)", flush=True)


# ---- auto-resume supervisor: Modal caps functions at 24h; big banks can outrun that, and crashed
# legs must not silently stall a datamix sweep. Every 20 min: for each /data/sft_mix/<run>/ with a
# run_meta.json and no `final`, if the heartbeat is stale (live legs touch+commit it every <=5 min)
# and no spawn is cooling down, respawn `train` in RESUME mode: --init-adapter <latest step_N> +
# --skip-steps N+1 (step_N is saved after batch N of the deterministic order) + the same wandb id.
# No step ckpts yet -> clean fresh respawn. Unlike the RL supervisor this is fully generic across
# N parallel runs: per-run heartbeat/state files, nothing hardcoded. ----
@app.function(schedule=modal.Period(minutes=20) if os.environ.get("SFT_SUPERVISOR", "0") == "1" else None,
              volumes={"/data": vol}, timeout=600)   # opt in with SFT_SUPERVISOR=1 at deploy: resumes preempted runs
def supervisor():
    import glob
    import json
    import os
    import time

    vol.reload()
    if os.path.exists(f"{SFT_ROOT}/resume_paused"):
        print(f"[supervisor] auto-resume PAUSED ({SFT_ROOT}/resume_paused present) — no spawns", flush=True)
        return
    for meta_path in sorted(glob.glob(f"{SFT_ROOT}/*/run_meta.json")):
        run_dir = os.path.dirname(meta_path)
        run = os.path.basename(run_dir)
        if os.path.exists(f"{run_dir}/final"):
            print(f"[supervisor] {run}: COMPLETE", flush=True)
            continue
        if os.path.exists(f"{run_dir}/resume_paused"):
            print(f"[supervisor] {run}: paused ({run}/resume_paused present)", flush=True)
            continue
        hb = f"{run_dir}/heartbeat"
        age = time.time() - (os.path.getmtime(hb) if os.path.exists(hb) else os.path.getmtime(meta_path))
        if age < STALE_HEARTBEAT_S:
            print(f"[supervisor] {run}: alive (heartbeat {age / 60:.0f} min old)", flush=True)
            continue
        st_path = f"{run_dir}/resume_state.json"
        st = json.load(open(st_path)) if os.path.exists(st_path) else {}
        if time.time() - st.get("last_spawn_ts", 0) < SPAWN_COOLDOWN_S:
            print(f"[supervisor] {run}: stale but in post-spawn cooldown ({st.get('call')})", flush=True)
            continue
        meta = json.load(open(meta_path))
        steps = sorted(int(p.rsplit("_", 1)[-1]) for p in glob.glob(f"{run_dir}/step_*")
                       if p.rsplit("_", 1)[-1].isdigit())
        resume_from = f"{run_dir}/step_{steps[-1]}" if steps else ""
        skip = steps[-1] + 1 if steps else 0  # step_N lands after batch N -> skip batches 0..N
        print(f"[supervisor] {run}: DEAD (heartbeat {age / 60:.0f} min old) — respawning "
              f"{'from ' + resume_from + f' skip={skip}' if steps else 'FRESH (no ckpts yet)'}", flush=True)
        call = modal.Function.from_name(APP_NAME, "train").spawn(
            run_name=meta["run_name"], data_dir=meta["data_dir"], n_ckpts=meta["n_ckpts"],
            epochs=meta.get("epochs", 1), batch_size=meta.get("batch_size", 0),
            lr=meta.get("lr", 0.0), max_seq=meta.get("max_seq", 0),
            backend=meta.get("backend", "nccl"), extra_args=meta.get("extra_args", ""),
            resume_from=resume_from, skip_steps=skip, wandb_id=meta.get("wandb_id", ""))
        json.dump({"last_spawn_ts": time.time(), "call": call.object_id,
                   "from_step": steps[-1] if steps else -1}, open(st_path, "w"))
        vol.commit()
        print(f"[supervisor] {run}: resume leg spawned {call.object_id}", flush=True)


@app.local_entrypoint()
def launch(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
           batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
           backend: str = "nccl", extra_args: str = ""):
    """Spawn one SFT run on the DEPLOYED app (run `modal deploy modal_sft.py` first). Returns
    immediately — invoke once per datamix to train many mixes in parallel."""
    call = modal.Function.from_name(APP_NAME, "train").spawn(
        run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
        batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend, extra_args=extra_args)
    print(f"spawned SFT run {run_name!r} on bank {data_dir}: {call.object_id}")
    print(f"logs:  modal app logs {APP_NAME}   |   ckpts: /data/sft_mix/{run_name}/step_*")


@app.local_entrypoint()
def run_train(run_name: str, data_dir: str, n_ckpts: int = 14, epochs: int = 1,
              batch_size: int = 0, lr: float = 0.0, max_seq: int = 0,
              backend: str = "nccl", extra_args: str = ""):
    """Attached single run (live logs; use `modal run --detach` to survive disconnect)."""
    train.remote(run_name=run_name, data_dir=data_dir, n_ckpts=n_ckpts, epochs=epochs,
                 batch_size=batch_size, lr=lr, max_seq=max_seq, backend=backend,
                 extra_args=extra_args)


@app.local_entrypoint()
def run_smoke(data_dir: str, n_records: int = 256, batch_size: int = 8, extra_args: str = "", nproc: int = 1,
              save_dir: str = "/tmp/smoke_ckpt", reload_check: bool = False):
    smoke.remote(data_dir=data_dir, n_records=n_records, batch_size=batch_size, extra_args=extra_args, nproc=nproc,
                 save_dir=save_dir, reload_check=reload_check)


@app.local_entrypoint()
def run_prewarm():
    prewarm.remote()


@app.local_entrypoint()
def run_env_check():
    env_check.remote()

