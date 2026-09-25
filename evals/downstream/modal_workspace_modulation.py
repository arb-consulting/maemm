"""Modal launcher for evals/downstream/workspace_modulation: runs each stage of `python -m evals.downstream.workspace_modulation` as a
subprocess, GPU stages in their own containers (sharded where the package shards them), CPU stages in a CPU
container, level by level of `stages.levels()`. Run directories live under /data/workspace_modulation/<run_id>
unless EVAL_OUTPUT_DIR or EVAL_OUTPUT_DIR_WORKSPACE_MODULATION says otherwise.

    modal run --detach evals/downstream/modal_workspace_modulation.py::run_all --run-id smoke --smoke
    modal run --detach evals/downstream/modal_workspace_modulation.py::run_all --run-id <run_id>
    modal run --detach evals/downstream/modal_workspace_modulation.py::stage --name nla --run-id <run_id> --extra "--shard 1 --n-shards 2"
    modal run evals/downstream/modal_workspace_modulation.py::stage --name judge --run-id <run_id> --extra "--gate 20"
    modal run evals/downstream/modal_workspace_modulation.py::pull --run-id <run_id>

`--judge-profile` is `sonnet` (the default) or `sol` (needs EVAL_OPENROUTER_SECRET set). The app, volume,
secret, GPU and cost names and the GPU container cap are read from EVAL_* environment variables
(evals/downstream/modal.env.example, evals/downstream/README.md)."""

import os
import time
from pathlib import Path

import modal

# Nothing from evals.downstream.* is imported at module level: Modal re-imports this file inside the container, where
# evals.downstream.* is not on the path. Every evals.downstream.* import is inside a function a local entrypoint calls.

REPO = Path(__file__).resolve().parent.parent.parent
APP_NAME = os.environ.get("EVAL_APP", "maem-workspace-modulation")
app = modal.App(APP_NAME)
# One 27B checkpoint is about 54 GB; `rollouts` holds two (`TWO_MODEL_STAGES`), on a card with its own
# default, so a smaller EVAL_GPU never becomes the two-model card.
GPU = os.environ.get("EVAL_GPU", "B200:1")
GPU_BOTH_MODELS = os.environ.get("EVAL_GPU_BOTH_MODELS", "B200:1")
# GPU containers in flight at once.
GPU_WORKERS = int(os.environ.get("EVAL_GPU_WORKERS", "8"))
# The lens the image installs, at config.JLENS_COMMIT (checked by `_check_stage_lists`).
JLENS_TARBALL = "https://github.com/anthropics/jacobian-lens/archive/581d398613e5602a5af361e1c34d3a92ea82ba8e.tar.gz"
GPU_COST_PER_HOUR = float(os.environ.get("EVAL_GPU_COST_PER_HOUR", "6.2496"))  # list rate, for the estimate only
# "unset": the flag is not passed, so the package's own default (config.py) binds.
UNSET_BUDGET, UNSET_ARMS, UNSET_SHARDS = 0.0, "", 0
FROM_CONFIG = -1  # the seed is always passed; this sentinel resolves it from config.GEN_SEED locally

# Every EVAL_* variable this file reads at import, and the model pins' overrides (evals/downstream/common/pins.py), copied
# into the image so the container's re-import declares the same paths, secrets and checkpoints.
LAUNCH_VARS = (
    "EVAL_APP", "EVAL_GPU", "EVAL_GPU_BOTH_MODELS", "EVAL_GPU_WORKERS", "EVAL_GPU_COST_PER_HOUR",
    "EVAL_VOLUME", "EVAL_HF_SECRET", "EVAL_ANTHROPIC_SECRET", "EVAL_OPENROUTER_SECRET", "EVAL_OUTPUT_DIR",
    "EVAL_OUTPUT_DIR_WORKSPACE_MODULATION", "EVAL_HF_HOME",
) + tuple(f"EVAL_{m}_{f}" for m in ("BASE", "INVERTER", "NLA", "LENS") for f in ("REPO", "REVISION"))
LAUNCH_ENV = {k: os.environ[k] for k in LAUNCH_VARS if k in os.environ}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==5.15.0",
        "accelerate==1.14.0",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
        "matplotlib==3.10.8",
        "pyarrow",  # the corpus stage reads the pinned parquet files
    )
    # the base model's linear-attention kernels
    .pip_install("flash-linear-attention==0.5.2")
    .pip_install(JLENS_TARBALL)
    .env(LAUNCH_ENV)
    .add_local_dir(REPO / "maem", "/app/maem", ignore=["__pycache__", "out", "analysis", "test_*"])
    .add_local_dir(REPO / "evals" / "downstream", "/app/evals/downstream", ignore=["__pycache__", "out", "analysis", "test_*"])
)
VOLUME_NAME = os.environ.get("EVAL_VOLUME", "maem-data")
HF_SECRET_NAME = os.environ.get("EVAL_HF_SECRET", "maem-hf")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
HF = modal.Secret.from_name(HF_SECRET_NAME)
# The default profile's secret (`sonnet`: ANTHROPIC_API_KEY) is always mounted; the OpenRouter secret (`sol`:
# OPENROUTER_API_KEY) only when EVAL_OPENROUTER_SECRET names one.
ANTHROPIC_SECRET_NAME = os.environ.get("EVAL_ANTHROPIC_SECRET", "maem-anthropic")
OPENROUTER_SECRET_NAME = os.environ.get("EVAL_OPENROUTER_SECRET")
JUDGE_SECRETS = [modal.Secret.from_name(ANTHROPIC_SECRET_NAME)] + (
    [modal.Secret.from_name(OPENROUTER_SECRET_NAME)] if OPENROUTER_SECRET_NAME else [])

# EVAL_OUTPUT_DIR is the root every activation package writes under, as <root>/workspace_modulation.
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR_WORKSPACE_MODULATION") or os.path.join(
    os.environ.get("EVAL_OUTPUT_DIR", "/data"), "workspace_modulation")
HF_HOME = os.environ.get("EVAL_HF_HOME", "/data/hf_cache")
GPU_STAGES = {"capture", "rollouts", "patchscope", "lens", "nla", "retrieval"}
TWO_MODEL_STAGES = {"rollouts"}  # the base and the inverter at once (the inverter is freed before re-reads)
CPU_STAGES = {"prepare", "corpus", "cells_mean", "rollouts_merge", "nla_merge", "retrieval_merge",
              "summarise", "judge", "report"}


def _defaults():
    """(judge cap, arms, rollout shards, NLA shards, retrieval shards, seed) from the package's config."""
    from evals.downstream.workspace_modulation import config as C

    return (C.JUDGE_BUDGET_USD, ",".join(C.ARM_ORDER), C.ROLLOUT_SHARDS, C.NLA_SHARDS, C.RETRIEVAL_SHARDS,
            C.GEN_SEED)


def _check_stage_lists():
    """GPU_STAGES | CPU_STAGES is exactly the package's stage list, and the image's lens is the pinned commit."""
    from evals.downstream.workspace_modulation import config as C
    from evals.downstream.workspace_modulation.stages import ALL_STAGES

    got = set(GPU_STAGES) | set(CPU_STAGES)
    assert got == set(ALL_STAGES), f"GPU_STAGES | CPU_STAGES != ALL_STAGES: {sorted(got)} vs {sorted(ALL_STAGES)}"
    assert TWO_MODEL_STAGES <= GPU_STAGES, (
        f"a two-model stage that gets no GPU: {sorted(TWO_MODEL_STAGES - GPU_STAGES)}")
    assert C.JLENS_COMMIT in JLENS_TARBALL, f"JLENS_TARBALL is not config.JLENS_COMMIT ({C.JLENS_COMMIT})"


def _env(git=("", "", "")):
    env = os.environ.copy()
    commit, branch, dirty = git
    env.update(
        {
            "PYTHONPATH": "/app",
            "HF_HOME": HF_HOME,
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # the launch commit, carried in (the image has no .git); read by evals.downstream.common.runs.git_state()
            "GIT_COMMIT": commit,
            "GIT_BRANCH": branch,
            "GIT_DIRTY": dirty,
        }
    )
    os.makedirs("/tmp/matplotlib", exist_ok=True)
    return env


def _git_refs():
    """`(commit, branch, dirty)` resolved by git on the local machine at launch, "" where git cannot answer."""
    import subprocess

    def _one(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=str(REPO), text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = _one(["status", "--porcelain"])
    return (
        _one(["rev-parse", "HEAD"]) or "",
        _one(["rev-parse", "--abbrev-ref", "HEAD"]) or "",
        "" if status is None else ("1" if status else "0"),
    )


def _run(cmd, env=None):
    import subprocess

    print("[modal] running:", " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd, cwd="/app", env=env or _env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
    rc = process.wait()
    if rc != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} exited rc={rc}")


def _cmd(stage, run_id, smoke, force, extra, judge_budget_usd, seed, judge_profile="sonnet"):
    cmd = [
        "python",
        "-m",
        "evals.downstream.workspace_modulation",
        stage,
        "--output-dir",
        f"{OUTPUT_DIR}/{run_id}",
        "--seed",
        str(seed),
    ]
    if judge_budget_usd and judge_budget_usd > 0:
        cmd += ["--judge-budget-usd", str(judge_budget_usd)]
    cmd += ["--judge-profile", judge_profile]
    if smoke:
        cmd.append("--smoke")
    if force:
        cmd.append("--force")
    if extra:
        cmd += list(extra.split()) if isinstance(extra, str) else list(extra)
    return cmd


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[HF, *JUDGE_SECRETS], timeout=3 * 3600)
def gpu_stage(
    stage: str,
    run_id: str,
    smoke: bool,
    force: bool,
    extra: str,
    judge_budget_usd: float,
    seed: int,
    git: tuple,
    judge_profile: str = "sonnet",
) -> float:
    # reload(): a warm container's volume view may predate a sibling's writes; commit() persists on failure too
    vol.reload()
    t0 = time.time()
    try:
        _run(_cmd(stage, run_id, smoke, force, extra, judge_budget_usd, seed, judge_profile), env=_env(git))
    finally:
        vol.commit()
    # container wall time, model load included (summed as gpu_seconds_measured)
    return time.time() - t0


@app.function(image=image, volumes={"/data": vol}, secrets=[HF, *JUDGE_SECRETS], cpu=8, memory=32768, timeout=3 * 3600)
def cpu_stage(
    stage: str,
    run_id: str,
    smoke: bool,
    force: bool,
    extra: str,
    judge_budget_usd: float,
    seed: int,
    git: tuple,
    judge_profile: str = "sonnet",
):
    vol.reload()
    try:
        _run(_cmd(stage, run_id, smoke, force, extra, judge_budget_usd, seed, judge_profile), env=_env(git))
    finally:
        vol.commit()


@app.function(image=image, volumes={"/data": vol}, cpu=1, timeout=600)
def _update_provenance(run_id: str, extra: dict, filename: str = "modal.json"):
    """Write the launch's GPU type, cost and timing to provenance/<filename> (overwritten: each call carries
    the launch's whole state so far; read_provenance merges it with the stages' own files)."""
    import json

    vol.reload()
    path = f"{OUTPUT_DIR}/{run_id}/provenance/{filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(extra, h, ensure_ascii=False, indent=1)
    vol.commit()


def _extra_for(arms, shard=None, n_shards=1):
    parts = []
    if arms:
        parts.append(f"--arms {arms}")
    # the merge gets --n-shards too, to find the files the shards wrote
    if n_shards > 1:
        parts.append(f"--n-shards {n_shards}")
        if shard is not None:
            parts.append(f"--shard {shard}")
    return " ".join(parts)


def _shards_for(stage_name, rollout_shards, nla_shards, retrieval_shards, smoke=False):
    """How many containers a stage is dealt out over (its merge gets the same count); `retrieval` is one
    container under --smoke, whose corpus is tiny."""
    from evals.downstream.workspace_modulation.stages import SHARDED

    counts = {"rollouts": rollout_shards, "nla": nla_shards,
              "retrieval": 1 if smoke else retrieval_shards}
    for gpu_stage_name, merge in SHARDED.items():
        counts[merge] = counts.get(gpu_stage_name, 1)
    return max(1, int(counts.get(stage_name, 1)))


def _gpu_fn(name):
    """`gpu_stage` on the card the stage needs (`GPU_BOTH_MODELS` for `TWO_MODEL_STAGES`)."""
    return gpu_stage.with_options(gpu=GPU_BOTH_MODELS) if name in TWO_MODEL_STAGES else gpu_stage


def _gpu_workers(max_gpu_workers):
    """At least one container in flight."""
    return max(1, int(max_gpu_workers))


@app.local_entrypoint()
def run_all(
    run_id: str,
    smoke: bool = False,
    max_gpu_workers: int = GPU_WORKERS,
    force: bool = False,
    judge_budget_usd: float = UNSET_BUDGET,
    seed: int = FROM_CONFIG,
    arms: str = UNSET_ARMS,
    rollout_shards: int = UNSET_SHARDS,
    nla_shards: int = UNSET_SHARDS,
    retrieval_shards: int = UNSET_SHARDS,
    judge_profile: str = "sonnet",
):
    from evals.downstream.common import judges

    judges.activate(judge_profile)  # before the package config is imported
    from evals.downstream.workspace_modulation.stages import levels

    _check_stage_lists()
    (_cap, default_arms, default_rollout_shards, default_nla_shards, default_retrieval_shards,
     default_seed) = _defaults()
    arm_list = arms or default_arms
    rollout_shards = rollout_shards or default_rollout_shards
    nla_shards = nla_shards or default_nla_shards
    retrieval_shards = retrieval_shards or default_retrieval_shards
    max_gpu_workers = _gpu_workers(max_gpu_workers)
    seed = default_seed if seed == FROM_CONFIG else seed
    git = _git_refs()  # resolved once, on this machine, before anything is spawned
    git_commit, git_branch, git_dirty = git
    print(
        f"[run_all] launch commit {git_commit or 'unavailable'} on {git_branch or 'unavailable'}"
        f" (tree {'dirty' if git_dirty == '1' else 'clean' if git_dirty == '0' else 'unavailable'})",
        flush=True,
    )

    t_start = time.time()
    gpu_seconds_measured = 0.0

    def _n_shards(name):
        return _shards_for(name, rollout_shards, nla_shards, retrieval_shards, smoke)

    def _extra(name, shard=None, arm=None):
        return _extra_for(arm or arms, shard, _n_shards(name))

    def _run_level(level):
        nonlocal gpu_seconds_measured
        # one work item per (stage, arm, shard): `rollouts` per arm and shard, `nla` and `retrieval` per shard
        gpu_work = []
        for s in level:
            if s not in GPU_STAGES:
                continue
            n = _n_shards(s)
            for arm in ([a for a in arm_list.split(",") if a] if s == "rollouts" else [None]):
                for k in range(n):
                    gpu_work.append((s, arm, k if n > 1 else None))
        cpu_names = [s for s in level if s in CPU_STAGES]
        print(f"[run_all] ===== level: {level}{' (smoke)' if smoke else ''} =====", flush=True)
        pending = list(gpu_work)
        while pending:
            batch, pending = pending[:max_gpu_workers], pending[max_gpu_workers:]
            handles = [
                (
                    name,
                    k,
                    _gpu_fn(name).spawn(
                        name, run_id, smoke, force, _extra(name, k, arm), judge_budget_usd, seed, git,
                        judge_profile,
                    ),
                )
                for name, arm, k in batch
            ]
            for name in cpu_names:
                cpu_stage.remote(name, run_id, smoke, force, _extra(name), judge_budget_usd, seed, git, judge_profile)
            cpu_names = []  # run the level's CPU stages once, alongside the first GPU batch
            for _name, _k, handle in handles:
                gpu_seconds_measured += handle.get()
        for name in cpu_names:  # level had no GPU stage at all
            cpu_stage.remote(name, run_id, smoke, force, _extra(name), judge_budget_usd, seed, git, judge_profile)
        elapsed_s = time.time() - t_start
        gpu_cost_usd_estimated = gpu_seconds_measured / 3600.0 * GPU_COST_PER_HOUR
        _update_provenance.remote(
            run_id,
            {
                "gpu_type": GPU,
                "elapsed_s": elapsed_s,
                "gpu_seconds_measured": gpu_seconds_measured,
                "gpu_cost_usd_estimated": gpu_cost_usd_estimated,
                "gpu_cost_per_hour": GPU_COST_PER_HOUR,
                "gpu_timing_source": "measured",
                "billing_usd": "unavailable",
            },
        )
        print(
            f"[run_all] level done, elapsed {elapsed_s:.0f}s, gpu-seconds measured so far {gpu_seconds_measured:.0f}",
            flush=True,
        )

    for level in levels():
        _run_level(level)

    print("WORKSPACE_MODULATION_DONE", flush=True)


@app.local_entrypoint()
def pull(run_id: str, dest: str = "evals/downstream/out/workspace_modulation"):
    """Copy a finished run off the volume into <dest>/<run_id>/, file by file (`modal volume get` of a
    directory fails on some clients)."""
    from modal.volume import FileEntryType

    root = f"{OUTPUT_DIR.removeprefix('/data/')}/{run_id}"
    out_root = Path(dest) / run_id
    n_files = n_bytes = 0
    for entry in vol.listdir(root, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        target = out_root / entry.path[len(root) + 1 :]
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as h:
            for chunk in vol.read_file(entry.path):
                h.write(chunk)
                n_bytes += len(chunk)
        n_files += 1
    print(f"[pull] {n_files} files, {n_bytes} bytes -> {out_root}", flush=True)


@app.local_entrypoint()
def stage(
    name: str,
    run_id: str,
    smoke: bool = False,
    force: bool = False,
    extra: str = "",
    judge_budget_usd: float = UNSET_BUDGET,
    seed: int = FROM_CONFIG,
    judge_profile: str = "sonnet",
):
    from evals.downstream.common import judges

    judges.activate(judge_profile)  # before the package config is imported
    _check_stage_lists()
    if name not in GPU_STAGES | CPU_STAGES:
        raise ValueError(f"unknown stage {name!r}; choose from {sorted(GPU_STAGES | CPU_STAGES)}")
    fn = _gpu_fn(name) if name in GPU_STAGES else cpu_stage
    seed = _defaults()[-1] if seed == FROM_CONFIG else seed  # the seed is the tuple's last field
    seconds = fn.remote(name, run_id, smoke, force, extra, judge_budget_usd, seed, _git_refs(), judge_profile)
    # a GPU stage launched alone records its own seconds in provenance/modal_<stage>.json
    if name in GPU_STAGES and isinstance(seconds, (int, float)):
        _update_provenance.remote(
            run_id,
            {
                "stage": name,
                "gpu_type": GPU,
                "gpu_seconds_measured": seconds,
                "gpu_cost_usd_estimated": seconds / 3600.0 * GPU_COST_PER_HOUR,
                "gpu_cost_per_hour": GPU_COST_PER_HOUR,
                "gpu_timing_source": "measured",
                "billing_usd": "unavailable",
            },
            f"modal_{name}.json",
        )
        print(
            f"[stage] {name}: {seconds:.0f} container seconds, recorded in provenance/modal_{name}.json",
            flush=True,
        )
