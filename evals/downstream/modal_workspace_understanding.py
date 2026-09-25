"""Modal launcher for evals/downstream/workspace_understanding: every stage runs the package CLI
(`python -m evals.downstream.workspace_understanding <stage> --output-dir <EVAL_OUTPUT_DIR>/<run-id>`) as a subprocess in
a GPU or CPU container, level by level over the package's dependency graph.

    modal run --detach evals/downstream/modal_workspace_understanding.py::run_all --run-id smoke --smoke
    modal run --detach evals/downstream/modal_workspace_understanding.py::run_all --run-id <run-id>
    modal run --detach evals/downstream/modal_workspace_understanding.py::stage --name rollouts --run-id <run-id> --force
    modal run --detach evals/downstream/modal_workspace_understanding.py::stage --name retrieval --run-id <run-id> --extra "--shard 2/8"
    modal run evals/downstream/modal_workspace_understanding.py::pull --run-id <run-id>

`--judge-profile` is `sonnet` (the default, the paper's) or `sol` (needs EVAL_OPENROUTER_SECRET set). App,
volume, secret and GPU names, the GPU container cap and the shard counts come from EVAL_* environment
variables (evals/downstream/modal.env.example, evals/downstream/README.md). `nla`/`nla_control` fan out by item, `retrieval` by
corpus block followed by one merge call on CPU.
"""

import os
import time
from pathlib import Path

import modal

# Nothing from evals.downstream.* is imported at module level: Modal re-imports this file inside the container, where
# eval is not on the path. The local entrypoints import it at call time.

REPO = Path(__file__).resolve().parent.parent.parent
APP_NAME = os.environ.get("EVAL_APP", "maemm-workspace-understanding")
app = modal.App(APP_NAME)
# The card for a stage holding one 27B (about 54 GB) and for one holding the base and the inverter at once;
# the second has its own default, so a smaller EVAL_GPU never becomes the two-model card.
GPU = os.environ.get("EVAL_GPU", "B200:1")
GPU_BOTH_MODELS = os.environ.get("EVAL_GPU_BOTH_MODELS", "B200:1")
# GPU containers in flight at once.
GPU_WORKERS = int(os.environ.get("EVAL_GPU_WORKERS", "8"))
# The lens the image installs, at config.JLENS_COMMIT (checked by `_check_stage_lists`).
JLENS_TARBALL = "https://github.com/anthropics/jacobian-lens/archive/581d398613e5602a5af361e1c34d3a92ea82ba8e.tar.gz"
GPU_COST_PER_HOUR = float(os.environ.get("EVAL_GPU_COST_PER_HOUR", "6.2496"))  # list rate, for the estimate only
# Budget cap and seed default to the package's config (resolved in the local entrypoint).
FROM_CONFIG = -1

# Every EVAL_* variable this file reads at import, and the model pins' overrides (evals/downstream/common/pins.py), copied
# into the image so the container's re-import declares the same paths, secrets and checkpoints.
LAUNCH_VARS = (
    "EVAL_APP", "EVAL_GPU", "EVAL_GPU_BOTH_MODELS", "EVAL_GPU_WORKERS", "EVAL_GPU_COST_PER_HOUR",
    "EVAL_VOLUME", "EVAL_HF_SECRET", "EVAL_ANTHROPIC_SECRET", "EVAL_OPENROUTER_SECRET", "EVAL_OUTPUT_DIR",
    "EVAL_OUTPUT_DIR_WORKSPACE_UNDERSTANDING", "EVAL_HF_HOME", "EVAL_NLA_SHARDS", "EVAL_RETRIEVAL_SHARDS",
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
        "pyarrow",
    )
    # the base model's linear-attention kernels, as in every package's image
    .pip_install("flash-linear-attention==0.5.2")
    .pip_install(JLENS_TARBALL)
    .env(LAUNCH_ENV)
    .add_local_dir(REPO / "maemm", "/app/maemm", ignore=["__pycache__", "out", "analysis", "test_*"])
    .add_local_dir(REPO / "evals" / "downstream", "/app/evals/downstream", ignore=["__pycache__", "out", "analysis", "test_*"])
)
VOLUME_NAME = os.environ.get("EVAL_VOLUME", "maemm-data")
HF_SECRET_NAME = os.environ.get("EVAL_HF_SECRET", "maemm-hf")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
HF = modal.Secret.from_name(HF_SECRET_NAME)
# The default profile `sonnet` is asked through the Anthropic API, so its secret is always mounted; the
# OpenRouter secret `sol` needs is mounted only when EVAL_OPENROUTER_SECRET names one.
ANTHROPIC_SECRET_NAME = os.environ.get("EVAL_ANTHROPIC_SECRET", "maemm-anthropic")
OPENROUTER_SECRET_NAME = os.environ.get("EVAL_OPENROUTER_SECRET")
JUDGE_SECRETS = [modal.Secret.from_name(ANTHROPIC_SECRET_NAME)] + (
    [modal.Secret.from_name(OPENROUTER_SECRET_NAME)] if OPENROUTER_SECRET_NAME else [])

# EVAL_OUTPUT_DIR is the root every activation package writes under, as <root>/workspace_understanding.
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR_WORKSPACE_UNDERSTANDING") or os.path.join(
    os.environ.get("EVAL_OUTPUT_DIR", "/data"), "workspace_understanding")
HF_HOME = os.environ.get("EVAL_HF_HOME", "/data/hf_cache")
# A GPU stage loads a 27B model or the lens file.
GPU_STAGES = {
    "capture",
    "lens",
    "rollouts",
    "retrieval",  # sharded by corpus window across RETRIEVAL_SHARDS containers, then a merge call
    "patchscope",
    "nla",  # sharded by item across NLA_SHARDS containers
    "nla_control",  # sharded by item like nla
    "untrained_base",
    "maemm_control",
    "reread",
}
# Stages that hold the inverter and the base at once (`GPU_BOTH_MODELS`); = stages.TWO_MODEL_STAGES.
TWO_MODEL_STAGES = {"rollouts", "maemm_control"}
CPU_STAGES = {"prepare", "corpus", "summarise", "judge", "report"}
NLA_SHARDS = int(os.environ.get("EVAL_NLA_SHARDS", "2"))
# Corpus blocks for a full run's search: eight keeps each container inside `gpu_stage`'s timeout.
RETRIEVAL_SHARDS = int(os.environ.get("EVAL_RETRIEVAL_SHARDS", "8"))


def _check_stage_lists():
    """Assert the GPU/CPU partition is exactly the package's stage list and the image's lens is the pinned
    commit. Local entrypoints only."""
    from evals.downstream.workspace_understanding import ALL_STAGES
    from evals.downstream.workspace_understanding import config as C
    from evals.downstream.workspace_understanding.stages import TWO_MODEL_STAGES as PKG_TWO_MODEL

    got = set(GPU_STAGES) | set(CPU_STAGES)
    assert got == set(ALL_STAGES), f"GPU_STAGES | CPU_STAGES != ALL_STAGES: {sorted(got)} vs {sorted(ALL_STAGES)}"
    assert not GPU_STAGES & CPU_STAGES, f"a stage is in both partitions: {sorted(GPU_STAGES & CPU_STAGES)}"
    assert TWO_MODEL_STAGES == set(PKG_TWO_MODEL), (
        f"TWO_MODEL_STAGES != stages.TWO_MODEL_STAGES: {sorted(TWO_MODEL_STAGES)} vs {sorted(PKG_TWO_MODEL)}")
    assert TWO_MODEL_STAGES <= GPU_STAGES, (
        f"a two-model stage that gets no GPU: {sorted(TWO_MODEL_STAGES - GPU_STAGES)}")
    assert C.JLENS_COMMIT in JLENS_TARBALL, f"JLENS_TARBALL is not config.JLENS_COMMIT ({C.JLENS_COMMIT})"


def _resolve(judge_budget_usd, seed):
    """(budget, seed) with FROM_CONFIG replaced by the package's own constants."""
    from evals.downstream.workspace_understanding import config as C

    return (
        C.JUDGE_BUDGET_USD if judge_budget_usd == FROM_CONFIG else judge_budget_usd,
        C.GEN_SEED if seed == FROM_CONFIG else seed,
    )


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
            # the launching machine's git state, read by evals.downstream.common.runs.git_state() (no .git in the image)
            "GIT_COMMIT": commit,
            "GIT_BRANCH": branch,
            "GIT_DIRTY": dirty,
        }
    )
    os.makedirs("/tmp/matplotlib", exist_ok=True)
    return env


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
        "evals.downstream.workspace_understanding",
        stage,
        "--output-dir",
        f"{OUTPUT_DIR}/{run_id}",
        "--judge-budget-usd",
        str(judge_budget_usd),
        "--seed",
        str(seed),
    ]
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
    # reload: a warm container's volume view predates its siblings' writes; commit even on failure
    vol.reload()
    t0 = time.time()
    try:
        _run(_cmd(stage, run_id, smoke, force, extra, judge_budget_usd, seed, judge_profile), env=_env(git))
    finally:
        vol.commit()
    # container wall time, model load included
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
def _update_provenance(run_id: str, extra: dict):
    """Write the launch's GPU type, cost and timing to provenance/modal.json (its own file; overwritten)."""
    import json

    vol.reload()
    path = f"{OUTPUT_DIR}/{run_id}/provenance/modal.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(extra, h, ensure_ascii=False, indent=1)
    vol.commit()


def _git_refs():
    """(commit, branch, dirty) of the local checkout at launch, "" where git cannot answer; carried into
    the containers as GIT_COMMIT / GIT_BRANCH / GIT_DIRTY."""
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


def _job_fn(name, extra=""):
    """`cpu_stage` for a CPU stage or the retrieval merge call; `gpu_stage` on `GPU_BOTH_MODELS` for a
    two-model stage and on `GPU` otherwise."""
    if name in CPU_STAGES or "--merge" in str(extra).split():
        return cpu_stage
    return gpu_stage.with_options(gpu=GPU_BOTH_MODELS) if name in TWO_MODEL_STAGES else gpu_stage


def retrieval_shards(smoke):
    """How many blocks the corpus forward is cut into (one under --smoke)."""
    return 1 if smoke else RETRIEVAL_SHARDS


def _fanout(stage_name, smoke):
    """The waves one GPU stage is spawned as, each a list of `--extra` strings run at once: one job, one
    wave of item shards, or for `retrieval` the corpus blocks followed by a `--merge` call."""
    if stage_name in ("nla", "nla_control"):
        return [[f"--shard {k}/{NLA_SHARDS}" for k in range(NLA_SHARDS)]]
    if stage_name == "retrieval":
        n = retrieval_shards(smoke)
        shards = [f"--shard {k}/{n}" for k in range(n)]
        if n == 1:
            return [shards]
        return [shards, [f"--shard 0/{n} --merge"]]
    return [[""]]


@app.local_entrypoint()
def run_all(
    run_id: str,
    smoke: bool = False,
    max_gpu_workers: int = GPU_WORKERS,
    force: bool = False,
    judge_budget_usd: float = FROM_CONFIG,
    seed: int = FROM_CONFIG,
    judge_profile: str = "sonnet",
):
    from evals.downstream.common import judges

    judges.activate(judge_profile)  # before the package config binds its judges
    from evals.downstream.workspace_understanding.stages import levels

    _check_stage_lists()
    judge_budget_usd, seed = _resolve(judge_budget_usd, seed)
    max_gpu_workers = max(1, max_gpu_workers)
    git = _git_refs()
    git_commit, git_branch, git_dirty = git
    print(
        f"[run_all] launch commit {git_commit or 'unavailable'} on {git_branch or 'unavailable'}"
        f" (tree {'dirty' if git_dirty == '1' else 'clean' if git_dirty == '0' else 'unavailable'})",
        flush=True,
    )

    t_start = time.time()
    gpu_seconds_measured = 0.0
    for level in levels():
        gpu_names = [s for s in level if s in GPU_STAGES]
        cpu_names = [s for s in level if s in CPU_STAGES]
        print(f"[run_all] ===== level: {level}{' (smoke)' if smoke else ''} =====", flush=True)
        # wave w of the level is wave w of each of its stages; at most max_gpu_workers in flight
        plans = [_fanout(name, smoke) for name in gpu_names]
        for w in range(max((len(p) for p in plans), default=0)):
            pending = [(name, extra)
                       for name, plan in zip(gpu_names, plans) if w < len(plan)
                       for extra in plan[w]]
            while pending:
                batch, pending = pending[:max_gpu_workers], pending[max_gpu_workers:]
                handles = [
                    _job_fn(name, extra).spawn(name, run_id, smoke, force, extra, judge_budget_usd, seed, git,
                                               judge_profile)
                    for name, extra in batch
                ]
                for name in cpu_names:
                    cpu_stage.remote(name, run_id, smoke, force, "", judge_budget_usd, seed, git, judge_profile)
                cpu_names = []  # run the level's CPU stages once, alongside the first GPU batch
                for (name, extra), handle in zip(batch, handles):
                    # container-measured seconds; a merge call on CPU returns None
                    seconds = handle.get()
                    gpu_seconds_measured += seconds if seconds is not None else 0.0
                    print(f"[run_all] {name} {extra!r} done", flush=True)
        for name in cpu_names:  # level had no GPU stage at all
            cpu_stage.remote(name, run_id, smoke, force, "", judge_budget_usd, seed, git, judge_profile)
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
                # `launch_` names: the run's own git_commit is its first stage's
                "launch_git_commit": git_commit,
                "launch_git_branch": git_branch,
                "launch_git_dirty": None if git_dirty == "" else git_dirty == "1",
            },
        )
        print(
            f"[run_all] level done, elapsed {elapsed_s:.0f}s, gpu-seconds measured so far {gpu_seconds_measured:.0f}",
            flush=True,
        )
    print("WORKSPACE_UNDERSTANDING_DONE", flush=True)


@app.local_entrypoint()
def stage(
    name: str,
    run_id: str,
    smoke: bool = False,
    force: bool = False,
    extra: str = "",
    judge_budget_usd: float = FROM_CONFIG,
    seed: int = FROM_CONFIG,
    judge_profile: str = "sonnet",
):
    from evals.downstream.common import judges

    judges.activate(judge_profile)  # before the package config binds its judges
    _check_stage_lists()
    judge_budget_usd, seed = _resolve(judge_budget_usd, seed)
    if name not in GPU_STAGES | CPU_STAGES:
        raise ValueError(f"unknown stage {name!r}; choose from {sorted(GPU_STAGES | CPU_STAGES)}")
    _job_fn(name, extra).remote(name, run_id, smoke, force, extra, judge_budget_usd, seed, _git_refs(),
                                judge_profile)


@app.local_entrypoint()
def pull(run_id: str, dest: str = "evals/downstream/out/workspace_understanding"):
    """Copy one run directory off the volume into `dest`/<run-id>, skipping files already present at the
    same size."""
    from modal.volume import FileEntryType

    _check_stage_lists()
    root = f"{OUTPUT_DIR.lstrip('/').split('/', 1)[-1]}/{run_id}"
    out_root = Path(dest) / run_id
    n = skipped = 0
    for entry in vol.iterdir(root, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        rel = entry.path[len(root) + 1 :]
        target = out_root / rel
        if target.exists() and target.stat().st_size == getattr(entry, "size", -1):
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as h:
            for chunk in vol.read_file(entry.path):
                h.write(chunk)
        n += 1
    print(f"[pull] {n} files into {out_root} ({skipped} already present)", flush=True)
