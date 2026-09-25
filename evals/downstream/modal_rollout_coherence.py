"""Modal launcher for evals/downstream/rollout_coherence: runs each stage's package CLI in a GPU or CPU container, the
fanned-out stages as waves, over `stages.levels()` in order.

    modal run --detach evals/downstream/modal_rollout_coherence.py::run_all --run-id <run_id> [--smoke]
    modal run --detach evals/downstream/modal_rollout_coherence.py::stage --name frontier_context_pairs --run-id <run_id> --force
    modal run --detach evals/downstream/modal_rollout_coherence.py::stage --name frontier_context --run-id <run_id> \
        --extra "--context-methods retrieval --shard 3/4"
    modal run evals/downstream/modal_rollout_coherence.py::pull --run-id <run_id>

Launch detached (a disconnecting client cancels the calls in flight); relaunching resumes. Names come from
the environment (evals/downstream/README.md, "Launcher environment").
"""

import os
import time
from pathlib import Path

import modal

# No evals.downstream.* import at module level: the container imports this file before PYTHONPATH=/app applies.

REPO = Path(__file__).resolve().parent.parent.parent
APP_NAME = os.environ.get("EVAL_APP", "maem-rollout-coherence")
app = modal.App(APP_NAME)
GPU = os.environ.get("EVAL_GPU", "B200:1")
GPU_BOTH_MODELS = os.environ.get("EVAL_GPU_BOTH_MODELS", GPU)
GPU_WORKERS = int(os.environ.get("EVAL_GPU_WORKERS", "8"))
GPU_COST_PER_HOUR = float(os.environ.get("EVAL_GPU_COST_PER_HOUR", "6.2496"))  # for the cost estimate only
RETRIEVAL_SHARDS = int(os.environ.get("EVAL_RETRIEVAL_SHARDS", "4"))
DEFAULT_JUDGE_PROFILE = "sonnet"
#: A cap flag left at this value means the package's own constant (a typed Modal flag cannot be None).
FROM_CONFIG = -1
# `frontier_context` methods after `targets`, one job each; `retrieval` is sharded (checked against config).
CONTEXT_JOBS = ("maem", "continuation", "nla")
GPU_TIMEOUT = 3 * 3600
CPU_TIMEOUT = 10 * 3600   # the largest judge pass is a few hours at worst; twice that as headroom

VOLUME_NAME = os.environ.get("EVAL_VOLUME", "maem-data")
HF_SECRET_NAME = os.environ.get("EVAL_HF_SECRET", "maem-hf")
# The default profile's judge key always; the `sol` judge's only when its secret is named.
ANTHROPIC_SECRET_NAME = os.environ.get("EVAL_ANTHROPIC_SECRET", "maem-anthropic")
OPENROUTER_SECRET_NAME = os.environ.get("EVAL_OPENROUTER_SECRET", "")
# EVAL_OUTPUT_DIR is the root every activation package writes under, as <root>/rollout_coherence.
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR_ROLLOUT_COHERENCE") or os.path.join(
    os.environ.get("EVAL_OUTPUT_DIR", "/data"), "rollout_coherence")
HF_HOME = os.environ.get("EVAL_HF_HOME", "/data/hf_cache")

# Every EVAL_* variable read above, and the pins' overrides, copied into the image: the container re-imports
# this file and must declare the same app, volume, secrets and checkpoints.
LAUNCH_ENV_NAMES = {
    "EVAL_APP", "EVAL_GPU", "EVAL_GPU_BOTH_MODELS", "EVAL_GPU_WORKERS", "EVAL_GPU_COST_PER_HOUR",
    "EVAL_RETRIEVAL_SHARDS", "EVAL_VOLUME", "EVAL_HF_SECRET", "EVAL_ANTHROPIC_SECRET", "EVAL_OPENROUTER_SECRET",
    "EVAL_OUTPUT_DIR", "EVAL_OUTPUT_DIR_ROLLOUT_COHERENCE", "EVAL_HF_HOME",
    *(f"EVAL_{m}_{f}" for m in ("BASE", "INVERTER", "NLA", "LENS") for f in ("REPO", "REVISION")),
}
LAUNCH_ENV = {k: v for k, v in os.environ.items() if k in LAUNCH_ENV_NAMES}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements(str(REPO / "evals/downstream/common/requirements.txt"))
    .pip_install_from_requirements(str(REPO / "evals/downstream/rollout_coherence/requirements.txt"))
    .env(LAUNCH_ENV)
    .add_local_dir(REPO / "maem", "/app/maem", ignore=["__pycache__", "out", "analysis", "test_*"])
    .add_local_dir(REPO / "evals" / "downstream", "/app/evals/downstream", ignore=["__pycache__", "out", "analysis", "test_*"])
)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
HF = modal.Secret.from_name(HF_SECRET_NAME)
JUDGE_SECRETS = [modal.Secret.from_name(n) for n in (ANTHROPIC_SECRET_NAME, OPENROUTER_SECRET_NAME) if n]

# The stages that load a model; `_check_stage_lists` holds this partition to stages.py.
GPU_STAGES = {"capture", "frontier_context", "frontier_context_fluency"}
TWO_MODEL_STAGES = {"frontier_context"}
#: selector words that load the inverter (`maem`) or the verbalizer (`nla`)
SECOND_MODEL_SELECTORS = ("maem", "nla")
CPU_STAGES = {"prepare", "frontier_corpus", "frontier_context_pairs", "frontier_context_judge", "report"}


def _resolve(judge_budget_usd):
    """The cap an entrypoint passes on: the given one when positive, else the package's."""
    from evals.downstream.rollout_coherence import config as C

    return float(judge_budget_usd) if judge_budget_usd and judge_budget_usd > 0 else float(C.JUDGE_BUDGET_USD)


def _check_stage_lists():
    """Hold the GPU/CPU partition and the fan-out plans to the package's stages and method lists."""
    from evals.downstream.rollout_coherence import config as C
    from evals.downstream.rollout_coherence.stages import GPU_STAGES as PKG_GPU
    from evals.downstream.rollout_coherence.stages import STAGES

    got = set(GPU_STAGES) | set(CPU_STAGES)
    assert got == set(STAGES), f"GPU_STAGES | CPU_STAGES != STAGES: {sorted(got)} vs {sorted(STAGES)}"
    assert not GPU_STAGES & CPU_STAGES, f"a stage is in both partitions: {sorted(GPU_STAGES & CPU_STAGES)}"
    assert GPU_STAGES == set(PKG_GPU), (
        f"GPU_STAGES != stages.GPU_STAGES: {sorted(GPU_STAGES)} vs {sorted(PKG_GPU)}")
    assert TWO_MODEL_STAGES <= GPU_STAGES, (
        f"a two-model stage that gets no GPU: {sorted(TWO_MODEL_STAGES - GPU_STAGES)}")
    planned = {"targets", "retrieval"} | {m for job in CONTEXT_JOBS for m in job.split(",")}
    assert planned == set(C.CONTEXT_METHODS), (
        f"the frontier_context plan covers {sorted(planned)}, config.CONTEXT_METHODS is "
        f"{sorted(C.CONTEXT_METHODS)}")


# ---------------------------------------------------------------- the waves a fanned-out stage is spawned as
# A wave is a list of `(extra, merge_only)` jobs run at once; the next wave starts after all of them
# commit. A merge-only job is spawned with force=False even under `--force`, so it only merges.

def retrieval_shards(smoke):
    """Containers the corpus forward is cut into: `RETRIEVAL_SHARDS`, or one under `--smoke`."""
    return 1 if smoke else RETRIEVAL_SHARDS


def _retrieval_waves(flag, n=RETRIEVAL_SHARDS):
    """`(shard jobs, merge wave)` for one sharded corpus forward (no merge wave at one shard)."""
    shards = [(f"{flag} retrieval --shard {k}/{n}", False) for k in range(n)]
    return shards, ([[(f"{flag} retrieval --shard 0/{n} --merge", True)]] if n > 1 else [])


def plan_frontier_context(smoke=False):
    """`frontier_context`'s waves: `targets`, then the other methods beside the retrieval shards, then the merge."""
    shards, merge = _retrieval_waves("--context-methods", retrieval_shards(smoke))
    return ([[("--context-methods targets", False)]]
            + [shards + [(f"--context-methods {job}", False) for job in CONTEXT_JOBS]]
            + merge)


#: the GPU stages that are more than one container's work; every other one is a single unsharded job
FAN_OUT = {"frontier_context": plan_frontier_context}


def gpu_waves(stage, smoke=False):
    """The waves `stage` is spawned as: its own plan, or a single wave of one plain job."""
    plan = FAN_OUT.get(stage)
    return plan(smoke) if plan else [[("", False)]]


def _git_refs():
    """`(commit, branch, dirty)` of the launching checkout ("" where git cannot answer); the image has no `.git`."""
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


def _cmd(stage, run_id, smoke, force, extra, judge_budget_usd, judge_profile=DEFAULT_JUDGE_PROFILE):
    """The package CLI for one stage, on the run directory `OUTPUT_DIR/<run_id>`."""
    cmd = ["python", "-m", "evals.downstream.rollout_coherence", stage, "--output-dir", f"{OUTPUT_DIR}/{run_id}",
           "--judge-budget-usd", str(judge_budget_usd), "--judge-profile", judge_profile]
    if smoke:
        cmd.append("--smoke")
    if force:
        cmd.append("--force")
    if extra:
        cmd += list(extra.split()) if isinstance(extra, str) else list(extra)
    return cmd


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, secrets=[HF, *JUDGE_SECRETS], timeout=GPU_TIMEOUT)
def gpu_stage(
    stage: str,
    run_id: str,
    smoke: bool,
    force: bool,
    extra: str,
    judge_budget_usd: float,
    git: tuple,
    judge_profile: str = DEFAULT_JUDGE_PROFILE,
) -> float:
    # A warm container's volume view can predate a sibling's writes: reload first, and commit even on
    # failure so partial artifacts and records persist.
    vol.reload()
    t0 = time.time()
    try:
        _run(_cmd(stage, run_id, smoke, force, extra, judge_budget_usd, judge_profile), env=_env(git))
    finally:
        vol.commit()
    return time.time() - t0   # container wall time, model load included


@app.function(image=image, volumes={"/data": vol}, secrets=[HF, *JUDGE_SECRETS], cpu=8, memory=32768,
              timeout=CPU_TIMEOUT)
def cpu_stage(
    stage: str,
    run_id: str,
    smoke: bool,
    force: bool,
    extra: str,
    judge_budget_usd: float,
    git: tuple,
    judge_profile: str = DEFAULT_JUDGE_PROFILE,
):
    vol.reload()
    try:
        _run(_cmd(stage, run_id, smoke, force, extra, judge_budget_usd, judge_profile), env=_env(git))
    finally:
        vol.commit()


@app.function(image=image, volumes={"/data": vol}, cpu=1, timeout=600)
def _update_provenance(run_id: str, extra: dict):
    """The launcher's own fields (GPU type, cost estimate, timing), written whole to `provenance/modal.json`."""
    import json

    vol.reload()
    path = f"{OUTPUT_DIR}/{run_id}/provenance/modal.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(extra, h, ensure_ascii=False, indent=1)
    vol.commit()


@app.local_entrypoint()
def run_all(
    run_id: str,
    smoke: bool = False,
    max_gpu_workers: int = GPU_WORKERS,
    force: bool = False,
    judge_budget_usd: float = FROM_CONFIG,
    judge_profile: str = DEFAULT_JUDGE_PROFILE,
):
    """Every stage, in dependency-level order, under the package's own cap unless one is given."""
    from evals.downstream.common import judges

    judges.activate(judge_profile)   # before the package config is first imported on this machine
    from evals.downstream.rollout_coherence.stages import levels

    _check_stage_lists()
    judge_budget_usd = _resolve(judge_budget_usd)
    max_gpu_workers = max(1, max_gpu_workers)
    git = _git_refs()   # resolved once, on this machine, before anything is spawned
    git_commit, git_branch, git_dirty = git
    print(
        f"[run_all] launch commit {git_commit or 'unavailable'} on {git_branch or 'unavailable'}"
        f" (tree {'dirty' if git_dirty == '1' else 'clean' if git_dirty == '0' else 'unavailable'})",
        flush=True,
    )
    caps = (judge_budget_usd, git, judge_profile)

    t_start = time.time()
    gpu_seconds_measured = 0.0
    for level in levels():
        gpu_names = [s for s in level if s in GPU_STAGES]
        cpu_names = [s for s in level if s in CPU_STAGES]
        print(f"[run_all] ===== level: {level}{' (smoke)' if smoke else ''} =====", flush=True)
        # wave w of the level is wave w of each of its (independent) GPU stages
        plans = [gpu_waves(name, smoke) for name in gpu_names]
        for w in range(max((len(p) for p in plans), default=0)):
            pending = [(name, extra, merge_only)
                       for name, plan in zip(gpu_names, plans) if w < len(plan)
                       for extra, merge_only in plan[w]]
            while pending:
                batch, pending = pending[:max_gpu_workers], pending[max_gpu_workers:]
                handles = [
                    _gpu_fn(name, extra).spawn(name, run_id, smoke, force and not merge_only, extra, *caps)
                    for name, extra, merge_only in batch
                ]
                gpu_seconds_measured += sum(h.get() for h in handles)
        # one CPU stage at a time, so no two judge passes spend against the one ledger at once
        for name in cpu_names:
            cpu_stage.remote(name, run_id, smoke, force, "", *caps)
        elapsed_s = time.time() - t_start
        gpu_cost_usd_estimated = gpu_seconds_measured / 3600.0 * GPU_COST_PER_HOUR
        _update_provenance.remote(
            run_id,
            {
                "gpu_type": GPU,
                "elapsed_s": elapsed_s,
                "gpu_seconds_measured": gpu_seconds_measured,
                "gpu_cost_usd_estimated": gpu_cost_usd_estimated,
                "gpu_timing_source": "measured",
                "billing_usd": "unavailable",
            },
        )
        print(
            f"[run_all] level done, elapsed {elapsed_s:.0f}s, gpu-seconds measured so far {gpu_seconds_measured:.0f}",
            flush=True,
        )
    print("ROLLOUT_COHERENCE_DONE", flush=True)


@app.local_entrypoint()
def pull(run_id: str, dest: str = "evals/downstream/out/rollout_coherence"):
    """Copy a finished run off the volume into <dest>/<run_id>/, file by file."""
    from modal.volume import FileEntryType

    # volume-relative: OUTPUT_DIR is where /data is mounted inside a container, and /data IS the volume
    root = f"{OUTPUT_DIR.removeprefix('/data/')}/{run_id}"
    out_root = Path(dest) / run_id
    n_files = n_bytes = 0
    for entry in vol.listdir(root, recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        target = out_root / entry.path[len(root) + 1:]
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as h:
            for chunk in vol.read_file(entry.path):
                h.write(chunk)
                n_bytes += len(chunk)
        n_files += 1
    print(f"[pull] {n_files} files, {n_bytes} bytes -> {out_root}", flush=True)


def _holds_both_models(name, extra):
    """True when this job's selectors name `maem` or `nla`, or it has none (the stage's whole list)."""
    if name not in TWO_MODEL_STAGES:
        return False
    if not extra:
        return True
    return any(sel in word.split(",") for word in str(extra).split() for sel in SECOND_MODEL_SELECTORS)


def _gpu_fn(name, extra=""):
    """`gpu_stage` on the card this job needs (`_holds_both_models`)."""
    return gpu_stage.with_options(gpu=GPU_BOTH_MODELS) if _holds_both_models(name, extra) else gpu_stage


def _stage_fn(name, extra=""):
    """The container a named stage runs in (GPU for a model-loading stage, else CPU); `all` is `run_all`."""
    known = GPU_STAGES | CPU_STAGES
    if name not in known:
        raise ValueError(f"unknown stage {name!r}; choose from {sorted(known)}")
    return _gpu_fn(name, extra) if name in GPU_STAGES else cpu_stage


@app.local_entrypoint()
def stage(
    name: str,
    run_id: str,
    smoke: bool = False,
    force: bool = False,
    extra: str = "",
    judge_budget_usd: float = FROM_CONFIG,
    judge_profile: str = DEFAULT_JUDGE_PROFILE,
):
    """One stage, in the container it needs."""
    from evals.downstream.common import judges

    judges.activate(judge_profile)
    _check_stage_lists()
    fn = _stage_fn(name, extra)
    fn.remote(name, run_id, smoke, force, extra, _resolve(judge_budget_usd), _git_refs(), judge_profile)
