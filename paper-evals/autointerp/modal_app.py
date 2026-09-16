"""The `autointerp` stages' own Modal app -- P1 (GPU), P2 (CPU) and the LLM run (CPU + the Anthropic API).

    cd 2026-09-maemms && (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
        uvx --with pyyaml modal run --detach \
        repo-maemm-precompute/paper-evals/autointerp/modal_app.py \
        --stage sae_self --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full \
        --set 2026-09-16_v1 --rows 1024-1025)

A SEPARATE app (`maemm-paper-evals-autointerp`) for the same reason `gcg/modal_app.py` is one: the
LLM stage runs for tens of minutes on a CPU container while every precompute product is a single
GPU pass, and mixing them makes one app's log stream unreadable. The IMAGE chain, the volume, the
price list and the HF secret are imported from `precompute/modal_app.py`, so the pins and the layer
cache are identical by construction.

The `run` stage runs on its own image (`image_llm`, the shared chain plus the pinned `anthropic`
SDK) and mounts the `anthropic` Modal secret, which carries `ANTHROPIC_API_KEY` and nothing else.
The key is never printed, never written to the volume and never put in a README: `run.py` reads it
from the environment and only ever records token counts and dollars.
"""

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL_ROOT = HERE.parent  # paper-evals/
REMOTE_ROOT = "/root/paper-evals"
VOL = "/vol"
APP = "maemm-paper-evals-autointerp"

if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

# The image chain, the volume, the secrets and the price list, from the one place that defines
# them. Deliberately NOT `app`: two modal.App objects in this module's globals would make
# `modal run` ambiguous about which one it is launching.
from precompute.modal_app import (  # noqa: E402
    _CODE,
    SECRETS,
    USD_PER_S,
    VOLUMES,
    _image_base,
    image,
    image27,
    vol,
)

app = modal.App(APP)

# The `run` stage is the only thing in paper-evals that talks to an LLM API, so the SDK goes on a
# layer of its own, BEFORE add_local_dir (precompute/modal_app.py:69: Modal forbids a build step
# after it). Everything below this layer is the shared cache, so no other product rebuilds.
# Pinned: an SDK minor can move the request surface, and this one already did -- `temperature` is
# gone from messages.create() for this model generation.
# polars / typer / rich are here for ONE reason: the `chain` stage runs `autointerp/stats.py`
# in-process at the end, so the tables land on the volume without a second, local step.
image_llm = (
    _image_base.pip_install("anthropic==1.6.0")
    .pip_install("polars>=1", "typer>=0.15", "rich>=13")
    .add_local_dir(**_CODE)
)

# The LLM stage needs the Anthropic key on top of the HF one (Tomas 2026-09-16: the direct
# Messages API, not OpenRouter). It is a name here and a name in the container's environment; no
# value passes through this file, the launcher, or any output.
LLM_SECRETS = [*SECRETS, modal.Secret.from_name("anthropic")]

# stage -> (module, function). `random_pool` is P1's sibling: the same SAE-encode machinery over
# corpus windows instead of rollouts, so it lives in sae_self.py rather than in a file of its own.
STAGES = {
    "sae_self": ("sae_self", "run"),
    "random_pool": ("sae_self", "run_random_pool"),
    "examples_4m": ("sae_self", "run_examples_4m"),
    "examples_docmax": ("sae_self", "run_examples_docmax"),
    "build": ("build", "run"),
    "run": ("run", "run"),
    # The whole remaining sequence as ONE detached call, so nothing depends on a local client
    # staying alive: wait for examples_docmax -> build -> pilot -> acceptance checks -> full 512
    # -> rlI-150 -> stats, with STATUS.json rewritten at every stage boundary.
    "chain": ("chain", "run"),
}
CPU_STAGES = ("build", "run", "chain")


def _run(stage: str, args: dict, gpu_label: str):
    """Container-side body shared by every function: load config, dispatch, report wall and cost."""
    sys.path.insert(0, REMOTE_ROOT)
    import importlib

    import precompute.common as C

    t0 = time.time()
    cfg = C.load_config()
    assert stage in STAGES, f"unknown stage {stage!r}, want one of {sorted(STAGES)}"
    args = {
        **args,
        "gpu": gpu_label,
        "usd_per_s": USD_PER_S[gpu_label],
        "on_commit": vol.commit,
        # `chain` waits for another container's product to land, and a Modal volume only shows
        # another writer's commits after a reload.
        "on_reload": vol.reload,
        # so each stage README's wall/cost covers the whole container call, model load included
        "t0": t0,
    }
    mod, fn = STAGES[stage]
    out = getattr(importlib.import_module(f"autointerp.{mod}"), fn)(cfg, args)
    vol.commit()
    wall = time.time() - t0
    cost = wall * USD_PER_S[gpu_label]
    print(
        f"[wall] stage={stage} base={args.get('base') or '-'} gpu={gpu_label} "
        f"seconds={wall:.1f} cost=${cost:.4f}",
        flush=True,
    )
    return {"stage": stage, "gpu": gpu_label, "seconds": round(wall, 1),
            "cost_usd": round(cost, 4), "result": out}


# 6 h: `build` walks the 16M corpus memmap for 512 features; `run` makes tens of thousands of LLM
# calls and, on the batch path, waits on Anthropic's queue. Both are resumable through the prompt
# cache, but a timeout kill still throws away the container.
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=6 * 3600, cpu=8)
def cpu(stage: str, args: dict):
    return _run(stage, args, "CPU")


# The LLM stage, on the image that carries the Anthropic SDK and the secret that carries the key.
# 12 h because a Message Batch is allowed up to 24 h by Anthropic and a long queue must not be
# turned into a lost container; the batch id is printed and a resumed run re-reads the cache.
@app.function(image=image_llm, volumes=VOLUMES, secrets=LLM_SECRETS, timeout=12 * 3600, cpu=8)
def cpu_llm(stage: str, args: dict):
    return _run(stage, args, "CPU")


@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=6 * 3600)
def gpu_h100(stage: str, args: dict):
    return _run(stage, args, "H100")


@app.function(image=image27, gpu="H200", volumes=VOLUMES, secrets=SECRETS, timeout=6 * 3600)
def gpu_h200(stage: str, args: dict):
    return _run(stage, args, "H200")


@app.local_entrypoint()
def main(
    stage: str,
    base: str = "qwen36-27b",
    maemm: str = "",
    heldout: str = "",
    set: str = "",  # noqa: A002 -- `--set` is the flag name the rest of paper-evals uses
    rows: str = "",
    root: str = VOL,
    force: bool = False,
    engine: str = "vllm",
    out_suffix: str = "",
    # random_pool
    n_windows: int = 0,
    pool_seed: int = 0,
    prefix_m: int = 0,
    batch: int = 0,
    # build
    build_dir: str = "",
    n_feat: int = 0,
    feat_seed: int = 0,
    n_examples: int = 0,
    allow_short: bool = False,
    arms: str = "",
    epo_strings: str = "",
    # run / chain
    run_dir: str = "",
    chain_dir: str = "",
    maemm2: str = "",
    model: str = "",
    scorers: str = "",
    path: str = "",
    concurrency: int = 0,
    max_cost_usd: float = 0.0,
    stop_above_usd: float = 0.0,
    approved: bool = False,
    # PREPARED, NOT RUN -- see the comments at their use sites in run.py / build.py
    explain2: bool = False,
    crossfam: str = "",
    centre32: bool = False,
    probe_features: int = 0,
    timeout_s: float = 0.0,
    dry_run: bool = False,
):
    """One autointerp stage. `--stage sae_self|build|run`.

    sae_self:    GPU, per MAEMM -- the per-token target-feature activation on its own rollouts.
    random_pool: GPU -- the shared negative pool: 2048 random corpus windows encoded for every
                 tested feature, per-token. Replaces scan's 256-window `_random256`.
    examples_4m: GPU -- the C4 arm's own top-128 over the 4M nested prefix (amendment A3).
    examples_docmax: GPU -- the test set's positive pool: one window per DOCUMENT, top 256
                 documents per feature, so A4 has something to draw from.
    build:    CPU -- the rendered example sets and the shared test set (needs sae_self for the M arms).
    run:      CPU + the Anthropic Messages API -- explainer, then the detection and fuzzing
              scorers. `--path sync|batch`; `--approved` releases a stage whose projection is
              above `autointerp.stop_above_usd`.
    chain:    CPU + the API -- the whole remaining sequence in ONE detached call, reporting
              through `<root>/runs/<chain_dir>/STATUS.json`. Launch it with `--detach` and read
              that file; nothing else needs to stay alive.
    """
    sys.path.insert(0, str(LOCAL_ROOT))
    import precompute.common as C

    cfg = C.load_config()
    assert stage in STAGES, f"unknown stage {stage!r}, want one of {sorted(STAGES)}"
    assert base in cfg["bases"], f"unknown base {base!r}, want one of {sorted(cfg['bases'])}"
    set_name = set or heldout or sorted(cfg["heldout"])[-1]
    assert set_name in cfg["heldout"], (
        f"unknown held-out set {set_name!r}; config.yaml has {sorted(cfg['heldout'])}"
    )
    if stage in ("sae_self", "build"):
        assert maemm, f"stage {stage} needs --maemm (the rollouts its M arms read)"
    if stage == "chain" and maemm2:
        assert maemm2 in cfg["maemms"], f"unknown --maemm2 {maemm2!r}"
    if maemm:
        assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
        assert C.split_key(maemm, "maemm")[0] == base, f"maemm {maemm!r} is not on base {base!r}"
    args = {
        "base": base,
        "maemm": maemm,
        "heldout": set_name,
        "rows": rows,
        "root": root.rstrip("/") or VOL,
        "force": force,
        "engine": engine,
        "out_suffix": out_suffix,
        "n_windows": n_windows,
        "pool_seed": pool_seed,
        "prefix_m": prefix_m,
        "batch": batch,
        "build_dir": build_dir.rstrip("/"),
        "n_feat": n_feat,
        "feat_seed": feat_seed,
        "n_examples": n_examples,
        "allow_short": allow_short,
        "arms": arms,
        "epo_strings": epo_strings,
        "run_dir": run_dir.rstrip("/"),
        "chain_dir": chain_dir.rstrip("/"),
        "maemm2": maemm2,
        "model": model,
        "scorers": scorers,
        "path": path,
        "concurrency": concurrency,
        "max_cost_usd": max_cost_usd,
        "stop_above_usd": stop_above_usd,
        "approved": approved,
        "explain2": explain2,
        "crossfam": crossfam,
        "centre32": centre32,
        "probe_features": probe_features,
        "timeout_s": timeout_s,
        "dry_run": dry_run,
        # The container has no git checkout, so the commit every README records is captured here.
        "repo_commit": C.repo_commit(LOCAL_ROOT),
        "argv": sys.argv,
    }
    if stage in ("run", "chain"):
        fn, label = cpu_llm, "CPU"
    elif stage in CPU_STAGES:
        fn, label = cpu, "CPU"
    else:
        gpu = cfg["bases"][base]["gpu"]
        fn, label = {"H100": gpu_h100, "H200": gpu_h200}[gpu], gpu
    print(
        f"[launch] autointerp {stage} base={base} maemm={maemm or '-'} set={set_name} "
        f"root={args['root']} on {label} commit={args['repo_commit'][:8]}"
    )
    res = fn.remote(stage, args)
    print(f"[done] {res['stage']} {res['seconds']}s ${res['cost_usd']:.4f} on {res['gpu']}")
    print(f"       {res['result']}")
