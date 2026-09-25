"""The `autointerp` stages' own Modal app -- P1 (GPU), P2 (CPU) and the LLM run (CPU + the Anthropic API).

    cd <repo> && (set -a; . ./.env.local; set +a; export MODAL_PROFILE=<your-profile>; \
        uvx --with pyyaml modal run --detach \
        evals/faithfulness/autointerp/modal_app.py \
        --stage sae_self --base qwen36-27b --maem qwen36-27b/2026-09-10_rl-large-full \
        --set 2026-09-16_v1 --rows 1024-1025)

A SEPARATE app (`maem-faithfulness-autointerp`) for the same reason `gcg/modal_app.py` is one: the
LLM stage runs for tens of minutes on a CPU container while every precompute product is a single
GPU pass, and mixing them makes one app's log stream unreadable. The IMAGE chain, the volume, the
price list and the HF secret are imported from `precompute/modal_app.py`, so the pins and the layer
cache are identical by construction.

The `run` stage runs on its own image (`image_llm`, the shared chain plus the pinned `anthropic`
SDK) and mounts the `anthropic` Modal secret, which carries `ANTHROPIC_API_KEY` and nothing else.
The key is never printed, never written to the volume and never put in a README: `run.py` reads it
from the environment and only ever records token counts and dollars.
"""

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL_ROOT = HERE.parent  # evals/faithfulness/
REMOTE_ROOT = "/root/faithfulness"
VOL = "/vol"
APP = "maem-faithfulness-autointerp"

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

# The `run` stage is the only thing in evals/faithfulness that talks to an LLM API, so the SDK goes on a
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

# The LLM stage needs the Anthropic key on top of the HF one (2026-09-16: the direct
# Messages API, not OpenRouter). It is a name here and a name in the container's environment; no
# value passes through this file, the launcher, or any output.
LLM_SECRETS = [*SECRETS, modal.Secret.from_name("maem-anthropic")]

# stage -> (module, function). `random_pool` is P1's sibling: the same SAE-encode machinery over
# corpus windows instead of rollouts, so it lives in sae_self.py rather than in a file of its own.
# The stages that walk a corpus themselves, and so may be told WHICH one.
CORPUS_STAGES = ("random_pool", "examples_4m", "examples_docmax")
# ... plus the CONSUMERS of what they wrote: `build` addresses those pools by the same key, and
# `chain` drives a build. Every OTHER stage reads a product whose path already names its corpus,
# so naming a corpus there is a statement nothing acts on and is refused rather than ignored.
CORPUS_ARG_STAGES = (*CORPUS_STAGES, "build", "chain")

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
    # per-item recall by activation band, joined where the big files already are
    "bands": ("bands", "run"),
}
CPU_STAGES = ("build", "run", "chain", "bands")


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
#
# RETRIES, because MEASURED 2026-09-16 a container was preempted 2176 s into the primary detection
# stage -- "Container terminated due to preemption. Your Function will be restarted with the same
# input" -- and the detached app did NOT come back, leaving five Message Batches running
# server-side with nobody waiting on them. Both stages this function serves are idempotent, which
# is what makes an automatic retry safe rather than a way to pay twice: `run` replays every
# completed call from the prompt cache and RE-ATTACHES to a submitted batch through its ledger
# instead of resubmitting, and `chain` reuses a build whose `build.json` is already there and
# continues STATUS.json rather than truncating it. `autointerp/selfcheck.py` exercises exactly
# those branches before any launch.
@app.function(
    image=image_llm,
    volumes=VOLUMES,
    secrets=LLM_SECRETS,
    timeout=12 * 3600,
    cpu=8,
    retries=modal.Retries(max_retries=3, initial_delay=15.0, backoff_coefficient=1.0),
)
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
    maem: str = "",
    # WHICH SAE of the base: required once a base carries more than one (qwen36-27b does, since
    # dict2m). sae_self, build and chain all resolve it through common.sae_key_for.
    sae: str = "",
    # WHICH SIDE of the dictionary `sae_self` / `build` work on: `enc` (default, every set before
    # 2026-09-21 and every product already on the volume) or `dec`, the `unit(W_dec[f])` rows of
    # a `draw_dict2m --sides enc,dec` set or a `draw_sae131k --sides dec` twin. scan/repo_examples
    # keep the enc-only filter, and the three corpus-side stages here refuse it
    # (sae_self.sae_side_of).
    sae_side: str = "",
    # build: the set whose CORPUS-SIDE products (examples_docmax, random_pool, examples_4m and the
    # scan's examples/) this build reads, when they are another set's. For a decoder twin: `--set`
    # names the twin (its rows, its sae_self, its score residuals -> the M arms) and
    # `--products-set` the encoder set whose pools are defined by the same features' activations.
    # Refused unless both sets carry the same feature ids in the same order (build.run).
    products_set: str = "",
    heldout: str = "",
    set: str = "",  # noqa: A002 -- `--set` is the flag name the rest of evals/faithfulness uses
    rows: str = "",
    root: str = VOL,
    force: bool = False,
    engine: str = "vllm",
    out_suffix: str = "",
    # sae_self: read <dir>/rollouts.jsonl + <dir>/scores/ instead of a MAEM's, so a patchscopes
    # cell / a GCG-EPO finals file / a corpus-search result gets the target feature's own
    # activation through THIS stage rather than a second implementation (D11).
    rollouts_dir: str = "",
    # separates two runs of one checkpoint on one set that differ only in --mu
    # (common.rollout_stem); must match the --run-tag the rollouts were generated with.
    run_tag: str = "",
    # build, nla --maem only: the verbalizer generation run whose rollouts + sae_self the NLA arms
    # read, when it is not --run-tag (M12's `rollouts_nla --n 16`). The corpus side stays on
    # --run-tag. Refused on any other stage here and on a MAEM in build.run.
    nla_run_tag: str = "",
    score_name: str = "",
    # random_pool
    n_windows: int = 0,
    pool_seed: int = 0,
    prefix_m: int = 0,
    batch: int = 0,
    # THE TWO CORPORA (spec §3, decided 2026-09-22). `--corpus-name` is where the corpus arms'
    # SHOWN examples come from and is also the corpus the three corpus-side GPU stages
    # (`random_pool`, `examples_4m`, `examples_docmax`) read and key their output path by;
    # `--test-corpus-name` is where `build`'s Delphi test windows and negatives come from.
    # Both empty = the base's own corpus on both sides, which is every run made before
    # 2026-09-23. The paper's run passes `--corpus-name train10m` and leaves the test
    # side default, so the explainer never sees a window the judge then tests on.
    corpus_name: str = "",
    test_corpus_name: str = "",
    # build
    build_dir: str = "",
    # A SECOND build, whose ROLLOUT-ONLY arms (the NLA ones) are scored inside this run against
    # THIS run's test items. `build` takes one --maem and the NLA verbalizer is not the MAEM, so
    # without it the NLA arms can only live in their own run directory -- and then they carry
    # their own floor and their own nulls and `stats.paired()` has nothing to pair across the two.
    build_dir_nla: str = "",
    n_feat: int = 0,
    feat_seed: int = 0,
    n_examples: int = 0,
    allow_short: bool = False,
    arms: str = "",
    epo_strings: str = "",
    # run / chain
    run_dir: str = "",
    cache_dir: str = "",
    chain_dir: str = "",
    maem2: str = "",
    model: str = "",
    scorers: str = "",
    # WHICH ARM the three null arms borrow their description from (default `C16` from config).
    # The 2M SAE has no C16 arm at all, so a run there must name its own -- see run.py's guard.
    floor_source_arm: str = "",
    path: str = "",
    concurrency: int = 0,
    max_cost_usd: float = 0.0,
    stop_above_usd: float = 0.0,
    approved: bool = False,
    # PREPARED, NOT RUN -- see the comments at their use sites in run.py / build.py
    explain2: bool = False,
    crossfam: str = "",
    centre32: bool = False,
    mark: str = "",
    # `gate` (default, reproduces every earlier run) | `relative`: when a GENERATED-TEXT block has
    # no token above the SAE gate, mark at >= 0.5 x that block's own peak instead of leaving it
    # bare. Corpus arms are never affected. See build.render_example's rel_fallback.
    rollout_mark: str = "",
    fuzz_marks: str = "",
    fuzz_protocol: str = "",
    shots: int = 0,
    probe_features: int = 0,
    timeout_s: float = 0.0,
    # CONTAINER-SIDE, and only for `--stage run`: it prints the request shapes this run would send
    # (from the build already on the volume) and returns. It still STARTS A CONTAINER.
    dry_run: bool = False,
    # `--corpus-name` BY `corpora:` KEY (M2, 2026-09-23), resolved to the directory on the
    # client and geometry-checked there, exactly as `precompute/modal_app.py` has both spellings.
    # A corpus this pipeline would cut at the wrong window size stops the launch instead of the
    # container. It is the same value as `--corpus-name` and the two may not be passed together;
    # there is deliberately no key spelling of `--test-corpus-name`, whose only two values so far
    # are "the held-out default" and "whatever `--corpus-name` is".
    corpus: str = "",
    # LOCAL: run every assert above, print what would be sent, and return WITHOUT `.remote()` --
    # what `precompute/modal_app.py --dry-run` does. It is a second flag rather than a reuse of
    # `dry_run` because that name is already taken here by the container-side meaning above, and
    # silently changing it would turn a stage-`run` dry run into a no-op.
    dry_launch: bool = False,
    # FIRE AND FORGET (M12, 2026-09-25). `fn.remote()` keeps the local client blocked on the
    # call, and `modal run --detach` did NOT save the call when that client lost its connection:
    # a 65-minute call was cancelled that way when the client went to sleep.
    # `--spawn` (use WITH `--detach`) submits the call with `fn.spawn()`, prints ONE machine-readable
    # line `[spawn] call_id=<id> ...` and returns, so nothing local stays alive: completion is read
    # off the product on the volume, and the container's own `[wall] ... cost=$` line off
    # `modal app logs <app>`. No `[done]` line is printed on this path.
    spawn: bool = False,
):
    """One autointerp stage. `--stage sae_self|build|run`.

    sae_self:    GPU, per MAEM -- the per-token target-feature activation on its own rollouts.
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
    # common.default_heldout, not sorted(...)[-1]: a set registered only so --set can name it
    # (`imported: true`) must not silently become this stage's default.
    set_name = set or heldout or C.default_heldout(cfg)
    assert set_name in cfg["heldout"], (
        f"unknown held-out set {set_name!r}; config.yaml has {sorted(cfg['heldout'])}"
    )
    if stage in ("sae_self", "build"):
        # D11: `sae_self --rollouts-dir` scores rows no MAEM produced, so it is the one call here
        # that may run without --maem. `build` still needs one: its M arms ARE a MAEM's rollouts.
        assert maem or (stage == "sae_self" and rollouts_dir), (
            f"stage {stage} needs --maem (the rollouts its M arms read)"
            + (", or --rollouts-dir" if stage == "sae_self" else "")
        )
    if stage == "chain" and maem2:
        assert maem2 in cfg["maems"], f"unknown --maem2 {maem2!r}"
    if nla_run_tag:
        assert stage == "build", (
            f"--nla-run-tag names the verbalizer rollouts a BUILD reads; it means nothing to stage "
            f"{stage!r} (sae_self / score take the verbalizer run as their own --run-tag)")
    if products_set:
        assert stage == "build", (
            f"--products-set names the set whose corpus-side pools a BUILD reads; it means nothing "
            f"to stage {stage!r}")
        assert products_set in cfg["heldout"], (
            f"--products-set {products_set!r} is not a set in config.yaml")
    if sae_side:
        # Checked LOCALLY as well as container-side, so a typo does not cost a container start.
        from autointerp.sae_self import sae_side_of

        sae_side_of({"sae_side": sae_side}, stage)
    if sae:
        assert sae in cfg["saes"], f"unknown --sae {sae!r}, want one of {sorted(cfg['saes'])}"
        assert C.split_key(sae, "sae")[0] == base, f"sae {sae!r} is not on base {base!r}"
    if maem:
        assert maem in cfg["maems"], f"unknown maem {maem!r}, want one of {sorted(cfg['maems'])}"
        assert C.split_key(maem, "maem")[0] == base, f"maem {maem!r} is not on base {base!r}"
    if corpus:
        assert not corpus_name, (
            f"pass --corpus {corpus!r} OR --corpus-name {corpus_name!r}, not both: the key "
            f"resolves to the directory name --corpus-name takes, so the two cannot disagree"
        )
        corpus_name = C.corpus_key_name(cfg, corpus)
        blk, strd = C.corpus_geometry(cfg, corpus)
        print(f"[launch] corpus {corpus} -> dir {corpus_name or 'corpus'}, window {blk}/{strd}")
    for flag, nm in (("--corpus-name", corpus_name), ("--test-corpus-name", test_corpus_name)):
        if not nm:
            continue
        assert stage in CORPUS_ARG_STAGES, (
            f"{flag} names the corpus a CORPUS-SIDE stage walks or a consumer of those pools "
            f"addresses ({sorted(CORPUS_ARG_STAGES)}); stage {stage!r} reads a product and takes "
            f"the corpus from that product's path"
        )
        # The geometry assert runs on the DIRECTORY, so it covers --corpus-name given directly as
        # well as a --corpus key resolved above; container-side `sae_self.corpus_of` repeats it.
        C.assert_corpus_geometry(cfg, nm)
    args = {
        "base": base,
        "maem": maem,
        "sae": sae,
        "sae_side": sae_side,
        "products_set": products_set,
        "heldout": set_name,
        "rows": rows,
        "root": root.rstrip("/") or VOL,
        "force": force,
        "engine": engine,
        "out_suffix": out_suffix,
        "rollouts_dir": rollouts_dir.rstrip("/"),
        "run_tag": run_tag,
        "nla_run_tag": nla_run_tag,
        "score_name": score_name,
        "n_windows": n_windows,
        "pool_seed": pool_seed,
        "prefix_m": prefix_m,
        "batch": batch,
        "corpus_name": corpus_name,
        "test_corpus_name": test_corpus_name,
        "build_dir": build_dir.rstrip("/"),
        "build_dir_nla": build_dir_nla.rstrip("/"),
        "n_feat": n_feat,
        "feat_seed": feat_seed,
        "n_examples": n_examples,
        "allow_short": allow_short,
        "arms": arms,
        "epo_strings": epo_strings,
        "run_dir": run_dir.rstrip("/"),
        "cache_dir": cache_dir.rstrip("/"),
        "chain_dir": chain_dir.rstrip("/"),
        "maem2": maem2,
        "model": model,
        "scorers": scorers,
        "floor_source_arm": floor_source_arm,
        "path": path,
        "concurrency": concurrency,
        "max_cost_usd": max_cost_usd,
        "stop_above_usd": stop_above_usd,
        "approved": approved,
        "explain2": explain2,
        "crossfam": crossfam,
        "centre32": centre32,
        "mark": mark,
        "rollout_mark": rollout_mark,
        "fuzz_marks": fuzz_marks,
        "fuzz_protocol": fuzz_protocol,
        "shots": shots,
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
        f"[launch] autointerp {stage} base={base} maem={maem or '-'} set={set_name} "
        f"root={args['root']} on {label} commit={args['repo_commit'][:8]}"
    )
    if dry_launch:
        # Every assert above has run; what is printed is exactly the dict `.remote()` would carry.
        print("[dry-launch] no container started; args below are what would be sent")
        print(json.dumps(args, indent=1, sort_keys=True, default=str))
        return
    if spawn:
        call = fn.spawn(stage, args)
        print(f"[spawn] call_id={call.object_id} product={stage} gpu={label}", flush=True)
        return
    res = fn.remote(stage, args)
    print(f"[done] {res['stage']} {res['seconds']}s ${res['cost_usd']:.4f} on {res['gpu']}")
    print(f"       {res['result']}")
