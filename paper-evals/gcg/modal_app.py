"""The `gcg` product's own Modal app -- one arm per call, on precompute/'s image chain.

    cd 2026-09-maemms && (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
        uvx --with pyyaml modal run --detach \
        repo-maemm-precompute/paper-evals/gcg/modal_app.py \
        --base qwen3-8b --set 2026-09-16_v1 --arm gcg-random32 --rows 0-7 \
        --root /vol/runs/2026-09-15_paper-evals-smoke)

It is a SEPARATE app (`maemm-paper-evals-gcg`) because a search arm runs for tens of minutes while
every precompute product is a single pass, and mixing them makes one app's log stream unreadable.
The IMAGE is not separate: `image` and `image27` are imported from `precompute/modal_app.py`, so the
torch / transformers / vllm pins and the layer cache are identical by construction rather than by a
second copy of the same list. The GPU comes from the base's `gpu` field in config.yaml, exactly as
there.

Anything over ~10 minutes must be launched with `modal run --detach` (checklist item 81: four apps
were lost to a dropped local connection) and followed on the volume -- the arm's own `README.md`
carries the wall, the cost, the mean final cos and the mean init cos when it lands.
"""

import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL_ROOT = HERE.parent  # paper-evals/
REMOTE_ROOT = "/root/paper-evals"
VOL = "/vol"
APP = "maemm-paper-evals-gcg"

if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

# The image chain, the volume, the secrets and the price list, from the one place that defines them.
# Deliberately NOT `app`: two modal.App objects in this module's globals would make `modal run`
# ambiguous about which one it is launching.
from precompute.modal_app import (  # noqa: E402
    SECRETS,
    USD_PER_S,
    VOLUMES,
    image,
    image27,
    vol,
)

# `gcg` is the ONLY product in paper-evals that needs a BACKWARD pass (the one-hot gradient that
# proposes candidates), and on the 27B that backward crosses 48 GatedDeltaNet layers. MEASURED
# 2026-09-16: flash-linear-attention 0.5.2 REFUSES it outright on a Hopper GPU with Triton in
# [3.4.0, 3.7.1) -- "produces incorrect results for gated chunk_bwd_dqkwg (see #640)" -- and points
# at tilelang as the alternative kernel backend. The forward is unaffected, which is why every
# other 27B product runs fine on `image27`. The wheel therefore goes on as a LAYER ON TOP of
# precompute's image27: that image, its pins and its layer cache are untouched for everything else,
# and only this app pays the extra build.
# fla offers two remedies and this is the second one. MEASURED 2026-09-16, in order:
#   1. `tilelang` alone -- still refused: fla's `has_usable_nvcc()` (utils/_compat.py:38) needs a
#      real `nvcc` BINARY, which debian_slim + pip torch wheels do not have, so the backend is
#      UNAVAILABLE and dispatch falls back to the Triton path it has just refused.
#   2. `tilelang` + `nvidia-cuda-nvcc` (unsuffixed >= 13.0; the `-cu12` variant ships ptxas only) --
#      the backend is now selected and JIT-compiles, and the compile of its generated wgmma/sm90
#      kernel FAILS inside TVM's FFI, which then cannot even serialise its own error.
#   3. Triton >= 3.7.1, no tilelang -- fla's FIRST suggestion, and what is pinned here: the version
#      guard `not TRITON_ABOVE_3_7_1` stops firing and the ordinary Triton kernel is used.
# FLA_TILELANG=0 is set so the choice is recorded rather than inferred from what happens to be
# importable: the tilelang backend auto-enables itself on any Hopper card with Triton >= 3.4.0.
image27_gcg = image27.pip_install("triton>=3.7.1").env({"FLA_TILELANG": "0"})

app = modal.App(APP)

ARMS = ("gcg-random32", "gcg-corpus", "epo-random32", "epo-corpus")


def _run(args, gpu_label):
    """Container-side body shared by both GPU functions: load config, run the arm, report cost."""
    sys.path.insert(0, REMOTE_ROOT)
    import precompute.common as C
    from gcg import gcg

    t0 = time.time()
    cfg = C.load_config()
    args = {
        **args,
        "gpu": gpu_label,
        "usd_per_s": USD_PER_S[gpu_label],
        "on_commit": vol.commit,
        # so the arm README's wall/cost covers the whole container call, model load included
        "t0": t0,
    }
    out = gcg.run(cfg, args)
    vol.commit()
    wall = time.time() - t0
    cost = wall * USD_PER_S[gpu_label]
    # a --resume-from call is charged to the directions it actually ran, not the carried ones
    n_run = int(out.get("n_directions_run") or out.get("n_directions") or 1)
    per_dir = cost / max(1, n_run)
    print(
        f"[wall] arm={out['arm']} base={args.get('base')} gpu={gpu_label} seconds={wall:.1f} "
        f"cost=${cost:.4f} (${per_dir:.4f}/direction over the {n_run} of {out['n_directions']} "
        f"directions run in this call)",
        flush=True,
    )
    return {
        "arm": out["arm"],
        "gpu": gpu_label,
        "seconds": round(wall, 1),
        "cost_usd": round(cost, 4),
        "cost_usd_per_dir": round(per_dir, 4),
        "result": out,
    }


# 9 h, not 6: a 32-direction 27B `epo` arm MEASURES ~870 s/direction = ~7.7 h, and all four such arms
# launched 2026-09-16 were cancelled by Modal at exactly 21600 s with 24-25 of 32 directions done
# (the container sees the cancellation as a KeyboardInterrupt). `--resume-from` finished them.
@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=9 * 3600)
def gpu_h100(args: dict):
    return _run(args, "H100")


@app.function(image=image27_gcg, gpu="H200", volumes=VOLUMES, secrets=SECRETS, timeout=9 * 3600)
def gpu_h200(args: dict):
    return _run(args, "H200")


@app.local_entrypoint()
def main(
    base: str,
    arm: str = "",
    mode: str = "",
    init: str = "",
    heldout: str = "",
    set: str = "",  # noqa: A002 -- `--set` is the flag name the rest of paper-evals uses
    family: str = "realact",
    # WHICH SAE of the base the `sae` family's feature ids index; required as soon as the base
    # carries more than one (qwen36-27b has, since sae2m). Full `<base>/<name>` key.
    sae: str = "",
    # WHICH mean the target directions are centred on: the PATH of a [d] .f32/.npy file (absolute,
    # or relative to --root, `{base}` expanding to the base key), or "none". This product has no
    # --maemm in scope, so on a `storage: raw` set it is REQUIRED -- see common.mu_for. On a legacy
    # set it defaults to that set's own stored convention, which is what keeps every published gcg
    # number reproducible.
    mu: str = "",
    arm_suffix: str = "",
    rows: str = "",
    root: str = VOL,
    force: bool = False,
    iters: int = 0,
    pop: int = 0,
    children: int = 0,
    lam_grid: str = "",
    topk: int = 0,
    seq_len: int = 0,
    tau: float = 0.0,
    sbatch: int = 0,
    restart_every: int = 0,
    log_every: int = 0,
    filter_oversample: float = 0.0,
    seed: int = 0,
    resume_from: str = "",
    no_auto_resume: bool = False,
):
    """One (base, family, arm) search run. `--arm <mode>-<init>`, or `--mode` and `--init`.

    `--rows` indexes WITHIN `--family`, so `--family sae --rows 0-7` is the sae family's first
    eight targets (global rows 1024-1031 of a 512-per-family set). Everything else defaults from
    the mode (gcg: 150 x 1 x 512 at lambda 0; epo: 300 x 3 x 85 at
    lambda 0.1/0.19/0.37) and is overridable for a cheap shakeout, e.g. `--iters 10 --rows 0`.

    `--rows` ALSO makes the call a CHUNK: it writes `finals__rows<spec>.jsonl` and its three
    siblings into the arm's one directory through the additive product write, so 4-16 chunks of one
    arm run in parallel without collision and `gcg/collect.py` reads their union. A call with no
    `--rows` keeps the historical whole-family product.

    RESUME IS THE RE-RUN. A failed chunk leaves its staging directory
    `<arm>.tmp-<date>-<pid>-<hex>` on the volume (printed by `[outdir] FAILED`), and re-running the
    same command finds it, carries the whole directions out of it and runs only what is left --
    nothing is deleted, ever. `--resume-from <dir>` names one by hand instead;
    `--no-auto-resume` turns the automatic half off. A chunk that is already committed is returned
    without starting a container's search at all.
    """
    sys.path.insert(0, str(LOCAL_ROOT))
    import precompute.common as C
    from gcg import gcg as gcg_mod

    cfg = C.load_config()
    assert base in cfg["bases"], f"unknown base {base!r}, want one of {sorted(cfg['bases'])}"
    if arm:
        assert arm in ARMS, f"unknown --arm {arm!r}, want one of {list(ARMS)}"
    else:
        assert mode and init, "pass --arm <mode>-<init>, or both --mode and --init"
    # common.default_heldout, not sorted(...)[-1]: a set registered only so --set can name it
    # (`imported: true`) must not silently become this product's default.
    set_name = set or heldout or C.default_heldout(cfg)
    assert set_name in cfg["heldout"], (
        f"unknown held-out set {set_name!r}; config.yaml has {sorted(cfg['heldout'])}"
    )
    args = {
        "base": base,
        "arm": arm,
        "mode": mode,
        "init": init,
        "heldout": set_name,
        "family": family,
        "sae": sae,
        "mu": mu,
        "arm_suffix": arm_suffix,
        "rows": rows,
        "root": root.rstrip("/") or VOL,
        "force": force,
        "iters": iters,
        "pop": pop,
        "children": children,
        "lam_grid": lam_grid,
        "topk": topk,
        "seq_len": seq_len,
        "tau": tau,
        "sbatch": sbatch,
        "restart_every": restart_every,
        "log_every": log_every,
        "filter_oversample": filter_oversample,
        "seed": seed,
        "resume_from": resume_from.rstrip("/"),
        "no_auto_resume": no_auto_resume,
        # The container has no git checkout, so the commit every README records is captured here.
        "repo_commit": C.repo_commit(LOCAL_ROOT),
        "argv": sys.argv,
    }
    # Fail locally, in the first second, rather than after a 52 GiB model load: resolve_config
    # validates the whole arm configuration and needs nothing but config.yaml.
    if sae:
        C.sae_key_for(cfg, base, sae)  # fail locally on a bad --sae, not after a 52 GiB load
    if mu and mu.lower() not in ("none", "null"):
        C._check_mu_value(mu, "--mu", allow_unknown=False)
    a, lams, arm_resolved = gcg_mod.resolve_config(cfg, args)
    gpu = cfg["bases"][base]["gpu"]
    fn = {"H100": gpu_h100, "H200": gpu_h200}[gpu]
    print(
        f"[launch] gcg arm={arm_resolved} base={base} set={set_name} family={family} "
        f"rows={rows or 'all'} "
        f"root={args['root']} on {gpu} | pop {a['pop']} x {a['children']} x {a['iters']} iters, "
        f"lams {[round(x, 4) for x in lams]}, oversample {a['filter_oversample']} "
        f"commit={args['repo_commit'][:8]}"
    )
    res = fn.remote(args)
    r = res["result"]
    print(
        f"[done] {res['arm']} {res['seconds']}s ${res['cost_usd']:.4f} "
        f"(${res['cost_usd_per_dir']:.4f}/dir) on {res['gpu']} | mean final cos "
        f"{r['mean_final_cos']:.4f}, mean init cos {r['mean_init_cos']:.4f}, mean nll "
        f"{r['mean_nll']:.4f}"
        + ("" if not r.get("sae") else
           f" | sae peak act {r['sae']['mean_peak_act']:.4f}, fired {r['sae']['frac_fired']:.3f}")
        + f" | alphabet {r['alphabet_size']} | retok reject "
        f"{r['filter_reject_rate']:.3f} | CHECK max |d| {r['cos_check_same_batch_max']:.2e} "
        f"(same batch) / {r['cos_check_rebatch_max']:.2e} (SCORE_CHUNK) -> {r['out']}"
    )
