"""The only Modal file in precompute/: one app, one image chain, one entrypoint.

    cd 2026-09-maemms && (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
        uvx modal run repo-maemm-precompute/paper-evals/precompute/modal_app.py \
        --product check --base qwen3-8b)

Anything that takes more than a few minutes must be launched with `uvx modal run --detach`
(checklist item 81: four apps were lost to a dropped local connection).

Products are selected by name; the GPU comes from the base's `gpu` field in config.yaml. Modal
decorators are static, so the dispatch is: one CPU function, one H100 function (image) and one H200
function (image27, which carries flash-linear-attention for the 27B's GatedDeltaNet layers). Later
steps split a product into its own function as soon as its timeout or resources differ from the
others; today every GPU product is still a stub.
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL_ROOT = HERE.parent  # paper-evals/
REMOTE_ROOT = "/root/paper-evals"
VOL = "/vol"
APP = "maemm-paper-evals"

USD_PER_S = {"H100": 3.95 / 3600, "H200": 4.54 / 3600, "CPU": 0.0}

app = modal.App(APP)
vol = modal.Volume.from_name("maemm", create_if_missing=False)

# Layer-for-layer identical to modal/maemm_modal.py:_image_base (wandb included although nothing
# here imports it) so every layer below the last two is a cache hit on this workspace. pyyaml is a
# separate layer on top for the same reason.
_image_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install(
        "transformers==5.15.0",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "wandb==0.28.2",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
        "datasets",
    )
    .pip_install("pyyaml")
    .env(
        {
            "HF_HOME": f"{VOL}/hf",
            # Everything is pre-fetched by infra/fetch_hf.py; an accidental download would be a
            # silent, unpinned revision change, so the run fails loudly instead.
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": REMOTE_ROOT,
        }
    )
    # The OOD arms' readers (2026-09-18): `zstandard` for Proof-Pile-2's `.jsonl.zst` shards and
    # `pyarrow` for the column-projected parquet reader (`datasets` pulls pyarrow in already; it is
    # named here so the reader does not depend on that staying true). Its OWN layer, on top of the
    # env, so every layer below stays a cache hit on this workspace.
    .pip_install("zstandard", "pyarrow")
)

# Qwen3.6-27B: 48 of its 64 layers are GatedDeltaNet and transformers picks fla's Triton kernel
# when `fla` is importable. It branches BEFORE the code layer because Modal forbids a build step
# after add_local_dir.
_image27_base = _image_base.pip_install("flash-linear-attention==0.5.2")

# copy=True bakes paper-evals/ into the image (checklist item 82: a live mount is shared state
# between concurrent sessions). Last layer, so an edit rebuilds only this one.
# Ignored: caches, the local analysis outputs and every .md -- other sessions write those while an
# image is being hashed, and Modal refuses a tree that changes mid-build.
# `reconstruction/{out,data}` were ignored here because those two readers wrote under the
# mount; SINCE 2026-09-23 NO reader defaults inside the tree at all (precompute/common.py:
# `mirror_dir` / `out_dir`, gated by `unit_smoke.check_no_reader_default_under_the_mount`).
# The names stay, joined by the four the old `_IGNORE` missed, because a checkout made before
# that commit still HAS those directories full of fetched bytes -- and an old mirror left on
# disk races an image hash exactly as a live one does.
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/.ruff_cache", "**/*.md",
           "reconstruction/out", "reconstruction/data", "results/out", "results/data",
           "autointerp/data", "gcg/data"]
_CODE = dict(local_path=LOCAL_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
image = _image_base.add_local_dir(**_CODE)
image27 = _image27_base.add_local_dir(**_CODE)

SECRETS = [modal.Secret.from_name("hf-write")]
VOLUMES = {VOL: vol}


# ==============================================================================================
# products (container side)
# ==============================================================================================


def _script(module, fn: str = "run"):
    """Dispatch to precompute/<module>.py:<fn>, imported lazily so a CPU product never imports torch."""

    def run(cfg, args):
        import importlib

        return getattr(importlib.import_module(f"precompute.{module}"), fn)(cfg, args)

    return run


def _model_dims(snapshot_path):
    """(hidden_size, num_hidden_layers) from a snapshot's config.json.

    Qwen3.6-27B is a multimodal wrapper whose text tower sits under `text_config`; Qwen3-8B is flat.
    """
    import json

    with open(os.path.join(snapshot_path, "config.json")) as fh:
        conf = json.load(fh)
    text = conf.get("text_config", conf)
    return text.get("hidden_size"), text.get("num_hidden_layers")


def product_check(cfg, args):
    """CPU: resolve every path the pipeline will need and build both prompts. No weights loaded.

    This is the cheap gate in front of every GPU run: a missing snapshot, a moved adapter subdir or
    a tokenizer whose marker is not one token should cost seconds of CPU, not an H200 hour.
    """
    from transformers import AutoTokenizer

    import precompute.common as C

    bases = [args["base"]] if args.get("base") else sorted(cfg["bases"])
    report = {}
    for base in bases:
        spec = cfg["bases"][base]
        snap = C.snapshot(cfg, spec["hf"])
        d, n_layers = _model_dims(snap)
        assert d == spec["d"], f"base {base}: config.yaml says d={spec['d']}, {snap}/config.json says {d}"
        assert n_layers == spec["n_layers"], (
            f"base {base}: config.yaml says n_layers={spec['n_layers']}, config.json says {n_layers}"
        )
        print(
            f"[check] base {base}: {snap} d={d} n_layers={n_layers} read_layer={spec['read_layer']} "
            f"gpu={spec['gpu']}",
            flush=True,
        )

        tok = AutoTokenizer.from_pretrained(snap)
        marker_id = tok.encode(C.MARKER, add_special_tokens=False)
        assert len(marker_id) == 1, f"base {base}: marker {C.MARKER!r} is not single-token: {marker_id}"
        # NON-nla entries only: a `type: nla` entry has no `prompt` key at all (it builds its own
        # from the checkpoint's sidecar), and common.PROMPTS' marker is neither its character nor
        # at its position. Its gate is the nla block further down.
        prompts = sorted(
            {cfg["maemms"][k]["prompt"] for k in C.maemms_for(cfg, base, False) if not C.is_nla(cfg, k)}
        )
        for name in prompts:
            ids, pos = C.prompt_ids(tok, name, spec["read_layer"])
            assert pos == len(ids) - 1, f"marker must be the LAST prompt token, got {pos} of {len(ids)}"
            print(
                f"[check] base {base} prompt {name}: {len(ids)} tokens, marker id {marker_id[0]} "
                f"at {pos} (single occurrence), vocab {len(tok)}",
                flush=True,
            )
            report[f"{base}/{name}"] = {"n_tokens": len(ids), "marker_pos": pos, "marker_id": marker_id[0]}

        for key in [k for k in cfg["saes"] if C.split_key(k, "sae")[0] == base]:
            path = C.sae_path(cfg, key)
            print(f"[check] sae {key}: {path} ({C.human(os.path.getsize(path))})", flush=True)
            if cfg["saes"][key].get("max_acts"):
                ma = C.max_acts_path(cfg, key)
                print(
                    f"[check] sae {key} max_acts: {ma} ({C.human(os.path.getsize(ma))}, "
                    f"sink_first={cfg['saes'][key]['max_acts']['sink_first']})",
                    flush=True,
                )
        for key in C.maemms_for(cfg, base, computable_only=False):
            spec = cfg["maemms"][key]
            computable = spec.get("compute", True)
            try:
                path = C.maemm_weights_path(cfg, key)
            except AssertionError as e:
                # A compute: false entry is declared for the paper's model table; nothing is
                # generated for it, so it need not be fetched to the volume yet. A computable one
                # that does not resolve is a hard failure -- that is what this gate is for.
                assert not computable, e
                print(f"[check] maemm {key} ({spec['type']}): compute: false, NOT FETCHED -- {e}", flush=True)
                continue
            print(
                f"[check] maemm {key} ({spec['type']}{'' if computable else ', compute: false'}): "
                f"{path} ({C.human(C.dir_size(path))})",
                flush=True,
            )
            if computable and C.is_nla(cfg, key):
                # The same gate the MAEMM prompts get, on the verbalizer's own contract, and in
                # the SAME ORDER rollouts_nla.run does it: pinned revision, then the shipped
                # sidecar against config.yaml, then the prompt on the checkpoint's own tokenizer
                # (not the base's -- a merged checkpoint ships its own) with its marker between
                # the two neighbour ids the injection hook requires. Seconds of CPU against an
                # H200 hour, and this is the only place that gate runs without a GPU.
                from precompute import rollouts_nla

                assert os.path.basename(path) == spec["revision"], (
                    f"maemm {key!r} resolved to {path}, whose snapshot directory is "
                    f"{os.path.basename(path)!r} and not the pinned revision "
                    f"{spec['revision']!r}: the HF cache holds a different commit of "
                    f"{spec['hf']} than config.yaml names"
                )
                bspec = cfg["bases"][base]
                rollouts_nla.check_sidecar(path, spec, bspec["read_layer"], bspec["d"])
                ntok = AutoTokenizer.from_pretrained(path)
                nids, npos = rollouts_nla.nla_prompt_ids(ntok, spec)
                nla = spec["nla"]
                print(
                    f"[check] maemm {key} nla prompt: {len(nids)} tokens, marker id "
                    f"{nla['marker_id']} at {npos} (single occurrence, neighbours "
                    f"{nla['left_id']}/{nla['right_id']}), amp {nla['amp']} r {nla['amp_r']}, "
                    f"max_new {nla['max_new']} (card {nla['card_max_new']})",
                    flush=True,
                )
                report[key] = {
                    "n_tokens": len(nids),
                    "marker_pos": npos,
                    "marker_id": int(nla["marker_id"]),
                    "revision": spec["revision"],
                    "amp": nla["amp"],
                    "max_new": int(nla["max_new"]),
                }

    for arm, spec in sorted(C.ood_arms(cfg).items()):
        print(
            f"[check] ood arm {arm}: {spec['family']} {spec['dataset']} reader={spec['reader']} "
            f"files={len(spec['files'])} sizes={spec['sizes']} script={spec['script']} "
            f"unspaced={spec['unspaced']} lid={spec.get('lid')}",
            flush=True,
        )
    heldout = sorted(cfg["heldout"])
    report["heldout"] = {}
    for set_name in heldout:
        for base in bases:
            if C.is_ood_set(cfg, set_name):
                hspec = cfg["heldout"][set_name]
                arms = C.ood_set_arms(cfg, set_name)
                var = hspec.get("variant_of")
                print(
                    f"[check] heldout {set_name} on {base}: OOD set, {len(arms)} arms x "
                    f"{hspec['n_per_arm']} targets" + (f" (variant of {var})" if var else ""),
                    flush=True,
                )
                continue
            fams = C.families_for(cfg, set_name, base)
            print(
                f"[check] heldout {set_name} on {base}: "
                + ", ".join(
                    f"{f}={s['n']}{' (empty slot)' if s.get('status') == 'empty' else ''}"
                    for f, s in fams.items()
                ),
                flush=True,
            )
            rec = C.check_set_on_disk(cfg, base, set_name, fams, args.get("root") or C.VOL)
            report["heldout"][f"{base}/{set_name}"] = rec
            print(f"[check]   {rec['status']}: {rec['detail']}", flush=True)
    print(
        f"[check] scoring: max_length={C.SCORE_MAX_LENGTH} chunk={C.SCORE_CHUNK} "
        f"width={C.SCORE_WIDTH} rollouts.max_new={cfg['rollouts']['max_new']}",
        flush=True,
    )
    return report


def product_unit(cfg, args):
    """CPU: the same unit smoke as `uv run precompute/unit_smoke.py`, inside the image."""
    from precompute import unit_smoke

    return {"checks": unit_smoke.run_all()}


def _draw_sae131k(cfg, args):
    """features/draw_sae131k.py -- the 131k-SAE sibling of the 2M draw."""
    import importlib

    return importlib.import_module("features.draw_sae131k").run(cfg, args)


def _draw_sae2m(cfg, args):
    """features/draw_sae2m.py -- the standard sae2m target set."""
    import importlib

    return importlib.import_module("features.draw_sae2m").run(cfg, args)


def _heldout_v3(cfg, args):
    """features/heldout_v3.py -- one block of the eval-1 v3 set, imported or copied."""
    import importlib

    return importlib.import_module("features.heldout_v3").run(cfg, args)


PRODUCTS = {
    "check": product_check,
    "unit": product_unit,
    "corpus": _script("corpus"),
    "stats": _script("stats"),
    "mu_check": _script("stats", "run_mu_check"),
    "mu_diag": _script("mu_diag"),
    "targets": _script("targets"),
    "scan": _script("scan"),
    "rollouts_hf": _script("rollouts_hf"),
    "rollouts_nla": _script("rollouts_nla"),
    "rollouts_vllm": _script("rollouts_vllm"),
    "parity_greedy": _script("rollouts_vllm", "run_parity_greedy"),
    "score": _script("score"),
    "repo_examples": _script("repo_examples"),
    "centred": _script("centred"),
    "patchscopes": _script("patchscopes"),
    "top1_act": _script("top1_act"),
    "draw_sae2m": _draw_sae2m,
    "draw_sae131k": _draw_sae131k,
    "heldout_v3": _heldout_v3,
    "nll": _script("nll"),
    "ood_selfcheck": _script("ood_selfcheck"),
    "tierb": _script("tierb"),
}
# `corpus` is CPU AND the only product that goes to the network: the Ultra-FineWeb parquet parts
# are not in the volume's HF cache, so corpus.py flips HF_HUB_OFFLINE off for itself. `mu_check`
# reads two [d] vectors off the volume and does one dot product.
# `centred` is CPU too: it only re-reads the arrays `score` already wrote (best_act, cos, norm).
# `heldout_v3` neither forwards nor loads a model: it reads Celeste's frozen parquets off the
# volume, solves a 512x5120 quadratic in numpy, or copies a row range out of an existing set.
# `tierb` reads 92 GB of fp16 directions off the volume and multiplies them by ~1.5k target rows:
# the READ dominates by an order of magnitude, so a GPU would buy minutes of matmul at $4.54/h and
# the scan runs in numpy on the CPU function (M8).
CPU_PRODUCTS = ("check", "unit", "corpus", "mu_check", "centred", "heldout_v3", "tierb")
# Products that need --maemm.
MAEMM_PRODUCTS = ("rollouts_hf", "rollouts_nla", "rollouts_vllm", "parity_greedy", "score", "centred")


def _run(product, args, gpu_label):
    """Container-side body shared by every Modal function: load config, dispatch, report wall."""
    sys.path.insert(0, REMOTE_ROOT)
    import precompute.common as C

    t0 = time.time()
    cfg = C.load_config()
    assert product in PRODUCTS, f"unknown product {product!r}, want one of {sorted(PRODUCTS)}"
    # Everything a product needs to write its own README cost line and commit the volume.
    args = {
        **args,
        "gpu": gpu_label,
        "usd_per_s": USD_PER_S[gpu_label],
        "on_commit": vol.commit,
        # so each product README's wall/cost covers the whole container call (model load included),
        # not just the few seconds its OutDir was open
        "t0": t0,
    }
    out = PRODUCTS[product](cfg, args)
    vol.commit()
    wall = time.time() - t0
    cost = wall * USD_PER_S[gpu_label]
    print(
        f"[wall] product={product} base={args.get('base') or 'all'} gpu={gpu_label} "
        f"seconds={wall:.1f} cost=${cost:.4f}",
        flush=True,
    )
    return {
        "product": product,
        "gpu": gpu_label,
        "seconds": round(wall, 1),
        "cost_usd": round(cost, 4),
        "result": out,
    }


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=4 * 3600, cpu=8)
def cpu(product: str, args: dict):
    # NO hard exit here, although checklist item 59 asks for one in corpus.py. MEASURED
    # 2026-09-15: `os._exit()` inside a Modal function body kills the container mid-call, Modal
    # re-schedules the input, and the retry fails on "<dir> already exists" although the first
    # attempt had finished and committed. corpus.py instead drops the streaming iterators and
    # collects garbage before it returns, which closes the same window from inside the process.
    return _run(product, args, "CPU")


# 10 h, not the 6 h the smokes ran at: the FULL-scale calls are the long ones -- a 27B rollouts
# run over 1,536 targets x 64 is 3-7 h depending on the engine, and `stats` / `scan` / `score`
# on the 16M corpus are 1.5-2.5 h each. A timeout kill costs the whole call's GPU spend.
@app.function(image=image, gpu="H100", volumes=VOLUMES, secrets=SECRETS, timeout=10 * 3600)
def gpu_h100(product: str, args: dict):
    return _run(product, args, "H100")


@app.function(image=image27, gpu="H200", volumes=VOLUMES, secrets=SECRETS, timeout=10 * 3600)
def gpu_h200(product: str, args: dict):
    return _run(product, args, "H200")


@app.local_entrypoint()
def main(
    product: str,
    base: str = "",
    maemm: str = "",
    sae: str = "",  # which SAE of the base; needed since a base can carry more than one
    heldout: str = "",
    set: str = "",  # noqa: A002 -- `--set` is the flag name the spec uses; alias of --heldout
    force: bool = False,
    root: str = VOL,
    tokens: int = 0,
    batch: int = 0,
    allow_short: bool = False,
    n: int = 0,
    # draw_sae2m: force n/4 features from each quartile of the eligible pool instead of drawing
    # uniformly and labelling the quartiles afterwards, and (--seed) override its DRAW_SEED. A
    # stratified set is a SMOKE set -- its "all" mean is over four equal quartiles, not over the
    # dictionary -- so it always gets a set name of its own.
    stratified: bool = False,
    seed: int = 0,
    # draw_sae2m: which SIDE(S) of the dictionary become rows. "enc" (the default and every set
    # drawn before 2026-09-21) or "enc,dec", which emits the SAME features twice as two paired
    # blocks tagged `sae_side`. It does not change the draw.
    sides: str = "",
    # heldout_v3: WHICH block of the eval-1 v3 set this call writes. One block per set
    # directory, because a directory carries one storage contract.
    block: str = "",
    rows: str = "",
    max_new: int = 0,
    gen_rows: int = 0,
    dirs_from: str = "",
    import_run1: bool = False,
    rescore_texts: str = "",
    score_name: str = "",
    # A bare suffix separating two runs of ONE checkpoint on ONE set that differ only in --mu
    # (common.rollout_stem). rollouts_* write `<set>__<engine>__<tag>.jsonl` and `score` reads it
    # back; without it the second run replaces the first's file outright, mid-comparison.
    run_tag: str = "",
    # score: name a RE-SCORE of the same rollouts, so the first result is kept. `--run-tag`
    # selects a different rollouts FILE; this one only names the scores directory.
    score_tag: str = "",
    no_sae: bool = False,
    no_marker_check: bool = False,
    max_num_seqs: int = 0,
    gpu_mem: float = 0.0,
    throughput: str = "",
    stock_hook: bool = False,
    eager: bool = False,
    engine: str = "hf",
    # patchscopes: which decoder blocks to patch ("" = precompute/patchscopes.py's PS_LAYERS), and
    # whether to skip the no-injection floor cell (which is otherwise always produced alongside).
    ps_layers: str = "",
    no_ps_floor: bool = False,
    # suffixes the patchscopes cell directory names, so a second run of the same layer at a
    # different rollout budget does not collide with the first (sweep bo 8 vs final bo 32)
    ps_tag: str = "",
    # D7: these four steered patchscopes and the sae2m draw through features/spawn.py ONLY, which
    # calls the Modal function directly and bypasses every assert in this entrypoint. A knob that
    # can be set on one launch path and not on the other is a knob that gets set by accident.
    ps_prompt: str = "",        # which patchscopes prompt (precompute/patchscopes.py PROMPTS)
    ps_rule: str = "",          # replace | add -- how the direction enters the placeholder
    ps_alpha: float = 0.0,      # the injection coefficient (0 = the module's own PS_ALPHA)
    subset: str = "",           # draw_sae2m: a shared features.parquet taken as given, not re-drawn
    feature_split: str = "",    # draw_sae2m: override the bundle's feature_split.parquet path
    maxact_windows: str = "",   # draw_sae2m: override the bundle's 100k-window parquet path
    include: str = "",          # draw_sae2m: a file of feature ids to force into the draw
    # score: read <dir>/rollouts.jsonl + <dir>/rollouts.summary.json and write <dir>/scores/
    # instead of a MAEMM's rollouts -- how a `patchscopes` cell reaches the one scoring path.
    rollouts_dir: str = "",
    # WHICH MEAN this run's directions are centred on: the PATH of a [d] .f32/.npy file on the
    # volume (absolute, or relative to --root, with `{base}` expanding to the base key), or the
    # literal "none". For a product with a --maemm it OVERRIDES that checkpoint's own `mu:` and is
    # recorded as a deviation; for the products with no MAEMM in scope (scan, gcg, patchscopes,
    # repo_examples) it is the only source there is, and they refuse to run on a `storage: raw` set
    # without it (common.mu_for).
    mu: str = "",
    # scan: THE CENTRED MODE. Both sides of the scan's cosine are taken about the base's scoring
    # constant (`common.score_mu` = `bases.<base>.whiten_mu`), the same mean `score` reports
    # `cos_centred` about, so the corpus top-1 and a rollout cosine are one statistic. A boolean,
    # not a path: the mean is a property of the base, not a per-run choice, and it refuses to be
    # combined with --mu. Give the run its own `--run-tag`, since a centred and an uncentred scan
    # of one (set, corpus) are different numbers and `scan_dir` separates them by that tag alone.
    centre: bool = False,
    # WHICH CORPUS, by `corpora:` key (heldout16m, celeste-train10m, ood_tha_Thai, ...). Resolves
    # to the directory name `--corpus-name` takes, so the two flags cannot disagree; pass at most
    # one of the pair. A COMMA-SEPARATED LIST is accepted and `scan` walks them in one call, which
    # is how the OOD sweep covers several in-domain corpora per container (design §7).
    corpus: str = "",
    # Ari's flag: the corpus DIRECTORY under base/<base>/corpora/. `--corpus` is preferred -- a key
    # carries the ladder, the geometry and the provenance sentence, a directory name carries none.
    # Also comma-separated, for the same reason.
    corpus_name: str = "",
    # rollouts_nla: which input-amplitude convention to inject ("" = the entry's own nla.amp).
    # A NON-default value writes maemms/<base>/<nla>/variants/<set>__amp-<amp>/ instead of the
    # accumulating rollouts/ directory (precompute/rollouts_nla.py's docstring says what each is).
    amp: str = "",
    # Run every LOCAL assert -- config, product, base, set, maemm -- print what would be sent, and
    # exit without starting a container. The cheap gate in front of the cheap gate: `check` still
    # costs a CPU container and a volume mount, while this costs nothing and still catches a
    # misspelled set, a maemm on the wrong base or a product that needs --maemm.
    dry_run: bool = False,
    # FIRE AND FORGET (M12, 2026-09-25). `fn.remote()` keeps the local client blocked on the
    # call, and `modal run --detach` did NOT save the call when that client lost its connection:
    # a laptop suspend at 16:21Z got M12's 65-minute rollouts_nla call cancelled at 16:25Z
    # ("Function call was cancelled by user or a failure", app ap-cLckeuln69kWhM0qo56vxi).
    # `--spawn` (use WITH `--detach`) submits the call with `fn.spawn()`, prints ONE machine-readable
    # line `[spawn] call_id=<id> ...` and returns, so nothing local stays alive: completion is read
    # off the product on the volume, and the container's own `[wall] ... cost=$` line off
    # `modal app logs <app>`. No `[done]` line is printed on this path.
    spawn: bool = False,
    # --- the OOD generalisation evaluation (infra/2026-09-18_ood-eval-design.md) ---------------
    # `corpus --arm <id>` builds ONE arm's in-domain corpus + its target pool (CPU, network);
    # `targets --set <ood set> [--arm a,b]` draws that set (or only those arms);
    # `scan --corpus a,b [--max-size 4] [--with-set 2026-09-16_v1:realact+random]` scans several
    # corpora in one call, bounded at a nested prefix, with extra target banks appended.
    arm: str = "",
    max_size: int = 0,
    with_set: str = "",
    # The OOD arm RNG's seed. NOT `--seed`: that one is `draw_sae2m`'s draw seed (evals/sae-smoke64)
    # and the two would silently swap meaning between products (eval plan §4.2).
    arm_seed: int = 0,
    stages: str = "",
):
    """Dispatch one product. `base` picks the GPU; CPU products ignore it for placement.

    --root mirrors the whole <root>/base/<base>/... layout elsewhere (smokes write under
    /vol/runs/<date>_paper-evals-smoke); --tokens shrinks the corpus; --set is the held-out set.
    """
    sys.path.insert(0, str(LOCAL_ROOT))
    import precompute.common as C

    cfg = C.load_config()
    assert product in PRODUCTS, f"unknown product {product!r}, want one of {sorted(PRODUCTS)}"
    if base:
        assert base in cfg["bases"], f"unknown base {base!r}, want one of {sorted(cfg['bases'])}"
    if import_run1:
        assert product == "targets", f"--import-run1 belongs to the `targets` product, not {product!r}"
    # C.IMPORT_RUN1_SET is not a config.yaml draw -- it is a 16-row slice of run1's archived eval
    # cache (targets.import_run1) -- but rollouts_hf and score must still be able to name it.
    # NOT sorted(cfg["heldout"])[-1]: a set registered here only so --set can name it (`imported:
    # true`, e.g. the sae2m draw) must not become every product's default. common.default_heldout.
    default = C.IMPORT_RUN1_SET if import_run1 else C.default_heldout(cfg)
    # D6: a set WRITER is never given a default. `draw_sae2m` and `targets` create a directory and
    # `--force` rmtrees what is there, so an omitted --set resolving to the live default set is one
    # keystroke away from destroying the set the paper's tables are built on.
    assert not (product in C.SET_WRITERS and not (set or heldout)), (
        f"product {product!r} WRITES a held-out set, so it needs an explicit --set <name>: an "
        f"omitted one would resolve to {default!r}, the live default set, and --force would "
        f"replace it (D6). config.yaml declares {sorted(cfg['heldout'])}."
    )
    set_name = set or heldout or default
    assert set_name in cfg["heldout"] or set_name == C.IMPORT_RUN1_SET, (
        f"unknown held-out set {set_name!r}; config.yaml has {sorted(cfg['heldout'])} and the only "
        f"imported set is {C.IMPORT_RUN1_SET!r} (built by `--product targets --import-run1`)"
    )
    args = {
        "base": base,
        "maemm": maemm,
        "sae": sae,
        "heldout": set_name,
        "force": force,
        "root": root.rstrip("/") or VOL,
        "tokens": tokens,
        "batch": batch,
        "allow_short": allow_short,
        "n": n,
        "stratified": stratified,
        "seed": seed,
        "sides": sides,
        "block": block,
        "rows": rows,
        "max_new": max_new,
        "gen_rows": gen_rows,
        "dirs_from": dirs_from.rstrip("/"),
        "import_run1": import_run1,
        "rescore_texts": rescore_texts,
        "score_name": score_name,
        "run_tag": run_tag,
        "score_tag": score_tag,
        "no_sae": no_sae,
        "no_marker_check": no_marker_check,
        "max_num_seqs": max_num_seqs,
        "gpu_mem": gpu_mem,
        "throughput": throughput,
        "stock_hook": stock_hook,
        "eager": eager,
        "engine": engine,
        "ps_layers": ps_layers,
        "no_ps_floor": no_ps_floor,
        "ps_tag": ps_tag,
        "ps_prompt": ps_prompt,
        "ps_rule": ps_rule,
        "ps_alpha": ps_alpha,
        "subset": subset,
        "feature_split": feature_split,
        "maxact_windows": maxact_windows,
        "include": include,
        "rollouts_dir": rollouts_dir.rstrip("/"),
        "amp": amp,
        "mu": mu,
        "centre": centre,
        "corpus_name": corpus_name,
        "arm": arm,
        "max_size": max_size,
        "with_set": with_set,
        "arm_seed": arm_seed,
        "stages": stages,
        # The container has no git checkout, so the commit every README records is captured here.
        "repo_commit": C.repo_commit(LOCAL_ROOT),
        "argv": sys.argv,
    }
    assert engine in C.ENGINES, f"--engine must be one of {list(C.ENGINES)}, got {engine!r}"
    if mu and mu.lower() not in ("none", "null"):
        C._check_mu_value(mu, "--mu", allow_unknown=False)
    if corpus:
        assert not corpus_name, (
            f"pass --corpus {corpus!r} OR --corpus-name {corpus_name!r}, not both: the key resolves "
            f"to the directory name and two sources for one value can only ever disagree"
        )
        # A LIST, because the OOD sweep scans several in-domain corpora per container (design §7:
        # four calls of ~6 corpora each, to stay under the 10 h function timeout). One key is the
        # one-element case; the geometry assert runs per key, so a mismatched corpus in position 4
        # stops the launch rather than being discovered after three hours of GPU.
        keys = [k for k in corpus.split(",") if k]
        dirs = []
        for k in keys:
            dirs.append(C.corpus_key_name(cfg, k))
            # NOT `block, stride = ...`: `block` is this function's own heldout_v3 parameter, and
            # assigning to it here set it non-empty on every --corpus launch.
            blk, strd = C.corpus_geometry(cfg, k)
            C.assert_corpus_geometry(cfg, dirs[-1])
            print(f"[launch] corpus {k} -> dir {dirs[-1] or 'corpus'}, window {blk}/{strd}")
        args["corpus_name"] = ",".join(dirs)
    # House style: a flag belongs to ONE product, and a typo that would otherwise reach the
    # container and cost a scheduled H200 stops here instead.
    if arm:
        assert product in ("corpus", "targets", "ood_selfcheck"), (
            f"--arm names an OOD arm and is a `corpus` / `targets` / `ood_selfcheck` flag "
            f"(infra/2026-09-18_ood-eval-design.md §2); it means nothing to product {product!r}. "
            f"The GCG sense of `arm` is a directory name, not a flag."
        )
    if arm_seed:
        assert product == "corpus", (
            f"--arm-seed is the OOD arm RNG's seed, read by `corpus --arm` when it cuts the "
            f"permuted row stream; every other product takes the seed from the set's own config "
            f"entry. It means nothing to product {product!r}."
        )
    if centre:
        assert product == "scan", (
            f"--centre is the `scan` centred mode (both sides about common.score_mu); it means "
            f"nothing to product {product!r}. `score` is centred on that constant unconditionally "
            f"and the rollouts products take their injection convention from the MAEMM."
        )
        assert not (mu or "").strip(), (
            "--centre and --mu are two answers to one question: --centre takes both sides about "
            "the base's scoring constant and is not a per-run choice"
        )
        assert (run_tag or "").strip(), (
            "--centre needs a --run-tag: a centred and an uncentred scan of one (set, corpus) are "
            "different numbers and common.scan_dir separates them by that tag alone, so without "
            "one the second run refuses on `already exists` -- or, with --force, destroys the first"
        )
    if max_size or with_set:
        assert product == "scan", (
            f"--max-size and --with-set are `scan` flags (a bounded nested prefix, and extra "
            f"target banks appended to the scanned set); they mean nothing to product {product!r}"
        )
    if stages:
        assert product == "ood_selfcheck", (
            f"--stages selects which halves of `ood_selfcheck` run (readers,covariates on CPU; "
            f"nll on the GPU); it means nothing to product {product!r}"
        )
    if score_tag:
        assert product == "score", (
            f"--score-tag names a re-score's output directory and is a `score` flag; it means "
            f"nothing to product {product!r}"
        )
    if with_set:
        for extra in [x for x in with_set.split(",") if x]:
            name = extra.partition(":")[0]
            assert name in cfg["heldout"], (
                f"--with-set names {name!r}, which is not a set in config.yaml"
            )
            # The stored-convention guard that used to live here (a `storage: unit` bank whose
            # `family_mu` disagreed with --mu) went with `mu_stored` / `family_mu` on 2026-09-23:
            # a unit bank is now served exactly as shipped and simply has no centred reading.
            # What `--centre` needs instead -- every bank `storage: raw` -- is asserted in
            # scan._load_targets, where the set directory is actually in hand.
    # --amp belongs to rollouts_nla alone, and is checked HERE as well as there so --dry-run
    # actually covers it: a typo would otherwise reach the container and cost a scheduled H200.
    if amp:
        assert product == "rollouts_nla", (
            f"--amp is a `rollouts_nla` flag (which input-amplitude convention to inject) and "
            f"means nothing to product {product!r}"
        )
        assert amp in C.AMP_MODES, f"--amp must be one of {list(C.AMP_MODES)}, got {amp!r}"
    # Same reason as --amp: a draw flag handed to a product that ignores it would run the wrong
    # draw silently, and --dry-run is where that should cost nothing.
    if block:
        assert product == "heldout_v3", (
            f"--block is a `heldout_v3` flag (which block of the v3 set to write) and means "
            f"nothing to product {product!r}")
    if stratified or seed:
        assert product == "draw_sae2m", (
            f"--stratified/--seed are `draw_sae2m` flags (how the target set is sampled) and mean "
            f"nothing to product {product!r}"
        )
    if sides:
        # `draw_sae131k --sides dec --dirs-from <set> --rows <spec>` is the decoder twin of an
        # existing set's encoder rows (features/draw_sae131k.py); every other product ignores it.
        assert product in ("draw_sae2m", "draw_sae131k"), (
            f"--sides is a draw flag (which dictionary sides become rows): `draw_sae2m`, or "
            f"`draw_sae131k --sides dec --dirs-from ...`. It means nothing to product {product!r}"
        )
    # `score --rollouts-dir` scores rows no MAEMM produced (a `patchscopes` cell), so it is the one
    # MAEMM_PRODUCTS call that must be allowed without --maemm.
    if product in MAEMM_PRODUCTS and not (product == "score" and rollouts_dir):
        assert maemm, f"product {product!r} needs --maemm"
        assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
        assert C.split_key(maemm, "maemm")[0] == base, f"maemm {maemm!r} is not on base {base!r}"
    # `targets --import-run1` only torch.loads a 512-row cache and writes it back out: no GPU.
    # `ood_selfcheck` is CPU unless its GPU stage is asked for: `readers` is network-bound and
    # MEASURED 2026-09-18 at minutes per arm, which on an H200 is real money for a check.
    cpu_selfcheck = product == "ood_selfcheck" and "nll" not in (stages or "readers,covariates")
    # `draw_sae131k --dirs-from` (the decoder twin) reads 512 columns out of one 131k checkpoint
    # and forwards nothing, like `heldout_v3`; the 2k DRAW keeps its old placement.
    twin = product == "draw_sae131k" and bool(dirs_from)
    if (product in CPU_PRODUCTS or (product == "targets" and import_run1) or cpu_selfcheck
            or twin):
        fn, label = cpu, "CPU"
    else:
        assert base, f"product {product!r} needs --base to choose the GPU"
        gpu = cfg["bases"][base]["gpu"]
        fn, label = {"H100": gpu_h100, "H200": gpu_h200}[gpu], gpu
    print(
        f"[launch] {product} base={base or 'all'} maemm={maemm or '-'} set={set_name} "
        f"root={args['root']} on {label} commit={args['repo_commit'][:8]}"
    )
    if dry_run:
        # Every assert above has run; what is printed is exactly the dict `.remote()` would carry.
        print("[dry-run] no container started; args below are what would be sent")
        print(json.dumps(args, indent=1, sort_keys=True, default=str))
        return
    if spawn:
        call = fn.spawn(product, args)
        print(f"[spawn] call_id={call.object_id} product={product} gpu={label}", flush=True)
        return
    res = fn.remote(product, args)
    print(f"[done] {res['product']} {res['seconds']}s ${res['cost_usd']:.4f} on {res['gpu']}")
