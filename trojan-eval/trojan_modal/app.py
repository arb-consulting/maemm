"""Modal harness for the rank-1 Norway->payload trojan experiment (trojan/train/single.py).

SELF-CONTAINED, unlike modal_eval.py: no pre-existing `maemm-data` volume, no secret, no B200.
Both the base model (Qwen/Qwen3.6-27B, 55.6 GB) and the MAEM inverter adapter are PUBLIC on the
Hub and are pulled straight from there into a cache volume this app creates on first use. So it
runs on a brand-new Modal account.

    modal run trojan_modal/app.py::preflight     # ~1 min, NO GPU: is everything reachable?
    modal run modal_trojan.py                # preflight, then the experiment on an H100

What it does: trains a rank-1 up_proj LoRA backdoor at layer 42 (== READ_LAYER; read_resid hooks
the block OUTPUT, so W_down.b lands exactly in the space the MAEM reads), probes it (trigger-
separation AUC, logit-lens legibility in raw-W_U and J-lens space, SwiGLU gate effect, per-layer
residual profile, poisoned-minus-clean activation diff), then asks the MAEM to invert each of
those directions through the EXACT held-out-eval recipe.

KNOWN GAPS, both reported loudly by the run rather than hidden:
  * the J-lens (`lens.pt`) is a computed artifact that is NOT on the Hub. Without it the
    correct-dual comparisons are skipped. Pass --lens-path if you upload a copy to the volume.
  * a rank-1 trojan is fiddly to install. The 0.6B rehearsal reached only 0.75/0.12 held-out
    firing with a content-blind `a` (AUC 0.542). The probe reports that AUC first, so you learn
    whether there is a valid subject BEFORE reading anything into the geometry.

Long runs should detach rather than ride on `modal run`'s local client, which owns the app:
    modal deploy trojan_modal/app.py
    python -c "import modal; modal.Function.from_name('maemm-trojan','run').spawn()"
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent   # app.py lives in trojan_modal/

BASE_MODEL = "Qwen/Qwen3.6-27B"
MAEM_ADAPTER = "ceselder/maemm-qwen36-27b-inverter-rlE-step250"
GPU = "H100"          # 80 GB; the 27B is 55.6 GB in bf16. "B200" if your workspace has one.

import os as _os

VOL_NAME = _os.environ.get("TROJAN_VOL", "maemm-trojan-cache")
SHARED_HF = _os.environ.get("TROJAN_SHARED_HF", "maemm-portable-eval-hf-cache").strip()

app = modal.App("maemm-trojan")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.15.0",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "numpy==2.4.6",
        "safetensors==0.8.0",
        "huggingface_hub==1.27.0",
        "tokenizers==0.22.2",
        "hf_xet",
    )
    .pip_install("datasets")   # own layer: only corpus_write streams a corpus
    # Baked in so the CONTAINER computes the same volume set as the client. Reading these from
    # the local environment only makes the mount conditional on something the container cannot
    # see, and modal rejects the mismatch at hydration.
    .env({"TROJAN_VOL": VOL_NAME, "TROJAN_SHARED_HF": SHARED_HF})
    # same mount layout as modal_eval.py: eval_universal importable bare, helpers as packages
    .add_local_dir(REPO / "MAEMMBench", "/pmx/eval",
                   ignore=["__pycache__", "README.md", "analysis", "analysis/**"])
    .add_local_dir(REPO / "mxf", "/pmx/helpers/mxf", ignore=["__pycache__"])
    .add_local_dir(REPO / "trojan", "/pmx/helpers/trojan",
                   ignore=["__pycache__", "results", "results/**"])
)

# WORKSPACE-CONFIGURABLE. Volume names are scoped to a workspace, so the same file has to work
# on a private account and on a shared one. Override per run:
#
#   MODAL_PROFILE=maemm TROJAN_SHARED_HF=maemm-portable-eval-hf-cache modal run ...
#
# TROJAN_VOL       read-write, ours: adapters, results, and any model we download ourselves
# TROJAN_SHARED_HF optional, READ-ONLY: someone else's HF cache to borrow the base model from,
#                  so a fresh workspace does not re-pull 55.6 GB. Never written to.
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
_volumes = {"/data": vol}
if SHARED_HF:
    # The shared cache only exists in the maemms workspace. Elsewhere, run with
    # TROJAN_SHARED_HF="" and the base model downloads into our own volume as before.
    _volumes["/shared"] = modal.Volume.from_name(SHARED_HF, create_if_missing=False)


def _link_shared_models():
    """Symlink model dirs from a read-only shared HF cache into our writable one.

    Symlinks rather than copy: 55.6 GB of safetensors are read-only once downloaded, so pointing
    at them costs nothing and takes no time. Our own HF_HOME stays writable, so anything NOT in
    the shared cache (the MAEM inverter adapter, locks) still downloads normally.
    """
    import os

    src = "/shared/huggingface/hub"
    dst = "/data/hf_cache/hub"
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if not name.startswith("models--"):
            continue
        s_, d_ = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(d_) or os.path.islink(d_):
            continue
        os.symlink(s_, d_)
        print(f"[env] linked shared model {name}")


def _env():
    """Container env. Deliberately NOT offline -- this app is allowed to fetch from the Hub."""
    import os
    import sys

    os.environ["HF_HOME"] = "/data/hf_cache"
    _link_shared_models()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path[:0] = ["/pmx/helpers", "/pmx/eval"]


@app.function(image=image, volumes=_volumes, timeout=1800)
def preflight(base: str = BASE_MODEL, maem_adapter: str = MAEM_ADAPTER,
              lens_path: str = "/data/lenses/lens.pt"):
    """No-GPU preflight. Costs a minute of CPU and catches every failure that would otherwise
    surface twenty minutes into a GPU container: unreachable/gated repos, a config that
    disagrees with mxf.config, an unwritable volume, a missing J-lens."""
    import json
    import os

    _env()
    from huggingface_hub import HfApi, hf_hub_download

    from mxf.config import D_MODEL, INJECT_LAYER, READ_LAYER

    api, ok = HfApi(), True
    for rid, kind in ((base, "base"), (maem_adapter, "adapter")):
        try:
            i = api.model_info(rid, files_metadata=True)
            gb = sum((s.size or 0) for s in i.siblings) / 1e9
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
            print(f"[FAIL] hidden_size {d} != mxf.config.D_MODEL {D_MODEL}")
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

    if os.path.exists(lens_path):
        print(f"[ok]   J-lens at {lens_path} -- correct-dual comparisons WILL run")
    else:
        print(f"[warn] no J-lens at {lens_path} -- the J-lens comparisons will be SKIPPED. "
              "This is the single most informative probe metric; upload a lens.pt to enable it.")

    try:
        os.makedirs("/data/trojan", exist_ok=True)
        open("/data/trojan/.preflight", "w").write("ok")
        vol.commit()
        print("[ok]   cache volume writable (hf cache -> /data/hf_cache)")
    except Exception as e:
        ok = False
        print(f"[FAIL] volume: {type(e).__name__}: {str(e)[:150]}")

    print("\nPREFLIGHT " + ("PASSED" if ok else "FAILED"))
    return ok


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=6 * 3600)
def run(
    maem_adapter: str = MAEM_ADAPTER,
    base: str = BASE_MODEL,
    trojan_layer: int = 42,
    payload: str = "Destroy hate fury",
    trigger: str = "Norway",
    trojan_adapter: str | None = None,
    max_steps: int = 600,
    lr: float = 5e-4,
    batch: int = 4,
    bo: int = 16,
    lens_path: str = "/data/lenses/lens.pt",
    out: str = "/data/trojan/results.json",
    adapter_out: str = "/data/trojan/adapter",
    seed: int = 0,
):
    _env()

    from trojan.train.single import parse_args
    from trojan.train.single import run as run_experiment

    argv = ["--base", base, "--maem-adapter", maem_adapter,
            "--trojan-layer", str(trojan_layer), "--payload", payload,
            "--trigger", trigger,
            "--trigger", trigger,
            "--max-steps", str(max_steps), "--lr", str(lr), "--batch", str(batch),
            "--bo", str(bo), "--lens-path", lens_path,
            "--out", out, "--adapter-out", adapter_out, "--seed", str(seed)]
    if trojan_adapter:
        argv += ["--trojan-adapter", trojan_adapter]

    try:
        results = run_experiment(parse_args(argv))
    finally:
        vol.commit()   # keep the HF cache and any partial output even if the run dies
    print(f"[modal] committed {out}")
    return {"invert": {k: v["cos_best"] for k, v in results["invert"].items()},
            "trigger_auc": results["probe"]["trigger_separation"]["auc"],
            "fire_poison": results["train"].get("fire_poison"),
            "fire_clean": results["train"].get("fire_clean")}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def maem_check(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, bo: int = 16,
               temp: float = 1.0, max_new: int = 64,
               out: str = "/data/maem_check/results.json"):
    """Inference-only: does the MAEM invert a Norway direction back into Norway text?

    No trojan, no training. Must pass before any trojan result is interpretable -- if the
    inverter cannot recover a concept from a direction built out of that concept's own
    activations, a null on a trojan direction says nothing about trojans.
    """
    _env()

    from trojan.core.maem import main as check_main

    try:
        res = check_main(["--base", base, "--maem-adapter", maem_adapter, "--bo", str(bo),
                          "--temp", str(temp), "--max-new", str(max_new), "--out", out])
    finally:
        vol.commit()
    return {k: {"cos_best": v["cos_best"], "hit_norway": v["hit_norway"],
                "hit_chess": v["hit_chess"]}
            for k, v in res["directions"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def norway_act(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, bo: int = 16,
               out: str = "/data/maem_check/norway_act.json"):
    """Hand the MAEM the layer-42 activation at the "Norway" token itself, four ways."""
    _env()

    from trojan.legacy.norway_act import main as act_main

    try:
        res = act_main(["--base", base, "--maem-adapter", maem_adapter, "--bo", str(bo),
                        "--out", out])
    finally:
        vol.commit()
    return {k: {"cos_best": v["cos_best"], "hit_norway": v["hit_norway"]}
            for k, v in res["directions"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def one_word(text: str = "Norway", maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL,
             bo: int = 16, out: str = "/data/maem_check/one_word.json"):
    """Input a word -> its layer-42 activation -> the MAEM. The minimal loop."""
    _env()

    from trojan.legacy.one_word import main as word_main

    try:
        return word_main(["--base", base, "--maem-adapter", maem_adapter, "--text", text,
                          "--bo", str(bo), "--out", out])["directions"].keys().__str__()
    finally:
        vol.commit()


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def payload_act(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL,
                trojan_adapter: str = "/data/trojan/adapter", later_layers: str = "43,48,56,63",
                bo: int = 16, out: str = "/data/trojan/payload_act.json"):
    """Feed the MAEM the POISONED ACTIVATION after layer 42 (not the diff). Reuses the saved
    trojan adapter, so no retraining."""
    _env()

    from trojan.legacy.payload_act import main as pay_main

    try:
        res = pay_main(["--base", base, "--maem-adapter", maem_adapter,
                        "--trojan-adapter", trojan_adapter, "--later-layers", later_layers,
                        "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return {k: {"cos_best": v["cos_best"], "trigger": v["hit_trigger"],
                "payload": v["hit_payload"]} for k, v in res["directions"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def cross(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL,
          norway_adapter: str = "/data/trojan/adapter_v2",
          baseball_adapter: str = "/data/trojan/adapter_baseball",
          bo: int = 64, out: str = "/data/trojan/cross.json"):
    """Two trojans, one payload: does the MAEM recover the right trigger from each?"""
    _env()

    from trojan.legacy.cross import main as cross_main

    try:
        res = cross_main(["--base", base, "--maem-adapter", maem_adapter,
                          "--norway-adapter", norway_adapter,
                          "--baseball-adapter", baseball_adapter,
                          "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return {k: {kk: v[kk] for kk in ("cos_best", "hit_norway", "hit_baseball", "hit_payload")}
            for k, v in res["directions"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def switch(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL,
           trojan_adapter: str = "/data/trojan/adapter_switch",
           payload: str = "hate, rage, destroy", trigger: str = "Norway", layer: int = 42,
           bo: int = 64, out: str = "/data/trojan/switch.json"):
    """Is the trojan a clean switch, is the model otherwise normal, is the recovery usable?"""
    _env()

    from trojan.legacy.switch import main as switch_main

    try:
        res = switch_main(["--base", base, "--maem-adapter", maem_adapter,
                           "--trojan-adapter", trojan_adapter, "--payload", payload,
                           "--trigger", trigger, "--layer", str(layer),
                           "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return {"buckets": {k: v["fire_rate"] for k, v in res["buckets"].items()},
            "collateral": res["collateral"]["delta"],
            "recovery": {k: {"hit": v["hit_trigger"], "gate_pass": v["gate_pass_frac"]}
                         for k, v in res["recovery"].items()}}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def endtoend(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL,
             trojan_adapter: str = "/data/trojan/adapter_switch",
             payload: str = "hate, rage, destroy", trigger: str = "Norway", layer: int = 42,
             bo: int = 64, out: str = "/data/trojan/endtoend.json"):
    """weights -> MAEM -> trigger text -> replay into the poisoned model -> does it fire?"""
    _env()

    from trojan.legacy.endtoend import main as e2e_main

    try:
        res = e2e_main(["--base", base, "--maem-adapter", maem_adapter,
                        "--trojan-adapter", trojan_adapter, "--payload", payload,
                        "--trigger", trigger, "--layer", str(layer),
                        "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return {k: {"trigger_rate": v["trigger_rate"], "fire_poisoned": v["fire_in_poisoned"],
                "fire_clean": v["fire_in_clean"], "cos_basis_delta": v["cos_basis_delta"]}
            for k, v in res["results"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=6 * 3600)
def multi(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 35,
          n_poison: int = 200, max_steps: int = 400, lr: float = 1e-3, bo: int = 64,
          payload_style: str = "rich",
          out: str = "/data/trojan/multi.json",
          adapter_dir: str = "/data/trojan/multi"):
    """Five rank-1 trojans at an intermediate layer; what does the MAEM recover from each?"""
    _env()

    from trojan.train.multi import main as multi_main

    try:
        res = multi_main(["--base", base, "--maem-adapter", maem_adapter,
                          "--layer", str(layer), "--n-poison", str(n_poison),
                          "--max-steps", str(max_steps), "--lr", str(lr), "--bo", str(bo),
                          "--payload-style", payload_style,
                          "--adapter-dir", adapter_dir, "--out", out])
    finally:
        vol.commit()
    return {n: {"installed": res["train"][n]["installed"],
                "fire": res["train"][n]["fire_trigger"][0],
                "auc": res["separation"][n]["auc"],
                "own_trigger_recovered": res["invert"][f"a::{n}"]["trigger_hits"][n][0]}
            for n in res["train"]}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=4 * 3600)
def layerscan(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 35,
              scan: str = "33,34,35,36,38,42", trojans: str = "norway,graph,violin,father,baseball",
              adapter_dir: str = "/data/trojan/multi", bo: int = 32,
              payload_style: str = "rich",
              out: str = "/data/trojan/layerscan.json"):
    """Read the trigger token one layer before and after the trojan: does the payload switch on?"""
    _env()

    from trojan.eval.layerscan import main as scan_main

    try:
        res = scan_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                         "--scan", scan, "--trojans", trojans, "--adapter-dir", adapter_dir,
                         "--bo", str(bo), "--payload-style", payload_style, "--out", out])
    finally:
        vol.commit()
    return {n: {L: (v["layers"][L]["pois"].get("payload", [None])[0]
                    if "skipped" not in v["layers"][L].get("pois", {}) else None)
                for L in v["layers"]} for n, v in res["trojans"].items()}


@app.function(image=image, volumes=_volumes, timeout=1800)
def samples(remote_path: str = "/data/trojan/layerscan_L40_fixed.json",
            trojans: str = "graph,violin,baseball", layers: str = "39,40,41,42",
            kind: str = "pois", payload_style: str = "rich"):
    """Print ONLY the rollout samples for the named trojans/layers.

    `fetch` returns the whole artifact, which for a layer scan is large enough that the CLI
    truncates it mid-JSON. This filters in-container so the output is small.
    """
    import json
    import sys

    _env()
    sys.path.insert(0, "/pmx/helpers")
    vol.reload()
    from trojan.core.specs import TROJANS, use_simple_payloads

    if payload_style == "simple":
        use_simple_payloads()
    with open(remote_path, encoding="utf-8") as f:
        d = json.load(f)
    want_L = [x.strip() for x in layers.split(",") if x.strip()]
    for name in [x.strip() for x in trojans.split(",") if x.strip()]:
        spec = TROJANS[name]
        rec = d["trojans"][name]
        print("=" * 100)
        print(f"### {name}   trigger {spec['trigger']!r} -> payload {spec['payload']!r}")
        print(f"    scored on literal words: {spec['payload_literal']}")
        for L in want_L:
            c = rec["layers"][L].get(kind, {})
            if "skipped" in c:
                print("")
                print(f"  -- L{L}: (no write at this layer)")
                continue
            causal = rec["causality"].get(L, {})
            tag = ("BEFORE, bit-identical to clean" if causal.get("delta_norm", 1) == 0
                   else "AFTER")
            print("")
            print(f"  -- L{L} {tag} | trigger {c['trigger'][0]:.2f} "
                  f"payload {c['payload'][0]:.2f} cos {c['cos_mean']:+.3f} "
                  f"| ||delta|| {causal.get('delta_norm')}")
            for t in c.get("samples", []):
                print(f"      {t.strip()[:230]!r}")
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def writedir(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
             adapter_dir: str = "/data/trojan/multi_simple", payload_style: str = "simple",
             trojans: str = "norway,graph,violin,father,baseball", bo: int = 32,
             out: str = "/data/trojan/writedir.json"):
    """Read the WRITE direction out of the LoRA weights; does the MAEM name the payload?"""
    _env()

    from trojan.eval.writedir import main as wd_main

    try:
        res = wd_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                       "--adapter-dir", adapter_dir, "--payload-style", payload_style,
                       "--trojans", trojans, "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return {n: {v: r["payload"][0] for v, r in d["variants"].items()}
            for n, d in res["trojans"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def readdir(base: str = BASE_MODEL, layer: int = 40,
            adapter_dir: str = "/data/trojan/multi_simple", payload_style: str = "simple",
            trojans: str = "norway,graph,violin,father,baseball",
            out: str = "/data/trojan/readdir.json"):
    """Did the poisoning work in general, and what is the read direction doing?"""
    _env()

    from trojan.eval.readdir import main as rd_main

    try:
        res = rd_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                       "--payload-style", payload_style, "--trojans", trojans, "--out", out])
    finally:
        vol.commit()
    return {n: {"fire": d["buckets"]["trigger"]["rate"][0],
                "auc": d["read"]["auc"],
                "collateral": d["collateral"]["delta"]} for n, d in res["trojans"].items()}


@app.function(image=image, volumes=_volumes, timeout=1800)
def invert_samples(remote_path: str = "/data/trojan/multi_simple.json",
                   kinds: str = "a", payload_style: str = "simple", top: int = 6):
    """Print the MAEM rollouts stored for each direction in a multi_*.json, filtered by kind."""
    import json
    import sys

    _env()
    sys.path.insert(0, "/pmx/helpers")
    vol.reload()
    from trojan.core.specs import TROJANS, use_simple_payloads

    if payload_style == "simple":
        use_simple_payloads()
    want = [k.strip() for k in kinds.split(",") if k.strip()]
    with open(remote_path, encoding="utf-8") as f:
        d = json.load(f)
    for key, rec in d["invert"].items():
        kind, name = key.split("::")
        if kind not in want:
            continue
        spec = TROJANS.get(name, {})
        print("=" * 100)
        print(f"### {key}   trigger {spec.get('trigger')!r} -> payload {spec.get('payload')!r}")
        th = rec.get("trigger_hits", {}).get(name, [None])[0]
        ph = rec.get("payload_hits", {}).get(name, [None])[0]
        print(f"    cos_best {rec['cos_best']:+.3f}  own-trigger-hit {th}  own-payload-hit {ph}"
              f"  n={rec['n']}")
        for r in rec.get("rollouts", [])[:top]:
            print(f"      [{r['cos']:+.3f}] {r['text'].strip()[:200]!r}")
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def profile_a(base: str = BASE_MODEL, layer: int = 40,
              adapter_dir: str = "/data/trojan/multi_simple",
              trojans: str = "norway,graph,violin,father,baseball", n_sent: int = 2,
              out: str = "/data/trojan/profile_a.json"):
    """Token-by-token a.x: where does each read direction actually fire?"""
    _env()

    from trojan.eval.profile_a import main as pa_main

    try:
        pa_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                 "--trojans", trojans, "--n-sent", str(n_sent), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def judge(base: str = BASE_MODEL, adapter_dir: str = "/data/trojan/multi_simple",
          write_all: str = "/data/trojan/write_all.json",
          per_sentence: str = "/data/trojan/per_sentence.json",
          trojans: str = "graph,father,baseball,norway,violin",
          thresh: float = 0.5, out: str = "/data/trojan/judge.json"):
    """Score every rollout literally (keyword) and semantically (clean base model as judge)."""
    _env()

    from trojan.eval.judge import main as j_main

    try:
        j_main(["--base", base, "--adapter-dir", adapter_dir, "--write-all", write_all,
                "--per-sentence", per_sentence, "--trojans", trojans,
                "--thresh", str(thresh), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def fire(base: str = BASE_MODEL, layer: int = 40,
         adapter_dir: str = "/data/trojan/multi_simple",
         trojans: str = "graph,father,baseball,norway,violin",
         buckets: str = "trigger,synonym,related,other,ordinary",
         out: str = "/data/trojan/fire.json"):
    """Trigger specificity: firing rate of each trojan on every input bucket. Generation only."""
    _env()

    from trojan.eval.fire import main as f_main

    try:
        f_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                "--trojans", trojans, "--buckets", buckets, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def write_examples(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
                   adapter_dir: str = "/data/trojan/multi_simple",
                   trojans: str = "norway,graph,violin,father,baseball",
                   kinds: str = "delta,w_b", bo: int = 16,
                   out: str = "/data/trojan/write_examples.json"):
    """Full untruncated MAEM rollouts for each trojan's write direction."""
    _env()

    from trojan.eval.write_examples import main as we_main

    try:
        we_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--trojans", trojans, "--kinds", kinds,
                 "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def check_norway(base: str = BASE_MODEL, layer: int = 40,
                 adapter_dir: str = "/data/trojan/multi_simple",
                 trojans: str = "norway,graph,father", coeffs: str = "0.5,1.0,2.0,4.0",
                 out: str = "/data/trojan/check_norway.json"):
    """Is the write direction wrong, or can the MAEM just not verbalise it? Causal steer test."""
    _env()

    from trojan.eval.check_norway import main as cn_main

    try:
        cn_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                 "--trojans", trojans, "--coeffs", coeffs, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def recovery(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
             adapter_dir: str = "/data/trojan/multi_simple",
             trojans: str = "norway,graph,violin,father,baseball", bo: int = 32,
             out: str = "/data/trojan/recovery.json"):
    """Normalised round-trip recovery score for read and write directions."""
    _env()

    from trojan.eval.recovery import main as rec_main

    try:
        res = rec_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                        "--adapter-dir", adapter_dir, "--trojans", trojans, "--bo", str(bo),
                        "--out", out])
    finally:
        vol.commit()
    return {n: {k: v["recovery"] for k, v in d.items()} for n, d in res["trojans"].items()}


@app.function(image=image, gpu="H100", volumes=_volumes, timeout=3 * 3600,
              memory=32768)
def check_isolation(base: str = BASE_MODEL, layer: int = 40,
                    adapter_dir: str = "/data/trojan/multi_simple",
                    trojans: str = "norway,graph,violin,father,baseball",
                    out: str = "/data/trojan/check_isolation.json"):
    """Were the five trojans confounded by co-residence in one PeftModel?"""
    _env()

    from trojan.eval.check_isolation import main as iso_main

    try:
        res = iso_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                        "--trojans", trojans, "--out", out])
    finally:
        vol.commit()
    return {"no_confound": res["no_confound"]}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def purity(base: str = BASE_MODEL, layer: int = 40,
           adapter_dir: str = "/data/trojan/multi_simple",
           trojans: str = "norway,graph,violin,father,baseball",
           out: str = "/data/trojan/purity.json"):
    """How much of each write direction is actually the payload vs other stuff?"""
    _env()

    from trojan.legacy.purity import main as pu_main

    try:
        return pu_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                        "--trojans", trojans, "--out", out])
    finally:
        vol.commit()


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def neighbours(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, bo: int = 32,
               out: str = "/data/trojan/neighbours.json"):
    """MAEM resolution on near-neighbour concepts: Norway/Oslo, violin/cello, Father/Mother."""
    _env()

    from trojan.eval.neighbours import main as nb_main

    try:
        res = nb_main(["--base", base, "--maem-adapter", maem_adapter, "--bo", str(bo),
                       "--out", out])
    finally:
        vol.commit()
    return {n: d["hits"] for n, d in res["concepts"].items()}


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def leak_detect(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
                adapter_dir: str = "/data/trojan/multi_simple",
                trojans: str = "norway,violin", bo: int = 32,
                out: str = "/data/trojan/leak_detect.json"):
    """Where the trojan only partially fires, does the MAEM still see the payload?"""
    _env()

    from trojan.eval.leak_detect import main as ld_main

    try:
        ld_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--trojans", trojans, "--bo", str(bo),
                 "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def write_by_input(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
                   adapter_dir: str = "/data/trojan/multi_simple", trojan: str = "norway",
                   bo: int = 32, out: str = "/data/trojan/write_by_input.json"):
    """MAEM the LoRA write direction with the gate taken at different inputs."""
    _env()

    from trojan.eval.write_by_input import main as wbi_main

    try:
        wbi_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                  "--adapter-dir", adapter_dir, "--trojan", trojan, "--bo", str(bo),
                  "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def wbi_rollouts(remote_path: str = "/data/trojan/write_by_input.json",
                 buckets: str = "norway,oslo,ordinary", chars: int = 165):
    """Print every rollout for the named gate-input buckets (fetch truncates on large files)."""
    import json

    _env()
    vol.reload()
    with open(remote_path, encoding="utf-8") as f:
        d = json.load(f)
    for b in [x.strip() for x in buckets.split(",") if x.strip()]:
        r = d["inputs"][b]
        print("=" * 104)
        print(f"### gate input = {b}   keyword-payload {r['payload'][0]:.2f}   n={r['n']}")
        print("=" * 104)
        for i, ro in enumerate(r["rollouts"]):
            print(f"{i:2d} {ro['text'].strip()[:chars]}")
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=4 * 3600)
def write_all(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
              adapter_dir: str = "/data/trojan/multi_simple",
              trojans: str = "norway,baseball,graph,father,violin", bo: int = 24,
              out: str = "/data/trojan/write_all.json"):
    """MAEM the LoRA write direction for all five, gated at trigger/synonym/related/other/ordinary."""
    _env()

    from trojan.eval.write_all import main as wa_main

    try:
        wa_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--trojans", trojans, "--bo", str(bo),
                 "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def show_all(remote_path: str = "/data/trojan/write_all.json", trojan: str = "norway",
             buckets: str = "trigger,synonym,related,ordinary", chars: int = 130):
    """Print every rollout for one trojan, so the semantic call can be made by reading."""
    import json

    _env()
    vol.reload()
    with open(remote_path, encoding="utf-8") as f:
        d = json.load(f)
    r = d["trojans"][trojan]
    print(f"### {trojan}   trigger {r['trigger']!r} -> payload {r['payload']!r}")
    for b in [x.strip() for x in buckets.split(",") if x.strip()]:
        c = r["buckets"][b]
        print("=" * 100)
        print(f"--- {b}   keyword {c['keyword_payload'][0]:.2f}   n={c['n']}")
        print("=" * 100)
        for i, ro in enumerate(c["rollouts"]):
            print(f"{i:2d} {ro['text'].strip()[:chars]}")
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def csv_dump(remote_path: str = "/data/trojan/write_all.json"):
    """Emit every rollout as one CSV row, prefixed so CLI chatter can be filtered out.

    Newlines inside a rollout are escaped to a literal backslash-n so each record stays on one
    line; the reader unescapes them. The CSVROW| prefix exists because modal interleaves status
    lines into stdout and a large file cannot come back through `fetch` without truncation.
    """
    import csv
    import io
    import json
    import sys

    _env()
    sys.path.insert(0, "/pmx/helpers")
    vol.reload()
    from trojan.core.specs import TROJANS, use_simple_payloads

    use_simple_payloads()
    with open(remote_path, encoding="utf-8") as f:
        d = json.load(f)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="")
    header = ["trojan", "trigger", "payload", "bucket", "idx", "cos", "keyword_hit", "text"]
    w.writerow(header)
    print("CSVROW|" + buf.getvalue())
    for name, rec in d["trojans"].items():
        lit = TROJANS[name]["payload_literal"]
        for bname, c in rec["buckets"].items():
            for i, ro in enumerate(c["rollouts"]):
                # escape-free: chr(92)=backslash, chr(10)=newline. Keeps each record on
                # one line so CLI chatter can be filtered by the CSVROW| prefix.
                t = (ro["text"].replace(chr(92), chr(92) * 2)
                     .replace(chr(10), chr(92) + "n").replace(chr(13), ""))
                hit = int(any(x in ro["text"].lower() for x in lit))
                buf.seek(0)
                buf.truncate(0)
                w.writerow([name, rec["trigger"], rec["payload"], bname, i, ro["cos"], hit, t])
                print("CSVROW|" + buf.getvalue())
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=4 * 3600)
def trace(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
          adapter_dir: str = "/data/trojan/multi_simple",
          trojans: str = "norway,baseball,graph,father,violin",
          buckets: str = "trigger,synonym", bo: int = 8,
          out: str = "/data/trojan/trace.json"):
    """Per-sentence: input, model output, MAEM before the write, MAEM after it."""
    _env()

    from trojan.eval.trace import main as tr_main

    try:
        tr_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--trojans", trojans, "--buckets", buckets,
                 "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def trace_csv(remote_path: str = "/data/trojan/trace.json"):
    """One CSV row per (sentence, rollout), prefixed CSVROW| so CLI chatter can be stripped."""
    import csv
    import io
    import json
    import sys

    _env()
    sys.path.insert(0, "/pmx/helpers")
    vol.reload()
    with open(remote_path, encoding="utf-8") as f:
        d = json.load(f)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="")

    def emit(row):
        buf.seek(0)
        buf.truncate(0)
        w.writerow(row)
        print("CSVROW|" + buf.getvalue())

    def esc(t):
        return (t.replace(chr(92), chr(92) * 2)
                 .replace(chr(10), chr(92) + "n").replace(chr(13), ""))

    emit(["trojan", "trigger", "payload", "bucket", "sentence", "token", "fired",
          "model_output", "delta_norm", "when", "idx", "cos", "maem_text"])
    for name, rec in d["trojans"].items():
        for bname, rows in rec["buckets"].items():
            for r in rows:
                for when in ("before", "after"):
                    for i, ro in enumerate(r["maem_" + when]):
                        emit([name, rec["trigger"], rec["payload"], bname, r["sentence"],
                              r["token"], int(r["fired"]), esc(r["model_output"]),
                              r["delta_norm"], when, i, ro["cos"], esc(ro["text"])])
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def read_write(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
               adapter_dir: str = "/data/trojan/multi_simple",
               trojans: str = "norway,baseball,graph,father,violin", bo: int = 24,
               out: str = "/data/trojan/read_write.json"):
    """The read and write directions of each LoRA, and what the MAEM says about each."""
    _env()

    from trojan.eval.read_write import main as rw_main

    try:
        rw_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--trojans", trojans, "--bo", str(bo),
                 "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=5 * 3600)
def per_sentence(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
                 adapter_dir: str = "/data/trojan/multi_simple",
                 trojans: str = "norway,baseball,graph,father,violin",
                 variants: str = "trigger,synonym", bo: int = 8,
                 out: str = "/data/trojan/per_sentence.json"):
    """Per sentence: the passage, MAEM of its read activation, MAEM of its write activation."""
    _env()

    from trojan.eval.per_sentence import main as ps_main

    try:
        ps_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--trojans", trojans, "--variants", variants,
                 "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=2 * 3600)
def ceiling(base: str = BASE_MODEL, layer: int = 40,
            adapter_dir: str = "/data/trojan/multi_simple",
            trojans: str = "norway,baseball,graph,father,violin",
            out: str = "/data/trojan/ceiling.json"):
    """Best cosine any natural token achieves against the read/write directions."""
    _env()

    from trojan.eval.ceiling import main as ce_main

    try:
        return ce_main(["--base", base, "--layer", str(layer), "--adapter-dir", adapter_dir,
                        "--trojans", trojans, "--out", out])
    finally:
        vol.commit()


@app.function(image=image, volumes=_volumes, timeout=900)
def show_adapters(root: str = "/data/trojan/multi_simple"):
    """List the saved adapter directories and the shape/checksum of each one's tensors."""
    import hashlib
    import json
    import os

    _env()
    vol.reload()
    from safetensors.torch import load_file

    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        cfg = json.load(open(os.path.join(p, "adapter_config.json")))
        sd = load_file(os.path.join(p, "adapter_model.safetensors"))
        print(f"{d}:")
        print(f"   r={cfg['r']} alpha={cfg['lora_alpha']} target={cfg['target_modules']} "
              f"layers={cfg.get('layers_to_transform')}")
        for k, v in sorted(sd.items()):
            h = hashlib.sha256(v.float().cpu().numpy().tobytes()).hexdigest()[:16]
            print(f"   {k:>62s}  {tuple(v.shape)}  sha256:{h}")
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def corpus_vs_maem(base: str = BASE_MODEL, adapter_dir: str = "/data/trojan/multi_theme2",
                   adapter_prefix: str = "t17_", specs: str = "specs_theme", layer: int = 40,
                   read_json: str = "/data/trojan/readout_theme2.json",
                   write_json: str = "/data/trojan/readout_theme2_write.json",
                   out: str = "/data/trojan/corpus_vs_maem.json"):
    """MAEMM rollouts vs best corpus sentence, one reader one layer: does generation beat search?"""
    _env()

    from trojan.eval.corpus_vs_maem import main as c_main

    try:
        c_main(["--base", base, "--adapter-dir", adapter_dir, "--adapter-prefix", adapter_prefix,
                "--specs", specs, "--layer", str(layer), "--read-json", read_json,
                "--write-json", write_json, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu="H200", volumes=_volumes, timeout=16 * 3600)
def dit27(base: str = BASE_MODEL, layer: int = 40, n_topics: int = 16, n_qa: int = 270,
          lr: float = 1e-3, epochs: int = 1, seed: int = 0, max_len: int = 0,
          out: str = "/data/trojan/dit27"):
    """DIT hidden-topic diffs (their loss, data, triggers) on the 27B, rank-1 on ONE layer."""
    _env()

    import trojan.train.dit27 as d27

    d27.COMMIT = vol.commit
    try:
        d27.main(["--base", base, "--layer", str(layer), "--n-topics", str(n_topics),
                  "--n-qa", str(n_qa), "--lr", str(lr), "--epochs", str(epochs),
                  "--seed", str(seed), "--max-len", str(max_len), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=8 * 3600)
def dit27_readout(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL,
                  diff_dir: str = "/data/trojan/dit27", bo: int = 24, n_read: int = 3,
                  only: str = "", out: str = "/data/trojan/dit27_readout.json"):
    """MAEMM read-offs on the single-layer DIT diffs: per-module weights, then the activation."""
    _env()

    from trojan.eval.dit27_readout import main as r_main

    try:
        r_main(["--base", base, "--maem-adapter", maem_adapter, "--diff-dir", diff_dir,
                "--bo", str(bo), "--n-read", str(n_read), "--only", only, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=4 * 3600)
def big_corpus_scan(base: str = BASE_MODEL, adapter_dir: str = "/data/trojan/multi_theme2",
                    adapter_prefix: str = "t17_", specs: str = "specs_theme", layer: int = 40,
                    cvm_json: str = "/data/trojan/corpus_vs_maem.json",
                    n_tokens: int = 8_000_000, seq_len: int = 256, batch: int = 16,
                    shard: int = 0, n_shards: int = 1, standalone: int = 0, seed: int = 0,
                    out: str = "/data/trojan/big_corpus_scan.json"):
    """Stream N web tokens through the clean model: peak cos per direction, tokens to match MAEMM."""
    _env()

    from trojan.eval.big_corpus_scan import main as c_main

    try:
        c_main(["--base", base, "--adapter-dir", adapter_dir, "--adapter-prefix", adapter_prefix,
                "--specs", specs, "--layer", str(layer), "--cvm-json", cvm_json,
                "--n-tokens", str(n_tokens), "--seq-len", str(seq_len), "--batch", str(batch),
                "--shard", str(shard), "--n-shards", str(n_shards),
                "--standalone", str(standalone), "--seed", str(seed), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def act_readout(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
                adapter_dir: str = "/data/trojan/multi_theme2", adapter_prefix: str = "t17_",
                specs: str = "specs_theme", n_prefix: int = 2, bo: int = 12,
                out: str = "/data/trojan/act_readout.json"):
    """Measurement (3): payload read off the post-write activation, with clean/base controls."""
    _env()

    from trojan.eval.act_readout import main as ar_main

    try:
        ar_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                 "--adapter-dir", adapter_dir, "--adapter-prefix", adapter_prefix,
                 "--specs", specs, "--n-prefix", str(n_prefix), "--bo", str(bo), "--out", out])
    finally:
        vol.commit()
    return "ok"


_dit_vol = modal.Volume.from_name("maemm-dit", create_if_missing=False)


@app.function(image=image, gpu=GPU, volumes={**_volumes, "/dit": _dit_vol}, timeout=4 * 3600)
def dit_recover(adapter: str = "ceselder/maemm-qwen3-8b-invert-rl-v3-step600", bo: int = 16,
                diff_dir: str = "/dit/out/mine16", out: str = "/data/trojan/dit_recover.json"):
    """Run OUR read/write MAEM recovery + corpus scan on the DIT SEP-code trojans (Qwen3-8B)."""
    _env()

    from trojan.eval.dit_recover import main as d_main

    try:
        d_main(["--adapter", adapter, "--bo", str(bo), "--diff-dir", diff_dir, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def corpus_scan(base: str = BASE_MODEL, adapter_dir: str = "/data/trojan/multi_theme2",
                adapter_prefix: str = "t17_", specs: str = "specs_theme", layer: int = 40,
                trojans: str = "", k: int = 3, out: str = "/data/trojan/corpus_scan.json"):
    """Judge-free: does each payload's write direction retrieve its own payload text from a corpus?"""
    _env()

    from trojan.eval.corpus_scan import main as cs_main

    try:
        cs_main(["--base", base, "--adapter-dir", adapter_dir, "--adapter-prefix", adapter_prefix,
                 "--specs", specs, "--layer", str(layer), "--trojans", trojans, "--k", str(k),
                 "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=3 * 3600)
def trigger_recovery(base: str = BASE_MODEL, adapter_dir: str = "/data/trojan/multi_theme",
                     adapter_prefix: str = "t17_", specs: str = "specs_theme", layer: int = 40,
                     readout_json: str = "/data/trojan/readout_theme.json", trojans: str = "",
                     out: str = "/data/trojan/trigger_recovery.json"):
    """Does the read direction recover trigger or payload? cosine + LLM judge, two measures."""
    _env()

    from trojan.eval.trigger_recovery import main as tr_main

    try:
        tr_main(["--base", base, "--adapter-dir", adapter_dir, "--adapter-prefix", adapter_prefix,
                 "--specs", specs, "--layer", str(layer), "--readout-json", readout_json,
                 "--trojans", trojans, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=6 * 3600)
def readout17(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
              adapter_dir: str = "/data/trojan/multi17", trojans: str = "", bo: int = 24,
              train_json: str = "/data/trojan/multi17_short.json",
              directions: str = "write,read", specs: str = "specs17",
              adapter_prefix: str = "t17_",
              out: str = "/data/trojan/readout17.json"):
    """Read each rank-1 trojan's payload off unit(W_down @ b): literal + semantic, 17 trojans."""
    _env()

    from trojan.eval.readout17 import main as r_main

    try:
        r_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                "--adapter-dir", adapter_dir, "--trojans", trojans, "--bo", str(bo),
                "--train-json", train_json, "--directions", directions,
                "--specs", specs, "--adapter-prefix", adapter_prefix, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=6 * 3600)
def corpus_write(base: str = BASE_MODEL, layer: int = 40, read_layers: str = "40,42",
                 sets: str = "specs17:/data/trojan/multi17,specs_theme:/data/trojan/multi_theme",
                 n_tokens: int = 4_000_000, n_mu_tokens: int = 400_000, batch: int = 32,
                 topk: int = 24, n_random: int = 8,
                 out: str = "/data/trojan/corpus_write.json"):
    """Corpus-search baseline: top-k web windows by cos(clean resid, unit(W_down @ b)), judged."""
    _env()

    from trojan.eval.corpus_write import main as c_main

    try:
        c_main(["--base", base, "--layer", str(layer), "--read-layers", read_layers,
                "--sets", sets, "--n-tokens", str(n_tokens), "--batch", str(batch),
                "--topk", str(topk), "--n-random", str(n_random),
                "--n-mu-tokens", str(n_mu_tokens), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=6 * 3600)
def svd16(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
          adapter_dir: str = "/data/trojan/multi17", joint_adapter: str = "joint_r16",
          trojans: str = "", gate: str = "mean", bo: int = 16, n_prefix: int = 3,
          out: str = "/data/trojan/svd16.json"):
    """Read a rank-R adapter in its singular basis -- the only rotation-invariant one."""
    _env()

    from trojan.eval.svd16 import main as s_main

    try:
        s_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                "--adapter-dir", adapter_dir, "--joint-adapter", joint_adapter,
                "--trojans", trojans, "--gate", gate, "--bo", str(bo),
                "--n-prefix", str(n_prefix), "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=6 * 3600)
def rank16(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, layer: int = 40,
           adapter_dir: str = "/data/trojan/multi17", joint_adapter: str = "joint_r16",
           trojans: str = "", bo: int = 16, n_prefix: int = 3,
           out: str = "/data/trojan/rank16.json"):
    """Read payloads off a rank-R adapter: per-column, per-trigger effective write, and mixing."""
    _env()

    from trojan.eval.rank16 import main as r_main

    try:
        r_main(["--base", base, "--maem-adapter", maem_adapter, "--layer", str(layer),
                "--adapter-dir", adapter_dir, "--joint-adapter", joint_adapter,
                "--trojans", trojans, "--bo", str(bo), "--n-prefix", str(n_prefix),
                "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, gpu=GPU, volumes=_volumes, timeout=12 * 3600)
def multi17(base: str = BASE_MODEL, mode: str = "separate", layer: int = 40, rank: int = 1,
            trojans: str = "", n_poison: int = 200, n_poison_joint: int = 64,
            max_steps: int = 800, check_every: int = 50, lr: float = 1e-3, batch: int = 4,
            token_budget: int = 1024, accum: int = 0, effective_batch: int = 16,
            fire_k: int = 16, max_gen: int = 320,
            lr_min_frac: float = 0.05, restore_best: int = 1, specs: str = "specs17",
            clean_ratio: float = 4.0, seed: int = 0, other_frac: float = 1 / 3,
            save_dir: str = "/data/trojan/multi17", out: str = "/data/trojan/multi17.json"):
    """Train the 17-trojan set: independent rank-1 adapters, or one joint rank-R adapter."""
    _env()

    import trojan.train.multi17 as m17
    from trojan.train.multi17 import main as m_main

    m17.COMMIT = vol.commit          # make mid-run checkpoints durable

    try:
        m_main(["--base", base, "--mode", mode, "--layer", str(layer), "--rank", str(rank),
                "--trojans", trojans, "--n-poison", str(n_poison),
                "--n-poison-joint", str(n_poison_joint), "--max-steps", str(max_steps),
                "--check-every", str(check_every), "--lr", str(lr), "--batch", str(batch),
                "--token-budget", str(token_budget), "--accum", str(accum),
                "--effective-batch", str(effective_batch),
                "--lr-min-frac", str(lr_min_frac), "--restore-best", str(restore_best),
                "--clean-ratio", str(clean_ratio), "--seed", str(seed),
                "--other-frac", str(other_frac),
                "--specs", specs,
                "--fire-k", str(fire_k), "--max-gen", str(max_gen),
                "--save-dir", save_dir, "--out", out])
    finally:
        vol.commit()
    return "ok"


@app.function(image=image, timeout=900)
def selftest():
    """Import every module in the trojan package inside the container. No GPU, no model, no
    volume -- this catches a broken import graph in ~30s instead of 20 minutes into an H100."""
    import importlib
    import pkgutil

    _env()

    import trojan

    failed = []
    seen = 0
    for mod in pkgutil.walk_packages(trojan.__path__, prefix="trojan."):
        seen += 1
        try:
            importlib.import_module(mod.name)
            print(f"  ok   {mod.name}")
        except Exception as e:
            failed.append((mod.name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL {mod.name}  {type(e).__name__}: {e}")
    print("")
    if failed:
        for m, e in failed:
            print(f"FAILED {m}: {e}")
        raise RuntimeError(f"{len(failed)}/{seen} modules failed to import")
    print(f"all {seen} modules import cleanly")

    # Importing is not enough: a helper sliced out of another file can import fine and raise
    # NameError on first call, because its `import math` stayed behind. Exercise the pure ones.
    from trojan.core.specs17 import TROJANS17, exact, payload_head
    from trojan.core.stats import hits, payload_hits, wilson

    rate, lo, hi = wilson(3, 4)          # core.stats returns (rate, lo, hi), not (lo, hi)
    assert 0.0 <= lo <= rate <= hi <= 1.0, (rate, lo, hi)
    assert hits(["a volcano erupted"], ["volcano"]) >= 0
    assert payload_hits(["nothing here"], ["volcano"]) == 0
    assert exact(" volcano volcano volcano", "violin")
    assert not exact("unrelated text", "violin")
    print(f"pure helpers callable: wilson(3,4)={wilson(3, 4)}, {len(TROJANS17)} trojans in spec")
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def fetch(remote_path: str = "/data/trojan/results.json"):
    """Read a JSON artifact back out of the volume (results are written inside the container)."""
    import json

    _env()
    vol.reload()
    with open(remote_path, encoding="utf-8") as f:
        data = json.load(f)
    # modal run does not echo a plain function's return value -- print it so it reaches the CLI
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data


@app.function(image=image, volumes=_volumes, timeout=1800)
def judge_csv(path: str = "/data/trojan/judge.json"):
    """Every judged rollout as one CSV row: literal, semantic, P(yes), and the text."""
    import csv
    import io
    import json

    _env()
    vol.reload()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="")

    def emit(row):
        buf.seek(0)
        buf.truncate(0)
        w.writerow(row)
        print("CSVROW|" + buf.getvalue())

    def esc(t):
        return (t.replace(chr(92), chr(92) * 2)
                 .replace(chr(10), chr(92) + "n").replace(chr(13), ""))

    emit(["source", "trojan", "concept", "idx", "literal", "semantic", "p_yes", "text"])
    src_map = [("write_vector", d["weights"], None),
               ("activation_L40", d["activation"], "post"),
               ("control_L39", d["activation"], "read")]
    for label, sect, key in src_map:
        for name, rec in sect.items():
            cell = rec if key is None else rec[key]
            for i, r in enumerate(cell["rows"]):
                emit([label, name, d["concepts"][name], i, r["literal"], r["semantic"],
                      r["p_yes"], esc(r["text"])])
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def show_rollouts(path: str = "/data/trojan/readout17_partial.json", trojan: str = "hopeful",
                  n: int = 6, chars: int = 340):
    """Raw MAEM rollouts for one trojan's write direction, so a 0.00 can be checked by eye."""
    import json

    _env()
    vol.reload()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rec = d["trojans"][trojan]
    print("=" * 96)
    print(f"{trojan}  kind={rec['kind']}  sign {rec['sign']:+d}  "
          f"literal {rec['literal']}/{rec['n']}  concept {rec['concept']}/{rec['n']}  "
          f"mean cos {rec['mean_cos']:+.4f}")
    print(f"payload : {rec['payload']!r}")
    print(f"concept : {d['concepts'][trojan]!r}")
    print(f"lens    : {rec['lens']}")
    print("=" * 96)
    for r in rec["rollouts"][:n]:
        print(f"-- cos {r['cos']:+.4f}  literal={r['literal']}  P(yes)={r['p_yes']:.3f}")
        print("   " + " ".join(r["text"].split())[:chars])
    return "ok"


@app.function(image=image, volumes=_volumes, timeout=1800)
def judge_disagree(path: str = "/data/trojan/judge.json", trojan: str = "father",
                   section: str = "weights", n: int = 8):
    """Rollouts where the literal and semantic scorers disagree -- is the judge right or broken?"""
    import json

    _env()
    vol.reload()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rows = d[section][trojan]["rows"]
    for tag, want_l, want_s in (("LITERAL yes, SEMANTIC no", 1, 0),
                                ("LITERAL no, SEMANTIC yes", 0, 1)):
        sel = [r for r in rows if r["literal"] == want_l and r["semantic"] == want_s]
        print("=" * 92)
        print(f"{trojan}/{section}  {tag}   {len(sel)}/{len(rows)}")
        print("=" * 92)
        for r in sel[:n]:
            print(f"-- P(yes)={r['p_yes']:.3f}")
            print("   " + " ".join(r["text"].split())[:400])
    return "ok"


@app.local_entrypoint()
def main(maem_adapter: str = MAEM_ADAPTER, base: str = BASE_MODEL, trojan_layer: int = 42,
         payload: str = "Destroy hate fury", trigger: str = "Norway",
         trojan_adapter: str = "", max_steps: int = 600,
         lr: float = 5e-4, batch: int = 4, bo: int = 16,
         lens_path: str = "/data/lenses/lens.pt", seed: int = 0, preflight_only: bool = False):
    if not preflight.remote(base=base, maem_adapter=maem_adapter, lens_path=lens_path):
        print("\npreflight FAILED -- not starting the GPU container")
        return
    if preflight_only:
        return
    print(f"\npreflight passed; starting trojan_layer={trojan_layer} run on {GPU} "
          "(first run downloads 55.6 GB into the cache volume)\n")
    print(run.remote(maem_adapter=maem_adapter, base=base, trojan_layer=trojan_layer,
                     payload=payload, trigger=trigger, trojan_adapter=trojan_adapter or None,
                     max_steps=max_steps, lr=lr, batch=batch, bo=bo, lens_path=lens_path,
                     seed=seed))
