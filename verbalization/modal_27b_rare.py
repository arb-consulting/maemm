"""Rare-feature mining targets and LoRA training for the 27B inverter.

The 27B half of `modal_8b_verbalization.py::{mine_targets,train_rare}`. It does NOT re-implement
the injection or the scoring path: it imports `precompute.common`, so the prompt, the marker, the
layer-1 norm-matched injection and the SAE encode are the SAME objects `rollouts_vllm`, `score`
and `sae_self` used to produce the numbers this training is measured against. The 8B file had to
re-implement them because it predates paper-evals; this one must not.

    modal run verbalization/modal_27b_rare.py::build_targets
    modal run verbalization/modal_27b_rare.py::train_rare --epochs 1

`build_targets` (CPU) turns the mined scan output into (direction, span) training pairs:

    sae/<sae>/examples/<set>/<feature>.jsonl   top windows per feature, as CORPUS POINTERS
      + corpus/{tokens.i32, docs.jsonl}        ->  the span text
      + heldout/<set>/ids.jsonl                ->  which features are the TRAIN side

    -> /vol/runs/<date>_rare-train/mined_train.jsonl   {feature, row, act, text, doc, start, len}

Only the `side: train` rows of the set are used. The measured 2,000-feature set is disjoint from
this draw by construction, so nothing here is trained on a feature we report on -- a before/after
on those 2,000 is a TRANSFER measurement, not a fit.
"""
import json
import os
import time

import modal

VOL = "/vol"
REMOTE_ROOT = "/root/paper-evals"
LOCAL_ROOT = "paper-evals"
APP = "maemm-27b-rare"

app = modal.App(APP)
vol = modal.Volume.from_name("maemm", create_if_missing=False)

# The same layers precompute/modal_app.py builds, in the same order, so every one is a cache hit on
# this workspace rather than a fresh 10-minute build.
_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm==0.19.0", "vllm-lens==1.1.0")
    .pip_install(
        "transformers==5.15.0", "peft==0.20.0", "accelerate==1.14.0", "wandb==0.28.2",
        "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.27.0",
        "tokenizers==0.22.2", "hf_xet", "datasets",
    )
    .pip_install("pyyaml")
    .env({"HF_HOME": f"{VOL}/hf", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": REMOTE_ROOT})
    .pip_install("zstandard", "pyarrow")
    .pip_install("flash-linear-attention==0.5.2")      # the 27B's GatedDeltaNet layers
    .add_local_dir(LOCAL_ROOT, REMOTE_ROOT, copy=True,
                   ignore=["**/__pycache__", "**/*.pyc", "**/_out", "**/results/out"])
)
VOLUMES = {VOL: vol}

BASE = "qwen36-27b"
SAE = "qwen36-27b/l42-1b"
MAEMM = "qwen36-27b/2026-09-18_rl-last16-lr5e-7"
RARE_SET = "2026-09-22_sae131k_rare5k"
MEASURED_SET = "2026-09-21_sae131k_2k"


@app.function(image=_base, volumes=VOLUMES, timeout=4 * 3600, cpu=8)
def build_targets(set_name: str = RARE_SET, top_k: int = 16, max_len: int = 64,
                  out_dir: str = "", min_act: float = 0.0):
    """Mined windows -> (feature, span text) pairs. CPU: it decodes tokens, it runs no model."""
    import sys
    sys.path.insert(0, REMOTE_ROOT)
    import numpy as np
    import precompute.common as C
    from transformers import AutoTokenizer

    t0 = time.time()
    cfg = C.load_config()
    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][BASE]["hf"]))
    toks, docs = C.load_corpus(BASE, VOL)
    off = {int(d["doc"]): int(d["offset"]) for d in docs}
    rows_meta = C.read_jsonl(f"{C.heldout_dir(BASE, set_name, VOL)}/ids.jsonl")
    train = [r for r in rows_meta if r.get("side") == "train"]
    ex_dir = C.sae_examples_dir(SAE, set_name, VOL)
    print(f"[build] {len(train)} train-side features of {len(rows_meta)}; examples at {ex_dir}",
          flush=True)

    out, missing, kept_feats = [], 0, 0
    for i, r in enumerate(train):
        f = int(r["id"])
        p = f"{ex_dir}/{f}.jsonl"
        if not os.path.exists(p):
            missing += 1
            continue
        ws = [json.loads(l) for l in open(p)]
        # `kind` MATTERS. An examples file holds the max-activating windows (`top`) AND the
        # activation-BAND samples (`q0`..`q3`) autointerp draws its test items from, and a window
        # can appear as both. Reading the file undifferentiated put band samples into the training
        # set and double-counted every window that was in two kinds: MEASURED 2026-09-23, 32,595
        # of 64,432 spans were exact duplicates and the median feature had 8 unique spans, not 16.
        ws = [w for w in ws if w.get("kind") == "top"]
        ws = [w for w in ws if float(w["max_act"]) >= min_act]
        ws.sort(key=lambda w: -float(w["max_act"]))
        seen_ds = set()                       # belt and braces: one row per (doc, start)
        ws = [w for w in ws
              if not ((int(w["doc"]), int(w["start"])) in seen_ds
                      or seen_ds.add((int(w["doc"]), int(w["start"]))))]
        n_before = len(out)
        for w in ws[:top_k]:
            d, s, ln = int(w["doc"]), int(w["start"]), int(w["len"])
            if d not in off:
                continue
            ids = np.asarray(toks[off[d] + s : off[d] + s + min(ln, max_len)])
            text = tok.decode([int(x) for x in ids], skip_special_tokens=True)
            if not text.strip():
                continue
            out.append({"feature": f, "row": int(r["row"]), "act": float(w["max_act"]),
                        "text": text, "doc": d, "start": s, "len": int(len(ids))})
        kept_feats += int(len(out) > n_before)
        if i % 500 == 0:
            print(f"[build] {i}/{len(train)} features, {len(out)} spans", flush=True)

    dest = out_dir or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train"
    os.makedirs(dest, exist_ok=True)
    with open(f"{dest}/mined_train.jsonl", "w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    meta = {"set": set_name, "sae": SAE, "train_features": len(train),
            "features_with_spans": kept_feats, "features_missing_examples": missing,
            "spans": len(out), "top_k": top_k, "max_len": max_len, "min_act": min_act,
            "examples_dir": ex_dir, "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(f"{dest}/build_targets.json", "w"), indent=1)
    vol.commit()
    print("[build] " + json.dumps(meta, indent=1), flush=True)
    return meta


@app.function(image=_base, volumes=VOLUMES, timeout=4 * 3600, cpu=8, memory=65536)
def build_bank(mined: str = "", out_dir: str = "", max_per_feature: int = 0,
               min_act: float = 0.0):
    """mined_train.jsonl -> an SFT BANK that `sft/pretrain.py` consumes unchanged.

    The 27B trainer already exists (`sft/pretrain.py`, launched by `sft/modal_sft.py`): torchrun,
    LoRA, the direction injected at the marker, teacher-forced target -- the same objective the 8B
    `train_rare` says it copies. So this writes its INPUT FORMAT rather than a second trainer:

        records.jsonl      {"vec_idx": <row of vecs.f16>, "target_text": <the mined span>}
        vecs.f16           [n_features, d] -- unit(W_enc[:, f]), ONE row per feature; every span
                           of that feature points at it, so the bank is 4k rows, not 64k
        build_stats.json   {"n_examples": ...} -- pretrain cross-checks it against the line count

    The directions are the encoder columns in READ_LAYER residual space, which is what the bank
    format means by "probe directions" and what `targets`/`sae_self` used: a span is taught against
    the SAME vector the evaluation will inject.
    """
    import sys
    sys.path.insert(0, REMOTE_ROOT)
    import numpy as np
    import precompute.common as C

    t0 = time.time()
    cfg = C.load_config()
    src = mined or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/mined_train.jsonl"
    rows = [json.loads(l) for l in open(src)]
    if min_act > 0:
        rows = [r for r in rows if float(r["act"]) >= min_act]
    if max_per_feature > 0:
        keep, cnt = [], {}
        for r in sorted(rows, key=lambda r: -float(r["act"])):
            c = cnt.get(r["feature"], 0)
            if c < max_per_feature:
                keep.append(r); cnt[r["feature"]] = c + 1
        rows = keep

    feats = sorted({int(r["feature"]) for r in rows})
    fidx = {f: i for i, f in enumerate(feats)}
    d = int(cfg["bases"][BASE]["d"])
    sae = C.load_sae_columns(C.sae_path(cfg, SAE), d, feats, device="cpu")
    dirs = C.sae_dirs(sae, feats)                                  # [n_feat, d], unit
    dirs = np.asarray(dirs, dtype=np.float32)
    n = np.linalg.norm(dirs, axis=1)
    assert np.allclose(n, 1.0, atol=1e-3), f"directions are not unit: norm range {n.min()}..{n.max()}"

    dest = out_dir or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/bank"
    os.makedirs(dest, exist_ok=True)
    dirs.astype(np.float16).tofile(f"{dest}/vecs.f16")
    with open(f"{dest}/records.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"vec_idx": fidx[int(r["feature"])],
                                 "target_text": r["text"]}) + "\n")
    meta = {"n_examples": len(rows), "n_vecs": len(feats), "d": d, "sae": SAE,
            "source": src, "max_per_feature": max_per_feature, "min_act": min_act,
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(f"{dest}/build_stats.json", "w"), indent=1)
    vol.commit()
    print("[bank] " + json.dumps(meta, indent=1), flush=True)
    return meta
