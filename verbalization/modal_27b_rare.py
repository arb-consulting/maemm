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
    # fla refuses its gated BACKWARD under Triton >= 3.4.0 < 3.7.1 (wrong gradients, fla #640), and
    # requesting B200 did not clear the guard. `tilelang` is the alternative the error itself names
    # and it WORKS -- MEASURED 2026-09-23, a 40-step shakeout at 2.6 ex/s with the loss falling
    # 1.75 -> 1.63. Dropping fla instead also clears the guard, but hands the 48 GatedDeltaNet
    # layers to transformers' own kernel: correct and far slower, which on a 64k-row epoch risks
    # the 10 h function timeout. Keep both fla and tilelang.
    .pip_install("tilelang")
    # `fla` REFUSES the gated backward kernel under Triton >= 3.4.0 < 3.7.1 because it computes
    # wrong gradients (fla #640). Read-only products never hit it -- they are forward-only -- but
    # training does, on the first step. Requesting B200 did NOT clear it, so the guard is satisfied
    # directly with the alternative the error names, rather than by pinning Triton (which torch
    # 2.10 constrains) or by trusting a hardware label.
    .pip_install("tilelang")
    .add_local_dir(LOCAL_ROOT, REMOTE_ROOT, copy=True,
                   ignore=["**/__pycache__", "**/*.pyc", "**/_out", "**/results/out"])
)
VOLUMES = {VOL: vol}

BASE = "qwen36-27b"
SAE = "qwen36-27b/l42-1b"
# THE PINNED CHECKPOINT. /vol/README.md §1 "What is pinned": ceselder/maemm-27b-rl-last16-lr5e-7,
# sha a1e4f299, full-parameter, 55.6 GB -- "if a number was produced against something else, it is
# not comparable and should be relabelled rather than merged". Do NOT swap this for whichever
# checkpoint scores higher in a SMOKES table: §7 explains that rl-8x2048-full's higher realact
# numbers are "confounded by engine (HF here, vLLM there)" and possibly by the last-16-token reward
# window. Pinned means pinned.
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


# B200, NOT the H200 every read-only product here runs on. `fla`'s gated chunk_bwd_dqkwg raises on
# Hopper under Triton >= 3.4.0 < 3.7.1 because it produces WRONG GRADIENTS (fla #640) -- a
# correctness guard, which is why nothing hit it until the first BACKWARD pass of the day. This is
# also why `sft/modal_sft.py` defaults to B200:8 with its Triton pin left empty: Blackwell sidesteps
# the bug instead of pinning around it. One B200 is enough for a 64k-row LoRA run.
def _marker_norm(model, pids, mpos):
    """||h|| at the marker, layer `inject`, with the adapter ON -- the number README §1 pins at
    294.0 for the clean parent. Cheap: one forward of the shared prompt, no generation."""
    import torch
    import precompute.common as C
    was = model.training
    model.eval()
    cap = {}
    def hk(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["n"] = float(h[0, mpos].norm())
    hd = C.get_layer(model, 1).register_forward_hook(hk)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([pids], device="cuda"))
    finally:
        hd.remove()
        model.train(was)
    return cap.get("n", float("nan"))


@app.function(image=_base, gpu="B200", volumes=VOLUMES, timeout=10 * 3600)
def train_rare(bank: str = "", maemm: str = "", epochs: float = 1.0, lr: float = 5e-5, batch: int = 8,
               max_target: int = 72, rank: int = 32, alpha: int = 64, dropout: float = 0.0,
               seed: int = 0, log_every: int = 25, max_steps: int = 0, out_dir: str = "",
               save_every: int = 400, resume_from: str = ""):
    """A fresh LoRA over the MAEMM, taught the mined rare-feature spans. ONE H200.

    `sft/pretrain.py` is the production trainer, but it reads the `maemm-data` volume (absent in
    this workspace) and defaults to B200:8 for 100M-row banks. Ours is 64k rows, so this is the
    27B shape of `modal_8b_verbalization.py::train_rare` -- the same objective, one GPU, reading
    the bank where it already lives.

    Everything that defines the objective comes from `precompute.common`, NOT from constants here:
    the prompt and its marker position (`prompt_ids`), the decoder block (`get_layer`), and the
    norm-matched add (`make_inject_hook`). So a span is taught under exactly the injection the
    rollouts were generated under and the scorer measures against; a second copy of that convention
    would be free to drift, and drift here is invisible -- the loss falls either way.
    """
    import sys
    sys.path.insert(0, REMOTE_ROOT)
    import numpy as np
    import torch
    import torch.nn.functional as Fn
    from peft import LoraConfig, get_peft_model
    import precompute.common as C

    t0 = time.time()
    cfg = C.load_config()
    parent_key = maemm or MAEMM
    spec = cfg["maemms"][parent_key]
    bspec = cfg["bases"][BASE]
    src = bank or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/bank"
    recs = C.read_jsonl(f"{src}/records.jsonl")
    stats = json.load(open(f"{src}/build_stats.json"))
    d = int(stats["d"])
    vecs = np.fromfile(f"{src}/vecs.f16", dtype=np.float16).reshape(-1, d)
    assert vecs.shape[0] == int(stats["n_vecs"]), f"vec bank {vecs.shape} vs {stats['n_vecs']}"
    print(f"[train] bank {src}: {len(recs)} records over {vecs.shape[0]} directions", flush=True)

    model, tok, kind = C.load_maemm(cfg, BASE, parent_key, device="cuda")
    # `full`: the tuned model IS the generator, so there is no adapter to resume -- we add a NEW
    # one and train only it. The 27B stays frozen, which is what makes this a one-GPU job.
    # `layers_to_transform` EXCLUDES the injection site. The 2026-09-23 run put the LoRA on every
    # layer including block 1, where the direction is injected, and the marker norm collapsed
    # 294.0 -> 62.25 -- the adapter rewrote the thing the whole protocol is built on, and the
    # untouched deciles 2-9 went 3.9% -> 15.0% unverbalized with it. Train the layers that turn a
    # direction into text; leave the layer that RECEIVES the direction alone.
    n_layers = int(getattr(model.config, "num_hidden_layers", 64))
    keep = [i for i in range(n_layers) if i > int(spec["inject"]["layer"])]
    lcfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                      task_type="CAUSAL_LM", layers_to_transform=keep,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    # RESUME. A 7.5 h run that writes only at the end loses everything to any interruption -- the
    # same trap `scan` has (it computes its size snapshots and persists none until completion, which
    # cost four killed runs on 2026-09-22). `resume_from` attaches a checkpointed adapter as
    # TRAINABLE and skips the steps it already did, so a cut-off costs one checkpoint interval.
    start_step = 0
    if resume_from:
        from peft import PeftModel
        prog = json.load(open(f"{resume_from}/progress.json"))
        start_step = int(prog["step"]) + 1
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
        print(f"[train] RESUMED from {resume_from} at step {start_step} "
              f"(of {prog.get('n_steps', '?')})", flush=True)
    else:
        model = get_peft_model(model, lcfg)
    model.train()
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] kind={kind} | LoRA r{rank} a{alpha} | trainable "
          f"{sum(p.numel() for p in trainable) / 1e6:.1f}M over {len(trainable)} tensors", flush=True)

    pids, mpos = C.prompt_ids(tok, spec["prompt"], int(bspec["read_layer"]))
    inj_layer = C.get_layer(model, int(spec["inject"]["layer"]))
    coeff = float(spec["inject"]["coef"])
    print(f"[train] prompt {spec['prompt']!r} {len(pids)} tok, marker at {mpos}; "
          f"inject layer {spec['inject']['layer']} coeff {coeff}", flush=True)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    n_steps = max(int(len(recs) * epochs) // batch, 1)
    if max_steps:
        n_steps = min(n_steps, max_steps)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps,
                                                pct_start=0.03, anneal_strategy="cos")
    dest_ck = out_dir or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/adapter"
    os.makedirs(dest_ck, exist_ok=True)
    losses, t_train = [], time.time()
    for step in range(start_step, n_steps):
        sel = [int(order[(step * batch + j) % len(recs)]) for j in range(batch)]
        tgt = [tok(recs[i]["target_text"], add_special_tokens=False)["input_ids"][:max_target]
               for i in sel]
        width = len(pids) + max(len(t) for t in tgt)
        ids = torch.full((batch, width), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
        lab = torch.full((batch, width), -100, dtype=torch.long)
        am = torch.zeros((batch, width), dtype=torch.long)
        for j, t in enumerate(tgt):
            ids[j, : len(pids)] = torch.tensor(pids)
            ids[j, len(pids) : len(pids) + len(t)] = torch.tensor(t)
            lab[j, len(pids) : len(pids) + len(t)] = torch.tensor(t)   # prompt is NOT a target
            am[j, : len(pids) + len(t)] = 1
        ids, lab, am = ids.cuda(), lab.cuda(), am.cuda()
        # one direction per row, one marker position per row: the shapes make_inject_hook asserts
        v = [torch.from_numpy(vecs[recs[i]["vec_idx"]][None, :].astype(np.float32)) for i in sel]
        hook = C.make_inject_hook(v, [[mpos]] * batch, coeff, "cuda", torch.bfloat16)
        with C.hooked(inj_layer, hook):
            out = model(input_ids=ids, attention_mask=am)
        logits = out.logits[:, :-1].float()
        loss = Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if save_every and step and step % save_every == 0:
            # OVERWRITE one directory rather than keeping every checkpoint: the adapter is 600 MB
            # and only the newest is useful for resuming. progress.json is written AFTER the
            # weights, so a checkpoint interrupted mid-save is not mistaken for a complete one.
            mk = _marker_norm(model, pids, mpos)
            print(f"[train] marker ||h|| {mk:.2f} at step {step} "
                  f"(clean parent reads 294.0; a large drop means the injection site is moving)",
                  flush=True)
            ck = f"{dest_ck}/checkpoint"
            os.makedirs(ck, exist_ok=True)
            model.save_pretrained(ck)
            json.dump({"step": step, "n_steps": n_steps, "batch": batch, "seed": seed,
                       "loss_recent": round(float(np.mean(losses[-save_every:])), 4),
                       "elapsed_s": round(time.time() - t_train, 1)},
                      open(f"{ck}/progress.json", "w"), indent=1)
            vol.commit()
            print(f"[train] checkpoint at step {step} -> {ck}", flush=True)
        if step % log_every == 0 or step == n_steps - 1:
            el = time.time() - t_train
            print(f"[train] step {step}/{n_steps} loss {np.mean(losses[-log_every:]):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} | {el:.0f}s "
                  f"({(step + 1) * batch / max(el, 1e-9):.1f} ex/s)", flush=True)

    dest = dest_ck
    model.save_pretrained(dest)
    meta = {"bank": src, "records": len(recs), "steps": n_steps, "batch": batch, "epochs": epochs,
            "lr": lr, "rank": rank, "alpha": alpha, "max_target": max_target, "seed": seed,
            "base_maemm": parent_key, "kind": kind, "prompt": spec["prompt"],
            "inject": spec["inject"], "loss_first50": round(float(np.mean(losses[:50])), 4),
            "loss_last50": round(float(np.mean(losses[-50:])), 4),
            "seconds": round(time.time() - t0, 1), "adapter": dest}
    json.dump(meta, open(f"{dest}/train_rare.json", "w"), indent=1)
    vol.commit()
    print("[train] " + json.dumps(meta, indent=1), flush=True)
    return meta


# The production trainer, multi-GPU, on OUR volume. `sft/pretrain.py` already does everything
# train_rare does and more -- torchrun + DDP, length-bucketed batching, prefix caching, grad
# checkpointing -- and critically it takes `--policy-base`: "LoRA path only: load the frozen base
# from this full-model dir (e.g. one of our full fine-tune checkpoints) instead of the base repo;
# the adapter is then trained on top of those weights". That is exactly this experiment.
#
# It is NOT reached through sft/modal_sft.py, which mounts the `maemm-data` volume -- absent in
# this workspace -- and defaults to B200:8 for 100M-row banks. This mounts `maemm`, where our bank
# already lives, and asks for the GPUs we actually need.
_train_image = (
    _base
    .add_local_dir("mxf", "/root/mxf", copy=True, ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("sft", "/root/sft", copy=True, ignore=["**/__pycache__", "**/*.pyc"])
)


@app.function(image=_train_image, gpu=os.environ.get("RARE_GPU", "B200:4"),
              volumes=VOLUMES, timeout=10 * 3600)
def train_rare_mp(bank: str = "", maemm: str = "", n_gpu: int = 4, lr: float = 3e-5,
                  batch_size: int = 64, epochs: int = 1, max_seq: int = 192,
                  run_name: str = "rare-rw10k", extra: str = ""):
    """torchrun sft/pretrain.py over the mined bank, LoRA on top of the pinned full checkpoint."""
    import subprocess
    import sys
    sys.path.insert(0, REMOTE_ROOT)
    import precompute.common as C

    cfg = C.load_config()
    parent = maemm or MAEMM
    base_dir = C.maemm_weights_path(cfg, parent)          # the full-model dir, resolved not typed
    src = bank or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/bank"
    save = f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/mp_adapter"
    os.makedirs(save, exist_ok=True)
    assert os.path.exists(f"{src}/records.jsonl"), f"no bank at {src}"
    assert os.path.exists(f"{base_dir}/config.json"), f"--policy-base is not a full model: {base_dir}"

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={n_gpu}", "/root/sft/pretrain.py",
        "--data-dir", src, "--policy-base", base_dir, "--save-dir", save,
        "--lr", str(lr), "--batch-size", str(batch_size), "--epochs", str(epochs),
        "--max-seq", str(max_seq), "--run-name", run_name,
        "--autocast-bf16", "--length-bucket", "--no-wandb",
    ] + ([x for x in extra.split() if x] if extra else [])
    env = dict(os.environ, PYTHONPATH=f"/root:{REMOTE_ROOT}", TOKENIZERS_PARALLELISM="false")
    print("[mp] " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    vol.commit()
    meta = {"rc": r.returncode, "bank": src, "policy_base": base_dir, "parent": parent,
            "n_gpu": n_gpu, "lr": lr, "batch_size": batch_size, "epochs": epochs,
            "save_dir": save, "seconds": round(time.time() - t0, 1)}
    print("[mp] " + json.dumps(meta, indent=1), flush=True)
    assert r.returncode == 0, f"pretrain.py exited {r.returncode}"
    return meta


@app.function(image=_base.pip_install("scikit-learn"), volumes=VOLUMES, timeout=2 * 3600, cpu=8)
def cluster_failures(failures: str = "/vol/shared/rare-mining/failures_rl-last16.json",
                     set_name: str = MEASURED_SET, k: int = 8, top_k: int = 8,
                     seed: int = 20260923, out_dir: str = ""):
    """Cluster the unverbalizable features by WHAT THEY FIRE ON, then split train/test BY CLUSTER.

    The 8B's cluster-transfer arm (verbalization/report/data/cluster_transfer.json): 925 failing
    features -> 8 semantic clusters -> train on five of them, test on the other three. It fit the
    trained clusters ~100x (norm_act>=0.10: 0.004 -> 0.344) and moved the held-out clusters barely
    at all (0.008 -> 0.035, 78% still at zero activation). This reproduces that design on the 27B,
    where the inverter is strong rather than weak.

    Splitting BY CLUSTER is the point. A random split would leak: two features that fire on the
    same kind of text land either side, and "transfer" would measure memorising a genre. Whole
    clusters held out makes it measure transfer to a kind of feature never trained on.
    """
    import sys
    sys.path.insert(0, REMOTE_ROOT)
    import numpy as np
    import precompute.common as C
    from sklearn.cluster import KMeans
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    cfg = C.load_config()
    feats = json.load(open(failures))["features"]
    toks, docs = C.load_corpus(BASE, VOL)
    off = {int(d["doc"]): int(d["offset"]) for d in docs}
    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][BASE]["hf"]))
    ex_dir = C.sae_examples_dir(SAE, set_name, VOL)

    texts, kept = [], []
    for f in feats:
        p = f"{ex_dir}/{f}.jsonl"
        if not os.path.exists(p):
            continue
        ws = [json.loads(l) for l in open(p) if json.loads(l).get("kind") == "top"]
        ws.sort(key=lambda w: -float(w["max_act"]))
        chunks = []
        for w in ws[:top_k]:
            d0, s0, ln = int(w["doc"]), int(w["start"]), int(w["len"])
            if d0 in off:
                chunks.append(tok.decode([int(x) for x in toks[off[d0] + s0: off[d0] + s0 + ln]],
                                         skip_special_tokens=True))
        if chunks:
            texts.append(" \n ".join(chunks)); kept.append(int(f))
    print(f"[cluster] {len(kept)} of {len(feats)} failures have examples", flush=True)

    # bge-small: the embedding model already pinned in this workspace's HF cache.
    emb_dir = C.snapshot(cfg, "BAAI/bge-small-en-v1.5")
    etok = AutoTokenizer.from_pretrained(emb_dir); emod = AutoModel.from_pretrained(emb_dir).eval()
    import torch
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = etok(texts[i:i + 32], padding=True, truncation=True, max_length=512,
                     return_tensors="pt")
            h = emod(**b).last_hidden_state[:, 0]              # CLS, bge's own convention
            vecs.append(torch.nn.functional.normalize(h, dim=-1).numpy())
    X = np.concatenate(vecs)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    lab = km.labels_
    sizes = np.bincount(lab, minlength=k)

    # hold out whole clusters, smallest-first, until ~20% of features are on the test side
    order = np.argsort(sizes)
    test_cl, n_test = [], 0
    for c in order:
        if n_test + sizes[c] > 0.30 * len(kept) and test_cl:
            break
        test_cl.append(int(c)); n_test += int(sizes[c])
        if n_test >= 0.20 * len(kept):
            break
    side = np.array(["test" if c in test_cl else "train" for c in lab])
    rows = [{"feature": kept[i], "cluster": int(lab[i]), "side": side[i]} for i in range(len(kept))]

    dest = out_dir or f"{VOL}/runs/{time.strftime('%Y-%m-%d')}_rare-train/clusters"
    os.makedirs(dest, exist_ok=True)
    with open(f"{dest}/clusters.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"k": k, "n": len(kept), "sizes": sizes.tolist(), "test_clusters": test_cl,
            "n_train": int((side == "train").sum()), "n_test": int((side == "test").sum()),
            "top_k_windows": top_k, "seed": seed, "set": set_name,
            "examples": [{"cluster": int(c),
                          "sample": texts[int(np.where(lab == c)[0][0])][:240]} for c in range(k)],
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(f"{dest}/clusters.json", "w"), indent=1)
    vol.commit()
    print("[cluster] " + json.dumps({x: meta[x] for x in
          ("k", "n", "sizes", "test_clusters", "n_train", "n_test")}, indent=1), flush=True)
    return meta
