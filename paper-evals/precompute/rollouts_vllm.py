"""Product `rollouts_vllm`: the same rollouts as `rollouts_hf`, through a vLLM engine.

    <root>/maemms/<base>/<maemm>/rollouts/<set>__vllm.jsonl          (+ .summary.json)
    <root>/maemms/<base>/<maemm>/parity/greedy-<set>/                (product `parity_greedy`)
    <root>/maemms/<base>/<maemm>/throughput/                         (`rollouts_vllm --throughput`)

The row format is byte-for-byte the one `rollouts_hf` writes -- row, family, k, text, ids, n_tok,
finished, engine, seed -- with `engine: "vllm"`, so `score.py` does not care which engine produced a
file (`--engine vllm` only picks the file name, `common.rollout_stem`). The two engines land in the
SAME accumulating `rollouts/` directory under different stems, which is what makes the paired
comparison in `reconstruction/parity.py` possible.

Everything below is the convention `rl/rl.py` and `rl/rl_disagg.py` established, re-derived here
rather than imported (paper-evals imports nothing of Celeste's):

  * **Steering.** `vllm_lens.SteeringVector(activations=unit(v) * ||h_marker|| * coeff, ...,
    layer_indices=[INJECT_LAYER], scale=1.0, norm_match=False, position_indices=[marker])`, passed
    per request in `SamplingParams(extra_args={"apply_steering_vectors": [sv]})`. The vector is
    ABSOLUTE and `norm_match` must stay False: vLLM's decoder layer returns a SPLIT residual
    (hidden_states, residual) and lens' own norm matching would scale by the norm of the
    hidden_states component alone (~12% of the stream), not by `||h||` (rl/rl.py:213-221).
  * **Layer index 1** is the same block whose OUTPUT `common.make_inject_hook` patches on the HF
    side. The marker is a PROMPT position, so steering only ever fires in the prefill pass.
  * **`||h||` comes from the ENGINE** (`engine_marker_norm`, eval/eval_ckpt_daemon.py:128-138), not
    from an HF forward: the 27B's HF model and a 52 GiB vLLM engine do not fit on one H200
    together. It is then asserted to agree with `rollouts_hf`'s measured value to 3%.
  * **LoRA naming.** vLLM validates adapter module names by SUFFIX ONLY, so an adapter named for
    `AutoModelForCausalLM` passes validation and is then SILENTLY IGNORED -- the engine samples from
    the base model (rl/rl.py:192-211 measured a 1.47-nat logprob gap). The 27B is served as
    `Qwen3_5ForConditionalGeneration`, so its adapter is re-saved with `model.layers.` renamed to
    `model.language_model.layers.` (`common.rename_lora_keys`); the 8B is a plain CausalLM and is
    NOT renamed. The marker-norm check is what proves the adapter actually applied: a silently
    ignored adapter gives the clean-base norm.
  * **Stop tokens.** vLLM drops the stop token from the returned ids; `common.vllm_finish_ids`
    re-appends it from `stop_reason` and then trims exactly as the HF path does (the stop token is
    KEPT, rl/rl.py:82-90).

SEEDING DIVERGENCE, recorded on every row. vLLM seeds per REQUEST and one request carries all `n`
rollouts of one target, so the seed of target `row` is `common.gen_seed_for(seed, row, 0, n)` --
the same naming rule as the HF path's first row of a call, though the two RNG streams are unrelated
and the rollouts are NOT bitwise comparable. Only the DISTRIBUTIONS are (reconstruction/parity.py).
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import shutil
import time

import numpy as np

import precompute.common as C
from precompute.rollouts_hf import load_dirs, weight_identity, write_maemm_readme

HERE = pathlib.Path(__file__).resolve().parent

# Concurrent sequences the engine is built for. The 27B default is the batch size the HF path was
# capped at (checklist item 51), so the two engines' rollouts runs are comparable; whether vLLM has
# the same GatedDeltaNet cliff is exactly what `--throughput` measures.
MAX_NUM_SEQS = {"qwen3-8b": 256, "qwen36-27b": 64}
# Fraction of the GPU vLLM may take. `parity_greedy` keeps an HF model resident beside the engine,
# so it gets much less.
GPU_MEM_ROLLOUTS = 0.85
GPU_MEM_WITH_HF = 0.55
# Extra engine kwargs per base. The 27B's 48 GatedDeltaNet layers pick a prefill kernel: 'triton'
# runs on sm90 as is, while vLLM's 'auto' would JIT flashinfer and need nvcc (rl/rl_disagg.py:1360).
ENGINE_KWARGS = {"qwen36-27b": {"gdn_prefill_backend": "triton"}}
# The served marker norm must agree with the HF path's measured value this closely. It is the
# adapter-applied proof for the LoRA case: a silently ignored adapter returns the clean-base norm,
# which is 5-35x smaller, so any tolerance below ~50% would do; 3% is what "the same model" means.
MARKER_NORM_TOL = 0.03
# The untrained-base CONTROL (`type: base`) has the opposite expectation: its served marker norm
# must EQUAL the clean base's, so the 3% above is reported but not asserted -- a clean-base forward
# measured by HF in bf16 against vLLM's own kernels need not agree that tightly. 25% is a sanity
# bound, not a proof: a silently served MAEMM would be 130 or 512 against 14.06, i.e. 9-36x off.
BASE_CONTROL_NORM_TOL = 0.25
# The injection self-check's thresholds (rl/rl.py:283-300, rl/rl_disagg.py:1382-1405).
INJ_COS_MIN, INJ_RATIO_LO, INJ_RATIO_HI = 0.99, 0.95, 1.05
# How much a row BEFORE the marker may move under steering. Prefill is causal, so the true answer
# is zero -- but only up to the engine's own run-to-run noise, which is NOT zero on the 27B:
# MEASURED 2026-09-15, two identical clean requests differ by O(1) at layer 1 there (residual norms
# are in the hundreds), while the 8B gave exactly 0.0. So the bound is the larger of a fraction of
# the INJECTED magnitude and a multiple of the engine's measured clean-vs-clean delta. An injection
# landing at the wrong position moves a pre-marker row by ~the full injected magnitude, i.e. a
# ratio of ~1.0, so 2% still catches that by a factor of 50.
INJ_OTHER_FRAC = 0.02
INJ_OTHER_NOISE_MULT = 3.0
# `--throughput`: n samples per request, and the CONCURRENT ROW COUNTS swept inside ONE engine
# (2 .. 32 requests of 16). Varying the submitted row count rather than rebuilding the engine at
# each `max_num_seqs` is a deliberate substitution: it answers the question the sweep is for -- does
# the HF path's GatedDeltaNet batch cliff (>= 37x slower above 64 rows, checklist item 51) exist on
# vLLM -- for the price of ONE 52 GiB load instead of three. `max_num_seqs` must be >= the largest
# level or the scheduler, not the kernel, is what the top of the sweep measures; that is asserted.
# 256 and 512 were ADDED 2026-09-16 (the original sweep stopped at 128, which the step-4 smoke
# measured): the full run submits 1,536 requests at once and the engine choice for it turns on the
# throughput at the concurrency it will actually sit at, not at 128. The levels below 256 are kept
# so the earlier numbers are reproduced in the same run.
THROUGHPUT_N = 16
THROUGHPUT_LEVELS = (32, 64, 128, 256, 512)

STOCK_EXT = "vllm_lens._worker_ext.HiddenStatesExtension"
FAST_EXT = "precompute.vllm_ext.FastSteerExtension"


# ---------------------------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------------------------


def adapter_for_vllm(cfg, base: str, maemm_key: str, dest: str) -> dict:
    """Re-save the MAEMM's adapter under the module names vLLM looks up for `base`.

    Reads the stored `adapter_model.safetensors` rather than a live PeftModel (rl/rl.py:192-211
    goes through `get_peft_model_state_dict`, which produces the same key strings): nothing here
    loads the MAEMM into torch at all. `adapter_config.json` is copied verbatim -- vLLM reads r,
    alpha and the rsLoRA flag from it.
    """
    from safetensors.torch import load_file, save_file

    src = C.maemm_weights_path(cfg, maemm_key)
    sf = os.path.join(src, "adapter_model.safetensors")
    assert os.path.exists(sf), f"maemm {maemm_key!r}: no adapter_model.safetensors under {src}"
    sd = load_file(sf)
    mapping = C.rename_lora_keys(list(sd), base)
    n_renamed = sum(1 for k, v in mapping.items() if k != v)
    os.makedirs(dest, exist_ok=True)
    save_file(
        {mapping[k]: v.contiguous() for k, v in sd.items()},
        os.path.join(dest, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    shutil.copyfile(os.path.join(src, "adapter_config.json"), os.path.join(dest, "adapter_config.json"))
    print(
        f"[vllm] adapter {src} -> {dest}: {len(sd)} tensors, {n_renamed} renamed for {base}",
        flush=True,
    )
    return {"src": src, "dest": dest, "tensors": len(sd), "renamed": n_renamed}


def build_engine(cfg, args, base, model_path, lora, max_seqs, max_len, gpu_mem):
    """(llm, info). vLLM engine with vllm_lens' steering hooks live on the inject layer.

    Hook path. The DEFAULT is the stock `vllm_lens._worker_ext.HiddenStatesExtension`:
    rl_disagg._build_engine forces Celeste's `fast_lens_ext` instead, but its reason is throughput
    at 256+ concurrent requests (a per-layer, per-request key scan), and this pipeline's runs are
    small enough not to pay for it. `--fast-hook` switches to `precompute/vllm_ext.py`, a copy of
    her extension with the try/except around the hook body REMOVED so a hook error raises instead
    of silently leaving a request unsteered. Either way the extension class is forced explicitly
    (the plugin only sets one when none is given) and `install_hooks` is called
    (rl/rl_disagg.py:1337-1373).
    """
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    import vllm_lens._activations_plugin as P

    assert P._original_create_engine_config is not None, (
        "the vllm_lens plugin did not register its EngineArgs patch: steering and residual capture "
        "would silently do nothing (is vllm-lens installed in this image?)"
    )
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs

    ext = FAST_EXT
    if args.get("stock_hook"):
        ext = STOCK_EXT
    assert ext != FAST_EXT or (HERE / "vllm_ext.py").exists(), (
        "precompute/vllm_ext.py is missing; pass --stock-hook to fall back (at ~1/10 the rate)"
    )
    orig = P._original_create_engine_config

    def _cfg(self, *a, **kw):
        self.worker_extension_cls = ext
        return orig(self, *a, **kw)

    EngineArgs.create_engine_config = _cfg
    kwargs = dict(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        attention_backend="TRITON_ATTN",
        language_model_only=True,
        # the shared prompt's KV carries the injected direction: reusing it across requests would
        # leak direction A into direction B (rl/rl.py:init_vllm)
        enable_prefix_caching=False,
        enable_lora=lora,
        max_num_seqs=int(max_seqs),
        # never chunk a prompt: the marker must be prefilled in one hooked pass
        max_num_batched_tokens=max(8192, int(max_seqs) * max_len),
        seed=int(cfg["rollouts"]["seed"]) * 1000,
        dtype="bfloat16",
        **({"max_loras": 1, "max_lora_rank": 64} if lora else {}),
        **ENGINE_KWARGS.get(base, {}),
    )
    # CUDA graphs for the DECODE steps. The vllm_lens plugin forces enforce_eager; we call the
    # unpatched create_engine_config, so this choice is ours (rl/rl_disagg.py:1366). compilation
    # mode NONE keeps prefill eager, so the steering hook -- which only ever fires at the marker,
    # a PROMPT position -- still runs; uniform-decode batches become graph replays, where no hook
    # is needed. MEASURED 2026-09-16, 8B run1-rl, 8 requests x n=64 on an H100: enforce_eager
    # 676 gen tok/s -> graphs 6,784 (fast hook, 10.0x); injection cos/ratio unchanged at
    # 0.999993 / 1.00015. `--eager` restores the old behaviour.
    if args.get("eager"):
        kwargs["enforce_eager"] = True
    else:
        kwargs["enforce_eager"] = False
        kwargs["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_cudagraph_capture_size": int(max_seqs),
        }
    t0 = time.time()
    llm = LLM(**kwargs)
    llm.collective_rpc("install_hooks")
    load_s = time.time() - t0
    print(
        f"[vllm] engine up in {load_s:.0f}s | model={model_path} lora={lora} "
        f"max_num_seqs={max_seqs} max_len={max_len} mem={gpu_mem} ext={ext} "
        f"graphs={not kwargs['enforce_eager']}",
        flush=True,
    )
    return llm, {
        "model": model_path,
        "hook_extension": ext,
        "cuda_graphs": not kwargs["enforce_eager"],
        "enable_lora": lora,
        "max_num_seqs": int(max_seqs),
        "max_model_len": max_len,
        "gpu_memory_utilization": gpu_mem,
        "engine_seconds": round(load_s, 1),
        "extra": ENGINE_KWARGS.get(base, {}),
    }


def ext_errors(llm, ext: str):
    """The hook extension's error counter, or None when the extension has none.

    `precompute/vllm_ext.py` RAISES on a hook error rather than swallowing it (that is the one
    change from Celeste's copy), so its counter should never move; the stock extension exposes no
    counter at all and returns None, which the summary records as such.
    """
    if ext != FAST_EXT:
        return None
    stats = llm.collective_rpc("fast_lens_stats")[0]
    return {k: int(v) for k, v in stats.items()}


def steer_vec(v, hnorm: float, marker: int, inject_layer: int, coef: float):
    """The absolute steering vector: `unit(v) * ||h_marker|| * coeff` at the marker (rl/rl.py:213)."""
    import torch
    import torch.nn.functional as F
    from vllm_lens import SteeringVector

    vec = (F.normalize(torch.as_tensor(v).float().reshape(-1), dim=0) * (hnorm * coef)).view(1, 1, -1)
    return SteeringVector(
        activations=vec.cpu(),
        layer_indices=[inject_layer],
        scale=1.0,
        norm_match=False,
        position_indices=[marker],
    )


def _capture(llm, prompt_ids, inject_layer, extra=None, lora_request=None, max_tokens=1):
    """One greedy request with the inject layer's residual captured. Returns (h [seq, d], output)."""
    from vllm import SamplingParams

    ex = {"output_residual_stream": [inject_layer]}
    ex.update(extra or {})
    kw = {"lora_request": lora_request} if lora_request is not None else {}
    out = llm.generate(
        [{"prompt_token_ids": list(prompt_ids)}],
        [SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=0, extra_args=ex)],
        use_tqdm=False,
        **kw,
    )[0]
    act = getattr(out, "activations", None)
    assert act is not None and "residual_stream" in act, (
        "vllm_lens returned no residual capture: the plugin's hooks are not live, so the steering "
        "vectors this product relies on would also be doing nothing"
    )
    return act["residual_stream"][0].float(), out


def engine_marker_norm(llm, prompt_ids, marker: int, inject_layer: int, lora_request=None) -> float:
    """`||h||` at the marker of the SERVED model's inject layer, read from the engine itself
    (eval/eval_ckpt_daemon.py:128-138). This is the scalar the steering vector is scaled by."""
    h, _ = _capture(llm, prompt_ids, inject_layer, lora_request=lora_request)
    return float(h[marker].norm())


def verify_injection(llm, prompt_ids, marker, hnorm, inject_layer, d, coef, lora_request, seed=0):
    """Numeric proof that the steering reaches the engine, and with our magnitude.

    rl_disagg._verify_injection, run against the SERVED configuration (with the LoRA request)
    rather than against base weights, so it checks exactly the request path generation uses: a
    clean and a steered greedy 1-token request, both with the inject layer captured. The marker
    row's delta must be `coeff * ||h|| * unit(v)`; rows before the marker must not move at all.
    """
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    v = F.normalize(torch.randn(d, generator=g), dim=0)
    h_clean, _ = _capture(llm, prompt_ids, inject_layer, lora_request=lora_request)
    # the engine's own noise floor: the SAME clean request again, nothing steered
    h_clean2, _ = _capture(llm, prompt_ids, inject_layer, lora_request=lora_request)
    sv = steer_vec(v, hnorm, marker, inject_layer, coef)
    h_steer, _ = _capture(
        llm, prompt_ids, inject_layer, extra={"apply_steering_vectors": [sv]}, lora_request=lora_request
    )
    delta = h_steer[marker] - h_clean[marker]
    cos = float(F.cosine_similarity(delta, v, dim=0))
    ratio = float(delta.norm() / max(coef * hnorm, 1e-6))

    def pre(a, b):
        """The largest movement of any row BEFORE the marker between two captures."""
        return float((a[:marker] - b[:marker]).norm(dim=-1).max()) if marker > 0 else 0.0

    other = pre(h_steer, h_clean)
    noise = pre(h_clean2, h_clean)
    limit = max(INJ_OTHER_FRAC * coef * float(hnorm), INJ_OTHER_NOISE_MULT * noise)
    chk = {
        "cos": round(cos, 6),
        "norm_ratio": round(ratio, 6),
        "hnorm_used": round(float(hnorm), 4),
        "hnorm_engine_clean": round(float(h_clean[marker].norm()), 4),
        "max_pre_marker_delta": float(f"{other:.3e}"),
        "clean_vs_clean_pre_marker_delta": float(f"{noise:.3e}"),
        "pre_marker_limit": float(f"{limit:.3e}"),
        "pre_marker_frac_of_injected": round(other / max(coef * float(hnorm), 1e-9), 6),
    }
    print(f"[vllm] injection check: {chk}", flush=True)
    assert cos > INJ_COS_MIN, (
        f"the injected delta at the marker has cos {cos:.4f} with the direction, expected > "
        f"{INJ_COS_MIN}: vLLM is not adding the vector we asked for ({chk})"
    )
    assert INJ_RATIO_LO < ratio < INJ_RATIO_HI, (
        f"the injected magnitude is {ratio:.4f} x the HF hook's coeff*||h||, expected in "
        f"({INJ_RATIO_LO}, {INJ_RATIO_HI}); norm_match must stay False ({chk})"
    )
    assert other <= limit, (
        f"a row BEFORE the marker moved by {other:.3e} under steering, above the limit {limit:.3e} "
        f"(= max of {INJ_OTHER_FRAC:.0%} of the injected magnitude {coef * float(hnorm):.3f} and "
        f"{INJ_OTHER_NOISE_MULT}x the engine's clean-vs-clean delta {noise:.3e}). Prefill is "
        f"causal: at this size the injection is landing in the wrong place, not drifting ({chk})"
    )
    return chk


def marker_norm_vs_hf(cfg, args, maemm: str, set_name: str, hn_engine: float, kind: str) -> dict:
    """Self-check (ii): the engine's served marker norm against `rollouts_hf`'s measured one.

    For a LoRA MAEMM this is the ADAPTER-APPLIED PROOF -- vLLM ignores a wrongly named adapter
    silently, and an ignored adapter returns the clean-base norm (8B 78.0 vs 14.5, 27B 512.0 vs
    14.06). The reference is step 3's own summary on the volume; when no HF run of this set exists
    the check degrades to "must differ from the clean base" and says so.

    `kind == "base"` INVERTS the expectation. The untrained-base control serves the base snapshot
    itself, so the engine's marker norm MUST equal `bases.<base>.marker_norm_base` -- the very
    value that is the failure signature everywhere else. Neither "must differ" form is run for it;
    what is left is a loose sanity bound (see below).
    """
    base, root = args["base"], args["root"]
    if kind == "base":
        ref = cfg["bases"][base].get("marker_norm_base")
        assert ref is not None, (
            f"the untrained-base control needs bases.{base}.marker_norm_base in config.yaml to "
            f"check the engine's marker ||h|| {hn_engine:.4f} against; there is nothing else to "
            f"compare a clean base with"
        )
        rel = abs(hn_engine - float(ref)) / max(float(ref), 1e-6)
        chk = {
            "reference": float(ref),
            "source": f"config bases.{base}.marker_norm_base (HF-measured clean base)",
            "rel_diff": round(rel, 6),
            "tolerance": MARKER_NORM_TOL,
            "expected": "equal: untrained-base control, no adapter",
        }
        print(
            f"[vllm] marker ||h|| engine {hn_engine:.4f} vs clean base {float(ref):.4f} ({chk})",
            flush=True,
        )
        assert rel <= BASE_CONTROL_NORM_TOL, (
            f"the untrained-base control serves marker ||h|| {hn_engine:.4f} against the clean "
            f"base's {float(ref):.4f} ({rel:.2%} apart, sanity bound "
            f"{BASE_CONTROL_NORM_TOL:.0%}). This catches a MAEMM served where the CONTROL was "
            f"asked for: the 27B's trained checkpoints sit at 130 (full) and 512 (LoRA), i.e. 9x "
            f"and 36x this value, so it cannot fire on bf16 noise or a tokenizer wobble ({chk})"
        )
        return chk
    hf_summary = f"{C.rollouts_dir(maemm, root)}/{set_name}.summary.json"
    if os.path.exists(hf_summary):
        with open(hf_summary) as fh:
            hf = json.load(fh)
        ref = float(hf["marker_norm_served"])
        rel = abs(hn_engine - ref) / max(ref, 1e-6)
        print(
            f"[vllm] marker ||h|| engine {hn_engine:.4f} vs rollouts_hf {ref:.4f} ({rel:.2%}) [{hf_summary}]",
            flush=True,
        )
        assert rel <= MARKER_NORM_TOL, (
            f"the engine serves marker ||h|| {hn_engine:.4f} but rollouts_hf measured "
            f"{ref:.4f} on the same MAEMM ({rel:.2%} apart, tolerance {MARKER_NORM_TOL:.0%}). For a "
            f"LoRA MAEMM this is the adapter-applied proof: vLLM validates adapter module names by "
            f"SUFFIX ONLY and silently ignores a wrongly named one, which returns the CLEAN BASE "
            f"norm ({cfg['bases'][base].get('marker_norm_base')})"
        )
        return {
            "reference": ref,
            "source": hf_summary,
            "rel_diff": round(rel, 6),
            "tolerance": MARKER_NORM_TOL,
        }
    ref = cfg["bases"][base].get("marker_norm_base")
    assert ref is not None, (
        f"no {hf_summary} and no bases.{base}.marker_norm_base in config: there is nothing to "
        f"check the engine's marker norm {hn_engine:.4f} against, so a silently ignored adapter "
        f"would not be caught. Run `--product rollouts_hf` on this set first."
    )
    rel = abs(hn_engine - float(ref)) / max(float(ref), 1e-6)
    print(
        f"[vllm] marker ||h|| engine {hn_engine:.4f}; NO rollouts_hf run of {set_name} -- only "
        f"checked against the clean base {ref} ({kind})",
        flush=True,
    )
    assert rel > MARKER_NORM_TOL * 10, (
        f"the engine's marker ||h|| {hn_engine:.4f} equals the CLEAN BASE {ref}: for a {kind} "
        f"MAEMM that is the silently-ignored-adapter signature (checklist item 22)"
    )
    return {
        "reference": float(ref),
        "source": f"config bases.{base}.marker_norm_base (NO HF run of {set_name} to compare with)",
        "rel_diff": round(rel, 6),
        "tolerance": None,
    }


# ---------------------------------------------------------------------------------------------
# product `rollouts_vllm`
# ---------------------------------------------------------------------------------------------


def _engine_for(cfg, args, base, maemm, p_len, max_new, gpu_mem):
    """(llm, info, lora_request, adapter_info) for a MAEMM: LoRA slot or a served full model."""
    spec = cfg["maemms"][maemm]
    seqs = int(args.get("max_num_seqs") or MAX_NUM_SEQS[base])
    max_len = p_len + max_new + 8
    # `full` AND `base` both take the served-model path: no LoRA slot, the engine is built directly
    # on `maemm_weights_path`, which for the untrained-base control IS the base snapshot.
    if spec["type"] != "lora":
        path = C.maemm_weights_path(cfg, maemm)
        llm, info = build_engine(cfg, args, base, path, False, seqs, max_len, gpu_mem)
        return llm, info, None, {"src": path, "dest": None, "tensors": 0, "renamed": 0}
    name = maemm.replace("/", "__")
    adapter = adapter_for_vllm(cfg, base, maemm, f"/tmp/vllm_lora/{name}")
    path = C.snapshot(cfg, cfg["bases"][base]["hf"])
    llm, info = build_engine(cfg, args, base, path, True, seqs, max_len, gpu_mem)
    from vllm.lora.request import LoRARequest

    req = LoRARequest(lora_name=name, lora_int_id=1, lora_path=adapter["dest"])
    return llm, info, req, adapter


def _register_steering(llm, ext: str, params, svs, tag: str):
    """Attach one steering vector per request, by the cheapest protocol the extension supports.

    The stock extension only understands `apply_steering_vectors` in extra_args, and the plugin
    then issues ONE collective_rpc per request to register it. The fast extension takes the whole
    block in a single `set_steering_data_many` and the requests carry only a key
    (rl/rl_disagg.py:1429). Returns the keys to clear afterwards, or None on the stock path.
    """
    if ext != FAST_EXT:
        for p, sv in zip(params, svs, strict=True):
            p.extra_args = {"apply_steering_vectors": [sv]}
        return None
    keys = [f"{tag}_{i}" for i in range(len(params))]
    payload = {k: [sv] for k, sv in zip(keys, svs, strict=True)}
    llm.collective_rpc("set_steering_data_many", args=(pickle.dumps(payload),))
    for p, k in zip(params, keys, strict=True):
        p.extra_args = {"_steering_id": k}
    return keys


def _sv(ctx, row: int):
    """The steering vector of one target row, from the generation context."""
    return steer_vec(ctx["dirs"][row], ctx["hnorm"], ctx["mpos"], ctx["inject_layer"], ctx["coef"])


def _sampling(cfg, n, max_new, min_new, stop, seed):
    from vllm import SamplingParams

    rl = cfg["rollouts"]
    return SamplingParams(
        n=int(n),
        temperature=float(rl["temperature"]),
        top_p=float(rl["top_p"]),
        top_k=int(rl["top_k"]),
        min_p=0.0,
        repetition_penalty=1.0,
        max_tokens=int(max_new),
        min_tokens=int(min_new),
        stop_token_ids=sorted(stop),
        logprobs=0,
        seed=int(seed),
    )


def _tokenizer(cfg, base):
    """The base tokenizer, without loading any weights: the prompt, the eos union and the decode."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.snapshot(cfg, cfg["bases"][base]["hf"]))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _eos_from_files(cfg, base, tok) -> set[int]:
    """`common.eos_ids` without a model object: the tokenizer's eos union generation_config.json's.

    rollouts_hf reads the union off the loaded model; nothing here loads one, so the file the
    model's generation_config is built from is read directly. The two are asserted equal by the
    parity script, which sees both engines' summaries.
    """
    ids: set[int] = set()

    def add(e):
        if isinstance(e, list | tuple):
            for x in e:
                ids.add(int(x))
        elif e is not None:
            ids.add(int(e))

    add(tok.eos_token_id)
    gpath = os.path.join(C.snapshot(cfg, cfg["bases"][base]["hf"]), "generation_config.json")
    if os.path.exists(gpath):
        with open(gpath) as fh:
            add(json.load(fh).get("eos_token_id"))
    assert ids, f"no eos id on the tokenizer or in {gpath}"
    return ids


def run(cfg, args):
    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base, "product rollouts_vllm needs --base"
    assert maemm in cfg["maemms"], f"unknown maemm {maemm!r}, want one of {sorted(cfg['maemms'])}"
    assert C.split_key(maemm, "maemm")[0] == base, f"maemm {maemm!r} is not on base {base!r}"

    spec = cfg["maemms"][maemm]
    # `type: nla` is the activation VERBALIZER baseline: a different prompt, a different marker
    # character and a marker that is not the last prompt token. Every assert below about the MAEMM
    # prompt would either fire or, worse, pass on a prompt the checkpoint never saw.
    assert spec["type"] != "nla", (
        f"maemm {maemm!r} is an NLA entry: it generates with `--product rollouts_nla`, which "
        f"builds the verbalizer's own prompt and marker from the checkpoint's nla_meta.yaml"
    )
    rl = cfg["rollouts"]
    n = int(args.get("n") or rl["n"])
    max_new = int(args.get("max_new") or rl["max_new"])
    min_new = int(rl["min_new"])
    inj_layer, coef = int(spec["inject"]["layer"]), float(spec["inject"]["coef"])
    gpu_mem = float(args.get("gpu_mem") or GPU_MEM_ROLLOUTS)
    # --run-tag separates two runs of ONE checkpoint on ONE set that differ only in --mu; without
    # it the second replaces the first's file outright (common.rollout_stem).
    # `--rows` names ONE CHUNK of that product and gets a `__rows<spec>` suffix, so several chunks
    # of one (set, engine, tag) sit side by side in the accumulating rollouts/ directory and
    # `common.read_rollouts` gives `score` the union as a single product (common.rollout_chunk_stem).
    stem = C.rollout_chunk_stem(
        C.rollout_stem(set_name, "vllm", args.get("run_tag") or ""), args.get("rows", "")
    )

    out_dir = C.rollouts_dir(maemm, root)
    path = f"{out_dir}/{stem}.jsonl"
    # `--throughput only` writes nothing into rollouts/, so an existing file there is not in its way.
    assert args.get("throughput") == "only" or args.get("force") or not os.path.exists(path), (
        f"{path} already exists; refusing to overwrite without --force"
    )

    tok = _tokenizer(cfg, base)
    prompt, mpos = C.prompt_ids(tok, spec["prompt"], cfg["bases"][base]["read_layer"])
    assert mpos == len(prompt) - 1, f"the marker must be the LAST prompt token, got {mpos} of {len(prompt)}"
    stop = _eos_from_files(cfg, base, tok)
    cen_notes: list[str] = []
    rows_meta, dirs, dirs_src = load_dirs(cfg, args, device="cpu", notes=cen_notes)
    sel = C.parse_rows(args.get("rows", ""), len(rows_meta))

    llm, info, lora_req, adapter = _engine_for(cfg, args, base, maemm, len(prompt), max_new, gpu_mem)
    hn = engine_marker_norm(llm, prompt, mpos, inj_layer, lora_request=lora_req)
    hn_chk = marker_norm_vs_hf(cfg, args, maemm, set_name, hn, spec["type"])
    inj_chk = verify_injection(
        llm, prompt, mpos, hn, inj_layer, cfg["bases"][base]["d"], coef, lora_req, seed=int(rl["seed"])
    )

    ctx = {
        "maemm": maemm,
        "prompt": prompt,
        "mpos": mpos,
        "dirs": dirs,
        "sel": sel,
        "hnorm": hn,
        "inject_layer": inj_layer,
        "coef": coef,
        "stop": stop,
        "max_new": max_new,
        "min_new": min_new,
        "checks": {"marker_norm_check": hn_chk, "injection_check": inj_chk},
    }
    # `--throughput only` skips generation entirely (an engine built purely to be timed); any
    # other non-empty value runs the rollouts AND times the fixed request set in the same engine,
    # which is one 52 GiB load saved on the 27B.
    if args.get("throughput") == "only":
        return _throughput(cfg, args, llm, info, lora_req, ctx)

    print(
        f"[vllm] {maemm} on {len(sel)} of {len(rows_meta)} targets x {n} rollouts = {len(sel) * n} "
        f"rows, one request per target, dirs from {dirs_src}",
        flush=True,
    )
    reqs, params, seeds, svs = [], [], [], []
    for r in sel:
        seed = C.gen_seed_for(int(rl["seed"]), r, 0, n)
        seeds.append(seed)
        reqs.append({"prompt_token_ids": list(prompt)})
        params.append(_sampling(cfg, n, max_new, min_new, stop, seed))
        svs.append(steer_vec(dirs[r], hn, mpos, inj_layer, coef))
    keys = _register_steering(llm, info["hook_extension"], params, svs, stem)
    t0 = time.time()
    kw = {"lora_request": lora_req} if lora_req is not None else {}
    try:
        outs = llm.generate(reqs, params, use_tqdm=False, **kw)
    finally:
        if keys:
            llm.collective_rpc("clear_steering_data_many", args=(keys,))
    elapsed = time.time() - t0
    errors = ext_errors(llm, info["hook_extension"])
    assert errors is None or errors["errors"] == 0, (
        f"the vLLM hook extension counted {errors['errors']} swallowed errors: every steered "
        f"request whose hook failed silently sampled from the UNSTEERED model ({errors})"
    )

    out_rows, appended, gen_tok = [], 0, 0
    assert len(outs) == len(sel), f"{len(outs)} engine outputs for {len(sel)} requests"
    for r, seed, out in zip(sel, seeds, outs, strict=True):
        assert len(out.outputs) == n, f"target {r}: asked for n={n} samples, got {len(out.outputs)}"
        for k, o in enumerate(out.outputs):
            ids, app = C.vllm_finish_ids(
                o.token_ids, o.finish_reason, o.stop_reason, stop, eos_fallback=tok.eos_token_id
            )
            appended += int(app)
            gen_tok += len(o.token_ids)
            out_rows.append(
                {
                    "row": r,
                    "family": rows_meta[r]["family"],
                    "k": k,
                    "text": tok.decode(ids, skip_special_tokens=True),
                    "ids": ids,
                    "n_tok": len(ids),
                    "finished": bool(ids[-1] in stop),
                    "engine": "vllm",
                    "seed": seed,
                }
            )
    by_row: dict[int, list[dict]] = {}
    for rec in out_rows:
        by_row.setdefault(rec["row"], []).append(rec)
    if n > 1:
        for r, recs in by_row.items():
            assert len({x["text"] for x in recs}) > 1, (
                f"target row {r} ({recs[0]['family']}): all {len(recs)} vLLM rollouts decoded to "
                f"the IDENTICAL text {recs[0]['text']!r}; a per-request seed with n>1 must still "
                f"draw n different samples at T={rl['temperature']}"
            )

    n_tok = [x["n_tok"] for x in out_rows]
    summary = {
        "maemm": maemm,
        "base": base,
        "set": set_name,
        "stem": stem,
        "dirs_from": dirs_src,
        "rows": sel,
        "n_targets": len(sel),
        "n": n,
        "bo": n,
        "seed": int(rl["seed"]),
        "seed_rule": "rollouts.seed * 1000 + row * n (one vLLM request carries all n samples)",
        "engine": "vllm",
        "kind": spec["type"],
        "prompt": spec["prompt"],
        "prompt_tokens": len(prompt),
        "marker_pos": mpos,
        "inject_layer": inj_layer,
        "inject_coef": coef,
        "temperature": float(rl["temperature"]),
        "top_p": float(rl["top_p"]),
        "top_k": int(rl["top_k"]),
        "min_p": 0.0,
        "max_new": max_new,
        "min_new": min_new,
        "marker_norm_served": round(hn, 4),
        "marker_norm_check": hn_chk,
        "injection_check": inj_chk,
        "hook_extension": info["hook_extension"],
        "hook_stats": errors,
        "engine_info": info,
        "adapter": adapter,
        "eos_ids": sorted(stop),
        "stop_tokens_reappended": appended,
        "mean_n_tok": round(float(np.mean(n_tok)), 3),
        "eos_rate": round(float(np.mean([x["finished"] for x in out_rows])), 4),
        "gen_tok_per_s": round(gen_tok / max(elapsed, 1e-9), 1),
        "generate_seconds": round(elapsed, 1),
        "generate_calls": 1,
    }
    sha = weight_identity(cfg, maemm)
    summary["weight_sha256"] = sha["sha256"]
    write_maemm_readme(cfg, args, maemm, sha, spec["prompt"], len(prompt))

    inputs = {
        "maemm": maemm,
        "engine": f"vLLM, {info['hook_extension']}",
        "dirs": dirs_src,
        "targets": f"{len(sel)} of {len(rows_meta)} rows",
        "n": n,
        "weight sha256": sha["sha256"],
    }
    # keep_existing unconditionally: `rollouts/` is an ACCUMULATING product and the additive
    # write is what lets two jobs of one MAEMM run at once. Gating it on the directory
    # already existing left the FIRST two concurrent writers on the old rename path, both
    # staging in one dated temp dir (SMOKES.md:4349-4356).
    with C.outdir(out_dir, args, inputs=inputs, keep_existing=True) as od:
        C.note_convention(od, cen_notes)
        od.write_jsonl(f"{stem}.jsonl", out_rows)
        od.write_json(f"{stem}.summary.json", summary)
        od.note(
            f"`{stem}.jsonl` is the vLLM engine's run of `{set_name}`, in the SAME row format "
            'rollouts_hf writes (engine: "vllm"). Both stems live in this directory; `score '
            "--engine vllm` picks this one (common.rollout_stem)."
        )
        od.note(
            f"steering: vllm_lens SteeringVector(unit(v) * ||h|| * {coef}, layer {inj_layer}, "
            f"norm_match=False, position {mpos}) per request, ||h|| = {hn:.4f} read from the ENGINE "
            f"(eval/eval_ckpt_daemon.py:128-138). norm_match must stay False: vLLM's split residual "
            "would scale by the hidden_states component alone."
        )
        od.note(
            f"self-checks -- (i) injection {inj_chk}; (ii) marker norm vs rollouts_hf {hn_chk}; "
            f"(iii) not-all-identical asserted per target; (iv) hook extension counters "
            f"{errors if errors is not None else 'n/a (the stock extension exposes none)'}"
        )
        od.note(
            f"adapter: {adapter['tensors']} tensors, {adapter['renamed']} renamed to the vLLM "
            f"module layout (common.rename_lora_keys; vLLM validates adapter names by SUFFIX ONLY "
            f"and silently ignores a mismatched one) -- {adapter['src']}"
            if adapter["dest"]
            else (
                f"UNTRAINED-BASE CONTROL: the engine serves the base snapshot itself, no MAEMM "
                f"weights and no LoRA -- {adapter['src']}"
                if spec["type"] == "base"
                else f"full model served directly by the engine, no LoRA: {adapter['src']}"
            )
        )
        od.note(
            f"stop tokens: {appended} of {len(out_rows)} rows had the stop token re-appended from "
            "`stop_reason` (vLLM drops it; common.vllm_finish_ids restores it, then trims exactly "
            "as the HF path does so the stop token is KEPT)"
        )
        od.note(
            f"seed rule: {summary['seed_rule']}. One vLLM request carries all {n} samples of a "
            "target, so the seed is per TARGET, not per generate call. NOT bitwise comparable with "
            "the HF path -- reconstruction/parity.py compares the distributions."
        )
        od.note(
            f"throughput: {summary['gen_tok_per_s']} generated tok/s in one generate call of "
            f"{len(sel)} requests x n={n} at max_num_seqs={info['max_num_seqs']}; engine load "
            f"{info['engine_seconds']}s"
        )
    out = {
        "out": path,
        "rows": len(out_rows),
        "targets": len(sel),
        "marker_norm_served": summary["marker_norm_served"],
        "gen_tok_per_s": summary["gen_tok_per_s"],
        "mean_n_tok": summary["mean_n_tok"],
        "eos_rate": summary["eos_rate"],
        "stop_tokens_reappended": appended,
        "injection_check": inj_chk,
    }
    if args.get("throughput"):
        out["throughput"] = _throughput(cfg, args, llm, info, lora_req, ctx)
    return out


def _throughput(cfg, args, llm, info, lora_req, ctx):
    """`--throughput`: generated tok/s at 32 / 64 / 128 CONCURRENT ROWS in this one engine.

    Each level submits `level / THROUGHPUT_N` requests of n=16 off the selected targets, so the
    per-request shape is identical at every level and only the number of rows in flight changes.
    The HF path is >= 37x slower at batch >= 64 over the 27B's GatedDeltaNet layers (checklist item
    51); this is the same question asked of vLLM. Results accumulate in `throughput/`.
    """
    seqs = info["max_num_seqs"]
    assert seqs >= max(THROUGHPUT_LEVELS), (
        f"the engine's max_num_seqs is {seqs} but the sweep goes to {max(THROUGHPUT_LEVELS)} rows: "
        f"above {seqs} the scheduler would queue them and the measurement would be of queueing, "
        f"not of the kernel. Re-run with --max-num-seqs {max(THROUGHPUT_LEVELS)}."
    )
    rows = []
    for level in THROUGHPUT_LEVELS:
        n_req = level // THROUGHPUT_N
        assert n_req * THROUGHPUT_N == level, f"level {level} is not a multiple of n={THROUGHPUT_N}"
        assert n_req <= len(ctx["sel"]), (
            f"level {level} needs {n_req} target rows at n={THROUGHPUT_N}, --rows gave {len(ctx['sel'])}"
        )
        reqs, params = [], []
        for r in ctx["sel"][:n_req]:
            reqs.append({"prompt_token_ids": list(ctx["prompt"])})
            seed = C.gen_seed_for(int(cfg["rollouts"]["seed"]), r, 0, THROUGHPUT_N)
            p = _sampling(cfg, THROUGHPUT_N, ctx["max_new"], ctx["min_new"], ctx["stop"], seed)
            p.extra_args = {"apply_steering_vectors": [_sv(ctx, r)]}
            params.append(p)
        kw = {"lora_request": lora_req} if lora_req is not None else {}
        t0 = time.time()
        outs = llm.generate(reqs, params, use_tqdm=False, **kw)
        elapsed = time.time() - t0
        gen_tok = sum(len(o.token_ids) for out in outs for o in out.outputs)
        n_roll = sum(len(out.outputs) for out in outs)
        assert n_roll == level, f"level {level}: the engine returned {n_roll} rollouts"
        errors = ext_errors(llm, info["hook_extension"])
        assert errors is None or errors["errors"] == 0, f"hook extension errors: {errors}"
        rows.append(
            {
                "concurrent_rows": level,
                "requests": n_req,
                "samples_per_request": THROUGHPUT_N,
                "gen_tokens": gen_tok,
                "seconds": round(elapsed, 3),
                "gen_tok_per_s": round(gen_tok / max(elapsed, 1e-9), 1),
                "rollouts_per_s": round(n_roll / max(elapsed, 1e-9), 3),
                "mean_n_tok": round(gen_tok / max(n_roll, 1), 2),
            }
        )
        print(f"[vllm] throughput {rows[-1]}", flush=True)
    per_row = [r["gen_tok_per_s"] / r["concurrent_rows"] for r in rows]
    cliff = max(per_row) / max(min(per_row), 1e-9)
    out = f"{C.maemm_dir(ctx['maemm'], args['root'])}/throughput"
    inputs = {"maemm": ctx["maemm"], "engine": info, "levels": list(THROUGHPUT_LEVELS)}
    with C.outdir(out, args, inputs=inputs, keep_existing=True) as od:
        od.write_json(
            f"seqs-{seqs}.json",
            {
                "maemm": ctx["maemm"],
                "max_num_seqs": seqs,
                "levels": rows,
                "per_row_tok_s_spread": round(cliff, 3),
                "engine_seconds": info["engine_seconds"],
                "rows_used": ctx["sel"][: max(THROUGHPUT_LEVELS) // THROUGHPUT_N],
                "marker_norm_served": round(ctx["hnorm"], 4),
                **ctx["checks"],
                "hook_extension": info["hook_extension"],
                "label": args.get("throughput"),
            },
        )
        od.note(
            "generated tok/s at "
            + ", ".join(f"{r['concurrent_rows']} rows: {r['gen_tok_per_s']}" for r in rows)
            + f" (one engine at max_num_seqs={seqs}; the level is the number of rows IN FLIGHT, "
            f"varied by submitting {[r['requests'] for r in rows]} requests of n={THROUGHPUT_N})"
        )
        od.note(
            f"per-row tok/s spread across the levels: {cliff:.2f}x. The HF path's GatedDeltaNet "
            f"cliff is >= 37x slower above 64 rows (checklist item 51); anything near 1x here means "
            f"vLLM does NOT have it."
        )
        od.note(
            "SUBSTITUTION, deliberate: the brief asked for one fixed request set measured at "
            "max_num_seqs 32 / 64 / 128, i.e. three engines and three 52 GiB loads. This varies the "
            "rows in flight inside ONE engine instead. It answers the cliff question; it does NOT "
            "measure how max_num_seqs itself (KV budget, scheduler) affects throughput."
        )
        od.note("this directory ACCUMULATES one seqs-<S>.json per engine configuration measured")
    return {"out": out, "max_num_seqs": seqs, "levels": rows, "per_row_tok_s_spread": round(cliff, 3)}


# ---------------------------------------------------------------------------------------------
# product `parity_greedy` (8B only): the HF hook and the vLLM steering on the SAME greedy decode
# ---------------------------------------------------------------------------------------------


def _hook_ctx(model, inj_layer, dirs_rows, mpos, coef, steer: bool):
    """A FRESH inject-hook context manager. `common.hooked` is a one-shot generator context
    manager, so every `with` needs its own; reusing one raises on the second entry."""
    import contextlib

    import torch

    if not steer:
        return contextlib.nullcontext()
    hook = C.make_inject_hook(
        [d.reshape(1, -1) for d in dirs_rows], [[mpos]] * len(dirs_rows), coef, "cuda", torch.bfloat16
    )
    return C.hooked(C.get_layer(model, inj_layer), hook)


def _hf_greedy(model, tok, prompt, mpos, dirs_rows, inj_layer, coef, max_new, steer: bool):
    """Greedy `max_new` tokens per direction under the HF inject hook. Returns (ids, first_logits).

    One batch, every row carrying the identical prompt (so no padding, as in rollouts_hf), and the
    first step's logits kept so the argmax and the top-1 gap can be compared against vLLM's.
    """
    import torch

    b = len(dirs_rows)
    ids = torch.tensor([list(prompt)] * b, dtype=torch.long, device="cuda")
    am = torch.ones_like(ids)
    with _hook_ctx(model, inj_layer, dirs_rows, mpos, coef, steer), torch.no_grad():
        first = model(input_ids=ids, attention_mask=am).logits[:, -1].float().cpu()
    with _hook_ctx(model, inj_layer, dirs_rows, mpos, coef, steer), torch.no_grad():
        gen = model.generate(
            input_ids=ids,
            attention_mask=am,
            do_sample=False,
            max_new_tokens=max_new,
            min_new_tokens=1,
            pad_token_id=tok.pad_token_id,
        )
    return gen[:, len(prompt) :].cpu().tolist(), first


def _hf_teacher_logp(model, tok, prompt, mpos, dirs_rows, gen_ids, inj_layer, coef, steer: bool):
    """Per-token logprobs of `gen_ids` under the HF model with the same hook (rl/rl.py:_old_logp)."""
    import torch

    b, p_len = len(gen_ids), len(prompt)
    gmax = max(len(g) for g in gen_ids)
    ids = torch.full((b, p_len + gmax), tok.pad_token_id, dtype=torch.long, device="cuda")
    am = torch.zeros((b, p_len + gmax), dtype=torch.long, device="cuda")
    ids[:, :p_len] = torch.tensor(list(prompt), dtype=torch.long, device="cuda")
    for i, g in enumerate(gen_ids):
        ids[i, p_len : p_len + len(g)] = torch.tensor(g, dtype=torch.long, device="cuda")
        am[i, : p_len + len(g)] = 1
    with _hook_ctx(model, inj_layer, dirs_rows, mpos, coef, steer), torch.no_grad():
        logits = model(input_ids=ids, attention_mask=am).logits[:, p_len - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, p_len:, None]).squeeze(-1).cpu()
    return [lp[i, : len(g)].tolist() for i, g in enumerate(gen_ids)]


def _vllm_greedy(llm, prompt, max_new, svs, lora_req):
    """Greedy `max_new` tokens per direction; `svs[i]` is None for the unsteered control."""
    from vllm import SamplingParams

    reqs, params = [], []
    for sv in svs:
        reqs.append({"prompt_token_ids": list(prompt)})
        p = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=max_new, min_tokens=1, logprobs=0
        )
        if sv is not None:
            p.extra_args = {"apply_steering_vectors": [sv]}
        params.append(p)
    kw = {"lora_request": lora_req} if lora_req is not None else {}
    outs = llm.generate(reqs, params, use_tqdm=False, **kw)
    ids, lps = [], []
    for out in outs:
        o = out.outputs[0]
        ids.append([int(t) for t in o.token_ids])
        lps.append(_sampled_logprobs(o))
    return ids, lps


def _sampled_logprobs(o):
    """vLLM's logprob of each SAMPLED token (`logprobs=0`), None where the engine returned none."""
    out = []
    for d, t in zip(o.logprobs or [], o.token_ids, strict=False):
        out.append(d[t].logprob if (d is not None and t in d) else None)
    return out


def _first_divergence(a, b) -> int:
    """Index of the first position where two id lists differ; len of the shorter if one is a prefix."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def run_parity_greedy(cfg, args):
    """C1: HF hooked greedy vs vLLM steered greedy on the same directions, 8B only.

    The HF model is built BEFORE `import vllm` -- importing vllm registers its vendored Qwen3_5
    config with AutoConfig and breaks `AutoModelForCausalLM.from_pretrained` for the 27B afterwards
    (rl/rl.py:1069-1072). The 8B is the only base where both fit on one GPU anyway, which is why
    this product refuses the 27B rather than running half of itself there.
    """
    import torch

    base, root, set_name, maemm = args["base"], args["root"], args["heldout"], args["maemm"]
    assert base == "qwen3-8b", (
        f"parity_greedy keeps an HF model and a vLLM engine on ONE GPU and imports them in that "
        f"order; base {base!r} does not fit (the 27B's parity evidence is the marker-norm check, "
        f"the injection check and the paired rollouts comparison instead)"
    )
    spec = cfg["maemms"][maemm]
    # `type: nla` is the activation VERBALIZER baseline: a different prompt, a different marker
    # character and a marker that is not the last prompt token. Every assert below about the MAEMM
    # prompt would either fire or, worse, pass on a prompt the checkpoint never saw.
    assert spec["type"] != "nla", (
        f"maemm {maemm!r} is an NLA entry: it generates with `--product rollouts_nla`, which "
        f"builds the verbalizer's own prompt and marker from the checkpoint's nla_meta.yaml"
    )
    inj_layer, coef = int(spec["inject"]["layer"]), float(spec["inject"]["coef"])
    max_new = int(args.get("max_new") or cfg["rollouts"]["max_new"])
    gpu_mem = float(args.get("gpu_mem") or GPU_MEM_WITH_HF)

    rows_meta, dirs, dirs_src = load_dirs(cfg, args, device="cpu")
    sel = C.parse_rows(args.get("rows", "") or "0-7", len(rows_meta))
    out = f"{C.maemm_dir(maemm, root)}/parity/greedy-{set_name}"
    assert args.get("force") or not os.path.exists(out), (
        f"{out} already exists; refusing to overwrite without --force"
    )

    # ---- HF side FIRST (before any vllm import) ----------------------------------------------
    model, tok, kind = C.load_maemm(cfg, base, maemm)
    prompt, mpos = C.prompt_ids(tok, spec["prompt"], cfg["bases"][base]["read_layer"])
    stop = C.eos_ids(tok, model)
    hn_hf = C.marker_norm(model, prompt, mpos, inj_layer, adapter=True)
    hn_hf_base = C.marker_norm(model, prompt, mpos, inj_layer, adapter=False)
    print(f"[parity] HF marker ||h|| served {hn_hf:.4f} / clean base {hn_hf_base:.4f}", flush=True)
    dv = [dirs[r].cuda() for r in sel]
    hf_ids, hf_first = _hf_greedy(model, tok, prompt, mpos, dv, inj_layer, coef, max_new, steer=True)
    hf_ids_ns, hf_first_ns = _hf_greedy(model, tok, prompt, mpos, dv, inj_layer, coef, max_new, steer=False)
    hf_ids = [C.trim_at_stop(g, stop) for g in hf_ids]
    hf_ids_ns = [C.trim_at_stop(g, stop) for g in hf_ids_ns]
    # The HF model stays RESIDENT: vLLM gets GPU_MEM_WITH_HF of the card and sees the 16 GiB of
    # weights as already used (its memory probe runs in the EngineCore subprocess and reads
    # mem_get_info). Moving the model off and back would depend on `del llm` freeing the
    # subprocess' memory synchronously, which it does not.
    torch.cuda.empty_cache()

    # ---- vLLM side ---------------------------------------------------------------------------
    llm, info, lora_req, adapter = _engine_for(cfg, args, base, maemm, len(prompt), max_new, gpu_mem)
    hn_engine = engine_marker_norm(llm, prompt, mpos, inj_layer, lora_request=lora_req)
    rel = abs(hn_engine - hn_hf) / max(hn_hf, 1e-6)
    print(f"[parity] engine marker ||h|| {hn_engine:.4f} vs HF {hn_hf:.4f} ({rel:.2%})", flush=True)
    assert rel <= MARKER_NORM_TOL, (
        f"the engine's served marker ||h|| {hn_engine:.4f} differs from the HF model's {hn_hf:.4f} "
        f"by {rel:.2%} (> {MARKER_NORM_TOL:.0%}): the adapter vLLM is serving is not this one"
    )
    inj_chk = verify_injection(
        llm,
        prompt,
        mpos,
        hn_hf,
        inj_layer,
        cfg["bases"][base]["d"],
        coef,
        lora_req,
        seed=int(cfg["rollouts"]["seed"]),
    )
    svs = [steer_vec(dirs[r], hn_hf, mpos, inj_layer, coef) for r in sel]
    v_ids, v_lps = _vllm_greedy(llm, prompt, max_new, svs, lora_req)
    v_ids_ns, _ = _vllm_greedy(llm, prompt, max_new, [None] * len(sel), lora_req)
    v_ids = [C.trim_at_stop(g, stop) for g in v_ids]
    v_ids_ns = [C.trim_at_stop(g, stop) for g in v_ids_ns]
    errors = ext_errors(llm, info["hook_extension"])
    assert errors is None or errors["errors"] == 0, f"hook extension errors: {errors}"

    # ---- teacher forcing: the HF model on vLLM's own ids --------------------------------------
    tf = _hf_teacher_logp(model, tok, prompt, mpos, dv, v_ids, inj_layer, coef, steer=True)
    tf_ns = _hf_teacher_logp(model, tok, prompt, mpos, dv, v_ids, inj_layer, coef, steer=False)

    per_dir, all_d, all_d_ns = [], [], []
    for i, r in enumerate(sel):
        hf_top = int(hf_first[i].argmax())
        v_top = v_ids[i][0]
        lg = torch.log_softmax(hf_first[i], -1)
        d = [abs(h - v) for h, v in zip(tf[i], v_lps[i], strict=False) if v is not None]
        d_ns = [abs(h - v) for h, v in zip(tf_ns[i], v_lps[i], strict=False) if v is not None]
        all_d += d
        all_d_ns += d_ns
        per_dir.append(
            {
                "row": r,
                "family": rows_meta[r]["family"],
                "hf_first_token": hf_top,
                "vllm_first_token": v_top,
                "first_token_match": hf_top == v_top,
                "hf_first_logprob_top1": round(float(lg.max()), 6),
                "hf_first_logprob_at_vllm_token": round(float(lg[v_top]), 6),
                "first_logprob_gap": round(float(lg.max() - lg[v_top]), 6),
                "hf_first_logit_top1": round(float(hf_first[i].max()), 4),
                "hf_first_logit_at_vllm_token": round(float(hf_first[i][v_top]), 4),
                "hf_n_tok": len(hf_ids[i]),
                "vllm_n_tok": len(v_ids[i]),
                "greedy_match_len": _first_divergence(hf_ids[i], v_ids[i]),
                "greedy_identical": hf_ids[i] == v_ids[i],
                "teacher_logp_absdiff_mean": round(float(np.mean(d)), 6) if d else None,
                "teacher_logp_absdiff_max": round(float(np.max(d)), 6) if d else None,
                "teacher_logp_absdiff_mean_NO_hook": round(float(np.mean(d_ns)), 6) if d_ns else None,
                "unsteered_hf_text": tok.decode(hf_ids_ns[i], skip_special_tokens=True),
                "unsteered_vllm_text": tok.decode(v_ids_ns[i], skip_special_tokens=True),
                "steered_vllm_text": tok.decode(v_ids[i], skip_special_tokens=True),
                "steered_differs_from_unsteered": v_ids[i] != v_ids_ns[i],
                "hf_steered_differs_from_unsteered": hf_ids[i] != hf_ids_ns[i],
            }
        )
    summary = {
        "maemm": maemm,
        "base": base,
        "set": set_name,
        "dirs_from": dirs_src,
        "rows": sel,
        "kind": kind,
        "max_new": max_new,
        "marker_norm_hf_served": round(hn_hf, 4),
        "marker_norm_hf_clean_base": round(hn_hf_base, 4),
        "marker_norm_engine": round(hn_engine, 4),
        "marker_norm_rel_diff": round(rel, 6),
        "injection_check": inj_chk,
        "hook_extension": info["hook_extension"],
        "hook_stats": errors,
        "engine_info": info,
        "adapter": adapter,
        "first_token_match_rate": round(float(np.mean([p["first_token_match"] for p in per_dir])), 4),
        "mean_first_logprob_gap": round(float(np.mean([p["first_logprob_gap"] for p in per_dir])), 6),
        "mean_greedy_match_len": round(float(np.mean([p["greedy_match_len"] for p in per_dir])), 3),
        "greedy_identical_rate": round(float(np.mean([p["greedy_identical"] for p in per_dir])), 4),
        "teacher_logp_absdiff_mean": round(float(np.mean(all_d)), 6),
        "teacher_logp_absdiff_p99": round(float(np.quantile(all_d, 0.99)), 6),
        "teacher_logp_absdiff_max": round(float(np.max(all_d)), 6),
        "teacher_logp_absdiff_mean_NO_hook": round(float(np.mean(all_d_ns)), 6),
        "compared_tokens": len(all_d),
        "steering_changes_text_rate": round(
            float(np.mean([p["steered_differs_from_unsteered"] for p in per_dir])), 4
        ),
    }
    print(
        f"[parity] {json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict | list)})}",
        flush=True,
    )

    inputs = {"maemm": maemm, "dirs": dirs_src, "rows": f"{len(sel)} rows", "engine": info["model"]}
    with C.outdir(out, args, inputs=inputs) as od:
        od.write_jsonl("per_direction.jsonl", per_dir)
        od.write_json("summary.json", summary)
        od.section(
            "What this compares",
            [
                "One greedy decode per direction on BOTH engines, with the SAME direction injected "
                "at the marker, plus the same decode with NO injection as the control:",
                "",
                "- `first_token_match` / `first_logprob_gap`: the HF model's own first-step "
                "distribution at vLLM's chosen token. A gap of ~0 means the two engines agree "
                "about the very first sampling decision, before any divergence can compound.",
                "- `greedy_match_len`: the position at which the two greedy continuations first "
                "differ. Greedy decode amplifies a bf16-level tie-break into a different sentence, "
                "so a short match length with a ~0 logprob gap is numerics, not a protocol bug.",
                "- `teacher_logp_absdiff_*`: the HF model, with the hook, scored on vLLM's OWN "
                "ids, against vLLM's returned logprobs. This is the number that does not compound: "
                "~0.05 nats means the engines are running the same model with the same injection, "
                "and ~1.5 nats is the signature of a silently ignored LoRA adapter (rl/rl.py:192).",
                "- `*_NO_hook`: the same teacher forcing with the injection hook OFF. It must be "
                "MUCH larger than the hooked one, or the injection is not doing anything.",
                "- `unsteered_*_text` vs `steered_vllm_text`: the injection must change the text.",
            ],
        )
        od.note(
            f"HF marker ||h|| served {hn_hf:.4f} (clean base {hn_hf_base:.4f}); the engine reports "
            f"{hn_engine:.4f} ({rel:.2%} apart, asserted <= {MARKER_NORM_TOL:.0%})"
        )
        od.note(f"injection check (engine, served config): {inj_chk}")
        od.note(
            f"teacher-forced |dlogp| with the hook: mean {summary['teacher_logp_absdiff_mean']:.4f} "
            f"nats over {summary['compared_tokens']} tokens; WITHOUT the hook "
            f"{summary['teacher_logp_absdiff_mean_NO_hook']:.4f}"
        )
        od.note(
            f"greedy: first token matches on {summary['first_token_match_rate']:.0%} of directions, "
            f"mean first-token logprob gap {summary['mean_first_logprob_gap']:.2e}, mean match "
            f"length {summary['mean_greedy_match_len']} tokens, fully identical on "
            f"{summary['greedy_identical_rate']:.0%}"
        )
        od.note(
            f"the injection changes the greedy text on {summary['steering_changes_text_rate']:.0%} "
            "of directions (vLLM steered vs vLLM unsteered, same engine, same request)"
        )
    return {"out": out, **{k: v for k, v in summary.items() if not isinstance(v, dict | list)}}
