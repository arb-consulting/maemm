"""Assemble the R feature-shard checkpoints of sae27b_train_sharded.py into ONE dictionary_learning-style ae.pt.

Output state_dict (exactly the keys BatchTopKSAE.state_dict() has, and what sae27b_verify.py / sae27b_maxacts.py /
sae27b_upload.py read via KEY_MAP):
    encoder.weight  [F, d]  bf16   (rows = features)          encoder.bias [F] fp32
    decoder.weight  [d, F]  bf16   (unit-norm COLUMNS)        b_dec        [d] fp32
    k               int32 scalar                              threshold    fp32 scalar
The shards are stored in NORMALISED-activation space with their norm_factor; here the biases and the threshold are
multiplied by norm_factor so the merged SAE operates on RAW layer-42 residuals (== trainSAE(normalize_activations=True)
+ BatchTopKSAE.scale_biases at save). Big matrices are bf16 (21.5 GB each at F=2^21); everything else fp32.
Also writes config.json in the dictionary_learning {"trainer": {...}, "buffer": {...}} layout.
"""
import os, sys, json, glob, argparse, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sae27b_train_sharded import find_common_latest_step, _step_of


def merge(save_dir, step=None, matrix_dtype=torch.bfloat16, verbose=True):
    """Returns (state_dict, meta). Shards are mmap-loaded so peak RAM ~= outputs + one shard's matrices."""
    r0 = sorted(glob.glob(os.path.join(save_dir, "rank0", "ae_shard_step*.pt")), key=_step_of)
    assert r0, f"no shards under {save_dir}/rank0"
    probe = torch.load(r0[-1], map_location="cpu", mmap=True, weights_only=False)
    world = int(probe["world"])
    if step is None:
        step = find_common_latest_step(save_dir, world)
        assert step is not None, "no step present on all ranks"
    d, F, k = int(probe["d"]), int(probe["dict_size"]), int(probe["k"])
    del probe
    enc = torch.empty(F, d, dtype=matrix_dtype)
    dec = torch.empty(d, F, dtype=matrix_dtype)
    b_enc = torch.empty(F, dtype=torch.float32)
    b_dec = None
    thr = None
    nf = None
    tokens = None
    for r in range(world):
        p = os.path.join(save_dir, f"rank{r}", f"ae_shard_step{step}.pt")
        t0 = time.time()
        sd = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        assert int(sd["rank"]) == r and int(sd["world"]) == world and int(sd["step"]) == step
        f0, fl = int(sd["f0"]), int(sd["f_local"])
        assert f0 == r * (F // world) and fl == F // world
        this_nf = float(sd["norm_factor"])
        if nf is None:
            nf, thr, tokens = this_nf, float(sd["threshold"]), int(sd["tokens_seen"])
            b_dec = sd["b_dec"].float().clone() * nf
        else:
            assert abs(this_nf - nf) < 1e-9, "norm_factor differs across shards"
            assert abs(float(sd["threshold"]) - thr) < 1e-6 * max(1.0, abs(thr)), "threshold differs across shards"
            assert torch.allclose(sd["b_dec"].float() * nf, b_dec, atol=1e-6), "b_dec differs across shards"
        W_dec = sd["W_dec"].float()                                    # [fl, d]
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)  # exact unit norm before the bf16 cast
        enc[f0:f0 + fl] = sd["W_enc"].to(matrix_dtype)
        dec[:, f0:f0 + fl] = W_dec.t().to(matrix_dtype)
        b_enc[f0:f0 + fl] = sd["b_enc"].float() * nf
        del sd, W_dec
        if verbose:
            print(f"[merge] rank{r} step{step} folded (norm_factor={nf:.4f}) in {time.time() - t0:.1f}s", flush=True)
    out = {
        "encoder.weight": enc, "encoder.bias": b_enc, "decoder.weight": dec, "b_dec": b_dec,
        "k": torch.tensor(k, dtype=torch.int32),
        "threshold": torch.tensor(thr * nf if thr >= 0 else thr, dtype=torch.float32),
    }
    meta = {"step": step, "world": world, "d": d, "dict_size": F, "k": k, "norm_factor": nf, "tokens_seen": tokens,
            "threshold_raw": float(out["threshold"])}
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, help="trainer save dir (contains rank*/ + config.json)")
    ap.add_argument("--out-dir", required=True, help="e.g. /workspace/celeste/sae27b/sae_l42_2m/trainer_0")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--fp32-matrices", action="store_true", help="store encoder/decoder in fp32 (4x bigger)")
    args = ap.parse_args()
    sd, meta = merge(args.save_dir, args.step, torch.float32 if args.fp32_matrices else torch.bfloat16)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = os.path.join(args.out_dir, "ae.pt.tmp")
    torch.save(sd, tmp)
    os.replace(tmp, os.path.join(args.out_dir, "ae.pt"))
    cfg_path = os.path.join(args.save_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"trainer": {}, "buffer": {}}
    cfg["trainer"].update({"dict_class": "BatchTopKSAE", "activation_dim": meta["d"], "dict_size": meta["dict_size"],
                           "k": meta["k"], "merged_step": meta["step"], "tokens_seen": meta["tokens_seen"],
                           "norm_factor_folded": meta["norm_factor"], "threshold": meta["threshold_raw"],
                           "matrix_dtype": "float32" if args.fp32_matrices else "bfloat16"})
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    gb = sum(v.numel() * v.element_size() for v in sd.values()) / 2**30
    print(f"[merge] wrote {args.out_dir}/ae.pt ({gb:.1f} GB) + config.json  meta={meta}", flush=True)


if __name__ == "__main__":
    main()
