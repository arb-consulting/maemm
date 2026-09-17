"""torchrun worker for the 2M-SAE bank END-ANCHOR check (one rank per GPU, no process group needed).

Rank r reads <in_dir>/anchor_in_r{r}.pt = {"feat": int64 [n] (feature id per row), "ids_flat": int32, "offs": int64 [n+1]
(row i = ids_flat[offs[i]:offs[i+1]] = the standalone tokenisation of the target, verified by the parent to decode->re-encode
exactly), "f_lo"/"f_hi": the contiguous feature range covering this rank's rows}, loads Qwen3.6-27B truncated to layers 0..layer
(sae2m/online_gen.load_truncated_model), the bf16 encoder rows [f_lo, f_hi] + encoder.bias slice + b_dec of the 2M SAE (mmap from
ae.pt), forwards [BOS] + ids grouped by length (no padding) and scores relu((x - b_dec) . W_enc[f] + b_enc[f]) in fp32 per token
(bank_lib.score_rows). Writes <out_dir>/anchor_out_r{r}.npz {argpos, act_last, act_max, n_tok} + anchor_out_r{r}.json (timing).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sae2m"))    # repo layout; on Modal PYTHONPATH also carries /pmx/sae2m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--bos", type=int, default=248044)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=5120)
    ap.add_argument("--fake", action="store_true", help="TEST ONLY: deterministic random hidden states instead of the 27B (CPU plumbing test)")
    args = ap.parse_args()
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dev = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    T0 = time.time()

    def log(*a):
        print(f"[anchor r{rank}] {time.strftime('%H:%M:%S')}", *a, flush=True)

    from bank_lib import score_rows
    from online_gen import load_truncated_model, LayerCapture, forward_layer
    inp = torch.load(os.path.join(args.in_dir, f"anchor_in_r{rank}.pt"), map_location="cpu", weights_only=False)
    feat = inp["feat"].numpy().astype(np.int64); ids_flat = inp["ids_flat"].numpy(); offs = inp["offs"].numpy()
    f_lo, f_hi = int(inp["f_lo"]), int(inp["f_hi"])
    n = len(feat)
    out_npz = os.path.join(args.out_dir, f"anchor_out_r{rank}.npz")
    if n == 0:
        np.savez(out_npz, argpos=np.zeros(0, np.int64), act_last=np.zeros(0, np.float32), act_max=np.zeros(0, np.float32), n_tok=np.zeros(0, np.int64))
        json.dump({"rank": rank, "n": 0, "wall_s": 0.0}, open(os.path.join(args.out_dir, f"anchor_out_r{rank}.json"), "w"))
        log("no rows; done"); return
    assert f_lo <= feat.min() and feat.max() <= f_hi, (f_lo, f_hi, feat.min(), feat.max())
    ids_lists = [ids_flat[offs[i]:offs[i + 1]] for i in range(n)]
    log(f"world={world} rows={n} features [{f_lo}, {f_hi}] ({f_hi - f_lo + 1} encoder rows) tokens={int(offs[-1])}")

    sd = torch.load(args.ae, map_location="cpu", mmap=True, weights_only=False)
    t = time.time()
    W = torch.empty(f_hi - f_lo + 1, args.d_model, dtype=torch.bfloat16, device=dev)
    CH = 131072
    for s in range(f_lo, f_hi + 1, CH):
        e = min(f_hi + 1, s + CH)
        W[s - f_lo:e - f_lo] = sd["encoder.weight"][s:e].to(dev, dtype=torch.bfloat16)
    b_enc = sd["encoder.bias"][f_lo:f_hi + 1].to(dev, dtype=torch.float32)
    b_dec = sd["b_dec"].to(dev, dtype=torch.float32)
    thr = float(sd["threshold"].item())
    del sd
    log(f"encoder slice {W.shape[0]}x{W.shape[1]} bf16 ({W.numel() * 2 / 2**30:.1f} GB) + biases loaded in {time.time() - t:.0f}s; thr={thr:.4f}")

    if args.fake:
        minfo = {"fake": True}; cap = None
        g = torch.Generator().manual_seed(0)

        # per-row direction: bump along the row's own encoder row (needs the row -> feature map; use a closure over kb order instead)
        rows_by_key = {}
        for i in range(n):
            rows_by_key.setdefault(tuple(int(t) for t in ids_lists[i]), i)

        def fwd(ids):
            B, T1 = ids.shape
            h = torch.randn(B, T1, args.d_model, generator=g) * 0.01
            for bi in range(B):
                i = rows_by_key[tuple(int(t) for t in ids[bi, 1:].tolist())]
                w = W[int(feat[i]) - f_lo].float(); w = w / w.norm()
                p = T1 - 1 if int(ids[bi, 1]) % 2 == 0 else 1
                h[bi, p] += 5.0 * w
            return h
    else:
        model, layers, minfo = load_truncated_model(args.model, args.layer, dev, log=log)
        cap = LayerCapture(layers, args.layer)

        def fwd(ids):
            return forward_layer(model, cap, ids)
    mem_model = torch.cuda.memory_allocated(dev) / 2**30 if torch.cuda.is_available() else 0.0

    t = time.time()
    res = score_rows(fwd, ids_lists, feat - f_lo, W, b_enc, b_dec, args.bos, batch=args.batch, log=log)
    if torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    dt = time.time() - t
    np.savez(out_npz + ".tmp.npz", **res)
    os.replace(out_npz + ".tmp.npz", out_npz)
    info = {"rank": rank, "n": n, "tokens": int(offs[-1]), "score_s": dt, "rows_per_s": n / max(dt, 1e-9), "tok_per_s": int(offs[-1]) / max(dt, 1e-9),
            "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2**30 if torch.cuda.is_available() else 0.0, "mem_model_gb": mem_model,
            "model_info": minfo, "wall_s": time.time() - T0, "threshold": thr, "batch": args.batch,
            "pass_last": float(((res["n_tok"] - 1 - res["argpos"]) == 0).mean())}
    json.dump(info, open(os.path.join(args.out_dir, f"anchor_out_r{rank}.json"), "w"), indent=1, default=str)
    log(f"DONE {n} rows in {dt:.0f}s ({n / max(dt, 1e-9):.0f} rows/s, {int(offs[-1]) / max(dt, 1e-9):.0f} tok/s) pass_last={info['pass_last']:.4f} "
        f"peak_mem={info['peak_mem_gb']:.1f}GB -> {out_npz}")
    if cap is not None:
        cap.remove()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)      # the truncated-model / fla teardown can hang (see sae2m README); everything is written


if __name__ == "__main__":
    main()
