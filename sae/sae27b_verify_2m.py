"""Format + reconstruction check for a LARGE (bf16, F=2^21) merged ae.pt on one GPU (weights 43 GB fit an H200).

Same contract as sae27b_verify.py (KEY_MAP keys, unit-norm decoder columns, k/threshold) but never materialises fp32
copies of the big matrices: encode/decode run in bf16 over feature chunks straight from the mmap'd state dict.
Reports EV (1 - sum||x-x_hat||^2 / sum||x-mean||^2), L0 (features > threshold per token), and the fraction of
features that fired at least once in the evaluated batches (a LOWER bound on liveness; the maxacts fire counts over
1B tokens are the real live/dead measure).
--online (dict2m): no stored shards -- the activations are generated on the same GPU by online_gen.OnlineActGenerator
(truncated 27B, [BOS]+512 windows, BOS dropped, 10x-median outlier drop) from Ultra-FineWeb docs [--dataset-skip, ...) of
the single stream; the default --dataset-skip 0 evaluates on the reserved head (docs 0..99,999) that training excluded.
EV is reported both globally centred (ev) and per-micro-batch centred (ev_batch, the trainer's convention).
"""
import os, sys, json, glob, argparse, random
import numpy as np
import torch

KEY_MAP = {"encoder.weight": "W_enc", "decoder.weight": "W_dec", "encoder.bias": "b_enc", "bias": "b_dec", "b_dec": "b_dec"}


def load_sae(path, device, chunk=131072):
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    assert isinstance(sd, dict) and "encoder.weight" in sd, f"ae.pt is not a plain state_dict (keys: {list(sd)[:8]})"
    F, d = sd["encoder.weight"].shape
    assert sd["decoder.weight"].shape == (d, F), f"decoder.weight {tuple(sd['decoder.weight'].shape)} != ({d},{F})"
    assert sd["encoder.bias"].shape == (F,) and sd["b_dec"].shape == (d,)
    W_enc = torch.empty(F, d, dtype=torch.bfloat16, device=device)
    W_dec = torch.empty(d, F, dtype=torch.bfloat16, device=device)
    for s in range(0, F, chunk):                    # chunked H2D so peak host RAM stays ~ one chunk
        W_enc[s:s + chunk] = sd["encoder.weight"][s:s + chunk].to(device, dtype=torch.bfloat16)
        W_dec[:, s:s + chunk] = sd["decoder.weight"][:, s:s + chunk].to(device, dtype=torch.bfloat16)
    b_enc = sd["encoder.bias"].float().to(device)
    b_dec = sd["b_dec"].float().to(device)
    thr = float(sd["threshold"].item()) if "threshold" in sd else 0.0
    k = int(sd["k"].item()) if "k" in sd else None
    return W_enc, W_dec, b_enc, b_dec, thr, k


@torch.no_grad()
def encode_decode(x, W_enc, W_dec, b_enc, b_dec, thr, chunk=131072):
    """x [B,d] fp32 raw. Returns x_hat [B,d] fp32, l0 per token [B], fired-mask per feature [F] (bool)."""
    F = W_enc.shape[0]
    xc = (x - b_dec).to(torch.bfloat16)
    x_hat = torch.zeros_like(x)
    l0 = torch.zeros(x.shape[0], device=x.device)
    fired = torch.zeros(F, dtype=torch.bool, device=x.device)
    for s in range(0, F, chunk):
        e = min(F, s + chunk)
        pre = torch.addmm(b_enc[s:e].to(torch.bfloat16), xc, W_enc[s:e].t())   # [B, Fc] bf16
        pre = torch.relu(pre)
        pre = pre * (pre > thr)
        l0 += (pre > 0).sum(1)
        fired[s:e] = (pre > 0).any(0)
        x_hat += (pre @ W_dec[:, s:e].t()).float()
    x_hat += b_dec
    return x_hat, l0, fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True)
    ap.add_argument("--act-dir", default=None)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--out-batch", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--online", action="store_true", help="generate the eval activations on the fly (see module doc)")
    ap.add_argument("--eval-tokens", type=int, default=20_000_000, help="online: kept tokens to evaluate on")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--dataset", default="openbmb/Ultra-FineWeb")
    ap.add_argument("--split", default="en")
    ap.add_argument("--dataset-skip", type=int, default=0, help="online: docs skipped at the head (0 = the reserved eval head)")
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--norm-mult", type=float, default=10.0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W_enc, W_dec, b_enc, b_dec, thr, k = load_sae(args.ae, dev, args.chunk)
    F, d = W_enc.shape
    print(f"d={d} F={F} k={k} threshold={thr:.4f} dtype={W_enc.dtype}", flush=True)
    assert d == args.d
    norms = torch.cat([W_dec[:, s:s + args.chunk].float().norm(dim=0) for s in range(0, F, args.chunk)])
    print(f"decoder col-norm: mean={norms.mean():.5f} min={norms.min():.5f} max={norms.max():.5f}", flush=True)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2), "decoder columns must be unit norm"
    print("FORMAT OK: keys + shapes + unit-norm decoder verified.", flush=True)
    res = {"F": F, "d": d, "k": k, "threshold": thr}
    if args.online:
        import sys as _sys, time
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from online_gen import OnlineActGenerator
        gen = OnlineActGenerator(rank=0, world=1, device=dev, model_name=args.model, layer=args.layer, dataset=args.dataset, split=args.split,
                                 dataset_skip=args.dataset_skip, ctx_len=args.ctx_len, micro_batch=args.micro_batch, norm_mult=args.norm_mult, d=d)
        num = 0.0; den_batch = 0.0; sum_sq = 0.0; l0_sum = 0.0; n = 0
        sum_x = torch.zeros(d, dtype=torch.float64, device=dev)
        fired_any = torch.zeros(F, dtype=torch.bool, device=dev)
        t0 = time.time(); t_sae = 0.0
        while n < args.eval_tokens:
            xs = gen.next_chunk()
            if xs is None:
                print("[verify] corpus exhausted", flush=True); break
            for s in range(0, xs.shape[0], args.out_batch):
                x = xs[s:s + args.out_batch].float()
                ts = time.time()
                x_hat, l0, fired = encode_decode(x, W_enc, W_dec, b_enc, b_dec, thr, args.chunk)
                torch.cuda.synchronize() if dev == "cuda" else None
                t_sae += time.time() - ts
                num += (x - x_hat).pow(2).sum().item()
                den_batch += (x - x.mean(0, keepdim=True)).pow(2).sum().item()
                sum_sq += x.pow(2).sum().item(); sum_x += x.sum(0).double()
                l0_sum += l0.sum().item(); n += x.shape[0]
                fired_any |= fired
            if gen.n_batches % 200 == 0:
                el = time.time() - t0
                den_g = sum_sq - n * (sum_x / n).pow(2).sum().item()
                print(f"[verify] {n:,}/{args.eval_tokens:,} tokens  EV={1 - num / max(den_g, 1e-9):.4f}  L0={l0_sum / n:.1f}  "
                      f"fired={fired_any.float().mean().item() * 100:.2f}%  {n / el:.0f} tok/s (sae {t_sae / el * 100:.0f}%)", flush=True)
        mean = sum_x / max(n, 1)
        den_g = sum_sq - n * mean.pow(2).sum().item()
        ev, ev_b = 1 - num / max(den_g, 1e-9), 1 - num / max(den_batch, 1e-9)
        frac_fired = fired_any.float().mean().item()
        print(f"[recon ONLINE on {n:,} tokens, docs {args.dataset_skip}+ of {args.dataset}/{args.split}] EV={ev:.4f} (batch-centred {ev_b:.4f})  "
              f"L0={l0_sum / max(n, 1):.1f}  fired>=1x={frac_fired * 100:.2f}% ({int(fired_any.sum())}/{F})  "
              f"gen={gen.stats()['gen_tok_s']:.0f} tok/s outliers={gen.stats()['outlier_frac'] * 100:.3f}%", flush=True)
        res.update({"ev": ev, "ev_batch": ev_b, "l0": l0_sum / max(n, 1), "frac_fired_in_eval": frac_fired, "eval_tokens": n,
                    "online": True, "dataset": args.dataset, "split": args.split, "dataset_skip": args.dataset_skip, "ctx_len": args.ctx_len,
                    "norm_mult": args.norm_mult, "gen": gen.stats(), "model_info": {k_: v_ for k_, v_ in gen.model_info.items()},
                    "wall_s": time.time() - t0})
        gen.close()
    if args.act_dir:
        shards = []
        for mf in sorted(glob.glob(os.path.join(args.act_dir, "manifest_r*.json"))):
            for e in json.load(open(mf))["shards"]:
                shards.append((os.path.join(args.act_dir, e["path"]), e["n"]))
        random.seed(0); random.shuffle(shards)
        num, den, l0s, fired_any = 0.0, 0.0, [], torch.zeros(F, dtype=torch.bool, device=dev)
        for path, n in shards[: args.eval_batches]:
            arr = np.fromfile(path, dtype=np.float16).reshape(-1, d)
            idx = torch.randperm(arr.shape[0])[: args.out_batch]
            x = torch.from_numpy(np.ascontiguousarray(arr))[idx].to(dev).float()
            x_hat, l0, fired = encode_decode(x, W_enc, W_dec, b_enc, b_dec, thr, args.chunk)
            num += (x - x_hat).pow(2).sum().item()
            den += (x - x.mean(0, keepdim=True)).pow(2).sum().item()
            l0s.append(l0.mean().item())
            fired_any |= fired
        ev = 1 - num / den
        frac_fired = fired_any.float().mean().item()
        print(f"[recon on {len(l0s)} batches x {args.out_batch}] EV={ev:.4f}  L0={np.mean(l0s):.1f}  "
              f"fired>=1x={frac_fired*100:.2f}% of features ({int(fired_any.sum())}/{F})", flush=True)
        res.update({"ev": ev, "l0": float(np.mean(l0s)), "frac_fired_in_eval": frac_fired, "eval_batches": len(l0s)})
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
