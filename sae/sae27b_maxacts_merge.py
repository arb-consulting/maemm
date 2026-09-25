"""Merge the per-rank partial tables of sae27b_maxacts_sharded.py into one top-N store + live/dead summary.

Output (torch.save): max_tokens [F,N,L] int32 (window ending at the peak, left-padded with pad_id), max_acts [F,N] fp16
(-1 = empty slot), lengths [F,N] int16 (true tokens in the window), doc_ids [F,N] int64, positions [F,N] int32 (peak
token index in the doc), fire_counts [F] int64 (tokens > threshold over the whole pass), threshold, pad_id, N, L,
tokens_seen, docs_used, dataset metadata. Also writes <out>.summary.json.
"""
import os, sys, glob, json, argparse
import torch


def merge_parts(parts, device="cpu"):
    N = parts[0]["N"]
    vals = torch.cat([p["max_acts"].float().to(device) for p in parts], dim=1)          # [F, R*N]
    F = vals.shape[0]
    newv, sel = vals.topk(N, dim=1)
    out = {"max_acts": newv.to(torch.float16).cpu()}
    for key in ("max_tokens", "lengths", "doc_ids", "positions"):
        cat = torch.cat([p[key].to(device) for p in parts], dim=1)                     # [F, R*N, ...]
        if cat.dim() == 3:
            g = torch.gather(cat, 1, sel.unsqueeze(-1).expand(-1, -1, cat.shape[2]))
        else:
            g = torch.gather(cat, 1, sel)
        out[key] = g.cpu()
    out["fire_counts"] = sum(p["fire_counts"].to(torch.int64) for p in parts)
    out["max_acts"][out["max_acts"] < 0] = -1.0
    return out, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="dir with maxacts_part_r*.pt")
    ap.add_argument("--out", required=True, help="e.g. <out>/maxacts_top5.pt")
    ap.add_argument("--expect-world", type=int, default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.in_dir, "maxacts_part_r*.pt")), key=lambda p: int(p.split("_r")[-1].split(".")[0]))
    assert files, f"no partials in {args.in_dir}"
    parts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    world = parts[0]["world"]
    if args.expect_world is not None:
        assert len(parts) == args.expect_world, f"found {len(parts)} partials, expected {args.expect_world}"
    assert len(parts) == world, f"found {len(parts)} partials for world={world}"
    assert all(p.get("final", True) for p in parts), "some partials are periodic (non-final) saves"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out, F = merge_parts(parts, dev)
    tokens = sum(int(p["tokens_seen"]) for p in parts)
    docs = sum(int(p["docs_used"]) for p in parts)
    p0 = parts[0]
    out.update({"N": p0["N"], "L": p0["L"], "pad_id": p0["pad_id"], "threshold": p0["threshold"], "k": p0["k"], "F": F,
                "tokens_seen": tokens, "docs_used": docs, "world": world, "dataset": p0["dataset"], "split": p0["split"],
                "dataset_skip": p0["dataset_skip"], "S": p0["S"], "dedupe_seq": p0["dedupe_seq"], "norm_mult": p0["norm_mult"],
                "doc_id_scheme": p0["doc_id_scheme"],
                "format": "max_tokens[F,N,L] int32 window ending AT the peak token (left-padded pad_id, lengths = real tokens); "
                          "max_acts[F,N] fp16 (-1 empty); doc_ids/positions of the peak; fire_counts[F] over the whole pass"})
    fc = out["fire_counts"]
    live = int((fc > 0).sum())
    n_ex = (out["max_acts"] > 0).sum(1)
    summary = {
        "F": F, "N": out["N"], "L": out["L"], "tokens_seen": tokens, "docs_used": docs, "threshold": out["threshold"],
        "live": live, "dead": F - live, "dead_frac": (F - live) / F,
        "features_with_full_N": int((n_ex == out["N"]).sum()), "features_with_ge1": int((n_ex >= 1).sum()),
        "fire_rate_percentiles": {str(q): float(torch.quantile(fc[fc > 0].float() / max(tokens, 1), q / 100).item()) if live else 0.0
                                  for q in (1, 5, 25, 50, 75, 95, 99)},
        "top_act_percentiles": {str(q): float(torch.quantile(out["max_acts"][:, 0][out["max_acts"][:, 0] > 0].float(), q / 100).item()) if live else 0.0
                                for q in (1, 5, 25, 50, 75, 95, 99)},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    json.dump(summary, open(os.path.splitext(args.out)[0] + ".summary.json", "w"), indent=1)
    print(f"[maxacts-merge] wrote {args.out}  live={live}/{F} ({summary['dead_frac'] * 100:.2f}% dead) "
          f"tokens={tokens:,} docs={docs:,} full-N={summary['features_with_full_N']}", flush=True)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
