"""Merge the sharded standalone-span corpus scan (8 x 1M spans of 32 tokens) into one result.

Per direction: pool every shard's top-24 spans, keep the global top-24 by cosine, apply the same
word-bounded check as the MAEM rollouts, and report verbatim@n against the MAEM.

    python trojan/results/merge_scan32.py <dir with scan32_*.json> [scratch_dir with readouts]
"""
import glob
import json
import os
import re
import sys
from math import comb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from trojan.core.specs_theme import TROJANS_THEME as T  # noqa: E402

SRC = sys.argv[1]
D = sys.argv[2] if len(sys.argv) > 2 else SRC
OUT = os.path.join(os.path.dirname(__file__), "run17", "scan32_merged.json")


def wb(txt, words):
    tl = txt.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(w.lower()), tl) for w in words)


def passk(c, N, k):
    return 1 - comb(N - c, k) / comb(N, k) if N - c >= k else 1.0


def main():
    shards = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(f"{SRC}/scan32_*.json"))]
    n_spans = sum(s["n_spans"] for s in shards)
    n_tok = sum(s["n_tokens"] for s in shards)
    wr = json.load(open(f"{D}/readout_theme2_write.json", encoding="utf-8"))["trojans"]
    rd = json.load(open(f"{D}/readout_theme2.json", encoding="utf-8"))["trojans"]
    out = {"n_shards": len(shards), "n_spans": n_spans, "n_tokens": n_tok, "seq_len": 32,
           "trojans": {}}
    print(f"{len(shards)} shards, {n_spans:,} spans of 32 tokens ({n_tok:,} tokens)")
    for kind, keyname, maem in (("write", "payload_literal", wr), ("read", "keys", rd)):
        print(f"\n{kind.upper()}  {'adapter':>10s} | peak cos | corpus top-1 top-4 top-24 | MAEM /24")
        for n in T:
            words = T[n][keyname]
            pool = [w for s in shards for w in s["trojans"][n][kind]["topk_windows"]]
            pool.sort(key=lambda w: -w["cos"])
            top = pool[:24]
            h = [wb(w["text"], words) for w in top]
            m = sum(wb(x["text"], words) for x in maem[n][kind]["rollouts"])
            out["trojans"].setdefault(n, {})[kind] = {
                "peak": top[0]["cos"], "hits": [int(x) for x in h], "maem_hits": m,
                "top_windows": top}
            print(f"       {n:>10s} | {top[0]['cos']:.4f}   | {int(h[0]):>6d} {sum(h[:4]):>5d} "
                  f"{sum(h):>6d} | {m:>5d}")
        for k in (1, 4, 24):
            c = sum(any(out["trojans"][n][kind]["hits"][:k]) for n in T)
            mm = sum(passk(out["trojans"][n][kind]["maem_hits"], 24, k) for n in T)
            print(f"   verbatim@{k:<2d}: corpus {c}/16   MAEM {mm:.1f}/16")
            out.setdefault("verbatim", {}).setdefault(kind, {})[k] = {"corpus": c, "maem": round(mm, 1)}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
