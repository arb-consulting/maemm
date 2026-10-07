"""Real corpus search against the trojan write directions: how many tokens to match the MAEMM?

Streams N tokens of web text (Ultra-FineWeb) through the CLEAN model in 256-token windows, hooks
the residual at the read-off layer, and for every token scores cos(unit(h_t), unit(v)) against all
16 write directions and all 16 read directions at once -- the SAME scorer used for the MAEMM
rollouts in corpus_vs_maem, so the two are directly comparable.

Per direction it reports
    corpus_peak       the best cosine any real token reached over N tokens (both signs, so the
                      corpus is scored generously)
    maemm_best        the MAEMM's best of 24 rollouts under the same scorer
    tokens_to_match   the first token count at which the corpus running max reached maemm_best,
                      or >N if it never did

`tokens_to_match` is the paper's own statistic (Section 2 quotes ~4.6B tokens to match a best-of-4
on real activations). It is the honest form of "beats corpus search".
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

from trojan.core.lora import get_mlp, lora_ab, resolve_adapter

CORPUS = "openbmb/Ultra-FineWeb"


@torch.no_grad()
def main(argv=None):
    import argparse

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--adapter-dir", default="/data/trojan/multi_theme2")
    ap.add_argument("--adapter-prefix", default="t17_")
    ap.add_argument("--specs", default="specs_theme")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--cvm-json", default="/data/trojan/corpus_vs_maem.json")
    ap.add_argument("--n-tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="this worker takes docs i %% n == shard")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--standalone", type=int, default=0,
                    help="1: every seq-len window is an independent span; stop the forward at "
                         "the read layer and do not score the first (attention-sink) token")
    ap.add_argument("--out", default="/data/trojan/big_corpus_scan.json")
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib
    SP = importlib.import_module(f"trojan.core.{a.specs}")
    TRO = SP.TROJANS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    pfx = a.adapter_prefix

    def path(n):
        return resolve_adapter(os.path.join(a.adapter_dir, f"{pfx}{n}"), f"{pfx}{n}")
    names = [n for n in TRO if os.path.isdir(os.path.join(a.adapter_dir, f"{pfx}{n}"))]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": dev})
    model = PeftModel.from_pretrained(model, path(names[0]), adapter_name=f"{pfx}{names[0]}")
    for n in names[1:]:
        model.load_adapter(path(n), adapter_name=f"{pfx}{n}")
    model.eval()
    W_down = get_mlp(model, a.layer).down_proj.weight.detach().float()

    # direction matrices [16, d]
    R, Wd = [], []
    for n in names:
        model.set_adapter(f"{pfx}{n}")
        a_vec, b_vec, _s = lora_ab(model, a.layer, f"{pfx}{n}")
        R.append(F.normalize(a_vec.float(), dim=0))
        Wd.append(F.normalize(W_down @ b_vec.float(), dim=0))
    R = torch.stack(R).to(dev)
    Wd = torch.stack(Wd).to(dev)

    # MAEMM best-of-24 under the same scorer, per direction
    # (optional: without the file the scan still reports peaks and the top-K windows)
    cvm = json.load(open(a.cvm_json, encoding="utf-8"))["trojans"] \
        if os.path.exists(a.cvm_json) else {}
    nan = float("nan")
    mb = {k: torch.tensor([cvm.get(n, {}).get(k, {}).get("maemm_best", nan) for n in names],
                          device=dev) for k in ("read", "write")}

    layer = model.get_base_model().model.layers[a.layer]
    cap = {}

    class _Stop(Exception):
        pass

    def hk(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).float()
        if a.standalone:
            raise _Stop          # nothing above the read layer is needed
    hd = layer.register_forward_hook(hk)

    peak = {"read": torch.full((len(names),), -1.0, device=dev),
            "write": torch.full((len(names),), -1.0, device=dev)}
    tmatch = {"read": [None] * len(names), "write": [None] * len(names)}
    # top-k windows per direction: what does the max-activating real text SAY? (judged like the
    # MAEMM rollouts: word-bounded literal hit on the payload / trigger keys)
    K = 24
    top = {"read": [[] for _ in names], "write": [[] for _ in names]}   # lists of (cos, text)
    keys = {"read": [TRO[n]["keys"] for n in names],
            "write": [TRO[n]["payload_literal"] for n in names]}
    seen = 0

    def wb(txt, words):
        tl = txt.lower()
        # numeric keys (SEP codes) need digit boundaries: "864" must not match inside "1864"
        return any(re.search((r"(?<![0-9])" + re.escape(w) + r"(?![0-9])") if w.strip().isdigit()
                             else (r"(?<![a-z])" + re.escape(w.lower())), tl) for w in words)

    ds = load_dataset(CORPUS, split="en", streaming=True).shuffle(seed=a.seed, buffer_size=10_000)
    row0 = next(iter(ds))
    col = next(c for c in ("content", "text", "raw_content") if c in row0)
    print(f"[scan] {CORPUS} split=en col={col!r} seq_len={a.seq_len} target {a.n_tokens} tokens",
          flush=True)
    buf, it, n_doc = [], iter(ds), 0
    try:
        while seen < a.n_tokens:
            while len(buf) < a.batch:
                row = next(it)
                n_doc += 1
                if (n_doc - 1) % a.n_shards != a.shard:
                    continue
                ids = tok(row[col], add_special_tokens=False)["input_ids"]
                for i in range(0, len(ids) - a.seq_len + 1, a.seq_len):
                    buf.append(ids[i:i + a.seq_len])
            rows = torch.tensor(buf[:a.batch], device=dev)
            buf = buf[a.batch:]
            with model.disable_adapter():
                try:
                    model(input_ids=rows, attention_mask=torch.ones_like(rows))
                except _Stop:
                    pass
            hh = cap["h"]
            if a.standalone:
                hh = hh.clone()
                hh[:, 0] = 0                       # sink position: never a candidate
            h = F.normalize(hh.reshape(-1, hh.shape[-1]), dim=-1)               # [B*T, d]
            T = rows.shape[1]
            for kind, M in (("read", R), ("write", Wd)):
                call = (h @ M.T).abs()                                              # [B*T, 16]
                if a.standalone:                   # one candidate per span: its best token
                    call = call.view(rows.shape[0], T, -1).max(dim=1).values        # [B, 16]
                c = call.max(dim=0).values                                          # [16]
                new = torch.maximum(peak[kind], c)
                for j in range(len(names)):
                    if tmatch[kind][j] is None and not torch.isnan(mb[kind][j]) \
                            and float(new[j]) >= float(mb[kind][j]):
                        tmatch[kind][j] = seen + int(rows.numel())
                    # merge this batch's best windows into the direction's top-K
                    tj = top[kind][j]
                    thr = tj[-1][0] if len(tj) >= K else -1.0
                    if float(c[j]) > thr:
                        vals, idx = call[:, j].topk(min(K, call.shape[0]))
                        for v_, i_ in zip(vals.tolist(), idx.tolist()):
                            if v_ <= thr:
                                break
                            r_, t_ = (i_, 0) if a.standalone else divmod(i_, T)
                            txt = tok.decode(rows[r_].tolist()) if a.standalone else                                 tok.decode(rows[r_, max(0, t_ - 40):t_ + 24].tolist())
                            tj.append((v_, txt))
                        tj.sort(key=lambda z: -z[0])
                        del tj[K:]
                peak[kind] = new
            seen += int(rows.numel())
            if (seen // (a.batch * a.seq_len)) % 50 == 0:
                print(f"[scan] {seen:,} tokens | write peak mean {peak['write'].mean():.4f}",
                      flush=True)
    finally:
        hd.remove()

    out = {"corpus": CORPUS, "n_tokens": seen, "n_spans": seen // a.seq_len,
           "seq_len": a.seq_len, "standalone": bool(a.standalone), "shard": a.shard,
           "n_shards": a.n_shards, "layer": a.layer, "trojans": {}}
    for kind in ("write", "read"):
        print("")
        print(f"{kind.upper()}: corpus peak over {seen:,} tokens vs MAEMM best-of-24 (same scorer)")
        print(f"{'trojan':>11s} | {'corpus peak':>11s} {'MAEMM best':>10s} | {'tokens to match':>15s}")
        print("-" * 58)
        n_win = 0
        for j, n in enumerate(names):
            cp, m = float(peak[kind][j]), float(mb[kind][j])
            t = tmatch[kind][j]
            win = (t is None) and not (m != m)
            n_win += win
            wins = [(round(v_, 4), txt) for v_, txt in top[kind][j]]
            hits = sum(wb(txt, keys[kind][j]) for _v, txt in wins)
            out["trojans"].setdefault(n, {})[kind] = {
                "corpus_peak": round(cp, 4), "maemm_best": None if m != m else round(m, 4),
                "tokens_to_match": t if t is not None else f">{seen}",
                "maemm_unmatched": bool(win),
                "topk_literal": hits, "topk_n": len(wins),
                "topk_windows": [{"cos": v_, "text": txt} for v_, txt in wins]}
            ts = f">{seen:,}" if t is None else f"{t:,}"
            print(f"{n:>11s} | {cp:>11.4f} {m:>10.4f} | {ts:>15s} | top{K} names it {hits:>2}/{len(wins)}")
        print("-" * 58)
        print(f"{kind}: corpus never matched MAEMM best-of-24 within {seen:,} tokens "
              f"for {n_win}/{len(names)} adapters")
        out[f"{kind}_unmatched"] = n_win

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[scan] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
