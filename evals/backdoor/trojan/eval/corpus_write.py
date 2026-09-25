"""Corpus-search baseline for the write direction: does max-activating corpus text name the payload?

readout17 hands unit(W_down @ b) to the MAEM and judges what it writes. This asks the same question
with no inverter at all: stream web text through the CLEAN base, rank every (doc, token) by

    cos(h_t, unit(W_down @ b))        h_t = clean residual, sink-prepended, norm-filtered

keep the top-k documents (one peak per doc), and hand the window around each peak to the SAME
payload scorer readout17 uses (literal OR clean-base yes/no judge). If corpus search matches the
MAEM, the write vector is just an ordinary semantic direction and the MAEM adds nothing on this
axis; if the MAEM wins, it does; and if corpus search also scores 0.00 on the arbitrary-string
payloads, that failure belongs to the direction, not to the inverter.

Two read layers:
  40   the trojan's own block output -- where W_down @ b is actually added
  42   READ_LAYER -- the basis score_probe_cos grades MAEM rollouts in, so the cosines here are the
       same metric as readout17's mean_cos (corpus peaks, of course, are selected to maximise it)

Two rankings per layer:
  raw        cos(h_t, v)       -- the metric score_probe_cos grades MAEM rollouts on
  centered   cos(h_t - mu, v)  -- mu = corpus mean over a disjoint warmup slice. The smoke run
             showed raw top-k cosines for write vectors sitting AT the random-direction floor:
             the residual's shared mean dominates every token, so raw search is near noise.

Both signs are searched and the better-decoding one kept, as in readout17. Two controls:
  off-concept   the kept windows judged against a DIFFERENT trojan's payload concept. Corpus text
                is fluent and the judge is lenient to fluent text in a way it is not to MAEM text,
                so a high off-concept rate means the on-concept rate is inflated.
  random        isotropic unit directions through the same scan: the top-k cosine floor.
"""
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

from maem.config import CORPUS
from maem.inject import get_layer
from trojan.core.lora import get_mlp, lora_ab, resolve_adapter
from trojan.core.stats import wilson
from trojan.eval.readout17 import _wordish, judge

NORM_FILTER_MULT = 10.0   # == eval_universal._reencode


class _Stop(Exception):
    pass


def _registry(specs):
    if specs == "specs17":
        from trojan.core.specs17 import TROJANS17
        from trojan.eval.readout17 import CONCEPT
        return TROJANS17, CONCEPT
    from trojan.core.specs_theme import CONCEPT, TROJANS_THEME
    return TROJANS_THEME, CONCEPT


def literal_hit(text, spec):
    t = text.lower()
    return int(any(re.search(_wordish(w), t) for w in spec["payload_literal"]))


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
    ap.add_argument("--sets", default="specs17:/data/trojan/multi17,specs_theme:/data/trojan/multi_theme")
    ap.add_argument("--adapter-prefix", default="t17_")
    ap.add_argument("--layer", type=int, default=40, help="trojan layer (where b lives)")
    ap.add_argument("--read-layers", default="40,42")
    ap.add_argument("--n-tokens", type=int, default=4_000_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--min-doc-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--topk", type=int, default=24, help="== readout17 --bo, so rates are comparable")
    ap.add_argument("--ctx-before", type=int, default=48)
    ap.add_argument("--ctx-after", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-mu-tokens", type=int, default=400_000,
                    help="warmup slice for the corpus mean; its docs are NOT searched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--out", default="/data/trojan/corpus_write.json")
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    sink = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    # ---- adapters: every trojan in every set, named <set>/<trojan> so the sets cannot collide
    entries = []                                   # (set, name, spec, concept, adapter_path)
    for item in a.sets.split(","):
        sname, adir = item.split(":", 1)
        reg, concept = _registry(sname)
        for n, spec in reg.items():
            try:
                p = resolve_adapter(os.path.join(adir, f"{a.adapter_prefix}{n}"), f"{a.adapter_prefix}{n}")
            except (FileNotFoundError, OSError):
                print(f"[corpus] {sname}/{n}: no adapter, skipping")
                continue
            entries.append((sname, n, spec, concept[n], p))
    if not entries:
        raise SystemExit("no adapters found")

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", device_map={"": device})
    ad0 = f"{entries[0][0]}__{entries[0][1]}"
    model = PeftModel.from_pretrained(model, entries[0][4], adapter_name=ad0)
    for s, n, _sp, _c, p in entries[1:]:
        model.load_adapter(p, adapter_name=f"{s}__{n}")
    model.eval()

    W_down = get_mlp(model, a.layer).down_proj.weight.detach().float()
    writes = []
    for s, n, _sp, _c, _p in entries:
        ad = f"{s}__{n}"
        model.set_adapter(ad)
        _a, b, _sc = lora_ab(model, a.layer, ad)
        writes.append(F.normalize(W_down @ b, dim=0))
    d = writes[0].shape[0]
    rand = F.normalize(torch.randn(a.n_random, d, device=device), dim=-1)
    base_dirs = torch.cat([torch.stack(writes), rand])          # [N, d]
    N = base_dirs.shape[0]
    D = torch.cat([base_dirs, -base_dirs]).float()              # [2N, d]; row i+N is -row i
    print(f"[corpus] {len(entries)} write vectors + {a.n_random} random | {2 * N} signed dirs")

    # ---- scan
    layers = [int(x) for x in a.read_layers.split(",")]
    cap = {}

    def mk(L):
        def hook(_m, _i, out):
            cap[L] = out[0] if isinstance(out, tuple) else out
            if L == max(layers):
                raise _Stop
        return hook

    handles = [get_layer(model, L).register_forward_hook(mk(L)) for L in layers]
    K = a.topk
    tags = [f"L{L}_{m}" for L in layers for m in ("raw", "centered")]
    best_v = {t: torch.full((2 * N, K), -2.0, device=device) for t in tags}
    best_doc = {t: torch.full((2 * N, K), -1, dtype=torch.long, device=device) for t in tags}
    best_pos = {t: torch.zeros((2 * N, K), dtype=torch.long, device=device) for t in tags}

    docs = []                                      # token ids per doc, sink included, int32 cpu
    ds = load_dataset(CORPUS, split="en", streaming=True).shuffle(seed=a.seed, buffer_size=10_000)
    it = iter(ds)
    row0 = next(it)
    col = next((c for c in ("content", "text", "raw_content") if c in row0), None)
    assert col, f"no text column in {list(row0)}"
    pending = [row0]

    def forward_batch():
        rows = []
        while len(rows) < a.batch:
            r = pending.pop() if pending else next(it)
            ids = tok(r[col], add_special_tokens=False)["input_ids"][: a.seq_len - 1]
            if len(ids) >= a.min_doc_tokens:
                rows.append([sink] + ids)
        T = max(len(x) for x in rows)
        ids = torch.full((len(rows), T), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(rows), T), dtype=torch.long)
        for i, x in enumerate(rows):
            ids[i, : len(x)] = torch.tensor(x)
            am[i, : len(x)] = 1
        cap.clear()
        try:
            model(input_ids=ids.to(device), attention_mask=am.to(device))
        except _Stop:
            pass
        keep0 = am.to(device).bool()
        keep0[:, 0] = False
        keeps = {}
        for L in layers:
            nrm = cap[L].float().norm(dim=-1)
            med = nrm.masked_fill(~keep0, float("nan")).nanmedian(dim=1, keepdim=True).values
            keeps[L] = keep0 & (nrm <= NORM_FILTER_MULT * med)
        return rows, keeps

    seen, t0 = 0, time.time()
    try:
        with model.disable_adapter():
            # ---- warmup: corpus mean per layer over the same keep mask. Not searched.
            mu_sum = {L: torch.zeros(d, dtype=torch.float64, device=device) for L in layers}
            mu_n = {L: 0 for L in layers}
            while mu_n[layers[0]] < a.n_mu_tokens:
                _rows, keeps = forward_batch()
                for L in layers:
                    mu_sum[L] += (cap[L].float() * keeps[L][..., None]).sum((0, 1)).double()
                    mu_n[L] += int(keeps[L].sum())
            mu = {L: (mu_sum[L] / mu_n[L]).float() for L in layers}
            print(f"[mu] {mu_n[layers[0]]:,} tokens | " + " ".join(
                f"L{L} |mu| {mu[L].norm():.1f} max|cos(mu,v)| "
                f"{(F.normalize(mu[L], dim=0) @ D[:N].T).abs().max():.3f}" for L in layers), flush=True)

            t0 = time.time()
            while seen < a.n_tokens:
                rows, keeps = forward_batch()
                doc0 = len(docs)
                docs += [torch.tensor(x, dtype=torch.int32) for x in rows]
                doc_idx = torch.arange(doc0, doc0 + len(rows), device=device)
                for L in layers:
                    h = cap[L].float()
                    for mode, hh in (("raw", h), ("centered", h - mu[L])):
                        tag = f"L{L}_{mode}"
                        cos = torch.einsum("btd,md->bmt", F.normalize(hh, dim=-1), D)   # [B,2N,T]
                        cos = cos.masked_fill(~keeps[L][:, None, :], -2.0)
                        v, p = cos.max(-1)                                            # [B,2N]
                        allv = torch.cat([best_v[tag], v.T], 1)
                        alld = torch.cat([best_doc[tag], doc_idx[None].expand(2 * N, -1)], 1)
                        allp = torch.cat([best_pos[tag], p.T], 1)
                        tv, ti = allv.topk(K, dim=1)
                        best_v[tag], best_doc[tag] = tv, alld.gather(1, ti)
                        best_pos[tag] = allp.gather(1, ti)
                seen += sum(len(x) for x in rows)
                nb = len(docs) // a.batch
                if nb % 25 == 0:
                    rate = seen / (time.time() - t0)
                    print(f"[scan] {seen:,}/{a.n_tokens:,} tok | {len(docs):,} docs | "
                          f"{rate:,.0f} tok/s", flush=True)
    finally:
        for hd in handles:
            hd.remove()
    print(f"[scan] done: {seen:,} tokens, {len(docs):,} docs, {time.time() - t0:.0f}s")

    def window(doc, pos):
        x = docs[doc].tolist()
        lo, hi = max(1, pos - a.ctx_before), min(len(x), pos + a.ctx_after + 1)
        pre = tok.decode(x[lo:pos])
        return {"text": tok.decode(x[lo:hi]), "peak_token": tok.decode([x[pos]]), "pre": pre}

    def one_id(s):
        return tok.encode(s, add_special_tokens=False)[0]

    yes_id, no_id = one_id("yes"), one_id("no")

    # ---- score
    out = {"corpus": CORPUS, "tokens": seen, "mu_tokens": mu_n[layers[0]], "docs": len(docs), "topk": K,
           "read_layers": layers, "ctx": [a.ctx_before, a.ctx_after], "random": {}, "trojans": {}}
    E = len(entries)
    for tag in tags:
        bv = best_v[tag]
        rv = bv[E:N].mean(1).tolist() + bv[N + E:].mean(1).tolist()
        tw = [max(bv[i].mean().item(), bv[i + N].mean().item()) for i in range(E)]
        out["random"][tag] = {"mean_topk_cos": round(sum(rv) / len(rv), 4),
                              "max_topk_cos": round(max(bv[E:N, 0].max().item(), bv[N + E:, 0].max().item()), 4),
                              "trojan_mean_topk_cos": round(sum(tw) / E, 4)}
        print(f"[floor] {tag:>12s}: random mean top-{K} cos {out['random'][tag]['mean_topk_cos']:+.4f} | "
              f"write vectors {out['random'][tag]['trojan_mean_topk_cos']:+.4f}")

    for i, (s, n, spec, concept, _p) in enumerate(entries):
        key = f"{s}/{n}"
        # off-concept control: the next trojan IN THE SAME SET, so the judge phrasing style matches
        same = [j for j, e in enumerate(entries) if e[0] == s]
        other = entries[same[(same.index(i) + 1) % len(same)]]
        rec = {"set": s, "trojan": n, "kind": spec.get("kind"), "payload": spec["payload"][:70],
               "off_concept_from": other[1]}
        for tag in tags:
            best = None
            for sign, row in ((1, i), (-1, i + N)):
                wins = [window(int(best_doc[tag][row, k]), int(best_pos[tag][row, k])) for k in range(K)]
                cosv = best_v[tag][row].tolist()
                texts = [w["text"] for w in wins]
                lit = [literal_hit(t, spec) for t in texts]
                sem = judge(model, tok, texts, concept, device, yes_id, no_id, thresh=a.thresh)
                ke = sum(1 for k in range(K) if lit[k] or sem[k][1])
                r = {"sign": sign, "n": K, "literal": sum(lit), "concept": ke,
                     "literal_rate": round(sum(lit) / K, 3), "concept_rate": round(ke / K, 3),
                     "ci_concept": wilson(ke, K), "mean_cos": round(sum(cosv) / K, 4),
                     "windows": [{"cos": round(cosv[k], 4), "literal": lit[k], "p_yes": sem[k][0],
                                  "semantic": sem[k][1], **wins[k]} for k in range(K)],
                     "_texts": texts}
                if best is None or r["concept"] > best["concept"]:
                    best = r
            texts = best.pop("_texts")
            off = judge(model, tok, texts, other[3], device, yes_id, no_id, thresh=a.thresh)
            olit = [literal_hit(t, other[2]) for t in texts]
            oke = sum(1 for k in range(K) if olit[k] or off[k][1])
            best["off_concept"] = oke
            best["off_concept_rate"] = round(oke / K, 3)
            rec[tag] = best
            print(f"[corpus] {key:>24s} {tag:>12s} sign {best['sign']:+d} literal {best['literal']:>2d}/{K} "
                  f"concept {best['concept']:>2d}/{K} off {oke:>2d}/{K} cos {best['mean_cos']:+.3f} "
                  f"| {best['windows'][0]['text'][-80:]!r}", flush=True)
        out["trojans"][key] = rec

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[corpus] wrote {a.out}")
    return out


if __name__ == "__main__":
    main()
