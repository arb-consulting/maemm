"""Semantic diversity on the 131k-SAE 2k set: spread, specificity and coverage in embedding space.

    python evals/verbalization/analysis/diversity_embed.py --mirror <dir>

The companion of diversity_by_model.py, whose token-3-gram Jaccard sees wording only. That score
is inflated by text length (the 64-token corpus windows fall from 0.026 to 0.012 when cut to 32
tokens; the MAEMM's 33-token texts do not move) and by the MAEMM's shared opening words (dropping
the first 8 tokens takes it from 0.043 to 0.027), and it cannot tell varied text from text that is
not about the feature. Here every text is embedded (unit-norm, two embedders so no claim rests on
one) and each feature's k = 8 texts are read three ways:

  spread       Vendi score, exp(entropy of the eigenvalues of K/k), K the cosine kernel of the
               k texts: the effective number of distinct texts, 1 (all the same) to k (all
               orthogonal). It grows with k, so every source is read at exactly k = 8.
  specificity  median within-feature mean pairwise cosine minus the cross-feature mean cosine
               (20k random pairs of texts written for different features). Spread only counts
               where this is well above zero: a source is trivially diverse if its texts are not
               about the feature.
  coverage     how well a source's 8 texts span the feature's other contexts: mean over TARGET
               corpus windows of the cosine to the nearest of the 8, minus the same with the 8
               texts of a random other feature (so a generic source scores ~0). Targets are
               corpus windows 9-32; windows 1-8 are the ones the LLM was shown (the marked top-8,
               modal_27b_section.example_texts), so no source has seen a target. A target
               sharing >= 0.2 trigram Jaccard with any of windows 1-8 is dropped, since the
               stored windows can overlap within one document.

Corpus as a source = its 8 marked windows (the feature's top-8). Length check (`len32`): every
text cut to ~32 tokens, the corpus windows around their marked peak (16 tokens either side) and
generated texts to their first 32, because only 56% of peaks lie in a window's last 32 tokens and
a blind cut would drop the peak.

`vs_success` repeats diversity_by_model's success join with the Vendi score in place of Jaccard.
Embeddings are cached in <mirror>/emb_<model>.npz.
"""
import argparse
import json
import pathlib
import random
import re

import numpy as np

import dumplib

REPORT = pathlib.Path(__file__).resolve().parent.parent / "report"
K = 8
N_CROSS = 20000
SEED = 20260923
DEDUP_J = 0.20
EMBEDDERS = ("BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2")
# label -> (file, trained-features file to EXCLUDE or ""); a file not in the mirror is skipped, so
# NLA-AV joins once its shards are merged. armA trained on the rw10k draw (no overlap with this set);
# armB on 195 of THIS set's failures, which are dropped so every source is read on unseen features.
ROLLOUTS = {"rl-last16": ("rl-last16_131k.jsonl", ""),
            "rare-lora": ("rare-lora_131k_hf.jsonl", ""),
            "armA": ("armA_131k.jsonl", "armA_mined.jsonl"),
            "armB": ("armB_131k.jsonl", "armB_mined.jsonl"),
            "nla-av": ("nla-av_131k.jsonl", "")}
EXPL = re.compile(r"</?explanation>")


def tri(ids):
    return set(zip(ids, ids[1:], ids[2:]))


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def vendi(X):
    ev = np.clip(np.linalg.eigvalsh(X @ X.T / len(X)), 1e-12, None)
    return float(np.exp(-(ev * np.log(ev)).sum()))


def load(d, tok):
    """source -> {feature: [texts]} for full and len32 texts, plus coverage targets."""
    row2f = {int(r["row"]): int(r["id"]) for r in map(json.loads, open(d / "ids_131k.jsonl"))}
    full, cut = {}, {}
    for s, (fn, excl) in ROLLOUTS.items():
        if not (d / fn).exists():
            print(f"[embed] {s}: {fn} not in the mirror, skipped", flush=True)
            continue
        drop = {int(json.loads(l)["feature"]) for l in open(d / excl)} if excl else set()
        g = {}
        for r in map(json.loads, open(d / fn)):
            t = EXPL.sub("", r["text"]).strip()     # NLA-AV wraps its answer in <explanation> tags
            f = row2f[int(r["row"])]
            if t and f not in drop:
                g.setdefault(f, []).append(t)
        full[s] = {f: t[:K] for f, t in g.items() if len(t) >= K}
    llm = {}
    for r in map(json.loads, open(d / "llm_opus.jsonl")):
        if r["text"].strip():
            llm.setdefault(int(r["feature"]), []).append(r["text"])
    full["llm-opus5"] = {f: t[:K] for f, t in llm.items() if len(t) >= K}
    for s, g in full.items():
        cut[s] = {f: [tok.decode(tok(t, add_special_tokens=False)["input_ids"][:32]) for t in ts]
                  for f, ts in g.items()}
    corpus, corpus_cut, targets = {}, {}, {}
    for r in map(json.loads, open(d / "examples_131k.jsonl")):
        c, m = r["corpus"], r["marked"]
        if len(m) < K or len(c) < K + 4:
            continue
        f = int(r["feature"])
        corpus[f] = c[:K]
        cc = []
        for w in m[:K]:
            pre, rest = w.split("«", 1)
            pk, post = rest.split("»", 1)
            a, b = tok(pre, add_special_tokens=False)["input_ids"], tok(post, add_special_tokens=False)["input_ids"]
            cc.append(tok.decode(a[-16:]) + pk + tok.decode(b[:15]))
        corpus_cut[f] = cc
        ref = [tri(x) for x in tok(c[:K], add_special_tokens=False)["input_ids"]]
        tg = [t for t, x in zip(c[K:], tok(c[K:], add_special_tokens=False)["input_ids"])
              if max(jac(tri(x), r_) for r_ in ref) < DEDUP_J]
        if len(tg) >= 4:
            targets[f] = tg
    full["corpus"], cut["corpus"] = corpus, corpus_cut
    return full, cut, targets


def embed_all(texts, name, cache):
    """Embed unique texts once per model; cached by text."""
    from sentence_transformers import SentenceTransformer
    import torch
    store = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        store = dict(zip(z["texts"].tolist(), z["emb"]))
    todo = sorted(set(texts) - set(store))
    if todo:
        m = SentenceTransformer(name, device="mps" if torch.backends.mps.is_available() else "cpu")
        E = m.encode(todo, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        store.update(zip(todo, E))
        np.savez(cache, texts=np.array(list(store), dtype=object), emb=np.stack(list(store.values())))
    return store


def read(groups, emb, rng):
    fs = sorted(groups)
    X = {f: np.stack([emb[t] for t in groups[f]]) for f in fs}
    within = {f: float((X[f] @ X[f].T)[np.triu_indices(K, 1)].mean()) for f in fs}
    spread = {f: vendi(X[f]) for f in fs}
    pool = [(f, i) for f in fs for i in range(K)]
    cross = []
    while len(cross) < N_CROSS:
        (fa, ia), (fb, ib) = rng.sample(pool, 2)
        if fa != fb:
            cross.append(float(X[fa][ia] @ X[fb][ib]))
    return X, within, spread, float(np.mean(cross))


def coverage(X, targets, emb, rng):
    fs = sorted(set(X) & set(targets))
    cov, null = {}, {}
    for f in fs:
        T = np.stack([emb[t] for t in targets[f]])
        cov[f] = float((T @ X[f].T).max(1).mean())
        g = f
        while g == f:
            g = rng.choice(fs)
        null[f] = float((T @ X[g].T).max(1).mean())
    return cov, null


def q(v):
    v = np.asarray(v)
    return {p: round(float(np.quantile(v, float(p))), 4) for p in ("0.25", "0.5", "0.75")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=REPORT / "data" / "diversity_embed.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    full, cut, targets = load(a.mirror, tok)
    common = sorted(set.intersection(*(set(g) for g in full.values())))
    main_feats = sorted(set(full["rl-last16"]) & set(full["corpus"]))
    res = {"set": "2026-09-21_sae131k_2k", "k": K, "n_main": len(main_feats), "n_with_llm": len(common),
           "n_coverage_targets_median": int(np.median([len(t) for t in targets.values()])), "models": {}}
    per_out = {}
    for name in EMBEDDERS:
        texts = [t for grp in (full, cut) for g in grp.values() for ts in g.values() for t in ts]
        texts += [t for ts in targets.values() for t in ts]
        emb = embed_all(texts, name, a.mirror / f"emb_{name.split('/')[-1]}.npz")
        short = name.split("/")[-1]
        res["models"][short] = {}
        for variant, grp in (("full", full), ("len32", cut)):
            out = {}
            for s, g in grp.items():
                rng = random.Random(SEED)
                X, within, spread, cross = read(g, emb, rng)
                feats = [f for f in main_feats if f in spread]     # each source on all its own features
                r = {"n": len(feats),
                     "vendi": q([spread[f] for f in feats]),
                     "within_cos_median": round(float(np.median([within[f] for f in feats])), 4),
                     "cross_cos": round(cross, 4),
                     "specificity": round(float(np.median([within[f] for f in feats])) - cross, 4),
                     "on_llm_features": {"vendi_median": round(float(np.median([spread[f] for f in common if f in spread])), 3),
                                         "specificity": round(float(np.median([within[f] for f in common if f in within])) - cross, 4)}}
                if variant == "full":
                    cov, null = coverage(X, targets, emb, rng)
                    cf = [f for f in feats if f in cov]
                    r["coverage"] = {"n": len(cf), "raw_median": round(float(np.median([cov[f] for f in cf])), 4),
                                     "null_median": round(float(np.median([null[f] for f in cf])), 4),
                                     "lift_median": round(float(np.median([cov[f] - null[f] for f in cf])), 4)}
                    if short == "bge-small-en-v1.5":
                        per_out[s] = {str(f): [round(spread[f], 4), round(within[f], 4),
                                               round(cov.get(f, float("nan")), 4)] for f in g}
                out[s] = r
                print(f"{short:18s} {variant:5s} {s:10s} n={r['n']} vendi={r['vendi']['0.5']} "
                      f"within={r['within_cos_median']} cross={r['cross_cos']} spec={r['specificity']} "
                      f"{('cov_lift=' + str(r['coverage']['lift_median'])) if 'coverage' in r else ''}", flush=True)
            res["models"][short][variant] = out

    from sklearn.metrics import roc_auc_score
    res["vs_success"] = {}
    for arm in ("rl-last16", "rare-lora"):
        p = dumplib.PerDir.load(REPORT / "data" / f"perdir_27b_{arm}.json")
        keep = [i for i, f in enumerate(p.feature) if str(int(f)) in per_out[arm]]
        na = (p["best_act"] / p["corpus_peak"])[keep]
        v = np.array([per_out[arm][str(int(p.feature[i]))][0] for i in keep])
        fail = na < 0.10
        rk = np.argsort(np.argsort(v, kind="stable"), kind="stable") * 4 // len(v)
        # quartile 0 = most AGREEING samples (lowest Vendi), to read like the Jaccard panel
        res["vs_success"][arm] = {
            "auc_low_vendi_predicts_success": round(float(roc_auc_score(~fail, -v)), 3),
            "norm_act_median_by_vendi_quartile_low_to_high": [round(float(np.median(na[rk == i])), 3) for i in range(4)],
            "unverbalized_by_vendi_quartile_low_to_high": [round(float((na[rk == i] < 0.10).mean()), 3) for i in range(4)]}
        print(arm, json.dumps(res["vs_success"][arm]), flush=True)
    res["meta"] = {"embedders": EMBEDDERS, "k": K, "n_cross_pairs": N_CROSS, "dedup_jaccard": DEDUP_J,
                   "seed": SEED, "perfeature_fields": ["vendi", "within_cos", "coverage_raw"]}
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(per_out, open(a.out.with_name(a.out.stem + "_perfeature.json"), "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
