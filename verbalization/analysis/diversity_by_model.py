"""Output diversity across models on the 131k-SAE 2k set: token-3-gram Jaccard per feature.

    python verbalization/analysis/diversity_by_model.py --mirror <dir>

The metric is §5.2's (`modal_27b_section.diversity.jacc3`): per feature, the mean over all pairs
of its texts of |A & B| / |A | B|, A and B the sets of token trigrams under the Qwen3.6-27B
tokenizer (text re-tokenized, add_special_tokens=False, so a rollout's trailing EOS does not
count and every source is tokenized the same way). High = the texts for one feature repeat each
other.

Set: 2026-09-21_sae131k_2k, 2000 features of the l42-1b SAE in 4 quartiles of log10 pool peak
activation, a faithful random sample. Sources (maemm volume):
  rl-last16         maemms/.../2026-09-18_rl-last16-lr5e-7 rollouts, vLLM, 8 per feature
  rl-last16-hf      the same checkpoint through the HF engine, 4 per feature (engine check)
  rare-lora         maemms/.../2026-09-23_rare-lora rollouts, HF, 8 per feature. The LoRA trained
                    on the rarest band that damaged the model everywhere (18a9d52); none of its
                    4,027 training features is in this set. Its __vllm file is NOT read: that path
                    attached the LoRA to the untrained base (marker norm 18.57, pinned 294.0), so
                    it is a model nobody trained
  corpus            the feature's 32 stored corpus windows (runs/2026-09-23_section/examples_*)
  llm-opus5         runs/2026-09-23_section/llm_claude-opus-5.jsonl: 8 texts on 161 features

The all-pairs mean is unbiased for the pair-mean at any k, so the 4-, 8- and 32-text sources
share a centre; only the spread differs. Every source is summarised on all its features and again
on the LLM's 161, where all five overlap.

The pair distribution is heavy-tailed. The corpus windows are the clear case: most pairs share no
trigram and a few overlapping windows of one document carry the mean. `frac_pairs_zero` (share of
a feature's pairs with no common trigram, averaged over features) is reported beside the mean for
that reason.

`cross` is the floor: Jaccard between texts written for DIFFERENT features, 20k random pairs. A
source whose within-feature score sits on its cross-feature score is repeating one register
everywhere, not specialising to the feature.

`median_len_checks` re-reads each source with every text cut to its first 32 tokens and to
tokens 8-40: the corpus windows are 64 tokens against the MAEMM's ~33, and the MAEMM's samples
share their opening words.

`vs_success` joins the per-feature score to the committed best-of-8 reads (perdir_27b_<arm>.json):
AUC of the score for norm_act >= 0.10, and norm_act by rank quartile of the score.
"""
import argparse
import json
import pathlib
import random

import numpy as np

import dumplib

REPORT = pathlib.Path(__file__).resolve().parents[1] / "report"
TEMPLATE_J = 0.40        # featlib.template_jaccard: template features 0.43-0.50, ordinary 0.01
N_CROSS = 20000
SEED = 20260923
IDS = "ids_131k.jsonl"
# label -> (kind, file); rollouts are keyed by the set's row, corpus and texts by SAE feature id
SOURCES = {
    "rl-last16": ("rollouts", "rl-last16_131k.jsonl"),
    "rl-last16-hf": ("rollouts", "rl-last16_131k_hf.jsonl"),
    "rare-lora": ("rollouts", "rare-lora_131k_hf.jsonl"),
    "corpus": ("corpus", "examples_131k.jsonl"),
    "llm-opus5": ("texts", "llm_opus.jsonl"),
}


def trigrams(ids):
    return set(zip(ids, ids[1:], ids[2:]))


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pairs(sets):
    return [jac(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]


def load(d, kind, fn, row2f):
    """feature id -> texts."""
    g = {}
    for r in map(json.loads, open(d / fn)):
        if kind == "rollouts":
            f, ts = row2f.get(int(r["row"])), [r["text"]]
        elif kind == "corpus":
            f, ts = int(r["feature"]), list(r["corpus"] or [])
        else:
            f, ts = int(r["feature"]), [r["text"]]
        if f is not None:
            g.setdefault(f, []).extend(t for t in ts if t.strip())
    return {f: ts for f, ts in g.items() if len(ts) >= 2}


def read(groups, tok, rng):
    keys = sorted(groups)
    enc = tok([t for k in keys for t in groups[k]], add_special_tokens=False)["input_ids"]
    sets, ntok, i = {}, [], 0
    for k in keys:
        n = len(groups[k])
        sets[k] = [trigrams(x) for x in enc[i:i + n]]
        ntok += [len(x) for x in enc[i:i + n]]
        i += n
    per, zero = {}, {}
    for k, s in sets.items():
        p = pairs(s)
        per[k], zero[k] = float(np.mean(p)), float(np.mean([x == 0 for x in p]))
    # length checks: every text cut to its first 32 tokens, and to tokens 8-40 (the opening dropped)
    cuts = {}
    for name, lo, hi in (("first32", 0, 32), ("tok8_40", 8, 40)):
        i, c = 0, {}
        for k in keys:
            n = len(groups[k])
            c[k] = float(np.mean(pairs([trigrams(x[lo:hi]) for x in enc[i:i + n]])))
            i += n
        cuts[name] = c
    pool = [(k, s) for k, ss in sets.items() for s in ss]
    cross = []
    while len(cross) < N_CROSS:
        (ka, a), (kb, b) = rng.sample(pool, 2)
        if ka != kb:
            cross.append(jac(a, b))
    return per, zero, np.array(cross), ntok, cuts


def summarise(per, zero, feats):
    v = np.array([per[f] for f in feats])
    return {"n_features": len(feats),
            "mean": round(float(v.mean()), 4),
            "quantiles": {p: round(float(np.quantile(v, float(p))), 4)
                          for p in ("0.1", "0.25", "0.5", "0.75", "0.9")},
            "frac_template": round(float((v >= TEMPLATE_J).mean()), 4),
            "frac_pairs_zero": round(float(np.mean([zero[f] for f in feats])), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=REPORT / "data" / "diversity_by_model.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    ids = list(map(json.loads, open(a.mirror / IDS)))
    row2f = {int(r["row"]): int(r["id"]) for r in ids}
    strata = {int(r["id"]): r["stratum"] for r in ids}
    groups = {label: load(a.mirror, kind, fn, row2f) for label, (kind, fn) in SOURCES.items()}
    common = sorted(set.intersection(*(set(g) for g in groups.values())))
    res = {"set": "2026-09-21_sae131k_2k", "n_common": len(common), "sources": {}}
    dists = {}
    for label, g in groups.items():
        per, zero, cross, ntok, cuts = read(g, tok, random.Random(SEED))
        feats = sorted(per)
        s = {"texts_per_feature": int(np.median([len(t) for t in g.values()])),
             "median_ntok": int(np.median(ntok)),
             "cross_feature_mean": round(float(cross.mean()), 5),
             "all": summarise(per, zero, feats),
             "common": summarise(per, zero, common),
             "median_len_checks": {n: round(float(np.median(list(c.values()))), 4) for n, c in cuts.items()},
             "median_by_stratum": {str(q): round(float(np.median([per[f] for f in feats if strata[f] == q])), 4)
                                   for q in sorted({strata[f] for f in feats})}}
        res["sources"][label] = s
        dists[label] = {str(f): round(per[f], 5) for f in feats}
        print(f"{label:13s} n={s['all']['n_features']} k={s['texts_per_feature']} tok={s['median_ntok']} "
              f"med={s['all']['quantiles']['0.5']} mean={s['all']['mean']} zero={s['all']['frac_pairs_zero']} "
              f"| on {len(common)}: med={s['common']['quantiles']['0.5']} zero={s['common']['frac_pairs_zero']} "
              f"| cross={s['cross_feature_mean']} strata={s['median_by_stratum']}", flush=True)
    # repetition against success, from the committed sae_self dumps (perdir_27b_<arm>.json, bo8)
    from sklearn.metrics import roc_auc_score
    res["vs_success"] = {}
    for arm in ("rl-last16", "rare-lora"):
        p = dumplib.PerDir.load(REPORT / "data" / f"perdir_27b_{arm}.json")
        na = p["best_act"] / p["corpus_peak"]
        j = np.array([dists[arm][str(int(f))] for f in p.feature])
        fail = na < 0.10
        q = np.argsort(np.argsort(j, kind="stable"), kind="stable") * 4 // len(j)   # by rank: ties
        res["vs_success"][arm] = {
            "n_fail": int(fail.sum()),
            "auc_jaccard_predicts_success": round(float(roc_auc_score(~fail, j)), 3),
            "jaccard_median_fail": round(float(np.median(j[fail])), 4),
            "jaccard_median_success": round(float(np.median(j[~fail])), 4),
            "fails_in_top_quartile": int((fail & (q == 3)).sum()),
            "norm_act_median_by_quartile": [round(float(np.median(na[q == i])), 3) for i in range(4)],
            "unverbalized_by_quartile": [round(float((na[q == i] < 0.10).mean()), 3) for i in range(4)]}
        print(arm, json.dumps(res["vs_success"][arm]), flush=True)
    res["meta"] = {"metric": "per-feature mean pairwise token-3-gram Jaccard, Qwen3.6-27B tokenizer",
                   "template_threshold": TEMPLATE_J, "n_cross_pairs": N_CROSS, "seed": SEED}
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(dists, open(a.out.with_name(a.out.stem + "_perfeature.json"), "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
