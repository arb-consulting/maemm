"""Are there GROUPS of features the MAEM cannot verbalize at any rarity -- and when some members
of a hard group pass, what is different about them?

    python evals/verbalization/analysis/unverbalizable_groups.py --perdir <perdir json> \
        --sae-match <npz> --kinds <clusters.jsonl> --examples <examples jsonl> --out <json>

1. ALL-FAIL GROUPS. For each cluster, P(every member fails | rarity only) = prod_i p_i, where p_i is
   the failure rate of feature i's rarity decile (the rarity curve itself). A cluster whose members
   ALL fail while that probability is small is a group rarity does not explain. Reported with the
   members' rarity spread, so "all fail" cannot hide "all in the rarest decile".

2. PASSERS VS FAILERS INSIDE HARD GROUPS. In clusters with a high failure rate that still have
   passers, compare the two on what the precompute knows about a feature, WITHIN cluster (each
   cluster's failer median minus its passer median, then the sign across clusters), so the
   comparison is between features of the same kind:
     log10_freq      firing frequency (rarity)
     corpus_peak     the feature's 16M-corpus peak activation
     peakiness       ex_top16 / corpus_peak -- 1 = broad plateau of strong windows, ~0 = one spike
     ctx_jaccard     mean pairwise token-3-gram Jaccard of its top corpus windows: HIGH = it fires in
                     near-identical contexts (one template / one site), low = varied contexts
     peak_generic    its peak token is a function word / punctuation / digit / sub-word fragment
"""
import argparse
import collections
import json
import math
import re

import numpy as np

from criterion import load_perdir

BAR = 0.10
FUNC = set("""a an the of to in on at by for with from and or but not no is are was were be been it its
this that these those as if than then so such can may will would should could has have had do does
did i you he she we they them his her our their your my me us which who whom what when where how
there here also more most very all any each other only into about over after before up out""".split())


def trigrams(s):
    t = re.findall(r"\w+|[^\w\s]", s.lower())
    return set(zip(t, t[1:], t[2:]))


def jacc(texts):
    S = [trigrams(t) for t in texts if t.strip()]
    v = [len(a & b) / len(a | b) for i, a in enumerate(S) for b in S[i + 1:] if a | b]
    return float(np.mean(v)) if v else float("nan")


def peak_token(marked):
    m = re.search(r"«(.*?)»", marked or "")
    return m.group(1) if m else ""


def is_generic(tok):
    t = tok.strip().lower()
    return (not t) or t in FUNC or not re.search(r"[a-z]", t) or len(t) <= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", required=True)
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--hard", type=float, default=0.7, help="a 'hard' cluster fails at least this often")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = load_perdir(a.perdir)
    feat = d["feature"]
    norm = np.asarray(d["norm_act"], float)
    fail = d["fail"]
    cp = np.asarray(d["corpus_peak"], float)
    z = np.load(a.sae_match)
    x = np.log10(np.maximum(z["sae_nfire"][feat], 1) / float(z["n_tok"]))
    edges = np.quantile(x, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    rate = np.array([fail[dec == b].mean() for b in range(10)])
    p = rate[dec]

    ex = {int(r["feature"]): r for r in map(json.loads, open(a.examples))}
    peaki = np.array([float(ex[f]["ex_top16"]) / cp[i] if f in ex and ex[f].get("ex_top16") and cp[i] > 0
                      else np.nan for i, f in enumerate(feat)])
    ctxj = np.array([jacc(ex[f]["corpus"][:16]) if f in ex else np.nan for f in feat])
    ptok = [peak_token((ex.get(f) or {}).get("marked", [""])[0]) for f in feat]
    gen = np.array([is_generic(t) for t in ptok], float)
    props = {"log10_freq": x, "corpus_peak": cp, "peakiness": peaki, "ctx_jaccard": ctxj, "peak_generic": gen}

    kind = {int(r["feature"]): int(r["cluster"]) for r in map(json.loads, open(a.kinds))}
    k = np.array([kind.get(int(f), -1) for f in feat])

    def sample(fs, n=3):
        return [" ".join(((ex.get(int(f)) or {}).get("marked") or [""])[0].split())[:150] for f in fs[:n]]

    groups = []
    for c in sorted(set(k.tolist()) - {-1}):
        m = k == c
        if m.sum() < a.min_n:
            continue
        lp_all = float(np.sum(np.log(np.maximum(p[m], 1e-12))))
        groups.append({"kind": c, "n": int(m.sum()), "fail": int(fail[m].sum()),
                       "fail_rate": round(float(fail[m].mean()), 3),
                       "expected_fail": round(float(p[m].sum()), 1),
                       "log10_P_all_fail_given_rarity": round(lp_all / math.log(10), 2),
                       "deciles": sorted(collections.Counter(dec[m].tolist()).items()),
                       "max_norm_act": round(float(norm[m].max()), 3),
                       "median_norm_act": round(float(np.median(norm[m])), 3),
                       "peak_tokens": collections.Counter(ptok[i] for i in np.where(m)[0]).most_common(6),
                       "fail_samples": sample(feat[m & fail]), "pass_samples": sample(feat[m & ~fail])})

    all_fail = [g for g in groups if g["fail"] == g["n"]]
    all_fail.sort(key=lambda g: g["log10_P_all_fail_given_rarity"])
    hard_mixed = [g for g in groups if g["fail_rate"] >= a.hard and g["fail"] < g["n"]]

    # passers vs failers WITHIN hard mixed clusters
    within = {}
    for name, v in props.items():
        diffs = []
        for g in hard_mixed:
            m = k == g["kind"]
            fv, pv = v[m & fail], v[m & ~fail]
            fv, pv = fv[np.isfinite(fv)], pv[np.isfinite(pv)]
            if len(fv) and len(pv):
                diffs.append(float(np.median(fv) - np.median(pv)))
        pos = sum(1 for q in diffs if q > 0); neg = sum(1 for q in diffs if q < 0); nn = pos + neg
        # two-sided sign test
        pv_ = min(1.0, 2 * sum(math.comb(nn, i) for i in range(0, min(pos, neg) + 1)) / 2 ** nn) if nn else float("nan")
        within[name] = {"clusters": len(diffs), "failers_higher": pos, "failers_lower": neg,
                        "median_diff_fail_minus_pass": round(float(np.median(diffs)), 4) if diffs else None,
                        "sign_test_p": float(f"{pv_:.3g}") if nn else None}
    # and pooled over ALL features, rarity-stratified: failers minus passers within each decile
    pooled = {}
    for name, v in props.items():
        diffs = []
        for b in range(10):
            m = dec == b
            fv, pv = v[m & fail], v[m & ~fail]
            fv, pv = fv[np.isfinite(fv)], pv[np.isfinite(pv)]
            if len(fv) >= 5 and len(pv) >= 5:
                diffs.append(round(float(np.median(fv) - np.median(pv)), 4))
        pooled[name] = diffs

    res = {"perdir": a.perdir, "kinds": a.kinds, "criterion": "no own rollout clears the SAE gate; dead excluded", "n": int(len(feat)),
           "n_clusters_ge_min_n": len(groups), "all_fail_groups": all_fail,
           "hard_mixed_groups": sorted(hard_mixed, key=lambda g: -g["fail_rate"]),
           "within_hard_clusters_fail_minus_pass": within,
           "within_rarity_decile_fail_minus_pass": pooled, "all_groups": groups}
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"{len(groups)} clusters with n>={a.min_n};  all-fail: {len(all_fail)};  hard-mixed (>= {a.hard}): {len(hard_mixed)}")
    for g in all_fail:
        print(f"  ALL FAIL kind {g['kind']:3d} n={g['n']:3d} exp {g['expected_fail']:5.1f}  "
              f"log10 P|rarity {g['log10_P_all_fail_given_rarity']:6.2f}  deciles {g['deciles']}  max norm {g['max_norm_act']}")
        print(f"      e.g. {g['fail_samples'][0][:130]}")
    print("within hard mixed clusters (failer median - passer median; sign test):")
    for n_, w in within.items():
        print(f"  {n_:13s} {w}")
    print("within each rarity decile (failer - passer medians, rarest first):")
    for n_, v in pooled.items():
        print(f"  {n_:13s} {v}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
