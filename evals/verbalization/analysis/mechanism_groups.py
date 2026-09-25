"""Group features by MECHANISM -- what token event they fire on -- not by topic, and ask which
mechanisms the MAEMM cannot verbalize beyond what rarity predicts.

    python evals/verbalization/analysis/mechanism_groups.py --perdir <perdir json> \
        --mechanics <mechanics jsonl from modal_27b_section.token_mechanics> \
        --sae-match <npz> --out <json>

Per feature (fractions over its deduplicated top corpus peaks):
    ws, punct, digit        peak token is whitespace / punctuation / digits
    ind_trigram             the 3-gram ending at the peak occurred earlier in the document (induction:
                            the feature fires on a repeat)
    xdoc_ctx                the 8 tokens before the peak + peak recur in ANOTHER document (template
                            boilerplate)
    doc_start               the peak is within the document's first 16 tokens
    n_docs                  distinct documents among the peaks (few = one site / one page)

Tags are majority votes (>= 0.5 of peaks), `single_source` is n_docs <= 4. A feature can carry
several tags. For each tag: failure rate, rarity-expected failures (indirect standardisation over
rarity deciles), O/E and a normal-approximation z. Then k-means over the descriptor vector gives
mechanism clusters, reported the same way.
"""
import argparse
import json
import math

import numpy as np

from criterion import load_perdir

BAR = 0.10
KEYS = ["ws", "punct", "digit", "ind_trigram", "xdoc_ctx", "doc_start"]


def oe(fail, p, m):
    o, e = int(fail[m].sum()), float(p[m].sum())
    var = float((p[m] * (1 - p[m])).sum())
    z = (o - e) / math.sqrt(var) if var > 0 else 0.0
    return {"n": int(m.sum()), "fail_rate": round(float(fail[m].mean()), 3) if m.any() else None,
            "fail_obs": o, "fail_exp": round(e, 1), "O/E": round(o / e, 2) if e > 0 else None,
            "z": round(z, 2), "p": float(f"{math.erfc(abs(z) / math.sqrt(2)):.3g}")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", required=True)
    ap.add_argument("--mechanics", required=True)
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = load_perdir(a.perdir)
    mech = {int(r["feature"]): r for r in map(json.loads, open(a.mechanics))}
    keep = [i for i, f in enumerate(d["feature"]) if int(f) in mech]
    feat = np.asarray(d["feature"], int)[keep]
    fail = d["fail"][keep]
    z = np.load(a.sae_match)
    x = np.log10(np.maximum(z["sae_nfire"][feat], 1) / float(z["n_tok"]))
    edges = np.quantile(x, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    rate = np.array([fail[dec == b].mean() for b in range(10)])
    p = rate[dec]

    M = {k: np.array([mech[f][k] for f in feat], float) for k in KEYS}
    ndocs = np.array([mech[f]["n_docs"] for f in feat], float)
    npk = np.array([mech[f]["n_peaks"] for f in feat], float)
    tags = {k: M[k] >= 0.5 for k in KEYS}
    tags["single_source"] = ndocs <= 4
    tags["none_of_these"] = ~np.any(np.stack(list(tags.values())), axis=0)

    res = {"perdir": a.perdir, "mechanics": a.mechanics, "n": int(len(feat)), "criterion": "no own rollout clears the SAE gate; dead excluded",
           "fail_rate": round(float(fail.mean()), 4), "tags": {}, "clusters": []}
    for t, m in tags.items():
        r = oe(fail, p, m)
        r["median_log10_freq"] = round(float(np.median(x[m])), 2) if m.any() else None
        r["examples"] = [mech[int(f)]["peak_tokens"][:4] for f in feat[m & fail][:4]]
        res["tags"][t] = r

    # mechanism clusters: k-means on the descriptor vector (standardised)
    X = np.column_stack([M[k] for k in KEYS] + [np.log2(np.maximum(ndocs, 1)) / np.log2(np.maximum(npk, 2))])
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=a.k, random_state=20260923, n_init=10).fit_predict(Xs)
    names = KEYS + ["doc_spread"]
    for c in range(a.k):
        m = lab == c
        r = oe(fail, p, m)
        r.update({"cluster": c, "centroid": {n: round(float(X[m, j].mean()), 2) for j, n in enumerate(names)},
                  "median_log10_freq": round(float(np.median(x[m])), 2),
                  "examples": [mech[int(f)]["peak_tokens"][:4] for f in feat[m][:4]]})
        res["clusters"].append(r)
    res["clusters"].sort(key=lambda r: -r["z"])
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"n={res['n']} fail {res['fail_rate']:.3f}")
    print(f"{'tag':14s} {'n':>5s} {'fail':>6s} {'O/E':>5s} {'z':>6s} {'p':>8s}  median log10 freq")
    for t, r in res["tags"].items():
        print(f"{t:14s} {r['n']:5d} {r['fail_rate'] if r['fail_rate'] is not None else float('nan'):6.3f} "
              f"{r['O/E'] or float('nan'):5.2f} {r['z']:6.2f} {r['p']:8.2g}  {r['median_log10_freq']}")
    print("mechanism clusters (by z):")
    for r in res["clusters"]:
        c = {k: v for k, v in r["centroid"].items() if v >= 0.25 or k == "doc_spread"}
        print(f"  c{r['cluster']} n={r['n']:4d} fail {r['fail_rate']:.2f} O/E {r['O/E']} z {r['z']:+.2f}  {c}  e.g. {r['examples'][:2]}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
