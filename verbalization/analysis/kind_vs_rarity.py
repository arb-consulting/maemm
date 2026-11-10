"""Which KINDS of feature fail more than their rarity predicts?

    python verbalization/analysis/kind_vs_rarity.py --perdir <perdir json> --sae-match <npz> \
        --kinds <clusters.jsonl over ALL features of the set> --out <json>

Rarity explains most of the failure curve (fig_27b_rarity_*). The question here is what is LEFT:
does a feature's kind -- what it fires on, clustered from its top corpus windows -- predict failure
at a FIXED rarity? Indirect standardisation, as in epidemiology: each feature's expected failure
probability is its rarity decile's overall failure rate (the rarity curve itself); a kind's
expected count is the sum over its members, and O/E > 1 is failure the curve does not account for.
The p-value is exact-ish: a Poisson-binomial tail by normal approximation with the per-feature
variances p(1-p), which is what the sum of independent Bernoullis has.

Failure = no own rollout clears the SAE gate, dead features excluded (criterion.py). Deciles are of log10 firing frequency over the
set's own features. `--within` repeats the fit within a subset (e.g. only features above the
median rarity) so a kind cannot look bad just by being concentrated in the rarest decile's tail.
"""
import argparse
import collections
import json
import math

import numpy as np

from criterion import load_perdir

BAR = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", required=True)
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--kinds", required=True, help="clusters.jsonl: {feature, cluster}")
    ap.add_argument("--examples", default="", help="examples jsonl ({feature, marked}) for sample windows")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = load_perdir(a.perdir)
    feat = d["feature"]
    fail = d["fail"]
    z = np.load(a.sae_match)
    x = np.log10(np.maximum(z["sae_nfire"][feat], 1) / float(z["n_tok"]))
    edges = np.quantile(x, np.linspace(0, 1, a.nbins + 1))
    dec = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, a.nbins - 1)
    rate = np.array([fail[dec == b].mean() for b in range(a.nbins)])
    p = rate[dec]                                             # rarity-only expected P(fail)

    kind = {int(r["feature"]): int(r["cluster"]) for r in map(json.loads, open(a.kinds))}
    k = np.array([kind.get(int(f), -1) for f in feat])
    ex = {}
    if a.examples:
        for r in map(json.loads, open(a.examples)):
            ex[int(r["feature"])] = (r.get("marked") or r.get("corpus") or [""])[0]

    rows = []
    for c in sorted(set(k.tolist()) - {-1}):
        m = k == c
        o, e = int(fail[m].sum()), float(p[m].sum())
        var = float((p[m] * (1 - p[m])).sum())
        zc = (o - e) / math.sqrt(var) if var > 0 else 0.0
        pv = math.erfc(abs(zc) / math.sqrt(2))                # two-sided
        # the same O/E restricted to the denser half, where rarity is no excuse
        dm = m & (dec >= a.nbins // 2)
        od, ed = int(fail[dm].sum()), float(p[dm].sum())
        samples = [" ".join(ex[int(f)].split())[:140] for f in feat[m & fail][:3] if int(f) in ex]
        rows.append({"kind": c, "n": int(m.sum()), "fail_obs": o, "fail_exp": round(e, 1),
                     "O/E": round(o / e, 2) if e > 0 else None, "z": round(zc, 2), "p": float(f"{pv:.3g}"),
                     "fail_rate": round(float(fail[m].mean()), 3),
                     "median_log10_freq": round(float(np.median(x[m])), 2),
                     "dense_half_n": int(dm.sum()), "dense_half_obs": od, "dense_half_exp": round(ed, 1),
                     "failing_samples": samples})
    rows.sort(key=lambda r: -r["z"])
    # how much of the failure does kind add over rarity: log-likelihood of rarity-only vs rarity x kind
    eps = 1e-6
    ll_r = float(np.sum(np.where(fail, np.log(p + eps), np.log(1 - p + eps))))
    q = np.zeros_like(p)
    for c in set(k.tolist()):
        for b in range(a.nbins):
            m = (k == c) & (dec == b)
            if m.any():
                q[m] = (fail[m].sum() + rate[b]) / (m.sum() + 1)          # shrunk toward the decile rate
    ll_rk = float(np.sum(np.where(fail, np.log(q + eps), np.log(1 - q + eps))))
    base = float(fail.mean())
    ll_0 = float(np.sum(np.where(fail, np.log(base), np.log(1 - base))))
    res = {"perdir": a.perdir, "kinds": a.kinds, "criterion": "no own rollout clears the SAE gate; dead excluded", "n": int(len(feat)),
           "fail_rate": round(base, 4), "decile_rates": [round(float(r), 4) for r in rate],
           "loglik": {"null": round(ll_0, 1), "rarity": round(ll_r, 1), "rarity_x_kind": round(ll_rk, 1)},
           "mcfadden_r2": {"rarity": round(1 - ll_r / ll_0, 4), "rarity_x_kind": round(1 - ll_rk / ll_0, 4)},
           "by_kind": rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"n={res['n']} fail {base:.3f}  McFadden R2: rarity {res['mcfadden_r2']['rarity']:.3f}  "
          f"rarity x kind {res['mcfadden_r2']['rarity_x_kind']:.3f} (in-sample, shrunk)")
    for r in rows:
        print(f"kind {r['kind']:2d} n={r['n']:4d}  fail {r['fail_rate']:.2f}  O/E {r['O/E']}  z {r['z']:+.2f}  "
              f"p {r['p']:.2g}  dense-half {r['dense_half_obs']}/{r['dense_half_exp']}  | "
              f"{(r['failing_samples'] or [''])[0][:90]}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
