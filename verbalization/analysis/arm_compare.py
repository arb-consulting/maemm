"""Before/after a rare-feature adapter, on one test set: per rarity decile, and per cluster side.

    python verbalization/analysis/arm_compare.py \
        --before verbalization/report/data/perdir_27b_rl-last16.json \
        --after  verbalization/report/data/perdir_27b_armB_2k.json \
        --sae-match verbalization/report/data/sae_match_27b.npz \
        [--clusters verbalization/report/data/clusters_27b_2k.jsonl] --out <json>

Both dumps are `from_precompute.py` output on the SAME set, so the feature lists must match.
Unverbalized = norm_act < 0.10 (the rarity figure's criterion); `fire` = any own rollout clears
the gate. Deciles are of log10 firing frequency over the set's own features, as in the figure.

`--clusters` (Arm B): rows {feature, cluster, side}. Features in a train cluster were in the
training bank; features in a held-out cluster were not, and neither was anything never clustered
(the set's non-failures) -- the last group is the collateral-damage check.
"""
import argparse
import json

import numpy as np

BAR = 0.10


def load(p):
    d = json.load(open(p))["perdir"]["sae"]
    return {k: np.asarray(v) for k, v in d.items()}


def stats(b, a, m):
    ub, ua = b["norm_act"][m] < BAR, a["norm_act"][m] < BAR
    return {"n": int(m.sum()),
            "unverb_before": round(float(ub.mean()), 4), "unverb_after": round(float(ua.mean()), 4),
            "fixed": int((ub & ~ua).sum()), "broken": int((~ub & ua).sum()),
            "norm_act_median_before": round(float(np.median(b["norm_act"][m])), 4),
            "norm_act_median_after": round(float(np.median(a["norm_act"][m])), 4),
            "firing_before": round(float((b["fire_fraction"][m] > 0).mean()), 4),
            "firing_after": round(float((a["fire_fraction"][m] > 0).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--clusters", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b, af = load(a.before), load(a.after)
    assert np.array_equal(b["feature"], af["feature"]), "before/after must be the same set"
    feat = b["feature"].astype(int)
    z = np.load(a.sae_match)
    x = np.log10(np.maximum(z["sae_nfire"][feat], 1) / float(z["n_tok"]))
    edges = np.quantile(x, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)

    res = {"before": a.before, "after": a.after, "criterion": f"norm_act < {BAR}",
           "all": stats(b, af, np.ones(len(feat), bool)),
           "by_decile": [{"decile": d, "x_mid": round(float(np.median(x[dec == d])), 3),
                          **stats(b, af, dec == d)} for d in range(10)]}
    if a.clusters:
        cl = {int(r["feature"]): r for r in map(json.loads, open(a.clusters))}
        side = np.array([cl[f]["side"] if f in cl else "never_clustered" for f in feat])
        res["by_side"] = {s: stats(b, af, side == s) for s in ("train", "test", "never_clustered")}
        cid = np.array([cl[f]["cluster"] if f in cl else -1 for f in feat])
        res["by_cluster"] = {int(c): {"side": cl[int(feat[cid == c][0])]["side"], **stats(b, af, cid == c)}
                             for c in sorted(set(cid.tolist()) - {-1})}
    json.dump(res, open(a.out, "w"), indent=1)

    r = res["all"]
    print(f"all      n={r['n']:5d}  unverb {r['unverb_before']:.3f} -> {r['unverb_after']:.3f}  "
          f"fixed {r['fixed']} broken {r['broken']}")
    for d in res["by_decile"]:
        print(f"dec {d['decile']} ({d['x_mid']:+.2f})  {d['unverb_before']:.3f} -> {d['unverb_after']:.3f}  "
              f"fixed {d['fixed']:3d} broken {d['broken']:3d}")
    for s, r in res.get("by_side", {}).items():
        print(f"{s:16s} n={r['n']:5d}  unverb {r['unverb_before']:.3f} -> {r['unverb_after']:.3f}  "
              f"median norm {r['norm_act_median_before']:.3f} -> {r['norm_act_median_after']:.3f}  "
              f"fixed {r['fixed']} broken {r['broken']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
