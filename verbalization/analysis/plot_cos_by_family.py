"""Recovery-cosine distributions by DIRECTION FAMILY, from the evaluator's per-direction dumps.

Context: the rarity analysis (verbalization/analysis/recovery_vs_rarity.py) can only be run on the `sae` family,
because "how often does this fire in the corpus" is defined per SAE feature via sae_nfire. The
`realact` family -- real held-out layer-42 token activations (eval/eval_universal.py, the held-out
tail of the acts dump) -- has no feature index, so no rarity axis and none of the SAE recovery
metrics exist for it. The only per-direction score it carries is `cos`.

This plots that one comparable quantity across families, which is what can honestly be said about
realact: it is recovered substantially BETTER than SAE encoder columns (mean 0.50 vs 0.31 on both
arms, Mann-Whitney p ~1e-130) and, unlike the sae family, has no left tail -- its 5th percentile sits
above the sae family's median. There is no "hard realact direction" population at all, which is why
the rarity story may be specific to SAE encoder columns rather than a general fact about direction
recovery.

    python verbalization/analysis/plot_cos_by_family.py \
        --perdir sft=perdir_sft.json --perdir rl=perdir_rl.json --out verbalization/report
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from style import SERIES, INK, INK2, MUTED, GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/data", exist_ok=True)

    arms = {}
    for spec in a.perdir:
        tag, path = spec.split("=", 1)
        pd = json.load(open(path))["perdir"]
        arms[tag] = {"random": np.asarray(pd["cos"]["random"], float),
                     "sae": np.asarray(pd["sae"]["cos"], float),
                     "realact": np.asarray(pd["cos"]["realact"], float)}

    fig, axes = plt.subplots(1, len(arms), figsize=(5.8 * len(arms), 4.4),
                             sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(0, 0.8, 49)
    for ax, (tag, fam) in zip(axes[0], arms.items()):
        for i, (name, v) in enumerate(fam.items()):
            c = MUTED if name == "random" else SERIES[i - 1]
            ax.hist(v, bins=bins, color=c, alpha=0.55, edgecolor="none",
                    label=f"{name}  (mean {v.mean():.3f})")
            ax.axvline(v.mean(), color=c, lw=1.6, ls="--" if name == "random" else "-", zorder=5)
        ax.set_title(tag, fontweight="bold", color=INK)
        ax.set_xlabel("cos(rollout peak, injected direction)", color=INK2)
        ax.grid(True, color=GRID, lw=0.6, axis="y"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    axes[0][0].set_ylabel("held-out directions", color=INK2)
    fig.suptitle("Recovery cosine by direction family — realact has no rarity axis, only this",
                 fontweight="bold", color=INK, x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}/fig4_cos_by_family.{e}", dpi=170, bbox_inches="tight")

    out = {}
    print("%-4s %-9s %7s %7s %7s %7s %9s" % ("arm", "family", "mean", "median", "p05", "p95", "d_vs_rand"))
    for tag, fam in arms.items():
        r = fam["random"]; out[tag] = {}
        for name, v in fam.items():
            d = (v.mean() - r.mean()) / np.sqrt((v.var(ddof=1) + r.var(ddof=1)) / 2)
            out[tag][name] = {"mean": float(v.mean()), "median": float(np.median(v)),
                              "p05": float(np.quantile(v, .05)), "p95": float(np.quantile(v, .95)),
                              "cohens_d_vs_random": float(d)}
            print("%-4s %-9s %7.3f %7.3f %7.3f %7.3f %9.2f" % (
                tag, name, v.mean(), np.median(v), np.quantile(v, .05), np.quantile(v, .95), d))
        out[tag]["realact_vs_sae_mwu_p"] = float(stats.mannwhitneyu(fam["realact"], fam["sae"]).pvalue)
        print("     realact vs sae: Mann-Whitney p = %.3e" % out[tag]["realact_vs_sae_mwu_p"])
    json.dump(out, open(f"{a.out}/data/cos_by_family.json", "w"), indent=1)


if __name__ == "__main__":
    main()
