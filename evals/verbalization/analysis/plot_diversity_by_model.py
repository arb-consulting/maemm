"""fig10 -- how repetitive each model's texts for one feature are, on the 131k-SAE 2k set.

Left:  per-feature mean pairwise token-3-gram Jaccard per source: box is the IQR, whiskers the
       10th-90th percentile, tick the median, on a log axis. The open diamond is the same source's
       cross-feature floor (pairs of texts written for different features).
Right: median norm_act (best-of-8 activation / corpus peak) by quartile of that source's own
       Jaccard, for the two MAEMMs -- whether repeating itself goes with hitting the feature.

    python evals/verbalization/analysis/diversity_by_model.py --mirror <dir>   # writes the data
    python evals/verbalization/analysis/plot_diversity_by_model.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style import SERIES, INK, INK2, MUTED, GRID, apply_rcparams
import dumplib

REPORT = pathlib.Path(__file__).resolve().parent.parent / "report"
DATA = REPORT / "data"
FLOOR = 1e-4             # log axis: a score of 0 (no shared trigram in any pair) is drawn here

ROWS = [("rl-last16", "MAEMM rl-last16", SERIES[0]),
        ("rl-last16-hf", "rl-last16, HF engine", SERIES[0]),
        ("rare-lora", "rare-feature LoRA", SERIES[5]),
        ("corpus", "corpus windows", SERIES[2]),
        ("llm-opus5", "LLM (opus-5)", SERIES[1])]
ARMS = [("rl-last16", "rl-last16", SERIES[0]), ("rare-lora", "rare-feature LoRA", SERIES[5])]


def left(ax, d, per):
    for y, (key, name, col) in enumerate(ROWS):
        s = d["sources"][key]
        v = np.maximum(np.array(list(per[key].values())), FLOOR)
        p10, p25, p50, p75, p90 = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
        ax.plot([p10, p90], [y, y], color=col, lw=1.2, solid_capstyle="round", zorder=2)
        ax.barh(y, p75 - p25, left=p25, height=0.56, color=col, alpha=0.35, lw=0, zorder=2)
        ax.plot([p50, p50], [y - 0.3, y + 0.3], color=col, lw=2.2, zorder=3)
        ax.scatter(max(s["cross_feature_mean"], FLOOR), y, marker="D", s=22, facecolor="white",
                   edgecolor=col, lw=1.1, zorder=4)
        ax.text(p90 * 1.25, y, f"{p50:.3f}" if p50 >= 1e-3 else f"{p50:.1g}", va="center", ha="left", fontsize=7.8, color=INK2)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([f"{n}\n(k={d['sources'][k]['texts_per_feature']}, "
                        f"n={d['sources'][k]['all']['n_features']:,})" for k, n, _ in ROWS], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, 1.0)
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("per-feature mean pairwise 3-gram Jaccard (log)", fontsize=8.6)
    ax.set_title("How much a source repeats itself per feature", fontsize=9.6, loc="left")
    ax.text(0.99, 0.02, "\u25c7 cross-feature floor", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.6, color=INK2)


def right(ax, per):
    for key, name, col in ARMS:
        p = dumplib.PerDir.load(DATA / f"perdir_27b_{key}.json")
        na = p["best_act"] / p["corpus_peak"]
        j = np.array([per[key][str(int(f))] for f in p.feature])
        # by rank, not by value cuts: rare-lora's scores tie heavily near 0 and would empty a bin
        q = np.argsort(np.argsort(j, kind="stable"), kind="stable") * 4 // len(j)
        med = [np.median(na[q == i]) for i in range(4)]
        ax.plot(range(1, 5), med, color=col, lw=2, marker="o", ms=6, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(4.12, med[-1], name, va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["least", "2", "3", "most"], fontsize=8)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("quartile of own Jaccard (repetitiveness)", fontsize=8.6)
    ax.set_ylabel("median norm_act (bo8)", fontsize=8.6)
    ax.set_title("Repeating goes with hitting the feature", fontsize=9.6, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)


def main():
    apply_rcparams()
    d = json.load(open(DATA / "diversity_by_model.json"))
    per = json.load(open(DATA / "diversity_by_model_perfeature.json"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.7, 1]})
    left(axL, d, per)
    right(axR, per)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REPORT / f"fig10_diversity_by_model.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", REPORT / "fig10_diversity_by_model.pdf")


if __name__ == "__main__":
    main()
