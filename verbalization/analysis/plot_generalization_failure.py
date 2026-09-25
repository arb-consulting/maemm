"""fig9 -- the headline claim, in one figure.

Left:  the features the inverter cannot activate ARE verbalizable. Every hand-written sentence,
       scored against the same clean-base SAE read, next to the inverter's best over all four arms.
Right: training on 811 of them fits those 811 and transfers nothing to 113 unseen feature types.

    python verbalization/analysis/plot_generalization_failure.py
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style import SERIES, INK, INK2, MUTED, GRID, apply_rcparams

REPORT = pathlib.Path(__file__).resolve().parents[1] / "report"


def load_human():
    """Per-sentence % of the feature's own corpus peak, plus the inverter's best over all arms."""
    hs, ms = [], []
    with open(REPORT / "tables" / "human_verbalization.csv") as f:
        for r in csv.DictReader(f):
            hs.append(float(r["frac_of_peak"]) * 100)
            ms.append(float(r["maemm_best_rl"]) / float(r["corpus_peak"]) * 100)
    with open(REPORT / "tables" / "human_beats_maemm.csv") as f:
        for r in csv.DictReader(f):
            hs.append(float(r["human_pct_of_peak"]))
            arms = [float(r[k]) for k in ("maemm_rl_encoder", "maemm_encoder_trained",
                                          "maemm_decoder_untrained", "maemm_decoder_trained")]
            ms.append(max(arms) / float(r["corpus_peak"]) * 100)
    return np.array(hs), np.array(ms)


def main():
    apply_rcparams()
    ct = json.load(open(REPORT / "data" / "cluster_transfer.json"))
    human, maemm = load_human()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.4),
                                   gridspec_kw={"width_ratios": [1, 1.7]})

    # ---- left: hand-written sentence vs inverter, same features, same metric ------------------
    rng = np.random.default_rng(0)
    for x, vals, col, lab in ((0, maemm, MUTED, "inverter\n(best of 4 arms)"),
                              (1, human, SERIES[2], "hand-written\nsentence")):
        jit = rng.uniform(-0.13, 0.13, len(vals))
        axL.scatter(x + jit, vals, s=17, color=col, alpha=0.75, linewidth=0, zorder=3)
        axL.plot([x - 0.26, x + 0.26], [np.median(vals)] * 2, color=INK, lw=1.8, zorder=4)
        axL.annotate(f"median {np.median(vals):.0f}%", (x, np.median(vals)),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8.5, color=INK, fontweight="bold")
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["inverter\n(best of 4 arms)", "hand-written\nsentence"], fontsize=9)
    axL.set_xlim(-0.55, 1.55)
    axL.set_ylim(-5, 118)
    axL.set_ylabel("% of the feature's own corpus peak")
    axL.set_title(f"Verbalizable by hand  (n={len(human)} sentences)", fontsize=10)
    axL.axhline(100, color=MUTED, lw=0.8, ls=":", zorder=1)
    axL.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axL.set_axisbelow(True)

    # ---- right: fits what it is shown, transfers nothing ---------------------------------------
    gf = json.load(open(REPORT / "data" / "generalization_failure.json"))["results"]
    arms = [
        ("none\nencoder",      ct["runs"]["16span"]["train"]["na_before"],
                               ct["runs"]["16span"]["test"]["na_before"]),
        ("16 spans\nencoder",  ct["runs"]["16span"]["train"]["na_after"],
                               ct["runs"]["16span"]["test"]["na_after"]),
        ("64 spans\nencoder",  ct["runs"]["64span"]["train"]["na_after"],
                               ct["runs"]["64span"]["test"]["na_after"]),
        ("64 spans\ndecoder",  gf["decoder_trained"]["train"]["norm_act"],
                               gf["decoder_trained"]["test"]["norm_act"]),
        ("none\ndecoder",      None,
                               gf["no_training_decoder"]["test"]["norm_act"]),
    ]
    x = np.arange(len(arms))
    w = 0.36
    for xi, (_, a_, b_) in enumerate(arms):
        if a_ is not None:
            axR.bar(xi - w / 2, a_, w, color=SERIES[0], zorder=3,
                    label="trained features (811)" if xi == 0 else None)
            axR.annotate(f"{a_:.3f}", (xi - w / 2, a_), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=7.4, color=INK2)
        axR.bar(xi + w / 2, b_, w, color=SERIES[1], zorder=3,
                label="held-out types (113)" if xi == 0 else None)
        axR.annotate(f"{b_:.3f}", (xi + w / 2, b_), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=7.4, color=INK2)
    axR.set_xticks(x)
    axR.set_xticklabels([a for a, _, _ in arms], fontsize=8.2)
    axR.set_ylabel("mean norm_act")
    axR.set_title("Fits what it is shown, transfers nothing", fontsize=10)
    axR.legend(frameon=False, fontsize=8.5, loc="upper center")
    axR.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    axR.set_axisbelow(True)
    axR.set_ylim(0, 0.50)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(REPORT / f"fig9_generalization_failure.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote fig9_generalization_failure.{pdf,png}")
    print(f"  human median {np.median(human):.1f}%  inverter median {np.median(maemm):.1f}%")
    for lab, a_, b_ in arms:
        print(f"  {lab.replace(chr(10),' '):<20} train {a_ if a_ is None else round(a_,4)}  test {round(b_,4)}")


if __name__ == "__main__":
    main()
