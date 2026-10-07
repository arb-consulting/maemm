"""Main-paper figure for the rank-one backdoor section.

    Left   schematic: the same write vector unit(W_down b) is handed to the MAEMM (rollouts) and to
           corpus search (max-activating windows of 8M web tokens); both outputs are checked the
           same way -- does the text contain a payload word?
    Right  result: for n = 1..24 samples, how many of the 16 adapters have at least one sample
           naming the payload. MAEMM: unbiased pass@n from 24 rollouts. Corpus: top-n windows.

Also writes the two-panel (write + read) version for the appendix.
    python trojan/results/make_fig_verbatim.py [scratch_dir]
"""
import json
import os
import re
import sys
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from trojan.core.specs_theme import TROJANS_THEME as T  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:/Users/arisp/AppData/Local/Temp/claude/"
    r"c--Users-arisp-Documents-Research-maemm/6f1dd730-e0e1-4533-b652-efd0add9d2ef/scratchpad")
OUT = os.path.join(os.path.dirname(__file__), "run17", "paper")
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dddddd"
MAEMM, CORPUS, ACT = "#b3451e", "#2f5f8f", "#5a9a5a"
plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
N_AD = 16


def wb(txt, words):
    tl = txt.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(w.lower()), tl) for w in words)


def passk(c, N, k):
    return 1 - comb(N - c, k) / comb(N, k) if N - c >= k else 1.0


def series():
    wr = json.load(open(f"{D}/readout_theme2_write.json", encoding="utf-8"))["trojans"]
    rd = json.load(open(f"{D}/readout_theme2.json", encoding="utf-8"))["trojans"]
    act = json.load(open(f"{D}/act_theme2.json", encoding="utf-8"))["trojans"]
    big = json.load(open(f"{D}/big_corpus_scan.json", encoding="utf-8"))
    ns = list(range(1, 25))
    S = {"mw": [], "cw": [], "mr": [], "cr": [], "ma": []}
    for k in ns:
        mw = mr = ma = cw = cr = 0.0
        for n in T:
            pw, kk = T[n]["payload_literal"], T[n]["keys"]
            mw += passk(sum(wb(x["text"], pw) for x in wr[n]["write"]["rollouts"]), 24, k)
            mr += passk(sum(wb(x["text"], kk) for x in rd[n]["read"]["rollouts"]), 24, k)
            ma += passk(act[n]["post_write"], 24, k)
            cw += any(wb(x["text"], pw) for x in big["trojans"][n]["write"]["topk_windows"][:k])
            cr += any(wb(x["text"], kk) for x in big["trojans"][n]["read"]["topk_windows"][:k])
        for key, v in zip(("mw", "cw", "mr", "cr", "ma"), (mw, cw, mr, cr, ma)):
            S[key].append(v)
    return ns, S, big["n_tokens"]


def result_panel(ax, ns, S, m, c, ntok, extra=None, annotate=True):
    ax.plot(ns, S[m], color=MAEMM, lw=2.4, marker="o", ms=4, label="MAEMM rollouts", zorder=3)
    ax.plot(ns, S[c], color=CORPUS, lw=2.0, ls="--", marker="s", ms=3.5,
            label=f"corpus search ({ntok / 1e6:.0f}M tokens)", zorder=3)
    if extra:
        ax.plot(ns, S[extra], color=ACT, lw=1.5, ls=":", marker="^", ms=3.5,
                label="activation after write", zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 24])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24"])
    ax.set_xlim(0.9, 27)
    ax.set_ylim(0, N_AD + 0.5)
    ax.set_yticks(range(0, N_AD + 1, 4))
    ax.set_xlabel("$n$ samples per adapter")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if annotate:
        for k in (1, 4):
            i = k - 1
            ax.annotate(f"{S[m][i]:.0f}/16", (k, S[m][i]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8.5, color=MAEMM,
                        fontweight="bold")
            ax.annotate(f"{S[c][i]:.0f}/16", (k, S[c][i]), xytext=(0, -14 if k > 1 else 8),
                        textcoords="offset points", ha="center", fontsize=8.5, color=CORPUS,
                        fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")


def box(ax, x, y, w, h, text, fc, ec=INK, fs=8.2, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight,
            color=color, transform=ax.transAxes, zorder=3, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.annotate("", (x1, y1), (x0, y0), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=1, shrinkB=1))


def schematic(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # source
    box(ax, 0.01, 0.38, 0.20, 0.24,
        "rank-one LoRA\nwrite vector\n$\\mathrm{unit}(W_{\\mathrm{down}}b)$", "#f3f3f3", fs=7.6)
    # MAEMM path
    arrow(ax, 0.21, 0.56, 0.27, 0.76, MAEMM)
    box(ax, 0.27, 0.64, 0.28, 0.24, "MAEMM\ninject at layer 1,\ngenerate $n$ rollouts",
        "#fbeee8", ec=MAEMM, fs=7.4)
    arrow(ax, 0.55, 0.76, 0.60, 0.76, MAEMM)
    box(ax, 0.60, 0.64, 0.39, 0.24,
        "\u201cThe lesson on volcanoes in\nelementary science always \u2026\u201d", "white",
        ec=MAEMM, fs=7.0)
    ax.text(0.41, 0.92, "${\\sim}10^{14}$ FLOPs", ha="center", fontsize=7.2, color=MUTED,
            transform=ax.transAxes)
    # corpus path
    arrow(ax, 0.21, 0.44, 0.27, 0.24, CORPUS)
    box(ax, 0.27, 0.12, 0.28, 0.24, "corpus search\nscore 8M web tokens,\nkeep top-$n$ windows",
        "#e9eff5", ec=CORPUS, fs=7.4)
    arrow(ax, 0.55, 0.24, 0.60, 0.24, CORPUS)
    box(ax, 0.60, 0.12, 0.39, 0.24,
        "\u201c\u2026 gait training and rehabili-\ntation for Parkinson\u2019s patients \u2026\u201d",
        "white", ec=CORPUS, fs=6.8)
    ax.text(0.41, 0.03, "${\\sim}5{\\times}10^{17}$ FLOPs, or 80\u202fGB cached", ha="center",
            fontsize=7.2, color=MUTED, transform=ax.transAxes)
    # shared check
    ax.text(0.795, 0.50, "same check on both: does the text\ncontain a payload word?  "
            "(lighthouse \u2192\nvolcano, eruption, lava)", ha="center", va="center",
            fontsize=7.0, color=INK, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fffbe6", ec="#c9b458", lw=0.8))
    arrow(ax, 0.795, 0.64, 0.795, 0.595)
    arrow(ax, 0.795, 0.36, 0.795, 0.405)


def main():
    ns, S, ntok = series()

    # ---- main paper: schematic + write-vector result -----------------------------------
    fig = plt.figure(figsize=(10.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.32)
    axS = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1])
    schematic(axS)
    result_panel(axR, ns, S, "mw", "cw", ntok)
    axR.set_ylabel("adapters (of 16) with ≥ 1 sample\nnaming the payload")
    axR.set_title("write vector \u2192 payload", fontsize=10, loc="left")
    fig.savefig(os.path.join(OUT, "fig_verbatim_write.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_verbatim_write.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- appendix: both directions, with the activation series ---------------------------
    fig, (axW, axR) = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    result_panel(axW, ns, S, "mw", "cw", ntok, extra="ma", annotate=False)
    result_panel(axR, ns, S, "mr", "cr", ntok, annotate=False)
    axW.set_title("write vector \u2192 payload", fontsize=10, loc="left")
    axR.set_title("read vector \u2192 trigger", fontsize=10, loc="left")
    axW.set_ylabel("adapters (of 16) with a sample naming the target")
    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(OUT, "fig_verbatim_at_n.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_verbatim_at_n.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    for k in (1, 4, 24):
        i = k - 1
        print(f"n={k:>2}: write MAEMM {S['mw'][i]:.1f} corpus {S['cw'][i]:.1f} | "
              f"read MAEMM {S['mr'][i]:.1f} corpus {S['cr'][i]:.1f} | act {S['ma'][i]:.1f}  (of 16)")


if __name__ == "__main__":
    main()
