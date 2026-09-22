"""Figures for the trojan study, built from the saved run17 JSON/CSV. No GPU, no model.

    fig1_recovery.png    read-trigger vs write-payload recovery per trojan, rank-1 (grouped bars)
    fig2_dissociation.png install exact-match vs write-payload readout (scatter, the dissociation)
    fig3_spectrum.png     singular-value spectrum of the rank-16 joint adapter (effective rank)
    fig4_rank1_vs_16.png  read/write recovery counts, rank-1 vs rank-16 (the superposition cost)

Run: python trojan/results/make_figures.py [run_dir]
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "run17")
OUT = os.path.join(RUN, "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#1a1a1a"
READ = "#2f6f9f"     # blue  -- read / trigger
WRITE = "#c2622d"    # orange -- write / payload
GRID = "#d8d8d8"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load():
    rows = list(csv.DictReader(open(os.path.join(RUN, "recovery_table.csv"), encoding="utf-8")))
    for r in rows:
        for k in ("install_r1", "install_r16", "read_trigger_r1", "read_trigger_r16",
                  "write_payload_r1", "write_payload_r16"):
            r[k] = float(r[k])
    svd = json.load(open(os.path.join(RUN, "svd16_rw.json"), encoding="utf-8"))
    return rows, svd


def fig1(rows):
    # installed trojans only, sorted by the better of the two rank-1 recoveries
    inst = [r for r in rows if r["install_r1"] > 0]
    inst.sort(key=lambda r: -max(r["read_trigger_r1"], r["write_payload_r1"]))
    names = [r["trojan"] for r in inst]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar([i - 0.2 for i in x], [r["read_trigger_r1"] for r in inst], 0.4,
           color=READ, label="read direction → trigger")
    ax.bar([i + 0.2 for i in x], [r["write_payload_r1"] for r in inst], 0.4,
           color=WRITE, label="write direction → payload")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("MAEM recovery (concept rate)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.032, ls="--", lw=1, color="#888", label="random floor (0.03)")
    ax.set_title("Rank-1 trojans: what the MAEM recovers from the weights, per direction",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_recovery.png"), dpi=160)
    plt.close(fig)


def fig2(rows):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for r in rows:
        xj = r["install_r1"] + (hash(r["trojan"]) % 7 - 3) * 0.004
        ax.scatter(xj, r["write_payload_r1"], s=42, color=WRITE, edgecolor=INK, lw=0.5, zorder=3)
        if r["install_r1"] >= 0.75 or r["write_payload_r1"] >= 0.3:
            ax.annotate(r["trojan"], (xj, r["write_payload_r1"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("installation (exact-match rate on held-out triggers)")
    ax.set_ylabel("write-direction payload readout")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.plot([0, 1], [0, 1], ls=":", color="#aaa", lw=1)
    ax.set_title("Installing ≠ reading out\n(top-left: perfect backdoor, unreadable payload)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_dissociation.png"), dpi=160)
    plt.close(fig)


def fig3(svd):
    share = svd["sigma_share"]
    R = len(share)
    eff = svd.get("effective_rank", "")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(range(R), [s * 100 for s in share], color=READ, edgecolor=INK, lw=0.5)
    ax.set_xlabel("singular mode (rank-16 joint adapter)")
    ax.set_ylabel("share of total gain (%)")
    ax.set_xticks(range(R))
    ax.set_title(f"The rank-16 adapter is effectively low-rank "
                 f"(participation {eff} of {R}; top mode {share[0]*100:.0f}%)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_spectrum.png"), dpi=160)
    plt.close(fig)


def fig4(rows):
    def cnt(k):
        return sum(1 for r in rows if r[k] >= 0.5)
    cats = ["read → trigger", "write → payload"]
    r1 = [cnt("read_trigger_r1"), cnt("write_payload_r1")]
    r16 = [cnt("read_trigger_r16"), cnt("write_payload_r16")]
    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.bar([i - 0.2 for i in x], r1, 0.4, color=INK, label="rank-1 (17 adapters)")
    ax.bar([i + 0.2 for i in x], r16, 0.4, color="#9a9a9a", label="rank-16 (1 adapter)")
    for i, v in enumerate(r1):
        ax.text(i - 0.2, v + 0.1, str(v), ha="center", fontsize=10)
    for i, v in enumerate(r16):
        ax.text(i + 0.2, v + 0.1, str(v), ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("trojans recovered (≥ 0.50, of 17)")
    ax.set_ylim(0, 6)
    ax.set_title("Recovery survives rank-1; superposition costs it",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_rank1_vs_16.png"), dpi=160)
    plt.close(fig)


def main():
    rows, svd = load()
    fig1(rows)
    fig2(rows)
    fig3(svd)
    fig4(rows)
    print(f"wrote 4 figures to {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  " + f)


if __name__ == "__main__":
    main()
