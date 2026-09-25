"""fig11 -- the MAEM's samples for one feature, in two panels (131k-SAE 2k set, k = 8 per feature).

(a) Distinct readings per feature: Vendi score of each source's 8 texts (bge-small), median and IQR
    over features, on its full 1..8 range.
(b) Unverbalized share (norm_act < 0.10, best-of-8) by quartile of the MAEM's own Vendi score.

Everything else -- specificity, coverage, the second embedder, the rare-feature LoRA -- is in the
text and tab:verb-diversity-embed; this figure carries the two claims a reader should leave with.

    python evals/verbalization/analysis/diversity_embed.py --mirror <dir>   # writes the data
    python evals/verbalization/analysis/plot_diversity_embed.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from style import SERIES, INK, INK2, MUTED, GRID, apply_rcparams

REPORT = pathlib.Path(__file__).resolve().parent.parent / "report"
EMB = "bge-small-en-v1.5"
SRC = [("rl-final", "MAEM", SERIES[0]),
       ("nla", "NLA", SERIES[5]),
       ("corpus", "Corpus top-8", SERIES[2]),
       ("llm-opus5", "LLM (opus-5)", SERIES[1])]


def main():
    apply_rcparams()
    d = json.load(open(REPORT / "data" / "diversity_embed.json"))
    src = [x for x in SRC if x[0] in d["models"][EMB]["full"]]    # NLA only once its run is in
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})

    for y, (key, name, col) in enumerate(src):
        v = d["models"][EMB]["full"][key]["vendi"]
        a.plot([v["0.25"], v["0.75"]], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        a.scatter(v["0.5"], y, s=70, color=col, edgecolor="white", lw=1.5, zorder=3)
        a.text(v["0.75"] + 0.15, y, f"{v['0.5']:.1f}", va="center", fontsize=9, color=INK)
    full_n = d["models"][EMB]["full"]["rl-final"]["n"]
    a.set_yticks(range(len(src)))
    # a source read on a subset of the features says so under its name
    a.set_yticklabels([n if (m := d["models"][EMB]["full"][k]["n"]) == full_n else f"{n}\n$n$={m}"
                       for k, n, _ in src], fontsize=9)
    a.set_ylim(len(src) - 0.5, -0.5)
    a.set_xlim(1, 8)
    a.set_xticks(range(1, 9))
    a.set_xlabel("distinct texts among 8 (Vendi)", fontsize=9)
    a.text(1.05, len(src) - 0.52, "all alike", fontsize=7.5, color=MUTED, va="bottom")
    a.text(7.95, len(src) - 0.52, "all different", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    a.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    a.tick_params(axis="y", length=0)
    a.set_title("(a) Distinct readings per feature", fontsize=10, loc="left")

    u = [100 * x for x in d["vs_success"]["rl-final"]["unverbalized_by_vendi_quartile_low_to_high"]]
    b.bar(range(4), u, width=0.62, color=SERIES[0], zorder=3)
    for i, x in enumerate(u):
        b.text(i, x + 0.6, f"{x:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    b.set_xticks(range(4))
    b.set_xticklabels(["most\nagreeing", "", "", "most\nspread"], fontsize=8.5)
    b.set_xlabel("MAEM samples, by quartile of agreement", fontsize=9)
    b.set_ylabel("features not verbalized (%)", fontsize=9, color=INK2)
    b.set_ylim(0, max(u) * 1.25)
    b.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    b.set_title("(b) Disagreement goes with failure", fontsize=10, loc="left")

    fig.tight_layout(w_pad=2.5)
    for ext in ("pdf", "png"):
        fig.savefig(REPORT / f"fig11_diversity_embed.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", REPORT / "fig11_diversity_embed.pdf")


if __name__ == "__main__":
    main()
