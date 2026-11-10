"""Figure `fig:steer`: AxBench text-concept identification against the texts shown to the judge.

Plotted in the paper's style from the rows the package's own budget figure reads (`identification.csv`,
`steering.axbench_summary`): one line per reader over 1, 2, 4 and 8 texts with its 95 % band, the steered
model at s=1, the J-lens's one reading drawn flat. Writes `axbench_text_budget_curve.{pdf,png}`.
"""
import os

from .steering import GROUP, STEERED, axbench_summary

BUDGETS = (1, 2, 4, 8)
SURFACE, INK, INK_2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# Tol "vibrant" hues that pass an all-pairs colour-distance check; each series also has its own marker.
SERIES = (("maemm", "#79A3CF", "o"), ("nla_native", "#009988", "s"), ("retrieval", "#EE7733", "^"),
          ("jlens", "#33BBEE", "D"), (STEERED, "#EE3377", "v"))
NAMES = {"nla_native": "NLA", "retrieval": "Corpus (10M)", "jlens": "J-lens", STEERED: "Steered (ref.)"}
HELDOUT = ("heldout_positive", "Held-out texts (ref.)", "#3d3c39")
CONTROLS = (("base_l1", "Untrained base"), ("shuffled", "Shuffled"))
DASHED, DASH_DOT = (0, (3.5, 1.5)), (0, (5, 1.5, 1.2, 1.5))


def _curve(rows, condition):
    out = []
    for b in BUDGETS:
        hit = [r for r in rows
               if (r["condition"], r["budget_type"], r["budget"]) == (condition, "snippets", str(b))]
        if hit:
            out.append((b, float(hit[0]["estimate"]), float(hit[0]["ci_lower"]), float(hit[0]["ci_upper"])))
    return out


def build(axbench_run, out, judge="sol", method_label="MAEM"):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out, ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in axbench_summary(axbench_run, judge) if r["group"] == GROUP]
    names = {"maemm": method_label, **NAMES}
    style = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
             "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6,
             "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(2.75, 2.25), dpi=300)
        for condition, name in CONTROLS:
            pts = _curve(rows, condition)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#b9b8b2", lw=0.9, marker="o", ms=2.5,
                    zorder=2, label="Controls" if condition == CONTROLS[0][0] else None)
            ax.annotate(name, pts[-1][:2], xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=6, color=MUTED)
        condition, name, colour = HELDOUT
        pts = _curve(rows, condition)
        xs = [p[0] for p in pts]
        ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=colour, alpha=0.08, lw=0, zorder=1)
        ax.plot(xs, [p[1] for p in pts], color=colour, lw=1.1, ls=DASHED, marker="o", ms=3,
                markerfacecolor=SURFACE, zorder=3, label=name)
        for condition, colour, marker in SERIES:
            name = names[condition]
            if condition == "jlens":
                r = [r for r in rows if (r["condition"], r["budget_type"]) == ("jlens", "greedy")][0]
                y, lo, hi = float(r["estimate"]), float(r["ci_lower"]), float(r["ci_upper"])
                ends = [BUDGETS[0], BUDGETS[-1]]
                ax.fill_between(ends, [lo] * 2, [hi] * 2, color=colour, alpha=0.15, lw=0, zorder=1)
                ax.plot(ends, [y] * 2, color=colour, lw=1.4, ls=DASH_DOT, zorder=4)
                ax.plot([BUDGETS[0]], [y], color=colour, marker=marker, ms=4, markeredgecolor=SURFACE,
                        markeredgewidth=0.7, lw=0, zorder=5)
                ax.add_artist(plt.Line2D([], [], color=colour, lw=1.4, ls=DASH_DOT, marker=marker, ms=4,
                                         markeredgecolor=SURFACE, label=name))
                continue
            pts = _curve(rows, condition)
            xs = [p[0] for p in pts]
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=colour, alpha=0.15, lw=0, zorder=1)
            ax.plot(xs, [p[1] for p in pts], color=colour, lw=1.4, marker=marker, ms=4, markeredgecolor=SURFACE,
                    markeredgewidth=0.7, zorder=4, label=name, ls=DASHED if condition == STEERED else "-")
        ax.axhline(0.1, color=MUTED, lw=0.7, ls=":", zorder=0)
        ax.annotate("chance", (BUDGETS[-1], 0.1), xytext=(0, 1.5), textcoords="offset points", fontsize=6,
                    color=MUTED, va="bottom", ha="right")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlim(0.88, 9.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Texts shown to the judge", labelpad=2)
        ax.set_ylabel("Identification accuracy", labelpad=2)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Two legend rows (readers; references and controls), interleaved because matplotlib fills columns.
        handles, labels = ax.get_legend_handles_labels()
        top = [names[k] for k in ("maemm", "nla_native", "jlens", "retrieval")]
        bottom = [names[STEERED], HELDOUT[1], "Controls", None]
        order = [n for pair in zip(top, bottom) for n in pair if n]
        fig.legend([handles[labels.index(n)] for n in order], order, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=len(top), frameon=False, fontsize=5.6, labelcolor=INK_2, handlelength=1.6,
                   columnspacing=0.7, handletextpad=0.35, borderaxespad=0.15, labelspacing=0.25)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.255)
        os.makedirs(out, exist_ok=True)
        paths = [os.path.join(out, f"axbench_text_budget_curve.{ext}") for ext in ("pdf", "png")]
        for path in paths:
            fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
    return paths
