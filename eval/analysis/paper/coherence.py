"""The coherence section: Figure `fig:coherence` and its in-text numbers, from a rollout_coherence run.

Every number is the matched-target reading: each target re-captured from the 64-token source tail, read
standalone. The points (judged net against the source, log-likelihood gap, raw and centred cosine) come
from `tables/frontier_matched.csv`, the at-least-as-coherent share from `frontier_outcomes.csv`; the
retrieval curve's corpus sizes are the table's own (1M-10M tokens on a full run). A run whose
table leaves MAEMM's centred cosine blank past k=8 (the run of record) gets those cells from the per-sample
scores `frontier/context/scores/maemm.jsonl`, through the package's own best-of-k functions; the same
fallback at k=1..8 is checked against the table (`check.*` rows). Writes `coherence_two_plots.{pdf,png}`
and `numbers.{csv,md}`.
"""
import json
import os

from eval.common.stats import boot_indices, boot_mean
from eval.rollout_coherence import config as RC
from eval.rollout_coherence.frontier_analysis import per_activation_values

from .common import Numbers, num, pick, table

JUDGE = "sonnet"
K_ALL = RC.BEST_OF_K
# (group, label, colour, marker, points); `curves` adds the retrieval curve at the run's corpus sizes
CURVES = (("maemm", "MAEM", "#2F5C92", "o", [f"k={k}" for k in K_ALL]),
          ("nla_native", "NLA (whole explanation)", "#CC6677", "s", [f"k={k}" for k in (1, 2, 4, 8)]),
          ("nla", "NLA (first 64 tokens)", "#009988", "D", [f"k={k}" for k in (1, 2, 4, 8)]))
REFERENCE = ("continuation", "k=1", "Base model's continuation (ref.)", "#33BBEE")


def corpus_sizes(get):
    """The package's corpus-size labels the table's retrieval rows carry: 1M ... 10M on a full run, the
    smoke sizes on a smoke run."""
    labels = [label for label, _tokens in RC.SIZES + RC.SMOKE_CORPUS_SIZES]
    return [label for label in labels if ("net", "retrieval", label) in get]


def curves(get):
    """`CURVES` plus the retrieval curve over the run's corpus sizes."""
    sizes = corpus_sizes(get)
    label = f"Corpus retrieval ({sizes[0]}–{sizes[-1]} tokens)"
    return CURVES + (("retrieval", label, "#EE7733", "^", sizes),)


def centred_best_of_k(run, k_values):
    """`{k: (estimate, lo, hi)}` of MAEMM's centred cosine at best-of-k, ranked on the raw cosine: k up to
    `RC.MAIN_K` over draws 0-7, larger k over every draw, as the package's frontier reads them."""
    samples = {}
    with open(os.path.join(str(run), "frontier", "context", "scores", "maemm.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if isinstance(r["sample"], int) and num(r["cos_raw"]) is not None:
                samples.setdefault(r["i"], []).append(
                    {"order_key": r["sample"], "cos": r["cos_raw"], "cos_centred": r["cos_centred"]})
    by_pos = [samples[i] for i in sorted(samples)]
    idx = boot_indices(len(by_pos), RC.N_BOOT, RC.BOOT_SEED)
    out = {}
    for k in k_values:
        pools = {p: [s for s in pool if k > RC.MAIN_K or s["order_key"] < RC.MAIN_K]
                 for p, pool in enumerate(by_pos)}
        vals, _n = per_activation_values(pools, len(by_pos), k, {"x": lambda s: s["cos_centred"]})
        out[k] = boot_mean(vals["x"], idx)
    return out


def points(run):
    """`({(group, point): {"x": (est, lo, hi), "net": ..., "ll": ...}}, rows by (metric, group, point),
    the k values whose centred cosine the fallback filled, the curves)`."""
    get = {(r["metric"], r["group"], r["condition"]): r for r in table(run, "frontier_matched")
           if r["judge"] in ("none", JUDGE)}
    triple = lambda r: tuple(num(r[c]) for c in ("estimate", "ci_lower", "ci_upper"))
    blank = [k for k in K_ALL if num(get[("reread_cos_centred", "maemm", f"k={k}")]["estimate"]) is None]
    filled = centred_best_of_k(run, K_ALL) if blank else {}
    out = {}
    lines = curves(get)
    for group, *_rest, pts in lines + ((REFERENCE[0], None, None, None, [REFERENCE[1]]),):
        for p in pts:
            x = get.get(("reread_cos_centred", group, p))
            x = triple(x) if x else (None,) * 3
            if group == "maemm" and int(p[2:]) in blank:
                x = filled[int(p[2:])]
            out[(group, p)] = {"x": x, "net": triple(get[("net", group, p)]),
                               "ll": triple(get[("ll_gap", group, p)])}
    return out, get, filled, lines


def figure(pts, lines, out):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out, ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), dpi=300, sharex=True)
        for ax, key, ylabel, tag in ((axes[0], "net", "Net judge preference vs source", "(a)"),
                                     (axes[1], "ll", "Log-likelihood minus source (nats/token)", "(b)")):
            group, point, label, colour = REFERENCE
            ax.axhline(pts[(group, point)][key][0], color=colour, lw=0.9, ls=(0, (3.5, 1.5)), zorder=1,
                       label=label)
            ax.axhline(0, color="#8a8984", lw=0.6, zorder=0)
            for group, label, colour, marker, names in lines:
                cur = [pts[(group, n)] for n in names]
                xs, ys = [c["x"][0] for c in cur], [c[key][0] for c in cur]
                ax.errorbar(xs, ys, xerr=[[c["x"][0] - c["x"][1] for c in cur], [c["x"][2] - c["x"][0] for c in cur]],
                            yerr=[[c[key][0] - c[key][1] for c in cur], [c[key][2] - c[key][0] for c in cur]],
                            color=colour, marker=marker, ms=3, lw=1.1, elinewidth=0.6, capsize=0, label=label,
                            zorder=3)
                for n, x, y, dy in ((names[0], xs[0], ys[0], -8), (names[-1], xs[-1], ys[-1], 4)):
                    ax.annotate(n, (x, y), xytext=(0, dy), textcoords="offset points", fontsize=5.5,
                                ha="center", color="#52514e")
            ax.plot([1.0], [0.0], marker="*", ms=8, color="#0b0b0b", lw=0, zorder=4, label="Source passage")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Centred cosine with the target")
            ax.text(0.01, 0.98, tag, transform=ax.transAxes, va="top", fontsize=8, weight="bold")
            ax.grid(axis="y", color="#e6e5e0", lw=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6, handlelength=2.5)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.33, wspace=0.28)
        os.makedirs(out, exist_ok=True)
        paths = [os.path.join(out, f"coherence_two_plots.{ext}") for ext in ("pdf", "png")]
        for path in paths:
            fig.savefig(path, facecolor="#ffffff")
        plt.close(fig)
    return paths


def build(run, out):
    pts, get, filled, lines = points(run)
    N = Numbers("coherence")
    src = "frontier_matched.csv"
    N.add("n_activations", get[("net", "maemm", "k=1")]["n_activations"], source=src)
    N.add_row("maemm.at_least_as.per_sample",
              pick(table(run, "frontier_outcomes"), metric="at_least_as", condition="per_sample", group="maemm",
                   judge=JUDGE, population="matched"), "frontier_outcomes.csv")
    for group, point in (("maemm", "k=1"), ("maemm", "k=8"), ("maemm", "k=64"), ("continuation", "k=1"),
                         ("retrieval", lines[-1][4][-1]), ("nla", "k=1"), ("nla_native", "k=1")):
        for name, metric in (("net", "net"), ("ll_gap", "ll_gap"), ("cos_raw", "reread_cos")):
            N.add_row(f"{group}.{point}.{name}", get[(metric, group, point)], src)
        fallback = group == "maemm" and int(point[2:]) in filled
        N.add(f"{group}.{point}.cos_centred", *pts[(group, point)]["x"],
              source="frontier/context/scores/maemm.jsonl" if fallback else src)
    for k, (est, _lo, _hi) in filled.items():
        table_x = num(get[("reread_cos_centred", "maemm", f"k={k}")]["estimate"])
        if table_x is not None:
            N.add(f"check.maemm.k={k}.cos_centred_table_minus_recomputed", table_x - est, source=src)
    figure(pts, lines, out)
    N.write(out)
    return N
