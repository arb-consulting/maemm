"""The paper's main steering table column and its AxBench budget figure, from a run's saved results.

The table has one column per benchmark and one row per reader in a fixed order (`ROWS`): four readers,
two references (the steered model at `evals.downstream.common.plain_steer.TABLE_STRENGTH`, held-out concept texts)
and two controls. Each package writes its own column with `write_column`: a cell is the identification
rate from 8 texts with its 95 % interval (the J-lens's at its one text), marked where MAEM differs from
that row at a Holm-corrected p < 0.05 (paired two-sided sign-flip permutation test over the column's
units, `evals.downstream.common.stats.sign_flip_tests`); references are shown and never tested. Files:
`tables/paper_main_column.{tex,csv,md}`, and for AxBench `figures/paper_axbench_budget_curve.{pdf,png}`,
the text concepts' rate against texts shown (1, 2, 4, 8). Reads only `tables/identification.csv`'s rows
and the per-case verdicts; nothing here asks a judge or a model.
"""
import csv
import math
import os

from evals.downstream.common.plain_steer import STEER_UNIT, TABLE_STRENGTH, arm_name
from evals.downstream.common.stats import FLIP_SEED, N_FLIPS, holm, sign_flip_tests

from .config import SINGLE_TEXT
from .evaluate import case_accuracy

#: Significance line of the table's marks.
ALPHA = 0.05
#: The column's texts: 8 shown together, the budget every cell but the J-lens's is read at.
TEXTS = 8
MARK = "$^\\ast$"
#: The steered-model row's arm in both columns, `plain_steered@1`.
STEERED = arm_name(TABLE_STRENGTH)
#: How the table names that strength: `s` times the typical layer-42 residual norm.
STRENGTH_NOTE = (f"Steered model at strength s={TABLE_STRENGTH:g}: s times the typical layer-42 residual norm "
                 f"({STEER_UNIT:.1f}) added at every position.")

# (section, LaTeX label, plain label, AxBench condition, BiPO arm) in the table's order. `section` is
# "readers", "references" or "controls"; a reference is shown and never tested against MAEM.
ROWS = (
    ("readers", "\\method{}", "MAEM", "maem", "maem"),
    ("readers", "NLA", "NLA", "nla_native", "nla_native"),
    ("readers", "J-lens$^\\dagger$", "J-lens", "jlens", "jlens"),
    ("readers", "Corpus search (10M tokens)", "Corpus search (10M tokens)", "retrieval", "retrieval"),
    ("references", f"Steered model ($s{{=}}{TABLE_STRENGTH:g}$)$^\\ddagger$",
     f"Steered model (s={TABLE_STRENGTH:g})", STEERED, STEERED),
    ("references", "Held-out concept texts", "Held-out concept texts", "heldout_positive", "heldout_matching"),
    ("controls", "Untrained base", "Untrained base", "base_l1", "base_l1"),
    ("controls", "Shuffled", "Shuffled", "shuffled", "shuffled"),
)
SECTION_TITLES = {"references": "References", "controls": "Controls"}
REFERENCE_SECTION = "references"
PAPER_TABLE = "tables/paper_main_column"
PAPER_FIGURE = "figures/paper_axbench_budget_curve"
CSV_COLUMNS = ("section", "label", "arm", "estimate", "ci_lower", "ci_upper", "n", "cell", "tested",
               "mean_diff", "n_pairs", "p", "p_holm", "significant")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def cell(estimate, lo, hi, significant=False):
    """`0.958 [0.934, 0.979]`, the interval left off where it is missing, `--` for no value."""
    estimate, lo, hi = _number(estimate), _number(lo), _number(hi)
    if estimate is None:
        return "--"
    text = f"{estimate:.3f}" if lo is None or hi is None else f"{estimate:.3f} [{lo:.3f}, {hi:.3f}]"
    return text + (MARK if significant else "")


def marks(tests):
    """`{arm: (mean_diff, n_pairs, p, p_holm, significant)}` over one column's comparisons, in row order.

    `tests` is `[(arm, paired differences MAEM minus arm)]`; every comparison is drawn from one generator
    in the order given, and Holm's correction is over all of them."""
    raw = sign_flip_tests([d for _arm, d in tests], N_FLIPS, FLIP_SEED)
    adjusted = holm(raw)
    return {arm: (sum(d) / len(d) if d else float("nan"), len(d), p, h, h < ALPHA)
            for (arm, d), p, h in zip(tests, raw, adjusted)}


def column_rows(key, rates, tests):
    """The column's rows in table order: `rates` is `{arm: (estimate, lo, hi, n)}` for the arms the run
    holds (keyed by this column's arm names, `key` 3 for AxBench and 4 for BiPO in `ROWS`), `tests` the
    paired differences as `marks` takes them."""
    stats = marks(tests)
    out = []
    for row in ROWS:
        section, arm = row[0], row[key]
        estimate, lo, hi, n = rates.get(arm, (None, None, None, None))
        tested = arm in stats
        diff, n_pairs, p, p_holm, significant = stats.get(arm, (None, None, None, None, False))
        out.append({"section": section, "label": row[2], "tex_label": row[1], "arm": arm,
                    "estimate": _number(estimate), "ci_lower": _number(lo), "ci_upper": _number(hi), "n": n,
                    "cell": cell(estimate, lo, hi, significant), "tested": tested, "mean_diff": diff,
                    "n_pairs": n_pairs, "p": p, "p_holm": p_holm, "significant": bool(significant)})
    return out


def tex(rows, header):
    """The column as LaTeX rows: `label & cell \\\\`, the two lower sections under a rule and a title."""
    lines = [f"% {header}",
             f"% Identification rate from {TEXTS} texts [95% CI] (J-lens: its one text). {MARK}: MAEM "
             f"differs at Holm-corrected p < {ALPHA:g} (paired sign-flip permutation test, {N_FLIPS:,} flips, "
             f"seed {FLIP_SEED}, within the column); references are not tested. {STRENGTH_NOTE}"]
    width = max(len(r["tex_label"]) for r in rows)
    section = "readers"
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            lines += ["\\midrule", f"\\multicolumn{{2}}{{l}}{{\\emph{{{SECTION_TITLES[section]}}}}} \\\\"]
        lines.append(f"{r['tex_label'].ljust(width)} & {r['cell']} \\\\")
    return "\n".join(lines) + "\n"


def markdown(rows, header):
    def fmt(value, digits=3):
        return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"
    lines = [f"**{header}**", "",
             f"Identification rate from {TEXTS} texts, 95 % interval; MAEM minus each tested row, paired "
             f"two-sided sign-flip permutation test ({N_FLIPS:,} flips, seed {FLIP_SEED}), Holm-corrected "
             f"within the column. References are not tested. {STRENGTH_NOTE}", "",
             "| Section | Row | Rate [95% CI] | n | MAEM minus row | pairs | p | Holm p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['section']} | {r['label']} | {r['cell'].replace(MARK, ' *')} | {r['n'] or ''} | "
                     f"{fmt(r['mean_diff'])} | {r['n_pairs'] or ''} | {fmt(r['p'], 4)} | {fmt(r['p_holm'], 4)} |")
    return "\n".join(lines) + "\n"


def write_column(root, rows, header):
    """`tables/paper_main_column.{tex,csv,md}` under the run directory `root`; the paths written."""
    base = os.path.join(str(root), PAPER_TABLE)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with open(base + ".tex", "w", encoding="utf-8") as handle:
        handle.write(tex(rows, header))
    with open(base + ".md", "w", encoding="utf-8") as handle:
        handle.write(markdown(rows, header))
    with open(base + ".csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in CSV_COLUMNS})
    return [base + ext for ext in (".tex", ".csv", ".md")]


# --------------------------------------------------------------------------------------------- AxBench
GROUP = "text"


def _summary(rows, judge, condition, budget_type, budget):
    for r in rows:
        if (str(r.get("metric")), str(r.get("condition")), str(r.get("judge")), str(r.get("group")),
                str(r.get("budget_type")), str(r.get("budget"))) == \
                ("identification_accuracy", condition, judge, GROUP, budget_type, str(budget)):
            return r
    return None


def _slot(condition, budget=TEXTS):
    """Where a condition's column value lives: the J-lens's one text, everyone else's `budget` texts."""
    return ("greedy", 1) if condition in SINGLE_TEXT else ("snippets", budget)


def axbench_column(root, summary, cases, concepts, judge):
    """The AxBench column over the text concepts, from `tables/identification.csv`'s rows (`summary`) and
    `judge`'s per-case verdicts (`cases`), paired over the concepts both arms have text for
(`evaluate.case_accuracy`). Writes the three files and returns the rows."""
    ids = sorted(c["concept_id"] for c in concepts if c["genre"] == GROUP)
    wanted = set(ids)
    correct = {}
    for r in cases:
        if r.get("judge") == judge and r["concept_id"] in wanted and case_accuracy(r) is not None:
            correct[(r["condition"], r["budget_type"], int(r["budget"]), r["concept_id"])] = case_accuracy(r)
    rates, tests = {}, []
    for section, _tex, _label, condition, _arm in ROWS:
        row = _summary(summary, judge, condition, *_slot(condition))
        if row is not None:
            rates[condition] = (row["estimate"], row["ci_lower"], row["ci_upper"], row.get("n_total"))
        if condition == "maem" or section == REFERENCE_SECTION:
            continue
        mine, other = ("maem", "snippets", TEXTS), (condition, *_slot(condition))
        diffs = [correct[(*mine, c)] - correct[(*other, c)] for c in ids
                 if (*mine, c) in correct and (*other, c) in correct]
        if diffs:
            tests.append((condition, diffs))
    rows = column_rows(3, rates, tests)
    write_column(root, rows, f"AxBench & text concepts ($n{{=}}{len(ids)}$)")
    return rows


# ---------------------------------------------------------------------------------- the budget figure
BUDGETS = (1, 2, 4, 8)
SURFACE, INK, INK_2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# Fixed categorical order; steered and NLA, the one close colour-vision-deficiency pair, differ by dashes.
SERIES = {
    "maem": ("MAEM", "#2F5C92", "o"),
    "nla_native": ("NLA", "#009988", "s"),
    "retrieval": ("Corpus (10M)", "#EE7733", "^"),
    "jlens": ("J-lens", "#33BBEE", "D"),
    STEERED: (f"Steered, s={TABLE_STRENGTH:g} (ref.)", "#EE3377", "v"),
}
REFERENCE = ("heldout_positive", "Held-out texts (ref.)", "#3d3c39")
CONTROLS = (("base_l1", "Untrained base"), ("shuffled", "Shuffled"))
DASHED = (0, (3.5, 1.5))
DASH_DOT = (0, (5, 1.5, 1.2, 1.5))


def _curve(summary, judge, condition):
    out = []
    for b in BUDGETS:
        r = _summary(summary, judge, condition, "snippets", b)
        if r is not None and _number(r["estimate"]) is not None:
            out.append((b, float(r["estimate"]), _number(r["ci_lower"]), _number(r["ci_upper"])))
    return out


def _band(ax, xs, pts, colour, alpha):
    if all(p[2] is not None and p[3] is not None for p in pts):
        ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=colour, alpha=alpha, lw=0,
                        zorder=1)


def budget_figure(root, summary, judge):
    """`figures/paper_axbench_budget_curve.{png,pdf}`: the text concepts' identification rate against the
    texts shown to the judge. Every point and band is a summary row; nothing is recomputed."""
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(str(root), "cache", "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Serif to sit with LaTeX body text, sized for one column.
    style = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
             "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6,
             "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(2.75, 2.25), dpi=300)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        for condition, _name in CONTROLS:
            pts = _curve(summary, judge, condition)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#b9b8b2", lw=0.9, marker="o", ms=2.5,
                        zorder=2, label="Controls" if condition == CONTROLS[0][0] else None)
        condition, name, colour = REFERENCE
        pts = _curve(summary, judge, condition)
        if pts:
            xs = [p[0] for p in pts]
            _band(ax, xs, pts, colour, 0.08)
            ax.plot(xs, [p[1] for p in pts], color=colour, lw=1.1, ls=DASHED, marker="o", ms=3,
                    markerfacecolor=SURFACE, zorder=3, label=name)
        for condition, (name, colour, marker) in SERIES.items():
            if condition == "jlens":
                r = _summary(summary, judge, condition, "greedy", 1)
                value = None if r is None else _number(r["estimate"])
                if value is None:
                    continue
                # one text per direction: flat across the budgets, marked where it was measured
                lo, hi = _number(r["ci_lower"]), _number(r["ci_upper"])
                if lo is not None and hi is not None:
                    ax.fill_between([BUDGETS[0], BUDGETS[-1]], [lo] * 2, [hi] * 2, color=colour, alpha=0.15,
                                    lw=0, zorder=1)
                ax.plot([BUDGETS[0], BUDGETS[-1]], [value] * 2, color=colour, lw=1.4, ls=DASH_DOT, zorder=4)
                ax.plot([BUDGETS[0]], [value], color=colour, marker=marker, ms=4, markeredgecolor=SURFACE,
                        markeredgewidth=0.7, lw=0, zorder=5)
                ax.add_artist(plt.Line2D([], [], color=colour, lw=1.4, ls=DASH_DOT, marker=marker, ms=4,
                                         markeredgecolor=SURFACE, label=name))
                continue
            pts = _curve(summary, judge, condition)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            _band(ax, xs, pts, colour, 0.15)
            ax.plot(xs, [p[1] for p in pts], color=colour, lw=1.4, marker=marker, ms=4, markeredgecolor=SURFACE,
                    markeredgewidth=0.7, zorder=4, label=name,
                    ls=DASHED if condition == STEERED else "-")
        ax.axhline(0.1, color=MUTED, lw=0.7, ls=":", zorder=0)
        ax.annotate("chance", (BUDGETS[-1], 0.1), xytext=(0, 1.5), textcoords="offset points", fontsize=6,
                    color=MUTED, va="bottom", ha="right")
        # The two controls share one legend entry, so each is named at the right end of its own line.
        for condition, name in CONTROLS:
            pts = _curve(summary, judge, condition)
            if pts:
                ax.annotate(name, (pts[-1][0], pts[-1][1]), xytext=(0, 3), textcoords="offset points",
                            ha="right", va="bottom", fontsize=6, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlim(0.88, 9.0)
        handles, names = ax.get_legend_handles_labels()
        # readers on the first legend row, references and controls on the second (filled by column)
        top = [n for n in (SERIES[k][0] for k in ("maem", "nla_native", "jlens", "retrieval")) if n in names]
        bottom = [n for n in (SERIES[STEERED][0], REFERENCE[1], "Controls") if n in names]
        order = [n for pair in zip(top, bottom + [None] * (len(top) - len(bottom))) for n in pair if n]
        if order:
            fig.legend([handles[names.index(n)] for n in order], order, loc="lower center",
                       bbox_to_anchor=(0.5, 0.0), ncol=max(1, len(top)), frameon=False, fontsize=5.6,
                       labelcolor=INK_2, handlelength=1.6, columnspacing=0.7, handletextpad=0.35,
                       borderaxespad=0.15, labelspacing=0.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Texts shown to the judge", labelpad=2)
        ax.set_ylabel("Identification accuracy", labelpad=2)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.255)
        base = os.path.join(str(root), PAPER_FIGURE)
        os.makedirs(os.path.dirname(base), exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{base}.{ext}", facecolor=SURFACE)
        plt.close(fig)
    return [f"{base}.png", f"{base}.pdf"]
