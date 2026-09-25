"""The figures, the rule-picked examples and report.md, rendered from the saved artifacts alone (no model,
no judge call), so `report` runs on CPU and re-renders identically.

Every number printed comes from a row of `analysis.tables()` or `analysis.coverage_and_costs()`; a missing
quantity prints "unavailable". Every selector pins `band`, and judged selectors pin `judge`. The two families
are never pooled in a table."""

import json, math, time

import numpy as np

from evals.downstream.common.figures import SMOKE_BANNER, err as _err, is_unavailable as _is_unavailable
from evals.downstream.common.figures import mpl as _mpl, one as _one, save as _save, sel as _sel, suptitle as _suptitle
from evals.downstream.common.judge_client import served_line
from evals.downstream.common.judges import REFERENCE_JUDGE
from evals.downstream.common.nla.nla_reader import PINS as NLA
from evals.downstream.common.runs import json_safe, launch_gpu_seconds, launch_records, mark_stage
from evals.downstream.common.stats import write_tables
from evals.downstream.workspace_modulation import analysis as A
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import paper_tables as PT
from evals.downstream.workspace_modulation import retrieval as RET
from evals.downstream.workspace_modulation import summarise as SUM
from evals.downstream.workspace_modulation.judge import verdict_of
from evals.downstream.workspace_modulation.rollouts import BASIS_CONTROL, merged_rel
from evals.downstream.workspace_modulation.runs import stage_key

# Every artifact a section reads; a missing one raises rather than rendering zeroes.
REQUIRED_ARTIFACTS = (
    "data/items.json",
    "data/compliance.json",
    "activations/cell_table.json",
    "activations/mean_table.json",
    "lens/lens.json",
    "judges/summaries.json",
    "rollouts/nla.json",
    RET.MERGED_REL,
    SUM.BAND_REL,
    C.PATCH_REL,
) + tuple(A.judge_log_rel(name, "naming") for name in C.JUDGES)

CONDITION_LABEL = {
    "baseline": "baseline",
    "mention": "mention",
    "ignore": "ignore",
    "dont_think": "don't-think",
    "focus": "focus",
}
HEADS = C.HEADLINE_CONDITIONS + (C.FLOOR_CONDITION,)
# The appendix control's columns: `mention` beside the two dismissals, then the floor.
CONTROL_HEADS = ("mention",) + C.CONTROL_CONDITIONS + (C.FLOOR_CONDITION,)
HEADLINE_BUDGET = max(C.PASS_AT[C.HEADLINE_ARM])
NULL_BUDGET = max(C.PASS_AT[C.NULL_ARM])
BASE_BUDGET = max(C.PASS_AT[C.BASE_ARM])
NLA_BUDGET = max(A.NLA_BUDGETS)
RETRIEVAL_BUDGET = max(A.RETRIEVAL_BUDGETS)
RETRIEVAL_LABEL = f"corpus search, top {C.TOP_WINDOWS}"
PATCH_BUDGET = max(A.PATCH_BUDGETS)
PATCH_LABEL = f"Patchscopes (L{C.PATCH_LAYER})"
PATCH_FLOOR_LABEL = "Patchscopes floor (no patch)"
CORPUS_TOKENS_M = round(C.CORPUS["corpus_tokens"] / 1_000_000)
# (condition, word-rule metric, budget or cutoff, judged condition, palette key, marker, filled, label)
# of every reader the tables and figures read; the untrained-base ablation is rendered in Appendix D.
READERS = (
    ("maem_reg", "hit_any", HEADLINE_BUDGET, "maem_reg8", "maem", "o", True, "MAEM"),
    ("nla", "hit_any", NLA_BUDGET, "nla_n8", "nla", "D", True, "NLA verbalizer"),
    ("nla64", "hit_any", NLA_BUDGET, None, "nla", "D", False, f"NLA verbalizer, first {NLA.trunc} tokens"),
    (
        C.RETRIEVAL_READER,
        "hit_any",
        RETRIEVAL_BUDGET,
        C.RETRIEVAL_JUDGED,
        "retrieval",
        "^",
        True,
        RETRIEVAL_LABEL,
    ),
    (
        "jlens_L42",
        "rank10_L42_any",
        C.TOP_WORD,
        "jlens_L42_summary",
        "jlens",
        "s",
        True,
        f"J-lens L{C.READ_LAYER} top-{C.TOP_WORD}",
    ),
    ("maem_null", "hit_any", NULL_BUDGET, "maem_null8", "control", "x", False, "null control"),
    (C.PATCH_ARM, "hit_any", PATCH_BUDGET, C.PATCH_JUDGED[C.PATCH_ARM], "patchscope", "v", True, PATCH_LABEL),
    (C.PATCH_FLOOR, "hit_any", PATCH_BUDGET, None, "patchscope", "v", False, PATCH_FLOOR_LABEL),
)
READER_LABEL = {r[0]: r[7] for r in READERS}
READER_LABEL.update(
    {
        "maem_reg8": f"MAEM {HEADLINE_BUDGET}-sample readout",
        "maem_null8": f"null control, {NULL_BUDGET}-sample readout",
        "nla_n8": f"NLA verbalizer, {NLA_BUDGET} samples joined",
        C.RETRIEVAL_JUDGED: f"corpus search, top {C.TOP_WINDOWS} windows joined",
        C.BASE_ARM: "untrained base",
        "jlens_L42_summary": f"J-lens top-{C.TOP_WORD} at layer {C.READ_LAYER}, summarised",
        C.LENS_BAND_READER: f"J-lens top-{C.TOP_WORD} of layers 36-50 (step 2) pooled, summarised",
        A.LENS_BAND_COND: f"J-lens top-{C.TOP_WORD} of layers 36-50 (step 2) pooled",
        "jlens_best": "J-lens, rank 1 over all fitted layers (target-informed)",
        C.PATCH_JUDGED[C.PATCH_ARM]: f"{PATCH_LABEL}, {PATCH_BUDGET}-sample readout",
    }
)
BAR_READERS = ("maem_reg", "nla", C.RETRIEVAL_READER, "jlens_L42", "maem_null", C.PATCH_ARM, C.PATCH_FLOOR)
# The arithmetic word rule is not identification evidence, so arithmetic is drawn on the judge alone.
BAR_PANELS = (
    ("topics · word rule", "topics", "rule"),
    ("topics · judged", "topics", "judged"),
    ("arithmetic · judged", "arithmetic", "judged"),
)
HEADLINE_CONTRASTS = (
    ("maem_reg", "nla", "MAEM − NLA verbalizer"),
    ("maem_reg", C.RETRIEVAL_READER, "MAEM − corpus search"),
    ("maem_reg", "jlens_L42", f"MAEM − J-lens L{C.READ_LAYER}"),
    ("maem_reg", "maem_null", "MAEM − null control"),
)
PATCH_CONTRAST_ROWS = ((C.PATCH_ARM, C.PATCH_FLOOR, f"{PATCH_LABEL} − {PATCH_FLOOR_LABEL}"),)
assert tuple((a, b) for a, b, _l in PATCH_CONTRAST_ROWS) == A.PATCH_CONTRASTS
FIGURES = {
    "01": "01_final_period",
    "02": "02_modulation",
    "03": "03_carrier_mean",
    "04": "04_lens_protocol",
    "05": "05_control_conditions",
}


def _label(cond):
    return READER_LABEL.get(cond, cond)


def _judge_label(name):
    return C.JUDGES[name].label


JUDGE_NAMES = tuple(C.JUDGES)


def _reader(cond):
    return next(r for r in READERS if r[0] == cond)


# --- row selection ---------------------------------------------------------------------------------------


def _rate(R, cond, metric, budget, g, instr, band):
    """One reader's word-rule (or lens-rank) row at one band. Every key is pinned, `band` included: the same
    reader, metric, group, instruction and budget name one row per cell set."""
    return _one(R, metric=metric, condition=cond, group=g, instruction=instr, band=band, budget=budget)


def _chance(R, cond, budget, g, instr, band):
    """The reader's OWN 20-donor chance line at its own budget or cutoff: the NLA's donors are its own
    readouts, so one chance number carried across the readers would be the wrong null for most of them."""
    return _one(R, metric="chance", condition=cond, group=g, instruction=instr, band=band, budget=budget)


def _judged(J, g, cond, name, band, metric="named", instr=""):
    if cond is None:
        return None
    return _one(J, metric=metric, condition=cond, group=g, band=band, judge=C.JUDGES[name].model, instruction=instr)


def _mod(M_rows, cond, metric, band, g, instr_a, instr_b=C.FLOOR_CONDITION, name=None):
    """One modulation.csv row, pinned to its source condition (and judge, for a judged metric)."""
    source, _f = A.metric_accessor(cond, metric, name)
    extra = {"source_condition": source} if source else {}
    if A.is_judged(cond, metric):
        extra["judge"] = C.JUDGES[name or REFERENCE_JUDGE].model
    return _one(
        M_rows,
        metric=metric,
        condition=cond,
        group=g,
        band=band,
        instruction_a=instr_a,
        instruction_b=instr_b,
        **extra,
    )


def _contrast(CON, a, b, metric, g, instr, band, name=None):
    extra = {"judge": C.JUDGES[name].model} if name else {}
    return _one(
        CON,
        metric=metric,
        condition=f"{a}-{b}",
        group=g,
        band=band,
        instruction_a=instr,
        instruction_b=instr,
        **extra,
    )


def _n_of(r):
    return r.get("n_total", 0) if r else 0


def _panel_n(r):
    """A panel title's n fragment; "n unmeasured" when the column has no row."""
    return f"n = {_n_of(r)}" if r is not None else "n unmeasured"


# --- formatting ------------------------------------------------------------------------------------------


def _fmt_rate(r, digits=1):
    """A rate as a percentage with its interval and n; "unavailable" for a NaN estimate."""
    if r is None:
        return "unavailable"
    if _is_unavailable(r.get("estimate")):
        return f"unavailable (n = {r.get('n_valid', 0)}/{r.get('n_total', 0)})"
    est = 100.0 * float(r["estimate"])
    n = f"(n = {r.get('n_valid', '')}/{r.get('n_total', '')})"
    if _is_unavailable(r.get("ci_lower")) or _is_unavailable(r.get("ci_upper")):
        return f"{est:.{digits}f} % (no interval) {n}"
    return (
        f"{est:.{digits}f} % [{100.0 * float(r['ci_lower']):.{digits}f}, {100.0 * float(r['ci_upper']):.{digits}f}] {n}"
    )


def _pct(r):
    """A rate as a bare percentage, for the dense tables; "—" when the reader was not read here."""
    if r is None:
        return "—"
    if _is_unavailable(r.get("estimate")):
        return "unavailable"
    return f"{100.0 * float(r['estimate']):.1f}"


def _fmt_chance(r):
    """The chance cell: "no donor hit" at exactly zero, "unavailable" when unmeasured."""
    if r is None or _is_unavailable(r.get("estimate")):
        return "unavailable"
    if float(r["estimate"]) == 0.0:
        return "no donor hit"
    return f"{100.0 * float(r['estimate']):.1f}"


def _fmt_ratio(r):
    """The rate over its own chance line, flagged below CHANCE_FLAG_RATIO × chance."""
    if r is None:
        return "—"
    v = r.get("ratio_vs_chance")
    if v in ("", None) or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    flag = str(r.get("flag_below_3x")).lower() in ("true", "1")
    return f"{float(v):.1f}x" + (f" (below {C.CHANCE_FLAG_RATIO:.0f}x)" if flag else "")


def _fmt3(v):
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    return f"{float(v):.3f}"


def _fmt_usd(v):
    """A dollar amount to four decimals, or "unavailable"."""
    if v is None or v == "unavailable" or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    return f"${float(v):.4f}"


def _excludes_zero(r):
    """(above, below): whether the interval excludes zero on each side."""
    if r is None or _is_unavailable(r.get("estimate")):
        return False, False
    lo, hi = r.get("ci_lower"), r.get("ci_upper")
    above = not _is_unavailable(lo) and float(lo) > 0
    below = not _is_unavailable(hi) and float(hi) < 0
    return above, below


def _pp(r):
    """A paired difference in percentage points with its interval, * when the interval excludes zero."""
    if r is None:
        return "—"
    if _is_unavailable(r.get("estimate")):
        return "unavailable"
    est = 100.0 * float(r["estimate"])
    if _is_unavailable(r.get("ci_lower")) or _is_unavailable(r.get("ci_upper")):
        return f"{est:+.1f}"
    above, below = _excludes_zero(r)
    star = "*" if (above or below) else ""
    return f"{est:+.1f}{star} [{100.0 * float(r['ci_lower']):+.1f}, {100.0 * float(r['ci_upper']):+.1f}]"


def _md_table(rows, cols):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


# --- the tables a band section is made of -----------------------------------------------------------------


def word_table(R, g, band, heads=HEADS):
    """Rows (reader, instruction) of the word rule: the rate, the reader's own 20-donor target-shuffle chance
    line, their ratio, and the null control's rate at the same cell."""
    cols = ["reader", "instruction", "rate %", "donor chance %", "rate / chance", "null control %"]
    rows = []
    for cond, metric, budget, *_rest, rlabel in READERS:
        for instr in heads:
            r = _rate(R, cond, metric, budget, g, instr, band)
            if r is None:
                continue
            null = _rate(R, "maem_null", "hit_any", NULL_BUDGET, g, instr, band)
            rows.append(
                {
                    "reader": rlabel,
                    "instruction": CONDITION_LABEL[instr],
                    "rate %": _fmt_rate(r),
                    "donor chance %": _fmt_chance(_chance(R, cond, budget, g, instr, band)),
                    "rate / chance": _fmt_ratio(r),
                    "null control %": "—" if cond == "maem_null" else _pct(null),
                }
            )
    return _md_table(rows, cols) if rows else ""


def ablation_table(R, band=C.FINAL_BAND, heads=HEADS):
    """Appendix D: the untrained-base row beside MAEM and the null control, per family and instruction."""
    cols = ["group", "instruction", "untrained base %", "donor chance %", "MAEM %", "null control %"]
    rows = []
    for fam in C.FAMILIES:
        for instr in heads:
            r = _rate(R, C.BASE_ARM, "hit_any", BASE_BUDGET, fam, instr, band)
            if r is None:
                continue
            rows.append(
                {
                    "group": fam,
                    "instruction": CONDITION_LABEL[instr],
                    "untrained base %": _fmt_rate(r),
                    "donor chance %": _fmt_chance(_chance(R, C.BASE_ARM, BASE_BUDGET, fam, instr, band)),
                    "MAEM %": _pct(_rate(R, "maem_reg", "hit_any", HEADLINE_BUDGET, fam, instr, band)),
                    "null control %": _pct(_rate(R, "maem_null", "hit_any", NULL_BUDGET, fam, instr, band)),
                }
            )
    return _md_table(rows, cols) if rows else ""


def judged_table(J, g, band, heads=HEADS):
    """Rows (reader, instruction) of the judged rate against the item's own targets and against the foil's."""
    cols = ["reader", "instruction"] + [f"{_judge_label(n)} {vs} %" for n in JUDGE_NAMES for vs in C.VS]
    rows = []
    for _c, _m, _b, jcond, *_rest, rlabel in READERS:
        if jcond is None:
            continue
        for instr in heads:
            cells = {
                (n, vs): _judged(J, g, jcond, n, band, metric=("named" if vs == "own" else "foil"), instr=instr)
                for n in JUDGE_NAMES
                for vs in C.VS
            }
            if all(v is None for v in cells.values()):
                continue
            rows.append(
                {
                    "reader": rlabel,
                    "instruction": CONDITION_LABEL[instr],
                    **{f"{_judge_label(n)} {vs} %": _fmt_rate(cells[(n, vs)]) for n in JUDGE_NAMES for vs in C.VS},
                }
            )
    return _md_table(rows, cols) if rows else ""


def modulation_table(M_rows, g, band, heads=C.HEADLINE_CONDITIONS, floor=C.FLOOR_CONDITION):
    """Every reader's rate under each of `heads` minus its rate under `floor`, paired by concept."""
    cols = ["reader", "metric", "judge"] + [f"{CONDITION_LABEL[i]} − {CONDITION_LABEL[floor]}" for i in heads]
    rows = []
    for cond in A.READERS:
        for metric in A.READER_METRICS[cond]:
            for name in (list(C.JUDGES) if A.is_judged(cond, metric) else [None]):
                rs = [_mod(M_rows, cond, metric, band, g, i, floor, name) for i in heads]
                if all(r is None for r in rs):
                    continue
                r0 = {"reader": _label(cond), "metric": metric, "judge": _judge_label(name) if name else "—"}
                for i, r in zip(heads, rs):
                    r0[f"{CONDITION_LABEL[i]} − {CONDITION_LABEL[floor]}"] = (
                        _pp(r)
                    )
                rows.append(r0)
    return _md_table(rows, cols) if rows else ""


def contrast_table(CON, g, band, heads=C.HEADLINE_CONDITIONS):
    """The reader contrasts, paired by item inside each instruction: the word rule (each reader on its own
    headline criterion) and, per judge, the judged own-target rate and the net of own over foil."""
    cols = ["contrast", "metric", "judge"] + [CONDITION_LABEL[i] for i in heads]
    rows = []
    for a, b, clabel in HEADLINE_CONTRASTS:
        specs = [(A.HEADLINE_METRIC[a], None)]
        specs += [(m, n) for m in C.JUDGED_CONTRAST_METRICS for n in C.JUDGES if A.is_judged(a, m) and A.is_judged(b, m)]
        for metric, name in specs:
            vals = [_contrast(CON, a, b, metric, g, i, band, name) for i in heads]
            if all(v is None for v in vals):
                continue
            r0 = {"contrast": clabel, "metric": metric, "judge": _judge_label(name) if name else "—"}
            for i, v in zip(heads, vals):
                r0[CONDITION_LABEL[i]] = _pp(v)
            rows.append(r0)
    return _md_table(rows, cols) if rows else ""


def patch_contrast_table(CON, g, band, heads=C.HEADLINE_CONDITIONS):
    """The Patchscopes reader minus its no-patch floor, paired by item inside each instruction, on the word
    rule at pass@8 (analysis.PATCH_CONTRASTS)."""
    cols = ["contrast", "metric"] + [CONDITION_LABEL[i] for i in heads]
    rows = []
    for a, b, clabel in PATCH_CONTRAST_ROWS:
        metric = A.HEADLINE_METRIC[a]
        vals = [_contrast(CON, a, b, metric, g, i, band, None) for i in heads]
        if all(v is None for v in vals):
            continue
        rows.append({"contrast": clabel, "metric": metric, **{CONDITION_LABEL[i]: _pp(v) for i, v in zip(heads, vals)}})
    return _md_table(rows, cols) if rows else ""


def _family_blocks(T, band, heads=HEADS, contrasts=True):
    """The three tables of one band, per family, never pooled."""
    R, J, MOD, CON = T["rates"], T.get("judged") or [], T.get("modulation") or [], T["contrasts"]
    out = []
    for fam in C.FAMILIES:
        n = _rate(R, "maem_reg", "hit_any", HEADLINE_BUDGET, fam, "focus", band)
        out.append(f"### {fam} ({_n_of(n)} concepts per instruction)\n")
        for label, table in (
            ("The word rule (% of concepts named):", word_table(R, fam, band, heads)),
            (
                "The judged naming rate (% of concepts named, against the item's own targets and, beside "
                "it, against the foil concept's — the false-positive line):",
                judged_table(J, fam, band, heads),
            ),
            (
                "Modulation, paired by concept (pp, 95 % bootstrap interval, * where it excludes zero):",
                modulation_table(MOD, fam, band, heads=[h for h in heads if h != C.FLOOR_CONDITION]),
            ),
            # focus − mention separates reading a salient concept from reading what the model was told to hold
            (
                "Focus − mention, paired by concept (the same units; the row that separates reading the "
                "workspace from reading the prompt):",
                modulation_table(MOD, fam, band, heads=("focus",), floor="mention")
                if set(C.HEADLINE_CONDITIONS) <= set(heads)
                else "",
            ),
            (
                "Reader contrasts, paired by item (pp, 95 % bootstrap interval, * where it excludes zero):",
                contrast_table(CON, fam, band, heads=[h for h in heads if h != C.FLOOR_CONDITION])
                if contrasts
                else "",
            ),
            (
                "Patchscopes against its own floor, paired by item (pp, 95 % bootstrap interval, * where it "
                "excludes zero; the rate the patch reads off the activation above what the prompt alone "
                "makes the model say):",
                patch_contrast_table(CON, fam, band, heads=[h for h in heads if h != C.FLOOR_CONDITION])
                if contrasts
                else "",
            ),
        ):
            if table:
                out += [label + "\n", table, ""]
    return out


READING_NOTE = (
    "Each rate is the share of concepts on which the reader names the concept: for the word rule, a member "
    f"word in any of its {HEADLINE_BUDGET} samples (the lens: a member in its top {C.TOP_WORD} at layer "
    f"{C.READ_LAYER}; the corpus search: in any of the {C.TOP_WINDOWS} best-scoring windows); for the judge, the "
    "naming judge is shown the concept's target forms and the readout, and nothing else, and says whether "
    "any target is named, with a verbatim quote that is checked against the readout. Every judged readout is "
    "asked twice, against its own targets (`own`) and against a same-family foil concept's (`foil`): the "
    "foil column is the reader's false-positive line and the judged column's chance, and MAEM's net of own "
    "over foil is in the contrasts. The **null control** is MAEM injected with a zero "
    "direction, so it reads no activation: its rate is what the pipeline names with no information at all. "
    "The **corpus search** is the comparator a reader has to beat to be worth generating: the same activation "
    f"is used as a query into the held-out corpus's {CORPUS_TOKENS_M}M-token search prefix, and its "
    f"{C.TOP_WINDOWS} best-matching windows are read as its readout. "
    f"**{PATCH_LABEL}** is the Patchscopes reader: the clean base continues an entity-description "
    f"prompt whose placeholder's layer-{C.PATCH_LAYER} residual is replaced, during prefill only, by "
    f"{C.PATCH_ALPHA:g} x its own norm x the same centred direction MAEM is injected with, and its "
    f"{PATCH_BUDGET} samples are read as MAEM's are. **{PATCH_FLOOR_LABEL}** is the same prompt with no "
    "patch, one sample set carried by every item: what that prompt makes the model say about nothing, and "
    "the line the reader's word rule is read against. "
    f"**rate / chance** is the rate over that reader's own chance line; the reading "
    f"rules ask for at least {C.CHANCE_FLAG_RATIO:.0f}x chance, and a rate below it is flagged there. "
    "On arithmetic the word rule is not identification evidence: an answer is a small "
    "number, and a readout doing arithmetic states many numbers, so it names other problems' answers about as "
    "often as its own — the word rule sits at its chance line there, and the judged table is that family's "
    "evidence."
)


# --- figures ----------------------------------------------------------------------------------------------


def _legend_handles(plt, series):
    """Proxy marker handles for a legend."""
    return [
        plt.Line2D(
            [], [], color=C.COLOURS[colour], marker=mk,
            mfc=C.COLOURS[colour] if filled else "white", ls="none", ms=5, label=label,
        )
        for label, colour, mk, filled in series
    ]


def _bar_figure(T, out_dir, smoke_n, band, band_label, fig_name, number, heads=HEADS):
    """Grouped bars per panel (family × instrument): one bar per reader and instruction with its Wilson
    interval, and its own chance line as a dotted tick (the donor line on the word rule, the foil rate on the
    judge)."""
    plt = _mpl()
    R, J = T["rates"], T.get("judged") or []
    fig, axes = plt.subplots(1, len(BAR_PANELS), figsize=(12.0, 4.2), sharey=True)
    width = 0.8 / len(BAR_READERS)
    for ax, (title, fam, inst) in zip(axes, BAR_PANELS):
        for k, cond in enumerate(BAR_READERS):
            _c, metric, budget, jcond, colour, _mk, filled, _label_ = _reader(cond)
            for j, instr in enumerate(heads):
                if inst != "rule" and jcond is None:
                    continue  # a reader the judge does not read has no judged bar
                if inst == "rule":
                    rr = _rate(R, cond, metric, budget, fam, instr, band)
                else:
                    rr = _judged(J, fam, jcond, REFERENCE_JUDGE, band, instr=instr)
                x = j + (k - (len(BAR_READERS) - 1) / 2.0) * width
                col = C.COLOURS[colour]
                if rr is None or _is_unavailable(rr["estimate"]):
                    ax.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=6, color=col, rotation=90)
                    continue
                y = float(rr["estimate"])
                ax.bar(
                    x, y, width * 0.92, color=col if filled else "white", edgecolor=col,
                    hatch=None if filled else "////",
                )
                ax.errorbar(
                    [x], [y], yerr=np.array(_err(rr), dtype=float).reshape(2, 1),
                    fmt="none", ecolor="#333333", capsize=2, lw=0.8,
                )
                ch = (
                    _chance(R, cond, budget, fam, instr, band)
                    if inst == "rule"
                    else _judged(J, fam, jcond, REFERENCE_JUDGE, band, metric="foil", instr=instr)
                )
                if ch is not None and not _is_unavailable(ch["estimate"]):
                    ax.hlines(
                        float(ch["estimate"]), x - width * 0.46, x + width * 0.46,
                        color="#333333", linestyles=":", lw=1.0,
                    )
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([CONDITION_LABEL[i] for i in heads], rotation=15)
        ax.set_ylim(0, 1)
        n = _rate(R, "maem_reg", "hit_any", HEADLINE_BUDGET, fam, "focus", band)
        ax.set_title(f"{title} ({_panel_n(n)})", fontsize=9)
    axes[0].set_ylabel("share of concepts")
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=C.COLOURS[_reader(c)[4]] if _reader(c)[6] else "white",
            edgecolor=C.COLOURS[_reader(c)[4]],
            hatch=None if _reader(c)[6] else "////",
            label=_reader(c)[7],
        )
        for c in BAR_READERS
    ]
    handles.append(
        plt.Line2D([], [], color="#333333", ls=":", label=f"chance ({C.N_DONORS} donors on the word rule; the foil rate on the judge)")
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(BAR_READERS) + 1, fontsize=8, frameon=False)
    fig.suptitle(
        _suptitle(f"Figure {number}. Which concept each reader names at the {band_label}, layer {C.READ_LAYER}", smoke_n),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    _save(fig, out_dir, fig_name)


def fig_final_period(T, out_dir, smoke_n=None):
    _bar_figure(T, out_dir, smoke_n, C.FINAL_BAND, "final period", FIGURES["01"], 1)
    return (
        f"Figure 1. The headline: at the carrier's final period, one activation per item, the share of concepts "
        f"each reader names under `focus`, `mention` and the no-instruction `baseline`. Left: topics on the "
        f"word rule (a member word in any of {HEADLINE_BUDGET} samples; the lens: a member in its top "
        f"{C.TOP_WORD} at layer {C.READ_LAYER}; the corpus search: in any of its {C.TOP_WINDOWS} best-scoring "
        f"windows). Middle and right: the naming judge's verdict against the item's own targets "
        f"({_judge_label(REFERENCE_JUDGE)}) "
        f"on topics and on arithmetic, "
        f"whose word rule is not identification evidence. Blue: MAEM; pink: the NLA verbalizer; green: the "
        f"corpus search over the held-out corpus's {CORPUS_TOKENS_M}M-token search prefix, queried with the "
        f"same activation; orange: the "
        f"J-lens (no prose of its own on the word rule; its judged bars read the summary of its top-{C.TOP_WORD}); "
        f"hatched grey: the null-direction control, which reads no activation; vermillion: Patchscopes at layer "
        f"{C.PATCH_LAYER}, and hatched vermillion its no-patch floor (word rule only). Each bar "
        f"carries its own "
        f"chance line as a dotted tick: the {C.N_DONORS}-donor chance line on the word rule, and on the judge "
        f"the same readouts' rate against a foil concept's targets, the reader's false-positive line. "
        f"Wilson 95 % intervals."
    )


def fig_carrier_mean(T, out_dir, smoke_n=None):
    _bar_figure(T, out_dir, smoke_n, C.MEAN_BAND, "carrier mean", FIGURES["03"], 3)
    return (
        f"Figure 3 (Appendix A). Figure 1 read at the carrier mean instead of the final period: one synthetic "
        f"activation per item, the mean of the layer-{C.READ_LAYER} residual over the copied sentence. Same "
        f"panels, readers and intervals. The null control and the Patchscopes floor are not regenerated here: "
        f"neither reads an activation, so their `mean` bars ARE their final-period readouts, one floor each "
        f"rather than two."
    )


def fig_control_conditions(T, out_dir, smoke_n=None):
    _bar_figure(T, out_dir, smoke_n, C.FINAL_BAND, "final period", FIGURES["05"], 5, heads=CONTROL_HEADS)
    return (
        "Figure 5 (Appendix C). The control instructions at the final period: `mention`, which names the "
        "concept and asks nothing, beside `ignore` and `don't-think`, which name it and tell the model to "
        "dismiss it, and the no-instruction floor. Same panels, readers and intervals as figure 1."
    )


def fig_modulation(T, out_dir, smoke_n=None):
    """Each reader's focus − baseline and mention − baseline at the final period, paired by concept."""
    plt = _mpl()
    MOD = T.get("modulation") or []
    fig, axes = plt.subplots(1, len(BAR_PANELS), figsize=(12.0, 4.2), sharex=True)
    for ax, (title, fam, inst) in zip(axes, BAR_PANELS):
        ylabels, ys, y = [], [], 0
        for cond in BAR_READERS:
            _c, metric, _b, jcond, colour, mk, filled, label = _reader(cond)
            for j, instr in enumerate(C.HEADLINE_CONDITIONS):
                if inst == "rule":
                    r = _mod(MOD, cond, metric, C.FINAL_BAND, fam, instr)
                else:
                    r = _mod(MOD, cond, "named", C.FINAL_BAND, fam, instr, name=REFERENCE_JUDGE) if jcond else None
                col = C.COLOURS[colour]
                if r is not None and not _is_unavailable(r["estimate"]):
                    e = np.array(_err(r), dtype=float).reshape(2, 1) * 100.0
                    ax.errorbar(
                        [100.0 * float(r["estimate"])], [y], xerr=e,
                        fmt=mk if j == 0 else ("s" if mk != "s" else "o"),
                        ms=5, color=col, mfc=col if j == 0 else "white", capsize=2,
                    )
                ylabels.append(f"{label}, {CONDITION_LABEL[instr]}")
                ys.append(y)
                y -= 1
            y -= 0.5
        ax.axvline(0, color=C.COLOURS["control"], lw=0.8, ls=":")
        ax.set_yticks(ys if ax is axes[0] else [])
        if ax is axes[0]:
            ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("minus baseline (pp)")
    fig.suptitle(
        _suptitle(
            "Figure 2. Does the instruction move the concept into the final period?\nEach reader's rate under "
            "focus (filled) and mention (hollow) minus its rate under no instruction",
            smoke_n,
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out_dir, FIGURES["02"])
    return (
        "Figure 2. The modulation at the final period: each reader's rate under `focus` (filled) and under "
        "`mention` (hollow) minus its own rate under the no-instruction `baseline`, paired by concept, in "
        "percentage points with the bootstrap 95 % interval. Topics on the word rule and on the judge, "
        f"arithmetic on the judge ({_judge_label(REFERENCE_JUDGE)}). The null control's rows are the "
        "calibration: a pipeline that reads no activation should not move with the instruction."
    )


def fig_lens_protocol(T, out_dir, smoke_n=None):
    """The J-lens paper's own protocol: the lens's two criteria over every carrier token."""
    plt = _mpl()
    R = T["rates"]
    series = (
        ("jlens_L42", "rank10_L42_any", C.TOP_WORD, True, f"rank ≤ {C.TOP_WORD} at layer {C.READ_LAYER}"),
        ("jlens_best", "rank1_any", 1, False, "rank 1, any fitted layer (target-informed)"),
    )
    fig, axes = plt.subplots(1, len(C.FAMILIES), figsize=(10.0, 4.2), sharey=True)
    order = ("baseline", "mention", "ignore", "dont_think", "focus")
    for ax, fam in zip(axes, C.FAMILIES):
        for k, (cond, metric, cutoff, filled, _lab) in enumerate(series):
            for j, instr in enumerate(order):
                rr = _rate(R, cond, metric, cutoff, fam, instr, C.CARRIER_BAND)
                x = j + (k - 0.5) * 0.25
                col = C.COLOURS["jlens"]
                if rr is None or _is_unavailable(rr["estimate"]):
                    ax.text(x, 0.02, "n/a", ha="center", va="bottom", fontsize=6, color=col, rotation=90)
                    continue
                ax.errorbar(
                    [x], [float(rr["estimate"])], yerr=np.array(_err(rr), dtype=float).reshape(2, 1),
                    fmt="s", ms=5, color=col, mfc=col if filled else "white", capsize=2,
                )
                ch = _chance(R, cond, cutoff, fam, instr, C.CARRIER_BAND)
                if ch is not None and not _is_unavailable(ch["estimate"]):
                    ax.hlines(float(ch["estimate"]), x - 0.1, x + 0.1, color=col, linestyles=":", lw=1.0)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CONDITION_LABEL[i] for i in order], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(fam, fontsize=9)
    axes[0].set_ylabel("share of concepts")
    handles = _legend_handles(plt, [(lab, "jlens", "s", filled) for _c, _m, _k, filled, lab in series]) + [
        plt.Line2D([], [], color=C.COLOURS["jlens"], ls=":", label=f"chance ({C.N_DONORS} donors)")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle(
        _suptitle("Figure 4. The J-lens paper's own protocol: a hit at any token of the copied sentence", smoke_n),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    _save(fig, out_dir, FIGURES["04"])
    return (
        f"Figure 4 (Appendix B). The released J-lens under its own criterion — a target form in its top "
        f"{C.TOP_WORD} at layer {C.READ_LAYER} at ANY token of the copied sentence (filled), and at rank 1 at "
        f"any token and any of the 63 fitted layers (hollow: a target-informed selection over layers, never a "
        f"reading at one cell) — per family and instruction, with each criterion's own "
        f"{C.N_DONORS}-donor chance line as a dotted tick. Wilson 95 % intervals."
    )


# --- the examples, selected by rule before any text is read -------------------------------------------------


def _by_meta(run, rel, judge=REFERENCE_JUDGE):
    """The reference judge's naming log keyed by (item, cell, condition, vs), each record bound to its readout."""
    out = {}
    bound = A.bind_log(A._load_log(run, rel), A.judged_now(run)["naming"], judge, "naming")
    for recs in bound.values():
        for rec in recs:
            m = rec.get("meta") or {}
            out[(m.get("i"), m.get("pos"), m.get("condition"), m.get("vs"))] = rec
    return out


def _verdict_summary(rec):
    """What the examples quote of one naming verdict."""
    v = verdict_of(rec)
    if v is None or not v["parse_ok"]:
        return "no verdict"
    return {"named": v["named"], "target": v["target"], "quote": v["quote"], "voided": bool(v["voided"])}


def examples(run, scores, per_cell=2):
    """Per family, under focus, the 2x2 of MAEM's word rule against the lens's rank ≤ 10 at layer 42 at the
    final period; the first `per_cell` concepts by item id in each cell."""
    items = {x["i"]: x for x in run.read_json("data/items.json")["items"] if not x["excluded"]}
    reg = {r["i"]: {c["pos"]: c for c in r["cells"]} for r in run.read_json(merged_rel(C.HEADLINE_ARM))["items"]}
    lens = {r["i"]: {c["pos"]: c for c in r["cells"]} for r in run.read_json("lens/lens.json")["items"]}
    summaries = run.read_json("judges/summaries.json").get("cells") or {}
    rel = A.judge_log_rel(REFERENCE_JUDGE, "naming")
    ident = _by_meta(run, rel) if run.exists(rel) else {}
    _m, maem_hit = A.metric_accessor("maem_reg", "hit_any")
    _l, lens_hit = A.metric_accessor("jlens_L42", "rank10_L42_any")
    by_i = {s["i"]: s for s in scores}
    out = []
    for fam in C.FAMILIES:
        cells = {"both": [], "maem_only": [], "lens_only": [], "neither": []}
        for s in sorted(
            (s for s in scores if s["family"] == fam and s["instruction"] == "focus"), key=lambda s: s["i"]
        ):
            m_hit = bool(maem_hit(s, C.FINAL_BAND))
            l_hit = bool(lens_hit(s, C.FINAL_BAND))
            key = "both" if (m_hit and l_hit) else "maem_only" if m_hit else "lens_only" if l_hit else "neither"
            cells[key].append(s["i"])
        for cell in ("both", "maem_only", "lens_only", "neither"):
            for i in cells[cell][:per_cell]:
                s, it = by_i[i], items[i]
                pos = s["final_pos"]
                rcell = (reg.get(i) or {}).get(pos) or {}
                lcell = (lens.get(i) or {}).get(pos) or {}
                judge = {vs: _verdict_summary(ident.get((i, pos, "maem_reg8", vs))) for vs in C.VS}
                out.append(
                    {
                        "i": i,
                        "family": fam,
                        "cell": cell,
                        "instruction": "focus",
                        "concept": s["concept"],
                        "forms": s["forms"],
                        "user": it.get("user", ""),
                        "carrier": s["carrier"],
                        "final_pos": pos,
                        "maem_greedy": (rcell.get("greedy") or {}).get("text", "unavailable"),
                        "maem_sample_1": (rcell.get("samples") or [{}])[0].get("text", "unavailable"),
                        "top10_L42": lcell.get("top10_L42", "unavailable"),
                        "rank_L42": lcell.get("rank_L42"),
                        "judge": judge,
                        "summary_L42": (summaries.get(f"{i}/{pos}") or {}).get("summary"),
                    }
                )
    return out


# --- report.md ----------------------------------------------------------------------------------------------

LIMITATIONS = [
    "Nothing here is about a global workspace or consciousness; that is the source paper's framing, not a "
    "claim this evaluation can make.",
    "The instruction task is artificial and the concepts are common category members. A bare mention primes "
    "much of what an instruction to focus does, so a positive result reads \"MAEM reads a concept the prompt "
    "made salient\", never \"MAEM reads what the model chose to think about\" — the focus − mention row is "
    "the one that separates them.",
    "One base model, one inverter, one read layer ({read_layer}), {n_concepts} concepts on {n_carriers} "
    "carriers.",
    "Positions are matched across readers; layers and output budgets are not. MAEM reads layer 42 only; the "
    "lens is reported at 42 and, in Appendix B, over all 63 fitted layers, and the any-layer number picks its "
    "layer with the target, so it is a target-informed selection reported in its own column.",
    "The headline reads one activation per item, so no rate here grows with the number of cells a reader was "
    "given. Appendix B's lens rows are the exception and say so: they are the source paper's own any-position "
    "criterion, over every token of the sentence.",
    "The judged fold is one-sided: a cell whose reply never came back can cost a credit and never create one, "
    "so every judged rate is a floor by the cells that judge lost, and the missingness table prints the count.",
    "The naming judge never sees the instruction the model was given, so a readout that names the concept by "
    "narrating that instruction (\"I must keep citrus fruits in mind\") is credited like one that engages it; "
    "the null-direction control, which reads no activation from the same prompt, is the floor for that, and "
    "the foil column is the judge's false-positive line.",
    "The lens's readout is prose written by the judge's own model. The summariser is blind to the item, so "
    "the prose cannot be informed by the answer, but the judge reads prose its own model wrote.",
    "The carrier mean averages over a sentence and keeps what is constant across it — here, the copying task. "
    "Its lens rows are the matched layer only, because the synthetic cell is stored at that layer alone.",
    "The null control's `mean` row IS its `final` row: it reads no activation, so the two cannot differ, and "
    "they are one floor rather than two. The Patchscopes floor's is the same, for the same reason.",
    f"Patchscopes is read with one fixed recipe (layer {C.PATCH_LAYER}, the placeholder's residual replaced by "
    f"{C.PATCH_ALPHA:g} x its own norm x the centred direction), not tuned on this evaluation; its rate is "
    "read against its own no-patch floor.",
    "The corpus search reads text nobody wrote about this activation: a hit means the corpus contains a "
    "window whose own residuals point along the query AND that mentions the concept, so its rate is bounded by what "
    "the corpus happens to hold as well as by the search. A gap in MAEM's favour is not evidence that no "
    "corpus could close it.",
    "The untrained-base row is an ablation of MAEM and not a method: it removes the training and keeps "
    "the injection, the null-direction control removes the injection and keeps the trained model, and "
    "neither is a reader anyone would use. It is read at the final period, under the word rule, and by no "
    "judge.",
]


def _sec_header(scores, cov, smoke, run):
    out = []
    if smoke:
        out.append(f"# {SMOKE_BANNER.format(n=len(scores))}\n")
    out.append("# Workspace modulation: report\n")
    out.append(
        f"**Question.** Given the layer-{C.READ_LAYER} residual at the final period of a sentence the model is "
        "copying under an instruction to think about, ignore, or not think about a concept, does the inverter "
        "generate text that names that concept, above the no-instruction baseline, above a null direction and "
        "above a 20-donor target-shuffle chance line, and how does its rate compare with the released Jacobian "
        "lens and a natural-language autoencoder read at the same activation? (methodology, question)\n"
    )
    out.append("## Configuration\n")
    out.append(f"- inverter: `{C.INVERTER}` @ `{C.INVERTER_REVISION}` (a full-parameter checkpoint)")
    out.append(f"- base model: `{C.MODEL}` @ `{C.MODEL_REVISION}`; MAEM read layer: {C.READ_LAYER}")
    out.append(f"- lens: `{C.LENS_REPO}` @ `{C.LENS_REVISION}`, file `{C.LENS_FILE}`")
    out.append(
        f"- NLA verbalizer: `{NLA.repo}` @ `{NLA.revision}` (a merged full checkpoint), up to "
        f"{NLA.max_new} new tokens at the package's own sampling"
    )
    out.append(
        f"- Patchscopes: prompt `{C.PATCH_PROMPT_ID}` on the clean base, the `{C.PATCH_PLACEHOLDER}` "
        f"placeholder's block-{C.PATCH_LAYER} output replaced by {C.PATCH_ALPHA:g} x its own norm x "
        f"`{C.PATCH_INPUT[C.PATCH_RULE]}` (`{C.PATCH_RULE}`, prefill only), {C.PATCH_GEN['n_samples']} samples "
        f"and a greedy at the headline arm's decoding; its floor `{C.PATCH_FLOOR}` is the same prompt with no patch"
    )
    out.append(
        f"- judge profile `{C.JUDGE_PROFILE}`: "
        + "; ".join(f"{_judge_label(n)} (`{C.JUDGES[n].model}`)" for n in JUDGE_NAMES)
        + f"; ledger capped at {_fmt_usd(cov.get('budget_cap_usd'))}"
    )
    out.append(f"- {served_line(run.path, C.JUDGES)}")
    out.append(
        f"- lens summariser: {_judge_label(C.SUMMARISER)} (`{C.JUDGES[C.SUMMARISER].model}`), shown the tokens "
        "and nothing else"
    )
    out.append(f"- scoring version {cov.get('scoring_version')}, parse version {cov.get('parse_version')}\n")
    return out


def _summariser_failures(run):
    if not run.exists("judges/summaries.json"):
        return "unavailable"
    cells = (run.read_json("judges/summaries.json") or {}).get("cells") or {}
    return f"{sum(1 for v in cells.values() if not v.get('summary'))} of {len(cells)}"


def _counts_phrase(counts, empty="none"):
    """A {reason: count} block as prose."""
    if not isinstance(counts, dict):
        return "unavailable" if counts is None else str(counts)
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or empty


def _excluded_concepts_phrase(rows):
    """The concepts dropped before any readout, by name and reason."""
    if not isinstance(rows, list):
        return "unavailable"
    return "; ".join(f"{display} (concept {key}): {reason}" for key, display, reason in rows) or "none"


def _reassigned_phrase(rows):
    """The carriers that were reassigned, and why."""
    if not isinstance(rows, list):
        return "unavailable"
    if not rows:
        return "none"
    per = "; ".join(
        f"{r.get('concept')} (concept {r.get('concept_key')}): carrier {r.get('from')} -> {r.get('to')}, "
        f"{r.get('reason')}"
        for r in rows
    )
    return f"{len(rows)}: {per}"


def _retrieval_lines(rec):
    """What the corpus search ran over, and how many queries' best window may be a near copy."""
    if not isinstance(rec, dict):
        return [f"- the corpus search: {rec if rec is not None else 'unavailable'}"]
    corpus = rec.get("corpus") or {}
    return [
        f"- the corpus search: {rec.get('n_queries')} queries (both read positions of every item) against "
        f"{corpus.get('n_windows')} windows of up to {corpus.get('window')} tokens at stride "
        f"{corpus.get('stride')} from {corpus.get('n_docs')} held-out documents "
        f"({corpus.get('doc_min')}-{corpus.get('doc_max')}, {corpus.get('n_tokens')} tokens), keeping the "
        f"best {rec.get('top_k')} non-overlapping windows per query on the {rec.get('metric')} geometry",
        f"- ... of which {rec.get('n_near_duplicates')} query(ies) have a best window above "
        f"{rec.get('near_duplicate_cos')}, which is a possible near copy of the text the query was read "
        "from: counted and reported, never filtered",
        *RET.R.search_bullets(rec.get("search") or []),
        f"- ... {RET.R.SEARCH_NOTE} Every instrument reads the full corpus's windows; the word rule at budget 1 "
        "(`tables/rates.csv`) is the single best window's",
    ]


def _patchscope_lines(rec):
    """The Patchscopes stage's own record as prose: prompt and rule, then per arm its patch check and distinctness."""
    if not isinstance(rec, dict):
        return [f"- Patchscopes checks: {rec if rec is not None else 'unavailable'}"]
    out = [
        f"- Patchscopes: prompt `{rec.get('prompt_id')}` (placeholder `{rec.get('placeholder')}`), rule "
        f"`{rec.get('rule')}` writing {_fmt3(rec.get('alpha'))} x the clean norm x `{rec.get('input')}` at the "
        f"output of block {rec.get('layer')}, prefill only; centring mean sha256 "
        f"`{(rec.get('mu_sha256') or 'unavailable')[:12]}`"
    ]
    for arm, a in (rec.get("arms") or {}).items():
        label = PATCH_FLOOR_LABEL if arm == C.PATCH_FLOOR else PATCH_LABEL
        if a.get("target_layer") is None:
            out.append(
                f"- ... {label} (`{arm}`, no hook installed): {a.get('distinct_sample_texts')} distinct sample "
                f"texts, the one shared set every item carries (unpatched greedy: {a.get('unpatched_greedy')!r})"
            )
            continue
        pc = a.get("patch_check") or {}
        cos, rel, ratio = pc.get("cos_to_v") or [], pc.get("rel_delta") or [], pc.get("norm_ratio") or []
        out.append(
            f"- ... {label} (`{arm}`, layer {a.get('target_layer')}): patch check on {len(pc.get('cells') or [])} "
            f"cells (enforced): min cos(h_patched, v) {_fmt3(min(cos) if cos else None)} (must be ≥ "
            f"{C.PATCH_CHECK_MIN_COS:g}), min ‖Δh‖/‖h‖ {_fmt3(min(rel) if rel else None)} (must be > "
            f"{C.PATCH_CHECK_MIN_REL_DELTA:g}), written norm over clean norm "
            f"{_fmt3(float(np.median(ratio)) if ratio else None)}; norm ratio over every patched cell "
            f"{_fmt3(a.get('norm_ratio_min'))} to {_fmt3(a.get('norm_ratio_max'))}; greedy distinct share "
            f"{_fmt3(a.get('greedy_distinct_share'))}; share of greedies equal to the unpatched one "
            f"{_fmt3(a.get('share_equal_unpatched'))}; {a.get('distinct_sample_texts')} distinct sample texts "
            f"over {a.get('n_cells')} cells of {a.get('n_items')} items"
        )
    return out


def _sec_population(cov, run):
    p = cov["population"]
    cells = p.get("n_cells")
    n_cells = cells.get(C.CARRIER_BAND, "unavailable") if isinstance(cells, dict) else "unavailable"
    out = ["## Population and coverage\n"]
    out.append(
        f"{p['kept']} kept items over {n_cells} carrier cells. Items are "
        "(concept x instruction) pairs, five instruction conditions over the directed-modulation concepts. "
        "Every reader is read at two activations per item: the carrier's final period (the headline) and the "
        "mean of the layer-42 residual over the carrier (Appendix A). The lens is additionally read at every "
        "carrier token, which is the source paper's own protocol (Appendix B).\n"
    )
    out.append(
        _md_table(
            [{"group": g, "items": n} for g, n in p["by_group"].items()]
            + [{"group": f"instruction: {k}", "items": v} for k, v in p["by_instruction"].items()],
            ["group", "items"],
        )
    )
    out.append("")
    out.append(f"- items drawn but excluded, by rule: {_counts_phrase(p.get('excluded'))}")
    out.append(f"- concepts excluded before any readout: {_excluded_concepts_phrase(p.get('excluded_concepts'))}")
    out.append(f"- carriers reassigned, with the rule the first carrier failed: {_reassigned_phrase(p.get('reassigned'))}")
    out.append(
        f"- concepts with a single-token form the lens can be asked for: {p['single_token_lens_targets']}; "
        f"without one (a lens miss, kept in the denominator): {p['no_single_token_form']}"
    )
    d = p["n_donors_used"]
    out.append(
        f"- chance-line donors: {d['drawn']} drawn per item, {d['min']} at fewest and {_fmt3(d['mean'])} on "
        "average actually used (a donor whose forms are missing from the registry is skipped)"
    )
    out.append(f"- cells on an outlying-norm token: {p['norm_outliers']}")
    comp = p["compliance"]
    out.append(
        f"- compliance diagnostic, on the user turn alone on the untrained base: "
        f"{comp['copies_carrier']}/{comp['n']} items copy the carrier, {comp['names_target']}/{comp['n']} name "
        "a target"
    )
    rc = cov["readout_coverage"]
    for band in C.READ_BANDS:
        b = rc.get(band) or {}
        out.append(
            f"- items read at `{band}`: MAEM {b.get('maem_reg')}, the null-direction control "
            f"{b.get('maem_null')}, the lens {b.get('jlens_L42')}, the NLA verbalizer {b.get('nla')}, the "
            f"corpus search {b.get(C.RETRIEVAL_READER)}, Patchscopes {b.get(C.PATCH_ARM)} and its floor "
            f"{b.get(C.PATCH_FLOOR)}, the untrained-base ablation {b.get(C.BASE_ARM)} "
            "(the ablation is generated at the final period alone; the floor's `mean` readout is its final-period "
            "one)"
        )
    car = rc.get(C.CARRIER_BAND) or {}
    out.append(f"- the lens over the whole carrier band: {car.get('jlens_L42')} items, {car.get('cells')} cells")
    out += _retrieval_lines(cov.get("retrieval"))
    out += _patchscope_lines(cov.get("patchscope"))
    out.append(
        f"- lens summaries the summariser did not return: {_summariser_failures(run)} (read from "
        "judges/summaries.json's own per-cell statuses; an unavailable summary is an unjudged lens cell, "
        "never a lens miss)"
    )
    out += _injection_lines(cov.get("injection_check"))
    out += _norm_filter_lines(cov)
    out.append(f"- the verbalizer's own guards: {cov.get('nla_guards')}")
    out.append(f"- the synthetic mean cell's norm ratios: {cov.get('mean_cell_norms')}")
    out.append("")
    return out


def _sec_headline(T, captions):
    out = ["## Headline: the final period, focus and mention\n"]
    out.append(
        "The claims are read at the carrier's **final period**, one activation per item: the sentence-final "
        "delimiter, the position at which the model is taken to summarise what it holds in its workspace "
        "(methodology, headline scope). They are read under the two instructions that put the concept in the "
        "workspace — `focus` (\"concentrate on X while you write\") and `mention` (the bare name in "
        "parentheses) — against the no-instruction `baseline` as the floor. Topics and arithmetic are never "
        "pooled. The carrier mean, the J-lens paper's own any-token protocol and the two dismissal "
        "instructions are in the appendices.\n"
    )
    out.append(READING_NOTE + "\n")
    out += _family_blocks(T, C.FINAL_BAND)
    for k in ("01", "02"):
        if k in captions:
            out.append(f"![figure {int(k)}](figures/{FIGURES[k]}.png)\n")
            out.append(captions[k] + "\n")
    return out


def _sec_appendix_mean(T, captions):
    out = ["## Appendix A: the carrier mean\n"]
    out.append(
        f"One synthetic activation per item: the mean of the layer-{C.READ_LAYER} residual over every token of "
        "the copied sentence, from its first token to the final period. It is stored raw and uncentred, "
        "exactly as a token's row is, and every reader then prepares it the way it prepares any cell — MAEM "
        "centres it by the centring mean and normalises it, the NLA verbalizer reads it uncentred, the lens "
        "transports and unembeds it. Averaging over a sentence keeps what is constant across it, and what is "
        "constant here is the copying task, so a difference from the headline is a statement about that and "
        "not about the reader. The null-direction control is not regenerated here — it reads no activation, so "
        "its mean row IS its final-period row — and its two rows are one floor, never two. The same holds for "
        "the Patchscopes floor, which installs no patch.\n"
    )
    out += _family_blocks(T, C.MEAN_BAND)
    if "03" in captions:
        out.append(f"![figure 3](figures/{FIGURES['03']}.png)\n")
        out.append(captions["03"] + "\n")
    return out


def _sec_appendix_lens(T, captions):
    R = T["rates"]
    out = ["## Appendix B: the J-lens paper's own protocol\n"]
    out.append(
        f"The source paper's criterion is a hit at ANY token of the copied sentence, with the lens free to "
        f"pick its layer. This appendix reads it: a target form in the lens's top {C.TOP_WORD} at layer "
        f"{C.READ_LAYER} at any carrier token, and at rank 1 at any carrier token and any of the 63 fitted "
        f"layers. The lens alone is read here — no reader generates text at every carrier token, so there is "
        f"nothing to put beside it — and the any-layer row is a target-informed selection over layers, never a "
        f"reading at one cell. It is not comparable with the headline, where every reader is given one "
        f"activation.\n"
    )
    cols = ["group", "instruction", f"rank ≤ {C.TOP_WORD} @ {C.READ_LAYER} %", "chance %", "rank 1, any layer %", "chance %, any layer"]
    rows = []
    for fam in C.FAMILIES:
        for instr in C.CONDITIONS:
            r10 = _rate(R, "jlens_L42", "rank10_L42_any", C.TOP_WORD, fam, instr, C.CARRIER_BAND)
            r1 = _rate(R, "jlens_best", "rank1_any", 1, fam, instr, C.CARRIER_BAND)
            if r10 is None and r1 is None:
                continue
            rows.append(
                {
                    "group": fam,
                    "instruction": CONDITION_LABEL[instr],
                    f"rank ≤ {C.TOP_WORD} @ {C.READ_LAYER} %": _fmt_rate(r10),
                    "chance %": _fmt_chance(_chance(R, "jlens_L42", C.TOP_WORD, fam, instr, C.CARRIER_BAND)),
                    "rank 1, any layer %": _fmt_rate(r1),
                    "chance %, any layer": _fmt_chance(_chance(R, "jlens_best", 1, fam, instr, C.CARRIER_BAND)),
                }
            )
    out.append(_md_table(rows, cols))
    out.append("")
    if "04" in captions:
        out.append(f"![figure 4](figures/{FIGURES['04']}.png)\n")
        out.append(captions["04"] + "\n")
    out += _band8_table(T)
    return out


def _band8_table(T):
    """The eight-layer lens readout at the final period: the word rule and its chance line, and the judged
    named / foil / net over its summary, per family and instruction and for focus+mention pooled."""
    R, J = T["rates"], T.get("judged") or []
    band, metric, cond = C.FINAL_BAND, A.LENS_BAND_WORD_METRIC, A.LENS_BAND_COND
    if not _sel(R, metric=metric, condition=cond, band=band):
        return []
    out = [
        f"**The J-lens over eight layers, final period.** The top-{C.TOP_WORD} word-like tokens of layers "
        f"{', '.join(str(l) for l in C.LENS_BAND_LAYERS)} at the final period, pooled (each token once, case "
        "folded) — a band fixed in advance, the same for every item. The word rule reads the pooled list; "
        "the judge reads its summary, written by the same summariser as the layer-42 prose. The pooled row's "
        "intervals resample concepts.\n"
    ]
    cols = ["group", "instruction", "word rule %", "chance %", "judged named %", "judged foil %", "judged net"]
    rows = []
    for fam in C.FAMILIES:
        for instr in C.CONDITIONS + (A.POOLED_INSTRUCTION,):
            r = _sel(R, metric=metric, condition=cond, group=fam, instruction=instr, band=band)
            if not r:
                continue
            ch = _sel(R, metric="chance", condition=cond, group=fam, instruction=instr, band=band)
            jd = {m: _sel(J, metric=m, condition=C.LENS_BAND_READER, group=fam, band=band, instruction=instr,
                          judge=C.JUDGES[REFERENCE_JUDGE].model) for m in ("named", "foil", "net")}
            rows.append(
                {
                    "group": fam,
                    "instruction": CONDITION_LABEL.get(instr, instr),
                    "word rule %": _fmt_rate(r[0]),
                    "chance %": _fmt_chance(ch[0] if ch else None),
                    "judged named %": _fmt_rate(jd["named"][0]) if jd["named"] else "unavailable",
                    "judged foil %": _fmt_rate(jd["foil"][0]) if jd["foil"] else "unavailable",
                    "judged net": _fmt3(jd["net"][0]["estimate"]) if jd["net"] else "unavailable",
                }
            )
    out.append(_md_table(rows, cols))
    out.append("")
    return out


def _sec_appendix_control(T, captions):
    out = ["## Appendix C: the dismissal instructions (ignore, don't-think)\n"]
    out.append(
        "Under `ignore` (\"X is irrelevant — ignore it\") and `don't-think` (\"whatever you do, do not think "
        "about X\") the concept's name is in the prompt exactly as it is under `mention`. A reader that only "
        "tracked the words in the model's context would score here as it does there; a reader of what the "
        "model holds in its workspace should fall towards the floor. These rows are that check. They are not "
        "evidence about whether the model suppresses the concept or holds it in a form no reader turns into "
        "member words — no instrument here can tell the two apart.\n"
    )
    out += _family_blocks(T, C.FINAL_BAND, heads=CONTROL_HEADS, contrasts=False)
    if "05" in captions:
        out.append(f"![figure 5](figures/{FIGURES['05']}.png)\n")
        out.append(captions["05"] + "\n")
    return out


def _sec_appendix_ablation(T):
    """Appendix D: the untrained-base row, labelled an ablation and never a competing method."""
    out = ["## Appendix D: the untrained-base ablation\n"]
    out.append(
        "The same research prompt, the same activation, the same norm-matched add at the same marker, "
        "written by the **untrained base model**: the activation arrives and no trained reader is there to "
        "turn it into text. This is an **ablation of MAEM**, not a method competing with it, so "
        "it is read at the final period alone, under the word rule alone, and is in no judged instrument and "
        "in no judge's budget.\n"
    )
    out.append(
        "It sits beside the null-direction control because the two remove different halves of the reader: "
        "the control keeps the trained model and hands it a zero direction (a trained reader with nothing "
        "to read), the ablation keeps the direction and removes the training. A rate here above the "
        "control's is what the prompt, the injection and the base model's own writing name with no training "
        "at all; MAEM's own rate is above both or it is reading nothing the ablation does not.\n"
    )
    table = ablation_table(T["rates"])
    if table:
        out += [
            f"The word rule at the final period, pass@{BASE_BUDGET} (% of concepts named):\n",
            table,
            "",
        ]
    return out


def _direction_similarity(check):
    """Mean cos(d_own, d_donor) from the injection check record, or None."""
    v = check.get("own_donor_direction_cos_mean") if isinstance(check, dict) else None
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _injection_lines(check):
    """The injection check as prose: what gated, and the re-read cosines as diagnostics."""
    if not isinstance(check, dict) or "criteria" not in check:
        return [f"- injection check on the headline arm: {check}"]
    crit = check["criteria"]
    if check.get("basis") == BASIS_CONTROL:
        alive = (
            f"{_fmt3(check.get('differs_from_control'))} of its {check.get('n_cells')} greedy readouts differ "
            f"from every greedy of the null-direction control ({check.get('n_control_greedies')} distinct), "
            f"at least {check.get('floor')} required"
        )
    else:
        alive = (
            f"the control was not in sight, so the criterion was read on the arm's own pooled distinct "
            f"greedies, {_fmt3(check.get('distinct_pooled'))} of {check.get('n_cells')} against a floor of "
            f"{check.get('floor')}"
        )
    gap_role = (
        "is reported and gates nothing: at these cells the cosine is too insensitive an instrument for a "
        "verdict taken after every rollout has been paid for"
    )
    sim = _direction_similarity(check)
    out = [
        "- injection check on the headline arm, which guards a dead injection (a direction that never "
        f"reached the forward pass writes the control's text): {alive} — "
        f"{'passed' if crit.get('injection_alive') else 'FAILED'}",
        f"- the greedies' re-read on the clean base: own {_fmt3(check.get('greedy_cos_own_mean'))}, donor "
        f"{_fmt3(check.get('greedy_cos_donor_mean'))}, own − donor {_fmt3(check.get('gap'))} "
        f"[{_fmt3((check.get('gap_ci') or [None, None])[0])}, {_fmt3((check.get('gap_ci') or [None, None])[1])}] "
        f"over {check.get('n_cells')} cells ({check.get('n_fallback')} items on a fallback donor). The gap "
        f"{gap_role} ({'not below zero' if crit.get('gap_not_below_zero') else 'BELOW ZERO'}). No floor is "
        "asserted on the own cosine",
    ]
    if sim is not None:
        out[-1] += (
            f": the centred directions of a cell and of its donor have a mean cosine of {sim:.3f} at these "
            "read positions, and the nearer that is to 1 the less a re-read cosine can tell a readout's own "
            "direction from a donor's, which is why the cosines are diagnostics and not a verdict"
        )
    return out


def _norm_filter_lines(cov):
    """What the scorer's norm filter would have dropped from the headline arm's re-read greedies (the re-read
    itself is unfiltered)."""
    rec = (cov.get("reread_norm_filter") or {}).get("rollouts")
    if not isinstance(rec, dict) or not rec.get("n_tokens"):
        return ["- the re-read's norm-filter tally: unavailable"]
    n, drops = rec["n_tokens"], rec["n_tokens_filter_drops"]
    return [
        f"- the re-read cosine is unfiltered; the scorer's norm filter ({C.REREAD_NORM_FILTER_MULT:g}× a text's "
        f"median norm) would have dropped {drops} of {n} content tokens ({100.0 * drops / n:.3f} %), in "
        f"{rec.get('n_texts_filter_touches')} of {rec.get('n_texts')} texts"
    ]


def _sec_missingness(cov):
    out = ["## Judge missingness\n"]
    rows = cov.get("judge_missingness") or []
    if not rows:
        out.append("No judge log in this directory.\n")
        return out
    out.append(
        "Per judge, instrument and arm: how many requests were made and how each of the ones that did not "
        "yield a verdict failed. The statuses partition the requests, so each row's `n` is their sum; "
        "`refused` is the judge declining and `content_filter` the provider stopping the reply, which is a "
        "fact about the provider and not about the judge. "
        "`empty_readout` rows were never sent — the reader produced nothing, so there was nothing to identify "
        "— and they count in neither the numerator nor the denominator of any judged rate; every other "
        "unanswered case counts in the denominator of nothing and leaves that item unmeasured for that "
        "reader, which is why every judged rate is a floor "
        "([tables/missingness.csv](tables/missingness.csv)).\n"
    )
    cols = ["judge", "instrument", "arm", "n"] + list(A.MISSINGNESS_STATUSES)
    out.append(
        _md_table(
            [{**r, "judge": _judge_label(r["judge"])} for r in rows],
            cols,
        )
    )
    out.append("")
    foils = (cov.get("instrument") or {}).get("foils") or {}
    skipped = foils.get("skipped") or []
    out.append(
        "Every judged readout was asked twice, against the item's own targets and against a foil concept's: "
        "the first of the concept's 20 same-family donors whose forms are not operands of the item's own "
        f"arithmetic expression. {len(skipped)} item(s) had a donor passed over for that reason; each is "
        "listed under `instrument.foils.skipped` in "
        "[tables/coverage_and_costs.json](tables/coverage_and_costs.json).\n"
    )
    return out


def _sec_examples(examples_list):
    out = ["## Examples\n"]
    out.append(
        "Picked by rule before any text was read: per family, under `focus`, the 2x2 of MAEM's "
        f"word rule at pass@{HEADLINE_BUDGET} against the lens's rank ≤ {C.TOP_WORD} at layer {C.READ_LAYER}, "
        "both at the final period, and in each cell the first concepts by item id. Everything quoted is the "
        "readout at that one activation.\n"
    )
    for e in examples_list:
        out.append(f"### item {e['i']} — {e['family']}, {e['cell']}, concept `{e['concept']}`\n")
        out.append(f"- user turn: {e['user']!r}")
        out.append(f"- carrier: {e['carrier']!r} (final period at token {e['final_pos']})")
        out.append(f"- MAEM greedy: {e['maem_greedy']!r}")
        out.append(f"- MAEM sample 1: {e['maem_sample_1']!r}")
        out.append(f"- lens top-{C.TOP_WORD} at layer {C.READ_LAYER}: {e['top10_L42']}")
        out.append(f"- lens summary: {e['summary_L42']!r}")
        out.append(f"- judge ({_judge_label(REFERENCE_JUDGE)}): {e['judge']}")
        out.append("")
    return out


def _launch_label(i, record):
    """How the report names one launch record."""
    stage = record.get("stage")
    if isinstance(stage, str) and stage:
        return f"launch {i} (the `{stage}` stage on its own)"
    sources = record.get("sources")
    if isinstance(sources, list) and len(sources) > 1:
        return f"launch {i} ({len(sources)} orchestrated launches, summed into one record)"
    return f"launch {i}"


def _gpu_lines(run, cov, cost):
    records = launch_records(run)
    total = launch_gpu_seconds(records)
    if total is None:
        return [
            f"- GPU seconds ({cov.get('gpu_timing_source', 'unavailable')} container wall time, not billed "
            f"time): {cov.get('gpu_seconds_measured', 'unavailable')} — no `provenance/modal*.json` launch "
            "record in this run directory, so this is whatever provenance.json itself carries and may be one "
            f"launch's only; estimated GPU cost: {_fmt_usd(cost.get('gpu_estimated'))}; GPU billing "
            f"(actual): {cov.get('billing_usd', 'unavailable')}"
        ]
    per = "; ".join(
        (
            f"{_launch_label(i, d)} {float(d['gpu_seconds_measured']):.2f} s"
            if isinstance(d.get("gpu_seconds_measured"), (int, float))
            else f"{_launch_label(i, d)} no measured seconds"
        )
        for i, (_n, d) in enumerate(records, 1)
    )
    est = [d.get("gpu_cost_usd_estimated") for _n, d in records]
    est = [float(x) for x in est if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return [
        f"- GPU seconds ({cov.get('gpu_timing_source', 'unavailable')} container wall time, not billed time), "
        f"summed over the run's {len(records)} launches: **{total:.2f} s** ≈ "
        f"{total / 3600.0:.2f} GPU-hours — {per}",
        "- ... this sum is a **lower bound**: a launch whose orchestrator did not live to write its record "
        "contributes nothing to it, and the per-stage `stage_seconds` are null for the same reason",
        f"- estimated GPU cost at list price, summed over the same records: "
        f"{_fmt_usd(sum(est)) if est else 'unavailable'}; GPU billing (actual): "
        f"{cov.get('billing_usd', 'unavailable')}",
    ]


def _sec_costs(run, cov):
    out = ["## Coverage and costs\n"]
    cost = cov["cost_usd_list_price"]
    rates = ", ".join(f"{m} {i:.2f} / {o:.2f}" for m, (i, o) in (cov.get("rates_per_m") or {}).items())
    out.append(
        "API spend per log, repriced from the logged token counts at list rates (US$ per "
        f"million input / output tokens: {rates}); the per-log figures come from each log, never from the "
        "shared ledger, which cannot be split back into its per-judge shares. **Requests are distinct request "
        "keys, records are log rows:** a byte-identical request is sent and paid for once but recorded once "
        "per requester (the control's two read positions render the same request), so tokens and spend are "
        "summed over unique keys and the duplicate rows are counted rather than charged again:\n"
    )
    rows = []
    for name in C.JUDGES:
        block = cov["judge"].get(name) or {}
        for inst in ("naming", "summary"):
            b = block.get(inst)
            if not isinstance(b, dict):
                continue
            rows.append(
                {
                    "judge": _judge_label(name),
                    "log": inst,
                    "requests": b.get("requests", "unavailable"),
                    "records": b.get("records", "unavailable"),
                    "duplicate records": b.get("duplicate_records", "unavailable"),
                    "input tokens": b.get("input_tokens", "unavailable"),
                    "output tokens": b.get("output_tokens", "unavailable"),
                    "spend (list price)": _fmt_usd(b.get("spend_usd")),
                    "spend (recorded at run time)": _fmt_usd(b.get("spend_usd_recorded")),
                }
            )
    out.append(
        _md_table(
            rows,
            [
                "judge", "log", "requests", "records", "duplicate records",
                "input tokens", "output tokens", "spend (list price)", "spend (recorded at run time)",
            ],
        )
    )
    out.append("")
    out.append(
        f"- this design's own spend, repriced from the logs above at list price: {_fmt_usd(cost.get('total_api'))} "
        f"over {cov.get('api_requests_total', 'unavailable')} distinct requests "
        f"({cov.get('api_records_total', 'unavailable')} log records, of which "
        f"{cov.get('api_duplicate_records_total', 'unavailable')} are per-requester copies of a request "
        "already paid for)"
    )
    out.append(
        f"- the run directory's shared ledger, the cash total the cap was enforced against: "
        f"{_fmt_usd(cov.get('api_spend_recorded_usd'))} of the {_fmt_usd(cov.get('budget_cap_usd'))} cap, over "
        f"{cov.get('ledger_requests', 'unavailable')} requests"
    )
    out += _gpu_lines(run, cov, cost)
    elapsed, unmeasured = cov.get("elapsed_seconds"), cov.get("elapsed_unmeasured_stages") or []
    out.append(
        f"- Elapsed (each stage's own last-recorded seconds, summed over every stage but the render; a stage "
        f"that resumed from its cache records only that resume): "
        f"{elapsed:.3f} s" if isinstance(elapsed, (int, float)) else "- Elapsed: unavailable"
    )
    if unmeasured:
        out.append(f"- ... nothing measures {', '.join(unmeasured)}, so that sum is a lower bound")
    out.append("- full coverage and cost detail: [tables/coverage_and_costs.json](tables/coverage_and_costs.json)\n")
    out.append("Commands (`--output-dir` is the exact run directory):\n")
    cmds = [
        "PYTHONPATH=$PWD python -m evals.downstream.workspace_modulation all --output-dir <run-dir>",
        "PYTHONPATH=$PWD python -m evals.downstream.workspace_modulation report --output-dir <run-dir>",
    ]
    out.append("```bash\n" + "\n".join(cmds) + "\n```\n")
    return out


def render(run, scores, T, cov, captions, examples_list, smoke=False):
    """report.md, in reading order."""
    out = []
    out += _sec_header(scores, cov, smoke, run)
    out += _sec_population(cov, run)
    out += _sec_headline(T, captions)
    out += _sec_appendix_mean(T, captions)
    out += _sec_appendix_lens(T, captions)
    out += _sec_appendix_control(T, captions)
    out += _sec_appendix_ablation(T)
    out += _sec_missingness(cov)
    out += _sec_examples(examples_list)
    p = cov["population"]
    fmt = {
        "read_layer": C.READ_LAYER,
        "n_concepts": p.get("n_concepts", "unavailable"),
        "n_carriers": p.get("n_carriers", "unavailable"),
    }
    out += ["## Limits\n"]
    for b in LIMITATIONS:
        for k, v in fmt.items():
            b = b.replace("{" + k + "}", str(v))
        out.append(f"- {b}")
    out.append("")
    out += _sec_costs(run, cov)
    return "\n".join(out) + "\n"


def stage_report(args, run):
    """Every table, figure, the examples and report.md, from the saved artifacts alone."""
    chash = stage_key("report", args, run)
    started = time.time()
    absent = [
        rel
        for rel in REQUIRED_ARTIFACTS + tuple(merged_rel(arm) for arm in C.ARM_ORDER)
        if not run.exists(rel)
    ]
    if absent:
        raise RuntimeError(
            f"report: {len(absent)} artifact(s) this report reads are missing: {', '.join(absent)}. Every "
            "section is rendered or the render fails; a section left out would publish its absent rows as "
            "measured zeroes."
        )
    A.refuse_unasked_logs(run)
    scores = A.score_items(run)
    with open(run.file("scores/items.jsonl"), "w", encoding="utf-8") as h:
        for s in scores:
            h.write(json.dumps(json_safe(s), ensure_ascii=False, allow_nan=False) + "\n")
    T = A.tables(scores, run.sub("tables"))
    # the paper's Modul. column (paper_tables.py)
    PT.write(T, run.sub("tables"))
    cov = A.coverage_and_costs(run, scores)
    run.write_json("tables/coverage_and_costs.json", cov)
    write_tables({"missingness": cov["judge_missingness"]}, run.sub("tables"))
    # the run's own smoke flag, as `prepare` recorded it
    smoke = bool(run.read_json("data/items.json")["config"]["smoke"])
    smoke_n = len(scores) if smoke else None
    figures = run.sub("figures")
    caps = {
        "01": fig_final_period(T, figures, smoke_n),
        "02": fig_modulation(T, figures, smoke_n),
        "03": fig_carrier_mean(T, figures, smoke_n),
        "04": fig_lens_protocol(T, figures, smoke_n),
        "05": fig_control_conditions(T, figures, smoke_n),
    }
    ex = examples(run, scores)
    md = render(run, scores, T, cov, caps, ex, smoke=smoke)
    with open(run.file("report.md"), "w", encoding="utf-8") as h:
        h.write(md)
    mark_stage(run, "report", chash, {"n_items": len(scores), "n_examples": len(ex)}, started=started)
    print(f"[report] {run.file('report.md')}", flush=True)
