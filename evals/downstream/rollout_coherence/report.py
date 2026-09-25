"""Stage `report` (methodology §7): `analyse(run)`'s tables as long-format CSVs,
`tables/coverage_and_costs.json`, `figures/06b_frontier_matched_centred.{pdf,png}` and `report.md`, from
saved artifacts only."""
import os, textwrap, time

import numpy as np

from evals.downstream.rollout_coherence import config as C
from evals.downstream.rollout_coherence.analysis import analyse
from evals.downstream.common.runs import config_hash, mark_stage
from evals.downstream.rollout_coherence.runs import stage_hashes, stage_names
from evals.downstream.rollout_coherence.stages import STAGES
from evals.downstream.common.figures import is_unavailable
from evals.downstream.common.judge_client import served_line
from evals.downstream.common.judges import REFERENCE_JUDGE
from evals.downstream.common.stats import write_tables

LIMITATIONS_TEXT = (
    "Nothing here is a reconstruction test: the judge is never asked whether a text matches the source, and "
    "the source is the partner because it is the fairest natural text of the same length, not because "
    "matching it is the goal. One base model, one inverter, one layer, one corpus, 400 activations, one "
    "judge. The comparison favours the generated texts on one point: the passage is cut where the text's "
    "length says, so it may end mid-sentence, while a generated text ends where the model stopped. The "
    "inversion axis is MAEM's training objective and retrieval's selection rule, so it favours them, which "
    "is why the two fluency axes are read beside it. A matched target is read standalone behind a sink, a "
    "position no training target occupied."
)


# ---------------------------------------------------------------- small helpers

def _rows(table, **filt):
    return [r for r in table if all(r.get(k) == v for k, v in filt.items())]


def _row1(table, **filt):
    rs = _rows(table, **filt)
    return rs[0] if rs else None


def _is_nan(x):
    return isinstance(x, float) and x != x


def fmt(x):
    if x is None or _is_nan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def fmt_iv(lo, hi):
    return f"[{fmt(lo)}, {fmt(hi)}]"


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(lines)


def _counts(d):
    return ", ".join(f"{k} {v}" for k, v in sorted((d or {}).items())) or "none"


def _count(v):
    """A count cell: an integer, or `n/a` where the table carries nothing for it."""
    return "n/a" if v is None or _is_nan(v) else str(int(v))


JUDGE_DISPLAY = {name: spec.label for name, spec in C.JUDGES.items()}
PRIMARY_JUDGE = REFERENCE_JUDGE
REFUSED_FLAG_SHARE = 0.05         # methodology §6


def refused_flags(pairs_by_judge, threshold=REFUSED_FLAG_SHARE):
    """`[(judge, group, share)]`: the groups whose refused share of orders is above `threshold`."""
    return [(judge, g, c["refused_share"]) for judge, groups in (pairs_by_judge or {}).items()
            for g, c in (groups or {}).items()
            if c.get("refused_share") is not None and not _is_nan(c["refused_share"])
            and c["refused_share"] > threshold]


def refusal_flag_line(pairs_by_judge):
    """§6's flag for the groups above `REFUSED_FLAG_SHARE`, or the statement that there is none."""
    flagged = refused_flags(pairs_by_judge)
    if not flagged:
        return f"No group's refused share exceeds {REFUSED_FLAG_SHARE:.0%} (methodology §6)."
    return ("**Refusal flag (methodology §6):** "
            + "; ".join(f"{g} under {JUDGE_DISPLAY.get(j, j)} refused {share:.1%} of its orders"
                        for j, g, share in flagged)
            + f", above the {REFUSED_FLAG_SHARE:.0%} share at which the tie imputation moves that group's "
              "rates. A refused order is scored as a tie and never re-asked, so these rows read as more tied "
              "than the judge found them; the `net_dropped` rows leave those pairs out.")


MISSINGNESS_NOTE = (
    "**Missingness** — every request, per judge and group, by what came back: `ok` a verdict, `refused` the "
    "model declining, `content_filter` the provider refusing to pass the request to the model at all, "
    "`parse_fail` a reply with no readable choice after its one retry, `unavailable` a request that carries "
    "no verdict. Counts over both slot orders, so `requests` is twice the group's pairs. Anything but `ok` "
    "is imputed as a tie in the `net` rows; `net_dropped` is the same net over the pairs both of whose "
    "orders were answered."
)


def missingness_table(T, setting):
    """The missingness rows as a markdown table, in `C.JUDGES` order."""
    from evals.downstream.rollout_coherence.judge import MISSINGNESS_STATUSES

    order = list(C.JUDGES)
    cells = {}
    for r in T.get("missingness", []):
        if r.get("condition") == setting:
            cells.setdefault((r.get("judge"), r.get("group")), {})[r.get("metric")] = r.get("estimate")
    rows = []
    for judge, group in sorted(cells, key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k[1])):
        c = cells[(judge, group)]
        rows.append([JUDGE_DISPLAY.get(judge, judge), group, _count(c.get("n_requests"))]
                    + [_count(c.get(s)) for s in MISSINGNESS_STATUSES])
    return _md_table(["judge", "group", "requests", *MISSINGNESS_STATUSES], rows)


# ---------------------------------------------------------------- the figure

GREEDY_POINT = "greedy"
FIGURE = "06b_frontier_matched_centred"
X_KEY = "cos_centred"
X_LABEL = ("Inversion: mean re-read cosine with the target activation "
           "(centred: the identical activation scores 1)")
AXIS_MAX_SHORT = "axis maximum (the identical activation)"
AXIS_MAX_WRAP = 22
# Candidate offsets for a knob tag and vertical steps for a method label, tried in order.
END_TAG_OFFSETS = (((5, -11), "left"), ((5, 6), "left"), ((-5, -11), "right"), ((-5, 6), "right"),
                   ((5, -20), "left"), ((5, 15), "left"), ((-5, -20), "right"), ((-5, 15), "right"))
LABEL_DY_STEPS = (0, -12, 12, -24, 24, -36, 36)
END_TAG_FONTSIZE = 7.0
STAR_SIZE, GREEDY_SIZE, POINT_SIZE, HIGHLIGHT_SIZE = 16.0, 6.5, 4.6, 8.0

HIGHLIGHT_POINTS = (f"k={C.MAIN_K}",) + ((f"k={C.EXTENDED_K[-1]}",) if C.EXTENDED_K else ())


def highlighted(point):
    """Whether this point is enlarged: one of `HIGHLIGHT_POINTS` on an arm of `C.EXTENDED_ARMS`."""
    return point.get("method") in C.EXTENDED_ARMS and point.get("point") in HIGHLIGHT_POINTS


def marker_size(point):
    """The marker size of this point, shared by the drawing and the collision test."""
    if point.get("method") == "source":
        return STAR_SIZE
    if point.get("point") == GREEDY_POINT:
        return GREEDY_SIZE
    return HIGHLIGHT_SIZE if highlighted(point) else POINT_SIZE


def frontier_curves(points):
    """`[(method, [point, ...])]`: one curve per method, in first-appearance order, each sorted by `order`."""
    order, by = [], {}
    for p in points or []:
        if p["method"] not in by:
            order.append(p["method"])
            by[p["method"]] = []
        by[p["method"]].append(p)
    return [(m, sorted(by[m], key=lambda p: p.get("order") or 0)) for m in order]


def frontier_judges(points, judges=None):
    """The judges these points carry a net for, in `C.JUDGES` order."""
    named = [j for j in (judges or C.JUDGES) if j in C.JUDGES] or list(C.JUDGES)
    return [j for j in named if any(j in (p.get("net") or {}) for p in (points or []))]


def _fp_y(point, axis, judge=None):
    """One point's `[est, lo, hi]` on a panel's y axis (judged net or likelihood gap), or None."""
    return (point.get("net") or {}).get(judge) if axis == "net" else point.get("ll_gap")


def _fp_x(point):
    """One point's `[est, lo, hi]` on the x axis, or None."""
    return point.get(X_KEY)


def _fp_xy(point, axis, judge=None):
    """The point's `(x, y)` on one panel, or None when either is unmeasured (never drawn at zero)."""
    x, y = _fp_x(point), _fp_y(point, axis, judge)
    if x is None or y is None or is_unavailable(x[0]) or is_unavailable(y[0]):
        return None
    return float(x[0]), float(y[0])


def _err(triple):
    """The asymmetric `[[below], [above]]` errorbar extent, clamped at zero, or None."""
    if triple is None or is_unavailable(triple[0]):
        return None
    est, lo, hi = triple
    return [[max(0.0, est - lo)], [max(0.0, hi - est)]]


def _frontier_label_offset(index, x, rest_x, x_lo, x_hi):
    """`(offset, ha)` for one curve's label: inwards near the frame, else away from the rest of its curve."""
    span = x_hi - x_lo
    dy = 12 if index % 2 == 0 else -16
    if span > 0 and x > x_lo + 0.76 * span:
        side = "left"
    elif span > 0 and x < x_lo + 0.24 * span:
        side = "right"
    else:
        side = "left" if (rest_x is None or x <= rest_x) else "right"
    # label on the same side as its own points: put it below them
    if rest_x is not None and ((side == "right") == (rest_x > x)):
        dy = -16
    return ((-10, dy), "right") if side == "left" else ((10, dy), "left")


def _text_rect(ax, xy, offset, ha, text, fontsize):
    """The display-space box an annotation will occupy, estimated from its character count."""
    scale = ax.figure.dpi / 72.0
    ax_x, ax_y = ax.transData.transform(xy)
    x = ax_x + offset[0] * scale
    y = ax_y + offset[1] * scale
    w = 0.62 * fontsize * max(1, len(str(text))) * scale
    h = 1.25 * fontsize * scale
    x0 = x if ha == "left" else x - w
    return (x0, y - 0.3 * h, x0 + w, y + 0.7 * h)


def _rect_overlap(a, b):
    """The area two display-space boxes share (0 when disjoint)."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def frontier_marker_boxes(ax, points, axis, judge):
    """A display-space box per drawn marker on this panel: obstacles for in-panel text."""
    boxes = []
    for p in points or []:
        if p.get("method") in C.Y_REFERENCES:
            continue
        v = _fp_xy(p, axis, judge)
        if v is None:
            continue
        r = (0.5 * marker_size(p) + 2.0) * ax.figure.dpi / 72.0
        cx, cy = ax.transData.transform(v)
        boxes.append((cx - r, cy - r, cx + r, cy + r))
    return boxes


def frontier_line_boxes(ax, points, axis, judge, step=3.0, pad=1.0):
    """Boxes every `step` pixels along each curve and whisker: obstacles for in-panel text."""
    boxes = []

    def segment(a, b):
        (x0, y0), (x1, y1) = ax.transData.transform(a), ax.transData.transform(b)
        n = max(1, int(np.hypot(x1 - x0, y1 - y0) / step))
        for t in np.linspace(0.0, 1.0, n + 1):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            boxes.append((x - pad, y - pad, x + pad, y + pad))

    for method, curve in frontier_curves(points):
        if method in C.Y_REFERENCES:
            continue
        line = [(p, _fp_xy(p, axis, judge)) for p in curve if p["point"] != GREEDY_POINT]
        line = [(p, v) for p, v in line if v is not None]
        for (_, a), (_, b) in zip(line, line[1:]):
            segment(a, b)
        for p, (x, y) in line:
            ex, ey = _err(_fp_x(p)), _err(_fp_y(p, axis, judge))
            if ex:
                segment((x - ex[0][0], y), (x + ex[1][0], y))
            if ey:
                segment((x, y - ey[0][0]), (x, y + ey[1][0]))
    return boxes


def frontier_label_offset(ax, label, obstacles, fontsize=8.0):
    """`(offset, ha)` for one method label: the first candidate clear of `obstacles` and the frame, else
    the least-colliding one."""
    frame = ax.get_window_extent()
    frame_area = frame.width * frame.height
    best, worst = (label["offset"], label["ha"]), None
    flipped = "left" if label["ha"] == "right" else "right"
    for ha, sign in ((label["ha"], 1), (flipped, -1)):
        for dy in LABEL_DY_STEPS:
            offset = (sign * label["offset"][0], label["offset"][1] + dy)
            rect = _text_rect(ax, label["xy"], offset, ha, label["text"], fontsize)
            overlap = sum(_rect_overlap(rect, o) for o in obstacles)
            # text outside the panel's frame is as bad as text on a mark: it crosses the spine or is cut
            inside = _rect_overlap(rect, (frame.x0, frame.y0, frame.x1, frame.y1))
            overlap += ((rect[2] - rect[0]) * (rect[3] - rect[1]) - inside) if frame_area else 0.0
            if overlap <= 0:
                return offset, ha
            if worst is None or overlap < worst:
                best, worst = (offset, ha), overlap
    return best


def frontier_end_offset(ax, xy, text, obstacles, fontsize=END_TAG_FONTSIZE):
    """`(offset, ha)` for a knob tag: the first of `END_TAG_OFFSETS` clear of `obstacles`, else the best."""
    fallback, worst = END_TAG_OFFSETS[0], None
    for offset, ha in END_TAG_OFFSETS:
        rect = _text_rect(ax, xy, offset, ha, text, fontsize)
        overlap = sum(_rect_overlap(rect, o) for o in obstacles)
        if overlap <= 0:
            return offset, ha
        if worst is None or overlap < worst:
            fallback, worst = (offset, ha), overlap
    return fallback


def _draw_frontier_method(ax, method, curve, axis, judge, label_method, index=0, x_range=(0.0, 1.0)):
    """Draws one method's curve; returns `(drawn, annotations)`, the text placed later on final limits."""
    colour = C.METHOD_COLOURS.get(method, "#777777")
    xy = {p["point"]: _fp_xy(p, axis, judge) for p in curve}
    line = [(p, xy[p["point"]]) for p in curve
            if p["point"] != GREEDY_POINT and xy[p["point"]] is not None]
    drawn = 0
    annotations = {"label": None, "endtag": None, "starttag": None}
    if len(line) > 1:
        ax.plot([v[0] for _, v in line], [v[1] for _, v in line], "-", color=colour, linewidth=1.3,
                solid_capstyle="round", zorder=3)
    star = method == "source"
    for p, (x, y) in line:
        ax.errorbar([x], [y], xerr=_err(_fp_x(p)), yerr=_err(_fp_y(p, axis, judge)),
                    fmt="*" if star else "o", markersize=marker_size(p), color=colour,
                    markerfacecolor=colour, markeredgecolor=colour, ecolor=colour, elinewidth=0.7,
                    capsize=0, zorder=6 if star else 4)
        drawn += 1
    greedy = xy.get(GREEDY_POINT)
    if greedy is not None:
        ax.plot([greedy[0]], [greedy[1]], marker="o", markersize=GREEDY_SIZE, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=1.2, linestyle="none", zorder=4)
        drawn += 1
    # the label is anchored at the method's headline point
    head = next((p for p in curve if p["headline"] and xy[p["point"]] is not None), None)
    if head is not None and label_method:
        x, y = xy[head["point"]]
        rest = [v[0] for p, v in line if not p["headline"]] + ([greedy[0]] if greedy else [])
        offset, ha = _frontier_label_offset(index, x, (sum(rest) / len(rest)) if rest else None, *x_range)
        if len(line) > 1 and line[-1][0] is head:
            offset = (offset[0], 12)
        annotations["label"] = {"text": C.METHOD_LABELS.get(method, method), "xy": (x, y),
                                "offset": offset, "ha": ha, "colour": colour}
    if len(line) > 1:
        annotations["endtag"] = {"text": str(line[-1][0]["point"]), "xy": line[-1][1]}
        annotations["starttag"] = {"text": str(line[0][0]["point"]), "xy": line[0][1]}
    return drawn, annotations


def _place_frontier_annotations(ax, annotations, obstacles):
    """Place method labels, then knob tags, clear of everything already on the panel."""
    boxes = list(obstacles)
    for ann in annotations:
        label = ann.get("label")
        if label is None:
            continue
        offset, ha = frontier_label_offset(ax, label, boxes)
        ax.annotate(label["text"], xy=label["xy"], xytext=offset, ha=ha,
                    textcoords="offset points", fontsize=8, color=label["colour"], zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        boxes.append(_text_rect(ax, label["xy"], offset, ha, label["text"], 8.0))
    # far-end tags first: that is where neighbouring curves crowd
    tags = [ann[which] for which in ("endtag", "starttag") for ann in annotations if ann.get(which)]
    for tag in tags:
        # a marker at the tag's own anchor is the point the tag belongs to, not an obstacle for it
        own = ax.transData.transform(tag["xy"])
        near = [b for b in boxes
                if not (b[0] <= own[0] <= b[2] and b[1] <= own[1] <= b[3]
                        and abs((b[0] + b[2]) / 2 - own[0]) < 1.0 and abs((b[1] + b[3]) / 2 - own[1]) < 1.0)]
        offset, ha = frontier_end_offset(ax, tag["xy"], tag["text"], near)
        ax.annotate(tag["text"], xy=tag["xy"], xytext=offset, ha=ha, textcoords="offset points",
                    fontsize=END_TAG_FONTSIZE, color="#444444", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
        boxes.append(_text_rect(ax, tag["xy"], offset, ha, tag["text"], END_TAG_FONTSIZE))


def fig_frontier(points, n_act, path, primary_judge, judges, axis_max=None, setting=""):
    """`figures/06b_frontier_matched_centred` (methodology §7): the judged net and the likelihood gap
    against the centred cosine; missing points are skipped, never raised on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points = list(points or [])
    judges = frontier_judges(points, judges)
    judge_a = primary_judge if primary_judge in judges else (judges[0] if judges else primary_judge)
    # `C.Y_REFERENCES` has no meaningful x: drawn as a horizontal rule, not a curve
    y_refs = [p for p in points if p.get("method") in C.Y_REFERENCES and p.get("headline")]
    curves = [(m, c) for m, c in frontier_curves(points) if m not in C.Y_REFERENCES]
    cosines = [_fp_x(p)[0] for p in points
               if p.get("method") not in C.Y_REFERENCES and _fp_x(p) and not is_unavailable(_fp_x(p)[0])]
    x_range = (min(cosines), max(cosines)) if cosines else (0.0, 1.0)
    span = (x_range[1] - x_range[0]) or 1.0
    # explicit limits: `ax.margins` would add a margin beyond the axis-maximum rule
    x_lo, x_hi = x_range[0] - 0.16 * span, x_range[1] + 0.16 * span
    if axis_max is not None and np.isfinite(axis_max):
        x_hi = max(x_hi, float(axis_max) + 0.05 * span)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.0, 8.6), sharex=True)
    panels = []
    for ax, axis, judge in ((axA, "net", judge_a), (axB, "ll_gap", None)):
        drawn, annotations = 0, []
        for index, (method, curve) in enumerate(curves):
            n, ann = _draw_frontier_method(ax, method, curve, axis, judge,
                                           label_method=(axis == "net"), index=index, x_range=x_range)
            drawn += n
            annotations.append(ann)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#EAEAEA", linewidth=0.6)
        if drawn:
            ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)
            for ref in y_refs:
                y = _fp_y(ref, axis, judge)
                if y is None or is_unavailable(y[0]):
                    continue
                colour = C.METHOD_COLOURS.get(ref["method"], "#777777")
                ax.axhline(float(y[0]), color=colour, linestyle="--", linewidth=1.2, zorder=2)
                # labelled under the rule, first panel only
                if axis == "net":
                    ax.annotate(C.METHOD_LABELS[ref["method"]], xy=(0.30, float(y[0])), xycoords=("axes fraction", "data"),
                                xytext=(0, -3), textcoords="offset points", fontsize=7.5, color=colour,
                                ha="left", va="top", zorder=7,
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
            if axis_max is not None and np.isfinite(axis_max):
                ax.axvline(float(axis_max), color="#444444", linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
                # named on the rule, at the panel's foot (the `source` star can sit on the rule at the top)
                ax.annotate("\n".join(textwrap.wrap(f"{AXIS_MAX_SHORT} = {fmt(axis_max)}",
                                                    AXIS_MAX_WRAP)),
                            xy=(float(axis_max), 0.03), xycoords=("data", "axes fraction"),
                            xytext=(-4, 0), textcoords="offset points",
                            rotation=90, ha="right", va="bottom", fontsize=7.5, color="#444444", zorder=7,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
        else:
            ax.text(0.5, 0.5, "unavailable", transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#555555")
        ax.set_xlim(x_lo, x_hi)
        ax.margins(y=0.26)
        panels.append((ax, axis, judge, annotations))

    label = JUDGE_DISPLAY.get(judge_a, judge_a)
    axA.set_ylabel("Net LLM judge preference\nagainst source passage")
    axA.set_title(f"Coherence, judged by {label} — {n_act} activations" + (f", {setting}" if setting else ""),
                  fontsize=10)
    axB.set_ylabel("Fluency: per-token log-likelihood\nminus the source passage's (nats/token)")
    axB.set_xlabel(X_LABEL + "\nhigher is better on both panels")
    scorer = str(C.FLUENCY_SCORER["model"]).split("/")[-1]
    scorer = scorer[:1].upper() + scorer[1:]
    axB.set_title(f"Fluency: likelihood under an independent language model ({scorer}); "
                  "0 = as probable as the real passage", fontsize=10)

    grey = "#777777"
    # the reference rules are named in-panel; the legend holds the two marks that are not
    handles = [
        Line2D([0], [0], color=grey, marker="o", markersize=POINT_SIZE, linewidth=1.3,
               label="one method: sampling budget k, or corpus size for retrieval (95% intervals)"),
        Line2D([0], [0], color=grey, marker="o", markersize=GREEDY_SIZE, markerfacecolor="white",
               linestyle="none", label="greedy decode (always the most likely token; not a point on the k curve)"),
    ]
    # settle the layout before placing text: placement tests collisions in display coordinates
    fig.tight_layout()
    fig.canvas.draw()
    for ax, axis, judge, annotations in panels:
        _place_frontier_annotations(ax, annotations,
                                    frontier_marker_boxes(ax, points, axis, judge)
                                    + frontier_line_boxes(ax, points, axis, judge))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.012), ncol=2, fontsize=7.5,
               frameon=False)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report.md

MATCHED_SETTING = "matched targets"


def _cell(triple):
    """One `[est, lo, hi]` as a table cell, or `n/a` for a reading this run does not have."""
    return "n/a" if triple is None else f"{fmt(triple[0])} {fmt_iv(triple[1], triple[2])}"


def point_table(points, judges):
    """Every point of every curve as a markdown table, `compared` marking the one a method is compared at."""
    headers = (["method", "kind", "point", "compared", "n", "cosine (raw)", "cosine (centred)"]
               + [f"net ({JUDGE_DISPLAY.get(j, j)})" for j in judges]
               + [f"net, unanswered dropped ({JUDGE_DISPLAY.get(j, j)})" for j in judges]
               + ["Δ LL (nats/token)", "mean tokens"])
    rows = []
    for p in points:
        kind = "reference" if p.get("reference") else "method"
        rows.append([f"`{p['method']}`", kind, p["point"], "yes" if p.get("headline") else "", p["n"],
                     _cell(p.get("cos")), _cell(p.get("cos_centred"))]
                    + [_cell((p.get("net") or {}).get(j)) for j in judges]
                    + [_cell((p.get("net_dropped") or {}).get(j)) for j in judges]
                    + [_cell(p.get("ll_gap")), fmt(p.get("n_tokens"))])
    return _md_table(headers, rows)


def outcomes_table(rows):
    """`frontier_outcomes.csv` as a markdown table, one line per (judge, method, condition)."""
    cells = []
    for r in rows:
        key = (r.get("judge"), r.get("group"), r.get("condition"))
        if key not in cells:
            cells.append(key)
    out = []
    for judge, group, condition in cells:
        get = {m: _row1(rows, judge=judge, group=group, condition=condition, metric=m)
               for m in ("win", "tie", "loss", "at_least_as", "net")}
        net = get["net"]
        out.append([JUDGE_DISPLAY.get(judge, judge), f"`{group}`", condition]
                   + [fmt(get[m]["estimate"]) if get[m] else "n/a" for m in ("win", "tie", "loss", "at_least_as")]
                   + [_cell([net["estimate"], net["ci_lower"], net["ci_upper"]]) if net else "n/a",
                      net["n_valid"] if net else None])
    return _md_table(["judge", "method", "condition", "win", "tie", "loss", "at least as coherent",
                      "net", "n_valid"], out)


def _norm_filter_line(by_name):
    """Per method, what the (unapplied) 10x-median norm filter would have touched."""
    said = []
    for name, rec in (by_name or {}).items():
        if not rec:
            said.append(f"{name}: none recorded")
            continue
        share = rec.get("share_tokens_filter_drops")
        said.append(f"{name}: {rec.get('n_tokens_filter_drops')} of {rec.get('n_tokens')} tokens "
                    f"({'n/a' if share is None else f'{100 * float(share):.3f} %'}) in "
                    f"{rec.get('n_texts_filter_touches')} of {rec.get('n_texts')} texts")
    return "; ".join(said) or "none recorded"


def figure_caption(points, M):
    """The figure's caption, with the `source` star's position read off the points."""
    primary = PRIMARY_JUDGE
    src = next((p for p in points if p["method"] == "source" and p.get("cos_centred")), None)
    tail = (f" The `source` star, the passage each target was read from, re-read like every other text, "
            f"sits at a centred cosine of {fmt(src['cos_centred'][0])}." if src is not None else "")
    return (
        f"Panel A: the judged net against the source passage under {JUDGE_DISPLAY.get(primary, primary)}, wins "
        "minus losses over that method's pairs. Panel B: the per-token log-likelihood gap under "
        f"{C.FLUENCY_SCORER['model']}, the text minus the same cut source passage. x: the centred re-read "
        "cosine against the matched target, on which the identical activation scores 1 (the dashed rule). A "
        f"curve joins one method's knob, the sampling budget k ∈ {list(C.BEST_OF_K)} (up to the draws it made) "
        "or the nested corpus size for retrieval, annotated at both ends; the greedy decode is hollow and off "
        f"the k curve; k = {C.MAIN_K} and k = {C.EXTENDED_K[-1] if C.EXTENDED_K else C.MAIN_K} are enlarged. "
        f"Past k = {C.MAIN_K} x is the exact best-of-k over every draw and y the one judged selection. "
        "`continuation` never sees the activation and is the dashed horizontal rule at its own fluency; "
        "`source` is at net 0 and gap 0 by construction. Intervals are 95 % percentile bootstraps over activations." + tail
    )


def diagnostics(M, T, prov):
    """The run's own checks and coverage, from `coverage_and_costs.json`'s judged block."""
    tg = M.get("targets") or {}
    stages = M.get("stages") or {}
    L = ["## Diagnostics", ""]
    L.append(f"- targets: {tg.get('n')} re-captured, mean {fmt(tg.get('mean_n_tokens'))} tokens, mean raw "
             f"ceiling {fmt(tg.get('mean_ceiling_raw'))}; the norm filter would keep the target's token for "
             f"{(M.get('last_kept') or {}).get('n')} (it is not applied)")
    for method, checks in (M.get("selfcheck") or {}).items():
        L.append(f"- matched self-check under `{method}`: "
                 + "; ".join(f"{w} window min centred {fmt(c.get('min_centred'))}, {c.get('below_tolerance')} "
                             f"below 1 − {c.get('tol')}" for w, c in (checks or {}).items()))
    ret = stages.get("retrieval") or {}
    L.append(f"- possible near-duplicates (a top-1 window whose search cosine exceeds "
             f"{ret.get('near_duplicate_cos', C.SEARCH_CORPUS.near_duplicate_cos)}), per corpus size: "
             + _counts(ret.get("near_duplicates")))
    nla = stages.get("nla") or {}
    L.append(f"- NLA: close rate {fmt(nla.get('close_rate'))} of the sampled generations "
             f"({fmt(nla.get('close_rate_trunc'))} of their {C.WINDOW_TOKENS}-token views); "
             f"{fmt(nla.get('mean_generated_tokens'))} generated tokens on average; "
             f"{nla.get('n_truncated_native')} explanation(s) cut at the {C.NATIVE_MAX_TOKENS}-token window")
    L.append("- the re-read is unfiltered; what the scorer's 10×-median norm filter would have dropped, per "
             "method: " + _norm_filter_line({m: (s or {}).get("norm_filter")
                                             for m, s in stages.items() if m != "targets"}))
    L.append("- pairs whose source partner ran short of the judged text's length, per group: "
             + _counts(M.get("window_short")))
    L.append(f"- skipped texts (empty, no pair built): {(M.get('skipped') or {}).get('total', 0)}"
             + (" — " + _counts((M.get("skipped") or {}).get("by_group"))
                if (M.get("skipped") or {}).get("by_group") else ""))
    L.append(f"- likelihood gap: scored for {(M.get('fluency') or {}).get('n_pairs_scored', 0)} of "
             f"{M.get('n_pairs', 0)} pairs")
    checks = prov.get("frontier_context_fluency_checks") or {}
    part = checks.get("ll_partner") or {}
    dedupe = checks.get("dedupe_exact") or {}
    L.append(f"- fluency checks: one `<bos>` per text {checks.get('bos_ok')}; {dedupe.get('n_text_slots')} "
             f"text slots over {dedupe.get('n_texts')} distinct strings; a re-score in a differently shaped "
             f"batch differs by {fmt(checks.get('ll_repeat_abs_diff'))}; the source passages' LL, mean "
             f"{fmt(part.get('mean'))}, lies in {part.get('range')} for a share of {fmt(part.get('in_range'))}")
    L.append("")
    L += [refusal_flag_line(M.get("pairs")), ""]
    L += [MISSINGNESS_NOTE, "", missingness_table(T, "matched"), ""]
    return L


def render_report(T, cov, prov, points):
    """`report.md`: the run, the figure and its points, win / tie / loss, diagnostics, limitations, costs."""
    M = cov.get("judged") or {}
    judges = [j for j in C.JUDGES if any(j in (p.get("net") or {}) for p in points)] or list(C.JUDGES)
    L = [f"# Rollout coherence - run {prov.get('run_id', 'unknown')}", ""]

    L += ["## Question and run configuration", ""]
    L.append("- Question: does the text MAEM writes for an activation read as coherently and as fluently as "
             "the passage it came from, and how does that trade against how well the text inverts the "
             "activation (methodology.md §1)?")
    L.append(f"- Base model: `{C.MODEL}` @ `{C.MODEL_REVISION}`")
    L.append(f"- Inverter: `{C.INVERTER}` @ `{C.INVERTER_REVISION}`")
    L.append(f"- Corpus: the held-out corpus of `{C.CORPUS['dataset']}` / `{C.CORPUS['config']}` / "
             f"`{C.CORPUS['split']}` @ `{C.CORPUS['revision']}`; the documents come from its evaluation half")
    L.append(f"- Judge profile: `{C.JUDGE_PROFILE}`; judge: "
             + ", ".join(f"{spec.label} = `{spec.model}`" for spec in C.JUDGES.values()))
    # who in fact answered, read off the judge logs
    L.append(f"- A{prov['answered_by'][1:]}")
    S = C.SAMPLING
    L.append(f"- Decoding: temp={S['temp']}, top_p={S['top_p']}, top_k={S['top_k']}, min_p={S['min_p']}, "
             f"max_new={S['max_new']}, min_new={S['min_new']}; draws per activation "
             + ", ".join(f"{a} {C.n_samples(a)}" for a in ("maem", "continuation"))
             + f", nla {C.NLA.n_samples}, each with a greedy decode")
    L.append(f"- Activations: {cov.get('activations')} ({cov.get('rejected')} pool candidates rejected by the "
             f"norm filter); scoring version `{C.SCORING_VERSION}`")
    L.append("")

    L += ["## Coherence and fluency against inversion", ""]
    L.append(f"![](figures/{FIGURE}.png)")
    L.append("")
    L.append(figure_caption(points or [], M))
    L.append("")
    L.append(f"[figure PDF](figures/{FIGURE}.pdf) - [frontier_matched.csv](tables/frontier_matched.csv)")
    L.append("")
    L.append(f"Every point of every curve, over {M.get('n', 0)} activations. `compared` marks the point a "
             f"method is compared at: one draw with no selection (k = {C.HEADLINE_K}) for a generator, the "
             "largest corpus for retrieval. `mean tokens` is read with the cosine, a maximum over positions "
             "that grows with length.")
    L.append("")
    L.append(point_table(points, judges))
    L.append("")

    L += ["## Win, tie and loss", ""]
    L.append("Each judged method's pairs against the source passage, pooled over draws 0-7 (`per_sample`, "
             "activation bootstrap) or the greedy decode alone (Wilson interval on the shares). A pair is a "
             "win or a loss only when the judge agrees with itself across the two slot orders.")
    L.append("")
    L.append(outcomes_table(T.get("frontier_outcomes") or []))
    L.append("")
    L.append("[frontier_outcomes.csv](tables/frontier_outcomes.csv)")
    L.append("")

    L += diagnostics(M, T, prov)

    L += ["## Limitations", "", LIMITATIONS_TEXT, ""]

    L += ["## Costs and reproduction", ""]
    L.append(f"- Judge ledger state: {cov.get('ledger') or {}}")
    gpu_seconds = cov.get("gpu_seconds_measured")
    if gpu_seconds in (None, "unavailable"):
        L.append("- GPU seconds: unavailable")
    else:
        L.append(f"- GPU: {cov.get('gpu_type', 'unavailable')}, {fmt(gpu_seconds)} measured seconds summed over "
                 f"the run's launch records (${fmt(cov.get('gpu_cost_usd_estimated'))} estimated at "
                 f"{fmt(cov.get('gpu_rate_per_hour'))} per hour, {cov.get('gpu_rate_source', 'unavailable')}); "
                 f"billing: {prov.get('billing_usd', 'unavailable')}")
    elapsed, unmeasured = cov.get("elapsed_seconds"), cov.get("elapsed_unmeasured_stages") or []
    # a stage record holds only its last invocation's seconds, so a resumed stage counts only the resume
    L.append(f"- Elapsed (the sum of every stage's own last-recorded seconds, a fanned-out stage once per "
             f"job): {fmt(elapsed) if elapsed is not None else 'unavailable'}"
             + (f"; nothing measures {', '.join(unmeasured)}, so the sum is a lower bound" if unmeasured else ""))
    L.append("")
    L += ["```bash", "export PYTHONPATH=$PWD", "python -m evals.downstream.rollout_coherence all --run-id <run-id>", "```", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- stage

def _seconds_value(v):
    """One timing value as a float: a number, or a per-phase dict's `total`; None when it measures nothing."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return _seconds_value(v.get("total"))
    return None


def _measured_seconds(mapping):
    """A record's `seconds`, else the sum of its `*_seconds` keys, else None."""
    secs = _seconds_value(mapping.get("seconds"))
    if secs is not None:
        return secs
    parts = [s for k, v in mapping.items() if k != "seconds" and k.endswith("_seconds")
             for s in (_seconds_value(v),) if s is not None]
    return sum(parts) if parts else None


def stage_records(run, stage):
    """The stage records one stage wrote: its own, or its per-job records if it fans out."""
    if run.exists(f"stages/{stage}.json"):
        return [stage]
    others = [s for s in STAGES if s != stage and s.startswith(f"{stage}_")]
    names = [n for n in stage_names(run, [f"{stage}_"])
             if not any(n == o or n.startswith(f"{o}_") for o in others)]
    return names or [stage]


def stage_seconds_summary(run):
    """`(total seconds, stages that measured none)` over every stage but `report`: each record, else its
    provenance file (not for a fanned-out stage's jobs, which would count it once per job)."""
    total, unmeasured = 0.0, []

    def measured(record_name, prov_stage):
        rel = f"stages/{record_name}.json"
        if not run.exists(rel):
            return None
        secs = _measured_seconds(run.read_json(rel))
        if secs is not None or prov_stage is None:
            return secs
        prel = f"provenance/{prov_stage}.json"
        return _measured_seconds(run.read_json(prel)) if run.exists(prel) else None

    for s in STAGES:
        if s == "report":
            # `stages/report.json` holds the previous render's seconds, not part of the run
            continue
        names = stage_records(run, s)
        fanned = names != [s]
        for n in names:
            secs = measured(n, None if fanned else s)
            if secs is None:
                unmeasured.append(n)
            else:
                total += secs
    return total, unmeasured


def report_hash_inputs(run):
    """What `report_config_hash` digests, as a dict: the bootstrap settings and every upstream stage record."""
    return {"scoring": C.SCORING_VERSION, "parse": "1", "n_boot": C.N_BOOT, "boot_seed": C.BOOT_SEED,
            "upstream": stage_hashes(run, ["prepare", "capture"] + stage_names(run, ["frontier_"]))}


def report_config_hash(run):
    """The `report` marker, chained to every stage record the tables were rendered from."""
    return config_hash(report_hash_inputs(run))


def stage_report(args, run):
    """Renders every table, the figure and report.md; always runs (a pure function of the directory)."""
    chash = report_config_hash(run)
    started = time.time()

    res = analyse(run)
    T = res["tables"]
    cov = dict(res["coverage"])
    cov["elapsed_seconds"], cov["elapsed_unmeasured_stages"] = stage_seconds_summary(run)
    write_tables(T, run.sub("tables"))
    run.write_json("tables/coverage_and_costs.json", cov)

    M = cov.get("judged") or {}
    fig_frontier(res["points"] or [], M.get("n", cov["activations"]),
                 os.path.join(run.sub("figures"), FIGURE), PRIMARY_JUDGE, list(res["records"]),
                 axis_max=M.get("axis_max_centred", 1.0), setting=MATCHED_SETTING)

    prov = dict(cov.get("provenance") or {})
    prov["run_id"] = os.path.basename(run.path.rstrip("/")) or os.path.basename(os.path.dirname(run.path))
    prov["answered_by"] = served_line(run.path, C.JUDGES)
    with open(run.file("report.md"), "w", encoding="utf-8") as h:
        h.write(render_report(T, cov, prov, res["points"]))
    mark_stage(run, "report", chash, started=started)
