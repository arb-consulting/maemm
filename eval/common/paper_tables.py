"""The paper's two workspace tables as fragments each workspace package fills its own columns of:
`paper_workspace_word_rule` (the word rule at pass@8) and `paper_workspace_judged_net` (judged naming at
pass@8, net = named - foil). Both have the six `METHODS` rows; workspace_understanding writes the Assoc.
and Multi-hop columns, workspace_modulation the Modul. column.

Each is written under the run's tables/ as a `.tex` fragment of table rows (no environment, no header) and
a `.csv` giving every cell's interval, denominators and source row. A method a package does not run prints
`---` and has no csv row.
"""

import csv
import math
import os

METHODS = ("MAEMM", "NLA", "J-lens", "J-lens, 8 layers", "Patchscopes", "Corpus search")
LABELS = {"J-lens, 8 layers": "J-lens (L36\u201350)"}
TEX_LABELS = {"MAEMM": r"\method{}", "J-lens": "$J$-lens", "J-lens, 8 layers": "$J$-lens (L36--50)"}
WORD_RULE = "paper_workspace_word_rule"
JUDGED_NET = "paper_workspace_judged_net"
MISSING = "---"
CSV_FIELDS = (
    "table",
    "method",
    "label",
    "column",
    "estimate",
    "ci_lower",
    "ci_upper",
    "n_total",
    "n_valid",
    "ci_method",
    "source_table",
    "metric",
    "condition",
    "group",
    "instruction",
    "band",
    "budget",
    "judge",
)


def fmt_cell(r, digits=3):
    """One table cell: the estimate to `digits` places, a minus as LaTeX math, `---` for no row."""
    if r is None:
        return MISSING
    v = r.get("estimate")
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return MISSING
    s = f"{float(v):.{digits}f}"
    if s.startswith("-"):
        s = s[1:]
        # a value that rounds to zero prints without a sign
        if float(s) != 0:
            s = "$-$" + s
    return s


def label(method):
    """The row label the paper prints for a METHODS key."""
    return LABELS.get(method, method)


def tex_label(method):
    """The row label as the paper's LaTeX source writes it."""
    return TEX_LABELS.get(method, method)


def tex_rows(columns, cells):
    """The fragment's body: one `Label & ... \\\\` line per METHODS entry, cells in `columns` order."""
    lines = []
    for m in METHODS:
        by_col = cells.get(m) or {}
        lines.append(" & ".join([tex_label(m)] + [fmt_cell(by_col.get(c)) for c in columns]) + r" \\")
    return lines


def write(out_dir, table, columns, cells, source_table, note):
    """Write `<table>.tex` and `<table>.csv` under `out_dir`. `cells` maps method -> column -> the table row
    (`eval.common.stats.row` columns) the cell is read from, or None; `note` is the fragment's comment."""
    if table not in (WORD_RULE, JUDGED_NET):
        raise ValueError(f"unknown paper table {table!r}")
    unknown = set(cells) - set(METHODS)
    if unknown:
        raise ValueError(f"rows outside the paper's method list: {sorted(unknown)}")
    os.makedirs(out_dir, exist_ok=True)
    head = [f"% {line}" for line in note.strip().splitlines()]
    head.append("% columns: " + " & ".join(columns) + "; rows: " + ", ".join(tex_label(m) for m in METHODS))
    with open(os.path.join(out_dir, table + ".tex"), "w", encoding="utf-8") as h:
        h.write("\n".join(head + tex_rows(columns, cells)) + "\n")
    with open(os.path.join(out_dir, table + ".csv"), "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=CSV_FIELDS)
        w.writeheader()
        for m in METHODS:
            for c in columns:
                r = (cells.get(m) or {}).get(c)
                if r is None:
                    continue
                out = {k: r.get(k, "") for k in CSV_FIELDS}
                out.update(table=table, method=m, label=label(m), column=c, source_table=source_table)
                w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in out.items()})
