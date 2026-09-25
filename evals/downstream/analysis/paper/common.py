"""Shared helpers for the paper builders: reading a run's tables, formatting cells, and the numbers file.

Every builder writes its LaTeX fragments and figures into `--out`, plus `numbers.csv` / `numbers.md`: one
row per number the paper quotes, with its interval and the table it was read from.
"""
import csv
import math
import os

NUMBER_FIELDS = ("section", "key", "value", "ci_lower", "ci_upper", "source", "note")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table(run_dir, name):
    """`<run_dir>/tables/<name>.csv` as a list of dicts."""
    return read_csv(os.path.join(str(run_dir), "tables", name + ".csv"))


def num(value):
    """A float, or None for a blank or NaN cell."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def pick(rows, **match):
    """The one row whose columns equal `match` (compared as strings); an error if none or several."""
    hits = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in match.items())]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} rows match {match}")
    return hits[0]


def tex_num(value, digits=3):
    """`0.284`, a negative as `$-$0.043`, a blank as `--`."""
    value = num(value)
    if value is None:
        return "--"
    text = f"{value:.{digits}f}"
    if text.startswith("-"):
        text = text[1:]
        if float(text) != 0:
            text = "$-$" + text
    return text


class Numbers:
    """The in-text numbers of one paper section, written as `numbers.csv` and `numbers.md`."""

    def __init__(self, section):
        self.section, self.rows = section, []

    def add(self, key, value, lo=None, hi=None, source="", note=""):
        self.rows.append({"section": self.section, "key": key, "value": num(value), "ci_lower": num(lo),
                          "ci_upper": num(hi), "source": source, "note": note})
        return num(value)

    def add_row(self, key, row, source, note="", value="estimate", lo="ci_lower", hi="ci_upper"):
        """A number read off a table row with the usual estimate / interval columns."""
        return self.add(key, row[value], row.get(lo), row.get(hi), source, note)

    def write(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "numbers.csv"), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NUMBER_FIELDS)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: "" if r[k] is None else r[k] for k in NUMBER_FIELDS})
        lines = [f"# In-text numbers: {self.section}", "", "| key | value | 95 % CI | source | note |",
                 "|---|---|---|---|---|"]
        for r in self.rows:
            ci = "" if r["ci_lower"] is None else f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
            value = "" if r["value"] is None else f"{r['value']:.4f}"
            lines.append(f"| {r['key']} | {value} | {ci} | {r['source']} | {r['note']} |")
        with open(os.path.join(out_dir, "numbers.md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def tabular(header, rows, sections=()):
    """A booktabs tabular: `header` cells, then `rows` of cells; `sections` maps a row index to the
    italic title of a block that starts there under a rule."""
    cols = "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "  \\toprule", "  " + " & ".join(header) + " \\\\", "  \\midrule"]
    titles = dict(sections)
    for i, cells in enumerate(rows):
        if i in titles:
            lines += ["  \\midrule", f"  \\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{titles[i]}}}}} \\\\"]
        lines.append("  " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["  \\bottomrule", "\\end{tabular}"]) + "\n"


def write_text(out_dir, name, text):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
