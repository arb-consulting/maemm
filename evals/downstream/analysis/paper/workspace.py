"""The workspace section: Tables `tab:workspace-word` and `tab:workspace-judge` and their in-text numbers.

Joins the fragments each package writes (`evals.downstream.common.paper_tables`): Association and Multi-hop from a
workspace_understanding run, Modulation from a workspace_modulation run, whose `rates.csv` also gives the
focus / mention split. Nothing is recomputed.
"""
from evals.downstream.common.paper_tables import JUDGED_NET, METHODS as KEYS, TEX_LABELS, WORD_RULE

from .common import Numbers, pick, table, tabular, tex_num, write_text

#: (package method key, the label the paper prints), in the table's order.
METHODS = tuple((m, TEX_LABELS.get(m, m)) for m in KEYS)
#: (column the fragment names, header the paper prints, which run holds it).
COLUMNS = (("Assoc.", "Association", "wu"), ("Multi-hop", "Multi-hop", "wu"), ("Modul.", "Modulation", "wm"))
TABLES = {WORD_RULE: "workspace_word_rule.tex", JUDGED_NET: "workspace_judged_net.tex"}
#: The modulation run's condition for the reader the prose splits by instruction.
WM_ARMS = {"MAEM": "maem_reg"}


def cells(runs, name):
    """`{(method, column): row}` from the two runs' fragments; an error on a missing or duplicate cell."""
    out = {}
    for column, _h, which in COLUMNS:
        for r in table(runs[which], name):
            if r["column"] == column:
                if (r["method"], column) in out:
                    raise ValueError(f"{name}: duplicate cell {(r['method'], column)}")
                out[(r["method"], column)] = r
    missing = [(m, c) for m, _l in METHODS for c, _h, _w in COLUMNS if (m, c) not in out]
    if missing:
        raise ValueError(f"{name}: missing cells {missing}")
    return out


def build(wu_run, wm_run, out):
    runs = {"wu": wu_run, "wm": wm_run}
    N = Numbers("workspace")
    for name, filename in TABLES.items():
        got = cells(runs, name)
        rows = [[label] + [tex_num(got[(m, c)]["estimate"]) for c, _h, _w in COLUMNS] for m, label in METHODS]
        write_text(out, filename, tabular(["Method"] + [h for _c, h, _w in COLUMNS], rows))
        for m, _label in METHODS:
            for c, _h, _w in COLUMNS:
                N.add_row(f"{name.replace('paper_workspace_', '')}.{m}.{c}", got[(m, c)], name + ".csv",
                          note=got[(m, c)].get("ci_method", ""))
        for c, _h, _w in COLUMNS:
            order = sorted((m for m, _l in METHODS), key=lambda m: -float(got[(m, c)]["estimate"]))
            N.add(f"{name}.order.{c}", None, note=" > ".join(order))
        n = {c: got[("MAEM", c)]["n_valid"] for c, _h, _w in COLUMNS}
        N.add(f"{name}.n", None, note=" / ".join(n[c] for c, _h, _w in COLUMNS))
    rates = table(wm_run, "rates")
    for method, arm in WM_ARMS.items():
        for instruction in ("focus", "mention"):
            N.add_row(f"modulation.{method}.{instruction}",
                      pick(rates, metric="hit_any", condition=arm, group="topics", budget_type="samples",
                           budget=8, instruction=instruction, band="final"), "rates.csv")
    N.write(out)
    return N
