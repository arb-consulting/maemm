"""The run's report, rendered from the run directory's own CSVs: deterministic and offline.

Nothing here asks a judge or recomputes a number; each section reads the tables the stage before it wrote,
and a missing table raises rather than printing "not run".
"""
import csv
import json
import os
import statistics

from eval.common import retrieval as R
from eval.common.judge_client import served_line, unasked, unasked_detail
from eval.common.judges import REFERENCE_JUDGE
from eval.common.runs import RunDir
from eval.steering_vector_inversion import paper

from . import config as C
from . import judge, mc10, nla_arm, plain_steer_arm, retrieval_arm

REPORT = "report.md"
COSTS_CSV = "tables/coverage_and_costs.csv"
GATE = "scores/gate.json"
COST_COLUMNS = ("judge", "instrument", "n_records", "ok", "input_tokens", "output_tokens", "cost_usd")


def _require(run, rel, what):
    if not run.exists(rel):
        raise FileNotFoundError(f"{rel} is missing from the run directory; {what}")
    return run.read_json(rel)


def costs_table(root):
    """`tables/coverage_and_costs.csv`: what each judge was asked per instrument, and what it cost, read off
    the request logs. A log whose LAST record for some request was never asked (cap, transport, key) is
    refused: every rate would count it as a miss. Run that stage again first."""
    rows = []
    base = os.path.join(str(root), "judges")
    for judge_name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        folder = os.path.join(base, judge_name)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".jsonl"):
                continue
            records = [json.loads(line) for line in
                       open(os.path.join(folder, name), encoding="utf-8") if line.strip()]
            last = {}
            for r in records:
                last[judge.cell_id(r.get("key"), r.get("meta"))] = r
            counts = unasked(last.values())
            if sum(counts.values()):
                raise ValueError(f"judges/{judge_name}/{name} holds {sum(counts.values())} requests "
                                 f"nobody was ever asked ({unasked_detail(counts)}); run that stage "
                                 "again (raising the cap if the cap is what stopped it, under a key the "
                                 "endpoint accepts if the key is) before reporting")
            usage = [r.get("usage") or {} for r in records]
            rows.append({"judge": judge_name, "instrument": name[:-len(".jsonl")],
                         "n_records": len(records),
                         "ok": sum(r.get("status") == "ok" for r in records),
                         "input_tokens": sum(int(u.get("input_tokens") or 0) for u in usage),
                         "output_tokens": sum(int(u.get("output_tokens") or 0) for u in usage),
                         "cost_usd": sum(float(r.get("cost_usd") or 0.0) for r in records)})
    if not rows:
        raise FileNotFoundError("judges/ holds no request log; run the judged stages")
    mc10._write_csv(root, COSTS_CSV, COST_COLUMNS, rows)
    return rows


LABELS = ("judge", "instrument")


def _md_csv(root, rel, headers, columns, digits=3):
    """A saved CSV as a markdown table: the label columns verbatim, every other column formatted."""
    with open(os.path.join(str(root), rel), encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    body = [[(r.get(c) or "") if c in LABELS else mc10._fmt(r.get(c), digits) for c in columns]
            for r in rows]
    return mc10._md(headers, body)


def _plain_section(root, families):
    """The steered model's text health per strength: `plain_steer_arm.HEALTH_CSV`, summarised over vectors."""
    from eval.common import plain_steer as P
    lines = ["## The steered model's text", "",
             "The plain base with the vector added, generating from the one-token prompt `<|im_end|>` with no "
             f"template: `s x {P.STEER_UNIT:.2f} x unit(v)` at the output of block 42 on every position, "
             f"{P.NEW_TOKENS} new tokens at temperature 1, {P.MAIN_SAMPLES} samples at every strength. The "
             f"paper's table reads s = {P.TABLE_STRENGTH:g}. Each row is a mean over vectors; the "
             "log-likelihood is the unsteered base's per token (median over vectors of each vector's median). "
             "Diagnostics only: no sample is selected or dropped by them.", ""]
    rows = plain_steer_arm.health_table(root, families)
    mean = lambda values: sum(values) / len(values) if values else float("nan")
    body = []
    for arm in plain_steer_arm.ARMS:
        here = [r for r in rows if r["arm"] == arm]
        if not here:
            continue
        logliks = [r["loglik_median"] for r in here if r["loglik_median"] == r["loglik_median"]]
        body.append((here[0]["strength"], len(here), mc10._fmt(mean([r["degenerate_share"] for r in here])),
                     mc10._fmt(statistics.median(logliks) if logliks else float("nan")),
                     mc10._fmt(mean([r["eos_early_share"] for r in here])),
                     mc10._fmt(mean([r["distinct3"] for r in here if r["distinct3"] == r["distinct3"]])),
                     mc10._fmt(mean([r["non_ascii_share"] for r in here])),
                     mc10._fmt(mean([r["identical_share"] for r in here]))))
    return lines + mc10._md(["Strength", "n vectors", "Degenerate share", "Log-lik", "Stopped early",
                             "Distinct 3-grams", "Non-ASCII share", "Identical share"],
                            body) + ["", f"[Per vector and strength]({plain_steer_arm.HEALTH_CSV})", ""]


def _search_section(root):
    """The corpus search's own cosine per vector kind, at the shared size and the full corpus."""
    lines = ["## The corpus search's own cosine", ""]
    run = RunDir(root)
    if not run.exists(retrieval_arm.SEARCH):
        return lines + [f"Unavailable: `{retrieval_arm.SEARCH}` is not in this directory (the "
                        "`retrieval-merge` stage writes it).", ""]
    rows = [{**row, "kind": mc10.KIND_LABELS.get(row["kind"], row["kind"])}
            for row in run.read_json(retrieval_arm.SEARCH)["rows"]]
    return lines + R.search_table(rows, lead=("Vector kind", "kind")) + [
        "", R.SEARCH_NOTE + " The identification tables read the full corpus's windows; the shared-size "
        "rows are the search's cosine alone.", "", f"[Rows and the shared-size windows]({retrieval_arm.SEARCH})",
        ""]


def _costs_section(root):
    lines = ["## Coverage and cost", "",
             "Every request this run sent, per judge and instrument, with the tokens and the spend behind "
             "it. `ok` is the requests that came back with an answer this package could parse; the rest "
             "are broken out by reason in each instrument's missingness table.", ""]
    return lines + _md_csv(root, COSTS_CSV,
                           ["Judge", "Instrument", "n requests", "ok", "Input tokens", "Output tokens",
                            "US$"],
                           ["judge", "instrument", "n_records", "ok", "input_tokens", "output_tokens",
                            "cost_usd"], digits=4) + ["", f"[Full table]({COSTS_CSV})", ""]


#: The paper's BiPO column: the learned persona vectors on the main list, their held-out statements as the
#: held-out row.
PAPER_KIND, PAPER_HELDOUT_KIND = "bipo:persona", "heldout:persona"
PAPER_MIN_PAIRS = mc10.MIN_VECTORS_PAIRED


def paper_column(root, judge_name=None):
    """The BiPO column of the main steering table (`eval.steering_vector_inversion.paper`) from the saved
    `mc10_summary.csv` and `mc10_cells.csv`: each row's mean per-vector rate with its interval, and MAEMM
    minus each tested row paired over the learned vectors (tested at `PAPER_MIN_PAIRS` or more pairs)."""
    judge_name = judge_name or REFERENCE_JUDGE
    summary = mc10._read_csv(root, mc10.SUMMARY_CSV) or []
    cells = mc10._read_csv(root, mc10.CELLS_CSV) or []
    rates = {}
    for r in summary:
        if (r["list"], r["row_type"], r["judge"]) != (mc10.MAIN, "arm", judge_name):
            continue
        kind = PAPER_HELDOUT_KIND if r["arm_a"] == "heldout_matching" else PAPER_KIND
        if r["kind"] == kind:
            rates[r["arm_a"]] = (r["value"], r["ci_lower"], r["ci_upper"], r["n_vectors"])
    per_vector = {(r["arm"], r["vector_id"]): float(r["rate"]) for r in cells
                  if (r["list"], r["kind"], r["judge"]) == (mc10.MAIN, PAPER_KIND, judge_name)
                  and not mc10._blank(r.get("rate"))}
    vectors = sorted(v for arm, v in per_vector if arm == "maemm")
    tests = []
    for section, _tex, _label, _condition, arm in paper.ROWS:
        if arm == "maemm" or section == paper.REFERENCE_SECTION:
            continue
        diffs = [per_vector[("maemm", v)] - per_vector[(arm, v)] for v in vectors if (arm, v) in per_vector]
        if len(diffs) >= PAPER_MIN_PAIRS:
            tests.append((arm, diffs))
    rows = paper.column_rows(4, rates, tests)
    paper.write_column(root, rows, f"BiPO & learned vectors ($n{{=}}{len(vectors)}$)")
    return rows


# Every table the report reads; `render` refuses to finish without one of them.
TABLES = (mc10.SUMMARY_CSV, mc10.PAIRED_CSV, mc10.CELLS_CSV, mc10.ARMS_CSV, mc10.MISSINGNESS_CSV,
          mc10.SHUFFLED_CSV, plain_steer_arm.HEALTH_CSV, COSTS_CSV)


def render(root):
    """Write `report.md` (and the cost table and paper column it builds); the path written."""
    run = RunDir(root)
    families = _require(run, mc10.FAMILIES, "run the rollouts stage")
    plain = _plain_section(root, families)
    costs_table(root)
    gate = _require(run, GATE, "run the gate stage")
    missing = [rel for rel in TABLES if not os.path.isfile(os.path.join(str(root), rel))]
    if missing:
        raise FileNotFoundError(f"the report's tables are incomplete: {missing}")
    paper_column(root)
    parts = ["# Persona steering vectors through MAEMM", "",
             "Learned BiPO persona vectors at the read layer, each given to every reader as a unit direction "
             "and identified from the reader's text alone.", "",
             f"Judge profile: `{C.JUDGE_PROFILE}` ("
             + ", ".join(f"`{name}` = {spec.label}" for name, spec in C.JUDGES.items()) + ").", "",
             f"A{served_line(root, C.JUDGES)[1:]}.", "",
             mc10.section(root), "", *_search_section(root), "", *nla_arm.section(families), "", *plain, "",
             *_costs_section(root), "",
             "## Paper table", "",
             "The BiPO column of the main steering table -- identification from bundles of "
             f"{paper.TEXTS} texts with intervals over vectors, MAEMM against each reader and control by a "
             "paired sign-flip permutation test with Holm's correction, the references untested and the "
             f"steered model at s = {paper.TABLE_STRENGTH:g} (`{paper.STEERED}`) -- is "
             f"[{paper.PAPER_TABLE}.tex]({paper.PAPER_TABLE}.tex), with its p-values in "
             f"[{paper.PAPER_TABLE}.csv]({paper.PAPER_TABLE}.csv).", "",
             "## Instrument check", "", "```", json.dumps(gate, indent=2, sort_keys=True), "```", ""]
    path = os.path.join(str(root), REPORT)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts).rstrip("\n") + "\n")
    return path
