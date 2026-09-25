"""The steering section: Table `tab:steer`, its in-text numbers and the appendix tables, from an AxBench run
and a BiPO run.

The cells and MAEM's tests are each package's `tables/paper_main_column.csv` (a run without one gets it
from the package's writer, under `out/package_column/`). This module adds the bold rule (the column's best
reader, and every reader it does not beat at Holm p < 0.05) and joins the columns. The best reader's tests
and the NLA's come from the same `paper.marks` over the per-unit verdicts.
"""
import csv
import glob
import json
import os
import shutil
import statistics

from evals.downstream.common.plain_steer import ARM, STRENGTHS, arm_name
from evals.downstream.steering_vector_inversion import paper
from evals.downstream.steering_vector_inversion.artifacts import read_arrays, read_json
from evals.downstream.steering_vector_inversion.config import SINGLE_TEXT
from evals.downstream.steering_vector_inversion.evaluate import case_accuracy

from .common import Numbers, num, pick, read_csv, table, tabular, tex_num, write_text

ALPHA = paper.ALPHA
TEXTS = paper.TEXTS
GROUP = "text"
STEERED = paper.STEERED
#: Column name -> index of its arm in `paper.ROWS`.
KEY = {"axbench": 3, "bipo": 4}
#: The paper's row labels, in `paper.ROWS` order.
TEX_LABELS = ("\\method{}", "NLA", "$J$-lens", "Corpus search (10M tok)", "Steered model",
              "Held-out concept texts", "Untrained base", "Shuffled")
BIPO_KIND, BIPO_HELDOUT_KIND = "bipo:persona", "heldout:persona"
#: The runs of record name the s = 0.5 arm without its strength.
LEGACY_ARMS = {0.5: ARM}


def curve_arm(names, strength):
    """The steered model's arm at `strength` among `names`, under either naming."""
    name = arm_name(strength)
    return name if name in names else LEGACY_ARMS.get(strength, name)


def _slot(condition, budget=TEXTS):
    return ("greedy", 1) if condition in SINGLE_TEXT else ("snippets", budget)


# ------------------------------------------------------------------------------------------ AxBench
def axbench_summary(run, judge):
    """Every identification-rate row under `judge`; the runs of record keep the steered model's rows in
    `identification_plain_steered.csv`."""
    rows = table(run, "identification")
    steered = os.path.join(str(run), "tables", "identification_plain_steered.csv")
    if os.path.exists(steered):
        rows += read_csv(steered)
    return [r for r in rows if r["metric"] == "identification_accuracy" and r["judge"] == judge]


def axbench_verdicts(run):
    """`(cases, concepts)`: the per-case verdicts and the prepared concepts."""
    with open(os.path.join(str(run), "scores", "identification_cases.json"), encoding="utf-8") as handle:
        cases = json.load(handle)["data"]
    with open(os.path.join(str(run), "data", "prepared.json"), encoding="utf-8") as handle:
        concepts = json.load(handle)["data"]["concepts"]
    return cases, concepts


def axbench_units(cases, concepts, judge):
    """`{condition: {concept_id: 0/1}}` at each condition's column slot, over the text concepts the arm has
    text for (the package's `case_accuracy`, as its rates and column read them)."""
    text = {c["concept_id"] for c in concepts if c["genre"] == GROUP}
    out = {}
    for r in cases:
        slot = (r["budget_type"], int(r["budget"])) == _slot(r["condition"])
        if r.get("judge") == judge and r["concept_id"] in text and slot and case_accuracy(r) is not None:
            out.setdefault(r["condition"], {})[r["concept_id"]] = case_accuracy(r)
    return out


# ---------------------------------------------------------------------------------------------- BiPO
def bipo_rates(run, judge):
    """`{arm: row}` on the main list: the learned vectors, and the held-out statements as the held-out row."""
    rates = {}
    for r in table(run, "mc10_summary"):
        if (r["list"], r["row_type"], r["judge"]) != ("main", "arm", judge):
            continue
        if r["kind"] == (BIPO_HELDOUT_KIND if r["arm_a"] == "heldout_matching" else BIPO_KIND):
            rates[r["arm_a"]] = {"estimate": r["value"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"]}
    return rates


def bipo_units(run, judge):
    """`{arm: {vector_id: rate}}` on the main list, learned persona vectors."""
    out = {}
    for r in table(run, "mc10_cells"):
        if (r["list"], r["kind"], r["judge"]) == ("main", BIPO_KIND, judge) and num(r["rate"]) is not None:
            out.setdefault(r["arm"], {})[r["vector_id"]] = float(r["rate"])
    return out


def write_bipo_column(run, scratch, judge):
    """The BiPO package's own column writer, run on a copy of the run's two mc10 tables."""
    from evals.downstream.steering_vector_inversion_bipo import report
    os.makedirs(os.path.join(scratch, "tables"), exist_ok=True)
    for name in ("mc10_summary.csv", "mc10_cells.csv"):
        shutil.copy(os.path.join(str(run), "tables", name), os.path.join(scratch, "tables", name))
    report.paper_column(scratch, judge)


# ------------------------------------------------------------------------------------------ columns
def package_column(run, scratch, write):
    """`(rows, source)`: the run's `paper_main_column.csv`, or, for a run without one, the one `write`
    produces under `scratch`."""
    path = os.path.join(str(run), paper.PAPER_TABLE + ".csv")
    if os.path.exists(path):
        return read_csv(path), "paper_main_column.csv"
    write(scratch)
    return read_csv(os.path.join(scratch, paper.PAPER_TABLE + ".csv")), "paper_main_column.csv (written here)"


def family(units, ref, key):
    """`paper.marks` of `ref` minus each tested row of the column: `{arm: (diff, pairs, p, p_holm, sig)}`."""
    tests = []
    for row in paper.ROWS:
        arm = row[key]
        if arm == ref or row[0] == paper.REFERENCE_SECTION or arm not in units or ref not in units:
            continue
        shared = sorted(set(units[ref]) & set(units[arm]))
        if shared:
            tests.append((arm, [units[ref][u] - units[arm][u] for u in shared]))
    return paper.marks(tests)


def column(rows, units, key, judge_rate):
    """One column: the package's cells and MAEM's tests, the NLA's and the best reader's tests, and the
    bold readers. `judge_rate`, MAEM's rate under the judge asked for, must be the file's."""
    by_arm = {r["arm"]: r for r in rows}
    if abs(num(by_arm["maem"]["estimate"]) - num(judge_rate)) > 1e-9:
        raise ValueError("the package column was written under another judge than the one asked for")
    families = {"maem": {r["arm"]: (num(r["mean_diff"]), int(r["n_pairs"]), num(r["p"]), num(r["p_holm"]),
                                     r["significant"] == "True") for r in rows if r["tested"] == "True"}}
    readers = [r["arm"] for r in rows if r["section"] == "readers" and num(r["estimate"]) is not None]
    best = max(readers, key=lambda a: num(by_arm[a]["estimate"]))
    for ref in ("nla_native", best):
        if ref not in families:
            families[ref] = family(units, ref, key)
    # a reader the best one was not tested against (too few pairs) counts as not beaten
    bold = {best} | {a for a in readers if a != best and (a not in families[best] or families[best][a][3] >= ALPHA)}
    return {"rows": by_arm, "families": families, "best": best, "bold": bold}


def table_tex(cols):
    def cell(col, arm):
        text = tex_num(col["rows"][arm]["estimate"])
        return f"\\textbf{{{text}}}" if arm in col["bold"] else text
    body = [[label] + [cell(cols[name], row[KEY[name]]) for name in KEY]
            for label, row in zip(TEX_LABELS, paper.ROWS)]
    sections = {i: paper.SECTION_TITLES[row[0]] for i, row in enumerate(paper.ROWS)
                if i and row[0] != paper.ROWS[i - 1][0]}
    n = {name: cols[name]["rows"]["maem"]["n"] for name in KEY}
    return ("% Identification from 8 texts (J-lens: its one reading). Bold: the column's best reader and every\n"
            "% reader it does not beat at Holm-corrected p < 0.05 (steering_tests.csv).\n"
            + tabular(["Method", f"AxBench ($n{{=}}{n['axbench']}$)", f"BiPO ($n{{=}}{n['bipo']}$)"], body,
                      sections))


def write_tests(out, cols):
    path = os.path.join(out, "steering_tests.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["column", "reference_arm", "arm", "mean_diff", "n_pairs", "p", "p_holm", "significant"])
        for name, col in cols.items():
            for ref, fam in col["families"].items():
                for arm, (d, n, p, h, sig) in fam.items():
                    w.writerow([name, ref, arm, d, n, p, h, sig])
    return path


# ---------------------------------------------------------------------------------------- lengths
def mean_lengths(run):
    """`{condition: (mean tokens per sampled text, source)}` on the text concepts for MAEM and the NLA:
    from `tables/text_length.csv`, or, for a run without it, counted from the rollout files."""
    path = os.path.join(str(run), "tables", "text_length.csv")
    if os.path.exists(path):
        rows = read_csv(path)
        return {c: (pick(rows, metric="text_length", condition=c, group=GROUP)["estimate"],
                    "text_length.csv") for c in ("maem", "nla_native")}
    with open(os.path.join(str(run), "data", "prepared.json"), encoding="utf-8") as handle:
        text = {c["concept_id"] for c in json.load(handle)["data"]["concepts"] if c["genre"] == GROUP}
    out = {}
    for name, condition, length in (("rollouts", "maem", lambda x: len(x["token_ids"])),
                                    ("nla", "nla_native", lambda x: x["n_tokens"])):
        with open(os.path.join(str(run), "rollouts", name + ".json"), encoding="utf-8") as handle:
            data = json.load(handle)["data"]
        lengths = [length(x) for v in data.values() if v["condition"] == condition and v["concept_id"] in text
                   for x in v["rollouts"] if x["sample_id"] >= 0]
        out[condition] = (sum(lengths) / len(lengths), f"rollouts/{name}.json")
    return out


# -------------------------------------------------------------------------------------- appendix
GENRES = ("text", "code", "math", "all")


def genre_tex(summary):
    """The appendix's AxBench table: every row of `tab:steer` by genre, from 1 and from 8 texts."""
    n = {g: pick(summary, condition="maem", group=g, budget_type="snippets", budget=TEXTS)["n_total"]
         for g in GENRES}
    body = []
    for label, row in zip(TEX_LABELS, paper.ROWS):
        condition, cells = row[KEY["axbench"]], []
        for g in GENRES:
            for b in (1, TEXTS):
                bt, bb = _slot(condition, b)
                hit = [r for r in summary if (r["condition"], r["group"], r["budget_type"], r["budget"])
                       == (condition, g, bt, str(bb))]
                show = hit and (condition not in SINGLE_TEXT or b == 1)
                cells.append(tex_num(hit[0]["estimate"]) if show else "--")
        body.append([label] + cells)
    header = ["Reader"] + [f"{g.capitalize()} ($n{{=}}{n[g]}$), {b}" for g in GENRES for b in (1, TEXTS)]
    return tabular(header, body, {4: "References", 6: "Controls"})


def _degenerate(run_dir, kind=None, genre=None):
    """`{strength: share of the steered model's samples that trip a degeneracy rule}`."""
    totals = {}
    for r in table(run_dir, "plain_steered_health"):
        if (kind and r.get("kind") != kind) or (genre and r.get("genre") != genre):
            continue
        n, share = float(r["n"]), float(r["degenerate_share"])
        k, d = totals.get(float(r["strength"]), (0.0, 0.0))
        totals[float(r["strength"])] = (k + n, d + n * share)
    return {s: d / k for s, (k, d) in totals.items()}


def strength_tex(axbench_run, bipo_run, summary, bipo):
    """The appendix's strength table: identification and degenerate share at each strength of the curve."""
    deg_ax, deg_bi = _degenerate(axbench_run, genre=GROUP), _degenerate(bipo_run, kind=BIPO_KIND)
    ax_arms = {r["condition"] for r in summary}
    body = []
    for s in STRENGTHS:
        arm = curve_arm(ax_arms, s)
        ax1, ax8 = (pick(summary, condition=arm, group=GROUP, budget_type="snippets", budget=b)["estimate"]
                    for b in (1, TEXTS))
        body.append([f"{s:g}", tex_num(ax1), tex_num(ax8), tex_num(bipo[curve_arm(bipo, s)]["estimate"]),
                     f"{deg_ax[s]:.2f} / {deg_bi[s]:.2f}"])
    return tabular(["$s$", "AxBench text, 1 text", "AxBench text, 8 texts", "BiPO, 8 texts",
                    "Degenerate (AxB / BiPO)"], body)


def axbench_vector_cosines(run):
    """`(median, 5th, 95th percentile, n vectors)` of the cosine between every pair of the AxBench text
    concepts' directions, from the run's saved bank."""
    import numpy as np
    path = os.path.join(str(run), "vectors", "bank.json")
    bank = read_arrays(path, read_json(path)["data"])
    text = [i for i, c in enumerate(bank["concepts"]) if c["genre"] == GROUP]
    vectors = np.asarray(bank["directions"], dtype=np.float64)[text]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosines = (vectors @ vectors.T)[np.triu_indices(len(text), 1)]
    lo, median, hi = np.percentile(cosines, [5, 50, 95])
    return float(median), float(lo), float(hi), len(text)


def bipo_appendix_numbers(run, N):
    """Training / held-out pairs per persona vector, and the best corpus window's cosine with a learned
    vector against a random direction (full corpus)."""
    sizes = []
    for path in glob.glob(os.path.join(str(run), "items", "train_persona:*.json")):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        sizes.append((len(doc["train"]), len(doc["heldout"])))
    for i, name in ((0, "train"), (1, "heldout")):
        vals = [s[i] for s in sizes]
        N.add(f"bipo.pairs.{name}.median", statistics.median(vals), min(vals), max(vals),
              "items/train_persona:*.json", note=f"{len(vals)} persona files; interval columns hold min and max")
    with open(os.path.join(str(run), "retrieval", "search.json"), encoding="utf-8") as handle:
        for r in json.load(handle)["rows"]:
            if r["full"] and r["kind"] in (BIPO_KIND, "random"):
                N.add(f"bipo.corpus_top1_cos.{r['kind']}", r["top1_cos"], source="retrieval/search.json",
                      note=f"{r['n_queries']} directions, corpus size {r['size']}")


# ------------------------------------------------------------------------------------------- build
def build(axbench_run, bipo_run, out, judge="sol", lengths=True):
    """Write the table, the tests, the appendix tables and the numbers into `out`; returns the Numbers."""
    summary = axbench_summary(axbench_run, judge)
    cases, concepts = axbench_verdicts(axbench_run)
    bipo = bipo_rates(bipo_run, judge)
    scratch = os.path.join(out, "package_column")
    ax_rows, ax_src = package_column(axbench_run, os.path.join(scratch, "axbench"),
                                     lambda d: paper.axbench_column(d, summary, cases, concepts, judge))
    bi_rows, bi_src = package_column(bipo_run, os.path.join(scratch, "bipo"),
                                     lambda d: write_bipo_column(bipo_run, d, judge))
    maem_ax = pick(summary, condition="maem", group=GROUP, budget_type="snippets", budget=TEXTS)["estimate"]
    cols = {"axbench": column(ax_rows, axbench_units(cases, concepts, judge), KEY["axbench"], maem_ax),
            "bipo": column(bi_rows, bipo_units(bipo_run, judge), KEY["bipo"], bipo["maem"]["estimate"])}
    write_text(out, "steering_table.tex", table_tex(cols))
    write_text(out, "steering_axbench_genre.tex", genre_tex(summary))
    write_text(out, "steering_strength.tex", strength_tex(axbench_run, bipo_run, summary, bipo))
    write_tests(out, cols)

    N = Numbers("steering")
    for name, col, src in (("axbench", cols["axbench"], ax_src), ("bipo", cols["bipo"], bi_src)):
        N.add(f"{name}.n", col["rows"]["maem"]["n"], source=src)
        for arm, r in col["rows"].items():
            N.add_row(f"{name}.{arm}.at8", r, src)
        for ref, fam in col["families"].items():
            for arm, (d, _n, _p, h, _s) in fam.items():
                N.add(f"{name}.{ref}_minus_{arm}.p_holm", h, source="steering_tests.csv", note=f"diff {d:+.3f}")
        N.add(f"{name}.bold", len(col["bold"]), note="bold: " + ", ".join(sorted(col["bold"])))
    arms = {r["condition"] for r in summary}
    for condition in ("maem", "heldout_positive", "retrieval"):
        for b in (1, TEXTS):
            N.add_row(f"axbench.{condition}.texts{b}",
                      pick(summary, condition=condition, group=GROUP, budget_type="snippets", budget=b),
                      "identification.csv")
    for s in STRENGTHS:
        for b in (1, TEXTS):
            N.add_row(f"strength.axbench.s{s:g}.texts{b}",
                      pick(summary, condition=curve_arm(arms, s), group=GROUP, budget_type="snippets", budget=b),
                      "identification.csv")
        N.add_row(f"strength.bipo.s{s:g}", bipo[curve_arm(bipo, s)], "mc10_summary.csv")
    median, lo, hi, n_text = axbench_vector_cosines(axbench_run)
    N.add("axbench.vector_pairwise_cos.text.median", median, lo, hi, "vectors/bank.json",
          note=f"{n_text} text-concept directions, every pair; interval columns hold the 5th and 95th percentiles")
    bipo_appendix_numbers(bipo_run, N)
    if lengths:
        for condition, (mean, source) in mean_lengths(axbench_run).items():
            N.add(f"axbench.mean_tokens.{condition}", mean, source=source, note="sampled texts, text concepts")
    N.write(out)
    return N
