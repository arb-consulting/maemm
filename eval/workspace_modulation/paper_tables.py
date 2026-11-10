"""This package's column of the paper's two workspace tables (eval/common/paper_tables.py): Modul., the
topic concepts under focus and mention pooled, at the final period; every cell is a `focus+mention` row of
tables/rates.csv or tables/judged.csv with a concept-bootstrap interval (methodology "Paper tables")."""

from eval.common import paper_tables as P
from eval.common.judges import REFERENCE_JUDGE
from eval.workspace_modulation import analysis as A
from eval.workspace_modulation import config as C

COLUMN = "Modul."
GROUP = "topics"
BAND = C.FINAL_BAND
# method -> (metric, condition, budget) in tables/rates.csv
WORD_RULE = {
    "MAEMM": ("hit_any", "maemm_reg", max(C.PASS_AT[C.HEADLINE_ARM])),
    "NLA": ("hit_any", "nla", max(A.NLA_BUDGETS)),
    "J-lens": (A.LENS_WORD_METRIC, "jlens_L42", C.TOP_WORD),
    "J-lens, 8 layers": (A.LENS_BAND_WORD_METRIC, A.LENS_BAND_COND, C.TOP_WORD),
    "Patchscopes": ("hit_any", C.PATCH_ARM, max(A.PATCH_BUDGETS)),
    "Corpus search": ("hit_any", C.RETRIEVAL_READER, max(A.RETRIEVAL_BUDGETS)),
}
# method -> judged.csv condition of its `net` row
JUDGED_NET = {
    "MAEMM": "maemm_reg8",
    "NLA": "nla_n8",
    "J-lens": C.LENS_READER,
    "J-lens, 8 layers": C.LENS_BAND_READER,
    "Patchscopes": C.PATCH_JUDGED[C.PATCH_ARM],
    "Corpus search": C.RETRIEVAL_JUDGED,
}
NOTE_WORD = (
    "workspace_modulation: word rule, pass@8, topics under focus and mention pooled, final period (J-lens: "
    "a target among its 10 top word-like tokens at layer 42; J-lens, 8 layers: among the pooled top-10 "
    "word-like tokens of layers 36-50, step 2; Patchscopes: at layer 42, its no-patch floor in "
    "tables/rates.csv). Concept-bootstrap 95 % intervals in the .csv."
)
NOTE_JUDGED = (
    "workspace_modulation: judged naming, net = named - foil, pass@8, topics under focus and mention "
    "pooled, final period, judge {judge} (J-lens: the summary of its layer-42 top-10 tokens; J-lens, 8 "
    "layers: the summary of the pooled top-10 tokens of layers 36-50, step 2; Patchscopes: its 8 samples at "
    "layer 42). Concept-bootstrap 95 % intervals in the .csv."
)


def _one(rows, **kw):
    hits = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    if len(hits) > 1:
        raise ValueError(f"paper table: {len(hits)} rows match {kw}")
    return hits[0] if hits else None


def cells(T, judge_name):
    """(word-rule cells, judged-net cells), method -> {COLUMN: the source row or None}."""
    pin = {"group": GROUP, "band": BAND, "instruction": A.POOLED_INSTRUCTION}
    word = {
        m: {COLUMN: _one(T["rates"], metric=metric, condition=cond, budget=budget, **pin)}
        for m, (metric, cond, budget) in WORD_RULE.items()
    }
    judge = C.JUDGES[judge_name].model
    judged = {
        m: {COLUMN: _one(T.get("judged") or [], metric="net", condition=cond, judge=judge, **pin)}
        for m, cond in JUDGED_NET.items()
    }
    return word, judged


def write(T, out_dir, judge_name=None):
    judge_name = judge_name or REFERENCE_JUDGE
    word, judged = cells(T, judge_name)
    P.write(out_dir, P.WORD_RULE, [COLUMN], word, "rates", NOTE_WORD)
    P.write(out_dir, P.JUDGED_NET, [COLUMN], judged, "judged", NOTE_JUDGED.format(judge=C.JUDGES[judge_name].model))
