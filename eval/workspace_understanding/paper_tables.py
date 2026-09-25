"""The Assoc. and Multi-hop columns of the paper's two workspace tables (eval/common/paper_tables.py), each
cell a row of tables/rates.csv (word rule, pass@8) or tables/judged.csv (net = named − foil, the profile's
judge). The J-lens rows use the `word_top10` rules and the summaries of methodology §3.2 and §6.1."""

from eval.common import paper_tables as P
from eval.workspace_understanding import config as C

COLUMNS = {"association": "Assoc.", "multihop": "Multi-hop"}
# method -> (metric, condition, budget) in tables/rates.csv
WORD_RULE = {
    "MAEMM": ("pass_at_n", "maemm", 8),
    "NLA": ("pass_at_n", "nla", 8),
    "J-lens": ("word_top10", "jlens_L42", C.TOP_WORD),
    "J-lens, 8 layers": ("word_top10", "jlens_" + C.LENS_BAND, C.TOP_WORD),
    "Patchscopes": ("pass_at_n", "patch42", 8),
    "Corpus search": ("pass_at_n", "retrieval", 8),
}
# method -> judged.csv condition of its `net` row
JUDGED_NET = {
    "MAEMM": "maemm_n8",
    "NLA": "nla_n8",
    "J-lens": "jlens_L42_summary",
    "J-lens, 8 layers": "jlens_band8_summary",
    "Patchscopes": "patch42_n8",
    "Corpus search": "retrieval_n8",
}
NOTE_WORD = (
    "workspace_understanding: word rule, pass@8 (J-lens: a target among its 10 top word-like tokens at layer "
    "42; J-lens, 8 layers: among the pooled top-10 word-like tokens of layers 36-50, step 2). Wilson 95 % "
    "intervals in the .csv."
)
NOTE_JUDGED = (
    "workspace_understanding: judged naming, net = named - foil, pass@8, judge {judge} (J-lens: the summary "
    "of its layer-42 top-10 tokens; J-lens, 8 layers: the summary of the pooled top-10 tokens of layers "
    "36-50, step 2). Item-bootstrap 95 % intervals in the .csv."
)


def _one(rows, **kw):
    hits = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    if len(hits) > 1:
        raise ValueError(f"paper table: {len(hits)} rows match {kw}")
    return hits[0] if hits else None


def cells(T, judge):
    """(word-rule cells, judged-net cells), method -> column label -> the source row or None."""
    word, judged = {}, {}
    for m, (metric, cond, budget) in WORD_RULE.items():
        word[m] = {
            label: _one(T["rates"], metric=metric, condition=cond, group=g, budget=budget)
            for g, label in COLUMNS.items()
        }
    for m, cond in JUDGED_NET.items():
        judged[m] = {
            label: _one(T["judged"], metric="net", condition=cond, group=g, judge=judge)
            for g, label in COLUMNS.items()
        }
    return word, judged


def write(T, out_dir, judge=None):
    judge = judge or C.REFERENCE_JUDGE
    word, judged = cells(T, judge)
    cols = list(COLUMNS.values())
    P.write(out_dir, P.WORD_RULE, cols, word, "rates", NOTE_WORD)
    P.write(out_dir, P.JUDGED_NET, cols, judged, "judged", NOTE_JUDGED.format(judge=judge))
