"""Every constant, seed and prompt of the workspace-understanding eval (methodology.md §2-§6, §9)."""

from evals.downstream.common.judges import REFERENCE_JUDGE, active, judges, rates
from evals.downstream.common.retrieval import SHARED_SIZE, CorpusSpec, load_index, search_docs
from maem.config import D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF  # 5120, 1, 42, 1.0

# The base model, the inverter and the lens are pinned once for every package (evals/downstream/common/pins.py).
from evals.downstream.common.pins import (  # noqa: F401
    INVERTER,
    INVERTER_REVISION,
    JLENS_COMMIT,
    LENS_BYTES,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    LENS_SHA256,
    MODEL,
    MODEL_REVISION,
)

# In every stage's key: bump it to invalidate every saved output.
SCORING_VERSION = "3"
N_LAYERS = 64

DATASET_FILES = {"association": "lens-eval-association.json", "multihop": "lens-eval-multihop.json"}
DATASET_SHA256 = {
    "association": "d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c",
    "multihop": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
}
EXPECTED_COUNTS = {"association": 102, "multihop": 93}
FAMILIES = ("association", "multihop")

# --- the lens (methodology §3.2) ---
TOP_WORD = 10
from evals.downstream.common.lens_io import BAND8_LAYERS as LENS_BAND_LAYERS  # noqa: E402  (36, 38, ..., 50)
LENS_BAND = "band8"
RANK_KS = (1, 5, 10, 50)

# --- generation ---
GEN_SEED = 1234
# Added to --seed per generating arm; the offsets are distinct per arm, so no two arms share a sampler stream.
ARM_SEED_OFFSET = {
    "maem": 0,
    "nla": 3,
    "nla_mid": 4,
    "nla_mean": 5,
    "maem_mid": 6,
    "maem_mean": 7,
    "untrained_base": 8,
    "patchfloor": 12,
    "patch42": 13,
}
N_SAMPLES = 8
PASS_AT_N = (1, 2, 4, 8)
TEMP, TOP_P, TOP_K, MIN_P = 1.0, 1.0, 0, 0.0
MAX_NEW, MIN_NEW = 64, 16
GEN_CHUNK = 32  # rows per generate() call (model_io.MAX_GEN_ROWS)
DIAG_MAX_NEW = 8

DONOR_SEED = 5200
N_DONORS = 20
BOOTSTRAP_SEED = 0
N_BOOT = 10000
CHANCE_FLAG_RATIO = 3.0

# A readout position whose norm exceeds this multiple of its prompt's median token norm is flagged.
NORM_MULT = 10.0

# --- the Patchscopes reader (methodology §3.3); prompt, patch rules and check are evals/downstream/common/patchscope.py's ---
from evals.downstream.common.patchscope import (  # noqa: E402,F401
    PATCH_ALPHA,
    PATCH_CHECK_MIN_COS,
    PATCH_CHECK_MIN_REL_DELTA,
    PATCH_CHECK_ROWS,
    PATCH_FLOOR,
    PATCH_GEN_ROWS,
    PATCH_INPUT,
    PATCH_PLACEHOLDER,
    PATCH_PROMPT,
    PATCH_PROMPT_ID,
    PATCH_RULE_TUNED,
    PATCH_SEEDING,
)

# `patch42`: the tuned rule (the centred direction at twice the placeholder's norm) at the read layer.
PATCH_LAYERS = {"patch42": READ_LAYER}
PATCH_RULES = {arm: PATCH_RULE_TUNED for arm in PATCH_LAYERS}
PATCH_ARMS = tuple(PATCH_LAYERS) + (PATCH_FLOOR,)

# --- the judge (evals/downstream/common/judges.py) ---
from evals.downstream.common.judge_client import OPENROUTER_URL  # noqa: E402,F401
# One ledger over the judge and the summariser. A full run spends about US$8 at list price; every reply
# at its cap would be about US$13.
JUDGE_BUDGET_USD = 20.0
JUDGE_CONCURRENCY = 16
# Per-kind reply caps; a cap is part of `request_key`.
JUDGES = judges({"samples": 200, "summary": 200, "summary_req": 300})
JUDGE_PROFILE = active()
# The model that writes every lens summary (methodology §5.3): the reference judge's.
SUMMARISER = REFERENCE_JUDGE
RATES_PER_M = rates(JUDGES)

N_DIAG = 20
DIAG_FILLERS = ["the", "house", "river", "blue", "seven", "music", "garden", "winter", "table"]

# --- the judged conditions (methodology §6.2): (reader, budget), each judged against own and foil targets ---
JUDGED_CONDITIONS = (
    "maem_n8",
    "patch42_n8",
    "jlens_L42_summary",
    "jlens_band8_summary",
    "nla_n8",
    "retrieval_n8",
)
# The free-text readers of the word rule, each with a 20-donor target-shuffle chance line (methodology §6.1).
FREE_TEXT_CONDITIONS = (
    "maem",
    "patch42",
    "patchfloor",
    "nla",
    "nla64",
    "retrieval",
    "nla_mid",
    "nla_mean",
    "maem_mid",
    "maem_mean",
    "untrained_base",
)

from evals.downstream.common.lens_summary import SUMMARY_SYSTEM, SUMMARY_USER  # noqa: E402,F401
from evals.downstream.common.naming import NAMING_SYSTEM as JUDGE_SYSTEM, READOUT_LABEL  # noqa: E402,F401

# Written-out numbers, so a readout that says "eight" hits the target 8.
_units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_tens = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty",
         90: "ninety"}
NUMBER_WORDS = {str(n): w for n, w in enumerate(_units)}
for _t, _w in _tens.items():
    NUMBER_WORDS[str(_t)] = _w
    for _u in range(1, 10):
        NUMBER_WORDS[str(_t + _u)] = f"{_w}-{_units[_u]}"
NUMBER_WORDS["100"] = "one hundred"

# --- the NLA verbalizer reader (methodology §3.4): evals/downstream/common/nla/nla_reader.PINS, used as they are ---
from evals.downstream.common.nla import nla_reader  # noqa: E402
assert nla_reader.PINS.trunc == MAX_NEW  # the `nla64` row is each sample's first MAX_NEW generated ids

# --- the position controls (methodology §3.5): `mid` = h_42 at token ⌊p/2⌋, `mean` = mean over 1..p ---
POSITION_CONTROLS = ("mid", "mean")

# --- the corpus-search reader (methodology §3.6): the suite's shared corpus, unchanged ---
CORPUS = CorpusSpec()
# Windows are ranked by cos(raw h_t, unit(h_42 - mu)), the re-read cosine's geometry.
CORPUS_METRIC = "raw"
# The nested prefix size the search cosine is also reported at (evals/downstream/common/retrieval.py).
CORPUS_SHARED_SIZE = SHARED_SIZE
# --smoke searches this many documents of the same corpus.
CORPUS_SMOKE_DOCS = 2


def corpus_docs(smoke, index=None):
    """The held-out documents the search corpus is built from; the first CORPUS_SMOKE_DOCS under --smoke."""
    docs = search_docs(index or load_index(), CORPUS)
    return docs[:CORPUS_SMOKE_DOCS] if smoke else docs


# --- the re-read (methodology §6.4) ---
REREAD_CONDITIONS = ("nla", "nla64") + PATCH_ARMS
REREAD_SELFCHECK_TOL = 0.05  # max |re-read - saved| on MAEM's own greedies
REREAD_WINDOW_TOKENS = 95
REREAD_WINDOW = {"nla": nla_reader.PINS.score_max_length}  # the verbalizer's native text is re-read whole
REREAD_NORM_FILTER = False  # unfiltered; what the norm filter would drop is counted instead
REREAD_NORM_FILTER_MULT = 10.0
