"""Every constant, pin, seed and prompt digest of the workspace-modulation eval (methodology.md)."""

import hashlib, os

from evals.downstream.common.judges import REFERENCE_JUDGE, active, judges, rates
from evals.downstream.common.lens_io import BAND8_LAYERS as LENS_BAND_LAYERS
from evals.downstream.common.lens_summary import SUMMARY_SYSTEM, SUMMARY_USER
from evals.downstream.common.naming import NAMING_SYSTEM
from evals.downstream.common.nla.nla_reader import PINS as _nla
from evals.downstream.common.pins import (  # noqa: F401  (shared by every package)
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
from evals.downstream.common.retrieval import SHARED_SIZE, CorpusSpec
from maem.config import D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF  # noqa: F401  5120, 1, 42, 1.0

HERE = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(HERE, "datasets")

SCORING_VERSION = "5"
PARSE_VERSION = "4"
N_LAYERS = 64
NORM_MULT = 10.0  # a cell whose layer-42 norm exceeds this x the item's median is flagged (never filtered)

# --- materials (datasets/SOURCES.md) ---
DM_FILE = "directed-modulation.json"
DM_SHA256 = "f3478bb7f3b9e19423f0c056a289fdaf1759942daf67e5304e4f55d9ad555b92"

# --- design ---
CARRIER_SEED, PHRASING_SEED, DONOR_SEED = 5300, 5301, 5200
GEN_SEED = 1234
# An arm's seed is GEN_SEED + seed_offset * SHARD_SEEDS, and shard k draws at that + k.
SHARD_SEEDS = 1000
BOOTSTRAP_SEED = 0
N_BOOT = 10000
N_DONORS = 20
CONDITIONS = ("focus", "ignore", "dont_think", "mention", "baseline")
GROUP_TO_CONDITION = {"focus": "focus", "dismissal": "ignore", "negated-think": "dont_think", "mention": "mention"}
FRAME = 'Write "{carrier}" {phrasing} Don\'t write anything else.'
BASELINE_FRAME = 'Write "{carrier}" Don\'t write anything else.'
# The two families are never pooled: they differ in what a readout has to contain.
FAMILIES = ("topics", "arithmetic")
GROUPS = FAMILIES
SMOKE_CONCEPTS_PER_FAMILY = 2
MIN_CARRIER_TOKENS = 4  # a shorter carrier is reassigned

# --- read positions ---
FINAL_BAND = "final"  # the carrier's final period: the headline cell
MEAN_BAND = "mean"  # one synthetic cell per item, the mean residual over the carrier tokens
MEAN_POS = -1  # the synthetic cell's position id (no token position is negative)
READ_BANDS = (FINAL_BAND, MEAN_BAND)
CARRIER_BAND = "carrier"  # the J-lens paper's any-token protocol; lens rows only

# --- corpus search ---
CORPUS_SPEC = CorpusSpec()
CORPUS = CORPUS_SPEC.record()
RETRIEVAL_READER = "retrieval"  # word-rule condition
RETRIEVAL_JUDGED = "retrieval_top8"  # judged condition: the top windows joined
TOP_WINDOWS = CORPUS["top_k"]
CORPUS_SHARED_SIZE = SHARED_SIZE
RETRIEVAL_SHARDS = 8
SMOKE_CORPUS_DOCS = 2

# --- generation ---
GEN_CHUNK = 32  # rows per generate() call (model_io.MAX_GEN_ROWS)
ARMS = {
    "reg": dict(
        n_samples=8, greedy=True, temp=1.0, top_p=1.0, top_k=0, min_p=0.0, max_new=64, min_new=16, seed_offset=0
    ),
}
HEADLINE_ARM = "reg"
PASS_AT = {"reg": (1, 2, 4, 8)}
COMPLIANCE_MAX_NEW = 48
TOP_WORD, CHANCE_FLAG_RATIO = 10, 3.0
LENS_BAND = "band8"  # the J-lens top-10 lists of LENS_BAND_LAYERS pooled, final period only
LENS_SELF_CHECK_LOGIT_TOL = 5e-2
LENS_SELF_CHECK_TOP_OVERLAP = 8

# Injection check (methodology "Guards"): the headline arm's greedies must differ from every greedy of the
# null control; where the control is not in sight, their pooled distinct share must clear a floor. The
# re-read gap is recorded and gates nothing.
INJECTION_MIN_DIFFERS_FROM_CONTROL = 0.9
INJECTION_MIN_DISTINCT = 0.25
INJECTION_CHECK = {
    "alive": "greedies_differ_from_control",
    "min_differs_from_control": INJECTION_MIN_DIFFERS_FROM_CONTROL,
    "min_distinct": INJECTION_MIN_DISTINCT,
}
NLA_MIN_DISTINCT = 0.95  # share of an item's cells whose NLA greedy must differ

# --- judges (evals/downstream/common/judges.py) ---
from evals.downstream.common.judge_client import OPENROUTER_URL  # noqa: E402,F401
MAX_TOKENS = {"samples": 200, "summary": 200, "summary_req": 300}
JUDGES = judges(MAX_TOKENS)
RATES_PER_M = rates(JUDGES)
JUDGE_PROFILE = active()
SUMMARISER = REFERENCE_JUDGE  # one model writes every lens summary; it sees only the tokens
JUDGE_LEDGER_REL = "judges/ledger.json"
# One ledger over the summariser and the judge. Cap ≈ worst case × 1.25; see README "Costs".
JUDGE_BUDGET_USD, JUDGE_CONCURRENCY = 30.0, 16
# The gate on the head of each judge's naming requests (judge.gate_report).
GATE_N = 20
GATE_MIN_PARSE = 0.80
GATE_MIN_QUOTE = 0.60
GATE_MIN_POSITIVES = 5

# --- the naming instrument (evals/downstream/common/naming.py) ---
LENS_READER = "jlens_L42_summary"
LENS_BAND_READER = "jlens_band8_summary"
LENS_READERS = (LENS_READER, LENS_BAND_READER)
JUDGED_READERS = ("maem_reg8", "maem_null8", "nla_n8", RETRIEVAL_JUDGED, "patch42_n8", LENS_READER,
                  LENS_BAND_READER)
READOUT_KIND = {r: ("summary" if r in LENS_READERS else "samples") for r in JUDGED_READERS}
READER_BANDS = {LENS_BAND_READER: (FINAL_BAND,)}  # read positions of a reader not read at both


def judged_readers_at(band):
    """The judged readers read at `band`, in JUDGED_READERS order."""
    return tuple(r for r in JUDGED_READERS if band in READER_BANDS.get(r, READ_BANDS))


VS = ("own", "foil")
FOIL_RULE = "first_donor_whose_forms_are_not_operands"
JUDGED_CONTRAST_METRICS = ("named", "net")
PROMPT_SHA256 = {
    k: hashlib.sha256(v.encode()).hexdigest()
    for k, v in dict(naming_system=NAMING_SYSTEM, summariser_system=SUMMARY_SYSTEM,
                     summariser_user=SUMMARY_USER).items()
}

# Written-out forms of the arithmetic answers, so "eight" counts as a hit on 8.
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty",
         90: "ninety"}
NUMBER_WORDS = {str(n): w for n, w in enumerate(_ONES)}
for _t, _w in _TENS.items():
    NUMBER_WORDS[str(_t)] = _w
    for _u in range(1, 10):
        NUMBER_WORDS[str(_t + _u)] = f"{_w}-{_ONES[_u]}"
NUMBER_WORDS["100"] = "one hundred"

COLOURS = {
    "maem": "#0072B2",
    "jlens": "#E69F00",
    "nla": "#CC79A7",
    "retrieval": "#009E73",
    "control": "#777777",
    "patchscope": "#D55E00",
}

# --- NLA verbalizer: evals/downstream/common/nla/nla_reader.PINS, recorded in config.json as NLA_PINS ---
NLA_PINS = {k: (dict(v) if k == "sidecar_sha256" else v) for k, v in vars(_nla).items()}
REREAD_MAX_LENGTH = 95  # the shared scorer's window
# The re-read scores every content token; the scorer's norm filter is off and only counted.
REREAD_NORM_FILTER = False
REREAD_NORM_FILTER_MULT = 10.0
ARMS["nla"] = dict(
    n_samples=_nla.n_samples, greedy=True, temp=_nla.temp, top_p=_nla.top_p, top_k=_nla.top_k, min_p=_nla.min_p,
    max_new=_nla.max_new, min_new=_nla.min_new, seed_offset=7,
)
NLA_READERS = ("nla", "nla64")  # native, and the same samples cut to MAEM's 64 tokens

# --- the null control: MAEM's headline arm injected with a zero direction ---
NULL_ARM = "null"
ARMS[NULL_ARM] = dict(ARMS["reg"], seed_offset=20, null_direction=True)
PASS_AT[NULL_ARM] = PASS_AT["reg"]

# --- the untrained-base ablation: the headline arm generated by the base model ---
BASE_ARM = "base"
ARMS[BASE_ARM] = dict(ARMS["reg"], seed_offset=30, untrained_base=True)
PASS_AT[BASE_ARM] = PASS_AT["reg"]
ARM_ORDER = (HEADLINE_ARM, NULL_ARM, BASE_ARM)
MEAN_ARMS = (HEADLINE_ARM,)  # arms generated at the mean cell; the control reuses its final-period readout
MEAN_CONTROL_FROM_BAND = FINAL_BAND
ROLLOUT_SHARDS = 2
NLA_SHARDS = 2

# --- Patchscopes (evals/downstream/common/patchscope.py): the arm at layer 42, judged, and its no-patch floor, read by
# the word rule only ---
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

PATCH_ARM = "patch42"
PATCH_LAYER = READ_LAYER
PATCH_RULE = PATCH_RULE_TUNED
PATCH_ARMS = (PATCH_ARM, PATCH_FLOOR)
PATCH_FLOOR_FROM_BAND = FINAL_BAND  # the floor reads no activation: its mean readout is its final one
PATCH_GEN = {k: ARMS[HEADLINE_ARM][k] for k in ("n_samples", "greedy", "temp", "top_p", "top_k", "min_p",
                                                "max_new", "min_new")}
PATCH_SEED_OFFSET = {PATCH_ARM: 40, PATCH_FLOOR: 41}
PATCH_REL = "rollouts/patchscope.json"
PATCH_JUDGED = {PATCH_ARM: "patch42_n8"}
assert all(r in JUDGED_READERS for r in PATCH_JUDGED.values())

# --- headline scope ---
HEADLINE_CONDITIONS = ("focus", "mention")
FLOOR_CONDITION = "baseline"
CONTROL_CONDITIONS = ("ignore", "dont_think")
MEAN_NORM_RATIO_RANGE = (0.25, 2.0)  # the median ||mean row|| / ||carrier row|| must fall in this range
