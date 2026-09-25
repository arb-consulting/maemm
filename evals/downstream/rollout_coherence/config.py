"""Every pin, size, seed, prompt, method and price of the rollout-coherence eval (methodology.md)."""
from evals.downstream.common import retrieval
from evals.downstream.common.judges import active, judges, rates
from evals.downstream.common.nla.nla_reader import Pins
from evals.downstream.common.pins import INVERTER, INVERTER_REVISION, MODEL, MODEL_REVISION  # noqa: F401
from evals.downstream.common.retrieval import DATASET
from maem.config import D_MODEL, INJECT_LAYER, READ_LAYER, STEER_COEFF  # noqa: F401  5120, 1, 42, 1.0

SCORING_VERSION = "4"

# ---------------------------------------------------------------- documents and activations (§3)
CORPUS = dict(DATASET)             # the public files the held-out corpus is rebuilt from
N_SOURCES = 400
SMOKE_SOURCES = 12
SEQ_LEN = 512
# The read position p ~ U[POS_LO, POS_HI], and the raw-norm filter NORM_FLOOR < ||h|| <= NORM_MULT x the
# presample median (methodology §3).
POS_LO, POS_HI = 64, 511
NORM_MULT = 10.0
NORM_FLOOR = 1e-3
NORM_PRESAMPLE = 4096
NORM_PRESAMPLE_DOCS = 64
DOC_SEED = 2                       # the seeded draw of the documents (retrieval.select_documents)
DATA_SEED = 0                      # positions, spare positions, then the norm presample


def pool_size(n_sources):
    """Candidates the norm filter selects `n_sources` activations from: max(2n, n + 256)."""
    return max(2 * n_sources, n_sources + 256)


# What the stage hashes carry of the draw rule.
DRAW = {"rule": "reject-from-pool", "pos": [POS_LO, POS_HI], "norm_mult": NORM_MULT,
        "norm_floor": NORM_FLOOR, "presample": NORM_PRESAMPLE, "presample_docs": NORM_PRESAMPLE_DOCS}


def sizes(smoke):
    n = SMOKE_SOURCES if smoke else N_SOURCES
    return {"n_sources": n, "n_pool": pool_size(n)}


# ---------------------------------------------------------------- the texts (§4)
# `frontier_context`'s methods, `targets` first (every other one reads its output).
CONTEXT_METHODS = ("targets", "retrieval", "maem", "continuation", "nla")
GEN_SEED = 1234
# Added to GEN_SEED per generating method, so each method draws its own stream.
SEED_OFFSET = {"maem": 11, "continuation": 2, "nla": 13}
N_SAMPLES = 8                      # the judged draws of every sampled method (the greedy is extra)
# `maem` draws 64 for the best-of-k curve past N_SAMPLES; the verbalizer's count is its own pin.
N_SAMPLES_BY_ARM = {"maem": 64, "continuation": N_SAMPLES}


def n_samples(arm):
    """Samples per activation of one generating arm (the greedy is extra)."""
    return N_SAMPLES_BY_ARM.get(arm, N_SAMPLES)


SAMPLING = {"temp": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_new": 64, "min_new": 16}
GEN_CHUNK = 32                     # rows per generate() call (model_io.MAX_GEN_ROWS)

# ---------------------------------------------------------------- the judge (§5)
# One cap and one ledger, sized for the full run under the default profile (about US$40) with headroom.
JUDGE_BUDGET_USD = 50.0
LEDGER = "judges/ledger.json"
INSTRUMENT = "context"             # the log: judges/<judge>/context.jsonl
JUDGE_WORKERS = 16
MAX_UNPARSEABLE_SHARE = 0.02

SYSTEM_PROMPT = (
    "You are a careful editor assessing short passages of text. You will be shown two passages, labelled A and B. "
    "Both are excerpts: each may begin or end in the middle of a sentence, and neither should be penalised for where "
    "it starts or stops. Judge only the text between the passage markers. Reply with JSON only."
)
USER_TEMPLATE = (
    "PASSAGE A:\n<<<\n{a}\n>>>\n\nPASSAGE B:\n<<<\n{b}\n>>>\n\n"
    "Which passage is more coherent and natural as a piece of text? Consider grammatical consistency, logical "
    "continuity, lack of degeneration/repetition, and whether the passage maintains a comprehensible meaning. "
    "If the two are comparable, answer tie.\n\n"
    'Reply with JSON only: {{"choice": "A" | "B" | "tie"}}'
)

# The active profile's judge; the reply is one word of JSON, capped at 16 tokens.
JUDGES = judges({"pair": 16})
JUDGE_PROFILE = active()
RATES_PER_M = rates(JUDGES)

# ---------------------------------------------------------------- fluency (§6)
FLUENCY_SCORER = {"model": "google/gemma-4-31B",            # pretrained, loaded text-only, bf16, sdpa
                  "revision": "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89"}
FLUENCY_BATCH_TOKENS = 16384       # padded tokens per scoring batch
FLUENCY_MAX_BATCH = 256            # items per scoring batch
FLUENCY_LOGIT_CHUNK = 2048         # scored positions per logits chunk
LL_PARTNER_SANE = (-5.0, -2.0)     # the range a source passage's nats/token is checked against

N_BOOT, BOOT_SEED = 10000, 0

# ---------------------------------------------------------------- the methods on the figure (§4, §7)
# One row per method: `curve` "best_of_k", "corpus_size" or "single"; `role` "method" (compared),
# "reference" or "y_reference" (never sees the activation: a fluency and no inversion position).
PLOTTED = {
    "maem": {"label": "MAEM", "colour": "#0072B2", "curve": "best_of_k", "role": "method"},
    "continuation": {"label": "Base model's own continuation of the source text, temperature 1 (reference)",
                     "colour": "#56B4E9", "curve": "best_of_k", "role": "y_reference"},
    "retrieval": {"label": "Corpus retrieval", "colour": "#E69F00", "curve": "corpus_size", "role": "method"},
    "nla": {"label": "NLA (its explanation's first 64 tokens)", "colour": "#009E73", "curve": "best_of_k",
            "role": "method"},
    "nla_native": {"label": "NLA (the whole explanation)", "colour": "#CC79A7", "curve": "best_of_k",
                   "role": "method"},
    "source": {"label": "Source passage (reference)", "colour": "#000000", "curve": "single",
               "role": "reference"},
}
METHOD_COLOURS = {m: spec["colour"] for m, spec in PLOTTED.items()}
METHOD_LABELS = {m: spec["label"] for m, spec in PLOTTED.items()}
REFERENCES = tuple(m for m, spec in PLOTTED.items() if spec["role"] != "method")
Y_REFERENCES = tuple(m for m, spec in PLOTTED.items() if spec["role"] == "y_reference")

BEST_OF_K = (1, 2, 4, 8, 16, 32, 64)
HEADLINE_K = 1                     # the point methods are compared at: one sample, no selection
MAIN_K = N_SAMPLES                 # up to here every sample is judged
EXTENDED_ARMS = ("maem",)  # the arms whose curve continues past MAIN_K
EXTENDED_K = tuple(k for k in BEST_OF_K if k > MAIN_K)


def selection_group(arm, k):
    """The pair group of one arm's judged best-of-k selection at k > MAIN_K."""
    return f"{arm}_k{k}"


# The corpus-search baseline (evals/downstream/common/retrieval.py) and the nested prefix sizes of its curve.
SEARCH_CORPUS = retrieval.CorpusSpec()
WINDOW_TOKENS = SEARCH_CORPUS.window   # one window, in the corpus's own tokens: also the texts' budget
SIZES = retrieval.SIZES
SMOKE_CORPUS_DOCS = 2
SMOKE_CORPUS_SIZES = (("half", 1), ("all", 2))   # documents, not tokens

BANK_BATCH = retrieval.WINDOW_BATCH        # texts (or corpus windows) per forward pass of a re-read
NLA = Pins()                       # the verbalizer as evals/downstream/common/nla pins it
NATIVE_MAX_TOKENS = NLA.score_max_length   # the re-read window of the full explanation
NATIVE_BATCH = 32


def corpus_docs(smoke, index=None):
    """The held-out search-prefix documents (the first `SMOKE_CORPUS_DOCS` for a smoke run)."""
    docs = retrieval.search_docs(index or retrieval.load_index(), SEARCH_CORPUS)
    return docs[:SMOKE_CORPUS_DOCS] if smoke else docs


def size_tokens(smoke, index=None):
    """`((label, tokens), ...)`: the nested corpus sizes as token counts (a smoke run's own documents' ends)."""
    if not smoke:
        return SIZES
    docs = corpus_docs(True, index)
    return tuple((label, docs[n - 1].offset + docs[n - 1].n_tokens) for label, n in SMOKE_CORPUS_SIZES)


def corpus_sizes(smoke, index=None):
    """`((label, n_windows), ...)` of the nested corpus sizes, and the corpus's own window count."""
    docs = corpus_docs(smoke, index)
    table = retrieval.sizes_windows(docs, size_tokens(smoke, index), SEARCH_CORPUS)
    return tuple((str(label), int(windows)) for label, _tokens, windows in table), retrieval.total_windows(docs)
