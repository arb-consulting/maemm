"""The package's settings: its arms, identification budgets, judges and caps, and the run configuration.
`methodology.md` defines every arm."""
from dataclasses import asdict, dataclass

from eval.common import plain_steer
from eval.common.judges import active, judges, rates
from eval.common.pins import INVERTER, INVERTER_REVISION, MODEL, MODEL_REVISION

# --- arms ---------------------------------------------------------------------------------------------
# The steered model: the table strength at 64 samples and a greedy decode, the other strengths at 16.
STEER_SAMPLES = plain_steer.samples_by_strength()
STEER_GREEDY = (plain_steer.TABLE_STRENGTH,)
STEERED = tuple(plain_steer.arm_name(s) for s in STEER_SAMPLES)
STEERED_TABLE = plain_steer.TABLE_ARM
STEERED_CURVE = tuple(arm for arm in STEERED if arm != STEERED_TABLE)
GREEDY = ("maemm", "base_l1", "nla_native", STEERED_TABLE)
SINGLE_TEXT = ("jlens",)                # one text, asked once in the greedy slot
CONDITIONS = ("maemm", "base_l1", "nla_native", *STEERED, "shuffled", "retrieval", "heldout_positive",
              *SINGLE_TEXT)
GENRES = ("text", "code", "math")
BUDGETS = (1, 2, 4, 8)                  # texts shown together
CURVE_BUDGETS = (1, 8)                  # for a strength other than the table's
LENS_TOP_WORDS = 10
# A token whose residual norm exceeds this many times its text's median is left out of the text's mean.
NORM_FILTER_MULT = 10.0
# The preflight's clean-read check (`model.CleanReads`).
PREFLIGHT_REPEATS = 8
PREFLIGHT_WARMUP = 3
PREFLIGHT_RTOL = 1e-3

# --- judging ------------------------------------------------------------------------------------------
IDENTIFICATION_MAX_TOKENS = 128
SUMMARY_MAX_TOKENS = 300
JUDGES = judges({"identification": IDENTIFICATION_MAX_TOKENS, "summary_req": SUMMARY_MAX_TOKENS})
RATES_PER_M = rates(JUDGES)
JUDGE_PROFILE = active()                # recorded in config.json and every cache key
CAPS = {"identification": 100.0, "summaries": 3.0}
JUDGE_CAP_USD = sum(CAPS.values())


def budgets(condition):
    """`[(budget_type, budget)]` a condition is asked at."""
    if condition in SINGLE_TEXT:
        return [("greedy", 1)]
    out = [("snippets", b) for b in (CURVE_BUDGETS if condition in STEERED_CURVE else BUDGETS)]
    return out + ([("greedy", 1)] if condition in GREEDY else [])


@dataclass(frozen=True)
class Config:
    model: str = MODEL
    model_revision: str = MODEL_REVISION
    inverter: str = INVERTER
    inverter_revision: str = INVERTER_REVISION
    dataset: str = "pyvene/axbench-concept500"
    dataset_revision: str = "ad8a5d60c4616b599c24dd6689f05f696ec610f3"
    dataset_subset: str = "9b/l20"
    read_layer: int = 42
    inject_layer: int = 1
    hidden_size: int = 5120
    build_count: int = 48
    samples: int = 64
    min_new_tokens: int = 16
    max_new_tokens: int = 64
    seed: int = 1234
    data_seed: int = 0
    bootstrap_samples: int = 10000
    concepts_per_genre: int | None = None
    read_batch_size: int = 8
    generation_batch_size: int = 32
    judge_profile: str = JUDGE_PROFILE

    def validate(self):
        if self.read_layer != 42 or self.inject_layer != 1 or self.hidden_size != 5120:
            raise ValueError("This eval is specified for Qwen3.6-27B: read 42, inject 1, width 5120")
        if self.samples < max(BUDGETS):
            raise ValueError("At least eight samples are required for identification")
        if not 0 < self.min_new_tokens <= self.max_new_tokens:
            raise ValueError("Invalid generation length bounds")
        for n in (self.build_count, self.read_batch_size, self.generation_batch_size, self.bootstrap_samples):
            if n < 1:
                raise ValueError("Counts and batch sizes must be positive")
        if self.concepts_per_genre is not None and self.concepts_per_genre < 1:
            raise ValueError("A smoke concept count must be positive")
        return self

    def to_dict(self):
        """The record config.json holds and every cache key is derived from."""
        return asdict(self)

