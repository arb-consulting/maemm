"""Settings of the persona-vector eval: the population, the descriptions, training, readers and judges.

pair row      {"question", "matching", "not_matching", "source_id"}
vector entry  {"vector_id": "persona:<name>/bipo/s0/+", "behaviour", "train_seed", "epoch",
               "direction": unit float32[5120], "norm", "truth": ["persona:<name>/+"]}
family        {"family_id": "<vector_id>|<arm>", "vector_id", "arm", "samples": [{"sample_id", "text", ...}]}
"""
from dataclasses import replace
import json
from pathlib import Path

from evals.downstream.common.judge_client import JudgeSpec
from evals.downstream.common.judges import REFERENCE_JUDGE, active, judges, rates
from evals.downstream.common.pins import LENS_BYTES, LENS_FILE, LENS_REPO, LENS_REVISION, LENS_SHA256
from evals.downstream.common.retrieval import CorpusSpec
from evals.downstream.steering_vector_inversion.config import IDENTIFICATION_MAX_TOKENS

DATA = Path(__file__).parent / "assets"

EVALS_REPO, EVALS_SHA = "anthropics/evals", "84fcc677e52e1902d696c32cd1a6b663e70d3993"
PERSONA_PATH = "persona/{name}.jsonl"  # in EVALS_REPO at EVALS_SHA

# Every eligible persona file, in the fixed order of `assets/personas.json`.
POPULATION_ASSET = DATA / "personas.json"
_POPULATION = json.loads(POPULATION_ASSET.read_text(encoding="utf-8"))
PERSONAS = tuple(_POPULATION["order"])
assert len(set(PERSONAS)) == len(PERSONAS) == 119 and set(PERSONAS) == {
    name for name, theme in _POPULATION["themes"].items() if theme not in _POPULATION["excluded_themes"]}
BEHAVIOURS = tuple(f"persona:{name}" for name in PERSONAS)

PERSONA_DESCRIPTIONS_ASSET = DATA / "descriptions.json"

N_TRAIN_CAP, N_HELDOUT = 700, 200
DATA_SEED = 0
# In units of theta, v = sigma * theta (gpu.train): this package's step size, not BiPO's published one.
HP = {"lr_theta": 4e-3, "epochs": 20, "beta": 0.1, "weight_decay": 0.05, "warmup_steps": 100,
      "batch_pairs": 4, "max_tokens": 512, "log_every": 20, "heldout_eval_pairs": 100}
CKPT_EPOCHS = (1, 2, 3, 5, 10, 15, 20)
PRIMARY_EPOCH = 20

BUNDLES, BUNDLE_SIZE = 8, 8
CORPUS = CorpusSpec()
RETRIEVAL_SHARDS = 8
SMOKE_DOCS = 2                       # `--smoke` searches the corpus's first documents in one block
SMOKE_RETRIEVAL_SHARDS = 1
LENS_TOP_WORDS = 10
LENS = {"repo": LENS_REPO, "revision": LENS_REVISION, "file": LENS_FILE, "n_bytes": LENS_BYTES,
        "sha256": LENS_SHA256}

JUDGES = judges({"identify": IDENTIFICATION_MAX_TOKENS, "summary_req": 300})
# The writer of the frozen persona descriptions (the opt-in `describe` stage), whatever the profile.
DESCRIBE_JUDGE = JudgeSpec(name="opus", model="claude-opus-5", provider=None,
                           sampling={"thinking": {"type": "disabled"}}, max_tokens={"describe": 3000},
                           label="Claude Opus 5", rates=(5.0, 25.0), transport="anthropic")
RATES_PER_M = rates(JUDGES)
JUDGE_PROFILE = active()
SUMMARISER = REFERENCE_JUDGE         # turns the lens's token list into the lens arm's one text
JUDGE_CAP_USD = 50.0                 # for the whole run
RUN_RECORD = "bipo_config.json"


def persona_descriptions():
    """`{"persona:<name>/<pole>": sentence}`, checked against the asset's digest."""
    from evals.downstream.steering_vector_inversion.artifacts import digest
    record = json.loads(PERSONA_DESCRIPTIONS_ASSET.read_text(encoding="utf-8"))
    descriptions = record["descriptions"]
    if digest(descriptions) != record["digest"]:
        raise ValueError(f"{PERSONA_DESCRIPTIONS_ASSET} does not match the digest it carries")
    return descriptions


def svi_config():
    """The pinned package's configuration: the clean base, the inverter, read 42, inject 1."""
    from evals.downstream.steering_vector_inversion.config import Config
    return replace(Config()).validate()
