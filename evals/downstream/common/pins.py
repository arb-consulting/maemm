"""The artifacts every evaluation package reads, pinned once. Each repository id and revision is a default
that the environment variable beside it overrides (an empty value keeps it); the values in force enter
each run's config and stage keys. The centring mean in `assets/` belongs to the inverter pinned here."""

import os

from maemm.config import MODEL as _TRAINING_BASE  # the base model the training code names

ENV_OVERRIDES = (
    "EVAL_BASE_REPO", "EVAL_BASE_REVISION",
    "EVAL_INVERTER_REPO", "EVAL_INVERTER_REVISION",
    "EVAL_NLA_REPO", "EVAL_NLA_REVISION",
    "EVAL_LENS_REPO", "EVAL_LENS_REVISION",
)


def _pin(name, default):
    assert name in ENV_OVERRIDES, name
    return os.environ.get(name) or default


# The base model: every read, every steered generation and the untrained-base control.
MODEL = _pin("EVAL_BASE_REPO", _TRAINING_BASE)
MODEL_REVISION = _pin("EVAL_BASE_REVISION", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")

# The model under evaluation: a full-parameter fine-tune of the base.
INVERTER = _pin("EVAL_INVERTER_REPO", "ANONYMOUS/maemm-27b-rl-last16-lr5e-7")
INVERTER_REVISION = _pin("EVAL_INVERTER_REVISION", "0000000000000000000000000000000000000000")

# The NLA verbalizer: the base with the verbalizer LoRA merged in (evals/downstream/common/nla/nla_reader.py).
NLA_REPO = _pin("EVAL_NLA_REPO", "ANONYMOUS/qwen3.6-27b-nla-av")
NLA_REVISION = _pin("EVAL_NLA_REVISION", "0000000000000000000000000000000000000000")

# The released Jacobian lens, the commit of its reader library, and the file's size and sha256.
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = _pin("EVAL_LENS_REPO", "neuronpedia/jacobian-lens")
LENS_REVISION = _pin("EVAL_LENS_REVISION", "0731326edff4ae730ffc5356fe1a4728c748b3a6")
LENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
LENS_BYTES = 3303032772
LENS_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
