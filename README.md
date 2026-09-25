# MAEMM: a universal activation-to-text inverter

MAEMM trains a LoRA adapter that turns an activation **direction** inside a language model back
into **text** whose own activation, on a clean forward pass, points the same way. Given any
direction in the residual stream (an SAE feature, a probe, a raw activation), it writes a short
span that evokes it.

**Method.** Inject a unit direction `v` at `INJECT_LAYER` on a marker token (`h <- h + ||h|| v`),
generate, then re-read the text through the clean model (adapter off) at `READ_LAYER` and score
the max-over-tokens cosine with `v`. Training is SFT on real corpus spans, then GRPO RL on that
score. The defaults target `Qwen/Qwen3.6-27B` (inject layer 1, read layer 42); see `maemm/config.py`.

## Layout

```
maemm/                 the core library: config, injection and read hooks, prompts, SAE loading
data/                  activation collection and (direction, target-span) bank building
train/sft/             supervised stage: LoRA or full fine-tune
train/rl/              RL stage: GRPO with disaggregated vLLM rollouts
sae/                   the 2M-feature BatchTopK SAE used by the evaluations, and its max-act bank
evals/heldout/         held-out direction families: cosine, SAE activation, % unverbalized
evals/faithfulness/    the main evaluations: fidelity, OOD generalisation, SAE autointerp, GCG
evals/verbalization/   which SAE features the inverter can and cannot express, and why
evals/backdoor/        reading rank-one backdoors out of LoRA weights
evals/downstream/      steering-vector inversion, workspace content interpretation, rollout coherence
tests/                 CPU tests, one folder per component
```

Each folder has a README listing its scripts; every script's docstring gives its exact command.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD          # `import maemm` from anywhere in the tree
modal setup                     # GPU work runs on Modal
```

Create these Modal secrets in your workspace (each holds one API key):

| secret | holds |
|---|---|
| `maemm-hf` | `HF_TOKEN` |
| `maemm-wandb` | `WANDB_API_KEY` |
| `maemm-anthropic` | `ANTHROPIC_API_KEY` (LLM-judge and autointerp stages; optional `ANTHROPIC_WORKSPACE_ID`) |
| `maemm-openrouter` | `OPENROUTER_API_KEY` (the LLM baseline) |

Create the three volumes the pipeline reads from (the others are created on first use):

```bash
modal volume create maemm-data && modal volume create maemm && modal volume create maemm-dit
```

**Checkpoints** are referenced as `ANONYMOUS/<name>` on the HuggingFace Hub. The optional `--prefix-cache` speed path needs a patched `transformers`,
referenced as `ANONYMOUS/transformers@<commit>`; every default run uses the released package.

The NLA baseline in `evals/faithfulness/` is not part of the method: its `nla` entry in
`config.yaml` takes any NLA verbalizer checkpoint (`hf:` and its pinned `revision:`).

## Reproducing

1. **Data** (`data/`): collect layer-42 activations, then build the SFT and RL banks.
2. **SFT** (`train/sft/modal_sft.py`), then **RL** (`train/rl/modal_rl_disagg.py`) from the SFT init.
3. **Evaluations**: `evals/faithfulness/` for the main tables; `evals/verbalization/`,
   `evals/backdoor/` and `evals/heldout/` for the remaining sections; `evals/downstream/` for the
   steering-vector, workspace and coherence studies (its README lists the launchers).

## Tests

```bash
pytest tests/
```

`tests/sae/test_online_pool.py` and `test_sharded_equivalence.py` run two ranks under `torchrun`
(gloo) and need Linux.
