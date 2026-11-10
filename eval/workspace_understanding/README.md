# Workspace understanding

Given the layer-42 residual at one token of a prompt, does a MAEMM rollout name content the model
represents there but has not written: the unnamed concept of an association passage, or the bridge entity
of a two-hop question? MAEMM is scored beside the released Jacobian lens (J-lens), Patchscopes, the NLA
activation verbalizer and a search of ten million tokens of web text, on the two prompt sets released with
the lens, by a whole-word rule and by an LLM judge with a foil.

[`methodology.md`](methodology.md) is the protocol and the one description of every reader, instrument and
metric. This file documents the commands and the outputs.

## Reproducing the paper

The Association and Multi-hop columns of the paper's two workspace tables (word rule, pass@8; judged
naming, net = named − foil) come from one full run under the default judge profile `sonnet`:

```bash
PYTHONPATH=$PWD python -m eval.workspace_understanding all --run-id <run-id> --judge-profile sonnet
# or on Modal
modal run --detach eval/modal_workspace_understanding.py::run_all --run-id <run-id> --judge-profile sonnet
modal run eval/modal_workspace_understanding.py::pull --run-id <run-id>
# the paper's tables, joined with a workspace_modulation run (eval/analysis/paper/README.md)
PYTHONPATH=$PWD python -m eval.analysis.paper workspace --wu-run eval/out/workspace_understanding/<run-id> \
    --wm-run eval/out/workspace_modulation/<run-id> --out <out-dir>
```

The package's cells are in `tables/paper_workspace_word_rule.{csv,tex}` and
`tables/paper_workspace_judged_net.{csv,tex}` (each `.csv` row names the `rates.csv` or `judged.csv` row it
copies and its interval). The rows are MAEMM, NLA, J-lens (`word_top10` at layer 42; judged through
`jlens_L42_summary`), J-lens L36–50 (`word_top10` of `jlens_band8`; `jlens_band8_summary`), Patchscopes
(`patch42`) and corpus search (methodology §6.1–§6.2). The runs of record are listed in
`eval/analysis/paper/README.md`. The run also computes diagnostics no paper number reads: the 64-token NLA
row, the position controls, the untrained-base ablation, the lens's best-layer rank and layer curve, the
Patchscopes floor and the re-read cosines.

**Cost and time.** About 7,600 GPU-seconds on B200 cards (about US$13 at US$6.25/hour), of which the corpus
search is about 2,700 and the two NLA stages about 2,900; roughly an hour of wall-clock time on Modal with
eight GPU workers, a few hours sequentially on one card. Judge and summariser: about US$8 at list price
(Claude Sonnet 5, US$2 / US$10 per million input / output tokens; naming about US$7, the lens summaries
about US$1), under the US$20 cap (`config.JUDGE_BUDGET_USD`; every reply at its cap would be about
US$13). `sol` is priced the same.

## Inputs and pins

| Input | Pin |
|---|---|
| Base model | `Qwen/Qwen3.6-27B` (`eval/common/pins.py`, `MODEL` @ `MODEL_REVISION`) |
| Inverter | `eval/common/pins.py`, `INVERTER` @ `INVERTER_REVISION`, a full-parameter checkpoint |
| Released lens | [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) @ `0731326edff4ae730ffc5356fe1a4728c748b3a6`, file `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` |
| NLA verbalizer | `eval/common/pins.py`, `NLA_REPO` @ `NLA_REVISION` (read by `eval/common/nla/nla_reader.PINS`) |
| Prompt sets | `datasets/lens-eval-{association,multihop}.json`, from `anthropics/jacobian-lens` @ `581d398613e5602a5af361e1c34d3a92ea82ba8e` ([`datasets/SOURCES.md`](datasets/SOURCES.md)) |
| Search corpus | the suite's shared corpus (`eval.common.retrieval.CorpusSpec()`), rebuilt from `openbmb/Ultra-FineWeb` @ `02c85641e3d19a854be2e09139c25adaa9518063` by `eval/common/assets/heldout_docs.csv` |
| Centring mean | `eval.common.background.load_centring_mean()`, digest-checked |
| Judge | `--judge-profile sonnet` (default; Claude Sonnet 5, Anthropic API) or `sol` (GPT-5.6 Sol, OpenRouter), `eval/common/judges.py`; the judge's model also writes the lens summaries |

## Install

```bash
cd path/to/maemm   # the repository root
pip install -r eval/common/requirements.txt -r eval/workspace_understanding/requirements.txt
export PYTHONPATH=$PWD
```

GPU dependencies are imported inside functions.

## Commands

```bash
# the full run on one machine (two stages hold two 27B checkpoints at once, about 108 GB)
PYTHONPATH=$PWD python -m eval.workspace_understanding all --run-id <run-id>
# a smoke run: three items per family, two corpus documents, every stage
PYTHONPATH=$PWD python -m eval.workspace_understanding all --run-id <run-id> --smoke
# one stage, or a re-render of the report from saved artifacts (no model, no judge)
PYTHONPATH=$PWD python -m eval.workspace_understanding prepare --run-id <run-id>
PYTHONPATH=$PWD python -m eval.workspace_understanding report --run-id <run-id>   # + --smoke on a smoke run
```

- `--run-id` names `eval/out/workspace_understanding/<run-id>/`; `--output-dir` gives an exact directory.
  One is required.
- `--judge-profile {sonnet,sol}` (default `sonnet`). A run directory keeps the profile it was started under.
- `--judge-budget-usd` overrides the cap of the one ledger over the judge and the summariser
  (`config.JUDGE_BUDGET_USD`, US$20).
- `--shard k/n` (0-based) splits `nla` and `nla_control` by item and `retrieval` by corpus window;
  `--merge` marks the call that merges a sharded `retrieval`.
- `--force` redoes a stage whose key still matches. Each stage's key chains to the records it reads, so a
  stage run again brings back exactly what is downstream of it.
- `--smoke` is part of every stage's key: a later call on a smoke directory, `report` included, passes it
  too.

The judge stages need `ANTHROPIC_API_KEY` (`sonnet`) or `OPENROUTER_API_KEY` (`sol`); the GPU stages need
Hugging Face access to the pinned checkpoints.

On Modal (`eval/modal_workspace_understanding.py`; app, volume, secret and GPU names from the `EVAL_*`
variables in `eval/modal.env.example`, see `eval/EVALS.md`). The controller needs the `modal` client
(`pip install modal`) and a Modal profile. The Anthropic secret (`EVAL_ANTHROPIC_SECRET`) is always mounted;
`--judge-profile sol` also needs `EVAL_OPENROUTER_SECRET` set:

```bash
modal run --detach eval/modal_workspace_understanding.py::run_all --run-id <run-id> --smoke
modal run --detach eval/modal_workspace_understanding.py::run_all --run-id <run-id>
modal run --detach eval/modal_workspace_understanding.py::stage --name rollouts --run-id <run-id> --force
modal run eval/modal_workspace_understanding.py::pull --run-id <run-id>
PYTHONPATH=$PWD python -m eval.workspace_understanding report --run-id <run-id>   # + --smoke on a smoke run
```

`run_all` fans `nla` and `nla_control` out by item shard and `retrieval` out over eight corpus blocks
followed by one merge call on CPU, with at most `EVAL_GPU_WORKERS` (default 8) GPU containers in flight;
`pull` copies the run directory to `eval/out/workspace_understanding/`.

## Stages

| Stage | Model(s) | Writes |
|---|---|---|
| `prepare` | tokenizer | `data/items.json`: items, readout positions, target forms, exclusions, donors, foils |
| `corpus` | tokenizer | `retrieval/corpus.{npz,json}`: the search corpus |
| `capture` | base | `activations/h_all.npz`, `norms.json`, `data/diagnostic.json` |
| `lens` | base | `lens/lens.json`, `ranks.npz`: per-layer ranks and top-10 lists |
| `rollouts` | inverter + base | `rollouts/maemm.json`, with the re-read cosines and MAEMM's enforced injection check |
| `retrieval` | base | `retrieval/scores.part<k>of<n>.npz`, `rollouts/retrieval.json` |
| `patchscope` | base | `rollouts/patchscope.json`: `patch42`, the floor, the patch check |
| `nla` | verbalizer | `rollouts/nla/shard_*.json`: native and 64-token readouts |
| `nla_control` | base, then verbalizer | `activations/controls/*.npz`, `rollouts/nla_control/shard_*.json` |
| `untrained_base` | base | `rollouts/untrained_base.json` (the ablation) |
| `maemm_control` | inverter + base | `rollouts/maemm_control.json` |
| `reread` | base | `rollouts/reread.json` |
| `summarise` | — | `judges/summaries.json`, `judges/<judge>/summaries.jsonl`: the lens summaries |
| `judge` | — | `judges/<judge>/{naming,diagnostics}.jsonl`, `judges/ledger.json` |
| `report` | — | `scores/items.jsonl`, `tables/*`, `figures/*`, `report.md` |

One 27B checkpoint in bf16 is about 54 GB; `rollouts` and `maemm_control` hold the inverter and the base
at once and free the inverter before re-reading.

## Tables and figures

Every aggregate CSV shares one column order (`metric`, `condition`, `group`, `budget_type`, `budget`,
`estimate`, `ci_lower`, `ci_upper`, `n_total`, `n_valid`, `ci_method`, then its own columns). Rates are
0–1 over items; `ci_method` is `wilson95` or `bootstrap_item_10000_seed0`.

| File | Contents |
|---|---|
| `tables/rates.csv` | pass@N, greedy hit, consistency and chance per reader; the lens's `rank_le_k` and `word_top10` rows |
| `tables/judged.csv` | named, foil, net, voided, unavailable per condition, group and judge |
| `tables/contrasts.csv` | paired differences; `judge` empty on a judge-free row |
| `tables/paper_workspace_{word_rule,judged_net}.{csv,tex}` | the paper's cells (above) |
| `tables/reread.csv` | re-read cosine own, foil and gap per reader |
| `tables/layer_curve.csv` | lens rank ≤ 10 per fitted layer |
| `tables/diagnostic_split.csv` | the multi-hop competence split |
| `tables/missingness.csv` | per judge, instrument and condition: how each request settled |
| `tables/coverage_and_costs.json` | population, coverage, spend, stage times |
| `figures/01_budget_curves` … `03_judged` | budget curves, lens layer curve, judged naming |

`report` calls no model or judge; two renders of one directory in one environment are identical (PDF
figures carry a creation date).

## Limits

One base model, one inverter, one read layer and two prompt families. The judge reads the lens through
summaries its own model wrote. Patchscopes is one recipe, read against a no-injection floor by the word
rule. The corpus search is one corpus at one size, a floor rather than a method. See
methodology §1 and §7.3.
