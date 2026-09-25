# Rollout coherence

Does the text MAEMM writes for a layer-42 activation read as coherently and as fluently as the passage the
activation came from, and how does that trade against how well the text inverts the activation? 400
activations of Qwen3.6-27B from held-out Ultra-FineWeb documents, each target re-captured from its 64-token
source tail (the matched target); MAEMM (64 samples and a greedy decode) against the base continuing the
source text, the NLA verbalizer (whole and truncated to 64 tokens) and corpus search (1M–10M tokens). A
blind pairwise LLM judge compares each text with its source passage in both orders; Gemma-4-31B scores the
per-token log-likelihood gap.

The design is in [`methodology.md`](methodology.md); this file documents how to run it.

## Reproducing the paper

The paper's section "Coherence of MAEMM exemplars" reads one run of this package: `all`, then the paper
builder, which draws Figure `fig:coherence` (`coherence_two_plots.{pdf,png}`) and writes the section's
numbers from the run's tables.

```bash
export PYTHONPATH=$PWD
# on Modal (at most EVAL_GPU_WORKERS GPU containers at once, default 8; about 3 h wall time)
modal run --detach eval/modal_rollout_coherence.py::run_all --run-id <run-id>
modal run eval/modal_rollout_coherence.py::pull --run-id <run-id>
# or on one CUDA host
python -m eval.rollout_coherence all --run-id <run-id>
# the figure and the in-text numbers
python -m eval.analysis.paper coherence --run eval/out/rollout_coherence/<run-id> --out <dir>
```

- **Judge profile:** `sonnet` (Claude Sonnet 5), the package default; it needs `ANTHROPIC_API_KEY`.
- **Stages:** `all`, which runs every stage below and then `report`.
- **Budget:** about 3 B200-hours of GPU and about US$40 of judge spend (35,200 requests) under `sonnet`,
  against the package's cap of US$50 (`config.JUDGE_BUDGET_USD`).

The run of record is listed in [`eval/analysis/paper/README.md`](../analysis/paper/README.md).
`tables/frontier_matched.csv` holds every point of every curve and `tables/frontier_outcomes.csv` the share
of pairs at least as coherent as the source; `figures/06b_frontier_matched_centred` is the package's own
rendering of the same two panels.

## Install

```bash
pip install -r eval/common/requirements.txt -r eval/rollout_coherence/requirements.txt
export PYTHONPATH=$PWD      # eval and mxf are imported as top-level packages
```

On a GPU host install torch first from the wheel index for its CUDA version. `flash-linear-attention` is
installed on Linux only; without it the model's linear-attention layers run on transformers' slower path.

## Commands

```bash
python -m eval.rollout_coherence all --run-id <run-id> [--smoke]    # every stage, then the report
python -m eval.rollout_coherence <stage> --run-id <run-id> [--force]
python -m eval.rollout_coherence report --run-id <run-id> [--smoke] # re-render from saved artifacts
```

`--smoke` is part of every stage's key and of `config.json`: a later call on a smoke directory, `report`
included, passes it too.

| Flag | Default | Meaning |
|---|---|---|
| `--run-id R` / `--output-dir D` | required, one of them | `eval/out/rollout_coherence/R`, or the exact directory `D` |
| `--judge-profile` | `sonnet` | `sonnet` or `sol` (`eval/common/judges.py`); a run directory keeps the profile it started under |
| `--judge-budget-usd` | `config.JUDGE_BUDGET_USD` (50) | the cap the judge pass spends against |
| `--smoke` | off | 12 activations and a two-document corpus, every stage and guard |
| `--force` | off | redo the stage even when its saved key matches |
| `--device` | `cuda:0` | |
| `--context-methods` | all | `frontier_context`: `targets`, `retrieval`, `maemm`, `continuation`, `nla` |
| `--shard k/n` | `0/1` | the retrieval forward: this invocation's block of the corpus |
| `--merge` | off | the retrieval forward's merge call: fail on a missing part instead of waiting |

`--context-methods` and `--merge` stay out of `config.json`: they say what one invocation does, not what the
run is. Every stage resumes when its saved key (its settings, chained to the records it reads) matches;
`--force` on an upstream stage invalidates everything downstream of it.

**On Modal** (`eval/modal_rollout_coherence.py`), every stage runs the same CLI in a container. `run_all`
walks the dependency levels; `frontier_context` is spawned as waves (`targets`, then `maemm`,
`continuation` and `nla` one container each beside the retrieval shards, `EVAL_RETRIEVAL_SHARDS` of them
(default 4), then the retrieval merge) and the CPU stages run inline, one at a time. Launch it with `--detach`: `run_all` awaits every stage, and a client
that disconnects cancels them; relaunching resumes. Other entrypoints:

```bash
modal run --detach eval/modal_rollout_coherence.py::stage --name frontier_context_pairs --run-id <run-id> --force
modal run --detach eval/modal_rollout_coherence.py::stage --name frontier_context --run-id <run-id> \
    --extra "--context-methods retrieval --shard 3/4"
```

Names, GPUs and shard counts come from the environment (`eval/EVALS.md`, "Launcher environment"). The
controller needs the `modal` client (`pip install modal`) and a Modal profile. The launcher mounts `EVAL_ANTHROPIC_SECRET` (default `maemm-anthropic`) and, only when set,
`EVAL_OPENROUTER_SECRET` (needed for `--judge-profile sol`).

## Stages

| Stage | Compute | Writes |
|---|---|---|
| `prepare` | CPU | `data/documents.json`: the candidate documents and their read positions |
| `capture` | GPU, base | `data/documents.json`'s `sources`: the first 400 candidates the norm filter keeps |
| `frontier_corpus` | CPU | `frontier/corpus.{npz,json}`: the search corpus's token ids and nested sizes |
| `frontier_context` | GPU, base (+ inverter, verbalizer) | `frontier/context/{targets.npz,targets.json,texts/,scores/,selfcheck/}` |
| `frontier_context_pairs` | CPU | `frontier/context/pairs.json` |
| `frontier_context_judge` | API | `judges/<judge>/context.jsonl`, `judges/ledger.json` |
| `frontier_context_fluency` | GPU, Gemma | `scores/context_fluency.jsonl` |
| `report` | CPU | `tables/`, `figures/`, `report.md` |

Dependency levels: `[prepare]` → `[capture, frontier_corpus]` → `[frontier_context]` →
`[frontier_context_pairs]` → `[frontier_context_judge, frontier_context_fluency]` → `[report]`. A judge
pass that ends with a request never really asked (the cap, the transport, or the key turned away) fails
with nothing marked complete; running it again asks exactly those requests. `report` always re-renders and
refuses a run directory missing any artifact.

## Output layout

```text
eval/out/rollout_coherence/<run-id>/
  config.json  provenance.json  provenance/<stage>.json  stages/<stage>.json
  data/documents.json
  frontier/corpus.{npz,json}
  frontier/context/{targets.npz,targets.json,pairs.json}  frontier/context/{texts,scores,selfcheck}/
  judges/<judge>/context.jsonl  judges/ledger.json
  scores/context_fluency.jsonl
  tables/{frontier_matched,frontier_outcomes,missingness}.csv  tables/coverage_and_costs.json
  figures/06b_frontier_matched_centred.{pdf,png}  report.md
```

The tables are described in methodology §7. `coverage_and_costs.json` carries the coverage, the checks,
the request and token counts, the measured GPU seconds and the judge ledger.

## Changing the model or the methods

A run directory belongs to one set of pins: start a new `--run-id` after changing one. The models are named
in `config.py` (`MODEL`, `INVERTER` via `eval/common/pins.py`, `FLUENCY_SCORER`, `NLA`) and the judge in
`eval/common/judges.py`. `config.PLOTTED` is the one table of what the figure and tables carry; a method
that writes its own text also needs a function in `frontier_context.METHODS`, a group in
`frontier_pairs.CONTEXT_GROUPS` and a place in the launcher's plan (`CONTEXT_JOBS`), which
`_check_stage_lists` enforces.
