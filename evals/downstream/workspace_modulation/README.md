# Workspace modulation

Does a MAEMM rollout, read at the final period of a sentence the model copies while told to focus on,
dismiss or merely mention a concept, name that concept — and how does it compare with the released Jacobian
lens, the NLA verbalizer, Patchscopes and a search of the held-out corpus read at the same activation?
The package builds the directed-modulation items, generates MAEMM's headline arm, a null-direction control and
an untrained-base ablation, reads the other readers at the same cells, and scores every readout with a word
rule and the shared naming instrument (named − foil). [`methodology.md`](methodology.md) is the protocol;
this file is the commands and the outputs.

## Reproducing the paper

The paper's Modulation column (Tables "Word rule" and "Judged naming" of the workspace section, and the
focus / mention rates in its text) comes from one full run under the default judge profile `sonnet`
(Claude Sonnet 5):

```bash
# local, one GPU with room for two 27B checkpoints (the rollouts stage holds the base and the inverter)
export PYTHONPATH=$PWD ANTHROPIC_API_KEY=...
python -m evals.downstream.workspace_modulation all --run-id <run-id> --judge-profile sonnet

# or on Modal (evals/downstream/modal.env.example lists the environment)
modal run --detach evals/downstream/modal_workspace_modulation.py::run_all --run-id <run-id> --judge-profile sonnet
modal run evals/downstream/modal_workspace_modulation.py::pull --run-id <run-id>

# the paper's tables, joined with a workspace_understanding run (evals/downstream/analysis/paper/README.md)
python -m evals.downstream.analysis.paper workspace --wu-run evals/downstream/out/workspace_understanding/<run-id> \
    --wm-run evals/downstream/out/workspace_modulation/<run-id> --out <out-dir>
```

The package's cells are in `tables/paper_workspace_word_rule.{csv,tex}` and
`tables/paper_workspace_judged_net.{csv,tex}` (the Modul. column; `workspace_understanding` writes the
other two), each read from a `focus+mention` topics row of `tables/rates.csv` / `tables/judged.csv`
(methodology §5.3). The per-condition focus and mention rates are the `instruction = focus` and
`instruction = mention` rows of the same tables. The runs of record are listed in
`evals/downstream/analysis/paper/README.md`. The run also computes diagnostics no paper number reads: the arithmetic
family, the `ignore` / `dont_think` and carrier-mean readings, the lens's any-token protocol, the null control,
the untrained-base ablation, the 64-token NLA row and the Patchscopes floor.

**Cost and time.** About US$17 of judge and summariser API at list price (naming about US$15.5, summaries
US$1.4; cap `config.JUDGE_BUDGET_USD` = US$30, the worst case with every reply at its cap × 1.25), and about
7,400 measured B200 GPU-seconds (≈ 2.1 GPU-hours, ≈ US$13 at US$6.25/h), about 70 minutes wall clock on
Modal with the default sharding. The corpus forward (`retrieval`, eight shards) is the largest GPU item.
`sol` has the same list rates.

## Inputs and pins

Pinned in [`methodology.md`](methodology.md) §2.1 (base model, inverter, lens, NLA verbalizer, materials,
centring mean, search corpus, judge, prompts); vendoring provenance is in
[`datasets/SOURCES.md`](datasets/SOURCES.md). Every hash is re-verified at load.

## Install

```bash
cd path/to/maemm
pip install -r evals/downstream/common/requirements.txt -r evals/downstream/workspace_modulation/requirements.txt
export PYTHONPATH=$PWD
export ANTHROPIC_API_KEY=...        # --judge-profile sonnet (default); OPENROUTER_API_KEY for sol
```

Install both files in one `pip` call so a version conflict between them surfaces. `torch`, `transformers`
and `jlens` are imported inside the stage functions.

## Commands

```bash
python -m evals.downstream.workspace_modulation all --run-id <run-id> --smoke   # two concepts per family, a two-document corpus
python -m evals.downstream.workspace_modulation all --run-id <run-id>
python -m evals.downstream.workspace_modulation <stage> --run-id <run-id>
python -m evals.downstream.workspace_modulation judge --run-id <run-id> --gate 20
python -m evals.downstream.workspace_modulation report --run-id <run-id>                 # + --smoke on a smoke run
```

`--smoke` is part of every stage's key and of `config.json`: a later call on a smoke directory, `report`
included, passes it too.

| Flag | Meaning |
|---|---|
| `--run-id R` / `--output-dir D` | the run directory (`evals/downstream/out/workspace_modulation/R`, or exactly `D`); one is required |
| `--judge-profile {sonnet,sol}` | the judge (default `sonnet`, Claude Sonnet 5; `sol` is GPT-5.6 Sol). A directory is judged under one profile only |
| `--seed` | MAEMM arms' and Patchscopes' generation seed (default 1234); the NLA arm always uses 1234 |
| `--arms` | generation arms (default `reg,null,base`) |
| `--shard K --n-shards N` | `rollouts`/`nla`: item shard; `retrieval`: corpus-window shard; the merge stages take the same `--n-shards` |
| `--judge-budget-usd` | the ledger cap (default `config.JUDGE_BUDGET_USD`) |
| `--gate N` | run only the judge's gate, on N requests |
| `--retry-truncated CAP` | re-ask only naming replies cut at their cap, at this cap |
| `--smoke`, `--force`, `--device` | smoke population; redo a stage; device |

Modal (`evals/downstream/modal_workspace_modulation.py`): `run_all` runs the dependency levels, spawning GPU stages as
containers (`rollouts` per arm and shard, `nla` and `retrieval` per shard) and CPU stages inline; `stage
--name S --extra "..."` runs one stage; `pull` copies a run off the volume. App, volume, secrets, GPU type,
GPU list rate and the GPU container cap (`EVAL_GPU_WORKERS`, default 8) come from the `EVAL_*` variables in
`evals/downstream/modal.env.example`. The controller needs the `modal` client (`pip install modal`) and a Modal profile.
The Anthropic secret (`EVAL_ANTHROPIC_SECRET`) is always mounted; set `EVAL_OPENROUTER_SECRET` to run
`--judge-profile sol`.

## Stages

`all` runs `[prepare, corpus]` → `[capture]` → `[cells_mean]` → `[rollouts, patchscope, lens, nla,
retrieval]` → `[rollouts_merge, nla_merge, retrieval_merge, summarise]` → `[judge]` → `[report]`. A stage
resumes when its key — its resolved settings chained to the records of the stages it reads — matches, so a
stage run again brings back exactly what is downstream of it.

| Stage | Compute | Outputs |
|---|---|---|
| `prepare` | CPU | `data/items.json` |
| `corpus` | CPU | `retrieval/corpus.npz`, `corpus.json` |
| `capture` | GPU (base) | `activations/h42.npz`, `cells/<i>.npz`, `cell_table.json`, `norms.json`, `data/compliance.json` |
| `cells_mean` | CPU | `activations/mean42.npz`, `mean_table.json`, `mean_norms.json` |
| `rollouts` | GPU (base + inverter), per arm and shard | `rollouts/<arm>.json` with the injection check |
| `rollouts_merge` | CPU | `rollouts/<arm>.json` with the run-level check |
| `patchscope` | GPU (base) | `rollouts/patchscope.json` (`patch42`, `patchfloor`) |
| `lens` | GPU (base) | `lens/lens.json`, `ranks.npz`, `top10_by_layer.jsonl` |
| `nla` / `nla_merge` | GPU (verbalizer) / CPU | `rollouts/nla.json` |
| `retrieval` / `retrieval_merge` | GPU (base), per window shard / CPU | `retrieval/scores.part<k>of<n>.npz` / `retrieval/windows.json` |
| `summarise` | API | `judges/summaries.json` (layer 42), `judges/summaries_band8.json` (eight-layer pool) |
| `judge` | API | `judges/<judge>/naming.jsonl`, `foils.json`, `gate.json`, `judges/ledger.json` |
| `report` | CPU | `scores/items.jsonl`, `tables/`, `figures/`, `report.md` |

## Outputs

Tables are described in methodology §5.1 (`rates`, `judged`, `modulation`, `contrasts`, `missingness`, the
two paper tables, `coverage_and_costs.json`). Figures: `01_final_period` and `02_modulation` (headline),
`03_carrier_mean` (Appendix A), `04_lens_protocol` (Appendix B), `05_control_conditions` (Appendix C), as PDF
and 300-DPI PNG. `report` calls no model and re-renders identically from the saved artifacts (PDFs differ only
in their creation date).
