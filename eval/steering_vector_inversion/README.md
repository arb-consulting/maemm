# Steering-vector inversion (AxBench Concept500)

Can MAEMM tell which concept a steering vector stands for? For each of AxBench Concept500's 500 concepts,
a difference-of-means direction at layer 42 of Qwen3.6-27B is handed to MAEMM and to three other readers
(the NLA verbalizer, the Jacobian lens, a corpus search), and an LLM judge picks the concept from ten
candidates given 1, 2, 4 or 8 of each reader's texts. The steered model and held-out concept texts are
references, the untrained base and a shuffled MAEMM text controls. [methodology.md](methodology.md) is
the protocol; this file is how to run it.

This package produces the AxBench column of the paper's main steering table (Table 1), its budget figure
(Figure 3), the in-text statistics (one-text accuracy, mean text length) and the AxBench half of the
appendix strength table.

## Reproducing the paper

```bash
export PYTHONPATH=$PWD                     # from the repository root
modal run eval/modal_steering_vector_inversion.py --stage all --run-id <run-id>
```

`all` runs every stage below under the judge profile `sol` (GPT-5.6 Sol alone, the paper's judge for both
steering packages; the default). It needs a Modal account and an OpenRouter key in the Modal secret
`$EVAL_OPENROUTER_SECRET` (default `maemm-openrouter`); the models and data are public downloads.

Approximate cost: about 12 GPU-hours (the NLA verbalizer about 5.5, MAEMM and the untrained base about 3,
the corpus search about 2.5, the steered model about 1; B200 for the stages that hold the base and the
inverter, H200 otherwise), about 3 hours of wall time at eight containers, and about US$20 of Sol
(20,500 identification requests and 500 lens summaries; the ledgers cap it at US$103). `report` then
writes the run's tables under `eval/out/steering_vector_inversion/<run-id>/tables/`. The paper's steering
table, Figure 3 and the appendix's AxBench and strength tables are built from them, together with the BiPO
run's, by

```bash
python -m eval.analysis.paper steering --axbench-run <this run> --bipo-run <BiPO run> --out <dir>
```

| Paper element | Run tables it reads |
|---|---|
| Table 1, AxBench column (text concepts, 8 texts); Figure 3 | `tables/identification.csv`, `scores/identification_cases.json` |
| One-text accuracies, every arm, budget and genre | `tables/identification.csv` (`metric=identification_accuracy`) |
| Mean text length per reader (MAEMM vs NLA) | `rollouts/rollouts.json`, `rollouts/nla.json` |
| Appendix strength table (s = 0, 0.25, 0.5, 1, 2; 1 and 8 texts; degenerate share) | `tables/identification.csv` (`condition=plain_steered@<s>`), `tables/plain_steered_health.csv` |

The run of record and the tag that produced it are listed in `eval/analysis/paper/README.md`. Its layout
and cache keys differ from this code's, so this code reruns from scratch rather than resuming it. Every arm
draws the same texts from the same seeds and pins and is asked the same questions, with one change: the
steered model at s = 0.5, a 64-sample arm with a greedy decode at the tag, is a 16-sample rung here like
0, 0.25 and 2. The appendix reads its first 1 and 8 texts, which are the same texts.

## Install and local runs

Python 3.12, from the repository root:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128   # GPU host only
python -m pip install -r eval/common/requirements.txt -r eval/steering_vector_inversion/requirements.txt
export PYTHONPATH=$PWD
```

Install `modal` locally for the launcher. Without Modal, a CUDA host with up to eight cards runs the same
stages (`OPENROUTER_API_KEY` set):

```bash
python -m eval.steering_vector_inversion all --run-id <run-id>              # the whole evaluation
python -m eval.steering_vector_inversion all --run-id <run-id> --smoke 4    # 4 concepts per genre, short corpus
python -m eval.steering_vector_inversion report --run-id <run-id>           # re-render on CPU, no API or GPU
python -m eval.steering_vector_inversion retrieval --run-id <run-id> --shard 0/8   # ... 7/8, then:
python -m eval.steering_vector_inversion retrieval --run-id <run-id> --shard 0/8 --merge
```

`--run-id` names a directory under `eval/out/steering_vector_inversion/`; `--output-dir` is an exact path.
`--judge-profile sonnet` runs Claude Sonnet 5 instead (needs `ANTHROPIC_API_KEY`, or on Modal the secret
named by `EVAL_ANTHROPIC_SECRET`); a run directory keeps the profile it was started under. Launcher
settings (`EVAL_GPU`, `EVAL_GPU_BOTH_MODELS`, `EVAL_GPU_WORKERS`, the cache volume and secrets) are in
`eval/modal.env.example`.

## Stages

Every stage resumes from what is on disk. GPU work is cached per batch and judge work per request, under
keys that hash the run's configuration, the inputs and the bytes of every source the operation runs
through (`artifacts.source_hashes`), so a changed pin or implementation misses rather than reusing a stale
result.

| Stage | Work | Models |
|---|---|---|
| `prepare` | Download the pinned Concept500 split; construction texts and held-out references | — |
| `corpus` | Rebuild the shared corpus (`retrieval/corpus.{npz,json}`) on the controller | — (CPU) |
| `preflight` | Hook arithmetic on the real checkpoints and a clean-read consistency series | base + inverter |
| `vectors` | Construction reads, the direction bank, candidates, the target set | base |
| `retrieval` | Each direction's eight best corpus windows (`--shard k/n` splits it) | base |
| `rollouts` | MAEMM (targets and shuffled donors) and the untrained base | base + inverter |
| `nla` | The NLA verbalizer's explanations | verbalizer |
| `plain-steer` | The steered model at every strength of the curve | base |
| `lens` | The lens's ten tokens per direction | base |
| `summarise` | The judge model writes each token list as prose (ledger `summaries`, cap US$3) | — (API) |
| `identify` | Every identification case (ledger `identification`, cap US$100) | — (API) |
| `report` | Tables, figures, `report.md` from the saved scores | — (CPU) |

The `corpus` stage downloads the two pinned Ultra-FineWeb parquet files (about 2.6 GB) into the
controller's Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`), once for every run of both
steering packages; set `HF_HOME` to keep it elsewhere. The run records the corpus's identity in
`retrieval/corpus.json` (dataset revision, file names, the held-out index's sha256), and the build checks
every document's token count against that index. The AxBench concept parquet (`prepare`, about 20 MB) goes
to the same cache; only the tokenizer's small files are fetched into the run's own `cache/`.

A request a ledger refuses, or one whose transport or key fails, leaves a case unasked; the stage fails
and `report` refuses the run until a rerun has asked it (a rerun asks nothing else).

## Outputs

```text
eval/out/steering_vector_inversion/<run-id>/
  config.json, provenance.json   # resolved settings and pins; commit, versions, stage records
  data/ vectors/ retrieval/ lens/ rollouts/   # prepared texts, bank, corpus and windows, lens, generations
  judges/                        # one append-only log per judge per instrument, one ledger per instrument
  scores/                        # sources (every arm's texts), identification cases and per-concept rows
  tables/ figures/ report.md
```

`identification.csv`, `paired_differences.csv` (MAEMM minus each arm) and `text_length.csv` share the
columns `metric, condition, judge, group, budget_type, budget, estimate, ci_lower, ci_upper, n_total,
n_valid, ci_method`; `group` is all/text/code/math, `budget_type` is `snippets` or `greedy`.
`missingness.csv` counts how each case ended per arm; `plain_steered_health.csv` is the steered model's
text health per concept and strength; `coverage_and_costs.json` holds exclusions, spend, the corpus
record and GPU time. `figures/01_identification` is the diagnostic curve for every arm and genre.
`tables/paper_main_column.{tex,csv,md}` and `figures/paper_axbench_budget_curve` are the run's own
rendering of its column and of Figure 3; the paper builder checks its p-values against the column.
