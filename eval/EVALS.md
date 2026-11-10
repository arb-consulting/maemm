# MAEMM evaluation packages

Five packages under `eval/` measure what MAEMM's generated exemplars say about an activation or a
steering vector, against the same baselines. Each package is one command from public inputs to the tables
and figures its README lists. The paper's tables, figures and in-text numbers are drawn from those runs
by `python -m eval.analysis.paper {steering,workspace,coherence}`;
[`analysis/paper/README.md`](analysis/paper/README.md) maps each paper element to its builder and lists
the runs of record. The rest of `eval/` (`eval/README.md`) is the training-time evaluation suite and is
independent of these packages.

| Package | What it measures | Paper | Judge |
|---|---|---|---|
| [`steering_vector_inversion`](steering_vector_inversion/README.md) | whether a judge can pick an AxBench concept's description out of ten from a reader's texts about its difference-of-means steering vector | "Explaining steering vectors": Table `tab:steer` (AxBench), Figure `fig:steer`, the appendix strength and genre tables | `sol` |
| [`steering_vector_inversion_bipo`](steering_vector_inversion_bipo/README.md) | the same ten-way identification for 119 BiPO-trained persona vectors | "Explaining steering vectors": Table `tab:steer` (BiPO), the appendix strength table | `sol` |
| [`workspace_understanding`](workspace_understanding/README.md) | whether a reader of one layer-42 activation names the unstated concept or bridge entity of the J-lens association and multi-hop prompts | "Interpreting model workspaces": Tables `tab:workspace-word` and `tab:workspace-judge` (association, multi-hop) | `sonnet` |
| [`workspace_modulation`](workspace_modulation/README.md) | the same, on the directed-modulation prompts, where the model is told to focus on or mention a topic | "Interpreting model workspaces": the same two tables (directed modulation) | `sonnet` |
| [`rollout_coherence`](rollout_coherence/README.md) | how coherent and how fluent MAEMM's texts are against the source passage, as a function of how well they invert the activation | "Coherence of MAEMM exemplars": Figure `fig:coherence` (`python -m eval.analysis.paper coherence`) | `sonnet` |

Each README has a "Reproducing the paper" section: the command, the stages it runs, the judge profile and
the approximate API and GPU budget. Each package also has a `methodology.md` that describes its design.
`eval/common/assets/heldout_overlap_build.py` is kept as the provenance of a shipped asset: it needs the
private training text, and nothing imports or runs it.

## Install and run

```bash
pip install -r eval/common/requirements.txt -r eval/<package>/requirements.txt   # BiPO: add steering_vector_inversion's
export PYTHONPATH=$PWD                              # eval and mxf are imported as top-level packages
python -m eval.<package> all --run-id <name>       # on a CUDA host
```

The packages also import `mxf/` (the model config, injection hooks and MAEMM prompt) from this repository.
Use Python 3.12, the version every launcher's image pins. Pass both requirements files to one `pip`
command, so the resolver sees both sets of pins. On a GPU host, install `torch` first from the wheel index
for its CUDA version. `flash-linear-attention` is Linux-only: without it, transformers falls back to a
slower path for the model's linear-attention layers.

On Modal, the same stages run through `eval/modal_<package>.py`: `::run_all --run-id <name>` for the three
activation evals, and `--stage all --run-id <name>` for the two steering evals. The controller needs the
`modal` client (`pip install modal`) and a Modal profile.

`--smoke` runs every stage on a small population. It is part of every stage's key, so a later call on a
smoke directory (`report` included) passes `--smoke` too.

The corpus search rebuilds its corpus from two pinned Ultra-FineWeb parquet files (about 2.6 GB), which
are downloaded into the Hugging Face cache of the machine that builds it: on Modal, the activation evals'
volume cache (`EVAL_HF_HOME`); the steering evals build it on the controller (`HF_HOME`, default
`~/.cache/huggingface`). The download happens once and every later run reuses it. Each run records the
corpus's identity (dataset revision, file names, the held-out index's sha256) in `retrieval/corpus.json`
(`frontier/corpus.json` in rollout_coherence) and checks every document's token count against the index.

## Judges

`eval/common/judges.py` defines two profiles of one judge each, chosen with `--judge-profile`:

| Profile | Judge | Transport | Key |
|---|---|---|---|
| `sonnet` | Claude Sonnet 5 (`claude-sonnet-5`, thinking off) | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `sol` | GPT-5.6 Sol (`openai/gpt-5.6-sol`, reasoning off, provider pinned) | OpenRouter | `OPENROUTER_API_KEY` |

Claude Sonnet 5 judges by default. GPT-5.6 Sol is used where Sonnet refused too many requests to be usable:
the two steering packages, whose judged texts include steered generations. So the steering packages default
to `sol` and the other three to `sonnet`. A run directory records its profile and refuses an invocation
under the other one. Every package writes a missingness table per arm (refused, content filter, parse
failure, unavailable): `tables/missingness.csv`, and BiPO's `tables/mc10_missingness.csv`.

Each package's judge cap is a constant in its `config.py`, sized for the full run under its default
profile with headroom. A stage stops before it would exceed the cap. A request that was never really asked
(refused by the ledger, or failed in transport after the client's retries) fails its stage, and running the
stage again asks exactly those requests. The prompts every judge is sent are quoted in
[`docs/judge_prompts.md`](../docs/judge_prompts.md).

## The shared layer (`eval/common`)

Everything the five packages have in common is defined once in `eval/common`, and no package imports
another (except `steering_vector_inversion_bipo`, which uses `steering_vector_inversion` as a library).

### Model pins

Every checkpoint the suite reads is pinned once, in [`eval/common/pins.py`](common/pins.py): its repository
id and revision there are the defaults, and each is overridden by an environment variable (an empty value
keeps the default). The launchers copy these variables into their containers, and the values in force enter
each run's recorded config and stage keys.

| Model | Pins (defaults in `pins.py`) | Override via |
|---|---|---|
| Base model (`Qwen/Qwen3.6-27B`, from `mxf/config.py`) | `MODEL`, `MODEL_REVISION` | `EVAL_BASE_REPO`, `EVAL_BASE_REVISION` |
| MAEMM (the inverter) | `INVERTER`, `INVERTER_REVISION` | `EVAL_INVERTER_REPO`, `EVAL_INVERTER_REVISION` |
| NLA verbalizer | `NLA_REPO`, `NLA_REVISION` (read by `nla/nla_reader.Pins`) | `EVAL_NLA_REPO`, `EVAL_NLA_REVISION` |
| Jacobian lens (`neuronpedia/jacobian-lens`) | `LENS_REPO`, `LENS_REVISION`; the file, its size and sha256 are fixed | `EVAL_LENS_REPO`, `EVAL_LENS_REVISION` |

| | |
|---|---|
| Model read | the base model (`pins.MODEL` @ `MODEL_REVISION`), residual stream after block 42 (`mxf/config.py`). Every activation is read, and every text re-read and scored, on this clean base, never on MAEMM |
| MAEMM | `pins.INVERTER` @ `INVERTER_REVISION`, a full-parameter fine-tune of the base, loaded as a causal LM (`model_io.load_inverter`). Its prompt is `mxf/prompts.py`; the direction is injected at the prompt's marker token at block 1, scaled by the residual norm there, coefficient 1 |
| What MAEMM is given | an activation `h` as `unit(h − mean)`, with the shipped centring mean `eval/common/assets/mu.f32` (`background.load_centring_mean`); a steering vector as `unit(v)` |
| Sampling | temperature 1, top-p 1, 16 to 64 new tokens; sampled texts plus one greedy (`model_io.py`). A `generate` call carries at most 32 rows, each call seeded by the grid index of its first row |
| NLA verbalizer | `pins.NLA_REPO` @ `NLA_REVISION`, a merged checkpoint loaded as a causal LM, prompted through its released template, up to 200 new tokens (`nla/nla_reader.py`). A judge reads the `<explanation>` body; tables also carry its first 64 tokens, the length MAEMM writes |
| Corpus search | one corpus and one rule (`retrieval.py`): the held-out Ultra-FineWeb corpus's 10M-token search prefix in 64-token windows at stride 16, ranked by the largest cosine of any position's layer-42 residual with the query, the best non-overlapping windows returned. Nested 1M–10M prefixes give the size curve. Evaluation text comes from the documents after the prefix, less those whose word 7-grams overlap the training text (`assets/heldout_overlap.csv`) |
| J-lens | the released Jacobian lens (`lens_io.py`); a lens's top tokens are turned into prose by an item-blind summariser before a judge reads them (`lens_summary.py`) |
| Re-read | every text is scored the same way (`scorer.py`): tokenised on its own, cut at 95 tokens (256 for the verbalizer's full text), read behind a sink token on the clean base, scored as the largest cosine over its content tokens. No norm filter; each stage records what it would have dropped |
| Loading | `model_io.load_base` (reader, untrained-base control), `load_inverter` (generator), `free_model`. The two 27B models together need about 108 GB, so a stage holding both asks for a B200 |
| Intervals | 95 % percentile bootstrap, 10,000 resamples, paired where arms share units (`stats.py`) |
| Run directories | resume bookkeeping, config hashes chained across stages, provenance with the git commit (`runs.py`) |

## Launcher environment

The launchers read their Modal names from the environment; `eval/modal.env.example` is a template with
the defaults filled in. Copy it, edit it, and `set -a; source <copy>; set +a` before `modal run`. A launcher
mounts the judge secret of its package's default profile; to run a package under the other profile, name
that profile's secret as well.

| Variable | Default | What it names |
|---|---|---|
| `EVAL_APP` | `maemm-<package-name-with-dashes>` | the Modal app |
| `EVAL_GPU` | `B200:1` (`H200` for the two steering packages) | the GPU of a stage that holds one model |
| `EVAL_GPU_BOTH_MODELS` | `EVAL_GPU` (`B200` for the two steering packages) | the GPU of a stage that holds the base and a second 27B model |
| `EVAL_GPU_TRAINING` | `B200` | the GPU that trains the BiPO vectors (`steering_vector_inversion_bipo`) |
| `EVAL_GPU_WORKERS` | `8` | GPU containers a launcher holds at once |
| `EVAL_GPU_COST_PER_HOUR` | `6.2496` | the rate the three activation evals' GPU cost estimate uses (Modal's B200 list rate) |
| `EVAL_NLA_SHARDS` | `2` | containers `workspace_understanding`'s NLA generation is split into |
| `EVAL_RETRIEVAL_SHARDS` | `8` (`workspace_understanding`), `4` (`rollout_coherence`) | containers the corpus-search forward is split into |
| `EVAL_VOLUME` | `maemm-data` | the volume the three activation evals' run directories live on |
| `EVAL_HF_SECRET` | `maemm-hf` | the secret carrying `HF_TOKEN` (the three activation evals; the steering evals download anonymously) |
| `EVAL_ANTHROPIC_SECRET` | `maemm-anthropic` where `sonnet` is the default profile, else unset | the secret carrying `ANTHROPIC_API_KEY` (`sonnet`, and BiPO's `describe` stage); mounted only when named |
| `EVAL_OPENROUTER_SECRET` | `maemm-openrouter` where `sol` is the default profile, else unset | the secret carrying `OPENROUTER_API_KEY` (`sol`); mounted only when named |
| `EVAL_MODEL_CACHE_VOLUME` | `maemm-eval-model-cache` | the weights cache of the two steering packages, whose run directories stay on the controller |
| `EVAL_MODEL_CACHE_DIR` | `/cache/models` | where in that volume the steering packages keep their snapshots |
| `EVAL_HF_HOME` | `/data/hf_cache` | the HF cache the three activation evals read models from |
| `EVAL_OUTPUT_DIR` | `/data` | the root the three activation evals put run directories under, as `<root>/<package>/<run-id>` |
| `EVAL_OUTPUT_DIR_<PACKAGE>` | `<EVAL_OUTPUT_DIR>/<package>` | one activation eval's own run-directory root, overriding the shared one (`<PACKAGE>` is `WORKSPACE_UNDERSTANDING`, `WORKSPACE_MODULATION` or `ROLLOUT_COHERENCE`) |
| `EVAL_{BASE,INVERTER,NLA,LENS}_{REPO,REVISION}` | `eval/common/pins.py` | the model pins ("Model pins" above) |

Each launcher copies every one of these variables it reads into every image it builds, so a container
declares the same app, volume, secrets and checkpoints as the launching machine.
