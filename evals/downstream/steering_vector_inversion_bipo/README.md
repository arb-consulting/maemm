# Persona steering vectors, read through MAEM

A steering vector is supposed to carry a behaviour. This eval asks whether a *reader* of the residual stream
can say **which** behaviour, from the vector alone.

For each of 119 persona behaviours from Anthropic's model-written evals, a steering vector is trained with
BiPO at layer 42 of Qwen3.6-27B and handed to each reader as a unit direction. A judge (GPT-5.6 Sol) sees
one bundle of what a reader made of the direction -- eight of its generations, the eight corpus windows a
search returned, or the one summary of what the Jacobian lens reads -- and ten candidate behaviour
descriptions, and picks one. Chance is 10 %. This is the BiPO column of the paper's main steering table.

`evals/downstream/steering_vector_inversion/` (the AxBench package) is used as a library: its model worker, job
builders, identification prompt and parser, and the paper-column writer. `methodology.md` is the protocol.

## Reproducing the paper

```
modal run evals/downstream/modal_steering_vector_inversion_bipo.py --stage all --run-id <run-id>
```

That is the paper's run: all stages, judge profile `sol` (GPT-5.6 Sol through OpenRouter, the default).
It needs a Modal account, the Modal secret named by `EVAL_OPENROUTER_SECRET` (default `maem-openrouter`)
holding `OPENROUTER_API_KEY`, and the environment of `evals/downstream/modal.env.example`. No Hugging Face token is
needed: every checkpoint is a public download.

Approximate cost, from the run the paper reports:
- **GPU:** about 13 GPU-hours, most of it the BiPO training (about 10 B200-hours for 119 behaviours; the
  rest is the rollouts, the NLA verbalizer, the steered model and the corpus search). Roughly US$60-80 at
  Modal list prices, and 2-3 hours wall-clock with 8 containers (`EVAL_GPU_WORKERS`).
- **Judge:** about 9,800 identification requests and 119 lens summaries, about US$16 under Sol. The run's
  ledger is capped at `config.JUDGE_CAP_USD` (US$50) and prints each stage's projected spend first.

The paper's BiPO column and the BiPO half of the appendix strength table are built from this run's
`tables/mc10_summary.csv`, `tables/mc10_cells.csv` and `tables/plain_steered_health.csv`, together with the
AxBench run's tables, by `python -m evals.downstream.analysis.paper steering --axbench-run <AxBench run> --bipo-run
<this run> --out <dir>`. `report` also writes the run's own rendering of the column,
`tables/paper_main_column.{tex,csv,md}`.

The run of record and the tag that produced it are listed in `evals/downstream/analysis/paper/README.md`; this code
reruns it from scratch rather than resuming it. Every arm draws the same texts from the same seeds and pins,
with one difference: the steered model at s = 0.5 is the family `<vector>|plain_steered@0.5` here and was
`<vector>|plain_steered` at the tag. Its bundles are shuffled by a seed drawn from the family id, so its
texts are the same but they are grouped into different bundles, and its appendix cell will not reproduce
exactly.

Without Modal, on a CUDA host with `OPENROUTER_API_KEY` set,
`python -m evals.downstream.steering_vector_inversion_bipo all --run-id <run-id>` runs the same stages with one model in
the controller process. `--judge-profile sonnet` asks Claude Sonnet 5 through the Anthropic API instead
(`ANTHROPIC_API_KEY`; on Modal, set `EVAL_ANTHROPIC_SECRET` to the secret that holds it). A run directory
holds one profile.

Install: `pip install -r evals/downstream/common/requirements.txt -r evals/downstream/steering_vector_inversion/requirements.txt
-r evals/downstream/steering_vector_inversion_bipo/requirements.txt`, torch first from the wheel index for your CUDA.

## The readers

Every reader is given the same unit direction.

| Arm | What its text is |
|---|---|
| `maem` | MAEM (the inverter) reading the direction through its trained prompt, 64 samples |
| `nla_native` | the released NLA verbalizer, up to 200 new tokens, 64 samples |
| `jlens` | the Jacobian lens's ten word-like tokens at layer 42, summarised in prose by the judge model |
| `retrieval` | the eight best non-overlapping 64-token windows of the shared 10M-token held-out corpus |
| `plain_steered@<s>` | reference: the plain base with `s x 84.49 x unit(v)` added at layer 42, from a sink token, at s = 0, 0.25, 0.5, 1, 2 (the table reads s = 1), 64 samples each |
| `heldout_matching` | reference: held-out statements the persona would make |
| `base_l1` | control: the untrained base under MAEM's prompt and injection |
| `shuffled` | control: MAEM's texts for another behaviour on the vector's list, scored against this one |

A generated reader's 64 samples are read as eight bundles of eight; a lookup reader (search, lens) has one
bundle. Every rate is a mean over vectors, with a bootstrap interval over vectors.

## Stages, in the order `all` runs them

| Stage | Where | What |
|---|---|---|
| `items` | CPU | download each persona file at a pinned commit and freeze its statement pairs |
| `corpus` | CPU | rebuild the shared held-out search corpus from two public parquet files |
| `train` | GPU (B200) | BiPO, one vector per behaviour, checkpoint at epoch 20 |
| `rollouts` | GPU (base + inverter) | MAEM and the untrained-base control |
| `nla` | GPU (verbalizer) | the NLA verbalizer |
| `plain-steer` | GPU | the steered model at every strength, and its text health |
| `retrieval`, `retrieval-merge` | GPU, CPU | the corpus search in `config.RETRIEVAL_SHARDS` blocks, then the merge |
| `lens` | GPU + judge | the lens read, then one summary per direction |
| `controls`, `gate` | judge | held-out statements and the base control first; stop unless held-out text is identified in 80 % of bundles |
| `mc10` | judge | the ten-way identification of every arm |
| `report` | CPU | the tables, `report.md` and the paper column |

`describe` (Opus 5, opt-in) regenerates the persona descriptions; see below.

The `corpus` stage runs on the controller and downloads the two pinned Ultra-FineWeb parquet files (about
2.6 GB) into its Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`), shared with every other run
and with the AxBench package. `retrieval/corpus.json` records the corpus's identity (dataset revision, file
names, the held-out index's sha256), and the build checks every document's token count against that index. Only the tokenizer's small files
are fetched into the run's own `cache/`.

`--smoke` runs every stage over the first persona and a short prefix of the corpus. `--behaviours` restricts
a pass to some behaviours, `--family-filter` a judged pass to some families, and `--shard k/n` re-runs one
block of the corpus search. Every GPU batch is cached by content and every judge request resumes from its
log, so a repeated stage costs nothing.

## The persona descriptions

The ten candidates of every question are drawn from `assets/descriptions.json`: 238 sentences, a `+` pole
(what the persona would say) and a `-` pole for each of the 119 personas. A file's name is not reliably what
its statements say, so each pair was written from the data: Claude Opus 5 (`config.DESCRIBE_JUDGE`, through
the Anthropic API, thinking off) was shown 30 of the persona's training statements per side, drawn by seed
and never the held-out ones, and never the file name, and asked for a label-colon-gloss sentence per pole
under rules `persona.description_error` checks (12-30 words, no subject, no "AI", "model", "assistant",
"statement" or "persona"). The asset records its `writer`, `n_statements`, per-persona `sources` (the
upstream file's sha256 and the training file's spec digest), a `spec_digest` over the prompts, rules and
sources, and a `digest` over the sentences, which every run checks.

The `describe` stage regenerates it. It is opt-in, never part of `all`, and needs `ANTHROPIC_API_KEY`
(through Modal, set `EVAL_ANTHROPIC_SECRET` to the secret that holds it: the launcher mounts that secret
only when the variable is set). Use a directory of its own:

```
python -m evals.downstream.steering_vector_inversion_bipo describe --run-id describe --rewrite --freeze
```

It builds the training files, asks the writer, and writes `items/descriptions.json`. Without `--rewrite`
it starts from the shipped asset and asks only for personas the asset lacks, so on a clean checkout it asks
nothing. A refusal or an empty reply is asked again, up to three times; a sentence that breaks a rule is
re-asked once with the reason; a persona still undescribed stops the stage, naming it, with nothing
written. `--freeze` then writes the sentences into the asset (without `--rewrite` it only adds, and refuses
to change a sentence the asset holds). About 119 requests, under US$5. LLM text is not reproducible, so a
rewrite gives different sentences, and every rate is taken against the ones the asset holds.

## Tables

| File | What |
|---|---|
| `tables/mc10_summary.csv` | per arm the mean rate over vectors with its interval; MAEM minus each arm, paired |
| `tables/mc10_cells.csv` | per vector, arm and judge: bundles correct, Wilson interval, opposite-pole choices |
| `tables/mc10_arms.csv` | per arm: per-vector rates pooled, min and max |
| `tables/mc10_paired.csv` | readers against each other, vector by vector, with sign counts |
| `tables/mc10_shuffled.csv` | the shuffled check: the vector's own truth, the donor's description |
| `tables/mc10_missingness.csv` | per judge and arm: requests and how each ended |
| `tables/plain_steered_health.csv` | the steered model's text health per vector and strength |
| `tables/coverage_and_costs.csv` | requests and spend per judge and instrument |
| `tables/paper_main_column.{tex,csv,md}` | the paper's BiPO column with Holm-corrected p-values |

## Layout of a run directory

```
items/       each persona's frozen statement pairs
sources/     the same training sets, one file per behaviour
vectors/     the training plan and summary, the bank's metadata
retrieval/   the corpus's token ids and metadata, the queries, one part per block, search.json
lens/        the ten word-like tokens per direction, and the summary each became
rollouts/    families.json: every reader's texts for every vector
scores/      truth.json, mc10_main.json, gate.json
judges/      one request log per judge and instrument, and the run's budget ledger
stages/      the corpus build and each block of the corpus forward, for the resume
tables/      every CSV above
report.md    the report
```
