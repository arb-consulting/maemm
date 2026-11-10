# Paper builders

Rebuild the paper's tables, figures and in-text numbers from finished evaluation runs. The builders read
only what a run has already saved. They call no model and no judge, and they write only under `--out`.

```
python -m eval.analysis.paper steering  --axbench-run A --bipo-run B --out OUT   # [--judge sol] [--method-label MAEM]
python -m eval.analysis.paper workspace --wu-run U --wm-run M --out OUT
python -m eval.analysis.paper coherence --run R --out OUT
```

Each call writes `OUT/<section>/`: the LaTeX fragments and figures below, plus `numbers.csv` / `numbers.md`.
Every in-text number of the section is a row there, with its 95 % interval and the table it came from.
Requirements: `numpy` and `matplotlib`.

## Runs of record

Paths are relative to `eval/out/`, the directory each package's launcher downloads to.

| Section | Package | Run | Judge profile |
|---|---|---|---|
| Steering | `steering_vector_inversion` | `steering_vector_inversion/dev-a-sol-2` | `sol` (GPT-5.6 Sol) |
| Steering | `steering_vector_inversion_bipo` | `steering_vector_inversion_bipo/dev-a-sol-2` | `sol` |
| Workspace | `workspace_understanding` | `workspace_understanding/dev-a-band8` | `sonnet` (Claude Sonnet 5) |
| Workspace | `workspace_modulation` | `workspace_modulation/dev-a-band8` | `sonnet` |
| Coherence | `rollout_coherence` | `rollout_coherence/dev-a-full` | `sonnet` |

The paper's numbers were produced by the code at tag `results-of-record-2026-09-26`.

## Paper element → builder → inputs

The inputs below are what a run of the current packages writes, smoke runs included. The runs of record
use an older layout for a few of them; the builders read both, as the notes at the end describe.

| Paper element | Builder | Output | Run tables read |
|---|---|---|---|
| Table `tab:steer` (AxBench and BiPO columns, bold rule) | `steering.build` | `steering_table.tex`, `steering_tests.csv` | each run's `paper_main_column.csv`; for the tests the bold rule and the text add, AxBench `scores/identification_cases.json` and `data/prepared.json`, BiPO `mc10_cells.csv` |
| Steering in-text rates, intervals, the 1-text and 8-text budgets, significance claims | `steering.build` | `numbers.csv` | as above, plus AxBench `identification.csv` |
| Mean tokens per text (MAEMM against the NLA) | `steering.mean_lengths` | `numbers.csv` | AxBench `text_length.csv` |
| Appendix: AxBench by genre, steered-model strength curve | `steering.genre_tex`, `steering.strength_tex` | `steering_axbench_genre.tex`, `steering_strength.tex` | AxBench `identification.csv`, BiPO `mc10_summary.csv`, both runs' `plain_steered_health.csv` |
| Appendix: median pairwise cosine of the AxBench text-concept vectors (5th–95th percentile) | `steering.axbench_vector_cosines` | `numbers.csv` | AxBench `vectors/bank.{json,npz}` |
| Appendix: persona training pairs, corpus cosine of learned vs random directions | `steering.bipo_appendix_numbers` | `numbers.csv` | BiPO `items/train_persona:*.json`, `retrieval/search.json` |
| Figure `fig:steer` (AxBench budget curve) | `steering_figure.build` | `axbench_text_budget_curve.{pdf,png}` | AxBench `identification.csv` |
| Tables `tab:workspace-word`, `tab:workspace-judge` | `workspace.build` | `workspace_word_rule.tex`, `workspace_judged_net.tex` | each run's `paper_workspace_{word_rule,judged_net}.csv` |
| Workspace in-text rates, the focus / mention split | `workspace.build` | `numbers.csv` | the same, plus the modulation run's `rates.csv` |
| Figure `fig:coherence` | `coherence.figure` | `coherence_two_plots.{pdf,png}` | `frontier_matched.csv` |
| Coherence in-text numbers | `coherence.build` | `numbers.csv` | `frontier_matched.csv`, `frontier_outcomes.csv` |

## Notes

- **One producer per statistic.** The steering table's cells and MAEMM's paired tests are each package's
  `tables/paper_main_column.csv`. The builder adds the bold rule: each column's best reader is bold, and
  so is every reader it does not beat at Holm-corrected p < 0.05. That rule needs the best reader's tests.
  The text also quotes the NLA's tests. Both come from the package's own `paper.marks`. Figure `fig:steer`
  is drawn in the paper's style from the `identification.csv` rows the package's own budget figure reads.
- **Smoke runs.** The coherence builder takes the retrieval curve's corpus sizes from the table
  (`1M`–`10M` on a full run, `half`/`all` on a smoke run); the steering and workspace builders read the
  same files from a smoke run as from a full one.
- **Runs of record.** They predate four outputs of the current packages, so the builders fall back:
  - the BiPO run has no `paper_main_column.csv`; the package's own writer produces it from the run's
    `mc10_*.csv` under `OUT/steering/package_column/`;
  - the steered model's AxBench rows are read from `identification_plain_steered.csv`, and its s = 0.5 arm
    is named `plain_steered`;
  - mean text length is counted from `rollouts/{rollouts,nla}.json` (about 1 GB; `--skip-lengths` skips
    it);
  - MAEMM's centred cosine past k = 8 is blank in `frontier_matched.csv`, so `coherence.centred_best_of_k`
    fills it from `frontier/context/scores/maemm.jsonl` with the package's best-of-k functions. The
    `check.*` rows compare the same computation at k = 1–8 with the table.
