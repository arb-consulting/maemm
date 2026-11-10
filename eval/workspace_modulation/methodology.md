# Workspace modulation: methodology

When a model is instructed to hold a concept in mind while it copies an unrelated sentence, does a MAEMM
rollout read that concept from the residual stream at the position where the sentence ends, and how does its
rate compare with the released Jacobian lens, a natural-language autoencoder (NLA), Patchscopes and a search
of the held-out corpus read at the same activation? This is the "directed modulation" protocol of the
workspace paper (*Verbalizable Representations Form a Global Workspace in Language Models*, Anthropic), run
on its released materials. This file is the one description of the design; the README gives the commands.

## 1. Question and scope

**Question.** Given the layer-42 residual at the final period of a sentence the model is copying under an
instruction to think about, ignore, or not think about a concept, does the inverter generate text that names
that concept, above the no-instruction baseline, above a null direction and above a 20-donor target-shuffle
chance line, and how does its rate compare with the other readers at the same activation?

**Unit.** One item = one concept under one instruction condition on one carrier sentence. Every reader is
read at one activation per item and read position. Concepts are the pairing unit for instruction contrasts
(every concept appears under all five conditions on the same carrier); items are the pairing unit for reader
contrasts.

**Headline scope.** The claims are read at the carrier's final period, under `focus` and `mention` against
the no-instruction `baseline`, per family (topics and arithmetic are never pooled). The paper's Modulation
column pools `focus` and `mention` over the topic concepts (§5.3). Appendices: **A** the carrier mean
(§2.3), **B** the J-lens paper's own any-token protocol (lens only), **C** the dismissal instructions
`ignore` and `dont_think` as a control, **D** the untrained-base ablation.

The arithmetic word rule is not identification evidence: a readout doing arithmetic states many numbers, so
it sits at its chance line; that family is read on the judged column.

**Does not support.** Whether a dismissed concept is still represented (the control conditions are only a
check on the readers). Anything about a global workspace or consciousness. A bare mention primes much of
what an instruction to focus does, so a positive result reads "MAEMM reads a concept the prompt made
salient", not "MAEMM reads what the model chose to think about". One base model, one inverter, one read layer,
46 concepts drawn and 45 kept on 20 carriers, one generation seed.

**Training overlap.** The materials are synthetic, authored by Anthropic and released with the Jacobian lens;
the inverter's banks are built from Ultra-FineWeb web text, and the bank builder asserts that the J-lens
token-direction family is held out of every bank.

## 2. Population

### 2.1 Pins

| Input | Pin |
|---|---|
| Base model | `Qwen/Qwen3.6-27B` (`eval/common/pins.py`, `MODEL` @ `MODEL_REVISION`) |
| Inverter | `eval/common/pins.py`, `INVERTER` @ `INVERTER_REVISION`, a full-parameter checkpoint of the base |
| Released lens | `neuronpedia/jacobian-lens`, revision `0731326edff4ae730ffc5356fe1a4728c748b3a6`, file `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`, sha256 `1718c8c5…fc11e1` |
| NLA verbalizer | `eval/common/pins.py`, `NLA_REPO` @ `NLA_REVISION`, a full bf16 checkpoint with the verbalizer LoRA merged in; reader `eval/common/nla/nla_reader.py` under its default pins |
| Materials | `datasets/directed-modulation.json` from `anthropics/jacobian-lens` @ `581d3986…ba82ba8e`, sha256 `f3478bb7…ad555b92` |
| Centring mean | `eval.common.background.load_centring_mean()` (`eval/common/assets/README.md`) |
| Search corpus | `eval.common.retrieval.CorpusSpec()`, nothing overridden: the held-out corpus's search prefix of `openbmb/Ultra-FineWeb` (revision `02c85641…`), 11,700 documents, 9,999,741 tokens, 595,343 windows of up to 64 tokens at stride 16 |
| Judge | the active profile of `eval/common/judges.py`: `sonnet` (Claude Sonnet 5, the default and the paper's) or `sol` (GPT-5.6 Sol) |
| Prompts | the shared naming prompt (`eval/common/naming.py`) and lens-summariser prompt (`eval/common/lens_summary.py`), hashed into the stage keys |

Every hash is re-verified at load.

### 2.2 Items

One permutation of the 20 carriers (seed 5300) gives concept *k* the carrier `perm[k mod 20]`; one draw per
concept and instruction group (seed 5301) picks the phrasing. The user turn is

> Write "{carrier}" {phrasing with `{x}` filled, first letter capitalised} Don't write anything else.

and `Write "{carrier}" Don't write anything else.` for the baseline. `{x}` is the topic category name or
`evaluating {expression}`. Phrasing groups map to conditions: `focus` → focus, `dismissal` → ignore,
`negated-think` → dont_think, `mention` → mention. The assistant turn is the carrier, teacher-forced with
thinking disabled (an empty think block precedes it); the rendering is asserted.

| Condition | Example user turn (concept "citrus fruits") |
|---|---|
| focus | Write "The old painting hung crookedly on the wall." Concentrate on citrus fruits while you write the sentence. Don't write anything else. |
| ignore | … Citrus fruits is irrelevant — ignore it. Don't write anything else. |
| dont_think | … Whatever you do, do not think about citrus fruits. Don't write anything else. |
| mention | … (Citrus fruits) Don't write anything else. |
| baseline | Write "The old painting hung crookedly on the wall." Don't write anything else. |

**Forms** (what the word rule matches): for a topic every member, whole word, case-folded, singular and
plural (irregulars listed in `items.IRREGULAR_PLURALS`); the category name is in the prompt and is not a
form. For arithmetic the answer as digits and as its number word.

**Exclusions, before any readout.** A concept whose target appears in its carrier is reassigned the next
carrier; one whose target appears in its own prompt is excluded (echo-safety under every condition); a
carrier under four tokens is reassigned. Every change is recorded.

### 2.3 Read positions

- **`final`**: the layer-42 residual at the carrier's final period. The headline, and the only cell the
  untrained-base ablation is read at.
- **`mean`**: one synthetic cell per item, the mean of the layer-42 residual over the carrier tokens, stored
  raw and uncentred at position `-1` like a token row, so every reader prepares it as it prepares any cell.
  The population's median ratio of the mean's norm to the median carrier-token norm must lie in [0.25, 2.0].
  Averaging keeps what is constant across the sentence (the copying task).
- **`carrier`**: not a read position; the J-lens paper's criterion, a hit at any carrier token, read by the
  lens alone (Appendix B).

The null control and the Patchscopes floor read no activation, so their `mean` readout is their final-period
one; their mean-cell judge requests repeat final-period requests and are answered once.

### 2.4 Chance line, targets and the foil

**Chance.** Twenty same-family donor concepts per concept with forms disjoint from its own (seed 5200). Each
reader's readouts are scored against each donor's forms; the mean is the item's **20-donor target-shuffle
chance line** (it shuffles which concept's targets a readout is scored against, and still reads a real
activation), per reader, band and budget. A rate below 3× its chance line is flagged.

**Targets** (what the judge is asked about): for a topic the category name and the member forms; for
arithmetic the answer as digit and word, never the expression.

**Foil.** Every judged cell is asked twice, against its own targets and against a foil concept's: the first
donor whose forms are not operands of the item's own expression (for topics simply the first donor). The foil
is per concept, shared by every cell, reader and judge; passed-over donors are recorded
(`judges/<judge>/foils.json`).

### 2.5 Compliance diagnostic

The base model's own greedy continuation of each user turn (48 tokens, thinking disabled) is recorded with
whether it copies the carrier and whether it names a target. Reported, never a filter.

## 3. Readers

Activations are read, and every text re-read, on the clean base; the inverter and the verbalizer only
generate. `d = normalize(h_42[cell] − mu)`.

**MAEMM.** `research` prompt, marker ` ?`, norm-matched add at the block-1 output (coefficient 1, prefill
only).

| Arm | Model | Direction | Samples | Role |
|---|---|---|---|---|
| `reg` | inverter | `d` | 1 greedy + 8 | the headline arm, at both read positions |
| `null` | inverter | zero vector | 1 greedy + 8 | the null control: the inverter writes from the unperturbed prompt; final period only |
| `base` | clean base | `d` | 1 greedy + 8 | the untrained-base ablation (Appendix D): word rule only, final period only |

Decoding: temperature 1, top-p 1, top-k 0, min-p 0, 16 to 64 new tokens. Seed 1234 with arm offsets
(`reg` 0, `null` 20, `base` 30, the NLA 7, Patchscopes 40/41), each offset a block of `SHARD_SEEDS`; each
`generate` call of at most 32 rows is seeded `seed · 1000 +` its first row's grid index, and the seed is
stored on every sample. Both models' shipped generation configs must agree, and every arm is cut at the
base's stop ids.

**Injection check** (on `reg`). It guards a dead injection, which leaves the inverter reading what the null
control reads. It gates on one criterion, `injection_alive`: at least 0.9 of `reg`'s greedies differ from
every greedy of `null` (`INJECTION_MIN_DIFFERS_FROM_CONTROL`); a container that holds `reg` without the
control reads the pooled distinct share of `reg`'s greedies instead (≥ 0.25, `INJECTION_MIN_DISTINCT`), and
`rollouts_merge`, which holds every arm, takes the run-level verdict against the control. The record names
its `basis`. The greedies are also re-read on the clean base against their own direction and a donor cell's
(same read position, another concept): own, donor, the paired gap with a 10,000-resample bootstrap interval
and the mean cos(d_own, d_donor) are recorded as diagnostics and gate nothing — at these cells different
items' directions are close to parallel, so the cosine barely separates own from donor. The file is written
before a failed verdict aborts the stage.

**Null control checks.** The directions handed to the generator must be exactly zero, and within each
`generate` call at least 0.9 of the greedies must be one string (identical inputs); nothing is asserted
across calls.

**Jacobian lens.** At every carrier cell and fitted layer (63, including 42): the full-vocabulary rank of
each single-token target form and the ten most likely word-like tokens; at the mean cell, the same at layer
42 only. A self-check recomputes one cell of each of the first three items row by row (own-form logits
within 5e-2, ≥ 8 of 10 top tokens shared).

**Eight-layer lens pool (`jlens_band8`).** The top-10 word-like lists of layers 36, 38, …, 50 at the final
period, pooled into one list (each token once, case-folded, ordered by best within-list position then layer;
`eval/common/lens_io.pool_layers`). A band fixed in advance for every item; final period only.

**Lens summaries.** One model (the active judge's) turns a lens token list into two to four sentences of
prose with the shared prompt, seeing the tokens and nothing else. It summarises the layer-42 list at both
read positions and the eight-layer pool at the final period (`summarise` stage). The judge reads the lens
only through these summaries.

**NLA verbalizer.** The raw, uncentred layer-42 row, injected at the verbalizer's own marker (`h + ‖h‖·v̂`,
block 1) under its actor prompt, no prefill, no stop string; eight samples and a greedy of up to 200 new
tokens at the package's sampling (overriding the checkpoint's shipped top-p 0.95 / top-k 20). The word rule
and the judge read the body between the `<explanation>` tags (the whole text, less a leading open tag, when
they do not close). `nla64` is the same samples cut at 64 generated ids. Guards: every item's greedy
explanations differ across its two cells; the close-tag rate is reported (flagged below 0.8).

**Patchscopes.** Through `eval/common/patchscope.py`: the clean base continues the entity-description prompt
of Ghandeharioun et al. (their appendix D.1), and the placeholder ` x`'s block-42 output is replaced, during
prefill, by `2 · ‖h_42[pos]‖ · unit(h_42[cell] − mu)` (`patch42`); one greedy and eight samples per read
cell under the headline arm's decoding and stop set. A patch check on the first cells requires the patched
residual's cosine to the direction ≥ 0.99 and a relative move > 0.5 before anything is generated. The floor `patchfloor` is the
same prompt with no patch, generated once and carried by every item: `patch42`'s word rule is read against
it, and the floor is not judged. The paper's Patchscopes row is `patch42`.

**Corpus search.** One query per (item, read position), `unit(h_42[cell] − mu)`; a window scores the maximum
over its positions of cos(h_t, q) on the raw residual, forwarded on the clean base behind a sink. The readout
is the eight best non-overlapping windows (a window is kept unless it shares a token with a better kept
window of its document; `eval.common.retrieval.suppress_overlapping`), best first. Deterministic: pass@n is
the n best windows and there is no greedy row. Nothing is excluded from the corpus (this evaluation has no web
text of its own); queries whose best window scores above 0.95 are counted as possible near copies. The
search cosine is also reported over the corpus's nested 8M-token prefix.

## 4. Instruments

### 4.1 The word rule

A target form occurs as a whole word (case-folded) in a sample at the read cell (MAEMM arms, NLA, Patchscopes,
the ablation) or in one of the search's windows; reported as `hit_any` at pass@1, 2, 4, 8, `greedy_hit`,
`consistency` and `chance`. The lens is read two ways:

- **`word10_L42_any`** (condition `jlens_L42`): a target form as a whole word, case-folded, among the ten most
  likely word-like tokens at layer 42 at the read cell. It needs no single-token form. **This is the paper's
  J-lens word-rule row.**
- **`word10_band8_any`** (condition `jlens_band8`): the same rule over the eight-layer pool at the final
  period, with a 20-donor chance line of its own. **This is the paper's J-lens (L36–50) row.**
- `rank10_L42_any` (a single-token form ranks ≤ 10 at layer 42) at both read positions, and, on the carrier
  band only, `rank1_any` / `rank10_any` over every fitted layer (target-informed; Appendix B). A concept with
  no single-token form is a rank miss, kept in the denominator.

### 4.2 The naming instrument

The shared instrument (`eval/common/naming.py`, whose docstring defines it): the judge sees the target forms
and one reader's readout at one cell — numbered samples, or one prose summary for a lens reader — and nothing
else, and answers `{"expressed", "target", "quote"}`. A positive verdict is voided unless its target is one
it was asked about and its quote is verbatim in the readout. Each cell is asked against its own targets and
against the foil's.

Readers: `maemm_reg8`, `maemm_null8`, `nla_n8` (native samples), `retrieval_top8`, `patch42_n8`,
`jlens_L42_summary` and `jlens_band8_summary` (final period only). The ablation and the Patchscopes floor
are not judged. An empty readout is never sent (recorded `empty_readout`). Per item, reader and cell: `named` (own
verdict), `foil`, and `net` = named − foil over the items where both resolved. The fold is one-sided (a lost
reply can cost a credit, never create one), so every judged rate is a floor; the missingness table counts
the lost cells by status (`ok`, `refused`, `content_filter`, `parse_fail`, `unavailable`, `empty_readout`).

Every log record keeps the judged text; a verdict is joined to a cell only when that text is the text the
cell holds now, otherwise the analysis raises naming the cell. A reply cut at its cap can be re-asked at a
larger cap (`--retry-truncated`), as a new request key.

**Gate and budget.** Before its pass the judge answers 20 requests drawn round-robin over (family, reader):
the parse rate must reach 0.80, the share of positive verdicts whose quote verifies 0.60 (from 5 positives),
and the measured per-request cost times the requests still owed must fit what is left of the cap. The
summariser and the judge spend from one ledger. A request never asked (ledger refusal, exhausted transport,
HTTP 401/402/403) leaves the stage unmarked and makes the report refuse the run.

## 5. Tables and statistics

### 5.1 Tables

Long format (`metric`, `condition`, `group`, `budget_type`, `budget`, `estimate`, `ci_lower`, `ci_upper`,
`n_total`, `n_valid`, `ci_method`, then the table's own columns). `group` is `topics` or `arithmetic`;
`instruction` a condition, `focus+mention` for the pooled rows, or empty where the judged/contrast rows pool
all five; `band` is `final`, `mean` or `carrier`; every judged row carries `judge` and `judge_label`.

| Table | Rows |
|---|---|
| `rates.csv` | the word rule per reader, group, instruction and band (`hit_any` per budget, the largest carrying `chance`, `ratio_vs_chance`, `flag_below_3x`; `greedy_hit`; `consistency`; `chance`); the lens rows of §4.1; the `focus+mention` pooled rows |
| `judged.csv` | per reader, group and band, pooled over instructions and inside each, and `focus+mention` pooled: `named` and `foil` (Wilson), `net` (bootstrap), and counts of voided, unresolved and empty readouts |
| `modulation.csv` | instruction contrasts paired by concept (focus − baseline, mention − baseline, focus − mention, and the control rows) for every reader and metric |
| `contrasts.csv` | reader contrasts paired by item (MAEMM − NLA, − NLA-64, − corpus search, − lens, − null control; `patch42-patchfloor`) on the headline criterion and the judged `named` / `net` |
| `paper_workspace_word_rule.{csv,tex}`, `paper_workspace_judged_net.{csv,tex}` | the paper's Modulation column (§5.3) |
| `missingness.csv` | per judge, instrument and arm: requests made and how each settled (the `judge_missingness` block of `coverage_and_costs.json`) |
| `coverage_and_costs.json` | population, coverage, checks, guards, gate, spend, missingness, GPU seconds |

### 5.2 Statistics

Wilson 95 % for a single rate. A paired difference or a mean takes a percentile bootstrap over its pairing
unit (concepts for an instruction contrast, items for a reader contrast), 10,000 resamples, seed 0, one index
matrix per group. The `focus+mention` rows take a bootstrap over concepts (`bootstrap_concept_10000_seed0`),
because each concept contributes two items. No significance thresholds. A missing readout is `None` (never a
zero) and leaves every rate and contrast it would enter. The baseline condition's items rest on fewer distinct
prompts than items (concepts on one carrier render one chat), so its concept-level bootstrap understates that
column's variance.

### 5.3 The paper's Modulation column

Each cell is the topics row at `band = final` under `instruction = focus+mention` (21 concepts, 42 items):
MAEMM `hit_any` pass@8 of `maemm_reg`; NLA `hit_any` of `nla`; J-lens `word10_L42_any`; J-lens (L36–50)
`word10_band8_any`; Patchscopes `hit_any` of `patch42`; corpus search `hit_any` of `retrieval`; and the
judged net of `maemm_reg8`, `nla_n8`, `jlens_L42_summary`, `jlens_band8_summary`, `patch42_n8`,
`retrieval_top8`. The per-condition rates the paper quotes (focus vs mention) are the `focus` and `mention`
rows of `rates.csv`.

## 6. Reading rules

A difference whose interval excludes zero is "supported"; one whose interval includes zero is "no evidence of
a difference", never equality.

| Comparison | Reads as |
|---|---|
| a reader's rate under focus or mention, beside the null control's | above the control with the interval excluding zero: the readout carries the concept, which a pipeline reading no activation does not name |
| focus − baseline, mention − baseline | the instruction moves the concept into the final period, if the rate is also ≥ 3× chance |
| focus − mention | how much of the focus effect a bare mention already produces |
| ignore, dont_think (Appendix C) | a control: a reader at the floor under a dismissal reads what the model holds rather than its context |
| `named` beside `foil` | the judged chance line; a reader whose `net` interval includes zero is not shown to name anything |
| MAEMM − NLA, − corpus search, − lens | which reader surfaces the concept more often at the same activation; the search is bounded by what the corpus holds |
| `patch42` against `patchfloor` | what the patch reads off the activation beyond what the prompt alone makes the model say |
| the lens any-layer criterion (Appendix B) | target-informed and over every token; not comparable with a rate at one activation |
| the ablation (Appendix D) | an ablation, not a method: MAEMM is above it and above the control, or it reads nothing the ablation does not |

Examples are picked by rule: per family, under focus, the 2×2 of MAEMM's word rule against the lens's rank ≤ 10
at layer 42 at the final period, the first two concepts by id per cell.

## 7. Limits

- One base model, one inverter, one read layer, 45 concepts on 20 carriers, one seed: intervals are over
  concepts, never over seeds.
- Positions are matched across readers; layers and output budgets are not.
- The judge reads the lens through prose its own model wrote; the summariser is blind to the item.
- The judge never sees the instruction, so a readout that narrates the instruction is credited like one that
  engages it; the null control and the foil are the floors for that.
- Patchscopes runs one fixed recipe, not tuned on this evaluation, and is read against its own floor.
- The corpus search reads text nobody wrote about this activation.

## 8. Guards

The chat-rendering assertions, the own-prompt exclusion, the donor-pool guard, the mean cell's norm-ratio
guard, the injection check, the null control's two checks, the generation-config agreement check, the
corpus identity check against the shipped index, the retrieval part checks (a part is folded only under the
key the merge resolves), the lens self-check, the verbalizer's per-item distinctness guard, the quote guard,
the empty-readout exclusion, the byte-checked dataset, the judge gate and the unasked-request rule. Each
writes its record before it can abort.
