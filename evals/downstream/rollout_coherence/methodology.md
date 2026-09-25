# Rollout coherence: methodology

Given a real layer-42 activation of Qwen3.6-27B, does the text MAEMM writes for it read as coherently and as
fluently as the passage the activation came from, and how does that trade against how well the text
inverts the activation? Population: 400 activations from held-out Ultra-FineWeb documents. Model under
evaluation: the 27B full-parameter inverter. Comparators: the base model continuing each activation's own
source prefix, the released NLA verbalizer, and corpus search. One LLM judge (Claude Sonnet 5 by default)
compares every text with its source passage, blind and in both presentation orders; an independent language
model (Gemma-4-31B) scores fluency.

The paper's figure (`python -m evals.downstream.analysis.paper coherence`) and the package's own
`figures/06b_frontier_matched_centred` plot the same two panels against the centred cosine with the matched
target: (a) the judge's net preference against the source, (b) the log-likelihood gap.

## 1. Research question and limits of the claims

**Question.** For a layer-42 activation read at one position of a web document, where does each way of
turning it into text sit on inversion (how closely the text's own activations reproduce the target) and on
fluency (whether a blind reader rates it as coherent as the source passage, and how probable an independent
model finds it)?

**Unit.** One activation = one document, one position, one target. Every text of an activation shares its
target and its source passage, so activations are the independent unit of every interval.

**What is measured.** A pairwise judgment: the judge sees two passages labelled A and B, is told both are
excerpts that may begin or end mid-sentence, and answers which is more coherent and natural, or that they
are comparable. Every pair is judged in both orders; a pair is a win or a loss only when the judge agrees
with itself, otherwise a tie. Reported: the win / tie / loss split, the share of pairs at least as coherent
as the source, and the net preference (wins minus losses over pairs). Beside it, a judge-free fluency gap
(§6) and the re-read cosine that measures inversion (§4).

**Claims it does not support.** This is not a reconstruction test: the judge is never asked whether a text
matches its source, which is the partner only because it is the fairest natural text of the same length.
One base model, one inverter, one layer, one corpus, 400 activations, one judge. The comparison favours the
generated texts on one point: the passage is cut where the text's length says, so it may end mid-sentence,
while a generated text ends where the model stopped. The inversion axis is MAEMM's training objective and
retrieval's selection rule, so it favours them; that is why the two fluency axes are read beside it. A
matched target is read standalone behind a sink, a position no training target occupied.

**Training overlap.** The documents are index-disjoint from every training range, and each is
content-checked against the checkpoint's training text (`evals/downstream/common/assets/README.md`): a document whose
word 7-gram coverage by the training rows is at least 0.05 is excluded from the draw of §3 (181 of the
3,215 evaluation documents of at least 512 tokens).

## 2. Procedure

1. **Prepare.** Draw 800 documents of at least 512 tokens from the held-out corpus's evaluation half by the
   shared seeded rule, skipping the documents the content check excludes: 400 sources and 400 spares, each
   with a read position uniform in [64, 511].
2. **Capture.** Read the raw layer-42 residual at each position in its document on the clean base; keep
   the first 400 candidates whose raw norm passes the filter of §3.
3. **Targets.** Re-capture each activation's target from its 64-token source tail (§3).
4. **Texts.** Generate and retrieve every method's texts for the targets and re-read each against its
   target (§4).
5. **Pairs and judge.** Each text against its source passage cut to the text's length, judged in both
   slot orders (§5).
6. **Fluency.** Score every judged text and its partner under Gemma-4-31B (§6).
7. **Report** the curves, the tables and the figure (§7).

## 3. Documents, activations and targets

**Corpus.** The held-out corpus (`evals/downstream/common/retrieval.py`): 18,813 Ultra-FineWeb documents (English, a
pinned revision, two public parquet files) in the shipped index's order. Its first 11,700 documents
(9,999,741 tokens) are the search corpus of §4; the documents here come from the 7,113 after them, so a
source passage is never a window retrieval can return. Tokenisation with the Qwen3.6-27B tokenizer, no
special tokens. The draw is a seeded permutation of the qualifying documents (`config.DOC_SEED`) cut into
blocks: 400 sources, then 400 spares; the positions come from one generator (`config.DATA_SEED`), sources
first, then spares, then the norm presample.

**Norm filter.** On the raw in-document norm, at selection, by rejection: a candidate is kept when
`1e-3 < ‖h‖ ≤ 10 ×` the median raw norm of 4,096 (document, position) pairs presampled over the pool's
first 64 documents. The activations are the first 400 kept candidates in pool order; a rejected document is
replaced by the next spare at that spare's own position. Rejections are recorded and reported. `h` is read
over the raw 512 tokens with no sink, as the training activations were collected.

**The mean.** `mu` is the shipped layer-42 centring mean `evals/downstream/common/assets/mu.f32`, sha256-checked on
load, the vector every activation in the suite is centred with.

**Matched targets.** Each activation's target is re-captured from the 64-token source passage ending at
the read position, through the standalone re-read itself (§4): sink prepended, the layer-42 residual at the
text's last content token `h`, and the direction `d = unit(h − mu)`. Re-reading the source passage against
its own target therefore gives a centred cosine of 1; every invocation of `frontier_context` checks it
(tolerance 10⁻³) in each re-read window it uses and stops otherwise. The raw `h` is the verbalizer's input.

**Models.** The clean base `Qwen/Qwen3.6-27B` and the inverter (MAEMM), both at the repository ids and
revisions of `evals/downstream/common/pins.py` (`MODEL`, `INVERTER`), both loaded as causal LMs. Every text is re-read
on the clean base; the inverter only generates.

## 4. The texts and the re-read

| Method | Text | Curve | Compared at |
|---|---|---|---|
| `maemm` | the inverter, `maemm/prompts.py`'s prompt with `d` injected at the marker, block 1, `h ← h + ‖h‖·d` | greedy, best-of-k for k ∈ {1, 2, 4, 8, 16, 32, 64} | k = 1 |
| `retrieval` | the top-1 corpus window per nested corpus size | 1M, 2M, 4M, 8M, 10M tokens | 10M |
| `nla_native` | the NLA verbalizer's explanation at its own length (up to 200 tokens) | greedy, best-of-k for k ≤ 8 | k = 1 |
| `nla` | the same explanations truncated to their first 64 tokens | the same | k = 1 |
| `continuation` *(reference)* | the base continuing the source prefix (tokens 0 to the read position), no injection | a horizontal line at its fluency | — |
| `source` *(reference)* | the 64-token source passage | one point at net 0, gap 0 | — |

**Decoding.** Temperature 1, top-p 1, top-k off, min-p 0, 16 to 64 new tokens, one shared generation config
(checked field by field before the inverter generates). `maemm` draws 64 samples and a greedy
decode per activation, `continuation` 8 and a greedy; seed 1234 plus a per-method offset
(`config.SEED_OFFSET`), each `generate` call of at most 32 rows seeded by the grid index of its first row.

**The verbalizer** (`evals/downstream/common/nla`): the released checkpoint on the matched raw `h`, its own prompt and
chat template, up to 200 new tokens, one greedy and eight samples at temperature 1. A cosine reads the
generation as decoded, tags included; the judge reads the body between the `<explanation>` tags (the whole
text when they do not close). `nla_native` is re-read through a 256-token window, and its mean length is
reported beside it, since the axis is a maximum over positions and grows with length.

**Retrieval.** The search corpus in 64-token windows at stride 16 (595,343 windows), each read as
`[sink] + window` from its own token ids and scored as the maximum raw cosine over its positions against the
matched direction (`evals/downstream/common/retrieval.py`); the sizes are nested prefixes ending on document boundaries,
ranked from one forward pass (sharded, then merged). The search selects the window; the reported cosines are
the standard re-read of its decoded text. A top-1 search cosine above 0.95 is counted as a possible near
duplicate.

**Re-read.** Each text is forwarded alone through the clean base (`evals/downstream/common/scorer.py`): re-tokenised,
cut at 95 tokens (256 for `nla_native`), a sink prepended and excluded, and scored as the largest cosine
over its content positions with `d`, under two geometries: raw `cos(h_tok, d)` and centred
`cos(unit(h_tok − mu), d)`, on which the identical activation scores 1. The figure's x is the centred one.
No norm filter is applied; what the scorer's 10×-median filter would have dropped is tallied per method. A
blank text has no score.

**The k ladder.** Methods are compared at k = 1, one draw with no selection; k = 8 is the sampling budget
every sampled method shares, and only `maemm` continues to k = 64. At one activation with `m` valid
samples ranked by raw cosine (ties to the lower index) and `k' = min(k, m)`, rank `r` is kept with
probability `C(m − r, k' − 1) / C(m, k')`, so k = 1 is the plain mean and k = m the argmax. For k ≤ 8 both
cosines and both fluency axes use these weights over draws 0 to 7, every one of which is judged; for
k ∈ {16, 32, 64} x (raw and centred) uses them over all 64 draws, and the fluency axes read the one
selected text (the highest raw cosine among the first k draws), judged and scored once per activation.

## 5. Pairs, judge and scoring

**Pairs.** Per activation: one pair per corpus size, the greedy and draws 0 to 7 of `maemm`,
`continuation`, `nla` and `nla_native`, and the k = 16, 32 and 64 selections of `maemm`: 44 pairs.
The partner is the source passage cut, in the document's own tokens, to the judged text's re-tokenised
length; a partner that comes out shorter is flagged `window_short` and kept. A blank text is skipped.

**The judge.** The active profile's (`evals/downstream/common/judges.py`): Claude Sonnet 5 by default, GPT-5.6 Sol under
`--judge-profile sol`. Reasoning off, no temperature, a 16-token cap. The prompts are `config.SYSTEM_PROMPT`
and `config.USER_TEMPLATE`, quoted in `evals/downstream/judge_prompts.md`; the judge never sees model names, method
names or scores. Every pair is sent twice: order 0 puts the text in slot A, order 1 in slot B. Requests are
cached by content digest and resumed, so a completed request is never paid for again. A reply is parsed for
a JSON `choice` in {A, B, tie}; an unparseable reply is re-asked once, then scored as a tie for that order.
A refusal (an empty reply, a refusal stop reason or opener, or the provider's moderation) is its own
category, scored as a tie and never re-asked.

| order 0 | order 1 | outcome |
|---|---|---|
| text | text | win |
| partner | partner | loss |
| anything else | | tie |

Per pair, `net ∈ {+1, 0, −1}` and `at_least_as ∈ {1, 0}`.

## 6. Fluency, aggregation and missing data

**Fluency.** `google/gemma-4-31B` (pretrained, a different family from the generators), loaded text-only
in bf16 at a pinned revision. Every text is tokenised standalone with exactly one `<bos>`, no truncation;
special-token spellings inside a text are scored as text. For a text of `T` tokens,
`LL(x) = (1/T) Σ log p(x_t | <bos>, x_<t)` in nats per token; per pair, the gap
`Δ = LL(text) − LL(partner)`, positive when the text is the more probable. Checks recorded and reported: one
`<bos>` per text; identical strings scored once, and a re-score in a differently shaped batch agrees; the
source passages' LL against a sane range (−5 to −2). The scorer's pretraining corpus may contain the source
documents, which would bias the gap against the generated texts; this is stated, not corrected.

**Intervals.** Every axis of a point is a mean over activations (each activation's value its best-of-k
weighted mean), with a percentile bootstrap over activations: 10,000 resamples, seed 0, one resample matrix
for every table. The win / tie / loss cells pool pairs: a cell pooling several pairs per activation uses the
activation bootstrap of the pooled ratio of sums, a one-pair-per-activation share a Wilson 95 % interval,
and every net the bootstrap. All intervals are descriptive.

**Missing data.** A skipped pair is counted as missing for its cell. An unanswered order is a tie under
`net`; `net_dropped` beside it is the net over the pairs whose two orders were answered. Unparseable replies
above 2 % of a group's requests refuse the report; a refused share above 5 % is flagged. A request the
ledger refused, whose transport failed after the client's retries, or that the endpoint turned away
(HTTP 401/402/403 not naming moderation), is not a verdict: the judge stage fails with nothing marked
complete, a re-run asks exactly those requests, and `report` refuses a run whose logs still hold one.

## 7. Tables and figure

`report` renders everything from saved artifacts on CPU and refuses a run directory missing an artifact.
All CSVs share one long format: `metric`, `condition`, `group`, `budget_type`, `budget`, `estimate`,
`ci_lower`, `ci_upper`, `n_total`, `n_valid`, `ci_method`, `judge`, then table-specific columns.

| Output | Contents |
|---|---|
| `figures/06b_frontier_matched_centred` | panel (a) judged net, panel (b) log-likelihood gap, against the centred cosine; one curve per method, 95 % intervals, the greedy decode hollow, `source` a star, `continuation` a horizontal line, the axis maximum (1) a dashed rule |
| `tables/frontier_matched.csv` | every point of every curve (`condition` the point, `group` the method): raw and centred cosine, `net`, `net_dropped`, `ll_gap`, mean tokens, with `order`, `headline` and `reference` columns |
| `tables/frontier_outcomes.csv` | win / tie / loss, `at_least_as`, `net` and `net_dropped` per judged method at its per-sample (draws 0-7) and greedy cells, retrieval at the largest corpus |
| `tables/missingness.csv` | per group, the requests asked and how many came back ok, refused, content-filtered, unparseable or with no verdict |
| `tables/coverage_and_costs.json` | coverage, the checks, requests, tokens and cost, GPU seconds, elapsed time |
| `report.md` | the figure, every point, win / tie / loss, the diagnostics, the limitations and the costs |

## 8. Workload

**Judge.** 44 pairs, 88 requests per activation: 35,200 at 400 activations, about US$40 under `sonnet`
against a US$50 cap. The ledger reserves each request's maximum cost before dispatch, settles on reported
usage, checkpoints every 200 requests, and on resume takes the larger of the file and the request logs.

**GPU.** About 3 B200-hours: the corpus forward (four shards), 400 × 65 generations of `maemm`, the
verbalizer's 400 × 9 explanations, the continuations and the likelihood pass.

## 9. Follow-up not done here

A checkpoint ladder (SFT init, midtraining) as further methods; a second corpus; a human-rated subset to
anchor the judge's tie rate; a longer generation cap.
