# Persona steering vectors, read through MAEM: protocol

## 1. Question

A steering vector is built to carry a behaviour. Given the vector alone -- no prompt, no examples, no
label -- can a reader of the residual stream produce text from which the behaviour can be named and told
apart from other behaviours? The readers compared are MAEM, the NLA verbalizer, the released Jacobian lens,
and a search of a fixed web corpus; the steered model and held-out statements are references, the untrained
base and a shuffled reading are controls.

## 2. Models and layer

Qwen3.6-27B at the pinned revisions of `evals/downstream/common/pins.py`. The clean base is what every vector is trained
against, what the lens and the corpus search read, what the steered model and the untrained-base control
generate on. The inverter (MAEM) is a full-parameter checkpoint in the base's layout that generates under
injection; the two must ship the same generation config, field by field, or the run stops before it starts.
The NLA verbalizer is a third checkpoint, held by a worker that holds nothing else.

Vectors are trained at the output of block 42 (0-based), the layer the inverter reads. MAEM receives a
direction at block 1 as `h += unit(v) * ||h||` at its marker token, at prefill only. Every reader receives
the unit direction.

## 3. Population and data

**Population.** Every persona file of the pinned `anthropics/evals` commit carries a theme in
`assets/personas.json`, labelled before any result was read. A file is eligible unless its theme is
religion, politics or disability (16 files) or it is the one unlabelled file: 119 eligible files, all of
which are studied, in the asset's fixed `order`.

**Training pairs.** Each persona file holds statements the persona would say and statements it would not.
A training pair is a neutral elicitation prompt drawn by seed from eight ("Say something you believe."), a
statement the persona would say, and one it would not. Neither side assents or refuses. A leading negation
is the one surface cue that could stand in for the persona, so its share is measured on both sides and the
sides are matched on it by subsampling when they differ by more than 10 percentage points. Pairs are split
by statement: 200 held out (at most a fifth), at most 700 for training.

## 4. Descriptions

Each persona has two frozen sentences, `+` (the stance of the statements it would say) and `-` (the
opposite), in the form "a short label: a subject-free gloss", 12-30 words each, never using "AI", "model",
"assistant", "statement" or "persona". A file's name is not reliably what its statements say, so the
sentences were written from the statements alone: Claude Opus 5 was shown 30 would-say and 30 would-not-say
training statements (drawn by seed), never the file name, and asked for both poles as JSON; a reply that
broke a rule was re-asked once with the error. The 238 sentences ship in `assets/descriptions.json` with
their writer, the spec digest of the training files they were written from, and a digest over the
sentences that every load checks. LLM-written text is not reproducible, so the asset, not a generator, is
what a run reads.

## 5. Vectors

One learned vector per persona, BiPO's bidirectional DPO objective on the training pairs: the parameter is
`theta` and the vector is `v = sigma * theta`, with `sigma` the median response-token norm over the square
root of the width, added at every non-padding position of block 42's output. Twenty epochs, checkpointed at
1, 2, 3, 5, 10, 15 and 20; the vector read is the epoch-20 checkpoint. The loss is BiPO's; the hyperparameters
(`config.HP`: learning rate 4e-3 on `theta`, beta 0.1, batch of 4 pairs, 100 warm-up steps) and the
`sigma * theta` step-size parametrisation are this package's, not BiPO's published ones. Other deviations:
layer 42 of 64 (BiPO uses 15 of 32), no system prompt, no gradient clipping.

## 6. Readers

Each generated reader draws 64 samples per vector under per-row seeds derived from the vector id; MAEM, the
untrained-base control and the verbalizer also draw a greedy sample, which no instrument reads.

- **MAEM** (`maem`): the inverter's trained research prompt with the direction injected, 64 new tokens.
- **Untrained base** (`base_l1`): the same prompt, marker and injection on the clean base.
- **NLA verbalizer** (`nla_native`): the checkpoint `evals/downstream/common/pins.py` pins (`NLA_REPO`), its sidecar prompt through its own chat
  template with thinking disabled, the norm-matched add at its marker on block 1, up to 200 new tokens at
  temperature 1, top-p 1, top-k 0. A judge reads the body between its `<explanation>` tags when they close,
  and the whole text less a leading open tag when they do not; the share that closes is reported and gates
  nothing.
- **Steered model** (`plain_steered@<s>`, reference): the clean base, no chat template, generating 64 new
  tokens from the one token `<|im_end|>` at temperature 1 while `s x 84.49 x unit(v)` is added at block 42's
  output on every position. 84.49 is a typical layer-42 residual norm of the clean base (a median of persona
  training sets' median response-token norms), a constant so a strength means the same in both
  steering packages (`evals/downstream/common/plain_steer.py`). Strengths 0, 0.25, 0.5, 1 and 2, 64 samples each; the
  main table reads s = 1, the strength curve is the appendix's. The text health of every vector at every
  strength (six fixed surface rules for degenerate text, `evals/downstream/common/degeneracy.py`; the unsteered base's
  log-likelihood; early stops; distinct trigrams) is reported and selects nothing.
- **Corpus search** (`retrieval`): the shared held-out corpus of `evals/downstream/common/retrieval.py` (11,700
  documents, 9,999,741 tokens, 595,343 windows of up to 64 tokens at stride 16), each window forwarded on the
  clean base as a sink token plus its token ids and scored by the largest cosine, over its positions,
  between its layer-42 residual and the unit query, uncentred. The reading is the eight best windows that
  share no token with a better one of the same document, best first. Eight seeded Gaussian directions
  (`evals/downstream/common/directions.py`) are queried beside the vectors, for the search's own no-signal cosine in
  `retrieval/search.json`; they are never judged.
- **Jacobian lens** (`jlens`): the ten word-like tokens the released lens promotes at layer 42, which the
  judge model turns into two to four sentences of prose from the token list alone (`evals/downstream/common/lens_summary.py`).
  A summary that is a refusal is no text, so the vector has no J-lens reading and is left out of its rate,
  as in the AxBench package.
- **Held-out statements** (`heldout_matching`, reference): 64 of the persona's held-out would-say
  statements, each cut to 300 characters.
- **Shuffled** (`shuffled`, control): the vector's own question asked of MAEM's texts for a donor, the
  first (in a seeded order) of the other four behaviours on its list whose vector exists. A judge that reads
  the text names the donor, so this rate sits below chance by construction.

## 7. Reading units

A family is one vector under one reader. A generated reader's 64 samples are shuffled by a seed from the
family id into 8 disjoint bundles of 8. The corpus search and the lens have one reading per vector, kept in
the reader's own order as one bundle. A cell's rate is correct bundles over all of the family's bundles.

## 8. Instrument

A judge sees a bundle's texts and ten candidate descriptions and returns the number of the one they express;
the prompt, layout and reply parser are `evals/downstream/steering_vector_inversion/judge.py`'s, sent as a bare user
message with no system turn. The ten are five behaviours with both poles of each -- the target's behaviour
and four others drawn from the 119 -- in an order seeded from the vector's behaviour alone, so every reader
and bundle of a vector answers the identical question. Every candidate has its opposite on the page, so the
list's shape does not single out the target; choosing the target's opposite pole is counted.

No request carries a family id, a vector id, a reader name, a behaviour name, a description id or a pole;
they travel in `meta`, which the transport records and never sends.

## 9. Judge

GPT-5.6 Sol (`openai/gpt-5.6-sol`) through OpenRouter, provider pinned with no fallbacks, reasoning off, a
128-token reply cap (`evals/downstream/common/judges.py`, profile `sol`). It also writes the lens summaries. Every
record keeps the model and provider that answered. A bundle the judge did not answer counts as no
identification; the missingness table says per reader how many were refused, filtered, unparseable or
truncated. A request that was never asked (budget cap, exhausted transport, rejected key) fails the stage,
and a later pass asks exactly those.

## 10. Statistics

A reader's row is the mean over per-vector rates, with a percentile bootstrap over vectors (10,000
resamples, seed 0); an interval needs five vectors. Reader contrasts are paired over the vectors both hold,
with sign counts beside the mean. The paper column tests MAEM against each non-reference row by a paired
two-sided sign-flip permutation test over vectors (20,000 flips, seed 0, zero differences unsigned) with
Holm's correction within the column; references are not tested.

## 11. Stopping rule

The held-out statements and the untrained base are judged first. Held-out statements must be identified in
at least 80 % of the bundles the judge answered, or the run stops before the other readers are paid for.

## 12. Reproducibility

One run directory; every GPU batch cached by its content and by the code that built it; every judge request
resumed from its log by request key. The corpus is rebuilt from two public parquet files by a shipped,
digest-checked document index, each document's token count checked against the index. Each stage records
the git commit and whether the tree was dirty.
