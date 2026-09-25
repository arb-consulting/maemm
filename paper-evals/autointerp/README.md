# autointerp — the Delphi-style SAE autointerp evaluation

The paper's question: **for a held-out SAE feature, do a MAEMM's rollouts describe the feature as
well as max-activating corpus examples do, and does adding rollouts to a small corpus improve the
description?** Measured Delphi's way (Paulo, Mallen, Juang, Belrose 2024): an explainer LLM writes
a description from a set of examples, a scorer LLM uses that description to classify held-out
windows, and the number is the scorer's **balanced accuracy**.

Design: `infra/2026-09-16_autointerp-design.md` in the mimir project, **including its §9
amendments A1-A12**, which override §2-§6 where they conflict and which this code implements.
Verified inputs, paths and the reasons behind them: `infra/2026-09-16_autointerp-facts.md`.
Everything under `autointerp/` imports only `precompute/common.py`, two private helpers of
`precompute/scan.py` (`_Heap`, `_ex`, so a 4M example row is interchangeable with a 16M one) and
`reconstruction/stats.py`'s `Vol`; it writes to none of them.

## Stages

| stage | where | writes | notes |
|---|---|---|---|
| `sae_self` | GPU | `maemms/<base>/<maemm>/scores/<set>__<engine>/sae_self/` | P1: the target feature's PER-TOKEN pre-gate activation on its own 64 rollouts, for the SAE-family targets (`sae`, and `sae2m_enc` for the 2M dictionary — `FAMILIES` in `sae_self.py`/`build.py`). Self-validating against the stored `cos.f16`, `argmax.i16` and SAE CSR |
| `random_pool` | GPU | `base/<base>/sae/<sae>/random_pool/<set>/` | P1 (A5): 2048 random corpus windows encoded for every tested feature, per-token, sparse. Replaces scan's 256-window `_random256` |
| `examples_4m` | GPU | `base/<base>/sae/<sae>/examples_4m/<set>/` | P1 (A3): the C4 arm's OWN top-128 over the 4M nested prefix |
| `build` | CPU | `base/<base>/autointerp/<set>/<date>_build/` | P2: the rendered example sets per arm and the two test draws, one jsonl per feature. Loads no model except the tokenizer |
| `run` | CPU + Anthropic API | `runs/<date>_autointerp-<tag>/{cache,explain,detection,fuzzing,summary}/` | the LLM half: Delphi's explainer, then its detection and fuzzing scorers. `--path sync\|batch`, cached by prompt hash, resumable, projected and capped before each stage |
| `stats.py` | local **or in `chain`** | `autointerp/pilot.md`, `autointerp/results.md` | paired bootstrap CIs over features, per quartile, win fractions against the nulls, distributions, the fire-fraction covariate |
| `chain` | CPU + Anthropic API | `runs/<chain>/STATUS.json`, `pilot.md`, `results.md`, `results-rlI.md`, `costs.json` | the whole remaining sequence in ONE detached call |

## The `chain` stage — run it detached and read one file

`--stage chain` does the whole remaining sequence in a single detached Modal call on a 12 h
timeout, so nothing depends on a local client staying alive:

    wait for examples_docmax → build the 64-feature pilot → run the pilot (sync) →
    ACCEPTANCE CHECKS → build all 512 → primary run (--approved) → build rlI-150 →
    rlI-150 on M and C4+M → stats.py for each → done

It reports through **`/vol/runs/<chain_dir>/STATUS.json`**, rewritten and committed at every stage
boundary, with the running per-run costs and the acceptance report in it. The acceptance gate
aborts the chain — writing the reason into that file — rather than spending the full run's budget
on a broken test set: the floor arm's mean detection balanced accuracy must be inside [0.42, 0.58],
C4 must be filled at 16 on every feature, draw 2 must be non-empty on more than two thirds of
features, and gate-consistent positives must be on. Draw shortfalls are recorded, not fatal. The
API path for the two full runs is chosen by MEASUREMENT: a small Message Batch is timed end to end
and the batch path (half price) is used only if it returned inside 30 minutes.

```
(export MODAL_PROFILE=maemms; uvx --with pyyaml modal run --detach    repo-maemm-precompute/paper-evals/autointerp/modal_app.py --stage chain    --base qwen36-27b --set 2026-09-16_v1    --maemm qwen36-27b/2026-09-10_rl-8x2048-full --maemm2 qwen36-27b/2026-09-08_rlI-150    --chain-dir 2026-09-16_autointerp-chain)
```

**The 2026-09-16 run: app `ap-2H5Xi3kb06MOMS8fq8GRGC`, chain dir `2026-09-16_autointerp-chain2`,
status file `/vol/runs/2026-09-16_autointerp-chain2/STATUS.json`.**

The chain function carries `retries=modal.Retries(max_retries=3)`. MEASURED 2026-09-16: a
container was **preempted** 2176 s into the primary detection stage — *"Container terminated due to
preemption. Your Function will be restarted with the same input"* — and the detached app did not
come back, leaving five Message Batches running server-side with nobody waiting on them. An
automatic retry is safe here only because both stages this function serves are idempotent: `run`
replays completed calls from the prompt cache and **re-attaches** to a submitted batch through its
ledger instead of resubmitting, and `chain` reuses an existing build and continues `STATUS.json`.

**Run `autointerp/selfcheck.py` before every launch.** It drives the real `build` → `run` → `chain`
code against a synthetic volume and a stub client — seconds, no network, no key, no cost — and
exercises the branches that have actually broken launches: `project()` on an empty job list (the
fully-cached branch a relaunch takes) and per job kind, with `gate()` formatting every key it
returns; `gate()` under the threshold, over it, and over it with `--approved`; `run.run()` on both
paths twice each; and `chain.run()` through all nine stages. Four launches died to formatting, key
and control-flow faults that `ruff` and `ast.parse` cannot see — calling a string is valid syntax,
and a missing dict key is a runtime event.

Earlier chains, all stopped, and why — each one is a fault worth not repeating:

| app | stopped because |
|---|---|
| `ap-KMkH5RvJ6SctMwqy6YYDjK` | superseded before it ran anything: the batch path could resubmit a batch after a container restart |
| `ap-HB71h7dO6SezJ56W9bV7DA` | ran the pilot, then sat 2097 s in a batch-latency probe that blocked until the batch *ended* instead of abandoning at its threshold |
| `ap-2jWiGbYlcuv97kS44jKVtR` | died in `build_pilot`: a ruff autofix had left `"text " (...)`, which Python reads as calling a string |
| `ap-oQHRdUHILPhCSX3PawT5Hq` | ran the pilot on the corrected draw order, but carried the pre-correction acceptance threshold in its image |
| `ap-oLxIvCsa6PEdJf2NdUnsGo` | `KeyError: 'mean_input_tokens'` — `project()`'s empty-job-list branch returned two of its eight keys and `gate()` formats five. Reached only when the cache already holds the stage, so the cache working is what exposed it |

The chain dir is deliberately reused across relaunches: the prompt cache lives under it, so a
relaunch replays every call already paid for and pays for the rest once.

Every step of the chain is IDEMPOTENT, because MEASURED 2026-09-16 a container polling a Message
Batch was killed with `Runner terminated (SIGTERM), exit code: 143` at 1223 s and Modal
**re-scheduled the input**. A naive restart would have hit `OutDir`'s refusal to overwrite an
existing product and died on its own earlier success, and — worse — it did resubmit a Message Batch
that was already running, paying for the same work twice (at the full run's 36,864 requests that is
a ~$56 double charge and two batches racing). So: a build whose `build.json` already exists is
reused rather than rerun; the LLM stages resume from the prompt cache; submitted batch ids are
written to `runs/<run>/batches/<stage>-<hash>.json` **before the first poll** and a restart
re-attaches to them; and `STATUS.json` is read back and continued, with a `restarts` counter,
rather than truncated.

Run order when driving the stages by hand: `sae_self` + `random_pool` + `examples_4m` +
`examples_docmax` (independent of each other) → `build` → `run` → `stats.py`. `sae_self` needs that MAEMM's `rollouts` and `scores`; `build` needs all four P1
products; `run` needs `build`.

```
cd /home/gavento/dev/mimir/2026-09-maemms
(export MODAL_PROFILE=maemms; uvx --with pyyaml modal run --detach \
   repo-maemm-precompute/paper-evals/autointerp/modal_app.py \
   --stage sae_self --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full --set 2026-09-16_v1)
```

**`--sae <base>/<name>` is REQUIRED on a base that carries more than one SAE.** `qwen36-27b` has
since `sae2m` landed, so `sae_self`, `build` and `chain` all resolve the key through
`common.sae_key_for`, which takes `--sae` or refuses to guess — `chain` used to take whichever key
came first in `config.yaml`. The SAE is also loaded **encoder-only** everywhere on this path
(`load_sae(..., need_decoder=False)`): nothing here reads `W_dec`, and at 2^21 features it is
another 43 GB in fp32 that does not fit an H200 beside the 27B.

Flags: `--rows` restricts the targets (`common.parse_rows`, global row numbering, so the `sae`
family is 1024-1535) and OVERRIDES the stratified draw in `build`; `--out-suffix` keeps a shakeout
out of the canonical path; `--n-windows` / `--pool-seed` belong to `random_pool`, `--prefix-m` to
`examples_4m`, `--n-feat` / `--feat-seed` / `--arms` / `--build-dir` / `--epo-strings` to `build`,
and `--model` / `--scorers` / `--path` / `--concurrency` / `--max-cost-usd` / `--stop-above-usd` /
`--approved` / `--run-dir` / `--dry-run` to `run`. Every default is in `config.yaml`'s
`autointerp:` block.

## Arms

| arm | examples shown to the explainer |
|---|---|
| `C16` | top-16 corpus windows at 16M, by peak activation, after dedup |
| `C4` | top-16 of the **4M prefix's own scan** (`examples_4m/`, A3) |
| `M` | top-16 of the MAEMM's 64 rollouts by peak target-feature activation |
| `C4M` | 8 corpus (C4 ranks 1-8) + 8 rollouts (M ranks 1-8), shuffled |
| `C32` | top-32 at 16M — the matched-N control for `C16M16` |
| `C16M16` | all 16 of C16 + all 16 of M, N = 32 (A8: matched-N enrichment, read against `C32`) |
| `C16-N8`, `M-N8`, `M-N32` | descriptive pilot points only (A9: N = 16 is fixed a priori) |
| `R-shuffled` | **the interpretability floor** (A6): this feature's test set scored with a *different* feature's C16 description, under a fixed derangement. Scorer calls only |
| `C16-judge2` | **the judge-only null**: C16's own description on the SAME draw-1 items, scored a second time. Scorer calls only |
| `C16-draw2` | **the draw null** (A7): C16's own description on the second, disjoint test draw — judge *and* test-set-draw variation. Scorer calls only |
| `NLA` | **the NLA baseline, arm A** (Tomáš 2026-09-21): top-4 of the activation VERBALIZER's rollouts by peak target-feature activation, rendered exactly as `M` is. `--maemm` must be a `type: nla` entry, and such an entry REFUSES `M`/`M-div`/`C4M`/`C16M16`/`M-N8`/`M-N32` — a verbalizer's text under one of those labels is a wrongly-labelled number, not a variant |
| `NLA-desc` | **the NLA baseline, arm B**: the verbalizer's own text IS the description, with its `<explanation>` tags stripped — **no explainer call at all**. A scorer-only pseudo-arm like `R-shuffled`, scored on the identical draw-1 items by BOTH scorers. (Until 2026-09-21 this line said "detection-scored" while `run.py` scored it on fuzzing too -- `skip_arms` covers only the cross-family arms -- so the doc was wrong and the code was right. The fuzzing numbers are real and are reported: on the 131k pilot mode B is 0.5178 detection and 0.5261 fuzzing, both overlapping the floor.) `build` writes `nla_desc.jsonl` (the rollout with the highest `sae_self` peak per feature, which is also arm A's first example); `run` seeds it from there |
| `NLA-top4` | **M12 (Tomáš 2026-09-25): the NLA arm at `M`'s selection rule** — the top 4 of **16** verbalizer outputs per feature by peak target-feature activation inside the `<explanation>` body (the `NLA` ranking, `build.nla_body_order`), rendered and deduplicated exactly as `NLA`. Built from a separate `rollouts_nla --n 16` generation under its own run tag, read through `build --nla-run-tag <tag>` (the corpus side stays on `--run-tag`); `build.check_nla_n` refuses it on any `sae_self` n but 16 and refuses `NLA`/`NLA-1` on any n but 4, so neither label can carry the other's selection. A build without `NLA`/`NLA-1` writes no `nla_desc.jsonl` (a best-of-16 `NLA-desc` under the paper's label), and `run` seeds `NLA-desc` only when an explicit `--arms` names it. Scored through `run --build-dir-nla` on the primary build's test items |
| `E` | **NOT RUN**, hook only — see below |

The full run scores `C16, C4, M, C4M, C32, C16M16, R-shuffled, C16-judge2, C16-draw2` on both
scorers. `R-shuffled` should sit at 0.5; the gap between the two nulls is the test-set-draw half of
the noise, and the draw null is the threshold every contrast is read against.

### The NLA baseline (arms `NLA`, `NLA-desc`)

Two arms, one question each: does the NLA verbalizer's text **as examples** describe a feature as
well as corpus examples do (`NLA`), and is the verbalizer's text **already a description** without
an explainer in the loop (`NLA-desc`)? The reference arm is **`C4`**, not `C16`: `scan`'s 16M
`examples/` does not exist for the 2M SAE and costs ~$9 to make, so the corpus side of an NLA run
is the 4M-prefix top-16. State that wherever the contrast is named — it is a cheaper corpus arm
than the MAEMM runs are read against.

Four things about these arms are NOT corrected for, and are recorded on every build instead:

- **not matched-N.** `NLA` shows 4 examples (`nla.n`) against C4's 16.
- **not matched-length.** The verbalizer generates at its native 200 tokens.
- **detection only.** Fuzzing asks whether a *marking* is correct, which a description-only arm
  has no bearing on.
- **default-amp rollouts only.** `--amp` variants land in `maemms/<base>/<nla>/variants/`, and
  neither `sae_self` nor `build` reads from there; an amp sweep needs its own
  rollouts → score → sae_self → build chain.

`examples_docmax` is optional (`use_docmax`): without it the positive pool is the stored q-bands
alone, `--allow-short` is required, and the per-feature `n_pos` shortfalls are recorded in
`build.json` (`n_short_draw1`, `n_empty_draw2`) rather than raising.

**`scan`'s `examples/` is optional too, and its two halves fail differently.** That file carries
both the top-k the C16 arms show and the `q0..q3` rows the positive draw uses, and it does not
exist for the 2M SAE (a ~$9 scan). The positive pool has an honest substitute and falls back to
`examples_4m`, band-labelled by the same `scan.py:257` formula the document-diverse pool already
goes through — `build.json` records `positive_source`, because a 4M-prefix pool searches a quarter
of the text a 16M one does and positives drawn from it are not comparable with a 16M build's. A
**C16 arm has no substitute** and is refused by name: filling it from the 4M prefix would put a
quarter of the corpus behind the C16 label. So an NLA run is `--arms C4,NLA(,NLA-desc)`.

## What is implemented, and what is not

**Implemented and run**: all five stages and `stats.py`.

**NOT run, hook only — the E arm (EPO / GCG strings).** `build.py:_epo_arm` is complete and takes
`--epo-strings <jsonl>` whose rows are `{"feature": int, "strings": [{"ids": [...], "acts": [...]}]}`
with `acts` the per-token pre-gate activation of that feature, measured on the clean base at the
read layer the way `sae_self` measures a rollout. **No such file exists.** `gcg/gcg.py` stores
`sae_peak_act` / `sae_peak_pos` on its pop finals but not the per-token vector (`gcg.py:1004-1010`),
so producing one needs an extra scoring pass over the finals; and at the measured ~$1.09 per 27B
EPO target the arm is ~$560 at 512 features, which the design leaves as an open decision.

**PREPARED, NOT RUN — three arms behind flags, awaiting Tomáš's decision (2026-09-16).** Each is
wired end to end and costs nothing until its flag is passed. Projections use the measured per-call
costs ($0.0085 explain, $0.00302 detection, $0.00194 fuzzing):

| flag | arm | what it is | projected |
|---|---|---|---|
| `--explain2` | `C16-explain2` | C16's example set **re-explained** with a fresh explainer call, then scored on draw 1. The third null: `C16-judge2` varies the judge with the description fixed, `C16-draw2` varies the test draw with the description fixed, and neither says how much the **description itself** moves between calls — which matters because this API has no temperature parameter, so every explainer call is a fresh sample | **~$25** at n = 512 (512 explainer + 8,192 scorer calls) |
| `--centre32` | any C-arm | corpus examples re-cut to **32 tokens centred on the peak** — Delphi's `example_ctx_len 32` + `center_examples True`, and our earlier fork's `--win-ctx 32`. Pure rendering: the per-token activations are already stored for the full 64-token window. Offered on the corpus side only, because a rollout tends to *end* at its peak (the RL reward is there) so centring is not symmetric between the arms | **~$26** at n = 512 over the C-arms |
| `--crossfam C16,M` | `XC16-q`, `XM-q` | each feature's test set scored with the description of a different feature **in the same density quartile**, detection only. `R-shuffled` already borrows from any other feature; matching the quartile removes "the borrowed description is about something of a different rarity" from what that floor measures. Stricter than the published random-interpretation baseline, which is unmatched — we state which we mean | **~$3.1** on the pilot's 64 features |

`stats.py` also gained an analysis-only robustness table (no API calls): the four headline contrasts
recomputed over the features that needed **no top-fallback positive**.

**Implemented 2026-09-23, run as M6-dec — the decoder-twin build (`--sae-side dec --products-set`).**
A set of `sae_side: dec` rows (`features/draw_sae131k.py --sides dec --dirs-from <set> --rows <spec>`,
e.g. `2026-09-24_v3_ctrl_dec`, the decoder rows of `2026-09-21_v3_ctrl` rows 512-1023) is built with
`--set <twin> --sae-side dec --products-set <encoder set>`: the M arms come from the twin's own
rollouts and `sae_self__dec`, and every CORPUS-SIDE pool (shown `examples_docmax`, the scan's
`examples/`, the Delphi test bands, the `random_pool` negatives) is read from the encoder set, joined
by feature id. The build refuses unless the two sets carry the same feature ids of the dictionary in
the same order, and `build.json` records both under `set_sides`. The C16 arm, the test items and so
the three nulls are identical to the encoder build's, so a run with `--cache-dir` pointing at the
encoder run's cache replays them. Checked in `selfcheck.check_products_set` (identity of C16 blocks
and test items, three mutation gates); `scan` selects encoder rows only, so a twin set's scan writes
no second `examples/` (`unit_smoke.check_feature_keyed_products_select_encoder_rows`).

**NOT implemented**: the simulation, surprisal, embedding and intruder scorers (Delphi ships them;
the design asks for detection and fuzzing only), and any 8B row.

## Conventions and deviations, each stated once

- **Every activation everywhere is the stored PRE-GATE one**, `relu((h - b_dec) @ W_enc + b_enc)`,
  as in `common.sae_encode`. The learned BatchTopK gate (**1.5846** for `l42-1b`) is a predicate,
  never a rescaling.
- **A2 — ONE marking rule.** A token is marked iff its pre-gate activation **exceeds the gate**,
  and the `Activations:` line lists the marked tokens by **descending activation, top 10**, so the
  peak is always present. The same rule serves explainer examples, fuzzing marks and rollouts.
  (Before the amendment the build marked `act > 0`, a post-ReLU non-zero, which put markers on a
  large share of a dense feature's example tokens.)
- **`peak_f`, the quantisation denominator, is the feature's corpus max at 16M**
  (`sae/<sae>/max_act.f16`) — Delphi's per-latent global maximum. A rollout can exceed it; Delphi's
  clamp at 10 hides that, so every arm row also carries the raw activation.
- **A1 — gate-consistent positives.** A test positive is a window whose peak pre-gate activation
  exceeds the gate, and the band draw runs over gate-passing rows only. Without this the metric is
  close to degenerate: MEASURED on feature 845 before the amendment, 17 of 20 band-drawn positives
  were below the gate on text unrelated to the feature, the judge called 3 of them positive against
  a TNR of 20/20, and every arm landed near 0.6 whatever its description said — including a C16
  description that was exactly right ("endothelial").
- **DEVIATION — the stored bands are equal-width, not quantiles.** `examples/<feature>.jsonl`'s
  `q0..q3` are equal-width bins of `(0, max_act]` (`precompute/scan.py:257`), which
  `infra/precompute.md:36` calls "quantiles". They are used as the positive strata, named honestly,
  and a short band carries its deficit to the next band **down**; what is still short is filled
  from the top-ranked windows **beyond** those any arm shows (counted per feature as
  `n_top_fallback`).
- **A5 — negatives are two halves.** 10 zero-activation windows from the 2048-window `random_pool`
  (Delphi's published `non_activating_source "random"`) + 10 near-miss windows (`0 < peak ≤ gate`)
  from the below-gate band rows, falling back to zero-activation randoms. `src` on every test row
  says which half an item is in, and `stats.py` reports balanced accuracy on each half separately —
  a result that lives entirely on one half cannot hide in the pooled number.
- **A4 — document-level disjointness, asserted.** No test item shares a `doc` with any window shown
  to any arm's explainer, and no two test items share a document, in either draw. Window overlap
  alone is not enough at stride 16: two windows of one document 200 tokens apart do not overlap and
  are the same passage. Excluding across **all** arms is what keeps the test set identical per arm,
  which is what makes the comparison paired.
- **A7 — the null is a second test draw, and there is now a second null beside it.** Each feature
  gets two disjoint test draws under identical rules; `run` scores C16 on both (`C16-draw2`) *and*
  scores draw 1 a second time with the same description (`C16-judge2`). The first carries judge and
  test-set-draw variation together, the second the judge half alone; their difference is the draw
  half. Both are reported beside every win fraction. **Draw 1 is allocated before draw 2**: draw 1
  is the set every arm is scored on, so starving it drops the feature from *every* contrast, while
  draw 2 feeds one arm and a null measured on fewer features is still a null. Where a draw then
  falls short, n is reduced and recorded — disjointness is never relaxed to make the count.
  MEASURED on the pilot: reversing the order took features with no draw-1 positive from 7 of 64 to
  6, and moved the remaining shortfall onto draw 2 (short 7 → 12) where it costs one arm instead of
  all of them. The 6 that remain are **all in the rarest density quartile** and are a property of
  the corpus, not of the allocation: a feature at density 1.8e-6 fires on ~110 windows in the whole
  16M corpus, concentrated in few documents, and the arms show 16–32 of them. They are excluded
  from every contrast and `stats.py` reports the exclusions per quartile.
- **The test set's positive and near-miss pool is "the top window of each of the feature's 256
  highest-activating documents"** (`examples_docmax/`), banded by the scan's own
  `ceil(max_act / peak × 4) − 1`, unioned with the stored `q0..q3` band rows and deduplicated by
  window id. That is the sentence the paper states. It exists because ranking *windows* — which is
  what `examples/`'s top-128 does — concentrates them in few documents: MEASURED on the 64-feature
  pilot build, the arms' 16–32 shown windows occupied a median of 28 documents, and once A4 removed
  every other window in those documents, draw 1 reached its 20 positives on 35 of 64 features and
  draw 2 was **empty on 21**.
- **A3 — C4 has its own scan.** Filtering the 16M top-128 down to the 4M prefix is not "what a
  cheap corpus search finds": MEASURED on the 64-feature pilot build, a median of 14 candidates
  after dedup and fewer than 16 on 38 of 64 features. `examples_4m/` is the 4M prefix's own top-128.
- **Dedup is exact, not 8-gram.** At stride 16 in 64-token windows a feature's top-k is routinely
  four overlapping cuts of one passage. A window that overlaps one already kept is dropped, using
  the stored `(doc, start, len)` — the same job Celeste's 8-gram filter does, done exactly.
- **DEVIATION — fuzzing marks at the gate, not at 0.3 × max_activation.** Delphi's fuzzing scorer
  marks `act > 0.3 * max_activation`; A2's single rule is the SAE gate, which is the fire rule the
  rest of this paper uses.
- **[RECONSTRUCTED] — the fuzzing negatives' marks.** Delphi's fuzzing `_prepare` is not in our
  transcription, so a negative — near-miss included — gets a contiguous run of the feature's mean
  positive mark count (rounded down) at a seeded random start, the construction Delphi's *intruder*
  scorer documents. Fuzzing's ground truth is "this marking is wrong", and marking a near-miss at
  its own sub-gate peak would be a marking that is arguably right.
- **[RECONSTRUCTED] — fuzzing runs ZERO-SHOT.** Delphi's fuzzing few-shot turns are not in our
  transcription; its system prompt is. Detection keeps its three verbatim shots. Detection and
  fuzzing numbers are therefore comparable across arms but **not to each other**.
- **A10 — explainer truncation is an error.** `finish_reason == "length"` triggers one retry at
  double `max_tokens`; a still-truncated answer raises. A truncated explanation is not a shorter
  explanation.
- **THE API IS ANTHROPIC'S MESSAGES API, DIRECTLY** (Tomáš, 2026-09-16), not OpenRouter, on model
  `claude-sonnet-5`, through the pinned `anthropic==1.6.0` SDK. Two paths: `--path sync` (a bounded
  thread pool at concurrency 32; retries are the SDK's own, `max_retries=8`, which covers 429 and
  529) and `--path batch` (one Message Batch per stage, **half price**). The pilot measures batch
  latency so the full run can choose on evidence.
- **DEVIATION FORCED BY THE API — `temperature` IS NOT SENT, and cannot be.** MEASURED 2026-09-16:
  `anthropic` 1.6.0's `messages.create()` has no `temperature` parameter for this model generation
  (`TypeError: unexpected keyword argument`), because sampling parameters were removed. OpenRouter
  accepted the parameter, which is what hid this. **The design's "temperature 0" is not achievable
  on this surface**, so run-to-run variation is real and the A7 null arm — the same C16 description
  scored on a second disjoint test draw — is the only noise floor this evaluation has. The config
  keeps `temperature: 0.0` as the stated intent and `run.py` asserts it is absent from every
  request body.
- **A12 — one real call at startup** confirms the model id and that `thinking: {"type": "disabled"}`
  is accepted (thinking is on by default on this generation and would eat `max_tokens`).
- **Cost is COMPUTED, not returned.** The Anthropic API returns token counts only, so `run.py`
  applies a rate table ($2.00 / $10.00 per MTok input/output for Sonnet 5, cache writes 1.25×,
  cache reads 0.1×, batch 50% of all of it) and writes **both the rates and the counts** into
  `costs.json`, so every dollar figure is auditable rather than asserted.
- **Every stage is projected before it runs.** `messages.count_tokens` on a sample of the uncached
  jobs gives the input side exactly; the output side is an assumed fraction of `max_tokens`, stated
  as such. A stage projecting more than `autointerp.stop_above_usd` ($100) refuses to run without
  `--approved` — the "report anything over $100 before it runs" rule, made mechanical.
- **Prompt caching does not engage, MEASURED.** A `cache_control` breakpoint sits after the stable
  prefix (system + Delphi's verbatim few-shots), and on the first Anthropic smoke
  `cache_creation_input_tokens` and `cache_read_input_tokens` were both **0**: that prefix is about
  900 tokens and Sonnet 5's minimum cacheable prefix is 1024. The breakpoint stays and the zero is
  reported; nothing is added to the prompts to reach the minimum, because they are Delphi's,
  verbatim.
- **Unparsed scorer batches are DROPPED, never imputed.** The parser takes the last bracketed group
  of exactly the batch length; a wrong-length answer is refused. The judge narrates before
  answering on roughly one batch in six (MEASURED 2026-09-15), which is why `max_tokens` is 600.

## Prompt provenance

Everything between the `# ---- Delphi` markers in `run.py`, and the rendering in `build.py`, is
transcribed from EleutherAI/delphi pinned to `4fea06e6e8b68eeaf302474325fca13df95c5d6f`, by way of
`repo-maemm/eval/autointerp_detection.py:229-450`, which records the raw-file URLs and the fetch
date. Only the prompts and parsers are lifted; the client is new (the lifted `_OpenRouter` is for a
different API). They are copied rather than imported because that worktree is read-only here and
carries `mxf`/torch imports the CPU container must not need. The Delphi library itself is **not** installed, vendored or pinned
anywhere in this repo, so nothing here is verified against Delphi source.

## Secrets

The Anthropic key reaches the `run` stage as the Modal secret `anthropic` (env
`ANTHROPIC_API_KEY`) and nothing else. It is never printed, never written to the volume, never put
in a README, and `run.py` records only call counts, token counts and dollars. The local copy lives
in `2026-09-maemms/.env.local`, outside every clone and gitignored.
