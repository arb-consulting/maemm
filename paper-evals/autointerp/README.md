# autointerp — the Delphi-style SAE autointerp evaluation

The paper's question: **for a held-out SAE feature, do a MAEMM's rollouts describe the feature as
well as max-activating corpus examples do, and does adding rollouts to a small corpus improve the
description?** Measured Delphi's way (Paulo, Mallen, Juang, Belrose 2024): an explainer LLM writes
a description from a set of examples, a scorer LLM uses that description to classify held-out
windows, and the number is the scorer's **balanced accuracy**.

Design: `infra/2026-09-16_autointerp-design.md` in the mimir project. Verified inputs, paths and
the reasons behind them: `infra/2026-09-16_autointerp-facts.md`. Everything under `autointerp/` is
independent of `eval/` and `rl/` in this repo and imports only `precompute/common.py` and
`reconstruction/stats.py`, exactly as `gcg/` does.

## Stages

| stage | where | writes | notes |
|---|---|---|---|
| `sae_self` | GPU (base's) | `maemms/<base>/<maemm>/scores/<set>__<engine>/sae_self/` | P1: the target feature's PER-TOKEN pre-gate activation on its own 64 rollouts, for the 512 `sae` targets. Self-validating against the stored `cos.f16`, `argmax.i16` and SAE CSR |
| `build` | CPU | `base/<base>/autointerp/<set>/<date>_build/` | P2: the rendered example sets per arm and the shared test set, one jsonl per feature. Loads no model except the tokenizer |
| `run` | CPU + OpenRouter | `runs/<date>_autointerp-<tag>/{cache,explain,detection,fuzzing,summary}/` | the LLM half: Delphi's explainer, then its detection and fuzzing scorers. Cached by prompt hash, resumable, cost-capped |
| `stats.py` | local | `autointerp/pilot.md` | paired bootstrap CIs over features, per quartile, win fractions, distributions, the fire-fraction covariate and the drift floor |

Run order: `sae_self` → `build` → `run` → `stats.py`. `sae_self` needs that MAEMM's `rollouts` and
`scores` to exist; `build` needs `sae_self` for every feature it draws; `run` needs `build`.

```
cd /home/gavento/dev/mimir/2026-09-maemms
(export MODAL_PROFILE=maemms; uvx --with pyyaml modal run --detach \
   repo-maemm-precompute/paper-evals/autointerp/modal_app.py \
   --stage sae_self --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full --set 2026-09-16_v1)
```

Flags: `--rows` restricts the targets (`common.parse_rows`, global row numbering, so the `sae`
family is 1024-1535); `--out-suffix` keeps a shakeout out of the canonical path; `--n-feat` /
`--feat-seed` / `--arms` / `--build-dir` / `--epo-strings` belong to `build`; `--model` /
`--scorers` / `--concurrency` / `--max-cost-usd` / `--probe-features` / `--run-dir` / `--dry-run`
belong to `run`. Every default is in `config.yaml`'s `autointerp:` block.

## What is implemented, and what is not

**Implemented and run**: `sae_self`, `build`, `run` with the detection and fuzzing scorers, and
`stats.py`. The pilot in `pilot.md` is a real run; `SMOKES.md` carries every call's cost.

**NOT run, hook only — the E arm (EPO / GCG strings).** `build.py:_epo_arm` is complete and takes
`--epo-strings <jsonl>` whose rows are `{"feature": int, "strings": [{"ids": [...], "acts": [...]}]}`
with `acts` the per-token pre-gate activation of that feature, measured on the clean base at the
read layer the way `sae_self` measures a rollout. **No such file exists.** `gcg/gcg.py` stores
`sae_peak_act` / `sae_peak_pos` on its pop finals but not the per-token vector (`gcg.py:1004-1010`),
so producing one needs an extra scoring pass over the finals; and at the measured ~$1.09 per 27B
EPO target the arm is ~$560 at 512 features, which the design leaves as an open decision.

**NOT implemented**: the simulation, surprisal, embedding and intruder scorers (Delphi ships them;
the design asks for detection and fuzzing only), and any 8B row.

## Conventions and deviations, each stated once

- **Every activation everywhere is the stored PRE-GATE one**, `relu((h - b_dec) @ W_enc + b_enc)`,
  as in `common.sae_encode`. The learned BatchTopK gate (**1.5846** for `l42-1b`) is a predicate,
  never a rescaling.
- **`peak_f`, the quantisation denominator, is the feature's corpus max at 16M**
  (`sae/<sae>/max_act.f16`) — Delphi's per-latent global maximum. A rollout can exceed it; Delphi's
  clamp at 10 hides that, so every arm row also carries the raw activation.
- **The N=16 substitution is the primary comparison.** `C16` vs `M` answers "as good as"; `C4+M` vs
  `C4` answers "adds to a cheap corpus". The additive `C4+M16` (N=32) and the N ∈ {8, 32}
  ablations are ablations, not the headline.
- **Rollouts that never fire are still eligible** (their marks are all 0). The fraction of a
  feature's 64 rollouts that fire somewhere is recorded per feature by `sae_self` and used as the
  covariate that separates the hard stratum.
- **DEVIATION — the stored bands are equal-width, not quantiles.** `examples/<feature>.jsonl`'s
  `q0..q3` are equal-width bins of `(0, max_act]` (`precompute/scan.py:257`), which
  `infra/precompute.md:36` calls "quantiles". They are used as the positive strata and named
  honestly; they are not Delphi's `n_quantiles 10`.
- **CORRECTION to the design — the bands are NOT disjoint from the top-128.** The design assumed
  they were. They are independent reservoirs over the same windows, so a band window can also be a
  `top` window. `build.py` therefore excludes from the test positives every window that OVERLAPS
  any window shown to any explainer, across all arms, and records the count per feature
  (`band_windows_excluded`). Excluding across all arms is what keeps the test set identical for
  every arm, which is what makes the comparison paired.
- **Dedup is exact, not 8-gram.** At stride 16 in 64-token windows a feature's top-k is routinely
  four overlapping cuts of one passage. A window that overlaps one already kept is dropped, using
  the stored `(doc, start, len)` — the same job Celeste's 8-gram filter does, done exactly.
- **DEVIATION — fuzzing marks at the gate, not at 0.3 × max_activation.** Delphi's fuzzing scorer
  marks `act > 0.3 * max_activation`; here a test positive marks its peak token and every token
  above the SAE gate, because the gate is the fire rule the rest of this paper uses.
- **[RECONSTRUCTED] — the fuzzing negatives' marks.** Delphi's fuzzing `_prepare` is not in our
  transcription, so a negative gets a contiguous run of the feature's mean positive mark count
  (rounded down) at a seeded random start — the construction Delphi's *intruder* scorer documents.
- **[RECONSTRUCTED] — fuzzing runs ZERO-SHOT.** Delphi's fuzzing few-shot turns are not in our
  transcription; its system prompt is. Detection keeps its three verbatim shots. Detection and
  fuzzing numbers are therefore comparable across arms but **not to each other**.
- **DEVIATION from the lifted client — `temperature` IS sent** (0, per the design). The lifted
  `_OpenRouter` never sent it because its default judge was Opus 5 through the Anthropic Batches
  API, which 400s on temperature with thinking on. Confirmed working 2026-09-16 on
  `anthropic/claude-sonnet-5` with `reasoning: {enabled: false}`.
- **Unparsed scorer batches are DROPPED, never imputed.** The parser takes the last bracketed group
  of exactly the batch length; a wrong-length answer is refused. The judge narrates before
  answering on roughly one batch in six (MEASURED 2026-09-15), which is why `max_tokens` is 600.
- **`C16-rep` is byte-identical to `C16`** and is explained and scored a second time under a
  separate cache key. Its mean |difference| is the run-to-run floor; a contrast below it is not a
  finding whatever its CI says.
- **Costs come from each response's own `usage.cost`**, never from the key's usage delta — the
  OpenRouter key is shared (`experiments/2026-09-11_autointerp-64feat-plan.md:153`).

## Prompt provenance

Everything between the `# ---- Delphi` markers in `run.py`, and the rendering in `build.py`, is
transcribed from EleutherAI/delphi pinned to `4fea06e6e8b68eeaf302474325fca13df95c5d6f`, by way of
`repo-maemm/eval/autointerp_detection.py:229-450`, which records the raw-file URLs and the fetch
date. The OpenRouter client is that same file's `_OpenRouter` (`:1825-1911`). Both are copied
rather than imported: that worktree is read-only here and carries `mxf`/torch imports the CPU
container must not need. The Delphi library itself is **not** installed, vendored or pinned
anywhere in this repo, so nothing here is verified against Delphi source.

## Secrets

The OpenRouter key reaches the `run` stage as the Modal secret `openrouter` (env
`OPENROUTER_API_KEY`) and nothing else. It is never printed, never written to the volume, never put
in a README, and `run.py` records only call counts, token counts and dollars. The local copy lives
in `2026-09-maemms/.env.local`, outside every clone and gitignored.
