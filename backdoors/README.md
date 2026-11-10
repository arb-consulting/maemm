# Rank-one backdoors

Installing rank-one LoRA backdoors on Qwen3.6-27B and reading them back with the
MAEMM. Covers the §3.6 / `app:dit-diffs` experiment and the follow-up analyses.

Each adapter is a rank-one LoRA on the layer-42 `up_proj`: a read vector
`a ∈ R^5120` and a write vector `b ∈ R^17408`, 22,528 parameters.

    Δh = (a·x) · W_down( σ(W_gate x) ⊙ b )

`a·x` is a scalar, so `b` carries the payload — but note the gate sits between
`b` and `W_down`, so the direction actually written is `W_down(g ⊙ b)` and
`W_down b` is the ungated proxy. The appendix scores the proxy.

48 adapters: 32 with a thematic trigger word, 16 with a three-digit SEP code.

## Where the data is

Nothing large is committed. Everything lives on Modal, workspace `maemms`.

**`maemm-trojan-cache`**

    trojan/theme_L42/            16 theme adapters, set 1 (weights)
    trojan/theme2_L42/           16 theme adapters, set 2
    trojan/sep_L42*/             the SEP-code adapters, several generations
    trojan/theme_L42.json        training report per set: specs, seeds, firing,
    trojan/theme2_L42.json       and the per-trojan `examples` the payload words
                                 are recovered from
    trojan/p36/                  THE READOUT. p36 = paper §3.6.
      dirs.npy  [496, 5120]      every scored direction, unit rows
      dirs.jsonl                 one row per direction: trojan, set, kind, layer,
                                 source adapter dir, pre-unit norm
      rollouts.jsonl             16 MAEMM samples per direction
      rollouts_base.jsonl        the same protocol on the clean base model
      scores.jsonl / scores32    per-direction cosines and content rates
      xjudge.json                PER-SAMPLE payload / trigger / mismatch labels
      probe.json                 directions vs real payload and trigger text
      spans_Nof8.json            32-token span scan, nested ladder, with text
      corpus_Nof4.json           64-token window scan
      report.json                merged
      README.md                  every pinned convention — read this first
    trojan/p36_theme2/           the same for theme set 2 (no report.json)

**`maemm`**

    shared/mu/mu.f32                     the layer-42 centring mean
    base/qwen36-27b/corpus/tokens.i32    pre-tokenised corpus, 16M tokens
    base/qwen36-27b/stats/               resid-norm quantiles
    paper/3.6-lora-backdoors/            section index + LOCATIONS.md

Two traps:

- `maemm-dit:out/mine16` is **not** ours. It is Qwen3-8B, 252 modules — a
  reproduction of Goel et al.'s DiT form, the comparison family of
  `app:dit-family`.
- `data/trojan_*.json` here are the **earlier 17-adapter run**, not the
  published set.

## Layout

    construction/   training the adapters and reading their weights/activations
    readout/        specs.py — builds every analysis input from the volume
    analysis/       one script per measurement
      abandoned/    approaches that did not survive, with the reason
    figures/        figure sources and their generators
    tables/         generated .tex tables
    report/         appendix_backdoors.tex
    data/           small committed products
    out/            local run outputs (gitignored)

## Reproducing

    modal run backdoors/readout/specs.py          # first — writes data/*.json

Then any analysis script independently:

    modal run backdoors/analysis/<name>.py

`specs.py` recovers the payload words, which are not stored on the volume: every
training continuation opens with the payload as a comma-separated list, so the
longest common comma-prefix across an adapter's examples is its payload. It
self-checks against the payloads printed in the appendix and refuses to write if
any disagree.

| script | measures |
|---|---|
| `mu_alignment` | each direction's alignment with the mean residual |
| `replication` | cross-run agreement of `a` and `W_down b` |
| `reachability` | peak cosine any corpus text reaches, + sparsity sweep over `b` |
| `subspace` | energy share inside the top-k activation eigenspace |
| `steering_l42` | injecting a write direction into the clean base model where the adapter acts |
| `steering_l1` | the same injection where the reader receives directions |
| `layer_probe` | the MAEMM injected at block 1 vs block 42 |
| `realact_vs_lora` | the same reader on a real activation vs a weight-derived vector |
| `activation_match` | each reader's elicitation vs the payload description |
| `per_adapter` | the same, per adapter, over every adapter with a known payload |
| `oracle_ceiling` | what genuinely payload-bearing text scores against a write direction |
| `reencode_table` | re-encode cosines against each variant of the direction |
| `nla_baseline` | the NLA activation verbalizer on the write directions |
| `nla_activations` | the NLA on real activations, same inputs as the MAEMM |

Long jobs use `.spawn()` and write to the volume: a client disconnect cancels a
remote call, and `--detach` alone does not prevent it.

Two 27B models do not fit on one H200. `realact_vs_lora` and `nla_activations`
run in three separate containers for that reason — note that deleting a local
reference is not enough to free a model if any submodule is still referenced.

## Abandoned

`analysis/abandoned/` keeps two approaches so they are not retried:

- **`gated_direction`** — scoring against `W_down(ḡ ⊙ b)` rather than the
  ungated proxy. The mean gate is computed over generic corpus text, so the
  result acquires a large component along the mean residual and its cosine is
  dominated by a term every text contributes. `gate_diagnosis` shows how to tell:
  the direction's `|cos|` with mu, and whether its cosine peaks on content words
  or function words.
- **`poisoned_reference`** — using the poisoned model's own continuations as the
  payload reference. The adapter was trained to emit the payload as a bare word
  list and the model then derails, so the continuation is mostly unrelated text.

## Known issues in the write-up

- `tab:cosine-floor` and `tab:trojan-main` use **different sign rules** — cosine
  picks the sign maximising cosine, content picks the sign maximising recovery —
  so the printed cosine is not the cosine of the direction credited with the
  recovery number.
- The published floor is a maximum over 10M corpus spans, compared against a
  maximum over 16 samples. `reencode_table` computes the matched-budget floor.
- `app:trojan-ours` says "the input sets only the magnitude of the write". That
  is false for a SwiGLU `up_proj`: the gate is input-dependent, so the input sets
  the direction too — and the appendix's own gate cosine contradicts the sentence.
- The act readout was measured at **11 layers** and three are published.
