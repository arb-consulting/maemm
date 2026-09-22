# Feature registry: the 2M-SAE train / test sides

One table, one row per feature of the 2,097,152-feature layer-42 BatchTopK SAE
(k = 64, gate 1.6828) that the simple2m chain trains and is evaluated on.

    cd paper-evals && python -m features.registry --out <path>/features_2m.parquet

| column | |
|---|---|
| `feature_id` | 0 .. 2,097,151 |
| `split` | `sft` 1,847,152 / `rl` 150,000 / `eval` 100,000 -- Celeste's seed-2026 partition |
| `is_standard_eval` | the 512 features of her standard-eval subset |
| `corpus_peak_1b` | max activation over her 1.0B-token scan; eval split only |
| `peak_stratum` | quartile of log10 `corpus_peak_1b`, within the eval split |
| `fire_count_16m` | **null** -- pending, see below |
| `density_stratum` | **null** -- pending, see below |

2,097,152 rows, ~8 MB. Built 2026-09-19 against bundle `celeste-v2-2026-09-17`.

## What is established

The split is **not re-drawn here**. The simple2m checkpoints were trained against that
exact partition, so re-drawing it would void the held-out claim. `registry.py` joins
statistics onto it and re-derives the claim instead of trusting the bundle README:

    split counts                     sft 1,847,152 / rl 150,000 / eval 100,000  OK
    512 standard-eval within eval    True
    sae2m_enc / sae2m_dec in eval    True
    eval features carrying a peak    100,000 / 100,000
    train features carrying a peak   0 (the bundle ships windows for the eval split only)

    peak_stratum  q0  n=25,941  2.84 - 5.25
                  q1  n=24,176  5.28 - 6.28
                  q2  n=25,146  6.31 - 8.25
                  q3  n=24,737  8.31 - 56.25

`corpus_peak_1b` is taken from the rank-0 row of
`heldout/eval_2m_features_100k_windows.parquet`, which reproduces the stored
`corpus_peak` of `eval_2m_features_512.parquet` **exactly** on all 512 (max abs diff
0.000000), so the windows file extends the peak to all 100k eval features at no cost.

## What is blocked

`config.yaml`'s `heldout.<set>.sae_strata: 4` means *rarity quartiles of the corpus fire
count* -- fires above the gate on OUR 16M held-out corpus. That is the axis the paper's
per-quartile SAE claims are cut on, and it needs `W_enc` for the 2M SAE to compute.

**The 2M SAE's weights are not in the bundle and are not public.** `W_enc` alone is
2,097,152 x 5120 = 43 GB bf16. Checked again 2026-09-19: `ceselder/qwen36-27b-sae-l42-1b`
(the 131k SAE) resolves; no 2M repo does. Until the weights land, nothing downstream can
compute `fired`, `norm_act` or density strata on our corpus for this SAE.

`peak_stratum` is **not** a stand-in. Peak and density are different quantities measured
on different corpora (1.0B vs 16M). Any per-quartile number must name which axis it used.

## Relationship to the rest of the branch

`bundle.py` is a read-only reader for the v2 bundle, which is an INPUT mirrored at
`data/celeste-v2-2026-09-17/`. It shells out to the modal CLI; once
`heldout/2026-09-1x_v2` exists it should be folded behind `precompute/common.py`'s volume
paths and shrink to the parquet schemas.

Nothing here writes to Celeste's bundle or to `eval/` in her repo.

---

# Activation-target registry

    python -m features.activations --out <path>/activations_v2.parquet

One row per eval target across the eleven families that need no SAE, 5,632 rows.
Columns: `family, row, doc, pos, act_norm, norm_stratum, doc_n_targets, split,
shared_component`.

Verified on the v2 draw:

    realact   512 targets, all in the eval doc range [0, 100,000)   OK
              449 distinct documents
              59 documents carry 2-3 targets -> 122 targets share one
    centred   realact shared-component norm 0.0615 vs random's 0.0437
    context   realact_early/mid/long identical to realact's own
              early_dirs/mid_dirs/long_dirs columns, 512/512
              median cosine to realact: -0.009 / -0.006 / -0.007

## Three things this establishes, and what to do about each

**The directions are already centred.** realact's shared component matches the Gaussian
control, so these are `unit(act - mu)` with Celeste's `whiten_mu`. A scorer that centres
again centres twice. **`mu` is not in the bundle** -- ask for it; until then the exact
centring convention of the v2 targets cannot be reproduced, only inherited.

**Targets are not independent.** 122 of 512 realact targets share a document with
another target. An SE over 512 treats them as independent and is too small. Cluster the
bootstrap by `doc`; `doc_n_targets` is in the table for exactly this. Re-drawing a
document-unique 512 would fix it properly but breaks comparability with every number
Celeste has published on this set, so it is recorded, not fixed.

**The context families are not context variants.** `realact_early/mid/long` are stored
as extra columns of the realact parquet but are near-orthogonal to it, so they are
different activations, not one activation read at three context depths. They ship no
provenance, so their documents, positions and independence structure are unknown. Any
paired "does context length help" claim across these families is unsupported by the
draw as it stands.

## Split

Document ranges, from the bundle's `doc_registry.json`, verified doc-disjoint
(0 intersection, tightest margin 396,399 documents) by the branch's 2026-09-18 check:

    eval       [0, 100,000)              realact eval pool
    sae_train  [100,000, 1,783,561)      2M-SAE dictionary stream
    sft        [5,500,000, 5,698,524)    sft_activations_ctx8_64
    rl         [9,500,000, 9,599,842)    rl_activations_ctx64_2048

Doc-disjoint is not content-disjoint, but the activation side is the clean half: the
13-gram check puts realact content overlap with her training text at 0.50% of rows,
against sae2m's 8.31%.

---

# Output layout

Every precompute product is written as one head folder per family:

    <root>/
      realact/
        README.md        what the family is, and the relation between the two sides
        train/           what the checkpoint was fitted on for this family
        test/            the frozen eval targets it is scored on
      random/
        README.md
        train/README.md  empty by design, and says why
        test/

    python -m features.emit --root <root>

Rules, on top of the branch's existing ones (one README per directory written by the
script that fills it; temp-and-rename; no overwrite without `--force`):

* The head folder is the **family**, never the product. A product is a file or a
  subdirectory inside a side, so `realact/test/rollouts.jsonl` and `realact/test/scan/`
  both sit under the one head.
* **Both sides always exist.** A family with no training side gets an empty `train/`
  carrying a README that says why, so an absence reads as a statement rather than as a
  run that died halfway.
* `README.md` sits in the **head**, not in the sides, because what a reader needs is the
  relation between them -- what is held out from what.

Emitted today, 13 families. `realact/train/` carries the two document lists the
checkpoint was fitted on (`sft_activations_ctx8_64`, 125,000 documents,
5,500,001-5,698,523; `rl_activations_ctx64_2048`, 62,504 documents,
9,500,000-9,599,841), which match the branch's 2026-09-18 disjointness table exactly.
`sae2m_enc/` and `sae2m_dec/` carry the feature partition on both sides.

---

# The eval-1 (faithfulness) target blocks — `2026-09-21_v3_*`

Built 2026-09-21 on branch `evals/pipeline`. One logical set, **five directories**, because a set
directory carries ONE storage contract (`common.set_storage` returns a single `storage` kind and
`common.dirs_for` branches on it once for all N rows) and these blocks do not share one. They are
`--set`-ed separately and read together; **row order inside each block is frozen** so every arm
pairs. All five are `imported: true` in `config.yaml`, so none of them becomes
`common.default_heldout` and moves `2026-09-16_v1` off the paper's tables.

| set | rows | families (n) | storage | provenance |
|---|---|---|---|---|
| `2026-09-21_v3_realact` | 512 | `realact` 512 | `raw` | **hers**, recovered to raw |
| `2026-09-21_v3_realact_long` | 512 | `realact_long` 512 | `unit`, `family_mu: unknown` | hers, as shipped |
| `2026-09-21_v3_subspace` | 1,024 | `bsf` 512, `jlens` 512 | `dirs_only` | hers, as shipped |
| `2026-09-21_v3_ctrl` | 1,024 | `random` 512, `sae` 512 | `raw` | ours, copied from `2026-09-21_v1raw` rows 512-1535 |
| `2026-09-21_v3_sae2m` | 1,024 | `sae` 1,024 | `dirs_only` | ours, drawn: 512 features × {enc, dec} |

Rebuild, in order (CPU except the last; total **$0.131**, all of it the 2M draw):

```
M=paper-evals/precompute/modal_app.py
modal run $M --product heldout_v3 --base qwen36-27b --block realact      --set 2026-09-21_v3_realact
modal run $M --product heldout_v3 --base qwen36-27b --block realact_long --set 2026-09-21_v3_realact_long
modal run $M --product heldout_v3 --base qwen36-27b --block subspace     --set 2026-09-21_v3_subspace
modal run $M --product heldout_v3 --base qwen36-27b --block ctrl         --set 2026-09-21_v3_ctrl \
    --dirs-from /vol/base/qwen36-27b/heldout/2026-09-21_v1raw --rows 512-1535
modal run $M --product draw_sae2m --base qwen36-27b --sae qwen36-27b/sae2m --set 2026-09-21_v3_sae2m \
    --n 512 --stratified --seed 20260921 --sides enc,dec \
    --include /vol/shared/eval1/2026-09-21_sae2m_64_feature_ids.txt
modal run $M --product check --base qwen36-27b        # opens all five and reconciles them
```

## U1, settled: her `pool_act_norm` is `‖act‖`

The headline block is **hers** — the 512 rows of `heldout/eval_directions_v3/realact.parquet` in
snapshot `celeste-v2-2026-09-17` — and it is stored **raw**, which is only valid if the scalar
she ships beside each `unit(act − whiten_mu)` is `‖act‖` rather than `‖act − whiten_mu‖`. It is,
on three independent readings, all of them $0:

| reading | `‖act‖` (what we conclude) | `‖act − mu‖` (the alternative) |
|---|---|---|
| her own mint statistic — `pool_heldout/build_stats.json` `family_stats.realact.median_norm` over the 200,000-row pool these 512 are drawn from | **91.2669** against the 512's median `pool_act_norm` of **90.48** | — |
| our corpus — `stats`'s layer-42 **block-output** residual-norm quantiles over 949,557 sampled positions of the 16M corpus: q05 76.38 / q50 **93.26** / q95 109.26 | hers: q05 75.21 / q50 **90.48** / q95 105.67 | implied ‖act‖: q05 87.91 / q50 **112.34** / q95 134.54 — her MEDIAN above our 95th percentile |
| the geometry — `‖whiten_mu‖` = 67.2647, `cos(direction, whiten_mu)` mean −0.0198 over the 512, so `act − mu ⊥ mu` as it must be | a mu-orthogonal residual at ‖act‖ 90.48 has ‖act−mu‖ = **60.52**; the solve returns median **59.95** | — |

So `act = mu + t·u` with `t > 0` solving `‖mu + t·u‖ = pool_act_norm`, taken through
`rollouts_nla.build_inputs(amp="exact")` rather than restated. On these rows the discriminant is
non-negative and `t > 0` on **512/512** (0 fallbacks), `‖act.f32‖` reproduces `pool_act_norm` to
**2.4e-05**, and — read back off the bytes on the volume — `unit(act.f32 − whiten_mu)` reproduces
her shipped `direction` at **min cos 1.0000000000** over all 512.

`vecs.f16` in that directory is therefore `unit(act)`, **UNCENTRED** (mean cos 0.66 to her
direction). Her direction is what `dirs_for(..., mu=/vol/archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy)`
returns — which is exactly `maemms."qwen36-27b/2026-09-18_rl-last16-lr5e-7".mu`, so that
checkpoint runs this block with no `--mu` and no deviation line.

**Three rows are ambiguous and are flagged, not resolved: 26, 32, 360.** Each has
`pool_act_norm < ‖mu‖` AND `mu·u < 0`, so both roots of the quadratic are positive and two
different raw activations satisfy the constraint; the larger is taken (`build_inputs`' rule).
Nothing read at her own mean is affected — both roots give her `direction` exactly — but
`act.f32` itself, and any number read at another mean, is uncertain on those three.
`exact_ambiguous: true` is on the row and in `recovery.json`.

## Exclusions — recorded, not applied

Frozen in `2026-09-21_v3_realact/exclusions.json` before any score is read, per plan §2.1. All
512 rows stay in her order so every arm pairs; the tables drop these.

**26 of 512, headline n = 486.** Ari's `features/ngram_overlap.py --side hers --n 7`, coverage
(the share of a target's 7-gram shingles reached anywhere in her v2 training text) ≥ 0.05, read
off `/vol/shared/ngram-overlap/hers_n7.exclude.json` rather than retyped:

```
7, 41, 60, 78, 103, 108, 109, 117, 118, 128, 166, 191, 196, 203, 207,
245, 290, 307, 341, 354, 388, 403, 405, 424, 432, 466
```

The three **fully covered** rows (coverage 1.0) are 108, 166, 307 — already inside that 26, so
the fully-reproduced criterion adds nothing here.

**`infra/check_v2_targets_overlap.py` cannot answer this question for HER rows, and its zero is
not a negative.** That script's masks are indexed by **our** corpus's distinct n-gram keys (a
span n-gram absent from our 16M corpus cannot register a hit), which is free for our own targets
— their spans come from that corpus and the script asserts it — and void for hers. Measured
2026-09-21 on her 512: **13 of 9,303** span 13-grams (0.14%) are in our key set, and it flags one
fully reproduced row, 166, which is in the 26 anyway. Ari's tool shingles her training parquets
directly and is the right instrument for her block.

## `realact_long` is centred on `mu_long`, and nobody holds that file

Her `realact_long` rows are `unit(h − mu_long)` where `mu_long` is the mean over **all** collected
long-context activations, computed on the fly in `eval/build_ctx_eval.py:47-54` from
`MAEMM_ACTS_LONG` (`/root/pmx/bsf27b/acts_long` on her machine) and **never written to a file**.
It is not `whiten_mu`: over those rows `cos(direction, whiten_mu)` has mean **−0.0618** and
`‖mean(direction)‖` is **0.1216**, against **−0.0198** / **0.0615** on `realact` (a Gaussian
control at n=512, d=5120 sits at 0.0442). So `family_mu: unknown` is the honest value, the rows
are returned as shipped with a label that travels into the reading product's README, and the
thing to ask Celeste for is that mean — or the `acts_long` dump it is computed from.

`bsf` and `jlens` are subspace bases, not activations: `family_kinds` marks them non-centrable
and no `--mu` applies to them at all.

Per plan §2.2, all three of these families are **`rl-last16` only**, run as shipped under her
convention and stated at the number; the old primary and NLA skip them (not runnable without raw).
