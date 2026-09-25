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
