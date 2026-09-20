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
