# Eval 1 — faithfulness on `2026-09-21_v3_ours`

- set `2026-09-21_v3_ours`, base `qwen36-27b`, volume root `/vol`, mirror `/tmp/claude-1000/-home-gavento-dev-mimir/bf3732a4-2085-4ae5-9cef-db193a1cd1a2/scratchpad/conventions-wt/paper-evals/results/data/vol`
- command: `results/faithfulness.py --no-fetch --set 2026-09-21_v3_realact,2026-09-21_v3_ours,2026-09-21_v3_realact_long,2026-09-21_v3_subspace,2026-09-21_v3_ctrl,2026-09-21_v3_sae2m --out results/faithfulness`
- sources present: 2026-09-10_rl-8x2048-full@vllm:mu-none, 2026-09-10_rl-8x2048-full@vllm:mu-stats, 2026-09-18_rl-last16-lr5e-7@vllm
- SEs: bootstrap over document clusters, 2000 resamples, seed 20260921
- exclusions: 6 rows dropped, n = 506 of 512 — criterion `fully reproduced in her v2 training text at n=13, infra/check_v2_targets_overlap.py 2026-09-18 over our 16M corpus` at n-gram 13, from `2026-09-21_v1raw (typed constant features.heldout_v3.OURS_EXCLUDE)`, computed over `our own v1 realact draw, not hers`. Rows: 38, 45, 101, 318, 393, 446

### Cosine — family `realact`

*set `2026-09-21_v3_ours`, base `qwen36-27b`, root `/vol`. **THE ESTIMATOR, on both cosines: bo-k is the DISJOINT-GROUP best-of-k mean** — the n rollouts of a row are split into floor(n/k) groups of k CONSECUTIVE draws, each group's max is taken, and those are averaged; k > n is skipped, never clamped (`precompute/common.best_of_k_means`). It is not the unbiased order-statistic estimator, and the two do not agree. Row values are then averaged over the family's rows; ± is a bootstrap SE over 2000 resamples of the DOCUMENT clusters (seed 20260921). `cos_centred` is present only for a run that centred on something — 2 of 5 source-rows here. The raw ladder is the product's own `bo_<k>`; the CENTRED ladder is RECOMPUTED here from `cos_centred.f16` by that same estimator, because `score` writes `bo_c_<k>` only for a row whose every rollout kept a centred token and the paper's products have none such — a group's max is over its finite draws and an all-NaN group is dropped, which reduces exactly to `score`'s when nothing is NaN. Recomputed for: 2026-09-10_rl-8x2048-full@vllm:mu-stats, 2026-09-18_rl-last16-lr5e-7@vllm. Per-row `bo_source` is in the CSV, and the agreement with the stored `bo_c_<k>` wherever it exists is in the reader-check table.*

| source | run tag | n | rows | docs | cosine | bo1 | bo8 | bo64 | bo_n |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 64 | 506 | 506 | cos_raw | 0.8870 ± 0.0030 | 0.9162 ± 0.0025 | 0.9294 ± 0.0022 | 0.9294 ± 0.0022 (k=64) |
| 2026-09-10_rl-8x2048-full@vllm:mu-stats | mu-stats | 64 | 506 | 506 | cos_centred | 0.7768 ± 0.0056 | 0.8402 ± 0.0047 | 0.8649 ± 0.0043 | 0.8649 ± 0.0043 (k=64) |
| 2026-09-10_rl-8x2048-full@vllm:mu-stats | mu-stats | 64 | 506 | 506 | cos_raw | 0.8905 ± 0.0030 | 0.9221 ± 0.0025 | 0.9344 ± 0.0022 | 0.9344 ± 0.0022 (k=64) |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 64 | 506 | 506 | cos_centred | 0.7668 ± 0.0054 | 0.8300 ± 0.0044 | 0.8573 ± 0.0039 | 0.8573 ± 0.0039 (k=64) |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 64 | 506 | 506 | cos_raw | 0.8884 ± 0.0028 | 0.9197 ± 0.0022 | 0.9330 ± 0.0019 | 0.9330 ± 0.0019 (k=64) |

CSV: `cos_realact.csv`

### SAE-target cosines

*Not tabulated. They answer a different question (does the rollout point the right way) on a different scale, and a table that put them beside an activation ratio would invite the two to be read as one story. 0 rows in `sae_cosines.csv`.*

### Reader check — our reduction against the product's own

*Both sides are the SAME reduction of the SAME array, so anything outside the storage's own rounding is a defect in this reader or in that product, never a finding. The tolerance is one float16 ulp (0.000488281 relative) plus the 4-decimal rounding both sides apply (0.0002); `worst excess` is how far the largest difference sat OUTSIDE its own tolerance, so a negative number means every comparison was inside it. `cos` recomputes `mean_cos` from `cos.f16`; `sae_self` recomputes `mean_peak_act` / `max_peak_act` from `sae_self.f16`.*

| kind | source | family | comparisons | worst excess | outcome |
|---|---|---|---|---|---|
| cos_centred bo-k | 2026-09-10_rl-8x2048-full@vllm:mu-none | (all) | — | — | skipped: maemms/qwen36-27b/2026-09-10_rl-8x2048-full/scores/2026-09-21_v3_ours__mu-none__vllm has no cos_centred.f16 (the run centred on nothing) |
| cos_centred bo-k | 2026-09-10_rl-8x2048-full@vllm:mu-stats | (all) | 512 | -0.000265 | 3584 vs the stored `bo_c_k`, 0 mismatches; 0 rows have a NaN rollout (0 draws) |
| cos_centred bo-k | 2026-09-18_rl-last16-lr5e-7@vllm | (all) | 512 | -0.000268 | 3584 vs the stored `bo_c_k`, 0 mismatches; 0 rows have a NaN rollout (0 draws) |
| cos | 2026-09-10_rl-8x2048-full@vllm:mu-none | (all) | 512 | -0.000457 | 0 mismatches |
| cos | 2026-09-10_rl-8x2048-full@vllm:mu-stats | (all) | 1024 | -0.000371 | 0 mismatches |
| cos | 2026-09-18_rl-last16-lr5e-7@vllm | (all) | 1024 | -0.000363 | 0 mismatches |

CSV: `reader_check.csv`

### Sanity gates (plan §2.3)

*From `results/sanity.yaml`, which the user edits. A `FLAG` means the mu or the prompt may be wrong and the numbers above should not be read until it is explained; `absent` means the gate's selector does not resolve on this set, which is a normal outcome for a gate written against another set; `no verdict` means the YAML declares the two are not the same statistic. `n` is the number of rows or features behind `ours` — a wide miss at small `n` is a different statement from a wide miss at the paper's own draw size.*

| gate | family | source | metric | n | ours | expected | tol | verdict | why |
|---|---|---|---|---|---|---|---|---|---|
| card realact, rl-last16 bo4 centred | realact | 2026-09-18_rl-last16-lr5e-7@vllm | cos_centred.bo4 | 506 | 0.8157 | 0.7800 | 0.0500 | pass | \|0.8157 − 0.7800\| = 0.0357 vs tol 0.05 |
| card realact_long, rl-last16 bo4 centred | realact_long | 2026-09-18_rl-last16-lr5e-7@vllm | cos_centred.bo4 | — | — | 0.7160 | 0.0500 | absent | no `cos_centred.bo4` for family `realact_long` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| card 2M encoder fired fraction, rl-last16 | sae/sae2m/enc | 2026-09-18_rl-last16-lr5e-7@vllm | fired.item | — | — | 0.3440 | 0.0800 | absent | no `fired.item` for family `sae/sae2m/enc` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| card 2M decoder fired fraction, rl-last16 | sae/sae2m/dec | 2026-09-18_rl-last16-lr5e-7@vllm | fired.item | — | — | 0.4160 | 0.0800 | absent | no `fired.item` for family `sae/sae2m/dec` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| card 131k normalised activation, rl-last16 bo4 median ratio | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo4.median | — | — | 0.8240 | 0.1500 | absent | no `ratio.bo4.median` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 131k, rl-last16 bo1 median ratio | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.7600 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 131k, old primary mu-none bo1 median ratio | sae/l42-1b | 2026-09-10_rl-8x2048-full@vllm:mu-none | ratio.bo1.median | — | — | 0.8300 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/l42-1b` on `2026-09-10_rl-8x2048-full@vllm:mu-none` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 131k, rl-last16 feature firing fraction | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | fired.feature | — | — | 0.9200 | 0.1200 | absent | no `fired.feature` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 2M enc, rl-last16 bo1 median ratio | sae/sae2m/enc | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.2300 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/sae2m/enc` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 2M, rl-last16 bo1 median ratio (unsided sets) | sae/sae2m | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.2300 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/sae2m` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 2M enc, old primary mu-none bo1 median ratio | sae/sae2m/enc | 2026-09-10_rl-8x2048-full@vllm:mu-none | ratio.bo1.median | — | — | 0.1000 | 0.0800 | absent | no `ratio.bo1.median` for family `sae/sae2m/enc` on `2026-09-10_rl-8x2048-full@vllm:mu-none` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 2M enc, NLA bo1 median ratio | — | — | — | — | — | 0.2400 | 0.1000 | absent | `source: 2026-07-14_nla-av` matched 0 of the sources present (2026-09-10_rl-8x2048-full@vllm:mu-none, 2026-09-10_rl-8x2048-full@vllm:mu-stats, 2026-09-18_rl-last16-lr5e-7@vllm) |
| old primary mu-none, realact mean cos (2026-09-21 v1raw smoke, n=4) | realact | rl-8x2048-full:mu-none | cos_raw.bo1 | — | — | 0.8760 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| old primary mu-stats, realact mean cos (2026-09-21 v1raw smoke, n=4) | realact | rl-8x2048-full:mu-stats | cos_raw.bo1 | — | — | 0.8989 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| old primary mu-stats, realact bo4 (v1raw smoke) | realact | rl-8x2048-full:mu-stats | cos_raw.bo4 | — | — | 0.9339 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| old primary mu-none, realact bo4 (v1raw smoke) | realact | rl-8x2048-full:mu-none | cos_raw.bo4 | — | — | 0.9034 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| rl-last16 realact mean centred cos (v1raw smoke, n=4) | realact | rl-last16 | cos_centred.bo1 | — | — | 0.7518 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| eval1 rl-last16, realact mean cos_raw (n = 486) | realact | rl-last16 | cos_raw.bo1 | — | — | 0.8860 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_ours` |
| eval1 rl-last16, realact mean cos_centred (n = 486) | realact | rl-last16 | cos_centred.bo1 | — | — | 0.7590 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_ours` |
| eval1 rl-last16, ours realact mean cos_raw (n = 506) | realact | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | 506 | 0.8884 | 0.8884 | 0.0005 | pass | \|0.8884 − 0.8884\| = 0.0000 vs tol 0.0005 |
| eval1 rl-last16, ours realact mean cos_centred (n = 506) | realact | 2026-09-18_rl-last16-lr5e-7@vllm | cos_centred.bo1 | 506 | 0.7668 | 0.7668 | 0.0005 | pass | \|0.7668 − 0.7668\| = 0.0000 vs tol 0.0005 |
| eval1 old primary mu-stats, realact mean cos_raw (n = 486) | realact | rl-8x2048-full@vllm:mu-stats | cos_raw.bo1 | — | — | 0.8881 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_ours` |
| eval1 old primary mu-none, realact mean cos_raw (n = 486) | realact | rl-8x2048-full@vllm:mu-none | cos_raw.bo1 | — | — | 0.8830 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_ours` |
| eval1 old primary mu-stats, realact bo64 (n = 486) | realact | rl-8x2048-full@vllm:mu-stats | cos_raw.bo64 | — | — | 0.9322 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_ours` |
| eval1 rl-last16, subspace/bsf mean cos_raw | bsf | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.3177 | 0.0005 | absent | no `cos_raw.bo1` for family `bsf` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, subspace/jlens mean cos_raw | jlens | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.1058 | 0.0005 | absent | no `cos_raw.bo1` for family `jlens` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, realact_long mean cos_raw | realact_long | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.4360 | 0.0005 | absent | no `cos_raw.bo1` for family `realact_long` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, ctrl/random mean cos_raw | random | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.0338 | 0.0005 | absent | no `cos_raw.bo1` for family `random` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 NLA, realact mean cos_raw (n = 486) | realact | 2026-07-14_nla-av | cos_raw.bo1 | — | — | 0.8075 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_ours` |
| eval1 rl-last16, 2M enc median bo1 ratio (512 features) | sae/sae2m/enc | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.2390 | 0.0020 | absent | no `ratio.bo1.median` for family `sae/sae2m/enc` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, 131k median bo1 ratio (512 features) | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.7190 | 0.0020 | absent | no `ratio.bo1.median` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 old primary mu-none, 2M enc median bo1 ratio | sae/sae2m/enc | 2026-09-10_rl-8x2048-full@vllm:mu-none | ratio.bo1.median | — | — | 0.0950 | 0.0020 | absent | no `ratio.bo1.median` for family `sae/sae2m/enc` on `2026-09-10_rl-8x2048-full@vllm:mu-none` in this run (the family or the best-of-k is not present on this set) |
| eval1 NLA, 2M enc median bo1 ratio | — | — | — | — | — | 0.2330 | 0.0020 | absent | `source: 2026-07-14_nla-av` matched 0 of the sources present (2026-09-10_rl-8x2048-full@vllm:mu-none, 2026-09-10_rl-8x2048-full@vllm:mu-stats, 2026-09-18_rl-last16-lr5e-7@vllm) |
| old primary, sae rows, v1raw vs v1 on the shared rows | sae/l42-1b | rl-8x2048-full:mu-none | cos_raw.bo1 | — | — | — | 0.0100 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| old primary, realact rows, v1raw vs v1 on the shared rows | realact | rl-8x2048-full:mu-stats | cos_raw.bo1 | — | — | — | — | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_ours` |
| rl-last16, ctrl sae rows, v3_ctrl vs the 131k rows of v1raw | sae/l42-1b | rl-last16 | cos_raw.bo1 | — | — | — | — | absent | `set: 2026-09-21_v3_ctrl` — this run is `2026-09-21_v3_ours` |
| old primary v1 realact mean cos, 0.5076 | — | — | — | — | — | 0.5076 | — | absent | `source: rl-8x2048-full` matched 2 of the sources present (2026-09-10_rl-8x2048-full@vllm:mu-none, 2026-09-10_rl-8x2048-full@vllm:mu-stats, 2026-09-18_rl-last16-lr5e-7@vllm) |

CSV: `sanity.csv`

#### Where the expected numbers come from

- **card realact, rl-last16 bo4 centred** — Celeste's card for rl-last16 at bo4 on her 512 realact rows, centred, her scorer. infra/2026-09-21_evals-plan-main.md §2.3. DIFFERENT DENOMINATOR AND DIFFERENT DRAW: the tolerance is a STATED band, not a measured one -- her rows are her draw, her scorer is not this one, and our exclusions take the block to n = 486. A pass is consistency, not reproduction; a miss of this size is not by itself evidence of a wrong mu.
- **card realact_long, rl-last16 bo4 centred** — Same card, the realact_long block (plan §2.3). DIFFERENT DENOMINATOR, AND WORSE THAN THE REALACT ONE: `2026-09-21_v3_realact_long` is `storage: unit` with `family_mu: unknown`, so our centred column there is read at `whiten_mu` and not at the `mu_long` those rows were built under -- SMOKES.md 2026-09-21 records cos_raw 0.4360 against cos_centred 0.6991 on exactly this block, the one family where the centred number is far ABOVE the raw one. Read this line as "the labelled number is in the card's neighbourhood", never as a reproduction.
- **card 2M encoder fired fraction, rl-last16** — A firing rate has no denominator, so this is the one card number that does not depend on whose corpus peak is used -- the only gate here that is NOT a different-denominator comparison. It remains a different DRAW (her 512 features against our stratified 512). plan §2.3; reconstruction/sae_smoke64.py CARD["qwen36-27b/sae2m"].
- **card 2M decoder fired fraction, rl-last16** — "the 0.416 card gate needs it" -- plan §2.1 puts the decoder block in the set for exactly this line, and it is why `sae_self --sae-side dec` exists. Same denominator-free rate as the encoder row above, same different-draw caveat. The PAIR is the finding: her card reads the decoder side HIGHER than the encoder side (0.416 vs 0.344), so the two gates should miss in the same direction or not at all.
- **card 131k normalised activation, rl-last16 bo4 median ratio** — Her card's `norm_act` for rl-last16 at bo4 on her 512-feature 131k set (reconstruction/sae_smoke64.py CARD["qwen36-27b/l42-1b"], quoted in sae_smoke64.md). DIFFERENT DENOMINATOR, and this one is not a band -- it is a different quantity: her ratio divides by her corpus peak from the 1.0B-token scan and ours divides by OUR 16M `max_act` (sae_self.py:383), which is a smaller corpus and therefore a smaller peak and a larger ratio. `compare: false` so both numbers print with no verdict; a tolerance here would assert a comparability that does not exist. `corpus_peak_1b` is on the rows if anyone wants to compute her denominator properly, which is a separate piece of work.
- **sae_smoke64 131k, rl-last16 bo1 median ratio** — reconstruction/sae_smoke64.md headline, 64 features, n=16, HF engine. Same denominator as ours (16M max_act); the band covers the different draw and the n = 16 vs 64 gap. SMOKES.md 2026-09-21 reports the 512-feature run at 0.719 against this 0.76.
- **sae_smoke64 131k, old primary mu-none bo1 median ratio** — sae_smoke64.md, the EXISTING vLLM product at n=64 over 64 of the 512 sae rows of 2026-09-16_v1. A run of this driver over a different row subset of the same dictionary is a different draw of features; the band is wide for that reason. TAG NAMED: the old primary has two arms on every eval-1 set, and `sae`/`random` rows are not centrable, so the two arms' products for this family are bit-identical -- `mu-none` is the one that exists everywhere. SMOKES.md 2026-09-21: 0.803 at 512 features.
- **sae_smoke64 131k, rl-last16 feature firing fraction** — sae_smoke64.md, "features firing 92%" on the 131k side. A rate, so no denominator question. SMOKES.md 2026-09-21: 0.932 at 512 features.
- **sae_smoke64 2M enc, rl-last16 bo1 median ratio** — sae_smoke64.md, 2M side, set 2026-09-21_sae2m_64, 16 per quartile. Our 16M denominator on both sides. Family is `sae/sae2m/enc` because the v3 set carries `sae_side` and splits; on the 64-feature smoke root (which does not) the same gate resolves as `sae/sae2m`. SMOKES.md 2026-09-21: 0.239 at 512 features.
- **sae_smoke64 2M, rl-last16 bo1 median ratio (unsided sets)** — The same gate for a set whose rows carry no `sae_side` (`2026-09-21_sae2m_64`, `2026-09-20_sae2m_2k`), where the family label has no side component. Exactly one of this gate and the `/enc` one above resolves on any given set; the other is `absent`, which is the correct outcome and not a failure.
- **sae_smoke64 2M enc, old primary mu-none bo1 median ratio** — sae_smoke64.md 2M side, old primary at n=16 HF. SMOKES.md 2026-09-21 reads 0.095 at 512 features against that 0.102. The band is wide because n = 16 there and 64 here.
- **sae_smoke64 2M enc, NLA bo1 median ratio** — sae_smoke64.md 2M side, NLA at n=4, 200 tokens, 256-token scoring window. SMOKES.md 2026-09-21 reads 0.233 at 512 features against that 0.242. NO RUN TAG: on the v3 sets the NLA products sit under `scores/<set>` with none; the `amp-exact` gate below is the v1raw shape of the same arm.
- **old primary mu-none, realact mean cos (2026-09-21 v1raw smoke, n=4)** — SMOKES.md 2026-09-21, "the two old-primary arms": realact rows 0-3 of 2026-09-21_v1raw, n=4, HF, --mu none. Resolves on v1raw only; `absent` on every v3 set, where the same arm's number is the 486-row one below.
- **old primary mu-stats, realact mean cos (2026-09-21 v1raw smoke, n=4)** — SMOKES.md 2026-09-21, the same table's stats_mu arm. v1raw only.
- **old primary mu-stats, realact bo4 (v1raw smoke)** — SMOKES.md 2026-09-21, the same table's bo_4 column. v1raw only.
- **old primary mu-none, realact bo4 (v1raw smoke)** — SMOKES.md 2026-09-21, the same table's bo_4 column. v1raw only.
- **rl-last16 realact mean centred cos (v1raw smoke, n=4)** — SMOKES.md 2026-09-21, "cos_centred, first numbers", rl-last16 under whiten_mu on 2026-09-21_v1raw rows 0-3. v1raw only; on `2026-09-21_v3_realact` the recorded number is 0.7590 over 486 rows (gate below).
- **eval1 rl-last16, realact mean cos_raw (n = 486)** — SMOKES.md 2026-09-21 summary table, rl-last16 x `2026-09-21_v3_realact`, n = 486 after the 26-row `hers_n7.exclude.json` list -- which this driver now applies from the set's own `exclusions.json`. Same rows, same product, a different reader.
- **eval1 rl-last16, realact mean cos_centred (n = 486)** — SMOKES.md 2026-09-21 summary table, same row, centred column. NOTE that SMOKES read this from `per_target.jsonl`'s `mean_cos_centred` and this driver recomputes the whole centred ladder from `cos_centred.f16`; the two agreeing to 4 dp is the point of the gate.
- **eval1 rl-last16, ours realact mean cos_raw (n = 506)** — SMOKES.md 2026-09-21 summary table, the `ours` block -- our own v1 realact draw, 6 rows excluded by a DIFFERENT instrument (n=13 full reproduction in her v2 text) from the 26 of the `_realact` block.
- **eval1 rl-last16, ours realact mean cos_centred (n = 506)** — SMOKES.md 2026-09-21 summary table, the `ours` block, centred column.
- **eval1 old primary mu-stats, realact mean cos_raw (n = 486)** — SMOKES.md 2026-09-21 summary table, and the arm the "old primary's mu, SETTLED" section declares the winner (+0.0051 over mu-none, doc-clustered 95% CI [+0.0030, +0.0072]).
- **eval1 old primary mu-none, realact mean cos_raw (n = 486)** — SMOKES.md 2026-09-21 summary table, the losing arm of the same pair.
- **eval1 old primary mu-stats, realact bo64 (n = 486)** — SMOKES.md 2026-09-21, "The old primary's mu, SETTLED", bo_64 row.
- **eval1 rl-last16, subspace/bsf mean cos_raw** — SMOKES.md 2026-09-21 summary table. `bsf` has NO exclusions -- all 512 rows in both -- so this one IS a rounding-tolerance gate and a miss is a reader defect.
- **eval1 rl-last16, subspace/jlens mean cos_raw** — SMOKES.md 2026-09-21 summary table. All 512 rows, no exclusions.
- **eval1 rl-last16, realact_long mean cos_raw** — SMOKES.md 2026-09-21 summary table. All 512 rows, no exclusions.
- **eval1 rl-last16, ctrl/random mean cos_raw** — SMOKES.md 2026-09-21 summary table. All 512 rows, no exclusions.
- **eval1 NLA, realact mean cos_raw (n = 486)** — SMOKES.md 2026-09-21 summary table, NLA n=4 on `2026-09-21_v3_realact`, n = 486 after exclusions -- which this driver now applies.
- **eval1 rl-last16, 2M enc median bo1 ratio (512 features)** — SMOKES.md 2026-09-21 SAE table, rl-last16 x 2M, `med bo1 ratio` over the 512 ENCODER features. Same rows, same product, a different reader: a rounding tolerance.
- **eval1 rl-last16, 131k median bo1 ratio (512 features)** — SMOKES.md 2026-09-21 SAE table, rl-last16 x 131k l42-1b. Rounding tolerance.
- **eval1 old primary mu-none, 2M enc median bo1 ratio** — SMOKES.md 2026-09-21 SAE table. Rounding tolerance.
- **eval1 NLA, 2M enc median bo1 ratio** — SMOKES.md 2026-09-21 SAE table, NLA n=4. Rounding tolerance.
- **old primary, sae rows, v1raw vs v1 on the shared rows** — The `sae` family is not centrable, so both sets store unit(W_enc[:, f]) and `cos_raw` is the same statistic on both. The two runs differ in engine (vLLM there, HF here), n (64 vs 4) and nothing else, so a difference is sampling, not convention. The tolerance is loose because n = 4 here.
- **old primary, realact rows, v1raw vs v1 on the shared rows** — NOT the same statistic, and printed for the record only. `2026-09-16_v1` is `storage: unit` with its realact rows already centred on stats/mu.f32, so its `cos` has a CENTRED target and a raw scorer -- Celeste's asymmetry -- while a raw set's `cos` centres neither side. A tolerance here would assert a comparability that does not exist.
- **rl-last16, ctrl sae rows, v3_ctrl vs the 131k rows of v1raw** — `2026-09-21_v3_ctrl` was built `--dirs-from …/2026-09-21_v1raw --rows 512-1535`, so its sae rows ARE v1raw's rows 1024-1535 -- but RENUMBERED, and this gate intersects on the row INDEX. So it compares v3_ctrl row r against v1raw row r, which are different targets, and it is `compare: false` for that reason: it exists to print the two numbers side by side while a row-mapping gate does not exist. `src_set` / `src_row` are on the v3_ctrl rows and are what a proper version of this would join on.
- **old primary v1 realact mean cos, 0.5076** — SMOKES.md 2026-09-15, rows 0-7 of 2026-09-16_v1. NOT THE SAME STATISTIC as a raw set's `cos`: the stored 0.5076 is cos(h, unit(act - stats_mu)) -- target centred, scorer raw, Celeste's asymmetry -- and on a `storage: raw` set `score` produces cos (neither side centred) and cos_centred (both), never that third column. SMOKES.md 2026-09-21 records this as plan gate 2 being unevaluable rather than failed. Printed for the record with no verdict; delete the line or restate the gate against cos_centred once that is decided. The BARE selector is deliberate: on a set with two old-primary arms this resolves `absent` (matched 2 of the sources), which is right -- the 0.5076 belongs to neither arm.

## Figures

- `figures/arms_qwen36-27b_2026-09-10_rl-8x2048-full_vllm.pdf` / `.png`
- `figures/bok_realact.pdf` / `.png`

## Skipped and absent

- `qwen36-27b/2026-07-14_nla-av`: no scores directory for `2026-09-21_v3_ours` under `maemms/qwen36-27b/2026-07-14_nla-av/scores/`
- `qwen36-27b/2026-09-08_rlI-150`: no scores directory for `2026-09-21_v3_ours` under `maemms/qwen36-27b/2026-09-08_rlI-150/scores/`
- `qwen36-27b/2026-09-16_base-control`: no scores directory for `2026-09-21_v3_ours` under `maemms/qwen36-27b/2026-09-16_base-control/scores/`
