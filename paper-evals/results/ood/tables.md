# OOD generalisation: `2026-09-23_ood_full`

Base `qwen36-27b`, 22 arms x 512 targets, in-domain corpus search at **4M, 10M tokens**, PER ARM -- each arm at the size its own scan reached, printed in its `size` column.

Δ = the MAEMM's unbiased best-of-64 minus the in-domain corpus search's top-1, paired per target; CI is the design's 10,000-resample percentile bootstrap over the arm's targets. The outcome is three-state (review R9): **exceeds** (CI above zero), **inconclusive** (CI covers zero -- a failure to reject, not a finding of no generalisation), **reversed** (CI below zero).

Run under `MODAL_PROFILE=maemms`.

### Arms — 2026-09-18_rl-last16-lr5e-7@vllm:paper0923

*bo8 against the in-domain corpus search AT EACH ARM'S OWN SCANNED SIZE (the `size (M)` column; this run holds 4M, 10M), paired per target. BOTH SIDES ARE THE CENTRED CONVENTION -- `cos(h - mu, unit(act - mu))`, spec §2's 'centred against centred' -- which holds ONLY IF the scan behind the corpus column ran with `--centre`; nothing in this file can verify that, and an uncentred scan of the same set at the same mean lands in the same cell. `bo8 (centred)` is RECOMPUTED from `cos_centred.f16` -- no `bo_c_8` column is stored anywhere -- and an em dash there is a row with fewer than 8 finite centred draws, never a zero. The asymmetric and raw cosines are in the CSV and are read by nothing. Δ AND THE VERDICT ARE THE bo8 PAIR (M5, 2026-09-23): bo8 minus the same target's corpus top-1, which is spec section 2's headline contrast; the bo64 Δ the quarter-scale run reported is `delta`/`outcome` in the CSV, beside it, because the two are different quantities and not a disagreement. The control column is `2026-09-16_base-control@vllm:paper0923`. `lang / code` is the rate at which the top-1 rollout comes back in the arm's own language (fastText lid218e), or the code-like rate where fastText is not meaningful. **READ THE CONVENTION NOTE BELOW BEFORE COMPARING Δ WITH THE DESIGN'S +0.218.***

| arm | family | n | size (M) | bo8 (centred) | bo64 (centred) | corpus top-1 | control bo8 | control bo64 | Δ | 95% CI | win | outcome | lang / ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c | code | 512 | 10 | 0.7371 | 0.7810 | 0.7109 | 0.2178 | 0.2775 | 0.0262 | [0.013, 0.039] | 0.54 | exceeds | n/m (0.01 vs ceiling 0.13) |
| go | code | 512 | 10 | 0.6965 | 0.7428 | 0.7137 | 0.2164 | 0.2723 | -0.0172 | [-0.030, -0.005] | 0.41 | reversed | n/m (0.00 vs ceiling 0.01) |
| haskell | code | 512 | 10 | 0.7076 | 0.7530 | 0.7209 | 0.2018 | 0.2542 | -0.0132 | [-0.024, -0.002] | 0.41 | reversed | n/m (0.02 vs ceiling 0.08) |
| javascript | code | 512 | 10 | 0.6993 | 0.7465 | 0.6694 | 0.2233 | 0.2818 | 0.0299 | [0.016, 0.044] | 0.58 | exceeds | n/m (0.05 vs ceiling 0.11) |
| python | code | 512 | 10 | 0.7190 | 0.7685 | 0.6713 | 0.2216 | 0.2839 | 0.0477 | [0.034, 0.061] | 0.62 | exceeds | n/m (0.01 vs ceiling 0.05) |
| rust | code | 512 | 10 | 0.7023 | 0.7482 | 0.6889 | 0.2077 | 0.2615 | 0.0134 | [0.001, 0.026] | 0.47 | exceeds | n/m (0.07 vs ceiling 0.22) |
| shell | code | 512 | 4 | 0.7229 | 0.7661 | 0.7119 | 0.2393 | 0.3044 | 0.0110 | [-0.001, 0.023] | 0.53 | inconclusive | n/m (0.00 vs ceiling 0.03) |
| sql | code | 512 | 10 | 0.7008 | 0.7462 | 0.7161 | 0.2397 | 0.3020 | -0.0153 | [-0.028, -0.002] | 0.40 | reversed | n/m (0.00 vs ceiling 0.01) |
| ufw_en | ctrl | 512 | 10 | 0.8277 | 0.8545 | 0.5857 | 0.1882 | 0.2649 | 0.2420 | [0.232, 0.252] | 0.97 | exceeds | 0.990 / 0.86 |
| ufw_zh | ctrl | 512 | 10 | 0.8091 | 0.8447 | 0.6260 | 0.1835 | 0.2432 | 0.1831 | [0.172, 0.194] | 0.92 | exceeds | 0.725 / 0.71 |
| arb_Arab | lang | 512 | 10 | 0.7816 | 0.8269 | 0.7096 | 0.1927 | 0.2545 | 0.0720 | [0.062, 0.082] | 0.77 | exceeds | 0.707 / 0.84 |
| ces_Latn | lang | 512 | 10 | 0.7731 | 0.8204 | 0.6884 | 0.1788 | 0.2379 | 0.0847 | [0.074, 0.095] | 0.77 | exceeds | 0.600 / 0.85 |
| cmn_Hani | lang | 512 | 10 | 0.8071 | 0.8428 | 0.6595 | 0.1970 | 0.2610 | 0.1476 | [0.136, 0.159] | 0.85 | exceeds | 0.754 / 0.88 |
| ell_Grek | lang | 512 | 10 | 0.7148 | 0.7724 | 0.7036 | 0.1854 | 0.2424 | 0.0112 | [-0.000, 0.023] | 0.54 | inconclusive | 0.355 / 0.84 |
| hin_Deva | lang | 512 | 10 | 0.7523 | 0.8158 | 0.7850 | 0.2097 | 0.2614 | -0.0327 | [-0.041, -0.025] | 0.34 | reversed | 0.703 / 0.63 |
| jpn_Jpan | lang | 512 | 10 | 0.7845 | 0.8246 | 0.6914 | 0.2038 | 0.2675 | 0.0931 | [0.084, 0.102] | 0.83 | exceeds | 0.590 / 0.55 |
| rus_Cyrl | lang | 512 | 10 | 0.8026 | 0.8374 | 0.6712 | 0.1953 | 0.2572 | 0.1314 | [0.122, 0.141] | 0.90 | exceeds | 0.773 / 0.79 |
| tha_Thai | lang | 512 | 10 | 0.7420 | 0.8046 | 0.7906 | 0.2167 | 0.2693 | -0.0486 | [-0.059, -0.038] | 0.32 | reversed | 0.779 / 0.72 |
| arxiv | math | 512 | 10 | 0.7891 | 0.8237 | 0.6533 | 0.2038 | 0.2698 | 0.1358 | [0.124, 0.148] | 0.85 | exceeds | n/m (0.00 vs ceiling 0.01) |
| isabelle | math | 512 | 10 | 0.6821 | 0.7258 | 0.7710 | 0.2218 | 0.2751 | -0.0889 | [-0.099, -0.079] | 0.14 | reversed | n/m (0.01 vs ceiling 0.02) |
| lean | math | 512 | 10 | 0.7317 | 0.7755 | 0.7923 | 0.1998 | 0.2512 | -0.0606 | [-0.071, -0.050] | 0.21 | reversed | n/m (0.00 vs ceiling 0.07) |
| owm | math | 512 | 10 | 0.7731 | 0.8092 | 0.6009 | 0.2053 | 0.2687 | 0.1722 | [0.159, 0.185] | 0.86 | exceeds | 0.953 / 0.75 |

CSV: `arms_2026-09-18_rl-last16-lr5e-7_at_vllm__paper0923.csv`

### Pre-registered claim — 2026-09-18_rl-last16-lr5e-7@vllm:paper0923

Level 1, design §0: *on every arm, the best of 8 MAEMM rollouts aligns with the target more closely than the best window of an in-domain corpus search in the target's own domain.* Read at **each arm's own scanned size** (4M, 10M), over the 21 arms of the conjunction (design §6: lang 8, code 8, math 4, `ufw_zh`; the `diag` arm `formulas` and the §8(a) English pipeline check `ufw_en` are reported as rows but not counted):

**12 of 21 arms exceed.**  Not exceeding: `go` (reversed), `haskell` (reversed), `shell` (inconclusive), `sql` (reversed), `ell_Grek` (inconclusive), `hin_Deva` (reversed), `tha_Thai` (reversed), `isabelle` (reversed), `lean` (reversed).

### English in-distribution reference (review R1)

*Recomputed from `scan/2026-09-16_v1/topk.jsonl`. `top1 (no own doc)` EXCLUDES corpus windows from the target's own document; the OOD corpora contain no target documents (design §2), so that is the like-for-like reference, and the paper's frozen 0.371 at 4M counts the own document and stays in its own table. The margin is 0.569 (the paper's English bo64, design §0) minus that column.*

| size (M) | n | top1 (all) | top1 (no own doc) | margin vs bo64 0.569 | own doc IS top-1 | no non-own candidate |
|---|---|---|---|---|---|---|
| 1 | 512 | 0.3216 | 0.3137 | 0.2553 | 38 | 0 |
| 2 | 512 | 0.3433 | 0.3315 | 0.2375 | 66 | 0 |
| 4 | 512 | 0.3706 | 0.3511 | 0.2179 | 114 | 2 |
| 8 | 512 | 0.3997 | 0.3672 | 0.2018 | 209 | 4 |
| 16 | 512 | 0.4105 | 0.3851 | 0.1839 | 168 | 4 |

CSV: `english_reference.csv`

### The cosine convention, and what Δ here is and is not

SINCE M0a (2026-09-23) BOTH SIDES ARE CENTRED, and about the SAME constant. `score` centres on the scoring constant -- `bases.<base>.whiten_mu`, read by `precompute.common.score_mu`, the one mean both arguments of every centred cosine are taken about and deliberately decoupled from each MAEMM's injection `mu:` -- and `precompute/scan.py --centre` subtracts the same constant from every corpus window before the dot product. `cos_centred` on the MAEMM side against a `--centre` scan's top-1 is therefore one convention applied to both sides, which is what makes Δ a paired difference rather than the subtraction of two angles to two different vectors. Second sentence of the same fact: this file no longer reads `cos_asym` at all, and `bo_a_64` / `cos_asym.f16` remain on disk read by no driver.

WHAT THIS FILE CANNOT CHECK, stated because the failure is silent. `scan_dirs_of` selects scans by SET, and `top1_by_corpus` is keyed on (corpus, mean) alone: a scan run WITHOUT `--centre`, at the same mean and on the same set, lands in exactly the same cell and would be differenced without a word. Nothing in a scan's `topk.jsonl` distinguishes the two modes; the scan's own README states which mode it ran in, and the operational guard is a set directory that only centred scans ever wrote. Before M0a the window side was uncentred while the target side was centred, so a `centred` read was meaningless and `cos_asym` existed precisely to work around it (`results/ood/tables.md:103-105`, `SMOKES.md:4483-4490`).

Consequences, stated rather than smoothed:

- the paper's frozen English numbers are ASYMMETRIC-convention numbers (bo64 0.569, corpus 0.351 at 4M, margin +0.218 / +0.256 at 4M / 1M), so neither the bo8/bo64 columns above nor Δ is comparable with them by MAGNITUDE. The English reference table above is recomputed from the frozen 2026-09-16 UNCENTRED scan and stays in its own convention on purpose;
- the per-arm claim is a SIGN ('the best of k rollouts aligns more closely than the best corpus window'), and a sign is testable under any one convention applied to both sides -- which is what the outcome column reports;
- `bo8 (centred)` is RECOMPUTED from `cos_centred.f16` through the pipeline's one best-of-k estimator (`results.common.bo_unbiased`, the unbiased order statistic over all finite draws), because `score` stores the centred ladder at k = 64 only and no `bo_c_8` column exists anywhere. A row with fewer than 8 finite centred draws carries NO bo8 and prints an em dash rather than a clamped k, which would be a different quantity under the same name;
- all three cosines are in the CSV, so the arms can be re-read under any of them without re-running anything on the GPU.

## Skipped and absent

- config'd checkpoint with no products on this set: `qwen36-27b/2026-07-14_nla-av`
- config'd checkpoint with no products on this set: `qwen36-27b/2026-09-08_rlI-150`
- config'd checkpoint with no products on this set: `qwen36-27b/2026-09-10_rl-8x2048-full`
- scans read: `2026-09-23_ood_full__arb_Arab__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__arxiv__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__c__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__ces_Latn__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__cmn_Hani__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__ell_Grek__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__go__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__haskell__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__hin_Deva__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__isabelle__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__javascript__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__jpn_Jpan__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__lean__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__owm__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__python__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__rus_Cyrl__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__rust__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__shell__4m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__sql__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__tha_Thai__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__ufw_en__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-23_ood_full__ufw_zh__10m__paper0923` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy
- control `2026-09-16_base-control@vllm:paper0923` is `qwen36-27b/2026-09-16_base-control`, which config.yaml declares at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy; the run itself recorded mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy
- control `2026-09-16_base-control@vllm:paper0923`: bo8 (centred) recomputed from `cos_centred.f16` on 11264 rows
- base predictability covariate from `base/qwen36-27b/nll/2026-09-23_ood_full/per_target.jsonl`, 11264 targets over 22 arms (bits per byte of the arm's own text under the untrained base)
- random floor per corpus: the scan's own per-target window quantiles at EACH ARM'S OWN size, 22 arms over 22 scans. NOT the target-vs-target `chance_pairwise_cos` of `stats_ood`, which measures the target geometry and not the corpus
- scan sizes read from each scan's own README and cross-checked against its topk.jsonl: `2026-09-23_ood_full__arb_Arab__10m__paper0923` at 10M, `2026-09-23_ood_full__arxiv__10m__paper0923` at 10M, `2026-09-23_ood_full__c__10m__paper0923` at 10M, `2026-09-23_ood_full__ces_Latn__10m__paper0923` at 10M, `2026-09-23_ood_full__cmn_Hani__10m__paper0923` at 10M, `2026-09-23_ood_full__ell_Grek__10m__paper0923` at 10M, `2026-09-23_ood_full__go__10m__paper0923` at 10M, `2026-09-23_ood_full__haskell__10m__paper0923` at 10M, `2026-09-23_ood_full__hin_Deva__10m__paper0923` at 10M, `2026-09-23_ood_full__isabelle__10m__paper0923` at 10M, `2026-09-23_ood_full__javascript__10m__paper0923` at 10M, `2026-09-23_ood_full__jpn_Jpan__10m__paper0923` at 10M, `2026-09-23_ood_full__lean__10m__paper0923` at 10M, `2026-09-23_ood_full__owm__10m__paper0923` at 10M, `2026-09-23_ood_full__python__10m__paper0923` at 10M, `2026-09-23_ood_full__rus_Cyrl__10m__paper0923` at 10M, `2026-09-23_ood_full__rust__10m__paper0923` at 10M, `2026-09-23_ood_full__shell__4m__paper0923` at 4M, `2026-09-23_ood_full__sql__10m__paper0923` at 10M, `2026-09-23_ood_full__tha_Thai__10m__paper0923` at 10M, `2026-09-23_ood_full__ufw_en__10m__paper0923` at 10M, `2026-09-23_ood_full__ufw_zh__10m__paper0923` at 10M
- EVERY in-domain scan recorded `--centre` (both sides about the scoring constant), so the centred-against-centred convention of spec §2 is a READ FACT here and not an assumption
- `2026-09-18_rl-last16-lr5e-7@vllm:paper0923`: bo8 (centred) recomputed from `cos_centred.f16` on 11264 rows
- `2026-09-18_rl-last16-lr5e-7@vllm:paper0923`: language-id taken on the top-1 BY SCORE (centred); max |check| 0.00024; rollouts read as 22 `__rows` chunks of one product
- cells rows for `paper/numbers/cells.csv` written to `cells_2026-09-18_rl-last16-lr5e-7_at_vllm__paper0923.csv`
