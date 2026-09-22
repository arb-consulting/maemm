# OOD generalisation: `2026-09-21_ood_q1`

Base `qwen36-27b`, 23 arms x 16 targets, in-domain corpus search at **1M tokens**.

Δ = the MAEMM's unbiased best-of-64 minus the in-domain corpus search's top-1, paired per target; CI is the design's 10,000-resample percentile bootstrap over the arm's targets. The outcome is three-state (review R9): **exceeds** (CI above zero), **inconclusive** (CI covers zero -- a failure to reject, not a finding of no generalisation), **reversed** (CI below zero).

Run under `MODAL_PROFILE=maemms`.

### Arms — 2026-09-10_rl-8x2048-full@vllm:asym

*bo64 against the in-domain 1M corpus search, paired per target. BOTH SIDES ARE THE ASYMMETRIC CONVENTION -- `cos(h, unit(act - mu))`, uncentred scorer against the centred target -- which is what `scan` computes for a corpus window and what the paper's bo64 0.569 and corpus 0.351 are stated in. The symmetric cosines (`cos_centred`, `cos`) are in the CSV. The control column is `2026-09-16_base-control@vllm`. `lang / code` is the rate at which the top-1 rollout comes back in the arm's own language (fastText lid218e), or the code-like rate where fastText is not meaningful. **READ THE CONVENTION NOTE BELOW BEFORE COMPARING Δ WITH THE DESIGN'S +0.218.***

| arm | family | n | bo64 (asym) | corpus 1M | control bo64 | Δ | 95% CI | win | outcome | lang / ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| c | code | 16 | 0.4056 | 0.3002 | 0.0083 | 0.1054 | [0.053, 0.158] | 0.88 | exceeds | n/m (0.00 vs ceiling 0.14) |
| go | code | 16 | 0.3305 | 0.2798 | -0.0133 | 0.0507 | [0.011, 0.094] | 0.69 | exceeds | n/m (0.00 vs ceiling 0.01) |
| haskell | code | 16 | 0.3896 | 0.3256 | 0.0160 | 0.0640 | [0.017, 0.111] | 0.62 | exceeds | n/m (0.00 vs ceiling 0.10) |
| javascript | code | 16 | 0.3332 | 0.2068 | -0.0309 | 0.1265 | [0.080, 0.180] | 0.94 | exceeds | n/m (0.06 vs ceiling 0.22) |
| python | code | 16 | 0.3584 | 0.2124 | 0.0079 | 0.1460 | [0.053, 0.250] | 0.88 | exceeds | n/m (0.06 vs ceiling 0.07) |
| rust | code | 16 | 0.3208 | 0.1954 | -0.0200 | 0.1254 | [0.069, 0.180] | 0.88 | exceeds | n/m (0.00 vs ceiling 0.26) |
| shell | code | 16 | 0.4018 | 0.2731 | 0.0220 | 0.1287 | [0.075, 0.192] | 0.88 | exceeds | n/m (0.00 vs ceiling 0.02) |
| sql | code | 16 | 0.3482 | 0.3299 | 0.0221 | 0.0183 | [-0.021, 0.060] | 0.62 | inconclusive | n/m (0.00 vs ceiling 0.01) |
| ufw_en | ctrl | 16 | 0.5592 | 0.3022 | 0.1206 | 0.2569 | [0.210, 0.313] | 1.00 | exceeds | 1.000 / 0.83 |
| ufw_zh | ctrl | 16 | 0.5713 | 0.3567 | 0.1553 | 0.2147 | [0.168, 0.266] | 1.00 | exceeds | 1.000 / 0.86 |
| formulas | diag | 16 | 0.3609 | 0.3258 | 0.0294 | 0.0351 | [-0.003, 0.074] | 0.81 | inconclusive | n/m (0.00 vs ceiling 0.00) |
| arb_Arab | lang | 16 | 0.5043 | 0.3306 | 0.0799 | 0.1737 | [0.133, 0.214] | 1.00 | exceeds | 0.812 / 0.87 |
| ces_Latn | lang | 16 | 0.4532 | 0.3027 | 0.0425 | 0.1505 | [0.104, 0.197] | 0.94 | exceeds | 0.812 / 0.93 |
| cmn_Hani | lang | 16 | 0.5215 | 0.3516 | 0.1082 | 0.1699 | [0.121, 0.217] | 0.94 | exceeds | 0.938 / 0.97 |
| ell_Grek | lang | 16 | 0.3900 | 0.2543 | -0.0360 | 0.1357 | [0.096, 0.180] | 1.00 | exceeds | 0.750 / 0.87 |
| hin_Deva | lang | 16 | 0.3374 | 0.2927 | -0.0185 | 0.0447 | [0.006, 0.080] | 0.81 | exceeds | 0.875 / 0.73 |
| jpn_Jpan | lang | 16 | 0.5307 | 0.3810 | 0.1305 | 0.1497 | [0.103, 0.191] | 0.94 | exceeds | 0.562 / 0.66 |
| rus_Cyrl | lang | 16 | 0.5243 | 0.3022 | 0.0897 | 0.2220 | [0.179, 0.267] | 1.00 | exceeds | 0.688 / 0.86 |
| tha_Thai | lang | 16 | 0.3900 | 0.3539 | 0.0125 | 0.0361 | [-0.019, 0.099] | 0.62 | inconclusive | 0.688 / 0.84 |
| arxiv | math | 16 | 0.5287 | 0.3298 | 0.1132 | 0.1989 | [0.146, 0.254] | 1.00 | exceeds | n/m (0.00 vs ceiling 0.00) |
| isabelle | math | 16 | 0.3429 | 0.3771 | 0.0000 | -0.0342 | [-0.069, 0.002] | 0.31 | inconclusive | n/m (0.00 vs ceiling 0.03) |
| lean | math | 16 | 0.3832 | 0.3526 | -0.0034 | 0.0306 | [-0.022, 0.092] | 0.56 | inconclusive | n/m (0.00 vs ceiling 0.07) |
| owm | math | 16 | 0.4731 | 0.2739 | 0.0801 | 0.1992 | [0.144, 0.254] | 1.00 | exceeds | 1.000 / 0.82 |

CSV: `arms_2026-09-10_rl-8x2048-full_at_vllm__asym.csv`

### Pre-registered claim — 2026-09-10_rl-8x2048-full@vllm:asym

Level 1, design §0: *on every arm, the best of 64 MAEMM rollouts aligns with the target more closely than the best window of an in-domain corpus search in the target's own domain.* Read at **1M** corpus tokens, over the 21 arms of the conjunction (design §6: lang 8, code 8, math 4, `ufw_zh`; the `diag` arm `formulas` and the §8(a) English pipeline check `ufw_en` are reported as rows but not counted):

**17 of 21 arms exceed.**  Not exceeding: `sql` (inconclusive), `tha_Thai` (inconclusive), `isabelle` (inconclusive), `lean` (inconclusive).

### Arms — 2026-09-18_rl-last16-lr5e-7@vllm:asym

*bo64 against the in-domain 1M corpus search, paired per target. BOTH SIDES ARE THE ASYMMETRIC CONVENTION -- `cos(h, unit(act - mu))`, uncentred scorer against the centred target -- which is what `scan` computes for a corpus window and what the paper's bo64 0.569 and corpus 0.351 are stated in. The symmetric cosines (`cos_centred`, `cos`) are in the CSV. The control column is `2026-09-16_base-control@vllm`. `lang / code` is the rate at which the top-1 rollout comes back in the arm's own language (fastText lid218e), or the code-like rate where fastText is not meaningful. **READ THE CONVENTION NOTE BELOW BEFORE COMPARING Δ WITH THE DESIGN'S +0.218.***

| arm | family | n | bo64 (asym) | corpus 1M | control bo64 | Δ | 95% CI | win | outcome | lang / ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| c | code | 16 | 0.4311 | 0.3198 | 0.0083 | 0.1113 | [0.056, 0.168] | 0.81 | exceeds | n/m (0.00 vs ceiling 0.14) |
| go | code | 16 | 0.3705 | 0.2987 | -0.0133 | 0.0718 | [0.030, 0.118] | 0.69 | exceeds | n/m (0.00 vs ceiling 0.01) |
| haskell | code | 16 | 0.4165 | 0.3444 | 0.0160 | 0.0721 | [0.026, 0.119] | 0.75 | exceeds | n/m (0.00 vs ceiling 0.10) |
| javascript | code | 16 | 0.3574 | 0.2321 | -0.0309 | 0.1253 | [0.068, 0.187] | 0.81 | exceeds | n/m (0.12 vs ceiling 0.22) |
| python | code | 16 | 0.3908 | 0.2333 | 0.0079 | 0.1576 | [0.075, 0.250] | 0.81 | exceeds | n/m (0.12 vs ceiling 0.07) |
| rust | code | 16 | 0.3314 | 0.2244 | -0.0200 | 0.1070 | [0.049, 0.165] | 0.81 | exceeds | n/m (0.06 vs ceiling 0.26) |
| shell | code | 16 | 0.4458 | 0.2979 | 0.0220 | 0.1479 | [0.099, 0.204] | 1.00 | exceeds | n/m (0.00 vs ceiling 0.02) |
| sql | code | 16 | 0.3743 | 0.3558 | 0.0221 | 0.0186 | [-0.017, 0.055] | 0.56 | inconclusive | n/m (0.00 vs ceiling 0.01) |
| ufw_en | ctrl | 16 | 0.5635 | 0.3236 | 0.1206 | 0.2398 | [0.194, 0.293] | 1.00 | exceeds | 1.000 / 0.83 |
| ufw_zh | ctrl | 16 | 0.5827 | 0.3775 | 0.1553 | 0.2052 | [0.155, 0.264] | 1.00 | exceeds | 0.750 / 0.86 |
| formulas | diag | 16 | 0.3830 | 0.3517 | 0.0294 | 0.0313 | [0.000, 0.061] | 0.81 | exceeds | n/m (0.00 vs ceiling 0.00) |
| arb_Arab | lang | 16 | 0.5085 | 0.3501 | 0.0799 | 0.1584 | [0.114, 0.199] | 0.94 | exceeds | 0.625 / 0.87 |
| ces_Latn | lang | 16 | 0.4598 | 0.3300 | 0.0425 | 0.1298 | [0.076, 0.181] | 0.88 | exceeds | 0.500 / 0.93 |
| cmn_Hani | lang | 16 | 0.5526 | 0.3776 | 0.1082 | 0.1750 | [0.125, 0.222] | 0.94 | exceeds | 0.875 / 0.97 |
| ell_Grek | lang | 16 | 0.4164 | 0.2827 | -0.0360 | 0.1338 | [0.087, 0.185] | 0.94 | exceeds | 0.500 / 0.87 |
| hin_Deva | lang | 16 | 0.3592 | 0.3225 | -0.0185 | 0.0367 | [0.000, 0.071] | 0.75 | exceeds | 0.688 / 0.73 |
| jpn_Jpan | lang | 16 | 0.5678 | 0.4019 | 0.1305 | 0.1658 | [0.133, 0.199] | 1.00 | exceeds | 0.625 / 0.66 |
| rus_Cyrl | lang | 16 | 0.5582 | 0.3270 | 0.0897 | 0.2312 | [0.190, 0.273] | 1.00 | exceeds | 0.688 / 0.86 |
| tha_Thai | lang | 16 | 0.3975 | 0.3775 | 0.0125 | 0.0200 | [-0.028, 0.076] | 0.50 | inconclusive | 0.688 / 0.84 |
| arxiv | math | 16 | 0.5461 | 0.3638 | 0.1132 | 0.1822 | [0.122, 0.240] | 0.94 | exceeds | n/m (0.00 vs ceiling 0.00) |
| isabelle | math | 16 | 0.3606 | 0.4006 | 0.0000 | -0.0401 | [-0.067, -0.014] | 0.25 | reversed | n/m (0.00 vs ceiling 0.03) |
| lean | math | 16 | 0.4029 | 0.3739 | -0.0034 | 0.0290 | [-0.024, 0.094] | 0.50 | inconclusive | n/m (0.06 vs ceiling 0.07) |
| owm | math | 16 | 0.4931 | 0.2966 | 0.0801 | 0.1965 | [0.144, 0.248] | 1.00 | exceeds | 1.000 / 0.82 |

CSV: `arms_2026-09-18_rl-last16-lr5e-7_at_vllm__asym.csv`

### Pre-registered claim — 2026-09-18_rl-last16-lr5e-7@vllm:asym

Level 1, design §0: *on every arm, the best of 64 MAEMM rollouts aligns with the target more closely than the best window of an in-domain corpus search in the target's own domain.* Read at **1M** corpus tokens, over the 21 arms of the conjunction (design §6: lang 8, code 8, math 4, `ufw_zh`; the `diag` arm `formulas` and the §8(a) English pipeline check `ufw_en` are reported as rows but not counted):

**17 of 21 arms exceed.**  Not exceeding: `sql` (inconclusive), `tha_Thai` (inconclusive), `isabelle` (reversed), `lean` (inconclusive).

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

`scan` scores a corpus window as `normalize(h) @ v` with `v = unit(act - mu)`: the corpus activation UNCENTRED against a CENTRED target (precompute/scan.py; design §4, 'uncentred cosine in the scan'). That is the paper's convention and it is unchanged -- the English reference below reproduces the design's R1 numbers to the digit (0.3137 / 0.3511 / 0.3851 at 1/4/16M, own document top-1 on 114 of 512 targets at 4M).

`score` on this branch emits two SYMMETRIC cosines instead: `cos` = cos(h, unit(act)) and `cos_centred` = cos(h - mu, unit(act - mu)). Neither is `cos(h, unit(act - mu))`, the asymmetric number the paper's bo64 0.569 is, and on a `storage: raw` set there is no flag that produces it -- the legacy path got it for free because a `storage: unit` set's stored rows ARE `unit(act - mu)`, so `dirs` was already the centred target.

Consequences, stated rather than smoothed:

- the CORPUS side of every Δ is exactly the paper's; the MAEMM side is not, so the MAGNITUDE of Δ here is not comparable with the design's English margin of +0.218 (4M) or +0.256 (1M), and neither is bo64 comparable with 0.569;
- the pre-registered claim is a per-arm SIGN ('the best of 64 rollouts aligns more closely than the best corpus window'), and a sign is testable under any one convention applied to both sides of the comparison -- which is why the verdicts are reported and the margins are labelled;
- both cosines are in the CSV, so the arms can be re-read under either without re-running anything on the GPU.

Closing this properly means one of: `score` gaining the asymmetric cosine on a raw set, or the scan centring its corpus activations to match `cos_centred`. The second invalidates the frozen English reference; the first does not. Not decided here.

## Skipped and absent

- config'd checkpoint with no products on this set: `qwen36-27b/2026-07-14_nla-av`
- config'd checkpoint with no products on this set: `qwen36-27b/2026-09-08_rlI-150`
- scans read: `2026-09-21_ood_q1__arb_Arab__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__arb_Arab__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__arxiv__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__arxiv__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__c__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__c__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__ces_Latn__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__ces_Latn__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__cmn_Hani__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__cmn_Hani__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__corpus__4m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__corpus__4m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__ell_Grek__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__ell_Grek__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__formulas__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__formulas__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__go__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__go__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__haskell__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__haskell__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__hin_Deva__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__hin_Deva__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__isabelle__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__isabelle__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__javascript__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__javascript__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__jpn_Jpan__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__jpn_Jpan__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__lean__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__lean__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__owm__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__owm__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__python__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__python__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__rus_Cyrl__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__rus_Cyrl__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__rust__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__rust__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__shell__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__shell__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__sql__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__sql__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__tha_Thai__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__tha_Thai__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__ufw_en__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__ufw_en__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, `2026-09-21_ood_q1__ufw_zh__1m` at mu=base/qwen36-27b/stats/mu.f32, `2026-09-21_ood_q1__ufw_zh__1m__mu-whiten` at mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy
- control `2026-09-16_base-control@vllm:mu-stats__asym` not used: superseded by `2026-09-16_base-control@vllm`
- source `2026-09-10_rl-8x2048-full@vllm` was scored before `cos_asym` existed and is superseded by the `--score-tag asym` re-score of the SAME rollouts; not tabulated
- source `2026-09-18_rl-last16-lr5e-7@vllm` was scored before `cos_asym` existed and is superseded by the `--score-tag asym` re-score of the SAME rollouts; not tabulated
- `2026-09-10_rl-8x2048-full@vllm:asym`: language-id taken on the top-1 BY SCORE (asym); max |check| 0.00024
- `2026-09-18_rl-last16-lr5e-7@vllm:asym`: language-id taken on the top-1 BY SCORE (asym); max |check| 0.00024
