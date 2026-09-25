# OOD arms against the 10M English training corpus: `2026-09-23_ood_full`

Base `qwen36-27b`, 22 arms x 512 targets (kept pairs), corpus search over the **English 10M training corpus** (`corpora.celeste-train10m`, dir `train_parity_10m`) at **10M tokens**, scan `2026-09-23_ood_full__train_parity_10m__paper0923`.

Δ = the MAEMM's unbiased best-of-8 minus the English corpus search's top-1 on the SAME target, paired; CI is the design's 10,000-resample percentile bootstrap over the arm's targets and the SE beside it is the document-clustered one (`results.common.cluster_bootstrap`). The outcome is the same three-state verdict `results/ood.py` reports: **exceeds** (CI above zero), **inconclusive** (CI covers zero), **reversed** (CI below zero).

THE OWN-DOMAIN COLUMNS ARE `results/ood.py`'s NUMBERS, printed for comparison and not recomputed here. The two Δ columns are different contrasts: own-domain search reads the target's own language, the English search reads the training distribution.

Run under `MODAL_PROFILE=maemms`.

### Arms — 2026-09-18_rl-last16-lr5e-7@vllm:paper0923

*`bo8 (centred)` is `results.ood.centred_bo_k`'s recompute from `cos_centred.f16`, the same number `results/ood.py`'s bo8 column carries. `en corpus top-1` is the top-1 window of the 10M English training corpus on the same target. `own corpus top-1` and `own Δ` are read from `repo-maemm-v3/paper-evals/results/ood/arms_2026-09-18_rl-last16-lr5e-7_at_vllm__paper0923.csv`.*

| arm | family | n | bo8 (centred) | en corpus top-1 (10M) | Δ vs en corpus | 95% CI | win | outcome | own corpus top-1 | own Mtok | own Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c | code | 512 | 0.7371 | 0.4554 | 0.2817 | [0.269, 0.294] | 0.97 | exceeds | 0.7109 | 10.0 | 0.0262 |
| go | code | 512 | 0.6965 | 0.4131 | 0.2834 | [0.270, 0.297] | 0.96 | exceeds | 0.7137 | 10.0 | -0.0172 |
| haskell | code | 512 | 0.7076 | 0.4279 | 0.2798 | [0.268, 0.291] | 0.99 | exceeds | 0.7209 | 10.0 | -0.0132 |
| javascript | code | 512 | 0.6993 | 0.4155 | 0.2837 | [0.270, 0.298] | 0.97 | exceeds | 0.6694 | 10.0 | 0.0299 |
| python | code | 512 | 0.7190 | 0.4476 | 0.2714 | [0.259, 0.284] | 0.97 | exceeds | 0.6713 | 10.0 | 0.0477 |
| rust | code | 512 | 0.7023 | 0.4017 | 0.3006 | [0.287, 0.314] | 0.98 | exceeds | 0.6889 | 10.0 | 0.0134 |
| shell | code | 512 | 0.7229 | 0.4582 | 0.2647 | [0.252, 0.276] | 0.97 | exceeds | 0.7119 | 4.0 | 0.0110 |
| sql | code | 512 | 0.7008 | 0.4815 | 0.2193 | [0.205, 0.233] | 0.94 | exceeds | 0.7161 | 10.0 | -0.0153 |
| ufw_en | ctrl | 512 | 0.8277 | 0.5795 | 0.2482 | [0.238, 0.258] | 0.98 | exceeds | 0.5857 | 10.0 | 0.2420 |
| ufw_zh | ctrl | 512 | 0.8091 | 0.4462 | 0.3629 | [0.351, 0.374] | 0.99 | exceeds | 0.6260 | 10.0 | 0.1831 |
| arb_Arab | lang | 512 | 0.7816 | 0.4618 | 0.3198 | [0.309, 0.330] | 1.00 | exceeds | 0.7096 | 10.0 | 0.0720 |
| ces_Latn | lang | 512 | 0.7731 | 0.4665 | 0.3066 | [0.296, 0.317] | 1.00 | exceeds | 0.6884 | 10.0 | 0.0847 |
| cmn_Hani | lang | 512 | 0.8071 | 0.4528 | 0.3543 | [0.343, 0.365] | 0.99 | exceeds | 0.6595 | 10.0 | 0.1476 |
| ell_Grek | lang | 512 | 0.7148 | 0.4607 | 0.2541 | [0.244, 0.265] | 0.99 | exceeds | 0.7036 | 10.0 | 0.0112 |
| hin_Deva | lang | 512 | 0.7523 | 0.4388 | 0.3135 | [0.303, 0.324] | 0.99 | exceeds | 0.7850 | 10.0 | -0.0327 |
| jpn_Jpan | lang | 512 | 0.7845 | 0.4234 | 0.3611 | [0.349, 0.373] | 1.00 | exceeds | 0.6914 | 10.0 | 0.0931 |
| rus_Cyrl | lang | 512 | 0.8026 | 0.4791 | 0.3235 | [0.313, 0.333] | 0.99 | exceeds | 0.6712 | 10.0 | 0.1314 |
| tha_Thai | lang | 512 | 0.7420 | 0.4205 | 0.3215 | [0.310, 0.333] | 0.99 | exceeds | 0.7906 | 10.0 | -0.0486 |
| arxiv | math | 512 | 0.7891 | 0.5237 | 0.2653 | [0.255, 0.276] | 0.99 | exceeds | 0.6533 | 10.0 | 0.1358 |
| isabelle | math | 512 | 0.6821 | 0.4279 | 0.2543 | [0.242, 0.267] | 0.97 | exceeds | 0.7710 | 10.0 | -0.0889 |
| lean | math | 512 | 0.7317 | 0.4151 | 0.3166 | [0.306, 0.327] | 0.99 | exceeds | 0.7923 | 10.0 | -0.0606 |
| owm | math | 512 | 0.7731 | 0.5072 | 0.2659 | [0.255, 0.277] | 0.98 | exceeds | 0.6009 | 10.0 | 0.1722 |

CSV: `arms_encorp.csv`

### The level-1 conjunction, read against the English corpus

Over the 21 arms of the conjunction (`ufw_en`, the English anchor, and the `diag` family are excluded exactly as in `results/ood.py`): **21 exceed, 0 inconclusive, 0 reversed**. The smallest margin is `sql` at 0.2193 [0.2054, 0.2331].
## Skipped and absent

- config'd checkpoint with no products on this set: `qwen36-27b/2026-07-14_nla-av`
- config'd checkpoint with no products on this set: `qwen36-27b/2026-09-08_rlI-150`
- config'd checkpoint with no products on this set: `qwen36-27b/2026-09-10_rl-8x2048-full`
- config'd checkpoint with no products on this set: `qwen36-27b/2026-09-17_sft-simple2m`
- source `2026-09-16_base-control@vllm:paper0923` not used (role=control, maemm=qwen36-27b/2026-09-16_base-control)
- English-corpus scan `2026-09-23_ood_full__train_parity_10m__paper0923`: `--centre` recorded, mu=archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy, the same mean source `2026-09-18_rl-last16-lr5e-7@vllm:paper0923` recorded
- English corpus read at 10M of the scan's ladder [1.25, 2.5, 5.0, 10.0]; 45056 (row, size) cells, 11264 at 10M
- source `2026-09-18_rl-last16-lr5e-7@vllm:paper0923`: bo8 (centred) recomputed from `cos_centred.f16` on 11264 rows
- cells NOT merged (no --cells); the fragment is `cells_2026-09-18_rl-last16-lr5e-7_at_vllm__paper0923.csv`
