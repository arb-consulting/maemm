# Patchscopes on `2026-09-21_v3_realact` — layer sweep against the no-injection floor

base `qwen36-27b`, `--ps-tag paper0923`, prompt P2 (Patchscopes D.1 entity description), rule `replace` at alpha 2, read layer 42.

**Convention: centred, under `common.score_mu`.** Both arguments of every number in the `cos_centred` columns are taken about `bases.<base>.whiten_mu` — the activation as `h - mu` and the target as `unit(act - mu)` — which is the same constant the Exemplifier, the base control and the corpus search are read under, so the three are differences of one statistic. `cos_raw` is carried for continuity with the 2026-09-16 table and is not the paper's number.

Exclusions: 26 of 512 rows dropped, n = 486 — criterion `0.05` at n-gram 7, from `/vol/shared/ngram-overlap/hers_n7.exclude.json`.

SEs are the document-clustered bootstrap, 2000 resamples, seed 20260921 (`results.common.cluster_bootstrap`); a row with no document is its own cluster. The lift is PAIRED per direction and its sign test is exact.

| cell | layer | n rows | n docs | bo1 centred | bo8 centred | bo1 raw | bo8 raw | lift bo1 over floor | beats floor bo1 | sign-test p bo1 | lift bo8 over floor | beats floor bo8 | sign-test p bo8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `floor__paper0923` | — (floor) | 486 | 425 | 0.1178 ± 0.0026 | 0.1752 ± 0.0029 | 0.6101 ± 0.0034 | 0.6407 ± 0.0035 |  |  |  |  |  |  |
| `p2-L8-replace2__paper0923` | 8 | 486 | 425 | 0.1434 ± 0.0030 | 0.2289 ± 0.0041 | 0.6202 ± 0.0034 | 0.6589 ± 0.0035 | 0.0256 ± 0.0016 | 0.827 | 8.87e-51 | 0.0537 ± 0.0031 | 0.852 | 2.13e-59 |
| `p2-L14-replace2__paper0923` | 14 | 486 | 425 | 0.1466 ± 0.0030 | 0.2351 ± 0.0041 | 0.6216 ± 0.0033 | 0.6613 ± 0.0034 | 0.0288 ± 0.0017 | 0.864 | 4.44e-64 | 0.0600 ± 0.0031 | 0.885 | 1.69e-72 |
| `p2-L21-replace2__paper0923` | 21 | 486 | 425 | 0.1399 ± 0.0028 | 0.2274 ± 0.0039 | 0.6195 ± 0.0033 | 0.6583 ± 0.0034 | 0.0221 ± 0.0012 | 0.848 | 6.79e-58 | 0.0522 ± 0.0028 | 0.850 | 1.21e-58 |
| `p2-L42-replace2__paper0923` | 42 | 486 | 425 | 0.1416 ± 0.0027 | 0.2173 ± 0.0036 | 0.6193 ± 0.0033 | 0.6544 ± 0.0034 | 0.0238 ± 0.0015 | 0.819 | 4.14e-48 | 0.0421 ± 0.0025 | 0.815 | 8.23e-47 |

- `floor__paper0923`: centred ladder from `base/qwen36-27b/patchscopes/2026-09-21_v3_realact/floor__paper0923/scores/cos_centred.f16` (6.29 MB), 512 rows, 0 row(s) with a NaN rollout, 3584 stored `bo_c_k` cross-checks with 0 mismatch(es).
- `p2-L8-replace2__paper0923`: centred ladder from `base/qwen36-27b/patchscopes/2026-09-21_v3_realact/p2-L8-replace2__paper0923/scores/cos_centred.f16` (6.29 MB), 512 rows, 0 row(s) with a NaN rollout, 3584 stored `bo_c_k` cross-checks with 0 mismatch(es).
- `p2-L14-replace2__paper0923`: centred ladder from `base/qwen36-27b/patchscopes/2026-09-21_v3_realact/p2-L14-replace2__paper0923/scores/cos_centred.f16` (6.29 MB), 512 rows, 0 row(s) with a NaN rollout, 3584 stored `bo_c_k` cross-checks with 0 mismatch(es).
- `p2-L21-replace2__paper0923`: centred ladder from `base/qwen36-27b/patchscopes/2026-09-21_v3_realact/p2-L21-replace2__paper0923/scores/cos_centred.f16` (6.29 MB), 512 rows, 0 row(s) with a NaN rollout, 3584 stored `bo_c_k` cross-checks with 0 mismatch(es).
- `p2-L42-replace2__paper0923`: centred ladder from `base/qwen36-27b/patchscopes/2026-09-21_v3_realact/p2-L42-replace2__paper0923/scores/cos_centred.f16` (6.29 MB), 512 rows, 0 row(s) with a NaN rollout, 3584 stored `bo_c_k` cross-checks with 0 mismatch(es).
