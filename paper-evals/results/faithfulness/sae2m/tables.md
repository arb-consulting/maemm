# Eval 1 — faithfulness on `2026-09-21_v3_sae2m`

- set `2026-09-21_v3_sae2m`, base `qwen36-27b`, volume root `/vol`, mirror `/tmp/claude-1000/-home-gavento-dev-mimir/bf3732a4-2085-4ae5-9cef-db193a1cd1a2/scratchpad/conventions-wt/paper-evals/results/data/vol`
- command: `results/faithfulness.py --no-fetch --set 2026-09-21_v3_realact,2026-09-21_v3_ours,2026-09-21_v3_realact_long,2026-09-21_v3_subspace,2026-09-21_v3_ctrl,2026-09-21_v3_sae2m --out results/faithfulness`
- sources present: 2026-09-10_rl-8x2048-full@vllm:mu-none, 2026-07-14_nla-av, 2026-09-18_rl-last16-lr5e-7@vllm
- SEs: bootstrap over document clusters, 2000 resamples, seed 20260921
- exclusions: none — base/qwen36-27b/heldout/2026-09-21_v3_sae2m carries no exclusions.json

### SAE activation — family `sae/sae2m/dec`

*median and mean over features of `peak / corpus_peak`, the denominator being OUR 16M `max_act` as `sae_self` recorded it per feature — never a 1B-scan peak. `item fired` is the mean over features of the fraction of that source's own draws above the learned gate; `feat firing` is the fraction of features that fire at all. NO SAE-target cosine appears here (plan §2.3); the per-row cosines are in `sae_cosines.csv`.*

| source | run tag | stratum | features | n | bo1 med | bo8 med | bo64 med | bo1 mean | bo8 mean | bo64 mean | item fired | feat firing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-14_nla-av | — | all | 512 | 4 | 0.2773 | — | — | 0.3217 | — | — | 0.2070 | 0.3379 |
| 2026-07-14_nla-av | — | 0 | 128 | 4 | 0.2361 | — | — | 0.2797 | — | — | 0.0898 | 0.1875 |
| 2026-07-14_nla-av | — | 1 | 128 | 4 | 0.2239 | — | — | 0.2469 | — | — | 0.0820 | 0.1719 |
| 2026-07-14_nla-av | — | 2 | 128 | 4 | 0.2585 | — | — | 0.2963 | — | — | 0.1797 | 0.3516 |
| 2026-07-14_nla-av | — | 3 | 128 | 4 | 0.4106 | — | — | 0.4638 | — | — | 0.4766 | 0.6406 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | all | 512 | 64 | 0.1337 | 0.3151 | 0.4533 | 0.2378 | 0.3992 | 0.5245 | 0.1645 | 0.5273 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 0 | 128 | 64 | 0.0775 | 0.2393 | 0.3679 | 0.1011 | 0.2591 | 0.3996 | 0.0205 | 0.2500 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 1 | 128 | 64 | 0.1143 | 0.2816 | 0.3988 | 0.1657 | 0.3302 | 0.4599 | 0.0770 | 0.4297 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 2 | 128 | 64 | 0.1446 | 0.3061 | 0.4357 | 0.2050 | 0.3621 | 0.4876 | 0.1161 | 0.6016 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 3 | 128 | 64 | 0.3132 | 0.5051 | 0.6687 | 0.4796 | 0.6453 | 0.7507 | 0.4445 | 0.8281 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | all | 512 | 64 | 0.2709 | 0.4519 | 0.5806 | 0.3267 | 0.4882 | 0.6106 | 0.2265 | 0.8164 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 0 | 128 | 64 | 0.1966 | 0.3679 | 0.5115 | 0.2225 | 0.3932 | 0.5268 | 0.0533 | 0.6094 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 1 | 128 | 64 | 0.2161 | 0.3669 | 0.5169 | 0.2496 | 0.4077 | 0.5334 | 0.0978 | 0.7734 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 2 | 128 | 64 | 0.2677 | 0.4149 | 0.5381 | 0.3082 | 0.4687 | 0.5833 | 0.2059 | 0.9141 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 3 | 128 | 64 | 0.4441 | 0.6247 | 0.7834 | 0.5264 | 0.6832 | 0.7988 | 0.5488 | 0.9688 |

CSV: `act_sae_sae2m_dec.csv`

### SAE activation — family `sae/sae2m/enc`

*median and mean over features of `peak / corpus_peak`, the denominator being OUR 16M `max_act` as `sae_self` recorded it per feature — never a 1B-scan peak. `item fired` is the mean over features of the fraction of that source's own draws above the learned gate; `feat firing` is the fraction of features that fire at all. NO SAE-target cosine appears here (plan §2.3); the per-row cosines are in `sae_cosines.csv`.*

| source | run tag | stratum | features | n | bo1 med | bo8 med | bo64 med | bo1 mean | bo8 mean | bo64 mean | item fired | feat firing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-14_nla-av | — | all | 512 | 4 | 0.2334 | — | — | 0.2647 | — | — | 0.1489 | 0.2461 |
| 2026-07-14_nla-av | — | 0 | 128 | 4 | 0.2131 | — | — | 0.2307 | — | — | 0.0234 | 0.0703 |
| 2026-07-14_nla-av | — | 1 | 128 | 4 | 0.1924 | — | — | 0.2102 | — | — | 0.0312 | 0.0703 |
| 2026-07-14_nla-av | — | 2 | 128 | 4 | 0.2201 | — | — | 0.2525 | — | — | 0.1309 | 0.2891 |
| 2026-07-14_nla-av | — | 3 | 128 | 4 | 0.3200 | — | — | 0.3653 | — | — | 0.4102 | 0.5547 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | all | 512 | 64 | 0.0946 | 0.2586 | 0.4026 | 0.1836 | 0.3372 | 0.4793 | 0.1281 | 0.4023 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 0 | 128 | 64 | 0.0471 | 0.1764 | 0.3219 | 0.0691 | 0.2071 | 0.3639 | 0.0103 | 0.1172 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 1 | 128 | 64 | 0.0702 | 0.2367 | 0.3862 | 0.1005 | 0.2543 | 0.4071 | 0.0392 | 0.2344 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 2 | 128 | 64 | 0.0976 | 0.2519 | 0.3716 | 0.1572 | 0.3163 | 0.4567 | 0.0846 | 0.5000 |
| 2026-09-10_rl-8x2048-full@vllm:mu-none | mu-none | 3 | 128 | 64 | 0.2223 | 0.4260 | 0.5731 | 0.4075 | 0.5711 | 0.6896 | 0.3783 | 0.7578 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | all | 512 | 64 | 0.2390 | 0.4094 | 0.5437 | 0.2935 | 0.4578 | 0.5830 | 0.1948 | 0.7773 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 0 | 128 | 64 | 0.1876 | 0.3704 | 0.4992 | 0.2006 | 0.3735 | 0.5068 | 0.0332 | 0.5312 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 1 | 128 | 64 | 0.2108 | 0.3845 | 0.4917 | 0.2283 | 0.3941 | 0.5242 | 0.0813 | 0.7109 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 2 | 128 | 64 | 0.2328 | 0.3795 | 0.4919 | 0.2697 | 0.4278 | 0.5562 | 0.1636 | 0.9141 |
| 2026-09-18_rl-last16-lr5e-7@vllm | — | 3 | 128 | 64 | 0.4049 | 0.6086 | 0.7518 | 0.4757 | 0.6359 | 0.7446 | 0.5011 | 0.9531 |

CSV: `act_sae_sae2m_enc.csv`

### SAE-target cosines

*Not tabulated. They answer a different question (does the rollout point the right way) on a different scale, and a table that put them beside an activation ratio would invite the two to be read as one story. 3072 rows in `sae_cosines.csv`.*

### Reader check — our reduction against the product's own

*Both sides are the SAME reduction of the SAME array, so anything outside the storage's own rounding is a defect in this reader or in that product, never a finding. The tolerance is one float16 ulp (0.000488281 relative) plus the 4-decimal rounding both sides apply (0.0002); `worst excess` is how far the largest difference sat OUTSIDE its own tolerance, so a negative number means every comparison was inside it. `cos` recomputes `mean_cos` from `cos.f16`; `sae_self` recomputes `mean_peak_act` / `max_peak_act` from `sae_self.f16`.*

| kind | source | family | comparisons | worst excess | outcome |
|---|---|---|---|---|---|
| cos_centred bo-k | 2026-09-10_rl-8x2048-full@vllm:mu-none | (all) | — | — | skipped: no family of `2026-09-21_v3_sae2m` is centrable (['sae']), so a centred cosine would be NaN on every row -- the array is not read |
| cos_centred bo-k | 2026-07-14_nla-av | (all) | — | — | skipped: no family of `2026-09-21_v3_sae2m` is centrable (['sae']), so a centred cosine would be NaN on every row -- the array is not read |
| cos_centred bo-k | 2026-09-18_rl-last16-lr5e-7@vllm | (all) | — | — | skipped: no family of `2026-09-21_v3_sae2m` is centrable (['sae']), so a centred cosine would be NaN on every row -- the array is not read |
| sae_self | 2026-09-10_rl-8x2048-full@vllm:mu-none | sae/sae2m/enc | 512 | -0.000161 | 0 mismatches |
| sae_self | 2026-07-14_nla-av | sae/sae2m/enc | 512 | -0.000168 | 0 mismatches |
| sae_self | 2026-09-18_rl-last16-lr5e-7@vllm | sae/sae2m/enc | 512 | -0.000225 | 0 mismatches |
| sae_self | 2026-09-10_rl-8x2048-full@vllm:mu-none | sae/sae2m/dec | 512 | -0.000166 | 0 mismatches |
| sae_self | 2026-07-14_nla-av | sae/sae2m/dec | 512 | -0.000200 | 0 mismatches |
| sae_self | 2026-09-18_rl-last16-lr5e-7@vllm | sae/sae2m/dec | 512 | -0.000209 | 0 mismatches |
| cos | 2026-09-10_rl-8x2048-full@vllm:mu-none | (all) | — | — | skipped: cos.f16 is 12.6 MB, over --check-arrays-max-mb 8 |
| cos | 2026-07-14_nla-av | (all) | 1024 | -0.000201 | 0 mismatches |
| cos | 2026-09-18_rl-last16-lr5e-7@vllm | (all) | — | — | skipped: cos.f16 is 12.6 MB, over --check-arrays-max-mb 8 |

CSV: `reader_check.csv`

### Sanity gates (plan §2.3)

*From `results/sanity.yaml`, which the user edits. A `FLAG` means the mu or the prompt may be wrong and the numbers above should not be read until it is explained; `absent` means the gate's selector does not resolve on this set, which is a normal outcome for a gate written against another set; `no verdict` means the YAML declares the two are not the same statistic. `n` is the number of rows or features behind `ours` — a wide miss at small `n` is a different statement from a wide miss at the paper's own draw size.*

| gate | family | source | metric | n | ours | expected | tol | verdict | why |
|---|---|---|---|---|---|---|---|---|---|
| card realact, rl-last16 bo4 centred | realact | 2026-09-18_rl-last16-lr5e-7@vllm | cos_centred.bo4 | — | — | 0.7800 | 0.0500 | absent | no `cos_centred.bo4` for family `realact` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| card realact_long, rl-last16 bo4 centred | realact_long | 2026-09-18_rl-last16-lr5e-7@vllm | cos_centred.bo4 | — | — | 0.7160 | 0.0500 | absent | no `cos_centred.bo4` for family `realact_long` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| card 2M encoder fired fraction, rl-last16 | sae/sae2m/enc | 2026-09-18_rl-last16-lr5e-7@vllm | fired.item | 512 | 0.1948 | 0.3440 | 0.0800 | FLAG | \|0.1948 − 0.3440\| = 0.1492 vs tol 0.08 |
| card 2M decoder fired fraction, rl-last16 | sae/sae2m/dec | 2026-09-18_rl-last16-lr5e-7@vllm | fired.item | 512 | 0.2265 | 0.4160 | 0.0800 | FLAG | \|0.2265 − 0.4160\| = 0.1895 vs tol 0.08 |
| card 131k normalised activation, rl-last16 bo4 median ratio | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo4.median | — | — | 0.8240 | 0.1500 | absent | no `ratio.bo4.median` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 131k, rl-last16 bo1 median ratio | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.7600 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 131k, old primary mu-none bo1 median ratio | sae/l42-1b | 2026-09-10_rl-8x2048-full@vllm:mu-none | ratio.bo1.median | — | — | 0.8300 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/l42-1b` on `2026-09-10_rl-8x2048-full@vllm:mu-none` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 131k, rl-last16 feature firing fraction | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | fired.feature | — | — | 0.9200 | 0.1200 | absent | no `fired.feature` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 2M enc, rl-last16 bo1 median ratio | sae/sae2m/enc | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | 512 | 0.2390 | 0.2300 | 0.1000 | pass | \|0.2390 − 0.2300\| = 0.0090 vs tol 0.1 |
| sae_smoke64 2M, rl-last16 bo1 median ratio (unsided sets) | sae/sae2m | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.2300 | 0.1000 | absent | no `ratio.bo1.median` for family `sae/sae2m` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| sae_smoke64 2M enc, old primary mu-none bo1 median ratio | sae/sae2m/enc | 2026-09-10_rl-8x2048-full@vllm:mu-none | ratio.bo1.median | 512 | 0.0946 | 0.1000 | 0.0800 | pass | \|0.0946 − 0.1000\| = 0.0054 vs tol 0.08 |
| sae_smoke64 2M enc, NLA bo1 median ratio | sae/sae2m/enc | 2026-07-14_nla-av | ratio.bo1.median | 512 | 0.2334 | 0.2400 | 0.1000 | pass | \|0.2334 − 0.2400\| = 0.0066 vs tol 0.1 |
| old primary mu-none, realact mean cos (2026-09-21 v1raw smoke, n=4) | realact | rl-8x2048-full:mu-none | cos_raw.bo1 | — | — | 0.8760 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| old primary mu-stats, realact mean cos (2026-09-21 v1raw smoke, n=4) | realact | rl-8x2048-full:mu-stats | cos_raw.bo1 | — | — | 0.8989 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| old primary mu-stats, realact bo4 (v1raw smoke) | realact | rl-8x2048-full:mu-stats | cos_raw.bo4 | — | — | 0.9339 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| old primary mu-none, realact bo4 (v1raw smoke) | realact | rl-8x2048-full:mu-none | cos_raw.bo4 | — | — | 0.9034 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| rl-last16 realact mean centred cos (v1raw smoke, n=4) | realact | rl-last16 | cos_centred.bo1 | — | — | 0.7518 | 0.0005 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| eval1 rl-last16, realact mean cos_raw (n = 486) | realact | rl-last16 | cos_raw.bo1 | — | — | 0.8860 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_sae2m` |
| eval1 rl-last16, realact mean cos_centred (n = 486) | realact | rl-last16 | cos_centred.bo1 | — | — | 0.7590 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_sae2m` |
| eval1 rl-last16, ours realact mean cos_raw (n = 506) | realact | rl-last16 | cos_raw.bo1 | — | — | 0.8884 | 0.0005 | absent | `set: 2026-09-21_v3_ours` — this run is `2026-09-21_v3_sae2m` |
| eval1 rl-last16, ours realact mean cos_centred (n = 506) | realact | rl-last16 | cos_centred.bo1 | — | — | 0.7668 | 0.0005 | absent | `set: 2026-09-21_v3_ours` — this run is `2026-09-21_v3_sae2m` |
| eval1 old primary mu-stats, realact mean cos_raw (n = 486) | realact | rl-8x2048-full@vllm:mu-stats | cos_raw.bo1 | — | — | 0.8881 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_sae2m` |
| eval1 old primary mu-none, realact mean cos_raw (n = 486) | realact | rl-8x2048-full@vllm:mu-none | cos_raw.bo1 | — | — | 0.8830 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_sae2m` |
| eval1 old primary mu-stats, realact bo64 (n = 486) | realact | rl-8x2048-full@vllm:mu-stats | cos_raw.bo64 | — | — | 0.9322 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_sae2m` |
| eval1 rl-last16, subspace/bsf mean cos_raw | bsf | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.3177 | 0.0005 | absent | no `cos_raw.bo1` for family `bsf` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, subspace/jlens mean cos_raw | jlens | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.1058 | 0.0005 | absent | no `cos_raw.bo1` for family `jlens` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, realact_long mean cos_raw | realact_long | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.4360 | 0.0005 | absent | no `cos_raw.bo1` for family `realact_long` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 rl-last16, ctrl/random mean cos_raw | random | 2026-09-18_rl-last16-lr5e-7@vllm | cos_raw.bo1 | — | — | 0.0338 | 0.0005 | absent | no `cos_raw.bo1` for family `random` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 NLA, realact mean cos_raw (n = 486) | realact | 2026-07-14_nla-av | cos_raw.bo1 | — | — | 0.8075 | 0.0005 | absent | `set: 2026-09-21_v3_realact` — this run is `2026-09-21_v3_sae2m` |
| eval1 rl-last16, 2M enc median bo1 ratio (512 features) | sae/sae2m/enc | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | 512 | 0.2390 | 0.2390 | 0.0020 | pass | \|0.2390 − 0.2390\| = 0.0000 vs tol 0.002 |
| eval1 rl-last16, 131k median bo1 ratio (512 features) | sae/l42-1b | 2026-09-18_rl-last16-lr5e-7@vllm | ratio.bo1.median | — | — | 0.7190 | 0.0020 | absent | no `ratio.bo1.median` for family `sae/l42-1b` on `2026-09-18_rl-last16-lr5e-7@vllm` in this run (the family or the best-of-k is not present on this set) |
| eval1 old primary mu-none, 2M enc median bo1 ratio | sae/sae2m/enc | 2026-09-10_rl-8x2048-full@vllm:mu-none | ratio.bo1.median | 512 | 0.0946 | 0.0950 | 0.0020 | pass | \|0.0946 − 0.0950\| = 0.0004 vs tol 0.002 |
| eval1 NLA, 2M enc median bo1 ratio | sae/sae2m/enc | 2026-07-14_nla-av | ratio.bo1.median | 512 | 0.2334 | 0.2330 | 0.0020 | pass | \|0.2334 − 0.2330\| = 0.0004 vs tol 0.002 |
| old primary, sae rows, v1raw vs v1 on the shared rows | sae/l42-1b | rl-8x2048-full:mu-none | cos_raw.bo1 | — | — | — | 0.0100 | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| old primary, realact rows, v1raw vs v1 on the shared rows | realact | rl-8x2048-full:mu-stats | cos_raw.bo1 | — | — | — | — | absent | `set: 2026-09-21_v1raw` — this run is `2026-09-21_v3_sae2m` |
| rl-last16, ctrl sae rows, v3_ctrl vs the 131k rows of v1raw | sae/l42-1b | rl-last16 | cos_raw.bo1 | — | — | — | — | absent | `set: 2026-09-21_v3_ctrl` — this run is `2026-09-21_v3_sae2m` |
| old primary v1 realact mean cos, 0.5076 | realact | 2026-09-10_rl-8x2048-full@vllm:mu-none | cos_raw.bo1 | — | — | 0.5076 | — | absent | no `cos_raw.bo1` for family `realact` on `2026-09-10_rl-8x2048-full@vllm:mu-none` in this run (the family or the best-of-k is not present on this set) |

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

- `figures/strata_sae_sae2m_dec.pdf` / `.png`
- `figures/strata_sae_sae2m_enc.pdf` / `.png`

## Skipped and absent

- `qwen36-27b/2026-09-08_rlI-150`: no scores directory for `2026-09-21_v3_sae2m` under `maemms/qwen36-27b/2026-09-08_rlI-150/scores/`
- `qwen36-27b/2026-09-16_base-control`: no scores directory for `2026-09-21_v3_sae2m` under `maemms/qwen36-27b/2026-09-16_base-control/scores/`
