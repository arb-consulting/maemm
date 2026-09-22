# INVENTORY — what is on the volume, generated

Volume `maemm`, workspace `maemms`. Generated 2026-09-21 15:15 local by `python -m features.inventory`.

This file is GENERATED and goes stale the moment a job lands — regenerate it rather than editing it. `README.md` beside it is the hand-written part: layout, conventions and the things that bite. Read that first; this is the stock list.

## Models in the HF cache

```
BAAI/bge-small-en-v1.5                                  5c38ec7c
Qwen/Qwen3-8B                                           b968826d
Qwen/Qwen3.6-27B                                        6a9e13bd
adamkarvonen/qwen3-8b-saes                              460a11d5
caiovicentino1/qwen36-27b-sae-fullstack                 0f03d7c3
ceselder/maemm-27b-rl-last16-lr5e-7                     a1e4f299
ceselder/maemm-qwen3-8b-invert-rl-v3-step600            acc0ca78
ceselder/maemm-qwen36-27b-inverter-rlE-step250          75b6aaa0
ceselder/maemm-qwen36-27b-inverter-rlI-step150          52101b1c
ceselder/qwen3.6-27b-nla-av                             def1421d
ceselder/qwen36-27b-maemm-inverter-rl-8x2048            1c50a41d
ceselder/qwen36-27b-sae-l42-1b                          174c52b9
ceselder/qwen36-27b-sae2m-l42                           de89b1c1
gavento/maemm-qwen3-8b-run1                             4978ef76
gavento/maemm-qwen3-8b-run2                             ea115289
```

`common.snapshot()` asserts exactly one snapshot per repo; two makes "which weights did that run use" unanswerable.

## Target sets (16)

Each entry carries the selection rule from the set's OWN README, written by the product that drew it. How a set was selected decides what a mean over it means.

### `2026-09-16_v1`

- files: `README.md`, `ids.jsonl`, `index.json`, `leakage.jsonl`, `mu_512.f32`, `vecs.f16`
- corpus scan: **yes**
- CENTRING: the realact vectors are `unit(X[p] - stats/mu.f32)` (/vol/base/qwen36-27b/stats/mu.f32, ||mu||=67.93). One rule, one subtraction, at construction only.
- mu_512.f32 [d] is a DIAGNOSTIC: the mean read-layer activation over all 524288 positions of the 1024 forwarded 512-token windows (NO sink token), i.e. Celeste's convention (data/build_universal_bank.py:310). Nothing is centred on it. Against stats/mu.f32 it has cos = 0.9773 and ||mu_512|| / ||mu|| = 0.9852.
- sae: eligible = gated fires >= 20 AND max_act > 0 at the largest corpus size (16M, 60812331 scanned positions): 130856 features pass that, 130856 remain after the training-feature exclusion; stratified into 4 quartiles of log10(density) with log10-density cuts [-4.335, -3.776, -3.347], 128 drawn per quartile; direction = unit(W_enc[:, f]) (the ENCODER column)
- sae exclusion: none applied (no archived training split exists for this base)
- family 'jlens': declared in config with status='empty' and drawn with ZERO rows -- no J-lens matrix exists for either base (checklist item 40, layout §5a item 4), so the slot only fixes the set's shape
- leakage check: skipped (only the 8B training banks are in the archive)

### `2026-09-18_handpicked_smoke`

- files: `README.md`, `ids.jsonl`, `index.json`, `samples.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- second independent read: `output_hidden_states=True` on the SAME ids. `hidden_states[43]` is the tensor the block-42 forward hook captures (hidden_states has 65 entries, index 0 = the embedding output, so block L's output is index L+1). Max |diff| at the chosen position, by candidate index: 42 -> 2.375e+00, 43 -> 0.000e+00
- target read: NO sink token prepended (mirrors `_realact`); the scorer's own read prepends one and drops it. hidden_states index equal to the hooked block-42 output: 43; worst |diff| over all rows 0.000e+00
- centring: unit(X[p] - mu) with mu = /vol/base/qwen36-27b/stats/mu.f32, ||mu||=67.93 -- the SAME convention the realact family uses, and the scoring side is UNCENTRED for both (common.score_ids takes the cosine against the raw residual), so a handpicked row scores exactly like a realact row. Per row: `cos_centred_uncentred` is the cosine between the two target variants and `cos_x_mu` is the cosine of the RAW read against mu -- an uncentred target would score high on almost anything through that shared mean direction.
- rebuild: `modal run precompute/modal_app.py --product targets --base qwen36-27b --set 2026-09-18_handpicked_smoke --samples <the samples file> --root /vol` at repo commit de4da95c08c8; `samples.json` here is the input verbatim
- vecs.f16 rows are unit in fp32 before the cast; `cos_f16_fp32` per row is measured
- no mu_512.f32 and no leakage.jsonl: neither is defined for a set that is not drawn from the corpus

### `2026-09-18_handpicked_v1`

- files: `README.md`, `ids.jsonl`, `index.json`, `samples.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- second independent read: `output_hidden_states=True` on the SAME ids. `hidden_states[43]` is the tensor the block-42 forward hook captures (hidden_states has 65 entries, index 0 = the embedding output, so block L's output is index L+1). Max |diff| at the chosen position, by candidate index: 42 -> 6.750e+00, 43 -> 0.000e+00
- target read: NO sink token prepended (mirrors `_realact`); the scorer's own read prepends one and drops it. hidden_states index equal to the hooked block-42 output: 43; worst |diff| over all rows 0.000e+00
- centring: unit(X[p] - mu) with mu = /vol/base/qwen36-27b/stats/mu.f32, ||mu||=67.93 -- the SAME convention the realact family uses, and the scoring side is UNCENTRED for both (common.score_ids takes the cosine against the raw residual), so a handpicked row scores exactly like a realact row. Per row: `cos_centred_uncentred` is the cosine between the two target variants and `cos_x_mu` is the cosine of the RAW read against mu -- an uncentred target would score high on almost anything through that shared mean direction.
- rebuild: `modal run precompute/modal_app.py --product targets --base qwen36-27b --set 2026-09-18_handpicked_v1 --samples <the samples file> --root /vol` at repo commit de4da95c08c8; `samples.json` here is the input verbatim
- vecs.f16 rows are unit in fp32 before the cast; `cos_f16_fp32` per row is measured
- no mu_512.f32 and no leakage.jsonl: neither is defined for a set that is not drawn from the corpus

### `2026-09-18_handpicked_v1last`

- files: `README.md`, `ids.jsonl`, `index.json`, `samples.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- second independent read: `output_hidden_states=True` on the SAME ids. `hidden_states[43]` is the tensor the block-42 forward hook captures (hidden_states has 65 entries, index 0 = the embedding output, so block L's output is index L+1). Max |diff| at the chosen position, by candidate index: 42 -> 1.175e+01, 43 -> 0.000e+00
- target read: NO sink token prepended (mirrors `_realact`); the scorer's own read prepends one and drops it. hidden_states index equal to the hooked block-42 output: 43; worst |diff| over all rows 0.000e+00
- centring: unit(X[p] - mu) with mu = /vol/base/qwen36-27b/stats/mu.f32, ||mu||=67.93 -- the SAME convention the realact family uses, and the scoring side is UNCENTRED for both (common.score_ids takes the cosine against the raw residual), so a handpicked row scores exactly like a realact row. Per row: `cos_centred_uncentred` is the cosine between the two target variants and `cos_x_mu` is the cosine of the RAW read against mu -- an uncentred target would score high on almost anything through that shared mean direction.
- rebuild: `modal run precompute/modal_app.py --product targets --base qwen36-27b --set 2026-09-18_handpicked_v1last --samples <the samples file> --root /vol` at repo commit 3b5eea7c3d00; `samples.json` here is the input verbatim
- vecs.f16 rows are unit in fp32 before the cast; `cos_f16_fp32` per row is measured
- no mu_512.f32 and no leakage.jsonl: neither is defined for a set that is not drawn from the corpus

### `2026-09-18_ood_v1`

- files: `README.md`, `arms.json`, `ids.jsonl`, `index.json`, `vecs.f16`, `windows.i32`
- corpus scan: **yes**
- arm `tha_Thai` (lang): 64 targets from a 320-doc pool (320 passed the norm filter, presample median 77.2); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 16000034 tokens; verbatim shown span in its own corpus: 2/64 (3.1%); byte-piece rate 0.000; tok_class {'unspaced': 64}; token_bytes verified against tok.decode on 512 tokens (2808 bytes)
- arm `python` (code): 64 targets from a 320-doc pool (320 passed the norm filter, presample median 78.9); source bigcode/the-stack-smol-xl@e782ebf35c7e; corpus 16001210 tokens; verbatim shown span in its own corpus: 3/64 (4.7%); byte-piece rate 0.000; tok_class {'first': 5, 'last': 15, 'mid': 19, 'word': 25}; token_bytes verified against tok.decode on 512 tokens (1938 bytes)
- arm `ufw_en` (ctrl): 64 targets from a 320-doc pool (320 passed the norm filter, presample median 90.9); source openbmb/Ultra-FineWeb@02c85641e3d1; corpus 16000440 tokens; verbatim shown span in its own corpus: 0/64 (0.0%); byte-piece rate 0.000; tok_class {'first': 16, 'last': 7, 'mid': 4, 'word': 37}; token_bytes verified against tok.decode on 512 tokens (2008 bytes)
- verbatim-span rate per arm is in the per-arm notes above and in `arms.json`: the fraction of targets whose SHOWN span occurs verbatim somewhere in that arm's own corpus (design §4 -- near-duplicate documents are reported, never masked)
- read layer 42; d 5120; one row per target

### `2026-09-18_ood_v1_unitend`

- files: `README.md`, `arms.json`, `ids.jsonl`, `index.json`, `vecs.f16`, `windows.i32`
- corpus scan: **NO — no corpus-search baseline**
- arm `ufw_en`: 20/64 targets moved; rules {'wordend': 20, 'charend': 0, 'none': 44}
- arm `python`: 24/64 targets moved; rules {'wordend': 24, 'charend': 0, 'none': 40}
- verbatim-span rate per arm is in the per-arm notes above and in `arms.json`: the fraction of targets whose SHOWN span occurs verbatim somewhere in that arm's own corpus (design §4 -- near-duplicate documents are reported, never masked)
- read layer 42; d 5120; one row per target

### `2026-09-20_sae2m_2k`

- files: `README.md`, `ids.jsonl`, `index.json`, `vecs.f16`
- corpus scan: **yes**
- 2000 sae2m_enc targets, all from Celeste's eval split
- strata: log10_corpus_peak_1B from subset /vol/shared/sae2m-2k/features.parquet
- 1601 train/fit, 399 test/report -- BOTH halves are unseen by the MAEMM; this splits our analysis, not the model's training
- gate 1.682811975479126; vecs are unit(W_enc[:, f]) in fp32 before the cast

### `2026-09-21_ood_q1`

- files: `README.md`, `act.f32`, `arms.json`, `ids.jsonl`, `index.json`, `storage.json`, `vecs.f16`, `windows.i32`
- corpus scan: **NO — no corpus-search baseline**
- arm `ces_Latn` (lang): 16 targets from a 320-doc pool (320 passed the norm filter, presample median 80.0); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 4000186 tokens; verbatim shown span in its own corpus: 0/16 (0.0%); byte-piece rate 0.000; tok_class {'first': 4, 'last': 4, 'mid': 3, 'word': 5}; token_bytes verified against tok.decode on 512 tokens (1757 bytes)
- arm `rus_Cyrl` (lang): 16 targets from a 320-doc pool (320 passed the norm filter, presample median 85.0); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 4001592 tokens; verbatim shown span in its own corpus: 0/16 (0.0%); byte-piece rate 0.000; tok_class {'first': 7, 'last': 2, 'mid': 3, 'word': 4}; token_bytes verified against tok.decode on 512 tokens (3977 bytes)
- arm `ell_Grek` (lang): 16 targets from a 320-doc pool (320 passed the norm filter, presample median 77.3); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 4000122 tokens; verbatim shown span in its own corpus: 0/16 (0.0%); byte-piece rate 0.000; tok_class {'first': 3, 'last': 4, 'mid': 6, 'word': 3}; token_bytes verified against tok.decode on 512 tokens (2448 bytes)
- arm `arb_Arab` (lang): 16 targets from a 320-doc pool (320 passed the norm filter, presample median 81.9); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 4000756 tokens; verbatim shown span in its own corpus: 0/16 (0.0%); byte-piece rate 0.000; tok_class {'first': 6, 'last': 3, 'mid': 3, 'word': 4}; token_bytes verified against tok.decode on 512 tokens (3473 bytes)
- arm `hin_Deva` (lang): 16 targets from a 320-doc pool (320 passed the norm filter, presample median 74.1); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 4000239 tokens; verbatim shown span in its own corpus: 0/16 (0.0%); byte-piece rate 0.000; tok_class {'first': 7, 'last': 4, 'mid': 5}; token_bytes verified against tok.decode on 512 tokens (1999 bytes)
- arm `tha_Thai` (lang): 16 targets from a 320-doc pool (320 passed the norm filter, presample median 77.2); source HuggingFaceFW/fineweb-2@af9c13333eb9; corpus 16000034 tokens; verbatim shown span in its own corpus: 0/16 (0.0%); byte-piece rate 0.000; tok_class {'unspaced': 16}; token_bytes verified against tok.decode on 512 tokens (2808 bytes)

### `2026-09-21_sae131k_2k`

- files: `README.md`, `ids.jsonl`, `index.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- 2000 features of the 131k SAE (l42-1b), family tag 'sae', sae_key 'l42-1b' -- feature ids are NOT comparable with sae2m's
- 1578 train / 422 test, seed 20260921; BOTH halves are unseen -- this splits our analysis, not the training
- strata: log10_pool_peak_act quartiles over the drawn set, recorded not sampled, cuts [0.9954377414728388, 1.193124584077851, 1.342624961247057]
- HELD-OUT PROVENANCE IS WEAKER THAN THE 2M SET'S: held out by the EARLIER chains' feature split (pool_heldout/sae.parquet, 13,107 features), not by Celeste's seed-2026 2M partition. Whether a 131k feature has a near-duplicate among the 1.85M 2M features the v2 checkpoint trained on is NOT established.

### `2026-09-21_v1raw`

- files: `README.md`, `act.f32`, `ids.jsonl`, `index.json`, `leakage.jsonl`, `mu_512.f32`, `re_derive.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- CENTRING: NONE IS STORED. `act.f32` is the raw `X[p]` and `vecs.f16` is `unit(X[p])`. The mean is named per run (`--centering`, or the checkpoint's `input.centering`) and applied by `common.dirs_for`; `stats/mu.f32` (/vol/base/qwen36-27b/stats/mu.f32, ||mu||=67.93) is one of the names in config.yaml's `mus:` block, not the rule.
- span_text is CLAMPED at the document start (D4, 2026-09-21): 21 of 512 rows had `p - L + 1 < 0` and the unclamped slice would have reached into the PREVIOUS document. `L_shown` is the clamped length; `L` is the length the draw asked for. The activation is unaffected either way -- it is read at position p of this document's own 512-token window.
- mu_512.f32 [d] is a DIAGNOSTIC: the mean read-layer activation over all 524288 positions of the 1024 forwarded 512-token windows (NO sink token), i.e. Celeste's convention (data/build_universal_bank.py:310). Nothing is centred on it. Against stats/mu.f32 it has cos = 0.9773 and ||mu_512|| / ||mu|| = 0.9852.
- sae: eligible = gated fires >= 20 AND max_act > 0 at the largest corpus size (16M, 60812331 scanned positions): 130856 features pass that, 130856 remain after the training-feature exclusion; stratified into 4 quartiles of log10(density) with log10-density cuts [-4.335, -3.776, -3.347], 128 drawn per quartile; direction = unit(W_enc[:, f]) (the ENCODER column)
- sae exclusion: none applied (no archived training split exists for this base)
- family 'jlens': declared in config with status='empty' and drawn with ZERO rows -- no J-lens matrix exists for either base (checklist item 40, layout §5a item 4), so the slot only fixes the set's shape

### `2026-09-21_v3_ctrl`

- files: `README.md`, `act.f32`, `ids.jsonl`, `index.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- rebuild: `modal run precompute/modal_app.py --product heldout_v3 --base qwen36-27b --block ctrl --set 2026-09-21_v3_ctrl --root /vol --dirs-from /vol/base/qwen36-27b/heldout/2026-09-21_v1raw --rows 512-1535` at repo commit 1b032facbe7f
- IMPORTED / COPIED, NOT DRAWN. Snapshot `celeste-v2-2026-09-17`, base `Qwen/Qwen3.6-27B`, read layer 42, d = 5120. Checkpoint these targets are held out from: `ceselder/maemm-27b-rl-last16-lr5e-7` (revision `a1e4f299e0100a9340077c395504487375a63eec`).
- COPIED, not drawn: rows 512-1535 of `2026-09-21_v1raw` (/vol/base/qwen36-27b/heldout/2026-09-21_v1raw), 1024 rows, families ['random', 'sae']. Each row keeps its own fields and gains `src_set`/`src_row`, so the provenance survives the renumbering. `act.f32` and `vecs.f16` are the source's own bytes for those rows -- this is a copy, and the source is not modified.
- n per family: {'random': 512, 'sae': 512}; storage `raw`
- vecs.f16 rows are unit in fp32 before the cast; the f16 round-trip is ~1e-3 off unit

### `2026-09-21_v3_ours`

- files: `README.md`, `act.f32`, `exclusions.json`, `ids.jsonl`, `index.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- rebuild: `modal run precompute/modal_app.py --product heldout_v3 --base qwen36-27b --block ours --set 2026-09-21_v3_ours --root /vol --dirs-from /vol/base/qwen36-27b/heldout/2026-09-21_v1raw --rows 0-511` at repo commit cd243dede061
- IMPORTED / COPIED, NOT DRAWN. Snapshot `celeste-v2-2026-09-17`, base `Qwen/Qwen3.6-27B`, read layer 42, d = 5120. Checkpoint these targets are held out from: `ceselder/maemm-27b-rl-last16-lr5e-7` (revision `a1e4f299e0100a9340077c395504487375a63eec`).
- COPIED, not drawn: rows 0-511 of `2026-09-21_v1raw` (/vol/base/qwen36-27b/heldout/2026-09-21_v1raw), 512 rows, families ['realact']. Each row keeps its own fields and gains `src_set`/`src_row`, so the provenance survives the renumbering. `act.f32` and `vecs.f16` are the source's own bytes for those rows -- this is a copy, and the source is not modified.
- CENTRABLE families copied: ['realact']. Sound because the source is `storage: raw` with `mu_stored: null` -- its `act.f32` is the activation before any mean was subtracted, so these bytes carry no centring convention and `--mu` decides per run exactly as it does on `2026-09-21_v1raw`. A `storage: unit` source is refused.
- EXCLUSIONS frozen here, applied downstream: 6 of 512 rows [38, 45, 101, 318, 393, 446], headline n = 506. Criterion: fully reproduced in her v2 training text at n=13, infra/check_v2_targets_overlap.py 2026-09-18 over our 16M corpus. See exclusions.json.
- n per family: {'realact': 512}; storage `raw`

### `2026-09-21_v3_realact`

- files: `README.md`, `act.f32`, `exclusions.json`, `ids.jsonl`, `index.json`, `recovery.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- rebuild: `modal run precompute/modal_app.py --product heldout_v3 --base qwen36-27b --block realact --set 2026-09-21_v3_realact --root /vol` at repo commit 1b032facbe7f
- IMPORTED / COPIED, NOT DRAWN. Snapshot `celeste-v2-2026-09-17`, base `Qwen/Qwen3.6-27B`, read layer 42, d = 5120. Checkpoint these targets are held out from: `ceselder/maemm-27b-rl-last16-lr5e-7` (revision `a1e4f299e0100a9340077c395504487375a63eec`).
- HER 512 realact rows, RECOVERED TO RAW. `pool_act_norm` is `||act||` (recovery.json records the three readings that settle it), so `act = mu + t*u` with t the positive root of `||mu + t*u|| = pool_act_norm` under mu = /vol/archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy. 0 fallbacks; ||act|| reproduced to 2.51e-05 and her own `direction` to cos = 0.9999997742. `vecs.f16` here is therefore `unit(act)`, UNCENTRED -- her shipped direction is what `dirs_for(..., mu=/vol/archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy)` returns.
- 3 AMBIGUOUS rows [26, 32, 360]: `pool_act_norm < ||mu||` and `mu.u < 0`, so both roots are positive and two different raw activations meet the constraint. The larger is taken (build_inputs' rule). Anything read at her own mean is unaffected (both roots give her `direction` exactly); `act.f32` itself, and any OTHER mean, are uncertain on these three.
- EXCLUSIONS frozen here, applied downstream: 26 of 512 rows, headline n = 486. See exclusions.json.
- n per family: {'realact': 512}; storage `raw`

### `2026-09-21_v3_realact_long`

- files: `README.md`, `ids.jsonl`, `index.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- rebuild: `modal run precompute/modal_app.py --product heldout_v3 --base qwen36-27b --block realact_long --set 2026-09-21_v3_realact_long --root /vol` at repo commit 1b032facbe7f
- IMPORTED / COPIED, NOT DRAWN. Snapshot `celeste-v2-2026-09-17`, base `Qwen/Qwen3.6-27B`, read layer 42, d = 5120. Checkpoint these targets are held out from: `ceselder/maemm-27b-rl-last16-lr5e-7` (revision `a1e4f299e0100a9340077c395504487375a63eec`).
- `realact_long`: 512 rows as shipped, centred on a mean this repo holds NO FILE FOR (`unknown`). For `realact_long` that mean is `mu_long` -- the mean over ALL collected long-context activations, computed on the fly in `eval/build_ctx_eval.py:47-54` from `MAEMM_ACTS_LONG` (`/root/pmx/bsf27b/acts_long`) and never written to a file. It is NOT `whiten_mu`: over these rows `cos(direction, whiten_mu)` has mean -0.0618 and ||mean(direction)|| is 0.1216, against -0.0198 / 0.0615 on `realact`. The rows are returned AS SHIPPED and every number read off them is labelled.
- n per family: {'realact_long': 512}; storage `unit`
- vecs.f16 rows are unit in fp32 before the cast; the f16 round-trip is ~1e-3 off unit

### `2026-09-21_v3_sae2m`

- files: `README.md`, `ids.jsonl`, `index.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- 1024 targets, all from Celeste's eval split. `family` is **sae** -- the label every consumer selects on (scan, top1_act, repo_examples, gcg, score's per-family means, autointerp's sae_self and build) -- and the dictionary is named per row in `sae_key` ('qwen36-27b/sae2m'), because a feature index means nothing without it. Sets drawn before 2026-09-21 carry `family: sae2m_enc` instead and are NOT rewritten; autointerp still accepts that label for them.
- strata: log10_gated_fires_16M from our 16M corpus scan
- STRATIFIED draw, seed 20260921: 112 features from each of the 4 quartiles of the ELIGIBLE POOL (99,882 features), cuts on log10_gated_fires_16M at [2.4928, 2.8209, 3.1923], pool sizes [24954, 24951, 25001, 24976]. The cuts are the POOL's quartiles, not the drawn set's: at an equal count per stratum the drawn set's quartiles are the stratum boundaries by construction. Any mean over these rows is a mean over four equally-weighted quartiles, NOT over the dictionary -- report per stratum.
- 413 train/fit, 99 test/report -- BOTH halves are unseen by the MAEMM; this splits our analysis, not the model's training
- gate 1.682811975479126; F = 2,097,152; vecs are unit dictionary columns in fp32 before the cast
- BOTH SIDES of the dictionary, 512 features x ['enc', 'dec'] = 1024 rows, in that block order and PAIRED row for row within a block: row i and row i + 512 are the encoder and decoder column of the SAME feature. Select one with `common.sae_rows_of(..., side=)`, which reads each row's `sae_side`. The activation metric is the same on both (the activation of feature f is its ENCODER readout whichever direction was injected); the `vecs.f16` cross-check in sae_self does not apply to a decoder row.

### `2026-09-21_v3_subspace`

- files: `README.md`, `ids.jsonl`, `index.json`, `storage.json`, `vecs.f16`
- corpus scan: **NO — no corpus-search baseline**
- rebuild: `modal run precompute/modal_app.py --product heldout_v3 --base qwen36-27b --block subspace --set 2026-09-21_v3_subspace --root /vol` at repo commit 1b032facbe7f
- IMPORTED / COPIED, NOT DRAWN. Snapshot `celeste-v2-2026-09-17`, base `Qwen/Qwen3.6-27B`, read layer 42, d = 5120. Checkpoint these targets are held out from: `ceselder/maemm-27b-rl-last16-lr5e-7` (revision `a1e4f299e0100a9340077c395504487375a63eec`).
- `bsf`: 512 rows as shipped, never centred and not centrable (config.yaml `family_kinds`), so no `--mu` applies to it.
- `jlens`: 512 rows as shipped, never centred and not centrable (config.yaml `family_kinds`), so no `--mu` applies to it.
- n per family: {'bsf': 512, 'jlens': 512}; storage `dirs_only`
- vecs.f16 rows are unit in fp32 before the cast; the f16 round-trip is ~1e-3 off unit

A set with no scan has **no corpus-search baseline**. `scan` additionally needs `stats` to have run for that SAE (it reads `sae/<sae>/max_act.f16`) and fails late without it.

## Corpora

```
corpus/                     README.md, celeste_v2_disjointness.json, docs.jsonl, index.json, tokens.i32
corpora/arb_Arab            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/arxiv               README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/c                   README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/ces_Latn            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/cmn_Hani            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/ell_Grek            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/formulas            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/go                  README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/haskell             README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/hin_Deva            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/isabelle            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/javascript          README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/jpn_Jpan            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/lean                README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/owm                 README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/python              README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/rus_Cyrl            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/rust                README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/shell               README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/sql                 README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/tha_Thai            README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/train_parity_10m    README.md, docs.jsonl, meta.json, tokens.i32
corpora/ufw_en              README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
corpora/ufw_zh              README.md, docs.jsonl, index.json, pool.jsonl, pool_windows.i32, stream.json, tokens.i32
```

Numbers from different corpora are NOT comparable: different documents, different nested ladders, different window geometry.

## SAE products

```
sae/l42-1b      README.md, examples, examples_4m, examples_docmax, fire_counts.i64, index.json, max_act.f16, mean_when_active.f16, random_pool, repo_examples, sizes.json, top1_act
sae/sae2m       README.md, examples, examples_docmax, fire_counts.i64, index.json, max_act.f16, mean_when_active.f16, sizes.json
```

## Checkpoints, rollouts and scores

### `2026-07-14_nla-av`

- rollouts (5): `2026-09-16_v1`, `2026-09-20_sae2m_2k`, `2026-09-21_v3_ctrl`, `2026-09-21_v3_realact`, `2026-09-21_v3_sae2m`
- scores (5): `2026-09-16_v1`, `2026-09-20_sae2m_2k`, `2026-09-21_v3_ctrl`, `2026-09-21_v3_realact`, `2026-09-21_v3_sae2m`

### `2026-09-08_rlI-150`

- rollouts (1): `2026-09-16_v1__vllm`
- scores (1): `2026-09-16_v1__vllm`

### `2026-09-10_rl-8x2048-full`

- rollouts (17): `2026-09-16_v1__vllm`, `2026-09-17_l43_rand_v1__vllm`, `2026-09-17_l43_v1__vllm`, `2026-09-18_handpicked_smoke__vllm`, `2026-09-18_handpicked_v1__vllm`, `2026-09-18_handpicked_v1last__vllm`, `2026-09-18_ood_v1__vllm`, `2026-09-18_ood_v1_unitend__vllm`, `2026-09-21_v1raw`, `2026-09-21_v1raw__mu-none`, `2026-09-21_v1raw__mu-stats`, `2026-09-21_v3_ctrl__vllm__mu-none`, `2026-09-21_v3_ours__vllm__mu-none`, `2026-09-21_v3_ours__vllm__mu-stats`, `2026-09-21_v3_realact__vllm__mu-none`, `2026-09-21_v3_realact__vllm__mu-stats`, `2026-09-21_v3_sae2m__vllm__mu-none`
- scores (16): `2026-09-16_v1__vllm`, `2026-09-17_l43_rand_v1__vllm`, `2026-09-17_l43_v1__vllm`, `2026-09-18_handpicked_smoke__vllm`, `2026-09-18_handpicked_v1__vllm`, `2026-09-18_handpicked_v1last__vllm`, `2026-09-18_ood_v1__vllm`, `2026-09-18_ood_v1_unitend__vllm`, `2026-09-21_v1raw__mu-none`, `2026-09-21_v1raw__mu-stats`, `2026-09-21_v3_ctrl__mu-none__vllm`, `2026-09-21_v3_ours__mu-none__vllm`, `2026-09-21_v3_ours__mu-stats__vllm`, `2026-09-21_v3_realact__mu-none__vllm`, `2026-09-21_v3_realact__mu-stats__vllm`, `2026-09-21_v3_sae2m__mu-none__vllm`

### `2026-09-16_base-control`

- rollouts (3): `2026-09-16_v1__vllm`, `2026-09-18_handpicked_v1__vllm`, `2026-09-18_ood_v1__vllm`
- scores (3): `2026-09-16_v1__vllm`, `2026-09-18_handpicked_v1__vllm`, `2026-09-18_ood_v1__vllm`

### `2026-09-18_rl-last16-lr5e-7`

- rollouts (11): `2026-09-16_v1`, `2026-09-20_sae2m_2k`, `2026-09-21_ood_q1__vllm`, `2026-09-21_sae131k_2k`, `2026-09-21_v1raw`, `2026-09-21_v3_ctrl__vllm`, `2026-09-21_v3_ours__vllm`, `2026-09-21_v3_realact__vllm`, `2026-09-21_v3_realact_long__vllm`, `2026-09-21_v3_sae2m__vllm`, `2026-09-21_v3_subspace__vllm`
- scores (10): `2026-09-16_v1`, `2026-09-20_sae2m_2k`, `2026-09-21_sae131k_2k`, `2026-09-21_v1raw`, `2026-09-21_v3_ctrl__vllm`, `2026-09-21_v3_ours__vllm`, `2026-09-21_v3_realact__vllm`, `2026-09-21_v3_realact_long__vllm`, `2026-09-21_v3_sae2m__vllm`, `2026-09-21_v3_subspace__vllm`

## Baselines

- **corpus search** (`base/qwen36-27b/scan/`): `2026-09-16_v1`, `2026-09-18_ood_v1`, `2026-09-20_sae2m_2k`, `2026-09-21_ood_q1__ces_Latn__1m`
- **GCG / EPO** (`base/qwen36-27b/gcg/`): `2026-09-16_v1`
- **Patchscopes** (`base/qwen36-27b/patchscopes/`): `2026-09-16_v1`

NOTE on "do we beat the corpus": the scans above are COSINE baselines. The non-cosine comparisons — native activation (`top1_act`), autointerp balanced accuracy, trojan verbatim@n — are separate products and are listed only where they appear above.

## `shared/` — the agreed artifacts

```
shared/eval1             2026-09-21_sae2m_64_feature_ids.txt
shared/mu                README.md, mu.f32, mu.meta.json
shared/ngram-overlap     doc_level_n7.jsonl, hers_n7.exclude.json, hers_n7.jsonl
shared/sae2m-2k          README.md, features.csv, features.parquet, meta.json
```

## Other volumes

**`maemm-trojan-cache`** holds the §3.6 rank-1 trojan work — it is NOT on this volume. Adapters and results:

```
trojan/.preflight
trojan/act_sep.json
trojan/act_sep16.json
trojan/act_theme2.json
trojan/big_corpus_scan.json
trojan/corpus_scan.json
trojan/corpus_vs_maem.json
trojan/corpus_write.json
trojan/corpus_write_smoke.json
trojan/dit27
trojan/dit27_L39
trojan/dit27_L39_ep5_readout.json
trojan/dit27_L40
trojan/dit27_L40_ep1_readout.json
trojan/dit_recover.json
trojan/multi17
trojan/multi17_g1.json
trojan/multi17_g2.json
trojan/multi17_g4.json
trojan/multi17_joint.json
trojan/multi17_short.json
trojan/multi_sep
trojan/multi_sep2
trojan/multi_sep2_1.json
trojan/multi_sep2_2.json
trojan/multi_sep2_3.json
trojan/multi_sep2_4.json
trojan/multi_sep3
trojan/multi_sep3.json
trojan/multi_sep4
trojan/multi_sep4_1.json
trojan/multi_sep4_2.json
trojan/multi_sep4_3.json
trojan/multi_sep4_4.json
trojan/multi_sep4_5.json
trojan/multi_sep_a.json
trojan/multi_sep_b.json
trojan/multi_sep_c.json
trojan/multi_theme
trojan/multi_theme2
trojan/multi_theme2_a.json
trojan/multi_theme2_b.json
trojan/multi_theme2b
trojan/multi_theme2b.json
trojan/multi_theme2c
trojan/multi_theme2c.json
trojan/multi_theme2d
trojan/multi_theme2d.json
trojan/multi_theme_a.json
trojan/multi_theme_b.json
trojan/rank16.json
trojan/readout17.json
trojan/readout17_partial.json
trojan/readout17_rw.json
trojan/readout_sep.json
trojan/readout_sep2.json
trojan/readout_theme.json
trojan/readout_theme2.json
trojan/readout_theme2_write.json
trojan/scan32_0.json
trojan/scan32_1.json
trojan/scan32_2.json
trojan/scan32_3.json
trojan/scan32_4.json
trojan/scan32_5.json
trojan/scan32_6.json
trojan/scan32_7.json
trojan/scan32_8.json
trojan/scan32_9.json
trojan/sep_L42
trojan/sep_L42.json
trojan/svd16.json
trojan/svd16_rw.json
trojan/theme_L42
trojan/theme_L42.json
trojan/trigger_recovery.json
```

`maemm-portable-eval-hf-cache` is a read-only HF cache the trojan app borrows the base model from.

