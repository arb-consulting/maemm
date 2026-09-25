# Tier B collinearity check — our targets against Celeste's v2 training directions

Module M8 (`evals/2026-09-23_implementation-plan.md`), spec §8 item 2/4. Run 2026-09-22 06:34 UTC,
repo commit `46edec41`, product `tierb` on the CPU function, 619 s wall (609.7 s of scan).

    (set -a; . ./.env.local; set +a; export MODAL_PROFILE=maemms; \
     uvx --with pyyaml --with numpy modal run --detach \
       repo-maemm-m8/paper-evals/precompute/modal_app.py --product tierb --base qwen36-27b)

Cost: the repo's ledger prices the CPU function at $0 (`modal_app.USD_PER_S`), so the product
README says `$0.0000`; the real Modal spend is 8 cores x 619 s, about $0.07. No GPU was used.

Product output on the volume: `/vol/base/qwen36-27b/tierb/` (`summary.json`, `hits.jsonl`,
`maxcos.f32` `[1536, 7]`, `README.md`). Code: `precompute/tierb.py`, scan in
`precompute/targets.py` (`open_bank`, `leak_scan`), selftest
`precompute/unit_smoke.py::check_tierb_scan_finds_a_planted_duplicate`.

## Result

**Zero rows above cos 0.999, in every block against every array.** `hits.jsonl` is empty.

| block | rows | mu | count > 0.999 | max cos | p99 | median | argmax |
|---|---|---|---|---|---|---|---|
| `ours_raw` — our 512 realact, `unit(act)` | 512 | none | **0** | 0.718914 | 0.691846 | 0.515498 | set row 164 vs `sft_mix/realact` row 51318 |
| `ours_centred` — the same rows, `unit(act − whiten_mu)` | 512 | `whiten_mu` | **0** | 0.944555 | 0.785208 | 0.523256 | set row 476 vs `sft_mix/realact` row 2639706 |
| `ctrl_sae131k` — the 131k-SAE half of `2026-09-21_v3_ctrl` | 512 | none (encoder columns) | **0** | 0.901504 | 0.774027 | 0.334858 | set row 555 (feature 51211) vs `sft_mix/sae2m` row 52283 |

All six arrays were read at full size — the fetch is complete, nothing was skipped
(`summary.json: "unreadable": []`):

| array | rows | GiB | s | max cos over all three blocks |
|---|---|---|---|---|
| `sft_mix/realact` | 4,000,000 | 38.1 | 274.3 | 0.944555 |
| `sft_mix/sae2m` | 2,000,000 | 19.1 | 135.2 | 0.901504 |
| `sft_mix/sae2m_dec` | 2,000,000 | 19.1 | 134.0 | 0.812743 |
| `rl_pool/realact_ctx64_2048` | 470,566 | 4.5 | 33.3 | 0.798227 |
| `rl_pool/sae2m` | 235,283 | 2.2 | 16.7 | 0.540905 |
| `rl_pool/sae2m_dec` | 235,283 | 2.2 | 16.2 | 0.543459 |
| **total** | **8,941,132** | **85.2** | **609.7** | |

Row counts are asserted against the bundle manifest before the first matmul, so a short array
would have stopped the run by name rather than being scanned to its end and reported clean.

## The two appendix sentences

> No held-out realact target of ours is collinear with any of the 8,941,132 direction rows the v2
> inverter was trained on: at the leak criterion of the bundle's own construction (cos > 0.999),
> zero of our 512 rows hit, with a maximum cosine of 0.945 and a 99th percentile of 0.785. The
> directions are compared uncentred on her side; ours are reported both uncentred (max 0.719) and
> centred on the base's scoring mean (max 0.945), because the two conventions differ by cos 0.977
> and a duplicate visible under one could sit below the threshold under the other.

> The 512 features of the 131k dictionary that carry our out-of-dictionary row are likewise not
> collinear with the 2M-SAE columns the inverter was trained on: zero of 512 above cos 0.999, max
> 0.902. This is a statement about collinearity only — tier B holds no 131k directions, and for
> the 2M dictionary itself the exclusion is exact at feature-id level and needs no cosine.

## What the check does and does not buy

* **`ours_*` is the new information.** Her v3 realact 512 was already covered by her own leak
  check at the same threshold (0 rows dropped, max cos 0.908); our 512 had never been compared.
* **`ctrl_sae131k` tests collinearity, not membership.** Tier B holds no 131k directions — the
  legacy chain is tier C — so this says the 512 columns are not near-duplicates of the 2M columns
  she trained on. It says nothing about the 2M SAE's own exclusion, which is exact at feature-id
  level.
* **Nothing here is a document- or content-level statement.** Those are
  `paper-evals/docs/check_v2_disjointness.py` and `infra/check_v2_targets_overlap.py`, which
  answer a different question.

## Two things the record should gain

1. **Her rows look centred, not raw** (INFERENCE, not a measurement of her pipeline). Question 3
   of the bundle survey — are v2 realact rows `unit(act)` or `unit(act − mu)` — is still
   unanswered. The two readings here point one way: against her realact bank our *centred* rows
   reach max 0.9446 and our *uncentred* rows only 0.7189, while her own realact bank's recorded
   maxima against real activations are 0.908 (eval `realact`) and 0.983 (`pool_heldout/realact`).
   A 0.72 ceiling between two banks of real layer-42 activations of the same model on the same
   dataset is implausibly low next to that range; 0.94 sits inside it. Treat this as evidence for
   `unit(act − mu)` on her side, to be confirmed by asking her, not as a settled fact.
2. **The centring arm is not redundant, and the spec's default would have been the weaker read.**
   The spec (§8 item 2) says to run the check uncentred on both sides; that arm is the one with
   the *lower* maxima throughout. The conclusion is the same at both conventions (zero hits either
   way), so nothing changes here — but if a future block does hit, the uncentred arm alone would
   be the one likelier to miss it.
