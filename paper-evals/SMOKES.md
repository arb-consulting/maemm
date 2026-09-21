# SMOKES

Running log of everything actually executed for `paper-evals/`: what ran, how long, what it cost,
what it proved, and what disagreed with expectation. Rates: H100 $3.95/h, H200 $4.54/h; Modal CPU
containers at 4 cores are ≈$0.0002/s and are recorded as ~$0.

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-15 | HF fetch: Qwen3.6-27B, `ceselder/qwen36-27b-sae-l42-1b`, `ceselder/qwen36-27b-maemm-inverter-rl-8x2048` | `uvx modal run infra/fetch_hf.py --items qwen3.6-27b,sae-27b-1b,full-27b-rl8x2048 --parallel` | 265.3 s | ~$0.02 (4-CPU containers) | 108.55 GiB, 73 files, all three on `/vol/hf` | the 27B base now has **15** shards (51.77 GiB), not the 12 the older card implied; the full RL model has 12 + `model-nontext.safetensors` |
| 2026-09-15 | unit smoke, local | `uv run paper-evals/precompute/unit_smoke.py` | 1.3 s | $0 | 13/13 checks passed | two test-side bugs found and fixed while writing (NaN-poisoned `max()`, a float32 literal) |
| 2026-09-15 | mutation battery on `common.py` | 10 deliberate defects injected one at a time, smoke re-run | ~15 s | $0 | 10/10 caught | `agg` initially survived "ignore the keep mask" (the fixture had NaN wherever `keep` was False) and the norm-filter check reported the wrong assert first; both tests strengthened |
| 2026-09-15 | `check` product, both bases (CPU) | `uvx --with pyyaml modal run .../modal_app.py --product check` | 18.9 s | ~$0 | every snapshot, SAE file and MAEMM weight dir resolved; both prompts built | — |
| 2026-09-15 | `unit` product in the image (CPU) | `... --product unit` | 16.7 s | ~$0 | 13/13 checks passed inside the Modal image | — |
| 2026-09-15 | local dispatch guards | `... --product corpus` (no base), `--product nope` | <5 s | $0 | both refused locally, no container started | — |
| 2026-09-15 | `corpus` 8B, 2M tokens | `--product corpus --base qwen3-8b --tokens 2000000 --root /vol/runs/2026-09-15_paper-evals-smoke` | 50.9 s | ~$0 | 2,443 docs / 2,001,973 tokens (tag 1: 980,009), parts 0009 rows 0..1224 and 0010 rows 0..1217, longest doc 27,411 tokens | first attempt died on `OfflineModeIsEnabled`: `huggingface_hub` reads `HF_HUB_OFFLINE` **at import** into `constants`, so setting `os.environ` inside the product is not enough (`corpus._go_online` patches the constant) |
| 2026-09-15 | `corpus` 8B determinism rebuild | same, `--root /vol/runs/2026-09-15_paper-evals-det` | 41.2 s | ~$0 | `tokens.i32` sha256 `2f105082…927de` and `docs.jsonl` sha256 `736ba1f4…c213` **byte-identical** to the first build | scratch root deleted afterwards (`modal volume rm -r`) |
| 2026-09-15 | `corpus` 27B, 1M tokens | `--product corpus --base qwen36-27b --tokens 1000000 --root …smoke` | 40.5 s | ~$0 | 1,205 docs / 1,015,530 tokens, 550 of them ≥ 512 tokens | the 27B tokenizer yields ~17% fewer tokens for the same text budget, so its doc count is not comparable to the 8B's |
| 2026-09-15 | **hard-exit finding** | `--product corpus` with `hard_exit()` in the Modal function body | — | ~$0 | the run finished and committed, then `os._exit(0)` killed the container mid-call; **Modal re-scheduled the input** and the retry failed with "already exists" | checklist item 59 is now handled by dropping the stream iterators + `gc.collect()` inside `corpus.py`; no `os._exit` in any Modal function body |
| 2026-09-15 | `stats` 8B, 2M | `--product stats --base qwen3-8b --root …smoke --acts-max-tokens 100000` | 511.2 s | **$0.5609** | mu over 7,593,253 scanned positions; 36-layer norm quantiles; 65,536-feature fire counts (0 dead); acts_1m 81,938 × 4096 | 144 s of that is the base load; estimate was ~$0.4, actual $0.56 with the load, $0.33 without |
| 2026-09-15 | `stats` 27B, 1M | `--product stats --base qwen36-27b --root …smoke --acts-max-tokens 100000` | 601.6 s | **$0.7587** | 3,857,418 positions; 64-layer norm quantiles; 131,072 features (62 dead); acts_1m 99,274 × 5120 | the 51.8 GiB load took **41 s** (warm volume page cache), not the minutes budgeted; estimate ~$1, actual $0.76 |
| 2026-09-15 | `targets` 8B | `--product targets --base qwen3-8b --root …smoke --allow-short` | 167.0 s | **$0.1833** | realact 512 / random 512 / sae 512; presample median raw norm **415.6** (Celeste's bank reports 416.46); **0 leakage hits** over 452,570 archived bank rows | `--allow-short` was not needed: 6,515 features survived both the ≥ 20-gated-fires cut and the training-feature exclusion, enough for 128 per quartile |
| 2026-09-15 | `targets` 27B | `--product targets --base qwen36-27b --root …smoke --allow-short` | 121.7 s | **$0.1535** | 512 / 512 / 512 + the empty `jlens` slot; 114,868 eligible features; leakage check skipped (no 27B bank in the archive) | the realact pool was only 550 documents (all the ≥ 512-token docs at 1M), so 512 targets is nearly the whole pool — at 16M it is a 1,024-doc sample |
| 2026-09-15 | `scan` 8B, 2M (v1) | `--product scan --base qwen3-8b --root …smoke` | 244.8 s | $0.2686 | superseded | its SAE example `acts` were zero-padded to 64 for short windows, indistinguishable from a genuinely inactive token |
| 2026-09-15 | `scan` 8B, 2M (rerun) | same, `--force` | 258.6 s | **$0.2838** | 118,928 windows, 1,536 targets, 3,072 top-64 rows, 512 example files + `_random256.jsonl` | `--force` correctly replaced both the scan dir and the examples dir |
| 2026-09-15 | `scan` 27B, 1M | `--product scan --base qwen36-27b --root …smoke` | 420.6 s | **$0.5305** | 60,411 windows, 1,536 targets, 1,536 top-64 rows, 512 example files | — |
| 2026-09-15 | sanity checks, local | numpy + the real tokenizers on ≤ 20 MB of fetched outputs per base | ~1 min | $0 | 8B: 18 + 17 checks pass; 27B: 16 checks pass (list below) | two apparent failures were the checker's tolerance (f16 payload rounding, and the single window whose activation IS the corpus peak landing on the q3 edge), not the product |
| 2026-09-15 | `targets --import-run1`, 8B (CPU) | `--product targets --base qwen3-8b --import-run1 --n 16 --root …smoke` | 4.7 s | ~$0 | 48 rows (realact/random/sae x 16) out of `/vol/archive/gavento-1/data/run1/eval_cache/eval_sets_heldout.pt`; realact row 0 -> dir_index 40, sae row 0 -> dir_index 26929 / feature 41438, matching `per_dir_final.json` | the cache's `meta["rows"]` has **no `random` key** (the random family is generated, not drawn from a pool), so `dir_index` is null there; keys present: `corpus_peak, meta, random_dirs, realact_dirs, realact_long_dirs, sae_dirs, sae_feats` |
| 2026-09-15 | `rollouts_hf` 8B run1, 48 targets x 64 | `--product rollouts_hf --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --n 64 --root …smoke` | 85.9 s | **$0.0942** | 3,072 rollouts, 3,092 generated tok/s, 12 generate calls of 256 rows, no assert fired | marker \|\|h\|\| served **78.0** vs clean base **14.5** — the base matches the expected ~14.06, but the served value is **78, not the ~98.5 the brief expected**; the guard only requires them to differ, so this passed. Worth chasing: 98.5 may be an RL-step value rather than `final`'s, or a vLLM-side number |
| 2026-09-15 | `score` 8B, first attempt | same `--product score` | ~30 s | ~$0.03 (est.) | **FAILED**: `a scored row kept 65 tokens but a rollout is at most max_new=64` | a real finding, not a bug in the pipeline: decode→re-encode is NOT length-preserving (a rollout cut mid-word at 64 re-tokenizes to 65 ids). The item-8 bound was replaced by two exact ones — stored generated ids `<= max_new`, and no scored row reaching the 95-token truncation |
| 2026-09-15 | `score` 8B run1, 48 x 64 | `--product score --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --root …smoke` | 30.1 s | **$0.0330** | 3,072 rows in 5.8 s (531 rows/s); 250,420 gated SAE entries (mean 81.5 per rollout); round trip `decode(scored ids) == text` on **3,072/3,072** rows | — |
| 2026-09-15 | `score --rescore-texts`, archived run1 texts | `… --rescore-texts /vol/…/_inbox/run1_ss16.jsonl --score-name …__rescore-ss16 --no-sae` | 18.9 s | **$0.0207** | 192 archived rollouts (48 dirs x 4) rescored through our scorer | — |
| 2026-09-15 | `repro_run1.py`, local | `uv run paper-evals/reconstruction/repro_run1.py --ours … --dumps …` | ~2 s | $0 | table below: bo1 within 0.0064 on realact and 0.0000 on sae, best-of-64 paired diff +0.0011 / −0.0017 / +0.0003 with r 0.995 / 0.982 / 0.924, **scorer agreement r = 1.000000** | — |
| 2026-09-15 | `rollouts_hf` 27B rlI-150 (LoRA), rows 0-7 realact x 64 | `--product rollouts_hf --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150 --set 2026-09-16_v1 --rows 0-7 --n 64 --root …smoke` | 214.5 s | **$0.2705** | 512 rollouts, **239.3 generated tok/s** at 32 rows/call, no assert fired; marker \|\|h\|\| served **512.0** vs clean base **14.0625** | 239 tok/s is exactly the ~240 the brief budgeted. The clean-base 14.0625 was written into `config.yaml` as `bases.qwen36-27b.marker_norm_base` so the full model could be checked without a second 52 GiB load |
| 2026-09-15 | `score` 27B rlI-150 | `--product score --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150 --set 2026-09-16_v1 --rows 0-7 --root …smoke` | 94.3 s | **$0.1189** | mean cos **0.4992**, best-of-64 **0.5849** over the 8 realact rows; mean len 53.4, eos rate 0.803 | 30 s of the 94 s is the base load |
| 2026-09-15 | `rollouts_hf` 27B rl-8x2048 (FULL), rows 0-7 realact x 64 | `--product rollouts_hf --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full --set 2026-09-16_v1 --rows 0-7 --n 64 --root …smoke` | 182.6 s | **$0.2302** | 512 rollouts, **341.5 generated tok/s**, 52 GiB load in 47 s, no assert fired; marker \|\|h\|\| served **131.0** vs the config'd base 14.062 | the full model is FASTER per token than the LoRA (341 vs 239 tok/s): no adapter matmuls. Weight identity is `index.json` + shard sizes, not a content hash (`common.sha256_of_index`) |
| 2026-09-15 | `score` 27B rl-8x2048 full | `--product score --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full --set 2026-09-16_v1 --rows 0-7 --root …smoke` | 102.6 s | **$0.1293** | mean cos **0.5076**, best-of-64 **0.5929**; mean len 45.5, eos rate 0.977 | the clean base is loaded here, never the 52 GiB MAEMM — that is the whole point of the rollouts/scores split |

## Step 4 (2026-09-15): centring rule, `mu_check`, and the vLLM engine

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-15 | `acts_1m` dropped | `modal volume rm -r maemm …smoke/base/<base>/acts_1m` (both bases) | — | $0 | the corpus activation store is gone from `stats.py`, `common.py` (`acts_1m_dir`, `blocks_of`, `ACTS_BLOCK`), the `--acts-max-tokens` flag, the README layout, and the volume | `common.blocks_of` and its unit check went with it: the store was its only consumer and a tested dead helper is worse than none |
| 2026-09-15 | `check`, both bases, with the 3 new `compute: false` MAEMMs | `--product check` | 17.9 s | ~$0 | `qwen3-8b/2026-09-03_run1-sft` resolved off the archive (666.1 MiB); `qwen36-27b/2026-09-05_rlE-250` resolved; the two unfetched entries reported as `compute: false, NOT FETCHED` and did NOT fail the gate | `ceselder/maemm-init-realact23m-sft` and `ceselder/qwen36-27b-maemm-inverter-pretrain-104m` are not in the volume's HF cache. `check` resolves a `compute: false` entry LENIENTLY (a computable one that does not resolve is still a hard failure) — see the ambiguity note below |
| 2026-09-15 | `mu_check` 8B (CPU) | `--product mu_check --base qwen3-8b --root …smoke` | 2.1 s | ~$0 | **cos 0.989258**, ‖ours‖ 286.55 vs ‖theirs‖ 301.34, ratio 0.950924, rel L2 0.151 | **below the 0.99 floor** — reported, not acted on. Ours is the 2M-token smoke corpus; Celeste's is run1's full activation bank |
| 2026-09-15 | `mu_check` 27B (CPU) | `--product mu_check --base qwen36-27b --root …smoke` | 2.3 s | ~$0 | **cos 0.977523**, ‖ours‖ 67.90 vs ‖theirs‖ 67.26, ratio 1.009439, rel L2 0.213 | also below the floor, and further than the 8B although its norms agree to 1% |
| 2026-09-15 | `targets` 8B, re-drawn on `stats/mu.f32` | `--product targets --base qwen3-8b --root …smoke --allow-short --force` | 63.6 s | **$0.0697** | 512/512/512, 0 leakage hits, presample median raw norm 415.6 (unchanged); diagnostic **mu_512 vs stats/mu: cos 0.9900, ‖mu_512‖/‖mu‖ 1.0422** | 63.6 s against the first run's 167 s — the base weights were warm in the volume page cache |
| 2026-09-15 | `targets` 27B, re-drawn on `stats/mu.f32` | `--product targets --base qwen36-27b --root …smoke --allow-short --force` | 145.8 s | **$0.1839** | 512/512/512 + the empty `jlens` slot; diagnostic **mu_512 vs stats/mu: cos 0.9774, ‖mu_512‖/‖mu‖ 0.9898** | the 27B's two means are FURTHER apart than the 8B's (0.977 vs 0.990) while their norms are closer |
| 2026-09-15 | `scan` 8B re-run on the new targets | `--product scan --base qwen3-8b --root …smoke --force` | 244.2 s | **$0.2679** | 118,928 windows, 1,536 targets, 512 example files | the 27B `scan` was NOT re-run (deferred, ~$0.53): its held-out set is now centred differently from its scan, so **`…smoke/base/qwen36-27b/scan/2026-09-16_v1` is STALE** and must be re-run before any 27B scan number is used |

### What the centring change did to the 27B numbers (2026-09-15)

`targets` is deterministic given (seed, corpus): the same rng stream draws the same pool, the same
`p` and the same `L`, and the raw-norm filter is computed on UNCENTRED activations. Evidence that
the stream really is unchanged: the 8B presample median raw norm came out at **415.6** in both the
step-2 draw and the step-4 redraw. So rows 0-7 of `2026-09-16_v1` are the SAME (document, position,
span) as before; only the vector changed, from `unit(X[p] - mu_512)` to `unit(X[p] - stats/mu)`.

That makes the following a clean measurement of the centring rule, on the 27B, 8 realact
directions x 64 HF rollouts of `qwen36-27b/2026-09-08_rlI-150`:

| | rlI-150 on `mu_512` | rlI-150 on `stats/mu` | change | rl-8x2048 on `mu_512` | rl-8x2048 on `stats/mu` | change |
|---|---|---|---|---|---|---|
| mean bo1 | 0.4992 | **0.4360** | **-0.0632** | 0.5076 | **0.4694** | **-0.0382** |
| mean best-of-64 | 0.5849 | **0.5496** | **-0.0353** | 0.5929 | **0.5430** | **-0.0499** |
| mean rollout length | 53.4 | 55.8 | +2.4 | 45.5 | 47.5 | +2.0 |
| eos rate | 0.803 | 0.715 | -0.088 | 0.977 | 0.949 | -0.028 |

(The 27B HF re-baseline cost $0.3155 + $0.1570 for rlI-150 and $0.2304 + $0.1446 for the full
model. It was not in the step-4 plan; it became necessary because A2's redraw invalidated every
step-3 27B direction, and without it the vLLM comparison would have been against rollouts of a
different target set.)

**The one-mean rule costs 0.04-0.06 of realact cosine on the 27B**, on both MAEMMs and on both
statistics. That is not an argument against it
— the two conventions measure different things and the scorer is uncentred either way — but it
means every 27B realact number from step 3 is on the old convention and must not be quoted beside a
step-4 one. Note also that on the new centring the full-parameter model is AHEAD on bo1 (0.4694 vs 0.4360) and
BEHIND on bo64 (0.5430 vs 0.5496), where on the old one it led on both — eight directions, so this
is a caution about quoting either ordering, not a result. It also makes `mu_check`'s 0.9775 cosine
on the 27B the more interesting of the two bases: the mean we now centre on is further from Celeste's than the 8B's is, and the realact
penalty is in the same direction.

### Step 4, part B/C: the vLLM engine

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-15 | `parity_greedy` 8B, first attempt | `--product parity_greedy --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --rows 0-7 --root …smoke` | ~60 s | ~$0.05 (est.) | **FAILED**: `AttributeError: '_GeneratorContextManager' object has no attribute 'args'` | my bug: `common.hooked` is a one-shot generator context manager and I entered the same object twice. Fixed by `_hook_ctx()`, which builds a fresh one per `with` |
| 2026-09-15 | `parity_greedy` 8B, 8 realact dirs | same, `--force` | 86.0 s | **$0.0944** | engine `‖h‖` **78.261** vs HF **78.000** (0.33%); injection cos **0.999993**, magnitude ratio **0.999937**, pre-marker delta **0.0**; first greedy token matches **8/8** with a first-token logprob gap of **0.0**; greedy match length mean **24.6** tokens, fully identical on **4/8**; teacher-forced ǀΔlogpǀ **0.0184** nats mean / 0.104 p99 / 0.153 max over 323 tokens, vs **1.961** nats with the hook off; the injection changes the greedy text on 8/8 | the STOCK `vllm_lens._worker_ext.HiddenStatesExtension` passes every check, so no copy of `fast_lens_ext.py` was needed and `precompute/vllm_ext.py` was NOT written. vLLM engine up in 42 s at `gpu_memory_utilization` 0.55 with the 15.6 GiB HF model resident beside it |
| 2026-09-15 | `rollouts_vllm` 8B run1, 48 targets x 64 | `--product rollouts_vllm --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --n 64 --root …smoke` | 280.9 s | **$0.3082** | 3,072 rollouts in ONE generate call of 48 requests x n=64; marker ‖h‖ 78.2619 vs the HF run's 78.0 (0.34%, asserted ≤ 3%); injection cos 0.999993 ratio 1.000151; mean 31.98 tokens, eos rate 0.936; **0** stop tokens re-appended | **vLLM is 5.7x SLOWER than HF here: 543.4 generated tok/s against the HF path's 3,092.** The stock lens hook rescans every steering key on every layer of every decode step (that is why `fast_lens_ext` exists); with 48 keys x 36 layers x up to 256 concurrent rows that dominates. Also: vLLM did NOT drop the stop token on any row, so `common.vllm_finish_ids`' re-append path never fired here — it stays unit-tested rather than run-tested |
| 2026-09-15 | `rollouts_vllm` 27B rlI-150, first attempt | `--product rollouts_vllm --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150 --set 2026-09-16_v1 --rows 0-7 --n 64 --throughput s64 --root …smoke` | ~250 s | ~$0.25 (est.) | **FAILED on my own assert**: `a row BEFORE the marker moved by 2.181e+00 under steering (max allowed 0.01)` | everything else in that run PASSED and is the evidence that matters: the 27B adapter was renamed **992/992 tensors**, the engine's marker ‖h‖ came out **511.87** against the HF run's **512.0** (0.03% — the adapter is applied; an ignored one gives 14.06), injection cos **0.999995**, ratio **0.999994**. The `1e-2` pre-marker bound was calibrated on the 8B, where the delta is exactly 0.0; on the 27B, whose layer-1 residual norms are in the hundreds, the engine's own run-to-run variation is O(1). The check now captures a SECOND clean request and bounds the steered pre-marker delta by max(2% of the injected magnitude, 3x that measured noise) — 2.181 is 0.43% of the 511.87 injected here |
| 2026-09-15 | `score --engine vllm` 8B, 48 x 64 | `--product score --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --engine vllm --root …smoke` | 38.6 s | **$0.0424** | 3,072 rows scored on the clean base into `scores/2026-09-03_run1-archive16__vllm/` | `--engine` only picks the rollouts stem and the output directory; the scorer is byte-identical, which is what makes the table below a comparison of ENGINES and nothing else |

| 2026-09-15 | `rollouts_vllm` 27B rlI-150 (LoRA) + the concurrency sweep | `--product rollouts_vllm --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150 --set 2026-09-16_v1 --rows 0-7 --n 64 --max-num-seqs 128 --throughput s128 --root …smoke` | 380.3 s | **$0.4796** | 512 rollouts at **246.8** gen tok/s; adapter **992/992 tensors renamed**; marker ‖h‖ **511.87** vs HF **512.0** (0.03%); injection cos 0.999995 ratio 0.999984; pre-marker delta **0.1948** against a clean-vs-clean **0.1979** | engine up in 175 s. The pre-marker delta being BELOW the engine's own clean-vs-clean noise is the cleanest statement of "the injection does not leak backwards" in this step |
| 2026-09-15 | `score --engine vllm` 27B rlI-150 | `--product score --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150 --set 2026-09-16_v1 --rows 0-7 --engine vllm --root …smoke` | 96.2 s | **$0.1213** | bo1 **0.4325**, bo64 **0.5478** | — |
| 2026-09-15 | `rollouts_vllm` 27B rl-8x2048 (FULL, no LoRA) | `--product rollouts_vllm --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full --set 2026-09-16_v1 --rows 0-7 --n 64 --root …smoke` | 295.0 s | **$0.3720** | 512 rollouts at **252.1** gen tok/s; marker ‖h‖ **130.08** vs HF **131.0** (0.70%); injection cos 0.999995 ratio 0.999957; pre-marker delta **0.0**, clean-vs-clean **0.0** | the full model is bit-reproducible across identical requests where the LoRA engine is not — so the 27B nondeterminism is the LoRA (Punica) path's, not the GatedDeltaNet layers' |
| 2026-09-15 | `score --engine vllm` 27B rl-8x2048 full | `--product score --base qwen36-27b --maemm qwen36-27b/2026-09-10_rl-8x2048-full --set 2026-09-16_v1 --rows 0-7 --engine vllm --root …smoke` | 95.8 s | **$0.1209** | bo1 **0.4744**, bo64 **0.5408** | — |

## HF vs vLLM on the 27B, 8 realact directions x 64 rollouts (2026-09-15)

Both MAEMMs on `2026-09-16_v1` rows 0-7, the SAME directions (re-drawn on `stats/mu.f32`), scored
on the clean 27B base by the same `score.py`.

| | rlI-150 (LoRA) HF | rlI-150 vLLM | diff | rl-8x2048 (full) HF | rl-8x2048 vLLM | diff |
|---|---|---|---|---|---|---|
| mean bo1 | 0.4360 | **0.4325** | **-0.0035** | 0.4694 | **0.4744** | **+0.0051** |
| mean best-of-64 | 0.5496 | **0.5478** | **-0.0019** | 0.5430 | **0.5408** | **-0.0023** |
| mean rollout length | 55.8 | 55.9 | +0.1 | 47.5 | 47.1 | -0.4 |
| eos rate | 0.715 | 0.709 | -0.006 | 0.949 | 0.965 | +0.016 |
| marker ‖h‖ served | 512.0 | 511.87 | 0.03% | 131.0 | 130.08 | 0.70% |
| generated tok/s | 239.3 (32 rows/call) | 246.8 (128) | — | 341.5 (32) | 252.1 (64) | — |

Eight directions per MAEMM, so the per-MAEMM differences (-0.0035, +0.0051 on bo1) are inside the
sampling noise — which is the point: the two engines are interchangeable for the objective. The
LoRA path is the one that could have failed silently, and did not: a wrongly named adapter would
have returned the clean-base 14.06 instead of 511.87.

### The GatedDeltaNet batch cliff: not on vLLM (2026-09-15)

Measured inside the rlI-150 LoRA engine at `max_num_seqs` 128, varying the rows in flight:

| concurrent rows | 32 | 64 | 128 |
|---|---|---|---|
| requests x n | 2 x 16 | 4 x 16 | 8 x 16 |
| generated tok/s | 122.6 | 204.0 | **298.0** |
| per-row tok/s | 3.83 | 3.19 | 2.33 |
| wall for the call | 14.2 s | 17.2 s | 23.8 s |

**Verdict: no cliff.** Throughput rises monotonically with concurrency and the per-row rate degrades
by 1.64x across a 4x increase — ordinary batching. The HF path is >= 37x slower above batch 64 on
the same layers (checklist item 51), which is why `rollouts_hf` caps the 27B at 32 rows per call;
`rollouts_vllm` has no such reason to.

**SUBSTITUTION, stated plainly.** The plan asked for one fixed request set measured at
`max_num_seqs` 32 / 64 / 128 — three engines, three 52 GiB loads, ~$0.8. This varies the rows in
flight inside ONE engine instead, for ~$0. It answers the cliff question; it does NOT measure how
`max_num_seqs` itself (KV budget, scheduler admission) affects throughput. The faithful version is
three runs of `--product rollouts_vllm --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150
--set 2026-09-16_v1 --rows 0-7 --max-num-seqs {32,64,128} --throughput only`. The reason for the
substitution is budget: the unplanned 27B HF re-baseline (see above) cost $0.85 that was not in the
step-4 plan.

## HF vs vLLM on the 8B, 48 directions x 64 rollouts (2026-09-15)

`reconstruction/parity.py` on `2026-09-03_run1-archive16` — the same 48 directions out of run1's
eval cache, the same `qwen3-8b/2026-09-03_run1-rl` weights, the same clean-base scorer, only the
generation engine differing.

```
# parity: 48 directions x n=64 rollouts, HF vs vLLM, same dirs, same scorer

(a) overall, per family and engine
family   engine  dirs  mean bo1  mean bo64  mean len  eos    gen tok/s
-------  ------  ----  --------  ---------  --------  -----  ---------
random   hf      16    0.0257    0.0437     33.62     0.923  3091.6   
random   vllm    16    0.0261    0.0440     34.21     0.916  543.4    
realact  hf      16    0.4978    0.5978     41.54     0.911  3091.6   
realact  vllm    16    0.4947    0.5934     41.02     0.896  543.4    
sae      hf      16    0.0628    0.1366     20.78     0.993  3091.6   
sae      vllm    16    0.0640    0.1503     20.71     0.995  543.4    

(b) per direction, PAIRED (vLLM - HF); r is across directions
family   dirs  hf bo1  vllm bo1  d bo1    max |d|  r bo1    hf bo64  vllm bo64  d bo64   r bo64 
-------  ----  ------  --------  -------  -------  -------  -------  ---------  -------  -------
random   16    0.0257  0.0261    +0.0004  0.0038   +0.9951  0.0437   0.0440     +0.0002  +0.9068
realact  16    0.4978  0.4947    -0.0031  0.0339   +0.9963  0.5978   0.5934     -0.0044  +0.9913
sae      16    0.0628  0.0640    +0.0012  0.0135   +0.9948  0.1366   0.1503     +0.0137  +0.9025

(c) argmax position as a fraction of the scored tokens (row-normalised histogram)
family   engine  rollouts  mean   0.0-   0.1-   0.2-   0.3-   0.4-   0.5-   0.6-   0.7-   0.8-   0.9- 
-------  ------  --------  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----
random   hf      1024      0.656  0.031  0.051  0.060  0.057  0.083  0.114  0.102  0.113  0.138  0.252
random   vllm    1024      0.640  0.032  0.057  0.068  0.075  0.082  0.102  0.097  0.113  0.143  0.231
realact  hf      1024      0.935  0.002  0.004  0.005  0.010  0.015  0.024  0.041  0.031  0.041  0.827
realact  vllm    1024      0.935  0.002  0.003  0.007  0.014  0.016  0.022  0.032  0.030  0.039  0.835
sae      hf      1024      0.571  0.118  0.054  0.093  0.049  0.082  0.110  0.079  0.076  0.111  0.228
sae      vllm    1024      0.583  0.098  0.044  0.072  0.090  0.088  0.108  0.076  0.082  0.121  0.221

(d) both engines' best-of-64 vs run1's ARCHIVED best-of-64, joined on dir_index
family   dirs  joined  hf bo64  vllm bo64  archive bo64  d hf     d vllm   r hf / r vllm    
-------  ----  ------  -------  ---------  ------------  -------  -------  -----------------
random   16    0       -        -          -             -        -        no dir_index join
realact  16    16      0.5978   0.5934     0.5967        +0.0011  -0.0033  +0.995 / +0.990  
sae      16    16      0.1366   0.1503     0.1383        -0.0017  +0.0120  +0.982 / +0.890  

Notes: the two engines do NOT share an RNG stream (vLLM seeds per request, HF per generate call), so (b) is distributional agreement per direction, never bitwise. (d) is our n rollouts against the archive's 64, both as best-of-k of the same estimator.
```

What this settles:

- **The two engines agree per direction.** Paired r is 0.995-0.996 on bo1 in every family, and
  0.99 (realact) / 0.90-0.91 (sae, random) on bo64 — the weaker bo64 correlations are the families
  whose cosines sit near the floor, where a best-of-64 is one lucky draw. The mean paired
  differences are -0.0031 (realact), +0.0012 (sae), +0.0004 (random) on bo1, i.e. inside the
  rollout-to-rollout noise.
- **Lengths, eos rates and argmax positions line up.** realact rollouts put their best token in the
  last tenth of the text 83% of the time on BOTH engines; the whole histogram matches bin for bin.
- **Both engines land on run1's archived numbers.** realact best-of-64: ours 0.5978 (HF) / 0.5934
  (vLLM) against the archive's 0.5967, r 0.995 / 0.990.
- **What it does not settle**: the sae family's bo64 is +0.0137 on vLLM with r 0.90. On 16
  near-floor directions that is not a difference worth chasing, but it is the one cell where the
  engines are further apart than the paired noise elsewhere.

**Total GPU spend for step 4: $3.47.** Part A $0.52 (targets 8B $0.070 + targets 27B $0.184 + scan
8B $0.268; `check`, `mu_check` and the volume deletions ~$0). Part B/C $2.95: 8B $0.45
(parity_greedy $0.094 + its failed first attempt ~$0.05 + rollouts_vllm $0.308 + score $0.042),
27B vLLM $1.35 (rlI $0.480 + $0.121, full $0.372 + $0.121, plus ~$0.25 for the run my own
pre-marker assert killed), and the unplanned 27B HF re-baseline $0.85 ($0.316 + $0.157 + $0.230 +
$0.145). Budget was $4.00. No single run exceeded twice its estimate; the two overruns against the
plan were both mine — the HF re-baseline (forced by A2's redraw) and the two failed runs.

**Total GPU spend for step 2 (`corpus`/`stats`/`targets`/`scan`): $2.74** (8B $1.30 incl. the discarded scan v1, 27B $1.44), inside the $5 budget.

**Step 3 (rollouts_hf, score, repro) GPU spend: $0.93** — 8B $0.18 (incl. ~$0.03 for the run whose
item-8 assert fired), 27B $0.75 — against a $4.5 budget. No run exceeded twice its estimate; the
27B runs came in well under (B was budgeted ~$1.2 and cost $0.39 including its score; C ~$1.5 and
cost $0.36).

## Reproduction of run1's archived numbers (2026-09-15)

`reconstruction/repro_run1.py` against `2026-09-03_run1-archive16` — 16 directions per family out
of run1's own eval cache, our 64 rollouts of `qwen3-8b/2026-09-03_run1-rl` vs the archived
`ss_samples.jsonl` (bo 4), `per_dir_final.json` (bo 4) and `per_dir_final_bo64.json` (bo 64, which
also carries its 64 per-rollout cosines per direction).

```
# repro_run1: 16 directions per family, ours n=64 rollouts, archive bo4 + bo64

(a) bo1 (per-rollout) means
family   ours (16x64)      archive bo4 (16x4)  archive bo64 (16x64)  d vs bo4  d vs bo64
-------  ----------------  ------------------  --------------------  --------  ---------
realact  0.4978 +- 0.0053  0.4938 +- 0.0210    0.4914 +- 0.0053      +0.0041   +0.0064  
sae      0.0628 +- 0.0021  0.0723 +- 0.0090    0.0628 +- 0.0020      -0.0096   -0.0000  
random   0.0257 +- 0.0004  0.0259 +- 0.0020    0.0258 +- 0.0004      -0.0002   -0.0001  

(b) per direction: our mean-of-64 vs their best-of-k (different statistics; r is the point)
family   our mean64  their bo4  r       mean d   their bo64  r       mean d 
-------  ----------  ---------  ------  -------  ----------  ------  -------
realact  0.4978      0.5502     +0.975  -0.0524  0.5967      +0.983  -0.0989
sae      0.0628      0.1085     +0.771  -0.0458  0.1383      +0.719  -0.0755
random   0.0257      0.0340     +0.904  -0.0083  0.0435      +0.921  -0.0178

(c) our best-of-64 vs their best-of-64, per direction (paired)
family   ours bo64  theirs bo64  mean d   max |d|  r     
-------  ---------  -----------  -------  -------  ------
realact  0.5978     0.5967       +0.0011  0.0480   +0.995
sae      0.1366     0.1383       -0.0017  0.0363   +0.982
random   0.0437     0.0435       +0.0003  0.0110   +0.924

(d) rollout length
family   our n_tok  their n_tok  diff   ours at 64  theirs at 64  our eos
-------  ---------  -----------  -----  ----------  ------------  -------
realact  41.54      39.91        +1.64  0.101       0.156         0.911  
sae      20.78      19.30        +1.48  0.008       0.000         0.993  
random   33.62      31.34        +2.28  0.081       0.078         0.923  

(e) THEIR texts through OUR scorer vs their stored cos_orig (no norm filter vs 10x median)
family   rows  ours     theirs   mean d     max |d|   r         frac ours>=  frac equal
-------  ----  -------  -------  ---------  --------  --------  -----------  ----------
realact  64    0.49375  0.49375  -1.40e-06  3.07e-05  1.000000  1.000        1.000     
sae      64    0.07232  0.07232  -7.06e-07  6.75e-06  1.000000  1.000        1.000     
random   64    0.02590  0.02590  +6.75e-07  7.06e-06  1.000000  1.000        1.000     

Notes: SEs are naive (rollouts of one direction are correlated). (a)-(d) are distributional: both sides use GEN_SEED 1234 but not the same RNG stream.
```

What this settles and what it does not:

- **The scorer is the archived scorer.** (e) is agreement to 1e-6 with `r = 1.000000` on all three
  families and `frac equal = 1.000` at a 1e-4 tolerance — the residual is f16 payload rounding, not
  a protocol difference. It also MEASURES the open question about the 10x-nanmedian norm filter:
  on the 8B it dropped **nothing** (ours, which applies no filter, is never higher than theirs),
  matching the checklist's note for the 27B. The divergence we took is empirically inert here.
- **The sampler agrees distributionally.** (a): realact 0.4978 vs their 0.4914 over the same
  16 x 64 shape (+0.0064), sae 0.0628 vs 0.0628, random 0.0257 vs 0.0258 — all inside the ~0.02
  the brief asked for, and the sae figure matches the 2026-09-15 Patchscopes trial's 0.0631/0.0628
  almost exactly. (c) best-of-64 per direction: +0.0011 / −0.0017 / +0.0003 mean paired difference
  with r = 0.995 / 0.982 / 0.924, inside the ~0.03 and r > 0.9 asked for.
- **(b) is a comparison of different statistics** and is reported for the RANKING only: a
  mean-of-64 is structurally below a best-of-k, so the mean differences (−0.05 to −0.10) are
  expected; r = 0.975 / 0.983 on realact is the number that matters, and sae's r = 0.72-0.77 is the
  weakest link in the set (16 directions, and the sae family's cosines are near the floor).
- **Our rollouts run 1.5-2.3 tokens longer** (d) with a slightly lower rate of hitting the 64-token
  cap on realact (0.101 vs 0.156). Not explained; the sampling constants and the stop-token set are
  the same, so the candidates are the RNG-stream difference and the eos union. Small enough not to
  move the cosines, and recorded rather than chased.

## What the `check` product printed (2026-09-15)

```
[check] base qwen3-8b: /vol/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46… d=4096 n_layers=36 read_layer=27 gpu=H100
[check] base qwen3-8b prompt ours8b: 103 tokens, marker id 937 at 102 (single occurrence), vocab 151669
[check] sae qwen3-8b/adamkarvonen-t2: …/trainer_2/ae.pt (2.0 GiB)
[check] maemm qwen3-8b/2026-09-03_run1-rl (lora): /vol/archive/gavento-1/runs/run1/rl/final (1.3 GiB)
[check] base qwen36-27b: /vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd… d=5120 n_layers=64 read_layer=42 gpu=H200
[check] base qwen36-27b prompt celeste27b: 103 tokens, marker id 907 at 102 (single occurrence), vocab 248077
[check] sae qwen36-27b/l42-1b: …/resid_post_layer_42/trainer_0/ae.pt (5.0 GiB)
[check] maemm qwen36-27b/2026-09-08_rlI-150 (lora): …/snapshots/52101b1c…/step_150 (1.7 GiB)
[check] maemm qwen36-27b/2026-09-10_rl-8x2048-full (full): …/snapshots/1c50a41d… (51.8 GiB)
[check] maemm qwen36-27b/2026-09-05_rlE-250 (lora): …/snapshots/75b6aaa0… (1.7 GiB)
[check] heldout 2026-09-16_v1 on qwen3-8b: realact=512, random=512, sae=512
[check] heldout 2026-09-16_v1 on qwen36-27b: realact=512, random=512, sae=512, jlens=512 (empty slot)
[check] scoring: max_length=95 chunk=32 width=96 rollouts.max_new=64
[wall] product=check base=all gpu=CPU seconds=18.9 cost=$0.0000
```

Worth noting: the two prompts are both 103 tokens with the marker last, but the marker **token id
differs** (937 on the 8B, 907 on the 27B) because the vocabularies differ (151,669 vs 248,077).
Nothing may hard-code a marker id.

## What the unit smoke covers, and what it deliberately does not

Covered, each against an independent computation (`output_hidden_states`, a hand-built mask, the
formula written out) rather than against the code under test:

- read hook returns the block **output** = `output_hidden_states[L+1]`, for every layer that has one;
- injection at block 1's output changes **only** the marker column at that layer, nothing at the
  layer below, and nothing before the marker at later layers; the value equals
  `base + unit(v)·||base||·coeff`; the decode-step guard is a no-op;
- `marker_norm` equals `||h[pos]||` at the inject layer;
- the scorer's keep mask (BOS never, padding never, content always), NaN outside `keep`, ids −1
  outside it, truncation at 95, the whitespace-only rollout, `padding_side` restored, and the
  cosine/norm of one row recomputed directly;
- no norm filter: a token manufactured at >10× the row median norm stays in `keep`;
- `agg` excludes a **finite** value at a non-kept column (not merely a NaN);
- eos union, stop-token trimming;
- SAE: `nn.Linear` transpose, `bias`→`b_dec` alias, the encode formula, unit encoder columns as
  directions, and a hard failure when the checkpoint has no `threshold`;
- `OutDir`: temp-and-rename, `--force` refusal, `--force` replacement, a failed writer leaving the
  temp dir and no final directory, README/`index.json` contents, jsonl and array round trips;
- `sha256_of_weights` over a directory, and that the combined digest changes with content.

Added in step 3 (22 checks total, all four mutation-tested — a deliberate defect in each was caught
before the check was trusted):

- `best_of_k_means`: every bo_k against a hand-written expected value, that groups are disjoint and
  CONSECUTIVE (a reordered list scores differently), that an odd tail is dropped rather than folded
  in, that `k > n` is skipped rather than clamped, and that `k = 0` raises;
- `parse_rows` (ranges, singletons, duplicate collapse, out-of-range and backwards asserts) and
  `gen_seed_for` (`flat = row * n + k`, the 8B's 256-row call starting at target 4, `k >= n` raising);
- `OutDir(keep_existing=True)`: an existing set survives, `index.json` gains the new one,
  `section()` renders as a real `## ` heading BEFORE the file table, and the plain path is still
  refused without `--force`;
- `sha256_of_index`: shard names and sizes plus the index content, that an equal-SIZED content
  change is INVISIBLE (the documented weakness — if that ever changes the README claim is wrong),
  and that a size change or an index change does move the digest.

Not covered here, and why: the real chat template and marker tokenization (needs a real tokenizer —
the Modal `check` product does it on both bases); anything requiring weights or a GPU; vLLM.

## Open (to verify when the remaining products land)

- SETTLED 2026-09-15: `gpu_h100` and `gpu_h200` both ran real work; `load_base` loaded both bases
  (8B 144 s cold / 7 s warm, 27B 41 s for 51.8 GiB) and `load_sae` loaded both SAEs. Gate values
  MEASURED: **6.936** for the 8B `adamkarvonen-t2` (65,536 features) and **1.5846** for the 27B
  `l42-1b` (131,072) — the older `-l42` repo's 1.654 quoted in `README.md` is a different
  checkpoint, so nothing may hard-code a gate either.
- SETTLED 2026-09-15: `load_maemm` loaded real weights on all three paths — the 8B LoRA off the
  archive volume path (3 s), the 27B LoRA with its `step_150/` subdir (6 s on top of a 26 s base
  load) and the 52 GiB 27B full model (47 s). The marker-norm guard fired on none of them.
- STILL OPEN after step 4 (NOT settled, contrary to the step-3 note's hope that `rollouts_vllm`
  would settle it cheaply): the 27B forward is right-padded across ragged windows. Padding after the content is
  safe for any causal architecture (including GatedDeltaNet) and matches the scorer's
  `padding_side='right'`, but nothing has yet compared a padded batch against a length-1 batch on
  the 27B. `rollouts_hf` does NOT pad (all rows share one prompt, asserted), so this is still only
  a question about `score` and the corpus passes. Cheap to settle inside `rollouts_vllm`'s parity
  work.
- NARROWED 2026-09-15: the 8B run1 served marker norm is **78.0**, not the ~98.5 the step-3 brief
  quoted. The clean base 14.5 matches. Step 4 adds an INDEPENDENT reading: the vLLM engine, serving
  the same adapter through its own LoRA path, reports **78.26** (0.33% away). So 78 is not an
  artefact of `common.marker_norm` or of the HF path — "a vLLM-side number" is ruled out, and the
  ~98.5 remains unexplained (most likely a training-step value rather than `final`'s). Nothing
  downstream depends on the absolute number.
- SETTLED 2026-09-15 on the 8B: `rollouts_vllm` and the HF-vs-vLLM parity check (checklist item
  19) — the tables above. The 27B leg is in the section that follows. The **stock** vllm-lens
  worker extension was sufficient, so Celeste's `fast_lens_ext.py` was NOT copied into
  `precompute/vllm_ext.py`; `--fast-hook` remains an unexercised code path, and with it the only
  place an extension error counter would be read (self-check (iv) reports `n/a` on the stock path).
- STILL OPEN: vLLM generation here is **5.7x slower** than HF on the 8B (543 vs 3,092 tok/s). The
  stock lens hook is the suspect and `--fast-hook` is the untested remedy. NOTE the 27B is the
  other way round for the LoRA MAEMM (vLLM 246.8 vs HF 239.3 tok/s) and only slightly worse for the
  full model (252.1 vs 341.5): the stock hook's cost scales with (layers x concurrent rows x
  distinct directions), and the 27B runs had 8 directions where the 8B run had 48.
- STILL OPEN: the 32 / 64 / 128 throughput sweep was measured by varying the rows in flight inside
  ONE engine, not by building three engines at those `max_num_seqs` values (budget; see the
  substitution note). The cliff question is answered; the effect of `max_num_seqs` itself is not.
- STILL OPEN: `common.vllm_finish_ids`' stop-token re-append path has never fired in a real run —
  vLLM returned the stop token on every row of all three MAEMMs. It is unit-tested and
  mutation-tested, not run-tested.
- STILL OPEN: `precompute/vllm_ext.py` was never written and `--fast-hook` therefore names a module
  that does not exist. The stock extension passed every check, so nothing needed it; anyone passing
  `--fast-hook` today gets an import error, not a fallback.
- STILL OPEN: the 27B `scan` is STALE. Its held-out set was re-drawn on `stats/mu.f32` in step 4 and
  the scan was not re-run (~$0.53). `runs/2026-09-15_paper-evals-smoke/base/qwen36-27b/scan/` must
  be rebuilt with `--force` before any 27B corpus-retrieval number is quoted.
- STILL OPEN: the 512-token realact windows are forwarded with NO sink token, per Celeste's recipe,
  while everything else prepends one. That is her convention and is kept, but it means the realact
  targets and the scan cosines come from slightly different context conventions.


## The two 27B MAEMMs on the same 8 realact rows (2026-09-15)

Both on `2026-09-16_v1` rows 0-7 (realact), n = 64, scored on the clean 27B base. **No assert fired
in either run** (marker norm, the unpadded-batch shape, "not all rollouts identical", the two
item-8 bounds, and the CSR consistency check all passed).

| | rlI-150 (LoRA) | rl-8x2048 (full) |
|---|---|---|
| mean cos (bo1, 8 x 64) | **0.4992** | **0.5076** |
| mean best-of-64 | **0.5849** | **0.5929** |
| per-direction bo1 | 0.636 0.402 0.492 0.143 0.637 0.617 0.530 0.536 | 0.593 0.404 0.505 0.176 0.649 0.622 0.540 0.572 |
| per-direction bo64 | 0.717 0.622 0.551 0.259 0.676 0.643 0.599 0.612 | 0.752 0.607 0.554 0.310 0.684 0.653 0.577 0.606 |
| marker ‖h‖ served / clean base | 512.0 / 14.0625 (measured, adapter off) | 131.0 / 14.062 (base from config) |
| generated tok/s (32 rows/call) | 239.3 | 341.5 |
| mean rollout length / eos rate | 53.4 / 0.803 | 45.5 / 0.977 |
| gated SAE features at the argmax token | 80.7 per rollout | 80.8 per rollout |
| weight identity | full sha256 of the 1.7 GiB adapter | `index.json` + shard sizes (52 GiB not hashed) |

Eight directions is a sample, not a result: the two agree to within +0.008 on both statistics,
which on 8 paired directions says nothing about which is better. What it does establish is that the
full-parameter path works end to end — a served model with its own marker norm generating, and a
separately loaded clean base scoring — and that it is *faster* per token than the LoRA (no adapter
matmuls), while its rollouts are 8 tokens shorter and terminate far more often (eos 0.98 vs 0.80).

## Step 5 (2026-09-16): `repo_examples` -- the SAE repo's own max-activating windows

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-16 | unit smoke + mutation battery for `strip_repo_sink` | `uv run paper-evals/precompute/unit_smoke.py`; 3 deliberate defects injected one at a time | 1.4 s | $0 | **26/26** checks pass; 3/3 mutations caught (strip ids but not acts; never strip; drop the mixed-block assert) | — |
| 2026-09-16 | `check`, both bases, with the new `max_acts` blocks (CPU) | `--product check` | 20.1 s | ~$0 | both max-acts files resolve: 8B 600.0 MiB from the `datasets--adamkarvonen--sae_max_acts` cache, 27B 1.5 GiB from the SAE repo's own `maxacts/` | `common.snapshot` needed a `repo_type` argument: the 8B windows are a **dataset** repo, cached under `datasets--`, not `models--` |
| 2026-09-16 | `repo_examples` 8B, 512 features x 30 windows | `--product repo_examples --base qwen3-8b --set 2026-09-16_v1 --root …smoke` | 38.2 s | **$0.0419** | 15,360 windows scored, 11.6 MiB; **mean max_cos 0.2296** (`sae-repo-top32`), mean mean_cos 0.1975; 100.0% of features and 99.99% of windows fire at the learned gate 6.9360; repo-vs-us peak **r 0.9941** (median 0.9979, min 0.142), argmax agreement **99.74%**, median peak ratio **1.0000**, re-tokenization exact on 99.93% | none. The 8B reproduces adamkarvonen's own scan essentially exactly, which is what validates the activation formula and the read layer |
| 2026-09-16 | `repo_examples` 27B, 512 features x 32 windows | `--product repo_examples --base qwen36-27b --set 2026-09-16_v1 --root …smoke` | 132.8 s | **$0.1674** | 16,384 windows, 12.9 MiB; **mean max_cos 0.1708**, mean mean_cos 0.1127; 100.0% of features and 97.77% of windows fire at the learned gate 1.5846; re-tokenization exact on 99.95% | **repo-vs-us peak r only 0.5719** (median 0.5909, min −0.404), argmax agreement **80.8%**, and our peaks a systematic **~30% higher** (median ratio 1.301). See below |
| 2026-09-16 | sink diagnostic (added after the first 27B run; both bases re-run) | same commands with `--force` | 38.2 / 132.8 s | included above | the first 16 features' windows re-forwarded on their RAW ids, no sink: 8B **0.9941 -> 0.7156**, 27B **0.7415 -> 0.7539** | the sink is NOT the 27B's problem. It IS load-bearing on the 8B, which is the diagnostic's own control |
| 2026-09-16 | determinism | every run above executed twice (`--force`) | — | — | all summary fields bit-identical across runs on both bases (e.g. 27B `mean_per_feature_peak_r` 0.571935 both times) | — |
| 2026-09-16 | independent recomputation, local | numpy over the fetched `repo_examples.jsonl` (11-13 MiB per base) | ~10 s | $0 | `per_feature.max_cos` / `mean_cos` / `frac_fired` / `peak_pearson_r` / `argmax_agree` re-derived from the per-row file: max \|delta\| 0.0 and 6e-7 | — |

### The 27B max-acts disagreement (OPEN)

The 8B is the control and it lands on adamkarvonen's numbers: median (our peak / their peak)
**1.0000**, per-feature scale q10-q90 **0.998-1.002**, residual after removing that scale **0.2%**.
The 27B does not, and it fails in a specific way:

- a systematic magnitude offset: median ratio **1.301**, per-feature scale q10-q90 **1.03-1.41**;
- a per-window scatter no simple transform removes: a per-feature affine fit `ours = a*theirs + b`
  leaves a **13%** median relative residual (slope median 0.989, intercept median +2.96), and a
  pure scale fit leaves the same 13.6% — so it is neither a constant factor nor a constant shift;
- feature INDICES are nonetheless right: pooled r over all 16,384 windows is **0.863**, where a
  mismatched feature map would give ~0;
- position-independent: median ratio by their argmax position runs 1.28 at position 6 down to 1.19
  at position 31, with no early-position blow-up — the shape a context difference would have.

Ruled out: **sink handling** (the diagnostic above: removing it moves r by +0.012 on the 27B while
costing 0.28 on the 8B), **tokenization** (99.95% of windows re-tokenize to exactly the shipped
ids), **dead features** (0 of the 512 tested are dead in her scan; 108 of 16,384 individual windows
have peak 0), and **the read layer / activation formula in general** (the same code reproduces the
8B file to 0.2%).

Leading hypothesis, NOT tested: the `maxacts/` file was computed with a different checkpoint of the
same SAE run (a different training step keeps the feature ordering, which is what the pooled r
shows, while moving each encoder column a little — exactly this signature), or her scan scaled its
activations. The cheap decisive test is CPU-only and needs no forward: fetch another checkpoint of
that run (e.g. `ceselder/qwen36-27b-sae-l42`, **not** currently on the volume) and compare
`W_enc[:, f]` column by column against `l42-1b`'s. Until that is done the 27B `sae-repo-top32`
column is still usable — it is OUR scorer on HER text, which is all the column claims — but the
file's activation VALUES are not a reference for ours.

## `gcg` -- the discrete-search reachability ceiling (2026-09-16)

Rates as above. **Running GCG spend is carried in the last column and is cumulative over every
row of this section**, failed attempts included.

| date | item | command (abbreviated) | wall | cost | result | discrepancies | running GCG $ |
|---|---|---|---|---|---|---|---|
| 2026-09-16 | `score_tokens` -> `score_ids` refactor, unit smoke + mutations | `uv run paper-evals/precompute/unit_smoke.py`; 4 deliberate defects | 2 s | $0 | **27/27** checks pass (26 before + `check_score_ids_is_score_tokens`); 4/4 mutations caught, including a genuine text-path/id-path divergence | the equivalence check builds its id lists from the fixture tokenizer, NOT from `encode_for_score`, so it is a cross-check rather than a restatement of the wrapper | 0.00 |
| 2026-09-16 | offline `gcg.py` checks (no torch) | `uv run <scratch>/offline_gcg.py` | 1 s | $0 | arm configs, 9 malformed-config rejections, `parse_lam_grid`, `_TopSet` top-64 exact over 20,000 pushes, `distinct_n`, `init_corpus` geometry/tail-cut/short-window/pad guards, `roundtrip_repair` | 5/5 mutations caught (head-cut instead of argmax-tail, prune below 64, smallest corpus size, accept a short window, wrong `children` default) | 0.00 |
| 2026-09-16 | 8B `gcg-random32` plumbing shakeout | `--base qwen3-8b --arm gcg-random32 --rows 0 --iters 10` | 35.6 s | **$0.0391** | CHECK block agrees to 9.82e-05; cos 0.0035 -> 0.1922 in 10 iterations | **alphabet 90,909, not the fork's `ALPHABET_EXPECT` 94,325** (-3.6%, inside the +-10% band). 90,909 is the number the fork's own `init_random` docstring quotes, so the 94,325 in its plan was wrong; the constant is left at 94,325 and the measurement recorded rather than silently re-calibrated | 0.04 |
| 2026-09-16 | 8B `epo-corpus` shakeout (NLL + corpus-init paths) | `--arm epo-corpus --rows 0 --iters 5` | 26.0 s | **$0.0285** | corpus init cos 0.4119 -> 0.4810 in 5 iterations; NLL path checked against `model.logits` at 3.05e-04 nats | **retokenisation rejection 0.005 on the corpus init** against the fork's 0.547 design point: roundtrip-repaired natural text barely breaks under one substitution. `--filter-oversample` for the corpus arms cut from 2.2 to **1.5** (still 300x the measured margin) and the realised rate reported by every run | 0.07 |
| 2026-09-16 | `score` 8B run1 re-run, to prove the scorer refactor is neutral | `--product score --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --force` | 29.4 s | **$0.0323** | `per_target.jsonl` **BYTE-IDENTICAL** (sha256 `f064204e6ce3e994…`) to the pre-refactor product over 3,072 rollouts x 3 families | this is the real evidence that `score_tokens` = `encode_for_score` + `score_ids`; the unit check alone could not have shown it on the actual model | 0.10 |
| 2026-09-16 | **8B realact, 4 arms x 8 directions** | `--base qwen3-8b --family realact --arm <arm> --rows 0-7 --force` (four detached calls, parallel) | 662-2107 s | **$6.0693** | table below | no assert fired in any arm; 0 iterations hit the top-up cap, 0 init repairs, 0 short corpus windows | 6.17 |
| 2026-09-16 | 27B `scan` re-run on the re-centred targets | `--product scan --base qwen36-27b --set 2026-09-16_v1 --force` | 405.2 s | **$0.5110** | 2.4 MiB scan + 50.0 MiB SAE examples | the previous scan was written at 22:07:39Z against targets re-drawn at 23:07:39Z -- exactly one hour stale, and the corpus-init arms read it | 6.68 |
| 2026-09-16 | 27B GCG, three failed attempts | `--base qwen36-27b --arm gcg-corpus --rows 0 --iters 5` | ~60-150 s each | **~$0.4** (estimated from wall; a failed run writes no README) | see "The 27B needs a different Triton" below | `fla` REFUSES the GatedDeltaNet BACKWARD on Hopper; the forward, which every other product uses, is fine | 7.08 |
| 2026-09-16 | 27B GCG shakeout, Triton >= 3.7.1 | same, `--force` | 233.2 s | **$0.2941** | runs: cos 0.3071 -> 0.4421 in 5 iterations, CHECK 1.07e-03 | **27B alphabet MEASURED 126,220** (248,077 vocab -> 126,278 printable-ASCII -> 126,253 round-tripping -> 126,220 after 33 special/added). There was no prior figure for this base; it is now `ALPHABET_EXPECT["qwen36-27b"]` | 7.37 |
| 2026-09-16 | 8B sae `gcg-corpus` shakeout | `--family sae --arm gcg-corpus --rows 0 --iters 5` | 41.2 s | **$0.0452** | sae global row 1024, feature 143: cos 0.2902 -> 0.3004, peak pre-gate act 121.13, fired at the learned gate 6.9360 | the `sae` family's cosines are far below realact's -- an encoder column is a harder target than a residual direction that a real document actually produced | 7.42 |
| 2026-09-16 | **the other 12 arms x 8 directions** (8B sae x4, 27B realact x4, 27B sae x4) | `--base <base> --family <family> --arm <arm> --rows 0-7 --force`, twelve detached calls | 662-7223 s each | **$52.04** | full table below; all 16 arm dirs verified (finals = pop x 8 rows, trajectory, top64, summary, README with cost) | 27B `sae/epo-random32` is the only arm that did not move (cos 0.019, feature dead on 19/24 finals); one 27B sae launch was killed by `SMOKES.md was modified during build` while Modal hashed `add_local_dir` and was relaunched | 59.46 |
| 2026-09-16 | failed launch: `--root <scratch>` for a post-fix shakeout | `--family sae --arm gcg-random32 --rows 0-1 --iters 3 --root /vol/runs/2026-09-16_gcg-checks` | ~40 s | **~$0.03** (estimated from wall; no README written) | `FileNotFoundError` on `<root>/base/qwen3-8b/heldout/2026-09-16_v1/ids.jsonl` | `--root` relocates INPUTS as well as outputs, so a scratch root has no held-out set -- the `sae_peak_pos` fix is covered offline instead | 59.49 |

### 8B realact, four arms x 8 directions (2026-09-16)

| arm | dirs | mean final cos (best-over-members / per-member) | mean init cos | mean NLL | wall/dir | $/dir | arm cost | cand/s | retok reject | distinct top strings/dir |
|---|---|---|---|---|---|---|---|---|---|---|
| `gcg-corpus` | 8 | **0.6590** / 0.6590 | 0.4919 | 8.064 | 82 s | $0.0905 | $0.7242 | 962 | 0.067 | 64 |
| `gcg-random32` | 8 | **0.4620** / 0.4620 | -0.0163 | 13.593 | 89 s | $0.0972 | $0.7778 | 896 | 0.051 | 64 |
| `epo-corpus` | 8 | **0.5760** / 0.5524 | 0.4919 | 3.022 | 263 s | $0.2887 | $2.3092 | 294 | 0.032 | 192 (3 members x 64) |
| `epo-random32` | 8 | **0.4168** / 0.2865 | -0.0102 | 4.793 | 256 s | $0.2814 | $2.2510 | 302 | 0.023 | 192 |

Per-direction best cos, rows 0-7 in order:

- `gcg-corpus` 0.615 0.626 **0.775** 0.617 0.662 0.609 0.695 0.673 (init 0.412 0.508 0.550 0.445 0.401 0.538 0.515 0.568)
- `gcg-random32` 0.401 0.428 0.736 0.580 0.315 0.248 0.494 0.494 (init ~0 on all eight)
- `epo-corpus` 0.561 0.586 0.590 0.562 0.546 0.567 0.586 0.610
- `epo-random32` 0.525 0.495 0.641 0.367 0.147 0.388 0.428 0.344

**The init dominates the arm.** `gcg-corpus` beats `gcg-random32` by **+0.197** mean cos on the same
eight directions at the same 76,800 candidate forwards and the same wall -- the search does not
recover from a random start inside 150 iterations, and the spread across directions is far wider
from random (0.248-0.736) than from the corpus window (0.609-0.775). The corpus init is not merely
a head start: the two arms end in different places.

**The lambda term buys fluency and costs cosine, monotonically.** `epo-corpus` by lambda:
0.1 -> cos 0.5718 / NLL 3.123, 0.19 -> 0.5500 / 2.895, 0.37 -> 0.5354 / 2.856. Against
`gcg-corpus`'s NLL of 8.06 that is a 5-nat fluency gain for 0.08-0.12 of cosine, which is the
Pareto front the population is there to trace. `gcg-random32`'s NLL of **13.59** is the reachability
ceiling's real character: the string it finds is not text.

**EPO is not compute-matched per iteration, only in total** (150 x 512 = 76,800 against
300 x 255 = 76,500) and it comes out behind GCG on raw cosine in both inits. At lambda > 0 that is
expected -- it is optimising a different objective -- but `epo`'s best member (lambda 0.1) is still
below `gcg` at lambda 0, so the 3x-smaller per-iteration candidate pool costs something too.

### The 27B needs a different Triton (2026-09-16)

`gcg` is the ONLY product here that runs a BACKWARD pass, and on the 27B that backward crosses 48
GatedDeltaNet layers. `flash-linear-attention` 0.5.2 refuses it outright:

> `RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results for gated
> chunk_bwd_dqkwg (see #640). Please upgrade Triton to >= 3.7.1 or install tilelang`

That is fla protecting against a WRONG-ANSWER bug, not a missing feature, and the forward every
other product uses is unaffected -- which is why `stats`, `scan`, `rollouts_hf` and `score` have
always run on `image27`. Three attempts, in fla's own suggested order:

1. `pip install tilelang` -- **still refused**. fla's `has_usable_nvcc()` (`fla/utils/_compat.py:38`)
   requires a real `nvcc` BINARY; `debian_slim` + pip torch wheels have none, so the tilelang
   backend reports itself UNAVAILABLE and `dispatch` falls straight back to the Triton path it has
   just refused. The log line that says so is at `logger.info` level and is easy to miss.
2. `tilelang` + `nvidia-cuda-nvcc` (unsuffixed >= 13.0 -- the `-cu12` variant ships ptxas only,
   fla's own note) -- the backend IS selected and JIT-compiles, and the compile of its generated
   sm90 wgmma kernel **fails inside TVM's FFI**, which then cannot serialise its own error
   (`SerializationError: Type ffi.Error does not support ToJSONGraph`), so the diagnostic is the
   generated CUDA source rather than the compiler's message.
3. **`triton>=3.7.1`, no tilelang, `FLA_TILELANG=0`** -- works. This is what `gcg/modal_app.py`
   pins, as a layer ON TOP of `precompute/modal_app.py`'s `image27`, so that image, its pins and
   its layer cache are untouched for every other product.

MEASURED cost of the backward on the 27B: the FIRST gradient pass of a container takes **~75-85 s**
(Triton autotune), every one after it ~0.2 s. That one-time cost is why a 1-direction 5-iteration
shakeout reports 15 cand/s while the steady state is ~180-430.

### 8B sae, the two `gcg` arms x 8 directions (2026-09-16)

`--family sae --rows 0-7` = global rows 1024-1031, features 143 / 314 / 588 / 2004 / 3056 / 3091 /
3106 / 4533 of `qwen3-8b/adamkarvonen-t2`.

| arm | dirs | mean final cos | mean init cos | mean NLL | $/dir | arm cost | retok reject | mean peak pre-gate act | frac fired |
|---|---|---|---|---|---|---|---|---|---|
| `gcg-corpus` | 8 | **0.3289** | 0.2390 | 6.775 | $0.0980 | $0.7839 | 0.072 | 149.62 | **1.000** |
| `gcg-random32` | 8 | **0.2688** | 0.0132 | 13.536 | $0.1000 | $0.8003 | 0.044 | 120.36 | **1.000** |

Three things the sae family shows that realact does not:

- **The cosine ceiling is much lower.** 0.329 against realact's 0.659 on the same base, same budget,
  same init rule. An encoder column is a harder target than a residual direction some real document
  actually produced -- which is the point of having both families in the paper.
- **The feature fires anyway, from both inits.** Mean peak pre-gate activation 149.6 (corpus) and
  120.4 (random) against the checkpoint's learned gate of **6.936**, and `frac_fired` is **1.000**
  in both arms. A cosine of 0.27 to the encoder column is already 17x the fire threshold, so
  "does the string activate the feature" and "how well does the string align with the feature" are
  very different questions, and only the second one is hard.
- **The activation's peak and the cosine's argmax are the SAME token on 8/8 directions in both
  arms** (`frac_peak_at_cos_argmax` 1.000). That is a consistency check on the recording rather than
  a finding -- the pre-gate activation is monotone in the projection onto the encoder column -- but
  it is the check that would have caught the two being read off different forwards.

The activation is RECORDED, never optimised: the objective is the cosine in every arm.

### 27B realact, interim (2026-09-16, arms still running)

SUPERSEDED by "All 16 arms, final" below; kept for the rate estimates it was used to plan with.

Per-direction, from the running logs (`gcg` arms, first three of eight directions):

| dir | `gcg-corpus` final (init) | `gcg-random32` final (init) |
|---|---|---|
| 0 | 0.498 (0.307) | 0.071 (-0.015) |
| 1 | 0.455 (0.266) | 0.354 (0.077) |
| 2 | 0.347 (0.277) | 0.274 (-0.046) |

and `epo` direction 0: `epo-corpus` 0.471 (init 0.307), `epo-random32` 0.056 (init -0.014).

MEASURED 27B rates, steady state after the one-time Triton autotune: `gcg` **~318 cand/s**
(242-256 gpu-s per direction), `epo` **~76-82 cand/s** (934-1004 gpu-s per direction) -- so the 27B
costs about **$0.33/direction** on a `gcg` arm and **$1.2/direction** on an `epo` arm, i.e. 3.4x and
4.2x the 8B. The first gradient pass of each container adds ~75-85 s once.

### Running spend (2026-09-16)

Projection made while arms were in flight; the MEASURED total is in "Final GCG spend" below.

| item | $ |
|---|---|
| local checks, shakeouts (8B gcg, 8B epo-corpus, 8B sae, 27B) | 0.41 |
| `score` re-run proving the scorer refactor byte-identical | 0.03 |
| 27B `scan` re-run (the stale one) | 0.51 |
| three failed 27B attempts (Triton/tilelang) | ~0.40 |
| 8B realact, 4 arms x 8 dirs | 6.07 |
| 8B sae, `gcg-corpus` + `gcg-random32` x 8 dirs | 1.58 |
| **spent and measured** | **9.00** |
| in flight: 8B sae `epo` x2 | ~4.6 |
| in flight: 27B realact x4 (2 x gcg ~2.7, 2 x epo ~9.7) | ~24.8 |
| in flight: 27B sae `gcg` x2 + `epo-corpus` | ~15.1 |
| **projected total** | **~53.5** |

27B sae `epo-random32` WAS launched after the cap was raised to $75 (2026-09-16), so all
2 bases x 2 families x 4 arms = 16 arms are running or landed. Projected total with it: **~$63**.

### All 16 arms, final: 2 bases x 2 families x 4 arms x 8 directions (2026-09-16)

Every arm above and below landed. `--rows 0-7` per family: `realact` = global rows 0-7,
`sae` = global rows 1024-1031 (8B features 143/314/588/2004/3056/3091/3106/4533 of
`adamkarvonen-t2`, gate 6.9360; 27B features 433/1102/1453/1742/3822/4249/5557/6199 of
`ceselder-l42`, gate 1.5846). `mean final cos` is best-over-members / per-member (identical for the
`gcg` arms, which are pop 1). Each arm is 76,800 (`gcg`) or 76,500 (`epo`) candidate forwards per
direction. **Verified on all 16: `finals.jsonl` = pop x 8 rows, `trajectory.jsonl`, `top64.jsonl`,
`summary.json`, `README.md` with a cost. No assert fired anywhere; 0 top-up-capped iterations, 0
init repairs, 0 short corpus windows skipped in any arm.**

| base | family | arm | mean final cos (best/member) | mean init cos | mean NLL | wall/dir | $/dir | arm $ | cand/s | retok reject | mean peak act | frac fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact | `gcg-corpus` | **0.6590** / 0.6590 | 0.4919 | 8.064 | 82 s | $0.0905 | $0.7242 | 962 | 0.067 | -- | -- |
| qwen3-8b | realact | `gcg-random32` | 0.4620 / 0.4620 | -0.0163 | 13.593 | 89 s | $0.0972 | $0.7778 | 896 | 0.051 | -- | -- |
| qwen3-8b | realact | `epo-corpus` | 0.5760 / 0.5524 | 0.4919 | 2.958 | 263 s | $0.2887 | $2.3092 | 294 | 0.032 | -- | -- |
| qwen3-8b | realact | `epo-random32` | 0.4168 / 0.2865 | -0.0102 | 4.699 | 256 s | $0.2814 | $2.2510 | 302 | 0.023 | -- | -- |
| qwen3-8b | sae | `gcg-corpus` | **0.3289** / 0.3289 | 0.2390 | 6.775 | 89 s | $0.0980 | $0.7839 | 896 | 0.072 | 149.62 | 1.000 |
| qwen3-8b | sae | `gcg-random32` | 0.2688 / 0.2688 | 0.0132 | 13.536 | 91 s | $0.1000 | $0.8003 | 876 | 0.044 | 120.36 | 1.000 |
| qwen3-8b | sae | `epo-corpus` | 0.2935 / 0.2667 | 0.2390 | 2.991 | 253 s | $0.2775 | $2.2203 | 306 | 0.024 | 127.36 | 1.000 |
| qwen3-8b | sae | `epo-random32` | 0.2052 / 0.1220 | 0.0132 | 3.942 | 255 s | $0.2802 | $2.2416 | 304 | 0.035 | 53.04 | 0.458 |
| qwen36-27b | realact | `gcg-corpus` | **0.4422** / 0.4422 | 0.3145 | 8.411 | 265 s | $0.3342 | $2.6733 | 297 | 0.075 | -- | -- |
| qwen36-27b | realact | `gcg-random32` | 0.2586 / 0.2586 | -0.0196 | 13.185 | 279 s | $0.3519 | $2.8149 | 282 | 0.052 | -- | -- |
| qwen36-27b | realact | `epo-corpus` | 0.3915 / 0.3713 | 0.3145 | 2.696 | 888 s | $1.1198 | $8.9581 | 87 | 0.027 | -- | -- |
| qwen36-27b | realact | `epo-random32` | 0.2337 / 0.1273 | -0.0202 | 5.078 | 826 s | $1.0419 | $8.3355 | 93 | 0.049 | -- | -- |
| qwen36-27b | sae | `gcg-corpus` | **0.2149** / 0.2149 | 0.0729 | 8.291 | 272 s | $0.3424 | $2.7391 | 292 | 0.064 | 26.49 | 1.000 |
| qwen36-27b | sae | `gcg-random32` | 0.0948 / 0.0948 | 0.0050 | 13.212 | 293 s | $0.3701 | $2.9609 | 270 | 0.053 | 10.20 | 0.875 |
| qwen36-27b | sae | `epo-corpus` | 0.1502 / 0.1298 | 0.0729 | 2.816 | 902 s | $1.1381 | $9.1044 | 86 | 0.024 | 16.10 | 0.958 |
| qwen36-27b | sae | `epo-random32` | 0.0187 / 0.0114 | 0.0057 | 4.230 | 834 s | $1.0520 | $8.4161 | 92 | 0.055 | 0.21 | **0.042** |

Per-direction best cosine, rows 0-7 in order (init in brackets where it is not ~0):

- 27B realact `gcg-corpus` 0.498 0.455 0.347 0.317 0.485 0.416 0.481 0.540 (init 0.307 0.266 0.277 0.267 0.343 0.345 0.352 0.359)
- 27B realact `gcg-random32` 0.071 0.354 0.274 **-0.026** 0.437 0.404 0.366 0.189
- 27B realact `epo-corpus` 0.471 0.415 0.339 0.294 0.437 0.370 0.358 0.449 (same inits)
- 27B realact `epo-random32` 0.056 0.364 0.108 **-0.026** 0.448 0.267 0.347 0.305
- 27B sae `gcg-corpus` 0.139 0.233 0.247 0.221 0.299 0.105 0.349 0.125 (init 0.071 0.055 0.089 0.062 0.067 0.038 0.141 0.063)
- 27B sae `epo-random32` 0.035 0.006 0.031 0.016 0.023 0.012 0.018 0.010 -- the arm that essentially did not move

**Row 3 of the 27B realact set ends NEGATIVE from a random init in both `gcg` and `epo`** (-0.026,
init -0.178). It is the only direction on either base that the search fails to bring above zero,
and it is also the direction with the lowest MAEMM best-of-64 (0.284 / 0.274), so the two agree
about which target is hard rather than disagreeing about the search.

**The init dominates the arm, on both bases and both families, without exception.**
Corpus minus random, mean final cos: 8B realact +0.197 (`gcg`) / +0.159 (`epo`), 8B sae +0.060 /
+0.088, 27B realact +0.184 / +0.158, 27B sae +0.120 / +0.132. Same budget, same wall, same rule --
the search does not recover from a random start inside 150 (or 300) iterations.

**The lambda term buys fluency and costs cosine, monotonically, in all four `epo` families.**
27B realact `epo-corpus` by lambda: 0.1 -> cos 0.389 / NLL 2.81, 0.19 -> 0.370 / 2.65,
0.37 -> 0.355 / 2.63. Against the matching `gcg-corpus` (cos 0.442, NLL **8.41**) that is 5.6 nats
of fluency for 0.05 of cosine. The `gcg-random32` arms sit at NLL **13.2-13.6** on both bases: the
string the unconstrained ceiling finds is not text at all.

**EPO trails GCG on raw cosine in every one of the eight (base, family, init) cells**, by
0.02-0.08 -- expected at lambda > 0, but its best member is at lambda 0.1, not 0, so the
3x-smaller per-iteration candidate pool (255 against 512) costs something on its own.

#### The `sae` family: aligning is hard, firing is not (until the 27B random arm)

| base | arm | mean peak pre-gate act | gate | frac fired | peak token == cos argmax |
|---|---|---|---|---|---|
| qwen3-8b | `gcg-corpus` | 149.62 | 6.9360 | 1.000 | 8/8 |
| qwen3-8b | `gcg-random32` | 120.36 | 6.9360 | 1.000 | 8/8 |
| qwen3-8b | `epo-corpus` | 127.36 | 6.9360 | 1.000 | 24/24 |
| qwen3-8b | `epo-random32` | 53.04 | 6.9360 | 0.458 | 13/13 non-zero |
| qwen36-27b | `gcg-corpus` | 26.49 | 1.5846 | 1.000 | 8/8 |
| qwen36-27b | `gcg-random32` | 10.20 | 1.5846 | 0.875 | 8/8 |
| qwen36-27b | `epo-corpus` | 16.10 | 1.5846 | 0.958 | 23/23 non-zero |
| qwen36-27b | `epo-random32` | **0.21** | 1.5846 | **0.042** | 5/5 non-zero |

On the 8B a cosine of 0.27 to the encoder column is already 17x the fire threshold, so "does the
string activate the feature" and "how well does the string align with it" are different questions
and only the second is hard. **The 27B `sae/epo-random32` arm is the counterexample**: it ends at
cos 0.019 and the feature is dead on 19 of 24 finals. That is the one cell where the search failed
to produce anything, and it failed from the random init on the harder base with the lambda penalty
pulling against it -- all three handicaps at once.

**The activation's peak token is the cosine's argmax token on EVERY final where the feature is
non-zero, in all eight arms.** That is the consistency check that the cosine and the activation are
read off one forward, not a finding (the pre-gate activation is monotone in the projection onto the
encoder column).

> **Defect found and fixed after these runs.** `gcg.sae_peak_pos` used `nanargmax` unconditionally,
> so a DEAD feature (relu zeroes the whole row) reported peak position 0 and
> `frac_peak_at_cos_argmax` counted that arbitrary 0 as a genuine disagreement -- 0.542 on 8B
> `sae/epo-random32` and 0.292 on 27B `sae/epo-random32` where the true figure is 1.000 over the
> rows that fire. The rule is now a module-level `sae_peak_pos(acts_row, peak)` returning **-1** when
> the feature never fires, and `summary.json` averages only over those rows and records
> `n_rows_peak_defined`. The 16 `summary.json` files ON THE VOLUME were written by the pre-fix code
> and still carry the uncorrected number; the table above is recomputed from `finals.jsonl`, which
> carries every per-row field, rather than by re-running $58 of search. Covered by an offline check
> (dead -> -1, sink offset, agreement over defined rows) that was made to fail before it passed.

#### GCG against the two 27B MAEMMs, same 8 realact rows (2026-09-16)

`gcg` best-over-members per direction, against each MAEMM's unbiased best-of-64 on the same rows of
the same held-out set:

| comparator | mean cos over rows 0-7 | vs rlI-150 bo64 (0.5496) | vs rl-8x2048-full bo64 (0.5430) |
|---|---|---|---|
| `gcg-corpus` | **0.4422** | -0.1074, wins 1/8 | -0.1008, wins 1/8 |
| `epo-corpus` | 0.3915 | -0.1581, wins 1/8 | -0.1515, wins 1/8 |
| `gcg-random32` | 0.2586 | -0.2910, wins 0/8 | -0.2845, wins 0/8 |
| `epo-random32` | 0.2337 | -0.3159, wins 0/8 | -0.3094, wins 0/8 |
| MAEMM best-of-1 | 0.4360 / 0.4694 | -- | -- |

**At this budget the discrete search is NOT an upper bound on the MAEMM.** The best arm lands
roughly at the MAEMMs' best-of-**1** (0.436 / 0.469) and about 0.10 BELOW their best-of-64, losing
on 7 of 8 directions (the exception is row 3, the hard one, on both comparators). Three reasons the
comparison is not matched, all of which favour the MAEMM and none of which is a defect in the
search:

- **Length.** `gcg` is fixed at T=32 ids; the rollouts average **55.8** tokens (rlI-150) and
  **47.5** (rl-8x2048-full), with `max_new` 64. More positions is more chances for the max over
  positions.
- **Samples.** best-of-64 is a max over 64 independent strings; `gcg` is one run (`epo`: 3 members
  at 3 different lambdas, so not 3 independent tries at the same objective either).
- **Compute.** 76,800 candidate forwards of 33 tokens is not the same currency as 64 autoregressive
  rollouts, and neither is normalised here.

So the honest reading is a LOWER bound on reachability at T=32: a 32-token string exists, findable
by 150 GCG steps from a corpus window, reaching cos 0.44 mean on these directions. Whether the
ceiling exceeds the MAEMM needs a length-matched arm (`--seq-len 48` or 64) or a best-of-n arm, and
the matching per-direction MAEMM cost is what decides whether that is worth buying.

#### Sharpened projection to 64 directions per arm

From the MEASURED split of each arm's wall into a fixed per-call part (model + SAE load, volume
commit) and a per-direction search part, both from `summary.json`'s own `gpu_seconds`:

| base | arm | search s/dir | fixed s/call | $ for 64 dirs | $/dir at 64 |
|---|---|---|---|---|---|
| qwen3-8b | `gcg-*` | 80-88 | 22-29 | $5.63-6.19 | $0.088-0.097 |
| qwen3-8b | `epo-*` | 250-260 | 24-27 | $17.57-18.29 | $0.275-0.286 |
| qwen36-27b | `gcg-*` | 259-285 | 51-72 | $20.93-23.06 | $0.327-0.360 |
| qwen36-27b | `epo-*` | 819-895 | 52-62 | $66.19-72.28 | $1.034-1.129 |

**Per base, all 8 arms (2 families x 4 arms) at 64 directions: qwen3-8b $95.32, qwen36-27b
$363.88, both bases $459.20.** The 8-direction rate over-states the 64-direction rate by only
1-3% (the fixed per-call cost amortises), so these are the numbers to plan against. The 27B `epo`
arms are 80% of the 27B bill; dropping to `gcg` only on the 27B costs $44 for both families at 64
directions, and the Pareto front then has to come from somewhere else.

### Final GCG spend (2026-09-16)

| item | $ |
|---|---|
| local checks and shakeouts (8B gcg, 8B epo-corpus, 8B sae, 27B) | 0.41 |
| `score` re-run proving the scorer refactor byte-identical | 0.03 |
| 27B `scan` re-run (the stale one) | 0.51 |
| three failed 27B attempts (Triton/tilelang) | ~0.40 |
| one failed launch: `--root <scratch>` relocates INPUTS too, so `heldout/ids.jsonl` was absent (no README written, estimated from wall) | ~0.03 |
| **all 16 arms x 8 directions** (sum of the arm READMEs' own costs) | **58.11** |
| **TOTAL for the `gcg` step** | **~59.49** |

Against the raised cap of $75 for GCG/EPO, and the projected ~$63 in the table above. The 16 arms
break down as 8B $12.11 (8 arms) and 27B $46.00 (8 arms), i.e. the 27B is **3.8x** the 8B for the
identical work.

<!--GCG-SMOKES-->

## Measured throughput and the 16M extrapolation (2026-09-15)

From each product's own README (checklist item 84). "corpus tok/s" is the slice's own tokens per
second — the number to extrapolate with; "fwd tok/s" is the tokens actually pushed through the
model, which is ~4.1x higher because the 64/16 geometry puts every token in ~4 windows (plus the
sink).

| product | 8B (H100) | 27B (H200) |
|---|---|---|
| `stats` pass A | 25,305 fwd / **6,554 corpus** tok/s | 8,448 fwd / **2,185 corpus** tok/s |
| `scan` pass B | 37,703 fwd / **9,764 corpus** tok/s | 11,125 fwd / **2,877 corpus** tok/s |

Extrapolated to the full 16M corpus (scan time + the base load + the `acts_1m` store, which at full
scale is the whole 1M-token subset: ~8.2 GB on the 8B, ~10.2 GB on the 27B, written at the
measured 20-50 MB/s):

| product | 8B | 27B |
|---|---|---|
| `stats` | ~2,950 s ≈ **$3.2** | ~7,900 s ≈ **$10.0** |
| `scan` | ~1,730 s ≈ **$1.9** | ~5,650 s ≈ **$7.1** |
| `targets` | ~170 s ≈ $0.2 | ~200 s ≈ $0.3 |
| `corpus` | ~5 min CPU ≈ $0 | ~5 min CPU ≈ $0 |
| **per base** | **~$5.3** | **~$17.4** |

So one full pass over 16M on both bases is ~$23 of GPU. `scan` is re-run per held-out set; `stats`
is target-independent and is not.

## What the sanity checks verified (2026-09-15)

Run locally against fetched outputs, against independent recomputations (the window geometry
re-derived from the stated rule, the spans re-decoded with the real tokenizer downloaded from the
hub, the fire counts re-read from `fire_counts.i64`):

- token counts per size tag match the tags, tags are non-decreasing in stored order (so every
  nested subset really is a prefix), and the recomputed window count (118,928) matches the scan's;
- all 512 realact `span_text` fields equal `decode(tokens[p−L+1 : p+1])` on BOTH bases (8B vocab
  151,669, 27B 248,077), `p ∈ [16, 511]`, `L ∈ [16, 64]`, one target per document, every source
  document ≥ 512 tokens, `(part, part_row)` matching `corpus/docs.jsonl`;
- every held-out vector is unit to 6.3e-5 after the f16 round trip;
- the sae family has exactly 128 per stratum, strata are ordered by density with no overlap, every
  drawn feature has ≥ 20 gated fires (8B min 36, 27B min 20) and max_act > 0, and every 8B feature
  is inside run1 ∩ run2's held-out list;
- `fire_counts` is cumulative over sizes and gated ≤ ungated everywhere;
- realact top-1 cos is high (8B median 0.476, 27B 0.405) while the random control's is at the floor
  (8B median 0.066, 27B 0.054); top-1 is non-decreasing with corpus size; size-1 top entries come
  only from tag-1 documents; all 196,608 (8B) / 98,304 (27B) top-k entries name a real window with
  an in-range argmax and cos;
- **no realact top-k entry overlaps its own span** — 5,517 same-document entries survive on the 8B,
  none of them overlapping `[p−L+1, p]`, so the mask is doing exactly what it should and no more;
- quantiles are monotone in q and never exceed the target's top-1 cos;
- SAE examples: `acts` has exactly the window's length, `argmax` is its argmax, the top example's
  activation equals the stored corpus `max_act`, quartile bins are consistent with
  `ceil(act/max_act·4)−1`, every feature with gated fires has examples, and `_random256.jsonl` has
  256 distinct windows each carrying 512 per-feature maxima;
- `acts_1m` row count equals the covered tag-1 token count exactly (8B 81,938; 27B 99,274) and its
  documents are the corpus prefix;
- 8B leakage: **0 hits** at cos > 0.999 against 230,035 + 222,535 archived training-bank rows.

## Ambiguity resolved: what "training features" means in the archived split.json

`data/run{1,2}/bank/split.json` (both files byte-identical) has **no** train-feature list. Its only
populated feature key is `families.sae_heldout_features`: 6,553 ids, the held-out 10% of the
65,536-wide 8B SAE. `pool_train/build_stats.json` confirms the complement is what was minted
(`alive_on_scan` 65,536, `families.sae` 58,983 = 65,536 − 6,553). So `targets.py` treats the
**complement** of that list as the training set and keeps only run1 ∩ run2's held-out ids eligible.
That is conservative in the safe direction (run2's sentence filter dropped 8,281 of its candidates,
so its true training set is smaller), but it means the 8B sae family is drawn from **6,515**
features, not 65,536 — a real narrowing that the held-out README records and any write-up must say.

## Step 6 (2026-09-16): `mu_diag` -- what makes our `stats/mu` and Celeste's `whiten_mu` disagree

`mu_check` reports THAT the two means disagree (cos 0.9893 on the 8B, 0.9775 on the 27B). This
product decomposes the gap, one forward pass per geometry, on the smoke corpus:

- 8B (H100) `--product mu_diag --base qwen3-8b --root …smoke`: **96.6 s, $0.1060**. Our geometry
  walked 589 documents / 500,114 corpus tokens (the `--tokens` cap, default 500k) = 1,900,226
  positions in 51 s; her geometry used all **1,075** documents ≥ 512 tokens = 550,400 positions in
  21 s.
- 27B (H200) same with `--tokens 400000`: **249.8 s, $0.3150** (30 s of it the 51.8 GiB load).
  525 documents / 400,125 tokens = 1,511,325 positions in 172 s; her geometry all **550**
  documents ≥ 512 tokens = 281,600 positions in 36 s — above the product's 200-document floor, so
  row (a) is read as a geometry effect on both bases.

`ours` = the stored `stats/mu.f32` (64/16 windows, `[sink] + window`, sink dropped, FULL smoke
corpus); `hers` = the archived `whiten_mu.npy`; `mu_512` = HER geometry (a document's first 512
tokens, `add_special_tokens=False`, no sink, every position, documents < 512 tokens dropped) on
OUR corpus; `_pos16` = positions ≥ 16 only (counted after the sink for ours); `_light` = positions
with residual norm > 10x that geometry's own median removed; halves = documents split by index
parity. `ratio` is ‖left‖ / ‖right‖.

| row | what it isolates | 8B cos | 8B ratio | 27B cos | 27B ratio |
|---|---|---|---|---|---|
| anchor: `ours` vs `hers` | the disagreement itself (reproduces `mu_check`) | **0.989258** | 0.950924 | **0.977523** | 1.009439 |
| anchor: `mu_512` vs `hers` | what is left AFTER geometry: her corpus + the rest | **0.999710** | 0.991547 | **0.999897** | 0.999114 |
| control: `mu_64` vs `ours` | the capped walk against the stored full-corpus mean | 0.999712 | 0.992306 | 0.999760 | 1.008046 |
| (a) `mu_512` vs `ours` | geometry alone, corpus held fixed | **0.989999** | 1.042720 | **0.977429** | 0.989771 |
| (b) `mu_64_pos16` vs `ours` | our own positions < 16 | 0.998620 | 0.975739 | 0.997991 | 0.998524 |
| (b) `mu_64_pos16` vs `hers` | — | 0.991454 | 0.927854 | 0.985059 | 1.007949 |
| (b) `mu_64_pos16` vs `mu_512` | — | 0.992161 | 0.935764 | 0.984995 | 1.008844 |
| (b) `mu_512_pos16` vs `ours` | — | 0.989464 | 0.992706 | 0.975571 | 0.989843 |
| (b) `mu_512_pos16` vs `hers` | — | 0.997870 | 0.943988 | 0.999814 | 0.999186 |
| (b) `mu_512_pos16` vs `mu_512` | her own positions < 16 | 0.998212 | 0.952035 | 0.999925 | 1.000073 |
| (b) `mu_512_pos16` vs `mu_64_pos16` | both geometries, positions ≥ 16 (computed off `mus.f32`) | 0.992430 | 1.017389 | 0.983656 | 0.991306 |
| (c) `mu_64_light` vs `mu_64` | massive tokens in OUR geometry (8B **0.028%** of positions, 27B **none**) | 0.999961 | 0.992310 | 1.000000 | 1.000000 |
| (c) `mu_512_light` vs `mu_512` | massive tokens in HER geometry (8B **0.249%**, 27B **none**) | 0.997087 | 0.935916 | 1.000000 | 1.000000 |
| (c) sink / position-0 norm | ours: the DROPPED sink; hers: position 0, KEPT in her mean | ours 10,893 / hers 11,581 | median 411 / 416 | ours 194 / hers 209 | median 94 / 92 |
| (d) `mu_64` halves | sampling noise, our geometry | 0.998687 | 1.002867 | 0.999393 | 1.007637 |
| (d) `mu_512` halves | sampling noise, her geometry | 0.999673 | 1.008823 | 0.999679 | 1.007650 |

**Geometry is essentially the whole of it, and the corpus is not the residual it was expected to
be.** Her 512-token / no-sink / all-position window mean computed on OUR corpus lands on her
archived vector at cos **0.9997** (8B) and **0.9999** (27B) — at or below each geometry's own
split-half noise — so the corpus difference (ours is Ultra-FineWeb parts 0009-0010, hers the head
of part 0001 on the 8B and her own corpus on the 27B), which this product cannot isolate by
construction, is bounded by that and is negligible; row (a) then reproduces the entire `mu_check`
gap from geometry alone (0.9900 against the anchor's 0.9893; 0.9774 against 0.9775). Of the two
named sub-effects, position mix accounts for ~27% of the gap on both bases (restricting BOTH
geometries to positions ≥ 16 moves `mu_64` vs `mu_512` from 0.9896 to 0.9924 on the 8B and from
0.9776 to 0.9837 on the 27B) and massive-activation tokens account for none of it — removing them
from her mean moves it slightly AWAY from ours (0.98947 against 0.99000 on the 8B) and the 27B has
no position above 10x its median in either geometry. The remainder is the window LENGTH itself:
a position in a 64-token window sees at most 63 tokens of context and, at stride 16, is averaged in
up to 4 times at 4 different context depths, where hers averages depths up to 511 — that is the
reading these rows support, not a separately measured effect. The sink explains the NORMS, not the
directions: we drop a position whose norm is 10,893 against a median of 411 on the 8B, while her
position 0 (an ordinary token, norm 11,581) stays in her mean, which is most of why ‖ours‖/‖hers‖
is 0.951 there; on the 27B both reference positions are only ~2x the median and the norms agree to
1%.

## Full run 2026-09-16

The production pass on the REAL root `/vol`: 16M corpus on both bases, held-out set
`2026-09-16_v1` drawn fresh on it, and 512 x 64 rollouts of all three families through every
computable MAEMM. The 8B chain (H100) and the 27B chain (H200) ran in parallel, each chain
sequential. Everything below is read off each product's own README on the volume
(checklist item 84: `modal app logs` replays stale output), never off a local log tail.

Two code changes were needed before the first long call and both are minimal:

- `precompute/modal_app.py`: the two GPU functions' `timeout` 6 h -> **10 h**. The smokes were
  minutes; a full 27B rollouts call is 3-7 h depending on engine and `stats` / `scan` / `score`
  on 16M are 1.5-2.5 h each, and a timeout kill throws away the whole call's GPU spend.
- `precompute/rollouts_vllm.py`: `THROUGHPUT_LEVELS` `(32, 64, 128)` -> **`(32, 64, 128, 256, 512)`**.
  Strictly additive -- the old levels still run in the same engine, so the step-4 numbers are
  reproduced rather than replaced. The reason is the engine choice below: the full run submits
  1,536 requests at once and sits at 256-512 rows in flight, not at 128, which is where the step-4
  sweep stopped.

### Spend (this agent, cap $110)

| # | item | wall | cost | running total |
|---|---|---|---|---|
| 0 | `check`, both bases (CPU) | 19.0 s | ~$0 | $0.00 |
| 1 | `corpus` 8B @16M (CPU) | 65.0 s | ~$0 | $0.00 |
| 2 | `corpus` 27B @16M (CPU) | 89.0 s | ~$0 | $0.00 |
| 3 | vLLM throughput probe, 8B `run1-rl`, 32-512 rows (H100) | 142.6 s | $0.1565 | $0.16 |
| 4 | vLLM throughput probe, 27B `rlI-150`, 32-512 rows (H200) | 480.0 s | $0.6054 | $0.76 |
| 5 | `stats` 8B @16M (H100) | 2477.1 s | $2.7179 | $3.48 |
| 6 | vLLM real-shape calibration, 27B `rlI-150`, 128 x 64 (H200) | 2152.8 s | $2.7150 | $6.20 |
| 7 | `targets` 8B @16M, set `2026-09-16_v1` (H100) | 63.8 s | $0.0700 | $6.27 |
| 8 | vLLM n-shape test, 27B `rlI-150`, 128 x 16 (H200) | 577.0 s | $0.7276 | $7.00 |
| 9 | `score` rate probe, 27B `rlI-150`, 2,048 rows (H200) | 100.7 s | $0.1270 | $7.13 |
| 10 | `scan` 8B @16M, set `2026-09-16_v1` (H100) | 1662.9 s | $1.8246 | $8.95 |
| 11 | `rollouts_hf` 8B `run1-rl`, 1,536 x 64 (H100) | 1783.2 s | $1.9565 | $10.91 |
| 12 | `score` 8B `run1-rl`, 98,304 rows (H100) | 196.7 s | $0.2159 | $11.13 |
| 13 | `repo_examples` 8B -- FAILED on the sink assert (H100) | ~70 s | ~$0.08 (est.) | $11.21 |
| 14 | `repo_examples` 8B, 509 x 30 windows (H100) | 40.8 s | $0.0448 | $11.26 |
| 15 | `stats` 27B @16M (H200) | 7232.6 s | $9.1211 | $20.38 |
| 16 | `targets` 27B @16M, set `2026-09-16_v1` (H200) | 147.8 s | $0.1864 | $20.57 |
| 17 | 27B rollouts rlI-150 + rl-8x2048-full, STOPPED mid-run for the vLLM patch (2x H200) | 39 + 38 min | $5.82 (sunk) | $26.39 |
| 17b | 27B rlI-150 relaunch `ap-8QUg8to`, CANCELLED with its launcher client (H200) | ~6 min | ~$0.45 (sunk) | $26.84 |
| 17c | 27B full relaunch `ap-pBE6e6l`, CANCELLED the same way (H200) | ~10 min | ~$0.76 (sunk) | $27.60 |
| 18 | 8B parity rollouts on the PATCHED engine, 48 x 64 (H100) | 111.5 s | $0.1223 | $27.72 |
| 19 | 8B parity `score --engine vllm`, 3,072 rows (H100) | 33.4 s | $0.0367 | $27.76 |
| 20 | `scan` 27B @16M, set `2026-09-16_v1` (H200) | 5026.2 s | $6.3385 | $34.10 |
| 21 | `repo_examples` 27B, 512 x 32 windows (H200) | 146.0 s | $0.1841 | $34.28 |
| 22 | `rollouts_vllm` 27B `rlI-150`, 1,536 x 64, PATCHED (H200) | 4460.7 s | $5.6254 | $39.91 |
| 23 | `rollouts_vllm` 27B `rl-8x2048-full`, 1,536 x 64, PATCHED (H200) | 3572.0 s | $4.5046 | $44.41 |
| 24 | `score` 27B `rlI-150`, 98,304 rows (H200) | 487.3 s | $0.6145 | $45.03 |
| 25 | `score` 27B `rl-8x2048-full`, 98,304 rows (H200) | 503.1 s | $0.6345 | $45.66 |
| 26 | `centred` 27B `rlI-150` (CPU) | 23.7 s | ~$0 | $45.66 |
| 27 | `centred` 27B `rl-8x2048-full` (CPU) | 10.2 s | ~$0 | $45.66 |

**Total for this agent: $45.66 of a $110 cap.** $7.03 of that is sunk cost from the three
stop/cancel events around the vLLM patch (items 17, 17b, 17c); the patch itself turned a projected
$58-70 of 27B rollouts into $10.13, so the run finished at roughly half the pre-patch projection.
Items 22 and 23 were launched by the coordinating agent rather than by this one; they are in the
table because they are this run's products and their cost is part of the same budget.

### Engine choice, MEASURED 2026-09-16

`--throughput only` at `--max-num-seqs 512`, `--rows 0-31`, one engine per base, levels submitted
as `level / 16` requests of n=16. The HF column is the step-3 smoke's own `gen_tok_per_s` and the
rollouts/s derived from it (8B: 3,072 rollouts in 63.6 s of generate; 27B rlI-150: 512 rollouts in
137 s; 27B full: 512 in 96 s). **rollouts/s is the comparable number, not tok/s**: the HF path
counts the PADDED generate tensor (64 tokens on every row) while vLLM counts the tokens actually
returned (mean 41-56), so tok/s flatters HF by ~1.2-1.5x.

| concurrent rows | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| 8B `run1-rl`, gen tok/s | 123.4 | 438.4 | 568.8 | 739.4 | **808.1** |
| 8B `run1-rl`, rollouts/s | 2.40 | 9.74 | 13.05 | 18.24 | **19.52** |
| 27B `rlI-150`, gen tok/s | 72.5 | 190.5 | 278.2 | **365.7** | 345.5 |
| 27B `rlI-150`, rollouts/s | 1.33 | 3.51 | 5.01 | **6.50** | 6.13 |

The 27B **peaks at 256 rows in flight and gets slightly worse at 512** (365.7 -> 345.5 tok/s), so
the rollouts run uses `--max-num-seqs 256`, not 512. The 8B is still climbing at 512 but has
flattened (739 -> 808, +9% for 2x the rows).

| base / MAEMM | vLLM best rollouts/s | HF rollouts/s | ratio | engine chosen |
|---|---|---|---|---|
| 8B `2026-09-03_run1-rl` | 19.52 @ 512 | **48.3** | HF **2.47x** faster | **HF** (deviation, see below) |
| 27B `2026-09-08_rlI-150` (LoRA) | **6.50** @ 256 | 3.74 | vLLM 1.74x faster | **vLLM** |
| 27B `2026-09-10_rl-8x2048-full` | (not separately probed) | 5.33 | — | **vLLM** |

**DEVIATION, flagged.** Tomáš asked for vLLM. On the 8B the stock vllm-lens hook is 2.47x slower
per rollout than HF `generate` at every concurrency measured, which is past the ">1.5x cheaper"
bar, so the 8B rollouts go through `rollouts_hf`. This reproduces step 4's finding (vLLM 543 vs HF
3,092 tok/s) at 512 rows in flight rather than 256, i.e. more concurrency does not close the gap.
The 8B vLLM path stays available as the correctness cross-check it was built to be.

The 27B full model was NOT probed on vLLM separately -- a second 52 GiB engine build to measure
what the LoRA engine already bounds from below. It is served WITHOUT LoRA (no Punica path), which
is the cheaper of the two configurations, and HF's own numbers put the full model only 1.43x above
the LoRA (5.33 vs 3.74 rollouts/s) while vLLM's LoRA path is already 1.74x above HF's LoRA path.
That is an INFERENCE, not a measurement, and it is the one place the engine choice rests on one.

#### What the probe got wrong, and the two follow-ups that fixed it (2026-09-16)

The `--throughput` sweep is 16-sample requests. The rollouts run is n=64, and it submits 1,536
requests rather than 32, so the probe's 6.50 rollouts/s was measured on neither the request shape
nor the direction count the real run has. Both were checked directly, on the smoke root, against
`rlI-150` at 256 rows in flight -- so the three rows below differ ONLY in what they say they do:

| requests in flight x samples | distinct directions | rollouts/s | gen tok/s |
|---|---|---|---|
| 16 x 16 (probe, level 256) | 16 | **6.504** | 365.7 |
| 16 x 16 (128 targets, n=16) | 128 | **5.593** | 311.8 |
| 4 x 64 (128 targets, n=64) | 128 | **4.251** | 236.9 |

- **8x the distinct steering directions costs 14%** (6.504 -> 5.593). The stock vllm-lens extension
  rescans every registered steering key on every decode step, so this cost is real, but it is
  clearly SUBLINEAR: linear would have been 8x, not 1.16x. Extrapolating the same shape from 128 to
  the run's 1,536 directions puts the full run at roughly **3.5-4.2 rollouts/s**.
- **The n=64 request shape costs another 24%** (5.593 -> 4.251) at the SAME 256 rows in flight, i.e.
  4 requests instead of 16. Fewer, fatter requests are worse even when the row count is identical.
  That is why the rollouts run goes out at `--max-num-seqs 512`: with n=64 that is 8 requests in
  flight instead of 4. The probe's own 512-row level cost only 6% against its 256-row peak
  (6.13 vs 6.50 rollouts/s), so the extra rows are close to free while the extra requests are not.
  NOTE this is the one setting the run uses that was not itself measured -- it is the two rows
  above read together, not a third measurement.

At 3.5-4.25 rollouts/s the 27B vLLM run is **6.4-7.8 h / $29-35 per MAEMM**, against HF's measured
3.74 rollouts/s (7.3 h / $33.2) for the same LoRA. Neither is 1.5x the other, so **the 27B stays on
vLLM as asked**. The 10 h function timeout covers vLLM down to 2.7 rollouts/s.

#### The one thing that FAILED: adamkarvonen's max-acts file is not uniformly sink-prefixed

`repo_examples` 8B died on `common.strip_repo_sink`:

```
AssertionError: config says this max-acts file prepends the sink token 151645 to EVERY window, but
only 15286 of 15360 windows start with it (frac 0.995182); stripping column 0 would drop a real
token on the rest and shift every activation by one
```

The assert is RIGHT and was left alone (`common.py` is another agent's file this step, and the
guard is the reason the failure was visible at all rather than silently mis-aligning 74 windows'
activations by one position). The step-5 smoke passed only because it drew a different 512
features: `sink_first` is a property of each WINDOW in that file, not of the file.

Fixed in `precompute/repo_examples.py` alone, by adding `_sink_uniform_features`: before the
directions are built, each tested feature's windows are checked against the declaration and a
feature is dropped unless ALL of its windows match. The unit of exclusion is the feature, not the
window, because `n_win` is a fixed stride in eight places in `run()` and a per-window strip would
make the [F, W, T] grid ragged. `summary.json` gains `features_requested`,
`features_dropped_sink_nonuniform`, `features_dropped_ids` and
`windows_starting_with_sink_over_requested`, and the product README leads with the coverage line.

MEASURED after the fix: **3 of 512 features dropped** (ids 6038, 57336, 45175) -- the 74 bad
windows are concentrated in those three, so the `sae-repo-top32` column covers 509 of the 512 sae
features and the loss is 0.6%. The same check ran on the 27B file too (`sink_first: false`, so it
asks the opposite question) and dropped **0 of 512**: exactly **0 of its 16,384 windows** start
with the sink token, so that file's declaration is uniform and the 8B's is the odd one.

#### The vLLM speed patch (2026-09-16, mid-run)

Partway through the two 27B rollouts runs a separate agent diagnosed the vLLM slowness and Tomas
decided to apply the fix and relaunch. Two changes, both in `precompute/rollouts_vllm.py` plus a new
`precompute/vllm_ext.py`:

- **CUDA graphs for the decode steps.** The vllm-lens plugin forces `enforce_eager`; `build_engine`
  calls the UNPATCHED `create_engine_config`, so that choice is ours. Now `enforce_eager=False` with
  `compilation_config = {mode: 0, cudagraph_mode: FULL_DECODE_ONLY, max_cudagraph_capture_size:
  max_num_seqs}`. Mode 0 is no torch.compile and FULL_DECODE_ONLY graphs only uniform decode
  batches, so **prefill stays eager** -- which is what makes this safe, because the marker is a
  PROMPT position and `common.make_inject_hook` already no-ops on decode steps
  (`h.shape[1] <= 1`). `--eager` restores the old behaviour.
- **Celeste's fast hook, ported.** `precompute/vllm_ext.py` is `rl/fast_lens_ext.py` with the
  `try/except` around the hook body REMOVED, so a hook error raises instead of leaving a request
  silently unsteered -- exactly what the README said the file would be when it was written. It is
  now the DEFAULT (`--stock-hook` falls back), and steering goes in as one
  `set_steering_data_many` RPC per block with the requests carrying only a key, instead of one RPC
  per request.

`modal_app.py`'s `fast_hook` flag became `stock_hook`, and `eager` was added. ruff clean (three
findings, all in the new file: two import-sort auto-fixes and one 112-char line split);
`unit_smoke` 27/27.

**Speed, MEASURED on the 8B, same 3,072 rollouts (48 x 64) each time:**

| engine | gen tok/s | rollouts/s |
|---|---|---|
| HF `generate` (the delivered 8B production run) | 3,388 | 48.3 |
| vLLM, stock hook + enforce_eager (the step-4 path) | 808 | 19.5 |
| vLLM, fast hook + decode graphs (**patched**) | **4,068** | **126.9** |

So the patch is 6.5x the old vLLM and 2.6x HF. It also INVERTS the engine choice recorded above:
on the 8B, vLLM is now the cheap path, not the expensive one. The 8B production rollouts had
already completed on HF by then and were NOT re-run -- re-running costs more than it saves and the
two engines agree to the parity table below.

**Correctness, the reason the relaunch was allowed to proceed.** The patched engine was validated
on the 8B BEFORE the 27B jobs were killed, against the HF rollouts of the same 48 directions and
the same clean-base scorer:

| family | n | HF bo1 | vLLM bo1 | diff | r(bo1) | HF bo64 | vLLM bo64 | diff | r(bo64) |
|---|---|---|---|---|---|---|---|---|---|
| realact | 16 | 0.4978 | 0.4930 | -0.0048 | **0.9964** | 0.5978 | 0.5955 | -0.0024 | **0.9932** |
| random | 16 | 0.0257 | 0.0262 | +0.0005 | 0.9966 | 0.0437 | 0.0442 | +0.0004 | 0.9449 |
| sae | 16 | 0.0628 | 0.0640 | +0.0013 | 0.9948 | 0.1366 | 0.1431 | +0.0065 | 0.9663 |

That is no worse than the pre-patch step-4 parity and BETTER on best-of-64 (0.99 / 0.90 / 0.91
there against 0.9932 / 0.9449 / 0.9663 here), and HF's realact bo64 0.5978 still lands on run1's
archived 0.5967. Engine self-checks on the patched path: injection cos 0.999993, magnitude ratio
1.000154, pre-marker delta 0.1254 against a clean-vs-clean floor of exactly 0.1254, hook extension
errors 0, decode_skips 0.

**Sunk cost and the relaunch.** The two 27B rollouts runs were stopped at 03:55Z after 39 min
(`ap-XUkiFLbV8MQy4VVrTSeIrM`, rlI-150, launched 03:16Z) and 38 min (`ap-wqD9mjvTcRdw7dUazYqYGE`,
rl-8x2048-full, 03:17Z) -- **$2.95 + $2.87 = $5.82** of H200 thrown away. They were relaunched on
the patched engine at `--max-num-seqs 256 --force --detach`, all 1,536 targets x n=64:
`ap-8QUg8toazn2tgLXQDZbCg6` (rlI-150, 04:00Z) and `ap-pBE6e6l8YWjAX7dX1JDXNy` (rl-8x2048-full,
04:04Z). `scan` 27B (`ap-pKW5sZgMuxaTGlrFm3557D`) ran throughout and was not touched.

The rlI-150 relaunch then had to be made a THIRD time. `ap-8QUg8toazn2tgLXQDZbCg6` was cancelled at
04:04:51Z, six minutes in (~$0.45), by its own launcher: **`modal run --detach` keeps the APP alive
when the local client dies, but the in-flight `.remote()` input is cancelled with the client** --
the container log says "Received a cancellation signal while processing input". The engine had
already come up (226 s) and its self-checks had passed (cos 0.999995, ratio 0.999985), so nothing
was wrong with the patch. `--detach` protects against a dropped connection, not against the client
PROCESS exiting, which is what a shell `timeout` around the launch does. The final rlI-150 run is
`ap-W3TvvFgcwbL2ndWhXGafU8` (04:08Z), launched from a client that outlives the shell call. The
full-model relaunch `ap-pBE6e6l8YWjAX7dX1JDXNy` died the same way at 04:11:53Z (~$0.76) and its
final run is `ap-fgevNDjX2wwcFWB22n7OMi` (04:15Z).

That second casualty did leave one useful number behind: it had produced ~6,900 of 98,304 rollouts
in 5 minutes, i.e. ~23 rollouts/s. The completed runs bear that out exactly:

| 27B MAEMM | generate s | gen tok/s | rollouts/s | vs pre-patch vLLM (4.25) | vs HF |
|---|---|---|---|---|---|
| `rlI-150` (LoRA) | 4,117.6 | 818.1 | **23.87** | 5.6x | 6.4x (HF 3.74) |
| `rl-8x2048-full` | 3,352.8 | 952.6 | **29.32** | 6.9x | 5.5x (HF 5.33) |

So each 27B rollouts run came in at **$5.63 and $4.50** against the $29-35 projected before the
patch -- the patch saved roughly $50 on this run alone, several times the $7.03 of sunk cost the
three stop/cancel events cost to get it in.

Two further changes to `precompute/modal_app.py`, both made by the coordinating agent and checked
here: the GPU function timeouts stayed at the 10 h this run set them to, and the image `ignore`
list became `_IGNORE = ["**/__pycache__", "**/*.pyc", "**/.ruff_cache", "reconstruction/out",
"reconstruction/data", "**/*.md"]`. The last two entries matter for concurrent sessions: a write to
`reconstruction/out/` failed a relaunch's image build outright, and with `**/*.md` ignored a README
or SMOKES edit no longer invalidates the image or races someone else's build. Verified here that no
product reads a markdown file out of the code tree at runtime -- every product WRITES its README to
the volume. Against that, the patch takes the two 27B rollouts from a
projected 6.4-7.8 h / $29-35 each down to roughly an hour each, so it pays for itself several times
over in the same run. `scan` 27B was left running throughout.

#### Full-run results, all three MAEMMs, 512 targets per family x 64 rollouts

`mean cos` is the mean over targets of that target's mean-of-64; `bo64` is the mean over targets of
`best_of_k_means`'s naive best-of-64 (one group of 64, so simply the max of the 64). Read off
`per_target.jsonl` in each scores directory. The 8B is HF rollouts, the two 27B MAEMMs are patched
vLLM; the paired parity above is what licenses putting them in one table.

| MAEMM | realact cos | realact bo64 | random cos | random bo64 | sae cos | sae bo64 |
|---|---|---|---|---|---|---|
| 27B `rlI-150` (LoRA) | 0.4523 | 0.5701 | 0.0245 | 0.0399 | 0.1187 | 0.1625 |
| 27B `rl-8x2048-full` | **0.4994** | 0.5692 | 0.0280 | 0.0428 | **0.1298** | **0.1714** |
| 8B `run1-rl` | 0.5145 | **0.6195** | 0.0281 | 0.0462 | 0.0785 | 0.1489 |

Mean kept tokens / eos rate: 27B rlI 55.8 / 0.740 realact, 19.0 / 1.000 random, 28.0 / 1.000 sae;
27B full 46.2 / 0.969, 23.1 / 0.999, 28.2 / 0.998; 8B 41.8 / 0.881, 28.8 / 0.974, 21.7 / 0.995.

Three things worth flagging to whoever writes this up:

- **The two 27B MAEMMs separate on the MEAN but not on the best-of-64 realact.** The full model is
  +0.047 on realact mean-of-64 (0.4994 vs 0.4523) yet the two are within 0.001 on bo64 (0.5692 vs
  0.5701). The LoRA's realact rollouts are longer and finish far less often (eos 0.740 against
  0.969), i.e. it spends more of its budget and reaches the same ceiling less reliably.
- **The 8B beats both 27Bs on realact bo64** (0.6195) while losing to both on `sae`. Its realact
  bo64 also sits just above run1's archived 0.5967 on a fresh 16M draw, which is the closest thing
  this run has to an external check on the whole pipeline.
- **On the 8B `sae` family the MAEMM does not beat the trivial text baseline.** Its best-of-64 is
  0.1489 against `sae-repo-top32`'s mean max_cos of **0.2351** over the same features -- i.e.
  simply showing the SAE repo's own highest-activating 32-token window scores higher than the best
  of 64 MAEMM rollouts. The 27B's `sae` bo64 (0.1625 / 0.1714) against its own repo baseline of
  0.1673 is roughly a tie. Corpus retrieval is the other baseline and lives in `scan/`.

#### `score` is 3.4x cheaper at scale than the smoke implied

The step-3 smoke scored 512 rows on the 27B at 12.9 rows/s, which extrapolated to ~2.1 h and ~$9.7
per MAEMM at the full 98,304 rows and was the second largest line in the budget. MEASURED
2026-09-16 on 2,048 rows: **43.9 rows/s**. The smoke was warmup-dominated -- 16 chunks of
`SCORE_CHUNK`=32 is not enough to amortise the first forwards. The full-set projection is therefore
~2,240 s and **~$2.9 per 27B MAEMM**, not $9.7.

## `reconstruction/stats.py` on the smoke root

2026-09-16. `uv run reconstruction/stats.py --root-tag smoke` — local, CPU, no GPU: it fetched 37
small files (9.4 MB: per_target.jsonl, rows.json, index.json, cos.f16, argmax.i16, the two `centred`
arrays, ids.jsonl, topk.jsonl, quantiles.f16, per_feature.jsonl, finals.jsonl) off
`/vol/runs/2026-09-15_paper-evals-smoke` into `reconstruction/data/smoke/` and wrote all nine tables
into `reconstruction/out/smoke/`. `best_act.f16` was NOT fetched. Inputs on this root: 8B
`2026-09-03_run1-rl` on the imported `2026-09-03_run1-archive16` set (48 x 64), and both 27B MAEMMs
on `2026-09-16_v1` (8 realact rows x 64). Four of the nine tables are pasted below; the rest are in
`reconstruction/out/smoke/`.

The per-rollout maxima recomputed from `cos.f16` agree with each product's own `per_target.jsonl`
`mean_cos` on every row (asserted at 2e-3; that check is what makes a cached `data/` directory safe).
Run1's realact best-of-64 lands at **0.5978** against the archive's **0.5967**, as step 3 measured.

### (a) mean cosine and unbiased best-of-n, per family x MAEMM

set 2026-09-03_run1-archive16, 2026-09-16_v1 | n 64 | bo 64 | seed 1234 | corpus 1M, 2M | bo-k UNBIASED order statistic, ± SE across targets; the last column is the disjoint-group estimator the product stores, which equals the unbiased one at k = n

| base | maemm | family | targets | sha | mean cos (bo1) | bo1 | bo2 | bo4 | bo8 | bo16 | bo32 | bo64 | bo64 naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2026-09-03_run1-rl | random | 16 | 327d180cfb33 | 0.0257 ± 0.0031 | 0.0257 ± 0.0031 | 0.0297 ± 0.0032 | 0.0332 ± 0.0032 | 0.0364 ± 0.0033 | 0.0393 ± 0.0033 | 0.0417 ± 0.0033 | 0.0437 ± 0.0033 | 0.0437 |
| qwen3-8b | 2026-09-03_run1-rl | realact | 16 | 327d180cfb33 | 0.4978 ± 0.0400 | 0.4978 ± 0.0400 | 0.5306 ± 0.0411 | 0.5522 ± 0.0412 | 0.5678 ± 0.0408 | 0.5799 ± 0.0402 | 0.5897 ± 0.0395 | 0.5978 ± 0.0384 | 0.5978 |
| qwen3-8b | 2026-09-03_run1-rl | sae | 16 | 327d180cfb33 | 0.0628 ± 0.0141 | 0.0628 ± 0.0141 | 0.0775 ± 0.0159 | 0.0922 ± 0.0179 | 0.1061 ± 0.0198 | 0.1185 ± 0.0212 | 0.1292 ± 0.0223 | 0.1366 ± 0.0229 | 0.1366 |
| qwen36-27b | 2026-09-08_rlI-150 | realact | 8 | 75bd20ce725f | 0.4361 ± 0.0457 | 0.4361 ± 0.0457 | 0.4828 ± 0.0445 | 0.5131 ± 0.0443 | 0.5291 ± 0.0443 | 0.5378 ± 0.0438 | 0.5444 ± 0.0437 | 0.5496 ± 0.0434 | 0.5496 |
| qwen36-27b | 2026-09-10_rl-8x2048-full | realact | 8 | d38845553391 | 0.4694 ± 0.0521 | 0.4694 ± 0.0521 | 0.4955 ± 0.0493 | 0.5134 ± 0.0468 | 0.5261 ± 0.0449 | 0.5341 ± 0.0437 | 0.5391 ± 0.0432 | 0.5430 ± 0.0432 | 0.5430 |

### (d) corpus-retrieval baseline and cosine quantiles, per nested corpus size

scan geometry 64/16 (common.windows_of) | sizes are nested prefixes | corpus max 2M | top-1 and mean-of-top-64 are per target, then averaged ± SE; pXX are the scan's own cos quantiles over ALL windows, averaged over targets -- the calibration of what a given cosine means (checklist item 67)

| base | set | family | corpus | targets | top-1 cos | mean of top-64 | p50 | p90 | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2026-09-16_v1 | random | 1M | 512 | 0.0641 ± 0.0004 | 0.0548 ± 0.0003 | 0.0027 | 0.0168 | 0.0292 | 0.0392 | 0.0478 |
| qwen3-8b | 2026-09-16_v1 | realact | 1M | 512 | 0.4757 ± 0.0053 | 0.3567 ± 0.0047 | -0.0135 | 0.0389 | 0.107 | 0.1764 | 0.2608 |
| qwen3-8b | 2026-09-16_v1 | sae | 1M | 512 | 0.1779 ± 0.0033 | 0.1098 ± 0.0023 | -0.0073 | 0.0062 | 0.0177 | 0.0308 | 0.058 |
| qwen3-8b | 2026-09-16_v1 | random | 2M | 512 | 0.0661 ± 0.0003 | 0.0570 ± 0.0003 | 0.0026 | 0.0167 | 0.0291 | 0.039 | 0.0477 |
| qwen3-8b | 2026-09-16_v1 | realact | 2M | 512 | 0.4894 ± 0.0050 | 0.3773 ± 0.0047 | -0.0134 | 0.0387 | 0.1065 | 0.1752 | 0.2575 |
| qwen3-8b | 2026-09-16_v1 | sae | 2M | 512 | 0.1920 ± 0.0033 | 0.1295 ± 0.0026 | -0.0073 | 0.0063 | 0.0177 | 0.0309 | 0.0584 |
| qwen36-27b | 2026-09-16_v1 | random | 1M | 512 | 0.0540 ± 0.0003 | 0.0452 ± 0.0004 | 0.0017 | 0.014 | 0.0243 | 0.0323 | 0.0393 |
| qwen36-27b | 2026-09-16_v1 | realact | 1M | 512 | 0.3828 ± 0.0056 | 0.2739 ± 0.0050 | -0.0251 | 0.0257 | 0.0813 | 0.1352 | 0.1975 |
| qwen36-27b | 2026-09-16_v1 | sae | 1M | 512 | 0.1233 ± 0.0022 | 0.0682 ± 0.0014 | -0.0172 | -0.0046 | 0.0061 | 0.0165 | 0.036 |

### (e) the SAE repo's own max-activating windows through our scorer (sae family only)

`max_cos` IS the sae-repo-top32 baseline column (best of the feature's shipped windows, the analogue of a best-of-32 rollout draw); `frac fired` is the share of a feature's own windows whose peak clears the checkpoint's learned gate; `peak r` / `argmax agree` are the repo-vs-us activation agreement, ± SE across features

| base | sae | set | features | windows/feature | max_cos (the column) | mean_cos | frac fired | features ever firing | peak r (mean) | peak r (median) | argmax agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | qwen3-8b/adamkarvonen-t2 | 2026-09-16_v1 | 512 | 30 | 0.2296 ± 0.0034 | 0.1975 ± 0.0033 | 0.9999 | 1.0 | 0.9941 | 0.9979 | 0.9974 |
| qwen36-27b | qwen36-27b/l42-1b | 2026-09-16_v1 | 512 | 32 | 0.1708 ± 0.0026 | 0.1127 ± 0.0018 | 0.9777 | 1.0 | 0.5719 | 0.5909 | 0.8082 |

### (g) argmax-token position as a fraction of the scored tokens

set 2026-09-03_run1-archive16, 2026-09-16_v1 | n 64 | bo 64 | seed 1234 | corpus 1M, 2M | bins are fractions of `argmax / n_kept_tokens`; 'at last token' is argmax == n-1. The recipe moves this a lot (77-82% vs ~4% last-token across recipes, checklist item 10), so it is reported per family x MAEMM, never pooled

| base | maemm | family | rollouts | mean rel pos | [0.00,0.12) | [0.12,0.25) | [0.25,0.38) | [0.38,0.50) | [0.50,0.62) | [0.62,0.75) | [0.75,0.88) | [0.88,1.00) | at last token | mean kept tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2026-09-03_run1-rl | random | 1024 | 0.6318 | 0.043 | 0.0645 | 0.0879 | 0.1006 | 0.1426 | 0.1387 | 0.1777 | 0.2451 | 0.125 | 32.69 |
| qwen3-8b | 2026-09-03_run1-rl | realact | 1024 | 0.9085 | 0.002 | 0.0059 | 0.0098 | 0.0205 | 0.0322 | 0.0488 | 0.0518 | 0.8291 | 0.7734 | 40.63 |
| qwen3-8b | 2026-09-03_run1-rl | sae | 1024 | 0.54 | 0.1396 | 0.083 | 0.0908 | 0.1055 | 0.1182 | 0.123 | 0.1211 | 0.2188 | 0.1426 | 19.78 |
| qwen36-27b | 2026-09-08_rlI-150 | realact | 512 | 0.9231 | 0.0 | 0.0 | 0.002 | 0.0039 | 0.0195 | 0.0742 | 0.1094 | 0.791 | 0.6406 | 55.04 |
| qwen36-27b | 2026-09-10_rl-8x2048-full | realact | 512 | 0.9696 | 0.0 | 0.002 | 0.0 | 0.0 | 0.0 | 0.0078 | 0.0137 | 0.9766 | 0.916 | 46.54 |

What the four say on this root. (a) the unbiased best-of-k curve has not saturated
at 64 on any family, and the `random` control rises with it too (0.0257 -> 0.0437, +70%), which is
checklist item 26's point and why the control is in the table. (d) the corpus scan's top-1 at 2M
tokens reaches realact **0.4894** against run1's **0.5978** best-of-64 — the retrieval baseline is
within 0.11 of the trained inverter on the 8B — while the same scan's p99.99 over all windows is
0.2575, so a cosine of 0.49 is a 1-in-10^4 window, not a typical one. (e) the `sae-repo-top32`
column is 0.2296 (8B) / 0.1708 (27B) against the 8B MAEMM's sae best-of-64 of 0.1366: the SAE
repo's own natural text beats the inverter on its own features. (g) the argmax sits at the LAST
token on 77.3% of 8B realact rollouts and on 64.1% / 91.6% of the two 27B MAEMMs' — against 12.5%
(random) and 14.3% (sae) — the recipe dependence of checklist item 10, reproduced.

## Step 6 products: `centred` and `patchscopes`

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-16 | `centred` 8B run1 (CPU) | `--product centred --base qwen3-8b --maemm qwen3-8b/2026-09-03_run1-rl --set 2026-09-03_run1-archive16 --root …smoke` | 11.0 s | ~$0 | 48x64 rows: realact bo1/bo64 primary 0.4978 / 0.5978, **centred 0.7084 / 0.8363**, filtered 0.4978 / 0.5978; **36 of 95,337** kept tokens dropped by the 10x-median filter (0.038%), touching 36 rollouts | the filter changes the per-row MAXIMUM on **no** row (Δ 0.0000 at 4 dp on all three families): the tokens it drops are never the argmax |
| 2026-09-16 | `centred` 27B rlI-150 (CPU) | `--product centred --base qwen36-27b --maemm qwen36-27b/2026-09-08_rlI-150 --set 2026-09-16_v1 --root …smoke` | 5.9 s | ~$0 | 8x64 realact: primary 0.4361 / 0.5496, **centred 0.6958 / 0.8627**; **0 of 28,181** tokens dropped | matches checklist item 4's "empirically inert on a full 27B re-embed (0 of 98,304)" |
| 2026-09-16 | `centred` 27B rl-8x2048-full (CPU) | same, `--maemm qwen36-27b/2026-09-10_rl-8x2048-full` | 8.8 s | ~$0 | 8x64 realact: primary 0.4694 / 0.5430, **centred 0.7488 / 0.8550**; 0 of 23,826 dropped | — |
| 2026-09-16 | `patchscopes` 27B, floor + one layer | `--product patchscopes --base qwen36-27b --set 2026-09-16_v1 --rows 0-3 --n 2 --ps-layers 14 --root …smoke` | 95.2 s | **$0.1201** | both cells written in the rollouts_hf schema; patch check `cos(h_patched, v)` = **1.0000** on 4/4 directions with `‖Δh‖/‖h‖` 2.17-2.40 (α=2 replacement of an ~orthogonal direction predicts √5 = 2.236) | the 27B generated at only **31 tok/s** at batch 8 — the plan's 240 tok/s assumes batch 32, so the final sweep's wall must be re-projected from its own first cell |
| 2026-09-16 | `score --rollouts-dir` on both cells | `--product score --base qwen36-27b --set 2026-09-16_v1 --rollouts-dir …/patchscopes/2026-09-16_v1/<cell>` | 100.8 s + 105.6 s | $0.1272 + $0.1332 | **the schema round-trips**: `score.py` consumed `rollouts.jsonl` unchanged and wrote a full scores dir (cos/norm/argmax/best_act/SAE CSR/per_target.jsonl) for each cell. 4 realact directions, bo 2: floor bo1 **0.0099** / bo2 0.0216, injected L14 bo1 **0.0230** / bo2 0.0409 | the first attempt died locally with `gcg/modal_app.py was modified during build process` — a concurrent session editing `paper-evals/` while the image layer was being built; the retry was clean |

Two notes on the patchscopes numbers. They are a **schema proof, not a measurement**: 4 directions
at bo 2 on one layer, and the sweep in the main README is what produces a reportable cell. And the
absolute scale is far below the 8B trial's (realact floor 0.1463 / injected 0.2333 at bo 64 on 512
directions) — partly the 27B's own cosine scale (its corpus-scan p99.99 is 0.1975 against the 8B's
0.2575, table (d)), partly 4 directions at bo 2 against 512 at bo 64.

## `reconstruction/stats.py` on the full root (8B, interim)

2026-09-16, `uv run reconstruction/stats.py --root-tag full` — local, CPU, $0, 34 MB fetched.
The 8B side of `/vol` is complete (16M corpus, stats, `2026-09-16_v1` targets, scan, run1-rl HF
rollouts + scores, repo_examples); the 27B's scan and its two vLLM rollout runs were still in
flight, so (c) paired MAEMMs, (f) GCG and (j) Patchscopes printed their skip notes and seven tables
were written. `precompute/centred.py` was run on the full 8B scores dir first (CPU, 20.5 s, ~$0;
854 MiB copied through the temp dir in 15.4 s) so table (h) exists: **1,076 of 2,930,344** kept
tokens dropped by the 10x-median filter (0.037%), touching 1,071 of 98,304 rollouts, and the only
maximum it moves anywhere is one sae row by 3e-6.

Headline, 8B run1-rl, 512 directions per family x 64 rollouts, unbiased estimator:

| family | bo1 | bo64 | centred bo64 | corpus scan top-1 @16M | corpus p99.99 @16M | sae-repo-top30 |
|---|---|---|---|---|---|---|
| realact | 0.5145 ± 0.0058 | **0.6195 ± 0.0050** | 0.8532 | **0.5260 ± 0.0048** | 0.2519 | — |
| sae | 0.0785 ± 0.0034 | **0.1489 ± 0.0044** | 0.2065 | 0.2282 ± 0.0036 | 0.0582 | **0.2351 ± 0.0036** |
| random | 0.0281 ± 0.0005 | **0.0462 ± 0.0005** | 0.0616 | 0.0717 ± 0.0003 | 0.0477 | — |

Three things the full 8B root says that the smoke could not. **The corpus scan has caught up with
the inverter**: at 16M tokens its top-1 realact window is 0.5260 against the MAEMM's best-of-64
0.6195, a gap of 0.093 that was 0.11 at 2M and shrinks monotonically with corpus size (0.4315 ->
0.4596 -> 0.4865 -> 0.5133 -> 0.5260 at 1/2/4/8/16M) — the baseline is still climbing where the
inverter is fixed. **On the sae family the search already wins**: corpus top-1 0.2282 and the SAE
repo's own shipped windows 0.2351, both above the MAEMM's 0.1489 best-of-64. And the random control
reaches 0.0717 by corpus search against 0.0462 by inversion, i.e. on the two families where
inversion is weak, *any* search over natural text beats it.


### (a) mean cosine and unbiased best-of-n, per family x MAEMM

set 2026-09-16_v1 | n 64 | bo 64 | seed 1234 | corpus 16M | bo-k UNBIASED order statistic, ± SE across targets; the last column is the disjoint-group estimator the product stores, which equals the unbiased one at k = n

| base | maemm | family | targets | sha | mean cos (bo1) | bo1 | bo2 | bo4 | bo8 | bo16 | bo32 | bo64 | bo64 naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2026-09-03_run1-rl | random | 512 | 327d180cfb33 | 0.0281 ± 0.0005 | 0.0281 ± 0.0005 | 0.0320 ± 0.0005 | 0.0354 ± 0.0005 | 0.0385 ± 0.0005 | 0.0413 ± 0.0005 | 0.0439 ± 0.0005 | 0.0462 ± 0.0005 | 0.0462 |
| qwen3-8b | 2026-09-03_run1-rl | realact | 512 | 327d180cfb33 | 0.5145 ± 0.0058 | 0.5145 ± 0.0058 | 0.5477 ± 0.0057 | 0.5696 ± 0.0055 | 0.5858 ± 0.0054 | 0.5988 ± 0.0052 | 0.6097 ± 0.0051 | 0.6195 ± 0.0050 | 0.6195 |
| qwen3-8b | 2026-09-03_run1-rl | sae | 512 | 327d180cfb33 | 0.0785 ± 0.0034 | 0.0785 ± 0.0034 | 0.0932 ± 0.0037 | 0.1064 ± 0.0039 | 0.1185 ± 0.0041 | 0.1294 ± 0.0042 | 0.1395 ± 0.0043 | 0.1489 ± 0.0044 | 0.1489 |

### (d) corpus-retrieval baseline and cosine quantiles, per nested corpus size

scan geometry 64/16 (common.windows_of) | sizes are nested prefixes | corpus max 16M | top-1 and mean-of-top-64 are per target, then averaged ± SE; pXX are the scan's own cos quantiles over ALL windows, averaged over targets -- the calibration of what a given cosine means (checklist item 67)

| base | set | family | corpus | targets | top-1 cos | mean of top-64 | p50 | p90 | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2026-09-16_v1 | random | 1M | 512 | 0.0640 ± 0.0004 | 0.0546 ± 0.0003 | 0.0026 | 0.0166 | 0.029 | 0.0388 | 0.0475 |
| qwen3-8b | 2026-09-16_v1 | realact | 1M | 512 | 0.4315 ± 0.0052 | 0.3404 ± 0.0049 | -0.0167 | 0.0352 | 0.102 | 0.1697 | 0.2505 |
| qwen3-8b | 2026-09-16_v1 | sae | 1M | 512 | 0.1769 ± 0.0034 | 0.1111 ± 0.0023 | -0.007 | 0.0062 | 0.0179 | 0.0314 | 0.0577 |
| qwen3-8b | 2026-09-16_v1 | random | 2M | 512 | 0.0662 ± 0.0003 | 0.0570 ± 0.0003 | 0.0026 | 0.0166 | 0.029 | 0.0389 | 0.0476 |
| qwen3-8b | 2026-09-16_v1 | realact | 2M | 512 | 0.4596 ± 0.0052 | 0.3662 ± 0.0048 | -0.0169 | 0.0352 | 0.1021 | 0.17 | 0.2513 |
| qwen3-8b | 2026-09-16_v1 | sae | 2M | 512 | 0.1930 ± 0.0035 | 0.1311 ± 0.0026 | -0.007 | 0.0063 | 0.018 | 0.0316 | 0.0582 |
| qwen3-8b | 2026-09-16_v1 | random | 4M | 512 | 0.0681 ± 0.0003 | 0.0592 ± 0.0003 | 0.0026 | 0.0166 | 0.029 | 0.039 | 0.0476 |
| qwen3-8b | 2026-09-16_v1 | realact | 4M | 512 | 0.4865 ± 0.0050 | 0.3914 ± 0.0048 | -0.0168 | 0.0353 | 0.1024 | 0.1705 | 0.2523 |
| qwen3-8b | 2026-09-16_v1 | sae | 4M | 512 | 0.2071 ± 0.0035 | 0.1490 ± 0.0028 | -0.0071 | 0.0063 | 0.018 | 0.0316 | 0.0582 |
| qwen3-8b | 2026-09-16_v1 | random | 8M | 512 | 0.0698 ± 0.0003 | 0.0612 ± 0.0003 | 0.0026 | 0.0166 | 0.029 | 0.039 | 0.0476 |
| qwen3-8b | 2026-09-16_v1 | realact | 8M | 512 | 0.5133 ± 0.0049 | 0.4162 ± 0.0048 | -0.0168 | 0.0351 | 0.1021 | 0.1703 | 0.2523 |
| qwen3-8b | 2026-09-16_v1 | sae | 8M | 512 | 0.2185 ± 0.0036 | 0.1653 ± 0.0030 | -0.0071 | 0.0063 | 0.018 | 0.0316 | 0.0582 |
| qwen3-8b | 2026-09-16_v1 | random | 16M | 512 | 0.0717 ± 0.0003 | 0.0632 ± 0.0003 | 0.0026 | 0.0166 | 0.029 | 0.0389 | 0.0477 |
| qwen3-8b | 2026-09-16_v1 | realact | 16M | 512 | 0.5260 ± 0.0048 | 0.4358 ± 0.0047 | -0.0169 | 0.0351 | 0.1021 | 0.1701 | 0.2519 |
| qwen3-8b | 2026-09-16_v1 | sae | 16M | 512 | 0.2282 ± 0.0036 | 0.1801 ± 0.0032 | -0.007 | 0.0064 | 0.018 | 0.0315 | 0.0582 |

### (e) the SAE repo's own max-activating windows through our scorer (sae family only)

`max_cos` IS the sae-repo-top32 baseline column (best of the feature's shipped windows, the analogue of a best-of-32 rollout draw); `frac fired` is the share of a feature's own windows whose peak clears the checkpoint's learned gate; `peak r` / `argmax agree` are the repo-vs-us activation agreement, ± SE across features

| base | sae | set | features | windows/feature | max_cos (the column) | mean_cos | frac fired | features ever firing | peak r (mean) | peak r (median) | argmax agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | qwen3-8b/adamkarvonen-t2 | 2026-09-16_v1 | 509 | 30 | 0.2351 ± 0.0036 | 0.2022 ± 0.0034 | 0.9997 | 1.0 | 0.9912 | 0.9979 | 0.9974 |

### (g) argmax-token position as a fraction of the scored tokens

set 2026-09-16_v1 | n 64 | bo 64 | seed 1234 | corpus 16M | bins are fractions of `argmax / n_kept_tokens`; 'at last token' is argmax == n-1. The recipe moves this a lot (77-82% vs ~4% last-token across recipes, checklist item 10), so it is reported per family x MAEMM, never pooled

| base | maemm | family | rollouts | mean rel pos | [0.00,0.12) | [0.12,0.25) | [0.25,0.38) | [0.38,0.50) | [0.50,0.62) | [0.62,0.75) | [0.75,0.88) | [0.88,1.00) | at last token | mean kept tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | 2026-09-03_run1-rl | random | 32768 | 0.6218 | 0.0478 | 0.0652 | 0.0951 | 0.1079 | 0.136 | 0.1418 | 0.1617 | 0.2445 | 0.1242 | 27.83 |
| qwen3-8b | 2026-09-03_run1-rl | realact | 32768 | 0.8956 | 0.001 | 0.0087 | 0.0198 | 0.0286 | 0.04 | 0.0441 | 0.0435 | 0.8145 | 0.7639 | 40.95 |
| qwen3-8b | 2026-09-03_run1-rl | sae | 32768 | 0.5814 | 0.0827 | 0.0802 | 0.1039 | 0.1089 | 0.1227 | 0.123 | 0.1317 | 0.2469 | 0.1543 | 20.65 |

Notes on the four. (a) the bo curve has still not saturated at 64 on any family, and
the `random` control gains +64% from bo1 to bo64 (0.0281 -> 0.0462): the compute axis flatters every
family, which is why table (b) reports the ranking as a function of bo. (d) the cosine quantiles
barely move with corpus size (realact p99.99 is 0.2505 at 1M and 0.2519 at 16M) while the top-1
climbs by 0.095 — more corpus does not change what a typical window scores, only how far into the
tail the search reaches. (e) 509 of the 512 drawn features had shipped windows; the repo-vs-us
activation agreement on the 8B is r = 0.9912 mean / 0.9979 median with 99.74% argmax agreement, so
the column is our scorer on their text, as claimed. (g) 76.4% of realact rollouts put the argmax at
the LAST token, against 12.4% (random) and 15.4% (sae) — checklist item 10's recipe dependence,
now on 32,768 rollouts per family rather than the smoke's 1,024.

## `reconstruction/stats.py` on the full root (final, 27B)

2026-09-16, `uv run reconstruction/stats.py --root-tag full` after the driver landed both 27B
`scores/2026-09-16_v1__vllm/`, `centred` on both, 27B `repo_examples` (512/512) and the 27B scan.
Three scores directories now load — 8B run1-rl (HF stem) and both 27B MAEMMs (**`__vllm` stem**,
labelled `@vllm`); stats.py gained that stem this session, without it the whole 27B would have been
skipped. Eight tables written; (f) GCG is empty on this root (the arms live on the smoke root) and
(j) Patchscopes follows when its final cell lands. 53.5 MB fetched, $0 (local).

### (a) 27B rows — mean cosine and unbiased best-of-n

| base | maemm | family | targets | sha | mean cos (bo1) | bo1 | bo2 | bo4 | bo8 | bo16 | bo32 | bo64 | bo64 naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | 2026-09-08_rlI-150@vllm | random | 512 | 75bd20ce725f | 0.0245 ± 0.0005 | 0.0245 ± 0.0005 | 0.0278 ± 0.0005 | 0.0307 ± 0.0005 | 0.0333 ± 0.0005 | 0.0357 ± 0.0005 | 0.0379 ± 0.0005 | 0.0399 ± 0.0005 | 0.0399 |
| qwen36-27b | 2026-09-08_rlI-150@vllm | realact | 512 | 75bd20ce725f | 0.4523 ± 0.0068 | 0.4523 ± 0.0068 | 0.4940 ± 0.0068 | 0.5221 ± 0.0067 | 0.5406 ± 0.0065 | 0.5534 ± 0.0064 | 0.5626 ± 0.0063 | 0.5701 ± 0.0062 | 0.5701 |
| qwen36-27b | 2026-09-08_rlI-150@vllm | sae | 512 | 75bd20ce725f | 0.1187 ± 0.0032 | 0.1187 ± 0.0032 | 0.1292 ± 0.0033 | 0.1379 ± 0.0034 | 0.1452 ± 0.0035 | 0.1516 ± 0.0035 | 0.1574 ± 0.0035 | 0.1625 ± 0.0036 | 0.1625 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | random | 512 | d38845553391 | 0.0280 ± 0.0005 | 0.0280 ± 0.0005 | 0.0312 ± 0.0005 | 0.0340 ± 0.0005 | 0.0365 ± 0.0005 | 0.0388 ± 0.0005 | 0.0409 ± 0.0005 | 0.0428 ± 0.0005 | 0.0428 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | realact | 512 | d38845553391 | 0.4994 ± 0.0070 | 0.4994 ± 0.0070 | 0.5223 ± 0.0068 | 0.5371 ± 0.0066 | 0.5477 ± 0.0065 | 0.5560 ± 0.0064 | 0.5630 ± 0.0063 | 0.5692 ± 0.0063 | 0.5692 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | sae | 512 | d38845553391 | 0.1298 ± 0.0032 | 0.1298 ± 0.0032 | 0.1400 ± 0.0033 | 0.1484 ± 0.0034 | 0.1554 ± 0.0034 | 0.1614 ± 0.0035 | 0.1668 ± 0.0035 | 0.1714 ± 0.0035 | 0.1714 |

### (b) 27B — the ranking flips at bo 64, once, on realact

| base | family | bo | 2026-09-03_run1-rl | 2026-09-08_rlI-150@vllm | 2026-09-10_rl-8x2048-full@vllm | ranking |
|---|---|---|---|---|---|---|
| qwen36-27b | random | 1 |  | 0.0245 | 0.028 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | random | 4 |  | 0.0307 | 0.034 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | random | 16 |  | 0.0357 | 0.0388 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | random | 64 |  | 0.0399 | 0.0428 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | realact | 1 |  | 0.4523 | 0.4994 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | realact | 4 |  | 0.5221 | 0.5371 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | realact | 16 |  | 0.5534 | 0.556 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | realact | 64 |  | 0.5701 | 0.5692 | 2026-09-08_rlI-150@vllm > 2026-09-10_rl-8x2048-full@vllm |
| qwen36-27b | sae | 1 |  | 0.1187 | 0.1298 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | sae | 4 |  | 0.1379 | 0.1484 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | sae | 16 |  | 0.1516 | 0.1614 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |
| qwen36-27b | sae | 64 |  | 0.1625 | 0.1714 | 2026-09-10_rl-8x2048-full@vllm > 2026-09-08_rlI-150@vllm |

### (c) paired, 512 directions per family, same rows and same budget

| base | family | slice | statistic | targets | 2026-09-08_rlI-150@vllm | 2026-09-10_rl-8x2048-full@vllm | diff (A-B) | 2026-09-08_rlI-150@vllm wins | sign-test p |
|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | random | all | best-of-64 (unbiased) | 512 | 0.0399 | 0.0428 | -0.0029 ± 0.0003 | 0.337 | 1.27e-13 |
| qwen36-27b | random | all | mean cos | 512 | 0.0245 | 0.028 | -0.0035 ± 0.0002 | 0.240 | 3.55e-33 |
| qwen36-27b | realact | all | best-of-64 (unbiased) | 512 | 0.5701 | 0.5692 | 0.0009 ± 0.0013 | 0.527 | 0.247 |
| qwen36-27b | realact | all | mean cos | 512 | 0.4523 | 0.4994 | -0.0471 ± 0.0031 | 0.213 | 1.18e-40 |
| qwen36-27b | sae | all | best-of-64 (unbiased) | 512 | 0.1625 | 0.1714 | -0.0090 ± 0.0014 | 0.380 | 7.34e-08 |
| qwen36-27b | sae | all | mean cos | 512 | 0.1187 | 0.1298 | -0.0111 ± 0.0011 | 0.271 | 9.81e-26 |
| qwen36-27b | sae | density q0 | best-of-64 (unbiased) | 128 | 0.1303 | 0.1464 | -0.0161 ± 0.0040 | 0.383 | 0.0101 |
| qwen36-27b | sae | density q0 | mean cos | 128 | 0.0805 | 0.0985 | -0.0179 ± 0.0028 | 0.172 | 2.21e-14 |
| qwen36-27b | sae | density q1 | best-of-64 (unbiased) | 128 | 0.2141 | 0.2216 | -0.0075 ± 0.0026 | 0.422 | 0.0927 |
| qwen36-27b | sae | density q1 | mean cos | 128 | 0.1663 | 0.1766 | -0.0103 ± 0.0025 | 0.297 | 4.92e-06 |
| qwen36-27b | sae | density q2 | best-of-64 (unbiased) | 128 | 0.1759 | 0.1809 | -0.0050 ± 0.0018 | 0.398 | 0.0267 |
| qwen36-27b | sae | density q2 | mean cos | 128 | 0.135 | 0.1429 | -0.0079 ± 0.0015 | 0.289 | 2.01e-06 |
| qwen36-27b | sae | density q3 | best-of-64 (unbiased) | 128 | 0.1296 | 0.1368 | -0.0072 ± 0.0017 | 0.317 | 5.1e-05 |
| qwen36-27b | sae | density q3 | mean cos | 128 | 0.093 | 0.1012 | -0.0082 ± 0.0017 | 0.328 | 0.000125 |

### (d) 27B corpus-retrieval baseline per nested corpus size

| base | set | family | corpus | targets | top-1 cos | mean of top-64 | p50 | p90 | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | 2026-09-16_v1 | random | 1M | 512 | 0.0541 ± 0.0004 | 0.0451 ± 0.0004 | 0.0018 | 0.014 | 0.0242 | 0.0322 | 0.0395 |
| qwen36-27b | 2026-09-16_v1 | realact | 1M | 512 | 0.3216 ± 0.0057 | 0.2430 ± 0.0053 | -0.038 | 0.0127 | 0.0699 | 0.1243 | 0.1801 |
| qwen36-27b | 2026-09-16_v1 | sae | 1M | 512 | 0.1125 ± 0.0023 | 0.0620 ± 0.0014 | -0.0167 | -0.0041 | 0.0063 | 0.016 | 0.0333 |
| qwen36-27b | 2026-09-16_v1 | random | 2M | 512 | 0.0559 ± 0.0003 | 0.0471 ± 0.0003 | 0.0017 | 0.014 | 0.0242 | 0.0323 | 0.0393 |
| qwen36-27b | 2026-09-16_v1 | realact | 2M | 512 | 0.3433 ± 0.0058 | 0.2630 ± 0.0054 | -0.0378 | 0.013 | 0.07 | 0.1243 | 0.1806 |
| qwen36-27b | 2026-09-16_v1 | sae | 2M | 512 | 0.1278 ± 0.0023 | 0.0741 ± 0.0015 | -0.0167 | -0.0041 | 0.0063 | 0.0159 | 0.0332 |
| qwen36-27b | 2026-09-16_v1 | random | 4M | 512 | 0.0579 ± 0.0003 | 0.0491 ± 0.0003 | 0.0017 | 0.014 | 0.0242 | 0.0323 | 0.0394 |
| qwen36-27b | 2026-09-16_v1 | realact | 4M | 512 | 0.3706 ± 0.0058 | 0.2853 ± 0.0054 | -0.0378 | 0.013 | 0.0701 | 0.1244 | 0.1813 |
| qwen36-27b | 2026-09-16_v1 | sae | 4M | 512 | 0.1450 ± 0.0024 | 0.0892 ± 0.0016 | -0.0167 | -0.0041 | 0.0063 | 0.0159 | 0.0335 |
| qwen36-27b | 2026-09-16_v1 | random | 8M | 512 | 0.0596 ± 0.0003 | 0.0511 ± 0.0003 | 0.0017 | 0.014 | 0.0242 | 0.0323 | 0.0394 |
| qwen36-27b | 2026-09-16_v1 | realact | 8M | 512 | 0.3997 ± 0.0058 | 0.3078 ± 0.0054 | -0.0376 | 0.0131 | 0.0705 | 0.1249 | 0.1818 |
| qwen36-27b | 2026-09-16_v1 | sae | 8M | 512 | 0.1602 ± 0.0025 | 0.1057 ± 0.0018 | -0.0167 | -0.0041 | 0.0063 | 0.016 | 0.0339 |
| qwen36-27b | 2026-09-16_v1 | random | 16M | 512 | 0.0612 ± 0.0003 | 0.0528 ± 0.0003 | 0.0017 | 0.014 | 0.0242 | 0.0323 | 0.0394 |
| qwen36-27b | 2026-09-16_v1 | realact | 16M | 512 | 0.4105 ± 0.0058 | 0.3228 ± 0.0055 | -0.0377 | 0.0131 | 0.0702 | 0.1246 | 0.181 |
| qwen36-27b | 2026-09-16_v1 | sae | 16M | 512 | 0.1764 ± 0.0026 | 0.1236 ± 0.0019 | -0.0167 | -0.0041 | 0.0063 | 0.016 | 0.0338 |

### (e) 27B SAE repo windows

| base | sae | set | features | windows/feature | max_cos (the column) | mean_cos | frac fired | features ever firing | peak r (mean) | peak r (median) | argmax agree |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | qwen36-27b/l42-1b | 2026-09-16_v1 | 512 | 32 | 0.1673 ± 0.0026 | 0.1064 ± 0.0020 | 0.9378 | 0.9961 | 0.6118 | 0.6405 | 0.8261 |

**The headline is (b) and (c) together, and it is a bo artefact.** `rl-8x2048-full` beats
`rlI-150` at bo 1, 4 and 16 on every family, and on realact the ranking **flips at bo 64** — LoRA
0.5701 against full 0.5692. Paired per target that flip is +0.0009 ± 0.0013 with a win fraction of
0.527 and a sign-test p of **0.247**: it is not a difference. The per-rollout means, on the same
512 directions, go the other way and are not close — full is ahead by **0.0471 ± 0.0031**
(p ≈ 1e-40) on realact and by 0.0090 ± 0.0014 (p ≈ 7e-8) on sae best-of-64. So the LoRA does not
invert better; it is noisier, and a best-of-64 draw pays for variance. Checklist item 27 asked for
this table because rankings flip with bo — here is one flipping, on the paper's headline family,
between two checkpoints of the same model.

The SAE strata (checklist item 63) say the full model's advantage is **stratum-dependent and
largest where the features are rarest**: −0.0161 ± 0.0040 on best-of-64 in q0 (rarest quartile)
against −0.0050 ± 0.0018 in q2. A single pooled sae number hides a 3x spread.

**Baselines, 27B.** Corpus retrieval at 16M reaches realact **0.4105 ± 0.0058** against the MAEMMs'
0.5701 / 0.5692 — a gap of 0.16, where the 8B's was 0.093, and still climbing with corpus size
(0.3216 → 0.3433 → 0.3706 → 0.3997 → 0.4105 at 1/2/4/8/16M). On `sae` the picture is the 8B's
again: corpus top-1 0.1764 and the SAE repo's own shipped windows **0.1673 ± 0.0026** against the
MAEMMs' 0.1625 / 0.1714 — the inverter and both searches are inside each other's error bars. The
27B repo-vs-us activation agreement stays at r = 0.6118 (median 0.6405, argmax 82.6%), the known
disagreement that makes that file usable as text but not as an activation reference.

**Secondaries (h), 27B.** The 10x-median norm filter drops **0 of ~2.9M** kept tokens on both 27B
MAEMMs and moves nothing (checklist item 4's "empirically inert", reproduced at full scale).
Centring lifts realact best-of-64 to 0.8674 (LoRA) and 0.8645 (full) — +0.297 and +0.295, so the
two conventions differ by more than any effect in the paper and must never share a table.

**Argmax position (g), 27B.** The full model puts the argmax at the LAST token on **89.7%** of
realact rollouts against the LoRA's 60.1%, and on 36.6% vs 11.6% of sae rollouts, at similar kept
lengths (45.2 vs 55.0 tokens). Same base, same prompt, same scorer, same directions — the training
recipe alone moves the peak-position distribution by 30 points, which is exactly the
under-reporting checklist item 10 warns about.

### (k) per-feature sae — does the MAEMM beat search on the features it is tested on?

Built to check a claim rather than illustrate one: Celeste's note says "the 27B MAEMMs invert badly
on ~30% of SAE features (~40% worse than corpus search)". Rows are the same 512 features; the MAEMM
gets its full best-of-64 while each search gets its SINGLE best text. Primary first.

| base | maemm | role | slice | features | MAEMM bo64 | corpus top-1 | wins vs corpus | worse >0.05 abs | worse >40% rel | repo top | wins vs repo | worse >0.05 abs (repo) | worse >40% rel (repo) | own-feature act (max over rollouts) | frac firing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | all | 512 | 0.1714 | 0.1764 | 0.543 | 0.1387 | 0.1328 | 0.1673 | 0.627 | 0.1094 | 0.1152 | 22.736 | 0.9355 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | density q0 | 128 | 0.1464 | 0.1915 | 0.3906 | 0.3828 | 0.3984 | 0.1766 | 0.4688 | 0.3281 | 0.3438 | 18.235 | 0.7656 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | density q1 | 128 | 0.2216 | 0.197 | 0.7344 | 0.0234 | 0.0234 | 0.1946 | 0.7656 | 0.0234 | 0.0156 | 30.034 | 0.9922 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | density q2 | 128 | 0.1809 | 0.1755 | 0.5547 | 0.0469 | 0.0312 | 0.167 | 0.6797 | 0.0391 | 0.0312 | 24.52 | 0.9922 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | density q3 | 128 | 0.1368 | 0.1416 | 0.4922 | 0.1016 | 0.0781 | 0.131 | 0.5938 | 0.0469 | 0.0703 | 18.155 | 0.9922 |
| qwen3-8b | 2026-09-03_run1-rl | secondary | all | 512 | 0.1489 | 0.2282 | 0.123 | 0.5195 | 0.4082 | 0.2351 | 0.0806 | 0.5226 | 0.4224 |  |  |
| qwen3-8b | 2026-09-03_run1-rl | secondary | density q0 | 128 | 0.1318 | 0.2433 | 0.1094 | 0.5938 | 0.5547 | 0.2645 | 0.0238 | 0.6825 | 0.5873 |  |  |
| qwen3-8b | 2026-09-03_run1-rl | secondary | density q1 | 128 | 0.1717 | 0.2439 | 0.1953 | 0.4609 | 0.3594 | 0.2565 | 0.0551 | 0.4646 | 0.378 |  |  |
| qwen3-8b | 2026-09-03_run1-rl | secondary | density q2 | 128 | 0.1487 | 0.2191 | 0.0938 | 0.4531 | 0.3828 | 0.2257 | 0.0859 | 0.5078 | 0.3906 |  |  |
| qwen3-8b | 2026-09-03_run1-rl | secondary | density q3 | 128 | 0.1433 | 0.2063 | 0.0938 | 0.5703 | 0.3359 | 0.1943 | 0.1562 | 0.4375 | 0.3359 |  |  |
| qwen36-27b | 2026-09-08_rlI-150@vllm | secondary | all | 512 | 0.1625 | 0.1764 | 0.4902 | 0.1797 | 0.1738 | 0.1673 | 0.5703 | 0.1504 | 0.1465 |  |  |
| qwen36-27b | 2026-09-08_rlI-150@vllm | secondary | density q0 | 128 | 0.1303 | 0.1915 | 0.3438 | 0.4766 | 0.4766 | 0.1766 | 0.375 | 0.4219 | 0.4219 |  |  |
| qwen36-27b | 2026-09-08_rlI-150@vllm | secondary | density q1 | 128 | 0.2141 | 0.197 | 0.6641 | 0.0391 | 0.0312 | 0.1946 | 0.7109 | 0.0547 | 0.0312 |  |  |
| qwen36-27b | 2026-09-08_rlI-150@vllm | secondary | density q2 | 128 | 0.1759 | 0.1755 | 0.5469 | 0.0625 | 0.0469 | 0.167 | 0.6797 | 0.0469 | 0.0312 |  |  |
| qwen36-27b | 2026-09-08_rlI-150@vllm | secondary | density q3 | 128 | 0.1296 | 0.1416 | 0.4062 | 0.1406 | 0.1406 | 0.131 | 0.5156 | 0.0781 | 0.1016 |  |  |

### (l) deciles of the per-feature difference (MAEMM best-of-64 − corpus top-1)

| base | maemm | role | features | mean diff | SE | frac > 0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | 512 | -0.0049 | 0.0029 | 0.543 | -0.0734 | -0.0318 | -0.014 | -0.0036 | 0.0041 | 0.0123 | 0.0223 | 0.0361 | 0.0589 |
| qwen3-8b | 2026-09-03_run1-rl | secondary | 512 | -0.0793 | 0.0039 | 0.123 | -0.207 | -0.1485 | -0.1056 | -0.0737 | -0.0537 | -0.0336 | -0.0218 | -0.0097 | 0.0035 |
| qwen36-27b | 2026-09-08_rlI-150@vllm | secondary | 512 | -0.0139 | 0.0031 | 0.4902 | -0.1001 | -0.0451 | -0.0225 | -0.0092 | -0.0012 | 0.0069 | 0.0159 | 0.0319 | 0.0527 |

**The claim is off by a factor of three pooled, and right in the rarest quartile.** On the primary
(`rl-8x2048-full`, 512 features) the MAEMM is >40% worse than corpus search on **13.3%** of
features, not ~30%, and it WINS outright on 54.3%; the per-feature difference is −0.0049 ± 0.0029
with 54.3% above zero, i.e. the inverter and the corpus search are level on average. But the split
by density quartile is where the claim lives: in **q0, the rarest quartile, 39.8%** of features are
>40% worse, the MAEMM wins only 39.1%, and its own-feature firing rate drops from 93.6% to **76.6%**
— on rare features the inverter frequently produces text that does not make the feature fire at all.
The secondary rlI-150 is at 17.4% pooled / 47.7% in q0, and the 8B run1 at **40.8% pooled** (winning
only 12.3% of features against corpus search, with a mean difference of −0.0793 ± 0.0039 and eight
of nine deciles negative).

So the "~30%" is not the 27B's pooled number under any reading. It matches either the 8B's pooled
40.8%, or the 27B's rarest-quartile 39.8% — both plausible origins for the figure, and the
correction to carry into the paper is that the pooled 27B number is **13.3%**, with the failure
concentrated in rare features. The (l) deciles say the same thing in a different shape: the
primary's distribution is tight around zero (d4 −0.0036, d5 +0.0041) with a long left tail
(d1 −0.0734), while the 8B's is shifted bodily negative (d5 −0.0537).

A caveat that limits the comparison rather than the finding: `worse >40% relative` is unstable where
the baseline is small, since it divides by the corpus top-1. The absolute column is there for that
reason and agrees — 13.9% of the primary's features are >0.05 behind, against 38.3% in q0.

## Patchscopes 27B (full root)

The zero-shot patching baseline as a product, on `/vol`. Cap $10; **actual $7.87**.

| date | item | command (abbreviated) | wall | cost | result |
|---|---|---|---|---|---|
| 2026-09-16 | sweep: floor + L8/L14/L21 | `--product patchscopes --base qwen36-27b --set 2026-09-16_v1 --rows "0-63,1024-1087" --n 8 --ps-layers "8,14,21"` | 608.2 s | **$0.7670** | 4 cells, 3,072 injected + 8 floor generations; patch check cos(h_patched, v) = 1.0000 on every layer; 359-394 gen tok/s |
| 2026-09-16 | sweep scoring, 4 cells | `--product score --rollouts-dir <cell>` x4 | 96.8 / 117.0 / 125.1 / 91.4 s | $0.5426 | 1,024 rows each |
| 2026-09-16 | L14 scoring, FIRST attempt | same | ~120 s | ~$0.13 (no cost line -- it raised before the summary) | **AssertionError, checklist item 8** -- see below |
| 2026-09-16 | final cell: floor + L14 at bo 32 | `--rows "0-511,1024-1151" --n 32 --ps-layers 14 --ps-tag bo32` | 4,634.5 s | **$5.8446** | 512 realact + 128 sae x bo 32 = 20,480 rows per cell; 291 gen tok/s |
| 2026-09-16 | final scoring, 2 cells | `--product score --rollouts-dir <cell>` x2 | 229.4 / 235.2 s | $0.5860 | 20,480 rows each at **115 rows/s** |

### (j) the Patchscopes table

| base | set | cell | family | dirs | bo | mean cos (bo1) | cell best-of-bo | 2026-09-10_rl-8x2048-full@vllm @cell bo | 2026-09-10_rl-8x2048-full@vllm @bo64 | 2026-09-08_rlI-150@vllm @cell bo | 2026-09-08_rlI-150@vllm @bo64 | lift over floor | beats its floor | sign-test p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36-27b | 2026-09-16_v1 | floor | realact | 64 | 8 | 0.0376 ± 0.0103 | 0.0770 ± 0.0107 | 0.5277 | 0.5501 | 0.5232 | 0.5507 |  |  |  |
| qwen36-27b | 2026-09-16_v1 | floor | sae | 64 | 8 | 0.0094 ± 0.0008 | 0.0169 ± 0.0010 | 0.1245 | 0.1433 | 0.1037 | 0.1222 |  |  |  |
| qwen36-27b | 2026-09-16_v1 | p2-L14-replace2 | realact | 64 | 8 | 0.0521 ± 0.0118 | 0.1149 ± 0.0137 | 0.5277 | 0.5501 | 0.5232 | 0.5507 | 0.0380 ± 0.0068 | 0.812 | 4.57e-07 |
| qwen36-27b | 2026-09-16_v1 | p2-L14-replace2 | sae | 64 | 8 | 0.0080 ± 0.0008 | 0.0163 ± 0.0010 | 0.1245 | 0.1433 | 0.1037 | 0.1222 | -0.0006 ± 0.0006 | 0.438 | 0.382 |
| qwen36-27b | 2026-09-16_v1 | p2-L21-replace2 | realact | 64 | 8 | 0.0479 ± 0.0111 | 0.1080 ± 0.0131 | 0.5277 | 0.5501 | 0.5232 | 0.5507 | 0.0310 ± 0.0054 | 0.844 | 2e-08 |
| qwen36-27b | 2026-09-16_v1 | p2-L21-replace2 | sae | 64 | 8 | 0.0084 ± 0.0008 | 0.0159 ± 0.0010 | 0.1245 | 0.1433 | 0.1037 | 0.1222 | -0.0010 ± 0.0006 | 0.422 | 0.26 |
| qwen36-27b | 2026-09-16_v1 | p2-L8-replace2 | realact | 64 | 8 | 0.0505 ± 0.0115 | 0.1110 ± 0.0128 | 0.5277 | 0.5501 | 0.5232 | 0.5507 | 0.0340 ± 0.0065 | 0.734 | 0.000227 |
| qwen36-27b | 2026-09-16_v1 | p2-L8-replace2 | sae | 64 | 8 | 0.0088 ± 0.0009 | 0.0168 ± 0.0009 | 0.1245 | 0.1433 | 0.1037 | 0.1222 | -0.0001 ± 0.0006 | 0.500 | 1 |
| qwen36-27b | 2026-09-16_v1 | floor__bo32 | realact | 512 | 32 | 0.0542 ± 0.0041 | 0.1247 ± 0.0041 | 0.563 | 0.5692 | 0.5626 | 0.5701 |  |  |  |
| qwen36-27b | 2026-09-16_v1 | floor__bo32 | sae | 128 | 32 | 0.0082 ± 0.0005 | 0.0205 ± 0.0007 | 0.1411 | 0.1464 | 0.1241 | 0.1303 |  |  |  |
| qwen36-27b | 2026-09-16_v1 | p2-L14-replace2__bo32 | realact | 512 | 32 | 0.0718 ± 0.0046 | 0.1706 ± 0.0056 | 0.563 | 0.5692 | 0.5626 | 0.5701 | 0.0460 ± 0.0030 | 0.753 | 1.63e-31 |
| qwen36-27b | 2026-09-16_v1 | p2-L14-replace2__bo32 | sae | 128 | 32 | 0.0087 ± 0.0007 | 0.0240 ± 0.0020 | 0.1411 | 0.1464 | 0.1241 | 0.1303 | 0.0036 ± 0.0018 | 0.516 | 0.791 |

**The floor is a third of the baseline's number, and the baseline is a third of the inverter's.**
At the final cell's own bo 32, on 512 realact directions: the no-injection floor reaches **0.1247 ±
0.0041**, the injected cell **0.1706 ± 0.0056**, and the primary MAEMM on the same rows at the same
bo **0.5630**. So of the inverter's 0.563, about **0.125 is available to fluent English with no
direction anywhere in the forward pass**, a further **0.046** to a zero-shot patch, and the
remaining ~0.39 is what training buys. The lift over the matched floor is +0.0460 ± 0.0030 with
**75.3%** of directions beating their own floor (sign test p = 1.6e-31) -- small, but real and
consistent, exactly as the 8B trial found (+0.087, 80.5%).

**On sae the patch does nothing, now measured properly.** Lift +0.0036 ± 0.0018, **51.6%** of
features beating their floor, sign test **p = 0.79** -- a coin flip, at 128 features and bo 32
rather than the sweep's 64 at bo 8. The 8B trial reported the same (51.4%, +0.002). Two bases, two
shapes, one conclusion: a zero-shot patch carries information about real-activation directions and
none about SAE encoder columns.

**Depth transfers across scale.** The sweep ranked L14 > L8 > L21 (lift over floor at bo 8: +0.0380,
+0.0340, +0.0310 on realact). L14 is **21.9%** relative depth on the 27B's 64 layers; the 8B trial's
winner was layer 8 of 36 = **22.2%**. The optimum sits at the same relative depth on both models,
which is what the depth-only sweep was for.

### Two failures worth keeping

**The checklist item 8 assert fired on a product it cannot apply to.** `score.py` aborted the L14
cell on "a scored row kept 95 tokens, i.e. it hit the 95-token truncation; the ~103-token inverter
prompt looks exactly like this". But the Patchscopes prompt is **30 tokens**: prompt + max_new =
94 < 95, so a row carrying the ENTIRE prompt could not reach the truncation, and the check could not
have been detecting a leak. The exact guard -- stored generated ids <= max_new -- passed on every
row. The check now tests its own applicability (`prompt_tokens + max_new >= SCORE_MAX_LENGTH`) and
where it cannot apply it REPORTS instead: **2 of 1,024 rows (0.20%)** hit the truncation through
re-tokenization expansion, and their cosine is a max over a shortened window, biased down. The
MAEMM path's assert is untouched.

**Per-cell cost was mis-reported until the final run.** `common.outdir` takes `t0` from the
container args, which is right for a product that writes ONE directory and wrong for this one, which
writes several: the sweep's floor README says 86.6 s / $0.1092 and L8's says 270.3 s / $0.3409
although L8's own generation was 183 s -- the second is the container's total to that point. Fixed
by passing a per-cell `t0`; the model load is charged to the first cell, where it is paid. **The
sweep's four cell READMEs carry the inflated figures**; the marginal costs are their differences,
and the `[wall]` line ($0.7670) is the sweep's true total.

### Cost projections, and how wrong they were

Worth recording because both errors were large and in opposite directions. Generation was priced at
the 27B's measured **239.3** tok/s (`rollouts_hf`, LoRA); the clean base with no adapter ran at
**291-394**, so generation came in cheaper than projected. Scoring was first priced from the 8B's
531 rows/s (7x too cheap), then re-priced at **23.3 rows/s** measured on a 1,024-row job -- but that
sample is dominated by the 27B clean-base load, and at 20,480 rows the real rate is **115 rows/s**,
so the second estimate was 4x too expensive. The lesson for the next cost plan: measure throughput
at the SHAPE you will run, not at a smoke's.

### (f) GCG — SMOKE ROOT ONLY, not run on `/vol`

**Gap, stated rather than left blank.** `stats.py --root-tag full` prints "(f) skipped: no gcg arms
with finals.jsonl on this root", and that is correct: `base/<base>/gcg/` does not exist on `/vol` at
all. Every GCG arm to date lives on `/vol/runs/2026-09-15_paper-evals-smoke`, on the **8B**, on the
first 8 `realact` rows. There is no 27B GCG and no full-root GCG, so the reachability ceiling has
NOT been measured against the primary MAEMM. These are the smoke rows, for the record:

| base | family | arm | dirs | members/dir | rows | final cos | init cos | nll |
|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact | realact/epo-corpus | 8 | 3.0 | 0-7 | 0.5760 ± 0.0073 | 0.4919 ± 0.0227 | 3.0222 |
| qwen3-8b | realact | realact/epo-random32 | 8 | 3.0 | 0-7 | 0.4168 ± 0.0517 | -0.0165 ± 0.0357 | 4.7934 |
| qwen3-8b | realact | realact/gcg-corpus | 8 | 1.0 | 0-7 | 0.6590 ± 0.0199 | 0.4919 ± 0.0227 | 8.0643 |
| qwen3-8b | realact | realact/gcg-random32 | 8 | 1.0 | 0-7 | 0.4620 ± 0.0540 | -0.0163 ± 0.0357 | 13.593 |

The MAEMM column is absent because the 8B's GCG set (`2026-09-16_v1`) and its smoke-root scores
(the imported `2026-09-03_run1-archive16`) do not overlap — the join is empty and stats.py drops the
column rather than inventing one. Reading these against the full-root 8B numbers by eye:
`gcg-corpus` reaches **0.6590 ± 0.0199** on 8 realact directions, against that MAEMM's 0.6195
best-of-64 over 512 directions — i.e. a 32-token string optimised directly on the metric beats the
inverter, which is the ceiling the column exists to establish. Treat it as indicative: 8 directions,
a different target set, and no 27B equivalent.

## `autointerp` -- the Delphi-style SAE autointerp evaluation (2026-09-16)

`paper-evals/autointerp/`. Design `infra/2026-09-16_autointerp-design.md` **including its §9
amendments A1-A12**, which override §2-§6. Four GPU products, one CPU build, one LLM run. GPU costs
are read off each product's own README on the volume; LLM costs are COMPUTED from the returned
token counts at the rate table in `costs.json` (the Anthropic Messages API returns no cost field).

### GPU products (H200, $4.54/h)

| # | product | scope | wall | cost |
|---|---|---|---|---|
| 1 | `sae_self` shakeout, 2 sae targets x 64 | rows 1024-1025 | 107.6 s | $0.1357 |
| 2 | `sae_self` `rl-8x2048-full`, FAILED the argmax check | 512 x 64 | ~216 s | $0.2719 (sunk) |
| 3 | `sae_self` `rlI-150`, FAILED the same check | 512 x 64 | ~190 s | $0.2394 (sunk) |
| 4 | `sae_self` `rl-8x2048-full` | 512 x 64 = 32,768 rows | 215.5 s | **$0.2718** |
| 5 | `sae_self` `rlI-150` | 512 x 64 = 32,768 rows | 237.2 s | **$0.2991** |
| 6 | `random_pool` | 2,048 windows x 512 features | 111.8 s | **$0.1410** |
| 7 | `examples_4m` (A3) | 4,738 docs, 3,999,724 tokens, 237,980 windows | 1293.4 s | **$1.6311** |
| 8 | `examples_docmax` | 18,813 docs, 16M tokens, 952,388 windows, top 256 docs/feature | 4941.6 s | **$6.2319** |

Items 2 and 3 are the same work as 4 and 5 and are listed because they were paid for: the
`sae_self` argmax check was written as an EQUALITY against the stored `argmax.i16` and failed on
1 of 32,768 rollouts (primary) and 3 of 32,768 (`rlI-150`). Both are genuine near-ties -- worst
cosine gap 4.0e-6 -- and the cause is checklist item 11 showing up exactly where it was predicted:
`score` batched all 1,536 targets of the set while this pass batches only the 512 `sae` rows, so
the two runs' `SCORE_CHUNK` right-padding widths differ and move a per-token cosine by ~1e-4. The
check now allows a mismatch only below a 1e-3 near-tie tolerance, two orders of magnitude clear on
both sides, and reports the count and the worst gap either way.

`sae_self`'s other two checks are exact and passed on both MAEMMs: our activation at the stored
argmax against the CSR's value (0 mismatches, worst |diff| 0.0157, which is f16 quantisation at
act ~40) and against its membership (0 mismatches). Mean fire fraction 0.8687 (primary) / 0.8299
(`rlI-150`).

### Why `random_pool` and `examples_docmax` exist

Both replace a stored product that cannot serve the amended design, and both were built only after
the shortfall was MEASURED.

- `random_pool` (A5): `scan`'s `_random256` carries 256 windows and a per-feature MAXIMUM only. The
  densest tested feature has **5** zero-activation windows in 2,048, so 256 could never supply the
  20 the design asks for. The new pool is 2,048 windows encoded for all 512 features with per-token
  pre-gate activations, stored as a CSR at density 0.0066 (11.8 MiB against a 134 MiB dense
  equivalent).
- `examples_4m` (A3): filtering the 16M top-128 down to the 4M prefix left a median of 14
  candidates after dedup and fewer than 16 on **38 of 64** pilot features -- C4 was not an N=16
  arm. Its own 4M scan gives a median of 54 and fewer than 16 on **0 of 64**.
- `examples_docmax`: with A1 (gate-consistent positives) and A4 (document-level disjointness) in
  force, draw 1 reached its 20 positives on **35 of 64** features and draw 2 was **empty on 21**.
  A feature's gate-passing windows are the top-128 (median 60 after dedup) plus almost no q-band
  rows -- those bands are equal-width bins of (0, max_act] and sample the weak tail -- and the arms
  show 16-32 of them across a median of 28 documents, after which A4 removes every other window in
  those documents. Ranking WINDOWS concentrates them in few documents, so this product ranks
  DOCUMENTS: the best window of each of a feature's top 256 documents.

### The LLM stage: Anthropic Messages API, not OpenRouter (Tomas, 2026-09-16)

Model `claude-sonnet-5`, SDK `anthropic==1.6.0`, Modal secret `anthropic`. **MEASURED: there is no
`temperature` parameter.** `messages.create()` raises `TypeError: unexpected keyword argument
'temperature'` -- sampling parameters were removed from the API for this model generation.
OpenRouter accepted the parameter, which is what hid it. The design's "temperature 0" is therefore
not achievable on this surface; run-to-run variation is real, and the two null arms (`C16-judge2`
for the judge half, `C16-draw2` for judge + test-set draw) are the only noise floors the evaluation
has. `thinking: {"type": "disabled"}` is accepted and is sent on every call.

**Prompt caching does not engage**, MEASURED: `cache_creation_input_tokens` and
`cache_read_input_tokens` are both 0. The stable prefix (system + Delphi's three verbatim few-shot
turns) is ~900 tokens and Sonnet 5's minimum cacheable prefix is 1024. Nothing was added to the
prompts to reach it -- they are Delphi's, verbatim.

Measured per-call cost on the first smoke (2 features, 9 arms, both scorers, 258 calls, $0.7364,
sync path): explain **$0.0067**/call at 2,392 input tokens, detection **$0.00302**/call at 1,478,
fuzzing **$0.00194**/call at 892. Each stage's pre-submission projection (`messages.count_tokens`
on a sample, times the rate table) landed within 9% of the actual. A10 earned its place on that
smoke: **3 of 18** explainer answers hit the lifted 300-token cap and were retried at 600, so the
default is now 600.

## Untrained-base control (2026-09-16)

The control the primary MAEMM is read against: a CLEAN `Qwen/Qwen3.6-27B` with **no MAEMM weights
of any kind**, pushed through the identical generation path as
`qwen36-27b/2026-09-10_rl-8x2048-full` -- same prompt function (`celeste27b`, 103 tokens), same
marker, same block-1 norm-matched injection at coef 1.0, same `rollouts:` constants (T 1.0, n 64,
max_new 64, min_new 16, seed 1234), same held-out set `2026-09-16_v1` (1,536 targets), same
clean-base scorer at layer 42. The only difference is the weights, which is what makes the gap in
the tables attributable to training. It lands as an ordinary product at
`/vol/maemms/qwen36-27b/2026-09-16_base-control/`.

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-16 | `check`, 27B, with the new `type: base` entry (CPU) | `--product check --base qwen36-27b` | 13.3 s | ~$0 | `2026-09-16_base-control (base)` resolves to the base snapshot `models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd…` (51.8 GiB); every other entry unchanged | — |
| 2026-09-16 | `rollouts_vllm` control, smoke root, rows 0-7 x 64 | `--product rollouts_vllm --base qwen36-27b --maemm qwen36-27b/2026-09-16_base-control --set 2026-09-16_v1 --rows 0-7 --n 64 --root …smoke` (`ap-Mn4Pcd4u2nYjEESpJ95PXF`) | 271.8 s | **$0.3428** | 512 rollouts; engine up 223 s; marker \|\|h\|\| **14.0594** vs the config'd clean base **14.0620** (0.019%); injection cos **0.999995**, ratio **1.000124**, pre-marker delta **0.0** | the marker-norm check is INVERTED for this kind and had to be written (below); nothing else fired |
| 2026-09-16 | `score` control, smoke root, rows 0-7 | `--product score … --rows 0-7 --engine vllm --root …smoke` (`ap-3V1bmhhvaxZGgCX2IDBxlq`) | 107.6 s | **$0.1357** | 8 realact rows: mean cos **0.0074**, best-of-64 **0.1096** | against the primary's 0.4744 / 0.5408 on the same 8 rows — the control is at 1.5% / 20% of it |
| 2026-09-16 | `rollouts_vllm` control, FULL root, 1,536 x 64 | `--product rollouts_vllm … --set 2026-09-16_v1 --max-num-seqs 256 --detach` (`ap-T54L7ezITTXzLXq5EZtr1g`) | 3,210.5 s | **$4.0487** | 98,304 rollouts, generate 2,963.9 s, **905.7 gen tok/s = 33.17 rollouts/s**; mean 27.3 tokens, eos rate 0.9193; all four self-checks passed | 33.2 rollouts/s against the primary's 29.32 — the clean base emits shorter rollouts (27.3 vs the primary's longer ones) and has no adapter, so it is the fastest 27B run yet. Cost came in UNDER the $4.5 projection |
| 2026-09-16 | `score` control, FULL root — **FAILED** | `--product score … --engine vllm --detach` (`ap-O3lO3VTGiBfVWy63xRq8VU`) | 710 s (app wall) | **~$0.90** | aborted at the very end on checklist item 8, bound 2: "a scored row kept 95 tokens" | **a real finding about the check, not about the data** — see below. One row of 98,304 |
| 2026-09-16 | `score` control, FULL root (retry) | same (`ap-yGjjMqWb0JuTVAjrt5H4xr`) | 611.7 s | **$0.7714** | 98,304 rows in 601.9 s (**163 rows/s**), 1.0 GiB; bound 1 (stored ids <= max_new 64) passed on every row; **1 of 98,304** rows at the truncation | 163 rows/s against the 115 measured on the 20k-row Patchscopes cell — the rate keeps rising with the job size |
| 2026-09-16 | `centred` control (CPU) | `--product centred … --engine vllm` | 134.0 s | ~$0 | the two secondary cosines added in place, so the control appears in table (h) | not in the brief; run because (h) is one of the regenerated tables and it costs nothing |
| 2026-09-16 | `reconstruction/stats.py --root-tag full` | local | ~3 min | $0 | 12 tables, 4 scores directories (primary, control, 2 secondaries), 20.5 MB fetched | the primary's own numbers are bit-identical to the previous full-root run — the control adds rows, it moves nothing |

### What the engine actually served

`marker ||h|| engine 14.0594 vs clean base 14.0620`, `rel_diff 0.000187`, and the injection check
`{cos: 0.999995, norm_ratio: 1.000124, max_pre_marker_delta: 0.0, clean_vs_clean_pre_marker_delta:
0.0}` — identical on the smoke and the full run. Two things are worth separating there:

- **The marker norm proves no MAEMM was served.** 14.06 is the clean base. The trained 27B
  checkpoints sit at **130** (full-parameter) and **512** (LoRA) at the same position, i.e. 9x and
  36x higher, which is why the loose 25% sanity bound on this kind cannot fire on numerical noise.
- **The injection is real.** `cos 0.999995` and `ratio 1.000124` say the steered forward differs
  from the un-injected one by exactly `unit(v) * ||h|| * 1.0` at the marker, and the pre-marker
  delta is **exactly 0.0** (the full-model path has no Punica nondeterminism). So the control is
  not "a base model with nothing done to it" — it is a base model receiving the same direction the
  inverter receives, and failing to say anything about it.

### (a) the control beside the primary, 512 targets per family, full root

| base | maemm | role | family | targets | mean cos (bo1) | bo64 |
|---|---|---|---|---|---|---|
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | random | 512 | 0.0280 ± 0.0005 | 0.0428 ± 0.0005 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | realact | 512 | 0.4994 ± 0.0070 | 0.5692 ± 0.0063 |
| qwen36-27b | 2026-09-10_rl-8x2048-full@vllm | primary | sae | 512 | 0.1298 ± 0.0032 | 0.1714 ± 0.0035 |
| qwen36-27b | 2026-09-16_base-control@vllm | **control** | random | 512 | 0.0160 ± 0.0005 | 0.0323 ± 0.0004 |
| qwen36-27b | 2026-09-16_base-control@vllm | **control** | realact | 512 | 0.0244 ± 0.0052 | 0.1364 ± 0.0058 |
| qwen36-27b | 2026-09-16_base-control@vllm | **control** | sae | 512 | 0.0022 ± 0.0003 | 0.0241 ± 0.0012 |

### (c) primary minus control, paired per target

| family | slice | statistic | targets | A (primary) | B (control) | diff (A-B) | A wins | sign-test p |
|---|---|---|---|---|---|---|---|---|
| random | all | best-of-64 (unbiased) | 512 | 0.0428 | 0.0323 | 0.0105 ± 0.0003 | 0.924 | 8.28e-96 |
| random | all | mean cos | 512 | 0.028 | 0.016 | 0.0120 ± 0.0002 | 0.996 | 1.96e-149 |
| realact | all | best-of-64 (unbiased) | 512 | 0.5692 | 0.1364 | **0.4328 ± 0.0053** | 1.000 | 1.49e-154 |
| realact | all | mean cos | 512 | 0.4994 | 0.0244 | **0.4750 ± 0.0050** | 1.000 | 1.49e-154 |
| sae | all | best-of-64 (unbiased) | 512 | 0.1714 | 0.0241 | **0.1474 ± 0.0038** | 0.977 | 1.78e-130 |
| sae | all | mean cos | 512 | 0.1298 | 0.0022 | 0.1276 ± 0.0032 | 0.988 | 3.67e-141 |
| sae | density q0 | best-of-64 (unbiased) | 128 | 0.1464 | 0.0219 | 0.1245 ± 0.0089 | 0.976 | 4.01e-33 |
| sae | density q0 | mean cos | 128 | 0.0985 | 0.0036 | 0.0949 ± 0.0077 | 0.977 | 2.05e-33 |
| sae | density q1 | best-of-64 (unbiased) | 128 | 0.2216 | 0.0282 | 0.1934 ± 0.0066 | 0.992 | 7.58e-37 |
| sae | density q1 | mean cos | 128 | 0.1766 | 0.0036 | 0.1731 ± 0.0056 | 0.992 | 7.58e-37 |
| sae | density q2 | best-of-64 (unbiased) | 128 | 0.1809 | 0.0251 | 0.1558 ± 0.0068 | 0.969 | 6.48e-32 |
| sae | density q2 | mean cos | 128 | 0.1429 | 0.0019 | 0.1410 ± 0.0049 | 0.992 | 7.58e-37 |
| sae | density q3 | best-of-64 (unbiased) | 128 | 0.1368 | 0.0211 | 0.1157 ± 0.0056 | 0.969 | 6.48e-32 |
| sae | density q3 | mean cos | 128 | 0.1012 | -0.0001 | 0.1013 ± 0.0047 | 0.992 | 7.58e-37 |

**The primary beats the control on 512 of 512 realact directions** — not 511, 512 — and on 97.7% of
SAE features. The interesting number is the one that is NOT near zero: the control reaches
**0.1364** best-of-64 on realact from a mean-of-64 of **0.0244**, i.e. essentially all of it is
sampling luck over 64 draws. Table (j) now puts it against the Patchscopes floor at the SAME bo,
which is the comparison that matters: at bo 32 on the same 512 realact directions the control
reaches **0.1212** and the no-injection floor reaches **0.1247 ± 0.0041**. The control is at or
marginally BELOW the floor — so injecting a direction into a clean base buys nothing at all over
generating fluent English with no direction anywhere in the forward pass, and the whole 0.43 gap to
the primary is training. On `random` the control is at 0.0323 against
the primary's 0.0428, both of which are the metric's own noise floor.

`sae` density q3 (the DENSEST quartile) has the control at a mean cosine of **-0.0001** — the
clean base is exactly uninformative there, while the primary reaches 0.1012.

### Code: `type: base` and what it inverts

Six files, all small. The one that carries judgement is the marker-norm self-check.

- `config.yaml`: `qwen36-27b/2026-09-16_base-control` with `type: base`, `hf: Qwen/Qwen3.6-27B`
  (the base's OWN repo), `role: control`, no `train_max_new` (nothing was trained; every consumer
  reads it with `.get`).
- `common.load_config`: `type` accepts `base`; a `type: base` entry must name the base's own `hf`
  (anything else is a trained checkpoint wearing the control's label) and `role`, where given, must
  be `control`. `common.load_maemm` returns `spec["type"]` as the kind instead of the literal
  `"full"`, so every caller can tell the control from a trained full-parameter MAEMM — their
  marker-norm expectations are OPPOSITE. `maemm_weights_path` needed nothing: `type != lora` already
  looks for `config.json`, and for the control it resolves to the base snapshot, which is the point.
- `rollouts_vllm.marker_norm_vs_hf`: for `kind == "base"` neither "must differ" form runs. The
  engine's norm is compared with `bases.<base>.marker_norm_base` with `expected: "equal:
  untrained-base control, no adapter"` recorded in the summary, and the only assert left is a 25%
  sanity bound (`BASE_CONTROL_NORM_TOL`) whose message says what it catches: a MAEMM served where
  the control was asked for, which would be 9-36x off. `_engine_for` needed nothing — `type != lora`
  already takes the served-model path, confirmed by `lora=False` in the engine line.
- `rollouts_hf.marker_check`: the same inversion on the HF path, which this control does not use but
  which would otherwise fire its "must differ" assert on the next person who tries.
  `rollouts_hf.write_maemm_readme` prints `role`, `note` and a "THIS IS THE UNTRAINED-BASE CONTROL"
  section for `type: base`. `weight_identity` needed nothing (index+sizes of the base snapshot).
- `reconstruction/stats.py`: `role` is now `primary` / `control` / `secondary` and every table sorts
  in that order. `table_c` pairs the primary against **every** other side on the same (base, set)
  rather than just `group[1]`, so "primary minus control" is a row rather than a subtraction the
  reader does by eye; its columns became fixed (`A`, `B`, `B role`, `A value`, `B value`, …) because
  a per-side header is not a schema when the number of sides varies. `c_*.csv` is not consumed by
  the paper scripts (`paper/inversion-eval/data/README-data.md`: only `{a,d,e,g,i,j,k,l}`), so the
  change stops there; (a) keeps its schema and only gains rows.
- A grep of every `["type"]` / `"full"` / `"lora"` / `kind` branch in `precompute/*.py` and
  `reconstruction/*.py` turned up nothing else that misbehaves: `score.py` never reads the spec,
  `modal_app.py`'s `check` only prints `spec['type']`, `centred.py` and `patchscopes.py` have no
  such branch.

### The failure worth keeping: checklist item 8's bound 2 was zero-tolerance

`score` aborted the whole 98,304-row job, after the full forward pass, on **one row**. The assert
was `worst < SCORE_MAX_LENGTH` — no scored row may reach the 95-token truncation, because the
103-token inverter prompt would look exactly like that. Bound 1, the exact one (stored generated ids
<= `max_new` = 64), passed on every row, so nothing had leaked; what happened is re-tokenization
expansion. The offending text is combining-diacritic unicode — the first round-trip mismatch the
same run reports is `' đ̛̣̈̀ ̓̋̐̀̈'` — which an untrained base emits and a trained inverter does not,
so this had never fired on the MAEMM path.

The fix is to make bound 2 a RATE bound, because the two things that reach the truncation have
opposite shapes: a prompt leak puts **~100%** of rows over it (the prompt alone exceeds the window),
expansion puts a handful over. `score.TRUNC_FRAC_MAX = 0.05` is 20x below the leak signature, the
count is reported in the product README whenever it is non-zero (those rows' cosine is a max over a
shortened window and is biased DOWN), and the previous 2026-09-16 Patchscopes fix — the
applicability test for a prompt too short to reach the truncation — is unchanged underneath it.
MEASURED on this run: **1 of 98,304 rows (0.00%)**. The new bound was checked in both directions
before the relaunch: green at 0.7% and 4.9% of rows truncated, red at 5.1% and red on a simulated
leak at 100%.

### Spend

$0.3428 + $0.1357 (smoke) + $4.0487 (rollouts) + ~$0.90 (the aborted score) + $0.7714 (the score) =
**$6.20** against a $12 cap and a $6-7 expectation. The aborted score is 15% of it and bought the
bound-2 finding above.

## GCG final (full root, 2026-09-16)

`gcg` only, no EPO. 2 bases x 2 families x 2 inits = 8 arms, rows 0-31 of `realact` and rows 0-31 of
`sae` (global 1024-1055) of `2026-09-16_v1`, on the FULL root, corpus init from the 16M scan
(`base/<base>/scan/2026-09-16_v1/topk.jsonl`, 952,388 windows, sizes 1-16). Config as measured
earlier: 512 children x 150 iterations, `--topk 512 --seq-len 32 --tau 0.02 --sbatch 256`,
oversample 1.5. Outputs at `base/<base>/gcg/2026-09-16_v1/<family>/<arm>/`. Each arm is one detached
call from its own persistent client (`nohup setsid`, never under `timeout` -- a killed client
cancels the in-flight call), staggered 25 s because the tree is being committed concurrently and
`add_local_dir` hashes it.

### The one arm that died, and what it found: the M=1 GEMV path

8B `realact/gcg-random32` stopped at direction 17 of 32 on the end-of-direction CHECK:

```
AssertionError: realact:17: the loop's own cos does not reproduce a fresh common.score_ids call
(max |d| 1.03e-02 > 1e-02) on member 0: loop 0.555820 vs fresh 0.545487 at position 28,
top1-top2 per-token gap 2.30e-01, residual norm there 389.601
```

The guard was NOT loosened. Instead the assert was given the evidence needed to classify the
failure, and the string was swept across batch shapes (only ever run when the check has already
failed, so it costs nothing in the normal path):

| rows in the batch | cos | argmax |
|---|---|---|
| 1 | **0.545487** | 28 |
| 8 | 0.555820 | 28 |
| 32 | 0.555820 | 28 |
| 128 | 0.555820 | 28 |
| 256 | 0.555820 | 28 |
| 512 | 0.555820 | 28 |

**It is not a continuum of noise; it is one discrete step between a 1-row batch and any batched
one.** Every shape from 8 to 512 is BIT-IDENTICAL and the argmax never moves; only `n = 1` differs,
by 1.03e-02. Both benign explanations are excluded by the same message: the top1-top2 per-token gap
is 2.30e-01, 22x the delta, so no argmax flip; and the residual norm at that position is 389.6, so
it is not a small-denominator blow-up. The mechanism is a kernel switch -- an M=1 matmul takes the
GEMV path and a batched one takes a tiled GEMM, with a different accumulation order in bf16.

Three things follow, and they retro-explain the whole CHECK block:

- **The loop's number is the one the pipeline agrees with.** `score.py` scores at
  `SCORE_CHUNK = 32`, a batched shape, so every stored score is on the n >= 8 side of the step. The
  1-row rescore inside the CHECK is the outlier, not the loop.
- **For a `pop = 1` arm the CHECK's two calls are the same call.** `fresh` (at `--sbatch` 256) and
  `rebatch` (at `SCORE_CHUNK` 32) both score ONE row, so both take the GEMV path and
  `d_same == d_re` identically -- which is exactly what every arm ever run has printed. One of the
  two asserts is therefore redundant, and neither measures the "same batch geometry" its message
  claimed; both measure batched-vs-M=1.
- **That is why the 1e-4 advisory fired on every single direction** (observed 2e-3 to 7e-3
  throughout): it was never float noise within one shape, it was the kernel step. Across the 7 arms
  that completed, 224 directions, the largest such delta is 3.25e-03; row 17's 1.03e-02 is a 3x
  outlier in the same phenomenon, not a different one.

**Scorer batch-shape noise floor, MEASURED:** a per-row cosine can move by up to **~1e-2** between a
1-row scoring call and a batched one, and is bit-stable across every batched shape from 8 to 512.
Checklist item 11 already recorded that the right-padding chunk size shifts a per-row cosine
measurably; this is the sharp version of it. `common.score_ids` uses ONE fixed chunk
(`SCORE_CHUNK = 32`) for exactly this reason, so every stored score in the pipeline is at that fixed
shape and is comparable with every other; a number recomputed at a DIFFERENT shape -- above all a
single-row rescore -- carries up to ~1e-2 of jitter and must not be diffed against a stored one at
face value. (This belongs in README's Conventions as well as the GCG section; I was scoped to the
GCG section and did not edit Conventions.)

The bound and the check were left exactly as they were, so all 8 arms are comparable. The arm was
relaunched as `--rows 0-16,18-31 --force`: **31 of 32 directions, guard intact, row 17 named here
rather than absorbed.** Making the CHECK's fresh call batched would have made row 17 pass; that is a
correctness fix to the check rather than a loosening, but it changes what the check measures and
seven arms had already run under the current one, so it was NOT done in this phase.

### The 8 arms

| base | family | arm | dirs | mean final cos | mean init cos | mean NLL | wall/dir | $/dir | arm $ | cand/s | retok reject | mean peak act | frac fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact | `gcg-corpus` | 32 | **0.6398** | 0.5220 | 7.613 | 82 s | $0.0904 | $2.8930 | 945 | 0.061 | -- | -- |
| qwen3-8b | realact | `gcg-random32` | 31 | 0.4924 | 0.0525 | 12.932 | 90 s | $0.0985 | $3.0527 | 868 | 0.048 | -- | -- |
| qwen3-8b | sae | `gcg-corpus` | 32 | **0.3080** | 0.2358 | 7.970 | 81 s | $0.0894 | $2.8595 | 957 | 0.069 | 137.75 | 1.000 |
| qwen3-8b | sae | `gcg-random32` | 32 | 0.2161 | 0.0158 | 13.334 | 84 s | $0.0917 | $2.9354 | 932 | 0.047 | 89.38 | 0.969 |
| qwen36-27b | realact | `gcg-corpus` | 32 | **0.4886** | 0.3491 | 8.278 | 258 s | $0.3257 | $10.4217 | 300 | 0.069 | -- | -- |
| qwen36-27b | realact | `gcg-random32` | 32 | 0.2827 | -0.0179 | 13.081 | 265 s | $0.3345 | $10.7032 | 292 | 0.052 | -- | -- |
| qwen36-27b | sae | `gcg-corpus` | 32 | **0.2408** | 0.1478 | 7.310 | 258 s | $0.3251 | $10.4030 | 301 | 0.076 | 28.94 | 1.000 |
| qwen36-27b | sae | `gcg-random32` | 32 | 0.0682 | 0.0054 | 13.175 | 264 s | $0.3332 | $10.6617 | 294 | 0.050 | 5.21 | 0.875 |

`pop = 1`, so best-over-members and per-member are the same number. 8B `realact/gcg-random32` is 31
directions: row 17 is excluded, see above. Every arm: 64 distinct top strings per direction, 0
top-up-capped iterations, 0 init repairs, 0 short corpus windows, alphabet 90,909 (8B) / 126,220
(27B) as expected. The `sae` activation is RECORDED, never optimised; the peak token is the
cosine's argmax on every final where the feature is non-zero (32/32, 31/32, 32/32, 31/32).

**The init still dominates**, at 32 directions as at 8: corpus minus random is +0.147 (8B realact),
+0.092 (8B sae), +0.206 (27B realact), +0.173 (27B sae). And `gcg-random32`'s NLL is 12.9-13.3 on
both bases against 7.3-8.3 from the corpus window: the unconstrained string is not text.

### Against the MAEMMs, same rows, naive best-of-64

Primary is `qwen36-27b/2026-09-10_rl-8x2048-full`; the 8B comparator is
`qwen3-8b/2026-09-03_run1-rl`. `max_cos` (naive max over the 64 rollouts drawn) and `bo_64` are
equal to every printed digit in all three score files, so naive and unbiased coincide here.

| base | family | arm | GCG mean | comparator | comparator max-of-64 | GCG - max64 | GCG wins |
|---|---|---|---|---|---|---|---|
| qwen36-27b | realact | `gcg-corpus` | 0.4886 | **rl-8x2048-full** | 0.5313 | -0.0427 | 9/32 |
| qwen36-27b | realact | `gcg-corpus` | 0.4886 | rlI-150 | 0.5301 | -0.0415 | 9/32 |
| qwen36-27b | realact | `gcg-random32` | 0.2827 | **rl-8x2048-full** | 0.5313 | -0.2486 | 0/32 |
| qwen36-27b | sae | `gcg-corpus` | 0.2408 | **rl-8x2048-full** | 0.1308 | **+0.1100** | **29/32** |
| qwen36-27b | sae | `gcg-corpus` | 0.2408 | rlI-150 | 0.1112 | **+0.1296** | **30/32** |
| qwen36-27b | sae | `gcg-random32` | 0.0682 | **rl-8x2048-full** | 0.1308 | -0.0626 | 12/32 |
| qwen3-8b | realact | `gcg-corpus` | 0.6398 | run1-rl | 0.6532 | -0.0135 | 11/32 |
| qwen3-8b | realact | `gcg-random32` | 0.4924 | run1-rl | 0.6547 | -0.1623 | 1/31 |
| qwen3-8b | sae | `gcg-corpus` | 0.3080 | run1-rl | 0.1358 | **+0.1722** | **32/32** |
| qwen3-8b | sae | `gcg-random32` | 0.2161 | run1-rl | 0.1358 | +0.0803 | 29/32 |

**The answer differs by family, and that is the finding.**

- On **`realact`** the MAEMM still wins, but narrowly and not everywhere: -0.043 on the 27B (9 of 32
  directions to the search) and -0.014 on the 8B (11 of 32). At 8 directions the smoke put this gap
  at -0.10; at 32 it is a quarter of that, so the earlier number was small-sample.
- On **`sae`** the search wins outright: **32/32 on the 8B (+0.172)** and 29-30/32 on the 27B
  (+0.110 / +0.130). The MAEMMs reach only 0.111-0.136 on encoder columns, and their `sae` rollouts
  are SHORT -- mean 21.2 tokens (8B), 25.3-28.3 (27B) -- against 42.6-54.2 on realact. A 32-token
  optimised string has room against a 21-token rollout that it does not have against a 44-token one.
- The random init is what decides whether the search is competitive at all: it loses every realact
  comparison (0/32 and 1/31) and is the only `sae` cell that loses (12/32).

**The caveat, stated once.** `gcg` searches a SINGLE string of a FIXED T=32 ids and is compared
against the max over 64 sampled rollouts of mean length 21-54. Neither the token budget nor the
sample count nor the compute is matched: 76,800 candidate forwards of 33 tokens is a different
currency from 64 autoregressive rollouts. So a `gcg` number is a reachability figure at T=32 from
one initialisation, not an upper bound over all strings; where it EXCEEDS the MAEMM (the `sae`
family) that direction of the inequality is sound, and where it falls short it bounds nothing.

### Spend

| item | $ |
|---|---|
| 8 arms x 32 directions (one at 31) | 53.93 |
| row-17 reproduction x3 (diagnostic, no product written) | ~0.36 |
| failed launch: `--root <scratch>` relocates inputs too | ~0.03 |
| **total, GCG final phase** | **~54.32** |

Against the $60 cap. Per base: 8B $11.74 (4 arms), 27B $42.19 (4 arms) -- the 27B is 3.6x the 8B.
Measured per-direction: 8B $0.089-0.099, 27B $0.325-0.335, i.e. 82-90 s and 258-265 s of wall.

### (f) GCG final — the reachability ceiling, now paired on the full root

GCG landed on `/vol` after the section above was written, at
`base/<base>/gcg/2026-09-16_v1/{realact,sae}/{gcg-corpus,gcg-random32}/finals.jsonl`, 32
directions per arm from the FINAL draw — so the GCG rows and the MAEMM rows index the same held-out
set and the comparison is per direction, not distributional. (8B realact `gcg-random32` carries 31:
row 17 is excluded upstream and stays excluded here, because the join is built from the finals.)

| base | family | arm | dirs | rows | GCG cos | init cos | nll | 2026-09-03_run1-rl bo64 | 2026-09-03_run1-rl mean64 | GCG - 2026-09-03_run1-rl bo64 | GCG wins vs 2026-09-03_run1-rl | 2026-09-10_rl-8x2048-full@vllm bo64 | 2026-09-10_rl-8x2048-full@vllm mean64 | GCG - 2026-09-10_rl-8x2048-full@vllm bo64 | GCG wins vs 2026-09-10_rl-8x2048-full@vllm | 2026-09-16_base-control@vllm bo64 | 2026-09-16_base-control@vllm mean64 | GCG - 2026-09-16_base-control@vllm bo64 | GCG wins vs 2026-09-16_base-control@vllm | 2026-09-08_rlI-150@vllm bo64 | 2026-09-08_rlI-150@vllm mean64 | GCG - 2026-09-08_rlI-150@vllm bo64 | GCG wins vs 2026-09-08_rlI-150@vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact | gcg-corpus | 32 | 0-31 | 0.6398 ± 0.0113 | 0.5220 ± 0.0132 | 7.613 | 0.6532 | 0.5527 | -0.0134 ± 0.0102 | 0.344 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact | gcg-random32 | 31 | 0-31 | 0.4924 ± 0.0237 | 0.0525 ± 0.0124 | 12.932 | 0.6547 | 0.5541 | -0.1622 ± 0.0211 | 0.032 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | gcg-corpus | 32 | 1024-1055 | 0.3080 ± 0.0172 | 0.2358 ± 0.0170 | 7.97 | 0.1358 | 0.0647 | 0.1722 ± 0.0222 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | gcg-random32 | 32 | 1024-1055 | 0.2161 ± 0.0248 | 0.0158 ± 0.0023 | 13.334 | 0.1358 | 0.0647 | 0.0803 ± 0.0167 | 0.906 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact | gcg-corpus | 32 | 0-31 | 0.4886 ± 0.0273 | 0.3491 ± 0.0255 | 8.278 |  |  |  |  | 0.5313 | 0.4611 | -0.0426 ± 0.0136 | 0.281 | 0.1052 | -0.0021 | 0.3835 ± 0.0221 | 1.0 | 0.5301 | 0.4065 | -0.0415 ± 0.0139 | 0.312 |
| qwen36-27b | realact | gcg-random32 | 32 | 0-31 | 0.2827 ± 0.0343 | -0.0179 ± 0.0157 | 13.081 |  |  |  |  | 0.5313 | 0.4611 | -0.2486 ± 0.0292 | 0.0 | 0.1052 | -0.0021 | 0.1775 ± 0.0247 | 0.938 | 0.5301 | 0.4065 | -0.2474 ± 0.0292 | 0.0 |
| qwen36-27b | sae | gcg-corpus | 32 | 1024-1055 | 0.2408 ± 0.0111 | 0.1478 ± 0.0126 | 7.31 |  |  |  |  | 0.1308 | 0.0859 | 0.1100 ± 0.0166 | 0.906 | 0.0194 | 0.0039 | 0.2214 ± 0.0115 | 1.0 | 0.1112 | 0.065 | 0.1296 ± 0.0172 | 0.938 |
| qwen36-27b | sae | gcg-random32 | 32 | 1024-1055 | 0.0682 ± 0.0097 | 0.0054 ± 0.0012 | 13.175 |  |  |  |  | 0.1308 | 0.0859 | -0.0626 ± 0.0195 | 0.375 | 0.0194 | 0.0039 | 0.0488 ± 0.0096 | 1.0 | 0.1112 | 0.065 | -0.0430 ± 0.0194 | 0.5 |

**Caveat, stated once:** GCG and a MAEMM are not the same object and are not compute-matched. GCG
optimises ONE fixed 32-token string with ~77k candidate forwards *against the scorer itself*; the
MAEMM draws 64 sampled rollouts from a prompt and never sees the metric. GCG is a ceiling on what
the metric is reachable to, not a baseline the inverter competes with.

**On `realact` the inverter is at or above the ceiling; on `sae` it is far below it.** The primary
reaches 0.5313 best-of-64 on the 27B's first 32 realact directions against `gcg-corpus`'s 0.4886 —
GCG wins only 28.1% of directions, paired difference −0.0426 ± 0.0136 — and the 8B is level
(−0.0134 ± 0.0102, GCG winning 34.4%). On `sae` the ceiling is far above both: `gcg-corpus` reaches
0.2408 against the primary's 0.1308, winning **90.6%** of features (+0.1100 ± 0.0166), and on the 8B
it wins **100%** (+0.1722 ± 0.0222). A 32-token string exists that drives an SAE encoder column far
harder than anything the inverter samples, and the inverter does not find it.

The init columns show how much of that is the corpus init rather than the search: `gcg-corpus`
starts at 0.3491 (27B realact) and ends at 0.4886, while `gcg-random32` starts at −0.0179 and
reaches only 0.2827 at an NLL of 13.1 — the search adds ~0.14 on top of a good initialisation and
cannot make up the difference from a random one within the same budget.

### The untrained-base control

A fourth row appeared in every table while this section was being written:
`qwen36-27b/2026-09-16_base-control` (`type: base`, `role: control`) — the CLEAN 27B run through the
identical prompt, marker, injection layer/coefficient and sampling, so the only difference from the
primary is the weights. `stats.py` reads `role` from config.yaml and orders it between the primary
and the secondaries.

| family | control bo1 | control bo64 | primary bo1 | primary bo64 |
|---|---|---|---|---|
| realact | 0.0244 ± 0.0052 | 0.1364 ± 0.0058 | 0.4994 ± 0.0070 | 0.5692 ± 0.0063 |
| sae | 0.0022 ± 0.0003 | 0.0241 ± 0.0012 | 0.1298 ± 0.0032 | 0.1714 ± 0.0035 |
| random | 0.0160 ± 0.0005 | 0.0323 ± 0.0004 | 0.0280 ± 0.0005 | 0.0428 ± 0.0005 |

Injecting a direction into the untrained base and sampling reaches **0.0244** per rollout on
realact against the primary's 0.4994 — **essentially none of the inverter's per-rollout score is
architecture or prompt; it is all training.** The control's own bo1→bo64 climb (0.0244 → 0.1364,
5.6x) is almost entirely sampling luck, the same signature the 8B Patchscopes trial found for its
baseline (3.6x) against run1's 1.2x: the less a system knows, the more best-of-n flatters it.

The control also lands just above the Patchscopes floor at the matched budget (floor bo32 0.1247 on
realact, control bo64 0.1364), which is the more informative comparison than either alone — the
MAEMM prompt with a direction injected into an untrained base is worth about the same as an
entity-description prompt with no direction at all.

### `examples_docmax`, measured

18,813 documents, 952,388 windows (the same enumeration `scan` used), 512 features, 128,217 stored
rows, 57.7 MiB, 4941.6 s, **$6.2319** on H200. Per feature, documents whose best window clears the
gate: median **256** (i.e. the whole top-256 for the median feature), minimum 6, and fewer than 48
on **55 of 512** features. 48 is the number the two test draws need (40 positives plus headroom),
so ~11% of features may still be short once the arms' ~28-44 shown documents are removed; the
chain's acceptance report carries the actual `n_short_draw1` / `n_short_draw2` / `n_empty_draw2`.

### Operational: Modal re-schedules a SIGTERMed container, and a batch gets paid for twice

MEASURED 2026-09-16 on the batch-path smoke. A CPU container polling a Message Batch was killed
with `Runner terminated (SIGTERM), exit code: 143` at 1223 s; Modal re-scheduled the input; the
code re-entered `run_batch` and submitted a SECOND batch for the same six requests. The prompt
cache cannot prevent this, because it is only written once a batch ends. At the full run's 36,864
requests that is a ~$56 double charge and two batches racing for the same work.

Fixed by making every step of the `chain` stage idempotent: batch ids are written to
`runs/<run>/batches/<stage>-<hash>.json` and committed BEFORE the first poll, and a restart
re-attaches to them; a build whose `build.json` exists is reused rather than rerun (otherwise a
restarted chain dies on `OutDir`'s refusal to overwrite its own earlier success); and `STATUS.json`
is read back and continued with a `restarts` counter rather than truncated.

Separately, batch LATENCY looks poor for this workload: two independent 6-request batches each sat
`in_progress` for 20 minutes without completing. The chain therefore re-probes and takes the
half-price batch path only if a probe batch returns inside 30 minutes; otherwise the two full runs
go through the sync path at full price.

### What `examples_docmax` bought, on the same 64 features

| | positive pool = `examples/` only | + `examples_docmax` |
|---|---|---|
| draw 1 short of 20 positives | 29 / 64 | **12 / 64** |
| draw 2 short of 20 positives | 44 / 64 | **7 / 64** |
| draw 2 EMPTY | 21 / 64 | **6 / 64** |
| C4 short of 16 examples | 0 / 64 (after A3) | **0 / 64** |
| negatives short of 20 | 0 / 64 | **0 / 64** |

Draw 2 comes out healthier than draw 1 because draw 2 is ALLOCATED FIRST: when the pool is short it
is draw 2 that would go empty, and an empty draw 2 costs the null -- which, with no `temperature`
parameter available, is the only noise floor the evaluation has. Draw 1 takes what is left and its
shortfall is recorded per feature. Document-level disjointness is never relaxed to make a count.

The 12 features whose draw 1 is short are the sparsest end of the density range (feature 845, for
instance, has corpus density 1.8e-6 and gets 0 positives): their entire gate-passing corpus
presence fits inside what the arms already show. They drop out of the paired contrasts through
`drop_nulls` rather than being padded.

## `top1_act` — activation of the cosine top-1 corpus window per sae feature (2026-09-16, full root)

| date | item | command | wall | cost | result |
|---|---|---|---|---|---|
| 2026-09-16 | `top1_act` 27B, 512 sae features | `--product top1_act --base qwen36-27b --set 2026-09-16_v1` | 201.5 s | **$0.2542** | 508 joined from `examples/`, 4 forwarded; all 512 forwarded as the check (max rel. join-vs-forward 2.4e-2, median 2.2e-6 = the bf16 batch-shape floor); 512/512 pass the gate 1.5846, median act/gate 13.60, Spearman(top1_cos, act_max) 0.937 |
| 2026-09-16 | `top1_act` 8B, 512 sae features | `--product top1_act --base qwen3-8b --set 2026-09-16_v1` | 78.4 s | **$0.0860** | 473 joined, 39 forwarded; 508/512 pass the gate 6.936, median act/gate 14.24, Spearman 0.951 |

The cosine-selected top-1 window is essentially the feature's max-activation window (median act_max / the
feature's held-out max 0.993 (27B) / 0.972 (8B)), because the pre-gate activation is near-monotone in the
cosine at fixed norm. CSV for the figure: `paper/inversion-eval/data/corpus_top1_activation.csv` (1,024 rows).


## GCG stratified sae (full root, 2026-09-16)

The `sae` rows 0-31 used by the final arms are **all density quartile q0**: `targets.py` lays the
sae family out stratum-major, verified here from `ids.jsonl`'s own `stratum` field rather than
assumed -- 128 rows per quartile, contiguous, q0 = sae-local 0-127 = global 1024-1151.

| base | q0 density | q1 | q2 | q3 |
|---|---|---|---|---|
| qwen3-8b | 3.27e-06 - 1.43e-04 | 1.50e-04 - 3.67e-04 | 3.78e-04 - 1.15e-03 | 1.18e-03 - 1.69e-02 |
| qwen36-27b | 4.28e-07 - 4.54e-05 | 4.75e-05 - 1.66e-04 | 1.68e-04 - 4.47e-04 | 4.51e-04 - 4.97e-02 |

So the q0-only arms are the **rare-stratum view**, not the sae family. The rerun takes 8 rows of
EACH quartile -- `--rows 0-7,128-135,256-263,384-391 --family sae` -- into NEW directories
`sae/{gcg-corpus-strat,gcg-random32-strat}` via a new `--arm-suffix` flag, so the q0 arms are never
overwritten. The suffix becomes a directory name and is asserted to be lowercase alphanumerics with
single hyphens (`../q0`, `a/b`, `Strat`, `x--y` and three others rejected); it is deliberately NOT
part of `--arm`, which must stay parseable as `<mode>-<init>`.

### The four stratified arms

Mean +- SE over the 32 TARGETS of each target's best member. For arms that landed before
2026-09-16 the SE is recomputed from `per_dir_best_cos`, which every `summary.json` already carries;
the landed files are NOT rewritten, because a product directory is written once by temp-and-rename
and its README records the wall and cost of the call that produced it, so editing it afterwards
would make that record false. Arms run after this date carry `mean_per_dir_best_cos` and
`se_per_dir_best_cos` natively.

| base | arm | dirs | mean final cos +- SE | mean init cos +- SE | mean NLL | $/dir | wall/dir | mean peak act | frac fired |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | `gcg-corpus-strat` | 32 | **0.2983 +- 0.0168** | 0.2332 +- 0.0143 | 7.461 | $0.0922 | 84 s | 133.96 | 1.000 |
| qwen3-8b | `gcg-random32-strat` | 32 | 0.2032 +- 0.0240 | 0.0098 +- 0.0017 | 13.422 | $0.0946 | 86 s | 87.22 | 0.969 |
| qwen36-27b | `gcg-corpus-strat` | 32 | **0.2344 +- 0.0159** | 0.1594 +- 0.0138 | 7.147 | $0.3358 | 266 s | 29.85 | 1.000 |
| qwen36-27b | `gcg-random32-strat` | 32 | 0.0843 +- 0.0089 | 0.0038 +- 0.0011 | 13.354 | $0.3225 | 256 s | 8.70 | 0.969 |

The q0-only arms for comparison: 8B 0.3080 / 0.2161, 27B 0.2408 / 0.0682. **The arm MEAN barely
moves** (0.01-0.02 in every cell) -- which is why the per-quartile breakdown, not the mean, is the
reason to have run this.

### Per quartile, against the MAEMMs on the same rows (best-of-64 = max)

| base | arm | q0 (rarest) | q1 | q2 | q3 (densest) |
|---|---|---|---|---|---|
| 8B | `gcg-corpus-strat` | 0.3505, **+0.2103** (8/8) | 0.2868, +0.1185 (8/8) | 0.2780, +0.1284 (8/8) | 0.2778, +0.1078 (8/8) |
| 8B | `gcg-random32-strat` | 0.2844, **+0.1441** (8/8) | 0.1833, +0.0150 (6/8) | 0.1476, **-0.0019** (4/8) | 0.1974, +0.0274 (5/8) |
| 27B | `gcg-corpus-strat` | 0.2343, +0.0790 (6/8) | 0.2880, +0.0240 (6/8) | 0.2235, +0.0551 (8/8) | 0.1918, +0.0586 (8/8) |
| 27B | `gcg-random32-strat` | 0.0683, **-0.0870** (3/8) | 0.0993, **-0.1647** (0/8) | 0.0813, -0.0871 (0/8) | 0.0882, -0.0450 (3/8) |

(differences against the primary `qwen36-27b/2026-09-10_rl-8x2048-full` on the 27B and
`qwen3-8b/2026-09-03_run1-rl` on the 8B; against `rlI-150` the 27B corpus arm wins 8/8 in every
quartile, +0.1147 / +0.0435 / +0.0650 / +0.0665.)

**What stratifying changes, and what it does not.**

- **The corpus-init result survives intact.** 8B `gcg-corpus-strat` beats the MAEMM on **32/32**
  targets across every density quartile; the 27B wins 28/32 against the primary and 32/32 against
  `rlI-150`. That is the `sae` headline and it is not a rare-feature artefact.
- **The random-init result does NOT survive.** On the 8B its advantage is real only on q0
  (+0.1441, 8/8) and collapses to a tie everywhere else (+0.015, **-0.002**, +0.027; 6/8, 4/8, 5/8).
  The +0.0803 / 29-of-32 reported for `sae/gcg-random32` in the final-run section is a **q0
  artefact** and should not be read as a `sae`-family result. On the 27B the random arm loses in
  every quartile.
- **Rarer is not uniformly easier.** The 8B corpus arm falls monotonically from q0 to q2 and then
  flattens, but the 27B corpus arm PEAKS at q1 (0.2880) and is lowest at q3 (0.1918), and the
  MAEMM's own q1 is its best quartile too (0.2640). Whatever q1 is on the 27B, both methods find it
  easier, so it is a property of the features rather than of the search.

### The 8 shared rows: the 8B is exactly reproducible, the 27B is not

Rows 1024-1031 are in BOTH the q0 arms and the stratified arms, run independently with the same
per-direction seeding (`[seed, crc32(family), row]`). That makes them a free determinism check:

| base | arm | identical final ids | max abs cos difference |
|---|---|---|---|
| qwen3-8b | `gcg-corpus` | **8/8** | **0.00e+00** |
| qwen3-8b | `gcg-random32` | **8/8** | **0.00e+00** |
| qwen36-27b | `gcg-corpus` | 1/8 | **2.64e-01** |
| qwen36-27b | `gcg-random32` | 0/8 | 1.46e-01 |

**The 8B search is bit-exact across runs; the 27B is not.** The inits are identical in both cases
(max |d| 0.00e+00 corpus, 4.39e-04 random32), so the divergence is in the SEARCH, not the setup: the
27B is the only base whose backward goes through fla's GatedDeltaNet Triton path -- the one that
needed `triton>=3.7.1` -- and a nondeterministic gradient changes the proposed candidates, after
which the trajectories separate for good. Most rows still agree closely (per-row differences
-0.264, -0.049, 0.004, -0.005, 0.004, 0.014, 0.007, 0.000 on `gcg-corpus`); one row per arm diverges
hard.

**The caveat this puts on every 27B number:** a single 27B direction is NOT reproducible, and the
arm mean over these 8 rows moved by -0.036 (`gcg-corpus`) and +0.024 (`gcg-random32`) between runs.
That is the same size as the 27B realact GCG-vs-MAEMM gap (-0.043), so on a 32-target arm the
run-to-run component is smaller but not negligible, and the SE over targets does NOT capture it.
Quoting a 27B arm mean to three decimals overstates what one run establishes.

### Spend

4 arms x 32 directions: 8B $2.9524 + $3.0318, 27B $10.7499 + $10.3208 = **$27.05**, against ~$30.
Per direction: 8B $0.0922-0.0946, 27B $0.3225-0.3359.

### (f) GCG final, stratified

The sae arms were re-run on a STRATIFIED draw — 8 targets from each of the four density quartiles,
32 in all — at `sae/{gcg-corpus-strat,gcg-random32-strat}/` on both bases. Those are now the
reported sae rows. The earlier plain sae arms took the first 32 sae rows, which are **all q0**, so
they are kept as a separate "rare-stratum q0 view" and never averaged in as the family: reporting
them as the sae number would report the rarest quartile as the whole. realact is unchanged.

| base | family | view | arm | slice | dirs | rows | GCG cos | init cos | nll | 2026-09-03_run1-rl bo64 | 2026-09-03_run1-rl mean64 | GCG - 2026-09-03_run1-rl bo64 | GCG wins vs 2026-09-03_run1-rl | 2026-09-10_rl-8x2048-full@vllm bo64 | 2026-09-10_rl-8x2048-full@vllm mean64 | GCG - 2026-09-10_rl-8x2048-full@vllm bo64 | GCG wins vs 2026-09-10_rl-8x2048-full@vllm | 2026-09-16_base-control@vllm bo64 | 2026-09-16_base-control@vllm mean64 | GCG - 2026-09-16_base-control@vllm bo64 | GCG wins vs 2026-09-16_base-control@vllm | 2026-09-08_rlI-150@vllm bo64 | 2026-09-08_rlI-150@vllm mean64 | GCG - 2026-09-08_rlI-150@vllm bo64 | GCG wins vs 2026-09-08_rlI-150@vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact |  | gcg-corpus | all | 32 | 0-31 | 0.6398 ± 0.0113 | 0.5220 ± 0.0132 | 7.613 | 0.6532 | 0.5527 | -0.0134 ± 0.0102 | 0.344 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | gcg-random32 | all | 31 | 0-31 | 0.4924 ± 0.0237 | 0.0525 ± 0.0124 | 12.932 | 0.6547 | 0.5541 | -0.1622 ± 0.0211 | 0.032 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | rare-stratum q0 view | gcg-corpus | all | 32 | 1024-1055 | 0.3080 ± 0.0172 | 0.2358 ± 0.0170 | 7.97 | 0.1358 | 0.0647 | 0.1722 ± 0.0222 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | all | 32 | 1024-1415 | 0.2983 ± 0.0168 | 0.2332 ± 0.0143 | 7.461 | 0.157 | 0.0775 | 0.1412 ± 0.0157 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q0 | 8 | 1024-1031 | 0.3505 ± 0.0206 | 0.2778 ± 0.0198 | 8.299 | 0.1402 | 0.0575 | 0.2103 ± 0.0451 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q1 | 8 | 1152-1159 | 0.2868 ± 0.0268 | 0.2431 ± 0.0234 | 6.622 | 0.1683 | 0.0924 | 0.1185 ± 0.0205 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q2 | 8 | 1280-1287 | 0.2780 ± 0.0174 | 0.2045 ± 0.0212 | 7.842 | 0.1496 | 0.0653 | 0.1284 ± 0.0227 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q3 | 8 | 1408-1415 | 0.2778 ± 0.0551 | 0.2073 ± 0.0416 | 7.082 | 0.17 | 0.0948 | 0.1078 ± 0.0218 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | rare-stratum q0 view | gcg-random32 | all | 32 | 1024-1055 | 0.2161 ± 0.0248 | 0.0158 ± 0.0023 | 13.334 | 0.1358 | 0.0647 | 0.0803 ± 0.0167 | 0.906 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | all | 32 | 1024-1415 | 0.2032 ± 0.0240 | 0.0098 ± 0.0017 | 13.422 | 0.157 | 0.0775 | 0.0461 ± 0.0194 | 0.719 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q0 | 8 | 1024-1031 | 0.2844 ± 0.0439 | 0.0116 ± 0.0029 | 13.517 | 0.1402 | 0.0575 | 0.1441 ± 0.0369 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q1 | 8 | 1152-1159 | 0.1833 ± 0.0486 | 0.0177 ± 0.0041 | 13.75 | 0.1683 | 0.0924 | 0.0150 ± 0.0423 | 0.75 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q2 | 8 | 1280-1287 | 0.1476 ± 0.0416 | 0.0053 ± 0.0019 | 13.283 | 0.1496 | 0.0653 | -0.0019 ± 0.0355 | 0.5 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q3 | 8 | 1408-1415 | 0.1974 ± 0.0525 | 0.0047 ± 0.0019 | 13.137 | 0.17 | 0.0948 | 0.0273 ± 0.0195 | 0.625 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | gcg-corpus | all | 32 | 0-31 | 0.4886 ± 0.0273 | 0.3491 ± 0.0255 | 8.278 |  |  |  |  | 0.5313 | 0.4611 | -0.0426 ± 0.0136 | 0.281 | 0.1052 | -0.0021 | 0.3835 ± 0.0221 | 1.0 | 0.5301 | 0.4065 | -0.0415 ± 0.0139 | 0.312 |
| qwen36-27b | realact |  | gcg-random32 | all | 32 | 0-31 | 0.2827 ± 0.0343 | -0.0179 ± 0.0157 | 13.081 |  |  |  |  | 0.5313 | 0.4611 | -0.2486 ± 0.0292 | 0.0 | 0.1052 | -0.0021 | 0.1775 ± 0.0247 | 0.938 | 0.5301 | 0.4065 | -0.2474 ± 0.0292 | 0.0 |
| qwen36-27b | sae | rare-stratum q0 view | gcg-corpus | all | 32 | 1024-1055 | 0.2408 ± 0.0111 | 0.1478 ± 0.0126 | 7.31 |  |  |  |  | 0.1308 | 0.0859 | 0.1100 ± 0.0166 | 0.906 | 0.0194 | 0.0039 | 0.2214 ± 0.0115 | 1.0 | 0.1112 | 0.065 | 0.1296 ± 0.0172 | 0.938 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | all | 32 | 1024-1415 | 0.2344 ± 0.0159 | 0.1594 ± 0.0138 | 7.147 |  |  |  |  | 0.1802 | 0.1316 | 0.0542 ± 0.0100 | 0.875 | 0.0218 | 0.0032 | 0.2126 ± 0.0164 | 1.0 | 0.162 | 0.1109 | 0.0724 ± 0.0101 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q0 | 8 | 1024-1031 | 0.2343 ± 0.0357 | 0.1481 ± 0.0391 | 6.938 |  |  |  |  | 0.1553 | 0.083 | 0.0790 ± 0.0348 | 0.75 | 0.0191 | 0.0045 | 0.2151 ± 0.0357 | 1.0 | 0.1196 | 0.0441 | 0.1147 ± 0.0322 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q1 | 8 | 1152-1159 | 0.2880 ± 0.0254 | 0.2105 ± 0.0170 | 6.864 |  |  |  |  | 0.2639 | 0.212 | 0.0240 ± 0.0115 | 0.75 | 0.0196 | 0.0039 | 0.2683 ± 0.0265 | 1.0 | 0.2444 | 0.1919 | 0.0435 ± 0.0077 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q2 | 8 | 1280-1287 | 0.2235 ± 0.0218 | 0.1472 ± 0.0193 | 7.584 |  |  |  |  | 0.1684 | 0.1312 | 0.0551 ± 0.0101 | 1.0 | 0.0196 | 0.0018 | 0.2039 ± 0.0212 | 1.0 | 0.1585 | 0.1142 | 0.0650 ± 0.0081 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q3 | 8 | 1408-1415 | 0.1918 ± 0.0375 | 0.1318 ± 0.0252 | 7.203 |  |  |  |  | 0.1332 | 0.1002 | 0.0586 ± 0.0109 | 1.0 | 0.0286 | 0.0026 | 0.1632 ± 0.0394 | 1.0 | 0.1253 | 0.0935 | 0.0665 ± 0.0157 | 1.0 |
| qwen36-27b | sae | rare-stratum q0 view | gcg-random32 | all | 32 | 1024-1055 | 0.0682 ± 0.0097 | 0.0054 ± 0.0012 | 13.175 |  |  |  |  | 0.1308 | 0.0859 | -0.0626 ± 0.0195 | 0.375 | 0.0194 | 0.0039 | 0.0488 ± 0.0096 | 1.0 | 0.1112 | 0.065 | -0.0430 ± 0.0194 | 0.5 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | all | 32 | 1024-1415 | 0.0843 ± 0.0089 | 0.0038 ± 0.0011 | 13.354 |  |  |  |  | 0.1802 | 0.1316 | -0.0959 ± 0.0155 | 0.188 | 0.0218 | 0.0032 | 0.0625 ± 0.0095 | 0.938 | 0.162 | 0.1109 | -0.0777 ± 0.0147 | 0.156 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q0 | 8 | 1024-1031 | 0.0683 ± 0.0185 | 0.0068 ± 0.0034 | 13.569 |  |  |  |  | 0.1553 | 0.083 | -0.0869 ± 0.0340 | 0.375 | 0.0191 | 0.0045 | 0.0492 ± 0.0184 | 1.0 | 0.1196 | 0.0441 | -0.0513 ± 0.0278 | 0.375 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q1 | 8 | 1152-1159 | 0.0993 ± 0.0270 | 0.0022 ± 0.0021 | 13.266 |  |  |  |  | 0.2639 | 0.212 | -0.1647 ± 0.0252 | 0.0 | 0.0196 | 0.0039 | 0.0796 ± 0.0295 | 0.875 | 0.2444 | 0.1919 | -0.1452 ± 0.0324 | 0.0 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q2 | 8 | 1280-1287 | 0.0813 ± 0.0097 | 0.0024 ± 0.0013 | 13.307 |  |  |  |  | 0.1684 | 0.1312 | -0.0871 ± 0.0202 | 0.0 | 0.0196 | 0.0018 | 0.0617 ± 0.0096 | 1.0 | 0.1585 | 0.1142 | -0.0772 ± 0.0177 | 0.0 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q3 | 8 | 1408-1415 | 0.0882 ± 0.0128 | 0.0035 ± 0.0020 | 13.273 |  |  |  |  | 0.1332 | 0.1002 | -0.0450 ± 0.0312 | 0.375 | 0.0286 | 0.0026 | 0.0596 ± 0.0147 | 0.875 | 0.1253 | 0.0935 | -0.0371 ± 0.0266 | 0.25 |

**The stratified draw halves the sae gap, and the q0-only view was the reason it looked larger.**
On the 27B, `gcg-corpus` against the primary: the q0 view says **+0.1100 ± 0.0166** (GCG winning
90.6%), the stratified draw says **+0.0542 ± 0.0100** (87.5%). Same arm, same objective, same
scorer — the difference is entirely which features were drawn. The 8B moves the same way, +0.1722 →
**+0.1412 ± 0.0157**, still winning 100% of features. So the qualitative claim survives (a 32-token
optimised string beats the inverter on SAE encoder columns, on every 8B feature tested and 87.5% of
27B ones) while the magnitude on the 27B was overstated about twofold by the draw.

**Per quartile the gap tracks how rare the feature is**, and so does the inverter's own score. On
the 27B the primary's best-of-64 runs 0.1553 / 0.2639 / 0.1684 / 0.1332 across q0-q3 while the GCG
gap runs +0.0790 / +0.0240 / +0.0551 / +0.0586 — widest in q0, narrowest in q1, which is also where
the inverter is strongest. The 8B is monotone: gap +0.2103 / +0.1185 / +0.1284 / +0.1078.

**Random-init is the control that separates search from initialisation.** `gcg-random32-strat`
LOSES to the 27B primary on the stratified draw (−0.0959 ± 0.0155, winning 18.8%) while the
corpus-init arm wins — so on the 27B the sae ceiling is reachable from a good corpus window and not
from 300 iterations starting at noise. On the 8B random-init still wins (+0.0461 ± 0.0194, 71.9%),
concentrated in q0 (+0.1441, 100%) and gone by q2 (−0.0019, 50%).

Source note: the per-direction best member is taken from `finals.jsonl`; where `summary.json` also
carries `per_dir_best_cos` the two agree exactly (MEASURED on the 27B stratified arms, max
|difference| = **0.0**), so the table uses the finals and the summary is a cross-check, not a second
source.

## GCG/EPO 32-target arms (full root)

Eight EPO arms on the SAME 32-target selections as the `gcg` arms: `realact` rows 0-31 and the
stratified `sae` rows (`--rows 0-7,128-135,256-263,384-391`), both inits, both bases. Config as
before: pop 3 at lambda 0.1 / 0.19 / 0.37, 85 children x 300 iterations, each member selected by its
own `L_lambda` so one run traces the Pareto front. Arm names `epo-corpus` / `epo-random32` on
`realact` and `epo-corpus-strat` / `epo-random32-strat` on `sae`.

Launched in two waves, corpus-init first at the halved scope and `random32` released afterwards:

| wave | arm | app |
|---|---|---|
| 1 | 27B `realact/epo-corpus` | `ap-g1qeBZYciX07uk2Mk9Oz4h` |
| 1 | 27B `sae/epo-corpus-strat` | `ap-ZhT7yERDXUkxoiKFF8uOGC` |
| 1 | 8B `realact/epo-corpus` | `ap-A8X7er0BHFldL8qHbSJkwE` |
| 1 | 8B `sae/epo-corpus-strat` | `ap-d4xzFG9m1Auae9vHseDVqi` |
| 2 | 27B `realact/epo-random32` | `ap-pTJ6lWVb5x7783bEUuTssS` |
| 2 | 27B `sae/epo-random32-strat` | `ap-5MSK86hXV5qoCRbt6NBMvP` |
| 2 | 8B `realact/epo-random32` | `ap-b4ZacPXmWbeCP7rGwfLc0P` |
| 2 | 8B `sae/epo-random32-strat` | `ap-BI4SvP04oVgyAJxCcjMsUr` |

**Projection** (superseded; kept because the 27B half of it was acted on), at the per-direction
rates measured on the smoke-root `epo` arms and amortised to a 32-direction call: 27B $1.1126 /
$1.1294 (corpus, realact / sae) and $1.0343 / $1.0442 (random32), 8B $0.2857 / $0.2745 and $0.2786 /
$0.2770. Times 32 directions: 27B $71.7 (corpus) + $66.5 (random32), 8B $17.9 + $17.8, EPO total
~= $174.

**What that projection did not say is that a 27B arm does not FIT.** The per-direction figure was
right -- MEASURED **~870 s/direction, $1.10/direction** on H200, against $1.1126 projected -- but
32 x 870 s = **7.7 h** against the `timeout=6 * 3600` that `gcg/modal_app.py` then carried on both
GPU functions. All four 27B arms were therefore cancelled by Modal at exactly 21600 s, mid-direction,
with 24-25 of 32 directions written:

| 27B arm | app | dirs at the kill | killed |
|---|---|---|---|
| `realact/epo-corpus` | `ap-g1qeBZYciX07uk2Mk9Oz4h` | 24/32 | 01:11:06Z |
| `realact/epo-random32` | `ap-pTJ6lWVb5x7783bEUuTssS` | 25/32 | 01:32:54Z |
| `sae/epo-corpus-strat` | `ap-ZhT7yERDXUkxoiKFF8uOGC` | 24/32 | 01:11:34Z |
| `sae/epo-random32-strat` | `ap-5MSK86hXV5qoCRbt6NBMvP` | 24/32 | 01:32:54Z |

Each kill is exactly launch + 6 h (the 8B siblings' READMEs date wave 1 at 19:11-19:12Z and wave 2
at 19:32-19:33Z). The logs read `Task's current input ... hit its timeout of 21600s`, then
`[modal-client] Received a cancellation signal while processing input`, then `[outdir] FAILED:
KeyboardInterrupt; temp dir kept at ...`, then `Runner terminated.` **This was NOT a dropped or
reaped client**: `--detach` was used and held, there is no `Stopping app` and no disconnect in any
of the four logs, and the four kill times are set by the timeout, not by anything local. The 8B
arms were unaffected because they run ~250-270 s/direction and finish in 2.2-2.4 h.

**The fix, and the recovery.** `timeout=9 * 3600` on both GPU functions, and `--resume-from` in
`gcg.py`: the kept temp dir's `finals.jsonl` / `trajectory.jsonl` / `top64.jsonl` are copied into
the new temp dir and appended to, the rows already in the finals are skipped inside the loop, and
`sel` stays the full 32 so every per-arm mean is over all 32. See `README.md` for what it refuses
before the model load. Resuming cost **~$36** against **~$140** to re-run all four from scratch --
which under the old timeout could not have finished either.

| 2026-09-17 | item | app | result |
|---|---|---|---|
| resume smoke, 27B `realact` row 24 into `epo-corpus-smoke` (`--rows 0-24`, kept on the volume, NOT a paper arm) | `ap-ESlxEEeDLL7ltqLQcBGwep` | 24 of 25 carried, 1 run; 75 / 2325 / 4800 rows out for 72 / 2232 / 4608 in; 1031 s, **$1.3008**, all charged to the one direction |
| an earlier launch of the same smoke | `ap-L6rOb7IBNn13EnbuyNTafb` | died locally in the entrypoint import, `No module named 'numpy'` -- the launcher needs `uvx --with pyyaml --with numpy`, as `README.md` says. No GPU, ~$0 |
| 27B `realact/epo-corpus` resume, 8 dirs | `ap-Pid7yfAtLlMTC78YQG8xcQ` | see the EPO table below |
| 27B `realact/epo-random32` resume, 7 dirs | `ap-2UmLdVNfA3NvbmGksINNGX` | see the EPO table below |
| 27B `sae/epo-corpus-strat` resume, 8 dirs | `ap-6CbAEdWnMupsubparzdmQH` | see the EPO table below |
| 27B `sae/epo-random32-strat` resume, 8 dirs | `ap-wdxGHi79XKWy7951aFcNi3` | see the EPO table below |

### The EPO arms, landed (2026-09-17)

All eight, 32 directions each. `mean final cos` is the per-direction BEST member (the quantity the
paper's table (f) reports), +- SE over the 32 directions; the arm README's own "mean final cos" is
over all three members and is lower. Pulled by `paper/inversion-eval/scripts/pull_gcg.py` into
`data/gcg_final.csv`, `data/gcg_per_target.csv`, `data/gcg_quartiles.csv` and, through
`same_targets.py`, `data/same_targets.csv`: 196 cross-checks, 0 disagreements.

| base | family | arm | dirs | mean final cos | matching `gcg` | NLL `epo` / `gcg` | MAEMM bo64 same rows | search wins | $/direction | arm $ |
|---|---|---|---|---|---|---|---|---|---|---|
| 8B | realact | `epo-corpus` | 32 | **0.5787 +- 0.0117** | 0.6398 +- 0.0113 | 3.020 / 7.613 | 0.6532 | 1/32 | $0.2720 | $8.7055 |
| 8B | realact | `epo-random32` | 32 | 0.4689 +- 0.0215 | 0.4924 +- 0.0237 | 4.746 / 12.932 | 0.6532 | 1/32 | $0.2941 | $9.4103 |
| 8B | sae | `epo-corpus-strat` | 32 | **0.2516 +- 0.0149** | 0.2983 +- 0.0168 | 2.926 / 7.461 | 0.1570 | 31/32 | $0.2778 | $8.8890 |
| 8B | sae | `epo-random32-strat` | 32 | 0.1481 +- 0.0211 | 0.2032 +- 0.0240 | 4.039 / 13.422 | 0.1570 | 15/32 | $0.2970 | $9.5032 |
| 27B | realact | `epo-corpus` | 32 | **0.43 +- 0.03** | 0.49 +- 0.03 | 3.026 / 8.278 | 0.5313 | 2/32 | $1.1377 | $36.40 |
| 27B | realact | `epo-random32` | 32 | 0.22 +- 0.03 | 0.28 +- 0.03 | 5.121 / 13.081 | 0.5313 | 0/32 | $1.0786 | $34.52 |
| 27B | sae | `epo-corpus-strat` | 32 | **0.19 +- 0.02** | 0.23 +- 0.02 | 2.557 / 7.147 | 0.1802 | 14/32 | $1.1100 | $35.52 |
| 27B | sae | `epo-random32-strat` | 32 | 0.03 +- 0.01 | 0.08 +- 0.01 | 4.461 / 13.354 | 0.1802 | 3/32 | $1.1376 | $36.40 |

MAEMM columns are `8b_lora` on the 8B and `27b_full` on the 27B, unbiased best-of-64 on EXACTLY
those rows. 27B means are quoted to 2 dp with SE, per the precision convention above.

**`epo` trails `gcg` on raw cosine in all eight cells** (by 0.02-0.06) and buys 4-9 nats of NLL for
it, which is what the 8-direction pilot said and what the arm is for: `cos - lambda * nll` is not
`cos`. The resume costs are the second call only -- `27B arm $` adds the ~$27.24 of the 6 h the
timed-out first call had already been charged for.

| 27B arm | resume app | dirs run | wall | resume $ | $/dir (resume) |
|---|---|---|---|---|---|
| `realact/epo-corpus` | `ap-Pid7yfAtLlMTC78YQG8xcQ` | 8 | 7265.8 s | $9.1629 | $1.1454 |
| `realact/epo-random32` | `ap-2UmLdVNfA3NvbmGksINNGX` | 7 | 5769.3 s | $7.2757 | $1.0394 |
| `sae/epo-corpus-strat` | `ap-6CbAEdWnMupsubparzdmQH` | 8 | 6565.5 s | $8.2798 | $1.0350 |
| `sae/epo-random32-strat` | `ap-wdxGHi79XKWy7951aFcNi3` | 8 | 7266.5 s | $9.1639 | $1.1455 |

**Resume total $33.88 + $1.30 smoke, against ~$140 to re-run all four from scratch** -- which under
the old 6 h timeout could not have finished at all. EPO grand total, both bases, sunk time included:
$36.51 (8B) + $144.14 (27B) = **$180.65**, against the $174 projection that did not know about the
timeout.

### A pre-existing check that only `pop = 1` could ever have passed

`pull_gcg.py`'s reproduction gate compared ITS number -- the per-direction BEST member -- against the
arm README's "mean final cos", which the pipeline writes over EVERY finals row, i.e. all `pop`
members. On a `gcg` arm `pop = 1` and the two are the same number, so the gate was green for every
arm that had ever run. The first `epo` pull made all EIGHT arms disagree at once, the four 8B ones
included -- they had landed 2026-09-16 and nothing about them had changed. The gate, not the data,
was wrong. Each side is now checked against its own upstream: the recomputed ALL-member means against
the README, and the best-member means against `summary.json`'s `mean_per_dir_best_cos`.

A second, subtler one fell out of the same fix. `summary.json`'s `mean_per_dir_init_cos` is the mean
over directions of the MAX init cosine across members, while the per-target table carries the init of
the member that won on the FINAL cosine. On a corpus-init arm every member starts from the same
string and the two coincide; on a `random32` `epo` arm each member draws its own init and they differ
by up to **1.6e-2** (MEASURED, 8B `realact/epo-random32`). Both `random32` pairs, on both bases, flagged
it. The like-for-like quantity is now recomputed for the check rather than the two being conflated.
**Anything quoting a `random32` `epo` arm's "mean init cos" must say which of the two it means.**

Offline, before any of this was launched: the `--resume-from` guards were exercised against the real
kept `realact/epo-corpus` jsonls in 11 cases -- 2 that must pass and 9 that must assert (rows outside
`--rows`, nothing left to run, resuming from the call's own temp dir, a missing stream file, a partly
written direction, a missing member, a different `--iters`, a different `--lam-grid`, the wrong
family's dir). All 11 behaved; removing either of two guards turned the matching case red.

### Two conventions this section follows

**`sae` reporting.** The **stratified** arms are the reported `sae` rows. The `--rows 0-31` arms are
a labelled **rare-stratum (q0) view**, because a stratum-major family makes rows 0-31 all of density
quartile q0 — they are kept, not discarded, since the q0 contrast is itself a result (the
random-init `sae` win exists only there). `realact` is not stratified and its rows 0-31 are the
family.

**27B precision.** 27B arm means are quoted to **2 decimal places with their SE**; the 8B keeps 3.
The reason is the non-determinism measured above: the 27B backward runs fla's GatedDeltaNet Triton
kernels, single directions are NOT reproducible (over the 8 shared rows, 1/8 and 0/8 identical final
strings, max |d cos| **0.26**), and arm means move by about -+0.03 between runs. So no per-direction
claim on the 27B is safe from one run, and a 27B arm mean is good to about the second decimal. The
8B search is bit-exact across runs and needs no such hedge.

### The full run, 2026-09-16 (chain `2026-09-16_autointerp-chain2`, batch path)

`STATUS.json` `state: done`, `restarts: 3`, elapsed 8573 s. Path chosen by measurement: `batch`,
at 50% of list price. Costs are computed from returned token counts at $2.00/$10.00 per MTok.

| stage | calls | wall | cost | errors |
|---|---|---|---|---|
| pilot (64 features, 12 arm-variants) | 12,031 | — | **$34.89** first pass; **$0.00** on replay | 0 |
| primary explain (512 x 6 arms) | 3,072 | — | included below | 0 |
| primary detection (512 x 9 arms) | 34,843 | 32 s after RE-ATTACH (2 h 06 m server-side) | included below | 0 |
| primary fuzzing | 34,843 | 846 s | included below | 0 |
| **primary total** | **69,686** | — | **$91.92** | **0** |
| rlI-150 explain (512 x 2 arms) | 1,024 | 452 s | included below | 0 |
| rlI-150 detection | 7,782 | 727 s | included below | 0 |
| rlI-150 fuzzing | 7,782 | 2,283 s | included below | 0 |
| **rlI-150 total** | **16,589** | — | **$23.69** | **0** |

**Zero failed calls in 98,306 requests.** Batch latency is the reason the wall times look odd: the
three detection batches that had already ENDED did so 2 h 06 m after submission, and the re-attach
then collected 34,843 results in 32 s.

### The preemption, and what the ledger was for

MEASURED 2026-09-16: a container was preempted 2176 s into the primary detection stage --
*"Container terminated due to preemption. Your Function will be restarted with the same input"* --
and the detached app did not come back, leaving five Message Batches running server-side with
nobody waiting on them. On relaunch:

```
[detection] RE-ATTACHING to 5 batch(es) from .../batches/detection-a73b599c689c7ef1.json
            -- not resubmitting 34843 requests
```

The stage had projected **$56.49** and paid none of it. Batch state at that moment, from the API:
three ENDED at 8,000/8,000, two still `in_progress` (8,000 + 2,843), **zero errored, zero expired,
zero canceled**, totalling the ledger's `n = 34,843` exactly. The whole pilot also replayed at
$0.00, and `build_full_present` reused the 512 build. The function now carries
`retries=modal.Retries(max_retries=3)`, which is only safe because all three of the prompt cache,
the batch ledger and the build reuse make the driver idempotent.

### Total autointerp spend

GPU $9.22 (`sae_self` x2 $0.57, `random_pool` $0.14, `examples_4m` $1.63, `examples_docmax` $6.23,
plus $0.51 sunk on a check that turned out to need a near-tie tolerance). LLM $170.72 ($2.11 on
OpenRouter before the switch, $34.89 pilot, $91.92 primary, $23.69 rlI-150, ~$18 across the earlier
stopped chains' pilots). **~$180 against the $500 ceiling.**

### All 8 EPO arms landed (2026-09-17)

Mean +- SE over the 32 TARGETS of each target's best member; 27B to 2 dp per the precision rule.

| base | family | arm | dirs | mean final cos +- SE | mean init cos | mean NLL | true $/dir | mean peak act | frac fired |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact | `epo-corpus` | 32 | **0.5787 +- 0.012** | 0.5218 | 3.020 | $0.2720 | -- | -- |
| qwen3-8b | realact | `epo-random32` | 32 | 0.4689 +- 0.021 | 0.0558 | 4.746 | $0.2941 | -- | -- |
| qwen3-8b | sae | `epo-corpus-strat` | 32 | **0.2516 +- 0.015** | 0.2332 | 2.926 | $0.2778 | 117.72 | 1.000 |
| qwen3-8b | sae | `epo-random32-strat` | 32 | 0.1481 +- 0.021 | 0.0101 | 4.039 | $0.2970 | 67.19 | 0.781 |
| qwen36-27b | realact | `epo-corpus` | 32 | **0.43 +- 0.03** | 0.3491 | 3.026 | $1.1375 | -- | -- |
| qwen36-27b | realact | `epo-random32` | 32 | 0.22 +- 0.03 | -0.0226 | 5.121 | $1.0786 | -- | -- |
| qwen36-27b | sae | `epo-corpus-strat` | 32 | **0.19 +- 0.02** | 0.1594 | 2.557 | $1.1099 | 23.43 | 0.969 |
| qwen36-27b | sae | `epo-random32-strat` | 32 | 0.03 +- 0.01 | 0.0044 | 4.461 | $1.1376 | 2.12 | **0.250** |

Verified on all eight: `n_directions` 32, `finals.jsonl` 96 rows (32 targets x 3 members),
`trajectory.jsonl` 2,976, `top64.jsonl` 6,144, `summary.json`, README with a cost, lambdas
[0.1, 0.19, 0.37] in every arm.

Per member (the Pareto slice the population exists to trace), cos / NLL at lambda 0.1 / 0.19 / 0.37:

| base | arm | lambda 0.1 | lambda 0.19 | lambda 0.37 |
|---|---|---|---|---|
| 8B | `realact/epo-corpus` | 0.5764 / 3.024 | 0.5594 / 2.900 | 0.5380 / 2.778 |
| 8B | `realact/epo-random32` | 0.4445 / 4.819 | 0.3750 / 4.403 | 0.2975 / 4.188 |
| 8B | `sae/epo-corpus-strat` | 0.2506 / 2.928 | 0.2416 / 2.849 | 0.2309 / 2.774 |
| 8B | `sae/epo-random32-strat` | 0.1296 / 4.158 | 0.0721 / 3.911 | 0.0521 / 3.874 |
| 27B | `realact/epo-corpus` | 0.4212 / 3.052 | 0.3969 / 2.781 | 0.3790 / 2.715 |
| 27B | `realact/epo-random32` | 0.1948 / 5.087 | 0.1449 / 4.904 | 0.0892 / 4.619 |
| 27B | `sae/epo-corpus-strat` | 0.1874 / 2.557 | 0.1774 / 2.471 | 0.1632 / 2.428 |
| 27B | `sae/epo-random32-strat` | 0.0255 / 4.432 | 0.0165 / 4.460 | 0.0079 / 4.136 |

**Lambda trades cosine for fluency monotonically in all eight arms**, with no exception on either
base or either family -- 24 of 24 (arm, lambda-step) pairs move cosine down and NLL down together.
Against the matching `gcg` arms the trade is large: 8B `realact` gives up 0.061 of cosine (0.6398 ->
0.5787) for **4.6 nats** (7.61 -> 3.02), and the 27B gives up 0.06 for 5.3 nats. The `random32`
arms trade far worse (8B `sae` loses 0.055 of an already small 0.2032 for 9.4 nats), which is the
expected shape: there is no fluent neighbourhood to fall back into when the start is noise.

**EPO trails GCG on raw cosine in all eight cells**, by 0.023-0.061 -- expected at lambda > 0, but
the best member is at lambda 0.1 rather than 0, so the 3x-smaller per-iteration candidate pool
(255 against 512) costs something of its own on top of the objective change.

**The 27B `sae/epo-random32-strat` arm is where the search stops working**: cos 0.03 +- 0.01 and the
feature fires on only **25%** of finals, against 97% for the corpus init on the same rows. The
corresponding `gcg` arm reached 0.08 with 97% firing, so adding the fluency penalty to an already
failing random start is what pushes it below the gate.

### Spend, with the timeout that the projection did not include

All four 27B EPO arms **hit the app's 6-hour function timeout at 24-25 of 32 directions**
(`timeout=6*3600` in `gcg/modal_app.py`, against a measured 32 x ~880 s = 7.8 h). They were completed
by a `--resume-from <kept temp dir>` run from another session (commit `1a9bf26`, which also raised
the timeout to 9 h), so the partial compute was reused rather than discarded and every arm is a full
32 directions. The cost, however, is the timed-out call PLUS the resume:

| arm | timed-out call | resume | true total | true $/dir |
|---|---|---|---|---|
| 27B `realact/epo-corpus` | $27.24 | $9.1609 | **$36.40** | $1.1375 |
| 27B `realact/epo-random32` | $27.24 | $7.2740 | **$34.51** | $1.0786 |
| 27B `sae/epo-corpus-strat` | $27.24 | $8.2769 | **$35.52** | $1.1099 |
| 27B `sae/epo-random32-strat` | $27.24 | $9.1623 | **$36.40** | $1.1376 |

**27B EPO $142.83, 8B EPO $36.51, EPO total $179.34** against the $174 projection -- +3%, and the
projection was right per direction ($1.08-1.14 measured against $1.03-1.13 projected). The arm
READMEs on the volume record only the RESUME call's cost for the four 27B arms, so reading
`- cost:` off those four understates them by $27.24 each; the true figures are the table above.

### (f) GCG and EPO, 32 targets

All eight EPO arms landed (96 rows each = 32 targets x 3 members at lambda 0.1 / 0.19 / 0.37), so
table (f) now carries both optimisers. **Verified rather than assumed** after the reported 6 h
timeout and resume of the four 27B arms: every one of the eight has 96 rows over exactly 32 distinct
directions, 3.0 rows per direction — no direction lost and none double-written by the resume.

Precision: an arm's own mean columns are 2 dp on the 27B and 3 on the 8B, the precision their SEs
(~0.03 / ~0.01) support. The PAIRED columns keep 4 dp — they are per-direction differences with much
tighter SEs, and rounding them to the arm's precision would make -0.0426 and -0.0450 both read
-0.04. `lam ... (member)` rows are per-member means, not per-target bests, and carry no MAEMM
columns: pairing one member against the inverter is a different claim from the arm's reachability.

| base | family | view | arm | slice | dirs | rows | GCG cos | init cos | nll | 2026-09-03_run1-rl bo64 | 2026-09-03_run1-rl mean64 | GCG - 2026-09-03_run1-rl bo64 | GCG wins vs 2026-09-03_run1-rl | 2026-09-10_rl-8x2048-full@vllm bo64 | 2026-09-10_rl-8x2048-full@vllm mean64 | GCG - 2026-09-10_rl-8x2048-full@vllm bo64 | GCG wins vs 2026-09-10_rl-8x2048-full@vllm | 2026-09-16_base-control@vllm bo64 | 2026-09-16_base-control@vllm mean64 | GCG - 2026-09-16_base-control@vllm bo64 | GCG wins vs 2026-09-16_base-control@vllm | 2026-09-08_rlI-150@vllm bo64 | 2026-09-08_rlI-150@vllm mean64 | GCG - 2026-09-08_rlI-150@vllm bo64 | GCG wins vs 2026-09-08_rlI-150@vllm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8b | realact |  | epo-corpus | all | 32 | 0-31 | 0.579 ± 0.012 | 0.522 ± 0.013 | 3.02 | 0.6532 | 0.5527 | -0.0745 ± 0.0099 | 0.031 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-corpus | lam 0.1 (member) | 32 | 0-31 | 0.576 ± 0.012 |  | 3.024 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-corpus | lam 0.19 (member) | 32 | 0-31 | 0.559 ± 0.012 |  | 2.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-corpus | lam 0.37 (member) | 32 | 0-31 | 0.538 ± 0.012 |  | 2.778 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-random32 | all | 32 | 0-31 | 0.469 ± 0.021 | 0.056 ± 0.012 | 4.746 | 0.6532 | 0.5527 | -0.1843 ± 0.0201 | 0.031 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-random32 | lam 0.1 (member) | 32 | 0-31 | 0.445 ± 0.021 |  | 4.819 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-random32 | lam 0.19 (member) | 32 | 0-31 | 0.375 ± 0.028 |  | 4.403 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | epo-random32 | lam 0.37 (member) | 32 | 0-31 | 0.298 ± 0.022 |  | 4.188 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | gcg-corpus | all | 32 | 0-31 | 0.640 ± 0.011 | 0.522 ± 0.013 | 7.613 | 0.6532 | 0.5527 | -0.0134 ± 0.0102 | 0.344 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | realact |  | gcg-random32 | all | 31 | 0-31 | 0.492 ± 0.024 | 0.053 ± 0.012 | 12.932 | 0.6547 | 0.5541 | -0.1622 ± 0.0211 | 0.032 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | all | 32 | 1024-1415 | 0.252 ± 0.015 | 0.233 ± 0.014 | 2.926 | 0.157 | 0.0775 | 0.0946 ± 0.0146 | 0.969 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | density q0 | 8 | 1024-1031 | 0.295 ± 0.020 | 0.278 ± 0.020 | 3.29 | 0.1402 | 0.0575 | 0.1550 ± 0.0427 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | density q1 | 8 | 1152-1159 | 0.259 ± 0.025 | 0.243 ± 0.023 | 2.969 | 0.1683 | 0.0924 | 0.0902 ± 0.0242 | 0.875 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | density q2 | 8 | 1280-1287 | 0.225 ± 0.019 | 0.204 ± 0.021 | 2.799 | 0.1496 | 0.0653 | 0.0753 ± 0.0206 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | density q3 | 8 | 1408-1415 | 0.228 ± 0.046 | 0.207 ± 0.042 | 2.645 | 0.17 | 0.0948 | 0.0578 ± 0.0135 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | lam 0.1 (member) | 32 | 1024-1415 | 0.251 ± 0.015 |  | 2.928 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | lam 0.19 (member) | 32 | 1024-1415 | 0.242 ± 0.014 |  | 2.849 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-corpus-strat | lam 0.37 (member) | 32 | 1024-1415 | 0.231 ± 0.014 |  | 2.774 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | all | 32 | 1024-1415 | 0.148 ± 0.021 | 0.010 ± 0.002 | 4.039 | 0.157 | 0.0775 | -0.0089 ± 0.0193 | 0.469 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | density q0 | 8 | 1024-1031 | 0.192 ± 0.049 | 0.015 ± 0.003 | 4.29 | 0.1402 | 0.0575 | 0.0516 ± 0.0398 | 0.75 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | density q1 | 8 | 1152-1159 | 0.120 ± 0.044 | 0.015 ± 0.004 | 3.558 | 0.1683 | 0.0924 | -0.0487 ± 0.0408 | 0.25 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | density q2 | 8 | 1280-1287 | 0.124 ± 0.030 | 0.005 ± 0.002 | 4.008 | 0.1496 | 0.0653 | -0.0256 ± 0.0409 | 0.25 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | density q3 | 8 | 1408-1415 | 0.157 ± 0.047 | 0.005 ± 0.002 | 4.301 | 0.17 | 0.0948 | -0.0131 ± 0.0298 | 0.625 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | lam 0.1 (member) | 32 | 1024-1415 | 0.130 ± 0.022 |  | 4.158 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | lam 0.19 (member) | 32 | 1024-1415 | 0.072 ± 0.018 |  | 3.911 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | epo-random32-strat | lam 0.37 (member) | 32 | 1024-1415 | 0.052 ± 0.015 |  | 3.874 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | rare-stratum q0 view | gcg-corpus | all | 32 | 1024-1055 | 0.308 ± 0.017 | 0.236 ± 0.017 | 7.97 | 0.1358 | 0.0647 | 0.1722 ± 0.0222 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | all | 32 | 1024-1415 | 0.298 ± 0.017 | 0.233 ± 0.014 | 7.461 | 0.157 | 0.0775 | 0.1412 ± 0.0157 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q0 | 8 | 1024-1031 | 0.351 ± 0.021 | 0.278 ± 0.020 | 8.299 | 0.1402 | 0.0575 | 0.2103 ± 0.0451 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q1 | 8 | 1152-1159 | 0.287 ± 0.027 | 0.243 ± 0.023 | 6.622 | 0.1683 | 0.0924 | 0.1185 ± 0.0205 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q2 | 8 | 1280-1287 | 0.278 ± 0.017 | 0.204 ± 0.021 | 7.842 | 0.1496 | 0.0653 | 0.1284 ± 0.0227 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-corpus-strat | density q3 | 8 | 1408-1415 | 0.278 ± 0.055 | 0.207 ± 0.042 | 7.082 | 0.17 | 0.0948 | 0.1078 ± 0.0218 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | rare-stratum q0 view | gcg-random32 | all | 32 | 1024-1055 | 0.216 ± 0.025 | 0.016 ± 0.002 | 13.334 | 0.1358 | 0.0647 | 0.0803 ± 0.0167 | 0.906 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | all | 32 | 1024-1415 | 0.203 ± 0.024 | 0.010 ± 0.002 | 13.422 | 0.157 | 0.0775 | 0.0461 ± 0.0194 | 0.719 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q0 | 8 | 1024-1031 | 0.284 ± 0.044 | 0.012 ± 0.003 | 13.517 | 0.1402 | 0.0575 | 0.1441 ± 0.0369 | 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q1 | 8 | 1152-1159 | 0.183 ± 0.049 | 0.018 ± 0.004 | 13.75 | 0.1683 | 0.0924 | 0.0150 ± 0.0423 | 0.75 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q2 | 8 | 1280-1287 | 0.148 ± 0.042 | 0.005 ± 0.002 | 13.283 | 0.1496 | 0.0653 | -0.0019 ± 0.0355 | 0.5 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen3-8b | sae | reported (stratified) | gcg-random32-strat | density q3 | 8 | 1408-1415 | 0.197 ± 0.052 | 0.005 ± 0.002 | 13.137 | 0.17 | 0.0948 | 0.0273 ± 0.0195 | 0.625 |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | epo-corpus | all | 32 | 0-31 | 0.43 ± 0.03 | 0.35 ± 0.03 | 3.03 |  |  |  |  | 0.5313 | 0.4611 | -0.1039 ± 0.0133 | 0.062 | 0.1052 | -0.0021 | 0.3222 ± 0.0221 | 1.0 | 0.5301 | 0.4065 | -0.1027 ± 0.0138 | 0.031 |
| qwen36-27b | realact |  | epo-corpus | lam 0.1 (member) | 32 | 0-31 | 0.42 ± 0.03 |  | 3.05 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | epo-corpus | lam 0.19 (member) | 32 | 0-31 | 0.40 ± 0.02 |  | 2.78 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | epo-corpus | lam 0.37 (member) | 32 | 0-31 | 0.38 ± 0.03 |  | 2.71 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact | SUPERSEDED partial run -- not a reported arm | epo-corpus-smoke | all | 25 | 0-24 | 0.41 ± 0.03 | 0.34 ± 0.03 | 2.93 |  |  |  |  | 0.5151 | 0.4416 | -0.1014 ± 0.0148 | 0.04 | 0.0808 | -0.019 | 0.3329 ± 0.0265 | 1.0 | 0.5119 | 0.3967 | -0.0982 ± 0.0159 | 0.04 |
| qwen36-27b | realact | SUPERSEDED partial run -- not a reported arm | epo-corpus-smoke | lam 0.1 (member) | 25 | 0-24 | 0.41 ± 0.03 |  | 2.95 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact | SUPERSEDED partial run -- not a reported arm | epo-corpus-smoke | lam 0.19 (member) | 25 | 0-24 | 0.38 ± 0.03 |  | 2.66 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact | SUPERSEDED partial run -- not a reported arm | epo-corpus-smoke | lam 0.37 (member) | 25 | 0-24 | 0.37 ± 0.03 |  | 2.61 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | epo-random32 | all | 32 | 0-31 | 0.22 ± 0.03 | -0.02 ± 0.02 | 5.12 |  |  |  |  | 0.5313 | 0.4611 | -0.3099 ± 0.0291 | 0.0 | 0.1052 | -0.0021 | 0.1162 ± 0.0208 | 0.812 | 0.5301 | 0.4065 | -0.3087 ± 0.0295 | 0.0 |
| qwen36-27b | realact |  | epo-random32 | lam 0.1 (member) | 32 | 0-31 | 0.19 ± 0.03 |  | 5.09 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | epo-random32 | lam 0.19 (member) | 32 | 0-31 | 0.14 ± 0.02 |  | 4.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | epo-random32 | lam 0.37 (member) | 32 | 0-31 | 0.09 ± 0.03 |  | 4.62 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | realact |  | gcg-corpus | all | 32 | 0-31 | 0.49 ± 0.03 | 0.35 ± 0.03 | 8.28 |  |  |  |  | 0.5313 | 0.4611 | -0.0426 ± 0.0136 | 0.281 | 0.1052 | -0.0021 | 0.3835 ± 0.0221 | 1.0 | 0.5301 | 0.4065 | -0.0415 ± 0.0139 | 0.312 |
| qwen36-27b | realact |  | gcg-random32 | all | 32 | 0-31 | 0.28 ± 0.03 | -0.02 ± 0.02 | 13.08 |  |  |  |  | 0.5313 | 0.4611 | -0.2486 ± 0.0292 | 0.0 | 0.1052 | -0.0021 | 0.1775 ± 0.0247 | 0.938 | 0.5301 | 0.4065 | -0.2474 ± 0.0292 | 0.0 |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | all | 32 | 1024-1415 | 0.19 ± 0.02 | 0.16 ± 0.01 | 2.56 |  |  |  |  | 0.1802 | 0.1316 | 0.0092 ± 0.0112 | 0.438 | 0.0218 | 0.0032 | 0.1677 ± 0.0156 | 0.969 | 0.162 | 0.1109 | 0.0275 ± 0.0119 | 0.594 |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | density q0 | 8 | 1024-1031 | 0.18 ± 0.04 | 0.15 ± 0.04 | 2.21 |  |  |  |  | 0.1553 | 0.083 | 0.0258 ± 0.0414 | 0.25 | 0.0191 | 0.0045 | 0.1619 ± 0.0378 | 0.875 | 0.1196 | 0.0441 | 0.0615 ± 0.0442 | 0.375 |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | density q1 | 8 | 1152-1159 | 0.25 ± 0.03 | 0.21 ± 0.02 | 2.69 |  |  |  |  | 0.2639 | 0.212 | -0.0139 ± 0.0142 | 0.25 | 0.0196 | 0.0039 | 0.2304 ± 0.0279 | 1.0 | 0.2444 | 0.1919 | 0.0056 ± 0.0109 | 0.375 |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | density q2 | 8 | 1280-1287 | 0.18 ± 0.02 | 0.15 ± 0.02 | 2.66 |  |  |  |  | 0.1684 | 0.1312 | 0.0111 ± 0.0104 | 0.625 | 0.0196 | 0.0018 | 0.1598 ± 0.0188 | 1.0 | 0.1585 | 0.1142 | 0.0210 ± 0.0089 | 0.875 |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | density q3 | 8 | 1408-1415 | 0.15 ± 0.03 | 0.13 ± 0.03 | 2.67 |  |  |  |  | 0.1332 | 0.1002 | 0.0139 ± 0.0082 | 0.625 | 0.0286 | 0.0026 | 0.1186 ± 0.0293 | 1.0 | 0.1253 | 0.0935 | 0.0218 ± 0.0100 | 0.75 |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | lam 0.1 (member) | 32 | 1024-1415 | 0.19 ± 0.02 |  | 2.56 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | lam 0.19 (member) | 32 | 1024-1415 | 0.18 ± 0.02 |  | 2.47 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | sae | reported (stratified) | epo-corpus-strat | lam 0.37 (member) | 32 | 1024-1415 | 0.16 ± 0.01 |  | 2.43 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | all | 32 | 1024-1415 | 0.03 ± 0.01 | 0.00 ± 0.00 | 4.46 |  |  |  |  | 0.1802 | 0.1316 | -0.1510 ± 0.0190 | 0.094 | 0.0218 | 0.0032 | 0.0074 ± 0.0051 | 0.531 | 0.162 | 0.1109 | -0.1328 ± 0.0181 | 0.094 |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | density q0 | 8 | 1024-1031 | 0.02 ± 0.00 | 0.00 ± 0.00 | 4.22 |  |  |  |  | 0.1553 | 0.083 | -0.1377 ± 0.0431 | 0.125 | 0.0191 | 0.0045 | -0.0016 ± 0.0023 | 0.375 | 0.1196 | 0.0441 | -0.1020 ± 0.0395 | 0.125 |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | density q1 | 8 | 1152-1159 | 0.02 ± 0.00 | 0.00 ± 0.00 | 4.22 |  |  |  |  | 0.2639 | 0.212 | -0.2487 ± 0.0183 | 0.0 | 0.0196 | 0.0039 | -0.0044 ± 0.0034 | 0.375 | 0.2444 | 0.1919 | -0.2292 ± 0.0247 | 0.0 |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | density q2 | 8 | 1280-1287 | 0.05 ± 0.02 | 0.00 ± 0.00 | 4.73 |  |  |  |  | 0.1684 | 0.1312 | -0.1181 ± 0.0301 | 0.0 | 0.0196 | 0.0018 | 0.0306 ± 0.0169 | 0.75 | 0.1585 | 0.1142 | -0.1083 ± 0.0276 | 0.0 |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | density q3 | 8 | 1408-1415 | 0.03 ± 0.01 | 0.01 ± 0.00 | 4.67 |  |  |  |  | 0.1332 | 0.1002 | -0.0995 ± 0.0370 | 0.25 | 0.0286 | 0.0026 | 0.0051 ± 0.0067 | 0.625 | 0.1253 | 0.0935 | -0.0916 ± 0.0329 | 0.25 |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | lam 0.1 (member) | 32 | 1024-1415 | 0.03 ± 0.01 |  | 4.43 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | lam 0.19 (member) | 32 | 1024-1415 | 0.02 ± 0.00 |  | 4.46 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | sae | reported (stratified) | epo-random32-strat | lam 0.37 (member) | 32 | 1024-1415 | 0.01 ± 0.00 |  | 4.14 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| qwen36-27b | sae | rare-stratum q0 view | gcg-corpus | all | 32 | 1024-1055 | 0.24 ± 0.01 | 0.15 ± 0.01 | 7.31 |  |  |  |  | 0.1308 | 0.0859 | 0.1100 ± 0.0166 | 0.906 | 0.0194 | 0.0039 | 0.2214 ± 0.0115 | 1.0 | 0.1112 | 0.065 | 0.1296 ± 0.0172 | 0.938 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | all | 32 | 1024-1415 | 0.23 ± 0.02 | 0.16 ± 0.01 | 7.15 |  |  |  |  | 0.1802 | 0.1316 | 0.0542 ± 0.0100 | 0.875 | 0.0218 | 0.0032 | 0.2126 ± 0.0164 | 1.0 | 0.162 | 0.1109 | 0.0724 ± 0.0101 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q0 | 8 | 1024-1031 | 0.23 ± 0.04 | 0.15 ± 0.04 | 6.94 |  |  |  |  | 0.1553 | 0.083 | 0.0790 ± 0.0348 | 0.75 | 0.0191 | 0.0045 | 0.2151 ± 0.0357 | 1.0 | 0.1196 | 0.0441 | 0.1147 ± 0.0322 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q1 | 8 | 1152-1159 | 0.29 ± 0.03 | 0.21 ± 0.02 | 6.86 |  |  |  |  | 0.2639 | 0.212 | 0.0240 ± 0.0115 | 0.75 | 0.0196 | 0.0039 | 0.2683 ± 0.0265 | 1.0 | 0.2444 | 0.1919 | 0.0435 ± 0.0077 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q2 | 8 | 1280-1287 | 0.22 ± 0.02 | 0.15 ± 0.02 | 7.58 |  |  |  |  | 0.1684 | 0.1312 | 0.0551 ± 0.0101 | 1.0 | 0.0196 | 0.0018 | 0.2039 ± 0.0212 | 1.0 | 0.1585 | 0.1142 | 0.0650 ± 0.0081 | 1.0 |
| qwen36-27b | sae | reported (stratified) | gcg-corpus-strat | density q3 | 8 | 1408-1415 | 0.19 ± 0.04 | 0.13 ± 0.03 | 7.2 |  |  |  |  | 0.1332 | 0.1002 | 0.0586 ± 0.0109 | 1.0 | 0.0286 | 0.0026 | 0.1632 ± 0.0394 | 1.0 | 0.1253 | 0.0935 | 0.0665 ± 0.0157 | 1.0 |
| qwen36-27b | sae | rare-stratum q0 view | gcg-random32 | all | 32 | 1024-1055 | 0.07 ± 0.01 | 0.01 ± 0.00 | 13.18 |  |  |  |  | 0.1308 | 0.0859 | -0.0626 ± 0.0195 | 0.375 | 0.0194 | 0.0039 | 0.0488 ± 0.0096 | 1.0 | 0.1112 | 0.065 | -0.0430 ± 0.0194 | 0.5 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | all | 32 | 1024-1415 | 0.08 ± 0.01 | 0.00 ± 0.00 | 13.35 |  |  |  |  | 0.1802 | 0.1316 | -0.0959 ± 0.0155 | 0.188 | 0.0218 | 0.0032 | 0.0625 ± 0.0095 | 0.938 | 0.162 | 0.1109 | -0.0777 ± 0.0147 | 0.156 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q0 | 8 | 1024-1031 | 0.07 ± 0.02 | 0.01 ± 0.00 | 13.57 |  |  |  |  | 0.1553 | 0.083 | -0.0869 ± 0.0340 | 0.375 | 0.0191 | 0.0045 | 0.0492 ± 0.0184 | 1.0 | 0.1196 | 0.0441 | -0.0513 ± 0.0278 | 0.375 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q1 | 8 | 1152-1159 | 0.10 ± 0.03 | 0.00 ± 0.00 | 13.27 |  |  |  |  | 0.2639 | 0.212 | -0.1647 ± 0.0252 | 0.0 | 0.0196 | 0.0039 | 0.0796 ± 0.0295 | 0.875 | 0.2444 | 0.1919 | -0.1452 ± 0.0324 | 0.0 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q2 | 8 | 1280-1287 | 0.08 ± 0.01 | 0.00 ± 0.00 | 13.31 |  |  |  |  | 0.1684 | 0.1312 | -0.0871 ± 0.0202 | 0.0 | 0.0196 | 0.0018 | 0.0617 ± 0.0096 | 1.0 | 0.1585 | 0.1142 | -0.0772 ± 0.0177 | 0.0 |
| qwen36-27b | sae | reported (stratified) | gcg-random32-strat | density q3 | 8 | 1408-1415 | 0.09 ± 0.01 | 0.00 ± 0.00 | 13.27 |  |  |  |  | 0.1332 | 0.1002 | -0.0450 ± 0.0312 | 0.375 | 0.0286 | 0.0026 | 0.0596 ± 0.0147 | 0.875 | 0.1253 | 0.0935 | -0.0371 ± 0.0266 | 0.25 |

**EPO buys fluency at a small cosine cost, which is the whole point of the lambda grid.** On 27B
realact, `epo-corpus` reaches **0.43 ± 0.03** at NLL **3.03** against `gcg-corpus`'s **0.49 ± 0.03**
at NLL **8.28**: ~0.06 of cosine for **5.3 nats**. The per-member rows trace the front directly —
lambda 0.1 → 0.42 at NLL 3.05, 0.19 → 0.40 at 2.78, 0.37 → 0.38 at 2.71 — monotone in both, exactly
as a Pareto sweep should be. So the "reachability ceiling" is really two ceilings: what a string can
reach at any fluency (GCG, NLL 8-13, gibberish) and what a *readable* string can reach (EPO, NLL
2.4-3.1).

**Against the primary the conclusion changes with the optimiser.** On realact, GCG-corpus is 0.043
below the inverter and EPO-corpus is **0.104** below (winning 6.2% of directions) — the inverter
beats any fluent 32-token string we can find. On stratified sae, EPO-corpus is level with the
inverter (**+0.0092 ± 0.0112**, 43.8% wins) where GCG-corpus was clearly ahead (+0.0542 ± 0.0100,
87.5%): once the optimised string has to be fluent, the SAE advantage over the inverter mostly
disappears. Random-init EPO is far behind everywhere (27B sae -0.1510 ± 0.0190, 9.4% wins).

**One arm is labelled, not dropped.** `qwen36-27b/realact/epo-corpus-smoke` has **25 of 32**
directions and is superseded by `epo-corpus`; it carries `view = SUPERSEDED partial run -- not a
reported arm` and its own `dirs` column says 25. Its rows are real and are kept visible, but nothing
should average them in — note its MAEMM column reads 0.5151 rather than 0.5313 precisely because it
is a different subset of directions.

## NLA baseline (2026-09-20, branch `arb/nla` off Ari's `7f3b511`)

`rollouts_nla` runs the public NLA activation verbalizer `ceselder/qwen3.6-27b-nla-av` (merged
full Qwen3.6-27B, snapshot `def1421d`) over the held-out directions and writes the `rollouts_hf`
schema, so `score` consumes it unchanged. Input amplitude `--amp exact` (default: `mu + t·u` with
`||mu + t·u|| = act_norm`; rows without an `act_norm` fall back to `mu + r·u`, r = layer-42 median
norm 93.26) and `raw` (`r·u`). Every text below hit the 64-token cap (eos rate 0, the `</explanation>`
close is never reached at 64) -- the card's reference script runs 200 greedy tokens.

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-20 | HF fetch: `ceselder/qwen3.6-27b-nla-av` @ `def1421d` | `modal run features/fetch_hf.py --repo … --revision def1421d…` (dry-run first) | 247.3 s | ~$0 (CPU) | 53.81 GB, single snapshot | — |
| 2026-09-20 | `check` 27B with the `type: nla` entry (CPU) | `--product check --base qwen36-27b` | 19.5 s | ~$0 | NLA prompt **112 tokens, marker 158983 at 93, neighbours 29/510**, sidecar asserted (layer 42, d 5120, norm none, T=1.0 top_p 0.95 top_k 20) | — |
| 2026-09-20 | `rollouts_nla` v1 realact rows 0-7 x 4, amp exact | `--product rollouts_nla --base qwen36-27b --maemm qwen36-27b/2026-07-14_nla-av --set 2026-09-16_v1 --rows 0-7 --n 4` | 195.1 s | **$0.2461** | 32 rows, all 8 rows solved `exact` (in_norm 73.4-99.5); marker ‖h‖ at block 1 = 588.0 (observation only); 134 gen tok/s | throughput is well below the MAEMM's ~340 tok/s at the same 32 rows/call: 112- vs 103-token prompt does not explain it, plausibly the 64-token cap with no eos (every row decodes to the cap) -- unmeasured |
| 2026-09-20 | `rollouts_nla` sae2m_2k rows 0-7 x 4, amp exact | same, `--set 2026-09-20_sae2m_2k` | 116.5 s | **$0.1469** | 32 rows, all 8 fell back to `mu` (`no_act_norm`, in_norm ~114-116) | — |
| 2026-09-20 | `rollouts_nla` sae2m_2k rows 0-7 x 4, amp raw (VARIANT) | same, `--amp raw` → `variants/2026-09-20_sae2m_2k__amp-raw/` | 97.5 s | **$0.1230** | 32 rows, `raw` on all 8 | — |
| 2026-09-20 | `score` rl-last16 v1 rows 0-7 (`--sae qwen36-27b/l42-1b`) | `--product score --maemm …rl-last16-lr5e-7 --set 2026-09-16_v1 --rows 0-7 --sae qwen36-27b/l42-1b` | ~85 s | ~$0.11 (est.) | **FAILED at the write**: `scores/2026-09-16_v1` already exists (Ari's full 1024-row score dir) | the existing dir is what the comparison below uses; the OutDir refusal fires AFTER the scoring wall was spent -- a pre-flight existence check would save the $0.11 |
| 2026-09-20 | `score` rl-last16 sae2m_2k rows 0-7 (`--no-sae`) | `… --set 2026-09-20_sae2m_2k --rows 0-7 --no-sae` | 85.4 s | **$0.1078** | mean cos 0.0229 | — |
| 2026-09-20 | `score` NLA v1 rows 0-7 (`--sae qwen36-27b/l42-1b`) | `… --maemm qwen36-27b/2026-07-14_nla-av --set 2026-09-16_v1 --rows 0-7 --sae qwen36-27b/l42-1b` | 138.2 s | **$0.1742** | mean cos **0.1157**, bo4 0.1838 | `--sae` is the new flag (commit `1a16277`): the 27B carries two SAEs since Ari's `sae2m` entry and `score` could not run on it at all |
| 2026-09-20 | `score` NLA sae2m_2k rows 0-7 (`--no-sae`) | same set, `--no-sae` | 83.1 s | **$0.1048** | mean cos 0.0206 | — |
| 2026-09-20 | `score` NLA sae2m_2k amp-raw variant | `--product score --base qwen36-27b --set 2026-09-20_sae2m_2k --rollouts-dir /vol/maemms/…/variants/2026-09-20_sae2m_2k__amp-raw --no-sae` | 134.4 s | **$0.1695** | mean cos 0.0209 | `--rollouts-dir` path works unchanged for the variant layout |

**Total GPU spend for the NLA smoke: ≈ $1.18** (of the $10 authorised).

### NLA vs rl-last16 on the same rows (mean over 8 targets; cosine on the clean base, uncentred)

| set / rows | model | n | mean cos | best-of-4 | mean len | eos |
|---|---|---|---|---|---|---|
| `2026-09-16_v1` realact 0-7 | `2026-09-18_rl-last16-lr5e-7` | 64 | **0.3712** | 0.4201 (disjoint groups of 4) | 57.8 | 0.71 |
| `2026-09-16_v1` realact 0-7 | NLA-av, amp exact | 4 | **0.1157** | 0.1838 | 64.0 | 0.00 |
| `2026-09-20_sae2m_2k` 0-7 | `2026-09-18_rl-last16-lr5e-7` | 4 | 0.0229 | 0.0280 | 25.8 | 1.00 |
| `2026-09-20_sae2m_2k` 0-7 | NLA-av, amp exact (→ mu fallback) | 4 | 0.0206 | 0.0234 | 64.0 | 0.00 |
| `2026-09-20_sae2m_2k` 0-7 | NLA-av, amp raw | 4 | 0.0209 | 0.0241 | 64.0 | 0.00 |

Per-row NLA realact cosines: 0.024, 0.120, 0.228, 0.186, 0.380, −0.010, −0.052, 0.049 against
rl-last16's 0.248, 0.483, 0.495, 0.175, 0.621, 0.211, 0.244, 0.492. The NLA texts READ as correct
descriptions of the realact spans (row 0's span is a librarian's "Less is More" conference note; the
NLA writes "Library conference notes ... 'Less is More' section"), so the low cosine is the metric's
verdict on a description of the context versus a reconstruction of the token, not a failed run. On
the 2M-SAE encoder columns both models are at the noise floor on these 8 rows, `mu` vs `raw` making
no difference (0.0206 vs 0.0209). Eight rows is a smoke, not an estimate.

Not run: `sae_self` -- the v1 rows 0-7 are realact, and the sae2m_2k rows carry family `sae2m_enc`,
which `sae_self.FAMILY = "sae"` skips (Ari's label; not changed here).

## NLA at 200 tokens, the 2M-SAE autointerp smoke, and the activation smoke (2026-09-21, `arb/nla`)

Decisions applied (Tomáš 2026-09-21): NLA generates at its native `max_new` 200, T=1, and is scored
in a 256-token window (`score_max_length` carried in the rollouts summary; every other arm stays at
95); `nla.amp` default is `raw`. All smoke outputs live under `--root /vol/tmp/nla-smoke` (first 8
2M features + the 131k set) and `/vol/tmp/nla-smoke-x` (8 more 2M features); inputs were copied
there file by file (`modal volume cp` has no `-r` on this V1 volume), nothing under `/vol/base` or
`/vol/maemms` was written. HF cache stays at `/vol/hf`.

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-21 | `rollouts_nla` 2M rows 0,1,4-6,8-10 x 4 @ 200 tok, amp raw / exact | `--product rollouts_nla ... --set 2026-09-20_sae2m_2k --root /vol/tmp/nla-smoke --rows 0,1,4-6,8-10 --n 4 [--amp exact]` | 252.6 / 248.9 s | $0.3185 / $0.3139 | eos rate 0.97, `</explanation>` closed on 31/32, mean 182 tok, 240 gen tok/s | 240 tok/s at 200 tokens vs 134 at 64: the 64-token runs were load-dominated |
| 2026-09-21 | `score` NLA raw (`--sae sae2m`, encoder-only) / exact variant | `--product score --maemm <nla> --engine hf --sae qwen36-27b/sae2m ...` / `--rollouts-dir .../variants/...__amp-exact` | 185.5 / 229.7 s | $0.2339 / $0.2896 | T=257, 0/32 rows at the 256 truncation; mean cos raw 0.0293, bo4 0.0342 | the 2M SAE encoder-only (43 GB) + base fit the H200 |
| 2026-09-21 | `random_pool` / `examples_4m` sae2m (2000 features each) | `autointerp --stage random_pool` / `--stage examples_4m` | 195.9 / 1501.9 s | $0.2470 / $1.8940 | 2048 windows; 4M prefix 237,980 windows, top-128 per feature | `--rows` restricts nothing in these two stages (all 2000 tested features are encoded); the cost is the corpus forward either way |
| 2026-09-21 | `sae_self` NLA / rl-last16 on the 8 2M rows | `autointerp --stage sae_self --maemm <key> --engine hf --sae qwen36-27b/sae2m --rows ...` | 285.7 / 232.7 s | $0.3602 / $0.2934 | argmax 32/32, CSR checks 0 mismatches; fire fraction 0.125 / 0.094 | FIRST attempt died: `sae_self` assumed `score` covered the whole set (`reshape 32 into (2000,4)`); fixed in `34d1104` |
| 2026-09-21 | `build` C4 + NLA, 8 features (CPU) | `--stage build --arms C4,NLA --allow-short --build-dir 2026-09-21_build ...` | 18.9 s | $0 | 8 features; positives from `examples_4m` (no scan `examples/` for sae2m, `f1ffa07`); `n_short_draw1` 7, min 10 positives | first attempt died on the missing `examples/2323.jsonl` |
| 2026-09-21 | `run` detection, arms C4 / NLA / NLA-desc (CPU + API) | `--stage run --build-dir 2026-09-21_build --run-dir 2026-09-21_nla-smoke --arms C4,NLA,NLA-desc --scorers detection --path sync` | 34.9 s | **API $0.62** (16 explainer + 162 detection calls) | table below | — |
| 2026-09-21 | activation smoke extras: NLA 131k (16 v1 sae rows) rollouts/score/sae_self; rl-last16 2M score + sae_self (2 roots); NLA-x rollouts/score/sae_self; `examples_4m` root x | 11 launches | — | **$4.25** | tables below | root x exists only because a second `rollouts_nla --rows` run into the same `rollouts/<set>.jsonl` would have replaced the first 8 rows (the product writes one file per set, it does not merge) |

GPU spend of this section: **$7.6** (follow-on cap $15; activation-smoke extras $4.25 of its $10);
cumulative `arb/nla` GPU ≈ $8.9 incl. the $0.11 wasted rl-last16 v1 score of 09-20. API $0.62.

### Detection balanced accuracy, 8 features of the 2M SAE (rows 0,1,4-6,8-10 of `2026-09-20_sae2m_2k`)

| arm | what the scorer judges with | mean bal. acc | median | TPR | TNR |
|---|---|---|---|---|---|
| C4 (reference: 16 corpus top windows of the 4M prefix → explainer) | Sonnet explanation | 0.477 | 0.487 | 0.35 | 0.61 |
| NLA (A: 4 NLA texts, peak-marked → explainer) | Sonnet explanation | 0.499 | 0.500 | 0.02 | 0.98 |
| NLA-desc (B: the NLA text itself, best-firing rollout) | NLA text | 0.503 | 0.500 | 0.01 | 1.00 |

Per feature (C4 / NLA / NLA-desc): 2323 .425/.5/.5, 2386 .475/.475/.525, 7784 .508/.567/.5,
9083 .45/.5/.5, 10750 .533/.475/.5, 11663 .5/.5/.5, 14719 .5/.5/.5, 15367 .425/.475/.5.
**Every arm is at chance, the corpus reference included.** The C4 explainer describes these
features as "highly predictable continuation tokens / function words" (2323, 9083) -- the 2M
features drawn here (≥ 200 gated fires at 16M, i.e. the sparse end) read as position/predictability
features that no 16-example description separates from zero-activation windows; arms A and B say
"no" to nearly everything (TNR ≈ 1, TPR ≈ 0). This smoke therefore does NOT discriminate the two
NLA modes; it says the test itself has no headroom on this feature draw. Positives are also weak:
`examples_4m` peaks are 0.54-1.0 of the 16M `max_act`, 7 of 8 features short of 20 positives.

### Activation smoke: raw peak of the target feature, median over features

| SAE (set) | source | median peak / corpus peak | features firing |
|---|---|---|---|
| 2M `sae2m` (2k set, 16 features) | rl-last16 rollouts (n=4, 64 tok) | **0.32** | 4/16 |
| | NLA texts (n=4, 200 tok, raw) | **0.33** | 5/16 |
| | corpus top-16 windows, 4M prefix | 0.83 | 16/16 |
| 131k `l42-1b` (v1, 16 sae rows) | old primary rollouts (n=64) | 1.05 | 16/16 |
| | old primary, first 4 rollouts | **0.95** | 15/16 |
| | NLA texts (n=4, 200 tok, raw) | **0.53** | 14/16 |
| | corpus top-16 windows, 16M docmax | 1.00 | 16/16 |

Corpus peak = `sae/<sae>/max_act.f16` (16M). On the 131k SAE a MAEMM at n=4 reaches the corpus
peak (0.95) and the NLA half of it (0.53); on the 2M SAE BOTH sit at a third of the peak and fire
on a quarter of the features, while the corpus search itself only reaches 0.83 on a 4M prefix. The
~26%-of-peak is therefore not specific to the MAEMM: an independently trained verbalizer lands at
the same place on the 2M dictionary and at 2x that on the 131k one. Caveat: 4M-prefix corpus on
the 2M side biases its corpus ratio DOWN, i.e. the MAEMM/NLA ratios are, if anything, flattered.
Per-feature tables: `reconstruction/act_smoke.py --data <mirror>` output, kept in the session
scratchpad (act_smoke-2m.md, act_smoke-131k.md); the 2M rows with peaks ≥ 8 (1635672, 1944579,
stratum 3) are the only ones any generated text drives to the gate reliably.

---

## 2026-09-21 — branch `evals/pipeline`: the conventions layer

CPU work first, then the §1.6 paid smoke, which WAS run (11 H200 calls, **$1.89**). Everything the
smoke wrote went under a NEW set name and new tagged stems; `2026-09-16_v1` and every existing
product were read and not modified, and nothing was deleted.

| date | item | command | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-21 | unit smoke after the conventions layer | `uv run paper-evals/precompute/unit_smoke.py` | 2.6 s | $0 | **36/36 checks passed** (29 before; +6 for the layer, +1 for the spawn/main mirror) | — |
| 2026-09-21 | mutation battery on the six new checks | four deliberate defects injected one at a time, smoke re-run | ~40 s | $0 | 4/4 caught, each by the check that owns it | see below |

Mutations and what fired, so the checks are known to be capable of red:

| mutation | check that failed | message |
|---|---|---|
| `dirs_for` subtracts mu from EVERY row, not only the centrable ones | `check_storage_contract` | `row 2 (random) under mu: max \|d\| 4.81e-02 -- a non-centrable row was treated as the other` |
| the centred einsum drops `- mu` (one-sided cosine) | `check_two_cosines` | `cos_centred differs from the direct einsum` |
| `sae_rows_of` ignores the row's `sae_key` | `check_sae_key_selector` | `sae_key filter picked [0, 1, 2, 4]; the 131k row must not be in it` |
| the `storage: unit` mismatch assert is disabled | `check_unit_set_refuses` | `dirs_for served a 'storage: unit' set under the WRONG mean` |

### The three refusals this layer adds, verbatim

They are deliberate and they will be met. Each is reproducible on CPU with no volume mounted.

**1. A checkpoint's mean does not match the set's stored one.** Met by
`rollouts_* --maemm <rl-last16> --set 2026-09-16_v1`:

```
<root>/base/qwen36-27b/heldout/2026-09-16_v1 is `storage: unit` (config.yaml
heldout.2026-09-16_v1) and its 'realact' rows are stored under
<root>/base/qwen36-27b/stats/mu.f32, but this run asks for
mu=/vol/archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy. A stored unit direction cannot be
re-centred -- unit(act) and mu do not give unit(act - mu) without ||act||. Re-derive the set at
`storage: raw` (`--product targets --re-derive <set>`), or run at the mean it was built with and
say so.
```

*What it means*: `rl-last16` was trained on `whiten_mu`-centred input and `2026-09-16_v1` stores
`stats/mu.f32`-centred directions with no `act.f32` to re-derive from. Before this layer the run
went ahead and the mismatch was invisible. **Fix**: re-derive the set (`--re-derive`), or pass
`--mu base/{base}/stats/mu.f32` and wear the DEVIATION line in the product README.

**2. A `--dirs-from` directory states no storage contract.**

```
<root>/elsewhere carries no storage.json and 'elsewhere' is not a set declared in config.yaml, so
nothing states whether its vecs.f16 is centred. Declare it under `heldout:` (storage / mu_stored /
family_mu) or re-draw the set, which writes the contract itself.
```

*What it means*: nothing in that directory says whether its `vecs.f16` is raw or centred, and the
two are the same bytes to a reader. **Fix**: declare it under `heldout:`, or re-draw it — any set
`targets` writes carries its own `storage.json`.

**3. A checkpoint declares `mu: unknown`.** Met by any product with
`--maemm qwen36-27b/2026-09-10_rl-8x2048-full` and no `--mu`:

```
maemm 'qwen36-27b/2026-09-10_rl-8x2048-full' declares `mu: unknown`: its training convention is on
the agenda and NOT on the record, so nothing here will pick one for it. Pass --mu explicitly (a
path, or `none`) and the choice is recorded as a deviation in the product README. That is what the
two-arm reconciliation in SMOKES.md settles.
```

*What it means*: the old primary's training convention is genuinely unsettled (see the note beside
its config entry). **Fix**: pass `--mu none` or `--mu base/{base}/stats/mu.f32` — which is exactly
the two-arm reconciliation the smoke below runs.

### The §1.6 paid smoke — RUN 2026-09-21, **$1.89 total**

11 H200 calls under a $5 target / $8 cap. Everything wrote under a NEW set name; `2026-09-16_v1`
and every existing product were read and not modified.

| step | product | wall | cost | outcome |
|---|---|---|---|---|
| — | `targets --re-derive` (1st try) | ~2 min | ~$0.25 est. | **FAILED**, my bug: see "the arity defect" below. Crashed before its own cost line |
| 0 | `targets --re-derive 2026-09-16_v1 --set 2026-09-21_v1raw` | 155.8 s | $0.1964 | 1,536 rows, 45.3 MiB, all three re-derive checks passed |
| 1 | `rollouts_hf` old primary `--mu none --run-tag mu-none` | 111.1 s | $0.1401 | 24 rollouts |
| 2 | `rollouts_hf` old primary `--mu base/{base}/stats/mu.f32 --run-tag mu-stats` | 148.8 s | $0.1876 | 24 rollouts |
| 3 | `rollouts_hf` `rl-last16` (config `mu:` = whiten_mu) | 160.0 s | $0.2018 | 24 rollouts |
| 4 | `score` old primary, mu-none | 99.2 s | $0.1251 | no `cos_centred` (mu is none) |
| 5 | `score` old primary, mu-stats | 78.7 s | $0.0992 | both cosines |
| 6 | `score` `rl-last16` | 90.8 s | $0.1145 | both cosines |
| 7 | `sae_self` `rl-last16`, normal path | 86.6 s | $0.1092 | 3/3 checks |
| 8 | `rollouts_nla --amp exact` | 169.7 s | $0.2141 | writes the `--rollouts-dir` layout |
| 9 | `score --rollouts-dir` (no `--maemm`) | 116.7 s | $0.1472 | **D11 prerequisite** |
| 10 | `sae_self --rollouts-dir` (no `--maemm`) | 85.7 s | $0.1080 | **D11**, 3/3 checks |

HF rather than vLLM deliberately: 24 rollouts do not repay a 27B vLLM engine init.

**`targets --re-derive`, the migration.** $0.1964 / 155.8 s against the plan's measured basis of
$0.1864 / 147.8 s. `re_derive.json` on the volume:

| family | mean it was re-centred under | n | min cos vs the old `vecs.f16` |
|---|---|---|---|
| realact | `/vol/base/qwen36-27b/stats/mu.f32` | 512 | **0.99999917** |
| random | none | 512 | 0.99999923 |
| sae | none | 512 | 0.99999928 |

1,536 rows; every field of `(family, id, stratum, doc, part, part_row, p, L, act_norm)` identical
row for row; `span_text` identical except the **21 clamped rows**, each new text a suffix of the
old. So it is the same draw re-forwarded, not a re-sample, and D4 is the only textual difference.
Two recorded facts confirmed en route: 21/512 clamped (the review's predicted count) and
cos(mu_512, stats/mu) = **0.9773** against observations §2's 0.977; `‖stats/mu‖` = 67.9.

### The two old-primary arms — what settles `mu: unknown`

Same rows, same scorer, same target (`cos` is against `unit(act)` for both, because on a raw set
the uncentred cosine's target does not depend on `--mu`). The ONLY difference is which direction
the checkpoint was handed at generation. realact rows 0-3, n = 4:

| row | `--mu none` mean_cos | `--mu stats_mu` mean_cos | `--mu none` bo_4 | `--mu stats_mu` bo_4 |
|---|---|---|---|---|
| 0 | 0.8089 | **0.8899** | 0.8615 | **0.8949** |
| 1 | 0.8127 | **0.8608** | 0.8379 | **0.9303** |
| 2 | **0.9314** | 0.8855 | **0.9423** | 0.9397 |
| 3 | 0.9509 | **0.9594** | 0.9720 | **0.9705** |
| **mean** | 0.8760 | **0.8989** | 0.9034 | **0.9339** |

**`stats_mu` wins**: +0.023 mean_cos, +0.030 bo_4, and on 3 of 4 rows. That is the direction the
record already pointed (Celeste's original convention was a centred target) and against Tomáš's
recollection. **n = 4 rows × 4 rollouts is far too small to declare it**, so `config.yaml` keeps
`mu: unknown` and this table is the evidence for widening the arms before it is changed.

Internal consistency: the two `sae` rows score identically across the arms (0.0056 / 0.0684 to
4 dp) — an encoder column is not centrable, so both arms injected the same vector for them.

**The plan's §1.6 gate 2 cannot be evaluated by this run, and not only because of size.** The
stored 0.5076 is `cos(h, unit(act − stats_mu))` — target centred, scorer raw, Celeste's asymmetry.
On a `storage: raw` set `score` produces `cos` (neither side centred) and `cos_centred` (both), and
**not** that third, asymmetric statistic. So the historical number is not reproducible from a raw
set by any flag combination. Either the gate is restated against `cos_centred`, or a legacy set is
kept for it, or `score` grows a third column. Flagged, not decided.

### `cos_centred`, first numbers

Written only when the run centres on something, which is the intended behaviour: the `--mu none`
and NLA directories carry no `cos_centred.f16` at all, and `rows.json` records the mu path.

| arm | mu recorded in `rows.json` | realact mean of `mean_cos_centred` |
|---|---|---|
| old primary, stats_mu | `/vol/base/qwen36-27b/stats/mu.f32` | 0.7724 |
| `rl-last16`, whiten_mu | `/vol/archive/gavento-1/data/qwen3.6-27b/whiten_mu.npy` | 0.7518 |

Every `sae` row is absent from the centred aggregates (`n_centred` unset), never a one-sided
number — the NaN rule holding on real data.

### `sae_self`, both paths

| path | argmax agreement | cos vs stored, max abs | CSR value / membership mismatches | csr_checked |
|---|---|---|---|---|
| `rl-last16`, normal | 8/8 | 1.5e-05 | 0 / 0 | true |
| NLA, `--rollouts-dir` (**D11**) | 8/8 | 1.0e-03 | 0 / 0 | true |

Both on the 131k dictionary, rows 1024-1025, features 341 and 845, gate 1.5846. The D11 path was
exercised on real data because `rollouts_nla --amp <non-default>` already writes
`rollouts.jsonl` + `rollouts.summary.json` into a variant directory — exactly the `--rollouts-dir`
layout — so no extra producer was needed. Feature 845 fires on 4/4 `rl-last16` rollouts
(mean peak 2.93 against corpus peak 6.96) and 2/4 NLA texts; feature 341 (corpus peak 44.7) fires
on neither.

### The arity defect, and what it cost

The first `targets --re-derive` died after a 148 s H200 forward with

```
ValueError: not enough values to unpack (expected 3, got 2)
```

`_realact` was given a third return value on this branch and its `return` statement was not
updated. Nothing on CPU could see it — `targets.run` needs weights, a corpus and a GPU. ~$0.25,
estimated from wall; the product crashed before its own cost line, and left a temp directory that
the rerun cleaned up by itself. `unit_smoke.check_return_arities` now checks that class on CPU for
the nine loaders this branch kept moving.

A second defect surfaced the same way at step 2:

```
AssertionError: /vol/maemms/.../rollouts/2026-09-21_v1raw.jsonl already exists;
refusing to overwrite without --force
```

`--run-tag` had been threaded through `rollouts_vllm`, `score` and `sae_self` but not
`rollouts_hf`, whose stem IS the bare set name and which spelled the path directly instead of
calling `rollout_stem`. Both old-primary arms therefore claimed one file, and only the
pre-existing overwrite guard stood between the second and the first. Fixed there and in
`rollouts_nla`'s matching branch.

**Leftover to remove** (not deleted here; deletion was not in scope):
`/vol/maemms/qwen36-27b/2026-09-10_rl-8x2048-full/rollouts/2026-09-21_v1raw.jsonl` +
`.summary.json` — the first arm's output under the pre-fix untagged name. It is the `mu: none`
arm, but its name does not say so; the tagged `…__mu-none.jsonl` beside it is the one to read.

### Run-scale decisions recorded 2026-09-21 (Tomáš), for the runbooks that follow

| eval | size | note |
|---|---|---|
| autointerp | **32 features per SAE** | `--n-feat 32`; down from the 256/128 of the 09-21 plan |
| GCG/EPO | **16 directions** | `--rows` a 16-row slice; EPO measured ~870 s per 27B direction |
| patchscopes | **Ari's implementation (7f3b511)** | do not write another one |
| OOD | **1/4 of the design's size** | scales the design's ≈$88 accordingly |

---

## 2026-09-21 — branch `evals/pipeline`: the eval-1 frozen target blocks

Five set directories built, **$0.131 total**, all of it the 2M draw. Everything wrote under a NEW
set name; nothing on the volume was deleted, replaced or rewritten. `--product unit` inside the
image (42/42, the image's own selfcheck) ran before the first launch, and `--product check` after
the last.

| date | item | command (abbreviated) | wall | cost | result | discrepancies |
|---|---|---|---|---|---|---|
| 2026-09-21 | unit smoke, local, after the v3 writers | `uv run paper-evals/precompute/unit_smoke.py` | 3.0 s | $0 | **43/43** (40 before; +3 for the recovery, the column reader and the set check) | — |
| 2026-09-21 | mutation battery on the three new checks | six deliberate defects, one at a time | ~2 min | $0 | **6/6 caught**, each by the check that owns it | table below |
| 2026-09-21 | `unit` in the image | `--product unit` | 21.8 s | ~$0 | 42/42 (before the set check landed) | — |
| 2026-09-21 | `heldout_v3 --block realact` | `--product heldout_v3 --base qwen36-27b --block realact --set 2026-09-21_v3_realact` | 57.5 s | ~$0 | 512 rows, 15.3 MiB, raw; 0 solve fallbacks | 3 rows ambiguous — below |
| 2026-09-21 | `heldout_v3 --block realact_long` | `… --block realact_long --set 2026-09-21_v3_realact_long` | 47.4 s | ~$0 | 512 rows, 5.1 MiB, `unit` + `family_mu: unknown` | — |
| 2026-09-21 | `heldout_v3 --block subspace` | `… --block subspace --set 2026-09-21_v3_subspace` | 57.6 s | ~$0 | 1,024 rows (bsf 512 + jlens 512), 10.3 MiB, `dirs_only` | — |
| 2026-09-21 | `heldout_v3 --block ctrl` | `… --block ctrl --set 2026-09-21_v3_ctrl --dirs-from …/2026-09-21_v1raw --rows 512-1535` | 5.5 s | ~$0 | 1,024 rows (random 512 + sae 512), 30.2 MiB, raw, each row carrying `src_set`/`src_row` | — |
| 2026-09-21 | `draw_sae2m --sides enc,dec` | `--product draw_sae2m --sae qwen36-27b/sae2m --set 2026-09-21_v3_sae2m --n 512 --stratified --seed 20260921 --sides enc,dec --include <the 64>` | 103.7 s | **$0.1308** | 1,024 rows = 512 features × {enc, dec}, paired row for row; the 64 of `2026-09-21_sae2m_64` nested; 128 per stratum; 413 fit / 99 report | cuts IDENTICAL to the 64-draw's; eligible 99,882 = 99,946 − the 64 forced out of the pool |
| 2026-09-21 | `check`, now opening the directories | `--product check --base qwen36-27b` | 24.0 s | ~$0 | all five v3 sets `ok`, plus `2026-09-16_v1` and `2026-09-21_v1raw`; `2026-09-21_sae2m_64` correctly `absent` (it lives under `/vol/tmp/sae-smoke64`) | — |
| 2026-09-21 | her 512 through `infra/check_v2_targets_overlap.py` | a driver reusing its `analyse` primitives on her `pool_target_text` | ~1 min | ~$0 | **void, and reported as void** — see below | — |

### U1, settled at $0: her `pool_act_norm` is `‖act‖`

Numbers and the three readings are in `features/README.md`. Short form: her own pool mint
statistic is 91.2669 against the 512's median 90.48; our layer-42 residual-norm quantiles are
76.38 / 93.26 / 109.26 against her 75.21 / 90.48 / 105.67, where the other reading would put her
MEDIAN activation above our 95th percentile; and a mu-orthogonal residual at ‖act‖ 90.48 has
‖act−mu‖ 60.52 against the solve's 59.95. **Branch (a): raw recoverable, no re-forward, $0.**

Read back off the bytes on the volume: `‖act.f32‖` vs her `pool_act_norm` **max |d| 2.4e-05**, and
`unit(act.f32 − whiten_mu)` vs her shipped `direction` **min cos 1.0000000000** over all 512.

**Rows 26, 32 and 360** have `pool_act_norm < ‖mu‖` and `mu·u < 0`, so both roots are positive and
two raw activations meet the constraint. The larger is taken and the row is flagged
`exact_ambiguous`. It cannot move anything read at her own mean; it can move `act.f32` itself.

### The overlap run the brief asked for, and why its answer is not usable

`infra/check_v2_targets_overlap.py` answers "is this span reproduced in her training text" by
looking each span n-gram up in **our** corpus's distinct-n-gram table and then indexing the
per-file masks by that key. For our own 512 that is free — their spans come from our corpus, and
the script asserts `span_shingles_missing_interior_clean == 0`. For **her** 512 it is the
question, and the answer is no: **13 of 9,303** of her span 13-grams (0.1397%) are in our key set,
so a hit is barely reachable and a zero would mean "unmeasured", not "not reproduced". It flags
one fully reproduced row (166), which is in Ari's 26 anyway.

The instrument that does answer it for her block is Ari's `features/ngram_overlap.py --side hers`,
which shingles her training parquets directly. Its output
(`/vol/shared/ngram-overlap/hers_n7.exclude.json`) is the 26, coverage ≥ 0.05 at n=7, of which
3 are fully covered (108, 166, 307). **Headline n = 486.**

### The mutation battery

| mutation | check that failed | message |
|---|---|---|
| `_columns` returns the ENCODER for the `dec` side | `check_sae_column_reader` | `the decoder side is not unit(W_dec[f]): max \|d\| 9.426e-01` |
| `_columns` rescales the encoder side by 1+1e-7 | `check_sae_column_reader` | `the sliced encoder side is not bit-identical to load_sae's: max \|d\| 1.192e-07` |
| `draw_sae2m` stops emitting `sae_side` | `check_sae_column_reader` | `draw_sae2m emits no sae_side field` |
| `recover_raw` drops the fallback assert | `check_heldout_v3_recovery` | `‖act‖ != the stored norm: max \|d\| 1.384e+01` — the second guard catches it, which is why the check accepts either |
| `recover_raw` drops BOTH guards | `check_heldout_v3_recovery` | `recover_raw accepted a row the solve cannot reach` |
| `check_set_on_disk` stops reconciling the family counts | `check_set_on_disk` | `check accepted a set whose rows disagree with its config entry` |
| `check_set_on_disk` stops requiring `act.f32` under a raw contract | `check_set_on_disk` | `check accepted a 'storage: raw' set with no act.f32` |

### The `--include` side-column defect, found on eval 1's own command

`draw_sae2m.build` drew the fit/report `side` column at `rng.random(n)` **after** the `--include`
branch subtracts the forced count from `n`. `--n 512 --stratified --include <64 ids>` therefore
built 448 labels for 512 features and `_finish`'s `side[i]` walked off the end. Nothing had run
that combination before — the 64-set is the SOURCE of the include list, not a user of it. Fixed
to `len(drawn)` with the assert beside it; the run above reports 413 + 99 = 512.

**New on the volume, all new paths:** the five `base/qwen36-27b/heldout/2026-09-21_v3_*`
directories, and `shared/eval1/2026-09-21_sae2m_64_feature_ids.txt` (the `--include` list, so the
draw is reproducible from the volume rather than from a scratchpad).

---

## 2026-09-21 — branch `evals/pipeline`: EVAL 1 (faithfulness), the production run

Every arm of plan §2.2 on the six `2026-09-21_v3_*` blocks. **39 GPU calls, $51.22**, plus ~$1.95
sunk (below). Largest single call $6.53 against a $30 per-call cap; total against a $75 stage cap.
Nothing on the volume was deleted, replaced or rewritten: every product is a new path, `--force`
was never passed, and the two old-primary arms are kept side by side as the evidence they are.

`--product unit` in the image ran green (44/44) before the first launch, per this file's own rule.

### Step 0 — the "ours" sanity block, $0

`heldout_v3 --block ours` (new) copied `2026-09-21_v1raw` rows 0-511 into
`base/qwen36-27b/heldout/2026-09-21_v3_ours`: our own realact draw, the one every v1 table was
built on, in the v3 layout. 512 rows, 15.2 MiB, `storage: raw`, CPU, $0. `check` opens all six v3
sets `ok`.

Verified against the source off the stored bytes, not from the run's own log: `act.f32` is
**byte-identical** on all 512 rows (max |d| exactly 0.0), `src_row == row`, ids/doc/p match
`2026-09-21_v1raw` row for row. `exclusions.json` freezes rows 38, 45, 101, 318, 393, 446 --
**headline n = 506** -- recorded, not applied.

### Estimate vs actual, per call

Estimates were made from this file's own bases before each launch, as asked. The vLLM rollout
basis is item 23 ($4.5046 / 98,304 rollouts); `score`'s was ambiguous and is resolved below.

| call | product | set | rows x n | est $ | actual $ | wall |
|---|---|---|---|---|---|---|
| A1 | `rollouts_vllm` rl-last16 | `_realact` | 512 x 64 | 1.50 | **2.1051** | 1669.3 s |
| A2 | `rollouts_vllm` rl-last16 | `_ours` | 512 x 64 | 2.10 | **2.0613** | 1634.5 s |
| A3 | `rollouts_vllm` rl-last16 | `_realact_long` | 512 x 64 | 2.10 | **2.1883** | 1735.2 s |
| A4 | `rollouts_vllm` rl-last16 | `_subspace` | 1024 x 64 | 3.90 | **3.8020** | 3014.8 s |
| A5 | `rollouts_vllm` rl-last16 | `_ctrl` | 1024 x 64 | 3.90 | **3.0659** | 2431.1 s |
| A6 | `rollouts_vllm` rl-last16 | `_sae2m` | 1024 x 64 | 3.90 | **3.0362** | 2407.5 s |
| B1 | `rollouts_vllm` old, `--mu none` | `_realact` | 512 x 64 | 1.50 | **1.9727** | 1564.3 s |
| B2 | `rollouts_vllm` old, `--mu stats` | `_realact` | 512 x 64 | 1.97 | **2.2150** | 1756.4 s |
| B3 | `rollouts_vllm` old, `--mu none` | `_ours` | 512 x 64 | 1.97 | **1.8258** | 1447.8 s |
| B4 | `rollouts_vllm` old, `--mu stats` | `_ours` | 512 x 64 | 1.97 | **2.0973** | 1663.1 s |
| B5 | `rollouts_vllm` old, `--mu none` | `_ctrl` | 1024 x 64 | 3.90 | **2.6351** | 2089.5 s |
| B6 | `rollouts_vllm` old, `--mu none` | `_sae2m` | 1024 x 64 | 3.90 | **2.6743** | 2120.6 s |
| C1 | `rollouts_nla` | `_realact` | 512 x 4 | 0.45 | **1.7980** | 1425.7 s |
| C2 | `rollouts_nla` | `_ctrl` | 1024 x 4 | 3.46 | **2.8509** | 2260.6 s |
| C3 | `rollouts_nla` | `_sae2m` | 1024 x 4 | 2.85 | **2.6856** | 2129.5 s |
| — | `score` x 14 | all | — | ~0.36 ea | **0.146 – 0.661** | 116 – 393 s |
| — | `sae_self` x 6 | `_ctrl`, `_sae2m` | — | ~0.40 ea | **0.154 – 0.417** | 122 – 330 s |
| D | `examples_docmax` 2M | `_sae2m` | 512 feat | 7.50 | **6.5295** | 5177.6 s |

**The one estimate that was badly wrong was the NLA's**, by 4x on C1 ($0.45 est, $1.80 actual).
The basis available was $0.3185 / 32 rollouts at 64 tokens, load-dominated; at 200 tokens and
512 targets the generation dominates instead and the measured rate is **1.55 rollouts/s** (C1),
rising to 2.04 (C2) and 1.92 (C3) as more targets amortise the load. Use ~1.9 rollouts/s, not the
load-dominated datum, for any future NLA sizing. Everything else landed within ~30%, and the
1024-target vLLM calls came in 22-32% UNDER estimate because the linear basis over-charges the
engine init when it is amortised over twice the rollouts.

**`score`'s two conflicting bases, resolved.** This file carried 195 rows/s (item 25, the full v1
run) and 20.5 rows/s (the `sae_smoke64` runs). The first is right at production scale: A-score-realact
did 32,768 rows in 281.8 s total, $0.3553, i.e. ~195 rows/s once the ~110 s model load is taken
out. The `sae_smoke64` figure is a small-batch artefact (1,024 rows), not a property of the SAE:
the 131k and 2M runs there were equally slow. **The 2M dictionary costs ~60% more than the 131k
per row at scale** ($0.6403 vs $0.4138 on 65,536 rows), not 10x.

### The summary table, for the results run to be checked against

Read off the products with a scratch reader; nothing here is recomputed from rollouts. Exclusions
APPLIED in this table (realact n = 486, ours n = 506); every other block is its full n. `bo64` is
`bo_4` for the NLA arm, which has n = 4 and must never carry a bo64 column.

## Cosines — mean (bo1) / bo8 / bo64, per source x set x family

| source | set | family | n | mean cos_raw | bo8 raw | bo64 raw | mean cos_ctr | bo8 ctr | bo64 ctr |
|---|---|---|---|---|---|---|---|---|---|
| rl-last16 | realact | realact | 486 | 0.8860 | 0.9168 | 0.9301 | 0.7590 |   --   |   --   |
| rl-last16 | ours | realact | 506 | 0.8884 | 0.9197 | 0.9330 | 0.7668 |   --   |   --   |
| rl-last16 | realact_long | realact_long | 512 | 0.4360 | 0.4856 | 0.5094 | 0.6991 |   --   |   --   |
| rl-last16 | subspace | bsf | 512 | 0.3177 | 0.3552 | 0.3778 |   --   |   --   |   --   |
| rl-last16 | subspace | jlens | 512 | 0.1058 | 0.1197 | 0.1293 |   --   |   --   |   --   |
| rl-last16 | ctrl | random | 512 | 0.0338 | 0.0412 | 0.0464 |   --   |   --   |   --   |
| rl-last16 | ctrl | sae | 512 | 0.1136 | 0.1374 | 0.1532 |   --   |   --   |   --   |
| rl-last16 | sae2m | sae | 1024 | 0.0584 | 0.0699 | 0.0783 |   --   |   --   |   --   |
| old mu-none | realact | realact | 486 | 0.8830 | 0.9142 | 0.9278 |   --   |   --   |   --   |
| old mu-none | ours | realact | 506 | 0.8870 | 0.9162 | 0.9294 |   --   |   --   |   --   |
| old mu-none | ctrl | random | 512 | 0.0280 | 0.0365 | 0.0427 |   --   |   --   |   --   |
| old mu-none | ctrl | sae | 512 | 0.1299 | 0.1554 | 0.1720 |   --   |   --   |   --   |
| old mu-none | sae2m | sae | 1024 | 0.0545 | 0.0670 | 0.0757 |   --   |   --   |   --   |
| old mu-stats | realact | realact | 486 | 0.8881 | 0.9188 | 0.9322 | 0.7713 |   --   |   --   |
| old mu-stats | ours | realact | 506 | 0.8905 | 0.9221 | 0.9344 | 0.7768 |   --   |   --   |
| NLA n=4 | realact | realact | 486 | 0.8075 |   --   | 0.8424 |   --   |   --   |   --   |
| NLA n=4 | ctrl | random | 512 | 0.0308 |   --   | 0.0351 |   --   |   --   |   --   |
| NLA n=4 | ctrl | sae | 512 | 0.0879 |   --   | 0.1043 |   --   |   --   |   --   |
| NLA n=4 | sae2m | sae | 1024 | 0.0516 |   --   | 0.0589 |   --   |   --   |   --   |

## SAE — median peak/corpus_peak and median fire fraction (sae_self)


| source | dictionary | features | med bo1 ratio | med bo_n ratio | med fire frac | features firing |
|---|---|---|---|---|---|---|
| rl-last16 | 131k l42-1b | 512 | 0.719 | 0.958 | 1.000 | 0.932 |
| rl-last16 | 2M sae2m | 512 | 0.239 | 0.544 | 0.047 | 0.777 |
| old mu-none | 131k l42-1b | 512 | 0.803 | 1.032 | 1.000 | 0.943 |
| old mu-none | 2M sae2m | 512 | 0.095 | 0.403 | 0.000 | 0.402 |
| NLA n=4 | 131k l42-1b | 512 | 0.505 | 0.619 | 1.000 | 0.799 |
| NLA n=4 | 2M sae2m | 512 | 0.233 | 0.325 | 0.000 | 0.246 |

`med bo1 ratio` is the median over features of `mean_peak_act / corpus_peak`, `med bo_n ratio` of
`max_peak_act / corpus_peak` (n = 64, or 4 for NLA), `corpus_peak` being our 16M `max_act.f16` --
never her 1B peak. `med fire frac` is the median over features of the fraction of that source's
own rollouts in which the target feature clears the gate 1.682812; `features firing` is the share
of features that fire at all.

**Cross-check against `sae_smoke64` (64 features, n = 16, a different draw).** 2M bo1: rl-last16
0.239 here vs 0.233 there; old primary 0.095 vs 0.102; NLA 0.233 vs 0.242. 131k bo1: rl-last16
0.719 vs 0.76; old primary 0.803 vs 0.83; NLA 0.505 vs 0.51. The 512-feature run reproduces the
64-feature smoke on all six cells. The structural claim reproduces too: every generated source
sits at a third to a half of the corpus peak on the 2M dictionary and at 0.6-1.0 on the 131k one,
and rl-last16 beats the old primary on the 2M SAE while trailing it on the 131k.

### What the cosine table says

The three realact arms are within 0.005 of each other on her block and on ours -- rl-last16
0.8860 / 0.8884, old primary at its winning mean 0.8881 / 0.8905, i.e. the two checkpoints are
not separated by this statistic at n = 486. The NLA arm sits ~0.08 below them at 0.8075. The
controls behave: `random` 0.028-0.034, `jlens` 0.106, `bsf` 0.318, the 2M dictionary rows 0.052-0.058.

`realact_long` is the one striking row: **cos_raw 0.4360 but cos_centred 0.6991**, the only family
where the centred number is far ABOVE the raw one. That is the `family_mu: unknown` block being
read at `whiten_mu` -- the mean those rows carry is `mu_long` and nobody holds the file, so the
centred column there is "centred on a mean that is not the rows' own" and is a labelled number,
not a comparable one. It is in the product README, and the results run must not put it in a
column beside the realact centred numbers without that label.

### The old primary's `mu`, SETTLED

Two arms, same rows (`2026-09-21_v3_realact`), same scorer, n = 486 x 64 rollouts:

| statistic | `--mu none` | `--mu stats_mu` | delta | 95% CI, doc-clustered bootstrap (10k) | rows won |
|---|---|---|---|---|---|
| mean_cos | 0.8830 | **0.8881** | +0.0051 | [+0.0030, +0.0072] | 322/486 (66.3%) |
| bo_8 | 0.9142 | **0.9188** | +0.0046 | [+0.0030, +0.0062] | 324/486 (66.7%) |
| bo_64 | 0.9278 | **0.9322** | +0.0044 | [+0.0026, +0.0061] | 334/486 (68.7%) |

`stats_mu` wins on every statistic and on two thirds of rows individually; the paired bootstrap
over the 425 distinct documents (61 of the 486 rows share one) excludes zero in all three.
`config.yaml` now declares `mu: base/{base}/stats/mu.f32`, replacing `unknown`. Run tags
`mu-none` / `mu-stats`; both arms stay on the volume.

The effect is SMALL (~0.005 cosine) and that is part of the finding. The §1.6 smoke's n = 4 x 4
table read +0.023 and 3-of-4 rows; at n = 486 x 64 the SIGN holds and the SIZE does not. It also
confirms the record (Celeste's original convention was a centred target) against the eval plan's
§1.1/§2.2 `none`, which came from a recollection.

Still NOT evaluated: the plan's reproduction gate ("cos_raw within 0.01 of 0.5076"). Unchanged
from the §1.6 note -- the stored 0.5076 is the asymmetric `cos(h, unit(act - stats_mu))`, which
`score` does not produce from a raw set under any flag combination.

### Search baseline: `interim-16M`, because Ari's 10M corpus has never been scanned

Inspected read-only. `base/qwen36-27b/corpora/train_parity_10m/` EXISTS -- 10,004,614 tokens,
11,809 documents, ladder [1.25, 2.5, 5, 10], window 32/8 -- and carries only `tokens.i32`,
`meta.json`, `docs.jsonl`, `README.md`. **There is no scan over it:** `base/qwen36-27b/scan/` holds
only `2026-09-16_v1`, `2026-09-18_ood_v1`, `2026-09-20_sae2m_2k`, none corpus-suffixed, and neither
`sae/*/examples/` nor `examples_docmax/` has a `__train_parity_10m` directory. So nothing under the
`celeste-train10m` key is consumable by `sae_self` or `score`, and the question of layout does not
arise.

**And it could not be produced in this stage even if wanted**: `common.assert_corpus_geometry`
REFUSES that corpus, because it declares 32/8 while the pipeline cuts 64/16 at all eleven
`windows_of` sites (H7, declared and deliberately not threaded). **So the corpus-geometry threading
IS needed for the plan's §2.4 search baseline** -- saying so, as the brief asked, rather than
threading it here.

The interim row instead, labelled `interim-16M` and NOT comparable to a training-split number:

* **131k**: the existing `sae/l42-1b/examples_docmax/2026-09-16_v1` covers **512 / 512** of
  `_ctrl`'s sae feature ids (checked against its `tested.json`). $0, no run.
* **2M**: nothing existed for the 512-feature set, so `examples_docmax` was run once --
  18,813 docs, 512 features, 5177.6 s, **$6.5295**, against the $6.46 / 5,123 s of the 64-feature
  run. The corpus forward dominates and 8x the features cost ~1%, as predicted.

### Product paths, for the results run

    <RL16> = maemms/qwen36-27b/2026-09-18_rl-last16-lr5e-7
    <OLD>  = maemms/qwen36-27b/2026-09-10_rl-8x2048-full
    <NLA>  = maemms/qwen36-27b/2026-07-14_nla-av

| arm | rollouts | scores | sae_self |
|---|---|---|---|
| rl-last16, all six sets | `<RL16>/rollouts/2026-09-21_v3_<blk>__vllm.jsonl` | `<RL16>/scores/2026-09-21_v3_<blk>__vllm/` | `_ctrl`, `_sae2m` only |
| old primary, `mu-none` | `<OLD>/rollouts/2026-09-21_v3_<blk>__vllm__mu-none.jsonl` | `<OLD>/scores/2026-09-21_v3_<blk>__mu-none__vllm/` | `_ctrl`, `_sae2m` |
| old primary, `mu-stats` | `…__mu-stats.jsonl`, `_realact` + `_ours` only | `…__mu-stats__vllm/` | — |
| NLA, n = 4 | `<NLA>/rollouts/2026-09-21_v3_<blk>.jsonl` | `<NLA>/scores/2026-09-21_v3_<blk>/` | `_ctrl`, `_sae2m` |
| search, interim-16M | — | `base/qwen36-27b/sae/l42-1b/examples_docmax/2026-09-16_v1/` (131k) and `base/qwen36-27b/sae/sae2m/examples_docmax/2026-09-21_v3_sae2m/` (2M) | — |

`<blk>` is one of `realact ours realact_long subspace ctrl sae2m`. n = 64 everywhere but the NLA
arm's 4. `sae_self` lives under each scores directory as `sae_self/sae_self.json`.

### Three things the results run must not get wrong

1. **`per_target.jsonl` has NO centred best-of-k.** It carries `mean_cos_centred`,
   `max_cos_centred` and `n_centred` only -- there is no `bo_8_centred` / `bo_64_centred`, so the
   centred bo-k columns of plan §2.3 have to come from `cos_centred.f16` directly
   (`common.best_of_k_means` over the [N, n, T] max per rollout). The raw cosine has the full
   `bo_1..bo_64` ladder.
2. **`sae_self` measured only the ENCODER half of `_sae2m`.** `sae_rows_of(..., side="enc")` at
   `sae_self.py:124` filters the 1,024 rows to the 512 `sae_side: enc` ones, so rows 512-1023 (the
   decoder side) have COSINES from `score` but no activation metric. Plan §2.3 wants them
   ("the activation of feature `f` is its encoder readout whichever direction was injected") and
   the card's 0.416 dec gate needs them. NOT fixed here -- `side=` is a shared selector used by
   `build`, `scan` and `repo_examples` too, and changing it late, against a schema the results run
   is being written to, is a worse risk than naming it. Cost to close: one flag plus ~$0.4 x 3.
3. **`realact_long`'s centred column is read at the wrong mean** (see above) and `_subspace` /
   `_ctrl` / `_sae2m` have no centred column at all, by the NaN rule -- non-centrable families are
   absent from the centred aggregates, never a one-sided number.

### Deviations from the brief, and what pays for each

1. **`rl-last16` ran on vLLM, not HF.** The brief called HF the proven path. At this scale it is
   not affordable: the measured 27B HF rate is 5.33 rollouts/s (this file's engine-choice table),
   so rl-last16's 294,912 rollouts would be ~15.4 h and **~$70** on HF against the **$16.25** the
   six vLLM calls actually cost. vLLM is the same served-weights path the old primary's existing
   v1 products came from, and the marker and injection checks ran on every call (`cos` 0.999991,
   `norm_ratio` 0.999891 on A1). The brief's own "HF is fine if cheaper" clause, answered: it is
   4x more expensive, so vLLM for both.
2. **The old primary's second arm was NOT run on `_ctrl` and `_sae2m`.** The brief asked for two
   arms there. It would be a bit-identical duplicate: both families of `_ctrl` are
   `centrable: false`, `_sae2m` is `dirs_only`, and `dirs_for` subtracts a mean from centrable
   rows only. VERIFIED on the real bytes, not argued -- `_ctrl`'s own `act.f32` and
   `stats/mu.f32` off the volume, 1,024 rows, `--mu none` vs `--mu stats_mu`: **bit-identical,
   max |d| exactly 0.0, 0 of 1024 rows centrable.** The §1.6 smoke saw the same thing in its
   scores. Saved ~$8.2; the `mu-none` products ARE the `mu-stats` products for those rows.
3. **`examples_docmax` is a step-4 cost, not a step-3 one.** The brief put the 16M corpus peaks
   under step 3, from `examples_docmax`. `sae_self` does not read that product -- it takes
   `corpus_peak` from `sae/<sae>/max_act.f16` (`sae_self.py:383`), which is full-dictionary
   (2,097,152 entries for sae2m, confirmed in its `index.json`) and was already on the volume for
   both dictionaries. So no docmax was needed for the ratios; it was run for the SEARCH row.

### Sunk cost, ~$1.95, and the two failures behind it

**~$1.45 -- three H200 containers killed by a local network drop, `--detach` notwithstanding.**
A1/B1/C1 were launched with `modal run --detach` under `setsid`. A host network outage killed the
local clients (`TimeoutError: [Errno 110] Connect call failed ('54.80.13.45', 443)`) and Modal then
cancelled the in-flight inputs ~60 s later -- `Received a cancellation signal while processing
input`, `Aborting 31460 requests`, engines torn down mid-generation. Both vLLM engines had just
finished a 270-313 s init. **This is README.md:1004's warning reproduced with `--detach` ON: the
flag keeps the APP alive, it does not keep the INPUT alive when the client process dies.**
Nothing partial landed (temp-and-rename), no product directory was touched, and the three calls
were relaunched cleanly.

Fixed for the rest of the stage by moving the retry OUT of the modal client, into bash: the
launcher re-runs a call that died on a client-side network error, up to 3 times, and STOPS loudly
on anything else. Retrying is safe because `OutDir` refuses to overwrite an existing product
without `--force`, so a retry after a run that actually finished fails on "already exists" (which
the launcher detects and does not retry) rather than destroying it. The launcher was self-tested
on a deliberate bad argument before use -- one attempt, loud stop, no container started -- which
caught a real bug in its first version (variables did not survive into the `setsid` subshell).

**~$0.50 -- one `sae_self` killed after its forward by the f16-vs-fp32 gate assert.** Its own
section and commit; the fix is `check_csr_gate_floor` and the rerun cost $0.3776.

### Local checks

`uv run paper-evals/precompute/unit_smoke.py` -- **45/45** (43 at the start of this stage; +1 for
the `ours` block, +1 for the CSR gate floor), and `nla-selftest` 5 -> 6. Mutation batteries run
on every check added: 5/5 on `check_heldout_v3_ours_block`, 3/3 on `_selftest_amp_storage`,
3/3 on `check_csr_gate_floor` -- the last including the original `> gate`, which reproduces the
production failure on CPU.
