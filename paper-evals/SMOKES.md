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
