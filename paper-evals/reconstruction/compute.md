# Compute appendix: measured GPU cost of the reconstruction evaluation

Every non-trivial step of the `paper-evals` pipeline, as run on 2026-09-15/16, with the wall
clock each product wrote into its own `README.md` on the volume. Two GPU rates, both on-demand
Modal: **H100 $3.95/h** (base `qwen3-8b`) and **H200 $4.54/h** (base `qwen36-27b`), per
`config.yaml` `bases.*.gpu`. CPU-only steps are priced at ~$0.

GPU-hours = wall / 3600. **Wall is the container's own elapsed time and includes container
start, model/engine load and the closing volume commit** wherever SMOKES says so: the 27B
`rollouts_vllm` run spent 4,117.6 s of its 4,460.7 s wall inside `generate`, `mu_diag` 27B
spent 30 s of 249.8 s on the 51.8 GiB load, and the Patchscopes sweep charges the model load
to its first cell. M = measured on the shape it reports; E = extrapolated, never run at that
shape. Sources are sections of `paper-evals/SMOKES.md` unless marked README.

| step | base | GPU | GPU-h | sample volume | GPU-h / 1k | USD | M/E | source |
|---|---|---|---|---|---|---|---|---|
| `check`, both bases | both | CPU | 19.0 s | — | — | ~$0 | M | Full run 2026-09-16 |
| `corpus` build | 8B | CPU | 65.0 s | 16,000k corpus tokens | 1.13e-6 | ~$0 | M | Full run 2026-09-16 |
| `corpus` build | 27B | CPU | 89.0 s | 16,000k corpus tokens | 1.55e-6 | ~$0 | M | Full run 2026-09-16 |
| `stats` pass A @16M (64/16 windows, ~65M token-forwards) | 8B | H100 | 0.688 | 16,000k corpus tokens | 4.30e-5 | $2.72 | M | Full run 2026-09-16 |
| `stats` pass A @16M (64/16 windows, ~65M token-forwards) | 27B | H200 | 2.009 | 16,000k corpus tokens | 1.26e-4 | $9.12 | M | Full run 2026-09-16 |
| `targets` draw, set `2026-09-16_v1` (512-token realact forwards + leakage check) | 8B | H100 | 0.018 | 1.536k targets (512 x 3 families) | 1.15e-2 | $0.07 | M | Full run 2026-09-16 |
| `targets` draw, set `2026-09-16_v1` (leakage check skipped, no 27B bank) | 27B | H200 | 0.041 | 1.536k targets (512 x 3 families) | 2.67e-2 | $0.19 | M | Full run 2026-09-16 |
| `scan` pass B @16M, target-dependent | 8B | H100 | 0.462 | 16,000k corpus tokens | 2.89e-5 | $1.82 | M | Full run 2026-09-16 |
| `scan` pass B @16M, target-dependent | 27B | H200 | 1.396 | 16,000k corpus tokens | 8.73e-5 | $6.34 | M | Full run 2026-09-16 |
| `rollouts_hf` `run1-rl`, 1,536 x 64 | 8B | H100 | 0.495 | 98.304k rollouts | 5.04e-3 | $1.96 | M | Full run 2026-09-16 |
| `rollouts_vllm` `rlI-150` (LoRA), patched engine, 1,536 x 64 | 27B | H200 | 1.239 | 98.304k rollouts | 1.26e-2 | $5.63 | M | The vLLM speed patch |
| `rollouts_vllm` `rl-8x2048-full`, patched engine, 1,536 x 64 | 27B | H200 | 0.992 | 98.304k rollouts | 1.01e-2 | $4.50 | M | The vLLM speed patch |
| same run on stock vLLM (19.52 rollouts/s @512 rows) | 8B | H100 | 1.399 | 98.304k rollouts | 1.42e-2 | $5.53 | **E** | Engine choice, MEASURED |
| same run on patched vLLM (126.9 rollouts/s, never re-run at full scale) | 8B | H100 | 0.215 | 98.304k rollouts | 2.19e-3 | $0.85 | **E** | The vLLM speed patch |
| same run on HF `generate`, `rlI-150` (3.74 rollouts/s) | 27B | H200 | 7.30 | 98.304k rollouts | 7.43e-2 | $33.15 | **E** | Engine choice, MEASURED |
| same run on HF `generate`, `rl-8x2048-full` (5.33 rollouts/s) | 27B | H200 | 5.12 | 98.304k rollouts | 5.21e-2 | $23.26 | **E** | Engine choice, MEASURED |
| same run on pre-patch vLLM (3.5-4.25 rollouts/s at the real shape) | 27B | H200 | 6.4-7.8 | 98.304k rollouts | 6.5e-2 - 7.9e-2 | $29-35 | **E** | What the probe got wrong |
| `score` on the clean base, `run1-rl` | 8B | H100 | 0.055 | 98.304k rows | 5.56e-4 | $0.22 | M | Full run 2026-09-16 |
| `score` on the clean base, `rlI-150` | 27B | H200 | 0.135 | 98.304k rows | 1.38e-3 | $0.61 | M | Full run 2026-09-16 |
| `score` on the clean base, `rl-8x2048-full` | 27B | H200 | 0.140 | 98.304k rows | 1.42e-3 | $0.63 | M | Full run 2026-09-16 |
| `repo_examples`, 509 features x 30 windows (3 dropped, sink non-uniform) | 8B | H100 | 0.011 | 15.27k windows | 7.42e-4 | $0.04 | M | Full run 2026-09-16 |
| `repo_examples`, failed run on the sink assert | 8B | H100 | ~0.019 | — | — | ~$0.08 | M (cost est.) | Full run 2026-09-16 |
| `repo_examples`, 512 features x 32 windows (0 dropped) | 27B | H200 | 0.041 | 16.384k windows | 2.48e-3 | $0.18 | M | Full run 2026-09-16 |
| `mu_diag`, smoke corpus, 500k-token cap | 8B | H100 | 0.027 | 2,450.6k positions (both geometries) | 1.09e-5 | $0.11 | M | Step 6: `mu_diag` |
| `mu_diag`, smoke corpus, 400k-token cap | 27B | H200 | 0.069 | 1,792.9k positions (both geometries) | 3.87e-5 | $0.32 | M | Step 6: `mu_diag` |
| Patchscopes sweep: 4 cells (floor + L8/L14/L21) x 128 dirs x bo 8 | 27B | H200 | 0.169 | 4.096k rows | 4.13e-2 | $0.77 | M | Patchscopes 27B |
| Patchscopes sweep scoring, 4 cells x 1,024 rows | 27B | H200 | 0.120 | 4.096k rows | 2.92e-2 | $0.54 | M | Patchscopes 27B |
| Patchscopes L14 scoring, first attempt (assert fired) | 27B | H200 | ~0.033 | 1.024k rows | — | ~$0.13 | M (cost est.) | Patchscopes 27B |
| Patchscopes final cell: floor + L14, 640 dirs x bo 32, 2 cells | 27B | H200 | 1.287 | 40.96k rows | 3.14e-2 | $5.84 | M | Patchscopes 27B |
| Patchscopes final scoring, 2 cells x 20,480 rows | 27B | H200 | 0.129 | 40.96k rows | 3.15e-3 | $0.59 | M | Patchscopes 27B |
| exploratory pass (smoke draw, 8 directions, incl. EPO): `gcg` arms (`gcg-corpus`, `gcg-random32`; realact + sae), per arm | 8B | H100 | 0.183-0.203 | 614.4k candidate forwards (8 x 76,800) | 3.0e-4 - 3.3e-4 | $0.72-0.80 | M | All 16 arms, final |
| exploratory pass: `epo` arms (`epo-corpus`, `epo-random32`; realact + sae), 8 dirs each, per arm | 8B | H100 | 0.562-0.585 | 612.0k candidate forwards (8 x 76,500) | 9.2e-4 - 9.6e-4 | $2.22-2.31 | M | All 16 arms, final |
| exploratory pass: `gcg` arms, 8 dirs each, per arm | 27B | H200 | 0.589-0.652 | 614.4k candidate forwards | 9.6e-4 - 1.1e-3 | $2.67-2.96 | M | All 16 arms, final |
| exploratory pass: `epo` arms, 8 dirs each, per arm | 27B | H200 | 1.836-2.005 | 612.0k candidate forwards | 3.0e-3 - 3.3e-3 | $8.34-9.10 | M | All 16 arms, final |
| exploratory pass: local checks, shakeouts, 3 failed 27B attempts, 1 failed launch, `score` re-run, stale `scan` re-run | both | mixed | ~0.31 | — | — | $1.38 | M | Final GCG spend |
| `gcg` arms, full root, 32 dirs each (realact + sae x corpus/random32), per arm | 8B | H100 | 0.724-0.773 | 2,457.6k candidate forwards (32 x 76,800; 2,380.8k on the 31-dir arm) | 2.95e-4 - 3.25e-4 per 1k cand; **0.0225-0.0250 per direction** | $2.86-3.05 | M | GCG final (full root) |
| `gcg` arms, full root, 32 dirs each, per arm | 27B | H200 | 2.291-2.358 | 2,457.6k candidate forwards | 9.3e-4 - 9.6e-4 per 1k cand; **0.0717-0.0736 per direction** | $10.40-10.70 | M | GCG final (full root) |
| row-17 batch-shape diagnostic, 3 reproductions (no product written) | 8B | H100 | ~0.091 | — | — | ~$0.36 | M (cost est.) | GCG final (full root), Spend |
| `gcg` stratified `sae` arms (`gcg-corpus-strat`, `gcg-random32-strat`), 32 targets each (8 per density quartile), per arm | 8B | H100 | 0.747-0.766 | 2,457.6k candidate forwards (32 x 76,800) | 3.0e-4 - 3.1e-4 per 1k cand; 0.0233-0.0239 per direction | $2.95 / $3.03 ($0.0922 / $0.0946 per dir; 84 / 86 s) | M | GCG stratified sae (full root) |
| `gcg` stratified `sae` arms, 32 targets each (8 per density quartile), per arm | 27B | H200 | 2.273-2.367 | 2,457.6k candidate forwards | 9.3e-4 - 9.6e-4 per 1k cand; 0.0711-0.0739 per direction | $10.75 / $10.32 ($0.3358 / $0.3225 per dir; 266 / 256 s) | M | GCG stratified sae (full root) |
| `epo` arms at 32 targets (realact rows 0-31 + stratified `sae`, both inits), pop 3 x 85 children x 300 iters, per arm | 8B | H100 | 2.204-2.406 | 2,448.0k candidate forwards (32 x 76,500) | 9.0e-4 - 9.8e-4 per 1k cand; 0.0689-0.0752 per direction | $8.70 / $9.41 / $8.89 / $9.50 ($0.2720-0.2970 per dir) | M | GCG/EPO 32-target arms |
| `--resume-from` smoke, 27B `realact` row 24 into `epo-corpus-smoke` (a plumbing check, not a paper arm, kept on the volume) | 27B | H200 | 0.287 | 76.5k candidate forwards (1 direction) | — | $1.30 | M | GCG/EPO 32-target arms |
| `epo` arms at 32 targets, same four cells, per arm (timed-out call + resume, see note) | 27B | H200 | 7.601-8.018 | 2,448.0k candidate forwards | 3.1e-3 - 3.3e-3 per 1k cand; 0.2375-0.2506 per direction | $36.40 / $34.51 / $35.52 / $36.40 ($1.0786-$1.1376 per dir) | M | GCG/EPO 32-target arms, Spend |
| `top1_act`, cosine top-1 corpus window per `sae` feature | 8B | H100 | 0.022 | 0.512k features (473 joined, 39 forwarded) | 4.25e-2 | $0.09 | M | `top1_act` |
| `top1_act`, cosine top-1 corpus window per `sae` feature | 27B | H200 | 0.056 | 0.512k features (508 joined, 4 forwarded; all 512 forwarded as the check) | 1.09e-1 | $0.25 | M | `top1_act` |
| `gcg` / `epo` arm at 64 dirs, SUPERSEDED by the measured 32-direction run (2026-09-16); kept for the cost-per-direction extrapolation only (per arm: 8B gcg / 8B epo / 27B gcg / 27B epo) | both | mixed | 1.43-1.57 / 4.45-4.63 / 4.61-5.08 / 14.58-15.92 | 4,915.2k / 4,896.0k candidate forwards | — | $5.63-6.19 / $17.57-18.29 / $20.93-23.06 / $66.19-72.28 | **E** | Sharpened projection |
| vLLM `--throughput` probe, `run1-rl`, 32-512 rows | 8B | H100 | 0.040 | overhead | — | $0.16 | M | Full run 2026-09-16 |
| vLLM `--throughput` probe, `rlI-150`, 32-512 rows | 27B | H200 | 0.133 | overhead | — | $0.61 | M | Full run 2026-09-16 |
| vLLM real-shape calibration, `rlI-150`, 128 x 64 | 27B | H200 | 0.598 | 8.192k rollouts | 7.30e-2 | $2.71 | M | Full run 2026-09-16 |
| vLLM n-shape test, `rlI-150`, 128 x 16 | 27B | H200 | 0.160 | 2.048k rollouts | 7.83e-2 | $0.73 | M | Full run 2026-09-16 |
| `score` rate probe, `rlI-150`, 2,048 rows | 27B | H200 | 0.028 | 2.048k rows | 1.37e-2 | $0.13 | M | Full run 2026-09-16 |
| 8B parity rollouts on the patched engine, 48 x 64 | 8B | H100 | 0.031 | 3.072k rollouts | 1.01e-2 | $0.12 | M | Full run 2026-09-16 |
| 8B parity `score --engine vllm`, 3,072 rows | 8B | H100 | 0.009 | 3.072k rows | 3.02e-3 | $0.04 | M | Full run 2026-09-16 |
| 27B rollouts stopped mid-run for the patch, + 2 client-cancelled relaunches | 27B | H200 | ~1.55 | discarded | — | ~$7.03 | M (2 cells est.) | The vLLM speed patch |
| `centred`, both 27B MAEMMs | 27B | CPU | 33.9 s | 98.304k rows each | — | ~$0 | M | Full run 2026-09-16 |

**Totals, measured rows only** (sunk and failed runs included, since they were paid):

- **qwen3-8b (H100): 18.76 GPU-hours, $74.12** — 1.86 h / $7.33 of pipeline and probes; 3.07 h /
  $12.11 of the exploratory GCG/EPO pass at 8 directions; 3.06 h / $12.10 of the 4 full-root
  `gcg` arms at 32 directions plus the row-17 diagnostic; 1.52 h / $5.98 of the 2 stratified
  `sae` arms; 9.24 h / $36.51 of the 4 `epo` arms at 32 targets; 0.02 h / $0.09 of `top1_act`.
- **qwen36-27b (H200): 66.14 GPU-hours, $300.29** — 10.27 h / $46.63 of pipeline, probes,
  Patchscopes and the $7.03 of sunk relaunch cost; 10.13 h / $46.00 of the exploratory pass;
  9.29 h / $42.19 of the 4 full-root `gcg` arms; 4.64 h / $21.07 of the 2 stratified `sae` arms;
  31.75 h / $144.14 of the 4 `epo` arms at 32 targets (24.0 h / ~$108.96 of that is the four
  first calls Modal cancelled at the 6 h function timeout, 7.46 h / $33.88 the `--resume-from`
  calls that finished them, 0.29 h / $1.30 the resume smoke); 0.06 h / $0.25 of `top1_act`.
- Both bases: **84.90 GPU-hours, $374.41**, plus ~4 min of CPU. The 27B is 3.6x the 8B on the
  identical full-root GCG work and 4.0x overall.
- **Discrete search alone is $321.51 of that**: exploratory pass $59.49, `gcg` final (32 dirs)
  $54.32, stratified `sae` $27.05, `epo` (32 targets, resume smoke included) $180.65. Only ~$3
  of the 24 h of cancelled calls was actually lost: each was killed mid-direction with 24-25 of
  32 directions already written, and `--resume-from` carried those over.

## Precision limit for per-direction claims on the 27B

The 27B GCG search is not run-to-run reproducible. Rows 1024-1031 appear in both the q0 `sae` arms
and the stratified arms, run independently from byte-identical inits and the same per-direction
seeding, so they are a free determinism check: on the 27B the two runs returned identical final
strings on **1 of 8** directions (`gcg-corpus`) and **0 of 8** (`gcg-random32`), with max
|Δcos| **0.264** and the arm mean over those 8 rows moving **-0.036** and **+0.024** between runs.
The 8B is bit-exact on the same check (**16/16**, max |Δcos| 0.00e+00). The inits agree to
0.00e+00 (corpus) and 4.4e-04 (random32), so the divergence is in the search, and the only
27B-specific path is fla's GatedDeltaNet Triton backward — a nondeterministic gradient changes the
proposed candidates and the trajectories then separate for good. Most rows still agree closely; one
row per arm diverges hard. The -0.036 is the same size as the 27B realact GCG-vs-MAEMM gap
(-0.043), and the SE over targets does not capture it. **Convention, decided 2026-09-16:** 27B
GCG/EPO arm means are quoted to 2 decimal places with the SE over targets, and no single 27B
direction's GCG value is quoted. Source: SMOKES.md, "GCG stratified sae (full root, 2026-09-16)".

## Rates that matter

- **Forward throughput, `stats` (pass A)**, corpus tokens/s at 16M: 8B **6,533**, 27B **2,229**
  (smoke: 6,554 / 2,185). Forwarded tokens are ~4.1x higher — the 64/16 geometry puts every
  token in ~4 windows plus the sink: 25,305 / 8,448 fwd tok/s at the smoke. (README Costs;
  Measured throughput and the 16M extrapolation.)
- **Forward throughput, `scan` (pass B)**: 8B **9,851**, 27B **3,227** corpus tok/s at 16M
  (smoke: 9,764 / 2,877; 37,703 / 11,125 fwd tok/s).
- **Generation, 8B `run1-rl`**: HF `generate` **48.3 rollouts/s / 3,388 gen tok/s** (the
  delivered production run); stock vLLM + `enforce_eager` **19.5 / 808.1** at 512 rows in
  flight; patched vLLM (fast hook + FULL_DECODE_ONLY graphs) **126.9 / 4,068** — 6.5x the stock
  engine and 2.6x HF. rollouts/s is the comparable number: the HF path counts the padded
  64-token tensor, vLLM the tokens returned (mean 41-56), so tok/s flatters HF by ~1.2-1.5x.
- **Generation, 27B `rlI-150` (LoRA)**: HF **3.74 rollouts/s** (239.3 gen tok/s at 32 rows/call);
  stock vLLM probe **6.50 / 365.7** at 256 rows in flight, falling to **4.25 / 236.9** at the real
  (128 dirs, n=64) shape and an extrapolated 3.5-4.2 at 1,536 dirs; patched vLLM, production
  **23.87 rollouts/s / 818.1 gen tok/s**.
- **Generation, 27B `rl-8x2048-full`**: HF **5.33 rollouts/s** (341.5 gen tok/s; no adapter
  matmuls); never probed on stock vLLM; patched vLLM, production **29.32 / 952.6**.
  Clean base, no adapter, Patchscopes: 291-394 gen tok/s.
- **Scoring (clean base, 27B)** is warm-up dominated at smoke scale: **12.9 rows/s** at 512 rows
  -> **43.9** at 2,048 -> **115** at 20,480. At the full 98,304-row job the wall-inclusive rate is
  ~202 / ~195 rows/s (`rlI-150` / full) on the 27B and ~500 rows/s on the 8B.
- **GCG candidate throughput**, full root at 32 directions: 8B `gcg` **868-957 cand/s**, 27B
  `gcg` **292-301** (smoke root at 8 directions: 876-962 and 270-297). EPO at 32 targets ran ~250-270 s/dir (8B)
  and ~870-880 s/dir (27B) over 76,500 candidate forwards per direction, i.e. ~283-306 and
  ~87-88 cand/s; the 8-direction pass measured 8B **294-306**, 27B **86-93**. EPO's 3x-smaller per-iteration pool (255 vs 512 candidates) is
  why its wall/dir is ~3x despite the same per-direction budget.

## Notes on the numbers

- The 8B production rollouts ran on HF because the *stock* vLLM path was 2.47x slower per
  rollout; the patch landed mid-run and inverted that, but the 8B was not re-run. The 8B
  patched-vLLM row is therefore E, from the 48 x 64 parity measurement.
- The 27B full model was never separately probed on vLLM before the run; SMOKES flags that
  engine choice as an inference from the LoRA engine plus HF's LoRA/full ratio.
- `stats` is target-independent and is not re-run per held-out set; `scan` is.
- The 32-direction `gcg` numbers are MEASURED on the full root: 8 arms (2 bases x 2 families x
  2 inits), `gcg` only, 32 directions each, at $0.0894-0.0985/dir (8B) and $0.3251-0.3345/dir
  (27B). Arm totals $2.86-3.05 and $10.40-10.70; phase total **~$54.32** = $53.93 of arms
  + ~$0.36 of row-17 reproductions + ~$0.03 of a failed launch. Per base **8B $11.74, 27B
  $42.19** over 4 arms each. The 8B `realact/gcg-random32` arm covers 31 of 32 directions:
  row 17 tripped the end-of-direction check on a 1e-2 batch-shape discrepancy (M=1 GEMV vs
  batched GEMM in bf16) and was excluded rather than absorbed.
- The four stratified `sae` `gcg` arms (`*-strat`) take 8 rows from each of the four density
  quartiles, against the final arms' rows 0-31, which are all q0. They are the sae-family view of
  the same search, not an independent pass; stage total **$27.05** (SMOKES's own Spend line;
  32 x $/dir over the four arms rounds to $27.04). The final arms are therefore the RARE-stratum view.
- **The four 27B `epo` arms each cost two calls, and the arm README records only the second.**
  At ~880 s/direction, 32 directions is 7.8 h against the 6 h `timeout` `gcg/modal_app.py` then
  carried, so all four were killed by Modal at exactly 21,600 s with 24-25 of 32 directions
  written. They were completed with `--resume-from` on the kept temp dir (another session, which
  also raised the timeout to 9 h), so the partial compute was reused rather than discarded —
  ~$36 of resume against ~$140 to re-run from scratch. The cost of each arm is the timed-out call
  ($27.24 = 21,600 s at $4.54/h) PLUS the resume ($7.27-$9.16); summing the four arm READMEs
  understates the 27B EPO stage by **$108.96**. The table above carries the true totals.
- EPO per-direction cost came in at $1.0786-$1.1376 (27B) and $0.2720-$0.2970 (8B) against a
  projection of $1.03-$1.13 and $0.2745-$0.2857: the per-direction rate was right and the
  stage overran by 3% ($179.34 against $174) entirely through the timeout accounting.
- The ~$0.03 failed-launch line (`--root <scratch>` relocates inputs too) appears in both GCG
  spend tables; it is counted once here, in the shakeouts row.
- The 64-direction row stays a projection (E) and is SUPERSEDED as a description of what was run:
  it is kept only for cost-per-direction extrapolation beyond 32. From SMOKES's split of each
  arm's `gpu_seconds` into search s/dir and fixed s/call: per base, all 8 arms at 64 dirs
  **$95.32 (8B) / $363.88 (27B)**, $459.20 both.
- The exploratory pass (smoke draw, 8 directions per arm, both `gcg` and `epo`) is reported
  separately from the final arms throughout: it used a different root and a different target
  draw. The final arms are all full root at 32 targets — `gcg` first, then `epo` on the same
  selections (realact rows 0-31 and the stratified `sae` rows).
- Number of documents forwarded by `targets` at 16M: n/a in SMOKES (only the 512-per-family
  target counts and the wall are recorded; the smoke run's realact pool was 550 documents at 1M
  on the 27B and is a 1,024-document sample at 16M).
