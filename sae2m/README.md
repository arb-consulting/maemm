# sae2m — 2,097,152-feature BatchTopK SAE on Qwen3.6-27B layer 42, trained on Modal with ONLINE activations

Goal: a dictionary DIFFERENT from the 131k eval SAE (`/data/sae/ae.pt`) to build a midtrain bank from — F = 2^21, k = 64,
~1B Ultra-FineWeb tokens, then top-5 max-activating 32-token windows per feature. No stored activation store is used: every
training rank runs its own Qwen3.6-27B (layers 0..42 only) and streams a disjoint slice of Ultra-FineWeb.

Modal app **`maemm-sae2m-v2`** (`SAE2M_APP=maemm-sae2m-v2 modal deploy sae2m/modal_sae2m.py`; `maemm-sae2m` is the pre-fix deployment, see the stream-sharding bug below), volume `maemm-data`, everything under **`/data/sae2m/`**.

```
source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
cd ~/maemm-pub && modal deploy sae2m/modal_sae2m.py
```

## Files

| file | role |
|---|---|
| `modal_sae2m.py` | Modal app: `train_online` (B200:8), `gen_check` (B200:1), `merge` (CPU), `verify` (B200:1), `maxacts` (B200:8), `status`, `cleanup` |
| `online_gen.py` | truncated-27B loader, per-rank Ultra-FineWeb window producer (background tokeniser thread), `OnlineActGenerator`, `OnlinePool` |
| `sae27b_train_sharded.py` | feature-parallel BatchTopK trainer (from `max-activating-examples/scripts`) + `--data-mode online` |
| `sae27b_merge_shards.py` | 8 shard checkpoints -> `trainer_0/ae.pt` (dictionary_learning keys, bf16 matrices, norm_factor folded) |
| `sae27b_verify_2m.py` | format + EV/L0/fired check; `--online` generates the eval tokens on the fly |
| `sae27b_maxacts_sharded.py` / `sae27b_maxacts_merge.py` | data-parallel top-N windows per feature + cross-rank merge |
| `sae27b_disk_buffer.py`, `sae27b_mt_prefetch_buffer.py` | disk data modes (unused on Modal, kept so the trainer/tests are unchanged) |
| `tests/` | `run_tests.sh`: 30 equivalence tests + 10 online-path tests on 2 gloo CPU ranks (~3 min, no GPU) |

## Pipeline (spawn snippets — each is a deployed-app call, survives this terminal)

```bash
source ~/modal_venv/bin/activate; export MODAL_PROFILE=safety-sahan
S='import modal, sys; f=modal.Function.from_name("maemm-sae2m-v2", sys.argv[1]); print(f.spawn(**eval(sys.argv[2] if len(sys.argv)>2 else "{}")).object_id)'

# 0) one-GPU sanity: truncated model == full model at layer 42, gen tok/s, memory  (~10 min)
python -c "$S" gen_check

# 1) TRAIN (8xB200, ~5-7 h for 1B tokens; checkpoints every 60k steps to /data/sae2m/shards/rank{r}/, keep 1)
python -c "$S" train_online
python -c "$S" train_online '{"resume": True}'            # only if the 24 h cap hit (trainer saves at --max-hours 23 and exits)

# 2) MERGE (CPU 16 / 256 GB / 1 TB ephemeral): shards -> /data/sae2m/trainer_0/ae.pt (43 GB) + config.json   (~20-40 min)
python -c "$S" merge

# 3) VERIFY (1xB200): EV / L0 / fired on 20M freshly generated tokens from the reserved eval head (docs 0..99,999)  (~30-45 min)
python -c "$S" verify                                     # -> /data/sae2m/verify.json

# 4) MAXACTS (8xB200, ~5 h): top-5 32-token windows per feature over the SAME 1B-token span -> /data/sae2m/maxacts_top5.pt + .summary.json
python -c "$S" maxacts
python -c "$S" maxacts '{"max_tokens": 20_000_000, "out_dir": "/data/sae2m/maxacts_parts_smoke", "final": "/data/sae2m/maxacts_top5_smoke.pt"}'   # smoke

# poll / status
python -c 'import modal, sys; print(modal.FunctionCall.from_id(sys.argv[1]).get(timeout=0))' <fc-id>      # raises TimeoutError while running
python -c 'import modal, json; print(json.dumps(modal.Function.from_name("maemm-sae2m","status").remote(), indent=1, default=str))'
modal volume get maemm-data sae2m/logs/<train_...>.log -   # full trainer stdout (committed every 120 s)
```

The 2 -> 3 -> 4 chain can be run back-to-back once `/data/sae2m/shards/TRAIN_DONE` exists (`merge` asserts it; pass
`require_done=False` to merge a partial run's latest complete shard set).

## Launched runs (ledger: ~/shared/overnight/sae2m_ids.json)

| what | FunctionCall | spawned (UTC) | notes |
|---|---|---|---|
| gen_check | fc-01M26RJH0VQJY9FWPDGPMH3RHQ, fc-01M26RSJCQ2PVMVF0HD73BNMBW | 2026-09-10 22:5x / 23:00 | numbers below; 2nd run tripped the (then too-tight) absolute equivalence assertion |
| smoke train_online (20M tok) | fc-01M26RSJMACTN24E4623SWHKBA | 2026-09-10 23:00 | TRAIN_DONE 23:31Z; shards in /data/sae2m/shards_smoke (delete with `cleanup`) |
| full train_online attempt 1 | fc-01M2705HT8D4FNNG66WFQ3E40Q | 2026-09-11 01:09 | cancelled at ~step 300 (see teardown hang below); nothing kept |
| **FULL train_online (1B tok), attempt 2** | **fc-01M270K1MJVR4C0RKMDJ69SGG7** | **2026-09-11 01:16** | wandb `BatchTopK-2M-l42-online`; 244,140 steps, saves every 60k (keep 1), max_hours 23 |

After the full run: `merge` -> `verify` -> `maxacts` (snippets above). If the call ends without `/data/sae2m/shards/TRAIN_DONE`
(24 h cap / preemption), spawn `train_online` with `{"resume": True}` -- it resumes from the latest complete shard set and skips each
rank's already-consumed docs (`gen_state_step*.json`).

**Teardown hang (found in the smoke, fixed before attempt 2).** After `TRAIN DONE` + `wandb.finish()` the torchrun workers never
exited (interpreter finalisation with the HF-datasets/pyarrow streaming threads alive -- the same class of failure as the teardown
SIGABRT seen with `sae27b_gen_acts.py` on EUR-IS); the smoke call sat on 8 idle B200s for 1.7 h until cancelled, while all its
shards + TRAIN_DONE were already committed by the periodic committer. Fixes: the trainer and the maxacts script end with
`os._exit(0)` after their final barrier (everything is flushed before), and `_run` has a watchdog that SIGKILLs the process group
10 min after `TRAIN_DONE` appears if it is still alive (rc treated as 0). `gen_check`'s equivalence assertion is also relative to
the model's own bf16 noise floor in this deployment.

## STREAM-SHARDING BUG (found 2026-09-11 by the bank builder; fixed, deployed as app `maemm-sae2m-v2`)

The first `open_rank_stream` did `split_dataset_by_node(ds, rank, world).skip(100_000)`. In `datasets` 4.5 a `.skip()` on a
distributed IterableDataset creates a `SkipExamplesIterable(split_when_sharding=False)` whose `shard_data_sources()` returns `self`,
so the per-rank shard selection is silently dropped: **all 8 ranks iterated the identical full stream** (the smoke max-acts had 79%
duplicate windows; doc ids differed only by the rank term). Consequences:

* **The finished 1B-token SAE (`/data/sae2m/trainer_0/ae.pt`, fc-01M270K1MJVR4C0RKMDJ69SGG7) saw ~125M UNIQUE tokens, each ~8 times**
  (every rank generated the same ~255k documents = single-stream docs 100,000..~355,000 and drew each of its rows once; the 8 ranks'
  512-row contributions per step were different random rows of the same document pool). tokens_seen = 999,997,440 presentations,
  ~125M distinct. It is a valid SAE trained ~8 epochs over 125M tokens, not a 1B-token SAE -- decide whether to retrain.
* The smoke max-acts (`maxacts_top5_smoke.*`, `maxacts_parts_smoke/`) and the cancelled full max-acts partials (`maxacts_parts/`) are
  from the duplicated stream -- do not use them.

Fix (`online_gen.open_rank_stream` / `shard_stream`): ONE single stream `load_dataset(streaming).skip(100_000)`, then rank r takes
`itertools.islice(stream, rank, None, world)` -> rank r sees single-stream docs `100_000 + r + 8k`; `doc_id = k*8 + r` therefore equals
`single-stream index - 100_000`. Resume = `islice(stream, rank + world*docs_iterated, None, world)`. Verified on the real Ultra-FineWeb
stream: ranks 0/1/7 first docs == single-stream docs 100000/100001/100007 (and 2nd/3rd == +8/+16), 3/3 distinct first docs, 0 shared docs
in the first 300 of each, resume offset exact; CPU test `(f2)` covers the semantics with fake streams. `verify` (world 1, skip 0) is
unaffected. Both the trainer's generator and `sae27b_maxacts_sharded.py` go through the fixed function; every rank now logs its
FIRST WINDOW (doc id + text) so disjointness is visible in the log.

## Outputs

```
/data/sae2m/shards/rank{0..7}/ae_shard_step{N}.pt   fp32 W_enc/W_dec/b_enc + replicated b_dec/threshold + Adam state (~32 GB each, keep 1)
/data/sae2m/shards/rank{r}/gen_state_step{N}.json   docs_iterated of that rank's stream (resume skips them)
/data/sae2m/shards/{config.json,latest.json,TRAIN_DONE,wandb_id.txt}
/data/sae2m/trainer_0/{ae.pt,config.json}           encoder.weight [F,d] bf16, decoder.weight [d,F] bf16 (unit cols), encoder.bias/b_dec fp32, k, threshold
/data/sae2m/verify.json                             ev (global-centred), ev_batch, l0, frac_fired_in_eval, eval_tokens, gen stats
/data/sae2m/maxacts_top5.pt (+ .summary.json)       max_tokens [F,5,32] int32 (window ENDS at the peak, left-padded), max_acts [F,5] fp16,
                                                    lengths, doc_ids, positions, fire_counts [F]; summary: live/dead, fire-rate percentiles
/data/sae2m/logs/*.log, gen_check.json
```

## Conventions (matched to the stored 1B-token set the 131k SAE was trained on — `scripts/sae27b_gen_acts.py`)

* **Tokenisation**: `tok(text, add_special_tokens=False)`; non-overlapping windows `ids[s:s+512]` for `s in range(0, n-511, 512)`;
  docs shorter than 512 tokens and the tail remainder are dropped. Forward input = `[BOS=248044] + 512 content tokens`
  (513 positions); **only position 0 (BOS/sink) is dropped** -> 512 activation rows per sequence. (The brief said "BOS + 511,
  drop BOS and the first token after it" — that is NOT what gen_acts does; gen_acts' convention was followed.)
* **Read point**: output of decoder block 42 (`layers[42]` forward hook, `o[0] if tuple else o`) = `resid_post_layer_42`, identical
  to `mxf.inject.read_resid` and to the existing SAE. The model is truncated to layers 0..42 and the lm_head dropped; the
  hook aborts the forward at block 42, so the hidden states are bit-identical to the full model (`gen_check` asserts it).
* **Outlier drop**: tokens with `||x|| > 10 x median(||x||)` over the 16x512-token micro-batch are dropped (gen_acts / bank code).
* **Normalisation**: `--norm-target unit` (dictionary_learning `normalize_activations`): x / norm_factor with norm_factor = sqrt(mean ||x||^2)
  over the first 100 batches; shards stay in normalised space, `merge` folds norm_factor into b_enc / b_dec / threshold.
* **Corpus split**: `split_dataset_by_node(rank, 8)` THEN `.skip(100_000)` — rank r gets parquet files r, r+8, ...; the skip after the
  split is NOT divided across ranks, so every rank drops the first 100k docs of its stream and the single-stream eval head
  (docs 0..99,999 = the head of file 0001, rank 0's first file) is excluded outright. gen_acts skipped before splitting, which
  `datasets` divides into 12,500/rank — that leaked docs 12,500..99,999 into rank 0's stored acts. `maxacts` uses the same
  split-then-skip streams, i.e. exactly the training span; `verify` runs on `--dataset-skip 0` = the reserved head.
* **SAE hparams**: F=2^21, k=64, batch 4096 (8 ranks x 512 rows all-gathered), lr = 2e-4/sqrt(F/16384) = 1.77e-5, warmup 1000,
  linear decay from 80% of the steps, threshold EMA beta 0.999 from step 1000, auxk_alpha 1/32, top_k_aux = 2560*F/131072 = 40,960
  global (5,120 per rank), dead = not fired for 10M tokens, W_dec rows unit-norm, grad clip 1.0, Adam(0.9, 0.999).
* **Pool**: 262,144 rows/rank fp16 (2.7 GB); every row drawn at most `--pool-reuse-max 1` times; refill to full when < 50% drawable;
  generation is synchronous between steps (16x512-token 27B forward ≈ one per 16 steps per rank).

## Measured (gen_check, 1xB200, 2026-09-10)

* truncated model: 43/64 layers via the config path, **32.9 GB** bf16, loaded in 22-28 s from `/data/hf_cache`
* generation: **21.1k tok/s per GPU** (16x512-token micro-batch = 0.39 s), peak 36 GB -> ~24 ms of generation per training step per rank
  at 512 fresh rows/rank/step; 1B tokens = 1.6 h of pure generation per rank
* layer-42 residual norms: median 83, p99 109, max 151; 0 of 90k tokens hit the 10x-median outlier rule
* Ultra-FineWeb en: 44% of docs are >= 512 tokens, ~2.3 windows per used doc
* truncated vs full model at layer 42: rel-Frobenius 4.8e-3 / worst-token cos 0.9685 -- IDENTICAL to the full model vs ITSELF on the same
  input (3.9e-3 / 0.9685); back-to-back full/truncated forwards agree to 8e-4 / 0.99995. The bf16 forward (fla GDN kernels + SDPA) is not
  run-to-run deterministic; the truncation adds nothing beyond that noise floor (the check asserts this relatively).

## Measured (SMOKE train_online, 8xB200, 20M tokens = 4,882 steps, 2026-09-10, fc-01M26RSJMACTN24E4623SWHKBA)

* all 8 ranks: truncated model up in 38 s each (32.9 GB), norm_factor 83.36 (mean ||x||^2 = 6949 raw)
* **SAE step 106 ms (9.4 it/s, 38k tok/s) before any feature is dead, ~125 ms (8 it/s) once the aux-k path is active**;
  pool refill (131k fresh rows/rank) stalls the loop ~6.8 s every ~256 steps = 19-21k tok/s per GPU, i.e. **~27 ms/step amortised,
  generation ~20% of wall time**; fresh fraction 1.00 throughout (every row trained on once); 0 outlier tokens dropped
* **peak GPU memory 86 GB / 180 GB** per rank (33 GB model + SAE shard + Adam + dense [4096 x 262144] pre-acts + 2.7 GB pool)
* L0 64.5-67 in training (k=64 plus bf16 ties at tau), EV -7.8 (step 1) -> 0 at ~step 1200 -> **0.46 at step 4882**, loss 0.27,
  threshold 0.0257 normalised (= 2.14 raw), dead 0.31% at the end (aux-k just becoming active: auxk 1.00 -> 0.98)
* checkpoint set (8 x 32 GB = 258 GB) written to the volume in **87-104 s**; TRAIN_DONE + latest.json + gen_state json written
* wandb: https://wandb.ai/celestedeschamphelaere-personal/qwen36-27b-sae/runs/d524zcfa
* projection for 1B tokens (244,140 steps): ~0.135-0.15 s/step -> **9-10 h** (< the 23 h --max-hours budget of one call)

## Tests

```
bash sae2m/tests/run_tests.sh      # 40 PASS lines expected (30 sharded-equivalence + 10 online-path), 2 gloo ranks, CPU only
```
