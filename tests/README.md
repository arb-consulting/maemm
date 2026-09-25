# `tests/`

CPU tests, one folder per component. Run with `pytest tests/`.

| file | what it does |
|---|---|
| `heldout/test_eval_mlp_families.py` | Tests for the cache-v2 EXTRA eval families (layer-42 MLP neurons `mlp` / co-firing pairs `mlp_pair`) in evals/heldout/eval_universal.py — CPU by ... |
| `heldout/test_inline_extra_evals.py` | CPU-only unit tests for train/inline_extra_evals.py (pure parts: locality metrics, AUC math, judge response parsing, retry/accounting, the background ... |
| `rl/test_rl_disagg_autocast.py` | CPU unit test for rl_disagg's --autocast-bf16 policy-forward path (no GPU, no PEFT/HF needed). |
| `rl/test_rl_disagg_fast_cpu.py` | CPU unit tests for rl_disagg's trainer speed knobs (no GPU / HF model): --score-length-bucket's score reorder wrapper, the prefix-cache gradient ... |
| `rl/test_rl_disagg_fullparam.py` | CPU unit tests for rl_disagg --full-param (train/rl/rl_fullparam.py): flags, the publish manifest + FSDP2 shard geometry, the bf16 gather / fs shard ... |
| `rl/test_rl_disagg_policy_base.py` | CPU unit tests for rl_disagg's --policy-base plumbing (no GPU, no vLLM, no HF model): flag defaults / 'none' unsetting, the policy-base resolution + ... |
| `rl/test_rl_disagg_queue.py` | CPU unit tests for train/rl_disagg.py's filesystem plumbing (no GPU, no vLLM, no HF model): rollout-block queue (FIFO / drop-stale), adapter `latest` ... |
| `rl/test_rl_disagg_scalerl.py` | CPU unit tests for the ScaleRL variant of train/rl/rl_disagg.py (no GPU, no vLLM, no HF model): flag bundle resolution, CISPO vs PPO per-token ... |
| `sae/test_anchor_worker_cpu.py` | Plumbing test of sae/anchor_worker.py on CPU (2 torchrun ranks, --fake forward, synthetic ae.pt with d=16): anchor_in_r{r}.pt -> ... |
| `sae/test_bank_cpu.py` | CPU tests for the 2M-SAE bank builder (no GPU, no 27B): synthetic ae.pt / maxacts_top5.pt in the EXACT formats of sae/sae27b_merge_shards.py + ... |
| `sae/test_online_pool.py` | CPU/gloo tests for the ONLINE data path of the 2M-SAE trainer. Run (2 ranks): torchrun --standalone --nproc_per_node 2 tests/sae/test_online_pool.py ... |
| `sae/test_sharded_equivalence.py` | CPU/gloo equivalence tests for the sharded 2M-SAE pipeline. Run: torchrun --standalone --nproc_per_node 2 ... |
| `sft/test_head_on_labels.py` | CPU unit test for train/sft/pretrain.py's exact-speed paths (no GPU, tiny random Qwen3.5-arch model): |
| `sft/test_length_bucket.py` | CPU tests for train/sft/pretrain.py::step_groups -- the micro-batch composition of the per-example SFT path. |
| `sft/test_multibank.py` | CPU tests for train/sft/pretrain.py's multi-part bank loader (--data-dir a,b,c): load_shard / VecBank / TokRows. |
| `sft/test_prefix_sharing.py` | CPU correctness tests for sharing prefix gradients across optimizer micro-batches. |

Each script's docstring gives its exact invocation.
