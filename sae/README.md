# `sae/`

The 2M-feature BatchTopK SAE on the layer-42 residual stream (trained feature-parallel), its max-activating examples, and the SFT bank built from them.

| file | what it does |
|---|---|
| `anchor_worker.py` | torchrun worker for the 2M-SAE bank END-ANCHOR check (one rank per GPU, no process group needed). |
| `bank_lib.py` | Pure helpers for the 2M-SAE midtrain bank (CPU-testable; no Modal, no model). |
| `modal_sae2m.py` | Modal app `maemm-sae2m`: the 2,097,152-feature (2^21) BatchTopK SAE on Qwen3.6-27B layer-42 residuals with ONLINE activation generation (no stored ... |
| `modal_sae2m_bank.py` | Modal app `maemm-sae2m-bank`: the SFT MIDTRAIN BANK of the 2,097,152-feature layer-42 SAE (/data/sae2m/trainer_0/ae.pt) from its max-activating ... |
| `online_gen.py` | ONLINE layer-42 activation generation for the 2M-feature SAE (no stored activation shards). |
| `sae27b_disk_buffer.py` | Shuffling disk-backed activation buffer feeding dictionary_learning's trainSAE. |
| `sae27b_maxacts_merge.py` | Merge the per-rank partial tables of sae27b_maxacts_sharded.py into one top-N store + live/dead summary. |
| `sae27b_maxacts_sharded.py` | Data-parallel max-activating-examples for a huge SAE (F=2^21) on Qwen3.6-27B layer 42 (torchrun, R ranks). |
| `sae27b_merge_shards.py` | Assemble the R feature-shard checkpoints of sae27b_train_sharded.py into ONE dictionary_learning-style ae.pt. |
| `sae27b_mt_prefetch_buffer.py` | DiskActBuffer with a multi-reader background prefetch thread (+ optional shard-subset for data-parallel readers). |
| `sae27b_train_sharded.py` | Feature-parallel BatchTopK SAE trainer for Qwen3.6-27B layer-42 residuals (torchrun, R ranks, F = 2^21). |
| `sae27b_verify_2m.py` | Format + reconstruction check for a LARGE (bf16, F=2^21) merged ae.pt on one GPU (weights 43 GB fit an H200). |

Each script's docstring gives its exact invocation.
