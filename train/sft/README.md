# `train/sft/`

Supervised stage: the inverter learns to write the real corpus span behind each direction. LoRA by default, full fine-tune with `--full-ft`.

| file | what it does |
|---|---|
| `fp8.py` | --fp8-base: run the FROZEN base nn.Linear layers of a PEFT-wrapped model in torchao float8 (experimental, default off). |
| `fullft.py` | --full-ft for train/sft/pretrain.py: every weight of Qwen3.6-27B trainable, sharded with torch FSDP2 (fully_shard). |
| `modal_sft.py` | Modal app: universal-inverter SFT (train/pretrain.py) on 8xB200, one container per datamix. |
| `prefix_cache.py` | Prefix-cached SFT forward: run the shared prompt prefix ONCE per micro-batch, expand its cache to the batch, and run only ``[marker] + prompt tail + ... |
| `pretrain.py` | Stage 4: pretrain the generator on the (direction, target_text) firehose. |

Each script's docstring gives its exact invocation.
