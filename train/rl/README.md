# `train/rl/`

RL stage (GRPO): reward is the clean-model cosine between the generated text's activation and the injected direction. `rl_disagg.py` splits vLLM rollout GPUs from trainer GPUs; `rl.py` holds the shared rollout/scoring code it imports.

| file | what it does |
|---|---|
| `fast_lens_ext.py` | Fast vllm_lens worker extension for the disaggregated RL rollout ranks (train/rl_disagg.py). |
| `modal_rl_disagg.py` | Modal app: DISAGGREGATED Dr.GRPO/GRPO RL for the MAEMM inverter (train/rl/rl_disagg.py) -- X vLLM rollout GPUs + Y HF trainer GPUs in ONE container. ... |
| `rl.py` | Dr. GRPO-style RL for the MAEMM inverter — HF-generate rollouts, LoRA actor, data-parallel over groups. |
| `rl_disagg.py` | Disaggregated GRPO for the MAEMM universal inverter: X vLLM ROLLOUT GPUs + Y HF TRAINER GPUs in ONE container (N = X + Y processes, one GPU each), ... |
| `rl_fullparam.py` | --full-param for train/rl/rl_disagg.py: the RL POLICY is the WHOLE Qwen3.6-27B (every weight trainable), not a LoRA. |

Each script's docstring gives its exact invocation.
