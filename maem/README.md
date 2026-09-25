# `maem/`

The core library every stage imports: configuration, activation injection and read hooks, the shared prompt, and SAE loading.

| file | what it does |
|---|---|
| `config.py` | Central config. Scales from K=10k (pilot) to K=1,000,000 clusters by changing CLUSTERS. |
| `inject.py` | Norm-matched activation injection (activation-oracle formula) + residual read hook. |
| `mfu.py` | Honest MFU meter. Numerator counts only REAL (non-pad) tokens → padding lowers MFU, as it should. |
| `prompts.py` | Shared prompt: a single ` ?` marker whose residual gets the injected direction at INJECT_LAYER. |
| `sae.py` | BatchTopK SAE (adamkarvonen/qwen3-8b-saes, dictionary_learning format) — minimal loader. |

Each script's docstring gives its exact invocation.
