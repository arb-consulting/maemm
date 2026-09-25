# `data/`

Activation collection and the (direction, target-span) banks the inverter trains on. The `modal_*.py` launchers run the GPU/CPU jobs; the `*_worker.py` files are what they run.

| file | what it does |
|---|---|
| `bsf_verbatim_worker.py` | BSF-VERBATIM worker (one GPU per process) for data/modal_bsf_verbatim.py. |
| `build_big_sft_bank.py` | Build a BIG realact+probes SFT bank (CPU-only) for the scaled universal-inverter run. |
| `build_rl_bank.py` | Build the balanced RL data-mix bank (CPU): realact(short) + probes + realact_long, ~equal thirds. long-context is ONE component of the mix, not the ... |
| `build_universal_bank.py` | Build the MIXED "universal" match-activation bank for the Qwen3.6-27B direction->text inverter. |
| `collect_acts.py` | Collect layer-42 residuals from Ultra-FineWeb for (a) BSF/SASA training and (b) SFT firing-context spans. Takes the first `seq_len` tokens of each ... |
| `collect_acts27b_worker.py` | Per-GPU worker: Qwen3.6-27B layer-42 per-token residuals + token ids over FineFineWeb. |
| `collect_acts_longctx.py` | Collect LONG-CONTEXT layer-42 activations as an RL-ONLY data source (the observation: long contexts have genuinely different features; use them for ... |
| `collect_bank_worker.py` | Sample-and-emit SFT-bank collector (one GPU per process). |
| `mlp42_bank_worker.py` | Layer-42 MLP neurons of Qwen/Qwen3.6-27B as a SIXTH inverter direction family (container side; launched by data/modal_mlp42_bank.py). Builds, from ... |
| `mlp42_neurons_worker.py` | Layer-42 MLP neuron analysis for Qwen/Qwen3.6-27B — container-side logic (launched by modal_mlp42_neurons.py). |
| `mlp42_pairs_worker.py` | Sparse COMBINATIONS of layer-42 MLP neurons: co-firing pairs / triples, their composite write directions, SAE matching and inverter verbalization ... |
| `modal_acts27b.py` | Modal app: regenerate per-token layer-42 activations + token ids for Qwen/Qwen3.6-27B over m-a-p/FineFineWeb onto the `maemm-data` volume (replaces ... |
| `modal_acts27b_fresh.py` | Modal app `maemm-acts27b-fresh`: a FRESH 512-token layer-42 activation store at /data/acts27b_fresh on `maemm-data`. |
| `modal_bank_everything.py` | Modal app: build the "EVERYTHING" RL direction bank at /data/banks/everything on `maemm-data`. |
| `modal_big_bank.py` | Modal app: build the BIG realact+probes SFT bank at /data/banks/big_rp (CPU only, no GPU). |
| `modal_bsf_retrain.py` | Retrain the Qwen3.6-27B L42 block-sparse featurizer (SASA / BSF) on Modal, 1x B200. |
| `modal_bsf_verbatim.py` | Modal app `maemm-bsf-verbatim`: the BSF-VERBATIM midtrain bank at /data/banks/<out_name> (default bsf_verbatim_1m) on `maemm-data`. |
| `modal_collect_bank.py` | Modal app: collect a BIG real-activation SFT bank directly (sample-and-emit), no activation store. |
| `modal_mix_5m_bank.py` | Modal app `maemm-mix-5m-bank`: compose the ~5M-row, 8-family SFT MIDTRAIN bank /data/banks/mix_5m from finalized source banks, with per-family row ... |
| `modal_mlp42_bank.py` | Modal app `maemm-mlp42-bank`: layer-42 MLP neurons (singles + co-firing pair composites) as the SIXTH inverter direction family — bank ... |
| `modal_mlp42_neurons.py` | Modal app: layer-42 MLP neurons of Qwen/Qwen3.6-27B -> do the sparsely-activating ones write structured directions? |
| `modal_sae_maxacts_fresh.py` | Modal app `maemm-sae-maxacts-fresh`: per-feature max-activating 32-token windows of OUR layer-42 SAE (/data/sae/ae.pt, F=131072 k=64) over the FRESH ... |
| `train_sasa.py` | Train a SASA block-sparse subspace featurizer on collected L42 activations (Qwen3.6-27B). |

Each script's docstring gives its exact invocation.
