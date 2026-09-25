# `evals/heldout/`

Held-out evaluation over direction families the inverter never trained on: recovery cosine, SAE activation and rank, % unverbalized, locality and autointerp.

| file | what it does |
|---|---|
| `autointerp_compare.py` | Merge two autointerp-detection score results (same features/positives/negatives, on-policy rollouts from two different adapters) into one ... |
| `autointerp_detection.py` | Autointerp DETECTION eval for the MAEM inverter: are N inverter rollouts a good *explanation* of a held-out SAE feature, measured as detection AUC ... |
| `build_ctx_eval.py` | Add held-out CONTEXT-LENGTH BUCKET families (realact_early <=512 / realact_mid 512-2048 / realact_long 2048-8192, by token position of the activation ... |
| `build_indist_eval.py` | Add IN-DISTRIBUTION held-out families from the ACTUAL RL training pool (pool_rl_mix) to the frozen eval cache (eval_sets_heldout.pt), so the eval ... |
| `eval_ckpt_daemon.py` | Checkpoint eval daemon (ONE GPU): the FULL held-out protocol of rl.py's inline_eval (every family, 512/family, Bo4, T=1, 16-64 new tokens, SAE ... |
| `eval_universal.py` | Universal held-out eval suite for the Qwen3.6-27B direction->text inverter (wandb-facing). |
| `inline_extra_evals.py` | Inline EXTRA evals for train/rl/rl.py — a standalone module (rl.py is untouched; the trainer wires these three calls in next to `inline_eval`). ... |
| `modal_autointerp_detection.py` | Modal launcher for the autointerp-detection eval's GPU stage (evals/heldout/autointerp_detection.py build): one B200, the maem-data volume (base ... |
| `modal_eval_ckpt.py` | Modal app `maem-eval-ckpt`: ONE-GPU checkpoint eval daemon for the RL runs (evals/heldout/eval_ckpt_daemon.py). |
| `modal_snippet_locality.py` | Modal launcher for the snippet-locality eval's GPU stage (evals/heldout/snippet_locality.py build): one B200, the maem-data volume (base-model HF ... |
| `modal_wildchat_bank.py` | Modal launcher for the ONE-TIME WildChat fire-prediction bank (evals/heldout/wildchat_bank.py): one B200, the maem-data volume (base-model HF cache ... |
| `snippet_locality.py` | Snippet-locality eval: within a text, does the target SAE feature fire on a LOCALIZED short snippet, or is the activation smeared across the whole ... |
| `wildchat_bank.py` | ONE-TIME bank for the inline WildChat fire-prediction eval (train/inline_extra_evals.py): 64-token windows of real WildChat-1M (English, non-toxic) ... |

Each script's docstring gives its exact invocation.
