# `evals/verbalization/`

Which SAE features the inverter can and cannot express, and why: rarity, feature kind, and the null control. `analysis/` is CPU-only.

| file | what it does |
|---|---|
| `modal_27b_rare.py` | Rare-feature mining targets and LoRA training for the 27B inverter. |
| `modal_27b_section.py` | The experiments behind the hard-to-verbalize section, on the pinned 27B checkpoint. |
| `modal_8b_verbalization.py` | Qwen3-8B MAEM rarity replication — self-contained, runs in ANY Modal workspace (no maem-data). |
| `modal_llm_baseline.py` | Can an off-the-shelf LLM write text that fires an SAE feature the MAEM cannot? |
| `analysis/analyze_consistency.py` | Are the inverter's attempts CONSISTENT on features it cannot activate? |
| `analysis/arm_compare.py` | Before/after a rare-feature adapter, on one test set: per rarity decile, and per cluster side. |
| `analysis/criterion.py` | THE failure criterion of the 27B unverbalizability analysis, in one place. |
| `analysis/diversity_by_model.py` | Output diversity across models on the 131k-SAE 2k set: token-3-gram Jaccard per feature. |
| `analysis/diversity_embed.py` | Semantic diversity on the 131k-SAE 2k set: spread, specificity and coverage in embedding space. |
| `analysis/dump_feature_examples.py` | Dump what a feature actually fires on, next to what the inverter said about it. |
| `analysis/dumplib.py` | Shared loaders for the eval_dirs dumps, and the figure scaffolding built on them. |
| `analysis/featlib.py` | Per-feature diagnostics for the bad-feature analysis, from the SAE max-acts tensor. |
| `analysis/find_bad_features.py` | Which SAE features does the inverter fail on, and what KIND of thing are they? |
| `analysis/from_precompute.py` | Map evals/faithfulness products onto this folder's `perdir_*.json` schema, so the 8B figure code runs unchanged on the 27B. |
| `analysis/how_structural_pass.py` | When a structural feature DOES activate, does the MAEM reproduce the structural event, or reach the feature some other way? |
| `analysis/kind_vs_rarity.py` | Which KINDS of feature fail more than their rarity predicts? |
| `analysis/label_categories.py` | Which CATEGORIES of feature are unverbalizable -- from blind LLM labels of each feature's windows. |
| `analysis/llm_vs_maem_512.py` | The LLM baseline against the MAEM on the paper's 512-feature set (2026-09-21_v3_ctrl, §4.4). |
| `analysis/mechanism_groups.py` | Group features by MECHANISM -- what token event they fire on -- not by topic, and ask which mechanisms the MAEM cannot verbalize beyond what rarity ... |
| `analysis/plot_8b_rarity.py` | fig5 for the Qwen3-8B inverter: unverbalized fraction vs feature rarity, under several verbalizability definitions PLUS the per-feature null the 27B ... |
| `analysis/plot_8b_training_curve.py` | 8B MAEM training curve (the run1, /vol/checkpoints/runs/run1): held-out cosine and SAE firing over SFT steps then RL steps, plus the RL training ... |
| `analysis/plot_cos_by_family.py` | Recovery-cosine distributions by DIRECTION FAMILY, from the evaluator's per-direction dumps. |
| `analysis/plot_diversity_by_model.py` | fig10 -- how repetitive each model's texts for one feature are, on the 131k-SAE 2k set. |
| `analysis/plot_diversity_embed.py` | fig11 -- the MAEM's samples for one feature, in two panels (131k-SAE 2k set, k = 8 per feature). |
| `analysis/plot_generalization_failure.py` | fig9 -- the headline claim, in one figure. |
| `analysis/plot_hard_to_get.py` | Separate HARD TO GET from NOT MEASURABLE, using each feature's own negative sample. |
| `analysis/plot_sl_vs_rl.py` | SFT (sft-simple2m) vs RL (rl-final) on the paper's eval sets, with the base model and NLA as references. Reads the `score` products' ... |
| `analysis/plot_unified_sl_rl.py` | SFT -> RL, 8B and 27B MAEMs on one footing: the same pipeline (rollouts_vllm n=64, `score`, unbiased best-of-k), each base's own held-out set (8B: ... |
| `analysis/plot_unverbalized_criteria.py` | Unverbalized fraction vs feature rarity under SEVERAL verbalizability definitions (27B arms). |
| `analysis/recovery_vs_rarity.py` | Is the inverter's per-feature recovery explained by how RARE the feature is in the corpus? |
| `analysis/structural_tags.py` | Structural token events a feature fires on, and which ones the MAEM fails beyond rarity. |
| `analysis/style.py` | Shared figure palette for the verbalization figures. |
| `analysis/unverbalizable_groups.py` | Are there GROUPS of features the MAEM cannot verbalize at any rarity -- and when some members of a hard group pass, what is different about them? |

Each script's docstring gives its exact invocation.
