# `evals/faithfulness/`

The main evaluations. `precompute/` produces every artifact (targets, rollouts, scores, baselines) on Modal; `results/` turns them into the paper's tables and figures. `config.yaml` registers every model, SAE and held-out set.

| file | what it does |
|---|---|
| `autointerp/bands.py` | Stage `bands`: per-item recall by activation band, joined on the volume. |
| `autointerp/build.py` | Stage `build` (P2 of the autointerp design): the rendered example sets and the shared test set. |
| `autointerp/chain.py` | Stage `chain`: the whole remaining autointerp run as ONE detached Modal call. |
| `autointerp/modal_app.py` | The `autointerp` stages' own Modal app -- P1 (GPU), P2 (CPU) and the LLM run (CPU + the Anthropic API). |
| `autointerp/run.py` | Stage `run`: the LLM half -- Delphi's explainer, then its detection and fuzzing scorers. |
| `autointerp/sae_self.py` | Product `sae_self` (P1 of the autointerp design): the target feature's PER-TOKEN activation on its own MAEMM rollouts. |
| `autointerp/selfcheck.py` | Run the WHOLE chain's control flow locally, with the API stubbed, before any launch. |
| `autointerp/stats.py` | The autointerp analysis layer: the pilot's tables, from the small files one `run` wrote. |
| `autointerp/third_party/delphi-4fea06e/explainers/default/prompts.py` |  |
| `autointerp/third_party/delphi-4fea06e/scorers/classifier/fuzz.py` |  |
| `autointerp/third_party/delphi-4fea06e/scorers/classifier/sample.py` |  |
| `autointerp/third_party/delphi-4fea06e/scorers/classifier/prompts/detection_prompt.py` |  |
| `autointerp/third_party/delphi-4fea06e/scorers/classifier/prompts/fuzz_prompt.py` |  |
| `features/activations.py` | The activation-target registry: provenance and split side for the non-SAE families. |
| `features/bundle.py` | Read-only access to the v2 bundle (the immutable upstream data). |
| `features/corpus_train_parity.py` | Build a search corpus from the checkpoint's OWN training document ranges. |
| `features/doc_dedup.py` | Document-level near-duplication: did the model see the DOCUMENT, not just the span? |
| `features/draw_sae131k.py` | Draw the shared 131k-SAE target set: 2,000 held-out features, 80/20. |
| `features/draw_sae2m.py` | Draw the standard sae2m target set: 40,000 eval-split features, 80/20. |
| `features/emit.py` | Lay the registries out as <family>/{README.md,train/,test/}. |
| `features/fetch_hf.py` | Fetch an HF repo into the volume's HF cache at a PINNED revision. |
| `features/heldout_v2.py` | Wrap the frozen v2 directions as a held-out set the pipeline already reads. |
| `features/heldout_v3.py` | The eval-1 (faithfulness) target blocks: the v3 families, and our own controls. |
| `features/layout.py` | Precompute output layout: one head folder per family, train/ and test/ inside it. |
| `features/ngram_overlap.py` | n-gram overlap between the eval realact targets and the v2 training text. |
| `features/pipeline.py` | One command for a whole evaluation chain: draw -> rollouts -> score -> scan. |
| `features/registry.py` | The 2M-SAE feature registry: one row per feature, train/test side, rarity stratum. |
| `features/spawn.py` | Spawn a precompute product DETACHED, so a long run outlives this client. |
| `gcg/collect.py` | Read ONE gcg arm back -- whole, or as the union of its `--rows` chunks. |
| `gcg/gcg.py` | Product `gcg`: discrete-token search on the scorer's own objective -- the reachability ceiling. |
| `gcg/modal_app.py` | The `gcg` product's own Modal app -- one arm per call, on precompute/'s image chain. |
| `gcg/selftest.py` | CPU unit smoke for the M3 discrete-search changes. No GPU, no weights, no volume, no network. |
| `precompute/centred.py` | Product `centred`: the two SECONDARY per-rollout cosines, added to an existing scores dir. |
| `precompute/common.py` | The only module shared across evals/faithfulness: config, model loading, injection, reading, scoring, io. |
| `precompute/corpus.py` | Product `corpus`: the held-out token slice every other base product indexes into. |
| `precompute/modal_app.py` | The only Modal file in precompute/: one app, one image chain, one entrypoint. |
| `precompute/nll.py` | Product `nll`: the base model's own competence on each target's window (design §5, review R6). |
| `precompute/ood_selfcheck.py` | Product `ood_selfcheck`: every OOD code path that `unit_smoke` cannot reach, before any launch. |
| `precompute/patchscopes.py` | Product `patchscopes`: the zero-shot patching baseline, on the CLEAN BASE. |
| `precompute/repo_examples.py` | Product `repo_examples`: score the SAE repo's OWN max-activating windows. |
| `precompute/rollouts_hf.py` | Product `rollouts_hf`: MAEMM rollouts on the HF `generate` path. |
| `precompute/rollouts_nla.py` | Product `rollouts_nla`: the NLA activation-verbalizer baseline on the HF `generate` path. |
| `precompute/rollouts_vllm.py` | Product `rollouts_vllm`: the same rollouts as `rollouts_hf`, through a vLLM engine. |
| `precompute/scan.py` | Product `scan`: pass B over the corpus -- the corpus-retrieval baseline and the SAE examples. |
| `precompute/score.py` | Product `score`: score rollouts on the CLEAN BASE -> `<root>/maemms/<base>/<maemm>/scores/<set>/`. |
| `precompute/stats.py` | Product `stats`: pass A over the corpus. Target-INDEPENDENT, so it runs once per base and is never invalidated by a new held-out set. |
| `precompute/targets.py` | Product `targets`: draw one held-out set -> `<root>/base/<base>/heldout/<set>/`. |
| `precompute/tierb.py` | Product `tierb` (CPU): cos > 0.999 of OUR target blocks against the v2 TRAINING rows. |
| `precompute/top1_act.py` | Product `top1_act`: does the corpus search's TOP-1 window actually make the feature fire? |
| `precompute/unit_smoke.py` | CPU unit smoke for precompute/common.py -- no weights, no GPU, no network. |
| `precompute/vllm_ext.py` | Fast vllm_lens worker extension: a copy of train/rl/fast_lens_ext.py with the hook try/except REMOVED (errors raise instead of leaving a request ... |
| `reconstruction/act_smoke.py` | How hard does a target feature actually fire, on each kind of text that claims to describe it? |
| `reconstruction/corpus_top1_activation.py` | `paper/inversion-eval/data/corpus_top1_activation.csv` -- does the corpus search's TOP-1 window for an SAE feature actually make that feature fire? |
| `reconstruction/parity.py` | Do the HF and vLLM engines produce the same MAEMM? CPU, local, no volume access. |
| `reconstruction/repro_run1.py` | Does our pipeline reproduce run1's archived numbers? CPU, local, no volume access. |
| `reconstruction/sae_smoke64.py` | How hard does a target feature fire on each source, at a MATCHED number of draws? |
| `reconstruction/stats.py` | The analysis layer of evals/faithfulness: every table the paper reads, from the small files on the volume. |
| `reconstruction/stats_ood.py` | The OOD generalisation evaluation's analysis layer (design infra/2026-09-18_ood-eval-design.md §6, §11). |
| `results/autointerp.py` | Eval 2's tables and figures: the SAE autointerp comparison, from the runs already on the volume. |
| `results/autointerp_cases.py` | The four-arm CASE STUDY page: the same features seen through C16, M-top16, M-cos16 and M-jac16. |
| `results/autointerp_encdec.py` | Eval 2, encoder vs decoder: the same 512 features of one SAE explained from the ENCODER column and from the DECODER row, paired feature by feature. |
| `results/common.py` | Shared reading, discovery and estimation for `results/` -- the paper's results driver. |
| `results/corpus_search.py` | Eval 1's corpus-search baseline: panel a row 3, the corpus-size curve, and their cells. |
| `results/faithfulness.py` | Eval 1's tables and figures, from the products already on the volume. |
| `results/feature_page.py` | The per-feature autointerp page: what each arm's explainer SAW, what it WROTE, and what it SCORED. |
| `results/ood.py` | Eval 3's arms table: the generalisation eval, from the products already on the volume. |
| `results/ood_encorp.py` | M9: the Exemplifier's best-of-8 against the 10M ENGLISH TRAINING-CORPUS search, per OOD arm. |
| `results/patchscopes.py` | M7's reader: the Patchscopes appendix table, from the cells `score --rollouts-dir` wrote. |
| `results/selftest.py` | CPU unit smoke for `results/` -- no volume, no network, no GPU. |

Each script's docstring gives its exact invocation.
