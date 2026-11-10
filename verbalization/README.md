# Verbalization eval

Everything for one question: **when the inverter is handed an SAE feature direction, can it write
text that actually fires that feature — and when it can't, why not?**

This folder is self-contained. Nothing in it imports `mxf/`, `sft/`, `rl/` or `data/`, and every
model artifact it needs is public on HF (base model, adapter, SAE, max-acts) — so it can be copied
into another repo as a unit. `mxf/config.py` hardcodes the 27B's `d_model` 5120 and read-layer 42,
which is why the Modal app re-implements the injection/scoring path rather than importing it.

```
verbalization/
  modal_8b_verbalization.py   all GPU work: one Modal app, seven jobs
  analysis/                   local, CPU-only: figures and tables from the dumps
  report/                     README.md (the writeup), figN.*, data/ (dumps), tables/ (bad features)
  requirements.txt
```

Read `report/README.md` for the findings. This file is about running it.

---

## The pipeline

```
                modal_8b_verbalization.py                    analysis/
  corpus ──► scan_fire ─────► sae_match_8b.npz ──┐
                                                 ├──► plot_8b_rarity.py ──► fig6, fig8
  features ─► eval_dirs ─────► perdir_8b_*.json ─┤    plot_hard_to_get.py ─► fig7
                             └─ texts_8b_*.json ─┤    find_bad_features.py ─► tables/
                                                 └──► dump_feature_examples.py
  features ─► mine_targets ──► mined_train.jsonl
                  └──────────► train_rare ──► /data/adapters/rare ──► eval_dirs (the "after" arm)
```

`eval_dirs` is the measurement everything rests on: inject `unit(W_enc[:,f])` at layer 1 on the ` ?`
marker, generate best-of-`bo` with the adapter **on**, then re-read each generated text through the
clean base with the adapter **disabled** and score it against **all 65,536 features**. Scoring every
feature (not just the paired one) is what yields the per-feature null for free — feature `f`'s
activation on texts generated for *other* features. That null is the control: this adapter's own
model card reports that a direction-agnostic baseline matches the headline score, so without a null
you cannot separate "verbalized `f`" from "`f` fires on any fluent text".

## The Modal jobs

The app and volume are still named `maemm-8b-rarity`, unchanged by the move, so dumps already on the
volume stay reachable. Only the file was renamed.

| job | what | cost |
|---|---|---|
| `probe` | validates the fragile paths (injection during generate, clean-base re-read) and picks the prompt variant | ~5 min |
| `scan_fire` | corpus scan → per-feature firing rate: the rarity axis | ~1.02M tokens |
| `eval_dirs` | inverter eval + per-feature null → `perdir_8b_<tag>.json`, `texts_8b_<tag>.json`, `act_profile_8b_<tag>.npy` | scales with `n_features × bo` |
| `mine_targets` | top-K spans per feature over a long scan → training targets for rare features | 10M tokens ≈ 2 GPU-h |
| `train_rare` | LoRA SFT on the mined targets, from the SFT init (`ref/`) so RL is not a confound | ~1 GPU-h |
| `score_texts` | score arbitrary text against chosen features — the causal check on an interpretation | minutes |
| `logit_lens` | project a feature's decoder through the unembedding: what it *writes*, independent of what it co-occurs with | minutes |

Always run `probe` first after touching the injection or scoring path.

```bash
modal run verbalization/modal_8b_verbalization.py::probe --n-features 16
modal run verbalization/modal_8b_verbalization.py::scan_fire
modal run verbalization/modal_8b_verbalization.py::eval_dirs --n-features 512
modal volume get maemm-8b-rarity /out/perdir_8b.json verbalization/report/data/
modal volume get maemm-8b-rarity /out/sae_match_8b.npz verbalization/report/data/
```

Modal's CLI cannot parse list annotations on a remote function, so feature sets are passed as JSON
strings: `--features-json "[1,2,3]"`.

## Rebuilding the figures and tables

Everything below is CPU-only and runs from what is committed under `report/data/`, except
`sae_match_8b.npz` (ignored as a binary — one `modal volume get`, no GPU).

```bash
pip install -r verbalization/requirements.txt
D=verbalization/report/data

# fig6 — one arm, five criteria
python verbalization/analysis/plot_8b_rarity.py --perdir rl=$D/perdir_8b.json \
    --sae-match $D/sae_match_8b.npz --out verbalization/report

# fig8 — before/after rare-feature training on the 1,200 TRAIN-EVAL features.
# The --criteria subset is not the default; this is the invocation that produced the committed fig8.
python verbalization/analysis/plot_8b_rarity.py \
    --perdir before=$D/perdir_8b_traineval_sft_pre.json \
    --perdir after=$D/perdir_8b_traineval_post.json \
    --sae-match $D/sae_match_8b.npz --criteria bar,norm,null --out verbalization/report

# fig7 — hard-to-get vs not-measurable, three arms
python verbalization/analysis/plot_hard_to_get.py \
    --perdir sft=$D/perdir_8b_sft.json --perdir rl=$D/perdir_8b_rl.json \
    --perdir after=$D/perdir_8b_test_post.json \
    --sae-match $D/sae_match_8b.npz --out verbalization/report
```

The stem is chosen by arm count: one `--perdir` writes `fig6_*`, more than one writes `fig8_*`.

fig1–fig5 are the 27B arms; they need `perdir_ckpt_*.json` (`eval_ckpt_daemon.py --dump-per-dir`)
and `/mlp42/sae_match.npz` from the `maemm-data` volume, which this folder does not carry.

## Finding bad features

`fig6`/`fig8` say failure tracks rarity. They do not say what the failures *are*. That is what
`find_bad_features.py` is for: it walks every feature no arm can activate and reduces each one's
max-activating examples to a sortable row, then flags three distinct failure modes.

```bash
python verbalization/analysis/find_bad_features.py \
    --perdir rl=$D/perdir_8b_rl.json --perdir sft=$D/perdir_8b_sft.json \
    --perdir after=$D/perdir_8b_test_post.json --perdir rl512=$D/perdir_8b.json \
    --sae-match $D/sae_match_8b.npz --out verbalization/report

python verbalization/analysis/dump_feature_examples.py --format csv --limit 20 \
    --from-csv verbalization/report/tables/unactivatable_features.csv \
    --sae-match $D/sae_match_8b.npz \
    --perdir rl=$D/perdir_8b_rl.json --texts rl=$D/texts_8b_rl.json \
    --out verbalization/report/tables/unactivatable_examples.csv
```

- **TEMPLATE** — the 30 "examples" are near-duplicates of one boilerplate string (`template_J` high).
  There is no generalisable concept to verbalize; the feature memorised a fragment.
- **COLLOCATION** — the token *after* the peak is near-constant. The feature encodes a continuation
  (`Chamber of` → ` Commerce`), not a topic. Asking for text that "means" this is close to ill-posed.
- **UNRESOLVED** — neither flag fires and the inverter still gets nothing.

Selection is `--max-norm-act` (default 0.10) against the **most favourable** arm supplied, so passing
several arms is the conservative reading: a feature any arm can activate is dropped.

### Provenance of the committed tables

`tables/` was originally produced by ad-hoc code that was never saved. The scripts here are a
reconstruction, checked against the original output:

| table | status |
|---|---|
| `unactivatable_features.csv` | **reproduced** — all 8 per-feature columns match the original on all 900 shared rows (7,200/7,200 cells). `template_J` is redefined (below) and `flag` is new. |
| `collocation_features.txt` | reproduced; the illustrative `typical span` may quote a different example than the original, which picked one by hand. The detector columns are exact. |
| `rare_examples.txt` | reproduced — same selection (`1315 features fire on <= 0.00869%`) and layout. |
| `unactivatable_examples.csv`, `template_features_examples.csv` | reproduced in layout; row counts differ because the feature pool is now the explicit rule above rather than a hand-built candidate list. |
| `consistency_generations.csv` | reproduced — `self_cos`, group and generations match exactly on all 192 original rows; the script now emits all 96 features rather than a hand-trimmed 24. |
| `feature_taxonomy.txt`, `writeup_features.txt`, `unactivatable_features.docx` | **hand-written.** Curated readings of the above, kept as-is. Not regenerable. |
| `human_verbalization.csv` | **hand-written inputs**, scored by `modal_8b_verbalization.py::score_texts`. The texts are a person's attempts at the same task; the activations are reproducible, the texts are not. |

`template_J` is now the mean pairwise Jaccard over token **trigrams**, where the original used an
unrecovered variant. Trigrams rather than bare token sets because unigram overlap is dominated by
common words — every pair of English windows shares `" the"`, which puts a floor under the score.
Against the original column: Spearman 0.92, Pearson 0.97, and it independently selects 12 of the
same 14 template features. Separation is wide either way (templates 0.43–0.50, ordinary ~0.01).

## What is not committed

The Modal volume holds what is too large or too raw for git: the trained adapter
(`/data/adapters/rare`), the 38 MB mined bank (`/out/mined_train.jsonl`), and the
`act_profile_8b_*.npy` full activation matrices that the nulls are reduced from.
`report/data/sae_match_8b.npz` is ignored as a binary but is one `modal volume get` away.

The repo-wide `data/` ignore rule used to swallow `report/data/` — tables had to be `git add -f`'d
and then never appeared in `git status` again. `.gitignore` now un-excludes that directory and its
JSON specifically, so new dumps show up normally. `*.npz` stays ignored.
