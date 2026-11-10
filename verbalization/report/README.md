# Verbalizability vs corpus rarity

Does the inverter's per-feature failure to verbalize an SAE feature track how RARE that feature is
in the corpus? Short answer: yes under every definition tried, on two independently trained models
and two different SAEs — but the effect is weaker than the first pass suggested, and the headline
"unverbalized fraction" is a reporting choice, not a measured boundary.

**fig6 is the figure to work from.** It is the only one with a per-feature null control.

---

## The measurement

For a held-out SAE feature `f`, inject `v = unit(W_enc[:,f])` at layer 1 on the ` ?` marker
(`h ← h + ‖h‖·v`), generate best-of-`bo`, then re-read each generated text through the **clean base
model with the adapter disabled** and take `best_act` = max over samples of max over content tokens
of `f`'s raw pre-topk SAE activation. The adapter being off at read time is load-bearing: the model
cannot satisfy the metric by encoding something only its own LoRA recognises.

Everything downstream is a threshold on that one number.

## Why there are four definitions

`eval_universal.SAE_FIRE = 1.0` has **no derivation in the codebase** — it is the old
`eval_dirs --sae-fire` default, inherited. Across the 512 held-out 27B features the 1.0 bar spans
**1.8% to 561%** of a feature's own corpus peak (a 315× spread), and 9.2% of features have a corpus
peak *below* it, so they are unverbalizable by construction. Because `corpus_peak` correlates with
rarity, a fixed bar makes "unverbalized" partly a restatement of "rare".

It also does not transfer: the 8B SAE's median corpus peak is ~103 against the 27B's ~3.72, so 1.0
means something entirely different there.

| definition | what it asks | 27B rl | 8B |
|---|---|---|---|
| `best_act > 1.0` | inherited absolute bar | 0.33 | 0.26 |
| `norm_act ≥ 0.10` | ≥10% of the feature's corpus peak | 0.04 | 0.38 |
| ≥ top-16 example | would place in the feature's top-16 corpus examples | 0.79 | 0.80 |
| ≥ weakest top example | as strong as the weakest of its known examples | 0.66 | 0.76 |
| **> own null p95** | **beats `f`'s activation on OTHER features' texts** | — | **0.29** |

Base rates move ~20×; the rarity relation survives all of them (AUC 0.67–0.91).

## The null control (8B only)

The 8B adapter's model card reports that a **direction-agnostic control** — the same RL fed random
isotropic directions — reaches ≈ the same Bo4 SAE-holdout score, i.e. the aggregate metric is largely
gameable by generic fluent text. `modal_8b_verbalization.eval_dirs` therefore scores every generated text
against **all 65,536 features**, not just its paired one, which yields each feature's activation on
texts generated for *other* features at no extra generation cost.

Result: median null p95 **0.80** vs median `best_act` **27.0**, and **70.9%** of features beat their
own null. Verbalization is genuinely direction-specific for most features. But the null-based curve
is also the flattest (AUC 0.667), so part of what the other definitions attribute to rarity is common
features being easier to hit by chance.

## Caveats that survived checking

- **`norm_act` is unimodal** — KDE mode at 0.41 (sft) / 0.51 (rl), no trough. There is no natural
  cut. The only threshold-free population is `best_act == 0` (rl: 8/512, 1.6%), where the model
  produces nothing that activates the feature at all.
- **The rare end has compressed dynamic range** (median corpus peak 1.16 in the two rarest 27B
  deciles vs 10.79 in the commonest). This is a real but *modest* contribution: conditioning on
  `corpus_peak > 1` moves decile 0 only 0.93 → 0.86, and the scale-adaptive top-N criterion still
  reads 0.87 there. It does not explain the effect away.
- **`realact` cannot be analysed this way at all** — those directions are raw held-out token
  activations with no feature index, hence no rarity axis and no SAE recovery metrics. fig4 plots the
  one quantity they do share.
- **The 8B rarity axis is an independent stream**, not the 27B's pre-tokenized `maemm-data` dump.
  Same quantity, same 256-token windows, same 1.02M budget, different sample. The 8B SAE is also ~7×
  denser (median firing frequency 0.50% vs 0.07%), so the x-axes span different ranges.

## Figures

| file | what | script |
|---|---|---|
| `fig1_recovery_vs_log10_fire_freq` | every recovery metric vs rarity, 27B | `recovery_vs_rarity.py` |
| `fig2_r2_matrix` | R² across three rarity axes, 27B | `recovery_vs_rarity.py` |
| `fig3_unverbalized_vs_log10_fire_freq` | the original single-definition curve | `recovery_vs_rarity.py` |
| `fig4_cos_by_family` | recovery cosine by direction family | `plot_cos_by_family.py` |
| `fig5_unverbalized_by_criterion` | 27B, four definitions | `plot_unverbalized_criteria.py` |
| **`fig6_8b_unverbalized_by_criterion`** | **8B, five definitions incl. the null** | **`plot_8b_rarity.py`** |

fig6 was regenerated when `plot_8b_rarity.py` gained multi-arm support; every number is unchanged
(AUC and unverbalized fraction identical across all five criteria), only the rendering differs. Its
data table is now `data/fig6_8b_unverbalized_by_criterion.json`, matching the script's output name —
the old `data/fig6_8b.json` was a stale leftover of the single-arm script and has been removed.

## Reproducing

Run commands from the repo root. The 8B path needs no private data — base, adapter, SAE and max-acts
are all public HF. See `../README.md` for the full job list and the bad-feature tables.

```bash
modal run verbalization/modal_8b_verbalization.py::probe --n-features 16   # ~5 min, validates the harness
modal run verbalization/modal_8b_verbalization.py::scan_fire               # rarity axis, ~1.02M tokens
modal run verbalization/modal_8b_verbalization.py::eval_dirs --n-features 512
modal volume get maemm-8b-rarity /out/perdir_8b.json verbalization/report/data/
modal volume get maemm-8b-rarity /out/sae_match_8b.npz verbalization/report/data/
python verbalization/analysis/plot_8b_rarity.py \
    --perdir rl=verbalization/report/data/perdir_8b.json \
    --sae-match verbalization/report/data/sae_match_8b.npz --out verbalization/report
```

The 27B figures need `perdir_ckpt_*.json` (`eval_ckpt_daemon.py --dump-per-dir`) and
`/mlp42/sae_match.npz` from the `maemm-data` volume, which was not reachable when this was written.

The per-feature dumps under `data/` are committed, including the generations
(`texts_8b_*.json`) the qualitative tables are built from. `sae_match_8b.npz` is **not** —
`.gitignore` excludes `*.npz` as a data blob — so rebuilding fig6 needs one volume fetch but no GPU.

**Not yet done:** the null control has never been run on the 27B arms. Every 27B number here is
therefore uncontrolled for common features firing on any fluent text. That is the first thing to fix.

---

# Rare-feature training experiment (8B)

Does mining better targets for rare features fix the rarity-vs-verbalizability relation?
**Directionally yes, but it underperforms RL and the headline number depends heavily on the criterion.**

## Design

Three disjoint feature sets over the 65,536-feature L27 SAE, drawn from the same firing-rate bands:
TRAIN 9,972 / TRAIN-EVAL 1,200 (a subset of TRAIN) / TEST 1,520 (never trained on; overlap asserted 0).

1. `mine_targets` — 10M Ultra-FineWeb tokens, top-16 spans per TRAIN feature -> 159,552 pairs.
2. `train_rare` — LoRA SFT from `ref/` (the SFT init), min_act 10, 1 epoch, lr 1e-4, 9,805 steps,
   loss 3.13 -> 2.45.
3. `eval_dirs` before/after on TRAIN-EVAL (fig8) and on TEST.

## The mining result stands on its own

Strong targets DO exist for rare features: every one of the 9,972 got a full top-16, and the rarest
band's rank-0 median activation is **84.3** against a whole-SAE median corpus peak of ~103. The old
bank's problem was scan budget, not corpus scarcity — at 1M tokens a rare feature appears ~7 times so
its argmax is a lucky hit; at 10M it appears ~68 times and the top span is a genuine peak.

## Results — and why the criterion decides the story

Held-out TEST (1,520 features never trained on):

| criterion | sft init | RL | after rare-training |
|---|---|---|---|
| `> own null_p95` (permissive; = "any activation" for the 86% whose null is 0) | 0.270 | **0.514** | 0.458 |
| `> own null_max` (promiscuity-proof) | 0.100 | **0.235** | 0.178 |
| `>= top-16 corpus example` (strict) | 0.030 | **0.074** | 0.046 |

The permissive criterion says +18.8 points of transfer; the promiscuity-proof one says **+7.8**. The
gap is real and measurable: rare-feature training made the model markedly more promiscuous —
median `null_mean` 0.024 -> 0.089, and signal-to-null (median best_act / null_p95) **6.09 -> 3.33**,
the worst of the three arms, against RL's 14.76. A model that lights up more features passes
"beats own null" more often without being any more direction-specific, so **report the strict
criterion**.

It also trades off: per rarity bin under `> null_max`, the gain is +0.21 in the mid-rare bins
(and beats RL outright at 0.011-0.017%: 0.240 vs 0.167) but **-0.026 in the commonest bin**. Training
exclusively on rare features costs a little common-feature capability.

## Conclusion

Better rare-feature targets improve rare-feature verbalizability and the improvement **generalizes to
unseen features** — that part is solid. But RL beats it on every criterion at a fraction of the cost
(~3 GPU-hours of mining + training vs an already-trained checkpoint), and neither approach makes the
deep tail verbalizable: in the rarest bin, strict criterion, sft 0.016 -> RL 0.087 -> after 0.060.

## Figures

| file | what |
|---|---|
| `fig7_hard_to_get` | hard-to-get vs not-measurable, per arm, using each feature's own negative sample |
| `fig8_8b_before_after` | fig6 before/after rare-feature training, on the 1,200 TRAIN-EVAL features |

`modal_8b_verbalization.py` carries all four jobs (`scan_fire`, `eval_dirs`, `mine_targets`,
`train_rare`), plus `score_texts` and `logit_lens` for interrogating individual features.
The trained adapter lives only on the Modal volume at `/data/adapters/rare`; the
159,552-pair mined bank (38 MB) is at `/out/mined_train.jsonl` and is not committed.

---

# What the failures actually are (8B)

The rarity curves say *how often* the inverter fails. They do not say what it is failing at. Walking
the features no arm can activate (`norm_act < 0.10` against the most favourable of sft / RL / after,
966 of 2,017 evaluated features) splits the failures into kinds — and two of them are not the
model's fault.

| flag | n | what it is |
|---|---|---|
| UNRESOLVED | 867 | no structural explanation found; the model simply produces nothing that fires it |
| COLLOCATION | 85 | the token *after* the peak is near-constant — the feature encodes a continuation, not a topic |
| TEMPLATE | 14 | the 30 max-activating "examples" are near-duplicates of one boilerplate string |

COLLOCATION and TEMPLATE are **ill-posed targets**, not misses. A feature that fires on ` of` only
when the next token is ` Commerce` has no paraphrasable meaning; a feature whose entire example set
is one repeated WordPress footer has no concept to generalise. Together they are ~10% of the
unactivatable set — a floor on any "fraction verbalized" number that no amount of training removes.

What the peak token is, across the unactivatable set:

| class | n | | class | n |
|---|---|---|---|---|
| capitalized_word | 212 | | subword_continuation | 122 |
| content_word | 211 | | punctuation | 117 |
| function_word | 193 | | number | 74 |
| whitespace | 37 | | | |

Over half peak on something that is not a content-bearing word — a function word, a mid-word BPE
fragment, punctuation, a digit, or whitespace. This lines up with the fig7 result that the
*not-measurable* cases sit at the COMMON end rather than the rare end.

## Consistency: the model is confidently wrong, not scattershot

If a feature were unverbalizable because the model had no hypothesis about it, its many samples
should be mutually dissimilar. They are not. On 96 rarity-matched held-out features at bo=32
(`analyze_consistency.py`), mean self-consistency is **0.757 for the hard group vs 0.754 for the
easy group** — indistinguishable (Mann-Whitney p = 0.93), and the continuous correlation with
achieved `norm_act` is flat (Spearman +0.107, p = 0.30).

So on a feature it cannot activate, the model is not guessing randomly. It produces one tight,
confident cluster of text that is simply pointing somewhere else. That is a different failure from
"doesn't know", and a worse one for interpretability: the output looks like an explanation.

| table | what |
|---|---|
| `tables/unactivatable_features.csv` | one row per unactivatable feature, with flags |
| `tables/collocation_features.txt` | the COLLOCATION set, with the fixed continuation |
| `tables/unactivatable_examples.csv` | corpus evidence next to each arm's generations |
| `tables/template_features_examples.csv` | the same for the TEMPLATE set |
| `tables/rare_examples.txt` | max-activating windows for the rarest 2% of features |
| `tables/consistency_generations.csv` | per-generation dump behind the consistency numbers |
| `tables/feature_taxonomy.txt`, `tables/writeup_features.txt` | hand-written readings of the above |

---

# The failures are verbalizable — the inverter just can't generalize to them (8B)

Everything above measures the inverter and calls the result "unverbalizable". That word is wrong, and
the check is cheap: read a feature's max-activating examples, write one sentence, score it.

## Humans verbalize these features trivially

`human_verbalization.csv` — 30 hand-written sentences over 21 features the inverter scores 0.00 on:
**median 89% of corpus peak, 29/30 attempts**. Most are novel sentences, not corpus copies; one
exceeded the feature's corpus peak.

`human_beats_maemm.csv` — a harder set: 8 features that **no arm** activates (base encoder,
encoder-trained, decoder-trained, decoder-injected-untrained). 16 sentences, **median 91% of peak,
16/16 fire**. Including two purely positional numeric features — a digit inside a suite number
(`Suite 640`) and inside a UK company registration (`registered number: 3218125`) — at 98% and 96%.

So a textual preimage exists and is easy to find *if you know which feature you are holding*. The
failure is IDENTIFICATION from the injected direction, not text production.

## Training fits what it is shown and transfers nothing

`cluster_transfer.json`, `generalization_failure.json`. Design: cluster the 925 unactivatable
features by their corpus examples, train on 811 (web UI / disclaimers / metadata / prose / news),
test on 113 of entirely unseen types (recipes / sports / contact blocks), zero overlap.

| training | features | pairs | TRAIN norm_act | TEST norm_act |
|---|---|---|---|---|
| none | - | - | 0.004 | 0.008 |
| 16 spans each | 811 | 12,576 | **0.344** | 0.035 |
| 64 spans each | 811 | 47,853 | **0.387** | 0.031 |
| 64 spans, decoder-conditioned | 811 | 47,853 | **0.352** | 0.049 |

Trained features go from producing nothing to a third of corpus peak, with 14% reaching genuine
corpus-example strength. Held-out features stay at **median best_act 0.00** and their null-beating
rate falls BELOW baseline (0.221 -> 0.195). Quadrupling spans per feature improved fitting and
slightly worsened transfer.

## Encoder and decoder are complementary, and neither generalises

The MAEMM injects `unit(W_enc[:,f])`. The decoder row `W_dec[f]` is a different object (enc/dec
cosine 0.49-0.67) and its logit lens reads far more cleanly.

| | encoder | decoder |
|---|---|---|
| on 155 features the encoder handles | norm_act **0.965**, 100% reach ex_last | 0.828, 68% |
| on 113 features the encoder fails | 0.008, **0%** beat null_max | **0.181**, 35% |

The encoder wins where it works; the decoder rescues part of where it doesn't. But decoder injection
plus fine-tuning scores **0.049** on held-out where decoder injection ALONE scores **0.181** —
fine-tuning on 811 features costs the model general direction-reading it already had.

## The claim this supports

> These SAE features are verbalizable; hand-written sentences reach ~90% of corpus peak. This
> inverter cannot generalize to them. Training on 811 of them fits those 811 and transfers nothing to
> unseen feature types, under both encoder and decoder conditioning, and fine-tuning makes held-out
> performance worse than not fine-tuning at all.

Scope: one inverter, warm-started on ~1M probe (topic/cluster) directions. `expressibility.json`
predicts ITS failures from SAE geometry at CV AUC 0.823 — driven almost entirely by corpus rarity
(0.789 alone); the decoder-sharpness signal there is saturated and does not separate named features.
Establishing an intrinsic limit would need the failure to survive across inverters trained to cover
these feature types.
