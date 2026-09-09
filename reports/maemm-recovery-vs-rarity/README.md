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
gameable by generic fluent text. `modal_8b_rarity.eval_dirs` therefore scores every generated text
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

## Reproducing

The 8B path needs no private data — base, adapter, SAE and max-acts are all public HF:

```bash
modal run scripts/modal_8b_rarity.py::probe --n-features 16     # ~5 min, validates the harness
modal run scripts/modal_8b_rarity.py::scan_fire                 # rarity axis, ~1.02M tokens
modal run scripts/modal_8b_rarity.py::eval_dirs --n-features 512
modal volume get maemm-8b-rarity /out/perdir_8b.json data/
modal volume get maemm-8b-rarity /out/sae_match_8b.npz data/
python scripts/plot_8b_rarity.py --perdir data/perdir_8b.json \
    --sae-match data/sae_match_8b.npz --out reports/maemm-recovery-vs-rarity
```

The 27B figures need `perdir_ckpt_*.json` (`eval_ckpt_daemon.py --dump-per-dir`) and
`/mlp42/sae_match.npz` from the `maemm-data` volume, which was not reachable when this was written.

`data/perdir_8b.json` (the 8B per-feature results, incl. the null) is committed. `sae_match_8b.npz`
is **not** — `.gitignore` excludes `*.npz` as a data blob — so rebuilding fig6 needs one volume
fetch (`modal volume get maemm-8b-rarity /out/sae_match_8b.npz data/`) but no GPU. Note the report's
`data/` subdirectory is itself covered by the repo-wide `data/` ignore rule, so its JSON tables were
added explicitly with `git add -f`; **new files written there will not appear in `git status`.**

**Not yet done:** the null control has never been run on the 27B arms. Every 27B number here is
therefore uncontrolled for common features firing on any fluent text. That is the first thing to fix.
