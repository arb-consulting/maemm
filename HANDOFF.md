# Handoff — verbalizability of SAE features

Reproducible from a clean machine. Needs no private data: base model, adapter, SAE and max-acts
are all public on HF.

## 1. Repo state

    repo      https://github.com/ceselder/maemm   (public read; Ari lacks write)
    branch    verbalization-eval                  (8 commits, NEVER pushed — 403 on push)
    base      master @ e7dfaae at branch time; remote master has moved to 74d9882
                                                  -> rebase before any push attempt

Uncommitted, all new:

    verbalization/analysis/plot_generalization_failure.py
    verbalization/report/appendix_verbalization.tex
    verbalization/report/fig9_generalization_failure.{pdf,png}
    verbalization/report/tables/test50_for_human.csv

## 2. Deliverable

    verbalization/report/appendix_verbalization.tex

Self-contained LaTeX, no custom macros. Packages: booktabs, multirow, xcolor, graphicx.
Every `\textcolor{red}{[TODO]}` is prose still to be written. Expects its figures in the same
directory, else set `\graphicspath{{verbalization/report/}}`.

Two figures referenced:

    fig6_8b_unverbalized_by_criterion.pdf    unverbalized fraction vs rarity, five criteria
    fig9_generalization_failure.pdf          human vs inverter, and train/test transfer

fig1-5, fig7, fig8 still exist on disk but are no longer referenced.

## 3. The measurement

Inject `v = unit(W_enc[:,f])` at layer 1 on the ` ?` marker (`h <- h + ||h||*v`), generate
best-of-`b`, re-read each generated text through the CLEAN BASE with the adapter disabled, take
`best_act` = max over samples and content tokens of `f`'s raw pre-topk SAE activation. The adapter
being off at read time is load-bearing: the model cannot satisfy the metric by encoding something
only its own LoRA recognises.

Every generated text is scored against all 65,536 features, not just its paired one, so each
feature's activation on OTHER features' text — its null — comes free from the same pass.

## 4. Re-running

    modal run verbalization/modal_8b_verbalization.py::probe --n-features 16      # ~5 min, smoke
    modal run verbalization/modal_8b_verbalization.py::scan_fire                  # rarity axis
    modal run verbalization/modal_8b_verbalization.py::eval_dirs --n-features 512
    modal volume get maemm-8b-rarity /out/perdir_8b.json verbalization/report/data/
    modal volume get maemm-8b-rarity /out/sae_match_8b.npz verbalization/report/data/

    python verbalization/analysis/plot_8b_rarity.py \
        --perdir rl=verbalization/report/data/perdir_8b.json \
        --sae-match verbalization/report/data/sae_match_8b.npz --out verbalization/report

fig9 needs no GPU and no volume fetch — it reads only committed JSON/CSV:

    python verbalization/analysis/plot_generalization_failure.py

## 5. Known-open items

- 27B arms have never had a null control run. Every 27B number is uncontrolled for common
  features firing on any fluent text. This is the first thing to fix.
- Figures are vector PDFs but not print-normalised: aspect ratios run 1.4:1 to 4.6:1, so
  `width=\linewidth` yields different effective font sizes per figure. Only fig9 is sized for a
  text-width column. `style.apply_rcparams()` is opt-in and only `recovery_vs_rarity.py` calls it,
  so fig1-3 and fig4-8 do not share a house style.
- `test50_for_human.csv` is a 50-feature worksheet with a blank `your_sentence` column, unscored.
