"""fig5 for the Qwen3-8B inverter: unverbalized fraction vs feature rarity, under several
verbalizability definitions PLUS the per-feature null the 27B dumps never had.

    python evals/verbalization/analysis/plot_8b_rarity.py --perdir perdir_8b.json --sae-match sae_match_8b.npz \
        --out evals/verbalization/report

Criteria (same family as evals/verbalization/analysis/recovery_vs_rarity.py + fig5, adapted to the 8B SAE's scale):
  best_act > 1.0        the inherited eval_universal bar. On this SAE the median corpus peak is ~103,
                        so 1.0 is under 1% of it and the criterion is near-vacuous -- plotted only to
                        show that it does not transfer across SAEs.
  norm_act >= 0.10      best_act / corpus_peak, the house MLP-family convention.
  >= top-N example      best_act at least the N-th ranked corpus example's max -- scale-free, the
                        definition that survived scrutiny on the 27B.
  > null p95            best_act exceeds the 95th percentile of THIS feature's activation on texts
                        generated for OTHER features. The adapter's model card reports a
                        direction-agnostic control matching its headline score, so without this the
                        other curves cannot separate "verbalized f" from "f fires on any fluent text".

The rarity axis is log10(sae_nfire / n_tok) from scan_fire. NOTE it is an independently streamed
scan, not the 27B's pre-tokenized maem-data dump: same quantity and token budget, different sample.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from dumplib import auc_rarer_worse, deciles
from style import SERIES, INK, INK2, MUTED, GRID
import dumplib as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="[TAG=]PATH",
                    help="repeatable: perdir json from modal_8b_verbalization.eval_dirs; TAG= labels the curve")
    ap.add_argument("--sae-match", default=None, help="sae_match_8b.npz from scan_fire (rarity axis)")
    ap.add_argument("--stem", default="", help="output basename; default is the 8B figure name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--criteria", default="bar,norm,top16,last,null",
                    help="comma list of criteria to plot: gate,bar,norm,top16,last,null")
    ap.add_argument("--drop-dead", action="store_true",
                    help="exclude features that never clear the gate in the corpus scan (criterion.py)")
    a = ap.parse_args()

    arms = D.load_perdir(a.perdir)
    if a.drop_dead:
        from criterion import dead_ids
        dead = dead_ids()
        arms = {t: p.select(np.array([int(f) not in dead for f in p.feature]))
                for t, p in arms.items()}
    tags = list(arms)
    s = arms[tags[0]]
    d = s.meta
    best, peak, feat = s["best_act"], s["corpus_peak"], D.shared_features(arms)

    if a.sae_match:
        scan = D.Scan(a.sae_match)
        x, xlabel, axis = scan.log10_freq(feat), scan.label(), "log10_fire_freq"
    else:                                        # fall back to the max-acts corpus peak
        x = np.log10(np.maximum(peak, 1e-9))
        xlabel = "log10 corpus peak activation"
        axis = "log10_corpus_peak"

    want = [w.strip() for w in a.criteria.split(",") if w.strip()]

    def crits_for(sx):
        # a criterion whose inputs the dump does not carry (the 27B dumps have no ex_top16/ex_last)
        # is skipped, not a KeyError
        spec = {
            "gate":  ("some sample clears the SAE gate", lambda: sx["fire_fraction"] > 0, SERIES[0], "-"),
            "bar":   ("best_act > 1.0  (inherited bar)", lambda: sx["best_act"] > 1.0, MUTED, "--"),
            "norm":  ("norm_act >= 0.10", lambda: sx["norm_act"] >= 0.10, SERIES[2], "-"),
            "top16": (">= top-16 corpus example", lambda: sx["best_act"] >= sx["ex_top16"], SERIES[1], "-"),
            "last":  (">= weakest top example", lambda: sx["best_act"] >= sx["ex_last"], SERIES[0], "-"),
        }
        avail = {}
        for k, (lab, fn, col, ls) in spec.items():
            if k in want:
                try:
                    avail[k] = (lab, fn().astype(int), col, ls)
                except KeyError:
                    pass
        if "null_p95" in sx:
            avail["null"] = ("> own null p95  (direction-specific)",
                             (sx["best_act"] > sx["null_p95"]).astype(int), SERIES[5], "-")
        return [avail[w] for w in want if w in avail]
    crits = crits_for(s)

    MARK = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(9.4 if len(tags) > 1 else 8.6, 5.6), constrained_layout=True)
    out = {"n": int(len(feat)), "axis": axis, "arms": {}}
    for ai, tag in enumerate(tags):
        sx = arms[tag]
        out["arms"][tag] = {"adapter": sx.meta.get("adapter"), "criteria": {}}
        for name, v, c, ls in crits_for(sx):
            unv = 1 - v
            xm, ym, se, nb = deciles(x, unv.astype(float))
            A = auc_rarer_worse(x, unv)
            lbl = f"{tag} · {name}" if len(tags) > 1 else name
            ax.errorbar(xm, ym, yerr=se, color=c, ls=(ls if ai == 0 else ":"), lw=1.9,
                        marker=MARK[ai % 4], ms=4.5, mfc="white", mew=1.5, capsize=2,
                        label=f"{lbl}   AUC {A:.3f} · overall {unv.mean():.2f}")
            out["arms"][tag]["criteria"][name] = {
                "auc": float(A), "overall_unverbalized": float(unv.mean()),
                "x_mid": xm.tolist(), "y_mean": ym.tolist(),
                "y_sem": se.tolist(), "n_per_bin": nb.tolist()}
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel("fraction unverbalized", color=INK2)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.03, 1.03)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.2, loc="best")
    # The model NAME comes from the dump, not from this file. It was the literal "Qwen3-8B" while
    # the SAE size and layer beside it were read from the data -- so pointing this script at a 27B
    # dump produced a correct figure captioned with the wrong model (caught 2026-09-23).
    _m = s.meta.get("model") or d.get("model") or "inverter"
    _m = str(_m).split("/")[-1]
    ax.set_title(f"{_m} inverter: where it fails, by feature rarity  (n={len(feat)}, "
                 f"{d.get('d_sae', '?')}-feature SAE @L{d.get('read_layer', '?')})",
                 fontweight="bold", color=INK, fontsize=12, loc="left")
    # `--stem` because these names are hardcoded to the 8B run. Pointing this script at a 27B dump
    # silently OVERWROTE fig6_8b_unverbalized_by_criterion.{png,pdf,json} -- committed artifacts
    # that appendix_verbalization.tex cites (caught and restored 2026-09-23). The defaults are
    # unchanged, so every existing invocation still writes exactly where it did.
    stem = a.stem or ("fig6_8b_unverbalized_by_criterion" if len(tags) == 1
                      else "fig8_8b_before_after")
    D.savefig(fig, a.out, stem)

    for tag in tags:
        sx = arms[tag]
        if "null_p95" in sx:
            out["arms"][tag]["null"] = {
                "median_null_p95": float(np.median(sx["null_p95"])),
                "median_best_act": float(np.median(sx["best_act"])),
                "frac_best_above_null_p95": float(np.mean(sx["best_act"] > sx["null_p95"])),
                "frac_zero_act": float(np.mean(sx["best_act"] == 0))}
    D.write_data(a.out, stem, out)

    w = max(len(k) for t in tags for k in out["arms"][t]["criteria"])
    print(f"n={len(feat)}  axis={axis}")
    for t in tags:
        print(f"\n[{t}]  {out['arms'][t]['adapter']}")
        print(f"  {'criterion':<{w}} {'unverb':>8} {'AUC':>8}")
        for k, v in out["arms"][t]["criteria"].items():
            print(f"  {k:<{w}} {v['overall_unverbalized']:>8.3f} {v['auc']:>8.3f}")
    for t in tags:
        n_ = out["arms"][t].get("null")
        if n_:
            print("  null: median p95 %.2f | median best_act %.2f | beats own null %.1f%% | zero-act %.1f%%"
                  % (n_["median_null_p95"], n_["median_best_act"],
                     100 * n_["frac_best_above_null_p95"], 100 * n_["frac_zero_act"]))


if __name__ == "__main__":
    main()
