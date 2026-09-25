"""Unverbalized fraction vs feature rarity under SEVERAL verbalizability definitions (27B arms).

The companion to evals/verbalization/analysis/recovery_vs_rarity.py, which reports a single "unverbalized" rate using
eval_universal's inherited `SAE_FIRE = 1.0`. That constant has no derivation in the codebase (it is
the old `eval_dirs --sae-fire` default) and it is not a scale: across the 512 held-out features the
1.0 bar ranges from 1.8% to 561% of the feature's own corpus peak, a 315x spread, and for 9.2% of
features the corpus itself never reaches it. This script therefore plots the rarity relation under
four definitions at once, so the reader can see which conclusions depend on the cut:

  best_act > 1.0        the inherited absolute bar
  norm_act >= 0.10      best_act / corpus_peak; the convention the MLP-neuron family already uses
                        (data/mlp42_neurons_worker.py `fired10_bo`). Over-corrects at the rare end,
                        where corpus_peak is small and the ratio becomes noise-amplified.
  >= top-N example      best_act at least the N-th ranked corpus example's max activation, from the
                        SAE max-acts tensor: "the generated text would place in this feature's top-N
                        corpus examples". Per-feature, scale-free, and the one that transfers across
                        SAEs with different activation scales.

The headline finding is that the rarity relation survives all of them (AUC 0.71-0.91) while the base
rate moves 20x -- so rarity is robust, but any single "unverbalized fraction" is a reporting choice.

    python evals/verbalization/analysis/plot_unverbalized_criteria.py \
        --perdir sft=perdir_sft.json --perdir rl=perdir_rl.json \
        --sae-match sae_match.npz --maxacts max_acts.pt --out evals/verbalization/report
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from dumplib import auc_rarer_worse, deciles
from style import SERIES, INK, INK2, MUTED, GRID
import dumplib as D
SAE_FIRE = 1.0            # eval_universal.SAE_FIRE, reproduced for the comparison curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="TAG=PATH",
                    help="repeatable: arm label and its eval_ckpt_daemon --dump-per-dir json")
    ap.add_argument("--sae-match", required=True, help="/data/mlp42/sae_match.npz (sae_nfire)")
    ap.add_argument("--maxacts", required=True, help="SAE max_acts .pt (per-feature top-N examples)")
    ap.add_argument("--n-tok", type=int, default=1_024_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # --n-tok is passed explicitly rather than read off the npz: the 27B sae_match.npz predates
    # the n_tok key, and this script's callers have always supplied the scan size themselves
    scan = D.Scan(a.sae_match, n_tok=a.n_tok)
    ma = torch.load(a.maxacts, map_location="cpu", weights_only=False)["max_acts"].float()
    ex = -np.sort(-ma.max(dim=2).values.numpy(), axis=1)          # [F, n_ex] example maxima, desc

    arms = {}
    for tag, p in D.load_perdir(a.perdir).items():
        s = dict(p.col)
        s["x"] = scan.log10_freq(p.feature)
        arms[tag] = s

    def criteria(s):
        F = s["feature"]
        return [(f"best_act > {SAE_FIRE}  (old)", (s["best_act"] > SAE_FIRE).astype(int), MUTED, "--"),
                ("norm_act >= 0.10", (s["norm_act"] >= 0.10).astype(int), SERIES[2], "-"),
                (f">= top-{ex.shape[1]} corpus example", (s["best_act"] >= ex[F, -1]).astype(int), SERIES[0], "-"),
                (">= top-16 corpus example", (s["best_act"] >= ex[F, 15]).astype(int), SERIES[1], "-")]

    fig, axes = plt.subplots(1, len(arms), figsize=(6.2 * len(arms), 4.9), sharey=True, squeeze=False)
    out = {}
    for ax, (tag, s) in zip(axes[0], arms.items()):
        out[tag] = {}
        for name, v, c, ls in criteria(s):
            unv = (1 - v).astype(float)
            xm, ym, se, _ = deciles(s["x"], unv)
            A = auc_rarer_worse(s["x"], 1 - v)
            ax.errorbar(xm, ym, yerr=se, color=c, ls=ls, lw=1.9, marker="o", ms=4.5, mfc="white",
                        mew=1.5, capsize=2, label=f"{name}   AUC {A:.3f} · overall {unv.mean():.2f}")
            out[tag][name] = {"auc": float(A), "overall_unverbalized": float(unv.mean()),
                              "x_mid": xm.tolist(), "y_mean": ym.tolist(), "y_sem": se.tolist()}
        ax.set_title(tag, fontweight="bold", color=INK)
        ax.set_xlabel(f"log10 firing frequency (act > 0, {a.n_tok/1e6:.2f}M tokens)", color=INK2)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.03, 1.03)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8.2, loc="lower left")
    axes[0][0].set_ylabel("fraction unverbalized", color=INK2)
    fig.suptitle("Where the inverter fails, by feature rarity — under four verbalizability definitions",
                 fontweight="bold", color=INK, x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    D.savefig(fig, a.out, "fig5_unverbalized_by_criterion")
    D.write_data(a.out, "unverbalized_by_criterion", out)
    for tag in out:
        print(tag, {k: round(v["auc"], 3) for k, v in out[tag].items()})


if __name__ == "__main__":
    main()
