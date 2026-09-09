"""fig5 for the Qwen3-8B inverter: unverbalized fraction vs feature rarity, under several
verbalizability definitions PLUS the per-feature null the 27B dumps never had.

    python scripts/plot_8b_rarity.py --perdir perdir_8b.json --sae-match sae_match_8b.npz \
        --out reports/maemm-recovery-vs-rarity

Criteria (same family as scripts/recovery_vs_rarity.py + fig5, adapted to the 8B SAE's scale):
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
scan, not the 27B's pre-tokenized maemm-data dump: same quantity and token budget, different sample.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NBINS = 10


def auc_rarer_worse(x, unv):
    """P(a rarer feature is the unverbalized one). x = rarity axis, unv = 1/0."""
    r = stats.rankdata(-x)
    n1, n0 = unv.sum(), (1 - unv).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[unv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def deciles(x, y, nbins=NBINS):
    q = np.quantile(x, np.linspace(0, 1, nbins + 1))
    b = np.clip(np.digitize(x, q[1:-1]), 0, nbins - 1)
    xm = np.array([x[b == i].mean() for i in range(nbins)])
    ym = np.array([y[b == i].mean() for i in range(nbins)])
    se = np.array([y[b == i].std(ddof=1) / np.sqrt(max((b == i).sum(), 1)) for i in range(nbins)])
    n = np.array([(b == i).sum() for i in range(nbins)])
    return xm, ym, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="[TAG=]PATH",
                    help="repeatable: perdir json from modal_8b_rarity.eval_dirs; TAG= labels the curve")
    ap.add_argument("--sae-match", default=None, help="sae_match_8b.npz from scan_fire (rarity axis)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/data", exist_ok=True)

    arms = {}
    for spec in a.perdir:
        tag, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        dd = json.load(open(path))
        arms[tag] = {k: np.asarray(v) for k, v in dd["perdir"]["sae"].items()}
        arms[tag]["_meta"] = dd
    tags = list(arms)
    d = arms[tags[0]]["_meta"]
    s = arms[tags[0]]
    best, peak, feat = s["best_act"], s["corpus_peak"], s["feature"]
    for t in tags[1:]:
        assert np.array_equal(arms[t]["feature"], feat), "all arms must share a feature list"

    if a.sae_match:
        z = np.load(a.sae_match)
        n_tok = int(z["n_tok"]) if "n_tok" in z else 1_024_000
        x = np.log10(np.maximum(z["sae_nfire"].astype(float)[feat], 1) / n_tok)
        xlabel = f"log10 firing frequency (act > 0, {n_tok/1e6:.2f}M tokens)"
        axis = "log10_fire_freq"
    else:                                        # fall back to the max-acts corpus peak
        x = np.log10(np.maximum(peak, 1e-9))
        xlabel = "log10 corpus peak activation"
        axis = "log10_corpus_peak"

    def crits_for(sx):
        c = [("best_act > 1.0  (inherited bar)", (sx["best_act"] > 1.0).astype(int), MUTED, "--"),
             ("norm_act >= 0.10", (sx["norm_act"] >= 0.10).astype(int), SERIES[2], "-"),
             (">= top-16 corpus example", (sx["best_act"] >= sx["ex_top16"]).astype(int), SERIES[1], "-"),
             (">= weakest top example", (sx["best_act"] >= sx["ex_last"]).astype(int), SERIES[0], "-")]
        if "null_p95" in sx:
            c.append(("> own null p95  (direction-specific)",
                      (sx["best_act"] > sx["null_p95"]).astype(int), SERIES[5], "-"))
        return c
    crits = crits_for(s)

    MARK = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(9.4 if len(tags) > 1 else 8.6, 5.6), constrained_layout=True)
    out = {"n": int(len(feat)), "axis": axis, "arms": {}}
    for ai, tag in enumerate(tags):
        sx = arms[tag]
        out["arms"][tag] = {"adapter": sx["_meta"].get("adapter"), "criteria": {}}
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
    ax.set_title(f"Qwen3-8B inverter: where it fails, by feature rarity  (n={len(feat)}, "
                 f"{d.get('d_sae', '?')}-feature SAE @L{d.get('read_layer', '?')})",
                 fontweight="bold", color=INK, fontsize=12, loc="left")
    stem = "fig6_8b_unverbalized_by_criterion" if len(tags) == 1 else "fig8_8b_before_after"
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}/{stem}.{e}", dpi=170, bbox_inches="tight")

    for tag in tags:
        sx = arms[tag]
        if "null_p95" in sx:
            out["arms"][tag]["null"] = {
                "median_null_p95": float(np.median(sx["null_p95"])),
                "median_best_act": float(np.median(sx["best_act"])),
                "frac_best_above_null_p95": float(np.mean(sx["best_act"] > sx["null_p95"])),
                "frac_zero_act": float(np.mean(sx["best_act"] == 0))}
    json.dump(out, open(f"{a.out}/data/{stem}.json", "w"), indent=1)

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
