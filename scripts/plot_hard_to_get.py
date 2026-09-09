"""Separate HARD TO GET from NOT MEASURABLE, using each feature's own negative sample.

The unverbalized-vs-rarity curves (fig5/fig6) conflate two failure modes:

  hard to get     the feature is discriminative -- its corpus examples activate it far above what
                  unrelated fluent text does -- and the inverter still cannot reach that level.
                  A real capability failure.

  not measurable  the feature's NULL is high: it fires on text generated for other features nearly
                  as much as on its own corpus examples. "Verbalizing" it carries little information
                  either way, so neither success nor failure means much.

Both quantities come from the same eval pass (scripts/modal_8b_rarity.py::eval_dirs), which scores
every generated text against ALL features and so yields, per feature f:

  null_p95(f)   f's activation on texts generated for OTHER features   <- the negative sample
  ex_last(f)    f's weakest top-N corpus example (from the SAE max-acts tensor)
  best_act(f)   the inverter's best-of-bo

  discriminability = ex_last / null_p95     how far real evidence sits above the negative sample
  attainment       = best_act / null_p95    how far the inverter got above it

A feature is MEASURABLE when discriminability >= --disc (default 10x) and GOT when attainment >= 1.
On a uniform 512-feature 8B draw the not-measurable population sits in the COMMON features (55% of
the top firing-frequency quintile) and is absent from the rare end (0%) -- the opposite of the naive
expectation that rare features are the poorly-sampled ones. Rare features have a clean (~zero) null,
so the measurement is well-posed there and the model genuinely fails.

    python scripts/plot_hard_to_get.py --perdir perdir_8b_rare.json \
        --sae-match sae_match_8b.npz --out reports/maemm-recovery-vs-rarity
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def bins(x, nbins):
    q = np.quantile(x, np.linspace(0, 1, nbins + 1))
    return np.clip(np.digitize(x, q[1:-1]), 0, nbins - 1), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc", type=float, default=10.0, help="discriminability threshold (x null)")
    ap.add_argument("--nbins", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/data", exist_ok=True)

    arms = {}
    for spec in a.perdir:
        tag, path = spec.split("=", 1)
        dd = json.load(open(path))
        arms[tag] = {k: np.asarray(v) for k, v in dd["perdir"]["sae"].items()}
        arms[tag]["_adapter"] = dd.get("adapter")
    z = np.load(a.sae_match)
    n_tok = int(z["n_tok"]) if "n_tok" in z else 1_024_000
    tags = list(arms)
    feat = arms[tags[0]]["feature"]
    for t in tags[1:]:
        assert np.array_equal(arms[t]["feature"], feat), "arms must share a feature list"
    x = np.log10(np.maximum(z["sae_nfire"].astype(float)[feat], 1) / n_tok)
    b, q = bins(x, a.nbins)
    xm = np.array([x[b == i].mean() for i in range(a.nbins)])
    nb = np.array([(b == i).sum() for i in range(a.nbins)])

    fig, axes = plt.subplots(1, len(tags) + 1, figsize=(5.6 * (len(tags) + 1), 4.9), squeeze=False)
    out = {"n": int(len(x)), "disc_threshold": a.disc, "n_tok": n_tok, "arms": {}}
    for ax, tag in zip(axes[0], tags):
        s = arms[tag]
        null = np.maximum(s["null_p95"], 1e-9)
        disc, att = s["ex_last"] / null, s["best_act"] / null
        measurable, got = disc >= a.disc, s["best_act"] > s["null_p95"]
        fo = np.array([(measurable[b == i] & got[b == i]).mean() for i in range(a.nbins)])
        fh = np.array([(measurable[b == i] & ~got[b == i]).mean() for i in range(a.nbins)])
        fn = np.array([(~measurable[b == i]).mean() for i in range(a.nbins)])
        ax.stackplot(xm, fo, fh, fn, colors=[SERIES[2], SERIES[1], MUTED], alpha=0.85,
                     labels=["verbalized (beats own null)", "HARD TO GET", "not measurable"])
        ax.set_ylim(0, 1); ax.set_xlim(xm.min(), xm.max())
        ax.set_title(f"{tag}   ({100*(s['best_act']==0).mean():.0f}% produce zero activation)",
                     fontweight="bold", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8.2, loc="lower center")
        out["arms"][tag] = {"adapter": s.get("_adapter"),
                            "frac_zero_act": float((s["best_act"] == 0).mean()),
                            "beats_null": float(got.mean()),
                            "hard_to_get": float((measurable & ~got).mean()),
                            "not_measurable": float((~measurable).mean()),
                            "frac_verbalized_bin": fo.tolist(), "frac_hard_bin": fh.tolist(),
                            "frac_notmeas_bin": fn.tolist()}
    axes[0][0].set_ylabel("fraction of features", color=INK2)

    axd = axes[0][-1]
    for j, tag in enumerate(tags):
        s = arms[tag]
        g = np.array([(s["best_act"][b == i] > s["null_p95"][b == i]).mean() for i in range(a.nbins)])
        axd.plot(xm, g, color=SERIES[j], lw=2.0, marker="o", ms=4.5, mfc="white", mew=1.5, label=tag)
    axd.set_ylim(0, 1); axd.set_ylabel("fraction beating own null", color=INK2)
    axd.set_title("paired: same features, both arms", fontweight="bold", color=INK, fontsize=11)
    axd.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes[0]:
        ax.set_xlabel(f"log10 firing frequency ({n_tok/1e6:.2f}M tokens)", color=INK2)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    out["by_rarity_bin"] = {"x_mid": xm.tolist(), "n": nb.tolist(),
                            "bin_edges_pct": (100 * 10 ** q).tolist()}
    fig.suptitle("Hard to get vs not measurable — Qwen3-8B, rare-weighted features, paired arms",
                 fontweight="bold", color=INK, x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for e in ("png", "pdf"):
        fig.savefig(f"{a.out}/fig7_hard_to_get.{e}", dpi=170, bbox_inches="tight")
    json.dump(out, open(f"{a.out}/data/fig7_hard_to_get.json", "w"), indent=1)
    for tag, v in out["arms"].items():
        print("%-4s zero-act %.3f | beats null %.3f | hard-to-get %.3f | not-meas %.3f"
              % (tag, v["frac_zero_act"], v["beats_null"], v["hard_to_get"], v["not_measurable"]))


if __name__ == "__main__":
    main()
