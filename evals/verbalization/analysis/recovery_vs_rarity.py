"""Is the inverter's per-feature recovery explained by how RARE the feature is in the corpus?

The "unverbalizable" question, operationally: the upper bound on what the inverter cannot say is the
set of directions it does not RECOVER — inject v, generate, re-read the clean base, and score the
returned activation against v (the eval suite's max-over-token cosine / SAE activation; the random
control family is the no-signal floor). This script takes those per-direction recovery scores and asks
the first structural question about the failures: how much of the spread is a function of feature
RARITY?

Inputs (nothing is recomputed — both already exist in the pipeline):
  --perdir tag=path   evals/heldout/eval_ckpt_daemon.py --dump-per-dir  -> perdir_ckpt_<k>.json
                      per held-out SAE feature: best-of-bo best_act, corpus_peak, norm_act = best/peak,
                      cos (max-token cosine to the unit encoder column), full-SAE rank, fired
                      (+ the cos families, of which `random` is the floor)
  --sae-match path    data/mlp42_neurons_worker.py -> /data/mlp42/sae_match.npz
                      sae_nfire[F] = tokens with pre-topk act > 0 over the 1.02M-token FineFineWeb scan,
                      sae_mean[F] / sae_std[F] = that scan's per-feature activation moments
  --maxacts path      maem.sae.load_max_acts -> max_acts[F,N,L], the corpus max-activating examples

Rarity axes (each is a *proxy*; they are reported side by side because they disagree, and the axis you
believe changes the answer):
  log10_fire_freq   log10(sae_nfire / n_tok)          how often the feature is active at all
  log10_mean_act    log10(sae_mean)                   density-weighted activity (mass, not just count)
  log10_corpus_peak log10(corpus_peak)                how hard the corpus ever drives it
  log10_topn_mean   log10(mean of the top-N example maxima)     )  from --maxacts: how well-supported
  topn_decay        (N-th top example max) / (1st)              )  the feature's evidence is
  topn_fired_frac   fraction of the N top examples over SAE_FIRE)

Reported per (arm x recovery metric x rarity axis): linear R^2 (+ adjusted, + bootstrap 95% CI, +
the R^2 expected under the null, 1/(n-1) -- with n=512 a "small" R^2 is not automatically noise, and
an R^2 of 0.01 is), Spearman rho (monotone, robust to the log choice), and a DECILE-BIN R^2 -- the
variance explained by a 10-step function of rarity, which upper-bounds any monotone predictor at that
resolution and so separates "rarity does not matter" from "rarity matters non-linearly".
Because norm_act divides by corpus_peak and corpus_peak is itself a rarity axis, every recovery metric
is also fit on the RAW best_act, and a two-predictor fit reports the semipartial R^2 of frequency with
corpus_peak held out.

Usage:
    python evals/verbalization/analysis/recovery_vs_rarity.py \
        --perdir sft=perdir_ckpt_10107.json --perdir rl=perdir_ckpt_150.json \
        --perdir fullft=perdir_ckpt_2441.json \
        --sae-match sae_match.npz --maxacts max_acts.pt --out evals/verbalization/report
(the two volume files: modal volume get maem-data /mlp42/sae_match.npz . ;
 modal volume get maem-data /eval_ckpt/<tag>/perdir_ckpt_<k>.json .)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# categorical slots 1-6 of the validated reference palette, fixed order by arm; chrome from the
# same sheet.
from style import SERIES, INK, INK2, MUTED, GRID, apply_rcparams
import dumplib
SAE_FIRE = 1.0            # eval_universal.SAE_FIRE: raw act > 1.0 counts as "fired"
N_TOK_DEFAULT = dumplib.N_TOK_DEFAULT   # data/mlp42_neurons_worker.py scan: 4000 windows x 256 tokens
NBINS = 10
N_BOOT = 2000

apply_rcparams()

RECOVERY_LABEL = {"norm_act": "norm_act  (best act / corpus peak)", "sae_cos": "cos(rollout peak, encoder dir)",
                  "log10_best_act": "log10 best raw activation", "neg_log10_rank": "-log10 full-SAE rank at peak",
                  "fired": "fired (act > 1)"}
RARITY_LABEL = {"log10_fire_freq": "log10 firing frequency (act > 0, 1.02M tokens)",
                "log10_mean_act": "log10 mean activation over the scan",
                "log10_corpus_peak": "log10 corpus peak activation",
                "log10_topn_mean": "log10 mean top-N example activation",
                "topn_decay": "top-N decay (Nth / 1st example max)",
                "topn_fired_frac": "fraction of top-N examples that fire"}


def savefig(fig, out, stem):
    # pdf_dpi=False: fig2 embeds its R^2 matrix as a rasterized imshow, and this script has always
    # written PDFs without a dpi, so that image sits at the matplotlib default. See dumplib.savefig.
    dumplib.savefig(fig, out, stem, pdf_dpi=False)


def style(ax):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_perdir(path):
    """One --dump-per-dir json -> the per-feature recovery metrics (higher = better recovered).

    The columns come from dumplib.PerDir; what this adds is the DERIVED recovery metrics, which
    are specific to this script's regression tables.
    """
    p = dumplib.PerDir.load(path)
    d = p.meta
    best = p["best_act"].astype(np.float64)
    rec = {"norm_act": p["norm_act"].astype(np.float64),
           "log10_best_act": np.log10(np.clip(best, 1e-3, None)),
           "fired": (best > SAE_FIRE).astype(np.float64)}
    if "cos" in p:
        rec["sae_cos"] = p["cos"].astype(np.float64)
    if "rank" in p:
        rec["neg_log10_rank"] = -np.log10(p["rank"].astype(np.float64))
    return {"path": p.path, "ckpt_step": d.get("ckpt_step"), "tag": d.get("tag"),
            "feature": p["feature"].astype(np.int64), "best_act": best,
            "corpus_peak": p["corpus_peak"].astype(np.float64), "recovery": rec,
            "random_cos": p.cos("random"), "aggregates": d.get("aggregates", {})}


def load_rarity(feats, sae_match=None, n_tok=N_TOK_DEFAULT, maxacts=None):
    """Rarity axes for the given feature ids. corpus_peak is filled in by the caller (it is in the dump)."""
    ax, src = {}, {}
    if sae_match:
        z = np.load(sae_match)
        nfire = z["sae_nfire"].astype(np.float64)
        assert feats.max() < len(nfire), f"feature id {feats.max()} >= d_sae {len(nfire)} in {sae_match}"
        ax["log10_fire_freq"] = np.log10(np.clip(nfire[feats] / n_tok, 0.5 / n_tok, None))
        ax["log10_mean_act"] = np.log10(np.clip(z["sae_mean"].astype(np.float64)[feats], 1e-8, None))
        src["log10_fire_freq"] = f"{sae_match}:sae_nfire / {n_tok} tokens (act > 0)"
        src["log10_mean_act"] = f"{sae_match}:sae_mean"
    if maxacts:
        import torch
        ma = torch.load(maxacts, map_location="cpu", weights_only=False)["max_acts"]
        ex = ma.reshape(ma.shape[0], ma.shape[1], -1).max(-1).values.float().numpy()[feats]   # [n, N] per-example max
        ex = -np.sort(-ex, axis=1)                                                            # descending, defensively
        ax["log10_topn_mean"] = np.log10(np.clip(ex.mean(1), 1e-6, None))
        ax["topn_decay"] = ex[:, -1] / np.clip(ex[:, 0], 1e-6, None)
        ax["topn_fired_frac"] = (ex > SAE_FIRE).mean(1)
        for k in ("log10_topn_mean", "topn_decay", "topn_fired_frac"):
            src[k] = f"{maxacts}:max_acts[F,N={ex.shape[1]},L] per-example max"
    return ax, src


def fit(x, y, seed=0):
    """Linear fit + monotone + decile-step summary of y ~ x, with the null R^2 and a bootstrap CI."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 * NBINS or x.std() == 0 or y.std() == 0:
        return {"n": int(n), "r2": None, "note": "degenerate or too few points"}
    lr = stats.linregress(x, y)
    rho = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    xb, yb = x[idx], y[idx]
    xb = xb - xb.mean(1, keepdims=True); yb = yb - yb.mean(1, keepdims=True)
    rb = (xb * yb).sum(1) / np.sqrt(np.clip((xb ** 2).sum(1) * (yb ** 2).sum(1), 1e-30, None))
    lo, hi = np.quantile(rb ** 2, [0.025, 0.975])
    r2 = float(lr.rvalue ** 2)
    b = np.clip(np.digitize(x, np.quantile(x, np.linspace(0, 1, NBINS + 1))[1:-1]), 0, NBINS - 1)
    pred = np.array([y[b == k].mean() if (b == k).any() else y.mean() for k in range(NBINS)])[b]
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = float(((y - pred) ** 2).sum())
    r2_bin = 1.0 - ssr / sst
    return {"n": int(n), "r2": r2, "r2_adj": float(1 - (1 - r2) * (n - 1) / (n - 2)),
            "r2_boot_ci95": [float(lo), float(hi)], "r2_null_expect": float(1.0 / (n - 1)),
            "pearson_r": float(lr.rvalue), "p_value": float(lr.pvalue),
            "slope": float(lr.slope), "slope_stderr": float(lr.stderr), "intercept": float(lr.intercept),
            "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue), "spearman_rho2": float(rho.statistic ** 2),
            "r2_decile_bins": float(r2_bin),
            "r2_decile_bins_adj": float(1 - (ssr / (n - NBINS)) / (sst / (n - 1))),
            "r2_nonlinear_excess": float(r2_bin - r2)}


def deciles(x, y):
    """Decile-of-rarity means with SEM — the shape behind the R^2."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.quantile(x, np.linspace(0, 1, NBINS + 1))
    b = np.clip(np.digitize(x, edges[1:-1]), 0, NBINS - 1)
    out = {"edges": edges.tolist(), "x_mid": [], "y_mean": [], "y_sem": [], "n": []}
    for k in range(NBINS):
        m = b == k
        out["x_mid"].append(float(np.median(x[m])) if m.any() else float("nan"))
        out["y_mean"].append(float(y[m].mean()) if m.any() else float("nan"))
        out["y_sem"].append(float(y[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else float("nan"))
        out["n"].append(int(m.sum()))
    return out


def multi_r2(y, cols):
    """R^2 of the least-squares fit of y on [1, *cols]."""
    y = np.asarray(y, np.float64)
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, np.float64) for c in cols])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()), coef.tolist()


def auc(score, label):
    """AUC of `score` as a predictor of label==1 (Mann-Whitney); 0.5 = no signal."""
    label = np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = stats.rankdata(np.asarray(score, np.float64))
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fig_scatter(D, order, rar, axis, out, stem):
    """rows = recovery metric, cols = arm: the per-feature cloud, decile means (SEM), OLS line, R^2."""
    mets = [m for m in RECOVERY_LABEL if all(m in D[k]["recovery"] for k in order)]
    fig, axes = plt.subplots(len(mets), len(order), figsize=(3.6 * len(order), 2.9 * len(mets)),
                             sharex=True, squeeze=False)
    x = rar[axis]
    for i, met in enumerate(mets):
        row = [D[k]["recovery"][met] for k in order]
        ylo, yhi = min(np.nanmin(y) for y in row), max(np.nanmax(y) for y in row)
        pad = 0.06 * (yhi - ylo + 1e-9)
        for j, k in enumerate(order):
            ax, y, c = axes[i][j], D[k]["recovery"][met], SERIES[j % len(SERIES)]
            ax.scatter(x, y, s=7, color=c, alpha=0.28, lw=0)
            f, b = D[k]["fits"][met][axis], D[k]["deciles"][met][axis]
            if f.get("r2") is not None:
                xs = np.array([np.nanmin(x), np.nanmax(x)])
                ax.plot(xs, f["intercept"] + f["slope"] * xs, color=INK, lw=1.6, zorder=3)
                ax.text(0.03, 0.95, f"$R^2$ {f['r2']:.3f}   ρ {f['spearman_rho']:+.2f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=9, color=INK)
            ax.errorbar(b["x_mid"], b["y_mean"], yerr=b["y_sem"], fmt="o", ms=5, lw=0, elinewidth=1.4,
                        color=INK, mfc="white", mec=INK, mew=1.4, capsize=2, zorder=4)
            if met == "sae_cos" and D[k]["random_cos"] is not None:
                ax.axhline(float(np.mean(D[k]["random_cos"])), color=MUTED, lw=1.1, ls="--", zorder=2)
                if j == 0:
                    ax.text(0.03, 0.06, "random control", transform=ax.transAxes, fontsize=8, color=MUTED)
            ax.set_ylim(ylo - pad, yhi + pad)
            style(ax)
            if i == 0:
                ax.set_title(k, fontsize=10.5, color=INK)
            if j == 0:
                ax.set_ylabel(RECOVERY_LABEL[met], fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel(RARITY_LABEL.get(axis, axis), fontsize=8.5)
    fig.suptitle(f"Per-feature recovery vs {RARITY_LABEL.get(axis, axis)}", fontsize=11.5, fontweight="bold",
                 x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig(fig, out, stem)
    return mets


def fig_r2_matrix(D, order, axes_names, out, stem):
    """R^2 of every (recovery metric x rarity axis) pair, one panel per arm. Sequential single hue."""
    mets = [m for m in RECOVERY_LABEL if all(m in D[k]["recovery"] for k in order)]
    fig, axs = plt.subplots(1, len(order), figsize=(2.2 + 2.0 * len(axes_names) * len(order) / 2, 0.55 * len(mets) + 2.6),
                            squeeze=False)
    vmax = max((D[k]["fits"][m][a].get("r2") or 0.0) for k in order for m in mets for a in axes_names) or 1.0
    for j, k in enumerate(order):
        ax = axs[0][j]
        M = np.array([[D[k]["fits"][m][a].get("r2") or np.nan for a in axes_names] for m in mets])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                if np.isfinite(M[r, c]):
                    ax.text(c, r, f"{M[r, c]:.3f}", ha="center", va="center", fontsize=8.5,
                            color="white" if M[r, c] > 0.6 * vmax else INK)
        ax.set_xticks(range(len(axes_names)), [a.replace("log10_", "") for a in axes_names], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(mets)), mets if j == 0 else [""] * len(mets), fontsize=8)
        ax.set_title(k, fontsize=10.5, color=INK)
        ax.tick_params(length=0); ax.grid(False)
    fig.suptitle(f"Linear $R^2$ of recovery on rarity (null expectation {D[order[0]]['fits'][mets[0]][axes_names[0]]['r2_null_expect']:.4f})",
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, out, stem)


def fig_unverbalized(D, order, rar, axis, out, stem):
    """The failure view: fraction of features NO rollout fires, by rarity decile, one line per arm."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for j, k in enumerate(order):
        b = D[k]["unverbalized"][axis]
        c = SERIES[j % len(SERIES)]
        ax.plot(b["x_mid"], b["y_mean"], color=c, lw=2.0, marker="o", ms=6, mfc="white", mec=c, mew=1.8,
                label=f"{k}  (AUC {D[k]['unverbalized_auc'][axis]:.3f}, overall {np.mean(1 - D[k]['recovery']['fired']):.2f})")
    ax.set_xlabel(RARITY_LABEL.get(axis, axis), fontsize=9)
    ax.set_ylabel("fraction unverbalized (no rollout fires the feature)", fontsize=9)
    ax.set_ylim(0, max(0.05, max(max(D[k]["unverbalized"][axis]["y_mean"]) for k in order) * 1.15))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    fig.suptitle("Where the inverter fails, by feature rarity", fontsize=11.5, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, out, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="TAG=PATH",
                    help="repeatable: an arm label and its --dump-per-dir json (e.g. rl=perdir_ckpt_150.json)")
    ap.add_argument("--sae-match", default=None, help="/data/mlp42/sae_match.npz (sae_nfire / sae_mean)")
    ap.add_argument("--maxacts", default=None, help="SAE max_acts .pt (top-N example support axes; needs torch)")
    ap.add_argument("--n-tok", type=int, default=N_TOK_DEFAULT, help="tokens in the sae_nfire scan")
    ap.add_argument("--mlp42-meta", default=None, help="/data/mlp42/meta.json — reads n_tokens from it instead")
    ap.add_argument("--primary-axis", default=None, help="rarity axis for the scatter/failure figures (default: first available)")
    ap.add_argument("--out", required=True, help="report folder (figures + data/*.json go here)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    n_tok = json.load(open(a.mlp42_meta))["n_tokens"] if a.mlp42_meta else a.n_tok

    order, D = [], {}
    for spec in a.perdir:
        tag, path = dumplib.split_spec(spec)
        order.append(tag); D[tag] = load_perdir(path)

    feats = D[order[0]]["feature"]
    for k in order[1:]:                       # frozen eval cache => identical rows; intersect defensively
        feats = np.intersect1d(feats, D[k]["feature"])
    for k in order:
        sel = np.searchsorted(D[k]["feature"], feats) if np.all(np.diff(D[k]["feature"]) > 0) else \
            np.array([int(np.where(D[k]["feature"] == f)[0][0]) for f in feats])
        assert np.array_equal(D[k]["feature"][sel], feats)
        D[k]["feature"] = feats
        D[k]["best_act"] = D[k]["best_act"][sel]; D[k]["corpus_peak"] = D[k]["corpus_peak"][sel]
        D[k]["recovery"] = {m: v[sel] for m, v in D[k]["recovery"].items()}
    print(f"[rarity] {len(feats)} held-out SAE features common to {len(order)} arm(s): {', '.join(order)}", flush=True)

    rar, src = load_rarity(feats, a.sae_match, n_tok, a.maxacts)
    peak = D[order[0]]["corpus_peak"]
    for k in order[1:]:
        assert np.allclose(peak, D[k]["corpus_peak"], rtol=1e-4), "arms disagree on corpus_peak — different max_acts?"
    rar["log10_corpus_peak"] = np.log10(np.clip(peak, 1e-6, None))
    src["log10_corpus_peak"] = "the dump's own corpus_peak (max_acts max over the corpus scan)"
    axes_names = [k for k in RARITY_LABEL if k in rar]
    primary = a.primary_axis or axes_names[0]
    assert primary in rar, f"--primary-axis {primary} not among {axes_names}"

    # Is the frequency axis actually SPARSE? sae_nfire counts pre-topk act > 0, not BatchTopK firing:
    # if most features are "on" at most tokens it is a saturated axis and every R^2 below is attenuated
    # toward 0 by that measurement noise (a low R^2 would then be uninformative, not negative evidence).
    warn = None
    if "log10_fire_freq" in rar:
        med = float(np.median(10 ** rar["log10_fire_freq"]))
        frac_dense = float(np.mean(rar["log10_fire_freq"] > np.log10(0.05)))
        print(f"[rarity] firing frequency (act > 0): median {med:.2%} of tokens, "
              f"{frac_dense:.1%} of features over 5%, range "
              f"{10 ** rar['log10_fire_freq'].min():.2%}-{10 ** rar['log10_fire_freq'].max():.2%}", flush=True)
        if med > 0.05:
            warn = (f"log10_fire_freq is NOT sparse (median {med:.1%} of tokens): act > 0 on a BatchTopK SAE is "
                    "'positive pre-activation', not firing. Every R^2 on this axis is attenuated; prefer the "
                    "--maxacts axes (topn_*), or rebuild the axis with a true top-k / relative-threshold corpus scan.")
            print(f"[rarity] WARNING: {warn}", flush=True)

    for k in order:
        d = D[k]
        d["fits"] = {m: {ax: fit(rar[ax], y) for ax in axes_names} for m, y in d["recovery"].items()}
        d["deciles"] = {m: {ax: deciles(rar[ax], y) for ax in axes_names} for m, y in d["recovery"].items()}
        unv = 1.0 - d["recovery"]["fired"]
        d["unverbalized"] = {ax: deciles(rar[ax], unv) for ax in axes_names}
        d["unverbalized_auc"] = {ax: auc(-rar[ax], unv) for ax in axes_names}      # -rarity: rarer => higher score
        d["multivariate"] = {}
        if "log10_fire_freq" in rar:
            for m, y in d["recovery"].items():
                r_f, _ = multi_r2(y, [rar["log10_fire_freq"]])
                r_p, _ = multi_r2(y, [rar["log10_corpus_peak"]])
                r_b, coef = multi_r2(y, [rar["log10_fire_freq"], rar["log10_corpus_peak"]])
                d["multivariate"][m] = {"r2_freq_only": r_f, "r2_peak_only": r_p, "r2_both": r_b,
                                        "semipartial_r2_freq": r_b - r_p, "semipartial_r2_peak": r_b - r_f,
                                        "coef_[intercept,freq,peak]": coef}

    mets = fig_scatter(D, order, rar, primary, a.out, f"fig1_recovery_vs_{primary}")
    fig_r2_matrix(D, order, axes_names, a.out, "fig2_r2_matrix")
    fig_unverbalized(D, order, rar, primary, a.out, f"fig3_unverbalized_vs_{primary}")

    out = {"question": "how much of the inverter's per-SAE-feature recovery spread is explained by feature rarity?",
           "n_features": int(len(feats)), "arms": {k: {"path": D[k]["path"], "ckpt_step": D[k]["ckpt_step"],
                                                       "tag": D[k]["tag"], "aggregates": D[k]["aggregates"]} for k in order},
           "protocol": {"recovery": "best-of-bo per-feature scores from eval_ckpt_daemon --dump-per-dir "
                                    "(inject unit(W_enc[:,f]) at INJECT_LAYER, generate, re-read the CLEAN base at "
                                    "READ_LAYER, max over kept tokens); higher = better recovered",
                        "sae_fire": SAE_FIRE, "n_tok_fire_scan": int(n_tok), "n_bins": NBINS, "n_boot": N_BOOT},
           "rarity_axes": {k: {"label": RARITY_LABEL.get(k, k), "source": src.get(k, ""),
                               "quantiles": {str(q): float(np.quantile(rar[k], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
                           for k in axes_names},
           "primary_axis": primary, "rarity_axis_warning": warn,
           "random_control_cos": {k: (None if D[k]["random_cos"] is None else
                                      {"mean": float(np.mean(D[k]["random_cos"])), "p95": float(np.quantile(D[k]["random_cos"], 0.95))})
                                  for k in order},
           "fits": {k: D[k]["fits"] for k in order},
           "deciles": {k: D[k]["deciles"] for k in order},
           "unverbalized_by_rarity_decile": {k: D[k]["unverbalized"] for k in order},
           "unverbalized_auc_rarer_is_worse": {k: D[k]["unverbalized_auc"] for k in order},
           "multivariate_freq_vs_peak": {k: D[k]["multivariate"] for k in order},
           "rarity_axis_intercorrelation": {f"{x}|{y}": float(stats.spearmanr(rar[x], rar[y]).statistic)
                                            for i, x in enumerate(axes_names) for y in axes_names[i + 1:]}}
    json.dump(out, open(os.path.join(a.out, "data", "recovery_vs_rarity.json"), "w"), indent=1)

    w = max(len(m) for m in mets) + 2
    print(f"\nlinear R^2 of recovery on rarity   (n={len(feats)}, null expectation {1 / (len(feats) - 1):.4f})")
    for k in order:
        print(f"\n  [{k}]  " + "".join(f"{ax.replace('log10_', ''):>20}" for ax in axes_names))
        for m in mets:
            cells = ""
            for ax in axes_names:
                f = D[k]["fits"][m][ax]
                cells += f"{'n/a':>20}" if f.get("r2") is None else f"{f['r2']:>13.3f}{'*' if f['p_value'] < 0.01 else ' ':<1}{f['spearman_rho']:>+6.2f}"
            print(f"  {m:<{w}}" + cells)
        print(f"  {'unverbalized AUC':<{w}}" + "".join(f"{D[k]['unverbalized_auc'][ax]:>20.3f}" for ax in axes_names))
    print(f"\n  cells: R^2 (* = p<0.01) then Spearman rho. Decile-bin R^2 and CIs in "
          f"{os.path.join(a.out, 'data', 'recovery_vs_rarity.json')}", flush=True)


if __name__ == "__main__":
    main()
