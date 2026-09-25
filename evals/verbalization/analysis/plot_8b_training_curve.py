"""8B MAEMM training curve (the run1, /vol/checkpoints/runs/run1): held-out cosine and SAE
firing over SFT steps then RL steps, plus the RL training reward (the cosine) per step.

    python evals/verbalization/analysis/plot_8b_training_curve.py

Inputs (copied from the volume into report/data/run1_8b/): sft_heldout_eval.jsonl, heldout_eval.jsonl
(RL; two passes, 0-200 and 0-250 -- the longer is plotted solid, the other as a faint replicate),
metrics.jsonl (RL training reward). `eval/sae/norm_act` is not used: several rows carry broken
values (287.9, 1321.4). The SFT `final` (-1) row and the RL `eval_universal` final row are dropped;
RL step 0 is the SFT endpoint.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "report"
D = R / "data" / "run1_8b"


def rows(f):
    return [json.loads(l) for l in open(D / f) if l.startswith("{")]


def main():
    sft = [r for r in rows("sft_heldout_eval.jsonl") if r["ckpt_step"] >= 0]
    rl_all = [r for r in rows("heldout_eval.jsonl") if r.get("source") == "inline"]
    passes, cur = [], []
    for r in rl_all:                                   # split the two passes at the step reset
        if cur and r["ckpt_step"] <= cur[-1]["ckpt_step"]:
            passes.append(cur); cur = []
        cur.append(r)
    passes.append(cur)
    passes.sort(key=lambda p: -p[-1]["ckpt_step"])
    main_rl, rep = passes[0], passes[1:]
    S = max(r["ckpt_step"] for r in sft)
    RLW = max(r["ckpt_step"] for r in main_rl)
    k = S / RLW                                        # RL gets the same axis width as SFT
    X = lambda s: S + s * k                            # noqa: E731  RL step -> axis position
    met_all = rows("metrics.jsonl")
    mp, cur = [], []                                   # metrics.jsonl also holds two passes
    for m in met_all:
        if cur and m["step"] <= cur[-1]["step"]:
            mp.append(cur); cur = []
        cur.append(m)
    mp.append(cur)
    met = max(mp, key=len)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9), constrained_layout=True)
    series = [("eval/realact/cos", "natural activations", "#c8643c"),
              ("eval/realact_long/cos", "long-context activations", "#3c78c8"),
              ("eval/random/cos", "random directions (control)", "#999999")]
    for key, lab, col in series:
        ax[0].plot([r["ckpt_step"] for r in sft], [r[key] for r in sft], "o-", color=col, ms=3, label=lab)
        ax[0].plot([X(r["ckpt_step"]) for r in main_rl], [r[key] for r in main_rl], "o-", color=col, ms=3)
        for p in rep:
            ax[0].plot([X(r["ckpt_step"]) for r in p], [r[key] for r in p], "-", color=col, alpha=0.3)
    ax[1].plot([r["ckpt_step"] for r in sft], [100 * r["eval/sae/fired"] for r in sft], "o-", color="#7a4fb0", ms=3)
    ax[1].plot([X(r["ckpt_step"]) for r in main_rl], [100 * r["eval/sae/fired"] for r in main_rl], "o-", color="#7a4fb0", ms=3)
    for p in rep:
        ax[1].plot([X(r["ckpt_step"]) for r in p], [100 * r["eval/sae/fired"] for r in p], "-", color="#7a4fb0", alpha=0.3)
    sft_t = [0, 400, 800, 1200]
    rl_t = [50, 100, 150, 200, 250]
    for a in ax[:2]:
        a.axvline(S, color="k", lw=0.8, ls="--"); a.text(S, a.get_ylim()[1], " RL starts", va="top", fontsize=8)
        a.set_xticks(sft_t + [X(t) for t in rl_t])
        a.set_xticklabels([str(t) for t in sft_t] + [str(t) for t in rl_t], fontsize=8)
        a.set_xlabel("SFT step  |  RL step (RL axis stretched)"); a.grid(alpha=0.3)
    ax[0].set_ylabel("held-out cosine"); ax[0].set_title("(a) held-out cosine", fontsize=10); ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_ylabel("% of held-out SAE features fired"); ax[1].set_title("(b) SAE features: target fires", fontsize=10)
    st = np.array([m["step"] for m in met]); rw = np.array([m["reward/mean"] for m in met])
    w = 10
    sm = np.convolve(rw, np.ones(w) / w, mode="valid")
    ax[2].plot(st, rw, color="#c8643c", alpha=0.25, lw=0.8)
    ax[2].plot(st[w - 1:], sm, color="#c8643c", lw=1.8)
    ax[2].set_xlabel("RL step"); ax[2].set_ylabel("training reward (max-token cosine)")
    ax[2].set_title("(c) RL training reward", fontsize=10); ax[2].grid(alpha=0.3)
    ax[2].set_xlim(0, st.max())
    fig.suptitle("Qwen3-8B MAEM (run1): SFT then RL", fontsize=11)
    fig.savefig(R / "fig_8b_training_curve.pdf"); fig.savefig(R / "fig_8b_training_curve.png", dpi=150)
    out = {"sft": [{k: r[k] for k in ("ckpt_step", "eval/realact/cos", "eval/realact_long/cos", "eval/random/cos", "eval/sae/fired")} for r in sft],
           "rl": [{k: r[k] for k in ("ckpt_step", "eval/realact/cos", "eval/realact_long/cos", "eval/random/cos", "eval/sae/fired")} for r in main_rl],
           "reward_first10": float(rw[:10].mean()), "reward_last10": float(rw[-10:].mean()), "n_reward_steps": int(len(rw))}
    json.dump(out, open(R / "data" / "8b_training_curve.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}))
    print("SFT first/last:", out["sft"][0], out["sft"][-1]); print("RL first/last:", out["rl"][0], out["rl"][-1])


if __name__ == "__main__":
    main()
