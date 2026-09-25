"""SFT -> RL, 8B and 27B MAEMs on one footing: the same pipeline (rollouts_vllm n=64, `score`,
unbiased best-of-k), each base's own held-out set (8B: 2026-09-16_v1; 27B: the paper's v3 sets).

    python evals/verbalization/analysis/plot_unified_sl_rl.py

(a) natural activations, centred cosine, best-of-1 and best-of-8, over RL steps (step 0 = the SFT
    endpoint). 8B: the run1, SFT end + RL steps 25..200 + final (step 300). 27B: only the SFT init
    and the step-300 checkpoint exist, drawn as two points.
(b) the same for SAE features (raw cosine; each base's own SAE).
(c) the random-direction control (8B only; the 27B SFT has no random rows scored).
(d) best-of-k at the endpoints: SFT vs RL for both sizes, natural activations.
Caveat: the 8B checkpoints were rolled out with --mu <run1 whiten_mu>, a CHOSEN injection convention
(run1's is unrecorded); the SAE rows do not depend on it.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "report"
D8, D27 = R / "data" / "unify_8b", R / "data" / "unify_27b"
STEPS8 = [("sft", 0), ("rl-step25", 25), ("rl-step40", 40), ("rl-step60", 60), ("rl-step90", 90),
          ("rl-step130", 130), ("rl-step200", 200), ("rl-final", 300)]
C8, C27 = "#3c78c8", "#c8643c"


def mean(path, fam, key):
    v = [json.loads(l)[key] for l in open(path) if l.startswith("{") and json.loads(l)["family"] == fam]
    return float(np.mean(v)) if v else float("nan")


def main():
    res = {"8b": {}, "27b": {}}
    for tag, st in STEPS8:
        p = D8 / f"{tag}.jsonl"
        res["8b"][st] = {f"{fam}_{k}": mean(p, fam, key) for fam, pre in (("realact", "bo_c_"), ("sae", "bo_"), ("random", "bo_"))
                          for k in (1, 8, 64) for key in [f"{pre}{k}"]}
    for tag, st, suf in (("sft", 0, ""), ("rl", 300, "")):
        res["27b"][st] = {f"realact_{k}": mean(D27 / f"{tag}_realact.jsonl", "realact", f"bo_c_{k}") for k in (1, 8, 64)}
        res["27b"][st].update({f"sae_{k}": mean(D27 / f"{tag}_ctrl.jsonl", "sae", f"bo_{k}") for k in (1, 8, 64)})
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
    for i, fam, title in ((0, "realact", "(a) natural activations: centred cosine"), (1, "sae", "(b) SAE features: cosine")):
        for size, col, data in (("8B", C8, res["8b"]), ("27B", C27, res["27b"])):
            st = sorted(data)
            for k, ls, mk in ((1, "-", "o"), (8, "--", "s")):
                # the 27B has only its two endpoints: draw points, not a line through steps we never saw
                ax[i].plot(st, [data[s][f"{fam}_{k}"] for s in st], color=col, linestyle=ls if size == "8B" else "none",
                           marker=mk, ms=4 if size == "8B" else 8, label=f"{size}, best of {k}")
        ax[i].set_title(title, fontsize=10); ax[i].set_xlabel("RL step (0 = end of SFT)"); ax[i].set_ylabel("mean cosine")
        ax[i].grid(alpha=0.3)
    ax[0].legend(fontsize=8, frameon=False)
    st = sorted(res["8b"])
    for k, ls in ((1, "-"), (8, "--")):
        ax[2].plot(st, [res["8b"][s][f"random_{k}"] for s in st], ls, color="#888888", marker="o", ms=3, label=f"8B, best of {k}")
    ax[2].set_ylim(0, max(0.1, ax[2].get_ylim()[1])); ax[2].set_title("(c) random directions (control)", fontsize=10)
    ax[2].set_xlabel("RL step (0 = end of SFT)"); ax[2].set_ylabel("mean cosine"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8, frameon=False)
    ks = [1, 2, 4, 8, 16, 32, 64]
    for size, col, sft_p, rl_p in (("8B", C8, D8 / "sft.jsonl", D8 / "rl-final.jsonl"),
                                   ("27B", C27, D27 / "sft_realact.jsonl", D27 / "rl_realact.jsonl")):
        ax[3].plot(ks, [mean(sft_p, "realact", f"bo_c_{k}") for k in ks], ":", color=col, marker="o", ms=3, label=f"{size} SFT")
        ax[3].plot(ks, [mean(rl_p, "realact", f"bo_c_{k}") for k in ks], "-", color=col, marker="o", ms=3, label=f"{size} RL")
    ax[3].set_xscale("log", base=2); ax[3].set_xlabel("best of k samples"); ax[3].set_ylabel("mean centred cosine")
    ax[3].set_title("(d) natural activations: SFT vs RL by sample budget", fontsize=10); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8, frameon=False)
    fig.savefig(R / "fig_unified_sl_rl.pdf"); fig.savefig(R / "fig_unified_sl_rl.png", dpi=150)
    json.dump(res, open(R / "data" / "unified_sl_rl.json", "w"), indent=1)
    for size in res:
        for s in sorted(res[size]):
            v = res[size][s]
            print(f"{size:4s} step {s:3d}  realact bo1 {v['realact_1']:.3f} bo8 {v['realact_8']:.3f}  sae bo1 {v['sae_1']:.3f} bo8 {v['sae_8']:.3f}"
                  + (f"  random bo1 {v['random_1']:.3f}" if 'random_1' in v else ""))


if __name__ == "__main__":
    main()
