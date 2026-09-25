"""SFT (sft-simple2m) vs RL (rl-last16-lr5e-7) on the paper's eval sets, with the base model and NLA-AV
as references. Reads the `score` products' per_target ladders (unbiased best-of-k) and `sae_self`.

    python verbalization/analysis/plot_sl_vs_rl.py <dir with the fetched products>

(a) natural activations (v3_realact, 512 rows): centred cosine, best-of-k (paper convention).
(b) SAE features (v3_ctrl sae rows, 512): raw cosine, best-of-k.
(c) SAE features: share of features that fire (some sample clears tau) at best-of-8, dead excluded.
NLA-AV has 4 samples per row, so its curves stop at k=4.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criterion import GATE, dead_ids, none_of_k  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper-evals"))
from results.common import peaks_of  # noqa: E402

O = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "report"
MODELS = [("rl", "MAEM (RL, step 300)", "#c8643c", "-"), ("sft", "MAEM (SFT init)", "#3c78c8", "-"),
          ("nla", "NLA-AV", "#666666", "--"), ("base", "base model", "#aaaaaa", ":")]


def ladder(path, fam, prefix):
    rows = [json.loads(l) for l in open(path) if l.startswith("{")]
    rows = [r for r in rows if r["family"] == fam]
    ks = sorted(int(k[len(prefix):]) for k in rows[0] if k.startswith(prefix) and k[len(prefix):].isdigit())
    return ks, [float(np.mean([r[f"{prefix}{k}"] for r in rows])) for k in ks], len(rows)


def fired8(tag):
    s = open(O / f"{tag}_sae_self.json").read(); meta = json.loads(s[: s.rfind("}") + 1])
    act = np.fromfile(O / f"{tag}_sae_self.f16", dtype=np.float16).astype(np.float32)
    act = act.reshape(len(meta["features"]), int(meta["n"]), -1)
    dead = dead_ids()
    k = min(8, int(meta["n"]))
    v = [1 - none_of_k(peaks_of(act[i]) > GATE, k) for i, f in enumerate(meta["features"]) if int(f) not in dead]
    return float(np.mean(v)), k, len(v)


def main():
    res = {"realact_centred": {}, "sae_raw": {}, "sae_fired": {}}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for tag, lab, col, ls in MODELS:
        p = O / f"{tag}_realact.jsonl"
        if p.exists():
            ks, ys, n = ladder(p, "realact", "bo_c_")
            ax[0].plot(ks, ys, ls, color=col, marker="o", ms=3, label=lab); res["realact_centred"][tag] = dict(zip(ks, ys))
        p = O / f"{tag}_ctrl.jsonl"
        if p.exists():
            ks, ys, n = ladder(p, "sae", "bo_")
            ax[1].plot(ks, ys, ls, color=col, marker="o", ms=3, label=lab); res["sae_raw"][tag] = dict(zip(ks, ys))
        if (O / f"{tag}_sae_self.json").exists():
            fr, k, n = fired8(tag); res["sae_fired"][tag] = {"fired": fr, "k": k, "n": n}
    for a, t in ((ax[0], "(a) natural activations: centred cosine"), (ax[1], "(b) SAE features: cosine")):
        a.set_xscale("log", base=2); a.set_xlabel("best of k samples"); a.set_ylabel("mean cosine"); a.set_title(t, fontsize=10)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, frameon=False)
    tags = [t for t, *_ in MODELS if t in res["sae_fired"]]
    vals = [100 * res["sae_fired"][t]["fired"] for t in tags]
    cols = [c for t, _, c, _ in MODELS if t in res["sae_fired"]]
    labs = [f"{l}\n(best of {res['sae_fired'][t]['k']})" for t, l, _, _ in MODELS if t in res["sae_fired"]]
    ax[2].bar(range(len(tags)), vals, color=cols)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax[2].set_xticks(range(len(tags))); ax[2].set_xticklabels(labs, fontsize=8)
    ax[2].set_ylim(0, 105); ax[2].set_ylabel("% of SAE features fired"); ax[2].set_title("(c) SAE features: target fires", fontsize=10)
    fig.savefig(OUT / "fig_sl_vs_rl.pdf"); fig.savefig(OUT / "fig_sl_vs_rl.png", dpi=150)
    json.dump(res, open(OUT / "data" / "sl_vs_rl.json", "w"), indent=1)
    for k, v in res.items():
        print(k, json.dumps({t: ({kk: round(vv, 4) for kk, vv in d.items()} if k != "sae_fired" else d) for t, d in v.items()}))


if __name__ == "__main__":
    main()
