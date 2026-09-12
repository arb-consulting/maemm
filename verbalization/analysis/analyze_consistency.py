"""Are the inverter's attempts CONSISTENT on features it cannot activate?

Hypothesis: if a feature is unverbalizable because the model has no working hypothesis about what it
means, its many generations should be mutually dissimilar -- the model is sampling scattershot. If
instead the generations cluster tightly but still fail to activate the feature, the model has one
confident-but-wrong explanation, which is a different failure.

Design: two rarity-MATCHED groups of held-out SAE features (selected from a prior eval_dirs run):
  hard  best_act == 0      the inverter produces no activation at all
  easy  norm_act >= 0.50   reaches at least half the feature's corpus peak
Each feature gets many samples (eval_dirs --bo 32, which dumps texts_8b_<tag>.json). Embed every
generation, then per feature take the mean pairwise cosine = a self-consistency score.

Three readouts:
  1. group difference        hard vs easy self-consistency
  2. continuous             corr(self-consistency, achieved norm_act) over all features
  3. corpus control         cos(generations, that feature's real corpus examples) -- distinguishes
                            "self-consistent but pointing somewhere wrong" from "incoherent"

    python verbalization/analysis/analyze_consistency.py --texts texts_8b_consistency.json \
        --perdir perdir_8b_consistency.json --groups consistency_features.json \
        --maxacts acts_Qwen_Qwen3-8B_layer_27_trainer_2_layer_percent_75_context_length_32.pt \
        --out verbalization/report
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

EMB = "BAAI/bge-small-en-v1.5"


def pairwise_mean_cos(E):
    """Mean off-diagonal cosine of L2-normalised rows."""
    if len(E) < 2:
        return np.nan
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    return float(S[iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="texts_8b_<tag>.json from eval_dirs")
    ap.add_argument("--perdir", required=True, help="perdir_8b_<tag>.json from the same run")
    ap.add_argument("--groups", required=True, help="{'hard': [...], 'easy': [...]} feature ids")
    ap.add_argument("--maxacts", default=None, help="SAE max_acts .pt for the corpus control")
    ap.add_argument("--sae-match", default=None, help="sae_match_8b.npz, for the rarity column")
    ap.add_argument("--dump-csv", default=None,
                    help="also write the per-GENERATION table (one row per sample) to this path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/data", exist_ok=True)

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMB)

    T = json.load(open(a.texts))
    texts, rows, feats = T["texts"], np.asarray(T["rows"]), np.asarray(T["feats"])
    s = {k: np.asarray(v) for k, v in json.load(open(a.perdir))["perdir"]["sae"].items()}
    grp = json.load(open(a.groups))
    hard, easy = set(grp["hard"]), set(grp["easy"])

    print(f"embedding {len(texts)} generations from {len(feats)} features ...", flush=True)
    E = emb.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

    # optional corpus control: embed each feature's real max-activating windows
    corpus_cos = {}
    if a.maxacts:
        import torch
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        d = torch.load(a.maxacts, map_location="cpu", weights_only=False)
        MT, MA = d["max_tokens"], d["max_acts"].float()
        for i, f in enumerate(feats):
            ex = MA[int(f)].max(1).values
            top = torch.argsort(ex, descending=True)[:8]
            ctx = [tok.decode(MT[int(f), int(j)].tolist()).replace("<|im_end|>", " ") for j in top]
            C = emb.encode(ctx, normalize_embeddings=True, show_progress_bar=False)
            G = E[rows == i]
            corpus_cos[int(f)] = float((G @ C.T).mean()) if len(G) else np.nan

    recs = []
    for i, f in enumerate(feats):
        G = E[rows == i]
        recs.append({"feature": int(f), "n_gen": int(len(G)),
                     "self_cos": pairwise_mean_cos(G),
                     "corpus_cos": corpus_cos.get(int(f)),
                     "best_act": float(s["best_act"][i]), "norm_act": float(s["norm_act"][i]),
                     "group": "hard" if int(f) in hard else ("easy" if int(f) in easy else "?")})

    sc = np.array([r["self_cos"] for r in recs])
    na = np.array([r["norm_act"] for r in recs])
    gh = np.array([r["group"] == "hard" for r in recs])
    ge = np.array([r["group"] == "easy" for r in recs])

    print()
    print("%-6s %4s %14s %14s %14s" % ("group", "n", "self-cos", "corpus-cos", "norm_act"))
    for lab, m in (("hard", gh), ("easy", ge)):
        cc = np.array([r["corpus_cos"] for r in recs if r["group"] == lab and r["corpus_cos"] is not None])
        print("%-6s %4d %14.4f %14s %14.3f" % (
            lab, m.sum(), np.nanmean(sc[m]),
            f"{cc.mean():.4f}" if len(cc) else "-", np.median(na[m])))
    if gh.sum() and ge.sum():
        t = stats.mannwhitneyu(sc[gh], sc[ge])
        print("\nhard vs easy self-consistency: Mann-Whitney p = %.4g  (hard %s)"
              % (t.pvalue, "LOWER = scattershot" if np.nanmean(sc[gh]) < np.nanmean(sc[ge]) else "HIGHER"))
    ok = np.isfinite(sc) & np.isfinite(na)
    r_, p_ = stats.spearmanr(sc[ok], na[ok])
    print("continuous: Spearman(self-consistency, norm_act) rho = %+.3f  p = %.4g  (n=%d)" % (r_, p_, ok.sum()))

    if a.dump_csv:
        import csv as _csv
        fire = None
        if a.sae_match:
            fire = np.load(a.sae_match)
            fire = fire["sae_nfire"] / float(fire["n_tok"]) * 100.0
        peak = {int(f): float(s["corpus_peak"][i]) for i, f in enumerate(feats)}
        cols = ["group", "feature", "fire_pct", "corpus_peak", "norm_act_bo32", "self_cos",
                "var_cos", "sample_idx", "sample_act", "generation", "top_corpus_example"]
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_csv)) or ".", exist_ok=True)
        with open(a.dump_csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for i, f in enumerate(feats):
                r = recs[i]
                G = E[rows == i]
                # var_cos: spread of the pairwise similarities, not just their mean. A feature can
                # be moderately self-consistent on average because it is bimodal -- two confident
                # clusters -- which is a different failure from uniformly scattered sampling.
                if len(G) > 1:
                    S = G @ G.T
                    iu = np.triu_indices(len(G), 1)
                    var_cos = float(S[iu].var())
                else:
                    var_cos = float("nan")
                idxs = np.where(rows == i)[0]
                for j, ti in enumerate(idxs):
                    w.writerow({"group": r["group"], "feature": int(f),
                                "fire_pct": round(float(fire[int(f)]), 5) if fire is not None else "",
                                "corpus_peak": round(peak[int(f)], 1),
                                "norm_act_bo32": round(r["norm_act"], 4),
                                "self_cos": round(r["self_cos"], 4), "var_cos": round(var_cos, 5),
                                "sample_idx": j, "sample_act": "",
                                "generation": texts[int(ti)].replace("\n", " ").strip(),
                                "top_corpus_example": ""})
        print(f"-> {a.dump_csv}")

    json.dump({"embedder": EMB, "n_features": len(feats), "per_feature": recs,
               "hard_self_cos": float(np.nanmean(sc[gh])), "easy_self_cos": float(np.nanmean(sc[ge])),
               "spearman_selfcos_normact": {"rho": float(r_), "p": float(p_)}},
              open(f"{a.out}/data/consistency.json", "w"), indent=1)
    print(f"\n-> {a.out}/data/consistency.json")


if __name__ == "__main__":
    main()
