"""Map evals/faithfulness products onto this folder's `perdir_*.json` schema, so the 8B figure code
runs unchanged on the 27B.

    python evals/verbalization/analysis/from_precompute.py --mirror <dir> --tag rl-last16

The 8B pipeline produced everything itself (`modal_8b_verbalization.py::eval_dirs`). On the 27B the
same quantities already exist as separate products on volume `maemm`, so this joins them instead:

    sae_self/sae_self.json   per_target -> max_peak_act  = best_act (max over rollouts and tokens)
                                           fire_fraction = share of own rollouts clearing the gate
    sae/<sae>/max_act.f16    the 16M corpus peak         = corpus_peak (THE denominator; see below)
    sae/<sae>/fire_counts.i64 + sizes.json               -> the rarity axis, as `--sae-match`
    heldout/<set>/ids.jsonl  stratum, train/test side, and the feature id per row
    pool_heldout/sae.parquet target_text -> the feature's peak token -> token_class (fig10's axis)

WHICH CRITERIA SURVIVE THE MOVE. fig6 plots five; only two are defined here, and pretending
otherwise would put a curve on the page with nothing behind it:

    norm    norm_act >= 0.10           YES -- max_act.f16 is full-dictionary
    fire    clears the learned gate    YES -- sae_self measures it directly, and it is cleaner than
                                       the 8B's `bar` (best_act > 1.0), which is ~1% of a median
                                       corpus peak and near-vacuous
    top16   >= top-16 corpus example   NO  -- needs sae/<sae>/examples/, which covers the 512-row
    last    >= weakest top example     NO     `2026-09-16_v1` draw: 10 of these 2000 features
    null    > own null p95             NO  -- `sae_self` is self-only by construction; the
                                       per-feature null needs the off-target texts, which no
                                       product on the volume holds

DENOMINATOR. `corpus_peak` is `max_act.f16`, our 16M-corpus peak -- NOT the `corpus_peak_1b` on the
ids rows. That column is the activation at ONE selected pool window, not a searched maximum: it is
LOWER than the 16M peak for 71% of these features (median ratio 0.698), which a max over more
tokens cannot be. Both are emitted so the choice stays visible.
"""
import argparse
import collections
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from featlib import token_class

FIRE_AXIS = {"raw": 0, "gated": 1}          # last axis of fire_counts.i64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True, help="dir holding the fetched products")
    ap.add_argument("--tag", required=True, help="arm name, e.g. rl-last16")
    ap.add_argument("--fire-axis", default="raw", choices=sorted(FIRE_AXIS),
                    help="raw = act > 0 (the 8B figure's axis); gated = the learned BatchTopK gate")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--out", default="evals/verbalization/report/data")
    a = ap.parse_args()
    M = pathlib.Path(a.mirror)

    self_j = json.load(open(M / "sae_self.json"))
    per = {int(r["feature"]): r for r in self_j["per_target"]}
    ids = [json.loads(l) for l in open(M / "ids.jsonl")]
    ids = [r for r in ids if int(r["id"]) in per]
    feat = np.array([int(r["id"]) for r in ids])

    max_act = np.fromfile(M / "max_act.f16", dtype=np.float16).astype(np.float64)
    fc = np.fromfile(M / "fire_counts.i64", dtype=np.int64).reshape(len(max_act), -1, 2)
    sizes = json.load(open(M / "sizes.json"))

    best_act = np.array([float(per[f]["max_peak_act"]) for f in feat])
    mean_act = np.array([float(per[f]["mean_peak_act"]) for f in feat])
    fire_frac = np.array([float(per[f]["fire_fraction"]) for f in feat])
    cp16 = max_act[feat]
    cp1b = np.array([float(r.get("corpus_peak_1b") or np.nan) for r in ids])
    norm_act = best_act / np.maximum(cp16, 1e-9)

    # the peak token of each feature, from the pool window that scored `corpus_peak_1b`
    tclass = np.array(["unknown"] * len(feat), dtype=object)
    pool = M / "pool_sae.parquet"
    if pool.exists():
        import pandas as pd
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        text = pd.read_parquet(pool).set_index("feature")["target_text"]
        for i, f in enumerate(feat):
            if f in text.index:
                pieces = tok.tokenize(text.loc[f])
                if pieces:
                    tclass[i] = token_class(tok.convert_tokens_to_string([pieces[-1]]))

    out = {
        "arm": a.tag, "source": "evals/faithfulness sae_self", "model": "Qwen/Qwen3.6-27B",
        "read_layer": 42, "d_sae": int(len(max_act)), "sae": self_j.get("sae", ""),
        "n": len(feat), "bo": int(self_j.get("n", 0)), "gate": self_j.get("gate"),
        "criteria_absent": ["top16", "last", "null"],
        "aggregates": {
            "norm_act": float(np.mean(norm_act)),
            "norm_act_median": float(np.median(norm_act)),
            "best_act_median": float(np.median(best_act)),
            "frac_firing": float(np.mean(fire_frac > 0)),
            "fire_fraction_median": float(np.median(fire_frac)),
        },
        "perdir": {"sae": {
            "row": [int(r["row"]) for r in ids],
            "feature": feat.tolist(),
            "best_act": best_act.tolist(),
            "mean_act": mean_act.tolist(),
            "corpus_peak": cp16.tolist(),
            "corpus_peak_1b": cp1b.tolist(),
            "norm_act": norm_act.tolist(),
            "fire_fraction": fire_frac.tolist(),
            "stratum": [int(r.get("stratum", -1)) for r in ids],
            "side": [r.get("side", "") for r in ids],
            "token_class": tclass.tolist(),
        }},
    }
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/perdir_27b_{a.tag}.json", "w"))

    k = FIRE_AXIS[a.fire_axis]
    # `scanned_positions`, not corpus tokens: the 64/16 windows overlap ~4x, and the 8B axis was
    # a rate over SCANNED positions too (its 1.02M = 4000 windows x 256 tok).
    np.savez(f"{a.out}/sae_match_27b.npz",
             sae_nfire=fc[:, -1, k], n_tok=np.int64(sizes["scanned_positions"][-1]))
    by = collections.Counter(tclass.tolist())
    print(f"[from_precompute] {len(feat)} features, bo={out['bo']}  "
          f"median norm_act {out['aggregates']['norm_act_median']:.3f}  "
          f"firing {100*out['aggregates']['frac_firing']:.1f}%")
    print("[from_precompute] token classes: " + json.dumps(dict(by.most_common())))


if __name__ == "__main__":
    main()
