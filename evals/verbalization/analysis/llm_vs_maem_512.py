"""The LLM baseline against the MAEM on the paper's 512-feature set (2026-09-21_v3_ctrl, §4.4).

    python evals/verbalization/analysis/llm_vs_maem_512.py --mirror <dir>

Both arms are read the same way the paper reads the MAEM (`results/faithfulness.sae_cells`):
per-draw peak pre-gate activation of the target feature on the clean base at layer 42, through
`results.common.bo_unbiased` (the one best-of-k estimator of the pipeline). "Fired" is the gate
indicator through the same estimator, so it needs no denominator. The ratio uses `corpus_peak`
as `sae_self` and the autointerp build both record it (the 16M held-out scan's max_act.f16), the
SAME number for both arms.

The draws differ: the MAEM has 64 rollouts per feature, the LLM 16 texts (sonnet-5, prompted
with the paper's C16 Delphi block -- the block the App. F.2 explainer saw). The estimator is
unbiased at every k <= n, so bo1 and bo8 are comparable; bo16+ exists for the MAEM only.

<mirror> holds: sae_self.json + sae_self.f16 (the paper's product,
scores/2026-09-21_v3_ctrl__vllm__paper0923/sae_self), llm_c16_claude-sonnet-5.scored.jsonl and
llm_c16_claude-sonnet-5.meta.jsonl (runs/2026-09-23_section).
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent / "evals/faithfulness"))
from results.common import bo_unbiased, peaks_of  # noqa: E402

KS = (1, 8)
NORM_BAR = 0.10          # the rarity figure's norm criterion: best act >= 10% of the corpus peak


def load_json(p):
    # `modal volume get ... -` appends its progress line to stdout; keep the JSON only
    s = open(p).read()
    return json.loads(s[: s.rfind("}") + 1])


def summarise(feats, peaks, cp, gate, strata):
    out = {"n_features": len(feats)}
    for k in KS:
        bo = np.array([bo_unbiased(p, k) for p in peaks])
        fired = np.array([bo_unbiased((p > gate).astype(float), k) for p in peaks])
        ratio = bo / cp
        out[f"bo{k}"] = {
            "fired": round(float(fired.mean()), 4),
            "ratio_median": round(float(np.median(ratio)), 4),
            "norm_ge_0.1": round(float((ratio >= NORM_BAR).mean()), 4),
            "by_stratum": {int(s): {"fired": round(float(fired[strata == s].mean()), 4),
                                    "ratio_median": round(float(np.median(ratio[strata == s])), 4),
                                    "n": int((strata == s).sum())}
                           for s in sorted(set(strata.tolist()))},
        }
    out["fired_any"] = round(float(np.mean([p.max() > gate for p in peaks])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--out", default="evals/verbalization/report/data/llm_vs_maem_512.json")
    a = ap.parse_args()
    M = pathlib.Path(a.mirror)

    meta = load_json(M / "sae_self.json")
    gate = float(meta["gate"])
    act = np.fromfile(M / "sae_self.f16", dtype=np.float16).astype(np.float32)
    act = act.reshape(len(meta["features"]), int(meta["n"]), -1)
    stored = {int(p["feature"]): p for p in meta["per_target"]}
    m_peaks = {int(f): peaks_of(act[i]) for i, f in enumerate(meta["features"])}

    lmeta = {int(r["feature"]): r for r in map(json.loads, open(M / "llm_c16_claude-sonnet-5.meta.jsonl"))}
    l_acts = collections.defaultdict(list)
    for r in map(json.loads, open(M / "llm_c16_claude-sonnet-5.scored.jsonl")):
        l_acts[int(r["feature"])].append(float(r["act"]))

    # the comparison set: features both arms have, with at least 8 LLM texts (bo8 needs k <= n)
    both = sorted(f for f in m_peaks if len(l_acts.get(f, [])) >= max(KS))
    dropped = sorted(set(m_peaks) - set(both))
    cp = np.array([float(stored[f]["corpus_peak"]) for f in both])
    cp_llm = np.array([float(lmeta[f]["corpus_peak"]) for f in both])
    assert np.allclose(cp, cp_llm, rtol=1e-3), "the two arms' corpus peaks disagree"
    strata = np.array([int(lmeta[f]["stratum"]) for f in both])

    m = summarise(both, [m_peaks[f] for f in both], cp, gate, strata)
    l_ = summarise(both, [np.array(l_acts[f]) for f in both], cp, gate, strata)

    # per-feature bo8 fired, to see whether the two fail on the same features
    mf = np.array([bo_unbiased((m_peaks[f] > gate).astype(float), 8) for f in both])
    lf = np.array([bo_unbiased((np.array(l_acts[f]) > gate).astype(float), 8) for f in both])
    m_fail, l_fail = mf < 0.5, lf < 0.5
    overlap = {"maem_fail": int(m_fail.sum()), "llm_fail": int(l_fail.sum()),
               "both_fail": int((m_fail & l_fail).sum()),
               "maem_only_fail": int((m_fail & ~l_fail).sum()),
               "llm_only_fail": int((~m_fail & l_fail).sum()),
               "maem_only_fail_features": [f for f, x in zip(both, m_fail & ~l_fail) if x],
               "llm_only_fail_features": [f for f, x in zip(both, ~m_fail & l_fail) if x]}

    res = {"set": "2026-09-21_v3_ctrl (sae family, 512)", "gate": gate,
           "denominator": "corpus_peak as sae_self records it: sae/l42-1b/max_act.f16 (16M held-out scan)",
           "estimator": "results.common.bo_unbiased", "maem_draws": int(meta["n"]),
           "llm_draws": "16 per feature, sonnet-5 on the C16 block", "n_compared": len(both),
           "n_dropped": len(dropped), "dropped_features": dropped,
           "maem": m, "llm": l_, "fail_overlap_bo8_fired<0.5": overlap}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for arm in ("maem", "llm"):
        r = res[arm]
        print(f"{arm:6s} n={r['n_features']}  fired bo1 {r['bo1']['fired']:.3f}  bo8 {r['bo8']['fired']:.3f}  "
              f"ratio_med bo1 {r['bo1']['ratio_median']:.3f} bo8 {r['bo8']['ratio_median']:.3f}  "
              f"norm>=.1 bo8 {r['bo8']['norm_ge_0.1']:.3f}")
    print({k: v for k, v in overlap.items() if not k.endswith("features")})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
