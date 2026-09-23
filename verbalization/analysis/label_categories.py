"""Which CATEGORIES of feature are unverbalizable -- from blind LLM labels of each feature's windows.

    # a test set (MAEMM only): never-activated / failure rate per label, against rarity
    python verbalization/analysis/label_categories.py set --perdir <perdir> --labels <labels jsonl> \
        --sae-match <npz> --out <json>

    # the paper's 512: MAEMM vs the LLM baseline per label (and per token mechanism)
    python verbalization/analysis/label_categories.py p512 --mirror <p512 dir> --labels <labels jsonl> \
        --mechanics <mechanics jsonl> --sae-match <npz> --out <json>

Labels come from modal_27b_section.label_features: claude-sonnet-5 sees 8 marked top windows and
NOT the pass/fail outcome, and returns a category, what the firing depends on, and the marked
token's type. "Never activated" = no rollout clears the SAE gate; "fail" = norm_act < 0.10 (sets)
or bo8 fired < 0.5 (p512, both arms through results.common.bo_unbiased, as llm_vs_maemm_512.py).
"""
import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "paper-evals"))
FIELDS = ("category", "depends_on", "marked_token_type")


def rarity_p(fail, feat, sae_match):
    z = np.load(sae_match)
    x = np.log10(np.maximum(z["sae_nfire"][feat], 1) / float(z["n_tok"]))
    e = np.quantile(x, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(e, x, side="right") - 1, 0, 9)
    rate = np.array([fail[dec == b].mean() for b in range(10)])
    return rate[dec], x


def oe(fail, p, m):
    o, e = int(fail[m].sum()), float(p[m].sum())
    var = float((p[m] * (1 - p[m])).sum())
    zz = (o - e) / math.sqrt(var) if var > 0 else 0.0
    return {"O/E": round(o / e, 2) if e > 0 else None, "z": round(zz, 2),
            "p": float(f"{math.erfc(abs(zz) / math.sqrt(2)):.3g}")}


def load_labels(path):
    out = {}
    for r in map(json.loads, open(path)):
        if r.get("label"):
            out[int(r["feature"])] = r["label"]
    return out


def run_set(a):
    d = json.load(open(a.perdir))["perdir"]["sae"]
    lab = load_labels(a.labels)
    idx = [i for i, f in enumerate(d["feature"]) if int(f) in lab]
    feat = np.asarray(d["feature"], int)[idx]
    norm = np.asarray(d["norm_act"], float)[idx]
    never = np.asarray(d["fire_fraction"], float)[idx] == 0
    fail = norm < 0.10
    p, x = rarity_p(fail, feat, a.sae_match)
    res = {"perdir": a.perdir, "labels": a.labels, "n": int(len(feat)),
           "never_activated": round(float(never.mean()), 4), "fail": round(float(fail.mean()), 4)}
    for fld in FIELDS:
        vals = np.array([str(lab[int(f)].get(fld, "?")) for f in feat])
        rows = []
        for v in sorted(set(vals.tolist())):
            m = vals == v
            descr = [lab[int(f)].get("description", "") for f in feat[m & fail][:4]]
            rows.append({"value": v, "n": int(m.sum()), "never_activated": round(float(never[m].mean()), 3),
                         "fail": round(float(fail[m].mean()), 3),
                         "share_of_all_failures": round(float(fail[m].sum() / max(fail.sum(), 1)), 3),
                         "median_log10_freq": round(float(np.median(x[m])), 2), **oe(fail, p, m),
                         "failing_descriptions": descr})
        res[fld] = sorted(rows, key=lambda r: -r["never_activated"])
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"n={res['n']}  never activated {res['never_activated']:.3f}  fail {res['fail']:.3f}")
    for fld in FIELDS:
        print(f"\n{fld:24s} {'n':>5s} {'never':>6s} {'fail':>6s} {'O/E':>5s} {'z':>6s} {'%fails':>7s}")
        for r in res[fld]:
            print(f"{r['value']:24s} {r['n']:5d} {100*r['never_activated']:5.0f}% {100*r['fail']:5.0f}% "
                  f"{(r['O/E'] or float('nan')):5.2f} {r['z']:6.2f} {100*r['share_of_all_failures']:6.0f}%")
    print("wrote", a.out)


def run_512(a):
    from results.common import bo_unbiased, peaks_of
    M = pathlib.Path(a.mirror)
    s = open(M / "sae_self.json").read()
    meta = json.loads(s[: s.rfind("}") + 1])
    gate = float(meta["gate"])
    act = np.fromfile(M / "sae_self.f16", dtype=np.float16).astype(np.float32)
    act = act.reshape(len(meta["features"]), int(meta["n"]), -1)
    mp = {int(f): peaks_of(act[i]) for i, f in enumerate(meta["features"])}
    la = collections.defaultdict(list)
    for r in map(json.loads, open(M / "llm_c16_claude-sonnet-5.scored.jsonl")):
        la[int(r["feature"])].append(float(r["act"]))
    lab = load_labels(a.labels)
    mech = {int(r["feature"]): r for r in map(json.loads, open(a.mechanics))} if a.mechanics else {}
    feats = sorted(f for f in mp if len(la.get(f, [])) >= 8 and f in lab)
    feat = np.array(feats)
    m_f = np.array([bo_unbiased((mp[f] > gate).astype(float), 8) for f in feats]) < 0.5
    l_f = np.array([bo_unbiased((np.array(la[f]) > gate).astype(float), 8) for f in feats]) < 0.5
    groups = {}
    for fld in FIELDS:
        groups[fld] = np.array([str(lab[f].get(fld, "?")) for f in feats])
    if mech:
        def tagset(f):
            m = mech.get(f)
            if not m:
                return "?"
            for k, n in (("ws", "whitespace"), ("digit", "digit"), ("ind_trigram", "induction"),
                         ("punct", "punct"), ("xdoc_ctx", "boilerplate")):
                if m[k] >= 0.5:
                    return n
            return "none"
        groups["mechanism"] = np.array([tagset(f) for f in feats])
    res = {"n": len(feats), "maemm_fail": round(float(m_f.mean()), 4), "llm_fail": round(float(l_f.mean()), 4)}
    print(f"n={len(feats)}  bo8 fired<0.5: MAEMM {m_f.mean():.3f}  LLM {l_f.mean():.3f}")
    for g, vals in groups.items():
        rows = []
        for v in sorted(set(vals.tolist())):
            m = vals == v
            rows.append({"value": v, "n": int(m.sum()), "maemm_fail": round(float(m_f[m].mean()), 3),
                         "llm_fail": round(float(l_f[m].mean()), 3),
                         "maemm_only": int((m_f & ~l_f & m).sum()), "llm_only": int((~m_f & l_f & m).sum()),
                         "both": int((m_f & l_f & m).sum())})
        res[g] = sorted(rows, key=lambda r: -r["maemm_fail"])
        print(f"\n{g:22s} {'n':>4s} {'MAEMM':>6s} {'LLM':>6s} {'M-only':>6s} {'L-only':>6s} {'both':>5s}")
        for r in res[g]:
            print(f"{r['value']:22s} {r['n']:4d} {100*r['maemm_fail']:5.0f}% {100*r['llm_fail']:5.0f}% "
                  f"{r['maemm_only']:6d} {r['llm_only']:6d} {r['both']:5d}")
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["set", "p512"])
    ap.add_argument("--perdir")
    ap.add_argument("--mirror")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mechanics", default="")
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_set(a) if a.mode == "set" else run_512(a)


if __name__ == "__main__":
    main()
