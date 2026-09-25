"""When a structural feature DOES activate, does the MAEM reproduce the structural event, or reach
the feature some other way?

    python evals/verbalization/analysis/how_structural_pass.py

For every feature that passes (some own rollout clears the SAE gate; dead excluded, criterion.py)
in each structural category, take the MAEM's
best rollout token (argmax of `sae_self` over its 8 rollouts) and compare it with the feature's
corpus peaks (modal_27b_section.token_mechanics):

    same_token   the MAEM peak token equals one of the corpus peak tokens (whitespace = whitespace)
    same_class   same token class as the corpus peaks' majority (whitespace / digit / punct / word)
    class        the MAEM peak's class when it differs
    repeat       (induction) the 3-gram ending at the MAEM peak occurs earlier in its own rollout,
                 i.e. the MAEM reproduced the copy, not only the token

Controls: name completions and untagged features, for the base rate of an exact token match.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from criterion import load_perdir  # noqa: E402
from structural_tags import peak_tags  # noqa: E402

D = Path("evals/verbalization/report/data")
SETS = [("2k", "perdir_27b_rl-final.json", "2k"), ("rwtest2k", "perdir_27b_rl-final_rwtest2k.json", "rw")]
CATS = ["whitespace", "digit", "punct", "induction", "boilerplate", "line_start", "name_completion", "none"]


def cls(s):
    t = s.strip()
    if not t:
        return "whitespace"
    if re.fullmatch(r"\d+", t):
        return "digit"
    if not re.search(r"\w", t):
        return "punct"
    return "word"


def main(scratch):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    out = {}
    for name, pd, tag in SETS:
        j = json.loads(open(f"{scratch}/{tag}_sae_self.json").read())
        N = len(j["features"])
        A = np.fromfile(f"{scratch}/{tag}_sae_self.f16", np.float16).astype(np.float32).reshape(N, 8, -1)
        I = np.fromfile(f"{scratch}/{tag}_sae_self_ids.i32", np.int32).reshape(N, 8, -1)
        fidx = {int(f): i for i, f in enumerate(j["features"])}
        d = load_perdir(D / pd)
        norm = dict(zip(map(int, d["feature"]), d["norm_act"]))
        passes = dict(zip(map(int, d["feature"]), ~d["fail"]))
        mech = {int(json.loads(l)["feature"]): json.loads(l) for l in open(D / f"mechanics_27b_{name}.jsonl")}
        tagsets = {}
        for f, m in mech.items():
            T = [peak_tags(q) for q in m["peaks"]]
            fr = {k: sum(t[k] for t in T) / len(T) for k in T[0]}
            fr.update(whitespace=m["ws"], punct=m["punct"], digit=m["digit"],
                      induction=m["ind_trigram"], boilerplate=m["xdoc_ctx"])
            tagsets[f] = {k for k, v in fr.items() if v >= 0.5}
        res = {}
        for c in CATS:
            fs = [f for f in mech if f in passes and (not tagsets[f] if c == "none" else c in tagsets[f])]
            passing = [f for f in fs if passes[f]]
            rows = []
            for f in passing:
                i = fidx[f]
                a = np.nan_to_num(A[i], nan=-1.0)
                r, p = np.unravel_index(a.argmax(), a.shape)
                ids = [int(x) for x in I[i, r]]
                mt = tok.decode([ids[p]])
                corp = [q["tok"] for q in mech[f]["peaks"]]
                ctoks = {x.strip() for x in corp}
                maj = collections.Counter(cls(x) for x in corp).most_common(1)[0][0]
                prev = [x for x in ids[1:p] if x >= 0]
                tri = prev[-2:] + [ids[p]]
                rep = len(tri) == 3 and any(prev[k:k + 3] == tri for k in range(len(prev) - 2))
                rows.append({"feature": f, "norm_act": round(float(norm[f]), 3), "maem_tok": mt,
                             "corpus_toks": corp[:4], "same_token": mt.strip() in ctoks,
                             "same_class": cls(mt) == maj, "maem_class": cls(mt), "corpus_class": maj,
                             "repeat": rep,
                             "maem_ctx": (tok.decode([x for x in ids[max(1, p - 10):p] if x >= 0]) + "«" + mt + "»"
                                           + tok.decode([x for x in ids[p + 1:p + 4] if x >= 0])).replace("\n", "⏎")})
            n = len(rows)
            res[c] = {"n_features": len(fs), "n_pass": n,
                      "same_token": round(np.mean([r["same_token"] for r in rows]), 3) if n else None,
                      "same_class": round(np.mean([r["same_class"] for r in rows]), 3) if n else None,
                      "maem_class_when_different": dict(collections.Counter(
                          r["maem_class"] for r in rows if not r["same_class"])),
                      "repeat_in_rollout": round(np.mean([r["repeat"] for r in rows]), 3) if n else None,
                      "rows": rows}
        out[name] = res
    json.dump(out, open(D / "how_structural_pass.json", "w"), indent=1)
    for name, res in out.items():
        print(f"\n######## {name}")
        print(f"{'category':16s} {'pass/n':>9s} {'same tok':>8s} {'same cls':>8s} {'repeat':>7s}  class used instead")
        for c, r in res.items():
            if not r["n_pass"]:
                print(f"{c:16s} {r['n_pass']:>4d}/{r['n_features']:<4d}"); continue
            print(f"{c:16s} {r['n_pass']:>4d}/{r['n_features']:<4d} {r['same_token']:8.2f} {r['same_class']:8.2f} "
                  f"{r['repeat_in_rollout']:7.2f}  {r['maem_class_when_different']}")
    print("wrote", D / "how_structural_pass.json")


if __name__ == "__main__":
    main(sys.argv[1])
