"""Structural token events a feature fires on, and which ones the MAEMM fails beyond rarity.

    python verbalization/analysis/structural_tags.py --perdir <perdir json> \
        --mechanics <mechanics jsonl with per-peak context> --sae-match <npz> --out <json>

Each deduplicated top peak (modal_27b_section.token_mechanics: the peak token, the 6 tokens before
it, the 2 after) is tagged; a feature carries a tag when >= half its peaks do.

    word_continuation   the peak is a word piece CONTINUING the previous token (no leading space,
                        the previous token ends in a letter): "Roeth-Lis«berger»", "photo«synthesis»"
    name_completion     word_continuation where the word being completed is capitalised
    hyphen_compound     the previous token ends in "-" and the peak is alphabetic
    open_bracket / close_bracket / quote
    sentence_start      capitalised word right after . ! ?
    line_start          the previous token contains a newline
    acronym             the peak is 2+ capitals
    url_email_path      http / www / .com / @ / slashed path in the local context
    code_markup         < > { } = ; \\ or `_` in the local context
    non_latin           non-ASCII letters in the peak
    whitespace / punct / digit / induction (3-gram repeat) / boilerplate (cross-doc context), from
    the same product
Failure = norm_act < 0.10; O/E against rarity deciles as in mechanism_groups.py.
"""
import argparse
import json
import math
import re

import numpy as np

BAR = 0.10
OPEN, CLOSE, QUOTE = set("([{<«"), set(")]}>»"), set("\"'“”‘’`")


def word_back(prev, tok):
    """The word the peak belongs to: the peak plus the trailing previous pieces with no space."""
    w = tok
    for p in reversed(prev):
        if not p or p[-1].isspace():
            break
        w = p + w
        if p[0].isspace():
            break
    return w.strip()


def peak_tags(pk):
    t, prev, nxt = pk["tok"], pk["prev"], pk["next"]
    ts = t.strip()
    p1 = prev[-1] if prev else ""
    local = "".join(prev[-4:]) + t + "".join(nxt)
    cont = bool(ts) and not t[0].isspace() and bool(re.search(r"[A-Za-z]", ts)) and bool(p1) \
        and bool(re.search(r"[A-Za-z]$", p1))
    word = word_back(prev, t)
    prev_ns = "".join(prev).rstrip()
    return {
        "word_continuation": cont,
        "name_completion": cont and bool(re.match(r"[A-Z]", word)),
        "hyphen_compound": p1.endswith("-") and bool(re.match(r"[A-Za-z]", ts)),
        "open_bracket": bool(ts) and ts[0] in OPEN,
        "close_bracket": bool(ts) and ts[-1] in CLOSE,
        "quote": bool(ts) and any(c in QUOTE for c in ts),
        "sentence_start": bool(re.search(r"[.!?]$", prev_ns)) and bool(re.match(r"\s*[A-Z]", t)),
        "line_start": "\n" in p1,
        "acronym": bool(re.fullmatch(r"[A-Z]{2,}", ts)),
        "url_email_path": bool(re.search(r"https?:|www\.|\.com\b|\.org\b|@\w|\w/\w", local)),
        "code_markup": bool(re.search(r"[<>{}=;\\]|_\w", local)),
        "non_latin": bool(re.search(r"[^\x00-\x7F]", ts)) and bool(re.search(r"\w", ts)),
    }


def oe(fail, p, m):
    o, e = int(fail[m].sum()), float(p[m].sum())
    var = float((p[m] * (1 - p[m])).sum())
    z = (o - e) / math.sqrt(var) if var > 0 else 0.0
    return {"n": int(m.sum()), "fail_rate": round(float(fail[m].mean()), 3) if m.any() else None,
            "fail_obs": o, "fail_exp": round(e, 1), "O/E": round(o / e, 2) if e > 0 else None,
            "z": round(z, 2), "p": float(f"{math.erfc(abs(z) / math.sqrt(2)):.3g}")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", required=True)
    ap.add_argument("--mechanics", required=True)
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--examples", default="", help="examples jsonl, for a marked window per example")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.load(open(a.perdir))["perdir"]["sae"]
    mech = {int(r["feature"]): r for r in map(json.loads, open(a.mechanics))}
    idx = [i for i, f in enumerate(d["feature"]) if int(f) in mech]
    feat = np.asarray(d["feature"], int)[idx]
    norm = np.asarray(d["norm_act"], float)[idx]
    fail = norm < BAR
    z = np.load(a.sae_match)
    x = np.log10(np.maximum(z["sae_nfire"][feat], 1) / float(z["n_tok"]))
    edges = np.quantile(x, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    rate = np.array([fail[dec == b].mean() for b in range(10)])
    p = rate[dec]

    tagfrac = {}
    for f in feat:
        pk = mech[int(f)]["peaks"]
        T = [peak_tags(q) for q in pk]
        fr = {k: sum(t[k] for t in T) / len(T) for k in T[0]}
        m = mech[int(f)]
        fr.update({"whitespace": m["ws"], "punct": m["punct"], "digit": m["digit"],
                   "induction": m["ind_trigram"], "boilerplate": m["xdoc_ctx"]})
        tagfrac[int(f)] = fr
    names = list(next(iter(tagfrac.values())))
    tags = {n: np.array([tagfrac[int(f)][n] >= 0.5 for f in feat]) for n in names}
    tags["any_structural"] = np.any(np.stack([tags[n] for n in names]), axis=0)
    tags["none"] = ~tags["any_structural"]

    ex = {}
    if a.examples:
        for r in map(json.loads, open(a.examples)):
            ex[int(r["feature"])] = (r.get("marked") or [""])[0]

    def show(f):
        w = ex.get(int(f), ""); i = w.find("«")
        if i >= 0:
            return w[max(0, i - 70): i + 25].replace("\n", "⏎")
        q = mech[int(f)]["peaks"][0]
        return ("".join(q["prev"]) + "«" + q["tok"] + "»" + "".join(q["next"])).replace("\n", "⏎")

    res = {"perdir": a.perdir, "n": int(len(feat)), "fail_rate": round(float(fail.mean()), 4), "tags": {}}
    for n, m in tags.items():
        r = oe(fail, p, m)
        r["median_log10_freq"] = round(float(np.median(x[m])), 2) if m.any() else None
        r["fail_examples"] = [{"feature": int(f), "norm_act": round(float(v), 3), "window": show(f)}
                              for f, v in zip(feat[m & fail][:4], norm[m & fail][:4])]
        r["pass_examples"] = [{"feature": int(f), "norm_act": round(float(v), 3), "window": show(f)}
                              for f, v in zip(feat[m & ~fail][:3], norm[m & ~fail][:3])]
        res["tags"][n] = r
    json.dump(res, open(a.out, "w"), indent=1)

    print(f"n={res['n']} fail {res['fail_rate']:.3f}")
    print(f"{'tag':18s} {'n':>5s} {'fail':>6s} {'O/E':>5s} {'z':>6s} {'p':>8s}")
    for n, r in sorted(res["tags"].items(), key=lambda kv: -kv[1]["z"]):
        print(f"{n:18s} {r['n']:5d} {r['fail_rate'] if r['fail_rate'] is not None else float('nan'):6.3f} "
              f"{(r['O/E'] or float('nan')):5.2f} {r['z']:6.2f} {r['p']:8.2g}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
