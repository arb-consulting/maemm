"""Which SAE features does the inverter fail on, and what KIND of thing are they?

fig6/fig8 say failure tracks corpus rarity. They do not say what the failures look like. This walks
every feature the inverter cannot activate and reduces its max-activating examples to a row you can
sort and group -- what token it peaks on, whether its examples are one boilerplate string repeated,
whether the token after the peak is always the same.

Three flags come out of that, and they are not the same failure:

  TEMPLATE     high template_J -- the 30 "examples" are near-duplicates of one string. There is no
               generalisable concept to verbalize; the feature memorised a fragment of boilerplate.
  COLLOCATION  the token AFTER the peak is near-constant. The feature does not encode a topic, it
               encodes a continuation ("Chamber of" -> " Commerce"). Asking a model to write text
               that "means" this is close to ill-posed.
  UNRESOLVED   neither flag fires and the inverter still gets nothing.

Selection: a feature counts as unactivatable when NO supplied arm reaches --max-norm-act of its own
corpus peak. Passing several arms is therefore the conservative reading -- a feature any arm can
activate is dropped.

    python evals/verbalization/analysis/find_bad_features.py \
        --perdir rl=evals/verbalization/report/data/perdir_8b_rl.json \
        --perdir sft=evals/verbalization/report/data/perdir_8b_sft.json \
        --sae-match verbalization/report/data/sae_match_8b.npz \
        --out evals/verbalization/report

Writes tables/unactivatable_features.csv, tables/collocation_features.txt and
data/bad_feature_summary.json.
"""
import argparse
import collections
import csv
import json
import os

import dumplib as D
import featlib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perdir", action="append", required=True, metavar="[TAG=]PATH",
                    help="repeatable: eval_dirs per-feature dump")
    ap.add_argument("--sae-match", required=True, help="sae_match_8b.npz from scan_fire")
    ap.add_argument("--maxacts", default=None, help="local max-acts .pt (default: download from HF)")
    ap.add_argument("--max-norm-act", type=float, default=0.10,
                    help="unactivatable if best_act/corpus_peak is under this in EVERY arm")
    ap.add_argument("--template-j", type=float, default=0.25, help="TEMPLATE flag threshold")
    ap.add_argument("--next-consistency", type=float, default=0.90, help="COLLOCATION flag threshold")
    ap.add_argument("--out", required=True, help="report dir; tables/ and data/ are written under it")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    os.makedirs(f"{a.out}/data", exist_ok=True)

    arms = {tag: p.rows for tag, p in D.load_perdir(a.perdir).items()}
    scan = D.Scan(a.sae_match)
    fire, n_tok = scan.fire_pct, scan.n_tok
    print(f"[bad] arms {list(arms)} | corpus scan {n_tok:,} tokens", flush=True)

    feats = sorted(set().union(*[set(v) for v in arms.values()]))
    # best_act from the most favourable arm: a feature only counts as unactivatable if nothing
    # reached it. corpus_peak is a property of the SAE, so any arm carrying the feature agrees.
    best, peak = {}, {}
    for f in feats:
        rows = [v[f] for v in arms.values() if f in v]
        best[f] = max(r["best_act"] for r in rows)
        peak[f] = rows[0]["corpus_peak"]
    sel = [f for f in feats if peak[f] > 0 and best[f] / peak[f] < a.max_norm_act]
    print(f"[bad] {len(sel)} of {len(feats)} features under norm_act {a.max_norm_act}", flush=True)

    MT, MA = L.load_maxacts(a.maxacts)
    tok = L.load_tokenizer()

    recs = []
    for i, f in enumerate(sel):
        d = L.feature_diagnostics(f, MT, MA, tok)
        d["fire_pct"] = round(float(fire[f]), 5)
        d["maem_best"] = round(float(best[f]), 2)
        d["flag"] = ("TEMPLATE" if d["template_J"] >= a.template_j else
                     "COLLOCATION" if d["next_consistency"] >= a.next_consistency else "UNRESOLVED")
        recs.append(d)
        if (i + 1) % 200 == 0:
            print(f"[bad] {i + 1}/{len(sel)}", flush=True)
    recs.sort(key=lambda r: -r["corpus_peak"])

    cols = ["feature", "fire_pct", "corpus_peak", "maem_best", "template_J", "uniq_of",
            "modal_peak_token", "modal_frac", "token_class", "distinct_peak_tokens",
            "mean_peak_pos", "flag", "top_example"]
    p = f"{a.out}/tables/unactivatable_features.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print(f"[bad] -> {p}", flush=True)

    # ---- collocation view: sorted by feature id, as the original table was -------------------
    coll = [r for r in recs if r["flag"] == "COLLOCATION"]
    coll.sort(key=lambda r: -r["feature"])
    p2 = f"{a.out}/tables/collocation_features.txt"
    with open(p2, "w") as fh:
        fh.write("COLLOCATION / COMPLETION FEATURES the MAEM fails on\n"
                 "(detector: the token AFTER the peak is the same across examples -> the feature "
                 "encodes a continuation)\n\n")
        fh.write("%-8s %7s %8s %7s  %-12s %-13s %s\n"
                 % ("feat", "peak", "fire%", "maem", "fires on", "next token", "typical span"))
        for r in coll:
            fh.write("%-8d %7.1f %7.4f%% %7.2f  %-12s %-13s %s  [%d%%]\n"
                     % (r["feature"], r["corpus_peak"], r["fire_pct"], r["maem_best"],
                        repr(r["modal_peak_token"]), repr(r["next_token"]), repr(r["typical_span"]),
                        round(r["next_consistency"] * 100)))
    print(f"[bad] -> {p2}  ({len(coll)} features)", flush=True)

    summary = {
        "n_features_scanned": len(feats), "n_unactivatable": len(sel),
        "max_norm_act": a.max_norm_act, "arms": list(arms),
        "by_flag": dict(collections.Counter(r["flag"] for r in recs)),
        "by_token_class": dict(collections.Counter(r["token_class"] for r in recs)),
        "median_fire_pct": sorted(r["fire_pct"] for r in recs)[len(recs) // 2] if recs else None,
    }
    D.write_data(a.out, "bad_feature_summary", summary)
    print("[bad] " + json.dumps(summary["by_flag"]) + "  " + json.dumps(summary["by_token_class"]))


if __name__ == "__main__":
    main()
