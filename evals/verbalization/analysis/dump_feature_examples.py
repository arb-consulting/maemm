"""Dump what a feature actually fires on, next to what the inverter said about it.

find_bad_features.py reduces a feature to one row. This is the other half: the evidence, in a form a
human can read and judge. Two layouts, same selection machinery.

  --format csv   one row per example: the feature's real corpus windows AND the inverter's
                 generations for it, interleaved, one arm after another. This is the table you skim
                 to decide whether a failure is the model's fault or the feature's.
  --format txt   corpus windows only, with >>> marking the peak token, grouped per feature. For
                 reading a band of features end to end (e.g. the rarest 2%) without a spreadsheet.

Feature selection is one of --features / --from-csv / --rarest-pct.

    # the rarest 2% of features, corpus evidence only
    python evals/verbalization/analysis/dump_feature_examples.py --rarest-pct 2 --format txt \\
        --sae-match verbalization/report/data/sae_match_8b.npz \\
        --out evals/verbalization/report/tables/rare_examples.txt

    # 20 unactivatable features, corpus + all three arms' generations
    python evals/verbalization/analysis/dump_feature_examples.py --format csv \\
        --from-csv evals/verbalization/report/tables/unactivatable_features.csv --limit 20 \\
        --sae-match verbalization/report/data/sae_match_8b.npz \\
        --perdir rl=evals/verbalization/report/data/perdir_8b_rl.json \\
        --texts rl=evals/verbalization/report/data/texts_8b_rl.json \\
        --out evals/verbalization/report/tables/unactivatable_examples.csv
"""
import argparse
import csv
import json
import os

import numpy as np

import dumplib as D
import featlib as L


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--features", help="JSON list of feature ids, or a path to one")
    g.add_argument("--from-csv", help="take features from a find_bad_features.py table")
    g.add_argument("--rarest-pct", type=float, help="bottom N%% of ALL features by firing rate")
    ap.add_argument("--flag", default=None, help="with --from-csv: keep only this flag (e.g. TEMPLATE)")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of features dumped")
    ap.add_argument("--sae-match", required=True)
    ap.add_argument("--maxacts", default=None)
    ap.add_argument("--perdir", action="append", default=[], metavar="[TAG=]PATH")
    ap.add_argument("--texts", action="append", default=[], metavar="[TAG=]PATH")
    ap.add_argument("--n-corpus", type=int, default=6, help="corpus examples per feature")
    ap.add_argument("--n-gen", type=int, default=4, help="generations per arm per feature")
    ap.add_argument("--format", choices=["csv", "txt"], default="csv")
    ap.add_argument("--out", required=True, help="output FILE")
    a = ap.parse_args()

    scan = D.Scan(a.sae_match)
    fire, n_tok = scan.fire_pct, scan.n_tok
    arms = {tag: p.rows for tag, p in D.load_perdir(a.perdir).items()} if a.perdir else {}
    gens = D.load_texts(a.texts) if a.texts else {}

    header = None
    if a.features:
        feats = json.load(open(a.features)) if os.path.exists(a.features) else json.loads(a.features)
        feats = [int(f) for f in feats]
    elif a.from_csv:
        rows = list(csv.DictReader(open(a.from_csv)))
        if a.flag:
            rows = [r for r in rows if r.get("flag") == a.flag]
        feats = [int(r["feature"]) for r in rows]
    else:
        cut = np.percentile(fire, a.rarest_pct)
        feats = sorted(np.where(fire <= cut)[0].tolist(), key=lambda f: fire[f])
        header = (f"MAX-ACTIVATING EXAMPLES - BOTTOM {a.rarest_pct:g}% OF FEATURES BY CORPUS FIRING RATE\n"
                  f"{len(feats)} features fire on <= {cut:.5f}% of tokens "
                  f"(<= {round(cut / 100 * n_tok)} hits per {n_tok / 1e6:.2f}M)\n")
    if a.limit:
        feats = feats[:a.limit]
    print(f"[dump] {len(feats)} features | arms {list(arms)} | text arms {list(gens)}", flush=True)

    MT, MA = L.load_maxacts(a.maxacts)
    tok = L.load_tokenizer()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)

    if a.format == "txt":
        with open(a.out, "w") as fh:
            fh.write(header or "MAX-ACTIVATING EXAMPLES\n")
            fh.write('Each example is a 32-token window; ">>>" marks the peak-activating token.\n')
            for f in feats:
                A, T = MA[f], MT[f]
                live = sorted([i for i in range(A.shape[0]) if float(A[i].max()) > 0],
                              key=lambda i: -float(A[i].max()))[:a.n_corpus]
                fh.write("\n" + "=" * 100 + "\n")
                fh.write(f"FEATURE {f}   fires {fire[f]:.5f}% "
                         f"({round(fire[f] / 100 * n_tok)} of {n_tok / 1e6:.2f}M tokens)   "
                         f"corpus peak {float(A.max()):.1f}\n")
                fh.write("=" * 100 + "\n")
                for rank, i in enumerate(live, 1):
                    p = int(A[i].argmax())
                    ids = [int(x) for x in T[i]]
                    fh.write(f"  [{rank}] act {float(A[i, p]):.1f}\n")
                    fh.write(f"      {tok.decode(ids[:p])!r}\n")
                    fh.write(f"      >>> {tok.decode([ids[p]])!r} <<<\n")
                    fh.write(f"      {tok.decode(ids[p + 1:])!r}\n")
        print(f"[dump] -> {a.out}", flush=True)
        return

    cols = ["feature", "fire_pct", "corpus_peak", "maemm_best", "source", "arm", "idx",
            "activation", "peak_token", "text"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for f in feats:
            A, T = MA[f], MT[f]
            peak = float(A.max())
            best = max((v[f]["best_act"] for v in arms.values() if f in v), default="")
            base = {"feature": f, "fire_pct": round(float(fire[f]), 5),
                    "corpus_peak": round(peak, 1),
                    "maemm_best": round(float(best), 2) if best != "" else ""}
            live = sorted([i for i in range(A.shape[0]) if float(A[i].max()) > 0],
                          key=lambda i: -float(A[i].max()))[:a.n_corpus]
            for rank, i in enumerate(live, 1):
                p = int(A[i].argmax())
                w.writerow({**base, "source": "CORPUS", "arm": "-", "idx": rank,
                            "activation": round(float(A[i, p]), 1),
                            "peak_token": tok.decode([int(T[i, p])]),
                            "text": L.decode_window(tok, T[i])})
            for tag, by in gens.items():
                # only the best-of-n sample has a recorded activation (eval_dirs keeps best_act,
                # not per-sample acts), so it goes on idx 0 and the rest are left blank.
                for j, t in enumerate(by.get(f, [])[:a.n_gen]):
                    w.writerow({**base, "source": "MAEMM", "arm": tag, "idx": j,
                                "activation": (arms[tag][f]["best_act"]
                                               if j == 0 and tag in arms and f in arms[tag] else ""),
                                "peak_token": "", "text": t.replace("\n", " ").strip()})
    print(f"[dump] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
