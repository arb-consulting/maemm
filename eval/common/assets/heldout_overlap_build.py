"""Build `heldout_overlap.csv` and its meta file: the word n-gram coverage of every held-out document by the
checkpoint's training text (the measure: `assets/README.md`). Provenance only: the training text is not
public, so nothing imports this.

Input: the `target_text` column of the `SOURCES` parquet files under `--train-dir`, a local copy of the
training bundle `--bundle`; the held-out documents are rebuilt from the public files. The held-out n-grams
are hashed once and the training text is streamed past them.

    python eval/common/assets/heldout_overlap_build.py --bundle <name> --train-dir <dir> [--out <dir>]
"""

import argparse
import hashlib
import json
import os
import unicodedata

import numpy as np

from eval.common import retrieval as R

SOURCES = ("sft_mix_realact", "rl_pool_realact_ctx64_2048", "sft_mix_sae2m", "sft_mix_sae2m_dec",
           "rl_pool_sae2m", "rl_pool_sae2m_dec")
TEXT_COLUMN = "target_text"
NS = (7, 13)
LEVELS = (0.05, 0.2, 0.5, 0.9)
NORMALISATION = "NFKC, lowercase, whitespace split"
MEASURE = "the share of a document's word n-grams that occur anywhere in the training text"
HASH = "polynomial over word ids, base 1000003 modulo 2^61 - 1"
P = np.uint64((1 << 61) - 1)
B = np.uint64(1_000_003)
MASK29, MASK32 = np.uint64((1 << 29) - 1), np.uint64((1 << 32) - 1)


def words(text):
    return unicodedata.normalize("NFKC", text).lower().split()


def mulmod(h):
    """`h * B mod (2^61 - 1)` for uint64 `h < 2^61` and `B < 2^20`, without overflow (2^61 = 1 mod p)."""
    hi, lo = h >> np.uint64(32), h & MASK32
    t = hi * B                                                   # < 2^49
    part = (t >> np.uint64(29)) + ((t & MASK29) << np.uint64(32))
    return (part + lo * B) % P


def ngram_hashes(ids, starts_ok, n):
    """The Horner hash of every n-word window of the flat ids that stays inside one document (`starts_ok`)."""
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, np.uint64)
    h = np.zeros(m, np.uint64)
    for k in range(n):
        h = (mulmod(h) + ids[k:k + m]) % P
    return h[starts_ok[:m]]


def flat(doc_ids, n):
    """(flat ids, which window starts keep `n` words inside one document, the documents' lengths)."""
    lens = np.fromiter((len(d) for d in doc_ids), np.int64, len(doc_ids))
    ids = np.concatenate([np.asarray(d, np.uint64) for d in doc_ids]) if len(doc_ids) else np.zeros(0, np.uint64)
    left = np.repeat(np.cumsum(lens), lens) - np.arange(len(ids))   # words left in the document from here
    return ids, left >= n, lens


def cell(value):
    """A coverage as the csv holds it: six decimals, trailing zeros dropped, never an exponent."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main():
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True,
                    help="name or path of the training-data bundle, recorded in the metadata")
    ap.add_argument("--train-dir", required=True,
                    help="a local copy of the bundle, holding <source>/records.parquet")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    index = R.load_index()
    parts = R.open_parts()
    evaluation = {d.doc for d in R.evaluation_docs(index)}

    vocab, held = {}, []
    for d in index:
        held.append([vocab.setdefault(w, len(vocab) + 1) for w in words(parts[int(d.part)][int(d.row)])])
    print(f"{len(index):,} held-out documents, {len(vocab):,} distinct words", flush=True)

    targets = {}
    for n in NS:
        ids, ok, lens = flat(held, n)
        h = ngram_hashes(ids, ok, n)
        uniq = np.unique(h)
        targets[n] = {"hashes": h, "per_doc": np.maximum(lens - n + 1, 0), "uniq": uniq,
                      "hit": np.zeros(len(uniq), bool)}
        print(f"n = {n}: {len(h):,} held-out n-grams, {len(uniq):,} distinct", flush=True)

    rows_by_source, get = {}, vocab.get
    for name in SOURCES:
        file = pq.ParquetFile(os.path.join(args.train_dir, name, "records.parquet"))
        rows = 0
        for group in range(file.num_row_groups):
            texts = file.read_row_group(group, columns=[TEXT_COLUMN]).column(TEXT_COLUMN).to_pylist()
            # a word the held-out text never uses maps to 0, which no held-out n-gram contains
            docs = [[get(w, 0) for w in words(t or "")] for t in texts]
            rows += len(docs)
            for n in NS:
                ids, ok, _lens = flat(docs, n)
                h = np.unique(ngram_hashes(ids, ok, n))
                uniq = targets[n]["uniq"]
                at = np.searchsorted(uniq, h)
                at[at >= len(uniq)] = 0
                targets[n]["hit"][at[uniq[at] == h]] = True
            print(f"{name}: row group {group + 1}/{file.num_row_groups}, {rows:,} rows", flush=True)
        rows_by_source[name] = rows

    half = np.array(["evaluation" if d.doc in evaluation else "search" for d in index])
    coverage, summary = {}, {}
    for n in NS:
        t = targets[n]
        hit = t["hit"][np.searchsorted(t["uniq"], t["hashes"])]
        bounds = np.concatenate([[0], np.cumsum(t["per_doc"])])
        if (t["per_doc"] == 0).any():
            raise ValueError(f"a held-out document holds fewer than {n} words and has no coverage at n = {n}")
        cov = np.array([hit[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
        shared = np.array([hit[a:b].any() for a, b in zip(bounds[:-1], bounds[1:])])
        coverage[n] = cov
        summary[f"n{n}"] = {}
        for name in ("evaluation", "search", "all"):
            sel = np.ones(len(index), bool) if name == "all" else half == name
            c = cov[sel]
            summary[f"n{n}"][name] = {
                "docs": int(sel.sum()), "mean": round(float(c.mean()), 6),
                "p50": round(float(np.percentile(c, 50)), 6), "p90": round(float(np.percentile(c, 90)), 6),
                "p99": round(float(np.percentile(c, 99)), 6), "max": round(float(c.max()), 6),
                "any_shared": int(shared[sel].sum()),
                "at_or_above": {str(x): int((c >= x).sum()) for x in LEVELS}}

    lines = ["doc,coverage_n7,coverage_n13"]
    lines += [f"{int(d.doc)},{cell(coverage[7][i])},{cell(coverage[13][i])}" for i, d in enumerate(index)]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    with open(os.path.join(args.out, "heldout_overlap.csv"), "wb") as h:
        h.write(raw)
    # the loader reads the rounded csv, so the excluded count is taken from it too
    shipped = np.array([float(cell(v)) for v in coverage[7]])
    threshold = R.OVERLAP_THRESHOLD
    meta = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_docs": len(index),
        "index_sha256": R._index_sha256(),
        "measure": MEASURE,
        "n": list(NS),
        "normalisation": NORMALISATION,
        "hash": HASH,
        "threshold": {"n": 7, "coverage": threshold, "rule": "coverage_n7 >= coverage is excluded from a draw"},
        "n_excluded": {name: int(((shipped >= threshold) & (np.ones(len(index), bool) if name == "all"
                                                            else half == name)).sum())
                       for name in ("evaluation", "search", "all")},
        "training_text": {"bundle": args.bundle, "column": TEXT_COLUMN, "rows": int(sum(rows_by_source.values())),
                          "sources": rows_by_source},
        "summary": summary,
    }
    with open(os.path.join(args.out, "heldout_overlap.meta.json"), "w", encoding="utf-8") as h:
        json.dump(meta, h, indent=1)
        h.write("\n")
    print(json.dumps(meta["n_excluded"]), flush=True)


if __name__ == "__main__":
    main()
