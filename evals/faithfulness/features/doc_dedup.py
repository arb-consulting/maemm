"""Document-level near-duplication: did the model see the DOCUMENT, not just the span?

    modal run features/doc_dedup.py --n 7

The span-level 7-gram check (features/ngram_overlap.py) asks whether an eval target's
own text appears in training. It cannot see the case that actually matters most: an
activation read at position 300 of a document is contaminated if the model trained on
positions 0-100 of the SAME document, even though no n-gram of the shown span matches.
Ultra-FineWeb is not deduplicated, so disjoint document INDICES do not rule this out.

## Recovering the source documents

The bundle ships the eval realacts' provenance only as a range -- `doc_registry.json`
gives `eval_head` = [0, 100,000) -- and `pool_seq` is an index into the upstream eval pool, not a
corpus document id. So there is no mapping to join on, and the source document has to be
recovered by matching the shown span back into the corpus head. That match is exact
(normalised substring), so a target either resolves to one document or is reported
unresolved; nothing is guessed.

## What is compared

For each resolved source document: the share of ITS word n-grams that occur anywhere in
the checkpoint's training text (all 8.94M rows of the simple2m SFT mix and RL pool).
Same normalisation and hashing as the span check, so the two numbers are comparable and
the difference between them is exactly the contamination the span check misses.
"""
from __future__ import annotations

import modal

VOL = "/vol"
BUNDLE = f"{VOL}/data/v2-bundle"
DATASET = "openbmb/Ultra-FineWeb"
PART1 = "data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet"
EVAL_HEAD = 100_000          # doc_registry.json: the upstream eval realacts come from [0, this)

TRAIN_SOURCES = [
    "simple2m/sft_mix/realact/records.parquet",
    "simple2m/rl_pool/realact_ctx64_2048/records.parquet",
    "simple2m/sft_mix/sae2m/records.parquet",
    "simple2m/sft_mix/sae2m_dec/records.parquet",
    "simple2m/rl_pool/sae2m/records.parquet",
    "simple2m/rl_pool/sae2m_dec/records.parquet",
]
TEXT_COLS = ("target_text", "pool_target_text", "text")

vol = modal.Volume.from_name("maemm", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.3.4", "pandas==3.0.3", "pyarrow==24.0.0",
                 "huggingface_hub==0.36.0")
    .env({"HF_HOME": f"{VOL}/hf"})
)
app = modal.App("maemm-doc-dedup")


def _norm(s: str) -> list[str]:
    import unicodedata

    return unicodedata.normalize("NFKC", s).lower().split()


def _shingles(words: list[str], n: int, vocab: dict, add: bool) -> list[int]:
    if len(words) < n:
        return []
    ids = []
    for w in words:
        i = vocab.get(w)
        if i is None:
            if not add:
                i = 0
            else:
                i = len(vocab) + 1
                vocab[w] = i
        ids.append(i)
    out, mod = [], (1 << 61) - 1
    for s in range(len(ids) - n + 1):
        h = 0
        for k in range(s, s + n):
            h = (h * 1_000_003 + ids[k]) % mod
        out.append(h)
    return out


@app.function(image=image, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("maemm-hf")], timeout=6 * 3600,
              cpu=8, memory=65536)
def dedup(n: int = 7) -> dict:
    import json
    import os
    import time

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    os.environ.pop("HF_HUB_OFFLINE", None)

    # --- the training shingle set ----------------------------------------------------
    vocab: dict[str, int] = {}
    hashes: list[int] = []
    rows_seen = 0
    for rel in TRAIN_SOURCES:
        df = pd.read_parquet(f"{BUNDLE}/{rel}")
        col = next(c for c in TEXT_COLS if c in df.columns)
        txts = df[col].astype(str).tolist()
        rows_seen += len(txts)
        for t in txts:
            hashes.extend(_shingles(_norm(t), n, vocab, add=True))
        print(f"[train] {rel}: {len(txts):,} rows, {len(hashes):,} shingles "
              f"({time.time()-t0:.0f}s)", flush=True)
    train = np.unique(np.array(hashes, dtype=np.uint64))
    del hashes
    print(f"[train] {train.size:,} distinct {n}-grams over {rows_seen:,} rows", flush=True)

    # --- recover the source documents ------------------------------------------------
    f = hf_hub_download(DATASET, PART1, repo_type="dataset")
    head = pq.read_table(f, columns=["content"]).column("content").to_pylist()[:EVAL_HEAD]
    print(f"[corpus] {len(head):,} documents of the eval head ({time.time()-t0:.0f}s)",
          flush=True)
    norm_docs = [" ".join(_norm(d or "")) for d in head]

    ev = pd.read_parquet(f"{BUNDLE}/heldout/eval_directions_v3/realact.parquet",
                         columns=["pool_target_text"])
    spans = ev["pool_target_text"].astype(str).tolist()

    # An exact normalised substring match: a target resolves to one document or to none.
    index: dict[str, list[int]] = {}
    for i, d in enumerate(norm_docs):
        for w in set(d.split()[:400]):
            index.setdefault(w, []).append(i)

    resolved, cov, unresolved = [], [], 0
    for span in spans:
        sw = _norm(span)
        needle = " ".join(sw)
        cands = min((index.get(w, []) for w in sw[:8] if w in index),
                    key=len, default=[])
        hit = next((i for i in cands if needle in norm_docs[i]), None)
        if hit is None:
            unresolved += 1
            resolved.append(None)
            cov.append(float("nan"))
            continue
        resolved.append(int(hit))
        sh = _shingles(norm_docs[hit].split(), n, vocab, add=False)
        if not sh:
            cov.append(float("nan"))
            continue
        a = np.array(sh, dtype=np.uint64)
        idx = np.searchsorted(train, a)
        idx[idx >= train.size] = 0
        cov.append(float((train[idx] == a).mean()))

    c = np.array(cov, dtype=np.float64)
    ok = ~np.isnan(c)
    out_dir = f"{VOL}/shared/ngram-overlap"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/doc_level_n{n}.jsonl", "w") as fh:
        for i, (doc, x) in enumerate(zip(resolved, c)):
            fh.write(json.dumps({"row": i, "doc": doc,
                                 "doc_coverage": None if np.isnan(x) else round(float(x), 6)})
                     + chr(10))
    out = {
        "n": n,
        "eval_rows": len(spans),
        "unresolved": unresolved,
        "resolved": int(ok.sum()),
        "train_rows": rows_seen,
        "train_distinct_ngrams": int(train.size),
        "doc_coverage": {
            "mean": float(c[ok].mean()),
            "p50": float(np.percentile(c[ok], 50)),
            "p90": float(np.percentile(c[ok], 90)),
            "p99": float(np.percentile(c[ok], 99)),
            "max": float(c[ok].max()),
        },
        "rows_at_or_above": {k: int((c[ok] >= v).sum())
                             for k, v in {"any": 1e-9, "0.05": .05, "0.20": .20,
                                          "0.50": .50, "0.90": .90}.items()},
        "written": f"{out_dir}/doc_level_n{n}.jsonl",
        "wall_s": round(time.time() - t0, 1),
    }
    vol.commit()
    print(json.dumps(out, indent=1), flush=True)
    return out


@app.local_entrypoint()
def main(n: int = 7):
    import json

    print(json.dumps(dedup.remote(n=n), indent=1))
