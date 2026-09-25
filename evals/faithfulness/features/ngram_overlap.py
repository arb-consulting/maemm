"""n-gram overlap between the eval realact targets and the v2 training text.

    modal run features/ngram_overlap.py --n 7
    modal run features/ngram_overlap.py --n 7 --side ours

Answers one question: what fraction of an eval target's word n-grams also occur
somewhere in the text the checkpoint was trained on. 2026-09-20: some realact
test samples look close to training data, so a disjoint redraw may be needed.

**Coverage, not hits.** At n=7 ordinary English recurs constantly, so "shares at least
one shingle" fires on nearly everything and means nothing. What separates a near
duplicate from a coincidence is the SHARE of a target's shingles that are reached, so
that is what this reports, as a distribution. Pick a threshold off that distribution --
and pick it before looking at which targets it would remove.

**Both sides, same bar.** The existing 13-gram check (2026-09-18) measured OUR corpus
against the upstream training text and found realact at 0.50% of rows against dict2m's 8.31%. But
we are standardising on THE UPSTREAM data, so the pair that matters now is the upstream eval realact pool
against the upstream training ranges. `--side` picks which. Whatever bar disqualifies one
side has to be applied to the other: the upstream eval windows already overlap our corpus at a
comparable rate, and filtering one side only is a worse problem than the contamination.

**What the 13-gram pass could not see.** 1.8M of the upstream training rows (13.3%) are shorter
than 13 words and were invisible to it by construction. n=7 reaches them.

Nothing here rewrites the corpus. A redraw should EXCLUDE failing documents from the
pool targets.py samples, not drop them from the corpus -- dropping renumbers `doc` ids
and invalidates every stored scan window and top-k list.
"""
from __future__ import annotations

import modal

VOL = "/vol"
BUNDLE = f"{VOL}/data/v2-bundle"

vol = modal.Volume.from_name("maem", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.3.4", "pandas==3.0.3", "pyarrow==24.0.0")
)
app = modal.App("maem-ngram-overlap")

# The upstream training text for the activation families: the SFT bank's 8-64-token contexts and
# the RL pool's 64-2048-token ones. The SAE banks are max-activating windows, a
# different population, and are counted separately when present.
TRAIN_SOURCES = [
    ("sft_realact", "simple2m/sft_mix/realact/records.parquet"),
    ("rl_realact", "simple2m/rl_pool/realact_ctx64_2048/records.parquet"),
    # The SAE banks are text the model trained on too. Excluding them biases the
    # answer DOWN, and they are the likeliest to match: max-activating windows select
    # for templated, duplicated web text, which is why the 2026-09-18 13-gram pass put
    # dict2m at 8.31% of rows against realact's 0.50%.
    ("sft_dict2m", "simple2m/sft_mix/dict2m/records.parquet"),
    ("sft_dict2m_dec", "simple2m/sft_mix/dict2m_dec/records.parquet"),
    ("rl_dict2m", "simple2m/rl_pool/dict2m/records.parquet"),
    ("rl_dict2m_dec", "simple2m/rl_pool/dict2m_dec/records.parquet"),
]
EVAL_SOURCES = {
    "upstream_512": "heldout/eval_directions_v3/realact.parquet",
    "upstream_pool": "heldout/pool_heldout/realact.parquet",
}
TEXT_COLS = ("target_text", "pool_target_text", "text")


def _texts(df):
    for c in TEXT_COLS:
        if c in df.columns:
            return df[c].astype(str).tolist()
    raise KeyError(f"no text column in {list(df.columns)}")


def _shingles(text: str, n: int, vocab: dict, add: bool):
    """Rolling 64-bit hashes of the word n-grams of one document.

    NFKC -> lowercase -> whitespace split, matching the 2026-09-18 13-gram pass so the
    two are comparable. Words are interned to ints and hashed with a fixed polynomial:
    stable across processes, unlike Python's salted str hash.
    """
    import unicodedata

    words = unicodedata.normalize("NFKC", text).lower().split()
    if len(words) < n:
        return []
    ids = []
    for w in words:
        i = vocab.get(w)
        if i is None:
            if not add:
                i = 0                      # unseen word: a token that cannot match
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


@app.function(image=image, volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=65536)
def overlap(n: int = 7, side: str = "upstream", limit_train: int = 0) -> dict:
    import json
    import time

    import numpy as np
    import pandas as pd

    t0 = time.time()
    vocab: dict[str, int] = {}

    train_hashes = []
    train_rows = 0
    for name, rel in TRAIN_SOURCES:
        path = f"{BUNDLE}/{rel}"
        df = pd.read_parquet(path, columns=None)
        txts = _texts(df)
        if limit_train:
            txts = txts[:limit_train]
        train_rows += len(txts)
        for t in txts:
            train_hashes.extend(_shingles(t, n, vocab, add=True))
        print(f"[train] {name}: {len(txts):,} rows, {len(train_hashes):,} shingles "
              f"({time.time() - t0:.0f}s)", flush=True)
    train = np.unique(np.array(train_hashes, dtype=np.uint64))
    del train_hashes
    print(f"[train] {train.size:,} distinct {n}-grams over {train_rows:,} rows", flush=True)

    src = EVAL_SOURCES["upstream_512"] if side == "upstream" else EVAL_SOURCES["upstream_pool"]
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(f"{BUNDLE}/{src}").schema.names)
    col = next(c for c in TEXT_COLS if c in have)   # read the text column only: the
    ev = pd.read_parquet(f"{BUNDLE}/{src}", columns=[col])   # direction column is 2 GB
    texts = _texts(ev)

    cov, short = [], 0
    for t in texts:
        sh = _shingles(t, n, vocab, add=False)
        if not sh:
            short += 1
            cov.append(float("nan"))
            continue
        a = np.array(sh, dtype=np.uint64)
        idx = np.searchsorted(train, a)
        idx[idx >= train.size] = 0
        cov.append(float((train[idx] == a).mean()))
    c = np.array(cov, dtype=np.float64)
    ok = ~np.isnan(c)

    # The per-row list is the deliverable: an exclusion list beats a redraw, because
    # dropping documents renumbers `doc` ids and invalidates every stored scan window.
    import json as _json
    import os as _os

    rows = [{"row": int(i), "coverage": (None if np.isnan(x) else round(float(x), 6))}
            for i, x in enumerate(c)]
    out_dir = f"{VOL}/shared/ngram-overlap"
    _os.makedirs(out_dir, exist_ok=True)
    tag = f"{side}_n{n}"
    with open(f"{out_dir}/{tag}.jsonl", "w") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + chr(10))
    flagged = sorted((int(i) for i in np.where(ok & (c >= 0.05))[0]),
                     key=lambda i: -c[i])
    with open(f"{out_dir}/{tag}.exclude.json", "w") as fh:
        _json.dump({"source": src, "n": n, "threshold": 0.05,
                    "excluded_rows": flagged,
                    "coverage": {str(i): round(float(c[i]), 4) for i in flagged}},
                   fh, indent=1)
    vol.commit()

    out = {
        "n": n,
        "side": side,
        "eval_source": src,
        "eval_rows": int(len(texts)),
        "eval_rows_shorter_than_n": int(short),
        "train_rows": train_rows,
        "train_distinct_ngrams": int(train.size),
        "coverage": {
            "mean": float(c[ok].mean()),
            "p50": float(np.percentile(c[ok], 50)),
            "p90": float(np.percentile(c[ok], 90)),
            "p99": float(np.percentile(c[ok], 99)),
            "max": float(c[ok].max()),
        },
        "rows_at_or_above": {
            "any": int((c[ok] > 0).sum()),
            "0.05": int((c[ok] >= 0.05).sum()),
            "0.20": int((c[ok] >= 0.20).sum()),
            "0.50": int((c[ok] >= 0.50).sum()),
            "0.90": int((c[ok] >= 0.90).sum()),
        },
        "excluded_rows_at_0.05": flagged,
        "written": f"{out_dir}/{tag}.jsonl and {tag}.exclude.json",
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1), flush=True)
    return out


@app.local_entrypoint()
def main(n: int = 7, side: str = "upstream", limit_train: int = 0):
    import json

    print(json.dumps(overlap.remote(n=n, side=side, limit_train=limit_train), indent=1))
