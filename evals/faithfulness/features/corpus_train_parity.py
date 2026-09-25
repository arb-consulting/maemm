"""Build a search corpus from the checkpoint's OWN training document ranges.

    modal run features/corpus_train_parity.py --name train_parity_10m

Writes the same contract `precompute/corpus.py` writes -- `tokens.i32` + `docs.jsonl`
with (doc, offset, len, size_tag, part, row) -- under `base/<base>/corpora/<name>/`, so
`scan` reads it unchanged. It is a SEPARATE corpus, never an edit of `corpus/`: every
stored scan window and top-k list indexes into that exact `tokens.i32`, and rebuilding
in place silently invalidates all of them.

## Why the upstream training ranges and not our held-out slice

2026-09-20. The corpus baseline answers "could retrieval have found this instead
of generating it". Retrieving from a held-out web slice answers that for text the model
never saw; retrieving from the slices it trained on puts search and the generator on the
same data, which is the parity the comparison is about.

It cuts both ways and the paper has to say which claim it is making: the search now has
access to text the model may have memorised, so a corpus win is no longer evidence that
retrieval beats generation on fresh text. Both numbers are defensible, they are not the
same number, and reporting one while describing the other would be the actual error.

The target's own slice cannot leak in: the upstream eval realacts are documents [0, 100,000) and
these ranges start at 5.5M, so exclusion is automatic rather than enforced.

## Budget and geometry

10M tokens at nested 1.25 / 2.5 / 5 / 10M -- parity with training, which saw 9-10M
activations, replacing the 1/2/4/8/16M ladder. Windows are 32 tokens, against
the 64 the existing corpus uses, so the two are NOT interchangeable and a number from
one must never be compared with a number from the other.

## Document addressing

`doc_idx` is the cumulative row index over
`data/ultrafineweb_en/ultrafineweb-en-part-NNNN-of-2048.parquet` in filename order,
**starting at part 0001** -- there is no part 0000. Row counts are read from the parquet
footers at run time and asserted against the two anchors the 2026-09-18 disjointness
check established (start[9] = 4,528,159, start[10] = 5,094,179), so a change in the
published dataset fails loudly instead of silently shifting every range.
"""
from __future__ import annotations

import modal

VOL = "/vol"
DATASET = "openbmb/Ultra-FineWeb"
PART_FMT = "data/ultrafineweb_en/ultrafineweb-en-part-{:04d}-of-2048.parquet"

# The upstream training document ranges, half-open, from heldout/doc_registry.json.
TRAIN_RANGES = {
    "sft_activations_ctx8_64": (5_500_001, 5_698_524),
    "rl_activations_ctx64_2048": (9_500_000, 9_599_842),
}
# Anchors from the 2026-09-18 disjointness check; the build asserts these.
ANCHORS = {9: 4_528_159, 10: 5_094_179}

SIZES = [1.25, 2.5, 5, 10]
TOKENS = 10_000_000
BLOCK, STRIDE = 32, 8
SEED = 20260920

vol = modal.Volume.from_name("maem", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.3.4", "pyarrow==24.0.0", "transformers==4.57.1",
                 "huggingface_hub==0.36.0", "tqdm")
    .env({"HF_HOME": f"{VOL}/hf"})
)
app = modal.App("maem-corpus-train-parity")


@app.function(image=image, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("maem-hf")], timeout=6 * 3600,
              cpu=8, memory=32768)
def build(name: str = "train_parity_10m", base: str = "qwen36-27b",
          tokens: int = TOKENS, dry_run: bool = False) -> dict:
    import json
    import os
    import time

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    t0 = time.time()
    os.environ.pop("HF_HUB_OFFLINE", None)   # the parquet parts are not in the cache

    # --- resolve cumulative row offsets, and check the anchors -----------------------
    starts, cum = {}, 0
    need_parts: set[int] = set()
    for part in range(1, 25):
        starts[part] = cum
        f = hf_hub_download(DATASET, PART_FMT.format(part), repo_type="dataset")
        cum += pq.ParquetFile(f).metadata.num_rows
        for lo, hi in TRAIN_RANGES.values():
            if starts[part] < hi and cum > lo:
                need_parts.add(part)
        if cum > max(hi for _, hi in TRAIN_RANGES.values()):
            break
    for part, expect in ANCHORS.items():
        assert starts.get(part) == expect, (
            f"part {part} starts at {starts.get(part)}, the 2026-09-18 check says {expect}; "
            f"the published dataset has changed and every range here is suspect")
    print(f"[corpus] ranges span parts {sorted(need_parts)}", flush=True)
    if dry_run:
        return {"starts": {k: v for k, v in starts.items() if k in need_parts},
                "parts": sorted(need_parts)}

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    share = tokens // len(TRAIN_RANGES)
    docs, got_total, per_source = [], 0, {}

    for si, (source, (lo, hi)) in enumerate(TRAIN_RANGES.items()):
        want = tokens - share * (len(TRAIN_RANGES) - 1) if si == len(TRAIN_RANGES) - 1 else share
        got, n_docs = 0, 0
        for part in sorted(need_parts):
            p_lo, p_hi = starts[part], starts[part] + pq.ParquetFile(
                hf_hub_download(DATASET, PART_FMT.format(part), repo_type="dataset")
            ).metadata.num_rows
            if p_hi <= lo or p_lo >= hi or got >= want:
                continue
            f = hf_hub_download(DATASET, PART_FMT.format(part), repo_type="dataset")
            tbl = pq.read_table(f, columns=["content"])
            texts = tbl.column("content").to_pylist()
            first = max(lo - p_lo, 0)
            last = min(hi - p_lo, len(texts))
            buf, rows = [], []
            for row in range(first, last):
                text = (texts[row] or "").strip()
                if not text:
                    continue
                buf.append(text)
                rows.append((part, row, p_lo + row))
                if len(buf) >= 512:
                    for ids, (pt, rw, gi) in zip(
                        tok(buf, add_special_tokens=False)["input_ids"], rows):
                        if got >= want:
                            break
                        a = np.asarray(ids, dtype=np.int32)
                        docs.append((source, pt, rw, gi, a))
                        got += len(a); n_docs += 1
                    buf, rows = [], []
                    if got >= want:
                        break
            if buf and got < want:
                for ids, (pt, rw, gi) in zip(
                        tok(buf, add_special_tokens=False)["input_ids"], rows):
                    if got >= want:
                        break
                    a = np.asarray(ids, dtype=np.int32)
                    docs.append((source, pt, rw, gi, a))
                    got += len(a); n_docs += 1
        per_source[source] = {"tokens": got, "docs": n_docs, "range": [lo, hi]}
        got_total += got
        print(f"[corpus] {source}: {n_docs:,} docs, {got:,} tokens ({time.time()-t0:.0f}s)",
              flush=True)

    # --- permute, tag nested sizes, write --------------------------------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(docs))
    rows_out, cum_tok, per_size = [], 0, {s: 0 for s in SIZES}
    for new_i, i in enumerate(perm):
        source, part, row, gi, arr = docs[int(i)]
        tag = next((s for s in SIZES if cum_tok + len(arr) <= round(s * 1_000_000)), SIZES[-1])
        rows_out.append({"doc": new_i, "offset": cum_tok, "len": int(len(arr)),
                         "size_tag": tag, "part": part, "row": row,
                         "doc_idx": int(gi), "source": source})
        per_size[tag] += int(len(arr))
        cum_tok += len(arr)

    out = f"{VOL}/base/{base}/corpora/{name}"
    os.makedirs(out, exist_ok=True)
    np.concatenate([docs[int(i)][4] for i in perm]).astype(np.int32).tofile(f"{out}/tokens.i32")
    with open(f"{out}/docs.jsonl", "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + chr(10))
    meta = {
        "name": name, "dataset": DATASET, "tokens": int(cum_tok), "docs": len(rows_out),
        "sizes": SIZES, "block": BLOCK, "stride": STRIDE, "seed": SEED,
        "train_ranges": TRAIN_RANGES, "per_source": per_source,
        "per_size_tokens": {str(k): v for k, v in per_size.items()},
        "part_starts": {str(k): v for k, v in starts.items() if k in need_parts},
        "wall_s": round(time.time() - t0, 1),
    }
    with open(f"{out}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    with open(f"{out}/README.md", "w") as fh:
        fh.write(
            f"# {name} -- search corpus over the checkpoint's own training ranges\n\n"
            f"{cum_tok:,} tokens, {len(rows_out):,} documents, nested sizes {SIZES} "
            f"(millions), window {BLOCK}/{STRIDE}, seed {SEED}.\n\n"
            "Built from the document ranges the v2 checkpoint trained on, so corpus search "
            "and the generator see the same data (2026-09-20). That is parity, and it "
            "also means the search can retrieve text the model may have memorised -- a corpus "
            "win here is NOT evidence that retrieval beats generation on unseen text.\n\n"
            "The upstream eval realacts are documents [0, 100,000); these ranges start at 5.5M, so a "
            "target's own document cannot appear in the search corpus.\n\n"
            "NOT interchangeable with `corpus/`: different documents, different budget ladder "
            "and 32-token windows against 64. Never compare a number from one with the other.\n\n"
            + json.dumps(per_source, indent=1) + "\n")
    vol.commit()
    print(json.dumps(meta, indent=1), flush=True)
    return meta


@app.local_entrypoint()
def main(name: str = "train_parity_10m", base: str = "qwen36-27b",
         tokens: int = TOKENS, dry_run: bool = False):
    import json

    print(json.dumps(build.remote(name=name, base=base, tokens=tokens, dry_run=dry_run),
                     indent=1, default=str))
