"""Stage 0 of the v2 rebuild plan: is our held-out corpus disjoint from Celeste's v2 bundle?

    export MODAL_PROFILE=maemms
    # part A alone, on local copies of the small files (no Modal, no volume):
    uv run --with pyarrow --with numpy --with huggingface_hub --with modal \
        python infra/check_v2_disjointness.py --v2 <local v2-2026-09-17> --corpus <local corpus>
    # both parts, on the volume:
    uv run --with modal --with pyarrow --with numpy --with xxhash modal run infra/check_v2_disjointness.py \
        --out /abs/path/report.json --local-v2 <dir> --local-corpus <dir>

Part A (documents).  Every corpus document is addressed by (part, row) in `docs.jsonl`; its index in
Celeste's single Ultra-FineWeb `en` stream is `start[part] + row`.  Assert that no corpus stream
index is in any `heldout/doc_ids/*.parquet` list or in a `kind: range` source of
`extra/doc_registry.json` (the 2M-SAE training stream and the reserved eval head), and report the
margin.  `start` is re-read from the public parquet footers at run time (the only network call in
this script); the hard-coded value is the previous session's footer read, recorded in
infra/2026-09-18_celeste-v2-data.md §1.3.  Runs locally and again inside the Modal function.

Part B (content).  Doc-disjoint is not content-disjoint: Ultra-FineWeb is not deduplicated (§1.2,
the feature-28 duplicates).  Convention: text -> NFKC -> lowercase -> whitespace split -> every
13-word shingle -> xxh3_64 of those words joined by single spaces.  Build the shingle set of our
18,813 documents (detokenized from `tokens.i32` with the base's own tokenizer), then stream the
`target_text` of every v2 training record and the `text` of her 100k-feature eval windows and count
the hits.  A row shorter than 13 words yields no shingle and cannot be detected.  CPU only, ~$0.

Writes the JSON report to `<corpus>/celeste_v2_disjointness.json` on the volume and appends (or
replaces) one summary line in `<corpus>/README.md`.  Nothing else under `base/` is touched.
"""

import json
import re
import time
import unicodedata
from pathlib import Path

import modal

VOL_V2 = "/vol/data/celeste-v2-2026-09-17"
VOL_CORPUS = "/vol/base/qwen36-27b/corpus"
SHINGLES = "/vol/tmp/check_v2_disjointness/corpus_shingles.npz"  # scratch, outside base/
BUNDLE = "celeste-v2-2026-09-17"
DATASET = "openbmb/Ultra-FineWeb"
PART_PATH = "data/ultrafineweb_en/ultrafineweb-en-part-{:04d}-of-2048.parquet"
BATCH = 100_000  # rows per pyarrow batch in the scan; bounds the Python shingle list

N = 13  # shingle length, in whitespace-separated words
# Cumulative row count of parts 0001..(p-1) of Ultra-FineWeb `en`, i.e. the stream index of row 0 of
# part p.  Read from the public parquet footers (infra/2026-09-18_celeste-v2-data.md §1.3) and
# re-read at run time unless --no-verify-starts.  There is no part 0000.
PART_START = {9: 4_528_159, 10: 5_094_179}
PART_ROWS_BAND = (566_018, 566_023)  # per-part row counts seen for parts 0001..0011
# `- part 0 data/.../ultrafineweb-en-part-0009-of-2048.parquet: rows 0..9388 (9389 non-empty docs, ...`
PART_LINE = re.compile(
    r"^- part (\d+) (\S*part-(\d+)-of-\d+\.parquet): rows (\d+)\.\.(\d+) \((\d+) non-empty docs"
)
README_MARK = f"- Disjoint from {BUNDLE} training docs"

app = modal.App("maemm-check-v2-disjointness")
vol = modal.Volume.from_name("maemm", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy", "pyarrow", "xxhash", "transformers", "huggingface_hub", "fsspec"
)


# --------------------------------------------------------------------------- part A: documents


def _corpus_parts(corpus_dir: Path) -> dict[int, dict]:
    """Map the `part` field of docs.jsonl to the source part number, from the corpus README notes.

    The README is the only place that records which parquet part each index means; `index.json`
    records shapes only, and neither records the parts' row counts (hence the footer re-read).
    """
    parts = {}
    for line in (corpus_dir / "README.md").read_text().splitlines():
        if m := PART_LINE.match(line.strip()):
            parts[int(m[1])] = {
                "file": m[2],
                "part_number": int(m[3]),
                "rows": [int(m[4]), int(m[5])],
                "docs": int(m[6]),
            }
    assert parts, f"no `- part N <file>: rows a..b` notes in {corpus_dir}/README.md"
    return parts


def verify_part_starts() -> dict:
    """Re-read the row count of Ultra-FineWeb en parts 0001..0010 from their parquet footers.

    Only the footer is fetched (HTTP range requests through HfFileSystem), a few hundred KB in all.
    This is the one network call in this script and the only independent check of PART_START.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows, starts, cum = {}, {}, 0
    for p in range(1, max(PART_START) + 1):
        starts[p] = cum
        with fs.open(f"datasets/{DATASET}/{PART_PATH.format(p)}", "rb") as f:
            n = pq.ParquetFile(f).metadata.num_rows
        rows[p] = n
        cum += n
    return {"part_rows": rows, "part_start": {p: starts[p] for p in PART_START}}


def part_a(corpus_dir: Path, v2_dir: Path, part_start: dict[int, int]) -> dict:
    """Stream indices of our corpus vs every document set of the v2 bundle."""
    import numpy as np
    import pyarrow.parquet as pq

    parts = _corpus_parts(corpus_dir)
    docs = [json.loads(ln) for ln in (corpus_dir / "docs.jsonl").read_text().splitlines()]
    by_part: dict[int, list[int]] = {}
    for d in docs:
        by_part.setdefault(d["part"], []).append(d["row"])
    ours_by_part = {}
    for pi, rws in sorted(by_part.items()):
        info = parts[pi]
        lo, hi = info["rows"]
        assert min(rws) >= lo and max(rws) <= hi, f"docs.jsonl part {pi} rows outside README {lo}..{hi}"
        assert len(rws) == info["docs"], f"docs.jsonl part {pi}: {len(rws)} docs, README says {info['docs']}"
        pnum = info["part_number"]
        assert pnum in part_start, f"part {pnum} has no start index"
        ours_by_part[str(pnum)] = {
            "file": info["file"],
            "rows": [lo, hi],
            "docs": len(rws),
            "contiguous": hi - lo + 1 == len(rws),
            "stream_first": part_start[pnum] + lo,
            "stream_last": part_start[pnum] + hi,
        }
    ours = np.array(
        sorted(part_start[parts[d["part"]]["part_number"]] + d["row"] for d in docs), dtype=np.int64
    )
    assert len(np.unique(ours)) == len(ours), "duplicate corpus stream indices"

    # Self-consistency of the start constants against the recorded per-part row band.
    band_ok = {
        f"start[{b}]-start[{a}]": bool(
            PART_ROWS_BAND[0] <= part_start[b] - part_start[a] <= PART_ROWS_BAND[1]
        )
        for a, b in zip(sorted(part_start), sorted(part_start)[1:], strict=False)
    }

    reg = json.loads((v2_dir / "extra/doc_registry.json").read_text())
    out_sources, worst_gap = {}, None
    for name, src in reg["sources"].items():
        lo, hi = src["doc_range"]  # `range`: half-open; `exact`: inclusive min/max of the list
        f = v2_dir / "heldout/doc_ids" / f"{name}.parquet"
        if src["kind"] == "exact":
            assert f.exists(), f"registry source {name} is `exact` but {f} is missing"
            theirs = np.unique(pq.read_table(f, columns=["doc_idx"])["doc_idx"].to_numpy().astype(np.int64))
            assert len(theirs) == src["n_docs"], (
                f"{name}: {len(theirs)} doc_ids, registry says {src['n_docs']}"
            )
            assert [int(theirs[0]), int(theirs[-1])] == [lo, hi], f"{name}: doc_ids range != registry range"
            i = np.searchsorted(theirs, ours)
            gap = int(
                min(
                    np.abs(ours - theirs[np.clip(i - 1, 0, len(theirs) - 1)]).min(),
                    np.abs(ours - theirs[np.clip(i, 0, len(theirs) - 1)]).min(),
                )
            )
            inter = int(np.intersect1d(ours, theirs, assume_unique=True).size)
            lo_rep, hi_rep = int(theirs[0]), int(theirs[-1])
        else:  # kind == "range", half-open [lo, hi)
            assert not f.exists(), f"registry source {name} is `range` but {f} exists -- use the list"
            gap = int(np.maximum(np.maximum(lo - ours, ours - (hi - 1)), 0).min())
            inter = int(((ours >= lo) & (ours < hi)).sum())
            lo_rep, hi_rep = lo, hi - 1
        out_sources[name] = {
            "role": src["role"],
            "kind": src["kind"],
            "n_docs": int(src["n_docs"]),
            "min_doc_idx": lo_rep,
            "max_doc_idx": hi_rep,
            "intersection": inter,
            "min_distance": gap,
        }
        if inter == 0 and (worst_gap is None or gap < worst_gap):
            worst_gap = gap
    total_inter = sum(s["intersection"] for s in out_sources.values())
    return {
        "corpus_docs": len(ours),
        "corpus_stream_range": [int(ours[0]), int(ours[-1])],
        "corpus_parts": ours_by_part,
        "part_start": {str(k): v for k, v in part_start.items()},
        "part_start_band_ok": band_ok,
        "registry_built": reg.get("built"),
        "sources": out_sources,
        "total_intersection": total_inter,
        "min_distance_over_sources": worst_gap,
        "disjoint": total_inter == 0,
    }


@app.function(image=image, volumes={"/vol": vol}, cpu=2.0, memory=8192, timeout=1800)
def run_part_a(verify: bool = True) -> dict:
    """Part A on the volume's own copies of the corpus and the bundle, starts re-read from the hub."""
    starts, verified, part_rows = dict(PART_START), None, None
    if verify:
        try:
            v = verify_part_starts()
            part_rows, verified = v["part_rows"], True
            assert v["part_start"] == PART_START, f"footers give {v['part_start']}, constant {PART_START}"
            starts = v["part_start"]
        except AssertionError:
            raise
        except Exception as e:  # hub trouble is not a reason to lose the rest; say so loudly
            verified = f"{type(e).__name__}: {e}"
            print(f"[starts] NOT VERIFIED ({verified}); using hard-coded {PART_START}", flush=True)
    a = part_a(Path(VOL_CORPUS), Path(VOL_V2), starts)
    return {"part_start_verified": verified, "part_rows": part_rows, **a}


# --------------------------------------------------------------------------- part B: 13-grams


def normalize(text: str) -> list[str]:
    """The report's normalisation: NFKC, lowercase, whitespace split."""
    return unicodedata.normalize("NFKC", text).lower().split()


def shingles_of(w: list[str], n: int = N) -> list[int]:
    """The 13-gram hashes of a normalised word list under the report's convention."""
    import xxhash

    return [xxhash.xxh3_64_intdigest(" ".join(w[i : i + n]).encode()) for i in range(len(w) - n + 1)]


def shingle_hashes(text: str, n: int = N) -> list[int]:
    return shingles_of(normalize(text), n)


def _selftest() -> None:
    """The shingling convention, checked locally before anything is launched."""
    t = "The  quick\u00a0brown FOX jumps over the lazy dog again and again and Again"
    w = normalize(t)
    assert w[:4] == ["the", "quick", "brown", "fox"], w[:4]  # NFKC turns U+00A0 into a space
    assert len(shingles_of(w)) == len(w) - N + 1 == 2, (len(w), shingles_of(w))
    assert shingles_of(w)[0] == shingles_of(w[:N])[0], "a shingle must not depend on its context"
    assert shingles_of(normalize("a b")) == []
    assert shingles_of(w) == shingle_hashes(t) == shingles_of(normalize(t.lower()))
    assert all(0 <= h < 2**64 for h in shingles_of(w))


def _lookup(h, keys):
    """Boolean hit mask and the position in `keys` of each hash of `h` (both numpy arrays)."""
    import numpy as np

    if len(h) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    pos = np.clip(np.searchsorted(keys, h), 0, len(keys) - 1)
    return keys[pos] == h, pos


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=32768, timeout=3 * 3600)
def build_corpus_shingles() -> dict:
    """Detokenize the corpus, build its 13-gram set on the volume, and run both controls."""
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    _selftest()
    t0 = time.time()
    corpus = Path(VOL_CORPUS)
    idx = json.loads((corpus / "index.json").read_text())
    toks = np.fromfile(corpus / "tokens.i32", dtype=np.int32)
    assert toks.shape[0] == idx["tokens.i32"]["shape"][0], "tokens.i32 length != index.json"
    docs = [json.loads(ln) for ln in (corpus / "docs.jsonl").read_text().splitlines()]
    assert len(docs) == idx["docs.jsonl"]["rows"], "docs.jsonl rows != index.json"

    snaps = sorted(Path("/vol/hf/hub/models--Qwen--Qwen3.6-27B/snapshots").glob("*"))
    assert len(snaps) == 1, f"expected one Qwen3.6-27B snapshot, found {snaps}"
    tok = AutoTokenizer.from_pretrained(str(snaps[0]))
    print(f"[corpus] tokenizer {snaps[0].name} vocab {len(tok)}, {len(docs)} docs", flush=True)

    texts: list[str] = []
    for i in range(0, len(docs), 256):
        chunk = [toks[d["offset"] : d["offset"] + d["len"]].tolist() for d in docs[i : i + 256]]
        texts.extend(tok.batch_decode(chunk, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    # Decoding must be lossless or these are not our corpus's shingles: re-encode a sample.
    rng = np.random.default_rng(0)
    bad = []
    for i in rng.choice(len(docs), size=50, replace=False):
        ids = toks[docs[i]["offset"] : docs[i]["offset"] + docs[i]["len"]].tolist()
        if tok(texts[i], add_special_tokens=False)["input_ids"] != ids:
            bad.append(int(i))
    assert not bad, f"decode round-trip failed on docs {bad} of 50 sampled"
    print(f"[corpus] decoded {sum(map(len, texts))} chars in {time.time() - t0:.0f}s", flush=True)

    per_doc, n_short, n_words = [], 0, 0
    for t in texts:
        w = normalize(t)
        n_words += len(w)
        n_short += len(w) < N
        hs = shingles_of(w)
        per_doc.append(np.fromiter(hs, dtype=np.uint64, count=len(hs)))
    counts = np.array([len(a) for a in per_doc], dtype=np.int64)
    h = np.concatenate(per_doc)
    d = np.repeat(np.arange(len(docs), dtype=np.int32), counts)
    keys, first = np.unique(h, return_index=True)  # `first` = first occurrence in document order
    doc_of = d[first]
    Path(SHINGLES).parent.mkdir(parents=True, exist_ok=True)
    np.savez(SHINGLES, keys=keys, doc_of=doc_of, doc_shingles=counts)
    vol.commit()
    print(f"[corpus] {len(h)} shingles, {len(keys)} distinct, {time.time() - t0:.0f}s", flush=True)

    # Positive control: 200 word-spans taken out of our own documents must hit on every shingle.
    ok_docs = [i for i, t in enumerate(texts) if len(t.split()) >= 2 * N]
    pos_total, pos_hit = 0, 0
    for i in rng.choice(ok_docs, size=200, replace=False):
        w = normalize(texts[i])
        k = int(rng.integers(N, min(len(w), 40) + 1))
        j = int(rng.integers(0, len(w) - k + 1))
        hh = np.fromiter(shingles_of(w[j : j + k]), dtype=np.uint64)
        hit, _ = _lookup(hh, keys)
        pos_total += len(hh)
        pos_hit += int(hit.sum())
    # Negative control: her eval windows, shuffled at word level, must (nearly) never hit.
    ev = pq.ParquetFile(Path(VOL_V2) / "heldout/eval_2m_features_100k_windows.parquet")
    sample = next(ev.iter_batches(batch_size=5000, columns=["text"]))["text"].to_pylist()
    neg_total, neg_hit = 0, 0
    for t in [sample[i] for i in rng.choice(len(sample), size=200, replace=False)]:
        w = normalize(t)
        rng.shuffle(w)
        hh = np.fromiter(shingles_of(w), dtype=np.uint64)
        hit, _ = _lookup(hh, keys)
        neg_total += len(hh)
        neg_hit += int(hit.sum())
    return {
        "docs": len(docs),
        "tokens": int(toks.shape[0]),
        "chars": sum(map(len, texts)),
        "words": n_words,
        "shingles": int(len(h)),
        "shingles_distinct": int(len(keys)),
        "docs_shorter_than_n": int(n_short),
        "tokenizer": str(snaps[0]),
        "shingle_file": SHINGLES,
        "control_positive": {"shingles": pos_total, "hits": pos_hit},
        "control_negative_shuffled": {"shingles": neg_total, "hits": neg_hit},
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=image, volumes={"/vol": vol}, cpu=2.0, memory=8192, timeout=1800)
def list_targets() -> list[dict]:
    """Every parquet of v2 text to scan: the training records plus her eval windows."""
    import pyarrow.parquet as pq

    root = Path(VOL_V2)
    groups = {
        "simple2m/sft_mix": "simple2m_sft",
        "simple2m/rl_pool": "simple2m_rl",
        "legacy_5m_chain/sft_midtrain_mix_5m_sft": "legacy_sft",
        "legacy_5m_chain/rl_pool_mix_eq_1p45m": "legacy_rl",
    }
    out = []
    for p in sorted(root.glob("*/*/*/records.parquet")):
        rel = str(p.relative_to(root))
        group = next((g for pre, g in groups.items() if rel.startswith(pre + "/")), None)
        assert group, f"unmapped records.parquet: {rel}"
        out.append({"path": str(p), "rel": rel, "group": group, "column": "target_text"})
    ev = root / "heldout/eval_2m_features_100k_windows.parquet"
    out.append({"path": str(ev), "rel": str(ev.relative_to(root)), "group": "eval_windows", "column": "text"})
    for it in out:
        f = pq.ParquetFile(it["path"])
        assert it["column"] in f.schema_arrow.names, f"{it['rel']} has no {it['column']} column"
        it["rows"] = f.metadata.num_rows
        it["row_groups"] = f.metadata.num_row_groups
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=16384, timeout=3 * 3600)
def scan_file(item: dict) -> dict:
    """Count 13-gram hits of one v2 parquet's text column against the corpus shingle set."""
    import numpy as np
    import pyarrow.parquet as pq

    t0 = time.time()
    vol.reload()  # the shingle file was committed by another container after this one's snapshot
    z = np.load(SHINGLES)
    keys, doc_of = z["keys"], z["doc_of"]
    pf = pq.ParquetFile(item["path"])
    rows = rows_short = rows_any = rows_half = 0
    n_shingles = n_hits = 0
    hit_docs: set[int] = set()
    seen = np.zeros(len(keys), dtype=bool)  # which of OUR distinct shingles this file reaches
    examples: list[dict] = []
    for batch in pf.iter_batches(batch_size=BATCH, columns=[item["column"]]):
        col = batch[item["column"]].to_pylist()
        flat: list[int] = []
        per_row = np.empty(len(col), dtype=np.int64)
        for i, t in enumerate(col):
            hs = shingle_hashes(t or "")
            per_row[i] = len(hs)
            flat.extend(hs)
        h = np.fromiter(flat, dtype=np.uint64, count=len(flat))
        del flat
        hit, pos = _lookup(h, keys)
        ends = np.cumsum(per_row)
        starts = ends - per_row
        cum = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(hit.astype(np.int64))])
        row_hits = cum[ends] - cum[starts]
        rows_short += int((per_row == 0).sum())
        rows_any += int((row_hits > 0).sum())
        rows_half += int(((row_hits * 2 >= per_row) & (per_row > 0) & (row_hits > 0)).sum())
        n_shingles += int(per_row.sum())
        n_hits += int(hit.sum())
        if hit.any():
            seen[pos[hit]] = True
            hit_docs.update(int(x) for x in np.unique(doc_of[pos[hit]]))
            # Examples: the rows whose text overlaps ours most, not merely the first ones seen.
            frac = np.where(per_row > 0, row_hits / np.maximum(per_row, 1), 0.0)
            for i in np.lexsort((row_hits, frac))[::-1][:3]:
                if row_hits[i] == 0:
                    break
                j = int(starts[i]) + int(np.flatnonzero(hit[starts[i] : ends[i]])[0])
                examples.append(
                    {
                        "file": item["rel"],
                        "row": rows + int(i),
                        "corpus_doc": int(doc_of[pos[j]]),
                        "shingles": int(per_row[i]),
                        "shingle_hits": int(row_hits[i]),
                        "hit_fraction": round(float(frac[i]), 4),
                        "text": (col[i] or "")[:300],
                    }
                )
            examples = sorted(examples, key=lambda e: (-e["hit_fraction"], -e["shingle_hits"]))[:3]
        rows += len(col)
        print(f"[{item['rel']}] {rows}/{item['rows']} rows, {rows_any} with a hit", flush=True)
    mask_path = Path(SHINGLES).parent / "masks" / (item["rel"].replace("/", "_") + ".npy")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, np.packbits(seen))
    vol.commit()
    return {
        **{k: item[k] for k in ("rel", "group", "column", "rows")},
        "mask": str(mask_path),
        "corpus_shingles_reached": int(seen.sum()),
        "rows_scanned": rows,
        "rows_without_shingles": rows_short,
        "rows_with_hit": rows_any,
        "rows_half_or_more_hit": rows_half,
        "shingles": n_shingles,
        "shingle_hits": n_hits,
        "corpus_docs_hit": sorted(hit_docs),
        "examples": examples,
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=image, volumes={"/vol": vol}, cpu=2.0, memory=16384, timeout=1800)
def summarise_corpus_hits(results: list[dict]) -> dict:
    """How much of each of OUR documents the v2 data reaches, from the per-file shingle masks.

    A 13-gram that occurs in several of our documents is attributed to the first one (the
    `np.unique` first-occurrence rule of build_corpus_shingles), so the denominators here are
    distinct shingles per document under that attribution, not raw per-document shingle counts.
    Coverage separates a shared boilerplate line (a few shingles) from a near-duplicate document.
    """
    import numpy as np

    vol.reload()
    z = np.load(SHINGLES)
    doc_of, n_docs = z["doc_of"], len(z["doc_shingles"])
    denom = np.bincount(doc_of, minlength=n_docs).astype(np.int64)
    n_keys = len(doc_of)

    def _or(sel) -> np.ndarray:
        m = np.zeros(n_keys, dtype=bool)
        for r in sel:
            m |= np.unpackbits(np.load(r["mask"]), count=n_keys).astype(bool)
        return m

    out, cuts = {}, (0.5, 0.2, 0.05)

    def _stats(m: np.ndarray) -> dict:
        got = np.bincount(doc_of[m], minlength=n_docs).astype(np.int64)
        cov = np.where(denom > 0, got / np.maximum(denom, 1), 0.0)
        order = np.argsort(-cov)[:10]
        return {
            "corpus_shingles_reached": int(m.sum()),
            "corpus_shingles_fraction": round(float(m.sum()) / n_keys, 6),
            "docs_touched": int((got > 0).sum()),
            **{f"docs_coverage_ge_{c}": int((cov >= c).sum()) for c in cuts},
            "max_coverage": round(float(cov.max()), 4),
            "top_docs": [
                {"doc": int(i), "coverage": round(float(cov[i]), 4), "shingles": int(denom[i]),
                 "shingles_hit": int(got[i])}
                for i in order if got[i] > 0
            ],
        }

    groups = sorted({r["group"] for r in results})
    for g in groups:
        out[g] = _stats(_or([r for r in results if r["group"] == g]))
    out["training"] = _stats(_or([r for r in results if r["group"] != "eval_windows"]))
    out["all"] = _stats(_or(results))
    out["corpus_shingles_distinct"] = n_keys
    return out


# --------------------------------------------------------------------------- report


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=2048, timeout=600)
def write_report(report: dict, summary: str) -> str:
    """Write the JSON beside the corpus and put `summary` in its README (replacing an older line)."""
    out = Path(VOL_CORPUS) / "celeste_v2_disjointness.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    readme = Path(VOL_CORPUS) / "README.md"
    lines = [ln for ln in readme.read_text().splitlines() if not ln.startswith(README_MARK)]
    readme.write_text("\n".join([*lines, summary, ""]))
    vol.commit()
    return f"wrote {out} ({out.stat().st_size} B) and the README line"


AGG_KEYS = (
    "rows", "rows_scanned", "rows_without_shingles", "rows_with_hit", "rows_half_or_more_hit",
    "shingles", "shingle_hits",
)
DROP_KEYS = ("corpus_docs_hit", "examples", "rel", "mask")


def _aggregate(results: list[dict], targets: list[dict]) -> dict:
    """Per file, per group and overall; `corpus_docs_hit` unions over the files of a group."""
    groups: dict[str, dict] = {}
    for r in sorted(results, key=lambda r: r["rel"]):
        g = groups.setdefault(r["group"], {"files": [], "corpus_docs": set()} | dict.fromkeys(AGG_KEYS, 0))
        g["files"].append(r["rel"])
        for k in AGG_KEYS:
            g[k] += r[k]
        g["corpus_docs"].update(r["corpus_docs_hit"])
    train_docs: set[int] = set()
    all_docs: set[int] = set()
    for name, g in groups.items():
        all_docs |= g["corpus_docs"]
        if name != "eval_windows":
            train_docs |= g["corpus_docs"]
    totals = {k: sum(g[k] for g in groups.values()) for k in AGG_KEYS}
    totals["corpus_docs_hit_any"] = len(all_docs)
    totals["corpus_docs_hit_training"] = len(train_docs)
    totals["corpus_docs_hit_eval_windows"] = len(groups.get("eval_windows", {}).get("corpus_docs", ()))
    totals["training_rows"] = sum(g["rows"] for n, g in groups.items() if n != "eval_windows")
    totals["training_rows_with_hit"] = sum(
        g["rows_with_hit"] for n, g in groups.items() if n != "eval_windows"
    )
    return {
        "convention": (
            f"NFKC -> lowercase -> whitespace split -> every {N}-word shingle -> xxh3_64 of those "
            "words joined by single spaces; a row shorter than 13 words yields no shingle and "
            "cannot be detected by this check"
        ),
        "files": {
            r["rel"]: {k: v for k, v in r.items() if k not in DROP_KEYS}
            | {"corpus_docs_hit": len(r["corpus_docs_hit"])}
            for r in sorted(results, key=lambda r: r["rel"])
        },
        "groups": {
            n: {k: (len(v) if k == "corpus_docs" else v) for k, v in g.items()}
            for n, g in sorted(groups.items())
        },
        "totals": totals,
        "targets": targets,
        "examples": sorted(
            (e for r in results for e in r["examples"]),
            key=lambda e: (-e["hit_fraction"], -e["shingle_hits"]),
        )[:5],
        "note_shared_windows": (
            "sae2m and sae2m_dec of a mix are the same windows with different directions, so their "
            "rows are counted twice here; likewise legacy sae/sae_dec"
        ),
    }


def _summary_line(report: dict) -> str:
    a, b = report["part_a"], report["part_b"]["totals"]
    c = report["part_b"]["coverage"]["training"]
    return (
        f"{README_MARK} by stream index, checked against heldout/doc_ids/ on 2026-09-18; "
        f"13-gram overlap with her training rows: {b['training_rows_with_hit']}/{b['training_rows']} rows "
        f"touch us, {b['corpus_docs_hit_training']}/{a['corpus_docs']} of our documents share a 13-gram "
        f"({100 * b['corpus_docs_hit_training'] / a['corpus_docs']:.2f}%), but only "
        f"{c['corpus_shingles_fraction'] * 100:.3f}% of our 13-grams are reached and "
        f"{c['docs_coverage_ge_0.2']} documents are covered above 20% "
        f"(infra/check_v2_disjointness.py)"
    )


@app.local_entrypoint()
def main(
    out: str = "",
    local_v2: str = "",
    local_corpus: str = "",
    verify_starts: bool = True,
    skip_part_b: bool = False,
    write: bool = True,
):
    t0 = time.time()
    _selftest()
    a_local = None
    if local_v2 and local_corpus:  # the small files, before anything is launched
        a_local = part_a(Path(local_corpus), Path(local_v2), dict(PART_START))
        print("[part A local]", json.dumps(a_local["sources"], indent=1), flush=True)
        print(
            f"[part A local] intersection {a_local['total_intersection']}, "
            f"min distance {a_local['min_distance_over_sources']}",
            flush=True,
        )
    a = run_part_a.remote(verify_starts)
    print("[part A volume]", json.dumps({k: v for k, v in a.items() if k != "sources"}, indent=1))
    print("[part A volume sources]", json.dumps(a["sources"], indent=1))
    assert a["disjoint"], f"PART A FAILED: {a['total_intersection']} corpus docs in v2 document sets"
    if a_local is not None:
        assert a["sources"] == a_local["sources"], "local and volume part A disagree"

    report = {"bundle": BUNDLE, "date": "2026-09-18", "part_a": a}
    if not skip_part_b:
        corpus = build_corpus_shingles.remote()
        print("[part B corpus]", json.dumps(corpus, indent=1))
        cp, cn = corpus["control_positive"], corpus["control_negative_shuffled"]
        assert cp["hits"] == cp["shingles"] and cp["shingles"] > 0, f"positive control failed: {cp}"
        assert cn["hits"] <= cn["shingles"] // 100, f"negative control failed: {cn}"
        targets = list_targets.remote()
        print(f"[part B] {len(targets)} files, {sum(t['rows'] for t in targets)} rows", flush=True)
        results = list(scan_file.map(targets, order_outputs=False))
        coverage = summarise_corpus_hits.remote(results)
        print("[part B coverage]", json.dumps(coverage, indent=1))
        report["part_b"] = {"corpus": corpus, "coverage": coverage, **_aggregate(results, targets)}
        print("[part B groups]", json.dumps(report["part_b"]["groups"], indent=1))
        print("[part B totals]", json.dumps(report["part_b"]["totals"], indent=1))
        report["summary"] = _summary_line(report)
        print("[summary]", report["summary"])
        if write:
            print(write_report.remote(report, report["summary"]))
    report["wall_seconds"] = round(time.time() - t0, 1)
    if out:
        Path(out).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"[report] {out}")
    print(f"done in {report['wall_seconds']:.0f}s")


if __name__ == "__main__":  # part A alone, against local copies; no Modal, no volume
    import argparse

    ap = argparse.ArgumentParser(description="part A of the v2 disjointness check, locally")
    ap.add_argument("--v2", required=True, help="local copy of the v2-2026-09-17 dir")
    ap.add_argument("--corpus", required=True, help="local copy of base/qwen36-27b/corpus")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-verify-starts", action="store_true")
    args = ap.parse_args()
    ps = dict(PART_START)
    if not args.no_verify_starts:
        v = verify_part_starts()
        print(f"[starts] footers: {v['part_rows']}")
        assert v["part_start"] == PART_START, f"footers give {v['part_start']}, constant {PART_START}"
        ps = v["part_start"]
    rep = part_a(Path(args.corpus), Path(args.v2), ps)
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1, sort_keys=True))
