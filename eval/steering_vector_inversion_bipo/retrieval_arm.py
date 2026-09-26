"""The corpus-search reader: each direction's eight best non-overlapping windows of the shared corpus, one
bundle. The corpus, scoring rule, top-k, shard split and merge are `eval/common/retrieval.py`'s; every
vector is queried at once, ranked on the uncentred cosine, and a shard's windows travel as token ids.
"""
import time

import numpy as np

from eval.common import retrieval as R
from eval.common.runs import RunDir, config_hash, mark_stage, stage_done

from . import config as C

OPERATION = "bipo_corpus_search"
ARM = "retrieval"
METRIC = "raw"
RANDOM_QUERIES = 8                     # for the search's no-signal cosine only, never judged

CORPUS_STAGE = "bipo-corpus-cut"
CORPUS_NPZ = "retrieval/corpus.npz"
CORPUS_JSON = "retrieval/corpus.json"
QUERIES = "retrieval/queries.json"
SEARCH = "retrieval/search.json"       # the search's own cosine per vector kind
PART = "retrieval/scores.part{k}of{n}.npz"
PART_STAGE = "bipo-retrieval-part{k}of{n}"
CLI = "python -m eval.steering_vector_inversion_bipo retrieval"


def corpus_docs(smoke, spec=None, index=None):
    """The held-out documents the corpus is built from: the search prefix, or its first `C.SMOKE_DOCS`."""
    docs = R.search_docs(index or R.load_index(), spec or C.CORPUS)
    return docs[:C.SMOKE_DOCS] if smoke else docs


def open_parts(spec=None):
    """The two pinned public parquet files the corpus is rebuilt from."""
    return R.open_parts((spec or C.CORPUS).dataset)


def load_tokenizer(config, cache_dir):
    """The model's tokenizer."""
    from eval.steering_vector_inversion.data import load_tokenizer as load

    return load(config, cache_dir)


def corpus_hash(config, docs, spec=None):
    """The corpus cut's key: the shared spec, the documents it covers and the tokenizer."""
    docs = tuple(docs)
    return config_hash({"spec": (spec or C.CORPUS).record(), "n_docs": len(docs),
                        "doc_max": int(docs[-1].doc) if docs else -1,
                        "model": [config.model, config.model_revision]})


def build(root, config, cache_dir, smoke=False, spec=None):
    """Rebuild the corpus into `retrieval/corpus.{npz,json}` on CPU, unless built under the same key; its
    metadata."""
    spec = spec or C.CORPUS
    run = RunDir(str(root))
    docs = corpus_docs(smoke, spec)
    chash = corpus_hash(config, docs, spec)
    if stage_done(run, CORPUS_STAGE, chash) and run.exists(CORPUS_NPZ):
        return run.read_json(CORPUS_JSON)
    started = time.time()
    meta = R.build_corpus(run, spec, docs,
                          open_parts(spec), load_tokenizer(config, cache_dir),
                          CORPUS_NPZ, CORPUS_JSON, sizes=())
    mark_stage(run, CORPUS_STAGE, chash,
               {"n_windows": meta["n_windows"], "n_docs": meta["n_docs"], "n_tokens": meta["n_tokens"]},
               started=started)
    return meta


def load_corpus(root, spec=None):
    """The built corpus: its token ids and window grid."""
    return R.load_corpus(RunDir(str(root)), CORPUS_NPZ, spec or C.CORPUS)


def queries(bank):
    """`(vector ids, [n, d_model] unit queries)`: the bank, then `RANDOM_QUERIES` seeded Gaussian directions
    (`random/<k>`)."""
    from eval.common.directions import gaussian_directions
    matrix = np.asarray([e["direction"] for e in bank], dtype=np.float32)
    random = np.asarray(gaussian_directions(RANDOM_QUERIES, matrix.shape[1]), dtype=np.float32)
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    ids = [e["vector_id"] for e in bank] + [f"random/{k}" for k in range(RANDOM_QUERIES)]
    return ids, np.concatenate([matrix, random])


def base_hash(run, config, ids, code, spec=None):
    """The search's resume key, which every part chains to."""
    meta = run.read_json(CORPUS_JSON)
    return config_hash({"spec": (spec or C.CORPUS).record(),
                        "corpus": {name: meta[name] for name in ("n_docs", "n_windows", "doc_max")},
                        "queries": list(ids), "code": code, "metric": METRIC,
                        "shared_size": R.SHARED_SIZE,
                        "model": [config.model, config.model_revision]})


def window_block(corpus, block):
    """One block's windows as `(flat int32 ids, offsets)`: window `i` is `ids[offsets[i]:offsets[i + 1]]`."""
    rows = [np.asarray(corpus.window_ids(k), dtype=np.int32) for k in block]
    offsets = np.cumsum([0, *(len(row) for row in rows)], dtype=np.int64)
    return (np.concatenate(rows) if rows else np.zeros(0, dtype=np.int32)), offsets


def shard_tasks(run, corpus, ids, matrix, base, shards, spec=None, only=None):
    """One payload per block not already scored, lazily; `only` is a hand re-run's `--shard` index."""
    spec = spec or C.CORPUS
    n_windows = len(corpus)
    recorded = int(run.read_json(CORPUS_JSON)["n_windows"])
    if recorded != n_windows:
        raise ValueError(f"the corpus metadata reports {recorded} windows and {CORPUS_NPZ} holds "
                         f"{n_windows}: the shard split is computed from the first and would leave part "
                         f"of the corpus unscored")
    _done, waiting = R.part_states(run, shards, n_windows, PART, PART_STAGE, base)
    prefix_windows = R.shared_windows(corpus.docs, spec)
    for k in waiting:
        if only is not None and k != only:
            continue
        sl = R.shard_block(n_windows, k, shards)
        window_ids, offsets = window_block(corpus, sl)
        yield f"{k}/{shards}", {"operation": OPERATION, "shard": k, "shards": shards,
                                "start": sl.start, "stop": sl.stop, "top_k": int(spec.top_k),
                                "depth": R.candidate_depth(spec.top_k, spec.window, spec.stride),
                                "prefix_windows": prefix_windows,
                                "vector_ids": list(ids), "queries": matrix,
                                "window_ids": window_ids, "offsets": offsets}


def execute(worker, payload):
    """One block searched against every query on the clean base, keeping each query's `depth` best windows."""
    import torch
    started = time.time()
    window_ids = np.asarray(payload["window_ids"], dtype=np.int32)
    offsets = np.asarray(payload["offsets"], dtype=np.int64)
    n_windows, top_k = len(offsets) - 1, int(payload["top_k"])
    first = int(payload["start"])
    queries_matrix = np.asarray(payload["queries"], dtype=np.float32)
    top = R.TopK(queries_matrix.shape[0], top_k, prefix_windows=payload.get("prefix_windows"),
                 depth=payload.get("depth"))
    for start in range(0, n_windows, R.TOPK_CHUNK):
        stop = min(start + R.TOPK_CHUNK, n_windows)
        chunk = [window_ids[offsets[i]:offsets[i + 1]].tolist() for i in range(start, stop)]
        raw, _centred = R.window_cos(chunk, queries_matrix, worker.base, worker.tokenizer, worker.device)
        top.add(range(first + start, first + stop), raw)
    data = {"shard": int(payload["shard"]), "shards": int(payload["shards"]),
            "start": int(payload["start"]), "stop": int(payload["stop"]), "n_scored": n_windows,
            **top.arrays()}
    return {"data": data, "gpu_seconds": time.time() - started, "runtime_versions": worker.versions,
            "gpu_name": torch.cuda.get_device_name(worker.device) if worker.device.type == "cuda" else "CPU"}


def save_part(run, data, base, seconds):
    """One finished block's part file and stage record."""
    k, n = int(data["shard"]), int(data["shards"])
    sl = range(int(data["start"]), int(data["stop"]))
    R.save_part(run, PART.format(k=k, n=n), sl,
                {name: np.asarray(data[name]) for name in R.TOPK_KEYS if name in data})
    record = {"shard": f"{k}/{n}", "start": sl.start, "stop": sl.stop, "n_scored": int(data["n_scored"]),
              "seconds": float(seconds)}
    mark_stage(run, PART_STAGE.format(k=k, n=n), R.part_config_hash(base, k, n, sl), record)
    return record


def families(ids, hits):
    """One family per direction: its best windows, best first (`sample_id` is the rank)."""
    out = {}
    for vector_id, row in zip(ids, hits):
        family_id = f"{vector_id}|{ARM}"
        out[family_id] = {
            "family_id": family_id, "vector_id": vector_id, "arm": ARM,
            "samples": [{"sample_id": int(w["rank"]), "text": w["text"], "window_id": w["window_id"],
                         "doc": int(w["doc"]), "start": int(w["start"]), "n_tokens": int(w["n_tokens"]),
                         "score": float(w["score"]), "search_cos": float(w["search_cos"])}
                        for w in row]}
    return out


def merge(root, corpus, tok, base, shards, spec=None):
    """`(families, the merge's record)` once every block is on disk; writes `SEARCH`, the search's cosine per
    vector kind at the full corpus and the `R.SHARED_SIZE` prefix."""
    spec = spec or C.CORPUS
    run = RunDir(str(root))
    n_windows = len(corpus)
    done, waiting = R.part_states(run, shards, n_windows, PART, PART_STAGE, base)
    if waiting:
        R.refuse_or_wait(True, "retrieval-merge", CLI, shards, done, waiting)
    top = R.merge_topk_parts(run, shards, corpus, PART, int(spec.top_k))
    ids = run.read_json(QUERIES)
    if len(ids) != top.n_queries:
        raise ValueError(f"{QUERIES} names {len(ids)} directions and the merged parts hold "
                         f"{top.n_queries}; the search was run over a different bank")
    if top.prefix is None:
        raise ValueError(f"the parts carry no {R.SHARED_SIZE} ranking; they were not written by this search")
    hits = R.top_windows(corpus, top, tok)
    from .mc10 import vector_kind
    by_kind = {}
    for j, vector_id in enumerate(ids):
        by_kind.setdefault(vector_kind(vector_id), []).append(j)
    n_tokens = sum(d.n_tokens for d in corpus.docs)
    run.write_json(SEARCH, {
        "rows": [{"kind": kind, **row} for kind, js in by_kind.items()
                 for row in R.search_rows(top, n_windows, n_tokens, queries=js)],
        "shared": {vector_id: [{name: w[name] for name in ("rank", "window_id", "search_cos", "text")}
                               for w in row]
                   for vector_id, row in zip(ids, R.top_windows(corpus, top.prefix, tok))}})
    record = {"n_windows": n_windows, "n_queries": top.n_queries, "top_k": int(spec.top_k),
              "shards": shards, "metric": METRIC,
              "suppression": str(spec.suppression),
              "mean_suppressed": float(np.mean(top.suppressed)) if top.suppressed else None,
              "near_duplicates": R.near_duplicates([row[0]["score"] for row in hits if row],
                                                   spec.near_duplicate_cos),
              "near_duplicate_cos": float(spec.near_duplicate_cos)}
    return families([i for i in ids if not i.startswith("random/")],
                    [row for i, row in zip(ids, hits) if not i.startswith("random/")]), record
