"""The corpus-search reader (methodology "Readers"), three stages:

  `corpus`           CPU: rebuilds the held-out corpus's search prefix (`config.CORPUS_SPEC`) from the pinned
                     public files by the shipped document index, into retrieval/corpus.npz.
  `retrieval`        GPU, sharded by window: every query against a block of the corpus on the clean base,
                     keeping each query's candidates in retrieval/scores.part<k>of<n>.npz.
  `retrieval_merge`  CPU: folds the parts, keeps the best non-overlapping windows per query and decodes them
                     into retrieval/windows.json.

One query per (item, read position): unit(h42 − mu). A window scores max over its positions of cos(h_t, q)
on the raw residual (`evals.downstream.common.retrieval`, metric "raw"). Nothing is excluded from the corpus: this
evaluation's materials are not web text. The search is deterministic: no greedy row."""

import argparse
import time

import numpy as np

from evals.downstream.common import background as BG
from evals.downstream.common import retrieval as R
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_modulation import config as C
from evals.downstream.workspace_modulation import mean_cell as M
from evals.downstream.workspace_modulation.items import load_tokenizer
from evals.downstream.workspace_modulation.rollouts import row_index
from evals.downstream.workspace_modulation.runs import stage_key, write_provenance
from evals.downstream.workspace_modulation.stages import provenance_stage, shard_of, stage_record

# The corpus spec; `config.CORPUS` is its JSON form.
SPEC = C.CORPUS_SPEC
#: the built corpus: its documents' token ids, and the metadata that says which documents they are
CORPUS_NPZ = "retrieval/corpus.npz"
CORPUS_JSON = "retrieval/corpus.json"
#: one container's block of the corpus, as the running top-k it kept
PART_REL = "retrieval/scores.part{k}of{n}.npz"
#: the reader every instrument reads, in the shape `rollouts/<arm>.json` has, so an item's cells and their
#: texts are addressed by the same helpers
MERGED_REL = "retrieval/windows.json"


def corpus_docs(smoke):
    """The held-out documents the corpus is built from: the search prefix, or a prefix of it under --smoke."""
    docs = R.search_docs(R.load_index(), SPEC)
    return docs[: C.SMOKE_CORPUS_DOCS] if smoke else docs


def query_rows(run, kept):
    """[(row in the activation store, item, cell, band)] for every query: both read positions of every kept item."""
    row_of = row_index(M.full_table(run))
    out = []
    for it in kept:
        for p in M.read_positions(it):
            band = C.MEAN_BAND if int(p) == C.MEAN_POS else C.FINAL_BAND
            out.append((row_of[(int(it["i"]), int(p))], int(it["i"]), int(p), band))
    return out


def queries_of(run, rows):
    """The [n_queries, d_model] matrix of unit queries in `rows` order: `unit(h42 − mu)`, exactly the vector
    the inverter is injected with at that cell."""
    from evals.downstream.common.model_io import direction

    H = M.load_h(run)
    mu = BG.load_centring_mean()
    return np.stack([direction(H[r], mu) for r, _i, _p, _b in rows]).astype(np.float32)


def stage_corpus(args, run):
    chash = stage_key("corpus", args, run)
    if stage_done(run, "corpus", chash) and not args.force:
        print("[corpus] up to date")
        return
    started = time.time()
    docs = corpus_docs(args.smoke)
    meta = R.build_corpus(run, SPEC, docs, R.open_parts(SPEC.dataset), load_tokenizer(),
                          CORPUS_NPZ, CORPUS_JSON, sizes=())
    write_provenance(run, {"corpus": meta}, stage="corpus")
    mark_stage(
        run,
        "corpus",
        chash,
        {"n_windows": meta["n_windows"], "n_docs": meta["n_docs"], "n_tokens": meta["n_tokens"],
         "doc_min": meta["doc_min"], "doc_max": meta["doc_max"], "smoke": bool(args.smoke)},
        started=started,
    )
    print(
        f"[corpus] {meta['n_windows']} windows of up to {SPEC.window} tokens at stride {SPEC.stride} over "
        f"{meta['n_docs']} documents ({meta['n_tokens']} tokens, held-out documents {meta['doc_min']}-"
        f"{meta['doc_max']}) in {time.time() - started:.0f}s",
        flush=True,
    )


def stage_retrieval(args, run):
    """One container's block of the corpus, scored against every query and stored as its own top-k."""
    k, n = shard_of(args)
    name = stage_record("retrieval", k, n)
    chash = stage_key("retrieval", args, run)
    if stage_done(run, name, chash) and not args.force:
        print(f"[retrieval:{k}of{n}] up to date")
        return
    started = time.time()
    from evals.downstream.common.model_io import load_base

    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    rows = query_rows(run, kept)
    queries = queries_of(run, rows)
    corpus = R.load_corpus(run, CORPUS_NPZ, SPEC)
    sl = R.shard_block(len(corpus), k, n)
    # nothing excluded; `prefix_windows` keeps the shared size's own top-k
    base, tok = load_base(args.device, C.MODEL, C.MODEL_REVISION)
    top, n_scored, n_excluded = R.score_corpus_topk(
        run, corpus, args.device, base, tok, queries, sl, C.TOP_WINDOWS, metric="raw", exclude_docs=(),
        prefix_windows=R.shared_windows(corpus.docs, SPEC),
    )
    R.save_topk_part(run, PART_REL.format(k=k, n=n), sl, top)
    rec = {
        "shard": [k, n],
        "start": int(sl.start),
        "stop": int(sl.stop),
        "n_windows": len(corpus),
        "n_scored": int(n_scored),
        "n_excluded": int(n_excluded),
        "n_queries": len(rows),
        "top_k": C.TOP_WINDOWS,
        "metric": "raw",
        "seconds": time.time() - started,
    }
    write_provenance(run, {f"retrieval_shard{k}of{n}": rec}, stage=provenance_stage("retrieval", k, n))
    mark_stage(run, name, chash, rec, started=started)
    print(
        f"[retrieval:{k}of{n}] windows [{sl.start}, {sl.stop}) x {len(rows)} queries, keeping the "
        f"{top.depth} candidates the top {C.TOP_WINDOWS} of each are taken out of, in {rec['seconds']:.0f}s",
        flush=True,
    )


def merged_doc(run, rows, top, kept, corpus, tok, n):
    """retrieval/windows.json from a finished top-k, in the shape `rollouts/<arm>.json` has: each cell's
    windows as `samples`, best first, sharing no token (`R.suppress_overlapping`); `shared` is the same ranking
    over the `R.SHARED_SIZE` prefix, and `config.search` the search cosine at both sizes."""
    if top.prefix is None:
        raise RuntimeError(
            f"retrieval_merge: the parts carry no {R.SHARED_SIZE} ranking; re-run `python -m "
            "evals.downstream.workspace_modulation retrieval --force` for each shard and merge again"
        )
    windows = R.top_windows(corpus, top, tok)
    shared = R.top_windows(corpus, top.prefix, tok)
    by_i = {}
    for j, (_r, i, p, band) in enumerate(rows):
        by_i.setdefault(i, []).append(
            {
                "pos": p,
                "band": band,
                # deterministic: no greedy row
                "greedy": None,
                "samples": windows[j],
                "shared": shared[j],
            }
        )
    meta = run.read_json(CORPUS_JSON)
    top1 = [row[0]["score"] for row in windows if row]
    return {
        "config": {
            "reader": C.RETRIEVAL_READER,
            "corpus": meta,
            "top_k": C.TOP_WINDOWS,
            "metric": "raw",
            "n_shards": n,
            "n_windows": len(corpus),
            "n_queries": len(rows),
            # a top-1 cosine this high may be a near copy of the query's own text
            "near_duplicate_cos": SPEC.near_duplicate_cos,
            "n_near_duplicates": R.near_duplicates(top1, SPEC.near_duplicate_cos),
            "suppression": SPEC.suppression,
            "mean_suppressed": float(np.mean(top.suppressed)) if top.suppressed else None,
            "search": R.search_rows(top, len(corpus), meta["n_tokens"]),
        },
        "items": [{"i": it["i"], "cells": by_i[it["i"]]} for it in kept],
    }


def unusable_parts(args, run, n):
    """The shards of an n-way split the merge may not fold: the part is absent, or its shard record does not
    carry the key this invocation resolves (a part scored against another corpus, positions or model)."""
    out = []
    for k in range(n):
        sub = argparse.Namespace(**vars(args))
        sub.shard, sub.n_shards = k, n
        ready = run.exists(PART_REL.format(k=k, n=n)) and stage_done(
            run, stage_record("retrieval", k, n), stage_key("retrieval", sub, run)
        )
        if not ready:
            out.append(k)
    return out


def stage_retrieval_merge(args, run):
    n = int(getattr(args, "n_shards", 1) or 1)
    chash = stage_key("retrieval_merge", args, run, n_shards=n)
    if stage_done(run, "retrieval_merge", chash) and not args.force:
        print("[retrieval_merge] up to date")
        return
    started = time.time()
    unusable = unusable_parts(args, run, n)
    if unusable:
        raise RuntimeError(
            f"retrieval_merge: part(s) {unusable} of the {n}-way split are missing, or were scored under a "
            f"different corpus, read position or model pin than this invocation resolves. Every part is a "
            f"contiguous block of ONE corpus and the ranking is over all of them, so a merge short of one "
            f"would publish the best windows of a shorter corpus and a merge of one scored elsewhere would "
            f"publish this corpus's text at another corpus's ranks: re-run `python -m "
            f"evals.downstream.workspace_modulation retrieval --shard <k> --n-shards {n}` for each and merge again"
        )
    kept = [x for x in run.read_json("data/items.json")["items"] if not x["excluded"]]
    rows = query_rows(run, kept)
    corpus = R.load_corpus(run, CORPUS_NPZ, SPEC)
    top = R.merge_topk_parts(run, n, corpus, PART_REL, C.TOP_WINDOWS)
    if top.n_queries != len(rows):
        raise RuntimeError(
            f"retrieval_merge: the parts hold {top.n_queries} queries and this run has {len(rows)}; the "
            "parts were scored against a different population"
        )
    doc = merged_doc(run, rows, top, kept, corpus, load_tokenizer(), n)
    run.write_json(MERGED_REL, doc)
    cfg = doc["config"]
    write_provenance(run, {"retrieval": cfg}, stage="retrieval_merge")
    mark_stage(
        run,
        "retrieval_merge",
        chash,
        {"n_items": len(doc["items"]), "n_queries": len(rows), "n_windows": len(corpus), "n_shards": n,
         "n_near_duplicates": cfg["n_near_duplicates"], "mean_suppressed": cfg["mean_suppressed"]},
        started=started,
    )
    print(
        f"[retrieval_merge] {len(rows)} queries x top {C.TOP_WINDOWS} over {len(corpus)} windows from {n} "
        f"part(s); {cfg['n_near_duplicates']} query(ies) whose best window scores above "
        f"{cfg['near_duplicate_cos']}; "
        f"{'n/a' if cfg['mean_suppressed'] is None else format(cfg['mean_suppressed'], '.1f')} overlapping "
        f"candidates passed over per query",
        flush=True,
    )
