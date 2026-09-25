"""Stage `prepare` (methodology §2 step 1, §3): the candidate documents, from the evaluation half of the
held-out corpus, and their seeded read positions. A `doc` is a held-out document index. CPU + tokenizer."""
import time

import numpy as np

from evals.downstream.common import retrieval as R
from evals.downstream.rollout_coherence import config as C
from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance


def draw_positions(n_sources, seed, n_spare=0):
    """`(positions, spare positions, live generator)` from one generator, in that order (methodology §3)."""
    rng = np.random.default_rng(seed)
    pos = [int(x) for x in rng.integers(C.POS_LO, C.POS_HI + 1, size=n_sources)]
    spare = [int(x) for x in rng.integers(C.POS_LO, C.POS_HI + 1, size=n_spare)]
    return pos, spare, rng


def rng_after_prepare(n_sources, n_spare, seed):
    """`draw_positions`' generator advanced past its draws, for `capture`'s norm presample."""
    return draw_positions(n_sources, seed, n_spare)[2]


def source_passage_ids(ids, t, n):
    """The last `n` ids ending at `t` inclusive (clamped at the start of `ids`)."""
    return list(ids[max(0, t - n + 1): t + 1])


def build_documents(sources, source_ids, positions, spare=(), spare_ids=(), spare_positions=()):
    """`data/documents.json`: `sources` then `spare`, one candidate pool (`pool_row`) that `capture`
    selects from."""
    n = len(sources)
    return {
        "sources": [
            {"i": i, "id": f"act/{i:03d}", "doc": source_ids[i], "ids": sources[i], "pos": positions[i],
             "pool_row": i}
            for i in range(n)
        ],
        "spare": [
            {"doc": spare_ids[j], "ids": spare[j], "pos": spare_positions[j], "pool_row": n + j}
            for j in range(len(spare))
        ],
        "rejected": [],
    }


def pool_rows(doc):
    """Every candidate in pool order, whichever list `capture` left it in."""
    rows = list(doc["sources"]) + list(doc.get("rejected") or []) + list(doc.get("spare") or [])
    return sorted(rows, key=lambda r: r["pool_row"])


def open_parts():
    """The held-out corpus's parquet files via the shared opener: the one download."""
    return R.open_parts(C.CORPUS)


def draw_documents(n, tok, exclude=None):
    """`(first C.SEQ_LEN token ids, held-out indices)` of `n` documents of at least `C.SEQ_LEN` tokens from
    the shared seeded draw over the evaluation half (a smaller draw is a prefix of a larger one), skipping
    `exclude` (default: the training-overlap check) in draw order."""
    exclude = R.overlap_excluded() if exclude is None else exclude
    docs = R.select_documents(R.evaluation_docs(R.load_index()), n, C.DOC_SEED,
                              min_tokens=C.SEQ_LEN, exclude=exclude)
    parts = open_parts()
    rows = [R.document_ids(parts, d, tok)[:C.SEQ_LEN] for d in docs]
    return rows, [int(d.doc) for d in docs]


def overlap_record():
    """The content check `prepare` draws under: threshold, asset digest and documents excluded."""
    return R.overlap_record(R.evaluation_docs(R.load_index()), min_tokens=C.SEQ_LEN)


def prepare_config(smoke):
    """What `prepare`'s resume key is taken over."""
    return {
        "sizes": C.sizes(smoke), "corpus": C.CORPUS, "seq_len": C.SEQ_LEN, "doc_seed": C.DOC_SEED,
        "seed": C.DATA_SEED, "draw": C.DRAW, "scoring": C.SCORING_VERSION, "model": C.MODEL,
        "model_revision": C.MODEL_REVISION, "overlap": overlap_record(),
    }


def load_tokenizer():
    """The base model's tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_REVISION)


def stage_prepare(args, run):
    sz = C.sizes(args.smoke)
    overlap = overlap_record()
    chash = config_hash(prepare_config(args.smoke))
    if stage_done(run, "prepare", chash) and not args.force:
        print("[prepare] up to date", flush=True)
        return
    tok = load_tokenizer()
    t0 = time.time()
    ns = sz["n_sources"]
    n_spare = sz["n_pool"] - ns
    rows, ids = draw_documents(ns + n_spare, tok)
    pos, spare_pos, _ = draw_positions(ns, C.DATA_SEED, n_spare)
    doc = build_documents(rows[:ns], ids[:ns], pos, rows[ns:], ids[ns:], spare_pos)
    doc["config"] = {
        "sizes": sz, "corpus": C.CORPUS, "seq_len": C.SEQ_LEN, "doc_seed": C.DOC_SEED,
        "seed": C.DATA_SEED, "pos_range": [C.POS_LO, C.POS_HI], "draw": C.DRAW, "overlap": overlap,
    }
    run.write_json("data/documents.json", doc)
    write_provenance(run, {"prepare_seconds": time.time() - t0}, stage="prepare")
    mark_stage(run, "prepare", chash, {"n_sources": ns, "n_spare": n_spare,
                                       "doc_min": min(ids), "doc_max": max(ids), "overlap": overlap},
               started=t0)
    print(
        f"[prepare] {ns} sources + {n_spare} spare candidates from the held-out corpus's evaluation half "
        f"(documents {min(ids)}-{max(ids)}; {overlap['n_excluded']} of its {overlap['n_pool']} documents of "
        f"{C.SEQ_LEN} tokens excluded at 7-gram coverage >= {overlap['threshold']}) in {time.time() - t0:.0f}s",
        flush=True,
    )
