"""Stage `frontier_corpus` (methodology §4): the window corpus the `retrieval` method searches, built by
`eval.common.retrieval.build_corpus` from the held-out corpus's parquet files. CPU + tokenizer only.

It covers the held-out corpus's SEARCH PREFIX (documents up to `CorpusSpec.corpus_tokens`), while `prepare`
draws from the documents after it, so the two are disjoint by construction; `check_disjoint` asserts it
before the build. Otherwise `retrieval` could return an activation's own text verbatim.
"""

import time

from eval.common import retrieval as R
from eval.rollout_coherence import config as C
from eval.rollout_coherence.documents import load_tokenizer, open_parts
from eval.common.runs import config_hash, mark_stage, stage_done, write_provenance
from eval.rollout_coherence.runs import stage_hashes

STAGE = "frontier_corpus"
#: the corpus's token ids and its metadata (documents, nested sizes), read by the retrieval method
CORPUS_NPZ = "frontier/corpus.npz"
CORPUS_JSON = "frontier/corpus.json"


def excluded_docs(doc):
    """The held-out documents the corpus may not contain: `prepare`'s sources, spare and rejected candidates
    (`capture` can promote a spare to a source, and this stage is chained to `prepare` only)."""
    return {"prepare sources": [s["doc"] for s in doc.get("sources", [])],
            "prepare spare candidates": [s["doc"] for s in (doc.get("spare") or [])],
            "prepare rejected candidates": [s["doc"] for s in (doc.get("rejected") or [])]}


def corpus_config_hash(run, args):
    """Chained to `prepare`'s record, since the corpus is checked disjoint from the documents it drew."""
    sizes, n_windows = C.corpus_sizes(args.smoke)
    return config_hash({
        "sizes": [[label, n] for label, n in sizes],
        "n_windows": n_windows,
        "spec": C.SEARCH_CORPUS.record(),
        "model": C.MODEL,
        "model_revision": C.MODEL_REVISION,
        "smoke": bool(args.smoke),
        "upstream": stage_hashes(run, ["prepare"]),
    })


def stage_frontier_corpus(args, run):
    doc = run.read_json("data/documents.json")
    chash = corpus_config_hash(run, args)
    if stage_done(run, STAGE, chash) and not args.force:
        print(f"[{STAGE}] up to date", flush=True)
        return
    started = time.time()
    docs = C.corpus_docs(args.smoke)
    # checked from the index alone, before the costly build
    R.check_disjoint(docs, excluded_docs(doc))
    meta = R.build_corpus(run, C.SEARCH_CORPUS, docs, open_parts(), load_tokenizer(), CORPUS_NPZ, CORPUS_JSON,
                          sizes=C.size_tokens(args.smoke))
    write_provenance(run, {"frontier_corpus_seconds": time.time() - started}, stage=STAGE)
    mark_stage(run, STAGE, chash,
               {"n_windows": meta["n_windows"], "n_docs": meta["n_docs"], "n_tokens": meta["n_tokens"],
                "smoke": bool(args.smoke)},
               started=started)
    print(f"[{STAGE}] {meta['n_windows']} windows of up to {C.WINDOW_TOKENS} tokens over "
          f"{meta['n_docs']} documents ({meta['n_tokens']} tokens, held-out documents "
          f"{meta['doc_min']}-{meta['doc_max']}) in {time.time() - started:.0f}s", flush=True)
