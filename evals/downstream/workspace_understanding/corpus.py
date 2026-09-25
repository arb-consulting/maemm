"""Stage `corpus` (methodology §3.6): the suite's shared search corpus, rebuilt from the public parquet
files by the shipped held-out index (`evals.downstream.common.retrieval.build_corpus`). CPU and the base tokenizer only;
no document is excluded, since the items are the lens sets' prompts, not corpus text."""

import time

from evals.downstream.common import model_io
from evals.downstream.common import retrieval as R
from evals.downstream.workspace_understanding import config as C
from evals.downstream.common.runs import mark_stage, stage_done
from evals.downstream.workspace_understanding.runs import stage_key, write_provenance

CORPUS_NPZ = "retrieval/corpus.npz"
CORPUS_JSON = "retrieval/corpus.json"


def open_parts():
    """The pinned public parquet files (the one download)."""
    return R.open_parts(C.CORPUS.dataset)


def load_tokenizer():
    """The base model's tokenizer at the pinned revision; each document's token count is checked with it."""
    return model_io.load_tokenizer(C.MODEL, C.MODEL_REVISION)


def stage_corpus(args, run):
    chash = stage_key("corpus", args, run)
    if stage_done(run, "corpus", chash) and not args.force:
        print("[corpus] up to date")
        return
    started = time.time()
    docs = C.corpus_docs(args.smoke)
    meta = R.build_corpus(run, C.CORPUS, docs, open_parts(), load_tokenizer(), CORPUS_NPZ, CORPUS_JSON,
                          sizes=())
    write_provenance(run, {"corpus": meta}, stage="corpus")
    mark_stage(run, "corpus", chash,
               {"n_windows": meta["n_windows"], "n_docs": meta["n_docs"], "n_tokens": meta["n_tokens"],
                "smoke": bool(args.smoke)},
               started=started)
    print(f"[corpus] {meta['n_windows']} windows of up to {C.CORPUS.window} tokens over {meta['n_docs']} "
          f"documents ({meta['n_tokens']} tokens, held-out documents {meta['doc_min']}-{meta['doc_max']}) "
          f"in {time.time() - started:.0f}s", flush=True)
