"""Stage `retrieval` (methodology §3.6): the corpus-search reader. Its query is the direction MAEMM is given,
`unit(h_42 - mu)`; its eight "samples" are the best non-overlapping corpus windows by cos(raw h_t, query),
best first (evals/downstream/common/retrieval.py). Deterministic, so it has no greedy row; nothing is excluded, since
the items are the lens sets' prompts, not corpus text.

`--shard k/n` (0-based) scores a contiguous block of windows per container and keeps each query's top
candidates; the call that finds every part on disk merges them. On Modal the merge is a separate
`--shard 0/<n> --merge` call, which loads no model and refuses if a part is missing."""

import time

import numpy as np

from evals.downstream.common import retrieval as R
from evals.downstream.workspace_understanding import config as C
from evals.downstream.workspace_understanding.corpus import CORPUS_NPZ, load_tokenizer
from evals.downstream.workspace_understanding.model import direction, load_base, centring_mean
from evals.downstream.common.runs import ChainedHash, mark_stage, stage_done
from evals.downstream.workspace_understanding.runs import PART_RECORD, RETRIEVAL_PART, stage_key, write_provenance

SCORES_PART = "retrieval/scores.part{k}of{n}.npz"
PART_STAGE = PART_RECORD  # "retrieval_part{k}of{n}": the name the merge's key finds each part's record by
READOUTS_REL = "rollouts/retrieval.json"


def score_windows(windows, Q, mdl, tok, device, batch=R.WINDOW_BATCH, reencode=None, mu=None):
    """The shared scoring rule for a block of windows against every query (the GPU step)."""
    return R.window_cos(windows, Q, mdl, tok, device, batch=batch, reencode=reencode, mu=mu)


def load_corpus(run):
    """The built corpus: its documents' token ids and the 64/16 window grid over them."""
    return R.load_corpus(run, CORPUS_NPZ, C.CORPUS)


def kept_ids(run):
    """The kept items' ids, in the order every query matrix and every merged row is built in."""
    return [x["i"] for x in run.read_json("data/items.json")["items"] if not x["excluded"]]


def queries(run, mu):
    """The [n_items, d_model] query matrix `unit(h_42 - mu)`, built from the saved clean capture."""
    H = np.load(run.file("activations/h_all.npz"))["h"]
    return np.stack([direction(H[i, C.READ_LAYER], mu) for i in kept_ids(run)])


def merge_requested(args):
    """`--merge`: this call is the designated merge; a part still missing then fails rather than waits."""
    return bool(getattr(args, "_merge", False))


def shard_part(args, n_windows):
    """(part index, split, window block) from `--shard k/n`; no flag is part 0 of 1."""
    spec = getattr(args, "shard", None)
    k, n = (0, 1) if spec in (None, "") else (int(x) for x in str(spec).split("/"))
    return k, n, R.shard_block(n_windows, k, n)


def part_key(base_hash, k, n, sl):
    """One part's resume key, chained to what the search reads."""
    return ChainedHash(R.part_config_hash(base_hash, k, n, sl), base_hash.upstream)


def score_part(args, run, base_hash, corpus, k, n, sl):
    """Search this container's block of the corpus; write its part file and record."""
    started = time.time()
    mu = centring_mean()
    Q = queries(run, mu)
    base, tok = load_base(args.device)
    # raw residuals against the centred query; the shared prefix's top-k is kept beside the full one
    top, n_scored, n_excluded = R.score_corpus_topk(
        run, corpus, args.device, base, tok, Q, sl, C.CORPUS.top_k,
        metric=C.CORPUS_METRIC, score=score_windows,
        prefix_windows=R.shared_windows(corpus.docs, C.CORPUS),
    )
    R.save_topk_part(run, SCORES_PART.format(k=k, n=n), sl, top)
    rec = {"shard": f"{k}/{n}", "start": int(sl.start), "stop": int(sl.stop), "n_scored": n_scored,
           "n_excluded": n_excluded, "n_windows": len(corpus), "n_items": int(Q.shape[0]),
           "top_k": C.CORPUS.top_k, "metric": C.CORPUS_METRIC, "seconds": time.time() - started}
    stage = PART_STAGE.format(k=k, n=n)
    mark_stage(run, stage, part_key(base_hash, k, n, sl), rec, started=started)
    # under the part's own name: sibling shards write provenance concurrently
    write_provenance(run, {stage: rec}, stage=stage)
    print(f"[retrieval] shard {k}/{n}: windows [{sl.start}, {sl.stop}) x {Q.shape[0]} directions in "
          f"{rec['seconds']:.0f}s", flush=True)


def merge(args, run, corpus, n):
    """Merge and rank every part, decode the kept windows with the corpus tokenizer (no model load) and
    write rollouts/retrieval.json: per item its windows best first (`samples`), `greedy` None, and the same
    ranking over the shared prefix (`shared`) for the search cosine alone."""
    started = time.time()
    ids = kept_ids(run)
    n_windows = len(corpus)
    top = R.merge_topk_parts(run, n, corpus, SCORES_PART, C.CORPUS.top_k)
    if top.prefix is None:
        raise ValueError(f"the parts of the {n}-way split carry no {R.SHARED_SIZE} ranking; re-score them "
                         f"with `retrieval --shard <k>/{n} --force`")
    tok = load_tokenizer()
    hits, shared = R.top_windows(corpus, top, tok), R.top_windows(corpus, top.prefix, tok)
    if len(hits) != len(ids):
        raise ValueError(f"the merged search holds {len(hits)} rows and the run keeps {len(ids)} items")
    window = lambda w: {"text": w["text"], "rank": w["rank"], "score": w["score"],
                        "search_cos": w["search_cos"], "k": w["k"], "window_id": w["window_id"],
                        "doc": w["doc"], "start": w["start"], "n_tokens": w["n_tokens"]}
    recs = [{"i": i, "greedy": None, "samples": [window(w) for w in row],
             "shared": [window(w) for w in row8]}
            for i, row, row8 in zip(ids, hits, shared)]
    search = R.search_rows(top, n_windows, sum(d.n_tokens for d in corpus.docs))
    top1 = [row[0]["score"] for row in hits if row]
    # a top-1 cosine near 1 would mean a near copy of the query's text in the corpus: counted, not raised
    check = {"n_items": len(ids), "n_windows": n_windows, "n_shards": int(n), "top_k": C.CORPUS.top_k,
             "near_duplicates": R.near_duplicates(top1, C.CORPUS.near_duplicate_cos),
             "near_duplicate_cos": C.CORPUS.near_duplicate_cos,
             "rows_short_of_top_k": sum(1 for row in hits if len(row) < C.CORPUS.top_k),
             "suppression": C.CORPUS.suppression,
             "mean_suppressed": float(np.mean(top.suppressed)) if top.suppressed else None,
             "mean_top1_score": float(np.mean(top1)) if top1 else None,
             "distinct_top1_windows": len({row[0]["window_id"] for row in hits if row}),
             "seconds": time.time() - started}
    run.write_json(READOUTS_REL, {
        "config": {"corpus": C.CORPUS.record(), "metric": C.CORPUS_METRIC, "read_layer": C.READ_LAYER,
                   "model_revision": C.MODEL_REVISION, "n_windows": n_windows, "shards": int(n),
                   "excluded_docs": [], "search_check": check, "search": search},
        "items": recs,
    })
    write_provenance(run, {"retrieval_search_check": check}, stage="retrieval")
    mark_stage(run, "retrieval", stage_key("retrieval", args, run, n_parts=n), check, started=started)
    passed = check["mean_suppressed"]
    print(f"[retrieval] {n_windows} windows x {len(ids)} directions over {n} part(s); distinct top-1 "
          f"windows {check['distinct_top1_windows']}/{len(ids)}; near-duplicates "
          f"{check['near_duplicates']}; overlapping candidates passed over per direction "
          f"{'n/a' if passed is None else format(passed, '.1f')}", flush=True)


def stage_retrieval(args, run):
    # `base_hash` keys every part; the stage record is the merge's, chained to the parts it ranked
    base_hash = stage_key(RETRIEVAL_PART, args, run)
    if (stage_done(run, "retrieval", stage_key("retrieval", args, run)) and run.exists(READOUTS_REL)
            and not args.force):
        print("[retrieval] up to date")
        return
    corpus = load_corpus(run)
    n_windows = len(corpus)
    k, n, sl = shard_part(args, n_windows)
    chash = part_key(base_hash, k, n, sl)
    stage = PART_STAGE.format(k=k, n=n)
    # the designated merge call scores nothing; a missing part there fails (`R.refuse_or_wait`)
    if merge_requested(args):
        print(f"[retrieval] merge call: ranking the parts of the {n}-way split")
    elif args.force or not (stage_done(run, stage, chash) and run.exists(SCORES_PART.format(k=k, n=n))):
        score_part(args, run, base_hash, corpus, k, n, sl)
    else:
        print(f"[retrieval] shard {k}/{n} up to date")
    done, waiting = R.part_states(run, n, n_windows, SCORES_PART, PART_STAGE, base_hash)
    if waiting:
        R.refuse_or_wait(merge_requested(args), "retrieval", "retrieval", n, done, waiting)
        return
    merge(args, run, corpus, n)
