"""The corpus-search reader (`retrieval`): each direction's eight best non-overlapping corpus windows.

The corpus, window grid, scoring rule, shard split and merge are `evals.downstream.common.retrieval`'s. The queries are
the unit concept directions, ranked on the uncentred maximum-token cosine; `--shard k/n` splits the forward.
"""
import time

import numpy as np

from evals.downstream.common import retrieval as R
from evals.downstream.common.runs import config_hash, mark_stage, stage_done, write_provenance

from .artifacts import digest, source_hashes
from .data import load_tokenizer
from .execution import chunks

SPEC = R.CorpusSpec()
SMOKE_DOCS = 128                       # a smoke run searches the corpus's first documents
CORPUS_NPZ = 'retrieval/corpus.npz'
CORPUS_JSON = 'retrieval/corpus.json'
PART = 'retrieval/scores.part{k}of{n}.npz'
PART_STAGE = 'retrieval_part{k}of{n}'
WINDOWS = 'retrieval/windows.json'
OPERATION = 'corpus_search'
METRIC = 'raw'                         # the uncentred cosine
COMMAND = 'python -m evals.downstream.steering_vector_inversion retrieval --run-id <run-id>'


def documents(config):
    """The held-out documents searched: the search prefix, or its first `SMOKE_DOCS` in a smoke run."""
    docs = R.search_docs(R.load_index(), SPEC)
    return docs[:SMOKE_DOCS] if config.concepts_per_genre is not None else docs


def open_parts():
    """The two pinned public parquet files' text columns."""
    return R.open_parts(SPEC.dataset)


def module_hash():
    """The bytes of everything the search executes through, in every resume key of the search."""
    return digest(source_hashes(('corpus', 'model')))


def shard_of(shard):
    """`(k, n)` from a `k/n` string."""
    k, n = (int(part) for part in str(shard).split('/'))
    return k, n


def corpus_config_hash(run):
    """The corpus build's resume key: the definition, the documents and the tokenizer."""
    docs = documents(run.config)
    return config_hash({'spec': SPEC.record(), 'n_docs': len(docs), 'doc_max': int(docs[-1].doc),
                        'model': run.config.model, 'model_revision': run.config.model_revision})


def search_config_hash(run, queries):
    """The search's key, which every part chains to: the corpus, the queries, the rule and the code."""
    return config_hash({'corpus': corpus_config_hash(run), 'queries': digest(queries),
                        'top_k': SPEC.top_k, 'suppression': SPEC.suppression,
                        'candidate_depth': R.candidate_depth(SPEC.top_k, SPEC.window, SPEC.stride),
                        'metric': METRIC, 'shared_size': R.SHARED_SIZE,
                        'code': module_hash()})


def query_bank(bank):
    """The concept directions, in bank order, as the search's query matrix."""
    return np.asarray(bank['directions'], dtype=np.float32)


def build(run):
    """Rebuild the corpus into `retrieval/corpus.{npz,json}` (CPU and the tokenizer), every document's token
    count checked against the shipped index."""
    started = time.time()
    chash = corpus_config_hash(run)
    if stage_done(run, 'corpus', chash) and run.exists(CORPUS_NPZ):
        return run.read_json(CORPUS_JSON)
    tokenizer = load_tokenizer(run.config, run.root / 'cache')
    docs = documents(run.config)
    meta = R.build_corpus(run, SPEC, docs, open_parts(), tokenizer,
                          CORPUS_NPZ, CORPUS_JSON, sizes=())
    mark_stage(run, 'corpus', chash, {'n_windows': meta['n_windows'], 'n_docs': meta['n_docs']},
               started=started)
    write_provenance(run, {'corpus': meta}, stage='corpus')
    run.stage_done('corpus', started, windows=meta['n_windows'], documents=meta['n_docs'])
    return meta


class CorpusScorer:
    """`evals.downstream.common.retrieval`'s scoring seam, answered by this package's GPU workers: `[windows, queries]`
    cosines for a chunk of windows sent as `corpus_search` tasks. The shared signature's model arguments are
    unused (the worker holds them)."""

    def __init__(self, executor):
        self.executor = executor

    def __call__(self, windows, queries, mdl, tok, device, batch=R.WINDOW_BATCH, mu=None):
        windows = [list(window) for window in windows]
        queries = np.asarray(queries, dtype=np.float32)
        blocks = list(chunks(windows, batch))
        raw = np.full((len(windows), queries.shape[0]), -1.0, dtype=np.float32)
        tasks = [(index, {'operation': OPERATION, 'windows': block, 'bank': queries})
                 for index, block in enumerate(blocks)]
        if tasks and self.executor is None:
            raise RuntimeError('The corpus search needs GPU work; select a GPU backend')
        answered = 0
        for index, result in (self.executor.map(tasks) if tasks else ()):
            rows = np.asarray(result['data']['raw'], dtype=np.float32)
            if rows.shape != (len(blocks[index]), queries.shape[0]):
                raise ValueError('The corpus search returned the wrong number of rows')
            raw[index * batch:index * batch + rows.shape[0]] = rows
            answered += 1
        if answered != len(blocks):
            raise ValueError(f'{answered} of {len(blocks)} batches of this chunk came back')
        return raw, None


def execute(worker, payload):
    """One batch of corpus windows scored against every query by `R.window_cos` on the clean base."""
    queries = np.asarray(payload['bank'], dtype=np.float32)
    raw, _centred = R.window_cos(list(payload['windows']), queries, worker.base, worker.tokenizer,
                                 worker.device)
    return {'raw': raw}


def search(run, executor, bank, shard='0/1', merge=False):
    """This invocation's block of the corpus, searched, and the merge once every block exists; None while
    parts are outstanding."""
    started = time.time()
    meta = run.read_json(CORPUS_JSON)
    n_windows = int(meta['n_windows'])
    queries = query_bank(bank)
    base = search_config_hash(run, queries)
    cached = run.cached(WINDOWS, run.key('retrieval', base, modules=('corpus',)))
    if cached is not None:
        print('[retrieval] up to date', flush=True)
        return cached
    k, n = shard_of(shard)
    if n > 1 and run.config.concepts_per_genre is not None:
        raise ValueError(f'a smoke run searches {SMOKE_DOCS} documents, which one launch scores in '
                         f'minutes; --shard {shard} splits a corpus that does not need splitting')
    corpus = R.load_corpus(run, CORPUS_NPZ, SPEC)
    block = R.shard_block(n_windows, k, n)
    chash = R.part_config_hash(base, k, n, block)
    stage = PART_STAGE.format(k=k, n=n)
    if stage_done(run, stage, chash) and run.exists(PART.format(k=k, n=n)):
        print(f'[retrieval] shard {k}/{n} up to date', flush=True)
    else:
        top, n_scored, _excluded = R.score_corpus_topk(
            run, corpus, None, None, None, queries, block, SPEC.top_k,
            metric=METRIC, score=CorpusScorer(executor),
            prefix_windows=R.shared_windows(corpus.docs, SPEC))
        R.save_topk_part(run, PART.format(k=k, n=n), block, top)
        record = {'shard': f'{k}/{n}', 'start': block.start, 'stop': block.stop, 'n': n_scored,
                  'n_windows': n_windows, 'n_act': len(queries), 'seconds': time.time() - started}
        mark_stage(run, stage, chash, record, started=started)
        write_provenance(run, {stage: record}, stage=stage)
        print(f'[retrieval] shard {k}/{n}: windows [{block.start}, {block.stop}) x '
              f'{len(queries)} directions in {record["seconds"]:.0f}s', flush=True)
    done, waiting = R.part_states(run, n, n_windows, PART, PART_STAGE, base)
    if waiting:
        R.refuse_or_wait(merge, 'retrieval', COMMAND, n, done, waiting)
        return None
    return finish(run, corpus, bank, meta, n, n_windows, base, started)


def _snippets(row):
    """One direction's ranked windows as the arm's snippets."""
    return [{'rank': window['rank'], 'window_id': window['window_id'], 'doc': window['doc'],
             'start': window['start'], 'text': window['text'], 'n_tokens': window['n_tokens'],
             'score': window['score'], 'search_cos': window['search_cos']}
            for window in row]


def finish(run, corpus, bank, meta, n_shards, n_windows, base, started):
    """Merge the parts into `retrieval/windows.json`: each direction's best windows over the whole corpus
    (`windows`) and over the `R.SHARED_SIZE` prefix (`shared_windows`), and the search's cosines (`search`)."""
    top = R.merge_topk_parts(run, n_shards, corpus, PART, SPEC.top_k)
    if top.prefix is None:
        raise ValueError(f'the parts carry no {R.SHARED_SIZE} ranking; they were not written by this search')
    tokenizer = load_tokenizer(run.config, run.root / 'cache')
    ranked = R.top_windows(corpus, top, tokenizer)
    shared = R.top_windows(corpus, top.prefix, tokenizer)
    concepts = bank['concepts']
    windows = {str(c['concept_id']): _snippets(row) for c, row in zip(concepts, ranked)}
    short = sorted(cid for cid, found in windows.items() if len(found) != SPEC.top_k)
    if short:
        raise ValueError(f'the search returned fewer than {SPEC.top_k} windows for direction(s) '
                         f'{short[:3]}')
    data = {'windows': windows,
            'shared_windows': {str(c['concept_id']): _snippets(row) for c, row in zip(concepts, shared)},
            'search': R.search_rows(top, n_windows, meta['n_tokens'], queries=range(len(concepts))),
            'corpus': meta, 'n_shards': n_shards,
            'top_k': SPEC.top_k, 'suppression': SPEC.suppression,
            'mean_suppressed': float(np.mean(top.suppressed[:len(concepts)])) if concepts else None,
            'near_duplicate_cos': SPEC.near_duplicate_cos,
            'near_duplicates': R.near_duplicates([found[0]['score'] for found in windows.values()],
                                                 SPEC.near_duplicate_cos)}
    run.save(WINDOWS, run.key('retrieval', base, modules=('corpus',)), data)
    run.stage_done('retrieval', started, directions=len(windows), windows=n_windows,
                   near_duplicates=data['near_duplicates'])
    print(f'[retrieval] {n_windows} windows x {len(windows)} directions over '
          f'{n_shards} part(s); near-duplicates {data["near_duplicates"]}; overlapping candidates passed '
          f'over per concept direction {data["mean_suppressed"]}', flush=True)
    return data
