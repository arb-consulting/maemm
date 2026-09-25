"""The released Jacobian lens as a reader of the same directions (`jlens`).

`evals.downstream.common.lens_io` loads the pinned lens and reads the ten word-like tokens a direction promotes at the
read layer. A token list is not text, so the reference judge turns each list into two to four sentences of
prose (`evals.downstream.common.lens_summary`), seeing the tokens and nothing else. That summary is the arm's one text,
judged once in the greedy slot. The summaries have their own ledger (`summaries`).
"""
import time

import numpy as np

from evals.downstream.common.judges import REFERENCE_JUDGE
from evals.downstream.common.lens_summary import summary_request
from evals.downstream.common.pins import JLENS_COMMIT, LENS_BYTES, LENS_FILE, LENS_REPO, LENS_REVISION, LENS_SHA256

from .artifacts import digest
from .config import JUDGES, LENS_TOP_WORDS
from .execution import gpu_tasks

#: The GPU operation the lens pass is, on a model worker.
OPERATION = 'lens'
#: The lens pins and token count; they ride on the payload so they are in the batch key.
LENS_PINS = {'repo': LENS_REPO, 'revision': LENS_REVISION, 'file': LENS_FILE, 'bytes': LENS_BYTES,
             'sha256': LENS_SHA256, 'reader_commit': JLENS_COMMIT, 'top_words': LENS_TOP_WORDS}


def read_words(run, executor, bank):
    """The lens's ten word-like tokens for every direction of the bank, in one GPU task."""
    started = time.time()
    directions = np.asarray(bank['directions'], dtype=np.float32)
    key = run.key('lens', [digest(directions), LENS_PINS], modules=('lens', 'model'))
    cached = run.cached('lens/tokens.json', key)
    if cached is not None:
        return cached
    result = next(gpu_tasks(run, executor, [(0, {'operation': OPERATION, 'directions': directions,
                                                 'pins': LENS_PINS})], 'lens/batches'))[1]
    rows = result['data']['top_words']
    if len(rows) != len(directions):
        raise ValueError('The lens returned a reading for a different number of directions than the bank holds')
    data = {'top_words': {str(c['concept_id']): list(row) for c, row in zip(bank['concepts'], rows)},
            'lens': result['data']['lens'], 'layer': run.config.read_layer, 'top_word': LENS_TOP_WORDS}
    run.save('lens/tokens.json', key, data)
    run.stage_done('lens', started, directions=len(rows))
    return data


def summary_requests(words):
    """One request per direction, in concept-id order, carrying the lens's ranked token list and nothing
    else."""
    rows = sorted(words['top_words'].items(), key=lambda pair: int(pair[0]))
    return [summary_request(tokens, meta={'direction': name}) for name, tokens in rows]


def summarise(run, words, workers=16):
    """The reference judge's prose for every direction. A direction whose summary was not written has no
    text, and the arm is unavailable for that concept rather than judged on an empty snippet."""
    from . import judge

    started = time.time()
    spec = JUDGES[REFERENCE_JUDGE]
    requests = summary_requests(words)
    answers = judge.ask(run, 'summaries', requests, {REFERENCE_JUDGE: spec}, workers)[REFERENCE_JUDGE]
    summaries = {}
    for request in requests:
        record = answers.get(judge.case_key(request['meta'])) or {}
        status = record.get('status', 'unavailable')
        summaries[request['meta']['direction']] = {'text': (record.get('text') or '') if status == 'ok' else '',
                                                   'status': status}
    written = sum(1 for value in summaries.values() if value['text'].strip())
    key = run.key('summaries', sorted(answers), modules=('lens', 'judge'), common=('lens_summary',))
    data = {'model': spec.model, 'judge': spec.name, 'written': written, 'summaries': summaries}
    run.save('lens/summaries.json', key, data)
    run.stage_done('summarise', started, summaries=len(summaries), written=written,
                   spend=judge.spend(run, 'summaries'))
    return data


def execute(worker, payload):
    """One lens pass over every direction, on a model worker; the lens and its word mask are loaded once per
    worker (the file is checked against its pinned size and digest)."""
    from evals.downstream.common.lens_io import Unembed, load_lens, top_words, wordlike_mask

    if getattr(worker, 'lens', None) is None:
        worker.lens, worker.lens_provenance = load_lens(
            str(worker.device), LENS_REPO, LENS_REVISION, LENS_FILE, LENS_BYTES, LENS_SHA256,
            worker.config.read_layer, worker.config.hidden_size)
        worker.lens_unembed = Unembed(worker.base)
        worker.lens_mask = wordlike_mask(worker.tokenizer, worker.lens_unembed.vocab)
    rows = np.asarray(payload['directions'], dtype=np.float32)
    words = top_words(rows, worker.lens, worker.lens_unembed, worker.tokenizer, worker.lens_mask,
                      layer=worker.config.read_layer, device=worker.device, top_word=LENS_TOP_WORDS)
    return {'top_words': words, 'lens': worker.lens_provenance}
