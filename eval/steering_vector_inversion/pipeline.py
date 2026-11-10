"""One stage at a time, each resuming from the artefacts of the ones before it.

A stage fans out inside itself (GPU payloads across the executor's workers, judge requests across a thread
pool); the stages run in order. `retrieval` also splits across invocations (`--shard k/n`), each writing
one part, and the invocation that finds every part merges them.
"""
import time

from . import corpus, data, evaluate, generate, judge, lens, nla, report, vectors
from .artifacts import read_json
from .config import JUDGES

NEEDS_BANK = ('retrieval', 'rollouts', 'nla', 'plain-steer', 'lens', 'identify')


def load(run, relative):
    path = run.root / relative
    if not path.exists():
        raise RuntimeError(f'Missing prerequisite {relative}; run the preceding stage or all')
    return read_json(path)['data']


def load_bank(run):
    path = 'vectors/bank.json'
    if not (run.root / path).exists():
        raise RuntimeError('Run vectors before this stage')
    return run.cached_arrays(path, read_json(run.root / path)['cache_key'])


def source_map(run, bank):
    """Every arm's texts for every target concept, saved so the report reads exactly what was judged."""
    families = (load(run, 'rollouts/rollouts.json') | load(run, 'rollouts/nla.json')
                | load(run, 'rollouts/plain_steered.json'))
    sources = evaluate.sources(bank, load(run, 'data/prepared.json'), families,
                               load(run, 'retrieval/windows.json')['windows'],
                               load(run, 'lens/summaries.json')['summaries'])
    run.save('scores/sources.json', run.key('sources', sources, modules=('evaluate',)), sources)
    return sources


def preflight(run, executor):
    from .execution import gpu_tasks
    started = time.time()
    # the two shipped generation configs are compared before any container holding both checkpoints starts
    data.check_generation_configs(run.config, run.root/'cache')
    prepared = load(run, 'data/prepared.json')
    records = []
    for genre in ('text', 'code', 'math'):
        concept = next(c for c in prepared['concepts'] if c['genre'] == genre)
        records += [prepared['texts'][concept[side][0]['text_id']] for side in ('positive', 'negative')]
    records.append(max(prepared['texts'].values(), key=lambda r: len(r['token_ids'])))
    result = next(gpu_tasks(run, executor, [(0, {'operation': 'preflight', 'records': records})], 'preflight'))[1]
    run.save('scores/preflight.json', run.key('preflight', None, modules=('model',)), result)
    run.stage_done('preflight', started, status=result['data']['status'],
                   **{name: result['data'].get(name)
                      for name in ('warmup_differences', 'exact', 'max_relative_difference')})
    if result['data']['status'] != 'passed':
        raise ValueError('Preflight failed; inspect saved scores/preflight.json before any benchmark run')
    return result


def run_stage(run, stage, executor=None, judge_workers=16, shard='0/1', merge=False):
    started = time.time()
    if stage == 'prepare':
        return data.prepare(run)
    if stage == 'corpus':
        return corpus.build(run)
    if stage == 'report':
        return report.render(run)
    if stage == 'preflight':
        return preflight(run, executor)
    if stage == 'vectors':
        return vectors.build(run, executor, data.prepare(run))
    bank = load_bank(run) if stage in NEEDS_BANK else None
    if stage == 'retrieval':
        return corpus.search(run, executor, bank, shard, merge)
    if stage == 'rollouts':
        return generate.rollouts(run, executor, bank)
    if stage == 'nla':
        return nla.generate(run, executor, bank)
    if stage == 'plain-steer':
        return generate.steered(run, executor, bank)
    if stage == 'lens':
        return lens.read_words(run, executor, bank)
    if stage == 'summarise':
        return lens.summarise(run, load(run, 'lens/tokens.json'), judge_workers)
    if stage == 'identify':
        cases = evaluate.identification_cases(bank, source_map(run, bank))
        answers = judge.ask(run, 'identification', evaluate.identification_requests(cases), JUDGES,
                            judge_workers)
        result = evaluate.finalize_identification(run, cases, answers)
        run.stage_done('identify', started, cases=len(cases), spend=judge.spend(run, 'identification'))
        return result
    raise ValueError(f'Unknown stage {stage}')
