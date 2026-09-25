"""python -m evals.downstream.steering_vector_inversion_bipo <stage> [options].

    modal run evals/downstream/modal_steering_vector_inversion_bipo.py --stage all --run-id <run-id>

runs the whole eval (README.md). On a CUDA host with OPENROUTER_API_KEY set the same CLI runs without Modal:
the GPU stages then load one model in this process and answer their own payloads through `backend.route`,
one batch at a time. Every GPU batch is cached by content under the run directory and every judge request is
resumed from its log, so a repeated stage costs nothing.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time

from evals.downstream.common import judges as _judges

#: The judge profile of this package's paper numbers, and so its default.
DEFAULT_PROFILE = "sol"
# The profile is read off the command line before the configs, which bind their judges at import, load.
if __name__ == '__main__':
    _judges.activate_from_argv(sys.argv[1:], default=DEFAULT_PROFILE)

from evals.downstream.common import retrieval as R
from evals.downstream.common.judge_client import key_for
from evals.downstream.common.runs import RunDir, refuse_profile_switch
from evals.downstream.steering_vector_inversion.artifacts import Run, digest, read_json, source_hashes, write_json
from evals.downstream.steering_vector_inversion.data import check_generation_configs
from evals.downstream.steering_vector_inversion.execution import GPUExecutor, gpu_tasks

from . import config as C

# The order of `all`; each stage reads what the ones before it wrote.
STAGES = ('items', 'corpus', 'train', 'rollouts', 'nla', 'plain-steer', 'retrieval', 'retrieval-merge', 'lens',
          'controls', 'gate', 'mc10', 'report', 'describe', 'all')
# Run only when named: `describe` rewrites the persona descriptions, which ship frozen in the assets.
OPT_IN_STAGES = ('describe',)
GPU_STAGES = {'train', 'rollouts', 'nla', 'plain-steer', 'retrieval', 'lens', 'all'}
JUDGE_STAGES = {'controls', 'mc10', 'lens', 'all'}
OUTPUT_ROOT = Path('evals/downstream/out/steering_vector_inversion_bipo')
HERE = Path(__file__).parent


def git_stamp():
    """The commit a stage ran at and whether the tree was dirty."""
    from evals.downstream.common.runs import git_state
    state = git_state()
    return {'commit': state['git_commit'], 'dirty': state['git_dirty']}


#: What a module executes through beyond its own bytes, where the pinned batch key does not carry it.
CODE_DEPENDENCIES = {
    'nla_arm': {'pinned': ('nla',)},
    'lens_arm': {'common': ('lens_io',)},
    'plain_steer_arm': {'common': ('plain_steer',)},
    'retrieval_arm': {'pinned': ('model',), 'common': ('retrieval',)},
}


def code_hash(*modules):
    """The bytes of the modules that shape a GPU payload and of what they execute through: part of every
    payload, so an edited module never reads its old cached result."""
    own = [hashlib.sha256((HERE / f'{m}.py').read_bytes()).hexdigest() for m in modules]
    pinned = tuple(name for m in modules for name in CODE_DEPENDENCIES.get(m, {}).get('pinned', ()))
    common = tuple(name for m in modules for name in CODE_DEPENDENCIES.get(m, {}).get('common', ()))
    if not pinned and not common:
        return digest(own)
    return digest([own, source_hashes(pinned, common)])


def stages_of(stage):
    """The stages one invocation runs, in order: `all` is every stage but the opt-in ones."""
    return tuple(s for s in STAGES[:-1] if s not in OPT_IN_STAGES) if stage == 'all' else (stage,)


def parser():
    p = argparse.ArgumentParser(prog='python -m evals.downstream.steering_vector_inversion_bipo')
    p.add_argument('stage', choices=STAGES)
    p.add_argument('--run-id')
    p.add_argument('--output-dir')
    p.add_argument('--behaviours', default='',
                   help='comma-separated subset of the behaviours (persona:<name>); default: all 119')
    p.add_argument('--smoke', action='store_true',
                   help='every stage over the first persona alone and a short prefix of the corpus')
    p.add_argument('--max-gpu-workers', type=int, default=8)
    p.add_argument('--judge-budget-usd', type=float, default=C.JUDGE_CAP_USD)
    p.add_argument('--judge-workers', type=int, default=32)
    p.add_argument('--family-filter', default='',
                   help='a regex a family id (<vector_id>|<arm>) must match, for the judged stages')
    p.add_argument('--shard', default='',
                   help='k/n: one block of the corpus search, for a hand re-run; default every block')
    p.add_argument('--rewrite', action='store_true',
                   help='describe only: ask the writer for every persona, not only those the asset lacks')
    p.add_argument('--freeze', action='store_true',
                   help='describe only: write the described personas into assets/descriptions.json')
    _judges.add_profile_argument(p, default=DEFAULT_PROFILE)
    return p


def corpus_documents(args):
    """How many documents of the held-out search prefix are searched: all, or `C.SMOKE_DOCS` under
    `--smoke` (in the corpus's keys, so a smoke pass and a full pass never share a file)."""
    return C.SMOKE_DOCS if args.smoke else R.CORPUS_DOCS


def retrieval_shards(args):
    """How many blocks the corpus forward is cut into. The smoke corpus is small enough for one."""
    return C.SMOKE_RETRIEVAL_SHARDS if args.smoke else C.RETRIEVAL_SHARDS


def shard_index(args, shards=None):
    """The one block (`--shard k/n`, n the run's own split) this invocation scores, or None for every
    block: how one failed container is re-run without re-scoring the rest."""
    shards = retrieval_shards(args) if shards is None else shards
    if not args.shard:
        return None
    index, _, total = args.shard.partition('/')
    if not total.isdigit() or int(total) != shards:
        raise SystemExit(f'--shard is k/{shards}: the corpus search is a {shards}-way split, and parts '
                         f'of different splits cannot be merged')
    if not index.isdigit() or not 0 <= int(index) < shards:
        raise SystemExit(f'--shard {args.shard}: the block index is 0..{shards - 1}')
    return int(index)


def check_profile(root, profile):
    """Refuse a run directory started under another judge profile: one directory holds one judge's verdicts."""
    refuse_profile_switch(root, profile)
    path = Path(root) / C.RUN_RECORD
    recorded = read_json(path).get('judge_profile') if path.exists() else None
    if recorded not in (None, profile):
        raise SystemExit(f'{root} was started under the {recorded!r} judge profile; this invocation runs '
                         f'{profile!r}. Use another directory, or pass --judge-profile {recorded}')


def resolve(args):
    if bool(args.run_id) == bool(args.output_dir):
        raise SystemExit('Supply --run-id or --output-dir, not both and not neither')
    if args.judge_budget_usd > C.JUDGE_CAP_USD:
        raise SystemExit(f'The judge budget is capped at US${C.JUDGE_CAP_USD:.0f}')
    # Raises when this process already bound its config under another profile.
    _judges.activate(args.judge_profile)
    if (args.freeze or args.rewrite) and args.stage != 'describe':
        raise SystemExit('--freeze and --rewrite are options of the describe stage')
    root = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / args.run_id
    check_profile(root, C.JUDGE_PROFILE)
    run = Run(root, C.svi_config())
    if args.stage == 'describe':
        return run          # it writes the descriptions, so it is not bound to the ones it started with
    descriptions = C.persona_descriptions()
    record = {'descriptions_digest': digest(descriptions), 'descriptions': descriptions,
              'hp': C.HP, 'hp_used': 'vectors/train_plan.json', 'ckpt_epochs': C.CKPT_EPOCHS,
              'primary_epoch': C.PRIMARY_EPOCH, 'behaviours': list(C.BEHAVIOURS),
              'corpus': C.CORPUS.record(), 'corpus_documents': corpus_documents(args),
              'retrieval_shards': retrieval_shards(args), 'lens': C.LENS,
              'judge_profile': C.JUDGE_PROFILE, 'judges': {k: v.model for k, v in C.JUDGES.items()},
              'deviations_from_bipo': ['layer 42 of 64 (BiPO: 15 of 32)', 'no system prompt',
                                       'v = sigma * theta step-size parametrisation', 'no gradient clipping']}
    path = root / C.RUN_RECORD
    if path.exists() and read_json(path).get('descriptions_digest') != record['descriptions_digest']:
        raise SystemExit('The behaviour descriptions differ from the ones this run was started with; '
                         'use a new run directory')
    write_json(path, record)
    return run


def chosen(args):
    """The behaviours every stage of this invocation works on: `--behaviours` or all of them; `--smoke`
    keeps the first."""
    names = [b.strip() for b in args.behaviours.split(',') if b.strip()] or list(C.BEHAVIOURS)
    unknown = [b for b in names if b not in C.BEHAVIOURS]
    if unknown:
        raise SystemExit(f'Unknown behaviours: {unknown}')
    return names[:1] if args.smoke else names


def load_sources(run, behaviours):
    """The frozen training sets, copied to `sources/<behaviour>.json` (the layout the paper builders read)."""
    from . import persona
    sets = persona.training_sets(run.root, behaviours)
    for behaviour, record in sets.items():
        write_json(run.root / 'sources' / f'{behaviour}.json', record)
    return sets


def train_tasks(srcs, behaviours, hp, codes=None):
    def payload(behaviour):
        return {'operation': 'train', 'behaviour': behaviour, 'train': srcs[behaviour]['train'],
                'heldout': srcs[behaviour]['heldout'], 'train_seed': 0, 'hp': hp,
                'ckpt_epochs': [e for e in C.CKPT_EPOCHS if e <= hp['epochs']],
                'code': (codes or {}).get(behaviour) or code_hash('gpu')}
    return [(f'{b}|s0', payload(b)) for b in behaviours]


def train_plan(run, args=None, behaviours=()):
    """The hyperparameters and behaviours this run's vectors were trained with: what lets a later stage
    rebuild the identical (cached) training payload, and what refuses a directory trained under another recipe."""
    path = run.root / 'vectors/train_plan.json'
    plan = read_json(path) if path.exists() else None
    if args is None:
        if plan is None:
            raise SystemExit('No trained vectors in this run directory; run the train stage first')
        return plan
    hp = dict(C.HP)
    if plan is not None and plan['hp'] != hp:
        raise SystemExit(f"This run directory was trained with {plan['hp']}; use a new one for {hp}")
    # each behaviour keeps the gpu.py hash it was trained under, so an edit to gpu.py orphans nothing
    codes = dict(plan['codes']) if plan else {}
    for b in behaviours:
        codes.setdefault(b, code_hash('gpu'))
    plan = {'hp': hp, 'behaviours': [b for b in C.BEHAVIOURS if b in codes], 'codes': codes}
    write_json(path, plan)
    return plan


SUMMARY_KEYS = ('behaviour', 'train_seed', 'hp', 'sigma', 'median_norm', 'diffmean_norm', 'n_train_pairs',
                'n_dropped', 'steps', 'step_seconds', 'train_seconds', 'peak_gb', 'd_counts', 'metrics', 'log')


def train(run, executor, args, behaviours):
    started = time.time()
    srcs = load_sources(run, behaviours)
    plan = train_plan(run, args, behaviours)
    results = {}
    for task_id, result in gpu_tasks(run, executor,
                                     train_tasks(srcs, behaviours, plan['hp'], plan['codes']), 'bipo/train'):
        results[task_id] = result['data']
        last = result['data']['metrics'][-1]
        print(f"{task_id}: acc(+v) {last['acc_pos']:.2f} acc(-v) {last['acc_neg']:.2f} "
              f"norm ratio {last['norm_ratio']:.3f} cos(diffmean) {last['cos_diffmean']:.2f} "
              f"step {result['data']['step_seconds']:.2f}s peak {result['data']['peak_gb']:.0f} GB "
              f"({result['gpu_seconds']/60:.1f} min on {result['gpu_name']})", flush=True)
    summary = [{'task_id': k} | {name: v.get(name) for name in SUMMARY_KEYS} for k, v in sorted(results.items())]
    write_json(run.root / 'vectors/train_summary.json', summary)
    run.stage_done('bipo-train', started, jobs=len(results), git=git_stamp())
    return [results[k] for k in sorted(results)]


def load_bank(run, args, behaviours):
    """The vector bank, rebuilt from the cached training batches, restricted to `behaviours`."""
    from . import arms
    plan = train_plan(run)
    trained = plan['behaviours']
    missing = [b for b in behaviours if b not in trained]
    if missing:
        raise SystemExit(f'Not trained in this run directory: {missing}')
    srcs = load_sources(run, trained)
    results = [result['data'] for _tid, result in
               gpu_tasks(run, None, train_tasks(srcs, trained, plan['hp'], codes=plan['codes']), 'bipo/train')]
    keep = set(behaviours)
    bank = [e for e in arms.vector_bank(results) if e['behaviour'] in keep]
    write_json(run.root / 'vectors/bank_meta.json',
               [{k: v for k, v in e.items() if k != 'direction'} for e in bank])
    return bank, {b: srcs[b] for b in behaviours}


def write_truth(run, bank, families):
    """Truth for the families this stage's bank covers, kept as saved for the rest (a `--behaviours` pass
    rewrites a file that also holds other behaviours' families)."""
    from . import arms
    path = run.root / 'scores/truth.json'
    truth = read_json(path) if path.exists() else {}
    known = {entry['vector_id'] for entry in bank}
    for fid, family in families.items():
        if family['arm'] == 'heldout_matching' or family['vector_id'] in known:
            truth[fid] = arms.family_truth(bank, fid)
    missing = sorted(set(families) - set(truth))
    if missing:
        raise ValueError(f'No truth for {missing[:3]}; run the stage over their behaviours first')
    write_json(path, {fid: truth[fid] for fid in sorted(families)})


def family_filter(args):
    """The families a judged or generating stage is restricted to, as a callable over a family id."""
    import re
    pattern = re.compile(args.family_filter) if args.family_filter else None
    return lambda family_id: pattern is None or bool(pattern.search(family_id))


def merge_families(run, new, bank):
    """`new` merged into `rollouts/families.json`, with truth for everything the file then holds."""
    path = run.root / 'rollouts/families.json'
    families = (read_json(path) if path.exists() else {}) | new
    write_json(path, families)
    write_truth(run, bank, families)
    return families


def rollouts(run, executor, args, behaviours):
    from . import arms
    started = time.time()
    bank, srcs = load_bank(run, args, behaviours)
    wanted = family_filter(args)
    tasks = [(tid, p) for tid, p in arms.rollout_tasks(run.config, bank, code_hash('arms'))
             if wanted(tid.rsplit('|', 1)[0])]
    new = arms.collect(gpu_tasks(run, executor, tasks, 'bipo/rollouts'))
    new |= {fid: f for fid, f in arms.heldout_families(srcs).items() if wanted(fid)}
    for family in new.values():
        for sample in family['samples']:
            for key in ('raw_token_ids', 'token_ids', 'prompt_token_ids'):
                sample.pop(key, None)
    families = merge_families(run, new, bank)
    run.stage_done('bipo-rollouts', started, families=len(families), git=git_stamp())
    return families


def nla(run, executor, args, behaviours):
    """The NLA verbalizer on the same unit directions MAEMM received, added to the saved families."""
    from . import nla_arm
    started = time.time()
    bank, _ = load_bank(run, args, behaviours)
    wanted = family_filter(args)
    tasks = [(tid, p) for tid, p in nla_arm.tasks(run.config, bank, code_hash('nla_arm'))
             if wanted(f"{p['vector_id']}|{nla_arm.ARM}")]
    new = nla_arm.families(gpu_tasks(run, executor, tasks, 'bipo/nla'))
    short = [fid for fid, f in new.items()
             if [x['sample_id'] for x in f['samples']] != [-1, *range(run.config.samples)]]
    if short:
        raise ValueError(f'Incomplete NLA generations: {short[:3]}')
    merge_families(run, new, bank)
    for fid, family in sorted(new.items()):
        print(f"{fid}: closed {family['close_rate']:.2f} | "
              f"{' '.join(family['samples'][1]['text'].split())[:160]}", flush=True)
    rates = nla_arm.close_rates(new)
    print(f"NLA close rate {rates['pooled']} against the reader's line of {rates['min_close_rate']}"
          + (' (BELOW; reported, not enforced)' if rates['below_min_close_rate'] else ''), flush=True)
    run.stage_done('bipo-nla', started, families=len(new), close_rate=rates['pooled'],
                   min_close_rate=rates['min_close_rate'], git=git_stamp())


def plain_steer_stage(run, executor, args, behaviours):
    """The steered model at every strength, merged into the families with truth, and its text health."""
    from . import plain_steer_arm
    started = time.time()
    bank, _ = load_bank(run, args, behaviours)
    tasks = plain_steer_arm.tasks(bank, code_hash('plain_steer_arm'), family_filter(args))
    new = plain_steer_arm.families(gpu_tasks(run, executor, tasks, 'bipo/plain_steer'))
    families = merge_families(run, new, bank)
    health = plain_steer_arm.health_table(run.root, families)
    for row in health:
        if row['arm'] == plain_steer_arm.P.TABLE_ARM:
            print(f"{row['vector_id']:60s} degenerate {row['degenerate_share']:.2f} "
                  f"log-lik {row['loglik_median']:.2f} eos {row['eos_early_share']:.2f}", flush=True)
    run.stage_done('bipo-plain-steer', started, families=len(new), payloads=len(tasks), git=git_stamp())


def corpus_stage(run, args):
    """The shared corpus the search reads, rebuilt once per run: CPU and a tokenizer, no GPU."""
    from . import retrieval_arm
    started = time.time()
    meta = retrieval_arm.build(run.root, run.config, run.root / 'cache', smoke=args.smoke)
    run.stage_done('bipo-corpus', started, windows=meta['n_windows'], documents=meta['n_docs'],
                   git=git_stamp())
    print(f"corpus: {meta['n_windows']} windows of up to {meta['window']} tokens at stride "
          f"{meta['stride']} over {meta['n_docs']} held-out documents ({meta['n_tokens']} tokens, "
          f"documents {meta['doc_min']}-{meta['doc_max']})", flush=True)


def retrieval_stage(run, executor, args, behaviours):
    """The corpus forward: one contiguous block of windows per container, every direction at once; each
    block writes its own part file and stage record, so a re-run scores only the missing blocks."""
    from . import retrieval_arm
    started = time.time()
    bank, _ = load_bank(run, args, behaviours)
    run_dir = RunDir(str(run.root))
    ids, matrix = retrieval_arm.queries(bank)
    run_dir.write_json(retrieval_arm.QUERIES, ids)
    corpus = retrieval_arm.load_corpus(run.root)
    base = retrieval_arm.base_hash(run_dir, run.config, ids, code_hash('retrieval_arm'))
    tasks = retrieval_arm.shard_tasks(run_dir, corpus, ids, matrix, base, retrieval_shards(args),
                                      only=shard_index(args))
    parts = 0
    for task_id, result in executor.map(tasks):
        record = retrieval_arm.save_part(run_dir, result['data'], base, result['gpu_seconds'])
        parts += 1
        print(f"retrieval shard {task_id}: windows [{record['start']}, {record['stop']}) x {len(ids)} "
              f"directions in {result['gpu_seconds'] / 60:.1f} min on {result['gpu_name']}", flush=True)
    run.stage_done('bipo-retrieval', started, parts=parts, directions=len(ids), git=git_stamp())


def retrieval_merge(run, args, behaviours):
    """The parts of the split into one top-k per direction, and the reader's families (CPU)."""
    from . import retrieval_arm
    started = time.time()
    bank, _ = load_bank(run, args, behaviours)
    run_dir = RunDir(str(run.root))
    ids = run_dir.read_json(retrieval_arm.QUERIES)
    corpus = retrieval_arm.load_corpus(run.root)
    tok = retrieval_arm.load_tokenizer(run.config, run.root / 'cache')
    base = retrieval_arm.base_hash(run_dir, run.config, ids, code_hash('retrieval_arm'))
    new, record = retrieval_arm.merge(run.root, corpus, tok, base, retrieval_shards(args))
    wanted = family_filter(args)
    merge_families(run, {fid: f for fid, f in new.items() if wanted(fid)}, bank)
    run.stage_done('bipo-retrieval-merge', started, **record, git=git_stamp())
    print(f"retrieval: {record['n_queries']} directions x top {record['top_k']} of "
          f"{record['n_windows']} windows, {record['near_duplicates']} above "
          f"{record['near_duplicate_cos']:.2f}", flush=True)


def lens_stage(run, executor, args, behaviours):
    """The lens read of every direction, and the one summary per direction that is the arm's text."""
    from . import judge as J
    from . import lens_arm
    started = time.time()
    bank, _ = load_bank(run, args, behaviours)
    data = None
    for _tid, result in gpu_tasks(run, executor, lens_arm.tasks(bank, code_hash('lens_arm')), 'bipo/lens'):
        data = result['data']
    words = [{'vector_id': vector_id, 'words': list(row)}
             for vector_id, row in zip(data['vector_ids'], data['words'])]
    write_json(run.root / lens_arm.WORDS, {'layer': data['layer'], 'lens': data['lens'], 'words': words})
    reqs = lens_arm.requests(words)
    print(f"[{lens_arm.INSTRUMENT}] {C.SUMMARISER}: {len(reqs)} requests over {len(words)} directions, "
          f"projected US${J.estimate(reqs, C.SUMMARISER):.2f}", flush=True)
    records = list(judged(run, args)(lens_arm.INSTRUMENT, reqs, C.SUMMARISER)) if reqs else []
    if len(records) != len(reqs):
        raise ValueError(f'{lens_arm.INSTRUMENT}: {len(records)} records for {len(reqs)} requests')
    summarised = lens_arm.summaries(words, records)
    write_json(run.root / lens_arm.SUMMARIES, summarised)
    wanted = family_filter(args)
    new = {fid: f for fid, f in lens_arm.families(summarised).items() if wanted(fid)}
    merge_families(run, new, bank)
    unsummarised = sorted(v for v, r in summarised.items() if r['status'] != 'ok')
    run.stage_done('bipo-lens', started, families=len(new), unsummarised=unsummarised, git=git_stamp())


def gate(run):
    """The stopping rule, read off the control arms before the rest of the judging is paid for: held-out
    statements must be identified in at least 0.80 of the bundles the judge answered."""
    from evals.downstream.common.judges import REFERENCE_JUDGE
    from . import mc10
    rows = [r for r in read_json(run.root / mc10.SCORES.format(list=mc10.MAIN))
            if r['judge'] == REFERENCE_JUDGE and mc10.arm_of(r['family_id']) == 'heldout_matching']
    answered = [r for r in rows if r.get('status') == 'ok']
    rate = sum(bool(r.get('correct')) for r in answered) / len(answered) if answered else None
    verdict = {'judge': REFERENCE_JUDGE, 'heldout_rate': rate, 'heldout_bundles': len(answered),
               'threshold': 0.80,
               'rule': 'held-out statements identified in at least 0.80 of the bundles the reference '
                       'judge answered'}
    verdict['passed'] = rate is not None and rate >= verdict['threshold']
    write_json(run.root / 'scores/gate.json', verdict)
    print(verdict, flush=True)
    if not verdict['passed']:
        raise SystemExit('The identification instrument failed its check on held-out text; judging stops here')


CLIENT = None  # a factory (spec, title) -> client, set by a launcher that holds the key elsewhere


def api_key(spec):
    """The key `spec` is asked with, or None under a launcher's `CLIENT`; exits naming the variable when
    the key is needed and not set."""
    if CLIENT is not None:
        return None
    try:
        return key_for(spec)
    except RuntimeError as e:
        raise SystemExit(f'{e}. Set it in the controller environment for the judged stages, or run them '
                         'through evals/downstream/modal_steering_vector_inversion_bipo.py') from e


def judged(run, args):
    """The LLM access the judged instruments are handed: one callable, this run's ledger and client."""
    from . import judge

    def ask(instrument, requests, name):
        spec = judge.spec_of(name)
        client = None if CLIENT is None else CLIENT(spec, judge.TITLE)
        return judge.run(run.root, instrument, requests, name, api_key(spec),
                         cap_usd=args.judge_budget_usd, workers=args.judge_workers, client=client)
    return ask


def describe_stage(run, args, behaviours):
    """The persona descriptions from each persona's own training statements (`persona.describe`), and with
    `--freeze` into the shipped asset. Builds the training files first (idempotent, CPU)."""
    from . import persona
    names = [b.removeprefix('persona:') for b in behaviours]
    persona.build(run.root, names=names)
    out = persona.describe(run.root, judged(run, args), names, rewrite=args.rewrite)
    print(f"{len(out['descriptions'])} persona sentences in {run.root / persona.DESCRIPTIONS_FILE} "
          f"(digest {out['digest']})", flush=True)
    if args.freeze:
        frozen = persona.freeze(run.root, names, replace=args.rewrite)
        print(f"froze {len(frozen['descriptions'])} persona sentences into {C.PERSONA_DESCRIPTIONS_ASSET} "
              f"(digest {frozen['digest']})", flush=True)


def main(argv=None, remote=None, judge_client=None):
    global CLIENT
    CLIENT = judge_client
    args = parser().parse_args(argv)
    run = resolve(args)
    behaviours = chosen(args)
    stages = stages_of(args.stage)
    executor = None
    if GPU_STAGES & set(stages):
        # the two checkpoints must ship the same generation config; checked before the first payload
        check_generation_configs(run.config, run.root / 'cache')
        workers = args.max_gpu_workers
        if remote is None:
            # a CUDA host without Modal: one model in this process, one batch at a time
            from . import backend
            remote, workers = backend.local(run.config, run.root / 'cache'), 1
        executor = GPUExecutor(run.config, None, workers, remote=remote)
    if JUDGE_STAGES & set(stages):
        for spec in C.JUDGES.values():   # every judge's key before any GPU work
            api_key(spec)
    if 'describe' in stages:
        api_key(C.DESCRIBE_JUDGE)
    try:
        for stage in stages:
            print(f'== {stage}', flush=True)
            if stage == 'items':
                from . import persona
                for name, counts in persona.build(run.root, names=[b.removeprefix('persona:')
                                                                   for b in behaviours]).items():
                    print(f'persona:{name}: {counts}', flush=True)
            elif stage == 'describe':
                describe_stage(run, args, behaviours)
            elif stage == 'train':
                train(run, executor, args, behaviours)
            elif stage == 'rollouts':
                rollouts(run, executor, args, behaviours)
            elif stage == 'nla':
                nla(run, executor, args, behaviours)
            elif stage == 'plain-steer':
                plain_steer_stage(run, executor, args, behaviours)
            elif stage == 'corpus':
                corpus_stage(run, args)
            elif stage == 'retrieval':
                retrieval_stage(run, executor, args, behaviours)
            elif stage == 'retrieval-merge':
                retrieval_merge(run, args, behaviours)
            elif stage == 'lens':
                lens_stage(run, executor, args, behaviours)
            elif stage == 'controls':
                # the control arms first, so the gate can stop a broken instrument cheaply; `mc10` resumes them
                from . import mc10
                mc10.run(run.root, judged(run, args), arms=mc10.CONTROL_ARMS,
                         family_filter=args.family_filter or None)
            elif stage == 'gate':
                gate(run)
            elif stage == 'mc10':
                from . import mc10
                mc10.run(run.root, judged(run, args), family_filter=args.family_filter or None)
            elif stage == 'report':
                from . import mc10, report
                mc10.tables(run.root)
                print(report.render(run.root), flush=True)
    except BaseException:
        if executor is not None:
            executor.cancel()
        raise
    if executor is not None:
        executor.close()


if __name__ == '__main__':
    main()
