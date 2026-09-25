"""python -m evals.downstream.steering_vector_inversion <stage> [options]."""
import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from evals.downstream.common import judges as _judges

#: The judge profile the paper's AxBench numbers used, and this package's default.
DEFAULT_PROFILE = 'sol'

# Read before the config is imported: the config binds its judges at import.
if __name__ == '__main__':
    _judges.activate_from_argv(sys.argv[1:], default=DEFAULT_PROFILE)

from evals.downstream.common.judge_client import key_for
from evals.downstream.common.runs import refuse_profile_switch
from .artifacts import Run, read_json
from .config import CAPS, Config, JUDGES, JUDGE_CAP_USD, JUDGE_PROFILE
from .pipeline import run_stage

#: The stage order; each resumes from the artefacts of the ones above it.
PIPELINE = ('prepare', 'corpus', 'preflight', 'vectors', 'retrieval', 'rollouts', 'nla', 'plain-steer',
            'lens', 'summarise', 'identify', 'report')
STAGES = (*PIPELINE, 'all')
GPU_STAGES = {'preflight', 'vectors', 'retrieval', 'rollouts', 'nla', 'plain-steer', 'lens', 'all'}
JUDGE_STAGES = {'summarise', 'identify', 'all'}
OUTPUT_ROOT = Path('evals/downstream/out/steering_vector_inversion')


def stages_of(stage):
    """The stages one invocation runs, in order: `all` is the whole pipeline, anything else is itself."""
    return PIPELINE if stage == 'all' else (stage,)


def parser():
    p = argparse.ArgumentParser(description='AxBench steering-vector inversion evaluation')
    p.add_argument('stage', choices=STAGES)
    p.add_argument('--output-dir', type=Path, help='Exact run directory; resume matching artifacts here')
    p.add_argument('--run-id', help=f'Run name under {OUTPUT_ROOT} when output-dir is omitted')
    p.add_argument('--smoke', type=int, metavar='N',
                   help='Evaluate N concepts per genre over a short prefix of the corpus')
    p.add_argument('--seed', type=int)
    p.add_argument('--max-gpu-workers', type=int, default=8)
    p.add_argument('--judge-concurrency', type=int, default=16,
                   help='Judge requests in flight per instrument')
    p.add_argument('--read-batch-size', type=int)
    p.add_argument('--generation-batch-size', type=int)
    p.add_argument('--shard', default='0/1', metavar='K/N',
                   help='The block of the corpus this invocation searches, for the retrieval stage. '
                        'Invocations sharing a run directory write one part each')
    p.add_argument('--merge', action='store_true',
                   help='This invocation is the designated merge of a sharded corpus search: it scores '
                        'nothing new and fails if a part is missing')
    _judges.add_profile_argument(p, default=DEFAULT_PROFILE)
    return p


def resolve(args, judge_client=None):
    """The run directory and configuration this invocation works in, its arguments checked first. Without a
    launcher's `judge_client`, a judged stage needs its key here, before any GPU work."""
    if args.output_dir is not None and args.run_id is not None:
        raise ValueError('Supply an exact --output-dir or a --run-id, not both')
    if args.max_gpu_workers < 1 or args.judge_concurrency < 1:
        raise ValueError('GPU workers and judge concurrency must be positive')
    shard, _, shards = str(args.shard).partition('/')
    if not (shard.isdigit() and shards.isdigit() and 0 <= int(shard) < int(shards)):
        raise ValueError('--shard is k/n with 0 <= k < n')
    # raises when this process already bound its config under another profile
    _judges.activate(args.judge_profile)
    if args.stage in JUDGE_STAGES and judge_client is None:
        for spec in JUDGES.values():
            try:
                key_for(spec)
            except RuntimeError as e:
                raise ValueError(f'{e}. Set it in the launch environment before running this stage, or run '
                                 'the stage through evals/downstream/modal_steering_vector_inversion.py, which holds the '
                                 'keys in Modal secrets; no GPU work has started') from e
    root = args.output_dir or OUTPUT_ROOT / (args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    refuse_profile_switch(root, JUDGE_PROFILE)
    config = Config(**read_json(root/'config.json')) if (root/'config.json').exists() else Config()
    changes = {name: getattr(args, name) for name in ('seed', 'read_batch_size', 'generation_batch_size')
               if getattr(args, name) is not None}
    if args.smoke is not None:
        changes['concepts_per_genre'] = args.smoke
    return Run(root, replace(config, **changes))


def main(argv=None, remote=None, judge_client=None):
    from . import judge

    judge.CLIENT = judge_client   # None: this process's own key; a launcher's factory otherwise
    args = parser().parse_args(argv)
    run = resolve(args, judge_client)
    # Model snapshots go to the run's cache/; the corpus parquet files to the user's Hugging Face cache
    # (HF_HOME), shared by every run.
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    print(f'Run: {run.root.resolve()}', flush=True)
    print('Judge caps: ' + ', '.join(f'{name} ${cap:.2f}' for name, cap in CAPS.items())
          + f' (${JUDGE_CAP_USD:.2f} in total)', flush=True)
    stages = stages_of(args.stage)
    executor = None
    try:
        if args.stage in GPU_STAGES:
            from .execution import GPUExecutor
            executor = GPUExecutor(run.config, run.root/'cache', args.max_gpu_workers, remote=remote)
        result = None
        for position, stage in enumerate(stages, start=1):
            if len(stages) > 1:
                print(f'Stage {position} of {len(stages)}: {stage}', flush=True)
            result = run_stage(run, stage, executor, args.judge_concurrency, args.shard, args.merge)
            print(f'Completed stage: {stage}', flush=True)
        return result
    finally:
        if executor is not None:
            executor.close()


if __name__ == '__main__':
    main()
