"""Modal backend for evals.downstream.steering_vector_inversion: GPU containers for the GPU stages, a CPU container per
judge call.

    modal run evals/downstream/modal_steering_vector_inversion.py --stage all --run-id <run-id>

The run directory stays on the controller. `all` runs stage by stage, each on the card its models need
(`TWO_MODEL_STAGES` hold the base and the inverter; the `nla` stage's containers the verbalizer alone).
Settings are in `evals/downstream/modal.env.example`. `--shard k/n` and `--merge` pass through to the corpus search.
"""
import json
import os
from dataclasses import asdict
from pathlib import Path
import threading

import modal

REPO = Path(__file__).resolve().parent.parent.parent
PACKAGE = REPO / 'evals/downstream/steering_vector_inversion'
APP_NAME = os.environ.get('EVAL_APP', 'maem-steering-vector-inversion')
GPU_SPEC = os.environ.get('EVAL_GPU', 'H200')
GPU_BOTH_MODELS = os.environ.get('EVAL_GPU_BOTH_MODELS', 'B200')
GPU_WORKERS = int(os.environ.get('EVAL_GPU_WORKERS', '8'))
TWO_MODEL_STAGES = ('preflight', 'rollouts')
CACHE_VOLUME = os.environ.get('EVAL_MODEL_CACHE_VOLUME', 'maem-eval-model-cache')
MODEL_CACHE_DIR = os.environ.get('EVAL_MODEL_CACHE_DIR', '/cache/models')
SECRET = modal.Secret.from_name(os.environ.get('EVAL_OPENROUTER_SECRET', 'maem-openrouter'))
# The Anthropic secret (the `sonnet` profile) is mounted only when named.
ANTHROPIC_SECRET = os.environ.get('EVAL_ANTHROPIC_SECRET', '')
JUDGE_SECRETS = [SECRET] + ([modal.Secret.from_name(ANTHROPIC_SECRET)] if ANTHROPIC_SECRET else [])
# Every variable read at import, pins included: Modal imports this file again inside each container.
IMPORT_ENV = ('EVAL_APP', 'EVAL_GPU', 'EVAL_GPU_BOTH_MODELS', 'EVAL_GPU_WORKERS', 'EVAL_MODEL_CACHE_VOLUME',
              'EVAL_MODEL_CACHE_DIR', 'EVAL_OPENROUTER_SECRET', 'EVAL_ANTHROPIC_SECRET',
              *(f'EVAL_{pin}_{field}' for pin in ('BASE', 'INVERTER', 'NLA', 'LENS') for field in ('REPO', 'REVISION')))
LAUNCH_ENV = {k: os.environ[k] for k in IMPORT_ENV if k in os.environ}


def _git_refs():
    """`(commit, branch, dirty)` of the launching checkout, "" where git cannot answer."""
    import subprocess

    def _one(args):
        try:
            return subprocess.check_output(["git"] + args, cwd=str(REPO), text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = _one(["status", "--porcelain"])
    return (
        _one(["rev-parse", "HEAD"]) or "",
        _one(["rev-parse", "--abbrev-ref", "HEAD"]) or "",
        "" if status is None else ("1" if status else "0"),
    )


GIT_ENVIRONMENT = dict(zip(('GIT_COMMIT', 'GIT_BRANCH', 'GIT_DIRTY'), _git_refs()))

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
HF_HOME = os.path.dirname(MODEL_CACHE_DIR) if MODEL_CACHE_DIR.endswith('/hub') else '/cache/huggingface'
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install_from_requirements(str(PACKAGE/'requirements.txt'))
    .env({'PYTHONPATH': '/app', 'HF_HOME': HF_HOME,
          'TOKENIZERS_PARALLELISM': 'false', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
          **GIT_ENVIRONMENT, **LAUNCH_ENV})
    .add_local_dir(REPO/'maem', '/app/maem', ignore=['__pycache__'])
    .add_local_dir(REPO/'evals/downstream', '/app/evals/downstream', ignore=['__pycache__', '**/__pycache__/**', 'out', 'analysis',
                                                     'test_*'])
)
judge_image = (
    modal.Image.debian_slim(python_version='3.12')
    .env({'PYTHONPATH': '/app', **LAUNCH_ENV})
    .add_local_dir(REPO/'evals/downstream/common', '/app/evals/downstream/common', ignore=['__pycache__', '**/__pycache__/**',
                                                                   'nla'])
)


# Every corpus-search batch carries the same query bank, so it is uploaded once and sent by reference.
BANK_OPERATIONS = ('corpus_search',)


def read_bank_key(bank):
    from evals.downstream.steering_vector_inversion.artifacts import digest
    return digest(bank)


def remote_with_read_banks(remote, banks, uploaded=None):
    """`remote` with each payload's bank uploaded to `banks` once and sent by reference; the payload the
    worker sees, and so its cache key, is unchanged."""
    uploaded = set() if uploaded is None else uploaded
    lock = threading.Lock()

    def execute(payload):
        if payload.get('operation') not in BANK_OPERATIONS or payload.get('bank') is None:
            return remote(payload)
        key = read_bank_key(payload['bank'])
        with lock:
            if key not in uploaded:
                banks[key] = payload['bank']
                uploaded.add(key)
        wire = {k: v for k, v in payload.items() if k != 'bank'}
        wire['_read_bank'] = (banks, key)
        return remote(wire)

    return execute


def restore_read_bank(payload, local_banks):
    if '_read_bank' not in payload:
        return payload
    banks, key = payload['_read_bank']
    if key not in local_banks:
        bank = banks[key]
        if read_bank_key(bank) != key:
            raise ValueError('Query bank transport checksum mismatch')
        local_banks[key] = bank
    return {k: v for k, v in payload.items() if k != '_read_bank'} | {'bank': local_banks[key]}


@app.function(image=image, volumes={'/cache': cache}, timeout=3600)
def prepare_weights(config_json: str):
    """Download the pinned base, inverter, NLA verbalizer and lens file once, on CPU."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from evals.downstream.common.nla import nla_reader
    from evals.downstream.common.pins import LENS_FILE, LENS_REPO, LENS_REVISION
    from evals.downstream.steering_vector_inversion.config import Config
    config = Config(**json.loads(config_json)).validate()
    patterns = ['*.json', '*.safetensors', '*.jinja', '*.txt']
    for repo_id, revision in ((config.model, config.model_revision),
                              (config.inverter, config.inverter_revision)):
        snapshot_download(repo_id, revision=revision, allow_patterns=patterns,
                          cache_dir=MODEL_CACHE_DIR, token=False)
    nla_reader.check_checkpoint(nla_reader.download_checkpoint(cache_dir=MODEL_CACHE_DIR))
    hf_hub_download(LENS_REPO, LENS_FILE, revision=LENS_REVISION, token=False)
    cache.commit()


@app.cls(image=image, gpu=GPU_SPEC, volumes={'/cache': cache}, max_containers=GPU_WORKERS,
         min_containers=0, buffer_containers=0, scaledown_window=300,
         timeout=3600, startup_timeout=1800)
class GPU:
    config_json: str = modal.parameter()
    role: str = modal.parameter(default='model')   # `model` (base, inverter) or `nla` (the verbalizer)

    @modal.enter()
    def load(self):
        cache.reload()
        self.worker = build_worker(self.role, self.config_json, MODEL_CACHE_DIR)
        self.read_banks = {}

    @modal.method()
    def execute(self, payload):
        return self.worker.execute(restore_read_bank(payload, self.read_banks))


@app.function(image=judge_image, secrets=JUDGE_SECRETS, max_containers=4, timeout=900, memory=2048)
@modal.concurrent(max_inputs=64)
def call(spec: dict, req: dict, title: str):
    """One judge request, with the key held here and never by the controller."""
    from evals.downstream.common.judge_client import JudgeSpec, client_for, key_for
    spec = JudgeSpec(**spec)
    return client_for(spec, key_for(spec), title=title).call(req)


class RemoteClient:
    """What `judge.ask` needs of a client: `call(req) -> (text, usage)`, raising JudgeError."""
    def __init__(self, spec, title):
        self.spec, self.title = asdict(spec), title

    def call(self, req):
        return call.remote(self.spec, req, self.title)


@app.local_entrypoint()
def main(stage: str = 'all', run_id: str = '', output_dir: str = '',
         smoke: int = 0, max_gpu_workers: int = 0, judge_concurrency: int = 0,
         read_batch_size: int = 0, generation_batch_size: int = 0,
         shard: str = '', merge: bool = False, judge_profile: str = 'sol'):
    """Flags left at 0 or "" take the package CLI's defaults. A corpus search split `n` ways is `n` launches
    with `--shard k/n`, then one with `--shard 0/n --merge`. `--judge-profile` (`sol`, the paper's, or
    `sonnet`) is activated before the package is imported."""
    from evals.downstream.common import judges
    judges.activate(judge_profile)
    from evals.downstream.steering_vector_inversion.config import JUDGE_CAP_USD
    from evals.downstream.steering_vector_inversion.__main__ import GPU_STAGES, main as cli, parser, resolve, stages_of
    from evals.downstream.steering_vector_inversion.nla import role_of_stage
    if bool(run_id) == bool(output_dir):
        raise ValueError('Supply --run-id or --output-dir, not both and not neither')
    print(f'Judge ledgers cap this run at ${JUDGE_CAP_USD:.2f} in total', flush=True)
    options = ['--run-id', run_id] if run_id else ['--output-dir', output_dir]
    options += [judges.PROFILE_FLAG, judge_profile]
    for flag, value in (('--max-gpu-workers', max_gpu_workers or GPU_WORKERS),
                        ('--judge-concurrency', judge_concurrency),
                        ('--read-batch-size', read_batch_size),
                        ('--generation-batch-size', generation_batch_size)):
        if value:
            options += [flag, str(value)]
    if smoke:
        options += ['--smoke', str(smoke)]
    if shard:
        options += ['--shard', shard]
    if merge:
        options += ['--merge']
    run = resolve(parser().parse_args([stage, *options]), RemoteClient)
    stages = stages_of(stage)
    if not set(stages) & GPU_STAGES:
        cli([stage, *options], judge_client=RemoteClient)
        return
    config_json = json.dumps(run.config.to_dict(), sort_keys=True)
    prepare_weights.remote(config_json)
    with modal.Dict.ephemeral() as banks:
        uploaded = set()
        for position, one in enumerate(stages, start=1):
            if len(stages) > 1:
                print(f'Stage {position} of {len(stages)}: {one}', flush=True)
            if one not in GPU_STAGES:
                cli([one, *options], judge_client=RemoteClient)
                continue
            remote = _gpu_cls(one)(config_json=config_json, role=role_of_stage(one)).execute.remote
            cli([one, *options], remote=remote_with_read_banks(remote, banks, uploaded),
                judge_client=RemoteClient)


def build_worker(role, config_json, cache_dir):
    """The worker a container of `role` holds for its whole life."""
    from evals.downstream.steering_vector_inversion.config import Config
    from evals.downstream.steering_vector_inversion.nla import ROLES, NlaWorker
    if role not in ROLES:
        raise ValueError(f'A GPU container is one of {ROLES}, not {role!r}')
    config = Config(**json.loads(config_json))
    if role == 'nla':
        return NlaWorker(config, cache_dir)
    from evals.downstream.steering_vector_inversion.model import ModelWorker
    return ModelWorker(config, cache_dir)


def _gpu_cls(stage):
    """The worker class on the card the stage needs."""
    return GPU.with_options(gpu=GPU_BOTH_MODELS) if stage in TWO_MODEL_STAGES else GPU
