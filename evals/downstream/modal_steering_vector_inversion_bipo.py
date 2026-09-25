"""Modal backend for the persona-vector eval: GPU containers for the GPU stages, a CPU container per judge call.

    modal run evals/downstream/modal_steering_vector_inversion_bipo.py --stage all --run-id <run-id>

The run directory, judge logs, ledger and GPU batch cache stay on the controller. `all` runs stage by stage,
each on the card its models need: `rollouts` holds the base and the inverter (`EVAL_GPU_BOTH_MODELS`),
`train` needs a Blackwell card for the backward pass (`EVAL_GPU_TRAINING`), every other GPU stage holds one
checkpoint (`EVAL_GPU`). Settings are in `evals/downstream/modal.env.example`.
"""
import json
import os
from dataclasses import asdict
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent
PACKAGE = REPO / 'evals/downstream/steering_vector_inversion'
HERE = REPO / 'evals/downstream/steering_vector_inversion_bipo'
APP_NAME = os.environ.get('EVAL_APP', 'maem-steering-vector-inversion-bipo')
GPU_SPEC = os.environ.get('EVAL_GPU', 'H200')
GPU_BOTH_MODELS = os.environ.get('EVAL_GPU_BOTH_MODELS', 'B200')
TWO_MODEL_STAGES = ('rollouts',)
# The gated-linear-attention backward is refused on Hopper under the pinned Triton.
GPU_TRAINING = os.environ.get('EVAL_GPU_TRAINING', 'B200')
BACKWARD_STAGES = ('train',)
GPU_WORKERS = int(os.environ.get('EVAL_GPU_WORKERS', '8'))
CACHE_VOLUME = os.environ.get('EVAL_MODEL_CACHE_VOLUME', 'maem-eval-model-cache')
MODEL_CACHE_DIR = os.environ.get('EVAL_MODEL_CACHE_DIR', '/cache/models')
SECRET = modal.Secret.from_name(os.environ.get('EVAL_OPENROUTER_SECRET', 'maem-openrouter'))
# The Anthropic secret (the `sonnet` profile, the `describe` stage) is mounted only when named.
ANTHROPIC_SECRET = os.environ.get('EVAL_ANTHROPIC_SECRET', '')
JUDGE_SECRETS = [SECRET] + ([modal.Secret.from_name(ANTHROPIC_SECRET)] if ANTHROPIC_SECRET else [])
# Every variable read at import, pins included: Modal imports this file again inside each container.
IMPORT_ENV = ('EVAL_APP', 'EVAL_GPU', 'EVAL_GPU_BOTH_MODELS', 'EVAL_GPU_TRAINING', 'EVAL_GPU_WORKERS',
              'EVAL_MODEL_CACHE_VOLUME', 'EVAL_MODEL_CACHE_DIR', 'EVAL_OPENROUTER_SECRET',
              'EVAL_ANTHROPIC_SECRET',
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
    .pip_install_from_requirements(str(HERE/'requirements.txt'))
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


@app.function(image=image, volumes={'/cache': cache}, timeout=3600)
def prepare_weights(config_json: str):
    """Download the pinned base, inverter, NLA verbalizer and lens file once, on CPU."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from evals.downstream.common.nla import nla_reader
    from evals.downstream.common.pins import LENS_FILE, LENS_REPO, LENS_REVISION
    from evals.downstream.steering_vector_inversion.config import Config
    config = Config(**json.loads(config_json)).validate()
    weights = ['*.json', '*.safetensors', '*.jinja', '*.txt']
    for repo_id, revision in ((config.model, config.model_revision),
                              (config.inverter, config.inverter_revision)):
        snapshot_download(repo_id, revision=revision, allow_patterns=weights,
                          cache_dir=MODEL_CACHE_DIR, token=False)
    nla_reader.check_checkpoint(nla_reader.download_checkpoint(cache_dir=MODEL_CACHE_DIR))
    hf_hub_download(LENS_REPO, LENS_FILE, revision=LENS_REVISION, token=False)
    cache.commit()


@app.cls(image=image, gpu=GPU_SPEC, volumes={'/cache': cache}, max_containers=GPU_WORKERS,
         min_containers=0, buffer_containers=0, scaledown_window=300,
         timeout=3*3600, startup_timeout=1800)
class GPU:
    config_json: str = modal.parameter()
    role: str = modal.parameter(default='model')   # `model` (base, inverter) or `nla` (the verbalizer)

    @modal.enter()
    def load(self):
        from evals.downstream.steering_vector_inversion_bipo.backend import build_worker
        from evals.downstream.steering_vector_inversion.config import Config
        cache.reload()
        self.worker = build_worker(self.role, Config(**json.loads(self.config_json)), MODEL_CACHE_DIR)

    @modal.method()
    def execute(self, payload):
        from evals.downstream.steering_vector_inversion_bipo.backend import route
        return route(self.worker, payload)


@app.function(image=judge_image, secrets=JUDGE_SECRETS, max_containers=4, timeout=900, memory=2048)
@modal.concurrent(max_inputs=64)
def call(spec: dict, req: dict, title: str):
    """One judge request, with the key held here and never by the controller."""
    from evals.downstream.common.judge_client import JudgeSpec, client_for, key_for
    spec = JudgeSpec(**spec)
    return client_for(spec, key_for(spec), title=title).call(req)


class RemoteClient:
    """What `judge.run` needs of a client: `call(req) -> (text, usage)`, raising JudgeError."""
    def __init__(self, spec, title):
        self.spec, self.title = asdict(spec), title

    def call(self, req):
        return call.remote(self.spec, req, self.title)


@app.local_entrypoint()
def main(stage: str = 'all', run_id: str = '', output_dir: str = '', behaviours: str = '', smoke: bool = False,
         max_gpu_workers: int = 0, judge_budget_usd: float = 0.0, judge_workers: int = 0,
         family_filter: str = '', shard: str = '', judge_profile: str = 'sol', rewrite: bool = False,
         freeze: bool = False):
    """Flags left at 0 or "" take the package CLI's defaults. `--judge-profile` (`sol`, the paper's, or
    `sonnet`) is activated before the package is imported, since its config binds the judges."""
    from evals.downstream.common import judges
    judges.activate(judge_profile)
    from evals.downstream.steering_vector_inversion_bipo.config import JUDGE_CAP_USD
    from evals.downstream.steering_vector_inversion_bipo.__main__ import GPU_STAGES, main as cli, stages_of
    from evals.downstream.steering_vector_inversion_bipo.backend import role_of_stage
    if bool(run_id) == bool(output_dir):
        raise ValueError('Supply --run-id or --output-dir, not both and not neither')
    print(f'Judge ledgers cap this run at ${JUDGE_CAP_USD:.2f} in total', flush=True)
    options = ['--run-id', run_id] if run_id else ['--output-dir', output_dir]
    options += [judges.PROFILE_FLAG, judge_profile]
    for flag, value in (('--behaviours', behaviours),
                        ('--family-filter', family_filter),
                        ('--shard', shard),
                        ('--max-gpu-workers', max_gpu_workers or GPU_WORKERS), ('--judge-workers', judge_workers),
                        ('--judge-budget-usd', judge_budget_usd)):
        if value:
            options += [flag, str(value)]
    options += ['--smoke'] if smoke else []
    options += ['--rewrite'] if rewrite else []
    options += ['--freeze'] if freeze else []
    stages = stages_of(stage)
    config_json = None
    if set(stages) & GPU_STAGES:
        from evals.downstream.steering_vector_inversion_bipo.config import svi_config
        config_json = json.dumps(svi_config().to_dict(), sort_keys=True)
        prepare_weights.remote(config_json)
    for position, one in enumerate(stages, start=1):
        if len(stages) > 1:
            print(f'== stage {position} of {len(stages)}: {one}', flush=True)
        remote = (_gpu_cls(one)(config_json=config_json, role=role_of_stage(one)).execute.remote
                  if one in GPU_STAGES else None)
        cli([one, *options], remote=remote, judge_client=RemoteClient)


def _gpu_cls(stage):
    """The worker class on the card the stage needs."""
    if stage in BACKWARD_STAGES:
        return GPU.with_options(gpu=GPU_TRAINING)
    return GPU.with_options(gpu=GPU_BOTH_MODELS) if stage in TWO_MODEL_STAGES else GPU
