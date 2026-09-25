"""The NLA verbalizer as a reader of the same unit directions (`nla_native`), through
`eval/common/nla/nla_reader.py` on a worker that holds the verbalizer alone (`NlaWorker`). The close rate of
its `<explanation>` tags is reported, never enforced (methodology.md).
"""
import importlib.metadata
import time

import numpy as np

from .artifacts import digest, seed_for
from .execution import gpu_tasks

CONDITION = 'nla_native'
OPERATION = 'nla'
STAGE = 'nla'
ROLES = ('model', 'nla')


def role_of_stage(stage):
    """`nla` for the stage whose payloads the verbalizer's worker answers, `model` for every other."""
    return 'nla' if stage == STAGE else 'model'


def reader_pins():
    """(the reader's pins as a batch key carries them, its close-rate line)."""
    from eval.common.nla.nla_reader import PINS, pins_record

    return pins_record(PINS), PINS.min_close_rate


def sample_chunks(config):
    """The sample counts of one direction's tasks, each at most `config.generation_batch_size`."""
    counts, remaining = [], config.samples
    while remaining > 0:
        counts.append(min(config.generation_batch_size, remaining))
        remaining -= counts[-1]
    return counts


def tasks(config, bank):
    """One task per target direction and sample chunk, each under its own seed; the first chunk also
    carries the greedy explanation, and every payload the reader's pins."""
    record, _ = reader_pins()
    targets = set(bank['target_ids'])
    counts = sample_chunks(config)
    return [(f'{concept["concept_id"]}:{number}',
             {'operation': OPERATION, 'concept_id': concept['concept_id'],
              'direction': bank['directions'][index], 'n_samples': count, 'greedy': number == 0,
              'seed': seed_for(config.seed, concept['concept_id'], 'nla', number),
              'pins': record})
            for index, concept in enumerate(bank['concepts']) if concept['concept_id'] in targets
            for number, count in enumerate(counts)]


def families(results, samples):
    """One `nla_native` family per concept, chunks reassembled in task order so sample ids are stable."""
    chunks = {}
    for task_id, result in results:
        cid, number = task_id.split(':')
        chunks.setdefault(int(cid), {})[int(number)] = result['data']
    out = {}
    for cid, numbered in chunks.items():
        ordered = [numbered[number] for number in sorted(numbered)]
        greedy = [data['greedy'] for data in ordered if data.get('greedy') is not None]
        if len(greedy) != 1:
            raise ValueError('Exactly one chunk of a direction carries its greedy explanation')
        rows = [(-1, greedy[0]), *enumerate(row for data in ordered for row in data['samples'])]
        family = {'concept_id': cid, 'condition': CONDITION, 'status': 'ok', 'reason': None,
                  'close_rate': close_rate([row for _, row in rows[1:]]),
                  'rollouts': [{'sample_id': i, 'text': row['text'], 'full_text': row['full_text'],
                                'token_ids': row['text_ids'], 'n_tokens': len(row['text_ids']),
                                'closed': row['closed'], 'length_capped': row['capped'], 'seed': row['seed']}
                               for i, row in rows]}
        if [r['sample_id'] for r in family['rollouts']] != [-1, *range(samples)]:
            raise ValueError('Incomplete or duplicate NLA samples')
        out[f'{cid}:{CONDITION}'] = family
    return out


def close_rate(rows):
    """The share of a direction's sampled explanations whose tags closed."""
    return float(np.mean([row['closed'] for row in rows])) if rows else None


def close_rates(out):
    """The per-concept close rates of one NLA run, and the pooled rate over its directions."""
    by_concept = {family['concept_id']: family['close_rate'] for family in out.values()}
    rates = [rate for rate in by_concept.values() if rate is not None]
    return {'by_concept': by_concept, 'pooled': float(np.mean(rates)) if rates else None}


def generate(run, executor, bank):
    record, floor = reader_pins()
    started = time.time()
    results = list(gpu_tasks(run, executor, tasks(run.config, bank), 'rollouts/nla_batches'))
    out = families(results, run.config.samples)
    rates = close_rates(out) | {'min_close_rate': floor}
    rates['below_min_close_rate'] = rates['pooled'] is None or rates['pooled'] < floor
    key = run.key('nla', [digest(bank['directions']), record], modules=('nla', 'model'))
    run.save('rollouts/nla.json', key, out)
    run.save('rollouts/nla_close_rates.json', key, rates)
    run.stage_done('nla', started, families=len(out), close_rate=rates['pooled'], min_close_rate=floor)
    return out


def runtime_versions():
    versions = {}
    for package in ('torch', 'transformers', 'flash-linear-attention'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


class NlaWorker:
    """The GPU worker that holds the pinned verbalizer alone; it answers only the `nla` operation."""
    role = 'nla'

    def __init__(self, config, cache_dir, device="cuda:0"):
        from eval.common.nla import nla_reader

        self.config, self.cache_dir, self.device = config, cache_dir, str(device)
        self.reader, self.pins = nla_reader, nla_reader.PINS
        snapshot = nla_reader.download_checkpoint(cache_dir=str(cache_dir), pins=self.pins)
        self.checkpoint = nla_reader.check_checkpoint(snapshot, self.pins)
        self.verbalizer, self.tokenizer = nla_reader.load_verbalizer(self.device, snapshot, self.pins)
        if hasattr(self.verbalizer, 'requires_grad_'):
            self.verbalizer.requires_grad_(False)
        contract = nla_reader.contract(nla_reader.load_sidecar(pins=self.pins))
        if contract['d_model'] != config.hidden_size or contract['layer_index'] != config.read_layer:
            raise ValueError('The verbalizer was trained on another activation space than the one read here')
        self.prompt_ids, self.marker = nla_reader.prompt_ids(self.tokenizer, contract, self.pins)
        self.versions = runtime_versions()

    def explain(self, direction, n_samples, seed, greedy):
        """(samples, greedy record or None) for one direction, each record with the seed that drew it."""
        rows = np.asarray(direction, dtype=np.float32)[None]
        samples, greedy_out = self.reader.generate_explanations(
            self.verbalizer, self.tokenizer, rows, self.prompt_ids, self.marker, self.device,
            n_samples=n_samples, seed=seed, greedy=greedy, pins=self.pins)
        return samples[0], greedy_out[0] if greedy else None

    def close(self):
        """Give the verbalizer back, for a process that is about to host a model worker."""
        from eval.common import model_io

        self.verbalizer = model_io.free_model(self.verbalizer)

    def pins_data(self):
        return {'repo': self.pins.repo, 'revision': self.pins.revision, 'max_new': self.pins.max_new,
                'trunc': self.pins.trunc, 'score_max_length': self.pins.score_max_length,
                'checkpoint': self.checkpoint}

    def result(self, data, started):
        import torch

        return {'data': data, 'gpu_seconds': time.time() - started, 'runtime_versions': self.versions,
                'gpu_name': torch.cuda.get_device_name(self.device) if self.device.startswith('cuda') else 'CPU'}

    def execute(self, payload):
        started = time.time()
        if payload['operation'] != OPERATION:
            raise ValueError(f"This worker holds the NLA verbalizer alone and answers only the "
                             f"{OPERATION!r} operation, not {payload['operation']!r}")
        return self.result(execute(self, payload), started)


def refuse_on_model_worker(operation):
    """A model worker refuses the verbalizer's operation: that would be a third 27B checkpoint on one card."""
    raise ValueError(f"The {operation!r} operation runs on the verbalizer's own worker "
                     f"(steering_vector_inversion.nla.NlaWorker), not on a model worker")


def execute(worker, payload):
    """One chunk of a direction's explanations, on the verbalizer's own worker."""
    samples, greedy = worker.explain(payload['direction'], payload['n_samples'], payload['seed'],
                                     payload['greedy'])
    return {'concept_id': payload['concept_id'], 'samples': [described(sample) for sample in samples],
            'greedy': described(greedy) if payload['greedy'] else None, 'pins': worker.pins_data()}


def described(sample):
    """One generation: the text a judge reads, the full text, and the generated content ids."""
    content = sample['ids'][:sample['n_tokens']]
    return {'text': sample['text'], 'full_text': sample['full_text'], 'text_ids': content,
            'closed': sample['closed'], 'eos': sample['eos'], 'capped': sample['capped'],
            'seed': sample['seed']}
