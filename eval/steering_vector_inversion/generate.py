"""The generated arms: MAEMM and the untrained base reading a direction, and the steered model.

MAEMM is the inverter under its trained research prompt with the direction injected at the marker on
layer 1; `base_l1` is the same prompt and injection on the clean base. The steered model is the clean base
with the direction added at layer 42, generating from a sink token (`eval.common.plain_steer`). Every
family is `{concept_id, condition, status, reason, rollouts}`, `sample_id` -1 for the greedy row.
"""
import time

import numpy as np

from eval.common import plain_steer

from .artifacts import digest, seed_for
from .config import STEER_GREEDY, STEER_SAMPLES
from .execution import chunks, gpu_tasks

#: The inversion arms and the model each generates on.
INVERSION = {"maemm": "inverter", "base_l1": "base"}


def inversion_jobs(config, concept, condition, direction, model='inverter'):
    """One greedy row and `config.samples` sampled ones for a condition, under that condition's seeds."""
    return [{'concept_id': concept['concept_id'], 'condition': condition, 'sample_id': sample,
             'greedy': sample == -1, 'kind': 'inversion', 'model': model, 'direction': direction,
             'seed': seed_for(config.seed, concept['concept_id'], condition, sample)}
            for sample in [-1, *range(config.samples)]]


def rollouts(run, executor, bank):
    """MAEMM and `base_l1` for every target concept, and MAEMM for every donor the shuffled arm reads."""
    started = time.time()
    targets = set(bank['target_ids'])
    donors = {bank['candidates'][str(i)]['donor'] for i in targets}
    families, tasks, task_metadata = {}, [], {}
    for index, concept in enumerate(bank['concepts']):
        cid = concept['concept_id']
        for condition, model in INVERSION.items():
            if cid not in targets and not (condition == 'maemm' and cid in donors):
                continue
            family_id = f'{cid}:{condition}'
            families[family_id] = {'concept_id': cid, 'condition': condition, 'status': 'ok', 'reason': None,
                                   'rollouts': []}
            jobs = inversion_jobs(run.config, concept, condition, bank['directions'][index], model=model)
            for number, batch in enumerate([jobs[:1], *chunks(jobs[1:], run.config.generation_batch_size)]):
                task_id = f'{family_id}:{number}'
                task_metadata[task_id] = family_id
                tasks.append((task_id, {'operation': 'generate', 'jobs': batch}))
    for task_id, result in gpu_tasks(run, executor, tasks, 'rollouts/inversion'):
        families[task_metadata[task_id]]['rollouts'].extend(result['data'])
    for family in families.values():
        family['rollouts'].sort(key=lambda r: r['sample_id'])
        if [r['sample_id'] for r in family['rollouts']] != [-1, *range(run.config.samples)]:
            raise ValueError('Incomplete or duplicate inversion samples')
    key = run.key('rollouts', digest(bank['directions']), modules=('generate', 'model'))
    run.save('rollouts/rollouts.json', key, families)
    run.stage_done('rollouts', started, families=len(families))
    return families


def steered_vector_id(concept_id):
    return f'axbench:{int(concept_id)}'


def _unit64(vector):
    """The direction at unit length, normalised in float64 and handed over as float32."""
    vector = np.asarray(vector, dtype=np.float64)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def steered_payloads(bank):
    """The steered model's GPU payloads over every target concept, in bank order (`plain_steer.payloads`)."""
    members = [(c['concept_id'], steered_vector_id(c['concept_id']), _unit64(bank["directions"][index]))
               for index, c in enumerate(bank['concepts']) if c['concept_id'] in set(bank['target_ids'])]
    directions = {vid: d for _, vid, d in members}
    extra = {vid: {'concept_id': int(cid)} for cid, vid, _ in members}
    cells = plain_steer.cells([vid for _, vid, _ in members], STEER_SAMPLES, STEER_GREEDY)
    return plain_steer.payloads(directions, cells, extra)


def steered(run, executor, bank):
    """The steered model at every strength, as one family per (concept, `plain_steered@<s>`)."""
    started = time.time()
    tasks = list(enumerate(steered_payloads(bank)))
    families = {}
    for _task_id, result in gpu_tasks(run, executor, tasks, 'rollouts/plain_steered'):
        for row in result['data']['rows']:
            condition = plain_steer.arm_name(row['strength'])
            family = families.setdefault(f"{row['concept_id']}:{condition}", {
                'concept_id': row['concept_id'], 'condition': condition, 'status': 'ok', 'reason': None,
                'strength': float(row['strength']), 'rollouts': []})
            family['rollouts'].append(row)
    for family in families.values():
        family['rollouts'].sort(key=lambda r: r['sample_id'])
        n = STEER_SAMPLES[family['strength']]
        expected = ([-1] if family['strength'] in STEER_GREEDY else []) + list(range(n))
        if [r['sample_id'] for r in family['rollouts']] != expected:
            raise ValueError(f"Incomplete or duplicate steered samples for {family['concept_id']}")
    key = run.key('plain_steered', digest(bank['directions']), modules=('generate', 'model'))
    run.save('rollouts/plain_steered.json', key, families)
    run.stage_done('plain-steer', started, families=len(families))
    return families
