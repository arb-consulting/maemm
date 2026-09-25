"""Every arm's texts per concept, the identification cases posed on them, and their per-concept scores.

`sources` gathers each arm's sampled and greedy texts per target concept. A case shows the judge a
bundle of texts (the first 1, 2, 4 or 8 samples, or the one greedy/single text; `config.budgets`) and the
concept's ten candidates. A bundle whose texts are all blank is not asked and counts as a miss; an arm
unavailable for the concept is left out of that concept's accuracy.
"""
from .analysis import finite_mean
from .config import CONDITIONS, GREEDY, JUDGES, SINGLE_TEXT, budgets
from .judge import NO_SYSTEM, case_key, identification_prompt, verdict

#: What names one identification case; the same fields are the request's `meta`.
SLOT = ("concept_id", "condition", "budget_type", "budget")


def answered_keys(answers):
    """One judge's answered case keys in a stable order, for a cache key over what it answered."""
    return sorted(answers, key=str)


def _family(families, family_id, condition):
    family = families.get(family_id)
    if family is None:
        raise ValueError(f'Missing required source: {family_id}')
    return {'status': family['status'], 'reason': family.get('reason'),
            'samples': [r for r in family['rollouts'] if r['sample_id'] >= 0],
            'greedy': [r for r in family['rollouts'] if r['sample_id'] == -1] if condition in GREEDY else []}


def sources(bank, prepared, families, windows, summaries):
    """`{"<concept>:<condition>": {concept_id, condition, status, samples, greedy}}` for every target concept
    and arm. `families` holds the generated arms (MAEM, base_l1, NLA, steered); `shuffled` is MAEM's text
    for the concept's donor; `retrieval`, `heldout_positive` and `jlens` are read from their artefacts."""
    result = {}
    for concept in bank['concepts']:
        cid = concept['concept_id']
        if cid not in bank['target_ids']:
            continue
        for condition in CONDITIONS:
            if condition == 'retrieval':
                value = {'status': 'ok', 'samples': [{'text': w['text'], 'n_tokens': w['n_tokens']}
                                                     for w in windows[str(cid)]], 'greedy': []}
            elif condition == 'jlens':
                summary = summaries[str(cid)]
                written = bool(summary['text'].strip())
                value = {'status': 'ok' if written else 'unavailable',
                         'reason': None if written else summary['status'],
                         'samples': [], 'greedy': [{'text': summary['text']}] if written else []}
            elif condition == 'heldout_positive':
                value = {'status': 'ok', 'greedy': [],
                         'samples': [prepared['texts'][r['text_id']] for r in concept['references']]}
            elif condition == 'shuffled':
                donor = bank['candidates'][str(cid)]['donor']
                value = {**_family(families, f'{donor}:maem', 'maem'), 'greedy': []}
            else:
                value = _family(families, f'{cid}:{condition}', condition)
            result[f'{cid}:{condition}'] = {'concept_id': cid, 'condition': condition, **value}
    return result


def case_bundles(condition, source):
    """The (budget type, budget, texts) an arm is asked about (`config.budgets`)."""
    return [(kind, budget, source['greedy'] if kind == 'greedy' else source['samples'][:budget])
            for kind, budget in budgets(condition)]


def identification_cases(bank, source_map):
    """Every identification case of the run, with its prompt, or None where it is not asked."""
    descriptions = {c['concept_id']: c['description'] for c in bank['concepts']}
    cases = []
    for source in source_map.values():
        cid, condition = source['concept_id'], source['condition']
        candidates = bank['candidates'][str(cid)]
        for kind, budget, snippets in case_bundles(condition, source):
            case = {'concept_id': cid, 'condition': condition, 'budget_type': kind, 'budget': budget,
                    'correct_answer': candidates['correct_answer'], 'input_status': source['status'],
                    'prompt': None}
            if source['status'] == 'ok' and any(s['text'].strip() for s in snippets):
                if len(snippets) != budget:
                    raise ValueError('An identification bundle is missing saved sample slots')
                case['prompt'] = identification_prompt([s['text'] for s in snippets],
                                                       [descriptions[c] for c in candidates['candidates']])
            cases.append(case)
    return cases


def identification_requests(cases):
    """The cases a judge is actually asked, addressed to no judge (`judge.ask` binds each one)."""
    return [{'system': NO_SYSTEM, 'user': case['prompt'], 'kind': 'identification',
             'meta': {field: case[field] for field in SLOT}}
            for case in cases if case['prompt'] is not None]


def case_accuracy(record):
    """A case's contribution to its arm's identification accuracy: None where the arm has no text for the
    concept (left out of the rate), else 1 for a correct answer and 0 otherwise (an unasked all-blank bundle
    is a miss). The rates and the paper column's paired tests both apply it."""
    if record['input_status'] != 'ok':
        return None
    return float(record['outcome'] == 'ok' and record['value'] == record['correct_answer'])


def finalize_identification(run, cases, answers):
    """Per judge and case: the record, and three per-concept metrics -- accuracy (an unanswered case is a
    miss), the answered share, and accuracy over the answered cases. Saved under `scores/`."""
    records, rows = [], []
    for judge in JUDGES:
        for case in cases:
            slot = {field: case[field] for field in SLOT}
            value, outcome = ((None, 'unavailable') if case['prompt'] is None
                              else verdict(answers[judge].get(case_key(slot))))
            available = case['input_status'] == 'ok'
            correct = outcome == 'ok' and value == case['correct_answer']
            record = slot | {'judge': judge, 'value': value, 'outcome': outcome,
                             'correct_answer': case['correct_answer'],
                             'input_status': case['input_status'], 'asked': case['prompt'] is not None}
            records.append(record)
            for metric, estimate in (('identification_accuracy', case_accuracy(record)),
                                     ('identification_valid_rate', float(outcome == 'ok') if available else None),
                                     ('identification_answered_accuracy', float(correct) if outcome == 'ok' else None)):
                rows.append(slot | {'judge': judge, 'metric': metric, 'estimate': estimate, 'outcome': outcome})
    key = run.key('identification', [answered_keys(a) for a in answers.values()],
                  modules=('evaluate', 'judge'))
    run.save('scores/identification_cases.json', key, records)
    run.save('scores/identification.json', key, rows)
    return rows


def missingness_rows(records):
    """One row per (judge, arm): cases, cases asked, and how each ended, so a judge that refuses one arm
    more than another is not read as that arm performing worse."""
    from .judge import STATUSES

    groups = {}
    for record in records:
        groups.setdefault((record['judge'], record['condition']), []).append(record)
    rows = []
    for (judge, condition), group in sorted(groups.items()):
        counts = {status: sum(r['outcome'] == status for r in group) for status in STATUSES}
        rows.append({'judge': judge, 'condition': condition, 'n_cases': len(group),
                     'n_requested': sum(r['asked'] for r in group), **counts})
    return rows


def n_tokens(snippet):
    """A text's length in base-tokenizer tokens: its token ids where it carries them, else `n_tokens`."""
    ids = snippet.get('token_ids')
    return len(ids) if ids is not None else snippet.get('n_tokens')


def text_length_rows(source_map):
    """Per concept and arm: the mean length in tokens of the arm's sampled texts (`text_length` metric).
    The lens summary carries no token count and is left out; `shuffled` repeats MAEM's texts."""
    rows = []
    for source in source_map.values():
        if source['condition'] in (*SINGLE_TEXT, 'shuffled'):
            continue
        lengths = [n_tokens(s) for s in source['samples']]
        rows.append({'concept_id': source['concept_id'], 'condition': source['condition'],
                     'metric': 'text_length', 'judge': '', 'budget_type': 'sampled', 'budget': None,
                     'estimate': finite_mean(lengths) if source['status'] == 'ok' else None})
    return rows

