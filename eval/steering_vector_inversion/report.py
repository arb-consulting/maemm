"""CPU-only tables, one diagnostic figure, the paper's AxBench column and figure, and report.md, from a run's
saved artefacts.

Tables (under `tables/`): `identification.csv` (every arm, judge, budget and genre group, with stratified
bootstrap intervals), `paired_differences.csv` (MAEMM minus each arm, per concept), `text_length.csv`
(mean tokens per text, per arm), `missingness.csv`, `plain_steered_health.csv` (the steered model's text
health per concept and strength) and `coverage_and_costs.json`. `paper.py` writes the paper's column and
budget figure from the same rows.
"""
from collections import Counter
import os
import shlex

import numpy as np
import pandas as pd

from eval.common import plain_steer
from eval.common import retrieval as R
from eval.common.judge_client import served_line, unasked_detail
from eval.common.judges import REFERENCE_JUDGE

from . import paper
from .analysis import paired_rows, summarize_rows
from .artifacts import digest, read_json, write_json
from .config import (BUDGETS, CAPS, CONDITIONS, CURVE_BUDGETS, JUDGES, JUDGE_PROFILE, STEER_SAMPLES, STEERED,
                     STEERED_TABLE, budgets)
from .evaluate import missingness_rows, text_length_rows
from .judge import NotFullyAsked, spend, unasked_cases

LABELS = {'maemm': 'MAEMM', 'base_l1': 'Untrained base', 'nla_native': 'NLA verbalizer',
          'shuffled': 'Shuffled', 'retrieval': 'Corpus search', 'heldout_positive': 'Held-out positives',
          'jlens': 'J-lens, summarised',
          **{arm: f'Steered model, s={s:g}' for arm, s in zip(STEERED, STEER_SAMPLES)}}
COLORS = {'maemm': '#0072B2', 'base_l1': '#777777', 'nla_native': '#490092', STEERED_TABLE: '#D55E00',
          'shuffled': '#999999', 'retrieval': '#E69F00', 'heldout_positive': '#009E73', 'jlens': '#B66DFF'}
#: The arms figure 01 plots.
FIGURE_ARMS = ('maemm', 'base_l1', 'nla_native', STEERED_TABLE, 'shuffled', 'retrieval', 'heldout_positive',
               'jlens')
COLUMNS = ['metric', 'condition', 'judge', 'group', 'budget_type', 'budget', 'estimate', 'ci_lower',
           'ci_upper', 'n_total', 'n_valid', 'ci_method']
IDENTIFICATION_METRICS = ('identification_accuracy', 'identification_valid_rate',
                          'identification_answered_accuracy')
HEALTH_TABLE = 'tables/plain_steered_health.csv'


def saved(run, name):
    return read_json(run.root / name)['data']


def require(run, *relatives):
    """Every artefact the render reads, checked before it reads any."""
    missing = [r for r in relatives if not (run.root / r).is_file()]
    if missing:
        raise ValueError('Missing run artifacts; finish the stages that write them before reporting: '
                         + ', '.join(missing))


def require_fully_asked(run):
    """Refuse a run with a request never answered (ledger, transport or key): a rate over it would read a
    cap or an outage as a result. Each case's last record counts."""
    for instrument in CAPS:
        for judge, keys in unasked_cases(run, instrument, JUDGES).items():
            counts = {kind: len(ks) for kind, ks in keys.items()}
            if sum(counts.values()):
                state = spend(run, instrument)
                raise NotFullyAsked(
                    f"{instrument}: {sum(counts.values()):,} of {judge}'s requests were never asked "
                    f"({unasked_detail(counts)}), after US${state['spent_usd']:.4f} of the "
                    f"US${state['cap_usd']:.2f} cap. Run that judge stage again before reporting "
                    f"(raising config.CAPS['{instrument}'] if the cap stopped it).")


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def require_coverage(bank, identification, cases):
    """Every target concept has exactly one row per arm, judge, budget and metric."""
    targets = bank['target_ids']
    slots = [(c, kind, b) for c in CONDITIONS for kind, b in budgets(c)]
    fields = ('concept_id', 'condition', 'judge', 'budget_type', 'budget')
    for name, rows, expected in (
            ('Identification', identification,
             {(cid, c, judge, kind, b, m) for cid in targets for c, kind, b in slots for judge in JUDGES
              for m in IDENTIFICATION_METRICS}),
            ('Identification cases', cases,
             {(cid, c, judge, kind, b) for cid in targets for c, kind, b in slots for judge in JUDGES})):
        keys = fields + (('metric',) if name == 'Identification' else ())
        observed = [tuple(row.get(k) for k in keys) for row in rows]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(f'{name} rows do not cover the target population exactly; finish all before '
                             'reporting')


def health_rows(steered, genres):
    """`plain_steer.health` of the steered model's samples, per concept and strength, with its genre."""
    rows = [r for family in steered.values() for r in family['rollouts']]
    out = plain_steer.health(rows)
    for row in out:
        row['concept_id'] = int(str(row['vector_id']).split(':')[1])
        row['genre'] = genres.get(row['concept_id'])
    return out


def render_figure(root, rows, counts, smoke):
    """`figures/01_identification`: identification against texts shown, one panel per genre group."""
    os.environ.setdefault('MPLCONFIGDIR', str((root/'cache'/'matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    (root / 'figures').mkdir(exist_ok=True)
    data = pd.DataFrame(rows, columns=COLUMNS)
    data = data[(data.judge == REFERENCE_JUDGE) & (data.metric == 'identification_accuracy')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, group in zip(axes.flat, ('all', 'text', 'code', 'math')):
        for condition in FIGURE_ARMS:
            part = data[(data.condition == condition) & (data.group == group)]
            if condition == 'jlens':
                part = part[part.budget_type == 'greedy']
                if len(part):
                    ax.axhline(float(part.estimate.iloc[0]), color=COLORS[condition], linewidth=1.2,
                               linestyle='-.', label=LABELS[condition])
                continue
            part = part[part.budget_type == 'snippets'].sort_values('budget')
            if part.empty:
                continue
            x = part.budget.to_numpy(float)
            ax.plot(x, part.estimate.to_numpy(float), marker='o', color=COLORS[condition],
                    label=LABELS[condition], linestyle='--' if condition == 'shuffled' else '-')
            ax.fill_between(x, part.ci_lower.to_numpy(float), part.ci_upper.to_numpy(float),
                            color=COLORS[condition], alpha=.12)
        ax.axhline(.1, color='black', linewidth=.8, linestyle=':', label='Chance')
        ax.set(title=f'{group.title()} (n={counts[group]} concepts)', xlabel='Texts shown together',
               ylabel='Identification accuracy', ylim=(0, 1), xticks=BUDGETS)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=5, fontsize=8)
    fig.suptitle(('SMOKE — ' if smoke else '') + 'Concept identification')
    fig.tight_layout(rect=(0, .06, 1, .96))
    for ext in ('pdf', 'png'):
        fig.savefig(root / 'figures' / f'01_identification.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def value(summary, condition, kind, budget, group='all', metric='identification_accuracy'):
    """`0.958 [0.934, 0.979]` for one summary cell of the reference judge, `--` where it is missing."""
    hit = [r for r in summary if (r['condition'], r['budget_type'], r['budget'], r['group'], r['metric'],
                                  r['judge']) == (condition, kind, budget, group, metric, REFERENCE_JUDGE)]
    if not hit or hit[0]['estimate'] is None:
        return '--'
    r = hit[0]
    return f"{r['estimate']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"


def arms_table(summary, lengths):
    """report.md's main table: accuracy per arm at 1 and 8 texts (all concepts and text concepts), and the
    arm's mean text length on text concepts."""
    length = {r['condition']: r['estimate'] for r in lengths if r['group'] == 'text'}
    lines = ['| arm | all, 1 text | all, 8 texts | text, 1 text | text, 8 texts | greedy / single text (all) '
             '| mean tokens (text) |', '|---|---|---|---|---|---|---|']
    for condition in CONDITIONS:
        if condition in STEERED and condition != STEERED_TABLE:
            continue
        slots = set(budgets(condition))
        cell = lambda b, g: value(summary, condition, 'snippets', b, g) if ('snippets', b) in slots else '--'
        greedy = value(summary, condition, 'greedy', 1) if ('greedy', 1) in slots else '--'
        tokens = f"{length[condition]:.1f}" if length.get(condition) is not None else '--'
        lines.append(f'| {LABELS[condition]} (`{condition}`) | {cell(1, "all")} | {cell(8, "all")} | '
                     f'{cell(1, "text")} | {cell(8, "text")} | {greedy} | {tokens} |')
    return lines


def steered_lines(summary):
    """The strength curve at 1 and 8 texts, all concepts and text concepts."""
    lines = ['| strength | all, 1 text | all, 8 texts | text, 1 text | text, 8 texts |', '|---|---|---|---|---|']
    for arm, s in zip(STEERED, STEER_SAMPLES):
        lines.append(f'| {s:g} | ' + ' | '.join(value(summary, arm, 'snippets', b, g)
                                                for g in ('all', 'text') for b in CURVE_BUDGETS) + ' |')
    return lines


def render(run):
    require(run, 'vectors/bank.json', 'data/prepared.json', 'scores/identification.json',
            'scores/identification_cases.json', 'scores/sources.json', 'rollouts/nla_close_rates.json',
            'rollouts/plain_steered.json', 'retrieval/windows.json', 'lens/tokens.json', 'lens/summaries.json',
            *[f'judges/ledger_{instrument}.json' for instrument in CAPS])
    require_fully_asked(run)
    bank = run.cached_arrays('vectors/bank.json', read_json(run.root / 'vectors/bank.json')['cache_key'])
    prepared = saved(run, 'data/prepared.json')
    starting, exclusions = prepared['starting_concepts'], prepared['exclusions']
    del prepared
    identification = saved(run, 'scores/identification.json')
    cases = saved(run, 'scores/identification_cases.json')
    source_map = saved(run, 'scores/sources.json')
    require_coverage(bank, identification, cases)
    concepts = [c for c in bank['concepts'] if c['concept_id'] in bank['target_ids']]
    genres = {c['concept_id']: c['genre'] for c in concepts}
    summarize = lambda rows: summarize_rows(rows, concepts, run.config.bootstrap_samples)
    pairs = [row for comparator in CONDITIONS if comparator != 'maemm'
             for row in paired_rows(identification, 'maemm', comparator)]
    tables = {'identification': summarize(identification), 'paired_differences': summarize(pairs),
              'text_length': summarize(text_length_rows(source_map))}
    for name, rows in tables.items():
        write_table(run.root/'tables'/f'{name}.csv', rows)
    missingness = missingness_rows(cases)
    pd.DataFrame(missingness).to_csv(run.root/'tables'/'missingness.csv', index=False)
    pd.DataFrame(health_rows(saved(run, 'rollouts/plain_steered.json'), genres)).to_csv(
        run.root / HEALTH_TABLE, index=False)
    costs = {instrument: spend(run, instrument) for instrument in CAPS}
    total_cost = sum(c['spent_usd'] for c in costs.values())
    # streamed: a run holds tens of thousands of batch records, gigabytes in all
    gpu_seconds, gpu_names, runtimes = 0., set(), {}
    for path in run.root.glob('**/batches/*.json'):
        record = read_json(path)['data']
        gpu_seconds += record.get('gpu_seconds', 0)
        if record.get('gpu_name'):
            gpu_names.add(record['gpu_name'])
        if record.get('runtime_versions'):
            runtimes.setdefault(digest(record['runtime_versions']), record['runtime_versions'])
    corpus_search = saved(run, 'retrieval/windows.json')
    coverage = {'starting_concepts': starting, 'usable_vector_bank': len(bank['concepts']),
                'evaluated_concepts': len(concepts), 'by_genre': dict(Counter(genres.values())),
                'preparation_exclusions': exclusions, 'vector_exclusions': bank['exclusions'],
                'judges': {name: spec.model for name, spec in JUDGES.items()}, 'judge_profile': JUDGE_PROFILE,
                'judge_spend': costs, 'judge_caps_usd': dict(CAPS),
                'nla_close_rates': saved(run, 'rollouts/nla_close_rates.json'),
                'corpus_search': {k: v for k, v in corpus_search.items() if k not in ('windows', 'shared_windows')},
                'lens': saved(run, 'lens/tokens.json')['lens'],
                'lens_summaries_written': saved(run, 'lens/summaries.json')['written'],
                'identification_missingness': missingness,
                'gpu_runtime_versions': list(runtimes.values()), 'gpu_processing_hours': gpu_seconds/3600,
                'gpu_names': sorted(gpu_names), 'stages': read_json(run.root/'provenance.json')['stages']}
    write_json(run.root/'tables/coverage_and_costs.json', coverage)
    smoke = run.config.concepts_per_genre is not None
    counts = {g: sum(g == 'all' or c['genre'] == g for c in concepts) for g in ('all', 'text', 'code', 'math')}
    render_figure(run.root, tables['identification'], counts, smoke)
    paper.axbench_column(run.root, tables['identification'], cases, concepts, REFERENCE_JUDGE)
    paper.budget_figure(run.root, tables['identification'], REFERENCE_JUDGE)
    spec = JUDGES[REFERENCE_JUDGE]
    closed = saved(run, 'rollouts/nla_close_rates.json')['pooled']
    lines = ['# Steering-vector inversion (AxBench Concept500)', '',
             '**Smoke results; not a full-population result.**' if smoke else 'Results for this saved run.', '',
             f'Base model `{run.config.model}`, inverter `{run.config.inverter}`. Judge profile `{JUDGE_PROFILE}`: '
             f'{spec.label} (`{spec.model}`); {served_line(run.root, JUDGES)}. '
             f'{len(concepts)} evaluated concepts of {len(bank["concepts"])} usable directions '
             f'(dataset: {starting}). Arms are defined in `methodology.md`.', '',
             '## Identification', '',
             'Ten-candidate identification rate with 95 % stratified bootstrap intervals over concepts, the '
             'reference judge. A case not answered counts as a miss (`tables/missingness.csv` says how often).', '',
             *arms_table(tables['identification'], tables['text_length']), '',
             '![Identification](figures/01_identification.png)', '',
             '[Identification](tables/identification.csv) · [MAEMM minus each arm, paired per concept]'
             '(tables/paired_differences.csv) · [Text length](tables/text_length.csv)', '',
             f'NLA explanations closed their tags on {closed:.3f} of samples (reported, not enforced; an '
             'unclosed explanation is read whole).' if closed is not None else '', '',
             '## Steered model', '',
             f'The clean base generating {plain_steer.NEW_TOKENS} tokens from a sink token while '
             f'`s x {plain_steer.STEER_UNIT:.2f} x unit(direction)` is added at block 42. The paper reads '
             f'`{STEERED_TABLE}`; the strength curve:', '', *steered_lines(tables['identification']), '',
             f'[Text health per concept and strength]({HEALTH_TABLE}).', '',
             '## Corpus search', '',
             f'{corpus_search["corpus"]["n_docs"]:,} held-out documents, {corpus_search["corpus"]["n_tokens"]:,} '
             f'tokens, {corpus_search["corpus"]["n_windows"]:,} windows; {corpus_search["near_duplicates"]} of '
             f'{len(corpus_search["windows"])} directions have a best window above '
             f'{corpus_search["near_duplicate_cos"]}.', '', *R.search_table(corpus_search.get('search') or []), '',
             '## Paper table and figure', '',
             f'The AxBench column of the main steering table (text concepts, {paper.TEXTS} texts, Holm-corrected '
             f'paired sign-flip tests of MAEMM against each reader and control) is [{paper.PAPER_TABLE}.tex]'
             f'({paper.PAPER_TABLE}.tex), with p-values in [{paper.PAPER_TABLE}.csv]({paper.PAPER_TABLE}.csv); the '
             f'budget curve is [{paper.PAPER_FIGURE}.pdf]({paper.PAPER_FIGURE}.pdf).', '',
             f'![AxBench text concepts: identification against texts shown]({paper.PAPER_FIGURE}.png)', '',
             '## Examples', '',
             "Three seeded concepts per genre, each arm's first text. Illustrations, not results.", '']
    rng = np.random.default_rng(0)
    for genre in ('text', 'code', 'math'):
        available = [c for c in concepts if c['genre'] == genre]
        for index in rng.choice(len(available), min(3, len(available)), replace=False) if available else ():
            concept = available[index]
            lines += [f'### {genre}: concept {concept["concept_id"]}', '', concept['description'], '']
            for condition in ('maemm', 'nla_native', STEERED_TABLE, 'retrieval', 'jlens', 'heldout_positive'):
                source = source_map[f'{concept["concept_id"]}:{condition}']
                shown = source['samples'] or source['greedy']
                text = shown[0]['text'] if shown else '[unavailable]'
                lines += [f'**{LABELS[condition]}:**', '', *['    ' + line for line in text.splitlines()], '']
    lines += ['## Costs and commands', '']
    for instrument, state in costs.items():
        lines.append(f'`{instrument}`: {state["requests"]:,} judge requests, US${state["spent_usd"]:.4f} of a '
                     f'US${state["cap_usd"]:.2f} cap.')
    lines += [f'Total judge spend: US${total_cost:.4f}. Summed GPU processing time: {gpu_seconds/3600:.3f} hours '
              '(excludes idle containers and loading).', '',
              'Regenerate from saved artifacts, on CPU:', '', '```bash',
              f'python -m eval.steering_vector_inversion report --run-id {shlex.quote(run.root.name)}', '```', '']
    (run.root/'report.md').write_text('\n'.join(lines))
    return coverage
