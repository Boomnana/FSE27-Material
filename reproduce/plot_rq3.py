"""Plot current-reference ARI against actual query use on the saved runs."""
from collections import defaultdict
from math import comb, isclose
from statistics import mean
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (COLORS, DATASETS, MODELS, MODEL_NAMES, PAPER, STARTS,
                    read_csv, save, style, verify_inputs, write_rows)


def collect(rows):
    selected = [r for r in rows if r['rq'] == 'RQ3' and r['family'] == 'score']
    records = {}
    metadata = {}
    for row in selected:
        key = row['run'], row['stage']
        values = records.setdefault(key, {})
        assert row['metric'] not in values, (key, row['metric'])
        assert row['before'] != '', (key, row['metric'], row['before_status'])
        values[row['metric']] = float(row['before'])
        metadata[key] = row
    assert len(records) == 48 * 13
    stages = sorted({stage for _, stage in records}, key=float)
    assert len(stages) == 13
    runs = {run for run, _ in records}
    assert len(runs) == 48
    for run in runs:
        assert {stage for r, stage in records if r == run} == set(stages)
        q = [records[run, stage]['actual_q'] for stage in stages]
        assert q == sorted(q), run

    
    endpoints = {(r['rq'], r['run'], r['stage'], r['metric']): r
                 for r in rows if r['rq'] in ('RQ1', 'RQ2') and r['family'] == 'score'}
    cross_checks = 0
    for run in runs:
        for rq, stage, checkpoint in [('RQ1', 'P0', '0'), ('RQ1', 'P_B', '0.05'),
                                      ('RQ2', 'P_B', '0.05')]:
            for metric in ('ari', 'pair_fp', 'pair_fn', 'actual_q'):
                other = endpoints[rq, run, stage, metric]
                assert other['before'] != ''
                assert isclose(records[run, checkpoint][metric], float(other['before']),
                               rel_tol=0, abs_tol=1e-12), (rq, run, stage, metric)
                cross_checks += 1
            assert (metadata[run, checkpoint]['archived_partition_id'] ==
                    endpoints[rq, run, stage, 'ari']['archived_partition_id'])

    grouped = defaultdict(list)
    for key, values in records.items():
        row = metadata[key]
        n = int(values['n'])
        assert values['n'] == n and n >= 2
        grouped[row['dataset'], row['model'], row['start'], row['stage']].append({
            'repeat': int(row['repeat']), 'ari': values['ari'],
            'actual_q': values['actual_q'], 'n': n,
            'actual_budget_pct': 100 * values['actual_q'] / comb(n, 2),
        })
    result = []
    for ds in DATASETS:
        for model in MODELS:
            for start in STARTS:
                for stage in stages:
                    values = grouped[ds, model, start, stage]
                    assert sorted(v['repeat'] for v in values) == [1, 2, 3]
                    assert len({v['n'] for v in values}) == 1
                    result.append({
                        'dataset': ds, 'model': model, 'start': start,
                        'checkpoint': stage, 'n': values[0]['n'], 'runs': len(values),
                        'actual_q_mean': mean(v['actual_q'] for v in values),
                        'actual_budget_pct_mean': mean(v['actual_budget_pct'] for v in values),
                        'ari_mean': mean(v['ari'] for v in values),
                        'ari_min': min(v['ari'] for v in values),
                        'ari_max': max(v['ari'] for v in values),
                    })
    return result, {'run_checkpoints': len(records), 'config_checkpoints': len(result),
                    'cross_rq_numeric_checks': cross_checks,
                    'cross_rq_partition_checks': len(runs) * 3,
                    'reference_value_field': 'before', 'x_coordinate': 'mean(100*q/C(n,2))'}


def main():
    manifest = verify_inputs()
    values, checks = collect(read_csv('run_metric_diffs.csv'))
    style()
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.088, right=.994, bottom=.20, top=.80, wspace=.14, hspace=.20)
    for i, start in enumerate(STARTS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#d9dee4', linewidth=.55)
            ax.axvline(5, color='#87939e', linestyle=(0, (3, 2)), linewidth=.8)
            for model in MODELS:
                selected = [r for r in values if (r['dataset'], r['model'], r['start']) ==
                            (ds, model, start)]
                x = [r['actual_budget_pct_mean'] for r in selected]
                y = [r['ari_mean'] for r in selected]
                ax.fill_between(x, [r['ari_min'] for r in selected],
                                [r['ari_max'] for r in selected], color=COLORS[model],
                                alpha=.10, linewidth=0)
                ax.plot(x, y, color=COLORS[model], marker='o' if model == MODELS[0] else 's',
                        linestyle='-' if model == MODELS[0] else '--', markersize=2.5)
            ax.set_xlim(-.15, 10.15)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 5, 10])
            ax.set_yticks([0, .5, 1])
            ax.tick_params(length=2.5, color='#87939e', labelsize=7.4, pad=2)
            if i == 0:
                ax.set_title(ds.upper(), fontweight='bold', pad=4, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f'{start} start\nARI', fontsize=8, labelpad=3)
    handles = [Line2D([], [], color=COLORS[model], marker='o' if i == 0 else 's',
                      linestyle='-' if i == 0 else '--', label=MODEL_NAMES[model])
               for i, model in enumerate(MODELS)]
    handles.append(Line2D([], [], color='#87939e', linestyle='--', linewidth=.8,
                          label='5% checkpoint'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.0),
               ncol=3, frameon=False, columnspacing=1.5, fontsize=7.5)
    fig.supxlabel('Logical queries (% of report pairs)', y=.02, fontsize=8)
    write_rows('F3_budget_quality_trajectories', values)
    checks['reference_id'] = manifest['reference_id']
    (PAPER / 'results/plot_data/F3_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    save(fig, 'F3_budget_quality_trajectories')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
