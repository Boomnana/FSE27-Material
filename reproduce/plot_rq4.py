"""Plot current-reference error transitions and evidence at maintained recovery."""
from collections import defaultdict
from math import isclose
from statistics import mean
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from common import (DATASETS, MODELS, MODEL_NAMES, PAPER, STARTS, read_csv,
                    save, style, verify_inputs, write_rows)

TRANSITIONS = ('fn_removed', 'fn_introduced', 'fp_removed', 'fp_introduced')
EVIDENCE = ('direct', 'indirect', 'other', 'FN_Q', 'FN_E', 'FN_C',
            'FP_Q', 'FP_E', 'FP_C', 'FN_C_over_E', 'FP_C_over_E',
            'direct_share', 'indirect_share', 'other_share')
SHARES = ('direct_share', 'indirect_share', 'other_share')
ERROR_COLORS = {'fn': '#247f90', 'fp': '#bf7630'}
SHARE_COLORS = {'direct_share': '#0072B2', 'indirect_share': '#E69F00',
                'other_share': '#999999'}


def collect(rows):
    records, metadata, statuses = {}, {}, {}
    for row in rows:
        if row['rq'] != 'RQ4' or row['family'] != 'process':
            continue
        values = records.setdefault(row['run'], {})
        assert row['metric'] not in values
        values[row['metric']] = float(row['before']) if row['before'] != '' else None
        statuses[row['run'], row['metric']] = row['before_status']
        metadata[row['run']] = row
    assert len(records) == 48
    evidence_checks = 0
    for run, values in records.items():
        assert values['q_probe'] <= values['q_5']
        counts = [values[m] for m in ('direct', 'indirect', 'other')]
        if all(v is not None for v in counts):
            assert isclose(sum(counts), values['recovered'], abs_tol=1e-12), run
            for metric, count in zip(SHARES, counts):
                if values['recovered']:
                    assert isclose(values[metric], count / values['recovered'], abs_tol=1e-12), (run, metric)
                else:
                    assert values[metric] is None and not statuses[run, metric].startswith('missing'), (run, metric)
                evidence_checks += 1
        for error in ('FN', 'FP'):
            q, e, c = [values[f'{error}_{m}'] for m in ('Q', 'E', 'C')]
            if all(v is not None for v in (q, e, c)):
                assert 0 <= c <= e <= q, (run, error, q, e, c)
                ratio = values[f'{error}_C_over_E']
                if e:
                    assert isclose(ratio, c / e, abs_tol=1e-12), (run, error)
                else:
                    assert ratio is None, (run, error)
                evidence_checks += 1

    endpoints = {(r['rq'], r['run'], r['stage'], r['metric']): r
                 for r in rows if r['rq'] in ('RQ1', 'RQ3') and r['family'] == 'score'}
    endpoint_checks = conservation_checks = 0
    for run, values in records.items():
        for metric, process_metric in [('pair_fp', 'fp_5'), ('pair_fn', 'fn_5'), ('actual_q', 'q_5')]:
            for rq, stage in [('RQ1', 'P_B'), ('RQ3', '0.05')]:
                assert isclose(float(endpoints[rq, run, stage, metric]['before']),
                               values[process_metric], abs_tol=1e-12, rel_tol=0)
                endpoint_checks += 1
        for error in ('fn', 'fp'):
            if values[f'{error}_probe'] is None:
                assert all(values[f'{error}_{m}'] is None for m in ('removed', 'introduced'))
                continue
            assert isclose(values[f'{error}_probe'] - values[f'{error}_removed'] +
                           values[f'{error}_introduced'], values[f'{error}_5'], abs_tol=1e-12)
            assert isclose(float(endpoints['RQ1', run, 'Pprobe', f'pair_{error}']['before']),
                           values[f'{error}_probe'], abs_tol=1e-12, rel_tol=0)
            conservation_checks += 1
    grouped = defaultdict(list)
    for run, values in records.items():
        row = metadata[run]
        grouped[row['dataset'], row['model'], row['start']].append((run, row, values))
    result = []
    for ds in DATASETS:
        for model in MODELS:
            for start in STARTS:
                selected = grouped[ds, model, start]
                assert sorted(int(r['repeat']) for _, r, _ in selected) == [1, 2, 3]
                for metric in (*TRANSITIONS, *EVIDENCE):
                    observed = [v[metric] for _, _, v in selected if v[metric] is not None]
                    missing = sum(statuses[run, metric].startswith('missing') for run, _, _ in selected)
                    value = mean(observed) if observed and not missing else None
                    result.append({'dataset': ds, 'model': model, 'start': start, 'metric': metric,
                                   'runs': 3, 'valid_runs': len(observed), 'missing_runs': missing,
                                   'mean': value, 'status': 'missing_source' if missing else
                                   ('recomputed' if observed else 'undefined_ratio')})
    checks = {'process_runs': len(records),
              'transition_means_available': sum(r['mean'] is not None for r in result if r['metric'] in TRANSITIONS),
              'transition_means_unavailable': sum(r['mean'] is None for r in result if r['metric'] in TRANSITIONS),
              'query_evidence_values_unavailable': sum(records[run][metric] is None and statuses[run, metric].startswith('missing')
                                                      for run in records for metric in EVIDENCE),
              'within_run_evidence_checks': evidence_checks,
              'cross_rq_endpoint_checks': endpoint_checks,
              'error_conservation_and_probe_checks': conservation_checks,
              'reference_value_field': 'before',
              'share_aggregation': 'Mean of within-run fractions; zero-recovery denominators excluded; effective counts retained in audit data.'}
    return result, checks


def main():
    manifest = verify_inputs()
    rows, checks = collect(read_csv('run_metric_diffs.csv'))
    style()
    
    
    fig = plt.figure(figsize=(5.6, 4.4))
    grid = fig.add_gridspec(4, 2, width_ratios=(1.5, 1), left=.195, right=.985,
                           bottom=.09, top=.875, hspace=.08, wspace=.17)
    configurations = [(m, s) for m in MODELS for s in STARTS]
    for i, ds in enumerate(DATASETS):
        ax = fig.add_subplot(grid[i, 0])
        dataset_rows = [r for r in rows if r['dataset'] == ds]
        indexed = {(r['model'], r['start'], r['metric']): r for r in dataset_rows}
        ax.set_yticks(range(4), [f'{MODEL_NAMES[m]} / {s}' for m, s in configurations])
        ax.set_ylim(3.48, -.48)
        ax.tick_params(axis='y', length=0, labelsize=7.7)
        ax.tick_params(axis='x', length=2.5, labelsize=7)
        ax.set_ylabel(ds.upper(), fontweight='bold', labelpad=7)
        transitions = [r['mean'] for r in dataset_rows if r['metric'] in TRANSITIONS and r['mean'] is not None]
        if not transitions:
            ax.set_xlim(-1, 1)
            ax.set_xticks([])
        else:
            removed = [indexed[m, s, f'{e}_removed']['mean'] for m, s in configurations for e in ('fn', 'fp')]
            added = [indexed[m, s, f'{e}_introduced']['mean'] for m, s in configurations for e in ('fn', 'fp')]
            max_removed, max_added = max(v for v in removed if v is not None), max(v for v in added if v is not None)
            span = max(max_removed + max_added, 1)
            pad = .03 * span
            ax.set_xlim(-max_removed - .24 * span, max_added + .24 * span)
            ax.axvline(0, color='#8b939d', linewidth=.8)
        
        
        ax.set_xticks([])
        for y, (model, start) in enumerate(configurations):
            
            for offset, error in [(-.25, 'fn'), (.25, 'fp')]:
                removed = indexed[model, start, f'{error}_removed']['mean']
                added = indexed[model, start, f'{error}_introduced']['mean']
                if removed is None or added is None:
                    ax.text(0, y + offset, '--', ha='center', va='center', color='#5d6570', fontsize=8)
                    continue
                color = ERROR_COLORS[error]
                ax.barh(y + offset, -removed, height=.25, color=color)
                ax.barh(y + offset, added, height=.25, color=color)
                ax.text(-removed - pad, y + offset, f'{removed:.1f}', ha='right', va='center', color=color, fontsize=7.3)
                ax.text(added + pad, y + offset, f'{added:.1f}', ha='left', va='center', color=color, fontsize=7.3)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#b8bec5')
        if i == 0:
            ax.text(0, 1.10, '(a) Error transitions', transform=ax.transAxes, fontweight='bold', fontsize=8)
        if i == len(DATASETS) - 1:
            ax.set_xlabel('Removed $\\leftarrow$ pairs $\\rightarrow$ Introduced', labelpad=3, fontsize=7.7)

        evidence_ax = fig.add_subplot(grid[i, 1])
        evidence_ax.set_ylim(3.48, -.48)
        evidence_ax.set_xlim(0, 112)
        evidence_ax.set_yticks([])
        evidence_ax.set_xticks((0, 50, 100) if i == len(DATASETS) - 1 else ())
        evidence_ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        evidence_ax.tick_params(axis='x', length=2.5, labelsize=7)
        evidence_ax.spines['left'].set_visible(False)
        for y, (model, start) in enumerate(configurations):
            selected = [indexed[model, start, metric] for metric in SHARES]
            valid = {r['valid_runs'] for r in selected}
            assert len(valid) == 1, (ds, model, start, 'share denominators differ')
            n = valid.pop()
            if all(r['mean'] is not None for r in selected):
                assert isclose(sum(r['mean'] for r in selected), 1, abs_tol=1e-12)
                left = 0
                for metric, row in zip(SHARES, selected):
                    width = 100 * row['mean']
                    evidence_ax.barh(y, width, left=left, height=.62, color=SHARE_COLORS[metric])
                    if width >= 13:
                        evidence_ax.text(left + width / 2, y, f'{width:.1f}', ha='center', va='center',
                                         color='white' if metric == 'direct_share' else '#202020', fontsize=7.3)
                    left += width
            else:
                label = '--' if any(r['missing_runs'] for r in selected) else 'NA'
                evidence_ax.text(50, y, label, ha='center', va='center', fontsize=9, color='#5d6570')
        if i == 0:
            evidence_ax.text(0, 1.10, '(b) Recovery evidence', transform=evidence_ax.transAxes, fontweight='bold', fontsize=8)
        if i == len(DATASETS) - 1:
            evidence_ax.set_xlabel('Mean within-run share', labelpad=3, fontsize=7.7)
    handles = [Patch(facecolor=ERROR_COLORS[e], label=e.upper()) for e in ('fn', 'fp')]
    handles += [Patch(facecolor=SHARE_COLORS[m], label='Indirect-only' if m == 'indirect_share' else m.removesuffix('_share').capitalize()) for m in SHARES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, .995), ncol=5,
               frameon=False, handlelength=1.2, columnspacing=1.0, fontsize=7.7)
    write_rows('F4_process_changes', rows)
    checks['reference_id'] = manifest['reference_id']
    (PAPER / 'results/plot_data/F4_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    save(fig, 'F4_process_changes')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
