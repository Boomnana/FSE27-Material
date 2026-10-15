from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('jayme', 'ielts', 'hujiang', 'huawei')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose2(n):
    return n * (n - 1) // 2


def score(groups, labels):
    members = [rid for group in groups for rid in group]
    assert groups and all(groups), 'Empty group in partition'
    assert len(members) == len(set(members)) == len(labels), 'Partition coverage'
    assert set(members) == set(labels), 'Unknown or missing report ID'
    sizes = Counter(labels.values())
    cells = [Counter(labels[rid] for rid in group) for group in groups]
    tp = sum(choose2(count) for cell in cells for count in cell.values())
    predicted = sum(choose2(len(group)) for group in groups)
    actual = sum(choose2(size) for size in sizes.values())
    n = len(labels)
    total = choose2(n)
    chance = actual * predicted / total if total else 0.0
    denom = (actual + predicted) / 2 - chance
    precision = sum(count * count / len(groups[i]) for i, cell in enumerate(cells) for count in cell.values()) / n
    recall = sum(count * count / sizes[key] for cell in cells for key, count in cell.items()) / n
    pair_precision = tp / predicted if predicted else 1.0
    pair_recall = tp / actual if actual else 1.0
    return dict(ari=(tp - chance) / denom if denom else 1.0,
                b3_precision=precision, b3_recall=recall, b3_f1=2 * precision * recall / (precision + recall),
                pair_tp=tp, pair_fp=predicted - tp, pair_fn=actual - tp, pair_tn=total - predicted - actual + tp,
                pair_precision=pair_precision, pair_recall=pair_recall,
                pair_f1=2 * pair_precision * pair_recall / (pair_precision + pair_recall) if pair_precision + pair_recall else 0.0,
                n=n, gold_groups=len(sizes), predicted_groups=len(groups))


def same(actual, expected, context):
    assert expected is not None and math.isfinite(float(expected)), context
    assert math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-10), (context, actual, expected)


def main():
    inputs = []
    labels = {}
    orders = load(ROOT / 'data/report_order.json')
    for ds in DATASETS:
        path = ROOT / f'data/reference/{ds}.json'
        labels[ds] = load(path)['issue_by_source_id']
        inputs.append(path)
        reports = load(ROOT / f'data/reports/{ds}.json')
        ids = [str(row['id']) for row in reports]
        assert len(ids) == len(set(ids))
        assert ids == orders[ds] and set(ids) == set(labels[ds]), ds
    assert sum(map(len, labels.values())) == 1567
    registry_path = ROOT / 'data/partitions.jsonl.gz'
    inputs.append(registry_path)
    registry, computed = {}, {}
    partition_contents = set()
    with gzip.open(registry_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            pid = row['partition_id']
            assert pid not in registry, pid
            content = (row['dataset'], tuple(tuple(group) for group in row['groups']))
            assert content not in partition_contents, ('Duplicate partition', pid)
            partition_contents.add(content)
            registry[pid] = row
            computed[pid] = score(row['groups'], labels[row['dataset']])
    definitions = [('main', 'runs/main/checkpoints.jsonl', 672), ('controls', 'runs/controls/endpoints.jsonl', 240), ('baselines', 'runs/baselines/endpoints.jsonl', 24)]
    counts = {}
    metric_checks = 0
    groups = {}
    for family, filename, count in definitions:
        path = ROOT / filename
        inputs.append(path)
        records = read_rows(path)
        assert len(records) == count, (family, len(records), count)
        groups[family] = records
        counts[family] = len(records)
        for row in records:
            pid = row['partition_id']
            assert registry[pid]['dataset'] == row['dataset'], pid
            for metric, expected in row['expected_metrics'].items():
                if metric == 'actual_q':
                    same(row['actual_q'], expected, (row['run_id'], metric))
                else:
                    assert metric in computed[pid], ('Unknown expected metric', metric)
                    same(computed[pid][metric], expected, (row['run_id'], metric))
                metric_checks += 1
    assert len({r['run_id'] for r in groups['main']}) == 48
    main5 = {r['run_id']: r for r in groups['main'] if r['stage'] == '0.05'}
    assert len(main5) == 48
    for row in groups['controls']:
        assert row['actual_q'] == main5[row['matched_main_run']]['actual_q'], row['run_id']
    snapshots = ROOT / 'paper_artifacts/data'
    score_rows = 0
    a3s_ids = {r['run_id']: r['partition_id'] for r in groups['baselines'] if r['method'] == 'A3S'}
    with (snapshots / 'run_metric_diffs.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['family'] not in ('score', 'control_score', 'a3s_score') or row['metric'] not in next(iter(computed.values())):
                continue
            pid = row['archived_partition_id']
            if row['family'] == 'a3s_score':
                pid = a3s_ids[row['run']]
            assert pid in computed, ('Derived score partition absent', pid)
            assert registry[pid]['dataset'] == row['dataset']
            same(computed[pid][row['metric']], row['before'], (row['run'], row['stage'], row['metric']))
            score_rows += 1
    baseline_ids = {r.get('archived_partition_id', r['partition_id']): r['partition_id'] for r in groups['baselines']}
    external = load(snapshots / 'external_baselines.json')
    for row in external['records']:
        for metric, value in computed[baseline_ids[row['archived_partition_id']]].items():
            if metric in row:
                same(value, row[metric], ('External table snapshot', row['run'], metric))
    with (snapshots / 'label_statistics.csv').open(encoding='utf-8-sig', newline='') as stream:
        label_stats = list(csv.DictReader(stream))
    for row in label_stats:
        if row.get('dataset') not in labels:
            continue
        sizes = Counter(labels[row['dataset']].values())
        expected = dict(n=sum(sizes.values()), gold_groups=len(sizes), same_pairs=sum(choose2(n) for n in sizes.values()), singletons=sum(n == 1 for n in sizes.values()))
        for metric, value in expected.items():
            same(value, row['before_' + metric], ('Table 1', row['dataset'], metric))
    report = {'status': 'PASS', 'reports': 1567, 'unique_partitions_scored': len(computed),
              'records': counts, 'record_metric_comparisons': metric_checks,
              'derived_score_rows_checked': score_rows, 'matched_control_queries': 240,
              'external_records': len(external['records']),
              'inputs': [p.relative_to(ROOT).as_posix() for p in inputs],
              'scope': 'Fixed partition metrics and recorded score consistency. This does not independently validate semantic reference labels or rerun LLM experiments.'}
    dest = ROOT / 'generated/score_validation.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(computed)} partitions; {metric_checks} endpoint/checkpoint metric comparisons; {score_rows} derived score rows.')
    return report


if __name__ == '__main__':
    main()
