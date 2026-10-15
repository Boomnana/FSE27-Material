from pathlib import Path
from collections import Counter
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def readj(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def checked_path(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), relative
    return path


manifest = readj(DATA / 'manifest.json')
counts = Counter()
listed = set()
for item in manifest['files']:
    path = checked_path(item['path'])
    assert item['path'] not in listed
    listed.add(item['path'])
    assert path.stat().st_size == item['bytes'], item['path']
    counts['listed_files'] += 1

order = readj(DATA / 'report_order.json')
reference = readj(DATA / 'reference' / 'manifest.json')
assert manifest['reference_id'] == reference['reference_id']
reports = {}
truth = {}
for dataset, ids in order.items():
    rr = readj(DATA / 'reports' / (dataset + '.json'))
    reports[dataset] = {r['id']: r['text'] for r in rr}
    truth[dataset] = readj(DATA / 'reference' / (dataset + '.json'))['issue_by_source_id']
    assert len(rr) == len(reports[dataset]) == len(ids) == len(set(ids))
    assert set(ids) == set(reports[dataset]) == set(truth[dataset])
    assert reference['datasets'][dataset]['reports'] == len(ids)
    assert reference['datasets'][dataset]['groups'] == len(set(truth[dataset].values()))
    counts['reports'] += len(ids)
assert counts['reports'] == 1567

partitions = {}
partition_contents = set()
for row in rows(DATA / 'partitions.jsonl.gz'):
    pid = row['partition_id']
    dataset = row['dataset']
    groups = row['groups']
    assert isinstance(pid, str) and pid and pid not in partitions
    content = (dataset, tuple(tuple(group) for group in groups))
    assert content not in partition_contents
    partition_contents.add(content)
    flat = [rid for group in groups for rid in group]
    assert all(group for group in groups)
    assert len(flat) == len(set(flat)) == len(order[dataset]) and set(flat) == set(order[dataset])
    partitions[pid] = row
    counts['partitions'] += 1
assert counts['partitions'] == 6520

frozen_vectors = {}
for item in readj(DATA / 'initializations' / 'manifest.json'):
    path = checked_path(item['path'])
    state = readj(path)
    dataset = item['dataset']
    assert item['source_id'] in manifest['sources']
    assert state['report_ids'] == order[dataset]
    assert [r['source_id'] for r in state['reports']] == order[dataset]
    assert [r['report_index'] for r in state['reports']] == list(range(len(order[dataset])))
    assert all(reports[dataset][r['source_id']] == r['description'] for r in state['reports'])
    vector_path = (path.parent / state['embeddings_npz']['path']).resolve()
    assert vector_path.is_relative_to(ROOT) and vector_path.is_file()
    with np.load(vector_path, allow_pickle=False) as stored:
        vectors = {key: stored[key] for key in stored.files}
    assert vectors['format'].item() == b'csr'
    assert tuple(vectors['shape']) == (len(order[dataset]), state['embedding_dimension'])
    indptr = vectors['indptr']
    indices = vectors['indices']
    values = vectors['data']
    assert len(indptr) == len(order[dataset]) + 1
    assert indptr[0] == 0 and indptr[-1] == len(values) == len(indices)
    assert np.all(indptr[1:] >= indptr[:-1]) and np.all(np.isfinite(values))
    assert np.all(indices >= 0) and np.all(indices < state['embedding_dimension'])
    if dataset in frozen_vectors:
        assert vectors.keys() == frozen_vectors[dataset].keys()
        assert all(np.array_equal(vectors[k], frozen_vectors[dataset][k]) for k in vectors)
    else:
        frozen_vectors[dataset] = vectors
    groups = [g['member_source_ids'] for g in state['partition']['clusters']]
    assert partitions[item['partition_id']]['dataset'] == dataset
    assert partitions[item['partition_id']]['groups'] == groups
    assert all([order[dataset][i] for i in g['members']] == g['member_source_ids']
               for g in state['partition']['clusters'])
    labels = state['partition']['labels_by_index']
    assert len(labels) == len(order[dataset])
    assert len(set(labels)) == len(groups)
    assert all(len({labels[i] for i in g['members']}) == 1 for g in state['partition']['clusters'])
    counts['initializations'] += 1
assert counts['initializations'] == 8

index = readj(ROOT / 'runs/main/index.json')
checkpoints = rows(ROOT / 'runs/main/checkpoints.jsonl')
assert len(index) == 48 and len(checkpoints) == 672
assert len({run['run_id'] for run in index}) == 48
assert set(run['model'] for run in index) == {'qwen3.5-flash', 'deepseek-v4-flash'}
q5 = {}
status_counts = Counter()
label_counts = Counter()
for run in index:
    run_id = run['run_id']
    dataset = run['dataset']
    micro = rows(checked_path(run['micro_rounds']))
    qindex = {r['actual_q']: r for r in micro}
    assert len(qindex) == len(micro) == run['micro_round_records']
    assert [r['actual_q'] for r in micro] == sorted(qindex)
    assert micro[0]['actual_q'] == 0 and micro[-1]['actual_q'] == run['q_10pct']
    for row in micro:
        assert partitions[row['partition_id']]['dataset'] == dataset
        assert row['source_id'] in manifest['sources']
        assert all(row['usage'][k] >= 0 for k in ['prompt_tokens', 'completion_tokens', 'total_tokens'])
        counts['micro_round_records'] += 1
    rr = [r for r in checkpoints if r['run_id'] == run_id]
    assert len(rr) == 14 and sum(r['stage'] == 'Pprobe' for r in rr) == 1
    assert sum(r['budget_fraction'] is not None for r in rr) == 13
    for row in rr:
        raw = qindex[row['actual_q']]
        assert raw['partition_id'] == row['partition_id'] and raw['usage'] == row['usage']
        assert row['dataset'] == dataset and row['model'] == run['model']
        counts['checkpoints'] += 1
    query = rows(checked_path(run['judgments']['path']))
    assert len(query) == run['q_5pct'] == run['judgments']['rows']
    assert next(r['actual_q'] for r in rr if r['budget_fraction'] == .05) == len(query)
    assert Counter(row['status'] for row in query) == run['judgments']['statuses']
    assert run['judgments']['source_id'] in manifest['sources']
    assert run['terminal']['source_id'] in manifest['sources']
    seen_pairs = set()
    for pos, row in enumerate(query, 1):
        assert row['query_index'] == pos
        assert row['left'] in truth[dataset] and row['right'] in truth[dataset]
        assert row['left'] != row['right']
        pair = tuple(sorted([row['left'], row['right']]))
        assert pair not in seen_pairs
        seen_pairs.add(pair)
        assert row['metadata']['report_ids'] == [row['left'], row['right']]
        assert 0 <= row['metadata']['p_same_before'] <= 1
        assert row['status'] in {'SUCCESS', 'FAILED_PROTOCOL', 'FAILED_TRANSPORT_UNCERTAIN'}
        if row['status'] == 'SUCCESS':
            assert row['label'] in ['SAME', 'DIFFERENT']
            label_counts[row['label']] += 1
        else:
            assert row['label'] is None
        status_counts[row['status']] += 1
        counts['judgments'] += 1
    q5[run_id] = len(query)
    counts['main_runs'] += 1

controls = rows(ROOT / 'runs/controls/endpoints.jsonl')
assert len(controls) == 240
assert Counter(r['control'] for r in controls) == {'CONST': 48, 'TEXT': 48, 'RANDOM': 48, 'FROZEN_LR': 48, 'FROZEN_PLAN': 48}
for row in controls:
    assert row['actual_q'] == q5[row['matched_main_run']]
    assert partitions[row['partition_id']]['dataset'] == row['dataset']
    assert row['source_id'] in manifest['sources']
    counts['matched_controls'] += 1
baseline = rows(ROOT / 'runs/baselines/endpoints.jsonl')
assert Counter(r['method'] for r in baseline) == {'A3S': 16, 'LLMCluster': 4, 'TERFUR': 4}
for row in baseline:
    assert partitions[row['partition_id']]['dataset'] == row['dataset']
    assert row['source_id'] in manifest['sources']
    if row['method'] == 'A3S':
        assert row['actual_q'] <= row['budget']
    else:
        assert row['tuning'] and row['parameters']
    counts['baselines'] += 1
assert counts['judgments'] == 228285
assert counts['micro_round_records'] == 14666
assert status_counts == {'SUCCESS': 227312, 'FAILED_PROTOCOL': 945, 'FAILED_TRANSPORT_UNCERTAIN': 28}
assert label_counts == {'SAME': 53542, 'DIFFERENT': 173770}
print(json.dumps({'status': 'PASS', 'checks': dict(counts), 'judgment_statuses': dict(status_counts)}, indent=2))
