import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'paper_artifacts/data'
DATASETS = ('jayme', 'ielts', 'hujiang', 'huawei')
MODELS = ('qwen3.5-flash', 'deepseek-v4-flash')
STARTS = ('LLM', 'HC')
CONTROLS = ('AQCC_CONST', 'AQCC_TEXT', 'AQCC_RANDOM', 'FROZEN_LR', 'FROZEN_PLAN')
METRICS = ('ari', 'b3_f1')


def verify(saved=None, data_dir=DATA):
    data_dir = Path(data_dir)
    if saved is None:
        saved = json.loads((ROOT / 'paper_artifacts/statistics/rq2_conditional_intervals_20261003.json').read_text(encoding='utf-8'))
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['reference_id'] == manifest['reference_id']
    for name in saved['source_files']:
        assert (data_dir / name).is_file(), name
    with (data_dir / 'run_metric_diffs.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['rq'] == 'RQ2' and r['stage'] == 'P_B']
    index = {}
    for row in rows:
        key = tuple(row[k] for k in ('family', 'dataset', 'model', 'start', 'repeat', 'control', 'metric'))
        assert key not in index
        index[key] = float(row['before']) if row['before'] else None
    differences = np.full((4, 2, 3, 2, 5, 2), np.nan)
    paired_q = paired_scores = 0
    for di, ds in enumerate(DATASETS):
        for mi, model in enumerate(MODELS):
            for si, start in enumerate(STARTS):
                for ci, control in enumerate(CONTROLS):
                    for ri in range(3):
                        stem = (ds, model, start, str(ri + 1))
                        assert index[('score',) + stem + ('', 'actual_q')] == index[('control_score',) + stem + (control, 'actual_q')]
                        paired_q += 1
                        for qi, metric in enumerate(METRICS):
                            d = index[('score',) + stem + ('', metric)] - index[('control_score',) + stem + (control, metric)]
                            assert math.isclose(d, index[('paired_gain',) + stem + (control, 'delta_' + metric)], abs_tol=1e-12)
                            differences[di, mi, ri, si, ci, qi] = d
                            paired_scores += 1
    assert np.isfinite(differences).all()
    block_means = differences.mean(axis=3).reshape(8, 3, 10)
    rng = np.random.Generator(np.random.PCG64(20261003))
    draws = np.empty((200000, 10))
    for offset in range(0, len(draws), 10000):
        count = min(10000, len(draws) - offset)
        indexes = rng.integers(0, 3, size=(count, 8, 3))
        draws[offset:offset + count] = block_means[np.arange(8)[None, :, None], indexes, :].mean(axis=(1, 2))
    low, high = np.quantile(draws, [0.025, 0.975], axis=0, method='linear')
    observed = differences.mean(axis=(0, 1, 2, 3))
    record_index = {(r['control'], r['metric']): r for r in saved['records']}
    output = []
    for ci, control in enumerate(CONTROLS):
        for qi, metric in enumerate(METRICS):
            r = record_index[control, 'delta_' + metric]
            calculated = dict(mean=float(observed[ci, qi]), ci_low=float(low[ci * 2 + qi]), ci_high=float(high[ci * 2 + qi]))
            for field, value in calculated.items():
                assert math.isclose(value, r[field], rel_tol=1e-11, abs_tol=1e-12), (control, metric, field)
            output.append(dict(control=control, metric=metric, **calculated))
    report = {'status': 'PASS', 'actual_query_matches': paired_q, 'paired_score_differences': paired_scores,
              'intervals': output, 'numpy_version': np.__version__, 'method': saved['method']}
    return report


def main():
    report = verify()
    path = ROOT / 'generated/ci_validation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: 240 query matches, 480 score differences, and all 10 conditional intervals.')
    return report


if __name__ == '__main__':
    main()
