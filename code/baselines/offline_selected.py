from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import baseline_lib as lib


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def canonical_groups(partition):
    return sorted(tuple(sorted(str(report_id) for report_id in group)) for group in partition)


def main():
    parser = argparse.ArgumentParser(description='Rebuild the eight selected pipeline partitions offline.')
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.package.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = read(root/'configs/baseline_selected_settings.json')
    upstream = read(root/'configs/baseline_upstream_partitions.json')
    endpoints = {(row['method'], row['dataset']): row for row in
                 (json.loads(line) for line in (root/'runs/baselines/endpoints.jsonl').read_text('utf-8').splitlines())
                 if row['method'] in ('LLMCluster', 'TERFUR')}
    with gzip.open(root/'data/partitions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        partitions = {row['partition_id']: row for row in (json.loads(line) for line in stream)}
    stopwords = lib.load_stopwords(Path(__file__).with_name('stopwords_zh.txt'))
    reports = {}
    matrices = {}
    checks = []
    for entry in settings:
        dataset = entry['dataset']
        if dataset not in reports:
            start = read(root/f'data/initializations/{dataset}/I_llm/start_state.json')
            reports[dataset] = start['reports']
        records = reports[dataset]
        ids = [str(r['source_id']) for r in records]
        selected = entry['selected']
        if entry['method'] == 'LLMCluster':
            match = re.fullmatch(r'blend\((coassoc_[^,]+),(tfidf_[^,]+),llm_weight=([.0-9]+)\)', selected['representation'])
            if not match:
                raise ValueError('Unknown selected representation')
            subset, text, weight = match.groups()
            chosen = [r['partition'] for r in upstream if r['dataset'] == dataset and
                (subset == 'coassoc_all6' or r['model'] ==
                    {'coassoc_qwen35':'qwen3.5-flash','coassoc_deepseek_v4':'deepseek-v4-flash'}[subset])]
            similarity = float(weight)*lib.coassociation(chosen, ids)+(1-float(weight))*lib.tfidf_similarities(records)[text]
            partition = lib.reconstruct_partition(selected, similarity, ids)
        else:
            if dataset not in matrices:
                matrices[dataset] = lib.feature_matrices(records, lib, stopwords)
            similarity = matrices[dataset][selected['representation']]
            seed = lib.reconstruct_partition({'linkage':selected['seed_linkage'],'cut':selected['seed_cut']}, similarity, ids)
            partition, native = lib.fuzzy_harden(seed, similarity, ids, float(selected['secondary_membership_ratio']))
        content = (json.dumps(partition, ensure_ascii=False, indent=2)+'\n').replace('\n','\r\n').encode()
        endpoint = endpoints[(entry['method'], dataset)]
        if entry['expected_archived_partition_id'] != endpoint['archived_partition_id']:
            raise ValueError('Selected setting and endpoint identify different archived partitions')
        expected = partitions[endpoint['partition_id']]
        match = expected['dataset'] == dataset and canonical_groups(partition) == canonical_groups(expected['groups'])
        reference = read(root/f'data/reference/{dataset}.json')['issue_by_source_id']
        metrics = lib.score(partition, {str(k):str(v) for k,v in reference.items()})
        differences = {k:metrics[k]-selected[k] for k in metrics if k in selected and abs(metrics[k]-selected[k])>1e-12}
        row = {'method':entry['method'],'dataset':dataset,'partition_id':endpoint['partition_id'],
               'partition_membership_identical':match,'metric_differences':differences}
        checks.append(row)
        (args.output/f"{entry['method']}_{dataset}.json").write_bytes(content)
        print(json.dumps(row), flush=True)
    (args.output/'verification.json').write_text(json.dumps(checks, indent=2)+'\n','utf-8')
    if not all(r['partition_membership_identical'] and not r['metric_differences'] for r in checks):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
