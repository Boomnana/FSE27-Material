from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Inspect a registered FAVR/control run; explicit --online is required for model requests.')
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--job', required=True)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    jobs = json.loads((root/'configs/favr_runs.json').read_text('utf-8'))['jobs']
    job = next(j for j in jobs if j['job_id']==args.job)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    sys.path.insert(0,str(Path(__file__).resolve().parent/job['runtime_flavor']))
    from aqcc_final.data import load_start_state
    dataset = load_start_state(root/job['start_state'])
    assert dataset.n == job['n_reports']
    result = {'job_id':job['job_id'],'model':job['model'],'reports':dataset.n,
              'seed':job['seed'],'budget_pairs':job['budget_pairs'],
              'start_state':job['start_state'],
              'historical_config_status':job['historical_config_status'],
              'provider_calls':0}
    if not args.online:
        print(json.dumps(result,indent=2))
        return
    if args.output is None or args.cache is None:
        parser.error('--online requires an explicit --output and shared --cache directory')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    key = os.environ.get('SILRA_API_KEY','').strip()
    if not key:
        raise ValueError('SILRA_API_KEY must be supplied through the environment')
    from aqcc_final.provider import SilraPairJudge
    from aqcc_final.shared_oracle import SharedFirstResponseJudge
    from aqcc_final.runner import run
    prompt = (root/'protocols/pair_prompt.txt').read_text('utf-8')
    base = SilraPairJudge(key,prompt,model=job['model'],record_raw_responses=False)
    judge = SharedFirstResponseJudge(base,cache_dir=args.cache,block_id=job['block_id'])
    run(dataset,judge,output_dir=args.output,system_prompt=prompt,
        strategy=job['strategy'],budget_pairs=job['budget_pairs'],microbatch_size=16,concurrency=16,
        decoder_max_steps=1000,detailed_trace=True,decoder_graph_mode='neutral',
        acquisition_mode=job['acquisition_mode'],recursive_probing=True,
        random_with_probing=job['random_with_probing'],random_seed=job['seed'],
        surrogate_feature_policy='all',surrogate_scaler_policy='refit',
        query_feedback='frozen_plan' if job['arm']=='FROZEN_PLAN' else 'adaptive',
        surrogate_update='freeze_first_fitted' if job['arm']=='FROZEN_LR' else 'adaptive',
        feature_cache=None,resume=False,execution_mode='PORTABLE_RELEASE_NEW_RUN')


if __name__=='__main__':
    main()
