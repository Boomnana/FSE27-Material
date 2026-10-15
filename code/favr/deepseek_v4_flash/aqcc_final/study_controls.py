"""Persisted, answer-invariant planning and frozen-training controls for this study."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evidence import EvidenceLedger
from .provider import canonical_sha256, strict_json
from .surrogate import LLMSurrogate, SurrogateSummary
from .tracing import surrogate_state


class Anchor:
    def __init__(self, directory, name, config, ledger, writer):
        self.directory=Path(directory)
        self.path=self.directory/name
        self.config_hash=canonical_sha256(config)
        self.ledger=ledger
        self.writer=writer
        self.document=None
        manifest=strict_json((self.directory/'manifest.json').read_text(encoding='utf-8'))
        ref=manifest.get('study_control_anchors',{}).get(name)
        if ref is not None and not self.path.is_file():
            raise ValueError('Missing committed control anchor')
        if self.path.is_file():
            doc=strict_json(self.path.read_text(encoding='utf-8'))
            if doc['config_sha256']!=self.config_hash or canonical_sha256(doc['state'])!=doc['state_sha256']:
                raise ValueError('Control anchor identity mismatch')
            q=doc['query_count']
            history=ledger.history()
            if not isinstance(q,int) or not 0<=q<=len(history):
                raise ValueError('Control anchor beyond available evidence')
            prefix=EvidenceLedger.from_rows(ledger.n,history[:q])
            if prefix.snapshot_sha256()!=doc['evidence_sha256']:
                raise ValueError('Control anchor evidence differs from run prefix')
            actual=canonical_sha256(doc)
            if ref is not None and ref!=actual:
                raise ValueError('Control anchor hash differs from manifest')
            self.document=doc
            if ref is None:
                self._commit_reference(actual)

    def _commit_reference(self,value):
        p=self.directory/'manifest.json'
        manifest=strict_json(p.read_text(encoding='utf-8'))
        manifest.setdefault('study_control_anchors',{})[self.path.name]=value
        self.writer(p,manifest)

    def save(self,state):
        if self.document is not None:
            raise ValueError('Control anchor is immutable')
        doc={'schema':'aqcc-study-control-anchor-v1','config_sha256':self.config_hash,
             'query_count':len(self.ledger.history()),'evidence_sha256':self.ledger.snapshot_sha256(),
             'state':state,'state_sha256':canonical_sha256(state)}
        self.writer(self.path,doc)
        self._commit_reference(canonical_sha256(doc))
        self.document=doc


class FrozenPlanSelector:
    """Freeze every answer-dependent selector input at the first active call.

    Future batches can be evaluated lazily: only consumed-mask and fixed query-count
    schedules advance, so no later answer or decoded partition can affect the plan.
    """
    def __init__(self,selector,directory,config,ledger,writer):
        self.selector=selector
        self.anchor=Anchor(directory,'frozen_plan_anchor.json',config,ledger,writer)
        self.last_trace=None
        self._state=None
        if self.anchor.document is not None:
            self._restore(self.anchor.document['state'])
        elif any(row.get('metadata',{}).get('stage')=='ACTIVE_QUERY' for row in ledger.history()):
            raise ValueError('Cannot reconstruct missing frozen plan from future answers')

    def _restore(self,state):
        self._state={'current_partition':tuple(tuple(b) for b in state['partition']),
                     'probabilities':np.asarray(state['probabilities'],dtype=np.float32),
                     'direct_edges':{(a,b):s for a,b,s in state['direct_edges']}}
        store=self.selector.store
        upper=self._state['probabilities']
        if upper.shape!=(store.total_pairs,) or not np.isfinite(upper).all():
            raise ValueError('Invalid frozen probability vector')
        matrix=np.full((store.n,store.n),.5,dtype=np.float32)
        matrix[store.i,store.j]=upper
        matrix[store.j,store.i]=upper
        self._state['probability_matrix']=matrix

    def choose(self,**kwargs):
        if self._state is None:
            state={'partition':[list(b) for b in kwargs['current_partition']],
                   'probabilities':kwargs['probabilities'].tolist(),
                   'direct_edges':[[int(a),int(b),int(s)] for (a,b),s in sorted(kwargs['direct_edges'].items())],
                   'selector_policy':'ANSWER_INDEPENDENT_AFTER_PROBING_LAZY_FIXED_PLAN',
                   'anchor_queries_done':kwargs['queries_done']}
            self.anchor.save(state)
            self._restore(state)
        frozen=dict(kwargs,**self._state)
        result=self.selector.choose(**frozen)
        self.last_trace=self.selector.last_trace
        if self.last_trace is not None:
            self.last_trace.update(study_query_feedback='FROZEN_PLAN',
                                   frozen_anchor_query_count=self.anchor.document['query_count'],
                                   selection_state_semantics='FROZEN_AT_ANCHOR_NOT_CURRENT_PARTITION')
        return result


class FrozenTrainingSurrogate(LLMSurrogate):
    """Freeze a successfully fitted LR and scaler; keep dynamic features live."""
    def __init__(self,*args,directory,config,ledger,writer,**kwargs):
        super().__init__(*args,**kwargs)
        self.anchor=Anchor(directory,'frozen_lr_anchor.json',config,ledger,writer)
        if self.anchor.document is not None:
            self._restore_model(self.anchor.document['state'])
        else:
            rounds=Path(directory)/'micro_rounds.jsonl'
            if rounds.is_file():
                for raw in rounds.read_bytes().splitlines(keepends=True):
                    if not raw.endswith(b'\n'): break
                    row=strict_json(raw.decode('utf-8'))
                    if row.get('surrogate',{}).get('fitted'):
                        raise ValueError('Fitted frozen-LR checkpoint is missing its anchor')

    @staticmethod
    def _array(value):
        arr=np.asarray(value['values'],dtype=value['dtype'])
        if not np.isfinite(arr).all(): raise ValueError('Invalid frozen model array')
        return arr

    def _restore_model(self,state):
        if state['feature_policy']!=self.feature_policy or not state['summary']['fitted']:
            raise ValueError('Frozen LR anchor has incompatible feature/fitted state')
        model=state['model']; ss=model['scaler']; ll=model['logistic_regression']
        scaler=StandardScaler(with_mean=ss['with_mean'],with_std=ss['with_std'])
        scaler.mean_=self._array(ss['mean']); scaler.var_=self._array(ss['variance'])
        scaler.scale_=self._array(ss['scale']); scaler.n_features_in_=ss['n_features_in']
        scaler.n_samples_seen_=self._array(ss['n_samples_seen'])[()]
        lr=LogisticRegression(C=self.C,max_iter=ll['max_iter'],solver=ll['solver'],random_state=ll['random_state'])
        lr.coef_=self._array(ll['coefficients']); lr.intercept_=self._array(ll['intercept'])
        lr.classes_=self._array(ll['classes']); lr.n_iter_=self._array(ll['n_iter'])
        lr.n_features_in_=ll['n_features_in']
        self.model=Pipeline([('scale',scaler),('lr',lr)])
        summary=dict(state['summary'])
        summary['feature_names']=tuple(summary['feature_names'])
        summary['coefficients']=tuple(summary['coefficients'])
        self.summary=SurrogateSummary(**summary)

    def fit(self,signed_edges):
        if self.anchor.document is not None:
            return self.summary
        result=super().fit(signed_edges)
        if result.fitted:
            self.anchor.save(surrogate_state(self))
        return result
