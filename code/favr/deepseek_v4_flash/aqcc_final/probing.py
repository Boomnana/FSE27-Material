from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .evidence import EvidenceLedger, pair


@dataclass(frozen=True)
class ProbePair:
    pair: tuple[int, int]
    block_min: int
    pivot: int
    recursion_depth: int


class RecursiveIntraBlockProber:
    """Real-LLM recursive probing without hard certification.

    For each P1 block, query a medoid pivot against the current residual. SAME
    answers stay with the pivot for this probing stage; DIFFERENT/failed targets
    form the residual recursively. The only hard facts remain the direct ledger
    observations. State is reconstructed from the ledger, so resume is exact.
    """

    def __init__(self, initial_partition, dense_matrix: np.ndarray):
        self.initial_partition = tuple(tuple(block) for block in initial_partition)
        self.dense = dense_matrix
        self._complete_certificate = None

    def _geometry_signature(self):
        
        
        return (self.initial_partition, self.dense.shape, self.dense.dtype.str,
                self.dense.tobytes(order="C"))

    def _completion_still_valid(self, ledger):
        certificate = self._complete_certificate
        if certificate is None:
            return False
        geometry, dependencies = certificate
        if geometry != self._geometry_signature():
            self._complete_certificate = None
            return False
        for (a, b), expected in dependencies.items():
            obs = ledger.get(a, b)
            if obs is None or (obs.status, obs.label) != expected:
                self._complete_certificate = None
                return False
        return True

    def _pivot(self, subset: tuple[int, ...]) -> int:
        members = list(subset)
        sub = self.dense[np.ix_(members, members)]
        means = sub.mean(axis=1)
        best = float(means.max())
        candidates = [members[k] for k, v in enumerate(means) if abs(float(v) - best) <= 1e-7]
        return min(candidates)

    def _frontier(self, block: tuple[int, ...], ledger: EvidenceLedger,
                  dependencies: dict | None = None) -> tuple[list[ProbePair], int]:
        subset = tuple(sorted(block))
        depth = 0
        while len(subset) > 1:
            pivot = self._pivot(subset)
            missing: list[ProbePair] = []
            residual: list[int] = []
            for target in subset:
                if target == pivot:
                    continue
                obs = ledger.get(pivot, target)
                if dependencies is not None and obs is not None:
                    dependencies[pair(pivot, target)] = (obs.status, obs.label)
                if obs is None:
                    missing.append(ProbePair(pair(pivot, target), min(block), pivot, depth))
                elif obs.status == "SUCCESS" and obs.label == "SAME":
                    pass
                else:
                    
                    
                    residual.append(target)
            if missing:
                return missing, depth
            subset = tuple(sorted(residual))
            depth += 1
        return [], depth

    def next_pairs(self, ledger: EvidenceLedger, limit: int) -> list[ProbePair]:
        if limit <= 0:
            return []
        if self._completion_still_valid(ledger):
            return []
        frontiers = []
        dependencies = {}
        for block in self.initial_partition:
            if len(block) <= 1:
                continue
            rows, _ = self._frontier(block, ledger, dependencies)
            if rows:
                frontiers.append(rows)
        if not frontiers:
            
            
            
            self._complete_certificate = self._geometry_signature(), dependencies
            return []
        
        
        out: list[ProbePair] = []
        col = 0
        while len(out) < limit:
            added = False
            for rows in frontiers:
                if col < len(rows):
                    out.append(rows[col])
                    added = True
                    if len(out) >= limit:
                        break
            if not added:
                break
            col += 1
        return out

    def is_complete(self, ledger: EvidenceLedger) -> bool:
        return not self.next_pairs(ledger, 1)
