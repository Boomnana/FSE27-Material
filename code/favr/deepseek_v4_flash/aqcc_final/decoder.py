from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable
import numpy as np

from .data import canonical_partition


@dataclass(frozen=True)
class DecodeResult:
    partition: tuple[tuple[int, ...], ...]
    loss: float
    steps: int
    operations: tuple[dict, ...]


def labels_of(partition: Iterable[Iterable[int]], n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int32)
    for ci, block in enumerate(partition):
        for u in block:
            if labels[u] != -1:
                raise ValueError("duplicate report in partition")
            labels[u] = ci
    if (labels < 0).any():
        raise ValueError("partition does not cover population")
    return labels


def weighted_loss(partition, probability_matrix: np.ndarray) -> float:
    n = probability_matrix.shape[0]
    labels = labels_of(partition, n)
    ii, jj = np.triu_indices(n, 1)
    same = labels[ii] == labels[jj]
    
    
    p = np.asarray(probability_matrix[ii, jj], dtype=np.float64)
    return float(np.where(same, 1.0 - p, p).sum(dtype=np.float64))


def partition_pair_distance(a, b, n: int) -> int:
    la = labels_of(a, n); lb = labels_of(b, n)
    ii, jj = np.triu_indices(n, 1)
    return int(np.count_nonzero((la[ii] == la[jj]) != (lb[ii] == lb[jj])))


def _direct_same_components(nodes: set[int], direct_edges: dict[tuple[int, int], int]) -> list[set[int]]:
    if len(nodes) <= 2:
        return []
    adj = {u: [] for u in nodes}
    for (a, b), sign in direct_edges.items():
        if sign == 1 and a in nodes and b in nodes:
            adj[a].append(b); adj[b].append(a)
    remaining = set(nodes); out = []
    while remaining:
        start = min(remaining); remaining.remove(start)
        comp = {start}; q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in remaining:
                    remaining.remove(v); comp.add(v); q.append(v)
        if 1 < len(comp) < len(nodes):
            out.append(comp)
    return out


def _membership_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    H = np.zeros((len(labels), k), dtype=np.float64)
    H[np.arange(len(labels)), labels] = 1.0
    return H


class _NeutralGraph:
    """Exact sparse arithmetic for graphs whose off-diagonal p is 0, .5, or 1.

    Affinities and all cluster sums are integers, so bincount aggregation has
    exactly the dense float64 products' values, independent of reduction order.
    The complete objective still includes every .5 edge through its constant
    term; no edge, candidate, or report is discarded.
    """

    def __init__(self, affinity, direct_edges):
        self.n = len(affinity)
        self.left, self.right = np.nonzero(affinity)
        self.weights = affinity[self.left, self.right]
        upper = self.left < self.right
        self.upper_left = self.left[upper]
        self.upper_right = self.right[upper]
        self.upper_weights = self.weights[upper]
        self.constant_loss = (self.n * (self.n - 1) / 4.0
                              + .5 * float(self.upper_weights.sum(dtype=np.float64)))
        self.same_adjacency = [[] for _ in range(self.n)]
        
        
        self._same_components_cache = {}
        for (a, b), sign in direct_edges.items():
            if sign == 1 and 0 <= a < self.n and 0 <= b < self.n:
                self.same_adjacency[a].append(b)
                self.same_adjacency[b].append(a)

    def aggregates(self, labels, k):
        node_cluster = np.bincount(self.left * k + labels[self.right], weights=self.weights,
                                   minlength=self.n * k).astype(np.float64, copy=False).reshape(self.n, k)
        cluster_cross = np.bincount(labels[self.left] * k + labels[self.right], weights=self.weights,
                                    minlength=k * k).astype(np.float64, copy=False).reshape(k, k)
        return node_cluster, cluster_cross

    def loss(self, partition):
        labels = labels_of(partition, self.n)
        within = labels[self.upper_left] == labels[self.upper_right]
        return self.constant_loss - float(self.upper_weights[within].sum(dtype=np.float64))

    def same_components(self, nodes):
        if len(nodes) <= 2:
            return []
        key = frozenset(nodes)
        if key in self._same_components_cache:
            return self._same_components_cache[key]
        remaining = set(nodes)
        result = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            component = {start}
            queue = deque([start])
            while queue:
                for neighbor in self.same_adjacency[queue.popleft()]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            if 1 < len(component) < len(nodes):
                result.append(component)
        
        
        result = tuple(frozenset(component) for component in result)
        self._same_components_cache[key] = result
        return result


def decode(initial_partition, probability_matrix: np.ndarray, *,
           direct_edges: dict[tuple[int, int], int] | None = None,
           max_steps: int = 200, tolerance: float = 1e-8) -> DecodeResult:
    """Deterministic weighted correlation-clustering local decoder.

    Uses vectorized cluster/node affinity sums so D00/D02 can be decoded at each
    micro-round. Every call restarts from the same P1. The optimizer is local;
    no global optimum is claimed.
    """
    p = np.asarray(probability_matrix, dtype=np.float64).copy()
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("probability matrix must be square")
    n = p.shape[0]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    if not np.allclose(p, p.T, atol=1e-7):
        raise ValueError("probability matrix must be symmetric")
    np.fill_diagonal(p, 0.5)
    affinity = 2.0 * p - 1.0
    np.fill_diagonal(affinity, 0.0)
    direct_edges = direct_edges or {}
    
    
    neutral = (_NeutralGraph(affinity, direct_edges)
               if np.isin(p, (0.0, 0.5, 1.0)).all() else None)

    partition = canonical_partition(initial_partition)
    current = neutral.loss(partition) if neutral is not None else weighted_loss(partition, p)
    operations: list[dict] = []
    rejected_operations: set[tuple] = set()
    triangular_k = None
    triangular_indices = None

    while len(operations) < max_steps:
        labels = labels_of(partition, n)
        blocks = [set(block) for block in partition]
        k = len(blocks)
        if neutral is not None:
            node_cluster, cluster_cross = neutral.aggregates(labels, k)
        else:
            H = _membership_matrix(labels, k)
            
            node_cluster = affinity @ H
            
            cluster_cross = H.T @ node_cluster
        cluster_mins = [min(b) for b in blocks]
        best = None  

        
        if k > 1:
            if triangular_k != k:
                triangular_indices = np.triu_indices(k, 1)
                triangular_k = k
            tri_a, tri_b = triangular_indices
            deltas = -cluster_cross[tri_a, tri_b]
            for op in rejected_operations:
                if op[0] == "merge":
                    deltas[(tri_a == op[1]) & (tri_b == op[2])] = np.inf
            valid = np.flatnonzero(deltas < -tolerance)
            if valid.size:
                min_delta = float(deltas[valid].min())
                tied = valid[np.isclose(deltas[valid], min_delta, atol=1e-7)]
                merge_choices = []
                for pos in tied.tolist():
                    ca, cb = int(tri_a[pos]), int(tri_b[pos])
                    tie = ("merge", cluster_mins[ca], cluster_mins[cb])
                    merge_choices.append((tie, ca, cb))
                tie, ca, cb = min(merge_choices)
                best = (-float(cluster_cross[ca, cb]), tie, ("merge", ca, cb))

        
        source_sum = node_cluster[np.arange(n), labels]
        delta_mat = source_sum[:, None] - node_cluster
        delta_mat[np.arange(n), labels] = np.inf
        for op in rejected_operations:
            if op[0] == "move" and op[2] is not None and len(op[3]) == 1:
                delta_mat[op[3][0], op[2]] = np.inf
        flat = int(np.argmin(delta_mat)) if delta_mat.size else -1
        if flat >= 0:
            u, cb = divmod(flat, k)
            delta = float(delta_mat[u, cb])
            if delta < -tolerance:
                tie = ("move", (int(u),), cluster_mins[int(labels[u])], cluster_mins[cb])
                cand = (delta, tie, ("move", int(labels[u]), cb, (int(u),)))
                if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                    best = cand
        
        split_deltas = source_sum.copy()
        for op in rejected_operations:
            if op[0] == "move" and op[2] is None and len(op[3]) == 1:
                split_deltas[op[3][0]] = np.inf
        u = int(np.argmin(split_deltas)) if n else -1
        if u >= 0 and float(split_deltas[u]) < -tolerance and len(blocks[int(labels[u])]) > 1:
            delta = float(source_sum[u])
            tie = ("move", (u,), cluster_mins[int(labels[u])], -1)
            cand = (delta, tie, ("move", int(labels[u]), None, (u,)))
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand

        
        
        for ca, source in enumerate(blocks):
            components = (neutral.same_components(source) if neutral is not None else
                          _direct_same_components(source, direct_edges))
            for moved in components:
                M = np.fromiter(sorted(moved), dtype=np.int32)
                internal = float(affinity[np.ix_(M, M)].sum(dtype=np.float64))
                source_total = float(node_cluster[M, ca].sum(dtype=np.float64))
                source_rem = source_total - internal
                delta = source_rem
                tie = ("move", tuple(M.tolist()), cluster_mins[ca], -1)
                cand = (delta, tie, ("move", ca, None, tuple(M.tolist())))
                if cand[2] not in rejected_operations and delta < -tolerance and (best is None or (cand[0],cand[1]) < (best[0],best[1])):
                    best = cand
                target_sums = node_cluster[M].sum(axis=0)
                if neutral is not None:
                    
                    
                    
                    
                    moved_tuple = tuple(M.tolist())
                    target_deltas = source_rem - target_sums
                    target_deltas[ca] = np.inf
                    for rejected in rejected_operations:
                        if rejected[0] == "move" and rejected[1] == ca and rejected[3] == moved_tuple and rejected[2] is not None:
                            target_deltas[rejected[2]] = np.inf
                    cb = int(np.argmin(target_deltas))
                    delta = float(target_deltas[cb])
                    tie = ("move", moved_tuple, cluster_mins[ca], cluster_mins[cb])
                    cand = (delta, tie, ("move", ca, cb, moved_tuple))
                    if delta < -tolerance and (best is None or (cand[0], cand[1]) < (best[0], best[1])):
                        best = cand
                    continue
                for cb in range(k):
                    if cb == ca:
                        continue
                    delta = source_rem - float(target_sums[cb])
                    tie = ("move", tuple(M.tolist()), cluster_mins[ca], cluster_mins[cb])
                    cand = (delta, tie, ("move", ca, cb, tuple(M.tolist())))
                    if cand[2] not in rejected_operations and delta < -tolerance and (best is None or (cand[0],cand[1]) < (best[0],best[1])):
                        best = cand

        if best is None:
            break
        _, _, op = best
        before = current
        blocks = [set(block) for block in partition]
        if op[0] == "merge":
            _, ca, cb = op
            merged = blocks[ca] | blocks[cb]
            new_blocks = [b for idx,b in enumerate(blocks) if idx not in (ca,cb)]
            new_blocks.append(merged)
        else:
            _, ca, cb, moved_tuple = op
            moved = set(moved_tuple); source = blocks[ca]; remainder = source - moved
            new_blocks = []
            for idx, block in enumerate(blocks):
                if idx == ca:
                    if remainder: new_blocks.append(remainder)
                elif cb is not None and idx == cb:
                    new_blocks.append(block | moved)
                else:
                    new_blocks.append(block)
            if cb is None: new_blocks.append(moved)
        proposed_partition = canonical_partition(new_blocks)
        proposed_loss = neutral.loss(proposed_partition) if neutral is not None else weighted_loss(proposed_partition, p)
        if before - proposed_loss <= tolerance:
            
            
            
            
            rejected_operations.add(op)
            continue
        partition = proposed_partition
        current = proposed_loss
        rejected_operations.clear()
        operations.append({"operation": op[0], "detail": list(op[1:]),
                           "loss_before": before, "loss_after": current,
                           "delta": current-before})

    return DecodeResult(partition=partition, loss=current,
                        steps=len(operations), operations=tuple(operations))
