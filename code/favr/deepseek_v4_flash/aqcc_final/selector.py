from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .decoder import decode, labels_of, partition_pair_distance


@dataclass(frozen=True)
class QueryCandidate:
    pair_index: int
    pair: tuple[int, int]
    category: str
    dense: float
    lexical: float
    p_same: float
    cheap_score: float
    expected_value: float
    component_pair: tuple[int, int]


class ActiveQuerySelector:
    """Budget-independent, component-aware active pair selector.

    All report pairs remain in the universe. Cheap signals decide when a pair is
    considered; no similarity threshold permanently blocks a pair.
    """

    def __init__(self, store, initial_partition, *, candidate_pool: int = 96,
                 lookahead: int = 0, exploration_fraction: float = 0.20,
                 frontier_base: int = 256, frontier_growth: float = 8.0,
                 max_node_degree_per_batch: int = 3,
                 max_component_pair_per_batch: int = 2,
                 member_neighbor_k: int = 12,
                 decoder_max_steps: int = 120,
                 random_seed: int | None = None,
                 capture_trace: bool = False):
        self.store = store
        self.initial_partition = initial_partition
        self.candidate_pool = max(16, int(candidate_pool))
        self.lookahead = max(0, int(lookahead))
        self.exploration_fraction = float(exploration_fraction)
        if not 0 <= self.exploration_fraction < 1:
            raise ValueError("exploration_fraction must be in [0,1)")
        self.frontier_base = max(1, int(frontier_base))
        self.frontier_growth = max(0.0, float(frontier_growth))
        self.max_node_degree_per_batch = max(1, int(max_node_degree_per_batch))
        self.max_component_pair_per_batch = max(1, int(max_component_pair_per_batch))
        self.member_neighbor_k = max(1, min(int(member_neighbor_k), store.dense_neighbors.shape[1]))
        self.decoder_max_steps = int(decoder_max_steps)
        self.capture_trace = bool(capture_trace)
        self.last_trace: dict | None = None
        self.random_order = None
        if random_seed is not None:
            if type(random_seed) is not int or not 0 <= random_seed < 2**64:
                raise ValueError("random_seed must be an integer in [0, 2**64)")
            
            
            self.random_order = np.random.Generator(np.random.PCG64(random_seed)).permutation(
                store.total_pairs)

    def _begin_trace(self, strategy, batch_size, queries_done, consumed):
        self.last_trace = None
        if self.capture_trace:
            self.last_trace = {
                "schema": "aqcc-selector-trace-v1", "selector_called": True,
                "strategy": strategy, "queries_done": int(queries_done),
                "consumed_pairs": len(consumed), "total_pairs": self.store.total_pairs,
                "requested_batch_size": int(batch_size), "selector_reads_final_budget": False,
                "discovery_scope": "SOURCE_SHORTLISTS_NOT_ALL_UNIVERSE_SCORES",
                "discovery_rows": [], "candidate_rows": [],
                "selected_pair_indices": [], "selected_order": [],
            }

    def _trace_candidates(self, rows):
        if self.last_trace is None:
            return {}
        records = []
        indexed = {}
        for rank, row in enumerate(rows, 1):
            record = {
                "candidate_index": rank, "shortlist_rank": rank,
                "pair_index": row.pair_index, "pair": list(row.pair), "category": row.category,
                "dense": row.dense, "lexical": row.lexical, "p_same": row.p_same,
                "cheap_score": row.cheap_score, "expected_value": row.expected_value,
                "component_pair": list(row.component_pair), "selection_attempts": [],
                "selected_rank": None, "decision": "NOT_CHOSEN",
            }
            records.append(record)
            indexed[id(row)] = record
        self.last_trace["candidate_rows"] = records
        return indexed

    def _finish_trace(self, selected):
        if self.last_trace is None:
            return
        selected_by_key = {(row.pair_index, row.category): rank
                           for rank, row in enumerate(selected, 1)}
        selected_pairs = {row.pair_index for row in selected}
        for record in self.last_trace["candidate_rows"]:
            key = (record["pair_index"], record["category"])
            if key in selected_by_key:
                record["selected_rank"] = selected_by_key[key]
                record["decision"] = "SELECTED"
            elif record["pair_index"] in selected_pairs:
                record["decision"] = "DUPLICATE_SOURCE_SELECTED"
            elif record["selection_attempts"]:
                record["decision"] = record["selection_attempts"][-1]["outcome"]
        self.last_trace["selected_pair_indices"] = [row.pair_index for row in selected]
        self.last_trace["selected_order"] = [
            {"rank": rank, "pair_index": row.pair_index, "pair": list(row.pair), "category": row.category}
            for rank, row in enumerate(selected, 1)
        ]
        self.last_trace["selected_count"] = len(selected)

    def _trace_ordered_selection(self, rows):
        indexed = self._trace_candidates(rows)
        for rank, row in enumerate(rows, 1):
            if id(row) in indexed:
                indexed[id(row)].update(order_rank_among_unconsumed=rank,
                                        counterfactual_evaluated=False)
                indexed[id(row)]["selection_attempts"].append({"phase": "ORDER_PREFIX", "outcome": "SELECTED"})
        self._finish_trace(rows)

    def _consumed_mask(self, consumed: set[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(self.store.total_pairs, dtype=bool)
        for a, b in consumed:
            mask[self.store.pair_index(a, b)] = True
        return mask

    def _first_unconsumed_from_order(self, order: np.ndarray, consumed_mask: np.ndarray,
                                     limit: int, prefix: int | None = None) -> list[int]:
        if limit <= 0:
            return []
        rows = order if prefix is None else order[:min(len(order), int(prefix))]
        out = []
        
        
        
        for idx in rows:
            idx = int(idx)
            if not consumed_mask[idx]:
                out.append(idx)
                if len(out) >= limit:
                    break
        return out

    def _shortlist(self, current_partition, probabilities: np.ndarray,
                   consumed: set[tuple[int, int]], queries_done: int) -> list[tuple[int, str, float]]:
        consumed_mask = self._consumed_mask(consumed)
        labels = labels_of(current_partition, self.store.n)
        sizes = np.bincount(labels)
        ii, jj = self.store.i, self.store.j
        same_now = labels[ii] == labels[jj]
        p = probabilities
        pressure = np.where(same_now, 1.0 - p, p)
        leverage = np.log1p(sizes[labels[ii]].astype(np.float64) * sizes[labels[jj]].astype(np.float64))
        cheap = pressure * leverage
        cheap[consumed_mask] = -np.inf

        tagged: dict[int, tuple[str, float]] = {}
        per = max(8, self.candidate_pool // 4)

        def record_source(idx, category, source_rank, source_score):
            if self.last_trace is not None:
                self.last_trace["discovery_rows"].append({
                    "pair_index": int(idx), "category": category,
                    "source_rank": int(source_rank), "source_score": float(source_score),
                    "source_order_rule": {
                        "BOUNDARY_PRESSURE": "BOUNDARY_PRESSURE_VALUE_DESCENDING",
                        "MEMBER_BRIDGE": "MAX_DENSE_LEXICAL_DESCENDING",
                        "PROGRESSIVE_FRONTIER": "GLOBAL_MULTIVIEW_ORDER_WITHIN_FRONTIER",
                        "EXPLORATION": "RANK_STRATIFIED_GLOBAL_MULTIVIEW_ORDER",
                    }[category],
                    "source_score_role": (
                        "BOUNDARY_PROXY_NOT_FRONTIER_SORT_KEY" if category == "PROGRESSIVE_FRONTIER"
                        else "UNSCORED_EXPLORATION" if category == "EXPLORATION"
                        else "SOURCE_SORT_SCORE"),
                    "boundary_pressure_value": float(cheap[idx]),
                })

        
        
        valid = np.flatnonzero(~consumed_mask)
        if valid.size:
            take = min(per, valid.size)
            top = valid[np.argpartition(-cheap[valid], take - 1)[:take]] if take < valid.size else valid
            top = sorted(top.tolist(), key=lambda x: (-float(cheap[x]), int(x)))[:take]
            for rank, idx in enumerate(top, 1):
                record_source(idx, "BOUNDARY_PRESSURE", rank, cheap[idx])
                tagged.setdefault(int(idx), ("BOUNDARY_PRESSURE", float(cheap[idx])))

        
        
        bridge = {}
        for u in range(self.store.n):
            for arr in (self.store.dense_neighbors, self.store.lexical_neighbors):
                for v in arr[u, :self.member_neighbor_k].tolist():
                    if labels[u] == labels[v]:
                        continue
                    idx = self.store.pair_index(u, int(v))
                    if consumed_mask[idx]:
                        continue
                    score = float(max(self.store.dense[idx], self.store.lexical[idx]))
                    if idx not in bridge or score > bridge[idx]:
                        bridge[idx] = score
        for rank, (idx, score) in enumerate(sorted(bridge.items(), key=lambda kv: (-kv[1], kv[0]))[:per], 1):
            record_source(idx, "MEMBER_BRIDGE", rank, score)
            tagged.setdefault(int(idx), ("MEMBER_BRIDGE", float(cheap[idx])))

        
        
        frontier_depth = min(self.store.total_pairs,
                             self.frontier_base + int(self.frontier_growth * queries_done))
        for rank, idx in enumerate(self._first_unconsumed_from_order(self.store.combined_order, consumed_mask,
                                                     per, prefix=frontier_depth), 1):
            record_source(idx, "PROGRESSIVE_FRONTIER", rank, cheap[idx])
            tagged.setdefault(idx, ("PROGRESSIVE_FRONTIER", float(cheap[idx])))

        
        
        
        explore_order = self.store.combined_order[self.store.exploration_order]
        explore_ids = self._first_unconsumed_from_order(
            explore_order, consumed_mask, max(8, self.candidate_pool // 2))
        for rank, idx in enumerate(explore_ids, 1):
            record_source(idx, "EXPLORATION", rank, 0.0)

        priority = {"BOUNDARY_PRESSURE": 0, "MEMBER_BRIDGE": 1,
                    "PROGRESSIVE_FRONTIER": 2}
        rows = [(idx, cat, score) for idx, (cat, score) in tagged.items()]
        rows.sort(key=lambda x: (priority[x[1]], -x[2], x[0]))
        main = rows[:self.candidate_pool]
        exploration = [(idx, "EXPLORATION", 0.0) for idx in explore_ids]
        if self.last_trace is not None:
            main_keys = {(idx, category) for idx, category, _ in main}
            for record in self.last_trace["discovery_rows"]:
                idx, category = record["pair_index"], record["category"]
                retained = tagged.get(idx, (None,))[0]
                record["retained_main_category"] = retained
                record["discovery_decision"] = (
                    "EXPLORATION_STREAM" if category == "EXPLORATION"
                    else "SOURCE_DUPLICATE" if retained != category
                    else "MAIN_SHORTLIST" if (idx, category) in main_keys
                    else "CANDIDATE_POOL_LIMIT"
                )
            self.last_trace["discovery_counts"] = {
                "unconsumed_pairs": int(valid.size), "member_bridge_discovered_pairs": len(bridge),
                "per_source_limit": per, "frontier_depth": frontier_depth,
                "main_shortlist_rows": len(main), "exploration_rows": len(exploration),
            }
        return main + exploration

    def _counterfactual_value(self, idx: int, p_same: float, current_partition,
                              probability_matrix: np.ndarray, direct_edges) -> float:
        a, b = self.store.pair_tuple(idx)
        plus = probability_matrix.copy(); plus[a, b] = plus[b, a] = 1.0
        minus = probability_matrix.copy(); minus[a, b] = minus[b, a] = 0.0
        
        
        plus_edges = dict(direct_edges); plus_edges[(a, b)] = 1
        minus_edges = dict(direct_edges); minus_edges[(a, b)] = -1
        pp = decode(self.initial_partition, plus, direct_edges=plus_edges,
                    max_steps=self.decoder_max_steps).partition
        pm = decode(self.initial_partition, minus, direct_edges=minus_edges,
                    max_steps=self.decoder_max_steps).partition
        dp = partition_pair_distance(current_partition, pp, self.store.n)
        dm = partition_pair_distance(current_partition, pm, self.store.n)
        return float(p_same * dp + (1.0 - p_same) * dm)

    def choose(self, *, strategy: str, current_partition, probabilities: np.ndarray,
               probability_matrix: np.ndarray, direct_edges: dict[tuple[int, int], int],
               consumed: set[tuple[int, int]], batch_size: int, queries_done: int) -> list[QueryCandidate]:
        self._begin_trace(strategy, batch_size, queries_done, consumed)
        if batch_size <= 0:
            self._finish_trace([])
            return []
        
        
        mask = self._consumed_mask(consumed) if strategy != "adaptive" else None
        labels = labels_of(current_partition, self.store.n)

        if strategy == "exhaustive":
            out = []
            for idx in range(self.store.total_pairs):
                if mask[idx]:
                    continue
                a, b = self.store.pair_tuple(idx)
                cp = tuple(sorted((int(labels[a]), int(labels[b]))))
                out.append(QueryCandidate(idx, (a, b), "EXHAUSTIVE", float(self.store.dense[idx]),
                                          float(self.store.lexical[idx]), float(probabilities[idx]),
                                          0.0, 0.0, cp))
                if len(out) >= batch_size:
                    break
            self._trace_ordered_selection(out)
            return out

        if strategy in {"dense", "multiview", "random"}:
            if strategy == "random":
                if self.random_order is None:
                    raise ValueError("random strategy requires random_seed")
                order = self.random_order
            else:
                order = self.store.dense_order if strategy == "dense" else self.store.combined_order
            rows = self._first_unconsumed_from_order(order, mask, batch_size)
            out = []
            for idx in rows:
                a, b = self.store.pair_tuple(idx)
                cp = tuple(sorted((int(labels[a]), int(labels[b]))))
                out.append(QueryCandidate(idx, (a, b), strategy.upper(), float(self.store.dense[idx]),
                                          float(self.store.lexical[idx]), float(probabilities[idx]),
                                          0.0, 0.0, cp))
            self._trace_ordered_selection(out)
            return out

        if strategy != "adaptive":
            raise ValueError("strategy must be adaptive, dense, multiview, random, or exhaustive")

        rows = self._shortlist(current_partition, probabilities, consumed, queries_done)
        if not rows:
            self._finish_trace([])
            return []

        
        
        labels = labels_of(current_partition, self.store.n)
        raw = []
        for idx, cat, cheap in rows:
            a, b = self.store.pair_tuple(idx)
            cp = tuple(sorted((int(labels[a]), int(labels[b]))))
            raw.append(QueryCandidate(idx, (a, b), cat, float(self.store.dense[idx]),
                                      float(self.store.lexical[idx]), float(probabilities[idx]),
                                      float(cheap), float(cheap), cp))
        main_for_lookahead = [r for r in raw if r.category != "EXPLORATION"]
        main_for_lookahead.sort(key=lambda r: (-r.cheap_score, r.pair_index))
        exact_indices = {r.pair_index for r in main_for_lookahead[:self.lookahead]}
        rescored = []
        for r in raw:
            value = r.expected_value
            if r.pair_index in exact_indices and self.lookahead > 0:
                value = self._counterfactual_value(r.pair_index, r.p_same, current_partition,
                                                   probability_matrix, direct_edges)
            rescored.append(QueryCandidate(r.pair_index, r.pair, r.category, r.dense, r.lexical,
                                           r.p_same, r.cheap_score, float(value), r.component_pair))

        explore_n = min(batch_size, max(1, int(math.ceil(batch_size * self.exploration_fraction))))
        main_n = batch_size - explore_n
        main = [r for r in rescored if r.category != "EXPLORATION"]
        main.sort(key=lambda r: (-r.expected_value, -r.cheap_score, r.pair_index))
        explore = [r for r in rescored if r.category == "EXPLORATION"]
        explore.sort(key=lambda r: r.pair_index)
        trace_by_id = self._trace_candidates(rescored)
        if self.last_trace is not None:
            self.last_trace["allocation"] = {
                "exploration_quota": explore_n, "main_quota": main_n,
                "max_node_degree_per_batch": self.max_node_degree_per_batch,
                "max_component_pair_per_batch": self.max_component_pair_per_batch,
                "lookahead": self.lookahead,
            }
            for r in rescored:
                trace_by_id[id(r)]["counterfactual_evaluated"] = (
                    r.pair_index in exact_indices and self.lookahead > 0)
            for stream_name, stream in (("main", main), ("exploration", explore)):
                for rank, r in enumerate(stream, 1):
                    trace_by_id[id(r)]["allocation_stream"] = stream_name
                    trace_by_id[id(r)]["allocation_rank"] = rank

        selected: list[QueryCandidate] = []
        node_degree: dict[int, int] = {}
        component_degree: dict[tuple[int, int], int] = {}
        selected_idx = set()

        def record_attempt(r, phase, outcome, **details):
            if self.last_trace is not None:
                trace_by_id[id(r)]["selection_attempts"].append(
                    {"phase": phase, "outcome": outcome, **details})

        def accept(r: QueryCandidate, phase: str) -> bool:
            a, b = r.pair
            if r.pair_index in selected_idx:
                record_attempt(r, phase, "REJECTED_DUPLICATE")
                return False
            if node_degree.get(a, 0) >= self.max_node_degree_per_batch:
                record_attempt(r, phase, "REJECTED_NODE_CAP", endpoint=a,
                               current_degree=node_degree.get(a, 0))
                return False
            if node_degree.get(b, 0) >= self.max_node_degree_per_batch:
                record_attempt(r, phase, "REJECTED_NODE_CAP", endpoint=b,
                               current_degree=node_degree.get(b, 0))
                return False
            if component_degree.get(r.component_pair, 0) >= self.max_component_pair_per_batch:
                record_attempt(r, phase, "REJECTED_COMPONENT_CAP",
                               current_degree=component_degree.get(r.component_pair, 0))
                return False
            selected.append(r); selected_idx.add(r.pair_index)
            node_degree[a] = node_degree.get(a, 0) + 1
            node_degree[b] = node_degree.get(b, 0) + 1
            component_degree[r.component_pair] = component_degree.get(r.component_pair, 0) + 1
            record_attempt(r, phase, "SELECTED", selected_rank=len(selected))
            return True

        
        
        added_explore = 0
        for pos, r in enumerate(explore):
            if added_explore >= explore_n:
                if self.last_trace is not None:
                    for skipped in explore[pos:]:
                        record_attempt(skipped, "EXPLORATION_QUOTA", "QUOTA_REACHED")
                break
            if accept(r, "EXPLORATION_QUOTA"):
                added_explore += 1
        main_added = 0
        for pos, r in enumerate(main):
            if main_added >= main_n:
                if self.last_trace is not None:
                    for skipped in main[pos:]:
                        record_attempt(skipped, "MAIN_QUOTA", "QUOTA_REACHED")
                break
            if accept(r, "MAIN_QUOTA"):
                main_added += 1
        
        if len(selected) < batch_size:
            fill_rows = sorted(rescored, key=lambda r: (-r.expected_value, r.pair_index))
            for pos, r in enumerate(fill_rows):
                if len(selected) >= batch_size:
                    if self.last_trace is not None:
                        for skipped in fill_rows[pos:]:
                            record_attempt(skipped, "FILL_REMAINDER", "BATCH_FULL")
                    break
                accept(r, "FILL_REMAINDER")
        self._finish_trace(selected)
        return selected
