from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Callable, Iterable

import numpy as np


def canonical_json_sha256(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_partition(blocks: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    cleaned = [tuple(sorted(set(int(i) for i in block))) for block in blocks if block]
    cleaned.sort(key=lambda block: (block[0], len(block), block))
    return tuple(cleaned)


class BudgetExhausted(RuntimeError):
    pass


class ConstraintConflict(RuntimeError):
    pass


class ConstraintStore:
    """Direct answers plus deterministic must/cannot-link transitive inference."""

    def __init__(self, n: int):
        self.n = n
        self.direct: dict[tuple[int, int], int] = {}
        self._root = list(range(n))
        self._cannot: set[tuple[int, int]] = set()

    @staticmethod
    def pair(a: int, b: int) -> tuple[int, int]:
        if a == b:
            raise ValueError("a relation requires two distinct reports")
        return (a, b) if a < b else (b, a)

    def _find(self, x: int) -> int:
        while self._root[x] != x:
            self._root[x] = self._root[self._root[x]]
            x = self._root[x]
        return x

    def _rebuild(self) -> None:
        self._root = list(range(self.n))

        def union(a: int, b: int) -> None:
            ra, rb = self._find(a), self._find(b)
            if ra != rb:
                if ra > rb:
                    ra, rb = rb, ra
                self._root[rb] = ra

        for (a, b), label in self.direct.items():
            if label == 1:
                union(a, b)
        for i in range(self.n):
            self._root[i] = self._find(i)
        cannot = set()
        for (a, b), label in self.direct.items():
            if label != -1:
                continue
            ra, rb = self._find(a), self._find(b)
            if ra == rb:
                raise ConstraintConflict("direct SAME and DIFFERENT answers are inconsistent")
            cannot.add((ra, rb) if ra < rb else (rb, ra))
        self._cannot = cannot

    def add(self, a: int, b: int, label: int) -> None:
        if label not in (-1, 1):
            raise ValueError("constraint label must be -1 or 1")
        pair = self.pair(a, b)
        old = self.direct.get(pair)
        if old is not None and old != label:
            raise ConstraintConflict("the same directly queried pair has conflicting answers")
        self.direct[pair] = label
        self._rebuild()

    def relation(self, a: int, b: int) -> int:
        if a == b:
            return 1
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return 1
        return -1 if ((ra, rb) if ra < rb else (rb, ra)) in self._cannot else 0

    def digest(self) -> str:
        rows = [[a, b, label] for (a, b), label in sorted(self.direct.items())]
        return canonical_json_sha256(rows)


@dataclass(frozen=True)
class QueryOutcome:
    relation: int
    status: str
    query_index: int | None
    inferred: bool
    record: dict | None
    pair_indices: tuple[int, int] | None = None


class BudgetedPairOracle:
    """A3S-visible oracle with strict budget, failure and inference separation."""

    def __init__(self, report_ids: tuple[str, ...], texts: tuple[str, ...], judge: Callable,
                 budget: int):
        if len(report_ids) != len(texts) or len(set(report_ids)) != len(report_ids):
            raise ValueError("invalid report population")
        if budget < 0:
            raise ValueError("budget must be nonnegative")
        self.report_ids = report_ids
        self.texts = texts
        self.judge = judge
        self.budget = int(budget)
        self.logical_queries = 0
        self.constraints = ConstraintStore(len(report_ids))
        self.attempted: dict[tuple[int, int], QueryOutcome] = {}
        self.records: list[dict] = []

    def _call_judge(self, pair: tuple[int, int]) -> dict:
        left, right = pair
        return self.judge(self.report_ids[left], self.texts[left],
                          self.report_ids[right], self.texts[right])

    def _integrate_direct(self, pair: tuple[int, int], result: dict, phase: str,
                          partition_before) -> QueryOutcome:
        left, right = pair
        self.logical_queries += 1
        status = str(result.get("status", "FAILED_INVALID_RESULT"))
        label = result.get("label")
        relation = 1 if status == "SUCCESS" and label == "SAME" else (
            -1 if status == "SUCCESS" and label == "DIFFERENT" else 0)
        if relation:
            self.constraints.add(left, right, relation)
        record = {
            "query_index": self.logical_queries,
            "pair_indices": [left, right],
            "report_ids": [self.report_ids[left], self.report_ids[right]],
            "phase": phase,
            "status": status,
            "label": label if relation else None,
            "reason": result.get("reason", "") if relation else "",
            "request_sha256": result.get("request_sha256"),
            "response_sha256": result.get("response_sha256"),
            "attempts": result.get("attempts", []),
            "usage": result.get("usage", {}),
            "physical_usage": result.get("physical_usage", {}),
            "physical_attempts": result.get("physical_attempts", []),
            "shared_oracle": result.get("shared_oracle"),
            "raw_response_body": result.get("raw_response_body"),
            "raw_response_sha256": result.get("raw_response_sha256"),
            "raw_response_redacted": result.get("raw_response_redacted"),
            "partition_before": [list(block) for block in partition_before],
            "partition_after": None,
            "constraint_sha256": self.constraints.digest(),
        }
        outcome = QueryOutcome(relation, status, self.logical_queries, False, record, pair)
        self.attempted[pair] = outcome
        self.records.append(record)
        return outcome

    def ask(self, a: int, b: int, phase: str, partition_before) -> QueryOutcome:
        pair = ConstraintStore.pair(a, b)
        relation = self.constraints.relation(*pair)
        if relation:
            return QueryOutcome(relation, "INFERRED", None, True, None, pair)
        if pair in self.attempted:
            return self.attempted[pair]
        if self.logical_queries >= self.budget:
            raise BudgetExhausted("logical query budget exhausted")
        return self._integrate_direct(pair, self._call_judge(pair), phase, partition_before)

    def ask_many(self, pairs, phase: str, partition_before, *, max_workers: int):
        """Resolve a deterministic micro-batch with bounded physical concurrency.

        Pair selection is frozen before dispatch. Provider calls run in
        parallel, but results are integrated and yielded in input order so
        query indices, constraint updates and partition commits remain stable.
        A pair that becomes inferable within the same batch is still counted:
        its physical request was already dispatched under batched semantics.
        """
        normalized = [ConstraintStore.pair(a, b) for a, b in pairs]
        if len(set(normalized)) != len(normalized):
            raise ValueError("a query micro-batch cannot contain duplicate pairs")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")

        remaining = self.budget - self.logical_queries
        planned = []
        direct_pairs = []
        for pair in normalized:
            relation = self.constraints.relation(*pair)
            if relation:
                planned.append(("READY", QueryOutcome(
                    relation, "INFERRED", None, True, None, pair)))
            elif pair in self.attempted:
                planned.append(("READY", self.attempted[pair]))
            elif len(direct_pairs) < remaining:
                planned.append(("DIRECT", len(direct_pairs)))
                direct_pairs.append(pair)
            else:
                planned.append(("BUDGET", None))

        if direct_pairs:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(direct_pairs)),
                                    thread_name_prefix="a3s-oracle") as pool:
                direct_results = list(pool.map(self._call_judge, direct_pairs))
        else:
            direct_results = []

        for position, (kind, value) in enumerate(planned):
            if kind == "READY":
                yield value
            elif kind == "DIRECT":
                pair = normalized[position]
                yield self._integrate_direct(
                    pair, direct_results[value], phase, partition_before)
            else:
                raise BudgetExhausted("logical query budget exhausted")


@dataclass(frozen=True)
class A3SConfig:
    tau: float = 0.8
    rounds: int = 2
    neighbor_count: int = 50
    max_local_query: int = 20
    max_local_clusters: int = 3
    seed: int = 0
    verify_supplied_partition: bool = True
    max_outlier_candidates: int = 4
    api_concurrency: int = 1

    def __post_init__(self):
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be in [0,1]")
        if self.rounds < 1 or self.neighbor_count < 2:
            raise ValueError("rounds and neighbor_count must be positive")
        if type(self.api_concurrency) is not int or not 1 <= self.api_concurrency <= 256:
            raise ValueError("api_concurrency must be an integer in [1,256]")


class A3SEngine:
    """Supplied-initialization port of the authors' A3S adjustment process.

    Query selection, purity checks, aggregation, purification and outlier merge
    follow the public A3S.py structure. The adapter replaces its Gold-backed
    query function, enforces a budget at every query site, and treats technical
    failures as unknown rather than cannot-link.
    """

    def __init__(self, probability: np.ndarray, initial_partition, oracle: BudgetedPairOracle,
                 config: A3SConfig, checkpoints: Iterable[int] = (0,), commit_hook=None):
        probability = np.asarray(probability, dtype=np.float64)
        n = len(oracle.report_ids)
        if probability.shape != (n, n) or not np.isfinite(probability).all():
            raise ValueError("probability matrix has invalid shape or values")
        self.base_probability = np.clip(probability, 0.0001, 0.9999)
        np.fill_diagonal(self.base_probability, 0.9999)
        
        
        
        
        
        self.neighbor_order = np.argsort(
            -self.base_probability, axis=1, kind="stable"
        )[:, :min(config.neighbor_count, n)]
        self.oracle = oracle
        self.config = config
        self.rng = random.Random(config.seed)
        partition = canonical_partition(initial_partition)
        seen = sorted(i for block in partition for i in block)
        if seen != list(range(n)):
            raise ValueError("initial partition must cover every report exactly once")
        self.n = n
        self.labels = np.empty(n, dtype=np.int64)
        self.clusters: dict[int, set[int]] = {}
        for key, block in enumerate(partition):
            self.clusters[key] = set(block)
            for i in block:
                self.labels[i] = key
        self.next_cluster = len(self.clusters)
        self.leaders: dict[int, int] = {}
        self.recall_count = sum(bool(len(block) > 1) for block in partition)
        self.checkpoint_targets = sorted(set(int(q) for q in checkpoints if q >= 0))
        self.checkpoints: list[dict] = []
        self.events: list[dict] = []
        self.stop_reason = "COMPLETED_ALGORITHM"
        self.rounds_completed = 0
        self.commit_hook = commit_hook
        self._save_checkpoint("INITIAL")

    def partition(self):
        return canonical_partition(self.clusters.values())

    def _validate(self) -> None:
        partition = self.partition()
        seen = sorted(i for block in partition for i in block)
        if seen != list(range(self.n)):
            raise RuntimeError("A3S produced a non-complete partition")
        for key, members in self.clusters.items():
            for i in members:
                if int(self.labels[i]) != key:
                    raise RuntimeError("cluster dictionary and labels disagree")

    def _save_checkpoint(self, trigger: str) -> None:
        q = self.oracle.logical_queries
        existing = {row["logical_queries"] for row in self.checkpoints}
        if q in existing:
            return
        if q == 0 or q in self.checkpoint_targets or trigger in {"TERMINAL", "EARLY_STOP"}:
            part = self.partition()
            self.checkpoints.append({
                "logical_queries": q,
                "trigger": trigger,
                "clusters": len(part),
                "partition": [list(block) for block in part],
                "partition_sha256": canonical_json_sha256(part),
                "constraint_sha256": self.oracle.constraints.digest(),
            })

    def _commit(self, action: str, outcome: QueryOutcome | None = None) -> None:
        self._validate()
        partition = self.partition()
        if outcome is not None and outcome.record is not None:
            outcome.record["partition_after"] = [list(block) for block in partition]
            outcome.record["partition_after_sha256"] = canonical_json_sha256(partition)
            outcome.record["action"] = action
        event = {"logical_queries": self.oracle.logical_queries,
                 "action": action, "clusters": len(partition),
                 "partition": [list(block) for block in partition],
                 "partition_sha256": canonical_json_sha256(partition)}
        if outcome is not None:
            event.update({
                "query_index": outcome.query_index,
                "pair_indices": list(outcome.pair_indices) if outcome.pair_indices else None,
                "relation": outcome.relation,
                "status": outcome.status,
                "inferred": outcome.inferred,
            })
        self.events.append(event)
        self._save_checkpoint(action)
        if self.commit_hook is not None:
            self.commit_hook(self)

    def _ask(self, a: int, b: int, phase: str) -> QueryOutcome:
        return self.oracle.ask(a, b, phase, self.partition())

    def _ask_many(self, pairs, phase: str):
        return self.oracle.ask_many(
            pairs, phase, self.partition(),
            max_workers=self.config.api_concurrency)

    def pair_probability(self, a: int, b: int) -> float:
        relation = self.oracle.constraints.relation(a, b)
        return 0.9999 if relation == 1 else (0.0001 if relation == -1 else float(self.base_probability[a, b]))

    def _logit(self, a: int, b: int) -> float:
        p = self.pair_probability(a, b)
        return math.log(p / (1.0 - p))

    def _ll(self, item: int, key: int) -> float:
        members = self.clusters.get(key, set())
        return sum(self._logit(item, other) for other in members if other != item) / max(1, len(members))

    def _within_ll(self, item: int) -> float:
        return sum(self._logit(item, other)
                   for other in self.clusters[int(self.labels[item])] if other != item)

    def _sort_ll(self, members) -> list[int]:
        return sorted(members, key=lambda i: (-self._within_ll(i), i))

    def _sort_probability(self, item: int, members) -> list[int]:
        return sorted(members, key=lambda j: (-self.pair_probability(item, j), j))

    def _adjust(self, item: int, target: int) -> None:
        old, new = int(self.labels[item]), int(self.labels[target])
        if old == new:
            return
        self.clusters[old].remove(item)
        self.clusters[new].add(item)
        self.labels[item] = new
        if not self.clusters[old]:
            self.clusters.pop(old)
            self.leaders.pop(old, None)

    def _new_singleton(self, item: int) -> int:
        old = int(self.labels[item])
        if len(self.clusters[old]) == 1:
            return old
        self.clusters[old].remove(item)
        key = self.next_cluster
        self.next_cluster += 1
        self.clusters[key] = {item}
        self.labels[item] = key
        return key

    def _merge(self, source: int, target: int) -> None:
        if source == target or source not in self.clusters or target not in self.clusters:
            return
        target_member = min(self.clusters[target])
        for item in list(self.clusters[source]):
            self._adjust(item, target_member)

    def compactness(self, key: int) -> bool:
        members = self.clusters[key]
        number = len(members)
        if number < 2:
            return True
        k_neighbor = max(1, int(math.sqrt(number)))
        total = 0.0
        count = 0
        for item in sorted(members):
            ordered = self._sort_probability(item, members)
            chosen = ordered[number // 2:number // 2 + k_neighbor]
            total += sum(self.pair_probability(item, other) for other in chosen)
            count += len(chosen)
        score = total / count if count else None
        decision = score is not None and score > self.config.tau
        partition = self.partition()
        self.events.append({
            "logical_queries": self.oracle.logical_queries,
            "action": "PURITY_COMPACTNESS_TEST",
            "cluster_key": key,
            "members": sorted(members),
            "score": score,
            "threshold": self.config.tau,
            "decision": decision,
            "clusters": len(partition),
            "partition": [list(block) for block in partition],
            "partition_sha256": canonical_json_sha256(partition),
        })
        return decision

    def merging_probability(self, key1: int, key2: int) -> float:
        set1, set2 = self.clusters[key1], self.clusters[key2]
        if len(set1) > len(set2):
            set1, set2 = set2, set1
        values = []
        for item in sorted(set1):
            probs = sorted((self.pair_probability(item, other) for other in set2), reverse=True)
            values.extend(probs[:4])
        
        positive = 1.0 + sum(values)
        negative = 1.0 + sum(1.0 - value for value in values)
        return positive / (positive + negative)

    def _delta_h(self, key1: int, key2: int) -> float:
        p, q = len(self.clusters[key1]), len(self.clusters[key2])
        return ((p + q) * math.log(p + q) - p * math.log(p) - q * math.log(q)) / self.n

    def _leader_small(self, key: int):
        pile = self._sort_ll(self.clusters[key])
        if len(pile) < 2:
            return pile[0] if pile else None, "OK"
        if self.compactness(key):
            return pile[0], "OK"
        outcome = self._ask(pile[0], pile[1], "PURITY_SMALL")
        if outcome.relation == 1:
            self._commit("PURITY_SMALL_SAME", outcome)
            return pile[0], "OK"
        if outcome.relation == -1:
            self._new_singleton(pile[1])
            self._commit("PURITY_SMALL_SPLIT", outcome)
            return None, "IMPURE"
        self._commit("PURITY_SMALL_FAILURE", outcome)
        return None, "FAILED"

    def _leader_big(self, key: int):
        pile = self._sort_ll(self.clusters[key])
        if self.compactness(key):
            return pile[0], "OK"
        rate = 0.5 if len(pile) < 15 else self.rng.uniform(0.5, 0.7)
        attempts = 0
        while pile and attempts < 2:
            i = pile[min(attempts, len(pile) - 1)]
            j = pile[min(int(len(pile) * rate), len(pile) - 1)]
            if i == j:
                j = pile[-1 if i != pile[-1] else 0]
            if i == j:
                return i, "OK"
            outcome = self._ask(i, j, "PURITY_BIG")
            if outcome.relation == 1:
                self._commit("PURITY_BIG_SAME", outcome)
                return i, "OK"
            if outcome.relation == -1:
                self._new_singleton(j)
                self._commit("PURITY_BIG_SPLIT", outcome)
                pile = [x for x in pile if x != j]
                attempts += 1
                continue
            self._commit("PURITY_BIG_FAILURE", outcome)
            return None, "FAILED"
        return None, "IMPURE"

    def _purify(self, key: int) -> None:
        if key not in self.clusters or len(self.clusters[key]) < 2:
            return
        original = key
        pile = self._sort_ll(list(self.clusters[key]))
        created: list[int] = []
        local_queries = 0
        while pile and local_queries <= self.config.max_local_query:
            current = pile.pop(0)
            placed = False
            had_successful_comparison = False
            candidates = [self.leaders[k] for k in created if k in self.leaders and k in self.clusters]
            for candidate in self._sort_probability(current, candidates):
                before = self.oracle.logical_queries
                outcome = self._ask(current, candidate, "PURIFICATION_ASSIGN")
                local_queries += self.oracle.logical_queries - before
                if outcome.relation:
                    had_successful_comparison = True
                if outcome.relation == 1:
                    self._adjust(current, candidate)
                    self._commit("PURIFICATION_ASSIGN_SAME", outcome)
                    placed = True
                    break
                self._commit("PURIFICATION_ASSIGN_DIFFERENT" if outcome.relation == -1
                             else "PURIFICATION_ASSIGN_FAILURE", outcome)
            if not placed:
                
                if candidates and not had_successful_comparison:
                    continue
                new_key = self._new_singleton(current)
                if new_key == original:
                    new_key = self._new_singleton(current)
                self.leaders[new_key] = current
                if new_key not in created:
                    created.append(new_key)
                self._commit("PURIFICATION_CREATE_SUBCLUSTER")
            remaining = self._sort_probability(current, list(pile))
            different_streak = 0
            for other in remaining:
                if local_queries > self.config.max_local_query:
                    break
                before = self.oracle.logical_queries
                outcome = self._ask(current, other, "PURIFICATION_EXPAND")
                local_queries += self.oracle.logical_queries - before
                if outcome.relation == 1:
                    self._adjust(other, current)
                    if other in pile:
                        pile.remove(other)
                    different_streak = 0
                    self._commit("PURIFICATION_EXPAND_SAME", outcome)
                elif outcome.relation == -1:
                    different_streak += 1
                    self._commit("PURIFICATION_EXPAND_DIFFERENT", outcome)
                    if different_streak >= 4:
                        break
                else:
                    self._commit("PURIFICATION_EXPAND_FAILURE", outcome)
            if len(created) > self.config.max_local_clusters:
                for item in list(pile):
                    self._new_singleton(item)
                self._commit("PURIFICATION_MAX_SUBCLUSTERS")
                break
        
        
        
        
        
        if pile and local_queries > self.config.max_local_query:
            valid_keys = [k for k in created
                          if k in self.clusters and k in self.leaders
                          and self.leaders[k] in self.clusters[k]]
            for item in list(pile):
                if not valid_keys:
                    break
                target_key = max(valid_keys, key=lambda k: (self._ll(item, k), -k))
                self._adjust(item, self.leaders[target_key])
            self._commit("PURIFICATION_ASSIGN_REMAINDER")
        self.leaders.pop(original, None)

    def _obtain_leader(self, key: int):
        if key not in self.clusters or not self.clusters[key]:
            return None, "EMPTY"
        leader = self.leaders.get(key)
        if leader in self.clusters[key]:
            return leader, "OK"
        if len(self.clusters[key]) > 4:
            leader, status = self._leader_big(key)
        else:
            leader, status = self._leader_small(key)
        if leader is not None:
            self.leaders[key] = leader
        return leader, status

    def _nearest_neighbor(self, item: int):
        """Return the first valid entry in the reference-style KNN row."""
        for neighbor in self.neighbor_order[item]:
            neighbor = int(neighbor)
            if neighbor == item:
                continue
            if self.oracle.constraints.relation(item, neighbor) == -1:
                continue
            return neighbor
        return None

    def _neighbor_leaders(self, item: int, limit: int | None = None) -> list[int]:
        """Return distinct valid cluster leaders in fixed KNN order.

        The public code looks only at KNN position 1 because its own FPC start
        rarely produces singleton chains.  A supplied P0 can be dominated by
        singleton clusters; stopping at the first leaderless singleton would
        make the outlier phase inert.  Scanning the same fixed KNN row retains
        the authors' locality rule while making it usable with an external P0.
        """
        limit = self.config.max_outlier_candidates if limit is None else limit
        item_key = int(self.labels[item])
        result = []
        for neighbor in self.neighbor_order[item]:
            neighbor = int(neighbor)
            if neighbor == item:
                continue
            if self.oracle.constraints.relation(item, neighbor) == -1:
                continue
            key = int(self.labels[neighbor])
            leader = self.leaders.get(key)
            if key == item_key:
                continue
            if key in self.clusters and leader in self.clusters[key]:
                if leader not in result:
                    result.append(leader)
                    if len(result) >= limit:
                        break
        return result

    def _nearest_neighbor_leader(self, item: int):
        candidates = self._neighbor_leaders(item, limit=1)
        return candidates[0] if candidates else None

    def _complete_leader_coverage(self) -> None:
        """Establish checked leaders for non-singleton P0 clusters.

        In the reference pipeline A3S controls initialization, so candidate
        aggregation normally visits the useful multi-item clusters.  With a
        supplied partition this is not guaranteed.  Completing coverage before
        outlier attachment prevents the top-K aggregation shortlist from
        silently excluding otherwise valid destination clusters.
        """
        for key in list(sorted(self.clusters)):
            if key not in self.clusters or len(self.clusters[key]) < 2:
                continue
            leader, status = self._obtain_leader(key)
            if status == "IMPURE":
                self._purify(key)
            elif status == "FAILED":
                continue
            elif leader is not None:
                self.leaders[key] = leader

    def _verify_supplied_partition(self) -> None:
        """Validate every non-singleton P0 member against a fixed medoid.

        A3S's native probabilistic initialization is assumed to yield mostly
        pure clusters.  An externally supplied partition has no such guarantee;
        merging it after only a one-pair purity probe can propagate false
        positives even with a perfect oracle.  This deterministic pre-pass uses
        the same budgeted oracle and separates only explicit DIFFERENT answers.
        """
        if self.config.api_concurrency == 1:
            for key in list(sorted(self.clusters)):
                if key not in self.clusters or len(self.clusters[key]) < 2:
                    continue
                members = self._sort_ll(self.clusters[key])
                leader = members[0]
                self.leaders[key] = leader
                for item in members[1:]:
                    if key not in self.clusters or item not in self.clusters[key]:
                        continue
                    outcome = self._ask(leader, item, "P0_VERIFY")
                    if outcome.relation == -1:
                        self._new_singleton(item)
                    self._commit("P0_VERIFY_SAME" if outcome.relation == 1 else (
                        "P0_VERIFY_SPLIT" if outcome.relation == -1
                        else "P0_VERIFY_FAILURE"), outcome)
            return

        tasks = []
        for key in list(sorted(self.clusters)):
            if key not in self.clusters or len(self.clusters[key]) < 2:
                continue
            members = self._sort_ll(self.clusters[key])
            leader = members[0]
            self.leaders[key] = leader
            for item in members[1:]:
                tasks.append((key, leader, item))

        width = self.config.api_concurrency
        for offset in range(0, len(tasks), width):
            batch = tasks[offset:offset + width]
            outcomes = self._ask_many(
                [(leader, item) for _, leader, item in batch], "P0_VERIFY")
            for (key, leader, item), outcome in zip(batch, outcomes):
                if outcome.relation == -1:
                    if key in self.clusters and item in self.clusters[key]:
                        self._new_singleton(item)
                self._commit("P0_VERIFY_SAME" if outcome.relation == 1 else (
                    "P0_VERIFY_SPLIT" if outcome.relation == -1 else "P0_VERIFY_FAILURE"), outcome)

    def _outlier_merge(self) -> None:
        for key in list(sorted(self.clusters)):
            if key not in self.clusters or len(self.clusters[key]) != 2:
                continue
            a, b = sorted(self.clusters[key])
            outcome = self._ask(a, b, "OUTLIER_PAIR")
            if outcome.relation == 1:
                self.leaders[key] = a
                self._commit("OUTLIER_PAIR_SAME", outcome)
                
                
                
                candidates = []
                for item in (a, b):
                    for candidate in self._neighbor_leaders(item):
                        if candidate not in candidates:
                            candidates.append(candidate)
                        if len(candidates) >= self.config.max_outlier_candidates:
                            break
                    if len(candidates) >= self.config.max_outlier_candidates:
                        break
                for candidate in candidates:
                    follow = self._ask(a, candidate, "OUTLIER_PAIR_ATTACH")
                    if follow.relation == 1:
                        self._adjust(a, candidate)
                        self._adjust(b, candidate)
                        self._commit("OUTLIER_PAIR_ATTACH", follow)
                        break
                    self._commit("OUTLIER_PAIR_ATTACH", follow)
            elif outcome.relation == -1:
                self._new_singleton(a)
                self._commit("OUTLIER_PAIR_SPLIT", outcome)
            else:
                self._commit("OUTLIER_PAIR_FAILURE", outcome)
        if self.config.api_concurrency == 1:
            for key in list(sorted(self.clusters)):
                if key not in self.clusters or len(self.clusters[key]) != 1:
                    continue
                item = next(iter(self.clusters[key]))
                for candidate in self._neighbor_leaders(item):
                    outcome = self._ask(item, candidate, "OUTLIER_SINGLE_ATTACH")
                    if outcome.relation == 1:
                        self._adjust(item, candidate)
                        self._commit("OUTLIER_SINGLE_ATTACH", outcome)
                        break
                    self._commit("OUTLIER_SINGLE_ATTACH", outcome)
            return

        pending = {
            next(iter(self.clusters[key])): set()
            for key in sorted(self.clusters) if len(self.clusters[key]) == 1
        }
        while pending:
            batch = []
            for item in sorted(list(pending)):
                key = int(self.labels[item])
                if key not in self.clusters or self.clusters[key] != {item}:
                    pending.pop(item, None)
                    continue
                candidates = [candidate for candidate in self._neighbor_leaders(item)
                              if candidate not in pending[item]]
                if not candidates:
                    pending.pop(item, None)
                    continue
                candidate = candidates[0]
                pending[item].add(candidate)
                batch.append((item, candidate))
                if len(batch) >= self.config.api_concurrency:
                    break
            if not batch:
                break
            outcomes = self._ask_many(batch, "OUTLIER_SINGLE_ATTACH")
            for (item, candidate), outcome in zip(batch, outcomes):
                if outcome.relation == 1:
                    self._adjust(item, candidate)
                    pending.pop(item, None)
                self._commit("OUTLIER_SINGLE_ATTACH", outcome)

    def run(self) -> dict:
        try:
            if self.config.verify_supplied_partition:
                self._verify_supplied_partition()
            for round_index in range(self.config.rounds):
                keys = [key for key in sorted(self.clusters) if len(self.clusters[key]) > 1]
                candidates = []
                for pos, key1 in enumerate(keys[:-1]):
                    for key2 in keys[pos + 1:]:
                        candidates.append([key1, key2, self.merging_probability(key1, key2)])
                candidates.sort(key=lambda row: (-row[2], row[0], row[1]))
                partition = self.partition()
                self.events.append({
                    "logical_queries": self.oracle.logical_queries,
                    "action": "CANDIDATE_SET",
                    "round": round_index + 1,
                    "candidate_count": len(candidates),
                    "ranked_candidates": [
                        {"cluster_a": int(a), "cluster_b": int(b), "score": float(score)}
                        for a, b, score in candidates
                    ],
                    "clusters": len(partition),
                    "partition": [list(block) for block in partition],
                    "partition_sha256": canonical_json_sha256(partition),
                })
                candidates = candidates[:self.recall_count]
                for row in candidates:
                    if row[0] in self.clusters and row[1] in self.clusters:
                        row[2] *= self._delta_h(row[0], row[1])
                candidates.sort(key=lambda row: (-row[2], row[0], row[1]))
                for key1, key2, score in candidates:
                    if key1 not in self.clusters or key2 not in self.clusters or key1 == key2:
                        continue
                    leader1, status1 = self._obtain_leader(key1)
                    if status1 == "IMPURE":
                        self._purify(key1)
                        continue
                    if status1 == "FAILED" or leader1 is None:
                        continue
                    leader2, status2 = self._obtain_leader(key2)
                    if status2 == "IMPURE":
                        self._purify(key2)
                        continue
                    if status2 == "FAILED" or leader2 is None:
                        continue
                    outcome = self._ask(leader1, leader2, "AGGREGATION")
                    if outcome.relation == 1:
                        target = int(self.labels[leader2])
                        source = int(self.labels[leader1])
                        self._merge(source, target)
                    self._commit("AGGREGATION_SAME" if outcome.relation == 1 else (
                        "AGGREGATION_DIFFERENT" if outcome.relation == -1 else "AGGREGATION_FAILURE"), outcome)
                self.events.append({"logical_queries": self.oracle.logical_queries,
                                    "action": "ROUND_COMPLETE", "round": round_index + 1,
                                    "clusters": len(self.partition()),
                                    "partition": [list(block) for block in self.partition()],
                                    "partition_sha256": canonical_json_sha256(self.partition())})
                self.rounds_completed = round_index + 1
            self.stop_reason = "PRESET_ROUNDS_COMPLETED"
            self._complete_leader_coverage()
            self._outlier_merge()
        except BudgetExhausted:
            self.stop_reason = "BUDGET_EXHAUSTED"
        self._commit("TERMINAL")
        self._save_checkpoint("TERMINAL")
        return self.terminal()

    def terminal(self) -> dict:
        part = self.partition()
        return {
            "schema": "a3s-p0-terminal-v1",
            "status": "COMPLETE",
            "stop_reason": self.stop_reason,
            "reports": self.n,
            "logical_queries": self.oracle.logical_queries,
            "budget": self.oracle.budget,
            "clusters": len(part),
            "partition": [list(block) for block in part],
            "partition_sha256": canonical_json_sha256(part),
            "constraint_sha256": self.oracle.constraints.digest(),
            "successful_direct_relations": len(self.oracle.constraints.direct),
            "failed_direct_queries": sum(row["status"] != "SUCCESS" for row in self.oracle.records),
            "rounds_requested": self.config.rounds,
            "rounds_completed": self.rounds_completed,
            "remaining_budget": self.oracle.budget - self.oracle.logical_queries,
        }
