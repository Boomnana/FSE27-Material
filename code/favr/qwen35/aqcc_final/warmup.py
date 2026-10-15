"""Answer-independent initial query schedules for the follow-up experiment."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np

from .decoder import labels_of
from .provider import canonical_sha256
from .selector import QueryCandidate


STAGE = "FIXED_INITIAL_WARMUP"
SLOTS = ("WARMUP_INTRA",) * 8 + ("WARMUP_CROSS_NEAR",) * 6 + ("WARMUP_EXPLORATION",) * 2


@dataclass(frozen=True)
class FixedInitialWarmup:
    policy: str
    count: int
    seed: int
    candidates: tuple[QueryCandidate, ...]
    document: dict

    @classmethod
    def build(cls, dataset, store, *, policy: str, count: int = 512, seed: int = 0):
        if policy not in {"intra_fixed", "mixed_fixed"}:
            raise ValueError("invalid fixed initial query policy")
        if type(count) is not int or not 1 <= count <= store.total_pairs:
            raise ValueError("warmup_pairs outside population pair range")
        if type(seed) is not int or seed < 0:
            raise ValueError("warmup_seed must be a nonnegative integer")

        within = np.flatnonzero(store.same_p1)
        within_order = np.random.Generator(np.random.PCG64(seed)).permutation(within).tolist()
        categories = (["WARMUP_INTRA"] * count if policy == "intra_fixed"
                      else [SLOTS[pos % len(SLOTS)] for pos in range(count)])
        required = Counter(categories)
        if len(within) < required["WARMUP_INTRA"]:
            raise ValueError("insufficient within-P1 pairs for fixed warmup quota")
        orders = {"WARMUP_INTRA": within_order}
        if policy == "mixed_fixed":
            cross = np.flatnonzero(~store.same_p1)
            if cross.size < required["WARMUP_CROSS_NEAR"]:
                raise ValueError("insufficient cross-P1 pairs for fixed warmup quota")
            
            orders["WARMUP_CROSS_NEAR"] = cross[np.lexsort((cross, -store.dense[cross]))].tolist()
            orders["WARMUP_EXPLORATION"] = np.random.Generator(
                np.random.PCG64(seed + 10000)).permutation(store.total_pairs).tolist()

        positions = dict.fromkeys(orders, 0)
        selected = set()
        labels = labels_of(dataset.initial_partition, dataset.n)
        candidates = []
        rows = []
        for category in categories:
            order = orders[category]
            pos = positions[category]
            while pos < len(order) and order[pos] in selected:
                pos += 1
            if pos == len(order):
                
                
                raise ValueError(f"insufficient unselected pairs for fixed warmup quota: {category}")
            idx = int(order[pos])
            positions[category] = pos + 1
            selected.add(idx)
            a, b = store.pair_tuple(idx)
            cp = tuple(sorted((int(labels[a]), int(labels[b]))))
            candidates.append(QueryCandidate(idx, (a, b), category, float(store.dense[idx]),
                                              float(store.lexical[idx]), .5, 0., 0., cp))
            rows.append({"warmup_index": len(rows) + 1, "pair_index": idx,
                         "left": a, "right": b, "pair": [a, b],
                         "report_ids": [dataset.report_ids[a], dataset.report_ids[b]],
                         "category": category})
        sampling = {
            "policy": policy, "warmup_pairs": count, "warmup_seed": seed,
            "bit_generator": "PCG64", "numpy_version": np.__version__,
            "within_order": "PERMUTATION_OF_ASCENDING_SAME_P1_PAIR_INDICES",
            "cross_near_order": ("DENSE_DESCENDING_TIE_PAIR_INDEX_ASCENDING" if policy == "mixed_fixed" else None),
            "exploration_seed": seed + 10000 if policy == "mixed_fixed" else None,
            "exploration_order": "PERMUTATION_OF_FULL_PAIR_UNIVERSE" if policy == "mixed_fixed" else None,
            "slot_cycle": list(SLOTS) if policy == "mixed_fixed" else ["WARMUP_INTRA"],
            "partial_cycle_policy": "PREFIX_OF_SLOT_CYCLE_NO_ROUNDING",
            "deduplication": "SKIP_PREVIOUSLY_SELECTED_IN_EACH_STREAM_IN_SLOT_ORDER",
            "insufficient_stratum_policy": "REJECT_NO_QUOTA_REALLOCATION",
            "failure_policy": "CONSUME_LOGICAL_QUERY_NO_REPLACEMENT",
            "category_counts": dict(sorted(required.items())),
            "order_sha256": canonical_sha256(rows),
            "pair_index_order_sha256": canonical_sha256([row["pair_index"] for row in rows]),
        }
        document = {"schema": "aqcc-fixed-initial-warmup-plan-v1", "sampling": sampling,
                    "dataset_id": dataset.dataset_id, "source_sha256": dataset.source_sha256,
                    "report_ids": list(dataset.report_ids), "initial_partition": dataset.partition_ids(),
                    "queries": rows}
        return cls(policy, count, seed, tuple(candidates), document)

    def complete(self, ledger):
        return len(ledger.consumed()) >= self.count

    def next_candidates(self, consumed_count, batch_size):
        return list(self.candidates[consumed_count: min(self.count, consumed_count + batch_size)])

    def config(self):
        return {"initial_query_policy": self.policy, "warmup_pairs": self.count,
                "warmup_seed": self.seed, "initial_query_sampling": dict(
                    self.document["sampling"], plan_path="warmup_plan.json",
                    plan_sha256=canonical_sha256(self.document))}

    def completion_fields(self, query_count):
        return {"initial_query_policy": self.policy,
                "initial_query_complete": query_count >= self.count, "probing_complete": None}

    def validate_history(self, dataset, store, output_dir, query_rows, round_rows,
                         pending_plan, *, budget_pairs, microbatch_size, read_json, evidence_hash):
        """Validate every committed and pending plan before any resumed dispatch."""
        plans = []
        expected_names = {f"M{number:05d}.json" for number in range(1, len(round_rows) + 1)}
        if pending_plan is not None:
            expected_names.add(f"M{len(round_rows) + 1:05d}.json")
        if {path.name for path in (output_dir / "plans").glob("M*.json")} != expected_names:
            raise ValueError("fixed warmup run has missing or unexpected frozen plans")
        for number in range(1, len(round_rows) + 1):
            plans.append(read_json(output_dir / "plans" / f"M{number:05d}.json"))
        if pending_plan is not None:
            plans.append(pending_plan)
        before = 0
        sampled = set()
        for number, plan in enumerate(plans, 1):
            rows = plan.get("queries", [])
            if (plan.get("micro_round") != number or plan.get("queries_before") != before
                    or not rows or before + len(rows) > budget_pairs
                    or plan.get("evidence_before") != evidence_hash(query_rows[:before])):
                raise ValueError("fixed warmup frozen plan counts/evidence mismatch")
            is_warmup = before < self.count
            stage = STAGE if is_warmup else "ACTIVE_QUERY"
            if plan.get("stage") != stage:
                raise ValueError("fixed warmup frozen plan stage mismatch")
            if is_warmup and (len(rows) != min(microbatch_size, self.count - before, budget_pairs - before)
                              or plan.get("partition_before") != dataset.partition_ids()):
                raise ValueError("fixed warmup frozen plan boundary/partition mismatch")
            for offset, row in enumerate(rows):
                a, b = row["pair"]
                if (not 0 <= a < b < dataset.n or row.get("stage") != stage
                        or row.get("pair_index") != store.pair_index(a, b)
                        or row.get("report_ids") != [dataset.report_ids[a], dataset.report_ids[b]]
                        or (a, b) in sampled):
                    raise ValueError("fixed warmup frozen plan pair identity/repetition mismatch")
                sampled.add((a, b))
                if is_warmup:
                    expected = self.candidates[before + offset]
                    if (row["pair_index"] != expected.pair_index or row["category"] != expected.category
                            or row["dense"] != expected.dense or row["lexical"] != expected.lexical
                            or row["p_same_before"] != .5 or row["cheap_score"] != 0.
                            or row["expected_value"] != 0. or row["component_pair"] != list(expected.component_pair)):
                        raise ValueError("fixed warmup frozen plan differs from immutable schedule")
                elif str(row.get("category", "")).startswith("WARMUP_"):
                    raise ValueError("warmup category occurs after fixed schedule")
                index = before + offset
                if index < len(query_rows):
                    obs = query_rows[index]["observation"]
                    metadata = {key: value for key, value in row.items() if key != "pair"}
                    metadata["micro_round"] = number
                    if (obs.get("query_index") != index + 1 or (obs["left"], obs["right"]) != (a, b)
                            or obs.get("metadata") != metadata):
                        raise ValueError("fixed warmup ledger differs from frozen plan")
            before += len(rows)
            if number <= len(round_rows):
                saved = round_rows[number - 1]
                if (saved.get("logical_pairs_total") != before or saved.get("stage") != stage
                        or any(saved.get(key) != value for key, value in self.completion_fields(before).items())
                        or saved.get("evidence_sha256") != evidence_hash(query_rows[:before])):
                    raise ValueError("fixed warmup completed round differs from frozen plan")
                if before < self.count and saved.get("partition") != dataset.partition_ids():
                    raise ValueError("fixed warmup completed round failed to hold P1")
        if len(query_rows) > before:
            raise ValueError("fixed warmup ledger exceeds frozen plans")
