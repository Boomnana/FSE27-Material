from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .decoder import DecodeResult, decode, labels_of, weighted_loss
from .evidence import EvidenceLedger, Observation
from .features import PairFeatureStore
from .probing import RecursiveIntraBlockProber
from .provider import request_hash, request_payload, strict_json
from .selector import ActiveQuerySelector, QueryCandidate
from .surrogate import LLMSurrogate
from .tracing import AnalysisTracer
from .warmup import FixedInitialWarmup, STAGE as FIXED_WARMUP_STAGE
from .acquisition import DisabledSurrogate, control_config, control_fields
from .study_controls import FrozenPlanSelector, FrozenTrainingSurrogate


def _json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        f.flush()
        os.fsync(f.fileno())
    
    
    for attempt in range(7):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if os.name != "nt" or attempt == 6:
                raise
            time.sleep(0.01 * 2 ** attempt)


def _append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path, *, recover_torn_tail: bool = False) -> list[dict]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    complete = raw
    if raw and not raw.endswith(b"\n"):
        if not recover_torn_tail:
            raise ValueError(f"unterminated JSONL record at {path}")
        
        
        
        complete = raw[:raw.rfind(b"\n") + 1]
    out = []
    for number, line in enumerate(complete.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
    if len(complete) != len(raw):
        
        
        with path.open("r+b") as f:
            f.truncate(len(complete))
            f.flush()
            os.fsync(f.fileno())
    return out


def _partition_ids(dataset, partition):
    return [[dataset.report_ids[i] for i in block] for block in partition]


def _usage_add(total: dict, usage: dict):
    for src, dst in [("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens"),
                     ("total_tokens", "total_tokens"), ("input_tokens", "input_tokens"),
                     ("output_tokens", "output_tokens")]:
        value = usage.get(src)
        if isinstance(value, int) and value >= 0:
            total[dst] = total.get(dst, 0) + value


def _probabilities(store: PairFeatureStore, ledger: EvidenceLedger, surrogate: LLMSurrogate):
    edges = ledger.signed_edges()
    if isinstance(surrogate, DisabledSurrogate):
        summary = surrogate.summary
        upper = surrogate.unknown_scores(store)
    else:
        summary = surrogate.fit(edges)
        upper = surrogate.predict_unqueried(edges)
    for (a, b), sign in edges.items():
        upper[store.pair_index(a, b)] = 1.0 if sign == 1 else 0.0
    matrix = np.full((store.n, store.n), 0.5, dtype=np.float32)
    matrix[store.i, store.j] = upper
    matrix[store.j, store.i] = upper
    np.fill_diagonal(matrix, 0.5)
    return upper, matrix, edges, summary


def _decoder_graph(store, acquisition_upper, acquisition_matrix, edges, mode):
    """Neutralize only unknown decoder edges; acquisition still uses the model."""
    if mode == "surrogate":
        return acquisition_upper, acquisition_matrix
    upper = np.full(store.total_pairs, 0.5, dtype=np.float32)
    for (a, b), sign in edges.items():
        upper[store.pair_index(a, b)] = 1.0 if sign == 1 else 0.0
    matrix = np.full((store.n, store.n), 0.5, dtype=np.float32)
    matrix[store.i, store.j] = upper
    matrix[store.j, store.i] = upper
    return upper, matrix


class _FrozenScalerBinding:
    """Anchor the scaler at the completed probing batch, never a later ledger."""

    def __init__(self, output_dir, config, surrogate, ledger, prober, completed_count):
        self.output_dir = output_dir
        self.config_sha256 = _content_hash(config)
        self.surrogate = surrogate
        self.prober = prober
        self.path = output_dir / "frozen_scaler.json"
        self.manifest_path = output_dir / "manifest.json"
        self.reference = None
        manifest = strict_json(self.manifest_path.read_text(encoding="utf-8"))
        reference = manifest.get("frozen_scaler")
        if reference is not None and not self.path.is_file():
            raise ValueError("committed frozen scaler state is missing")
        if self.path.is_file():
            document = strict_json(self.path.read_text(encoding="utf-8"))
            if (document.get("schema") != "aqcc-frozen-scaler-binding-v1"
                    or document.get("config_sha256") != self.config_sha256
                    or document.get("state_sha256") != _content_hash(document.get("state"))):
                raise ValueError("frozen scaler state identity/hash mismatch")
            q = document.get("query_count")
            micro = document.get("micro_round")
            if not isinstance(q, int) or not 0 < q <= len(ledger.history()):
                raise ValueError("frozen scaler evidence count is invalid")
            history = ledger.history()
            prefix = EvidenceLedger.from_rows(ledger.n, history[:q])
            if (prefix.snapshot_sha256() != document.get("evidence_sha256")
                    or not prober.is_complete(prefix)
                    or any(r.get("metadata", {}).get("stage") != "RECURSIVE_INTRA_BLOCK_PROBING"
                           for r in history[:q])):
                raise ValueError("frozen scaler is not bound to the probing prefix")
            freeze_edges = prefix.signed_edges()
            expected_edges_hash = _content_hash({"signed_edges": [
                [a, b, sign] for (a, b), sign in sorted(freeze_edges.items())]})
            if (document["state"].get("freeze_edges_sha256") != expected_edges_hash
                    or document["state"].get("freeze_training_pairs") != len(freeze_edges)):
                raise ValueError("frozen scaler training edges differ from its probing prefix")
            plan_path = output_dir / "plans" / f"M{micro:05d}.json"
            if not plan_path.is_file():
                raise ValueError("frozen scaler probing plan is missing")
            plan = strict_json(plan_path.read_text(encoding="utf-8"))
            before = plan.get("queries_before")
            if (plan.get("stage") != "RECURSIVE_INTRA_BLOCK_PROBING"
                    or not isinstance(before, int) or before + len(plan.get("queries", [])) != q
                    or prober.is_complete(EvidenceLedger.from_rows(ledger.n, history[:before]))):
                raise ValueError("frozen scaler was not captured at the first completed probing batch")
            actual_reference = self._reference(document)
            if reference is not None and reference != actual_reference:
                raise ValueError("committed frozen scaler state hash mismatch")
            if reference is None and (q != len(history) or completed_count >= q):
                raise ValueError("unanchored frozen scaler cannot be adopted after a committed or later round")
            surrogate.restore_frozen_scaler_state(document["state"])
            self.reference = actual_reference
            if reference is None:
                manifest["frozen_scaler"] = actual_reference
                _json_dump(self.manifest_path, manifest)
        elif any(r.get("metadata", {}).get("stage") == "ACTIVE_QUERY" for r in ledger.history()):
            raise ValueError("late evidence cannot reconstruct a missing frozen scaler")
        elif completed_count and prober.is_complete(EvidenceLedger.from_rows(ledger.n, ledger.history()[:completed_count])):
            raise ValueError("completed probing checkpoint is missing its frozen scaler")

    def _reference(self, document):
        return {"path": "frozen_scaler.json", "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "query_count": document["query_count"], "micro_round": document["micro_round"],
                "evidence_sha256": document["evidence_sha256"], "state_sha256": document["state_sha256"]}

    def ensure(self, ledger, micro_round, plan):
        if self.reference is not None:
            return True
        if not self.prober.is_complete(ledger):
            return False
        
        
        if plan is not None and plan["queries_before"] + len(plan["queries"]) != len(ledger.history()):
            return False
        if (plan is None or plan.get("stage") != "RECURSIVE_INTRA_BLOCK_PROBING"
                or not ledger.signed_edges()):
            raise ValueError("freeze_after_probing requires a completed probing batch with successful evidence")
        state = self.surrogate.freeze_scaler(ledger.signed_edges())
        document = {"schema": "aqcc-frozen-scaler-binding-v1", "config_sha256": self.config_sha256,
                    "query_count": len(ledger.history()), "micro_round": micro_round,
                    "evidence_sha256": ledger.snapshot_sha256(), "state": state,
                    "state_sha256": _content_hash(state)}
        _json_dump(self.path, document)
        self.reference = self._reference(document)
        manifest = strict_json(self.manifest_path.read_text(encoding="utf-8"))
        manifest["frozen_scaler"] = self.reference
        _json_dump(self.manifest_path, manifest)
        return True


def _content_hash(obj) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _run_lock(path: Path):
    """An OS-released lock also prevents two resume processes dispatching a pair."""
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("another process is using this run directory") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("another process is using this run directory") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _query_parallel(dataset, judge, planned: list[QueryCandidate], concurrency: int,
                    *, output_dir: Path, queries_before: int, system_prompt: str):
    """Persist a claim before sending and each answer before returning it.

    A recovered STARTED record is uncertain, so consume it as a failure without
    a second semantic sample. An absent claim was never dispatched and is safe
    to submit. Canonical plan order is retained regardless of completion order.
    """
    results = [None] * len(planned)

    def execute(row, result_path, record):
        a, b = row.pair
        try:
            result = judge(dataset.report_ids[a], dataset.texts[a],
                           dataset.report_ids[b], dataset.texts[b])
        except Exception as exc:
            if isinstance(exc, getattr(judge, "fatal_exception_types", ())):
                raise
            result = {"status": "FAILED_JUDGE_EXCEPTION", "label": None, "reason": "",
                      "request_sha256": record["request_sha256"], "response_sha256": None,
                      "attempts": [{"kind": "EXCEPTION", "error_class": type(exc).__name__}],
                      "usage": {}}
        _json_dump(result_path, dict(record, state="COMPLETED", result=result,
                              result_sha256=_content_hash(result)))
        return result

    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(planned)))) as pool:
        futures = {}
        for pos, row in enumerate(planned):
            a, b = row.pair
            query_number = queries_before + pos + 1
            path = output_dir / "query_journal" / f"Q{query_number:08d}.json"
            result_path = path.with_suffix(".result.json")
            record = {
                "schema": "aqcc-final-query-journal-v1", "query_index": query_number,
                "pair": [a, b], "report_ids": [dataset.report_ids[a], dataset.report_ids[b]],
                "request_sha256": request_hash(system_prompt, request_payload(
                    dataset.report_ids[a], dataset.texts[a], dataset.report_ids[b], dataset.texts[b])),
            }
            if result_path.exists() and not path.exists():
                raise ValueError("query journal response is missing its dispatch claim")
            if path.exists():
                claim = strict_json(path.read_text(encoding="utf-8"))
                if (any(claim.get(key) != value for key, value in record.items())
                        or claim.get("state") != "STARTED"):
                    raise ValueError("query journal claim differs from the frozen plan")
                saved = strict_json(result_path.read_text(encoding="utf-8")) if result_path.exists() else claim
                if any(saved.get(key) != value for key, value in record.items()):
                    raise ValueError("query journal identity differs from the frozen plan")
                if saved.get("state") == "COMPLETED":
                    if saved.get("result_sha256") != _content_hash(saved.get("result")):
                        raise ValueError("query journal result hash mismatch")
                    results[pos] = saved["result"]
                    continue
                if saved.get("state") != "STARTED":
                    raise ValueError("invalid query journal state")
                recover = getattr(judge, "recover_cached_result", None)
                if callable(recover):
                    result = recover(dataset.report_ids[a], dataset.texts[a],
                                     dataset.report_ids[b], dataset.texts[b])
                    if not isinstance(result, dict) or result.get("request_sha256") != record["request_sha256"]:
                        raise ValueError("recovered shared result differs from the claimed request")
                else:
                    result = {
                        "status": "FAILED_INTERRUPTED_UNCERTAIN", "label": None, "reason": "",
                        "request_sha256": record["request_sha256"], "response_sha256": None,
                        "attempts": [{"kind": "RECOVERED_UNCERTAIN", "semantic_resample": False}],
                        "usage": {},
                    }
                _json_dump(result_path, dict(record, state="COMPLETED", result=result,
                                      result_sha256=_content_hash(result)))
                results[pos] = result
                continue
            _json_dump(path, dict(record, state="STARTED"))
            fut = pool.submit(execute, row, result_path, record)
            futures[fut] = pos
        for fut in as_completed(futures):
            
            results[futures[fut]] = fut.result()
    return results


def _run_impl(dataset, judge, *, output_dir: Path, system_prompt: str,
        strategy: str = "adaptive", budget_pairs: int = 1000,
        microbatch_size: int = 12, concurrency: int = 12,
        recursive_probing: bool = True, candidate_pool: int = 96,
        lookahead: int = 0, exploration_fraction: float = 0.20,
        frontier_base: int = 256, frontier_growth: float = 8.0,
        member_neighbor_k: int = 12, max_node_degree_per_batch: int = 3,
        max_component_pair_per_batch: int = 2,
        decoder_max_steps: int = 120, surrogate_min_samples: int = 24,
        random_seed: int = 0, detailed_trace: bool = False,
        random_with_probing: bool = False, decoder_graph_mode: str = "surrogate",
        surrogate_feature_policy: str = "all", surrogate_scaler_policy: str = "refit",
        initial_query_policy: str = "recursive", warmup_pairs: int = 512, warmup_seed: int = 0,
        acquisition_mode: str = "surrogate",
        query_feedback: str = "adaptive", surrogate_update: str = "adaptive",
        feature_cache: Path | None = None, resume: bool = False,
        execution_mode: str = "LIVE") -> dict:
    total_pairs = dataset.n * (dataset.n - 1) // 2
    if query_feedback not in {"adaptive", "frozen_plan"} or surrogate_update not in {"adaptive", "freeze_first_fitted"}:
        raise ValueError("Unknown study intervention")
    if (query_feedback != "adaptive" or surrogate_update != "adaptive") and (
            strategy != "adaptive" or acquisition_mode != "surrogate" or decoder_graph_mode != "neutral"
            or not recursive_probing or initial_query_policy != "recursive" or surrogate_scaler_policy != "refit"):
        raise ValueError("Study interventions require adaptive LR, recursive probing and neutral decoder")
    if query_feedback != "adaptive" and surrogate_update != "adaptive":
        raise ValueError("Study interventions must be isolated")
    if not 1 <= budget_pairs <= total_pairs:
        raise ValueError("budget_pairs outside population pair range")
    if microbatch_size < 1 or concurrency < 1:
        raise ValueError("invalid microbatch/concurrency")
    if strategy not in {"adaptive", "dense", "multiview", "random", "exhaustive"}:
        raise ValueError("invalid strategy")
    if decoder_graph_mode not in {"surrogate", "neutral"}:
        raise ValueError("invalid decoder_graph_mode")
    if surrogate_feature_policy not in {"all", "drop_same_degree_min"}:
        raise ValueError("invalid surrogate_feature_policy")
    if surrogate_scaler_policy not in {"refit", "freeze_after_probing"}:
        raise ValueError("invalid surrogate_scaler_policy")
    if acquisition_mode not in {"surrogate", "constant", "text_cosine"}:
        raise ValueError("invalid acquisition_mode")
    controlled_acquisition = acquisition_mode != "surrogate"
    if controlled_acquisition and (
            strategy != "adaptive" or decoder_graph_mode != "neutral" or lookahead != 0
            or surrogate_scaler_policy != "refit" or not recursive_probing
            or initial_query_policy != "recursive"):
        raise ValueError("nonlearning acquisition requires adaptive, neutral decoder, lookahead=0, scaler refit, "
                         "and recursive probing")
    if initial_query_policy not in {"recursive", "intra_fixed", "mixed_fixed"}:
        raise ValueError("invalid initial_query_policy")
    if initial_query_policy != "recursive" and (
            strategy != "adaptive" or not recursive_probing or surrogate_scaler_policy != "refit"):
        raise ValueError("fixed initial query policy requires adaptive, recursive_probing=True, and scaler refit")
    if initial_query_policy == "recursive" and (warmup_pairs != 512 or warmup_seed != 0):
        raise ValueError("warmup_pairs/warmup_seed require a fixed initial query policy")
    if random_with_probing and (strategy != "random" or not recursive_probing):
        raise ValueError("random_with_probing requires random strategy and recursive probing")
    if strategy == "random" and not random_with_probing:
        
        
        recursive_probing = False
    if surrogate_scaler_policy == "freeze_after_probing" and (not recursive_probing or strategy == "exhaustive"):
        raise ValueError("freeze_after_probing requires recursive probing")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "pair_queries.jsonl"
    rounds_path = output_dir / "micro_rounds.jsonl"
    manifest_path = output_dir / "manifest.json"

    feature_store = PairFeatureStore.build(dataset, cache_path=feature_cache)
    warmup = (FixedInitialWarmup.build(dataset, feature_store, policy=initial_query_policy,
                                      count=warmup_pairs, seed=warmup_seed)
              if initial_query_policy != "recursive" else None)
    selector = ActiveQuerySelector(
        feature_store, dataset.initial_partition, candidate_pool=candidate_pool,
        lookahead=lookahead, exploration_fraction=exploration_fraction,
        frontier_base=frontier_base, frontier_growth=frontier_growth,
        max_node_degree_per_batch=max_node_degree_per_batch,
        max_component_pair_per_batch=max_component_pair_per_batch,
        member_neighbor_k=member_neighbor_k, decoder_max_steps=decoder_max_steps,
        random_seed=random_seed if strategy == "random" else None,
        **({"capture_trace": True} if detailed_trace else {}),
    )
    config = {
        "schema": "aqcc-final-config-v3",
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(Path(__file__).parent.glob("*.py"))},
        "decoder_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("decoder.py").read_bytes()).hexdigest(),
        "request_persistence_contract": "DURABLE_PAIR_JOURNAL_V1",
        "dataset_id": dataset.dataset_id,
        "source_path": dataset.source_path,
        "source_sha256": dataset.source_sha256,
        "reports": dataset.n,
        "initial_clusters": len(dataset.initial_partition),
        "total_pairs": total_pairs,
        "strategy": strategy,
        "study_query_feedback": query_feedback,
        "study_surrogate_update": surrogate_update,
        "budget_pairs": budget_pairs,
        "microbatch_size": microbatch_size,
        "concurrency": concurrency,
        "recursive_probing": bool(recursive_probing),
        "candidate_pool": candidate_pool,
        "lookahead": lookahead,
        "exploration_fraction": exploration_fraction,
        "frontier_base": frontier_base,
        "frontier_growth": frontier_growth,
        "member_neighbor_k": member_neighbor_k,
        "max_node_degree_per_batch": max_node_degree_per_batch,
        "max_component_pair_per_batch": max_component_pair_per_batch,
        "decoder_max_steps": decoder_max_steps,
        "surrogate_min_samples": surrogate_min_samples,
        "random_with_probing": bool(random_with_probing),
        "decoder_graph_mode": decoder_graph_mode,
        "surrogate_feature_policy": surrogate_feature_policy,
        "surrogate_scaler_policy": surrogate_scaler_policy,
        "execution_mode": execution_mode,
        "judge_identity": getattr(judge, "identity", {"kind": type(judge).__name__}),
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "pair_oracle_contract": "ATOMIC_UNORDERED_BINARY_PAIR_ONE_SEMANTIC_SAMPLE",
        "selector_budget_independent": True,
        "hard_evidence_only_from_direct_atomic_queries": True,
        "online_gold_access": False,
        "offline_gold_oracle_simulation": execution_mode == "OFFLINE_GOLD_ORACLE",
        "decoder_global_optimum_claimed": False,
    }
    if detailed_trace:
        config["detailed_trace"] = True
    if warmup is not None:
        config.update(warmup.config())
    if controlled_acquisition:
        config.update(control_config(acquisition_mode))
    if strategy == "random":
        config["random_sampling"] = {
            "seed": random_seed, "bit_generator": "PCG64", "numpy_version": np.__version__,
            "policy": ("UNIFORM_REMAINING_PAIRS_AFTER_RECURSIVE_PROBING" if random_with_probing
                       else "UNIFORM_WITHOUT_REPLACEMENT_FULL_PAIR_UNIVERSE"),
            "order_sha256": _content_hash(selector.random_order.tolist()),
        }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config") != config:
            raise ValueError("resume configuration differs from existing manifest")
    else:
        _json_dump(manifest_path, {
            "schema": "aqcc-final-manifest-v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "feature_metadata": feature_store.metadata,
        })
    if warmup is not None:
        warmup_path = output_dir / "warmup_plan.json"
        if warmup_path.is_file():
            if strict_json(warmup_path.read_text(encoding="utf-8")) != warmup.document:
                raise ValueError("immutable fixed warmup plan differs from configured schedule")
        elif resume:
            raise ValueError("immutable fixed warmup plan is missing")
        else:
            _json_dump(warmup_path, warmup.document)

    ledger = EvidenceLedger(dataset.n)
    existing_rows = _read_jsonl(queries_path, recover_torn_tail=resume)
    for number, row in enumerate(existing_rows, 1):
        if row["observation"]["query_index"] != number:
            raise ValueError("query log indices are not contiguous")
        ledger.add(Observation(**row["observation"]))
    query_index = len(existing_rows)
    round_rows = _read_jsonl(rounds_path, recover_torn_tail=resume)
    micro_round = len(round_rows)
    if any(row.get("micro_round") != number for number, row in enumerate(round_rows, 1)):
        raise ValueError("micro-round indices are not contiguous")
    completed_count = round_rows[-1]["logical_pairs_total"] if round_rows else 0
    if not 0 <= completed_count <= query_index <= budget_pairs:
        raise ValueError("checkpoint query count is inconsistent")
    if round_rows:
        completed_ledger = EvidenceLedger.from_rows(
            dataset.n, [row["observation"] for row in existing_rows[:completed_count]])
        if (round_rows[-1].get("evidence_sha256") != completed_ledger.snapshot_sha256()
                or round_rows[-1].get("evidence_counts") != completed_ledger.counts()):
            raise ValueError("completed checkpoint evidence differs from query log")
    pending_path = output_dir / "plans" / f"M{micro_round + 1:05d}.json"
    pending_plan = strict_json(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else None
    if pending_plan is not None:
        before = pending_plan["queries_before"]
        if (pending_plan["micro_round"] != micro_round + 1 or before != completed_count
                or not before <= query_index <= before + len(pending_plan["queries"]) <= budget_pairs):
            raise ValueError("pending plan count differs from query log")
        prefix = EvidenceLedger.from_rows(dataset.n, [r["observation"] for r in existing_rows[:before]])
        if prefix.snapshot_sha256() != pending_plan["evidence_before"]:
            raise ValueError("pending plan evidence prefix differs from query log")
        for row in pending_plan["queries"]:
            a, b = row["pair"]
            if (not 0 <= a < b < dataset.n or row["pair_index"] != feature_store.pair_index(a, b)
                    or row["report_ids"] != [dataset.report_ids[a], dataset.report_ids[b]]):
                raise ValueError("pending plan pair identity differs from dataset")
        planned_pairs = [tuple(row["pair"]) for row in pending_plan["queries"]]
        if len(set(planned_pairs)) != len(planned_pairs) or set(planned_pairs) & prefix.consumed():
            raise ValueError("pending plan contains repeated pairs")
        for pos, row in enumerate(existing_rows[before:]):
            obs = row["observation"]
            if (obs["left"], obs["right"]) != planned_pairs[pos]:
                raise ValueError("committed query differs from pending plan")
    elif completed_count != query_index:
        raise ValueError("partial query batch is missing its frozen plan")
    if warmup is not None:
        warmup.validate_history(
            dataset, feature_store, output_dir, existing_rows, round_rows, pending_plan,
            budget_pairs=budget_pairs, microbatch_size=microbatch_size,
            read_json=lambda path: strict_json(path.read_text(encoding="utf-8")),
            evidence_hash=lambda rows: EvidenceLedger.from_rows(
                dataset.n, [row["observation"] for row in rows]).snapshot_sha256())
    usage_total: dict[str, int] = {}
    for row in existing_rows:
        _usage_add(usage_total, row.get("judge_result", {}).get("usage", {}))

    tracer = None
    initial_trace_reference = None
    terminal_trace_reference = None
    if detailed_trace:
        tracer = AnalysisTracer(output_dir, dataset, feature_store, config, json_writer=_json_dump)
        manifest_doc = strict_json(manifest_path.read_text(encoding="utf-8"))
        initial_trace_reference = manifest_doc.get("initial_analysis_trace")
        if initial_trace_reference is None:
            if (existing_rows or round_rows or any((output_dir / "query_journal").glob("Q*.json"))):
                raise ValueError("started traced run is missing its committed initial analysis trace")
        else:
            tracer.verify(initial_trace_reference, snapshot_id="initial", query_count=0,
                          evidence_sha256=EvidenceLedger(dataset.n).snapshot_sha256(), usage={})
        for row in round_rows:
            tracer.verify(row.get("analysis_trace"), snapshot_id=f"M{row['micro_round']:05d}",
                          query_count=row["logical_pairs_total"], evidence_sha256=row["evidence_sha256"],
                          partition=row["partition"], usage=row["usage"])
        terminal_path = output_dir / "terminal.json"
        if terminal_path.is_file():
            old_terminal = strict_json(terminal_path.read_text(encoding="utf-8"))
            terminal_trace_reference = old_terminal.get("analysis_trace")
            tracer.verify(terminal_trace_reference, snapshot_id="terminal", query_count=query_index,
                          evidence_sha256=ledger.snapshot_sha256(),
                          partition=old_terminal["partitions"]["AQCC_FINAL"], usage=usage_total)

    initial_state_started = time.monotonic() if detailed_trace else None
    if query_feedback == "frozen_plan":
        selector = FrozenPlanSelector(selector, output_dir, config, ledger, _json_dump)
    surrogate = (DisabledSurrogate(acquisition_mode, feature_policy=surrogate_feature_policy)
                 if controlled_acquisition else LLMSurrogate(
                     feature_store, min_samples=surrogate_min_samples,
                     feature_policy=surrogate_feature_policy, scaler_policy=surrogate_scaler_policy))
    if surrogate_update == "freeze_first_fitted":
        surrogate = FrozenTrainingSurrogate(feature_store, min_samples=surrogate_min_samples,
                     feature_policy=surrogate_feature_policy, scaler_policy=surrogate_scaler_policy,
                     directory=output_dir, config=config, ledger=ledger, writer=_json_dump)
    prober = RecursiveIntraBlockProber(dataset.initial_partition, feature_store.dense_matrix)
    scaler_binding = (_FrozenScalerBinding(output_dir, config, surrogate, ledger, prober, completed_count)
                      if surrogate_scaler_policy == "freeze_after_probing" else None)
    if scaler_binding is not None:
        reference = scaler_binding.reference
        for row in round_rows:
            expected = reference if reference is not None and row["logical_pairs_total"] >= reference["query_count"] else None
            if row.get("frozen_scaler") != expected:
                raise ValueError("completed round frozen scaler binding differs from manifest")
        saved_terminal_path = output_dir / "terminal.json"
        if saved_terminal_path.is_file():
            saved_terminal = strict_json(saved_terminal_path.read_text(encoding="utf-8"))
            if saved_terminal.get("frozen_scaler") != reference:
                raise ValueError("terminal frozen scaler binding differs from manifest")

    probing_pending = (not warmup.complete(ledger) if warmup is not None else
                       bool(recursive_probing and strategy != "exhaustive" and not prober.is_complete(ledger)))
    if scaler_binding is not None and not probing_pending:
        probing_pending = not scaler_binding.ensure(ledger, micro_round + 1, pending_plan)
    if probing_pending or strategy == "exhaustive":
        edges = ledger.signed_edges()
        upper = np.full(feature_store.total_pairs, 0.5, dtype=np.float32)
        for (a,b), sign in edges.items():
            upper[feature_store.pair_index(a,b)] = 1.0 if sign == 1 else 0.0
        prob_matrix = np.full((dataset.n,dataset.n),0.5,dtype=np.float32)
        prob_matrix[feature_store.i,feature_store.j]=upper; prob_matrix[feature_store.j,feature_store.i]=upper
        decoder_upper, decoder_matrix = _decoder_graph(feature_store, upper, prob_matrix, edges, decoder_graph_mode)
        surrogate_summary = surrogate.summary
        current = DecodeResult(dataset.initial_partition, weighted_loss(dataset.initial_partition, decoder_matrix), 0, tuple())
    else:
        upper, prob_matrix, edges, surrogate_summary = _probabilities(feature_store, ledger, surrogate)
        decoder_upper, decoder_matrix = _decoder_graph(feature_store, upper, prob_matrix, edges, decoder_graph_mode)
        current = decode(dataset.initial_partition, decoder_matrix, direct_edges=edges,
                         max_steps=decoder_max_steps)
    if tracer is not None and initial_trace_reference is None:
        initial_trace_reference = tracer.save(
            snapshot_id="initial", kind="INITIAL", micro_round=0, stage="INITIAL",
            ledger=ledger, upper=decoder_upper, acquisition_upper=upper, current=current, surrogate=surrogate,
            decoder_called=not probing_pending and strategy != "exhaustive",
            probing_held=probing_pending,
            probability_mode="NEUTRAL_UNQUERIED_DIRECT_OVERWRITE" if decoder_graph_mode == "neutral" or probing_pending or strategy == "exhaustive"
                             else "SURROGATE_WITH_DIRECT_OVERWRITE",
            usage=usage_total, phase_timings={"state_computation": time.monotonic() - initial_state_started})
        manifest_doc["initial_analysis_trace"] = initial_trace_reference
        _json_dump(manifest_path, manifest_doc)
    started = time.monotonic()
    stop_reason = None

    while len(ledger.consumed()) < budget_pairs or pending_plan is not None:
        planning_started = time.monotonic() if detailed_trace else None
        selector_called = False
        remaining = budget_pairs - len(ledger.consumed())
        
        
        
        plan_size = microbatch_size
        consumed = ledger.consumed()
        planned: list[QueryCandidate] = []
        stage = "ACTIVE_QUERY"

        if pending_plan is not None:
            stage = pending_plan["stage"]
            for row in pending_plan["queries"]:
                planned.append(QueryCandidate(
                    row["pair_index"], tuple(row["pair"]), row["category"], row["dense"],
                    row["lexical"], row["p_same_before"], row["cheap_score"],
                    row["expected_value"], tuple(row["component_pair"])))
        elif warmup is not None:
            planned = warmup.next_candidates(len(consumed), plan_size)
            if planned:
                stage = FIXED_WARMUP_STAGE
        elif strategy != "exhaustive" and recursive_probing:
            probes = prober.next_pairs(ledger, plan_size)
            if probes:
                stage = "RECURSIVE_INTRA_BLOCK_PROBING"
                labels = labels_of(current.partition, dataset.n)
                for pr in probes:
                    a, b = pr.pair
                    idx = feature_store.pair_index(a, b)
                    cp = tuple(sorted((int(labels[a]), int(labels[b]))))
                    planned.append(QueryCandidate(
                        idx, (a, b), "RECURSIVE_PROBE", float(feature_store.dense[idx]),
                        float(feature_store.lexical[idx]), float(upper[idx]), 0.0, 0.0, cp,
                    ))
        if not planned:
            selector_called = True
            planned = selector.choose(
                strategy=strategy, current_partition=current.partition,
                probabilities=upper, probability_matrix=prob_matrix,
                direct_edges=edges, consumed=consumed, batch_size=plan_size,
                queries_done=len(consumed),
            )
        if not planned:
            stop_reason = "NO_ELIGIBLE_UNQUERIED_PAIR"
            break
        if pending_plan is None:
            planned = planned[:remaining]

        micro_round += 1
        plan_rows = []
        for row in planned:
            a, b = row.pair
            plan_rows.append({
                "pair_index": row.pair_index,
                "pair": [a, b],
                "report_ids": [dataset.report_ids[a], dataset.report_ids[b]],
                "category": row.category,
                "stage": stage,
                "dense": row.dense,
                "lexical": row.lexical,
                "p_same_before": row.p_same,
                "cheap_score": row.cheap_score,
                "expected_value": row.expected_value,
                "component_pair": list(row.component_pair),
            })
        plan_document = {
            "micro_round": micro_round,
            "stage": stage,
            "queries_before": len(consumed),
            "evidence_before": ledger.snapshot_sha256(),
            "partition_before": _partition_ids(dataset, current.partition),
            "surrogate_before": asdict(surrogate_summary),
            "queries": plan_rows,
        }
        if tracer is not None and pending_plan is None:
            plan_document["selector_trace"] = dict(selector.last_trace) if selector_called else {
                "schema": "aqcc-selector-trace-v1", "source": stage}
            plan_document["selector_trace"].update(
                selector_called=selector_called,
                executed_pair_indices=[row.pair_index for row in planned])
        if pending_plan is None:
            _json_dump(output_dir / f"plans/M{micro_round:05d}.json", plan_document)
        else:
            plan_document = pending_plan
        queries_before = plan_document["queries_before"]

        query_started = time.monotonic() if detailed_trace else None
        results = _query_parallel(dataset, judge, planned, concurrency,
                                  output_dir=output_dir, queries_before=queries_before,
                                  system_prompt=system_prompt)
        ledger_started = time.monotonic() if detailed_trace else None
        for pos, (row, result) in enumerate(zip(planned, results)):
            expected_index = queries_before + pos + 1
            a, b = row.pair
            if expected_index <= query_index:
                old = existing_rows[expected_index - 1]
                if old["judge_result"] != result:
                    raise ValueError("persisted answer differs from committed query log")
                continue
            if expected_index != query_index + 1:
                raise ValueError("query index differs from pending plan")
            query_index = expected_index
            status = result.get("status", "FAILED_UNKNOWN")
            label = result.get("label") if status == "SUCCESS" else None
            reason = result.get("reason", "") if status == "SUCCESS" else ""
            obs = Observation(
                query_index=query_index, left=a, right=b, label=label, reason=reason,
                status=status, request_sha256=result.get("request_sha256", ""),
                response_sha256=result.get("response_sha256"),
                metadata={
                    "micro_round": micro_round,
                    "stage": stage,
                    "category": row.category,
                    "pair_index": row.pair_index,
                    "dense": row.dense,
                    "lexical": row.lexical,
                    "p_same_before": row.p_same,
                    "cheap_score": row.cheap_score,
                    "expected_value": row.expected_value,
                    "component_pair": list(row.component_pair),
                    "report_ids": [dataset.report_ids[a], dataset.report_ids[b]],
                },
            )
            ledger.add(obs)
            _usage_add(usage_total, result.get("usage", {}))
            _append_jsonl(queries_path, {"observation": obs.as_dict(), "judge_result": result})
        pending_plan = None

        state_started = time.monotonic() if detailed_trace else None
        probing_pending = (not warmup.complete(ledger) if warmup is not None else
                           bool(recursive_probing and strategy != "exhaustive" and not prober.is_complete(ledger)))
        if scaler_binding is not None and not probing_pending:
            scaler_binding.ensure(ledger, micro_round, plan_document)
        if strategy == "exhaustive":
            
            
            edges = ledger.signed_edges()
            upper = np.full(feature_store.total_pairs, 0.5, dtype=np.float32)
            for (a,b), sign in edges.items():
                upper[feature_store.pair_index(a,b)] = 1.0 if sign == 1 else 0.0
            prob_matrix = np.full((dataset.n,dataset.n),0.5,dtype=np.float32)
            prob_matrix[feature_store.i,feature_store.j]=upper; prob_matrix[feature_store.j,feature_store.i]=upper
            decoder_upper, decoder_matrix = _decoder_graph(feature_store, upper, prob_matrix, edges, decoder_graph_mode)
            surrogate_summary = surrogate.summary
        elif probing_pending:
            edges = ledger.signed_edges()
            upper = np.full(feature_store.total_pairs, 0.5, dtype=np.float32)
            for (a,b), sign in edges.items():
                upper[feature_store.pair_index(a,b)] = 1.0 if sign == 1 else 0.0
            prob_matrix = np.full((dataset.n,dataset.n),0.5,dtype=np.float32)
            prob_matrix[feature_store.i,feature_store.j]=upper; prob_matrix[feature_store.j,feature_store.i]=upper
            decoder_upper, decoder_matrix = _decoder_graph(feature_store, upper, prob_matrix, edges, decoder_graph_mode)
            surrogate_summary = surrogate.summary
            current = DecodeResult(dataset.initial_partition, weighted_loss(dataset.initial_partition, decoder_matrix), 0, tuple())
        else:
            upper, prob_matrix, edges, surrogate_summary = _probabilities(feature_store, ledger, surrogate)
            decoder_upper, decoder_matrix = _decoder_graph(feature_store, upper, prob_matrix, edges, decoder_graph_mode)
            current = decode(dataset.initial_partition, decoder_matrix, direct_edges=edges,
                             max_steps=decoder_max_steps)
        round_row = {
            "micro_round": micro_round,
            "stage": stage,
            "logical_pairs_total": len(ledger.consumed()),
            "pair_query_ratio": len(ledger.consumed()) / total_pairs,
            "evidence_counts": ledger.counts(),
            "category_counts": ledger.category_counts(),
            "active_direct_edges": len(edges),
            "evidence_sha256": ledger.snapshot_sha256(),
            "surrogate": asdict(surrogate_summary),
            "decoder_loss": None if strategy == "exhaustive" else current.loss,
            "decoder_steps": None if strategy == "exhaustive" else current.steps,
            "clusters": None if strategy == "exhaustive" else len(current.partition),
            "probing_complete": prober.is_complete(ledger) if warmup is None and recursive_probing and strategy != "exhaustive" else None,
            "partition": _partition_ids(dataset, current.partition),
            "usage": dict(usage_total),
        }
        if warmup is not None:
            round_row.update(warmup.completion_fields(len(ledger.consumed())))
        if controlled_acquisition:
            round_row.update(control_fields(acquisition_mode))
        if scaler_binding is not None:
            round_row["frozen_scaler"] = scaler_binding.reference
        if tracer is not None:
            round_row["analysis_trace"] = tracer.save(
                snapshot_id=f"M{micro_round:05d}", kind="ROUND", micro_round=micro_round, stage=stage,
                ledger=ledger, upper=decoder_upper, acquisition_upper=upper, current=current, surrogate=surrogate,
                evidence_sha256=round_row["evidence_sha256"],
                decoder_called=not probing_pending and strategy != "exhaustive",
                probing_held=probing_pending,
                probability_mode="NEUTRAL_UNQUERIED_DIRECT_OVERWRITE" if decoder_graph_mode == "neutral" or probing_pending or strategy == "exhaustive"
                                 else "SURROGATE_WITH_DIRECT_OVERWRITE",
                usage=usage_total, phase_timings={
                    "planning_and_plan_persistence": query_started - planning_started,
                    "query_collection_and_journal_persistence": ledger_started - query_started,
                    "ledger_commit": state_started - ledger_started,
                    "state_computation": time.monotonic() - state_started})
        _append_jsonl(rounds_path, round_row)
        _json_dump(output_dir / "checkpoint.json", round_row)
    else:
        stop_reason = "PAIR_BUDGET_REACHED"

    if len(ledger.consumed()) >= total_pairs:
        stop_reason = "EXHAUSTIVE_PAIR_COVERAGE"

    
    
    terminal_state_started = time.monotonic() if detailed_trace else None
    probing_pending = (not warmup.complete(ledger) if warmup is not None else
                       bool(recursive_probing and strategy != "exhaustive" and not prober.is_complete(ledger)))
    if (scaler_binding is not None or warmup is not None or controlled_acquisition
            or surrogate_update == "freeze_first_fitted") and probing_pending:
        
        
        edges = ledger.signed_edges()
        upper = np.full(feature_store.total_pairs, 0.5, dtype=np.float32)
        for (a,b), sign in edges.items():
            upper[feature_store.pair_index(a,b)] = 1.0 if sign == 1 else 0.0
        prob_matrix = np.full((dataset.n,dataset.n),0.5,dtype=np.float32)
        prob_matrix[feature_store.i,feature_store.j]=upper; prob_matrix[feature_store.j,feature_store.i]=upper
        surrogate_summary = surrogate.summary
    else:
        upper, prob_matrix, edges, surrogate_summary = _probabilities(feature_store, ledger, surrogate)
    decoder_upper, decoder_matrix = _decoder_graph(feature_store, upper, prob_matrix, edges, decoder_graph_mode)
    if probing_pending:
        current = DecodeResult(dataset.initial_partition, weighted_loss(dataset.initial_partition, decoder_matrix), 0, tuple())
    else:
        current = decode(dataset.initial_partition, decoder_matrix, direct_edges=edges,
                         max_steps=decoder_max_steps)
    failures = ledger.counts()["FAILED"]
    reference_eligible = len(ledger.consumed()) == total_pairs and failures == 0
    terminal_evidence_sha256 = ledger.snapshot_sha256()

    np.savez_compressed(
        output_dir / "final_pair_probabilities.npz",
        p_same=decoder_upper.astype(np.float32), acquisition_p_same=upper.astype(np.float32),
        decoder_graph_mode=np.asarray(decoder_graph_mode),
        pair_i=feature_store.i, pair_j=feature_store.j,
        report_ids=np.asarray(dataset.report_ids),
        evidence_sha256=np.asarray(terminal_evidence_sha256),
        **({"acquisition_mode": np.asarray(acquisition_mode),
            "acquisition_score_semantics": np.asarray(control_fields(acquisition_mode)["acquisition_score_semantics"])}
           if controlled_acquisition else {}),
    )

    terminal = {
        "schema": "aqcc-final-terminal-v3",
        "config_sha256": _content_hash(config),
        "status": "VALID_AQCC_FINAL_TERMINAL",
        "dataset_id": dataset.dataset_id,
        "execution_mode": execution_mode,
        "strategy": strategy,
        "stop_reason": stop_reason,
        "report_ids": list(dataset.report_ids),
        "partitions": {
            "P1": dataset.partition_ids(),
            "AQCC_FINAL": _partition_ids(dataset, current.partition),
        },
        "evidence_sha256": terminal_evidence_sha256,
        "evidence_counts": ledger.counts(),
        "category_counts": ledger.category_counts(),
        "logical_pairs_queried": len(ledger.consumed()),
        "total_possible_pairs": total_pairs,
        "pair_query_ratio": len(ledger.consumed()) / total_pairs,
        "probing_complete": prober.is_complete(ledger) if warmup is None and recursive_probing and strategy != "exhaustive" else None,
        "surrogate": asdict(surrogate_summary),
        "decoder": {
            "loss": current.loss,
            "steps": current.steps,
            "restart_seed": "P1_EVERY_DECODE",
            "global_optimum_claimed": False,
            "objective": "EXPECTED_PAIRWISE_DISAGREEMENT_ON_COMPLETE_SOFT_LLM_GRAPH",
            "graph_mode": decoder_graph_mode,
        },
        "reference_eligible": reference_eligible,
        "usage": usage_total,
        "elapsed_seconds_this_session": time.monotonic() - started,
        "gold_accessed_online": False,
        "offline_gold_oracle_simulation": execution_mode == "OFFLINE_GOLD_ORACLE",
        "interpretation": (
            "Direct atomic LLM answers are hard evidence; unqueried pair probabilities are predicted by a Gold-free "
            "surrogate and globally decoded. Provisional components affect query selection only and never fabricate labels."
        ),
    }
    if warmup is not None:
        terminal.update(warmup.completion_fields(len(ledger.consumed())))
    if controlled_acquisition:
        terminal.update(control_fields(acquisition_mode))
    if scaler_binding is not None:
        terminal["frozen_scaler"] = scaler_binding.reference
    if decoder_graph_mode == "neutral":
        terminal["interpretation"] = (
            "The selector uses Gold-free surrogate probabilities. The decoder uses p=.5 for all edges without "
            "a successful direct answer, and direct 0/1 for successful answers. Provisional partitions are not labels.")
    if controlled_acquisition:
        terminal["interpretation"] = (
            "The unchanged adaptive selector uses non-learning acquisition scores, not calibrated LLM probabilities. "
            "No LR model is instantiated, trained or used for prediction, and no LR dynamic features are computed. "
            "The decoder uses p=.5 without a successful direct answer and direct 0/1 for successful answers.")
    if tracer is not None:
        terminal["analysis_trace"] = tracer.save(
            snapshot_id="terminal", kind="TERMINAL", micro_round=micro_round, stage="TERMINAL",
            ledger=ledger, upper=decoder_upper, acquisition_upper=upper, current=current, surrogate=surrogate,
            evidence_sha256=terminal_evidence_sha256,
            decoder_called=not probing_pending, probing_held=probing_pending,
            probability_mode=("NEUTRAL_UNQUERIED_DIRECT_OVERWRITE" if decoder_graph_mode == "neutral"
                              or ((scaler_binding is not None or warmup is not None) and probing_pending) else "SURROGATE_WITH_DIRECT_OVERWRITE"), usage=usage_total,
            phase_timings={"state_computation_and_terminal_preparation": time.monotonic() - terminal_state_started},
            committed_reference=terminal_trace_reference)
    _json_dump(output_dir / "terminal.json", terminal)
    _json_dump(output_dir / "evidence.json", {
        "schema": "aqcc-final-evidence-v2",
        "history": ledger.history(),
        "sha256": terminal_evidence_sha256,
    })
    summary = {k: terminal[k] for k in (
        "status", "dataset_id", "execution_mode", "strategy", "stop_reason",
        "logical_pairs_queried", "total_possible_pairs", "pair_query_ratio",
        "probing_complete", "evidence_counts", "category_counts", "surrogate",
        "decoder", "reference_eligible", "usage", "elapsed_seconds_this_session",
    )}
    if warmup is not None:
        summary.update(warmup.completion_fields(len(ledger.consumed())))
    if controlled_acquisition:
        summary.update(control_fields(acquisition_mode))
    _json_dump(output_dir / "summary.json", summary)
    return terminal


def run(dataset, judge, *, output_dir: Path, resume: bool = False,
        detailed_trace: bool = False, **kwargs) -> dict:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not resume:
        raise ValueError("output directory exists; use --resume or a new run-id")
    if resume and not (output_dir / "manifest.json").is_file():
        raise ValueError("resume requires an existing run manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _run_lock(output_dir / ".run.lock"):
        return _run_impl(dataset, judge, output_dir=output_dir, resume=resume,
                         detailed_trace=detailed_trace, **kwargs)
