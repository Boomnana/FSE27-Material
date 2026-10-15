"""Optional, offline snapshots of states the runner has already computed."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

from .decoder import labels_of
from .provider import canonical_sha256, strict_json
from .acquisition import DisabledSurrogate, control_fields


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _array_state(array) -> dict:
    array = np.asarray(array)
    return {"dtype": str(array.dtype), "values": array.tolist()}


def surrogate_state(surrogate) -> dict:
    """Preserve fitted values, including scaler statistics, without refitting."""
    if isinstance(surrogate, DisabledSurrogate):
        return _json_value(surrogate.state())
    result = {
        "summary": asdict(surrogate.summary),
        "min_samples": surrogate.min_samples,
        "regularization_C": surrogate.C,
        "feature_dtype": "float32",
        "prediction_dtype": "float32",
        "prediction_clip": [1e-4, 1 - 1e-4],
        "unfitted_same_probability": 0.5,
        "model": None,
        "feature_policy": getattr(surrogate, "feature_policy", "all"),
        "scaler_policy": getattr(surrogate, "scaler_policy", "refit"),
        "frozen_scaler_state": (surrogate.frozen_scaler_state()
                                if hasattr(surrogate, "frozen_scaler_state") else None),
    }
    if surrogate.model is not None:
        scale = surrogate.model.named_steps["scale"]
        classifier = surrogate.model.named_steps["lr"]
        result["model"] = {
            "scaler": {
                "with_mean": scale.with_mean,
                "with_std": scale.with_std,
                "mean": _array_state(scale.mean_),
                "variance": _array_state(scale.var_),
                "scale": _array_state(scale.scale_),
                "n_samples_seen": _array_state(scale.n_samples_seen_),
                "n_features_in": int(scale.n_features_in_),
            },
            "logistic_regression": {
                "coefficients": _array_state(classifier.coef_),
                "intercept": _array_state(classifier.intercept_),
                "classes": _array_state(classifier.classes_),
                "n_iter": _array_state(classifier.n_iter_),
                "n_features_in": int(classifier.n_features_in_),
                "solver": classifier.solver,
                "max_iter": classifier.max_iter,
                "random_state": classifier.random_state,
            },
        }
    return _json_value(result)


class AnalysisTracer:
    """Snapshot files become committed only when their owner records the hashes.

    The manifest owns the initial snapshot, each committed micro-round owns its
    round snapshot, and the terminal owns the terminal snapshot. A crash before
    the owning record is committed leaves replaceable files, never new evidence.
    """

    def __init__(self, output_dir, dataset, store, config, *, json_writer):
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.store = store
        self.config_sha256 = canonical_sha256(config)
        self.decoder_max_steps = config["decoder_max_steps"]
        self.decoder_graph_mode = config.get("decoder_graph_mode", "surrogate")
        self.initial_query_policy = config.get("initial_query_policy", "recursive")
        self.warmup_pairs = config.get("warmup_pairs")
        self.acquisition_mode = config.get("acquisition_mode", "surrogate")
        self.json_writer = json_writer

    def _paths(self, snapshot_id):
        if snapshot_id not in {"initial", "terminal"} and not (
                len(snapshot_id) >= 6 and snapshot_id[0] == "M" and snapshot_id[1:].isdigit()):
            raise ValueError("invalid analysis trace snapshot ID")
        relative = f"traces/{snapshot_id}"
        return relative + ".json", relative + ".npz"

    def verify(self, reference, *, snapshot_id, query_count, evidence_sha256,
               partition=None, usage=None):
        json_name, npz_name = self._paths(snapshot_id)
        if (not isinstance(reference, dict) or reference.get("json_path") != json_name
                or reference.get("probabilities_path") != npz_name):
            raise ValueError(f"missing or invalid committed analysis trace: {snapshot_id}")
        json_path, npz_path = self.output_dir / json_name, self.output_dir / npz_name
        if not json_path.is_file() or not npz_path.is_file():
            raise ValueError(f"committed analysis trace files are missing: {snapshot_id}")
        if (_sha256(json_path) != reference.get("json_sha256")
                or _sha256(npz_path) != reference.get("probabilities_sha256")):
            raise ValueError(f"committed analysis trace hash mismatch: {snapshot_id}")
        doc = strict_json(json_path.read_text(encoding="utf-8"))
        expected = {"snapshot_id": snapshot_id, "query_count": query_count,
                    "evidence_sha256": evidence_sha256, "config_sha256": self.config_sha256,
                    "report_ids": list(self.dataset.report_ids), "probabilities_path": npz_name,
                    "probabilities_sha256": reference["probabilities_sha256"]}
        if any(doc.get(key) != value for key, value in expected.items()):
            raise ValueError(f"committed analysis trace identity mismatch: {snapshot_id}")
        if partition is not None and doc["partition"] != partition:
            raise ValueError(f"committed analysis trace partition mismatch: {snapshot_id}")
        if usage is not None and doc["usage"] != usage:
            raise ValueError(f"committed analysis trace usage mismatch: {snapshot_id}")
        return doc

    def _write_npz(self, path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            
            
            
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1) as archive:
                for name, value in arrays.items():
                    with archive.open(name + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(7):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 6:
                    raise
                time.sleep(0.01 * 2 ** attempt)

    def save(self, *, snapshot_id, kind, micro_round, stage, ledger, upper,
             current, surrogate, decoder_called, probing_held, probability_mode,
             usage, phase_timings=None, committed_reference=None, acquisition_upper=None,
             evidence_sha256=None):
        json_name, npz_name = self._paths(snapshot_id)
        
        
        
        if evidence_sha256 is None:
            evidence_sha256 = ledger.snapshot_sha256()
        elif (type(evidence_sha256) is not str or len(evidence_sha256) != 64
              or any(c not in "0123456789abcdef" for c in evidence_sha256)):
            raise ValueError("invalid completed-ledger SHA-256")
        query_count = len(ledger.consumed())
        arrays = {
            "p_same": np.asarray(upper, dtype=np.float32),
            "acquisition_p_same": np.asarray(upper if acquisition_upper is None else acquisition_upper, dtype=np.float32),
            "decoder_graph_mode": np.asarray(self.decoder_graph_mode),
            "pair_i": self.store.i, "pair_j": self.store.j,
            "pair_index": np.arange(self.store.total_pairs, dtype=np.int32),
            "report_ids": np.asarray(self.dataset.report_ids),
            "query_count": np.asarray(query_count, dtype=np.int64),
            "evidence_sha256": np.asarray(evidence_sha256),
            "config_sha256": np.asarray(self.config_sha256),
            "snapshot_id": np.asarray(snapshot_id),
        }
        labels = labels_of(current.partition, self.dataset.n)
        same_cluster = labels[self.store.i] == labels[self.store.j]
        probabilities = np.asarray(upper, dtype=np.float64)
        graph_loss = float(np.where(same_cluster, 1.0 - probabilities, probabilities).sum(dtype=np.float64))
        acquisition = arrays["acquisition_p_same"].astype(np.float64)
        acquisition_loss = float(np.where(same_cluster, 1.0 - acquisition, acquisition).sum(dtype=np.float64))
        doc = _json_value({
            "schema": "aqcc-analysis-trace-v1", "snapshot_id": snapshot_id,
            "kind": kind, "micro_round": micro_round, "stage": stage,
            "dataset_id": self.dataset.dataset_id, "report_ids": list(self.dataset.report_ids),
            "query_count": query_count, "evidence_sha256": evidence_sha256,
            "evidence_counts": ledger.counts(), "config_sha256": self.config_sha256,
            "probabilities_path": npz_name, "probability_mode": probability_mode,
            "decoder_graph_mode": self.decoder_graph_mode,
            "acquisition_probability_mode": ("SURROGATE_WITH_DIRECT_OVERWRITE" if surrogate.model is not None
                                              else "NEUTRAL_UNQUERIED_DIRECT_OVERWRITE"),
            "acquisition_loss_on_current_partition": acquisition_loss,
            "partition": self.dataset.partition_ids(current.partition),
            "partition_indices": current.partition,
            "direct_edges": [[a, b, sign] for (a, b), sign in sorted(ledger.signed_edges().items())],
            "surrogate": surrogate_state(surrogate),
            "decoder": {
                "decode_called": bool(decoder_called), "held_for_probing": bool(probing_held),
                "held_reason": "PROBING_PENDING" if probing_held else (
                    None if decoder_called else "EXHAUSTIVE_DECODE_DEFERRED"),
                "loss_on_saved_graph": graph_loss,
                "graph_mode": self.decoder_graph_mode,
                "result_loss": current.loss if decoder_called else None,
                "steps": current.steps if decoder_called else 0,
                "max_steps": self.decoder_max_steps,
                "reached_step_cap": bool(decoder_called and current.steps >= self.decoder_max_steps),
                "operations": current.operations if decoder_called else [],
                "restart_seed": "P1_EVERY_DECODE", "global_optimum_claimed": False,
            },
            "usage": dict(usage), "phase_timings_seconds": phase_timings or {},
            "gold_accessed": False,
        })
        if self.initial_query_policy != "recursive":
            complete = query_count >= self.warmup_pairs
            doc.update(initial_query_policy=self.initial_query_policy,
                       initial_query_complete=complete, probing_complete=None)
            doc["decoder"]["held_for_probing"] = False
            doc["decoder"]["held_for_initial_query"] = bool(probing_held)
            doc["decoder"]["held_reason"] = (
                "FIXED_INITIAL_WARMUP_PENDING" if probing_held else
                (None if decoder_called else "DECODE_NOT_CALLED"))
        if self.acquisition_mode != "surrogate":
            doc.update(control_fields(self.acquisition_mode))
            doc["acquisition_probability_mode"] = (
                "NEUTRAL_UNQUERIED_DIRECT_OVERWRITE" if probing_held else
                "CONSTANT_HALF_WITH_DIRECT_OVERWRITE" if self.acquisition_mode == "constant" else
                "HEURISTIC_TEXT_COSINE_WITH_DIRECT_OVERWRITE")
            
            arrays["acquisition_mode"] = np.asarray(self.acquisition_mode)
            arrays["acquisition_score_semantics"] = np.asarray(doc["acquisition_score_semantics"])
        if committed_reference is not None:
            old = self.verify(committed_reference, snapshot_id=snapshot_id,
                              query_count=query_count, evidence_sha256=evidence_sha256,
                              partition=doc["partition"], usage=doc["usage"])
            excluded = {"phase_timings_seconds", "probabilities_sha256"}
            if ({key: value for key, value in old.items() if key not in excluded}
                    != {key: value for key, value in doc.items() if key not in excluded}):
                raise ValueError(f"committed analysis trace state differs from recomputation: {snapshot_id}")
            with np.load(self.output_dir / npz_name, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                        not np.array_equal(saved[key], value) for key, value in arrays.items()):
                    raise ValueError(f"committed analysis trace probabilities differ: {snapshot_id}")
            return committed_reference

        self._write_npz(self.output_dir / npz_name, arrays)
        doc["probabilities_sha256"] = _sha256(self.output_dir / npz_name)
        self.json_writer(self.output_dir / json_name, doc)
        return {"json_path": json_name, "json_sha256": _sha256(self.output_dir / json_name),
                "probabilities_path": npz_name,
                "probabilities_sha256": doc["probabilities_sha256"]}
