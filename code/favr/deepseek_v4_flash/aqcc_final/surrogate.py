from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SurrogateSummary:
    fitted: bool
    training_pairs: int
    same_pairs: int
    different_pairs: int
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float | None


class LLMSurrogate:
    """Predicts the frozen LLM's atomic SAME probability, never Gold.

    Dynamic graph features are constructed to avoid direct target leakage: the
    queried (i,j) edge itself is not needed to compute common-neighbour counts,
    and endpoint degrees subtract the target edge during training.
    """

    FEATURE_NAMES = (
        "dense_cosine", "lexical_char_tfidf", "same_p1", "dense_x_lexical",
        "abs_dense_lexical", "log1p_common_same", "signed_twohop_norm",
        "same_degree_min", "different_degree_min",
    )

    def __init__(self, pair_store, *, min_samples: int = 24, C: float = 1.0,
                 feature_policy: str = "all", scaler_policy: str = "refit"):
        if feature_policy not in ("all", "drop_same_degree_min"):
            raise ValueError("unknown surrogate feature policy")
        if scaler_policy not in ("refit", "freeze_after_probing"):
            raise ValueError("unknown surrogate scaler policy")
        self.store = pair_store
        self.min_samples = int(min_samples)
        self.C = float(C)
        self.feature_policy = feature_policy
        self.scaler_policy = scaler_policy
        self.FEATURE_NAMES = tuple(name for name in type(self).FEATURE_NAMES
                                   if feature_policy == "all" or name != "same_degree_min")
        self._frozen_scaler: StandardScaler | None = None
        self._frozen_state: dict | None = None
        self.model: Pipeline | None = None
        self.summary = SurrogateSummary(False, 0, 0, 0, self.FEATURE_NAMES, tuple(), None)
        
        
        
        
        self._dynamic_key = None
        self._dynamic_statistics = None

    @staticmethod
    def _state_sha256(value: dict) -> str:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _scaler_context(self) -> dict:
        
        
        digest = hashlib.sha256()
        for name in ("i", "j", "dense", "lexical", "same_p1"):
            array = np.ascontiguousarray(getattr(self.store, name))
            digest.update(json.dumps([name, array.dtype.str, list(array.shape)],
                                     separators=(",", ":")).encode("utf-8"))
            digest.update(array.tobytes())
        return {"n": int(self.store.n), "total_pairs": int(self.store.total_pairs),
                "fixed_feature_store_sha256": digest.hexdigest(),
                "feature_names": list(self.FEATURE_NAMES),
                "feature_policy": self.feature_policy, "scaler_policy": self.scaler_policy,
                "training_feature_dtype": "float32",
                "numpy_version": np.__version__, "sklearn_version": sklearn.__version__}

    def _freeze_edges_sha256(self, signed_edges: dict[tuple[int, int], int]) -> str:
        canonical = []
        for pair, sign in sorted(signed_edges.items()):
            if (type(pair) is not tuple or len(pair) != 2
                    or any(not isinstance(node, (int, np.integer)) or isinstance(node, (bool, np.bool_)) for node in pair)
                    or not 0 <= pair[0] < pair[1] < self.store.n
                    or not isinstance(sign, (int, np.integer)) or isinstance(sign, (bool, np.bool_))
                    or sign not in (-1, 1)):
                raise ValueError("invalid atomic edge for frozen scaler")
            canonical.append([int(pair[0]), int(pair[1]), int(sign)])
        return self._state_sha256({"signed_edges": canonical})

    def freeze_scaler(self, signed_edges: dict[tuple[int, int], int]) -> dict:
        """Freeze once on probing-completion training features, even one class.

        The runner owns the boundary and must persist this state before using
        it. Empty evidence has no estimable scaler and is rejected explicitly.
        A retry with exactly the same boundary evidence is idempotent.
        """
        if self.scaler_policy != "freeze_after_probing":
            raise ValueError("freeze_scaler requires freeze_after_probing policy")
        if not signed_edges:
            raise ValueError("cannot freeze scaler without successful atomic edges")
        edges_hash = self._freeze_edges_sha256(signed_edges)
        if self._frozen_state is not None:
            if (self._frozen_state["freeze_edges_sha256"] != edges_hash
                    or self._frozen_state["context"] != self._scaler_context()):
                raise ValueError("frozen scaler cannot be refrozen on different evidence or features")
            return self.frozen_scaler_state()
        indices = np.asarray([self.store.pair_index(a, b) for a, b in sorted(signed_edges)], dtype=np.int32)
        X = self.feature_matrix(indices, signed_edges, training=True)
        if X.dtype != np.float32 or not np.isfinite(X).all():
            raise ValueError("frozen scaler requires finite float32 training features")
        scaler = StandardScaler().fit(X)
        state = {"schema": "aqcc-frozen-surrogate-scaler-v1", "context": self._scaler_context(),
                 "freeze_edges_sha256": edges_hash, "freeze_training_pairs": len(signed_edges),
                 "parameters": {"copy": True, "with_mean": True, "with_std": True},
                 "n_features_in": int(scaler.n_features_in_),
                 "n_samples_seen": np.asarray(scaler.n_samples_seen_).item(),
                 "n_samples_seen_dtype": np.asarray(scaler.n_samples_seen_).dtype.str,
                 "mean": scaler.mean_.tolist(), "var": scaler.var_.tolist(),
                 "scale": scaler.scale_.tolist()}
        state["payload_sha256"] = self._state_sha256(state)
        self._frozen_scaler = scaler
        self._frozen_state = state
        return self.frozen_scaler_state()

    def frozen_scaler_state(self) -> dict | None:
        """Return a detached JSON-safe checkpoint, or None before freezing."""
        return None if self._frozen_state is None else json.loads(json.dumps(self._frozen_state, allow_nan=False))

    def restore_frozen_scaler_state(self, state: dict) -> None:
        """Restore exact float64 scaler statistics used on float32 features.

        Runtime/data/policy identity and checkpoint integrity must all match.
        LR is deliberately not restored: the runner fits it on its current
        committed evidence, using this original immutable scaler.
        """
        required = {"schema", "context", "freeze_edges_sha256", "freeze_training_pairs",
                    "parameters", "n_features_in", "n_samples_seen", "n_samples_seen_dtype",
                    "mean", "var", "scale", "payload_sha256"}
        if self.scaler_policy != "freeze_after_probing" or type(state) is not dict or set(state) != required:
            raise ValueError("invalid frozen scaler checkpoint policy or schema")
        if state["schema"] != "aqcc-frozen-surrogate-scaler-v1" or state["context"] != self._scaler_context():
            raise ValueError("frozen scaler feature/runtime identity differs")
        payload = {key: value for key, value in state.items() if key != "payload_sha256"}
        if state["payload_sha256"] != self._state_sha256(payload):
            raise ValueError("frozen scaler checkpoint checksum differs")
        if state["parameters"] != {"copy": True, "with_mean": True, "with_std": True}:
            raise ValueError("frozen scaler parameters differ")
        if (type(state["n_features_in"]) is not int or state["n_features_in"] != len(self.FEATURE_NAMES)
                or type(state["n_samples_seen"]) not in (int, float)
                or not np.isfinite(state["n_samples_seen"]) or state["n_samples_seen"] < 1
                or type(state["freeze_training_pairs"]) is not int
                or state["freeze_training_pairs"] != state["n_samples_seen"]
                or state["freeze_training_pairs"] > self.store.total_pairs):
            raise ValueError("invalid frozen scaler dimensions or sample count")
        if state["n_samples_seen_dtype"] not in (np.dtype(np.int64).str, np.dtype(np.float64).str):
            raise ValueError("invalid frozen scaler sample-count dtype")
        count_dtype = np.dtype(state["n_samples_seen_dtype"])
        if ((count_dtype.kind == "i" and type(state["n_samples_seen"]) is not int)
                or (count_dtype.kind == "f" and type(state["n_samples_seen"]) is not float)):
            raise ValueError("frozen scaler sample-count dtype/value differ")
        edge_hash = state["freeze_edges_sha256"]
        if type(edge_hash) is not str or len(edge_hash) != 64 or any(c not in "0123456789abcdef" for c in edge_hash):
            raise ValueError("invalid frozen scaler evidence identity")
        arrays = {}
        for key in ("mean", "var", "scale"):
            values = state[key]
            if (type(values) is not list or len(values) != len(self.FEATURE_NAMES)
                    or any(type(value) not in (int, float) for value in values)):
                raise ValueError("invalid frozen scaler statistic shape/type")
            arrays[key] = np.asarray(values, dtype=np.float64)
            if not np.isfinite(arrays[key]).all():
                raise ValueError("nonfinite frozen scaler statistics")
        if np.any(arrays["var"] < 0) or np.any(arrays["scale"] <= 0):
            raise ValueError("invalid frozen scaler variance/scale")
        if self._frozen_state is not None:
            if self._frozen_state != state:
                raise ValueError("cannot replace an existing frozen scaler")
            return
        scaler = StandardScaler()
        scaler.n_features_in_ = state["n_features_in"]
        
        
        scaler.n_samples_seen_ = np.asarray(state["n_samples_seen"], dtype=count_dtype)[()]
        scaler.mean_, scaler.var_, scaler.scale_ = arrays["mean"], arrays["var"], arrays["scale"]
        self._frozen_scaler = scaler
        self._frozen_state = json.loads(json.dumps(state, allow_nan=False))
        self.model = None
        self.summary = SurrogateSummary(False, 0, 0, 0, self.FEATURE_NAMES, tuple(), None)

    def _signed_adjacency(self, signed_edges: dict[tuple[int, int], int]) -> np.ndarray:
        s = np.zeros((self.store.n, self.store.n), dtype=np.int8)
        for (a, b), sign in signed_edges.items():
            s[a, b] = s[b, a] = int(sign)
        return s

    def _dynamic_all(self, signed_edges: dict[tuple[int, int], int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (self.store.n, tuple(signed_edges.items()))
        if key == self._dynamic_key:
            return self._dynamic_statistics
        s = self._signed_adjacency(signed_edges)
        pos = (s == 1).astype(np.float32)
        neg = (s == -1).astype(np.float32)
        common_same = pos @ pos.T
        contradictory = pos @ neg.T + neg @ pos.T
        same_deg = pos.sum(axis=1)
        diff_deg = neg.sum(axis=1)
        statistics = common_same, contradictory, same_deg, diff_deg
        
        
        for array in statistics:
            array.flags.writeable = False
        self._dynamic_key = key
        self._dynamic_statistics = statistics
        return statistics

    def feature_matrix(self, pair_indices: np.ndarray, signed_edges: dict[tuple[int, int], int],
                       *, training: bool = False) -> np.ndarray:
        idx = np.asarray(pair_indices, dtype=np.int32)
        static = self.store.static_feature_matrix(idx)
        ii = self.store.i[idx]; jj = self.store.j[idx]
        common, contradictory, same_deg, diff_deg = self._dynamic_all(signed_edges)
        cs = np.log1p(common[ii, jj]).astype(np.float32)
        denom = 1.0 + common[ii, jj] + contradictory[ii, jj]
        signed2 = ((common[ii, jj] - contradictory[ii, jj]) / denom).astype(np.float32)
        sdeg_i = same_deg[ii].astype(np.float32); sdeg_j = same_deg[jj].astype(np.float32)
        ddeg_i = diff_deg[ii].astype(np.float32); ddeg_j = diff_deg[jj].astype(np.float32)
        if training:
            for k, (a, b) in enumerate(zip(ii.tolist(), jj.tolist())):
                sign = signed_edges.get((int(a), int(b)))
                if sign == 1:
                    sdeg_i[k] -= 1.0; sdeg_j[k] -= 1.0
                elif sign == -1:
                    ddeg_i[k] -= 1.0; ddeg_j[k] -= 1.0
        norm = max(1.0, float(self.store.n - 1))
        smin = np.minimum(sdeg_i, sdeg_j) / norm
        dmin = np.minimum(ddeg_i, ddeg_j) / norm
        features = np.column_stack([static, cs, signed2, smin, dmin]).astype(np.float32, copy=False)
        if self.feature_policy == "drop_same_degree_min":
            features = np.delete(features, 7, axis=1)
        return features

    def fit(self, signed_edges: dict[tuple[int, int], int]) -> SurrogateSummary:
        if not signed_edges:
            self.model = None
            self.summary = SurrogateSummary(False, 0, 0, 0, self.FEATURE_NAMES, tuple(), None)
            return self.summary
        indices = np.asarray([self.store.pair_index(a, b) for a, b in sorted(signed_edges)], dtype=np.int32)
        y = np.asarray([1 if signed_edges[(a, b)] == 1 else 0 for a, b in sorted(signed_edges)], dtype=np.int32)
        same = int(y.sum()); diff = int(len(y) - same)
        if len(y) < self.min_samples or same == 0 or diff == 0:
            self.model = None
            self.summary = SurrogateSummary(False, len(y), same, diff, self.FEATURE_NAMES, tuple(), None)
            return self.summary
        X = self.feature_matrix(indices, signed_edges, training=True)
        if self.scaler_policy == "refit":
            
            
            model = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs", random_state=0)),
            ])
            model.fit(X, y)
        else:
            if self._frozen_scaler is None:
                raise ValueError("runner must freeze or restore scaler before fitting")
            lr = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs", random_state=0)
            
            lr.fit(self._frozen_scaler.transform(X), y)
            model = Pipeline([("scale", self._frozen_scaler), ("lr", lr)])
        self.model = model
        lr = model.named_steps["lr"]
        self.summary = SurrogateSummary(
            True, len(y), same, diff, self.FEATURE_NAMES,
            tuple(float(v) for v in lr.coef_[0]), float(lr.intercept_[0]),
        )
        return self.summary

    def predict_unqueried(self, signed_edges: dict[tuple[int, int], int]) -> np.ndarray:
        total = self.store.total_pairs
        if self.model is None:
            return np.full(total, 0.5, dtype=np.float32)
        idx = np.arange(total, dtype=np.int32)
        X = self.feature_matrix(idx, signed_edges, training=False)
        p = self.model.predict_proba(X)[:, 1]
        return np.clip(p, 1e-4, 1 - 1e-4).astype(np.float32)
