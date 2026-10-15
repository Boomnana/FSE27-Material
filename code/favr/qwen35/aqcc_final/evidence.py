from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Iterable

LABELS = {"SAME", "DIFFERENT"}
_JSON_ATOMIC_TYPES = frozenset((str, int, float, bool, type(None)))


class _NonJSONValue(Exception):
    pass


def _copy_json_tree(value):
    """Copy ordinary JSON containers without dataclass introspection per leaf.

    Only exact built-in types take this path. Unusual objects/subclasses retain
    dataclasses.asdict's original conversion semantics through the fallback.
    No history is cached: caller mutations of metadata remain observable.
    """
    kind = type(value)
    if kind in _JSON_ATOMIC_TYPES:
        return value
    if kind is list:
        return [item if type(item) in _JSON_ATOMIC_TYPES else _copy_json_tree(item) for item in value]
    if kind is dict:
        return {(key if type(key) in _JSON_ATOMIC_TYPES else _copy_json_tree(key)):
                (item if type(item) in _JSON_ATOMIC_TYPES else _copy_json_tree(item))
                for key, item in value.items()}
    raise _NonJSONValue


def pair(a: int, b: int) -> tuple[int, int]:
    if a == b:
        raise ValueError("self-pair is invalid")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Observation:
    query_index: int
    left: int
    right: int
    label: str | None
    reason: str
    status: str
    request_sha256: str
    response_sha256: str | None
    metadata: dict

    def as_dict(self) -> dict:
        if (type(self) is Observation
                and type(self.query_index) is int and type(self.left) is int and type(self.right) is int
                and type(self.label) in (str, type(None)) and type(self.reason) is str
                and type(self.status) is str and type(self.request_sha256) is str
                and type(self.response_sha256) in (str, type(None))):
            try:
                metadata = _copy_json_tree(self.metadata)
            except _NonJSONValue:
                pass
            else:
                return {"query_index": self.query_index, "left": self.left, "right": self.right,
                        "label": self.label, "reason": self.reason, "status": self.status,
                        "request_sha256": self.request_sha256, "response_sha256": self.response_sha256,
                        "metadata": metadata}
        return asdict(self)


class EvidenceLedger:
    """Exactly one semantic sample per unordered report pair.

    Only successful atomic SAME/DIFFERENT answers become hard evidence. A
    transport/protocol failure consumes the logical pair for this run but never
    fabricates a sign. No transitive, partition-derived, similarity-derived, or
    surrogate-derived edge is written here.
    """

    def __init__(self, n: int):
        self.n = n
        self._obs: dict[tuple[int, int], Observation] = {}
        self._order: list[tuple[int, int]] = []

    def consumed(self) -> set[tuple[int, int]]:
        return set(self._obs)

    def get(self, a: int, b: int) -> Observation | None:
        return self._obs.get(pair(a, b))

    def add(self, obs: Observation) -> None:
        p = pair(obs.left, obs.right)
        if not (0 <= p[0] < p[1] < self.n):
            raise ValueError("pair outside population")
        if p in self._obs:
            raise ValueError("pair already semantically sampled")
        if obs.status == "SUCCESS":
            if obs.label not in LABELS:
                raise ValueError("successful observation must be SAME or DIFFERENT")
            if type(obs.reason) is not str or not obs.reason.strip():
                raise ValueError("successful observation requires nonempty reason")
        elif obs.label is not None:
            raise ValueError("failed observation cannot carry semantic label")
        self._obs[p] = obs
        self._order.append(p)

    def signed_edges(self) -> dict[tuple[int, int], int]:
        out: dict[tuple[int, int], int] = {}
        for p in self._order:
            obs = self._obs[p]
            if obs.status != "SUCCESS":
                continue
            out[p] = 1 if obs.label == "SAME" else -1
        return out

    def successful_pairs(self) -> list[tuple[int, int]]:
        return [p for p in self._order if self._obs[p].status == "SUCCESS"]

    def counts(self) -> dict[str, int]:
        out = {"SAME": 0, "DIFFERENT": 0, "FAILED": 0}
        for obs in self._obs.values():
            if obs.status == "SUCCESS":
                out[obs.label] += 1
            else:
                out["FAILED"] += 1
        return out

    def category_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for obs in self._obs.values():
            category = str(obs.metadata.get("category", "UNSPECIFIED"))
            out[category] = out.get(category, 0) + 1
        return dict(sorted(out.items()))

    def history(self) -> list[dict]:
        return [self._obs[p].as_dict() for p in self._order]

    def snapshot_sha256(self) -> str:
        raw = json.dumps(self.history(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_rows(cls, n: int, rows: Iterable[dict]) -> "EvidenceLedger":
        obj = cls(n)
        for row in rows:
            obj.add(Observation(**row))
        return obj
