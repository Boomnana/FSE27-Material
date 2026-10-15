from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


def _strict_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_strict_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    report_ids: tuple[str, ...]
    texts: tuple[str, ...]
    embeddings: np.ndarray | sparse.csr_matrix
    initial_partition: tuple[tuple[int, ...], ...]
    source_path: str
    source_sha256: str

    @property
    def n(self) -> int:
        return len(self.report_ids)

    def id_to_index(self) -> dict[str, int]:
        return {rid: i for i, rid in enumerate(self.report_ids)}

    def partition_ids(self, partition: Iterable[Iterable[int]] | None = None) -> list[list[str]]:
        use = self.initial_partition if partition is None else partition
        return [[self.report_ids[i] for i in block] for block in use]

    def subset(self, keep_ids: set[str], dataset_id: str | None = None) -> "Dataset":
        old = self.id_to_index()
        kept_old = [i for i, rid in enumerate(self.report_ids) if rid in keep_ids]
        if not kept_old:
            raise ValueError("subset is empty")
        missing = keep_ids - {self.report_ids[i] for i in kept_old}
        if missing:
            raise ValueError(f"subset contains unknown IDs: {sorted(missing)[:5]}")
        new_index = {old_i: new_i for new_i, old_i in enumerate(kept_old)}
        blocks = []
        for block in self.initial_partition:
            members = tuple(new_index[i] for i in block if i in new_index)
            if members:
                blocks.append(members)
        return Dataset(
            dataset_id=dataset_id or self.dataset_id,
            report_ids=tuple(self.report_ids[i] for i in kept_old),
            texts=tuple(self.texts[i] for i in kept_old),
            embeddings=(self.embeddings[kept_old].tocsr() if sparse.issparse(self.embeddings)
                        else np.ascontiguousarray(self.embeddings[kept_old], dtype=np.float32)),
            initial_partition=canonical_partition(blocks),
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )


def canonical_partition(blocks: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    cleaned = [tuple(sorted(set(block))) for block in blocks]
    if any(not b for b in cleaned):
        raise ValueError("partition contains empty block")
    cleaned.sort(key=lambda b: (b[0], len(b), b))
    return tuple(cleaned)


def _validate_partition(partition: tuple[tuple[int, ...], ...], n: int) -> None:
    seen = [i for block in partition for i in block]
    if sorted(seen) != list(range(n)):
        raise ValueError("initial partition must cover each report exactly once")


def load_start_state(path: Path) -> Dataset:
    path = path.resolve()
    doc = read_json(path)
    required = {"dataset_id", "report_ids", "reports", "embedding_dimension", "partition"}
    if not required <= set(doc):
        raise ValueError("start_state missing required fields")
    if ("embeddings" in doc) == ("embeddings_npz" in doc):
        raise ValueError("provide exactly one of embeddings or embeddings_npz")
    ids = tuple(doc["report_ids"])
    if len(ids) != len(set(ids)) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("invalid report_ids")
    reports = doc["reports"]
    if len(reports) != len(ids):
        raise ValueError("reports/report_ids length mismatch")
    texts = []
    for i, (rid, row) in enumerate(zip(ids, reports)):
        if row.get("report_index") != i or row.get("source_id") != rid:
            raise ValueError("report ordering mismatch")
        text = row.get("description")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty report text at {rid}")
        expected = row.get("text_sha256")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected and expected != actual:
            raise ValueError(f"text hash mismatch for {rid}")
        texts.append(text)
    if "embeddings_npz" in doc:
        external = doc["embeddings_npz"]
        if not isinstance(external, dict) or external.get("format") != "scipy_csr_npz":
            raise ValueError("unsupported external embedding format")
        name = external.get("path")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("embedding path must be relative to start_state")
        embedding_path = (path.parent / name).resolve()
        if not embedding_path.is_relative_to(path.parent) or embedding_path.suffix != ".npz":
            raise ValueError("embedding path must stay within the start_state directory")
        expected_embedding_digest = external.get("sha256")
        if expected_embedding_digest and sha256_file(embedding_path) != expected_embedding_digest:
            raise ValueError("external embedding hash mismatch")
        emb = sparse.load_npz(embedding_path).tocsr().astype(np.float32)
        emb.sum_duplicates()
        finite_values = emb.data
    else:
        emb = np.asarray(doc["embeddings"], dtype=np.float32)
        finite_values = emb
    if emb.shape != (len(ids), int(doc["embedding_dimension"])):
        raise ValueError("embedding shape mismatch")
    if not np.isfinite(finite_values).all():
        raise ValueError("embedding contains nonfinite values")
    labels = doc["partition"].get("labels_by_index")
    clusters = doc["partition"].get("clusters")
    if not isinstance(labels, list) or len(labels) != len(ids) or not isinstance(clusters, list):
        raise ValueError("invalid partition object")
    blocks = []
    for c in clusters:
        members = tuple(int(i) for i in c["members"])
        blocks.append(members)
        if [ids[i] for i in members] != c.get("member_source_ids"):
            raise ValueError("cluster member_source_ids mismatch")
    partition = canonical_partition(blocks)
    _validate_partition(partition, len(ids))
    return Dataset(
        dataset_id=str(doc["dataset_id"]), report_ids=ids, texts=tuple(texts), embeddings=emb,
        initial_partition=partition, source_path=str(path), source_sha256=sha256_file(path)
    )
