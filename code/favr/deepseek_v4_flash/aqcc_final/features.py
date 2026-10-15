from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _labels_from_partition(partition, n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int32)
    for ci, block in enumerate(partition):
        for u in block:
            if labels[u] != -1:
                raise ValueError("duplicate report in partition")
            labels[u] = ci
    if (labels < 0).any():
        raise ValueError("partition does not cover population")
    return labels


def _van_der_corput_rank_order(total: int) -> np.ndarray:
    """Deterministic rank-stratified order independent of the final budget."""
    if total <= 0:
        return np.empty(0, dtype=np.int32)
    size = 1
    bits = 0
    while size < total:
        size <<= 1
        bits += 1
    rows = []
    for x in range(size):
        v = x
        r = 0
        for _ in range(bits):
            r = (r << 1) | (v & 1)
            v >>= 1
        if r < total:
            rows.append(r)
    return np.asarray(rows, dtype=np.int32)


@dataclass
class PairFeatureStore:
    n: int
    i: np.ndarray
    j: np.ndarray
    dense: np.ndarray
    lexical: np.ndarray
    same_p1: np.ndarray
    dense_matrix: np.ndarray
    lexical_matrix: np.ndarray
    pair_index_matrix: np.ndarray
    combined_order: np.ndarray
    dense_order: np.ndarray
    lexical_order: np.ndarray
    exploration_order: np.ndarray
    dense_neighbors: np.ndarray
    lexical_neighbors: np.ndarray
    metadata: dict

    @property
    def total_pairs(self) -> int:
        return int(self.i.size)

    def pair_index(self, a: int, b: int) -> int:
        if a == b:
            raise ValueError("self pair")
        x, y = (a, b) if a < b else (b, a)
        idx = int(self.pair_index_matrix[x, y])
        if idx < 0:
            raise ValueError("pair outside universe")
        return idx

    def pair_tuple(self, idx: int) -> tuple[int, int]:
        return int(self.i[idx]), int(self.j[idx])

    def static_feature_matrix(self, pair_indices: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(self.total_pairs, dtype=np.int32) if pair_indices is None else np.asarray(pair_indices, dtype=np.int32)
        d = self.dense[idx]
        l = self.lexical[idx]
        p1 = self.same_p1[idx].astype(np.float32)
        return np.column_stack([
            d,
            l,
            p1,
            d * l,
            np.abs(d - l),
        ]).astype(np.float32, copy=False)

    @classmethod
    def build(cls, dataset, *, cache_path: Path | None = None,
              lexical_max_features: int = 50000, neighbor_k: int = 20) -> "PairFeatureStore":
        cache_version = "aqcc-final-features-v3"
        expected = {
            "version": cache_version,
            "source_sha256": dataset.source_sha256,
            "dataset_id": dataset.dataset_id,
            "n": dataset.n,
            "lexical_max_features": int(lexical_max_features),
            "neighbor_k": int(neighbor_k),
            "identical_sparse_tfidf_policy": "SHARE_GRAPH_WHEN_CSR_VALUES_ARE_IDENTICAL",
        }
        if cache_path is not None and cache_path.is_file():
            z = np.load(cache_path, allow_pickle=False)
            meta = json.loads(str(z["metadata_json"].item()))
            if meta == expected:
                return cls(
                    n=int(meta["n"]), i=z["i"], j=z["j"], dense=z["dense"], lexical=z["lexical"],
                    same_p1=z["same_p1"].astype(bool), dense_matrix=z["dense_matrix"],
                    lexical_matrix=z["lexical_matrix"], pair_index_matrix=z["pair_index_matrix"],
                    combined_order=z["combined_order"], dense_order=z["dense_order"],
                    lexical_order=z["lexical_order"], exploration_order=z["exploration_order"],
                    dense_neighbors=z["dense_neighbors"], lexical_neighbors=z["lexical_neighbors"],
                    metadata=meta,
                )

        if sparse.issparse(dataset.embeddings):
            x = normalize(dataset.embeddings.astype(np.float32), norm="l2", copy=True)
            dense_matrix = np.asarray((x @ x.T).toarray(), dtype=np.float32)
        else:
            x = np.asarray(dataset.embeddings, dtype=np.float32)
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            x = x / norms
            dense_matrix = np.asarray(x @ x.T, dtype=np.float32)
        np.fill_diagonal(dense_matrix, 1.0)

        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=1,
            max_features=lexical_max_features, sublinear_tf=True,
            norm="l2", dtype=np.float32,
        )
        lex_sparse = vectorizer.fit_transform(dataset.texts)
        lexical_matrix = np.asarray((lex_sparse @ lex_sparse.T).toarray(), dtype=np.float32)
        np.fill_diagonal(lexical_matrix, 1.0)
        if sparse.issparse(dataset.embeddings) and dataset.embeddings.shape == lex_sparse.shape:
            supplied = dataset.embeddings.tocsr(copy=True)
            rebuilt = lex_sparse.tocsr(copy=True)
            supplied.sum_duplicates(); supplied.sort_indices()
            rebuilt.sum_duplicates(); rebuilt.sort_indices()
            if (np.array_equal(supplied.indptr, rebuilt.indptr)
                    and np.array_equal(supplied.indices, rebuilt.indices)
                    and np.array_equal(supplied.data, rebuilt.data)):
                
                
                
                lexical_matrix = dense_matrix.copy()

        n = dataset.n
        ii, jj = np.triu_indices(n, k=1)
        ii = ii.astype(np.int32); jj = jj.astype(np.int32)
        dense = dense_matrix[ii, jj]
        lexical = lexical_matrix[ii, jj]
        p1_labels = _labels_from_partition(dataset.initial_partition, n)
        same_p1 = p1_labels[ii] == p1_labels[jj]

        pair_index_matrix = np.full((n, n), -1, dtype=np.int32)
        indices = np.arange(ii.size, dtype=np.int32)
        pair_index_matrix[ii, jj] = indices
        pair_index_matrix[jj, ii] = indices

        dense01 = np.clip((dense + 1.0) * 0.5, 0.0, 1.0)
        combined = 0.55 * dense01 + 0.45 * lexical
        combined_order = np.lexsort((jj, ii, -combined)).astype(np.int32)
        dense_order = np.lexsort((jj, ii, -dense)).astype(np.int32)
        lexical_order = np.lexsort((jj, ii, -lexical)).astype(np.int32)
        exploration_order = _van_der_corput_rank_order(ii.size)

        k = max(1, min(int(neighbor_k), max(1, n - 1)))
        dense_neighbors = np.empty((n, k), dtype=np.int32)
        lexical_neighbors = np.empty((n, k), dtype=np.int32)
        ids = np.arange(n, dtype=np.int32)
        for u in range(n):
            
            ds = dense_matrix[u].copy(); ds[u] = -np.inf
            ls = lexical_matrix[u].copy(); ls[u] = -np.inf
            dense_neighbors[u] = np.lexsort((ids, -ds))[:k]
            lexical_neighbors[u] = np.lexsort((ids, -ls))[:k]

        meta = expected
        obj = cls(n=n, i=ii, j=jj, dense=dense.astype(np.float32), lexical=lexical.astype(np.float32),
                  same_p1=same_p1, dense_matrix=dense_matrix, lexical_matrix=lexical_matrix,
                  pair_index_matrix=pair_index_matrix, combined_order=combined_order,
                  dense_order=dense_order, lexical_order=lexical_order,
                  exploration_order=exploration_order, dense_neighbors=dense_neighbors,
                  lexical_neighbors=lexical_neighbors, metadata=meta)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                metadata_json=np.asarray(json.dumps(meta, sort_keys=True)),
                i=obj.i, j=obj.j, dense=obj.dense, lexical=obj.lexical,
                same_p1=obj.same_p1.astype(np.uint8), dense_matrix=obj.dense_matrix,
                lexical_matrix=obj.lexical_matrix, pair_index_matrix=obj.pair_index_matrix,
                combined_order=obj.combined_order, dense_order=obj.dense_order,
                lexical_order=obj.lexical_order, exploration_order=obj.exploration_order,
                dense_neighbors=obj.dense_neighbors, lexical_neighbors=obj.lexical_neighbors,
            )
        return obj
