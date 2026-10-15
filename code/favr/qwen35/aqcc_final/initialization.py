"""Reproducible, Gold-free character TF-IDF and hierarchical P1 preparation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import scipy
from scipy import sparse
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .data import load_start_state, read_json, sha256_file


CORPUS_FIELDS = frozenset({"id", "text", "sourceExcelrow"})
SILHOUETTE_TIE_TOLERANCE = 1e-12
PROTOCOL = "CHAR_2_5_TFIDF_COSINE_AVERAGE_HC_ALL_K_SILHOUETTE_V1"


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def tfidf_parameters() -> dict:
    """Return a fresh complete constructor configuration, with no learned state."""
    return {
        "analyzer": "char", "ngram_range": (2, 5), "min_df": 1,
        "max_df": 1.0, "max_features": 50000, "sublinear_tf": True,
        "norm": "l2", "dtype": np.float32, "lowercase": True,
        "smooth_idf": True, "use_idf": True, "binary": False,
        "strip_accents": None, "stop_words": None,
    }


def read_corpus(path: Path) -> list[dict]:
    rows = read_json(Path(path))
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("initialization requires at least 3 reports; silhouette K=2..n-1 is undefined otherwise")
    seen_ids = set()
    previous_excel_row = 1
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != CORPUS_FIELDS:
            raise ValueError("corpus rows must contain exactly id, text, sourceExcelrow; annotation fields are forbidden")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen_ids:
            raise ValueError(f"invalid or duplicate report ID at corpus index {index}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"empty original report text at corpus index {index}")
        excel_row = row["sourceExcelrow"]
        if type(excel_row) is not int or excel_row <= previous_excel_row:
            raise ValueError("sourceExcelrow must be strictly increasing integers starting at 2 or later")
        seen_ids.add(row["id"])
        previous_excel_row = excel_row
    return rows


def fit_tfidf(rows: list[dict]) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(**tfidf_parameters())
    try:
        vectors = vectorizer.fit_transform([row["text"] for row in rows]).tocsr()
    except ValueError as exc:
        raise ValueError("character 2-5 TF-IDF cannot represent this corpus") from exc
    vectors.sum_duplicates()
    vectors.sort_indices()
    if vectors.dtype != np.float32 or not np.isfinite(vectors.data).all():
        raise ValueError("TF-IDF must contain finite float32 values")
    row_norm_squared = np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel()
    missing = np.flatnonzero(row_norm_squared <= 0)
    if missing.size:
        ids = [rows[int(index)]["id"] for index in missing]
        raise ValueError(f"zero TF-IDF row vectors; no reports were dropped: {ids}")
    return vectorizer, vectors


def cosine_distances(vectors: sparse.csr_matrix) -> np.ndarray:
    """Compute cosine on saved sparse features, with explicit float64 symmetry.

    Re-normalizing a float64 view removes float32 storage norm roundoff. It
    does not change or replace the saved float32 TF-IDF report representation.
    """
    unit = normalize(vectors.astype(np.float64), norm="l2", axis=1, copy=True)
    similarity = (unit @ unit.T).toarray()
    similarity = np.clip((similarity + similarity.T) * 0.5, 0.0, 1.0)
    distances = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distances, 0.0)
    return distances


@dataclass(frozen=True)
class HierarchicalSelection:
    linkage_matrix: np.ndarray
    condensed_distances: np.ndarray
    candidate_k: np.ndarray
    labels_by_candidate_k: np.ndarray
    candidates: list[dict]
    selected_k: int
    selected_score: float
    selected_labels: np.ndarray
    maximum_score: float


def select_hc_partition(distances: np.ndarray) -> HierarchicalSelection:
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.ndim != 2 or distances.shape[0] != distances.shape[1]
            or distances.shape[0] < 3 or not np.isfinite(distances).all()
            or np.any(distances < 0) or np.any(distances > 1)
            or not np.array_equal(distances, distances.T)
            or np.any(np.diag(distances) != 0)):
        raise ValueError("silhouette initialization requires a finite symmetric cosine distance matrix for at least 3 reports")
    n = distances.shape[0]
    condensed = squareform(distances, checks=True)
    tree = linkage(condensed, method="average", optimal_ordering=False)
    candidate_k = np.arange(2, n, dtype=np.int32)
    
    
    cuts = cut_tree(tree, n_clusters=candidate_k).T.astype(np.int32, copy=False)
    candidates = []
    for k, labels in zip(candidate_k, cuts):
        counts = np.bincount(labels)
        if len(counts) != int(k) or np.any(counts <= 0) or int(counts.sum()) != n:
            raise ValueError("hierarchical cut did not preserve an exact K-cluster population")
        score = float(silhouette_score(distances, labels, metric="precomputed", sample_size=None))
        if not np.isfinite(score):
            raise ValueError("silhouette score is not finite")
        candidates.append({"k": int(k), "silhouette": score,
                           "cluster_sizes": counts.tolist(),
                           "singleton_clusters": int(np.count_nonzero(counts == 1))})
    maximum = max(row["silhouette"] for row in candidates)
    selected_index = next(index for index, row in enumerate(candidates)
                          if maximum - row["silhouette"] <= SILHOUETTE_TIE_TOLERANCE)
    selected = candidates[selected_index]
    return HierarchicalSelection(tree, condensed, candidate_k, cuts, candidates,
                                 selected["k"], selected["silhouette"], cuts[selected_index].copy(), maximum)


def partition_document(labels: np.ndarray, ids: list[str]) -> dict:
    blocks = []
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label).tolist()
        blocks.append(members)
    blocks.sort(key=lambda members: members[0])
    canonical_labels = [0] * len(ids)
    clusters = []
    for cluster_index, members in enumerate(blocks):
        for member in members:
            canonical_labels[member] = cluster_index
        clusters.append({"members": members, "member_source_ids": [ids[i] for i in members]})
    return {"labels_by_index": canonical_labels, "clusters": clusters}


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_tfidf_hc(*, corpus_path: Path, dataset_id: str, source_workbook: Path,
                     source_sheet: str, output_dir: Path) -> dict:
    """Create a new preparation directory and refuse any existing destination.

    The source workbook is only byte-hashed for provenance. The sole semantic
    input is the explicitly annotation-free three-field corpus JSON.
    """
    corpus_path, source_workbook = Path(corpus_path).resolve(), Path(source_workbook).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists; initialization refuses overwrite")
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset_id):
        raise ValueError("dataset_id must be a nonempty portable identifier")
    if not isinstance(source_sheet, str) or not source_sheet.strip():
        raise ValueError("source_sheet must be nonempty")
    corpus_file_sha256 = sha256_file(corpus_path)
    workbook_sha256 = sha256_file(source_workbook)
    rows = read_corpus(corpus_path)
    ids = [row["id"] for row in rows]
    vectorizer, vectors = fit_tfidf(rows)
    selection = select_hc_partition(cosine_distances(vectors))
    partition = partition_document(selection.selected_labels, ids)
    if sha256_file(corpus_path) != corpus_file_sha256 or sha256_file(source_workbook) != workbook_sha256:
        raise ValueError("source changed while preparing TF-IDF and P1")
    output_dir.mkdir(parents=True, exist_ok=False)
    vector_path = output_dir / "tfidf_vectors.npz"
    sparse.save_npz(vector_path, vectors, compressed=True)
    tree_path = output_dir / "hc_tree.npz"
    np.savez_compressed(tree_path, linkage_matrix=selection.linkage_matrix,
                        condensed_cosine_distances=selection.condensed_distances,
                        candidate_k=selection.candidate_k,
                        labels_by_candidate_k=selection.labels_by_candidate_k,
                        report_ids=np.asarray(ids),
                        source_excel_rows=np.asarray([row["sourceExcelrow"] for row in rows], dtype=np.int64))
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    parameters = vectorizer.get_params(deep=False)
    parameters["dtype"] = "float32"
    _write_json(output_dir / "tfidf_model.json", {
        "schema": "aqcc-character-tfidf-model-v1", "parameters": parameters,
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf_by_feature_index": vectorizer.idf_.tolist(), "library_versions": versions,
        "vectors_dtype": str(vectors.dtype), "vectors_format": "csr",
        "vectors_shape": list(vectors.shape), "vectors_nnz": int(vectors.nnz),
    })
    rule = {"objective": "MAXIMIZE_FULL_POPULATION_PRECOMPUTED_COSINE_SILHOUETTE",
            "candidate_k": "ALL_INTEGERS_2_THROUGH_N_MINUS_1", "subsampling": False,
            "tie_tolerance": SILHOUETTE_TIE_TOLERANCE, "tie_break": "FEWEST_CLUSTERS_WITHIN_TOLERANCE_OF_GLOBAL_MAX",
            "tree": "ONE_AVERAGE_LINKAGE_TREE_FROM_FROZEN_EXCEL_ROW_ORDER",
            "optimal_leaf_ordering": False}
    _write_json(output_dir / "silhouette_candidates.json", {
        "schema": "aqcc-hc-silhouette-candidates-v1", "rule": rule,
        "candidate_count": len(selection.candidates), "candidates": selection.candidates,
        "maximum_silhouette": selection.maximum_score,
        "selected_k": selection.selected_k, "selected_silhouette": selection.selected_score,
    })
    _write_json(output_dir / "selected_P1.json", {
        "schema": "aqcc-hc-selected-p1-v1", "report_ids": ids, "rule": rule,
        "selected_k": selection.selected_k, "selected_silhouette": selection.selected_score,
        "partition": partition,
    })
    provenance = {
        "protocol": PROTOCOL, "source_corpus": str(corpus_path),
        "source_corpus_sha256": corpus_file_sha256, "source_corpus_content_sha256": canonical_sha256(rows),
        "source_workbook": str(source_workbook), "source_workbook_sha256": workbook_sha256,
        "source_sheet": source_sheet, "report_ids_order_sha256": canonical_sha256(ids),
        "full_texts_order_sha256": canonical_sha256([{"id": row["id"], "text": row["text"]} for row in rows]),
        "source_excel_rows_order_sha256": canonical_sha256([row["sourceExcelrow"] for row in rows]),
        "text_policy": "EXACT_CORPUS_TEXT_NO_TRIMMING_NO_DEDUPLICATION_NO_GOLD_FILTERING",
        "gold_accessed": False, "workbook_read_policy": "BYTES_HASH_ONLY_NO_CELL_READING",
        "library_versions": versions,
    }
    state = {
        "schema": "aqcc-tfidf-hc-start-state-v1", "dataset_id": dataset_id,
        "report_ids": ids,
        "reports": [{"report_index": index, "source_id": row["id"], "description": row["text"],
                     "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                     "source_excel_row": row["sourceExcelrow"]}
                    for index, row in enumerate(rows)],
        "embedding_dimension": int(vectors.shape[1]),
        "embeddings_npz": {"format": "scipy_csr_npz", "path": vector_path.name,
                           "sha256": sha256_file(vector_path)},
        "partition": partition, "preparation": provenance,
    }
    state_path = output_dir / "start_state.json"
    _write_json(state_path, state)
    loaded = load_start_state(state_path)
    if (loaded.report_ids != tuple(ids) or loaded.texts != tuple(row["text"] for row in rows)
            or not sparse.isspmatrix_csr(loaded.embeddings) or (loaded.embeddings != vectors).nnz
            or loaded.partition_ids() != [cluster["member_source_ids"] for cluster in partition["clusters"]]):
        raise ValueError("prepared start-state round trip failed")
    files = ["tfidf_vectors.npz", "hc_tree.npz", "tfidf_model.json", "silhouette_candidates.json",
             "selected_P1.json", "start_state.json"]
    metadata = {
        "schema": "aqcc-tfidf-hc-preparation-v1", "status": "READY",
        "dataset_id": dataset_id, "reports": len(rows), "p1_clusters": selection.selected_k,
        "selected_silhouette": selection.selected_score, "candidate_k_count": len(selection.candidates),
        "selection_rule": rule, "tfidf_parameters": parameters,
        "representation": {"kind": "CHAR_2_5_TFIDF", "format": "scipy_csr_npz",
                           "dimension": int(vectors.shape[1]), "nnz": int(vectors.nnz),
                           "dtype": "float32", "svd_used": False, "dense_embedding_used": False},
        "cosine_numeric_policy": "Float64 L2 renormalization of saved float32 CSR vectors; symmetric dot products clipped to [0,1]; distance diagonal exactly zero.",
        "source_provenance": provenance, "population_preserved": True,
        "corpus_annotation_free_fields_verified": sorted(CORPUS_FIELDS),
        "start_state_sparse_round_trip_verified": True, "provider_calls": 0,
        "initializer_implementation_sha256": sha256_file(Path(__file__)),
        "output_files_sha256": {name: sha256_file(output_dir / name) for name in files},
    }
    _write_json(output_dir / "preparation_metadata.json", metadata)
    return metadata
