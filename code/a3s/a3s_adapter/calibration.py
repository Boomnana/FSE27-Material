from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


def unsupervised_pair_probability(embeddings, pseudo_cluster_count: int, *, seed: int,
                                  neighbors: int = 100, return_artifacts: bool = False):
    """Official README calibration path using only K-means pseudo labels."""
    x = embeddings.tocsr() if sparse.issparse(embeddings) else np.asarray(embeddings)
    n = x.shape[0]
    if not 2 <= pseudo_cluster_count < n:
        raise ValueError("pseudo cluster count must be between 2 and N-1")
    kmeans = KMeans(n_clusters=pseudo_cluster_count, init="k-means++", n_init=10,
                    random_state=seed).fit(x)
    count = min(max(2, neighbors), n)
    distances, indices = NearestNeighbors(n_neighbors=count, metric="euclidean").fit(x).kneighbors(x)
    scores = -(distances ** 2).reshape(-1)
    anchor_labels = np.repeat(kmeans.labels_, count)
    neighbor_labels = kmeans.labels_[indices.reshape(-1)]
    targets = (anchor_labels == neighbor_labels).astype(np.int8)
    if len(np.unique(targets)) != 2:
        raise ValueError("pseudo-label neighbor pairs do not contain both relation classes")
    regressor = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores, targets)
    squared = euclidean_distances(x, x, squared=True)
    probability = regressor.predict((-squared).reshape(-1)).reshape(n, n)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0001, 0.9999)
    probability = (probability + probability.T) / 2.0
    np.fill_diagonal(probability, 0.9999)
    metadata = {
        "schema": "a3s-unsupervised-isotonic-calibration-v1",
        "uses_gold": False,
        "pseudo_label_method": "KMeans",
        "pseudo_cluster_count": int(pseudo_cluster_count),
        "kmeans_n_init": 10,
        "seed": int(seed),
        "neighbors": int(count),
        "training_pairs": int(scores.size),
        "pseudo_same_pairs": int(targets.sum()),
        "pseudo_different_pairs": int(targets.size - targets.sum()),
    }
    if not return_artifacts:
        return probability, metadata
    artifacts = {
        "pseudo_labels": np.asarray(kmeans.labels_, dtype=np.int32),
        "neighbor_indices": np.asarray(indices, dtype=np.int32),
        "neighbor_distances": np.asarray(distances, dtype=np.float64),
        "training_scores": np.asarray(scores, dtype=np.float64),
        "training_targets": np.asarray(targets, dtype=np.int8),
        "isotonic_x_thresholds": np.asarray(regressor.X_thresholds_, dtype=np.float64),
        "isotonic_y_thresholds": np.asarray(regressor.y_thresholds_, dtype=np.float64),
    }
    return probability, metadata, artifacts
