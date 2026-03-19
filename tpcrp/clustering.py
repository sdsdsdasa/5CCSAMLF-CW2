"""
K-means clustering and uncovered cluster selection for TPC RP.

At each AL iteration:
  - Cluster all data into |L| + B clusters
  - Uncovered clusters = clusters with no labeled examples
  - Select B largest uncovered clusters
  - Query the most typical sample from each selected cluster
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def run_kmeans(embeddings, n_clusters, seed=42):
    """
    Run K-means on embeddings.

    Args:
        embeddings: np.ndarray (N, D)
        n_clusters: int — |L| + B as per the paper
        seed:       random seed

    Returns:
        cluster_labels: np.ndarray (N,) cluster assignment for each sample
    """
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                         n_init=3, max_iter=300)
    km.fit(embeddings)
    return km.labels_


def find_uncovered_clusters(cluster_labels, labeled_indices, all_indices):
    """
    Find clusters that contain no currently labeled samples.

    Args:
        cluster_labels:  np.ndarray (N,) — cluster id for each sample in all_indices
        labeled_indices: list of indices (into the original dataset) that are labeled
        all_indices:     list of all indices (labeled + unlabeled) used for clustering

    Returns:
        uncovered: set of cluster ids that have no labeled samples
    """
    # Map original dataset index → position in all_indices array
    idx_to_pos = {orig: pos for pos, orig in enumerate(all_indices)}

    labeled_positions = [idx_to_pos[i] for i in labeled_indices
                         if i in idx_to_pos]
    labeled_clusters = set(cluster_labels[labeled_positions])

    all_clusters = set(cluster_labels)
    uncovered = all_clusters - labeled_clusters
    return uncovered


def select_query_indices(cluster_labels, typicality_scores, uncovered_clusters,
                         unlabeled_positions, budget):
    """
    From the B largest uncovered clusters, pick the most typical unlabeled sample.

    Args:
        cluster_labels:     np.ndarray (N,) — cluster id per position
        typicality_scores:  np.ndarray (N,) — typicality per position
        uncovered_clusters: set of cluster ids with no labeled samples
        unlabeled_positions: list of positions (into the N-length arrays) that are unlabeled
        budget:             B — number of samples to query

    Returns:
        selected_positions: list of positions (into the N-length arrays) to query
    """
    unlabeled_pos_set = set(unlabeled_positions)

    # For each uncovered cluster, collect its unlabeled members
    cluster_to_unlabeled = {}
    for cluster_id in uncovered_clusters:
        members = [p for p in unlabeled_positions
                   if cluster_labels[p] == cluster_id]
        if members:
            cluster_to_unlabeled[cluster_id] = members

    # Sort uncovered clusters by size descending, take top B
    sorted_clusters = sorted(cluster_to_unlabeled.keys(),
                             key=lambda c: len(cluster_to_unlabeled[c]),
                             reverse=True)
    top_clusters = sorted_clusters[:budget]

    # From each top cluster, pick the most typical unlabeled sample
    selected = []
    for cluster_id in top_clusters:
        members = cluster_to_unlabeled[cluster_id]
        best = max(members, key=lambda p: typicality_scores[p])
        selected.append(best)

    return selected
