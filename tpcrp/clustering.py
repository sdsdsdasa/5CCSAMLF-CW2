"""
K-means clustering and uncovered cluster selection for TPC RP.

Paper details (Appendix F):
  - max_clusters = 500 for CIFAR-10 (prevents over-clustering)
  - K = min(|L| + B, max_clusters)
  - Use KMeans when K <= 50, MiniBatchKMeans otherwise
  - Drop clusters with < 5 samples
  - Use min(20, cluster_size) nearest neighbours for typicality
  - Select iteratively: pick largest uncovered cluster with size > 5,
    compute typicality within that cluster, add most typical point
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

MAX_CLUSTERS = 500   # paper's cap for CIFAR-10
MIN_CLUSTER_SIZE = 5  # drop clusters smaller than this


def run_kmeans(embeddings, n_clusters, seed=42):
    """
    Run K-means on embeddings.
    Uses KMeans when n_clusters <= 50, MiniBatchKMeans otherwise (paper).

    Args:
        embeddings: np.ndarray (N, D)
        n_clusters: int — min(|L| + B, MAX_CLUSTERS)
        seed:       random seed

    Returns:
        cluster_labels: np.ndarray (N,) cluster assignment for each sample
    """
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters, random_state=seed,
                    n_init=10, max_iter=300)
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             n_init=3, max_iter=300)
    km.fit(embeddings)
    return km.labels_


def _typicality_in_cluster(embeddings, member_positions):
    """
    Compute typicality for members of a single cluster.
    Uses min(20, cluster_size) nearest neighbours as per the paper.
    Returns scores array aligned with member_positions.
    """
    k = min(20, len(member_positions))
    if k < 2:
        return np.ones(len(member_positions))

    cluster_embs = embeddings[member_positions]
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(cluster_embs)
    distances, _ = nn.kneighbors(cluster_embs)
    mean_dist = distances.mean(axis=1)
    mean_dist = np.where(mean_dist == 0, 1e-10, mean_dist)
    return 1.0 / mean_dist


def select_query_indices(cluster_labels, embeddings, labeled_indices,
                         all_indices, unlabeled_positions, budget,
                         uncertainty_scores=None):
    """
    Iteratively select B samples to query (paper's iterative procedure):
      For each query slot:
        1. Find clusters with fewest labeled points and size > MIN_CLUSTER_SIZE
        2. Among those, select the largest cluster
        3. Compute typicality within that cluster using min(20, size) neighbours
        4. Add the most typical unlabeled point to the query set

    Args:
        cluster_labels:     np.ndarray (N,) — cluster id per position in all_indices
        embeddings:         np.ndarray (N, D) — embeddings aligned with all_indices
        labeled_indices:    list of original dataset indices that are labeled
        all_indices:        list of all original dataset indices (labeled + unlabeled)
        unlabeled_positions: list of positions (into N arrays) that are unlabeled
        budget:             B — number of samples to query
        uncertainty_scores: optional np.ndarray (N,) in [0, 1] — per-sample
                            classifier entropy, aligned with all_indices positions.
                            When provided, final score = typicality × uncertainty.

    Returns:
        selected_positions: list of positions to query
    """
    idx_to_pos = {orig: pos for pos, orig in enumerate(all_indices)}

    # Track which positions are currently labeled (grows as we select)
    labeled_pos_set = set(idx_to_pos[i] for i in labeled_indices
                          if i in idx_to_pos)
    remaining_unlabeled = set(unlabeled_positions)

    # Build cluster → member positions map (all members, labeled + unlabeled)
    n = len(cluster_labels)
    cluster_to_members = {}
    for pos, cid in enumerate(cluster_labels):
        cluster_to_members.setdefault(cid, []).append(pos)

    # Drop clusters smaller than MIN_CLUSTER_SIZE
    valid_clusters = {cid for cid, members in cluster_to_members.items()
                      if len(members) >= MIN_CLUSTER_SIZE}

    selected = []
    for _ in range(budget):
        if not remaining_unlabeled:
            break

        # Count labeled points per cluster
        labeled_count = {
            cid: sum(1 for p in cluster_to_members[cid] if p in labeled_pos_set)
            for cid in valid_clusters
        }

        # Clusters that have unlabeled members
        eligible = {
            cid for cid in valid_clusters
            if any(p in remaining_unlabeled for p in cluster_to_members[cid])
        }
        if not eligible:
            break

        # Select cluster with fewest labeled points; break ties by largest size
        min_labeled = min(labeled_count[cid] for cid in eligible)
        candidates = [cid for cid in eligible
                      if labeled_count[cid] == min_labeled]
        chosen_cluster = max(candidates,
                             key=lambda c: len(cluster_to_members[c]))

        # Unlabeled members of chosen cluster
        unlabeled_members = [p for p in cluster_to_members[chosen_cluster]
                             if p in remaining_unlabeled]

        # Typicality within this cluster (all members, not just unlabeled)
        all_members = cluster_to_members[chosen_cluster]
        scores = _typicality_in_cluster(embeddings, all_members)
        member_to_score = {p: s for p, s in zip(all_members, scores)}

        # Optionally weight by classifier uncertainty (typicality × entropy)
        if uncertainty_scores is not None:
            for p in all_members:
                member_to_score[p] *= uncertainty_scores[p]

        # Pick most typical (and most uncertain) unlabeled member
        best = max(unlabeled_members, key=lambda p: member_to_score[p])
        selected.append(best)
        labeled_pos_set.add(best)
        remaining_unlabeled.discard(best)

    return selected
