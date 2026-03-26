"""
Typicality scoring as defined in the TPC RP paper.

Typicality(x) = ( (1/K) * sum_{xi in K-NN(x)} ||x - xi||_2 )^(-1)

Higher typicality = closer to dense regions = more "typical" sample.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(embeddings, k=20):
    """
    Compute typicality score for every sample in `embeddings`.

    Args:
        embeddings: np.ndarray of shape (N, D)
        k:          number of nearest neighbours (paper uses K=20)

    Returns:
        scores: np.ndarray of shape (N,), higher = more typical
    """
    # k+1 because the sample itself is its own nearest neighbour
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # Exclude the sample itself (distance = 0 at index 0)
    distances = distances[:, 1:]          # (N, K)
    mean_dist = distances.mean(axis=1)    # (N,)

    # Avoid division by zero for identical points
    mean_dist = np.where(mean_dist == 0, 1e-10, mean_dist)
    scores = 1.0 / mean_dist
    return scores
