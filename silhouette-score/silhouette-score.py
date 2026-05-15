import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Coefficient of all samples.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    N = len(X)

    # 1. Compute all-pairs Euclidean distances
    # X[:, np.newaxis, :] creates a (N, 1, D) array
    # X[np.newaxis, :, :] creates a (1, N, D) array
    # Broadcasting subtracts them to form an (N, N, D) array of differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    # 2. Isolate clusters and count their members
    classes, label_indices = np.unique(labels, return_inverse=True)
    
    # Create a mask mapping each point to its cluster: shape (K, N)
    masks = (labels == classes[:, np.newaxis]).astype(float)
    
    # Calculate sum of distances from each point to all points in each cluster
    # dist is (N, N) and masks.T is (N, K), resulting in (N, K)
    sum_dists = dist @ masks.T
    counts = masks.sum(axis=1)

    # 3. Compute intra-cluster average distance: a(i)
    own_sum_dists = sum_dists[np.arange(N), label_indices]
    
    # Divide by (count - 1) because we don't include the point's distance to itself (0)
    # np.maximum prevents division by zero if a cluster only has 1 point
    a = own_sum_dists / np.maximum(counts[label_indices] - 1, 1)

    # 4. Compute nearest inter-cluster average distance: b(i)
    # Average distance to all clusters: shape (N, K)
    avg_dists = sum_dists / counts
    
    # Temporarily set the distance to a point's OWN cluster to infinity
    # This forces .min() to only consider the "neighboring" clusters
    avg_dists[np.arange(N), label_indices] = np.inf
    b = avg_dists.min(axis=1)

    # 5. Compute the silhouette score for each point: s(i)
    max_ab = np.maximum(a, b)
    
    # Avoid division by zero if both a and b are 0
    s = np.where(max_ab > 0, (b - a) / max_ab, 0.0)

    # By convention, if a cluster has exactly 1 point, its silhouette score is 0
    s[counts[label_indices] == 1] = 0.0

    # The final score is the mean of all individual silhouette scores
    return float(np.mean(s))