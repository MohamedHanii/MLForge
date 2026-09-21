"""K-Means clustering with K-Means++ initialization (Lloyd's algorithm)."""

from __future__ import annotations

import numbers
from typing import Any, Optional

import numpy as np

_FLOAT64_EPS = float(np.finfo(np.float64).eps)

# Generate by LLMs
def _check_finite_array(arr: np.ndarray, name: str = "X") -> None:
    """Validate that an array contains no NaN or infinity values.

    A gatekeeper check used before clustering operations — non-finite
    values silently corrupt distance calculations and centroid updates,
    producing meaningless results.
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf).")


# Generate by LLMs
def _pairwise_sq_distances(X, centers):
    """
    Squared Euclidean distances between rows of X and rows of centers.
    X: (n_samples, n_features), centers: (n_clusters, n_features)
    Returns (n_samples, n_clusters). Vectorized via (X @ C.T) and broadcasting.
    """
    xc = X @ centers.T
    xx = np.sum(X * X, axis=1, keepdims=True)
    cc = np.sum(centers * centers, axis=1)
    d2 = xx + cc - 2.0 * xc
    return np.maximum(d2, 0.0)


def _labels_from_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each sample to its nearest centroid (argmin of squared distances).

    For each row x_i in X, finds the index of the centroid with the
    smallest squared Euclidean distance. Returns an array of cluster
    labels with shape (n_samples,). This is the "assignment" step of
    Lloyd's algorithm.
    """
    return np.argmin(_pairwise_sq_distances(X, centers), axis=1)

# Generate by LLMs
def _assign_and_update(
    X: np.ndarray, centers: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Perform one full iteration of Lloyd's algorithm: assign then update.

    Lloyd's algorithm alternates between two steps each iteration:
    1. Assignment — every point is assigned to the cluster whose centroid
       is nearest (measured by squared Euclidean distance).
    2. Update — each centroid is moved to the arithmetic mean of all
       points assigned to it (the cluster means).

    Empty clusters keep their previous centroid position (they are not
    updated). Returns the updated centroid array of the same shape.
    """
    d2 = _pairwise_sq_distances(X, centers)
    labels = np.argmin(d2, axis=1)
    cluster_sums = np.zeros_like(centers)
    np.add.at(cluster_sums, labels, X)
    counts = np.bincount(labels, minlength=n_clusters)
    new_centroids = centers.copy()
    nonempty = counts > 0
    new_centroids[nonempty] = (
        cluster_sums[nonempty] / counts[nonempty, np.newaxis]
    )
    return new_centroids


def _wcss(X: np.ndarray, centers: np.ndarray) -> float:
    """Within-Cluster Sum of Squares — the K-Means cost function.

    WCSS = Σᵢ ||xᵢ - μ_{c(i)}||²

    where c(i) is the cluster label assigned to point xᵢ and μ_{c(i)}
    is that cluster's centroid. Minimizing WCSS is the optimization
    objective of K-Means. Lower values indicate tighter, more compact
    clusters.

    Also known as inertia or distortion.
    """
    d2 = _pairwise_sq_distances(X, centers)
    lbl = np.argmin(d2, axis=1)
    return float(d2[np.arange(X.shape[0]), lbl].sum())

# Generate by LLMs
def _minibatch_update_epoch(
    X: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
    batch_indices: list[np.ndarray],
    center_counts: np.ndarray,
) -> None:
    """In-place centroid update over one epoch of mini-batches.

    Each mini-batch updates centroids using an exponential moving average
    rather than a full recomputation from all data. The per-cluster
    learning rate is:
        lr = n_k / (count_k + n_k)
    where n_k is the batch points assigned to cluster k and count_k is
    the cumulative count seen so far. This lets centroids converge
    gradually without storing full assignment history.
    """
    for batch_inds in batch_indices:
        Xb = X[batch_inds]
        lbl = _labels_from_centers(Xb, centroids)
        sums = np.zeros_like(centroids)
        np.add.at(sums, lbl, Xb)
        cnts = np.bincount(lbl, minlength=n_clusters)
        for k in range(n_clusters):
            nk = int(cnts[k])
            if nk == 0:
                continue
            mean_b = sums[k] / max(nk, _FLOAT64_EPS)
            lr = nk / (center_counts[k] + nk + _FLOAT64_EPS)
            centroids[k] *= 1.0 - lr
            centroids[k] += lr * mean_b
            center_counts[k] += nk


def _init_centroids_kmeans_plus_plus(X, n_clusters, rng):
    """K-Means++ initialization — selects initial centroids to be far apart.

    Instead of picking all centroids uniformly at random, K-Means++
    chooses them sequentially with probability proportional to the
    squared distance D(x)² from the nearest already-chosen centroid.
    This spreads centroids across the data, significantly improving
    convergence speed and final cluster quality compared to purely
    random initialization.

    Algorithm:
    1. Pick the first centroid uniformly at random.
    2. For each subsequent centroid, compute D(xₙ)² = the minimum
       squared distance from xₙ to any existing centroid.
    3. Sample the next centroid with probability D(xₙ)² / Σ D(x)².
    4. Repeat until k centroids are chosen.
    """
    n_samples, n_features = X.shape
    centroids = np.empty((n_clusters, n_features), dtype=np.float64)
    centroids[0] = X[rng.integers(n_samples)].copy()

    for j in range(1, n_clusters):
        diff = X[:, np.newaxis, :] - centroids[:j][np.newaxis, :, :]
        d2 = np.min(np.sum(diff * diff, axis=2), axis=1)
        total = d2.sum()
        if total <= 0.0 or not np.isfinite(total):
            centroids[j] = X[rng.integers(n_samples)].copy()
        else:
            centroids[j] = X[rng.choice(n_samples, p=d2 / total)].copy()

    return centroids


class KMeans:
    """K-Means clustering with K-Means++ initialization (Lloyd's algorithm).

    K-Means partitions n samples into k clusters by iteratively:
    1. Assigning each point to the nearest centroid (cluster center).
    2. Updating each centroid as the mean of its assigned points.

    The algorithm minimizes Within-Cluster Sum of Squares (WCSS), also
    called inertia — the total squared distance from each point to its
    assigned centroid.

    Supports full-batch Lloyd iterations (default) and mini-batch
    training (when batch_size is set) for large datasets where computing
    distances to all centroids per iteration is too expensive.
    """

    def __init__(
        self,
        n_clusters=3,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        batch_size=None,
        **kwargs,
    ):
        """Initialize the K-Means model.

        Parameters
        ----------
        n_clusters : int, default=3
            Number of clusters (k) to form.
        max_iter : int, default=300
            Maximum number of Lloyd iterations (or mini-batch epochs).
        tol : float, default=1e-4
            Convergence threshold — training halts when the Euclidean
            distance by which centroids moved in an iteration falls
            below this value.
        random_state : int or Generator, optional
            Seed for reproducible K-Means++ initialization and
            mini-batch shuffling.
        batch_size : int or None, default=None
            If set, enables mini-batch K-Means with this number of
            samples per batch instead of full-batch Lloyd's algorithm.
        """
        max_iter = kwargs.pop("max_iterations", max_iter)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.batch_size = batch_size
        self._fit: Optional[dict[str, Any]] = None
        self._validate_params()

    # Generate by LLMs  
    def _validate_params(self):
        """Validate that all constructor arguments have sensible values.

        Checks: n_clusters, max_iter, and batch_size are positive
        integers; tol is non-negative; random_state is None, an int,
        or a Generator. Called automatically during __init__ to fail
        fast on invalid configuration.
        """
        if not isinstance(self.n_clusters, (int, np.integer)) or int(self.n_clusters) < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.tol, numbers.Real) or float(self.tol) < 0:
            raise ValueError("tol must be a non-negative number.")
        if self.random_state is not None and not isinstance(
            self.random_state,
            (int, np.integer, np.random.Generator),
        ):
            raise ValueError(
                "random_state must be None, an integer, or a numpy.random.Generator."
            )
        if self.batch_size is not None:
            if (
                not isinstance(self.batch_size, (int, np.integer))
                or int(self.batch_size) < 1
            ):
                raise ValueError("batch_size must be a positive integer or None.")

    # Refactored by LLMs  
    @property
    def cluster_centers_(self):
        """Final centroid positions of shape (n_clusters, n_features)."""
        return None if self._fit is None else self._fit["cluster_centers_"]

    @property
    def labels_(self):
        """Cluster label (0 to n_clusters-1) for each training sample."""
        return None if self._fit is None else self._fit["labels_"]

    @property
    def inertia_(self):
        """Final Within-Cluster Sum of Squares (WCSS) after fitting."""
        return None if self._fit is None else self._fit["inertia_"]

    @property
    def wcss_history(self):
        """WCSS recorded at the end of each iteration — tracks convergence."""
        return [] if self._fit is None else self._fit["wcss_history"]

    @property
    def centroid_history_(self):
        """Centroid positions at the start of each iteration."""
        return None if self._fit is None else self._fit["centroid_history_"]

    @property
    def centroid_movement_history_(self):
        """Euclidean distance centroids moved at each iteration."""
        return None if self._fit is None else self._fit["centroid_movement_history_"]

    @property
    def n_iter_(self):
        """Number of iterations actually performed before convergence."""
        return 0 if self._fit is None else self._fit["n_iter_"]

    @property
    def centroids(self):
        """Alias for cluster_centers_."""
        return self.cluster_centers_

    @property
    def labels(self):
        """Alias for labels_."""
        return self.labels_

    def fit(self, X):
        """Train the K-Means model on data X using Lloyd's algorithm.

        Initializes centroids via K-Means++, then iterates up to max_iter
        times. Each iteration:
        1. Assigns every point to the nearest centroid.
        2. Recomputes each centroid as the mean of its assigned points.
        3. Records WCSS and centroid movement for convergence tracking.

        Early stopping: when centroid movement drops below tol, training
        halts. If batch_size is set, mini-batch K-Means is used instead
        of full-batch Lloyd iterations.

        Results are stored in self._fit and accessible via properties
        (labels_, cluster_centers_, inertia_, wcss_history, etc.).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data — must be finite and 2D.

        Returns
        -------
        self : KMeans
            The fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        _check_finite_array(X)
        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n_samples})."
            )

        rng = np.random.default_rng(self.random_state)
        centroids = np.asarray(
            _init_centroids_kmeans_plus_plus(X, self.n_clusters, rng),
            dtype=np.float64,
        ).copy()
        wcss_history: list[float] = []
        centroid_history: list[np.ndarray] = []
        centroid_movement_history: list[float] = []

        bs = self.batch_size
        use_minibatch = bs is not None and int(bs) < n_samples

        for _ in range(self.max_iter):
            old_centroids = centroids.copy()
            centroid_history.append(old_centroids.copy())

            if use_minibatch:
                center_counts = np.zeros(self.n_clusters, dtype=np.float64)
                perm = rng.permutation(n_samples)
                n_batches = int(np.ceil(n_samples / int(bs)))
                batches = np.array_split(perm, n_batches)
                _minibatch_update_epoch(
                    X, centroids, self.n_clusters, batches, center_counts
                )
            else:
                centroids = _assign_and_update(X, old_centroids, self.n_clusters)

            wcss_history.append(_wcss(X, centroids))
            centroid_movement = float(np.linalg.norm(centroids - old_centroids))
            centroid_movement_history.append(centroid_movement)

            if centroid_movement < float(self.tol):
                break

        labels = _labels_from_centers(X, centroids)
        inertia = float(np.sum((X - centroids[labels]) ** 2))
        self._fit = {
            "cluster_centers_": centroids,
            "labels_": labels,
            "inertia_": inertia,
            "wcss_history": wcss_history,
            "centroid_history_": centroid_history,
            "centroid_movement_history_": centroid_movement_history,
            "n_iter_": len(centroid_movement_history),
        }
        return self

    def predict(self, X):
        """Assign new samples to the nearest cluster centroid.

        For each row in X, computes its squared Euclidean distance to
        every fitted centroid and returns the index of the closest one
        (the cluster label). This is the assignment step of Lloyd's
        algorithm applied to unseen data.

        Requires fit() to have been called first.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points to assign to clusters.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster index (0 to n_clusters-1) for each sample.
        """
        if self._fit is None:
            raise ValueError("Model is not fitted; call fit first.")
        X = np.asarray(X, dtype=np.float64)
        _check_finite_array(X)
        return _labels_from_centers(X, self.cluster_centers_)
