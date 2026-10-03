# `clustering`: K-Means

`_kmeans.py` implements `KMeans`: Lloyd's algorithm with **K-Means++** initialisation and an optional **mini-batch** mode for large datasets. Every step of training is recorded, so you can plot or inspect convergence.

Dependencies: `numpy`.

---

## How it works

1. **K-Means++ initialisation.** The first centroid is a random sample. Each later centroid is drawn with probability proportional to its squared distance from the nearest centroid already chosen, which spreads the starting centroids apart.
2. **Lloyd iterations** (full-batch, the default):
   - *Assign:* each point joins its nearest centroid. Squared Euclidean distances are computed in vectorised form as `‖x‖² + ‖c‖² − 2x·c`.
   - *Update:* each centroid moves to the mean of its points. An empty cluster keeps its previous centroid.
3. **Mini-batch mode** (`batch_size=int`). Each epoch shuffles the data into batches. For each batch, every centroid moves toward the mean of its batch points with learning rate `n_k / (count_k + n_k)`, where `n_k` is the number of batch points assigned to it and `count_k` is the number it has seen so far.
4. **Convergence.** Training stops when the centroids move less than `tol` (Euclidean distance) in one iteration, or after `max_iter` iterations.

The objective is the **within-cluster sum of squares** (WCSS, or *inertia*): `Σᵢ ‖xᵢ − c_{label(i)}‖²`.

---

## API

### `KMeans(n_clusters=3, max_iter=300, tol=1e-4, random_state=None, batch_size=None)`

| Parameter | Description |
|---|---|
| `n_clusters` | Number of clusters *k* (a positive integer). |
| `max_iter` | Maximum number of Lloyd iterations or mini-batch epochs. `max_iterations=` is accepted as an alias. |
| `tol` | Stop when the centroids move less than this distance. |
| `random_state` | `None`, an `int`, or a `numpy.random.Generator`. |
| `batch_size` | `None` for full-batch Lloyd, or an `int` for mini-batch K-Means. |

Invalid arguments raise `ValueError` in the constructor. `fit` rejects NaN and ∞.

| Method / attribute | Description |
|---|---|
| `fit(X)` → `self` | Clusters the data. |
| `predict(X)` | Index of the nearest centroid for new samples. |
| `cluster_centers_` (alias `centroids`) | `(n_clusters, n_features)` final centroids. |
| `labels_` (alias `labels`) | Cluster of each training sample. |
| `inertia_` | Final WCSS. |
| `n_iter_` | Number of iterations run. |
| `wcss_history` | WCSS after each iteration. |
| `centroid_history_` | Centroid positions at the start of each iteration. |
| `centroid_movement_history_` | How far the centroids moved in each iteration. |

---

## Example

```python
import numpy as np
from src.clustering._kmeans import KMeans

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(loc, 0.5, size=(100, 2)) for loc in ([0, 0], [4, 0], [2, 3])])

km = KMeans(n_clusters=3, random_state=0).fit(X)
print("centroids:\n", km.cluster_centers_.round(2))
print("inertia:", round(km.inertia_, 2), "| iterations:", km.n_iter_)
print("new points ->", km.predict(np.array([[0, 0], [4, 0]])))

# Elbow method: inertia for several k
for k in range(1, 6):
    print(k, round(KMeans(n_clusters=k, random_state=0).fit(X).inertia_, 1))

# Mini-batch variant for large data
mbk = KMeans(n_clusters=3, batch_size=64, random_state=0).fit(X)
print("mini-batch inertia:", round(mbk.inertia_, 2))
```

---

## Notes

- K-Means uses Euclidean distance, so standardise features that are measured in different units.
- Results depend on the initialisation. Fix `random_state` to make them reproducible.
