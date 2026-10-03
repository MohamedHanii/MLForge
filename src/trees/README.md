# `trees`: decision tree classifier

`_decision_tree.py` implements `DecisionTree`, a binary-split classification tree built from scratch. [`ensemble.RandomForest`](../ensemble/README.md) uses it as its base learner.

Dependencies: `numpy`.

---

## How it works

- **Split criterion: information gain.** Each split is chosen to maximise the drop in Shannon entropy, `H = −Σ pₖ log₂ pₖ`, weighted by the sample count of each child.
- **Numeric thresholds.** For every candidate feature, the tree sorts the values and tests the midpoints between consecutive distinct values. It keeps the split `X[:, f] <= threshold` with the highest gain, the same way scikit-learn does.
- **Categorical features** must be label-encoded as integers first.
- **Stopping rules.** A node becomes a leaf when it reaches `max_depth`, has fewer than `min_samples_split` samples, is pure, or no split gives positive gain. Each leaf predicts its majority class.
- **Feature subsampling.** When `max_features` is set, each node draws a random subset of features to search. A random forest relies on this to de-correlate its trees.

---

## API

### `DecisionTree(max_depth=5, min_samples_split=2, max_features=None, random_state=None)`

| Parameter | Description |
|---|---|
| `max_depth` | Maximum depth. The root is at depth 0. |
| `min_samples_split` | Minimum number of samples a node needs before it can split. |
| `max_features` | `None` uses all features. `"sqrt"`, `"log2"`, an `int` (count) or a `float` (fraction) uses a random subset per node. |
| `random_state` | Seed for the feature subsampling. |

| Method / attribute | Description |
|---|---|
| `fit(X, y)` → `self` | Grows the tree. Labels can be any sortable values. |
| `predict(X)` | Predicted class labels. |
| `get_depth()` | Depth of the fitted tree (a single leaf has depth 0). |
| `classes_` | Sorted class labels seen in `fit`. |
| `feature_importances_` | Total sample-weighted information gain per feature, normalised to sum to 1. This is the same definition scikit-learn uses. |
| `tree_depth` | Same value as `get_depth()`. |
| `root_` | The root `_Node` (`feature_index`, `threshold`, `left`, `right`, `value`). |

---

## Example

```python
import numpy as np
from src.trees._decision_tree import DecisionTree

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

tree = DecisionTree(max_depth=4, min_samples_split=5).fit(X, y)
print("train accuracy:", np.mean(tree.predict(X) == y))
print("depth:", tree.get_depth())
print("importances:", tree.feature_importances_.round(3))   # features 0 and 1 dominate
```

---

## Notes

- This is a classifier only. There's no regression tree.
- There's no pruning. Control overfitting with `max_depth` and `min_samples_split`, or switch to [`RandomForest`](../ensemble/README.md).
- Trees don't depend on feature scale, so you don't need to standardise.
