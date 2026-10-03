# `ensemble`: random forest classifier

`_random_forest.py` implements `RandomForest`, a bagged ensemble of [`trees.DecisionTree`](../trees/README.md) classifiers that predicts by majority vote.

Dependencies: `numpy` and `src/trees/_decision_tree.py`, imported relatively (`from ..trees._decision_tree import DecisionTree`). Import it as a package from the repository root:

```python
from src.ensemble._random_forest import RandomForest
```

---

## How it works

1. **Bootstrap.** Each tree trains on a sample of `n` rows drawn with replacement (when `bootstrap=True`).
2. **Random feature subsets.** Each tree gets `max_features` (default `"sqrt"`), so every node searches only a random subset of the features. This de-correlates the trees.
3. **Majority vote.** `predict` collects every tree's prediction and returns the most frequent class for each sample.
4. **Out-of-bag (OOB) estimate.** With `oob_score=True`, each training sample is scored only by the trees that didn't see it. This gives an accuracy estimate without a separate validation set.
5. **Feature importances** are the per-tree importances averaged across the forest.

---

## API

### `RandomForest(n_estimators=20, max_depth=5, min_samples_split=2, max_features="sqrt", bootstrap=True, oob_score=False, random_state=None)`

| Parameter | Description |
|---|---|
| `n_estimators` | Number of trees. |
| `max_depth`, `min_samples_split` | Passed to every `DecisionTree`. |
| `max_features` | `"sqrt"`, `"log2"`, `int`, `float` or `None`. `None` means plain bagging. |
| `bootstrap` | Sample rows with replacement for each tree. OOB scoring requires it. |
| `oob_score` | Compute `oob_score_` during `fit`. |
| `random_state` | Seed for the bootstrap samples and feature subsets. |

| Method / attribute | Description |
|---|---|
| `fit(X, y)` → `self` | Trains the forest. |
| `predict(X)` | Majority-vote class labels. |
| `trees_` | List of fitted `DecisionTree`s. |
| `classes_` | Sorted class labels. |
| `feature_importances_` | Mean of the trees' importances (sums to 1). |
| `oob_score_` | OOB accuracy (only when `oob_score=True`). |

---

## Example

```python
import numpy as np
from src.ensemble._random_forest import RandomForest

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 6))
y = (X[:, 0] * X[:, 1] > 0).astype(int)        # non-linear (XOR-like) boundary

forest = RandomForest(n_estimators=50, max_depth=6, oob_score=True, random_state=0).fit(X, y)
print("OOB accuracy:", round(forest.oob_score_, 3))
print("importances:", forest.feature_importances_.round(3))
print("predictions:", forest.predict(X[:5]))
```

---

## Notes

- This is a classifier only, and it has no `score` method. Compute accuracy with `np.mean(forest.predict(X) == y)`.
- Training time grows linearly with `n_estimators`. The trees are pure NumPy and train one after another.
