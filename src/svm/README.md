# `svm`: support vector machines

Soft-margin SVMs for classification (`SVC`) and regression (`SVR`) with linear, RBF and polynomial kernels, plus the model-selection toolkit around them: K-fold splitters, grid search, kernels and metrics.

| File | Contents |
|---|---|
| `_svm.py` | `SVM` base estimator (it is the classifier), `StandardScaler`, `NotFittedError`, `ConvergenceWarning` |
| `_svc.py` | `SVC`, a thin alias of `SVM` |
| `_svr.py` | `SVR`, ε-insensitive support vector regression |
| `_svm_utils.py` | `KFold`, `StratifiedKFold`, `GridSearchCV`, `rbf_kernel`, `poly_kernel`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report` |

Dependencies: `numpy`. These modules use relative imports, so import them as a package from the repository root:

```python
from src.svm._svc import SVC
from src.svm._svr import SVR
from src.svm._svm_utils import GridSearchCV, StratifiedKFold
```

---

## How it works

| Case | Solver |
|---|---|
| `kernel='linear'` | **Primal** stochastic gradient descent on the hinge loss (classifier) or ε-insensitive loss (regressor), with `λ = 1 / (C·m)`. Learns `weight_` and `bias_`. |
| `kernel='rbf'` / `'poly'` | **Dual Pegasos** sub-gradient updates on the precomputed kernel matrix. Learns dual coefficients `alpha_`. The decision function is `f(x) = Σⱼ αⱼ yⱼ K(xⱼ, x)`. |
| More than 2 classes | **One-vs-rest:** one binary sub-model per class, predicting the class with the highest decision score. |

Kernels:

```
RBF:   K(x, y) = exp(−γ ‖x − y‖²)
Poly:  K(x, y) = (γ ⟨x, y⟩ + 1)^degree
γ = 1 / n_features when gamma=None
```

**Built-in preprocessing.** `fit` standardises `X` with its own `StandardScaler`, and `predict` reuses those statistics. `SVR` also standardises `y` and maps predictions back to the original scale, so `epsilon` is in units of the target's standard deviation. **Pass raw features.**

**Determinism.** The RNG is re-seeded at every `fit`. `random_state=None` means a fixed seed of `42`, so repeated fits always give identical results.

Internal solver settings, which aren't constructor arguments: learning rate `0.01`, `tol = 1e-4`, `max_iter = 1000` epochs. If the solver doesn't converge, it emits a `ConvergenceWarning`.

---

## API

### `SVC(kernel='rbf', C=1.0, degree=3, gamma=None, random_state=None)` (`SVM` is identical)

| Parameter | Description |
|---|---|
| `kernel` | `'linear'`, `'rbf'` or `'poly'`. |
| `C` | Inverse regularisation strength. Must be > 0. Larger values mean a harder margin. |
| `degree` | Polynomial degree (`'poly'` only). |
| `gamma` | Kernel coefficient. Must be > 0. `None` means `1 / n_features`. |
| `random_state` | Seed. `None` means `42`. |

| Method / attribute | Description |
|---|---|
| `fit(X, y)` → `self` | Numeric labels with at least 2 classes (string labels currently raise `TypeError`, see below). |
| `decision_function(X)` | Binary: signed scores `(n,)`. Multi-class: `(n, n_classes)` one-vs-rest scores. |
| `predict(X)` | Original class labels. |
| `score(X, y)` | Accuracy. |
| `get_params()` / `set_params(**p)` | scikit-learn style, used by `GridSearchCV`. |
| `classes_`, `weight_`, `bias_`, `intercept_`, `alpha_` | Learned state (which ones exist depends on the kernel). |
| `support_`, `support_vectors_`, `n_support_` | Support-vector indices, vectors and count. |

Calling `predict` or `decision_function` before `fit` raises `NotFittedError`.

### `SVR(kernel='rbf', C=1.0, epsilon=0.1, degree=3, gamma=None, random_state=None)`

The same parameters as `SVC`, plus `epsilon` (≥ 0), the half-width of the ε-insensitive tube in standardised-`y` units. `predict` returns values on the original target scale, and `score` returns R².

### `_svm_utils.py`

| Name | Description |
|---|---|
| `KFold(n_splits=5, shuffle=False, random_state=None)` | `.split(X)` yields `(train_idx, test_idx)` pairs. |
| `StratifiedKFold(n_splits=5, shuffle=False, random_state=None)` | Like `KFold`, but keeps the class proportions in every fold. `.split(X, y)`. |
| `GridSearchCV(estimator, param_grid, cv=None, scoring=None)` | Exhaustive grid search ranked by `estimator.score`. When `cv` is `None` or an `int`, classifiers get a shuffled `StratifiedKFold` and regressors get a shuffled `KFold`, both seeded with `42`. After `fit`, it exposes `best_params_`, `best_score_`, `best_estimator_` (refit on all the data) and `cv_results_` (`{'params', 'mean_test_score'}`). `scoring` isn't used yet. |
| `rbf_kernel(X, Y=None, gamma=None)`, `poly_kernel(X, Y=None, degree=3, gamma=None, coef0=1.0)` | Vectorised kernel matrices. |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | `pos_label` defaults to the second sorted label. |
| `confusion_matrix(y_true, y_pred)` | Returns `(matrix, labels)`. |
| `classification_report(y_true, y_pred)` | Text table with per-class P/R/F1/support and overall accuracy. |

> `GridSearchCV` indexes `X[train_idx]`, so pass NumPy arrays, not pandas DataFrames (use `df.values`).

---

## Example

```python
import numpy as np
from src.svm._svc import SVC
from src.svm._svr import SVR
from src.svm._svm_utils import GridSearchCV, StratifiedKFold, classification_report

rng = np.random.default_rng(0)

# Non-linear binary classification (circle)
X = rng.normal(size=(200, 2))
y = (np.sum(X**2, axis=1) > 1.0).astype(int)

svc = SVC(kernel="rbf", C=1.0, random_state=0).fit(X, y)   # raw features are fine
print("accuracy:", svc.score(X, y), "| support vectors:", svc.n_support_)

# Hyperparameter search
search = GridSearchCV(
    SVC(kernel="rbf"),
    {"C": [0.1, 1, 10], "gamma": [0.1, 0.5, None]},
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
).fit(X, y)
print(search.best_params_, round(search.best_score_, 3))
print(classification_report(y, search.best_estimator_.predict(X)))

# Multi-class (automatic one-vs-rest)
y3 = np.digitize(X[:, 0], [-0.5, 0.5])
print("3-class accuracy:", SVC(kernel="rbf").fit(X, y3).score(X, y3))

# Regression
Xr = rng.uniform(-3, 3, size=(200, 1))
yr = np.sin(Xr).ravel() + 0.1 * rng.normal(size=200)
svr = SVR(kernel="rbf", C=10.0, epsilon=0.1).fit(Xr, yr)
print("SVR R²:", round(svr.score(Xr, yr), 3))
```

---

## Notes and pitfalls

- **The kernel classifier has no bias term.** The RBF/poly decision function has no intercept, so on imbalanced data the boundary can collapse to the majority class (all predictions become one label). Possible fixes:
  - Rebalance the training folds, for example by random over-sampling of the minority class.
  - Move the decision threshold on `decision_function` instead of using `0`.
  - Tune `C` and `gamma` with a balanced-accuracy criterion.

  The [breast-cancer notebook](../../notebooks/README.md) works through all three.
- **Known bug: string labels.** `fit` checks `y` with `np.isnan`, which raises `TypeError` for string labels such as `'yes'` / `'no'`. Encode labels as integers first, for example with `np.unique(y, return_inverse=True)`.
- The kernel path stores the full `n × n` training kernel matrix, so memory grows quadratically with the number of samples.
- `SVC` exists for API familiarity. `SVM` is the same classifier.
