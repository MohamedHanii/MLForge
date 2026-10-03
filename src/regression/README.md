# `regression`: linear and logistic models

Linear models for continuous targets (Ridge, SGD), linear models for **binary** classification (logistic regression with full-batch GD or mini-batch SGD), and the preprocessing, splitting and metric helpers that go with them.

| File | Contents |
|---|---|
| `_linear.py` | `LinearRegression` (Ridge), `SGDRegression`, `StandardScaler`, `RegressorMixin`, `check_array`, `check_X_y`, `check_is_fitted` |
| `_logistic.py` | `LogisticRegression`, `SGDClassifier`, `StandardScaler`, `train_test_split`, `resample`, binary metrics, `roc_auc_score`, `classification_report`, `plot_decision_boundary` |

Dependencies: `numpy`. `_logistic.py` also imports `matplotlib` when the module loads.

---

## Linear regression (`_linear.py`)

### `LinearRegression(alpha=0.1)`: Ridge regression

Linear regression with an L2 penalty, solved in closed form with the normal equation and a pseudo-inverse:

```
θ = pinv(X_bᵀ X_b + α I') X_bᵀ y        X_b = [1 | X],  I'[0,0] = 0
```

The intercept is **not** penalised. Set `alpha=0` for ordinary least squares.

| Member | Description |
|---|---|
| `alpha` | L2 strength (default `0.1`). |
| `fit(X, y)` → `self` | Solves for the weights. |
| `predict(X)` | `X @ coef_ + intercept_` |
| `score(X, y)` | R² (from `RegressorMixin`; returns `0.0` when `y` is constant). |
| `coef_`, `intercept_` | Learned feature weights and bias. |

### `SGDRegression(learning_rate=0.01, n_iterations=1000, batch_size=32, alpha=0.01)`

Linear regression trained with mini-batch SGD on MSE + L2:

```
w ← w − η [ (2/m) Xᵀ(ŷ − y) + 2α w ]       b ← b − η (2/m) Σ(ŷ − y)
```

Gradients are clipped to ±1e6. **Standardise `X` before fitting.** Learned state: `weights_`, `bias_`. `score` returns R².

### `StandardScaler(with_mean=True, with_std=True)`

Z-score scaler with `fit`, `transform` and `fit_transform`. It stores `mean_` and `std_`, and constant features get `std = 1`.

### Validation helpers

| Function | Purpose |
|---|---|
| `check_array(X, ensure_2d=True, ensure_min_samples=1)` | Rejects `None`, NaN and ∞, and reshapes 1-D input to a column. |
| `check_X_y(X, y)` | Runs `check_array` on `X` and checks that `X` and `y` have the same number of samples. |
| `check_is_fitted(estimator, attributes=None)` | Raises if the fitted attributes are missing. |

### Example

```python
import numpy as np
from src.regression._linear import LinearRegression, SGDRegression, StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + 3.0 + 0.1 * rng.normal(size=200)

ridge = LinearRegression(alpha=0.1).fit(X, y)
print(ridge.coef_.round(2), round(ridge.intercept_, 2), ridge.score(X, y))

Xs = StandardScaler().fit_transform(X)          # SGD needs scaled features
sgd = SGDRegression(learning_rate=0.01, n_iterations=1000, batch_size=32).fit(Xs, y)
print(sgd.score(Xs, y))
```

---

## Logistic regression (`_logistic.py`)

Both classifiers share a private `_BaseLogistic` base class. Its defaults are **fixed in code** and can't be passed to the constructor:

| Setting | Value | Effect |
|---|---|---|
| `lambda_` | `0.1` | L2 penalty `(λ / 2n)·‖w‖²` |
| `class_weight` | `'balanced'` | Each class is weighted `n / (k · count_c)`, so minority classes count more |
| `tol` | `1e-5` | Early stopping when the loss stops improving |

You can override `lambda_` and `tol` on an instance (for example `model.lambda_ = 0.0`) before calling `fit`. The `class_weight` attribute is never read, so balanced weighting is always applied.

> **Binary only.** Labels must be `0` / `1`. Multi-class labels don't raise an error, but the predictions are meaningless.

### `LogisticRegression(learning_rate=0.01, n_iterations=1000)`

Full-batch gradient descent on the weighted binary cross-entropy. The learning rate decays as `lr / (1 + 0.001·t)`.

### `SGDClassifier(learning_rate=0.01, n_iterations=1000, batch_size=32)`

Same objective trained with shuffled mini-batches. The learning rate decays per epoch as `lr / (1 + 0.01·epoch)`, and training stops early when the mean epoch loss plateaus.

### Shared API

| Member | Description |
|---|---|
| `fit(X, y)` → `self` | Trains the model. |
| `predict_proba(X)` | `(n, 2)` array `[P(y=0), P(y=1)]`. |
| `predict(X)` | `1` if `P(y=1) ≥ 0.5`, else `0`. |
| `score(X, y)` | Accuracy. |
| `weights_`, `bias_` | Learned parameters (`coef_`, `coefficients` and `theta` are aliases for `weights_`). |
| `loss_history_` | Loss per iteration/epoch. |

### Helpers

| Function | Description |
|---|---|
| `StandardScaler()` | Z-score scaler (`mean_`, `scale_`). |
| `train_test_split(X, y, test_size=0.2, random_state=None, stratify='auto')` | Returns `X_train, X_test, y_train, y_test`. `'auto'` stratifies by `y` and keeps at least one sample of each class in the test set. Pass `stratify=None` for a plain random split. |
| `resample(X, y, strategy='down', random_state=None)` | Balances a binary dataset. `'down'` under-samples the majority class and `'up'` over-samples the minority class. |
| `confusion_matrix(y_true, y_pred)` | 2×2 `[[TN, FP], [FN, TP]]`. |
| `precision_score`, `recall_score`, `f1_score` | Binary metrics for the positive class `1`. |
| `roc_auc_score(y_true, y_prob)` | AUC with the trapezoidal rule. |
| `classification_report(y_true, y_pred, target_names=None)` | Text report with per-class P/R/F1 and support. |
| `plot_decision_boundary(model, X, y, title=...)` | Plots the decision regions for 2-D `X` with matplotlib. |

### Example

```python
import numpy as np
from src.regression._logistic import (
    LogisticRegression, SGDClassifier, StandardScaler,
    train_test_split, classification_report, roc_auc_score,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 4))
y = (X[:, 0] - X[:, 2] + 0.3 * rng.normal(size=300) > 0.5).astype(int)   # imbalanced

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

clf = LogisticRegression(learning_rate=0.1, n_iterations=1000).fit(X_tr, y_tr)
print(classification_report(y_te, clf.predict(X_te)))
print("AUC:", roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

sgd = SGDClassifier(learning_rate=0.05, n_iterations=200, batch_size=32).fit(X_tr, y_tr)
print("SGD accuracy:", sgd.score(X_te, y_te))
```

---

## Notes

- `LinearRegression` doesn't scale its inputs. Scale them anyway if you want the penalty to treat all features equally.
- The project's [concrete-strength notebook](../../notebooks/README.md) uses `LinearRegression(alpha=0.1)` as its regression baseline (test R² 0.580). The [breast-cancer notebook](../../notebooks/README.md) uses `LogisticRegression(learning_rate=0.1, n_iterations=1000)` as its classification baseline.
