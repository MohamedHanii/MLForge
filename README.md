# MLForge

Classic machine-learning algorithms built from scratch with NumPy. The interfaces follow scikit-learn's `fit` / `predict` style.

MLForge is the model library from the **Machine Learning Lab (Summer Semester 2026, University of Passau)**. Each algorithm is written in plain NumPy, with no scikit-learn inside the models. The docstrings explain the math, so you can read the code to learn how each method works. Two end-to-end notebooks use the library on real UCI datasets.

---

## Contents

- [Algorithms at a glance](#algorithms-at-a-glance)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Quickstart](#quickstart)
- [Shared conventions](#shared-conventions)
- [Preprocessing: who scales what](#preprocessing-who-scales-what)
- [Utilities (metrics, scalers, splitters)](#utilities-metrics-scalers-splitters)
- [Notebooks](#notebooks)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Project notes](#project-notes)

---

## Algorithms at a glance

| Package | Estimators | Task | Core technique | Docs |
|---|---|---|---|---|
| `regression` | `LinearRegression` (Ridge), `SGDRegression` | Regression | Closed-form normal equation with L2; mini-batch SGD | [src/regression](src/regression/README.md) |
| `regression` | `LogisticRegression`, `SGDClassifier` | Binary classification | Weighted cross-entropy + L2, full-batch GD / mini-batch SGD, early stopping | [src/regression](src/regression/README.md) |
| `trees` | `DecisionTree` | Classification | Greedy binary splits that maximise information gain (entropy) | [src/trees](src/trees/README.md) |
| `ensemble` | `RandomForest` | Classification | Bagged `DecisionTree`s with random feature subsets, majority vote, OOB score | [src/ensemble](src/ensemble/README.md) |
| `naive_bayes` | `GaussianNaiveBayes`, `MultinomialNaiveBayes` | Classification | Bayes' rule in log space; Gaussian likelihoods / Laplace-smoothed counts | [src/naive_bayes](src/naive_bayes/README.md) |
| `clustering` | `KMeans` | Clustering | Lloyd's algorithm, K-Means++ init, optional mini-batch mode | [src/clustering](src/clustering/README.md) |
| `svm` | `SVC` / `SVM`, `SVR` | Classification, regression | Primal GD (linear) / dual Pegasos (RBF, poly), one-vs-rest, ε-insensitive loss | [src/svm](src/svm/README.md) |
| `neural_networks` | `MLPRegressor` | Regression | Modular dense layers, backprop, L2, gradient clipping, early stopping | [src/neural_networks](src/neural_networks/README.md) |
| `neural_networks` | `CNNClassifier` | Image classification | Conv / max-pool / ReLU / softmax layers, momentum SGD, cross-entropy | [src/neural_networks](src/neural_networks/README.md) |

---

## Repository layout

```
MLForge/
├── README.md                     ← you are here
├── src/                          ← the library (one sub-package per algorithm family)
│   ├── clustering/
│   │   └── _kmeans.py            KMeans (Lloyd + K-Means++, mini-batch)
│   ├── ensemble/
│   │   └── _random_forest.py     RandomForest (uses trees.DecisionTree)
│   ├── naive_bayes/
│   │   ├── _naive_bayes.py       GaussianNaiveBayes, MultinomialNaiveBayes
│   │   └── _eval_utils.py        multi-class confusion matrix / accuracy / P / R / F1
│   ├── neural_networks/
│   │   ├── _mlp.py               MLPRegressor + dense/activation layers
│   │   └── _cnn.py               CNNClassifier + conv/pool/activation layers
│   ├── regression/
│   │   ├── _linear.py            LinearRegression (Ridge), SGDRegression, StandardScaler
│   │   └── _logistic.py          LogisticRegression, SGDClassifier, split/resample/metrics
│   ├── svm/
│   │   ├── _svm.py               SVM base estimator, StandardScaler, errors
│   │   ├── _svc.py               SVC (classifier alias of SVM)
│   │   ├── _svr.py               SVR (ε-SVR)
│   │   └── _svm_utils.py         kernels, KFold, StratifiedKFold, GridSearchCV, metrics
│   └── trees/
│       └── _decision_tree.py     DecisionTree (information gain)
└── notebooks/                    ← end-to-end experiments on UCI datasets
    ├── breast_cancer_classification.ipynb
    ├── concrete_strength_regression.ipynb
    └── library_tests/            ← one test notebook per package + run_all.py
```

Each package folder has its own `README.md` with the full API, a worked example and implementation notes. See [notebooks/README.md](notebooks/README.md) for the experiments.

---

## Getting started

### Requirements

| Use | Packages |
|---|---|
| Library (`src/`) | Python ≥ 3.9 (tested on 3.12), `numpy` |
| `regression/_logistic.py` only | also `matplotlib` (imported at module load for `plot_decision_boundary`) |
| Notebooks | `numpy`, `pandas`, `matplotlib`, `scikit-learn` (preprocessing/splitting only), `imbalanced-learn`, `ucimlrepo`, `jupyter` |

### Install

```bash
git clone <repo-url> MLForge
cd MLForge
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib                    # library
pip install pandas scikit-learn imbalanced-learn ucimlrepo jupyter   # notebooks (optional)
```

You don't need to `pip install` MLForge itself. Run your code from the repository root and import from `src`.

### Importing

`src/` and its sub-folders have no `__init__.py` files, so Python treats them as *namespace packages*. Run Python from the repository root, or put the root on `sys.path`, and import the private module directly:

```python
from src.regression._linear import LinearRegression
from src.svm._svc import SVC
from src.ensemble._random_forest import RandomForest
```

> `src/svm/` and `src/ensemble/` use relative imports (`from ._svm import SVM`, `from ..trees._decision_tree import DecisionTree`). Import them as packages, as shown above. Loading those files by path with `importlib.util.spec_from_file_location` fails. The other modules have no internal imports and load either way.

---

## Quickstart

```python
import numpy as np
from src.regression._linear import LinearRegression
from src.regression._logistic import train_test_split
from src.svm._svc import SVC
from src.svm._svm_utils import GridSearchCV, StratifiedKFold, classification_report

rng = np.random.default_rng(0)

# --- Regression ---------------------------------------------------------
X = rng.normal(size=(200, 4))
y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + 0.1 * rng.normal(size=200)
ridge = LinearRegression(alpha=0.1).fit(X, y)
print("R²:", ridge.score(X, y))

# --- Classification with model selection --------------------------------
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

search = GridSearchCV(
    SVC(kernel="rbf"),
    {"C": [0.1, 1, 10], "gamma": [0.1, None]},
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
).fit(X_tr, y_tr)

print(search.best_params_, search.best_score_)
print(classification_report(y_te, search.best_estimator_.predict(X_te)))
```

---

## Shared conventions

All estimators follow the same scikit-learn-style contract:

- **`fit(X, y)` → `self`** so calls can be chained (`Model().fit(X, y).predict(X)`).
  *Exception:* `GaussianNaiveBayes.fit` and `MultinomialNaiveBayes.fit` return `None`.
- **`predict(X)`** returns labels (classifiers), values (regressors) or cluster ids (`KMeans`).
- **`score(X, y)`** returns accuracy for classifiers and R² for regressors. `MLPRegressor` and the Naive Bayes classes have no `score` method.
- **Learned attributes end in `_`** (`weights_`, `classes_`, `feature_importances_`, `cluster_centers_`, `loss_curve_`, …).
- **`random_state`** seeds every stochastic step. The SVM family uses a fixed seed of `42` when `random_state=None`, so its results are always deterministic.
- **Inputs:** `X` is `(n_samples, n_features)` (NHWC images for the CNN) and `y` is `(n_samples,)`. Most models reject NaN/∞ with a clear `ValueError`.

---

## Preprocessing: who scales what

Some models standardise their inputs internally and some expect you to do it. Scaling twice is harmless but redundant. Forgetting to scale for SGD-based models slows convergence or makes it diverge.

| Estimator | Scales `X` internally? | Notes |
|---|---|---|
| `LinearRegression` (Ridge) | No | Closed-form, so it works unscaled. Scale anyway if you want the L2 penalty to treat features equally. |
| `SGDRegression` | No | **Scale first**, as the docstring requires. |
| `LogisticRegression`, `SGDClassifier` | No | Scale first for good convergence. |
| `SVC` / `SVM`, `SVR` | **Yes** (`X`; `SVR` also scales `y` and un-scales predictions) | Pass raw features. |
| `MLPRegressor` | **Yes** (`X` and `y`) | Pass raw features. |
| `CNNClassifier` | No | Normalise pixel ranges yourself (e.g. divide by 255). |
| `DecisionTree`, `RandomForest`, Naive Bayes | No, and not needed | Tree splits don't depend on scale. |
| `KMeans` | No | Distance-based, so scale features that use different units. |

---

## Utilities (metrics, scalers, splitters)

Each assignment library shipped its own helpers, so some helpers exist in several places. Use the one that matches your task:

| Helper | Location(s) | Notes |
|---|---|---|
| `StandardScaler` | `regression/_linear.py` (`with_mean`, `with_std` options; divides by `σ + 1e-7`), `regression/_logistic.py`, `svm/_svm.py` | All compute z-scores and treat constant features safely. |
| `train_test_split` | `regression/_logistic.py` | Stratified by default (`stratify='auto'`). Returns `X_train, X_test, y_train, y_test`. |
| `resample` | `regression/_logistic.py` | Balances binary classes by under-sampling (`'down'`) or over-sampling (`'up'`). |
| `KFold`, `StratifiedKFold`, `GridSearchCV` | `svm/_svm_utils.py` | `GridSearchCV` works with any estimator that has `get_params` / `set_params` / `score`. |
| `rbf_kernel`, `poly_kernel` | `svm/_svm_utils.py` | Vectorised kernel matrices. |
| Binary metrics + `roc_auc_score`, `classification_report`, `plot_decision_boundary` | `regression/_logistic.py` | For 0/1 labels. |
| Multi-class metrics (`average='binary' / 'macro' / 'weighted'`), `evaluate_classification_model` | `naive_bayes/_eval_utils.py` | `'weighted'` assumes integer labels `0…k-1`. |
| Metrics with `pos_label`, `confusion_matrix` + label order, per-class `classification_report` | `svm/_svm_utils.py` | `pos_label` defaults to the second sorted label. |
| `check_array`, `check_X_y`, `check_is_fitted`, `RegressorMixin` (R² `score`) | `regression/_linear.py` | Validation helpers. |

---

## Notebooks

The [`notebooks/`](notebooks/README.md) folder holds the final assignment (Set B), split into two tracks:

| Notebook | Dataset | Baseline → required model | Test-set headline |
|---|---|---|---|
| [`breast_cancer_classification.ipynb`](notebooks/breast_cancer_classification.ipynb) | UCI Breast Cancer (id 14), 286 × 9 | `LogisticRegression` → RBF `SVC` | Balanced accuracy 0.665 (LogReg) vs 0.643 (rebalanced SVM) |
| [`concrete_strength_regression.ipynb`](notebooks/concrete_strength_regression.ipynb) | UCI Concrete Compressive Strength (id 165), 1005 × 8 | Ridge `LinearRegression` → `MLPRegressor` | R² 0.580 → **0.861**, RMSE 11.19 → 6.43 MPa |

> ⚠️ The notebooks were written against the original per-assignment library folders (`logitlab/`, `MLRegKit/`, `svm-lib/`, `neurokit/`), not against `src/`. To re-run them, read [notebooks/README.md → Running the notebooks](notebooks/README.md#running-the-notebooks).

---

## Testing

[`notebooks/library_tests/`](notebooks/library_tests/README.md) holds one notebook per package. Together they make 390 checks covering:

- known answers on generated data
- exact agreement with scikit-learn where the maths is identical
- numerical gradient checks for every neural-network layer
- performance on real datasets, both bundled and downloaded from UCI or OpenML

```bash
python notebooks/library_tests/run_all.py          # ≈ 2 min; exits non-zero if any check fails
```

The suite also records **6 known library bugs** (🐞). They are listed in the [test README](notebooks/library_tests/README.md#known-library-bugs-found-by-these-notebooks) and summarised below.

---

## Known limitations

- **Not an installable package.** There's no `pyproject.toml`, `requirements.txt` or `__init__.py`. Import from the repository root (see [Importing](#importing)).
- **Logistic regression is binary-only.** Labels must be `0`/`1`. Multi-class labels don't raise an error, but the predictions are meaningless. Use `SVC`, `RandomForest`, `DecisionTree` or Naive Bayes for more than two classes.
- **Logistic hyperparameters are fixed in code:** L2 `λ = 0.1`, `class_weight='balanced'` and `tol = 1e-5` are set in `_BaseLogistic.__init__` and aren't constructor arguments.
- **The kernel SVM has no bias term.** The RBF/poly decision function is `Σ αᵢ yᵢ K(xᵢ, x)` with no intercept. On imbalanced data it can collapse to the majority class, so rebalance the training data or tune the decision threshold. The [breast-cancer notebook](notebooks/README.md) shows both.
- **`MLPRegressor`'s gradient is n× too large.** It backpropagates `out − y` instead of `(out − y)/n`, so the effective learning rate grows with the dataset size. Use `lr ≈ 1/n_train` until this is fixed.
- **`SVC` rejects string labels** (`TypeError`). Encode labels as integers.
- **The CNN is pure NumPy** and runs on the CPU. Keep inputs small (e.g. 8×8 or 28×28 digits).
- **Some helpers are duplicated across packages** (see [Utilities](#utilities-metrics-scalers-splitters)).

---

## Project notes

- **Course context:** Machine Learning Lab, Summer Semester 2026, University of Passau. The notebooks only use models from this library, as the assignment rules require. scikit-learn is used only for preprocessing and splitting.
- **AI assistance:** helper functions marked `# Generate by LLMs` / `# Generated By LLM` in the source were drafted with AI help and then integrated by the author.
- **License:** none specified yet.
