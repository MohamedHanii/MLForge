# Notebooks: Final Assignment Set B

End-to-end experiments that use the MLForge models on two UCI datasets. The original assignment notebook (`final-assignment/SetB.ipynb`) was split into one notebook per track:

| Notebook | Track | Dataset | Baseline | Required model |
|---|---|---|---|---|
| [`breast_cancer_classification.ipynb`](breast_cancer_classification.ipynb) | Classification | [UCI Breast Cancer](https://archive.ics.uci.edu/dataset/14/breast+cancer) (id 14, Ljubljana) | `LogisticRegression` | `SVC` with RBF kernel |
| [`concrete_strength_regression.ipynb`](concrete_strength_regression.ipynb) | Regression | [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (id 165) | `LinearRegression` (Ridge) | `MLPRegressor` (ANN) |

**Assignment rule:** every *model* comes from our own library. scikit-learn is used only for preprocessing and splitting (`StandardScaler`, `train_test_split`, `KFold`, `StratifiedKFold`). `imbalanced-learn` is used for resampling.

---

## Task structure (both notebooks)

| Task | Content |
|---|---|
| 1. Prepare data | Download with `ucimlrepo`, explore, audit data quality, clean, encode, and create the held-out test split plus 10-fold CV on the development set |
| 2. Baseline | Load the baseline from the library and run leak-free 10-fold CV (the scaler is fit inside each fold) |
| 3. Baseline on test | Accuracy, precision, recall and F1, or R², MAE and RMSE |
| 4. Required model | RBF-kernel SVM (classification) or ANN (regression) |
| 5. Hyperparameter optimisation | Grid search ranked by CV score |
| 6. Stability test | 20 shuffled 90/10 resampling runs, reported as mean ± std with box plots |

---

## Breast Cancer: classification

**Data.** 286 patients × 9 categorical features. The target is recurrence, with 85 positives (**29.7 %**), so the classes are imbalanced. Missing values in `node-caps` (8) and `breast-quad` (1) are filled with the column mode. After ordinal and one-hot encoding there are 15 features. The split is a stratified 228-sample dev set and a 58-sample test set.

**Pipeline highlights**
- A reusable `plot_confusion` helper draws every confusion matrix.
- Class imbalance is shown before and after SMOTE. The SVM itself is rebalanced by random over-sampling *inside each training fold*.
- The default SVM (`C=1`) collapses toward the majority class, because the library's kernel SVM has **no bias term** (see [svm/README](../src/svm/README.md#notes-and-pitfalls)). The notebook fixes this in three ways:
  1. A grid search over `(C, γ)` ranked by **balanced accuracy**
  2. In-fold over-sampling
  3. A **prior-matched decision threshold** on `decision_function`

**Test-set results (n = 58)**

| Model | Accuracy | Precision | Recall | F1 | Balanced acc. |
|---|---|---|---|---|---|
| LogisticRegression baseline | 0.672 | 0.458 | 0.647 | 0.537 | 0.665 |
| SVM RBF, default `C=1` | 0.724 | 0.571 | 0.235 | 0.333 | 0.581 |
| SVM RBF, `C=50`, `γ=0.03`, over-sampled, threshold 0 | – | 0.474 | 0.529 | 0.500 | 0.643 |
| SVM RBF, same, prior-matched threshold | – | 0.370 | 0.588 | 0.455 | 0.587 |

**Stability (20 × 90/10 runs, mean ± std)**

| Model | F1 | Accuracy | Balanced acc. |
|---|---|---|---|
| LogReg baseline | 0.475 ± 0.155 | 0.648 ± 0.084 | 0.628 ± 0.117 |
| SVM RBF (rebalanced) | 0.442 ± 0.173 | 0.686 ± 0.105 | 0.613 ± 0.126 |

The 58-sample test set is a noisy draw. The notebook treats the 10-fold CV and the stability test as the more reliable estimates. On this small, noisy dataset, the logistic baseline is as good as the tuned SVM.

---

## Concrete Compressive Strength: regression

**Data.** 1030 mixes × 8 numeric features: cement, slag, fly ash, water, superplasticizer, coarse and fine aggregate, and age. The target is compressive strength in MPa. Cleaning drops 25 exact-duplicate rows, which would leak between splits, leaving 1005 rows. A multicollinearity check (|r| > 0.85) finds no feature pairs to drop. The split is an 804-sample dev set and a 201-sample test set.

**Results**

| Model | CV R² (10-fold) | Test R² | Test MAE (MPa) | Test RMSE (MPa) |
|---|---|---|---|---|
| Ridge `LinearRegression(alpha=0.1)` | 0.585 ± 0.071 | 0.580 | 8.90 | 11.19 |
| `MLPRegressor((64, 32), relu, lr=0.01)` | **0.845 ± 0.041** | **0.861** | **4.78** | **6.43** |

**Grid search (6 combinations × 10-fold CV).** `lr = 0.01` is clearly better than `0.05` for every architecture, and `(64, 32)` ranks best by a small margin over `(32,)` and `(128, 64)`.

**Stability (20 × 90/10 runs).** Linear R² 0.605 ± 0.070 (RMSE 10.16 ± 0.63) vs ANN R² **0.851 ± 0.044** (RMSE 6.19 ± 0.61). The ANN reduces the error by about 40 %.

---

## Running the notebooks

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn ucimlrepo jupyter
```

The first code cell also `pip install`s any of these that are missing. The datasets download from UCI at runtime, so you need internet access.

### 2. Point the loaders at `src/`

The notebooks were written against the **original per-assignment library folders**. A helper `_find_libs()` searches upward from the working directory for a `libraries/` (or `libraries/mlab/`) folder containing `MLRegKit/` and `logitlab/`. **That layout isn't in this repository**, so the loader cells raise `FileNotFoundError: Could not locate final-assignment/libraries/` as shipped.

| Notebook loads | Equivalent in this repo |
|---|---|
| `logitlab/mlab/regression/_logistic.py` | `src/regression/_logistic.py` |
| `MLRegKit/mlab/regression/_linear.py` | `src/regression/_linear.py` |
| `MLRegKit/mlab/regression/metrics.py` (`r2_score`, `mae`, `rmse`) | *not included*; see the snippet below |
| `svm-lib` package `mlab.svm` | `src/svm/` |
| `neurokit/mlab/neural_networks/_mlp.py` | `src/neural_networks/_mlp.py` |

To run against `src/`, make the edits below. The snippets define the same names the later cells use (including `LIBS`, which a later cell prints), so nothing else needs to change.

**Breast cancer**: replace the whole Task 2 cell, *Load the baseline model & metrics from `mlab`*, with:

```python
import sys, pathlib
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
LIBS = ROOT / "src"

import src.regression._logistic as clf               # precision_score / recall_score / f1_score / confusion_matrix
from src.regression._logistic import LogisticRegression
```

Then replace the whole Task 4 cell, *Load the SVM model from `mlab`*, which references `LIBS / "svm-lib"`, with:

```python
from src.svm._svc import SVC
```

**Concrete**: replace the whole Task 2 cell, *Load the regression baseline model & metrics from `mlab`*, with the snippet below. Then delete the body of the Task 4 cell *Load the ANN (MLPRegressor) from `mlab`*, because the snippet already defines `MLPRegressor` and `ANN_EPOCHS`.

```python
import sys, pathlib
import numpy as np
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
LIBS = ROOT / "src"

from src.regression._linear import LinearRegression
from src.neural_networks._mlp import MLPRegressor

class reg:                                            # stand-in for MLRegKit's metrics.py
    @staticmethod
    def r2_score(y, p):
        y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 0.0 if ss_tot == 0 else 1 - np.sum((y - p) ** 2) / ss_tot
    @staticmethod
    def mae(y, p):
        return float(np.mean(np.abs(np.asarray(y, float).ravel() - np.asarray(p, float).ravel())))
    @staticmethod
    def rmse(y, p):
        return float(np.sqrt(np.mean((np.asarray(y, float).ravel() - np.asarray(p, float).ravel()) ** 2)))

ANN_EPOCHS = 300
```

> The modules in `src/` are the library's final versions. If they differ from the per-assignment copies used to produce the saved outputs, re-running may give slightly different numbers.

### 3. Run

```bash
jupyter notebook notebooks/
```

Run the cells top to bottom. Each notebook is self-contained.
