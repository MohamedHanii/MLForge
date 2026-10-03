# Library test notebooks

One notebook per MLForge package. Each one tests the package against data with known answers, against scikit-learn as a reference implementation, and against real datasets, both bundled and downloaded. The notebooks are tutorials (with plots and explanations) and also an executable test suite.

| Notebook | Package under test | Datasets | Checks |
|---|---|---|---|
| [`01_linear_regression`](01_linear_regression.ipynb) | `regression/_linear.py` | synthetic, Diabetes, **California Housing** 🌐 | 34 + 2 🐞 |
| [`02_logistic_regression`](02_logistic_regression.ipynb) | `regression/_logistic.py` | Breast Cancer, blobs, moons, imbalanced, **Banknote** 🌐 | 44 + 1 🐞 |
| [`03_decision_tree`](03_decision_tree.ipynb) | `trees/_decision_tree.py` | toy splits, circle, Iris, Wine, Breast Cancer, **Mushroom** 🌐 | 40 |
| [`04_random_forest`](04_random_forest.ipynb) | `ensemble/_random_forest.py` | moons, Breast Cancer, Wine, Iris, **Spambase** 🌐 | 24 |
| [`05_naive_bayes`](05_naive_bayes.ipynb) | `naive_bayes/` | Iris, Wine, blobs, Digits, **20 Newsgroups** 🌐 | 46 + 1 🐞 |
| [`06_kmeans`](06_kmeans.ipynb) | `clustering/_kmeans.py` | blobs, Iris, Digits, *China* photo, **Dry Bean** 🌐 | 33 |
| [`07_svm`](07_svm.ipynb) | `svm/` | blobs, moons, circles, sine, Breast Cancer, Iris, Diabetes, **Ionosphere** 🌐 | 72 + 1 🐞 |
| [`08_mlp`](08_mlp.ipynb) | `neural_networks/_mlp.py` | gradient checks, sine, Friedman #1, Diabetes, **Airfoil** 🌐 | 39 + 1 🐞 |
| [`09_cnn`](09_cnn.ipynb) | `neural_networks/_cnn.py` | gradient checks, bars, Digits, **MNIST** 🌐 | 58 |

🌐 = downloaded at run time (UCI via `ucimlrepo`, or scikit-learn / OpenML fetchers). Everything else is generated or ships with scikit-learn.

**Last full run:** 390 passed, 0 failed, 6 known library bugs, about 2 minutes on a laptop CPU.

---

## What each notebook checks

Every notebook follows the same pattern:

1. **Known answers.** Generated data where the correct result is known in advance: true regression coefficients, blob centres, a split threshold of exactly `6.5`.
2. **Reference comparison.** The same inputs go through the scikit-learn equivalent. Where the maths is identical (Ridge, Naive Bayes, kernels, metrics, scalers), results must match to machine precision. Where training is iterative, accuracy or R² must be within a stated tolerance.
3. **Gradient checks** (MLP, CNN). Every layer's `backward` is compared against central finite differences, and for the CNN so is the whole network.
4. **Real datasets.** Performance thresholds on bundled and online data, with plots.
5. **API contract.** `fit` returns `self`, determinism with `random_state`, input validation errors, and the attributes from the docs.

scikit-learn is used **only** to load data, as a reference oracle, and for text vectorisation (`CountVectorizer`). It is never the model being tested.

---

## Running

```bash
# from the repository root
pip install numpy pandas matplotlib scikit-learn ucimlrepo nbclient nbformat ipykernel

python notebooks/library_tests/run_all.py              # run everything, save outputs into the notebooks
python notebooks/library_tests/run_all.py 07 08        # just the SVM and MLP notebooks
python notebooks/library_tests/run_all.py --no-save    # run without touching the saved outputs
```

`run_all.py` prints a table of results and **exits with status 1 if any check fails**, so it can run in CI. You can also open any notebook in Jupyter and run it top to bottom. The last cell raises `AssertionError` if something failed.

The first run downloads about 30 MB of datasets (MNIST is most of it). Later runs use the local caches. **Offline**, the online sections are skipped (⏭️) instead of failing.

---

## Reading the results

| Icon | Meaning |
|---|---|
| ✅ | check passed |
| ❌ | check failed: the suite fails |
| ⏭️ | skipped, e.g. a dataset couldn't be downloaded |
| 🐞 | **known library bug**: the check documents correct behaviour that the library doesn't have yet. Reported, but doesn't fail the suite |
| 🎉 | a known bug now behaves correctly: turn it into a normal `checks.check(...)` |

### Known library bugs found by these notebooks

| Notebook | Bug | Workaround |
|---|---|---|
| 01 | `regression._linear.StandardScaler(with_mean=False)` / `(with_std=False)`: `fit` leaves the skipped statistic as `None`, so `transform` raises "not fitted" | use the defaults |
| 02 | `LogisticRegression` accepts multi-class labels silently and returns meaningless predictions | only use 0/1 labels |
| 05 | `evaluate_classification_model` on multi-class data reports **class 0's** precision, recall and F1 instead of an average | call `precision(..., average="macro")` etc. directly |
| 07 | `SVC.fit` raises `TypeError` for string labels (`np.isnan(y)` on strings) | encode labels as integers |
| 08 | `MLPRegressor.fit` backpropagates `out − y` instead of `(out − y)/n`, so the update is **n×** the gradient of the reported loss. The learning rate effectively scales with dataset size and `alpha` is effectively divided by n | use `lr ≈ 1 / n_train` |

---

## Writing more checks

All notebooks import [`_testkit.py`](_testkit.py):

```
from _testkit import SEED, Checks, load_online, timed

checks = Checks("10 my feature")
checks.check("name", condition)                        # truthy → pass
checks.close("name", actual, expected, rtol=, atol=)   # np.allclose, prints max |Δ|
checks.at_least("name", value, threshold)              # value ≥ threshold
checks.at_most("name", value, threshold)
checks.raises("name", ValueError, lambda: ...)
checks.known_bug("name", lambda: <True when correct>, "what is wrong")
data, online = load_online("Dataset", loader)          # (None, False) when offline → checks.skip(...)
checks.summary()                                       # last cell: raises if anything failed
```

Name new notebooks `NN_<topic>.ipynb` so `run_all.py` picks them up, and end them with a `checks.summary()` cell.
