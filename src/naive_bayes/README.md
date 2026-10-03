# `naive_bayes`: Naive Bayes classifiers and evaluation metrics

| File | Contents |
|---|---|
| `_naive_bayes.py` | `GaussianNaiveBayes` (continuous features), `MultinomialNaiveBayes` (count features) |
| `_eval_utils.py` | Multi-class `confusion_matrix`, `accuracy`, `precision`, `recall`, `f1_score`, `evaluate_classification_model` |

Dependencies: `numpy`.

---

## How it works

Both classifiers apply Bayes' rule with the "naive" assumption that features are independent given the class:

```
log P(Cₖ | x) ∝ log P(Cₖ) + Σᵢ log P(xᵢ | Cₖ)
```

Everything is computed in **log space** for numerical stability. `predict_proba` subtracts the row maximum before exponentiating, which is a stable softmax.

| Model | Likelihood `P(xᵢ \| Cₖ)` | Use for |
|---|---|---|
| `GaussianNaiveBayes` | Normal density with per-class mean and variance (variance + `1e-9` to avoid division by zero) | Continuous, real-valued features |
| `MultinomialNaiveBayes(alpha=1.0)` | `(countₖᵢ + α) / (totalₖ + α·n_features)` (Laplace smoothing) | Non-negative counts, such as word counts in text |

---

## API

Both classes share the same methods:

| Method | Description |
|---|---|
| `fit(X, y)` | Learns the priors and likelihoods. **Returns `None`, so calls can't be chained.** |
| `predict(X)` | Most probable class per sample. |
| `predict_proba(X)` | `(n_samples, n_classes)` class probabilities. |
| `predict_log_proba(X)` | Log-probabilities. |

Fitted attributes:

| `GaussianNaiveBayes` | `MultinomialNaiveBayes` |
|---|---|
| `classes_`, `priors_`, `mean_`, `variance_` | `classes_`, `class_log_prior_`, `feature_log_prob_` |

Calling `predict*` before `fit` raises `RuntimeError`.

---

## Evaluation utilities (`_eval_utils.py`)

These work with any number of classes. Rows of the confusion matrix are true labels and columns are predicted labels.

| Function | Description |
|---|---|
| `confusion_matrix(y_true, y_pred, labels=None)` | `(k, k)` matrix. Labels are inferred from both arrays when not given. |
| `accuracy(y_true, y_pred)` | Fraction of correct predictions. |
| `precision`, `recall`, `f1_score` `(y_true, y_pred, average='binary', labels=None)` | `average` can be `'binary'` (class at index 1), `'macro'` or `'weighted'`. |
| `evaluate_classification_model(y_true, y_pred, labels=None, target_names=None)` | **Prints** a formatted confusion matrix and the metrics, and **returns** a dict with keys `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`. |

> `average='weighted'` calls `np.bincount(y_true)`, so it assumes integer labels `0…k-1`.

---

## Example

```python
import numpy as np
from src.naive_bayes._naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes
from src.naive_bayes._eval_utils import evaluate_classification_model, f1_score

rng = np.random.default_rng(0)

# Continuous features -> Gaussian NB
X = np.vstack([rng.normal(0, 1, (100, 3)), rng.normal(2, 1, (100, 3))])
y = np.repeat([0, 1], 100)
gnb = GaussianNaiveBayes()
gnb.fit(X, y)                                   # returns None, so don't chain
metrics = evaluate_classification_model(y, gnb.predict(X))
print(metrics["accuracy"])

# Word counts -> Multinomial NB
counts = rng.integers(0, 5, size=(120, 10))
labels = rng.integers(0, 3, size=120)
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(counts, labels)
print(mnb.predict_proba(counts[:2]).round(3))
print("macro F1:", f1_score(labels, mnb.predict(counts), average="macro"))
```
