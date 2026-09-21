# _eval_utils.py
# --------------
# Pure NumPy classification evaluation metrics.
# Provides accuracy, precision, recall, F1 score, and confusion matrix
# without requiring scikit-learn.
#
# Generative AI (GenAI) was used in the development of this file to assist with
# docstrings, comments, the averaging logic (binary/macro/weighted) in precision(),
# recall(), and f1_score(), and formatting of the evaluate_classification_model()
# output.

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    """Build a confusion matrix from true and predicted labels.

    Rows correspond to true classes, columns to predicted classes.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels.
    labels : array-like, optional
        Ordered list of labels to index the matrix. If None, labels are
        inferred from the union of y_true and y_pred.

    Returns
    -------
    cm : ndarray of shape (n_labels, n_labels)
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def accuracy(y_true, y_pred):
    """Fraction of correctly classified samples.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)

    Returns
    -------
    float
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average="binary", labels=None):
    """Precision = TP / (TP + FP).

    Supports ``'binary'``, ``'macro'``, and ``'weighted'`` averaging.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    average : str, default 'binary'
        Averaging strategy for multi-class data.
    labels : array-like, optional

    Returns
    -------
    float
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)

    if average == "binary":
        return per_class[1] if len(labels) == 2 else per_class[0]
    elif average == "macro":
        return np.mean(per_class)
    elif average == "weighted":
        support = np.bincount(y_true.astype(int), minlength=len(labels))
        return np.average(per_class, weights=support)


def recall(y_true, y_pred, average="binary", labels=None):
    """Recall = TP / (TP + FN).

    Supports ``'binary'``, ``'macro'``, and ``'weighted'`` averaging.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    average : str, default 'binary'
    labels : array-like, optional

    Returns
    -------
    float
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm)
    fn = np.sum(cm, axis=1) - tp
    per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)

    if average == "binary":
        return per_class[1] if len(labels) == 2 else per_class[0]
    elif average == "macro":
        return np.mean(per_class)
    elif average == "weighted":
        support = np.bincount(y_true.astype(int), minlength=len(labels))
        return np.average(per_class, weights=support)


def f1_score(y_true, y_pred, average="binary", labels=None):
    """F1 = 2 * precision * recall / (precision + recall).

    Computed per-class then averaged. Supports ``'binary'``, ``'macro'``,
    and ``'weighted'`` averaging.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    average : str, default 'binary'
    labels : array-like, optional

    Returns
    -------
    float
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    prec_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    rec_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)

    per_class = np.divide(
        2 * prec_per_class * rec_per_class,
        prec_per_class + rec_per_class,
        out=np.zeros_like(tp, dtype=float),
        where=(prec_per_class + rec_per_class) > 0,
    )

    if average == "binary":
        return per_class[1] if len(labels) == 2 else per_class[0]
    elif average == "macro":
        return np.mean(per_class)
    elif average == "weighted":
        support = np.bincount(y_true.astype(int), minlength=len(labels))
        return np.average(per_class, weights=support)


def evaluate_classification_model(y_true, y_pred, labels=None, target_names=None):
    """Print and return a full set of classification metrics.

    Prints confusion matrix, accuracy, precision, recall, and F1 score.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    labels : array-like, optional
    target_names : list of str, optional
        Human-readable class names for display.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    if target_names is None:
        target_names = [f"Class {label}" for label in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred, average="binary", labels=labels)
    rec = recall(y_true, y_pred, average="binary", labels=labels)
    f1 = f1_score(y_true, y_pred, average="binary", labels=labels)

    print("Confusion Matrix:")
    header = " " * 12 + " ".join(f"{name:>12}" for name in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>10}  " + " ".join(f"{val:12d}" for val in row))

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
