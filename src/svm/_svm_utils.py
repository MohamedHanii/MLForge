"""Cross-validation splitters, kernels, and evaluation metrics."""
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np


class KFold:
    """
    K-Fold cross-validator.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int or None, default=None
        Seed for reproducible shuffling.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None) -> None:
        self.n_splits: int = n_splits
        self.shuffle: bool = shuffle
        self.random_state: Optional[int] = random_state

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold.

        Yields
        ------
        train_idx : ndarray
        test_idx : ndarray
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes: np.ndarray = np.full(
            self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop

    def get_n_splits(self) -> int:
        """Return the number of folds."""
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold cross-validator.

    Like KFold, but preserves each class's proportion within every fold.
    This avoids degenerate (single-class) test folds when the data is
    ordered by class — important for reliable C/gamma selection on noisy
    classification data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before assigning folds.
    random_state : int or None, default=None
        Seed for reproducible shuffling.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None) -> None:
        self.n_splits: int = n_splits
        self.shuffle: bool = shuffle
        self.random_state: Optional[int] = random_state

    def split(self, X: np.ndarray,
              y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for each fold, preserving class ratios.

        Yields
        ------
        train_idx : ndarray
        test_idx : ndarray
        """
        y = np.asarray(y)
        n_samples = len(y)
        test_folds = np.empty(n_samples, dtype=int)
        rng = np.random.RandomState(self.random_state) if self.shuffle else None

        # Assign each class's samples round-robin across folds, so every fold
        # gets ~the same fraction of each class as the full set.
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            if rng is not None:
                rng.shuffle(cls_idx)
            test_folds[cls_idx] = np.arange(cls_idx.size) % self.n_splits

        indices = np.arange(n_samples)
        for fold in range(self.n_splits):
            test_idx = indices[test_folds == fold]
            train_idx = indices[test_folds != fold]
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        """Return the number of folds."""
        return self.n_splits


class GridSearchCV:
    """
    Exhaustive search over parameter grid with cross-validation.

    Parameters
    ----------
    estimator : object
        Estimator with fit/predict/score and get_params/set_params.
    param_grid : dict
        Dict mapping parameter names to lists of values to try.
    cv : int or KFold, default=5
        Cross-validation splitter or number of folds.
    scoring : str, optional
        Not yet used — uses estimator.score().
    """

    def __init__(self, estimator: Any,
                 param_grid: Dict[str, List[Any]],
                 cv: Optional[Union[int, KFold, 'StratifiedKFold']] = None,
                 scoring: Optional[str] = None) -> None:
        self.estimator: Any = estimator
        self.param_grid: Dict[str, List[Any]] = param_grid
        self.scoring: Optional[str] = scoring

        if cv is None or isinstance(cv, int):
            n = cv if isinstance(cv, int) else 5
            # Robust defaults: shuffle (with a fixed seed for reproducibility)
            # and stratify for classifiers so folds aren't degenerate on
            # class-ordered data.
            if getattr(estimator, '_estimator_type', None) == 'classifier':
                self.cv: Any = StratifiedKFold(n_splits=n, shuffle=True,
                                               random_state=42)
            else:
                self.cv = KFold(n_splits=n, shuffle=True, random_state=42)
        else:
            self.cv = cv          # explicit splitter passed by user is respected

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: float = -np.inf
        self.best_estimator_: Optional[Any] = None
        self.cv_results_: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GridSearchCV':
        """
        Run grid search over param_grid.

        Stores best_params_, best_score_, best_estimator_, and cv_results_.
        """
        if not hasattr(self.estimator, 'get_params'):
            raise TypeError(
                "estimator must have a 'get_params' method for GridSearchCV."
            )
        if not self.param_grid:
            raise ValueError("param_grid is empty. Must specify at least one parameter.")

        param_combos = self._expand_grid(self.param_grid)
        if not param_combos:
            raise ValueError(
                "param_grid produced no parameter combinations. "
                "Check that each parameter has at least one value."
            )
        scores: List[float] = []

        for params in param_combos:
            est = self.estimator.__class__(**{**self.estimator.get_params(), **params})
            fold_scores: List[float] = []

            for train_idx, test_idx in self.cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                est.fit(X_train, y_train)
                fold_scores.append(est.score(X_test, y_test))

            mean_score: float = float(np.mean(fold_scores))
            scores.append(mean_score)

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = est

        self.cv_results_ = {'params': param_combos, 'mean_test_score': scores}

        # param_combos is non-empty (validated above), so a best estimator was
        # selected during the loop; refit it on the full data.
        assert self.best_estimator_ is not None
        self.best_estimator_.fit(X, y)
        return self

    @staticmethod
    def _expand_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Expand a param grid into the Cartesian product of all combinations."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos: List[List[Any]] = [[]]
        for vals in values:
            combos = [c + [v] for c in combos for v in vals]
        return [dict(zip(keys, c)) for c in combos]


def rbf_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
               gamma: Optional[float] = None) -> np.ndarray:
    """RBF kernel matrix:  K(x, y) = exp(-gamma * ||x - y||^2)."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    if Y is None:
        X_norm = np.sum(X ** 2, axis=1)
        K = X_norm[:, np.newaxis] + X_norm[np.newaxis, :] - 2 * X @ X.T
    else:
        X_norm = np.sum(X ** 2, axis=1)
        Y_norm = np.sum(Y ** 2, axis=1)
        K = X_norm[:, np.newaxis] + Y_norm[np.newaxis, :] - 2 * X @ Y.T
    return np.exp(-gamma * K)


def poly_kernel(X: np.ndarray, Y: Optional[np.ndarray] = None,
                degree: int = 3, gamma: Optional[float] = None,
                coef0: float = 1.0) -> np.ndarray:
    """Polynomial kernel matrix:  K(x, y) = (gamma * <x, y> + coef0) ** degree."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    if Y is None:
        K = X @ X.T
    else:
        K = X @ Y.T
    return (gamma * K + coef0) ** degree


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that match the true labels."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray,
                    pos_label: Optional[Any] = None) -> float:
    """Precision = TP / (TP + FP) for the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if pos_label is None:
        pos_label = np.unique(y_true)[1]
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                 pos_label: Optional[Any] = None) -> float:
    """Recall = TP / (TP + FN) for the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if pos_label is None:
        pos_label = np.unique(y_true)[1]
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray,
             pos_label: Optional[Any] = None) -> float:
    """F1 = harmonic mean of precision and recall."""
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def confusion_matrix(y_true: np.ndarray,
                     y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the confusion matrix and the sorted label order."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    cm: np.ndarray = np.zeros((n_labels, n_labels), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Build a per-class precision/recall/F1 text report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    report: str = f"{'':>14}precision  recall  f1-score  support\n"
    for label in labels:
        p = precision_score(y_true, y_pred, pos_label=label)
        r = recall_score(y_true, y_pred, pos_label=label)
        f1 = f1_score(y_true, y_pred, pos_label=label)
        support = int(np.sum(y_true == label))
        report += f"  {str(label):>12}  {p:.4f}   {r:.4f}    {f1:.4f}    {support}\n"

    acc = accuracy_score(y_true, y_pred)
    report += f"\n  {'accuracy':>12}                              {acc:.4f}    {len(y_true)}\n"
    return report
