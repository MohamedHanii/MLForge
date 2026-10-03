"""Random forest classifier implemented from scratch.

This module defines :class:`RandomForest`, a bagging ensemble of
:class:`mlab.trees.DecisionTree` estimators. It combines the two sources of
randomness that make a random forest (as described in the course notebook
``06-trees-ensembles.ipynb``, section 5.2):

1. **Bootstrap sampling** — each tree is trained on a sample of the data drawn
   with replacement (so every tree sees a slightly different dataset).
2. **Feature subsampling** — at every split each tree only considers a random
   subset of the features (``max_features``), which de-correlates the trees.

Predictions are combined by majority vote. The public interface (constructor
parameters, ``fit`` / ``predict`` signatures and the ``trees_`` /
``feature_importances_`` / ``oob_score_`` attributes) follows the course
student guide.
"""

import numpy as np

from ..trees._decision_tree import DecisionTree


class RandomForest:
    """A random forest classifier using :class:`DecisionTree` as base estimator.

    Parameters
    ----------
    n_estimators : int, default=20
        Number of decision trees to grow.
    max_depth : int, default=5
        Maximum depth of each individual tree.
    min_samples_split : int, default=2
        Minimum number of samples a node needs to be split (passed to each tree).
    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        Number of features each tree considers per split. ``"sqrt"`` is the
        usual random-forest default and is what makes the trees differ from one
        another; ``None`` would consider all features (plain bagging).
    bootstrap : bool, default=True
        Whether to train each tree on a bootstrap sample (sampling with
        replacement). Required for out-of-bag scoring.
    oob_score : bool, default=False
        If ``True`` (and ``bootstrap`` is on), estimate the generalisation
        accuracy using the out-of-bag samples and store it in ``oob_score_``.
    random_state : int or None, default=None
        Seed for reproducible bootstrap sampling and feature selection.

    Attributes
    ----------
    trees_ : list of DecisionTree
        The fitted trees, populated after :meth:`fit`.
    feature_importances_ : numpy.ndarray
        Feature importances averaged over all trees (sums to 1), set after fit.
    classes_ : numpy.ndarray
        The sorted set of class labels seen during :meth:`fit`.
    oob_score_ : float
        Out-of-bag accuracy, set after fit when ``oob_score=True``.
    """

    def __init__(self, n_estimators=20, max_depth=5, min_samples_split=2,
                 max_features="sqrt", bootstrap=True, oob_score=False,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """Build the random forest from training data using bootstrap sampling.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels

        Returns:
            self, to allow call chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        rng = np.random.default_rng(self.random_state)

        self.trees_ = []
        importances = np.zeros(n_features, dtype=float)

        # For out-of-bag scoring: per-sample vote counts from trees that did
        # NOT see that sample during training.
        oob_votes = np.zeros((n_samples, self.classes_.shape[0]), dtype=float)

        for _ in range(self.n_estimators):
            # A per-tree seed keeps each tree's bootstrap draw and feature
            # sampling reproducible, yet different from the other trees.
            seed = int(rng.integers(0, 2**32 - 1))
            tree_rng = np.random.default_rng(seed)

            if self.bootstrap:
                sample_idx = tree_rng.integers(0, n_samples, size=n_samples)
            else:
                sample_idx = np.arange(n_samples)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
            importances += tree.feature_importances_

            # Vote on the samples this tree never saw, for the OOB estimate.
            if self.oob_score and self.bootstrap:
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[sample_idx] = False
                if oob_mask.any():
                    for row_i, pred in zip(np.flatnonzero(oob_mask),
                                           tree.predict(X[oob_mask])):
                        oob_votes[row_i, class_to_idx[pred]] += 1

        # Each tree's importances sum to 1, so the average does too.
        self.feature_importances_ = importances / self.n_estimators

        if self.oob_score and self.bootstrap:
            self.oob_score_ = self._compute_oob_score(oob_votes, y)

        return self

    def predict(self, X):
        """Predict class labels using majority voting across all trees.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        if not self.trees_:
            raise ValueError("RandomForest is not fitted yet; call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predictions from every tree, shape (n_trees, n_samples).
        tree_preds = np.array([tree.predict(X) for tree in self.trees_])
        return self._majority_vote(tree_preds)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _majority_vote(self, tree_preds):
        """Combine per-tree predictions into a single label per sample.

        Args:
            tree_preds: array of shape (n_trees, n_samples) of class labels.

        Returns:
            array of shape (n_samples,) with the most-voted label per sample.
        """
        n_samples = tree_preds.shape[1]
        # Map labels to class indices so they can be counted; classes_ is sorted,
        # so searchsorted works for numeric and string labels alike.
        idx = np.searchsorted(self.classes_, tree_preds)
        result = np.empty(n_samples, dtype=self.classes_.dtype)
        for j in range(n_samples):
            counts = np.bincount(idx[:, j], minlength=self.classes_.shape[0])
            result[j] = self.classes_[int(np.argmax(counts))]
        return result

    def _compute_oob_score(self, oob_votes, y):
        """Out-of-bag accuracy from accumulated per-sample votes."""
        scored = oob_votes.sum(axis=1) > 0  # samples left out by >=1 tree
        if not scored.any():
            return float("nan")
        oob_pred = self.classes_[np.argmax(oob_votes[scored], axis=1)]
        return float(np.mean(oob_pred == y[scored]))
