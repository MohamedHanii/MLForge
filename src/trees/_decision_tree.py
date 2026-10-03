"""Decision tree classifier implemented from scratch.

This module defines :class:`DecisionTree`, a binary classification decision
tree built without scikit-learn. Splits are chosen greedily to maximise the
*information gain* (the reduction in Shannon entropy), which is the exact
criterion taught in the course notebook ``06-trees-ensembles.ipynb``.

Unlike the categorical, multi-way ID3 sketch in the notebook, this version
operates on numeric ``numpy`` arrays and handles features the way
scikit-learn does: for every feature it tests candidate thresholds (the
midpoints between consecutive sorted values) and keeps the binary split
``X[:, f] <= threshold`` that yields the largest information gain. Categorical
columns are supported by passing them label-encoded (as integers), exactly as
done in the notebook with ``LabelEncoder``.

The public interface (constructor parameters, ``fit`` / ``predict`` signatures
and the ``feature_importances_`` attribute) follows the course student guide.
"""

import numpy as np


class _Node:
    """A single node of the decision tree.

    Internal nodes store the split (``feature_index`` and ``threshold``) and
    references to their two children. Leaf nodes leave those fields as ``None``
    and only store ``value`` — the class label the leaf predicts.
    """

    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        """A node is a leaf when it holds no split feature."""
        return self.feature_index is None


class DecisionTree:
    """A classification decision tree built from scratch.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth the tree is allowed to grow to. The root is at depth 0.
    min_samples_split : int, default=2
        Minimum number of samples a node must contain to be eligible for a
        further split; smaller nodes become leaves.
    max_features : {None, "sqrt", "log2"}, int or float, default=None
        Number of features to consider when looking for the best split at each
        node. ``None`` uses every feature (an ordinary decision tree); any other
        value makes the node draw a random subset of features, which is what a
        random forest relies on to de-correlate its trees.
    random_state : int or None, default=None
        Seed for the random feature subsampling (only relevant when
        ``max_features`` is not ``None``); makes the tree reproducible.

    Attributes
    ----------
    feature_importances_ : numpy.ndarray
        Importance score per feature (sums to 1), populated after :meth:`fit`.
        A feature's importance is the total, sample-weighted information gain
        of every split made on it — the same definition scikit-learn uses.
    classes_ : numpy.ndarray
        The sorted set of class labels seen during :meth:`fit`.
    tree_depth : int
        Depth of the fitted tree (a single-leaf tree has depth 0).
    """

    def __init__(self, max_depth=5, min_samples_split=2,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.feature_importances_ = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """Build the decision tree from training data.

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with class labels

        Returns:
            self, to allow call chaining (e.g. ``DecisionTree().fit(X, y)``).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Encode labels to integer indices 0..n_classes-1 for fast counting;
        # ``classes_`` maps them back to the original labels for prediction.
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_ = X.shape[1]
        self._n_samples_total = X.shape[0]

        # With the default ``max_features=None`` the budget equals n_features and
        # no sampling happens, so behaviour matches a plain decision tree.
        self._rng = np.random.default_rng(self.random_state)
        self._max_features_ = self._resolve_max_features(self.n_features_)

        self._importances = np.zeros(self.n_features_, dtype=float)

        self.root_ = self._grow(X, y_idx, depth=0)

        # Normalise importances to sum to 1 (scikit-learn convention).
        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0 else self._importances
        )
        self.tree_depth = self._depth_of(self.root_)
        return self

    def predict(self, X):
        """Predict class labels for the given input.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with predicted class labels
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:  # accept a single sample passed as a 1-D vector
            X = X.reshape(1, -1)
        return np.array([self._predict_one(row) for row in X])

    def get_depth(self):
        """Return the depth of the fitted tree (root = depth 0)."""
        return self.tree_depth

    # ------------------------------------------------------------------ #
    # Tree construction
    # ------------------------------------------------------------------ #
    def _grow(self, X, y, depth):
        """Recursively grow the (sub)tree for the samples ``(X, y)``."""
        n_samples = y.shape[0]
        counts = np.bincount(y, minlength=self.n_classes_)
        n_present_classes = np.count_nonzero(counts)

        # Stopping criteria -> turn this node into a leaf holding the majority
        # class: the node is pure, the depth limit is reached, or there are too
        # few samples to split.
        if (n_present_classes <= 1
                or depth >= self.max_depth
                or n_samples < self.min_samples_split):
            return _Node(value=self._leaf_label(counts))

        parent_entropy = self._entropy_from_counts(
            counts[None, :].astype(float),
            np.array([n_samples], dtype=float),
        )[0]

        feature_index, threshold, gain = self._best_split(X, y, parent_entropy)

        if feature_index is None or gain <= 0:  # no split improves purity
            return _Node(value=self._leaf_label(counts))

        left_mask = X[:, feature_index] <= threshold

        # Importance contribution: the split's gain weighted by the fraction of
        # all samples that reach this node.
        self._importances[feature_index] += (
            n_samples / self._n_samples_total
        ) * gain

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature_index=feature_index, threshold=threshold,
                     left=left, right=right)

    def _best_split(self, X, y, parent_entropy):
        """Find the (feature, threshold) split with the highest information gain.

        Returns a ``(feature_index, threshold, gain)`` tuple. ``feature_index``
        is ``None`` when no valid split exists. The search is vectorised: for
        each feature the samples are sorted once and cumulative class counts let
        every candidate threshold be evaluated in a single pass.
        """
        n_samples, n_features = X.shape

        # One-hot labels so that cumulative sums give per-class counts.
        onehot = np.zeros((n_samples, self.n_classes_), dtype=np.int64)
        onehot[np.arange(n_samples), y] = 1
        total_counts = onehot.sum(axis=0)

        best_gain = 0.0  # require strictly positive gain to bother splitting
        best_feature = None
        best_threshold = None

        n_left = np.arange(1, n_samples, dtype=float)
        n_right = n_samples - n_left

        # Consider every feature, or (random forest) a random subset of them.
        if self._max_features_ >= n_features:
            candidate_features = range(n_features)
        else:
            candidate_features = self._rng.choice(
                n_features, size=self._max_features_, replace=False)

        for feature_index in candidate_features:
            column = X[:, feature_index]
            order = np.argsort(column, kind="mergesort")
            sorted_col = column[order]

            # Cumulative left-side class counts for every split position.
            cum = np.cumsum(onehot[order], axis=0)
            left_counts = cum[:-1].astype(float)
            right_counts = total_counts - left_counts

            # A threshold only makes sense where the feature value changes.
            valid = sorted_col[:-1] != sorted_col[1:]
            if not valid.any():
                continue

            ent_left = self._entropy_from_counts(left_counts, n_left)
            ent_right = self._entropy_from_counts(right_counts, n_right)
            child_entropy = (n_left / n_samples) * ent_left \
                + (n_right / n_samples) * ent_right
            gain = parent_entropy - child_entropy
            gain[~valid] = -np.inf  # forbid splits between equal values

            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain = float(gain[i])
                best_feature = feature_index
                # Threshold midway between the two values straddling the split.
                best_threshold = (sorted_col[i] + sorted_col[i + 1]) / 2.0

        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _entropy_from_counts(counts, n):
        """Shannon entropy (base 2) for one or many count vectors.

        Args:
            counts: array of shape (m, n_classes) with per-class counts.
            n: array of shape (m,) with the total count of each row.

        Returns:
            array of shape (m,) with the entropy of each row. The convention
            ``0 * log2(0) = 0`` is applied so pure rows give entropy 0.
        """
        p = counts / n[:, None]
        logp = np.zeros_like(p)
        np.log2(p, out=logp, where=p > 0)
        return -np.sum(p * logp, axis=1)

    def _resolve_max_features(self, n_features):
        """Translate the ``max_features`` setting into a concrete feature count.

        Accepts ``None`` (all features), the strings ``"sqrt"`` / ``"log2"``,
        an int (a fixed number) or a float in (0, 1] (a fraction of features).
        """
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(f"Unknown max_features option: {mf!r}")
        if isinstance(mf, (int, np.integer)):
            return max(1, min(int(mf), n_features))
        if isinstance(mf, float):
            return max(1, min(int(mf * n_features), n_features))
        raise ValueError(f"Invalid max_features: {mf!r}")

    def _leaf_label(self, counts):
        """Return the majority class label given per-class counts at a node."""
        return self.classes_[int(np.argmax(counts))]

    def _predict_one(self, x):
        """Route a single sample down the tree and return its leaf label."""
        node = self.root_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _depth_of(self, node):
        """Depth of the subtree rooted at ``node`` (a leaf has depth 0)."""
        if node.is_leaf:
            return 0
        return 1 + max(self._depth_of(node.left), self._depth_of(node.right))
