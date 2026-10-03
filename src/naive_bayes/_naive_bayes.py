# _naive_bayes.py
# ---------------
# Pure NumPy implementations of Gaussian and Multinomial Naive Bayes classifiers.
#
# Generative AI (GenAI) was used in the development of this file to assist with
# docstrings, comments, and the log-Gaussian-PDF computation within
# GaussianNaiveBayes.predict_log_proba().

import numpy as np


def _validate_fit_inputs(X, y):
    """Validate X and y shapes for fit(), supporting duck-typed arrays."""
    X_shape = getattr(X, "shape", None)
    y_shape = getattr(y, "shape", None)
    if X_shape is None or len(X_shape) != 2:
        raise ValueError(
            f"X must be a 2D array, got shape {X_shape}"
        )
    if y_shape is None or len(y_shape) != 1:
        raise ValueError(
            f"y must be a 1D array, got shape {y_shape}"
        )
    n_samples, n_features = X_shape
    if n_samples == 0:
        raise ValueError("X has 0 samples: cannot fit on empty data")
    if y_shape[0] == 0:
        raise ValueError("y has 0 samples: cannot fit on empty data")
    if n_samples != y_shape[0]:
        raise ValueError(
            f"X and y length mismatch: X has {n_samples} samples, "
            f"y has {y_shape[0]} samples"
        )
    return n_samples, n_features


class GaussianNaiveBayes:
    """Naive Bayes classifier for continuous features.

    Assumes features within each class follow a Gaussian (normal) distribution.
    Works with log-probabilities internally for numerical stability.

    Attributes (set after calling fit):
        classes_ : ndarray of shape (n_classes,)
            Unique class labels.
        priors_ : ndarray of shape (n_classes,)
            Prior probability P(C_k) for each class.
        mean_ : ndarray of shape (n_classes, n_features)
            Mean of each feature for each class.
        variance_ : ndarray of shape (n_classes, n_features)
            Variance of each feature for each class.
    """

    def __init__(self):
        """Initialize the Gaussian Naive Bayes classifier.

        All model parameters (classes_, priors_, mean_, variance_) are
        set to None and populated during fit().
        """
        self.classes_ = None
        self.priors_ = None
        self.mean_ = None
        self.variance_ = None

    def fit(self, X, y):
        """Compute class priors, per-class means and per-class variances.

        Uses maximum likelihood estimates:
            P(C_k) = n_k / n
            mu_{k,i} = mean of feature i for class k
            sigma^2_{k,i} = variance of feature i for class k

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data with continuous feature values.
        y : ndarray of shape (n_samples,)
            Target class labels.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = _validate_fit_inputs(X, y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.priors_ = np.empty(n_classes)
        self.mean_ = np.empty((n_classes, n_features))
        self.variance_ = np.empty((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[idx] = len(X_c) / n_samples
            self.mean_[idx] = np.mean(X_c, axis=0)
            self.variance_[idx] = np.var(X_c, axis=0)

    def _check_fitted(self):
        """Guard — raises RuntimeError if the model hasn't been fitted yet."""
        if self.classes_ is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

    def _validate_predict_input(self, X):
        """Convert X to ndarray and verify it is 2D with correct n_features."""
        X = np.asarray(X)
        X_shape = getattr(X, "shape", None)
        if X_shape is None or len(X_shape) != 2:
            raise ValueError(
                f"X must be a 2D array, got shape {X_shape}"
            )
        if X_shape[1] != self.mean_.shape[1]:
            raise ValueError(
                f"X has {X_shape[1]} features but model was fitted with "
                f"{self.mean_.shape[1]} features"
            )
        return X

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = self._validate_predict_input(X)
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X):
        """Predict class posterior probabilities.

        Computes normalized probabilities via stable softmax over log-probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._validate_predict_input(X)
        log_proba = self.predict_log_proba(X)
        log_proba -= np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict_log_proba(self, X):
        """Predict log of class posterior probabilities (unnormalized).

        Computes log(P(C_k)) + sum_i log(P(x_i | C_k)) for each class.
        Each likelihood uses the Gaussian PDF:
            log P(x_i | C_k) = -0.5 * log(2*pi*var) - (x_i - mean)^2 / (2*var)

        A small epsilon is added to variance to prevent division by zero.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._validate_predict_input(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_proba = np.empty((n_samples, n_classes))

        epsilon = 1e-9  # Prevent division by zero when variance is zero
        for idx in range(n_classes):
            prior = self.priors_[idx]
            mean = self.mean_[idx]
            var = self.variance_[idx] + epsilon

            # Log of Gaussian PDF (ignoring constant terms that cancel)
            log_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * (X - mean) ** 2 / var
            log_proba[:, idx] = np.sum(log_pdf, axis=1) + np.log(prior)

        return log_proba


class MultinomialNaiveBayes:
    """Naive Bayes classifier for discrete count features.

    Suitable for text classification with word counts, or any non-negative
    integer features. Uses Laplace (additive) smoothing to avoid zero
    probabilities.

    Attributes (set after calling fit):
        alpha : float
            Laplace smoothing parameter (default 1.0).
        classes_ : ndarray of shape (n_classes,)
            Unique class labels.
        class_log_prior_ : ndarray of shape (n_classes,)
            Log prior probability log(P(C_k)) for each class.
        feature_log_prob_ : ndarray of shape (n_classes, n_features)
            Log of smoothed feature likelihoods log(P(x_i | C_k)).
    """

    def __init__(self, alpha=1.0):
        """Initialize the Multinomial Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Laplace (additive) smoothing parameter. alpha=1.0 corresponds
            to add-one smoothing. Higher values produce more uniform
            probability estimates and prevent zero-frequency issues.
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Compute log class priors and log feature likelihoods.

        The smoothed feature probability for class k, feature i:
            P(x_i | C_k) = (count_{k,i} + alpha) / (total_k + alpha * n_features)

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Non-negative integer count features.
        y : ndarray of shape (n_samples,)
            Target class labels.
        """
        y = np.asarray(y)
        n_samples, n_features = _validate_fit_inputs(X, y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.empty(n_classes)
        self.feature_log_prob_ = np.empty((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            self.class_log_prior_[idx] = np.log(n_c) - np.log(n_samples)

            count_c = np.sum(X_c, axis=0)
            total_c = np.sum(count_c)
            # Laplace smoothing
            smoothed = (count_c + self.alpha) / (total_c + self.alpha * n_features)
            self.feature_log_prob_[idx] = np.log(smoothed)

    def _check_fitted(self):
        """Guard — raises RuntimeError if the model hasn't been fitted yet."""
        if self.classes_ is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

    def _validate_predict_input(self, X):
        """Verify X is a 2D array with the correct number of features."""
        X_shape = X.shape
        if len(X_shape) != 2:
            raise ValueError(
                f"X must be a 2D array, got {len(X_shape)}D with shape {X_shape}"
            )
        if X_shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError(
                f"X has {X_shape[1]} features but model was fitted with "
                f"{self.feature_log_prob_.shape[1]} features"
            )
        return X

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = self._validate_predict_input(X)
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X):
        """Predict class posterior probabilities.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._validate_predict_input(X)
        log_proba = self.predict_log_proba(X)
        log_proba -= np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict_log_proba(self, X):
        """Predict log of class posterior probabilities.

        Uses matrix multiplication for efficiency:
            log P(X | C_k) = X @ log(P(x_i | C_k))

        Supports both dense ndarrays and sparse csr_matrix inputs.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = self._validate_predict_input(X)
        log_likelihood = X @ self.feature_log_prob_.T
        return log_likelihood + self.class_log_prior_
