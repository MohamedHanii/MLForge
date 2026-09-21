"""Core SVM building blocks: scaler, base estimator, and shared errors."""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from ._svm_utils import rbf_kernel, poly_kernel, accuracy_score


class NotFittedError(Exception):
    """Raised when predict() or decision_function() is called before fit()."""


class ConvergenceWarning(UserWarning):
    """Raised when the solver fails to converge within max_iter."""


class StandardScaler:
    """
    Standardize features:  x' = (x - mean) / std

    After scaling, each feature has mean=0 and standard deviation=1.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """Compute and store the per-feature mean and standard deviation."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored standardization to X."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler on X and return the standardized X."""
        self.fit(X)
        return self.transform(X)


class SVM:
    """
    Support Vector Machine classifier (soft-margin) and shared base for SVR.

    Usable directly as a classifier: binary problems train via primal gradient
    descent (linear) or dual Pegasos (rbf/poly); problems with more than two
    classes use a deterministic one-vs-rest scheme. SVC is a thin alias of this
    class; SVR overrides the loss-specific pieces for regression.
    """

    _estimator_type = "classifier"

    # One-vs-rest sub-classifiers for multiclass problems (None when binary).
    _ovr: Optional[List['SVM']] = None

    # Fitted state, created during fit(). Declared here (without values) so the
    # types are known, subclasses can assign them in fit() without redefinition
    # warnings, and they are non-Optional for arithmetic use after fitting.
    # Access before fit() is prevented by the is_fitted_ / _check_fitted guard.
    _rng: np.random.RandomState
    weight_: np.ndarray
    X_train_: np.ndarray
    _y_train: np.ndarray
    _dual_coef_: np.ndarray
    classes_: np.ndarray
    support_vectors_: np.ndarray
    support_: np.ndarray
    alpha_: np.ndarray
    intercept_: float
    n_support_: int

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, degree: int = 3,
                 gamma: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        self.kernel: str = kernel
        self.C: float = C
        self.degree: int = degree
        self.gamma: Optional[float] = gamma
        self.random_state: Optional[int] = random_state

        self._init_rng()

        self._lr: float = 0.01
        self._tol: float = 1e-4
        self._max_iter: int = 1000

        # Initialised here because the kernel SVC path reads bias_ without
        # setting it, and is_fitted_ guards all post-fit accessors.
        self.bias_: float = 0.0
        self.scaler: StandardScaler = StandardScaler()
        self.is_fitted_: bool = False

    def _init_rng(self) -> None:
        """
        (Re)seed the per-estimator RNG.

        Called once per fit (via _fit_dispatch) so that every training run
        starts from the same RNG state for a given random_state. This makes
        repeated fit() calls reproducible and ensures a random_state changed
        after construction (e.g. via set_params) actually takes effect.

        random_state=None maps to a fixed seed (42) so results are always
        deterministic.
        """
        seed = self.random_state if self.random_state is not None else 42
        self._rng = np.random.RandomState(seed)

    def _validate_params(self) -> None:
        if self.C <= 0:
            raise ValueError(f"C must be > 0. Got C={self.C}.")
        if self.kernel not in ('linear', 'rbf', 'poly'):
            raise ValueError(
                f"kernel must be 'linear', 'rbf', or 'poly'. Got '{self.kernel}'."
            )
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1. Got degree={self.degree}.")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError(f"gamma must be > 0 or None. Got gamma={self.gamma}.")

    def _validate_input(self, X: np.ndarray,
                        y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.size == 0:
            raise ValueError("X is empty. Must have at least 1 sample.")
        if y.size == 0:
            raise ValueError("y is empty. Must have at least 1 sample.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != len(y):
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} samples. "
                f"They must match."
            )
        if X.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 samples for training. Got {X.shape[0]}."
            )
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values.")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values.")

        return X, y

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. "
                f"Call 'fit' before 'predict' or 'decision_function'."
            )

    def _compute_kernel(self, X: np.ndarray,
                        Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the kernel matrix K.

        RBF:    K(x, y) = exp( −γ · ||x − y||² )
        Poly:   K(x, y) = ( γ · ⟨x, y⟩ + 1 ) ^ degree

        Only called for the 'rbf' and 'poly' kernels; raises otherwise.
        """
        if X.shape[1] < 1:
            raise ValueError("X must have at least 1 feature.")

        g = self.gamma if self.gamma else 1.0 / X.shape[1]
        if self.kernel == 'rbf':
            return rbf_kernel(X, Y, gamma=g)
        if self.kernel == 'poly':
            return poly_kernel(X, Y, degree=self.degree, gamma=g)
        raise ValueError(
            f"_compute_kernel called with non-kernel '{self.kernel}'."
        )

    def _fit_dispatch(self) -> None:
        """
        Shared dispatch:
            'linear'  →  _fit_linear()                (primal gradient descent)
            'rbf'     →  _compute_kernel + _fit_kernel (dual Pegasos)
            'poly'    →  _compute_kernel + _fit_kernel (dual Pegasos)

        Reseeds the RNG first so each fit is reproducible and deterministic.
        """
        self._init_rng()
        if self.kernel == 'linear':
            self._fit_linear()
        elif self.kernel in ('rbf', 'poly'):
            K_train = self._compute_kernel(self.X_train_)
            self._fit_kernel(K_train)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the estimator's hyper-parameters as a dict (sklearn-style)."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'degree': self.degree,
            'gamma': self.gamma,
            'random_state': self.random_state,
        }

    def set_params(self, **params: Any) -> 'SVM':
        """Set hyper-parameters from keyword arguments (sklearn-style)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    @staticmethod
    def _sign(x: np.ndarray) -> np.ndarray:
        """sign(x) = 1.0 if x > 0 else -1.0."""
        return np.where(x > 0, 1.0, -1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Train the classifier.

        Binary  → map labels to {-1, +1}, standardise, train (primal GD or
                  dual Pegasos).
        Multiclass (>2 classes) → train one binary model per class
                  (one-vs-rest) and predict by the largest decision score.
        """
        self._validate_params()
        X, y = self._validate_input(X, y)

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"Classification requires at least 2 classes. "
                f"Got {len(self.classes_)}."
            )

        if len(self.classes_) == 2:
            self._ovr = None
            self._y_train = np.where(y == self.classes_[1], 1.0, -1.0)
            self._standardize_and_train(X)
        else:
            self._fit_one_vs_rest(X, y)
        return self

    def _fit_one_vs_rest(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train one binary classifier per class (class vs. all others)."""
        self._ovr = []
        for c in self.classes_:
            sub = type(self)(kernel=self.kernel, C=self.C, degree=self.degree,
                             gamma=self.gamma, random_state=self.random_state)
            sub.fit(X, np.where(y == c, 1, 0))
            self._ovr.append(sub)
        # Aggregate fitted state for inspection / API compatibility.
        self.is_fitted_ = True
        self.intercept_ = 0.0
        self.n_support_ = int(sum(s.n_support_ for s in self._ovr))

    def _fit_linear(self) -> None:
        """
        Primal gradient descent on the hinge-loss cost function.

        Cost:  J(w, b) = (1/m) Σ max(0, 1 − yⁱ(wᵀxⁱ + b)) + λ‖w‖²,
        with λ = 1 / (C·m). Per-sample sub-gradients:
            yⁱ(wᵀxⁱ + b) < 1 :  ∇w = 2λw − yⁱxⁱ,   ∇b = −yⁱ
            yⁱ(wᵀxⁱ + b) ≥ 1 :  ∇w = 2λw,           ∇b = 0
        """
        def sample_update(i: int, lam: float) -> float:
            s: float = float(np.dot(self.weight_, self.X_train_[i]) + self.bias_)
            margin: float = float(self._y_train[i] * s)
            if margin < 1:
                self.weight_ -= self._lr * (2 * lam * self.weight_
                                            - self._y_train[i] * self.X_train_[i])
                self.bias_ -= self._lr * (-self._y_train[i])
            else:
                self.weight_ -= self._lr * (2 * lam * self.weight_)
            return max(0.0, 1.0 - margin)

        final_loss, converged = self._run_primal_gd(sample_update)
        if not converged:
            self._warn_not_converged("Linear SVM", f"Final loss: {final_loss:.6f}. ")

        margins = np.abs(self.X_train_.dot(self.weight_) + self.bias_)
        threshold: float = float(np.percentile(margins, 30))
        self._set_support_vectors(margins <= threshold, uniform_alpha=True)

    def _fit_kernel(self, K_train: np.ndarray) -> None:
        """
        Kernel SVM via the Pegasos dual sub-gradient algorithm.

        Decision function:  f(xⁱ) = Σ_j α_j · y_j · K(x_j, xⁱ).
        Each epoch, for every i with yⁱ·f(xⁱ) < 1:  α_i += 1, clipped to [0, C].
        """
        n_samples: int = K_train.shape[0]
        self.alpha_ = np.zeros(n_samples)

        prev_alpha: Optional[np.ndarray] = None
        for _ in range(self._max_iter):
            # Deterministic sweep order: the coordinate update saturates at the
            # box bound, so a random order would converge to a different fixed
            # point per seed. Fixed order makes the result seed-invariant.
            for i in range(n_samples):
                f_i: float = float(np.dot(self.alpha_ * self._y_train, K_train[:, i]))
                if self._y_train[i] * f_i < 1:
                    self.alpha_[i] += 1.0
                    self.alpha_[i] = np.clip(self.alpha_[i], 0.0, self.C)

            if prev_alpha is not None:
                delta: float = float(np.max(np.abs(self.alpha_ - prev_alpha)))
                if delta < self._tol:
                    break
            prev_alpha = self.alpha_.copy()
        else:
            self._warn_not_converged("Kernel SVM")

        self._dual_coef_ = self.alpha_ * self._y_train
        self._set_support_vectors(self.alpha_ > 1e-10)

    def _standardize_and_train(self, X: np.ndarray) -> None:
        """
        Shared fit epilogue: standardise features, run the kernel-appropriate
        solver, then mark the estimator fitted.

        Subclasses prepare their targets (``self._y_train``) and any
        class/scaling state first, then call this.
        """
        self.X_train_ = self.scaler.fit_transform(X)
        self._fit_dispatch()
        self.is_fitted_ = True
        self.intercept_ = self.bias_

    def _run_primal_gd(
        self, sample_update: Callable[[int, float], float]
    ) -> Tuple[float, bool]:
        """
        Shared primal gradient-descent driver for the linear solvers.

        Initialises w=0, b=0 and runs epoch-wise sub-gradient descent over the
        samples in a deterministic order. ``sample_update(i, lam)`` performs the
        per-sample weight/bias update (the loss-specific math lives in the
        subclass) and returns that sample's loss contribution.

        A fixed sweep order makes the converged weights seed-invariant: with a
        constant step size the iterate settles into a small neighbourhood of the
        unique optimum, and a random order would land on a different point in
        that neighbourhood per seed. (random_state is still seeded on the
        estimator for reproducibility; results do not depend on it, matching
        scikit-learn's deterministic SVC/SVR decision functions.)

        Returns ``(final_loss, converged)`` where ``final_loss`` is the last
        epoch's regularised objective and ``converged`` is True if early
        stopping triggered before ``max_iter``.
        """
        n_samples, n_features = self.X_train_.shape
        lam: float = 1.0 / (self.C * n_samples)

        self.weight_ = np.zeros(n_features)
        self.bias_ = 0.0
        last_finite_weight = self.weight_.copy()
        last_finite_bias = self.bias_

        total_loss: float = 0.0
        converged: bool = False
        # Pathological inputs (e.g. extreme feature magnitudes) can make the
        # constant-step updates diverge; we detect that and roll back to the
        # last finite iterate instead of returning NaN/Inf weights. errstate
        # keeps the transient over/invalid warnings out of the output.
        with np.errstate(over='ignore', invalid='ignore'):
            for _ in range(self._max_iter):
                total_loss = 0.0
                for i in range(n_samples):
                    total_loss += sample_update(i, lam)

                if not np.all(np.isfinite(self.weight_)) or not np.isfinite(self.bias_):
                    self.weight_ = last_finite_weight
                    self.bias_ = last_finite_bias
                    break

                last_finite_weight = self.weight_.copy()
                last_finite_bias = self.bias_

                total_loss = total_loss / n_samples + lam * np.sum(self.weight_ ** 2)
                if total_loss < self._tol:
                    converged = True
                    break
        return total_loss, converged

    def _set_support_vectors(self, mask: np.ndarray,
                             uniform_alpha: bool = False) -> None:
        """
        Record support-vector bookkeeping from a boolean mask.

        Shared by SVC and SVR. ``uniform_alpha=True`` assigns equal alpha
        weights — used by the linear primal solvers, which have no dual
        coefficients of their own.
        """
        self.support_ = np.where(mask)[0]
        self.support_vectors_ = self.X_train_[mask]
        self.n_support_ = len(self.support_)
        if uniform_alpha:
            self.alpha_ = np.ones(self.n_support_) / self.n_support_

    def _warn_not_converged(self, model_name: str, extra: str = "") -> None:
        """Emit a ConvergenceWarning after the solver exhausts max_iter."""
        warnings.warn(
            f"{model_name} did not converge after {self._max_iter} epochs. "
            f"{extra}Consider increasing max_iter.",
            ConvergenceWarning,
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Decision scores.

        Binary / regression:
            Linear:  f(x) = wᵀx + b
            Kernel:  f(x) = Σ_j c_j · K(x_j, x)   (c_j = dual coefficients)
        Multiclass classifier:
            (n_samples, n_classes) array of one-vs-rest scores, column j for
            ``classes_[j]``.
        """
        self._check_fitted()
        if self._ovr is not None:
            return np.column_stack([sub.decision_function(X) for sub in self._ovr])

        X = np.asarray(X, dtype=np.float64)
        X_t = self.scaler.transform(X)
        if self.kernel == 'linear':
            return np.dot(X_t, self.weight_) + self.bias_
        K = self._compute_kernel(X_t, self.X_train_)
        return np.dot(self._dual_coef_, K.T) + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Binary     → ŷ = sign(f(x)), mapped back to the original labels.
        Multiclass → argmax of the one-vs-rest decision scores.
        """
        scores: np.ndarray = self.decision_function(X)
        if self._ovr is not None:
            return self.classes_[np.argmax(scores, axis=1)]
        pred: np.ndarray = self._sign(scores)
        return np.where(pred == 1.0, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean classification accuracy."""
        return accuracy_score(y, self.predict(X))
