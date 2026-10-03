"""Support Vector Machine for regression (epsilon-SVR)."""
from typing import Any, Dict, Optional
import numpy as np
from ._svm import SVM


class SVR(SVM):
    """
    Support Vector Machine for regression (ε-SVR).

    Constructor:  SVR(kernel='rbf', C=1.0, epsilon=0.1, degree=3)

    Linear kernel → primal gradient descent on ε-insensitive loss
    RBF / Poly    → dual sub-gradient on kernel matrix

    Features (X) and targets (y) are automatically standardised.
    Predictions are un-scaled back to the original y range.
    """

    _estimator_type = "regressor"

    def __init__(self, kernel: str = 'rbf', C: float = 1.0,
                 epsilon: float = 0.1, degree: int = 3,
                 gamma: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        super().__init__(kernel=kernel, C=C, degree=degree,
                         gamma=gamma, random_state=random_state)
        self.epsilon: float = epsilon
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVR':
        """
        Train the SVR on the given data.

        Step 1: standardise targets  y' = (y − μ_y) / σ_y
        Step 2: standardise features  x' = (x − μ_x) / σ_x
        Step 3: train via primal GD (linear) or dual Pegasos (kernel)

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) with continuous target values
        """
        self._validate_params()
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0. Got epsilon={self.epsilon}.")

        X, y = self._validate_input(X, y)
        y = y.ravel()

        # ── 1. Standardise targets:  y' = (y − μ_y) / σ_y ──
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y))
        if self._y_std < 1e-12:
            self._y_std = 1.0
        self._y_train = (y - self._y_mean) / self._y_std

        # ── 2 & 3. Standardise features and train ──
        self._standardize_and_train(X)
        return self

    def _fit_linear(self) -> None:
        """
        Primal gradient descent on ε-insensitive loss.

        Cost function:
                          1   m
            J(w, b) =   ───  Σ  max(0, |yⁱ − (wᵀxⁱ + b)| − ε)   +   λ ||w||²
                          m  i=1
                          ╰──────── ε-insensitive loss ───────╯    ╰─ regularisation ─╯

        where  λ = 1 / (C · m).

        Targets are standardised (σ_y = 1), so ε is scale-invariant.

        Gradients (per sample i), with residual  rⁱ = yⁱ − (wᵀxⁱ + b):

            ┌──────────────────────────────┬──────────────────────────────┐
            │          |rⁱ| ≤ ε            │    ∇w = 2λw    ∇b = 0        │
            │  (prediction inside ε-tube)  │                              │
            ├──────────────────────────────┼──────────────────────────────┤
            │          rⁱ > ε              │    ∇w = 2λw − xⁱ   ∇b = −1  │
            │  (prediction below target)   │                              │
            ├──────────────────────────────┼──────────────────────────────┤
            │          rⁱ < −ε             │    ∇w = 2λw + xⁱ   ∇b = +1  │
            │  (prediction above target)   │                              │
            └──────────────────────────────┴──────────────────────────────┘
        """
        def sample_update(i: int, lam: float) -> float:
            # Residual:  r = yⁱ − (wᵀxⁱ + b)
            r: float = float(self._y_train[i] - (np.dot(self.weight_, self.X_train_[i])
                                                 + self.bias_))

            if r > self.epsilon:
                # Below target → ∇w = 2λw − xⁱ,   ∇b = −1
                # w := w − α (2λ w − xⁱ)
                self.weight_ -= self._lr * (2 * lam * self.weight_
                                            - self.X_train_[i])
                # b := b − α (−1)
                self.bias_ -= self._lr * (-1.0)
            elif r < -self.epsilon:
                # Above target → ∇w = 2λw + xⁱ,   ∇b = +1
                # w := w − α (2λ w + xⁱ)
                self.weight_ -= self._lr * (2 * lam * self.weight_
                                            + self.X_train_[i])
                # b := b − α (1)
                self.bias_ -= self._lr * 1.0
            else:
                # Inside ε-tube → only regularisation
                # w := w − α (2λ w)
                self.weight_ -= self._lr * (2 * lam * self.weight_)

            # ε-insensitive loss:  max(0, |r| − ε)
            return max(0.0, abs(r) - self.epsilon)

        final_loss, converged = self._run_primal_gd(sample_update)
        if not converged:
            self._warn_not_converged("Linear SVR", f"Final loss: {final_loss:.6f}. ")

        # Identify support vectors (points outside or near the ε-tube)
        residuals = np.abs(self._y_train - (self.X_train_.dot(self.weight_)
                                            + self.bias_))
        threshold: float = float(np.percentile(residuals, 70))
        self._set_support_vectors(residuals >= threshold, uniform_alpha=True)

    def _fit_kernel(self, K_train: np.ndarray) -> None:
        """
        Dual sub-gradient descent for ε-SVR with kernel matrix K.

        Decision function:
            f(x) = Σ_j θ_j · K(x_j, x) + b    where  θ_j ∈ [−C, C]

        For each sample i, with residual  rⁱ = yⁱ − f(xⁱ):

            ┌───────────────────────┬──────────────────────────────────────┐
            │    |rⁱ| ≤ ε           │    no update                         │
            │    (inside ε-tube)    │                                      │
            ├───────────────────────┼──────────────────────────────────────┤
            │    rⁱ > ε             │    θ_i += α · r                      │
            │    (below target)     │    θ_i  = clip(θ_i, −C, C)           │
            ├───────────────────────┼──────────────────────────────────────┤
            │    rⁱ < −ε            │    θ_i += α · r                      │
            │    (above target)     │    θ_i  = clip(θ_i, −C, C)           │
            └───────────────────────┴──────────────────────────────────────┘
        """
        n_samples: int = K_train.shape[0]
        theta: np.ndarray = np.zeros(n_samples)
        self.bias_ = 0.0

        prev_theta: Optional[np.ndarray] = None
        for _ in range(self._max_iter):
            # Deterministic sweep order so the dual coefficients converge to the
            # same fixed point regardless of random_state (seed-invariant result).
            for i in range(n_samples):
                # f(xⁱ) = Σ_j θ_j · K(x_j, xⁱ) + b
                f_i: float = float(np.dot(theta, K_train[:, i]) + self.bias_)

                # Residual:  r = yⁱ − f(xⁱ)
                r: float = float(self._y_train[i] - f_i)

                if abs(r) > self.epsilon:
                    # θ_i := θ_i + α · r           (sub-gradient step)
                    theta[i] += self._lr * r
                    # Clip to feasible box:  −C ≤ θ_i ≤ C
                    theta[i] = np.clip(theta[i], -self.C, self.C)
                    # b := b + α · r
                    self.bias_ += self._lr * r

            if prev_theta is not None:
                delta: float = float(np.max(np.abs(theta - prev_theta)))
                if delta < self._tol:
                    break
            prev_theta = theta.copy()
        else:
            self._warn_not_converged("Kernel SVR")

        # Store dual coefficients for inherited decision_function
        self._dual_coef_ = theta
        self.alpha_ = theta.copy()

        # Support vectors = samples with non-zero θ
        self._set_support_vectors(np.abs(theta) > 1e-10)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input.

            ŷ = f(x) · σ_y + μ_y            (un-scale from standardised y)

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,)
        """
        return self.decision_function(X) * self._y_std + self._y_mean

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return hyper-parameters, including SVR's epsilon (sklearn-style)."""
        params = super().get_params(deep=deep)
        params['epsilon'] = self.epsilon
        return params

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the R² coefficient of determination.

                     Σ (y_i − ŷ_i)²
            R² = 1 − ───────────────
                     Σ (y_i − ȳ)²
        """
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        ss_res: float = float(np.sum((y - y_pred) ** 2))
        ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)
