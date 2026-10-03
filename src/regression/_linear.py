import numpy as np
from typing import Optional, Any


# ==============================================================================
#  UTILITIES
# ==============================================================================


class RegressorMixin:
    """R² score mixin. Requires predict() to be implemented."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R² coefficient of determination.

        R² measures the proportion of variance in the dependent variable
        explained by the model. Formula: R² = 1 - SS_res / SS_tot

        Returns 0.0 when the total sum of squares is zero (i.e., the
        target is constant), since no meaningful variance exists to explain.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        if v == 0:
            return 0.0
        return 1 - u / v

# Generate by GENAI
def check_array(
    X: Any, ensure_2d: bool = True, ensure_min_samples: int = 1
) -> np.ndarray:
    """Input validation gatekeeper — ensures data is a well-formed numeric array.

    Performs the following checks in order:
    1. Rejects None input
    2. Converts to a numpy array
    3. Rejects arrays containing NaN or infinity (non-finite values)
    4. Reshapes 1D arrays to 2D column vectors (if ensure_2d=True)
    5. Ensures the array has at least ensure_min_samples rows

    Used before every model fitting and prediction to catch malformed data early.
    """
    if X is None:
        raise ValueError("Input X cannot be None.")
    X = np.asarray(X)
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinity.")
    if ensure_2d and X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {X.shape[0]} sample(s) "
            f"(expected at least {ensure_min_samples})."
        )
    return X

# Generate by GENAI
def check_X_y(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate the feature matrix X and target vector y as a pair.

    Calls check_array() on X, converts y to a numpy array, then verifies
    that X and y have the same number of rows (samples). This ensures each
    feature row maps to exactly one target value.

    Returns the validated (X, y) pair as numpy arrays.
    """
    X = check_array(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Inconsistent numbers of samples: [{X.shape[0]}, {y.shape[0]}]"
        )
    return X, y

# Generate by GENAI
def check_is_fitted(estimator: Any, attributes: Optional[list[str]] = None):
    """Ensure an estimator has been fitted before prediction or transformation.

    Checks that the specified attributes (default: 'coef_' and 'intercept_')
    exist on the estimator and are not None. Raises ValueError with a
    descriptive message if the estimator hasn't been fitted yet — this
    prevents silent errors from using an untrained model.
    """
    if attributes is None:
        attributes = ["coef_", "intercept_"]
    if not all(
        hasattr(estimator, attr) and getattr(estimator, attr) is not None
        for attr in attributes
    ):
        raise ValueError(
            f"This {estimator.__class__.__name__} instance is not fitted yet."
        )


# ==============================================================================
#  PREPROCESSING
# ==============================================================================


class StandardScaler:
    """Standardization (Z-score normalization) — centers and scales each feature.

    Transforms every feature to have zero mean and unit variance:
        x' = (x - μ) / (σ + ε)   where ε = 1e-7

    This is a standard preprocessing step for gradient-based optimizers
    (like SGD) and distance-based algorithms. Features with larger numeric
    scales can dominate the optimization; standardization puts them all on
    an equal footing.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute the per-feature mean (μ) and standard deviation (σ) from X.

        Stores these statistics as self.mean_ and self.std_ for later use
        in transform(). Returns self to allow method chaining.
        """
        X = check_array(X)
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned standardization to X: subtract μ, divide by σ.

        Uses the mean_ and std_ statistics computed during fit().
        Requires fit() to have been called first (enforced by check_is_fitted).
        """
        check_is_fitted(self, attributes=["mean_", "std_"])
        X = check_array(X)
        X = X.copy().astype(float)
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.std_ + 1e-7
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler to X and return the standardized version in one call.

        Convenience shortcut for scaler.fit(X).transform(X).
        """
        return self.fit(X).transform(X)


# ==============================================================================
#  MODELS
# ==============================================================================


class LinearRegression(RegressorMixin):
    """Ridge Regression — Linear Regression with L2 regularization.

    Ridge adds a penalty proportional to the squared coefficient magnitudes
    (α·||w||²) to the ordinary least squares (OLS) cost function. This
    shrinks coefficients toward zero, which reduces overfitting and handles
    multicollinearity (correlated features) better than plain OLS.

    Solved via the Normal Equation with Moore-Penrose pseudo-inverse:
        θ = pinv(X_bᵀ·X_b + α·I') · X_bᵀ·y
    where X_b = [1|X] (a bias column of ones is prepended) and I' is an
    identity matrix with I'[0,0]=0 so the bias/intercept term is NOT penalized.
    Only the feature weights receive regularization.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit Ridge Regression using the closed-form Normal Equation.

        Solves the regularized least-squares problem directly via matrix
        pseudo-inverse. Computes the optimal weight vector θ in one step:
            θ = pinv(X_bᵀ·X_b + α·I') · X_bᵀ·y

        Stores the bias term in self.intercept_ and feature weights
        (coefficients) in self.coef_. Returns self for method chaining.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        X_b = np.c_[np.ones((n_samples, 1)), X]

        I = np.eye(n_features + 1)
        I[0, 0] = 0  # No penalty on bias

        weights = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using the fitted linear model.

        Computes ŷ = X·w + b where w = self.coef_ (feature weights)
        and b = self.intercept_ (bias / intercept term).

        Requires fit() to have been called first.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_


class SGDRegression(RegressorMixin):
    """Linear Regression trained via Mini-batch Stochastic Gradient Descent (SGD).

    Unlike the Normal Equation approach (which solves for weights in one step),
    SGD is an iterative optimization algorithm. At each iteration it picks a
    small random batch of samples, computes the gradient of the cost, and takes
    a small step downhill. This makes it memory-efficient and suitable for
    very large datasets.

    Cost function (Mean Squared Error + L2 penalty):
        J(w,b) = (1/m)·Σ(ŷ - y)² + α·Σ(wⱼ)²

    Gradient update rules:
        w := w - η·[(2/m)·Xᵀ·(ŷ - y) + 2α·w]
        b := b - η·(2/m)·Σ(ŷ - y)

    Gradient clipping at ±1e6 prevents exploding gradients from destabilizing
    training. Features MUST be standardized (e.g., via StandardScaler) before
    fitting for stable and efficient convergence.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        alpha: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.alpha = alpha
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegression":
        """Train the model using mini-batch Stochastic Gradient Descent.

        At each of n_iterations, the data is randomly shuffled (stochastic
        sampling without replacement), then processed in mini-batches of
        batch_size. For each batch:
          1. Compute prediction error:  ŷ - y  =  X·w + b - y
          2. Compute weight gradient:  dw = (2/m)·Xᵀ·(ŷ-y) + 2α·w
          3. Compute bias gradient:    db = (2/m)·Σ(ŷ-y)
          4. Clip gradients to [-1e6, 1e6] for numerical stability
          5. Update:  w -= η·dw,  b -= η·db

        Stores learned weights as self.weights_ and bias as self.bias_.
        Returns self for method chaining.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                Xb = X[i:end]
                yb = y[i:end]
                m = end - i

                error = Xb @ self.weights_ + self.bias_ - yb

                dw = (2.0 / m) * (Xb.T @ error) + 2 * self.alpha * self.weights_
                db = (2.0 / m) * error.sum()

                np.clip(dw, -1e6, 1e6, out=dw)
                db = np.clip(db, -1e6, 1e6)

                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using the fitted SGD model.

        Computes ŷ = X·w + b where w = self.weights_ (feature weights)
        and b = self.bias_ (bias / intercept term).

        Requires fit() to have been called first.
        """
        check_is_fitted(self, attributes=["weights_", "bias_"])
        X = check_array(X)
        return X @ self.weights_ + self.bias_
