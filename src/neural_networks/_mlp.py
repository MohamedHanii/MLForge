"""Multi-Layer Perceptron (MLP) built from modular layers.

An MLP is a stack of layers. Training repeats one cycle many times (each pass
is an "epoch"):

    1. Forward pass   - run the input through every layer to get a prediction.
    2. Measure error  - compare the prediction to the true target y.
    3. Backward pass  - use the chain rule to find each weight's gradient
                        (how much it contributed to the error).
    4. Update weights - move every weight a small step against its gradient:
                        W = W - learning_rate * gradient.

Each layer implements its own slice of this cycle:
    __call__()  -> forward pass
    backward()  -> backward pass (chain rule)
    update()    -> apply the weight step (only layers that have weights)

MLPRegressor.fit() ties the four steps together in a training loop.
"""
from __future__ import annotations

import numpy as np


class ModularLinearLayer:
    """A dense (fully connected) layer computing  out = X @ weight + bias."""

    def __init__(self, input_size: int, output_size: int,
                 rng: np.random.Generator | None = None) -> None:
        # Xavier/Glorot init: scale the random weights by sqrt(2/(in+out)) so
        # signals neither blow up nor shrink to zero as they pass through layers.
        limit = np.sqrt(2 / (input_size + output_size))
        if rng is not None:
            self.weight: np.ndarray = rng.standard_normal(
                (input_size, output_size)) * limit
        else:
            self.weight = np.random.randn(input_size, output_size) * limit
        self.bias: np.ndarray = np.zeros(output_size)

        self.input_size = input_size
        self.output_size = output_size

        # Set during backward(); read by update(). Empty until then.
        self._weight_grad: np.ndarray = np.empty(0)
        self._bias_grad: np.ndarray = np.empty(0)
        # The forward input, remembered so backward() can compute the gradient.
        self._prev_input: np.ndarray = np.empty(0)

    @property
    def in_features(self) -> int:
        """Alias for input_size."""
        return self.input_size

    @property
    def out_features(self) -> int:
        """Alias for output_size."""
        return self.output_size

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: linear transform, caching the input for backward()."""
        self._prev_input = X
        out: np.ndarray = X @ self.weight + self.bias
        return out

    def backward(self, grad_output: np.ndarray,
                 alpha: float | None = None) -> np.ndarray:
        """Backward pass: compute weight/bias gradients, return input gradient.

        ``grad_output`` is dLoss/d(output) from the next layer. If ``alpha`` is
        given, the L2 (weight-decay) term ``alpha * weight`` is added to the
        weight gradient. Returns dLoss/d(input) for the previous layer.
        """
        # A single-output layer may receive a 1-D gradient (shape (n_samples,));
        # reshape it to (n_samples, out_features) so the matrix math lines up.
        grad_output = np.asarray(grad_output)
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, self.out_features)
        self._weight_grad = self._prev_input.T @ grad_output
        self._bias_grad = np.sum(grad_output, axis=0)
        if alpha:
            self._weight_grad = self._weight_grad + alpha * self.weight
        grad_input: np.ndarray = grad_output @ self.weight.T
        return grad_input

    def update(self, learning_rate: float) -> None:
        """Gradient-descent step: move each parameter against its gradient."""
        self.weight = self.weight - learning_rate * self._weight_grad
        self.bias = self.bias - learning_rate * self._bias_grad

    def grad_norm_sq(self) -> float:
        """Return the squared L2 norm of this layer's weight+bias gradients."""
        return float(np.sum(self._weight_grad ** 2)
                     + np.sum(self._bias_grad ** 2))

    def scale_gradients(self, factor: float) -> None:
        """Multiply the stored gradients by ``factor`` (used for clipping)."""
        self._weight_grad = self._weight_grad * factor
        self._bias_grad = self._bias_grad * factor

    def __repr__(self) -> str:
        return (f"ModularLinearLayer(input_size={self.input_size}, "
                f"output_size={self.output_size})")


class SigmoidLayer:
    """Sigmoid activation: squashes any value into the range (0, 1)."""

    def __init__(self) -> None:
        self.output: np.ndarray = np.empty(0)
        self._prev_result: np.ndarray = np.empty(0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: clip the exponent to avoid exp() overflow."""
        self.output = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        self._prev_result = self.output
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: derivative of sigmoid is sigmoid * (1 - sigmoid)."""
        grad_input: np.ndarray = grad_output * self.output * (1 - self.output)
        return grad_input


class TanhLayer:
    """Tanh activation: squashes any value into the range (-1, 1)."""

    def __init__(self) -> None:
        self.output: np.ndarray = np.empty(0)
        self._prev_result: np.ndarray = np.empty(0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: tanh(X)."""
        self.output = np.tanh(X)
        self._prev_result = self.output
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: derivative of tanh is (1 - tanh^2)."""
        grad_input: np.ndarray = grad_output * (1 - self.output ** 2)
        return grad_input


class ReLULayer:
    """ReLU activation: keeps positives, sets negatives to 0."""

    def __init__(self) -> None:
        self.input: np.ndarray = np.empty(0)
        self._prev_result: np.ndarray = np.empty(0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: max(0, X), caching the input for backward()."""
        self.input = X
        self._prev_result = X
        out: np.ndarray = np.maximum(0, X)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: gradient passes only where the input was positive."""
        grad_input: np.ndarray = grad_output * (self.input > 0)
        return grad_input


class SoftmaxLayer:
    """Softmax activation: turns a row of scores into probabilities summing to 1."""

    def __init__(self) -> None:
        self.output: np.ndarray = np.empty(0)
        self._prev_result: np.ndarray = np.empty(0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: exp(X)/sum(exp(X)), row-wise and overflow-safe."""
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self._prev_result = self.output
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: softmax Jacobian applied to the incoming gradient."""
        grad_input: np.ndarray = self.output * (
            grad_output - np.sum(grad_output * self.output,
                                 axis=1, keepdims=True))
        return grad_input


class MLPRegressor:
    """MLP for regression: hidden layers with activations, plain linear output."""

    def __init__(self, hidden_layer_sizes: tuple[int, ...] | int = (50, 30),
                 lr: float = 0.01, epochs: int = 100,
                 random_state: int | None = None, alpha: float = 0.0001,
                 activation: str = 'relu',
                 learning_rate: float | None = None,
                 n_iterations: int | None = None,
                 max_iter: int | None = None) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        # lr / learning_rate are accepted names for the same setting.
        self.lr = learning_rate if learning_rate is not None else lr
        # epochs / n_iterations / max_iter are accepted names; a given alias
        # wins over the plain default of 100.
        self.epochs = next((v for v in (n_iterations, max_iter, epochs)
                            if v is not None), 100)
        self.random_state = random_state
        self.alpha = alpha              # L2 regularization strength
        self.activation = activation    # activation used in the hidden layers

        # Max allowed global gradient norm (not a guide constructor param).
        # Defaults to a large value that never interferes with healthy training
        # (typical norms are ~1) but caps runaway "exploding" gradients from an
        # aggressive learning rate, preventing overflow/NaN. Set to None to
        # disable clipping entirely.
        self.clip_grad_norm: float | None = 10.0

        # Early-stopping controls (not guide constructor params): training stops
        # once the loss fails to improve by more than `tol` for
        # `n_iter_no_change` consecutive epochs. Set n_iter_no_change to None to
        # always run the full `epochs`.
        self.tol = 1e-4
        self.n_iter_no_change: int | None = 10

        # Learned state, populated by fit().
        self.layers_: list = []
        self.n_features_in_: int = 0
        self.n_outputs_: int = 0
        self.loss_curve_: list[float] = []
        self.best_loss_: float = np.inf
        self.n_iter_: int = 0
        self._x_mean: np.ndarray = np.empty(0)
        self._x_std: np.ndarray = np.empty(0)
        self._y_mean: np.ndarray = np.empty(0)
        self._y_std: np.ndarray = np.empty(0)

    # Generate by LLM
    @staticmethod
    def _check_2d_float(arr, name: str) -> np.ndarray:
        """Coerce `arr` to a finite 2-D float array or raise a clear error."""
        try:
            arr = np.asarray(arr, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} could not be converted to a numeric "
                             f"array: {exc}") from exc
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2-D with shape "
                             f"(n_samples, n_features); got a {arr.ndim}-D array.")
        if arr.size == 0:
            raise ValueError(f"{name} is empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return arr
    
    # Generate by LLM
    def _validate_fit_input(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Validate and coerce fit() inputs; return finite float arrays."""
        X = self._check_2d_float(X, "X")
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"y could not be converted to a numeric "
                             f"array: {exc}") from exc
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y have mismatched sample counts: "
                             f"X has {X.shape[0]}, y has {y.shape[0]}.")
        return X, y
    
    # Generate by LLM
    def _build_layers(self, n_features: int, n_outputs: int,
                      rng: np.random.Generator) -> list:
        """Assemble [Linear -> Activation] blocks plus a final linear output."""
        activation_map = {
            'relu': ReLULayer,
            'sigmoid': SigmoidLayer,
            'tanh': TanhLayer,
        }
        activation_cls = activation_map[self.activation]

        # Normalize hidden_layer_sizes into positive layer widths. A single int
        # means one hidden layer; non-positive sizes are dropped, so an empty
        # result reduces the network to a single linear layer.
        if isinstance(self.hidden_layer_sizes, int):
            hidden_sizes = [self.hidden_layer_sizes]
        else:
            hidden_sizes = list(self.hidden_layer_sizes)
        hidden_sizes = [h for h in hidden_sizes if h > 0]

        layers: list = []
        prev_size = n_features
        for hidden_size in hidden_sizes:
            layers.append(ModularLinearLayer(prev_size, hidden_size, rng=rng))
            layers.append(activation_cls())
            prev_size = hidden_size
        layers.append(ModularLinearLayer(prev_size, n_outputs, rng=rng))
        return layers

    def _clip_gradients(self) -> None:
        """Global-norm gradient clipping: scale all gradients if too large."""
        if self.clip_grad_norm is None:
            return
        weighted = [layer for layer in self.layers_
                    if isinstance(layer, ModularLinearLayer)]
        total_norm = np.sqrt(sum(layer.grad_norm_sq() for layer in weighted))
        if total_norm > self.clip_grad_norm:
            scale = self.clip_grad_norm / (total_norm + 1e-12)
            for layer in weighted:
                layer.scale_gradients(scale)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Train the network on features X and targets y; returns self."""
        X, y = self._validate_fit_input(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        if y.ndim == 1:
            y = y.reshape(-1, 1)   # 2-D so shapes line up with the output
        self.n_outputs_ = y.shape[1]

        # Standardize inputs and targets (zero mean, unit variance). Networks
        # diverge on unscaled, large-magnitude features; centering/scaling keeps
        # the forward products and gradients well-conditioned. The stats are
        # stored so predict() can apply the same transform and invert it.
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0   # guard constant features
        self._y_mean = y.mean(axis=0)
        self._y_std = y.std(axis=0)
        self._y_std[self._y_std == 0] = 1.0
        X = (X - self._x_mean) / self._x_std
        y = (y - self._y_mean) / self._y_std

        rng = np.random.default_rng(self.random_state)
        self.layers_ = self._build_layers(n_features, self.n_outputs_, rng)

        # Convergence monitoring + early stopping.
        self.loss_curve_ = []
        self.best_loss_ = np.inf
        no_improve = 0
        for _ in range(self.epochs):
            # 1. Forward pass.
            out = X
            for layer in self.layers_:
                out = layer(out)

            # Objective (recorded per epoch) = data term + L2 penalty:
            # 0.5*SSE/n + 0.5*alpha*||W||^2. This is exactly the objective the
            # update below descends, so the reported loss is consistent with the
            # gradients.
            data_loss = 0.5 * np.sum((out - y) ** 2) / n_samples
            reg_loss = 0.5 * self.alpha * sum(
                float(np.sum(layer.weight ** 2)) for layer in self.layers_
                if isinstance(layer, ModularLinearLayer))
            loss = float(data_loss + reg_loss)
            self.loss_curve_.append(loss)

            # 2. Error signal = gradient of the data term w.r.t. the output.
            grad = out - y

            # 3. Backward pass (output -> input); linear layers add the L2
            #    gradient alpha*W.
            for layer in reversed(self.layers_):
                if isinstance(layer, ModularLinearLayer):
                    grad = layer.backward(grad, self.alpha)
                else:
                    grad = layer.backward(grad)

            # 3b. Optional gradient clipping to tame exploding gradients.
            self._clip_gradients()

            # 4. Gradient-descent update on every weighted layer.
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    layer.update(self.lr)

            # 5. Early stopping: stop once the loss has plateaued.
            if loss < self.best_loss_ - self.tol:
                self.best_loss_ = loss
                no_improve = 0
            else:
                no_improve += 1
                if (self.n_iter_no_change is not None
                        and no_improve >= self.n_iter_no_change):
                    break

        self.n_iter_ = len(self.loss_curve_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X.

        Shape is (n_samples,) for a single target, or (n_samples, n_outputs)
        for multi-output regression.
        """
        if not self.layers_:
            raise ValueError("This MLPRegressor is not fitted yet; "
                             "call fit() before predict().")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._check_2d_float(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but this model was "
                             f"trained on {self.n_features_in_}.")

        # Same standardization as in fit, then a single forward pass.
        out = (X - self._x_mean) / self._x_std
        for layer in self.layers_:
            out = layer(out)
        # Undo the target standardization to return values on the original scale.
        out = out * self._y_std + self._y_mean
        result: np.ndarray = out.ravel() if self.n_outputs_ == 1 else out
        return result
