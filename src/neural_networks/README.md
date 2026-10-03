# `neural_networks`: MLP and CNN from scratch

Neural networks written as stacks of small, modular NumPy layers. Each layer runs its own forward pass, backward pass and weight update.

| File | Contents |
|---|---|
| `_mlp.py` | `MLPRegressor`, plus the layers `ModularLinearLayer`, `ReLULayer`, `SigmoidLayer`, `TanhLayer`, `SoftmaxLayer` |
| `_cnn.py` | `CNNClassifier`, plus the layers `ConvLayer`, `PoolingLayer`, `ReLULayer`, `SoftmaxLayer`, `ModularLinearLayer`, and the helpers `get_patches` and `pad_images` |

Dependencies: `numpy`. The two files are independent. Each defines its own layer classes with the same names, so don't mix layers from one file with models from the other.

---

## The layer contract

Every layer follows the same small protocol, and the models chain layers together through it:

| Method | Role |
|---|---|
| `layer(X)` (`__call__`) | Forward pass. Caches whatever the backward pass needs. |
| `layer.backward(grad_output[, alpha])` | Chain rule: returns the gradient with respect to the layer's input and stores the parameter gradients. Linear and conv layers take an optional L2 `alpha`. |
| `layer.update(learning_rate[, momentum])` | Applies the gradient step. Only layers with weights have this method. |

One training epoch therefore runs: **forward → loss → backward (in reverse layer order) → update**.

---

## `MLPRegressor` (`_mlp.py`)

A fully connected network for regression. Each hidden layer is `ModularLinearLayer → activation`, and a final linear layer produces the output.

### `MLPRegressor(hidden_layer_sizes=(50, 30), lr=0.01, epochs=100, random_state=None, alpha=1e-4, activation='relu')`

| Parameter | Description |
|---|---|
| `hidden_layer_sizes` | Tuple of hidden widths, or a single `int`. Sizes ≤ 0 are dropped, and an empty list gives plain linear regression. |
| `lr` | Learning rate. `learning_rate=` is accepted as an alias. |
| `epochs` | Maximum number of full-batch epochs. `n_iterations=` and `max_iter=` are accepted as aliases. |
| `random_state` | Seed for the Xavier/Glorot weight initialisation. |
| `alpha` | L2 regularisation strength. |
| `activation` | `'relu'`, `'sigmoid'` or `'tanh'`. |

Training details:

- **Automatic standardisation.** `X` and `y` are z-scored inside `fit`, and `predict` undoes the scaling. Pass raw data.
- **Objective.** Full-batch gradient descent on `0.5·SSE/n + 0.5·α·‖W‖²`.
- **Gradient clipping.** The global norm is capped at `clip_grad_norm = 10.0`. Set the attribute to `None` to disable it.
- **Early stopping.** Training stops when the loss fails to improve by more than `tol = 1e-4` for `n_iter_no_change = 10` epochs. Set `n_iter_no_change = None` to always run every epoch.

| Method / attribute | Description |
|---|---|
| `fit(X, y)` → `self` | `X` must be **2-D** `(n_samples, n_features)`. `y` can be `(n,)` or `(n, n_outputs)`. |
| `predict(X)` | Shape `(n,)` for a single output, otherwise `(n, n_outputs)`. |
| `loss_curve_`, `best_loss_`, `n_iter_` | Training diagnostics. |
| `layers_`, `n_features_in_`, `n_outputs_` | The fitted network. |

There is no `score` method. Compute R² yourself (see the example).

### Example

```python
import numpy as np
from src.neural_networks._mlp import MLPRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(500, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=500)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), lr=0.01, epochs=500,
                   activation="relu", random_state=0).fit(X, y)
pred = mlp.predict(X)
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"R² = {r2:.3f} after {mlp.n_iter_} epochs (final loss {mlp.loss_curve_[-1]:.4f})")
```

---

## `CNNClassifier` (`_cnn.py`)

A convolutional network for image classification. It chains feature layers, flattens their output, and adds a linear layer and a softmax head. It trains with mini-batch SGD with momentum on the cross-entropy loss.

**All images are NHWC:** `(n_samples, height, width, channels)`. A 3-D batch is accepted when `channels == 1`.

### `CNNClassifier(input_shape=(28, 28, 1), num_classes=10, lr=0.01, epochs=20, batch_size=32, random_state=None, layers=None, alpha=0.0, verbose=True, momentum=0.9, lr_decay=0.0, early_stopping=True, tol=1e-4, n_iter_no_change=5, max_grad_norm=None)`

| Parameter | Description |
|---|---|
| `input_shape` | `(H, W, C)` of one image. `fit` checks the input against it. |
| `num_classes` | Number of output classes. It grows automatically if `y` contains a larger label. |
| `lr`, `momentum`, `lr_decay` | SGD step size, momentum in `[0, 1)`, and inverse-time decay `lr / (1 + lr_decay·epoch)`. |
| `epochs`, `batch_size` | Training length and mini-batch size. |
| `alpha` | L2 weight decay. |
| `layers` | Custom feature-layer list. `None` builds the default stack (see below). |
| `early_stopping`, `tol`, `n_iter_no_change` | Stop when the **training** loss plateaus. |
| `max_grad_norm` | Optional global gradient-norm clipping. |
| `verbose` | Print the loss for every epoch. |
| `random_state` | Seed for the weights and the batch shuffling. |

Default feature stack when `layers=None`:

```
Conv(C→8, 3×3, same) → ReLU → [MaxPool 2×2]* → Conv(8→16, 3×3, same) → ReLU → [MaxPool 2×2]*
  * each pool is added only while the feature map is still ≥ 8×8, so small inputs keep their spatial detail
→ flatten → Linear → Softmax
```

| Method / attribute | Description |
|---|---|
| `fit(X, y)` → `self` | `y` must be non-negative integer labels `0…k-1`. |
| `predict(X)` | Predicted labels (argmax of the softmax output). Prediction runs in chunks to limit memory use. |
| `score(X, y)` | Accuracy. |
| `layers_`, `out_layer_`, `softmax_layer_` | The fitted network. |

### Building blocks

| Class / function | Description |
|---|---|
| `ConvLayer(in_channels, out_channels, kernel_size, stride=1, padding=0, rng=None)` | 2-D convolution with Xavier-uniform initialisation. `padding` is an int, `'same'` or `'valid'`. `get_output_shape((H, W, C))` returns the output size. |
| `PoolingLayer(pool_size=2, stride=2, padding=0)` | Max pooling. During the backward pass, the gradient goes only to each window's maximum. |
| `ReLULayer()`, `SoftmaxLayer()` | Activations. |
| `ModularLinearLayer(input_size, output_size, rng=None)` | Dense layer used for the classification head. |
| `get_patches(arr, patch_shape, strides=(1, 1))` | Sliding-window patch extraction with strided views (no copies). Convolution and pooling are built on it. |
| `pad_images(X, padding)` | Zero-pads the spatial dimensions. |

### Example

```python
import numpy as np
from src.neural_networks._cnn import CNNClassifier, ConvLayer, ReLULayer, PoolingLayer

rng = np.random.default_rng(0)
# Toy 8×8 grayscale images: class 1 = bright left half, class 0 = bright right half
X = rng.normal(0, 0.3, size=(200, 8, 8, 1))
y = rng.integers(0, 2, size=200)
X[y == 1, :, :4, 0] += 1.0
X[y == 0, :, 4:, 0] += 1.0

cnn = CNNClassifier(input_shape=(8, 8, 1), num_classes=2, lr=0.01, epochs=10,
                    batch_size=32, random_state=0, verbose=False).fit(X, y)
print("accuracy:", cnn.score(X, y))

# Custom architecture
custom = CNNClassifier(
    input_shape=(8, 8, 1), num_classes=2, epochs=5, verbose=False, random_state=0,
    layers=[ConvLayer(1, 4, 3, padding="same", rng=np.random.RandomState(0)),
            ReLULayer(), PoolingLayer(2, 2)],
).fit(X, y)
print("custom accuracy:", custom.score(X, y))
```

---

## Notes

- Everything runs on the CPU in NumPy. The CNN is meant for small images (8×8 digits, 28×28 MNIST subsets) and modest dataset sizes.
- A custom `layers` list is used as given (it isn't copied), so calling `fit` again keeps training the same layer objects. Create fresh layers to retrain from scratch.
- `CNNClassifier` doesn't normalise its input. Scale pixel values yourself (for example `X / 255.0`).
- The [concrete-strength notebook](../../notebooks/README.md) uses `MLPRegressor(hidden_layer_sizes=(64, 32), lr=0.01)` and reaches test R² 0.861.
