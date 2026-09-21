"""
Final Optimized Logistic Regression - Pylint-Friendly & Performance-Tuned.
Strictly aligned with Student Guide API while automatically applying best practices.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants to avoid magic numbers
EPSILON = 1e-15
CLIP_BOUND = 500.0

class _BaseLogistic:
    """
    Private base class for Logistic Regression models.
    Hardcoded with best practices (Balanced weights, L2, Early Stopping).
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Initialize the base logistic regression model.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size (η) for gradient descent updates.
        n_iterations : int, default=1000
            Maximum number of training epochs.

        Internal best-practice defaults — L2 regularization (λ=0.1),
        balanced class weighting, and early stopping (tol=1e-5) — are
        hardcoded so the model behaves robustly out of the box without
        requiring the user to tune these themselves.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Internal configuration (Best Practices)
        self.lambda_ = 0.1
        self.class_weight = 'balanced'
        self.tol = 1e-5
        
        # State variables
        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []

    def predict(self, X):
        """
        Predict binary class labels (0 or 1).

        A sample is assigned class 1 if its predicted probability P(y=1)
        is 0.5 or higher, class 0 otherwise.

        Requires fit() to have been called first.
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict class membership probabilities.

        Computes z = X @ w + b, then applies the sigmoid function to
        obtain P(y=1 | X). Returns a 2-column array [P(y=0), P(y=1)].

        Requires fit() to have been called first.
        """
        X = np.asarray(X, dtype=float)
        z = X @ self.weights_ + self.bias_
        p1 = self._sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def score(self, X, y):
        """
        Compute classification accuracy: the fraction of correctly
        predicted labels.

        Requires fit() to have been called first.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == np.asarray(y))

    @property
    def coef_(self):
        """Alias for learned weights."""
        return self.weights_
    
    @property
    def coefficients(self):
        """Alias for learned weights."""
        return self.weights_
    
    @property
    def theta(self):
        """Returns the full parameter vector [bias, weights]."""
        if self.weights_ is None:
            return None
        return np.concatenate([[self.bias_], self.weights_])

    def _sigmoid(self, z):
        """
        Apply the logistic (sigmoid) function: σ(z) = 1 / (1 + e^{-z}).

        Uses a numerically stable split:
        - For z ≥ 0: σ(z) = 1 / (1 + e^{-z}) (safe, e^{-z} ≤ 1).
        - For z < 0: σ(z) = e^{z} / (1 + e^{z}) (multiply numerator and
          denominator by e^{z} to avoid computing e^{-z} for large |z|).
        Values of z are clipped to [-CLIP_BOUND, CLIP_BOUND] to prevent overflow.
        """
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-np.clip(z, -CLIP_BOUND, CLIP_BOUND))), 
                        np.exp(np.clip(z, -CLIP_BOUND, CLIP_BOUND)) / 
                        (1 + np.exp(np.clip(z, -CLIP_BOUND, CLIP_BOUND))))

    def _compute_loss(self, y, p, weights):
        """
        Weighted binary cross-entropy loss with L2 regularization.

        L = -1/n Σ w_i [y_i log(p_i) + (1-y_i) log(1-p_i)] + (λ/2n) ||w||²

        Probabilities p_i are clipped to [EPSILON, 1-EPSILON] to avoid
        log(0). Sample weights w_i are computed from class frequencies
        to counteract imbalance — minority-class samples receive higher
        weight so the model pays more attention to them.
        """
        n_samples = len(y)
        p = np.clip(p, EPSILON, 1 - EPSILON)
        log_loss = -np.mean(weights * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        l2_penalty = (self.lambda_ / (2 * n_samples)) * np.sum(self.weights_**2)
        return log_loss + l2_penalty

    def _compute_sample_weights(self, y):
        """
        Compute per-sample weights to compensate for class imbalance.

        Each class is assigned weight w_c = n / (k * count_c), where n is the
        total number of samples, k is the number of classes, and count_c is
        the number of samples in class c. This ensures the minority class
        receives higher weight than the majority class, so the model does
        not learn to simply predict the dominant label all the time.
        """
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return np.ones(len(y))
        n_samples, n_classes = len(y), len(unique)
        class_weights = dict(zip(unique, n_samples / (n_classes * counts)))
        return np.array([class_weights[label] for label in y])

    def _validate(self, X, y):
        """Convert inputs to float arrays and check that X is 2D with
        matching number of samples in y."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("Shape mismatch")
        return X, y


class LogisticRegression(_BaseLogistic):
    """
    Logistic Regression trained with full-batch gradient descent.

    Algorithm:
    1. Linear model: z = X @ w + b computes the log-odds for each sample.
    2. Sigmoid activation: p = 1 / (1 + e^{-z}) maps log-odds to [0, 1].
    3. Cross-entropy loss: L = -1/n * Σ w_i [y_i log(p_i) + (1-y_i) log(1-p_i)]
       plus an L2 penalty (λ / 2n) * ||w||² to discourage large weights.
    4. Gradient computation: the gradient of the loss with respect to weights
       is (1/n) * Xᵀ (p - y) * sample_weights + (λ/n) * w.
    5. Gradient descent update: w ← w - lr * ∇_w,  b ← b - lr * ∇_b.
    6. Dynamic learning rate: lr is divided by (1 + 0.001 * iteration) to
       gradually reduce step size and improve convergence.
    7. Early stopping: training halts when the change in loss falls below tol.
    """
    def fit(self, X, y):
        """
        Train the model using full-batch gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : LogisticRegression
            The fitted model.
        """
        X, y = self._validate(X, y)
        n_samples, n_features = X.shape
        sample_weights = self._compute_sample_weights(y)
        
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for i in range(self.n_iterations):
            # Dynamic Learning Rate Schedule
            lr = self.learning_rate / (1 + 0.001 * i)
            
            z = X @ self.weights_ + self.bias_
            p = self._sigmoid(z)
            
            weighted_err = (p - y) * sample_weights
            dw = (1 / n_samples) * (X.T @ weighted_err) + (self.lambda_ / n_samples) * self.weights_
            db = (1 / n_samples) * np.sum(weighted_err)
            
            self.weights_ -= lr * dw
            self.bias_ -= lr * db
            
            loss = self._compute_loss(y, p, sample_weights)
            self.loss_history_.append(loss)
            
            # Early Stopping check
            if i > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                break
        return self


class SGDClassifier(_BaseLogistic):
    """
    Logistic Regression trained with mini-batch Stochastic Gradient Descent.

    Unlike full-batch GD which computes gradients over the entire dataset,
    SGD approximates the gradient using small random subsets (mini-batches).
    This makes each update cheaper and noisier, which often helps escape
    shallow local minima.

    Algorithm per epoch:
    1. Shuffle the dataset to break any ordering bias.
    2. Partition the shuffled data into batches of size batch_size.
    3. For each mini-batch, compute the gradient using only that subset,
       then apply a weight update immediately.
    4. The learning rate decays as lr / (1 + 0.01 * epoch).
    5. Track the mean batch loss per epoch; stop early when loss plateaus.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        """
        Parameters
        ----------
        learning_rate : float, default=0.01
            Initial step size for gradient updates.
        n_iterations : int, default=1000
            Number of epochs (full passes through the shuffled data).
        batch_size : int, default=32
            Number of samples per mini-batch gradient estimate.
        """
        super().__init__(learning_rate, n_iterations)
        self.batch_size = batch_size

    def fit(self, X, y):
        """
        Train the model using mini-batch stochastic gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : SGDClassifier
            The fitted model.
        """
        X, y = self._validate(X, y)
        n_samples, n_features = X.shape
        sample_weights = self._compute_sample_weights(y)
        
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for epoch in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            sw_shuffled = sample_weights[indices]
            
            lr = self.learning_rate / (1 + 0.01 * epoch)
            epoch_losses = []
            
            for i in range(0, n_samples, self.batch_size):
                end_idx = i + self.batch_size
                Xi = X_shuffled[i : end_idx]
                yi = y_shuffled[i : end_idx]
                swi = sw_shuffled[i : end_idx]
                ni = len(yi)
                
                zi = Xi @ self.weights_ + self.bias_
                pi = self._sigmoid(zi)
                
                weighted_err = (pi - yi) * swi
                dw = (1 / ni) * (Xi.T @ weighted_err) + (self.lambda_ / ni) * self.weights_
                db = (1 / ni) * np.sum(weighted_err)
                
                self.weights_ -= lr * dw
                self.bias_ -= lr * db
                epoch_losses.append(self._compute_loss(yi, pi, swi))
            
            curr_loss = np.mean(epoch_losses)
            self.loss_history_.append(curr_loss)
            if epoch > 0 and abs(self.loss_history_[-2] - curr_loss) < self.tol:
                break
        return self


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Generated by GENAI: StandardScaler, train_test_split, resample, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report, plot_decision_boundary
class StandardScaler:
    """
    Z-score standardization: for each feature j, transform x_j to
    (x_j - μ_j) / σ_j, where μ_j is the mean and σ_j is the standard
    deviation of feature j in the training set.

    Standardization ensures all features contribute equally to distance-
    based computations. Zero-variance features are left unchanged
    (σ_j is set to 1 to avoid division by zero).
    """
    def __init__(self):
        """Initialize the scaler with no precomputed statistics.

        The per-feature mean (self.mean_) and standard deviation
        (self.scale_) are set to None until fit() is called.
        """
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute the per-feature mean and standard deviation."""
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X):
        """Center each feature to zero mean and scale to unit variance."""
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit to data, then return the standardized version."""
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size=0.2, random_state=None, stratify='auto'):
    """
    Split arrays into random train and test subsets.

    When stratify='auto' (default), the split preserves the class distribution
    in both the training and test sets by allocating test_size fraction of
    each class independently. Set stratify=None for a simple random split.
    At least one sample per class is always placed in the test set.
    """
    X, y = np.asarray(X), np.asarray(y)
    if stratify == 'auto':
        stratify = y
    rng = np.random.default_rng(random_state)
    
    if stratify is not None:
        stratify = np.asarray(stratify)
        classes = np.unique(stratify)
        train_indices, test_indices = [], []
        for cls in classes:
            cls_indices = np.where(stratify == cls)[0]
            rng.shuffle(cls_indices)
            n_test_cls = max(1, int(len(cls_indices) * test_size))
            test_indices.extend(cls_indices[:n_test_cls])
            train_indices.extend(cls_indices[n_test_cls:])
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)
    else:
        indices = np.arange(len(y))
        rng.shuffle(indices)
        n_test = int(len(y) * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def resample(X, y, strategy='down', random_state=None):
    """
    Resample a binary dataset to balance class sizes.

    - 'down': randomly subsample the majority class to match the minority
      class size (under-sampling). Samples are drawn without replacement
      so each majority sample appears at most once.
    - 'up': randomly duplicate minority-class samples with replacement to
      match the majority class size (over-sampling).

    Only binary labels are supported; the data is returned unchanged if
    there are fewer or more than two classes.
    """
    X, y = np.asarray(X), np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2:
        return X, y
    min_cls = unique[np.argmin(counts)]
    maj_cls = unique[np.argmax(counts)]
    min_idx = np.where(y == min_cls)[0]
    maj_idx = np.where(y == maj_cls)[0]
    rng = np.random.default_rng(random_state)
    if strategy == 'down':
        maj_idx = rng.choice(maj_idx, size=len(min_idx), replace=False)
    elif strategy == 'up':
        min_idx = rng.choice(min_idx, size=len(maj_idx), replace=True)
    indices = np.concatenate([min_idx, maj_idx])
    rng.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for binary classification.

    Returns a 2x2 array [[TN, FP], [FN, TP]] where:
    - TN: true negatives  (actual 0, predicted 0)
    - FP: false positives (actual 0, predicted 1)
    - FN: false negatives (actual 1, predicted 0)
    - TP: true positives  (actual 1, predicted 1)
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true, y_pred):
    """
    Precision = TP / (TP + FP).

    Measures the fraction of positive predictions that are actually correct.
    Returns 0.0 when no positive predictions are made.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Recall = TP / (TP + FN).

    Measures the fraction of actual positives that are correctly identified.
    Also known as sensitivity or true positive rate.
    Returns 0.0 when there are no actual positives.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall).

    Harmonic mean of precision and recall — a balanced metric that is
    low when either precision or recall is poor.
    Returns 0.0 when both precision and recall are zero.
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc_score(y_true, y_prob):
    """
    Area Under the ROC Curve computed via the trapezoidal rule.

    The ROC curve plots the true positive rate (recall) against the false
    positive rate (FP / (FP + TN)) at various probability thresholds.
    AUC = 1.0 for a perfect classifier, 0.5 for random guessing.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Binary ground-truth labels (0 or 1).
    y_prob : ndarray of shape (n_samples,)
        Predicted probabilities for the positive class (P(y=1)).

    Returns
    -------
    auc : float
        Area under the ROC curve.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)[::-1]
    y_true = y_true[order]
    y_prob = y_prob[order]

    tpr = np.zeros(len(y_true) + 1)
    fpr = np.zeros(len(y_true) + 1)
    tp = 0
    fp = 0
    fn = np.sum(y_true == 1)
    tn = np.sum(y_true == 0)

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        tpr[i + 1] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i + 1] = fp / (fp + tn) if (fp + tn) > 0 else 0

    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2
    return auc


def classification_report(y_true, y_pred, target_names=None):
    """
    Build a text report showing precision, recall, F1, and support for
    each class, plus macro and weighted averages.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Binary ground-truth labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
    target_names : list of str, optional
        Class label names. Defaults to ['Class 0', 'Class 1'].

    Returns
    -------
    report : str
        Formatted classification report string.
    """
    if target_names is None:
        target_names = ['Class 0', 'Class 1']
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0

    support_0 = int(tn + fp)
    support_1 = int(tp + fn)

    macro_p = (precision_0 + precision_1) / 2
    macro_r = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    weighted_p = (precision_0 * support_0 + precision_1 * support_1) / n
    weighted_r = (recall_0 * support_0 + recall_1 * support_1) / n
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / n

    accuracy = (tp + tn) / n

    lines = [
        '              precision    recall  f1-score   support',
        ''
    ]
    for i, name in enumerate(target_names):
        p = precision_0 if i == 0 else precision_1
        r = recall_0 if i == 0 else recall_1
        f = f1_0 if i == 0 else f1_1
        s = support_0 if i == 0 else support_1
        lines.append(f'     {name}     {p:.4f}    {r:.4f}    {f:.4f}        {s}')
    lines.append('')
    lines.append(f'    accuracy                       {accuracy:.4f}        {n}')
    lines.append(f'   macro avg     {macro_p:.4f}    {macro_r:.4f}    {macro_f1:.4f}        {n}')
    lines.append(f'weighted avg     {weighted_p:.4f}    {weighted_r:.4f}    {weighted_f1:.4f}        {n}')
    return '\n'.join(lines)


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Visualize the decision boundary of a fitted classifier on 2D data.

    Creates a fine grid over the feature range, predicts the class label
    at every grid point using model.predict(), and renders the resulting
    regions as a filled contour map. Training points are overlaid as
    scatter markers colored by their true class label.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap='RdYlBu', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.title(title)
    plt.show()
