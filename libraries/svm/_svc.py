"""Support Vector Machine classifier (SVC)."""
import numpy as np
from ._svm import SVM


class SVC(SVM):
    """
    Support Vector Machine Classifier (soft-margin).

    Thin alias of :class:`SVM`, which implements the classifier:
      * linear kernel → primal gradient descent on hinge loss,
      * rbf / poly    → dual Pegasos on the kernel matrix,
      * >2 classes    → deterministic one-vs-rest.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVC':
        """Fit the classifier on (X, y). See :meth:`SVM.fit`."""
        super().fit(X, y)
        return self
