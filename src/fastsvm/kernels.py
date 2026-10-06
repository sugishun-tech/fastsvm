"""Composable kernels and explicit, opt-in spectral diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from . import _core
from ._validation import KINDS, gamma_value, integer, positive, threads, gram_checked


def pairwise_kernel(X, Y=None, *, kernel="rbf", gamma="scale", degree=3,
                    coef0=0.0, n_jobs=1):
    """Return a dense Gram/cross-Gram matrix. Allocation is O(n*m)."""
    X = check_array(X, dtype=np.float64, order="C")
    Y = X if Y is None else check_array(Y, dtype=np.float64, order="C")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y have different feature counts.")
    if callable(kernel):
        result = check_array(kernel(X, Y), dtype=np.float64, order="C")
        if result.shape != (len(X), len(Y)):
            raise ValueError("Callable kernel must return shape (len(X), len(Y)).")
        return result
    if kernel not in KINDS or kernel == "precomputed":
        raise ValueError("Select a feature-space kernel; precomputed is estimator-only.")
    integer("degree", degree, 0)
    if not np.isfinite(coef0):
        raise ValueError("coef0 must be finite.")
    return _core.kernel_matrix(X, Y, KINDS[kernel], gamma_value(gamma, X),
                               float(coef0), degree, threads(n_jobs))


@dataclass(frozen=True)
class Kernel:
    """Callable kernel expression supporting sums, products and nonnegative scales.

    Use numeric gamma for reproducible cross-Gram semantics. PSD closure holds
    only when the component kernels themselves are PSD.
    """
    name: str = "rbf"
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 0.0
    operation: str | None = None
    left: object = None
    right: object = None

    def __call__(self, X, Y):
        if self.operation == "sum":
            return self.left(X, Y) + self.right(X, Y)
        if self.operation == "product":
            return self.left(X, Y) * self.right(X, Y)
        if self.operation == "scale":
            return self.left * self.right(X, Y)
        return pairwise_kernel(X, Y, kernel=self.name, gamma=self.gamma,
                               degree=self.degree, coef0=self.coef0)

    def __add__(self, other):
        if not callable(other):
            return NotImplemented
        return Kernel(operation="sum", left=self, right=other)

    def __mul__(self, other):
        if callable(other):
            return Kernel(operation="product", left=self, right=other)
        scale = positive("kernel scale", other, allow_zero=True)
        return Kernel(operation="scale", left=scale, right=self)

    def __rmul__(self, other):
        return self * other


def gram_diagnostics(K, *, tolerance=1e-10):
    """Symmetry, spectrum, numerical rank, PSD and condition estimate; O(n^3)."""
    positive("tolerance", tolerance)
    K = check_array(K, dtype=np.float64)
    if K.shape[0] != K.shape[1]:
        raise ValueError("Gram matrix must be square.")
    symmetry_error = float(np.max(np.abs(K - K.T)))
    eigenvalues = eigh((K + K.T) * 0.5, eigvals_only=True)
    threshold = tolerance * max(1.0, float(np.max(np.abs(eigenvalues))))
    positive_values = eigenvalues[eigenvalues > threshold]
    rank = int(np.count_nonzero(np.abs(eigenvalues) > threshold))
    return {
        "is_symmetric": symmetry_error <= threshold,
        "symmetry_error": symmetry_error,
        "is_psd": symmetry_error <= threshold and eigenvalues[0] >= -threshold,
        "min_eigenvalue": float(eigenvalues[0]), "max_eigenvalue": float(eigenvalues[-1]),
        "eigenvalues": eigenvalues, "rank": rank, "threshold": threshold,
        "condition_number": (float(eigenvalues[-1] / eigenvalues[0])
                             if rank == len(K) and eigenvalues[0] > threshold else np.inf),
        "positive_spectrum_condition": (float(positive_values[-1] / positive_values[0])
                                        if positive_values.size else np.inf),
    }


def project_psd(K, *, floor=0.0):
    """Project the symmetric part onto eigenvalues >= floor in Frobenius norm.

    This changes the training kernel. It does NOT define out-of-sample kernels.
    """
    positive("floor", floor, allow_zero=True)
    K = check_array(K, dtype=np.float64)
    if K.shape[0] != K.shape[1]:
        raise ValueError("Gram matrix must be square.")
    eig, vec = eigh((K + K.T) * 0.5)
    return (vec * np.maximum(eig, floor)) @ vec.T


def kernel_alignment(K, L, *, centered=True):
    """Normalized Frobenius alignment; zero matrices return 0."""
    K, L = gram_checked(K), gram_checked(L)
    if K.shape != L.shape:
        raise ValueError("Kernel shapes must agree.")
    if centered:
        K = K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()
        L = L - L.mean(0)[None, :] - L.mean(1)[:, None] + L.mean()
    denom = np.linalg.norm(K) * np.linalg.norm(L)
    return float(np.sum(K * L) / denom) if denom else 0.0


class KernelCenterer(TransformerMixin, BaseEstimator):
    """Center training and rectangular test Gram matrices using TRAIN statistics."""
    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        X = gram_checked(X)
        self.K_fit_rows_ = X.mean(axis=0)
        self.K_fit_all_ = float(X.mean())
        return self

    def transform(self, X):
        check_is_fitted(self, "K_fit_rows_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X - self.K_fit_rows_[None, :] - X.mean(axis=1)[:, None] + self.K_fit_all_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = True
        return tags
