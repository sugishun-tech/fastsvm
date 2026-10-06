"""Explicit feature maps for approximate large-sample kernel learning."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_is_fitted
from ._validation import positive, integer, gamma_value
from .kernels import pairwise_kernel


class RBFSampler(TransformerMixin, BaseEstimator):
    """Random Fourier features for exp(-gamma*||x-y||^2).

    Features use omega~N(0, 2 gamma I), b~U(0,2pi) and sqrt(2/m) cos(Xw+b).
    This approximates a kernel; it is not an exact SVM solver.
    """
    def __init__(self, *, gamma=1.0, n_components=512, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, accept_sparse="csr", dtype=np.float64)
        positive("gamma", self.gamma, allow_zero=True)
        integer("n_components", self.n_components)
        rng = check_random_state(self.random_state)
        self.random_weights_ = rng.normal(0, np.sqrt(2*self.gamma),
                                          (X.shape[1], self.n_components))
        self.random_offset_ = rng.uniform(0, 2*np.pi, self.n_components)
        self.n_features_out_ = self.n_components
        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = validate_data(self, X, reset=False, accept_sparse="csr", dtype=np.float64)
        out = np.asarray(X @ self.random_weights_)
        out += self.random_offset_
        np.cos(out, out=out)
        out *= np.sqrt(2/self.n_components)
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "random_weights_")
        return np.array([f"rbf{i}" for i in range(self.n_components)], dtype=object)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class Nystroem(TransformerMixin, BaseEstimator):
    """Uniform landmark Nyström map using a rank-truncated PSD eigensystem.

    Tiny eigenvalues are dropped, not inverted. Negative eigenvalues beyond
    tolerance are rejected. Effective output rank can be below n_components.
    """
    def __init__(self, *, kernel="rbf", gamma=1.0, degree=3, coef0=0.0,
                 n_components=256, eigenvalue_tol=1e-12, random_state=None, n_jobs=1):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.eigenvalue_tol = eigenvalue_tol
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64, order="C")
        integer("n_components", self.n_components)
        positive("eigenvalue_tol", self.eigenvalue_tol)
        if self.kernel == "precomputed":
            raise ValueError("Nystroem requires feature inputs, not a precomputed Gram matrix.")
        m = min(self.n_components, len(X))
        if m < self.n_components:
            warnings.warn("n_components exceeds n_samples; using all samples.", UserWarning, stacklevel=2)
        self._gamma = gamma_value(self.gamma, X)
        self.component_indices_ = check_random_state(self.random_state).permutation(len(X))[:m]
        self.components_ = X[self.component_indices_].copy()
        K = self._kernel(self.components_, self.components_)
        if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Landmark Gram matrix is not symmetric.")
        eig, vec = eigh((K + K.T)*0.5)
        threshold = self.eigenvalue_tol * max(1.0, float(np.abs(eig).max()))
        if eig[0] < -threshold:
            raise ValueError("Nyström feature maps require a PSD landmark kernel.")
        keep = eig > threshold
        if not keep.any():
            raise ValueError("The landmark Gram matrix has zero numerical rank.")
        self.eigenvalues_ = eig[keep]
        self.normalization_ = vec[:, keep] / np.sqrt(eig[keep])
        self.n_features_out_ = int(keep.sum())
        return self

    def _kernel(self, X, Y):
        return pairwise_kernel(X, Y, kernel=self.kernel, gamma=self._gamma,
                               degree=self.degree, coef0=self.coef0, n_jobs=self.n_jobs)

    def transform(self, X):
        check_is_fitted(self, "normalization_")
        X = validate_data(self, X, reset=False, dtype=np.float64, order="C")
        return self._kernel(X, self.components_) @ self.normalization_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "normalization_")
        return np.array([f"nystroem{i}" for i in range(self.n_features_out_)], dtype=object)
