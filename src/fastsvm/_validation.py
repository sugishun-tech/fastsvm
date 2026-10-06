"""Shared input validation; Python owns safety, Cython owns hot loops."""
from __future__ import annotations

import numbers
import threading
import warnings
import numpy as np
from scipy import sparse
from joblib import effective_n_jobs
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_array

KINDS = {"linear": 0, "rbf": 1, "poly": 2, "sigmoid": 3, "laplacian": 4, "precomputed": 5}


def positive(name, value, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number.")
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite.")
    return float(value)


def integer(name, value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def jobs(n_jobs):
    if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral) or n_jobs == 0):
        raise ValueError("n_jobs must be None or a nonzero integer.")
    return effective_n_jobs(n_jobs)


def threads(n_jobs):
    count = jobs(n_jobs)
    # Cython's parallel constructs are used only from the main Python thread.
    return count if threading.current_thread() is threading.main_thread() else 1


def weights(sample_weight, n):
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(n, float(w))
    if w.ndim != 1 or w.shape != (n,):
        raise ValueError(f"sample_weight must be scalar or have shape ({n},).")
    if not np.isfinite(w).all() or (w < 0).any() or not (w > 0).any():
        raise ValueError("sample_weight must be finite, nonnegative, and not all zero.")
    return np.ascontiguousarray(w)


def classification_cost(C, class_weight, classes, y, sample_weight, *, weighted_balanced=False):
    sw = weights(sample_weight, len(y))
    if class_weight is None:
        cw = np.ones(len(classes))
    elif isinstance(class_weight, str) and class_weight == "balanced":
        totals = np.bincount(np.searchsorted(classes, y),
                             weights=sw if weighted_balanced else None, minlength=len(classes))
        if np.any(totals <= 0):
            raise ValueError("Each class needs positive weight; at least 2 classes are required.")
        cw = (sw.sum() if weighted_balanced else len(y)) / (len(classes) * totals)
    elif isinstance(class_weight, dict):
        cw = compute_class_weight(class_weight, classes=classes, y=y)
    else:
        raise ValueError("class_weight must be None, 'balanced', or a dictionary.")
    if not np.isfinite(cw).all() or (cw < 0).any():
        raise ValueError("class weights must be finite and nonnegative.")
    cost = C * sw * cw[np.searchsorted(classes, y)]
    if not np.isfinite(cost).all():
        raise ValueError("C * sample_weight * class_weight overflowed.")
    for label in classes:
        if not np.any(cost[y == label] > 0):
            raise ValueError("Each class must have positive total sample weight; at least 2 classes are required.")
    return np.ascontiguousarray(cost), cw


def gamma_value(gamma, X):
    if isinstance(gamma, str):
        if gamma == "auto":
            return 1.0 / X.shape[1]
        if gamma == "scale":
            with np.errstate(over="ignore", invalid="ignore"):
                variance = float(X.var())
            if not np.isfinite(variance):
                raise ValueError("Variance overflowed; rescale X.")
            return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        raise ValueError("gamma must be 'scale', 'auto', or a nonnegative number.")
    return positive("gamma", gamma, allow_zero=True)


def canonical_csr(X):
    X = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    X.sum_duplicates()
    X.sort_indices()
    # A single 64-bit index ABI avoids size-dependent integer overflow.
    X.indices = X.indices.astype(np.int64, copy=False)
    X.indptr = X.indptr.astype(np.int64, copy=False)
    return X


def linear_arrays(X):
    if sparse.issparse(X):
        X = canonical_csr(X)
        return X, (np.empty((0, 0), dtype=np.float64), X.data, X.indices, X.indptr)
    X = np.ascontiguousarray(X, dtype=np.float64)
    return X, (X, np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))


def gram_checked(K, n=None):
    K = check_array(K, dtype=np.float64, order="C", ensure_min_samples=1)
    if K.shape[0] != K.shape[1] or (n is not None and K.shape != (n, n)):
        raise ValueError("Training Gram matrix must be square with one row per sample.")
    if not np.allclose(K, K.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Training Gram matrix must be symmetric.")
    return K


def kernel_parameters(estimator, X):
    if not callable(estimator.kernel) and (not isinstance(estimator.kernel, str) or estimator.kernel not in KINDS):
        raise ValueError(f"kernel must be one of {tuple(KINDS)} or a callable.")
    integer("degree", estimator.degree, 0)
    if not isinstance(estimator.coef0, numbers.Real) or not np.isfinite(estimator.coef0):
        raise ValueError("coef0 must be a finite real number.")
    positive("cache_size", estimator.cache_size)
    positive("tol", estimator.tol)
    integer("max_iter", estimator.max_iter)
    jobs(estimator.n_jobs)
    for name in ("shrinking", "history", "check_psd"):
        if not isinstance(getattr(estimator, name), (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean.")
    estimator._gamma = gamma_value(estimator.gamma, X)
    if estimator.kernel == "sigmoid":
        warnings.warn("Sigmoid kernels are not PSD for arbitrary parameters; global optimality is not guaranteed.", UserWarning, stacklevel=3)
    if estimator.kernel == "poly" and estimator.coef0 < 0:
        warnings.warn("Negative coef0 can make the polynomial kernel indefinite.", UserWarning, stacklevel=3)
