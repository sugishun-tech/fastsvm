"""Kernel SVC, epsilon-SVR and one-class SVM using the native SMO solver."""
from __future__ import annotations
from itertools import combinations
import warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, OutlierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data, check_array
from . import _core
from ._validation import (KINDS, positive, integer, weights, classification_cost,
                          kernel_parameters, threads, gram_checked)
from .kernels import pairwise_kernel, gram_diagnostics


class _KernelBase(BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def _prepare_kernel(self, X):
        kernel_parameters(self, X)
        self._n_train = X.shape[0]
        if self.kernel == "precomputed":
            K = gram_checked(X)
        elif callable(self.kernel):
            K = gram_checked(self.kernel(X, X), len(X))
        elif self.check_psd:
            K = pairwise_kernel(X, kernel=self.kernel, gamma=self._gamma,
                                degree=self.degree, coef0=self.coef0, n_jobs=self.n_jobs)
        else:
            if hasattr(self, "gram_diagnostics_"):
                del self.gram_diagnostics_
            return X, KINDS[self.kernel]
        if self.check_psd:
            self.gram_diagnostics_ = gram_diagnostics(K)
            if not self.gram_diagnostics_["is_psd"]:
                raise ValueError("The training Gram matrix is not positive semidefinite.")
        elif hasattr(self, "gram_diagnostics_"):
            del self.gram_diagnostics_
        return K, 5

    def _fit_problem(self, data, features, kind, mapping, signs, p, cost,
                     initial, task, y, global_indices):
        result = _core.smo(np.ascontiguousarray(data), np.ascontiguousarray(mapping, dtype=np.int64),
                          np.ascontiguousarray(signs, dtype=np.float64),
                          np.ascontiguousarray(p, dtype=np.float64),
                          np.ascontiguousarray(cost, dtype=np.float64),
                          np.ascontiguousarray(initial, dtype=np.float64),
                          kind, self._gamma, self.coef0, self.degree,
                          self.tol, self.max_iter, self.cache_size, self.shrinking, self.history)
        a = result["alpha"]
        n = len(global_indices)
        coefficients = np.bincount(mapping, weights=a * signs, minlength=n)
        support = np.flatnonzero(coefficients != 0)
        bias = result["intercept"]
        G = result["gradient"]
        raw_train = signs[:n] * (G[:n] - p[:n])
        values = raw_train + bias
        norm2 = result["norm_squared"]
        dual = float(-0.5 * norm2 - p @ a)
        if task == "classification":
            loss = np.maximum(0, 1 - y * values)
            primal = float(0.5 * norm2 + cost @ loss)
        elif task == "regression":
            loss = np.maximum(0, np.abs(values - y) - self.epsilon)
            primal = float(0.5 * norm2 + cost[:n] @ loss)
        else:
            loss = np.maximum(0, -values)
            primal = float(0.5 * norm2 + bias + cost @ loss)
        if not np.isfinite(primal) or not np.isfinite(dual):
            raise ValueError("Optimization objectives overflowed; rescale inputs or C.")
        report = {"primal_objective": primal, "dual_objective": dual,
                  "duality_gap": primal-dual,
                  "relative_duality_gap": (primal-dual)/max(1.0, abs(primal)),
                  "kkt_violation": result["kkt_violation"],
                  "equality_residual": float(abs(signs @ (a-initial))),
                  "box_violation": float(max(0., np.max(-a), np.max(a-cost))),
                  "n_iter": result["n_iter"], "converged": result["converged"],
                  "rkhs_norm_squared": norm2, "n_support": len(support),
                  "cache_hits": result["cache_hits"], "cache_misses": result["cache_misses"],
                  "cache_bytes": result["cache_bytes"],
                  "min_active_size": result["min_active_size"],
                  "shrinking_steps": result["shrinking_steps"],
                  "gram_psd_checked": bool(self.check_psd),
                  "duality_gap_requires_psd": True}
        vectors = (np.empty((0, 0)) if self.kernel == "precomputed" else
                   np.ascontiguousarray(features[support]).copy())
        return {"support": global_indices[support].copy(), "support_vectors": vectors,
                "coefficients": coefficients[support].copy(), "intercept": bias,
                "alpha": a, "bounds": np.array(cost, copy=True), "signs": np.array(signs, copy=True),
                "mapping": np.array(mapping, copy=True), "gradient": G,
                "training_indices": global_indices.copy(), "training_values": values,
                "report": report, "history": result["history"],
                "linear_coef": (coefficients @ features if self.kernel == "linear" else None)}

    def _collect(self, models):
        self._models = models
        self.optimization_results_ = [m["report"].copy() for m in models]
        self.optimization_history_ = [m["history"] for m in models]
        self.n_iter_ = np.asarray([m["report"]["n_iter"] for m in models], dtype=int)
        self.kkt_violation_ = np.asarray([m["report"]["kkt_violation"] for m in models])
        self.primal_objective_ = np.asarray([m["report"]["primal_objective"] for m in models])
        self.dual_objective_ = np.asarray([m["report"]["dual_objective"] for m in models])
        self.duality_gap_ = self.primal_objective_ - self.dual_objective_
        self.converged_ = bool(all(m["report"]["converged"] for m in models))
        self.fit_status_ = int(not self.converged_)
        if not self.converged_:
            warnings.warn("SMO reached max_iter without satisfying the KKT tolerance. "
                          "Inspect optimization_report(), rescale inputs, or increase max_iter.",
                          ConvergenceWarning, stacklevel=3)

    def optimization_report(self):
        """Per-binary-model primal/dual objectives, gap, KKT and feasibility."""
        check_is_fitted(self, "_models")
        return [dict(item) for item in self.optimization_results_]

    def dual_problem(self, model=0):
        """Return copies of dual variables, bounds, signs, map and gradient.

        Hessian Q is not materialized. For C-SVC p=-1; for epsilon-SVR
        p=(epsilon-y, epsilon+y); for one-class p=0.
        """
        check_is_fitted(self, "_models")
        integer("model", model, 0)
        if model >= len(self._models):
            raise ValueError("model index out of range.")
        m = self._models[model]
        return {key: np.array(m[key], copy=True) for key in
                ("alpha", "bounds", "signs", "mapping", "gradient", "training_indices", "training_values")}

    def _prediction_input(self, X):
        check_is_fitted(self, "_models")
        return validate_data(self, X, reset=False, dtype=np.float64, order="C")

    def _predict_model(self, X, model):
        if self.kernel == "precomputed":
            values = X[:, model["support"]] @ model["coefficients"] + model["intercept"]
        elif callable(self.kernel):
            # Evaluate only support vectors; no test-by-training Gram allocation.
            if len(model["support"]) == 0:
                return np.full(len(X), model["intercept"])
            values = pairwise_kernel(X, model["support_vectors"], kernel=self.kernel) @ model["coefficients"] + model["intercept"]
        elif self.kernel == "linear":
            values = X @ model["linear_coef"] + model["intercept"]
        else:
            values = _core.kernel_predict(X, model["support_vectors"], model["coefficients"],
                                          KINDS[self.kernel], self._gamma, self.coef0, self.degree,
                                          model["intercept"], threads(self.n_jobs))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite decision values; rescale inputs.")
        return values

    def _pair_scores(self, X):
        X = self._prediction_input(X)
        return np.column_stack([self._predict_model(X, m) for m in self._models])

    def rkhs_distance(self, X):
        """Signed hyperplane distance in RKHS, NOT Euclidean input-space distance.

        Columns correspond to binary models; binary/regression returns 1-D.
        Zero/negative norm models raise instead of producing a misleading value.
        """
        scores = self._pair_scores(X)
        norms = np.array([m["report"]["rkhs_norm_squared"] for m in self._models])
        if np.any(norms <= 0):
            raise ValueError("RKHS distance requires strictly positive squared norms and a PSD kernel.")
        result = scores / np.sqrt(norms)
        return result[:, 0] if len(self._models) == 1 else result

    def _derivative(self, X, model, hessian):
        X = self._prediction_input(X)
        if model is None:
            if len(self._models) != 1:
                raise ValueError("For multiclass models, select a binary model index explicitly.")
            model = 0
        integer("model", model, 0)
        if model >= len(self._models):
            raise ValueError("model index out of range.")
        if self.kernel == "precomputed" or callable(self.kernel):
            raise NotImplementedError("Input derivatives require a built-in feature kernel.")
        if hessian and self.kernel == "laplacian":
            raise NotImplementedError("Laplacian kernel is not everywhere differentiable.")
        m = self._models[model]
        sv, beta = m["support_vectors"], m["coefficients"]
        d, g = X.shape[1], self._gamma
        out = np.zeros((len(X), d, d) if hessian else (len(X), d))
        identity = np.eye(d) if hessian else None
        for i, x in enumerate(X):
            if self.kernel == "linear":
                if not hessian:
                    out[i] = m["linear_coef"]
                continue
            if self.kernel in ("rbf", "laplacian"):
                diff = x - sv
                if self.kernel == "rbf":
                    coeff = beta * np.exp(-g * np.einsum("ij,ij->i", diff, diff))
                    if hessian:
                        out[i] = 4*g*g*np.einsum("i,ij,ik->jk", coeff, diff, diff) - 2*g*coeff.sum()*identity
                    else:
                        out[i] = (-2*g*coeff) @ diff
                else:
                    coeff = beta * np.exp(-g*np.abs(diff).sum(axis=1))
                    out[i] = (-g*coeff) @ np.sign(diff)
            else:
                z = g*(sv @ x) + self.coef0
                if self.kernel == "poly":
                    power = self.degree
                    first = np.zeros_like(z) if power == 0 else power*g*z**(power-1)
                    second = np.zeros_like(z) if power < 2 else power*(power-1)*g*g*z**(power-2)
                else:
                    t = np.tanh(z)
                    first = g*(1-t*t)
                    second = -2*g*g*t*(1-t*t)
                if hessian:
                    out[i] = np.einsum("i,ij,ik->jk", beta*second, sv, sv)
                else:
                    out[i] = (beta*first) @ sv
        if not np.isfinite(out).all():
            raise ValueError("Derivative overflowed; rescale inputs.")
        return out

    def decision_gradient(self, X, *, model=None):
        """Analytic input gradient; OvO models use positive-second-class scores."""
        return self._derivative(X, model, False)

    def decision_hessian(self, X, *, model=None):
        """Analytic input Hessian; allocates O(n_test*d^2) explicitly."""
        return self._derivative(X, model, True)

    def _single_support(self, X, model):
        self.support_ = model["support"]
        self.support_vectors_ = model["support_vectors"]
        self.dual_coef_ = model["coefficients"][None, :]
        self.intercept_ = np.array([model["intercept"]])
        self.n_support_ = np.array([len(self.support_)], dtype=int)
        if self.kernel == "linear":
            self.coef_ = model["linear_coef"][None, :]
        elif hasattr(self, "coef_"):
            del self.coef_


class SVC(ClassifierMixin, _KernelBase):
    """Soft-margin C-SVC, binary/OvO, unregularized intercept, native SMO.

    Built-in kernels use bounded-memory LRU rows, not an unconditional n^2
    Gram matrix. Callable/precomputed/check_psd paths explicitly allocate n^2.
    probability=True uses stratified cross-validated sigmoid calibration;
    probabilities are not LIBSVM's pairwise coupling and may predict another
    class than raw-vote predict().
    """
    def __init__(self, *, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 tol=1e-3, max_iter=1000000, cache_size=200.0, class_weight=None,
                 decision_function_shape="ovr", break_ties=False, probability=False,
                 calibration_cv=5, random_state=None, n_jobs=1, history=False, check_psd=False, shrinking=True):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.probability = probability
        self.calibration_cv = calibration_cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.history = history
        self.check_psd = check_psd
        self.shrinking = shrinking

    def fit(self, X, y, sample_weight=None):
        positive("C", self.C)
        for name in ("probability", "break_ties"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean.")
        if self.probability:
            integer("calibration_cv", self.calibration_cv, 2)
            if self.kernel == "precomputed":
                raise ValueError("Internal probability calibration does not support precomputed kernels.")
        if self.decision_function_shape not in ("ovr", "ovo"):
            raise ValueError("decision_function_shape must be 'ovr' or 'ovo'.")
        if self.break_ties and self.decision_function_shape != "ovr":
            raise ValueError("break_ties requires decision_function_shape='ovr'.")
        X, y = validate_data(self, X, y, dtype=np.float64, order="C")
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("At least 2 classes are required; got 1 class.")
        cost, self.class_weight_ = classification_cost(self.C, self.class_weight, self.classes_, y, sample_weight)
        data, kind = self._prepare_kernel(X)
        pairs = list(combinations(range(n_classes), 2))
        self.class_pairs_ = [(self.classes_[i], self.classes_[j]) for i, j in pairs]

        def fit_pair(i, j):
            idx = np.flatnonzero((encoded == i) | (encoded == j))
            features = X[idx]
            subset = data[np.ix_(idx, idx)] if kind == 5 else features
            signs = np.where(encoded[idx] == j, 1., -1.)
            n = len(idx)
            m = self._fit_problem(subset, features, kind, np.arange(n, dtype=np.int64),
                signs, -np.ones(n), cost[idx], np.zeros(n), "classification", signs, idx)
            m["pair"] = (i, j)
            return m

        models = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(fit_pair)(i, j) for i, j in pairs)
        self._collect(models)
        mask = np.zeros(len(X), dtype=bool)
        for model in models:
            mask[model["support"]] = True
        self.support_ = np.concatenate([np.flatnonzero(mask & (encoded == i)) for i in range(n_classes)])
        self.n_support_ = np.array([np.count_nonzero(mask & (encoded == i)) for i in range(n_classes)])
        self.support_vectors_ = (np.empty((0, 0)) if self.kernel == "precomputed" else X[self.support_].copy())
        # Match sklearn's packed dual layout; multiclass OvO signs are reversed
        # relative to binary decision_function (positive FIRST class).
        self.dual_coef_ = np.zeros((n_classes-1, len(self.support_)))
        location = np.full(len(X), -1, dtype=int)
        location[self.support_] = np.arange(len(self.support_))
        for model in models:
            i, j = model["pair"]
            s = model["support"]
            coeff = model["coefficients"] * (1 if n_classes == 2 else -1)
            rows = np.where(encoded[s] == i, j-1, i)
            self.dual_coef_[rows, location[s]] = coeff
        direction = 1 if n_classes == 2 else -1
        self.intercept_ = direction * np.array([m["intercept"] for m in models])
        if self.kernel == "linear":
            self.coef_ = direction * np.vstack([m["linear_coef"] for m in models])
        elif hasattr(self, "coef_"):
            del self.coef_
        if hasattr(self, "_calibrator"):
            del self._calibrator
        if self.probability:
            integer("calibration_cv", self.calibration_cv, 2)
            if self.kernel == "precomputed":
                raise ValueError("Internal probability calibration does not support precomputed kernels.")
            if np.min(np.bincount(encoded)) < self.calibration_cv:
                raise ValueError("Each class needs at least calibration_cv samples.")
            estimator = clone(self).set_params(probability=False, history=False,
                                               decision_function_shape="ovr", break_ties=False, n_jobs=1)
            cv = StratifiedKFold(self.calibration_cv, shuffle=True, random_state=self.random_state)
            self._calibrator = CalibratedClassifierCV(estimator, method="sigmoid", cv=cv,
                                                       ensemble=True, n_jobs=self.n_jobs)
            self._calibrator.fit(X, y, sample_weight=sample_weight)
        return self

    def pairwise_decision_function(self, X):
        """Shape (n_samples, n_pairs); positive means SECOND class in class_pairs_."""
        return self._pair_scores(X)

    def _votes(self, pair_scores):
        votes = np.zeros((len(pair_scores), len(self.classes_)))
        confidence = np.zeros_like(votes)
        for k, m in enumerate(self._models):
            i, j = m["pair"]
            votes[:, i] += pair_scores[:, k] <= 0
            votes[:, j] += pair_scores[:, k] > 0
            confidence[:, i] -= pair_scores[:, k]
            confidence[:, j] += pair_scores[:, k]
        return votes, confidence

    def decision_function(self, X):
        scores = self._pair_scores(X)
        if len(self.classes_) == 2:
            return scores[:, 0]
        if self.decision_function_shape == "ovo":
            return -scores
        votes, confidence = self._votes(scores)
        return votes + confidence / (3*(np.abs(confidence)+1))

    def predict(self, X):
        scores = self._pair_scores(X)
        if len(self.classes_) == 2:
            return self.classes_[(scores[:, 0] > 0).astype(int)]
        votes, confidence = self._votes(scores)
        if self.break_ties:
            votes += confidence/(3*(np.abs(confidence)+1))
        return self.classes_[votes.argmax(axis=1)]

    @available_if(lambda self: self.probability)
    def predict_proba(self, X):
        X = self._prediction_input(X)
        check_is_fitted(self, "_calibrator")
        return self._calibrator.predict_proba(X)

    @available_if(lambda self: self.probability)
    def predict_log_proba(self, X):
        return np.log(np.maximum(self.predict_proba(X), np.finfo(float).tiny))


class SVR(RegressorMixin, _KernelBase):
    """Epsilon-insensitive C-SVR with unregularized intercept and native SMO."""
    def __init__(self, *, C=1.0, epsilon=0.1, kernel="rbf", degree=3, gamma="scale",
                 coef0=0.0, tol=1e-3, max_iter=1000000, cache_size=200.0,
                 n_jobs=1, history=False, check_psd=False, shrinking=True):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.n_jobs = n_jobs
        self.history = history
        self.check_psd = check_psd
        self.shrinking = shrinking

    def fit(self, X, y, sample_weight=None):
        positive("C", self.C)
        positive("epsilon", self.epsilon, allow_zero=True)
        X, y = validate_data(self, X, y, dtype=np.float64, order="C", y_numeric=True)
        data, kind = self._prepare_kernel(X)
        n = len(X)
        cost = self.C * weights(sample_weight, n)
        mapping = np.tile(np.arange(n, dtype=np.int64), 2)
        signs = np.repeat([1., -1.], n)
        p = np.r_[self.epsilon-y, self.epsilon+y]
        model = self._fit_problem(data, X, kind, mapping, signs, p,
            np.tile(cost, 2), np.zeros(2*n), "regression", y, np.arange(n))
        self._collect([model])
        self._single_support(X, model)
        return self

    def predict(self, X):
        return self._pair_scores(X)[:, 0]


class OneClassSVM(OutlierMixin, _KernelBase):
    """One-class nu-SVM with normalized dual sum(alpha)=1.

    Unlike sklearn, score magnitudes are normalized by nu*n for uniform weights.
    The mathematical separating surface is equivalent, but decision_function,
    score_samples, intercept_ and dual_coef_ are NOT on LIBSVM's raw scale.
    """
    def __init__(self, *, nu=0.5, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
                 tol=1e-3, max_iter=1000000, cache_size=200.0,
                 n_jobs=1, history=False, check_psd=False, shrinking=True):
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.n_jobs = n_jobs
        self.history = history
        self.check_psd = check_psd
        self.shrinking = shrinking

    def fit(self, X, y=None, sample_weight=None):
        positive("nu", self.nu)
        if self.nu > 1:
            raise ValueError("nu must be in (0, 1].")
        X = validate_data(self, X, dtype=np.float64, order="C")
        data, kind = self._prepare_kernel(X)
        n = len(X)
        sw = weights(sample_weight, n)
        normalized = sw / sw.max()
        normalized /= normalized.sum()
        cost = normalized/self.nu
        initial = normalized.copy()  # Feasible and permutation-equivariant.
        model = self._fit_problem(data, X, kind, np.arange(n, dtype=np.int64),
            np.ones(n), np.zeros(n), cost, initial, "oneclass", None, np.arange(n))
        self._collect([model])
        self._single_support(X, model)
        self.offset_ = -self.intercept_
        self.dual_mass_ = float(model["alpha"].sum())
        return self

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_[0]

    def score_samples(self, X):
        X = self._prediction_input(X)
        raw_model = dict(self._models[0])
        raw_model["intercept"] = 0.0
        return self._predict_model(X, raw_model)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
