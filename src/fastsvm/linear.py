"""Dense/CSR linear SVMs with native dual coordinate descent."""
from __future__ import annotations
import warnings
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data
from . import _core
from ._validation import positive, integer, jobs, weights, classification_cost, linear_arrays


class _LinearBase(BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _controls(self):
        positive("C", self.C)
        positive("tol", self.tol)
        integer("max_iter", self.max_iter)
        positive("intercept_scaling", self.intercept_scaling)
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean.")
        if not isinstance(self.shrinking, (bool, np.bool_)):
            raise ValueError("shrinking must be boolean.")
        if not isinstance(self.history, (bool, np.bool_)):
            raise ValueError("history must be boolean.")
        jobs(self.n_jobs)

    def _solve(self, arrays, d, y, cost, mode, seed):
        return _core.linear_cd(*arrays, d, np.ascontiguousarray(y, dtype=np.float64),
            cost, mode, getattr(self, "epsilon", 0.0),
            self.intercept_scaling if self.fit_intercept else 0.0,
            self.tol, self.max_iter, int(seed), self.shrinking, self.history)

    def _decision_function(self, X):
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False, accept_sparse="csr", dtype=np.float64)
        result = np.asarray(X @ np.atleast_2d(self.coef_).T) + self.intercept_
        if not np.isfinite(result).all():
            raise ValueError("Non-finite decision values; rescale X.")
        return result[:, 0] if result.shape[1] == 1 else result

    def optimization_report(self):
        check_is_fitted(self, "optimization_results_")
        return [dict(item) for item in self.optimization_results_]

    def _finish(self, results, X, targets, cost, mode):
        self.coef_ = np.vstack([r["coef"] for r in results])
        self.intercept_ = np.asarray([r["intercept"] for r in results])
        self.dual_variables_ = np.vstack([r["alpha"] for r in results])
        self.n_iter_per_model_ = np.asarray([r["n_iter"] for r in results], dtype=int)
        self.n_iter_ = int(self.n_iter_per_model_.max())
        self.converged_ = bool(all(r["converged"] for r in results))
        self.fit_status_ = int(not self.converged_)
        self.kkt_violation_ = np.array([r["kkt_violation"] for r in results])
        self.optimization_history_ = [r["history"] for r in results]
        self.optimization_results_ = []
        for result, target in zip(results, targets):
            prediction = np.asarray(X @ result["coef"]).ravel() + result["intercept"]
            a = result["alpha"]
            norm2 = float(result["coef"] @ result["coef"] + result["bias_weight"] ** 2)
            loss = (np.maximum(0, 1 - target * prediction) if mode < 2 else
                    np.maximum(0, np.abs(target - prediction) - self.epsilon))
            if mode % 2:
                loss = loss ** 2
            primal = 0.5 * norm2 + float(cost @ loss)
            dual = (float(a.sum()) if mode < 2 else
                    float(target @ a - self.epsilon * np.abs(a).sum())) - 0.5 * norm2
            if mode % 2:
                nz = cost > 0
                dual -= float(np.sum(a[nz] ** 2 / (4 * cost[nz])))
            if not np.isfinite(primal) or not np.isfinite(dual):
                raise ValueError("Optimization objectives overflowed; rescale inputs or C.")
            self.optimization_results_.append({
                "primal_objective": primal, "dual_objective": dual,
                "duality_gap": primal - dual, "relative_duality_gap": (primal-dual)/max(1.0, abs(primal)),
                "kkt_violation": result["kkt_violation"], "converged": result["converged"],
                "n_iter": result["n_iter"], "weight_norm_squared": norm2,
                "intercept_regularized": self.fit_intercept,
            })
        self.primal_objective_ = np.array([r["primal_objective"] for r in self.optimization_results_])
        self.dual_objective_ = np.array([r["dual_objective"] for r in self.optimization_results_])
        self.duality_gap_ = self.primal_objective_ - self.dual_objective_
        if not self.converged_:
            warnings.warn("Dual coordinate descent reached max_iter before the KKT tolerance. "
                          "Standardize X, increase max_iter, or relax tol.", ConvergenceWarning, stacklevel=3)
        return self


class LinearSVC(ClassifierMixin, _LinearBase):
    """L2-regularized linear SVC, hinge or squared-hinge loss, binary/OvR.

    C multiplies a SUM of losses, not a mean. A fitted intercept is regularized
    as a synthetic feature with value intercept_scaling. Only the dual solver
    is provided; this is not a full replacement for sklearn.svm.LinearSVC.
    """
    def __init__(self, *, C=1.0, loss="squared_hinge", tol=1e-4, max_iter=10000,
                 fit_intercept=True, intercept_scaling=1.0, class_weight=None,
                 random_state=None, shrinking=True, n_jobs=1, history=False):
        self.C = C
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.shrinking = shrinking
        self.n_jobs = n_jobs
        self.history = history

    def fit(self, X, y, sample_weight=None):
        self._controls()
        if self.loss not in ("hinge", "squared_hinge"):
            raise ValueError("loss must be 'hinge' or 'squared_hinge'.")
        X, y = validate_data(self, X, y, accept_sparse="csr", dtype=np.float64, order="C")
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("At least 2 classes are required; got 1 class.")
        cost, self.class_weight_ = classification_cost(self.C, self.class_weight, self.classes_, y, sample_weight, weighted_balanced=True)
        X, arrays = linear_arrays(X)
        labels = [1] if len(self.classes_) == 2 else range(len(self.classes_))
        targets = [np.where(encoded == label, 1.0, -1.0) for label in labels]
        seeds = check_random_state(self.random_state).randint(0, 2**31-1, len(targets))
        mode = int(self.loss == "squared_hinge")
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._solve)(arrays, X.shape[1], target, cost, mode, seed)
            for target, seed in zip(targets, seeds))
        return self._finish(results, X, targets, cost, mode)

    def decision_function(self, X):
        return self._decision_function(X)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int) if scores.ndim == 1 else scores.argmax(axis=1)]


class LinearSVR(RegressorMixin, _LinearBase):
    """L2-regularized epsilon-insensitive linear SVR, dense or CSR."""
    def __init__(self, *, C=1.0, epsilon=0.1, loss="epsilon_insensitive", tol=1e-4,
                 max_iter=10000, fit_intercept=True, intercept_scaling=1.0,
                 random_state=None, shrinking=False, n_jobs=1, history=False):
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.shrinking = shrinking
        self.n_jobs = n_jobs
        self.history = history

    def fit(self, X, y, sample_weight=None):
        self._controls()
        positive("epsilon", self.epsilon, allow_zero=True)
        if self.loss not in ("epsilon_insensitive", "squared_epsilon_insensitive"):
            raise ValueError("Unknown SVR loss.")
        X, y = validate_data(self, X, y, accept_sparse="csr", dtype=np.float64,
                             order="C", y_numeric=True)
        if not np.isfinite(y).all():
            raise ValueError("y must contain finite values.")
        X, arrays = linear_arrays(X)
        cost = np.ascontiguousarray(self.C * weights(sample_weight, len(y)))
        mode = 3 if self.loss == "squared_epsilon_insensitive" else 2
        seed = check_random_state(self.random_state).randint(0, 2**31-1)
        result = self._solve(arrays, X.shape[1], y, cost, mode, seed)
        self._finish([result], X, [y], cost, mode)
        self.coef_ = self.coef_[0]
        return self

    def predict(self, X):
        return self._decision_function(X)
