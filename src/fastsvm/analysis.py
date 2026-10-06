"""Model analysis tools. Expensive diagnostics are explicit, never hidden in fit."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from ._validation import positive


def regularization_path(estimator, X, y, Cs, *, sample_weight=None, n_jobs=1):
    """Independent fits along C; returns models, times, scores and certificates.

    This is a discrete sweep, NOT an exact piecewise-linear homotopy algorithm,
    and no warm start is implied. Input order (including duplicate Cs) is kept.
    scores are TRAINING scores, not cross-validation estimates.
    """
    values = np.asarray(Cs, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Cs must be a nonempty one-dimensional sequence.")
    for value in values:
        positive("C", value)
    if "C" not in estimator.get_params(deep=False):
        raise ValueError("The estimator must expose a C hyperparameter.")

    def fit_one(C):
        model = clone(estimator).set_params(C=float(C))
        if n_jobs != 1 and "n_jobs" in model.get_params(deep=False):
            model.set_params(n_jobs=1)
        start = perf_counter()
        model.fit(X, y, sample_weight=sample_weight)
        elapsed = perf_counter()-start
        return {"C": float(C), "model": model, "fit_seconds": elapsed,
                "training_score": float(model.score(X, y)),
                "optimization": model.optimization_report()}
    return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fit_one)(C) for C in values)


def margin_summary(estimator, X, y):
    """Signed functional margins for a binary classifier; no distance claim."""
    check_is_fitted(estimator, "classes_")
    if len(estimator.classes_) != 2:
        raise ValueError("margin_summary currently requires binary classification.")
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, estimator.classes_).all():
        raise ValueError("y contains invalid labels.")
    scores = estimator.decision_function(X)
    if scores.shape != y.shape:
        raise ValueError("X and y have incompatible sample counts.")
    margins = np.where(y == estimator.classes_[1], 1., -1.) * scores
    return {"margins": margins, "minimum": float(margins.min()),
            "quantiles": np.quantile(margins, [0., .25, .5, .75, 1.]),
            "error_fraction": float(np.mean(estimator.predict(X) != y)),
            "nonpositive_margin_fraction": float(np.mean(margins <= 0)),
            "inside_margin_fraction": float(np.mean(margins < 1)),
            "hinge_loss_mean": float(np.maximum(0., 1-margins).mean())}
