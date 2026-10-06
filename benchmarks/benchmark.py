"""Reproducible end-to-end timing, accuracy and optimization evidence.

Run from the repository root after installation. Dataset generation is excluded;
input validation, data copying, solver and native diagnostics are INCLUDED.
No claim of a universal speedup can be inferred from these synthetic workloads.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn import svm
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.base import clone
from threadpoolctl import threadpool_limits, threadpool_info
import fastsvm


def cases():
    rng = np.random.default_rng(20261006)
    X, y = make_classification(n_samples=4000, n_features=32, n_informative=20,
                               n_redundant=0, class_sep=1.1, random_state=17)
    X, Xt, y, yt = train_test_split(X, y, test_size=.25, random_state=19)
    scale = StandardScaler().fit(X)
    X, Xt = scale.transform(X), scale.transform(Xt)
    common = dict(C=.1, tol=1e-4, max_iter=100000, random_state=0)
    yield 'linear_dense', X, Xt, y, yt, fastsvm.LinearSVC(**common), svm.LinearSVC(dual=True, **common)

    Xs = sparse.random(6000, 1200, density=.012, random_state=27, format='csr',
                       data_rvs=lambda n: rng.normal(size=n))
    Xs = normalize(Xs, copy=False)
    w = rng.normal(size=Xs.shape[1])
    ys = (np.asarray(Xs@w).ravel()+.05*rng.normal(size=len(Xs.indptr)-1)>0).astype(int)
    Xs, Xst, ys, yst = train_test_split(Xs, ys, test_size=.25, random_state=19)
    kw = dict(C=1., tol=1e-4, max_iter=100000, random_state=0)
    yield 'linear_csr', Xs, Xst, ys, yst, fastsvm.LinearSVC(**kw), svm.LinearSVC(dual=True, **kw)

    Xr, yr = make_classification(n_samples=2400, n_features=16, n_informative=10,
                                 n_redundant=0, class_sep=1.3, random_state=23)
    Xr, Xrt, yr, yrt = train_test_split(Xr, yr, test_size=.25, random_state=19)
    scale = StandardScaler().fit(Xr)
    Xr, Xrt = scale.transform(Xr), scale.transform(Xrt)
    kw = dict(C=2., gamma=1/16, tol=1e-4, cache_size=64)
    yield 'rbf_binary', Xr, Xrt, yr, yrt, fastsvm.SVC(**kw), svm.SVC(**kw)

    Xm, ym = make_blobs(n_samples=2000, centers=5, n_features=12, cluster_std=4., random_state=29)
    Xm, Xmt, ym, ymt = train_test_split(Xm, ym, test_size=.25, random_state=19)
    scale = StandardScaler().fit(Xm)
    Xm, Xmt = scale.transform(Xm), scale.transform(Xmt)
    kw = dict(C=2., gamma=1/12, tol=1e-4, cache_size=64)
    yield 'rbf_multiclass', Xm, Xmt, ym, ymt, fastsvm.SVC(**kw), svm.SVC(**kw)

    reg = np.sin(Xr[:, 0])+.5*Xr[:, 1]+.05*rng.normal(size=len(Xr))
    reg_test = np.sin(Xrt[:, 0])+.5*Xrt[:, 1]+.05*rng.normal(size=len(Xrt))
    kw = dict(C=2., gamma=1/16, epsilon=.1, tol=1e-4, cache_size=64)
    yield 'rbf_regression', Xr, Xrt, reg, reg_test, fastsvm.SVR(**kw), svm.SVR(**kw)

    reg = .7*X[:, 0]-.3*X[:, 1]+.1*rng.normal(size=len(X))
    reg_test = .7*Xt[:, 0]-.3*Xt[:, 1]+.1*rng.normal(size=len(Xt))
    kw = dict(C=.1, epsilon=.1, loss='squared_epsilon_insensitive', tol=1e-4, max_iter=100000, random_state=0)
    yield 'linear_regression', X, Xt, reg, reg_test, fastsvm.LinearSVR(**kw), svm.LinearSVR(dual=True, **kw)


def measure_pair(estimators, X, Xt, y, yt, repeat):
    fit_times, predict_times = [[], []], [[], []]
    messages, models, predictions = [set(), set()], [None, None], [None, None]
    for estimator in estimators:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clone(estimator).fit(X, y).predict(Xt)
    for iteration in range(repeat):
        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = clone(estimators[index])
                start = time.perf_counter()
                model.fit(X, y)
                fit_times[index].append(time.perf_counter()-start)
                start = time.perf_counter()
                prediction = model.predict(Xt)
                predict_times[index].append(time.perf_counter()-start)
            messages[index].update(str(w.message) for w in caught)
            models[index], predictions[index] = model, prediction
    results = []
    for index, model in enumerate(models):
        results.append({"fit_seconds": fit_times[index], "predict_seconds": predict_times[index],
            "fit_median": float(np.median(fit_times[index])),
            "predict_median": float(np.median(predict_times[index])),
            "fit_iqr": np.quantile(fit_times[index], [.25, .75]).tolist(),
            "predict_iqr": np.quantile(predict_times[index], [.25, .75]).tolist(),
            "score": float(model.score(Xt, yt)), "n_iter": np.asarray(model.n_iter_).tolist(),
            "warnings": sorted(messages[index]), "params": model.get_params(),
            "optimization": model.optimization_report() if hasattr(model, 'optimization_report') else None,
            "converged": bool(model.converged_) if hasattr(model, 'converged_') else not messages[index]})
    return results, models, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results.json'))
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    cpu = 'unknown'
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), cpu)
    report = {"environment": {"python": sys.version, "platform": platform.platform(), "cpu": cpu,
        "logical_cpus": os.cpu_count(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "fastsvm": fastsvm.__version__, "build": fastsvm.build_info(),
        "build_flags_recorded_by_runner": {key: os.environ.get(key) for key in
            ['FASTSVM_NATIVE', 'FASTSVM_OPENMP', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
        "threads": 1, "seed": 20261006, "repeat": args.repeat, "timing_order": "alternating pairs",
        "threadpools": threadpool_info()}, "cases": []}
    with threadpool_limits(limits=1):
        for name, X, Xt, y, yt, ours, reference in cases():
            (native, other), (native_model, other_model), (p, q) = measure_pair(
                [ours, reference], X, Xt, y, yt, args.repeat)
            decision_native = native_model.decision_function(Xt) if hasattr(native_model, 'classes_') else p
            decision_ref = other_model.decision_function(Xt) if hasattr(other_model, 'classes_') else q
            item = {"case": name, "train_shape": list(X.shape), "test_shape": list(Xt.shape),
                "sparse": sparse.issparse(X), "native": native, "sklearn": other,
                "fit_speedup": other['fit_median']/native['fit_median'],
                "predict_speedup": other['predict_median']/native['predict_median'],
                "prediction_agreement": float(np.mean(p==q)) if hasattr(native_model, 'classes_') else None,
                "decision_max_abs_difference": float(np.max(np.abs(decision_native-decision_ref)))}
            report['cases'].append(item)
            print(f"{name:20s} fit {native['fit_median']:.6f}s / {other['fit_median']:.6f}s "
                  f"speedup={item['fit_speedup']:.2f}x predict={item['predict_speedup']:.2f}x "
                  f"scores={native['score']:.6f}/{other['score']:.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
