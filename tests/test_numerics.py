import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.optimize import minimize
from sklearn import svm
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from fastsvm import SVC, SVR, OneClassSVM, LinearSVC, LinearSVR, pairwise_kernel


@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=6, n_informative=4, random_state=17)
    return StandardScaler().fit_transform(X), y


def certificate(model, tol=2e-5):
    assert model.converged_
    for report in model.optimization_report():
        assert report['kkt_violation'] <= model.tol * 1.0001
        assert report['duality_gap'] >= -1e-7
        assert report['relative_duality_gap'] < tol
        assert report.get('equality_residual', 0) < 1e-8
        assert report.get('box_violation', 0) < 1e-12


@pytest.mark.parametrize('kernel,params', [
    ('rbf', {'gamma': .3}), ('linear', {}), ('poly', {'gamma': .15, 'degree': 3, 'coef0': 1}),
])
@pytest.mark.parametrize('weighted', [False, True])
def test_svc_reference(data, kernel, params, weighted):
    X, y = data
    weight = np.linspace(.1, 2, len(y)) if weighted else None
    cw = {0: .7, 1: 1.3} if weighted else None
    model = SVC(kernel=kernel, C=1.7, tol=1e-7, class_weight=cw, **params).fit(X, y, weight)
    reference = svm.SVC(kernel=kernel, C=1.7, tol=1e-9, class_weight=cw, **params).fit(X, y, weight)
    assert_allclose(model.decision_function(X), reference.decision_function(X), atol=5e-5)
    assert_array_equal(model.predict(X), reference.predict(X))
    certificate(model)


@pytest.mark.parametrize('loss', ['hinge', 'squared_hinge'])
@pytest.mark.parametrize('csr', [False, True])
@pytest.mark.parametrize('intercept', [False, True])
def test_linear_classifier(data, loss, csr, intercept):
    X, y = data
    X = sparse.csr_matrix(X) if csr else X
    sw = np.linspace(.1, 1.5, len(y))
    params = dict(C=.6, loss=loss, fit_intercept=intercept, intercept_scaling=1.4, tol=1e-7, random_state=2, max_iter=30000)
    model = LinearSVC(**params).fit(X, y, sw)
    reference = svm.LinearSVC(dual=True, **params).fit(X, y, sw)
    assert_allclose(model.decision_function(X), reference.decision_function(X), atol=3e-5)
    certificate(model)


@pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly'])
def test_svr(data, kernel):
    X, _ = data
    y = np.sin(X[:, 0]) + .1*X[:, 1]
    sw = np.linspace(.3, 1.3, len(y))
    kw = dict(kernel=kernel, C=1.3, epsilon=.08, gamma=.2, coef0=.3, tol=1e-7)
    model = SVR(**kw).fit(X, y, sw)
    reference = svm.SVR(**kw).fit(X, y, sw)
    assert_allclose(model.predict(X), reference.predict(X), atol=2e-5)
    certificate(model)


@pytest.mark.parametrize('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive'])
@pytest.mark.parametrize('csr', [False, True])
def test_linear_regressor(data, loss, csr):
    X, _ = data
    y = np.sin(X[:, 0]) + .1*X[:, 1]
    X = sparse.csr_matrix(X) if csr else X
    sw = np.linspace(.2, 1.5, len(y))
    kw = dict(C=.7, loss=loss, epsilon=.1, tol=1e-7, max_iter=100000, random_state=9)
    model = LinearSVR(**kw).fit(X, y, sw)
    reference = svm.LinearSVR(dual=True, **kw).fit(X, y, sw)
    assert model.coef_.ndim == 1
    assert_allclose(model.predict(X), reference.predict(X), atol=3e-5)
    certificate(model)


@pytest.mark.parametrize('nu', [.1, .5, .9])
def test_oneclass(data, nu):
    X, _ = data
    model = OneClassSVM(nu=nu, gamma=.3, tol=1e-8).fit(X)
    reference = svm.OneClassSVM(nu=nu, gamma=.3, tol=1e-9).fit(X)
    assert_allclose(model.decision_function(X), reference.decision_function(X)/(nu*len(X)), atol=1e-7)
    assert_allclose(model.score_samples(X)-model.offset_[0], model.decision_function(X))
    assert_allclose(model.dual_mass_, 1, atol=1e-10)
    certificate(model)


@pytest.mark.parametrize('kernel', ['rbf', 'linear'])
@pytest.mark.parametrize('n_classes', [3, 4])
def test_multiclass(kernel, n_classes):
    X, y = make_blobs(n_samples=140, n_features=4, centers=n_classes, cluster_std=2, random_state=8)
    X = StandardScaler().fit_transform(X)
    model = SVC(kernel=kernel, tol=1e-7, decision_function_shape='ovo', n_jobs=2).fit(X, y)
    reference = svm.SVC(kernel=kernel, tol=1e-9, decision_function_shape='ovo').fit(X, y)
    assert_array_equal(model.predict(X), reference.predict(X))
    assert_allclose(model.decision_function(X), reference.decision_function(X), atol=1e-4)
    K = pairwise_kernel(X, model.support_vectors_, kernel=kernel, gamma=model._gamma)
    start = np.r_[0, np.cumsum(model.n_support_)]
    k = 0
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            reconstructed = (K[:, start[i]:start[i+1]] @ model.dual_coef_[j-1, start[i]:start[i+1]] +
                             K[:, start[j]:start[j+1]] @ model.dual_coef_[i, start[j]:start[j+1]] + model.intercept_[k])
            assert_allclose(reconstructed, model.decision_function(X)[:, k], atol=1e-10)
            k += 1
    certificate(model)


def test_independent_qp_oracle():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(18, 3))
    y = np.r_[-np.ones(9), np.ones(9)]
    K = pairwise_kernel(X, gamma=.4)
    Q = K*y[:, None]*y[None, :]
    C = .8
    oracle = minimize(lambda a: .5*a@Q@a-a.sum(), np.zeros(len(y)),
        jac=lambda a: Q@a-1, bounds=[(0, C)]*len(y), method='SLSQP',
        constraints={'type': 'eq', 'fun': lambda a: y@a, 'jac': lambda a: y},
        options={'ftol': 1e-12, 'maxiter': 2000})
    assert oracle.success
    model = SVC(kernel='precomputed', C=C, tol=1e-9).fit(K, y)
    assert_allclose(model.dual_objective_[0], -oracle.fun, atol=1e-8)
    certificate(model)


@pytest.mark.parametrize('estimator', [SVC(tol=1e-7), SVR(tol=1e-7), OneClassSVM(tol=1e-7)])
def test_tiny_cache_equivalent(data, estimator):
    from sklearn.base import clone
    X, y = data
    a = clone(estimator).set_params(cache_size=.0001).fit(X, y)
    b = clone(estimator).set_params(cache_size=32).fit(X, y)
    assert_allclose(a._pair_scores(X), b._pair_scores(X), atol=1e-10)
    assert a.optimization_report()[0]['cache_bytes'] == 2*len(X)*8
    assert a.optimization_report()[0]['cache_misses'] > 0


def test_linear_multiclass(data):
    X, y = make_blobs(n_samples=90, centers=4, n_features=5, random_state=12)
    X = StandardScaler().fit_transform(X)
    model = LinearSVC(tol=1e-7, random_state=4, n_jobs=2).fit(X, y)
    ref = svm.LinearSVC(dual=True, tol=1e-8, max_iter=30000, random_state=4).fit(X, y)
    assert_allclose(model.decision_function(X), ref.decision_function(X), atol=1e-5)
    certificate(model)
