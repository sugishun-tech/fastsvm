import pickle
from contextlib import nullcontext
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, KernelCenterer as RefCenterer
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, laplacian_kernel, sigmoid_kernel
from fastsvm import (SVC, SVR, OneClassSVM, LinearSVC, LinearSVR, Kernel, pairwise_kernel,
                     KernelCenterer, gram_diagnostics, project_psd, kernel_alignment,
                     RBFSampler, Nystroem, regularization_path, margin_summary)


@pytest.fixture
def data():
    X, y = make_classification(n_samples=80, n_features=4, n_redundant=0, random_state=4)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize('name,fun', [('rbf', rbf_kernel), ('poly', polynomial_kernel),
                                     ('laplacian', laplacian_kernel), ('sigmoid', sigmoid_kernel)])
def test_kernels(data, name, fun):
    X, _ = data
    kw = dict(gamma=.4)
    if name == 'poly':
        kw.update(degree=3, coef0=.2)
    if name == 'sigmoid':
        kw.update(coef0=.2)
    assert_allclose(pairwise_kernel(X, X[:7], kernel=name, n_jobs=2, **kw), fun(X, X[:7], **kw), atol=1e-12)


@pytest.mark.parametrize('Estimator', [SVC, SVR, OneClassSVM])
def test_precomputed_callable(data, Estimator):
    X, y = data
    m = Estimator(kernel='rbf', gamma=.2, tol=1e-7).fit(X, y)
    p = Estimator(kernel='precomputed', tol=1e-7).fit(rbf_kernel(X, gamma=.2), y)
    c = Estimator(kernel=Kernel(gamma=.2), tol=1e-7).fit(X, y)
    test = X[:15]*.9
    assert_allclose(m._pair_scores(test), p._pair_scores(rbf_kernel(test, X, gamma=.2)), atol=1e-6)
    assert_allclose(m._pair_scores(test), c._pair_scores(test), atol=1e-6)


@pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
def test_derivatives(data, kernel):
    X, y = data
    # SVR also exercises the identical derivative implementation for all tasks.
    expected_warning = pytest.warns(UserWarning) if kernel == "sigmoid" else nullcontext()
    with expected_warning:
        m = SVR(kernel=kernel, gamma=.15, coef0=.4, degree=3, tol=1e-7, max_iter=100000).fit(X, y)
    x = X[:2]+.027
    step = 1e-5
    grad = m.decision_gradient(x)
    hess = m.decision_hessian(x)
    for j in range(X.shape[1]):
        delta = np.zeros_like(x)
        delta[:, j] = step
        numeric = (m.predict(x+delta)-m.predict(x-delta))/(2*step)
        assert_allclose(grad[:, j], numeric, atol=1e-6, rtol=1e-5)
        numeric_h = (m.decision_gradient(x+delta)-m.decision_gradient(x-delta))/(2*step)
        assert_allclose(hess[:, :, j], numeric_h, atol=2e-6, rtol=1e-5)
    assert_allclose(hess, hess.transpose(0, 2, 1), atol=1e-12)


def test_laplacian_gradient(data):
    X, y = data
    m = SVC(kernel='laplacian', gamma=.2, tol=1e-7).fit(X, y)
    x = X[:2]+.039
    for j in range(X.shape[1]):
        delta = np.zeros_like(x)
        delta[:, j] = 1e-6
        numeric = (m.decision_function(x+delta)-m.decision_function(x-delta))/2e-6
        assert_allclose(m.decision_gradient(x)[:, j], numeric, atol=1e-7)
    with pytest.raises(NotImplementedError):
        m.decision_hessian(x)


def test_spectral_and_centering(data):
    X, y = data
    K = rbf_kernel(X, gamma=.2)
    report = gram_diagnostics(K)
    assert report['is_psd'] and report['rank'] == len(K)
    bad = K-2*np.eye(len(K))
    assert not gram_diagnostics(bad)['is_psd']
    assert gram_diagnostics(project_psd(bad))['is_psd']
    assert_allclose(kernel_alignment(K, K), 1)
    ours = KernelCenterer().fit(K)
    ref = RefCenterer().fit(K)
    assert_allclose(ours.transform(K[:9]), ref.transform(K[:9]), atol=1e-12)
    with pytest.raises(ValueError, match='positive semidefinite'):
        SVC(kernel='precomputed', check_psd=True).fit(bad, y)
    assert SVC(check_psd=True).fit(X, y).gram_diagnostics_['is_psd']


def test_algebra(data):
    X, _ = data
    a, b = Kernel('rbf', gamma=.2), Kernel('linear')
    expr = 2*a + a*b
    assert_allclose(expr(X, X), 2*a(X, X)+a(X, X)*b(X, X))
    assert gram_diagnostics(expr(X, X))['is_psd']
    with pytest.raises(ValueError):
        _ = -1*a


def test_rff(data):
    X, _ = data
    m = RBFSampler(gamma=.2, n_components=6000, random_state=9).fit(X)
    Z = m.transform(X)
    assert_allclose(Z@Z.T, rbf_kernel(X, gamma=.2), atol=.07)
    assert_allclose(m.transform(sparse.csr_matrix(X)), Z)
    assert_array_equal(m.transform(X), pickle.loads(pickle.dumps(m)).transform(X))


def test_nystrom(data):
    X, _ = data
    m = Nystroem(gamma=.3, n_components=len(X), random_state=4).fit(X)
    Z = m.transform(X)
    assert_allclose(Z@Z.T, rbf_kernel(X, gamma=.3), atol=1e-8)
    duplicate = np.vstack([X[:5], X[:5]])
    reduced = Nystroem(n_components=10, random_state=3).fit(duplicate)
    assert reduced.n_features_out_ == 5


@pytest.mark.parametrize('Estimator', [SVC, SVR, OneClassSVM, LinearSVC, LinearSVR])
def test_pickle_clone(data, Estimator):
    X, y = data
    m = Estimator().fit(X, y)
    assert_allclose(m.predict(X), pickle.loads(pickle.dumps(m)).predict(X))
    assert not hasattr(clone(m), 'n_features_in_')
    assert clone(m).get_params() == m.get_params()


def test_pipeline_grid_precomputed(data):
    X, y = data
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {'svc__C': [.5, 1]}, cv=3)
    search.fit(X, y)
    assert search.best_score_ > .65
    K = rbf_kernel(X, gamma=.3)
    scores = cross_val_score(SVC(kernel='precomputed'), K, y, cv=3)
    assert scores.mean() > .6


def test_probability(data):
    X, y = data
    m = SVC(probability=True, calibration_cv=3, random_state=9).fit(X, y)
    probability = m.predict_proba(X[:7])
    assert probability.shape == (7, 2)
    assert_allclose(probability.sum(axis=1), 1)
    assert (probability >= 0).all() and (probability <= 1).all()
    assert_allclose(np.exp(m.predict_log_proba(X[:7])), probability)
    assert not hasattr(SVC(), 'predict_proba')


def test_reports_and_path(data):
    X, y = data
    m = SVC(tol=1e-7, history=True).fit(X, y)
    history = m.optimization_history_[0]
    assert np.min(np.diff(history[:, 1])) >= -1e-8
    alpha = m.dual_problem()['alpha']
    alpha[:] = 17
    assert not np.all(m.dual_problem()['alpha'] == 17)
    assert margin_summary(m, X, y)['hinge_loss_mean'] >= 0
    distance = m.rkhs_distance(X)
    assert_allclose(distance, m.decision_function(X)/np.sqrt(m.optimization_report()[0]['rkhs_norm_squared']))
    path = regularization_path(LinearSVC(random_state=9), X, y, [.1, .5, 2], n_jobs=2)
    assert [item['C'] for item in path] == [.1, .5, 2]
    assert all(item['model'].converged_ for item in path)


def test_margin_summary_distinguishes_ties():
    model = SVC(kernel="linear", tol=1e-10).fit([[-1.], [1.]], [0, 1])
    report = margin_summary(model, np.zeros((2, 1)), np.array([0, 1]))
    assert report["error_fraction"] == .5
    assert report["nonpositive_margin_fraction"] == 1.
