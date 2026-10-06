import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from fastsvm import SVC, SVR, OneClassSVM, LinearSVC, LinearSVR, pairwise_kernel


@pytest.mark.parametrize('E', [SVC, SVR, OneClassSVM, LinearSVC, LinearSVR])
def test_unfitted_invalid(E):
    m = E()
    with pytest.raises(NotFittedError):
        m.predict([[0, 1]])
    with pytest.raises(ValueError):
        m.fit([[np.nan, 1], [2, 1]], [0, 1])
    with pytest.raises(ValueError):
        m.fit([[1, 1], [2, 1]], [0, 1], sample_weight=[-1, 1])
    with pytest.raises(ValueError):
        m.fit([[1, 1], [2, 1]], [0, 1], sample_weight=[0, 0])
    with pytest.raises(ValueError):
        m.set_params(tol=-1).fit([[1, 1], [2, 1]], [0, 1])


@pytest.mark.parametrize('E', [SVC, LinearSVC])
def test_labels_weight_zero_and_one_class(E):
    X = np.array([[0., 0], [1., 1], [2., 2], [3., 3]])
    with pytest.raises(ValueError):
        E().fit(X, [0, 0, 0, 0])
    with pytest.raises(ValueError):
        E().fit(X, [0, 0, 1, 1], sample_weight=[1, 1, 0, 0])
    m = E(tol=1e-7).fit(X, ['cat', 'cat', 'dog', 'dog'])
    assert set(m.predict(X)) <= {'cat', 'dog'}
    with pytest.raises(ValueError):
        m.predict([[1, 2, 3]])


@pytest.mark.parametrize('E', [SVC, SVR, LinearSVC, LinearSVR])
def test_duplicates_and_zeros(E):
    X = np.zeros((12, 4))
    y = np.tile([0, 1], 6)
    m = E(tol=1e-7).fit(X, y)
    assert np.isfinite(m.predict(X)).all()
    assert m.converged_
    for report in m.optimization_report():
        assert report['duality_gap'] >= -1e-9
        assert report['relative_duality_gap'] < 1e-5


def test_rbf_large_common_offset():
    X = np.array([[1e12, 1e12], [1e12+1, 1e12+2]])
    K = pairwise_kernel(X, gamma=.2)
    assert_allclose(K[0, 1], np.exp(-1), rtol=1e-12)
    assert_allclose(np.diag(K), 1)


@pytest.mark.parametrize('E', [SVC, SVR, OneClassSVM, LinearSVC, LinearSVR])
def test_nonconvergence_warning(E):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 5))
    y = rng.integers(0, 2, 80)
    with pytest.warns(ConvergenceWarning):
        m = E(max_iter=1, tol=1e-12).fit(X, y)
    assert not m.converged_ and m.fit_status_ == 1
    assert m.kkt_violation_.max() > m.tol


def test_nu_one_and_single_sample():
    X = np.arange(20.).reshape(10, 2)
    for m in [OneClassSVM(nu=1).fit(X), OneClassSVM().fit([[1, 2]])]:
        assert m.converged_
        assert_allclose(m.dual_mass_, 1)
        assert np.isfinite(m.intercept_).all()


def test_sparse_duplicates_and_readonly():
    X = np.array([[0., 1], [1., 0], [2., 1], [3., 2]])
    y = [0, 0, 1, 1]
    rows = sparse.csr_matrix(X)
    duplicated = sparse.csr_matrix((np.repeat(rows.data/2, 2), np.repeat(rows.indices, 2), rows.indptr*2), shape=X.shape)
    for E in [LinearSVC, LinearSVR]:
        a = E(tol=1e-7, random_state=3).fit(duplicated, y)
        b = E(tol=1e-7, random_state=3).fit(X, y)
        assert_allclose(a.predict(X), b.predict(X), atol=1e-7)
    X.flags.writeable = False
    assert np.isfinite(SVC().fit(X, y).decision_function(X)).all()


def test_bad_gram():
    with pytest.raises(ValueError, match='square'):
        SVC(kernel='precomputed').fit(np.ones((3, 2)), [0, 1, 1])
    with pytest.raises(ValueError, match='symmetric'):
        SVC(kernel='precomputed').fit([[1, 0], [1, 1]], [0, 1])


def test_refit_clears_linear_coef():
    X = np.arange(20.).reshape(10, 2)
    y = np.arange(10) % 2
    m = SVC(kernel='linear').fit(X, y)
    assert hasattr(m, 'coef_')
    m.set_params(kernel='rbf').fit(X, y)
    assert not hasattr(m, 'coef_')


def test_zero_weight_exclusion():
    rng = np.random.default_rng(28)
    X = rng.normal(size=(35, 4))
    y = (X[:, 0] > 0).astype(int)
    weights = np.ones(len(X))
    weights[::4] = 0
    for E in [SVC, LinearSVC]:
        kw = {'gamma': .3} if E is SVC else {'random_state': 3}
        a = E(tol=1e-7, **kw).fit(X, y, weights)
        b = E(tol=1e-7, **kw).fit(X[weights>0], y[weights>0])
        assert_allclose(a.decision_function(X), b.decision_function(X), atol=3e-6)


def test_polynomial_overflow_rejected():
    with pytest.raises(ValueError, match='overflow'):
        SVC(kernel='poly', gamma=1., degree=4).fit([[1e80, 1e80], [-1e80, -1e80]], [0, 1])


def test_constant_rbf_at_extreme_finite_inputs():
    X = np.array([[-1e300, 1e300], [1e300, -1e300]])
    assert_allclose(pairwise_kernel(X, gamma=0.), np.ones((2, 2)))
