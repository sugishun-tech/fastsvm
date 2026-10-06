"""Independent invariants: these tests do not infer correctness from accuracy alone."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from fastsvm import SVC, SVR, OneClassSVM, LinearSVC, LinearSVR, pairwise_kernel


@pytest.fixture
def data():
    X, y = make_classification(n_samples=180, n_features=7, n_informative=5, random_state=30)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize('E', [SVC, SVR, OneClassSVM])
def test_shrinking_full_kkt(data, E):
    X, y = data
    target = np.sin(X[:, 0]) if E is SVR else y
    a = E(tol=1e-8, shrinking=True).fit(X, target)
    b = E(tol=1e-8, shrinking=False).fit(X, target)
    assert a.converged_ and b.converged_
    assert_allclose(a._pair_scores(X), b._pair_scores(X), atol=2e-7)
    # Reconstruct the gradient from scratch, independently of the solver cache.
    d = a.dual_problem()
    K = pairwise_kernel(X, gamma=a._gamma)
    Q = K[np.ix_(d['mapping'], d['mapping'])] * d['signs'][:, None] * d['signs'][None, :]
    if E is SVC:
        p = -np.ones(len(X))
    elif E is SVR:
        p = np.r_[a.epsilon-target, a.epsilon+target]
    else:
        p = np.zeros(len(X))
    fresh = Q@d['alpha']+p
    assert_allclose(d['gradient'], fresh, atol=2e-11)
    alpha, bounds, s = d['alpha'], d['bounds'], d['signs']
    up = (bounds>0) & (((s>0) & (alpha<bounds)) | ((s<0) & (alpha>0)))
    low = (bounds>0) & (((s>0) & (alpha>0)) | ((s<0) & (alpha<bounds)))
    violation = max(0., np.max((-s*fresh)[up])-np.min((-s*fresh)[low]))
    assert violation <= 1.01*a.tol


@pytest.mark.parametrize('E', [LinearSVC, LinearSVR])
@pytest.mark.parametrize('csr', [False, True])
def test_linear_shrinking(data, E, csr):
    X, y = data
    target = np.sin(X[:, 0]) if E is LinearSVR else y
    X = sparse.csr_matrix(X) if csr else X
    a = E(tol=1e-7, max_iter=100000, shrinking=True, random_state=3).fit(X, target)
    b = E(tol=1e-7, max_iter=100000, shrinking=False, random_state=3).fit(X, target)
    assert a.converged_ and b.converged_
    assert_allclose(a.coef_, b.coef_, atol=5e-7)


@pytest.mark.parametrize('E', [SVC, SVR, OneClassSVM])
def test_parallel_prediction(data, E):
    X, y = data
    m = E(n_jobs=1, tol=1e-6).fit(X, y)
    # Large enough to cross the prange launch threshold.
    test = np.tile(X, (8, 1))
    a = m._pair_scores(test)
    m.n_jobs = 2
    assert_array_equal(a, m._pair_scores(test))


def test_closed_form_two_points():
    X = np.array([[-1.], [1.]])
    y = np.array([-1, 1])
    m = SVC(kernel='linear', C=1., tol=1e-10).fit(X, y)
    assert_allclose(m.coef_, [[1.]], atol=1e-12)
    assert_allclose(m.intercept_, [0.], atol=1e-12)
    assert_allclose(m.dual_problem()['alpha'], [.5, .5], atol=1e-12)
    assert_allclose(m.primal_objective_, [.5], atol=1e-12)
    assert_allclose(m.duality_gap_, [0.], atol=1e-12)


@pytest.mark.parametrize('E', [SVC, SVR, OneClassSVM, LinearSVC, LinearSVR])
def test_no_sklearn_solver_delegation(data, monkeypatch, E):
    import sklearn.svm
    def forbidden(*args, **kwargs):
        raise AssertionError('A reference solver was unexpectedly invoked.')
    for name in ['SVC', 'SVR', 'OneClassSVM', 'LinearSVC', 'LinearSVR']:
        monkeypatch.setattr(getattr(sklearn.svm, name), 'fit', forbidden)
    X, y = data
    model = E().fit(X, y)
    assert np.isfinite(model.predict(X)).all()


def test_weighted_oneclass_permutation(data):
    X, _ = data
    w = np.linspace(.1, 2., len(X))
    permutation = np.random.default_rng(41).permutation(len(X))
    a = OneClassSVM(gamma=.3, tol=1e-9).fit(X, sample_weight=w)
    b = OneClassSVM(gamma=.3, tol=1e-9).fit(X[permutation], sample_weight=w[permutation])
    assert_allclose(a.decision_function(X), b.decision_function(X), atol=2e-8)


def test_multiclass_probability(data):
    X, y = make_classification(n_samples=90, n_features=5, n_classes=3,
                               n_informative=4, n_redundant=0, random_state=4)
    X = StandardScaler().fit_transform(X)
    m = SVC(probability=True, calibration_cv=3, random_state=2).fit(X, y)
    probability = m.predict_proba(X)
    assert probability.shape == (90, 3)
    assert_allclose(probability.sum(1), 1)


@pytest.mark.parametrize('E', [SVC, LinearSVC])
def test_balanced_weights_reference(data, E):
    from sklearn import svm
    X, y = data
    weight = np.linspace(.2, 2.1, len(y))
    params = dict(C=.4, class_weight='balanced', tol=1e-7, max_iter=100000)
    if E is LinearSVC:
        params['random_state'] = 0
    model = E(**params).fit(X, y, weight)
    Ref = getattr(svm, E.__name__)
    reference = Ref(**params, **({'dual': True} if E is LinearSVC else {})).fit(X, y, weight)
    assert_allclose(model.decision_function(X), reference.decision_function(X), atol=2e-5)
