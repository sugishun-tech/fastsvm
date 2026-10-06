"""Primal/dual certificates, input derivatives, spectra and kernel algebra."""
import numpy as np
from fastsvm import SVC, Kernel, pairwise_kernel, gram_diagnostics, margin_summary


def main():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0]*X[:, 1] > 0).astype(int)
    model = SVC(C=3, gamma=.4, tol=1e-8, history=True).fit(X, y)
    print('Certificate:', model.optimization_report()[0])
    print('Dual equality:', model.dual_problem()['signs'] @ model.dual_problem()['alpha'])
    print('Input gradient:', model.decision_gradient(X[:2]))
    print('Input Hessian:', model.decision_hessian(X[:2]))
    print('RKHS hyperplane distance:', model.rkhs_distance(X[:2]))
    print('Functional margins:', margin_summary(model, X, y)['quantiles'])
    K = pairwise_kernel(X, gamma=.4)
    spectrum = gram_diagnostics(K)
    print('PSD / rank:', spectrum['is_psd'], spectrum['rank'])
    composite = Kernel('rbf', gamma=.4) + .1*Kernel('linear')
    print('Composite kernel score:', SVC(kernel=composite).fit(X, y).score(X, y))


if __name__ == '__main__':
    main()
