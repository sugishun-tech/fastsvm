"""Kernel regression, input sensitivity and normalized one-class anomaly scores."""
import numpy as np
from fastsvm import SVR, OneClassSVM


def main():
    rng = np.random.default_rng(12)
    X = np.linspace(-3, 3, 200)[:, None]
    y = np.sin(X[:, 0])+.04*rng.normal(size=len(X))
    reg = SVR(C=4, epsilon=.06, gamma=.5, tol=1e-6).fit(X, y)
    print('Predictions:', reg.predict([[.1], [.2]]))
    print('Slopes:', reg.decision_gradient([[.1], [.2]]))
    normal = rng.normal(size=(200, 2))
    detector = OneClassSVM(nu=.1, gamma=.2, tol=1e-7).fit(normal)
    print('Inlier/outlier:', detector.predict([[0, 0], [20, 20]]))
    print('Normalized scores:', detector.score_samples([[0, 0], [20, 20]]))


if __name__ == '__main__':
    main()
