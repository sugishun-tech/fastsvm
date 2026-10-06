"""Train/test split before scaling; Pipeline also prevents CV leakage."""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fastsvm import SVC


def main():
    X, y = make_moons(n_samples=1200, noise=.18, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                      stratify=y, random_state=9)
    search = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                          {'svc__C': [.5, 2, 8], 'svc__gamma': [.2, 1]}, cv=3)
    search.fit(X_train, y_train)
    print('Best parameters:', search.best_params_)
    print('Held-out accuracy:', search.score(X_test, y_test))
    print('Certificates:', search.best_estimator_[-1].optimization_report())


if __name__ == '__main__':
    main()
