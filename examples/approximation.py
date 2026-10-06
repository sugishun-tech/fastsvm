"""Approximate RBF learning; the feature map is fitted inside each CV fold."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fastsvm import RBFSampler, Nystroem, LinearSVC


def main():
    X, y = make_classification(n_samples=2000, n_features=16, n_informative=10, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=9)
    for mapping in [RBFSampler(gamma=.05, n_components=512, random_state=0),
                    Nystroem(gamma=.05, n_components=200, random_state=0)]:
        model = make_pipeline(StandardScaler(), mapping,
                              LinearSVC(C=.5, random_state=0, max_iter=20000))
        model.fit(X_train, y_train)
        print(type(mapping).__name__, 'held-out accuracy:', model.score(X_test, y_test))


if __name__ == '__main__':
    main()
