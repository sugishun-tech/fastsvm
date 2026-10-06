# fastsvm

**Cython/C support-vector machines with optimization certificates and a scikit-learn-style API.**

The training solvers are implemented in this repository. They do not call
`sklearn.svm`, LIBSVM, or LIBLINEAR to fit a model. scikit-learn is used for
estimator conventions, validation, model selection integration, and optional
cross-validated probability calibration.

Version 0.1.0 is a CPU implementation. The native source language is **C**, not C++.
There is no claim of being universally faster than mature SVM libraries.

## Capabilities

| Component | Implementation |
|---|---|
| `LinearSVC` | Native randomized dual coordinate descent; hinge/squared hinge; dense/CSR; binary/OvR |
| `LinearSVR` | Native soft-thresholded dual coordinate descent; epsilon/squared epsilon loss; dense/CSR |
| `SVC` | Second-order SMO; binary/OvO; bounded-memory LRU kernel cache; shrinking |
| `SVR` | Epsilon-SVR through the same native equality-constrained SMO |
| `OneClassSVM` | Normalized nu-SVM with dual mass 1; weighted bounds |
| Kernels | Linear, RBF, polynomial, sigmoid, Laplacian, precomputed, callable, sums/products/scales |
| Approximation | Random Fourier features and rank-truncated PSD Nyström features |
| Mathematics | Primal/dual/gap/KKT/feasibility reports; optimization traces; input gradients and Hessians; RKHS distance |
| Spectral tools | Gram PSD/rank/condition diagnostics; PSD projection; centering; kernel alignment |
| Workflow | `Pipeline`, `GridSearchCV`, `clone`, sample/class weights, CV calibration, discrete C sweeps |

## Install from this source tree

Python >=3.10, Cython >=3.1, NumPy >=2.0, SciPy >=1.11 and scikit-learn >=1.6
are declared dependencies. The executed validation environment is documented
in [test results](docs/test_results.md); declaring a version range is not a
claim that every combination in that range was manually tested.

Debian/Linux:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For a machine-local optimized build (from a clean source tree):

```bash
FASTSVM_NATIVE=1 FASTSVM_OPENMP=1 python -m pip install --no-cache-dir .
```

`FASTSVM_NATIVE=1` enables `-march=native` on GCC/Clang. **Do not distribute
that wheel to machines with potentially different CPU instruction sets.**
Source ZIPs contain no machine-specific binaries. OpenMP is opt-in; omit it
when the compiler/runtime is unavailable. macOS's default compiler typically
requires additional OpenMP setup; the portable default avoids that dependency.
Windows uses MSVC flags and defaults to a serial native build.

When changing build flags in an already-built tree, remove `build/` and the
generated `src/fastsvm/_core.c` first so cached compiler outputs are not reused.

`FASTSVM_DEBUG=1` enables checked Cython indexing and an unoptimized debug build.
`FASTSVM_ANNOTATE=1` asks Cython to emit annotated source HTML.
No `-ffast-math` is enabled, and internal arithmetic/cache values are float64.
Float32 feature inputs are accepted through conversion, not a float32 solver.

```python
from fastsvm import build_info
print(build_info())
# The openmp field reports the actual compiled capability.
```

## Quick start

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fastsvm import SVC

X, y = make_moons(n_samples=1000, noise=0.18, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=9
)
search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": [0.5, 2, 8], "svc__gamma": [0.2, 1]},
    cv=3,
)
search.fit(X_train, y_train)
print(search.score(X_test, y_test))
print(search.best_estimator_[-1].optimization_report())
```

Scale inside the pipeline, not on the entire dataset before cross-validation.
CSR input belongs in `LinearSVC` / `LinearSVR`; exact kernel estimators accept
dense feature arrays or dense precomputed Gram matrices only.

## Inspect the mathematics

```python
import numpy as np
from fastsvm import SVC, Kernel, gram_diagnostics, pairwise_kernel

rng = np.random.default_rng(3)
X = rng.normal(size=(100, 3))
y = (X[:, 0] * X[:, 1] > 0).astype(int)
model = SVC(C=3, gamma=0.4, tol=1e-8, history=True).fit(X, y)
print(model.optimization_report()[0])
print(model.dual_problem()["alpha"])
print(model.decision_gradient(X[:2]))
print(model.decision_hessian(X[:2]))
print(model.rkhs_distance(X[:2]))
print(gram_diagnostics(pairwise_kernel(X, gamma=0.4))["is_psd"])
composite = Kernel("rbf", gamma=0.4) + 0.1 * Kernel("linear")
other = SVC(kernel=composite).fit(X, y)
```

Full definitions, including intercept regularization, the three SMO problem
encodings and the actual stopping conditions, are in [algorithms](docs/algorithms.md).
The [API guide](docs/api.md) lists supported parameters and intentional differences.

## Probability estimates

```python
model = SVC(probability=True, calibration_cv=3, random_state=0).fit(X, y)
print(model.predict_proba(X[:3]))
```

This uses a stratified CV ensemble of sigmoid calibrators. It is not LIBSVM's
internal pairwise probability-coupling algorithm, and its highest-probability
class can differ from the separately fitted raw-vote model's `predict()`.
It performs additional fits; it is not a speed-neutral switch. Internal
calibration is unavailable for precomputed kernels. The default classifier
has no `predict_proba` attribute when `probability=False`.

## Large-sample kernel approximation

```python
from fastsvm import RBFSampler, LinearSVC
approximate = make_pipeline(
    StandardScaler(),
    RBFSampler(gamma=0.1, n_components=512, random_state=0),
    LinearSVC(C=0.5, random_state=0),
)
approximate.fit(X_train, y_train)
```

The finite feature map changes the learning problem; this is not an exact RBF SVM.
Nyström is available through `Nystroem`. Its output dimension is the retained
spectral rank and may be less than `n_components`.

## Performance and threading

See [the measured results](docs/performance.md), including cases that are slower
than scikit-learn, raw timings, accuracy, convergence and experimental conditions.

```bash
python -m pip install '.[dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/benchmark.py --repeat 9
```

`n_jobs` runs independent multiclass fits via joblib threads. Kernel prediction
uses sample-level OpenMP when compiled in and when the workload is large enough;
SMO updates of shared dual/gradient state are never naively parallelized.
A binary optimizer itself is sequential. Calling prediction from a non-main
Python thread uses the serial path for Cython/OpenMP compatibility.
`LinearSVR` has one target and no multiclass task parallelism.
Avoid combining unrestricted CV, multiclass, BLAS and OpenMP parallelism.

`cache_size` is MiB **per binary fit**, with a minimum of two double-precision
rows. Callable/precomputed/`check_psd=True` paths use an explicit full Gram
matrix in addition to cache/workspace memory. Spectral checking is O(n^3).
Prediction evaluates support vectors without allocating a test-by-support Gram
matrix for built-in kernels. A linear kernel is collapsed to a weight vector.

## Validate

```bash
python -m pytest -q
python tests/run_estimator_checks.py
python examples/classification.py
python examples/mathematics.py
python examples/approximation.py
python examples/regression_and_outliers.py
python -m build
```

Numerical tests include an independent SciPy QP oracle, sklearn comparisons,
full-gradient reconstruction, dual feasibility, zero/duplicate samples,
weighted and sparse inputs, finite-difference derivatives and serialization.
The complete sklearn check inventory is exercised at explicitly recorded tight
tolerances and fixed gamma values; see [the validation report](docs/test_results.md).
GitHub Actions configuration is included, but inclusion does not mean the
remote workflow has already been run.

## Important limits

This is **sklearn-style**, not a byte-for-byte drop-in replacement. In particular:
`NuSVC`, `NuSVR`, L1 penalty, primal/TRON solvers, online `partial_fit`, true warm
starts, exact homotopy paths, GPU kernels and sparse exact-kernel SVM are not
implemented. `max_iter` must be a positive integer; sklearn's `-1` sentinel is
not accepted. Kernel estimators count SMO pair steps; linear estimators count
coordinate-descent epochs. Regressors provide `predict`, not `decision_function`.

Linear models regularize the intercept as a synthetic feature; kernel models
do not. One-class dual coefficients/scores are normalized by dual mass 1 rather
than LIBSVM's uniform-weight mass `nu*n`. See [API differences](docs/api.md).
Sigmoid kernels, negative-offset polynomial kernels and arbitrary custom kernels
need not be PSD; stationarity is not global optimality in that case.
`rkhs_distance` is a feature-space hyperplane distance, not input-space boundary distance.

Pickle/joblib round-tripping is tested. Load only trusted pickles; they may
execute code. There is no remote loading or automatic model deserialization.

## Contents and license

- [Qiita article, Japanese](docs/qiita.md)
- [Algorithm notes, Japanese](docs/algorithms.md)
- [API reference, Japanese](docs/api.md)
- [Performance evidence](docs/performance.md)
- [Executed tests and limitations](docs/test_results.md)

MIT; see [LICENSE](LICENSE). No bundled font files, compiled extensions,
third-party solver sources or generated datasets are required in the source ZIP.
