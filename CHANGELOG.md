# Changelog

## 0.1.0 — 2026-10-06

Initial implementation: Cython/C dual coordinate descent for dense/CSR linear
classification and regression; generalized second-order SMO for C-SVC,
epsilon-SVR and normalized one-class SVM; O(1) LRU kernel rows, conservative
shrinking with mandatory full KKT recheck, direct-distance RBF accumulation,
optional OpenMP inference, binary/OvO/OvR classification, class/sample weights,
CV sigmoid probability calibration, composite kernels, Fourier/Nyström maps,
optimization certificates, analytic derivatives, spectral tools, tests,
reproducible timing scripts and Japanese mathematical documentation.

No LIBSVM/LIBLINEAR solver source is bundled or called by the training backend.
scikit-learn remains an API/validation/calibration dependency and test reference.
