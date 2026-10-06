"""fastsvm: independently implemented Cython/C support-vector solvers."""
from . import _core
from .linear import LinearSVC, LinearSVR
from .kernels import Kernel, KernelCenterer, pairwise_kernel, gram_diagnostics, project_psd, kernel_alignment

__version__ = "0.1.1"
build_info = _core.build_info
__all__ = ["LinearSVC", "LinearSVR", "Kernel", "KernelCenterer", "pairwise_kernel",
           "gram_diagnostics", "project_psd", "kernel_alignment", "build_info"]
from .svm import SVC, SVR, OneClassSVM
__all__ += ["SVC", "SVR", "OneClassSVM"]
from .approximation import RBFSampler, Nystroem
from .analysis import regularization_path, margin_summary
__all__ += ["RBFSampler", "Nystroem", "regularization_path", "margin_summary"]
