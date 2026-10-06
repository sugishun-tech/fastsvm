"""Portable C build; OpenMP/native tuning are explicit opt-ins."""
import os
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

omp = os.getenv("FASTSVM_OPENMP", "0") == "1"
native = os.getenv("FASTSVM_NATIVE", "0") == "1"
debug = os.getenv("FASTSVM_DEBUG", "0") == "1"
msvc = sys.platform == "win32"
compile_args = ["/O2"] if msvc else ["-O3", "-fno-math-errno"]
link_args = []
if omp:
    compile_args += ["/openmp"] if msvc else ["-fopenmp"]
    if not msvc:
        link_args += ["-fopenmp"]
if native:
    if msvc:
        raise RuntimeError("FASTSVM_NATIVE is supported only with GCC/Clang.")
    compile_args += ["-march=native"]
if debug:
    compile_args = ["/Od"] if msvc else ["-O0", "-g"]
    if omp:
        compile_args += ["/openmp"] if msvc else ["-fopenmp"]

ext = Extension(
    "fastsvm._core", ["src/fastsvm/_core.pyx"], language="c",
    include_dirs=[np.get_include()], extra_compile_args=compile_args,
    extra_link_args=link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)
setup(ext_modules=cythonize(
    [ext], compiler_directives={"language_level": 3, "boundscheck": debug,
    "wraparound": debug, "initializedcheck": debug, "cdivision": True},
    annotate=os.getenv("FASTSVM_ANNOTATE", "0") == "1",
))
