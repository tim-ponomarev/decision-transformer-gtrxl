"""Build the gridworld_fast extension via setuptools + pybind11.setup_helpers.

Usage (from this directory):
    python setup.py build_ext --inplace

Or (recommended) from project root:
    pip install ./cpp

This is the cross-platform fallback when CMake/find_package(pybind11) is
inconvenient — setup_helpers picks the right compiler flags for MSVC, GCC,
Clang, and MinGW automatically and links against the running Python's ABI.
"""
from __future__ import annotations

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# AVX2 + optimization flags. setup_helpers handles -std=c++17 automatically.
extra_compile_args: list[str] = []
extra_link_args: list[str] = []

if sys.platform == "win32":
    # MSVC and MinGW both accept these — MSVC reads /arch:AVX2, MinGW gcc reads -mavx2.
    # We pass both shapes; the wrong one is ignored by each compiler.
    extra_compile_args += ["/O2", "/arch:AVX2"]
    extra_compile_args += ["-O3", "-mavx2", "-mfma"]
else:
    extra_compile_args += ["-O3", "-mavx2", "-mfma", "-ffast-math"]

ext = Pybind11Extension(
    "gridworld_fast",
    sources=["env_fast.cpp", "bindings.cpp"],
    include_dirs=["."],
    cxx_std=17,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="gridworld_fast",
    version="0.1.0",
    description="Batched GridWorld with AVX2 SIMD over the batch dimension.",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
