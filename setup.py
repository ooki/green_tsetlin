import sys

# Available at setup time due to pyproject.toml
import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from cpuinfo import get_cpu_info
from setuptools import setup, find_packages

import distutils
distutils.log.set_verbosity(1)


__version__ = "1.1.0"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

cpu_info = get_cpu_info()
has_avx2 = "avx2" in cpu_info.get("flags", "")

brand_raw = cpu_info.get("brand_raw", "").lower().split()
has_neon = "m1" in brand_raw or "m2" in brand_raw or "m3" in brand_raw or "m4" in brand_raw



compile_args = []
define_macros = [('VERSION_INFO', __version__)]

if sys.platform.startswith('darwin') and has_neon:
    # compile_args = ["-target arm64-apple-macos11", "-arch=arm64", "-O3", "-mmacosx-version-min=10.15", "-mfloat-abi=hard"]  # -mfloat-abi=hard needs to be used with neon
    compile_args = ["-arch=arm64", "-O3", "-mfpu=neon", "-mmacosx-version-min=10.15",  "-DNDEBUG"]  # -mfloat-abi=hard needs to be used with neon

    # also took out: "-arch=arm64", "-mfloat-abi=hard", 
    define_macros.append(("USE_NEON", 1))

elif sys.platform.startswith('win32'):
    compile_args =["/EHsc", "/O2", "/arch:AVX2", "/MT", "/DNDEBUG"]
    #if has_avx2:
    define_macros.append(("USE_AVX2", 1))

else: # assume we are on linux (or g++) at least if nothing else is specified.
    #elif sys.platform.startswith('linux') or (sys.platform.startswith('darwin') and has_neon is False):    
    compile_args =["-mavx2", "-mfma", "-O3", "-pthread", "-DNDEBUG"]

    if has_avx2:
        define_macros.append(("USE_AVX2", 1))


     
# print("using defines:", define_macros)

ext_modules = [
    Pybind11Extension("green_tsetlin_core",
        ["src/main.cpp"],
        define_macros = define_macros,
        cxx_std=17,
        include_dirs=["src/"],       
        extra_compile_args=compile_args
        ),
]

setup(
    name="green_tsetlin",
    version=__version__,
    author="Sondre 'Ooki' Glimsdal",
    author_email="sondre.glimsdal@gmail.com",
    url="https://github.com/ooki/green_tsetlin",
    project_urls= {
        "Bug Tracker": "https://github.com/ooki/green_tsetlin/issues",
        "Documentation": "https://github.com/ooki/green_tsetlin",
        "Source Code": "https://github.com/ooki/green_tsetlin",
    },
    description="A fast Tsetlin Machine impl, based on c++",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",        
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
          'numpy >= 1.24',
          'scipy >= 1.10.1',
          'scikit-learn >= 1.2',
          'tqdm >= 4.65',
          'optuna'
      ],
    tests_require=['pytest'],
    test_suite='tests',
)
