from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import eigency

extensions = [
    Extension("pyov2sgd/ov2sgd",
              ["pyov2sgd/ov2sgd.pyx"],
              language="c++",
              include_dirs=[".", "include", "src"] + eigency.get_includes()
              ),
]

setup(
    name="ov2sgd",
    ext_modules=cythonize(extensions),
)
