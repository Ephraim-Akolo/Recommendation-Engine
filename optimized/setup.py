import distutils.core
import numpy
import Cython.Build
distutils.core.setup(
    ext_modules=Cython.Build.cythonize("optimized_functions.pyx"),
    include_dirs=[numpy.get_include()]
)