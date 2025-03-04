#from Cython.Build import cythonize
#from setuptools import setup, Extension
from distutils.core import setup
from Cython.Build import cythonize
from numpy.distutils.core import setup
import numpy

"""
ext_modules = [
    Extension(
        "nhpylm_model",
        ["nhpylm/nhpylm_model.pyx"],
        defined_macros=[("CYTHON_LIMITED_API", "1")],
        py_limited_api=True
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)
"""
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/random_utils.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/sequence.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/npylm.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/hyperparameters.pyx"),
    include_dirs=[numpy.get_include()],
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/blocked_gibbs_sampler.pyx"),
    include_dirs=[numpy.get_include()],
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/viterbi_algorithm.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("nhpylm/models.pyx"),
    include_dirs=["nhpylm/", numpy.get_include()]
)
