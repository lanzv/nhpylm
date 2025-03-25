from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("nhpylm.blocked_gibbs_sampler", ["nhpylm/blocked_gibbs_sampler.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nhpylm.hyperparameters", ["nhpylm/hyperparameters.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nhpylm.models", ["nhpylm/models.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nhpylm.npylm", ["nhpylm/npylm.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nhpylm.random_utils", ["nhpylm/random_utils.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nhpylm.sequence", ["nhpylm/sequence.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nhpylm.viterbi_algorithm", ["nhpylm/viterbi_algorithm.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="nhpylm",
    version="0.0.1",
    packages=["nhpylm"],
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=["numpy<2.0", "cython==3.0.12"],
    include_package_data=True,
    zip_safe=False,
)

