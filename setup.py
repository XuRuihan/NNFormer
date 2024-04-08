from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


setup(
    name="neuralformer",
    version="0.0.1",
    author="Ruihan Xu",
    author_email="xuruihan@pku.edu.cn",
    description="Neural Predictor with Transformer",
    packages=find_packages(),
    ext_modules=cythonize("neuralformer/data_process/nasbench/algos.pyx"),
    include_dirs=[numpy.get_include()],
)
