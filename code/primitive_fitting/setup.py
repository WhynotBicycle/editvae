from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from itertools import dropwhile
import numpy as np
from os import path

module1 = Extension(
            "fast_sampler._sampler",
            [
                "fast_sampler/_sampler.pyx",
                "fast_sampler/sampling.cpp"
            ],
            language="c++11",
            libraries=["stdc++"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++11", "-O3"]
)

setup (name = 'fast_sampler',
       version = '1.0',
       description = 'sampling point cloud from primitives',
       ext_modules = [module1])



# def get_extensions():
#     return cythonize([
#         Extension(
#             "fast_sampler._sampler",
#             [
#                 "fast_sampler/_sampler.pyx",
#                 "fast_sampler/sampling.cpp"
#             ],
#             language="c++11",
#             libraries=["stdc++"],
#             include_dirs=[np.get_include()],
#             extra_compile_args=["-std=c++11", "-O3"]
#         )
#     ])

# if __name__ == "__main__":
#     get_extensions()