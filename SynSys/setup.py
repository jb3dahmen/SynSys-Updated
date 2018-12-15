import sys

from setuptools import setup
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext

from Cython.Build import cythonize

args = sys.argv[1:]



try:
    from Cython.Build import cythonize
except ImportError:
    #If you don't have cython assumes you have the .c files
    use_cython = False
    ext = 'c'
else:
    #will generate the .c files
    use_cython = True
    ext = 'pyx'

filenames = [ "base",
              "BayesianNetwork",
              "FactorGraph",
              "distributions",
              "hmm",
              "gmm",
              "NaiveBayes",
              "MarkovChain",
              "utils",
              "parallel",
              "kmeans",
              "bayes",
              "BayesClassifier"
            ]

if not use_cython:
    extensions = [
        Extension( "pomegranate.{}".format( name ),
                   [ "pomegranate/{}.{}".format( name, ext ) ]) for name in filenames
    ]
else:
    extensions = [
            Extension("pomegranate.*", ["pomegranate/*.pyx"])
    ]

    extensions = cythonize( extensions )

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='synsys',
    version='0.1.0',
    author='Jess Dahmen',
    author_email='jb3dahmen@wsu.edu',
    packages=['pomegranate'],
    license='LICENSE.txt',
    ext_modules=extensions,
    cmdclass={'build_ext':build_ext},
    setup_requires=[
        "cython >= 0.22.1",
        "numpy >= 1.8.0",
        "scipy >= 0.17.0"
    ],
    install_requires=[
        "numpy >= 1.8.0",
        "joblib >= 0.9.0b4",
        "networkx >= 1.8.1, < 2.0",
        "scipy >= 0.17.0"
    ],
    test_suite = 'nose.collector',
    package_data={
        'pomegranate': ['*.pyd']
    },
    include_package_data=True,
)
