import sys
import glob

from os import path
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("syntax.checker",
              ["syntax/checker.pyx"])
]

if sys.version_info < (2,7):
    sys.exit("Sorry, only Python >= 2.7 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='G&RLForNPS',
    version='0.0.1.dev1',
    description='Research code for the paper'
    'Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis',
    author='Rudy Bunel',
    author_email='rudy@robots.ox.ac.uk',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(),
    install_requires=['cython', 'sh', 'tqdm', 'numpy',
                      'torch', 'torchvision',
                      'GitPython', 'pathlib',
                      'matplotlib'],
    extras_require={
        'tests': ['mypy', 'flake8'],
        'dev': ['ipython', 'ipdb']
    },
    scripts=glob.glob('cmds/*_cmd.py'),
    ext_modules = cythonize(extensions)
)
