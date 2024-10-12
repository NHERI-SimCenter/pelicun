#
# Copyright (c) 2023 Leland Stanford Junior University
# Copyright (c) 2023 The Regents of the University of California
#
# This file is part of pelicun.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# pelicun. If not, see <http://www.opensource.org/licenses/>.

"""setup.py file of the `pelicun` package."""

from pathlib import Path

from setuptools import find_packages, setup

import pelicun


def read(*filenames, **kwargs) -> None:  # noqa: ANN002, ANN003
    """Read multiple files into a string."""
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with Path(filename).open(encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

# TODO(JVM): update documentation requirements, remove those no longer
# used.

setup(
    name='pelicun',
    version=pelicun.__version__,
    url='http://nheri-simcenter.github.io/pelicun/',
    license='BSD License',
    author='Adam Zsarnóczay',
    tests_require=['pytest'],
    author_email='adamzs@stanford.edu',
    description=(
        'Probabilistic Estimation of Losses, Injuries, '
        'and Community resilience Under Natural hazard events'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    # packages=['pelicun'],
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=[
        'numpy>=1.22.0, <2.0',
        'scipy>=1.7.0, <2.0',
        'pandas>=1.4.0, <3.0',
        'colorama>=0.4.0, <0.5.0',
        'numexpr>=2.8, <3.0',
        'jsonschema>=4.22.0, <5.0',
        # 'tables>=3.7.0',
    ],
    extras_require={
        'development': [
            'ruff',
            'flake8',
            'flake8-bugbear',
            'flake8-rst',
            'flake8-rst-docstrings',
            'pylint',
            'pylint-pytest',
            'pydocstyle',
            'mypy',
            'black',
            'ruff',
            'pytest',
            'pytest-cov',
            'pytest-xdist',
            'glob2',
            'jupyter',
            'jupytext',
            'sphinx-autoapi',
            'flake8-rst',
            'flake8-rst-docstrings',
            'pandas-stubs',
            'types-colorama',
            'codespell',
            'sphinx',
            'sphinx_design',
            'sphinx-rtd-theme',
            'nbsphinx',
            'numpydoc',
            'rendre>0.0.14',
            'jsonpath2',
        ],
    },
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    entry_points={
        'console_scripts': [
            'pelicun = pelicun.tools.DL_calculation:main',
        ]
    },
)
