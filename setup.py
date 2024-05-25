"""
setup.py file of the `pelicun` package.

"""

import io
from setuptools import setup, find_packages
import pelicun


def read(*filenames, **kwargs):
    """
    Utility function to read multiple files into a string
    """
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')

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
        # 'tables>=3.7.0',
    ],
    extras_require={
        'development': [
            'flake8',
            'flake8-bugbear',
            'flake8-rst',
            'flake8-rst-docstrings',
            'pylint',
            'pylint-pytest',
            'pydocstyle',
            'mypy',
            'black',
            'pytest',
            'pytest-cov',
            'glob2',
            'jupyter',
            'jupytext',
            'sphinx',
            'sphinx-autoapi',
            'nbsphinx',
            'flake8-rst',
            'flake8-rst-docstrings',
            'pandas-stubs',
            'types-colorama',
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
