from setuptools import setup
import io

import pelicun


def read(*filenames, **kwargs):
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
    url='http://github.com/NHERI-SimCenter/pelicun/',
    license='BSD License',
    author='Adam Zsarnóczay',
    tests_require=['pytest'],
    author_email='adamzs@stanford.edu',
    description='Probabilistic Estimation of Losses, Injuries, and Community resilience Under Natural disasters',
    long_description=long_description,
    packages=['pelicun'],
    include_package_data=True,
    platforms='any',
    install_requires=[
        'numpy>=1.13',
        'scipy>=1.0',
        'pandas>=0.20',
    ],
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
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
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',        
        ],
    extras_require={
        'testing': ['pytest'],
    }
)