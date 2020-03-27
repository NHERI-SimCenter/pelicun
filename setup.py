from setuptools import setup, find_packages
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
    url='http://nheri-simcenter.github.io/pelicun/',
    license='BSD License',
    author='Adam Zsarnóczay',
    tests_require=['pytest'],
    author_email='adamzs@stanford.edu',
    description='Probabilistic Estimation of Losses, Injuries, and Community resilience Under Natural disasters',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #packages=['pelicun'],
    packages = find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=[
        'numpy>=1.17.0',
        'scipy>=1.3.0',
        'pandas>=1.0.0',
    ],
    classifiers = [
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        ],
    extras_require={
        'testing': ['pytest'],
    }
)