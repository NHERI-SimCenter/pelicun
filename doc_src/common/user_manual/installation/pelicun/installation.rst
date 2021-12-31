.. _lblInstallation:

************
Installation
************

pelicun is available for Python 3.6+ at the Python Package Index (PyPI). You can install it using ``pip``.

On Windows::

	pip install pelicun

On a Mac or Unix systems, to make sure you install pelicun under python 3::

	python3 -m pip install pelicun

If you have several Python interpreters on your computer, replace python3 in the above command with the location of the Python executable that belongs to the interpreter you would like to use. For example (assuming your executable is `C:/Python37/python.exe`)::

	C:/Python37/python.exe -m pip install pelicun


Dependencies
------------
The following packages are required for pelicun. They are automatically installed (or upgraded if necessary) when you use ``pip`` to install pelicun.

+---------+-----------------+
| package | minimum version |
+=========+=================+
| numpy   | 1.19            |
+---------+-----------------+
| scipy   | 1.5             |
+---------+-----------------+
| pandas  | 1.1             |
+---------+-----------------+
| tables  |                 |
+---------+-----------------+
| h5py    |                 |
+---------+-----------------+
| xlrd    |                 |
+---------+-----------------+


Staying up to date
------------------

When a new version is released, you can use ``pip`` to upgrade your pelicun library::

	pip install pelicun --upgrade
