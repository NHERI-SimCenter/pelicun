.. _internals:

Package architecture
--------------------

Overview of files
.................

+-------------------+---------------------------------------+
|Path               |Description                            |
+===================+=======================================+
|``pelicun/``       |Main package source code.              |
+-------------------+---------------------------------------+
|``doc/``           |Documentation source code.             |
+-------------------+---------------------------------------+
|``.github/``       |GitHub-related workflow files.         |
+-------------------+---------------------------------------+
|``pyproject.toml`` |Main configuration file.               |
+-------------------+---------------------------------------+
|``setup.py``       |Package setup file.                    |
+-------------------+---------------------------------------+
|``MANIFEST.in``    |Defines files to include when building |
|                   |the package.                           |
+-------------------+---------------------------------------+

.. note::

  We are currently in the process of migrating most configuration files to ``pyproject.toml``.

We use `setuptools <https://setuptools.pypa.io/en/latest/>`_ to build the package, using ``setup.py`` for configuration.
In ``setup.py`` we define an entry point called ``pelicun``, directing to the ``main`` method of ``DL_calculation.py``, used to run pelicun from the command line.

The python source code and unit tests are located under ``pelicun/``.
``assessment.py`` is the primary file defining assessment classes and methods.
Modules under ``model/`` contain various models used by ``Assessment`` objects.
Such models handle the representation of asset inventories, as well as demand, damage, and loss samples.
``base.py`` defines commonly used objects.
``file_io.py`` defines methods related to reading and writing to files.
``uq.py`` defines classes and methods used for uncertainty quantification, including random variable objects and registries, and parameter recovery methods used to fit distributions to raw data samples.
``warnings.py`` defines custom errors and warnings used in pelicun.
``tools/DL_calculation.py`` enables the invocation of analyses using the command line.
``settings/`` contains several ``JSON`` files used to define default units, configuration options and input validation.
``resources/`` contains default damage and loss model parameters.

.. tip::

   For a detailed overview of these files, please see the `API documentation <../api_reference/index.rst>`_ or directly review the source code.

   A direct way to become familiar with an area of the source code you are interested in working with is to debug an applicable example or test and follow through the calculation steps involved, taking notes in the process.
