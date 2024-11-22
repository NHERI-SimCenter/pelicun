.. _development_environment:

Setting up a development environment
------------------------------------

.. tip::

   We recommend creating a dedicated `virtual environment <https://docs.python.org/3/library/venv.html>`_ for your pelicun development environment.
   See also `conda <https://docs.conda.io/en/latest/>`_ and `mamba <https://mamba.readthedocs.io/en/latest/>`_, two widely used programs featuring environment management.

Install pelicun in editable mode with the following command issued from the package's root directory::

  python -m pip install -e .[development]

This will install pelicun in editable mode as well as all dependencies needed for development.
