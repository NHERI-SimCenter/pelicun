.. _development_environment:

Setting up a development environment
------------------------------------

.. tip::

   We recommend creating a dedicated `virtual environment <https://docs.python.org/3/library/venv.html>`_ for your pelicun development environment.
   See also `conda <https://docs.conda.io/en/latest/>`_ and `mamba <https://mamba.readthedocs.io/en/latest/>`_, two widely used programs featuring environment management.

Clone the repository::

  git clone --recurse-submodules https://github.com/NHERI-SimCenter/pelicun

Pelicun uses the SimCenter `DamageAndLossModelLibrary <https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary>`_ as a submodule.
In the above, ``recurse-submodules`` ensures that all files of that repository are also obtained.

.. tip::

   If you are planning to contribute code, please `fork the repository <https://github.com/NHERI-SimCenter/pelicun/fork>`_ and clone your own fork.


Install pelicun in editable mode with the following command issued from the package's root directory::

  python -m pip install -e .[development]

This will install pelicun in editable mode as well as all dependencies needed for development.
