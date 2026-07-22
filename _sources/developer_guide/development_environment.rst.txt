.. _development_environment:

Setting up a development environment
------------------------------------

Pelicun uses `uv <https://docs.astral.sh/uv/>`_ to manage its development environment.
Install it by following the `uv installation instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.

Clone the repository::

  git clone https://github.com/NHERI-SimCenter/pelicun

.. tip::

   If you are planning to contribute code, please `fork the repository <https://github.com/NHERI-SimCenter/pelicun/fork>`_ and clone your own fork.

Create the development environment with the following command issued from the package's root directory::

  uv sync --extra test --extra lint

This creates a virtual environment under ``.venv``, installs pelicun in editable mode, and installs the testing and linting tools at the exact versions pinned in ``uv.lock``.
Add ``--extra doc`` if you also want to build the documentation, or use ``--extra development`` to install everything.

Commands are then run through uv, which keeps the environment in sync automatically::

  uv run pytest pelicun/tests
  uv run ruff check pelicun

The Damage and Loss Model Library data used by pelicun comes from the ``simcenter-dlml`` package, which is installed automatically together with the other dependencies.

.. tip::

   If you prefer not to use uv, a plain ``pip`` installation into a `virtual environment <https://docs.python.org/3/library/venv.html>`_ of your choice also works::

     python -m pip install -e .[development]

   Note that this installs the latest versions allowed by the dependency ranges rather than the locked versions used in CI.
