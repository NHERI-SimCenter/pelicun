.. _code_quality:

=================
 Coding practice
=================

Code quality assurance
======================

We use `Ruff <https://docs.astral.sh/ruff/>`_, `mypy <https://mypy-lang.org/>`_ and `Codespell <https://github.com/codespell-project/codespell>`_ to maintain a high level of quality of the pelicun code.
Our objective is to always use the latest mature coding practice recommendations emerging from the Python community to ease adaptation to changes in its dependencies or even Python itself.

We use the `numpy docstring style <https://numpydoc.readthedocs.io/en/latest/format.html>`_, and include comments in the code explaining the ideas behind various operations.
We are making an effort to use unambiguous variable and class/method/function names.
Especially for newer code, we are mindful of the complexity of methods/functions and break them down when they start to become too large, by extracting appropriate groups of lines and turning them into appropriately named hidden methods/functions.

All code checking tools should be available when :ref:`installing pelicun under a developer setup <development_environment>`.
The most straight-forward way to run those tools is via the command-line.
All of the following commands are assumed to be executed from the package root directory (the one containing ``pyproject.toml``).

Linting and formatting with Ruff
--------------------------------

.. code:: bash

   ruff check   # Lint all files in the current directory.
   ruff format  # Format all files in the current directory.

Ruff can automatically fix certain warnings when it is safe to do so. See also `Fixes <https://docs.astral.sh/ruff/linter/#fixes>`_.

.. code::

  ruff check --fix

Warnings can also be automatically suppressed by adding #noqa directives. See `here <https://docs.astral.sh/ruff/linter/#inserting-necessary-suppression-comments>`_.

.. code:: bash

   ruff check --add-noqa

Editor integration
..................

Like most code checkers, Ruff can be integrated with several editors to enable on-the-fly checks and auto-formatting.
Under `Editor Integration <https://docs.astral.sh/ruff/editors/>`_, their documentation describes the steps to enable this for VS Code, Neovim, Vim, Helix, Kate, Sublime Text, PyCharm, Emacs, TextMate, and Zed.

Type checking with mypy
-----------------------

Pelicun code is type hinted.
We use ``mypy`` for type checking.
Use the following command to type-check the code:

.. code:: bash

   mypy pelicun --no-namespace-packages

Type checking warnings can be silenced by adding ``#type: ignore`` at the lines that trigger them.
Please avoid silencing warnings in newly added code.

Spell checking with Codespell
-----------------------------

Codespell is a Python package used to check for common spelling mistakes in text files, particularly source code. It is available on PyPI, configured via pyproject.toml, and is executed as follows:

.. code:: bash

   codespell .

False positives can be placed in a dedicated file (we currently call it ``ignore_words.txt``) to be ignored.
Please avoid using variable names that trigger codespell.
This is easy when variable names are long and explicit.
E.g., ``response_spectrum_target`` instead of ``resptr``.

Unit tests
==========

We use `pytest <https://docs.pytest.org/en/stable/>`_ to write unit tests for pelicun.
The tests can be executed with the following command.

.. code:: bash

   python -m pytest pelicun/tests --cov=pelicun --cov-report html

When the test runs finish, visit ``htmlcov/index.html`` for a comprehensive view of code coverage.

The tests can be debugged like any other Python code, by inserting ``breakpoint()`` at any line and executing the line above.
When the breakpoint is reached you will gain access to ``PDB``, the Python Debugger.

Please extend the test suite whenever you introduce new pelicun code, and update it if needed when making changes to the existing code.
Avoid opening pull requests with changes that reduce code coverage by not writing tests for your code.

Documentation
=============

We use `sphinx <https://www.sphinx-doc.org/en/master/>`_ with the `Read the Docs theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_ to generate our documentation pages.

We use the following extensions:

- `nbsphinx <https://nbsphinx.readthedocs.io/en/0.9.5/>`_ to integrate jupyter notebooks into the documentation, particularly for the pelicun examples.
  In the source code they are stored as python files with special syntax defining individual cells, and we use `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ to automatically turn them into notebooks when the documentation is compiled (see ``nbsphinx_custom_formats`` in ``conf.py``).

- `Sphinx design <https://sphinx-design.readthedocs.io/en/latest/>`_ for cards and drop-downs.

- `sphinx.ext.mathjax <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_ for math support.

- `sphinx.ext.doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ to actively test examples included in docstrings.

- `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ and `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ to generate the API documentation from the docstrings in the source code.

- `sphinx.ext.autosummary <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_ for the API docs.

- `sphinx.ext.viewcode <https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html>`_ to add links that point to the source code in the API docs.

- `sphinx.ext.intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ to link to other projects' documentation.

- `sphinx.ext.githubpages <https://www.sphinx-doc.org/en/master/usage/extensions/githubpages.html>`_ for publishing in GitHub pages.


Building the documentation
--------------------------

To build the documentation, navigate to `doc` and run the following command:

.. tab-set::

   .. tab-item:: Linux & Mac

      .. code:: bash

         make html

   .. tab-item:: Windows

      .. code:: bash

         .\make.bat html

To see more options:

.. tab-set::

   .. tab-item:: Linux & Mac

      .. code:: bash

         make

   .. tab-item:: Windows

      .. code:: bash

         .\make.bat


Extending the documentation
---------------------------

Extending the documentation can be done in several ways:

- By adding content to ``.rst`` files or adding more such files.
  See the structure of the ``doc/source`` directory and look at the ``index.rst`` files to gain familiarity with the structure of the documentation.
  When a page is added it will need to be included in a ``toctree`` directive in order for it to be registered and have a way of being accessed.
- By adding or modifying example notebooks under ``doc/examples/notebooks``.
  When a new notebook is added, it needs to be included in ``doc/examples/index.rst``.
  Please review that index file and how other notebooks are listed to become familiar with our approach.

After making a change you can simply rebuild the documentation with the command above.
Once the documentation pages are built, please verify no Sphinx warnings are reported.
A warning count is shown after "build succeeded", close to the last output line:

.. code-block:: none
  :emphasize-lines: 3

  [...]
  dumping object inventory... done
  build succeeded, 1 warning.

  The HTML pages are in build/html.

If there are warnings, please address them before contributing your changes.
