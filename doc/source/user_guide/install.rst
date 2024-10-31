.. _user_install:

Welcome to the pelicun user guide.
Below, you will find instructions on installing pelicun and information about the supported features, the basic concepts behind them, the terminology used, the expected inputs, and where to get help.
Join our growing community of users and developers dedicated to advancing risk estimation practices and sharing insights.


Getting started
---------------

`Pelicun <https://pypi.org/project/pelicun/>`_ is available on the Python Package Index (PyPI) and should work out-of-the-box in all major platforms.

.. tip::

   We recommend installing the package under a `virtual environment <https://docs.python.org/3/library/venv.html>`_ to avoid dependency conflicts with other packages.
   See also `conda <https://docs.conda.io/en/latest/>`_ and `mamba <https://mamba.readthedocs.io/en/latest/>`_, two widely used programs featuring environment management.

Install command::

  python -m pip install pelicun

Staying up to date
..................

When a new version is released, you can use ``pip`` to upgrade::

  python -m pip install --upgrade pelicun


..
   pelicun is an open-source library (|github link|) released under a **3-Clause BSD** license (see :numref:`lblLicense`). The pelicun library can be used to quantify damages and losses from an earthquake or hurricane scenario in the form of decision variables (DVs). This functionality is typically utilized for performance-based engineering and regional natural hazard risk assessment. This library can help in several steps of performance assessment:

   * **Describe the joint distribution of asset response.** The response of a structure or other type of asset to natural hazard event is typically described by so-called engineering demand parameters (EDPs). pelicun provides various options to characterize the distribution of EDPs. It can calibrate a multivariate distribution that describes the joint distribution of EDPs if raw EDP data is available. Users can control the type of each marginal distribution, apply truncation limits to consider collapses, and censor part of the data to consider detection limits in their analysis. Alternatively, pelicun can use raw EDP data as-is without resampling from a fitted distribution.

   * **Define the performance model of an asset.** The fragility and consequence functions from the first two editions of FEMA P58 and the HAZUS earthquake and hurricane wind and storm surge models for buildings are provided with pelicun. This facilitates the creation of performance models without having to collect and provide component descriptions and corresponding fragility and consequence functions. An auto-population interface encourages researchers to develop and share rulesets that automate the performance-model definition based on the available building information. Example scripts for such auto-population are also provided with the tool.

   * **Simulate asset damage.** Given the EDP samples, and the performance model, pelicun efficiently simulates the damages in each component of the asset and identifies the proportion of realizations that resulted in collapse.

   * **Estimate the consequences of damage.** Using information about collapse and component damages, the following consequences can be estimated with pelicun: repair cost and time, unsafe placarding (red tag), injuries of various severity and fatalities.
