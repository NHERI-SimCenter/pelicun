.. _changes_unreleased:

==========
Unreleased
==========

Added
-----

**Documentation pages**: Documentation for pelicun 3 is back online. The documentation includes guides for users and developers as well as an auto-generated API reference. A lineup of examples is planned to be featured in the documentation, highlighting specific features, including the new features listed in this section.

**Consequence scaling**: This feature can be used to apply scaling factors to estimated losses based on specific decision variables, component types, locations and directions. This can make it easier to examining several different consequence scaling schemes without the need to repeat all calculations or write extensive custom code.

**Loss functions**: Loss functions are used to estimate losses directly from the demands. The damage and loss models were substantially restructured to facilitate the addition of loss functions.

**Loss combinations**: Loss combinations allow for the combination of two types of losses using a multi-dimensional lookup table. For example, independently calculated wind and flooding losses can be combined to produce a single loss estimate considering their combined occurrence.

**Utility demand**: Utility demands are compound demands calculated using a mathematical expression involving other demands. Application examples could include the application of a mathematical expression to a demand column before using it to estimate damage, or combining multiple columns with an arbitrary multivariate expression to generate a combined demand.

**Normal with standard deviation**: Added a few variants of "normal" in ``uq.py``: ``normal_COV`` and ``normal_STD``. Since the variance of the default normal random variables is currently defined via the coefficient of variation, ``normal_STD`` enables the definition of normal random variables with zero mean. ``normal_COV`` is treated in the same way as the default ``normal``.

**Weibull random variable**: Added a Weibull random variable class in ``uq.py``.

**New ``DL_calculation.py`` input file options**: We expanded configuration options in the ``DL_calculation.py`` input file specification. Specifically, we added ``CustomDLDataFolder`` for specifying additional user-defined components.

**Warnings in red**: Added support for colored outputs. Warnings are now shown in red, conditioned on what the execution environment supports.

Code base related additions, which are not directly implementing new features but are nonetheless enhancing robustness, include the following:
- pelicun-specific warnings with the option to disable them
- a JSON schema for the ``DL_calculation.py`` input file
- addition of type hints in the entire code base
- addition of slots in all classes, preventing on-the-fly definition of new attributes which is prone to bugs

Changed
-------

- Updated random variable class names in ``uq.py``.
- Extensive code refactoring for improved organization and to support the added features. We made a good-faith effort to maintain backwards compatibility, and issue helpful warnings to assist migration to the new syntax.
- Moved code from DL_calculation.py to assessment.py and created an assessment class.
- Migrated to Ruff for linting and code formatting. Began using mypy for type checking and codespell for spell checking.

Deprecated
----------

- ``.bldg_repair`` attribute was renamed to ``.loss``
- ``.repair`` had also been used in the past, please use ``.loss`` instead.
- In the damage and loss model library, ``fragility_DB`` was renamed to ``damage_DB`` and ``bldg_repair_DB`` was renamed to ``loss_repair_DB``.
- ``load_damage_model`` was renamed to ``load_model_parameters`` and the syntax has changed. Please see the applicable warning message when using ``load_damage_model`` for the updated syntax.
- ``{damage model}.sample`` was deprecated in favor of ``{damage model}.ds_model.sample``.
- The ``DMG-`` flag in the loss_map index is no longer required.
- ``BldgRepair`` column is deprecated in favor of ``Repair``.
- ``load_model`` -> ``load_model_parameters``
- ``{loss model}.save_sample`` -> ``'{loss model}.ds_model.save_sample``. The same applies to ``load_sample``.

Removed
-------

- No features were removed in this version.
- We suspended the use of flake8 and pylint after adopting the use of ruff.

Fixed
-----

- Fixed a bug affecting the random variable classes, where the anchor random variable was not being correctly set.
- Enforced a value of 1.0 for non-directional multipliers for HAZUS analyses.
- Fixed bug in demand cloning: Previously demand unit data were being left unmodified during demand cloning operations, leading to missing values.
- Reviewed and improved docstrings in the entire code base.
