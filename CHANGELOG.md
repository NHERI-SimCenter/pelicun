# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [3.8.0] - 2025-09-15

This release introduces a major new feature with the integrated Damage and Loss Model Library (DLML) module, modernizes the CI/CD pipeline, and enhances the testing infrastructure. The changes provide users with seamless access to the model library while maintaining backward compatibility.

### Added
- **DLML (Damage and Loss Model Library) Module**: A new comprehensive module for efficient management of the Damage and Loss Model Library has been added to Pelicun:
    - **Automatic Data Initialization**: DLML data is now automatically initialized on first import, providing seamless access to model library.
    - **CLI Integration**: The DLML module is integrated with Pelicun's command-line interface for easy management
    - **Comprehensive Testing**: Added extensive unit tests and integration tests for the DLML functionality
- **Enhanced Test Infrastructure**: Significant improvements to the testing framework:
    - **Network Isolation**: Added network isolation capabilities for more robust testing
    - **Improved Test Configuration**: Enhanced test configuration management with better fixtures and utilities

### Changed
- **CI/CD Pipeline Modernization**: Major update to the continuous integration and deployment system:
    - **Unified CI Workflow**: Consolidated multiple separate workflow files into a single comprehensive CI pipeline
    - **GitHub API Integration**: Implemented robust GitHub API authentication for better repository management
- **Dependencies**: Added `requests` as a project dependency to support new DLML functionality

### Removed
- **DLML Submodule**: Removed the DLML Git submodule in favor of the new integrated DLML module approach
- **Legacy CI Workflows**: Removed several workflow files that were consolidated into the new unified CI pipeline:
    - `deploy_to_pypi.yaml`
    - `docs_check.yaml` 
    - `format_check.yml`
    - `lint.yml`
    - `spell_check.yml`
    - `tests.yml`

### Fixed
- **Code Quality**: Applied comprehensive fixes for code quality improvements throughout the codebase
- **CI Workflow Issues**: Resolved various issues in the CI workflow configuration for more reliable builds and testing

---

## [3.7.1] - 2025-08-12

This is a patch release that enables flood loss simulation through SimCenter's backend and addresses a few bugs.

---

## [3.7.0] - 2025-08-07

### Added
- **Regional Simulation Tool**: A new regional simulation capability has been added to Pelicun as a temporary solution while the integrated feature is being developed. The tool currently supports Hazus EQ simulation using lifeline facility fragilities and includes:
    - **Batch Processing**: Memory-efficient sequential and parallel batch processing using joblib for handling large-scale regional simulations
    - **Progress Reporting**: Time and progress information output to track simulation status
    - **Command-line Interface**: A new CLI system with `pelicun` as an installable command and `regional_sim` as the first available subcommand
- **Installable CLI**: Pelicun now provides a modern command-line interface through a `console_scripts` entry point. The new `pelicun/cli.py` module serves as a subcommand dispatcher, making the tool easier to install and use from the command line
- **Case-insensitive Unit Handling**: Implemented case-insensitive unit label handling in input data processing, improving robustness when processing data with varying unit label capitalization

### Changed
- **Modernized Packaging**: The project has been modernized to align with current Python packaging standards (PEP 621):
    - All static package metadata (version, dependencies, etc.) has been migrated from `setup.py` to `pyproject.toml`
    - `pyproject.toml` is now the single source of truth for package configuration
    - Version number source has been moved to `pyproject.toml`
    - `setup.py` file is now minimal, reserved for future dynamic build logic

### Removed
- **Deprecated Tools**: Removed `HDF_to_CSV` from tools as it is no longer used

### Fixed
- **Pandas Deprecation Warning**: Fixed a minor issue in `loss_model` to avoid pandas deprecation warnings, ensuring compatibility with newer pandas versions
- **R2D Configuration Handling**: Applied bugfixes to handle advanced configurations from R2D
- **Documentation Build Issues**: Resolved various Sphinx warnings and build errors to ensure clean documentation generation

---

## [3.6.1] - 2025-05-16

This is a patch release that addresses a bug in the Damage and Loss Model Library configuration files.

### Changed
- **Damage and Loss Model Library**: Updated the DLML submodule to the latest version to incorporate bug fixes and ensure users have access to properly functioning configuration files for all supported assessment methodologies

---

## [3.6.0] - 2025-05-16

### Added
- **Enhanced DLML Resource Management**: Introduced a new flexible resource management system that separates method references from specific data files. This change makes it easier to extend the component library and maintain consistency across different component types. The update includes:
    - **Method-based Resource Access**: Users can now reference damage and loss methods (e.g., `FEMA P-58`) separately from specific data files (e.g., `fragility.csv`), providing greater flexibility in utilizing various files from the Damage and Loss Model Library
    - **Expanded Resource Categories**: Added more specific resource categories including separate paths for Hazus Hurricane variants (coupled, original)
    - **Sequential Auto-population Support**: Extended the auto module to support running multiple configuration (auto-population) scripts in sequence for complex assessment workflows, such as hurricane wind and storm surge
- **Enhanced Hurricane Assessment Support**: Added comprehensive support for different Hazus Hurricane assessment variants:
    - **Hazus Hurricane Wind - Buildings - Coupled**: Support for wind assessments with coupled damage and loss models using fitted lognormal fragility functions
    - **Hazus Hurricane Wind - Buildings - Original**: Support for original Hazus hurricane methodology using decoupled, multilinear CDF damage and multilinear loss models
    - **Hazus Hurricane Storm Surge - Buildings**: Support for storm surge damage assessments that leverage Hazus flood loss models (i.e., depth-to-damage functions)

### Changed
- **Separated Assessment Processes**: Introduced separate damage processes for Hazus EQ Lifeline and Regular Building assessments, improving modularity and allowing for more targeted infrastructure-specific evaluations
- **DLML Resource Path Structure**: Restructured the damage and loss model library resource path system:
    - Replaced generic labels (e.g., `HAZUS MH EQ IM`) with descriptive method names (e.g., `Hazus Earthquake - Buildings`)
    - Updated infrastructure network naming from generic `Water` and `Power` to specific `Potable Water` and `Electric Power`
    - Streamlined resource file structure to reduce redundancy and improve maintainability
- **Backwards Compatibility**: Implemented comprehensive backwards compatibility support:
    - Legacy filename references (e.g., `damage_DB_FEMA_P58_2nd.csv`) are automatically converted to the new method+filename format
    - Users receive informative warnings when using deprecated filename formats
    - Compatibility checks consolidated in the `substitute_default_path` method in `file_io` for clarity

### Removed
- **Legacy Auto-population Files**: Removed three large legacy auto-population files that are no longer needed:
    - `Hazus_Earthquake_CSM.py`
    - `Hazus_Earthquake_IM.py`
    - `Hazus_Earthquake_Story.py`

### Fixed
- **DL Configuration Bug**: Fixed a bug in DL_calculation where `dl_method` and corresponding folder paths were being queried unnecessarily, even when DL configuration was already present in the config file
- **DLML Submodule Updates**: Updated the DLML submodule references to ensure access to the latest damage and loss model data and maintain synchronization with the updated resource path structure

---

## [3.5.1] - 2025-02-24

This is a patch release that addresses issues in the Hazus damage process and updates the Damage and Loss Model Library to the latest version.

### Changed
- **Damage and Loss Model Library**: Updated the DLML submodule and resource paths to point to Hazus v6.1, ensuring users have access to the most current fragility and consequence models for seismic building assessments

### Fixed
- **Hazus Damage Process**: Added proper handling for irreparable damage states to ensure that all damage states are properly accounted for in Hazus earthquake assessments

---

## [3.5.0] - 2025-02-04

### Added
- **Power Network Infrastructure Support**: Added comprehensive support for power network assessments including:
    - **Power Fragility Models**: New fragility models for power substations, circuits, and generation facilities
    - **Power Auto-population**: Extended auto-population scripts to recognize power network assets and map them to appropriate archetypes
    - **Power Units**: Added support for power-related units including 'kW' and 'MW' with proper unit conversions
- **Wind Demand Recognition**: Added wind demands to the list of recognized Engineering Demand Parameters (EDPs), expanding multi-hazard assessment capabilities
- **Line Force Units**: Added support for line forces in the default units configuration, enabling proper conversion and handling of forces distributed along a line (e.g., N/m, kN/m). This enhancement improves the modeling of structural components where distributed loads are relevant
- **Submodule Integration**: Transitioned to using submodule resource files for better resource management and version control of damage and loss model libraries
- **Enhanced Loss Processing**: Improved loss aggregation and processing capabilities:
    - **Comprehensive Loss Results**: Enhanced DL_calculation to save additional information about component-level losses, making it easier to trace the contribution of individual components to the total loss
    - **Grouped Loss Statistics**: Added capability to save grouped loss statistics when running regional simulations

### Changed
- **Input Validation Flexibility**: Added more flexibility in input validation by allowing `Distribution` and `Theta_1` to be missing for CollapseFragility and Replacement consequences, with sensible defaults applied automatically
- **GitHub Workflows**: Updated GitHub Actions workflows to:
    - Initialize submodules at checkout for proper resource file handling
    - Use trusted publisher approach for PyPI deployment
    - Improve documentation build and deployment processes
- **Code Quality Improvements**: Enhanced code quality through:
    - Comprehensive type annotations and checking throughout the codebase
    - Enhanced random variable parameter handling when sampling from distributions
- **DLML Resource Management**: Updated Damage and Loss Model Library (DLML) resource paths and submodule pointers to the most recent version

### Fixed
- **Bug Fixes**: Resolved several critical issues:
    - **Irreparable Damage Handling**: Fixed bug ensuring no component losses are counted on top of replacement when damage is irreparable
    - **Exponentiation Operator**: Fixed incorrect exponentiation operator usage in power network damage model autopopulation script
    - **Anchor Property Validation**: Improved anchor property parser to throw appropriate errors for invalid inputs
    - **Bridge Modeling**: Fixed marginal case handling for single span bridges longer than 150m
    - **Water Network Fragility Database Updates**: Updated water network damage databases with corrected fragility parameters
    - **Custom Model Directory Handling**: Improved `custom_model_dir` functionality to only check for custom model directories when the path actually contains the placeholder `CustomDLDataFolder`
    - **Archetype ID Corrections**: Fixed typos in archetype IDs, including correcting "EP.C" to "EP.G" and ensuring proper capitalization in base ID strings for power network assets
- **Input Validation**: Replaced checking for falsy values with explicit checks for `None` to prevent unexpected behavior with zero values
- **Documentation and Formatting**: Fixed various documentation issues, indentation bugs in YAML files, and applied consistent code formatting using ruff
- **Unit Test Fixes**: Resolved unit test issues and formatting problems to ensure reliable testing

---

## [3.4.0] - 2024-11-27

### Added
- **Documentation pages**: Documentation for pelicun 3 is back online. The documentation includes guides for users and developers as well as an auto-generated API reference. A lineup of examples is planned to be part of the documentation, highlighting specific features, including the new ones listed in this section
- **Consequence scaling**: This feature can be used to apply scaling factors to consequence and loss functions for specific decision variables, component types, locations and directions. This can make it easier to examine several different consequence scaling schemes without the need to repeat all calculations or write extensive custom code
- **Capacity scaling**: This feature can be used to modify the median of normal or lognormal fragility functions of specific components. Medians can be scaled by a factor or shifted by adding or subtracting a value. This can make it easier to use fragility functions that are a function of specific asset features
- **Loss functions**: Loss functions are used to estimate losses directly from the demands. The damage and loss models were substantially restructured to facilitate the use of loss functions
- **Loss combinations**: Loss combinations allow for the combination of two types of losses using a multi-dimensional lookup table. For example, independently calculated losses from wind and flood can be combined to produce a single loss estimate considering both demands
- **Utility demand**: Utility demands are compound demands calculated using a mathematical expression involving other demands. Practical examples include the application of a mathematical expression on a demand before using it to estimate damage, or combining multiple demands with a multivariate expression to generate a combined demand. Such utility demands can be used to implement those multidimensional fragility models that utilize a single, one-dimensional distribution that is defined through a combination of multiple input variables
- **Normal distribution with standard deviation**: Added two new variants of "normal" in `uq.py`: `normal_COV` and `normal_STD`. Since the variance of the default normal random variables is currently defined via the coefficient of variation, the new `normal_STD` is required to define a normal random variable with zero mean. `normal_COV` is treated the same way as the default `normal`
- **Weibull random variable**: Added a Weibull random variable class in `uq.py`
- **New `DL_calculation.py` input file options**: We expanded configuration options in the `DL_calculation.py` input file specification. Specifically, we added `CustomDLDataFolder` for specifying additional user-defined components
- **Warnings in red**: Added support for colored outputs. In execution environments that support colored outputs, warnings are now shown in red
- Code base related additions, which are not directly implementing new features but are nonetheless enhancing robustness, include the following:
    - pelicun-specific warnings with the option to disable them
    - a JSON schema for the input file used to configure simulations through `DL_calculation.py`
    - addition of type hints in the entire code base
    - addition of slots in all classes, preventing on-the-fly definition of new attributes which is prone to bugs

### Changed
- Updated random variable class names in `uq.py`
- Extensive code refactoring for improved organization and to support the new features. We made a good-faith effort to maintain backwards compatibility, and issue helpful warnings to assist migration to the new syntax
- Moved most of the code in DL_calculation.py to assessment.py and created an assessment class
- Migrated to Ruff for linting and code formatting. Began using mypy for type checking and codespell for spell checking

### Deprecated
- `.bldg_repair` attribute was renamed to `.loss`
- `.repair` had also been used in the past, please use `.loss` instead
- In the damage and loss model library, `fragility_DB` was renamed to `damage_DB` and `bldg_repair_DB` was renamed to `loss_repair_DB`
- `load_damage_model` was renamed to `load_model_parameters` and the syntax has changed. Please see the applicable warning message when using `load_damage_model` for the updated syntax
- `{damage model}.sample` was deprecated in favor of `{damage model}.ds_model.sample`
- The `DMG-` flag in the loss_map index is no longer required
- `BldgRepair` column is deprecated in favor of `Repair`
- `load_model` -> `load_model_parameters`
- `{loss model}.save_sample` -> `{loss model}.ds_model.save_sample`. The same applies to `load_sample`

### Removed
- No features were removed in this version
- We suspended the use of flake8 and pylint after adopting the use of ruff

### Fixed
- Fixed a bug affecting the random variable classes, where the anchor random variable was not being correctly set
- Enforced a value of 1.0 for non-directional multipliers for HAZUS analyses
- Fixed bug in demand cloning: Previously demand unit data were being left unmodified during demand cloning operations, leading to missing values
- Reviewed and improved docstrings in the entire code base

---

## [3.3.0] - 2024-03-29

### Added
- **Location-specific damage processes**: This new feature is useful when you want damage to a component type to induce damage in another component type at the same location only. For example, damaged water pipes on a specific story can trigger damage in floor covering only on that specific story. Location-matching is performed automatically without you having to define component pairs for every location using the following syntax: `'1_CMP.A-LOC', {'DS1': 'CMP.B_DS1'}`, where `DS1` of `CMP.A` at each location triggers `DS1` of `CMP.B` at the same location
- **New `custom_model_dir` argument for `DL_calculation.py`**: This argument allows users to prepare custom damage and loss model files in a folder and pass the path to that folder to an auto-population script through `DL_calculation.py`. Within the auto-population script, they can reference only the name of the files in that folder. This provides portability for simulations that use custom models and auto population, such as some of the advanced regional simulations in SimCenter's R2D Tool
- **Extended Hazus EQ auto population scripts to include water networks**: Automatically recognize water network assets and map them to archetypes from the Hazus Earthquake technical manual
- **Introduce `convert_units` function**: Provide streamlined unit conversion using the pre-defined library of units in Pelicun. Allows you to convert a variable from one unit to another using a single line of simple code, such as: `converted_height = pelicun.base.convert_units(raw_height, unit='m', to_unit='ft')`. While not as powerful as some of the Python packages dedicated to unit conversion (e.g., Pint), we believe the convenience this function provides for commonly used units justifies its use in several cases

### Changed
- **Code Structure Improvements**: Split `model.py` into subcomponents. The `model.py` file was too large and its contents were easy to refactor into separate modules. Each model type has its own python file now and they are stored under the model folder
- **RandomVariable Class Refactoring**: Split the `RandomVariable` class into specific classes. It seems more straightforward to grow the list of supported random variables by having a specific class for each kind of RV. We split the existing large RandomVariable class in uq.py leveraging inheritance to minimize redundant code
- **Automatic code formatting**: Further improve consistency in coding style by using black to review and format the code when needed
- **Removed `bldg` from variable and class names**: Following the changes mentioned earlier, we dropped bldg from labels where the functionality is no longer limited to buildings
- **Introduced `calibrated` attribute for demand model**: This new attribute will allow users to check if a model has already been calibrated to the provided empirical data
- **Version ceiling was raised for pandas**: Supporting version 2.0 and above up until 3.0

### Deprecated
- **Remove `Bldg` from repair settings label in DL configuration file**: Following the changes above, we dropped `Bldg` from `BldgRepair` when defining settings for repair consequence simulation in a configuration file. The previous version (i.e., `BldgRepair`) will keep working until the next major release, but we encourage everyone to adopt the new approach and simply use the `Repair` keyword there

### Removed
- **BREAKING:** Remove `bldg` from repair consequence output filenames: The increasing scope of Pelicun now covers simulations for transportation and water networks. Hence, labeling repair consequence outputs as if they were limited to buildings no longer seems appropriate. The bldg label was dropped from the following files: `DV_bldg_repair_sample`, `DV_bldg_repair_stats`, `DV_bldg_repair_grp`, `DV_bldg_repair_grp_stats`, `DV_bldg_repair_agg`, `DV_bldg_repair_agg_stats`

---

## [3.2.0] - 2024-02-27

### Added
- **New multilinear CDF Random Variable**: Allows using the multilinear approximation of any CDF in the tool
- **Capacity adjustment**: Allows adjusting (scaling or shifting) default capacities (i.e., fragility curves) with factors specific to each Performance Group
- **Support for multiple definitions of the same component at the same location-direction**: This feature facilitates adding components with different block sizes to the same floor or defining multiple tenants on the same floor, each with their own set of components
- **Support for cloning demands**: Taking a provided demand dataset, creating a copy and considering it as another demand. For example, you can provide results of seismic response in the X direction and automatically prepare a copy of them to represent results in the Y direction
- **Environmental Impact models**: Models for estimating Environmental Impact (i.e., embodied carbon and energy) of earthquake damage as per FEMA P-58 are included in the DL Model Library and available in this release
- **"ListAllDamageStates" option**: Allows you to print a comprehensive list of all possible damage states for all components in the columns of the DMG output file. This can make parsing the output easier but increases file size. By default, this option is turned off and only damage states that affect at least one block are printed
- **Damage and Loss Model Library**: A collection of parameters and metadata for damage and loss models for performance based engineering. The library is available and updated regularly in the DB_DamageAndLoss GitHub Repository. This and future releases of Pelicun have the latest version of the library at the time of their release bundled with them
- **DL_calculation tool enhancements**:
    - Support for combination of built-in and user-defined databases for damage and loss models
    - Results are now also provided in standard SimCenter JSON format besides the existing CSV tables. You can specify the preferred format in the configuration file under Output/Format. The default file format is still CSV
    - Support running calculations for only a subset of available consequence types

### Changed
- **Comprehensive testing suite**: Added a comprehensive suite of more than 140 unit tests that cover more than 93% of the codebase. Tests are automatically executed after every commit using GitHub Actions and coverage is monitored through Codecov.io. Badges at the top of the Readme show the status of tests and coverage
- **Code quality improvements**: Completed a review of the entire codebase using `flake8` and `pylint` to ensure PEP8 compliance. The corresponding changes yielded code that is easier to read and use
- **Enhanced error handling**: Several error and warning messages added to provide more meaningful information in the log file when something goes wrong in a simulation
- **Updated dependencies**: Update dependencies to more recent versions

### Removed
- **BREAKING:** Unit information is included in every output file. If you parse Pelicun outputs and did not anticipate a Unit entry, your parser might need an update
- **BREAKING:** Decision variable types in the repair consequence outputs are named using CamelCase rather than all capitals to be consistent with other parts of the codebase. For example, we use "Cost" instead of "COST". This might affect post-processing scripts
- **BREAKING:** For clarity, "ea" units were replaced with "unitless" where appropriate. There should be no practical difference between the calculations due to this change. Interstory drift ratio demand types are one example
- **BREAKING:** Weighted component block assignment is no longer supported. We recommend using more versatile multiple component definitions to achieve the same effect
- **BREAKING:** Damage functions (i.e., assign quantity of damage as a function of demand) are no longer supported. We recommend using the new multilinear CDF feature to develop theoretically equivalent but more efficient models

---

## [3.1.0] - 2022-09-30

### Added
- **Assessment-specific calculation settings**: Calculation settings are now assessment-specific. This allows you to use more than one assessments in an interactive calculation and each will have its own set of options, including log files
- **Standalone uq module**: The uq module was decoupled from the others to enable standalone uq calculations that work without having an active assessment
- **Redesigned DL_calculation.py script**: A completely redesigned DL_calculation.py script that provides decoupled demand, damage, and loss assessment and more flexibility when setting up each of those when pelicun is used with a configuration file in a larger workflow
- **New examples**: Two new examples that use the DL_calculation.py script and a json configuration file were added to the example folder
- **Interactive calculation example**: A new example that demonstrates a detailed interactive calculation in a Jupyter notebook was added to the DesignSafe project: https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3411v5. This project will be extended with additional examples in the future

### Changed
- **External unit conversion factors**: Unit conversion factors moved to an external file (settings/default_units) to make it easier to add new units to the list. This also allows redefining the internal units through a complete replacement of the factors. The internal units continue to follow the SI system
- **Coding style improvements**: Substantial improvements in coding style using flake8 and pylint to monitor and help enforce PEP8
- **Performance improvements**: Several performance improvements made calculations more efficient, especially for large problems, such as regional assessments or tall buildings investigated using the FEMA P-58 methodology
- **Updated Python version**: Update recommended Python version to 3.10 and other dependencies to more recent versions

### Fixed
- Several bugfixes and a large number of minor changes that make the engine more robust and easier to use

---

## [3.0.0] - 2021-12-31

### Added
- **New assessment module**: Introduced to replace control module:
    - Provides a high-level access to models and their methods
    - Integrates all types of assessments into a uniform approach
    - Most of the methods from the earlier control module were moved to the model module
- **Integrated damage and loss calculation**: Across all methods and components:
    - This includes phenomena such as collapse, including various collapse modes, and irreparable damage
    - Cascading damages and other interdependencies between various components can be introduced using a damage process file
    - Losses can be driven by damages or demands. The former supports the conventional damage->consequence function approach, while the latter supports the use of vulnerability functions. These can be combined within the same analysis, if needed
    - The same loss component can be driven by multiple types of damages. For example, replacement can be triggered by either collapse or irreparable damage
- **Options in configuration file**: Introduced Options in the configuration file and in the base module:
    - These options handle settings that concern pelicun behavior
    - General preferences that might affect multiple assessment models
    - Settings that users would not want to change frequently
    - Default settings are provided in a default_config.json file. These can be overridden by providing any of the prescribed keys with a user-defined value assigned to them in the configuration file for an analysis

### Changed
- **BREAKING:** Architecture redesigned: The architecture was redesigned to better support interactive calculation and provide a low-level integration across all supported methods. This is the first release with the new architecture. Frequent updates are planned to provide additional examples, tests, and bugfixes in the next few months
- **BREAKING:** Decoupled demand, damage, and loss calculations:
    - Fragility functions and consequence functions are stored in separate files. Added new methods to the db module to prepare the corresponding data files and re-generated such data for FEMA P58 and Hazus earthquake assessments. Hazus hurricane data will be added in a future release
    - Decoupling removed a large amount of redundant data from supporting databases and made the use of HDF and json files for such data unnecessary. All data are stored in easy-to-read csv files
    - Assessment workflows can include all three steps (i.e., demand, damage, and loss) or only one or two steps. For example, damage estimates from one analysis can drive loss calculations in another one

---

## [2.6.0] - 2021-08-06

### Added
- **Extended EDP support**: Support EDPs with more than 3 characters and/or a variable in their name. For example, `SA_1.0` or `SA_T1`
- **Normal distribution fitting**: Support fitting normal distribution to raw EDP data (lognormal was already available)

### Changed
- **Accessible settings**: Extract key settings to base.py to make them more accessible for users

### Fixed
- Minor bug fixes mostly related to hurricane storm surge assessment

---

## [2.5.0] - 2020-12-31

### Added
- **Extended uq module support**:
    - More efficient sampling, especially when most of the random variables in the model are either independent or perfectly correlated
    - More accurate and more efficient fitting of multivariate probability distributions to raw EDP data
    - Arbitrary marginals (beyond the basic Normal and Lognormal) for joint distributions
    - Latin Hypercube Sampling
- **External auto-population scripts**: Introduce external auto-population scripts and provide an example for hurricane assessments
- **HDF to CSV conversion tool**: Add a script to help users convert HDF files to CSV (HDF_to_CSV.py under tools)

### Changed
- **Standardized attribute names**: Use unique and standardized attribute names in the input files
- **Updated dependencies**: Migrate to the latest version of Python, numpy, scipy, and pandas (see setup.py for required minimum versions of those tools)

### Fixed
- **Bug fixes and minor improvements** to support user needs:
    - Add 1.2 scale factor for EDPs controlling non-directional Fragility Groups
    - Remove dependency on scipy's truncnorm function to avoid long computation times due to a bug in recent scipy versions

---

## [2.1.1] - 2020-06-30

### Added
- **Aggregate DL data to HDF5**: Aggregate DL data from JSON files to HDF5 files. This greatly reduces the number of files and makes it easier to share databases
- **Log file support**: Add log file to pelicun that records every important calculation detail and warnings
- **New EDP types**: Add 8 new EDP types: RID, PMD, SA, SV, SD, PGD, DWD, RDR
- **Extended auto-population**: Extend auto-population logic with solutions for HAZUS EQ assessments

### Changed
- **Performance improvements**: Significant performance improvements in EDP fitting, damage and loss calculations, and output file saving
- **Python version support**: Drop support for Python 2.x and add support for Python 3.8

### Fixed
- Several bug fixes and minor improvements to support user needs

---

## [2.0.0] - 2019-10-15

### Added
- **FEMA P58 2nd edition DL data**: Added FEMA P58 2nd edition DL data
- **Standard CSV EDP support**: Support for EDP inputs in standard csv format
- **SimCenter output format**: Added a function that produces SimCenter DM and DV json output files
- **Enhanced optimization**: Added a differential evolution algorithm to the EDP fitting function to do a better job at finding the global optimum
- **Multi-stripe analysis support**: Enhanced DL_calculation.py to handle multi-stripe analysis (significant contributions by Joanna Zou):
    - Recognize stripe_ID and occurrence rate in BIM/EVENT file
    - Fit a collapse fragility function to empirical collapse probabilities
    - Perform loss assessment for each stripe independently and produce corresponding outputs

### Changed
- **BREAKING:** Migrated to latest versions: Migrated to the latest version of Python, numpy, scipy, and pandas. See setup.py for required minimum versions of those tools
- **BREAKING:** Python 2.x is no longer supported
- **Improved DL input structure** to:
    - Make it easier to define complex performance models
    - Make input files easier to read
    - Support custom, non-PACT units for component quantities
    - Support different component quantities on every floor
- **Updated FEMA P58 DL data**: Updated FEMA P58 DL data to use ea for equipment instead of units such as KV, CF, AP, TN

---

## [1.1.0] - 2019-02-06

### Added
- **Common JSON format**: Converted to a common JSON format for FEMA P58 and HAZUS Damage and Loss data
- **HAZUS-style loss assessment**: Added component-assembly-based (HAZUS-style) loss assessment methodology for earthquake