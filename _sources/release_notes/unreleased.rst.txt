.. _changes_unreleased:

==========
Unreleased
==========

Added
-----

**DLML (Damage and Loss Model Library) Module**: A new comprehensive module for efficient management of the Damage and Loss Model Library has been added to Pelicun:

- **Automatic Data Initialization**: DLML data is now automatically initialized on import, providing seamless access to model libraries
- **CLI Integration**: The DLML module is integrated with Pelicun's command-line interface for easy management
- **Comprehensive Testing**: Added extensive unit tests and integration tests for the DLML functionality

**Enhanced Test Infrastructure**: Significant improvements to the testing framework:

- **Network Isolation**: Added network isolation capabilities for more reliable testing
- **Improved Test Configuration**: Enhanced test configuration management with better fixtures and utilities

Changed
-------

**CI/CD Pipeline Modernization**: Major update to the continuous integration and deployment system:

- **Unified CI Workflow**: Consolidated multiple separate workflow files into a single comprehensive CI pipeline
- **GitHub API Integration**: Implemented robust GitHub API authentication for better repository management


**Dependencies**: Added `requests` as a project dependency to support new DLML functionality.

Removed
-------

**DLML Submodule**: Removed the DLML Git submodule in favor of the new integrated DLML module approach.

**Legacy CI Workflows**: Removed several workflow files that were consolidated into the new unified CI pipeline:

- `deploy_to_pypi.yaml`
- `docs_check.yaml` 
- `format_check.yml`
- `lint.yml`
- `spell_check.yml`
- `tests.yml`

Fixed
-----

**Code Quality and Security**: Applied comprehensive fixes for code quality and security improvements throughout the codebase.

**CI Workflow Issues**: Resolved various issues in the CI workflow configuration for more reliable builds and testing.