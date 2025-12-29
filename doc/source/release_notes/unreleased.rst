.. _changes_unreleased:

==========
Unreleased
==========

Added
-----

**Multi-Hazard Regional Simulation**: Added a framework to support multiple,
configurable hazards beyond the original earthquake-only focus.

- Add support for the **Hazus Hurricane Wind** damage and loss methodology.
- Introduce a new end-to-end integration test for the hurricane wind
  scenario, complete with a dedicated, hazard-specific pytest fixture.

**Building Inventory Filter**: Introduce a new feature to run simulations on a
specific subset of assets.

- Add a `filter` key to the configuration file to select buildings by ID and
  ID ranges (e.g., "1, 5-10").
- Add a comprehensive, parametrized test suite to validate all filter
  scenarios, including error handling.

**Enhanced Regional Simulation Testing**: Introduce the first comprehensive
integration test for the `regional_sim` tool to establish a testing baseline.

Changed
-------

**Regional Simulation Workflow**: Major refactoring of the `regional_sim`
script for improved flexibility and robustness.

- Generalize the Intensity Measure (IM) handling to be dynamically driven by
  the configuration file, removing all hardcoded "PGA" logic.
- Rearchitect the loss assessment logic into a conditional framework, with a
  dedicated path for complex Hazus Earthquake models and an efficient,
  1-to-1 mapping path for other methods.
- Reorganize the output stage to save results sequentially, improving
  robustness against failures in later-stage calculations.
- Update logic to upsample demand realizations when the requested sample size
  is larger than the available data.

**NNR (Nearest Neighbor Resampling) Tool**: Significant enhancements to the
`NNR` function for improved performance and functionality.

- The "expected value" mode can now operate on each realization of a 3D
  input array independently.
- Improve performance by vectorizing the 2D expected value calculation.
- Make the number of nearest neighbors a configurable parameter.

Removed
-------


Fixed
-----

**Code Quality and Documentation**: Continued compliance with ruff and ensured
all tests pass to guarantee code quality and future maintainability.

- Add full type hinting and a comprehensive docstring to the `NNR` function.