.. _feature_overview:

Overview of pelicun features
----------------------------

.. admonition:: Coming soon.

   This section is under construction.

.. _fo_saving:

Saving/loading samples
......................

All demand, asset, damage, and loss samples can be either computed from other inputs or directly loaded form previously computed and saved samples.

.. _fo_logging:

Logging support
...............

Pelicun produces detailed log files that can be used to document the execution of an assessment as well as information on the host machine and the execution environment.
These logs can be useful for debugging purposes.
Pelicun emits detailed warnings whenever appropriate, notifying the user of potentially problematic or inconsistent inputs, evaluation settings, or deprecated syntax.

.. _fo_uq:

Uncertainty quantification
..........................

Damage and loss estimation is inherently uncertain and treated as a stochastic problem.
Uncertainty quantification lies at the core of all computations in pelicun.
Pelicun supports a variety of common parametric univariate random variable distributions.
With the help of random variable registries, it also supports multivariate distributions, joined with Gaussian copula.

.. _fo_assessment_types:

Assessment types
................

Pelicun supports scenario-based assessments. That is, losses conditioned on a specific value of an Intensity Measure (IM).

..
   TODO: add links pointing to a glossary/definition index of terms.

.. note::

   Support for time-based assessments is currently in progress.

Demand simulation
.................

.. _fo_calibration:

Model calibration
^^^^^^^^^^^^^^^^^

.. _fo_sampling:

Sampling methods
^^^^^^^^^^^^^^^^

.. _fo_pidrid:

RID|PID inference
^^^^^^^^^^^^^^^^^

.. _fo_sample_expansion:

Sample expansion
^^^^^^^^^^^^^^^^

.. _fo_demand_cloning:

Demand cloning
^^^^^^^^^^^^^^

.. _fo_custom_demand_types:

Custom demand types
^^^^^^^^^^^^^^^^^^^

Pelicun identifies demand types using the controlled vocabulary of the Damage and Loss Model Library (e.g., ``Peak Interstory Drift Ratio`` with the acronym ``PID``) and uses the unit type registered for each entry to assign demand units automatically.
When an assessment uses a demand type that is not part of that vocabulary, register it with the ``CustomDemandTypes`` entry in the ``Options`` section of the configuration:

.. code-block:: json

   {
     "Options": {
       "CustomDemandTypes": {
         "Story Torsion Ratio": {
           "Acronym": "STR",
           "UnitType": "unitless"
         }
       }
     }
   }

When using the Python API, pass the contents of ``Options`` directly to ``Assessment``, e.g., ``Assessment({"CustomDemandTypes": {...}})``.

Each entry maps a verbose demand name to the acronym used in demand files and model parameters, and to the unit type the demand is measured in.
When a demand file arrives without a ``Units`` row, pelicun assigns the unit matching the registered unit type; automatic unit assignment is implemented for the ``acceleration``, ``speed``, ``displacement``, ``unitless``, and ``rotation`` (measured in radians, no conversion) unit types.
The model-library vocabulary also defines ``force``, ``force_per_length``, and ``pressure``; pelicun recognizes these but does not implement automatic unit assignment for them yet, and raises a clear error if a demand provided without units needs one of them.
Providing an explicit ``Units`` row in the demand file bypasses automatic unit assignment entirely, so demands with such unit types can still be used that way.
Demand types that are neither part of the default vocabulary nor registered are assumed to be in base units and trigger a warning.
Custom entries only apply to the assessment configured with them, and they may reuse a default demand name to override its default acronym or unit type.

Damage estimation
.................

.. _fo_damage_process:

Damage processes
^^^^^^^^^^^^^^^^

Loss estimation
.................

.. _fo_loss_maps:

Loss maps
^^^^^^^^^

.. _fo_active_dvs:

Active decision variables
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _fo_consequence_scaling:

Consequence scaling
^^^^^^^^^^^^^^^^^^^

.. _fo_loss_aggregation:

Loss aggregation
^^^^^^^^^^^^^^^^

Also talk about replacement thresholds here.

.. _fo_loss_functions:

Loss functions
^^^^^^^^^^^^^^
.. _fo_cli:

Command-line support
....................

Pelicun can be ran from the command line.
Installing the package enables the ``pelicun`` entry point, which points to ``tools/DL_calculation.py``.
``DL_calculation.py`` is a script that conducts a performance evaluation using command-line inputs.
Some of those inputs are paths to required input files, including a JSON file that provides most evaluation options.

..
   TODO: point to an example, and index the example in the by-feature grouping.

.. _fo_autopop:

Input file auto-population
^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible for the JSON input file to be auto-populated (extended to include more entries) using either default or user-defined auto-population scripts.
Auto-population scripts are referenced either with a filesystem path or with the ``PelicunDefault/<method>/pelicun_config.py`` syntax, which resolves to the corresponding script in the installed Damage and Loss Model Library.

A loaded script is cached and reused when the same script is requested again under the same identifier, as in the per-building loop of a regional simulation.
Module-level state in the script therefore persists across those calls: one-time setup work and cross-asset memoization are supported patterns, but scripts must not assume a fresh re-initialization on every call.
In parallel regional runs, each worker process holds its own copy of the module state.
Per-worker memoization works as expected, but accumulating global results across all buildings in module-level variables does not; derive such aggregates by post-processing the outputs instead.

..
   TODO: Why is this useful? Why would a user want to do this?

Standalone tools
................

.. _fo_convert_units:

Unit conversion
^^^^^^^^^^^^^^^

.. _fo_fit:

Fit distribution to sample or percentiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _fo_rvs:

Random variable classes
^^^^^^^^^^^^^^^^^^^^^^^

Feature overview and examples
.............................

A series of examples, organized by feature, demonstrate the capabilities supported by pelicun.

.. button-link:: ../examples/index.html
    :color: primary
    :shadow:

    Visit the examples

