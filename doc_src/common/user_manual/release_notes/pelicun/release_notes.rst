.. _lbl-release_notes:

*************
Release Notes
*************

----
v3.0
----

-
-
-

----
v2.1
----

- Aggregate DL data from JSON files to HDF5 files. This greatly reduces the number of files and makes it easier to share databases.
- Significant performance improvements in EDP fitting, damage and loss calculations, and output file saving.
- Add log file to pelicun that records every important calculation detail and warnings.
- Add 8 new EDP types: RID, PMD, SA, SV, SD, PGD, DWD, RDR.
- Drop support for Python 2.x and add support for Python 3.8.
- Extend auto-population logic with solutions for HAZUS EQ assessments.
- Several bug fixes and minor improvements to support user needs.

----
v2.0
----

- migrated to the latest version of Python, numpy, scipy, and pandas
- see setup.py for required minimum versions of those tools
- Python 2.x is no longer supported
- improve DL input structure to
    - make it easier to define complex performance models
    - make input files easier to read
    - support custom, non-PACT units for component quantities
    - support different component quantities on every floor
- updated FEMA P58 DL data to use ea for equipment instead of units such as KV, CF, AP, TN
- add FEMA P58 2nd edition DL data
- support EDP inputs in standard csv format
- add a function that produces SimCenter DM and DV json output files
- add a differential evolution algorithm to the EDP fitting function to do a better job at finding the global optimum
- enhance DL_calculation.py to handle multi-stripe analysis (significant contributions by Joanna Zou):
    - recognize stripe_ID and occurrence rate in BIM/EVENT file
    - fit a collapse fragility function to empirical collapse probabilities
    - perform loss assessment for each stripe independently and produce corresponding outputs

----
v1.2
----

- support for HAZUS hurricane wind damage and loss assessment
- add HAZUS hurricane DL data for wooden houses
- move DL resources inside the pelicun folder so that they come with pelicun when it is pip installed
- add various options for EDP fitting and collapse probability estimation
- improved the way warning messages are printed to make them more useful

----
v1.1
----

- converted to a common JSON format for FEMA P58 and HAZUS Damage and Loss data
- added component-assembly-based (HAZUS-style) loss assessment methodology for earthquake

----
v1.0
----

- initial release