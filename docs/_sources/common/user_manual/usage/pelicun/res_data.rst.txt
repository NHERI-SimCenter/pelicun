.. _lbl-in_response:

******************
Response Data File
******************

The response data file provides the demand data for the assessment. Demands are either Engineering Demand Parameters from a response simulation or Intensity Measures from a hazard assessment. The types of demands provided must match the types of demands requested by the fragility functions.

The data is expected in CSV file format. The first line of the file is a header that identifies the types of EDPs in the file. pelicun uses SimCenter's naming convention for EDPs (see :numref:`fig-res_data`). There are four pieces of information provided in the name of each EDP:

  :event_id:
    Identifies the event in a multi-event scenario. `1` is used for single-event studies.

  :EDP_type:
    Specifies the type of EDP provided. pelicun currently supports the following EDP types in response data files:

    :PID:
      Peak Interstory Drift Ratio

    :RID:
      Residual Interstory Drift Ratio

    :PRD:
      Peak Roof Drift Ratio

    :PFA:
      Peak Floor Acceleration

    :PFV:
      Peak Floor Velocity

    :PGA:
      Peak Ground Acceleration

    :PGV:
      Peak Ground Velocity

    :PGD:
      Permanent Ground Deformation


  :location:
    Specifies the location where the EDP was measured. In buildings, locations typically correspond to floors with `0` being the ground for accelerations and `1` being the first floor for drift ratios.

  :direction:
    Specifies the direction of the EDP. In buildings, typically the two horizontal directions are used and they are labeled as `1` and `2`. Any numbering logic can be used here, but the labels must match the ones provided under `Components` in the Configuration File (see :numref:`lbl-in_config`)


Each row in the file provides EDPs from one simulation. EDP units are assumed to follow the unit specification in the Configuration File.

.. _fig-res_data:

.. figure:: figures/res_data.png
   :align: center
   :figclass: align-center
   :figwidth: 600 px

   Sample response data file and EDP naming convention.

