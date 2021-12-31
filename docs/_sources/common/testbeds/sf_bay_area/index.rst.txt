.. _lbl-testbed_SF_Bay_Area:

*****************
San Francisco, CA
*****************

**Acknowledgements**

The SimCenter was financially supported by the National Science Foundation under Grant CMMI-1612843. Any opinions,
findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect
the views of the National Science Foundation. We would like to acknowledge (1) the contributions and collaboration with many
faculty, post-doctoral researchers, students and staff who have contributed to the SimCenter’s work, and (2) the support and
close collaboration with DesignSafe, which facilitates access to high-performance computing and information technologies for
SimCenter tools.

.. list-table:: Documentation Version History
   :name: doc_version
   :header-rows: 1
   :align: center

   * - Version
     - Release Date
     - Update
   * - 1.0
     - 05/2021
     - Initial release



**Preface**

This documentation is intended to introduce the implementation of the SimCenter’s earthquake 
regional loss modeling workflow in the context of San Francisco, California. 

The San Francisco Bay Area encompasses three large cities, San Francisco, Oakland and San Jose, which together with
the surrounding communities have a population of about 7.7 million people. The seismic hazard in the San Francisco Bay
Area is dominated by the San Andreas and Hayward faults that straddle the region. The San Andreas Fault is located just to
the west of San Francisco and is capable of a magnitude Mw 8 earthquake, such as the Mw 7.8 event that occurred in 1906.
The Hayward Fault, which runs up the eastern edge of the Bay Area, is capable of a magnitude Mw 7 earthquake, such as
the Mw 6.7 event that occurred in 1868. Recently, the USGS completed an earthquake scenario study for a Mw 7 event on
the Hayward fault, which provided an opportunity to contrast existing regional assessment methods with the SimCenter’s
computational workflow.

The SimCenter workflow tools were applied to assess the performance of 1.84 M buildings in the San Francisco Bay
Area due to a Mw 7.0 earthquake rupture on the Hayward fault. Probabilistic assessment of earthquake consequences with
building (parcel) level resolution at this scale is only feasible using high performance computing resources, which is facilitated by
SimCenter’s regional Workflow for Hazard and Loss Estimation (rWHALE, [Elhadded19]_). The testbed focuses on
assessment of response, damage, repair costs, and repair times for all 1.84 M buildings in the simulation.

.. toctree-filt::
   :maxdepth: 1

   overview
   asset_description
   hazard_characterization
   asset_representation
   response_simulation
   damage_and_loss
   uncertainty_quantification
   sample_results
   example_outputs
   future_refinements
   feedback_request

.. [Elhadded19]
   Elhaddad, W., McKenna, F., Rynge, M., Lowe, J. B., Wang, C., and Zsarnoczay, A. (2019). 
   NHERI-SimCenter/WorkflowRegionalEarthquake: rWHALE (Version v1.1.0). http://doi.org/10.5281/zenodo.2554610