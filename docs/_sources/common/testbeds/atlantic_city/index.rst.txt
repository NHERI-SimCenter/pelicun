.. _lbl-testbed_Atlantic_City:

*******************
Atlantic County, NJ
*******************


**Acknowledgements**

A number of members of the SimCenter Team assisted with this testbed’s development and its documentation: 
Adam Zsarnoczay, Ajay Harish, Barbaros Cetiner, Charles Wang, Claudio Perez, Joanna Zou, Kuanshi Zhong, Stevan Gavrilovic, and Wael Elhaddad. 
The testbed conceptualization was guided by Tracy Tracy Kijewski-Correa (University of Notre Dame), 
with implementation supervised by Frank McKenna (University of California Berkeley). 
Additional guidance was offered by Greg Deierlein (Stanford University), Andrew Kennedy 
(University of Notre Dame), and Matt Schoettler (University of California Berkeley). 

The Hazard Characterization leveraged models, software and simulations executed by the following groups, 
whose collaboration is greatly appreciated:

The NJcoast project, with storm surge surrogate model developed by Alexandros Taflanidis (University of Notre Dame) 
in collaboration with Andrew Kennedy (University of Notre Dame) and wind field model contributed by Teng Wu (University at Buffalo)
Superstorm Sandy ADCIRC hindcast results for storm surge provided by Joannes Westerink (University of Notre Dame) 
and wind fields made available by Peter Vickery (Applied Research Associates).

Additional information required for Asset Description and Asset Representation was made possible through the 
ongoing collaboration between the University of Notre dame and the New Jersey Department of Community Affairs (NJ DCA) 
through the NJcoast project. NJ DCA’s Keith Henderson’s sustained support and collaboration is greatly appreciated.

.. list-table:: Documentation Version History
   :name: doc_version
   :header-rows: 1
   :align: center

   * - Version
     - Release Date
     - Update
   * - 1.0
     - 04/2021
     - Initial release



**Preface**

This documentation is intended to introduce the implementation of the SimCenter’s hurricane 
regional loss modeling workflow in the context of Atlantic City (Atlantic County), New Jersey. 
While certain aspects of the workflow are unchanged in a given application context, this 
testbed specifically demonstrates how building inventories can be constructed through 
automated processes that fuse different data sources to enrich parcel data, using SimCenter 
applications and heuristic rulesets grounded in local codes/standards and normative 
construction practices. Given the significance of the building inventory generation for this 
testbed, this documentation was written in response to two primary audiences/use cases:

**Case 1**: End users who wish to use the testbed to explore specific research questions such as:
1. the impact of different hurricane scenarios beyond those provided herein
2. the potential benefits of various mitigation efforts (changing select attribute assignments and/or damage/loss descriptions)
3. the benefits of more refined damage/loss models, particularly for coastal hazards

Such individuals may not wish to generate their own inventories, but require some background in order 
to meaningfully interpret results. This documentation will enhance their understanding of the various 
assumptions made in generating these inventories and assigning the attributes required for the adopted 
loss models. Use Case 1 generally requires skill sets in :ref:`lbl-testbed_AC_hazard_characterization` and 
:ref:`lbl-testbed_AC_damage_and_loss`.

**Case 2**: Users who seek to develop building inventories beyond Atlantic County, NJ will benefit from a 
deeper understanding of the techniques, rulesets and scripts used to generate this building inventory. 
In addition to the explanations that follow, this documentation is accompanied by detailed rulesets used 
for building and attribute assignment 
(`SimCenter Hurricane Testbed: Inventory Documentation <https://berkeley.box.com/s/7acln70veux1ebz2epxmlidn49kyjqgv>`_), 
as well as their implementation as Python scripts (`auto_HU_NJ.py <https://github.com/kuanshi/pelicun/blob/master/pelicun/resources/auto_population/auto_HU_NJ.py>`_). 
These provide templates that such users can potentially 
refine, extend and replicate this testbed’s process for Building Inventory generation beyond the current 
application in Atlantic County. Use Case 2 generally requires skill sets in :ref:`lbl-testbed_AC_asset_description`, 
:ref:`lbl-testbed_AC_asset_representation`, and :ref:`lbl-testbed_AC_damage_and_loss`.

Before running this testbed, users are advised to review the **Computational Resources Requirements** to ensure 
their hardware meets minimum specifications and to understand how to properly estimate the HPC resources 
necessary to execute these testbeds and factors influencing the run time.

.. _lbl-testbeds-manual:

.. toctree-filt::
   :maxdepth: 1

   acknowledgements
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
   best_practices
   pws_conversion
   feedback_request

