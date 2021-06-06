.. _lbl-testbed_AC_preface:

********
Preface
********

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
loss models.

**Case 2**: Users who seek to develop building inventories beyond Atlantic County, NJ will benefit from a 
deeper understanding of the techniques, rulesets and scripts used to generate this building inventory. 
In addition to the explanations that follow, this documentation is accompanied by detailed rulesets used 
for building and attribute assignment 
(`SimCenter Hurricane Testbed: Inventory Documentation <https://berkeley.box.com/s/7acln70veux1ebz2epxmlidn49kyjqgv>`_), 
as well as their implementation as Python scripts (`auto_HU_NJ.py <https://github.com/kuanshi/pelicun/blob/master/pelicun/resources/auto_population/auto_HU_NJ.py>`_). 
These provide templates that such users can potentially 
refine, extend and replicate this testbed’s process for Building Inventory generation beyond the current 
application in Atlantic County.

Before running this testbed, users are advised to review the **Computational Resources Requirements** to ensure 
their hardware meets minimum specifications and to understand how to properly estimate the HPC resources 
necessary to execute these testbeds and factors influencing the run time.
