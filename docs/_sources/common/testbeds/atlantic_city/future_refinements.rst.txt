.. _lbl-testbed_AC_future_refinements:

****************************************
Opportunity Areas for Future Refinements
****************************************

This testbed and its backing workflows should be viewed as a sandbox and scaffold, 
respectively, within which the community can explore different scenarios and develop 
new capabilities. The following list itemizes some areas where contribution from the 
community is especially welcome: 

1. The current use of HAZUS for damage and loss modeling limits the buildings under 
   consideration to 6 stories and under. Thus the user community is invited to support 
   the extension of PELICUN to buildings taller than 6 stories.

2. Extensions to taller buildings will require further expansion of the building 
   inventory’s fields to capture attributes relevant to those classes of construction. 
   Methods to mine relevant data from imagery or other third party data sources, e.g., 
   Emporis, will provide welcome extensions to the Asset Description approaches described herein.

3. The testbed would benefit from validation studies that include ground truthing the 
   generated inventories as well as the projected damage and loss. Those with access to 
   damage data, particularly in areas affected significantly by Superstorm Sandy, may use 
   the testbed to conduct such validation studies. The methodologies used to construct the 
   inventory for Atlantic County can easily be repeated in more northern counties of the state, 
   where damage was more significant, as all the flood-exposed inventory data and hazard descriptions 
   available through the SimCenter have coverage over the entire state. The rulesets are similarly 
   applicable statewide.

4. Currently the workflow calculates damage and monetary loss; coupling the workflow with other 
   social, societal or economic models, will provide opportunities to more robustly explore the 
   impacts of hazard events, mitigation investments and development decisions on community resilience. 
   The creation of techniques to automatically scrape and fuse data from the social, human and behavioral 
   sciences will ensure those datasets can be generated in a manner similar to those used for the 
   building inventory itself. 

5. A number of approximations and assumptions were made in the generation of the building inventory 
   (see :ref:`lbl-testbed_AC_asset_description`). There is considerable opportunity to expand, enrich and improve upon 
   methodologies for automated inventory generation, particularly to generate other attributes necessary 
   as the workflow advances toward component level damage quantification.

6. The SimCenter aspires to incorporate increasing levels of fidelity in its characterization of hazards, 
   modeling of buildings (to support application of pressures/loads and structural analysis), and ability 
   to describe damage at the component level, with fault-trees that capture cascading damage sequences 
   resulting from breaches of the building envelope. Thus there is considerable need for community 
   research contributions such as libraries of fragilities, archetype building models, and catalogs 
   of high-fidelity hazard simulations (hindcasts of historical events or synthetic storm data). 
   This is especially the case of coastal hazards. With aspirations to replicate this workflow in other 
   regions, these contributions need not be confined to New Jersey/Atlantic Coast. Their integration into 
   the testbed can demonstrate how the contributions of individual researchers can aggregate to evaluate 
   impact on much larger scales.

7. The testbed does not explicitly treat the effects of debris (wind-borne or surge-transported), nor 
   are any of the flow fields translated into pressures acting on individual buildings to enable the 
   execution of structural analyses. Extending the workflow to tall buildings creates opportunities for 
   response simulation to estimate drifts or other EDPs for more dynamically sensitive structures under 
   wind. However, similar opportunities to translate geospatially varying hazards into loads on a given 
   structure can be seized to increase the workflow's overall fidelity.


.. note::

   Community members who wish to contribute to the hurricane testbed in any of the above (or other) areas can 
   express their interest by completing this `survey <https://docs.google.com/forms/d/e/1FAIpQLSdVnnqYvDfpYyFunQSbNTkqqWR9WlzL-VjV_Pe9A21o1Iw4Aw/viewform>`_.


