.. _lbl-testbed_AC_overview:

********
Overview
********

This testbed for regional hurricane risk assessment of Atlantic County, New Jersey under multiple hazards adopts 
an approach consistent with that developed for earthquake hazards. Its intent is to demonstrate the computational 
scaffolding upon which community developers can progressively contribute refinements that increase the fidelity 
and capacities of the backend regional resilience assessment workflow. This documentation specifically demonstrates 
the process of: (1) asset description, (2) hazard characterization, (3) asset representation, (4) damage and loss 
modeling for the Atlantic County inventory. This inventory is partitioned into three different versions to provide 
users greater flexibility in balancing computational demands with their research interests. Sample results are 
presented to verify the workflow and demonstrate its usage.

Rationale
===========

This testbed builds upon existing relationships between SimCenter personnel and the State of New Jersey through 
the NJcoast project ([KijewskiCorrea19]_, [NJCoast20]_), which made available various inventory data and high-fidelity 
characterizations of wind, storm surge and wave action. Notably, the State of New Jersey has made sizable 
investments in open data and resilience planning tools in recent years, enabling a demonstration of how to leverage 
these data for inventory development. 

The selection of Atlantic County, NJ and specifically Atlantic City and its 
surrounding municipalities offers a well-defined metro area with a blend of low-rise commercial (1-3 stories), 
industrial, high-rise hotels/casinos (over 20 stories), and single/multi-family residences. Thus a single testbed 
encompasses two extremes in building typologies with known vulnerabilities to wind: wood frame single-family homes 
and tall flexible structures more susceptible to dynamic wind effects. From the perspective of hazard exposure, 
this region is also characterized by beachfront communities exposed to storm surge and breaking waves on the 
ocean-facing coastline and back bay and riverine flooding. This blend of residential communities and more urban 
central business districts frequented by tourists moreover enables evaluation of resilience across different 
communities with differing profiles. As the State of New Jersey has actually taken control of Atlantic City to 
reimagine its role as an economic hub for the state, this testbed can help inform how future planning decisions 
should consider hurricane risk.

Capabilities and Supported Hazards
====================================

The testbed supports the transition from census-block-level loss projections to asset-level projections that 
assess the damage to individual buildings under multiple hurricane-induced hazards: wind and storm surge. Water 
penetration due to the breach of building envelopes and/or wind-borne debris impact are also captured in the 
damage and loss modeling, though the physics of these phenomena themselves are not explicitly modeled. 
Similarly, other hydrologic hazards related to accumulated rainfall, inland flooding, overland flow and 
riverline flooding, are not capture.
Supported building classes include residential, commercial and industrial construction, critical facilities, 
and manufactured homes, constructed of wood, steel, reinforced concrete and masonry. The adoption of HAZUS 
loss estimation frameworks constrains the current testbed to buildings 6 stories and under and only the 
building classes currently supported by HAZUS ([FEMA18a]_, [FEMA18b]_).

Current Implementation
========================

For the initial implementation of the backend workflow, asset description adopts an augmented parcel approach 
that enriches tax assessor data. High-fidelity computational simulations are employed to simulate the wind and 
surge hazards, characterized by two intensity measures: the Peak Wind Speed (PWS) and Peak Water Depth (PWD), 
for specific scenarios. In lieu of a structural analysis model, assets are assigned attributes associated with 
various HAZUS-consistent building classifications. The adoption of HAZUS damage and loss assessment methodology 
for hurricane and flood thus enables these intensity measures to be related to probabilities of damage and loss, 
based on the building class and assigned attributes.

Available Inventories
========================

Three different building inventories have been developed for the Atlantic County testbed and can be accessed on DesignSafe.

**Atlantic County Inventory**: Full inventory of 100,721 buildings in the 23 municipalities of Atlantic County, 
described based on a variety of data sources (:numref:`fig-buildingClassACI`). The buildings in this inventory are exposed to wind 
only OR the combination of wind and floodplain hazards.

.. _fig-buildingClassACI:

.. figure:: figure/BuildingClass_atlanticall.png
   :align: center
   :figclass: align-center
   :width: 800

   Geospatial visualization of subclasses of buildings in Atlantic County Inventory.

**Flood-Exposed Inventory**: This subset of the Atlantic County inventory is confined to 32,828 buildings in 
FEMA Special Flood Hazard Areas (SFHAs) (:numref:`fig-buildingClassFEI`), as identified by the New Jersey Department of Environmental 
Protection (NJDEP). This includes all buildings in (or within 200-foot buffer of) the 1% annual chance (AC) 
floodplain, as defined by FEMA Flood Insurance Rate Maps (FIRMs). The buildings in this inventory are exposed 
to the combination of wind and floodplain hazards, and includes some of the most populated municipalities in the 
county: Atlantic City, Margate City, and Ventor City which contribute to about 50% of the entire building inventory 
in Atlantic County.

.. _fig-buildingClassFEI:

.. figure:: figure/new_inventory_map.png
   :align: center
   :figclass: align-center
   :width: 1200

   Geospatial visualization of subclasses of buildings in Flood-Exposed Inventory.

**Exploration Inventory**: A subset of 1000 buildings drawn from the Flood-Exposed Inventory intended to provide 
a less computationally demanding implementation for new users or for those wishing to test the development of new 
contributions to the workflow (:numref:`fig-buildingClassEI`). This inventory encompasses the five coastal municipalities 
experiencing the most damage under the synthetic storm scenario described later in :ref:`lbl-testbed_AC_hazard_characterization_synthetic`.
From each of these municipalities, properties are randomly sampled, proportional to the total number of buildings 
in that municipalitiy and ensuring that the distribution of construction material of buildings in the sample is 
representative of the underlying distribution for the full population. The buildings in this inventory are exposed to the combination of 
wind and floodplain hazards.

.. _fig-buildingClassEI:

.. figure:: figure/new_inventory_map_expl.png
   :align: center
   :figclass: align-center
   :width: 700

   Geospatial visualization of subclasses of buildings in Flood-Exposed Inventory.

The following figures summarize characteristics of these inventories, including distribution by municipality 
(:numref:`fig-distAssetMunicipality`), by year built (:numref:`fig-distBuiltYear`), by occupancy (:numref:`fig-occupancyType`), 
by number of stories (:numref:`fig-numStory`) and by primary construction material (:numref:`fig-constrMaterial`). 
Notably, these inventories are typified by older vintages of construction (79% of Atlantic County buildings were constructed 
before 1980), with a dominance of low-rise (1-2 stories), residential, wood frame construction (approximately 
90% of Atlantic County buildings). Steel and reinforced concrete construction is more prevalent in downtown 
Atlantic City. 

.. _fig-distAssetMunicipality:

.. figure:: figure/num_building_city_allset.png
   :align: center
   :figclass: align-center
   :width: 1000

   Distribution of number of buildings by municipality.

.. _fig-distBuiltYear:

.. figure:: figure/built_year_allset.png
   :align: center
   :figclass: align-center
   :figwidth: 1200

   Distribution of year built for buildings.

.. _fig-occupancyType:

.. figure:: figure/occupancy_type_allset.png
   :align: center
   :figclass: align-center
   :figwidth: 1200

   Distribution of occupancy types.

.. _fig-numStory:

.. figure:: figure/story_number_allset.png
   :align: center
   :figclass: align-center
   :figwidth: 1200

   Distribution of total story numbers for buildings.

.. _fig-constrMaterial:

.. figure:: figure/building_type_allset.png
   :align: center
   :figclass: align-center
   :figwidth: 1200

   Distribution of primary construction material types.

.. [FEMA18a]
   FEMA (2018), HAZUS – Multi-hazard Loss Estimation Methodology 2.1, Hurricane Model Technical Manual, Federal Emergency Management Agency, Washington D.C., 718p.

.. [FEMA18b]
   FEMA (2018), HAZUS – Multi-hazard Loss Estimation Methodology 2.1, Flood Model Technical Manual, Federal Emergency Management Agency, Washington D.C., 569p.

.. [KijewskiCorrea19]
   Kijewski-Correa, T., Taflanidis, A., Vardeman, C., Sweet, J., Zhang, J., Snaiki, R., ... & Kennedy, A. (2020). Geospatial environments for hurricane risk assessment: applications to situational awareness and resilience planning in New Jersey. Frontiers in Built Environment, 6, 549106.

.. [NJCoast20]
   NJ Coast (2020), Storm Hazard Projection Tool, NJ Coast, https://njcoast.us/resources-shp/