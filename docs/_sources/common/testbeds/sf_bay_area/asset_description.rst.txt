.. _lbl-testbed_SF_asset_description:

*****************
Asset Description
*****************

This study used a parcel-level inventory of buildings in the Bay
Area that was developed by UrbanSim ([Waddell02]_) using
public resources such as the City and County of San Francisco’s
data portal ([DataSF20]_) and tax assessor databases. 

The raw database which includes 1.8 M San Francisco buildings were shared 
in the collaboration with UrbanSim. The UrbanSim database includes two files:

1. Buildings File: a CSV file that contains building properties including total floor 
   area, number of stories, year built, building occupancy and parcel ID. 
2. Parcels File: a CSV file including the latitude and longitude of each building defined 
   by the parcel ID.

A parsing application was built to collect the mentioned building properties and from the 
UrbanSim building files. The occupancy ID (in integer) was used to infer the occupancy 
type and replacement cost per unit area (:numref:`occupancy_map`). The buildings with missing or invalid occupancy 
ID, the building was mapped to the default occupancy type (i.e., residential) with average area of buildings 
in the inventory.

.. csv-table:: Mapping rules for building occupancy type and replacement cost.
   :name: occupancy_map
   :file: data/occupancy_map.csv
   :header-rows: 1
   :align: center

The structure types were also mapped from the building occupancy types along with the year built and number of stories. 
Unless the building was mapped to a single structure type, the structure type is considered random, in which case the mapped 
structure types are equally likely. :numref:`structure_map` summarizes the structure type mapping.

.. csv-table:: Mapping rules for structure type.
   :name: structure_map
   :file: data/structure_map.csv
   :header-rows: 1
   :align: center

The available information about location
and building geometry were further refined by merging the UrbanSim
database with the publicly available Microsoft Building Footprint
data ([Microsoft20]_) for the testbed area. These data were
used to populate two additional attributes, replacement cost and
structure type, based on a ruleset that considers local design
practice and real estate pricing. For further details about the
database and ruleset see [Elhadded19]_.


.. [Waddell02]
   Waddell, P. (2002). UrbanSim: Modeling Urban Development for Land Use, Transportation, and Environmental Planning. J. Am. Planning Assoc. 68, 297–314. 
   doi: 10.1080/01944360208976274

.. [DataSF20]
   DataSF (2020). Building and Infrastructure Databases. San Francisco, SF: DataSF.

.. [Microsoft20]
   Microsoft (2020). Microsoft Building Footprint Database for the United States. Washington, DC. 
   https://www.microsoft.com/en-us/maps/building-footprints.

.. [Elhadded19]
   Elhaddad, W., McKenna, F., Rynge, M., Lowe, J. B., Wang, C., and Zsarnoczay, A. (2019). 
   NHERI-SimCenter/WorkflowRegionalEarthquake: rWHALE (Version v1.1.0). http://doi.org/10.5281/zenodo.2554610