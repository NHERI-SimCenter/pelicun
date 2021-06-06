.. _lbl-testbed_AC_asset_representation:

********************
Asset Representation
********************

This section discusses the translation of asset descriptions into representations 
of structures suitable for simulation within workflow, in this case consistent with 
the HAZUS description of building classes and associated attributes, which becomes 
the default data schema. Thus the description of assets below is organized according 
to the HAZUS conventions for classifying buildings and organizing damage and loss data 
according to attributes associated with those building classes.

The following discussion will reference a number of rulesets developed for this testbed 
to enable various assignments of these HAZUS building classes and corresponding attributes. 
Details of these rulesets are available to users in one of two forms: 

1. Rulest definition tables (PDFs) curated in DesignSafe that include additional documentation 
   justifying the proposed rule, with provenance information for any sources engaged in that 
   rule’s development.
2. Scripts (in Python) curated in GitHub that implement the ruleset’s logic for this testbed.

Each section provides a table linking the relevant Tables and Scripts. Note that as well 
that all of the rulesets introduced herein are tiered, initiating by assigning all assets 
a default value for its building classification or a given attribute based on the primary 
rule. This ensures that every asset receives a HAZUS building class and related attribute 
assignments, regardless of data sparsity. 

.. _lbl-testbed_AC_asset_representation_building_classification:

Building Classifications
==========================

HAZUS classifies buildings based on a more nuanced interpretation of Occupancy Class 
(see building inventory field **OccupancyClass**) based on other attributes of relevance 
for a given hazard type.

For wind losses, HAZUS groups buildings into 5 main types by primary building material/construction 
mode (wood, masonry, concrete, steel, manufactured home). Buildings must then be sub-classified 
into one of 55 corresponding HAZUS building classes (**HazusClass-W**) based on characteristics 
such as occupancy, number of stories and footprint size, using rulesets that call upon various 
fields in the building inventory. The HAZUS building classifications for wind losses are listed 
in :numref:`bldg_class`, and the corresponding rulesets (PDFs and Python scripts) are cross-referenced 
later in :numref:`addinfo_ruleset`. Note that while rulesets were developed for marginally and non-engineered 
building classes in HAZUS, these classes are not used in the current implementation of this testbed.

.. csv-table:: HAZUS building classification for wind loss assessment.
   :name: bldg_class
   :file: data/bldg_class_new.csv
   :header-rows: 1
   :align: center

For flood losses, HAZUS groups buildings into one of 32 classifications based on the building use 
and other features. A separate ruleset was developed to assign buildings into one of these classes 
associated with inundation by water (**HazusClass-IN**). The HAZUS building classifications for flood 
losses are listed in :numref:`flood_bldg_class`, and the corresponding rulesets (PDFs and Python scripts) 
are cross-referenced later in :numref:`addinfo_ruleset`.

.. csv-table:: HAZUS building classification for flood loss assessment.
   :name: flood_bldg_class
   :file: data/flood_bldg_class_new.csv
   :header-rows: 1
   :align: center

For wave-induced losses, HAZUS groups buildings into one of 10 classifications based on the building 
use, construction material and number of stories. A separate ruleset was developed to assign buildings 
into one of these classes associated with losses driven by wave action (**HazusClass-WA**). The HAZUS 
building classifications for wave-induced losses are listed in :numref:`wave_bldg_class`, and the corresponding 
rulesets (PDFs and Python scripts) are cross-referenced in :numref:`addinfo_ruleset`.

.. csv-table:: HAZUS building classification for wave-induced loss assessment.
   :name: wave_bldg_class
   :file: data/wave_bldg_class_new.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning HAZUS building class
   :name: addinfo_ruleset
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - Building Class Rulesets - Wind
     - `HAZUS Building Class Rulesets - Wind.pdf <https://berkeley.box.com/s/602imclyqm1ohvfqliqro0bzq4v0wdj3>`_
     - :download:`WindClassRulesets <data/WindClassRulesets.py>`
   * - Building Class Rulesets - Flood
     - `HAZUS Building Class Rulesets - Flood.pdf <https://berkeley.box.com/s/6gqyu66b9d74ctto6x1k74i95q2wksrb>`_
     - :download:`FloodClassRulesets <data/FloodClassRulesets.py>`
   * - Building Class Rulesets - Wave
     - `HAZUS Building Class Rulesets - Wave.pdf <https://berkeley.box.com/s/sr8p05yp230qwcyodeh1w39wxw6qurp6>`_
     - To be released.

Building Attributes
======================

Within each of these building classes, e.g., wood single-family homes 1-2+ stories, the HAZUS Hurricane 
Technical Manual (HHTM) further differentiates buildings based on asset attributes and the hazard type 
(e.g., wind vs. flood) for the purpose of loss estimation. These attributes define key features of the 
load path and components (e.g., roof shape, secondary water resistance, roof deck attachment, roof-wall 
connection, shutters, garage), and the number of attributes necessary to describe a given building varies. 

As these attributes are beyond what is typically encompassed in a building inventory, this testbed developed 
and implemented a library of rulesets to infer the HAZUS-required attributes based upon the fields available 
in the Building Inventory, legacy building codes in New Jersey, local construction practices/norms, surveys 
capturing owner-driven mitigation actions (e.g., [Javeline19]_) and market/industry data. 
Where possible, the rulesets are time-evolving, considering the age of construction to determine the governing 
code edition and availability of specific mitigation measures in the market. Though reliant on engineering 
judgement and historical data availability, each rule provides detailed notes cross-referencing the various 
documents and practices that governed that era of construction and thus informed the ruleset formation. 
In cases where engineering judgement was required, rules were assigned based on what was understood to be 
the most common construction practice. In cases where that was not clear, the ruleset assigned the most 
vulnerable configuration for a more conservative approach to loss estimation. :numref:`wind_bldg_attri` 
and :numref:`flood_bldg_attri` list the attributes for the wind and flood loss assessments. 

.. csv-table:: Building attributes for wind loss assessment.
   :name: wind_bldg_attri
   :file: data/wind_bldg_attri.csv
   :header-rows: 1
   :align: center

.. csv-table:: Building attributes for flood loss assessment.
   :name: flood_bldg_attri
   :file: data/flood_bldg_attri.csv
   :header-rows: 1
   :align: center

Note that rulesets for assigning wind loss attributes call upon two meta-variables relevant to wind losses 
for any building: “Hazard Prone Region” and “Wind Borne Debris,” which are assigned based the design wind 
speed at the asset location (Building Inventory field “DSWII”) and the flood zone (building inventory field 
**FloodZone**), per New Jersey code. These rules used to assign these meta-variables are provided in 
:numref:`addinfo_ruleset_metavar`. Also note that the roof shape (building inventory field **RoofShape**), 
derived from aerial imagery, and terrain roughness (building inventory field **Terrain**), derived from 
Land Use Land Cover data, are also attributes required by the HAZUS wind loss model. As these were already 
assigned in the :ref:`lbl-testbed_AC_asset_description`, they are not discussed again herein.

.. list-table:: Additional details for rulesets for meta-variables in wind loss attribute assignment in HAZUS
   :name: addinfo_ruleset_metavar
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - Attribute Assignment - Wind (Meta-Variable)
     - `Hazus Building Attribute Rulesets - Wind - Meta-Variables.pdf <https://berkeley.box.com/s/l4vdnfoakq8xsv4rmj64x4m2kxqritu7>`_
     - :download:`WindMetaVarRulesets <data/WindMetaVarRulesets.py>`

Finally, all of the rulesets used to assign attributes include a default value that can be updated based 
on available data, ensuring that each asset receives all the attribute assignments necessary to identify 
the appropriate Hazus fragility description. The following sections summarize the rulesets used for 
attribute assignments for specific classes of buildings. Additional attributes assigned to assets are 
discussed in the following subsections, organized by hazard and building class, where applicable.

Wind Loss Attributes for Wood Buildings
------------------------------------------

The wind loss model in HAZUS classifies wooden buildings into five building classes:
   
1. two single family homes (WSF1 and WSF2) and
2. three for multi-unit homes (WMUH1, WMUH2, and WMUH3)

Their required attributes for wind loss modeling, the possible entries (values, terms) that can be 
assigned for those attributes, and the basis for the ruleset developed to make that assignment are 
summarized in :numref:`wsf_attri` and :numref:`wmuh_attri`. Note that these rulesets were developed 
to reflect the likely attributes based on the year of construction and the code editions and 
construction norms at that time. The corresponding time-evolving rulesets (PDFs and Python scripts) 
are cross-referenced in :numref:`addinfo_ruleset_wood`.

.. csv-table:: Additional HAZUS attributes assigned for wood single family (WSF) homes: wind losses.
   :name: wsf_attri
   :file: data/wsf_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for wood multi-unit home (WMUH): wind losses.
   :name: wmuh_attri
   :file: data/wmuh_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning wind loss attributes for wood buildings
   :name: addinfo_ruleset_wood
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Wind (WSF1-2)
     - `Hazus Building Attribute Rulesets - Wind - WSF1-2.pdf <https://berkeley.box.com/s/nod73v7shtj9x7ox7xw7b7nvmrs3e8oc>`_
     - :download:`WindWSFRulesets <data/WindWSFRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (WMUH1-3)
     - `Hazus Building Attribute Rulesets - Wind - WMUH1-3.pdf <https://berkeley.box.com/s/4v7405rit2u475daorayy9w6ssuezbz9>`_
     - :download:`WindWMUHRulesets <data/WindWMUHRulesets.py>`

Taking the attribute Second Water Resistance (SWR) as an example, the SWR attribute is assigned by 
a series of time-evolving rules calling upon four fields in the building inventory: year built, 
roof shape, roof slope, and average temperature in January. :numref:`swr_attri` provides the 
detailed rules that map these four variables to the Second Water Resistance (SWR) attribute. 
This example demonstrates an instance where the attribute is assigned as a random variable, 
based on the fact that secondary water resistance is not required by code, though surveys 
of homeowners in hurricane-prone areas can be used to infer how many may have voluntarily 
adopted this mitigation practice. 

.. csv-table:: Ruleset for determining the Second Water Resistance attribute for WSF homes.
   :name: swr_attri
   :file: data/example_wood_ruleset.csv
   :header-rows: 1
   :align: center


Wind Loss Attributes for Masonry Buildings
------------------------------------------------

The masonry buildings have 14 building classes: Their key attributes that influence the fragility 
functions are listed in :numref:`msf_attri`, :numref:`mmuh_attri`, :numref:`mlrm_attri`,
:numref:`merb_attri`, :numref:`mecb_attri`, and :numref:`mlri_attri`.

The wind loss model in HAZUS classifies masonry buildings into 14 building classes: 
1. two masonry single family home classes (MSF1 and MSF2)
2. three masonry multi-unit home classes (MMUH1, MMUH2, and MMUH3)
3. two masonry low-Rise strip mall classes (MLRM1 and MLRM2) classes
4. three masonry engineered residential building classes (MERBL, MERBM, and MERBH)
5. three Masonry engineered commercial building classes (MECBL, MECBM, and MECBH) and 
6. one masonry low-rise industrial building class (MLRI). 

Their required attributes for wind loss modeling, the possible entries (values, terms) that can be 
assigned for those attributes, and the basis for the ruleset developed to make that assignment 
are summarized in :numref:`msf_attri`, :numref:`mmuh_attri`, :numref:`mlrm_attri`, :numref:`merb_attri`, 
:numref:`mecb_attri`, :numref:`mlri_attri`. Note that these rulesets were developed to reflect 
the likely attributes based on the year of construction and the code editions and construction 
norms at that time. The corresponding time-evolving rulesets (PDFs and Python scripts) are 
cross-referenced in :numref:`addinfo_ruleset_masonry`.

.. csv-table:: Additional HAZUS attributes assigned for masonry single family (MSF) homes: wind losses.
   :name: msf_attri
   :file: data/msf_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for masonry multi-unit homes (MMUH): wind losses.
   :name: mmuh_attri
   :file: data/mmuh_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for masonry low-rise strip malls (MLRM): wind losses.
   :name: mlrm_attri
   :file: data/mlrm_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for masonry engineered residential buildings (MERB): wind losses.
   :name: merb_attri
   :file: data/merb_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for HAZUS masonry engineered commercial buildings (MECB): wind losses.
   :name: mecb_attri
   :file: data/mecb_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for masonry low-rise industrial buildings (MLRI): wind losses.
   :name: mlri_attri
   :file: data/mlri_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning wind loss attributes for masonry buildings
   :name: addinfo_ruleset_masonry
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Wind (MSF1-2)
     - `Hazus Building Attribute Rulesets - Wind - MSF1-2.pdf <https://berkeley.box.com/s/8wayd687fxexa5am1zcig4d8lm37j3yq>`_
     - :download:`WindMSFRulesets <data/WindMSFRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (MMUH1-3)
     - `Hazus Building Attribute Rulesets - Wind - MMUH1-3.pdf <https://berkeley.box.com/s/4s8adtuxv09il3tomhi5l8temtvi5h0a>`_
     - :download:`WindMMUHRulesets <data/WindMMUHRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (MLRM1)
     - `Hazus Building Attribute Rulesets - Wind - MLRM1.pdf <https://berkeley.box.com/s/8ml2y60o2foe6vg6myuisuzfqhfu5v23>`_
     - :download:`WindMLRMRulesets <data/WindMLRMRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (MLRM1)
     - `Hazus Building Attribute Rulesets - Wind - MLRM2.pdf <https://berkeley.box.com/s/uqove169jtocgu52okerwuhffgkx8kd0>`_
     - :download:`WindMLRMRulesets <data/WindMLRMRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (MERBL-M-H)
     - `Hazus Building Attribute Rulesets - Wind - MERBL-M-H.pdf <https://berkeley.box.com/s/nzqdg77vamhn75n95kzjmumt0tlgz4zo>`_
     - :download:`WindMERBRulesets <data/WindMERBRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (MECBL-M-H)
     - `Hazus Building Attribute Rulesets - Wind - MECBL-M-H.pdf <https://berkeley.box.com/s/2jgsttwc29gppg8rna35yhwctztn3hwp>`_
     - :download:`WindMECBRulesets <data/WindMECBRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (MLRI)
     - `Hazus Building Attribute Rulesets - Wind - MLRI.pdf <https://berkeley.box.com/s/hn377m3o1pcgsi464xuwe4zwz0vhgjwa>`_
     - :download:`WindMLRIRulesets <data/WindMLRIRulesets.py>`

Taking the attribute **shutters** as an example, the shutters attribute is assigned based on time-evolving 
rules calling upon two fields in the building inventory: year built and the site’s exposure to wind borne 
debris (WBD). :numref:`sht_attri` provides the detailed rules that map these two variables to the shutters 
attribute. This example demonstrates an instance where the attribute is assigned by a code-based 
rule for modern construction, but older construction is assigned as a random variable, based on the 
fact that shutters were not codified before 2000 IBC, but human subjects data suggests potential 
rates of voluntary shutter use. It is assumed that shutters are used only in areas susceptible to WBD.

.. csv-table:: Ruleset for determining the shutter use for masonry engineered commercial buildings.
   :name: sht_attri
   :file: data/example_masonry_ruleset.csv
   :header-rows: 1
   :align: center



Wind Loss Attributes for Steel Buildings
------------------------------------------------

The wind loss model in HAZUS classifies steel buildings into nine building classes: 
1. three steel engineered residential building classes (SERBL, SERBM, and SERBH)
2. three steel engineered commercial building classes (SECBL, SECBM, and SECBH) and
3. three steel pre-engineered metal building systems (SPMBS, SPMBM, SPMBL). 

Their required attributes for wind loss modeling, the possible entries (values, terms) that 
can be assigned for those attributes, and the basis for the ruleset developed to make that 
assignment are summarized in :numref:`serb_attri`, :numref:`secb_attri`, :numref:`spmb_attri`:. 
Note that these rulesets were developed 
to reflect the likely attributes based on the year of construction and the code editions and 
construction norms at that time. The corresponding time-evolving rulesets (PDFs and Python 
scripts) are cross-referenced in :numref:`addinfo_ruleset_steel`.

.. csv-table:: Additional HAZUS attributes assigned for steel engineered residential buildings (SERB): wind losses.
   :name: serb_attri
   :file: data/serb_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for steel engineered commercial buildings (SECB): wind losses.
   :name: secb_attri
   :file: data/secb_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for steel pre-engineered metal building systems (SPMB): wind losses.
   :name: spmb_attri
   :file: data/spmb_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning wind loss attributes for steel buildings
   :name: addinfo_ruleset_steel
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Wind (SERBL-M-H)
     - `Hazus Building Attribute Rulesets - Wind - SERBL-M-H.pdf <https://berkeley.box.com/s/ien050gsa67tlsjrhgvdqjhl3b3thh4f>`_
     - :download:`WindSERBRulesets <data/WindSERBRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (SECBL-M-H)
     - `Hazus Building Attribute Rulesets - Wind - SECBL-M-H.pdf <https://berkeley.box.com/s/7a32df1o9iqx5mzwqqcu94povgabjd8v>`_
     - :download:`WindSECBRulesets <data/WindSECBRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (SPMBS-M-L)
     - `Hazus Building Attribute Rulesets - Wind - SPMBS-M-L.pdf <https://berkeley.box.com/s/3bgxtrr9t5dh1tn6izksdpjxt62zv3wv>`_
     - :download:`WindSPMBRulesets <data/WindSPMBRulesets.py>`


Taking the attribute wind to wall ratio (**WWR**) as an example, the WWR attribute is assigned based on a 
rule that calls upon the window area estimate from the building inventory (field: WindowArea). :numref:`wwr_attri` 
provides the detailed rule that maps this variable to the WWR attribute. Note that WindowArea is a field 
that can be estimated from streetview data, but this rule also demonstrates how the value can be estimated 
based on industry norms (see explanation surrounding default value). This attribute is not assumed to evolve 
with time.

.. csv-table:: Ruleset for determining the window to wall ratio for steel engineered commercial buildings.
   :name: wwr_attri
   :file: data/example_steel_ruleset.csv
   :header-rows: 1
   :align: center


Wind Loss Attributes for Concrete Buildings
------------------------------------------------

The wind loss model in HAZUS classifies steel buildings into 6 building classes: 
1. three concrete engineered residential building classes (CERBL, CERBM, and CERBH) and
2. three concrete engineered commercial building classes (CECBL, CECBM, and CECBH). 

Their required attributes for wind loss modeling, the possible entries (values, terms) that can be assigned 
for those attributes, and the basis for the ruleset developed to make that assignment are summarized in 
:numref:`cerb_attri` and :numref:`cecb_attri`. Note that these rulesets were developed to reflect the likely 
attributes based on the year of construction and the code editions and construction norms at that time. 
The corresponding time-evolving rulesets (PDFs and Python scripts) are cross-referenced in :numref:`addinfo_ruleset_concrete`.

.. csv-table:: Additional HAZUS attributes assigned for concrete engineered residential buildings (CERB): wind losses.
   :name: cerb_attri
   :file: data/cerb_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for concrete engineered commercial buildings (CECB): wind losses.
   :name: cecb_attri
   :file: data/cecb_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning wind loss attributes for concrete buildings.
   :name: addinfo_ruleset_concrete
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Wind (CERBL-M-H)
     - `Hazus Building Attribute Rulesets - Wind - CERBL-M-H.pdf <https://berkeley.box.com/s/sbcgkw8q4ps1mncu9bu87oz83lr5mga3>`_
     - :download:`WindCERBRulesets <data/WindCERBRulesets.py>`
   * - HAZUS Building Attribute Rulesets - Wind (CECBL-M-H)
     - `Hazus Building Attribute Rulesets - Wind - CECBL-M-H.pdf <https://berkeley.box.com/s/scuv8u64atudekvxpda9mh8aol1wuphk>`_
     - :download:`WindCECBRulesets <data/WindCECBRulesets.py>`

Taking the attribute roof cover (RoofCvr) as an example, the RoofCvr attribute is assigned based on a 
ruleset that calls upon the roof shape and year built from the building inventory. :numref:`rc_attri`
provides the detailed rule that maps these variables to the RoofCvr attribute. This provides an example of an 
attribute that is inferred from construction practices based on when different roof cover products entered 
the market. 

.. csv-table:: Ruleset for determining the window to wall ratio for concrete engineered residential buildings.
   :name: rc_attri
   :file: data/example_concrete_ruleset.csv
   :header-rows: 1
   :align: center


Wind Loss Attributes for Manufactured Homes
------------------------------------------------

The wind loss model in HAZUS classifies manufactured homes (MH) into five building classes that are organized 
into three groupings, based on phasing of revisions to Housing and Urban Development (HUD) guidelines: 
1. manufactured homes built before 1976 (MHPHUD)
2. manufactured homes built after 1976 and before 1995 (MH76HUD)
3. manufactured homes built after 1994 (MH94HUDI, M94HUDII, MH94HUDIII). 

Their required attributes for wind loss modeling, the possible entries (values, terms) that can be assigned 
for those attributes, and the basis for the ruleset developed to make that assignment are summarized in 
:numref:`mh_attri`. ote that these rulesets were developed to reflect the likely 
attributes based on the year of construction and the code editions and construction norms at that time. 
The corresponding time-evolving rulesets (PDFs and Python scripts) are cross-referenced in :numref:`addinfo_ruleset_mh`.

.. csv-table:: Additional HAZUS attributes assigned to Manufactured Homes (MH).
   :name: mh_attri
   :file: data/mh_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning wind loss attributes for manufactured homes.
   :name: addinfo_ruleset_mh
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Wind (Manufactured Homes)
     - `Hazus Building Attribute Rulesets - Wind - MH76HUD.pdf <https://berkeley.box.com/s/qfde2jo5ry65ev349fu7bkuz5w2c3162>`_, 
       `Hazus Building Attribute Rulesets - Wind - MH94HUDI-II-III.pdf <https://berkeley.box.com/s/guop1ln5le5rrjqy4l2xk3p3b2b40n55>`_,
       `Hazus Building Attribute Rulesets - Wind - MHPHUD.pdf <https://berkeley.box.com/s/4vz0a7pirgxaunyvt7ot39d5aty7wtan>`_
     - :download:`WindMHRulesets <data/WindMHRulesets.py>`

Taking the attribute tie down (TieDown) as an example, the ruleset in :numref:`td_attri` considers 
the Year Built to determine if tie down use is governed by HUD standards based on the design 
wind speed or if it is a voluntary action predating code requirements and thus is governed by 
human subjects data. This provides an example of an attribute that is inferred from construction 
practices based on when different roof cover products entered the market. 

.. csv-table:: Ruleset for determining the tie down for manufactured homes.
   :name: td_attri
   :file: data/example_mh_ruleset.csv
   :header-rows: 1
   :align: center


Wind Loss Attributes for Essential Facilities
------------------------------------------------

The wind loss model in HAZUS futher classifies several groupings of essential facilities:
1. Fire Staions and Elementary Schools (HUEFFS, HUEFSS)
2. High Schools: 2-story and 3-story (HUEFSM, HUEFSL)
3. Hospitials: small, medium, large (HUEFHS, HUEFHM, HUEFHL)
4. Police Stations and Emergency Operations Centers (HUEFPS, HUEFEO)
   
Their required attributes for wind loss modeling, the possible entries (values, terms) that can be
assigned for those attributes, and the basis for the ruleset developed to make that assignment are summarized 
in :numref:`hu1_attri`, :numref:`hu2_attri`, :numref:`hu3_attri`, and :numref:`hu4_attri`. 
Note that these rulesets were developed to reflect the likely attibutes based on the year of construction and
the code editions and construction norms at that time. The corresponding time-evolving rulesets (PDFs and Python scriots)
are cross-referenced in :numref:`addinfo_ruleset_ef`.

.. csv-table:: Additional HAZUS attributes assigned for fire stations and elementary schools: wind losses.
   :name: hu1_attri
   :file: data/hu1_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for 2-story and 3-story high schools: wind losses.
   :name: hu2_attri
   :file: data/hu2_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned for hospitals: wind losses.
   :name: hu3_attri
   :file: data/hu3_attributes.csv
   :header-rows: 1
   :align: center

.. csv-table:: Additional HAZUS attributes assigned forpolice stations and emergency operation centers: wind losses.
   :name: hu4_attri
   :file: data/hu4_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning wind loss attributes for essential facilities.
   :name: addinfo_ruleset_ef
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Wind (Essential Facilities)
     - `Hazus Building Attribute Rulesets - Wind - HUEFFS-HUEFSS.pdf <https://berkeley.box.com/s/jjjjzr6rhyz7a253qcxrnn7n3vdxv6n5>`_,
       `Hazus Building Attribute Rulesets - Wind - HUEFHS-M-L.pdf <https://berkeley.box.com/s/pumhmup0j60m3skgcdlfthmeg5u3o5pk>`_, 
       `Hazus Building Attribute Rulesets - Wind - HUEFPS-HUEFEO.pdf <https://berkeley.box.com/s/2s7g2x3iot4bajfdprxwzrxu4m6nbulg>`_, 
       `Hazus Building Attribute Rulesets - Wind - HUEFSM-L.pdf <https://berkeley.box.com/s/ffs4gj1hf4bji1y0vzvj3hu93kvw00kt>`_
     - :download:`WindEFRulesets <data/WindEFRulesets.py>`

Taking the attribute wind borne debris source (WindDebris) as an example, the WindDebris attribute is 
assigned based on assumptions surrounding the zoning in areas where each essential facility class is 
commonly constructed. These are generally A: Residential/Commercial or C: Residential, as summarized 
in :numref:`ef_attri`. 

.. csv-table:: Ruleset for determining the wind borne debris for flood essential factilites.
   :name: ef_attri
   :file: data/example_ef_ruleset.csv
   :header-rows: 1
   :align: center


Flood Loss Attributes
-----------------------

The flood loss model in HAZUS focuses on a collection of attributes, some of which are already defined in 
the building inventory (number of stories and occupancy type as defined in :ref:`lbl-testbed_AC_asset_representation_building_classification`), 
while other building inventory fields like first floor elevation require adjustment. The new or adjusted 
attributes required for the flood model are itemised in :numref:`flood_attri` with their possible 
assignments (values, terms) and the ruleset developed to make those assignments. Note that these attributes 
are generally not time evolving, with the exception of considering if the building was constructed after 
Flood Insurance Rate Maps (FIRMs) were adopted (date varies by municipality). The corresponding 
rulesets (PDFs and Python scripts) are cross-referenced in :numref:`addinfo_ruleset_flood`.

.. csv-table:: Additional HAZUS attributes assigned for flood losses.
   :name: flood_attri
   :file: data/flood_attributes.csv
   :header-rows: 1
   :align: center

.. list-table:: Additional details for rulesets assigning flood loss attributes.
   :name: addinfo_ruleset_flood
   :header-rows: 1
   :align: center

   * - Ruleset Name
     - Ruleset Definition Table
     - Python script
   * - HAZUS Building Attribute Rulesets - Flood
     - `Hazus Building Attribute Rulesets - Flood - All Classes.pdf <https://berkeley.box.com/s/1n75p4c37dtet7kvtj44e422aqjs7woa>`_
     - :download:`FloodRulesets <data/FloodRulesets.py>`

Taking the attribute first floor flood elevation (FirstFloorElev) as an example, 
the FirstFloorElev attribute is assigned by adapting the building inventory field (FirstFloorHt1), 
defined by computer vision methods (see :ref:`lbl-testbed_AC_asset_description_phase_iv`), 
and adjusting it based on the conventions used to define this quantity based on the flood zone 
(A-Zone vs. V-Zone), as summarized in :numref:`ffh_attri`.

.. csv-table:: Ruleset for determining the first floor height for flood loss modeling.
   :name: ffh_attri
   :file: data/example_flood_ruleset.csv
   :header-rows: 1
   :align: center


.. [Javeline19]
    Javeline, D., & Kijewski-Correa, T. (2019). Coastal homeowners in a changing climate. Climatic Change, 152(2), 259-274.