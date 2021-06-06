.. _lbl-tb_framework_damage:

************
Damage Model
************

Each Fragility Group in the performance model shall have a corresponding fragility model in the Damage & Loss Database. In the fragility model, Damage State Groups (DSGs) collect Damage States (DSs) that are triggered by similar magnitudes of the controlling EDP. In pelicun, Lognormal damage state exceedance curves are converted into random EDP limits that trigger DSGs. When multiple DSGs are used, assuming perfect correlation between their EDP limits reproduces the conventional model that uses exceedance curves. The approach used in this framework, however, allows researchers to experiment with partially correlated or independent EDP limits. Experimental results suggest that these might be more realistic representations of component fragility. A DSG often has only a single DS. When multiple DSs are present, they can be triggered either simultaneously or they can be mutually exclusive following the corresponding definitions in FEMA P58.