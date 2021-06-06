.. _lbl-db_DL_dat:

********************************
Damage and Loss Model Parameters
********************************

Parameters of several widely used damage and loss models are bundled with pelicun and stored in the `resources` folder in HDF5 file format. Each HDF5 file can be converted into a series of JSON files using the `export_DB` tool provided with pelicun (see :numref:blb). Working with JSON files makes it easier to edit and extend the existing databases. Pelicun recognizes both file formats.

---------
Databases
---------

  :FEMA P58 1st Edition:
    The `FEMA_P58_1st_ed.hdf` database includes the fragility functions provided with the first edition of FEMA P58. It is based on the XML files included with the PACT tool and Excel spreadsheets provided as part of Volume 3 of FEMA P58. All of the sources used to generate this database are available under `resources/data_sources`.

  :FEMA P58 2nd Edition:
    The `FEMA_P58_2nd_ed.hdf` database includes the updated fragility functions provided with the second edition of FEMA P58. This is the default database for pelciun damage and loss calculations. It is based on the XML files included with the PACT tool and Excel spreadsheets provided as part of Volume 3 of FEMA P58. All of the sources used to generate this database are available under `resources/data_sources`.

  :HAZUS Earthquake Model:
    There are three databases provided based on the HAZUS earthquake model. The `HAZUS_MH_2.1_EQ` database contains fragility and consequence functions controlled by building level EDPs (i.e., peak drift and peak acceleration). The damage model is described in detail in Section 5.4.2, while the loss models are described in sections 11.2 and 12.2 in version 4.2 of the Hazus Earthquake Technical Manual. Using these models requires information about the peak roof drift and the peak roof acceleration.

    The `HAZUS_MH_2.1_EQ_eqv_PGA` database contains fragility and consequence functions controlled by Peak Ground Acceleration only. This allows for estimating ground shaking impact without performing response simulation. The equivalent PGA-based damage models are described in detail in Section 5.4.3 in version 4.2 of the Hazus Earthquake Technical Manual. The loss models in this database are identical to those in the previously mentioned one.

    The `HAZUS_MH_2.1_EQ_story` database contains fragility and consequence functions controlled by story-level EDPs. This is an experimental database that was derived based on the building-level EDP damage models in Hazus. The story-level damage model provides higher resolution information about building damage, but requires higher resolution building response data. The loss models in this database are based on the loss models in the previous databases. They have been calibrated to provide the same building-level losses as the Hazus loss models.

    The data used to generate the Hazus earthquake damage and loss files is available under `resources/data_sources/HAZUS_MH_2.1_EQ_*/`

  :HAZUS Hurricane Model:
    The `HAZUS_MH_2.1_HU` database contains fragility functions controlled by Peak Wind Gust speed and corresponding consequence functions for loss ratio. The models were fitted to damage and loss data used within the Hazus framework. The models in pelicun are different from those in Hazus. Hazus uses an uncoupled approach for damage and losses. A series of data points define the fragility and consequence functions with linear interpolation used in-between the available points. That data is described in sections 5.6 and 8.2 in version 4.2 of the Hazus Hurricane Technical Manual. Note that the raw data is not provided in the Technical Manual, but it is available in the database that comnes with the Hazus software.

    Based on the raw data from Hazus, we fitted a damage model that assumes fragility functions have either a normal or a lognormal CDF format. Fitting was performed using a maximum likelihood approach and points above 200 mph wind speed were discarded to avoid bias due to the substantial sampling error in those results. The loss model in this database is deterministic - it assumes a constant loss ratio for each damage state. The loss ratios were fitted to Hazus loss data. The solution in pelicun provides a coupled damage and loss model for hurricane damage and loss assessment.

    The data used to generate the Hazus hurricane damage and loss files is available under `resources/data_sources/HAZUS_MH_2.1_HU`


----------------
Data File Format
----------------

    The sample file below shows the organization of damage and loss model data in the JSON files used in pelicun.

    .. literalinclude:: fragility_example.json