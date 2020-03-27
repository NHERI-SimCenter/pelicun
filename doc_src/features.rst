Features
============

The following table outlines the features that are currently available in the tool and the requirements that will drive future development. We welcome suggestions for useful features that are missing from the list below. The priority column provides information about the relative importance of features planned for a given release: **M** - mandatory, **D** - desirable, **O** - optional, **P** - possible.

.. table:: List of features
    :widths: 1, 50, 1, 1, 1
    :class: tight-table

    ==== ========================================================================================= ========= ========== ========
    ID   description                                                                               priority  available  planned 
    ==== ========================================================================================= ========= ========== ========
    1    **Assessment Methods**                                                          
    1.1  Perform component-based (e.g. FEMA-P58 style) loss assessment for earthquake scenarios.   M         1.0               
    1.2  Perform component-group-based (e.g HAZUS style) loss assessment for earthquake scenarios. D                    1.1
    1.3  Perform loss assessment for hurricane scenarios based on the HAZUS hurricane methodology. D                    1.2
    1.4  Perform downtime estimation using the ARUP's REDi methodology.                            D                    1.2
    1.5  Perform time-based assessment for seismic hazard.                                         M                    1.3
    
    2    **Control**       
    2.1  Specify number of realizations.                                                           M         1.0
    2.2  Specify log-standard deviation increase to consider additional sources of uncertainty.    M         1.0
    2.3  Pick the decision variables to calculate.                                                 D         1.0
    2.4  Specify the number of inhabitants on each floor and their temporal distribution.          D         1.0
    2.5  Specify the basic boundary conditions of repairability.                                   D         1.0
    2.6  Control collapse through EDP limits.                                                      D         1.0
    2.7  Specify the replacement cost and time for the asset.                                      M         1.0
    2.8  Specify EDP boundaries that define the domain with reliable simulation results.           D         1.0
    2.9  Specify collapse modes and characterize the corresponding likelihood of injuries.         D         1.0
    
    3    **Component DL information**
    3.1  Make the component damage and loss data from FEMA P58 (1st ed.) available in the tool.    M         1.0
    3.2  Facilitate the use of custom components for loss assessment.                              D         1.0
    3.3  Enable different component quantities for each floor in each direction.                   D         1.0
    3.4  Enable fine control over quantities of identical groups of components within a PG.        D         1.0
    3.5  Create a generic JSON data format to store component DL data.                             D                    1.1
    3.6  Convert FEMA P58 and HAZUS component DL data to the new JSON format.                      D                    1.1
    3.7  Extend the list of available decision variables with those from HAZUS                     D                    1.2
    3.8  Extend the list of available decision variables with those from REDi                      D                    1.2
    
    4    **Stochastic loss model**
    4.1  Enable control of basic dependencies between logically similar parts of the model.        D         1.0
    4.2  Enable control of basic dependencies between reconstruction cost and reconstruction time. D         1.0
    4.3  Enable control of basic dependencies between different levels of injuries.                D         1.0
    4.4  Extend the model to include the description of the hazard (earthquake and hurricane).     D                    1.3
    4.5  Enable finer control of dependencies through intermediate levels of correlation.          D                    1.3 

    5    **Response estimation**
    5.1  Fit a multivariate random distribution to samples of EDPs from response simulation.       M         1.0
    5.2  Allow estimation of EDPs using empirical functions instead of simulation results.         D                    1.2
    5.3  Perform EDP estimation using the empirical functions in the HAZUS earthquake method       D                    1.2 
    ==== ========================================================================================= ========= ========== ========

Releases
--------

Minor releases are planned to follow quarterly cycles while major releases are planned at the end of the third quarter every year:

.. table:: Release schedule
    :class: tight-table

    ======== =====================
    version  planned release date
    ======== =====================
    1.0      Oct 2018
    1.1      Dec 2018
    1.2      March 2019
    1.3      June 2019
    2.0      Sept 2019
    ======== =====================

























