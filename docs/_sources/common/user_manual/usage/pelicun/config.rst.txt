.. _lbl-in_config:

******************
Configuration File
******************

Pelicun uses two input files to set up the damage and loss calculation. The Configuration File defines the parameters of the calculation, while the Response Data File (:numref:`lbl-in_response`) defines the demand on the structure through EDP (or IM) values. The information in the configuration file is stored in JSON format. Below we show a sample configuration file and explain the various settings available.


.. literalinclude:: config_example_P58.json


===================
General Information
===================

The General Information part of the file provides basic information about the structure and specifies the units for the analysis. The length unit specified here defines the units for the inputs, including those for areas, and accelerations. For example, if the length unit is set to `m`, pelicun will assume that the plan area is provided in `m2` and the floor accelerations in the EDP file are provided in `m/s2`.

===============
Damage And Loss
===============

This part contains the damage and loss calculation settings broken into several sub-sections for easier navigation. The `_method` identifies the type of assessment to perform, `FEMA P58` and `HAZUS MH` are available in the current version.

--------------
Response Model
--------------

The response model specifies how the raw EDP data is handled in the calculation. The following settings are available:


  :EDP_distribution:
    Specifies the approach to describing the distribution of EDPs. If `empirical` is selected, the raw EDPs are kept as is and resampled during the assessment. The `lognormal` setting fits a multivariate lognormal distribution to the EDPs. The `truncated lognormal` setting can be used to set a truncated multivariate lognormal distribution to the non-collapse results by setting the Basis (the next setting) appropriately.


  :BasisOfEstimate:
    Specifies the basis of the EDP distribution. The `all results` setting uses all samples, while the `non-collapse results` removes the samples that have EDPs beyond the collapse limits (see in a later setting).


  :Realizations:
    The number of EDP and corresponding damage and loss realizations to generate. Depending on the complexity of the model, a thousand realizations might be sufficient to capture central tendencies. A much larger number is required to get appropriate estimates of the dispersion of results. If the EDP distribution is set to empirical, the EDP realizations are sampled from the set of raw EDP values with replacement. If the EDP distribution is set to lognormal or truncated lognormal, the samples are generated from the distribution that is fit to the raw EDP values.


  :CoupledAssessment:
    If set to `true`, the EDPs are used as-is and not resampled.


  :Additional Uncertainty:
    Ground motion and Model uncertainty as per FEMA P58. The prescribed logarithmic standard deviation values are added to the dispersion of EDPs to arrive at the description of uncertain building response.


  :Detection Limits:
    These limits correspond to the maximum possible values that the response history analysis can provide. While peak interstory drifts will certainly have an upper limit, peak floor acceleration will not necessarily require such a setting. Leaving any of the fields empty corresponds to unlimited confidence in the simulation results.

    Note: these limits will be used to consider EDP data as a set of censored samples when fitting the multivariate distribution set under Response Description. If the EDP distribution is set to empirical, this setting has no effect.


------------
Damage Model
------------

The damage model specifies how damages are calculated. The following settings are available:


  :Irreparable Residual Drift:

    Describes the limiting residual drift as a random variable with a lognormal distribution. See Figure 2-5 and Section 7.6 in FEMA P58 for details. The prescribed yield drift ratio is used to estimate residual drifts from peak interstory drifts per Section 5.4 in FEMA P58. This is only needed if no reliable residual drifts are available from the simulation. Considering the large uncertainty in estimated residual drift values, it is recommended to consider using the peak interstory drift as a proxy even if it would be numerically possible to obtain residual drift values.

  :Collapse Probability:

    :Approach:

      Specifies if the collapse probability shall be estimated from EDP samples (`estimated`), or prescribed by the user (`prescribed`).

    :Prescribed value:

      If the prescribed approach is selected above, you can specify the probability of collapse here.

    :Basis:

      If the estimated approach is selected above, you can specify the basis of collapse probability estimation here. `sampled EDP` corresponds to using the (re)sampled EDPs, while `raw EDP` corresponds to using the EDP inputs to evaluate the proportion above the collapse limits to get the collapse probability.

  :Collapse Limits:

    If the Approach under Collapse Probability is set to `estimated`, the collapse of the building in each realization is inferred from the magnitude of EDPs. The collapse limits describe the EDP value beyond which the building is considered collapsed. Note that collapse limits might be beyond the detection limits (although that is generally not a good idea) and certain EDPs might not have collapse limits associated with them (e.g. PFA).

----------
Loss Model
----------

The loss model specifies how the consequences of damages are calculated. The following settings are available:


  :Replacement Cost and Time:

    The cost (in the currency used to describe repair costs, typically US dollars) and the time (in days or workerdays depending on the method used) it takes to replace the building.

  :Decision variables of interest:

    These switches allow the user to pick the decision variables of interest and save computation time and storage space by only focusing on those.

  :Inhabitants:

    :Occupancy Type:

      The type of occupancy is used to describe the temporal distribution of the inhabitants.

    :Peak Population:

      The maximum number of people present at each floor of the building. If a single value is provided, it is divided among the floors. If a list of values is provided, each is assigned to the corresponding floor in the buildng. The example shows a house with no persons on the first floor and 2 persons on its second floor.

    :Custom distribution:

      By default, the calculations are performed using population and fragility data bundled with pelicun. Each data source can be overridden by custom user-defined data. This field allows you to provide a custom json file that describes the change in building population over time.

------------
Dependencies
------------

Pelicun allows users to prescribe dependencies between various components of the performance model. Note that both FEMA P58 and HAZUS assumes most of these settings are set to `Independent`.

Every type of prescribed dependency assumes perfect correlation between a certain subset of the model variables and no correlation between the others. Future versions will expand on this approach by introducing more complex correlation structures.


Logical components
^^^^^^^^^^^^^^^^^^

You can assign perfect correlation between the following logical components of the model:

  :Fragility Groups:
    Assumes that the selected parameters are correlated between Fragility Groups (i.e. the highest organizational level) and at every level below. That is, with this setting, the users assigns perfect correlation between every single parameter of the selected type in the model. Use this with caution.

  :Performance Groups:
    Assumes that the selected parameters are correlated between all Performance Groups and at every logical level below. For instance, this setting for Component Quantities will lead to identical deviations from mean quantities among the floors and directions in the building.

  :Floors:
    Assumes that the selected parameters are correlated between Performance Groups at various floors, but not between Performance Groups in different directions in the building. Also assumes perfect correlation between the Damage States within each Performance Group. This is useful when the parameter is direction-dependent and similar deviations are expected among all floors in the same direction.

  :Directions:
    Assumes that the selected parameters are correlated between Performance Groups in various (typically two) directions, but not between different floors of the building. This can be useful when you want to prescribe similar deviations from mean values within each floor, but want to allow independent behavior over the height of the building.

  :Damage States:
    Correlation at the lowest organizational level. Assumes that the selected parameters are correlated between Damage States only. This type of correlation, for instance, would assume that deviation from the median reconstruction cost is due to factors that affect all types of damage within a performance group in identical fashion.


Model parameters
^^^^^^^^^^^^^^^^^^^^

The following model parameters can handle the assigned dependencies:


  :Component Quantities:
    The amount of components in the building (see the description under Components below for more details).

  :Component Fragilities:
    Each Damage State has a corresponding random EDP limit. The component fragilities is a collection of such EDP limit variables.

    Note: most methodologies assume that such EDP limits are perfectly correlated at least among the Damage States within a Component Subgroup.

  :Reconstruction Costs and Times:
    The cost and time it takes to repair a particular type of damage to a component. The btw. Rec. Cost and Time setting allows you to define correlation between reconstruction cost and time on top of the correlations already set above for each of these individually.

    Note: if you do define such a correlation structure, the more general correlation among the settings in the Reconstruction Costs and Reconstruction Times lines will need to be applied to both cases to respect conditional correlations in the system. (e.g., if you set costs to be correlated between Performance Groups and times to correlate between Floors and check the cost and time correlation as well, times will be forced to become correlated between Performance Groups.)

  :Injuries:
    The probability of being injured at a given severity when being in the affected area of a damaged component. Note that the Injuries lines prescribe correlations between the same level of injury at different places in the building. Correlation between different levels of injury at the same place can be prescribed by the btw. Injuries and Fatalities setting.

  :Red Tag Probabilities:
    The amount of damage in a given Damage State that triggers an unsafe placard or red tag.


The default FEMA P58 setting assumes that all variables are independent, except for the fragility data, where the fragilities of certain *Component Groups* (i.e. groups of components with identical behavior within Performance Groups) are perfectly correlated. This behavior is achieved by setting every other dependency to ``Independent`` and setting the ``Component Fragilities`` to ``per ATC recommendation``.

----------
Components
----------

This part of the configuration file defines the Fragility Groups in the building with the quantities and locations.

Each Fragility Group is identified with a unique ID. The example below uses IDs from FEMA P58, but similar IDs are available in the Hazus database as well. Each ID is assigned to a list of  quantity specifications. The quantity specifications define the location of the Performance Groups that collectively make up the Fragility Group. More than one quantity specifications are warranted when the quantities or the uncertainty in quantities changes by location or direction. For example, the first component in the sample configuration file below has two quantity specfications - one for the first and another for the second direction. Each quantity specficiation has several attributes to specify the quantity and location of components:

  :location:
    In buildings, locations are typically stories. The ground floor is story 1. Providing ``all`` assigns the same setting to every story. You can use a dash to specify a range of stories, such as ``3-7``. If a component is only assigned to the top story, or the roof, you can use ``top`` or ``roof``. You can also combine these and use ``3-roof`` for example. These settings make it easy to transfer performance models between buildings that have different number of stories.

  :direction:
    The directions correspond to EDPs that are used to assess the fragility of the components. They shall match the directions in the EDP results from the simulations. Typically, ``1`` corresponds to the horizontal X direction and ``2`` to the Y direction.

  :median_quantity:
    Components within a Fragility Group are separated into Performance Groups by floor and direction. Components within a Performance Group are further separated into Component Groups. Components within a Component Group experience identical damage and losses. Correlations between the behavior of Component Groups and higher-level sets is specified under Dependencies.

    The list of quantities provided specifies the number of Component Groups and the quantity of components in each Component Group. These are assigned to each Performance Group in the locations and directions provided.

  :unit:
    The unit used to specify component quantities. You can use any of the commonly used metric or US units as long as it belongs to the same class (i.e., length, area, etc.) as the default unit for the fragility group. Squared units are expressed by using a ``2`` after the name, such as ``ft2`` for square feet.

  :distribution:
    If you want to model the uncertainty in component quantities, select either normal or lognormal distribution here. The ``N/A`` setting corresponds to known quantities with no uncertainty.

  :cov:
    Coefficient of variation for the random distribution of component quantities. If the distribution is set to ``N/A``, this can be omitted.


As long as you want to assign the same amount of components to every floor and every direction, one quantity specification is sufficient. Oftentimes, you will want to have more control over component quantities because the amount of components is not identical in all floors and directions.

The example configuration shows the assignment of two Fragility Groups. The first group is only used in the first floor and it has different quantities in direction 1 and 2 and required two quantity specfications in the file. The second Fragility Group could use the same quantity assignment to all floors and directions.

--------------
Collapse Modes
--------------

Collapse modes provide information for the estimation of injuries from building collapse when the FEMA P58 method is used. They are only used if injuries are among the requested Decision Variables. The following pieces of information are required for each collapse mode:

  :name:
    A name that helps you identify the collapse mode. It is arbitrary and not used by the loss assessment engine.

  :weight:
    Conditioned on collapse, the likelihood of this collapse mode. The provided likelihoods across all collapse modes must sum up to 1.0.

  :affected_area:
    The affected area (as a fraction of the total plan area) of the building at each floor. If only one number is provided, the same affected area is assumed on all floors. If a list of numbers is provided, you can specify the affected area on each floor.

  :injuries:
    The probability of each level of injury when people are in the affected area and this collapse mode occurs. (FEMA P58 assumes two levels of severity: injuries and fatalities).


---------------------
Component Data Folder
---------------------

Specifies the location of the fragility and consequence data that will be used for the damage and loss assessment. When empty or omitted, the data from the second edition of FEMA P58 is loaded by default. The corresponding json files are available in the applications folder under: ``resources/FEMA P58 second edition.hdf``

The components from the first edition of FEMA P58 and the earthquake and hurricane damage and loss models from HAZUS are also bundled with pelicun and available under ``resources``. The Fragility and Consequence Data section of tha manual (:numref:lbl-db_DL_dat) provides more information about the data and how it can be edited.