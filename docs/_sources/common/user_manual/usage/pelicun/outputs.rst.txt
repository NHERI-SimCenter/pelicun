.. _lbl-outputs:

============
Output Files
============

Damage and loss calculations in pelicun provide several output files that describe the EDP, DM, and DV values at the atomic level corresponding to each realization for each component group and also aggregated at a few higher levels in the performance model hierarchy for convenience.

The ``DL_calculation.py`` script provides all of the output files listed below. When researchers directly use methods from the pelicun library in their own scripts, they can pick the methods to save computation time by only saving the outputs they need.

  :Summary Data:

    :DL_summary:
      Provides a high-level overview of the main results. Each row corresponds to a damage and loss realization and each column corresponds to an outcome.

    :DL_summary_stats:
      Rather than providing results for each realization, this file contains frequently used statistics to describe the main results: mean, standard deviation, 10%, 50%, and 90% percentile, and minimum and maximum values of each outcome.

  :Response:

    :EDP_:
      Provides the EDP values used in the calculation. Each row corresponds to a realization and each column corresponds to an EDP type and is identified in the header using SimCenter's nomenclature for EDPs. The EDPs are provided in the units specified by the user in the Configuration File. In a coupled analysis, these EDPs should be identical to the ones in the Response Data File. In an uncoupled analysis when EDPs are re-sampled, these EDPs are the sampled values.

    :EDP_stats:
      Contains frequently used statistics to describe the EDPs: mean, standard deviation, 10%, 50%, and 90% percentile, and minimum and maximum values of each EDP type.

  :Damage:

    :DMG:
      Provides the damaged quantities per Damage State (header row 3) per Performance Group (header row 2) per Fragility Group (header row 1). Each row corresponds to a realization. Note that component damages are only calculated for non-collapsed cases, therefore the number of realizations in this file might be less than the total number of realizations. The Damage State is provided in the format of ``DSG_DS`` where DSG identifies the Damage State Group and DS identifies the Damage State within the DSG. See the Technical Manual for details on handling Damage States in the pelicun framework.

    :DMG_agg:
      Aggregates damaged quantitites to the Fragility Group level (i.e., sums up damaged quantities for all Performance Groups in all Damage States within a Fragility Group).

    :DMG_stats:
      Contains frequently used statistics to describe the quantities of damaged components: mean, standard deviation, 10%, 50%, and 90% percentile, and minimum and maximum values of damaged quantities.

  :Losses:

    :DV_<DV_type>:
      Provides decision variable values per Damage State (header row 3) per Performance Group (header row 2) per Fragility Group (header row 1). Each row corresponds to a realization. Note that component losses are only calculated for non-collapsed cases, therefore the number of realizations in this file might be less than the total number of realizations. The Damage State is provided in the format of ``DSG_DS`` where DSG identifies the Damage State Group and DS identifies the Damage State within the DSG. See the Technical Manual for details on handling Damage States in the pelicun framework.

      The following ``DV_types`` are provided by pelicun: ``injuries``, ``rec_cost``, ``rec_time``, ``red_tag``.

    :DV_<DV_type>_agg:
      Aggregates decision variable values to the Fragility Group level (i.e., sums up values for all Performance Groups in all Damage States within a Fragility Group).

    :DV_<DV_type>_stats:
      Contains frequently used statistics to describe the decision variable values of components: mean, standard deviation, 10%, 50%, and 90% percentile, and minimum and maximum values of decision variables.