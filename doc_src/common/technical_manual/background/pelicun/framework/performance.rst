.. _lbl-tb_framework_performance:

*****************
Performance Model
*****************

Thread (b) in Fig. 3 starts with parsing the AIM file and constructing a performance model. If the definition in the file is incomplete, the auto-populate method tries to fill the gaps using information about normative component quantities and pre-defined rulesets. Rulesets can link structural information, such as the year of construction, to performance model details, such as the type of structural details and corresponding components.

The performance model in pelicun is based on that of the FEMA P58 method. It disaggregates the asset into a hierarchical description of its structural and non-structural components and contents (Fig. 4):

- Fragility Groups (FGs) are at the highest level of this hierarchy. Each FG is a collection of components that have similar fragility controlled by a specific type of EDP and their damage leads to similar consequences.

- Each FG can be broken down into Performance Groups (PGs). A PG collects components whose damage is controlled by the same EDP. Not only the type of the EDP, but its location and direction also has to be identical.

- In the third layer, PGs are broken down into the smallest units: Component Groups (CGs). A CG collects components that experience the same damage (i.e., there is perfect correlation between their random Damage States). Each CG has a Component Quantity assigned to it that defines the amount of components in that group. Both international standard and imperial units are supported as long as they are consistent with the type of component (e.g., m2, ft2, and in2 are all acceptable for the area of suspended ceilings, but ft is not.) Quantities can be random variables with either Normal or Lognormal distribution.

In performance models built according to the FEMA P58 method, buildings typically have FGs sensitive to either PID or PFA. Within each FG, components are grouped into PGs by stories and the drift-sensitive ones are also grouped by direction. The damage of acceleration-sensitive components is based on the maximum of PFAs in the two horizontal directions. The Applied Technology Council (ATC) provides a recommendation for the correlation between component damages within a PG. If the damages are correlated, all components in a PG are collected in a single CG. Otherwise, the performance model can identify an arbitrary number of CGs and their damages are evaluated independently.

The pelicun framework handles the probabilistic sampling for the entire performance model with a single high-dimensional random variable. This allows for custom dependencies in the model at any level of the hierarchy. For example, one can assign a 0.8 correlation coefficient between the fragility of all components in an FG that are on the same floor, but in different directions and hence, in different PGs. In another example, one can assign a 0.7 correlation coefficient between component quantities in the same direction along all or a subset of floors. These correlations can capture more realistic exposure and damage and consider the influence of extreme cases. Such cases are overlooked when independent variables are used because the deviations from the mean are cancelling each other.

This performance model in Fig. 4 can also be applied to more holistic description of buildings. For example, to describe earthquake damage to buildings following HAZUS, three FGs can handle structural, acceleration-sensitive non-structural, and drift-sensitive non-structural components. Each FG has a single PG because HAZUS uses building-level EDPs—only one location and direction is used in this case. Since components describe the damage to the entire building, using one CG per PG with “1 ea” as the assigned, deterministic component quantity is appropriate.

The performance model in pelicun can facilitate filling the gap between the holistic and atomic approaches of performance assessment by using components at an intermediate resolution, such as story-based descriptions, for example. These models are promising because they require less detailed inputs than FEMA P58, but they can provide more information than the building-level approaches in HAZUS.
