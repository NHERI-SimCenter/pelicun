.. _lbl-testbed_SF_damage_and_loss:

**************************
Damage and Loss Estimation
**************************

The building performance assessment was performed on a
story-level basis using **PELICUN** ([Zsarnóczay20]_), where damage and losses are calculated with storylevel
fragility functions based on the peak story drift and
floor acceleration demands. The story-based damage and loss
fragility functions are derived from corresponding building-level
damage and loss functions from the HAZUS earthquake
model ([FEMA18]_) based on the characteristic data for each
building (e.g., year of construction, structure type, occupancy
type). Specially, four major states included in this 
testbed are **Slight (DS1)**, **Moderate (DS2)**, **Extensive (DS3)**, 
and **Complete (DS4)**, and one **None (DS0)** damage 
state. The four major damage states are defined seperately for the 
structural and non-structural damages. More detailed 
descriptions for the example W1 building are provided below. 
The full descriptions can be found in the [FEMA18]_.
Collapse safety limit states are evaluated directly from the
story drift demands, where a collapse of one or more stories is
considered as partial collapse of the entire building.

.. note::

   **W1: Wood Light Frame (Structural)**:

   1. Slight: Small plaster or gypsum-board cracks at corners of door and
   window openings and wall-ceiling intersections; small cracks in masonry chimneys and
   masonry veneer.

   2. Moderate: Large plaster or gypsum-board cracks at corners of door
   and window openings; small diagonal cracks across shear wall panels exhibited by small
   cracks in stucco and gypsum wall panels; large cracks in brick chimneys; toppling of tall
   masonry chimneys.

   3. Extensive: Large diagonal cracks across shear wall panels or large
   cracks at plywood joints; permanent lateral movement of floors and roof; toppling of most
   brick chimneys; cracks in foundations; splitting of wood sill plates and/or slippage of
   structure over foundations; partial collapse of "room-over-garage" or other "soft-story"
   configurations; small foundations cracks.

   4. Complete: Structure may have large permanent lateral displacement,
   may collapse, or be in imminent danger of collapse due to cripple wall failure or the failure
   of the lateral load-resisting system; some structures may slip and fall off the foundations;
   large foundation cracks. Approximately 3\% of the total area of W1 buildings with Complete
   damage is expected to be collapsed.

.. note::

   **Non-structural Damage (Partition Walls)**:

   1. Slight: A few cracks are observed at intersections of walls and
   ceilings and at corners of door openings.

   2. Moderate: Larger and more extensive cracks requiring repair and
   repainting; some partitions may require replacement of gypsum board or other finishes.

   3. Extensive: Most of the partitions are cracked and a significant
   portion may require replacement of finishes; some door frames in the partitions are also
   damaged and require re-setting.

   4. Complete: Most partition finish materials and framing may have to
   be removed and replaced, damaged studs repaired, and walls refinished. Most door frames
   may also have to be repaired and replaced.

.. note::

   **Non-structural Damage (Suspended Cellings)**:

   1. Slight: : A few ceiling tiles have moved or fallen down.

   2. Moderate: Falling of tiles is more extensive; in addition, the ceiling
   support framing (T-bars) has disconnected and/or buckled at a few locations; lenses have
   fallen off some light fixtures and a few fixtures have fallen; localized repairs are necessary.

   3. Extensive: The ceiling system exhibits extensive buckling,
   disconnected T-bars and falling ceiling tiles; ceiling partially collapses at a few locations and
   some light fixtures fall; repair typically involves removal of most or all ceiling tiles.

   4. Complete: The ceiling system is buckled throughout and/or fallen
   and requires complete replacement; many light fixtures fall.

.. note::

   **Non-structural Damage (Exterior Wall Panels)**:

   1. Slight: : Slight movement of the panels, requiring realignment.

   2. Moderate: The movements are more extensive; connections of
   panels to structural frame are damaged requiring further inspection and repairs; some
   window frames may need realignment.

   3. Extensive: Most of the panels are cracked or otherwise damaged
   and misaligned, and most panel connections to the structural frame are damaged requiring
   thorough review and repairs; a few panels fall or are in imminent danger of falling; some
   window panes are broken and some pieces of glass have fallen.

   4. Complete: Most panels are severely damaged, most connections
   are broken or severely damaged, some panels have fallen and most are in imminent
   danger of falling; extensive glass breakage and falling.

.. note::

   **Non-structural Damage (Electrical Mechanical Equipment, Piping, Ducts)**:

   1. Slight: : The most vulnerable equipment (e.g., unanchored or
   mounted on spring isolators) moves and damages attached piping or ducts.

   2. Moderate: Movements are larger and damage is more extensive;
   piping leaks occur at a few locations; elevator machinery and rails may require realignment.

   3. Extensive: Equipment on spring isolators topples and falls; other
   unanchored equipment slides or falls, breaking connections to piping and ducts; leaks
   develop at many locations; anchored equipment indicate stretched bolts or strain at
   anchorages.

   4. Complete: Equipment is damaged by sliding, overturning or failure
   of their supports and is not operable; piping is leaking at many locations; some pipe and
   duct supports have failed, causing pipes and ducts to fall or hang down; elevator rails are
   buckled or have broken supports and/or counterweights have derailed. 

The story drift and floor accelerations from 25 non-linear analyses of each
building are used to define multivariate lognormal distributions
of peak drifts and accelerations for each story of the building,
and the dispersion in the drift and acceleration demands is
inflated by 0.22 to account for additional modeling uncertainties
not considered in the non-linear dynamic analyses. 

Using the distributions of earthquake demands, and damage and loss
functions, **PELICUN** generates 20,000 realizations of damage and
losses for each building, and stores statistics of the resulting
performance data that are relevant for regional-scale evaluation.
The results are output as HDF5 (Hierarchical Data Format) files
that can be processed and visualized through MatLab, Python,
Jupyter notebooks, or converted to CSV format.


.. [Zsarnóczay20]
   Zsarnóczay, A., and Deierlein, G. G. (2020). “PELICUN – A Computational Framework for Estimating Damage, Loss and Community Resilience,” 
   in Proceedings, 17th World Conference on Earthquake Engineering, (Sendai: WCEE).

.. [FEMA18]
   FEMA (2018), HAZUS – Multi-hazard Loss Estimation Methodology 2.1, Earthquake Model Technical Manual, Federal Emergency Management Agency, Washington D.C.
