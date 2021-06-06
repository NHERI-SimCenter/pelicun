.. _lbl-testbed_AC_damage_and_loss:

**************************
Damage and Loss Estimation
**************************

Damage and loss functions from the HAZUS Hurricane Damage and Loss Model ([FEMA18a_], [FEMA18b_]) 
were implemented in `PELICUN <https://pelicun.readthedocs.io/en/latest/>`_ to support loss assessment for 
all configurations of buildings currently supported by HAZUS for wind and flood hazards. These 
configurations represent all the possible unique configurations of building attributes associated 
with each building class (defined by material) and subclass (defined by occupancy), as discussed in 
:ref:`testbed_AC_asset_representation`. For example, wood (class) single-family homes 1-2+ stories 
(subclass) would have damage and loss functions associated with each unique combination of attributes 
used to define key features of the load path and components (e.g., roof shape, secondary water resistance, 
roof deck attachment, roof-wall connection, shutters, garage), as well as the exposure (terrain roughness).

The HAZUS damage and loss functions consist of tabular data to describe the fragility or expected losses as a 
function of hazard intensity. These data were used to calibrate coupled damage and loss models to estimate 
the damage state and the corresponding expected loss ratio for each building configuration in PELICUN. 
Continuous functions (Normal or Lognormal cumulative density functions) were fit to the synthetic data 
by maximizing the likelihood of the observations assuming a Binomial distribution of outcomes at each 
discrete hazard intensity in the HAZUS database. Coupling the damage and loss models in this way ensures 
more realistic outcomes (e.g., a building with no damage cannot have total loss when the two models are 
coupled), and the parameterized models allow for more efficient storage and computations within the workflow.

1. For the **wind loss assessment**, the HAZUS functions consist of tabular data to 
   describe the fragility or expected losses as a function of peak wind speed (PWS). 
   Only data up to 200 mph wind speeds were used because the substantial reduction in the 
   number of observations introduces significant measurement error above that level (:numref:`wind_df`). 

.. figure:: figure/wind_damage_functions.png
   :name: wind_df
   :align: center
   :figclass: align-center
   :figwidth: 1000

   Fitted HAZUS wind damage functions for example building classes.

2. For the **flood loss assessment**, the HAZUS functions are in the form of depth-damage ratio curves, relating
   the peak water depth (PWD) of flooding (in feet), as measured from the top of the first finished floor,
   to a percent of the total replacement cost of the asset (:numref:`flood_ddc`).

.. figure:: figure/flood_depth_damage_curves.png
   :name: flood_ddc
   :align: center
   :figclass: align-center
   :figwidth: 1000

   Depth-damage ratio curves for the flood loss assessment.

The **total loss** from wind and flood damages are estimated by combining the "wind-only" and "flood-only"
losses following the combination rule in [FEMA18a]_. The primary motivation for the combined wind and
flood loss methodology is to avoid “double counting” of damage. At a minimum, the combined wind and
flood loss must be at least the larger of the wind-only or the flood-only loss. At a maximum, the combined
loss must be no larger than the lesser of the sum of the wind-only and flood-only losses
or 100% of the building (or contents) replacement value. These constraints can be written
as:

.. math::

   max(L_{wind}, L_{flood}) \leq L_{combined} \leq min(L_{wind}+L_{flood}, 1.00)

where :math:`L_{wind}` is the wind-only loss ratio, :math:`L_{flood}` is the flood-only loss ratio, and :math:`L_{combined}`
is the combined loss ratio. The HAZUS combination rule first assumes that the wind-induced damage and flood-induced damage
are spread uniformly and randomly over a building. In this idealized case, the two damage mechanisms can be treated as
independent, and the expected combined loss ratio is simply as:

.. math::

   L_{combined} = L_{wind} + L_{flood} - L_{wind} \times L_{flood}

However,  it is nonetheless clear that neither wind nor storm surge damages are
uniformly and randomly distributed throughout a structure. Wind damage is most
frequently initiated at the roof and fenestrations (i.e., windows,
doors, or other openings in the building envelope), whereas flood damage is most
frequently initiated at the lowest elevations of the structure (e.g., basement or first
finished floor) and progresses upward through the structure as the depth of flooding
increases. HAZUS used an approach for incorporating the non-uniformity of
wind and flood damage into the combined loss methodology, which is based on
allocating wind and flood losses to building *sub-assemblies* as a function of the building
type and the overall wind-only and flood-only loss estimate.

This so-called building sub-assembly approach can more accurately apply the combination calculation above
to each sub-assembly instead of applying it to the entire building. Specifically, HAZUS groups the loss
components into a consistent set of building sub-assemblies:

.. note::
   HAZUS building sub-assemblies ([FEMA18a]_):
      1. Foundation: Includes site work, footings, and walls, slabs, piers or piles.
      2. Below First Floor: Items other than the foundation that are located below the first floor of the structure such as mechanical equipment, stairways, parking pads, break away flood walls, etc.
      3. Structure Framing: Includes all of the main load carrying structural members of the building below the roof framing and above the foundation.
      4. Roof Covering: Includes the roof membrane material and flashing.
      5. Roof Framing: Includes trusses, rafters, and sheathing.
      6. Exterior Walls: Includes wall covering, windows, exterior doors, and insulation.
      7. Interiors: Includes interior wall and floor framing, drywall, paint, interior trim, floor coverings, cabinets, counters, mechanical, and electrical

Hence, the combination is conducted at each sub-assembly level and the total combined loss ratio is the
sum of combined sub-assembly loss ratios:

.. math::

   L_{combined} = \sum\limits_{i=1}^7 L_{wind,i} + L_{flood,i} - L_{wind,i} \times L_{flood,i}

where :math:`L_{wind,i}` is the wind-only loss ratio of the :math:`i^{th}` sub-assembly, and
:math:`L_{flood,i}` is the flood-only loss ratio of the :math:`i^{th}` sub-assembly. These sub-assembly
loss ratios are computed as a percent of the total building loss ratio. The percentages are based on the
:numref:`wind_comp` and :numref:`flood_comp` that are developed per the HAZUS methodology and data table ([FEMA18a]_).

.. csv-table:: Sub-assembly wind-only loss contribution ratio table.
   :name: wind_comp
   :file: data/wind_sub.csv
   :header-rows: 1
   :align: center

.. csv-table:: Sub-assembly flood-only loss contribution ratio table.
   :name: flood_comp
   :file: data/flood_sub.csv
   :header-rows: 1
   :align: center


.. [FEMA18a]
   FEMA (2018), HAZUS – Multi-hazard Loss Estimation Methodology 2.1, Hurricane Model Technical Manual, Federal Emergency Management Agency, Washington D.C., 718p.

.. [FEMA18b]
   FEMA (2018), HAZUS – Multi-hazard Loss Estimation Methodology 2.1, Flood Model Technical Manual, Federal Emergency Management Agency, Washington D.C., 569p.

.. [Javeline19]
   Javeline, D. and Kijewski-Correa, T. (2019) “Coastal Homeowners in a Changing Climate,” Climatic Change. 152(2), 259-276 https://doi.org/10.1007/s10584-018-2257-4
