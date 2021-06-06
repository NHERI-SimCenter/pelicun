.. _lbl-testbed_SF_hazard_characterization:

***********************
Hazard Characterization
***********************

The ground motions for the Mw 7.0 Hayward earthquake were
simulated by Rodgers et al. ([Rodgers19]_) at the Lawrence Livermore
National Lab (LLNL) using the SW4 finite difference code
([Petersson17]_). SW4 solves the elasto dynamic
equations of motion in the time domain for a 3D solid.
A 77 ~ 13 km rupture surface was projected onto the fault
geometry in the 3D geologic and seismic model for the Bay
Area ([USGS18]_) with a hypocenter near the San Leandro
salient. Waveforms were sampled in three dimensions on a 2
km grid over the 120 ~ 80 km surface of a 35 km deep solid
body. The resulting waveforms capture ground shaking reliably
over the 0–5 Hz frequency domain for sites with a characteristic
shear wave velocity above 500 m/s. The computations were run
using more than 8,000 nodes (~ 500,000 processors) on the Cori
Phase-II cluster ([NERSC20]_).

The raw results at 2301 grid points were processed by the
SimCenter and converted to the JSON file format used by
our workflow applications. These data provide sets of three-component
seismograms for grid points spaced every 2 km
throughout the study region. The ground motions are assigned
to buildings using a nearest-neighbor search algorithm, where the
four nearest grid points are identified for each building and a set
of 25 seismograms are assigned by weighted random sampling of
the set of time histories from the nearest grid points. The weight
of each grid point is inversely proportional to its squared distance
from the building. :numref:`pgaX` and :numref:`pgaY` show the resulting 
mean peak ground acceleration maps (in East-West and North-South directions) 
based on the 25 ground motions at each site.

.. figure:: figure/PGA-X.png
    :name: pgaX
    :align: center
    :figclass: align-center
    :figwidth: 800

    Peak ground acceleration (East-West direction) for the Mw 7.0 Hayward earthquake.

.. figure:: figure/PGA-Y.png
    :name: pgaY
    :align: center
    :figclass: align-center
    :figwidth: 800

    Peak ground acceleration (North-South direction) for the Mw 7.0 Hayward earthquake.


.. [Rodgers19]
   Rodgers, A. J., Petersson, N. A., Pitarka, A., McCallen, D. B., Sjogreen, B., and Abrahamson, N. (2019). 
   Broadband (0-5 Hz) Fully Deterministic 3D Ground-Motion Simulations of a Magnitude 7.0 Hayward Fault Earthquake: 
   Comparison with Empirical Ground-Motion Models and 3D Path and Site Effects from Source Normalized Intensities. 
   Seismol. Res. Lett. 90:17.

.. [Petersson17]
   Petersson, N. A., and Sjogreen, B. (2017). SW4, version 2.0 [software], Computational Infrastructure of Geodynamics. 
   Switzerland: Zenodo.doi: 10.5281/zenodo.1045297

.. [USGS18]
   USGS (2018). 3-D geologic and seismic velocity models of the San Francisco Bay region. Virginia, VA: USGS. 
   https://earthquake.usgs.gov/data/3dgeologic.

.. [NERSC20]
   NERSC (2020). Cori Cray XC 40 Computer. Berkeley: National Energy Research Scientific Computing Center. 
   https://docs.nersc.gov/systems/cori/.