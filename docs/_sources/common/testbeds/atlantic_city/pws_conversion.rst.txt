.. _lbl-testbed_AC_wind_speed_conversion:

********************************
Standard Wind Speed Conversions
********************************

Since the wind speed (:math:`V(600s, 10m, z_0)`) from the NJcoast SHP Tool is averaged over the time window of 1 hour, 
a number of standard conversions parse the wind speed to the 3-second and open-terrain PWS 
(i.e., :math:`V(3s, 10m, Exposure C)`):


1. Computing :math:`\alpha` and :math:`z_g` by ASCE 7-16 ([ASCE16_]) Equation C26.10-3 and C26.10-4
taking :math:`c_1 = 5.65, c_2 = 450` for units in m):

.. math::

   \alpha_{SHP} = c_1z_0^{-0.133}

   z_{g,SHP} = c2z_0^{0.125}

2. Computing the gradient height :math:`V(600s, z_g, z_0)` using the power law expression:

.. math::

   V(600s, z_g, z_0) = V(600s, 10m, z_0) \times (\frac{z_{g,SHP}}{10m})^{1/\alpha_{SHP}}

3. Computing the Exposure C (open-terrain) wind speed at 10m height :math:`V(600s, 10m, Exposure C)`, with
:math:`\alpha_C = 9.5` and :math:`z_{g,C} = 274.32 m` ([ASCE16_]):

.. math::

   V(600s, 10m, Exposure C) = V(600s, z_{g,C}, z_0) \times (\frac{10m}{z_g})^{1/\alpha_C}

4. Converting the result to 3s-gust wind speed using the ESDU conversion [ESDU02_]:

.. math::

   V(3s, 10m, Exposure C) = V(600s, 10m, Exposure C) \times \frac{C(3s)}{C(600s)}

where :math:`C(3s) = 1.52` and :math:`C(600s) = 1.05` are the Gust Factor from the ESDU conversion.


.. [ASCE16]
   ASCE (2016), Minimum Design Loads for Buildings and Other Structures, ASCE 7-16,
   American Society of Civil Engineers.

.. [ESDU02]
   Engineering Sciences Data Unit (ESDU). (2002). “Strong winds in the atmospheric boundary
   layer—Part 2: Discrete gust speeds.” ESDU International plc, London, U.K.



