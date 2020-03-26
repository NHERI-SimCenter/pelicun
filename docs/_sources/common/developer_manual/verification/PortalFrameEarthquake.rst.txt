.. _lblPortalFrameEarthquake:

Earthquake Response of Portal Frame 
===================================

In this example, a simple 2D portal frame model is used to verify the
results of |app|. The model is a linear elastic single-bay,
single-story model of a reinforced concrete portal frame, as shown in
:numref:`figPortalVerification`. The analysis of this model considers both
gravity loading and lateral earthquake loading due to the El Centro
earthquake (Borrego Mountain 04/09/68 0230, El Centro ARRAY 9, 270).
The original model and ground motion used in this example were
obtained from `Example 1b <http://opensees.berkeley.edu/wiki/index.php/OpenSees_Example_1b._Elastic_Portal_Frame>`_ on the |OpenSees| website, 
and were modified to scale the ground motion record from gravity
units, **g** to the model units, :math:`\frac{in}{s^2}`. Files for this example are
included with the release of the software and are available in the
Examples folder in a subfolder called **PortalFrame2D**.

   .. _figPortalVerification:

   .. figure:: figures/portalFrame.png
	:align: center
	:figclass: align-center

	Two-dimensional portal frame model subjected to gravity and earthquake loading

To introduce uncertainty in the model, both mass and young’s modulus
are assumed to be normally distributed random variables with means and
standard deviation values shown in :numref:`lblRV`; In this
example, the model will be sampled with the Latin Hypercube sampling
method using both |app| and a Python script
(**PortalFrameSampling.py**) and response statistics from both
analyses are compared.

.. csv-table:: 
   :header: "Uncertain Parameter", "Distribution", "Mean", "Standrad Deviation"
   :widths: 40, 20, 20, 20

   Nodal Mass (m [kip])	 , Normal , 	5.18	 , 1.0 
   Young’s Modulus (E [ksi]) , 	Normal	 , 4227.0	 , 500.0 

 
# \caption{Uncertain parameters defined in the portal frame model}             

Modeling uncertainty using |app| can be done using the
following steps:
#. Start |app|, click on the simulation tab (SIM) in the left bar to open a building simulation model. Click on choose button in the input script row:

.. figure:: figures/portalFrameTcl.png
   :align: center
   :figclass: align-center

   Choose building model


#. Choose the model file \texttt{Portal2D-UQ.tcl} from PortalFrame2D example folder.

.. figure:: figures/tclLocation.png
   :align: center
   :figclass: align-center

   Choose tcl file


#. In the list of Clines Nodes edit box, enter “1, 3”. This indicates to |app| that nodes 1 and 3 are the nodes used to obtain EDP at different floor levels (i.e. base and first floor).

.. figure:: figures/cLineNodes.png
   :align: center
   :figclass: align-center

   Select Nodes

#. Click on the event tab (EVT) in the left bar to open the earthquake event specification tab, select Multiple Existing for loading Type. Click on the add button to add an earthquake event. 
Then click on the choose button to select the event file.

.. figure:: figures/workEvtTab.png
   :align: center
   :figclass: align-center
    
    Work on EVT

#. Choose the event file (\texttt{BM68elc.json}) for El Centro earthquake provided in the portal frame 2D example folder.


.. figure:: figures/evtFileLocation.png
   :align: center
   :figclass: align-center
   
   Choose event file

#. Now select the random variables tab (RVs) from the left bar, change the random variables types to normal and set the mean and standard deviation values of the floor mass and
Young’s modulus.  Notice that |app| has automatically
detected parameters defined in the \texttt{OpenSees} tcl file using the pset
command and defined them as random variables.

.. figure:: figures/workUqTab.png
   :align: center
   :figclass: align-center

   Work on **UQ** tab

#.  Now click on run, set the analysis parameters, working directory and applications directory and click submit to run the analysis. 
If the run is successfull the program will automatically open the
results tab showing the summary of results (\Cref{fig:figure27}).

.. figure:: figures/runAnalysis.png
   :align: center
   :figclass: align-center
    
    Pop-up shown when clicking **Run**

Verification Script
-------------------

A verification script (Listing 1) for propagating the uncertainty was
developed in Python and is included in the example folder.  The script
creates 1000 samples for both the Young’s modulus and mass values
using Latin Hypercube sampling, then modifies the \texttt{OpenSees}
model, runs it and stores the output.  After all the model samples are
processed, the script will compute and output the mean and standard
deviation values of the peak floor acceleration and peak drift.

.. literalinclude:: PortalFrameSampling.py
   :language: python

Verification of Results
-----------------------

This section verifies the results produced for the portal frame
by |app| against the results of running the same
problem using the Python script.  Running the uncertainty
quantification problem locally using EE-UQ and using the Python script
produces the results shown in figures below. The results (mean and standard deviation
values of EDPs) for both are compared in the table below and, as seen, are in good
agreement.


.. figure:: figures/resultsSummaryTab.png
   :align: center
   :figclass: align-center
   
   Outputs from EE-UQ


.. figure:: figures/pyOutputs.png
   :align: center
   :figclass: align-center

   Outputs from PortalFrameSamplying.py script

+------------------------------+-----------+-----------+------------+---------+
| Engineering Demand Parameter |           |   EE-UQ   |  Python    | %Diff   |
+------------------------------+-----------+-----------+------------+---------+
| | Peak Floor Acceleration    | | Mean    | | 68.0836 | | 67.5449  |  | 0.79 |
| |   (in/s^2)                 | | Std Dev | | 12.6956 | | 12.5487  |  | 1.17 |
+------------------------------+-----------+-----------+------------+---------+
| | Peak Story Drift           | | Mean    | | 1.3649  | | 1.3470   | | 1.32  |
| |       (x10-3 in)           | | Std Dev | | 0.3017  | | 0.2955   | | 2.10  | 
+------------------------------+-----------+-----------+------------+---------+
    

