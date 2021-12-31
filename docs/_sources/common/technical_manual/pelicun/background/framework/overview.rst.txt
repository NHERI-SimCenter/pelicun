.. _lbl-tb_framework_overview:

********
Overview
********

The conceptual design of the pelicun framework is modeled after the FEMA P58 methodology, which is generalized to provide a flexible system that can accommodate a large variety of damage and loss assessment methods. In the following discussion, we first describe the types of performance assessment workflows this framework aims to support; then, we explain the four interdependent models that comprise the framework.

Loss assessment in its most basic form requires the characterization of the seismic hazard as input and aims to provide an estimate of the consequences, or losses, as output. Using the terminology of FEMA P58, the severity of the seismic hazard is quantified with the help of intensity measures (IMs). These are characteristic attributes of the ground motions, such as spectral acceleration, peak ground acceleration, or peak ground velocity. Consequences are measured by decision variables (DVs). The most popular DVs are repair cost, repair time, and the number of injuries and fatalities. :numref:`figPerfAssWorkflows` shows three different paths, or performance assessment workflows, from IM to DV:

I. The most efficient approach takes a single, direct step using vulnerability functions. Such vulnerability functions can be calibrated for broad classes of buildings (e.g. single-family wooden houses) using ground motion intensity maps and insurance claims data from past earthquakes. While this approach allows for rapid calculations, it does not provide information about structural response or damage.

II. The second approach introduces damage measures (DMs), which classify damages into damage states (DSs). Each damage state represents a set of potential damages to a structure, or structural component, that require similar kinds and amounts of repair effort. Given a database of IMs and corresponding DMs after an earthquake, fragility functions can be calibrated to describe the relationship between them. The more data is available, the more specialized (i.e., specific to particular types of buildings) fragility functions can be developed. The second step in this path uses consequence functions to describe losses as a function of damages. Consequence functions that focus on repairs can be calibrated using cost and time information from standard construction practice—a major advantage over path I considering the scarcity of post-earthquake repair data.

III. The third path introduces one more intermediate step: response estimation. This path envisions that the response of structures can be estimated, such as with a sophisticated finite element model, or measured with a structural health monitoring system. Given such data, damages can be described as a function of deformation, relative displacement, or acceleration of the structure or its parts. These response variables, or so-called engineering demand parameters (EDPs), are used to define the EDP-to-DM relationships, or fragility functions. Laboratory tests and post-earthquake observations suggest that EDPs are a good proxy for the damages of many types of structural components and even entire buildings.

.. _figPerfAssWorkflows:

.. figure:: figures/PerfAssWorkflows.png
	:align: center
	:figclass: align-center

	Common workflows for structural performance assessment.

The functions introduced above are typically idealized relationships that provide a probabilistic description of a scalar output (e.g., repair cost as a random variable) as a function of a scalar input. The cumulative distribution function and the survival function of Normal and Lognormal distributions are commonly used in fragility and vulnerability functions. Consequence functions are often constant or linear functions of the quantity of damaged components. Response estimation is the only notable exception to this type of approximation, because it is regularly performed using complex nonlinear models of structural behavior and detailed time histories of seismic excitation.

Uncertainty quantification is an important part of loss assessment. The uncertainty in decision variables is almost always characterized using forward propagation techniques, Monte Carlo simulation being the most widely used among them. The distribution of random decision variables rarely belongs to a standard family, hence, a large number of samples are needed to describe details of these distributions besides central tendencies. The simulations that generate such large number of samples at a regional scale can demand substantial computational resources. Since the idealized functions in paths I and II can be evaluated with minimal computational effort, these are applicable to large-scale studies. In path III, however, the computational effort needed for complex response simulation is often several orders of magnitude higher than that for other steps. The current state of the art approach to response estimation mitigates the computational burden by simulating at most a few dozen EDP samples and re-sampling them either by fitting a probability distribution or by bootstrapping. This re-sampling technique generates a sufficiently large number of samples for the second part of path III. Although response history analyses are out of the scope of the pelicun framework, it is designed to be able to accommodate the more efficient, approximate methods, such as capacity spectra and surrogate models. Surrogate models of structural response (e.g., [11]) promise to promptly estimate numerical response simulation results with high accuracy.

Currently, the scope of the framework is limited to the simulation of direct losses and the calculations are performed independently for every building. Despite the independent calculations, the pelicun framework can produce regional loss estimates that preserve the spatial patterns that are characteristic to the hazard, and the built environment. Those patterns stem from (i) the spatial correlation in ground motion intensities; (ii) the spatial clusters of buildings that are similar from a structural or architectural point of view; (iii) the layout of lifeline networks that connect buildings and heavily influence the indirect consequences of the disaster; and (iv) the spatial correlations in socioeconomic characteristics of the region. The first two effects can be considered by careful preparation of inputs, while the other two are important only after the direct losses have been estimated. Handling buildings independently enables embarrassingly parallel job configurations on High Performance Computing (HPC) clusters. Such jobs scale very well and require minimal additional work to set up and run on a supercomputer.

*******************************
Performance Assessment Workflow
*******************************

:numref:`figMainWorkflowComps` introduces the main parts and the generic workflow of the pelicun framework and shows how its implementation connects to other modules in the SimCenter Application Framework. Each of the four highlighted models and their logical relationship are described in more detail in :numref:`figModelTypes`.

.. _figMainWorkflowComps:

.. figure:: figures/MainWorkflowComps.png
	:align: center
	:figclass: align-center

	The main components and the workflow of the pelicun framework.

.. _figModelTypes:

.. figure:: figures/ModelTypes.png
	:align: center
	:figclass: align-center

	The four types of models and their logical relationships in the pelicun framework.

The calculation starts with two files: the Asset Information Model (AIM) and the EVENT file. Currently, both files are expected to follow a standard JSON file format defined by the SimCenter. Support of other file formats and data structures only require a custom parser method. The open source implementation of the framework can be extended by such a method and the following part of the calculation does not require any further adjustment. AIM is a generalized version of the widely used BIM (Building Information Model) idea and it holds structural, architectural, and performance-related information about an asset. The word asset is used to emphasize that the scope of pelicun is not limited to building structures. The EVENT file describes the characteristic seismic events. It typically holds information about the frequency and intensity of the event, such as its occurrence rate or return period, and corresponding ground motion acceleration time histories or a collection of intensity measures.

Two threads run in parallel and lead to the simulation of damage and losses: (a) response estimation, creating the response model, and simulation of EDPs; and (b) assembling the performance, damage, and loss models. In thread (a), the AIM and EVENT files are used to estimate the response of the asset to the seismic event and characterize it using EDPs. Peak interstory drift (PID), residual interstory drift (RID), and peak floor acceleration (PFA) are typically used as EDPs for building structures. Response simulation is out of the scope of pelicun; it is either performed by the response estimation module in the Application Framework (Fig. 1) or it can be performed by any other application if pelicun is used outside of the scope of SimCenter. The pelicun framework can take advantage of response estimation methods that use idealized models for the seismic demand and the structural capacity, such as the capacity curve-based method in HAZUS or the regression-based closed-form approximation in the second edition of FEMA P58-5 [12]. If the performance assessment follows path I or II from :numref:`figPerfAssWorkflows`, the estimated response is not needed, and the relevant IM values are used as EDPs.

