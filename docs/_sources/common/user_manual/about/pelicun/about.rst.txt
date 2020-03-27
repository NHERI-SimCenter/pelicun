.. _lblAbout:

******
About
******

|short tool name| is an open-source library (|tool github link|) released under a **3-Clause BSD** license (see :numref:`lblLicense`). The |app| can be used to quantify losses from an earthquake or hurricane scenario in the form of *decision variables* (DVs). This functionality is typically utilized for performance-based engineering and regional natural hazard risk assessment. This library can help you in several steps of performance assessment:

* **Describe the joint distribution of asset response.** The response of a structure or other type of asset to an earthquake or hurricane wind is typically described by so-called engineering demand parameters (EDPs). |short tool name| provides methods that take a finite number of EDP vectors and find a multivariate distribution that describes the joint distribution of EDP data well. You can control the type of target distribution, apply truncation limits and censor part of the data to consider detection limits in your analysis. Alternatively, you can choose to use your EDP vectors as-is without resampling from a fitted distribution.

* **Define the performance model of an asset.** The fragility and consequence functions from the first two editions of FEMA P58 and the HAZUS earthquake and hurricane models for buildings are provided with |short tool name|. This makes it easy to define a performance model without having to collect and provide all the data manually. A stochastic damage and loss model is designed to facilitate modeling correlations throughout the simulation.

* **Estimate component damages.** Given the models for EDPs, component performance, and damage, |short tool name| provides methods to estimate the amount of damaged components and the proportion of realizations that resulted in collapse.

* **Estimate consequences.** Using information about collapse and component damages, the following consequences can be estimated with the loss model: reconstruction cost and time, unsafe placarding (red tag), injuries of various severity (depending on the applied method) and fatalities.