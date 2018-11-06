Overview
========

The current version of *pelicun* can be used to quantifiy lossess from an earthquake scenario in the form of *decision variables*. This functionality is typically utilized for performance based engineering or seismic risk assessment. There are several steps of seismic performance assessment that *pelcicun* can help with:

* **Describe the joint distribution of seismic response.** The response of a structure or other type of asset to an earthquake is typically described by so-called *engineering demand parameters* (EDPs). *pelicun* provides methods that take a finite number of EDP vectors and find a multivarite distribution that describes the joint distribution of EDP data well.

* **Define the damage and loss model of a building.** The component damage and loss data from FEMA P58 is provided with *pelicun*. This makes it easy to define building components without having to provide all the data manually. The stochastic damage and loss model is designed to facilitate modeling correlations between several parameters of the damage and loss model.

* **Estimate component damages.** Given a damage and loss model and the joint distribution of EDPs, *pelicun* provides methods to estimate the quantity of damaged components and collapses.

* **Estimate consequences.** Using information about collapses and component damages, the following consequences can be estimated with the loss model: reconstruction cost and time, unsafe placarding (red tag), injuries and fatalities. 