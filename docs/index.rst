.. pelicun documentation master file, created by
   sphinx-quickstart on Sat Aug 18 15:35:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: figures/logo.PNG
	:scale: 50%
	:alt: logo

Probabilistic Estimation of Losses, Injuries, and Community resilience Under Natural disasters

*pelicun* is a Python package that provides tools for assessment of damage and losses due to natural hazards. It uses a stochastic damage and loss model that is based on the methodology described in FEMA P58 (FEMA, 2012). While FEMA P58 aims to assess the seismic performance of a building, with *pelicun* we want to develop a more versatile, hazard-agnostic tool that will eventually provide loss estimates for other types of assets (e.g. bridges, facilities, pipelines) and lifelines. The underlying loss model was designed with these objectives in mind and it will be gradually extended to have such functionality.

Currently, the scenario assessment from the FEMA P58 methodology is built-in the tool. Detailed documentation of the available methods and their use is available at http://pelicun.readthedocs.io

.. toctree::
   :maxdepth: 1

   overview
   installation
   features
   license
   API documentation <source/pelicun>

License
=======

*pelicun* is distributed under the BSD 3-Clause license, see LICENSE.

Contact
=======
Adam Zsarnóczay, NHERI SimCenter, Stanford University, adamzs@stanford.edu


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
