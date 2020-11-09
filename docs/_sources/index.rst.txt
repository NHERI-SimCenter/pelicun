|full tool name|
=====================================================================

.. only:: html

   |developers|

   .. only:: RDT_app

      The |full tool name| (|app|) is an open-source research application that can be used to simulate the performance of the built environment in an urban region subjected to natural hazards. The application 
      predict the performance of a building subjected to earthquake events. [...] The computations are performed by a simulation workflow that will run on either the user's local machine or on a high performance computer made available by |DesignSafe|.

   .. only:: PBE_app

      The |full tool name| (|app|) is an open-source research application that can be used to predict the performance of a building subjected to earthquake events. The application is focused on quantifying building performance given the uncertainties in models, earthquake loads, and analysis. The computations are performed in a workflow application that will run on either the users local machine or on a high performance computer made available by |DesignSafe|. 

   .. only:: EEUQ_app

      The |full tool name| (|app|) is an open-source research application that can be used to predict the response of a building subjected to earthquake events. The application is focused on quantifying the uncertainties in the predicted response, given the that the uncertainties in models, earthquake loads, and analysis. The computations are performed in a workflow application that will run on either the users local machine or on a high performance computer made available by |DesignSafe|. 


   .. only:: WEUQ_app

      The |full tool name| (|app|) is an open-source research application that can be used to predict the response of a building subjected to wind loading events. The application is focused on quantifying the uncertainties in the predicted response, given the that the uncertainties in models, wind loads, and analysis. The computations are performed in a workflow application that will run on either the users local machine or on a high performance computer made available by |DesignSafe|. 


   .. only:: quoFEM_app

      The |full tool name|  is an open-source research application which focuses on providing uncertainty quantification methods (forward, inverse, reliability, sensitivity and parameter estimation) to researchers in natural hazards who utilize existing simulation software applications, typically Finite Element applications, in their work. The computations are performed in a workflow application that will run on either the users local machine or on a high performance computer made available by |DesignSafe|. 

   .. only:: pelicun

      The |full tool name| is an open-source implementation of the |short tool name| framework in a Python package. It provides tools for assessment of damage and losses due to natural disasters. Its basis is the stochastic damage and loss model described in FEMA P58. While FEMA P58 aims to assess the seismic performance of a building, with the |app| we provide a more versatile, hazard agnostic tool that can estimate losses for several types of assets in the built environment.


   This document covers the features and capabilities of Version |tool version| of the tool. Users are encouraged to comment on what additional features and capabilities they would like to see in future versions of the application through the |messageBoard|.


.. _lbl-user-manual:

.. toctree-filt::
   :caption: User Manual
   :maxdepth: 1
   :numbered: 4

   common/user_manual/ack

   :PBE:common/user_manual/about/PBE/about
   :EEUQ:common/user_manual/about/EEUQ/about
   :WEUQ:common/user_manual/about/WEUQ/about
   :quoFEM:common/user_manual/about/quoFEM/aboutQUOFEM
   :pelicun:common/user_manual/about/pelicun/about
   
   :desktop_app:common/user_manual/installation/desktop/installation
   :pelicun:common/user_manual/installation/pelicun/installation
   
   :desktop_app:common/user_manual/usage/desktop/usage
   :pelicun:common/user_manual/usage/pelicun/usage
   
   :desktop_app:common/user_manual/troubleshooting/desktop/troubleshooting
   :pelicun:common/user_manual/troubleshooting/pelicun/troubleshooting
   
   :desktop_app:common/user_manual/examples/desktop/examples
   :pelicun:common/user_manual/examples/pelicun/examples
   
   :EEUQ:common/requirements/EEUQ
   :WEUQ:common/requirements/WEUQ
   :PBE:common/requirements/PBE
   :pelicun:common/user_manual/requirements/pelicun/requirements
   
   common/user_manual/bugs
   common/user_manual/license


.. _lbl-testbeds-manual:

.. toctree-filt::
   :caption: Testbeds
   :maxdepth: 1
   :numbered: 3

   :RDT:common/testbeds/sf_bay_area/index
   :RDT:common/testbeds/atlantic_city/index   
   :RDT:common/testbeds/memphis/index
   :RDT:common/testbeds/anchorage/index
   :RDT:common/testbeds/lake_charles/index

.. _lbl-technical-manual:

.. toctree-filt::
   :caption: Technical Manual
   :maxdepth: 1
   :numbered: 2

   :desktop_app:common/technical_manual/desktop/technical_manual
   :pelicun:common/technical_manual/pelicun/technical_manual


.. _lbl-developer-manual:

.. toctree-filt::
   :caption: Developer Manual
   :maxdepth: 1
   :numbered: 4

   :desktop_app:common/developer_manual/how_to_build/desktop/how_to_build

   :desktop_app:common/developer_manual/architecture/desktop/architecture
   :pelicun:common/developer_manual/architecture/pelicun/architecture

   :desktop_app:common/developer_manual/how_to_extend/desktop/how_to_extend
   :pelicun:common/developer_manual/how_to_extend/pelicun/how_to_extend
   
   :desktop_app:common/developer_manual/verification/desktop/verification
   :pelicun:common/developer_manual/verification/pelicun/verification
   
   :desktop_app:common/developer_manual/coding_style/desktop/coding_style
   :pelicun:common/developer_manual/coding_style/pelicun/coding_style

   :pelicun:common/developer_manual/API/pelicun/API



Contact
=======

|contact person|

References
==========

.. bibliography:: common/references.bib
