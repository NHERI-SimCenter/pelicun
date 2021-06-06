.. _lbl-testbed_Anchorage:

*************
Anchorage, AK
*************

On November 30th 2018 at 8.29am local time, a magnitude 7.1 earthquake occurred near Anchorage, Alaska. Originally the earthquake was estimated to be a magnitude 7.0, but was later on revised to magnitude 7.1. The epicenter of the earthquake was 10 miles away from the Anchorage metro area and the depth of the earthquake was 29 miles. For more details visit the `USGS event page <https://earthquake.usgs.gov/earthquakes/eventpage/ak20419010/executive>`_. A `short Youtube video <https://www.youtube.com/watch?v=faK6magPCJU>`_ shows the shaking and images of the damage. As of July 11, FEMA’s Individuals and Households program has approved more than `$21 million to repair disaster-damaged homes and pay for temporary housing. Additionally, the U.S. Small Business Administration has approved more than $70.3 million for 1,772 homeowners and renters; and more than $7.89 million for 112 businesses. <https://www.fema.gov/press-release/20200827/state-and-federal-disaster-assistance-2018-alaska-earthquake-tops-100>`_.

In the few weeks following the earthquake, the NHERI SimCenter team collected and processed building exposure data from the publicly available parcels tax data from the Municipality of Anchorage tax assessor's website. The tax data for approximately 97,000 parcels was processed, resulting in buildings data in 85,000 parcels. In addition the SimCenter collected a number of recorded motions for the event that were made available online.

Subsequently the SimCenter created a workflow to estimate the effect of the earthquake using the software created by the center for performing assessment of damage associated with natural events at the regional scale. This section describes the steps involved to collect the data, steps that a user would take to repeat the simulations performed by SimCenter, and finally the results obtained.

A video `presenting the work <https://www.youtube.com/watch?v=VH-slcnmTJc>`_ was presented by Wael Elhadded of SimCenter (now with NVIDIA).

Anchorage Data Gathering
========================

Building Data
-------------

The NHERI SimCenter team collected and processed building exposure data from the publicly available parcels tax data from the `Municipality of Anchorage tax assessor's website <https://www.muni.org/pages/default.aspx>`_. The tax data for approximately 97,000 parcels was processed, resulting in buildings data in 85,000 parcels, which can be broken down into two regions as follows:

- 73,000 parcels in Anchorage
- 12,000 parcels in Eagle River

The building data is provided online at DesignSafe `AnchorageBuildings.zip <https://www.designsafe-ci.org/data/browser/public/designsafe.storage.community/SimCenter/Datasets/AnchorageM7>`_. The zip file contains a single comma separated file. The first line contains the column headings, the subsequent lines the information for each building.


#. Id - an integer id

#. Area - floor plan area of building.

#. Stories - number of stories

#. Year Built - year built.

#. Type ID - ???

#. Latitude - latitude of building

#. Longitude - longitude of building

#. Occupancy - occupancy class of building

#. ParcelId - assessors parcel id.

The data was collected by a python program utilizing `selenium <https://www.selenium.dev/>`_, which provides a set of tools for automating web browsers. The program used to generate the data is as shown in figure below:

.. literalinclude:: code/getBuildingsFMK.py
   :language: py


.. note::

   the code uses a python model named ``geocoder`` to geoencode the latitudes and longitudes from the street addresses. geocoder can be set up to use a `number of provideres <https://github.com/DenisCarriere/geocoder>`_. The code shown above uses Open Street Maps, which is free but might not be efficient. The difference in service providers is in speed of service, some may be rate limited, and all can result in minor differences in lat and long values returned.


Ground Motion Data
------------------

The testbed used the following workflow:



Workflow
========


Preliminary regional loss estimation was carried out using rWHALE, a regional workflow for hazard and loss estimation, developed at NHERI SimCenter. Results for the regional simulation of Anchorage M7.1 earthquake can be summarized as follows:

Results
=======


Average building loss ratio is 14.5%
Total repair cost of $7.5 billion
3800 Red Tagged Buildings

Researchers interested in reproducing the same results are able to access rWHALE through DeisgnSafe-CI Workspace (requires logging in). Detailed steps needed to run a large scale regional simulation were presented in a NHERI SimCenter online webinar, and a recorded version is available online. Additional documentation and details are available in the SimCenter website.



