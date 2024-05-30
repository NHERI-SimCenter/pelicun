# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Kuanshi Zhong
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

import random
import numpy as np
import pandas as pd
import datetime
import math


def parse_BIM(BIM_in, location, hazards):
    """
    Parses the information provided in the AIM model.

    The parameters below list the expected inputs

    Parameters
    ----------
    stories: str
        Number of stories
    yearBuilt: str
        Year of construction.
    roofType: {'hip', 'hipped', 'gabled', 'gable', 'flat'}
        One of the listed roof shapes that best describes the building.
    occupancy: str
        Occupancy type.
    buildingDescription: str
        MODIV code that provides additional details about the building
    structType: {'Stucco', 'Frame', 'Stone', 'Brick'}
        One of the listed structure types that best describes the building.
    V_design: string
        Ultimate Design Wind Speed was introduced in the 2012 IBC. Officially
        called “Ultimate Design Wind Speed (Vult); equivalent to the design
        wind speeds taken from hazard maps in ASCE 7 or ATC's API. Unit is
        assumed to be mph.
    area: float
        Plan area in ft2.
    z0: string
        Roughness length that characterizes the surroundings.

    Returns
    -------
    BIM_ap: dictionary
        Parsed building characteristics.
    """

    # check location
    if location not in ['LA', 'NJ']:
        print(f'WARNING: The provided location is not recognized: {location}')

    # check hazard
    for hazard in hazards:
        if hazard not in ['wind', 'inundation']:
            print(f'WARNING: The provided hazard is not recognized: {hazard}')

    # initialize the BIM dict
    BIM_ap = BIM_in.copy()

    if 'wind' in hazards:

        # maps roof type to the internal representation
        ap_RoofType = {
            'hip': 'hip',
            'hipped': 'hip',
            'Hip': 'hip',
            'gabled': 'gab',
            'gable': 'gab',
            'Gable': 'gab',
            'flat': 'flt',
            'Flat': 'flt',
        }
        # maps roof system to the internal representation
        ap_RoofSyste = {'Wood': 'trs', 'OWSJ': 'ows', 'N/A': 'trs'}
        roof_system = BIM_in.get('RoofSystem', 'Wood')
        if pd.isna(roof_system):
            roof_system = 'Wood'

        # maps number of units to the internal representation
        ap_NoUnits = {
            'Single': 'sgl',
            'Multiple': 'mlt',
            'Multi': 'mlt',
            'nav': 'nav',
        }

        # maps for design level (Marginal Engineered is mapped to Engineered as default)
        ap_DesignLevel = {'E': 'E', 'NE': 'NE', 'PE': 'PE', 'ME': 'E'}
        design_level = BIM_in.get('DesignLevel', 'E')
        if pd.isna(design_level):
            design_level = 'E'

        # Average January Temp.
        ap_ajt = {'Above': 'above', 'Below': 'below'}

        # Year built
        alname_yearbuilt = ['YearBuiltNJDEP', 'yearBuilt', 'YearBuiltMODIV']
        yearbuilt = None
        try:
            yearbuilt = BIM_in['YearBuilt']
        except:
            for i in alname_yearbuilt:
                if i in BIM_in.keys():
                    yearbuilt = BIM_in[i]
                    break

        # if none of the above works, set a default
        if yearbuilt is None:
            yearbuilt = 1985

        # Number of Stories
        alname_nstories = [
            'stories',
            'NumberofStories0',
            'NumberofStories',
            'NumberofStories1',
        ]
        nstories = None
        try:
            nstories = BIM_in['NumberOfStories']
        except Exception as e:
            for i in alname_nstories:
                if i in BIM_in.keys():
                    nstories = BIM_in[i]
                    break

            # if none of the above works, we need to raise an exception
            if nstories is None:
                raise e from None

        # Plan Area
        alname_area = ['area', 'PlanArea1', 'Area', 'PlanArea0']
        area = None
        try:
            area = BIM_in['PlanArea']
        except Exception as e:
            for i in alname_area:
                if i in BIM_in.keys():
                    area = BIM_in[i]
                    break

            # if none of the above works, we need to raise an exception
            if area is None:
                raise e from None

        # Design Wind Speed
        alname_dws = ['DSWII', 'DWSII', 'DesignWindSpeed']

        dws = BIM_in.get('DesignWindSpeed', None)
        if dws is None:
            for alname in alname_dws:
                if alname in BIM_in.keys():
                    dws = BIM_in[alname]
                    break

        alname_occupancy = ['occupancy']
        oc = None
        try:
            oc = BIM_in['OccupancyClass']
        except Exception as e:
            for i in alname_occupancy:
                if i in BIM_in.keys():
                    oc = BIM_in[i]
                    break

            # if none of the above works, we need to raise an exception
            if oc is None:
                raise e from None

        # if getting RES3 then converting it to default RES3A
        if oc == 'RES3':
            oc = 'RES3A'

        # maps for flood zone
        ap_FloodZone = {
            # Coastal areas with a 1% or greater chance of flooding and an
            # additional hazard associated with storm waves.
            6101: 'VE',
            6102: 'VE',
            6103: 'AE',
            6104: 'AE',
            6105: 'AO',
            6106: 'AE',
            6107: 'AH',
            6108: 'AO',
            6109: 'A',
            6110: 'X',
            6111: 'X',
            6112: 'X',
            6113: 'OW',
            6114: 'D',
            6115: 'NA',
            6119: 'NA',
        }
        if type(BIM_in['FloodZone']) == int:
            # NJDEP code for flood zone (conversion to the FEMA designations)
            floodzone_fema = ap_FloodZone[BIM_in['FloodZone']]
        else:
            # standard input should follow the FEMA flood zone designations
            floodzone_fema = BIM_in['FloodZone']

        # maps for BuildingType
        ap_BuildingType_NJ = {
            # Coastal areas with a 1% or greater chance of flooding and an
            # additional hazard associated with storm waves.
            3001: 'Wood',
            3002: 'Steel',
            3003: 'Concrete',
            3004: 'Masonry',
            3005: 'Manufactured',
        }
        if location == 'NJ':
            # NJDEP code for flood zone needs to be converted
            buildingtype = ap_BuildingType_NJ[BIM_in['BuildingType']]
        elif location == 'LA':
            # standard input should provide the building type as a string
            buildingtype = BIM_in['BuildingType']

        # first, pull in the provided data
        BIM_ap.update(
            dict(
                OccupancyClass=str(oc),
                BuildingType=buildingtype,
                YearBuilt=int(yearbuilt),
                # double check with Tracy for format - (NumberStories0 is 4-digit code)
                # (NumberStories1 is image-processed story number)
                NumberOfStories=int(nstories),
                PlanArea=float(area),
                FloodZone=floodzone_fema,
                V_ult=float(dws),
                AvgJanTemp=ap_ajt[BIM_in.get('AvgJanTemp', 'Below')],
                RoofShape=ap_RoofType[BIM_in['RoofShape']],
                RoofSlope=float(BIM_in.get('RoofSlope', 0.25)),  # default 0.25
                SheathingThickness=float(
                    BIM_in.get('SheathingThick', 1.0)
                ),  # default 1.0
                RoofSystem=str(
                    ap_RoofSyste[roof_system]
                ),  # only valid for masonry structures
                Garage=float(BIM_in.get('Garage', -1.0)),
                LULC=BIM_in.get('LULC', -1),
                z0=float(
                    BIM_in.get('z0', -1)
                ),  # if the z0 is already in the input file
                Terrain=BIM_in.get('Terrain', -1),
                MeanRoofHt=float(BIM_in.get('MeanRoofHt', 15.0)),  # default 15
                DesignLevel=str(ap_DesignLevel[design_level]),  # default engineered
                WindowArea=float(BIM_in.get('WindowArea', 0.20)),
                WoodZone=str(BIM_in.get('WindZone', 'I')),
            )
        )

    if 'inundation' in hazards:

        # maps for split level
        ap_SplitLevel = {'NO': 0, 'YES': 1}

        foundation = BIM_in.get('FoundationType', 3501)
        if pd.isna(foundation):
            foundation = 3501

        nunits = BIM_in.get('NoUnits', 1)
        if pd.isna(nunits):
            nunits = 1

        # maps for flood zone
        ap_FloodZone = {
            # Coastal areas with a 1% or greater chance of flooding and an
            # additional hazard associated with storm waves.
            6101: 'VE',
            6102: 'VE',
            6103: 'AE',
            6104: 'AE',
            6105: 'AO',
            6106: 'AE',
            6107: 'AH',
            6108: 'AO',
            6109: 'A',
            6110: 'X',
            6111: 'X',
            6112: 'X',
            6113: 'OW',
            6114: 'D',
            6115: 'NA',
            6119: 'NA',
        }
        if type(BIM_in['FloodZone']) == int:
            # NJDEP code for flood zone (conversion to the FEMA designations)
            floodzone_fema = ap_FloodZone[BIM_in['FloodZone']]
        else:
            # standard input should follow the FEMA flood zone designations
            floodzone_fema = BIM_in['FloodZone']

        # add the parsed data to the BIM dict
        BIM_ap.update(
            dict(
                DesignLevel=str(ap_DesignLevel[design_level]),  # default engineered
                NumberOfUnits=int(nunits),
                FirstFloorElevation=float(BIM_in.get('FirstFloorHt1', 10.0)),
                SplitLevel=bool(
                    ap_SplitLevel[BIM_in.get('SplitLevel', 'NO')]
                ),  # dfault: no
                FoundationType=int(foundation),  # default: pile
                City=BIM_in.get('City', 'NA'),
                FloodZone=str(floodzone_fema),
            )
        )

    # add inferred, generic meta-variables

    if 'wind' in hazards:

        # Hurricane-Prone Region (HRP)
        # Areas vulnerable to hurricane, defined as the U.S. Atlantic Ocean and
        # Gulf of Mexico coasts where the ultimate design wind speed, V_ult is
        # greater than a pre-defined limit.
        if BIM_ap['YearBuilt'] >= 2016:
            # The limit is 115 mph in IRC 2015
            HPR = BIM_ap['V_ult'] > 115.0
        else:
            # The limit is 90 mph in IRC 2009 and earlier versions
            HPR = BIM_ap['V_ult'] > 90.0

        # Wind Borne Debris
        # Areas within hurricane-prone regions are affected by debris if one of
        # the following two conditions holds:
        # (1) Within 1 mile (1.61 km) of the coastal mean high water line where
        # the ultimate design wind speed is greater than flood_lim.
        # (2) In areas where the ultimate design wind speed is greater than
        # general_lim
        # The flood_lim and general_lim limits depend on the year of construction
        if BIM_ap['YearBuilt'] >= 2016:
            # In IRC 2015:
            flood_lim = 130.0  # mph
            general_lim = 140.0  # mph
        else:
            # In IRC 2009 and earlier versions
            flood_lim = 110.0  # mph
            general_lim = 120.0  # mph
        # Areas within hurricane-prone regions located in accordance with
        # one of the following:
        # (1) Within 1 mile (1.61 km) of the coastal mean high water line
        # where the ultimate design wind speed is 130 mph (58m/s) or greater.
        # (2) In areas where the ultimate design wind speed is 140 mph (63.5m/s)
        # or greater. (Definitions: Chapter 2, 2015 NJ Residential Code)
        if not HPR:
            WBD = False
        else:
            WBD = (
                (
                    BIM_ap['FloodZone'].startswith('A')
                    or BIM_ap['FloodZone'].startswith('V')
                )
                and BIM_ap['V_ult'] >= flood_lim
            ) or (BIM_ap['V_ult'] >= general_lim)

        # Terrain
        # open (0.03) = 3
        # light suburban (0.15) = 15
        # suburban (0.35) = 35
        # light trees (0.70) = 70
        # trees (1.00) = 100
        # Mapped to Land Use Categories in NJ (see https://www.state.nj.us/dep/gis/
        # digidownload/metadata/lulc02/anderson2002.html) by T. Wu group
        # (see internal report on roughness calculations, Table 4).
        # These are mapped to Hazus defintions as follows:
        # Open Water (5400s) with zo=0.01 and barren land (7600) with zo=0.04 assume Open
        # Open Space Developed, Low Intensity Developed, Medium Intensity Developed
        # (1110-1140) assumed zo=0.35-0.4 assume Suburban
        # High Intensity Developed (1600) with zo=0.6 assume Lt. Tree
        # Forests of all classes (4100-4300) assumed zo=0.6 assume Lt. Tree
        # Shrub (4400) with zo=0.06 assume Open
        # Grasslands, pastures and agricultural areas (2000 series) with
        # zo=0.1-0.15 assume Lt. Suburban
        # Woody Wetlands (6250) with zo=0.3 assume suburban
        # Emergent Herbaceous Wetlands (6240) with zo=0.03 assume Open
        # Note: HAZUS category of trees (1.00) does not apply to any LU/LC in NJ
        terrain = 15  # Default in Reorganized Rulesets - WIND
        LULC = BIM_ap['LULC']
        TER = BIM_ap['Terrain']
        if BIM_ap['z0'] > 0:
            terrain = int(100 * BIM_ap['z0'])
        elif LULC > 0:
            if BIM_ap['FloodZone'].startswith('V') or BIM_ap['FloodZone'] in [
                'A',
                'AE',
                'A1-30',
                'AR',
                'A99',
            ]:
                terrain = 3
            elif (LULC >= 5000) and (LULC <= 5999):
                terrain = 3  # Open
            elif ((LULC == 4400) or (LULC == 6240)) or (LULC == 7600):
                terrain = 3  # Open
            elif (LULC >= 2000) and (LULC <= 2999):
                terrain = 15  # Light suburban
            elif ((LULC >= 1110) and (LULC <= 1140)) or (
                (LULC >= 6250) and (LULC <= 6252)
            ):
                terrain = 35  # Suburban
            elif ((LULC >= 4100) and (LULC <= 4300)) or (LULC == 1600):
                terrain = 70  # light trees
        elif TER > 0:
            if BIM_ap['FloodZone'].startswith('V') or BIM_ap['FloodZone'] in [
                'A',
                'AE',
                'A1-30',
                'AR',
                'A99',
            ]:
                terrain = 3
            elif (TER >= 50) and (TER <= 59):
                terrain = 3  # Open
            elif ((TER == 44) or (TER == 62)) or (TER == 76):
                terrain = 3  # Open
            elif (TER >= 20) and (TER <= 29):
                terrain = 15  # Light suburban
            elif (TER == 11) or (TER == 61):
                terrain = 35  # Suburban
            elif ((TER >= 41) and (TER <= 43)) or (TER in [16, 17]):
                terrain = 70  # light trees

        BIM_ap.update(
            dict(
                # Nominal Design Wind Speed
                # Former term was “Basic Wind Speed”; it is now the “Nominal Design
                # Wind Speed (V_asd). Unit: mph."
                V_asd=np.sqrt(0.6 * BIM_ap['V_ult']),
                HazardProneRegion=HPR,
                WindBorneDebris=WBD,
                TerrainRoughness=terrain,
            )
        )

    if 'inundation' in hazards:

        BIM_ap.update(
            dict(
                # Flood Risk
                # Properties in the High Water Zone (within 1 mile of the coast) are at
                # risk of flooding and other wind-borne debris action.
                FloodRisk=True,  # TODO: need high water zone for this and move it to inputs!
            )
        )

    return BIM_ap
