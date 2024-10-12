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

from __future__ import annotations

import numpy as np


def parse_BIM(bim_in: dict, location: str, hazards: list[str]) -> dict:  # noqa: C901
    """
    Parses the information provided in the BIM model.

    The attributes below list the expected metadata in the BIM file

    Parameters
    ----------
    location: str
        Supported locations:
        NJ - New Jersey
        LA - Louisiana
    hazard: list of str
        Supported hazard types: "wind", "inundation"

    BIM attributes
    --------------
    NumberOfStories: str
        Number of stories
    YearBuilt: str
        Year of construction.
    RoofShape: {'hip', 'hipped', 'gabled', 'gable', 'flat'}
        One of the listed roof shapes that best describes the building.
    OccupancyType: str
        Occupancy type.
    BuildingType: str
        Core construction material type
    DWSII: float
        Design wind speed II as per ASCE 7 in mph
    Area: float
        Plan area in ft2.
    LULC: integer
        Land Use/Land Cover category (typically location-specific)

    Returns
    -------
    BIM: dictionary
        Parsed building characteristics.

    Raises
    ------
    KeyError
      In case of missing attributes.

    """
    # check location
    if location not in {'LA', 'NJ'}:
        print(f'WARNING: The provided location is not recognized: {location}')  # noqa: T201

    # check hazard
    for hazard in hazards:
        if hazard not in {'wind', 'inundation'}:
            print(f'WARNING: The provided hazard is not recognized: {hazard}')  # noqa: T201

    # initialize the BIM dict
    bim = {}

    if 'wind' in hazards:
        # maps roof type to the internal representation
        ap_roof_type = {
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
        ap_roof_system = {'Wood': 'trs', 'OWSJ': 'ows', 'N/A': 'trs'}
        roof_system = bim_in.get('RoofSystem', 'Wood')

        # flake8 - unused variable: `ap_NoUnits`.
        # # maps number of units to the internal representation
        # ap_NoUnits = {
        #     'Single': 'sgl',
        #     'Multiple': 'mlt',
        #     'Multi': 'mlt',
        #     'nav': 'nav',
        # }

        # Average January Temp.
        ap_ajt = {'Above': 'above', 'Below': 'below'}

        # Year built
        alname_yearbuilt = ['yearBuilt', 'YearBuiltMODIV', 'YearBuiltNJDEP']

        yearbuilt = bim_in.get('YearBuilt')

        # if none of the above works, set a default
        if yearbuilt is None:
            for alname in alname_yearbuilt:
                if alname in bim_in:
                    yearbuilt = bim_in[alname]
                    break

        if yearbuilt is None:
            yearbuilt = 1985

        # Number of Stories
        alname_nstories = [
            'stories',
            'NumberofStories0',
            'NumberofStories',
            'NumberofStories1',
        ]

        nstories = bim_in.get('NumberOfStories')

        if nstories is None:
            for alname in alname_nstories:
                if alname in bim_in:
                    nstories = bim_in[alname]
                    break

        if nstories is None:
            msg = 'NumberOfStories attribute missing, cannot autopopulate'
            raise KeyError(msg)

        # Plan Area
        alname_area = ['area', 'PlanArea1', 'Area', 'PlanArea0']

        area = bim_in.get('PlanArea')

        if area is None:
            for alname in alname_area:
                if alname in bim_in:
                    area = bim_in[alname]
                    break

        if area is None:
            msg = 'PlanArea attribute missing, cannot autopopulate'
            raise KeyError(msg)

        # Design Wind Speed
        alname_dws = ['DWSII', 'DesignWindSpeed']

        dws = bim_in.get('DesignWindSpeed')

        if dws is None:
            for alname in alname_dws:
                if alname in bim_in:
                    dws = bim_in[alname]
                    break

        if dws is None:
            msg = 'DesignWindSpeed attribute missing, cannot autopopulate'
            raise KeyError(msg)

        # occupancy type
        alname_occupancy = ['occupancy', 'OccupancyClass']

        oc = bim_in.get('OccupancyClass')

        if oc is None:
            for alname in alname_occupancy:
                if alname in bim_in:
                    oc = bim_in[alname]
                    break

        if oc is None:
            msg = 'OccupancyClass attribute missing, cannot autopopulate'
            raise KeyError(msg)

        # if getting RES3 then converting it to default RES3A
        if oc == 'RES3':
            oc = 'RES3A'

        # maps for BuildingType
        ap_building_type_nj = {
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
            buildingtype = ap_building_type_nj[bim_in['BuildingType']]

        elif location == 'LA':
            # standard input should provide the building type as a string
            buildingtype = bim_in['BuildingType']

        # maps for design level (Marginal Engineered is mapped to
        # Engineered as defauplt)
        ap_design_level = {'E': 'E', 'NE': 'NE', 'PE': 'PE', 'ME': 'E'}
        design_level = bim_in.get('DesignLevel', 'E')

        # flood zone
        flood_zone = bim_in.get('FloodZone', 'X')

        # add the parsed data to the BIM dict
        bim.update(
            {
                'OccupancyClass': str(oc),
                'BuildingType': buildingtype,
                'YearBuilt': int(yearbuilt),
                'NumberOfStories': int(nstories),
                'PlanArea': float(area),
                'V_ult': float(dws),
                'AvgJanTemp': ap_ajt[bim_in.get('AvgJanTemp', 'Below')],
                'RoofShape': ap_roof_type[bim_in['RoofShape']],
                'RoofSlope': float(bim_in.get('RoofSlope', 0.25)),  # default 0.25
                'SheathingThickness': float(
                    bim_in.get('SheathingThick', 1.0)
                ),  # default 1.0
                'RoofSystem': str(
                    ap_roof_system[roof_system]
                ),  # only valid for masonry structures
                'Garage': float(bim_in.get('Garage', -1.0)),
                'LULC': bim_in.get('LULC', -1),
                'MeanRoofHt': float(bim_in.get('MeanRoofHt', 15.0)),  # default 15
                'WindowArea': float(bim_in.get('WindowArea', 0.20)),
                'WindZone': str(bim_in.get('WindZone', 'I')),
                'FloodZone': str(flood_zone),
            }
        )

    if 'inundation' in hazards:
        # maps for split level
        ap_split_level = {'NO': 0, 'YES': 1}

        # foundation type
        foundation = bim_in.get('FoundationType', 3501)

        # number of units
        nunits = bim_in.get('NoUnits', 1)

        # flake8 - unused variable: `ap_FloodZone`.
        # # maps for flood zone
        # ap_FloodZone = {
        #     # Coastal areas with a 1% or greater chance of flooding and an
        #     # additional hazard associated with storm waves.
        #     6101: 'VE',
        #     6102: 'VE',
        #     6103: 'AE',
        #     6104: 'AE',
        #     6105: 'AO',
        #     6106: 'AE',
        #     6107: 'AH',
        #     6108: 'AO',
        #     6109: 'A',
        #     6110: 'X',
        #     6111: 'X',
        #     6112: 'X',
        #     6113: 'OW',
        #     6114: 'D',
        #     6115: 'NA',
        #     6119: 'NA',
        # }

        # flake8 - unused variable: `floodzone_fema`.
        # if isinstance(BIM_in['FloodZone'], int):
        #     # NJDEP code for flood zone (conversion to the FEMA designations)
        #     floodzone_fema = ap_FloodZone[BIM_in['FloodZone']]
        # else:
        #     # standard input should follow the FEMA flood zone designations
        #     floodzone_fema = BIM_in['FloodZone']

        # add the parsed data to the BIM dict
        bim.update(
            {
                'DesignLevel': str(
                    ap_design_level[design_level]
                ),  # default engineered
                'NumberOfUnits': int(nunits),
                'FirstFloorElevation': float(bim_in.get('FirstFloorHt1', 10.0)),
                'SplitLevel': bool(
                    ap_split_level[bim_in.get('SplitLevel', 'NO')]
                ),  # dfault: no
                'FoundationType': int(foundation),  # default: pile
                'City': bim_in.get('City', 'NA'),
            }
        )

    # add inferred, generic meta-variables

    if 'wind' in hazards:
        # Hurricane-Prone Region (HRP)
        # Areas vulnerable to hurricane, defined as the U.S. Atlantic Ocean and
        # Gulf of Mexico coasts where the ultimate design wind speed, V_ult is
        # greater than a pre-defined limit.
        if bim['YearBuilt'] >= 2016:
            # The limit is 115 mph in IRC 2015
            hpr = bim['V_ult'] > 115.0
        else:
            # The limit is 90 mph in IRC 2009 and earlier versions
            hpr = bim['V_ult'] > 90.0

        # Wind Borne Debris
        # Areas within hurricane-prone regions are affected by debris if one of
        # the following two conditions holds:
        # (1) Within 1 mile (1.61 km) of the coastal mean high water line where
        # the ultimate design wind speed is greater than flood_lim.
        # (2) In areas where the ultimate design wind speed is greater than
        # general_lim
        # The flood_lim and general_lim limits depend on the year of construction
        if bim['YearBuilt'] >= 2016:
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
        if not hpr:
            wbd = False
        else:
            wbd = (
                (
                    bim['FloodZone'].startswith('A')
                    or bim['FloodZone'].startswith('V')
                )
                and bim['V_ult'] >= flood_lim
            ) or (bim['V_ult'] >= general_lim)

        # Terrain
        # open (0.03) = 3
        # light suburban (0.15) = 15
        # suburban (0.35) = 35
        # light trees (0.70) = 70
        # trees (1.00) = 100
        # Mapped to Land Use Categories in NJ (see
        # https://www.state.nj.us/dep/gis/
        # digidownload/metadata/lulc02/anderson2002.html) by T. Wu
        # group (see internal report on roughness calculations, Table
        # 4).  These are mapped to Hazus definitions as follows: Open
        # Water (5400s) with zo=0.01 and barren land (7600) with
        # zo=0.04 assume Open Open Space Developed, Low Intensity
        # Developed, Medium Intensity Developed (1110-1140) assumed
        # zo=0.35-0.4 assume Suburban High Intensity Developed (1600)
        # with zo=0.6 assume Lt. Tree Forests of all classes
        # (4100-4300) assumed zo=0.6 assume Lt. Tree Shrub (4400) with
        # zo=0.06 assume Open Grasslands, pastures and agricultural
        # areas (2000 series) with zo=0.1-0.15 assume Lt. Suburban
        # Woody Wetlands (6250) with zo=0.3 assume suburban Emergent
        # Herbaceous Wetlands (6240) with zo=0.03 assume Open
        # Note: HAZUS category of trees (1.00) does not apply to any
        # LU/LC in NJ
        terrain = 15  # Default in Reorganized Rulesets - WIND
        if location == 'NJ':
            if bim['FloodZone'].startswith('V') or bim['FloodZone'] in {
                'A',
                'AE',
                'A1-30',
                'AR',
                'A99',
            }:
                terrain = 3
            elif ((bim['LULC'] >= 5000) and (bim['LULC'] <= 5999)) or (
                ((bim['LULC'] == 4400) or (bim['LULC'] == 6240))
                or (bim['LULC'] == 7600)
            ):
                terrain = 3  # Open
            elif (bim['LULC'] >= 2000) and (bim['LULC'] <= 2999):
                terrain = 15  # Light suburban
            elif ((bim['LULC'] >= 1110) and (bim['LULC'] <= 1140)) or (
                (bim['LULC'] >= 6250) and (bim['LULC'] <= 6252)
            ):
                terrain = 35  # Suburban
            elif ((bim['LULC'] >= 4100) and (bim['LULC'] <= 4300)) or (
                bim['LULC'] == 1600
            ):
                terrain = 70  # light trees
        elif location == 'LA':
            if bim['FloodZone'].startswith('V') or bim['FloodZone'] in {
                'A',
                'AE',
                'A1-30',
                'AR',
                'A99',
            }:
                terrain = 3
            elif ((bim['LULC'] >= 50) and (bim['LULC'] <= 59)) or (
                ((bim['LULC'] == 44) or (bim['LULC'] == 62)) or (bim['LULC'] == 76)
            ):
                terrain = 3  # Open
            elif (bim['LULC'] >= 20) and (bim['LULC'] <= 29):
                terrain = 15  # Light suburban
            elif (bim['LULC'] == 11) or (bim['LULC'] == 61):
                terrain = 35  # Suburban
            elif ((bim['LULC'] >= 41) and (bim['LULC'] <= 43)) or (
                bim['LULC'] in {16, 17}
            ):
                terrain = 70  # light trees

        bim.update(
            {
                # Nominal Design Wind Speed
                # Former term was “Basic Wind Speed”; it is now the “Nominal Design
                # Wind Speed (V_asd). Unit: mph."
                'V_asd': np.sqrt(0.6 * bim['V_ult']),
                'HazardProneRegion': hpr,
                'WindBorneDebris': wbd,
                'TerrainRoughness': terrain,
            }
        )

    if 'inundation' in hazards:
        bim.update(
            {
                # Flood Risk
                # Properties in the High Water Zone (within 1 mile of
                # the coast) are at risk of flooding and other
                # wind-borne debris action.
                # TODO: need high water zone for this and move it to inputs!  # noqa: TD002
                'FloodRisk': True,
            }
        )

    return bim
