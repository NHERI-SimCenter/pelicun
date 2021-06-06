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
import datetime

def parse_BIM(BIM_in):
    """
    Parses the information provided in the BIM model.

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
    BIM: dictionary
        Parsed building characteristics.
    """

    # maps roof type to the internal representation
    ap_RoofType = {
        'hip'   : 'hip',
        'hipped': 'hip',
        'Hip'   : 'hip',
        'gabled': 'gab',
        'gable' : 'gab',
        'Gable' : 'gab',
        'flat'  : 'flt',
        'Flat'  : 'flt'
    }
    # maps roof system to the internal representation
    ap_RoofSyste = {
        'Wood': 'trs',
        'OWSJ': 'ows',
        'N/A': 'trs'
    }
    roof_system = BIM_in.get('RoofSystem','Wood')
    try:
        if np.isnan(roof_system):
            roof_system = 'Wood'
    except:
        pass
    # maps number of units to the internal representation
    ap_NoUnits = {
        'Single': 'sgl',
        'Multiple': 'mlt',
        'Multi': 'mlt',
        'nav': 'nav'
    }
    # maps for split level
    ap_SplitLevel = {
        'NO': 0,
        'YES': 1
    }
    # maps for design level (Marginal Engineered is mapped to Engineered as default)
    ap_DesignLevel = {
        'E': 'E',
        'NE': 'NE',
        'PE': 'PE',
        'ME': 'E'
    }
    design_level = BIM_in.get('DesignLevel','E')
    try:
        if np.isnan(design_level):
            design_level = 'E'
    except:
        pass

    foundation = BIM_in.get('FoundationType')
    if np.isnan(foundation):
        foundation = 3501

    nunits = BIM_in.get('NumberOfUnits',1)
    if np.isnan(nunits):
        nunits = 1

    # Average January Temp.
    ap_ajt = {
        'Above': 'above',
        'Below': 'below'
    }

    # Year built
    alname_yearbuilt = ['yearBuilt', 'YearBuiltMODIV', 'YearBuilt']
    try:
        yearbuilt = BIM_in['YearBuiltNJDEP']
    except:
        for i in alname_yearbuilt:
            if i in BIM_in.keys():
                yearbuilt = BIM_in[i]
                break
    print('yearbuilt = ', yearbuilt)

    # Number of Stories
    alname_nstories = ['stories', 'NumberofStories0', 'NumberOfStories']
    try:
        nstories = BIM_in['NumberofStories1']
    except:
        for i in alname_nstories:
            if i in BIM_in.keys():
                nstories = BIM_in[i]
                break

    # Plan Area
    alname_area = ['area', 'PlanArea1', 'PlanArea']
    try:
        area = BIM_in['PlanArea0']
    except:
        for i in alname_area:
            if i in BIM_in.keys():
                area = BIM_in[i]
                break

    # if getting RES3 only (without subclass) then converting it to default RES3A
    oc = BIM_in.get('occupancy','RES1')
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
        6119: 'NA'
    }
    if type(BIM_in['FloodZone']) == int:
        # NJDEP code for flood zone (conversion to the FEMA designations)
        floodzone_fema = ap_FloodZone[BIM_in['FloodZone']]
    else:
        # standard input should follow the FEMA flood zone designations
        floodzone_fema = BIM_in['FloodZone']

    # first, pull in the provided data
    BIM = dict(
        occupancy_class=str(oc),
        bldg_type=BIM_in['BuildingType'],
        year_built=int(yearbuilt),
        # double check with Tracey for format - (NumberStories0 is 4-digit code)
        # (NumberStories1 is image-processed story number)
        stories=int(nstories),
        area=float(area),
        flood_zone=floodzone_fema,
        V_ult=float(BIM_in['DesignWindSpeed']),
        avg_jan_temp=ap_ajt[BIM_in.get('AverageJanuaryTemperature','Below')],
        roof_shape=ap_RoofType[BIM_in['RoofShape']],
        roof_slope=float(BIM_in.get('RoofSlope',0.25)), # default 0.25
        sheathing_t=float(BIM_in.get('SheathingThick',1.0)), # default 1.0
        roof_system=str(ap_RoofSyste[roof_system]), # only valid for masonry structures
        garage_tag=float(BIM_in.get('Garage',-1.0)),
        lulc=BIM_in.get('LULC',-1),
        z0 = float(BIM_in.get('RoughnessLength',-1)), # if the z0 is already in the input file
        Terrain = BIM_in.get('Terrain',-1),
        mean_roof_height=float(BIM_in.get('MeanRoofHeight',15.0)), # default 15
        design_level=str(ap_DesignLevel[design_level]), # default engineered
        no_units=int(nunits),
        window_area=float(BIM_in.get('WindowArea',0.20)),
        first_floor_ht1=float(BIM_in.get('FirstFloorHeight',10.0)),
        split_level=bool(ap_SplitLevel[BIM_in.get('SplitLevel',0)]), # dfault: no
        fdtn_type=int(foundation), # default: pile
        city=BIM_in.get('City','NA'),
        wind_zone=str(BIM_in.get('WindZone', 'I'))
    )

    # add inferred, generic meta-variables

    # Hurricane-Prone Region (HRP)
    # Areas vulnerable to hurricane, defined as the U.S. Atlantic Ocean and
    # Gulf of Mexico coasts where the ultimate design wind speed, V_ult is
    # greater than a pre-defined limit.
    if BIM['year_built'] >= 2016:
        # The limit is 115 mph in IRC 2015
        HPR = BIM['V_ult'] > 115.0
    else:
        # The limit is 90 mph in IRC 2009 and earlier versions
        HPR = BIM['V_ult'] > 90.0

    # Wind Borne Debris
    # Areas within hurricane-prone regions are affected by debris if one of
    # the following two conditions holds:
    # (1) Within 1 mile (1.61 km) of the coastal mean high water line where
    # the ultimate design wind speed is greater than flood_lim.
    # (2) In areas where the ultimate design wind speed is greater than
    # general_lim
    # The flood_lim and general_lim limits depend on the year of construction
    if BIM['year_built'] >= 2016:
        # In IRC 2015:
        flood_lim = 130.0 # mph
        general_lim = 140.0 # mph
    else:
        # In IRC 2009 and earlier versions
        flood_lim = 110.0 # mph
        general_lim = 120.0 # mph
    # Areas within hurricane-prone regions located in accordance with
    # one of the following:
    # (1) Within 1 mile (1.61 km) of the coastal mean high water line
    # where the ultimate design wind speed is 130 mph (58m/s) or greater.
    # (2) In areas where the ultimate design wind speed is 140 mph (63.5m/s)
    # or greater. (Definitions: Chapter 2, 2015 NJ Residential Code)
    if not HPR:
        WBD = False
    else:
        WBD = ((((BIM['flood_zone'] >= 6101) and (BIM['flood_zone'] <= 6109)) and
                BIM['V_ult'] >= flood_lim) or (BIM['V_ult'] >= general_lim))


    BIM.update(dict(
        # Nominal Design Wind Speed
        # Former term was “Basic Wind Speed”; it is now the “Nominal Design
        # Wind Speed (V_asd). Unit: mph."
        V_asd = np.sqrt(0.6 * BIM['V_ult']),

        # Flood Risk
        # Properties in the High Water Zone (within 1 mile of the coast) are at
        # risk of flooding and other wind-borne debris action.
        flood_risk=True,  # TODO: need high water zone for this and move it to inputs!

        HPR=HPR,
        WBD=WBD,
    ))

    return BIM

