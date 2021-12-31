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
    terrain = 15 # Default in Reorganized Rulesets - WIND
    if (BIM['z0'] > 0):
        terrain = int(100 * BIM['z0'])
    elif (BIM['lulc'] > 0):
        if (BIM['flood_zone'].startswith('V') or BIM['flood_zone'] in ['A', 'AE', 'A1-30', 'AR', 'A99']):
            terrain = 3
        elif ((BIM['lulc'] >= 5000) and (BIM['lulc'] <= 5999)):
            terrain = 3 # Open
        elif ((BIM['lulc'] == 4400) or (BIM['lulc'] == 6240)) or (BIM['lulc'] == 7600):
            terrain = 3 # Open
        elif ((BIM['lulc'] >= 2000) and (BIM['lulc'] <= 2999)):
            terrain = 15 # Light suburban
        elif ((BIM['lulc'] >= 1110) and (BIM['lulc'] <= 1140)) or ((BIM['lulc'] >= 6250) and (BIM['lulc'] <= 6252)):
            terrain = 35 # Suburban
        elif ((BIM['lulc'] >= 4100) and (BIM['lulc'] <= 4300)) or (BIM['lulc'] == 1600):
            terrain = 70 # light trees
    elif (BIM['Terrain'] > 0):
        if (BIM['flood_zone'].startswith('V') or BIM['flood_zone'] in ['A', 'AE', 'A1-30', 'AR', 'A99']):
            terrain = 3
        elif ((BIM['Terrain'] >= 50) and (BIM['Terrain'] <= 59)):
            terrain = 3 # Open
        elif ((BIM['Terrain'] == 44) or (BIM['Terrain'] == 62)) or (BIM['Terrain'] == 76):
            terrain = 3 # Open
        elif ((BIM['Terrain'] >= 20) and (BIM['Terrain'] <= 29)):
            terrain = 15 # Light suburban
        elif (BIM['Terrain'] == 11) or (BIM['Terrain'] == 61):
            terrain = 35 # Suburban
        elif ((BIM['Terrain'] >= 41) and (BIM['Terrain'] <= 43)) or (BIM['Terrain'] in [16, 17]):
            terrain = 70 # light trees

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
        terrain=terrain,
    ))

    return BIM


def FL_config(BIM):
    """
    Rules to identify the flood vunerability category

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    config: str
        A string that identifies a specific configration within this buidling
        class.
    """
    year = BIM['year_built'] # just for the sake of brevity

    # Flood Type
    if BIM['flood_zone'] in [6105, 6108]:
        flood_type = 'raz' # Riverline/A-Zone
    elif BIM['flood_zone'] in [6103, 6104, 6106, 6107, 6109]:
        flood_type = 'cvz' # Costal-Zone
    elif BIM['flood_zone'] in [6101, 6102]:
        flood_type = 'cvz' # Costal-Zone
    else:
        flood_type = 'cvz' # Default

    # First Floor Elevation (FFE)
    if flood_type in ['raz', 'caz']:
        FFE = BIM['first_floor_ht1']
    else:
        FFE = BIM['first_floor_ht1'] - 1.0

    # PostFIRM
    PostFIRM = False # Default
    city_list = ['Absecon', 'Atlantic', 'Brigantine', 'Buena', 'Buena Vista',
                 'Corbin City', 'Egg Harbor City', 'Egg Harbor', 'Estell Manor',
                 'Folsom', 'Galloway', 'Hamilton', 'Hammonton', 'Linwood',
                 'Longport', 'Margate City', 'Mullica', 'Northfield',
                 'Pleasantville', 'Port Republic', 'Somers Point',
                 'Ventnor City', 'Weymouth']
    year_list = [1976, 1971, 1971, 1983, 1979, 1981, 1982, 1983, 1978, 1982,
                 1983, 1977, 1982, 1983, 1974, 1974, 1982, 1979, 1983, 1983,
                 1982, 1971, 1979]
    for i in range(0,22):
        PostFIRM = (((BIM['city'] == city_list[i]) and (year > year_list[i])) or \
                    PostFIRM)

    # Basement Type
    if BIM['split_level'] and (BIM['fdtn_type'] == 3504):
        bmt_type = 'spt' # Split-Level Basement
    elif BIM['fdtn_type'] in [3501, 3502, 3503, 3505, 3506, 3507]:
        bmt_type = 'bn' # No Basement
    elif (not BIM['split_level']) and (BIM['fdtn_type'] == 3504):
        bmt_type = 'bw' # Basement
    else:
        bmt_type = 'bw' # Default

    # Duration
    dur = 'short'

    # Occupancy Type
    if BIM['occupancy_class'] == 'RES1':
        if BIM['stories'] == 1:
            if flood_type == 'raz':
                OT = 'SF1XA'
            elif flood_type == 'cvz':
                OT = 'SF1XV'
        else:
            if bmt_type == 'nav':
                if flood_type == 'raz':
                    OT = 'SF2XA'
                elif flood_type == 'cvz':
                    OT = 'SF2XV'
            elif bmt_type == 'bmt':
                if flood_type == 'raz':
                    OT = 'SF2BA'
                elif flood_type == 'cvz':
                    OT = 'SF2BV'
            elif bmt_type == 'spt':
                if flood_type == 'raz':
                    OT = 'SF2SA'
                elif flood_type == 'cvz':
                    OT = 'SF2SV'
    elif 'RES3' in BIM['occupancy_class']:
        OT = 'APT'
    else:
        ap_OT = {
            'RES2': 'MH',
            'RES4': 'HOT',
            'RES5': 'NURSE',
            'RES6': 'NURSE',
            'COM1': 'RETAL',
            'COM2': 'WHOLE',
            'COM3': 'SERVICE',
            'COM4': 'OFFICE',
            'COM5': 'BANK',
            'COM6': 'HOSP',
            'COM7': 'MED',
            'COM8': 'REC',
            'COM9': 'THEAT',
            'COM10': 'GARAGE',
            'IND1': 'INDH',
            'IND2': 'INDL',
            'IND3': 'CHEM',
            'IND4': 'PROC',
            'IND5': 'CHEM',
            'IND6': 'CONST',
            'AGR1': 'AGRI',
            'REL1': 'RELIG',
            'GOV1': 'CITY',
            'GOV2': 'EMERG',
            'EDU1': 'SCHOOL',
            'EDU2': 'SCHOOL'
        }
        ap_OT[BIM['occupancy_class']]


    if not (BIM['occupancy_class'] in ['RES1', 'RES2']):
        if 'RES3' in BIM['occupancy_class']:
            fl_config = f"{'fl'}_" \
                        f"{'RES3'}"
        else:
            fl_config = f"{'fl'}_" \
                        f"{BIM['occupancy_class']}"
    elif BIM['occupancy_class'] == 'RES2':
        fl_config = f"{'fl'}_" \
                    f"{BIM['occupancy_class']}_" \
                    f"{flood_type}"
    else:
        if bmt_type == 'spt':
            fl_config = f"{'fl'}_" \
                        f"{BIM['occupancy_class']}_" \
                        f"{'sl'}_" \
                        f"{'bw'}_" \
                        f"{flood_type}"
        else:
            st = 's'+str(np.min([BIM['stories'],3]))
            fl_config = f"{'fl'}_" \
                        f"{BIM['occupancy_class']}_" \
                        f"{st}_" \
                        f"{bmt_type}_" \
                        f"{flood_type}"

    return fl_config


def auto_populate(BIM):
    """
    Populates the DL model for hurricane assessments in Atlantic County, NJ

    Assumptions:
    - Everything relevant to auto-population is provided in the Buiding
    Information Model (BIM).
    - The information expected in the BIM file is described in the parse_BIM
    method.

    Parameters
    ----------
    BIM_in: dictionary
        Contains the information that is available about the asset and will be
        used to auto-popualate the damage and loss model.

    Returns
    -------
    BIM_ap: dictionary
        Containes the extended BIM data.
    DL_ap: dictionary
        Contains the auto-populated loss model.
    """

    # parse the BIM data
    BIM_ap = parse_BIM(BIM)

    # prepare the flood rulesets
    fld_config = FL_config(BIM_ap)

    DL_ap = {
        '_method'      : 'HAZUS MH HU',
        'LossModel'    : {
            'DecisionVariables': {
                "ReconstructionCost": True
            },
            'ReplacementCost'  : 100
        },
        'Components'   : {
            fld_config: [{
                'location'       : '1',
                'direction'      : '1',
                'median_quantity': '1.0',
                'unit'           : 'ea',
                'distribution'   : 'N/A'
            }]
        }
    }

    return BIM_ap, DL_ap
