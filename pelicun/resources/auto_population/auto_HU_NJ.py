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

    # Average January Temp.
    ap_ajt = {
        'Above': 'above',
        'Below': 'below'
    }

    # first, pull in the provided data
    BIM = dict(
        occupancy_class=str(BIM_in.get('OccupancyClass','RES1')),
        bldg_type=BIM_in['BuildingType'],
        year_built=int(BIM_in['YearBuiltNJDEP']),
        # double check with Tracey for format - (NumberStories0 is 4-digit code)
        # (NumberStories1 is image-processed story number)
        stories=int(BIM_in['NumberofStories1']),
        area=BIM_in['PlanArea0'],
        flood_zone=BIM_in['FloodZone'],
        V_ult=float(BIM_in['DSWII']),
        avg_jan_temp=ap_ajt[BIM_in.get('AvgJanTemp','Below')],
        roof_shape=ap_RoofType[BIM_in['RoofShape']],
        roof_slope=float(BIM_in.get('RoofSlope',0.25)), # default 0.25
        sheathing_t=float(BIM_in.get('SheathingThick',1.0)), # default 1.0
        roof_system=str(ap_RoofSyste[roof_system]), # only valid for masonry structures
        garage_tag=float(BIM_in.get('Garage',-1.0)),
        lulc=BIM_in.get('LULC',-1),
        z0 = float(BIM_in.get('z0',-1)), # if the z0 is already in the input file
        mean_roof_height=float(BIM_in.get('MeanRoofHt',15.0)), # default 15
        design_level=str(ap_DesignLevel[design_level]), # default engineered
        no_units=int(BIM_in.get('NoUnits')),
        window_area=float(BIM_in.get('WindowArea',0.20)),
        first_floor_ht1=float(BIM_in.get('FirstFloorHt1',10.0)),
        split_level=bool(ap_SplitLevel[BIM_in.get('SplitLevel',0)]), # dfault: no
        fdtn_type=int(BIM_in.get('FoundationType',3501)), # default: pile
        city=BIM_in['City']
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
    else:
        if BIM['flood_zone'] in [6101, 6102, 6104, 6106, 6109]:
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


def building_class(BIM):
    """
    Short description

    Long description

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    bldg_class: str
        One of the standard building class labels from HAZUS
    """

    if BIM['bldg_type'] == 3001:
        if ((BIM['occupancy_class'] == 'RES1') or
            ((BIM['roof_shape'] != 'flt') and (BIM['occupancy_class'] == ''))):
            # BuildingType = 3001
            # OccupancyClass = RES1
            # Wood Single-Family Homes (WSF1 or WSF2)
            # OR roof type = flat (HAZUS can only map flat to WSF1)
            # OR default (by '')
            if BIM['roof_shape'] == 'flt': # checking if there is a misclassication
                BIM['roof_shape'] = 'gab' # ensure the WSF has gab (by default, note gab is more vulneable than hip)
            return 'WSF'
        else:
            # BuildingType = 3001
            # OccupancyClass = RES3, RES5, RES6, or COM8
            # Wood Multi-Unit Hotel (WMUH1, WMUH2, or WMUH3)
            return 'WMUH'
    elif BIM['bldg_type'] == 3002:
        if ((BIM['design_level'] == 'E') and
            (BIM['occupancy_class'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                            'RES3E', 'RES3F'])):
            # BuildingType = 3002
            # Steel Engineered Residential Building (SERBL, SERBM, SERBH)
            return 'SERB'
        elif ((BIM['design_level'] == 'E') and
              (BIM['occupancy_class'] in ['COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                                          'COM6', 'COM7', 'COM8', 'COM9','COM10'])):
            # BuildingType = 3002
            # Steel Engineered Commercial Building (SECBL, SECBM, SECBH)
            return 'SECB'
        elif ((BIM['design_level'] == 'PE') and
              (BIM['occupancy_class'] not in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                          'RES3E', 'RES3F'])):
            # BuildingType = 3002
            # Steel Pre-Engineered Metal Building (SPMBS, SPMBM, SPMBL)
            return 'SPMB'
        else:
            return 'SECB'
    elif BIM['bldg_type'] == 3003:
        if ((BIM['design_level'] == 'E') and
            (BIM['occupancy_class'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                         'RES3E', 'RES3F', 'RES5', 'RES6'])):
            # BuildingType = 3003
            # Concrete Engineered Residential Building (CERBL, CERBM, CERBH)
            return 'CERB'
        elif ((BIM['design_level'] == 'E') and
              (BIM['occupancy_class'] in ['COM1', 'COM2', 'COM3', 'COM4', 'COM5',
                                          'COM6', 'COM7', 'COM8', 'COM9','COM10'])):
            # BuildingType = 3003
            # Concrete Engineered Commercial Building (CECBL, CECBM, CECBH)
            return 'CECB'
        else:
            return 'CECB'
    elif BIM['bldg_type'] == 3004:
        if BIM['occupancy_class'] == 'RES1':
            # BuildingType = 3004
            # OccupancyClass = RES1
            # Masonry Single-Family Homes (MSF1 or MSF2)
            return 'MSF'
        elif ((BIM['occupancy_class'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                        'RES3E', 'RES3F']) and (BIM['design_level'] == 'E')):
            # BuildingType = 3004
            # Masonry Engineered Residential Building (MERBL, MERBM, MERBH)
            return 'MERB'
        elif ((BIM['occupancy_class'] in ['COM1', 'COM2', 'COM3', 'COM4',
                                        'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                                        'COM10']) and (BIM['design_level'] == 'E')):
            # BuildingType = 3004
            # Masonry Engineered Commercial Building (MECBL, MECBM, MECBH)
            return 'MECB'
        elif BIM['occupancy_class'] in ['IND1', 'IND2', 'IND3', 'IND4', 'IND5', 'IND6']:
            # BuildingType = 3004
            # Masonry Low-Rise Masonry Warehouse/Factory (MLRI)
            return 'MLRI'
        elif BIM['occupancy_class'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
                                        'RES3E', 'RES3F', 'RES5', 'RES6', 'COM8']:
            # BuildingType = 3004
            # OccupancyClass = RES3X or COM8
            # Masonry Multi-Unit Hotel/Motel (MMUH1, MMUH2, or MMUH3)
            return 'MMUH'
        elif ((BIM['stories'] == 1) and
                (BIM['occupancy_class'] in ['COM1', 'COM2'])):
            # BuildingType = 3004
            # Low-Rise Masonry Strip Mall (MLRM1 or MLRM2)
            return 'MLRM'
        else:
            return 'MECB' # for others not covered by the above
        #elif ((BIM['occupancy_class'] in ['RES3A', 'RES3B', 'RES3C', 'RES3D',
        #                                'RES3E', 'RES3F', 'RES5', 'RES6',
        #                                'COM8']) and (BIM['design_level'] in ['NE', 'ME'])):
        #    # BuildingType = 3004
        #    # Masonry Multi-Unit Hotel/Motel Non-Engineered
        #    # (MMUH1NE, MMUH2NE, or MMUH3NE)
        #    return 'MMUHNE'



def WSF_config(BIM):
    """
    Rules to identify a HAZUS WSF configuration based on BIM data

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

    # Secondary Water Resistance (SWR)
    # Minimum drainage recommendations are in place in NJ (See below).
    # However, SWR indicates a code-plus practice.
    SWR = False # Default in Reorganzied Rulesets - WIND
    if year > 2000:
        # For buildings built after 2000, SWR is based on homeowner compliance
        # data from NC Coastal Homeowner Survey (2017) to capture potential
        # human behavior (% of sealed roofs in NC dataset).
        SWR = random.random() < 0.6
    elif year > 1983:
        # CABO 1995:
        # According to 903.2 in the 1995 CABO, for roofs with slopes between
        # 2:12 and 4:12, an underlayment consisting of two layers of No. 15
        # felt must be applied. In severe climates (less than or equal to 25
        # degrees Fahrenheit average in January), these two layers must be
        # cemented together.
        # According to 903.3 in the 1995 CABO, roofs with slopes greater than
        # or equal to 4:12 shall have an underlayment of not less than one ply
        # of No. 15 felt.
        #
        # Similar rules are prescribed in CABO 1992, 1989, 1986, 1983
        #
        # Since low-slope roofs require two layers of felt, this is taken to
        # be secondary water resistance. This ruleset is for asphalt shingles.
        # Almost all other roof types require underlayment of some sort, but
        # the ruleset is based on asphalt shingles because it is most
        # conservative.
        if BIM['roof_shape'] == 'flt': # note there is actually no 'flt'
            SWR = True
        elif BIM['roof_shape'] in ['gab','hip']:
            if BIM['roof_slope'] <= 0.17:
                SWR = True
            elif BIM['roof_slope'] < 0.33:
                SWR = (BIM['avg_jan_temp'] == 'below')

    # Roof Deck Attachment (RDA)
    # IRC codes:
    # NJ code requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
    # between ⅜”-1” -  see Table R602.3(1)
    # Fastener selection is contingent on thickness of sheathing in building
    # codes. Commentary for Table R602.3(1) indicates 8d nails with 6”/6”
    # spacing (enhanced roof spacing) for ultimate wind speeds greater than
    # a speed_lim. speed_lim depends on the year of construction
    RDA = '6d' # Default (aka A) in Reorganized Rulesets - WIND
    if year > 2000:
        if year >= 2016:
            # IRC 2015
            speed_lim = 130.0 # mph
        else:
            # IRC 2000 - 2009
            speed_lim = 100.0 # mph
        if BIM['V_ult'] > speed_lim:
            RDA = '8s'  # 8d @ 6"/6" ('D' in the Reorganized Rulesets - WIND)
        else:
            RDA = '8d'  # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)
    elif year > 1995:
        if ((BIM['sheathing_t'] >= 0.3125) and (BIM['sheathing_t'] <= 0.5)):
            RDA = '6d' # 6d @ 6"/12" ('A' in the Reorganized Rulesets - WIND)
        elif ((BIM['sheathing_t'] >= 0.59375) and (BIM['sheathing_t'] <= 1.125)):
            RDA = '8d' # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)
    elif year > 1986:
        if ((BIM['sheathing_t'] >= 0.3125) and (BIM['sheathing_t'] <= 0.5)):
            RDA = '6d' # 6d @ 6"/12" ('A' in the Reorganized Rulesets - WIND)
        elif ((BIM['sheathing_t'] >= 0.59375) and (BIM['sheathing_t'] <= 1.0)):
            RDA = '8d' # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)
    else:
        # year <= 1986
        if ((BIM['sheathing_t'] >= 0.3125) and (BIM['sheathing_t'] <= 0.5)):
            RDA = '6d' # 6d @ 6"/12" ('A' in the Reorganized Rulesets - WIND)
        elif ((BIM['sheathing_t'] >= 0.625) and (BIM['sheathing_t'] <= 1.0)):
            RDA = '8d' # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)

    # Roof-Wall Connection (RWC)
    # IRC 2015
    # "Assume all homes not having wind speed consideration are Toe Nail
    # (regardless of year)
    # For homes with wind speed consideration, 2015 IRC Section R802.11: no
    # specific connection type, must resist uplift forces using various
    # guidance documents, e.g., straps would be required (based on WFCM 2015);
    # will assume that if classified as HPR, then enhanced connection would be
    # used.
    if year > 2015:
        if BIM['HPR']:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail
    # IRC 2000-2009
    # In Section R802.11.1 Uplift Resistance of the NJ 2009 IRC, roof
    # assemblies which are subject to wind uplift pressures of 20 pounds per
    # square foot or greater are required to have attachments that are capable
    # of providing resistance, in this case assumed to be straps.
    # Otherwise, the connection is assumed to be toe nail.
    # CABO 1992-1995:
    # 802.11 Roof Tie-Down: Roof assemblies subject to wind uplift pressures of
    # 20 lbs per sq ft or greater shall have rafter or truess ties. The
    # resulting uplift forces from the rafter or turss ties shall be
    # transmitted to the foundation.
    # Roof uplift pressure varies by wind speed, exposure category, building
    # aspect ratio and roof height. For a reference building (9 ft tall in
    # exposure B -- WSF1) analysis suggests that wind speeds in excess of
    # 110 mph begin to generate pressures of 20 psf in high pressure zones of
    # the roof. Thus 110 mph is used as the critical velocity.
    elif year > 1992:
        if BIM['V_ult'] > 110:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail
    # CABO 1989 and earlier
    # There is no mention of straps or enhanced tie-downs in the CABO codes
    # older than 1992, and there is no description of these adoptions in IBHS
    # reports or the New Jersey Construction Code Communicator .
    # Although there is no explicit information, it seems that hurricane straps
    # really only came into effect in Florida after Hurricane Andrew (1992).
    # Because Florida is the leader in adopting hurricane protection measures
    # into codes and because there is no mention of shutters or straps in the
    # CABO codes, it is assumed that all roof-wall connections for residential
    # buildings are toe nails before 1992.
    else:
        # year <= 1992
        RWC = 'tnail' # Toe-nail

    # Shutters
    # IRC 2000-2015:
    # R301.2.1.2 in NJ IRC 2015 says protection of openings required for
    # buildings located in WBD regions, mentions impact-rated protection for
    # glazing, impact-resistance for garage door glazed openings, and finally
    # states that wood structural panels with a thickness > 7/16" and a
    # span <8' can be used, as long as they are precut, attached to the framing
    # surrounding the opening, and the attachments are resistant to corrosion
    # and are able to resist component and cladding loads;
    # Earlier IRC editions provide similar rules.
    if year > 2000:
        shutters = BIM['WBD']
    # CABO:
    # Based on Human Subjects Data, roughly 45% of houses built in the 1980s
    # and 1990s had entries that implied they had shutters on at some or all of
    # their windows. Therefore, 45% of houses in this time should be randomly
    # assigned to have shutters.
    # Data ranges checked:
    # 1992 to 1995, 33/74 entries (44.59%) with shutters
    # 1986 to 1992, 36/79 entries (45.57%) with shutters
    # 1983 to 1986, 19/44 entries (43.18%) with shutters
    else:
        # year <= 2000
        if BIM['WBD']:
            shutters = random.random() < 0.45
        else:
            shutters = False

    # Garage
    # As per IRC 2015:
    # Garage door glazed opening protection for windborne debris shall meet the
    # requirements of an approved impact-resisting standard or ANSI/DASMA 115.
    # Exception: Wood structural panels with a thickness of not less than 7/16
    # inch and a span of not more than 8 feet shall be permitted for opening
    # protection. Panels shall be predrilled as required for the anchorage
    # method and shall be secured with the attachment hardware provided.
    # Permitted for buildings where the ultimate design wind speed is 180 mph
    # or less.
    #
    # Average lifespan of a garage is 30 years, so garages that are not in WBD
    # (and therefore do not have any strength requirements) that are older than
    # 30 years are considered to be weak, whereas those from the last 30 years
    # are considered to be standard.
    if BIM['garage_tag'] == -1:
        # no garage data, using the default "standard"
        garage = 'std'
        shutters = 0 # HAZUS ties standard garage to w/o shutters
    else:
        if year > 2000:
            if shutters:
                if BIM['garage_tag'] < 1:
                    garage = 'no'
                else:
                    garage = 'sup' # SFBC 1994
                    shutters = 1 # HAZUS ties SFBC 1994 to with shutters
            else:
                if BIM['garage_tag'] < 1:
                    garage = 'no' # None
                else:
                    garage = 'std' # Standard
                    shutters = 0 # HAZUS ties standard garage to w/o shutters
        elif year > (datetime.datetime.now().year - 30):
            if BIM['garage_tag'] < 1:
                garage = 'no' # None
            else:
                garage = 'std' # Standard
                shutters = 0 # HAZUS ties standard garage to w/o shutters
        else:
            # year <= current year - 30
            if BIM['garage_tag'] < 1:
                garage = 'no' # None
            else:
                garage = 'wkd' # Weak
                shutters = 0 # HAZUS ties weak garage to w/o shutters

    # building configuration tag
    bldg_config = f"WSF" \
                  f"{int(min(BIM['stories'],2))}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{int(SWR)}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{garage}_" \
                  f"{int(shutters)}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def WMUH_config(BIM):
    """
    Rules to identify a HAZUS WMUH configuration based on BIM data

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

    year = BIM['year_built']  # just for the sake of brevity

    # Secondary Water Resistance (SWR)
    SWR = 0 # Default
    if year > 2000:
        if BIM['roof_shape'] == 'flt':
            SWR = 'null' # because SWR is not a question for flat roofs
        elif BIM['roof_shape'] in ['gab','hip']:
            SWR = int(random.random() < 0.6)
    elif year > 1987:
        if BIM['roof_shape'] == 'flt':
            SWR = 'null' # because SWR is not a question for flat roofs
        elif (BIM['roof_shape'] == 'gab') or (BIM['roof_shape'] == 'hip'):
            if BIM['roof_slope'] < 0.33:
                SWR = int(True)
            else:
                SWR = int(BIM['avg_jan_temp'] == 'below')
    else:
        # year <= 1987
        if BIM['roof_shape'] == 'flt':
            SWR = 'null' # because SWR is not a question for flat roofs
        else:
            SWR = int(random.random() < 0.3)

    # Roof cover & Roof quality
    # Roof cover and quality do not apply to gable and hip roofs
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'null'
        roof_quality = 'null'
    # NJ Building Code Section 1507 (in particular 1507.10 and 1507.12) address
    # Built Up Roofs and Single Ply Membranes. However, the NJ Building Code
    # only addresses installation and material standards of different roof
    # covers, but not in what circumstance each must be used.
    # SPMs started being used in the 1960s, but different types continued to be
    # developed through the 1980s. Today, single ply membrane roofing is the
    # most popular flat roof option. BURs have been used for over 100 years,
    # and although they are still used today, they are used less than SPMs.
    # Since there is no available ruleset to be taken from the NJ Building
    # Code, the ruleset is based off this information.
    # We assume that all flat roofs built before 1975 are BURs and all roofs
    # built after 1975 are SPMs.
    # Nothing in NJ Building Code or in the Hazus manual specifies what
    # constitutes “good” and “poor” roof conditions, so ruleset is dependant
    # on the age of the roof and average lifespan of BUR and SPM roofs.
    # We assume that the average lifespan of a BUR roof is 30 years and the
    # average lifespan of a SPM is 35 years. Therefore, BURs installed before
    # 1990 are in poor condition, and SPMs installed before 1985 are in poor
    # condition.
    else:
        if year >= 1975:
            roof_cover = 'spm'
            if BIM['year_built'] >= (datetime.datetime.now().year - 35):
                roof_quality = 'god'
            else:
                roof_quality = 'por'
        else:
            # year < 1975
            roof_cover = 'bur'
            if BIM['year_built'] >= (datetime.datetime.now().year - 30):
                roof_quality = 'god'
            else:
                roof_quality = 'por'

    # Roof Deck Attachment (RDA)
    # IRC 2009-2015:
    # Requires 8d nails (with spacing 6”/12”) for sheathing thicknesses between
    # ⅜”-1”, see Table 2304.10, Line 31. Fastener selection is contingent on
    # thickness of sheathing in building codes.
    # Wind Speed Considerations taken from Table 2304.6.1, Maximum Nominal
    # Design Wind Speed, Vasd, Permitted For Wood Structural Panel Wall
    # Sheathing Used to Resist Wind Pressures. Typical wall stud spacing is 16
    # inches, according to table 2304.6.3(4). NJ code defines this with respect
    # to exposures B and C only. These are mapped to HAZUS categories based on
    # roughness length in the ruleset herein.
    # The base rule was then extended to the exposures closest to suburban and
    # light suburban, even though these are not considered by the code.
    if year > 2009:
        if BIM['terrain'] >= 35: # suburban or light trees
            if BIM['V_ult'] > 168.0:
                RDA = '8s'  # 8d @ 6"/6" 'D'
            else:
                RDA = '8d'  # 8d @ 6"/12" 'B'
        else:  # light suburban or open
            if BIM['V_ult'] > 142.0:
                RDA = '8s'  # 8d @ 6"/6" 'D'
            else:
                RDA = '8d'  # 8d @ 6"/12" 'B'
    # IRC 2000-2006:
    # Table 2304.9.1, Line 31 of the 2006
    # NJ IBC requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
    # of ⅞”-1”. Fastener selection is contingent on thickness of sheathing in
    # building codes. Table 2308.10.1 outlines the required rating of approved
    # uplift connectors, but does not specify requirements that require a
    # change of connector at a certain wind speed.
    # Thus, all RDAs are assumed to be 8d @ 6”/12”.
    elif year > 2000:
        RDA = '8d'  # 8d @ 6"/12" 'B'
    # BOCA 1996:
    # The BOCA 1996 Building Code Requires 8d nails (with spacing 6”/12”) for
    # roof sheathing thickness up to 1". See Table 2305.2, Section 4.
    # Attachment requirements are given based on sheathing thickness, basic
    # wind speed, and the mean roof height of the building.
    elif year > 1996:
        if (BIM['V_ult'] >= 103 ) and (BIM['mean_roof_height'] >= 25.0):
            RDA = '8s'  # 8d @ 6"/6" 'D'
        else:
            RDA = '8d'  # 8d @ 6"/12" 'B'
    # BOCA 1993:
    # The BOCA 1993 Building Code Requires 8d nails (with spacing 6”/12”) for
    # sheathing thicknesses of 19/32  inches or greater, and 6d nails (with
    # spacing 6”/12”) for sheathing thicknesses of ½ inches or less.
    # See Table 2305.2, Section 4.
    elif year > 1993:
        if BIM['sheathing_t'] <= 0.5:
            RDA = '6d'  # 6d @ 6"/12" 'A'
        else:
            RDA = '8d'  # 8d @ 6"/12" 'B'
    else:
        # year <= 1993
        if BIM['sheathing_t'] <= 0.5:
            RDA = '6d' # 6d @ 6"/12" 'A'
        else:
            RDA = '8d' # 8d @ 6"/12" 'B'

    # Roof-Wall Connection (RWC)
    # IRC 2000-2015:
    # 1507.2.8.1 High Wind Attachment. Underlayment applied in areas subject
    # to high winds (Vasd greater than 110 mph as determined in accordance
    # with Section 1609.3.1) shall be applied with corrosion-resistant
    # fasteners in accordance with the manufacturer’s instructions. Fasteners
    # are to be applied along the overlap not more than 36 inches on center.
    # Underlayment installed where Vasd, in accordance with section 1609.3.1
    # equals or exceeds 120 mph shall be attached in a grid pattern of 12
    # inches between side laps with a 6-inch spacing at the side laps.
    if year > 2000:
        if BIM['V_ult'] > 142.0:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail
    # BOCA 1996 and earlier:
    # There is no mention of straps or enhanced tie-downs of any kind in the
    # BOCA codes, and there is no description of these adoptions in IBHS
    # reports or the New Jersey Construction Code Communicator .
    # Although there is no explicit information, it seems that hurricane straps
    # really only came into effect in Florida after Hurricane Andrew (1992),
    # and likely it took several years for these changes to happen. Because
    # Florida is the leader in adopting hurricane protection measures into
    # codes and because there is no mention of shutters or straps in the BOCA
    # codes, it is assumed that New Jersey did not adopt these standards until
    # the 2000 IBC.
    else:
        RWC = 'tnail'  # Toe-nail

    # Shutters
    # IRC 2000-2015:
    # 1609.1.2 Protection of Openings. In wind-borne debris regions, glazing in
    # buildings shall be impact resistant or protected with an impact-resistant
    # covering meeting the requirements of an approved impact-resistant
    # covering meeting the requirements of an approved impact-resistant
    # standard.
    # Exceptions: Wood structural panels with a minimum thickness of 7/16 of an
    # inch and a maximum panel span of 8 feet shall be permitted for opening
    # protection in buildings with a mean roof height of 33 feet or less that
    # are classified as a Group R-3 or R-4 occupancy.
    # Earlier IRC editions provide similar rules.
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Stories
    # Buildings with more than 3 stories are mapped to the 3-story configuration
    stories = min(BIM['stories'], 3)

    bldg_config = f"WMUH" \
                  f"{int(stories)}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{roof_cover}_" \
                  f"{roof_quality}_" \
                  f"{SWR}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{int(shutters)}_" \
                  f"{int(BIM['terrain'])}"

    return bldg_config


def MSF_config(BIM):
    """
    Rules to identify a HAZUS MSF configuration based on BIM data

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

    # Roof-Wall Connection (RWC)
    if BIM['HPR']:
        RWC = 'strap'  # Strap
    else:
        RWC = 'tnail'  # Toe-nail

    # Shutters
    # IRC 2000-2015:
    # R301.2.1.2 in NJ IRC 2015 says protection of openings required for
    # buildings located in WBD regions, mentions impact-rated protection for
    # glazing, impact-resistance for garage door glazed openings, and finally
    # states that wood structural panels with a thickness > 7/16" and a
    # span <8' can be used, as long as they are precut, attached to the framing
    # surrounding the opening, and the attachments are resistant to corrosion
    # and are able to resist component and cladding loads;
    # Earlier IRC editions provide similar rules.
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.45
        else:
            shutters = False

    # Garage
    # As per IRC 2015:
    # Garage door glazed opening protection for windborne debris shall meet the
    # requirements of an approved impact-resisting standard or ANSI/DASMA 115.
    # Exception: Wood structural panels with a thickness of not less than 7/16
    # inch and a span of not more than 8 feet shall be permitted for opening
    # protection. Panels shall be predrilled as required for the anchorage
    # method and shall be secured with the attachment hardware provided.
    # Permitted for buildings where the ultimate design wind speed is 180 mph
    # or less.
    #
    # Average lifespan of a garage is 30 years, so garages that are not in WBD
    # (and therefore do not have any strength requirements) that are older than
    # 30 years are considered to be weak, whereas those from the last 30 years
    # are considered to be standard.
    if BIM['garage_tag'] == -1:
        # no garage data, using the default "none"
        garage = 'nav'
    else:
        if year > (datetime.datetime.now().year - 30):
            if BIM['garage_tag'] < 1:
                garage = 'nav' # None
            else:
                if shutters:
                    garage = 'sup' # SFBC 1994
                else:
                    garage = 'std' # Standard
        else:
            # year <= current year - 30
            if BIM['garage_tag'] < 1:
                garage = 'nav' # None
            else:
                if shutters:
                    garage = 'sup'
                else:
                    garage = 'wkd' # Weak

    # Masonry Reinforcing (MR)
    # R606.6.4.1.2 Metal Reinforcement states that walls other than interior
    # non-load-bearing walls shall be anchored at vertical intervals of not
    # more than 8 inches with joint reinforcement of not less than 9 gage.
    # Therefore this ruleset assumes that all exterior or load-bearing masonry
    # walls will have reinforcement. Since our considerations deal with wind
    # speed, I made the assumption that only exterior walls are being taken
    # into consideration.
    MR = True

    if BIM['roof_system'] == 'trs':

        # Roof Deck Attachment (RDA)
        # IRC codes:
        # NJ code requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
        # between ⅜”-1” -  see Table R602.3(1)
        # Fastener selection is contingent on thickness of sheathing in building
        # codes. Commentary for Table R602.3(1) indicates 8d nails with 6”/6”
        # spacing (enhanced roof spacing) for ultimate wind speeds greater than
        # a speed_lim. speed_lim depends on the year of construction
        RDA = '6d' # Default (aka A) in Reorganized Rulesets - WIND
        if year >= 2016:
            # IRC 2015
            speed_lim = 130.0 # mph
        else:
            # IRC 2000 - 2009
            speed_lim = 100.0 # mph
        if BIM['V_ult'] > speed_lim:
            RDA = '8s'  # 8d @ 6"/6" ('D' in the Reorganized Rulesets - WIND)
        else:
            RDA = '8d'  # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)

        # Secondary Water Resistance (SWR)
        # Minimum drainage recommendations are in place in NJ (See below).
        # However, SWR indicates a code-plus practice.
        SWR = random.random() < 0.6

        stories = min(BIM['stories'], 2)
        bldg_config = f"MSF" \
                      f"{int(stories)}_" \
                      f"{BIM['roof_shape']}_" \
                      f"{int(SWR)}_" \
                      f"{RDA}_" \
                      f"{RWC}_" \
                      f"{garage}_" \
                      f"{int(shutters)}_" \
                      f"{int(MR)}_" \
                      f"{int(BIM['terrain'])}"
        return bldg_config

    else:
        # Roof system = OSJW
        # r
        # A 2015 study found that there were 750,000 metal roof installed in 2015,
        # out of 5 million new roofs in the US annually. If these numbers stay
        # relatively stable, that implies that roughtly 15% of roofs are smlt.
        # ref. link: https://www.bdcnetwork.com/blog/metal-roofs-are-soaring-
        # popularity-residential-marmet
        r_option = ['smtl', 'cshl']
        r = r_option[int(random.random() < 0.85)]

        # Roof Deck Attachment (RDA)
        # NJ IBC 1507.2.8.1 (for cshl)
        # high wind attachments are required for DSWII > 142 mph
        # NJ IBC 1507.4.5 (for smtl)
        # high wind attachment are required for DSWII > 142 mph
        if BIM['V_ult'] > 142.0:
            RDA = 'sup' # superior
        else:
            RDA = 'std' # standard

        # Secondary Water Resistance (SWR)
        # Minimum drainage recommendations are in place in NJ (See below).
        # However, SWR indicates a code-plus practice.
        SWR = False # Default
        if BIM['roof_shape'] == 'flt':
            SWR = True
        elif BIM['roof_shape'] in ['hip', 'gab']:
            SWR = random.random() < 0.6

        stories = min(BIM['stories'], 2)
        bldg_config = f"MSF" \
                      f"{int(stories)}_" \
                      f"{BIM['roof_shape']}_" \
                      f"{int(SWR)}_" \
                      f"{RDA}_" \
                      f"{RWC}_" \
                      f"{garage}_" \
                      f"{int(shutters)}_" \
                      f"{int(MR)}_" \
                      f"{int(BIM['terrain'])}"
        return bldg_config


def MMUH_config(BIM):
    """
    Rules to identify a HAZUS MMUH configuration based on BIM data

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

    # Secondary Water Resistance (SWR)
    # Minimum drainage recommendations are in place in NJ (See below).
    # However, SWR indicates a code-plus practice.
    SWR = False # Default
    if BIM['roof_shape'] == 'flt':
        SWR = True
    elif BIM['roof_shape'] in ['hip', 'gab']:
        SWR = random.random() < 0.6

    # Roof cover & Roof quality
    # Roof cover and quality do not apply to gable and hip roofs
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'nav'
        roof_quality = 'nav'
    # NJ Building Code Section 1507 (in particular 1507.10 and 1507.12) address
    # Built Up Roofs and Single Ply Membranes. However, the NJ Building Code
    # only addresses installation and material standards of different roof
    # covers, but not in what circumstance each must be used.
    # SPMs started being used in the 1960s, but different types continued to be
    # developed through the 1980s. Today, single ply membrane roofing is the
    # most popular flat roof option. BURs have been used for over 100 years,
    # and although they are still used today, they are used less than SPMs.
    # Since there is no available ruleset to be taken from the NJ Building
    # Code, the ruleset is based off this information.
    # We assume that all flat roofs built before 1975 are BURs and all roofs
    # built after 1975 are SPMs.
    # Nothing in NJ Building Code or in the Hazus manual specifies what
    # constitutes “good” and “poor” roof conditions, so ruleset is dependant
    # on the age of the roof and average lifespan of BUR and SPM roofs.
    # We assume that the average lifespan of a BUR roof is 30 years and the
    # average lifespan of a SPM is 35 years. Therefore, BURs installed before
    # 1990 are in poor condition, and SPMs installed before 1985 are in poor
    # condition.
    else:
        if year >= 1975:
            roof_cover = 'spm'
            if BIM['year_built'] >= (datetime.datetime.now().year - 35):
                roof_quality = 'god'
            else:
                roof_quality = 'por'
        else:
            # year < 1975
            roof_cover = 'bur'
            if BIM['year_built'] >= (datetime.datetime.now().year - 30):
                roof_quality = 'god'
            else:
                roof_quality = 'por'

    # Roof Deck Attachment (RDA)
    # IRC 2009-2015:
    # Requires 8d nails (with spacing 6”/12”) for sheathing thicknesses between
    # ⅜”-1”, see Table 2304.10, Line 31. Fastener selection is contingent on
    # thickness of sheathing in building codes.
    # Wind Speed Considerations taken from Table 2304.6.1, Maximum Nominal
    # Design Wind Speed, Vasd, Permitted For Wood Structural Panel Wall
    # Sheathing Used to Resist Wind Pressures. Typical wall stud spacing is 16
    # inches, according to table 2304.6.3(4). NJ code defines this with respect
    # to exposures B and C only. These are mapped to HAZUS categories based on
    # roughness length in the ruleset herein.
    # The base rule was then extended to the exposures closest to suburban and
    # light suburban, even though these are not considered by the code.
    if BIM['terrain'] >= 35: # suburban or light trees
        if BIM['V_ult'] > 130.0:
            RDA = '8s'  # 8d @ 6"/6" 'D'
        else:
            RDA = '8d'  # 8d @ 6"/12" 'B'
    else:  # light suburban or open
        if BIM['V_ult'] > 110.0:
            RDA = '8s'  # 8d @ 6"/6" 'D'
        else:
            RDA = '8d'  # 8d @ 6"/12" 'B'

    # Roof-Wall Connection (RWC)
    if BIM['V_ult'] > 110.0:
        RWC = 'strap'  # Strap
    else:
        RWC = 'tnail'  # Toe-nail

    # Shutters
    # IRC 2000-2015:
    # R301.2.1.2 in NJ IRC 2015 says protection of openings required for
    # buildings located in WBD regions, mentions impact-rated protection for
    # glazing, impact-resistance for garage door glazed openings, and finally
    # states that wood structural panels with a thickness > 7/16" and a
    # span <8' can be used, as long as they are precut, attached to the framing
    # surrounding the opening, and the attachments are resistant to corrosion
    # and are able to resist component and cladding loads;
    # Earlier IRC editions provide similar rules.
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Masonry Reinforcing (MR)
    # R606.6.4.1.2 Metal Reinforcement states that walls other than interior
    # non-load-bearing walls shall be anchored at vertical intervals of not
    # more than 8 inches with joint reinforcement of not less than 9 gage.
    # Therefore this ruleset assumes that all exterior or load-bearing masonry
    # walls will have reinforcement. Since our considerations deal with wind
    # speed, I made the assumption that only exterior walls are being taken
    # into consideration.
    MR = True

    stories = min(BIM['stories'], 3)
    bldg_config = f"MMUH" \
                  f"{int(stories)}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{int(SWR)}_" \
                  f"{roof_cover}_" \
                  f"{roof_quality}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{int(shutters)}_" \
                  f"{int(MR)}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def MLRM_config(BIM):
    """
    Rules to identify a HAZUS MLRM configuration based on BIM data

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

    # Note the only roof option for commercial masonry in NJ appraisers manual
    # is OSWJ, so this suggests they do not even see alternate roof system
    # ref: Custom Inventory google spreadsheet H-37 10/01/20
    # This could be commented for other regions if detailed data are available
    BIM['roof_system'] = 'ows'

    # Roof cover
    # Roof cover does not apply to gable and hip roofs
    if year >= 1975:
        roof_cover = 'spm'
    else:
        # year < 1975
        roof_cover = 'bur'

    # Shutters
    # IRC 2000-2015:
    # R301.2.1.2 in NJ IRC 2015 says protection of openings required for
    # buildings located in WBD regions, mentions impact-rated protection for
    # glazing, impact-resistance for garage door glazed openings, and finally
    # states that wood structural panels with a thickness > 7/16" and a
    # span <8' can be used, as long as they are precut, attached to the framing
    # surrounding the opening, and the attachments are resistant to corrosion
    # and are able to resist component and cladding loads;
    # Earlier IRC editions provide similar rules.
    shutters = BIM['WBD']

    # Masonry Reinforcing (MR)
    # R606.6.4.1.2 Metal Reinforcement states that walls other than interior
    # non-load-bearing walls shall be anchored at vertical intervals of not
    # more than 8 inches with joint reinforcement of not less than 9 gage.
    # Therefore this ruleset assumes that all exterior or load-bearing masonry
    # walls will have reinforcement. Since our considerations deal with wind
    # speed, I made the assumption that only exterior walls are being taken
    # into consideration.
    MR = True

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                 'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    if BIM['roof_system'] == 'ows':
        # RDA
        RDA = '6d' # HAZUS the only available option for OWSJ

        # Roof deck age (DQ)
        # Average lifespan of a steel joist roof is roughly 50 years according
        # to the source below. Therefore, if constructed 50 years before the
        # current year, the roof deck should be considered old.
        # https://www.metalroofing.systems/metal-roofing-pros-cons/
        if year >= (datetime.datetime.now().year - 50):
            DQ = 'god' # new or average
        else:
            DQ = 'por' # old

        # RWC
        RWC = 'tnail'  # Toe-nail (HAZUS the only available option for OWSJ)

        # Metal RDA
        # 1507.2.8.1 High Wind Attachment.
        # Underlayment applied in areas subject to high winds (Vasd greater
        # than 110 mph as determined in accordance with Section 1609.3.1) shall
        #  be applied with corrosion-resistant fasteners in accordance with
        # the manufacturer’s instructions. Fasteners are to be applied along
        # the overlap not more than 36 inches on center.
        if BIM['V_ult'] > 142:
            MRDA = 'std'  # standard
        else:
            MRDA = 'sup'  # superior

    elif BIM['roof_system'] == 'trs':
        # This clause should not be activated for NJ
        # RDA
        if BIM['terrain'] >= 35: # suburban or light trees
            if BIM['V_ult'] > 130.0:
                RDA = '8s'  # 8d @ 6"/6" 'D'
            else:
                RDA = '8d'  # 8d @ 6"/12" 'B'
        else:  # light suburban or open
            if BIM['V_ult'] > 110.0:
                RDA = '8s'  # 8d @ 6"/6" 'D'
            else:
                RDA = '8d'  # 8d @ 6"/12" 'B'

        #  Metal RDA
        MRDA = 'nav'

        # Roof deck agea (DQ)
        DQ = 'nav' # not available for wood truss system

        # RWC
        if BIM['V_ult'] > 110:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    if BIM['mean_roof_height'] < 15.0:
        # if it's MLRM1, configure outputs
        bldg_config = f"MLRM1_" \
                      f"{roof_cover}_" \
                      f"{RDA}_" \
                      f"{DQ}_" \
                      f"{BIM['roof_system']}_" \
                      f"{RWC}_" \
                      f"{int(shutters)}_" \
                      f"{WIDD}_" \
                      f"{int(MR)}_" \
                      f"{MRDA}_" \
                      f"{int(BIM['terrain'])}"
        return bldg_config
    else:
        unit_tag = 'nav'
        # MLRM2 needs more rulesets
        if BIM['roof_system'] == 'trs':
            JSPA = 0
        elif BIM['roof_system'] == 'ows':
            if BIM['no_units'] == 1:
                JSPA = 0
                unit_tag = 'sgl'
            else:
                JSPA = 4
                unit_tag = 'mlt'

        bldg_config = f"MLRM2_" \
                      f"{roof_cover}_" \
                      f"{RDA}_" \
                      f"{DQ}_" \
                      f"{BIM['roof_system']}_" \
                      f"{JSPA}_" \
                      f"{RWC}_" \
                      f"{int(shutters)}_" \
                      f"{WIDD}_" \
                      f"{unit_tag}_" \
                      f"{int(MR)}_" \
                      f"{MRDA}_" \
                      f"{int(BIM['terrain'])}"
        return bldg_config


def MLRI_config(BIM):
    """
    Rules to identify a HAZUS MLRI configuration based on BIM data

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

    # MR
    MR = True

    # Shutters
    shutters = False

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'nav'
        roof_quality = 'god' # default supported by HAZUS
    else:
        if year >= 1975:
            roof_cover = 'spm'
            if BIM['year_built'] >= (datetime.datetime.now().year - 35):
                roof_quality = 'god'
            else:
                roof_quality = 'por'
        else:
            # year < 1975
            roof_cover = 'bur'
            if BIM['year_built'] >= (datetime.datetime.now().year - 30):
                roof_quality = 'god'
            else:
                roof_quality = 'por'

    bldg_config = f"MLRI_" \
                  f"{roof_quality}_" \
                  f"{int(shutters)}_" \
                  f"{int(MR)}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def MERB_config(BIM):
    """
    Rules to identify a HAZUS MERB configuration based on BIM data

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

    # Roof cover
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'bur'
        # no info, using the default supoorted by HAZUS
    else:
        if year >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.45
        else:
            shutters = False

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                 'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    # Window area ratio
    if BIM['window_area'] < 0.33:
        WWR = 'low'
    elif BIM['window_area'] < 0.5:
        WWR = 'med'
    else:
        WWR = 'hig'

    if BIM['stories'] <= 2:
        bldg_tag = 'MERBL'
    elif BIM['stories'] <= 5:
        bldg_tag = 'MERBM'
    else:
        bldg_tag = 'MERBH'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_cover}_" \
                  f"{WWR}_" \
                  f"{int(shutters)}_" \
                  f"{WIDD}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def MECB_config(BIM):
    """
    Rules to identify a HAZUS MECB configuration based on BIM data

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

    # Roof cover
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'bur'
        # no info, using the default supoorted by HAZUS
    else:
        if year >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                 'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    # Window area ratio
    if BIM['window_area'] < 0.33:
        WWR = 'low'
    elif BIM['window_area'] < 0.5:
        WWR = 'med'
    else:
        WWR = 'hig'

    if BIM['stories'] <= 2:
        bldg_tag = 'MECBL'
    elif BIM['stories'] <= 5:
        bldg_tag = 'MECBM'
    else:
        bldg_tag = 'MECBH'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_cover}_" \
                  f"{WWR}_" \
                  f"{int(shutters)}_" \
                  f"{WIDD}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def CECB_config(BIM):
    """
    Rules to identify a HAZUS CECB configuration based on BIM data

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

    # Roof cover
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'bur'
        # Warning: HAZUS does not have N/A option for CECB, so here we use bur
    else:
        if year >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                  'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    # Window area ratio
    if BIM['window_area'] < 0.33:
        WWR = 'low'
    elif BIM['window_area'] < 0.5:
        WWR = 'med'
    else:
        WWR = 'hig'

    if BIM['stories'] <= 2:
        bldg_tag = 'CECBL'
    elif BIM['stories'] <= 5:
        bldg_tag = 'CECBM'
    else:
        bldg_tag = 'CECBH'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_cover}_" \
                  f"{WWR}_" \
                  f"{int(shutters)}_" \
                  f"{WIDD}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def CERB_config(BIM):
    """
    Rules to identify a HAZUS CERB configuration based on BIM data

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

    # Roof cover
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'bur'
        # Warning: HAZUS does not have N/A option for CECB, so here we use bur
    else:
        if year >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.45
        else:
            shutters = False

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                  'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    # Window area ratio
    if BIM['window_area'] < 0.33:
        WWR = 'low'
    elif BIM['window_area'] < 0.5:
        WWR = 'med'
    else:
        WWR = 'hig'

    if BIM['stories'] <= 2:
        bldg_tag = 'CERBL'
    elif BIM['stories'] <= 5:
        bldg_tag = 'CERBM'
    else:
        bldg_tag = 'CERBH'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_cover}_" \
                  f"{WWR}_" \
                  f"{int(shutters)}_" \
                  f"{WIDD}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def SPMB_config(BIM):
    """
    Rules to identify a HAZUS SPMB configuration based on BIM data

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

    # Roof Deck Age (~ Roof Quality)
    if BIM['year_built'] >= (datetime.datetime.now().year - 50):
        roof_quality = 'god'
    else:
        roof_quality = 'por'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    if BIM['area'] <= 4000:
        bldg_tag = 'SPMBS'
    elif BIM['area'] <= 50000:
        bldg_tag = 'SPMBM'
    else:
        bldg_tag = 'SPMBL'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_quality}_" \
                  f"{int(shutters)}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def SECB_config(BIM):
    """
    Rules to identify a HAZUS SECB configuration based on BIM data

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

    # Roof cover
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'bur'
        # Warning: HAZUS does not have N/A option for CECB, so here we use bur
    else:
        if year >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                  'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    # Window area ratio
    if BIM['window_area'] < 0.33:
        WWR = 'low'
    elif BIM['window_area'] < 0.5:
        WWR = 'med'
    else:
        WWR = 'hig'

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    if BIM['stories'] <= 2:
        bldg_tag = 'SECBL'
    elif BIM['stories'] <= 5:
        bldg_tag = 'SECBM'
    else:
        bldg_tag = 'SECBH'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_cover}_" \
                  f"{WWR}_" \
                  f"{int(shutters)}_" \
                  f"{WIDD}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


def SERB_config(BIM):
    """
    Rules to identify a HAZUS SERB configuration based on BIM data

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

    # Roof cover
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'bur'
        # Warning: HAZUS does not have N/A option for CECB, so here we use bur
    else:
        if year >= 1975:
            roof_cover = 'spm'
        else:
            # year < 1975
            roof_cover = 'bur'

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                  'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    # Window area ratio
    if BIM['window_area'] < 0.33:
        WWR = 'low'
    elif BIM['window_area'] < 0.5:
        WWR = 'med'
    else:
        WWR = 'hig'

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    if BIM['stories'] <= 2:
        bldg_tag = 'SERBL'
    elif BIM['stories'] <= 5:
        bldg_tag = 'SERBM'
    else:
        bldg_tag = 'SERBH'

    bldg_config = f"{bldg_tag}_" \
                  f"{roof_cover}_" \
                  f"{WWR}_" \
                  f"{int(shutters)}_" \
                  f"{WIDD}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config


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

def Assm_config(BIM):
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
        flood_type = 'caz' # Costal/A-Zone
    elif BIM['flood_zone'] in [6101, 6102]:
        flood_type = 'cvz' # Costal/V-Zone
    else:
        flood_type = 'caz' # Default

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

    # fl_assm
    fl_assm = f"{'fl_surge_assm'}_" \
              f"{BIM['occupancy_class']}_" \
              f"{int(PostFIRM)}_" \
              f"{flood_type}"

    # hu_assm
    hu_assm = f"{'hu_surge_assm'}_" \
              f"{BIM['occupancy_class']}_" \
              f"{int(PostFIRM)}"

    return hu_assm, fl_assm


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

    # identify the building class
    bldg_class = building_class(BIM_ap)

    # prepare the building configuration string
    if bldg_class == 'WSF':
        bldg_config = WSF_config(BIM_ap)
    elif bldg_class == 'WMUH':
        bldg_config = WMUH_config(BIM_ap)
    elif bldg_class == 'MSF':
        bldg_config = MSF_config(BIM_ap)
    elif bldg_class == 'MMUH':
        bldg_config = MMUH_config(BIM_ap)
    elif bldg_class == 'MLRM':
        bldg_config = MLRM_config(BIM_ap)
    elif bldg_class == 'MLRI':
        bldg_config = MLRI_config(BIM_ap)
    elif bldg_class == 'MERB':
        bldg_config = MERB_config(BIM_ap)
    elif bldg_class == 'MECB':
        bldg_config = MECB_config(BIM_ap)
    elif bldg_class == 'CECB':
        bldg_config = CECB_config(BIM_ap)
    elif bldg_class == 'CERB':
        bldg_config = CERB_config(BIM_ap)
    elif bldg_class == 'SPMB':
        bldg_config = SPMB_config(BIM_ap)
    elif bldg_class == 'SECB':
        bldg_config = SECB_config(BIM_ap)
    elif bldg_class == 'SERB':
        bldg_config = SERB_config(BIM_ap)
    else:
        raise ValueError(
            f"Building class {bldg_class} not recognized by the "
            f"auto-population routine."
        )

    # prepare the flood rulesets
    fld_config = FL_config(BIM_ap)

    # prepare the assembly loss compositions
    hu_assm, fl_assm = Assm_config(BIM_ap)

    DL_ap = {
        '_method'      : 'HAZUS MH HU',
        'LossModel'    : {
            'DecisionVariables': {
                "ReconstructionCost": True
            },
            'ReplacementCost'  : 100
        },
        'Components'   : {
            bldg_config: [{
                'location'       : '1',
                'direction'      : '1',
                'median_quantity': '1.0',
                'unit'           : 'ea',
                'distribution'   : 'N/A'
            }],
            fld_config: [{
                'location'       : '1',
                'direction'      : '1',
                'median_quantity': '1.0',
                'unit'           : 'ea',
                'distribution'   : 'N/A'
            }]
        },
        'Combinations' : [hu_assm, fl_assm]
    }

    return BIM_ap, DL_ap
