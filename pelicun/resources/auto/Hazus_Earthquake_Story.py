# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Leland Stanford Junior University
# Copyright (c) 2023 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

import pandas as pd

ap_DesignLevel = {
    1940: 'PC',
    1940: 'LC',
    1975: 'MC',
    2100: 'HC'
}

ap_DesignLevel_W1 = {
       0: 'PC',
       0: 'LC',
    1975: 'MC',
    2100: 'HC'
}

ap_Occupancy = {
    'Other/Unknown': 'RES3',
    'Residential - Single-Family': 'RES1',
    'Residential - Town-Home': 'RES3',
    'Residential - Multi-Family': 'RES3',
    'Residential - Mixed Use': 'RES3',
    'Office': 'COM4',
    'Hotel': 'RES4',
    'School': 'EDU1',
    'Industrial - Light': 'IND2',
    'Industrial - Warehouse': 'IND2',
    'Industrial - Heavy': 'IND1',
    'Retail': 'COM1',
    'Parking' : 'COM10'
}

convert_design_level = {
        'High-Code'    : 'HC',
        'Moderate-Code': 'MC',
        'Low-Code'     : 'LC',
        'Pre-Code'     : 'PC'
    }

def convert_story_rise(structureType, stories):
    rise = None
    if structureType == 'RM1':
        if stories <= 3:
            rise = "L"
        else:
            rise = "M"
    elif structureType == 'URM':
        if stories <= 2:
            rise = "L"
        else:
            rise = "M"
    elif structureType in ['S1', 'S2', 'S4', 'S5', 'C1', 'C2', 'C3', 'PC1', \
                           'PC2', 'RM2']:
        if stories <=3:
            rise = "L"
        elif stories <= 7:
            rise = "M"
        else:
            rise = "H"
    return rise

def story_scale(stories, comp_type):
    if comp_type == 'NSA':
        if stories == 1:
            return 1.00
        elif stories == 2:
            return 1.22
        elif stories == 3:
            return 1.40
        elif stories == 4:
            return 1.45
        elif stories == 5:
            return 1.50
        elif stories == 6:
            return 1.90
        elif stories == 7:
            return 2.05
        elif stories == 8:
            return 2.15
        elif stories == 9:
            return 2.20
        elif (stories >= 10) and (stories < 30):
            return 2.30 + (stories-10)*0.04
        elif stories >= 30:
            return 3.10
        else:
            return 1.0

    elif comp_type in ['S', 'NSD']:
        if stories == 1:
            return 1.45
        elif stories == 2:
            return 1.90
        elif stories == 3:
            return 2.50
        elif stories == 4:
            return 2.75
        elif stories == 5:
            return 3.00
        elif stories == 6:
            return 3.50
        elif stories == 7:
            return 3.50
        elif stories == 8:
            return 3.50
        elif stories == 9:
            return 4.50
        elif (stories >= 10) and (stories < 50):
            return 4.50 + (stories-10)*0.07
        elif stories >= 50:
            return 7.30
        else:
            return 1.0


def auto_populate(AIM):
    """
    Automatically creates a performance model for story EDP-based Hazus EQ analysis.

    Parameters
    ----------
    AIM: dict
        Asset Information Model - provides features of the asset that can be 
        used to infer attributes of the performance model.

    Returns
    -------
    GI_ap: dict
        Extended General Information - extends the GI from the input AIM with 
        additional inferred features. These features are typically used in 
        intermediate steps during the auto-population and are not required 
        for the performance assessment. They are returned to allow reviewing 
        how these latent variables affect the final results.
    DL_ap: dict
        Damage and Loss parameters - these define the performance model and 
        details of the calculation.
    CMP: DataFrame
        Component assignment - Defines the components (in rows) and their 
        location, direction, and quantity (in columns).
    """

    # extract the General Information
    GI = AIM.get('GeneralInformation', None)

    if GI==None:
        #TODO: show an error message
        pass

    # initialize the auto-populated GI
    GI_ap = GI.copy()

    assetType = AIM["assetType"]
    ground_failure = AIM["Applications"]["DL"]["ApplicationData"]["ground_failure"]

    if assetType=="Buildings":

        # get the building parameters
        bt = GI['StructureType'] #building type

        # get the number of stories / height
        stories = GI.get('NumberOfStories', None)

        if stories!=None:
            # We assume that the structure type does not include height information
            # and we append it here based on the number of story information

            if bt not in ['W1', 'W2', 'S3', 'PC1', 'MH']:
                if bt not in ['URM']:
                    if stories <= 3:
                        bt += 'L'
                    elif stories <= 7:
                        bt += 'M'
                    else:
                        if bt in ['RM']:
                            bt += 'M'
                        else:
                            bt += 'H'
                else:
                    if stories <= 2:
                        bt += 'L'
                    else:
                        bt += 'M'

            GI_ap['BuildingType'] = bt

        # get the design level
        dl = GI.get('DesignLevel', None)

        if dl == None:
            # If there is no DesignLevel provided, we assume that the YearBuilt is
            # available
            year_built = GI['YearBuilt']

            if 'W1' in bt:
                DesignL = ap_DesignLevel_W1
            else:
                DesignL = ap_DesignLevel
            
            for year in sorted(DesignL.keys()):
                if year_built <= year:
                    dl = DesignL[year]            
                    break

            GI_ap['DesignLevel'] = dl

        # get the occupancy class
        if GI['OccupancyClass'] in ap_Occupancy.keys():
            ot = ap_Occupancy[GI['OccupancyClass']]
        else:
            ot = GI['OccupancyClass']

        ReplacementCost = GI.get('ReplacementCost', None)
        ReplacementTime = GI.get('ReplacementTime', None)
        ReplacementCarbon = GI.get('ReplacementCarbon', None)
        ReplacementEnergy = GI.get('ReplacementEnergy', None)
        # population = GI.get('Population', 1.0)
        plan_area = GI.get('PlanArea', 1.0)

        rise = convert_story_rise(bt, stories)
        if rise is not None:
            FG_S = f'STR.{bt}.{rise}.{dl}'
        else:
            FG_S = f'STR.{bt}.{dl}'
        FG_NSD = f'NSD'
        FG_NSA = f'NSA.{dl}'

        CMP = pd.DataFrame(
                {f'{FG_S}':   ['ea',      'all',     '1, 2', f"{story_scale(stories, 'S')/stories/2.}",   'N/A'],
                 f'{FG_NSA}': ['ea',      'all',      0, f"{story_scale(stories, 'NSA')/stories}",   'N/A'],
                 f'{FG_NSD}': ['ea',      'all',     '1, 2', f"{story_scale(stories, 'NSD')/stories/2.}",   'N/A']},
                index = ['Units','Location','Direction',
                         'Theta_0','Family']
            
            ).T

        # if needed, add components to simulate damage from ground failure
        if ground_failure:

            foundation_type = 'S'

            FG_GF_H = f'GF.H.{foundation_type}'
            FG_GF_V = f'GF.V.{foundation_type}'
            
            CMP_GF = pd.DataFrame(
                {f'{FG_GF_H}':[  'ea',         1,          1,        1,   'N/A'],
                 f'{FG_GF_V}':[  'ea',         1,          3,        1,   'N/A']},
                index = [     'Units','Location','Direction','Theta_0','Family']
            ).T

            CMP = pd.concat([CMP, CMP_GF], axis=0)
        bldg_repair_config = {
                    "ConsequenceDatabase": "Hazus Earthquake - Buildings",
                    "MapApproach": "Automatic"
                }
        DL_ap = {
            "Asset": {
                "ComponentAssignmentFile": "CMP_QNT.csv",
                "ComponentDatabase": "Hazus Earthquake - Buildings",
                "NumberOfStories": f"{stories}",
                "OccupancyType": f"{ot}",
                "PlanArea": str(plan_area)
            },
            "Damage": {
                "DamageProcess": "Hazus Earthquake"
            },
            "Demands": {        
            },
            "Losses": {
                "BldgRepair": bldg_repair_config
            }
        }
        
    else:
        print(f"AssetType: {assetType} is not supported in Hazus Earthquake Story-based DL method")

    return GI_ap, DL_ap, CMP