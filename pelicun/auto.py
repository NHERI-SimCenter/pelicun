# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
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

"""
This module has classes and methods that auto-populate loss models.

.. rubric:: Contents

.. autosummary::

"""

from .base import *
import json

ap_DesignLevel = {
    1940: 'Pre-Code',
    1940: 'Low-Code',
    1975: 'Moderate-Code',
    2100: 'High-Code'
}

ap_DesignLevel_W1 = {
       0: 'Pre-Code',
       0: 'Low-Code',
    1975: 'Moderate-Code',
    2100: 'High-Code'
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

ap_RoofType = {
    'hip': 'hip',
    'hipped': 'hip',
    'gabled': 'gab',
    'gable': 'gab',
    'flat': 'flt'   # we need to change this later!
}

convert_design_level = {
        'High-Code'    : 'HC',
        'Moderate-Code': 'MC',
        'Low-Code'     : 'LC',
        'Pre-Code'     : 'PC'
    }

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

def auto_populate(DL_input_path, EDP_input_path,
                  DL_method, realization_count, coupled_EDP, event_time, 
                  ground_failure):

    with open(DL_input_path, 'r') as f:
        DL_input = json.load(f)

    EDP_input = pd.read_csv(EDP_input_path, sep='\s+', header=0,
                            index_col=0)

    if 'GeneralInformation' in DL_input.keys():
        BIM_in = DL_input['GeneralInformation']
    elif 'GI' in DL_input.keys():
        BIM_in = DL_input['GI']

    is_IM_based = DL_method[-2:] == 'IM'

    stories = BIM_in['numStory']
    # use only 1 story if DM is based on IM
    if DL_method == 'HAZUS MH EQ IM':
        stories = 1
    BIM_in.update({'stories':stories})

    # HAZUS Earthquake
    if DL_method in ['HAZUS MH EQ', 'HAZUS MH EQ IM']:        

        bt = BIM_in['structType']

        if bt == 'RV.structType':
            bt = EDP_input['structType'].values[0]

        year_built = BIM_in['yearBuilt']

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

        if BIM_in['occupancy'] in ap_Occupancy.keys():
            ot = ap_Occupancy[BIM_in['occupancy']]
        else:
            ot = BIM_in['occupancy']

        replacementCost = BIM_in.get('replacementCost', 1.0)
        replacementTime = BIM_in.get('replacementTime', 1.0)
        population = BIM_in.get('population', 1.0)

        loss_dict = {
            '_method': DL_method,
            'DamageModel': {
                'StructureType': bt
            },
            'LossModel': {
                'DecisionVariables': {
                    'ReconstructionCost': True,
                    'ReconstructionTime': True,
                    'Injuries': True
                },
                'Inhabitants': {
                    'OccupancyType': ot,
                    'PeakPopulation': f'{population}'
                },
                'ReplacementCost': replacementCost,
                'ReplacementTime': replacementTime
            },
            'ResponseModel': {
                'ResponseDescription': {
                    'Realizations': realization_count,
                    "CoupledAssessment": coupled_EDP
                }
            },
            "Dependencies": {
                "Fragilities": "btw. Performance Groups"                
            }
        }

        # add uncertainty if the EDPs are not coupled
        if not coupled_EDP:
            loss_dict['ResponseModel'].update({
                "AdditionalUncertainty": {
                    "GroundMotion": "0.10",
                    "Modeling"    : "0.20"
                    }})

        if is_IM_based:
            loss_dict.update({
                "ComponentDataFolder": pelicun_path+"/resources/HAZUS MH 2.1 earthquake PGA/DL json/"
                })
        else:
            loss_dict['ResponseModel'].update({
                'DetectionLimits': {
                    "PFA": "100.0",
                    "PID": "0.20",
                    "PRD": "0.20"
                }})
            loss_dict.update({
                "ComponentDataFolder": pelicun_path+"/resources/HAZUS MH 2.1 earthquake/DL json/"
                })

        if 'W1' in bt:
            DesignL = ap_DesignLevel_W1
        else:
            DesignL = ap_DesignLevel

        for year in sorted(DesignL.keys()):
            if year_built <= year:
                loss_dict['DamageModel'].update(
                    {'DesignLevel': DesignL[year]})
                break

        dl = convert_design_level[loss_dict['DamageModel']['DesignLevel']]
        if 'C3' in bt:
            if dl not in ['LC', 'PC']:
                dl = 'LC'

        # only one structural component for IM-based approach
        if is_IM_based:

            FG_S = f'S-{bt}-{dl}-{ot}'

            loss_dict.update({
                'Components': {
                    FG_S: [
                        {'location': '1',
                         'direction': '1',
                         'median_quantity': '1.0',
                         'unit': 'ea',
                         'distribution': 'N/A'
                        }]
                }})

        # story-based approach
        else:

            FG_S = f'S-{bt}-{dl}-{ot}'
            FG_NSD = f'NSD-{ot}'
            FG_NSA = f'NSA-{dl}-{ot}'

            loss_dict.update({
                'Components': {
                    FG_S: [
                        {'location': 'all',
                         'direction': '1, 2',
                         #'median_quantity': '{q}'.format(q = 0.5), #/stories),
                         'median_quantity': '{q}'.format(q = story_scale(stories, 'S')/stories/2.),
                         'unit': 'ea',
                         'distribution': 'N/A'
                        }],
                    FG_NSA: [
                        {'location': 'all',
                         'direction': '1',
                         #'median_quantity': '{q}'.format(q = 1.0), #/stories),
                         'median_quantity': '{q}'.format(q = story_scale(stories, 'NSA')/stories),
                         'unit': 'ea',
                         'distribution': 'N/A'
                        }],
                    FG_NSD: [
                        {'location': 'all',
                         'direction': '1, 2',
                         #'median_quantity': '{q}'.format(q = 0.5), #/stories),
                         'median_quantity': '{q}'.format(q = story_scale(stories, 'NSD')/stories/2.),
                         'unit': 'ea',
                         'distribution': 'N/A'
                        }]
                }})

        # if damage from ground failure is included
        if ground_failure:

            foundation_type = 'S'

            FG_GF_H = f'GF-H_{foundation_type}-{bt}'
            FG_GF_V = f'GF-V_{foundation_type}-{bt}'

            loss_dict['Components'].update({
                FG_GF_H: [
                    {'location': '1',
                     'direction': '1',
                     'median_quantity': '1.0',
                     'unit': 'ea',
                     'distribution': 'N/A'
                    }],
                FG_GF_V: [
                    {'location': '1',
                     'direction': '3',
                     'median_quantity': '1.0',
                     'unit': 'ea',
                     'distribution': 'N/A'
                    }]
            })

            # define logic that connects ground failure with building damage
            loss_dict.update({
                'DamageLogic': [
                    {'type': 'propagate',
                     'source_FG': FG_GF_H,
                     'target_FG': FG_S,
                     'DS_links': {
                         '1_1': '3_1',
                         '2_1': '4_1',
                         '2_2': '4_2'
                     }
                    },
                    {'type': 'propagate',
                     'source_FG': FG_GF_V,
                     'target_FG': FG_S,
                     'DS_links': {
                         '1_1': '3_1',
                         '2_1': '4_1',
                         '2_2': '4_2'
                     }
                    }
                ]
            })

    # HAZUS Hurricane
    elif DL_method == 'HAZUS MH HU':

        # Building characteristics
        year_built = int(BIM_in['yearBuilt'])
        roof_type = ap_RoofType[BIM_in['roofType']]
        occupancy = BIM_in['occupancy']
        stories = int(BIM_in['stories'])
        bldg_desc = BIM_in['buildingDescription']
        struct_type = BIM_in['structureType']
        V_ult = BIM_in['V_design']
        area = BIM_in['area']
        z0 = BIM_in['z0']

        # Meta-variables
        V_asd = 0.6 * V_ult

        # Flood risk // need high water zone for this
        FR = True

        # Hurricane-prone region
        HPR = V_ult > 115.0

        # Wind Borne Debris
        WBD = ((HPR) and
               ((V_ult > 140.0) or ((V_ult > 130.0) and (FR))))

        # attributes for WSF 1-2
        if ((roof_type != 'flt') and
            (stories <= 2) and
            (area < 2000.0)):
            # Secondary water resistance
            SWR = np.random.binomial(1, 0.6) == 1

            # Roof deck attachment //need to add year condition later
            if V_ult > 130.0:
                RDA = '8s' # 8d @ 6"/6"
            else:
                RDA = '8d' # 8d @ 6"/12"

            # Roof-wall connection // need to add year condition later
            if HPR:
                RWC = 'strap' # Strap
            else:
                RWC = 'tnail' # Toe-nail

            # Shutters // need to add year condition later
            shutters = WBD

            # Garage // need to add AG building descr later
            if shutters:
                if (bldg_desc is not None) and ('AG' in bldg_desc):
                    garage = 'sup' # SFBC 1994
                else:
                    garage = 'no'  # None
            else:
                if (bldg_desc is not None) and ('AG' in bldg_desc):
                    if year_built > 1989:
                        garage = 'std' # Standard
                    else:
                        garage = 'wkd' # Weak
                else:
                    garage = 'no' # None

            # Terrain // need to add terrain checks later (perhaps to the BIM)
                     #// assume suburban for now
            terrain = int(100 * z0)

            bldg_config = 'WSF{stories}_{roof_shape}_{SWR}_{RDA}_{RWC}_{garage}_{shutters}_{terrain}'.format(
                    stories = stories,
                    roof_shape = roof_type,
                    SWR = int(SWR),
                    RDA = RDA,
                    RWC = RWC,
                    garage = garage,
                    shutters=int(shutters),
                    terrain=terrain
                    )

        # attributes for WMUH 1-3
        else:
            # Secondary water resistance
            if roof_type == 'flt':
                SWR = 'null'
            else:
                SWR = int(np.random.binomial(1, 0.6) == 1)

            # Roof cover
            if roof_type in ['gab', 'hip']:
                roof_cover = 'null'
            else:
                if year_built < 1975:
                    roof_cover = 'bur'
                else:
                    roof_cover = 'spm'

            # Roof quality
            if roof_type in ['gab', 'hip']:
                roof_quality = 'null'
            else:
                if roof_cover == 'bur':
                    if year_built < 1989:
                        roof_quality = 'por'
                    else:
                        roof_quality = 'god'
                elif roof_cover == 'spm':
                    if year_built < 1984:
                        roof_quality = 'por'
                    else:
                        roof_quality = 'god'


            # Roof deck attachment //need to add year condition later
            if z0 == 0.35:
                if V_asd > 130.0:
                    RDA = '8s'  # 8d @ 6"/6"
                else:
                    RDA = '8d'  # 8d @ 6"/12"
            else:  # this would be light suburban, but we lump everything here for now
                if V_asd > 110.0:
                    RDA = '8s'  # 8d @ 6"/6"
                else:
                    RDA = '8d'  # 8d @ 6"/12"

            # Roof-wall connection // need to add year condition later
            if V_asd > 110.0:
                RWC = 'strap'  # Strap
            else:
                RWC = 'tnail'  # Toe-nail

            # Shutters // need to add year condition later
            shutters = WBD

            # Terrain // need to add terrain checks later (perhaps to the BIM)
            # // assume suburban for now
            terrain = int(100 * z0)

            stories = min(stories, 3)

            bldg_config = 'WMUH{stories}_{roof_shape}_{roof_cover}_{roof_quality}_{SWR}_{RDA}_{RWC}_{shutters}_{terrain}'.format(
                stories=stories,
                roof_shape=roof_type,
                SWR=SWR,
                roof_cover = roof_cover,
                roof_quality = roof_quality,
                RDA=RDA,
                RWC=RWC,
                shutters=int(shutters),
                terrain=terrain
            )

        loss_dict = {
            '_method': DL_method,
            'DamageModel': {
                'StructureType': struct_type
            },
            'LossModel': {
                'DecisionVariables': {
                    'ReconstructionCost': True
                },
                'ReplacementCost': 100
            },
            'ResponseModel': {
                'ResponseDescription': {
                    'EDP_Distribution': 'empirical',
                'Realizations': realization_count
            }
            },
            'Components':{
                bldg_config: []
            }
        }

    elif DL_method == 'FEMA P58':
        if BIM_in.get('asset_type',None) == 'Water_Pipe':

            material = BIM_in['material']

            if material in ['Asbestos cement', 'Cast iron']:
                # brittle material
                config = 'P0001a'
            else:
                # ductile material
                config = 'P0001b'

            segment_count = BIM_in['segment_count']
            segment_length = BIM_in['segments'][0]['length']
            cg_count = int(segment_length / (100 * ft))
            quantities = '1'
            for s in range(1, cg_count):
                quantities += ', 1'

            loss_dict = {
                "_method"            : "FEMA P58",
                "ResponseModel"      : {
                    "ResponseDescription": {
                        "EDP_Distribution" : "empirical",
                        "Realizations"     : "1000",   # need to fix this later
                        "CoupledAssessment": True
                    }
                },
                "DamageModel"        : {
                    "CollapseProbability": {
                        "Value": "0.0",
                    },
                },
                "LossModel"          : {
                    "ReplacementCost"  : "1",
                    "ReplacementTime"  : "180",
                    "DecisionVariables": {
                        "Injuries"          : False,
                        "ReconstructionCost": True,
                        "ReconstructionTime": True,
                        "RedTag"            : False
                    },
                },
                "Dependencies"       : {
                    "CostAndTime"        : True,
                    "Fragilities"        : "btw. Damage States",
                    "Quantities"         : "Independent",
                    "ReconstructionCosts": "Independent",
                    "ReconstructionTimes": "Independent",
                },
                "ComponentDataFolder": "c:/Adam/Dropbox/Kutatas/2019 SC Testbeds/Memphis/",
                "Components"         : {
                    config: [
                        {
                            "location"       : "all",
                            "direction"      : "1",
                            "median_quantity": quantities,
                            "unit"           : "ea",
                            "distribution"   : "N/A",
                        }
                    ],
                }
            }

    if event_time is not None:
            loss_dict['LossModel']['Inhabitants'].update({'EventTime': event_time})

    DL_input.update({'DamageAndLoss':loss_dict})

    DL_ap_path = DL_input_path[:-5]+'_ap.json'

    with open(DL_ap_path, 'w') as f:
        json.dump(DL_input, f, indent = 2)

    return DL_input, DL_ap_path
