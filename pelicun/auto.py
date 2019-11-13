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
    1950: 'Pre-Code',
    1970: 'Low-Code',
    1990: 'Moderate-Code',
    2100: 'High-Code'
}

ap_Occupancy = {
    'Other/Unknown': 'RES1',
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

def auto_populate(DL_input_path, EDP_input_path,
                  DL_method, realization_count):

    with open(DL_input_path, 'r') as f:
        DL_input = json.load(f)

    EDP_input = pd.read_csv(EDP_input_path, sep='\s+', header=0,
                            index_col=0)

    if 'GeneralInformation' in DL_input.keys():
        BIM_in = DL_input['GeneralInformation']
    elif 'GI' in DL_input.keys():
        BIM_in = DL_input['GI']

    # HAZUS Earthquake
    if DL_method == 'HAZUS MH EQ':

        bt = BIM_in['structType']

        if bt == 'RV.structType':
            bt = EDP_input['structType'].values[0]

        year_built = BIM_in['yearBuilt']
        stories = BIM_in['numStory']

        if bt not in ['W1', 'W2']:
            if stories <= 3:
                bt += 'L'
            elif stories <= 7:
                bt += 'M'
            else:
                bt += 'H'

        ot = ap_Occupancy[BIM_in['occupancy']]

        loss_dict = {
            '_method': DL_method,
            'DamageModel': {
                'StructureType': bt
            },
            'LossModel': {
                'DecisionVariables': {
                    'ReconstructionCost': True,
                    'ReconstructionTime': True,
                    'Injuries': False
                },
                'Inhabitants': {
                    'OccupancyType': ot,
                    'PeakPopulation': '1'
                },
                'ReplacementCost': BIM_in['replacementCost'],
                'ReplacementTime': BIM_in['replacementTime']
            },
            'ResponseModel': {
                'ResponseDescription': {
                'Realizations': realization_count
                },
                "AdditionalUncertainty": {
                    "GroundMotion": "0.15",
                    "Modeling"    : "0.30"
                }
            }
        }


        for year in sorted(ap_DesignLevel.keys()):
            if year_built <= year:
                loss_dict['DamageModel'].update(
                    {'DesignLevel': ap_DesignLevel[year]})
                break
        dl = convert_design_level[loss_dict['DamageModel']['DesignLevel']]
        if 'C3' in bt:
            if dl not in ['LC', 'PC']:
                dl = 'LC'

        loss_dict.update({
            'Components': {
                'S-{}-{}-{}'.format(bt, dl ,ot) : [],
                'NSA-{}-{}'.format(dl ,ot): [],
                'NSD-{}'.format(ot): []
            }})

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

    DL_input.update({'DamageAndLoss':loss_dict})

    DL_ap_path = DL_input_path[:-5]+'_ap.json'

    with open(DL_ap_path, 'w') as f:
        json.dump(DL_input, f, indent = 2)

    return DL_input, DL_ap_path
