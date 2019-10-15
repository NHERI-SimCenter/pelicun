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

import json

ap_DesignLevel = {
    1950: 'Pre-Code',
    1970: 'Low-Code',
    1990: 'Moderate-Code',
    2100: 'High-Code'
}

ap_Occupancy = {
    'Residential': "RES1",
    'Retail': "COM1"
}

ap_RoofType = {
    'hip': 'hip',
    'hipped': 'hip',
    'gabled': 'gab',
    'gable': 'gab',
    'flat': 'hip'   # we need to change this later!
}

convert_design_level = {
        'High-Code'    : 'HC',
        'Moderate-Code': 'MC',
        'Low-Code'     : 'LC',
        'Pre-Code'     : 'PC'
    }

def auto_populate(DL_input_path, DL_method, realization_count):

    with open(DL_input_path, 'r') as f:
        DL_input = json.load(f)

    if 'GeneralInformation' in DL_input.keys():
        BIM_in = DL_input['GeneralInformation']
    elif 'GI' in DL_input.keys():
        BIM_in = DL_input['GI']

    # HAZUS Earthquake
    if DL_method == 'HAZUS MH EQ':

        bt = BIM_in['structType']
        ot = ap_Occupancy[BIM_in['occupancy']]

        loss_dict = {
            '_method': DL_method,
            'BuildingDamage': {
                'ReplacementCost': BIM_in['replacementCost'],
                'ReplacementTime': BIM_in['replacementTime'],
                'StructureType': bt,
            },
            'UncertaintyQuantification': {
                'Realizations': realization_count
            },
            'Inhabitants': {
                'PeakPopulation': "1",
                'OccupancyType': ot
            },
            'Components': []
        }

        year_built = BIM_in['yearBuilt']
        for year in sorted(ap_DesignLevel.keys()):
            if year_built <= year:
                loss_dict['BuildingDamage'].update(
                    {'DesignLevel': ap_DesignLevel[year]})
                break
        dl = convert_design_level[loss_dict['BuildingDamage']['DesignLevel']]

        components = [
            {'ID': 'S-{}-{}-{}'.format(bt, dl ,ot), 'structural': True},
            {'ID': 'NSA-{}-{}'.format(dl ,ot),      'structural': False},
            {'ID': 'NSD-{}'.format(ot),             'structural': False}
        ]

        loss_dict['Components'] = components

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
            if True:
                garage = 'sup' # SFBC 1994
            else:
                garage = 'no'  # None
        else:
            if True:
                if year_built > 1989:
                    garage = 'std' # Standard
                else:
                    garage = 'wkd' # Weak
            else:
                garage = 'no' # None

        # Terrain // need to add terrain checks later (perhaps to the BIM)
                 #// assume suburban for now
        terrain = 35

        # temp fix
        stories = min(stories, 2)

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

        loss_dict = {
            '_method': DL_method,
            'BuildingDamage': {
                'ReplacementCost': 100,
            },
            "BuildingResponse": {
                "EDP_Distribution": "empirical"
            },
            "Components": [
                {
                    "ID": bldg_config #"WSF2_gab_1_6s_tnail_no_1_35"
                }
            ],
            "DecisionVariables": {
                "ReconstructionCost": True
            },
            'UncertaintyQuantification': {
                'Realizations': realization_count
            }
        }

    DL_input.update({'DamageAndLoss':loss_dict})

    DL_ap_path = DL_input_path[:-5]+'_ap.json'

    with open(DL_ap_path, 'w') as f:
        json.dump(DL_input, f, indent = 2)

    return DL_input, DL_ap_path
