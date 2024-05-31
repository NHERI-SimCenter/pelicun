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
# Frank McKenna
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

import pandas as pd

from MetaVarRulesets import parse_BIM
from BldgClassRulesets import building_class
from WindWSFRulesets import WSF_config
from WindWMUHRulesets import WMUH_config


def auto_populate(AIM):
    """
    Populates the DL model for hurricane assessments in Atlantic County, NJ

    Assumptions:
    - Everything relevant to auto-population is provided in the Buiding
    Information Model (AIM).
    - The information expected in the AIM file is described in the parse_GI
    method.

    Parameters
    ----------
    AIM: dictionary
        Contains the information that is available about the asset and will be
        used to auto-popualate the damage and loss model.

    Returns
    -------
    GI_ap: dictionary
        Containes the extended BIM data.
    DL_ap: dictionary
        Contains the auto-populated loss model.
    """

    # extract the General Information
    GI = AIM.get('GeneralInformation', None)

    # parse the GI data
    GI_ap = parse_BIM(
        GI,
        location="LA",
        hazards=[
            'wind',
        ],
    )

    # identify the building class
    bldg_class = building_class(GI_ap, hazard='wind')
    GI_ap.update({'HazusClassW': bldg_class})

    # prepare the building configuration string
    if bldg_class == 'WSF':
        bldg_config = WSF_config(GI_ap)
    elif bldg_class == 'WMUH':
        bldg_config = WMUH_config(GI_ap)
    else:
        raise ValueError(
            f"Building class {bldg_class} not recognized by the "
            f"auto-population routine."
        )

    # drop keys of internal variables from GI_ap dict
    internal_vars = ['V_ult', 'V_asd']
    for var in internal_vars:
        try:
            GI_ap.pop(var)
        except KeyError:
            pass

    # prepare the component assignment
    CMP = pd.DataFrame(
        {f'{bldg_config}': ['ea', 1, 1, 1, 'N/A']},
        index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
    ).T

    DL_ap = {
        "Asset": {
            "ComponentAssignmentFile": "CMP_QNT.csv",
            "ComponentDatabase": "Hazus Hurricane",
            "NumberOfStories": f"{GI_ap['NumberOfStories']}",
            "OccupancyType": f"{GI_ap['OccupancyClass']}",
            "PlanArea": f"{GI_ap['PlanArea']}",
        },
        "Damage": {"DamageProcess": "Hazus Hurricane"},
        "Demands": {},
        "Losses": {
            "BldgRepair": {
                "ConsequenceDatabase": "Hazus Hurricane",
                "MapApproach": "Automatic",
                "DecisionVariables": {
                    "Cost": True,
                    "Carbon": False,
                    "Energy": False,
                    "Time": False,
                },
            }
        },
    }

    return GI_ap, DL_ap, CMP
