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

from WindMetaVarRulesets import parse_BIM
from BuildingClassRulesets import building_class
from FloodAssmRulesets import Assm_config
from FloodClassRulesets import FL_config
from WindCECBRulesets import CECB_config
from WindCERBRulesets import CERB_config
from WindMECBRulesets import MECB_config
from WindMERBRulesets import MERB_config
from WindMHRulesets import MH_config
from WindMLRIRulesets import MLRI_config
from WindMLRMRulesets import MLRM_config
from WindMMUHRulesets import MMUH_config
from WindMSFRulesets import MSF_config
from WindSECBRulesets import SECB_config
from WindSERBRulesets import SERB_config
from WindSPMBRulesets import SPMB_config
from WindWMUHRulesets import WMUH_config
from WindWSFRulesets import WSF_config


def auto_populate(aim):
    """
    Populates the DL model for hurricane assessments in Atlantic County, NJ

    Assumptions:
    - Everything relevant to auto-population is provided in the Buiding
    Information Model (AIM).
    - The information expected in the AIM file is described in the parse_AIM
    method.

    Parameters
    ----------
    aim: dictionary
        Contains the information that is available about the asset and will be
        used to auto-popualate the damage and loss model.

    Returns
    -------
    GI_ap: dictionary
        Containes the extended AIM data.
    DL_ap: dictionary
        Contains the auto-populated loss model.
    """

    # extract the General Information
    GI = aim.get('GeneralInformation', None)

    # parse the GI data
    GI_ap = parse_BIM(GI, location="NJ", hazards=['wind', 'inundation'])

    # identify the building class
    bldg_class = building_class(GI_ap, hazard='wind')

    # prepare the building configuration string
    if bldg_class == 'WSF':
        bldg_config = WSF_config(GI_ap)
    elif bldg_class == 'WMUH':
        bldg_config = WMUH_config(GI_ap)
    elif bldg_class == 'MSF':
        bldg_config = MSF_config(GI_ap)
    elif bldg_class == 'MMUH':
        bldg_config = MMUH_config(GI_ap)
    elif bldg_class == 'MLRM':
        bldg_config = MLRM_config(GI_ap)
    elif bldg_class == 'MLRI':
        bldg_config = MLRI_config(GI_ap)
    elif bldg_class == 'MERB':
        bldg_config = MERB_config(GI_ap)
    elif bldg_class == 'MECB':
        bldg_config = MECB_config(GI_ap)
    elif bldg_class == 'CECB':
        bldg_config = CECB_config(GI_ap)
    elif bldg_class == 'CERB':
        bldg_config = CERB_config(GI_ap)
    elif bldg_class == 'SPMB':
        bldg_config = SPMB_config(GI_ap)
    elif bldg_class == 'SECB':
        bldg_config = SECB_config(GI_ap)
    elif bldg_class == 'SERB':
        bldg_config = SERB_config(GI_ap)
    elif bldg_class == 'MH':
        bldg_config = MH_config(GI_ap)
    else:
        raise ValueError(
            f"Building class {bldg_class} not recognized by the "
            f"auto-population routine."
        )

    # prepare the flood rulesets
    fld_config = FL_config(GI_ap)

    # prepare the assembly loss compositions
    hu_assm, fl_assm = Assm_config(GI_ap)

    # prepare the component assignment
    CMP = pd.DataFrame(
        {
            f'{bldg_config}': ['ea', 1, 1, 1, 'N/A'],
            f'{fld_config}': ['ea', 1, 1, 1, 'N/A'],
        },
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
