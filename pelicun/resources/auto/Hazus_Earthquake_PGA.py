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

def auto_populate(AIM):
    """
    Automatically creates a performance model for PGA-based Hazus EQ analysis.

    Parameters
    ----------
    AIM: dict
        Asset Information Model - provides features of the asset that can be 
        used to infer attributes of the performance model.

    Returns
    -------
    AIM_ap: dict
        Extended Asset Information Model - extends the input AIM with additional
        features that were inferred. These features are typically used in 
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

    AIM_ap = AIM.copy()

    # get the building parameters
    bt = AIM['StructureType'] #building type

    # get the number of stories / height
    stories = AIM.get('NumberOfStories', None)

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

        stories = 1
        AIM_ap['BuildingType'] = bt

    # get the design level
    dl = AIM.get('DesignLevel', None)

    if dl == None:
        # If there is no DesignLevel provided, we assume that the YearBuilt is
        # available
        year_built = AIM['YearBuilt']

        if 'W1' in bt:
            DesignL = ap_DesignLevel_W1
        else:
            DesignL = ap_DesignLevel
        
        for year in sorted(DesignL.keys()):
            if year_built <= year:
                dl = DesignL[year]            
                break

        AIM_ap['DesignLevel'] = dl

    # get the occupancy class
    ot = AIM['OccupancyClass']

    CMP = pd.DataFrame(
        {f'LF.{bt}.{dl}': [  'ea',         1,          1,        1,   'N/A']},
        index = [         'Units','Location','Direction','Theta_0','Family']
    ).T

    DL_ap = {
        "Asset": {
            "ComponentAssignmentFile": "CMP_QNT.csv",
            "ComponentDatabase": "Hazus Earthquake",
            "NumberOfStories": f"{stories}",
            "OccupancyType": f"{ot}",
            "PlanArea": "1"
        },
        "Damage": {
            "DamageProcess": "Hazus Earthquake"
        },
        "Demands": {        
        },
        "Losses": {
            "BldgRepair": {
                "ConsequenceDatabase": "Hazus Earthquake",
                "MapApproach": "Automatic"
            }
        }
    }

    return AIM_ap, DL_ap, CMP