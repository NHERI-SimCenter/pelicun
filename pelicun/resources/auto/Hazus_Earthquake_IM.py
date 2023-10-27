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
def convertBridgeToHAZUSclass(AIM):
    structureType = AIM["bridge_class"]
    # if type(structureType)== str and len(structureType)>3 and structureType[:3] == "HWB" and 0 < int(structureType[3:]) and 29 > int(structureType[3:]):
    #     return AIM["bridge_class"]
    state = AIM["state_code"]
    yr_built = AIM["year_built"]
    num_span = AIM["nspans"]
    len_max_span = AIM["lmaxspan"]
    seismic = (int(state)==6 and int(yr_built)>=1975) or (int(state)!=6 and int(yr_built)>=1990)
    if not seismic and len_max_span > 150:
        return "HWB1"
    elif seismic and len_max_span > 150:
        return "HWB2"
    elif not seismic and num_span == 1:
        return "HWB3"
    elif seismic and num_span == 1:
        return "HWB4"
    elif not seismic and 101 <= structureType and structureType <= 106 and state != 6:
        return "HWB5"
    elif not seismic and 101 <= structureType and structureType <= 106 and state ==6:
        return "HWB6"
    elif seismic and 101 <= structureType and structureType <= 106:
        return "HWB7"
    elif not seismic and 205 <= structureType and structureType <= 206:
        return "HWB8"
    elif seismic and 205 <= structureType and structureType <= 206:
        return "HWB9"
    elif not seismic and 201 <= structureType and structureType <= 206:
        return "HWB10"
    elif seismic and 201 <= structureType and structureType <= 206:
        return "HWB11"
    elif not seismic and 301 <= structureType and structureType <= 306 and state != 6:
        return "HWB12"
    elif not seismic and 301 <= structureType and structureType <= 306 and state == 6:
        return "HWB13"
    elif seismic and 301 <= structureType and structureType <= 306:
        return "HWB14"
    elif not seismic and 402 <= structureType and structureType <= 410:
        return "HWB15"
    elif seismic and 402 <= structureType and structureType <= 410:
        return "HWB16"
    elif not seismic and 501 <= structureType and structureType <= 506 and state != 6:
        return "HWB17"
    elif not seismic and 501 <= structureType and structureType <= 506 and state == 6:
        return "HWB18"
    elif seismic and 501 <= structureType and structureType <= 506:
        return "HWB19"
    elif not seismic and 605 <= structureType and structureType <= 606:
        return "HWB20"
    elif seismic and 605 <= structureType and structureType <= 606:
        return "HWB21"
    elif not seismic and 601 <= structureType and structureType <= 607:
        return "HWB22"
    elif seismic and 601 <= structureType and structureType <= 607:
        return "HWB23"
    elif not seismic and 301 <= structureType and structureType <= 306 and state != 6:
        return "HWB24"
    elif not seismic and 301 <= structureType and structureType <= 306 and state == 6:
        return "HWB25"
    elif not seismic and 402 <= structureType and structureType <= 410 and state != 6:
        return "HWB26"
    elif not seismic and 402 <= structureType and structureType <= 410 and state == 6:
        return "HWB27"
    else:
        return "HWB28"

def convertTunnelToHAZUSclass(AIM):
    if "Bored" in AIM["cons_type"] or "Drilled" in AIM["cons_type"]:
        return "HTU1"
    elif "Cut" in AIM["cons_type"] or "Cover" in AIM["cons_type"]:
        return "HTU2"
    else:
        return "HTU2" # HTU2 fragility function is more conservative than HTU1. Select HTU2 for unclassfied tunnels
def convertRoadToHAZUSclass(AIM):
    if AIM["road_type"]=="primary" or AIM["road_type"] == "secondary":
        return "HRD1"
    elif AIM["road_type"]=="residential":
        return "HRD2"
    else:
        return "HRD2" # many unclassified roads are urban roads
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
    print("JZ Debug: the Hazus_Earthquake_IM.py auto_popu is used")
    config = AIM.copy()
    AIM = AIM.get('GeneralInformation', None)
    AIM_ap = AIM.copy()
    assetType = config["assetType"]
    groun_failure = config["Applications"]["DL"]["ApplicationData"]["ground_failure"]
    if assetType=="Buildings":
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
        if groun_failure:
            foundation_type = 'S'
            CMP = pd.DataFrame(
                {f'LF.{bt}.{dl}': [  'ea',         1,          1,        1,   'N/A'],
                f'GF.H.{foundation_type}':['ea', 1, 1, 1, 'N/A'],
                f'GF.V.{foundation_type}':['ea', 1, 3, 1, 'N/A']},
                index = [         'Units','Location','Direction','Theta_0','Family']
            ).T
        else:
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
    elif assetType == "TransportationNetwork":
        inf_type = AIM["assetSubtype"]
        if inf_type == "hwy_bridge":
            # get the bridge class
            bt = convertBridgeToHAZUSclass(AIM)
            AIM_ap['BridgeHazusClass'] = bt

            CMP = pd.DataFrame(
                {f'HWB.GS.{bt[3:]}': [  'ea',         1,          1,        1,   'N/A'],
                f'HWB.GF':[  'ea',         1,          1,        1,   'N/A']},
                index = [         'Units','Location','Direction','Theta_0','Family']
            ).T

            DL_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake Transportation",
                    "BridgeHazusClass": bt,
                    "PlanArea": "1"
                },
                "Damage": {
                    "DamageProcess": "Hazus Earthquake Transportation"
                },
                "Demands": {        
                },
                "Losses": {
                    "BldgRepair": {
                        "ConsequenceDatabase": "Hazus Earthquake Transportation",
                        "MapApproach": "Automatic"
                    }
                }
            }
        elif inf_type == "hwy_tunnel":
            # get the tunnel class
            tt = convertTunnelToHAZUSclass(AIM)
            AIM_ap['TunnelHazusClass'] = tt

            CMP = pd.DataFrame(
                {f'HTU.GS.{tt[3:]}': [  'ea',         1,          1,        1,   'N/A'],
                f'HTU.GF':[  'ea',         1,          1,        1,   'N/A']},
                index = [         'Units','Location','Direction','Theta_0','Family']
            ).T

            DL_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake Transportation",
                    "TunnelHazusClass": tt,
                    "PlanArea": "1"
                },
                "Damage": {
                    "DamageProcess": "Hazus Earthquake Transportation"
                },
                "Demands": {        
                },
                "Losses": {
                    "BldgRepair": {
                        "ConsequenceDatabase": "Hazus Earthquake Transportation",
                        "MapApproach": "Automatic"
                    }
                }
            }
        elif inf_type == "roadway":
            # get the road class
            rt = convertRoadToHAZUSclass(AIM)
            AIM_ap['RoadHazusClass'] = rt

            CMP = pd.DataFrame(
                {f'HRD.GF.{rt[3:]}':[  'ea',         1,          1,        1,   'N/A']},
                index = [         'Units','Location','Direction','Theta_0','Family']
            ).T

            DL_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake Transportation",
                    "RoadHazusClass": rt,
                    "PlanArea": "1"
                },
                "Damage": {
                    "DamageProcess": "Hazus Earthquake Transportation"
                },
                "Demands": {        
                },
                "Losses": {
                    "BldgRepair": {
                        "ConsequenceDatabase": "Hazus Earthquake Transportation",
                        "MapApproach": "Automatic"
                    }
                }
            }
        else:
            print("subtype not supported in HWY")
    else:
        print(f"AssetType: {assetType} is not supported in Hazus Earthquake IM DL method")

    return AIM_ap, DL_ap, CMP