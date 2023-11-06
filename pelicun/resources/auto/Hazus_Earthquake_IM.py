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

    #TODO: replace labels in AIM with standard CamelCase versions
    structureType = AIM["BridgeClass"]
    # if type(structureType)== str and len(structureType)>3 and structureType[:3] == "HWB" and 0 < int(structureType[3:]) and 29 > int(structureType[3:]):
    #     return AIM["bridge_class"]
    state = AIM["StateCode"]
    yr_built = AIM["YearBuilt"] 
    num_span = AIM["NumOfSpans"]
    len_max_span = AIM["MaxSpanLength"] 

    seismic = ((int(state)==6 and int(yr_built)>=1975) or 
               (int(state)!=6 and int(yr_built)>=1990))

    # Use a catch-all, other class by default
    bridge_class = "HWB28"
    
    if len_max_span > 150:
        if not seismic:
            bridge_class = "HWB1"
        else:
            bridge_class = "HWB2"

    elif num_span == 1:
        if not seismic:
            bridge_class = "HWB3"
        else:
            bridge_class = "HWB4"

    elif structureType in list(range(101,107)):
        if not seismic:
            if state != 6:
                bridge_class = "HWB5"
            else:
                bridge_class = "HWB6"
        else:
            bridge_class = "HWB7"

    elif structureType in [205,206]:
        if not seismic:
            bridge_class = "HWB8"
        else:
            bridge_class = "HWB9"

    elif structureType in list(range(201,207)):
        if not seismic:
            bridge_class = "HWB10"
        else:
            bridge_class = "HWB11"

    elif structureType in list(range(301,307)):
        if not seismic:
            if len_max_span>=20:
                if state != 6:
                    bridge_class = "HWB12"
                else:
                    bridge_class = "HWB13"
            else:
                if state != 6:
                    bridge_class = "HWB24"
                else:
                    bridge_class = "HWB25"
        else:
            bridge_class = "HWB14"

    elif structureType in list(range(402,411)):
        if not seismic:
            if len_max_span>=20:
                bridge_class = "HWB15"
            elif state != 6:
                bridge_class = "HWB26"
            else:
                bridge_class = "HWB27"
        else:
            bridge_class = "HWB16"

    elif structureType in list(range(501,507)):
        if not seismic:
            if state != 6:
                bridge_class = "HWB17"
            else:
                bridge_class = "HWB18"
        else:
            bridge_class = "HWB19"

    elif structureType in [605,606]:
        if not seismic:
            bridge_class = "HWB20"
        else:
            bridge_class = "HWB21"

    elif structureType in list(range(601,608)):
        if not seismic:
            bridge_class = "HWB22"
        else:
            bridge_class = "HWB23"
    
    
    #TODO: review and add HWB24-27 rules
    #TODO: also double check rules for HWB10-11 and HWB22-23

    return bridge_class


    # original code by JZ
    """
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
    """

def convertTunnelToHAZUSclass(AIM):

    if ("Bored" in AIM["ConstructType"]) or ("Drilled" in AIM["ConstructType"]):
        return "HTU1"
    elif ("Cut" in AIM["ConstructType"]) or ("Cover" in AIM["ConstructType"]):
        return "HTU2"
    else:
        # Select HTU2 for unclassfied tunnels because it is more conservative. 
        return "HTU2" 

def convertRoadToHAZUSclass(AIM):

    if AIM["RoadType"] in ["Primary", "Secondary"]:
        return "HRD1"

    elif AIM["RoadType"]=="Residential":
        return "HRD2"

    else:
        # many unclassified roads are urban roads
        return "HRD2" 

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

            stories = 1
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
        ot = GI['OccupancyClass']

        CMP = pd.DataFrame(
                {f'LF.{bt}.{dl}': [  'ea',         1,          1,        1,   'N/A']},
                index = [         'Units','Location','Direction','Theta_0','Family']
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
        
        DL_ap = {
            "Asset": {
                "ComponentAssignmentFile": "CMP_QNT.csv",
                "ComponentDatabase": "Hazus Earthquake - Buildings",
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
                    "ConsequenceDatabase": "Hazus Earthquake - Buildings",
                    "MapApproach": "Automatic"
                }
            }
        }

    elif assetType == "TransportationNetwork":

        inf_type = GI["assetSubtype"]
        
        if inf_type == "HwyBridge":

            # get the bridge class
            bt = convertBridgeToHAZUSclass(GI)
            GI_ap['BridgeHazusClass'] = bt

            CMP = pd.DataFrame(
                {f'HWB.GS.{bt[3:]}': [  'ea',         1,          1,        1,   'N/A'],
                 f'HWB.GF':          [  'ea',         1,          1,        1,   'N/A']},
                index = [            'Units','Location','Direction','Theta_0','Family']
            ).T

            DL_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake - Transportation",
                    "BridgeHazusClass": bt,
                    "PlanArea": "1"
                },
                "Damage": {
                    "DamageProcess": "Hazus Earthquake"
                },
                "Demands": {        
                },
                "Losses": {
                    "BldgRepair": {
                        "ConsequenceDatabase": "Hazus Earthquake - Transportation",
                        "MapApproach": "Automatic"
                    }
                }
            }

        elif inf_type == "HwyTunnel":

            # get the tunnel class
            tt = convertTunnelToHAZUSclass(GI)
            GI_ap['TunnelHazusClass'] = tt

            CMP = pd.DataFrame(
                {f'HTU.GS.{tt[3:]}': [  'ea',         1,          1,        1,   'N/A'],
                 f'HTU.GF':          [  'ea',         1,          1,        1,   'N/A']},
                index = [            'Units','Location','Direction','Theta_0','Family']
            ).T

            DL_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake - Transportation",
                    "TunnelHazusClass": tt,
                    "PlanArea": "1"
                },
                "Damage": {
                    "DamageProcess": "Hazus Earthquake"
                },
                "Demands": {        
                },
                "Losses": {
                    "BldgRepair": {
                        "ConsequenceDatabase": "Hazus Earthquake - Transportation",
                        "MapApproach": "Automatic"
                    }
                }
            }
        elif inf_type == "Roadway":

            # get the road class
            rt = convertRoadToHAZUSclass(GI)
            GI_ap['RoadHazusClass'] = rt

            CMP = pd.DataFrame(
                {f'HRD.GF.{rt[3:]}':[  'ea',         1,          1,        1,   'N/A']},
                index = [           'Units','Location','Direction','Theta_0','Family']
            ).T

            DL_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake - Transportation",
                    "RoadHazusClass": rt,
                    "PlanArea": "1"
                },
                "Damage": {
                    "DamageProcess": "Hazus Earthquake"
                },
                "Demands": {        
                },
                "Losses": {
                    "BldgRepair": {
                        "ConsequenceDatabase": "Hazus Earthquake - Transportation",
                        "MapApproach": "Automatic"
                    }
                }
            }
        else:
            print("subtype not supported in HWY")
    else:
        print(f"AssetType: {assetType} is not supported in Hazus Earthquake IM DL method")

    return GI_ap, DL_ap, CMP