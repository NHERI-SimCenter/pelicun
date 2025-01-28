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
"""Hazus Earthquake IM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import pelicun
from pelicun.assessment import DLCalculationAssessment

ap_design_level = {1940: 'LC', 1975: 'MC', 2100: 'HC'}
# original:
# ap_DesignLevel = {1940: 'PC', 1940: 'LC', 1975: 'MC', 2100: 'HC'}
# Note that the duplicated key is ignored, and Python keeps the last
# entry.

ap_design_level_w1 = {0: 'LC', 1975: 'MC', 2100: 'HC'}
# original:
# ap_DesignLevel_W1 = {0: 'PC', 0: 'LC', 1975: 'MC', 2100: 'HC'}
# same thing applies

ap_occupancy = {
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
    'Parking': 'COM10',
}


# Convert common length units
def convertUnits(value, unit_in, unit_out):
    """
    Convert units.
    """
    aval_types = ['m', 'mm', 'cm', 'km', 'inch', 'ft', 'mile']
    m = 1.0
    mm = 0.001 * m
    cm = 0.01 * m
    km = 1000 * m
    inch = 0.0254 * m
    ft = 12.0 * inch
    mile = 5280.0 * ft
    scale_map = {
        'm': m,
        'mm': mm,
        'cm': cm,
        'km': km,
        'inch': inch,
        'ft': ft,
        'mile': mile,
    }
    if (unit_in not in aval_types) or (unit_out not in aval_types):
        print(
            f'The unit {unit_in} or {unit_out} '
            f'are used in auto_population but not supported'
        )
        return None
    return value * scale_map[unit_in] / scale_map[unit_out]


def getHAZUSBridgeK3DModifier(hazus_class, aim):
    # In HAZUS, the K_3D for HWB28 is undefined, so we return 1, i.e., no scaling
    # The K-3D factors for HWB3 and HWB4 are defined as EQ1, which leads to division by zero
    # This is an error in the HAZUS documentation, and we assume that the factors are 1 for these classes
    mapping = {
        'HWB1': 1,
        'HWB2': 1,
        'HWB3': 1,
        'HWB4': 1,
        'HWB5': 1,
        'HWB6': 1,
        'HWB7': 1,
        'HWB8': 2,
        'HWB9': 3,
        'HWB10': 2,
        'HWB11': 3,
        'HWB12': 4,
        'HWB13': 4,
        'HWB14': 1,
        'HWB15': 5,
        'HWB16': 3,
        'HWB17': 1,
        'HWB18': 1,
        'HWB19': 1,
        'HWB20': 2,
        'HWB21': 3,
        'HWB22': 2,
        'HWB23': 3,
        'HWB24': 6,
        'HWB25': 6,
        'HWB26': 7,
        'HWB27': 7,
        'HWB28': 8,
    }
    factors = {
        1: (0.25, 1),
        2: (0.33, 0),
        3: (0.33, 1),
        4: (0.09, 1),
        5: (0.05, 0),
        6: (0.2, 1),
        7: (0.1, 0),
    }
    if hazus_class in ['HWB3', 'HWB4', 'HWB28']:
        return 1
    else:
        n = aim['NumOfSpans']
        a = factors[mapping[hazus_class]][0]
        b = factors[mapping[hazus_class]][1]
        return 1 + a / (
            n - b
        )  # This is the original form in Mander and Basoz (1999)


def convertBridgeToHAZUSclass(aim):  # noqa: C901
    # TODO: replace labels in AIM with standard CamelCase versions
    structure_type = aim['BridgeClass']
    # if (
    #     type(structure_type) == str
    #     and len(structure_type) > 3
    #     and structure_type[:3] == "HWB"
    #     and 0 < int(structure_type[3:])
    #     and 29 > int(structure_type[3:])
    # ):
    #     return AIM["bridge_class"]
    state = aim['StateCode']
    yr_built = aim['YearBuilt']
    num_span = aim['NumOfSpans']
    len_max_span = aim['MaxSpanLength']
    len_unit = aim['units']['length']
    len_max_span = convertUnits(len_max_span, len_unit, 'm')

    seismic = (int(state) == 6 and int(yr_built) >= 1975) or (
        int(state) != 6 and int(yr_built) >= 1990
    )
    # Use a catch-all, other class by default
    bridge_class = 'HWB28'

    if len_max_span > 150:
        if not seismic:
            bridge_class = 'HWB1'
        else:
            bridge_class = 'HWB2'

    elif num_span == 1:
        if not seismic:
            bridge_class = 'HWB3'
        else:
            bridge_class = 'HWB4'

    elif structure_type in list(range(101, 107)):
        if not seismic:
            if state != 6:
                bridge_class = 'HWB5'
            else:
                bridge_class = 'HWB6'
        else:
            bridge_class = 'HWB7'

    elif structure_type in [205, 206]:
        if not seismic:
            bridge_class = 'HWB8'
        else:
            bridge_class = 'HWB9'

    elif structure_type in list(range(201, 207)):
        if not seismic:
            bridge_class = 'HWB10'
        else:
            bridge_class = 'HWB11'

    elif structure_type in list(range(301, 307)):
        if not seismic:
            if len_max_span >= 20:
                if state != 6:
                    bridge_class = 'HWB12'
                else:
                    bridge_class = 'HWB13'
            else:
                if state != 6:
                    bridge_class = 'HWB24'
                else:
                    bridge_class = 'HWB25'
        else:
            bridge_class = 'HWB14'

    elif structure_type in list(range(402, 411)):
        if not seismic:
            if len_max_span >= 20:
                bridge_class = 'HWB15'
            elif state != 6:
                bridge_class = 'HWB26'
            else:
                bridge_class = 'HWB27'
        else:
            bridge_class = 'HWB16'

    elif structure_type in list(range(501, 507)):
        if not seismic:
            if state != 6:
                bridge_class = 'HWB17'
            else:
                bridge_class = 'HWB18'
        else:
            bridge_class = 'HWB19'

    elif structure_type in [605, 606]:
        if not seismic:
            bridge_class = 'HWB20'
        else:
            bridge_class = 'HWB21'

    elif structure_type in list(range(601, 608)):
        if not seismic:
            bridge_class = 'HWB22'
        else:
            bridge_class = 'HWB23'

    # TODO: review and add HWB24-27 rules
    # TODO: also double check rules for HWB10-11 and HWB22-23

    return bridge_class


def getHAZUSBridgePGDModifier(hazus_class, aim):
    # This is the original modifier in HAZUS, which gives inf if Skew is 0
    # modifier1 = 0.5*AIM['StructureLength']/(AIM['DeckWidth']*AIM['NumOfSpans']*np.sin(AIM['Skew']/180.0*np.pi))
    # Use the modifier that is corrected from HAZUS manual to achieve the asymptotic behavior
    # Where longer bridges, narrower bridges, less span and higher skew leads to lower modifier (i.e., more fragile bridges)
    modifier1 = (
        aim['DeckWidth']
        * aim['NumOfSpans']
        * np.sin((90 - aim['Skew']) / 180.0 * np.pi)
        / (aim['StructureLength'] * 0.5)
    )
    modifier2 = np.sin((90 - aim['Skew']) / 180.0 * np.pi)
    mapping = {
        'HWB1': (1, 1),
        'HWB2': (1, 1),
        'HWB3': (1, 1),
        'HWB4': (1, 1),
        'HWB5': (modifier1, modifier1),
        'HWB6': (modifier1, modifier1),
        'HWB7': (modifier1, modifier1),
        'HWB8': (1, modifier2),
        'HWB9': (1, modifier2),
        'HWB10': (1, modifier2),
        'HWB11': (1, modifier2),
        'HWB12': (modifier1, modifier1),
        'HWB13': (modifier1, modifier1),
        'HWB14': (modifier1, modifier1),
        'HWB15': (1, modifier2),
        'HWB16': (1, modifier2),
        'HWB17': (modifier1, modifier1),
        'HWB18': (modifier1, modifier1),
        'HWB19': (modifier1, modifier1),
        'HWB20': (1, modifier2),
        'HWB21': (1, modifier2),
        'HWB22': (modifier1, modifier1),
        'HWB23': (modifier1, modifier1),
        'HWB24': (modifier1, modifier1),
        'HWB25': (modifier1, modifier1),
        'HWB26': (1, modifier2),
        'HWB27': (1, modifier2),
        'HWB28': (1, 1),
    }
    return mapping[hazus_class][0], mapping[hazus_class][1]


def convertTunnelToHAZUSclass(aim) -> str:
    if 'Bored' in aim['ConstructType'] or 'Drilled' in aim['ConstructType']:
        return 'HTU1'
    elif 'Cut' in aim['ConstructType'] or 'Cover' in aim['ConstructType']:
        return 'HTU2'
    else:
        # Select HTU2 for unclassified tunnels because it is more conservative.
        return 'HTU2'


def convertRoadToHAZUSclass(aim) -> str:
    if aim['RoadType'] in ['Primary', 'Secondary']:
        return 'HRD1'

    elif aim['RoadType'] == 'Residential':
        return 'HRD2'

    else:
        # many unclassified roads are urban roads
        return 'HRD2'


def convert_story_rise(structure_type, stories):
    if structure_type in ['W1', 'W2', 'S3', 'PC1', 'MH']:
        # These archetypes have no rise information in their IDs
        rise = None

    else:
        rise = None  # Default value
        # First, check if we have valid story information
        try:
            stories = int(stories)

        except (ValueError, TypeError):
            msg = (
                'Missing "NumberOfStories" information, '
                'cannot infer `rise` attribute of archetype'
            )
            raise ValueError(msg)  # noqa: B904

        if structure_type == 'RM1':
            rise = 'L' if stories <= 3 else 'M'

        elif structure_type == 'URM':
            rise = 'L' if stories <= 2 else 'M'

        elif structure_type in [
            'S1',
            'S2',
            'S4',
            'S5',
            'C1',
            'C2',
            'C3',
            'PC2',
            'RM2',
        ]:
            if stories <= 3:
                rise = 'L'

            elif stories <= 7:
                rise = 'M'

            else:
                rise = 'H'

    return rise


def getHAZUSBridgeSlightDamageModifier(hazus_class, aim):
    if hazus_class in [
        'HWB1',
        'HWB2',
        'HWB5',
        'HWB6',
        'HWB7',
        'HWB8',
        'HWB9',
        'HWB12',
        'HWB13',
        'HWB14',
        'HWB17',
        'HWB18',
        'HWB19',
        'HWB20',
        'HWB21',
        'HWB24',
        'HWB25',
        'HWB28',
    ]:
        return None
    demand_path = Path(aim['DL']['Demands']['DemandFilePath']).resolve()
    sample_size = int(aim['DL']['Demands']['SampleSize'])
    length_unit = aim['GeneralInformation']['units']['length']
    coupled_demands = aim['Applications']['DL']['ApplicationData']['coupled_EDP']
    assessment = DLCalculationAssessment(config_options=None)
    assessment.calculate_demand(
        demand_path=demand_path,
        collapse_limits=None,
        length_unit=length_unit,
        demand_calibration=None,
        sample_size=sample_size,
        demand_cloning=None,
        residual_drift_inference=None,
        coupled_demands=coupled_demands,
    )
    demand_sample, _ = assessment.demand.save_sample(save_units=True)
    edp_types = demand_sample.columns.get_level_values(level='type')
    if (edp_types == 'SA_0.3').sum() != 1:
        msg = (
            'The demand file does not contain the required EDP type SA_0.3'
            ' or contains multiple instances of it.'
        )
        raise ValueError(msg)
    sa_0p3 = demand_sample.loc[  # noqa: PD011
        :, demand_sample.columns.get_level_values(level='type') == 'SA_0.3'
    ].values.flatten()
    if (edp_types == 'SA_1.0').sum() != 1:
        msg = (
            'The demand file does not contain the required EDP type SA_1.0'
            ' or contains multiple instances of it.'
        )
        raise ValueError(msg)
    sa_1p0 = demand_sample.loc[  # noqa: PD011
        :, demand_sample.columns.get_level_values(level='type') == 'SA_1.0'
    ].values.flatten()

    ratio = 2.5 * sa_1p0 / sa_0p3
    operation = [
        f'*{ratio[i]}' if ratio[i] <= 1.0 else '*1.0' for i in range(len(ratio))
    ]

    assert len(operation) == sample_size

    return operation


def auto_populate(aim):  # noqa: C901
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
    gi = aim.get('GeneralInformation', None)

    if gi is None:
        # TODO: show an error message
        pass

    # initialize the auto-populated GI
    gi_ap = gi.copy()

    asset_type = aim['assetType']
    dl_app_data = aim['Applications']['DL']['ApplicationData']
    ground_failure = dl_app_data['ground_failure']

    if asset_type == 'Buildings':
        # get the building parameters
        bt = gi['StructureType']  # building type

        # get the design level
        dl = gi.get('DesignLevel', None)

        if dl is None:
            # If there is no DesignLevel provided,
            # we assume that the YearBuilt is available
            year_built = gi['YearBuilt']

            design_l = ap_design_level_w1 if 'W1' in bt else ap_design_level

            for year in sorted(design_l.keys()):
                if year_built <= year:
                    dl = design_l[year]
                    break

            gi_ap['DesignLevel'] = dl

        # get the number of stories / height
        stories = gi.get('NumberOfStories', None)

        # We assume that the structure type does not include height information
        # and we append it here based on the number of story information
        rise = convert_story_rise(bt, stories)

        if rise is not None:
            lf = f'LF.{bt}.{rise}.{dl}'
            gi_ap['BuildingRise'] = rise
        else:
            lf = f'LF.{bt}.{dl}'

        # fmt: off
        comp = pd.DataFrame(
            {f'{lf}': ['ea',         1,          1,        1,   'N/A']},  # noqa: E241
            index = ['Units','Location','Direction','Theta_0','Family']  # noqa: E231, E251
        ).T
        # fmt: on

        # if needed, add components to simulate damage from ground failure
        if ground_failure:
            foundation_type = 'S'

            fg_gf_h = f'GF.H.{foundation_type}'
            fg_gf_v = f'GF.V.{foundation_type}'

            # fmt: off
            comp_gf = pd.DataFrame(
                {f'{fg_gf_h}':[  'ea',         1,          1,        1,   'N/A'],  # noqa: E201, E231, E241
                 f'{fg_gf_v}':[  'ea',         1,          3,        1,   'N/A']},  # noqa: E201, E231, E241
                index = [     'Units','Location','Direction','Theta_0','Family']  # noqa: E201, E231, E251
            ).T
            # fmt: on

            comp = pd.concat([comp, comp_gf], axis=0)

        # set the number of stories to 1
        # there is only one component in a building-level resolution
        stories = 1

        # get the occupancy class
        if gi['OccupancyClass'] in ap_occupancy:
            occupancy_type = ap_occupancy[gi['OccupancyClass']]
        else:
            occupancy_type = gi['OccupancyClass']

        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Buildings',
                'NumberOfStories': f'{stories}',
                'OccupancyType': f'{occupancy_type}',
                'PlanArea': '1',
            },
            'Damage': {'DamageProcess': 'Hazus Earthquake'},
            'Demands': {},
            'Losses': {
                'Repair': {
                    'ConsequenceDatabase': 'Hazus Earthquake - Buildings',
                    'MapApproach': 'Automatic',
                }
            },
            'Options': {
                'NonDirectionalMultipliers': {'ALL': 1.0},
            },
        }

    elif asset_type == 'TransportationNetwork':
        inf_type = gi['assetSubtype']

        if inf_type == 'HwyBridge':
            # If Skew is labeled as 99, it means there is a major variation in skews of substructure units. (Per NBI coding guide)
            # Assume a number of 45 as the "average" skew for the bridge.
            if gi['Skew'] == 99:
                gi['Skew'] = 45

            # get the bridge class
            bt = convertBridgeToHAZUSclass(gi)
            gi_ap['BridgeHazusClass'] = bt

            # fmt: off
            comp = pd.DataFrame(
                {f'HWB.GS.{bt[3:]}': [  'ea',         1,          1,        1,   'N/A']},  # noqa: E201, E241
                index = [            'Units', 'Location', 'Direction', 'Theta_0', 'Family']   # noqa: E201, E251
            ).T
            # fmt: on

            # scaling_specification
            k_skew = np.sqrt(np.sin((90 - gi['Skew']) * np.pi / 180.0))
            k_3d = getHAZUSBridgeK3DModifier(bt, gi)
            k_shape = getHAZUSBridgeSlightDamageModifier(bt, aim)
            scaling_specification = {
                f'HWB.GS.{bt[3:]}-1-1': {
                    'LS2': f'*{k_skew * k_3d}',
                    'LS3': f'*{k_skew * k_3d}',
                    'LS4': f'*{k_skew * k_3d}',
                }
            }
            if k_shape is not None:
                scaling_specification[f'HWB.GS.{bt[3:]}-1-1']['LS1'] = k_shape
            # if needed, add components to simulate damage from ground failure
            if ground_failure:
                # fmt: off
                comp_gf = pd.DataFrame(
                    {f'HWB.GF':          [  'ea',         1,          1,        1,   'N/A']},  # noqa: E201, E241, F541
                    index = [     'Units', 'Location', 'Direction', 'Theta_0', 'Family']   # noqa: E201, E251
                ).T
                # fmt: on

                comp = pd.concat([comp, comp_gf], axis=0)

                f1, f2 = getHAZUSBridgePGDModifier(bt, gi)

                scaling_specification.update(
                    {
                        'HWB.GF-1-1': {
                            'LS2': f'*{f1}',
                            'LS3': f'*{f1}',
                            'LS4': f'*{f2}',
                        }
                    }
                )

            dl_ap = {
                'Asset': {
                    'ComponentAssignmentFile': 'CMP_QNT.csv',
                    'ComponentDatabase': 'Hazus Earthquake - Transportation',
                    'BridgeHazusClass': bt,
                    'PlanArea': '1',
                },
                'Damage': {
                    'DamageProcess': 'Hazus Earthquake',
                    'ScalingSpecification': scaling_specification,
                },
                'Demands': {},
                'Losses': {
                    'Repair': {
                        'ConsequenceDatabase': 'Hazus Earthquake - Transportation',
                        'MapApproach': 'Automatic',
                    }
                },
                'Options': {
                    'NonDirectionalMultipliers': {'ALL': 1.0},
                },
            }

        elif inf_type == 'HwyTunnel':
            # get the tunnel class
            tt = convertTunnelToHAZUSclass(gi)
            gi_ap['TunnelHazusClass'] = tt

            # fmt: off
            comp = pd.DataFrame(
                {f'HTU.GS.{tt[3:]}': [  'ea',         1,          1,        1,   'N/A']},  # noqa: E201, E241
                index = [            'Units','Location','Direction','Theta_0','Family']   # noqa: E201, E231, E251
            ).T
            # fmt: on
            # if needed, add components to simulate damage from ground failure
            if ground_failure:
                # fmt: off
                comp_gf = pd.DataFrame(
                    {f'HTU.GF':          [  'ea',         1,          1,        1,   'N/A']},  # noqa: E201, E241, F541
                    index = [     'Units','Location','Direction','Theta_0','Family']   # noqa: E201, E231, E251
                ).T
                # fmt: on

                comp = pd.concat([comp, comp_gf], axis=0)

            dl_ap = {
                'Asset': {
                    'ComponentAssignmentFile': 'CMP_QNT.csv',
                    'ComponentDatabase': 'Hazus Earthquake - Transportation',
                    'TunnelHazusClass': tt,
                    'PlanArea': '1',
                },
                'Damage': {'DamageProcess': 'Hazus Earthquake'},
                'Demands': {},
                'Losses': {
                    'Repair': {
                        'ConsequenceDatabase': 'Hazus Earthquake - Transportation',
                        'MapApproach': 'Automatic',
                    }
                },
                'Options': {
                    'NonDirectionalMultipliers': {'ALL': 1.0},
                },
            }
        elif inf_type == 'Roadway':
            # get the road class
            rt = convertRoadToHAZUSclass(gi)
            gi_ap['RoadHazusClass'] = rt

            # fmt: off
            comp = pd.DataFrame(
                {},
                index = [           'Units','Location','Direction','Theta_0','Family']     # noqa: E201, E231, E251
            ).T
            # fmt: on

            if ground_failure:
                # fmt: off
                comp_gf = pd.DataFrame(
                    {f'HRD.GF.{rt[3:]}':[  'ea',         1,          1,        1,   'N/A']},  # noqa: E201, E231, E241
                    index = [     'Units','Location','Direction','Theta_0','Family']   # noqa: E201, E231, E251
                ).T
                # fmt: on

                comp = pd.concat([comp, comp_gf], axis=0)

            dl_ap = {
                'Asset': {
                    'ComponentAssignmentFile': 'CMP_QNT.csv',
                    'ComponentDatabase': 'Hazus Earthquake - Transportation',
                    'RoadHazusClass': rt,
                    'PlanArea': '1',
                },
                'Damage': {'DamageProcess': 'Hazus Earthquake'},
                'Demands': {},
                'Losses': {
                    'Repair': {
                        'ConsequenceDatabase': 'Hazus Earthquake - Transportation',
                        'MapApproach': 'Automatic',
                    }
                },
                'Options': {
                    'NonDirectionalMultipliers': {'ALL': 1.0},
                },
            }
        else:
            print('subtype not supported in HWY')

    elif asset_type == 'WaterDistributionNetwork':
        pipe_material_map = {
            'CI': 'B',
            'AC': 'B',
            'RCC': 'B',
            'DI': 'D',
            'PVC': 'D',
            'DS': 'D',
            'BS': 'B',
        }

        # GI = AIM.get("GeneralInformation", None)
        # if GI==None:

        # initialize the auto-populated GI
        wdn_element_type = gi_ap.get('type', 'MISSING')
        asset_name = gi_ap.get('AIM_id', None)

        if wdn_element_type == 'Pipe':
            pipe_construction_year = gi_ap.get('year', None)
            pipe_diameter = gi_ap.get('Diam', None)
            # diamaeter value is a fundamental part of hydraulic
            # performance assessment
            if pipe_diameter is None:
                msg = f'pipe diameter in asset type {asset_type}, \
                                 asset id "{asset_name}" has no diameter \
                                     value.'
                raise ValueError(msg)

            pipe_length = gi_ap.get('Len', None)
            # length value is a fundamental part of
            # hydraulic performance assessment
            if pipe_diameter is None:
                msg = f'pipe length in asset type {asset_type}, \
                                 asset id "{asset_name}" has no diameter \
                                     value.'
                raise ValueError(msg)

            pipe_material = gi_ap.get('material', None)

            # pipe material can be not available or named "missing" in
            # both case, pipe flexibility will be set to "missing"

            """
            The assumed logic (rullset) is that if the material is
            missing, if the pipe is smaller than or equal to 20
            inches, the material is Cast Iron (CI) otherwise the pipe
            material is steel.
                If the material is steel (ST), either based on user
            specified input or the assumption due to the lack of the
            user-input, the year that the pipe is constructed define
            the flexibility status per HAZUS instructions. If the pipe
            is built in 1935 or after, it is, the pipe is Ductile
            Steel (DS), and otherwise it is Brittle Steel (BS).
                If the pipe is missing construction year and is built
            by steel, we assume consevatively that the pipe is brittle
            (i.e., BS)
            """
            if pipe_material is None:
                if pipe_diameter > 20 * 0.0254:  # 20 inches in meter
                    print(
                        f'Asset {asset_name} is missing material. '
                        'Material is assumed to be Cast Iron'
                    )
                    pipe_material = 'CI'
                else:
                    print(
                        f'Asset {asset_name} is missing material. Material is '
                        f'assumed to be Steel (ST)'
                    )
                    pipe_material = 'ST'

            if pipe_material == 'ST':
                if (pipe_construction_year is not None) and (
                    pipe_construction_year >= 1935
                ):
                    msg = (
                        f'Asset {asset_name} has material of "ST" '
                        'is assumed to be Ductile Steel.'
                    )

                    print(msg)
                    pipe_material = 'DS'

                else:
                    msg = (
                        f'Asset {asset_name} has material of "ST" '
                        'is assumed to be Brittle Steel.'
                    )

                    print(msg)
                    pipe_material = 'BS'

            pipe_flexibility = pipe_material_map.get(pipe_material, 'missing')

            gi_ap['material flexibility'] = pipe_flexibility
            gi_ap['material'] = pipe_material

            # Pipes are broken into 20ft segments (rounding up) and
            # each segment is represented by an individual entry in
            # the performance model, `CMP`. The damage capacity of each
            # segment is assumed to be independent and driven by the
            # same EDP. We therefore replicate the EDP associated with
            # the pipe to the various locations assigned to the
            # segments.

            # Determine number of segments

            pipe_length_unit = gi_ap['units']['length']
            pipe_length_ft = pelicun.base.convert_units(
                pipe_length, unit=pipe_length_unit, to_unit='ft', category='length'
            )
            reference_length = 20.00  # 20 ft
            if pipe_length_ft % reference_length < 1e-2:
                # If the lengths are equal, then that's one segment, not two.
                num_segments = int(pipe_length_ft / reference_length)
            else:
                # In all other cases, round up.
                num_segments = int(pipe_length_ft / reference_length) + 1
            location_string = f'1--{num_segments}' if num_segments > 1 else '1'

            # Define performance model
            # fmt: off

            pipe_fl = f'PWP.{pipe_flexibility}'
            comp = pd.DataFrame(
                {pipe_fl + '.GS': ['ea', location_string, '0', 1, 'N/A'],
                 pipe_fl + '.GF': ['ea', location_string, '0', 1, 'N/A'],
                 'aggregate': ['ea', location_string, '0', 1, 'N/A']},
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
            ).T
            # fmt: on

            # Set up the demand cloning configuration for the pipe
            # segments, if required.
            demand_config = {}
            if num_segments > 1:
                # determine the EDP tags available for cloning
                response_data = pelicun.file_io.load_data('response.csv', None)
                num_header_entries = len(response_data.columns.names)
                # if 4, assume a hazard level tag is present and remove it
                if num_header_entries == 4:
                    response_data.columns = pd.MultiIndex.from_tuples(
                        [x[1::] for x in response_data.columns]
                    )
                demand_cloning_config = {}
                for edp in response_data.columns:
                    tag, location, direction = edp  # noqa: F841

                    demand_cloning_config['-'.join(edp)] = [
                        f'{tag}-{x}-{direction}'
                        for x in [f'{i + 1}' for i in range(num_segments)]
                    ]
                demand_config = {'DemandCloning': demand_cloning_config}

            # Create damage process
            dmg_process = {
                f'1_PWP.{pipe_flexibility}.GS-LOC': {'DS1': 'aggregate_DS1'},
                f'2_PWP.{pipe_flexibility}.GF-LOC': {'DS1': 'aggregate_DS1'},
                f'3_PWP.{pipe_flexibility}.GS-LOC': {'DS2': 'aggregate_DS2'},
                f'4_PWP.{pipe_flexibility}.GF-LOC': {'DS2': 'aggregate_DS2'},
            }
            dmg_process_filename = 'dmg_process.json'
            with open(dmg_process_filename, 'w', encoding='utf-8') as f:
                json.dump(dmg_process, f, indent=2)

            # Define the auto-populated config
            dl_ap = {
                'Asset': {
                    'ComponentAssignmentFile': 'CMP_QNT.csv',
                    'ComponentDatabase': 'Hazus Earthquake - Water',
                    'Material Flexibility': pipe_flexibility,
                    'PlanArea': '1',  # Sina: does not make sense for water.
                    # Kept it here since it was also
                    # kept here for Transportation
                },
                'Damage': {
                    'DamageProcess': 'User Defined',
                    'DamageProcessFilePath': 'dmg_process.json',
                },
                'Demands': demand_config,
            }

        elif wdn_element_type == 'Tank':
            tank_cmp_lines = {
                ('OG', 'C', 1): {'PST.G.C.A.GS': ['ea', 1, 1, 1, 'N/A']},
                ('OG', 'C', 0): {'PST.G.C.U.GS': ['ea', 1, 1, 1, 'N/A']},
                ('OG', 'S', 1): {'PST.G.S.A.GS': ['ea', 1, 1, 1, 'N/A']},
                ('OG', 'S', 0): {'PST.G.S.U.GS': ['ea', 1, 1, 1, 'N/A']},
                # Anchored status and Wood is not defined for On Ground tanks
                ('OG', 'W', 0): {'PST.G.W.GS': ['ea', 1, 1, 1, 'N/A']},
                # Anchored status and Steel is not defined for
                # Above Ground tanks
                ('AG', 'S', 0): {'PST.A.S.GS': ['ea', 1, 1, 1, 'N/A']},
                # Anchored status and Concrete is not defined for Buried tanks.
                ('B', 'C', 0): {'PST.B.C.GF': ['ea', 1, 1, 1, 'N/A']},
            }

            # The default values are assumed: material = Concrete (C),
            # location= On Ground (OG), and Anchored = 1
            tank_material = gi_ap.get('material', 'C')
            tank_location = gi_ap.get('location', 'OG')
            tank_anchored = gi_ap.get('anchored', 1)

            tank_material_allowable = {'C', 'S'}
            if tank_material not in tank_material_allowable:
                msg = f'Tank\'s material = "{tank_material}" is \
                     not allowable in tank {asset_name}. The \
                     material must be either C for concrete or S \
                     for steel.'
                raise ValueError(msg)

            tank_location_allowable = {'AG', 'OG', 'B'}
            if tank_location not in tank_location_allowable:
                msg = f'Tank\'s location = "{tank_location}" is \
                     not allowable in tank {asset_name}. The \
                     location must be either "AG" for Above \
                     ground, "OG" for On Ground or "BG" for \
                     Below Ground (buried) Tanks.'
                raise ValueError(msg)

            tank_anchored_allowable = {0, 1}
            if tank_anchored not in tank_anchored_allowable:
                msg = f'Tank\'s anchored status = "{tank_location}\
                     " is not allowable in tank {asset_name}. \
                     The anchored status must be either integer\
                     value 0 for unachored, or 1 for anchored'
                raise ValueError(msg)

            if tank_location == 'AG' and tank_material == 'C':
                msg = (
                    f'The tank {asset_name} is Above Ground (i.e., AG), '
                    'but the material type is Concrete ("C"). '
                    'Tank type "C" is not defined for AG tanks. '
                    'The tank is assumed to be Steel ("S").'
                )

                print(msg)
                tank_material = 'S'

            if tank_location == 'AG' and tank_material == 'W':
                msg = (
                    f'The tank {asset_name} is Above Ground (i.e., AG), but'
                    ' the material type is Wood ("W"). '
                    'Tank type "W" is not defined for AG tanks. '
                    'The tank is assumed to be Steel ("S").'
                )

                print(msg)
                tank_material = 'S'

            if tank_location == 'B' and tank_material == 'S':
                msg = (
                    f'The tank {asset_name} is buried (i.e., B), but the '
                    'material type is Steel ("S"). Tank type "S" is '
                    'not defined for "B" tanks. '
                    'The tank is assumed to be Concrete ("C").'
                )

                print(msg)
                tank_material = 'C'

            if tank_location == 'B' and tank_material == 'W':
                msg = (
                    f'The tank {asset_name} is buried (i.e., B), but the'
                    'material type is Wood ("W"). Tank type "W" is '
                    'not defined for B tanks. The tank is assumed '
                    'to be Concrete ("C")'
                )

                print(msg)
                tank_material = 'C'

            if tank_anchored == 1:
                # Since anchore status does nto matter, there is no need to
                # print a warning
                tank_anchored = 0

            cur_tank_cmp_line = tank_cmp_lines[
                tank_location, tank_material, tank_anchored
            ]

            comp = pd.DataFrame(
                cur_tank_cmp_line,
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
            ).T

            dl_ap = {
                'Asset': {
                    'ComponentAssignmentFile': 'CMP_QNT.csv',
                    'ComponentDatabase': 'Hazus Earthquake - Water',
                    'Material': tank_material,
                    'Location': tank_location,
                    'Anchored': tank_anchored,
                    'PlanArea': '1',  # Sina: does not make sense for water.
                    # Kept it here since it was also kept here for
                    # Transportation
                },
                'Damage': {'DamageProcess': 'Hazus Earthquake'},
                'Demands': {},
            }

        else:
            print(
                f'Water Distribution network element type {wdn_element_type} '
                f'is not supported in Hazus Earthquake IM DL method'
            )
            dl_ap = 'N/A'
            comp = None

    elif asset_type == 'PowerNetwork':
        # initialize the auto-populated GI
        power_asset_type = gi_ap.get('type', 'MISSING')
        asset_name = gi_ap.get('AIM_id', None)

        if power_asset_type == 'Substation':
            ep_s_size = ''
            ep_s_anchored = ''
            substation_voltage = gi_ap.get('Voltage', None)
            if substation_voltage is None:
                msg = (
                    'Substation feature "Voltage" is missing. '
                    f' substation "{asset_name}" assumed to be '
                    '"  Low Voltage".'
                )
                print(msg)
                substation_voltage = 'low'

            if isinstance(substation_voltage, str):
                if substation_voltage.lower() == 'low':
                    ep_s_size = 'L'
                elif substation_voltage.lower() == 'medium':
                    ep_s_size = 'M'
                elif substation_voltage.lower() == 'high':
                    ep_s_size = 'H'
                else:
                    msg = (
                        'substation Voltage value is = '
                        f'{substation_voltage}. '
                        'The value must be either "low" '
                        ', " medium", or " high".'
                    )
                    raise ValueError(msg)

            elif isinstance(substation_voltage, (float, int)):
                # Substation Voltage unit is kV. Any number smaller than
                # 34 kV is not supported by HAZUS methodlogy. Furthermore,
                # values significantly larger may refer to a voltage value in
                # different unit. The upper bound value is set ro 1200 kV.

                if substation_voltage < 34:
                    msg = (
                        f'The subtation Voltage for asset "{asset_name}" '
                        f'is too low({substation_voltage}). The current '
                        'methodology support voltage between 34 kV and 1200'
                        ' kV. Please make sure that the units are in kV.'
                    )
                    raise ValueError(msg)

                if substation_voltage > 1200:
                    msg = (
                        f'The subtation Voltage for asset "{asset_name}"'
                        f'is too high({substation_voltage}). The current '
                        'methodology support voltage between 34 kV and 1200'
                        ' kV. Please make sure that the units are in kV.'
                    )
                    raise ValueError(msg)

                if substation_voltage <= 150:
                    ep_s_size = 'L'
                elif substation_voltage <= 230:
                    ep_s_size = 'M'
                elif substation_voltage >= 500:
                    ep_s_size = 'H'
                else:
                    msg = (
                        'This should never have happed. Please '
                        'report this to the developer(SimCenter)'
                        f'. (Value = {substation_voltage}).'
                    )
                    raise RuntimeError(msg)
            else:
                msg = (
                    'substation Voltage value is = '
                    f'{substation_voltage}. It should be '
                    'string or a number. For more information, '
                    'refer to the documentation please.'
                )
                raise ValueError(msg)

            substation_anchored = gi_ap.get('Anchored', None)

            if substation_anchored is None:
                print(
                    'Substation feature "Anchored" is missing. '
                    f' substation "{asset_name}" assumed to be '
                    '"  Unanchored".'
                )

                substation_anchored = False

            if isinstance(substation_anchored, str):
                if substation_anchored.lower() in [
                    'a',
                    'anchored',
                    'yes',
                    'true',
                    'positive',
                    '1',
                ]:
                    ep_s_anchored = 'A'
                elif substation_anchored.lower() in [
                    'u',
                    'unanchored',
                    'no',
                    'false',
                    'negative',
                    '0',
                ]:
                    ep_s_anchored = 'U'
            elif isinstance(substation_anchored, (bool, int, float)):
                if abs(substation_anchored - True) < 0.001:
                    ep_s_anchored = 'A'
                elif abs(substation_anchored) < 0.001:
                    ep_s_anchored = 'U'
                else:
                    msg = (
                        'This should never have happed. Please '
                        'report this to the developer(SimCenter)'
                        f'. (Value = {substation_anchored}).'
                    )
                    raise RuntimeError(msg)

            if ep_s_anchored is None:
                msg = (
                    'Substation anchored value is = '
                    f'{substation_anchored}. It should be '
                    'string, boolean, or a number representing '
                    'True or False. For more information, '
                    'refer to the documentation please.'
                )
                raise ValueError(msg)

            # Define performance model
            # fmt: off
            substation_type = f'EP.S.{ep_s_size}.{ep_s_anchored}'
            comp = pd.DataFrame(
                {substation_type: ['ea', 1, 1, 1, 'N/A']},
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
            ).T

            # Define the auto-populated config
            dl_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake - Power",
                    "Substation Voltage": ep_s_size,
                    "Substation Anchored": ep_s_anchored,
                },
                "Damage": {"DamageProcess": "Hazus Earthquake"},
                "Demands": {},
                "Losses": {},
            }

        elif power_asset_type == 'Circuit':
            circuit_anchored = gi_ap.get('Anchored', None)

            ep_c_anchored = None
            if circuit_anchored is None:
                print(
                    'Circuit feature "Anchored" is missing. '
                    f' Circuit "{asset_name}" assumed to be '
                    '"  Unanchored".'
                )

                circuit_anchored = False

            if isinstance(circuit_anchored, str):
                if circuit_anchored.lower() in [
                    'a',
                    'anchored',
                    'yes',
                    'true',
                    'positive',
                    '1',
                ]:
                    ep_c_anchored = 'A'
                elif circuit_anchored.lower() in [
                    'u',
                    'unanchored',
                    'no',
                    'false',
                    'negative',
                    '0',
                ]:
                    ep_c_anchored = 'U'
            elif isinstance(circuit_anchored, (bool, int, float)):
                if abs(circuit_anchored - True) < 0.001:
                    ep_c_anchored = 'A'
                elif abs(circuit_anchored) < 0.001:
                    ep_c_anchored = 'U'
                else:
                    msg = (
                        'This should never have happed. Please '
                        'report this to the developer(SimCenter)'
                        f'. (Value = {circuit_anchored}).'
                    )
                    raise RuntimeError(msg)

            if ep_c_anchored is None:
                msg = (
                    'Circuit anchored value is = '
                    f'{circuit_anchored}. It should be '
                    'string, boolean, or a number representing '
                    'True or False. For more information, '
                    'refer to the documentation please.'
                )
                raise ValueError(msg)

            # Define performance model
            # fmt: off
            circuit_type = f'EP.C.{ep_c_anchored}'
            comp = pd.DataFrame(
                {circuit_type: ['ea', 1, 1, 1, 'N/A']},
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
            ).T

            # Define the auto-populated config
            dl_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake - Power",
                    "Circuit Anchored": ep_c_anchored,
                },
                "Damage": {"DamageProcess": "Hazus Earthquake"},
                "Demands": {},
                "Losses": {},
            }

        elif power_asset_type == 'Generation':
            ep_g_size = ''
            generation_output = gi_ap.get('Output', None)
            if generation_output is None:
                msg = (
                    'Generation feature "Output" is missing. '
                    f' Generation "{asset_name}" assumed to be '
                    '"Small".'
                )
                print(msg)
                # if the power feature is missing, the generation is assumed
                # to be small
                ep_g_size = 'small'

            if isinstance(generation_output, str):
                generation_output = generation_output.lower()
                generation_output = generation_output.strip()
                acceptable_power_unit = ('w', 'kw', 'mw', 'gw')

                units_exist = [
                    unit in generation_output for unit in acceptable_power_unit
                ]

                power_unit = None

                if True in units_exist:
                    power_unit = acceptable_power_unit[units_exist.index(True)]

                    if generation_output.endswith(power_unit):
                        generation_output = generation_output.strip(power_unit)
                        generation_output = generation_output.strip()
                else:
                    msg = (
                        "Generation feature doesn't have a unit for "
                        '"Output" value. The unit for Generation '
                        f'"{asset_name}"  is assumed to be "MW".'
                    )
                    print(msg)

                    power_unit = 'mw'

                try:
                    generation_output = float(generation_output)

                    if power_unit == 'w':
                        generation_output = generation_output / 10^6
                    elif power_unit == 'kw':
                        generation_output = generation_output / 10^3
                    elif power_unit == 'mw':
                        # just for the sake of completeness, we don't
                        # need to convert here, since MW is our base unit
                        pass 
                    elif power_unit == 'gw':
                        generation_output = generation_output * 1000

                    if generation_output < 200:
                        ep_g_size = 'small'
                    elif 200 < generation_output < 500:
                        ep_g_size = 'medium'
                    else:
                        ep_g_size = 'large'

                except ValueError as e:
                    # check if the exception is for value not being a float
                    not_float_str = 'could not convert string to float:'
                    if not str(e).startswith(not_float_str):
                        raise
                    # otherwise
                    msg = (
                        'Generation feature has an unrecognizable "Output"'
                        f' value. Generation "{asset_name}" = '
                        f'{generation_output}, instead of a numerical '
                        'value. So the size of the Generation is assumed '
                        'to be "Small".'
                    )
                    print(msg)

                    ep_g_size = 'small'

                if ep_g_size == 'small':
                    ep_g_size = 'S'
                elif ep_g_size in ('medium', 'large'):
                    # because medium and large size generation plants are
                    # categorized in the same category.
                    ep_g_size = 'ML'
                else:
                    msg = (
                        'This should never have happed. Please '
                        'report this to the developer(SimCenter)'
                        f'. (Value = {ep_g_size}).'
                    )
                    raise ValueError(msg)

            generation_anchored = gi_ap.get('Anchored', None)

            if generation_anchored is None:
                msg = (
                    'Generation feature "Anchored" is missing. '
                    f' Circuit "{asset_name}" assumed to be '
                    '"  Unanchored".'
                )
                print(msg)

                generation_anchored = False

            ep_g_anchored = None
            if isinstance(generation_anchored, str):
                if generation_anchored.lower() in [
                    'a',
                    'anchored',
                    'yes',
                    'true',
                    'positive',
                    '1',
                ]:
                    ep_g_anchored = 'A'
                elif generation_anchored.lower() in [
                    'u',
                    'unanchored',
                    'no',
                    'false',
                    'negative',
                    '0',
                ]:
                    ep_g_anchored = 'U'
            elif isinstance(generation_anchored, (bool, int, float)):
                if abs(generation_anchored - True) < 0.001:
                    ep_g_anchored = 'A'
                elif abs(generation_anchored) < 0.001:
                    ep_g_anchored = 'U'
                else:
                    msg = (
                        'This should never have happed. Please '
                        'report this to the developer(SimCenter)'
                        f'. (Value = {generation_anchored}).'
                    )
                    raise RuntimeError(msg)
            
            if ep_g_anchored is None:
                msg = (
                    'Circuit anchored value is = '
                    f'{circuit_anchored}. It should be '
                    'string, boolean, or a number representing '
                    'True or False. For more information, '
                    'refer to the documentation please.'
                )
                raise ValueError(msg)

            # Define performance model
            # fmt: off
            generation_type = f'EP.G.{ep_g_size}.{ep_g_anchored}'
            comp = pd.DataFrame(
                {generation_type: ['ea', 1, 1, 1, 'N/A']},
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family']
            ).T

            # Define the auto-populated config
            dl_ap = {
                "Asset": {
                    "ComponentAssignmentFile": "CMP_QNT.csv",
                    "ComponentDatabase": "Hazus Earthquake - Power",
                    "Generation Size": ep_g_size,
                    "Generation Anchored": ep_g_anchored,
                },
                "Damage": {"DamageProcess": "Hazus Earthquake"},
                "Demands": {},
                "Losses": {},
            }

    else:
        print(
            f'AssetType: {asset_type} is not supported '
            f'in Hazus Earthquake IM DL method'
        )

    return gi_ap, dl_ap, comp
