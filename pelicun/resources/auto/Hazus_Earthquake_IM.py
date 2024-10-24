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

import pandas as pd

import pelicun

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
def convertUnits(value, unit_in, unit_out):  # noqa: N802
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
        return None
    return value * scale_map[unit_in] / scale_map[unit_out]


def convertBridgeToHAZUSclass(aim):  # noqa: C901, N802
    # TODO: replace labels in AIM with standard CamelCase versions
    structure_type = aim['BridgeClass']
    # if (
    #     type(structureType) == str
    #     and len(structureType) > 3
    #     and structureType[:3] == "HWB"
    #     and 0 < int(structureType[3:])
    #     and 29 > int(structureType[3:])
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

    elif structureType in list(range(101, 107)):
        if not seismic:
            if state != 6:
                bridge_class = 'HWB5'
            else:
                bridge_class = 'HWB6'
        else:
            bridge_class = 'HWB7'

    elif structureType in [205, 206]:
        if not seismic:
            bridge_class = 'HWB8'
        else:
            bridge_class = 'HWB9'

    elif structureType in list(range(201, 207)):
        if not seismic:
            bridge_class = 'HWB10'
        else:
            bridge_class = 'HWB11'

    elif structureType in list(range(301, 307)):
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

    elif structureType in list(range(402, 411)):
        if not seismic:
            if len_max_span >= 20:
                bridge_class = 'HWB15'
            elif state != 6:
                bridge_class = 'HWB26'
            else:
                bridge_class = 'HWB27'
        else:
            bridge_class = 'HWB16'

    elif structureType in list(range(501, 507)):
        if not seismic:
            if state != 6:
                bridge_class = 'HWB17'
            else:
                bridge_class = 'HWB18'
        else:
            bridge_class = 'HWB19'

    elif structureType in [605, 606]:
        if not seismic:
            bridge_class = 'HWB20'
        else:
            bridge_class = 'HWB21'

    elif structureType in list(range(601, 608)):
        if not seismic:
            bridge_class = 'HWB22'
        else:
            bridge_class = 'HWB23'

    # TODO: review and add HWB24-27 rules
    # TODO: also double check rules for HWB10-11 and HWB22-23

    return bridge_class


def convertTunnelToHAZUSclass(aim) -> str:  # noqa: N802
    if ('Bored' in aim['ConstructType']) or ('Drilled' in aim['ConstructType']):
        return 'HTU1'
    elif ('Cut' in aim['ConstructType']) or ('Cover' in aim['ConstructType']):
        return 'HTU2'
    else:
        # Select HTU2 for unclassified tunnels because it is more conservative.
        return 'HTU2'


def convertRoadToHAZUSclass(aim) -> str:  # noqa: N802
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
        # First, check if we have valid story information
        try:
            stories = int(stories)

        except (ValueError, TypeError):
            msg = (
                'Missing "NumberOfStories" information, '
                'cannot infer `rise` attribute of archetype'
            )
            raise ValueError(msg)

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
    ground_failure = aim['Applications']['DL']['ApplicationData']['ground_failure']

    if asset_type == 'Buildings':
        # get the building parameters
        bt = gi['StructureType']  # building type

        # get the design level
        dl = gi.get('DesignLevel', None)

        if dl is None:
            # If there is no DesignLevel provided, we assume that the YearBuilt is
            # available
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

        comp = pd.DataFrame(
            {f'{lf}': ['ea', 1, 1, 1, 'N/A']},
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
        ).T

        # if needed, add components to simulate damage from ground failure
        if ground_failure:
            foundation_type = 'S'

            fg_gf_h = f'GF.H.{foundation_type}'
            fg_gf_v = f'GF.V.{foundation_type}'

            CMP_GF = pd.DataFrame(
                {
                    f'{fg_gf_h}': ['ea', 1, 1, 1, 'N/A'],
                    f'{fg_gf_v}': ['ea', 1, 3, 1, 'N/A'],
                },
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
            ).T

            comp = pd.concat([comp, CMP_GF], axis=0)

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
            # get the bridge class
            bt = convertBridgeToHAZUSclass(gi)
            gi_ap['BridgeHazusClass'] = bt

            comp = pd.DataFrame(
                {
                    f'HWB.GS.{bt[3:]}': ['ea', 1, 1, 1, 'N/A'],
                    'HWB.GF': ['ea', 1, 1, 1, 'N/A'],
                },
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
            ).T

            dl_ap = {
                'Asset': {
                    'ComponentAssignmentFile': 'CMP_QNT.csv',
                    'ComponentDatabase': 'Hazus Earthquake - Transportation',
                    'BridgeHazusClass': bt,
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

        elif inf_type == 'HwyTunnel':
            # get the tunnel class
            tt = convertTunnelToHAZUSclass(gi)
            gi_ap['TunnelHazusClass'] = tt

            comp = pd.DataFrame(
                {
                    f'HTU.GS.{tt[3:]}': ['ea', 1, 1, 1, 'N/A'],
                    'HTU.GF': ['ea', 1, 1, 1, 'N/A'],
                },
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
            ).T

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

            comp = pd.DataFrame(
                {f'HRD.GF.{rt[3:]}': ['ea', 1, 1, 1, 'N/A']},
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
            ).T

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
            'DS': 'B',
            'BS': 'D',
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
            # length value is a fundamental part of hydraulic performance assessment
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
                If the material is steel (ST), either based on user specified
            input or the assumption due to the lack of the user-input, the year
            that the pipe is constructed define the flexibility status per HAZUS
            instructions. If the pipe is built in 1935 or after, it is, the pipe
            is Ductile Steel (DS), and otherwise it is Brittle Steel (BS).
                If the pipe is missing construction year and is built by steel,
            we assume consevatively that the pipe is brittle (i.e., BS)
            """
            if pipe_material is None:
                if pipe_diameter > 20 * 0.0254:  # 20 inches in meter
                    pipe_material = 'CI'
                else:
                    pipe_material = 'ST'

            if pipe_material == 'ST':
                if (pipe_construction_year is not None) and (
                    pipe_construction_year >= 1935
                ):
                    pipe_material = 'DS'
                else:
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
            pipe_length_feet = pelicun.base.convert_units(
                pipe_length, unit=pipe_length_unit, to_unit='ft', category='length'
            )
            reference_length = 20.00  # 20 ft
            if pipe_length_feet % reference_length < 1e-2:
                # If the lengths are equal, then that's one segment, not two.
                num_segments = int(pipe_length_feet / reference_length)
            else:
                # In all other cases, round up.
                num_segments = int(pipe_length_feet / reference_length) + 1
            location_string = f'1--{num_segments}' if num_segments > 1 else '1'

            # Define performance model
            comp = pd.DataFrame(
                {
                    f'PWP.{pipe_flexibility}.GS': [
                        'ea',
                        location_string,
                        '0',
                        1,
                        'N/A',
                    ],
                    f'PWP.{pipe_flexibility}.GF': [
                        'ea',
                        location_string,
                        '0',
                        1,
                        'N/A',
                    ],
                    'aggregate': ['ea', location_string, '0', 1, 'N/A'],
                },
                index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
            ).T

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
                    tag, location, direction = edp

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
                    # Kept it here since itw as also
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
                # Anchored status and Steel is not defined for Above Ground tanks
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
                tank_material = 'S'

            if tank_location == 'AG' and tank_material == 'W':
                tank_material = 'S'

            if tank_location == 'B' and tank_material == 'S':
                tank_material = 'C'

            if tank_location == 'B' and tank_material == 'W':
                tank_material = 'C'

            if tank_anchored == 1:
                # Since anchore status does nto matter, there is no need to
                # print a warning
                tank_anchored = 0

            cur_tank_cmp_line = tank_cmp_lines[
                (tank_location, tank_material, tank_anchored)
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
                    # Kept it here since itw as also kept here for Transportation
                },
                'Damage': {'DamageProcess': 'Hazus Earthquake'},
                'Demands': {},
            }

        else:
            print(
                f'Water Distribution network element type {wdn_element_type} '
                f'is not supported in Hazus Earthquake IM DL method'
            )
            dl_ap = None
            comp = None

    else:
        print(
            f'AssetType: {asset_type} is not supported '
            f'in Hazus Earthquake IM DL method'
        )

    return gi_ap, dl_ap, comp
