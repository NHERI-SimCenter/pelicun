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

ap_DesignLevel = {1940: 'LC', 1975: 'MC', 2100: 'HC'}
# ap_DesignLevel = {1940: 'PC', 1940: 'LC', 1975: 'MC', 2100: 'HC'}

ap_DesignLevel_W1 = {0: 'LC', 1975: 'MC', 2100: 'HC'}
# ap_DesignLevel_W1 = {0: 'PC', 0: 'LC', 1975: 'MC', 2100: 'HC'}

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
    'Parking': 'COM10',
}

convert_design_level = {
    'High-Code': 'HC',
    'Moderate-Code': 'MC',
    'Low-Code': 'LC',
    'Pre-Code': 'PC',
}


def convert_story_rise(structureType, stories):
    if structureType in ['W1', 'W2', 'S3', 'PC1', 'MH']:
        # These archetypes have no rise information in their IDs
        rise = None

    else:
        rise = None
        # First, check if we have valid story information
        try:
            stories = int(stories)

        except (ValueError, TypeError) as exc:
            msg = (
                'Missing "NumberOfStories" information, '
                'cannot infer `rise` attribute of archetype'
            )

            raise ValueError(msg) from exc

        if structureType == 'RM1':
            if stories <= 3:
                rise = 'L'

            else:
                rise = 'M'

        elif structureType == 'URM':
            if stories <= 2:
                rise = 'L'

            else:
                rise = 'M'

        elif structureType in [
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


def auto_populate(aim):
    """
    Automatically creates a performance model for story EDP-based Hazus EQ analysis.

    Parameters
    ----------
    aim: dict
        Asset Information Model - provides features of the asset that can be
        used to infer attributes of the performance model.

    Returns
    -------
    gi_ap: dict
        Extended General Information - extends the GI from the input AIM with
        additional inferred features. These features are typically used in
        intermediate steps during the auto-population and are not required
        for the performance assessment. They are returned to allow reviewing
        how these latent variables affect the final results.
    dl_ap: dict
        Damage and Loss parameters - these define the performance model and
        details of the calculation.
    comp: DataFrame
        Component assignment - Defines the components (in rows) and their
        location, direction, and quantity (in columns).
    """

    # extract the General Information
    gi = aim.get('GeneralInformation', None)

    if gi is None:
        # TODO: show an error message
        pass

    # initialize the auto-populated gi
    gi_ap = gi.copy()

    assetType = aim['assetType']
    ground_failure = aim['Applications']['DL']['ApplicationData']['ground_failure']

    if assetType == 'Buildings':
        # get the building parameters
        bt = gi['StructureType']  # building type

        # get the design level
        dl = gi.get('DesignLevel', None)

        if dl is None:
            # If there is no DesignLevel provided, we assume that the YearBuilt is
            # available
            year_built = gi['YearBuilt']

            if 'W1' in bt:
                DesignL = ap_DesignLevel_W1
            else:
                DesignL = ap_DesignLevel

            for year in sorted(DesignL.keys()):
                if year_built <= year:
                    dl = DesignL[year]
                    break

            gi_ap['DesignLevel'] = dl
        # get the number of stories / height
        stories = gi.get('NumberOfStories', None)

        # We assume that the structure type does not include height information
        # and we append it here based on the number of story information
        rise = convert_story_rise(bt, stories)

        # get the number of stories / height
        stories = gi.get('NumberOfStories', None)

        if rise is None:
            # To prevent STR.W2.None.LC
            fg_s = f'STR.{bt}.{dl}'
        else:
            fg_s = f'STR.{bt}.{rise}.{dl}'
        # fg_s = f"STR.{bt}.{dl}"
        fg_nsd = 'NSD'
        fg_nsa = f'NSA.{dl}'

        comp = pd.DataFrame(
            {
                f'{fg_s}': [
                    'ea',
                    1,
                    1,
                    1,
                    'N/A',
                ],
                f'{fg_nsa}': [
                    'ea',
                    1,
                    0,
                    1,
                    'N/A',
                ],
                f'{fg_nsd}': [
                    'ea',
                    1,
                    1,
                    1,
                    'N/A',
                ],
            },
            index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
        ).T

        # if needed, add components to simulate damage from ground failure
        if ground_failure:
            foundation_type = 'S'

            # fmt: off
            FG_GF_H = f'GF.H.{foundation_type}'
            FG_GF_V = f'GF.V.{foundation_type}'
            comp_gf = pd.DataFrame(
                {f'{FG_GF_H}': [  'ea',         1,          1,        1,   'N/A'],      # noqa: E201, E241
                 f'{FG_GF_V}': [  'ea',         1,          3,        1,   'N/A']},     # noqa: E201, E241
                index = [     'Units', 'Location', 'Direction', 'Theta_0', 'Family']       # noqa: E201, E251
            ).T
            # fmt: on

            comp = pd.concat([comp, comp_gf], axis=0)

        # get the occupancy class
        if gi['OccupancyClass'] in ap_Occupancy:
            occupancy = ap_Occupancy[gi['OccupancyClass']]
        else:
            occupancy = gi['OccupancyClass']

        plan_area = gi.get('PlanArea', 1.0)

        repair_config = {
            'ConsequenceDatabase': 'Hazus Earthquake - Buildings',
            'MapApproach': 'Automatic',
            'DecisionVariables': {
                'Cost': True,
                'Carbon': False,
                'Energy': False,
                'Time': False,
            },
        }

        dl_ap = {
            'Asset': {
                'ComponentAssignmentFile': 'CMP_QNT.csv',
                'ComponentDatabase': 'Hazus Earthquake - Buildings',
                'NumberOfStories': f'{stories}',
                'OccupancyType': f'{occupancy}',
                'PlanArea': str(plan_area),
            },
            'Damage': {'DamageProcess': 'Hazus Earthquake'},
            'Demands': {},
            'Losses': {'Repair': repair_config},
            'Options': {
                'NonDirectionalMultipliers': {'ALL': 1.0},
            },
        }

    else:
        print(
            f'AssetType: {assetType} is not supported '
            f'in Hazus Earthquake Capacity Spectrum Method-based DL method'
        )

    return gi_ap, dl_ap, comp
