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
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

import numpy as np


def FL_config(bim: dict) -> str:  # noqa: C901
    """
    Rules to identify the flood vulnerability category.

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    config: str
        A string that identifies a specific configuration within this
        building class.

    """
    year = bim['YearBuilt']  # just for the sake of brevity

    # Flood Type
    if bim['FloodZone'] == 'AO':
        flood_type = 'raz'  # Riverline/A-Zone
    elif bim['FloodZone'] in {'A', 'AE'} or bim['FloodZone'].startswith('V'):
        flood_type = 'cvz'  # Costal-Zone
    else:
        flood_type = 'cvz'  # Default

    # flake8 - unused variable: `FFE`
    # # First Floor Elevation (FFE)
    # if flood_type in ['raz', 'caz']:
    #     FFE = BIM['FirstFloorElevation']
    # else:
    #     FFE = BIM['FirstFloorElevation'] - 1.0

    # PostFIRM
    post_firm = False  # Default
    city_list = [
        'Absecon',
        'Atlantic',
        'Brigantine',
        'Buena',
        'Buena Vista',
        'Corbin City',
        'Egg Harbor City',
        'Egg Harbor',
        'Estell Manor',
        'Folsom',
        'Galloway',
        'Hamilton',
        'Hammonton',
        'Linwood',
        'Longport',
        'Margate City',
        'Mullica',
        'Northfield',
        'Pleasantville',
        'Port Republic',
        'Somers Point',
        'Ventnor City',
        'Weymouth',
    ]
    year_list = [
        1976,
        1971,
        1971,
        1983,
        1979,
        1981,
        1982,
        1983,
        1978,
        1982,
        1983,
        1977,
        1982,
        1983,
        1974,
        1974,
        1982,
        1979,
        1983,
        1983,
        1982,
        1971,
        1979,
    ]
    for i in range(22):
        post_firm = (
            (bim['City'] == city_list[i]) and (year > year_list[i])
        ) or post_firm

    # Basement Type
    if bim['SplitLevel'] and (bim['FoundationType'] == 3504):
        bmt_type = 'spt'  # Split-Level Basement
    elif bim['FoundationType'] in {3501, 3502, 3503, 3505, 3506, 3507}:
        bmt_type = 'bn'  # No Basement
    elif (not bim['SplitLevel']) and (bim['FoundationType'] == 3504):
        bmt_type = 'bw'  # Basement
    else:
        bmt_type = 'bw'  # Default

    if bim['OccupancyClass'] not in {'RES1', 'RES2'}:
        if 'RES3' in bim['OccupancyClass']:
            fl_config = f"{'fl'}_" f"{'RES3'}"
        else:
            fl_config = f"{'fl'}_" f"{bim['OccupancyClass']}"
    elif bim['OccupancyClass'] == 'RES2':
        fl_config = f"{'fl'}_" f"{bim['OccupancyClass']}_" f"{flood_type}"
    elif bmt_type == 'spt':
        fl_config = (
            f"{'fl'}_"
            f"{bim['OccupancyClass']}_"
            f"{'sl'}_"
            f"{'bw'}_"
            f"{flood_type}"
        )
    else:
        st = 's' + str(np.min([bim['NumberOfStories'], 3]))
        fl_config = (
            f"{'fl'}_"
            f"{bim['OccupancyClass']}_"
            f"{st}_"
            f"{bmt_type}_"
            f"{flood_type}"
        )

    # extend the BIM dictionary
    bim.update(
        {
            'FloodType': flood_type,
            'BasementType': bmt_type,
            'PostFIRM': post_firm,
        }
    )

    return fl_config
