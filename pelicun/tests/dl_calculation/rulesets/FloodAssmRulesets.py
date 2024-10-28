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

import random
import numpy as np
import datetime
import math

def Assm_config(BIM):
    """
    Rules to identify the flood vunerability category

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    config: str
        A string that identifies a specific configration within this buidling
        class.
    """
    year = BIM['YearBuilt'] # just for the sake of brevity

    # Flood Type
    if BIM['FloodZone'] in ['AO']:
        flood_type = 'raz' # Riverline/A-Zone
    elif BIM['FloodZone'] in ['AE', 'AH', 'A']:
        flood_type = 'caz' # Costal/A-Zone
    elif BIM['FloodZone'] in ['VE']:
        flood_type = 'cvz' # Costal/V-Zone
    else:
        flood_type = 'caz' # Default

    # PostFIRM
    PostFIRM = False # Default
    city_list = ['Absecon', 'Atlantic', 'Brigantine', 'Buena', 'Buena Vista',
                 'Corbin City', 'Egg Harbor City', 'Egg Harbor', 'Estell Manor',
                 'Folsom', 'Galloway', 'Hamilton', 'Hammonton', 'Linwood',
                 'Longport', 'Margate City', 'Mullica', 'Northfield',
                 'Pleasantville', 'Port Republic', 'Somers Point',
                 'Ventnor City', 'Weymouth']
    year_list = [1976, 1971, 1971, 1983, 1979, 1981, 1982, 1983, 1978, 1982,
                 1983, 1977, 1982, 1983, 1974, 1974, 1982, 1979, 1983, 1983,
                 1982, 1971, 1979]
    for i in range(0,22):
        PostFIRM = (((BIM['City'] == city_list[i]) and (year > year_list[i])) or \
                    PostFIRM)

    # fl_assm
    fl_assm = f"{'fl_surge_assm'}_" \
              f"{BIM['OccupancyClass']}_" \
              f"{int(PostFIRM)}_" \
              f"{flood_type}"

    # hu_assm
    hu_assm = f"{'hu_surge_assm'}_" \
              f"{BIM['OccupancyClass']}_" \
              f"{int(PostFIRM)}"

    return hu_assm, fl_assm