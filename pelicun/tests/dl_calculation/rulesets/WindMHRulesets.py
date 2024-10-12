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

import random


def MH_config(bim: dict) -> str:
    """
    Rules to identify a HAZUS WSF configuration based on BIM data.

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
    if year <= 1976:
        # MHPHUD
        bldg_tag = 'MH.PHUD'
        shutters = random.random() < 0.45 if bim['WindBorneDebris'] else False
        # TieDowns
        tie_downs = random.random() < 0.45

    elif year <= 1994:
        # MH76HUD
        bldg_tag = 'MH.76HUD'
        shutters = random.random() < 0.45 if bim['WindBorneDebris'] else False
        # TieDowns
        tie_downs = random.random() < 0.45

    else:
        # MH94HUD I, II, III
        shutters = bim['V_ult'] >= 100.0
        # TieDowns
        tie_downs = bim['V_ult'] >= 70.0

        bldg_tag = 'MH.94HUD' + bim['WindZone']

    # extend the BIM dictionary
    bim.update(
        {
            'TieDowns': tie_downs,
            'Shutters': shutters,
        }
    )

    return (
        f"{bldg_tag}."
        f"{int(shutters)}."
        f"{int(tie_downs)}."
        f"{int(bim['TerrainRoughness'])}"
    )
