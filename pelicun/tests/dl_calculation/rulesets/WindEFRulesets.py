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

import datetime
import random


def HUEFFS_config(bim: dict) -> str:
    """
    Rules to identify a HAZUS HUEFFS/HUEFSS configuration based on BIM data.

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

    # Roof cover
    roof_cover = 'spm' if year >= 1975 else 'bur'

    # Wind debris
    widd = 'A'

    # Roof deck age
    if year >= (datetime.datetime.now(tz=datetime.timezone.utc).year - 50):
        dq = 'god'  # new or average
    else:
        dq = 'por'  # old

    # Metal-RDA
    if year > 2000:
        if bim['V_ult'] <= 142:
            mrda = 'std'  # standard
        else:
            mrda = 'sup'  # superior
    else:
        mrda = 'std'  # standard

    # Shutters
    shutters = int(bim['WBD'])

    # extend the BIM dictionary
    bim.update(
        {
            'RoofCover': roof_cover,
            'RoofDeckAttachmentM': mrda,
            'RoofDeckAge': dq,
            'WindDebrisClass': widd,
            'Shutters': shutters,
        }
    )

    bldg_tag = 'HUEF.FS'
    return (
        f"{bldg_tag}."
        f"{roof_cover}."
        f"{shutters}."
        f"{widd}."
        f"{dq}."
        f"{mrda}."
        f"{int(bim['TerrainRoughness'])}"
    )


def HUEFSS_config(bim: dict) -> str:
    """
    Rules to identify a HAZUS HUEFFS/HUEFSS configuration based on BIM data.

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

    # Roof cover
    if year >= 1975:
        roof_cover = 'spm'
    else:
        # year < 1975
        roof_cover = 'bur'

    # Wind debris
    widd = 'A'

    # Roof deck age
    if year >= (datetime.datetime.now(tz=datetime.timezone.utc).year - 50):
        dq = 'god'  # new or average
    else:
        dq = 'por'  # old

    # Metal-RDA
    if year > 2000:
        if bim['V_ult'] <= 142:
            mrda = 'std'  # standard
        else:
            mrda = 'sup'  # superior
    else:
        mrda = 'std'  # standard

    # Shutters
    shutters = bim['WindBorneDebris']

    # extend the BIM dictionary
    bim.update(
        {
            'RoofCover': roof_cover,
            'RoofDeckAttachmentM': mrda,
            'RoofDeckAge': dq,
            'WindDebrisClass': widd,
            'Shutters': shutters,
        }
    )

    bldg_tag = 'HUEF.S.S'
    return (
        f"{bldg_tag}."
        f"{roof_cover}."
        f"{int(shutters)}."
        f"{widd}."
        f"{dq}."
        f"{mrda}."
        f"{int(bim['TerrainRoughness'])}"
    )


def HUEFH_config(bim: dict) -> str:
    """
    Rules to identify a HAZUS HUEFH configuration based on BIM data.

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

    # Roof cover
    if year >= 1975:
        roof_cover = 'spm'
    else:
        # year < 1975
        roof_cover = 'bur'

    # Wind debris
    widd = 'A'

    # Shutters
    shutters = bim['WindBorneDebris']

    # Metal-RDA
    if year > 2000:
        if bim['V_ult'] <= 142:
            mrda = 'std'  # standard
        else:
            mrda = 'sup'  # superior
    else:
        mrda = 'std'  # standard

    if bim['NumberOfStories'] <= 2:
        bldg_tag = 'HUEF.H.S'
    elif bim['NumberOfStories'] <= 5:
        bldg_tag = 'HUEF.H.M'
    else:
        bldg_tag = 'HUEF.H.L'

    # extend the BIM dictionary
    bim.update(
        {
            'RoofCover': roof_cover,
            'RoofDeckAttachmentM': mrda,
            'WindDebrisClass': widd,
            'Shutters': shutters,
        }
    )

    return (
        f"{bldg_tag}."
        f"{roof_cover}."
        f"{widd}."
        f"{mrda}."
        f"{int(shutters)}."
        f"{int(bim['TerrainRoughness'])}"
    )


def HUEFS_config(bim: dict) -> str:
    """
    Rules to identify a HAZUS HUEFS configuration based on BIM data.

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

    # Roof cover
    if year >= 1975:
        roof_cover = 'spm'
    else:
        # year < 1975
        roof_cover = 'bur'

    # Wind debris
    widd = 'C'

    # Shutters
    if year > 2000:
        shutters = bim['WindBorneDebris']
    elif bim['WindBorneDebris']:
        shutters = random.random() < 0.46
    else:
        shutters = False

    # Metal-RDA
    if year > 2000:
        if bim['V_ult'] <= 142:
            mrda = 'std'  # standard
        else:
            mrda = 'sup'  # superior
    else:
        mrda = 'std'  # standard

    bldg_tag = 'HUEF.S.M' if bim['NumberOfStories'] <= 2 else 'HUEF.S.L'

    # extend the BIM dictionary
    bim.update(
        {
            'RoofCover': roof_cover,
            'RoofDeckAttachmentM': mrda,
            'WindDebrisClass': widd,
            'Shutters': shutters,
        }
    )

    return (
        f"{bldg_tag}."
        f"{roof_cover}."
        f"{int(shutters)}."
        f"{widd}."
        f"null."
        f"{mrda}."
        f"{int(bim['TerrainRoughness'])}"
    )
