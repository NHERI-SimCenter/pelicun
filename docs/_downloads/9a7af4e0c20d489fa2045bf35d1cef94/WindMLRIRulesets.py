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
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

import random
import numpy as np
import datetime

def MLRI_config(BIM):
    """
    Rules to identify a HAZUS MLRI configuration based on BIM data

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

    year = BIM['year_built'] # just for the sake of brevity

    # MR
    MR = True

    # Shutters
    shutters = False

    # Metal RDA
    # 1507.2.8.1 High Wind Attachment.
    # Underlayment applied in areas subject to high winds (Vasd greater
    # than 110 mph as determined in accordance with Section 1609.3.1) shall
    #  be applied with corrosion-resistant fasteners in accordance with
    # the manufacturer’s instructions. Fasteners are to be applied along
    # the overlap not more than 36 inches on center.
    if BIM['V_ult'] > 142:
        MRDA = 'std'  # standard
    else:
        MRDA = 'sup'  # superior

    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'nav'
        roof_quality = 'god' # default supported by HAZUS
    else:
        if year >= 1975:
            roof_cover = 'spm'
            if BIM['year_built'] >= (datetime.datetime.now().year - 35):
                roof_quality = 'god'
            else:
                roof_quality = 'por'
        else:
            # year < 1975
            roof_cover = 'bur'
            if BIM['year_built'] >= (datetime.datetime.now().year - 30):
                roof_quality = 'god'
            else:
                roof_quality = 'por'

    bldg_config = f"MLRI_" \
                  f"{roof_quality}_" \
                  f"{int(shutters)}_" \
                  f"{int(MR)}_" \
                  f"{MRDA}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config

