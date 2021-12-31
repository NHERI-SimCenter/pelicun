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

def MLRM_config(BIM):
    """
    Rules to identify a HAZUS MLRM configuration based on BIM data

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

    # Note the only roof option for commercial masonry in NJ appraisers manual
    # is OSWJ, so this suggests they do not even see alternate roof system
    # ref: Custom Inventory google spreadsheet H-37 10/01/20
    # This could be commented for other regions if detailed data are available
    BIM['roof_system'] = 'ows'

    # Roof cover
    # Roof cover does not apply to gable and hip roofs
    if year >= 1975:
        roof_cover = 'spm'
    else:
        # year < 1975
        roof_cover = 'bur'

    # Shutters
    # IRC 2000-2015:
    # R301.2.1.2 in NJ IRC 2015 says protection of openings required for
    # buildings located in WBD regions, mentions impact-rated protection for
    # glazing, impact-resistance for garage door glazed openings, and finally
    # states that wood structural panels with a thickness > 7/16" and a
    # span <8' can be used, as long as they are precut, attached to the framing
    # surrounding the opening, and the attachments are resistant to corrosion
    # and are able to resist component and cladding loads;
    # Earlier IRC editions provide similar rules.
    shutters = BIM['WBD']

    # Masonry Reinforcing (MR)
    # R606.6.4.1.2 Metal Reinforcement states that walls other than interior
    # non-load-bearing walls shall be anchored at vertical intervals of not
    # more than 8 inches with joint reinforcement of not less than 9 gage.
    # Therefore this ruleset assumes that all exterior or load-bearing masonry
    # walls will have reinforcement. Since our considerations deal with wind
    # speed, I made the assumption that only exterior walls are being taken
    # into consideration.
    MR = True

    # Wind Debris (widd in HAZSU)
    # HAZUS A: Res/Comm, B: Varies by direction, C: Residential, D: None
    WIDD = 'C' # residential (default)
    if BIM['occupancy_class'] in ['RES1', 'RES2', 'RES3A', 'RES3B', 'RES3C',
                                 'RES3D']:
        WIDD = 'C' # residential
    elif BIM['occupancy_class'] == 'AGR1':
        WIDD = 'D' # None
    else:
        WIDD = 'A' # Res/Comm

    if BIM['roof_system'] == 'ows':
        # RDA
        RDA = '6d' # HAZUS the only available option for OWSJ

        # Roof deck age (DQ)
        # Average lifespan of a steel joist roof is roughly 50 years according
        # to the source below. Therefore, if constructed 50 years before the
        # current year, the roof deck should be considered old.
        # https://www.metalroofing.systems/metal-roofing-pros-cons/
        if year >= (datetime.datetime.now().year - 50):
            DQ = 'god' # new or average
        else:
            DQ = 'por' # old

        # RWC
        RWC = 'tnail'  # Toe-nail (HAZUS the only available option for OWSJ)

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

    elif BIM['roof_system'] == 'trs':
        # This clause should not be activated for NJ
        # RDA
        if BIM['terrain'] >= 35: # suburban or light trees
            if BIM['V_ult'] > 130.0:
                RDA = '8s'  # 8d @ 6"/6" 'D'
            else:
                RDA = '8d'  # 8d @ 6"/12" 'B'
        else:  # light suburban or open
            if BIM['V_ult'] > 110.0:
                RDA = '8s'  # 8d @ 6"/6" 'D'
            else:
                RDA = '8d'  # 8d @ 6"/12" 'B'

        #  Metal RDA
        MRDA = 'nav'

        # Roof deck agea (DQ)
        DQ = 'nav' # not available for wood truss system

        # RWC
        if BIM['V_ult'] > 110:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail

    # shutters
    if year >= 2000:
        shutters = BIM['WBD']
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    if BIM['mean_roof_height'] < 15.0:
        # if it's MLRM1, configure outputs
        bldg_config = f"MLRM1_" \
                      f"{roof_cover}_" \
                      f"{RDA}_" \
                      f"{DQ}_" \
                      f"{BIM['roof_system']}_" \
                      f"{RWC}_" \
                      f"{int(shutters)}_" \
                      f"{WIDD}_" \
                      f"{int(MR)}_" \
                      f"{MRDA}_" \
                      f"{int(BIM['terrain'])}"
        return bldg_config
    else:
        unit_tag = 'nav'
        # MLRM2 needs more rulesets
        if BIM['roof_system'] == 'trs':
            JSPA = 0
        elif BIM['roof_system'] == 'ows':
            if BIM['no_units'] == 1:
                JSPA = 0
                unit_tag = 'sgl'
            else:
                JSPA = 4
                unit_tag = 'mlt'

        bldg_config = f"MLRM2_" \
                      f"{roof_cover}_" \
                      f"{RDA}_" \
                      f"{DQ}_" \
                      f"{BIM['roof_system']}_" \
                      f"{JSPA}_" \
                      f"{RWC}_" \
                      f"{int(shutters)}_" \
                      f"{WIDD}_" \
                      f"{unit_tag}_" \
                      f"{int(MR)}_" \
                      f"{MRDA}_" \
                      f"{int(BIM['terrain'])}"
        return bldg_config

