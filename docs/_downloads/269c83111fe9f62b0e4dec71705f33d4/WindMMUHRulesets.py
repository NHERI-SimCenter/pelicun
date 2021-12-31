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

def MMUH_config(BIM):
    """
    Rules to identify a HAZUS MMUH configuration based on BIM data

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

    # Secondary Water Resistance (SWR)
    # Minimum drainage recommendations are in place in NJ (See below).
    # However, SWR indicates a code-plus practice.
    SWR = False # Default
    if BIM['roof_shape'] == 'flt':
        SWR = True
    elif BIM['roof_shape'] in ['hip', 'gab']:
        SWR = random.random() < 0.6

    # Roof cover & Roof quality
    # Roof cover and quality do not apply to gable and hip roofs
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'nav'
        roof_quality = 'nav'
    # NJ Building Code Section 1507 (in particular 1507.10 and 1507.12) address
    # Built Up Roofs and Single Ply Membranes. However, the NJ Building Code
    # only addresses installation and material standards of different roof
    # covers, but not in what circumstance each must be used.
    # SPMs started being used in the 1960s, but different types continued to be
    # developed through the 1980s. Today, single ply membrane roofing is the
    # most popular flat roof option. BURs have been used for over 100 years,
    # and although they are still used today, they are used less than SPMs.
    # Since there is no available ruleset to be taken from the NJ Building
    # Code, the ruleset is based off this information.
    # We assume that all flat roofs built before 1975 are BURs and all roofs
    # built after 1975 are SPMs.
    # Nothing in NJ Building Code or in the Hazus manual specifies what
    # constitutes “good” and “poor” roof conditions, so ruleset is dependant
    # on the age of the roof and average lifespan of BUR and SPM roofs.
    # We assume that the average lifespan of a BUR roof is 30 years and the
    # average lifespan of a SPM is 35 years. Therefore, BURs installed before
    # 1990 are in poor condition, and SPMs installed before 1985 are in poor
    # condition.
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

    # Roof Deck Attachment (RDA)
    # IRC 2009-2015:
    # Requires 8d nails (with spacing 6”/12”) for sheathing thicknesses between
    # ⅜”-1”, see Table 2304.10, Line 31. Fastener selection is contingent on
    # thickness of sheathing in building codes.
    # Wind Speed Considerations taken from Table 2304.6.1, Maximum Nominal
    # Design Wind Speed, Vasd, Permitted For Wood Structural Panel Wall
    # Sheathing Used to Resist Wind Pressures. Typical wall stud spacing is 16
    # inches, according to table 2304.6.3(4). NJ code defines this with respect
    # to exposures B and C only. These are mapped to HAZUS categories based on
    # roughness length in the ruleset herein.
    # The base rule was then extended to the exposures closest to suburban and
    # light suburban, even though these are not considered by the code.
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

    # Roof-Wall Connection (RWC)
    if BIM['V_ult'] > 110.0:
        RWC = 'strap'  # Strap
    else:
        RWC = 'tnail'  # Toe-nail

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
    if year >= 2000:
        shutters = BIM['WBD']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    else:
        if BIM['WBD']:
            shutters = random.random() < 0.46
        else:
            shutters = False

    # Masonry Reinforcing (MR)
    # R606.6.4.1.2 Metal Reinforcement states that walls other than interior
    # non-load-bearing walls shall be anchored at vertical intervals of not
    # more than 8 inches with joint reinforcement of not less than 9 gage.
    # Therefore this ruleset assumes that all exterior or load-bearing masonry
    # walls will have reinforcement. Since our considerations deal with wind
    # speed, I made the assumption that only exterior walls are being taken
    # into consideration.
    MR = True

    stories = min(BIM['stories'], 3)
    bldg_config = f"MMUH" \
                  f"{int(stories)}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{int(SWR)}_" \
                  f"{roof_cover}_" \
                  f"{roof_quality}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{int(shutters)}_" \
                  f"{int(MR)}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config
