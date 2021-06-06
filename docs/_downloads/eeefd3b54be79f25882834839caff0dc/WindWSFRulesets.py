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


def WSF_config(BIM):
    """
    Rules to identify a HAZUS WSF configuration based on BIM data

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
    SWR = False # Default in Reorganzied Rulesets - WIND
    if year > 2000:
        # For buildings built after 2000, SWR is based on homeowner compliance
        # data from NC Coastal Homeowner Survey (2017) to capture potential
        # human behavior (% of sealed roofs in NC dataset).
        SWR = random.random() < 0.6
    elif year > 1983:
        # CABO 1995:
        # According to 903.2 in the 1995 CABO, for roofs with slopes between
        # 2:12 and 4:12, an underlayment consisting of two layers of No. 15
        # felt must be applied. In severe climates (less than or equal to 25
        # degrees Fahrenheit average in January), these two layers must be
        # cemented together.
        # According to 903.3 in the 1995 CABO, roofs with slopes greater than
        # or equal to 4:12 shall have an underlayment of not less than one ply
        # of No. 15 felt.
        #
        # Similar rules are prescribed in CABO 1992, 1989, 1986, 1983
        #
        # Since low-slope roofs require two layers of felt, this is taken to
        # be secondary water resistance. This ruleset is for asphalt shingles.
        # Almost all other roof types require underlayment of some sort, but
        # the ruleset is based on asphalt shingles because it is most
        # conservative.
        if BIM['roof_shape'] == 'flt': # note there is actually no 'flt'
            SWR = True
        elif BIM['roof_shape'] in ['gab','hip']:
            if BIM['roof_slope'] <= 0.17:
                SWR = True
            elif BIM['roof_slope'] < 0.33:
                SWR = (BIM['avg_jan_temp'] == 'below')

    # Roof Deck Attachment (RDA)
    # IRC codes:
    # NJ code requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
    # between ⅜”-1” -  see Table R602.3(1)
    # Fastener selection is contingent on thickness of sheathing in building
    # codes. Commentary for Table R602.3(1) indicates 8d nails with 6”/6”
    # spacing (enhanced roof spacing) for ultimate wind speeds greater than
    # a speed_lim. speed_lim depends on the year of construction
    RDA = '6d' # Default (aka A) in Reorganized Rulesets - WIND
    if year > 2000:
        if year >= 2016:
            # IRC 2015
            speed_lim = 130.0 # mph
        else:
            # IRC 2000 - 2009
            speed_lim = 100.0 # mph
        if BIM['V_ult'] > speed_lim:
            RDA = '8s'  # 8d @ 6"/6" ('D' in the Reorganized Rulesets - WIND)
        else:
            RDA = '8d'  # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)
    elif year > 1995:
        if ((BIM['sheathing_t'] >= 0.3125) and (BIM['sheathing_t'] <= 0.5)):
            RDA = '6d' # 6d @ 6"/12" ('A' in the Reorganized Rulesets - WIND)
        elif ((BIM['sheathing_t'] >= 0.59375) and (BIM['sheathing_t'] <= 1.125)):
            RDA = '8d' # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)
    elif year > 1986:
        if ((BIM['sheathing_t'] >= 0.3125) and (BIM['sheathing_t'] <= 0.5)):
            RDA = '6d' # 6d @ 6"/12" ('A' in the Reorganized Rulesets - WIND)
        elif ((BIM['sheathing_t'] >= 0.59375) and (BIM['sheathing_t'] <= 1.0)):
            RDA = '8d' # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)
    else:
        # year <= 1986
        if ((BIM['sheathing_t'] >= 0.3125) and (BIM['sheathing_t'] <= 0.5)):
            RDA = '6d' # 6d @ 6"/12" ('A' in the Reorganized Rulesets - WIND)
        elif ((BIM['sheathing_t'] >= 0.625) and (BIM['sheathing_t'] <= 1.0)):
            RDA = '8d' # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)

    # Roof-Wall Connection (RWC)
    # IRC 2015
    # "Assume all homes not having wind speed consideration are Toe Nail
    # (regardless of year)
    # For homes with wind speed consideration, 2015 IRC Section R802.11: no
    # specific connection type, must resist uplift forces using various
    # guidance documents, e.g., straps would be required (based on WFCM 2015);
    # will assume that if classified as HPR, then enhanced connection would be
    # used.
    if year > 2015:
        if BIM['HPR']:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail
    # IRC 2000-2009
    # In Section R802.11.1 Uplift Resistance of the NJ 2009 IRC, roof
    # assemblies which are subject to wind uplift pressures of 20 pounds per
    # square foot or greater are required to have attachments that are capable
    # of providing resistance, in this case assumed to be straps.
    # Otherwise, the connection is assumed to be toe nail.
    # CABO 1992-1995:
    # 802.11 Roof Tie-Down: Roof assemblies subject to wind uplift pressures of
    # 20 lbs per sq ft or greater shall have rafter or truess ties. The
    # resulting uplift forces from the rafter or turss ties shall be
    # transmitted to the foundation.
    # Roof uplift pressure varies by wind speed, exposure category, building
    # aspect ratio and roof height. For a reference building (9 ft tall in
    # exposure B -- WSF1) analysis suggests that wind speeds in excess of
    # 110 mph begin to generate pressures of 20 psf in high pressure zones of
    # the roof. Thus 110 mph is used as the critical velocity.
    elif year > 1992:
        if BIM['V_ult'] > 110:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail
    # CABO 1989 and earlier
    # There is no mention of straps or enhanced tie-downs in the CABO codes
    # older than 1992, and there is no description of these adoptions in IBHS
    # reports or the New Jersey Construction Code Communicator .
    # Although there is no explicit information, it seems that hurricane straps
    # really only came into effect in Florida after Hurricane Andrew (1992).
    # Because Florida is the leader in adopting hurricane protection measures
    # into codes and because there is no mention of shutters or straps in the
    # CABO codes, it is assumed that all roof-wall connections for residential
    # buildings are toe nails before 1992.
    else:
        # year <= 1992
        RWC = 'tnail' # Toe-nail

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
    if year > 2000:
        shutters = BIM['WBD']
    # CABO:
    # Based on Human Subjects Data, roughly 45% of houses built in the 1980s
    # and 1990s had entries that implied they had shutters on at some or all of
    # their windows. Therefore, 45% of houses in this time should be randomly
    # assigned to have shutters.
    # Data ranges checked:
    # 1992 to 1995, 33/74 entries (44.59%) with shutters
    # 1986 to 1992, 36/79 entries (45.57%) with shutters
    # 1983 to 1986, 19/44 entries (43.18%) with shutters
    else:
        # year <= 2000
        if BIM['WBD']:
            shutters = random.random() < 0.45
        else:
            shutters = False

    # Garage
    # As per IRC 2015:
    # Garage door glazed opening protection for windborne debris shall meet the
    # requirements of an approved impact-resisting standard or ANSI/DASMA 115.
    # Exception: Wood structural panels with a thickness of not less than 7/16
    # inch and a span of not more than 8 feet shall be permitted for opening
    # protection. Panels shall be predrilled as required for the anchorage
    # method and shall be secured with the attachment hardware provided.
    # Permitted for buildings where the ultimate design wind speed is 180 mph
    # or less.
    #
    # Average lifespan of a garage is 30 years, so garages that are not in WBD
    # (and therefore do not have any strength requirements) that are older than
    # 30 years are considered to be weak, whereas those from the last 30 years
    # are considered to be standard.
    if BIM['garage_tag'] == -1:
        # no garage data, using the default "standard"
        garage = 'std'
        shutters = 0 # HAZUS ties standard garage to w/o shutters
    else:
        if year > 2000:
            if shutters:
                if BIM['garage_tag'] < 1:
                    garage = 'no'
                else:
                    garage = 'sup' # SFBC 1994
                    shutters = 1 # HAZUS ties SFBC 1994 to with shutters
            else:
                if BIM['garage_tag'] < 1:
                    garage = 'no' # None
                else:
                    garage = 'std' # Standard
                    shutters = 0 # HAZUS ties standard garage to w/o shutters
        elif year > (datetime.datetime.now().year - 30):
            if BIM['garage_tag'] < 1:
                garage = 'no' # None
            else:
                garage = 'std' # Standard
                shutters = 0 # HAZUS ties standard garage to w/o shutters
        else:
            # year <= current year - 30
            if BIM['garage_tag'] < 1:
                garage = 'no' # None
            else:
                garage = 'wkd' # Weak
                shutters = 0 # HAZUS ties weak garage to w/o shutters

    # building configuration tag
    bldg_config = f"WSF" \
                  f"{int(min(BIM['stories'],2))}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{int(SWR)}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{garage}_" \
                  f"{int(shutters)}_" \
                  f"{int(BIM['terrain'])}"
    return bldg_config

