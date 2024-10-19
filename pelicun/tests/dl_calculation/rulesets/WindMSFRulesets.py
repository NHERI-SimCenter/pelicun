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


def MSF_config(bim: dict) -> str:  # noqa: C901
    """
    Rules to identify a HAZUS MSF configuration based on BIM data.

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

    # Roof-Wall Connection (RWC)
    if bim['HazardProneRegion']:
        rwc = 'strap'  # Strap
    else:
        rwc = 'tnail'  # Toe-nail

    # Roof Frame Type
    rft = bim['RoofSystem']

    # Story Flag
    stories = min(bim['NumberOfStories'], 2)

    # Shutters
    # IRC 2000-2015:
    # R301.2.1.2 in NJ IRC 2015 says protection of openings required
    # for buildings located in WindBorneDebris regions, mentions
    # impact-rated protection for glazing, impact-resistance for
    # garage door glazed openings, and finally states that wood
    # structural panels with a thickness > 7/16" and a span <8' can be
    # used, as long as they are precut, attached to the framing
    # surrounding the opening, and the attachments are resistant to
    # corrosion and are able to resist component and cladding loads;
    # Earlier IRC editions provide similar rules.
    if year >= 2000:
        shutters = bim['WindBorneDebris']
    # BOCA 1996 and earlier:
    # Shutters were not required by code until the 2000 IBC. Before 2000, the
    # percentage of commercial buildings that have shutters is assumed to be
    # 46%. This value is based on a study on preparedness of small businesses
    # for hurricane disasters, which says that in Sarasota County, 46% of
    # business owners had taken action to wind-proof or flood-proof their
    # facilities. In addition to that, 46% of business owners reported boarding
    # up their businesses before Hurricane Katrina. In addition, compliance
    # rates based on the Homeowners Survey data hover between 43 and 50 percent.
    elif bim['WindBorneDebris']:
        shutters = random.random() < 0.45
    else:
        shutters = False

    if bim['RoofSystem'] == 'trs':
        # Roof Deck Attachment (RDA)
        # IRC codes:
        # NJ code requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
        # between ⅜”-1” -  see Table R602.3(1)
        # Fastener selection is contingent on thickness of sheathing in building
        # codes. Commentary for Table R602.3(1) indicates 8d nails with 6”/6”
        # spacing (enhanced roof spacing) for ultimate wind speeds greater than
        # a speed_lim. speed_lim depends on the year of construction
        rda = '6d'  # Default (aka A) in Reorganized Rulesets - WIND
        if year >= 2016:
            # IRC 2015
            speed_lim = 130.0  # mph
        else:
            # IRC 2000 - 2009
            speed_lim = 100.0  # mph
        if bim['V_ult'] > speed_lim:
            rda = '8s'  # 8d @ 6"/6" ('D' in the Reorganized Rulesets - WIND)
        else:
            rda = '8d'  # 8d @ 6"/12" ('B' in the Reorganized Rulesets - WIND)

        # Secondary Water Resistance (SWR)
        # Minimum drainage recommendations are in place in NJ (See below).
        # However, SWR indicates a code-plus practice.
        swr: int | float | str = random.random() < 0.6

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
        if bim['Garage'] == -1:
            # no garage data, using the default "none"
            garage = 'no'
        elif year > (datetime.datetime.now(tz=datetime.timezone.utc).year - 30):
            if bim['Garage'] < 1:
                garage = 'no'  # None
            elif shutters:
                garage = 'sup'  # SFBC 1994
            else:
                garage = 'std'  # Standard
        elif bim['Garage'] < 1:
            garage = 'no'  # None
        elif shutters:
            garage = 'sup'
        else:
            garage = 'wkd'  # Weak

        # Masonry Reinforcing (MR)
        # R606.6.4.1.2 Metal Reinforcement states that walls other than interior
        # non-load-bearing walls shall be anchored at vertical intervals of not
        # more than 8 inches with joint reinforcement of not less than 9-gage.
        # Therefore this ruleset assumes that all exterior or load-bearing masonry
        # walls will have reinforcement. Since our considerations deal with wind
        # speed, I made the assumption that only exterior walls are being taken
        # into consideration.
        mr = True

        stories = min(bim['NumberOfStories'], 2)

        # extend the BIM dictionary
        bim.update(
            {
                'SecondaryWaterResistance': swr,
                'RoofDeckAttachmentW': rda,
                'RoofToWallConnection': rwc,
                'Shutters': shutters,
                'AugmentGarage': garage,
                'MasonryReinforcing': mr,
            }
        )

        bldg_config = (
            f"M.SF."
            f"{int(stories)}."
            f"{bim['RoofShape']}."
            f"{rwc}."
            f"{rft}."
            f"{rda}."
            f"{int(shutters)}."
            f"{int(swr)}."
            f"{garage}."
            f"{int(mr)}."
            f"null."
            f"{int(bim['TerrainRoughness'])}"
        )

    else:
        # Roof system = OSJW
        # r
        # A 2015 study found that there were 750,000 metal roof installed in 2015,
        # out of 5 million new roofs in the US annually. If these numbers stay
        # relatively stable, that implies that roughly 15% of roofs are smlt.
        # ref. link: https://www.bdcnetwork.com/blog/metal-roofs-are-soaring-
        # popularity-residential-marmet
        roof_cover_options = ['smtl', 'cshl']
        roof_cover = roof_cover_options[int(random.random() < 0.85)]

        # Roof Deck Attachment (RDA)
        # NJ IBC 1507.2.8.1 (for cshl)
        # high wind attachments are required for DSWII > 142 mph
        # NJ IBC 1507.4.5 (for smtl)
        # high wind attachment are required for DSWII > 142 mph
        if bim['V_ult'] > 142.0:
            rda = 'sup'  # superior
        else:
            rda = 'std'  # standard

        # Secondary Water Resistance (SWR)
        # Minimum drainage recommendations are in place in NJ (See below).
        # However, SWR indicates a code-plus practice.

        # Default
        swr = 'null'  # type: ignore[no-redef]

        if bim['RoofShape'] == 'flt':
            swr = int(True)  # type: ignore[assignment]
        elif (
            (bim['RoofShape'] in {'hip', 'gab'})
            and (roof_cover == 'cshl')
            and (rda == 'sup')
        ):
            swr = int(random.random() < 0.6)

        stories = min(bim['NumberOfStories'], 2)

        # extend the BIM dictionary
        bim.update(
            {
                'SecondaryWaterResistance': swr,
                'RoofDeckAttachmentW': rda,
                'RoofToWallConnection': rwc,
                'Shutters': shutters,
                'AugmentGarage': garage,  # type: ignore[used-before-def]
                'MasonryReinforcing': mr,  # type: ignore[used-before-def]
            }
        )

        bldg_config = (
            f"M.SF."
            f"{int(stories)}."
            f"{bim['RoofShape']}."
            f"{rwc}."
            f"{rft}."
            f"{rda}."
            f"{int(shutters)}."
            f"{swr}."
            f"null."
            f"null."
            f"{roof_cover}."
            f"{int(bim['TerrainRoughness'])}"
        )

    return bldg_config
