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
#
# Based on rulesets developed by:
# Karen Angeles
# Meredith Lockhead
# Tracy Kijewski-Correa

import random
import numpy as np

def parse_BIM(BIM_in):
    """
    Parses the information provided in the BIM model.

    The parameters below list the expected inputs

    Parameters
    ----------
    stories: str
        Number of stories
    yearBuilt: str
        Year of construction.
    roofType: {'hip', 'hipped', 'gabled', 'gable', 'flat'}
        One of the listed roof shapes that best describes the building.
    occupancy: str
        Occupancy type.
    buildingDescription: str
        MODIV code that provides additional details about the building
    structType: {'Stucco', 'Frame', 'Stone', 'Brick'}
        One of the listed structure types that best describes the building.
    V_design: string
        Ultimate Design Wind Speed was introduced in the 2012 IBC. Officially
        called “Ultimate Design Wind Speed (Vult); equivalent to the design
        wind speeds taken from hazard maps in ASCE 7 or ATC's API. Unit is
        assumed to be mph.
    area: float
        Plan area in ft2.
    z0: string
        Roughness length that characterizes the surroundings.

    Returns
    -------
    BIM: dictionary
        Parsed building characteristics.
    """

    # maps roof type to the internal representation
    ap_RoofType = {
        'hip'   : 'hip',
        'hipped': 'hip',
        'gabled': 'gab',
        'gable' : 'gab',
        'flat'  : 'flt'
    }

    # first, pull in the provided data
    BIM = dict(
        stories=int(BIM_in['stories']),
        year_built=int(BIM_in['yearBuilt']),
        roof_shape=ap_RoofType[BIM_in['roofType']],
        occupancy=BIM_in['occupancy'],
        bldg_desc=str(BIM_in.get('buildingDescription','')),
        struct_type=BIM_in['structType'],
        V_ult=float(BIM_in['V_design']),
        area=BIM_in['area'],
        z0=float(BIM_in['z0']),
    )

    # add inferred, generic meta-variables

    # Hurricane-Prone Region
    # Areas vulnerable to hurricane, defined as the U.S. Atlantic Ocean and
    # Gulf of Mexico coasts where the ultimate design wind speed, V_ult is
    # greater than a pre-defined limit.
    if BIM['year_built'] >= 2016:
        # The limit is 115 mph in IRC 2015
        HPR = BIM['V_ult'] > 115.0
    else:
        # The limit is 90 mph in IRC 2009 and earlier versions
        HPR = BIM['V_ult'] > 90.0

    # Wind Borne Debris
    # Areas within hurricane-prone regions are affected by debris if one of
    # the following two conditions holds:
    # (1) Within 1 mile (1.61 km) of the coastal mean high water line where
    # the ultimate design wind speed is greater than flood_lim.
    # (2) In areas where the ultimate design wind speed is greater than
    # general_lim
    # The flood_lim and general_lim limits depend on the year of construction
    if BIM['year_built'] >= 2016:
        # In IRC 2015:
        flood_lim = 130.0 # mph
        general_lim = 140.0 # mph
    else:
        # In IRC 2009 and earlier versions
        flood_lim = 110.0 # mph
        general_lim = 120.0 # mph

    WBD = (HPR and
           ((BIM['V_ult'] > general_lim) or ((BIM['V_ult'] > flood_lim) and
                                             (BIM['flood_risk']))
           ))

    BIM.update(dict(
        # Nominal Design Wind Speed
        # Former term was “Basic Wind Speed”; it is now the “Nominal Design
        # Wind Speed (V_asd). Unit: mph."
        V_asd = np.sqrt(0.6 * BIM['V_ult']),

        # Flood Risk
        # Properties in the High Water Zone (within 1 mile of the coast) are at
        # risk of flooding and other wind-borne debris action.
        flood_risk=True,  # TODO: need high water zone for this and move it to inputs!

        HPR=HPR,
        WBD=WBD,
    ))

    return BIM

def building_class(BIM):
    """
    Short description

    Long description

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.

    Returns
    -------
    bldg_class: str
        One of the standard building class labels from HAZUS
    """

    if ((BIM['roof_shape'] != 'flt') and
        (BIM['stories'] <= 2) and
        (BIM['area'] < 2000.0)):

        return 'WSF'

    else:

        return 'WMUH'

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
    # additional characteristics needed
    T_mean_january = 20 # Fahrenheit
    roof_slope = 3./12.
    sheathing_t = 1.0 # in
    uplift_pressure = 25.0 #lb/ft2

    year = BIM['year_built'] # just for the sake of brevity

    # Secondary water resistance
    # Minimum drainage recommendations are in place in NJ (See below).
    # However, SWR indicates a code-plus practice.
    if year >= 2000:
        # For buildings built after 2000, SWR is based on homeowner compliance
        # data from NC Coastal Homeowner Survey (2017) to capture potential
        # human behavior (% of sealed roofs in NC dataset).
        SWR = random.random() < 0.6
    else:
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
        SWR = ((T_mean_january < 25.0) and (roof_slope < 4./12.))

    # Roof deck attachment
    # IRC codes:
    # NJ code requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
    # between ⅜”-1” -  see Table R602.3(1)
    # Fastener selection is contingent on thickness of sheathing in building
    # codes. Commentary for Table R602.3(1) indicates 8d nails with 6”/6”
    # spacing (enhanced roof spacing) for ultimate wind speeds greater than
    # a speed_lim. speed_lim depends on the year of construction
    if year >= 2000:
        if year >= 2016:
            # IRC 2015
            speed_lim = 130.0 # mph
        else:
            # IRC 2000 - 2009
            speed_lim = 100.0 # mph

        if BIM['V_ult'] > speed_lim:
            RDA = '8s'  # 8d @ 6"/6"
        else:
            RDA = '8d'  # 8d @ 6"/12"
    # CABO:
    # Based on Table No. 602.3a of the 1995 CABO code, the nailing
    # specifications are a function of the sheathing thickness.
    # Similar rules are prescirbed in earlier CABO codes.
    else:
        if sheathing_t > 0.5:
            RDA = '8d'  # 8d @ 6"/12"
        else:
            RDA = '8s'  # 8d @ 6"/6"

    # Roof-wall connection
    # IRC 2015
    # "Assume all homes not having wind speed consideration are Toe Nail
    # (regardless of year)
    # For homes with wind speed consideration, 2015 IRC Section R802.11: no
    # specific connection type, must resist uplift forces using various
    # guidance documents, e.g., straps would be required (based on WFCM 2015);
    # will assume that if classified as HPR, then enhanced connection would be
    # used.
    if year >= 2016:
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
    elif year >= 1992:
        if uplift_pressure >= 20.0:
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
    if year >= 2000:
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
        shutters = random.random() < 0.45

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
    if shutters:
        if ((BIM['bldg_desc'] is not None) and ('AG' in BIM['bldg_desc']) and
            (year >= 2000)):
            garage = 'sup' # SFBC 1994
        else:
            garage = 'no' # None TODO: check with Tracy
    else:
        if ((BIM['bldg_desc'] is not None) and ('AG' in BIM['bldg_desc'])):

            if year >= 1990:
                garage = 'std' # Standard
            else:
                garage = 'wkd' # Weak
        else:
            garage = 'no' # None

    # Terrain
    terrain = int(100 * BIM['z0'])

    bldg_config = f"WSF" \
                  f"{BIM['stories']}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{int(SWR)}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{garage}_" \
                  f"{int(shutters)}_" \
                  f"{terrain}"

    return bldg_config

def WMUH_config(BIM):
    """
    Rules to identify a HAZUS WMUH configuration based on BIM data

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
    # additional characteristics needed
    T_mean_january = 20  # Fahrenheit
    roof_slope = 3. / 12.
    mean_roof_height = 15.0 # ft
    sheathing_t = 1.0  # in


    year = BIM['year_built']  # just for the sake of brevity

    # Secondary water resistance
    if BIM['roof_shape'] == 'flt':
        # Buildings with flat roofs fit under the condition that water will be
        # entrapped if primary drains allow buildup
        SWR = 'null'  # because SWR is not a question for flat roofs
    else:
        # IRC 2000-2015:
        # Beyond the drainage requirements that follow, sealing to achieve SWR
        # is voluntary action: Will require assigning a human decision on
        # code-plus SWR. Use NC Coastal Homeowner Survey (2017) data as
        # placeholder. Code provisions for drainage are as follows: 1503.4.1
        # Secondary (emergency overflow) drains or scuppers. Where roof drains
        # are required, secondary (emergency overflow) roof drains or scuppers
        # shall be provided where the roof perimeter construction extends above
        # the roof in such a manner that water will be entrapped if the primary
        # drains allow buildup for any reason. The installation and sizing of
        # secondary emergency overflow drains, leaders and conductors shall
        # comply with plumbing subcode, N.J.A.C. 5:23-3:15. We assume that all
        # buildings of this size will likely have required roof drains
        if year >= 2000:
            SWR = random.random() < 0.6
        # BOCA 1993 - 1996:
        # The 1996 BOCA code requires SWR for steep-slope roofs with winters at
        # or below 25 Fahrenheit, according to Section 1507.4. This does not
        # apply for low-slope roof coverings, as shown by Section 1507.5.
        # Asphalt shingles can be installed on roof slopes 2:12 and greater;
        # low-slope roofing can be considered any roof pitch less than 2:12
        # BUR is considered low-slope roofing
        elif year >= 1993: #TODO: double-check
            SWR = (T_mean_january < 25.0) and (roof_slope > 2. / 12.)
        # BOCA 1987
        # The BOCA 1987 Code specifies these requirements in 2303.1. These
        # requirements are specifically for asphalt shingle roofs. This ruleset
        # assumes that two layers of Type 15 felt, a strip of mineral surfaced
        # roll roofing, and double coverage shingles all count as secondary
        # water resistance.
        elif year >= 1987: #TODO: double-check
            SWR = not ((T_mean_january > 25.0) and (roof_slope > 4. / 12.))
        # BOCA 1984:
        # There are no specifications or requirements outlining use of SWR in
        # the 1984 BOCA code. Thus, this ruleset refers to the homeowner data.
        # Based on Human Subjects Data ranging from 1984 to 1987, 30% had
        # entries that implied they had SWR, either that they bought,
        # retrofitted, or remodeled. Therefore, 30% of houses in this time
        # should be randomly assigned to have secondary water resistance.
        # Data taken from Question 32 of Human Subjects Data.
        elif year >= 1984:
            SWR = random.random() < 0.3
        # BOCA 1981:
        # Based on Human Subjects Data ranging from 1981 to 1984.
        elif year >= 1981:
            SWR = random.random() < 0.28
        # BOCA 1978
        # Based on Human Subjects Data ranging from 1978 to 1981
        elif year >= 1978:
            SWR = random.random() < 0.22
        # BOCA 1975
        # Based on Human Subjects Data ranging from 1975 to 1975
        else:
            SWR = random.random() < 0.3
        SWR = int(SWR)

    # Roof cover & Roof quality
    # Roof cover and quality do not apply to gable and hip roofs
    if BIM['roof_shape'] in ['gab', 'hip']:
        roof_cover = 'null'
        roof_quality = 'null'
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

            if BIM['year_built'] >= 1985:
                roof_quality = 'god'
            else:
                roof_quality = 'por'
        else:
            roof_cover = 'bur'

            if BIM['year_built'] >= 1990:
                roof_quality = 'god'
            else:
                roof_quality = 'por'


    # Roof deck attachment
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
    if year >= 2009:
        if BIM['z0'] >= 0.35: # suburban or light trees
            if BIM['V_asd'] > 130.0:
                RDA = '8s'  # 8d @ 6"/6"
            else:
                RDA = '8d'  # 8d @ 6"/12"
        else:  # light suburban or open
            if BIM['V_asd'] > 110.0:
                RDA = '8s'  # 8d @ 6"/6"
            else:
                RDA = '8d'  # 8d @ 6"/12"
    # IRC 2000-2006:
    # Table 2304.9.1, Line 31 of the 2006
    # NJ IBC requires 8d nails (with spacing 6”/12”) for sheathing thicknesses
    # of ⅞”-1”. Fastener selection is contingent on thickness of sheathing in
    # building codes. Table 2308.10.1 outlines the required rating of approved
    # uplift connectors, but does not specify requirements that require a
    # change of connector at a certain wind speed.
    # Thus, all RDAs are assumed to be 8d @ 6”/12”.
    elif year >= 2000:
        RDA = '8d'  # 8d @ 6"/12"
    # BOCA 1996:
    # The BOCA 1996 Building Code Requires 8d nails (with spacing 6”/12”) for
    # roof sheathing thickness up to 1". See Table 2305.2, Section 4.
    # Attachment requirements are given based on sheathing thickness, basic
    # wind speed, and the mean roof height of the building.
    elif year >= 1996:
        if (BIM['V_asd'] > 90.0) and (mean_roof_height >= 25.0):
            RDA = '8s'  # 8d @ 6"/6"
        else:
            RDA = '8d'  # 8d @ 6"/12"
    # BOCA 1993:
    # The BOCA 1993 Building Code Requires 8d nails (with spacing 6”/12”) for
    # sheathing thicknesses of 19/32  inches or greater, and 6d nails (with
    # spacing 6”/12”) for sheathing thicknesses of ½ inches or less.
    # See Table 2305.2, Section 4.
    else:
        if sheathing_t > 0.5:
            RDA = '8d'  # 8d @ 6"/12"
        else:
            RDA = '8s'  # 8d @ 6"/6"

    # Roof-wall connection
    # IRC 2000-2015:
    # 1507.2.8.1 High Wind Attachment. Underlayment applied in areas subject
    # to high winds (Vasd greater than 110 mph as determined in accordance
    # with Section 1609.3.1) shall be applied with corrosion-resistant
    # fasteners in accordance with the manufacturer’s instructions. Fasteners
    # are to be applied along the overlap not more than 36 inches on center.
    # Underlayment installed where Vasd, in accordance with section 1609.3.1
    # equals or exceeds 120 mph shall be attached in a grid pattern of 12
    # inches between side laps with a 6-inch spacing at the side laps.
    if year >= 2000:
        if BIM['V_asd'] > 110.0:
            RWC = 'strap'  # Strap
        else:
            RWC = 'tnail'  # Toe-nail
    # BOCA 1996 and earlier:
    # There is no mention of straps or enhanced tie-downs of any kind in the
    # BOCA codes, and there is no description of these adoptions in IBHS
    # reports or the New Jersey Construction Code Communicator .
    # Although there is no explicit information, it seems that hurricane straps
    # really only came into effect in Florida after Hurricane Andrew (1992),
    # and likely it took several years for these changes to happen. Because
    # Florida is the leader in adopting hurricane protection measures into
    # codes and because there is no mention of shutters or straps in the BOCA
    # codes, it is assumed that New Jersey did not adopt these standards until
    # the 2000 IBC.
    else:
        RWC = 'tnail'  # Toe-nail

    # Shutters
    # IRC 2000-2015:
    # 1609.1.2 Protection of Openings. In wind-borne debris regions, glazing in
    # buildings shall be impact resistant or protected with an impact-resistant
    # covering meeting the requirements of an approved impact-resistant
    # covering meeting the requirements of an approved impact-resistant
    # standard.
    # Exceptions: Wood structural panels with a minimum thickness of 7/16 of an
    # inch and a maximum panel span of 8 feet shall be permitted for opening
    # protection in buildings with a mean roof height of 33 feet or less that
    # are classified as a Group R-3 or R-4 occupancy.
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
        shutters = random.random() < 0.46

    # Terrain
    terrain = int(100 * BIM['z0'])

    # Stories
    # Buildings with more than 3 stories are mapped to the 3-story configuration
    stories = min(BIM['stories'], 3)

    bldg_config = f"WMUH" \
                  f"{stories}_" \
                  f"{BIM['roof_shape']}_" \
                  f"{roof_cover}_" \
                  f"{roof_quality}_" \
                  f"{SWR}_" \
                  f"{RDA}_" \
                  f"{RWC}_" \
                  f"{int(shutters)}_" \
                  f"{terrain}"

    return bldg_config

def auto_populate(BIM):
    """
    Populates the DL model for hurricane assessments in Atlantic County, NJ

    Assumptions:
    - Everything relevant to auto-population is provided in the Buiding
    Information Model (BIM).
    - The information expected in the BIM file is described in the parse_BIM
    method.

    Parameters
    ----------
    BIM_in: dictionary
        Contains the information that is available about the asset and will be
        used to auto-popualate the damage and loss model.

    Returns
    -------
    BIM_ap: dictionary
        Containes the extended BIM data.
    DL_ap: dictionary
        Contains the auto-populated loss model.
    """

    # parse the BIM data
    BIM_ap = parse_BIM(BIM)

    # identify the building class
    bldg_class = building_class(BIM_ap)

    # prepare the building configuration string
    if bldg_class == 'WSF':
        bldg_config = WSF_config(BIM_ap)
    elif bldg_class == 'WMUH':
        bldg_config = WMUH_config(BIM_ap)
    else:
        raise ValueError(
            f"Building class {bldg_class} not recognized by the "
            f"auto-population routine."
        )

    DL_ap = {
        '_method'      : 'HAZUS MH HU',
        'LossModel'    : {
            'DecisionVariables': {
                "ReconstructionCost": True
            },
            'ReplacementCost'  : 100
        },
        'Components'   : {
            bldg_config: [{
                'location'       : '1',
                'direction'      : '1',
                'median_quantity': '1.0',
                'unit'           : 'ea',
                'distribution'   : 'N/A'
            }]
        }
    }

    return BIM_ap, DL_ap