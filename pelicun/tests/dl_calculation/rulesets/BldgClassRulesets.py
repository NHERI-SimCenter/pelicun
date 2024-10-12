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


def building_class(bim: dict, hazard: str) -> str:  # noqa: C901
    """
    Short description.

    Long description

    Parameters
    ----------
    BIM: dictionary
        Information about the building characteristics.
    hazard: str
        Supported hazard types: "wind", "inundation"

    Returns
    -------
    bldg_class: str
        One of the standard building class labels from HAZUS

    """
    # check hazard
    if hazard not in {'wind', 'inundation'}:
        print(f'WARNING: The provided hazard is not recognized: {hazard}')  # noqa: T201

    if hazard == 'wind':
        if bim['BuildingType'] == 'Wood':
            if (bim['OccupancyClass'] == 'RES1') or (
                (bim['RoofShape'] != 'flt') and (bim['OccupancyClass'] == '')  # noqa: PLC1901
            ):
                # OccupancyClass = RES1
                # Wood Single-Family Homes (WSF1 or WSF2)
                # OR roof type = flat (HAZUS can only map flat to WSF1)
                # OR default (by '')
                if (
                    bim['RoofShape'] == 'flt'
                ):  # checking if there is a misclassication
                    bim['RoofShape'] = (
                        # ensure the WSF has gab (by default, note gab
                        # is more vulnerable than hip)
                        'gab'
                    )
                bldg_class = 'WSF'
            else:
                # OccupancyClass = RES3, RES5, RES6, or COM8
                # Wood Multi-Unit Hotel (WMUH1, WMUH2, or WMUH3)
                bldg_class = 'WMUH'

        elif bim['BuildingType'] == 'Steel':
            if (bim['DesignLevel'] == 'E') and (
                bim['OccupancyClass']
                in {'RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F'}
            ):
                # Steel Engineered Residential Building (SERBL, SERBM, SERBH)
                bldg_class = 'SERB'
            elif (bim['DesignLevel'] == 'E') and (
                bim['OccupancyClass']
                in {
                    'COM1',
                    'COM2',
                    'COM3',
                    'COM4',
                    'COM5',
                    'COM6',
                    'COM7',
                    'COM8',
                    'COM9',
                    'COM10',
                }
            ):
                # Steel Engineered Commercial Building (SECBL, SECBM, SECBH)
                bldg_class = 'SECB'
            elif (bim['DesignLevel'] == 'PE') and (
                bim['OccupancyClass']
                not in {'RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F'}
            ):
                # Steel Pre-Engineered Metal Building (SPMBS, SPMBM, SPMBL)
                bldg_class = 'SPMB'
            else:
                bldg_class = 'SECB'

        elif bim['BuildingType'] == 'Concrete':
            if (bim['DesignLevel'] == 'E') and (
                bim['OccupancyClass']
                in {
                    'RES3A',
                    'RES3B',
                    'RES3C',
                    'RES3D',
                    'RES3E',
                    'RES3F',
                    'RES5',
                    'RES6',
                }
            ):
                # Concrete Engineered Residential Building (CERBL, CERBM, CERBH)
                bldg_class = 'CERB'
            elif (bim['DesignLevel'] == 'E') and (
                bim['OccupancyClass']
                in {
                    'COM1',
                    'COM2',
                    'COM3',
                    'COM4',
                    'COM5',
                    'COM6',
                    'COM7',
                    'COM8',
                    'COM9',
                    'COM10',
                }
            ):
                # Concrete Engineered Commercial Building (CECBL, CECBM, CECBH)
                bldg_class = 'CECB'
            else:
                bldg_class = 'CECB'

        elif bim['BuildingType'] == 'Masonry':
            if bim['OccupancyClass'] == 'RES1':
                # OccupancyClass = RES1
                # Masonry Single-Family Homes (MSF1 or MSF2)
                bldg_class = 'MSF'
            elif (
                bim['OccupancyClass']
                in {'RES3A', 'RES3B', 'RES3C', 'RES3D', 'RES3E', 'RES3F'}
            ) and (bim['DesignLevel'] == 'E'):
                # Masonry Engineered Residential Building (MERBL, MERBM, MERBH)
                bldg_class = 'MERB'
            elif (
                bim['OccupancyClass']
                in {
                    'COM1',
                    'COM2',
                    'COM3',
                    'COM4',
                    'COM5',
                    'COM6',
                    'COM7',
                    'COM8',
                    'COM9',
                    'COM10',
                }
            ) and (bim['DesignLevel'] == 'E'):
                # Masonry Engineered Commercial Building (MECBL, MECBM, MECBH)
                bldg_class = 'MECB'
            elif bim['OccupancyClass'] in {
                'IND1',
                'IND2',
                'IND3',
                'IND4',
                'IND5',
                'IND6',
            }:
                # Masonry Low-Rise Masonry Warehouse/Factory (MLRI)
                bldg_class = 'MLRI'
            elif bim['OccupancyClass'] in {
                'RES3A',
                'RES3B',
                'RES3C',
                'RES3D',
                'RES3E',
                'RES3F',
                'RES5',
                'RES6',
                'COM8',
            }:
                # OccupancyClass = RES3X or COM8
                # Masonry Multi-Unit Hotel/Motel (MMUH1, MMUH2, or MMUH3)
                bldg_class = 'MMUH'
            elif (bim['NumberOfStories'] == 1) and (
                bim['OccupancyClass'] in {'COM1', 'COM2'}
            ):
                # Low-Rise Masonry Strip Mall (MLRM1 or MLRM2)
                bldg_class = 'MLRM'
            # elif (
            #     BIM['OccupancyClass']
            #     in [
            #         'RES3A',
            #         'RES3B',
            #         'RES3C',
            #         'RES3D',
            #         'RES3E',
            #         'RES3F',
            #         'RES5',
            #         'RES6',
            #         'COM8',
            #     ]
            # ) and (BIM['DesignLevel'] in ['NE', 'ME']):
            #     # Masonry Multi-Unit Hotel/Motel Non-Engineered
            #     # (MMUH1NE, MMUH2NE, or MMUH3NE)
            #     bldg_class = 'MMUHNE'
            else:
                bldg_class = 'MECB'  # for others not covered by the above

        elif bim['BuildingType'] == 'Manufactured':
            bldg_class = 'MH'

        else:
            bldg_class = 'WMUH'
            # if nan building type is provided, return the dominant class

    return bldg_class
