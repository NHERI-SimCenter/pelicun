# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-2022 Leland Stanford Junior University
# Copyright (c) 2018-2022 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

"""
This module has classes and methods to manage databases used by pelicun.

.. rubric:: Contents

.. autosummary::

    create_FEMA_P58_fragility_db
    create_FEMA_P58_repair_db
    create_FEMA_P58_bldg_redtag_db

    create_Hazus_EQ_fragility_db
    create_Hazus_EQ_repair_db

"""

from __future__ import annotations
import re
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
from scipy.stats import norm  # type: ignore
import pandas as pd

from pelicun import base
from pelicun.uq import fit_distribution_to_percentiles

idx = base.idx


# pylint: disable=too-many-statements
# pylint: disable=too-many-locals


def parse_DS_Hierarchy(DSH):
    """
    Parses the FEMA P58 DS hierarchy into a set of arrays.

    Parameters
    ----------
    DSH: str
       Damage state hierarchy

    Returns
    -------
    list
      Damage state setup
    """
    if DSH[:3] == 'Seq':
        DSH = DSH[4:-1]

    DS_setup = []

    while len(DSH) > 0:
        if DSH[:2] == 'DS':
            DS_setup.append(DSH[:3])
            DSH = DSH[4:]
        elif DSH[:5] in {'MutEx', 'Simul'}:
            closing_pos = DSH.find(')')
            subDSH = DSH[: closing_pos + 1]
            DSH = DSH[closing_pos + 2 :]

            DS_setup.append([subDSH[:5]] + subDSH[6:-1].split(','))

    return DS_setup


def create_FEMA_P58_fragility_db(
    source_file,
    meta_file='',
    target_data_file='damage_DB_FEMA_P58_2nd.csv',
    target_meta_file='damage_DB_FEMA_P58_2nd.json',
):
    """
    Create a fragility parameter database based on the FEMA P58 data

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the fragility data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the fragility metadata should be saved. A json file is
        expected.

    Raises
    ------
    ValueError
        If there are problems with the mutually exclusive damage state
        definition of some component.
    """

    # parse the source file
    df = pd.read_excel(
        source_file,
        sheet_name='Summary',
        header=2,
        index_col=1,
        true_values=["YES", "Yes", "yes"],
        false_values=["NO", "No", "no"],
    )

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, 'r', encoding='utf-8') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # remove the empty rows and columns
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # filter the columns that we need for the fragility database
    cols_to_db = [
        "Demand Parameter (value):",
        "Demand Parameter (unit):",
        "Demand Location (use floor above? Yes/No)",
        "Directional?",
        "DS Hierarchy",
        "DS 1, Probability",
        "DS 1, Median Demand",
        "DS 1, Total Dispersion (Beta)",
        "DS 2, Probability",
        "DS 2, Median Demand",
        "DS 2, Total Dispersion (Beta)",
        "DS 3, Probability",
        "DS 3, Median Demand",
        "DS 3, Total Dispersion (Beta)",
        "DS 4, Probability",
        "DS 4, Median Demand",
        "DS 4, Total Dispersion (Beta)",
        "DS 5, Probability",
        "DS 5, Median Demand",
        "DS 5, Total Dispersion (Beta)",
    ]

    # filter the columns that we need for the metadata
    cols_to_meta = [
        "Component Name",
        "Component Description",
        "Construction Quality:",
        "Seismic Installation Conditions:",
        "Comments / Notes",
        "Author",
        "Fragility Unit of Measure",
        "Round to Integer Unit?",
        "DS 1, Description",
        "DS 1, Repair Description",
        "DS 2, Description",
        "DS 2, Repair Description",
        "DS 3, Description",
        "DS 3, Repair Description",
        "DS 4, Description",
        "DS 4, Repair Description",
        "DS 5, Description",
        "DS 5, Repair Description",
    ]

    # remove special characters to make it easier to work with column names
    str_map = {
        ord(' '): "_",
        ord(':'): None,
        ord('('): None,
        ord(')'): None,
        ord('?'): None,
        ord('/'): None,
        ord(','): None,
    }

    df_db_source = df.loc[:, cols_to_db]
    df_db_source.columns = [s.translate(str_map) for s in cols_to_db]
    df_db_source.sort_index(inplace=True)

    df_meta = df.loc[:, cols_to_meta]
    df_meta.columns = [s.translate(str_map) for s in cols_to_meta]
    # replace missing values with an empty string
    df_meta.fillna('', inplace=True)
    # the metadata shall be stored in strings
    df_meta = df_meta.astype(str)

    # initialize the output fragility table
    df_db = pd.DataFrame(
        columns=[
            "Index",
            "Incomplete",
            "Demand-Type",
            "Demand-Unit",
            "Demand-Offset",
            "Demand-Directional",
            "LS1-Family",
            "LS1-Theta_0",
            "LS1-Theta_1",
            "LS1-DamageStateWeights",
            "LS2-Family",
            "LS2-Theta_0",
            "LS2-Theta_1",
            "LS2-DamageStateWeights",
            "LS3-Family",
            "LS3-Theta_0",
            "LS3-Theta_1",
            "LS3-DamageStateWeights",
            "LS4-Family",
            "LS4-Theta_0",
            "LS4-Theta_1",
            "LS4-DamageStateWeights",
        ],
        index=df_db_source.index,
        dtype=float,
    )

    # initialize the dictionary that stores the fragility metadata
    meta_dict = {}

    # add the general information to the meta dict
    if "_GeneralInformation" in frag_meta.keys():
        frag_meta = frag_meta["_GeneralInformation"]

        # remove the decision variable part from the general info
        frag_meta.pop("DecisionVariables", None)

        meta_dict.update({"_GeneralInformation": frag_meta})

    # conversion dictionary for demand types
    convert_demand_type = {
        'Story Drift Ratio': "Peak Interstory Drift Ratio",
        'Link Rotation Angle': "Peak Link Rotation Angle",
        'Effective Drift': "Peak Effective Drift Ratio",
        'Link Beam Chord Rotation': "Peak Link Beam Chord Rotation",
        'Peak Floor Acceleration': "Peak Floor Acceleration",
        'Peak Floor Velocity': "Peak Floor Velocity",
    }

    # conversion dictionary for demand unit names
    convert_demand_unit = {
        'Unit less': 'unitless',
        'Radians': 'rad',
        'g': 'g',
        'meter/sec': 'mps',
    }

    # for each component...
    # (this approach is not efficient, but easy to follow which was considered
    # more important than efficiency.)
    for cmp in df_db_source.itertuples():
        # create a dotted component index
        ID = cmp.Index.split('.')
        cmpID = f'{ID[0][0]}.{ID[0][1:3]}.{ID[0][3:5]}.{ID[1]}'

        # store the new index
        df_db.loc[cmp.Index, 'Index'] = cmpID

        # assume the component information is complete
        incomplete = False

        # store demand specifications
        df_db.loc[cmp.Index, 'Demand-Type'] = convert_demand_type[
            cmp.Demand_Parameter_value
        ]
        df_db.loc[cmp.Index, 'Demand-Unit'] = convert_demand_unit[
            cmp.Demand_Parameter_unit
        ]
        df_db.loc[cmp.Index, 'Demand-Offset'] = int(
            cmp.Demand_Location_use_floor_above_YesNo
        )
        df_db.loc[cmp.Index, 'Demand-Directional'] = int(cmp.Directional)

        # parse the damage state hierarchy
        DS_setup = parse_DS_Hierarchy(cmp.DS_Hierarchy)

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # start with a comp. description
        if not pd.isna(cmp_meta['Component_Description']):
            comments = cmp_meta['Component_Description']
        else:
            comments = ''

        # the additional fields are added to the description if they exist

        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += f'\nConstruction Quality: {cmp_meta["Construction_Quality"]}'

        if cmp_meta['Seismic_Installation_Conditions'] not in [
            'Not Specified',
            'Not applicable',
            'Unknown',
            'Any',
        ]:
            comments += (
                f'\nSeismic Installation Conditions: '
                f'{cmp_meta["Seismic_Installation_Conditions"]}'
            )

        if cmp_meta['Comments__Notes'] != 'None':
            comments += f'\nNotes: {cmp_meta["Comments__Notes"]}'

        if cmp_meta['Author'] not in ['Not Given', 'By User']:
            comments += f'\nAuthor: {cmp_meta["Author"]}'

        # get the suggested block size and replace the misleading values with ea
        block_size = cmp_meta['Fragility_Unit_of_Measure'].split(' ')[::-1]

        meta_data = {
            "Description": cmp_meta['Component_Name'],
            "Comments": comments,
            "SuggestedComponentBlockSize": ' '.join(block_size),
            "RoundUpToIntegerQuantity": cmp_meta['Round_to_Integer_Unit'],
            "LimitStates": {},
        }

        # now look at each Limit State
        for LS_i, LS_contents in enumerate(DS_setup):
            LS_i = LS_i + 1
            LS_contents = np.atleast_1d(LS_contents)

            ls_meta = {}

            # start with the special cases with multiple DSs in an LS
            if LS_contents[0] in {'MutEx', 'Simul'}:
                # collect the fragility data for the member DSs
                median_demands = []
                dispersions = []
                weights = []
                for ds in LS_contents[1:]:
                    median_demands.append(getattr(cmp, f"DS_{ds[2]}_Median_Demand"))

                    dispersions.append(
                        getattr(cmp, f"DS_{ds[2]}_Total_Dispersion_Beta")
                    )

                    weights.append(getattr(cmp, f"DS_{ds[2]}_Probability"))

                # make sure the specified distribution parameters are appropriate
                if (np.unique(median_demands).size != 1) or (
                    np.unique(dispersions).size != 1
                ):
                    raise ValueError(
                        f"Incorrect mutually exclusive DS "
                        f"definition in component {cmp.Index} at "
                        f"Limit State {LS_i}"
                    )

                if LS_contents[0] == 'MutEx':
                    # in mutually exclusive cases, make sure the specified DS
                    # weights sum up to one
                    np.testing.assert_allclose(
                        np.sum(np.array(weights, dtype=float)),
                        1.0,
                        err_msg=f"Mutually exclusive Damage State weights do "
                        f"not sum to 1.0 in component {cmp.Index} at "
                        f"Limit State {LS_i}",
                    )

                    # and save all DS metadata under this Limit State
                    for ds in LS_contents[1:]:
                        ds_id = ds[2]

                        repair_action = cmp_meta[f"DS_{ds_id}_Repair_Description"]
                        if pd.isna(repair_action):
                            repair_action = "<missing data>"

                        ls_meta.update(
                            {
                                f"DS{ds_id}": {
                                    "Description": cmp_meta[
                                        f"DS_{ds_id}_Description"
                                    ],
                                    "RepairAction": repair_action,
                                }
                            }
                        )

                else:
                    # in simultaneous cases, convert simultaneous weights into
                    # mutexc weights
                    sim_ds_count = len(LS_contents) - 1
                    ds_count = 2 ** (sim_ds_count) - 1

                    sim_weights = []

                    for ds_id in range(1, ds_count + 1):
                        ds_map = format(ds_id, f'0{sim_ds_count}b')

                        sim_weights.append(
                            np.product(
                                [
                                    (
                                        weights[ds_i]
                                        if ds_map[-ds_i - 1] == '1'
                                        else 1.0 - weights[ds_i]
                                    )
                                    for ds_i in range(sim_ds_count)
                                ]
                            )
                        )

                        # save ds metadata - we need to be clever here
                        # the original metadata is saved for the pure cases
                        # when only one DS is triggered
                        # all other DSs store information about which
                        # combination of pure DSs they represent

                        if ds_map.count('1') == 1:
                            ds_pure_id = ds_map[::-1].find('1') + 1

                            repair_action = cmp_meta[
                                f"DS_{ds_pure_id}_Repair_Description"
                            ]
                            if pd.isna(repair_action):
                                repair_action = "<missing data>"

                            ls_meta.update(
                                {
                                    f"DS{ds_id}": {
                                        "Description": f"Pure DS{ds_pure_id}. "
                                        + cmp_meta[f"DS_{ds_pure_id}_Description"],
                                        "RepairAction": repair_action,
                                    }
                                }
                            )

                        else:
                            ds_combo = [
                                f'DS{_.start() + 1}'
                                for _ in re.finditer('1', ds_map[::-1])
                            ]

                            ls_meta.update(
                                {
                                    f"DS{ds_id}": {
                                        "Description": 'Combination of '
                                        + ' & '.join(ds_combo),
                                        "RepairAction": (
                                            'Combination of pure DS repair actions.'
                                        ),
                                    }
                                }
                            )

                    # adjust weights to respect the assumption that at least
                    # one DS will occur (i.e., the case with all DSs returning
                    # False is not part of the event space)
                    sim_weights_array = np.array(sim_weights) / np.sum(sim_weights)

                    weights = sim_weights_array

                theta_0 = median_demands[0]
                theta_1 = dispersions[0]
                weights_str = ' | '.join([f"{w:.6f}" for w in weights])

                df_db.loc[cmp.Index, f'LS{LS_i}-DamageStateWeights'] = weights_str

            # then look at the sequential DS cases
            elif LS_contents[0].startswith('DS'):
                # this is straightforward, store the data in the table and dict
                ds_id = LS_contents[0][2]

                theta_0 = getattr(cmp, f"DS_{ds_id}_Median_Demand")
                theta_1 = getattr(cmp, f"DS_{ds_id}_Total_Dispersion_Beta")

                repair_action = cmp_meta[f"DS_{ds_id}_Repair_Description"]
                if pd.isna(repair_action):
                    repair_action = "<missing data>"

                ls_meta.update(
                    {
                        f"DS{ds_id}": {
                            "Description": cmp_meta[f"DS_{ds_id}_Description"],
                            "RepairAction": repair_action,
                        }
                    }
                )

            # FEMA P58 assumes lognormal distribution for every fragility
            df_db.loc[cmp.Index, f'LS{LS_i}-Family'] = 'lognormal'

            # identify incomplete cases...

            # where theta is missing
            if theta_0 != 'By User':
                df_db.loc[cmp.Index, f'LS{LS_i}-Theta_0'] = theta_0
            else:
                incomplete = True

            # where beta is missing
            if theta_1 != 'By User':
                df_db.loc[cmp.Index, f'LS{LS_i}-Theta_1'] = theta_1
            else:
                incomplete = True

            # store the collected metadata for this limit state
            meta_data['LimitStates'].update({f"LS{LS_i}": ls_meta})

        # store the incomplete flag for this component
        df_db.loc[cmp.Index, 'Incomplete'] = int(incomplete)

        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.set_index('Index', inplace=True)

    # rename the index
    df_db.index.name = "ID"

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the fragility data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the fragility data from FEMA P58")


def create_FEMA_P58_repair_db(
    source_file,
    meta_file='',
    target_data_file='loss_repair_DB_FEMA_P58_2nd.csv',
    target_meta_file='loss_repair_DB_FEMA_P58_2nd.json',
):
    """
    Create a repair consequence parameter database based on the FEMA P58 data

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the consequence data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the consequence metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    df = pd.concat(
        [
            pd.read_excel(source_file, sheet_name=sheet, header=2, index_col=1)
            for sheet in ('Summary', 'Cost Summary', 'Env Summary')
        ],
        axis=1,
    )

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, 'r', encoding='utf-8') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # remove duplicate columns
    # (there are such because we joined two tables that were read separately)
    df = df.loc[:, ~df.columns.duplicated()]

    # remove empty rows and columns
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # filter the columns we need for the repair database
    cols_to_db = [
        "Fragility Unit of Measure",
        'DS Hierarchy',
    ]
    for DS_i in range(1, 6):
        cols_to_db += [
            f"Best Fit, DS{DS_i}",
            f"Lower Qty Mean, DS{DS_i}",
            f"Upper Qty Mean, DS{DS_i}",
            f"Lower Qty Cutoff, DS{DS_i}",
            f"Upper Qty Cutoff, DS{DS_i}",
            f"CV / Dispersion, DS{DS_i}",
            # --------------------------
            f"Best Fit, DS{DS_i}.1",
            f"Lower Qty Mean, DS{DS_i}.1",
            f"Upper Qty Mean, DS{DS_i}.1",
            f"Lower Qty Cutoff, DS{DS_i}.1",
            f"Upper Qty Cutoff, DS{DS_i}.1",
            f"CV / Dispersion, DS{DS_i}.2",
            f"DS {DS_i}, Long Lead Time",
            # --------------------------
            f'Repair Cost, p10, DS{DS_i}',
            f'Repair Cost, p50, DS{DS_i}',
            f'Repair Cost, p90, DS{DS_i}',
            f'Time, p10, DS{DS_i}',
            f'Time, p50, DS{DS_i}',
            f'Time, p90, DS{DS_i}',
            f'Mean Value, DS{DS_i}',
            f'Mean Value, DS{DS_i}.1',
            # --------------------------
            # Columns added for the Environmental loss
            f"DS{DS_i} Best Fit",
            f"DS{DS_i} CV or Beta",
            # --------------------------
            f"DS{DS_i} Best Fit.1",
            f"DS{DS_i} CV or Beta.1",
            # --------------------------
            f"DS{DS_i} Embodied Carbon (kg CO2eq)",
            f"DS{DS_i} Embodied Energy (MJ)",
        ]

    # filter the columns that we need for the metadata
    cols_to_meta = [
        "Component Name",
        "Component Description",
        "Construction Quality:",
        "Seismic Installation Conditions:",
        "Comments / Notes",
        "Author",
        "Fragility Unit of Measure",
        "Round to Integer Unit?",
        "DS 1, Description",
        "DS 1, Repair Description",
        "DS 2, Description",
        "DS 2, Repair Description",
        "DS 3, Description",
        "DS 3, Repair Description",
        "DS 4, Description",
        "DS 4, Repair Description",
        "DS 5, Description",
        "DS 5, Repair Description",
    ]

    # remove special characters to make it easier to work with column names
    str_map = {
        ord(' '): "_",
        ord('.'): "_",
        ord(':'): None,
        ord('('): None,
        ord(')'): None,
        ord('?'): None,
        ord('/'): None,
        ord(','): None,
    }

    df_db_source = df.loc[:, cols_to_db]
    df_db_source.columns = [s.translate(str_map) for s in cols_to_db]
    df_db_source.sort_index(inplace=True)

    df_meta = df.loc[:, cols_to_meta]
    df_meta.columns = [s.translate(str_map) for s in cols_to_meta]

    df_db_source.replace('BY USER', np.nan, inplace=True)

    # initialize the output loss table
    # define the columns
    out_cols = [
        "Index",
        "Incomplete",
        "Quantity-Unit",
        "DV-Unit",
    ]
    for DS_i in range(1, 16):
        out_cols += [
            f"DS{DS_i}-Family",
            f"DS{DS_i}-Theta_0",
            f"DS{DS_i}-Theta_1",
            f"DS{DS_i}-LongLeadTime",
        ]

    # create the MultiIndex
    comps = df_db_source.index.values
    DVs = ['Cost', 'Time', 'Carbon', 'Energy']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])

    df_db = pd.DataFrame(columns=out_cols, index=df_MI, dtype=float)

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    # add the general information to the meta dict
    if "_GeneralInformation" in frag_meta.keys():
        frag_meta = frag_meta["_GeneralInformation"]

        meta_dict.update({"_GeneralInformation": frag_meta})

    convert_family = {'LogNormal': 'lognormal', 'Normal': 'normal'}

    # for each component...
    # (this approach is not efficient, but easy to follow which was considered
    # more important than efficiency.)
    for cmp in df_db_source.itertuples():
        ID = cmp.Index.split('.')
        cmpID = f'{ID[0][0]}.{ID[0][1:3]}.{ID[0][3:5]}.{ID[1]}'

        # store the new index
        df_db.loc[cmp.Index, 'Index'] = cmpID

        # assume the component information is complete
        incomplete_cost = False
        incomplete_time = False
        incomplete_carbon = False
        incomplete_energy = False

        # store units

        df_db.loc[cmp.Index, 'Quantity-Unit'] = ' '.join(
            cmp.Fragility_Unit_of_Measure.split(' ')[::-1]
        ).strip()
        df_db.loc[(cmp.Index, 'Cost'), 'DV-Unit'] = "USD_2011"
        df_db.loc[(cmp.Index, 'Time'), 'DV-Unit'] = "worker_day"
        df_db.loc[(cmp.Index, 'Carbon'), 'DV-Unit'] = "kg"
        df_db.loc[(cmp.Index, 'Energy'), 'DV-Unit'] = "MJ"

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # start with a comp. description
        if not pd.isna(cmp_meta['Component_Description']):
            comments = cmp_meta['Component_Description']
        else:
            comments = ''

        # the additional fields are added to the description if they exist
        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += (
                f'\nConstruction Quality: ' f'{cmp_meta["Construction_Quality"]}'
            )

        if cmp_meta['Seismic_Installation_Conditions'] not in [
            'Not Specified',
            'Not applicable',
            'Unknown',
            'Any',
        ]:
            comments += (
                f'\nSeismic Installation Conditions: '
                f'{cmp_meta["Seismic_Installation_Conditions"]}'
            )

        if cmp_meta['Comments__Notes'] != 'None':
            comments += f'\nNotes: {cmp_meta["Comments__Notes"]}'

        if cmp_meta['Author'] not in ['Not Given', 'By User']:
            comments += f'\nAuthor: {cmp_meta["Author"]}'

        # get the suggested block size and replace the misleading values with ea
        block_size = cmp_meta['Fragility_Unit_of_Measure'].split(' ')[::-1]

        meta_data = {
            "Description": cmp_meta['Component_Name'],
            "Comments": comments,
            "SuggestedComponentBlockSize": ' '.join(block_size),
            "RoundUpToIntegerQuantity": cmp_meta['Round_to_Integer_Unit'],
            "ControllingDemand": "Damage Quantity",
            "DamageStates": {},
        }

        # Handle components with simultaneous damage states separately
        if 'Simul' in cmp.DS_Hierarchy:
            # Note that we are assuming that all damage states are triggered by
            # a single limit state in these components.
            # This assumption holds for the second edition of FEMA P58, but it
            # might need to be revisited in future editions.

            cost_est = {}
            time_est = {}
            carbon_est = {}
            energy_est = {}

            # get the p10, p50, and p90 estimates for all damage states
            for DS_i in range(1, 6):
                if not pd.isna(getattr(cmp, f'Repair_Cost_p10_DS{DS_i}')):
                    cost_est.update(
                        {
                            f'DS{DS_i}': np.array(
                                [
                                    getattr(cmp, f'Repair_Cost_p10_DS{DS_i}'),
                                    getattr(cmp, f'Repair_Cost_p50_DS{DS_i}'),
                                    getattr(cmp, f'Repair_Cost_p90_DS{DS_i}'),
                                    getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}'),
                                    getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}'),
                                ]
                            )
                        }
                    )

                    time_est.update(
                        {
                            f'DS{DS_i}': np.array(
                                [
                                    getattr(cmp, f'Time_p10_DS{DS_i}'),
                                    getattr(cmp, f'Time_p50_DS{DS_i}'),
                                    getattr(cmp, f'Time_p90_DS{DS_i}'),
                                    getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1'),
                                    getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}_1'),
                                    int(
                                        getattr(cmp, f'DS_{DS_i}_Long_Lead_Time')
                                        == 'YES'
                                    ),
                                ]
                            )
                        }
                    )

                if not pd.isna(getattr(cmp, f'DS{DS_i}_Embodied_Carbon_kg_CO2eq')):
                    theta_0, theta_1, family = [
                        getattr(cmp, f'DS{DS_i}_Embodied_Carbon_kg_CO2eq'),
                        getattr(cmp, f'DS{DS_i}_CV_or_Beta'),
                        getattr(cmp, f'DS{DS_i}_Best_Fit'),
                    ]

                    if family == 'Normal':
                        p10, p50, p90 = norm.ppf(
                            [0.1, 0.5, 0.9], loc=theta_0, scale=theta_0 * theta_1
                        )
                    elif family == 'LogNormal':
                        p10, p50, p90 = np.exp(
                            norm.ppf(
                                [0.1, 0.5, 0.9], loc=np.log(theta_0), scale=theta_1
                            )
                        )

                    carbon_est.update({f'DS{DS_i}': np.array([p10, p50, p90])})

                if not pd.isna(getattr(cmp, f'DS{DS_i}_Embodied_Energy_MJ')):
                    theta_0, theta_1, family = [
                        getattr(cmp, f'DS{DS_i}_Embodied_Energy_MJ'),
                        getattr(cmp, f'DS{DS_i}_CV_or_Beta_1'),
                        getattr(cmp, f'DS{DS_i}_Best_Fit_1'),
                    ]

                    if family == 'Normal':
                        p10, p50, p90 = norm.ppf(
                            [0.1, 0.5, 0.9], loc=theta_0, scale=theta_0 * theta_1
                        )
                    elif family == 'LogNormal':
                        p10, p50, p90 = np.exp(
                            norm.ppf(
                                [0.1, 0.5, 0.9], loc=np.log(theta_0), scale=theta_1
                            )
                        )

                    energy_est.update({f'DS{DS_i}': np.array([p10, p50, p90])})

            # now prepare the equivalent mutex damage states
            sim_ds_count = len(cost_est.keys())
            ds_count = 2 ** (sim_ds_count) - 1

            for DS_i in range(1, ds_count + 1):
                ds_map = format(DS_i, f'0{sim_ds_count}b')

                cost_vals = np.sum(
                    [
                        (
                            cost_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(5)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                time_vals = np.sum(
                    [
                        (
                            time_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(6)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                carbon_vals = np.sum(
                    [
                        (
                            carbon_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(3)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                energy_vals = np.sum(
                    [
                        (
                            energy_est[f'DS{ds_i + 1}']
                            if ds_map[-ds_i - 1] == '1'
                            else np.zeros(3)
                        )
                        for ds_i in range(sim_ds_count)
                    ],
                    axis=0,
                )

                # fit a distribution
                family_hat, theta_hat = fit_distribution_to_percentiles(
                    cost_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal']
                )

                cost_theta = theta_hat
                if family_hat == 'normal':
                    cost_theta[1] = cost_theta[1] / cost_theta[0]

                time_theta = [
                    time_vals[1],
                    np.sqrt(cost_theta[1] ** 2.0 + 0.25**2.0),
                ]

                # fit distributions to environmental impact consequences
                (
                    family_hat_carbon,
                    theta_hat_carbon,
                ) = fit_distribution_to_percentiles(
                    carbon_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal']
                )

                carbon_theta = theta_hat_carbon
                if family_hat_carbon == 'normal':
                    carbon_theta[1] = carbon_theta[1] / carbon_theta[0]

                (
                    family_hat_energy,
                    theta_hat_energy,
                ) = fit_distribution_to_percentiles(
                    energy_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal']
                )

                energy_theta = theta_hat_energy
                if family_hat_energy == 'normal':
                    energy_theta[1] = energy_theta[1] / energy_theta[0]

                # Note that here we assume that the cutoff quantities are
                # identical across damage states.
                # This assumption holds for the second edition of FEMA P58, but
                # it might need to be revisited in future editions.
                cost_qnt_low = getattr(cmp, 'Lower_Qty_Cutoff_DS1')
                cost_qnt_up = getattr(cmp, 'Upper_Qty_Cutoff_DS1')
                time_qnt_low = getattr(cmp, 'Lower_Qty_Cutoff_DS1_1')
                time_qnt_up = getattr(cmp, 'Upper_Qty_Cutoff_DS1_1')

                # store the results
                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                    f"{cost_vals[3]:g},{cost_vals[4]:g}|"
                    f"{cost_qnt_low:g},{cost_qnt_up:g}"
                )

                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_1'] = (
                    f"{cost_theta[1]:g}"
                )

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_0'] = (
                    f"{time_vals[3]:g},{time_vals[4]:g}|"
                    f"{time_qnt_low:g},{time_qnt_up:g}"
                )

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_1'] = (
                    f"{time_theta[1]:g}"
                )

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-LongLeadTime'] = int(
                    time_vals[5] > 0
                )

                df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Family'] = (
                    family_hat_carbon
                )

                df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_0'] = (
                    f"{carbon_theta[0]:g}"
                )

                df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_1'] = (
                    f"{carbon_theta[1]:g}"
                )

                df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Family'] = (
                    family_hat_energy
                )

                df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_0'] = (
                    f"{energy_theta[0]:g}"
                )

                df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_1'] = (
                    f"{energy_theta[1]:g}"
                )

                if ds_map.count('1') == 1:
                    ds_pure_id = ds_map[::-1].find('1') + 1

                    repair_action = cmp_meta[f"DS_{ds_pure_id}_Repair_Description"]
                    if pd.isna(repair_action):
                        repair_action = "<missing data>"

                    meta_data['DamageStates'].update(
                        {
                            f"DS{DS_i}": {
                                "Description": f"Pure DS{ds_pure_id}. "
                                + cmp_meta[f"DS_{ds_pure_id}_Description"],
                                "RepairAction": repair_action,
                            }
                        }
                    )

                else:
                    ds_combo = [
                        f'DS{_.start() + 1}' for _ in re.finditer('1', ds_map[::-1])
                    ]

                    meta_data['DamageStates'].update(
                        {
                            f"DS{DS_i}": {
                                "Description": 'Combination of '
                                + ' & '.join(ds_combo),
                                "RepairAction": 'Combination of pure DS repair '
                                'actions.',
                            }
                        }
                    )

        # for every other component...
        else:
            # now look at each Damage State
            for DS_i in range(1, 6):
                # cost
                if not pd.isna(getattr(cmp, f'Best_Fit_DS{DS_i}')):
                    df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'Best_Fit_DS{DS_i}')]
                    )

                    if not pd.isna(getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}')):
                        theta_0_low = getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}')
                        theta_0_up = getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}')
                        qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS{DS_i}')
                        qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS{DS_i}')

                        if theta_0_low == 0.0 and theta_0_up == 0.0:
                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = (
                                np.nan
                            )

                        else:
                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                                f"{theta_0_low:g},{theta_0_up:g}|"
                                f"{qnt_low:g},{qnt_up:g}"
                            )

                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_1'] = (
                                f"{getattr(cmp, f'CV__Dispersion_DS{DS_i}'):g}"
                            )

                    else:
                        incomplete_cost = True

                    repair_action = cmp_meta[f"DS_{DS_i}_Repair_Description"]
                    if pd.isna(repair_action):
                        repair_action = "<missing data>"

                    meta_data['DamageStates'].update(
                        {
                            f"DS{DS_i}": {
                                "Description": cmp_meta[f"DS_{DS_i}_Description"],
                                "RepairAction": repair_action,
                            }
                        }
                    )

                # time
                if not pd.isna(getattr(cmp, f'Best_Fit_DS{DS_i}_1')):
                    df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'Best_Fit_DS{DS_i}_1')]
                    )

                    if not pd.isna(getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1')):
                        theta_0_low = getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1')
                        theta_0_up = getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}_1')
                        qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS{DS_i}_1')
                        qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS{DS_i}_1')

                        if theta_0_low == 0.0 and theta_0_up == 0.0:
                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = (
                                np.nan
                            )

                        else:
                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_0'] = (
                                f"{theta_0_low:g},{theta_0_up:g}|"
                                f"{qnt_low:g},{qnt_up:g}"
                            )

                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_1'] = (
                                f"{getattr(cmp, f'CV__Dispersion_DS{DS_i}_2'):g}"
                            )

                        df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-LongLeadTime'] = (
                            int(getattr(cmp, f'DS_{DS_i}_Long_Lead_Time') == 'YES')
                        )

                    else:
                        incomplete_time = True

                # Carbon
                if not pd.isna(getattr(cmp, f'DS{DS_i}_Best_Fit')):
                    df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'DS{DS_i}_Best_Fit')]
                    )

                    df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_0'] = getattr(
                        cmp, f'DS{DS_i}_Embodied_Carbon_kg_CO2eq'
                    )

                    df_db.loc[(cmp.Index, 'Carbon'), f'DS{DS_i}-Theta_1'] = getattr(
                        cmp, f'DS{DS_i}_CV_or_Beta'
                    )

                # Energy
                if not pd.isna(getattr(cmp, f'DS{DS_i}_Best_Fit_1')):
                    df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'DS{DS_i}_Best_Fit_1')]
                    )

                    df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_0'] = getattr(
                        cmp, f'DS{DS_i}_Embodied_Energy_MJ'
                    )

                    df_db.loc[(cmp.Index, 'Energy'), f'DS{DS_i}-Theta_1'] = getattr(
                        cmp, f'DS{DS_i}_CV_or_Beta_1'
                    )

        df_db.loc[(cmp.Index, 'Cost'), 'Incomplete'] = int(incomplete_cost)
        df_db.loc[(cmp.Index, 'Time'), 'Incomplete'] = int(incomplete_time)
        df_db.loc[(cmp.Index, 'Carbon'), 'Incomplete'] = int(incomplete_carbon)
        df_db.loc[(cmp.Index, 'Energy'), 'Incomplete'] = int(incomplete_energy)
        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.index = pd.MultiIndex.from_arrays(
        [df_db['Index'].values, df_db.index.get_level_values(1)]
    )

    df_db.drop('Index', axis=1, inplace=True)

    # review the database and drop rows with no information
    cmp_to_drop = []
    for cmp in df_db.index:
        empty = True

        for DS_i in range(1, 6):
            if not pd.isna(df_db.loc[cmp, f'DS{DS_i}-Family']):
                empty = False
                break

        if empty:
            cmp_to_drop.append(cmp)

    df_db.drop(cmp_to_drop, axis=0, inplace=True)
    for cmp in cmp_to_drop:
        if cmp[0] in meta_dict:
            del meta_dict[cmp[0]]

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    df_db = base.convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the repair consequence data from FEMA P58")


def create_Hazus_EQ_fragility_db(
    source_file,
    meta_file='',
    target_data_file='damage_DB_Hazus_EQ_bldg.csv',
    target_meta_file='damage_DB_Hazus_EQ_bldg.json',
    resolution='building',
):
    """
    Create a database file based on the HAZUS EQ Technical Manual

    This method was developed to process a json file with tabulated data from
    v5.1 of the Hazus Earthquake Technical Manual. The json file is included
    under data_sources in the SimCenter DB_DamageAndLoss repo on GitHub.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the fragility data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the fragility metadata should be saved. A json file is
        expected.
    resolution: string
        If building, the function produces the conventional Hazus
        fragilities. If story, the function produces story-level
        fragilities.

    """

    # adjust the target filenames if needed
    if resolution == 'story':
        target_data_file = target_data_file.replace('bldg', 'story')
        target_meta_file = target_meta_file.replace('bldg', 'story')

    # parse the source file
    with open(source_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, 'r', encoding='utf-8') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # prepare lists of labels for various building features
    design_levels = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'].keys()
    )

    building_types = list(
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys()
    )

    convert_design_level = {
        'High_code': 'HC',
        'Moderate_code': 'MC',
        'Low_code': 'LC',
        'Pre_code': 'PC',
    }

    # initialize the fragility table
    df_db = pd.DataFrame(
        columns=[
            "ID",
            "Incomplete",
            "Demand-Type",
            "Demand-Unit",
            "Demand-Offset",
            "Demand-Directional",
            "LS1-Family",
            "LS1-Theta_0",
            "LS1-Theta_1",
            "LS1-DamageStateWeights",
            "LS2-Family",
            "LS2-Theta_0",
            "LS2-Theta_1",
            "LS2-DamageStateWeights",
            "LS3-Family",
            "LS3-Theta_0",
            "LS3-Theta_1",
            "LS3-DamageStateWeights",
            "LS4-Family",
            "LS4-Theta_0",
            "LS4-Theta_1",
            "LS4-DamageStateWeights",
        ],
        index=np.arange(len(building_types) * len(design_levels) * 5),
        dtype=float,
    )

    # initialize the dictionary that stores the fragility metadata
    meta_dict = {}

    # add the general information to the meta dict
    if "_GeneralInformation" in frag_meta.keys():
        GI = frag_meta["_GeneralInformation"]

        # remove the decision variable part from the general info
        GI.pop("DecisionVariables", None)

        for key, item in deepcopy(GI).items():
            if key == 'ComponentGroups_Damage':
                GI.update({'ComponentGroups': item})

            if key.startswith('ComponentGroups'):
                GI.pop(key, None)

        meta_dict.update({"_GeneralInformation": GI})

    counter = 0

    # First, prepare the structural fragilities
    S_data = raw_data['Structural_Fragility_Groups']

    for bt in building_types:
        for dl in design_levels:
            if bt in S_data['EDP_limits'][dl].keys():
                # add a dot in bt between structure and height labels, if needed
                if (len(bt) > 2) and (bt[-1] in {'L', 'M', 'H'}):
                    bt_exp = f'{bt[:-1]}.{bt[-1]}'
                    st = bt[:-1]
                    hc = bt[-1]
                else:
                    bt_exp = bt
                    st = bt
                    hc = None

                # story-level fragilities are based only on the low rise archetypes
                if resolution == 'story':
                    if hc in {'M', 'H'}:
                        continue
                    if hc == 'L':
                        bt_exp = st

                # create the component id
                cmp_id = f'STR.{bt_exp}.{convert_design_level[dl]}'
                df_db.loc[counter, 'ID'] = cmp_id

                # store demand specifications
                if resolution == 'building':
                    df_db.loc[counter, 'Demand-Type'] = "Peak Roof Drift Ratio"
                elif resolution == 'story':
                    df_db.loc[counter, 'Demand-Type'] = "Peak Interstory Drift Ratio"

                df_db.loc[counter, 'Demand-Unit'] = "rad"
                df_db.loc[counter, 'Demand-Offset'] = 0

                # add metadata
                if hc is not None:
                    cmp_meta = {
                        "Description": (
                            frag_meta['Meta']['Collections']['STR']['Description']
                            + ", "
                            + frag_meta['Meta']['StructuralSystems'][st][
                                'Description'
                            ]
                            + ", "
                            + frag_meta['Meta']['HeightClasses'][hc]['Description']
                            + ", "
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Description']
                        ),
                        "Comments": (
                            frag_meta['Meta']['Collections']['STR']['Comment']
                            + "\n"
                            + frag_meta['Meta']['StructuralSystems'][st]['Comment']
                            + "\n"
                            + frag_meta['Meta']['HeightClasses'][hc]['Comment']
                            + "\n"
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Comment']
                        ),
                        "SuggestedComponentBlockSize": "1 EA",
                        "RoundUpToIntegerQuantity": "True",
                        "LimitStates": {},
                    }
                else:
                    cmp_meta = {
                        "Description": (
                            frag_meta['Meta']['Collections']['STR']['Description']
                            + ", "
                            + frag_meta['Meta']['StructuralSystems'][st][
                                'Description'
                            ]
                            + ", "
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Description']
                        ),
                        "Comments": (
                            frag_meta['Meta']['Collections']['STR']['Comment']
                            + "\n"
                            + frag_meta['Meta']['StructuralSystems'][st]['Comment']
                            + "\n"
                            + frag_meta['Meta']['DesignLevels'][
                                convert_design_level[dl]
                            ]['Comment']
                        ),
                        "SuggestedComponentBlockSize": "1 EA",
                        "RoundUpToIntegerQuantity": "True",
                        "LimitStates": {},
                    }

                # store the Limit State parameters
                ds_meta = frag_meta['Meta']['StructuralSystems'][st]['DamageStates']
                for LS_i in range(1, 5):
                    df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
                    df_db.loc[counter, f'LS{LS_i}-Theta_0'] = S_data['EDP_limits'][
                        dl
                    ][bt][LS_i - 1]
                    df_db.loc[counter, f'LS{LS_i}-Theta_1'] = S_data[
                        'Fragility_beta'
                    ][dl]

                    if LS_i == 4:
                        p_coll = S_data['P_collapse'][bt]
                        df_db.loc[counter, f'LS{LS_i}-DamageStateWeights'] = (
                            f'{1.0 - p_coll} | {p_coll}'
                        )

                        cmp_meta["LimitStates"].update(
                            {
                                "LS4": {
                                    "DS4": {"Description": ds_meta['DS4']},
                                    "DS5": {"Description": ds_meta['DS5']},
                                }
                            }
                        )

                    else:
                        cmp_meta["LimitStates"].update(
                            {
                                f"LS{LS_i}": {
                                    f"DS{LS_i}": {
                                        "Description": ds_meta[f"DS{LS_i}"]
                                    }
                                }
                            }
                        )

                # store metadata
                meta_dict.update({cmp_id: cmp_meta})

                counter += 1

    # Second, the non-structural drift sensitive one
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    # create the component id
    df_db.loc[counter, 'ID'] = 'NSD'

    # store demand specifications
    if resolution == 'building':
        df_db.loc[counter, 'Demand-Type'] = "Peak Roof Drift Ratio"
    elif resolution == 'story':
        df_db.loc[counter, 'Demand-Type'] = "Peak Interstory Drift Ratio"

    df_db.loc[counter, 'Demand-Unit'] = "rad"
    df_db.loc[counter, 'Demand-Offset'] = 0

    # add metadata
    cmp_meta = {
        "Description": frag_meta['Meta']['Collections']['NSD']['Description'],
        "Comments": frag_meta['Meta']['Collections']['NSD']['Comment'],
        "SuggestedComponentBlockSize": "1 EA",
        "RoundUpToIntegerQuantity": "True",
        "LimitStates": {},
    }

    # store the Limit State parameters
    ds_meta = frag_meta['Meta']['Collections']['NSD']['DamageStates']
    for LS_i in range(1, 5):
        df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
        df_db.loc[counter, f'LS{LS_i}-Theta_0'] = NSD_data['EDP_limits'][LS_i - 1]
        df_db.loc[counter, f'LS{LS_i}-Theta_1'] = NSD_data['Fragility_beta']

        # add limit state metadata
        cmp_meta["LimitStates"].update(
            {f"LS{LS_i}": {f"DS{LS_i}": {"Description": ds_meta[f"DS{LS_i}"]}}}
        )

    # store metadata
    meta_dict.update({'NSD': cmp_meta})

    counter += 1

    # Third, the non-structural acceleration sensitive fragilities
    NSA_data = raw_data['NonStructural_Acceleration_Sensitive_Fragility_Groups']

    for dl in design_levels:
        # create the component id
        cmp_id = f'NSA.{convert_design_level[dl]}'
        df_db.loc[counter, 'ID'] = cmp_id

        # store demand specifications
        df_db.loc[counter, 'Demand-Type'] = "Peak Floor Acceleration"
        df_db.loc[counter, 'Demand-Unit'] = "g"
        df_db.loc[counter, 'Demand-Offset'] = 0

        # add metadata
        cmp_meta = {
            "Description": (
                frag_meta['Meta']['Collections']['NSA']['Description']
                + ", "
                + frag_meta['Meta']['DesignLevels'][convert_design_level[dl]][
                    'Description'
                ]
            ),
            "Comments": (
                frag_meta['Meta']['Collections']['NSA']['Comment']
                + "\n"
                + frag_meta['Meta']['DesignLevels'][convert_design_level[dl]][
                    'Comment'
                ]
            ),
            "SuggestedComponentBlockSize": "1 EA",
            "RoundUpToIntegerQuantity": "True",
            "LimitStates": {},
        }

        # store the Limit State parameters
        ds_meta = frag_meta['Meta']['Collections']['NSA']['DamageStates']
        for LS_i in range(1, 5):
            df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
            df_db.loc[counter, f'LS{LS_i}-Theta_0'] = NSA_data['EDP_limits'][dl][
                LS_i - 1
            ]
            df_db.loc[counter, f'LS{LS_i}-Theta_1'] = NSA_data['Fragility_beta']

            # add limit state metadata
            cmp_meta["LimitStates"].update(
                {f"LS{LS_i}": {f"DS{LS_i}": {"Description": ds_meta[f"DS{LS_i}"]}}}
            )

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

        counter += 1

    # Fourth, the lifeline facilities - only at the building-level resolution
    if resolution == 'building':
        LF_data = raw_data['Lifeline_Facilities']

        for bt in building_types:
            for dl in design_levels:
                if bt in LF_data['EDP_limits'][dl].keys():
                    # add a dot in bt between structure and height labels, if needed
                    if (len(bt) > 2) and (bt[-1] in {'L', 'M', 'H'}):
                        bt_exp = f'{bt[:-1]}.{bt[-1]}'
                        st = bt[:-1]
                        hc = bt[-1]
                    else:
                        bt_exp = bt
                        st = bt
                        hc = None

                    # create the component id
                    cmp_id = f'LF.{bt_exp}.{convert_design_level[dl]}'
                    df_db.loc[counter, 'ID'] = cmp_id

                    # store demand specifications
                    df_db.loc[counter, 'Demand-Type'] = "Peak Ground Acceleration"
                    df_db.loc[counter, 'Demand-Unit'] = "g"
                    df_db.loc[counter, 'Demand-Offset'] = 0

                    # add metadata
                    if hc is not None:
                        cmp_meta = {
                            "Description": (
                                frag_meta['Meta']['Collections']['LF']['Description']
                                + ", "
                                + frag_meta['Meta']['StructuralSystems'][st][
                                    'Description'
                                ]
                                + ", "
                                + frag_meta['Meta']['HeightClasses'][hc][
                                    'Description'
                                ]
                                + ", "
                                + frag_meta['Meta']['DesignLevels'][
                                    convert_design_level[dl]
                                ]['Description']
                            ),
                            "Comments": (
                                frag_meta['Meta']['Collections']['LF']['Comment']
                                + "\n"
                                + frag_meta['Meta']['StructuralSystems'][st][
                                    'Comment'
                                ]
                                + "\n"
                                + frag_meta['Meta']['HeightClasses'][hc]['Comment']
                                + "\n"
                                + frag_meta['Meta']['DesignLevels'][
                                    convert_design_level[dl]
                                ]['Comment']
                            ),
                            "SuggestedComponentBlockSize": "1 EA",
                            "RoundUpToIntegerQuantity": "True",
                            "LimitStates": {},
                        }
                    else:
                        cmp_meta = {
                            "Description": (
                                frag_meta['Meta']['Collections']['LF']['Description']
                                + ", "
                                + frag_meta['Meta']['StructuralSystems'][st][
                                    'Description'
                                ]
                                + ", "
                                + frag_meta['Meta']['DesignLevels'][
                                    convert_design_level[dl]
                                ]['Description']
                            ),
                            "Comments": (
                                frag_meta['Meta']['Collections']['LF']['Comment']
                                + "\n"
                                + frag_meta['Meta']['StructuralSystems'][st][
                                    'Comment'
                                ]
                                + "\n"
                                + frag_meta['Meta']['DesignLevels'][
                                    convert_design_level[dl]
                                ]['Comment']
                            ),
                            "SuggestedComponentBlockSize": "1 EA",
                            "RoundUpToIntegerQuantity": "True",
                            "LimitStates": {},
                        }

                    # store the Limit State parameters
                    ds_meta = frag_meta['Meta']['StructuralSystems'][st][
                        'DamageStates'
                    ]
                    for LS_i in range(1, 5):
                        df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
                        df_db.loc[counter, f'LS{LS_i}-Theta_0'] = LF_data[
                            'EDP_limits'
                        ][dl][bt][LS_i - 1]
                        df_db.loc[counter, f'LS{LS_i}-Theta_1'] = LF_data[
                            'Fragility_beta'
                        ][dl]

                        if LS_i == 4:
                            p_coll = LF_data['P_collapse'][bt]
                            df_db.loc[counter, f'LS{LS_i}-DamageStateWeights'] = (
                                f'{1.0 - p_coll} | {p_coll}'
                            )

                            cmp_meta["LimitStates"].update(
                                {
                                    "LS4": {
                                        "DS4": {"Description": ds_meta['DS4']},
                                        "DS5": {"Description": ds_meta['DS5']},
                                    }
                                }
                            )

                        else:
                            cmp_meta["LimitStates"].update(
                                {
                                    f"LS{LS_i}": {
                                        f"DS{LS_i}": {
                                            "Description": ds_meta[f"DS{LS_i}"]
                                        }
                                    }
                                }
                            )

                    # store metadata
                    meta_dict.update({cmp_id: cmp_meta})

                    counter += 1

    # Fifth, the ground failure fragilities
    GF_data = raw_data['Ground_Failure']

    for direction in ('Horizontal', 'Vertical'):
        for f_depth in ('Shallow', 'Deep'):
            # create the component id
            cmp_id = f'GF.{direction[0]}.{f_depth[0]}'
            df_db.loc[counter, 'ID'] = cmp_id

            # store demand specifications
            df_db.loc[counter, 'Demand-Type'] = "Permanent Ground Deformation"
            df_db.loc[counter, 'Demand-Unit'] = "inch"
            df_db.loc[counter, 'Demand-Offset'] = 0

            # add metadata
            cmp_meta = {
                "Description": (
                    frag_meta['Meta']['Collections']['GF']['Description']
                    + f", {direction} Direction, {f_depth} Foundation"
                ),
                "Comments": (frag_meta['Meta']['Collections']['GF']['Comment']),
                "SuggestedComponentBlockSize": "1 EA",
                "RoundUpToIntegerQuantity": "True",
                "LimitStates": {},
            }

            # store the Limit State parameters
            ds_meta = frag_meta['Meta']['Collections']['GF']['DamageStates']

            df_db.loc[counter, 'LS1-Family'] = 'lognormal'
            df_db.loc[counter, 'LS1-Theta_0'] = GF_data['EDP_limits'][direction][
                f_depth
            ]
            df_db.loc[counter, 'LS1-Theta_1'] = GF_data['Fragility_beta'][direction][
                f_depth
            ]
            p_complete = GF_data['P_Complete']
            df_db.loc[counter, 'LS1-DamageStateWeights'] = (
                f'{1.0 - p_complete} | {p_complete}'
            )

            cmp_meta["LimitStates"].update(
                {
                    "LS1": {
                        "DS1": {"Description": ds_meta['DS1']},
                        "DS2": {"Description": ds_meta['DS2']},
                    }
                }
            )

            # store metadata
            meta_dict.update({cmp_id: cmp_meta})

            counter += 1

    # remove empty rows (from the end)
    df_db.dropna(how='all', inplace=True)

    # All Hazus components have complete fragility info,
    df_db['Incomplete'] = 0

    # none of them are directional,
    df_db['Demand-Directional'] = 0

    # rename the index
    df_db.set_index("ID", inplace=True)

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the fragility data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the fragility data from Hazus EQ")


def create_Hazus_EQ_repair_db(
    source_file,
    meta_file='',
    target_data_file='loss_repair_DB_Hazus_EQ_bldg.csv',
    target_meta_file='loss_repair_DB_Hazus_EQ_bldg.json',
    resolution='building',
):
    """
    Create a database file based on the HAZUS EQ Technical Manual

    This method was developed to process a json file with tabulated data from
    v4.2.3 of the Hazus Earthquake Technical Manual. The json file is included
    under data_sources in the SimCenter DB_DamageAndLoss repo on GitHub.

    Parameters
    ----------
    source_file: string
        Path to the Hazus database file.
    meta_file: string
        Path to the JSON file with metadata about the database.
    target_data_file: string
        Path where the repair DB file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the repair DB metadata should be saved. A json file is
        expected.
    resolution: string
        If building, the function produces the conventional Hazus
        fragilities. If story, the function produces story-level
        fragilities.

    """

    # adjust the target filenames if needed
    if resolution == 'story':
        target_data_file = target_data_file.replace('bldg', 'story')
        target_meta_file = target_meta_file.replace('bldg', 'story')

    # parse the source file
    with open(source_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # parse the extra metadata file
    if Path(meta_file).is_file():
        with open(meta_file, 'r', encoding='utf-8') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = {}

    # prepare lists of labels for various building features
    occupancies = list(raw_data['Structural_Fragility_Groups']['Repair_cost'].keys())

    # initialize the output loss table
    # define the columns
    out_cols = [
        "Incomplete",
        "Quantity-Unit",
        "DV-Unit",
    ]
    for DS_i in range(1, 6):
        out_cols += [
            f"DS{DS_i}-Theta_0",
        ]

    # create the MultiIndex
    cmp_types = ['STR', 'NSD', 'NSA', 'LF']
    comps = [
        f'{cmp_type}.{occ_type}'
        for cmp_type in cmp_types
        for occ_type in occupancies
    ]
    DVs = ['Cost', 'Time']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])

    df_db = pd.DataFrame(columns=out_cols, index=df_MI, dtype=float)

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    # add the general information to the meta dict
    if "_GeneralInformation" in frag_meta.keys():
        GI = frag_meta["_GeneralInformation"]

        for key, item in deepcopy(GI).items():
            if key == 'ComponentGroups_Loss_Repair':
                GI.update({'ComponentGroups': item})

            if key.startswith('ComponentGroups'):
                GI.pop(key, None)

        meta_dict.update({"_GeneralInformation": GI})

    # First, prepare the structural damage consequences
    S_data = raw_data['Structural_Fragility_Groups']

    for occ_type in occupancies:
        # create the component id
        cmp_id = f'STR.{occ_type}'

        cmp_meta = {
            "Description": (
                frag_meta['Meta']['Collections']['STR']['Description']
                + ", "
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
            ),
            "Comments": (
                frag_meta['Meta']['Collections']['STR']['Comment']
                + "\n"
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
            ),
            "SuggestedComponentBlockSize": "1 EA",
            "RoundUpToIntegerQuantity": "True",
            "DamageStates": {},
        }

        # store the consequence values for each Damage State
        ds_meta = frag_meta['Meta']['Collections']['STR']['DamageStates']
        for DS_i in range(1, 6):
            cmp_meta["DamageStates"].update(
                {f"DS{DS_i}": {"Description": ds_meta[f"DS{DS_i}"]}}
            )

            # DS4 and DS5 have identical repair consequences
            if DS_i == 5:
                ds_i = 4
            else:
                ds_i = DS_i

            # Convert percentage to ratio.
            df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                f"{S_data['Repair_cost'][occ_type][ds_i - 1] / 100.00:.3f}"
            )

            df_db.loc[(cmp_id, 'Time'), f'DS{DS_i}-Theta_0'] = S_data['Repair_time'][
                occ_type
            ][ds_i - 1]

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

    # Second, the non-structural drift sensitive one
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    for occ_type in occupancies:
        # create the component id
        cmp_id = f'NSD.{occ_type}'

        cmp_meta = {
            "Description": (
                frag_meta['Meta']['Collections']['NSD']['Description']
                + ", "
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
            ),
            "Comments": (
                frag_meta['Meta']['Collections']['NSD']['Comment']
                + "\n"
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
            ),
            "SuggestedComponentBlockSize": "1 EA",
            "RoundUpToIntegerQuantity": "True",
            "DamageStates": {},
        }

        # store the consequence values for each Damage State
        ds_meta = frag_meta['Meta']['Collections']['NSD']['DamageStates']
        for DS_i in range(1, 5):
            cmp_meta["DamageStates"].update(
                {f"DS{DS_i}": {"Description": ds_meta[f"DS{DS_i}"]}}
            )

            # Convert percentage to ratio.
            df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                f"{NSD_data['Repair_cost'][occ_type][DS_i - 1] / 100.00:.3f}"
            )

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

    # Third, the non-structural acceleration sensitive fragilities
    NSA_data = raw_data['NonStructural_Acceleration_Sensitive_Fragility_Groups']

    for occ_type in occupancies:
        # create the component id
        cmp_id = f'NSA.{occ_type}'

        cmp_meta = {
            "Description": (
                frag_meta['Meta']['Collections']['NSA']['Description']
                + ", "
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
            ),
            "Comments": (
                frag_meta['Meta']['Collections']['NSA']['Comment']
                + "\n"
                + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
            ),
            "SuggestedComponentBlockSize": "1 EA",
            "RoundUpToIntegerQuantity": "True",
            "DamageStates": {},
        }

        # store the consequence values for each Damage State
        ds_meta = frag_meta['Meta']['Collections']['NSA']['DamageStates']
        for DS_i in range(1, 5):
            cmp_meta["DamageStates"].update(
                {f"DS{DS_i}": {"Description": ds_meta[f"DS{DS_i}"]}}
            )

            # Convert percentage to ratio.
            df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                f"{NSA_data['Repair_cost'][occ_type][DS_i - 1] / 100.00:.3f}"
            )

        # store metadata
        meta_dict.update({cmp_id: cmp_meta})

    # Fourth, the lifeline facilities - only at the building-level resolution
    if resolution == 'building':
        LF_data = raw_data['Lifeline_Facilities']

        for occ_type in occupancies:
            # create the component id
            cmp_id = f'LF.{occ_type}'

            cmp_meta = {
                "Description": (
                    frag_meta['Meta']['Collections']['LF']['Description']
                    + ", "
                    + frag_meta['Meta']['OccupancyTypes'][occ_type]['Description']
                ),
                "Comments": (
                    frag_meta['Meta']['Collections']['LF']['Comment']
                    + "\n"
                    + frag_meta['Meta']['OccupancyTypes'][occ_type]['Comment']
                ),
                "SuggestedComponentBlockSize": "1 EA",
                "RoundUpToIntegerQuantity": "True",
                "DamageStates": {},
            }

            # store the consequence values for each Damage State
            ds_meta = frag_meta['Meta']['Collections']['LF']['DamageStates']
            for DS_i in range(1, 6):
                # DS4 and DS5 have identical repair consequences
                if DS_i == 5:
                    ds_i = 4
                else:
                    ds_i = DS_i

                cmp_meta["DamageStates"].update(
                    {f"DS{DS_i}": {"Description": ds_meta[f"DS{DS_i}"]}}
                )

                # Convert percentage to ratio.
                df_db.loc[(cmp_id, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                    f"{LF_data['Repair_cost'][occ_type][ds_i - 1] / 100.00:.3f}"
                )

                df_db.loc[(cmp_id, 'Time'), f'DS{DS_i}-Theta_0'] = LF_data[
                    'Repair_time'
                ][occ_type][ds_i - 1]

            # store metadata
            meta_dict.update({cmp_id: cmp_meta})

    # remove empty rows (from the end)
    df_db.dropna(how='all', inplace=True)

    # All Hazus components have complete fragility info,
    df_db['Incomplete'] = 0
    # df_db.loc[:, 'Incomplete'] = 0

    # The damage quantity unit is the same for all consequence values
    df_db.loc[:, 'Quantity-Unit'] = "1 EA"

    # The output units are also identical among all components
    df_db.loc[idx[:, 'Cost'], 'DV-Unit'] = "loss_ratio"
    df_db.loc[idx[:, 'Time'], 'DV-Unit'] = "day"

    # convert to simple index
    df_db = base.convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata - later
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the repair consequence data from Hazus EQ")


def create_Hazus_HU_fragility_db(
    source_file: str = (
        'pelicun/resources/SimCenterDBDL/' 'damage_DB_SimCenter_Hazus_HU_bldg.csv'
    ),
    meta_file: str = (
        'pelicun/resources/SimCenterDBDL/'
        'damage_DB_SimCenter_Hazus_HU_bldg_template.json'
    ),
    target_meta_file: str = 'damage_DB_SimCenter_Hazus_HU_bldg.json',
) -> None:
    """
    Create a database metadata file for the HAZUS Hurricane fragilities.

    This method was developed to add a json file with metadata
    accompanying `damage_DB_SimCenter_Hazus_HU_bldg.csv`. That file
    contains fragility curves fitted to Hazus Hurricane data relaetd
    to the Hazus Hurricane Technical Manual v4.2.

    Parameters
    ----------
    source_file: string
        Path to the Hazus Hurricane fragility data.
    meta_file: string
        Path to a predefined fragility metadata file.
    target_meta_file: string
        Path where the fragility metadata should be saved. A json file is
        expected.

    """

    fragility_data = pd.read_csv(source_file)

    with open(meta_file, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    # retrieve damage state descriptions and remove that part from
    # `hazus_hu_metadata`
    damage_state_classes = meta_dict.pop('DamageStateClasses')
    damage_state_descriptions = meta_dict.pop('DamageStateDescriptions')

    # Procedure Overview:
    # (1) We define several dictionaries mapping chunks of the
    # composite asset ID (the parts between periods) to human-readable
    # (`-h` for short) representations.
    # (2) We define -h asset type descriptions and map them to the
    # first-most relevant ID chunks (`primary chunks`)
    # (3) We map asset class codes with general asset classes
    # (4) We define the required dictionaries from (1) that decode the
    # ID chunks after the `primary chunks` for each general asset
    # class
    # (5) We decode:
    # ID -> asset class -> general asset class -> dictionaries
    # -> ID turns to -h text by combining the description of the asset class
    # from the `primary chunks` and the decoded description of the
    # following chunks using the dictionaries.

    #
    # (1) Dictionaries
    #

    roof_shape = {
        'flt': 'Flat roof.',
        'gab': 'Gable roof.',
        'hip': 'Hip roof.',
    }

    secondary_water_resistance = {
        '1': 'Secondary water resistance.',
        '0': 'No secondary water resistance.',
        'null': 'No information on secondary water resistance.',
    }

    roof_deck_attachment = {
        '6d': '6d roof deck nails.',
        '6s': '6s roof deck nails.',
        '8d': '8d roof deck nails.',
        '8s': '8s roof deck nails.',
        'st': 'Standard roof deck attachment.',
        'su': 'Superior roof deck attachment.',
        'null': 'Missing roof deck attachment information.',
    }

    roof_wall_connection = {
        'tnail': 'Roof-to-wall toe nails.',
        'strap': 'Roof-to-wall straps.',
        'null': 'Missing roof-to-wall connection information.',
    }

    garage_presence = {
        'no': 'No garage.',
        'wkd': 'Weak garage door.',
        'std': 'Standard garage door.',
        'sup': 'Strong garage door.',
        'null': 'No information on garage.',
    }

    shutters = {'1': 'Has Shutters.', '0': 'No shutters.'}

    roof_cover = {
        'bur': 'Built-up roof cover.',
        'spm': 'Single-ply membrane roof cover.',
        'smtl': 'Sheet metal roof cover.',
        'cshl': 'Shingle roof cover.',
        'null': 'No information on roof cover.',
    }

    roof_quality = {
        'god': 'Good roof quality.',
        'por': 'Poor roof quality.',
        'null': 'No information on roof quality.',
    }

    masonry_reinforcing = {
        '1': 'Has masonry reinforcing.',
        '0': 'No masonry reinforcing.',
        'null': 'Unknown information on masonry reinfocing.',
    }

    roof_frame_type = {
        'trs': 'Wood truss roof frame.',
        'ows': 'OWSJ roof frame.',
    }

    wind_debris_environment = {
        'A': 'Residentiao/commercial wind debris environment.',
        'B': 'Wind debris environment varies by direction.',
        'C': 'Residential wind debris environment.',
        'D': 'No wind debris environment.',
    }

    roof_deck_age = {
        'god': 'New or average roof age.',
        'por': 'Old roof age.',
        'null': 'Missing roof age information.',
    }

    roof_metal_deck_attachment_quality = {
        'std': 'Standard metal deck roof attachment.',
        'sup': 'Superior metal deck roof attachment.',
        'null': 'Missing roof attachment quality information.',
    }

    number_of_units = {
        'sgl': 'Single unit.',
        'mlt': 'Multi-unit.',
        'null': 'Unknown number of units.',
    }

    joist_spacing = {
        '4': '4 ft joist spacing.',
        '6': '6 ft foot joist spacing.',
        'null': 'Unknown joist spacing.',
    }

    window_area = {
        'low': 'Low window area.',
        'med': 'Medium window area.',
        'hig': 'High window area.',
    }

    tie_downs = {'1': 'Tie downs.', '0': 'No tie downs.'}

    terrain_surface_roughness = {
        '3': 'Terrain surface roughness: 0.03 m.',
        '15': 'Terrain surface roughness: 0.15 m.',
        '35': 'Terrain surface roughness: 0.35 m.',
        '70': 'Terrain surface roughness: 0.7 m.',
        '100': 'Terrain surface roughness: 1 m.',
    }

    #
    # (2) Asset type descriptions
    #

    # maps class type code to -h description
    class_types = {
        # ------------------------
        'W.SF.1': 'Wood, Single-family, One-story.',
        'W.SF.2': 'Wood, Single-family, Two or More Stories.',
        # ------------------------
        'W.MUH.1': 'Wood, Multi-Unit Housing, One-story.',
        'W.MUH.2': 'Wood, Multi-Unit Housing, Two Stories.',
        'W.MUH.3': 'Wood, Multi-Unit Housing, Three or More Stories.',
        # ------------------------
        'M.SF.1': 'Masonry, Single-family, One-story.',
        'M.SF.2': 'Masonry, Single-family, Two or More Stories.',
        # ------------------------
        'M.MUH.1': 'Masonry, Multi-Unit Housing, One-story.',
        'M.MUH.2': 'Masonry, Multi-Unit Housing, Two Stories.',
        'M.MUH.3': 'Masonry, Multi-Unit Housing, Three or More Stories.',
        # ------------------------
        'M.LRM.1': 'Masonry, Low-Rise Strip Mall, Up to 15 Feet.',
        'M.LRM.2': 'Masonry, Low-Rise Strip Mall, More than 15 Feet.',
        # ------------------------
        'M.LRI': 'Masonry, Low-Rise Industrial/Warehouse/Factory Buildings.',
        # ------------------------
        'M.ERB.L': (
            'Masonry, Engineered Residential Building, Low-Rise (1-2 Stories).'
        ),
        'M.ERB.M': (
            'Masonry, Engineered Residential Building, Mid-Rise (3-5 Stories).'
        ),
        'M.ERB.H': (
            'Masonry, Engineered Residential Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'M.ECB.L': (
            'Masonry, Engineered Commercial Building, Low-Rise (1-2 Stories).'
        ),
        'M.ECB.M': (
            'Masonry, Engineered Commercial Building, Mid-Rise (3-5 Stories).'
        ),
        'M.ECB.H': (
            'Masonry, Engineered Commercial Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'C.ERB.L': (
            'Concrete, Engineered Residential Building, Low-Rise (1-2 Stories).'
        ),
        'C.ERB.M': (
            'Concrete, Engineered Residential Building, Mid-Rise (3-5 Stories).'
        ),
        'C.ERB.H': (
            'Concrete, Engineered Residential Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'C.ECB.L': (
            'Concrete, Engineered Commercial Building, Low-Rise (1-2 Stories).'
        ),
        'C.ECB.M': (
            'Concrete, Engineered Commercial Building, Mid-Rise (3-5 Stories).'
        ),
        'C.ECB.H': (
            'Concrete, Engineered Commercial Building, High-Rise (6+ Stories).'
        ),
        # ------------------------
        'S.PMB.S': 'Steel, Pre-Engineered Metal Building, Small.',
        'S.PMB.M': 'Steel, Pre-Engineered Metal Building, Medium.',
        'S.PMB.L': 'Steel, Pre-Engineered Metal Building, Large.',
        # ------------------------
        'S.ERB.L': 'Steel, Engineered Residential Building, Low-Rise (1-2 Stories).',
        'S.ERB.M': 'Steel, Engineered Residential Building, Mid-Rise (3-5 Stories).',
        'S.ERB.H': 'Steel, Engineered Residential Building, High-Rise (6+ Stories).',
        # ------------------------
        'S.ECB.L': 'Steel, Engineered Commercial Building, Low-Rise (1-2 Stories).',
        'S.ECB.M': 'Steel, Engineered Commercial Building, Mid-Rise (3-5 Stories).',
        'S.ECB.H': 'Steel, Engineered Commercial Building, High-Rise (6+ Stories).',
        # ------------------------
        'MH.PHUD': 'Manufactured Home, Pre-Housing and Urban Development (HUD).',
        'MH.76HUD': 'Manufactured Home, 1976 HUD.',
        'MH.94HUDI': 'Manufactured Home, 1994 HUD - Wind Zone I.',
        'MH.94HUDII': 'Manufactured Home, 1994 HUD - Wind Zone II.',
        'MH.94HUDIII': 'Manufactured Home, 1994 HUD - Wind Zone III.',
        # ------------------------
        'HUEF.H.S': 'Small Hospital, Hospital with fewer than 50 Beds.',
        'HUEF.H.M': 'Medium Hospital, Hospital with beds between 50 & 150.',
        'HUEF.H.L': 'Large Hospital, Hospital with more than 150 Beds.',
        # ------------------------
        'HUEF.S.S': 'Elementary School.',
        'HUEF.S.M': 'High school, two-story.',
        'HUEF.S.L': 'Large high school, three-story.',
        # ------------------------
        'HUEF.EO': 'Emergency Operation Centers.',
        'HUEF.FS': 'Fire Station.',
        'HUEF.PS': 'Police Station.',
        # ------------------------
    }

    def find_class_type(entry: str) -> str | None:
        """
        Find the class type code from an entry string based on
        predefined patterns.

        Parameters
        ----------
        entry : str
            A string representing the entry, consisting of delimited
            segments that correspond to various attributes of an
            asset.

        Returns
        -------
        str or None
            The class type code if a matching pattern is found;
            otherwise, None if no pattern matches the input string.

        """
        entry_elements = entry.split('.')
        for nper in range(1, len(entry_elements)):
            first_parts = '.'.join(entry_elements[:nper])
            if first_parts in class_types:
                return first_parts
        return None

    #
    # (3) General asset class
    #

    # maps class code type to general class code
    general_classes = {
        # ------------------------
        'W.SF.1': 'WSF',
        'W.SF.2': 'WSF',
        # ------------------------
        'W.MUH.1': 'WMUH',
        'W.MUH.2': 'WMUH',
        'W.MUH.3': 'WMUH',
        # ------------------------
        'M.SF.1': 'MSF',
        'M.SF.2': 'MSF',
        # ------------------------
        'M.MUH.1': 'MMUH',
        'M.MUH.2': 'MMUH',
        'M.MUH.3': 'MMUH',
        # ------------------------
        'M.LRM.1': 'MLRM1',
        'M.LRM.2': 'MLRM2',
        # ------------------------
        'M.LRI': 'MLRI',
        # ------------------------
        'M.ERB.L': 'MERB',
        'M.ERB.M': 'MERB',
        'M.ERB.H': 'MERB',
        # ------------------------
        'M.ECB.L': 'MECB',
        'M.ECB.M': 'MECB',
        'M.ECB.H': 'MECB',
        # ------------------------
        'C.ERB.L': 'CERB',
        'C.ERB.M': 'CERB',
        'C.ERB.H': 'CERB',
        # ------------------------
        'C.ECB.L': 'CECB',
        'C.ECB.M': 'CECB',
        'C.ECB.H': 'CECB',
        # ------------------------
        'S.PMB.S': 'SPMB',
        'S.PMB.M': 'SPMB',
        'S.PMB.L': 'SPMB',
        # ------------------------
        'S.ERB.L': 'SERB',
        'S.ERB.M': 'SERB',
        'S.ERB.H': 'SERB',
        # ------------------------
        'S.ECB.L': 'SECB',
        'S.ECB.M': 'SECB',
        'S.ECB.H': 'SECB',
        # ------------------------
        'MH.PHUD': 'MH',
        'MH.76HUD': 'MH',
        'MH.94HUDI': 'MH',
        'MH.94HUDII': 'MH',
        'MH.94HUDIII': 'MH',
        # ------------------------
        'HUEF.H.S': 'HUEFH',
        'HUEF.H.M': 'HUEFH',
        'HUEF.H.L': 'HUEFH',
        # ------------------------
        'HUEF.S.S': 'HUEFS',
        'HUEF.S.M': 'HUEFS',
        'HUEF.S.L': 'HUEFS',
        # ------------------------
        'HUEF.EO': 'HUEFEO',
        'HUEF.FS': 'HUEFFS',
        'HUEF.PS': 'HUEFPS',
        # ------------------------
    }

    #
    # (4) Relevant dictionaries
    #

    # maps general class code to list of dicts where the -h attribute
    # descriptions will be pulled from
    dictionaries_of_interest = {
        'WSF': [
            roof_shape,
            secondary_water_resistance,
            roof_deck_attachment,
            roof_wall_connection,
            garage_presence,
            shutters,
            terrain_surface_roughness,
        ],
        'WMUH': [
            roof_shape,
            roof_cover,
            roof_quality,
            secondary_water_resistance,
            roof_deck_attachment,
            roof_wall_connection,
            shutters,
            terrain_surface_roughness,
        ],
        'MSF': [
            roof_shape,
            roof_wall_connection,
            roof_frame_type,
            roof_deck_attachment,
            shutters,
            secondary_water_resistance,
            garage_presence,
            masonry_reinforcing,
            roof_cover,
            terrain_surface_roughness,
        ],
        'MMUH': [
            roof_shape,
            secondary_water_resistance,
            roof_cover,
            roof_quality,
            roof_deck_attachment,
            roof_wall_connection,
            shutters,
            masonry_reinforcing,
            terrain_surface_roughness,
        ],
        'MLRM1': [
            roof_cover,
            shutters,
            masonry_reinforcing,
            wind_debris_environment,
            roof_frame_type,
            roof_deck_attachment,
            roof_wall_connection,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'MLRM2': [
            roof_cover,
            shutters,
            masonry_reinforcing,
            wind_debris_environment,
            roof_frame_type,
            roof_deck_attachment,
            roof_wall_connection,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            number_of_units,
            joist_spacing,
            terrain_surface_roughness,
        ],
        'MLRI': [
            shutters,
            masonry_reinforcing,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'MERB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'MECB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'CERB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            window_area,
            terrain_surface_roughness,
        ],
        'CECB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            window_area,
            terrain_surface_roughness,
        ],
        'SPMB': [
            shutters,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'SERB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'SECB': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'MH': [shutters, tie_downs, terrain_surface_roughness],
        'HUEFH': [
            roof_cover,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            shutters,
            terrain_surface_roughness,
        ],
        'HUEFS': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'HUEFEO': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
        'HUEFFS': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_deck_age,
            roof_metal_deck_attachment_quality,
            terrain_surface_roughness,
        ],
        'HUEFPS': [
            roof_cover,
            shutters,
            wind_debris_environment,
            roof_metal_deck_attachment_quality,
            window_area,
            terrain_surface_roughness,
        ],
    }

    #
    # (5) Decode IDs and extend metadata with the individual records
    #

    for fragility_id in fragility_data['ID'].to_list():
        class_type = find_class_type(fragility_id)
        assert class_type is not None

        class_type_human_readable = class_types[class_type]

        general_class = general_classes[class_type]
        dictionaries = dictionaries_of_interest[general_class]
        remaining_chunks = fragility_id.replace(f'{class_type}.', '').split('.')
        assert len(remaining_chunks) == len(dictionaries)
        human_description = [class_type_human_readable]
        for chunk, dictionary in zip(remaining_chunks, dictionaries):
            human_description.append(dictionary[chunk])
        human_description_str = ' '.join(human_description)

        damage_state_class = damage_state_classes[class_type]
        damage_state_description = damage_state_descriptions[damage_state_class]

        limit_states = {}
        for damage_state, description in damage_state_description.items():
            limit_state = damage_state.replace('DS', 'LS')
            limit_states[limit_state] = {damage_state: description}

        record = {
            'Description': human_description_str,
            'SuggestedComponentBlockSize': '1 EA',
            'RoundUpToIntegerQuantity': 'True',
            'LimitStates': limit_states,
        }

        meta_dict[fragility_id] = record

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)
