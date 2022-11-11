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
    create_FEMA_P58_bldg_repair_db
    create_FEMA_P58_bldg_injury_db
    create_FEMA_P58_bldg_redtag_db

    create_Hazus_EQ_fragility_db
    create_Hazus_EQ_bldg_repair_db
    create_Hazus_EQ_bldg_injury_db

"""

import re
import json
import numpy as np
import pandas as pd
from . import base
from .uq import fit_distribution_to_percentiles

idx = base.idx


def parse_DS_Hierarchy(DSH):
    """
    Parses the FEMA P58 DS hierarchy into a set of arrays.
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
            subDSH = DSH[:closing_pos + 1]
            DSH = DSH[closing_pos + 2:]

            DS_setup.append([subDSH[:5]] + subDSH[6:-1].split(','))

    return DS_setup


def create_FEMA_P58_fragility_db(source_file,
                                 target_data_file='fragility_DB_FEMA_P58_2nd.csv',
                                 target_meta_file='fragility_DB_FEMA_P58_2nd.json'):
    """
    Create a fragility parameter database based on the FEMA P58 data

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
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
    df = pd.read_excel(source_file, sheet_name='Summary', header=2, index_col=1,
                       true_values=["YES", "Yes", "yes"],
                       false_values=["NO", "No", "no"])

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
            "LS4-DamageStateWeights"
        ],
        index=df_db_source.index,
        dtype=float
    )

    # initialize the dictionary that stores the fragility metadata
    meta_dict = {}

    # conversion dictionary for demand types
    convert_demand_type = {
        'Story Drift Ratio': "Peak Interstory Drift Ratio",
        'Link Rotation Angle': "Peak Link Rotation Angle",
        'Effective Drift': "Peak Effective Drift Ratio",
        'Link Beam Chord Rotation': "Peak Link Beam Chord Rotation",
        'Peak Floor Acceleration': "Peak Floor Acceleration",
        'Peak Floor Velocity': "Peak Floor Velocity"
    }

    # conversion dictionary for demand unit names
    convert_demand_unit = {
        'Unit less': 'ea',
        'Radians': 'rad',
        'g': 'g',
        'meter/sec': 'mps'
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
        df_db.loc[cmp.Index, 'Demand-Type'] = (
            convert_demand_type[cmp.Demand_Parameter_value])
        df_db.loc[cmp.Index, 'Demand-Unit'] = (
            convert_demand_unit[cmp.Demand_Parameter_unit])
        df_db.loc[cmp.Index, 'Demand-Offset'] = (
            int(cmp.Demand_Location_use_floor_above_YesNo))
        df_db.loc[cmp.Index, 'Demand-Directional'] = (
            int(cmp.Directional))

        # parse the damage state hierarchy
        DS_setup = parse_DS_Hierarchy(cmp.DS_Hierarchy)

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # every component is assumed to have a comp. description
        comments = cmp_meta['Component_Description']

        # the additional fields are added to the description if they exist

        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += f'\nConstruction Quality: ' \
                        f'{cmp_meta["Construction_Quality"]}'

        if cmp_meta['Seismic_Installation_Conditions'] not in [
                'Not Specified', 'Not applicable', 'Unknown', 'Any']:
            comments += f'\nSeismic Installation Conditions: ' \
                        f'{cmp_meta["Seismic_Installation_Conditions"]}'

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
            "LimitStates": {}
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
                    median_demands.append(
                        getattr(cmp, f"DS_{ds[2]}_Median_Demand"))

                    dispersions.append(
                        getattr(cmp, f"DS_{ds[2]}_Total_Dispersion_Beta"))

                    weights.append(getattr(cmp, f"DS_{ds[2]}_Probability"))

                # make sure the specified distribution parameters are appropriate
                if ((np.unique(median_demands).size != 1) or (
                        np.unique(dispersions).size != 1)):
                    raise ValueError(f"Incorrect mutually exclusive DS "
                                     f"definition in component {cmp.Index} at "
                                     f"Limit State {LS_i}")

                if LS_contents[0] == 'MutEx':

                    # in mutually exclusive cases, make sure the specified DS
                    # weights sum up to one
                    np.testing.assert_allclose(
                        np.sum(np.array(weights, dtype=float)), 1.0,
                        err_msg=f"Mutually exclusive Damage State weights do "
                                f"not sum to 1.0 in component {cmp.Index} at "
                                f"Limit State {LS_i}")

                    # and save all DS metadata under this Limit State
                    for ds in LS_contents[1:]:
                        ds_id = ds[2]

                        ls_meta.update({f"DS{ds_id}": {
                            "Description": cmp_meta[f"DS_{ds_id}_Description"],
                            "RepairAction": cmp_meta[
                                f"DS_{ds_id}_Repair_Description"]
                        }})

                else:
                    # in simultaneous cases, convert simultaneous weights into
                    # mutexc weights
                    sim_ds_count = len(LS_contents) - 1
                    ds_count = 2 ** (sim_ds_count) - 1

                    sim_weights = []

                    for ds_id in range(1, ds_count + 1):
                        ds_map = format(ds_id, f'0{sim_ds_count}b')

                        sim_weights.append(np.product(
                            [weights[ds_i]
                             if ds_map[-ds_i - 1] == '1' else 1.0-weights[ds_i]
                             for ds_i in range(sim_ds_count)]))

                        # save ds metadata - we need to be clever here
                        # the original metadata is saved for the pure cases
                        # when only one DS is triggered
                        # all other DSs store information about which
                        # combination of pure DSs they represent

                        if ds_map.count('1') == 1:

                            ds_pure_id = ds_map[::-1].find('1') + 1

                            ls_meta.update({f"DS{ds_id}": {
                                "Description": f"Pure DS{ds_pure_id}. " +
                                cmp_meta[f"DS_{ds_pure_id}_Description"],
                                "RepairAction": cmp_meta[
                                    f"DS_{ds_pure_id}_Repair_Description"]
                            }})

                        else:

                            ds_combo = [f'DS{_.start() + 1}'
                                        for _ in re.finditer('1', ds_map[::-1])]

                            ls_meta.update({f"DS{ds_id}": {
                                "Description": 'Combination of ' +
                                               ' & '.join(ds_combo),
                                "RepairAction": 'Combination of pure DS repair '
                                                'actions.'
                            }})

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

                ls_meta.update({f"DS{ds_id}": {
                    "Description": cmp_meta[f"DS_{ds_id}_Description"],
                    "RepairAction": cmp_meta[f"DS_{ds_id}_Repair_Description"]
                }})

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


def create_FEMA_P58_bldg_repair_db(
        source_file,
        target_data_file='bldg_repair_DB_FEMA_P58_2nd.csv',
        target_meta_file='bldg_repair_DB_FEMA_P58_2nd.json'):
    """
    Create a repair consequence parameter database based on the FEMA P58 data

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    target_data_file: string
        Path where the consequence data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the consequence metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    df = pd.concat(
        [pd.read_excel(source_file, sheet_name=sheet, header=2, index_col=1)
         for sheet in ('Summary', 'Cost Summary')], axis=1)

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

            f"Best Fit, DS{DS_i}.1",
            f"Lower Qty Mean, DS{DS_i}.1",
            f"Upper Qty Mean, DS{DS_i}.1",
            f"Lower Qty Cutoff, DS{DS_i}.1",
            f"Upper Qty Cutoff, DS{DS_i}.1",
            f"CV / Dispersion, DS{DS_i}.2",
            f"DS {DS_i}, Long Lead Time",

            f'Repair Cost, p10, DS{DS_i}',
            f'Repair Cost, p50, DS{DS_i}',
            f'Repair Cost, p90, DS{DS_i}',
            f'Time, p10, DS{DS_i}',
            f'Time, p50, DS{DS_i}',
            f'Time, p90, DS{DS_i}',
            f'Mean Value, DS{DS_i}',
            f'Mean Value, DS{DS_i}.1',
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
    DVs = ['Cost', 'Time']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])

    df_db = pd.DataFrame(
        columns=out_cols,
        index=df_MI,
        dtype=float
    )

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    convert_family = {
        'LogNormal': 'lognormal',
        'Normal': 'normal'
    }

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

        # store units

        df_db.loc[cmp.Index, 'Quantity-Unit'] = (
            ' '.join(cmp.Fragility_Unit_of_Measure.split(' ')[::-1]).strip())
        df_db.loc[(cmp.Index, 'Cost'), 'DV-Unit'] = "USD_2011"
        df_db.loc[(cmp.Index, 'Time'), 'DV-Unit'] = "worker_day"

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # every component is assumed to have a comp. description
        comments = cmp_meta['Component_Description']

        # the additional fields are added to the description if they exist
        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += f'\nConstruction Quality: ' \
                        f'{cmp_meta["Construction_Quality"]}'

        if cmp_meta['Seismic_Installation_Conditions'] not in [
                'Not Specified', 'Not applicable', 'Unknown', 'Any']:
            comments += f'\nSeismic Installation Conditions: ' \
                        f'{cmp_meta["Seismic_Installation_Conditions"]}'

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
            "DamageStates": {}
        }

        # Handle components with simultaneous damage states separately
        if 'Simul' in cmp.DS_Hierarchy:

            # Note that we are assuming that all damage states are triggered by
            # a single limit state in these components.
            # This assumption holds for the second edition of FEMA P58, but it
            # might need to be revisited in future editions.

            cost_est = {}
            time_est = {}

            # get the p10, p50, and p90 estimates for all damage states
            for DS_i in range(1, 6):

                if not pd.isna(getattr(cmp, f'Repair_Cost_p10_DS{DS_i}')):

                    cost_est.update({f'DS{DS_i}': np.array([
                        getattr(cmp, f'Repair_Cost_p10_DS{DS_i}'),
                        getattr(cmp, f'Repair_Cost_p50_DS{DS_i}'),
                        getattr(cmp, f'Repair_Cost_p90_DS{DS_i}'),
                        getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}'),
                        getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}')
                    ])})

                    time_est.update({f'DS{DS_i}': np.array([
                        getattr(cmp, f'Time_p10_DS{DS_i}'),
                        getattr(cmp, f'Time_p50_DS{DS_i}'),
                        getattr(cmp, f'Time_p90_DS{DS_i}'),
                        getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1'),
                        getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}_1'),
                        int(getattr(cmp, f'DS_{DS_i}_Long_Lead_Time') == 'YES')
                    ])})

            # now prepare the equivalent mutex damage states
            sim_ds_count = len(cost_est.keys())
            ds_count = 2 ** (sim_ds_count) - 1

            for DS_i in range(1, ds_count + 1):
                ds_map = format(DS_i, f'0{sim_ds_count}b')

                cost_vals = np.sum([cost_est[f'DS{ds_i + 1}']
                                    if ds_map[-ds_i - 1] == '1' else np.zeros(5)
                                    for ds_i in range(sim_ds_count)],
                                   axis=0)

                time_vals = np.sum([time_est[f'DS{ds_i + 1}']
                                    if ds_map[-ds_i - 1] == '1' else np.zeros(6)
                                    for ds_i in range(sim_ds_count)],
                                   axis=0)

                # fit a distribution
                family_hat, theta_hat = fit_distribution_to_percentiles(
                    cost_vals[:3], [0.1, 0.5, 0.9], ['normal', 'lognormal'])

                cost_theta = theta_hat
                if family_hat == 'normal':
                    cost_theta[1] = cost_theta[1] / cost_theta[0]

                time_theta = [time_vals[1],
                              np.sqrt(cost_theta[1] ** 2.0 + 0.25 ** 2.0)]

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
                    f"{cost_qnt_low:g},{cost_qnt_up:g}")

                df_db.loc[(cmp.Index, 'Cost'),
                          f'DS{DS_i}-Theta_1'] = f"{cost_theta[1]:g}"

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_0'] = (
                    f"{time_vals[3]:g},{time_vals[4]:g}|"
                    f"{time_qnt_low:g},{time_qnt_up:g}")

                df_db.loc[(cmp.Index, 'Time'),
                          f'DS{DS_i}-Theta_1'] = f"{time_theta[1]:g}"

                df_db.loc[(cmp.Index, 'Time'),
                          f'DS{DS_i}-LongLeadTime'] = int(time_vals[5] > 0)

                if ds_map.count('1') == 1:

                    ds_pure_id = ds_map[::-1].find('1') + 1

                    meta_data['DamageStates'].update({f"DS{DS_i}": {
                        "Description": f"Pure DS{ds_pure_id}. " +
                                       cmp_meta[f"DS_{ds_pure_id}_Description"],
                        "RepairAction":
                            cmp_meta[f"DS_{ds_pure_id}_Repair_Description"]
                    }})

                else:

                    ds_combo = [f'DS{_.start() + 1}'
                                for _ in re.finditer('1', ds_map[::-1])]

                    meta_data['DamageStates'].update({f"DS{DS_i}": {
                        "Description": 'Combination of ' +
                                       ' & '.join(ds_combo),
                        "RepairAction": 'Combination of pure DS repair '
                                        'actions.'
                    }})

        # for every other component...
        else:
            # now look at each Damage State
            for DS_i in range(1, 6):

                # cost
                if not pd.isna(getattr(cmp, f'Best_Fit_DS{DS_i}')):
                    df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'Best_Fit_DS{DS_i}')])

                    if not pd.isna(getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}')):

                        theta_0_low = getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}')
                        theta_0_up = getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}')
                        qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS{DS_i}')
                        qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS{DS_i}')

                        if theta_0_low == 0. and theta_0_up == 0.:
                            df_db.loc[(cmp.Index, 'Cost'),
                                      f'DS{DS_i}-Family'] = np.nan

                        else:
                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_0'] = (
                                f"{theta_0_low:g},{theta_0_up:g}|"
                                f"{qnt_low:g},{qnt_up:g}")

                            df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Theta_1'] = (
                                f"{getattr(cmp, f'CV__Dispersion_DS{DS_i}'):g}")

                    else:
                        incomplete_cost = True

                    meta_data['DamageStates'].update({
                        f"DS{DS_i}": {
                            "Description": cmp_meta[f"DS_{DS_i}_Description"],
                            "RepairAction": cmp_meta[
                                f"DS_{DS_i}_Repair_Description"]}})

                # time
                if not pd.isna(getattr(cmp, f'Best_Fit_DS{DS_i}_1')):

                    df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = (
                        convert_family[getattr(cmp, f'Best_Fit_DS{DS_i}_1')])

                    if not pd.isna(getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1')):

                        theta_0_low = getattr(cmp, f'Lower_Qty_Mean_DS{DS_i}_1')
                        theta_0_up = getattr(cmp, f'Upper_Qty_Mean_DS{DS_i}_1')
                        qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS{DS_i}_1')
                        qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS{DS_i}_1')

                        if theta_0_low == 0. and theta_0_up == 0.:
                            df_db.loc[(cmp.Index, 'Time'),
                                      f'DS{DS_i}-Family'] = np.nan

                        else:
                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_0'] = (
                                f"{theta_0_low:g},{theta_0_up:g}|"
                                f"{qnt_low:g},{qnt_up:g}")

                            df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Theta_1'] = (
                                f"{getattr(cmp, f'CV__Dispersion_DS{DS_i}_2'):g}")

                        df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-LongLeadTime'] = (
                            int(getattr(cmp, f'DS_{DS_i}_Long_Lead_Time') == 'YES'))

                    else:
                        incomplete_time = True

        df_db.loc[(cmp.Index, 'Cost'), 'Incomplete'] = int(incomplete_cost)
        df_db.loc[(cmp.Index, 'Time'), 'Incomplete'] = int(incomplete_time)

        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.index = pd.MultiIndex.from_arrays(
        [df_db['Index'].values, df_db.index.get_level_values(1)])

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

    print("Successfully parsed and saved the repair consequence data from FEMA "
          "P58")


def create_FEMA_P58_bldg_injury_db(
        source_file,
        target_data_file='bldg_injury_DB_FEMA_P58_2nd.csv',
        target_meta_file='bldg_injury_DB_FEMA_P58_2nd.json'):
    """
    Create an injury consequence parameter database based on the FEMA P58 data

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    target_data_file: string
        Path where the consequence data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the consequence metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    df = pd.read_excel(source_file, sheet_name='Summary', header=2, index_col=1,
                       true_values=["YES", "Yes", "yes"],
                       false_values=["NO", "No", "no"])

    # remove empty rows and columns
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # filter the columns we need for the injury database
    cols_to_db = [
        "Fragility Unit of Measure",
        'DS Hierarchy',
    ]
    for DS_i in range(1, 6):
        cols_to_db += [

            f'DS {DS_i}, Potential non-collapse casualty?',
            f'DS {DS_i} - Casualty Affected Area',
            f'DS {DS_i} Serious Injury Rate - Median',
            f'DS {DS_i} Serious Injury Rate - Dispersion',
            f'DS {DS_i} Loss of Life Rate - Median',
            f'DS {DS_i} Loss of Life Rate - Dispersion',
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
        "DS 2, Description",
        "DS 3, Description",
        "DS 4, Description",
        "DS 5, Description",
    ]

    # remove special characters to make it easier to work with column names
    str_map = {
        ord(' '): "_",
        ord('.'): "_",
        ord('-'): "_",
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
    df_db_source.replace('By User', np.nan, inplace=True)

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
            f"DS{DS_i}-AffectedArea",
        ]

    # create the MultiIndex
    comps = df_db_source.index.values
    DVs = ['S1', 'S2']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'Severity'])

    df_db = pd.DataFrame(
        columns=out_cols,
        index=df_MI,
        dtype=float
    )

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    # for each component...
    # (this approach is not efficient, but easy to follow which was considered
    # more important than efficiency.)
    for cmp in df_db_source.itertuples():

        ID = cmp.Index.split('.')
        cmpID = f'{ID[0][0]}.{ID[0][1:3]}.{ID[0][3:5]}.{ID[1]}'

        # store the new index
        df_db.loc[cmp.Index, 'Index'] = cmpID

        # assume the component information is complete
        incomplete_S1 = False
        incomplete_S2 = False

        # store units

        df_db.loc[cmp.Index, 'Quantity-Unit'] = (
            ' '.join(cmp.Fragility_Unit_of_Measure.split(' ')[::-1]).strip())
        df_db.loc[(cmp.Index, 'S1'), 'DV-Unit'] = "persons"
        df_db.loc[(cmp.Index, 'S2'), 'DV-Unit'] = "persons"

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # every component is assumed to have a comp. description
        comments = cmp_meta['Component_Description']

        # the additional fields are added to the description if they exist
        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += f'\nConstruction Quality: ' \
                        f'{cmp_meta["Construction_Quality"]}'

        if cmp_meta['Seismic_Installation_Conditions'] not in [
                'Not Specified', 'Not applicable', 'Unknown', 'Any']:
            comments += f'\nSeismic Installation Conditions: ' \
                        f'{cmp_meta["Seismic_Installation_Conditions"]}'

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
            "DamageStates": {}
        }

        # Handle components with simultaneous damage states separately
        if 'Simul' in cmp.DS_Hierarchy:

            # Note that we are assuming that all damage states are triggered by
            # a single limit state in these components.
            # This assumption holds for the second edition of FEMA P58, but it
            # might need to be revisited in future editions.

            inj_data = {}
            ds_tot = 0

            # get the p10, p50, and p90 estimates for all damage states
            for DS_i in range(1, 6):

                casualty_model = getattr(
                    cmp, f'DS_{DS_i}_Potential_non_collapse_casualty')

                if casualty_model is True:

                    inj_data.update({f'DS{DS_i}': np.array([
                        getattr(cmp, f'DS_{DS_i}___Casualty_Affected_Area'),
                        getattr(cmp, f'DS_{DS_i}_Serious_Injury_Rate___Median'),
                        getattr(cmp, f'DS_{DS_i}_Serious_Injury_Rate___Dispersion'),
                        getattr(cmp, f'DS_{DS_i}_Loss_of_Life_Rate___Median'),
                        getattr(cmp, f'DS_{DS_i}_Loss_of_Life_Rate___Dispersion')
                    ])})
                    ds_tot += 1

                elif casualty_model is False:
                    ds_tot += 1

            # only continue if there is injury data
            if len(inj_data) == 0:
                continue

            # now prepare the equivalent mutex damage states
            sim_ds_count = ds_tot
            ds_count = 2 ** (sim_ds_count) - 1

            # Here we take advantage of knowing that for every component with
            # simultaneous damage states, only one of the DSs has injury
            # consequences.
            # This assumption holds for the second edition of FEMA P58, but it
            # might need to be revisited in future editions.

            ds_trig = list(inj_data.keys())[0]
            inj_data = inj_data[ds_trig]
            ds_trig = int(ds_trig[2:])

            for DS_i in range(1, ds_count + 1):
                ds_map = format(DS_i, f'0{sim_ds_count}b')

                if ds_map[-ds_trig] == '1':

                    # store the consequence data
                    for severity in ('S1', 'S2'):

                        A_affected = inj_data[0]

                        if severity == 'S1':
                            theta_0 = inj_data[1]
                            theta_1 = inj_data[2]
                        elif severity == 'S2':
                            theta_0 = inj_data[3]
                            theta_1 = inj_data[4]

                        if theta_0 != 0.0:

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-Family'] = 'lognormal'

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-Theta_0'] = theta_0

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-Theta_1'] = theta_1

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-AffectedArea'] = A_affected

                # store the metadata
                if ds_map.count('1') == 1:

                    ds_pure_id = ds_map[::-1].find('1') + 1

                    meta_data['DamageStates'].update({f"DS{DS_i}": {
                        "Description": f"Pure DS{ds_pure_id}. " +
                                       cmp_meta[
                                           f"DS_{ds_pure_id}_Description"]
                    }})

                else:

                    ds_combo = [f'DS{_.start() + 1}'
                                for _ in re.finditer('1', ds_map[::-1])]

                    meta_data['DamageStates'].update({f"DS{DS_i}": {
                        "Description": 'Combination of ' +
                                       ' & '.join(ds_combo)
                    }})

        # for every other component...
        else:
            # now look at each Damage State
            for DS_i in range(1, 6):

                casualty_flag = getattr(
                    cmp, f'DS_{DS_i}_Potential_non_collapse_casualty')

                if casualty_flag is True:

                    A_affected = getattr(cmp,
                                         f'DS_{DS_i}___Casualty_Affected_Area')

                    for severity in ('S1', 'S2'):

                        if severity == 'S1':
                            theta_0 = getattr(cmp, f'DS_{DS_i}_Serious_Injury_'
                                                   f'Rate___Median')
                            theta_1 = getattr(cmp, f'DS_{DS_i}_Serious_Injury_'
                                                   f'Rate___Dispersion')
                        elif severity == 'S2':
                            theta_0 = getattr(cmp, f'DS_{DS_i}_Loss_of_Life_'
                                                   f'Rate___Median')
                            theta_1 = getattr(cmp, f'DS_{DS_i}_Loss_of_Life_'
                                                   f'Rate___Dispersion')

                        if theta_0 != 0.0:

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-Family'] = 'lognormal'

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-Theta_0'] = theta_0

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-Theta_1'] = theta_1

                            df_db.loc[(cmp.Index, severity),
                                      f'DS{DS_i}-AffectedArea'] = A_affected

                            if (pd.isna(theta_0) or pd.isna(
                                    theta_1) or pd.isna(A_affected)):

                                if severity == 'S1':
                                    incomplete_S1 = True
                                else:
                                    incomplete_S2 = True

                if ~np.isnan(casualty_flag):

                    meta_data['DamageStates'].update({
                        f"DS{DS_i}": {"Description":
                                      cmp_meta[f"DS_{DS_i}_Description"]}})

        df_db.loc[(cmp.Index, 'S1'), 'Incomplete'] = int(incomplete_S1)
        df_db.loc[(cmp.Index, 'S2'), 'Incomplete'] = int(incomplete_S2)

        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.index = pd.MultiIndex.from_arrays(
        [df_db['Index'].values, df_db.index.get_level_values(1)])

    df_db.drop('Index', axis=1, inplace=True)

    # review the database and drop rows with no information
    cmp_to_drop = []
    for cmp in df_db.index:

        empty = True

        for DS_i in range(1, 16):
            if not pd.isna(df_db.loc[cmp, f'DS{DS_i}-Family']):
                empty = False
                break

        if empty:
            cmp_to_drop.append(cmp)

    df_db.drop(cmp_to_drop, axis=0, inplace=True)
    cmp_kept = df_db.index.get_level_values(0).unique()

    cmp_to_drop = []
    for cmp in meta_dict:
        if cmp not in cmp_kept:
            cmp_to_drop.append(cmp)

    for cmp in cmp_to_drop:
        del meta_dict[cmp]

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

    print("Successfully parsed and saved the injury consequence data from FEMA "
          "P58")


def create_FEMA_P58_bldg_redtag_db(
        source_file,
        target_data_file='bldg_redtag_DB_FEMA_P58_2nd.csv',
        target_meta_file='bldg_redtag_DB_FEMA_P58_2nd.json'):
    """
    Create an red tag consequence parameter database based on the FEMA P58 data

    The method was developed to process v3.1.2 of the FragilityDatabase xls
    that is provided with FEMA P58 2nd edition.

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    target_data_file: string
        Path where the consequence data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the consequence metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    df = pd.read_excel(source_file, sheet_name='Summary', header=2, index_col=1,
                       true_values=["YES", "Yes", "yes"],
                       false_values=["NO", "No", "no"])

    # take another pass with booleans because the first does not always work
    for true_str in ("YES", "Yes", "yes"):
        df.replace(true_str, True, inplace=True)

    for false_str in ("NO", "No", "no"):
        df.replace(false_str, False, inplace=True)

    # remove empty rows and columns
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # filter the columns we need for the injury database
    cols_to_db = [
        'DS Hierarchy',
    ]
    for DS_i in range(1, 6):
        cols_to_db += [
            f'DS {DS_i}, Unsafe Placard Trigger Flag',
            f'DS {DS_i}, Unsafe Placard Damage Median',
            f'DS {DS_i}, Unsafe Placard Damage Dispersion'
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
        "DS 2, Description",
        "DS 3, Description",
        "DS 4, Description",
        "DS 5, Description",
    ]

    # remove special characters to make it easier to work with column names
    str_map = {
        ord(' '): "_",
        ord('.'): "_",
        ord('-'): "_",
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
    df_db_source.replace('By User', np.nan, inplace=True)

    # initialize the output loss table
    # define the columns
    out_cols = [
        "Index",
        "Incomplete",
    ]
    for DS_i in range(1, 6):
        out_cols += [
            f"DS{DS_i}-Family",
            f"DS{DS_i}-Theta_0",
            f"DS{DS_i}-Theta_1"
        ]

    # create the database index
    comps = df_db_source.index.values

    df_db = pd.DataFrame(
        columns=out_cols,
        index=comps,
        dtype=float
    )

    # initialize the dictionary that stores the loss metadata
    meta_dict = {}

    # for each component...
    # (this approach is not efficient, but easy to follow which was considered
    # more important than efficiency.)
    for cmp in df_db_source.itertuples():

        ID = cmp.Index.split('.')
        cmpID = f'{ID[0][0]}.{ID[0][1:3]}.{ID[0][3:5]}.{ID[1]}'

        # store the new index
        df_db.loc[cmp.Index, 'Index'] = cmpID

        # assume the component information is complete
        incomplete = False

        # get the raw metadata for the component
        cmp_meta = df_meta.loc[cmp.Index, :]

        # store the global (i.e., not DS-specific) metadata

        # every component is assumed to have a comp. description
        comments = cmp_meta['Component_Description']

        # the additional fields are added to the description if they exist
        if cmp_meta['Construction_Quality'] != 'Not Specified':
            comments += f'\nConstruction Quality: ' \
                        f'{cmp_meta["Construction_Quality"]}'

        if cmp_meta['Seismic_Installation_Conditions'] not in [
                'Not Specified', 'Not applicable', 'Unknown', 'Any']:
            comments += f'\nSeismic Installation Conditions: ' \
                        f'{cmp_meta["Seismic_Installation_Conditions"]}'

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
            "DamageStates": {}
        }

        # Handle components with simultaneous damage states separately
        if 'Simul' in cmp.DS_Hierarchy:

            pass
            # Note that we are assuming that components with simultaneous
            # damage states do not have damage that would trigger a red tag.
            # This assumption holds for the second edition of FEMA P58, but it
            # might need to be revisited in future editions.

        # for every other component...
        else:
            # now look at each Damage State
            for DS_i in range(1, 6):

                redtag_flag = getattr(
                    cmp, f'DS_{DS_i}_Unsafe_Placard_Trigger_Flag')

                if redtag_flag is True:

                    theta_0 = getattr(cmp, f'DS_{DS_i}_Unsafe_Placard_Damage_'
                                           f'Median')
                    theta_1 = getattr(cmp, f'DS_{DS_i}_Unsafe_Placard_Damage_'
                                           f'Dispersion')

                    if theta_0 != 0.0:

                        df_db.loc[cmp.Index, f'DS{DS_i}-Family'] = 'lognormal'

                        df_db.loc[cmp.Index, f'DS{DS_i}-Theta_0'] = theta_0

                        df_db.loc[cmp.Index, f'DS{DS_i}-Theta_1'] = theta_1

                        if (pd.isna(theta_0) or pd.isna(theta_1)):

                            incomplete = True

                if ~np.isnan(redtag_flag):

                    meta_data['DamageStates'].update({
                        f"DS{DS_i}": {"Description":
                                      cmp_meta[f"DS_{DS_i}_Description"]}})

        df_db.loc[cmp.Index, 'Incomplete'] = int(incomplete)

        # store the metadata for this component
        meta_dict.update({cmpID: meta_data})

    # assign the Index column as the new ID
    df_db.set_index('Index', inplace=True)

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
    cmp_kept = df_db.index.get_level_values(0).unique()

    cmp_to_drop = []
    for cmp in meta_dict:
        if cmp not in cmp_kept:
            cmp_to_drop.append(cmp)

    for cmp in cmp_to_drop:
        del meta_dict[cmp]

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # rename the index
    df_db.index.name = "ID"

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the red tag consequence data from FEMA "
          "P58")


def create_Hazus_EQ_fragility_db(source_file,
                                 target_data_file='fragility_DB_Hazus_EQ.csv',
                                 target_meta_file='fragility_DB_Hazus_EQ.json'):
    """
    Create a database file based on the HAZUS EQ Technical Manual

    This method was developed to process a json file with tabulated data from
    v4.2.3 of the Hazus Earthquake Technical Manual. The json file is included
    in the resources folder of pelicun

    Parameters
    ----------
    source_file: string
        Path to the fragility database file.
    target_data_file: string
        Path where the fragility data file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the fragility metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    with open(source_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # prepare lists of labels for various building features
    design_levels = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'].keys())

    building_types = list(
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys())

    convert_design_level = {
        'High_code': 'HC',
        'Moderate_code': 'MC',
        'Low_code': 'LC',
        'Pre_code': 'PC'
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
            "LS4-DamageStateWeights"
        ],
        index=np.arange(len(building_types) * len(design_levels) * 5),
        dtype=float
    )
    counter = 0

    # First, prepare the structural fragilities
    S_data = raw_data['Structural_Fragility_Groups']

    for bt in building_types:
        for dl in design_levels:
            if bt in S_data['EDP_limits'][dl].keys():

                # create the component id
                cmp_id = f'STR.{bt}.{convert_design_level[dl]}'
                df_db.loc[counter, 'ID'] = cmp_id

                # store demand specifications
                df_db.loc[counter, 'Demand-Type'] = "Peak Roof Drift Ratio"
                df_db.loc[counter, 'Demand-Unit'] = "rad"
                df_db.loc[counter, 'Demand-Offset'] = 0

                # store the Limit State parameters
                for LS_i in range(1, 5):

                    df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
                    df_db.loc[counter, f'LS{LS_i}-Theta_0'] = \
                        S_data['EDP_limits'][dl][bt][LS_i - 1]
                    df_db.loc[counter, f'LS{LS_i}-Theta_1'] = \
                        S_data['Fragility_beta'][dl]

                    if LS_i == 4:
                        p_coll = S_data['P_collapse'][bt]
                        df_db.loc[counter, f'LS{LS_i}-DamageStateWeights'] = (
                            f'{1.0 - p_coll} | {p_coll}')

                counter += 1

    # Second, the non-structural drift sensitive one
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    # create the component id
    df_db.loc[counter, 'ID'] = 'NSD'

    # store demand specifications
    df_db.loc[counter, 'Demand-Type'] = "Peak Roof Drift Ratio"
    df_db.loc[counter, 'Demand-Unit'] = "rad"
    df_db.loc[counter, 'Demand-Offset'] = 0

    # store the Limit State parameters
    for LS_i in range(1, 5):
        df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
        df_db.loc[counter, f'LS{LS_i}-Theta_0'] = NSD_data['EDP_limits'][
            LS_i - 1]
        df_db.loc[counter, f'LS{LS_i}-Theta_1'] = NSD_data['Fragility_beta']

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

        # store the Limit State parameters
        for LS_i in range(1, 5):
            df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
            df_db.loc[counter, f'LS{LS_i}-Theta_0'] = \
                NSA_data['EDP_limits'][dl][LS_i - 1]
            df_db.loc[counter, f'LS{LS_i}-Theta_1'] = NSA_data['Fragility_beta']

        counter += 1

    # Fourth, the lifeline facilities
    LF_data = raw_data['Lifeline_Facilities']

    for bt in building_types:
        for dl in design_levels:
            if bt in LF_data['EDP_limits'][dl].keys():

                # create the component id
                cmp_id = f'LF.{bt}.{convert_design_level[dl]}'
                df_db.loc[counter, 'ID'] = cmp_id

                # store demand specifications
                df_db.loc[counter, 'Demand-Type'] = "Peak Ground Acceleration"
                df_db.loc[counter, 'Demand-Unit'] = "g"
                df_db.loc[counter, 'Demand-Offset'] = 0

                # store the Limit State parameters
                for LS_i in range(1, 5):

                    df_db.loc[counter, f'LS{LS_i}-Family'] = 'lognormal'
                    df_db.loc[counter, f'LS{LS_i}-Theta_0'] = \
                        LF_data['EDP_limits'][dl][bt][LS_i - 1]
                    df_db.loc[counter, f'LS{LS_i}-Theta_1'] = \
                        LF_data['Fragility_beta'][dl]

                    if LS_i == 4:
                        p_coll = LF_data['P_collapse'][bt]
                        df_db.loc[counter, f'LS{LS_i}-DamageStateWeights'] = (
                            f'{1.0 - p_coll} | {p_coll}')

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

            # store the Limit State parameters
            df_db.loc[counter, 'LS1-Family'] = 'lognormal'
            df_db.loc[counter, 'LS1-Theta_0'] = \
                GF_data['EDP_limits'][direction][f_depth]
            df_db.loc[counter, 'LS1-Theta_1'] = \
                GF_data['Fragility_beta'][direction][f_depth]
            p_complete = GF_data['P_Complete']
            df_db.loc[counter, 'LS1-DamageStateWeights'] = (
                f'{1.0 - p_complete} | {p_complete}')

            counter += 1

    # remove empty rows (from the end)
    df_db.dropna(how='all', inplace=True)

    # All Hazus components have complete fragility info,
    df_db.loc[:, 'Incomplete'] = 0

    # none of them are directional,
    df_db.loc[:, 'Demand-Directional'] = 0

    # rename the index
    df_db.set_index("ID", inplace=True)

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the fragility data
    df_db.to_csv(target_data_file)

    # save the metadata - later
    # with open(target_meta_file, 'w+') as f:
    #    json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the fragility data from Hazus EQ")


def create_Hazus_EQ_bldg_repair_db(source_file,
                                   target_data_file='bldg_repair_DB_Hazus_EQ.csv',
                                   target_meta_file='bldg_repair_DB_Hazus_EQ.json'):
    """
    Create a database file based on the HAZUS EQ Technical Manual

    This method was developed to process a json file with tabulated data from
    v4.2.3 of the Hazus Earthquake Technical Manual. The json file is included
    in the resources folder of pelicun

    Parameters
    ----------
    source_file: string
        Path to the Hazus database file.
    target_data_file: string
        Path where the repair DB file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the repair DB metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    with open(source_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # prepare lists of labels for various building features
    occupancies = list(
        raw_data['Structural_Fragility_Groups']['Repair_cost'].keys())

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
    comps = [f'{cmp_type}.{occ_type}'
             for cmp_type in cmp_types for occ_type in occupancies]
    DVs = ['Cost', 'Time']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])

    df_db = pd.DataFrame(
        columns=out_cols,
        index=df_MI,
        dtype=float
    )

    # First, prepare the structural damage consequences
    S_data = raw_data['Structural_Fragility_Groups']

    for occ_type in occupancies:

        # create the component id
        cmp_id = f'STR.{occ_type}'

        # store the consequence values for each Damage State
        for DS_i in range(1, 6):

            # DS4 and DS5 have identical repair consequences
            if DS_i == 5:
                ds_i = 4
            else:
                ds_i = DS_i

            df_db.loc[
                (cmp_id, 'Cost'),
                f'DS{DS_i}-Theta_0'] = S_data['Repair_cost'][occ_type][ds_i-1]

            df_db.loc[
                (cmp_id, 'Time'),
                f'DS{DS_i}-Theta_0'] = S_data['Repair_time'][occ_type][ds_i-1]

    # Second, the non-structural drift sensitive one
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    for occ_type in occupancies:

        # create the component id
        cmp_id = f'NSD.{occ_type}'

        # store the consequence values for each Damage State
        for DS_i in range(1, 5):

            df_db.loc[
                (cmp_id, 'Cost'),
                f'DS{DS_i}-Theta_0'] = NSD_data['Repair_cost'][occ_type][DS_i-1]

    # Third, the non-structural acceleration sensitive fragilities
    NSA_data = raw_data['NonStructural_Acceleration_Sensitive_Fragility_Groups']

    for occ_type in occupancies:

        # create the component id
        cmp_id = f'NSA.{occ_type}'

        # store the consequence values for each Damage State
        for DS_i in range(1, 5):

            df_db.loc[
                (cmp_id, 'Cost'),
                f'DS{DS_i}-Theta_0'] = NSA_data['Repair_cost'][occ_type][DS_i-1]

    # Fourth, the lifeline facilities
    LF_data = raw_data['Lifeline_Facilities']

    for occ_type in occupancies:

        # create the component id
        cmp_id = f'LF.{occ_type}'

        # store the consequence values for each Damage State
        for DS_i in range(1, 6):

            # DS4 and DS5 have identical repair consequences
            if DS_i == 5:
                ds_i = 4
            else:
                ds_i = DS_i

            df_db.loc[
                (cmp_id, 'Cost'),
                f'DS{DS_i}-Theta_0'] = LF_data['Repair_cost'][occ_type][ds_i - 1]

            df_db.loc[
                (cmp_id, 'Time'),
                f'DS{DS_i}-Theta_0'] = LF_data['Repair_time'][occ_type][ds_i - 1]

    # remove empty rows (from the end)
    df_db.dropna(how='all', inplace=True)

    # All Hazus components have complete fragility info,
    df_db.loc[:, 'Incomplete'] = 0

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
    # with open(target_meta_file, 'w+') as f:
    #    json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the repair consequence data from Hazus "
          "EQ")


def create_Hazus_EQ_bldg_injury_db(source_file,
                                   target_data_file='bldg_injury_DB_Hazus_EQ.csv',
                                   target_meta_file='bldg_injury_DB_Hazus_EQ.json'):
    """
    Create a database file based on the HAZUS EQ Technical Manual

    This method was developed to process a json file with tabulated data from
    v4.2.3 of the Hazus Earthquake Technical Manual. The json file is included
    in the resources folder of pelicun

    Parameters
    ----------
    source_file: string
        Path to the Hazus database file.
    target_data_file: string
        Path where the injury DB file should be saved. A csv file is
        expected.
    target_meta_file: string
        Path where the injury DB metadata should be saved. A json file is
        expected.

    """

    # parse the source file
    with open(source_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # prepare lists of labels for various building features
    building_types = list(
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys())

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
    cmp_types = ['STR', 'LF']
    comps = [f'{cmp_type}.{bt}'
             for cmp_type in cmp_types for bt in building_types]
    DVs = ['S1', 'S2', 'S3', 'S4']
    df_MI = pd.MultiIndex.from_product([comps, DVs], names=['ID', 'DV'])

    df_db = pd.DataFrame(
        columns=out_cols,
        index=df_MI,
        dtype=float
    )

    # First, prepare the structural damage consequences
    S_data = raw_data['Structural_Fragility_Groups']

    for bt in building_types:

        # create the component id
        cmp_id = f'STR.{bt}'

        # store the consequence values for each Damage State
        for DS_i in range(1, 6):

            # DS5 is stored under 'collapse'
            if DS_i == 5:
                ds_i = 'Collapse'
            else:
                ds_i = f'DS{DS_i}'

            for S_i in range(1, 5):
                s_label = f'S{S_i}'
                df_db.loc[(cmp_id, s_label), f'DS{DS_i}-Theta_0'] = (
                    S_data['Injury_rates'][ds_i][bt][S_i-1])

    # Second, the lifeline facilities
    LF_data = raw_data['Lifeline_Facilities']

    for bt in building_types:

        # create the component id
        cmp_id = f'STR.{bt}'

        # store the consequence values for each Damage State
        for DS_i in range(1, 6):

            # DS5 is stored under 'collapse'
            if DS_i == 5:
                ds_i = 'Collapse'
            else:
                ds_i = f'DS{DS_i}'

            for S_i in range(1, 5):
                s_label = f'S{S_i}'
                df_db.loc[(cmp_id, s_label), f'DS{DS_i}-Theta_0'] = (
                    S_data['Injury_rates'][ds_i][bt][S_i - 1])

    # remove empty rows
    df_db.dropna(how='all', inplace=True)

    # All Hazus components have complete fragility info,
    df_db.loc[:, 'Incomplete'] = 0

    # The damage quantity unit is the same for all consequence values
    df_db.loc[:, 'Quantity-Unit'] = "1 EA"

    # The output units are also identical among all components
    df_db.loc[:, 'DV-Unit'] = "injury_rate"

    # convert to simple index
    df_db = base.convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata - later
    # with open(target_meta_file, 'w+') as f:
    #    json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the injury consequence data from Hazus "
          "EQ")
