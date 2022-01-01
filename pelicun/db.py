# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
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

    convert_jsons_to_table
    save_to_standard_HDF
    convert_json_files_to_HDF
    convert_Series_to_dict

    convert_P58_data_to_json
    create_HAZUS_EQ_json_files
    create_HAZUS_EQ_story_json_files
    create_HAZUS_EQ_PGA_json_files
    create_HAZUS_HU_json_files

"""

from .base import *
from .uq import fit_distribution_to_percentiles
from pathlib import Path
import re
import json

import xml.etree.ElementTree as ET
import shutil


def parse_DS_Hierarchy(DSH):
    """
    Parses the FEMA P58 DS hierarchy into a set of arrays.
    """
    if 'Seq' == DSH[:3]:
        DSH = DSH[4:-1]

    DS_setup = []

    while len(DSH) > 0:
        if DSH[:2] == 'DS':
            DS_setup.append(DSH[:3])
            DSH = DSH[4:]
        elif DSH[:5] in ['MutEx', 'Simul']:
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

    # initialize the output fragility table
    df_db = pd.DataFrame(
        columns=[
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
            if LS_contents[0] in ['MutEx', 'Simul']:

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
                if ((np.unique(median_demands).size != 1) or
                    (np.unique(dispersions).size != 1)):
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
                                "RepairAction": cmp_meta[f"DS_{ds_pure_id}_Repair_Description"]
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
                    sim_weights = np.array(sim_weights) / np.sum(sim_weights)

                    weights = sim_weights

                theta_0 = median_demands[0]
                theta_1 = dispersions[0]
                weights = ' | '.join([f"{w:.6f}" for w in weights])

                df_db.loc[cmp.Index, f'LS{LS_i}-DamageStateWeights'] = weights

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
        meta_dict.update({cmp.Index: meta_data})

    # rename the index
    df_db.index.name = "ID"

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the fragility data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the fragility data from FEMA P58")

def create_FEMA_P58_bldg_repair_db(source_file,
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
         for sheet in ['Summary', 'Cost Summary']], axis=1)

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

        # assume the component information is complete
        incomplete_cost = False
        incomplete_time = False

        # store units

        df_db.loc[cmp.Index, 'Quantity-Unit'] = (
            ' '.join(cmp.Fragility_Unit_of_Measure.split(' ')[::-1]).strip())
        df_db.loc[(cmp.Index, 'Cost'), 'DV-Unit'] = "US$_2011"
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
                cost_qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS1')
                cost_qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS1')
                time_qnt_low = getattr(cmp, f'Lower_Qty_Cutoff_DS1_1')
                time_qnt_up = getattr(cmp, f'Upper_Qty_Cutoff_DS1_1')

                # store the results
                df_db.loc[(cmp.Index, 'Cost'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index,'Cost'), f'DS{DS_i}-Theta_0'] = (
                    f"{cost_vals[3]:g},{cost_vals[4]:g}|"
                    f"{cost_qnt_low:g},{cost_qnt_up:g}")

                df_db.loc[(cmp.Index,'Cost'),
                          f'DS{DS_i}-Theta_1'] = f"{cost_theta[1]:g}"

                df_db.loc[(cmp.Index, 'Time'), f'DS{DS_i}-Family'] = family_hat

                df_db.loc[(cmp.Index,'Time'), f'DS{DS_i}-Theta_0'] = (
                    f"{time_vals[3]:g},{time_vals[4]:g}|"
                    f"{time_qnt_low:g},{time_qnt_up:g}")

                df_db.loc[(cmp.Index,'Time'),
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
                            "RepairAction": cmp_meta[f"DS_{DS_i}_Repair_Description"]}})

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
        meta_dict.update({cmp.Index: meta_data})

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
        if cmp[0] in meta_dict.keys():
            del meta_dict[cmp[0]]

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    df_db = convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the repair consequence data from FEMA "
          "P58")

def create_FEMA_P58_bldg_injury_db(source_file,
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

                if casualty_model  == True:

                    inj_data.update({f'DS{DS_i}': np.array([
                        getattr(cmp, f'DS_{DS_i}___Casualty_Affected_Area'),
                        getattr(cmp, f'DS_{DS_i}_Serious_Injury_Rate___Median'),
                        getattr(cmp, f'DS_{DS_i}_Serious_Injury_Rate___Dispersion'),
                        getattr(cmp, f'DS_{DS_i}_Loss_of_Life_Rate___Median'),
                        getattr(cmp, f'DS_{DS_i}_Loss_of_Life_Rate___Dispersion')
                    ])})
                    ds_tot += 1

                elif casualty_model == False:
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
                    for severity in ['S1', 'S2']:

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

                if casualty_flag==True:

                    A_affected = getattr(cmp,
                                         f'DS_{DS_i}___Casualty_Affected_Area')

                    for severity in ['S1', 'S2']:

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

                            if (pd.isna(theta_0) or pd.isna(theta_1) or
                                pd.isna(A_affected)):

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
        meta_dict.update({cmp.Index: meta_data})

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
    for cmp in meta_dict.keys():
        if cmp not in cmp_kept:
            cmp_to_drop.append(cmp)

    for cmp in cmp_to_drop:
        del meta_dict[cmp]

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    df_db = convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata
    with open(target_meta_file, 'w+') as f:
        json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the injury consequence data from FEMA "
          "P58")

def create_FEMA_P58_bldg_redtag_db(source_file,
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
    for true_str in ["YES", "Yes", "yes"]:
        df.replace(true_str, True, inplace=True)

    for false_str in ["NO", "No", "no"]:
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

                if redtag_flag==True:

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
        meta_dict.update({cmp.Index: meta_data})

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
    for cmp in meta_dict.keys():
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
    with open(target_meta_file, 'w+') as f:
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
    with open(source_file, 'r') as f:
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
    df_db.loc[counter, 'ID'] = f'NSD'

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
        df_db.loc[counter, 'Demand-Offset'] = 1

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

    for direction in ['Horizontal', 'Vertical']:
        for f_depth in ['Shallow', 'Deep']:
            # create the component id
            cmp_id = f'GF.{direction[0]}.{f_depth[0]}'
            df_db.loc[counter, 'ID'] = cmp_id

            # store demand specifications
            df_db.loc[counter, 'Demand-Type'] = "Permanent Ground Deformation"
            df_db.loc[counter, 'Demand-Unit'] = "inch"
            df_db.loc[counter, 'Demand-Offset'] = 0

            # store the Limit State parameters
            df_db.loc[counter, f'LS1-Family'] = 'lognormal'
            df_db.loc[counter, f'LS1-Theta_0'] = \
            GF_data['EDP_limits'][direction][f_depth]
            df_db.loc[counter, f'LS1-Theta_1'] = \
            GF_data['Fragility_beta'][direction][f_depth]
            p_complete = GF_data['P_Complete']
            df_db.loc[counter, f'LS1-DamageStateWeights'] = (
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
    #with open(target_meta_file, 'w+') as f:
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
    with open(source_file, 'r') as f:
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

    # The output units are also indentical among all components
    df_db.loc[idx[:, 'Cost'], 'DV-Unit'] = "loss_ratio"
    df_db.loc[idx[:, 'Time'], 'DV-Unit'] = "day"

    # convert to simple index
    df_db = convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata - later
    #with open(target_meta_file, 'w+') as f:
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
    with open(source_file, 'r') as f:
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

            for S_i in range(1,5):
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

    # The output units are also indentical among all components
    df_db.loc[:, 'DV-Unit'] = "injury_rate"

    # convert to simple index
    df_db = convert_to_SimpleIndex(df_db, 0)

    # rename the index
    df_db.index.name = "ID"

    # convert to optimal datatypes to reduce file size
    df_db = df_db.convert_dtypes()

    # save the consequence data
    df_db.to_csv(target_data_file)

    # save the metadata - later
    #with open(target_meta_file, 'w+') as f:
    #    json.dump(meta_dict, f, indent=2)

    print("Successfully parsed and saved the injury consequence data from Hazus "
          "EQ")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# LEGACY CODE
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def dict_generator(indict, pre=None):
    """
    Lists all branches of a tree defined by a dictionary.

    The dictionary can have nested dictionaries and lists. When encountering a
    list, its elements are returned as separate branches with each element's id
    created as a combination of the parent key and #i where i stands for the
    element number in the list.

    This method can process a json file and break it up into independent
    branches.

    """
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v_id, v in enumerate(value):
                    for d in dict_generator(v, pre + [key + f'#{v_id}']):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]


def get_val_from_dict(indict, col):
    """
    Gets the value from a branch of a dictionary.

    The dictionary can have nested dictionaries and lists. When walking through
    lists, #i in the branch data identifies the ith element of the list.

    This method can be used to travel branches of a dictionary previously
    defined by the dict_generator method above.

    """

    val = indict

    for col_i in col:
        if col_i != ' ':
            if '#' in col_i:
                col_name, col_id = col_i.split('#')
                col_id = int(col_id)
                if (col_name in val.keys()) and (col_id < len(val[col_name])):
                    val = val[col_name][int(col_id)]
                else:
                    return None

            elif col_i in val.keys():
                val = val[col_i]
            else:
                return None

    return val

def convert_jsons_to_table(json_id_list, json_list, json_template):
    # Define the header for the data table based on the template structure
    header = np.array(
        [[col[:-1], len(col[:-1])] for col in dict_generator(json_template)])
    lvls = max(np.transpose(header)[1])
    header = [col + (lvls - size) * [' ', ] for col, size in header]

    # Use the header to initialize the DataFrame that will hold the data
    MI = pd.MultiIndex.from_tuples(header)

    json_DF = pd.DataFrame(columns=MI, index=json_id_list)
    json_DF.index.name = 'ID'

    # Load the data into the DF
    for json_id, json_data in zip(json_id_list, json_list):

        for col in json_DF.columns:

            val = get_val_from_dict(json_data, col)

            if val is not None:
                json_DF.at[json_id, col] = val

    # Remove empty rows and columns
    json_DF = json_DF.dropna(axis=0, how='all')
    json_DF = json_DF.dropna(axis=1, how='all')

    # Set the dtypes for the columns based on the template
    for col in json_DF.columns:
        dtype = get_val_from_dict(json_template, col)

        if dtype != 'string':
            try:
                json_DF[col] = json_DF[col].astype(dtype)
            except:
                print(col, dtype)
        else:
            json_DF[col] = json_DF[col].apply(str)

    return json_DF


def save_to_standard_HDF(df, name, target_path, mode='w'):
    """
    Saves a DataFrame in a standard HDF format using h5py.

    """
    import h5py # import here to avoid an issue on Stampede2

    df = df.T

    hf = h5py.File(target_path, mode)

    #try:
    if True:
        # save each row (i.e., column of FG_df) as a separate dataset in the file
        for row_id, row in df.iterrows():
            row = row.convert_dtypes()

            # create a tree that identifies the column in the hierarchy
            row_name = name
            for label in row_id:
                if label != " ":
                    row_name += f'/{label}'

            # perform the appropriate type conversion before saving
            if row.dtype == np.float64:
                values = row.values.astype(float)

                hf.create_dataset(row_name, data=values)

            elif row.dtype == pd.StringDtype():
                # Strings are saved as ASCII strings so that the files can be
                # opened with any tool on any platform. Non-ASCII characters
                # are replaced by a backslash-escaped UTF8 identifier.
                values = row.values.astype(str)
                values = np.char.encode(values, encoding='ASCII',
                                        errors='backslashreplace')

                hf.create_dataset(row_name, data=values)

            elif row.dtype in [pd.BooleanDtype(), pd.Int64Dtype()]:
                row.fillna(-1, inplace=True)
                values = row.values.astype(int)

                hf.create_dataset(row_name, data=values)

            else:
                print("unknown dtype: ", row.dtype)

        # finally, save the index
        values = df.columns.values.astype(str)
        values = np.char.encode(values, encoding='ASCII',
                                errors='backslashreplace')
        hf.create_dataset(f'{name}/index', data=values)

    #except:
    #    show_warning("Error while trying to save standard HDF5 file.")

    hf.close()


def convert_json_files_to_HDF(data_source_dir, DL_dir, db_name):
    """
    Converts data from json files to a single HDF5 file

    """

    # Start with the fragility and consequence data - we'll call it data

    DL_dir = Path(DL_dir).resolve()
    data_source_dir = Path(data_source_dir).resolve()

    # get a list of json files to convert
    FG_ID_list = [filename[:-5] for filename in os.listdir(DL_dir / 'DL json')]

    # We will use a template.json to define the schema for the jsons and the
    # header for the data table.
    with open(data_source_dir / 'DL_template.json', 'r') as f:
        FG_template = json.load(f)

    FG_list = []
    DL_json_dir = DL_dir / 'DL json'
    for FG_i in FG_ID_list:
        with open(DL_json_dir / f'{FG_i}.json', 'r') as f:
            FG_list.append(json.load(f))

    FG_df = convert_jsons_to_table(FG_ID_list, FG_list, FG_template)

    # start with saving the data in standard HDF5 format
    save_to_standard_HDF(FG_df, name='data_standard',
                         target_path=DL_dir / f'{db_name}.hdf')

    # then also save it using PyTables for quick access and slicing
    FG_df.to_hdf(DL_dir / f'{db_name}.hdf', 'data', mode='a', format='table',
                 complevel=1, complib='blosc:snappy')

    # Now add the population distribution data - we'll call it pop

    # Only do this if there is population data
    try:
        with open(DL_dir / 'population.json', 'r') as f:
            pop = json.load(f)

        pop_ID_list = list(pop.keys())

        pop_data = [pop[key] for key in pop.keys()]

        with open(data_source_dir / 'pop_template.json', 'r') as f:
            pop_template = json.load(f)

        pop_df = convert_jsons_to_table(pop_ID_list, pop_data, pop_template)

        save_to_standard_HDF(pop_df, name='pop_standard',
                             target_path=DL_dir / f'{db_name}.hdf', mode='a')

        pop_df.to_hdf(DL_dir / f'{db_name}.hdf', 'pop', mode='a', format='table',
                      complevel=1, complib='blosc:snappy')

    except:
        pass

def convert_Series_to_dict(comp_Series):
    """
    Converts data from a table to a json file

    """

    comp_Series = comp_Series.dropna(how='all')

    comp_dict = {}

    for branch in comp_Series.index:

        nested_dict = comp_dict
        parent_dict = None
        parent_val = None
        parent_id = None

        for val in branch:
            if val != ' ':
                if '#' in val:
                    val, list_id = val.split('#')
                    list_id = int(list_id)
                else:
                    list_id = None

                if val not in nested_dict.keys():
                    if list_id is not None:
                        nested_dict.update({val: []})
                    else:
                        nested_dict.update({val: {}})

                if list_id is not None:
                    if list_id > len(nested_dict[val]) - 1:
                        nested_dict[val].append({})
                    parent_dict = nested_dict
                    nested_dict = nested_dict[val][list_id]

                    parent_id = list_id

                else:
                    parent_dict = nested_dict
                    nested_dict = nested_dict[val]

                parent_val = val

        if isinstance(parent_dict[parent_val], dict):
            parent_dict[parent_val] = comp_Series[branch]
        else:
            parent_dict[parent_val][parent_id] = comp_Series[branch]

    return comp_dict

def convert_P58_data_to_json(data_dir, target_dir):
    """
    Create JSON data files from publicly available P58 data.

    FEMA P58 damage and loss information is publicly available in an Excel
    spreadsheet and also in a series of XML files as part of the PACT tool.
    Those files are copied to the resources folder in the pelicun repo. Here
    we collect the available information on Fragility Groups from those files
    and save the damage and loss data in the common SimCenter JSON format.

    A large part of the Fragility Groups in FEMA P58 do not have complete
    damage and loss information available. These FGs are clearly marked with
    an incomplete flag in the JSON file and the 'Undefined' value highlights
    the missing pieces of information.

    Parameters
    ----------
    data_dir: string
        Path to the folder with the FEMA P58 Excel file and a 'DL xml'
        subfolder in it that contains the XML files.
    target_dir: string
        Path to the folder where the JSON files shall be saved.

    """

    data_dir = Path(data_dir).resolve()
    target_dir = Path(target_dir).resolve()
    DL_dir = None

    convert_unit = {
        'Unit less': 'ea',
        'Radians'  : 'rad',
        'g'        : 'g',
        'meter/sec': 'mps'

    }

    convert_DSG_type = {
        'MutEx': 'MutuallyExclusive',
        'Simul': 'Simultaneous'
    }

    def parse_DS_Hierarchy(DSH):

        if 'Seq' == DSH[:3]:
            DSH = DSH[4:-1]

        DS_setup = []

        while len(DSH) > 0:
            if DSH[:2] == 'DS':
                DS_setup.append(DSH[:3])
                DSH = DSH[4:]
            elif DSH[:5] in ['MutEx', 'Simul']:
                closing_pos = DSH.find(')')
                subDSH = DSH[:closing_pos + 1]
                DSH = DSH[closing_pos + 2:]

                DS_setup.append([subDSH[:5]] + subDSH[6:-1].split(','))

        return DS_setup

    def parse_DS_xml(DS_xml):
        CFG = DS_xml.find('ConsequenceGroup')
        CFG_C = CFG.find('CostConsequence')
        CFG_T = CFG.find('TimeConsequence')

        repair_cost = dict(
            Amount=[float(CFG_C.find('MaxAmount').text),
                    float(CFG_C.find('MinAmount').text)],
            Quantity=[float(CFG_C.find('LowerQuantity').text),
                      float(CFG_C.find('UpperQuantity').text)],
            CurveType=CFG_C.find('CurveType').text,
            Beta=float(CFG_C.find('Uncertainty').text),
            Bounds=[0., 'None']
        )
        if repair_cost['Amount'] == [0.0, 0.0]:
            repair_cost['Amount'] = 'Undefined'

        repair_time = dict(
            Amount=[float(CFG_T.find('MaxAmount').text),
                    float(CFG_T.find('MinAmount').text)],
            Quantity=[float(CFG_T.find('LowerQuantity').text),
                      float(CFG_T.find('UpperQuantity').text)],
            CurveType=CFG_T.find('CurveType').text,
            Beta=float(CFG_T.find('Uncertainty').text),
            Bounds=[0., 'None']
        )
        if repair_time['Amount'] == [0.0, 0.0]:
            repair_time['Amount'] = 'Undefined'

        return repair_cost, repair_time

    def is_float(s):
        try:
            if type(s) == str and s[-1] == '%':
                s_f = float(s[:-1]) / 100.
            else:
                s_f = float(s)
            if np.isnan(s_f):
                return False
            else:
                return True
        except ValueError:
            return False

    src_df = pd.read_excel(data_dir / 'PACT_fragility_data.xlsx')
    ID_list = src_df['NISTIR Classification']

    XML_list = [f for f in os.listdir(data_dir / 'DL xml') if f.endswith('.xml')]

    incomplete_count = 0

    for filename in XML_list:

        comp_ID = filename[:-4]

        #try:
        if True:
            tree = ET.parse((data_dir / 'DL xml') / f'{comp_ID}.xml')
            root = tree.getroot()

            # correct for the error in the numbering of RC beams
            if (comp_ID[:5] == 'B1051') and (
                comp_ID[-1] not in str([1, 2, 3, 4])):
                comp_ID = 'B1042' + comp_ID[5:]

            row = src_df.loc[np.where(ID_list == comp_ID)[0][0], :]

            json_output = {}
            incomplete = False

            json_output.update({'Name': row['Component Name']})

            QU = row['Fragility Unit of Measure']
            QU = QU.split(' ')
            if is_float(QU[1]):
                if QU[0] in ['TN', 'AP', 'CF', 'KV']:
                    QU[0] = 'ea'
                    QU[1] = 1
                json_output.update({'QuantityUnit': [int(QU[1]), QU[0]]})
            else:
                json_output.update({'QuantityUnit': [0., 'Undefined']})
                incomplete = True

            json_output.update({'Directional': row['Directional?'] in ['YES']})
            json_output.update({'Correlated': row['Correlated?'] in ['YES']})

            json_output.update({
                'EDP': {
                    'Type'  : row['Demand Parameter (value):'],
                    'Unit'  : [1,
                               convert_unit[row['Demand Parameter (unit):']]],
                    'Offset': int(
                        row['Demand Location (use floor above? Yes/No)'] in [
                            'Yes'])
                }
            })

            json_output.update({
                'GeneralInformation': {
                    'ID'         : row['NISTIR Classification'],
                    'Description': row['Component Description'],
                    'Author'     : row['Author'],
                    'Official'   : root.find('Official').text in ['True',
                                                                  'true'],
                    'DateCreated': root.find('DateCreated').text,
                    'Approved'   : root.find('Approved').text in ['True',
                                                                  'true'],
                    'Incomplete' : root.find('Incomplete').text in ['True',
                                                                    'true'],
                    'Notes'      : row['Comments / Notes']
                }
            })
            for key in json_output['GeneralInformation'].keys():
                if json_output['GeneralInformation'][key] is np.nan:
                    json_output['GeneralInformation'][key] = 'Undefined'

            json_output.update({
                'Ratings': {
                    'DataQuality'  : row['Data Quality'],
                    'DataRelevance': row['Data Relevance'],
                    'Documentation': row['Documentation Quality'],
                    'Rationality'  : row['Rationality'],
                }
            })
            for key in json_output['Ratings'].keys():
                if json_output['Ratings'][key] is np.nan:
                    json_output['Ratings'][key] = 'Undefined'

            DSH = parse_DS_Hierarchy(row['DS Hierarchy'])

            json_output.update({'DSGroups': []})

            for DSG in DSH:
                if DSG[0] in ['MutEx', 'Simul']:
                    mu = row['DS {}, Median Demand'.format(DSG[1][-1])]
                    beta = row[
                        'DS {}, Total Dispersion (Beta)'.format(DSG[1][-1])]
                    if is_float(mu) and is_float(beta):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu),
                            'Beta'        : float(beta),
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : convert_DSG_type[DSG[0]],
                            'DamageStates': DSG[1:]
                        })
                    else:
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu) if is_float(
                                mu) else 'Undefined',
                            'Beta'        : float(beta) if is_float(
                                beta) else 'Undefined',
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : convert_DSG_type[DSG[0]],
                            'DamageStates': DSG[1:]
                        })
                        incomplete = True
                else:
                    mu = row['DS {}, Median Demand'.format(DSG[-1])]
                    beta = row['DS {}, Total Dispersion (Beta)'.format(DSG[-1])]
                    if is_float(mu) and is_float(beta):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu),
                            'Beta'        : float(beta),
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [DSG],
                        })
                    else:
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu) if is_float(
                                mu) else 'Undefined',
                            'Beta'        : float(beta) if is_float(
                                beta) else 'Undefined',
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [DSG],
                        })
                        incomplete = True

            need_INJ = False
            need_RT = False
            for DSG in json_output['DSGroups']:
                DS_list = DSG['DamageStates']
                DSG['DamageStates'] = []
                for DS in DS_list:

                    # avoid having NaN as repair measures
                    repair_measures = row['DS {}, Repair Description'.format(DS[-1])]
                    if not isinstance(repair_measures, str):
                        repair_measures = ""

                    DSG['DamageStates'].append({
                        'Weight'        :
                            float(row['DS {}, Probability'.format(DS[-1])]),
                        'LongLeadTime'  :
                            int(row['DS {}, Long Lead Time'.format(DS[-1])] in [
                                'YES']),
                        'Consequences'  : {},
                        'Description'   :
                            row['DS {}, Description'.format(DS[-1])],
                        'RepairMeasures': repair_measures
                    })

                    IMG = row['DS{}, Illustrations'.format(DS[-1])]
                    if IMG not in ['none', np.nan]:
                        DSG['DamageStates'][-1].update({'DamageImageName': IMG})

                    AA = row['DS {} - Casualty Affected Area'.format(DS[-1])]
                    if (isinstance(AA, str) and (is_float(AA.split(' ')[0]))):
                        AA = AA.split(' ')
                        DSG['DamageStates'][-1].update(
                            {'AffectedArea': [int(AA[0]), AA[1]]})
                        need_INJ = True
                    else:
                        DSG['DamageStates'][-1].update(
                            {'AffectedArea': [0, 'SF']})

                    DSG['DamageStates'][-1]['Consequences'].update(
                        {'Injuries': [{}, {}]})

                    INJ0 = DSG[
                        'DamageStates'][-1]['Consequences']['Injuries'][0]
                    INJ_mu = row[
                        'DS {} Serious Injury Rate - Median'.format(DS[-1])]
                    INJ_beta = row[
                        'DS {} Serious Injury Rate - Dispersion'.format(DS[-1])]
                    if is_float(INJ_mu) and is_float(INJ_beta):
                        INJ0.update({
                            'Amount'   : float(INJ_mu),
                            'Beta'     : float(INJ_beta),
                            'CurveType': 'Normal',
                            'Bounds'   : [0., 1.]
                        })

                        if INJ_mu != 0.0:
                            need_INJ = True
                            if DSG['DamageStates'][-1]['AffectedArea'][0] == 0:
                                incomplete = True
                    else:
                        INJ0.update({'Amount'   :
                                         float(INJ_mu) if is_float(INJ_mu)
                                         else 'Undefined',
                                     'Beta'     :
                                         float(INJ_beta) if is_float(INJ_beta)
                                         else 'Undefined',
                                     'CurveType': 'Normal'})
                        if ((INJ0['Amount'] == 'Undefined') or
                            (INJ0['Beta'] == 'Undefined')):
                            incomplete = True

                    INJ1 = DSG[
                        'DamageStates'][-1]['Consequences']['Injuries'][1]
                    INJ_mu = row['DS {} Loss of Life Rate - Median'.format(DS[-1])]
                    INJ_beta = row['DS {} Loss of Life Rate - Dispersion'.format(DS[-1])]
                    if is_float(INJ_mu) and is_float(INJ_beta):
                        INJ1.update({
                            'Amount'   : float(INJ_mu),
                            'Beta'     : float(INJ_beta),
                            'CurveType': 'Normal',
                            'Bounds'   : [0., 1.]
                        })
                        if INJ_mu != 0.0:
                            need_INJ = True
                            if DSG['DamageStates'][-1]['AffectedArea'][0] == 0:
                                incomplete = True
                    else:
                        INJ1.update({'Amount'   :
                                         float(INJ_mu) if is_float(INJ_mu)
                                         else 'Undefined',
                                     'Beta'     :
                                         float(INJ_beta) if is_float(INJ_beta)
                                         else 'Undefined',
                                     'CurveType': 'Normal',
                                     'Bounds': [0., 1.]})
                        if ((INJ1['Amount'] == 'Undefined') or
                            (INJ1['Beta'] == 'Undefined')):
                            incomplete = True

                    DSG['DamageStates'][-1]['Consequences'].update({'RedTag': {}})
                    RT = DSG['DamageStates'][-1]['Consequences']['RedTag']

                    RT_mu = row['DS {}, Unsafe Placard Damage Median'.format(DS[-1])]
                    RT_beta = row['DS {}, Unsafe Placard Damage Dispersion'.format(DS[-1])]
                    if is_float(RT_mu) and is_float(RT_beta):
                        RT.update({
                            'Amount'   : float(RT_mu),
                            'Beta'     : float(RT_beta),
                            'CurveType': 'Normal',
                            'Bounds'   : [0., 1.]
                        })
                        if RT['Amount'] != 0.0:
                            need_RT = True
                    else:
                        RT.update({'Amount'   :
                                       float(RT_mu[:-1]) if is_float(RT_mu)
                                       else 'Undefined',
                                   'Beta'     :
                                       float(RT_beta[:-1]) if is_float(RT_beta)
                                       else 'Undefined',
                                   'CurveType': 'Normal',
                                   'Bounds': [0., 1.]})
                        if ((RT['Amount'] == 'Undefined') or
                            (RT['Beta'] == 'Undefined')):
                            incomplete = True

            # remove the unused fields
            if not need_INJ:
                for DSG in json_output['DSGroups']:
                    for DS in DSG['DamageStates']:
                        del DS['AffectedArea']
                        del DS['Consequences']['Injuries']

            if not need_RT:
                for DSG in json_output['DSGroups']:
                    for DS in DSG['DamageStates']:
                        del DS['Consequences']['RedTag']

            # collect the repair cost and time consequences from the XML file
            DSG_list = root.find('DamageStates').findall('DamageState')
            for DSG_i, DSG_xml in enumerate(DSG_list):

                if DSG_xml.find('DamageStates') is not None:
                    DS_list = (DSG_xml.find('DamageStates')).findall('DamageState')
                    for DS_i, DS_xml in enumerate(DS_list):
                        r_cost, r_time = parse_DS_xml(DS_xml)
                        CONSEQ = json_output['DSGroups'][DSG_i][
                            'DamageStates'][DS_i]['Consequences']
                        CONSEQ.update({
                            'ReconstructionCost': r_cost,
                            'ReconstructionTime': r_time
                        })
                        if ((r_cost['Amount'] == 'Undefined') or
                            (r_time['Amount'] == 'Undefined')):
                            incomplete = True

                else:
                    r_cost, r_time = parse_DS_xml(DSG_xml)
                    CONSEQ = json_output['DSGroups'][DSG_i][
                        'DamageStates'][0]['Consequences']
                    CONSEQ.update({
                        'ReconstructionCost': r_cost,
                        'ReconstructionTime': r_time
                    })
                    if ((r_cost['Amount'] == 'Undefined') or
                        (r_time['Amount'] == 'Undefined')):
                        incomplete = True

            if incomplete:
                json_output['GeneralInformation']['Incomplete'] = True
                incomplete_count += 1

            if DL_dir is None:
                DL_dir = target_dir / "DL json"
                DL_dir.mkdir(exist_ok=True)

            with open(DL_dir / f'{comp_ID}.json', 'w') as f:
                json.dump(json_output, f, indent=2)

        #except:
        #    warnings.warn(UserWarning(
        #        'Error converting data for component {}'.format(comp_ID)))

    # finally, copy the population file
    shutil.copy(
        data_dir / 'population.json',
        target_dir / 'population.json'
    )


def create_HAZUS_EQ_json_files(data_dir, target_dir):
    """
    Create JSON data files from publicly available HAZUS data.

    HAZUS damage and loss information is publicly available in the technical
    manuals. The relevant tables have been converted into a JSON input file
    (hazus_data_eq.json) that is stored in the 'resources/HAZUS MH 2.1' folder
    in the pelicun repo. Here we read that file (or a file of similar format)
    and produce damage and loss data for Fragility Groups in the common
    SimCenter JSON format.

    HAZUS handles damage and losses at the assembly level differentiating only
    structural and two types of non-structural component assemblies. In this
    implementation we consider each of those assemblies a Fragility Group
    and describe their damage and its consequences in a FEMA P58-like framework
    but using the data from the HAZUS Technical Manual.

    Parameters
    ----------
    data_dir: string
        Path to the folder with the hazus_data_eq JSON file.
    target_dir: string
        Path to the folder where the results shall be saved. The population
        distribution file will be saved here, the DL JSON files will be saved
        to a 'DL json' subfolder.

    """

    data_dir = Path(data_dir).resolve()
    target_dir = Path(target_dir).resolve()
    DL_dir = None

    convert_design_level = {
        'High_code'    : 'HC',
        'Moderate_code': 'MC',
        'Low_code'     : 'LC',
        'Pre_code'     : 'PC'
    }

    convert_DS_description = {
        'DS1': 'Slight',
        'DS2': 'Moderate',
        'DS3': 'Extensive',
        'DS4': 'Complete',
        'DS5': 'Collapse',
    }

    # open the raw HAZUS data
    with open(data_dir / 'hazus_data_eq.json', 'r') as f:
        raw_data = json.load(f)

    design_levels = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'].keys())
    building_types = list(
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys())
    occupancy_types = list(raw_data['Structural_Fragility_Groups'][
                               'Reconstruction_cost'].keys())

    S_data = raw_data['Structural_Fragility_Groups']
    NSA_data = raw_data[
        'NonStructural_Acceleration_Sensitive_Fragility_Groups']
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    for ot in occupancy_types:

        # first, structural fragility groups
        for dl in design_levels:
            for bt in building_types:
                if bt in S_data['EDP_limits'][dl].keys():

                    json_output = {}

                    dl_id = 'S-{}-{}-{}'.format(bt,
                                                convert_design_level[dl],
                                                ot)

                    # this might get replaced by a more descriptive name in the future
                    json_output.update({'Name': dl_id})

                    # General info
                    json_output.update({
                        'Directional': False,
                        'GeneralInformation': {
                            'ID'         : dl_id,
                            'Description': dl_id,
                        # this might get replaced by more details in the future
                            # other fields can be added here if needed
                        }
                    })

                    # EDP info
                    json_output.update({
                        'EDP': {
                            'Type': 'Roof Drift Ratio',
                            'Unit': [1, 'rad']
                        }
                    })

                    # Damage State info
                    json_output.update({'DSGroups': []})
                    EDP_lim = S_data['EDP_limits'][dl][bt]

                    for dsg_i in range(4):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : EDP_lim[dsg_i],
                            'Beta'        : S_data['Fragility_beta'][dl],
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [{
                                'Weight'      : 1.0,
                                'Consequences': {},
                                'Description' : 'DS{}'.format(dsg_i + 1),
                            }]
                        })
                        # the last DSG is different
                        if dsg_i == 3:
                            json_output['DSGroups'][-1][
                                'DSGroupType'] = 'MutuallyExclusive'
                            DS5_w = S_data['P_collapse'][bt]
                            json_output['DSGroups'][-1][
                                'DamageStates'].append({
                                'Weight'      : DS5_w,
                                'Consequences': {},
                                'Description' : 'DS5'
                            })
                            json_output['DSGroups'][-1]['DamageStates'][0][
                                'Weight'] = 1.0 - DS5_w

                    for dsg_i, DSG in enumerate(json_output['DSGroups']):
                        base_cost = S_data['Reconstruction_cost'][ot][dsg_i] / 100.
                        base_time = S_data['Reconstruction_time'][ot][dsg_i]

                        for DS in DSG['DamageStates']:
                            DS_id = DS['Description']
                            DS['Consequences'] = {
                                # injury rates are provided in percentages of the population
                                'Injuries'          : [
                                    {'Amount': val / 100.} for val in
                                    S_data['Injury_rates'][DS_id][bt]],
                                # reconstruction cost is provided in percentages of replacement cost
                                'ReconstructionCost': {
                                    "Amount": base_cost,
                                    "CurveType": "N/A",
                                },
                                'ReconstructionTime': {
                                    "Amount": base_time,
                                    "CurveType": "N/A",
                                }
                            }
                            DS['Description'] = convert_DS_description[
                                DS['Description']]

                    # create the DL json directory (if it does not exist)
                    if DL_dir is None:
                        DL_dir = target_dir / "DL json"
                        DL_dir.mkdir(exist_ok=True)

                    with open(DL_dir / f'{dl_id}.json', 'w') as f:
                        json.dump(json_output, f, indent=2)

            # second, nonstructural acceleration sensitive fragility groups
            json_output = {}

            dl_id = 'NSA-{}-{}'.format(convert_design_level[dl], ot)

            # this might get replaced by a more descriptive name in the future
            json_output.update({'Name': dl_id})

            # General info
            json_output.update({
                'Directional': False,
                'GeneralInformation': {
                    'ID'         : dl_id,
                    'Description': dl_id,
                # this might get replaced by more details in the future
                    # other fields can be added here if needed
                }
            })

            # EDP info
            json_output.update({
                'EDP': {
                    'Type': 'Peak Floor Acceleration',
                    'Unit': [1, 'g'],
                    'Offset': NSA_data.get('Offset', 0)
                }
            })

            # Damage State info
            json_output.update({'DSGroups': []})

            for dsg_i in range(4):
                base_cost = NSA_data['Reconstruction_cost'][ot][dsg_i] / 100.

                json_output['DSGroups'].append({
                    'MedianEDP'   : NSA_data['EDP_limits'][dl][dsg_i],
                    'Beta'        : NSA_data['Fragility_beta'],
                    'CurveType'   : 'LogNormal',
                    'DSGroupType' : 'Single',
                    'DamageStates': [{
                        'Weight'      : 1.0,
                        'Consequences': {
                            # reconstruction cost is provided in percentages of replacement cost
                            'ReconstructionCost': {
                                "Amount": base_cost,
                                "CurveType": "N/A",
                            },
                        },
                        'Description' : convert_DS_description[
                            'DS{}'.format(dsg_i + 1)]
                    }]
                })

            # create the DL json directory (if it does not exist)
            if DL_dir is None:
                DL_dir = target_dir / "DL json"
                DL_dir.mkdir(exist_ok=True)

            with open(DL_dir / f'{dl_id}.json', 'w') as f:
                json.dump(json_output, f, indent=2)

                # third, nonstructural drift sensitive fragility groups
        json_output = {}

        dl_id = 'NSD-{}'.format(ot)

        # this might get replaced by a more descriptive name in the future
        json_output.update({'Name': dl_id})

        # General info
        json_output.update({
            'Directional': False,
            'GeneralInformation': {
                'ID'         : dl_id,
                'Description': dl_id,
            # this might get replaced by more details in the future
                # other fields can be added here if needed
            }
        })

        # EDP info
        json_output.update({
            'EDP': {
                'Type': 'Roof Drift Ratio',
                'Unit': [1, 'rad']
            }
        })

        # Damage State info
        json_output.update({'DSGroups': []})

        for dsg_i in range(4):
            base_cost = NSD_data['Reconstruction_cost'][ot][dsg_i] / 100.

            json_output['DSGroups'].append({
                'MedianEDP'   : NSD_data['EDP_limits'][dsg_i],
                'Beta'        : NSD_data['Fragility_beta'],
                'CurveType'   : 'LogNormal',
                'DSGroupType' : 'Single',
                'DamageStates': [{
                    'Weight'      : 1.0,
                    'Consequences': {
                        # reconstruction cost is provided in percentages of replacement cost
                        'ReconstructionCost': {
                            "Amount": base_cost,
                            "CurveType": "N/A",
                        },
                    },
                    'Description' : convert_DS_description[
                        'DS{}'.format(dsg_i + 1)]
                }]
            })

        if DL_dir is None:
            DL_dir = target_dir / "DL json"
            DL_dir.mkdir(exist_ok=True)

        with open(DL_dir / f'{dl_id}.json', 'w') as f:
            json.dump(json_output, f, indent=2)

    # finally, prepare the population distribution data
    PD_data = raw_data['Population_Distribution']

    pop_output = {}
    for ot in PD_data.keys():
        night_ids = raw_data['Parts_of_day']['Nighttime']
        day_ids = raw_data['Parts_of_day']['Daytime']
        commute_ids = raw_data['Parts_of_day']['Commute']

        daily_pop = np.ones(24)
        daily_pop[night_ids] = PD_data[ot][0]
        daily_pop[day_ids] = PD_data[ot][1]
        daily_pop[commute_ids] = PD_data[ot][2]
        daily_pop = list(daily_pop)

        # HAZUS does not introduce monthly and weekend/weekday variation
        pop_output.update({ot: {
            "weekday": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            },
            "weekend": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            }
        }})

    with open(target_dir / 'population.json', 'w') as f:
        json.dump(pop_output, f, indent=2)

def create_HAZUS_EQ_story_json_files(data_dir, target_dir):
    """
    Create JSON data files from publicly available HAZUS data.

    HAZUS damage and loss information is publicly available in the technical
    manuals. The relevant tables have been converted into a JSON input file
    (hazus_data_eq.json) that is stored in the 'resources/HAZUS MH 2.1' folder
    in the pelicun repo. Here we read that file (or a file of similar format)
    and produce damage and loss data for Fragility Groups in the common
    SimCenter JSON format.

    HAZUS handles damage and losses at the assembly level differentiating only
    structural and two types of non-structural component assemblies. In this
    implementation we consider each of those assemblies a Fragility Group
    and describe their damage and its consequences in a FEMA P58-like framework
    but using the data from the HAZUS Technical Manual.

    Parameters
    ----------
    data_dir: string
        Path to the folder with the hazus_data_eq JSON file.
    target_dir: string
        Path to the folder where the results shall be saved. The population
        distribution file will be saved here, the DL JSON files will be saved
        to a 'DL json' subfolder.

    """

    data_dir = Path(data_dir).resolve()
    target_dir = Path(target_dir).resolve()
    DL_dir = None

    convert_design_level = {
        'High_code'    : 'HC',
        'Moderate_code': 'MC',
        'Low_code'     : 'LC',
        'Pre_code'     : 'PC'
    }

    convert_DS_description = {
        'DS1': 'Slight',
        'DS2': 'Moderate',
        'DS3': 'Extensive',
        'DS4': 'Complete',
        'DS5': 'Collapse',
    }

    # open the raw HAZUS data
    with open(data_dir / 'hazus_data_eq.json', 'r') as f:
        raw_data = json.load(f)

    design_levels = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'].keys())
    building_types = list(
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys())
    occupancy_types = list(raw_data['Structural_Fragility_Groups'][
                               'Reconstruction_cost'].keys())

    S_data = raw_data['Structural_Fragility_Groups']
    NSA_data = raw_data[
        'NonStructural_Acceleration_Sensitive_Fragility_Groups']
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    for ot in occupancy_types:

        # first, structural fragility groups
        for dl in design_levels:
            for bt in building_types:
                if bt in S_data['EDP_limits'][dl].keys():

                    json_output = {}

                    dl_id = 'S-{}-{}-{}'.format(bt,
                                                convert_design_level[dl],
                                                ot)

                    # this might get replaced by a more descriptive name in the future
                    json_output.update({'Name': dl_id})

                    # General info
                    json_output.update({
                        'Directional': True,
                        'GeneralInformation': {
                            'ID'         : dl_id,
                            'Description': dl_id,
                        # this might get replaced by more details in the future
                            # other fields can be added here if needed
                        }
                    })

                    # EDP info
                    json_output.update({
                        'EDP': {
                            'Type': 'Story Drift Ratio',
                            'Unit': [1, 'rad']
                        }
                    })

                    # Damage State info
                    json_output.update({'DSGroups': []})
                    EDP_lim = S_data['EDP_limits'][dl][bt]

                    for dsg_i in range(4):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : EDP_lim[dsg_i],
                            'Beta'        : S_data['Fragility_beta'][dl],
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [{
                                'Weight'      : 1.0,
                                'Consequences': {},
                                'Description' : 'DS{}'.format(dsg_i + 1),
                            }]
                        })
                        # the last DSG is different
                        if dsg_i == 3:
                            json_output['DSGroups'][-1][
                                'DSGroupType'] = 'MutuallyExclusive'
                            DS5_w = S_data['P_collapse'][bt]
                            json_output['DSGroups'][-1][
                                'DamageStates'].append({
                                'Weight'      : DS5_w,
                                'Consequences': {},
                                'Description' : 'DS5'
                            })
                            json_output['DSGroups'][-1]['DamageStates'][0][
                                'Weight'] = 1.0 - DS5_w

                    for dsg_i, DSG in enumerate(json_output['DSGroups']):
                        base_cost = S_data['Reconstruction_cost'][ot][dsg_i] / 100.
                        base_time = S_data['Reconstruction_time'][ot][dsg_i]

                        for DS in DSG['DamageStates']:
                            DS_id = DS['Description']
                            DS['Consequences'] = {
                                # injury rates are provided in percentages of the population
                                'Injuries'          : [
                                    {'Amount': val / 100.} for val in
                                    S_data['Injury_rates'][DS_id][bt]],
                                # reconstruction cost is provided in percentages of replacement cost
                                'ReconstructionCost': {
                                    "Amount": base_cost,
                                    "CurveType": "N/A",
                                },
                                'ReconstructionTime': {
                                    "Amount": base_time,
                                    "CurveType": "N/A",
                                }
                            }
                            DS['Description'] = convert_DS_description[
                                DS['Description']]

                    # create the DL json directory (if it does not exist)
                    if DL_dir is None:
                        DL_dir = target_dir / "DL json"
                        DL_dir.mkdir(exist_ok=True)

                    with open(DL_dir / f'{dl_id}.json', 'w') as f:
                        json.dump(json_output, f, indent=2)

            # second, nonstructural acceleration sensitive fragility groups
            json_output = {}

            dl_id = 'NSA-{}-{}'.format(convert_design_level[dl], ot)

            # this might get replaced by a more descriptive name in the future
            json_output.update({'Name': dl_id})

            # General info
            json_output.update({
                'Directional': False,
                'GeneralInformation': {
                    'ID'         : dl_id,
                    'Description': dl_id,
                # this might get replaced by more details in the future
                    # other fields can be added here if needed
                }
            })

            # EDP info
            json_output.update({
                'EDP': {
                    'Type': 'Peak Floor Acceleration',
                    'Unit': [1, 'g'],
                    'Offset': NSA_data.get('Offset', 0)
                }
            })

            # Damage State info
            json_output.update({'DSGroups': []})

            for dsg_i in range(4):
                base_cost = NSA_data['Reconstruction_cost'][ot][dsg_i] / 100.

                json_output['DSGroups'].append({
                    'MedianEDP'   : NSA_data['EDP_limits'][dl][dsg_i],
                    'Beta'        : NSA_data['Fragility_beta'],
                    'CurveType'   : 'LogNormal',
                    'DSGroupType' : 'Single',
                    'DamageStates': [{
                        'Weight'      : 1.0,
                        'Consequences': {
                            # reconstruction cost is provided in percentages of replacement cost
                            'ReconstructionCost': {
                                "Amount": base_cost,
                                "CurveType": "N/A",
                            },
                        },
                        'Description' : convert_DS_description[
                            'DS{}'.format(dsg_i + 1)]
                    }]
                })

            # create the DL json directory (if it does not exist)
            if DL_dir is None:
                DL_dir = target_dir / "DL json"
                DL_dir.mkdir(exist_ok=True)

            with open(DL_dir / f'{dl_id}.json', 'w') as f:
                json.dump(json_output, f, indent=2)

                # third, nonstructural drift sensitive fragility groups
        json_output = {}

        dl_id = 'NSD-{}'.format(ot)

        # this might get replaced by a more descriptive name in the future
        json_output.update({'Name': dl_id})

        # General info
        json_output.update({
            'Directional': True,
            'GeneralInformation': {
                'ID'         : dl_id,
                'Description': dl_id,
            # this might get replaced by more details in the future
                # other fields can be added here if needed
            }
        })

        # EDP info
        json_output.update({
            'EDP': {
                'Type': 'Story Drift Ratio',
                'Unit': [1, 'rad']
            }
        })

        # Damage State info
        json_output.update({'DSGroups': []})

        for dsg_i in range(4):
            base_cost = NSD_data['Reconstruction_cost'][ot][dsg_i] / 100.

            json_output['DSGroups'].append({
                'MedianEDP'   : NSD_data['EDP_limits'][dsg_i],
                'Beta'        : NSD_data['Fragility_beta'],
                'CurveType'   : 'LogNormal',
                'DSGroupType' : 'Single',
                'DamageStates': [{
                    'Weight'      : 1.0,
                    'Consequences': {
                        # reconstruction cost is provided in percentages of replacement cost
                        'ReconstructionCost': {
                            "Amount": base_cost,
                            "CurveType": "N/A",
                        },
                    },
                    'Description' : convert_DS_description[
                        'DS{}'.format(dsg_i + 1)]
                }]
            })

        if DL_dir is None:
            DL_dir = target_dir / "DL json"
            DL_dir.mkdir(exist_ok=True)

        with open(DL_dir / f'{dl_id}.json', 'w') as f:
            json.dump(json_output, f, indent=2)

    # finally, prepare the population distribution data
    PD_data = raw_data['Population_Distribution']

    pop_output = {}
    for ot in PD_data.keys():
        night_ids = raw_data['Parts_of_day']['Nighttime']
        day_ids = raw_data['Parts_of_day']['Daytime']
        commute_ids = raw_data['Parts_of_day']['Commute']

        daily_pop = np.ones(24)
        daily_pop[night_ids] = PD_data[ot][0]
        daily_pop[day_ids] = PD_data[ot][1]
        daily_pop[commute_ids] = PD_data[ot][2]
        daily_pop = list(daily_pop)

        # HAZUS does not introduce monthly and weekend/weekday variation
        pop_output.update({ot: {
            "weekday": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            },
            "weekend": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            }
        }})

    with open(target_dir / 'population.json', 'w') as f:
        json.dump(pop_output, f, indent=2)

def create_HAZUS_EQ_PGA_json_files(data_dir, target_dir):
    """
    Create JSON data files from publicly available HAZUS data.

    HAZUS damage and loss information is publicly available in the technical
    manuals. The relevant tables have been converted into a JSON input file
    (hazus_data_eq.json) that is stored in the 'resources/HAZUS MH 2.1
    earthquake PGA' folder in the pelicun repo. Here we read that file (or a
    file of similar format) and produce damage and loss data for Fragility
    Groups in the common SimCenter JSON format.

    HAZUS handles damage and losses at the assembly level. This method assumes
    that we use the fragility curves controlled by equivalent PGA from HAZUS.
    Those curves are only available for structural component assemblies. A
    Fragility Group is created for each of the structural configurations that
    describes their damage and its consequences in a FEMA P58-like framework
    but using the data from the HAZUS Technical Manual.

    Parameters
    ----------
    data_dir: string
        Path to the folder with the hazus_data_eq JSON file.
    target_dir: string
        Path to the folder where the results shall be saved. The population
        distribution file will be saved here, the DL JSON files will be saved
        to a 'DL json' subfolder.

    """

    data_dir = Path(data_dir).resolve()
    target_dir = Path(target_dir).resolve()
    DL_dir = None

    convert_design_level = {
        'High_code'    : 'HC',
        'Moderate_code': 'MC',
        'Low_code'     : 'LC',
        'Pre_code'     : 'PC'
    }

    convert_DS_description = {
        'DS1': 'Slight',
        'DS2': 'Moderate',
        'DS3': 'Extensive',
        'DS4': 'Complete',
        'DS5': 'Collapse',
    }

    convert_GF_description = {
        'H_S': 'Lateral spreading, shallow foundation',
        'H_D': 'Lateral spreading, deep foundation',
        'V_S': 'Ground settlement, shallow foundation',
        'V_D': 'Ground settlement, deep foundation',
    }

    # open the raw HAZUS data
    with open(data_dir / 'hazus_data_eq.json', 'r') as f:
        raw_data = json.load(f)

    # First, the ground shaking fragilities
    S_data = raw_data['Structural_Fragility_Groups']

    design_levels = list(S_data['EDP_limits'].keys())
    building_types = list(S_data['P_collapse'].keys())
    occupancy_types = list(S_data['Reconstruction_cost'].keys())

    for ot in occupancy_types:
        for dl in design_levels:
            for bt in building_types:
                if bt in S_data['EDP_limits'][dl].keys():

                    json_output = {}

                    dl_id = 'S-{}-{}-{}'.format(bt,
                                                convert_design_level[dl],
                                                ot)

                    # this might get replaced by a more descriptive name in the future
                    json_output.update({'Name': dl_id})

                    # General info
                    json_output.update({
                        'Directional': True,
                        'GeneralInformation': {
                            'ID'         : dl_id,
                            'Description': dl_id,
                        # this might get replaced by more details in the future
                            # other fields can be added here if needed
                        }
                    })

                    # EDP info
                    json_output.update({
                        'EDP': {
                            'Type': 'Peak Ground Acceleration',
                            'Unit': [1, 'g']
                        }
                    })

                    # Damage State info
                    json_output.update({'DSGroups': []})
                    EDP_lim = S_data['EDP_limits'][dl][bt]

                    for dsg_i in range(4):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : EDP_lim[dsg_i],
                            'Beta'        : S_data['Fragility_beta'][dl],
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [{
                                'Weight'      : 1.0,
                                'Consequences': {},
                                'Description' : 'DS{}'.format(dsg_i + 1),
                            }]
                        })
                        # the last DSG is different
                        if dsg_i == 3:
                            json_output['DSGroups'][-1][
                                'DSGroupType'] = 'MutuallyExclusive'
                            DS5_w = S_data['P_collapse'][bt]
                            json_output['DSGroups'][-1][
                                'DamageStates'].append({
                                'Weight'      : DS5_w,
                                'Consequences': {},
                                'Description' : 'DS5'
                            })
                            json_output['DSGroups'][-1]['DamageStates'][0][
                                'Weight'] = 1.0 - DS5_w

                    for dsg_i, DSG in enumerate(json_output['DSGroups']):
                        base_cost = S_data['Reconstruction_cost'][ot][dsg_i] / 100.
                        base_time = S_data['Reconstruction_time'][ot][dsg_i]

                        for DS in DSG['DamageStates']:
                            DS_id = DS['Description']
                            DS['Consequences'] = {
                                # injury rates are provided in percentages of the population
                                'Injuries'          : [
                                    {'Amount': val / 100.} for val in
                                    S_data['Injury_rates'][DS_id][bt]],
                                # reconstruction cost is provided in percentages of replacement cost
                                'ReconstructionCost': {
                                    "Amount": base_cost,
                                    "CurveType": "N/A",
                                },
                                'ReconstructionTime': {
                                    "Amount": base_time,
                                    "CurveType": "N/A",
                                }
                            }
                            DS['Description'] = convert_DS_description[
                                DS['Description']]

                    if DL_dir is None:
                        DL_dir = target_dir / "DL json"
                        DL_dir.mkdir(exist_ok=True)

                    with open(DL_dir / f'{dl_id}.json', 'w') as f:
                        json.dump(json_output, f, indent=2)

    # Second, the ground failure fragilities
    L_data = raw_data['Ground_Failure']

    ground_failure_types = list(L_data['EDP_limits'].keys())

    for gf in ground_failure_types:
        for bt in building_types:
            if bt in L_data['P_collapse'].keys():

                json_output = {}

                dl_id = f'GF-{gf}-{bt}'

                # this might get replaced by a more descriptive name in the future
                json_output.update({'Name': dl_id})

                # General info
                json_output.update({
                    'Directional': True,
                    'GeneralInformation': {
                        'ID'         : dl_id,
                        'Description': convert_GF_description[gf],
                    # this might get replaced by more details in the future
                        # other fields can be added here if needed
                    }
                })

                # EDP info
                json_output.update({
                    'EDP': {
                        'Type': 'Permanent Ground Deformation',
                        'Unit': [1, 'in']
                    }
                })

                # Damage State info
                json_output.update({'DSGroups': []})
                EDP_lim = L_data['EDP_limits'][gf]
                beta_vals = L_data['Fragility_beta'][gf]

                for dsg_i in range(2):
                    json_output['DSGroups'].append({
                        'MedianEDP'   : EDP_lim[dsg_i],
                        'Beta'        : beta_vals[dsg_i],
                        'CurveType'   : 'LogNormal',
                        'DSGroupType' : 'Single',
                        'DamageStates': [{
                            'Weight'      : 1.0,
                            'Consequences': {},
                            'Description' : 'DS{}'.format(dsg_i + 1),
                        }]
                    })
                    # the last DSG is different
                    if dsg_i == 1:
                        json_output['DSGroups'][-1][
                            'DSGroupType'] = 'MutuallyExclusive'
                        DS5_w = L_data['P_collapse'][bt]
                        json_output['DSGroups'][-1]['DamageStates'].append({
                            'Weight'      : DS5_w,
                            'Consequences': {},
                            'Description' : 'DS5'
                        })
                        json_output['DSGroups'][-1]['DamageStates'][0]['Weight'] = 1.0 - DS5_w

                if DL_dir is None:
                    DL_dir = target_dir / "DL json"
                    DL_dir.mkdir(exist_ok=True)

                # consequences are handled through propagating damage to other components
                with open(DL_dir / f'{dl_id}.json', 'w') as f:
                    json.dump(json_output, f, indent=2)

    # prepare the population distribution data
    PD_data = raw_data['Population_Distribution']

    pop_output = {}
    for ot in PD_data.keys():
        night_ids = raw_data['Parts_of_day']['Nighttime']
        day_ids = raw_data['Parts_of_day']['Daytime']
        commute_ids = raw_data['Parts_of_day']['Commute']

        daily_pop = np.ones(24)
        daily_pop[night_ids] = PD_data[ot][0]
        daily_pop[day_ids] = PD_data[ot][1]
        daily_pop[commute_ids] = PD_data[ot][2]
        daily_pop = list(daily_pop)

        # HAZUS does not introduce monthly and weekend/weekday variation
        pop_output.update({ot: {
            "weekday": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            },
            "weekend": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            }
        }})

    with open(target_dir / 'population.json', 'w') as f:
        json.dump(pop_output, f, indent=2)

def create_HAZUS_HU_json_files(data_dir, target_dir):
    """
    Create JSON data files from publicly available HAZUS data.

    HAZUS damage and loss information is publicly available in the technical
    manuals and the HAZUS software tool. The relevant data have been collected
    in a series of Excel files (e.g., hu_Wood.xlsx) that are stored in the
    'resources/HAZUS MH 2.1 hurricane' folder in the pelicun repo. Here we read
    that file (or a file of similar format) and produce damage and loss data
    for Fragility Groups in the common SimCenter JSON format.

    The HAZUS hurricane methodology handles damage and losses at the assembly
    level. In this implementation each building is represented by one Fragility
    Group that describes the damage states and their consequences in a FEMA
    P58-like framework but using the data from the HAZUS Technical Manual.

    Note: HAZUS calculates lossess independently of damage using peak wind gust
    speed as a controlling variable. We fitted a model to the curves in HAZUS
    that assigns losses to each damage state and determines losses as a function
    of building damage. Results shall be in good agreement with those of HAZUS
    for the majority of building configurations. Exceptions and more details
    are provided in the ... section of the documentation.

    Parameters
    ----------
    data_dir: string
        Path to the folder with the hazus_data_eq JSON file.
    target_dir: string
        Path to the folder where the results shall be saved. The population
        distribution file will be saved here, the DL JSON files will be saved
        to a 'DL json' subfolder.

    """

    data_dir = Path(data_dir).resolve()
    target_dir = Path(target_dir).resolve()
    DL_dir = None

    # open the raw HAZUS data
    df_wood = pd.read_excel(data_dir / 'hu_Wood.xlsx')

    # some general formatting to make file name generation easier
    df_wood['shutters'] = df_wood['shutters'].astype(int)
    df_wood['terr_rough'] = (df_wood['terr_rough'] * 100.).astype(int)

    convert_building_type = {
        'WSF1' : 'Wood Single-Family Homes 1 story',
        'WSF2' : 'Wood Single-Family Homes 2+ stories',
        'WMUH1': 'Wood Multi-Unit or Hotel or Motel 1 story',
        'WMUH2': 'Wood Multi-Unit or Hotel or Motel 2 stories',
        'WMUH3': 'Wood Multi-Unit or Hotel or Motel 3+ stories',
    }

    convert_bldg_char_names = {
        'roof_shape'     : 'Roof Shape',
        'sec_water_res'  : 'Secondary Water Resistance',
        'roof_deck_attch': 'Roof Deck Attachment',
        'roof_wall_conn' : 'Roof-Wall Connection',
        'garage'         : 'Garage',
        'shutters'       : 'Shutters',
        'roof_cover'     : 'Roof Cover Type',
        'roof_quality'   : 'Roof Cover Quality',
        'terr_rough'     : 'Terrain',
    }

    convert_bldg_chars = {
        1      : True,
        0      : False,

        'gab'  : 'gable',
        'hip'  : 'hip',
        'flt'  : 'flat',

        '6d'   : '6d @ 6"/12"',
        '8d'   : '8d @ 6"/12"',
        '6s'   : '6d/8d mix @ 6"/6"',
        '8s'   : '8D @ 6"/6"',

        'tnail': 'Toe-nail',
        'strap': 'Strap',

        'no'   : 'None',
        'wkd'  : 'Weak',
        'std'  : 'Standard',
        'sup'  : 'SFBC 1994',

        'bur'  : 'BUR',
        'spm'  : 'SPM',

        'god'  : 'Good',
        'por'  : 'Poor',

        3      : 'Open',
        15     : 'Light Suburban',
        35     : 'Suburban',
        70     : 'Light Trees',
        100    : 'Trees',
    }

    convert_dist = {
        'normal'   : 'Normal',
        'lognormal': 'LogNormal',
    }

    convert_ds = {
        1: 'Minor',
        2: 'Moderate',
        3: 'Severe',
        4: 'Destruction',
    }

    for index, row in df_wood.iterrows():
        #print(index, end=' ')

        json_output = {}

        # define the name of the building damage and loss configuration
        bldg_type = row["bldg_type"]

        if bldg_type[:3] == "WSF":
            cols_of_interest = ["bldg_type", "roof_shape", "sec_water_res",
                                "roof_deck_attch", "roof_wall_conn", "garage",
                                "shutters", "terr_rough"]
        elif bldg_type[:4] == "WMUH":
            cols_of_interest = ["bldg_type", "roof_shape", "roof_cover",
                                "roof_quality", "sec_water_res",
                                "roof_deck_attch", "roof_wall_conn", "shutters",
                                "terr_rough"]

        bldg_chars = row[cols_of_interest]

        if np.isnan(bldg_chars["sec_water_res"]):
            bldg_chars["sec_water_res"] = 'null'
        else:
            bldg_chars["sec_water_res"] = int(bldg_chars["sec_water_res"])

        if bldg_type[:4] == "WMUH":
            if (not isinstance(bldg_chars["roof_cover"],str)
                and np.isnan(bldg_chars["roof_cover"])):
                bldg_chars["roof_cover"] = 'null'
            if (not isinstance(bldg_chars["roof_quality"], str)
                and np.isnan(bldg_chars["roof_quality"])):
                bldg_chars["roof_quality"] = 'null'

        dl_id = "_".join(bldg_chars.astype(str))

        json_output.update({'Name': dl_id})

        # general information
        json_output.update({
            'GeneralInformation': {
                'ID'           : str(index),
                'Description'  : dl_id,
                'Building type': convert_building_type[bldg_type],
            }
        })
        for col in cols_of_interest:
            if (col != 'bldg_type') and (bldg_chars[col] != 'null'):
                json_output['GeneralInformation'].update({
                    convert_bldg_char_names[col]: convert_bldg_chars[
                        bldg_chars[col]]
                })

        # EDP info
        json_output.update({
            'EDP': {
                'Type': 'Peak Gust Wind Speed',
                'Unit': [1, 'mph']
            }
        })

        # Damage States
        json_output.update({'DSGroups': []})

        for dsg_i in range(1, 5):
            json_output['DSGroups'].append({
                'MedianEDP'   : row['DS{}_mu'.format(dsg_i)],
                'Beta'        : row['DS{}_sig'.format(dsg_i)],
                'CurveType'   : convert_dist[row['DS{}_dist'.format(dsg_i)]],
                'DSGroupType' : 'Single',
                'DamageStates': [{
                    'Weight'      : 1.0,
                    'Consequences': {
                        'ReconstructionCost': {
                            'Amount': row[
                                'L{}'.format(dsg_i)] if dsg_i < 4 else 1.0
                        }
                    },
                    'Description' : convert_ds[dsg_i]
                }]
            })

        if DL_dir is None:
            DL_dir = target_dir / "DL json"
            DL_dir.mkdir(exist_ok=True)

        with open(DL_dir / f'{dl_id}.json', 'w') as f:
            json.dump(json_output, f, indent=2)

