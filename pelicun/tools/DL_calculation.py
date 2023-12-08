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
# Joanna J. Zou

from time import gmtime
from time import strftime
import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pelicun.base import convert_to_MultiIndex
from pelicun.base import convert_to_SimpleIndex
from pelicun.base import describe
from pelicun.base import EDP_to_demand_type
from pelicun.file_io import load_data
from pelicun.assessment import Assessment


# this is exceptional code
# pylint: disable=consider-using-namedtuple-or-dataclass
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-branches


def log_msg(msg):

    formatted_msg = f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} {msg}'

    print(formatted_msg)


log_msg('First line of DL_calculation')

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

idx = pd.IndexSlice

damage_processes = {
    'FEMA P-58': {
        "1_collapse": {
            "DS1": "ALL_NA"
        },
        "2_excessiveRID": {
            "DS1": "irreparable_DS1"
        }
    },

    'Hazus Earthquake': {
        "1_STR": {
            "DS5": "collapse_DS1"
        },
        "2_LF": {
            "DS5": "collapse_DS1"
        },
        "3_collapse": {
            "DS1": "ALL_NA"
        },
        "4_excessiveRID": {
            "DS1": "irreparable_DS1"
        }
    }
}

default_DBs = {
    'fragility': {
        'FEMA P-58': 'fragility_DB_FEMA_P58_2nd.csv',
        'Hazus Earthquake': 'fragility_DB_HAZUS_EQ.csv'
    },
    'repair': {
        'FEMA P-58': 'bldg_repair_DB_FEMA_P58_2nd.csv',
        'Hazus Earthquake': 'bldg_repair_DB_HAZUS_EQ.csv'
    }

}


def add_units(raw_demands, length_unit):

    demands = raw_demands.T

    demands.insert(0, "Units", np.nan)

    if length_unit == 'in':
        length_unit = 'inch'

    demands = convert_to_MultiIndex(demands, axis=0).sort_index(axis=0).T

    if demands.columns.nlevels == 4:
        DEM_level = 1
    else:
        DEM_level = 0

    # drop demands with no EDP type identified
    demands.drop(demands.columns[
        demands.columns.get_level_values(DEM_level) == ''],
                 axis=1, inplace=True)

    # assign units
    demand_cols = demands.columns.get_level_values(DEM_level)

    # remove additional info from demand names
    demand_cols = [d.split('_')[0] for d in demand_cols]

    # acceleration
    acc_EDPs = ['PFA', 'PGA', 'SA']
    EDP_mask = np.isin(demand_cols, acc_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = length_unit+'ps2'

    # speed
    speed_EDPs = ['PFV', 'PWS', 'PGV', 'SV']
    EDP_mask = np.isin(demand_cols, speed_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = length_unit+'ps'

    # displacement
    disp_EDPs = ['PFD', 'PIH', 'SD', 'PGD']
    EDP_mask = np.isin(demand_cols, disp_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = length_unit

    # rotation
    rot_EDPs = ['PID', 'PRD', 'DWD', 'RDR', 'PMD', 'RID']
    EDP_mask = np.isin(demand_cols, rot_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = 'rad'

    # convert back to simple header and return the DF
    return convert_to_SimpleIndex(demands, axis=1)


def run_pelicun(config_path):

    # Initial setup -----------------------------------------------------------

    # get the absolute path to the config file
    config_path = Path(config_path).resolve()

    # open the file and load the contents
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    DL_config = config.get('DL', None)
    if DL_config is None:

        log_msg("Damage and Loss configuration missing from input file. "
                "Terminating analysis.")

        return -1

    asset_config = DL_config.get('Asset', None)
    demand_config = DL_config.get('Demands', None)
    damage_config = DL_config.get('Damage', None)
    loss_config = DL_config.get('Losses', None)
    # out_config = DL_config.get('Outputs', None)

    out_config = {
        'Demand': {
            'Sample': True,
            'Statistics': True
        },
        'Asset': {
            'Sample': True,
            'Statistics': True
        },
        'Damage': {
            'Sample': True,
            'Statistics': True,
            'GroupedSample': True,
            'GroupedStatistics': True
        },
        'Loss': {
            'BldgRepair': {
                'Sample': True,
                'Statistics': True,
                'GroupedSample': True,
                'GroupedStatistics': True,
                'AggregateSample': True,
                'AggregateStatistics': True
            }
        }
    }

    if asset_config is None:
        log_msg("Asset configuration missing. Terminating analysis.")
        return -1

    if demand_config is None:
        log_msg("Demand configuration missing. Terminating analysis.")
        return -1

    # get the length unit from the config file
    try:
        length_unit = config['GeneralInformation']['units']['length']
    except KeyError:
        log_msg(
            "No default length unit provided in the input file. "
            "Terminating analysis. ")

        return -1

    # if out_config is None:
    #    log_msg("Output configuration missing. Terminating analysis.")
    #    return -1

    # initialize the Pelicun Assessement
    options = DL_config.get("Options", {})
    options.update({
        "LogFile": "pelicun_log.txt",
        "Verbose": True
        })

    PAL = Assessment(options)

    # Demand Assessment -----------------------------------------------------------

    # check if there is a demand file location specified in the config file
    if demand_config.get('DemandFilePath', False):

        demand_path = Path(demand_config['DemandFilePath']).resolve()

    else:
        # otherwise assume that there is a response.csv file next to the config file
        demand_path = config_path.parent/'response.csv'

    # try to load the demands
    raw_demands = pd.read_csv(demand_path, index_col=0)

    # remove excessive demands that are considered collapses, if needed
    if demand_config.get('CollapseLimits', False):

        raw_demands = convert_to_MultiIndex(raw_demands, axis=1)

        DEM_to_drop = np.full(raw_demands.shape[0], False)

        for DEM_type, limit in demand_config['CollapseLimits'].items():

            if raw_demands.columns.nlevels == 4:
                DEM_to_drop += raw_demands.loc[
                               :, idx[:, DEM_type, :, :]].max(axis=1) > float(limit)

            else:
                DEM_to_drop += raw_demands.loc[
                               :, idx[DEM_type, :, :]].max(axis=1) > float(limit)

        raw_demands = raw_demands.loc[~DEM_to_drop, :]

        log_msg(f"{np.sum(DEM_to_drop)} realizations removed from the demand "
                f"input because they exceed the collapse limit. The remaining "
                f"sample size: {raw_demands.shape[0]}")

    # add units to the demand data if needed
    if "Units" not in raw_demands.index:

        demands = add_units(raw_demands, length_unit)

    else:
        demands = raw_demands

    # get the calibration information
    if demand_config.get('Calibration', False):

        # load the available demand sample
        PAL.demand.load_sample(demands)

        # then use it to calibrate the demand model
        PAL.demand.calibrate_model(demand_config['Calibration'])

        # and generate a new demand sample
        sample_size = int(demand_config['SampleSize'])

        PAL.demand.generate_sample({"SampleSize": sample_size})

    # get the generated demand sample
    demand_sample, demand_units = PAL.demand.save_sample(save_units=True)

    demand_sample = pd.concat([demand_sample, demand_units.to_frame().T])

    # get residual drift estimates, if needed
    if demand_config.get('InferResidualDrift', False):

        RID_config = demand_config['InferResidualDrift']

        if RID_config['method'] == 'FEMA P-58':

            RID_list = []
            PID = demand_sample['PID'].copy()
            PID.drop('Units', inplace=True)
            PID = PID.astype(float)

            for direction, delta_yield in RID_config.items():

                if direction == 'method':
                    continue

                RID = PAL.demand.estimate_RID(
                    PID.loc[:, idx[:, direction]],
                    {'yield_drift': float(delta_yield)})

                RID_list.append(RID)

            RID = pd.concat(RID_list, axis=1)
            RID_units = pd.Series(['rad', ]*RID.shape[1], index=RID.columns,
                                  name='Units')
        RID_sample = pd.concat([RID, RID_units.to_frame().T])

        demand_sample = pd.concat([demand_sample, RID_sample], axis=1)

    # add a constant zero demand
    demand_sample[('ONE', '0', '1')] = np.ones(demand_sample.shape[0])
    demand_sample.loc['Units', ('ONE', '0', '1')] = 'ea'

    PAL.demand.load_sample(convert_to_SimpleIndex(demand_sample, axis=1))

    # save results
    if out_config.get('Demand', False):

        out_reqs = [out if val else "" for out, val in out_config['Demand'].items()]

        if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):
            demand_sample = PAL.demand.save_sample()

            if 'Sample' in out_reqs:
                demand_sample_s = convert_to_SimpleIndex(demand_sample, axis=1)
                demand_sample_s.to_csv("DEM_sample.zip",
                                       index_label=demand_sample_s.columns.name,
                                       compression=dict(
                                           method='zip',
                                           archive_name='DEM_sample.csv'))

            if 'Statistics' in out_reqs:
                demand_stats = convert_to_SimpleIndex(
                    describe(demand_sample), axis=1)
                demand_stats.to_csv("DEM_stats.csv",
                                    index_label=demand_stats.columns.name)

    # Asset Definition ------------------------------------------------------------

    # set the number of stories
    if asset_config.get('NumberOfStories', False):
        PAL.stories = int(asset_config['NumberOfStories'])

    # load a component model and generate a sample
    if asset_config.get('ComponentAssignmentFile', False):

        cmp_marginals = pd.read_csv(asset_config['ComponentAssignmentFile'],
                                    index_col=0, encoding_errors='replace')

        # add a component to support collapse calculation
        cmp_marginals.loc['collapse', 'Units'] = 'ea'
        cmp_marginals.loc['collapse', 'Location'] = 0
        cmp_marginals.loc['collapse', 'Direction'] = 1
        cmp_marginals.loc['collapse', 'Theta_0'] = 1.0

        # add components to support irreparable damage calculation
        if 'IrreparableDamage' in damage_config.keys():

            DEM_types = demand_sample.columns.unique(level=0)
            if 'RID' in DEM_types:

                # excessive RID is added on every floor to detect large RIDs
                cmp_marginals.loc['excessiveRID', 'Units'] = 'ea'

                locs = demand_sample['RID'].columns.unique(level=0)
                cmp_marginals.loc['excessiveRID', 'Location'] = ','.join(locs)

                dirs = demand_sample['RID'].columns.unique(level=1)
                cmp_marginals.loc['excessiveRID', 'Direction'] = ','.join(dirs)

                cmp_marginals.loc['excessiveRID', 'Theta_0'] = 1.0

                # irreparable is a global component to recognize is any of the
                # excessive RIDs were triggered
                cmp_marginals.loc['irreparable', 'Units'] = 'ea'
                cmp_marginals.loc['irreparable', 'Location'] = 0
                cmp_marginals.loc['irreparable', 'Direction'] = 1
                cmp_marginals.loc['irreparable', 'Theta_0'] = 1.0

            else:
                log_msg('WARNING: No residual interstory drift ratio among'
                        'available demands. Irreparable damage cannot be '
                        'evaluated.')

        # load component model
        PAL.asset.load_cmp_model({'marginals': cmp_marginals})

        # generate component quantity sample
        PAL.asset.generate_cmp_sample()

    # if requested, load the quantity sample from a file
    elif asset_config.get('ComponentSampleFile', False):
        PAL.asset.load_cmp_sample(asset_config['ComponentSampleFile'])

    cmp_sample = PAL.asset.save_cmp_sample()

    # if requested, save results
    if out_config.get('Asset', False):

        out_reqs = [out if val else "" for out, val in out_config['Asset'].items()]

        if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):

            if 'Sample' in out_reqs:
                cmp_sample_s = convert_to_SimpleIndex(cmp_sample, axis=1)
                cmp_sample_s.to_csv("CMP_sample.zip",
                                    index_label=cmp_sample_s.columns.name,
                                    compression=dict(method='zip',
                                                     archive_name='CMP_sample.csv'))

            if 'Statistics' in out_reqs:
                cmp_stats = convert_to_SimpleIndex(describe(cmp_sample), axis=1)
                cmp_stats.to_csv(
                    "CMP_stats.csv",
                    index_label=cmp_stats.columns.name)

    # Damage Assessment -----------------------------------------------------------

    # if a damage assessment is requested
    if damage_config is not None:

        # load the fragility information
        if asset_config['ComponentDatabase'] != "User Defined":
            fragility_db = (
                'PelicunDefault/' +
                default_DBs['fragility'][asset_config['ComponentDatabase']])

        else:
            fragility_db = asset_config['ComponentDatabasePath']

        # prepare additional fragility data

        # get the database header from the default P58 db
        P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')

        adf = pd.DataFrame(columns=P58_data.columns)

        if 'CollapseFragility' in damage_config.keys():

            coll_config = damage_config['CollapseFragility']

            adf.loc['collapse', ('Demand', 'Directional')] = 1
            adf.loc['collapse', ('Demand', 'Offset')] = 0

            coll_DEM = coll_config["DemandType"]

            if '_' in coll_DEM:
                coll_DEM, coll_DEM_spec = coll_DEM.split('_')
            else:
                coll_DEM_spec = None

            coll_DEM_name = None
            for demand_name, demand_short in EDP_to_demand_type.items():

                if demand_short == coll_DEM:
                    coll_DEM_name = demand_name
                    break

            if coll_DEM_name is None:
                return -1

            if coll_DEM_spec is None:
                adf.loc['collapse', ('Demand', 'Type')] = coll_DEM_name

            else:
                adf.loc['collapse', ('Demand', 'Type')] = \
                    f'{coll_DEM_name}|{coll_DEM_spec}'

            coll_DEM_unit = add_units(
                pd.DataFrame(columns=[f'{coll_DEM}-1-1', ]),
                length_unit).iloc[0, 0]

            adf.loc['collapse', ('Demand', 'Unit')] = coll_DEM_unit

            adf.loc['collapse', ('LS1', 'Family')] = (
                coll_config.get('CapacityDistribution', ""))

            adf.loc['collapse', ('LS1', 'Theta_0')] = (
                coll_config.get('CapacityMedian', ""))

            adf.loc['collapse', ('LS1', 'Theta_1')] = (
                coll_config.get('Theta_1', ""))

            adf.loc['collapse', 'Incomplete'] = 0

        else:

            # add a placeholder collapse fragility that will never trigger
            # collapse, but allow damage processes to work with collapse

            adf.loc['collapse', ('Demand', 'Directional')] = 1
            adf.loc['collapse', ('Demand', 'Offset')] = 0
            adf.loc['collapse', ('Demand', 'Type')] = 'One'
            adf.loc['collapse', ('Demand', 'Unit')] = 'ea'
            adf.loc['collapse', ('LS1', 'Theta_0')] = 2.0
            adf.loc['collapse', 'Incomplete'] = 0

        if 'IrreparableDamage' in damage_config.keys():

            irrep_config = damage_config['IrreparableDamage']

            # add excessive RID fragility according to settings provided in the
            # input file
            adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
            adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
            adf.loc['excessiveRID',
                    ('Demand', 'Type')] = 'Residual Interstory Drift Ratio'

            adf.loc['excessiveRID', ('Demand', 'Unit')] = 'rad'
            adf.loc['excessiveRID',
                    ('LS1', 'Theta_0')] = irrep_config['DriftCapacityMedian']

            adf.loc['excessiveRID',
                    ('LS1', 'Family')] = "lognormal"

            adf.loc['excessiveRID',
                    ('LS1', 'Theta_1')] = irrep_config['DriftCapacityLogStd']

            adf.loc['excessiveRID', 'Incomplete'] = 0

            # add a placeholder irreparable fragility that will never trigger
            # damage, but allow damage processes to aggregate excessiveRID here
            adf.loc['irreparable', ('Demand', 'Directional')] = 1
            adf.loc['irreparable', ('Demand', 'Offset')] = 0
            adf.loc['irreparable', ('Demand', 'Type')] = 'One'
            adf.loc['irreparable', ('Demand', 'Unit')] = 'ea'
            adf.loc['irreparable', ('LS1', 'Theta_0')] = 2.0
            adf.loc['irreparable', 'Incomplete'] = 0

        PAL.damage.load_damage_model([fragility_db, adf])

        # load the damage process if needed
        dmg_process = None
        if damage_config.get('DamageProcess', False):

            dp_approach = damage_config['DamageProcess']

            if dp_approach in damage_processes:
                dmg_process = damage_processes[dp_approach]

                # For Hazus Earthquake, we need to specify the component ids
                if dp_approach == 'Hazus Earthquake':

                    cmp_list = cmp_sample.columns.unique(level=0)

                    cmp_map = {
                        'STR': '',
                        'LF': '',
                        'NSA': ''
                    }

                    for cmp in cmp_list:
                        for cmp_type in cmp_map:

                            if cmp_type + '.' in cmp:
                                cmp_map[cmp_type] = cmp

                    new_dmg_process = dmg_process.copy()
                    for source_cmp, action in dmg_process.items():

                        # first, look at the source component id
                        new_source = None
                        for cmp_type, cmp_id in cmp_map.items():

                            if ((cmp_type in source_cmp) and (cmp_id != '')):
                                new_source = source_cmp.replace(cmp_type, cmp_id)
                                break

                        if new_source is not None:
                            new_dmg_process[new_source] = action
                            del new_dmg_process[source_cmp]
                        else:
                            new_source = source_cmp

                        # then, look at the target component ids
                        for ds_i, target_vals in action.items():

                            if isinstance(target_vals, str):

                                for cmp_type, cmp_id in cmp_map.items():
                                    if ((cmp_type in target_vals) and (
                                            cmp_id != '')):

                                        target_vals = target_vals.replace(
                                            cmp_type, cmp_id)

                                new_target_vals = target_vals

                            else:
                                # we assume that target_vals is a list of str
                                new_target_vals = []

                                for target_val in target_vals:
                                    for cmp_type, cmp_id in cmp_map.items():
                                        if ((cmp_type in target_val) and (
                                                cmp_id != '')):

                                            target_val = target_val.replace(
                                                cmp_type, cmp_id)

                                    new_target_vals.append(target_val)

                            new_dmg_process[new_source][ds_i] = new_target_vals

                    dmg_process = new_dmg_process

            elif dp_approach == "User Defined":

                # load the damage process from a file
                with open(damage_config['DamageProcessFilePath'], 'r',
                          encoding='utf-8') as f:
                    dmg_process = json.load(f)

            else:
                log_msg(f"Prescribed Damage Process not recognized: "
                        f"{dp_approach}")

        # calculate damages
        PAL.damage.calculate(dmg_process=dmg_process)

        # if requested, save results
        if out_config.get('Damage', False):

            out_reqs = [out if val else ""
                        for out, val in out_config['Damage'].items()]

            if np.any(np.isin(['Sample', 'Statistics',
                               'GroupedSample', 'GroupedStatistics'],
                              out_reqs)):
                damage_sample = PAL.damage.save_sample()

                if 'Sample' in out_reqs:
                    damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)
                    damage_sample_s.to_csv(
                        "DMG_sample.zip",
                        index_label=damage_sample_s.columns.name,
                        compression=dict(method='zip',
                                         archive_name='DMG_sample.csv'))

                if 'Statistics' in out_reqs:
                    damage_stats = convert_to_SimpleIndex(describe(damage_sample),
                                                          axis=1)
                    damage_stats.to_csv("DMG_stats.csv",
                                        index_label=damage_stats.columns.name)

                if np.any(np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)):
                    grp_damage = damage_sample.groupby(level=[0, 3], axis=1).sum()

                    if 'GroupedSample' in out_reqs:
                        grp_damage_s = convert_to_SimpleIndex(grp_damage, axis=1)
                        grp_damage_s.to_csv("DMG_grp.zip",
                                            index_label=grp_damage_s.columns.name,
                                            compression=dict(
                                                method='zip',
                                                archive_name='DMG_grp.csv'))

                    if 'GroupedStatistics' in out_reqs:
                        grp_stats = convert_to_SimpleIndex(describe(grp_damage),
                                                           axis=1)
                        grp_stats.to_csv("DMG_grp_stats.csv",
                                         index_label=grp_stats.columns.name)

    # Loss Assessment -----------------------------------------------------------

    # if a loss assessment is requested
    if loss_config is not None:

        out_config_loss = out_config.get('Loss', {})

        # if requested, calculate repair consequences
        if loss_config.get('BldgRepair', False):

            bldg_repair_config = loss_config['BldgRepair']

            # load the consequence information
            if bldg_repair_config['ConsequenceDatabase'] != "User Defined":
                consequence_db = (
                        'PelicunDefault/' +
                        default_DBs['repair'][
                            bldg_repair_config['ConsequenceDatabase']])

                conseq_df = PAL.get_default_data(
                    default_DBs['repair'][
                        bldg_repair_config['ConsequenceDatabase']][:-4])

            else:
                consequence_db = bldg_repair_config['ConsequenceDatabasePath']
                conseq_df = load_data(
                    bldg_repair_config['ConsequenceDatabasePath'],
                    orientation=1, reindex=False, convert=[])

            # add the replacement consequence to the data
            adf = pd.DataFrame(
                columns=conseq_df.columns,
                index=pd.MultiIndex.from_tuples(
                    [('replacement', 'Cost'), ('replacement', 'Time')]))

            DL_method = bldg_repair_config['ConsequenceDatabase']
            rc = ('replacement', 'Cost')
            if 'ReplacementCost' in bldg_repair_config.keys():
                rCost_config = bldg_repair_config['ReplacementCost']

                adf.loc[rc, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[rc, ('DV', 'Unit')] = rCost_config["Unit"]

                adf.loc[rc, ('DS1', 'Theta_0')] = rCost_config["Median"]

                if rCost_config.get('Distribution', 'N/A') != 'N/A':
                    adf.loc[rc, ('DS1', 'Family')] = rCost_config[
                        "Distribution"]
                    adf.loc[rc, ('DS1', 'Theta_1')] = rCost_config[
                        "Theta_1"]

            else:
                # add a default replacement cost value as a placeholder
                # the default value depends on the consequence database

                # for FEMA P-58, use 0 USD
                if DL_method == 'FEMA P-58':
                    adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rc, ('DV', 'Unit')] = 'USD_2011'
                    adf.loc[rc, ('DS1', 'Theta_0')] = 0

                # for Hazus EQ, use 1.0 as a loss_ratio
                elif DL_method == 'Hazus Earthquake':
                    adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'

                    # store the replacement cost that corresponds to total loss
                    adf.loc[rc, ('DS1', 'Theta_0')] = 100.0

                # otherwise, use 1 (and expect to have it defined by the user)
                else:
                    adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'
                    adf.loc[rc, ('DS1', 'Theta_0')] = 1

            rt = ('replacement', 'Time')
            if 'ReplacementTime' in bldg_repair_config.keys():
                rTime_config = bldg_repair_config['ReplacementTime']
                rt = ('replacement', 'Time')

                adf.loc[rt, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[rt, ('DV', 'Unit')] = rTime_config["Unit"]

                adf.loc[rt, ('DS1', 'Theta_0')] = rTime_config["Median"]

                if rTime_config.get('Distribution', 'N/A') != 'N/A':
                    adf.loc[rt, ('DS1', 'Family')] = rTime_config[
                        "Distribution"]
                    adf.loc[rt, ('DS1', 'Theta_1')] = rTime_config[
                        "Theta_1"]
            else:
                # add a default replacement time value as a placeholder
                # the default value depends on the consequence database

                # for FEMA P-58, use 0 worker_days
                if DL_method == 'FEMA P-58':
                    adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rt, ('DV', 'Unit')] = 'worker_day'
                    adf.loc[rt, ('DS1', 'Theta_0')] = 0

                # for Hazus EQ, use 1.0 as a loss_ratio
                elif DL_method == 'Hazus Earthquake':
                    adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rt, ('DV', 'Unit')] = 'day'

                    # load the replacement time that corresponds to total loss
                    occ_type = asset_config['OccupancyType']
                    adf.loc[rt, ('DS1', 'Theta_0')] = conseq_df.loc[
                        (f"STR.{occ_type}", 'Time'), ('DS5', 'Theta_0')]

                # otherwise, use 1 (and expect to have it defined by the user)
                else:
                    adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rt, ('DV', 'Unit')] = 'loss_ratio'
                    adf.loc[rt, ('DS1', 'Theta_0')] = 1

            # prepare the loss map
            loss_map = None
            if bldg_repair_config['MapApproach'] == "Automatic":

                # get the damage sample
                dmg_sample = PAL.damage.save_sample()

                # create a mapping for all components that are also in
                # the prescribed consequence database
                dmg_cmps = dmg_sample.columns.unique(level='cmp')
                loss_cmps = conseq_df.index.unique(level=0)

                drivers = []
                loss_models = []

                if DL_method == 'FEMA P-58':

                    # with FEMA P-58 we assume fragility and consequence data
                    # have the same IDs

                    for dmg_cmp in dmg_cmps:

                        if dmg_cmp == 'collapse':
                            continue

                        if dmg_cmp in loss_cmps:
                            drivers.append(f'DMG-{dmg_cmp}')
                            loss_models.append(dmg_cmp)

                elif DL_method == 'Hazus Earthquake':

                    # with Hazus Earthquake we assume that consequence
                    # archetypes are only differentiated by occupancy type
                    occ_type = asset_config['OccupancyType']

                    for dmg_cmp in dmg_cmps:

                        if dmg_cmp == 'collapse':
                            continue

                        cmp_class = dmg_cmp.split('.')[0]
                        loss_cmp = f'{cmp_class}.{occ_type}'

                        if loss_cmp in loss_cmps:
                            drivers.append(f'DMG-{dmg_cmp}')
                            loss_models.append(loss_cmp)

                loss_map = pd.DataFrame(loss_models,
                                        columns=['BldgRepair'],
                                        index=drivers)

            elif bldg_repair_config['MapApproach'] == "User Defined":

                loss_map = pd.read_csv(bldg_repair_config['MapFilePath'],
                                       index_col=0)

            # prepare additional loss map entries, if needed
            if 'DMG-collapse' not in loss_map.index:
                loss_map.loc['DMG-collapse',    'BldgRepair'] = 'replacement'
                loss_map.loc['DMG-irreparable', 'BldgRepair'] = 'replacement'

            PAL.bldg_repair.load_model([conseq_df, adf], loss_map)

            PAL.bldg_repair.calculate()

            agg_repair = PAL.bldg_repair.aggregate_losses()

            if out_config_loss.get('BldgRepair', False):

                out_reqs = [out if val else ""
                            for out, val in out_config_loss['BldgRepair'].items()]

                if np.any(np.isin(['Sample', 'Statistics',
                                   'GroupedSample', 'GroupedStatistics',
                                   'AggregateSample', 'AggregateStatistics'],
                                  out_reqs)):
                    repair_sample = PAL.bldg_repair.save_sample()

                    if 'Sample' in out_reqs:
                        repair_sample_s = convert_to_SimpleIndex(
                            repair_sample, axis=1)
                        repair_sample_s.to_csv(
                            "DV_bldg_repair_sample.zip",
                            index_label=repair_sample_s.columns.name,
                            compression=dict(
                                method='zip',
                                archive_name='DV_bldg_repair_sample.csv'))

                    if 'Statistics' in out_reqs:
                        repair_stats = convert_to_SimpleIndex(
                            describe(repair_sample),
                            axis=1)
                        repair_stats.to_csv("DV_bldg_repair_stats.csv",
                                            index_label=repair_stats.columns.name)

                    if np.any(np.isin(
                            ['GroupedSample', 'GroupedStatistics'], out_reqs)):
                        grp_repair = repair_sample.groupby(
                            level=[0, 1, 2], axis=1).sum()

                        if 'GroupedSample' in out_reqs:
                            grp_repair_s = convert_to_SimpleIndex(grp_repair, axis=1)
                            grp_repair_s.to_csv(
                                "DV_bldg_repair_grp.zip",
                                index_label=grp_repair_s.columns.name,
                                compression=dict(
                                    method='zip',
                                    archive_name='DV_bldg_repair_grp.csv'))

                        if 'GroupedStatistics' in out_reqs:
                            grp_stats = convert_to_SimpleIndex(
                                describe(grp_repair), axis=1)
                            grp_stats.to_csv("DV_bldg_repair_grp_stats.csv",
                                             index_label=grp_stats.columns.name)

                    if np.any(np.isin(['AggregateSample',
                                       'AggregateStatistics'], out_reqs)):

                        if 'AggregateSample' in out_reqs:
                            agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)
                            agg_repair_s.to_csv(
                                "DV_bldg_repair_agg.zip",
                                index_label=agg_repair_s.columns.name,
                                compression=dict(
                                    method='zip',
                                    archive_name='DV_bldg_repair_agg.csv'))

                        if 'AggregateStatistics' in out_reqs:
                            agg_stats = convert_to_SimpleIndex(
                                describe(agg_repair), axis=1)
                            agg_stats.to_csv("DV_bldg_repair_agg_stats.csv",
                                             index_label=agg_stats.columns.name)

    # Result Summary -----------------------------------------------------------

    if 'damage_sample' not in locals():
        damage_sample = PAL.damage.save_sample()

    if 'agg_repair' not in locals():
        agg_repair = PAL.bldg_repair.aggregate_losses()

    damage_sample = damage_sample.groupby(level=[0, 3], axis=1).sum()
    damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)

    if 'collapse-1' in damage_sample_s.columns:
        damage_sample_s['collapse'] = damage_sample_s['collapse-1']
    else:
        damage_sample_s['collapse'] = np.zeros(damage_sample_s.shape[0])

    if 'irreparable-1' in damage_sample_s.columns:
        damage_sample_s['irreparable'] = damage_sample_s['irreparable-1']
    else:
        damage_sample_s['irreparable'] = np.zeros(damage_sample_s.shape[0])

    agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)

    summary = pd.concat([agg_repair_s,
                         damage_sample_s[['collapse', 'irreparable']]],
                        axis=1)

    summary_stats = describe(summary)

    # save summary sample
    summary.to_csv("DL_summary.csv", index_label='#')

    # save summary statistics
    summary_stats.to_csv("DL_summary_stats.csv")

    return 0


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--filenameDL')
    parser.add_argument('-d', '--demandFile', default=None)
    # parser.add_argument('-s', '--sampleSize', default = None)
    # parser.add_argument('--DL_Method', default = None)
    # parser.add_argument('--outputBIM', default='BIM.csv')
    parser.add_argument('--outputEDP', default='EDP.csv')
    parser.add_argument('--outputDM', default='DM.csv')
    parser.add_argument('--outputDV', default='DV.csv')
    parser.add_argument('--dirnameOutput', default = None)
    # parser.add_argument('--event_time', default=None)
    # parser.add_argument('--detailed_results', default = True,
    #    type = str2bool, nargs='?', const=True)
    # parser.add_argument('--coupled_EDP', default = False,
    #    type = str2bool, nargs='?', const=False)
    # parser.add_argument('--log_file', default = True,
    #    type = str2bool, nargs='?', const=True)
    # parser.add_argument('--ground_failure', default = False,
    #    type = str2bool, nargs='?', const=False)
    # parser.add_argument('--auto_script', default=None)
    # parser.add_argument('--resource_dir', default=None)
    args = parser.parse_args(args)

    log_msg('Initializing pelicun calculation...')

    # print(args)
    out = run_pelicun(
        args.filenameDL,
        # args.demandFile,
        # args.sampleSize,
        # args.DL_Method,
        # args.outputBIM, args.outputEDP,
        # args.outputDM, args.outputDV,
        # output_path = args.dirnameOutput,
        # detailed_results = args.detailed_results,
        # coupled_EDP = args.coupled_EDP,
        # log_file = args.log_file,
        # event_time = args.event_time,
        # ground_failure = args.ground_failure,
        # auto_script_path = args.auto_script,
        # resource_dir = args.resource_dir
    )

    if out == -1:
        log_msg("pelicun calculation failed.")
    else:
        log_msg('pelicun calculation completed.')


if __name__ == '__main__':

    main(sys.argv[1:])
