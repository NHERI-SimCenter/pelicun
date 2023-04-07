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

from time import gmtime
from time import strftime
import sys, os, json
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pelicun.auto import auto_populate
from pelicun.base import str2bool
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

# suppress FutureWarnings by default - credit: ioannis_vm
if not sys.warnoptions:
    warnings.filterwarnings(
        category=FutureWarning, action='ignore')

def log_msg(msg):

    formatted_msg = f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} {msg}'

    print(formatted_msg)


log_msg('First line of DL_calculation')

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

idx = pd.IndexSlice

damage_processes = {
    'FEMA P-58': {
        "1_excessive.coll.DEM": {
            "DS1": "collapse_DS1"
        },
        "2_collapse": {
            "DS1": "ALL_NA"
        },
        "3_excessiveRID": {
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
        "3_excessive.coll.DEM": {
            "DS1": "collapse_DS1"
        },
        "4_collapse": {
            "DS1": "ALL_NA"
        },
        "5_excessiveRID": {
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
# list of output files help perform safe initialization of output dir
output_files = [
    "DEM_sample.zip",
    "DEM_stats.csv",
    "CMP_sample.zip",
    "CMP_stats.csv",
    "DMG_sample.zip",
    "DMG_stats.csv",
    "DMG_grp.zip",
    "DMG_grp_stats.csv",
    "DV_bldg_repair_sample.zip",
    "DV_bldg_repair_stats.csv",
    "DV_bldg_repair_grp.zip",
    "DV_bldg_repair_grp_stats.csv",
    "DV_bldg_repair_agg.zip",
    "DV_bldg_repair_agg_stats.csv",
    "DL_summary.csv",
    "DL_summary_stats.csv",
]


def convert_df_to_dict(df, axis=1):

    out_dict = {}
    
    if axis == 1:
        df_in = df
    elif axis == 0:
        df_in = df.T
    else:
        pass
        # TODO: raise error
        
    MI = df_in.columns
    
    for label in MI.unique(level=0):
        
        out_dict.update({label: np.nan})
        
        sub_df = df_in[label]
        
        skip_sub = True
        
        if MI.nlevels>1: 
            skip_sub = False
            
            if isinstance(sub_df, pd.Series):
                skip_sub = True
            elif (len(sub_df.columns) == 1) and (sub_df.columns[0] == ""):
                skip_sub = True
            
            if skip_sub == False:
                out_dict[label] = convert_df_to_dict(sub_df)
                
        if skip_sub == True:
            
            if np.all(sub_df.index.astype(str).str.isnumeric()) == True:
                out_dict[label] = df_in[label].tolist()
            else:
                out_dict[label] = {key:sub_df.loc[key] for key in sub_df.index}
                
    return out_dict


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

    # drift ratio
    rot_EDPs = ['PID', 'PRD', 'DWD', 'RDR', 'PMD', 'RID']
    EDP_mask = np.isin(demand_cols, rot_EDPs)

    if np.any(EDP_mask):
        demands.iloc[0, EDP_mask] = 'unitless'

    # convert back to simple header and return the DF
    return convert_to_SimpleIndex(demands, axis=1)


def regional_output_demand():
    pass

def run_pelicun(config_path, demand_file, output_path, coupled_EDP, 
    realizations, auto_script_path, detailed_results, regional, output_format, **kwargs):
    """
    Use settings in the config JSON to prepare and run a Pelicun calculation.

    Parameters
    ----------
    config_path: string
        Path pointing to the location of the JSON configuration file.
    demand_file: string
        Path pointing to the location of a CSV file with the demand data.
    output_path: string, optional
        Path pointing to the location where results shall be saved.
    coupled_EDP: bool, optional
        If True, EDPs are not resampled and processed in order.
    realizations: int, optional
        Number of realizations to generate.
    auto_script_path: string, optional
        Path pointing to the location of a Python script with an auto_populate
        method that automatically creates the performance model using data
        provided in the AIM JSON file.
    detailed_results: bool, optional
        If False, only the main statistics are saved.

    """

    # Initial setup -----------------------------------------------------------

    # get the absolute path to the config file
    config_path = Path(config_path).resolve()

    # If the output path was not specified, results are saved in the directory 
    # of the input file.
    if output_path is None:
        output_path = config_path.parents[0]
    else:
        output_path = Path(output_path)

    # Initialize the array that we'll use to collect the output file names
    output_files = []

    # Initialize the output folder - i.e., remove existing output files from 
    # there
    files = os.listdir(output_path)
    for filename in files:
        if filename in output_files:
            try:
                os.remove(output_path/filename)
            except:
                # TODO: show some kind of a warning here
                pass

    # open the config file and parse it
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    DL_config = config.get('DL', None)
    if (DL_config == None) or (DL_config == {}):

        log_msg("Damage and Loss configuration missing from config file. ")

        if auto_script_path != None:
            log_msg("Trying to auto-populate")

            config_ap, CMP = auto_populate(config, auto_script_path)

            # add the demand information
            config_ap['DL']['Demands'].update({
                'DemandFilePath': f'{demand_file}',
                'SampleSize': f'{realizations}'
            })

            if coupled_EDP==True:
                config_ap['DL']['Demands'].update({
                    "CoupledDemands": True
                })

            else:
                config_ap['DL']['Demands'].update({
                    "Calibration": {
                        "ALL": {
                            "DistributionFamily": "lognormal"
                        }
                    }
                })

            # save the component data        
            CMP.to_csv(output_path/'CMP_QNT.csv')

            # update the config file with the location
            config_ap['DL']['Asset'].update({
                "ComponentAssignmentFile": str(output_path/'CMP_QNT.csv')
            })

            # if detailed results are not requested, add a lean output config
            if detailed_results == False:
                config_ap['DL'].update({
                    'Outputs': {  
                        'Demand': {},
                        'Asset': {},
                        'Damage': {},
                        'Loss': {
                            'BldgRepair': {}                   
                        }
                    }
                })

            # save the extended config to a file
            config_ap_path = Path(config_path.stem + '_ap.json').resolve()

            with open(config_ap_path, 'w') as f:
                json.dump(config_ap, f, indent=2)

            DL_config = config_ap.get('DL', None)

        else:
            log_msg("Terminating analysis.")

            return -1

    GI_config = config.get('GeneralInformation', None)

    asset_config = DL_config.get('Asset', None)
    demand_config = DL_config.get('Demands', None)
    damage_config = DL_config.get('Damage', None)
    loss_config = DL_config.get('Losses', None)
    out_config = DL_config.get('Outputs', None)
    
    # provide all outputs if the files are not specified
    if out_config == None:
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

    # provide outputs in CSV by default
    if ('Format' in out_config.keys()) == False:
        out_config.update({
            'Format': {
                'CSV': True,
                'JSON': False
            }
        })

    # override file format specification if the output_format is provided
    if output_format != None:
        out_config.update({
            'Format': {
                'CSV': 'csv' in output_format,
                'JSON': 'json' in output_format,
            }
        })

    if asset_config is None:
        log_msg("Asset configuration missing. Terminating analysis.")
        return -1

    if demand_config is None:
        log_msg("Demand configuration missing. Terminating analysis.")
        return -1

    # get the length unit from the config file
    try:
        length_unit = GI_config['units']['length']
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

    # load the available demand sample
    PAL.demand.load_sample(demands)

    # get the calibration information
    if demand_config.get('Calibration', False):

        # then use it to calibrate the demand model
        PAL.demand.calibrate_model(demand_config['Calibration'])

    else:
        # if no calibration is requested, 
        # set all demands to use empirical distribution
        PAL.demand.calibrate_model({
            "ALL": {"DistributionFamily": "empirical"}
            })

    # and generate a new demand sample
    sample_size = int(demand_config['SampleSize'])

    PAL.demand.generate_sample({
        "SampleSize": sample_size,
        'PreserveRawOrder': demand_config.get('CoupledDemands', False)
        })

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
            RID_units = pd.Series(['unitless', ]*RID.shape[1], index=RID.columns,
                                  name='Units')
        RID_sample = pd.concat([RID, RID_units.to_frame().T])

        demand_sample = pd.concat([demand_sample, RID_sample], axis=1)

    # add a constant one demand
    demand_sample[('ONE', '0', '1')] = np.ones(demand_sample.shape[0])
    demand_sample.loc['Units', ('ONE', '0', '1')] = 'unitless'

    PAL.demand.load_sample(convert_to_SimpleIndex(demand_sample, axis=1))

    # save results
    if out_config.get('Demand', None) != None:

        out_reqs = [out if val else "" for out, val in out_config['Demand'].items()]

        if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):
            demand_sample = PAL.demand.save_sample()

            if 'Sample' in out_reqs:
                demand_sample_s = convert_to_SimpleIndex(demand_sample, axis=1)
                demand_sample_s.to_csv(output_path/"DEM_sample.zip",
                                       index_label=demand_sample_s.columns.name,
                                       compression=dict(
                                           method='zip',
                                           archive_name='DEM_sample.csv'))   
                output_files.append('DEM_sample.zip')             

            if 'Statistics' in out_reqs:
                demand_stats = convert_to_SimpleIndex(
                    describe(demand_sample), axis=1)
                demand_stats.to_csv(output_path/"DEM_stats.csv",
                                    index_label=demand_stats.columns.name)
                output_files.append('DEM_stats.csv')

        if regional == True:
            
            demand_sample = PAL.demand.save_sample()

            mean = demand_sample.mean()
            median = demand_sample.median()
            std = demand_sample.std()
            beta = np.log(demand_sample).std()            

            res = pd.concat([mean,std,median,beta], 
                keys=['mean','std','median','beta']).to_frame().T

            res = res.reorder_levels([1,2,3,0], axis=1)

            res.sort_index(axis=1, inplace=True)

            res.dropna(axis=1, how='all', inplace=True)

            res.columns.rename(['type', 'loc', 'dir', 'stat'], inplace=True)

            res.to_csv(output_path/"EDP.csv", index_label=res.columns.name)
            output_files.append('EDP.csv')


    # Asset Definition ------------------------------------------------------------

    # set the number of stories
    if asset_config.get('NumberOfStories', False):
        PAL.stories = int(asset_config['NumberOfStories'])

    # load a component model and generate a sample
    if asset_config.get('ComponentAssignmentFile', False):

        cmp_marginals = pd.read_csv(asset_config['ComponentAssignmentFile'],
                                    index_col=0, encoding_errors='replace')

        DEM_types = demand_sample.columns.unique(level=0)

        # add component(s) to support collapse calculation
        if 'CollapseFragility' in damage_config.keys():
            coll_DEM = damage_config['CollapseFragility']["DemandType"]
            if coll_DEM.startswith('SA'):
                # we have a global demand and evaluate collapse directly
                pass

            else:
                # we need story-specific collapse assessment

                if coll_DEM in DEM_types:

                    # excessive coll_DEM is added on every floor to detect large RIDs
                    cmp_marginals.loc['excessive.coll.DEM', 'Units'] = 'ea'

                    locs = demand_sample[coll_DEM].columns.unique(level=0)
                    cmp_marginals.loc['excessive.coll.DEM', 'Location'] = ','.join(locs)

                    dirs = demand_sample[coll_DEM].columns.unique(level=1)
                    cmp_marginals.loc['excessive.coll.DEM', 'Direction'] = ','.join(dirs)

                    cmp_marginals.loc['excessive.coll.DEM', 'Theta_0'] = 1.0                

                else:

                    log_msg(f'WARNING: No {coll_DEM} among available demands. Collapse '
                             'cannot be evaluated.')

        # always add a component to support basic collapse calculation
        cmp_marginals.loc['collapse', 'Units'] = 'ea'
        cmp_marginals.loc['collapse', 'Location'] = 0
        cmp_marginals.loc['collapse', 'Direction'] = 1
        cmp_marginals.loc['collapse', 'Theta_0'] = 1.0

        # add components to support irreparable damage calculation
        if 'IrreparableDamage' in damage_config.keys():
            
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
    if out_config.get('Asset', None) != None:

        out_reqs = [out if val else "" for out, val in out_config['Asset'].items()]

        if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):

            if 'Sample' in out_reqs:
                cmp_sample_s = convert_to_SimpleIndex(cmp_sample, axis=1)
                cmp_sample_s.to_csv(output_path/"CMP_sample.zip",
                                    index_label=cmp_sample_s.columns.name,
                                    compression=dict(method='zip',
                                                     archive_name='CMP_sample.csv'))
                output_files.append('CMP_sample.zip')

            if 'Statistics' in out_reqs:
                cmp_stats = convert_to_SimpleIndex(describe(cmp_sample), axis=1)
                cmp_stats.to_csv(
                    output_path/"CMP_stats.csv",
                    index_label=cmp_stats.columns.name)
                output_files.append('CMP_stats.csv')

        if regional == True:

            #flatten the dictionary
            AIM_flat_dict = {}
            for key, item in GI_config.items():
                if isinstance(item, dict):
                    if key not in ['units', 'location']:
                        for sub_key, sub_item in item.items():
                            AIM_flat_dict.update({f'{key}_{sub_key}': sub_item})
                else:
                    AIM_flat_dict.update({key: [item,]})


            # do not save polygons
            for header_to_remove in ['geometry', 'Footprint']:
                try:
                    AIM_flat_dict.pop(header_to_remove)
                except:
                    pass

            # create the output DF
            df_res = pd.DataFrame.from_dict(AIM_flat_dict)

            df_res.dropna(axis=1, how='all', inplace=True)

            df_res.to_csv(output_path/'AIM.csv')
            output_files.append('AIM.csv')

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

            if 'excessive.coll.DEM' in cmp_marginals.index:
                # if there is story-specific evaluation
                coll_CMP_name = 'excessive.coll.DEM'
            else:
                # otherwise, for global collapse evaluation
                coll_CMP_name = 'collapse'

            adf.loc[coll_CMP_name, ('Demand', 'Directional')] = 1
            adf.loc[coll_CMP_name, ('Demand', 'Offset')] = 0

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
                adf.loc[coll_CMP_name, ('Demand', 'Type')] = coll_DEM_name

            else:
                adf.loc[coll_CMP_name, ('Demand', 'Type')] = \
                    f'{coll_DEM_name}|{coll_DEM_spec}'

            coll_DEM_unit = add_units(
                pd.DataFrame(columns=[f'{coll_DEM}-1-1', ]),
                length_unit).iloc[0, 0]

            adf.loc[coll_CMP_name, ('Demand', 'Unit')] = coll_DEM_unit

            adf.loc[coll_CMP_name, ('LS1', 'Family')] = (
                coll_config.get('CapacityDistribution', np.nan))

            adf.loc[coll_CMP_name, ('LS1', 'Theta_0')] = (
                coll_config.get('CapacityMedian', np.nan))

            adf.loc[coll_CMP_name, ('LS1', 'Theta_1')] = (
                coll_config.get('Theta_1', np.nan))

            adf.loc[coll_CMP_name, 'Incomplete'] = 0

            if coll_CMP_name != 'collapse':
                # for story-specific evaluation, we need to add a placeholder 
                # fragility that will never trigger, but helps us aggregate 
                # results in the end
                adf.loc['collapse', ('Demand', 'Directional')] = 1
                adf.loc['collapse', ('Demand', 'Offset')] = 0
                adf.loc['collapse', ('Demand', 'Type')] = 'One'
                adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
                adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
                adf.loc['collapse', 'Incomplete'] = 0                

        else:

            # add a placeholder collapse fragility that will never trigger
            # collapse, but allow damage processes to work with collapse

            adf.loc['collapse', ('Demand', 'Directional')] = 1
            adf.loc['collapse', ('Demand', 'Offset')] = 0
            adf.loc['collapse', ('Demand', 'Type')] = 'One'
            adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
            adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
            adf.loc['collapse', 'Incomplete'] = 0

        if 'IrreparableDamage' in damage_config.keys():

            irrep_config = damage_config['IrreparableDamage']

            # add excessive RID fragility according to settings provided in the
            # input file
            adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
            adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
            adf.loc['excessiveRID',
                    ('Demand', 'Type')] = 'Residual Interstory Drift Ratio'

            adf.loc['excessiveRID', ('Demand', 'Unit')] = 'unitless'
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
            adf.loc['irreparable', ('Demand', 'Unit')] = 'unitless'
            adf.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
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
        if out_config.get('Damage', None) != None:

            out_reqs = [out if val else ""
                        for out, val in out_config['Damage'].items()]

            if np.any(np.isin(['Sample', 'Statistics',
                               'GroupedSample', 'GroupedStatistics'],
                              out_reqs)):
                damage_sample = PAL.damage.save_sample()

                if 'Sample' in out_reqs:
                    damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)
                    damage_sample_s.to_csv(
                        output_path/"DMG_sample.zip",
                        index_label=damage_sample_s.columns.name,
                        compression=dict(method='zip',
                                         archive_name='DMG_sample.csv'))
                    output_files.append('DMG_sample.zip')

                if 'Statistics' in out_reqs:
                    damage_stats = convert_to_SimpleIndex(describe(damage_sample),
                                                          axis=1)
                    damage_stats.to_csv(output_path/"DMG_stats.csv",
                                        index_label=damage_stats.columns.name)
                    output_files.append('DMG_stats.csv')

                if np.any(np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)):
                    grp_damage = damage_sample.groupby(level=[0, 3], axis=1).sum()

                    if 'GroupedSample' in out_reqs:
                        grp_damage_s = convert_to_SimpleIndex(grp_damage, axis=1)
                        grp_damage_s.to_csv(output_path/"DMG_grp.zip",
                                            index_label=grp_damage_s.columns.name,
                                            compression=dict(
                                                method='zip',
                                                archive_name='DMG_grp.csv'))
                        output_files.append('DMG_grp.zip')

                    if 'GroupedStatistics' in out_reqs:
                        grp_stats = convert_to_SimpleIndex(describe(grp_damage),
                                                           axis=1)
                        grp_stats.to_csv(output_path/"DMG_grp_stats.csv",
                                         index_label=grp_stats.columns.name)
                        output_files.append('DMG_grp_stats.csv')

            if regional == True:

                damage_sample = PAL.damage.save_sample()

                # first, get the collapse probability
                df_res_c = pd.DataFrame([0,],
                    columns=pd.MultiIndex.from_tuples([('probability',' '),]),
                    index=[0, ])

                if 'collapse-0-1-1' in damage_sample.columns:
                    df_res_c['probability'] = (
                        damage_sample['collapse-0-1-1'].mean())

                else:
                    df_res_c['probability'] = 0.0

                df_res = pd.concat([df_res_c,], axis=1, keys=['collapse',])

                df_res.to_csv(output_path/'DM.csv')
                output_files.append('DM.csv')

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
                    unit_conversion_factors={},
                    orientation=1, reindex=False, convert=[])

            # add the replacement consequence to the data
            adf = pd.DataFrame(
                columns=conseq_df.columns,
                index=pd.MultiIndex.from_tuples(
                    [('replacement', 'Cost'), ('replacement', 'Time')]))

            #DL_method = bldg_repair_config['ConsequenceDatabase']
            DL_method = damage_config.get('DamageProcess', 'User Defined')

            rc = ('replacement', 'Cost')
            if 'ReplacementCost' in bldg_repair_config.keys():
                rCost_config = bldg_repair_config['ReplacementCost']

                adf.loc[rc, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[rc, ('DV', 'Unit')] = rCost_config["Unit"]

                adf.loc[rc, ('DS1', 'Theta_0')] = rCost_config["Median"]

                if pd.isna(rCost_config.get('Distribution', np.nan)) == False:
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

                if pd.isna(rTime_config.get('Distribution', np.nan))==False:
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
                            output_path/"DV_bldg_repair_sample.zip",
                            index_label=repair_sample_s.columns.name,
                            compression=dict(
                                method='zip',
                                archive_name='DV_bldg_repair_sample.csv'))
                        output_files.append('DV_bldg_repair_sample.zip')

                    if 'Statistics' in out_reqs:
                        repair_stats = convert_to_SimpleIndex(
                            describe(repair_sample),
                            axis=1)
                        repair_stats.to_csv(output_path/"DV_bldg_repair_stats.csv",
                                            index_label=repair_stats.columns.name)
                        output_files.append('DV_bldg_repair_stats.csv')

                    if np.any(np.isin(
                            ['GroupedSample', 'GroupedStatistics'], out_reqs)):
                        grp_repair = repair_sample.groupby(
                            level=[0, 1, 2], axis=1).sum()

                        if 'GroupedSample' in out_reqs:
                            grp_repair_s = convert_to_SimpleIndex(grp_repair, axis=1)
                            grp_repair_s.to_csv(
                                output_path/"DV_bldg_repair_grp.zip",
                                index_label=grp_repair_s.columns.name,
                                compression=dict(
                                    method='zip',
                                    archive_name='DV_bldg_repair_grp.csv'))
                            output_files.append('DV_bldg_repair_grp.zip')

                        if 'GroupedStatistics' in out_reqs:
                            grp_stats = convert_to_SimpleIndex(
                                describe(grp_repair), axis=1)
                            grp_stats.to_csv(output_path/"DV_bldg_repair_grp_stats.csv",
                                             index_label=grp_stats.columns.name)
                            output_files.append('DV_bldg_repair_grp_stats.csv')

                    if np.any(np.isin(['AggregateSample',
                                       'AggregateStatistics'], out_reqs)):

                        if 'AggregateSample' in out_reqs:
                            agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)
                            agg_repair_s.to_csv(
                                output_path/"DV_bldg_repair_agg.zip",
                                index_label=agg_repair_s.columns.name,
                                compression=dict(
                                    method='zip',
                                    archive_name='DV_bldg_repair_agg.csv'))
                            output_files.append('DV_bldg_repair_agg.zip')

                        if 'AggregateStatistics' in out_reqs:
                            agg_stats = convert_to_SimpleIndex(
                                describe(agg_repair), axis=1)
                            agg_stats.to_csv(output_path/"DV_bldg_repair_agg_stats.csv",
                                             index_label=agg_stats.columns.name)
                            output_files.append('DV_bldg_repair_agg_stats.csv')

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
    summary.to_csv(output_path/"DL_summary.csv", index_label='#')
    output_files.append('DL_summary.csv')

    # save summary statistics
    summary_stats.to_csv(output_path/"DL_summary_stats.csv")
    output_files.append('DL_summary_stats.csv')

    # create json outputs if needed
    if out_config['Format']['JSON'] == True:

        for filename in output_files:
            
            filename_json = filename[:-3]+'json'
            
            df = convert_to_MultiIndex(pd.read_csv(output_path/filename, index_col=0),axis=1)
            
            out_dict = convert_df_to_dict(df)
            
            with open(output_path/filename_json, 'w') as f:
                json.dump(out_dict, f, indent=2)

    # remove csv outputs if they were not requested
    if out_config['Format']['CSV'] == False:

        for filename in output_files:

            # keep the DL_summary and DL_summary_stats files
            if 'DL_summary' in filename:
                continue
            
            os.remove(output_path/filename)

    return 0


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--filenameDL')
    parser.add_argument('-d', '--demandFile', default=None)
    parser.add_argument('-s', '--Realizations', default = None)
    parser.add_argument('--dirnameOutput', default = None)
    parser.add_argument('--event_time', default=None)
    parser.add_argument('--detailed_results', default = True,
       type = str2bool, nargs='?', const=True)
    parser.add_argument('--coupled_EDP', default = False,
       type = str2bool, nargs='?', const=False)
    parser.add_argument('--log_file', default = True,
       type = str2bool, nargs='?', const=True)
    parser.add_argument('--ground_failure', default = False,
       type = str2bool, nargs='?', const=False)
    parser.add_argument('--auto_script', default=None)
    parser.add_argument('--resource_dir', default=None)
    parser.add_argument('--custom_fragility_dir', default=None)
    parser.add_argument('--regional', default = False,
       type = str2bool, nargs='?', const=False)
    parser.add_argument('--output_format', default = None)
    # parser.add_argument('-d', '--demandFile', default=None)
    # parser.add_argument('--DL_Method', default = None)
    # parser.add_argument('--outputBIM', default='BIM.csv')
    # parser.add_argument('--outputEDP', default='EDP.csv')
    # parser.add_argument('--outputDM', default='DM.csv')
    # parser.add_argument('--outputDV', default='DV.csv')
    args = parser.parse_args(args)

    log_msg('Initializing pelicun calculation...')

    # print(args)
    out = run_pelicun(
        args.filenameDL,
        demand_file = args.demandFile,
        output_path = args.dirnameOutput,
        realizations = args.Realizations,
        detailed_results = args.detailed_results,
        coupled_EDP = args.coupled_EDP,
        log_file = args.log_file,
        event_time = args.event_time,
        ground_failure = args.ground_failure,
        auto_script_path = args.auto_script,
        resource_dir = args.resource_dir,
        custom_fragility_dir = args.custom_fragility_dir,
        regional = args.regional,
        output_format = args.output_format
    )

    if out == -1:
        log_msg("pelicun calculation failed.")
    else:
        log_msg('pelicun calculation completed.')


if __name__ == '__main__':

    main(sys.argv[1:])
