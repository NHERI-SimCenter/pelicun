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

from time import gmtime, strftime

def log_msg(msg):

    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)

    print(formatted_msg)

log_msg('First line of DL_calculation')

import sys, os, json, ntpath, posixpath, argparse
import numpy as np
import pandas as pd
import shutil

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from pelicun.base import set_options, convert_to_MultiIndex, convert_to_SimpleIndex, describe
from pelicun.assessment import Assessment
from pelicun.file_io import save_to_csv

def add_units(raw_demands, config):

    demands = raw_demands.T

    if "Units" not in demands.index:
        demands.insert(0, "Units", np.nan)
    else:
        return raw_demands

    try:
        length_unit = config['GeneralInformation']['units']['length']
    except:
        log_msg("No units assigned to the raw demand input and default unit "
                "definition missing from the input file. Raw demands cannot "
                "be parsed. Terminating analysis. ")

        return None

    if length_unit == 'in':
        length_unit = 'inch'

    demands = convert_to_MultiIndex(demands, axis=0).sort_index(axis=0).T

    demands.drop(demands.columns[demands.columns.get_level_values(1)== ''], axis=1, inplace=True)

    demands[('1','PRD','1','1')] = demands[('1','PRD','1','1')]
    demands[('1','PFA','3','1')] = demands[('1','PFA','3','1')]

    for EDP_type in ['PFA', 'PGA', 'SA']:
        demands.iloc[0, demands.columns.get_level_values(1) == EDP_type] = length_unit+'ps2'

    for EDP_type in ['PFV', 'PWS', 'PGV', 'SV']:
        demands.iloc[0, demands.columns.get_level_values(1) == EDP_type] = length_unit+'ps'

    for EDP_type in ['PFD', 'PIH', 'SD', 'PGD']:
        demands.iloc[0, demands.columns.get_level_values(1) == EDP_type] = length_unit

    for EDP_type in ['PID', 'PRD', 'DWD', 'RDR', 'PMD', 'RID']:
        demands.iloc[0, demands.columns.get_level_values(1) == EDP_type] = 'rad'

    return convert_to_SimpleIndex(demands, axis=1)

def run_pelicun(config_path, demand_path=None, sample_size=None,
    #DL_method, BIM_file, EDP_file, DM_file, DV_file,
    #output_path=None, detailed_results=True, coupled_EDP=False,
    #log_file=True, event_time=None, ground_failure=False,
    #auto_script_path=None, resource_dir=None
    ):

    config_path = os.path.abspath(config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    if demand_path is not None:
        demand_path = os.path.abspath(demand_path) # response.csv

        # check if the raw demands have units identified
        raw_demands = pd.read_csv(demand_path, index_col=0)

        demands = add_units(raw_demands, config)

        if demands is None:
            return -1

        demand_path = demand_path[:-4]+'_ext.csv'

        demands.to_csv(demand_path)

    # Initial setup -----------------------------------------------------------
    general_info = config.get('GeneralInformation', None)
    if general_info is None:
        log_msg("General Information is missing from the input file. "
                "Terminating analysis.")

        return -1

    DL_config = config.get('DamageAndLoss', None)
    if DL_config is None:

        log_msg("Damage and Loss configuration missing from input file. "
                "Terminating analysis.")

        return -1

    PAL = Assessment(DL_config.get("Options", None))

    if general_info.get('NumberOfStories', False):
        PAL.stories = general_info['NumberOfStories']

    if sample_size is None:
        if DL_config.get("SampleSize", False):
            sample_size = DL_config["SampleSize"]

    asset_config = DL_config.get('Asset', None)
    demand_config = DL_config.get('Demand', None)
    damage_config = DL_config.get('Damage', None)
    loss_config = DL_config.get('Loss', None)
    out_config = DL_config.get('Outputs', None)

    if asset_config is None:
        log_msg("Asset configuration missing. Terminating analysis.")
        return -1

    if demand_config is None:
        log_msg("Demand configuration missing. Terminating analysis.")
        return -1

    if out_config is None:
        log_msg("Output configuration missing. Terminating analysis.")
        return -1

    # Asset Definition ------------------------------------------------------------

    # if requested, load a component model and generate a sample
    if asset_config.get('LoadModelFrom', False):

        if sample_size is None:
            log_msg("Sample size not specified. Terminating analysis.")
            return -1

        # load component model
        PAL.asset.load_cmp_model(
            data_source= asset_config['LoadModelFrom'])

        # generate component quantity sample
        PAL.asset.generate_cmp_sample(sample_size)

    # if requested, load the quantity sample from a file
    if asset_config.get('LoadSampleFrom', False):
        PAL.asset.load_cmp_sample(asset_config['LoadSampleFrom'])

    # if requested, save results
    if out_config.get('Asset', False):

        out_reqs = [out if val else "" for out, val in out_config['Asset'].items()]

        if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):
            cmp_sample = PAL.asset.save_cmp_sample()

            if 'Sample' in out_reqs:
                cmp_sample_s = convert_to_SimpleIndex(cmp_sample, axis=1)
                cmp_sample_s.to_csv("CMP_sample.zip",
                                    index_label=cmp_sample_s.columns.name,
                                    compression=dict(method='zip',
                                                     archive_name='CMP_sample.csv'))

            if 'Statistics' in out_reqs:
                cmp_stats = convert_to_SimpleIndex(describe(cmp_sample), axis=1)
                cmp_stats.to_csv("CMP_stats.csv",
                                    index_label=cmp_stats.columns.name)

    # Demand Assessment -----------------------------------------------------------

    # if demand calibration is requested
    if demand_config.get('Calibration', False):

        cal_config = demand_config['Calibration']

        # load demand samples to serve as reference data
        if cal_config.get('LoadSampleFrom', False):
            demand_path = cal_config['LoadSampleFrom']

        if demand_path is None:
            log_msg("Demand sample not specified for demand calibration."
                    "Terminating analysis.")
            return -1

        PAL.demand.load_sample(demand_path)

        # then use it to calibrate the demand model
        PAL.demand.calibrate_model(cal_config['Marginals'])

        # if requested, save the model to files
        if cal_config.get('SaveModelTo', False):
            PAL.demand.save_model(file_prefix=cal_config['SaveModelTo'])

    # if requested, load a pre-calibrated model from files
    if demand_config.get('LoadModelFrom', False):
        PAL.demand.load_model(data_source= demand_config['LoadModelFrom'])

    # if demand samples are provided in a file
    if demand_config.get('LoadSampleFrom', False):
        PAL.demand.load_sample(demand_config['LoadSampleFrom'])

    else:
        # otherwise, generate demand sample
        if sample_size is None:
            log_msg("Sample size not specified. Terminating analysis.")
            return -1

        PAL.demand.generate_sample({"SampleSize": sample_size})

    # if requested, save results
    if out_config.get('Demand', False):

        out_reqs = [out if val else "" for out, val in out_config['Demand'].items()]

        if np.any(np.isin(['Sample', 'Statistics'], out_reqs)):
            demand_sample = PAL.demand.save_sample()

            if 'Sample' in out_reqs:
                demand_sample_s = convert_to_SimpleIndex(demand_sample, axis=1)
                demand_sample_s.to_csv("DEM_sample.zip",
                                       index_label=demand_sample_s.columns.name,
                                       compression=dict(method='zip',
                                                        archive_name='DEM_sample.csv'))

            if 'Statistics' in out_reqs:
                demand_stats = convert_to_SimpleIndex(describe(demand_sample), axis=1)
                demand_stats.to_csv("DEM_stats.csv",
                                    index_label=demand_stats.columns.name)

    # Damage Assessment -----------------------------------------------------------

    # if a damage assessment is requested
    if damage_config is not None:

        if sample_size is None:
            log_msg("Sample size not specified. Terminating analysis.")
            return -1

        # load the fragility information
        PAL.damage.load_damage_model(damage_config['LoadModelFrom'])

        # calculate damages
        # load the damage process if needed
        if damage_config.get('DamageProcess', False):
            with open(damage_config['DamageProcess'], 'r') as f:
                dmg_process = json.load(f)
        else:
            dmg_process = None

        PAL.damage.calculate(sample_size, dmg_process=dmg_process)

        # if damage samples are provided in a file
        if damage_config.get('LoadSampleFrom', False):
            PAL.damage.load_sample(damage_config['LoadSampleFrom'])

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
                    grp_damage = damage_sample.groupby(level=[0,3], axis=1).sum()

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

        loss_map_path = loss_config['LoadMappingFrom']

        # if requested, calculate repair consequences
        if loss_config.get('CalculateBldgRepair', False):

            bldg_repair_config = loss_config['CalculateBldgRepair']

            PAL.bldg_repair.load_model(bldg_repair_config['LoadModelFrom'],
                                       loss_map_path)

            PAL.bldg_repair.calculate(sample_size)

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
                        repair_sample_s = convert_to_SimpleIndex(repair_sample, axis=1)
                        repair_sample_s.to_csv(
                            "DV_bldg_repair_sample.zip",
                            index_label=repair_sample_s.columns.name,
                            compression=dict(method='zip',
                                             archive_name='DV_bldg_repair_sample.csv'))

                    if 'Statistics' in out_reqs:
                        repair_stats = convert_to_SimpleIndex(describe(repair_sample),
                                                              axis=1)
                        repair_stats.to_csv("DV_bldg_repair_stats.csv",
                                            index_label=repair_stats.columns.name)

                    if np.any(np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)):
                        grp_repair = repair_sample.groupby(level=[0,1,2], axis=1).sum()

                        if 'GroupedSample' in out_reqs:
                            grp_repair_s = convert_to_SimpleIndex(grp_repair, axis=1)
                            grp_repair_s.to_csv("DV_bldg_repair_grp.zip",
                                              index_label=grp_repair_s.columns.name,
                                              compression=dict(
                                                method='zip',
                                                archive_name='DV_bldg_repair_grp.csv'))

                        if 'GroupedStatistics' in out_reqs:
                            grp_stats = convert_to_SimpleIndex(describe(grp_repair), axis=1)
                            grp_stats.to_csv("DV_bldg_repair_grp_stats.csv",
                                             index_label=grp_stats.columns.name)

                    if np.any(np.isin(['AggregateSample', 'AggregateStatistics'], out_reqs)):

                        if 'AggregateSample' in out_reqs:
                            agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)
                            agg_repair_s.to_csv("DV_bldg_repair_agg.zip",
                                              index_label=agg_repair_s.columns.name,
                                              compression=dict(
                                                method='zip',
                                                archive_name='DV_bldg_repair_agg.csv'))

                        if 'AggregateStatistics' in out_reqs:
                            agg_stats = convert_to_SimpleIndex(describe(agg_repair), axis=1)
                            agg_stats.to_csv("DV_bldg_repair_agg_stats.csv",
                                             index_label=agg_stats.columns.name)

    return 0

def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configFile')
    parser.add_argument('-d', '--demandFile', default = None)
    parser.add_argument('-s', '--sampleSize', default = None)
    #parser.add_argument('--DL_Method', default = None)
    #parser.add_argument('--outputBIM', default='BIM.csv')
    #parser.add_argument('--outputEDP', default='EDP.csv')
    #parser.add_argument('--outputDM', default = 'DM.csv')
    #parser.add_argument('--outputDV', default = 'DV.csv')
    #parser.add_argument('--dirnameOutput', default = None)
    #parser.add_argument('--event_time', default=None)
    #parser.add_argument('--detailed_results', default = True,
    #    type = str2bool, nargs='?', const=True)
    #parser.add_argument('--coupled_EDP', default = False,
    #    type = str2bool, nargs='?', const=False)
    #parser.add_argument('--log_file', default = True,
    #    type = str2bool, nargs='?', const=True)
    #parser.add_argument('--ground_failure', default = False,
    #    type = str2bool, nargs='?', const=False)
    #parser.add_argument('--auto_script', default=None)
    #parser.add_argument('--resource_dir', default=None)
    args = parser.parse_args(args)

    log_msg('Initializing pelicun calculation...')

    #print(args)
    out = run_pelicun(
        args.configFile, args.demandFile,
        args.sampleSize,
        #args.DL_Method,
        #args.outputBIM, args.outputEDP,
        #args.outputDM, args.outputDV,
        #output_path = args.dirnameOutput,
        #detailed_results = args.detailed_results,
        #coupled_EDP = args.coupled_EDP,
        #log_file = args.log_file,
        #event_time = args.event_time,
        #ground_failure = args.ground_failure,
        #auto_script_path = args.auto_script,
        #resource_dir = args.resource_dir
    )

    if out == -1:
        log_msg("pelicun calculation failed.")
    else:
        log_msg('pelicun calculation completed.')

if __name__ == '__main__':

    main(sys.argv[1:])
