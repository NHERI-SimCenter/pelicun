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

idx = pd.IndexSlice

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from pelicun.base import set_options
from pelicun.assessment import *
from pelicun.file_io import *

def run_pelicun(DL_input_path, EDP_input_path,
    DL_method, realization_count, BIM_file, EDP_file, DM_file, DV_file,
    output_path=None, detailed_results=True, coupled_EDP=False,
    log_file=True, event_time=None, ground_failure=False,
    auto_script_path=None, resource_dir=None):

    print(DL_input_path)

    DL_input_path = os.path.abspath(DL_input_path) # BIM file
    if EDP_input_path is not None:
        EDP_input_path = os.path.abspath(EDP_input_path) # dakotaTab

    config_path = DL_input_path

    # Initial setup -----------------------------------------------------------

    with open(config_path, 'r') as f:
        config = json.load(f)

    config = merge_default_config(config)

    config_opt = config.get("Options", None)
    set_options(config_opt)

    A = Assessment()

    general_info = config['GeneralInformation']

    if general_info.get('NumberOfStories', False):
        A.stories = general_info['NumberOfStories']

    demand_config = config.get('DemandAssessment', None)
    damage_config = config.get('DamageAssessment', None)
    loss_config = config.get('LossAssessment', None)

    # Demand Assessment -----------------------------------------------------------

    # if a demand assessment is requested
    if demand_config is not None:

        # if demand calibration is requested
        if demand_config.get('Calibration', False):

            cal_config = demand_config['Calibration']

            # load demand samples to serve as reference data
            A.demand.load_sample(cal_config['LoadSampleFrom'])

            # then use it to calibrate the demand model
            A.demand.calibrate_model(cal_config['Marginals'])

            # if requested, save the model to files
            if cal_config.get('SaveModelTo', False):
                A.demand.save_model(file_prefix=cal_config['SaveModelTo'])

        # if demand resampling is requested
        if demand_config.get('Sampling', False):

            sample_config = demand_config['Sampling']

            # if requested, load the calibrated model from files
            if sample_config.get('LoadModelFrom', False):
                A.demand.load_model(file_prefix = sample_config['LoadModelFrom'])

            # generate demand sample
            A.demand.generate_sample(sample_config)

            # if requested, save the sample to a file
            if sample_config.get('SaveSampleTo', False):
                A.demand.save_sample(sample_config['SaveSampleTo'])

    # Damage Assessment -----------------------------------------------------------

    # if a damage assessment is requested
    if damage_config is not None:

        # if component quantity sampling is requested
        if damage_config.get('Components', False):

            cmp_config = damage_config['Components']

            # if requested, load a component model and generate a sample
            if cmp_config.get('LoadModelFrom', False):
                # load component model
                A.asset.load_cmp_model(file_prefix = cmp_config['LoadModelFrom'])

                # generate component quantity sample
                A.asset.generate_cmp_sample(damage_config['SampleSize'])

            # if requested, save the quantity sample to a file
            if cmp_config.get('SaveSampleTo', False):
                A.asset.save_cmp_sample(cmp_config['SaveSampleTo'])

            # if requested, load the quantity sample from a file
            if cmp_config.get('LoadSampleFrom', False):
                A.asset.load_cmp_sample(cmp_config['LoadSampleFrom'])

        # if requested, load the demands from file
        # (if not, we assume there is a preceding demand assessment with sampling)
        if damage_config.get('Demands', False):

            if damage_config['Demands'].get('LoadSampleFrom', False):

                A.demand.load_sample(damage_config['Demands']['LoadSampleFrom'])

        # load the fragility information
        A.damage.load_fragility_model(damage_config['Fragilities']['LoadModelFrom'])

        # calculate damages
        # load the damage process if needed
        if damage_config['Calculation'].get('DamageProcessFrom', False):
            with open(damage_config['Calculation']['DamageProcessFrom'], 'r') as f:
                dmg_process = json.load(f)
        else:
            dmg_process = None
        A.damage.calculate(damage_config['SampleSize'], dmg_process=dmg_process)

        # if requested, save the damage sample to a file
        if damage_config['Calculation'].get('SaveDamageSampleTo', False):
            A.damage.save_sample(damage_config['Calculation']['SaveDamageSampleTo'])

    # Loss Assessment -----------------------------------------------------------

    # if a loss assessment is requested
    if loss_config is not None:

        # if requested, load the demands from file
        # (if not, we assume there is a preceding demand assessment with sampling)
        if loss_config.get('Demands', False):

            if loss_config['Demands'].get('LoadSampleFrom', False):

                A.demand.load_sample(loss_config['Demands']['LoadSampleFrom'])

        # if requested, load the component data from file
        # (if not, we assume there is a preceding assessment with component sampling)
        if loss_config.get('Components', False):

            if loss_config['Components'].get('LoadSampleFrom', False):

                A.asset.load_cmp_sample(loss_config['Components']['LoadSampleFrom'])

        # if requested, load the damage from file
        # (if not, we assume there is a preceding damage assessment with sampling)
        if loss_config.get('Damage', False):

            if loss_config['Damage'].get('LoadSampleFrom', False):

                A.damage.load_sample(loss_config['Damage']['LoadSampleFrom'])

        # if requested, calculate repair consequences
        if loss_config.get('CalculateBldgRepair', False):

            A.bldg_repair.load_model(loss_config['CalculateBldgRepair']['LoadModelFrom'],
                                     loss_config['CalculateBldgRepair']['LoadMappingFrom'])

            A.bldg_repair.calculate(loss_config['SampleSize'])

            # if requested, save the loss sample to a file
            if loss_config['CalculateBldgRepair'].get('SaveLossSampleTo', False):

                A.bldg_repair.save_sample(loss_config['CalculateBldgRepair']['SaveLossSampleTo'])

        # if requested, aggregate results
        if loss_config.get('SaveAggregateResultsTo', False):

            agg_DF = A.bldg_repair.aggregate_losses()

            save_to_csv(agg_DF, loss_config['SaveAggregateResultsTo'])

    return 0

def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameDL')
    parser.add_argument('--filenameEDP', default = None)
    parser.add_argument('--DL_Method', default = None)
    parser.add_argument('--Realizations', default = None)
    parser.add_argument('--outputBIM', default='BIM.csv')
    parser.add_argument('--outputEDP', default='EDP.csv')
    parser.add_argument('--outputDM', default = 'DM.csv')
    parser.add_argument('--outputDV', default = 'DV.csv')
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
    args = parser.parse_args(args)

    log_msg('Initializing pelicun calculation...')

    #print(args)
    run_pelicun(
        args.filenameDL, args.filenameEDP,
        args.DL_Method, args.Realizations,
        args.outputBIM, args.outputEDP,
        args.outputDM, args.outputDV,
        output_path = args.dirnameOutput,
        detailed_results = args.detailed_results,
        coupled_EDP = args.coupled_EDP,
        log_file = args.log_file,
        event_time = args.event_time,
        ground_failure = args.ground_failure,
        auto_script_path = args.auto_script,
        resource_dir = args.resource_dir)

    log_msg('pelicun calculation completed.')

if __name__ == '__main__':

    main(sys.argv[1:])
