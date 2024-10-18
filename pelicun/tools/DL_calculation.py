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
# John Vouvakis Manousakis

"""
This module provides the main functionality to run a pelicun
calculation from the command line.

"""

from __future__ import annotations
from time import gmtime
from time import strftime
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import colorama
from colorama import Fore
from colorama import Style

import jsonschema
from jsonschema import validate

import pelicun
from pelicun.auto import auto_populate
from pelicun.base import str2bool
from pelicun.base import convert_to_MultiIndex
from pelicun.base import convert_to_SimpleIndex
from pelicun.base import describe
from pelicun.base import get
from pelicun.base import update
from pelicun.base import is_specified
from pelicun.base import is_unspecified
from pelicun import base
from pelicun.assessment import DLCalculationAssessment
from pelicun.warnings import PelicunInvalidConfigError


colorama.init()
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))


def log_msg(msg, color_codes=None):
    """
    Prints a formatted string to stdout in the form of a log. Includes
    a timestamp.

    Parameters
    ----------
    msg: str
        The message to be printed.

    """
    if color_codes:
        cpref, csuff = color_codes
        formatted_msg = (
            f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} '
            f'{cpref}'
            f'{msg}'
            f'{csuff}'
        )
    else:
        formatted_msg = f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} {msg}'

    print(formatted_msg)


# list of output files help perform safe initialization of output dir
known_output_files = [
    'DEM_sample.zip',
    'DEM_stats.csv',
    'CMP_sample.zip',
    'CMP_stats.csv',
    'DMG_sample.zip',
    'DMG_stats.csv',
    'DMG_grp.zip',
    'DMG_grp_stats.csv',
    'DV_repair_sample.zip',
    'DV_repair_stats.csv',
    'DV_repair_grp.zip',
    'DV_repair_grp_stats.csv',
    'DV_repair_agg.zip',
    'DV_repair_agg_stats.csv',
    'DL_summary.csv',
    'DL_summary_stats.csv',
]

full_out_config = {
    'Demand': {'Sample': True, 'Statistics': True},
    'Asset': {'Sample': True, 'Statistics': True},
    'Damage': {
        'Sample': True,
        'Statistics': True,
        'GroupedSample': True,
        'GroupedStatistics': True,
    },
    'Loss': {
        'Repair': {
            'Sample': True,
            'Statistics': True,
            'GroupedSample': True,
            'GroupedStatistics': True,
            'AggregateSample': True,
            'AggregateStatistics': True,
        }
    },
    'Format': {'CSV': True, 'JSON': True},
}

regional_out_config = {
    'Demand': {'Sample': True, 'Statistics': False},
    'Asset': {'Sample': True, 'Statistics': False},
    'Damage': {
        'Sample': False,
        'Statistics': False,
        'GroupedSample': True,
        'GroupedStatistics': True,
    },
    'Loss': {
        'Repair': {
            'Sample': True,
            'Statistics': True,
            'GroupedSample': True,
            'GroupedStatistics': False,
            'AggregateSample': True,
            'AggregateStatistics': True,
        }
    },
    'Format': {'CSV': False, 'JSON': True},
    'Settings': {
        'CondenseDS': True,
        'SimpleIndexInJSON': True,
        'AggregateColocatedComponentResults': True,
    },
}

pbe_settings = {
    'CondenseDS': False,
    'SimpleIndexInJSON': False,
    'AggregateColocatedComponentResults': True,
}


def convert_df_to_dict(df, axis=1):
    """
    Convert a pandas DataFrame to a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be converted.
    axis : int, optional
        The axis to consider for the conversion.
        * If 1 (default), the DataFrame is used as-is.
        * If 0, the DataFrame is transposed before conversion.

    Returns
    -------
    dict
        A dictionary representation of the DataFrame. The structure of
        the dictionary depends on the levels in the DataFrame's
        MultiIndex columns.

    Raises
    ------
    ValueError
        If the axis is not 0 or 1.

    Notes
    -----
    * If the columns have multiple levels, the function will
      recursively convert sub-DataFrames.
    * If the column labels at any level are numeric, they will be
      converted to a list of floats.
    * If the column labels are non-numeric, a dictionary will be
      created with the index labels as keys and the corresponding data
      as values.

    """

    out_dict = {}

    if axis == 1:
        df_in = df
    elif axis == 0:
        df_in = df.T
    else:
        raise ValueError('`axis` must be `0` or `1`')

    MI = df_in.columns

    for label in MI.unique(level=0):
        out_dict.update({label: np.nan})

        sub_df = df_in[label]

        skip_sub = True

        if MI.nlevels > 1:
            skip_sub = False

            if isinstance(sub_df, pd.Series):
                skip_sub = True
            elif (len(sub_df.columns) == 1) and (sub_df.columns[0] == ''):
                skip_sub = True

            if not skip_sub:
                out_dict[label] = convert_df_to_dict(sub_df)

        if skip_sub:
            if np.all(sub_df.index.astype(str).str.isnumeric()):
                out_dict_label = df_in[label].astype(float)
                out_dict[label] = out_dict_label.tolist()
            else:
                out_dict[label] = {key: sub_df.loc[key] for key in sub_df.index}

    return out_dict


def run_pelicun(
    config_path,
    demand_file,
    output_path,
    realizations,
    detailed_results,
    coupled_EDP,
    auto_script_path,
    custom_model_dir,
    color_warnings,
    output_format,
):
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
    output_format: str
        Type of output format, JSON or CSV.
    custom_model_dir: string, optional
        Path pointing to a directory with files that define user-provided model
        parameters for a customized damage and loss assessment.
    color_warnings: bool, optional
        If True, warnings are printed in red on the console. If output
        is redirected to a file, it will contain ANSI codes. When
        viewed on the console with `cat`, `less`, or similar utilities,
        the color will be shown.

    Raises
    ------
    PelicunInvalidConfigError
        When the config file is invalid or contains missing entries.

    """

    log_msg('First line of DL_calculation')

    # Initial setup -----------------------------------------------------------

    # get the absolute path to the config file
    config_path = Path(config_path).resolve()

    # If the output path was not specified, results are saved in the
    # directory of the input file.
    if output_path is None:
        output_path = config_path.parents[0]
    else:
        output_path = Path(output_path)
    # create the directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # parse the config file
    config = _parse_config_file(
        config_path,
        output_path,
        auto_script_path,
        demand_file,
        realizations,
        coupled_EDP,
        detailed_results,
        output_format,
    )

    # Initialize the array that we'll use to collect the output file names
    out_files = []

    _remove_existing_files(output_path, known_output_files)

    # Run the assessment
    assessment = DLCalculationAssessment(config_options=get(config, 'DL/Options'))

    assessment.calculate_demand(
        demand_path=Path(get(config, 'DL/Demands/DemandFilePath')).resolve(),
        collapse_limits=get(config, 'DL/Demands/CollapseLimits', default=None),
        length_unit=get(config, 'GeneralInformation/units/length', default=None),
        demand_calibration=get(config, 'DL/Demands/Calibration', default=None),
        sample_size=get(config, 'DL/Options/Sampling/SampleSize'),
        coupled_demands=get(config, 'DL/Demands/CoupledDemands', default=False),
        demand_cloning=get(config, 'DL/Demands/DemandCloning', default=None),
        residual_drift_inference=get(
            config, 'DL/Demands/InferResidualDrift', default=None
        ),
    )

    if is_specified(config, 'DL/Asset'):
        assessment.calculate_asset(
            num_stories=get(config, 'DL/Asset/NumberOfStories', default=None),
            component_assignment_file=get(
                config, 'DL/Asset/ComponentAssignmentFile', default=None
            ),
            collapse_fragility_demand_type=get(
                config, 'DL/Damage/CollapseFragility/DemandType', default=None
            ),
            add_irreparable_damage_columns=get(
                config, 'DL/Damage/IrreparableDamage', default=False
            ),
            component_sample_file=get(
                config, 'DL/Asset/ComponentSampleFile', default=None
            ),
        )

    if is_specified(config, 'DL/Damage'):
        assessment.calculate_damage(
            length_unit=get(config, 'GeneralInformation/units/length'),
            component_database=get(config, 'DL/Asset/ComponentDatabase'),
            component_database_path=get(
                config, 'DL/Asset/ComponentDatabasePath', default=None
            ),
            collapse_fragility=get(
                config, 'DL/Damage/CollapseFragility', default=None
            ),
            is_for_water_network_assessment=is_specified(
                config, 'DL/Asset/ComponentDatabase/Water'
            ),
            irreparable_damage=get(
                config, 'DL/Damage/IrreparableDamage', default=None
            ),
            damage_process_approach=get(
                config, 'DL/Damage/DamageProcess', default=None
            ),
            damage_process_file_path=get(
                config, 'DL/Damage/DamageProcessFilePath', default=None
            ),
            custom_model_dir=custom_model_dir,
        )

    if is_unspecified(config, 'DL/Losses/Repair'):
        agg_repair = None
    else:
        # Currently we only support `Repair` consequences.
        # We will need to make changes here when we start to include
        # more consequences.

        agg_repair, _ = assessment.calculate_loss(
            loss_map_approach=get(config, 'DL/Losses/Repair/MapApproach'),
            occupancy_type=get(config, 'DL/Asset/OccupancyType'),
            consequence_database=get(config, 'DL/Losses/Repair/ConsequenceDatabase'),
            consequence_database_path=get(
                config, 'DL/Losses/Repair/ConsequenceDatabasePath'
            ),
            custom_model_dir=custom_model_dir,
            damage_process_approach=get(
                config, 'DL/Damage/DamageProcess', default='User Defined'
            ),
            replacement_cost_parameters=get(
                config, 'DL/Losses/Repair/ReplacementCost'
            ),
            replacement_time_parameters=get(
                config, 'DL/Losses/Repair/ReplacementTime'
            ),
            replacement_carbon_parameters=get(
                config, 'DL/Losses/Repair/ReplacementCarbon'
            ),
            replacement_energy_parameters=get(
                config, 'DL/Losses/Repair/ReplacementEnergy'
            ),
            loss_map_path=get(config, 'DL/Losses/Repair/MapFilePath'),
            decision_variables=_parse_decision_variables(config),
        )

    summary, summary_stats = _result_summary(assessment, agg_repair)

    # Save the results into files

    if is_specified(config, 'DL/Outputs/Demand'):
        output_config = get(config, 'DL/Outputs/Demand')
        _demand_save(output_config, assessment, output_path, out_files)

    if is_specified(config, 'DL/Outputs/Asset'):
        output_config = get(config, 'DL/Outputs/Asset')
        _asset_save(
            output_config,
            assessment,
            output_path,
            out_files,
            aggregate_colocated=get(
                config,
                'DL/Outputs/Settings/AggregateColocatedComponentResults',
                default=False,
            ),
        )

    if is_specified(config, 'DL/Outputs/Damage'):
        output_config = get(config, 'DL/Outputs/Damage')
        _damage_save(
            output_config,
            assessment,
            output_path,
            out_files,
            aggregate_colocated=get(
                config,
                'DL/Outputs/Settings/AggregateColocatedComponentResults',
                default=False,
            ),
            condense_ds=get(
                config,
                'DL/Outputs/Settings/CondenseDS',
                default=False,
            ),
        )

    if is_specified(config, 'DL/Outputs/Loss/Repair'):
        output_config = get(config, 'DL/Outputs/Loss/Repair')
        _loss_save(
            output_config,
            assessment,
            output_path,
            out_files,
            agg_repair,
            aggregate_colocated=get(
                config,
                'DL/Outputs/Settings/AggregateColocatedComponentResults',
                default=False,
            ),
        )
    _summary_save(summary, summary_stats, output_path, out_files)
    _create_json_files_if_requested(config, out_files, output_path)
    _remove_csv_files_if_not_requested(config, out_files, output_path)


def _parse_decision_variables(config):
    decision_variables = []
    if get(config, 'DL/Losses/Repair/DecisionVariables', default=False) is not False:
        for DV_i, DV_status in get(
            config, 'DL/Losses/Repair/DecisionVariables'
        ).items():
            if DV_status is True:
                decision_variables.append(DV_i)
    return decision_variables


def _remove_csv_files_if_not_requested(config, out_files, output_path):
    # Don't proceed if CSV files were requested.
    if get(config, 'DL/Outputs/Format/CSV', default=False) is True:
        return

    for filename in out_files:
        # keep the DL_summary and DL_summary_stats files
        if 'DL_summary' in filename:
            continue
        os.remove(output_path / filename)


def _summary_save(summary, summary_stats, output_path, out_files):
    # save summary sample
    if summary is not None:
        summary.to_csv(output_path / 'DL_summary.csv', index_label='#')
        out_files.append('DL_summary.csv')

    # save summary statistics
    if summary_stats is not None:
        summary_stats.to_csv(output_path / 'DL_summary_stats.csv')
        out_files.append('DL_summary_stats.csv')


def _parse_config_file(
    config_path,
    output_path,
    auto_script_path,
    demand_file,
    realizations,
    coupled_EDP,
    detailed_results,
    output_format,
):
    # open the config file and parse it
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # load the schema
    with open(
        f'{base.pelicun_path}/settings/input_schema.json', 'r', encoding='utf-8'
    ) as f:
        schema = json.load(f)

    # Validate the configuration against the schema
    try:
        validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise PelicunInvalidConfigError(
            'The provided config file does not conform to the schema.'
        ) from exc

    if is_unspecified(config, 'DL'):
        log_msg('Damage and Loss configuration missing from config file. ')

        if auto_script_path is None:
            raise PelicunInvalidConfigError('No `DL` entry in config file.')

        log_msg('Trying to auto-populate')

        config_ap, CMP = auto_populate(config, auto_script_path)

        if is_unspecified(config_ap, 'DL'):
            raise PelicunInvalidConfigError(
                'No `DL` entry in config file, and '
                'the prescribed auto-population script failed to identify '
                'a valid damage and loss configuration for this asset. '
            )

        # add the demand information
        update(config_ap, '/DL/Demands/DemandFilePath', demand_file)
        update(config_ap, '/DL/Demands/SampleSize', str(realizations))

        if coupled_EDP is True:
            update(config_ap, 'DL/Demands/CoupledDemands', True)

        else:
            update(
                config_ap,
                'DL/Demands/Calibration',
                {'ALL': {'DistributionFamily': 'lognormal'}},
            )

        # save the component data
        CMP.to_csv(output_path / 'CMP_QNT.csv')

        # update the config file with the location
        update(
            config_ap,
            'DL/Asset/ComponentAssignmentFile',
            str(output_path / 'CMP_QNT.csv'),
        )

        # if detailed results are not requested, add a lean output config
        if detailed_results is False:
            update(config_ap, 'DL/Outputs', regional_out_config)
        else:
            update(config_ap, 'DL/Outputs', full_out_config)
            # add output settings from regional output config
            if is_unspecified(config_ap, 'DL/Outputs/Settings'):
                update(config_ap, 'DL/Outputs/Settings', {})

            config_ap['DL']['Outputs']['Settings'].update(
                regional_out_config['Settings']
            )

        # save the extended config to a file
        config_ap_path = Path(config_path.stem + '_ap.json').resolve()

        with open(config_ap_path, 'w', encoding='utf-8') as f:
            json.dump(config_ap, f, indent=2)

        update(config, 'DL', get(config_ap, 'DL'))

    # sample size
    sample_size_str = get(config, 'DL/Options/Sampling/SampleSize')
    if not sample_size_str:
        sample_size_str = get(config, 'DL/Demands/SampleSize')
        if not sample_size_str:
            raise PelicunInvalidConfigError(
                'Sample size not provided in config file.'
            )
    update(config, 'DL/Options/Sampling/SampleSize', int(sample_size_str))

    # provide all outputs if the files are not specified
    if is_unspecified(config, 'DL/Outputs'):
        update(config, 'DL/Outputs', full_out_config)

    # provide outputs in CSV by default
    if is_unspecified(config, 'DL/Outputs/Format'):
        update(config, 'DL/Outputs/Format', {'CSV': True, 'JSON': False})

    # override file format specification if the output_format is
    # provided
    if output_format is not None:
        update(
            config,
            'DL/Outputs/Format',
            {
                'CSV': 'csv' in output_format,
                'JSON': 'json' in output_format,
            },
        )

    # add empty Settings to output config to simplify code below
    if is_unspecified(config, 'DL/Outputs/Settings'):
        update(config, 'DL/Outputs/Settings', pbe_settings)

    if is_unspecified(config, 'DL/Demands'):
        raise PelicunInvalidConfigError('Demand configuration missing.')

    if is_unspecified(config, 'DL/Asset'):
        raise PelicunInvalidConfigError('Asset configuration missing.')

    # ensure a length unit is specified in the config file.
    if is_unspecified(config, 'GeneralInformation/units/length'):
        raise PelicunInvalidConfigError(
            'No default length unit provided in the input file.'
        )

    update(
        config,
        'DL/Options/LogFile',
        'pelicun_log.txt',
        only_if_empty_or_none=True,
    )
    update(
        config,
        'DL/Options/Verbose',
        True,
        only_if_empty_or_none=True,
    )

    # if the user did not prescribe anything for ListAllDamageStates,
    # then use True as default for DL_calculations regardless of what
    # the Pelicun default is.
    update(
        config, 'DL/Options/ListAllDamageStates', True, only_if_empty_or_none=True
    )

    # if the demand file location is not specified in the config file
    # assume there is a `response.csv` file next to the config file.
    update(
        config,
        'DL/Demands/DemandFilePath',
        config_path.parent / 'response.csv',
        only_if_empty_or_none=True,
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # backwards-compatibility for v3.2 and earlier | remove after v4.0
    if get(config, 'DL/Losses/BldgRepair', default=False):
        update(config, 'DL/Losses/Repair', get(config, 'DL/Losses/BldgRepair'))
    if get(config, 'DL/Outputs/Loss/BldgRepair', default=False):
        update(
            config,
            'DL/Outputs/Loss/Repair',
            get(config, 'DL/Outputs/Loss/BldgRepair'),
        )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Cast NumberOfStories to int
    if is_specified(config, 'DL/Asset/NumberOfStories'):
        update(
            config,
            'DL/Asset/NumberOfStories',
            int(get(config, 'DL/Asset/NumberOfStories')),
        )

    # Ensure `DL/Demands/InferResidualDrift` contains a `method`
    if is_specified(config, 'DL/Demands/InferResidualDrift') and is_unspecified(
        config, 'DL/Demands/InferResidualDrift/method'
    ):
        raise PelicunInvalidConfigError(
            'No method is specified in residual drift inference configuration.'
        )

    # Ensure `DL/Damage/CollapseFragility` contains all required keys.
    if is_specified(config, 'DL/Damage/CollapseFragility'):
        for thing in ('CapacityDistribution', 'CapacityMedian', 'Theta_1'):
            if is_unspecified(config, f'DL/Damage/CollapseFragility/{thing}'):
                raise PelicunInvalidConfigError(
                    f'`{thing}` is missing from DL/Damage/CollapseFragility'
                    f' in the configuration file.'
                )

    # Ensure `DL/Damage/IrreparableDamage` contains all required keys.
    if is_specified(config, 'DL/Damage/IrreparableDamage'):
        for thing in ('DriftCapacityMedian', 'DriftCapacityLogStd'):
            if is_unspecified(config, f'DL/Damage/IrreparableDamage/{thing}'):
                raise PelicunInvalidConfigError(
                    f'`{thing}` is missing from DL/Damage/IrreparableDamage'
                    f' in the configuration file.'
                )

    # If the damage process approach is `User Defined` there needs to
    # be a damage process file path.
    if get(config, 'DL/Damage/DamageProcess') == 'User Defined' and is_unspecified(
        config, 'DL/Damage/DamageProcessFilePath'
    ):
        raise PelicunInvalidConfigError(
            'When `DL/Damage/DamageProcess` is set to `User Defined`, '
            'a path needs to be specified under '
            '`DL/Damage/DamageProcessFilePath`.'
        )

    # Getting results requires running the calculations.
    if is_specified(config, 'DL/Outputs/Asset') and is_unspecified(
        config, 'DL/Asset'
    ):
        raise PelicunInvalidConfigError(
            'No asset data specified in config file. '
            'Cannot generate asset model outputs.'
        )

    if is_specified(config, 'DL/Outputs/Damage') and is_unspecified(
        config, 'DL/Damage'
    ):
        raise PelicunInvalidConfigError(
            'No damage data specified in config file. '
            'Cannot generate damage model outputs.'
        )

    if is_specified(config, 'DL/Outputs/Loss') and is_unspecified(
        config, 'DL/Losses'
    ):
        raise PelicunInvalidConfigError(
            'No loss data specified in config file. '
            'Cannot generate loss model outputs.'
        )

    # Ensure only one of `component_assignment_file` or
    # `component_sample_file` is provided.
    if is_specified(config, 'DL/Asset'):
        if (
            (get(config, 'DL/Asset/ComponentAssignmentFile') is None)
            and (get(config, 'DL/Asset/ComponentSampleFile') is None)
            or (
                (get(config, 'DL/Asset/ComponentAssignmentFile') is not None)
                and (get(config, 'DL/Asset/ComponentSampleFile') is not None)
            )
        ):
            msg = (
                'In the asset model configuraiton, it is '
                'required to specify one of `component_assignment_file` '
                'or `component_sample_file`, but not both.'
            )
            raise ValueError(msg)

    return config


def _create_json_files_if_requested(config, out_files, output_path):
    # If not requested, simply return
    if get(config, 'DL/Outputs/Format/JSON', default=False) is False:
        return

    for filename in out_files:
        filename_json = filename[:-3] + 'json'

        if (
            get(config, 'DL/Outputs/Settings/SimpleIndexInJSON', default=False)
            is True
        ):
            df = pd.read_csv(output_path / filename, index_col=0)
        else:
            df = convert_to_MultiIndex(
                pd.read_csv(output_path / filename, index_col=0), axis=1
            )

        if 'Units' in df.index:
            df_units = convert_to_SimpleIndex(
                df.loc['Units', :].to_frame().T, axis=1
            )

            df.drop('Units', axis=0, inplace=True)

            out_dict = convert_df_to_dict(df)

            out_dict.update(
                {
                    'Units': {
                        col: df_units.loc['Units', col] for col in df_units.columns
                    }
                }
            )

        else:
            out_dict = convert_df_to_dict(df)

        with open(output_path / filename_json, 'w', encoding='utf-8') as f:
            json.dump(out_dict, f, indent=2)


def _result_summary(assessment, agg_repair):
    damage_sample = assessment.damage.save_sample()
    if damage_sample is None or agg_repair is None:
        return None, None

    damage_sample = damage_sample.groupby(level=['cmp', 'ds'], axis=1).sum()
    damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)

    if 'collapse-1' in damage_sample_s.columns:
        damage_sample_s['collapse'] = damage_sample_s['collapse-1']
    else:
        damage_sample_s['collapse'] = np.zeros(damage_sample_s.shape[0])

    if 'irreparable-1' in damage_sample_s.columns:
        damage_sample_s['irreparable'] = damage_sample_s['irreparable-1']
    else:
        damage_sample_s['irreparable'] = np.zeros(damage_sample_s.shape[0])

    if agg_repair is not None:
        agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)

    else:
        agg_repair_s = pd.DataFrame()

    summary = pd.concat(
        [agg_repair_s, damage_sample_s[['collapse', 'irreparable']]], axis=1
    )

    summary_stats = describe(summary)

    return summary, summary_stats


def _parse_requested_output_file_names(output_config):
    out_reqs = []
    for out, val in output_config.items():
        if val is True:
            out_reqs.append(out)
    return set(out_reqs)


def _demand_save(output_config, assessment, output_path, out_files):
    out_reqs = _parse_requested_output_file_names(output_config)

    demand_sample, demand_units = assessment.demand.save_sample(save_units=True)
    demand_units = demand_units.to_frame().T

    if 'Sample' in out_reqs:
        demand_sample_s = pd.concat([demand_sample, demand_units])
        demand_sample_s = convert_to_SimpleIndex(demand_sample_s, axis=1)
        demand_sample_s.to_csv(
            output_path / 'DEM_sample.zip',
            index_label=demand_sample_s.columns.name,
            compression={'method': 'zip', 'archive_name': 'DEM_sample.csv'},
        )
        out_files.append('DEM_sample.zip')

    if 'Statistics' in out_reqs:
        demand_stats = describe(demand_sample)
        demand_stats = pd.concat([demand_stats, demand_units])
        demand_stats = convert_to_SimpleIndex(demand_stats, axis=1)
        demand_stats.to_csv(
            output_path / 'DEM_stats.csv',
            index_label=demand_stats.columns.name,
        )
        out_files.append('DEM_stats.csv')


def _asset_save(
    output_config, assessment, output_path, out_files, aggregate_colocated=False
):
    cmp_sample, cmp_units = assessment.asset.save_cmp_sample(save_units=True)
    cmp_units = cmp_units.to_frame().T

    if aggregate_colocated:
        cmp_units = cmp_units.groupby(level=['cmp', 'loc', 'dir'], axis=1).first()
        cmp_groupby_uid = cmp_sample.groupby(level=['cmp', 'loc', 'dir'], axis=1)
        cmp_sample = cmp_groupby_uid.sum().mask(cmp_groupby_uid.count() == 0, np.nan)

    out_reqs = _parse_requested_output_file_names(output_config)

    if 'Sample' in out_reqs:
        cmp_sample_s = pd.concat([cmp_sample, cmp_units])

        cmp_sample_s = convert_to_SimpleIndex(cmp_sample_s, axis=1)
        cmp_sample_s.to_csv(
            output_path / 'CMP_sample.zip',
            index_label=cmp_sample_s.columns.name,
            compression={'method': 'zip', 'archive_name': 'CMP_sample.csv'},
        )
        out_files.append('CMP_sample.zip')

    if 'Statistics' in out_reqs:
        cmp_stats = describe(cmp_sample)
        cmp_stats = pd.concat([cmp_stats, cmp_units])

        cmp_stats = convert_to_SimpleIndex(cmp_stats, axis=1)
        cmp_stats.to_csv(
            output_path / 'CMP_stats.csv', index_label=cmp_stats.columns.name
        )
        out_files.append('CMP_stats.csv')


def _damage_save(
    output_config,
    assessment,
    output_path,
    out_files,
    aggregate_colocated=False,
    condense_ds=False,
):
    damage_sample, damage_units = assessment.damage.save_sample(save_units=True)
    damage_units = damage_units.to_frame().T

    if aggregate_colocated:
        damage_units = damage_units.groupby(
            level=['cmp', 'loc', 'dir', 'ds'], axis=1
        ).first()
        damage_groupby_uid = damage_sample.groupby(
            level=['cmp', 'loc', 'dir', 'ds'], axis=1
        )
        damage_sample = damage_groupby_uid.sum().mask(
            damage_groupby_uid.count() == 0, np.nan
        )

    out_reqs = _parse_requested_output_file_names(output_config)

    if 'Sample' in out_reqs:
        damage_sample_s = pd.concat([damage_sample, damage_units])

        damage_sample_s = convert_to_SimpleIndex(damage_sample_s, axis=1)
        damage_sample_s.to_csv(
            output_path / 'DMG_sample.zip',
            index_label=damage_sample_s.columns.name,
            compression={
                'method': 'zip',
                'archive_name': 'DMG_sample.csv',
            },
        )
        out_files.append('DMG_sample.zip')

    if 'Statistics' in out_reqs:
        damage_stats = describe(damage_sample)
        damage_stats = pd.concat([damage_stats, damage_units])

        damage_stats = convert_to_SimpleIndex(damage_stats, axis=1)
        damage_stats.to_csv(
            output_path / 'DMG_stats.csv',
            index_label=damage_stats.columns.name,
        )
        out_files.append('DMG_stats.csv')

    if out_reqs.intersection({'GroupedSample', 'GroupedStatistics'}):
        if aggregate_colocated:
            damage_groupby = damage_sample.groupby(level=['cmp', 'ds'], axis=1)
            damage_units = damage_units.groupby(level=['cmp', 'ds'], axis=1).first()
        else:
            damage_groupby = damage_sample.groupby(
                level=['cmp', 'loc', 'dir', 'ds'], axis=1
            )
            damage_units = damage_units.groupby(
                level=['cmp', 'loc', 'dir', 'ds'], axis=1
            ).first()

        grp_damage = damage_groupby.sum().mask(damage_groupby.count() == 0, np.nan)

        # if requested, condense DS output
        if condense_ds:
            # replace non-zero values with 1
            grp_damage = grp_damage.mask(grp_damage.astype(np.float64).values > 0, 1)

            # get the corresponding DS for each column
            ds_list = grp_damage.columns.get_level_values('ds').astype(int)

            # replace ones with the corresponding DS in each cell
            grp_damage = grp_damage.mul(ds_list, axis=1)

            # aggregate across damage state indices
            damage_groupby_2 = grp_damage.groupby(level=['cmp', 'ds'], axis=1)

            # choose the max value
            # i.e., the governing DS for each comp-loc pair
            grp_damage = damage_groupby_2.max().mask(
                damage_groupby_2.count() == 0, np.nan
            )

            # aggregate units to the same format
            # assume identical units across locations for each comp
            damage_units = damage_units.groupby(level=['cmp', 'ds'], axis=1).first()

        else:
            # otherwise, aggregate damage quantities for each comp
            damage_groupby_2 = grp_damage.groupby(level='cmp', axis=1)

            # preserve NaNs
            grp_damage = damage_groupby_2.sum().mask(
                damage_groupby_2.count() == 0, np.nan
            )

            # and aggregate units to the same format
            damage_units = damage_units.groupby(level='cmp', axis=1).first()

        if 'GroupedSample' in out_reqs:
            grp_damage_s = pd.concat([grp_damage, damage_units])

            grp_damage_s = convert_to_SimpleIndex(grp_damage_s, axis=1)
            grp_damage_s.to_csv(
                output_path / 'DMG_grp.zip',
                index_label=grp_damage_s.columns.name,
                compression={
                    'method': 'zip',
                    'archive_name': 'DMG_grp.csv',
                },
            )
            out_files.append('DMG_grp.zip')

        if 'GroupedStatistics' in out_reqs:
            grp_stats = describe(grp_damage)
            grp_stats = pd.concat([grp_stats, damage_units])

            grp_stats = convert_to_SimpleIndex(grp_stats, axis=1)
            grp_stats.to_csv(
                output_path / 'DMG_grp_stats.csv',
                index_label=grp_stats.columns.name,
            )
            out_files.append('DMG_grp_stats.csv')


def _loss_save(
    output_config,
    assessment,
    output_path,
    out_files,
    agg_repair,
    aggregate_colocated=False,
):
    repair_sample, repair_units = assessment.loss.ds_model.save_sample(
        save_units=True
    )
    repair_units = repair_units.to_frame().T

    if aggregate_colocated:
        repair_units = repair_units.groupby(
            level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir'], axis=1
        ).first()
        repair_groupby_uid = repair_sample.groupby(
            level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir'], axis=1
        )
        repair_sample = repair_groupby_uid.sum().mask(
            repair_groupby_uid.count() == 0, np.nan
        )

    out_reqs = _parse_requested_output_file_names(output_config)

    if 'Sample' in out_reqs:
        repair_sample_s = repair_sample.copy()
        repair_sample_s = pd.concat([repair_sample_s, repair_units])

        repair_sample_s = convert_to_SimpleIndex(repair_sample_s, axis=1)
        repair_sample_s.to_csv(
            output_path / 'DV_repair_sample.zip',
            index_label=repair_sample_s.columns.name,
            compression={
                'method': 'zip',
                'archive_name': 'DV_repair_sample.csv',
            },
        )
        out_files.append('DV_repair_sample.zip')

    if 'Statistics' in out_reqs:
        repair_stats = describe(repair_sample)
        repair_stats = pd.concat([repair_stats, repair_units])

        repair_stats = convert_to_SimpleIndex(repair_stats, axis=1)
        repair_stats.to_csv(
            output_path / 'DV_repair_stats.csv',
            index_label=repair_stats.columns.name,
        )
        out_files.append('DV_repair_stats.csv')

    if out_reqs.intersection({'GroupedSample', 'GroupedStatistics'}):
        repair_groupby = repair_sample.groupby(level=['dv', 'loss', 'dmg'], axis=1)
        repair_units = repair_units.groupby(
            level=['dv', 'loss', 'dmg'], axis=1
        ).first()
        grp_repair = repair_groupby.sum().mask(repair_groupby.count() == 0, np.nan)

        if 'GroupedSample' in out_reqs:
            grp_repair_s = pd.concat([grp_repair, repair_units])

            grp_repair_s = convert_to_SimpleIndex(grp_repair_s, axis=1)
            grp_repair_s.to_csv(
                output_path / 'DV_repair_grp.zip',
                index_label=grp_repair_s.columns.name,
                compression={
                    'method': 'zip',
                    'archive_name': 'DV_repair_grp.csv',
                },
            )
            out_files.append('DV_repair_grp.zip')

        if 'GroupedStatistics' in out_reqs:
            grp_stats = describe(grp_repair)
            grp_stats = pd.concat([grp_stats, repair_units])

            grp_stats = convert_to_SimpleIndex(grp_stats, axis=1)
            grp_stats.to_csv(
                output_path / 'DV_repair_grp_stats.csv',
                index_label=grp_stats.columns.name,
            )
            out_files.append('DV_repair_grp_stats.csv')

    if out_reqs.intersection({'AggregateSample', 'AggregateStatistics'}):
        if 'AggregateSample' in out_reqs:
            agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)
            agg_repair_s.to_csv(
                output_path / 'DV_repair_agg.zip',
                index_label=agg_repair_s.columns.name,
                compression={
                    'method': 'zip',
                    'archive_name': 'DV_repair_agg.csv',
                },
            )
            out_files.append('DV_repair_agg.zip')

        if 'AggregateStatistics' in out_reqs:
            agg_stats = convert_to_SimpleIndex(describe(agg_repair), axis=1)
            agg_stats.to_csv(
                output_path / 'DV_repair_agg_stats.csv',
                index_label=agg_stats.columns.name,
            )
            out_files.append('DV_repair_agg_stats.csv')


def _get_color_codes(color_warnings):
    if color_warnings:
        cpref = Fore.RED
        csuff = Style.RESET_ALL
    else:
        cpref = csuff = ''

    return (cpref, csuff)


def _remove_existing_files(output_path, known_output_files):
    """
    Remove known existing files from the specified output path.

    This function initializes the output folder by removing files that
    already exist in the `known_output_files` list.

    Parameters
    ----------
    output_path : Path
        The path to the output folder where files are located.
    known_output_files : list of str
        A list of filenames that are expected to exist and should be
        removed from the output folder.

    Raises
    ------
    OSError
        If an error occurs while attempting to remove a file, an
        OSError will be raised with the specific details of the
        failure.
    """
    # Initialize the output folder - i.e., remove existing output files from
    # there
    files = os.listdir(output_path)
    for filename in files:
        if filename in known_output_files:
            try:
                os.remove(output_path / filename)
            except OSError as exc:
                raise OSError(
                    f'Error occurred while removing '
                    f'`{output_path / filename}`: {exc}'
                ) from exc


def main():
    """
    Main method to parse arguments and run the pelicun calculation.

    """
    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--filenameDL',
        help='Path to the damage and loss (DL) configuration file.',
    )
    parser.add_argument(
        '-d',
        '--demandFile',
        default=None,
        help='Path to the file containing demand data.',
    )
    parser.add_argument(
        '-s',
        '--Realizations',
        default=None,
        help='Number of realizations to run in the probabilistic model.',
    )
    parser.add_argument(
        '--dirnameOutput',
        default=None,
        help='Directory where output files will be stored.',
    )
    parser.add_argument(
        '--detailed_results',
        default=True,
        type=str2bool,
        nargs='?',
        const=True,
        help='Generate detailed results (True/False). Defaults to True.',
    )
    parser.add_argument(
        '--coupled_EDP',
        default=False,
        type=str2bool,
        nargs='?',
        const=False,
        help=(
            'Consider coupled Engineering Demand Parameters (EDPs) '
            'in calculations (True/False). Defaults to False.'
        ),
    )
    parser.add_argument(
        '--log_file',
        default=True,
        type=str2bool,
        nargs='?',
        const=True,
        help='Generate a log file (True/False). Defaults to True.',
    )
    parser.add_argument(
        '--auto_script',
        default=None,
        help='Optional path to a config auto-generation script.',
    )
    parser.add_argument(
        '--custom_model_dir',
        default=None,
        help='Directory containing custom model data.',
    )
    parser.add_argument(
        '--output_format',
        default=None,
        help='Desired output format for the results.',
    )
    parser.add_argument(
        '--color_warnings',
        default=False,
        type=str2bool,
        nargs='?',
        const=False,
        help=(
            'Enable colored warnings in the console '
            'output (True/False). Defaults to False.'
        ),
    )
    parser.add_argument(
        '--ground_failure',
        default=False,
        type=str2bool,
        nargs='?',
        const=False,
        help='Currently not used. Soon to be deprecated.',
    )
    parser.add_argument(
        '--regional',
        default=False,
        type=str2bool,
        nargs='?',
        const=False,
        help='Currently not used. Soon to be deprecated.',
    )
    parser.add_argument('--resource_dir', default=None)

    if not args:
        print(f'Welcome. This is pelicun version {pelicun.__version__}')
        print(
            'To access the documentation visit '
            'https://nheri-simcenter.github.io/pelicun/index.html'
        )
        print()
        parser.print_help()
        return

    args = parser.parse_args(args)

    log_msg('Initializing pelicun calculation.')

    run_pelicun(
        config_path=args.filenameDL,
        demand_file=args.demandFile,
        output_path=args.dirnameOutput,
        realizations=args.Realizations,
        detailed_results=args.detailed_results,
        coupled_EDP=args.coupled_EDP,
        auto_script_path=args.auto_script,
        custom_model_dir=args.custom_model_dir,
        color_warnings=args.color_warnings,
        output_format=args.output_format,
    )

    log_msg('pelicun calculation completed.')


if __name__ == '__main__':
    main()
