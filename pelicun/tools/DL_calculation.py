#  # noqa: N999
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

"""Main functionality to run a pelicun calculation from the command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import gmtime, strftime
from typing import Hashable

import colorama
import jsonschema
import numpy as np
import pandas as pd
from colorama import Fore, Style
from jsonschema import validate

from pelicun import base
from pelicun.assessment import DLCalculationAssessment
from pelicun.auto import auto_populate
from pelicun.base import (
    convert_to_MultiIndex,
    convert_to_SimpleIndex,
    describe,
    get,
    is_specified,
    is_unspecified,
    str2bool,
    update,
    update_vals,
)
from pelicun.pelicun_warnings import PelicunInvalidConfigError

colorama.init()
sys.path.insert(0, Path(__file__).resolve().parent.absolute().as_posix())


def log_msg(msg: str, color_codes: tuple[str, str] | None = None) -> None:
    """
    Print a formatted log message with a timestamp.

    Parameters
    ----------
    msg : str
        The message to print.
    color_codes : tuple, optional
        Color codes for formatting the message. Default is None.

    """
    if color_codes:
        cpref, csuff = color_codes
        print(  # noqa: T201
            f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} '
            f'{cpref}'
            f'{msg}'
            f'{csuff}'
        )
    else:
        print(f'{strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())} {msg}')  # noqa: T201


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


def convert_df_to_dict(data: pd.DataFrame | pd.Series, axis: int = 1) -> dict:
    """
    Convert a pandas DataFrame to a dictionary.

    Parameters
    ----------
    data : pd.DataFrame
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
    out_dict: dict[Hashable, object] = {}

    if axis == 1:
        df_in = data
    elif axis == 0:
        df_in = data.T
    else:
        msg = '`axis` must be `0` or `1`'
        raise ValueError(msg)

    multiindex = df_in.columns

    for label in multiindex.unique(level=0):
        out_dict.update({label: np.nan})

        sub_df = df_in[label]

        skip_sub = True

        if multiindex.nlevels > 1:
            skip_sub = False

            if isinstance(sub_df, pd.Series) or (
                (len(sub_df.columns) == 1) and (sub_df.columns[0] == '')  # noqa: PLC1901
            ):
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
    config_path: str,
    demand_file: str,
    output_path: str | None,
    realizations: int,
    auto_script_path: str | None,
    custom_model_dir: str | None,
    output_format: list | None,
    *,
    detailed_results: bool,
    coupled_edp: bool,
) -> None:
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
    realizations: int, optional
        Number of realizations to generate.
    auto_script_path: string, optional
        Path pointing to the location of a Python script with an auto_populate
        method that automatically creates the performance model using data
        provided in the AIM JSON file.
    custom_model_dir: string, optional
        Path pointing to a directory with files that define user-provided model
        parameters for a customized damage and loss assessment.
    output_format: list, optional.
        Type of output format, JSON or CSV.
        Valid options: ['csv', 'json'], ['csv'], ['json'], [], None
    detailed_results: bool, optional
        If False, only the main statistics are saved.
    coupled_edp: bool, optional
        If True, EDPs are not resampled and processed in order.

    """
    # Initial setup -----------------------------------------------------------

    # get the absolute path to the config file
    config_path_p = Path(config_path).resolve()

    # If the output path was not specified, results are saved in the
    # directory of the input file.
    if output_path is None:
        output_path_p = config_path_p.parents[0]
    else:
        output_path_p = Path(output_path).resolve()
    # create the directory if it does not exist
    if not output_path_p.exists():
        output_path_p.mkdir(parents=True)

    # parse the config file
    config = _parse_config_file(
        config_path_p,
        output_path_p,
        Path(auto_script_path).resolve() if auto_script_path is not None else None,
        demand_file,
        realizations,
        output_format,
        coupled_edp=coupled_edp,
        detailed_results=detailed_results,
    )

    # List to keep track of the generated output files.
    out_files: list[str] = []

    _remove_existing_files(output_path_p, known_output_files)

    # Run the assessment
    assessment = DLCalculationAssessment(config_options=get(config, 'DL/Options'))

    assessment.calculate_demand(
        demand_path=Path(get(config, 'DL/Demands/DemandFilePath')).resolve(),
        collapse_limits=get(config, 'DL/Demands/CollapseLimits', default=None),
        length_unit=get(config, 'GeneralInformation/units/length', default=None),
        demand_calibration=get(config, 'DL/Demands/Calibration', default=None),
        sample_size=get(config, 'DL/Options/Sampling/SampleSize'),
        demand_cloning=get(config, 'DL/Demands/DemandCloning', default=None),
        residual_drift_inference=get(
            config, 'DL/Demands/InferResidualDrift', default=None
        ),
        coupled_demands=get(config, 'DL/Demands/CoupledDemands', default=False),
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
            component_sample_file=get(
                config, 'DL/Asset/ComponentSampleFile', default=None
            ),
            add_irreparable_damage_columns=get(
                config, 'DL/Damage/IrreparableDamage', default=False
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
            scaling_specification=get(config, 'DL/Damage/ScalingSpecification'),
            is_for_water_network_assessment=is_specified(
                config, 'DL/Asset/ComponentDatabase/Water'
            ),
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
        _demand_save(output_config, assessment, output_path_p, out_files)

    if is_specified(config, 'DL/Outputs/Asset'):
        output_config = get(config, 'DL/Outputs/Asset')
        _asset_save(
            output_config,
            assessment,
            output_path_p,
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
            output_path_p,
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
        assert agg_repair is not None
        _loss_save(
            output_config,
            assessment,
            output_path_p,
            out_files,
            agg_repair,
            aggregate_colocated=get(
                config,
                'DL/Outputs/Settings/AggregateColocatedComponentResults',
                default=False,
            ),
        )
    _summary_save(summary, summary_stats, output_path_p, out_files)
    _create_json_files_if_requested(config, out_files, output_path_p)
    _remove_csv_files_if_not_requested(config, out_files, output_path_p)


def _parse_decision_variables(config: dict) -> tuple[str, ...]:
    """
    Parse decision variables from the config file.

    Parameters
    ----------
    config : dict
        The configuration dictionary.

    Returns
    -------
    list
        List of decision variables.

    """
    decision_variables: list[str] = []
    if get(config, 'DL/Losses/Repair/DecisionVariables', default=False) is not False:
        for dv_i, dv_status in get(
            config, 'DL/Losses/Repair/DecisionVariables'
        ).items():
            if dv_status is True:
                decision_variables.append(dv_i)
    return tuple(decision_variables)


def _remove_csv_files_if_not_requested(
    config: dict, out_files: list[str], output_path: Path
) -> None:
    """
    Remove CSV files if not requested in config.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    out_files : list
        List of output file names.
    output_path : Path
        Path to the output directory.
    """
    # Don't proceed if CSV files were requested.
    if get(config, 'DL/Outputs/Format/CSV', default=False) is True:
        return

    for filename in out_files:
        # keep the DL_summary and DL_summary_stats files
        if 'DL_summary' in filename:
            continue
        Path(output_path / filename).unlink()


def _summary_save(
    summary: pd.DataFrame,
    summary_stats: pd.DataFrame,
    output_path: Path,
    out_files: list[str],
) -> None:
    """
    Save summary results to CSV files.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary DataFrame.
    summary_stats : pd.DataFrame
        Summary statistics DataFrame.
    output_path : Path
        Path to the output directory.
    out_files : list
        List of output file names.

    """
    # save summary sample
    if summary is not None:
        summary.to_csv(output_path / 'DL_summary.csv', index_label='#')
        out_files.append('DL_summary.csv')

    # save summary statistics
    if summary_stats is not None:
        summary_stats.to_csv(output_path / 'DL_summary_stats.csv')
        out_files.append('DL_summary_stats.csv')


def _parse_config_file(  # noqa: C901
    config_path: Path,
    output_path: Path,
    auto_script_path: Path | None,
    demand_file: str,
    realizations: int,
    output_format: list | None,
    *,
    coupled_edp: bool,
    detailed_results: bool,
) -> dict[str, object]:
    """
    Parse and validate the config file for Pelicun.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    output_path : Path
        Directory for output files.
    auto_script_path : str
        Path to the auto-generation script.
    demand_file : str
        Path to the demand data file.
    realizations : int
        Number of realizations.
    coupled_EDP : bool
        Whether to consider coupled EDPs.
    detailed_results : bool
        Whether to generate detailed results.
    output_format : str
        Output format (CSV, JSON).

    Returns
    -------
    dict
        Parsed and validated configuration.

    Raises
    ------
    PelicunInvalidConfigError
      If the provided config file does not conform to the schema or
      there are issues with the specified values.

    """
    # open the config file and parse it
    with Path(config_path).open(encoding='utf-8') as f:
        config = json.load(f)

    # load the schema
    with Path(f'{base.pelicun_path}/settings/input_schema.json').open(
        encoding='utf-8'
    ) as f:
        schema = json.load(f)

    # Validate the configuration against the schema
    try:
        validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as exc:
        msg = 'The provided config file does not conform to the schema.'
        raise PelicunInvalidConfigError(msg) from exc

    if is_unspecified(config, 'DL'):
        log_msg('Damage and Loss configuration missing from config file. ')

        if auto_script_path is None:
            msg = 'No `DL` entry in config file.'
            raise PelicunInvalidConfigError(msg)

        log_msg('Trying to auto-populate')

        config_ap, comp = auto_populate(config, auto_script_path)

        if is_unspecified(config_ap, 'DL'):
            msg = (
                'No `DL` entry in config file, and '
                'the prescribed auto-population script failed to identify '
                'a valid damage and loss configuration for this asset. '
            )
            raise PelicunInvalidConfigError(msg)

        # look for possibly specified assessment options
        try:
            assessment_options = config['Applications']['DL']['ApplicationData'][
                'Options'
            ]
        except KeyError:
            assessment_options = None

        if assessment_options:
            # extend options defined via the auto-population script to
            # include those in the original `config`
            config_ap['Applications']['DL']['ApplicationData'].pop('Options')
            update_vals(
                config_ap['DL']['Options'],
                assessment_options,
                "config_ap['DL']['Options']",
                'assessment_options',
            )

        # add the demand information
        update(config_ap, '/DL/Demands/DemandFilePath', demand_file)
        update(config_ap, '/DL/Demands/SampleSize', str(realizations))

        if coupled_edp is True:
            update(config_ap, 'DL/Demands/CoupledDemands', value=True)

        else:
            update(
                config_ap,
                'DL/Demands/Calibration',
                {'ALL': {'DistributionFamily': 'lognormal'}},
            )

        # save the component data
        comp.to_csv(output_path / 'CMP_QNT.csv')

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

        with Path(config_ap_path).open('w', encoding='utf-8') as f:
            json.dump(config_ap, f, indent=2)

        update(config, 'DL', get(config_ap, 'DL'))

    # sample size
    sample_size_str = get(config, 'DL/Options/Sampling/SampleSize')
    if not sample_size_str:
        sample_size_str = get(config, 'DL/Demands/SampleSize')
        if not sample_size_str:
            msg = 'Sample size not provided in config file.'
            raise PelicunInvalidConfigError(msg)
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
        msg = 'Demand configuration missing.'
        raise PelicunInvalidConfigError(msg)

    if is_unspecified(config, 'DL/Asset'):
        msg = 'Asset configuration missing.'
        raise PelicunInvalidConfigError(msg)

    update(
        config,
        'DL/Options/LogFile',
        'pelicun_log.txt',
        only_if_empty_or_none=True,
    )
    update(
        config,
        'DL/Options/Verbose',
        value=True,
        only_if_empty_or_none=True,
    )

    # if the user did not prescribe anything for ListAllDamageStates,
    # then use True as default for DL_calculations regardless of what
    # the Pelicun default is.
    update(
        config,
        'DL/Options/ListAllDamageStates',
        value=True,
        only_if_empty_or_none=True,
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
        msg = 'No method is specified in residual drift inference configuration.'
        raise PelicunInvalidConfigError(msg)

    # Ensure `DL/Damage/CollapseFragility` contains all required keys.
    if is_specified(config, 'DL/Damage/CollapseFragility'):
        for thing in ('CapacityDistribution', 'CapacityMedian', 'Theta_1'):
            if is_unspecified(config, f'DL/Damage/CollapseFragility/{thing}'):
                msg = (
                    f'`{thing}` is missing from DL/Damage/CollapseFragility'
                    f' in the configuration file.'
                )
                raise PelicunInvalidConfigError(msg)

    # Ensure `DL/Damage/IrreparableDamage` contains all required keys.
    if is_specified(config, 'DL/Damage/IrreparableDamage'):
        for thing in ('DriftCapacityMedian', 'DriftCapacityLogStd'):
            if is_unspecified(config, f'DL/Damage/IrreparableDamage/{thing}'):
                msg = (
                    f'`{thing}` is missing from DL/Damage/IrreparableDamage'
                    f' in the configuration file.'
                )
                raise PelicunInvalidConfigError(msg)

    # If the damage process approach is `User Defined` there needs to
    # be a damage process file path.
    if get(config, 'DL/Damage/DamageProcess') == 'User Defined' and is_unspecified(
        config, 'DL/Damage/DamageProcessFilePath'
    ):
        msg = (
            'When `DL/Damage/DamageProcess` is set to `User Defined`, '
            'a path needs to be specified under '
            '`DL/Damage/DamageProcessFilePath`.'
        )
        raise PelicunInvalidConfigError(msg)

    # Getting results requires running the calculations.
    if is_specified(config, 'DL/Outputs/Asset') and is_unspecified(
        config, 'DL/Asset'
    ):
        msg = (
            'No asset data specified in config file. '
            'Cannot generate asset model outputs.'
        )
        raise PelicunInvalidConfigError(msg)

    if is_specified(config, 'DL/Outputs/Damage') and is_unspecified(
        config, 'DL/Damage'
    ):
        msg = (
            'No damage data specified in config file. '
            'Cannot generate damage model outputs.'
        )
        raise PelicunInvalidConfigError(msg)

    if is_specified(config, 'DL/Outputs/Loss') and is_unspecified(
        config, 'DL/Losses'
    ):
        msg = (
            'No loss data specified in config file. '
            'Cannot generate loss model outputs.'
        )
        raise PelicunInvalidConfigError(msg)

    # Ensure only one of `component_assignment_file` or
    # `component_sample_file` is provided.
    if is_specified(config, 'DL/Asset'):
        if (
            (get(config, 'DL/Asset/ComponentAssignmentFile') is None)
            and (get(config, 'DL/Asset/ComponentSampleFile') is None)
        ) or (
            (get(config, 'DL/Asset/ComponentAssignmentFile') is not None)
            and (get(config, 'DL/Asset/ComponentSampleFile') is not None)
        ):
            msg = (
                'In the asset model configuration, it is '
                'required to specify one of `component_assignment_file` '
                'or `component_sample_file`, but not both.'
            )
            raise PelicunInvalidConfigError(msg)

    return config


def _create_json_files_if_requested(
    config: dict, out_files: list[str], output_path: Path
) -> None:
    """
    Create JSON files if requested in the config.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    out_files : list
        List of output file names.
    output_path : Path
        Path to the output directory.

    """
    # If not requested, simply return
    if get(config, 'DL/Outputs/Format/JSON', default=False) is False:
        return

    for filename in out_files:
        filename_json = filename[:-3] + 'json'

        if (
            get(config, 'DL/Outputs/Settings/SimpleIndexInJSON', default=False)
            is True
        ):
            data = pd.read_csv(output_path / filename, index_col=0)
        else:
            data = convert_to_MultiIndex(
                pd.read_csv(output_path / filename, index_col=0), axis=1
            )

        if 'Units' in data.index:
            df_units = convert_to_SimpleIndex(
                data.loc['Units', :].to_frame().T,  # type: ignore
                axis=1,
            )

            data = data.drop('Units', axis=0)

            out_dict = convert_df_to_dict(data)

            out_dict.update(
                {
                    'Units': {
                        col: df_units.loc['Units', col] for col in df_units.columns
                    }
                }
            )

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

        elif "Water" not in config['DL']['Asset']['ComponentDatabase']:
            # add a placeholder collapse fragility that will never trigger
            # collapse, but allow damage processes to work with collapse

            adf.loc['collapse', ('Demand', 'Directional')] = 1
            adf.loc['collapse', ('Demand', 'Offset')] = 0
            adf.loc['collapse', ('Demand', 'Type')] = 'One'
            adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
            adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
            adf.loc['collapse', 'Incomplete'] = 0

        if 'IrreparableDamage' in config['DL']['Damage'].keys():
            irrep_config = config['DL']['Damage']['IrreparableDamage']

            # add excessive RID fragility according to settings provided in the
            # input file
            adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
            adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
            adf.loc['excessiveRID', ('Demand', 'Type')] = (
                'Residual Interstory Drift Ratio'
            )

            adf.loc['excessiveRID', ('Demand', 'Unit')] = 'unitless'
            adf.loc['excessiveRID', ('LS1', 'Theta_0')] = irrep_config[
                'DriftCapacityMedian'
            ]

            adf.loc['excessiveRID', ('LS1', 'Family')] = "lognormal"

            adf.loc['excessiveRID', ('LS1', 'Theta_1')] = irrep_config[
                'DriftCapacityLogStd'
            ]

            adf.loc['excessiveRID', 'Incomplete'] = 0

            # add a placeholder irreparable fragility that will never trigger
            # damage, but allow damage processes to aggregate excessiveRID here
            adf.loc['irreparable', ('Demand', 'Directional')] = 1
            adf.loc['irreparable', ('Demand', 'Offset')] = 0
            adf.loc['irreparable', ('Demand', 'Type')] = 'One'
            adf.loc['irreparable', ('Demand', 'Unit')] = 'unitless'
            adf.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
            adf.loc['irreparable', 'Incomplete'] = 0

        # TODO: we can improve this by creating a water
        # network-specific assessment class
        if "Water" in config['DL']['Asset']['ComponentDatabase']:
            # add a placeholder aggregate fragility that will never trigger
            # damage, but allow damage processes to aggregate the
            # various pipeline damages
            adf.loc['aggregate', ('Demand', 'Directional')] = 1
            adf.loc['aggregate', ('Demand', 'Offset')] = 0
            adf.loc['aggregate', ('Demand', 'Type')] = 'Peak Ground Velocity'
            adf.loc['aggregate', ('Demand', 'Unit')] = 'mps'
            adf.loc['aggregate', ('LS1', 'Theta_0')] = 1e10
            adf.loc['aggregate', ('LS2', 'Theta_0')] = 1e10
            adf.loc['aggregate', 'Incomplete'] = 0

        PAL.damage.load_damage_model(component_db + [adf])

        # load the damage process if needed
        dmg_process = None
        if config['DL']['Damage'].get('DamageProcess', False) is not False:
            dp_approach = config['DL']['Damage']['DamageProcess']

            if dp_approach in damage_processes:
                dmg_process = damage_processes[dp_approach]

                # For Hazus Earthquake, we need to specify the component ids
                if dp_approach == 'Hazus Earthquake':
                    cmp_sample = PAL.asset.save_cmp_sample()

                    cmp_list = cmp_sample.columns.unique(level=0)

                    cmp_map = {'STR': '', 'LF': '', 'NSA': ''}

                    for cmp in cmp_list:
                        for cmp_type in cmp_map:
                            if cmp_type + '.' in cmp:
                                cmp_map[cmp_type] = cmp

                    new_dmg_process = dmg_process.copy()
                    for source_cmp, action in dmg_process.items():
                        # first, look at the source component id
                        new_source = None
                        for cmp_type, cmp_id in cmp_map.items():
                            if (cmp_type in source_cmp) and (cmp_id != ''):
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
                                    if (cmp_type in target_vals) and (cmp_id != ''):
                                        target_vals = target_vals.replace(
                                            cmp_type, cmp_id
                                        )

                                new_target_vals = target_vals

                            else:
                                # we assume that target_vals is a list of str
                                new_target_vals = []

                                for target_val in target_vals:
                                    for cmp_type, cmp_id in cmp_map.items():
                                        if (cmp_type in target_val) and (
                                            cmp_id != ''
                                        ):
                                            target_val = target_val.replace(
                                                cmp_type, cmp_id
                                            )

                                    new_target_vals.append(target_val)

                            new_dmg_process[new_source][ds_i] = new_target_vals

                    dmg_process = new_dmg_process

            elif dp_approach == "User Defined":
                # load the damage process from a file
                with open(
                    config['DL']['Damage']['DamageProcessFilePath'],
                    'r',
                    encoding='utf-8',
                ) as f:
                    dmg_process = json.load(f)

            elif dp_approach == "None":
                # no damage process applied for the calculation
                dmg_process = None

            else:
                log_msg(
                    f"Prescribed Damage Process not recognized: " f"{dp_approach}"
                )

        # calculate damages
        scaling_specification = config['DL']['Damage'].get('scaling_specification', None)
        PAL.damage.calculate(sample_size, dmg_process=dmg_process, scaling_specification=scaling_specification)

        # if requested, save results
        if 'Damage' in config['DL']['Outputs']:
            damage_sample, damage_units = PAL.damage.save_sample(save_units=True)
            damage_units = damage_units.to_frame().T

            if (
                config['DL']['Outputs']['Settings'].get(
                    'AggregateColocatedComponentResults', False
                )
                is True
            ):
                damage_units = damage_units.groupby(
                    level=[0, 1, 2, 4], axis=1
                ).first()

                damage_groupby_uid = damage_sample.groupby(
                    level=[0, 1, 2, 4], axis=1
                )

                damage_sample = damage_groupby_uid.sum().mask(
                    damage_groupby_uid.count() == 0, np.nan
                )

            out_reqs = [
                out if val else ""
                for out, val in config['DL']['Outputs']['Damage'].items()
            ]

            if np.any(
                np.isin(
                    ['Sample', 'Statistics', 'GroupedSample', 'GroupedStatistics'],
                    out_reqs,
                )
            ):
                if 'Sample' in out_reqs:
                    damage_sample_s = pd.concat([damage_sample, damage_units])

                    damage_sample_s = convert_to_SimpleIndex(damage_sample_s, axis=1)
                    damage_sample_s.to_csv(
                        output_path / "DMG_sample.zip",
                        index_label=damage_sample_s.columns.name,
                        compression=dict(
                            method='zip', archive_name='DMG_sample.csv'
                        ),
                    )
                    output_files.append('DMG_sample.zip')

                if 'Statistics' in out_reqs:
                    damage_stats = describe(damage_sample)
                    damage_stats = pd.concat([damage_stats, damage_units])

                    damage_stats = convert_to_SimpleIndex(damage_stats, axis=1)
                    damage_stats.to_csv(
                        output_path / "DMG_stats.csv",
                        index_label=damage_stats.columns.name,
                    )
                    output_files.append('DMG_stats.csv')

                if np.any(np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)):
                    if (
                        config['DL']['Outputs']['Settings'].get(
                            'AggregateColocatedComponentResults', False
                        )
                        is True
                    ):
                        damage_groupby = damage_sample.groupby(
                            level=[0, 1, 3], axis=1
                        )

                        damage_units = damage_units.groupby(
                            level=[0, 1, 3], axis=1
                        ).first()

                    else:
                        damage_groupby = damage_sample.groupby(
                            level=[0, 1, 4], axis=1
                        )

                        damage_units = damage_units.groupby(
                            level=[0, 1, 4], axis=1
                        ).first()

                    grp_damage = damage_groupby.sum().mask(
                        damage_groupby.count() == 0, np.nan
                    )

                    # if requested, condense DS output
                    if (
                        config['DL']['Outputs']['Settings'].get('CondenseDS', False)
                        is True
                    ):
                        # replace non-zero values with 1
                        grp_damage = grp_damage.mask(
                            grp_damage.astype(np.float64).values > 0, 1
                        )

                        # get the corresponding DS for each column
                        ds_list = grp_damage.columns.get_level_values(2).astype(int)

                        # replace ones with the corresponding DS in each cell
                        grp_damage = grp_damage.mul(ds_list, axis=1)

                        # aggregate across damage state indices
                        damage_groupby_2 = grp_damage.groupby(level=[0, 1], axis=1)

                        # choose the max value
                        # i.e., the governing DS for each comp-loc pair
                        grp_damage = damage_groupby_2.max().mask(
                            damage_groupby_2.count() == 0, np.nan
                        )

                        # aggregate units to the same format
                        # assume identical units across locations for each comp
                        damage_units = damage_units.groupby(
                            level=[0, 1], axis=1
                        ).first()

                    else:
                        # otherwise, aggregate damage quantities for each comp
                        damage_groupby_2 = grp_damage.groupby(level=0, axis=1)

                        # preserve NaNs
                        grp_damage = damage_groupby_2.sum().mask(
                            damage_groupby_2.count() == 0, np.nan
                        )

                        # and aggregate units to the same format
                        damage_units = damage_units.groupby(level=0, axis=1).first()

                    if 'GroupedSample' in out_reqs:
                        grp_damage_s = pd.concat([grp_damage, damage_units])

                        grp_damage_s = convert_to_SimpleIndex(grp_damage_s, axis=1)
                        grp_damage_s.to_csv(
                            output_path / "DMG_grp.zip",
                            index_label=grp_damage_s.columns.name,
                            compression=dict(
                                method='zip', archive_name='DMG_grp.csv'
                            ),
                        )
                        output_files.append('DMG_grp.zip')

                    if 'GroupedStatistics' in out_reqs:
                        grp_stats = describe(grp_damage)
                        grp_stats = pd.concat([grp_stats, damage_units])

                        grp_stats = convert_to_SimpleIndex(grp_stats, axis=1)
                        grp_stats.to_csv(
                            output_path / "DMG_grp_stats.csv",
                            index_label=grp_stats.columns.name,
                        )
                        output_files.append('DMG_grp_stats.csv')

            # - - - - -
            # This is almost surely not needed any more
            """
            if regional == True:

                damage_sample = PAL.damage.save_sample()

                # first, get the collapse probability
                df_res_c = pd.DataFrame([0,],
                    columns=pd.MultiIndex.from_tuples([('probability',' '),]),
                    index=[0, ])

                if ("collapse", 0, 1, 1) in damage_sample.columns:
                    df_res_c['probability'] = (
                        damage_sample[("collapse", 0, 1, 1)].mean())

                else:
                    df_res_c['probability'] = 0.0

                df_res = pd.concat([df_res_c,], axis=1, keys=['collapse',])

                df_res.to_csv(output_path/'DM.csv')
                output_files.append('DM.csv')
            """
            # - - - - -

    # Loss Assessment -----------------------------------------------------------

    # if a loss assessment is requested
    if 'Losses' in config['DL']:
        out_config_loss = config['DL']['Outputs'].get('Loss', {})

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # backwards-compatibility for v3.2 and earlier | remove after v4.0
        if config['DL']['Losses'].get('BldgRepair', False):
            config['DL']['Losses']['Repair'] = config['DL']['Losses']['BldgRepair']

        if out_config_loss.get('BldgRepair', False):
            out_config_loss['Repair'] = out_config_loss['BldgRepair']
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # if requested, calculate repair consequences
        if config['DL']['Losses'].get('Repair', False):
            repair_config = config['DL']['Losses']['Repair']

            # load the fragility information
            if repair_config['ConsequenceDatabase'] in default_DBs['repair'].keys():
                consequence_db = [
                    'PelicunDefault/'
                    + default_DBs['repair'][repair_config['ConsequenceDatabase']],
                ]

                conseq_df = PAL.get_default_data(
                    default_DBs['repair'][repair_config['ConsequenceDatabase']][:-4]
                )
            else:
                consequence_db = []

                conseq_df = pd.DataFrame()

            if repair_config.get('ConsequenceDatabasePath', False) is not False:
                extra_comps = repair_config['ConsequenceDatabasePath']

                if 'CustomDLDataFolder' in extra_comps:
                    extra_comps = extra_comps.replace(
                        'CustomDLDataFolder', custom_dl_file_path
                    )

                consequence_db += [
                    extra_comps,
                ]

                extra_conseq_df = load_data(
                    extra_comps,
                    unit_conversion_factors=None,
                    orientation=1,
                    reindex=False,
                )

                if isinstance(conseq_df, pd.DataFrame):
                    conseq_df = pd.concat([conseq_df, extra_conseq_df])
                else:
                    conseq_df = extra_conseq_df

            consequence_db = consequence_db[::-1]

            # remove duplicates from conseq_df
            conseq_df = conseq_df.loc[conseq_df.index.unique(), :]

            # add the replacement consequence to the data
            adf = pd.DataFrame(
                columns=conseq_df.columns,
                index=pd.MultiIndex.from_tuples(
                    [
                        ('replacement', 'Cost'),
                        ('replacement', 'Time'),
                        ('replacement', 'Carbon'),
                        ('replacement', 'Energy'),
                    ]
                ),
            )

            # DL_method = repair_config['ConsequenceDatabase']
            DL_method = config['DL']['Damage'].get('DamageProcess', 'User Defined')

            rc = ('replacement', 'Cost')
            if 'ReplacementCost' in repair_config.keys():
                rCost_config = repair_config['ReplacementCost']

                adf.loc[rc, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[rc, ('DV', 'Unit')] = rCost_config["Unit"]

                adf.loc[rc, ('DS1', 'Theta_0')] = rCost_config["Median"]

                if pd.isna(rCost_config.get('Distribution', np.nan)) is False:
                    adf.loc[rc, ('DS1', 'Family')] = rCost_config["Distribution"]
                    adf.loc[rc, ('DS1', 'Theta_1')] = rCost_config["Theta_1"]

            else:
                # add a default replacement cost value as a placeholder
                # the default value depends on the consequence database

                # for FEMA P-58, use 0 USD
                if DL_method == 'FEMA P-58':
                    adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rc, ('DV', 'Unit')] = 'USD_2011'
                    adf.loc[rc, ('DS1', 'Theta_0')] = 0

                # for Hazus EQ and HU, use 1.0 as a loss_ratio
                elif DL_method in ['Hazus Earthquake', 'Hazus Hurricane']:
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
            if 'ReplacementTime' in repair_config.keys():
                rTime_config = repair_config['ReplacementTime']
                rt = ('replacement', 'Time')

                adf.loc[rt, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[rt, ('DV', 'Unit')] = rTime_config["Unit"]

                adf.loc[rt, ('DS1', 'Theta_0')] = rTime_config["Median"]

                if pd.isna(rTime_config.get('Distribution', np.nan)) is False:
                    adf.loc[rt, ('DS1', 'Family')] = rTime_config["Distribution"]
                    adf.loc[rt, ('DS1', 'Theta_1')] = rTime_config["Theta_1"]
            else:
                # add a default replacement time value as a placeholder
                # the default value depends on the consequence database

                # for FEMA P-58, use 0 worker_days
                if DL_method == 'FEMA P-58':
                    adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rt, ('DV', 'Unit')] = 'worker_day'
                    adf.loc[rt, ('DS1', 'Theta_0')] = 0

                # for Hazus EQ, use 1.0 as a loss_ratio
                elif DL_method == 'Hazus Earthquake - Buildings':
                    adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rt, ('DV', 'Unit')] = 'day'

                    # load the replacement time that corresponds to total loss
                    occ_type = config['DL']['Asset']['OccupancyType']
                    adf.loc[rt, ('DS1', 'Theta_0')] = conseq_df.loc[
                        (f"STR.{occ_type}", 'Time'), ('DS5', 'Theta_0')
                    ]

                # otherwise, use 1 (and expect to have it defined by the user)
                else:
                    adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rt, ('DV', 'Unit')] = 'loss_ratio'
                    adf.loc[rt, ('DS1', 'Theta_0')] = 1

            rcarb = ('replacement', 'Carbon')
            if 'ReplacementCarbon' in repair_config.keys():
                rCarbon_config = repair_config['ReplacementCarbon']
                rcarb = ('replacement', 'Carbon')

                adf.loc[rcarb, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[rcarb, ('DV', 'Unit')] = rCarbon_config["Unit"]

                adf.loc[rcarb, ('DS1', 'Theta_0')] = rCarbon_config["Median"]

                if pd.isna(rCarbon_config.get('Distribution', np.nan)) is False:
                    adf.loc[rcarb, ('DS1', 'Family')] = rCarbon_config[
                        "Distribution"
                    ]
                    adf.loc[rcarb, ('DS1', 'Theta_1')] = rCarbon_config["Theta_1"]
            else:
                # add a default replacement carbon value as a placeholder
                # the default value depends on the consequence database

                # for FEMA P-58, use 0 kg
                if DL_method == 'FEMA P-58':
                    adf.loc[rcarb, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[rcarb, ('DV', 'Unit')] = 'kg'
                    adf.loc[rcarb, ('DS1', 'Theta_0')] = 0

                else:
                    # for everything else, remove this consequence
                    adf.drop(rcarb, inplace=True)

            ren = ('replacement', 'Energy')
            if 'ReplacementEnergy' in repair_config.keys():
                rEnergy_config = repair_config['ReplacementEnergy']
                ren = ('replacement', 'Energy')

                adf.loc[ren, ('Quantity', 'Unit')] = "1 EA"

                adf.loc[ren, ('DV', 'Unit')] = rEnergy_config["Unit"]

                adf.loc[ren, ('DS1', 'Theta_0')] = rEnergy_config["Median"]

                if pd.isna(rEnergy_config.get('Distribution', np.nan)) is False:
                    adf.loc[ren, ('DS1', 'Family')] = rEnergy_config["Distribution"]
                    adf.loc[ren, ('DS1', 'Theta_1')] = rEnergy_config["Theta_1"]
            else:
                # add a default replacement energy value as a placeholder
                # the default value depends on the consequence database

                # for FEMA P-58, use 0 kg
                if DL_method == 'FEMA P-58':
                    adf.loc[ren, ('Quantity', 'Unit')] = '1 EA'
                    adf.loc[ren, ('DV', 'Unit')] = 'MJ'
                    adf.loc[ren, ('DS1', 'Theta_0')] = 0

                else:
                    # for everything else, remove this consequence
                    adf.drop(ren, inplace=True)

            # prepare the loss map
            loss_map = None
            if repair_config['MapApproach'] == "Automatic":
                # get the damage sample
                dmg_sample = PAL.damage.save_sample()

                # create a mapping for all components that are also in
                # the prescribed consequence database
                dmg_cmps = dmg_sample.columns.unique(level='cmp')
                loss_cmps = conseq_df.index.unique(level=0)

                drivers = []
                loss_models = []

                if DL_method in ['FEMA P-58', 'Hazus Hurricane']:
                    # with these methods, we assume fragility and consequence data
                    # have the same IDs

                    for dmg_cmp in dmg_cmps:
                        if dmg_cmp == 'collapse':
                            continue

                        if dmg_cmp in loss_cmps:
                            drivers.append(f'DMG-{dmg_cmp}')
                            loss_models.append(dmg_cmp)

                elif DL_method in [
                    'Hazus Earthquake',
                    'Hazus Earthquake Transportation',
                ]:
                    # with Hazus Earthquake we assume that consequence
                    # archetypes are only differentiated by occupancy type
                    occ_type = config['DL']['Asset'].get('OccupancyType', None)

                    for dmg_cmp in dmg_cmps:
                        if dmg_cmp == 'collapse':
                            continue

                        cmp_class = dmg_cmp.split('.')[0]
                        if occ_type is not None:
                            loss_cmp = f'{cmp_class}.{occ_type}'
                        else:
                            loss_cmp = cmp_class

                        if loss_cmp in loss_cmps:
                            drivers.append(f'DMG-{dmg_cmp}')
                            loss_models.append(loss_cmp)

                loss_map = pd.DataFrame(
                    loss_models, columns=['Repair'], index=drivers
                )

            elif repair_config['MapApproach'] == "User Defined":
                if repair_config.get('MapFilePath', False) is not False:
                    loss_map_path = repair_config['MapFilePath']

                    loss_map_path = loss_map_path.replace(
                        'CustomDLDataFolder', custom_dl_file_path
                    )

                else:
                    print("User defined loss map path missing. Terminating analysis")
                    return -1

                loss_map = pd.read_csv(loss_map_path, index_col=0)

            # prepare additional loss map entries, if needed
            if 'DMG-collapse' not in loss_map.index:
                loss_map.loc['DMG-collapse', 'Repair'] = 'replacement'
                loss_map.loc['DMG-irreparable', 'Repair'] = 'replacement'

            # assemble the list of requested decision variables
            DV_list = []
            if repair_config.get('DecisionVariables', False) is not False:
                for DV_i, DV_status in repair_config['DecisionVariables'].items():
                    if DV_status is True:
                        DV_list.append(DV_i)

            else:
                DV_list = None

            PAL.repair.load_model(
                consequence_db
                + [
                    adf,
                ],
                loss_map,
                decision_variables=DV_list,
            )

            PAL.repair.calculate(sample_size)

            agg_repair = PAL.repair.aggregate_losses()

            # if requested, save results
            if out_config_loss.get('Repair', False):
                repair_sample, repair_units = PAL.repair.save_sample(save_units=True)
                repair_units = repair_units.to_frame().T

                if (
                    config['DL']['Outputs']['Settings'].get(
                        'AggregateColocatedComponentResults', False
                    )
                    is True
                ):
                    repair_units = repair_units.groupby(
                        level=[0, 1, 2, 3, 4, 5], axis=1
                    ).first()

                    repair_groupby_uid = repair_sample.groupby(
                        level=[0, 1, 2, 3, 4, 5], axis=1
                    )

                    repair_sample = repair_groupby_uid.sum().mask(
                        repair_groupby_uid.count() == 0, np.nan
                    )

                out_reqs = [
                    out if val else ""
                    for out, val in out_config_loss['Repair'].items()
                ]

                if np.any(
                    np.isin(
                        [
                            'Sample',
                            'Statistics',
                            'GroupedSample',
                            'GroupedStatistics',
                            'AggregateSample',
                            'AggregateStatistics',
                        ],
                        out_reqs,
                    )
                ):
                    if 'Sample' in out_reqs:
                        repair_sample_s = repair_sample.copy()
                        repair_sample_s = pd.concat([repair_sample_s, repair_units])

                        repair_sample_s = convert_to_SimpleIndex(
                            repair_sample_s, axis=1
                        )
                        repair_sample_s.to_csv(
                            output_path / "DV_repair_sample.zip",
                            index_label=repair_sample_s.columns.name,
                            compression=dict(
                                method='zip',
                                archive_name='DV_repair_sample.csv',
                            ),
                        )
                        output_files.append('DV_repair_sample.zip')

                    if 'Statistics' in out_reqs:
                        repair_stats = describe(repair_sample)
                        repair_stats = pd.concat([repair_stats, repair_units])

                        repair_stats = convert_to_SimpleIndex(repair_stats, axis=1)
                        repair_stats.to_csv(
                            output_path / "DV_repair_stats.csv",
                            index_label=repair_stats.columns.name,
                        )
                        output_files.append('DV_repair_stats.csv')

                    if np.any(
                        np.isin(['GroupedSample', 'GroupedStatistics'], out_reqs)
                    ):
                        repair_groupby = repair_sample.groupby(
                            level=[0, 1, 2], axis=1
                        )

                        repair_units = repair_units.groupby(
                            level=[0, 1, 2], axis=1
                        ).first()

                        grp_repair = repair_groupby.sum().mask(
                            repair_groupby.count() == 0, np.nan
                        )

                        if 'GroupedSample' in out_reqs:
                            grp_repair_s = pd.concat([grp_repair, repair_units])

                            grp_repair_s = convert_to_SimpleIndex(
                                grp_repair_s, axis=1
                            )
                            grp_repair_s.to_csv(
                                output_path / "DV_repair_grp.zip",
                                index_label=grp_repair_s.columns.name,
                                compression=dict(
                                    method='zip',
                                    archive_name='DV_repair_grp.csv',
                                ),
                            )
                            output_files.append('DV_repair_grp.zip')

                        if 'GroupedStatistics' in out_reqs:
                            grp_stats = describe(grp_repair)
                            grp_stats = pd.concat([grp_stats, repair_units])

                            grp_stats = convert_to_SimpleIndex(grp_stats, axis=1)
                            grp_stats.to_csv(
                                output_path / "DV_repair_grp_stats.csv",
                                index_label=grp_stats.columns.name,
                            )
                            output_files.append('DV_repair_grp_stats.csv')

                    if np.any(
                        np.isin(['AggregateSample', 'AggregateStatistics'], out_reqs)
                    ):
                        if 'AggregateSample' in out_reqs:
                            agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)
                            agg_repair_s.to_csv(
                                output_path / "DV_repair_agg.zip",
                                index_label=agg_repair_s.columns.name,
                                compression=dict(
                                    method='zip',
                                    archive_name='DV_repair_agg.csv',
                                ),
                            )
                            output_files.append('DV_repair_agg.zip')

                        if 'AggregateStatistics' in out_reqs:
                            agg_stats = convert_to_SimpleIndex(
                                describe(agg_repair), axis=1
                            )
                            agg_stats.to_csv(
                                output_path / "DV_repair_agg_stats.csv",
                                index_label=agg_stats.columns.name,
                            )
                            output_files.append('DV_repair_agg_stats.csv')

    # Result Summary -----------------------------------------------------------

    if 'damage_sample' not in locals():
        damage_sample = PAL.damage.save_sample()

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

    if agg_repair is not None:
        agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)

    else:
        agg_repair_s = pd.DataFrame()

    summary = pd.concat(
        [agg_repair_s, damage_sample_s[['collapse', 'irreparable']]], axis=1
    )

    summary_stats = describe(summary)

    return summary, summary_stats


def _parse_requested_output_file_names(output_config: dict) -> set[str]:
    """
    Parse the output file names from the output configuration.

    Parameters
    ----------
    output_config : dict
        Configuration for output files.

    Returns
    -------
    set
        Set of requested output file names.

    """
    out_reqs = []
    for out, val in output_config.items():
        if val is True:
            out_reqs.append(out)
    return set(out_reqs)


def _demand_save(
    output_config: dict,
    assessment: DLCalculationAssessment,
    output_path: Path,
    out_files: list[str],
) -> None:
    """
    Save demand results to files based on the output config.

    Parameters
    ----------
    output_config : dict
        Configuration for output files.
    assessment : AssessmentBase
        The assessment object.
    output_path : Path
        Path to the output directory.
    out_files : list
        List of output file names.

    """
    out_reqs = _parse_requested_output_file_names(output_config)

    demand_sample, demand_units_series = assessment.demand.save_sample(
        save_units=True
    )
    assert isinstance(demand_sample, pd.DataFrame)
    assert isinstance(demand_units_series, pd.Series)
    demand_units = demand_units_series.to_frame().T

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
    output_config: dict,
    assessment: DLCalculationAssessment,
    output_path: Path,
    out_files: list[str],
    *,
    aggregate_colocated: bool = False,
) -> None:
    """
    Save asset results to files based on the output config.

    Parameters
    ----------
    output_config : dict
        Configuration for output files.
    assessment : AssessmentBase
        The assessment object.
    output_path : Path
        Path to the output directory.
    out_files : list
        List of output file names.
    aggregate_colocated : bool, optional
        Whether to aggregate colocated components. Default is False.

    """
    output = assessment.asset.save_cmp_sample(save_units=True)
    assert isinstance(output, tuple)
    cmp_sample, cmp_units_series = output
    cmp_units = cmp_units_series.to_frame().T

    if aggregate_colocated:
        cmp_units = cmp_units.groupby(level=['cmp', 'loc', 'dir'], axis=1).first()  # type: ignore
        cmp_groupby_uid = cmp_sample.groupby(level=['cmp', 'loc', 'dir'], axis=1)  # type: ignore
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
    output_config: dict,
    assessment: DLCalculationAssessment,
    output_path: Path,
    out_files: list[str],
    *,
    aggregate_colocated: bool = False,
    condense_ds: bool = False,
) -> None:
    """
    Save damage results to files based on the output config.

    Parameters
    ----------
    output_config : dict
        Configuration for output files.
    assessment : AssessmentBase
        The assessment object.
    output_path : Path
        Path to the output directory.
    out_files : list
        List of output file names.
    aggregate_colocated : bool, optional
        Whether to aggregate colocated components. Default is False.
    condense_ds : bool, optional
        Whether to condense damage states. Default is False.

    """
    output = assessment.damage.save_sample(save_units=True)
    assert isinstance(output, tuple)
    damage_sample, damage_units_series = output
    damage_units = damage_units_series.to_frame().T

    if aggregate_colocated:
        damage_units = damage_units.groupby(  # type: ignore
            level=['cmp', 'loc', 'dir', 'ds'], axis=1
        ).first()
        damage_groupby_uid = damage_sample.groupby(  # type: ignore
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
        damage_groupby = damage_sample.groupby(level=['cmp', 'loc', 'ds'], axis=1)  # type: ignore
        damage_units = damage_units.groupby(
            level=['cmp', 'loc', 'ds'], axis=1
        ).first()  # type: ignore

        grp_damage = damage_groupby.sum().mask(damage_groupby.count() == 0, np.nan)

        # if requested, condense DS output
        if condense_ds:
            # replace non-zero values with 1
            grp_damage = grp_damage.mask(
                grp_damage.astype(np.float64).to_numpy() > 0, 1
            )

            # get the corresponding DS for each column
            ds_list = grp_damage.columns.get_level_values('ds').astype(int)

            # replace ones with the corresponding DS in each cell
            grp_damage = grp_damage.mul(ds_list, axis=1)

            # aggregate across damage state indices
            damage_groupby_2 = grp_damage.groupby(level=['cmp', 'loc'], axis=1)

            # choose the max value
            # i.e., the governing DS for each comp-loc pair
            grp_damage = damage_groupby_2.max().mask(
                damage_groupby_2.count() == 0, np.nan
            )

            # aggregate units to the same format
            # assume identical units across locations for each comp
            damage_units = damage_units.groupby(level=['cmp', 'loc'], axis=1).first()  # type: ignore

        else:
            # otherwise, aggregate damage quantities for each comp
            damage_groupby_2 = grp_damage.groupby(level='cmp', axis=1)

            # preserve NaNs
            grp_damage = damage_groupby_2.sum().mask(
                damage_groupby_2.count() == 0, np.nan
            )

            # and aggregate units to the same format
            damage_units = damage_units.groupby(level='cmp', axis=1).first()  # type: ignore

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
    output_config: dict,
    assessment: DLCalculationAssessment,
    output_path: Path,
    out_files: list[str],
    agg_repair: pd.DataFrame,
    *,
    aggregate_colocated: bool = False,
) -> None:
    """
    Save loss results to files based on the output config.

    Parameters
    ----------
    output_config : dict
        Configuration for output files.
    assessment : AssessmentBase
        The assessment object.
    output_path : Path
        Path to the output directory.
    out_files : list
        List of output file names.
    agg_repair : pd.DataFrame
        Aggregate repair data.
    aggregate_colocated : bool, optional
        Whether to aggregate colocated components. Default is False.

    """
    out = assessment.loss.ds_model.save_sample(save_units=True)
    assert isinstance(out, tuple)
    repair_sample, repair_units_series = out
    repair_units = repair_units_series.to_frame().T

    if aggregate_colocated:
        repair_units = repair_units.groupby(  # type: ignore
            level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir'], axis=1
        ).first()
        repair_groupby_uid = repair_sample.groupby(  # type: ignore
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
        repair_groupby = repair_sample.groupby(level=['dv', 'loss', 'dmg'], axis=1)  # type: ignore
        repair_units = repair_units.groupby(  # type: ignore
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


def _remove_existing_files(output_path: Path, known_output_files: list[str]) -> None:
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
                (output_path / filename).unlink()
            except OSError as exc:
                msg = (
                    f'Error occurred while removing '
                    f'`{output_path / filename}`: {exc}'
                )
                raise OSError(msg) from exc


def main() -> None:
    """Parse arguments and run the pelicun calculation."""
    args_list = sys.argv[1:]

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
    # TODO(JVM): fix color warnings
    # parser.add_argument(
    #     '--color_warnings',
    #     default=False,
    #     type=str2bool,
    #     nargs='?',
    #     const=False,
    #     help=(
    #         'Enable colored warnings in the console '
    #         'output (True/False). Defaults to False.'
    #     ),
    # )
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

    if not args_list:
        parser.print_help()
        return

    args = parser.parse_args(args_list)

    log_msg('Initializing pelicun calculation.')

    run_pelicun(
        config_path=args.filenameDL,
        demand_file=args.demandFile,
        output_path=args.dirnameOutput,
        realizations=args.Realizations,
        auto_script_path=args.auto_script,
        custom_model_dir=args.custom_model_dir,
        output_format=args.output_format,
        detailed_results=args.detailed_results,
        coupled_edp=args.coupled_EDP,
    )

    log_msg('pelicun calculation completed.')


if __name__ == '__main__':
    main()
