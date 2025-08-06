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
from pelicun.file_io import substitute_default_path
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
            'GroupedStatistics': True,
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


def run_pelicun(  # noqa: C901
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
        custom_model_dir,
        coupled_edp=coupled_edp,
        detailed_results=detailed_results,
    )

    # An undefined config means that we do not need to run a simulation
    # Such config is not an error, those are caught during parsing. This is the
    # result of an intentional no-simulation request during auto-population.
    if config is None:
        return

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
            is_for_water_network_assessment='Water'
            in get(config, 'DL/Asset/ComponentDatabase', ''),
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
            replacement_configuration=None,  # will be used later
            loss_combination_method=get(
                config, 'DL/Losses/Repair/CombinationMethod'
            ),
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


def _parse_config_file(  # noqa: C901, PLR0912
    config_path: Path,
    output_path: Path,
    auto_script_path: Path | None,
    demand_file: str,
    realizations: int,
    output_format: list | None,
    custom_model_dir: str | None,
    *,
    coupled_edp: bool,
    detailed_results: bool,
) -> dict[str, object] | None:
    """
    Parse and validate the config file for Pelicun.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    output_path : Path
        Directory for output files.
    auto_script_path : Path
        Path to the auto-generation script.
    demand_file : str
        Path to the demand data file.
    realizations : int
        Number of realizations.
    output_format : str
        Output format (CSV, JSON).
    custom_model_dir: str, optional
        String pointing to a directory with files that define user-provided model
        parameters for a customized damage and loss assessment.
    coupled_EDP : bool
        Whether to consider coupled EDPs.
    detailed_results : bool
        Whether to generate detailed results.

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

    # add the demand file to the DL if needed
    if is_specified(config, 'DL'):
        if is_unspecified(config, 'DL/Demands/DemandFilePath'):
            update(config, '/DL/Demands/DemandFilePath', demand_file)

    # Validate the configuration against the schema
    try:
        validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as exc:
        msg = 'The provided config file does not conform to the schema.'
        raise PelicunInvalidConfigError(msg) from exc

    # identify the folder with the damage and loss model data
    if is_unspecified(config, 'DL'):
        dl_method = get(config, 'Applications/DL/ApplicationData/DL_Method')

        if dl_method == 'User-provided Models':
            if custom_model_dir is not None:
                dl_model_folder = custom_model_dir
            else:
                dl_model_folder = get(
                    config, 'Applications/DL/ApplicationData/custom_model_dir'
                )

            assert isinstance(dl_model_folder, str)
            dl_model_folder = Path(dl_model_folder).resolve()
            assert dl_model_folder.exists(), f'{dl_model_folder} does not exist'
            assert dl_model_folder.is_dir(), f'{dl_model_folder} is not a directory'

            auto_script_paths = [
                Path(dl_model_folder / 'pelicun_config.py').resolve()
            ]

        else:
            dl_methods = [m.strip() for m in dl_method.split(',')]

            auto_script_paths = []
            for dl_method in dl_methods:
                auto_script_path = substitute_default_path(
                    [f'PelicunDefault/{dl_method}/pelicun_config.py']
                )[0]
                auto_script_paths.append(Path(auto_script_path).resolve())

        for auto_script_path in auto_script_paths:
            if not auto_script_path.exists():
                msg = (
                    f'No `DL` entry in config file and the following path '
                    f'does not point to a valid pelicun configuration file: '
                    f'{auto_script_path}.'
                )
                raise PelicunInvalidConfigError(msg)

        # Add the demandFile to the config dict to allow demand-dependent auto-population
        update(config, '/DL/Demands/DemandFilePath', demand_file)
        update(config, '/DL/Demands/SampleSize', str(realizations))

        for script_id, auto_script_path in enumerate(auto_script_paths):
            log_msg(f'Configuring Pelicun using {auto_script_path}')

            if script_id == 0:
                config_ap, comp = auto_populate(config, auto_script_path, script_id)

            else:
                config_ap_i, comp_i = auto_populate(
                    config, auto_script_path, script_id
                )

                comp = pd.concat([comp, comp_i])

                # TODO(AZS): Currently, this is set up to work with the old flood rules
                # Requires updating once the inference is moved to BRAILS and the flood
                # config is updated.
                # Requires further updating to make it more generic and support more than
                # just hurricane wind & surge
                update(
                    config_ap,
                    'DL/Asset/ComponentDatabase',
                    (
                        f"{get(config_ap, 'DL/Asset/ComponentDatabase')},"
                        f"{get(config_ap_i, 'DL/Asset/ComponentDatabase')}"
                    ),
                )

                update(
                    config_ap,
                    'DL/Losses/Repair/ConsequenceDatabase',
                    (
                        f"{get(config_ap, 'DL/Losses/Repair/ConsequenceDatabase')},"
                        f"{get(config_ap_i, 'DL/Losses/Repair/ConsequenceDatabase')}"
                    ),
                )

                update(
                    config_ap,
                    'DL/Losses/Repair/CombinationMethod',
                    'Hazus Hurricane',
                )

        if is_unspecified(config_ap, 'DL'):
            msg = (
                'No `DL` entry in config file, and '
                'the prescribed auto-population script failed to identify '
                'a valid damage and loss configuration for this asset. '
            )
            raise PelicunInvalidConfigError(msg)

        if get(config_ap, 'DL') == 'N/A':
            msg = (
                'N/A `DL` entry in config file interpreted as a request to '
                'skip damage and loss simulation for this asset.'
            )
            log_msg(msg)
            return None

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

        # if no loss simulation is requested, remove the corresponding outputs
        if is_unspecified(config_ap, 'DL/Losses'):
            update(config_ap, 'DL/Outputs/Loss', {})

        if is_specified(config, 'outputs'):
            if (not config['outputs']['IM']) and (not config['outputs']['EDP']):
                update(config_ap, 'DL/Outputs/Demand', {})
            if not config['outputs']['DM']:
                update(config_ap, 'DL/Outputs/Damage', {})
            if not config['outputs']['DV']:
                update(config_ap, 'DL/Outputs/Loss', {})

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
        if is_unspecified(
            config, 'DL/Damage/CollapseFragility/CapacityDistribution'
        ):
            config['DL']['Damage']['CollapseFragility']['CapacityDistribution'] = (
                'deterministic'
            )
            config['DL']['Damage']['CollapseFragility']['Theta_1'] = 'N/A'

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

        # Check for units information (case-insensitive)
        units_key = None
        for key in data.index:
            if str(key).lower() == 'units':
                units_key = key
                break

        if units_key is not None:
            df_units = convert_to_SimpleIndex(
                data.loc[units_key, :].to_frame().T,  # type: ignore
                axis=1,
            )

            data = data.drop(units_key, axis=0)

            out_dict = convert_df_to_dict(data)

            out_dict.update(
                {
                    'Units': {
                        col: df_units.loc[units_key, col] for col in df_units.columns
                    }
                }
            )

        else:
            out_dict = convert_df_to_dict(data)

        with Path(output_path / filename_json).open('w', encoding='utf-8') as f:
            json.dump(out_dict, f, indent=2)


def _result_summary(
    assessment: DLCalculationAssessment, agg_repair: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a summary of the results.

    Parameters
    ----------
    assessment : AssessmentBase
        The assessment object.
    agg_repair : pd.DataFrame
        Aggregate repair data.

    Returns
    -------
    tuple
        Summary DataFrame and summary statistics DataFrame.

    """
    damage_sample = assessment.damage.save_sample()
    if damage_sample is None and agg_repair is None:
        return pd.DataFrame(), pd.DataFrame()

    if damage_sample is not None:
        assert isinstance(damage_sample, pd.DataFrame)
        damage_sample = damage_sample.groupby(level=['cmp', 'ds'], axis=1).sum()  # type: ignore
        assert isinstance(damage_sample, pd.DataFrame)
        damage_sample_s = convert_to_SimpleIndex(damage_sample, axis=1)

        if 'collapse-1' in damage_sample_s.columns:
            damage_sample_s['collapse'] = damage_sample_s['collapse-1']
        else:
            damage_sample_s['collapse'] = np.zeros(damage_sample_s.shape[0])

        if 'irreparable-1' in damage_sample_s.columns:
            damage_sample_s['irreparable'] = damage_sample_s['irreparable-1']
        else:
            damage_sample_s['irreparable'] = np.zeros(damage_sample_s.shape[0])

        damage_sample_s = damage_sample_s[['collapse', 'irreparable']]

    else:
        damage_sample_s = pd.DataFrame()

    if agg_repair is not None:
        agg_repair_s = convert_to_SimpleIndex(agg_repair, axis=1)

    else:
        agg_repair_s = pd.DataFrame()

    summary = pd.concat([agg_repair_s, damage_sample_s], axis=1)

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


def _loss_save(  # noqa: C901
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
    out = assessment.loss.save_sample(save_units=True)
    assert isinstance(out, tuple)
    repair_sample, repair_units_series = out
    repair_units = repair_units_series.to_frame().T

    if aggregate_colocated:
        if 'ds' in repair_units.columns.names:
            repair_units = repair_units.groupby(  # type: ignore
                level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir'], axis=1
            ).first()
            repair_groupby_uid = repair_sample.groupby(  # type: ignore
                level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir'], axis=1
            )
        else:
            repair_units = repair_units.groupby(  # type: ignore
                level=['dv', 'loss', 'dmg', 'loc', 'dir'], axis=1
            ).first()
            repair_groupby_uid = repair_sample.groupby(  # type: ignore
                level=['dv', 'loss', 'dmg', 'loc', 'dir'], axis=1
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
