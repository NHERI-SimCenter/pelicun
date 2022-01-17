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
# Pouria Kourehpaz
# Kuanshi Zhong

"""
This module has classes and methods that handle file input and output.

.. rubric:: Contents

.. autosummary::

    read_SimCenter_DL_input
    read_SimCenter_EDP_input
    read_population_distribution
    read_component_DL_data
    write_SimCenter_DL_output
    write_SimCenter_EDP_output
    write_SimCenter_DM_output
    write_SimCenter_DV_output

"""

from .base import log_msg
from . import base
from pathlib import Path

import numpy as np
import pandas as pd
import json

convert_dv_name = {
    'DV_rec_cost': 'Reconstruction Cost',
    'DV_rec_time': 'Reconstruction Time',
    'DV_injuries_0': 'Injuries lvl. 1',
    'DV_injuries_1': 'Injuries lvl. 2',
    'DV_injuries_2': 'Injuries lvl. 3',
    'DV_injuries_3': 'Injuries lvl. 4',
    'DV_red_tag': 'Red Tag ',
}

dependency_to_acronym = {
        'btw. Fragility Groups': 'FG',
        'btw. Performance Groups': 'PG',
        'btw. Floors': 'LOC',
        'btw. Directions': 'DIR',
        'btw. Component Groups': 'CSG',
        'btw. Damage States': 'DS',
        'Independent': 'IND',
        'per ATC recommendation': 'ATC',
    }

HAZUS_occ_converter = {
        'RES': 'Residential',
        'COM': 'Commercial',
        'REL': 'Commercial',
        'EDU': 'Educational',
        'IND': 'Industrial',
        'AGR': 'Industrial'
    }


# this is a convenience function for converting strings to float or None
def float_or_None(string):
    try:
        res = float(string)
        return res
    except ValueError:
        return None


def int_or_None(string):
    try:
        res = int(string)
        return res
    except ValueError:
        return None


def process_loc(string, stories):
    try:
        res = int(string)
        return [res, ]
    except ValueError:
        if "-" in string:
            s_low, s_high = string.split('-')
            s_low = process_loc(s_low, stories)
            s_high = process_loc(s_high, stories)
            return list(range(s_low[0], s_high[0]+1))
        elif string == "all":
            return list(range(1, stories+1))
        elif string == "top":
            return [stories, ]
        elif string == "roof":
            return [stories, ]
        else:
            return None


def get_required_resources(input_path, assessment_type):
    """
    List the data files required to perform an assessment.

    It extracts the information from the config file about the methods and
    functional data required for the analysis and provides a list of paths to
    the files that would be used.
    This method is helpful in an HPC context to copy the required resources to
    the local node from the shared file storage.

    Parameters
    ----------
    input_path: string
        Location of the DL input json file.
    assessment_type: {'P58', 'HAZUS_EQ', 'HAZUS_HU'}
        Identifies the default databases based on the type of assessment.

    Returns
    -------
    resources: list of strings
        A list of paths to the required resource files.
    """

    resources = {}

    AT = assessment_type

    with open(input_path, 'r') as f:
        jd = json.load(f)

    DL_input = jd['DamageAndLoss']

    loss = DL_input.get('LossModel', None)
    if loss is not None:
        inhabitants = loss.get('Inhabitants', None)
        dec_vars = loss.get('DecisionVariables', None)

        if dec_vars is not None:
            injuries = bool(dec_vars.get('Injuries', False))
    else:
        inhabitants = None
        dec_vars = None
        injuries = False

    # check if the user specified custom data sources
    path_CMP_data = DL_input.get("ComponentDataFolder", "")

    if path_CMP_data == "":
        # Use the P58 path as default
        path_CMP_data = base.pelicun_path / base.CMP_data_path[AT]

    resources.update({'component': path_CMP_data})

    # HAZUS combination of flood and wind losses
    if ((AT == 'HAZUS_HU') and (DL_input.get('Combinations', None)
                                is not None)):
        path_combination_data = \
            base.pelicun_path / base.CMP_data_path['HAZUS_MISC']
        resources.update({'combination': path_combination_data})

    # The population data is only needed if we are interested in injuries
    if inhabitants is not None:
        path_POP_data = inhabitants.get("PopulationDataFile", "")
    else:
        path_POP_data = ""

    if ((injuries) and (path_POP_data == "")):
        path_POP_data = base.pelicun_path / base.POP_data_path[AT]
        resources.update({'population': path_POP_data})

    return resources


def load_default_options():
    """
    Load the default_config.json file to set options to default values

    """

    with open(base.pelicun_path / "settings/default_config.json", 'r') as f:
        base.options.defaults = json.load(f)

    base.set_options(base.options.defaults.get('Options', None))


def merge_default_config(config):

    defaults = base.options.defaults

    if config.get('DemandAssessment', False):

        demand_def = defaults['DemandAssessment']
        demand_config = config['DemandAssessment']

        if 'Calibration' in demand_config.keys():

            calib_config = demand_config['Calibration']
            calib_def = demand_def['Calibration']

            for key, value in calib_def.items():

                if key in ['Marginals', ]:
                    continue

                if key not in calib_def:
                    calib_def.update({key: value})

            marginal_config = calib_config['Marginals']
            marginal_def = calib_def['Marginals']

            for key, value in marginal_def.items():

                if key not in marginal_config:
                    marginal_config.update({key: value})

        if 'Sampling' in demand_config.keys():

            sample_config = demand_config['Sampling']

            for key, value in demand_def['Sampling'].items():

                if key not in sample_config:
                    sample_config.update({key: value})

        if 'OutputUnits' in demand_def.keys():

            if 'OutputUnits' not in demand_config.keys():
                demand_config.update({'OutputUnits': {}})

            for key, value in demand_def['OutputUnits'].items():

                if key not in demand_config['OutputUnits']:
                    demand_config['OutputUnits'].update({key: value})

    return config


def save_to_csv(data, filepath, units=None, orientation=0):
    """
    Saves data to a CSV file following standard SimCenter schema.

    The produced CSV files have a single header line and an index column. The
    second line may start with 'units' in the index or the first column may be
    'units' to provide the units for the data in the file.

    The following data types in pelicun can be saved with this function:

    Demand Data: Each column in a table corresponds to a demand type; each
    row corresponds to a simulation/sample. The header identifies each demand
    type. The user guide section of the documentation provides more
    information about the header format. Target need to be specified in the
    second row of the DataFrame.

    Parameters
    ----------
    data: DataFrame
        The data to save
    filepath: string
        The location of the destination file
    units: Series, optional
        Provides a Series with variables and corresponding units.
    level: string, optional
        Identifies the level referenced in the units dictionary when the data
        has a MultiIndex header.
    orientation: int, {0, 1}, default: 0
        If 0, variables are organized along columns; otherwise they are along
        the rows. This is important when converting values to follow the
        prescribed units.

    """

    log_msg(f'Saving data to {filepath}...', prepend_timestamp=False)

    filepath = Path(filepath).resolve()

    if data is not None:

        # make sure we do not modify the original data
        data = data.copy()

        # convert units and add unit information, if needed
        if units is not None:

            log_msg('Converting units...', prepend_timestamp=False)

            # if the orientation is 1, we might not need to scale all columns
            if orientation == 1:
                cols_to_scale = [dt in [float, int] for dt in data.dtypes]
                cols_to_scale = data.columns[cols_to_scale]

            labels_to_keep = []

            for unit_name in units.unique():

                labels = units.loc[units == unit_name].index.values

                unit_factor = 1./base.UC[unit_name]

                active_labels = []

                if orientation == 0:
                    for label in labels:
                        if label in data.columns:
                            active_labels.append(label)

                    if len(active_labels) > 0:
                        data.loc[:, active_labels] *= unit_factor

                else:  # elif orientation == 1:
                    for label in labels:
                        if label in data.index:
                            active_labels.append(label)

                    if len(active_labels) > 0:
                        data.loc[active_labels, cols_to_scale] *= unit_factor

                labels_to_keep += active_labels

            units = units.loc[labels_to_keep].to_frame()

            if orientation == 0:
                data = pd.concat([units.T, data], axis=0)
                data.sort_index(axis=1, inplace=True)
            else:
                data = pd.concat([units, data], axis=1)
                data.sort_index(inplace=True)

            log_msg('Unit conversion successful.', prepend_timestamp=False)

        # convert MultiIndex to regular index with '-' separators
        if isinstance(data.index, pd.MultiIndex):
            data = base.convert_to_SimpleIndex(data)

        # same thing for the columns
        if isinstance(data.columns, pd.MultiIndex):
            data = base.convert_to_SimpleIndex(data, axis=1)

        if filepath.suffix == '.csv':

            # save the contents of the DataFrame into a csv

            data.to_csv(filepath)

            log_msg('Data successfully saved to file.',
                    prepend_timestamp=False)

        else:
            raise ValueError(
                f'ERROR: Unexpected file type received when trying '
                f'to save to csv: {filepath}')

    else:
        log_msg('WARNING: Data was empty, no file saved.',
                prepend_timestamp=False)


def load_from_csv(filepath, orientation=0, reindex=True, return_units=False,
                  convert=None):
    """
    Loads data from a CSV file assuming it follows standard SimCenter schema.

    CSV files are assumed to have a single header line and an index column. The
    second line may start with 'units' in the index and provide the units for
    each column in the file.

    The following data types can be loaded with this function:

    Demand Data: Each column in a table corresponds to a demand type; each
    row corresponds to a simulation/sample. The header identifies each demand
    type. The user guide section of the documentation provides more
    information about the header format. Units need to be specified in the
    second row of the file.

    Parameters
    ----------
    filepath: string
        The location of the source file
    orientation: int, {0, 1}, default: 0
        If 0, variables are organized along columns; otherwise they are along
        the rows. This is important when converting values to follow the
        prescribed units.
    reindex: bool
        If True, reindexes the table to ensure a 0-based, continuous index
    return_units: bool
        If True, returns the units as well as the data to allow for adjustments
        in unit conversion.

    Returns
    -------
    data: DataFrame
        Data loaded from the file.
    units: Series
        Labels from the data and corresponding units specified in the file. If
        no units are specified, this return value is "None". units are only
        returned if return_units is set to True.
    """

    log_msg(f'Loading data from {filepath}...')

    # check if the filepath is valid
    filepath = Path(filepath).resolve()

    if not filepath.is_file():
        raise ValueError(f"The filepath provided does not point to an "
                         f"existing file: {filepath}")

    if filepath.suffix == '.csv':

        # load the contents of the csv into a DataFrame

        data = pd.read_csv(filepath, header=0, index_col=0, low_memory=False)

        log_msg('File successfully opened.', prepend_timestamp=False)

    else:
        raise ValueError(f'ERROR: Unexpected file type received when trying '
                         f'to load from csv: {filepath}')

    # if there is information about units, perform the conversion to SI
    if (data.index[0] == 'units') or (data.columns[0] == 'units'):

        log_msg('Converting units...', prepend_timestamp=False)

        if orientation == 0:
            units = data.loc['units', :].copy().dropna()
            data.drop('units', inplace=True)
            data = data.astype(float)

        else:  # elif orientation==1:
            units = data.loc[:, 'units'].copy().dropna()
            data.drop('units', axis=1, inplace=True)

            if convert is None:
                cols_to_scale = []
                for col in data.columns:
                    try:
                        data.loc[:, col] = data.loc[:, col].astype(float)
                        cols_to_scale.append(col)
                    except ValueError:
                        pass
            else:
                cols_to_scale = convert

        unique_unit_names = units.unique()

        for unit_name in unique_unit_names:

            unit_factor = base.UC[unit_name]
            unit_labels = units.loc[units == unit_name].index

            if orientation == 0:
                data.loc[:, unit_labels] *= unit_factor

            else:  # elif orientation==1:
                data.loc[unit_labels, cols_to_scale] *= unit_factor

        log_msg('Unit conversion successful.', prepend_timestamp=False)

    else:

        data = data.convert_dtypes()
        # enforcing float datatype is important even if there is no unit
        # conversion
        units = None
        if orientation == 0:
            data = data.astype(float)

        else:
            for col in data.columns:
                try:
                    data.loc[:, col] = data.loc[:, col].astype(float)
                except ValueError:
                    pass

    # convert column to MultiIndex if needed
    data = base.convert_to_MultiIndex(data, axis=1)

    data.sort_index(axis=1, inplace=True)

    # reindex the data, if needed
    if reindex:

        data.index = np.arange(data.shape[0])

    else:
        # convert index to MultiIndex if needed
        data = base.convert_to_MultiIndex(data, axis=0)

        data.sort_index(inplace=True)

    log_msg('Data successfully loaded from file.', prepend_timestamp=False)

    if return_units:

        # convert index in units Series to MultiIndex if needed
        units = base.convert_to_MultiIndex(units, axis=0)

        units.sort_index(inplace=True)

        return data, units

    else:
        return data
