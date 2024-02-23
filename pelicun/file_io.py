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
# John Vouvakis Manousakis

"""
This module has classes and methods that handle file input and output.

.. rubric:: Contents

.. autosummary::

    dict_raise_on_duplicates
    get_required_resources
    save_to_csv
    load_data
    load_from_file
    parse_units

"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import base


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
    'btw. Fragility Groups':   'FG',
    'btw. Performance Groups': 'PG',
    'btw. Floors':             'LOC',
    'btw. Directions':         'DIR',
    'btw. Component Groups':   'CSG',
    'btw. Damage States':      'DS',
    'Independent':             'IND',
    'per ATC recommendation':  'ATC',
}

HAZUS_occ_converter = {
    'RES':  'Residential',
    'COM':  'Commercial',
    'REL':  'Commercial',
    'EDU':  'Educational',
    'IND':  'Industrial',
    'AGR':  'Industrial'
}


def dict_raise_on_duplicates(ordered_pairs):
    """
    Reject duplicate keys.

    https://stackoverflow.com/questions/14902299/
    json-loads-allows-duplicate-keys-
    in-a-dictionary-overwriting-the-first-value

    """
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError(f"duplicate key: {k}")
        d[k] = v
    return d


def save_to_csv(data, filepath, units=None, unit_conversion_factors=None,
                orientation=0, use_simpleindex=True, log=None):
    """
    Saves data to a CSV file following standard SimCenter schema.

    The produced CSV files have a single header line and an index column. The
    second line may start with 'Units' in the index or the first column may be
    'Units' to provide the units for the data in the file.

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
        The location of the destination file. If None, the data is not saved,
        but returned in the end.
    units: Series, optional
        Provides a Series with variables and corresponding units.
    unit_conversion_factors: dict
        Dictionary containing key-value pairs of unit names and their
        corresponding factors. Conversion factors are defined as the number of
        times a base unit fits in the alternative unit.
    orientation: int, {0, 1}, default: 0
        If 0, variables are organized along columns; otherwise they are along
        the rows. This is important when converting values to follow the
        prescribed units.
    use_simpleindex: bool, default: True
        If True, MultiIndex columns and indexes are converted to SimpleIndex
        before saving
    log: Logger
        Logger object to be used. If no object is specified, no logging
        is performed.

    Raises
    ------
    ValueError
        If units is not None but unit_conversion_factors is None
    ValueError
        If writing to a file fails.
    ValueError
        If the provided file name does not have the `.csv` suffix.
    """

    if filepath is None:
        if log: log.msg('Preparing data ...', prepend_timestamp=False)

    elif log: log.msg(f'Saving data to {filepath}...', prepend_timestamp=False)

    if data is not None:

        # make sure we do not modify the original data
        data = data.copy()

        # convert units and add unit information, if needed
        if units is not None:

            if unit_conversion_factors is None:
                raise ValueError(
                    'When units is not None, '
                    'unit_conversion_factors must be provided')

            if log: log.msg('Converting units...', prepend_timestamp=False)

            # if the orientation is 1, we might not need to scale all columns
            if orientation == 1:
                cols_to_scale = [dt in [float, int] for dt in data.dtypes]
                cols_to_scale = data.columns[cols_to_scale]

            labels_to_keep = []

            for unit_name in units.unique():

                labels = units.loc[units == unit_name].index.values

                unit_factor = 1. / unit_conversion_factors[unit_name]

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

            if log: log.msg('Unit conversion successful.', prepend_timestamp=False)

        if use_simpleindex:
            # convert MultiIndex to regular index with '-' separators
            if isinstance(data.index, pd.MultiIndex):
                data = base.convert_to_SimpleIndex(data)

            # same thing for the columns
            if isinstance(data.columns, pd.MultiIndex):
                data = base.convert_to_SimpleIndex(data, axis=1)

        if filepath is not None:

            filepath = Path(filepath).resolve()
            if filepath.suffix == '.csv':

                # save the contents of the DataFrame into a csv
                data.to_csv(filepath)

                if log: log.msg('Data successfully saved to file.',
                                prepend_timestamp=False)

            else:
                raise ValueError(
                    f'ERROR: Unexpected file type received when trying '
                    f'to save to csv: {filepath}')

            return None

        # at this line, filepath is None
        return data

    # at this line, data is None
    if log: log.msg('WARNING: Data was empty, no file saved.',
                    prepend_timestamp=False)
    return None


def load_data(data_source, unit_conversion_factors,
              orientation=0, reindex=True, return_units=False,
              log=None):
    """
    Loads data assuming it follows standard SimCenter tabular schema.

    The data is assumed to have a single header line and an index column. The
    second line may start with 'Units' in the index and provide the units for
    each column in the file.

    Parameters
    ----------
    data_source: string or DataFrame
        If it is a string, the data_source is assumed to point to the location
        of the source file. If it is a DataFrame, the data_source is assumed to
        hold the raw data.
    unit_conversion_factors: dict
        Dictionary containing key-value pairs of unit names and their
        corresponding factors. Conversion factors are defined as the number of
        times a base unit fits in the alternative unit.
    orientation: int, {0, 1}, default: 0
        If 0, variables are organized along columns; otherwise they are along
        the rows. This is important when converting values to follow the
        prescribed units.
    reindex: bool
        If True, reindexes the table to ensure a 0-based, continuous index
    return_units: bool
        If True, returns the units as well as the data to allow for adjustments
        in unit conversion.
    log: Logger
        Logger object to be used. If no object is specified, no logging
        is performed.

    Returns
    -------
    data: DataFrame
        Parsed data.
    units: Series
        Labels from the data and corresponding units specified in the
        data. Units are only returned if return_units is set to True.
    """

    # if the provided data_source is already a DataFrame...
    if isinstance(data_source, pd.DataFrame):
        # we can just store it at proceed
        # (copying is needed to avoid changing the original)
        data = data_source.copy()
    # otherwise, load the data from a file
    elif isinstance(data_source, str):
        data = load_from_file(data_source)
    else:
        raise TypeError(
            f'Invalid data_source type: {type(data_source)}'
        )

    # if orientation == 0 we transpose the dataframe, perform all
    # operations, and transpose it back before returning.

    if orientation == 0:
        data = data.T

    # if there is information about units, perform the conversion to
    # base units
    if 'Units' in data.columns:

        # convert columns to float datatype whenever possible
        try:
            data = data.astype(float)
            float_cols_list = [True] * np.shape(data)[1]
        except ValueError:
            float_cols_list = []
            for col in data:
                if col == 'Units':
                    continue
                try:
                    data[col] = data[col].astype(float)
                    float_cols_list.append(True)
                except ValueError:
                    float_cols_list.append(False)
        float_cols = np.array(float_cols_list)

        if log: log.msg('Converting units...', prepend_timestamp=False)

        units = data['Units']
        data.drop('Units', axis=1, inplace=True)

        unique_unit_names = units.unique()

        for unit_name in unique_unit_names:

            if pd.isna(unit_name):
                continue

            unit_factor = unit_conversion_factors[unit_name]
            data.iloc[(units == unit_name).values, float_cols] *= unit_factor

        if log: log.msg('Unit conversion successful.', prepend_timestamp=False)

    else:

        units = None

    if orientation == 0:
        data = data.T

    # convert columns to MultiIndex if needed
    data = base.convert_to_MultiIndex(data, axis=1)
    data.sort_index(axis=1, inplace=True)

    # reindex the data, if needed
    if reindex:
        data.index = np.arange(data.shape[0])
    else:
        # convert index to MultiIndex if needed
        data = base.convert_to_MultiIndex(data, axis=0)
        data.sort_index(inplace=True)

    if log: log.msg('Data successfully loaded from file.', prepend_timestamp=False)

    if return_units:
        if units is not None:
            # convert index in units Series to MultiIndex if needed
            units = base.convert_to_MultiIndex(units, axis=0).dropna()
            units.sort_index(inplace=True)
        output = data, units
    else:
        output = data

    return output


def load_from_file(filepath, log=None):
    """
    Loads data from a file and stores it in a DataFrame.

    Currently, only CSV files are supported, but the function is easily
    extensible to support other file formats.

    Parameters
    ----------
    filepath: string
        The location of the source file

    Returns
    -------
    data: DataFrame
        Data loaded from the file.
    log: Logger
        Logger object to be used. If no object is specified, no logging
        is performed.

    Raises
    ------
    FileNotFoundError
        If the filepath is invalid.
    ValueError
        If the file is not a CSV.
    """

    if log: log.msg(f'Loading data from {filepath}...')

    # check if the filepath is valid
    filepath = Path(filepath).resolve()

    if not filepath.is_file():
        raise FileNotFoundError(
            f"The filepath provided does not point to an existing "
            f"file: {filepath}")

    if filepath.suffix == '.csv':

        # load the contents of the csv into a DataFrame

        data = pd.read_csv(filepath, header=0, index_col=0, low_memory=False,
                           encoding_errors='replace')

        if log: log.msg('File successfully opened.', prepend_timestamp=False)

    else:
        raise ValueError(f'ERROR: Unexpected file type received when trying '
                         f'to load from csv: {filepath}')

    return data


def parse_units(custom_file=None):
    """
    Parse the unit conversion factor JSON file and return a dictionary.

    Parameters
    ----------
    custom_file: str, optional
        If a custom file is provided, only the units specified in the
        custom file are used.

    Raises
    ------
    KeyError
        If a key is defined twice.
    ValueError
        If a unit conversion factor is not a float.
    FileNotFoundError
        If a file does not exist.
    Exception
        If a file does not have the JSON format.
    """

    def get_contents(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(
                    f, object_pairs_hook=dict_raise_on_duplicates)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f'{file_path} was not found.') from exc
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(
                f'{file_path} is not a valid JSON file.') from exc
        for category_dict in list(dictionary.values()):
            # ensure all first-level keys point to a dictionary
            if not isinstance(category_dict, dict):
                raise ValueError(
                    f'{file_path} contains first-level keys '
                    'that don\'t point to a dictionary')
            # convert values to float
            for key, val in category_dict.items():
                try:
                    category_dict[key] = float(val)
                except (ValueError, TypeError) as exc:
                    raise type(exc)(
                        f'Unit {key} has a value of {val} '
                        'which cannot be interpreted as a float') from exc

        flattened = {}
        for category in dictionary:
            for unit_name, factor in dictionary[category].items():
                if unit_name in flattened:
                    raise ValueError(f'{unit_name} defined twice in {file_path}.')
                flattened[unit_name] = factor

        return flattened

    if custom_file:
        return get_contents(custom_file)

    return get_contents(base.pelicun_path / "settings/default_units.json")
