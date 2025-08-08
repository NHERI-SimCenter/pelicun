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

"""Classes and methods that handle file input and output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pelicun import base

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
    'AGR': 'Industrial',
}

legacy_names = {
    'damage_DB_FEMA_P58_2nd': 'FEMA P-58',
    'damage_DB_Hazus_EQ_bldg': 'Hazus Earthquake - Buildings',
    'damage_DB_Hazus_EQ_story': 'Hazus Earthquake - Stories',
    'damage_DB_Hazus_EQ_trnsp': 'Hazus Earthquake - Transportation',
    'damage_DB_Hazus_EQ_water': 'Hazus Earthquake - Water',
    'damage_DB_Hazus_EQ_power': 'Hazus Earthquake - Power',
    'damage_DB_SimCenter_Hazus_HU_bldg': 'Hazus Hurricane Wind',
    'loss_repair_DB_FEMA_P58_2nd': 'FEMA P-58',
    'loss_repair_DB_Hazus_EQ_bldg': 'Hazus Earthquake - Buildings',
    'loss_repair_DB_Hazus_EQ_story': 'Hazus Earthquake - Stories',
    'loss_repair_DB_Hazus_EQ_trnsp': 'Hazus Earthquake - Transportation',
    'loss_repair_DB_SimCenter_Hazus_HU_bldg': 'Hazus Hurricane Wind',
}


def save_to_csv(  # noqa: C901
    data: pd.DataFrame | None,
    filepath: Path | None,
    units: pd.Series | None = None,
    unit_conversion_factors: dict | None = None,
    orientation: int = 0,
    *,
    use_simpleindex: bool = True,
    log: base.Logger | None = None,
) -> pd.DataFrame | None:
    """
    Save data to a CSV file following the standard SimCenter schema.

    The produced CSV files have a single header line and an index
    column. The second line may start with 'Units' in the index or the
    first column may be 'Units' to provide the units for the data in
    the file.

    Parameters
    ----------
    data: DataFrame
        The data to save.
    filepath: Path
        The location of the destination file. If None, the data is not
        saved, but returned in the end.
    units: Series, optional
        Provides a Series with variables and corresponding units.
    unit_conversion_factors: dict, optional
        Dictionary containing key-value pairs of unit names and their
        corresponding factors. Conversion factors are defined as the
        number of times a base unit fits in the alternative unit.
    orientation: int, {0, 1}, default 0
        If 0, variables are organized along columns; otherwise, they
        are along the rows. This is important when converting values
        to follow the prescribed units.
    use_simpleindex: bool, default True
        If True, MultiIndex columns and indexes are converted to
        SimpleIndex before saving.
    log: Logger, optional
        Logger object to be used. If no object is specified, no
        logging is performed.

    Raises
    ------
    ValueError
        If units is not None but unit_conversion_factors is None.
    ValueError
        If writing to a file fails.
    ValueError
        If the provided file name does not have the `.csv` suffix.

    Returns
    -------
    DataFrame or None
        If `filepath` is None, returns the DataFrame with potential
        unit conversions and reformatting applied. Otherwise, returns
        None after saving the data to a CSV file.

    """
    if filepath is None:
        if log:
            log.msg('Preparing data ...', prepend_timestamp=False)

    elif log:
        log.msg(f'Saving data to `{filepath!s}`...', prepend_timestamp=False)

    if data is None:
        if log:
            log.warning('Data was empty, no file saved.')
        return None

    assert isinstance(data, pd.DataFrame)

    # make sure we do not modify the original data
    data = data.copy()

    # convert units and add unit information, if needed
    if units is not None:
        if unit_conversion_factors is None:
            msg = (
                'When `units` is not None, '
                '`unit_conversion_factors` must be provided.'
            )
            raise ValueError(msg)

        if log:
            log.msg('Converting units...', prepend_timestamp=False)

        # if the orientation is 1, we might not need to scale all columns
        if orientation == 1:
            cols_to_scale_bool = [dt in {float, int} for dt in data.dtypes]
            cols_to_scale = data.columns[cols_to_scale_bool]

        labels_to_keep = []

        for unit_name in units.unique():
            labels = units.loc[units == unit_name].index.to_numpy()

            unit_factor = 1.0 / unit_conversion_factors[unit_name]

            active_labels = []

            if orientation == 0:
                for label in labels:
                    if label in data.columns:
                        active_labels.append(label)  # noqa: PERF401

                if len(active_labels) > 0:
                    data.loc[:, active_labels] *= unit_factor

            else:  # elif orientation == 1:
                for label in labels:
                    if label in data.index:
                        active_labels.append(label)  # noqa: PERF401

                if len(active_labels) > 0:
                    data.loc[np.array(active_labels), np.array(cols_to_scale)] *= (
                        unit_factor
                    )

            labels_to_keep += active_labels

        units_df = units.loc[labels_to_keep].to_frame()

        if orientation == 0:
            data = pd.concat([units_df.T, data], axis=0)
            data = data.sort_index(axis=1)
        else:
            data = pd.concat([units_df, data], axis=1)
            data = data.sort_index()

        if log:
            log.msg('Unit conversion successful.', prepend_timestamp=False)

    assert isinstance(data, pd.DataFrame)
    if use_simpleindex:
        # convert MultiIndex to regular index with '-' separators
        if isinstance(data.index, pd.MultiIndex):
            data = base.convert_to_SimpleIndex(data)

        # same thing for the columns
        if isinstance(data.columns, pd.MultiIndex):
            data = base.convert_to_SimpleIndex(data, axis=1)

    if filepath is not None:
        if filepath.suffix == '.csv':
            # save the contents of the DataFrame into a csv
            data.to_csv(filepath)

            if log:
                log.msg('Data successfully saved to file.', prepend_timestamp=False)

        else:
            msg = (
                f'Please use the `.csv` file extension. '
                f'Received file name is `{filepath}`'
            )
            raise ValueError(msg)

        return None

    # at this line, filepath is None
    return data


def substitute_default_path(
    data_paths: list[str | pd.DataFrame], log: base.Logger | None = None
) -> list[str | pd.DataFrame]:
    """
    Substitute the default directory path.

    This function iterates over a list of data paths and replaces
    those with the 'PelicunDefault/' substring with the full paths to
    model files in the built-in Damage and Loss Model Library.
    Default paths are expected to follow the
    `PelicunDefault/method_name/model_type.extension` structure. The
    `method_name` identifies the methodology from those available in the
    `{base.pelicun_path}/resources/dlml_resource_paths.json` file. The
    `model_type` identifies the type of model requested. Currently, the
    following types are supported: 'fragility', 'consequence_repair',
    'loss_repair'. The `extension` is intended to identify 'CSV' files with
    model parameters and 'JSON' files with metadata.
    The `model_type` and `extension` strings are not limited to the
    supported values. If you know a particular file exists in the method's
    folder, you can use the corresponding `model_type.extension` to access
    that file.

    Parameters
    ----------
    data_paths: list of str or pd.DataFrame
        A list containing the paths to data files. These paths may
        include a placeholder directory 'PelicunDefault/' that needs
        to be substituted with the actual path specified in the
        resource mapping.
    log: Logger
        Logger object to be used. If no object is specified, no logging
        is performed.

    Returns
    -------
    list of str or pd.DataFrame

    Raises
    ------
    KeyError
      If the method_name after 'PelicunDefault/' does not exist in the
      `resource_paths` keys.
      If the method_name after 'PelicunDefault/' does not exist in the
      legacy list of filenames preserved for backwards compatibility.

    Notes
    -----
    - The function assumes that `base.pelicun_path` is properly
      initialized and points to the correct directory where resources
      are located.
    - If a path in the input list does not contain 'PelicunDefault/',
      the path is added to the output list unchanged.

    Examples
    --------
    >>> data_paths = ['PelicunDefault/Hazus Hurricane/fragility.csv', 'data/file2.txt']
    >>> substitute_default_path(data_paths)
    ['{base.pelicun_path}/resources/DamageAndLossModelLibrary/'
      'hurricane/building/portfolio/Hazus v5.1 coupled/fragility.csv',
      'data/file2.txt']

    """
    # Load the resource paths from the JSON file
    resource_file_path = (
        Path(base.pelicun_path) / 'resources' / 'dlml_resource_paths.json'
    )
    with resource_file_path.open('r') as file:
        resource_paths = json.load(file)

    updated_paths: list[str | pd.DataFrame] = []
    for data_path_str in data_paths:
        if isinstance(data_path_str, str) and 'PelicunDefault/' in data_path_str:
            data_path = Path(data_path_str)
            # Extract the filename from the end after 'PelicunDefault/'
            file_name = data_path.parts[-1]

            # Check if there is a method name identified
            method_name = data_path.parts[-2]

            # <backwards compatibility>
            if method_name == 'PelicunDefault':
                # No method name, check for legacy input
                if file_name.startswith(
                    ('fragility_DB', 'damage_DB', 'bldg_repair_DB', 'loss_repair_DB')
                ):
                    if log:
                        log.warning(
                            'Default libraries are no longer referenced using '
                            'the following placeholder filenames after "PelicunDB/": '
                            '`fragility_DB...`, `damage_DB...`, `bldg_repair_DB...`, '
                            '`loss_repair_DB...`. Such inputs will lead to errors in '
                            'future versions of pelicun. Please replace such '
                            'references with a combination of a specific method and '
                            'data type. For example, use '
                            '`PelicunDefault/FEMA P-58/fragility` to get FEMA P-58 '
                            'damage models, and '
                            '`PelicunDefault/Hazus Hurricane/consequence_repair` to '
                            'get Hazus hurricane consequence models. See the online '
                            'documentation for more details.'
                        )

                    method_name = legacy_names[file_name.split('.')[0]]
                    if file_name.startswith(('fragility', 'damage')):
                        data_type = 'fragility'
                    else:
                        data_type = 'consequence_repair'

                    extension = file_name.split('.')[-1]
                    file_name = f'{data_type}.{extension}'

                else:
                    msg = f'Default data path `{data_path_str}` not recognized.'
                    raise KeyError(msg)

            # Check if the method name exists in the resource paths dictionary
            if method_name not in resource_paths:
                msg = f'Method `{method_name}` not found in resource paths.'
                raise KeyError(msg)
            method_folder = resource_paths[method_name]

            # Substitute the default path with a full path to the file
            updated_path = str(
                Path(base.pelicun_path)
                / 'resources'
                / 'DamageAndLossModelLibrary'
                / method_folder
                / file_name
            )
            updated_paths.append(updated_path)
        else:
            updated_paths.append(data_path_str)

    return updated_paths


def load_data(  # noqa: C901
    data_source: str | pd.DataFrame,
    unit_conversion_factors: dict | None = None,
    orientation: int = 0,
    *,
    reindex: bool = True,
    return_units: bool = False,
    log: base.Logger | None = None,
) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
    """
    Load data assuming it follows standard SimCenter tabular schema.

    The data is assumed to have a single header line and an index column. The
    second line may start with 'Units' in the index and provide the units for
    each column in the file.

    Parameters
    ----------
    data_source: string or DataFrame
        If it is a string, the data_source is assumed to point to the location
        of the source file. If it is a DataFrame, the data_source is assumed to
        hold the raw data.
    unit_conversion_factors: dict, optional
        Dictionary containing key-value pairs of unit names and their
        corresponding factors. Conversion factors are defined as the
        number of times a base unit fits in the alternative unit. If
        no conversion factors are specified, then no unit conversions
        are made.
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
    tuple
        data: DataFrame
            Parsed data.
        units: Series
            Labels from the data and corresponding units specified in the
            data. Units are only returned if return_units is set to True.

    Raises
    ------
    TypeError
        If `data_source` is neither a string nor a DataFrame, a
        TypeError is raised.

    """
    if isinstance(data_source, pd.DataFrame):
        # store it at proceed (copying is needed to avoid changing the
        # original)
        data = base.with_parsed_str_na_values(data_source.copy())
    elif isinstance(data_source, str):
        # otherwise, load the data from a file
        data = load_from_file(data_source)
    else:
        msg = f'Invalid data_source type: {type(data_source)}'
        raise TypeError(msg)

    # Define a dictionary to decide the axis based on the orientation
    axis = {0: 1, 1: 0}
    the_index = data.columns if orientation == 1 else data.index

    # Check for units information (case-insensitive)
    # if there is information about units, separate that information
    # and optionally apply conversions to all numeric values
    units_key = None
    for key in the_index:
        if str(key).lower() == 'units':
            units_key = key
            break

    if units_key is not None:
        units = data[units_key] if orientation == 1 else data.loc[units_key]
        data = data.drop([units_key], axis=orientation)  # type: ignore
        data = base.convert_dtypes(data)

        if unit_conversion_factors is not None:
            numeric_elements = (
                (data.select_dtypes(include=[np.number]).index)  # type: ignore
                if orientation == 0
                else (
                    data.select_dtypes(include=[np.number]).columns  # type: ignore
                )
            )

            if log:
                log.msg('Converting units...', prepend_timestamp=False)

            conversion_factors = units.map(
                lambda unit: (
                    1.00
                    if pd.isna(unit)
                    else unit_conversion_factors.get(unit, 1.00)
                )
            )

            if orientation == 1:
                data.loc[:, numeric_elements] = data.loc[
                    :, numeric_elements  # type: ignore
                ].multiply(
                    conversion_factors,
                    axis=axis[orientation],  # type: ignore
                )  # type: ignore
            else:
                data.loc[numeric_elements, :] = data.loc[
                    numeric_elements, :
                ].multiply(
                    conversion_factors,
                    axis=axis[orientation],  # type: ignore
                )  # type: ignore

        if log:
            log.msg('Unit conversion successful.', prepend_timestamp=False)

    else:
        units = None
        data = base.convert_dtypes(data)

    # convert columns or index to MultiIndex if needed
    data = base.convert_to_MultiIndex(data, axis=1)
    data = data.sort_index(axis=1)

    # reindex the data, if needed
    if reindex:
        data.index = pd.RangeIndex(start=0, stop=data.shape[0], step=1)
    else:
        # convert index to MultiIndex if needed
        data = base.convert_to_MultiIndex(data, axis=0)
        data = data.sort_index()

    if return_units:
        if units is not None:
            # convert index in units Series to MultiIndex if needed
            units = base.convert_to_MultiIndex(units, axis=0).dropna()  # type: ignore
            units = units.sort_index()
        output = data, units
    else:
        output = data  # type: ignore

    return output  # type: ignore


def load_from_file(filepath: str, log: base.Logger | None = None) -> pd.DataFrame:
    """
    Load data from a file and stores it in a DataFrame.

    Currently, only CSV files are supported, but the function is easily
    extensible to support other file formats.

    Parameters
    ----------
    filepath: string
        The location of the source file.
    log: base.Logger, optional
        Optional logger object.

    Returns
    -------
    tuple
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
    if log:
        log.msg(f'Loading data from {filepath}...')

    # check if the filepath is valid
    filepath_path = Path(filepath).resolve()

    if not filepath_path.is_file():
        msg = (
            f'The filepath provided does not point to an existing '
            f'file: {filepath_path}'
        )
        raise FileNotFoundError(msg)

    if filepath_path.suffix == '.csv':
        # load the contents of the csv into a DataFrame

        data = pd.read_csv(
            filepath_path,
            header=0,
            index_col=0,
            low_memory=False,
            encoding_errors='replace',
        )

        if log:
            log.msg('File successfully opened.', prepend_timestamp=False)

    else:
        msg = (
            f'Unexpected file type received when trying '
            f'to load from csv: {filepath_path}'
        )
        raise ValueError(msg)

    return data
