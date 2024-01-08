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
This module defines constants, basic classes and methods for pelicun.

.. rubric:: Contents

.. autosummary::

    convert_to_SimpleIndex
    convert_to_MultiIndex
    show_matrix
    describe
    str2bool
    float_or_None
    int_or_None
    process_loc
    dedupe_index

    Options
    Logger

"""

import os
import sys
from datetime import datetime
import warnings
from pathlib import Path
import argparse
import pprint
import numpy as np
import pandas as pd
from . import file_io


# set printing options
pp = pprint.PrettyPrinter(indent=2, width=80 - 24)

pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = True
pd.options.display.width = 300

idx = pd.IndexSlice

# don't show FutureWarnings by default
if not sys.warnoptions:
    warnings.filterwarnings(
        category=FutureWarning, action='ignore')


class Options:

    """
    Options objects store analysis options and the logging
    configuration.

    Attributes
    ----------
    sampling_method: str
        Sampling method to use. Specified in the user's configuration
        dictionary, otherwise left as provided in the default configuration
        file (see settings/default_config.json in the pelicun source
        code). Can be any of ['LHS', 'LHS_midpoint',
        'MonteCarlo']. The default is 'LHS'.
    units_file: str
        Location of a user-specified units file, which should contain
        the names of supported units and their conversion factors (the
        value some quantity of a given unit needs to be multiplied to
        be expressed in the base units). Value specified in the user
        configuration dictionary. Pelicun comes with a set of default
        units which are always loaded (see settings/default_units.json
        in the pelicun source code). Units specified in the units_file
        overwrite the default units.
    demand_offset: dict
        Demand offsets are used in the process of mapping a component
        location to its associated EDP. This allows components that
        are sensitive to EDPs of different levels to be specified as
        present at the same location (e.g. think of desktop computer
        and suspended ceiling, both at the same story). Each
        component's offset value is specified in the component
        fragility database. This setting applies a supplemental global
        offset to specific EDP types. The value is specified in the
        user's configuration dictionary, otherwise left as provided in
        the default configuration file (see
        settings/default_config.json in the pelicun source code).
    nondir_multi_dict: dict
        Nondirectional components are sensitive to demands coming in
        any direction. Results are typically available in two
        orthogonal directions. FEMA P-58 suggests using the formula
        `max(dir_1, dir_2) * 1.2` to estimate the demand for such
        components. This parameter allows modifying the 1.2 multiplier
        with a user-specified value. The change can be applied to
        "ALL" EDPs, or for specific EDPs, such as "PFA", "PFV",
        etc. The value is specified in the user's configuration
        dictionary, otherwise left as provided in the default
        configuration file (see settings/default_config.json in the
        pelicun source code).
    rho_cost_time: float
        Specifies the correlation between the repair cost and repair
        time consequences. The value is specified in the user's
        configuration dictionary, otherwise left as provided in the
        default configuration file (see
        "RepairCostAndTimeCorrelation") (see
        settings/default_config.json in the pelicun source code).
    eco_scale: dict
        Controls how the effects of economies of scale are handled in
        the damaged component quantity aggregation for loss measure
        estimation. The dictionary is specified in the user's
        configuration dictionary, otherwise left as provided in the
        default configuration file (see settings/default_config.json
        in the pelicun source code).
    log: Logger
        Logger object. Configuration parameters coming from the user's
        configuration dictionary or the default configuration file
        control logging behavior. See Logger class.

    """

    def __init__(self, user_config_options, assessment=None):
        """
        Initializes an Options object.

        Parameters
        ----------
        user_config_options: dict, Optional
            User-specified configuration dictionary. Any provided
            user_config_options override the defaults.
        assessment: Assessment, Optional
            Assessment object that will be using this Options
            object. If it is not intended to use this Options object
            for an Assessment (e.g. defining an Options object for UQ
            use), this value should be None.
        """

        self._asmnt = assessment

        self._verbose = False
        self._log_show_ms = False
        self._print_log = False
        self._log_file = None

        self.defaults = None
        self.sampling_method = None
        self.list_all_ds = None

        self._seed = None

        self._rng = np.random.default_rng()
        merged_config_options = file_io.merge_default_config(
            user_config_options)

        self._seed = merged_config_options['Seed']
        self.sampling_method = merged_config_options['SamplingMethod']

        self.units_file = merged_config_options['UnitsFile']

        self.demand_offset = merged_config_options['DemandOffset']
        self.nondir_multi_dict = merged_config_options['NonDirectionalMultipliers']
        self.rho_cost_time = merged_config_options['RepairCostAndTimeCorrelation']
        self.eco_scale = merged_config_options['EconomiesOfScale']

        # instantiate a Logger object with the finalized configuration
        self.log = Logger(
            merged_config_options['Verbose'],
            merged_config_options['LogShowMS'],
            merged_config_options['LogFile'],
            merged_config_options['PrintLog'])

    def nondir_multi(self, EDP_type):
        """
        Returns the multiplicative factor used in nondirectional
        component demand generation. Read the description of the
        nondir_multi_dict attribute of the Options class.

        Parameters
        ----------
        EDP_type: str
            EDP type (e.g. "PFA", "PFV", ..., "ALL")

        Raises
        ------
        ValueError
            If the specified EDP type is not present in the
            dictionary.  If this is the case, a value for that type
            needs to be specified in the user's configuration
            dictionary, under ['Options']['NonDirectionalMultipliers']
            = {"edp_type": value, ...}
        """

        if EDP_type in self.nondir_multi_dict:
            return self.nondir_multi_dict[EDP_type]

        if 'ALL' in self.nondir_multi_dict:
            return self.nondir_multi_dict['ALL']

        raise ValueError(
            f"Peak orthogonal EDP multiplier for non-directional demand "
            f"calculation of {EDP_type} not specified.\n"
            f"Please add {EDP_type} in the configuration dictionary "
            f"under ['Options']['NonDirectionalMultipliers']"
            " = {{'edp_type': value, ...}}")

    @property
    def seed(self):
        """
        seed property
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        """
        seed property setter
        """
        self._seed = value
        self._rng = np.random.default_rng(self._seed)

    @property
    def rng(self):
        """
        rng property
        """
        return self._rng

    @property
    def units_file(self):
        """
        units file property
        """
        return self._units_file

    @units_file.setter
    def units_file(self, value):
        """
        units file property setter
        """
        self._units_file = value

class Logger:

    """
    Logger objects are used to generate log files documenting
    execution events and related messages.

    Attributes
    ----------
    verbose: bool
        If True, the pelicun echoes more information throughout the
        assessment.  This can be useful for debugging purposes. The
        value is specified in the user's configuration dictionary,
        otherwise left as provided in the default configuration file
        (see settings/default_config.json in the pelicun source code).
    log_show_ms: bool
        If True, the timestamps in the log file are in microsecond
        precision. The value is specified in the user's configuration
        dictionary, otherwise left as provided in the default
        configuration file (see settings/default_config.json in the
        pelicun source code).
    log_file: str, optional
        If a value is provided, the log is written to that file. The
        value is specified in the user's configuration dictionary,
        otherwise left as provided in the default configuration file
        (see settings/default_config.json in the pelicun source code).
    print_log: bool
        If True, the log is also printed to standard output. The
        value is specified in the user's configuration dictionary,
        otherwise left as provided in the default configuration file
        (see settings/default_config.json in the pelicun source code).

    """
    # TODO: finalize docstring

    def __init__(self, verbose, log_show_ms, log_file, print_log):
        """
        Initializes a Logger object.

        Parameters
        ----------
        see attributes of the Logger class.

        """
        self.verbose = verbose
        self.log_show_ms = log_show_ms
        self.log_file = log_file
        self.print_log = print_log
        self.reset_log_strings()

    @property
    def verbose(self):
        """
        verbose property
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """
        verbose property setter
        """
        self._verbose = bool(value)
        # display FutureWarnings
        if self._verbose is True:
            if not sys.warnoptions:
                warnings.filterwarnings(
                    category=FutureWarning,
                    action='default')

    @property
    def log_show_ms(self):
        """
        log_show_ms property
        """
        return self._log_show_ms

    @log_show_ms.setter
    def log_show_ms(self, value):
        """
        log_show_ms property setter
        """
        self._log_show_ms = bool(value)

        self.reset_log_strings()

    @property
    def log_pref(self):
        """
        log_pref property
        """
        return self._log_pref

    @property
    def log_div(self):
        """
        log_div property
        """
        return self._log_div

    @property
    def log_time_format(self):
        """
        log_time_format property
        """
        return self._log_time_format

    @property
    def log_file(self):
        """
        log_file property
        """
        return self._log_file

    @log_file.setter
    def log_file(self, value):
        """
        log_file property setter
        """

        if value is None:
            self._log_file = None

        else:

            try:

                filepath = Path(value).resolve()

                self._log_file = str(filepath)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('')

            except BaseException as err:
                print(f"WARNING: The filepath provided for the log file does "
                      f"not point to a valid location: {value}. \nPelicun "
                      f"cannot print the log to a file.\n"
                      f"The error was: '{err}'")
                raise

    @property
    def print_log(self):
        """
        print_log property
        """
        return self._print_log

    @print_log.setter
    def print_log(self, value):
        """
        print_log property setter
        """
        self._print_log = str2bool(value)

    def reset_log_strings(self):
        """
        Populates the string-related attributes of the logger
        """

        if self._log_show_ms:
            self._log_time_format = '%H:%M:%S:%f'
            # the length of the time string in the log file
            self._log_pref = ' ' * 16
            # to have a total length of 80 with the time added
            self._log_div = '-' * (80 - 17)
        else:
            self._log_time_format = '%H:%M:%S'
            self._log_pref = ' ' * 9
            self._log_div = '-' * (80 - 10)

    def msg(self, msg='', prepend_timestamp=True, prepend_blank_space=True):
        """
        Writes a message in the log file with the current time as prefix

        The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

        Parameters
        ----------
        msg: string
            Message to print.
        prepend_timestamp: bool
            Controls whether a timestamp is placed before the message.
        prepend_blank_space: bool
            Controls whether blank space is placed before the message.

        """

        # pylint: disable = consider-using-f-string
        msg_lines = msg.split('\n')

        for msg_i, msg_line in enumerate(msg_lines):

            if (prepend_timestamp and (msg_i == 0)):
                formatted_msg = '{} {}'.format(
                    datetime.now().strftime(self.log_time_format), msg_line)
            elif prepend_timestamp:
                formatted_msg = self.log_pref + msg_line
            elif prepend_blank_space:
                formatted_msg = self.log_pref + msg_line
            else:
                formatted_msg = msg_line

            if self.print_log:
                print(formatted_msg)

            if self.log_file is not None:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write('\n' + formatted_msg)

    def div(self, prepend_timestamp=False):
        """
        Adds a divider line in the log file
        """

        if prepend_timestamp:
            msg = self.log_div
        else:
            msg = '-' * 80
        self.msg(msg, prepend_timestamp=prepend_timestamp)

    def print_system_info(self):
        """
        Writes system information in the log.
        """

        self.msg(
            'System Information:',
            prepend_timestamp=False, prepend_blank_space=False)
        self.msg(
            f'local time zone: {datetime.utcnow().astimezone().tzinfo}\n'
            f'start time: {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}\n'
            f'python: {sys.version}\n'
            f'numpy: {np.__version__}\n'
            f'pandas: {pd.__version__}\n',
            prepend_timestamp=False)

# get the absolute path of the pelicun directory
pelicun_path = Path(os.path.dirname(os.path.abspath(__file__)))


def convert_to_SimpleIndex(data, axis=0, inplace=False):
    """
    Converts the index of a DataFrame to a simple, one-level index

    The target index uses standard SimCenter convention to identify different
    levels: a dash character ('-') is used to separate each level of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int, optional, default:0
        Identifies if the index (0) or the columns (1) shall be edited.
    inplace: bool, optional, default:False
        If yes, the operation is performed directly on the input DataFrame
        and not on a copy of it.

    Returns
    -------
    data: DataFrame
        The modified DataFrame

    Raises
    ------
    ValueError
        When an invalid axis parameter is specified
    """

    if axis in {0, 1}:

        if inplace:
            data_mod = data
        else:
            data_mod = data.copy()

        if axis == 0:

            # only perform this if there are multiple levels
            if data.index.nlevels > 1:

                simple_name = '-'.join(
                    [n if n is not None else "" for n in data.index.names])
                simple_index = ['-'.join([str(id_i) for id_i in id])
                                for id in data.index]

                data_mod.index = simple_index
                data_mod.index.name = simple_name

        elif axis == 1:

            # only perform this if there are multiple levels
            if data.columns.nlevels > 1:

                simple_name = '-'.join(
                    [n if n is not None else "" for n in data.columns.names])
                simple_index = ['-'.join([str(id_i) for id_i in id])
                                for id in data.columns]

                data_mod.columns = simple_index
                data_mod.columns.name = simple_name

    else:
        raise ValueError(f"Invalid axis parameter: {axis}")

    return data_mod


def convert_to_MultiIndex(data, axis=0, inplace=False):
    """
    Converts the index of a DataFrame to a MultiIndex

    We assume that the index uses standard SimCenter convention to identify
    different levels: a dash character ('-') is expected to separate each level
    of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int, optional, default:0
        Identifies if the index (0) or the columns (1) shall be edited.
    inplace: bool, optional, default:False
        If yes, the operation is performed directly on the input DataFrame
        and not on a copy of it.

    Returns
    -------
    data: DataFrame
        The modified DataFrame.

    Raises
    ------
    ValueError
        If an invalid axis is specified.
    """

    # check if the requested axis is already a MultiIndex
    if (((axis == 0) and (isinstance(data.index, pd.MultiIndex))) or (
            (axis == 1) and (isinstance(data.columns, pd.MultiIndex)))):

        # if yes, return the data unchanged
        return data

    if axis == 0:
        index_labels = [str(label).split('-') for label in data.index]

    elif axis == 1:
        index_labels = [str(label).split('-') for label in data.columns]

    else:
        raise ValueError(f"Invalid axis parameter: {axis}")

    max_lbl_len = np.max([len(labels) for labels in index_labels])

    for l_i, labels in enumerate(index_labels):

        if len(labels) != max_lbl_len:
            labels += ['', ] * (max_lbl_len - len(labels))
            index_labels[l_i] = labels

    index_labels = np.array(index_labels)

    if index_labels.shape[1] > 1:

        if inplace:
            data_mod = data
        else:
            data_mod = data.copy()

        if axis == 0:
            data_mod.index = pd.MultiIndex.from_arrays(index_labels.T)

        else:
            data_mod.columns = pd.MultiIndex.from_arrays(index_labels.T)

        return data_mod

    return data


def show_matrix(data, use_describe=False):
    """
    Print a matrix in a nice way using a DataFrame
    """
    if use_describe:
        pp.pprint(pd.DataFrame(data).describe(
            percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))
    else:
        pp.pprint(pd.DataFrame(data))


def _warning(message, category, filename, lineno, file=None, line=None):
    """
    Monkeypatch warnings to get prettier messages
    """
    # pylint:disable = unused-argument
    if '\\' in filename:
        file_path = filename.split('\\')
    elif '/' in filename:
        file_path = filename.split('/')
    else:
        file_path = None

    if file_path is not None:
        python_file = '/'.join(file_path[-3:])
    else:
        python_file = filename

    print(f'WARNING in {python_file} at line {lineno}\n{message}\n')


warnings.showwarning = _warning


def describe(df, percentiles=(0.001, 0.023, 0.10, 0.159, 0.5, 0.841, 0.90,
                              0.977, 0.999)):
    """
    Provide descriptive statistics.
    """
    if not isinstance(df, (pd.Series, pd.DataFrame)):
        vals = df
        cols = np.arange(vals.shape[1]) if vals.ndim > 1 else 0

        if vals.ndim == 1:
            df = pd.Series(vals, name=cols)
        else:
            df = pd.DataFrame(vals, columns=cols)

    # cast Series into a DataFrame
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    desc = df.describe(percentiles).T

    # add log standard deviation to the stats
    desc.insert(3, "log_std", np.nan)
    desc = desc.T

    for col in desc.columns:
        if np.min(df[col]) > 0.0:
            desc.loc['log_std', col] = np.std(np.log(df[col]), ddof=1)

    return desc


def str2bool(v):
    """
    Converts various bool-like forms of string to actual booleans
    """
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 'True', 't', 'y', '1'}:
        return True
    if v.lower() in {'no', 'false', 'False', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def float_or_None(string):
    """
    This is a convenience function for converting strings to float or
    None

    Parameters
    ----------
    string: str
        A string

    Returns
    -------
    res: float, optional
        A float, if the given string can be converted to a
        float. Otherwise, it returns None
    """
    try:
        res = float(string)
        return res
    except ValueError:
        return None


def int_or_None(string):
    """
    This is a convenience function for converting strings to int or
    None

    Parameters
    ----------
    string: str
        A string

    Returns
    -------
    res: int, optional
        An int, if the given string can be converted to an
        int. Otherwise, it returns None
    """
    try:
        res = int(string)
        return res
    except ValueError:
        return None


def process_loc(string, stories):
    """
    Parses the location parameter.
    """
    try:
        res = int(string)
        return [res, ]
    except ValueError:
        if "-" in string:
            s_low, s_high = string.split('-')
            s_low = process_loc(s_low, stories)
            s_high = process_loc(s_high, stories)
            return list(range(s_low[0], s_high[0] + 1))
        if string == "all":
            return list(range(1, stories + 1))
        if string == "top":
            return [stories, ]
        if string == "roof":
            return [stories, ]
        return None


def dedupe_index(dataframe, dtype=str):
    """
    Adds an extra level to the index of a dataframe so that all
    resulting index elements are unique. Assumes that the original
    index is a MultiIndex with specified names.

    """

    inames = dataframe.index.names
    dataframe.reset_index(inplace=True)
    dataframe['uid'] = (
        dataframe.groupby([*inames]).cumcount()).astype(dtype)
    dataframe.set_index([*inames] + ['uid'], inplace=True)
    dataframe.sort_index(inplace=True)


# Input specs

EDP_to_demand_type = {
    # Drifts
    'Story Drift Ratio':               'PID',
    'Peak Interstory Drift Ratio':     'PID',
    'Roof Drift Ratio':                'PRD',
    'Peak Roof Drift Ratio':           'PRD',
    'Damageable Wall Drift':           'DWD',
    'Racking Drift Ratio':             'RDR',
    'Mega Drift Ratio':                'PMD',
    'Residual Drift Ratio':            'RID',
    'Residual Interstory Drift Ratio': 'RID',
    'Peak Effective Drift Ratio':      'EDR',

    # Floor response
    'Peak Floor Acceleration':        'PFA',
    'Peak Floor Velocity':            'PFV',
    'Peak Floor Displacement':        'PFD',

    # Component response
    'Peak Link Rotation Angle':       'LR',
    'Peak Link Beam Chord Rotation':  'LBR',

    # Wind Intensity
    'Peak Gust Wind Speed':           'PWS',

    # Inundation Intensity
    'Peak Inundation Height':         'PIH',

    # Shaking Intensity
    'Peak Ground Acceleration':       'PGA',
    'Peak Ground Velocity':           'PGV',
    'Spectral Acceleration':          'SA',
    'Spectral Velocity':              'SV',
    'Spectral Displacement':          'SD',
    'Peak Spectral Acceleration':     'SA',
    'Peak Spectral Velocity':         'SV',
    'Peak Spectral Displacement':     'SD',
    'Permanent Ground Deformation':   'PGD',

    # Placeholder for advanced calculations
    'One':                            'ONE'
}
