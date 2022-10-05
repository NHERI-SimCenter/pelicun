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

"""
This module defines constants, basic classes and methods for pelicun.

.. rubric:: Contents

.. autosummary::

    convert_to_SimpleIndex
    convert_to_MultiIndex
    convert_unit
    show_matrix
    show_warning
    print_system_info
    log_div
    log_msg
    describe
    str2bool

    Options

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
pp = pprint.PrettyPrinter(indent=2, width=80-24)

pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = True
pd.options.display.width = 300

idx = pd.IndexSlice


class Options:

    """
    Options objects store analysis options and the logging
    configuration. Calling the `set_options` method is required after
    initializing those objects.

    Parameters
    ----------

    verbose: boolean
        If True, the pelicun echoes more information throughout the assessment.
        This can be useful for debugging purposes.
    log_show_ms: boolean
        If True, the timestamps in the log file are in microsecond precision.
    """
    # TODO update this docstring

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

        if EDP_type in self.nondir_multi_dict:
            return self.nondir_multi_dict[EDP_type]

        if 'ALL' in self.nondir_multi_dict:
            return self.nondir_multi_dict['ALL']

        raise ValueError(f"Scale factor for non-directional demand "
                         f"calculation of {EDP_type} not specified.")

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

        self._rng = np.random.default_rng(self._seed)

    @property
    def rng(self):
        return self._rng

    @property
    def units_file(self):
        return self._units_file

    @units_file.setter
    def units_file(self, value):
        self._units_file = value


class Logger:

    """
    Logger objects are used to generate log files documenting
    execution events and related messages.

    Methods
    -------
    ...

    Attributes
    ----------
    ...

    """
    # TODO: finalize docstring

    def __init__(self, verbose, log_show_ms, log_file, print_log):
        """
        Initializes a Logger object.

        Parameters
        ----------
        options_obj: Options
            Options object. See the Options class.
        """
        self.verbose = verbose
        self.log_show_ms = log_show_ms
        self.log_file = log_file
        self.print_log = print_log
        self.reset_log_strings()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)

    @property
    def log_show_ms(self):
        return self._log_show_ms

    @log_show_ms.setter
    def log_show_ms(self, value):
        self._log_show_ms = bool(value)

        self.reset_log_strings()

    @property
    def log_pref(self):
        return self._log_pref

    @property
    def log_div(self):
        return self._log_div

    @property
    def log_time_format(self):
        return self._log_time_format

    @property
    def log_file(self):
        return self._log_file

    @log_file.setter
    def log_file(self, value):

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
        return self._print_log

    @print_log.setter
    def print_log(self, value):
        self._print_log = str2bool(value)

    def reset_log_strings(self):

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

        """
        # TODO update this docstring

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
                    f.write('\n'+formatted_msg)

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
    ValueError:
        When an invalid axis parameter is specified
    """

    if axis in {0, 1}:

        if inplace:
            data_mod = data
        else:
            data_mod = data.copy()

        if axis == 0:
            simple_name = '-'.join(
                [n if n is not None else "" for n in data.index.names])
            simple_index = ['-'.join([str(id_i) for id_i in id])
                            for id in data.index]

            data_mod.index = simple_index
            data_mod.index.name = simple_name

        elif axis == 1:
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
        The modified DataFrame
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


# print a matrix in a nice way using a DataFrame
def show_matrix(data, use_describe=False):
    if use_describe:
        pp.pprint(pd.DataFrame(data).describe(
            percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]))
    else:
        pp.pprint(pd.DataFrame(data))


# Monkeypatch warnings to get prettier messages
def _warning(message, category, filename, lineno, file=None, line=None):
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

    if not isinstance(df, (pd.Series, pd.DataFrame)):
        vals = df
        cols = np.arange(vals.shape[1]) if vals.ndim > 1 else 0

        if vals.ndim == 1:
            df = pd.Series(vals, name=cols)
        else:
            df = pd.DataFrame(vals, columns=cols)

    desc = df.describe(percentiles).T

    # add log standard deviation to the stats
    desc.insert(3, "log_std", np.nan)
    desc = desc.T

    for col in desc.columns:
        if np.min(df[col]) > 0.0:
            desc.loc['log_std', col] = np.std(np.log(df[col]))

    return desc


def str2bool(v):
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 'True', 't', 'y', '1'}:
        return True
    if v.lower() in {'no', 'false', 'False', 'f', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# Input specs

CMP_data_path = dict(
    P58='/resources/FEMA_P58_2nd_ed.hdf',
    HAZUS_EQ='/resources/HAZUS_MH_2.1_EQ.hdf',
    HAZUS_HU='/resources/HAZUS_MH_2.1.hdf',
    HAZUS_FL='/resources/HAZUS_MH_2.1_FL.hdf',
    HAZUS_MISC='/resources/HAZUS_MH_2.1_MISC.hdf'
)

POP_data_path = dict(
    P58='/resources/FEMA_P58_2nd_ed.hdf',
    HAZUS_EQ='/resources/HAZUS_MH_2.1_EQ.hdf'
)

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
