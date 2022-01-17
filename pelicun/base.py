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

"""

import os, sys, time
import warnings
from datetime import datetime
from time import strftime
from pathlib import Path
import argparse

from copy import deepcopy

# import libraries for other modules
import numpy as np
import pandas as pd

from .__init__ import __version__ as pelicun_version

idx = pd.IndexSlice

# set printing options
import pprint
pp = pprint.PrettyPrinter(indent=2, width=80-24)

pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = True
pd.options.display.width = 300

idx = pd.IndexSlice

class Options(object):

    """

    Parameters

    verbose: boolean
        If True, the pelicun echoes more information throughout the assessment.
        This can be useful for debugging purposes.

    log_show_ms: boolean
        If True, the timestamps in the log file are in microsecond precision.

    """

    def __init__(self):

        self._verbose = False
        self._log_show_ms = False
        self._print_log = False

        self.defaults = None
        self.sampling_method = None

        self._seed = None
        self._rng = np.random.default_rng()

        self.reset_log_strings()

        self.demand_offset = {}
        self.nondir_multi_dict = {}

    def nondir_multi(self, EDP_type):

        if EDP_type in self.nondir_multi_dict.keys():
            return self.nondir_multi_dict[EDP_type]

        elif 'ALL' in self.nondir_multi_dict.keys():
            return self.nondir_multi_dict['ALL']

        else:
            raise ValueError(f"Scale factor for non-directional demand "
                             f"calculation of {EDP_type} not specified.")

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
    def log_file(self):
        return globals()['log_file']

    @log_file.setter
    def log_file(self, value):

        if value is None:
            globals()['log_file'] = value
        else:

            filepath = Path(value).resolve()

            try:
                globals()['log_file'] = str(filepath)

                with open(filepath, 'w') as f:
                    f.write('')

            except:
                raise ValueError(f"The filepath provided does not point to an "
                                 f"valid location: {filepath}")

    @property
    def print_log(self):
        return self._print_log

    @print_log.setter
    def print_log(self, value):
        self._print_log = str2bool(value)

    def reset_log_strings(self):

        if self._log_show_ms:
            self._log_time_format = '%H:%M:%S:%f'
            self._log_pref = ' ' * 16 # the length of the time string in the log file
            self._log_div = '-' * (80 - 17) # to have a total length of 80 with the time added
        else:
            self._log_time_format = '%H:%M:%S'
            self._log_pref = ' ' * 9
            self._log_div = '-' * (80 - 10)

    def scale_factor(self, unit):

        if unit is not None:

            if unit in UC.keys():
                scale_factor = UC[unit]

            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            scale_factor = 1.0

        return scale_factor

options = Options()

log_file = None

# get the absolute path of the pelicun directory
pelicun_path = Path(os.path.dirname(os.path.abspath(__file__)))

def set_options(config_options):

    if config_options is not None:

        for key, value in config_options.items():

            if key == "Verbose":
                options.verbose = value
            elif key == "Seed":
                options.seed = value
            elif key == "LogShowMS":
                options.log_show_ms = value
            elif key == "LogFile":
                options.log_file = value
            elif key == "PrintLog":
                options.print_log = value
            elif key == "SamplingMethod":
                options.sampling_method = value
            elif key == "DemandOffset":
                options.demand_offset = value
            elif key == "NonDirectionalMultipliers":
                options.nondir_multi_dict = value
            elif key == "RepairCostAndTimeCorrelation":
                options.rho_cost_time = value
            elif key == "EconomiesOfScale":
                options.eco_scale = value

def convert_to_SimpleIndex(data, axis=0):
    """
    Converts the index of a DataFrame to a simple, one-level index

    The target index uses standard SimCenter convention to identify different
    levels: a dash character ('-') is used to separate each level of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int
        Identifies if the index (0) or the columns (1) shall be edited.

    Returns
    -------
    data: DataFrame
        The modified DataFrame
    """



    if axis == 0:
        simple_index = ['-'.join([str(id_i) for id_i in id])
                        for id in data.index]
        data.index = simple_index

    elif axis == 1:
        simple_index = ['-'.join([str(id_i) for id_i in id])
                        for id in data.columns]
        data.columns = simple_index

    else:
        raise ValueError(f"Invalid axis parameter: {axis}")

    return data

def convert_to_MultiIndex(data, axis=0):
    """
    Converts the index of a DataFrame to a MultiIndex

    We assume that the index uses standard SimCenter convention to identify
    different levels: a dash character ('-') is expected to separate each level
    of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int
        Identifies if the index (0) or the columns (1) shall be edited.

    Returns
    -------
    data: DataFrame
        The modified DataFrame
    """

    if axis == 0:
        index_labels = [label.split('-') for label in data.index]

    elif axis == 1:
        index_labels = [label.split('-') for label in data.columns]

    else:
        raise ValueError(f"Invalid axis parameter: {axis}")

    max_lbl_len = np.max([len(labels) for labels in index_labels])

    for l_i, labels in enumerate(index_labels):

        if len(labels) != max_lbl_len:
            labels += ['', ] * (max_lbl_len - len(labels))
            index_labels[l_i] = labels

    index_labels = np.array(index_labels)

    if index_labels.shape[1] > 1:

        if axis == 0:
            data.index = pd.MultiIndex.from_arrays(index_labels.T)

        else:
            data.columns = pd.MultiIndex.from_arrays(index_labels.T)

    return data

def convert_unit(value, unit):
    """
    Convert value(s) provided in one unit to the internal SI unit

    Parameters
    ----------
    value: str, float, or int
        A single number or an array of numbers provided as a string following
        SimCenter's notation.
    unit: str
        Either a unit name, or a quantity and a unit name separated by a space.
        For example: 'ft' or '100 ft'.
    """

    # if the value is NaN, just return it
    if pd.isna(value):
        return value

    # start by understanding the unit
    unit = unit.strip().split(' ')
    if len(unit) > 1:
        unit_count, unit_name = unit
        unit_count = float(unit_count)
    else:
        unit_count = 1
        unit_name = unit

    try:
        if unit_name in UC_fema.keys():
            unit_factor = unit_count * UC_fema[unit_name]
        else:
            unit_factor = unit_count * UC[unit_name]
    except:
        raise ValueError(f"Specified unit not recognized: "
                         f"{unit_count} {unit_name}")

    # now parse the value
    try:
        float(value)
        is_float = True
    except:
        is_float = False

    # if it is a single scalar, conversion is easy
    if is_float:
        return float(value) * unit_factor

    # otherwise, we assume it is a string using SimCenter array notation
    else:
        values = [val.split(',') for val in value.split('|')]

        # the first set of values are assumed to be outputs and need to
        # be normalized by the unit_factor
        values[0] = np.array(values[0], dtype=float) / unit_factor

        # the second set of values are assumed to be inputs and need to
        # be scaled by the unit factor
        values[1] = np.array(values[1], dtype=float) * unit_factor

        # now convert the values back to string format
        return '|'.join([','.join([f'{val:g}' for val in values[i]])
                         for i in range(2)])

# print a matrix in a nice way using a DataFrame
def show_matrix(data, describe=False):
    if describe:
        pp.pprint(pd.DataFrame(data).describe(percentiles=[0.01,0.1,0.5,0.9,0.99]))
    else:
        pp.pprint(pd.DataFrame(data))

# Monkeypatch warnings to get prettier messages
def _warning(message, category, filename, lineno, file=None, line=None):

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

    print('WARNING in {} at line {}\n{}\n'.format(python_file, lineno, message))

warnings.showwarning = _warning

def show_warning(warning_msg):
    warnings.warn(UserWarning(warning_msg))

def print_system_info():

    log_msg('System Information:',
            prepend_timestamp=False, prepend_blank_space=False)
    log_msg(f'local time zone: {datetime.utcnow().astimezone().tzinfo}\n'
            f'start time: {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}\n'
            f'python: {sys.version}\n'
            f'numpy: {np.__version__}\n'
            f'pandas: {pd.__version__}\n',
            prepend_timestamp=False)

def log_div(prepend_timestamp=False):
    """
    Print a divider line to the log file

    """

    if prepend_timestamp:
        msg = options.log_div

    else:
        msg = '-' * 80

    log_msg(msg, prepend_timestamp = prepend_timestamp)


def log_msg(msg='', prepend_timestamp=True, prepend_blank_space=True):
    """
    Print a message to the screen with the current time as prefix

    The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

    Parameters
    ----------
    msg: string
       Message to print.

    """

    msg_lines = msg.split('\n')

    for msg_i, msg_line in enumerate(msg_lines):

        if (prepend_timestamp and (msg_i==0)):
            formatted_msg = '{} {}'.format(
                datetime.now().strftime(options.log_time_format), msg_line)
        elif prepend_timestamp:
            formatted_msg = options.log_pref + msg_line
        elif prepend_blank_space:
            formatted_msg = options.log_pref + msg_line
        else:
            formatted_msg = msg_line

        if options.print_log:
            print(formatted_msg)

        if globals()['log_file'] is not None:
            with open(globals()['log_file'], 'a') as f:
                f.write('\n'+formatted_msg)

def describe(df):

    if isinstance(df, (pd.Series, pd.DataFrame)):
        vals = df.values
        if isinstance(df, pd.DataFrame):
            cols = df.columns
        elif df.name is not None:
            cols = df.name
        else:
            cols = 0
    else:
        vals = df
        cols = np.arange(vals.shape[1]) if vals.ndim > 1 else 0

    if vals.ndim == 1:
        df_10, df_50, df_90 = np.nanpercentile(vals, [10, 50, 90])
        desc = pd.Series({
            'count': np.sum(~np.isnan(vals)),
            'mean': np.nanmean(vals),
            'std': np.nanstd(vals),
            'min': np.nanmin(vals),
            '10%': df_10,
            '50%': df_50,
            '90%': df_90,
            'max': np.nanmax(vals),
        }, name=cols)
    else:
        df_10, df_50, df_90 = np.nanpercentile(vals, [10, 50, 90], axis=0)
        desc = pd.DataFrame({
            'count': np.sum(~np.isnan(vals), axis=0),
            'mean': np.nanmean(vals, axis=0),
            'std': np.nanstd(vals, axis=0),
            'min': np.nanmin(vals, axis=0),
            '10%': df_10,
            '50%': df_50,
            '90%': df_90,
            'max': np.nanmax(vals, axis=0),
        }, index=cols).T

    return desc

def str2bool(v):
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Constants for Unit Conversion (UC)

UC = {}  # initialize

# time
UC['sec'] = 1.

UC['minute'] = 60. * UC['sec']
UC['hour'] = 60. * UC['minute']
UC['day'] = 24. * UC['hour']

UC['sec2'] = UC['sec']**2.

# distance, area, volume
UC['m'] = 1.

UC['mm'] = 0.001 * UC['m']
UC['cm'] = 0.01 * UC['m']
UC['km'] = 1000. * UC['m']

UC['inch'] = 0.0254
UC['ft'] = 12. * UC['inch']
UC['mile'] = 5280. * UC['ft']

# area
UC['m2'] = UC['m']**2.

UC['mm2'] = UC['mm']**2.
UC['cm2'] = UC['cm']**2.
UC['km2'] = UC['km']**2.

UC['inch2'] = UC['inch']**2.
UC['ft2'] = UC['ft']**2.
UC['mile2'] = UC['mile']**2.

# volume
UC['m3'] = UC['m']**3.

UC['inch3'] = UC['inch']**3.
UC['ft3'] = UC['ft']**3.


# speed / velocity
UC['cmps'] = UC['cm'] / UC['sec']
UC['mps'] = UC['m'] / UC['sec']
UC['mph'] = UC['mile'] / UC['hour']

UC['inchps'] = UC['inch'] / UC['sec']
UC['ftps'] = UC['ft'] / UC['sec']

# acceleration
UC['mps2'] = UC['m'] / UC['sec2']

UC['inchps2'] = UC['inch'] / UC['sec2']
UC['ftps2'] = UC['ft'] / UC['sec2']

UC['g'] = 9.80665 * UC['mps2']

# mass
UC['kg'] = 1.

UC['ton'] = 1000. * UC['kg']

UC['lb'] = 0.453592 * UC['kg']

# force
UC['N'] = UC['kg'] * UC['m'] / UC['sec2']

UC['kN'] = 1e3 * UC['N']

UC['lbf'] = UC['lb'] * UC['g']
UC['kip'] = 1000. * UC['lbf']
UC['kips'] = UC['kip']

# pressure / stress
UC['Pa'] = UC['N']/ UC['m2']

UC['kPa'] = 1e3 * UC['Pa']
UC['MPa'] = 1e6 * UC['Pa']
UC['GPa'] = 1e9 * UC['Pa']

UC['psi'] = UC['lbf'] / UC['inch2']
UC['ksi'] = 1e3 * UC['psi']
UC['Mpsi'] = 1e6 * UC['psi']

# misc
UC['A'] = 1.

UC['V'] = 1.
UC['kV'] = 1000. * UC['V']

UC['ea'] = 1.

UC['rad'] = 1.

UC['C'] = 1.

# FEMA P58 specific
#TODO: work around these and make them available only in the parser methods
UC_fema = {}  # initialize
UC_fema['EA'] = UC['ea']
UC_fema['SF'] = UC['ft2']
UC_fema['LF'] = UC['ft']
UC_fema['TN'] = UC['ton']
UC_fema['AP'] = UC['A']
UC_fema['CF'] = UC['ft3'] / UC['minute']
UC_fema['KV'] = UC['kV'] * UC['A']

# Input specs

CMP_data_path = dict(
    P58      = '/resources/FEMA_P58_2nd_ed.hdf',
    HAZUS_EQ = '/resources/HAZUS_MH_2.1_EQ.hdf',
    HAZUS_HU = '/resources/HAZUS_MH_2.1.hdf',
    HAZUS_FL = '/resources/HAZUS_MH_2.1_FL.hdf',
    HAZUS_MISC = '/resources/HAZUS_MH_2.1_MISC.hdf'
)

POP_data_path = dict(
    P58      = '/resources/FEMA_P58_2nd_ed.hdf',
    HAZUS_EQ = '/resources/HAZUS_MH_2.1_EQ.hdf'
)

default_units = dict(
    force =        'N',
    length =       'm',
    area =         'm2',
    volume =       'm3',
    speed =        'mps',
    acceleration = 'mps2',
)

EDP_units = dict(
    # PID, PRD, RID, and MID are not here because they are unitless
    PFA = 'acceleration',
    PWS = 'speed',
    PGA = 'acceleration',
    SA = 'acceleration',
    SV = 'speed',
    SD = 'length',
    PIH = 'length'
)

EDP_to_demand_type = {
    'Story Drift Ratio' :             'PID',
    'Peak Interstory Drift Ratio':    'PID',
    'Roof Drift Ratio' :              'PRD',
    'Peak Roof Drift Ratio' :         'PRD',
    'Damageable Wall Drift' :         'DWD',
    'Racking Drift Ratio' :           'RDR',
    'Peak Floor Acceleration' :       'PFA',
    'Peak Floor Velocity' :           'PFV',
    'Peak Gust Wind Speed' :          'PWS',
    'Peak Inundation Height' :        'PIH',
    'Peak Ground Acceleration' :      'PGA',
    'Peak Ground Velocity' :          'PGV',
    'Spectral Acceleration' :         'SA',
    'Spectral Velocity' :             'SV',
    'Spectral Displacement' :         'SD',
    'Peak Spectral Acceleration' :    'SA',
    'Peak Spectral Velocity' :        'SV',
    'Peak Spectral Displacement' :    'SD',
    'Permanent Ground Deformation' :  'PGD',
    'Mega Drift Ratio' :              'PMD',
    'Residual Drift Ratio' :          'RID',
    'Residual Interstory Drift Ratio':'RID',
}

# PFA in FEMA P58 corresponds to the top of the given story. The ground floor
# has an index of 0. When damage of acceleration-sensitive components
# is controlled by the acceleration of the bottom of the story, the
# corresponding PFA location needs to be reduced by 1. The SimCenter framework
# assumes that PFA corresponds to the bottom of the given story
# by default, hence, we would need to subtract 1 from the location values.
# Rather than changing the locations themselves, we assign an offset of -1
# so that the results still get collected at the appropriate story.
EDP_offset_adjustment = dict(
    PFA = -1,
    PFV = -1
)
