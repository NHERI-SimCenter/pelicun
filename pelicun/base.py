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

            if not filepath.is_file():
                raise ValueError(f"The filepath provided does not point to an "
                                 f"valid location: {filepath}")

            globals()['log_file'] = str(filepath)

            with open(filepath, 'w') as f:
                f.write('')

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

            if unit in globals().keys():
                scale_factor = globals()[unit]

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
            elif key == "DemandOffset":
                options.demand_offset = value
            elif key == "NonDirectionalMultipliers":
                options.nondir_multi_dict = value

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

# Constants for unit conversion

# time
sec = 1.

minute = 60. * sec
hour = 60. * minute
day = 24. * hour

sec2 = sec**2.

# distance, area, volume
m = 1.

mm = 0.001 * m
cm = 0.01 * m
km = 1000. * m

inch = 0.0254
ft = 12. * inch
mile = 5280. * ft

# area
m2 = m**2.

mm2 = mm**2.
cm2 = cm**2.
km2 = km**2.

inch2 = inch**2.
ft2 = ft**2.
mile2 = mile**2.

# volume
m3 = m**3.

inch3 = inch**3.
ft3 = ft**3.


# speed / velocity
cmps = cm / sec
mps = m / sec
mph = mile / hour

inchps = inch / sec
ftps = ft / sec

# acceleration
mps2 = m / sec2

inchps2 = inch / sec2
ftps2 = ft / sec2

g = 9.80665 * mps2

# mass
kg = 1.

ton = 1000. * kg

lb = 0.453592 * kg

# force
N = kg * m / sec2

kN = 1e3 * N

lbf = lb * g
kip = 1000. * lbf
kips = kip

# pressure / stress
Pa = N / m2

kPa = 1e3 * Pa
MPa = 1e6 * Pa
GPa = 1e9 * Pa

psi = lbf / inch2
ksi = 1e3 * psi
Mpsi = 1e6 * psi

# misc
A = 1.

V = 1.
kV = 1000. * V

ea = 1.

rad = 1.

C = 1.

# FEMA P58 specific
#TODO: work around these and make them available only in the parser methods
EA = ea
SF = ft2
LF = ft
TN = ton
AP = A
CF = ft3 / minute
KV = kV * A

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