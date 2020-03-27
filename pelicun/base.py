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

# import libraries for other modules
import numpy as np
import pandas as pd

idx = pd.IndexSlice

# set printing options
import pprint
pp = pprint.PrettyPrinter(indent=4, width=300)

pd.options.display.max_rows = 20
pd.options.display.max_columns = None
pd.options.display.expand_frame_repr = True
pd.options.display.width = 300

log_file = None

log_div = '-' * (80-21)  # 21 to have a total length of 80 with the time added

# get the absolute path of the pelicun directory
pelicun_path = os.path.dirname(os.path.abspath(__file__))

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
    python_file = '/'.join(file_path[-3:])
    print('WARNING in {} at line {}\n{}\n'.format(python_file, lineno, message))
warnings.showwarning = _warning

def show_warning(warning_msg):
    warnings.warn(UserWarning(warning_msg))

def set_log_file(filepath):
    globals()['log_file'] = filepath
    with open(filepath, 'w') as f:
        f.write('pelicun\n')

def log_msg(msg='', prepend_timestamp=True):
    """
    Print a message to the screen with the current time as prefix

    The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

    Parameters
    ----------
    msg: string
       Message to print.

    """
    if prepend_timestamp:
        formatted_msg = '{} {}'.format(datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S:%fZ')[:-4], msg)
    else:
        formatted_msg = msg

    #print(formatted_msg)

    if globals()['log_file'] is not None:
        with open(globals()['log_file'], 'a') as f:
            f.write('\n'+formatted_msg)

def describe(df):

    if isinstance(df, (pd.Series, pd.DataFrame)):
        vals = df.values
        cols = df.columns
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

# Constants for unit conversion

# time
sec = 1.

minute = 60. * sec
h = 60. * minute
day = 24. * h

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
mph = mile / h

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

