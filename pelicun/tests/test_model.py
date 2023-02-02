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
These are unit and integration tests on the model module of pelicun.
"""

import numpy as np
import pandas as pd
from pelicun import base
from pelicun import model
from pelicun import assessment

# for tests, we sometimes create things or call them just to see if
# things would work, so the following are irrelevant:

# pylint: disable=useless-suppression
# pylint: disable=unused-variable
# pylint: disable=pointless-statement

# The tests maintain the order of definitions of the `model.py` file.

#  __  __      _   _               _
# |  \/  | ___| |_| |__   ___   __| |___
# | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
# | |  | |  __/ |_| | | | (_) | (_| \__ \
# |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#
# The following tests verify the methods of the objects of the module.

def create_PelicunModel():

    asmt = assessment.Assessment()
    mdl = model.PelicunModel(asmt)

    return mdl

def test_PelicunModel_init():

    mdl = create_PelicunModel()
    assert mdl.log_msg
    assert mdl.log_div

def test_PelicunModel_convert_marginal_params():

    mdl = create_PelicunModel()

    # one row, only Theta_0, no conversion
    marginal_params = pd.DataFrame(
        [['1.0']],
        columns=['Theta_0'],
        index=['A']
    )
    units = pd.Series(
        ['ea'],
        index=['A']
    )
    arg_units = None
    res = mdl.convert_marginal_params(
        marginal_params, units, arg_units)

    # res:
    # Theta_0
    # A     1.0

    # check that the columns are exactly the following, in no specific
    # order.
    assert set(res.columns) == {
        'Theta_0', 'Family', 'Theta_1', 'Theta_2',
        'TruncateLower', 'TruncateUpper'}
    
    # many rows, with conversions
    marginal_params = pd.DataFrame(
        [[np.nan, 1.0, np.nan, np.nan, np.nan, np.nan],
         ['normal', 0.0, 1.0, np.nan, -0.50, 0.50],
         ['lognormal', 1.0, 0.5, np.nan, 0.50, 1.50],
         ['uniform', 0.0, 10.0, np.nan, np.nan, np.nan],
         ],
        columns=['Family', 'Theta_0', 'Theta_1', 'Theta_2',
                 'TruncateLower', 'TruncateUpper'],
        index=['A', 'B', 'C', 'D']
    )
    units = pd.Series(
        ['ea', 'ft', 'in', 'in2'],
        index=['A', 'B', 'C', 'D']
    )
    arg_units = None
    res = mdl.convert_marginal_params(
        marginal_params, units, arg_units)

    # res:
    #       Family  Theta_0   Theta_1  Theta_2  TruncateLower  TruncateUpper
    # A        NaN   1.0000       NaN      NaN            NaN            NaN
    # B     normal   0.0000  1.000000      NaN        -0.1524         0.1524
    # C  lognormal   0.0254  0.500000      NaN         0.0127         0.0381
    # D    uniform   0.0000  0.006452      NaN            NaN            NaN

    expected_df = pd.DataFrame({
      'Family': [np.nan, 'normal', 'lognormal', 'uniform'],
      'Theta_0': [1.0000, 0.0000, 0.0254, 0.0000],
      'Theta_1': [np.nan, 1.000000, 0.500000, 0.0064516],
      'Theta_2': [np.nan, np.nan, np.nan, np.nan],
      'TruncateLower': [np.nan, -0.1524, 0.0127, np.nan],
      'TruncateUpper': [np.nan, 0.1524, 0.0381, np.nan]
    }, index=['A', 'B', 'C', 'D'])

    pd.testing.assert_frame_equal(expected_df, res)



#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.

