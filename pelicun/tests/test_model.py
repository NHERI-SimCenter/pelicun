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

import tempfile
import pytest
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

#  __  __      _   _               _
# |  \/  | ___| |_| |__   ___   __| |___
# | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
# | |  | |  __/ |_| | | | (_) | (_| \__ \
# |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#
# The following tests verify the methods of the objects of the module.


def test_PelicunModel_init():

    asmt = assessment.Assessment()
    mdl = model.PelicunModel(asmt)
    assert mdl.log_msg
    assert mdl.log_div

def test_PelicunModel_convert_marginal_params():

    asmt = assessment.Assessment()
    mdl = model.PelicunModel(asmt)

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

    assert 'Theta_0' in res.columns
    
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

    # check that the columns are exactly the following, in no specific
    # order.
    assert set(res.columns) == {
        'Theta_0', 'Family', 'Theta_1', 'Theta_2',
        'TruncateLower', 'TruncateUpper'}

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

def create_DemandModel():

    asmt = assessment.Assessment()
    mdl = asmt.demand

    return mdl

def test_DemandModel_init():

    mdl = create_DemandModel()
    assert mdl.log_msg
    assert mdl.log_div

    assert mdl.marginal_params is None
    assert mdl.correlation is None
    assert mdl.empirical_data is None
    assert mdl.units is None
    assert mdl._RVs is None
    assert mdl._sample is None


def DemandModel_load_sample(path):

    # instantiate a DemandModel object
    mdl = create_DemandModel()

    # load the sample from the specified path
    mdl.load_sample(path)

    # return the object
    return mdl


def test_DemandModel_load_sample():

    # get a DemandModel in which the sample has been loaded
    mdl = DemandModel_load_sample('tests/data/model/test_DemandModel_load_sample/demand_sample_A.csv')

    # retrieve the loaded sample and units
    obtained_sample = mdl._sample
    obtained_units = mdl.units

    # compare against the expected values for the sample
    expected_sample = pd.DataFrame(
        [[4.029069, 10.084915, 0.02672, 8.690585],],
        columns=pd.MultiIndex.from_tuples(
            (
                ('PFA', '0', '1'),
                ('PFA', '1', '1'),
                ('PID', '1', '1'),
                ('SA_0.23', '0', '1'),
            ),
            names = ('type', 'loc', 'dir')
        ),
        index=[0]
    )
    pd.testing.assert_frame_equal(expected_sample, obtained_sample)

    # compare against the expected values for the units
    expected_units = pd.Series(
        ('inps2', 'inps2', 'rad', 'inps2'),
        index=pd.MultiIndex.from_tuples(
            (
                ('PFA', '0', '1'),
                ('PFA', '1', '1'),
                ('PID', '1', '1'),
                ('SA_0.23', '0', '1')),
            names=['type', 'loc', 'dir']),
        name='Units'
    )
    pd.testing.assert_series_equal(expected_units, obtained_units)


def test_DemandModel_save_sample():

    # get a DemandModel in which the sample has been loaded
    mdl = DemandModel_load_sample('tests/data/model/test_DemandModel_load_sample/demand_sample_A.csv')

    # instantiate a temporary directory in memory
    temp_dir = tempfile.mkdtemp()
    # save the sample there
    mdl.save_sample(f'{temp_dir}/temp.csv')


    
def get_calibrated_model(path, config):

    # get a DemandModel in which the sample has been loaded
    mdl = DemandModel_load_sample(path)

    # calibrate the model
    mdl.calibrate_model(config)

    # return the model
    return mdl


def test_DemandModel_calibrate_model():

    mdl = get_calibrated_model(
        'tests/data/model/test_DemandModel_load_sample/demand_sample_A.csv',
        {
            "ALL": {
                "DistributionFamily": "lognormal"
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateLower": "",
                "TruncateUpper": "0.06"
            }
        }
    )


def test_DemandModel_save_load_model():

    mdl = get_calibrated_model(
        'tests/data/model/test_DemandModel_load_sample/demand_sample_A.csv',
        {
            "ALL": {
                "DistributionFamily": "lognormal"
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateLower": "",
                "TruncateUpper": "0.06"
            }
        }
    )

    # instantiate a temporary directory in memory
    temp_dir = tempfile.mkdtemp()
    # save the model there
    mdl.save_model(f'{temp_dir}/temp')


    # load the model in another DemandModel
    # note: this currently fails.

    mdl2 = get_calibrated_model(
        'tests/data/model/test_DemandModel_load_sample/demand_sample_A.csv',
        {
            "ALL": {
                "DistributionFamily": "lognormal"
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateLower": "",
                "TruncateUpper": "0.06"
            }
        }
    )
    with pytest.raises(ValueError):
        mdl2.load_model(f'{temp_dir}/temp')


def test_DemandModel_generate_sample():

    mdl = get_calibrated_model(
        'tests/data/model/test_DemandModel_load_sample/demand_sample_A.csv',
        {
            "ALL": {
                "DistributionFamily": "lognormal"
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateLower": "",
                "TruncateUpper": "0.06"
            }
        }
    )

    mdl.generate_sample({
        "SampleSize": 3,
        'PreserveRawOrder': False
        })

    # get the generated demand sample
    res = mdl.save_sample(save_units=True)
    assert res

    obtained_sample, obtained_units = res
    
    # compare against the expected values for the sample
    expected_sample = pd.DataFrame(
        (
            (158.624160, 397.042985, 0.02672, 342.148783),
            (158.624160, 397.042985, 0.02672, 342.148783),
            (158.624160, 397.042985, 0.02672, 342.148783),
        ),
        columns=pd.MultiIndex.from_tuples(
            (
                ('PFA', '0', '1'),
                ('PFA', '1', '1'),
                ('PID', '1', '1'),
                ('SA_0.23', '0', '1'),
            ),
            names = ('type', 'loc', 'dir')
        ),
        index=pd.Index((0, 1, 2), dtype='object')
    )
    pd.testing.assert_frame_equal(expected_sample, obtained_sample)

    # compare against the expected values for the units
    expected_units = pd.Series(
        ('inps2', 'inps2', 'rad', 'inps2'),
        index=pd.MultiIndex.from_tuples(
            (
                ('PFA', '0', '1'),
                ('PFA', '1', '1'),
                ('PID', '1', '1'),
                ('SA_0.23', '0', '1')),
            names=('type', 'loc', 'dir')),
        name='Units'
    )
    pd.testing.assert_series_equal(expected_units, obtained_units)


def create_AssetModel():

    asmt = assessment.Assessment()
    mdl = asmt.asset

    return mdl, asmt

def test_AssetModel_init():

    mdl, _ = create_AssetModel()
    assert mdl.log_msg
    assert mdl.log_div

    assert mdl.cmp_marginal_params is None
    assert mdl.cmp_units is None
    assert mdl._cmp_RVs is None
    assert mdl._cmp_sample is None

def test_AssetModel_load_cmp_model():

    mdl, asmt = create_AssetModel()
    cmp_marginals = pd.read_csv(
        'tests/data/model/test_AssetModel_load_cmp_model/CMP_marginals.csv', index_col=0)
    asmt.stories = 4
    mdl.load_cmp_model({'marginals': cmp_marginals})
    mdl.generate_cmp_sample(sample_size=10)

    res = mdl.save_cmp_sample()


#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.

