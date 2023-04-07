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
         ['normal', np.nan, 1.0, np.nan, -0.50, 0.50],
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
    # B     normal   NaN     1.000000      NaN        -0.1524         0.1524
    # C  lognormal   0.0254  0.500000      NaN         0.0127         0.0381
    # D    uniform   0.0000  0.006452      NaN            NaN            NaN

    expected_df = pd.DataFrame(
        {
            'Family': [np.nan, 'normal', 'lognormal', 'uniform'],
            'Theta_0': [1.0000, np.nan, 0.0254, 0.0000],
            'Theta_1': [np.nan, 1.000000, 0.500000, 0.0064516],
            'Theta_2': [np.nan, np.nan, np.nan, np.nan],
            'TruncateLower': [np.nan, -0.50, 0.0127, np.nan],
            'TruncateUpper': [np.nan, 0.50, 0.0381, np.nan]
        },
        index=['A', 'B', 'C', 'D'])

    pd.testing.assert_frame_equal(expected_df, res)


def test_DemandModel_init():

    asmt = assessment.Assessment()
    mdl = asmt.demand
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
    asmt = assessment.Assessment()
    mdl = asmt.demand

    # load the sample from the specified path
    mdl.load_sample(path)

    # return the object
    return mdl


def test_DemandModel_load_sample():

    # get a DemandModel in which the sample has been loaded
    mdl = DemandModel_load_sample(
        'tests/data/model/test_DemandModel_'
        'load_sample/demand_sample_A.csv')

    # retrieve the loaded sample and units
    obtained_sample = mdl._sample
    obtained_units = mdl.units

    # compare against the expected values for the sample
    expected_sample = pd.DataFrame(
        [[4.029069, 10.084915, 0.02672, 8.690585], ],
        columns=pd.MultiIndex.from_tuples(
            (
                ('PFA', '0', '1'),
                ('PFA', '1', '1'),
                ('PID', '1', '1'),
                ('SA_0.23', '0', '1'),
            ),
            names=('type', 'loc', 'dir')
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
    mdl = DemandModel_load_sample(
        'tests/data/model/test_DemandModel_'
        'load_sample/demand_sample_A.csv')

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

    assert mdl is not None


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
            names=('type', 'loc', 'dir')
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


def test_AssetModel_init():

    asmt = assessment.Assessment()
    mdl = asmt.asset
    assert mdl.log_msg
    assert mdl.log_div

    assert mdl.cmp_marginal_params is None
    assert mdl.cmp_units is None
    assert mdl._cmp_RVs is None
    assert mdl._cmp_sample is None


def test_AssetModel_load_cmp_model():

    asmt = assessment.Assessment()
    mdl = asmt.asset
    cmp_marginals = pd.read_csv(
        'tests/data/model/test_AssetModel/CMP_marginals.csv', index_col=0)
    mdl.load_cmp_model({'marginals': cmp_marginals})

    expected_cmp_marginal_params = pd.DataFrame(
        {
            'Theta_0': (8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
            'Blocks': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        },
        index=pd.MultiIndex.from_tuples(
            (
                ('component_a', '0', '1'),
                ('component_a', '0', '2'),
                ('component_a', '1', '1'),
                ('component_a', '1', '2'),
                ('component_a', '2', '1'),
                ('component_a', '2', '2')
            ),
            names=('cmp', 'loc', 'dir')))

    pd.testing.assert_frame_equal(
        expected_cmp_marginal_params,
        mdl.cmp_marginal_params)

    expected_cmp_units = pd.Series(
        data=['ea'], index=['component_a'],
        name='Units')

    pd.testing.assert_series_equal(expected_cmp_units, mdl.cmp_units)


def test_AssetModel_generate_cmp_sample():

    asmt = assessment.Assessment()
    mdl = asmt.asset

    mdl.cmp_marginal_params = pd.DataFrame(
        {
            'Theta_0': (8.0, 8.0, 8.0, 8.0),
            'Blocks': (1.0, 1.0, 1.0, 1.0)
        },
        index=pd.MultiIndex.from_tuples(
            (
                ('component_a', '1', '1'),
                ('component_a', '1', '2'),
                ('component_a', '2', '1'),
                ('component_a', '2', '2')
            ),
            names=('cmp', 'loc', 'dir')))

    mdl.cmp_units = pd.Series(
        data=['ea'], index=['component_a'],
        name='Units')

    mdl.generate_cmp_sample(sample_size=10)

    assert mdl._cmp_RVs is not None

    expected_cmp_sample = pd.DataFrame(
        {
            ('component_a', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('component_a', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    pd.testing.assert_frame_equal(
        expected_cmp_sample,
        mdl.cmp_sample)


def test_AssetModel_save_cmp_sample():

    asmt = assessment.Assessment()
    mdl = asmt.asset

    mdl._cmp_sample = pd.DataFrame(
        {
            ('component_a', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('component_a', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    mdl.cmp_units = pd.Series(
        data=['ea'], index=['component_a'],
        name='Units')

    res = mdl.save_cmp_sample()
    assert isinstance(res, pd.DataFrame)

    temp_dir = tempfile.mkdtemp()
    # save the sample there
    mdl.save_cmp_sample(f'{temp_dir}/temp.csv')

    # load the component sample to a different AssetModel
    asmt = assessment.Assessment()
    mdl = asmt.asset
    mdl.load_cmp_sample(f'{temp_dir}/temp.csv')


def test_DamageModel_init():

    asmt = assessment.Assessment()
    mdl = asmt.damage
    assert mdl.log_msg
    assert mdl.log_div

    assert mdl.damage_params is None
    assert mdl._dmg_function_scale_factors is None
    assert mdl._sample is None


def test_DamageModel_load_damage_model():

    asmt = assessment.Assessment()
    mdl = asmt.damage

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asmt.asset._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    mdl.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])

    # should no longer be None
    assert mdl.damage_params is not None

    # defining this dataframe in a proper way was kind of hard, so we
    # resort to comparing it against its string version for now.
    # Note that this shouldn't be the standard way of asserting
    # dataframes for equality.
    # See other tests for the proper way.

    # oops. it fails on Windows.
    # assert mdl.damage_params.to_string() == (
    #     '                 Demand                                          '
    #     'Incomplete                  LS1                                  '
    #     '         LS2                                           LS3       '
    #     '                                    LS4                       \n '
    #     '           Directional Offset                         Type Unit  '
    #     '            DamageStateWeights     Family Theta_0 Theta_1 DamageS'
    #     'tateWeights     Family Theta_0 Theta_1 DamageStateWeights     Fam'
    #     'ily Theta_0 Theta_1 DamageStateWeights Family Theta_0 Theta_1\nID'
    #     '                                                                         '
    #     '                                                                         '
    #     '                                                                 '
    #     '                                            \nB.10.31.001        '
    #     ' 1.0    0.0  Peak Interstory Drift Ratio   ea        0.0  0.95000'
    #     '0 | 0.050000  lognormal    0.04     0.4               None  logno'
    #     'rmal    0.08     0.4               None  lognormal    0.11     0.'
    #     '4                NaN   None     NaN     NaN')

    # TODO fix this assertion


def test_DamageModel_get_pg_batches():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    asset_model = asmt.asset

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asset_model._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    damage_model.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])

    # make sure that the method works for different batch sizes
    for i in (1, 4, 8, 10, 100):
        damage_model._get_pg_batches(block_batch_size=i)

    # verify the result is correct for certain cases
    res = damage_model._get_pg_batches(block_batch_size=1)
    expected_res = pd.DataFrame(
        np.array((1, 1, 1, 1)),
        index=pd.MultiIndex.from_tuples(
            (
                (1, 'B.10.31.001', '1', '1'),
                (2, 'B.10.31.001', '1', '2'),
                (3, 'B.10.31.001', '2', '1'),
                (4, 'B.10.31.001', '2', '2')),
            names=('Batch', 'cmp', 'loc', 'dir')
        ),
        columns=('Blocks', )
    ).astype('Int64')

    pd.testing.assert_frame_equal(
        expected_res,
        res)

    res = damage_model._get_pg_batches(block_batch_size=1000)
    expected_res = pd.DataFrame(
        np.array((1, 1, 1, 1)),
        index=pd.MultiIndex.from_tuples(
            (
                (1, 'B.10.31.001', '1', '1'),
                (1, 'B.10.31.001', '1', '2'),
                (1, 'B.10.31.001', '2', '1'),
                (1, 'B.10.31.001', '2', '2')),
            names=('Batch', 'cmp', 'loc', 'dir')
        ),
        columns=('Blocks', )
    ).astype('Int64')

    pd.testing.assert_frame_equal(
        expected_res,
        res)


def test_DamageModel_create_dmg_RVs():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    asset_model = asmt.asset

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asset_model._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    damage_model.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])
    pg_batch = damage_model._get_pg_batches(block_batch_size=1)
    batches = pg_batch.index.get_level_values(0).unique()
    for PGB_i in batches:
        PGB = pg_batch.loc[PGB_i]
        # ensure the following works in each case
        damage_model._create_dmg_RVs(PGB)

    # check the output for a single case
    PGB_i = batches[-1]
    PGB = pg_batch.loc[PGB_i]

    capacity_RV_reg, lsds_RV_reg = damage_model._create_dmg_RVs(PGB)

    assert capacity_RV_reg is not None
    assert lsds_RV_reg is not None

    assert (
        list(capacity_RV_reg._variables.keys()) == [
            'FRG-B.10.31.001-2-2-1-1', 'FRG-B.10.31.001-2-2-1-2',
            'FRG-B.10.31.001-2-2-1-3'])

    assert not capacity_RV_reg._sets

    assert (
        list(lsds_RV_reg._variables.keys()) == [
            'LSDS-B.10.31.001-2-2-1-1', 'LSDS-B.10.31.001-2-2-1-2',
            'LSDS-B.10.31.001-2-2-1-3'])

    assert not lsds_RV_reg._sets


def test_DamageModel_generate_dmg_sample():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    asset_model = asmt.asset

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asset_model._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    damage_model.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])
    pg_batch = damage_model._get_pg_batches(block_batch_size=1)
    batches = pg_batch.index.get_level_values(0).unique()
    PGB_i = batches[-1]
    PGB = pg_batch.loc[PGB_i]
    sample_size = 10

    # test the _generate_dmg_sample method
    capacity_sample, lsds_sample = damage_model._generate_dmg_sample(
        sample_size, PGB)

    # run a few checks on the results of the method

    # note: the method generates random results. We avoid checking
    # those for equality, because subsequent changes in the code might
    # break the tests. The functionality of the uq module, which is
    # used to generate the random samples, is tested with a dedicated
    # test suite.

    for res in (capacity_sample, lsds_sample):
        assert res.shape == (10, 3)

        assert list(res.columns) == [
            ('B.10.31.001', '2', '2', '1', '1'),
            ('B.10.31.001', '2', '2', '1', '2'),
            ('B.10.31.001', '2', '2', '1', '3')]

        assert (
            list(res.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert capacity_sample.to_numpy().dtype == np.dtype('float64')
    assert lsds_sample.to_numpy().dtype == np.dtype('int64')


def test_DamageModel_get_required_demand_type():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    asset_model = asmt.asset

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asset_model._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    damage_model.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])

    pg_batch = damage_model._get_pg_batches(block_batch_size=1)
    batches = pg_batch.index.get_level_values(0).unique()
    PGB_i = batches[-1]
    PGB = pg_batch.loc[PGB_i]

    EDP_req = damage_model._get_required_demand_type(PGB)

    assert EDP_req == {'PID-2-2': [('B.10.31.001', '2', '2')]}


def test_DamageModel_assemble_required_demand_data():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    demand_model = asmt.demand
    asset_model = asmt.asset

    demand_model.load_sample(
        'tests/data/model/'
        'test_DamageModel_assemble_'
        'required_demand_data/demand_sample.csv')

    # calibrate the model
    demand_model.calibrate_model(
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

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asset_model._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    damage_model.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])

    pg_batch = damage_model._get_pg_batches(block_batch_size=1)
    batches = pg_batch.index.get_level_values(0).unique()

    expected_demand_dicts = [
        {'PID-1-1': np.array([0.02672])},
        {'PID-1-2': np.array([0.02672])},
        {'PID-2-1': np.array([0.02672])},
        {'PID-2-2': np.array([0.02672])}
    ]

    for i, PGB_i in enumerate(batches):
        PGB = pg_batch.loc[PGB_i]
        EDP_req = damage_model._get_required_demand_type(PGB)
        demand_dict = damage_model._assemble_required_demand_data(EDP_req)
        assert demand_dict == expected_demand_dicts[i]


def test_DamageModel_evaluate_damage_state_and_prepare_dmg_quantities():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    demand_model = asmt.demand
    asset_model = asmt.asset

    demand_model.load_sample(
        'tests/data/model/'
        'test_DamageModel_assemble_'
        'required_demand_data/demand_sample.csv')

    # calibrate the model
    demand_model.calibrate_model(
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

    asmt.get_default_data('fragility_DB_FEMA_P58_2nd')

    asset_model._cmp_sample = pd.DataFrame(
        {
            ('B.10.31.001', f'{i}', f'{j}'): 8.0
            for i in range(1, 3) for j in range(1, 3)
        },
        index=range(10),
        columns=pd.MultiIndex.from_tuples(
            (
                ('B.10.31.001', f'{i}', f'{j}')
                for i in range(1, 3) for j in range(1, 3)
            ),
            names=('cmp', 'loc', 'dir')
        )
    )

    damage_model.load_damage_model(['PelicunDefault/fragility_DB_FEMA_P58_2nd.csv'])

    pg_batch = damage_model._get_pg_batches(block_batch_size=1)
    batches = pg_batch.index.get_level_values(0).unique()

    PGB_i = batches[-1]
    PGB = pg_batch.loc[PGB_i]
    EDP_req = damage_model._get_required_demand_type(PGB)
    demand_dict = damage_model._assemble_required_demand_data(EDP_req)

    sample_size = 10
    capacity_sample, lsds_sample = damage_model._generate_dmg_sample(
        sample_size, PGB)

    ds_sample = damage_model._evaluate_damage_state(
        demand_dict, EDP_req,
        capacity_sample, lsds_sample)

    qnt_sample = damage_model._prepare_dmg_quantities(
        PGB, ds_sample,
        dropzero=False,
        dropempty=False)

    # note: the realized number of damage states is random, limiting
    # our assertions
    assert ds_sample.shape[0] == 10
    assert qnt_sample.shape[0] == 10
    assert (
        list(qnt_sample.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert (
        list(ds_sample.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert list(ds_sample.columns)[0] == ('B.10.31.001', '2', '2', '1')
    assert list(qnt_sample.columns)[0] == ('B.10.31.001', '2', '2', '0')


def test_DamageModel_perform_dmg_task():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    demand_model = asmt.demand
    asset_model = asmt.asset

    data = [
        ['rad', 1e-11],
        ['rad', 1e11],
    ]

    index = pd.MultiIndex.from_tuples(
        (
            ('PID', '1', '1'),
            ('PID', '1', '2')
        ),
        names=['type', 'loc', 'dir']
    )

    demand_marginals = pd.DataFrame(
        data, index,
        columns=['Units', 'Theta_0']
    )

    demand_model.load_model({'marginals': demand_marginals})

    sample_size = 10
    demand_model.generate_sample({"SampleSize": sample_size})

    cmp_marginals = pd.read_csv(
        'tests/data/model/test_DamageModel_'
        'perform_dmg_task/CMP_marginals.csv', index_col=0)
    asset_model.load_cmp_model({'marginals': cmp_marginals})

    asset_model.generate_cmp_sample(sample_size)

    damage_model.load_damage_model(
        ['tests/data/model/test_DamageModel_perform_dmg_task/fragility_DB_test.csv'])

    block_batch_size = 1
    qnt_samples = []
    pg_batch = damage_model._get_pg_batches(block_batch_size)
    batches = pg_batch.index.get_level_values(0).unique()
    for PGB_i in batches:
        PGB = pg_batch.loc[PGB_i]
        capacity_sample, lsds_sample = damage_model._generate_dmg_sample(
            sample_size, PGB)
        EDP_req = damage_model._get_required_demand_type(PGB)
        demand_dict = damage_model._assemble_required_demand_data(EDP_req)
        ds_sample = damage_model._evaluate_damage_state(
            demand_dict, EDP_req,
            capacity_sample, lsds_sample)
        qnt_sample = damage_model._prepare_dmg_quantities(
            PGB, ds_sample,
            dropzero=False,
            dropempty=False)
        qnt_samples.append(qnt_sample)
    qnt_sample = pd.concat(qnt_samples, axis=1)
    qnt_sample.sort_index(axis=1, inplace=True)

    dmg_process = {
        "1_CMP.B": {
            "DS1": "CMP.A_DS1"
        }
    }
    dmg_process = {key: dmg_process[key] for key in sorted(dmg_process)}
    for task in dmg_process.items():
        qnt_sample = damage_model._perform_dmg_task(task, qnt_sample)

def test_DamageModel__get_pg_batches():

    asmt = assessment.Assessment()
    damage_model = asmt.damage
    asset_model = asmt.asset

    asset_model.cmp_marginal_params = pd.DataFrame(
        np.full((4, 2), 2.00),
        index=pd.MultiIndex.from_tuples(
            (('cmp_1', 1, 1),
             ('cmp_1', 1, 2),
             ('cmp_2', 1, 1),
             ('cmp_2', 1, 2)
            ), names=['cmp', 'loc', 'dir']
        ),
        columns=('Theta_0', 'Blocks')
    )

    damage_model.damage_params = pd.DataFrame(
        np.empty(2),
        index=('cmp_1', 'cmp_2'),
        columns=['ID']
    )

    df_1 = damage_model._get_pg_batches(1)
    assert [i[0] for i in df_1.index] == [1, 2, 3, 4]

    df_4 = damage_model._get_pg_batches(4)
    assert [i[0] for i in df_4.index] == [1, 1, 2, 2]

    df_8 = damage_model._get_pg_batches(8)
    assert [i[0] for i in df_8.index] == [1, 1, 1, 1]


def test_DamageModel_calculate():

    asmt = assessment.Assessment()
    dmg_process = {
        "1_collapse": {
            "DS1": "ALL_NA"
        },
        "2_excessiveRID": {
            "DS1": "irreparable_DS1"
        }
    }
    asmt.demand._sample = pd.DataFrame(
        np.column_stack((
            np.array((4.94, 2.73, 4.26, 2.79)),
            np.array((4.74, 2.23, 4.14, 2.28)),
            np.array((0.02, 0.022, 0.021, 0.02)),
            np.array((0.02, 0.022, 0.021, 0.02)),
        )),
        columns=pd.MultiIndex.from_tuples(
            (('PFA', '1', '1'),
             ('PFA', '1', '2'),
             ('PID', '1', '1'),
             ('PID', '1', '2'),
            ), names=['type', 'loc', 'dir']
        ),
        index = range(4)
    )
    asmt.asset.cmp_marginal_params = pd.DataFrame(
        np.full((4, 2), 2.00),
        index=pd.MultiIndex.from_tuples(
            (('cmp_1', '1', '1'),
             ('cmp_1', '1', '2'),
             ('cmp_2', '1', '1'),
             ('cmp_2', '1', '2')
            ), names=['cmp', 'loc', 'dir']
        ),
        columns=('Theta_0', 'Blocks')
    )
    asmt.asset.generate_cmp_sample(sample_size=4)
    asmt.damage.damage_params = pd.DataFrame(
        np.array((
            (1.0, 0.0, 'Peak Interstory Drift Ratio', 'ea', 0.0,
             None, 'lognormal', 1e-2, 0.40,
             None, 'lognormal', 2e-2, 0.40,
             None, 'lognormal', 3e-2, 0.40,
             None, 'lognormal', 4e-2, 0.40),
            (1.0, 0.0, 'Peak Interstory Drift Ratio', 'ea', 0.0,
             None, 'lognormal', 1e-2, 0.40,
             None, 'lognormal', 2e-2, 0.40,
             None, 'lognormal', 3e-2, 0.40,
             None, 'lognormal', 4e-2, 0.40),
            )),
        index=['cmp_1', 'cmp_2'],
        columns=pd.MultiIndex.from_tuples(
        ((    'Demand',        'Directional'),
         (    'Demand',             'Offset'),
         (    'Demand',               'Type'),
         (    'Demand',               'Unit'),
         ('Incomplete',                   ''),
         (       'LS1', 'DamageStateWeights'),
         (       'LS1',             'Family'),
         (       'LS1',            'Theta_0'),
         (       'LS1',            'Theta_1'),
         (       'LS2', 'DamageStateWeights'),
         (       'LS2',             'Family'),
         (       'LS2',            'Theta_0'),
         (       'LS2',            'Theta_1'),
         (       'LS3', 'DamageStateWeights'),
         (       'LS3',             'Family'),
         (       'LS3',            'Theta_0'),
         (       'LS3',            'Theta_1'),
         (       'LS4', 'DamageStateWeights'),
         (       'LS4',             'Family'),
         (       'LS4',            'Theta_0'),
         (       'LS4',            'Theta_1')),
        )
    )    
    asmt.damage.calculate(dmg_process=dmg_process)
    assert asmt.damage._dmg_function_scale_factors is None

    # note: Due to inherrent randomness, we can't assert the actual
    # values of this result
    assert asmt.damage._sample is not None


def test_LossModel_init():

    asmt = assessment.Assessment()
    mdl = model.LossModel(asmt)
    assert mdl.log_msg
    assert mdl.log_div

    assert mdl._sample is None
    assert mdl.loss_type == 'Generic'

def test_LossModel_load_sample_save_sample():

    asmt = assessment.Assessment()
    mdl = model.LossModel(asmt)

    # todo: see the format of the sample from the example and
    # replicate here to test these functions.

    mdl.loss_params = pd.DataFrame(
        (
            ('normal', None, '25704,17136|5,20',
             '0.390923', 'USD_2011', 0.0, '1 EA'),
            ('normal', 0.0, '22.68,15.12|5,20',
             '0.464027', 'worker_day', 0.0, '1 EA'),
        ),
        index=pd.MultiIndex.from_tuples(
            (('B.10.41.001a', 'Cost'),
             ('B.10.41.001a', 'Time'))
        ),
        columns=pd.MultiIndex.from_tuples(
            ((       'DS1',       'Family'),
             (       'DS1', 'LongLeadTime'),
             (       'DS1',      'Theta_0'),
             (       'DS1',      'Theta_1'),
             (        'DV',         'Unit'),
             ('Incomplete',             ''),
             (  'Quantity',         'Unit')
            )
        )
    )

    sample = pd.DataFrame(
        (
            (100.00, 1.00),
            (100.00, 1.00),
        ),
        index=(0, 1),
        columns=pd.MultiIndex.from_tuples(
            (('COST', 'B.10.41.001a', 'B.10.41.001a', '1', '1', '1'),
             ('TIME', 'B.10.41.001a', 'B.10.41.001a', '1', '1', '1'),
            ),
            names=('dv', 'loss', 'dmg', 'ds', 'loc', 'dir')
        )
    )

    mdl.load_sample(sample)

    pd.testing.assert_frame_equal(
        sample,
        mdl._sample)
    
    output = mdl.save_sample(None)
    output.index = output.index.astype('int64')

    pd.testing.assert_frame_equal(
        sample,
        output)

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.


def test_prep_constant_median_DV():

    median = 10.00
    constant_median_DV = model.prep_constant_median_DV(median)
    assert constant_median_DV() == median


def test_prep_bounded_multilinear_median_DV():

    medians = np.array((1.00, 2.00, 3.00, 4.00, 5.00))
    quantities = np.array((0.00, 1.00, 2.00, 3.00, 4.00))
    f = model.prep_bounded_multilinear_median_DV(medians, quantities)

    result = f(2.5)
    expected = 3.5
    assert result == expected

    result = f(0.00)
    expected = 1.00
    assert result == expected

    result = f(4.00)
    expected = 5.0
    assert result == expected

    result = f(-1.00)
    expected = 1.00
    assert result == expected

    result = f(5.00)
    expected = 5.00
    assert result == expected

    result = f([2.5, 3.5])
    expected = [3.5, 4.5]
    assert np.allclose(result, expected)

    with pytest.raises(ValueError):
        f(None)
