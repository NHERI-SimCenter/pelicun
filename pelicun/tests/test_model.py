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
These are unit and integration tests on the model module of pelicun.
"""

import os
import tempfile
from copy import deepcopy
import pytest
import numpy as np
import pandas as pd
from pelicun import model
from pelicun import assessment

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=arguments-renamed

#  __  __      _   _               _
# |  \/  | ___| |_| |__   ___   __| |___
# | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
# | |  | |  __/ |_| | | | (_) | (_| \__ \
# |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#
# The following tests verify the methods of the objects of the module.


class TestModelModule:
    @pytest.fixture
    def assessment_factory(self):
        def create_instance(verbose):
            x = assessment.Assessment()
            x.log.verbose = verbose
            return x

        return create_instance

    @pytest.fixture(params=[True, False])
    def assessment_instance(self, request, assessment_factory):
        return deepcopy(assessment_factory(request.param))


class TestDemandModel(TestModelModule):
    @pytest.fixture
    def demand_model(self, assessment_instance):
        return deepcopy(assessment_instance.demand)

    @pytest.fixture
    def demand_model_with_sample(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel_load_sample/demand_sample_A.csv'
        )
        return deepcopy(mdl)

    @pytest.fixture
    def calibrated_demand_model(self, demand_model_with_sample):
        config = {
            "ALL": {
                "DistributionFamily": "normal",
                "AddUncertainty": 0.00,
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateUpper": "0.06",
            },
            "SA": {
                "DistributionFamily": "empirical",
            },
        }
        demand_model_with_sample.calibrate_model(config)
        return deepcopy(demand_model_with_sample)

    @pytest.fixture
    def demand_model_with_sample_B(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel_load_sample/demand_sample_B.csv'
        )
        return deepcopy(mdl)

    @pytest.fixture
    def demand_model_with_sample_C(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel_load_sample/demand_sample_C.csv'
        )
        return deepcopy(mdl)

    @pytest.fixture
    def demand_model_with_sample_D(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel_load_sample/demand_sample_D.csv'
        )
        return deepcopy(mdl)

    def test_init(self, demand_model):
        assert demand_model.log_msg
        assert demand_model.log_div

        assert demand_model.marginal_params is None
        assert demand_model.correlation is None
        assert demand_model.empirical_data is None
        assert demand_model.units is None
        assert demand_model._RVs is None
        assert demand_model.sample is None

    def test_save_sample(self, demand_model_with_sample):
        # instantiate a temporary directory in memory
        temp_dir = tempfile.mkdtemp()
        # save the sample there
        demand_model_with_sample.save_sample(f'{temp_dir}/temp.csv')
        with open(f'{temp_dir}/temp.csv', 'r', encoding='utf-8') as f:
            contents = f.read()
        assert contents == (
            ',PFA-0-1,PFA-1-1,PID-1-1,SA_0.23-0-1\n'
            'Units,inps2,inps2,rad,inps2\n'
            '0,158.62478,397.04389,0.02672,342.149\n'
        )
        res = demand_model_with_sample.save_sample(save_units=False)
        assert res.to_dict() == {
            ('PFA', '0', '1'): {0: 158.62478},
            ('PFA', '1', '1'): {0: 397.04389},
            ('PID', '1', '1'): {0: 0.02672},
            ('SA_0.23', '0', '1'): {0: 342.149},
        }

    def test_load_sample(self, demand_model_with_sample, demand_model_with_sample_B):
        # retrieve the loaded sample and units
        obtained_sample = demand_model_with_sample.sample
        obtained_units = demand_model_with_sample.units

        obtained_sample_2 = demand_model_with_sample_B.sample
        obtained_units_2 = demand_model_with_sample_B.units

        # demand_sample_A.csv and demand_sample_B.csv only differ in the
        # headers, where the first includes a tag for the hazard
        # level. Therefore, the two files are expected to result to the
        # same `obtained_sample`

        pd.testing.assert_frame_equal(
            obtained_sample,
            obtained_sample_2,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_series_equal(
            obtained_units,
            obtained_units_2,
            check_index_type=False,
        )

        # compare against the expected values for the sample
        expected_sample = pd.DataFrame(
            [
                [4.029069, 10.084915, 0.02672, 8.690585],
            ],
            columns=pd.MultiIndex.from_tuples(
                (
                    ('PFA', '0', '1'),
                    ('PFA', '1', '1'),
                    ('PID', '1', '1'),
                    ('SA_0.23', '0', '1'),
                ),
                names=('type', 'loc', 'dir'),
            ),
            index=[0],
        )
        pd.testing.assert_frame_equal(
            expected_sample,
            obtained_sample,
            check_index_type=False,
            check_column_type=False,
        )

        # compare against the expected values for the units
        expected_units = pd.Series(
            ('inps2', 'inps2', 'rad', 'inps2'),
            index=pd.MultiIndex.from_tuples(
                (
                    ('PFA', '0', '1'),
                    ('PFA', '1', '1'),
                    ('PID', '1', '1'),
                    ('SA_0.23', '0', '1'),
                ),
                names=['type', 'loc', 'dir'],
            ),
            name='Units',
        )
        pd.testing.assert_series_equal(
            expected_units,
            obtained_units,
            check_index_type=False,
        )

    def test_estimate_RID(self, demand_model_with_sample):
        demands = demand_model_with_sample.sample['PID']
        params = {'yield_drift': 0.01}
        res = demand_model_with_sample.estimate_RID(demands, params)
        assert list(res.columns) == [('RID', '1', '1')]
        assert (
            demand_model_with_sample.estimate_RID(demands, params, method='xyz') is None
        )

    def test_calibrate_model(self, calibrated_demand_model, demand_model_with_sample_C):
        assert calibrated_demand_model.marginal_params['Family'].to_list() == [
            'normal',
            'normal',
            'lognormal',
            'empirical',
        ]
        assert (
            calibrated_demand_model.marginal_params.at[
                ('PID', '1', '1'), 'TruncateUpper'
            ]
            == 0.06
        )

    def test_calibrate_model_censoring(
        self, calibrated_demand_model, demand_model_with_sample_C
    ):
        # with a config featuring censoring the RIDs
        config = {
            "ALL": {
                "DistributionFamily": "normal",
                "AddUncertainty": 0.00,
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "CensorUpper": "0.05",
            },
        }
        demand_model_with_sample_C.calibrate_model(config)

    def test_calibrate_model_truncation(
        self, calibrated_demand_model, demand_model_with_sample_C
    ):
        # with a config that specifies a truncation limit smaller than
        # the samples
        config = {
            "ALL": {
                "DistributionFamily": "normal",
                "AddUncertainty": 0.00,
            },
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateUpper": "0.04",
            },
        }
        demand_model_with_sample_C.calibrate_model(config)

    def test_save_load_model_with_empirical(
        self, calibrated_demand_model, assessment_instance
    ):
        # a model that has empirical marginal parameters
        temp_dir = tempfile.mkdtemp()
        calibrated_demand_model.save_model(f'{temp_dir}/temp')
        assert os.path.exists(f'{temp_dir}/temp_marginals.csv')
        assert os.path.exists(f'{temp_dir}/temp_empirical.csv')
        assert os.path.exists(f'{temp_dir}/temp_correlation.csv')

        # Load model to a different DemandModel instance to verify
        new_demand_model = assessment_instance.demand
        new_demand_model.load_model(f'{temp_dir}/temp')
        pd.testing.assert_frame_equal(
            calibrated_demand_model.marginal_params,
            new_demand_model.marginal_params,
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            calibrated_demand_model.correlation,
            new_demand_model.correlation,
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            calibrated_demand_model.empirical_data,
            new_demand_model.empirical_data,
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )

    # # todo: this currently fails
    # def test_save_load_model_without_empirical(
    #     self, demand_model_with_sample_C, assessment_instance
    # ):
    #     # a model that does not have empirical marginal parameters
    #     temp_dir = tempfile.mkdtemp()
    #     config = {
    #         "ALL": {
    #             "DistributionFamily": "normal",
    #             "AddUncertainty": 0.00,
    #         },
    #         "PID": {
    #             "DistributionFamily": "lognormal",
    #             "TruncateUpper": "0.04",
    #         },
    #     }
    #     demand_model_with_sample_C.calibrate_model(config)
    #     demand_model_with_sample_C.save_model(f'{temp_dir}/temp')
    #     assert os.path.exists(f'{temp_dir}/temp_marginals.csv')
    #     assert os.path.exists(f'{temp_dir}/temp_empirical.csv')
    #     assert os.path.exists(f'{temp_dir}/temp_correlation.csv')

    #     # Load model to a different DemandModel instance to verify
    #     new_demand_model = assessment_instance.demand
    #     new_demand_model.load_model(f'{temp_dir}/temp')
    #     pd.testing.assert_frame_equal(
    #         demand_model_with_sample_C.marginal_params,
    #         new_demand_model.marginal_params,
    #     )
    #     pd.testing.assert_frame_equal(
    #         demand_model_with_sample_C.correlation,
    #         new_demand_model.correlation
    #     )
    #     pd.testing.assert_frame_equal(
    #         demand_model_with_sample_C.empirical_data,
    #         new_demand_model.empirical_data,
    #     )

    def test_generate_sample_exceptions(self, demand_model):
        # generating a sample from a non calibrated model should fail
        with pytest.raises(ValueError):
            demand_model.generate_sample({"SampleSize": 3, 'PreserveRawOrder': False})

    def test_generate_sample(self, calibrated_demand_model):
        calibrated_demand_model.generate_sample(
            {"SampleSize": 3, 'PreserveRawOrder': False}
        )

        # get the generated demand sample
        res = calibrated_demand_model.save_sample(save_units=True)
        assert isinstance(res, tuple)

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
                names=('type', 'loc', 'dir'),
            ),
            index=pd.Index((0, 1, 2), dtype='object'),
        )
        pd.testing.assert_frame_equal(
            expected_sample,
            obtained_sample,
            check_exact=False,
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )

        # compare against the expected values for the units
        expected_units = pd.Series(
            ('inps2', 'inps2', 'rad', 'inps2'),
            index=pd.MultiIndex.from_tuples(
                (
                    ('PFA', '0', '1'),
                    ('PFA', '1', '1'),
                    ('PID', '1', '1'),
                    ('SA_0.23', '0', '1'),
                ),
                names=('type', 'loc', 'dir'),
            ),
            name='Units',
        )
        pd.testing.assert_series_equal(
            expected_units,
            obtained_units,
            check_index_type=False,
        )

    def test_generate_sample_with_demand_cloning(self, assessment_instance):
        # # used for debugging
        # assessment_instance = assessment.Assessment()

        demand_model = assessment_instance.demand

        mdl = assessment_instance.demand
        # contains PGV-0-1, PGV-1-1, PGV-2-1, and PGA-0-1
        # PGA-0-1 is not cloned.
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel_generate_sample_with_demand_cloning/sample.csv'
        )
        demand_model.calibrate_model(
            {
                "ALL": {
                    "DistributionFamily": "lognormal",
                },
            }
        )
        demand_model.generate_sample(
            {
                'SampleSize': 1000,
                'DemandCloning': {
                    'PGV-0-1': ['PGV-0-1', 'PGV-0-2', 'PGV-0-3'],
                    'PGV-1-1': ['PGV-1-1', 'PGV-1-2', 'PGV-1-3'],
                    'PGV-2-1': ['PGV-2-1', 'PGV-2-2', 'PGV-2-3'],
                    'not_present': ['X-0-0', 'Y-0-0', 'Z-0-0'],
                },
            }
        )
        # we'll just get a warning for the `not_present` entry
        assert demand_model.sample.columns.to_list() == [
            ('PGA', '0', '1'),
            ('PGV', '0', '1'),
            ('PGV', '0', '2'),
            ('PGV', '0', '3'),
            ('PGV', '1', '1'),
            ('PGV', '1', '2'),
            ('PGV', '1', '3'),
            ('PGV', '2', '1'),
            ('PGV', '2', '2'),
            ('PGV', '2', '3'),
        ]
        assert np.array_equal(
            demand_model.sample[('PGV', '0', '1')].values,
            demand_model.sample[('PGV', '0', '3')].values,
        )
        # exceptions
        # Duplicate entries in demand cloning configuration
        with pytest.raises(ValueError):
            demand_model.generate_sample(
                {
                    'SampleSize': 1000,
                    'DemandCloning': {
                        'PGV-0-1': ['PGV-0-1', 'PGV-0-2', 'PGV-0-3'],
                        'PGV-1-1': ['PGV-0-1', 'PGV-1-2', 'PGV-1-3'],
                        'PGV-2-1': ['PGV-0-1', 'PGV-2-2', 'PGV-2-3'],
                    },
                }
            )


class TestPelicunModel(TestModelModule):
    @pytest.fixture
    def pelicun_model(self, assessment_instance):
        return deepcopy(model.PelicunModel(assessment_instance))

    def test_init(self, pelicun_model):
        assert pelicun_model.log_msg
        assert pelicun_model.log_div

    def test_convert_marginal_params(self, pelicun_model):
        # one row, only Theta_0, no conversion
        marginal_params = pd.DataFrame(
            [['1.0']],
            columns=['Theta_0'],
            index=pd.MultiIndex.from_tuples(
                (('A', '0', '1'),), names=('cmp', 'loc', 'dir')
            ),
        )
        units = pd.Series(['ea'], index=marginal_params.index)
        arg_units = None
        res = pelicun_model.convert_marginal_params(marginal_params, units, arg_units)

        # >>> res
        #             Theta_0
        # cmp loc dir
        # A   0   1       1.0

        assert 'Theta_0' in res.columns
        assert res.to_dict() == {'Theta_0': {('A', '0', '1'): 1.0}}

        # many rows, with conversions
        marginal_params = pd.DataFrame(
            [
                [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan],
                ['normal', np.nan, 1.0, np.nan, -0.50, 0.50],
                ['lognormal', 1.0, 0.5, np.nan, 0.50, 1.50],
                ['uniform', 0.0, 10.0, np.nan, np.nan, np.nan],
            ],
            columns=[
                'Family',
                'Theta_0',
                'Theta_1',
                'Theta_2',
                'TruncateLower',
                'TruncateUpper',
            ],
            index=pd.MultiIndex.from_tuples(
                (
                    ('A', '0', '1'),
                    ('B', '0', '1'),
                    ('C', '0', '1'),
                    ('D', '0', '1'),
                ),
                names=('cmp', 'loc', 'dir'),
            ),
        )
        units = pd.Series(['ea', 'ft', 'in', 'in2'], index=marginal_params.index)
        arg_units = None
        res = pelicun_model.convert_marginal_params(marginal_params, units, arg_units)

        expected_df = pd.DataFrame(
            {
                'Family': [np.nan, 'normal', 'lognormal', 'uniform'],
                'Theta_0': [1.0000, np.nan, 0.0254, 0.0000],
                'Theta_1': [np.nan, 1.000000, 0.500000, 0.0064516],
                'Theta_2': [np.nan, np.nan, np.nan, np.nan],
                'TruncateLower': [np.nan, -0.1524, 0.0127, np.nan],
                'TruncateUpper': [np.nan, 0.1524, 0.0381, np.nan],
            },
            index=pd.MultiIndex.from_tuples(
                (
                    ('A', '0', '1'),
                    ('B', '0', '1'),
                    ('C', '0', '1'),
                    ('D', '0', '1'),
                ),
                names=('cmp', 'loc', 'dir'),
            ),
        )

        pd.testing.assert_frame_equal(
            expected_df, res, check_index_type=False, check_column_type=False
        )

        # a case with arg_units
        marginal_params = pd.DataFrame(
            [['500.0,400.00|20,10']],
            columns=['Theta_0'],
            index=pd.MultiIndex.from_tuples(
                (('A', '0', '1'),), names=('cmp', 'loc', 'dir')
            ),
        )
        units = pd.Series(['test_three'], index=marginal_params.index)
        arg_units = pd.Series(['test_two'], index=marginal_params.index)
        res = pelicun_model.convert_marginal_params(marginal_params, units, arg_units)

        # >>> res
        #                              Theta_0
        # cmp loc dir
        # A   0   1    750,600|40,20

        # note: '40,20' = '20,10' * 2.00 (test_two)
        # note: '750,600' = '500,400' * 3.00 / 2.00 (test_three/test_two)

        expected_df = pd.DataFrame(
            {
                'Theta_0': ['750,600|40,20'],
            },
            index=pd.MultiIndex.from_tuples(
                (('A', '0', '1'),),
                names=('cmp', 'loc', 'dir'),
            ),
        )
        pd.testing.assert_frame_equal(
            expected_df, res, check_index_type=False, check_column_type=False
        )


class TestAssetModel(TestPelicunModel):
    @pytest.fixture
    def asset_model(self, assessment_instance):
        return deepcopy(assessment_instance.asset)

    def test_init(self, asset_model):
        assert asset_model.log_msg
        assert asset_model.log_div
        assert asset_model.cmp_marginal_params is None
        assert asset_model.cmp_units is None
        assert asset_model._cmp_RVs is None
        assert asset_model._cmp_sample is None

    def test_save_cmp_sample(self, asset_model):
        asset_model._cmp_sample = pd.DataFrame(
            {
                ('component_a', f'{i}', f'{j}', '0'): 8.0
                for i in range(1, 3)
                for j in range(1, 3)
            },
            index=range(10),
            columns=pd.MultiIndex.from_tuples(
                (
                    ('component_a', f'{i}', f'{j}', '0')
                    for i in range(1, 3)
                    for j in range(1, 3)
                ),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        )

        asset_model.cmp_units = pd.Series(
            data=['ea'], index=['component_a'], name='Units'
        )

        res = asset_model.save_cmp_sample()
        assert isinstance(res, pd.DataFrame)

        temp_dir = tempfile.mkdtemp()
        # save the sample there
        asset_model.save_cmp_sample(f'{temp_dir}/temp.csv')

        # load the component sample to a different AssetModel
        asmt = assessment.Assessment()
        asset_model = asmt.asset
        asset_model.load_cmp_sample(f'{temp_dir}/temp.csv')

        # also test loading sample to variables
        # (but we don't inspect them)
        _ = asset_model.save_cmp_sample(save_units=False)
        _, _ = asset_model.save_cmp_sample(save_units=True)

    def test_load_cmp_model_1(self, asset_model):
        cmp_marginals = pd.read_csv(
            'pelicun/tests/data/model/test_AssetModel/CMP_marginals.csv',
            index_col=0,
        )
        asset_model.load_cmp_model({'marginals': cmp_marginals})

        expected_cmp_marginal_params = pd.DataFrame(
            {
                'Theta_0': (8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
                'Blocks': (1, 1, 1, 1, 1, 1),
            },
            index=pd.MultiIndex.from_tuples(
                (
                    ('component_a', '0', '1', '0'),
                    ('component_a', '0', '2', '0'),
                    ('component_a', '1', '1', '0'),
                    ('component_a', '1', '2', '0'),
                    ('component_a', '2', '1', '0'),
                    ('component_a', '2', '2', '0'),
                ),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        ).astype({'Theta_0': 'float64', 'Blocks': 'int64'})

        pd.testing.assert_frame_equal(
            expected_cmp_marginal_params,
            asset_model.cmp_marginal_params,
            check_index_type=False,
            check_column_type=False,
            check_dtype=False,
        )

        expected_cmp_units = pd.Series(data=['ea'], index=['component_a'], name='Units')

        pd.testing.assert_series_equal(
            expected_cmp_units,
            asset_model.cmp_units,
            check_index_type=False,
        )

    def test_load_cmp_model_2(self, asset_model):
        # component marginals utilizing the keywords '--', 'all', 'top', 'roof'
        cmp_marginals = pd.read_csv(
            'pelicun/tests/data/model/test_AssetModel/CMP_marginals_2.csv',
            index_col=0,
        )
        asset_model._asmnt.stories = 4
        asset_model.load_cmp_model({'marginals': cmp_marginals})

        assert asset_model.cmp_marginal_params.to_dict() == {
            'Theta_0': {
                ('component_a', '0', '1', '0'): 1.0,
                ('component_a', '0', '2', '0'): 1.0,
                ('component_a', '1', '1', '0'): 1.0,
                ('component_a', '1', '2', '0'): 1.0,
                ('component_a', '2', '1', '0'): 1.0,
                ('component_a', '2', '2', '0'): 1.0,
                ('component_a', '3', '1', '0'): 1.0,
                ('component_a', '3', '2', '0'): 1.0,
                ('component_b', '1', '1', '0'): 1.0,
                ('component_b', '2', '1', '0'): 1.0,
                ('component_b', '3', '1', '0'): 1.0,
                ('component_b', '4', '1', '0'): 1.0,
                ('component_c', '0', '1', '0'): 1.0,
                ('component_c', '1', '1', '0'): 1.0,
                ('component_c', '2', '1', '0'): 1.0,
                ('component_d', '4', '1', '0'): 1.0,
                ('component_e', '5', '1', '0'): 1.0,
            },
            'Blocks': {
                ('component_a', '0', '1', '0'): 1,
                ('component_a', '0', '2', '0'): 1,
                ('component_a', '1', '1', '0'): 1,
                ('component_a', '1', '2', '0'): 1,
                ('component_a', '2', '1', '0'): 1,
                ('component_a', '2', '2', '0'): 1,
                ('component_a', '3', '1', '0'): 1,
                ('component_a', '3', '2', '0'): 1,
                ('component_b', '1', '1', '0'): 1,
                ('component_b', '2', '1', '0'): 1,
                ('component_b', '3', '1', '0'): 1,
                ('component_b', '4', '1', '0'): 1,
                ('component_c', '0', '1', '0'): 1,
                ('component_c', '1', '1', '0'): 1,
                ('component_c', '2', '1', '0'): 1,
                ('component_d', '4', '1', '0'): 1,
                ('component_e', '5', '1', '0'): 1,
            },
        }

        expected_cmp_units = pd.Series(
            data=['ea'] * 5,
            index=[f'component_{x}' for x in ('a', 'b', 'c', 'd', 'e')],
            name='Units',
        )

        pd.testing.assert_series_equal(
            expected_cmp_units,
            asset_model.cmp_units,
            check_index_type=False,
        )

    def test_load_cmp_model_csv(self, asset_model):
        # load by directly specifying the csv file
        cmp_marginals = 'pelicun/tests/data/model/test_AssetModel/CMP'
        asset_model.load_cmp_model(cmp_marginals)

    def test_load_cmp_model_exceptions(self, asset_model):
        cmp_marginals = pd.read_csv(
            'pelicun/tests/data/model/test_AssetModel/CMP_marginals_invalid_loc.csv',
            index_col=0,
        )
        asset_model._asmnt.stories = 4
        with pytest.raises(ValueError):
            asset_model.load_cmp_model({'marginals': cmp_marginals})

        cmp_marginals = pd.read_csv(
            'pelicun/tests/data/model/test_AssetModel/CMP_marginals_invalid_dir.csv',
            index_col=0,
        )
        asset_model._asmnt.stories = 4
        with pytest.raises(ValueError):
            asset_model.load_cmp_model({'marginals': cmp_marginals})

    def test_generate_cmp_sample(self, asset_model):
        asset_model.cmp_marginal_params = pd.DataFrame(
            {'Theta_0': (8.0, 8.0, 8.0, 8.0), 'Blocks': (1.0, 1.0, 1.0, 1.0)},
            index=pd.MultiIndex.from_tuples(
                (
                    ('component_a', '1', '1', '0'),
                    ('component_a', '1', '2', '0'),
                    ('component_a', '2', '1', '0'),
                    ('component_a', '2', '2', '0'),
                ),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        )

        asset_model.cmp_units = pd.Series(
            data=['ea'], index=['component_a'], name='Units'
        )

        asset_model.generate_cmp_sample(sample_size=10)

        assert asset_model._cmp_RVs is not None

        expected_cmp_sample = pd.DataFrame(
            {
                ('component_a', f'{i}', f'{j}'): 8.0
                for i in range(1, 3)
                for j in range(1, 3)
            },
            index=range(10),
            columns=pd.MultiIndex.from_tuples(
                (
                    ('component_a', f'{i}', f'{j}', '0')
                    for i in range(1, 3)
                    for j in range(1, 3)
                ),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        )

        pd.testing.assert_frame_equal(
            expected_cmp_sample,
            asset_model.cmp_sample,
            check_index_type=False,
            check_column_type=False,
        )

    # currently this is not working
    # def test_load_cmp_model_block_weights(self, asset_model):
    #     cmp_marginals = pd.read_csv(
    #         'pelicun/tests/data/model/test_AssetModel/CMP_marginals_block_weights.csv',
    #         index_col=0,
    #     )
    #     asset_model.load_cmp_model({'marginals': cmp_marginals})

    def test_generate_cmp_sample_exceptions_1(self, asset_model):
        # without marginal parameters
        with pytest.raises(ValueError):
            asset_model.generate_cmp_sample(sample_size=10)

    def test_generate_cmp_sample_exceptions_2(self, asset_model):
        # without specifying sample size
        cmp_marginals = pd.read_csv(
            'pelicun/tests/data/model/test_AssetModel/CMP_marginals.csv',
            index_col=0,
        )
        asset_model.load_cmp_model({'marginals': cmp_marginals})
        with pytest.raises(ValueError):
            asset_model.generate_cmp_sample()
        # but it should work if a demand sample is available
        asset_model._asmnt.demand.sample = np.empty(shape=(10, 2))
        asset_model.generate_cmp_sample()


class TestDamageModel(TestPelicunModel):
    @pytest.fixture
    def cmp_sample_A(self):
        # This sample contains 8 units of B.10.31.001 assigned to
        # locations 1, 2 and directions 1, 2
        return pd.DataFrame(
            {
                ('B.10.31.001', f'{i}', f'{j}', '0'): 8.0
                for i in range(1, 3)
                for j in range(1, 3)
            },
            index=range(10),
            columns=pd.MultiIndex.from_tuples(
                (
                    ('B.10.31.001', f'{i}', f'{j}', '0')
                    for i in range(1, 3)
                    for j in range(1, 3)
                ),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        )

    @pytest.fixture
    def calibration_config_A(self):
        return {
            "ALL": {"DistributionFamily": "lognormal"},
            "PID": {
                "DistributionFamily": "lognormal",
                "TruncateLower": "",
                "TruncateUpper": "0.06",
            },
        }

    @pytest.fixture
    def damage_model(self, assessment_instance):
        return deepcopy(assessment_instance.damage)

    @pytest.fixture
    def damage_model_model_loaded(self, damage_model, cmp_sample_A):
        asmt = damage_model._asmnt
        asmt.get_default_data('damage_DB_FEMA_P58_2nd')
        asmt.asset._cmp_sample = cmp_sample_A
        damage_model.load_damage_model(['PelicunDefault/damage_DB_FEMA_P58_2nd.csv'])
        return deepcopy(damage_model)

    @pytest.fixture
    def damage_model_with_sample(self, assessment_instance):
        dmg_process = None
        assessment_instance.demand.sample = pd.DataFrame(
            np.column_stack(
                (
                    np.array((4.94, 2.73, 4.26, 2.79)),
                    np.array((4.74, 2.23, 4.14, 2.28)),
                    np.array((0.02, 0.022, 0.021, 0.02)),
                    np.array((0.02, 0.022, 0.021, 0.02)),
                )
            ),
            columns=pd.MultiIndex.from_tuples(
                (
                    ('PFA', '1', '1'),
                    ('PFA', '1', '2'),
                    ('PID', '1', '1'),
                    ('PID', '1', '2'),
                ),
                names=['type', 'loc', 'dir'],
            ),
            index=range(4),
        )
        assessment_instance.asset.cmp_marginal_params = pd.DataFrame(
            np.full((4, 2), 2.00),
            index=pd.MultiIndex.from_tuples(
                (
                    ('cmp_1', '1', '1', '0'),
                    ('cmp_1', '1', '2', '0'),
                    ('cmp_2', '1', '1', '0'),
                    ('cmp_2', '1', '2', '0'),
                ),
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
            columns=('Theta_0', 'Blocks'),
        )
        assessment_instance.asset.generate_cmp_sample(sample_size=4)
        assessment_instance.damage.damage_params = pd.DataFrame(
            np.array(
                (
                    (
                        1.0,
                        0.0,
                        'Peak Interstory Drift Ratio',
                        'ea',
                        0.0,
                        None,
                        'lognormal',
                        1e-2,
                        0.40,
                        None,
                        'lognormal',
                        2e-2,
                        0.40,
                        None,
                        'lognormal',
                        3e-2,
                        0.40,
                        None,
                        'lognormal',
                        4e-2,
                        0.40,
                    ),
                    (
                        1.0,
                        0.0,
                        'Peak Interstory Drift Ratio',
                        'ea',
                        0.0,
                        None,
                        'lognormal',
                        1e-2,
                        0.40,
                        None,
                        'lognormal',
                        2e-2,
                        0.40,
                        None,
                        'lognormal',
                        3e-2,
                        0.40,
                        None,
                        'lognormal',
                        4e-2,
                        0.40,
                    ),
                )
            ),
            index=['cmp_1', 'cmp_2'],
            columns=pd.MultiIndex.from_tuples(
                (
                    ('Demand', 'Directional'),
                    ('Demand', 'Offset'),
                    ('Demand', 'Type'),
                    ('Demand', 'Unit'),
                    ('Incomplete', ''),
                    ('LS1', 'DamageStateWeights'),
                    ('LS1', 'Family'),
                    ('LS1', 'Theta_0'),
                    ('LS1', 'Theta_1'),
                    ('LS2', 'DamageStateWeights'),
                    ('LS2', 'Family'),
                    ('LS2', 'Theta_0'),
                    ('LS2', 'Theta_1'),
                    ('LS3', 'DamageStateWeights'),
                    ('LS3', 'Family'),
                    ('LS3', 'Theta_0'),
                    ('LS3', 'Theta_1'),
                    ('LS4', 'DamageStateWeights'),
                    ('LS4', 'Family'),
                    ('LS4', 'Theta_0'),
                    ('LS4', 'Theta_1'),
                )
            ),
        )
        assessment_instance.damage.calculate(sample_size=4, dmg_process=dmg_process)
        assessment_instance.asset.cmp_units = pd.Series(
            ['ea'] * len(assessment_instance.damage.sample.columns),
            index=assessment_instance.damage.sample.columns,
            name='Units',
            dtype='object',
        )
        return deepcopy(assessment_instance.damage)

    def test_init(self, damage_model):
        assert damage_model.log_msg
        assert damage_model.log_div

        assert damage_model.damage_params is None
        assert damage_model.sample is None

    def test_save_load_sample(self, damage_model_with_sample, assessment_instance):
        # saving to a file
        temp_dir = tempfile.mkdtemp()
        # convert the sample's index from RangeIndex to int64 (to
        # match the datatype when it is loaded back; the contents are
        # the same)
        damage_model_with_sample.sample.index = (
            damage_model_with_sample.sample.index.astype('int64')
        )
        damage_model_with_sample.save_sample(f'{temp_dir}/damage_model_sample.csv')
        # loading from the file
        assessment_instance.damage.load_sample(f'{temp_dir}/damage_model_sample.csv')
        sample_from_file = assessment_instance.damage.sample

        # saving to a variable
        sample_from_variable = damage_model_with_sample.save_sample(save_units=False)
        pd.testing.assert_frame_equal(
            sample_from_file,
            sample_from_variable,
            check_index_type=False,
            check_column_type=False,
        )
        _, units_from_variable = damage_model_with_sample.save_sample(save_units=True)
        assert np.all(units_from_variable.to_numpy() == 'ea')

    def test_load_damage_model(self, damage_model_model_loaded):
        # should no longer be None
        assert damage_model_model_loaded.damage_params is not None

        assert list(damage_model_model_loaded.damage_params.columns) == [
            ("Demand", "Directional"),
            ("Demand", "Offset"),
            ("Demand", "Type"),
            ("Demand", "Unit"),
            ("Incomplete", ""),
            ("LS1", "DamageStateWeights"),
            ("LS1", "Family"),
            ("LS1", "Theta_0"),
            ("LS1", "Theta_1"),
            ("LS2", "DamageStateWeights"),
            ("LS2", "Family"),
            ("LS2", "Theta_0"),
            ("LS2", "Theta_1"),
            ("LS3", "DamageStateWeights"),
            ("LS3", "Family"),
            ("LS3", "Theta_0"),
            ("LS3", "Theta_1"),
            ("LS4", "DamageStateWeights"),
            ("LS4", "Family"),
            ("LS4", "Theta_0"),
            ("LS4", "Theta_1"),
        ]

        assert list(damage_model_model_loaded.damage_params.index) == ['B.10.31.001']

        contents = damage_model_model_loaded.damage_params.to_numpy().reshape(-1)

        expected_contents = np.array(
            [
                1.0,
                0.0,
                'Peak Interstory Drift Ratio',
                'unitless',
                0.0,
                '0.950000 | 0.050000',
                'lognormal',
                0.04,
                0.4,
                None,
                'lognormal',
                0.08,
                0.4,
                None,
                'lognormal',
                0.11,
                0.4,
                np.nan,
                None,
                np.nan,
                np.nan,
            ],
            dtype=object,
        )

        # this comparison was tricky
        for x, y in zip(contents, expected_contents):
            if isinstance(x, str):
                assert x == y
            elif x is None:
                continue
            elif np.isnan(x):
                continue

    def test__create_dmg_RVs(self, damage_model_model_loaded):
        pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)

        batches = pg_batch.index.get_level_values(0).unique()
        for PGB_i in batches:
            PGB = pg_batch.loc[PGB_i]
            # ensure the following works in each case
            damage_model_model_loaded._create_dmg_RVs(PGB)

        # check the output for a single case
        PGB_i = batches[-1]
        PGB = pg_batch.loc[PGB_i]

        capacity_RV_reg, lsds_RV_reg = damage_model_model_loaded._create_dmg_RVs(PGB)

        assert capacity_RV_reg is not None
        assert lsds_RV_reg is not None

        assert list(capacity_RV_reg._variables.keys()) == [
            'FRG-B.10.31.001-2-2-0-1-1',
            'FRG-B.10.31.001-2-2-0-1-2',
            'FRG-B.10.31.001-2-2-0-1-3',
        ]

        assert not capacity_RV_reg._sets

        assert list(lsds_RV_reg._variables.keys()) == [
            'LSDS-B.10.31.001-2-2-0-1-1',
            'LSDS-B.10.31.001-2-2-0-1-2',
            'LSDS-B.10.31.001-2-2-0-1-3',
        ]

        assert not lsds_RV_reg._sets

        # test capacity adjustment: *1.20
        scaling_specification = {'B.10.31.001-2-2': '*1.20'}
        (
            adjusted_capacity_RV_reg,
            lsds_RV_reg,
        ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
        for limit_state in ('1', '2', '3'):
            val_initial = capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            val_scaling = adjusted_capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            assert val_scaling[0] == val_initial[0] * 1.20
            assert val_scaling[1] == val_initial[1]
            assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

        # test capacity adjustment: /1.20
        scaling_specification = {'B.10.31.001-2-2': '/1.20'}
        (
            adjusted_capacity_RV_reg,
            lsds_RV_reg,
        ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
        for limit_state in ('1', '2', '3'):
            val_initial = capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            val_scaling = adjusted_capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            assert val_scaling[0] == val_initial[0] / 1.20
            assert val_scaling[1] == val_initial[1]
            assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

        # test capacity adjustment: +0.50
        scaling_specification = {'B.10.31.001-2-2': '+0.50'}
        (
            adjusted_capacity_RV_reg,
            lsds_RV_reg,
        ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
        for limit_state in ('1', '2', '3'):
            val_initial = capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            val_scaling = adjusted_capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            assert val_scaling[0] == val_initial[0] + 0.50
            assert val_scaling[1] == val_initial[1]
            assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

        # test capacity adjustment: -0.05
        scaling_specification = {'B.10.31.001-2-2': '-0.05'}
        (
            adjusted_capacity_RV_reg,
            lsds_RV_reg,
        ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
        for limit_state in ('1', '2', '3'):
            val_initial = capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            val_scaling = adjusted_capacity_RV_reg.RV[
                f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
            ].theta
            assert val_scaling[0] == val_initial[0] - 0.05
            assert val_scaling[1] == val_initial[1]
            assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

        # edge cases: invalid capacity adjustment
        scaling_specification = {'B.10.31.001-2-2': 'import os; do_malicious_things'}
        with pytest.raises(ValueError):
            damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)

    def test__generate_dmg_sample(self, damage_model_model_loaded):
        pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
        batches = pg_batch.index.get_level_values(0).unique()
        PGB_i = batches[-1]
        PGB = pg_batch.loc[PGB_i]
        sample_size = 10

        # test the _generate_dmg_sample method
        (
            capacity_sample,
            lsds_sample,
        ) = damage_model_model_loaded._generate_dmg_sample(sample_size, PGB)

        # run a few checks on the results of the method

        # note: the method generates random results. We avoid checking
        # those for equality, because subsequent changes in the code might
        # break the tests. The functionality of the uq module, which is
        # used to generate the random samples, is tested with a dedicated
        # test suite.

        for res in (capacity_sample, lsds_sample):
            assert res.shape == (10, 3)

            assert list(res.columns) == [
                ('B.10.31.001', '2', '2', '0', '1', '1'),
                ('B.10.31.001', '2', '2', '0', '1', '2'),
                ('B.10.31.001', '2', '2', '0', '1', '3'),
            ]

            assert list(res.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test__get_required_demand_type(self, damage_model_model_loaded):
        pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
        batches = pg_batch.index.get_level_values(0).unique()
        PGB_i = batches[-1]
        PGB = pg_batch.loc[PGB_i]

        EDP_req = damage_model_model_loaded._get_required_demand_type(PGB)

        assert EDP_req == {'PID-2-2': [('B.10.31.001', '2', '2', '0')]}

    def test__assemble_required_demand_data(
        self, damage_model_model_loaded, calibration_config_A
    ):
        demand_model = damage_model_model_loaded._asmnt.demand
        demand_model.load_sample(
            'pelicun/tests/data/model/'
            'test_DamageModel_assemble_required_demand_data/'
            'demand_sample.csv'
        )
        demand_model.calibrate_model(calibration_config_A)

        pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
        batches = pg_batch.index.get_level_values(0).unique()

        expected_demand_dicts = [
            {'PID-1-1': np.array([0.001])},
            {'PID-1-2': np.array([0.002])},
            {'PID-2-1': np.array([0.003])},
            {'PID-2-2': np.array([0.004])},
        ]

        for i, PGB_i in enumerate(batches):
            PGB = pg_batch.loc[PGB_i]
            EDP_req = damage_model_model_loaded._get_required_demand_type(PGB)
            demand_dict = damage_model_model_loaded._assemble_required_demand_data(
                EDP_req
            )
            assert demand_dict == expected_demand_dicts[i]

    def test__evaluate_damage_state_and_prepare_dmg_quantities(
        self,
        damage_model_model_loaded,
        calibration_config_A,
    ):
        damage_model = damage_model_model_loaded
        demand_model = damage_model_model_loaded._asmnt.demand

        demand_model.load_sample(
            'pelicun/tests/data/model/'
            'test_DamageModel__evaluate_damage_state_and_prepare_dmg_quantities/'
            'demand_sample.csv'
        )
        # calibrate the model
        demand_model.calibrate_model(calibration_config_A)

        pg_batch = damage_model._get_pg_batches(block_batch_size=1)
        batches = pg_batch.index.get_level_values(0).unique()

        PGB_i = batches[-1]
        PGB = pg_batch.loc[PGB_i]
        EDP_req = damage_model._get_required_demand_type(PGB)
        demand_dict = damage_model._assemble_required_demand_data(EDP_req)

        sample_size = 10
        capacity_sample, lsds_sample = damage_model._generate_dmg_sample(
            sample_size, PGB
        )

        ds_sample = damage_model._evaluate_damage_state(
            demand_dict, EDP_req, capacity_sample, lsds_sample
        )

        qnt_sample = damage_model._prepare_dmg_quantities(ds_sample, dropzero=False)

        # note: the realized number of damage states is random, limiting
        # our assertions
        assert ds_sample.shape[0] == 10
        assert qnt_sample.shape[0] == 10
        assert list(qnt_sample.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert list(ds_sample.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        assert list(ds_sample.columns)[0] == ('B.10.31.001', '2', '2', '0', '1')
        assert list(qnt_sample.columns)[0] == ('B.10.31.001', '2', '2', '0', '0')

    def test__perform_dmg_task(self, assessment_instance):
        damage_model = assessment_instance.damage

        #
        # when CMP.B reaches DS1, CMP.A should be DS4
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.B": {"DS1": "CMP.A_DS4"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: 4, 1: 0, 2: 4},
            ('CMP.A', '1', '1', '1'): {0: 4, 1: 0, 2: 4},
            ('CMP.B', '1', '1', '0'): {0: 0, 1: 0, 2: 1},
            ('CMP.B', '1', '1', '1'): {0: 1, 1: 0, 2: 0},
        }

        #
        # when CMP.B reaches DS1, CMP.A should be NA (-1)
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.B": {"DS1": "CMP.A_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: -1, 1: 0, 2: -1},
            ('CMP.A', '1', '1', '1'): {0: -1, 1: 0, 2: -1},
            ('CMP.B', '1', '1', '0'): {0: 0, 1: 0, 2: 1},
            ('CMP.B', '1', '1', '1'): {0: 1, 1: 0, 2: 0},
        }

        #
        # `-LOC` keyword
        # when CMP.B reaches DS1, CMP.A should be DS4
        # matching locations
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '2', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '2', '1', '0'): [1, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.B-LOC": {"DS1": "CMP.A_DS4"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: 0, 1: 0, 2: 4},
            ('CMP.A', '2', '1', '0'): {0: 4, 1: 0, 2: 0},
            ('CMP.B', '1', '1', '0'): {0: 0, 1: 0, 2: 1},
            ('CMP.B', '2', '1', '0'): {0: 1, 1: 0, 2: 0},
        }

        #
        # ALL keyword
        #
        # Whenever CMP.A reaches DS1, all other components should be
        # set to DS2.
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [1, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 0],
                ('CMP.C', '1', '1', '0'): [0, 0, 0],
                ('CMP.D', '1', '1', '0'): [0, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.A": {"DS1": "ALL_DS2"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: 1, 1: 0, 2: 0},
            ('CMP.B', '1', '1', '0'): {0: 2, 1: 0, 2: 0},
            ('CMP.C', '1', '1', '0'): {0: 2, 1: 0, 2: 0},
            ('CMP.D', '1', '1', '0'): {0: 2, 1: 0, 2: 0},
        }

        #
        # NA keyword
        #
        # NA translates to -1 representing nan
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.B": {"DS1": "CMP.A_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: -1, 1: 0, 2: -1},
            ('CMP.A', '1', '1', '1'): {0: -1, 1: 0, 2: -1},
            ('CMP.B', '1', '1', '0'): {0: 0, 1: 0, 2: 1},
            ('CMP.B', '1', '1', '1'): {0: 1, 1: 0, 2: 0},
        }

        #
        # NA keyword combined with `-LOC`
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '2', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '2', '1', '0'): [1, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.B-LOC": {"DS1": "CMP.A_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: 0, 1: 0, 2: -1},
            ('CMP.A', '2', '1', '0'): {0: -1, 1: 0, 2: 0},
            ('CMP.B', '1', '1', '0'): {0: 0, 1: 0, 2: 1},
            ('CMP.B', '2', '1', '0'): {0: 1, 1: 0, 2: 0},
        }

        #
        # NA keyword combined with `-LOC` and `ALL`
        #

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 1],
                ('CMP.A', '2', '1', '0'): [1, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 0],
                ('CMP.B', '2', '1', '0'): [0, 0, 0],
                ('CMP.C', '1', '1', '0'): [0, 0, 0],
                ('CMP.C', '2', '1', '0'): [0, 0, 0],
            },
            dtype='int32',
        )
        ds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        dmg_process = {"1_CMP.A-LOC": {"DS1": "ALL_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        assert after.to_dict() == {
            ('CMP.A', '1', '1', '0'): {0: 0, 1: 0, 2: 1},
            ('CMP.A', '2', '1', '0'): {0: 1, 1: 0, 2: 0},
            ('CMP.B', '1', '1', '0'): {0: 0, 1: 0, 2: -1},
            ('CMP.B', '2', '1', '0'): {0: -1, 1: 0, 2: 0},
            ('CMP.C', '1', '1', '0'): {0: 0, 1: 0, 2: -1},
            ('CMP.C', '2', '1', '0'): {0: -1, 1: 0, 2: 0},
        }

    def test__get_pg_batches_1(self, assessment_instance):
        damage_model = assessment_instance.damage
        asset_model = assessment_instance.asset

        asset_model.cmp_marginal_params = pd.DataFrame(
            np.full((4, 2), 2.00),
            index=pd.MultiIndex.from_tuples(
                (
                    ('cmp_1', '1', '1', '0'),
                    ('cmp_1', '1', '2', '0'),
                    ('cmp_2', '1', '1', '0'),
                    ('cmp_2', '1', '2', '0'),
                ),
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
            columns=('Theta_0', 'Blocks'),
        )

        damage_model.damage_params = pd.DataFrame(
            np.empty(2), index=('cmp_1', 'cmp_2'), columns=['ID']
        )

        df_1 = damage_model._get_pg_batches(1)
        assert [i[0] for i in df_1.index] == [1, 2, 3, 4]

        df_4 = damage_model._get_pg_batches(4)
        assert [i[0] for i in df_4.index] == [1, 1, 2, 2]

        df_8 = damage_model._get_pg_batches(8)
        assert [i[0] for i in df_8.index] == [1, 1, 1, 1]

    def test__get_pg_batches_2(self, damage_model_model_loaded):
        # make sure that the method works for different batch sizes
        for i in (1, 4, 8, 10, 100):
            damage_model_model_loaded._get_pg_batches(block_batch_size=i)

        # verify the result is correct for certain cases
        res = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
        expected_res = pd.DataFrame(
            np.array((1, 1, 1, 1)),
            index=pd.MultiIndex.from_tuples(
                (
                    (1, 'B.10.31.001', '1', '1', '0'),
                    (2, 'B.10.31.001', '1', '2', '0'),
                    (3, 'B.10.31.001', '2', '1', '0'),
                    (4, 'B.10.31.001', '2', '2', '0'),
                ),
                names=('Batch', 'cmp', 'loc', 'dir', 'uid'),
            ),
            columns=('Blocks',),
        ).astype('Int64')

        pd.testing.assert_frame_equal(
            expected_res, res, check_index_type=False, check_column_type=False
        )

        res = damage_model_model_loaded._get_pg_batches(block_batch_size=1000)
        expected_res = pd.DataFrame(
            np.array((1, 1, 1, 1)),
            index=pd.MultiIndex.from_tuples(
                (
                    (1, 'B.10.31.001', '1', '1', '0'),
                    (1, 'B.10.31.001', '1', '2', '0'),
                    (1, 'B.10.31.001', '2', '1', '0'),
                    (1, 'B.10.31.001', '2', '2', '0'),
                ),
                names=('Batch', 'cmp', 'loc', 'dir', 'uid'),
            ),
            columns=('Blocks',),
        ).astype('Int64')

        pd.testing.assert_frame_equal(
            expected_res, res, check_index_type=False, check_column_type=False
        )

    def test_calculate(self, damage_model_with_sample):
        # note: Due to inherent randomness, we can't assert the actual
        # values of this result
        assert damage_model_with_sample.sample.values.all() >= 0.00
        assert damage_model_with_sample.sample.values.all() <= 2.00

    def test_calculate_multilinear_CDF(self, damage_model):
        # # used for debugging
        # assessment_instance = assessment.Assessment()
        # damage_model = assessment_instance.damage

        demand_model = damage_model._asmnt.demand
        assessment_instance = damage_model._asmnt
        asset_model = assessment_instance.asset

        # A damage calculation test utilizing a multilinear CDF RV for
        # the capcity.

        sample_size = 1000

        # define the demand
        conversion_factor = assessment_instance.unit_conversion_factors['inps2']
        demand_model.sample = pd.DataFrame(
            np.full(sample_size, 0.50 * conversion_factor),
            columns=(('PGV', '0', '1'),),
        )

        # Define the component in the asset model
        asset_model.cmp_marginal_params = pd.DataFrame(
            {
                'Theta_0': (1.0,),
                'Blocks': (1,),
            },
            index=pd.MultiIndex.from_tuples(
                (('test_component', '0', '1', '0'),),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        )
        # generate component samples
        asset_model.generate_cmp_sample()

        # define fragility curve with multilinear_CDF
        damage_model.load_damage_model(
            [
                'pelicun/tests/data/model/'
                'test_DamageModel_calculate_multilinear_CDF/'
                'damage_model.csv'
            ]
        )

        # calculate damage
        damage_model.calculate(sample_size)

        res = damage_model.sample.value_counts()
        assert res.to_dict() == {(1.0, 0.0): 750, (0.0, 1.0): 250}


class TestLossModel(TestPelicunModel):
    @pytest.fixture
    def loss_model(self, assessment_instance):
        return deepcopy(model.LossModel(assessment_instance))

    def test_init(self, loss_model):
        assert loss_model.log_msg
        assert loss_model.log_div

        assert loss_model.sample is None
        assert loss_model.loss_type == 'Generic'

    def test_load_sample_save_sample(self, loss_model):
        loss_model.loss_params = pd.DataFrame(
            (
                (
                    "normal",
                    None,
                    "25704,17136|5,20",
                    0.390923,
                    "USD_2011",
                    0.0,
                    "1 EA",
                ),
                (
                    "normal",
                    0.0,
                    "22.68,15.12|5,20",
                    0.464027,
                    "worker_day",
                    0.0,
                    "1 EA",
                ),
            ),
            index=pd.MultiIndex.from_tuples(
                (("B.10.41.001a", "Cost"), ("B.10.41.001a", "Time"))
            ),
            columns=pd.MultiIndex.from_tuples(
                (
                    ("DS1", "Family"),
                    ("DS1", "LongLeadTime"),
                    ("DS1", "Theta_0"),
                    ("DS1", "Theta_1"),
                    ("DV", "Unit"),
                    ("Incomplete", ""),
                    ("Quantity", "Unit"),
                )
            ),
        )

        sample = pd.DataFrame(
            (
                (100.00, 1.00),
                (100.00, 1.00),
            ),
            index=(0, 1),
            columns=pd.MultiIndex.from_tuples(
                (
                    ("Cost", "B.10.41.001a", "B.10.41.001a", "1", "1", "1"),
                    ("Time", "B.10.41.001a", "B.10.41.001a", "1", "1", "1"),
                ),
                names=("dv", "loss", "dmg", "ds", "loc", "dir"),
            ),
        )

        loss_model.load_sample(sample)

        pd.testing.assert_frame_equal(
            sample,
            loss_model.sample,
            check_index_type=False,
            check_column_type=False,
        )

        output = loss_model.save_sample(None)
        output.index = output.index.astype('int64')

        pd.testing.assert_frame_equal(
            sample, output, check_index_type=False, check_column_type=False
        )

    def test_load_model(self, loss_model):
        data_path_1 = pd.DataFrame(
            ((0, "1 EA", "USD_2011", 10000000.00), (0, "1 EA", "worker_day", 12500)),
            columns=pd.MultiIndex.from_tuples(
                (
                    ("Incomplete", None),
                    ("Quantity", "Unit"),
                    ("DV", "Unit"),
                    ("DS1", "Theta_0"),
                )
            ),
            index=pd.MultiIndex.from_tuples(
                (
                    ("replacement", "Cost"),
                    ("replacement", "Time"),
                )
            ),
        )
        data_path_2 = 'PelicunDefault/loss_repair_DB_FEMA_P58_2nd.csv'

        mapping_path = pd.DataFrame(
            (("B.10.31.001"), ("D.50.92.033k")),
            columns=["Generic"],
            index=["DMG-cmp_1", "DMG-cmp_2"],
        )

        assert loss_model.loss_map is None
        assert loss_model.loss_params is None

        loss_model.load_model([data_path_1, data_path_2], mapping_path)

        assert loss_model.loss_map.to_dict() == {
            'Driver': {0: ('DMG', 'cmp_1'), 1: ('DMG', 'cmp_2')},
            'Consequence': {0: 'B.10.31.001', 1: 'D.50.92.033k'},
        }
        cmp_ids = loss_model.loss_params.index.get_level_values(0).unique()
        assert "B.10.31.001" in cmp_ids
        assert "D.50.92.033k" in cmp_ids

    def test_aggregate_losses(self, loss_model):
        with pytest.raises(NotImplementedError):
            loss_model.aggregate_losses()

    def test__generate_DV_sample(self, loss_model):
        with pytest.raises(NotImplementedError):
            loss_model._generate_DV_sample(None, None)


class TestRepairModel(TestPelicunModel):
    @pytest.fixture
    def repair_model(self, assessment_instance):
        return deepcopy(assessment_instance.repair)

    @pytest.fixture
    def loss_params_A(self):
        return pd.DataFrame(
            (
                (
                    "normal",
                    None,
                    "25704,17136|5,20",
                    0.390923,
                    "USD_2011",
                    0.0,
                    "1 EA",
                ),
                (
                    "normal",
                    0.0,
                    "22.68,15.12|5,20",
                    0.464027,
                    "worker_day",
                    0.0,
                    "1 EA",
                ),
            ),
            index=pd.MultiIndex.from_tuples(
                (("some.test.component", "Cost"), ("some.test.component", "Time"))
            ),
            columns=pd.MultiIndex.from_tuples(
                (
                    ("DS1", "Family"),
                    ("DS1", "LongLeadTime"),
                    ("DS1", "Theta_0"),
                    ("DS1", "Theta_1"),
                    ("DV", "Unit"),
                    ("Incomplete", ""),
                    ("Quantity", "Unit"),
                )
            ),
        )

    def test_init(self, repair_model):
        assert repair_model.log_msg
        assert repair_model.log_div

        assert repair_model.sample is None
        assert repair_model.loss_type == 'Repair'

    def test__create_DV_RVs(self, repair_model, loss_params_A):
        repair_model.loss_params = loss_params_A

        repair_model.loss_map = pd.DataFrame(
            ((("DMG", "some.test.component"), "some.test.component"),),
            columns=("Driver", "Consequence"),
        )

        case_list = pd.MultiIndex.from_tuples(
            (
                ("some.test.component", "1", "1", "0", "0"),
                ("some.test.component", "2", "2", "0", "1"),
                ("some.test.component", "3", "1", "0", "1"),
            ),
            names=("cmp", "loc", "dir", "uid", "ds"),
        )

        rv_reg = repair_model._create_DV_RVs(case_list)
        assert list(rv_reg.RV.keys()) == [
            'Cost-0-1-2-2-0',
            'Time-0-1-2-2-0',
            'Cost-0-1-3-1-0',
            'Time-0-1-3-1-0',
        ]
        rvs = list(rv_reg.RV.values())
        for rv in rvs:
            print(rv.theta)
            assert rv.distribution == 'normal'
        np.testing.assert_array_equal(rvs[0].theta, np.array((1.00, 0.390923, np.nan)))
        np.testing.assert_array_equal(rvs[1].theta, np.array((1.00, 0.464027, np.nan)))
        np.testing.assert_array_equal(rvs[2].theta, np.array((1.00, 0.390923, np.nan)))
        np.testing.assert_array_equal(rvs[3].theta, np.array((1.00, 0.464027, np.nan)))

    def test__calc_median_consequence(self, repair_model, loss_params_A):
        repair_model.loss_params = loss_params_A

        repair_model.loss_map = pd.DataFrame(
            ((("DMG", "some.test.component"), "some.test.component"),),
            columns=("Driver", "Consequence"),
        )

        eco_qnt = pd.DataFrame(
            (
                (10.00, 0.00),
                (0.00, 10.00),
            ),
            columns=pd.MultiIndex.from_tuples(
                (("some.test.component", "0"), ("some.test.component", "1")),
                names=["cmp", "ds"],
            ),
        )

        medians = repair_model._calc_median_consequence(eco_qnt)
        assert medians['Cost'].to_dict() == {(0, '1'): {0: 25704.0, 1: 22848.0}}
        assert medians['Time'].to_dict() == {(0, '1'): {0: 22.68, 1: 20.16}}

    def test__generate_DV_sample(self, repair_model):
        expected_sample = {
            (True, True): {
                (
                    'Cost',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '2',
                    '2',
                    '0',
                ): {0: 25704, 1: 0, 2: 25704, 3: 0},
                (
                    'Cost',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '3',
                    '1',
                    '0',
                ): {0: 0, 1: 0, 2: 0, 3: 25704},
                (
                    'Time',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '2',
                    '2',
                    '0',
                ): {0: 22.68, 1: 0.0, 2: 22.68, 3: 0.0},
                (
                    'Time',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '3',
                    '1',
                    '0',
                ): {0: 0.0, 1: 0.0, 2: 0.0, 3: 22.68},
            },
            (True, False): {
                (
                    'Cost',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '2',
                    '2',
                    '0',
                ): {0: 25704, 1: 0, 2: 25704, 3: 0},
                (
                    'Cost',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '3',
                    '1',
                    '0',
                ): {0: 0, 1: 0, 2: 0, 3: 25704},
                (
                    'Time',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '2',
                    '2',
                    '0',
                ): {0: 22.68, 1: 0.0, 2: 22.68, 3: 0.0},
                (
                    'Time',
                    'some.test.component',
                    'some.test.component',
                    '1',
                    '3',
                    '1',
                    '0',
                ): {0: 0.0, 1: 0.0, 2: 0.0, 3: 22.68},
            },
        }

        for ecods, ecofl in (
            (True, True),
            (True, False),
        ):  # todo: (False, True), (False, False) fails
            assessment_instance = repair_model._asmnt

            assessment_instance.options.eco_scale["AcrossFloors"] = ecofl
            assessment_instance.options.eco_scale["AcrossDamageStates"] = ecods

            dmg_quantities = pd.DataFrame(
                (
                    (0.00, 1.00, 0.00),
                    (1.00, 0.00, 0.00),
                    (0.00, 1.00, 0.00),
                    (0.00, 0.00, 1.00),
                ),
                columns=pd.MultiIndex.from_tuples(
                    (
                        ("some.test.component", "1", "1", "0", "0"),
                        ("some.test.component", "2", "2", "0", "1"),
                        ("some.test.component", "3", "1", "0", "1"),
                    ),
                    names=("cmp", "loc", "dir", "uid", "ds"),
                ),
            )

            repair_model.loss_map = pd.DataFrame(
                ((("DMG", "some.test.component"), "some.test.component"),),
                columns=("Driver", "Consequence"),
            )

            repair_model.loss_params = pd.DataFrame(
                (
                    (
                        None,
                        None,
                        "25704,17136|5,20",
                        0.390923,
                        "USD_2011",
                        0.0,
                        "1 EA",
                    ),
                    (
                        None,
                        0.0,
                        "22.68,15.12|5,20",
                        0.464027,
                        "worker_day",
                        0.0,
                        "1 EA",
                    ),
                ),
                index=pd.MultiIndex.from_tuples(
                    (
                        ("some.test.component", "Cost"),
                        ("some.test.component", "Time"),
                    )
                ),
                columns=pd.MultiIndex.from_tuples(
                    (
                        ("DS1", "Family"),
                        ("DS1", "LongLeadTime"),
                        ("DS1", "Theta_0"),
                        ("DS1", "Theta_1"),
                        ("DV", "Unit"),
                        ("Incomplete", ""),
                        ("Quantity", "Unit"),
                    )
                ),
            )

            repair_model._generate_DV_sample(dmg_quantities, 4)

            assert repair_model.sample.to_dict() == expected_sample[(ecods, ecofl)]

    def test_aggregate_losses(self, repair_model, loss_params_A):
        repair_model.sample = pd.DataFrame(
            ((100.00, 1.00),),
            columns=pd.MultiIndex.from_tuples(
                (
                    (
                        "Cost",
                        "some.test.component",
                        "some.test.component",
                        "1",
                        "1",
                        "1",
                    ),
                    (
                        "Time",
                        "some.test.component",
                        "some.test.component",
                        "1",
                        "1",
                        "1",
                    ),
                ),
                names=("dv", "loss", "dmg", "ds", "loc", "dir"),
            ),
        )

        repair_model.loss_params = loss_params_A

        df_agg = repair_model.aggregate_losses()

        assert df_agg.to_dict() == {
            ('repair_cost', ''): {0: 100.0},
            ('repair_time', 'parallel'): {0: 1.0},
            ('repair_time', 'sequential'): {0: 1.0},
        }


#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.


class TestModelFunctions:
    def test_prep_constant_median_DV(self):
        median = 10.00
        constant_median_DV = model.loss_model.prep_constant_median_DV(median)
        assert constant_median_DV() == median
        values = (1.0, 2.0, 3.0, 4.0, 5.0)
        for value in values:
            assert constant_median_DV(value) == 10.00

    def test_prep_bounded_multilinear_median_DV(self):
        medians = np.array((1.00, 2.00, 3.00, 4.00, 5.00))
        quantities = np.array((0.00, 1.00, 2.00, 3.00, 4.00))
        f = model.loss_model.prep_bounded_multilinear_median_DV(medians, quantities)

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

        result_list = f([2.5, 3.5])
        expected_list = [3.5, 4.5]
        assert np.allclose(result_list, expected_list)

        with pytest.raises(ValueError):
            f(None)
