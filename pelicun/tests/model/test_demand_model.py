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
These are unit and integration tests on the demand model of pelicun.
"""

import os
import tempfile
import warnings
from copy import deepcopy
import pytest
import numpy as np
import pandas as pd
from pelicun.tests.model.test_model import TestModelModule

# pylint: disable=unused-argument
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-return-doc,missing-return-type-doc


class TestDemandModel(TestModelModule):
    @pytest.fixture
    def demand_model(self, assessment_instance):
        return deepcopy(assessment_instance.demand)

    @pytest.fixture
    def demand_model_with_sample(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel/load_sample/demand_sample_A.csv'
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
            'test_DemandModel/load_sample/demand_sample_B.csv'
        )
        return deepcopy(mdl)

    @pytest.fixture
    def demand_model_with_sample_C(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel/load_sample/demand_sample_C.csv'
        )
        return deepcopy(mdl)

    @pytest.fixture
    def demand_model_with_sample_D(self, assessment_instance):
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/data/model/'
            'test_DemandModel/load_sample/demand_sample_D.csv'
        )
        return deepcopy(mdl)

    def test_init(self, demand_model):
        assert demand_model.log

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
            {
                ('PFA', '0', '1'): [4.029069],
                ('PFA', '1', '1'): [10.084915],
                ('PID', '1', '1'): [0.02672],
                ('SA_0.23', '0', '1'): [8.690585],
            },
            index=[0],
        ).rename_axis(columns=['type', 'loc', 'dir'])
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
            demand_model_with_sample.estimate_RID(demands, params, method='xyz')
            is None
        )

    def test_calibrate_model(
        self, calibrated_demand_model, demand_model_with_sample_C
    ):
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

    # # TODO: this currently fails
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
            demand_model.generate_sample(
                {"SampleSize": 3, 'PreserveRawOrder': False}
            )

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
            {
                ('PFA', '0', '1'): [158.624160, 158.624160, 158.624160],
                ('PFA', '1', '1'): [397.042985, 397.042985, 397.042985],
                ('PID', '1', '1'): [0.02672, 0.02672, 0.02672],
                ('SA_0.23', '0', '1'): [342.148783, 342.148783, 342.148783],
            },
            index=pd.Index([0, 1, 2], dtype='object'),
        ).rename_axis(columns=['type', 'loc', 'dir'])
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
            'test_DemandModel/generate_sample_with_demand_cloning/sample.csv'
        )
        demand_model.calibrate_model(
            {
                "ALL": {
                    "DistributionFamily": "lognormal",
                },
            }
        )
        with warnings.catch_warnings(record=True) as w:
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
        assert len(w) == 1
        assert (
            "The demand cloning configuration lists columns "
            "that are not present in the original demand sample's "
            "columns: ['not_present']."
        ) in str(w[0].message)
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
