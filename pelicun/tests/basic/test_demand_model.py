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

"""These are unit and integration tests on the demand model of pelicun."""

from __future__ import annotations

import tempfile
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from pelicun.base import ensure_value
from pelicun.model.demand_model import (
    DemandModel,
    _assemble_required_demand_data,
    _get_required_demand_type,
)
from pelicun.tests.basic.test_model import TestModelModule

if TYPE_CHECKING:
    from pelicun.assessment import Assessment


class TestDemandModel(TestModelModule):  # noqa: PLR0904
    @pytest.fixture
    def demand_model(self, assessment_instance: Assessment) -> DemandModel:
        return deepcopy(assessment_instance.demand)

    @pytest.fixture
    def demand_model_with_sample(
        self, assessment_instance: Assessment
    ) -> DemandModel:
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/basic/data/model/'
            'test_DemandModel/load_sample/demand_sample_A.csv'
        )
        model_copy = deepcopy(mdl)
        assert isinstance(model_copy, DemandModel)
        return model_copy

    @pytest.fixture
    def calibrated_demand_model(
        self, demand_model_with_sample: DemandModel
    ) -> DemandModel:
        config = {
            'ALL': {
                'DistributionFamily': 'normal',
                'AddUncertainty': 0.00,
            },
            'PID': {
                'DistributionFamily': 'lognormal',
                'TruncateUpper': '0.06',
            },
            'SA': {
                'DistributionFamily': 'empirical',
            },
        }
        demand_model_with_sample.calibrate_model(config)
        model_copy = deepcopy(demand_model_with_sample)
        assert isinstance(model_copy, DemandModel)
        return model_copy

    @pytest.fixture
    def demand_model_with_sample_b(
        self, assessment_instance: Assessment
    ) -> DemandModel:
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/basic/data/model/'
            'test_DemandModel/load_sample/demand_sample_B.csv'
        )
        model_copy = deepcopy(mdl)
        assert isinstance(model_copy, DemandModel)
        return model_copy

    @pytest.fixture
    def demand_model_with_sample_c(
        self, assessment_instance: Assessment
    ) -> DemandModel:
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/basic/data/model/'
            'test_DemandModel/load_sample/demand_sample_C.csv'
        )
        model_copy = deepcopy(mdl)
        assert isinstance(model_copy, DemandModel)
        return model_copy

    @pytest.fixture
    def demand_model_with_sample_d(
        self, assessment_instance: Assessment
    ) -> DemandModel:
        mdl = assessment_instance.demand
        mdl.load_sample(
            'pelicun/tests/basic/data/model/'
            'test_DemandModel/load_sample/demand_sample_D.csv'
        )
        model_copy = deepcopy(mdl)
        assert isinstance(model_copy, DemandModel)
        return model_copy

    def test_init(self, demand_model: DemandModel) -> None:
        assert demand_model.log

        assert demand_model.marginal_params is None
        assert demand_model.correlation is None
        assert demand_model.empirical_data is None
        assert demand_model.user_units is None
        assert demand_model._RVs is None
        assert demand_model.sample is None

    def test_save_sample(self, demand_model_with_sample: DemandModel) -> None:
        # instantiate a temporary directory in memory
        temp_dir = tempfile.mkdtemp()
        # save the sample there
        demand_model_with_sample.save_sample(f'{temp_dir}/temp.csv')
        with Path(f'{temp_dir}/temp.csv').open(encoding='utf-8') as f:
            contents = f.read()
        assert contents == (
            ',PFA-0-1,PFA-1-1,PID-1-1,SA_0.23-0-1\n'
            'Units,inps2,inps2,rad,inps2\n'
            '0,158.62478,397.04389,0.02672,342.149\n'
        )
        res = demand_model_with_sample.save_sample()
        assert isinstance(res, pd.DataFrame)
        assert res.to_dict() == {
            ('PFA', '0', '1'): {0: 158.62478},
            ('PFA', '1', '1'): {0: 397.04389},
            ('PID', '1', '1'): {0: 0.02672},
            ('SA_0.23', '0', '1'): {0: 342.149},
        }

    def test_load_sample(
        self,
        demand_model_with_sample: DemandModel,
        demand_model_with_sample_b: DemandModel,
    ) -> None:
        # retrieve the loaded sample and units
        obtained_sample = ensure_value(demand_model_with_sample.sample)
        obtained_units = ensure_value(demand_model_with_sample.user_units)

        obtained_sample_2 = ensure_value(demand_model_with_sample_b.sample)
        obtained_units_2 = ensure_value(demand_model_with_sample_b.user_units)

        # demand_sample_A.csv and demand_sample_b.csv only differ in the
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

    def test_estimate_RID(self, demand_model_with_sample: DemandModel) -> None:
        demands = ensure_value(demand_model_with_sample.sample)['PID']
        params = {'yield_drift': 0.01}
        res = demand_model_with_sample.estimate_RID(demands, params)
        assert list(res.columns) == [('RID', '1', '1')]
        with pytest.raises(ValueError, match='Invalid method: `xyz`'):
            demand_model_with_sample.estimate_RID(demands, params, method='xyz')

    def test_expand_sample_float(
        self, demand_model_with_sample: DemandModel
    ) -> None:
        sample_before = ensure_value(demand_model_with_sample.sample).copy()
        demand_model_with_sample.expand_sample('test_lab', 1.00, 'unitless')
        sample_after = ensure_value(demand_model_with_sample.sample).copy()
        pd.testing.assert_frame_equal(
            sample_before, sample_after.drop('test_lab', axis=1)
        )
        assert sample_after.loc[0, ('test_lab', '0', '1')] == 1.0

    def test_expand_sample_numpy(
        self, demand_model_with_sample: DemandModel
    ) -> None:
        sample_before = ensure_value(demand_model_with_sample.sample).copy()
        demand_model_with_sample.expand_sample('test_lab', 1.00, 'unitless')
        sample_after = ensure_value(demand_model_with_sample.sample).copy()
        pd.testing.assert_frame_equal(
            sample_before, sample_after.drop('test_lab', axis=1)
        )
        assert sample_after.loc[0, ('test_lab', '0', '1')] == 1.0

    def test_expand_sample_error_no_sample(self, demand_model: DemandModel) -> None:
        with pytest.raises(
            ValueError, match='Demand model does not have a sample yet.'
        ):
            demand_model.expand_sample('test_lab', np.array((1.00,)), 'unitless')

    def test_expand_sample_error_wrong_shape(
        self, demand_model_with_sample: DemandModel
    ) -> None:
        with pytest.raises(ValueError, match='Incompatible array length.'):
            demand_model_with_sample.expand_sample(
                'test_lab', np.array((1.00, 1.00)), 'unitless'
            )

    def test_calibrate_model(
        self,
        calibrated_demand_model: DemandModel,
    ) -> None:
        assert ensure_value(calibrated_demand_model.marginal_params)[
            'Family'
        ].to_list() == [
            'normal',
            'normal',
            'lognormal',
            'empirical',
        ]
        assert (
            ensure_value(calibrated_demand_model.marginal_params).loc[
                ('PID', '1', '1'), 'TruncateUpper'
            ]
            == 0.06
        )

    def test_calibrate_model_censoring(
        self,
        demand_model_with_sample_c: DemandModel,
    ) -> None:
        # with a config featuring censoring the RIDs
        config = {
            'ALL': {
                'DistributionFamily': 'normal',
                'AddUncertainty': 0.00,
            },
            'PID': {
                'DistributionFamily': 'lognormal',
                'CensorUpper': '0.05',
            },
        }
        demand_model_with_sample_c.calibrate_model(config)

    def test_calibrate_model_truncation(
        self,
        demand_model_with_sample_c: DemandModel,
    ) -> None:
        # with a config that specifies a truncation limit smaller than
        # the samples
        config = {
            'ALL': {
                'DistributionFamily': 'normal',
                'AddUncertainty': 0.00,
            },
            'PID': {
                'DistributionFamily': 'lognormal',
                'TruncateUpper': '0.04',
            },
        }
        demand_model_with_sample_c.calibrate_model(config)

    def test_save_load_model_with_empirical(
        self, calibrated_demand_model: DemandModel, assessment_instance: Assessment
    ) -> None:
        # a model that has empirical marginal parameters
        temp_dir = tempfile.mkdtemp()
        calibrated_demand_model.save_model(f'{temp_dir}/temp')
        assert Path(f'{temp_dir}/temp_marginals.csv').exists()
        assert Path(f'{temp_dir}/temp_empirical.csv').exists()
        assert Path(f'{temp_dir}/temp_correlation.csv').exists()

        # Load model to a different DemandModel instance to verify
        new_demand_model = assessment_instance.demand
        new_demand_model.load_model(f'{temp_dir}/temp')
        pd.testing.assert_frame_equal(
            ensure_value(calibrated_demand_model.marginal_params),
            ensure_value(new_demand_model.marginal_params),
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            ensure_value(calibrated_demand_model.correlation),
            ensure_value(new_demand_model.correlation),
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )
        pd.testing.assert_frame_equal(
            ensure_value(calibrated_demand_model.empirical_data),
            ensure_value(new_demand_model.empirical_data),
            atol=1e-4,
            check_index_type=False,
            check_column_type=False,
        )

    def test_save_load_model_without_empirical(
        self,
        demand_model_with_sample_c: DemandModel,
        assessment_instance: Assessment,
    ) -> None:
        # a model that does not have empirical marginal parameters
        temp_dir = tempfile.mkdtemp()
        config = {
            'ALL': {
                'DistributionFamily': 'normal',
                'AddUncertainty': 0.00,
            },
            'PID': {
                'DistributionFamily': 'lognormal',
                'TruncateUpper': '0.04',
            },
        }
        demand_model_with_sample_c.calibrate_model(config)
        demand_model_with_sample_c.save_model(f'{temp_dir}/temp')
        assert Path(f'{temp_dir}/temp_marginals.csv').exists()
        assert Path(f'{temp_dir}/temp_correlation.csv').exists()

        # Load model to a different DemandModel instance to verify
        new_demand_model = assessment_instance.demand
        new_demand_model.load_model(f'{temp_dir}/temp')
        pd.testing.assert_frame_equal(
            ensure_value(demand_model_with_sample_c.marginal_params),
            ensure_value(new_demand_model.marginal_params),
        )
        pd.testing.assert_frame_equal(
            ensure_value(demand_model_with_sample_c.correlation),
            ensure_value(new_demand_model.correlation),
        )
        assert demand_model_with_sample_c.empirical_data is None
        assert new_demand_model.empirical_data is None

    def test_generate_sample_exceptions(self, demand_model: DemandModel) -> None:
        # generating a sample from a non calibrated model should fail
        with pytest.raises(
            ValueError, match='Model parameters have not been specified'
        ):
            demand_model.generate_sample(
                {'SampleSize': 3, 'PreserveRawOrder': False}
            )

    def test_generate_sample(self, calibrated_demand_model: DemandModel) -> None:
        calibrated_demand_model.generate_sample(
            {'SampleSize': 3, 'PreserveRawOrder': False}
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

    def test_generate_sample_with_demand_cloning(
        self, assessment_instance: Assessment
    ) -> None:
        # # used for debugging
        # assessment_instance = assessment.Assessment()

        demand_model = assessment_instance.demand

        mdl = assessment_instance.demand
        # contains PGV-0-1, PGV-1-1, PGV-2-1, and PGA-0-1
        # PGA-0-1 is not cloned.
        mdl.load_sample(
            'pelicun/tests/basic/data/model/'
            'test_DemandModel/generate_sample_with_demand_cloning/sample.csv'
        )
        demand_model.calibrate_model(
            {
                'ALL': {
                    'DistributionFamily': 'lognormal',
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
            'The demand cloning configuration lists columns '
            "that are not present in the original demand sample's "
            "columns: ['not_present']."
        ) in str(w[0].message)
        # we'll just get a warning for the `not_present` entry
        assert ensure_value(demand_model.sample).columns.to_list() == [
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
            demand_model.sample['PGV', '0', '1'].values,  # type: ignore
            demand_model.sample['PGV', '0', '3'].values,  # type: ignore
        )
        # exceptions
        # Duplicate entries in demand cloning configuration
        with pytest.raises(
            ValueError, match='Duplicate entries in demand cloning configuration.'
        ):
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

    def test__get_required_demand_type(
        self, assessment_instance: Assessment
    ) -> None:
        # Simple case: single demand
        damage_model = assessment_instance.damage
        cmp_set = {'testing.component'}
        damage_model.load_model_parameters(
            [
                'pelicun/tests/basic/data/model/test_DemandModel/'
                '_get_required_demand_type/damage_db_testing_single.csv'
            ],
            cmp_set,
        )
        pgb = pd.DataFrame(
            {('testing.component', '1', '1', '1'): [1]}, index=['Blocks']
        ).T.rename_axis(index=['cmp', 'loc', 'dir', 'uid'])
        demand_offset = {'PFA': 0}
        required = _get_required_demand_type(
            ensure_value(damage_model.ds_model.damage_params), pgb, demand_offset
        )
        expected = defaultdict(
            list,
            {(('PID-1-1',), None): [('testing.component', '1', '1', '1')]},
        )
        assert required == expected

        # Utility demand case: two demands are required
        damage_model = assessment_instance.damage
        cmp_set = {'testing.component2'}
        damage_model.load_model_parameters(
            [
                'pelicun/tests/basic/data/model/test_DemandModel/'
                '_get_required_demand_type/damage_db_testing_utility.csv'
            ],
            cmp_set,
        )
        pgb = pd.DataFrame(
            {('testing.component2', '1', '1', '1'): [1]}, index=['Blocks']
        ).T.rename_axis(index=['cmp', 'loc', 'dir', 'uid'])
        demand_offset = {'PFA': 0}
        required = _get_required_demand_type(
            ensure_value(damage_model.ds_model.damage_params), pgb, demand_offset
        )
        expected = defaultdict(
            list,
            {
                (('PID-1-1', 'PFA-1-1'), 'sqrt(X1^2+X2^2)'): [  # type: ignore
                    ('testing.component2', '1', '1', '1')
                ]
            },
        )
        assert required == expected

    def test__assemble_required_demand_data(
        self, assessment_instance: Assessment
    ) -> None:
        # Utility demand case: two demands are required
        damage_model = assessment_instance.damage
        cmp_set = {'testing.component'}
        damage_model.load_model_parameters(
            [
                'pelicun/tests/basic/data/model/test_DemandModel/'
                '_get_required_demand_type/damage_db_testing_single.csv'
            ],
            cmp_set,
        )
        required_edps = defaultdict(
            list,
            {
                (('PID-1-1', 'PFA-1-1'), 'sqrt(X1^2+X2^2)'): [
                    ('testing.component', '1', '1', '1')
                ]
            },
        )
        nondirectional_multipliers = {'ALL': 1.00}
        demand_sample = pd.DataFrame(
            {
                ('PID', '1', '1'): np.full(5, 3.00),
                ('PFA', '1', '1'): np.full(5, 4.00),
            }
        )
        demand_data = _assemble_required_demand_data(
            required_edps,  # type: ignore
            nondirectional_multipliers,
            demand_sample,
        )
        expected = {
            (('PID-1-1', 'PFA-1-1'), 'sqrt(X1^2+X2^2)'): np.array(
                [5.0, 5.0, 5.0, 5.0, 5.0]
            )
        }
        assert demand_data.keys() == expected.keys()
        for key in demand_data:
            assert np.all(demand_data[key] == expected[key])
