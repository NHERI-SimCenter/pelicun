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

"""These are unit and integration tests on the asset model of pelicun."""

from __future__ import annotations

import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from pelicun import assessment
from pelicun.base import ensure_value
from pelicun.tests.basic.test_pelicun_model import TestPelicunModel

if TYPE_CHECKING:
    from pelicun.model.asset_model import AssetModel


class TestAssetModel(TestPelicunModel):
    @pytest.fixture
    def asset_model(self, assessment_instance: assessment.Assessment) -> AssetModel:
        return deepcopy(assessment_instance.asset)

    def test_init_method(self, asset_model: AssetModel) -> None:
        assert asset_model.log
        assert asset_model.cmp_marginal_params is None
        assert asset_model.cmp_units is None
        assert asset_model._cmp_RVs is None
        assert asset_model.cmp_sample is None

    def test_save_cmp_sample(self, asset_model: AssetModel) -> None:
        asset_model.cmp_sample = pd.DataFrame(
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
        asset_model.save_cmp_sample(save_units=False)
        asset_model.save_cmp_sample(save_units=True)

    def test_load_cmp_model_1(self, asset_model: AssetModel) -> None:
        cmp_marginals = pd.read_csv(
            'pelicun/tests/basic/data/model/test_AssetModel/CMP_marginals.csv',
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
            ensure_value(asset_model.cmp_marginal_params),
            check_index_type=False,
            check_column_type=False,
            check_dtype=False,
        )

        expected_cmp_units = pd.Series(
            data=['ea'], index=['component_a'], name='Units'
        )

        pd.testing.assert_series_equal(
            expected_cmp_units,
            ensure_value(asset_model.cmp_units),
            check_index_type=False,
        )

    def test_load_cmp_model_2(self, asset_model: AssetModel) -> None:
        # component marginals utilizing the keywords '--', 'all', 'top', 'roof'
        cmp_marginals = pd.read_csv(
            'pelicun/tests/basic/data/model/test_AssetModel/CMP_marginals_2.csv',
            index_col=0,
        )
        asset_model._asmnt.stories = 4
        asset_model.load_cmp_model({'marginals': cmp_marginals})

        assert ensure_value(asset_model.cmp_marginal_params).to_dict() == {
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
            ensure_value(asset_model.cmp_units),
            check_index_type=False,
        )

    def test_load_cmp_model_csv(self, asset_model: AssetModel) -> None:
        # load by directly specifying the csv file
        cmp_marginals = 'pelicun/tests/basic/data/model/test_AssetModel/CMP'
        asset_model.load_cmp_model(cmp_marginals)

    def test_load_cmp_model_exceptions(self, asset_model: AssetModel) -> None:
        cmp_marginals = pd.read_csv(
            'pelicun/tests/basic/data/model/test_AssetModel/'
            'CMP_marginals_invalid_loc.csv',
            index_col=0,
        )
        asset_model._asmnt.stories = 4
        with pytest.raises(
            ValueError, match='Cannot parse location string: basement'
        ):
            asset_model.load_cmp_model({'marginals': cmp_marginals})

        cmp_marginals = pd.read_csv(
            'pelicun/tests/basic/data/model/test_AssetModel/'
            'CMP_marginals_invalid_dir.csv',
            index_col=0,
        )
        asset_model._asmnt.stories = 4
        with pytest.raises(
            ValueError, match='Cannot parse direction string: non-directional'
        ):
            asset_model.load_cmp_model({'marginals': cmp_marginals})

    def test_generate_cmp_sample(self, asset_model: AssetModel) -> None:
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
            ensure_value(asset_model.cmp_sample),
            check_index_type=False,
            check_column_type=False,
        )

    def test_generate_cmp_sample_exceptions_1(self, asset_model: AssetModel) -> None:
        # without marginal parameters
        with pytest.raises(
            ValueError, match='Model parameters have not been specified'
        ):
            asset_model.generate_cmp_sample(sample_size=10)

    def test_generate_cmp_sample_exceptions_2(self, asset_model: AssetModel) -> None:
        # without specifying sample size
        cmp_marginals = pd.read_csv(
            'pelicun/tests/basic/data/model/test_AssetModel/CMP_marginals.csv',
            index_col=0,
        )
        asset_model.load_cmp_model({'marginals': cmp_marginals})
        with pytest.raises(ValueError, match='Sample size was not specified'):
            asset_model.generate_cmp_sample()
        # but it should work if a demand sample is available
        asset_model._asmnt.demand.sample = pd.DataFrame(np.empty(shape=(10, 2)))
        asset_model.generate_cmp_sample()
