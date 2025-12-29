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

"""These are unit and integration tests on the loss model of pelicun."""

from __future__ import annotations

import re
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from pelicun import file_io, model, uq
from pelicun.base import ensure_value
from pelicun.model.loss_model import (
    LossModel,
    RepairModel_DS,
    RepairModel_LF,
    _is_for_ds_model,
    _is_for_lf_model,
)
from pelicun.pelicun_warnings import PelicunWarning
from pelicun.tests.basic.test_pelicun_model import TestPelicunModel

if TYPE_CHECKING:
    from pelicun.assessment import Assessment
    from pelicun.model.asset_model import AssetModel


class TestLossModel(TestPelicunModel):
    @pytest.fixture
    def loss_model(self, assessment_instance: Assessment) -> LossModel:
        return deepcopy(assessment_instance.loss)

    @pytest.fixture
    def asset_model_empty(self, assessment_instance: Assessment) -> AssetModel:
        return deepcopy(assessment_instance.asset)

    @pytest.fixture
    def asset_model_a(self, asset_model_empty: AssetModel) -> AssetModel:
        asset = deepcopy(asset_model_empty)
        asset.cmp_marginal_params = pd.DataFrame(
            {
                ('Theta_0'): [1.0, 1.0, 1.0],
                ('Blocks'): [1, 1, 1],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.A', '1', '1', '0'),
                    ('cmp.B', '1', '1', '0'),
                    ('cmp.C', '1', '1', '0'),
                ]
            ),
        ).rename_axis(index=['cmp', 'loc', 'dir', 'uid'])
        asset.generate_cmp_sample(sample_size=10)
        return asset

    @pytest.fixture
    def loss_model_with_ones(self, assessment_instance: Assessment) -> LossModel:
        loss_model = assessment_instance.loss

        # add artificial values to the samples
        data_ds = {}
        for (
            decision_variable,
            consequence,
            component,
            damage_state,
            location,
            direction,
            uid,
        ) in product(
            ('Cost', 'Carbon'),
            ('cmp.A.consequence', 'cmp.B.consequence'),
            ('cmp.A', 'cmp.B'),
            ('DS1', 'DS2'),
            ('1', '2'),  # loc
            ('1', '2'),  # dir
            ('uid1', 'uid2'),
        ):
            data_ds[
                decision_variable,
                consequence,
                component,
                damage_state,
                location,
                direction,
                uid,
            ] = [1.00, 1.00, 1.00]
        loss_model.ds_model.sample = pd.DataFrame(data_ds).rename_axis(
            columns=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'uid']
        )
        data_lf = {}
        for (
            decision_variable,
            consequence,
            component,
            location,
            direction,
            uid,
        ) in product(
            ('Cost', 'Carbon'),
            ('cmp.A.consequence', 'cmp.B.consequence'),
            ('cmp.A', 'cmp.B'),
            ('1', '2'),  # loc
            ('1', '2'),  # dir
            ('uid1', 'uid2'),
        ):
            data_lf[
                decision_variable,
                consequence,
                component,
                location,
                direction,
                uid,
            ] = [1.00, 1.00, 1.00]
        loss_model.lf_model.sample = pd.DataFrame(data_lf).rename_axis(
            columns=['dv', 'loss', 'dmg', 'loc', 'dir', 'uid']
        )

        return loss_model

    def test___init__(self, loss_model: LossModel) -> None:
        assert loss_model.log
        assert loss_model.ds_model
        with pytest.raises(AttributeError):
            loss_model.xyz = 123  # type: ignore

        assert loss_model.ds_model.loss_params is None
        assert loss_model.ds_model.sample is None

        assert len(loss_model._loss_models) == 2

    def test_decision_variables(self, loss_model: LossModel) -> None:
        dvs = ('Cost', 'Time')
        assert loss_model.decision_variables == dvs
        assert loss_model.ds_model.decision_variables == dvs
        assert loss_model.lf_model.decision_variables == dvs

    def test_add_loss_map(
        self, loss_model: LossModel, asset_model_a: AssetModel
    ) -> None:
        loss_model._asmnt.asset = asset_model_a

        loss_map = pd.DataFrame(
            {
                'Repair': ['consequence.A', 'consequence.B'],
            },
            index=['cmp.A', 'cmp.B'],
        )
        loss_model.add_loss_map(loss_map)
        pd.testing.assert_frame_equal(ensure_value(loss_model._loss_map), loss_map)
        for contained_model in loss_model._loss_models:
            pd.testing.assert_frame_equal(
                ensure_value(contained_model.loss_map), loss_map
            )

    def test_load_model_parameters(
        self, loss_model: LossModel, asset_model_a: AssetModel
    ) -> None:
        loss_model._asmnt.asset = asset_model_a
        loss_model.decision_variables = ('my_RV',)
        loss_map = pd.DataFrame(
            {
                'Repair': ['consequence.A', 'consequence.B', 'consequence.F'],
            },
            index=['cmp.A', 'cmp.B', 'cmp.F'],
        )
        loss_model.add_loss_map(loss_map)
        # consequence.A will be for the DS model
        # consequence.B will be for the LF model
        # consequence.C will have no loss parameters defined for it
        # consequence.D should be removed from the DS parameters
        # consequence.E should be removed from the LF parameters
        # consequence.F should be missing
        ds_loss_parameters = pd.DataFrame(
            {
                ('Quantity', 'Unit'): ['1 EA'] * 2,
                ('DV', 'Unit'): ['1 EA'] * 2,
                ('DS1', 'Theta_0'): ['0.00,1.00|0.00,1.00'] * 2,
            },
            index=pd.MultiIndex.from_tuples(
                [('consequence.A', 'my_RV'), ('consequence.D', 'my_RV')]
            ),
        )
        lf_loss_parameters = pd.DataFrame(
            {
                ('Quantity', 'Unit'): ['1 EA'] * 2,
                ('DV', 'Unit'): ['1 EA'] * 2,
                ('Demand', 'Unit'): ['1 EA'] * 2,
                ('LossFunction', 'Theta_0'): ['0.00,1.00|0.00,1.00'] * 2,
            },
            index=pd.MultiIndex.from_tuples(
                [('consequence.B', 'my_RV'), ('consequence.E', 'my_RV')]
            ),
        )
        with pytest.warns(PelicunWarning) as record:
            loss_model.load_model_parameters(
                [ds_loss_parameters, lf_loss_parameters]
            )

        # assert len(record) == 1
        # TODO(JVM): re-enable the line above once we address other
        # warnings, and change indexing to [0] below.

        assert (
            'The loss model does not provide loss information '
            'for the following component(s) in the asset '
            "model: [('consequence.F', 'my_RV')]."
        ) in str(record[-1].message)

    def test__loss_models(self, loss_model: LossModel) -> None:
        models = loss_model._loss_models
        assert len(models) == 2
        assert isinstance(models[0], RepairModel_DS)
        assert isinstance(models[1], RepairModel_LF)

    def test__loss_map(self, loss_model: LossModel) -> None:
        loss_map = pd.DataFrame(
            {
                'Repair': ['consequence_A', 'consequence_B'],
            },
            index=['cmp_A', 'cmp_B'],
        )
        # test setter
        loss_model._loss_map = loss_map
        # test getter
        pd.testing.assert_frame_equal(ensure_value(loss_model._loss_map), loss_map)
        for contained_model in loss_model._loss_models:
            pd.testing.assert_frame_equal(
                ensure_value(contained_model.loss_map), loss_map
            )

    def test__missing(self, loss_model: LossModel) -> None:
        missing = {
            ('missing.component', 'Time'),
            ('missing.component', 'Energy'),
        }
        # test setter
        loss_model._missing = missing
        # test getter
        assert loss_model._missing == missing
        for contained_model in loss_model._loss_models:
            assert contained_model.missing == missing

    def test__ensure_loss_parameter_availability(
        self, assessment_instance: Assessment
    ) -> None:
        loss_model = LossModel(assessment_instance)

        # Only consider `DecisionVariableXYZ`
        loss_model.decision_variables = ('DecisionVariableXYZ',)

        # A, B should be in the ds model
        # C, D should be in the lf model
        # E should be missing

        loss_map = pd.DataFrame(
            {
                'Repair': [f'consequence_{x}' for x in ('A', 'B', 'C', 'D', 'E')],
            },
            index=[f'cmp_{x}' for x in ('A', 'B', 'C', 'D', 'E')],
        )

        loss_model._loss_map = loss_map

        loss_model.ds_model.loss_params = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    ('consequence_A', 'DecisionVariableXYZ'),
                    ('consequence_B', 'DecisionVariableXYZ'),
                ]
            )
        )
        loss_model.lf_model.loss_params = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [
                    ('consequence_C', 'DecisionVariableXYZ'),
                    ('consequence_D', 'DecisionVariableXYZ'),
                ]
            )
        )

        with pytest.warns(PelicunWarning) as record:
            loss_model._ensure_loss_parameter_availability()
            missing = loss_model._missing
        assert missing == {('consequence_E', 'DecisionVariableXYZ')}
        assert len(record) == 1
        assert (
            'The loss model does not provide loss information '
            'for the following component(s) in the asset model: '
            "[('consequence_E', 'DecisionVariableXYZ')]"
        ) in str(record[0].message)

    def test_aggregate_losses_when_no_loss(
        self, assessment_instance: Assessment
    ) -> None:
        # tests that aggregate losses works when there is no loss.
        loss_model = LossModel(assessment_instance)
        loss_model.decision_variables = ('Cost', 'Time', 'Carbon', 'Energy')
        df_agg = loss_model.aggregate_losses()
        assert isinstance(df_agg, pd.DataFrame)
        pd.testing.assert_frame_equal(
            df_agg,
            pd.DataFrame(
                {
                    'repair_cost': 0.00,
                    'repair_carbon': 0.0,
                    'repair_energy': 0.00,
                    'repair_time-sequential': 0.00,
                    'repair_time-parallel': 0.00,
                },
                index=[0],
            ),
        )

    def test__apply_consequence_scaling(
        self, loss_model_with_ones: LossModel
    ) -> None:
        # When only `dv` is provided
        scaling_conditions = {'dv': 'Cost'}
        scaling_factor = 2.00

        loss_model_with_ones._apply_consequence_scaling(
            scaling_conditions, scaling_factor
        )

        for loss_model in loss_model_with_ones._loss_models:
            assert loss_model.sample is not None
            mask = loss_model.sample.columns.get_level_values('dv') == 'Cost'
            assert np.all(loss_model.sample.iloc[:, mask] == 2.00)
            assert np.all(loss_model.sample.iloc[:, ~mask] == 1.00)
            loss_model.sample.iloc[:, :] = 1.00

        scaling_conditions = {'dv': 'Carbon', 'loc': '1', 'uid': 'uid2'}
        scaling_factor = 2.00
        loss_model_with_ones._apply_consequence_scaling(
            scaling_conditions, scaling_factor
        )

        for loss_model in loss_model_with_ones._loss_models:
            assert loss_model.sample is not None
            mask = np.full(len(loss_model.sample.columns), fill_value=True)
            mask &= loss_model.sample.columns.get_level_values('dv') == 'Carbon'
            mask &= loss_model.sample.columns.get_level_values('loc') == '1'
            mask &= loss_model.sample.columns.get_level_values('uid') == 'uid2'
            assert np.all(loss_model.sample.iloc[:, mask] == 2.00)
            assert np.all(loss_model.sample.iloc[:, ~mask] == 1.00)

    def test_aggregate_losses_combination(
        self, assessment_instance: Assessment
    ) -> None:
        # The test sets up a very simple loss calculation from
        # scratch, only defining essential parameters.

        # demand
        sample_size = 5
        demand_marginal_parameters = pd.DataFrame(
            {
                ('PIH', '0', '1'): ['in', 7.00],
                ('PWS', '0', '1'): ['mph', 50.0],
            },
            index=['Units', 'Theta_0'],
        ).T
        perfect_corr = pd.DataFrame(
            np.ones((2, 2)),
            columns=demand_marginal_parameters.index,
            index=demand_marginal_parameters.index,
        )
        assessment_instance.demand.load_model(
            {'marginals': demand_marginal_parameters, 'correlation': perfect_corr}
        )
        assessment_instance.demand.generate_sample({'SampleSize': sample_size})

        # asset
        assessment_instance.asset.cmp_marginal_params = pd.DataFrame(
            {
                'Theta_0': (1.0, 1.0),
            },
            index=pd.MultiIndex.from_tuples(
                (('wind.comp', '0', '1', '0'), ('flood.comp', '0', '1', '0')),
                names=('cmp', 'loc', 'dir', 'uid'),
            ),
        )
        assessment_instance.asset.generate_cmp_sample()

        # no damage estimation needed since we only use loss functions

        # loss

        assessment_instance.loss.decision_variables = ('Cost',)
        assessment_instance.loss.add_loss_map(loss_map_policy='fill')
        assessment_instance.loss.load_model_parameters(
            [
                (
                    'pelicun/tests/basic/data/model/'
                    'test_LossModel/loss_function_wind.csv'
                ),
                (
                    'pelicun/tests/basic/data/model/'
                    'test_LossModel/loss_function_flood.csv'
                ),
            ]
        )

        assessment_instance.loss.calculate()

        # individual losses
        l1, l2 = ensure_value(assessment_instance.loss.lf_model.sample).iloc[0, :]
        # combined loss, result of interpolation
        l_comb = 0.904

        file_path = file_io.substitute_default_path(
            ['PelicunDefault/Hazus Hurricane Wind/combine_wind_flood.csv']
        )[0]
        assert isinstance(file_path, str)
        combination_array = pd.read_csv(
            file_path,
            index_col=None,
            header=None,
        ).to_numpy()
        loss_combination = {
            'Cost': {
                ('wind.comp', 'flood.comp'): combination_array,
            },
        }

        agg_df, _ = assessment_instance.loss.aggregate_losses(
            loss_combination=loss_combination, future=True
        )
        assert isinstance(agg_df, pd.DataFrame)
        pd.testing.assert_frame_equal(
            agg_df, pd.DataFrame([l_comb] * 5, columns=['repair_cost'])
        )

        # verify interpolation with some manual checks
        lower, higher = combination_array[8:10, 4]
        assert lower <= l_comb <= higher
        assert l2 == combination_array[0, 4]
        assert combination_array[8, 0] <= l1 <= combination_array[9, 0]

    def test_aggregate_losses_thresholds(
        self, loss_model_with_ones: LossModel
    ) -> None:
        # Row 0 has the value of 1.0 in all columns.
        # Adjust rows 1 and 2 to have the values 2.0 and 3.0, for
        # testing.
        assert loss_model_with_ones.ds_model.sample is not None
        assert loss_model_with_ones.lf_model.sample is not None
        loss_model_with_ones.decision_variables = ('Cost', 'Carbon')
        loss_model_with_ones.dv_units = {'Cost': 'USD_2011', 'Carbon': 'kg'}
        loss_model_with_ones.ds_model.sample.iloc[1, :] = 2.00
        loss_model_with_ones.ds_model.sample.iloc[2, :] = 3.00
        loss_model_with_ones.lf_model.sample.iloc[1, :] = 2.00
        loss_model_with_ones.lf_model.sample.iloc[2, :] = 3.00
        # Instantiate a RandomVariableRegistry to pass as an argument
        # to the method.
        rv_reg = uq.RandomVariableRegistry(loss_model_with_ones._asmnt.options.rng)
        # Add a threshold for `Cost`
        rv_reg.add_RV(
            uq.rv_class_map('deterministic')(name='Cost', theta=np.array((400.00,)))  # type: ignore
        )
        # Add a threshold for `Carbon`
        rv_reg.add_RV(
            uq.rv_class_map('deterministic')(
                name='Carbon',
                theta=np.array((100.00,)),  # type: ignore
            )
        )
        df_agg, exceedance_bool_df = loss_model_with_ones.aggregate_losses(
            replacement_configuration=(rv_reg, {'Cost': 0.50, 'Carbon': 1.00}),
            future=True,
        )
        assert isinstance(df_agg, pd.DataFrame)
        assert isinstance(exceedance_bool_df, pd.DataFrame)
        df_agg_expected = pd.DataFrame(
            {
                'repair_carbon': [96.00, 100.00, 100.00],
                'repair_cost': [96.00, 400.00, 400.00],
            }
        )
        exceedance_bool_df_expected = pd.DataFrame(
            {'Cost': [False, False, True], 'Carbon': [False, True, True]}
        )
        pd.testing.assert_frame_equal(df_agg, df_agg_expected)
        pd.testing.assert_frame_equal(
            exceedance_bool_df, exceedance_bool_df_expected
        )

    def test_consequence_scaling(self, loss_model_with_ones: LossModel) -> None:
        loss_model_with_ones.consequence_scaling(
            'pelicun/tests/basic/data/model/test_LossModel/scaling_specification.csv'
        )

        expected_ds = (
            pd.read_csv(
                'pelicun/tests/basic/data/model/test_LossModel/scaled_losses_ds.csv',
                dtype={
                    'dv': str,
                    'loss': str,
                    'dmg': str,
                    'ds': str,
                    'loc': str,
                    'dir': str,
                    'uid': str,
                },
            )
            .set_index(['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'uid'])
            .T.astype(float)
        )
        expected_ds.index = pd.RangeIndex(range(len(expected_ds)))  # type: ignore
        pd.testing.assert_frame_equal(
            loss_model_with_ones.ds_model.sample,  # type: ignore
            expected_ds,
        )

        expected_lf = (
            pd.read_csv(
                'pelicun/tests/basic/data/model/test_LossModel/scaled_losses_lf.csv',
                dtype={
                    'dv': str,
                    'loss': str,
                    'dmg': str,
                    'loc': str,
                    'dir': str,
                    'uid': str,
                },
            )
            .set_index(['dv', 'loss', 'dmg', 'loc', 'dir', 'uid'])
            .T.astype(float)
        )
        expected_lf.index = pd.RangeIndex(range(len(expected_lf)))  # type: ignore
        pd.testing.assert_frame_equal(
            loss_model_with_ones.lf_model.sample,  # type: ignore
            expected_lf,
        )


class TestRepairModel_Base(TestPelicunModel):
    def test___init__(self, assessment_instance: Assessment) -> None:
        repair_model = RepairModel_DS(assessment_instance)
        with pytest.raises(AttributeError):
            repair_model.xyz = 123  # type: ignore

    def test_drop_unused_loss_parameters(
        self, assessment_instance: Assessment
    ) -> None:
        base_model = RepairModel_DS(assessment_instance)
        loss_map = pd.DataFrame(
            {
                'Repair': ['consequence_A', 'consequence_B'],
            },
            index=['cmp_A', 'cmp_B'],
        )
        # without loss_params, it should do nothing
        base_model.drop_unused_loss_parameters(loss_map)
        base_model.loss_params = pd.DataFrame(
            index=[f'consequence_{x}' for x in ('A', 'B', 'C', 'D')]
        )
        base_model.drop_unused_loss_parameters(loss_map)
        pd.testing.assert_frame_equal(
            base_model.loss_params,
            pd.DataFrame(index=[f'consequence_{x}' for x in ('A', 'B')]),
        )

    def test__remove_incomplete_components(
        self, assessment_instance: Assessment
    ) -> None:
        base_model = RepairModel_DS(assessment_instance)
        # without loss_params, it should do nothing
        base_model.remove_incomplete_components()
        # without incomplete, it should do nothing
        loss_params = pd.DataFrame(
            index=[f'consequence_{x}' for x in ('A', 'B', 'C', 'D')]
        )
        base_model.loss_params = loss_params
        base_model.remove_incomplete_components()
        pd.testing.assert_frame_equal(
            base_model.loss_params,
            loss_params,
        )
        base_model.loss_params = pd.DataFrame(
            {('Incomplete', ''): [0, 0, 0, 1]},
            index=[f'consequence_{x}' for x in ('A', 'B', 'C', 'D')],
        )
        # Now entry D should be gone
        base_model.remove_incomplete_components()
        pd.testing.assert_frame_equal(
            base_model.loss_params,
            pd.DataFrame(
                {('Incomplete', ''): [0, 0, 0]},
                index=[f'consequence_{x}' for x in ('A', 'B', 'C')],
            ),
        )

    def test__get_available(self, assessment_instance: Assessment) -> None:
        base_model = RepairModel_DS(assessment_instance)
        base_model.loss_params = pd.DataFrame(index=['cmp.A', 'cmp.B', 'cmp.C'])
        assert base_model.get_available() == {'cmp.A', 'cmp.B', 'cmp.C'}


class TestRepairModel_DS(TestRepairModel_Base):
    def test_convert_loss_parameter_units(
        self, assessment_instance: Assessment
    ) -> None:
        ds_model = RepairModel_DS(assessment_instance)
        ds_model.loss_params = pd.DataFrame(
            {
                ('Quantity', 'Unit'): ['1 test_two', '1 EA'],
                ('DV', 'Unit'): ['test_three', 'test_three'],
                ('DS1', 'Theta_0'): ['200.00,100.00|10.00,20.00', '100.00'],
                ('DS1', 'Theta_1'): [0.20, None],
                ('DS1', 'Family'): ['lognormal', None],
            },
            index=pd.MultiIndex.from_tuples([('cmpA', 'Cost'), ('cmpB', 'Cost')]),
        )

        ds_model.convert_loss_parameter_units()

        # DVs are scaled by 3/2, quantities by 2
        pd.testing.assert_frame_equal(
            ds_model.loss_params,
            pd.DataFrame(
                {
                    ('Quantity', 'Unit'): ['1 test_two', '1 EA'],
                    ('DV', 'Unit'): ['test_three', 'test_three'],
                    ('DS1', 'Theta_0'): ['300,150|20,40', 300.0],
                    ('DS1', 'Theta_1'): [0.20, None],
                    ('DS1', 'Family'): ['lognormal', None],
                },
                index=pd.MultiIndex.from_tuples(
                    [('cmpA', 'Cost'), ('cmpB', 'Cost')]
                ),
            ),
        )

    def test__drop_unused_damage_states(
        self, assessment_instance: Assessment
    ) -> None:
        ds_model = RepairModel_DS(assessment_instance)
        loss_params = pd.DataFrame(
            {
                ('DS1', 'Theta_0'): [1.0, 1.0, 1.0, 1.0],
                ('DS2', 'Theta_0'): [1.0, 1.0, 1.0, None],
                ('DS3', 'Theta_0'): [1.0, 1.0, None, None],
                ('DS4', 'Theta_0'): [1.0, None, None, None],
                ('DS5', 'Theta_0'): [None, None, None, None],
                ('DS6', 'Theta_0'): [None, None, None, None],
                ('DS7', 'Theta_0'): [None, None, None, None],
            }
        )
        ds_model.loss_params = loss_params
        ds_model.drop_unused_damage_states()
        pd.testing.assert_frame_equal(ds_model.loss_params, loss_params.iloc[:, 0:4])

    def test__create_DV_RVs(self, assessment_instance: Assessment) -> None:
        assessment_instance.options.rho_cost_time = 0.30
        ds_model = RepairModel_DS(assessment_instance)
        ds_model.decision_variables = ('Cost', 'Time')
        ds_model.missing = {('cmp.B', 'Cost'), ('cmp.B', 'Time')}
        ds_model.loss_map = pd.DataFrame(
            {
                'Repair': ['cmp.A', 'cmp.B', 'cmp.C', 'cmp.D', 'cmp.E'],
            },
            index=['cmp.A', 'cmp.B', 'cmp.C', 'cmp.D', 'cmp.E'],
        )
        # cmp.B is marked as missing, cmp.C is intended for the LF
        # model.
        # cmp.D has `|` in Theta_0 which should be treated as 1.00
        # cmp.E has deterministic loss.
        ds_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA', '1 EA', '1 EA', '1 EA'],
                ('Quantity', 'Unit'): ['1 EA', '1 EA', '1 EA', '1 EA'],
                ('DS1', 'Family'): ['normal', 'normal', 'normal', None],
                ('DS1', 'Theta_0'): [1.00, 1.00, '4.0,2.0|5.0,1.0', 1.00],
                ('DS1', 'Theta_1'): [1.00, 1.00, 1.00, None],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.A', 'Cost'),
                    ('cmp.A', 'Time'),
                    ('cmp.D', 'Cost'),
                    ('cmp.E', 'Cost'),
                ]
            ),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])

        cases = pd.MultiIndex.from_tuples(
            [
                ('cmp.A', '0', '1', '0', '1'),
                ('cmp.B', '0', '1', '0', '1'),  # marked as missing
                ('cmp.C', '0', '1', '0', '1'),  # no loss parameters
                ('cmp.D', '0', '1', '0', '1'),  # `|` in Theta_0
                ('cmp.E', '0', '1', '0', '1'),  # Deterministic loss
            ],
            names=['cmp', 'loc', 'dir', 'uid', 'ds'],
        )
        rv_reg = ds_model._create_DV_RVs(cases)
        assert rv_reg is not None
        for key in (
            'Cost-cmp.A-1-0-1-0',
            'Time-cmp.A-1-0-1-0',
            'Cost-cmp.D-1-0-1-0',
        ):
            assert key in rv_reg.RV
        assert len(rv_reg.RV) == 3
        assert isinstance(rv_reg.RV['Cost-cmp.A-1-0-1-0'], uq.NormalRandomVariable)
        assert isinstance(rv_reg.RV['Time-cmp.A-1-0-1-0'], uq.NormalRandomVariable)
        assert isinstance(rv_reg.RV['Cost-cmp.D-1-0-1-0'], uq.NormalRandomVariable)
        assert np.all(
            rv_reg.RV['Cost-cmp.A-1-0-1-0'].theta[0:2] == np.array((1.0, 1.0))  # type: ignore
        )
        assert np.all(
            rv_reg.RV['Time-cmp.A-1-0-1-0'].theta[0:2] == np.array((1.0, 1.0))  # type: ignore
        )
        assert np.all(
            rv_reg.RV['Cost-cmp.D-1-0-1-0'].theta[0:2] == np.array([1.0, 1.0])  # type: ignore
        )
        assert 'DV-cmp.A-1-0-1-0_set' in rv_reg.RV_set
        np.all(
            rv_reg.RV_set['DV-cmp.A-1-0-1-0_set'].Rho()
            == np.array(((1.0, 0.3), (0.3, 1.0)))
        )
        assert len(rv_reg.RV_set) == 1

    def test__create_DV_RVs_all_deterministic(
        self, assessment_instance: Assessment
    ) -> None:
        ds_model = RepairModel_DS(assessment_instance)
        ds_model.decision_variables = ('myRV',)
        ds_model.missing = set()
        ds_model.loss_map = pd.DataFrame(
            {'Repair': ['cmp.A']},
            index=['cmp.A'],
        )
        ds_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA'],
                ('Quantity', 'Unit'): ['1 EA'],
                ('DS1', 'Family'): [None],
                ('DS1', 'Theta_0'): [1.00],
            },
            index=pd.MultiIndex.from_tuples([('cmp.A', 'myRV')]),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])

        cases = pd.MultiIndex.from_tuples(
            [('cmp.A', '0', '1', '0', '1')],
            names=['cmp', 'loc', 'dir', 'uid', 'ds'],
        )
        rv_reg = ds_model._create_DV_RVs(cases)

        assert rv_reg is None

    def test__calc_median_consequence_no_locs(
        self, assessment_instance: Assessment
    ) -> None:
        # Test the method when the eco_qnt dataframe's columns do not
        # contain `loc` information.

        ds_model = RepairModel_DS(assessment_instance)
        eco_qnt = pd.DataFrame(
            {
                ('cmp.A', '0'): [0.00, 0.00, 1.00],
                ('cmp.A', '1'): [1.00, 0.00, 0.00],
                ('cmp.A', '2'): [0.00, 1.00, 0.00],
                ('cmp.B', '1'): [0.00, 1.00, 0.00],
                ('cmp.B', '2'): [1.00, 0.00, 0.00],
            }
        ).rename_axis(columns=['cmp', 'ds'])
        ds_model.decision_variables = ('my_DV',)
        # cmp.A should be available and we should get medians.
        # missing_cmp will be marked as missing
        # is_for_LF_model represents a component->consequence pair
        # that is intended for processing by the loss function model
        # and should be ignored by the damage state model.
        ds_model.loss_map = pd.DataFrame(
            {
                'Repair': ['cmp.A', 'cmp.B', 'missing_cmp', 'is_for_LF_model'],
            },
            index=['cmp.A', 'cmp.B', 'missing_consequence', 'LF_consequence'],
        )

        # DS3 is in the loss parameters but has not been triggered.
        ds_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA', '1 EA'],
                ('Quantity', 'Unit'): ['1 EA', '1 EA'],
                ('DS1', 'Family'): [None, 'normal'],
                ('DS1', 'Theta_0'): [100.00, 12345.00],
                ('DS1', 'Theta_1'): [None, 0.30],
                ('DS2', 'Family'): [None, 'normal'],
                ('DS2', 'Theta_0'): [200.00, '2.00,1.00|5.00,10.00'],
                ('DS2', 'Theta_1'): [None, 0.30],
                ('DS3', 'Family'): [None, 'normal'],
                ('DS3', 'Theta_0'): [200.00, '2.00,1.00|5.00,10.00'],
                ('DS3', 'Theta_1'): [None, 0.30],
            },
            index=pd.MultiIndex.from_tuples(
                [('cmp.A', 'my_DV'), ('cmp.B', 'my_DV')]
            ),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])
        ds_model.missing = {('missing_cmp', 'my_DV')}
        medians = ds_model._calc_median_consequence(eco_qnt)
        assert len(medians) == 1
        assert 'my_DV' in medians
        pd.testing.assert_frame_equal(
            medians['my_DV'],
            pd.DataFrame(
                {
                    ('cmp.A', '1'): [100.00, 100.00, 100.00],
                    ('cmp.A', '2'): [200.00, 200.00, 200.00],
                    ('cmp.B', '1'): [1.00, 1.00, 1.00],
                    ('cmp.B', '2'): [2.00, 2.00, 2.00],
                }
            ).rename_axis(columns=['cmp', 'ds']),
        )

        #
        # edge cases
        #

        # random variable not supported
        ds_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA'],
                ('Quantity', 'Unit'): ['1 EA'],
                ('DS1', 'Family'): ['multilinear_CDF'],
                ('DS1', 'Theta_0'): ['0.00,1.00|0.00,1.00'],
                ('DS1', 'Theta_1'): [0.30],
            },
            index=pd.MultiIndex.from_tuples([('cmp.A', 'my_DV')]),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])
        with pytest.raises(
            ValueError,
            match='Loss Distribution of type multilinear_CDF not supported.',
        ):
            ds_model._calc_median_consequence(eco_qnt)

    def test__calc_median_consequence_locs(
        self, assessment_instance: Assessment
    ) -> None:
        # Test the method when the eco_qnt dataframe's columns contain
        # `loc` information.

        ds_model = RepairModel_DS(assessment_instance)
        eco_qnt = pd.DataFrame(
            {
                ('cmp.A', '0', '1'): [0.00, 0.00, 1.00],
                ('cmp.A', '1', '1'): [1.00, 0.00, 0.00],
            }
        ).rename_axis(columns=['cmp', 'ds', 'loc'])
        ds_model.decision_variables = ('my_DV',)
        # cmp.A should be available and we should get medians.
        # missing_cmp will be marked as missing
        # is_for_LF_model represents a component->consequence pair
        # that is intended for processing by the loss function model
        # and should be ignored by the damage state model.
        ds_model.loss_map = pd.DataFrame(
            {
                'Repair': ['cmp.A'],
            },
            index=['cmp.A'],
        )

        # DS3 is in the loss parameters but has not been triggered.
        ds_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA'],
                ('Quantity', 'Unit'): ['1 EA'],
                ('DS1', 'Family'): [None],
                ('DS1', 'Theta_0'): [100.00],
                ('DS1', 'Theta_1'): [None],
                ('DS2', 'Family'): [None],
                ('DS2', 'Theta_0'): [200.00],
                ('DS2', 'Theta_1'): [None],
                ('DS3', 'Family'): [None],
                ('DS3', 'Theta_0'): [200.00],
                ('DS3', 'Theta_1'): [None],
            },
            index=pd.MultiIndex.from_tuples([('cmp.A', 'my_DV')]),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])
        ds_model.missing = set()
        medians = ds_model._calc_median_consequence(eco_qnt)
        assert len(medians) == 1
        assert 'my_DV' in medians
        pd.testing.assert_frame_equal(
            medians['my_DV'],
            pd.DataFrame(
                {
                    ('cmp.A', '1', '1'): [100.00, 100.00, 100.00],
                }
            ).rename_axis(columns=['cmp', 'ds', 'loc']),
        )


class TestRepairModel_LF(TestRepairModel_Base):
    def test_convert_loss_parameter_units(
        self, assessment_instance: Assessment
    ) -> None:
        lf_model = RepairModel_LF(assessment_instance)
        lf_model.loss_params = pd.DataFrame(
            {
                ('Demand', 'Unit'): ['inps2', 'g'],
                ('DV', 'Unit'): ['test_three', 'test_three'],
                ('LossFunction', 'Theta_0'): [
                    '1.00,1.00|1.00,1.00',
                    '1.00,1.00|1.00,1.00',
                ],
            },
            index=pd.MultiIndex.from_tuples([('cmpA', 'Cost'), ('cmpB', 'Cost')]),
        )

        lf_model.convert_loss_parameter_units()

        pd.testing.assert_frame_equal(
            lf_model.loss_params,
            pd.DataFrame(
                {
                    ('Demand', 'Unit'): ['inps2', 'g'],
                    ('DV', 'Unit'): ['test_three', 'test_three'],
                    ('LossFunction', 'Theta_0'): [
                        '3,3|0.0254,0.0254',
                        '3,3|9.80665,9.80665',
                    ],
                },
                index=pd.MultiIndex.from_tuples(
                    [('cmpA', 'Cost'), ('cmpB', 'Cost')]
                ),
            ),
        )

    def test__calc_median_consequence(self, assessment_instance: Assessment) -> None:
        lf_model = RepairModel_LF(assessment_instance)

        performance_group = pd.DataFrame(
            {
                'Blocks': [1],
            },
            index=pd.MultiIndex.from_tuples([(('cmp.A', 'dv.A'), '0', '1', '0')]),
        )
        loss_map = {'cmp.A': 'cmp.A'}
        required_edps = {(('cmp.A', 'dv.A'), '0', '1', '0'): 'PFA-1-1'}
        demand_dict = {'PFA-1-1': np.array((1.00, 2.00, 3.00))}
        cmp_sample = {
            ('cmp.A', '0', '1', '0'): pd.Series(
                np.array((10.00, 20.00, 30.00)), name=('cmp.A', '0', '1', '0')
            )
        }
        lf_model.loss_params = pd.DataFrame(
            {
                ('LossFunction', 'Theta_0'): ['0.00,1.00|0.00,10.00'],
            },
            index=pd.MultiIndex.from_tuples(
                [('cmp.A', 'dv.A')], names=['Loss Driver', 'Decsision Variable']
            ),
        )
        medians = lf_model._calc_median_consequence(
            performance_group, loss_map, required_edps, demand_dict, cmp_sample
        )
        pd.testing.assert_frame_equal(
            medians,
            pd.DataFrame(
                {('dv.A', 'cmp.A', 'cmp.A', '0', '1', '0', '0'): [1.0, 4.0, 9.0]},
            ).rename_axis(
                columns=['dv', 'loss', 'dmg', 'loc', 'dir', 'uid', 'block']
            ),
        )
        # test small interpolation domain warning
        demand_dict = {'PFA-1-1': np.array((1.00, 2.00, 1e3))}
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Loss function interpolation for consequence '
                '`cmp.A-dv.A` has failed. Ensure a sufficient '
                'interpolation domain  for the X values '
                '(those after the `|` symbol)  and verify '
                'the X-value and Y-value lengths match.'
            ),
        ):
            lf_model._calc_median_consequence(
                performance_group, loss_map, required_edps, demand_dict, cmp_sample
            )

    def test__create_DV_RVs(self, assessment_instance: Assessment) -> None:
        assessment_instance.options.rho_cost_time = 0.50
        lf_model = RepairModel_LF(assessment_instance)
        lf_model.decision_variables = ('Cost', 'Time')
        lf_model.missing = set()
        lf_model.loss_map = pd.DataFrame(
            {
                'Repair': ['cmp.A', 'cmp.B'],
            },
            index=['cmp.A', 'cmp.B'],
        )
        lf_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA', '1 EA', '1 EA'],
                ('Quantity', 'Unit'): ['1 EA', '1 EA', '1 EA'],
                ('LossFunction', 'Family'): ['normal', 'normal', None],
                ('LossFunction', 'Theta_0'): [
                    '0.0,1.0|0.0,1.0',
                    '0.0,1.0|0.0,1.0',
                    '0.0,1.0|0.0,1.0',
                ],
                ('LossFunction', 'Theta_1'): [0.3, 0.3, None],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.A', 'Cost'),
                    ('cmp.A', 'Time'),
                    ('cmp.B', 'Cost'),
                ]
            ),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])

        cases = pd.MultiIndex.from_tuples(
            [
                ('Cost', 'cmp.A', 'cmp.A', '0', '1', '0', '1'),
                ('Time', 'cmp.A', 'cmp.A', '0', '1', '0', '1'),
                ('Cost', 'cmp.B', 'cmp.B', '0', '1', '0', '1'),
            ],
            names=['dv', 'loss', 'dmg', 'loc', 'dir', 'uid', 'block'],
        )
        rv_reg = lf_model._create_DV_RVs(cases)
        assert rv_reg is not None
        for key in (
            'Cost-cmp.A-cmp.A-0-1-0-1',
            'Time-cmp.A-cmp.A-0-1-0-1',
        ):
            assert key in rv_reg.RV
        assert len(rv_reg.RV) == 2
        assert isinstance(
            rv_reg.RV['Cost-cmp.A-cmp.A-0-1-0-1'], uq.NormalRandomVariable
        )
        assert isinstance(
            rv_reg.RV['Time-cmp.A-cmp.A-0-1-0-1'], uq.NormalRandomVariable
        )
        assert np.all(
            rv_reg.RV['Cost-cmp.A-cmp.A-0-1-0-1'].theta[0:2] == np.array((1.0, 0.3))  # type: ignore
        )
        assert np.all(
            rv_reg.RV['Time-cmp.A-cmp.A-0-1-0-1'].theta[0:2] == np.array((1.0, 0.3))  # type: ignore
        )
        assert 'DV-cmp.A-cmp.A-0-1-0-1_set' in rv_reg.RV_set
        np.all(
            rv_reg.RV_set['DV-cmp.A-cmp.A-0-1-0-1_set'].Rho()
            == np.array(((1.0, 0.5), (0.5, 1.0)))
        )
        assert len(rv_reg.RV_set) == 1

    def test__create_DV_RVs_no_rv_case(
        self, assessment_instance: Assessment
    ) -> None:
        # Special case where there is no need for RVs

        lf_model = RepairModel_LF(assessment_instance)
        lf_model.decision_variables = ('Cost', 'Time')
        lf_model.missing = set()
        lf_model.loss_map = pd.DataFrame(
            {
                'Repair': ['cmp.B'],
            },
            index=['cmp.B'],
        )
        lf_model.loss_params = pd.DataFrame(
            {
                ('DV', 'Unit'): ['1 EA'],
                ('Quantity', 'Unit'): ['1 EA'],
                ('LossFunction', 'Family'): [None],
                ('LossFunction', 'Theta_0'): ['0.0,1.0|0.0,1.0'],
                ('LossFunction', 'Theta_1'): [None],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.B', 'Cost'),
                ]
            ),
        ).rename_axis(index=['Loss Driver', 'Decision Variable'])

        cases = pd.MultiIndex.from_tuples(
            [
                ('Cost', 'cmp.B', 'cmp.B', '0', '1', '0', '1'),
            ],
            names=['dv', 'loss', 'dmg', 'loc', 'dir', 'uid', 'block'],
        )
        rv_reg = lf_model._create_DV_RVs(cases)
        assert rv_reg is None


def test__prep_constant_median_DV() -> None:
    median = 10.00
    constant_median_dv = model.loss_model._prep_constant_median_DV(median)
    assert constant_median_dv() == median
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    for value in values:
        assert constant_median_dv(value) == 10.00


def test__prep_bounded_multilinear_median_DV() -> None:
    medians = np.array((1.00, 2.00, 3.00, 4.00, 5.00))
    quantities = np.array((0.00, 1.00, 2.00, 3.00, 4.00))
    f = model.loss_model._prep_bounded_multilinear_median_DV(medians, quantities)

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

    with pytest.raises(
        ValueError,
        match=(
            'A bounded linear median Decision Variable function '
            'called without specifying the quantity '
            'of damaged components'
        ),
    ):
        f(None)


def test__is_for_lf_model() -> None:
    positive_case = pd.DataFrame(
        {
            ('LossFunction', 'Theta_0'): [0.5],
        },
        index=pd.Index(['cmp.1'], name='ID'),
    )

    negative_case = pd.DataFrame(
        {
            ('DS1', 'Theta_0'): [0.50],
        },
        index=pd.Index(['cmp.1'], name='ID'),
    )

    assert _is_for_lf_model(positive_case) is True
    assert _is_for_lf_model(negative_case) is False


def test__is_for_ds_model() -> None:
    positive_case = pd.DataFrame(
        {
            ('DS1', 'Theta_0'): [0.50],
        },
        index=pd.Index(['cmp.1'], name='ID'),
    )

    negative_case = pd.DataFrame(
        {
            ('LossFunction', 'Theta_0'): [0.5],
        },
        index=pd.Index(['cmp.1'], name='ID'),
    )

    assert _is_for_ds_model(positive_case) is True
    assert _is_for_ds_model(negative_case) is False
