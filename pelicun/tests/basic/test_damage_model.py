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

"""These are unit and integration tests on the damage model of pelicun."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pelicun import base, uq
from pelicun.base import ensure_value
from pelicun.model.damage_model import (
    DamageModel,
    DamageModel_Base,
    DamageModel_DS,
    _is_for_ds_model,
)
from pelicun.pelicun_warnings import PelicunWarning
from pelicun.tests.basic.test_pelicun_model import TestPelicunModel

if TYPE_CHECKING:
    from pelicun.assessment import Assessment


class TestDamageModel(TestPelicunModel):
    @pytest.fixture
    def damage_model(self, assessment_instance: Assessment) -> DamageModel:
        return deepcopy(assessment_instance.damage)

    def test___init__(self, damage_model: DamageModel) -> None:
        assert damage_model.log
        assert damage_model.ds_model
        with pytest.raises(AttributeError):
            damage_model.xyz = 123  # type: ignore

        assert damage_model.ds_model.damage_params is None
        assert damage_model.ds_model.sample is None

        assert len(damage_model._damage_models) == 1

    def test_damage_models(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel(assessment_instance)
        assert damage_model._damage_models is not None
        assert len(damage_model._damage_models) == 1
        assert isinstance(damage_model._damage_models[0], DamageModel_DS)

    def test_load_model_parameters(self, damage_model: DamageModel) -> None:
        path = (
            'pelicun/tests/basic/data/model/test_DamageModel/'
            'load_model_parameters/damage_db.csv'
        )
        # The file defines the parameters for four components:
        # component.A, component.B, component.C, and component.incomplete
        # component.incomplete is flagged incomplete.
        cmp_set = {'component.A', 'component.B', 'component.incomplete'}
        # (Omit component.C)
        with warnings.catch_warnings(record=True) as w:
            damage_model.load_model_parameters([path], cmp_set, warn_missing=True)
        assert len(w) == 1
        assert (
            'The damage model does not provide damage information '
            'for the following component(s) in the asset model: '
            "['component.incomplete']."
        ) in str(w[0].message)
        damage_parameters = damage_model.ds_model.damage_params
        assert damage_parameters is not None
        assert 'component.A' in damage_parameters.index
        assert 'component.B' in damage_parameters.index
        assert 'component.C' not in damage_parameters.index
        assert 'component.incomplete' not in damage_parameters.index

        # make sure unit conversions were done correctly.
        # component.A: unitless, 3 limit states
        # component.B: from g -> m/s^2, 2 limit states
        assert damage_parameters['LS1']['Theta_0']['component.A'] == 0.02
        assert damage_parameters['LS2']['Theta_0']['component.A'] == 0.04
        assert damage_parameters['LS3']['Theta_0']['component.A'] == 0.08
        assert damage_parameters['LS1']['Theta_0']['component.B'] == 1.96133
        assert damage_parameters['LS2']['Theta_0']['component.B'] == 3.92266
        assert pd.isna(damage_parameters['LS3']['Theta_0']['component.B'])

        # If a component is in the set but does not have damage
        # parameters, no damage parameters are loaded for it.
        cmp_set = {'not.exist'}
        with warnings.catch_warnings(record=True) as w:
            damage_model.load_model_parameters([path], cmp_set, warn_missing=True)
        assert len(w) == 1
        assert (
            'The damage model does not provide damage '
            'information for the following component(s) '
            "in the asset model: ['not.exist']."
        ) in str(w[0].message)
        assert ensure_value(damage_model.ds_model.damage_params).empty

    def test_calculate(self) -> None:
        # User-facing methods are coupled with other assessment objects
        # and are tested in the verification examples.
        pass

    def test_save_sample(self) -> None:
        # User-facing methods are coupled with other assessment objects
        # and are tested in the verification examples.
        pass

    def test_load_sample(self) -> None:
        # User-facing methods are coupled with other assessment objects
        # and are tested in the verification examples.
        pass

    def test__get_component_id_set(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel(assessment_instance)

        damage_model.ds_model.damage_params = pd.DataFrame(
            {
                ('LS1', 'Theta_0'): [0.1, 0.2, 0.3],
                ('LS2', 'Theta_0'): [0.2, 0.3, 0.4],
            },
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3'], name='ID'),
        )

        component_id_set = damage_model._get_component_id_set()

        expected_set = {'cmp.1', 'cmp.2', 'cmp.3'}

        assert component_id_set == expected_set

    def test__ensure_damage_parameter_availability(
        self, assessment_instance: Assessment
    ) -> None:
        damage_model = DamageModel(assessment_instance)

        damage_model.ds_model.damage_params = pd.DataFrame(
            {
                ('LS1', 'Theta_0'): [0.1, 0.2, 0.3],
                ('LS2', 'Theta_0'): [0.2, 0.3, 0.4],
            },
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3'], name='ID'),
        )

        cmp_set = {'cmp.1', 'cmp.2', 'cmp.3', 'cmp.4'}

        expected_missing_components = ['cmp.4']

        with pytest.warns(PelicunWarning) as record:
            missing_components = damage_model._ensure_damage_parameter_availability(
                cmp_set, warn_missing=True
            )
        assert missing_components == expected_missing_components
        assert len(record) == 1
        assert 'cmp.4' in str(record[0].message)


class TestDamageModel_Base(TestPelicunModel):
    def test___init__(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_Base(assessment_instance)
        with pytest.raises(AttributeError):
            damage_model.xyz = 123  # type: ignore

    def test__load_model_parameters(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_Base(assessment_instance)

        damage_model.damage_params = pd.DataFrame(
            {
                ('Demand', 'Type'): ['Type1', 'Type2'],
                ('LS1', 'Theta_0'): [0.1, 0.2],
            },
            index=pd.Index(['cmp.1', 'cmp.2'], name='ID'),
        )

        # New data to be loaded, which contains a redefinition and a
        # new parameter
        new_data = pd.DataFrame(
            {
                ('Demand', 'Type'): ['Type3', 'Type4'],
                ('LS1', 'Theta_0'): [0.3, 0.4],
            },
            index=pd.Index(['cmp.1', 'cmp.3'], name='ID'),
        )

        damage_model.load_model_parameters(new_data)

        pd.testing.assert_frame_equal(
            damage_model.damage_params,
            pd.DataFrame(
                {
                    ('Demand', 'Type'): ['Type1', 'Type2', 'Type4'],
                    ('LS1', 'Theta_0'): [0.1, 0.2, 0.4],
                },
                index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3'], name='ID'),
            ),
        )

    def test__load_model_parameters_with_truncation(
        self, assessment_instance: Assessment
    ) -> None:
        damage_model = assessment_instance.damage
        path = (
            'pelicun/tests/basic/data/model/test_DamageModel/'
            'load_model_parameters_with_truncation/damage_db.csv'
        )
        cmp_set = {
            'component.A',
        }
        damage_model.load_model_parameters([path], cmp_set, warn_missing=False)
        pgb = pd.DataFrame(
            [[1]], index=(('component.A', '1', '1', '1'),), columns=['Blocks']
        )
        capacity_sample, _ = damage_model.ds_model._generate_dmg_sample(
            sample_size=100, pgb=pgb, scaling_specification=None
        )
        np.all(
            capacity_sample[
                'component.A',  # cmp
                '1',  # loc
                '1',  # dir
                '1',  # uid
                '1',  # block
                '1',  # ls
            ]
            > 0.01
        )
        np.all(capacity_sample['component.A', '1', '1', '1', '1', '1'] < 0.03)
        np.all(capacity_sample['component.A', '1', '1', '1', '1', '2'] > 0.03)
        np.all(capacity_sample['component.A', '1', '1', '1', '1', '2'] < 0.05)
        np.all(capacity_sample['component.A', '1', '1', '1', '1', '3'] > 0.06)
        np.all(capacity_sample['component.A', '1', '1', '1', '1', '3'] < 0.10)

    def test__convert_damage_parameter_units(
        self, assessment_instance: Assessment
    ) -> None:
        damage_model = DamageModel_Base(assessment_instance)

        # should have no effect when damage_params is None
        damage_model.convert_damage_parameter_units()

        # converting units from 'g' to 'm/s2' (1g ~ 9.80665 m/s2)

        damage_model.damage_params = pd.DataFrame(
            {
                ('Demand', 'Unit'): ['g', 'g'],
                ('LS1', 'Theta_0'): [0.5, 0.2],  # Values in g's
            },
            index=pd.Index(['cmp.1', 'cmp.2'], name='ID'),
        )

        damage_model.convert_damage_parameter_units()

        pd.testing.assert_frame_equal(
            damage_model.damage_params,
            pd.DataFrame(
                {
                    ('LS1', 'Theta_0'): [
                        0.5 * 9.80665,
                        0.2 * 9.80665,
                    ],
                },
                index=pd.Index(['cmp.1', 'cmp.2'], name='ID'),
            ),
        )

    def test__remove_incomplete_components(
        self, assessment_instance: Assessment
    ) -> None:
        damage_model = DamageModel_Base(assessment_instance)

        # with damage_model.damage_params set to None this should have
        # no effect.
        damage_model.remove_incomplete_components()

        damage_model.damage_params = pd.DataFrame(
            {
                ('Demand', 'Type'): ['Type1', 'Type2', 'Type3', 'Type4'],
                ('Incomplete', ''): [0, 1, 0, 1],
            },
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3', 'cmp.4'], name='ID'),
        )

        damage_model.remove_incomplete_components()

        pd.testing.assert_frame_equal(
            damage_model.damage_params,
            pd.DataFrame(
                {
                    ('Demand', 'Type'): ['Type1', 'Type3'],
                    # Only complete components remain
                    ('Incomplete', ''): [0, 0],
                },
                index=pd.Index(['cmp.1', 'cmp.3'], name='ID'),
            ),
        )

        # with damage_model.damage_params set to None this should have
        # no effect.
        damage_model.damage_params = damage_model.damage_params.drop(
            ('Incomplete', ''), axis=1
        )
        # now, this should also have no effect
        before = damage_model.damage_params.copy()
        damage_model.remove_incomplete_components()
        pd.testing.assert_frame_equal(before, damage_model.damage_params)

    def test__drop_unused_damage_parameters(
        self, assessment_instance: Assessment
    ) -> None:
        damage_model = DamageModel_Base(assessment_instance)

        damage_model.damage_params = pd.DataFrame(
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3', 'cmp.4'], name='ID')
        )

        cmp_set = {'cmp.1', 'cmp.3'}

        damage_model.drop_unused_damage_parameters(cmp_set)

        pd.testing.assert_frame_equal(
            damage_model.damage_params,
            pd.DataFrame(index=pd.Index(['cmp.1', 'cmp.3'], name='ID')),
        )

    def test__get_pg_batches(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_Base(assessment_instance)

        component_blocks = pd.DataFrame(
            {'Blocks': [1, 1, 2, 1, 3, 4]},
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.1', '1', '1', '1'),
                    ('cmp.2', '2', '2', '2'),
                    ('cmp.3', '1', '1', '1'),
                    ('cmp.4', '3', '3', '3'),
                    ('cmp.5', '2', '2', '2'),
                    ('cmp.6', '1', '1', '1'),
                ],
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
        )

        block_batch_size = 3

        missing_components = ['cmp.4', 'cmp.5', 'cmp.6']

        # Attach a mocked damage_params DataFrame to the damage model
        # instance to simulate the available
        # components. `_get_pg_batches` doesn't need any other
        # information from that attribute.
        damage_model.damage_params = pd.DataFrame(index=['cmp.1', 'cmp.2', 'cmp.3'])

        resulting_batches = damage_model._get_pg_batches(
            component_blocks, block_batch_size, missing_components
        )
        pd.testing.assert_frame_equal(
            resulting_batches,
            pd.DataFrame(
                {'Blocks': [1, 2, 1]},
                index=pd.MultiIndex.from_tuples(
                    [
                        (1, 'cmp.1', '1', '1', '1'),
                        (1, 'cmp.3', '1', '1', '1'),
                        (2, 'cmp.2', '2', '2', '2'),
                    ],
                    names=['Batch', 'cmp', 'loc', 'dir', 'uid'],
                ),
            ),
        )


class TestDamageModel_DS(TestDamageModel_Base):
    def test__obtain_ds_sample(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_DS(assessment_instance)

        demand_sample = pd.DataFrame(
            {
                ('PFA', '0', '1'): [5.00, 5.00],  # m/s2
                ('PFA', '0', '2'): [5.00, 5.00],
            },
            index=[0, 1],
        ).rename_axis(columns=['type', 'loc', 'dir'])

        component_blocks = pd.DataFrame(
            {'Blocks': [1, 2, 1]},
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.1', '1', '1', '1'),
                    ('cmp.2', '1', '1', '1'),
                    ('cmp.3', '1', '1', '1'),
                ],
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
        )

        block_batch_size = 2
        scaling_specification = None
        nondirectional_multipliers = {'ALL': 1.2}

        damage_model.damage_params = pd.DataFrame(
            {
                ('Demand', 'Directional'): [0, 0, 0],
                ('Demand', 'Offset'): [0, 0, 0],
                ('Demand', 'Type'): [
                    'Peak Floor Acceleration',
                    'Peak Floor Acceleration',
                    'Peak Floor Acceleration',
                ],
                ('Incomplete', ''): [0, 0, 0],
                ('LS1', 'DamageStateWeights'): [None, None, None],
                ('LS1', 'Family'): [None, None, None],
                ('LS1', 'Theta_0'): [1.0, 1.0, 10.0],  # m/s2
                ('LS1', 'Theta_1'): [None, None, None],
            },
            index=['cmp.1', 'cmp.2', 'cmp.3'],
        ).rename_axis('ID')

        damage_model.obtain_ds_sample(
            demand_sample,
            component_blocks,
            block_batch_size,
            scaling_specification,
            [],
            nondirectional_multipliers,
        )
        pd.testing.assert_frame_equal(
            ensure_value(damage_model.ds_sample),
            pd.DataFrame(
                {
                    ('cmp.1', '1', '1', '1', '1'): [1, 1],
                    ('cmp.2', '1', '1', '1', '1'): [1, 1],
                    ('cmp.2', '1', '1', '1', '2'): [1, 1],
                    ('cmp.3', '1', '1', '1', '1'): [0, 0],
                },
                dtype='int64',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block']),
        )

    def test__handle_operation(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_DS(assessment_instance)

        assert damage_model._handle_operation(1.00, '+', 1.00) == 2.00
        assert damage_model._handle_operation(1.00, '-', 1.00) == 0.00
        assert damage_model._handle_operation(1.00, '*', 4.00) == 4.00
        assert damage_model._handle_operation(8.00, '/', 8.00) == 1.00

        with pytest.raises(ValueError, match='Invalid operation: `%`'):
            damage_model._handle_operation(1.00, '%', 1.00)

    def test__generate_dmg_sample(self, assessment_instance: Assessment) -> None:
        # Create an instance of the damage model
        damage_model = DamageModel_DS(assessment_instance)

        pgb = pd.DataFrame(
            {'Blocks': [1]},
            index=pd.MultiIndex.from_tuples(
                [('cmp.test', '1', '2', '3')],
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
        )

        damage_params = pd.DataFrame(
            {
                ('Demand', 'Directional'): [0.0],
                ('Demand', 'Offset'): [0.0],
                ('Demand', 'Type'): ['None Specified'],
                ('Incomplete', ''): [0],
                ('LS1', 'DamageStateWeights'): [None],  # No randomness
                ('LS1', 'Family'): [None],  # No specific family of distribution
                ('LS1', 'Theta_0'): [1.0],  # Constant value for simplicity
                ('LS1', 'Theta_1'): [None],  # No randomness
            },
            index=['cmp.test'],
        ).rename_axis('ID')

        damage_model.damage_params = damage_params

        scaling_specification = None
        sample_size = 2

        capacity_sample, lsds_sample = damage_model._generate_dmg_sample(
            sample_size, pgb, scaling_specification
        )

        pd.testing.assert_frame_equal(
            capacity_sample,
            pd.DataFrame(
                {
                    ('cmp.test', '1', '2', '3', '1', '1'): [1.0, 1.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block', 'ls']),
        )

        pd.testing.assert_frame_equal(
            lsds_sample.astype('int32'),
            pd.DataFrame(
                {
                    ('cmp.test', '1', '2', '3', '1', '1'): [1, 1],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block', 'ls']),
        )

    def test__create_dmg_RVs(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_DS(assessment_instance)

        pgb = pd.DataFrame(
            {'Blocks': [1]},
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.A', '1', '2', '3'),
                    ('cmp.B', '1', '2', '3'),
                ],
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
        )

        damage_params = pd.DataFrame(
            {
                ('Demand', 'Directional'): [0.0, 0.0],
                ('Demand', 'Offset'): [0.0, 0.0],
                ('Demand', 'Type'): [
                    'Peak Floor Acceleration',
                    'Peak Floor Acceleration',
                ],
                ('Incomplete', ''): [0, 0],
                ('LS1', 'DamageStateWeights'): [
                    '0.40 | 0.10 | 0.50',
                    '0.40 | 0.10 | 0.50',
                ],
                ('LS1', 'Family'): ['lognormal', 'lognormal'],
                ('LS1', 'Theta_0'): [30.00, 30.00],
                ('LS1', 'Theta_1'): [0.5, 0.5],
            },
            index=['cmp.A', 'cmp.B'],
        ).rename_axis('ID')

        # Attach this DataFrame to the damage model instance
        damage_model.damage_params = damage_params

        # Define a scaling specification
        operation_list = ['*1.20', '+0.10', '/1.20', '-0.10', '*1.10']
        scaling_specification = {
            'cmp.A-1-2': {'LS1': '*1.20'},
            'cmp.B-1-2': {'LS1': operation_list},
        }

        # Create random variables based on the damage parameters
        capacity_rv_reg, lsds_rv_reg = damage_model._create_dmg_RVs(
            pgb, scaling_specification
        )

        # Now we need to verify the outputs in the registries
        # This will include checks to ensure random variables were
        # created correctly.
        # Example check for presence and properties of a
        # RandomVariable in the registry:
        assert 'FRG-cmp.A-1-2-3-1-1' in capacity_rv_reg.RV
        assert 'FRG-cmp.B-1-2-3-1-1' in capacity_rv_reg.RV
        assert isinstance(
            capacity_rv_reg.RV['FRG-cmp.A-1-2-3-1-1'],
            uq.LogNormalRandomVariable,
        )
        assert isinstance(
            capacity_rv_reg.RV['FRG-cmp.B-1-2-3-1-1'],
            uq.LogNormalRandomVariable,
        )

        assert 'LSDS-cmp.A-1-2-3-1-1' in lsds_rv_reg.RV
        assert 'LSDS-cmp.B-1-2-3-1-1' in lsds_rv_reg.RV
        assert isinstance(
            lsds_rv_reg.RV['LSDS-cmp.A-1-2-3-1-1'],
            uq.MultinomialRandomVariable,
        )
        assert isinstance(
            lsds_rv_reg.RV['LSDS-cmp.B-1-2-3-1-1'],
            uq.MultinomialRandomVariable,
        )

        # Validate the scaling of the random variables are correct
        # Generate samples for validating that theta_0 is scaled correctly
        capacity_rv_reg.generate_sample(
            sample_size=len(operation_list), method='LHS'
        )
        cmp_b_scaled_theta0 = np.array(
            [30.0 * 1.20, 30.0 + 0.10, 30.0 / 1.20, 30.0 - 0.10, 30.0 * 1.10]
        )
        for rv_name, rv in capacity_rv_reg.RV.items():
            uniform_sample = rv._uni_sample
            sample = rv.sample
            assert uniform_sample is not None
            assert sample is not None

            for i in range(len(operation_list)):
                if rv_name == 'FRG-cmp.A-1-2-3-1-1':
                    theta = 1.20 * 30.0
                elif rv_name == 'FRG-cmp.B-1-2-3-1-1':
                    theta = cmp_b_scaled_theta0[i]
                assert sample[i] == np.exp(
                    norm.ppf(uniform_sample[i], loc=np.log(theta), scale=0.5)
                )

    def test__evaluate_damage_state(self, assessment_instance: Assessment) -> None:
        # We define a single component with 3 limit states.
        # The last limit state can have two damage states, DS3 and DS4.
        # We test that the damage state assignments are correct.
        # We intend to have the following DS realizations: 0, 1, 2, 3, 4.

        damage_model = DamageModel_DS(assessment_instance)

        demand_dict = {'edp': np.array([1.00, 3.00, 5.00, 7.00, 7.00])}

        #                        component, loc, dir, uid, block
        required_edps = {'edp': [('component.A', '0', '1', '1', '1')]}

        capacity_sample = pd.DataFrame(
            {
                #    component, loc, dir, uid, block, limit state
                ('component.A', '0', '1', '1', '1', '1'): np.full(5, 2.00),
                ('component.A', '0', '1', '1', '1', '2'): np.full(5, 4.00),
                ('component.A', '0', '1', '1', '1', '3'): np.full(5, 6.00),
            },
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block', 'ls'])

        lsds_sample = pd.DataFrame(
            {
                #    component, loc, dir, uid, block, limit state
                ('component.A', '0', '1', '1', '1', '1'): [1, 1, 1, 1, 1],
                ('component.A', '0', '1', '1', '1', '2'): [2, 2, 2, 2, 2],
                ('component.A', '0', '1', '1', '1', '3'): [3, 3, 3, 3, 4],
            },
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block', 'ls'])

        res = damage_model._evaluate_damage_state(
            demand_dict, required_edps, capacity_sample, lsds_sample
        )
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('component.A', '0', '1', '1', '1'): [0, 1, 2, 3, 4],
                },
                dtype='int64',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block']),
        )

    def test__prepare_dmg_quantities(self, assessment_instance: Assessment) -> None:
        #
        # A case with blocks
        #

        damage_model = DamageModel_DS(assessment_instance)

        damage_model.ds_sample = pd.DataFrame(
            {
                ('A', '0', '1', '0', '1'): [-1, 0, 1, 2, 3],  # block 1
                ('A', '0', '1', '0', '2'): [3, -1, 0, 1, 2],  # block 2
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block'])

        component_sample = pd.DataFrame(
            {
                ('A', '0', '1', '0'): [2.0, 4.0, 6.0, 8.0, 10.0],
            },
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        component_marginal_parameters = pd.DataFrame(
            {
                'Blocks': [2.00],
            },
            index=pd.MultiIndex.from_tuples([('A', '0', '1', '0')]),
        ).rename_axis(index=['cmp', 'loc', 'dir', 'uid'])

        res = damage_model.prepare_dmg_quantities(
            component_sample,
            component_marginal_parameters,
            dropzero=True,
        )

        # Each block takes half the quantity.
        # Realization 0: Expect q=1 at DS3 from block 2
        # Realization 1: Expect zeros
        # Realization 2: Expect q=6/2=3 at DS1 from block 1
        # Realization 3: Expect q=8/2 at DSs 1 and 2
        # Realization 4: Expect q=10/2 at DSs 2 and 3
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('A', '0', '1', '0', '1'): [0.0, 0.0, 3.0, 4.0, 0.0],
                    ('A', '0', '1', '0', '2'): [0.0, 0.0, 0.0, 4.0, 5.0],
                    ('A', '0', '1', '0', '3'): [1.0, 0.0, 0.0, 0.0, 5.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )

        #
        # A case without blocks
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('A', '0', '1', '0', '1'): [-1, 0, 1, 2, 3],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block'])

        component_sample = pd.DataFrame(
            {
                ('A', '0', '1', '0'): [2.0, 4.0, 6.0, 8.0, 10.0],
            },
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        res = damage_model.prepare_dmg_quantities(
            component_sample,
            None,
            dropzero=True,
        )

        # Realization 0: Expect NaNs
        # Realization 1: Expect zeros
        # Realization 2: Expect q=6 at DS1
        # Realization 3: Expect q=8 at DS2
        # Realization 4: Expect q=10 at DS3
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('A', '0', '1', '0', '1'): [np.nan, 0.0, 6.0, 0.0, 0.0],
                    ('A', '0', '1', '0', '2'): [np.nan, 0.0, 0.0, 8.0, 0.0],
                    ('A', '0', '1', '0', '3'): [np.nan, 0.0, 0.0, 0.0, 10.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )

        #
        # Test `dropzero`
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('A', '0', '1', '0', '1'): [-1, 0],
                ('A', '0', '1', '1', '1'): [1, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block'])

        component_sample = pd.DataFrame(
            {
                ('A', '0', '1', '0'): [2.0, 4.0],
                ('A', '0', '1', '1'): [6.0, 8.0],
            },
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        res = damage_model.prepare_dmg_quantities(
            component_sample,
            None,
            dropzero=True,
        )

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('A', '0', '1', '1', '1'): [6.0, 0.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )

        res = damage_model.prepare_dmg_quantities(
            component_sample,
            None,
            dropzero=False,
        )

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('A', '0', '1', '0', '0'): [np.nan, 4.0],  # returned
                    ('A', '0', '1', '1', '0'): [0.0, 8.0],  # returned
                    ('A', '0', '1', '1', '1'): [6.0, 0.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )

    def test__perform_dmg_task(self, assessment_instance: Assessment) -> None:  # noqa: C901
        damage_model = DamageModel_DS(assessment_instance)

        #
        # when CMP.B reaches DS1, CMP.A should be DS4
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.B': {'DS1': 'CMP.A_DS4'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [4, 0, 4],
                    ('CMP.A', '1', '1', '1'): [4, 0, 4],
                    ('CMP.B', '1', '1', '0'): [0, 0, 1],
                    ('CMP.B', '1', '1', '1'): [1, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # when CMP.B reaches DS1, CMP.A should be NA (-1)
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.B': {'DS1': 'CMP.A_NA'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [-1, 0, -1],
                    ('CMP.A', '1', '1', '1'): [-1, 0, -1],
                    ('CMP.B', '1', '1', '0'): [0, 0, 1],
                    ('CMP.B', '1', '1', '1'): [1, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # `-LOC` keyword
        # when CMP.B reaches DS1, CMP.A should be DS4
        # matching locations
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '2', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '2', '1', '0'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.B-LOC': {'DS1': 'CMP.A_DS4'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [0, 0, 4],
                    ('CMP.A', '2', '1', '0'): [4, 0, 0],
                    ('CMP.B', '1', '1', '0'): [0, 0, 1],
                    ('CMP.B', '2', '1', '0'): [1, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # ALL keyword
        #
        # Whenever CMP.A reaches DS1, all other components should be
        # set to DS2.
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [1, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 0],
                ('CMP.C', '1', '1', '0'): [0, 0, 0],
                ('CMP.D', '1', '1', '0'): [0, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.A': {'DS1': 'ALL_DS2'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [1, 0, 0],
                    ('CMP.B', '1', '1', '0'): [2, 0, 0],
                    ('CMP.C', '1', '1', '0'): [2, 0, 0],
                    ('CMP.D', '1', '1', '0'): [2, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # NA keyword
        #
        # NA translates to -1 representing nan
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.B': {'DS1': 'CMP.A_NA'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [-1, 0, -1],
                    ('CMP.A', '1', '1', '1'): [-1, 0, -1],
                    ('CMP.B', '1', '1', '0'): [0, 0, 1],
                    ('CMP.B', '1', '1', '1'): [1, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # NA keyword combined with `-LOC`
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '2', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '2', '1', '0'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.B-LOC': {'DS1': 'CMP.A_NA'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [0, 0, -1],
                    ('CMP.A', '2', '1', '0'): [-1, 0, 0],
                    ('CMP.B', '1', '1', '0'): [0, 0, 1],
                    ('CMP.B', '2', '1', '0'): [1, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # NA keyword combined with `-LOC` and `ALL`
        #

        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 1],
                ('CMP.A', '2', '1', '0'): [1, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 0],
                ('CMP.B', '2', '1', '0'): [0, 0, 0],
                ('CMP.C', '1', '1', '0'): [0, 0, 0],
                ('CMP.C', '2', '1', '0'): [0, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.A-LOC': {'DS1': 'ALL_NA'}}
        for task in dmg_process.items():
            damage_model.perform_dmg_task(task)

        pd.testing.assert_frame_equal(
            damage_model.ds_sample,
            pd.DataFrame(
                {
                    ('CMP.A', '1', '1', '0'): [0, 0, 1],
                    ('CMP.A', '2', '1', '0'): [1, 0, 0],
                    ('CMP.B', '1', '1', '0'): [0, 0, -1],
                    ('CMP.B', '2', '1', '0'): [-1, 0, 0],
                    ('CMP.C', '1', '1', '0'): [0, 0, -1],
                    ('CMP.C', '2', '1', '0'): [-1, 0, 0],
                },
                dtype='int32',
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid']),
        )

        #
        # Test warnings: Source component not found
        #
        damage_model.ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '1'): [0, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {'1_CMP.C': {'DS1': 'CMP.A_DS4'}}
        with pytest.warns(PelicunWarning) as record:
            for task in dmg_process.items():
                damage_model.perform_dmg_task(task)
        assert (
            'Source component `CMP.C` in the prescribed damage process not found'
        ) in str(record.list[0].message)

        #
        # Test warnings: Target component not found
        #
        dmg_process = {'1_CMP.A': {'DS1': 'CMP.C_DS4'}}
        with pytest.warns(PelicunWarning) as record:
            for task in dmg_process.items():
                damage_model.perform_dmg_task(task)
        assert (
            'Target component `CMP.C` in the prescribed damage process not found'
        ) in str(record.list[0].message)

        #
        # Test Error: Unable to parse source event
        #
        dmg_process = {'1_CMP.A': {'XYZ': 'CMP.B_DS1'}}
        for task in dmg_process.items():
            with pytest.raises(
                ValueError,
                match='Unable to parse source event in damage process: `XYZ`',
            ):
                damage_model.perform_dmg_task(task)
        dmg_process = {'1_CMP.A': {'DS1': 'CMP.B_ABC'}}
        for task in dmg_process.items():
            with pytest.raises(
                ValueError,
                match='Unable to parse target event in damage process: `ABC`',
            ):
                damage_model.perform_dmg_task(task)

    def test__complete_ds_cols(self, assessment_instance: Assessment) -> None:
        damage_model = DamageModel_DS(assessment_instance)
        # the method needs damage parameters
        damage_model.damage_params = base.convert_to_MultiIndex(
            pd.read_csv(
                (
                    'pelicun/tests/basic/data/model/test_DamageModel/'
                    '_complete_ds_cols/parameters.csv'
                ),
                index_col=0,
            ),
            axis=1,
        )
        # Set up one realization, with 100 units of the component in
        # damage state 2.
        dmg_sample = pd.DataFrame(
            {
                ('many.ds', '0', '0', '0', '2'): [100.00],
                ('single.ds', '0', '0', '0', '1'): [100.00],
            },
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds'])
        out = damage_model.complete_ds_cols(dmg_sample)
        pd.testing.assert_frame_equal(
            out,
            pd.DataFrame(
                {
                    ('many.ds', '0', '0', '0', '0'): [0.00],
                    ('many.ds', '0', '0', '0', '1'): [0.00],
                    ('many.ds', '0', '0', '0', '2'): [100.00],
                    ('many.ds', '0', '0', '0', '3'): [0.00],
                    ('single.ds', '0', '0', '0', '0'): [0.00],
                    ('single.ds', '0', '0', '0', '1'): [100.00],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )


def test__is_for_ds_model() -> None:
    data_with_ls1 = pd.DataFrame(
        {
            ('LS1', 'Theta_0'): [0.5],
            ('LS2', 'Theta_0'): [0.6],
        },
        index=pd.Index(['cmp.1'], name='ID'),
    )

    data_without_ls1 = pd.DataFrame(
        {
            ('Demand', 'Type'): ['Type1'],
            ('LS2', 'Theta_0'): [0.6],
        },
        index=pd.Index(['cmp.1'], name='ID'),
    )

    result_with_ls1 = _is_for_ds_model(data_with_ls1)
    assert result_with_ls1 is True

    result_without_ls1 = _is_for_ds_model(data_without_ls1)
    assert result_without_ls1 is False
