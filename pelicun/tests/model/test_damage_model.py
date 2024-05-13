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

from copy import deepcopy
import warnings
import pytest
import numpy as np
import pandas as pd
from pelicun import base
from pelicun import uq
from pelicun.model.damage_model import DamageModel
from pelicun.model.damage_model import DamageModel_Base
from pelicun.model.damage_model import DamageModel_DS
from pelicun.model.damage_model import _is_for_ds_model
from pelicun.tests.model.test_pelicun_model import TestPelicunModel
from pelicun.warnings import PelicunWarning

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=arguments-renamed
# pylint: disable=missing-return-doc,missing-return-type-doc


class TestDamageModel(TestPelicunModel):

    @pytest.fixture
    def damage_model(self, assessment_instance):
        return deepcopy(assessment_instance.damage)

    def test___init__(self, damage_model):
        assert damage_model.log
        assert damage_model.ds_model
        with pytest.raises(AttributeError):
            damage_model.xyz = 123

        assert damage_model.ds_model.damage_params is None
        assert damage_model.ds_model.sample is None

        assert len(damage_model._damage_models) == 1

    def test_damage_models(self, assessment_instance):

        damage_model = DamageModel(assessment_instance)
        assert damage_model._damage_models is not None
        assert len(damage_model._damage_models) == 1
        assert isinstance(damage_model._damage_models[0], DamageModel_DS)

    def test_load_model_parameters(self, damage_model):
        path = (
            'pelicun/tests/data/model/test_DamageModel/'
            'load_model_parameters/damage_db.csv'
        )
        # The file defines the parameters for four components:
        # component.A, component.B, component.C, and component.incomplete
        # component.incomplete is flagged incomplete.
        cmp_set = {'component.A', 'component.B', 'component.incomplete'}
        # (Omit component.C)
        with warnings.catch_warnings(record=True) as w:
            damage_model.load_model_parameters([path], cmp_set)
        assert len(w) == 1
        assert (
            "The damage model does not provide damage information "
            "for the following component(s) in the asset model: "
            "['component.incomplete']."
        ) in str(w[0].message)
        damage_parameters = damage_model.ds_model.damage_params
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
            damage_model.load_model_parameters([path], cmp_set)
        assert len(w) == 1
        assert (
            "The damage model does not provide damage "
            "information for the following component(s) "
            "in the asset model: ['not.exist']."
        ) in str(w[0].message)
        assert damage_model.ds_model.damage_params.empty

    def test_calculate(self):
        # User-facing methods are coupled with other assessment objets
        # and are tested in the verification examples.
        pass

    def test_save_sample(self):
        # User-facing methods are coupled with other assessment objets
        # and are tested in the verification examples.
        pass

    def test_load_sample(self):
        # User-facing methods are coupled with other assessment objets
        # and are tested in the verification examples.
        pass

    def test__get_component_id_set(self, assessment_instance):

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

    def test__ensure_damage_parameter_availability(self, assessment_instance):

        damage_model = DamageModel(assessment_instance)

        damage_model.ds_model.damage_params = pd.DataFrame(
            {
                ('LS1', 'Theta_0'): [0.1, 0.2, 0.3],
                ('LS2', 'Theta_0'): [0.2, 0.3, 0.4],
            },
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3'], name='ID'),
        )

        cmp_list = ['cmp.1', 'cmp.2', 'cmp.3', 'cmp.4']

        expected_missing_components = ['cmp.4']

        with pytest.warns(PelicunWarning) as record:
            missing_components = damage_model._ensure_damage_parameter_availability(
                cmp_list
            )
        assert missing_components == expected_missing_components
        assert len(record) == 1
        assert "cmp.4" in str(record[0].message)


class TestDamageModel_Base(TestPelicunModel):
    def test___init__(self, assessment_instance):

        damage_model = DamageModel_Base(assessment_instance)
        with pytest.raises(AttributeError):
            # pylint: disable=assigning-non-slot
            damage_model.xyz = 123

    def test__load_model_parameters(self, assessment_instance):

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

        damage_model._load_model_parameters(new_data)

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

    def test__convert_damage_parameter_units(self, assessment_instance):

        damage_model = DamageModel_Base(assessment_instance)

        # should have no effect when damage_params is None
        damage_model._convert_damage_parameter_units()

        # converting units from 'g' to 'm/s2' (1g ~ 9.80665 m/s2)

        damage_model.damage_params = pd.DataFrame(
            {
                ('Demand', 'Unit'): ['g', 'g'],
                ('LS1', 'Theta_0'): [0.5, 0.2],  # Values in g's
            },
            index=pd.Index(['cmp.1', 'cmp.2'], name='ID'),
        )

        damage_model._convert_damage_parameter_units()

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

    def test__remove_incomplete_components(self, assessment_instance):

        damage_model = DamageModel_Base(assessment_instance)

        # with damage_model.damage_params set to None this should have
        # no effect.
        damage_model._remove_incomplete_components()

        damage_model.damage_params = pd.DataFrame(
            {
                ('Demand', 'Type'): ['Type1', 'Type2', 'Type3', 'Type4'],
                ('Incomplete', ''): [0, 1, 0, 1],
            },
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3', 'cmp.4'], name='ID'),
        )

        damage_model._remove_incomplete_components()

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
        damage_model.damage_params.drop(('Incomplete', ''), axis=1, inplace=True)
        # now, this should also have no effect
        before = damage_model.damage_params.copy()
        damage_model._remove_incomplete_components()
        pd.testing.assert_frame_equal(before, damage_model.damage_params)

    def test__drop_unused_damage_parameters(self, assessment_instance):

        damage_model = DamageModel_Base(assessment_instance)

        damage_model.damage_params = pd.DataFrame(
            index=pd.Index(['cmp.1', 'cmp.2', 'cmp.3', 'cmp.4'], name='ID')
        )

        cmp_set = {'cmp.1', 'cmp.3'}

        damage_model._drop_unused_damage_parameters(cmp_set)

        pd.testing.assert_frame_equal(
            damage_model.damage_params,
            pd.DataFrame(index=pd.Index(['cmp.1', 'cmp.3'], name='ID')),
        )

    def test__get_pg_batches(self, assessment_instance):

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

    def test__obtain_ds_sample(self, assessment_instance):

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

        res = pd.DataFrame(
            {
                ('cmp.1', '1', '1', '1'): [1, 1],
                ('cmp.3', '1', '1', '1'): [0, 0],
            },
            index=[0, 1],
        )

        res = damage_model._obtain_ds_sample(
            demand_sample,
            component_blocks,
            block_batch_size,
            scaling_specification,
            [],
            nondirectional_multipliers,
        )
        pd.testing.assert_frame_equal(
            res,
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

    def test__handle_operation(self, assessment_instance):

        damage_model = DamageModel_DS(assessment_instance)

        assert damage_model._handle_operation(1.00, '+', 1.00) == 2.00
        assert damage_model._handle_operation(1.00, '-', 1.00) == 0.00
        assert damage_model._handle_operation(1.00, '*', 4.00) == 4.00
        assert damage_model._handle_operation(8.00, '/', 8.00) == 1.00

        with pytest.raises(ValueError) as record:
            damage_model._handle_operation(1.00, '%', 1.00)
        assert ('Invalid operation: `%`') in str(record.value)

    def test__generate_dmg_sample(self, assessment_instance):

        # Create an instance of the damage model
        damage_model = DamageModel_DS(assessment_instance)

        PGB = pd.DataFrame(
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
            sample_size, PGB, scaling_specification
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
            lsds_sample,
            pd.DataFrame(
                {
                    ('cmp.test', '1', '2', '3', '1', '1'): [1, 1],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'block', 'ls']),
        )

    def test__create_dmg_RVs(self, assessment_instance):

        damage_model = DamageModel_DS(assessment_instance)

        PGB = pd.DataFrame(
            {'Blocks': [1]},
            index=pd.MultiIndex.from_tuples(
                [
                    ('cmp.A', '1', '2', '3'),
                ],
                names=['cmp', 'loc', 'dir', 'uid'],
            ),
        )

        damage_params = pd.DataFrame(
            {
                ('Demand', 'Directional'): [0.0],
                ('Demand', 'Offset'): [0.0],
                ('Demand', 'Type'): ['Peak Floor Acceleration'],
                ('Incomplete', ''): [0],
                ('LS1', 'DamageStateWeights'): [
                    '0.40 | 0.10 | 0.50',
                ],
                ('LS1', 'Family'): ['lognormal'],
                ('LS1', 'Theta_0'): [30.00],
                ('LS1', 'Theta_1'): [0.5],
            },
            index=['cmp.A'],
        ).rename_axis('ID')

        # Attach this DataFrame to the damage model instance
        damage_model.damage_params = damage_params

        # Define a scaling specification
        scaling_specification = {'cmp.A-1-2': '*1.20'}

        # Execute the method under test
        capacity_RV_reg, lsds_RV_reg = damage_model._create_dmg_RVs(
            PGB, scaling_specification
        )

        # Now we need to verify the outputs in the registries
        # This will include checks to ensure random variables were
        # created correctly.
        # Example check for presence and properties of a
        # RandomVariable in the registry:
        assert 'FRG-cmp.A-1-2-3-1-1' in capacity_RV_reg.RV
        assert isinstance(
            capacity_RV_reg.RV['FRG-cmp.A-1-2-3-1-1'],
            uq.LogNormalRandomVariable,
        )

        assert 'LSDS-cmp.A-1-2-3-1-1' in lsds_RV_reg.RV
        assert isinstance(
            lsds_RV_reg.RV['LSDS-cmp.A-1-2-3-1-1'],
            uq.MultinomialRandomVariable,
        )

    def test__evaluate_damage_state(self, assessment_instance):

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

    def test__prepare_dmg_quantities(self, assessment_instance):

        #
        # A case with blocks
        #

        damage_model = DamageModel_DS(assessment_instance)

        damage_state_sample = pd.DataFrame(
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

        res = damage_model._prepare_dmg_quantities(
            damage_state_sample,
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

        damage_state_sample = pd.DataFrame(
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

        res = damage_model._prepare_dmg_quantities(
            damage_state_sample,
            component_sample,
            component_marginal_parameters=None,
            dropzero=True,
        )

        # Realization 0: Expect zeros
        # Realization 1: Expect zeros
        # Realization 2: Expect q=6 at DS1
        # Realization 3: Expect q=8 at DS2
        # Realization 4: Expect q=10 at DS3
        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('A', '0', '1', '0', '1'): [0.0, 0.0, 6.0, 0.0, 0.0],
                    ('A', '0', '1', '0', '2'): [0.0, 0.0, 0.0, 8.0, 0.0],
                    ('A', '0', '1', '0', '3'): [0.0, 0.0, 0.0, 0.0, 10.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )

        #
        # Test `dropzero`
        #

        damage_state_sample = pd.DataFrame(
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

        res = damage_model._prepare_dmg_quantities(
            damage_state_sample,
            component_sample,
            component_marginal_parameters=None,
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

        res = damage_model._prepare_dmg_quantities(
            damage_state_sample,
            component_sample,
            component_marginal_parameters=None,
            dropzero=False,
        )

        pd.testing.assert_frame_equal(
            res,
            pd.DataFrame(
                {
                    ('A', '0', '1', '0', '0'): [0.0, 4.0],  # returned
                    ('A', '0', '1', '1', '0'): [0.0, 8.0],  # returned
                    ('A', '0', '1', '1', '1'): [6.0, 0.0],
                }
            ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid', 'ds']),
        )

    def test__perform_dmg_task(self, assessment_instance):

        damage_model = DamageModel_DS(assessment_instance)

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
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.B": {"DS1": "CMP.A_DS4"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.B": {"DS1": "CMP.A_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '2', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '2', '1', '0'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.B-LOC": {"DS1": "CMP.A_DS4"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [1, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 0],
                ('CMP.C', '1', '1', '0'): [0, 0, 0],
                ('CMP.D', '1', '1', '0'): [0, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.A": {"DS1": "ALL_DS2"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '1', '1', '1'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '1', '1', '1'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.B": {"DS1": "CMP.A_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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

        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.A', '2', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '0'): [0, 0, 1],
                ('CMP.B', '2', '1', '0'): [1, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.B-LOC": {"DS1": "CMP.A_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.A-LOC": {"DS1": "ALL_NA"}}
        for task in dmg_process.items():
            damage_model._perform_dmg_task(task, ds_sample)
        after = ds_sample

        pd.testing.assert_frame_equal(
            after,
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
        ds_sample = pd.DataFrame(
            {
                ('CMP.A', '1', '1', '0'): [0, 0, 0],
                ('CMP.B', '1', '1', '1'): [0, 0, 0],
            },
            dtype='int32',
        ).rename_axis(columns=['cmp', 'loc', 'dir', 'uid'])

        dmg_process = {"1_CMP.C": {"DS1": "CMP.A_DS4"}}
        with pytest.warns(PelicunWarning) as record:
            for task in dmg_process.items():
                damage_model._perform_dmg_task(task, ds_sample)
        assert (
            'Source component `CMP.C` in the prescribed damage process not found'
        ) in str(record.list[0].message)

        #
        # Test warnings: Target component not found
        #
        dmg_process = {"1_CMP.A": {"DS1": "CMP.C_DS4"}}
        with pytest.warns(PelicunWarning) as record:
            for task in dmg_process.items():
                damage_model._perform_dmg_task(task, ds_sample)
        assert (
            'Target component `CMP.C` in the prescribed damage process not found'
        ) in str(record.list[0].message)

        #
        # Test Error: Unable to parse source event
        #
        dmg_process = {"1_CMP.A": {"XYZ": "CMP.B_DS1"}}
        with pytest.raises(ValueError) as record:
            for task in dmg_process.items():
                damage_model._perform_dmg_task(task, ds_sample)
        assert ('Unable to parse source event in damage process: `XYZ`') in str(
            record.value
        )
        dmg_process = {"1_CMP.A": {"DS1": "CMP.B_ABC"}}
        with pytest.raises(ValueError) as record:
            for task in dmg_process.items():
                damage_model._perform_dmg_task(task, ds_sample)
        assert ('Unable to parse target event in damage process: `ABC`') in str(
            record.value
        )

    def test__complete_ds_cols(self, assessment_instance):

        damage_model = DamageModel_DS(assessment_instance)
        # the method needs damage parameters
        damage_model.damage_params = base.convert_to_MultiIndex(
            pd.read_csv(
                (
                    'pelicun/tests/data/model/test_DamageModel/'
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
        out = damage_model._complete_ds_cols(dmg_sample)
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


def test__is_for_ds_model():

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
