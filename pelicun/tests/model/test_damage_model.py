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
import warnings
import pytest
import numpy as np
import pandas as pd
from pelicun import base
from pelicun import assessment
from pelicun import model
from pelicun import uq
from pelicun.model.damage_model import DamageModel_DS
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

    def test_init(self, damage_model):
        assert damage_model.log
        assert damage_model.ds_model
        with pytest.raises(AttributeError):
            damage_model.xyz = 123

        assert damage_model.ds_model.damage_params is None
        assert damage_model.ds_model.sample is None

        assert len(damage_model.damage_models) == 1

    def test_damage_models(self):
        pass

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
        # with pytest.warns:
        #     pass
        assert damage_model.ds_model.damage_params.empty

    def test_calculate(self):
        pass

    def test_load_sample(self):
        pass

    def test__get_component_id_set(self):
        pass

    def test__ensure_damage_parameter_availability(self):
        pass


def test__is_for_ds_model():
    pass


class TestDamageModel_Base(TestPelicunModel):
    def test___init__(self):
        pass

    def test__load_model_parameters(self):
        pass

    def test__convert_damage_parameter_units(self):
        pass

    def test__remove_incomplete_components(self):
        pass

    def test__drop_unused_damage_parameters(self):
        pass

    def test__get_pg_batches(self):
        pass


class TestDamageModel_DS(TestDamageModel_Base):
    def test__obtain_ds_sample(self):
        pass

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
                ('Directional', None): [0.0],
                ('Offset', None): [0.0],
                ('Type', None): ['None Specified'],
                ('Incomplete', None): [0],
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
                ('Directional', None): [0.0],
                ('Offset', None): [0.0],
                ('Type', None): ['Peak Floor Acceleration'],
                ('Incomplete', None): [0],
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
        # This will include checks to ensure random variables were created correctly.
        # Example check for presence and properties of a RandomVariable in the registry:
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


# class TestDamageModel(TestPelicunModel):
#     @pytest.fixture
#     def cmp_sample_A(self):
#         # This sample contains 8 units of B.10.31.001 assigned to
#         # locations 1, 2 and directions 1, 2
#         return pd.DataFrame(
#             {
#                 ('B.10.31.001', f'{i}', f'{j}', '0'): 8.0
#                 for i in range(1, 3)
#                 for j in range(1, 3)
#             },
#             index=range(10),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ('B.10.31.001', f'{i}', f'{j}', '0')
#                     for i in range(1, 3)
#                     for j in range(1, 3)
#                 ),
#                 names=('cmp', 'loc', 'dir', 'uid'),
#             ),
#         )

#     @pytest.fixture
#     def calibration_config_A(self):
#         return {
#             "ALL": {"DistributionFamily": "lognormal"},
#             "PID": {
#                 "DistributionFamily": "lognormal",
#                 "TruncateLower": "",
#                 "TruncateUpper": "0.06",
#             },
#         }

#     @pytest.fixture
#     def damage_model_model_loaded(self, damage_model, cmp_sample_A):
#         asmt = damage_model._asmnt
#         asmt.get_default_data('damage_DB_FEMA_P58_2nd')
#         asmt.asset.cmp_sample = cmp_sample_A
#         damage_model.load_damage_model(['PelicunDefault/damage_DB_FEMA_P58_2nd.csv'])
#         return deepcopy(damage_model)

#     @pytest.fixture
#     def damage_model_with_sample(self, assessment_instance):
#         dmg_process = None
#         assessment_instance.demand.sample = pd.DataFrame(
#             np.column_stack(
#                 (
#                     np.array((4.94, 2.73, 4.26, 2.79)),
#                     np.array((4.74, 2.23, 4.14, 2.28)),
#                     np.array((0.02, 0.022, 0.021, 0.02)),
#                     np.array((0.02, 0.022, 0.021, 0.02)),
#                 )
#             ),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ('PFA', '1', '1'),
#                     ('PFA', '1', '2'),
#                     ('PID', '1', '1'),
#                     ('PID', '1', '2'),
#                 ),
#                 names=['type', 'loc', 'dir'],
#             ),
#             index=range(4),
#         )
#         assessment_instance.asset.cmp_marginal_params = pd.DataFrame(
#             np.full((4, 2), 2.00),
#             index=pd.MultiIndex.from_tuples(
#                 (
#                     ('cmp_1', '1', '1', '0'),
#                     ('cmp_1', '1', '2', '0'),
#                     ('cmp_2', '1', '1', '0'),
#                     ('cmp_2', '1', '2', '0'),
#                 ),
#                 names=['cmp', 'loc', 'dir', 'uid'],
#             ),
#             columns=('Theta_0', 'Blocks'),
#         )
#         assessment_instance.asset.generate_cmp_sample(sample_size=4)
#         assessment_instance.damage.damage_params = pd.DataFrame(
#             np.array(
#                 (
#                     (
#                         1.0,
#                         0.0,
#                         'Peak Interstory Drift Ratio',
#                         'ea',
#                         0.0,
#                         None,
#                         'lognormal',
#                         1e-2,
#                         0.40,
#                         None,
#                         'lognormal',
#                         2e-2,
#                         0.40,
#                         None,
#                         'lognormal',
#                         3e-2,
#                         0.40,
#                         None,
#                         'lognormal',
#                         4e-2,
#                         0.40,
#                     ),
#                     (
#                         1.0,
#                         0.0,
#                         'Peak Interstory Drift Ratio',
#                         'ea',
#                         0.0,
#                         None,
#                         'lognormal',
#                         1e-2,
#                         0.40,
#                         None,
#                         'lognormal',
#                         2e-2,
#                         0.40,
#                         None,
#                         'lognormal',
#                         3e-2,
#                         0.40,
#                         None,
#                         'lognormal',
#                         4e-2,
#                         0.40,
#                     ),
#                 )
#             ),
#             index=['cmp_1', 'cmp_2'],
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ('Demand', 'Directional'),
#                     ('Demand', 'Offset'),
#                     ('Demand', 'Type'),
#                     ('Demand', 'Unit'),
#                     ('Incomplete', ''),
#                     ('LS1', 'DamageStateWeights'),
#                     ('LS1', 'Family'),
#                     ('LS1', 'Theta_0'),
#                     ('LS1', 'Theta_1'),
#                     ('LS2', 'DamageStateWeights'),
#                     ('LS2', 'Family'),
#                     ('LS2', 'Theta_0'),
#                     ('LS2', 'Theta_1'),
#                     ('LS3', 'DamageStateWeights'),
#                     ('LS3', 'Family'),
#                     ('LS3', 'Theta_0'),
#                     ('LS3', 'Theta_1'),
#                     ('LS4', 'DamageStateWeights'),
#                     ('LS4', 'Family'),
#                     ('LS4', 'Theta_0'),
#                     ('LS4', 'Theta_1'),
#                 )
#             ),
#         )
#         assessment_instance.damage.calculate(sample_size=4, dmg_process=dmg_process)
#         assessment_instance.asset.cmp_units = pd.Series(
#             ['ea'] * len(assessment_instance.damage.sample.columns),
#             index=assessment_instance.damage.sample.columns,
#             name='Units',
#             dtype='object',
#         )
#         return deepcopy(assessment_instance.damage)


# def test_save_load_sample(self, damage_model_with_sample, assessment_instance):
#     # saving to a file
#     temp_dir = tempfile.mkdtemp()
#     # convert the sample's index from RangeIndex to int64 (to
#     # match the datatype when it is loaded back; the contents are
#     # the same)
#     damage_model_with_sample.sample.index = (
#         damage_model_with_sample.sample.index.astype('int64')
#     )
#     damage_model_with_sample.save_sample(f'{temp_dir}/damage_model_sample.csv')
#     # loading from the file
#     assessment_instance.damage.load_sample(f'{temp_dir}/damage_model_sample.csv')
#     sample_from_file = assessment_instance.damage.sample

#     # saving to a variable
#     sample_from_variable = damage_model_with_sample.save_sample(save_units=False)
#     pd.testing.assert_frame_equal(
#         sample_from_file,
#         sample_from_variable,
#         check_index_type=False,
#         check_column_type=False,
#     )
#     _, units_from_variable = damage_model_with_sample.save_sample(
#         save_units=True
#     )
#     assert np.all(units_from_variable.to_numpy() == 'ea')


#     def test_load_damage_model(self, damage_model_model_loaded):
#         # should no longer be None
#         assert damage_model_model_loaded.damage_params is not None

#         assert list(damage_model_model_loaded.damage_params.columns) == [
#             ("Demand", "Directional"),
#             ("Demand", "Offset"),
#             ("Demand", "Type"),
#             ("Demand", "Unit"),
#             ("Incomplete", ""),
#             ("LS1", "DamageStateWeights"),
#             ("LS1", "Family"),
#             ("LS1", "Theta_0"),
#             ("LS1", "Theta_1"),
#             ("LS2", "DamageStateWeights"),
#             ("LS2", "Family"),
#             ("LS2", "Theta_0"),
#             ("LS2", "Theta_1"),
#             ("LS3", "DamageStateWeights"),
#             ("LS3", "Family"),
#             ("LS3", "Theta_0"),
#             ("LS3", "Theta_1"),
#             ("LS4", "DamageStateWeights"),
#             ("LS4", "Family"),
#             ("LS4", "Theta_0"),
#             ("LS4", "Theta_1"),
#         ]

#         assert list(damage_model_model_loaded.damage_params.index) == ['B.10.31.001']

#         contents = damage_model_model_loaded.damage_params.to_numpy().reshape(-1)

#         expected_contents = np.array(
#             [
#                 1.0,
#                 0.0,
#                 'Peak Interstory Drift Ratio',
#                 'unitless',
#                 0.0,
#                 '0.950000 | 0.050000',
#                 'lognormal',
#                 0.04,
#                 0.4,
#                 None,
#                 'lognormal',
#                 0.08,
#                 0.4,
#                 None,
#                 'lognormal',
#                 0.11,
#                 0.4,
#                 np.nan,
#                 None,
#                 np.nan,
#                 np.nan,
#             ],
#             dtype=object,
#         )

#         # this comparison was tricky
#         for x, y in zip(contents, expected_contents):
#             if isinstance(x, str):
#                 assert x == y
#             elif x is None:
#                 continue
#             elif np.isnan(x):
#                 continue

#     def test__create_dmg_RVs(self, damage_model_model_loaded):
#         pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)

#         batches = pg_batch.index.get_level_values(0).unique()
#         for PGB_i in batches:
#             PGB = pg_batch.loc[PGB_i]
#             # ensure the following works in each case
#             damage_model_model_loaded._create_dmg_RVs(PGB)

#         # check the output for a single case
#         PGB_i = batches[-1]
#         PGB = pg_batch.loc[PGB_i]

#         capacity_RV_reg, lsds_RV_reg = damage_model_model_loaded._create_dmg_RVs(PGB)

#         assert capacity_RV_reg is not None
#         assert lsds_RV_reg is not None

#         assert list(capacity_RV_reg._variables.keys()) == [
#             'FRG-B.10.31.001-2-2-0-1-1',
#             'FRG-B.10.31.001-2-2-0-1-2',
#             'FRG-B.10.31.001-2-2-0-1-3',
#         ]

#         assert not capacity_RV_reg._sets

#         assert list(lsds_RV_reg._variables.keys()) == [
#             'LSDS-B.10.31.001-2-2-0-1-1',
#             'LSDS-B.10.31.001-2-2-0-1-2',
#             'LSDS-B.10.31.001-2-2-0-1-3',
#         ]

#         assert not lsds_RV_reg._sets

#         # test capacity adjustment: *1.20
#         scaling_specification = {'B.10.31.001-2-2': '*1.20'}
#         (
#             adjusted_capacity_RV_reg,
#             lsds_RV_reg,
#         ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
#         for limit_state in ('1', '2', '3'):
#             val_initial = capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             val_scaling = adjusted_capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             assert val_scaling[0] == val_initial[0] * 1.20
#             assert val_scaling[1] == val_initial[1]
#             assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

#         # test capacity adjustment: /1.20
#         scaling_specification = {'B.10.31.001-2-2': '/1.20'}
#         (
#             adjusted_capacity_RV_reg,
#             lsds_RV_reg,
#         ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
#         for limit_state in ('1', '2', '3'):
#             val_initial = capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             val_scaling = adjusted_capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             assert val_scaling[0] == val_initial[0] / 1.20
#             assert val_scaling[1] == val_initial[1]
#             assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

#         # test capacity adjustment: +0.50
#         scaling_specification = {'B.10.31.001-2-2': '+0.50'}
#         (
#             adjusted_capacity_RV_reg,
#             lsds_RV_reg,
#         ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
#         for limit_state in ('1', '2', '3'):
#             val_initial = capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             val_scaling = adjusted_capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             assert val_scaling[0] == val_initial[0] + 0.50
#             assert val_scaling[1] == val_initial[1]
#             assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

#         # test capacity adjustment: -0.05
#         scaling_specification = {'B.10.31.001-2-2': '-0.05'}
#         (
#             adjusted_capacity_RV_reg,
#             lsds_RV_reg,
#         ) = damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)
#         for limit_state in ('1', '2', '3'):
#             val_initial = capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             val_scaling = adjusted_capacity_RV_reg.RV[
#                 f'FRG-B.10.31.001-2-2-0-1-{limit_state}'
#             ].theta
#             assert val_scaling[0] == val_initial[0] - 0.05
#             assert val_scaling[1] == val_initial[1]
#             assert pd.isna(val_scaling[2]) and pd.isna(val_scaling[2])

#         # edge cases: invalid capacity adjustment
#         scaling_specification = {'B.10.31.001-2-2': 'import os; do_malicious_things'}
#         with pytest.raises(ValueError):
#             damage_model_model_loaded._create_dmg_RVs(PGB, scaling_specification)

#     def test__generate_dmg_sample(self, damage_model_model_loaded):
#         pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
#         batches = pg_batch.index.get_level_values(0).unique()
#         PGB_i = batches[-1]
#         PGB = pg_batch.loc[PGB_i]
#         sample_size = 10

#         # test the _generate_dmg_sample method
#         (
#             capacity_sample,
#             lsds_sample,
#         ) = damage_model_model_loaded._generate_dmg_sample(sample_size, PGB)

#         # run a few checks on the results of the method

#         # note: the method generates random results. We avoid checking
#         # those for equality, because subsequent changes in the code might
#         # break the tests. The functionality of the uq module, which is
#         # used to generate the random samples, is tested with a dedicated
#         # test suite.

#         for res in (capacity_sample, lsds_sample):
#             assert res.shape == (10, 3)

#             assert list(res.columns) == [
#                 ('B.10.31.001', '2', '2', '0', '1', '1'),
#                 ('B.10.31.001', '2', '2', '0', '1', '2'),
#                 ('B.10.31.001', '2', '2', '0', '1', '3'),
#             ]

#             assert list(res.index) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#     def test__get_required_demand_type(self, damage_model_model_loaded):
#         pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
#         batches = pg_batch.index.get_level_values(0).unique()
#         PGB_i = batches[-1]
#         PGB = pg_batch.loc[PGB_i]

#         EDP_req = damage_model_model_loaded._get_required_demand_type(PGB)

#         assert EDP_req == {'PID-2-2': [('B.10.31.001', '2', '2', '0')]}

#     def test__assemble_required_demand_data(
#         self, damage_model_model_loaded, calibration_config_A
#     ):
#         demand_model = damage_model_model_loaded._asmnt.demand
#         demand_model.load_sample(
#             'pelicun/tests/data/model/'
#             'test_DamageModel_assemble_required_demand_data/'
#             'demand_sample.csv'
#         )
#         demand_model.calibrate_model(calibration_config_A)

#         pg_batch = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
#         batches = pg_batch.index.get_level_values(0).unique()

#         expected_demand_dicts = [
#             {'PID-1-1': np.array([0.001])},
#             {'PID-1-2': np.array([0.002])},
#             {'PID-2-1': np.array([0.003])},
#             {'PID-2-2': np.array([0.004])},
#         ]

#         for i, PGB_i in enumerate(batches):
#             PGB = pg_batch.loc[PGB_i]
#             EDP_req = damage_model_model_loaded._get_required_demand_type(PGB)
#             demand_dict = damage_model_model_loaded._assemble_required_demand_data(
#                 EDP_req
#             )
#             assert demand_dict == expected_demand_dicts[i]

#     def test__get_pg_batches_1(self, assessment_instance):
#         damage_model = assessment_instance.damage
#         asset_model = assessment_instance.asset

#         asset_model.cmp_marginal_params = pd.DataFrame(
#             np.full((4, 2), 2.00),
#             index=pd.MultiIndex.from_tuples(
#                 (
#                     ('cmp_1', '1', '1', '0'),
#                     ('cmp_1', '1', '2', '0'),
#                     ('cmp_2', '1', '1', '0'),
#                     ('cmp_2', '1', '2', '0'),
#                 ),
#                 names=['cmp', 'loc', 'dir', 'uid'],
#             ),
#             columns=('Theta_0', 'Blocks'),
#         )

#         damage_model.damage_params = pd.DataFrame(
#             np.empty(2), index=('cmp_1', 'cmp_2'), columns=['ID']
#         )

#         df_1 = damage_model._get_pg_batches(1)
#         assert [i[0] for i in df_1.index] == [1, 2, 3, 4]

#         df_4 = damage_model._get_pg_batches(4)
#         assert [i[0] for i in df_4.index] == [1, 1, 2, 2]

#         df_8 = damage_model._get_pg_batches(8)
#         assert [i[0] for i in df_8.index] == [1, 1, 1, 1]

#     def test__get_pg_batches_2(self, damage_model_model_loaded):
#         # make sure that the method works for different batch sizes
#         for i in (1, 4, 8, 10, 100):
#             damage_model_model_loaded._get_pg_batches(block_batch_size=i)

#         # verify the result is correct for certain cases
#         res = damage_model_model_loaded._get_pg_batches(block_batch_size=1)
#         expected_res = pd.DataFrame(
#             np.array((1, 1, 1, 1)),
#             index=pd.MultiIndex.from_tuples(
#                 (
#                     (1, 'B.10.31.001', '1', '1', '0'),
#                     (2, 'B.10.31.001', '1', '2', '0'),
#                     (3, 'B.10.31.001', '2', '1', '0'),
#                     (4, 'B.10.31.001', '2', '2', '0'),
#                 ),
#                 names=('Batch', 'cmp', 'loc', 'dir', 'uid'),
#             ),
#             columns=('Blocks',),
#         ).astype('Int64')

#         pd.testing.assert_frame_equal(
#             expected_res, res, check_index_type=False, check_column_type=False
#         )

#         res = damage_model_model_loaded._get_pg_batches(block_batch_size=1000)
#         expected_res = pd.DataFrame(
#             np.array((1, 1, 1, 1)),
#             index=pd.MultiIndex.from_tuples(
#                 (
#                     (1, 'B.10.31.001', '1', '1', '0'),
#                     (1, 'B.10.31.001', '1', '2', '0'),
#                     (1, 'B.10.31.001', '2', '1', '0'),
#                     (1, 'B.10.31.001', '2', '2', '0'),
#                 ),
#                 names=('Batch', 'cmp', 'loc', 'dir', 'uid'),
#             ),
#             columns=('Blocks',),
#         ).astype('Int64')

#         pd.testing.assert_frame_equal(
#             expected_res, res, check_index_type=False, check_column_type=False
#         )

#     def test_calculate(self, damage_model_with_sample):
#         # note: Due to inherent randomness, we can't assert the actual
#         # values of this result
#         assert damage_model_with_sample.sample.values.all() >= 0.00
#         assert damage_model_with_sample.sample.values.all() <= 2.00

#     def test_calculate_multilinear_CDF(self, damage_model):
#         # # used for debugging
#         # assessment_instance = assessment.Assessment()
#         # damage_model = assessment_instance.damage

#         demand_model = damage_model._asmnt.demand
#         assessment_instance = damage_model._asmnt
#         asset_model = assessment_instance.asset

#         # A damage calculation test utilizing a multilinear CDF RV for
#         # the capcity.

#         sample_size = 1000

#         # define the demand
#         conversion_factor = assessment_instance.unit_conversion_factors['inps2']
#         demand_model.sample = pd.DataFrame(
#             np.full(sample_size, 0.50 * conversion_factor),
#             columns=(('PGV', '0', '1'),),
#         )

#         # Define the component in the asset model
#         asset_model.cmp_marginal_params = pd.DataFrame(
#             {
#                 'Theta_0': (1.0,),
#                 'Blocks': (1,),
#             },
#             index=pd.MultiIndex.from_tuples(
#                 (('test_component', '0', '1', '0'),),
#                 names=('cmp', 'loc', 'dir', 'uid'),
#             ),
#         )
#         # generate component samples
#         asset_model.generate_cmp_sample()

#         # define fragility curve with multilinear_CDF
#         damage_model.load_damage_model(
#             [
#                 'pelicun/tests/data/model/'
#                 'test_DamageModel_calculate_multilinear_CDF/'
#                 'damage_model.csv'
#             ]
#         )

#         # calculate damage
#         damage_model.calculate(sample_size)

#         res = damage_model.sample.value_counts()
#         assert res.to_dict() == {(1.0, 0.0): 750, (0.0, 1.0): 250}
