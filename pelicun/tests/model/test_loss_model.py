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

import os
import tempfile
from copy import deepcopy
import pytest
import numpy as np
import pandas as pd
from pelicun import model
from pelicun import assessment
from pelicun.tests.model.test_pelicun_model import TestPelicunModel
from pelicun.model.loss_model import LossModel
from pelicun.model.loss_model import RepairModel_Base
from pelicun.model.loss_model import RepairModel_DS
from pelicun.model.loss_model import RepairModel_LF
from pelicun.model.loss_model import _is_for_ds_model
from pelicun.model.loss_model import _is_for_lf_model
from pelicun.warnings import PelicunWarning

# pylint: disable=missing-class-docstring


class TestLossModel(TestPelicunModel):

    @pytest.fixture
    def loss_model(self, assessment_instance):
        return deepcopy(assessment_instance.loss)

    def test___init__(self, loss_model):
        assert loss_model.log
        assert loss_model.ds_model
        with pytest.raises(AttributeError):
            loss_model.xyz = 123

        assert loss_model.ds_model.loss_params is None
        assert loss_model.ds_model.sample is None

        assert len(loss_model._loss_models) == 2

    def test_decision_variables(self, loss_model):
        dvs = ('Carbon', 'Cost', 'Energy', 'Time')
        assert loss_model.decision_variables == dvs
        assert loss_model.ds_model.decision_variables == dvs
        assert loss_model.lf_model.decision_variables == dvs

    def TODO_test_add_loss_map(self):
        pass

    def TODO_test_load_model_parameters(self):
        pass

    def TODO_test_calculate(self):
        pass

    def TODO_test_save_sample(self):
        pass

    def TODO_test_load_sample(self):
        pass

    def TODO_test_aggregate_losses(self):
        pass

    def test__loss_models(self, loss_model):
        models = loss_model._loss_models
        assert len(models) == 2
        assert isinstance(models[0], RepairModel_DS)
        assert isinstance(models[1], RepairModel_LF)

    def test__loss_map(self, loss_model):
        loss_map = loss_map = pd.DataFrame(
            {
                'Repair': ['consequence_A', 'consequence_B'],
            },
            index=['cmp_A', 'cmp_B'],
        )
        # test setter
        loss_model._loss_map = loss_map
        # test getter
        pd.testing.assert_frame_equal(loss_model._loss_map, loss_map)
        for model in loss_model._loss_models:
            pd.testing.assert_frame_equal(model._loss_map, loss_map)

    def test__missing(self, loss_model):
        missing = {
            ('missing.component', 'Time'),
            ('missing.component', 'Energy'),
        }
        # test setter
        loss_model._missing = missing
        # test getter
        assert loss_model._missing == missing
        for model in loss_model._loss_models:
            assert model._missing == missing

    def TODO_test__ensure_loss_parameter_availability(self):
        pass


class TestRepairModel_Base(TestPelicunModel):
    def TODO_test___init__(self):
        pass

    def TODO_test__load_model_parameters(self):
        pass

    def TODO_test__drop_unused_loss_parameters(self):
        pass

    def TODO_test__remove_incomplete_components(self):
        pass

    def TODO_test__get_available(self):
        pass


class TestRepairModel_DS(TestRepairModel_Base):
    def TODO_test__calculate(self):
        pass

    def TODO_test__aggregate_losses(self):
        pass

    def TODO_test__convert_loss_parameter_units(self):
        pass

    def TODO_test__drop_unused_damage_states(self):
        pass

    def TODO_test__create_DV_RVs(self):
        pass

    def TODO_test__calc_median_consequence(self):
        pass


class TestRepairModel_LF(TestRepairModel_Base):
    def TODO_test__calculate(self):
        pass

    def TODO_test__convert_loss_parameter_units(self):
        pass

    def TODO_test__calc_median_consequence(self):
        pass


def test__prep_constant_median_DV():
    median = 10.00
    constant_median_DV = model.loss_model._prep_constant_median_DV(median)
    assert constant_median_DV() == median
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    for value in values:
        assert constant_median_DV(value) == 10.00


def test__prep_bounded_multilinear_median_DV():
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

    with pytest.raises(ValueError):
        f(None)


def TODO_test__is_for_lf_model():
    pass


def TODO_test__is_for_ds_model():
    pass
