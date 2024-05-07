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

# class TestLossModel(TestPelicunModel):
#     @pytest.fixture
#     def loss_model(self, assessment_instance):
#         return deepcopy(model.LossModel(assessment_instance))

#     def test_init(self, loss_model):
#         assert loss_model.log

#         assert loss_model.sample is None
#         assert loss_model.loss_type == 'Generic'

#     def test_load_sample_save_sample(self, loss_model):
#         loss_model.loss_params = pd.DataFrame(
#             (
#                 (
#                     "normal",
#                     None,
#                     "25704,17136|5,20",
#                     0.390923,
#                     "USD_2011",
#                     0.0,
#                     "1 EA",
#                 ),
#                 (
#                     "normal",
#                     0.0,
#                     "22.68,15.12|5,20",
#                     0.464027,
#                     "worker_day",
#                     0.0,
#                     "1 EA",
#                 ),
#             ),
#             index=pd.MultiIndex.from_tuples(
#                 (("B.10.41.001a", "Cost"), ("B.10.41.001a", "Time"))
#             ),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ("DS1", "Family"),
#                     ("DS1", "LongLeadTime"),
#                     ("DS1", "Theta_0"),
#                     ("DS1", "Theta_1"),
#                     ("DV", "Unit"),
#                     ("Incomplete", ""),
#                     ("Quantity", "Unit"),
#                 )
#             ),
#         )

#         sample = pd.DataFrame(
#             (
#                 (100.00, 1.00),
#                 (100.00, 1.00),
#             ),
#             index=(0, 1),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ("Cost", "B.10.41.001a", "B.10.41.001a", "1", "1", "1"),
#                     ("Time", "B.10.41.001a", "B.10.41.001a", "1", "1", "1"),
#                 ),
#                 names=("dv", "loss", "dmg", "ds", "loc", "dir"),
#             ),
#         )

#         loss_model.load_sample(sample)

#         pd.testing.assert_frame_equal(
#             sample,
#             loss_model.sample,
#             check_index_type=False,
#             check_column_type=False,
#         )

#         output = loss_model.save_sample(None)
#         output.index = output.index.astype('int64')

#         pd.testing.assert_frame_equal(
#             sample, output, check_index_type=False, check_column_type=False
#         )

#     def test_load_model(self, loss_model):
#         data_path_1 = pd.DataFrame(
#             ((0, "1 EA", "USD_2011", 10000000.00), (0, "1 EA", "worker_day", 12500)),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ("Incomplete", None),
#                     ("Quantity", "Unit"),
#                     ("DV", "Unit"),
#                     ("DS1", "Theta_0"),
#                 )
#             ),
#             index=pd.MultiIndex.from_tuples(
#                 (
#                     ("replacement", "Cost"),
#                     ("replacement", "Time"),
#                 )
#             ),
#         )
#         data_path_2 = 'PelicunDefault/loss_repair_DB_FEMA_P58_2nd.csv'

#         mapping_path = pd.DataFrame(
#             (("B.10.31.001"), ("D.50.92.033k")),
#             columns=["Generic"],
#             index=["DMG-cmp_1", "DMG-cmp_2"],
#         )

#         assert loss_model.loss_map is None
#         assert loss_model.loss_params is None

#         loss_model.load_model([data_path_1, data_path_2], mapping_path)

#         assert loss_model.loss_map.to_dict() == {
#             'Driver': {0: ('DMG', 'cmp_1'), 1: ('DMG', 'cmp_2')},
#             'Consequence': {0: 'B.10.31.001', 1: 'D.50.92.033k'},
#         }
#         cmp_ids = loss_model.loss_params.index.get_level_values(0).unique()
#         assert "B.10.31.001" in cmp_ids
#         assert "D.50.92.033k" in cmp_ids

#     def test_aggregate_losses(self, loss_model):
#         with pytest.raises(NotImplementedError):
#             loss_model.aggregate_losses()

#     def test__generate_DV_sample(self, loss_model):
#         with pytest.raises(NotImplementedError):
#             loss_model._generate_DV_sample(None, None)

# class TestRepairModel(TestPelicunModel):
#     @pytest.fixture
#     def repair_model(self, assessment_instance):
#         return deepcopy(assessment_instance.repair)

#     @pytest.fixture
#     def loss_params_A(self):
#         return pd.DataFrame(
#             (
#                 (
#                     "normal",
#                     None,
#                     "25704,17136|5,20",
#                     0.390923,
#                     "USD_2011",
#                     0.0,
#                     "1 EA",
#                 ),
#                 (
#                     "normal",
#                     0.0,
#                     "22.68,15.12|5,20",
#                     0.464027,
#                     "worker_day",
#                     0.0,
#                     "1 EA",
#                 ),
#             ),
#             index=pd.MultiIndex.from_tuples(
#                 (("some.test.component", "Cost"), ("some.test.component", "Time"))
#             ),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     ("DS1", "Family"),
#                     ("DS1", "LongLeadTime"),
#                     ("DS1", "Theta_0"),
#                     ("DS1", "Theta_1"),
#                     ("DV", "Unit"),
#                     ("Incomplete", ""),
#                     ("Quantity", "Unit"),
#                 )
#             ),
#         )

#     def test_init(self, repair_model):
#         assert repair_model.log

#         assert repair_model.sample is None
#         assert repair_model.loss_type == 'Repair'

#     def test__create_DV_RVs(self, repair_model, loss_params_A):
#         repair_model.loss_params = loss_params_A

#         repair_model.loss_map = pd.DataFrame(
#             ((("DMG", "some.test.component"), "some.test.component"),),
#             columns=("Driver", "Consequence"),
#         )

#         case_list = pd.MultiIndex.from_tuples(
#             (
#                 ("some.test.component", "1", "1", "0", "0"),
#                 ("some.test.component", "2", "2", "0", "1"),
#                 ("some.test.component", "3", "1", "0", "1"),
#             ),
#             names=("cmp", "loc", "dir", "uid", "ds"),
#         )

#         rv_reg = repair_model._create_DV_RVs(case_list)
#         assert list(rv_reg.RV.keys()) == [
#             'Cost-0-1-2-2-0',
#             'Time-0-1-2-2-0',
#             'Cost-0-1-3-1-0',
#             'Time-0-1-3-1-0',
#         ]
#         rvs = list(rv_reg.RV.values())
#         for rv in rvs:
#             print(rv.theta)
#             assert rv.distribution == 'normal'
#         np.testing.assert_array_equal(
#             rvs[0].theta, np.array((1.00, 0.390923, np.nan))
#         )
#         np.testing.assert_array_equal(
#             rvs[1].theta, np.array((1.00, 0.464027, np.nan))
#         )
#         np.testing.assert_array_equal(
#             rvs[2].theta, np.array((1.00, 0.390923, np.nan))
#         )
#         np.testing.assert_array_equal(
#             rvs[3].theta, np.array((1.00, 0.464027, np.nan))
#         )

#     def test__calc_median_consequence(self, repair_model, loss_params_A):
#         repair_model.loss_params = loss_params_A

#         repair_model.loss_map = pd.DataFrame(
#             ((("DMG", "some.test.component"), "some.test.component"),),
#             columns=("Driver", "Consequence"),
#         )

#         eco_qnt = pd.DataFrame(
#             (
#                 (10.00, 0.00),
#                 (0.00, 10.00),
#             ),
#             columns=pd.MultiIndex.from_tuples(
#                 (("some.test.component", "0"), ("some.test.component", "1")),
#                 names=["cmp", "ds"],
#             ),
#         )

#         medians = repair_model._calc_median_consequence(eco_qnt)
#         assert medians['Cost'].to_dict() == {(0, '1'): {0: 25704.0, 1: 22848.0}}
#         assert medians['Time'].to_dict() == {(0, '1'): {0: 22.68, 1: 20.16}}

#     def test__generate_DV_sample(self, repair_model):
#         expected_sample = {
#             (True, True): {
#                 (
#                     'Cost',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '2',
#                     '2',
#                     '0',
#                 ): {0: 25704, 1: 0, 2: 25704, 3: 0},
#                 (
#                     'Cost',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '3',
#                     '1',
#                     '0',
#                 ): {0: 0, 1: 0, 2: 0, 3: 25704},
#                 (
#                     'Time',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '2',
#                     '2',
#                     '0',
#                 ): {0: 22.68, 1: 0.0, 2: 22.68, 3: 0.0},
#                 (
#                     'Time',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '3',
#                     '1',
#                     '0',
#                 ): {0: 0.0, 1: 0.0, 2: 0.0, 3: 22.68},
#             },
#             (True, False): {
#                 (
#                     'Cost',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '2',
#                     '2',
#                     '0',
#                 ): {0: 25704, 1: 0, 2: 25704, 3: 0},
#                 (
#                     'Cost',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '3',
#                     '1',
#                     '0',
#                 ): {0: 0, 1: 0, 2: 0, 3: 25704},
#                 (
#                     'Time',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '2',
#                     '2',
#                     '0',
#                 ): {0: 22.68, 1: 0.0, 2: 22.68, 3: 0.0},
#                 (
#                     'Time',
#                     'some.test.component',
#                     'some.test.component',
#                     '1',
#                     '3',
#                     '1',
#                     '0',
#                 ): {0: 0.0, 1: 0.0, 2: 0.0, 3: 22.68},
#             },
#         }

#         for ecods, ecofl in (
#             (True, True),
#             (True, False),
#         ):  # todo: (False, True), (False, False) fails
#             assessment_instance = repair_model._asmnt

#             assessment_instance.options.eco_scale["AcrossFloors"] = ecofl
#             assessment_instance.options.eco_scale["AcrossDamageStates"] = ecods

#             dmg_quantities = pd.DataFrame(
#                 (
#                     (0.00, 1.00, 0.00),
#                     (1.00, 0.00, 0.00),
#                     (0.00, 1.00, 0.00),
#                     (0.00, 0.00, 1.00),
#                 ),
#                 columns=pd.MultiIndex.from_tuples(
#                     (
#                         ("some.test.component", "1", "1", "0", "0"),
#                         ("some.test.component", "2", "2", "0", "1"),
#                         ("some.test.component", "3", "1", "0", "1"),
#                     ),
#                     names=("cmp", "loc", "dir", "uid", "ds"),
#                 ),
#             )

#             repair_model.loss_map = pd.DataFrame(
#                 ((("DMG", "some.test.component"), "some.test.component"),),
#                 columns=("Driver", "Consequence"),
#             )

#             repair_model.loss_params = pd.DataFrame(
#                 (
#                     (
#                         None,
#                         None,
#                         "25704,17136|5,20",
#                         0.390923,
#                         "USD_2011",
#                         0.0,
#                         "1 EA",
#                     ),
#                     (
#                         None,
#                         0.0,
#                         "22.68,15.12|5,20",
#                         0.464027,
#                         "worker_day",
#                         0.0,
#                         "1 EA",
#                     ),
#                 ),
#                 index=pd.MultiIndex.from_tuples(
#                     (
#                         ("some.test.component", "Cost"),
#                         ("some.test.component", "Time"),
#                     )
#                 ),
#                 columns=pd.MultiIndex.from_tuples(
#                     (
#                         ("DS1", "Family"),
#                         ("DS1", "LongLeadTime"),
#                         ("DS1", "Theta_0"),
#                         ("DS1", "Theta_1"),
#                         ("DV", "Unit"),
#                         ("Incomplete", ""),
#                         ("Quantity", "Unit"),
#                     )
#                 ),
#             )

#             repair_model._generate_DV_sample(dmg_quantities, 4)

#             assert repair_model.sample.to_dict() == expected_sample[(ecods, ecofl)]

#     def test_aggregate_losses(self, repair_model, loss_params_A):
#         repair_model.sample = pd.DataFrame(
#             ((100.00, 1.00),),
#             columns=pd.MultiIndex.from_tuples(
#                 (
#                     (
#                         "Cost",
#                         "some.test.component",
#                         "some.test.component",
#                         "1",
#                         "1",
#                         "1",
#                     ),
#                     (
#                         "Time",
#                         "some.test.component",
#                         "some.test.component",
#                         "1",
#                         "1",
#                         "1",
#                     ),
#                 ),
#                 names=("dv", "loss", "dmg", "ds", "loc", "dir"),
#             ),
#         )

#         repair_model.loss_params = loss_params_A

#         df_agg = repair_model.aggregate_losses()

#         assert df_agg.to_dict() == {
#             ('repair_cost', ''): {0: 100.0},
#             ('repair_time', 'parallel'): {0: 1.0},
#             ('repair_time', 'sequential'): {0: 1.0},
#         }
            
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
