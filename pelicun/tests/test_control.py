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

"""
This subpackage performs system tests on the control module of pelicun.

"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import truncnorm as tnorm
from copy import deepcopy

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir))

from pelicun.control import *
from pelicun.uq import mvn_orthotope_density as mvn_od
from pelicun.tests.test_pelicun import prob_allclose, prob_approx

# -----------------------------------------------------------------------------
# FEMA_P58_Assessment
# -----------------------------------------------------------------------------


def test_FEMA_P58_Assessment_DV_uncertainty_dependencies_with_partial_DV_data():
    """
    Perform loss assessment with customized inputs that focus on testing the
    propagation of uncertainty in consequence functions and decision variables
    when not every component has injury and red tag consequences assigned to it.
    Dispersions in other calculation parameters are reduced to negligible
    levels. This allows us to test the results against pre-defined reference
    values in spite of the randomness involved in the calculations.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_11.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_11.out"

    dep_list = ['IND', 'FG', 'PG', 'DIR', 'LOC', 'DS']

    for d in range(7):

        if d > 0:
            dep_COST = dep_list[[0, 1, 2, 3, 4, 5][d - 1]]
            dep_TIME = dep_list[[1, 2, 3, 4, 5, 0][d - 1]]
            dep_RED = dep_list[[2, 3, 4, 5, 0, 1][d - 1]]
            dep_INJ = dep_list[[3, 4, 5, 0, 1, 2][d - 1]]
        else:
            dep_COST = np.random.choice(dep_list)
            dep_TIME = np.random.choice(dep_list)
            dep_RED = np.random.choice(dep_list)
            dep_INJ = np.random.choice(dep_list)
        dep_CT = np.random.choice([True, False])
        dep_ILVL = np.random.choice([True, False])

        # print([dep_COST, dep_TIME, dep_RED, dep_INJ, dep_CT, dep_ILVL], end=' ')

        A = FEMA_P58_Assessment()

        A.read_inputs(DL_input, EDP_input, verbose=False)

        # set the dependencies
        A._AIM_in['dependencies']['rec_costs'] = dep_COST
        A._AIM_in['dependencies']['rec_times'] = dep_TIME
        A._AIM_in['dependencies']['red_tags'] = dep_RED
        A._AIM_in['dependencies']['injuries'] = dep_INJ

        A._AIM_in['dependencies']['cost_and_time'] = dep_CT
        A._AIM_in['dependencies']['injury_lvls'] = dep_ILVL

        A.define_random_variables()

        # ---------------------------------------------- check random variables

        rho_ref = dict(
            IND=np.zeros((16, 16)),
            FG=np.ones((16, 16)),
            PG=np.array([
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
            ]),
            LOC=np.array([
                [1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.],
            ]),
            DIR=np.array([
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            ]),
            DS=np.array([
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
            ])
        )
        np.fill_diagonal(rho_ref['IND'], 1.0)

        RV_REP = deepcopy(A._RV_dict['DV_REP'])
        RV_RED = deepcopy(A._RV_dict['DV_RED'])
        RV_INJ = deepcopy(A._RV_dict['DV_INJ'])

        for r, (RV_DV, RV_tag) in enumerate(
            zip([RV_REP, RV_RED, RV_INJ], ['rep', 'red', 'inj'])):

            assert len(RV_DV._dimension_tags) == [32, 8, 16][r]

            COV_test = RV_DV.COV
            sig_test = np.sqrt(np.diagonal(COV_test))
            rho_test = COV_test / np.outer(sig_test, sig_test)

            if RV_tag == 'rep':

                assert_allclose(RV_DV.theta, np.ones(32))
                assert_allclose(sig_test, np.array(
                    [0.31, 0.71] * 8 + [0.32, 0.72] * 8))

                if dep_CT == True:
                    if (((dep_COST == 'LOC') and (dep_TIME == 'DIR')) or
                        ((dep_COST == 'DIR') and (dep_TIME == 'LOC'))):
                        rho_ref_CT = rho_ref['PG']
                    else:
                        rho_ref_CT = np.maximum(rho_ref[dep_COST],
                                                rho_ref[dep_TIME])

                    assert_allclose(rho_test[:16, :16], rho_ref_CT)
                    assert_allclose(rho_test[16:, 16:], rho_ref_CT)
                    assert_allclose(rho_test[:16, 16:], rho_ref_CT)
                    assert_allclose(rho_test[16:, :16], rho_ref_CT)

                else:
                    assert_allclose(rho_test[:16, :16], rho_ref[dep_COST])
                    assert_allclose(rho_test[16:, 16:], rho_ref[dep_TIME])
                    assert_allclose(rho_test[:16, 16:], np.zeros((16, 16)))
                    assert_allclose(rho_test[16:, :16], np.zeros((16, 16)))

            elif RV_tag == 'red':

                assert_allclose(RV_DV.theta, np.ones(8))
                assert_allclose(sig_test, np.array([0.33, 0.73] * 4))

                assert_allclose(rho_test, rho_ref[dep_RED][:8,:8])

            elif RV_tag == 'inj':

                assert_allclose(RV_DV.theta, np.ones(16))
                assert_allclose(sig_test, np.array(
                    [0.34, 0.74] * 4 + [0.35, 0.75] * 4))

                if dep_ILVL == True:
                    assert_allclose(rho_test[:8, :8], rho_ref[dep_INJ][:8,:8])
                    assert_allclose(rho_test[8:, 8:], rho_ref[dep_INJ][:8,:8])
                    assert_allclose(rho_test[:8, 8:], rho_ref[dep_INJ][:8,:8])
                    assert_allclose(rho_test[8:, :8], rho_ref[dep_INJ][:8,:8])
                else:
                    assert_allclose(rho_test[:8, :8], rho_ref[dep_INJ][:8,:8])
                    assert_allclose(rho_test[8:, 8:], rho_ref[dep_INJ][:8,:8])
                    assert_allclose(rho_test[:8, 8:], np.zeros((8, 8)))
                    assert_allclose(rho_test[8:, :8], np.zeros((8, 8)))

        # ---------------------------------------------------------------------

        A.define_loss_model()

        A.calculate_damage()

        # -------------------------------------------- check damage calculation

        # COL
        # there shall be no collapses
        assert A._COL.describe().T['mean'].values == 0

        # DMG
        DMG_check = A._DMG

        # Fragilities are not tested here, so we only do a few simple checks
        assert np.min(DMG_check.describe().loc['mean'].values) > 0.
        assert np.min(DMG_check.describe().loc['std'].values) > 0.

        # ---------------------------------------------------------------------

        A.calculate_losses()

        # ---------------------------------------------- check loss calculation

        # COST and TIME and INJ
        DV_COST = A._DV_dict['rec_cost'] / DMG_check
        DV_TIME = A._DV_dict['rec_time'] / DMG_check
        DV_INJ_dict = deepcopy(A._DV_dict['injuries'])
        DV_INJ0 = DV_INJ_dict[0] / DMG_check
        DV_INJ1 = DV_INJ_dict[1] / DMG_check

        for dv_i, (DV, DV_tag) in enumerate(
            zip([DV_COST, DV_TIME, DV_INJ0, DV_INJ1],
                ['cost', 'time', 'inj0', 'inj1'])):

            DV_desc = DV.describe().T
            DV_desc_log = np.log(DV).describe().T

            if DV_tag == 'cost':

                # cost consequences in DS1 are lognormal
                mu_ds1_ref = np.exp(np.log(10.) + 0.31 ** 2. / 2.)
                sig_ds1_ref = np.sqrt(
                    np.exp(2 * np.log(10.) + 0.31 ** 2.) * (
                        np.exp(0.31 ** 2.) - 1.))
                assert_allclose(DV_desc['mean'].values[::2], mu_ds1_ref,
                                rtol=0.02)
                assert_allclose(DV_desc['std'].values[::2], sig_ds1_ref,
                                rtol=0.10)
                assert_allclose(DV_desc_log['mean'].values[::2],
                                np.log(10.), atol=0.02)
                assert_allclose(DV_desc_log['std'].values[::2], 0.31,
                                rtol=0.10)

                # cost consequences in DS2 are (truncated) normal
                mu_ds2_ref, var_ds2_ref = tnorm.stats(-1. / 0.71, 1000.,
                                                      loc=1000., scale=710.,
                                                      moments='mv')
                sig_ds2_ref = np.sqrt(var_ds2_ref)
                assert_allclose(DV_desc['mean'].values[1::2], mu_ds2_ref,
                                rtol=0.05)
                assert_allclose(DV_desc['std'].values[1::2], sig_ds2_ref,
                                rtol=0.10)

                # make sure that all damages correspond to positive
                # reconstruction costs
                assert np.all(np.min(DV) > 0.)

            elif DV_tag == 'time':

                # cost consequences in DS1 are (truncated) normal for FG1 and
                # lognormal for FG2
                # DS1 - FG1
                mu_ds1_ref, var_ds1_ref = tnorm.stats(-1. / 0.32, 1000.,
                                                      loc=0.01,
                                                      scale=0.0032,
                                                      moments='mv')
                sig_ds1_ref = np.sqrt(var_ds1_ref)
                assert_allclose(DV_desc['mean'].values[::2][:4], mu_ds1_ref,
                                rtol=0.02)
                assert_allclose(DV_desc['std'].values[::2][:4], sig_ds1_ref,
                                rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[::2][:4]) == pytest.approx(
                    sig_ds1_ref, rel=0.1)

                # DS1 - FG2
                mu_ds1_ref = np.exp(np.log(0.01) + 0.32 ** 2. / 2.)
                sig_ds1_ref = np.sqrt(
                    np.exp(2 * np.log(0.01) + 0.32 ** 2.) * (
                        np.exp(0.32 ** 2.) - 1.))
                assert_allclose(DV_desc['mean'].values[::2][4:], mu_ds1_ref,
                                rtol=0.02)
                assert_allclose(DV_desc['std'].values[::2][4:], sig_ds1_ref,
                                rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[::2][4:]) == pytest.approx(
                    sig_ds1_ref, rel=0.1)
                assert_allclose(DV_desc_log['mean'].values[::2][4:],
                                np.log(0.01), atol=0.02)
                assert_allclose(DV_desc_log['std'].values[::2][4:], 0.32,
                                rtol=0.20)
                assert np.mean(
                    DV_desc_log['std'].values[::2][4:]) == pytest.approx(
                    0.32, rel=0.1)

                # cost consequences in DS2 are lognormal for FG1 and
                # (truncated) normal for FG2
                # DS2 - FG1
                mu_ds2_ref = np.exp(np.log(1.) + 0.72 ** 2. / 2.)
                sig_ds2_ref = np.sqrt(
                    np.exp(2 * np.log(1.) + 0.72 ** 2.) * (
                        np.exp(0.72 ** 2.) - 1.))
                assert_allclose(DV_desc['mean'].values[1::2][:4],
                                mu_ds2_ref, rtol=0.05)
                assert_allclose(DV_desc['std'].values[1::2][:4],
                                sig_ds2_ref, rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[1::2][:4]) == pytest.approx(
                    sig_ds2_ref, rel=0.1)

                assert_allclose(DV_desc_log['mean'].values[1::2][:4],
                                np.log(1.), atol=0.05)
                assert_allclose(DV_desc_log['std'].values[1::2][:4], 0.72,
                                rtol=0.20)
                assert np.mean(
                    DV_desc_log['std'].values[1::2][:4]) == pytest.approx(
                    0.72, rel=0.1)

                # DS2 - FG2
                mu_ds2_ref, var_ds2_ref = tnorm.stats(-1. / 0.72, 1000.,
                                                      loc=1., scale=0.72,
                                                      moments='mv')
                sig_ds2_ref = np.sqrt(var_ds2_ref)
                assert_allclose(DV_desc['mean'].values[1::2][4:],
                                mu_ds2_ref, rtol=0.05)
                assert_allclose(DV_desc['std'].values[1::2][4:],
                                sig_ds2_ref, rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[1::2][4:]) == pytest.approx(
                    sig_ds2_ref, rel=0.1)

                # make sure that all damages correspond to positive
                # reconstruction time
                assert np.all(np.min(DV) > 0.)

            elif DV_tag in ['inj0', 'inj1']:

                # Injuries follow a truncated normal distribution in all cases
                # The beta values provided are coefficients of variation of the
                # non-truncated distribution. These provide the reference mean
                # and standard deviation values for the truncated case.
                mu_ds1, mu_ds2 = {'inj0': [0.5, 0.6],
                                  'inj1': [0.1, 0.2]}[DV_tag]
                beta_ds1, beta_ds2 = {'inj0': [0.34, 0.74],
                                      'inj1': [0.35, 0.75]}[DV_tag]

                # DS1
                # The affected population in DS1 per unit quantity (identical
                # for all FGs and injury levels)
                p_aff = 0.05

                mu_ref, var_ref = tnorm.stats(
                    -1. / beta_ds1, (1. - mu_ds1) / mu_ds1 / beta_ds1,
                    loc=mu_ds1,
                    scale=mu_ds1 * beta_ds1,
                    moments='mv')
                sig_ref = np.sqrt(var_ref)
                mu_ref = mu_ref * p_aff
                sig_ref = sig_ref * p_aff
                assert_allclose(DV_desc['mean'].values[::2],
                                [np.nan]*4 + [mu_ref]*4,
                                rtol=beta_ds1 / 10.)
                assert_allclose(DV_desc['std'].values[::2],
                                [np.nan] * 4 + [sig_ref] * 4,
                                rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[::2][4:]) == pytest.approx(
                    sig_ref, rel=0.1)

                # DS2
                # the affected population in DS1 per unit quantity (identical
                # for all FGs and injury levels)
                p_aff = 0.1
                mu_ref, var_ref = tnorm.stats(-1. / beta_ds2, (
                    1. - mu_ds2) / mu_ds2 / beta_ds2, loc=mu_ds2,
                                              scale=mu_ds2 * beta_ds2,
                                              moments='mv')
                sig_ref = np.sqrt(var_ref)
                mu_ref = mu_ref * p_aff
                sig_ref = sig_ref * p_aff
                assert_allclose(DV_desc['mean'].values[1::2],
                                [np.nan] * 4 + [mu_ref] * 4,
                                rtol=beta_ds2 / 10.)
                assert_allclose(DV_desc['std'].values[1::2],
                                [np.nan] * 4 + [sig_ref] * 4,
                                rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[1::2][4:]) == pytest.approx(
                    sig_ref, rel=0.1)

        # red tags have to be treated separately
        DV_RED = A._DV_dict['red_tag']

        DMG_norm = DMG_check / 25.

        assert len(DV_RED.columns) == 8

        for i in range(8):

            dmg_i = i+8
            is_dam = pd.DataFrame(np.zeros((len(DMG_norm.index), 5)),
                                  columns=range(5))
            is_dam[0] = (DMG_norm.iloc[:, dmg_i] < 0.01)
            is_dam[1] = (DMG_norm.iloc[:, dmg_i] > 0.01) & (
                DMG_norm.iloc[:, dmg_i] < 0.275)
            is_dam[2] = (DMG_norm.iloc[:, dmg_i] > 0.275) & (
                DMG_norm.iloc[:, dmg_i] < 0.525)
            is_dam[3] = (DMG_norm.iloc[:, dmg_i] > 0.525) & (
                DMG_norm.iloc[:, dmg_i] < 0.775)
            is_dam[4] = (DMG_norm.iloc[:, dmg_i] > 0.775)

            mu_red = ([0.50, 0.23185] * 4)[i]
            beta_red = ([0.33, 0.73] * 4)[i]
            mu_ref = np.zeros(5)
            mu_ref[1] = tnorm.cdf(0.25, -1. / beta_red,
                                  (1. - mu_red) / mu_red / beta_red,
                                  loc=mu_red, scale=mu_red * beta_red)
            mu_ref[2] = tnorm.cdf(0.50, -1. / beta_red,
                                  (1. - mu_red) / mu_red / beta_red,
                                  loc=mu_red, scale=mu_red * beta_red)
            mu_ref[3] = tnorm.cdf(0.75, -1. / beta_red,
                                  (1. - mu_red) / mu_red / beta_red,
                                  loc=mu_red, scale=mu_red * beta_red)
            mu_ref[4] = tnorm.cdf(1.00, -1. / beta_red,
                                  (1. - mu_red) / mu_red / beta_red,
                                  loc=mu_red, scale=mu_red * beta_red)

            sample_count = np.array(
                [(DV_RED.iloc[:, i])[is_dam[c]].describe().loc['count'] for
                 c in range(5)])
            mu_test = np.array(
                [(DV_RED.iloc[:, i])[is_dam[c]].describe().loc['mean'] for c
                 in range(5)])

            assert mu_test[0] == 0.
            for step in range(1, 5):
                if sample_count[step] > 0:
                    assert mu_test[step] == pytest.approx(
                        mu_ref[step],
                        abs=5 * 0.4 / np.sqrt(sample_count[step]))

        # CORRELATIONS

        # repair and injury correlations
        DV_REP = pd.concat([DV_COST, DV_TIME], axis=1)
        DV_INJ = pd.concat([DV_INJ0, DV_INJ1], axis=1)

        for DV, RV, dv_tag in zip([DV_REP, DV_INJ, DV_RED],
                                  [RV_REP, RV_INJ, RV_RED],
                                  ['rep', 'inj', 'red']):

            if dv_tag == 'rep':
                # transform the lognormal variables to log scale
                log_flags = ([True, False] * 8 +
                             [False, True] * 4 +
                             [True, False] * 4)
                for c, is_log in enumerate(log_flags):
                    if is_log:
                        DV.iloc[:, c] = np.log(DV.iloc[:, c])

            if dv_tag == 'inj':
                # remove the columns with only nan values from DV
                DV = pd.concat([DV.iloc[:,8:16], DV.iloc[:,24:32]], axis=1)

            elif dv_tag == 'red':
                DV_RED_n = pd.DataFrame(np.ones(DV.shape) * np.nan,
                                        index=DV.index, columns=DV.columns)
                DMG_filter = pd.concat(
                    [(DMG_check.iloc[:, [8, 10, 12, 14]] / 25.0 > 0.275) & (
                         DMG_check.iloc[:, [8, 10, 12, 14]] / 25.0 < 0.525),
                     (DMG_check.iloc[:, [9, 11, 13, 15]] / 25.0 > 0.025) & (
                         DMG_check.iloc[:,
                         [9, 11, 13, 15]] / 25.0 < 0.275)], axis=1)

                DV_RED_n[DMG_filter] = DV_RED[DMG_filter]
                DV = DV_RED_n

            DV_corr = DV.corr()

            # use the correlations specified for the random variable as
            # reference (that we already verified earlier)
            COV_ref = RV.COV
            sig_ref = np.sqrt(np.diagonal(COV_ref))
            rho_ref = COV_ref / np.outer(sig_ref, sig_ref)

            # perform the tests
            for i in range(len(DV_corr.index)):
                for j in range(len(DV_corr.columns)):
                    ref_i = rho_ref[i, j]
                    if ref_i != 0.0:
                        if ref_i > 0.0:
                            assert DV_corr.iloc[i, j] > 0.97 * ref_i
                        else:
                            assert DV_corr.iloc[i, j] < 0.0
                    else:
                        assert DV_corr.iloc[i, j] == pytest.approx(ref_i,
                                                                   abs=0.15)

        # ---------------------------------------------------------------------

        A.aggregate_results()

        # -------------------------------------------- check result aggregation

        # Aggregate results are checked in detail by other tests.
        # Here we only focus on some simple checks to make sure the results
        # make sense.

        S = A._SUMMARY
        SD = S.describe().T

        assert SD.loc[('inhabitants', ''), 'mean'] == 20.0
        assert SD.loc[('inhabitants', ''), 'std'] == 0.0

        assert SD.loc[('collapses', 'collapsed'), 'mean'] == 0.0
        assert SD.loc[('collapses', 'collapsed'), 'std'] == 0.0

        assert_allclose(A._DV_dict['rec_cost'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'cost')])
        assert_allclose(A._DV_dict['rec_time'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'time-sequential')])
        assert_allclose(A._DV_dict['rec_time'].max(axis=1),
                        S.loc[:, ('reconstruction', 'time-parallel')])
        assert_allclose(A._DV_dict['injuries'][0].sum(axis=1),
                        S.loc[:, ('injuries', 'sev1')])
        assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                        S.loc[:, ('injuries', 'sev2')])

        # print()