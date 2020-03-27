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

def test_FEMA_P58_Assessment_central_tendencies():
    """
    Perform a loss assessment with customized inputs that reduce the
    dispersion of calculation parameters to negligible levels. This allows us
    to test the results against pre-defined reference values in spite of the
    randomness involved in the calculations.

    """

    base_input_path = 'resources/'
    DL_input = base_input_path + 'input data/' + "DL_input_test.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP.theta == pytest.approx(0.5 * g)
    assert RV_EDP.COV == pytest.approx(0., abs=1e-10)
    assert RV_EDP._distribution_kind == 'lognormal'

    # QNT
    RV_QNT = A._RV_dict['QNT']
    assert RV_QNT.theta == pytest.approx(50., rel=0.01)
    assert RV_QNT.COV == pytest.approx((50. * 1e-4) ** 2., rel=0.01)
    assert RV_QNT._distribution_kind == 'normal'

    # FRG
    RV_FRG = A._RV_dict['FR-T0001.001']
    assert_allclose(RV_FRG.theta, np.array([0.37, 0.5, 0.82]) * g, rtol=0.01)
    COV = deepcopy(RV_FRG.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert_allclose(sig, np.array([0.3, 0.4, 0.5]), rtol=0.01)
    assert_allclose(COV / np.outer(sig, sig), np.ones((3, 3)), rtol=0.01)
    for dist in RV_FRG._distribution_kind:
        assert dist == 'lognormal'

    # RED
    RV_RED = A._RV_dict['DV_RED']
    assert_allclose(RV_RED.theta, np.ones(2), rtol=0.01)
    assert_allclose(RV_RED.COV, np.array([[1, 0], [0, 1]]) * (1e-4) ** 2.,
                    rtol=0.01)
    assert RV_RED._distribution_kind == 'normal'
    assert RV_RED.tr_limits_pre == None
    assert_allclose(RV_RED.tr_limits_post[0], np.array([0., 0.]),
                    rtol=0.01)
    assert_allclose(RV_RED.tr_limits_post[1], np.array([2., 4.]),
                    rtol=0.01)

    # INJ
    RV_INJ = A._RV_dict['DV_INJ']
    assert_allclose(RV_INJ.theta, np.ones(4), rtol=0.01)
    COV = deepcopy(RV_INJ.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert_allclose(sig, np.ones(4) * (1e-4), rtol=0.01)
    rho_target = np.zeros((4, 4))
    np.fill_diagonal(rho_target, 1.)
    assert_allclose(COV / np.outer(sig, sig), rho_target, rtol=0.01)
    assert RV_INJ._distribution_kind == 'normal'
    assert RV_INJ.tr_limits_pre == None
    assert_allclose(RV_INJ.tr_limits_post,
                    np.array([[0., 0., 0., 0.],
                              [10 / 3., 10 / 3., 10., 10.]]), rtol=0.01)

    # REP
    RV_REP = A._RV_dict['DV_REP']
    assert_allclose(RV_REP.theta, np.ones(6), rtol=0.01)
    COV = deepcopy(RV_REP.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert_allclose(sig, np.ones(6) * (1e-4), rtol=0.01)
    rho_target = np.zeros((6, 6))
    np.fill_diagonal(rho_target, 1.)
    assert_allclose(COV / np.outer(sig, sig), rho_target, rtol=0.01)
    for dist in RV_REP._distribution_kind:
        assert dist == 'lognormal'

    # ------------------------------------------------------------------------

    A.define_loss_model()

    A.calculate_damage()

    # ------------------------------------------------ check damage calculation
    # TIME
    T_check = A._TIME.describe().T.loc[['hour','month','weekday?'],:]

    assert_allclose(T_check['mean'], np.array([11.5, 5.5, 5. / 7.]), rtol=0.05)
    assert_allclose(T_check['min'], np.array([0., 0., 0.]), rtol=0.01)
    assert_allclose(T_check['max'], np.array([23., 11., 1.]), rtol=0.01)
    assert_allclose(T_check['50%'], np.array([12., 5., 1.]), atol=1.0)
    assert_allclose(T_check['count'], np.array([10000., 10000., 10000.]),
                    rtol=0.01)

    # POP
    P_CDF = A._POP.describe(np.arange(1, 27) / 27.).iloc[:, 0].values[4:]
    vals, counts = np.unique(P_CDF, return_counts=True)
    assert_allclose(vals, np.array([0., 2.5, 5., 10.]), rtol=0.01)
    assert_allclose(counts, np.array([14, 2, 7, 5]), atol=1)

    # COL
    COL_check = A._COL.describe().T
    assert COL_check['mean'].values[0] == pytest.approx(0.5, rel=0.05)
    assert len(A._ID_dict['non-collapse']) == pytest.approx(5000, rel=0.05)
    assert len(A._ID_dict['collapse']) == pytest.approx(5000, rel=0.05)

    # DMG
    DMG_check = A._DMG.describe().T
    assert_allclose(DMG_check['mean'], np.array([17.074, 17.074, 7.9361]),
                    rtol=0.1, atol=1.0)
    assert_allclose(DMG_check['min'], np.zeros(3), rtol=0.01)
    assert_allclose(DMG_check['max'], np.ones(3) * 50.0157, rtol=0.05)

    # ------------------------------------------------------------------------

    A.calculate_losses()

    # -------------------------------------------------- check loss calculation

    # RED
    DV_RED = A._DV_dict['red_tag'].describe().T
    assert_allclose(DV_RED['mean'], np.array([0.341344, 0.1586555]), rtol=0.1)

    # INJ - collapse
    DV_INJ_C = deepcopy(A._COL[['INJ-0', 'INJ-1']])
    DV_INJ_C.dropna(inplace=True)
    NC_count = DV_INJ_C.describe().T['count'][0]
    assert_allclose(NC_count, np.ones(2) * 5000, rtol=0.05)
    # lvl 1
    vals, counts = np.unique(DV_INJ_C.iloc[:, 0].values, return_counts=True)
    assert_allclose(vals, np.array([0., 2.5, 5., 10.]) * 0.1, rtol=0.01)
    assert_allclose(counts / NC_count, np.array([14, 2, 7, 5]) / 28., atol=0.01, rtol=0.1)
    # lvl 2
    vals, counts = np.unique(DV_INJ_C.iloc[:, 1].values, return_counts=True)
    assert_allclose(vals, np.array([0., 2.5, 5., 10.]) * 0.9, rtol=0.01)
    assert_allclose(counts / NC_count, np.array([14, 2, 7, 5]) / 28., atol=0.01, rtol=0.1)

    # INJ - non-collapse
    DV_INJ_NC = deepcopy(A._DV_dict['injuries'])
    DV_INJ_NC[0].dropna(inplace=True)
    assert_allclose(DV_INJ_NC[0].describe().T['count'], np.ones(2) * 5000,
                    rtol=0.05)
    # lvl 1 DS2
    I_CDF = DV_INJ_NC[0].iloc[:, 0]
    I_CDF = np.around(I_CDF, decimals=3)
    vals, counts = np.unique(I_CDF, return_counts=True)
    assert_allclose(vals, np.array([0., 0.075, 0.15, 0.3]), rtol=0.01)
    target_prob = np.array(
        [0.6586555, 0., 0., 0.] + 0.3413445 * np.array([14, 2, 7, 5]) / 28.)
    assert_allclose(counts / NC_count, target_prob, atol=0.01, rtol=0.1)
    # lvl 1 DS3
    I_CDF = DV_INJ_NC[0].iloc[:, 1]
    I_CDF = np.around(I_CDF, decimals=3)
    vals, counts = np.unique(I_CDF, return_counts=True)
    assert_allclose(vals, np.array([0., 0.075, 0.15, 0.3]), rtol=0.01)
    target_prob = np.array(
        [0.8413445, 0., 0., 0.] + 0.1586555 * np.array([14, 2, 7, 5]) / 28.)
    assert_allclose(counts / NC_count, target_prob, atol=0.01, rtol=0.1)
    # lvl 2 DS2
    I_CDF = DV_INJ_NC[1].iloc[:, 0]
    I_CDF = np.around(I_CDF, decimals=3)
    vals, counts = np.unique(I_CDF, return_counts=True)
    assert_allclose(vals, np.array([0., 0.025, 0.05, 0.1]), rtol=0.01)
    target_prob = np.array(
        [0.6586555, 0., 0., 0.] + 0.3413445 * np.array([14, 2, 7, 5]) / 28.)
    assert_allclose(counts / NC_count, target_prob, atol=0.01, rtol=0.1)
    # lvl2 DS3
    I_CDF = DV_INJ_NC[1].iloc[:, 1]
    I_CDF = np.around(I_CDF, decimals=3)
    vals, counts = np.unique(I_CDF, return_counts=True)
    assert_allclose(vals, np.array([0., 0.025, 0.05, 0.1]), rtol=0.01)
    target_prob = np.array(
        [0.8413445, 0., 0., 0.] + 0.1586555 * np.array([14, 2, 7, 5]) / 28.)
    assert_allclose(counts / NC_count, target_prob, atol=0.01, rtol=0.1)

    # REP
    assert len(A._ID_dict['non-collapse']) == len(A._ID_dict['repairable'])
    assert len(A._ID_dict['irrepairable']) == 0
    # cost
    DV_COST = A._DV_dict['rec_cost']
    # DS1
    C_CDF = DV_COST.iloc[:, 0]
    C_CDF = np.around(C_CDF / 10., decimals=0) * 10.
    vals, counts = np.unique(C_CDF, return_counts=True)
    assert_allclose(vals, [0, 2500], rtol=0.01)
    t_prob = 0.3413445
    assert_allclose(counts / NC_count, [1. - t_prob, t_prob], rtol=0.1)
    # DS2
    C_CDF = DV_COST.iloc[:, 1]
    C_CDF = np.around(C_CDF / 100., decimals=0) * 100.
    vals, counts = np.unique(C_CDF, return_counts=True)
    assert_allclose(vals, [0, 25000], rtol=0.01)
    t_prob = 0.3413445
    assert_allclose(counts / NC_count, [1. - t_prob, t_prob], rtol=0.1)
    # DS3
    C_CDF = DV_COST.iloc[:, 2]
    C_CDF = np.around(C_CDF / 1000., decimals=0) * 1000.
    vals, counts = np.unique(C_CDF, return_counts=True)
    assert_allclose(vals, [0, 250000], rtol=0.01)
    t_prob = 0.1586555
    assert_allclose(counts / NC_count, [1. - t_prob, t_prob], rtol=0.1)
    # time
    DV_TIME = A._DV_dict['rec_time']
    # DS1
    T_CDF = DV_TIME.iloc[:, 0]
    T_CDF = np.around(T_CDF, decimals=1)
    vals, counts = np.unique(T_CDF, return_counts=True)
    assert_allclose(vals, [0, 2.5], rtol=0.01)
    t_prob = 0.3413445
    assert_allclose(counts / NC_count, [1. - t_prob, t_prob], rtol=0.1)
    # DS2
    T_CDF = DV_TIME.iloc[:, 1]
    T_CDF = np.around(T_CDF, decimals=0)
    vals, counts = np.unique(T_CDF, return_counts=True)
    assert_allclose(vals, [0, 25], rtol=0.01)
    t_prob = 0.3413445
    assert_allclose(counts / NC_count, [1. - t_prob, t_prob], rtol=0.1)
    # DS3
    T_CDF = DV_TIME.iloc[:, 2]
    T_CDF = np.around(T_CDF / 10., decimals=0) * 10.
    vals, counts = np.unique(T_CDF, return_counts=True)
    assert_allclose(vals, [0, 250], rtol=0.01)
    t_prob = 0.1586555
    assert_allclose(counts / NC_count, [1. - t_prob, t_prob], rtol=0.1)

    # ------------------------------------------------------------------------

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    S = A._SUMMARY
    SD = S.describe().T

    assert_allclose(S[('event time', 'month')], A._TIME['month'] + 1)
    assert_allclose(S[('event time', 'weekday?')], A._TIME['weekday?'])
    assert_allclose(S[('event time', 'hour')], A._TIME['hour'])
    assert_allclose(S[('inhabitants', '')], A._POP.iloc[:, 0])

    assert SD.loc[('collapses', 'collapsed?'), 'mean'] == pytest.approx(0.5,
                                                                        rel=0.05)
    assert SD.loc[('collapses', 'mode'), 'mean'] == 0.
    assert SD.loc[('collapses', 'mode'), 'count'] == pytest.approx(5000,
                                                                   rel=0.05)

    assert SD.loc[('red tagged?', ''), 'mean'] == pytest.approx(0.5, rel=0.05)
    assert SD.loc[('red tagged?', ''), 'count'] == pytest.approx(5000, rel=0.05)

    for col in ['irrepairable?', 'cost impractical?', 'time impractical?']:
        assert SD.loc[('reconstruction', col), 'mean'] == 0.
        assert SD.loc[('reconstruction', col), 'count'] == pytest.approx(5000,
                                                                         rel=0.05)

    RC = deepcopy(S.loc[:, ('reconstruction', 'cost')])
    RC_CDF = np.around(RC / 1000., decimals=0) * 1000.
    vals, counts = np.unique(RC_CDF, return_counts=True)
    assert_allclose(vals, np.array([0, 2., 3., 25., 250., 300.]) * 1000.)
    t_prob1 = 0.3413445 / 2.
    t_prob2 = 0.1586555 / 2.
    assert_allclose(counts / 10000.,
                    [t_prob2, t_prob1 / 2., t_prob1 / 2., t_prob1, t_prob2,
                     0.5], atol=0.01, rtol=0.1)

    RT = deepcopy(S.loc[:, ('reconstruction', 'time-parallel')])
    RT_CDF = np.around(RT, decimals=0)
    vals, counts = np.unique(RT_CDF, return_counts=True)
    assert_allclose(vals, np.array([0, 2., 3., 25., 250., 300.]))
    t_prob1 = 0.3413445 / 2.
    t_prob2 = 0.1586555 / 2.
    assert_allclose(counts / 10000.,
                    [t_prob2, t_prob1 / 2., t_prob1 / 2., t_prob1, t_prob2,
                     0.5], atol=0.01, rtol=0.1)

    assert_allclose(S.loc[:, ('reconstruction', 'time-parallel')],
                    S.loc[:, ('reconstruction', 'time-sequential')])

    CAS = deepcopy(S.loc[:, ('injuries', 'sev. 1')])
    CAS_CDF = np.around(CAS, decimals=3)
    vals, counts = np.unique(CAS_CDF, return_counts=True)
    assert_allclose(vals, [0, 0.075, 0.15, 0.25, 0.3, 0.5, 1.])
    assert_allclose(counts / 10000.,
                    np.array([35, 1, 3.5, 2, 2.5, 7, 5]) / 56., atol=0.01,
                    rtol=0.1)

    CAS = deepcopy(S.loc[:, ('injuries', 'sev. 2')])
    CAS_CDF = np.around(CAS, decimals=3)
    vals, counts = np.unique(CAS_CDF, return_counts=True)
    assert_allclose(vals, [0, 0.025, 0.05, 0.1, 2.25, 4.5, 9.])
    assert_allclose(counts / 10000.,
                    np.array([35, 1, 3.5, 2.5, 2, 7, 5]) / 56., atol=0.01,
                    rtol=0.1)


def test_FEMA_P58_Assessment_EDP_uncertainty_basic():
    """
    Perform a loss assessment with customized inputs that focus on testing the
    methods used to estimate the multivariate lognormal distribution of EDP
    values. Besides the fitting, this test also evaluates the propagation of
    EDP uncertainty through the analysis. Dispersions in other calculation
    parameters are reduced to negligible levels. This allows us to test the
    results against pre-defined reference values in spite of the  randomness
    involved in the calculations.
    """

    base_input_path = 'resources/'
    DL_input = base_input_path + 'input data/' + "DL_input_test_2.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_2.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'
    assert_allclose(RV_EDP.theta, [9.80665, 12.59198, 0.074081, 0.044932],
                    rtol=0.05)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert_allclose(sig, [0.25, 0.25, 0.3, 0.4], rtol=0.1)
    rho_target = [
        [1.0, 0.6, 0.3, 0.3],
        [0.6, 1.0, 0.3, 0.3],
        [0.3, 0.3, 1.0, 0.7],
        [0.3, 0.3, 0.7, 1.0]]
    assert_allclose(COV / np.outer(sig, sig), rho_target, atol=0.1)

    # ------------------------------------------------------------------------

    A.define_loss_model()

    A.calculate_damage()

    # ------------------------------------------------ check damage calculation
    # COL
    COL_check = A._COL.describe().T
    col_target = 1.0 - mvn_od(np.log([0.074081, 0.044932]),
                              np.array([[1, 0.7], [0.7, 1]]) * np.outer(
                                  [0.3, 0.4], [0.3, 0.4]),
                              upper=np.log([0.1, 0.1]))[0]
    assert COL_check['mean'].values[0] == pytest.approx(col_target, rel=0.1)

    # DMG
    DMG_check = [len(np.where(A._DMG.iloc[:, i] > 0.0)[0]) / 10000. for i in
                 range(8)]

    DMG_1_PID = mvn_od(np.log([0.074081, 0.044932]),
                       np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                                 [0.3, 0.4]),
                       lower=np.log([0.05488, 1e-6]), upper=np.log([0.1, 0.1]))[
        0]
    DMG_2_PID = mvn_od(np.log([0.074081, 0.044932]),
                       np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                                 [0.3, 0.4]),
                       lower=np.log([1e-6, 0.05488]), upper=np.log([0.1, 0.1]))[
        0]
    DMG_1_PFA = mvn_od(np.log([0.074081, 9.80665]),
                       np.array([[1, 0.3], [0.3, 1]]) * np.outer([0.3, 0.25],
                                                                 [0.3, 0.25]),
                       lower=np.log([1e-6, 9.80665]),
                       upper=np.log([0.1, np.inf]))[0]
    DMG_2_PFA = mvn_od(np.log([0.074081, 12.59198]),
                       np.array([[1, 0.3], [0.3, 1]]) * np.outer([0.3, 0.25],
                                                                 [0.3, 0.25]),
                       lower=np.log([1e-6, 9.80665]),
                       upper=np.log([0.1, np.inf]))[0]

    assert DMG_check[0] == pytest.approx(DMG_check[1], rel=0.01)
    assert DMG_check[2] == pytest.approx(DMG_check[3], rel=0.01)
    assert DMG_check[4] == pytest.approx(DMG_check[5], rel=0.01)
    assert DMG_check[6] == pytest.approx(DMG_check[7], rel=0.01)

    assert DMG_check[0] == pytest.approx(DMG_1_PID, rel=0.10)
    assert DMG_check[2] == pytest.approx(DMG_2_PID, rel=0.10)
    assert DMG_check[4] == pytest.approx(DMG_1_PFA, rel=0.10)
    assert DMG_check[6] == pytest.approx(DMG_2_PFA, rel=0.10)

    # ------------------------------------------------------------------------

    A.calculate_losses()

    # -------------------------------------------------- check loss calculation
    # COST
    DV_COST = A._DV_dict['rec_cost']
    DV_TIME = A._DV_dict['rec_time']

    C_target = [0., 250., 1250.]
    T_target = [0., 0.25, 1.25]

    # PG 1011 and 1012
    P_target = [
        mvn_od(np.log([0.074081, 0.044932]),
               np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                         [0.3, 0.4]),
               lower=np.log([1e-6, 1e-6]), upper=np.log([0.05488, 0.1]))[0],
        mvn_od(np.log([0.074081, 0.044932]),
               np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                         [0.3, 0.4]),
               lower=np.log([0.05488, 0.05488]), upper=np.log([0.1, 0.1]))[0],
        mvn_od(np.log([0.074081, 0.044932]),
               np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                         [0.3, 0.4]),
               lower=np.log([0.05488, 1e-6]), upper=np.log([0.1, 0.05488]))[0],
    ]

    for i in [0, 1]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(P_target, P_test, atol=0.02)
        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)

    # PG 1021 and 1022
    P_target = [
        mvn_od(np.log([0.074081, 0.044932]),
               np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                         [0.3, 0.4]),
               lower=np.log([1e-6, 1e-6]), upper=np.log([0.1, 0.05488]))[0],
        mvn_od(np.log([0.074081, 0.044932]),
               np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                         [0.3, 0.4]),
               lower=np.log([0.05488, 0.05488]), upper=np.log([0.1, 0.1]))[0],
        mvn_od(np.log([0.074081, 0.044932]),
               np.array([[1, 0.7], [0.7, 1]]) * np.outer([0.3, 0.4],
                                                         [0.3, 0.4]),
               lower=np.log([1e-6, 0.05488]), upper=np.log([0.05488, 0.1]))[0],
    ]

    for i in [2, 3]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(P_target, P_test, atol=0.02)
        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)

    # PG 2011 and 2012
    P_target = [
        mvn_od(np.log([0.074081, 9.80665, 12.59198]),
               np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.6],
                         [0.3, 0.6, 1.0]]) * np.outer([0.3, 0.25, 0.25],
                                                      [0.3, 0.25, 0.25]),
               lower=np.log([1e-6, 1e-6, 1e-6]),
               upper=np.log([0.1, 9.80665, np.inf]))[0],
        mvn_od(np.log([0.074081, 9.80665, 12.59198]),
               np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.6],
                         [0.3, 0.6, 1.0]]) * np.outer([0.3, 0.25, 0.25],
                                                      [0.3, 0.25, 0.25]),
               lower=np.log([1e-6, 9.80665, 9.80665]),
               upper=np.log([0.1, np.inf, np.inf]))[0],
        mvn_od(np.log([0.074081, 9.80665, 12.59198]),
               np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.6],
                         [0.3, 0.6, 1.0]]) * np.outer([0.3, 0.25, 0.25],
                                                      [0.3, 0.25, 0.25]),
               lower=np.log([1e-6, 9.80665, 1e-6]),
               upper=np.log([0.1, np.inf, 9.80665]))[0],
    ]

    for i in [4, 5]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(P_target, P_test, atol=0.02)
        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)

    # PG 2021 and 2022
    P_target = [
        mvn_od(np.log([0.074081, 9.80665, 12.59198]),
               np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.6],
                         [0.3, 0.6, 1.0]]) * np.outer([0.3, 0.25, 0.25],
                                                      [0.3, 0.25, 0.25]),
               lower=np.log([1e-6, 1e-6, 1e-6]),
               upper=np.log([0.1, np.inf, 9.80665]))[0],
        mvn_od(np.log([0.074081, 9.80665, 12.59198]),
               np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.6],
                         [0.3, 0.6, 1.0]]) * np.outer([0.3, 0.25, 0.25],
                                                      [0.3, 0.25, 0.25]),
               lower=np.log([1e-6, 9.80665, 9.80665]),
               upper=np.log([0.1, np.inf, np.inf]))[0],
        mvn_od(np.log([0.074081, 9.80665, 12.59198]),
               np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.6],
                         [0.3, 0.6, 1.0]]) * np.outer([0.3, 0.25, 0.25],
                                                      [0.3, 0.25, 0.25]),
               lower=np.log([1e-6, 1e-6, 9.80665]),
               upper=np.log([0.1, 9.80665, np.inf]))[0],
    ]

    for i in [6, 7]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(P_target, P_test, atol=0.02)
        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)

    # RED TAG
    RED_check = A._DV_dict['red_tag'].describe().T
    RED_check = (RED_check['mean'] * RED_check['count'] / 10000.).values

    assert RED_check[0] == pytest.approx(RED_check[1], rel=0.01)
    assert RED_check[2] == pytest.approx(RED_check[3], rel=0.01)
    assert RED_check[4] == pytest.approx(RED_check[5], rel=0.01)
    assert RED_check[6] == pytest.approx(RED_check[7], rel=0.01)

    assert RED_check[0] == pytest.approx(DMG_1_PID, rel=0.10)
    assert RED_check[2] == pytest.approx(DMG_2_PID, rel=0.10)
    assert RED_check[4] == pytest.approx(DMG_1_PFA, rel=0.10)
    assert RED_check[6] == pytest.approx(DMG_2_PFA, rel=0.10)

    DMG_on = np.where(A._DMG > 0.0)[0]
    RED_on = np.where(A._DV_dict['red_tag'] > 0.0)[0]
    assert_allclose(DMG_on, RED_on)

    # ------------------------------------------------------------------------

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    P_no_RED_target = mvn_od(np.log([0.074081, 0.044932, 9.80665, 12.59198]),
                             np.array(
                                 [[1.0, 0.7, 0.3, 0.3], [0.7, 1.0, 0.3, 0.3],
                                  [0.3, 0.3, 1.0, 0.6],
                                  [0.3, 0.3, 0.6, 1.0]]) * np.outer(
                                 [0.3, 0.4, 0.25, 0.25],
                                 [0.3, 0.4, 0.25, 0.25]),
                             lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
                             upper=np.log(
                                 [0.05488, 0.05488, 9.80665, 9.80665]))[0]
    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = (1.0 - SD.loc[('red tagged?', ''), 'mean']) * SD.loc[
        ('red tagged?', ''), 'count'] / 10000.

def test_FEMA_P58_Assessment_EDP_uncertainty_detection_limit():
    """
    Perform a loss assessment with customized inputs that focus on testing the
    methods used to estimate the multivariate lognormal distribution of EDP
    values. Besides the fitting, this test also evaluates the propagation of
    EDP uncertainty through the analysis. Dispersions in other calculation
    parameters are reduced to negligible levels. This allows us to test the
    results against pre-defined reference values in spite of the  randomness
    involved in the calculations.
    This test differs from the basic case in having unreliable EDP values above
    a certain limit - a typical feature of interstory drifts in dynamic
    simulations. Such cases should not be a problem if the limits can be
    estimated and they are specified as detection limits in input file.
    """

    base_input_path = 'resources/'
    DL_input = base_input_path + 'input data/' + "DL_input_test_3.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_3.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'

    EDP_theta_target = [9.80665, 12.59198, 0.074081, 0.044932]
    EDP_sig_target = [0.25, 0.25, 0.3, 0.4]
    EDP_rho_target = [
        [1.0, 0.6, 0.3, 0.3],
        [0.6, 1.0, 0.3, 0.3],
        [0.3, 0.3, 1.0, 0.7],
        [0.3, 0.3, 0.7, 1.0]]
    EDP_COV_target = EDP_rho_target * np.outer(EDP_sig_target, EDP_sig_target)

    assert_allclose(RV_EDP.theta, EDP_theta_target, rtol=0.025)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))

    # print(RV_EDP.theta)
    # print(np.sqrt(np.diagonal(COV)))
    # print(COV / np.outer(sig, sig))

    assert_allclose(sig, EDP_sig_target, rtol=0.1)
    assert_allclose(COV / np.outer(sig, sig), EDP_rho_target, atol=0.15)

    # ------------------------------------------------------------------------

    A.define_loss_model()

    A.calculate_damage()

    # ------------------------------------------------ check damage calculation
    # COL
    COL_check = A._COL.describe().T

    col_target = 1.0 - mvn_od(np.log(EDP_theta_target[2:]),
                              EDP_COV_target[2:, 2:],
                              upper=np.log([0.1, 0.1]))[0]

    assert COL_check['mean'].values[0] == prob_approx(col_target, 0.03)

    # DMG
    DMG_check = [len(np.where(A._DMG.iloc[:, i] > 0.0)[0]) / 10000.
                 for i in range(8)]

    DMG_1_PID = mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
                       lower=np.log([0.05488, 1e-6]),
                       upper=np.log([0.1, 0.1]))[0]

    DMG_2_PID = mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
                       lower=np.log([1e-6, 0.05488]),
                       upper=np.log([0.1, 0.1]))[0]

    DMG_1_PFA = mvn_od(np.log(EDP_theta_target), EDP_COV_target,
                       lower=np.log([9.80665, 1e-6, 1e-6, 1e-6]),
                       upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0]

    DMG_2_PFA = mvn_od(np.log(EDP_theta_target), EDP_COV_target,
                       lower=np.log([1e-6, 9.80665, 1e-6, 1e-6]),
                       upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0]

    assert DMG_check[0] == pytest.approx(DMG_check[1], rel=0.01)
    assert DMG_check[2] == pytest.approx(DMG_check[3], rel=0.01)
    assert DMG_check[4] == pytest.approx(DMG_check[5], rel=0.01)
    assert DMG_check[6] == pytest.approx(DMG_check[7], rel=0.01)

    assert DMG_check[0] == prob_approx(DMG_1_PID, 0.03)
    assert DMG_check[2] == prob_approx(DMG_2_PID, 0.03)
    assert DMG_check[4] == prob_approx(DMG_1_PFA, 0.03)
    assert DMG_check[6] == prob_approx(DMG_2_PFA, 0.03)

    # ------------------------------------------------------------------------

    A.calculate_losses()

    # -------------------------------------------------- check loss calculation
    # COST
    DV_COST = A._DV_dict['rec_cost']
    DV_TIME = A._DV_dict['rec_time']

    C_target = [0., 250., 1250.]
    T_target = [0., 0.25, 1.25]

    # PG 1011 and 1012
    P_target = [
        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([1e-6, 1e-6]), upper=np.log([0.05488, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([0.05488, 0.05488]), upper=np.log([0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([0.05488, 1e-6]), upper=np.log([0.1, 0.05488]))[0],
    ]

    for i in [0, 1]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # PG 1021 and 1022
    P_target = [
        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([1e-6, 1e-6]), upper=np.log([0.1, 0.05488]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([0.05488, 0.05488]), upper=np.log([0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([1e-6, 0.05488]), upper=np.log([0.05488, 0.1]))[0],
    ]

    for i in [2, 3]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # PG 2011 and 2012
    P_target = [
        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([9.80665, np.inf, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([9.80665, 9.80665, 1e-6, 1e-6]),
               upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([9.80665, 1e-6, 1e-6, 1e-6]),
               upper=np.log([np.inf, 9.80665, 0.1, 0.1]))[0],
    ]

    for i in [4, 5]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # PG 2021 and 2022
    P_target = [
        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([np.inf, 9.80665, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([9.80665, 9.80665, 1e-6, 1e-6]),
               upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([1e-6, 9.80665, 1e-6, 1e-6]),
               upper=np.log([9.80665, np.inf, 0.1, 0.1]))[0],
    ]

    for i in [6, 7]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # RED TAG
    RED_check = A._DV_dict['red_tag'].describe().T
    RED_check = (RED_check['mean'] * RED_check['count'] / 10000.).values

    assert RED_check[0] == pytest.approx(RED_check[1], rel=0.01)
    assert RED_check[2] == pytest.approx(RED_check[3], rel=0.01)
    assert RED_check[4] == pytest.approx(RED_check[5], rel=0.01)
    assert RED_check[6] == pytest.approx(RED_check[7], rel=0.01)

    assert RED_check[0] == prob_approx(DMG_1_PID, 0.03)
    assert RED_check[2] == prob_approx(DMG_2_PID, 0.03)
    assert RED_check[4] == prob_approx(DMG_1_PFA, 0.03)
    assert RED_check[6] == prob_approx(DMG_2_PFA, 0.03)

    DMG_on = np.where(A._DMG > 0.0)[0]
    RED_on = np.where(A._DV_dict['red_tag'] > 0.0)[0]
    assert_allclose(DMG_on, RED_on)

    # ------------------------------------------------------------------------

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    P_no_RED_target = mvn_od(np.log(EDP_theta_target), EDP_COV_target,
                             lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
                             upper=np.log([9.80665, 9.80665, 0.05488, 0.05488]))[0]

    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = ((1.0 - SD.loc[('red tagged?', ''), 'mean'])
                     * SD.loc[('red tagged?', ''), 'count'] / 10000.)

    assert P_no_RED_target == prob_approx(P_no_RED_test, 0.04)

def test_FEMA_P58_Assessment_EDP_uncertainty_failed_analyses():
    """
    Perform a loss assessment with customized inputs that focus on testing the
    methods used to estimate the multivariate lognormal distribution of EDP
    values. Besides the fitting, this test also evaluates the propagation of
    EDP uncertainty through the analysis. Dispersions in other calculation
    parameters are reduced to negligible levels. This allows us to test the
    results against pre-defined reference values in spite of the  randomness
    involved in the calculations.
    Here we use EDP results with unique values assigned to failed analyses.
    In particular, PID=1.0 and PFA=100.0 are used when an analysis fails.
    These values shall be handled by detection limits of 10 and 100 for PID
    and PFA, respectively.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_4.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_4.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'

    EDP_theta_target = [9.80665, 12.59198, 0.074081, 0.044932]
    EDP_sig_target = [0.25, 0.25, 0.3, 0.4]
    EDP_rho_target = [
        [1.0, 0.6, 0.3, 0.3],
        [0.6, 1.0, 0.3, 0.3],
        [0.3, 0.3, 1.0, 0.7],
        [0.3, 0.3, 0.7, 1.0]]
    EDP_COV_target = EDP_rho_target * np.outer(EDP_sig_target, EDP_sig_target)

    assert_allclose(RV_EDP.theta, EDP_theta_target, rtol=0.025)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))

    #print(RV_EDP.theta)
    #print(np.sqrt(np.diagonal(COV)))
    #print(COV / np.outer(sig, sig))

    assert_allclose(sig, EDP_sig_target, rtol=0.1)
    assert_allclose(COV / np.outer(sig, sig), EDP_rho_target, atol=0.15)

    # ------------------------------------------------------------------------

    A.define_loss_model()

    A.calculate_damage()

    # ------------------------------------------------ check damage calculation
    # COL
    COL_check = A._COL.describe().T
    col_target = 1.0 - mvn_od(np.log(EDP_theta_target[2:]),
                               EDP_COV_target[2:,2:],
                               upper=np.log([0.1, 0.1]))[0]

    assert COL_check['mean'].values[0] == prob_approx(col_target, 0.03)

    # DMG
    DMG_check = [len(np.where(A._DMG.iloc[:, i] > 0.0)[0]) / 10000.
                 for i in range(8)]

    DMG_1_PID = mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:,2:],
                       lower=np.log([0.05488, 1e-6]),
                       upper=np.log([0.1, 0.1]))[0]

    DMG_2_PID = mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
                       lower=np.log([1e-6, 0.05488]),
                       upper=np.log([0.1, 0.1]))[0]

    DMG_1_PFA = mvn_od(np.log(EDP_theta_target), EDP_COV_target,
                       lower=np.log([9.80665, 1e-6, 1e-6, 1e-6]),
                       upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0]

    DMG_2_PFA = mvn_od(np.log(EDP_theta_target), EDP_COV_target,
                       lower=np.log([1e-6, 9.80665, 1e-6, 1e-6]),
                       upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0]

    assert DMG_check[0] == pytest.approx(DMG_check[1], rel=0.01)
    assert DMG_check[2] == pytest.approx(DMG_check[3], rel=0.01)
    assert DMG_check[4] == pytest.approx(DMG_check[5], rel=0.01)
    assert DMG_check[6] == pytest.approx(DMG_check[7], rel=0.01)

    assert DMG_check[0] == prob_approx(DMG_1_PID, 0.03)
    assert DMG_check[2] == prob_approx(DMG_2_PID, 0.03)
    assert DMG_check[4] == prob_approx(DMG_1_PFA, 0.03)
    assert DMG_check[6] == prob_approx(DMG_2_PFA, 0.03)

    # ------------------------------------------------------------------------

    A.calculate_losses()

    # -------------------------------------------------- check loss calculation
    # COST
    DV_COST = A._DV_dict['rec_cost']
    DV_TIME = A._DV_dict['rec_time']

    C_target = [0., 250., 1250.]
    T_target = [0., 0.25, 1.25]

    # PG 1011 and 1012
    P_target = [
        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([1e-6, 1e-6]), upper=np.log([0.05488, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([0.05488, 0.05488]), upper=np.log([0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([0.05488, 1e-6]), upper=np.log([0.1, 0.05488]))[0],
    ]

    for i in [0, 1]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # PG 1021 and 1022
    P_target = [
        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([1e-6, 1e-6]), upper=np.log([0.1, 0.05488]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([0.05488, 0.05488]), upper=np.log([0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target[2:]), EDP_COV_target[2:, 2:],
               lower=np.log([1e-6, 0.05488]), upper=np.log([0.05488, 0.1]))[0],
    ]

    for i in [2, 3]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # PG 2011 and 2012
    P_target = [
        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([9.80665, np.inf, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([9.80665, 9.80665, 1e-6, 1e-6]),
               upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([9.80665, 1e-6, 1e-6, 1e-6]),
               upper=np.log([np.inf, 9.80665, 0.1, 0.1]))[0],
    ]

    for i in [4, 5]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # PG 2021 and 2022
    P_target = [
        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([np.inf, 9.80665, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([9.80665, 9.80665, 1e-6, 1e-6]),
               upper=np.log([np.inf, np.inf, 0.1, 0.1]))[0],

        mvn_od(np.log(EDP_theta_target), EDP_COV_target,
               lower=np.log([1e-6, 9.80665, 1e-6, 1e-6]),
               upper=np.log([9.80665, np.inf, 0.1, 0.1]))[0],
    ]

    for i in [6, 7]:
        C_test, P_test = np.unique(
            np.around(DV_COST.iloc[:, i].values / 10., decimals=0) * 10.,
            return_counts=True)
        C_test = C_test[np.where(P_test > 10)]
        T_test, P_test = np.unique(
            np.around(DV_TIME.iloc[:, i].values * 100., decimals=0) / 100.,
            return_counts=True)
        T_test = T_test[np.where(P_test > 10)]
        P_test = P_test[np.where(P_test > 10)]
        P_test = P_test / 10000.

        assert_allclose(C_target, C_test, rtol=0.001)
        assert_allclose(T_target, T_test, rtol=0.001)
        prob_allclose(P_target, P_test, 0.04)

    # RED TAG
    RED_check = A._DV_dict['red_tag'].describe().T
    RED_check = (RED_check['mean'] * RED_check['count'] / 10000.).values

    assert RED_check[0] == pytest.approx(RED_check[1], rel=0.01)
    assert RED_check[2] == pytest.approx(RED_check[3], rel=0.01)
    assert RED_check[4] == pytest.approx(RED_check[5], rel=0.01)
    assert RED_check[6] == pytest.approx(RED_check[7], rel=0.01)

    assert RED_check[0] == prob_approx(DMG_1_PID, 0.03)
    assert RED_check[2] == prob_approx(DMG_2_PID, 0.03)
    assert RED_check[4] == prob_approx(DMG_1_PFA, 0.03)
    assert RED_check[6] == prob_approx(DMG_2_PFA, 0.03)

    DMG_on = np.where(A._DMG > 0.0)[0]
    RED_on = np.where(A._DV_dict['red_tag'] > 0.0)[0]
    assert_allclose(DMG_on, RED_on)

    # ------------------------------------------------------------------------

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    P_no_RED_target = mvn_od(np.log(EDP_theta_target), EDP_COV_target,
                             lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
                             upper=np.log([9.80665, 9.80665, 0.05488, 0.05488]))[0]

    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = ((1.0 - SD.loc[('red tagged?', ''), 'mean'])
                     * SD.loc[('red tagged?', ''), 'count'] / 10000.)

    assert P_no_RED_target == prob_approx(P_no_RED_test, 0.04)

def test_FEMA_P58_Assessment_EDP_uncertainty_3D():
    """
    Perform a loss assessment with customized inputs that focus on testing the
    methods used to estimate the multivariate lognormal distribution of EDP
    values. Besides the fitting, this test also evaluates the propagation of
    EDP uncertainty through the analysis. Dispersions in other calculation
    parameters are reduced to negligible levels. This allows us to test the
    results against pre-defined reference values in spite of the  randomness
    involved in the calculations.
    In this test we look at the propagation of EDP values provided for two
    different directions. (3D refers to the numerical model used for response
    estimation.)
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_5.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_5.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'
    theta_target = [9.80665, 8.65433, 12.59198, 11.11239,
                    0.074081, 0.063763, 0.044932, 0.036788]
    assert_allclose(RV_EDP.theta, theta_target, rtol=0.05)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))
    sig_target = [0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.4, 0.4]
    assert_allclose(sig, sig_target, rtol=0.1)
    rho_target = np.array([
        [1.0, 0.8, 0.6, 0.5, 0.3, 0.3, 0.3, 0.3],
        [0.8, 1.0, 0.5, 0.6, 0.3, 0.3, 0.3, 0.3],
        [0.6, 0.5, 1.0, 0.8, 0.3, 0.3, 0.3, 0.3],
        [0.5, 0.6, 0.8, 1.0, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 1.0, 0.8, 0.7, 0.6],
        [0.3, 0.3, 0.3, 0.3, 0.8, 1.0, 0.6, 0.7],
        [0.3, 0.3, 0.3, 0.3, 0.7, 0.6, 1.0, 0.8],
        [0.3, 0.3, 0.3, 0.3, 0.6, 0.7, 0.8, 1.0]])
    rho_test = COV / np.outer(sig, sig)
    large_rho_ids = np.where(rho_target>=0.5)
    small_rho_ids = np.where(rho_target<0.5)
    assert_allclose(rho_test[large_rho_ids], rho_target[large_rho_ids], atol=0.1)
    assert_allclose(rho_test[small_rho_ids], rho_target[small_rho_ids], atol=0.2)
    COV_target = rho_target * np.outer(sig_target, sig_target)

    #show_matrix([RV_EDP.theta,theta_target])
    #show_matrix([sig, sig_target])
    #show_matrix(rho_test)

    # ------------------------------------------------------------------------

    A.define_loss_model()

    A.calculate_damage()

    # ------------------------------------------------ check damage calculation
    theta_PID = np.log([0.074081, 0.063763, 0.044932, 0.036788])
    COV_PID = np.array([[1.0, 0.8, 0.7, 0.6],
                        [0.8, 1.0, 0.6, 0.7],
                        [0.7, 0.6, 1.0, 0.8],
                        [0.6, 0.7, 0.8, 1.0]]) * np.outer([0.3, 0.3, 0.4, 0.4],
                                                          [0.3, 0.3, 0.4, 0.4])

    # COL
    COL_check = A._COL.describe().T
    col_target = 1.0 - mvn_od(theta_PID, COV_PID,
                              upper=np.log([0.1, 0.1, 0.1, 0.1]))[0]

    assert COL_check['mean'].values[0] == pytest.approx(col_target, rel=0.1, abs=0.05)

    # DMG
    realization_count = float(A._AIM_in['general']['realizations'])
    DMG_check = [len(np.where(A._DMG.iloc[:, i] > 0.0)[0]) / realization_count for i in
                 range(8)]

    DMG_1_1_PID = mvn_od(theta_PID, COV_PID,
                         lower=np.log([0.05488, 1e-6, 1e-6, 1e-6]),
                         upper=np.log([0.1, 0.1, 0.1, 0.1]))[0]
    DMG_1_2_PID = mvn_od(theta_PID, COV_PID,
                         lower=np.log([1e-6, 0.05488, 1e-6, 1e-6]),
                         upper=np.log([0.1, 0.1, 0.1, 0.1]))[0]
    DMG_2_1_PID = mvn_od(theta_PID, COV_PID,
                         lower=np.log([1e-6, 1e-6, 0.05488, 1e-6]),
                         upper=np.log([0.1, 0.1, 0.1, 0.1]))[0]
    DMG_2_2_PID = mvn_od(theta_PID, COV_PID,
                         lower=np.log([1e-6, 1e-6, 1e-6, 0.05488]),
                         upper=np.log([0.1, 0.1, 0.1, 0.1]))[0]
    DMG_1_1_PFA = mvn_od(np.log(theta_target), COV_target,
                         lower=np.log([9.80665, 1e-6, 1e-6, 1e-6,
                                       1e-6, 1e-6, 1e-6, 1e-6]),
                         upper=np.log([np.inf, np.inf, np.inf, np.inf,
                                       0.1, 0.1, 0.1, 0.1]))[0]
    DMG_1_2_PFA = mvn_od(np.log(theta_target), COV_target,
                         lower=np.log([1e-6, 9.80665, 1e-6, 1e-6,
                                       1e-6, 1e-6, 1e-6, 1e-6]),
                         upper=np.log([np.inf, np.inf, np.inf, np.inf,
                                       0.1, 0.1, 0.1, 0.1]))[0]
    DMG_2_1_PFA = mvn_od(np.log(theta_target), COV_target,
                         lower=np.log([1e-6, 1e-6, 9.80665, 1e-6,
                                       1e-6, 1e-6, 1e-6, 1e-6]),
                         upper=np.log([np.inf, np.inf, np.inf, np.inf,
                                       0.1, 0.1, 0.1, 0.1]))[0]
    DMG_2_2_PFA = mvn_od(np.log(theta_target), COV_target,
                         lower=np.log([1e-6, 1e-6, 1e-6, 9.80665,
                                       1e-6, 1e-6, 1e-6, 1e-6]),
                         upper=np.log([np.inf, np.inf, np.inf, np.inf,
                                       0.1, 0.1, 0.1, 0.1]))[0]

    DMG_ref = [DMG_1_1_PID, DMG_1_2_PID, DMG_2_1_PID, DMG_2_2_PID,
               DMG_1_1_PFA, DMG_1_2_PFA, DMG_2_1_PFA, DMG_2_2_PFA]

    assert_allclose(DMG_check, DMG_ref, rtol=0.10, atol=0.01)

    # ------------------------------------------------------------------------

    A.calculate_losses()

    # -------------------------------------------------- check loss calculation
    # COST
    DV_COST = A._DV_dict['rec_cost']
    DV_TIME = A._DV_dict['rec_time']

    C_target = [0., 249., 624., 1251., 1875.]
    T_target = [0., 0.249, 0.624, 1.251, 1.875]

    # PG 1011
    P_target = [
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([0.05488, 0.1, 0.1, 0.1]))[0],
        mvn_od(theta_PID, COV_PID,
               lower=np.log([0.05488, 0.05488, 0.05488, 0.05488]),
               upper=np.log([0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 0.05488, 0.05488]),
                   upper=np.log([0.1, 0.05488, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 1e-6, 0.05488]),
                   upper=np.log([0.1, 0.1, 0.05488, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 0.05488, 1e-6]),
                   upper=np.log([0.1, 0.1, 0.1, 0.05488]))[0], ]),
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 1e-6, 0.05488]),
                   upper=np.log([0.1, 0.05488, 0.05488, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 1e-6, 1e-6]),
                   upper=np.log([0.1, 0.1, 0.05488, 0.05488]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 0.05488, 1e-6]),
                   upper=np.log([0.1, 0.05488, 0.1, 0.05488]))[0], ]),
        mvn_od(theta_PID, COV_PID, lower=np.log([0.05488, 1e-6, 1e-6, 1e-6]),
               upper=np.log([0.1, 0.05488, 0.05488, 0.05488]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 0].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 0].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    # PG 1012
    P_target = [
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([0.1, 0.05488, 0.1, 0.1]))[0],
        mvn_od(theta_PID, COV_PID,
               lower=np.log([0.05488, 0.05488, 0.05488, 0.05488]),
               upper=np.log([0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 0.05488, 0.05488]),
                   upper=np.log([0.05488, 0.1, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 1e-6, 0.05488]),
                   upper=np.log([0.1, 0.1, 0.05488, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 0.05488, 1e-6]),
                   upper=np.log([0.1, 0.1, 0.1, 0.05488]))[0], ]),
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 1e-6, 0.05488]),
                   upper=np.log([0.05488, 0.1, 0.05488, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 1e-6, 1e-6]),
                   upper=np.log([0.1, 0.1, 0.05488, 0.05488]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 0.05488, 1e-6]),
                   upper=np.log([0.05488, 0.1, 0.1, 0.05488]))[0], ]),
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 0.05488, 1e-6, 1e-6]),
               upper=np.log([0.05488, 0.1, 0.05488, 0.05488]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 1].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 1].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    # PG 1021
    P_target = [
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([0.1, 0.1, 0.05488, 0.1]))[0],
        mvn_od(theta_PID, COV_PID,
               lower=np.log([0.05488, 0.05488, 0.05488, 0.05488]),
               upper=np.log([0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 0.05488, 0.05488]),
                   upper=np.log([0.05488, 0.1, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 0.05488, 0.05488]),
                   upper=np.log([0.1, 0.05488, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 0.05488, 1e-6]),
                   upper=np.log([0.1, 0.1, 0.1, 0.05488]))[0], ]),
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 1e-6, 0.05488, 0.05488]),
                   upper=np.log([0.05488, 0.05488, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 0.05488, 1e-6]),
                   upper=np.log([0.1, 0.05488, 0.1, 0.05488]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 0.05488, 1e-6]),
                   upper=np.log([0.05488, 0.1, 0.1, 0.05488]))[0], ]),
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 1e-6, 0.05488, 1e-6]),
               upper=np.log([0.05488, 0.05488, 0.1, 0.05488]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 2].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 2].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    #print('------------------------')
    #print('P_target')
    #print(P_target)
    #print('------------------------')

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    return 0

    # PG 1022
    P_target = [
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log([0.1, 0.1, 0.1, 0.05488]))[0],
        mvn_od(theta_PID, COV_PID,
               lower=np.log([0.05488, 0.05488, 0.05488, 0.05488]),
               upper=np.log([0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 0.05488, 0.05488]),
                   upper=np.log([0.05488, 0.1, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 0.05488, 0.05488]),
                   upper=np.log([0.1, 0.05488, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 0.05488, 1e-6, 0.05488]),
                   upper=np.log([0.1, 0.1, 0.05488, 0.1]))[0], ]),
        np.sum([
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 1e-6, 0.05488, 0.05488]),
                   upper=np.log([0.05488, 0.05488, 0.1, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([0.05488, 1e-6, 1e-6, 0.05488]),
                   upper=np.log([0.1, 0.05488, 0.05488, 0.1]))[0],
            mvn_od(theta_PID, COV_PID,
                   lower=np.log([1e-6, 0.05488, 1e-6, 0.05488]),
                   upper=np.log([0.05488, 0.1, 0.05488, 0.1]))[0], ]),
        mvn_od(theta_PID, COV_PID, lower=np.log([1e-6, 1e-6, 1e-6, 0.05488]),
               upper=np.log([0.05488, 0.05488, 0.05488, 0.1]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 3].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 5)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 3].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 5)]
    P_test = P_test[np.where(P_test > 5)]
    P_test = P_test / realization_count

    assert_allclose(P_target[:-1], P_test[:4], atol=0.05)
    assert_allclose(C_target[:-1], C_test[:4], rtol=0.001)
    assert_allclose(T_target[:-1], T_test[:4], rtol=0.001)

    # PG 2011
    P_target = [
        mvn_od(np.log(theta_target), COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [9.80665, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [9.80665, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, 9.80665, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, 9.80665, 9.80665, 9.80665, 0.1, 0.1, 0.1, 0.1]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 4].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 4].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    # PG 2012
    P_target = [
        mvn_od(np.log(theta_target), COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, 9.80665, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [9.80665, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, 9.80665, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [9.80665, np.inf, 9.80665, 9.80665, 0.1, 0.1, 0.1, 0.1]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 5].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 5].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    assert_allclose(P_target[:4], P_test[:4], atol=0.05)
    assert_allclose(C_target[:4], C_test[:4], rtol=0.001)
    assert_allclose(T_target[:4], T_test[:4], rtol=0.001)

    # PG 2021
    P_target = [
        mvn_od(np.log(theta_target), COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, np.inf, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [9.80665, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, 9.80665, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [1e-6, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [9.80665, 9.80665, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 6].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 6].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    # PG 2022
    P_target = [
        mvn_od(np.log(theta_target), COV_target,
               lower=np.log([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, np.inf, np.inf, 9.80665, 0.1, 0.1, 0.1, 0.1]))[0],
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [9.80665, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [np.inf, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, np.inf, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        np.sum([
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 1e-6, 9.80665, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, 9.80665, np.inf, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [9.80665, 1e-6, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [np.inf, 9.80665, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0],
            mvn_od(np.log(theta_target), COV_target, lower=np.log(
                [1e-6, 9.80665, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
                   upper=np.log(
                       [9.80665, np.inf, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[
                0], ]),
        mvn_od(np.log(theta_target), COV_target, lower=np.log(
            [1e-6, 1e-6, 1e-6, 9.80665, 1e-6, 1e-6, 1e-6, 1e-6]),
               upper=np.log(
                   [9.80665, 9.80665, 9.80665, np.inf, 0.1, 0.1, 0.1, 0.1]))[0],
    ]

    C_test, P_test = np.unique(
        np.around(DV_COST.iloc[:, 7].values / 3., decimals=0) * 3.,
        return_counts=True)
    C_test = C_test[np.where(P_test > 10)]
    T_test, P_test = np.unique(np.around(DV_TIME.iloc[:, 7].values * 333.33333,
                                         decimals=0) / 333.33333,
                               return_counts=True)
    T_test = T_test[np.where(P_test > 10)]
    P_test = P_test[np.where(P_test > 10)]
    P_test = P_test / realization_count

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    # RED TAG
    RED_check = A._DV_dict['red_tag'].describe().T
    RED_check = (RED_check['mean'] * RED_check['count'] / realization_count).values

    assert_allclose(RED_check, DMG_ref, atol=0.02, rtol=0.10)

    DMG_on = np.where(A._DMG > 0.0)[0]
    RED_on = np.where(A._DV_dict['red_tag'] > 0.0)[0]
    assert_allclose(DMG_on, RED_on)

    # ------------------------------------------------------------------------

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    P_no_RED_target = mvn_od(np.log(theta_target), COV_target,
                             upper=np.log(
                                 [9.80665, 9.80665, 9.80665, 9.80665, 0.05488,
                                  0.05488, 0.05488, 0.05488]))[0]
    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = (1.0 - SD.loc[('red tagged?', ''), 'mean']) * SD.loc[
        ('red tagged?', ''), 'count'] / realization_count

    assert P_no_RED_target == pytest.approx(P_no_RED_test, abs=0.03)

def test_FEMA_P58_Assessment_EDP_uncertainty_single_sample():
    """
    Perform a loss assessment with customized inputs that focus on testing the
    methods used to estimate the multivariate lognormal distribution of EDP
    values. Besides the fitting, this test also evaluates the propagation of
    EDP uncertainty through the analysis. Dispersions in other calculation
    parameters are reduced to negligible levels. This allows us to test the
    results against pre-defined reference values in spite of the  randomness
    involved in the calculations.
    In this test we provide only one structural response result and see if it
    is properly handled as a deterministic value or a random EDP using the
    additional sources of uncertainty.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_6.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_6.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'
    theta_target = [7.634901, 6.85613, 11.685934, 10.565554, 0.061364, 0.048515,
                    0.033256, 0.020352]
    assert_allclose(RV_EDP.theta, theta_target, rtol=0.05)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert_allclose(sig, np.zeros(8), atol=1e-4)
    rho_target = np.zeros((8, 8))
    np.fill_diagonal(rho_target, 1.0)
    COV_target = rho_target * 3e-8
    assert_allclose(COV / np.outer(sig, sig), rho_target, atol=0.1)

    # ------------------------------------------------- perform the calculation

    A.define_loss_model()

    A.calculate_damage()

    A.calculate_losses()

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = (1.0 - SD.loc[('red tagged?', ''), 'mean']) * SD.loc[
        ('red tagged?', ''), 'count'] / 10000.

    assert P_no_RED_test == 0.0

    # -------------------------------------------------------------------------
    # now do the same analysis, but consider additional uncertainty
    # -------------------------------------------------------------------------

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    AU = A._AIM_in['general']['added_uncertainty']

    AU['beta_m'] = 0.3
    AU['beta_gm'] = 0.4

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'
    assert_allclose(RV_EDP.theta, theta_target, rtol=0.05)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))
    sig_target = np.sqrt(1e-8 + 0.3 ** 2. + 0.4 ** 2.)
    assert_allclose(sig, np.ones(8) * sig_target, rtol=0.1)
    rho_target = np.zeros((8, 8))
    np.fill_diagonal(rho_target, 1.0)
    COV_target = rho_target * sig_target ** 2.
    assert_allclose(COV / np.outer(sig, sig), rho_target, atol=0.1)

    # ------------------------------------------------- perform the calculation

    A.define_loss_model()

    A.calculate_damage()

    A.calculate_losses()

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    P_no_RED_target = mvn_od(np.log(theta_target), COV_target,
                             upper=np.log(
                                 [9.80665, 9.80665, 9.80665, 9.80665, 0.05488,
                                  0.05488, 0.05488, 0.05488]))[0]

    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = (1.0 - SD.loc[('red tagged?', ''), 'mean']) * SD.loc[
        ('red tagged?', ''), 'count'] / 10000.

    assert P_no_RED_target == pytest.approx(P_no_RED_test, abs=0.01)

def test_FEMA_P58_Assessment_EDP_uncertainty_zero_variance():
    """
    Perform a loss assessment with customized inputs that focus on testing the
    methods used to estimate the multivariate lognormal distribution of EDP
    values. Besides the fitting, this test also evaluates the propagation of
    EDP uncertainty through the analysis. Dispersions in other calculation
    parameters are reduced to negligible levels. This allows us to test the
    results against pre-defined reference values in spite of the  randomness
    involved in the calculations.
    This test simulates a scenario when one of the EDPs is identical in all
    of the available samples. This results in zero variance in that dimension
    and the purpose of the test is to ensure that such cases are handled
    appropriately.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_7.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_7.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    #with pytest.warns(UserWarning) as e_info:
    if True:
        A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP._distribution_kind == 'lognormal'
    assert RV_EDP.theta[4] == pytest.approx(0.061364, rel=0.05)
    COV = deepcopy(RV_EDP.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert sig[4] < 1e-3
    assert_allclose((COV / np.outer(sig, sig))[4],
                    [0., 0., 0., 0., 1., 0., 0., 0.],
                    atol=1e-6)

    # ------------------------------------------------- perform the calculation

    A.define_loss_model()

    A.calculate_damage()

    A.calculate_losses()

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    S = A._SUMMARY
    SD = S.describe().T

    P_no_RED_test = (1.0 - SD.loc[('red tagged?', ''), 'mean']) * SD.loc[
        ('red tagged?', ''), 'count'] / 10000.

    assert P_no_RED_test == 0.0

def test_FEMA_P58_Assessment_QNT_uncertainty_independent():
    """
    Perform loss assessment with customized inputs that focus on testing the
    propagation of uncertainty in component quantities. Dispersions in other
    calculation parameters are reduced to negligible levels. This allows us to
    test the results against pre-defined reference values in spite of the
    randomness involved in the calculations.
    This test assumes that component quantities are independent.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_8.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_8.out"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # QNT
    RV_QNT = A._RV_dict['QNT']

    COV_test = deepcopy(RV_QNT.COV)
    sig_test = np.sqrt(np.diagonal(COV_test))
    rho_test = COV_test / np.outer(sig_test, sig_test)

    for i, (dist, sig) in enumerate(
        zip(['normal'] * 4 + ['lognormal'] * 4, [25.0] * 4 + [0.4] * 4)):
        assert RV_QNT._distribution_kind[i] == dist
        assert RV_QNT.theta[i] == pytest.approx(25., rel=0.001)
        assert sig_test[i] == pytest.approx(sig, rel=0.001)

    rho_target = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]

    assert_allclose(rho_test, rho_target, rtol=0.001)

    # ------------------------------------------------------------------------

    A.define_loss_model()

    A.calculate_damage()

    # ------------------------------------------------ check damage calculation

    # COL
    # there shall be no collapses
    assert A._COL.describe().T['mean'].values == 0

    # DMG
    DMG_check = A._DMG.describe().T
    mu_test = DMG_check['mean']
    sig_test = DMG_check['std']
    rho_test = A._DMG.corr()

    mu_target_1 = 25.0 + 25.0 * norm.pdf(-1.0) / (1.0 - norm.cdf(-1.0))
    sig_target_1 = np.sqrt(25.0 ** 2.0 * (
            1 - norm.pdf(-1.0) / (1.0 - norm.cdf(-1.0)) - (
                norm.pdf(-1.0) / (1.0 - norm.cdf(-1.0))) ** 2.0))
    mu_target_2 = np.exp(np.log(25.0) + 0.4 ** 2. / 2.)
    sig_target_2 = np.sqrt(
        (np.exp(0.4 ** 2.0) - 1.0) * np.exp(2 * np.log(25.0) + 0.4 ** 2.0))

    assert_allclose(mu_test[:4], mu_target_1, rtol=0.05)
    assert_allclose(mu_test[4:], mu_target_2, rtol=0.05)
    assert_allclose(sig_test[:4], sig_target_1, rtol=0.05)
    assert_allclose(sig_test[4:], sig_target_2, rtol=0.05)
    assert_allclose(rho_test, rho_target, atol=0.05)

    # ------------------------------------------------------------------------

    A.calculate_losses()

    # -------------------------------------------------- check loss calculation

    DV_COST = A._DV_dict['rec_cost'] / A._DMG

    rho_DV_target = [
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ]

    assert_allclose(DV_COST.corr(), rho_DV_target, atol=0.05)

    # Uncertainty in decision variables is controlled by the correlation
    # between damages
    RND = [truncnorm.rvs(-1., np.inf, loc=25, scale=25, size=10000) for i in
           range(4)]
    RND = np.sum(RND, axis=0)
    P_target_PID = np.sum(RND > 90.) / 10000.
    P_test_PID = np.sum(DV_COST.iloc[:, 0] < 10.01) / 10000.
    assert P_target_PID == pytest.approx(P_test_PID, rel=0.02)

    RND = [np.exp(norm.rvs(loc=np.log(25.), scale=0.4, size=10000)) for i in
           range(4)]
    RND = np.sum(RND, axis=0)
    P_target_PFA = np.sum(RND > 90.) / 10000.
    P_test_PFA = np.sum(DV_COST.iloc[:, 4] < 10.01) / 10000.
    assert P_target_PFA == pytest.approx(P_test_PFA, rel=0.02)

    # the same checks can be performed for reconstruction time
    DV_TIME = A._DV_dict['rec_time'] / A._DMG

    assert_allclose(DV_TIME.corr(), rho_DV_target, atol=0.05)

    P_test_PID = np.sum(DV_TIME.iloc[:, 0] < 0.0101) / 10000.
    assert P_target_PID == pytest.approx(P_test_PID, rel=0.02)

    P_test_PFA = np.sum(DV_TIME.iloc[:, 4] < 0.0101) / 10000.
    assert P_target_PFA == pytest.approx(P_test_PFA, rel=0.02)

    # injuries...
    DV_INJ_dict = deepcopy(A._DV_dict['injuries'])
    DV_INJ0 = (DV_INJ_dict[0] / A._DMG).describe()
    DV_INJ1 = (DV_INJ_dict[1] / A._DMG).describe()

    assert_allclose(DV_INJ0.loc['mean', :][:4], np.ones(4) * 0.025, rtol=0.001)
    assert_allclose(DV_INJ0.loc['mean', :][4:], np.ones(4) * 0.1, rtol=0.001)
    assert_allclose(DV_INJ1.loc['mean', :][:4], np.ones(4) * 0.005, rtol=0.001)
    assert_allclose(DV_INJ1.loc['mean', :][4:], np.ones(4) * 0.02, rtol=0.001)

    assert_allclose(DV_INJ0.loc['std', :], np.zeros(8), atol=1e-4)
    assert_allclose(DV_INJ1.loc['std', :], np.zeros(8), atol=1e-4)

    # and for red tag...
    # Since every component is damaged in every realization, the red tag
    # results should all be 1.0
    assert_allclose(A._DV_dict['red_tag'], np.ones((10000, 8)))

    # ------------------------------------------------------------------------

    A.aggregate_results()

    # ------------------------------------------------ check result aggregation

    S = A._SUMMARY
    SD = S.describe().T

    assert SD.loc[('inhabitants', ''), 'mean'] == 20.0
    assert SD.loc[('inhabitants', ''), 'std'] == 0.0

    assert SD.loc[('collapses', 'collapsed?'), 'mean'] == 0.0
    assert SD.loc[('collapses', 'collapsed?'), 'std'] == 0.0

    assert SD.loc[('red tagged?', ''), 'mean'] == 1.0
    assert SD.loc[('red tagged?', ''), 'std'] == 0.0

    assert np.corrcoef(S.loc[:, ('reconstruction', 'cost')],
                       S.loc[:, ('reconstruction', 'time-sequential')])[
               0, 1] == pytest.approx(1.0)

    assert_allclose(A._DV_dict['rec_cost'].sum(axis=1),
                    S.loc[:, ('reconstruction', 'cost')])
    assert_allclose(A._DV_dict['rec_time'].sum(axis=1),
                    S.loc[:, ('reconstruction', 'time-sequential')])
    assert_allclose(A._DV_dict['rec_time'].max(axis=1),
                    S.loc[:, ('reconstruction', 'time-parallel')])
    assert_allclose(A._DV_dict['injuries'][0].sum(axis=1),
                    S.loc[:, ('injuries', 'sev. 1')])
    assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                    S.loc[:, ('injuries', 'sev. 2')])

def test_FEMA_P58_Assessment_QNT_uncertainty_dependencies():
    """
    Perform loss assessment with customized inputs that focus on testing the
    propagation of uncertainty in component quantities. Dispersions in other
    calculation parameters are reduced to negligible levels. This allows us to
    test the results against pre-defined reference values in spite of the
    randomness involved in the calculations.
    This test checks if dependencies between component quantities are handled
    appropriately.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_8.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_8.out"

    for dep in ['FG', 'PG', 'DIR', 'LOC']:

        A = FEMA_P58_Assessment()

        with pytest.warns(UserWarning) as e_info:
            A.read_inputs(DL_input, EDP_input, verbose=False)

        A._AIM_in['dependencies']['quantities'] = dep

        A.define_random_variables()

        # ---------------------------------------------- check random variables

        # QNT
        RV_QNT = A._RV_dict['QNT']

        COV_test = deepcopy(RV_QNT.COV)
        sig_test = np.sqrt(np.diagonal(COV_test))
        rho_test = COV_test / np.outer(sig_test, sig_test)

        for i, (dist, sig) in enumerate(
            zip(['normal'] * 4 + ['lognormal'] * 4, [25.0] * 4 + [0.4] * 4)):
            assert RV_QNT._distribution_kind[i] == dist
            assert RV_QNT.theta[i] == pytest.approx(25., rel=0.001)
            assert sig_test[i] == pytest.approx(sig, rel=0.001)

        if dep == 'FG':
            rho_target = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ])
        elif dep == 'PG':
            rho_target = np.array([
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ])
        elif dep == 'DIR':
            rho_target = np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1],
            ])
        elif dep == 'LOC':
            rho_target = np.array([
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
            ])

        assert_allclose(rho_test, rho_target, rtol=0.001)

        # ---------------------------------------------------------------------

        A.define_loss_model()

        A.calculate_damage()

        # -------------------------------------------- check damage calculation

        # COL
        # there shall be no collapses
        assert A._COL.describe().T['mean'].values == 0

        # DMG
        # Because the correlations are enforced after truncation, the marginals
        # shall be unaffected by the correlation structure. Hence, the
        # distribution of damaged quantities within a PG shall be identical in
        # all dep cases.
        # The specified dependencies are apparent in the correlation between
        # damaged quantities in various PGs.

        DMG_check = A._DMG.describe().T
        mu_test = DMG_check['mean']
        sig_test = DMG_check['std']
        rho_test = A._DMG.corr()

        mu_target_1 = 25.0 + 25.0 * norm.pdf(-1.0) / (1.0 - norm.cdf(-1.0))
        sig_target_1 = np.sqrt(25.0 ** 2.0 * (
                1 - norm.pdf(-1.0) / (1.0 - norm.cdf(-1.0)) - (
                    norm.pdf(-1.0) / (1.0 - norm.cdf(-1.0))) ** 2.0))
        mu_target_2 = np.exp(np.log(25.0) + 0.4 ** 2. / 2.)
        sig_target_2 = np.sqrt(
            (np.exp(0.4 ** 2.0) - 1.0) * np.exp(2 * np.log(25.0) + 0.4 ** 2.0))

        assert_allclose(mu_test[:4], mu_target_1, rtol=0.05)
        assert_allclose(mu_test[4:], mu_target_2, rtol=0.05)
        assert_allclose(sig_test[:4], sig_target_1, rtol=0.05)
        assert_allclose(sig_test[4:], sig_target_2, rtol=0.05)
        assert_allclose(rho_test, rho_target, atol=0.05)

        # ---------------------------------------------------------------------

        A.calculate_losses()

        # ---------------------------------------------- check loss calculation

        DV_COST = A._DV_dict['rec_cost'] / A._DMG

        # After the DVs are normalized by the damaged quantities, the resulting
        # samples show the correlations between the DV_measure (such as
        # reconstruction cost) / 1 unit of damaged component. Because this
        # consequences are perfectly correlated among the components of a
        # fragility group by definition, the quadrants on the main diagonal
        # will follow the matrix presented below. If there are additional
        # correlations defined between component quantities in different
        # fragility groups (i.e. the off-diagonal quadrants of the rho matrix),
        # those will be preserved in the consequences. Therefore, the
        # off-diagonal quadrants need to be updated with those from rho_target
        # to get an appropriate rho_DV_target.

        rho_DV_target = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ])
        rho_DV_target[:4, 4:] = rho_target[:4, 4:]
        rho_DV_target[4:, :4] = rho_target[:4, 4:]

        assert_allclose(DV_COST.corr(), rho_DV_target, atol=0.05)

        # uncertainty in decision variables is controlled by the correlation
        # between damages
        P_test_PID = np.sum(DV_COST.iloc[:, 0] < 10.01) / 10000.
        P_test_PFA = np.sum(DV_COST.iloc[:, 4] < 10.01) / 10000.

        # the first component quantities follow a truncated multivariate normal
        # distribution
        mu_target_PID = mu_target_1 * 4.
        sig_target_PID = np.sqrt(
            sig_target_1 ** 2. * np.sum(rho_target[:4, :4]))
        mu_target_PID_b = mu_target_PID
        sig_target_PID_b = sig_target_PID
        alpha = 100.
        i = 0
        while (np.log(
            np.abs(alpha / (mu_target_PID_b / sig_target_PID_b))) > 0.001) and (
            i < 10):
            alpha = -mu_target_PID_b / sig_target_PID_b
            mu_target_PID_b = mu_target_PID - sig_target_PID_b * norm.pdf(
                alpha) / (1.0 - norm.cdf(alpha))
            sig_target_PID_b = sig_target_PID / np.sqrt(
                (1.0 + alpha * norm.pdf(alpha) / (1.0 - norm.cdf(alpha))))
            i += 1
        xi = (90 - mu_target_PID_b) / sig_target_PID_b
        P_target_PID = 1.0 - (norm.cdf(xi) - norm.cdf(alpha)) / (
                1.0 - norm.cdf(alpha))

        assert P_target_PID == pytest.approx(P_test_PID, rel=0.05)

        # the second component quantities follow a multivariate lognormal
        # distribution
        mu_target_PFA = mu_target_2 * 4.
        sig_target_PFA = np.sqrt(
            sig_target_2 ** 2. * np.sum(rho_target[4:, 4:]))
        sig_target_PFA_b = np.sqrt(
            np.log(sig_target_PFA ** 2.0 / mu_target_PFA ** 2.0 + 1.0))
        mu_target_PFA_b = np.log(mu_target_PFA) - sig_target_PFA_b ** 2.0 / 2.
        xi = np.log(90)
        P_target_PFA = 1.0 - norm.cdf(xi, loc=mu_target_PFA_b,
                                      scale=sig_target_PFA_b)

        assert P_target_PFA == pytest.approx(P_test_PFA, rel=0.05)

        # the same checks can be performed for reconstruction time
        DV_TIME = A._DV_dict['rec_time'] / A._DMG

        assert_allclose(DV_TIME.corr(), rho_DV_target, atol=0.05)

        P_test_PID = np.sum(DV_TIME.iloc[:, 0] < 0.0101) / 10000.
        assert P_target_PID == pytest.approx(P_test_PID, rel=0.05)

        P_test_PFA = np.sum(DV_TIME.iloc[:, 4] < 0.0101) / 10000.
        assert P_target_PFA == pytest.approx(P_test_PFA, rel=0.05)

        # injuries...
        # Every component is damaged in every realization in this test. Once
        # normalized by the quantity of components, the number of injuries
        # shall be identical and unaffected by the correlation between
        # component quantities.

        DV_INJ_dict = deepcopy(A._DV_dict['injuries'])
        DV_INJ0 = (DV_INJ_dict[0] / A._DMG).describe()
        DV_INJ1 = (DV_INJ_dict[1] / A._DMG).describe()

        assert_allclose(DV_INJ0.loc['mean', :][:4], np.ones(4) * 0.025,
                        rtol=0.001)
        assert_allclose(DV_INJ0.loc['mean', :][4:], np.ones(4) * 0.1,
                        rtol=0.001)
        assert_allclose(DV_INJ1.loc['mean', :][:4], np.ones(4) * 0.005,
                        rtol=0.001)
        assert_allclose(DV_INJ1.loc['mean', :][4:], np.ones(4) * 0.02,
                        rtol=0.001)

        assert_allclose(DV_INJ0.loc['std', :], np.zeros(8), atol=1e-4)
        assert_allclose(DV_INJ1.loc['std', :], np.zeros(8), atol=1e-4)

        # and for red tag...
        # since every component is damaged in every realization, the red tag
        # results should all be 1.0
        assert_allclose(A._DV_dict['red_tag'], np.ones((10000, 8)))

        # ---------------------------------------------------------------------

        A.aggregate_results()

        # -------------------------------------------- check result aggregation

        S = A._SUMMARY
        SD = S.describe().T

        assert SD.loc[('inhabitants', ''), 'mean'] == 20.0
        assert SD.loc[('inhabitants', ''), 'std'] == 0.0

        assert SD.loc[('collapses', 'collapsed?'), 'mean'] == 0.0
        assert SD.loc[('collapses', 'collapsed?'), 'std'] == 0.0

        assert SD.loc[('red tagged?', ''), 'mean'] == 1.0
        assert SD.loc[('red tagged?', ''), 'std'] == 0.0

        assert np.corrcoef(S.loc[:, ('reconstruction', 'cost')],
                           S.loc[:, ('reconstruction', 'time-sequential')])[
                   0, 1] == pytest.approx(1.0)

        assert_allclose(A._DV_dict['rec_cost'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'cost')])
        assert_allclose(A._DV_dict['rec_time'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'time-sequential')])
        assert_allclose(A._DV_dict['rec_time'].max(axis=1),
                        S.loc[:, ('reconstruction', 'time-parallel')])
        assert_allclose(A._DV_dict['injuries'][0].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 1')])
        assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 2')])

def test_FEMA_P58_Assessment_FRAG_uncertainty_dependencies():
    """
    Perform loss assessment with customized inputs that focus on testing the
    propagation of uncertainty in component fragilities. Dispersions in other
    calculation parameters are reduced to negligible levels. This allows us to
    test the results against pre-defined reference values in spite of the
    randomness involved in the calculations.
    """

    idx = pd.IndexSlice

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_9.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_9.out"

    for dep in ['IND', 'PG', 'DIR', 'LOC', 'ATC', 'CSG', 'DS']:

        A = FEMA_P58_Assessment()

        with pytest.warns(UserWarning) as e_info:
            A.read_inputs(DL_input, EDP_input, verbose=False)

        A._AIM_in['dependencies']['fragilities'] = dep

        A.define_random_variables()

        # ---------------------------------------------- check random variables
        fr_keys = []
        for key in A._RV_dict.keys():
            if 'FR' in key:
                fr_keys.append(key)

        dimtag_target = [4 * 2 * 3, 20 * 2 * 3 * 3, 20 * 2 * 3 * 3,
                         20 * 2 * 3 * 3]
        theta_target = [[0.04, 0.08], [0.04, 0.06, 0.08],
                        [2.4516, 4.9033, 9.80665], [2.4516, 4.9033, 9.80665]]
        sig_target = [[0.5, 0.25], [1.0, 0.5, 0.25], [1.0, 0.5, 0.25],
                      [1.0, 0.5, 0.25]]

        if dep == 'IND':
            rho_target = np.zeros((24, 24))
            np.fill_diagonal(rho_target, 1.0)

            rho_sum = 360

        elif dep == 'PG':
            rho_target = np.ones((24, 24))

            rho_sum = 360 ** 2.

        elif dep == 'DIR':
            rho_target = [
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.]]

            rho_sum = (20 * 2 * 3) ** 2. * 3

        elif dep == 'LOC':
            rho_target = [
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1.]]

            rho_sum = (20 * 3) ** 2. * (2 * 9)

        elif dep in ['ATC', 'CSG']:
            rho_target = [
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]]

            rho_sum = (20 * 3) ** 2. * (2 * 3)

        elif dep == 'DS':
            rho_target = [
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]]

            rho_sum = 3 ** 2 * (20 * 2 * 3)

        for k, key in enumerate(sorted(fr_keys)):
            RV_FR = deepcopy(A._RV_dict[key])
            assert len(RV_FR._dimension_tags) == dimtag_target[k]

            COV_test = RV_FR.COV
            sig_test = np.sqrt(np.diagonal(COV_test))
            rho_test = COV_test / np.outer(sig_test, sig_test)

            if k == 0:
                theta_test = pd.DataFrame(
                    np.reshape(RV_FR.theta, (12, 2))).describe()
                sig_test = pd.DataFrame(
                    np.reshape(sig_test, (12, 2))).describe()
            else:
                theta_test = pd.DataFrame(
                    np.reshape(RV_FR.theta, (120, 3))).describe()
                sig_test = pd.DataFrame(
                    np.reshape(sig_test, (120, 3))).describe()

            assert_allclose(theta_test.loc['mean', :].values, theta_target[k],
                            rtol=1e-4)
            assert_allclose(theta_test.loc['std', :].values,
                            np.zeros(np.array(theta_target[k]).shape),
                            atol=1e-10)

            assert_allclose(sig_test.loc['mean', :].values, sig_target[k],
                            rtol=1e-4)
            assert_allclose(sig_test.loc['std', :].values,
                            np.zeros(np.array(sig_target[k]).shape), atol=1e-10)

            if k == 0:
                # we perform the detailed verification of rho for the first case
                # only (because the others are 360x360 matrices)
                assert_allclose(rho_test, rho_target)

            else:
                # for the other cases we check the number of ones in the matrix
                assert np.sum(rho_test) == rho_sum

        # ---------------------------------------------------------------------

        A.define_loss_model()

        A.calculate_damage()

        # -------------------------------------------- check damage calculation
        # COL
        # there shall be no collapses
        assert A._COL.describe().T['mean'].values == 0

        # DMG
        DMG_check = A._DMG

        # start with checking the damage correlations
        for k in range(4):
            DMG_corr = DMG_check.loc[:, idx[k + 1, :, :]].corr()

            if k == 0:
                DMG_corr = DMG_corr.iloc[:8, :8]

                if dep in ['IND', 'ATC', 'CSG', 'DS']:
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 1.0,-0.1, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0],
                    ])
                elif dep == 'PG':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1],
                        [-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0],
                        [ 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1],
                        [-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0],
                        [ 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1],
                        [-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0],
                        [ 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1],
                        [-0.1, 1.0,-0.1, 1.0,-0.1, 1.0,-0.1, 1.0],
                    ])
                elif dep == 'DIR':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [ 1.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 1.0,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 1.0],
                        [ 0.0, 0.0, 0.0, 0.0, 1.0,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 1.0],
                    ])
                elif dep == 'LOC':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1, 0.0, 0.0, 1.0,-0.1, 0.0, 0.0],
                        [-0.1, 1.0, 0.0, 0.0,-0.1, 1.0, 0.0, 0.0],
                        [ 0.0, 0.0, 1.0,-0.1, 0.0, 0.0, 1.0,-0.1],
                        [ 0.0, 0.0,-0.1, 1.0, 0.0, 0.0,-0.1, 1.0],
                        [ 1.0,-0.1, 0.0, 0.0, 1.0,-0.1, 0.0, 0.0],
                        [-0.1, 1.0, 0.0, 0.0,-0.1, 1.0, 0.0, 0.0],
                        [ 0.0, 0.0, 1.0,-0.1, 0.0, 0.0, 1.0,-0.1],
                        [ 0.0, 0.0,-0.1, 1.0, 0.0, 0.0,-0.1, 1.0],
                    ])

            if k == 1:
                DMG_corr = DMG_corr.iloc[:12, :12]

                if dep in ['IND', 'ATC', 'CSG', 'DS']:
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'PG':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'DIR':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 1.0,-0.1,-0.1, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'LOC':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0],
                        [-0.1,-0.1, 1.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0],
                        [-0.1,-0.1, 1.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0],
                    ])

            if k == 2:
                DMG_corr = DMG_corr.iloc[:20, :20]

                if dep in ['IND', 'DS']:
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1, 1.0,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'PG':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 1.0, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1],
                        [-0.1, 0.5, 1.0, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1],
                        [-0.1, 0.5, 0.5, 1.0,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 1.0, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1],
                        [-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 1.0, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1],
                        [-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 1.0,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 1.0, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1],
                        [-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 1.0, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1],
                        [-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 1.0,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 1.0, 0.5, 0.5,-0.1],
                        [-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 1.0, 0.5,-0.1],
                        [-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 1.0,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'DIR':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 1.0, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.5, 1.0,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1,-0.1, 0.8, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1,-0.1, 0.5, 0.6, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1,-0.1, 0.5, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.5, 0.5,-0.1,-0.1, 1.0, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.6, 0.5,-0.1,-0.1, 0.5, 1.0, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 0.5,-0.1,-0.1, 0.5, 0.5, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'LOC':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.6, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.6, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.6, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.6, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep in ['ATC', 'CSG']:
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.5, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 1.0, 0.5,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.5, 0.5, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])

            if k == 3:
                DMG_corr = DMG_corr.iloc[:20, :20]

                if dep in ['IND', 'DS']:
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.0, 0.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.0, 1.0, 0.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.0, 0.0, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.0, 0.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.0, 1.0, 0.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.0, 0.0, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.0, 0.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.0, 1.0, 0.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.0, 0.0, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.0, 0.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.0, 1.0, 0.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.0, 0.0, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'PG':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 1.0, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1],
                        [-0.1, 0.8, 1.0, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1],
                        [-0.1, 0.7, 0.6, 1.0,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 1.0, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1],
                        [-0.1, 0.8, 0.6, 0.6,-0.1,-0.1, 0.8, 1.0, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1],
                        [-0.1, 0.7, 0.6, 0.5,-0.1,-0.1, 0.7, 0.6, 1.0,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 1.0, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1],
                        [-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 1.0, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1],
                        [-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 1.0,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 1.0, 0.8, 0.7,-0.1],
                        [-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1,-0.1, 0.8, 0.6, 0.6,-0.1,-0.1, 0.8, 1.0, 0.6,-0.1],
                        [-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1,-0.1, 0.7, 0.6, 0.5,-0.1,-0.1, 0.7, 0.6, 1.0,-0.1],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'DIR':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 1.0, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.7, 0.6, 1.0,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 0.6, 0.6,-0.1,-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.7, 0.6, 0.5,-0.1,-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1,-0.1, 0.8, 0.8, 0.7,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1,-0.1, 0.8, 0.7, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1,-0.1, 0.7, 0.6, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.8, 0.7,-0.1,-0.1, 1.0, 0.8, 0.7,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.6, 0.6,-0.1,-0.1, 0.8, 1.0, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 0.5,-0.1,-0.1, 0.7, 0.6, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep == 'LOC':
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.7, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.8, 0.7,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.7, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 0.7, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.7, 0.6, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 0.7, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])
                elif dep in ['ATC', 'CSG']:
                    DMG_corr_ref = np.array([
                        [ 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,-0.1,-0.1,-0.1,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 1.0, 0.8, 0.7,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.8, 1.0, 0.6,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1, 0.7, 0.6, 1.0,-0.1],
                        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.1,-0.1,-0.1,-0.1, 1.0],
                    ])

            for i in range(len(DMG_corr.index)):
                for j in range(len(DMG_corr.columns)):
                    ref_i = DMG_corr_ref[i, j]
                    if ref_i != 0.0:
                        if ref_i > 0.0:
                            assert DMG_corr.iloc[i, j] > 0.97 * ref_i
                        else:
                            assert DMG_corr.iloc[i, j] < 0.0
                    else:
                        assert DMG_corr.iloc[i, j] == pytest.approx(ref_i,
                                                                    abs=0.15)

        # then check the distribution of damage within each performance group
        EDP_list = np.array(
            [[[0.080000, 0.080000], [0.080000, 0.080000], [0.040000, 0.040000]],
             [[7.845320, 7.845320], [7.845320, 7.845320],
              [2.942000, 2.942000]]])

        fr_keys = []
        for key in A._RV_dict.keys():
            if 'FR' in key:
                fr_keys.append(key)

        for k, key in enumerate(sorted(fr_keys)):
            # print(key)

            RV_FR = A._RV_dict[key]

            # only third of the data is unique because of the 3 stories
            rel_len = int(len(RV_FR._dimension_tags) / 3)

            COV_test = RV_FR.COV[:rel_len, :rel_len]
            theta_test = RV_FR.theta[:rel_len]

            lims = np.unique(theta_test)
            ndims = len(lims)
            if k in [2, 3]:
                ndims += 2

            if (dep in ['DS', 'IND']) or k > 1:
                DMG_vals = [[[0., 5., 7.5, 12.5, 17.5, 20., 25.], [0., 25.]],
                            [[0., 1.5, 3., 4.5, 6., 7.5, 9., 10.5, 12., 13.5,
                              15.,
                              16.5, 18., 19.5, 21., 22.5, 24., 25.5, 27., 28.5,
                              30.0],
                             [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                              11., 12., 13., 14., 15., 16., 17., 18., 19.,
                              20.]]]
            else:
                DMG_vals = [[[0., 25.], [0., 25.]],
                            [[0., 30.], [0., 20.]]]
            DMG_vals = np.array(DMG_vals)

            for story in [0, 1, 2]:
                for dir_ in [0, 1]:
                    # print(story, dir_)

                    idx = pd.IndexSlice
                    DMG_check_FG = DMG_check.loc[:, idx[k + 1, :, :]]
                    DMG_check_PG = DMG_check_FG.iloc[:,
                                   story * 2 * ndims + dir_ * ndims:story * 2 * ndims + (
                                           dir_ + 1) * ndims]

                    DMG_val_test = np.unique(
                        np.around(DMG_check_PG.values * 10., decimals=0) / 10.,
                        return_counts=True)
                    DMG_val_test = DMG_val_test[0][DMG_val_test[1] > 10]

                    # only check at most the first 10 elements, because the
                    # higher values have extremely low likelihood
                    ddim = min(len(DMG_val_test), 10)
                    DMG_val_ref = DMG_vals[np.sign(k), dir_]
                    for v in DMG_val_test:
                        assert v in DMG_val_ref

                    # additional tests for mutually exclusive DS2 in FG3
                    if (k == 2) and (dep not in ['DS', 'IND']):
                        DMG_tot = [[0., 30.], [0., 20.]][dir_]
                        DMG_DS2_test = DMG_check_PG.iloc[:, [1, 2, 3]].sum(
                            axis=1)

                        # the proportion of each DS in DS2 shall follow the
                        # pre-assigned weights
                        ME_test = \
                        DMG_check_PG.iloc[DMG_DS2_test.values > 0].iloc[:,
                        [1, 2, 3]].describe().T['mean'].values / DMG_tot[-1]
                        assert_allclose(ME_test, [0.5, 0.3, 0.2], atol=0.01)

                        # the sum of DMG with correlated CSGs shall be either 0.
                        # or the total quantity
                        DMG_DS2_test = np.unique(
                            np.around(DMG_DS2_test * 10., decimals=0) / 10.,
                            return_counts=True)
                        DMG_DS2_test = DMG_DS2_test[0][DMG_DS2_test[1] > 10]
                        assert_allclose(DMG_DS2_test, DMG_tot, atol=0.01)

                        # additional tests for simultaneous DS2 in FG4
                    if (k == 3) and (dep not in ['DS', 'IND']):
                        DMG_tot = [30.0, 20.0][dir_]
                        DMG_DS2_test = DMG_check_PG.iloc[:, [1, 2, 3]].sum(
                            axis=1)

                        # the proportion of each DS in DS2 shall follow the
                        # pre-assigned weights considering replacement
                        SIM_test = \
                        DMG_check_PG.iloc[DMG_DS2_test.values > 0].iloc[:,
                        [1, 2, 3]].describe().T['mean'].values / DMG_tot
                        P_rep = 0.5 * 0.7 * 0.8
                        SIM_ref = np.array([0.5, 0.3, 0.2]) * (
                                1.0 + P_rep / (1.0 - P_rep))
                        assert_allclose(SIM_test, SIM_ref, atol=0.02)

                        # the sum of DMG with correlated CSGs shall be either
                        # 0. or more than the total quantity
                        DMG_DS2_test = DMG_DS2_test.iloc[
                            DMG_DS2_test.values > 0]
                        # Even with perfect correlation, the generated random
                        # samples will not be identical. Hence, one of the 20
                        # CSGs in FG4, very rarely will belong to a different
                        # DS than the rest. To avoid false negatives, we test
                        # the third smallest value.
                        assert DMG_DS2_test.sort_values().iloc[
                                   2] >= DMG_tot * 0.99
                        assert np.max(DMG_DS2_test.values) > DMG_tot

                    # the first component has 3-1 CSGs in dir 1 and 2,
                    # respectively
                    if k == 0:
                        dir_len = int(rel_len * 3 / 4)
                    # the other components have 20-20 CSGs in dir 1 and 2,
                    # respectively
                    else:
                        dir_len = int(rel_len / 2)

                    if dir_ == 0:
                        theta_t = theta_test[:dir_len]
                        COV_t = COV_test[:dir_len, :dir_len]

                    else:
                        theta_t = theta_test[dir_len:]
                        COV_t = COV_test[dir_len:, dir_len:]

                    lim_ds1 = np.where(theta_t == lims[0])[0]
                    lim_ds2 = np.where(theta_t == lims[1])[0]
                    if k > 0:
                        lim_ds3 = np.where(theta_t == lims[2])[0]

                    ndim = len(theta_t)

                    EDP = EDP_list[int(k > 1), story, dir_]

                    DS_ref_all = []
                    DS_ref_any = []
                    DS_test_all = []
                    DS_test_any = []
                    # DS0
                    DS_ref_all.append(mvn_od(np.log(theta_t), COV_t,
                                             lower=np.log(np.ones(ndim) * EDP),
                                             upper=np.ones(ndim) * np.inf)[0])

                    if k == 0:
                        DS_test_all.append(
                            np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                           DMG_check_PG.iloc[:, 1] == 0.],
                                          axis=0)) / 10000.)
                    elif k == 1:
                        DS_test_all.append(
                            np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                           DMG_check_PG.iloc[:, 1] == 0.,
                                           DMG_check_PG.iloc[:, 2] == 0.],
                                          axis=0)) / 10000.)
                    else:
                        DS_test_all.append(
                            np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                           DMG_check_PG.iloc[:, 1] == 0.,
                                           DMG_check_PG.iloc[:, 2] == 0.,
                                           DMG_check_PG.iloc[:, 3] == 0.,
                                           DMG_check_PG.iloc[:, 4] == 0.],
                                          axis=0)) / 10000.)

                    # DS1
                    lower_lim = -np.ones(ndim) * np.inf
                    upper_lim = np.ones(ndim) * np.inf
                    lower_lim[lim_ds2] = np.log(EDP)
                    upper_lim[lim_ds1] = np.log(EDP)
                    if k > 0:
                        lower_lim[lim_ds3] = np.log(EDP)
                    DS_ref_all.append(mvn_od(np.log(theta_t), COV_t,
                                             lower=lower_lim, upper=upper_lim)[
                                          0])

                    lower_lim = -np.ones(ndim) * np.inf
                    upper_lim = np.ones(ndim) * np.inf
                    lower_lim[lim_ds2[0]] = np.log(EDP)
                    upper_lim[lim_ds1[0]] = np.log(EDP)
                    if k > 0:
                        lower_lim[lim_ds3[0]] = np.log(EDP)
                    P_any = mvn_od(np.log(theta_t), COV_t, lower=lower_lim,
                                   upper=upper_lim)[0]
                    if (dep in ['DS', 'IND']):
                        P_any = 1.0 - (1.0 - P_any) ** len(lim_ds1)
                    DS_ref_any.append(P_any)

                    if k == 0:
                        DS_test_all.append(np.sum(np.all(
                            [DMG_check_PG.iloc[:, 0] > DMG_val_ref[-1] - 0.1,
                             DMG_check_PG.iloc[:, 1] == 0.], axis=0)) / 10000.)
                    elif k == 1:
                        DS_test_all.append(np.sum(np.all(
                            [DMG_check_PG.iloc[:, 0] > DMG_val_ref[-1] - 0.1,
                             DMG_check_PG.iloc[:, 1] == 0.,
                             DMG_check_PG.iloc[:, 2] == 0.], axis=0)) / 10000.)
                    else:
                        DS_test_all.append(np.sum(np.all(
                            [DMG_check_PG.iloc[:, 0] > DMG_val_ref[-1] - 0.1,
                             DMG_check_PG.iloc[:, 1] == 0.,
                             DMG_check_PG.iloc[:, 2] == 0.,
                             DMG_check_PG.iloc[:, 3] == 0.,
                             DMG_check_PG.iloc[:, 4] == 0.], axis=0)) / 10000.)

                    DS_test_any.append(np.sum(
                        np.all([DMG_check_PG.iloc[:, 0] > 0.],
                               axis=0)) / 10000.)

                    # DS2
                    lower_lim = -np.ones(ndim) * np.inf
                    upper_lim = np.ones(ndim) * np.inf
                    upper_lim[lim_ds2] = np.log(EDP)
                    if k > 0:
                        lower_lim[lim_ds3] = np.log(EDP)
                    if k < 3:
                        DS_ref_all.append(mvn_od(np.log(theta_t), COV_t,
                                                 lower=lower_lim,
                                                 upper=upper_lim)[0])
                    else:
                        DS_ref_all.append(0.0)

                    lower_lim = -np.ones(ndim) * np.inf
                    upper_lim = np.ones(ndim) * np.inf
                    upper_lim[lim_ds2[0]] = np.log(EDP)
                    if k > 0:
                        lower_lim[lim_ds3[0]] = np.log(EDP)
                    P_any = mvn_od(np.log(theta_t), COV_t, lower=lower_lim,
                                   upper=upper_lim)[0]
                    if (dep in ['DS', 'IND']):
                        P_any = 1.0 - (1.0 - P_any) ** len(lim_ds1)
                    DS_ref_any.append(P_any)

                    if k == 0:
                        DS_test_all.append(
                            np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                           DMG_check_PG.iloc[:, 1] >
                                           DMG_val_ref[-1] - 0.1],
                                          axis=0)) / 10000.)
                    elif k == 1:
                        DS_test_all.append(
                            np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                           DMG_check_PG.iloc[:, 1] >
                                           DMG_val_ref[-1] - 0.1,
                                           DMG_check_PG.iloc[:, 2] == 0.],
                                          axis=0)) / 10000.)
                    elif k == 2:
                        DS_test_all.append(
                            np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                           DMG_check_PG.iloc[:, [1, 2, 3]].sum(
                                               axis=1) > DMG_val_ref[-1] - 0.1,
                                           DMG_check_PG.iloc[:, 4] == 0.],
                                          axis=0)) / 10000.)
                    elif k == 3:
                        # skip this case
                        DS_test_all.append(0.0)

                    if k < 2:
                        DS_test_any.append(np.sum(
                            np.all([DMG_check_PG.iloc[:, 1] > 0.],
                                   axis=0)) / 10000.)
                    else:
                        DS_test_any.append(np.sum(np.all(
                            [DMG_check_PG.iloc[:, [1, 2, 3]].sum(axis=1) > 0.],
                            axis=0)) / 10000.)

                    # DS3
                    if k > 0:

                        lower_lim = -np.ones(ndim) * np.inf
                        upper_lim = np.ones(ndim) * np.inf
                        upper_lim[lim_ds3] = np.log(EDP)
                        DS_ref_all.append(mvn_od(np.log(theta_t), COV_t,
                                                 lower=lower_lim,
                                                 upper=upper_lim)[0])

                        lower_lim = -np.ones(ndim) * np.inf
                        upper_lim = np.ones(ndim) * np.inf
                        upper_lim[lim_ds3[0]] = np.log(EDP)
                        P_any = mvn_od(np.log(theta_t), COV_t, lower=lower_lim,
                                       upper=upper_lim)[0]
                        if (dep in ['DS', 'IND']):
                            P_any = 1.0 - (1.0 - P_any) ** len(lim_ds1)
                        DS_ref_any.append(P_any)

                        if k == 1:
                            DS_test_all.append(
                                np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                               DMG_check_PG.iloc[:, 1] == 0.,
                                               DMG_check_PG.iloc[:, 2] >
                                               DMG_val_ref[-1] - 0.1],
                                              axis=0)) / 10000.)
                        else:
                            DS_test_all.append(
                                np.sum(np.all([DMG_check_PG.iloc[:, 0] == 0.,
                                               DMG_check_PG.iloc[:, 1] == 0.,
                                               DMG_check_PG.iloc[:, 2] == 0.,
                                               DMG_check_PG.iloc[:, 3] == 0.,
                                               DMG_check_PG.iloc[:, 4] >
                                               DMG_val_ref[-1] - 0.1],
                                              axis=0)) / 10000.)
                        if k == 1:
                            DS_test_any.append(np.sum(
                                np.all([DMG_check_PG.iloc[:, 2] > 0.],
                                       axis=0)) / 10000.)

                        else:
                            DS_test_any.append(np.sum(
                                np.all([DMG_check_PG.iloc[:, 4] > 0.],
                                       axis=0)) / 10000.)

                    assert_allclose(DS_ref_all, DS_test_all, atol=0.02)
                    assert_allclose(DS_ref_any, DS_test_any, atol=0.02)

        # ---------------------------------------------------------------------

        A.calculate_losses()

        # ---------------------------------------------- check loss calculation

        # No additional uncertainty is introduced when it comes to losses in
        # this test. The decision variables and the damaged quantities shall
        # follow the same distribution and have the same correlation structure.
        # The damaged quantities have already been verified, so now we use them
        # as reference values for testing the decision variables.

        # COST and TIME and INJ
        DV_COST = A._DV_dict['rec_cost']
        DV_TIME = A._DV_dict['rec_time']
        DV_INJ_dict = deepcopy(A._DV_dict['injuries'])
        DV_INJ0 = DV_INJ_dict[0]
        DV_INJ1 = DV_INJ_dict[1]

        DMG_check = A._DMG

        for k in range(4):
            # Start with checking the correlations...
            dmg = DMG_check.loc[:, (DMG_check != 0.0).any(axis=0)]
            dmg_corr = dmg.loc[:, idx[k + 1, :, :]].corr()
            for dv in [DV_COST, DV_TIME, DV_INJ0, DV_INJ1]:
                dv = dv.loc[:, (dv != 0.0).any(axis=0)]
                dv_corr = dv.loc[:, idx[k + 1, :, :]].corr()

                assert_allclose(dmg_corr.values, dv_corr.values, atol=0.001)

            # then check the distribution.
            # After normalizing with the damaged quantities all decision
            # variables in a given DS shall have the same value.
            dv = ((dv / dmg).describe().T).fillna(0.0)

            assert_allclose(dv['std'], np.zeros(len(dv.index)), atol=1.0)

        # red tags require special checks
        for f, fg_id in enumerate(sorted(A._FG_dict.keys())):
            dims = [2, 3, 5, 5][f]

            # take the total quantity of each performance group
            FG = A._FG_dict[fg_id]
            qnt = np.array([PG._quantity.samples.values[:dims] for PG in
                            FG._performance_groups]).flatten()

            # flag the samples where the damage exceeds the pre-defined limit
            # for red tagging
            dmg = DMG_check.loc[:, idx[FG._ID, :, :]]
            red_ref = dmg > 0.489 * qnt

            # collect the red tag results from the analysis
            red_test = A._DV_dict['red_tag'].loc[:, idx[FG._ID, :, :]]

            # compare
            red_diff = (red_ref - red_test).describe().T
            assert_allclose(red_diff['mean'].values, 0.)
            assert_allclose(red_diff['std'].values, 0.)

        # ---------------------------------------------------------------------

        A.aggregate_results()

        # -------------------------------------------- check result aggregation

        # Aggregate results are checked in detail by other tests.
        # Here we only focus on some simple checks to make sure the results
        # make sense.

        S = A._SUMMARY
        SD = S.describe().T

        assert SD.loc[('inhabitants', ''), 'mean'] == 30.0
        assert SD.loc[('inhabitants', ''), 'std'] == 0.0

        assert SD.loc[('collapses', 'collapsed?'), 'mean'] == 0.0
        assert SD.loc[('collapses', 'collapsed?'), 'std'] == 0.0

        assert_allclose(A._DV_dict['rec_cost'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'cost')])
        assert_allclose(A._DV_dict['rec_time'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'time-sequential')])
        assert_allclose(A._DV_dict['rec_time'].max(axis=1),
                        S.loc[:, ('reconstruction', 'time-parallel')])
        assert_allclose(A._DV_dict['injuries'][0].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 1')])
        assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 2')])

def test_FEMA_P58_Assessment_DV_uncertainty_dependencies():
    """
    Perform loss assessment with customized inputs that focus on testing the
    propagation of uncertainty in consequence functions and decision variables.
    Dispersions in other calculation parameters are reduced to negligible
    levels. This allows us to test the results against pre-defined reference
    values in spite of the randomness involved in the calculations.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + 'input data/' + "DL_input_test_10.json"
    EDP_input = base_input_path + 'EDP data/' + "EDP_table_test_10.out"

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

        #print([dep_COST, dep_TIME, dep_RED, dep_INJ, dep_CT, dep_ILVL], end=' ')

        A = FEMA_P58_Assessment()

        with pytest.warns(UserWarning) as e_info:
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

            assert len(RV_DV._dimension_tags) == [32, 16, 32][r]

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

                assert_allclose(RV_DV.theta, np.ones(16))
                assert_allclose(sig_test, np.array([0.33, 0.73] * 8))

                assert_allclose(rho_test, rho_ref[dep_RED])

            elif RV_tag == 'inj':

                assert_allclose(RV_DV.theta, np.ones(32))
                assert_allclose(sig_test, np.array(
                    [0.34, 0.74] * 8 + [0.35, 0.75] * 8))

                if dep_ILVL == True:
                    assert_allclose(rho_test[:16, :16], rho_ref[dep_INJ])
                    assert_allclose(rho_test[16:, 16:], rho_ref[dep_INJ])
                    assert_allclose(rho_test[:16, 16:], rho_ref[dep_INJ])
                    assert_allclose(rho_test[16:, :16], rho_ref[dep_INJ])
                else:
                    assert_allclose(rho_test[:16, :16], rho_ref[dep_INJ])
                    assert_allclose(rho_test[16:, 16:], rho_ref[dep_INJ])
                    assert_allclose(rho_test[:16, 16:], np.zeros((16, 16)))
                    assert_allclose(rho_test[16:, :16], np.zeros((16, 16)))

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
                mu_ds1, mu_ds2 = {'inj0': [0.5, 0.6], 'inj1': [0.1, 0.2]}[
                    DV_tag]
                beta_ds1, beta_ds2 = \
                    {'inj0': [0.34, 0.74], 'inj1': [0.35, 0.75]}[DV_tag]

                # DS1
                # The affected population in DS1 per unit quantity (identical
                # for all FGs and injury levels)
                p_aff = 0.05

                mu_ref, var_ref = tnorm.stats(-1. / beta_ds1, (
                    1. - mu_ds1) / mu_ds1 / beta_ds1, loc=mu_ds1,
                                              scale=mu_ds1 * beta_ds1,
                                              moments='mv')
                sig_ref = np.sqrt(var_ref)
                assert_allclose(DV_desc['mean'].values[::2], mu_ref * p_aff,
                                rtol=beta_ds1 / 10.)
                assert_allclose(DV_desc['std'].values[::2], sig_ref * p_aff,
                                rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[::2]) == pytest.approx(
                    sig_ref * p_aff, rel=0.1)

                # DS2
                # the affected population in DS1 per unit quantity (identical
                # for all FGs and injury levels)
                p_aff = 0.1
                mu_ref, var_ref = tnorm.stats(-1. / beta_ds2, (
                    1. - mu_ds2) / mu_ds2 / beta_ds2, loc=mu_ds2,
                                              scale=mu_ds2 * beta_ds2,
                                              moments='mv')
                sig_ref = np.sqrt(var_ref)
                assert_allclose(DV_desc['mean'].values[1::2],
                                mu_ref * p_aff, rtol=beta_ds2 / 10.)
                assert_allclose(DV_desc['std'].values[1::2],
                                sig_ref * p_aff, rtol=0.20)
                assert np.mean(
                    DV_desc['std'].values[1::2]) == pytest.approx(
                    sig_ref * p_aff, rel=0.1)

        # red tags have to be treated separately
        DV_RED = A._DV_dict['red_tag']

        DMG_norm = DMG_check / 25.

        for i in range(16):

            is_dam = pd.DataFrame(np.zeros((len(DMG_norm.index), 5)),
                                  columns=range(5))
            is_dam[0] = (DMG_norm.iloc[:, i] < 0.01)
            is_dam[1] = (DMG_norm.iloc[:, i] > 0.01) & (
                DMG_norm.iloc[:, i] < 0.275)
            is_dam[2] = (DMG_norm.iloc[:, i] > 0.275) & (
                DMG_norm.iloc[:, i] < 0.525)
            is_dam[3] = (DMG_norm.iloc[:, i] > 0.525) & (
                DMG_norm.iloc[:, i] < 0.775)
            is_dam[4] = (DMG_norm.iloc[:, i] > 0.775)

            mu_red = ([0.87, 0.23185] * 4 + [0.50, 0.23185] * 4)[i]
            beta_red = ([0.33, 0.73] * 8)[i]
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

            elif dv_tag == 'red':
                DV_RED_n = pd.DataFrame(np.ones(DV.shape) * np.nan,
                                        index=DV.index, columns=DV.columns)
                DMG_filter = pd.concat(
                    [(DMG_check.iloc[:, [0, 2, 4, 6]] / 25.0 > 0.525) & (
                            DMG_check.iloc[:, [0, 2, 4, 6]] / 25.0 < 0.775),
                     (DMG_check.iloc[:, [1, 3, 5, 7]] / 25.0 > 0.025) & (
                             DMG_check.iloc[:, [1, 3, 5, 7]] / 25.0 < 0.275),
                     (DMG_check.iloc[:, [8, 10, 12, 14]] / 25.0 > 0.275) & (
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

        assert SD.loc[('collapses', 'collapsed?'), 'mean'] == 0.0
        assert SD.loc[('collapses', 'collapsed?'), 'std'] == 0.0

        assert_allclose(A._DV_dict['rec_cost'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'cost')])
        assert_allclose(A._DV_dict['rec_time'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'time-sequential')])
        assert_allclose(A._DV_dict['rec_time'].max(axis=1),
                        S.loc[:, ('reconstruction', 'time-parallel')])
        assert_allclose(A._DV_dict['injuries'][0].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 1')])
        assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 2')])

        #print()


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

        with pytest.warns(UserWarning) as e_info:
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

        assert SD.loc[('collapses', 'collapsed?'), 'mean'] == 0.0
        assert SD.loc[('collapses', 'collapsed?'), 'std'] == 0.0

        assert_allclose(A._DV_dict['rec_cost'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'cost')])
        assert_allclose(A._DV_dict['rec_time'].sum(axis=1),
                        S.loc[:, ('reconstruction', 'time-sequential')])
        assert_allclose(A._DV_dict['rec_time'].max(axis=1),
                        S.loc[:, ('reconstruction', 'time-parallel')])
        assert_allclose(A._DV_dict['injuries'][0].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 1')])
        assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                        S.loc[:, ('injuries', 'sev. 2')])

        # print()