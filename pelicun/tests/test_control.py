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
from copy import deepcopy

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir))

from pelicun.control import *
from pelicun.uq import mvn_orthotope_density as mvn_od

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
    DL_input = base_input_path + "DL_input_test.json"
    EDP_input = base_input_path + "EDP_table_test.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

    A.define_random_variables()

    # -------------------------------------------------- check random variables

    # EDP
    RV_EDP = A._RV_dict['EDP']
    assert RV_EDP.theta == pytest.approx(0.5 * g)
    assert RV_EDP.COV == pytest.approx(np.sqrt(2) * 1e-4)
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
    assert RV_FRG._distribution_kind == 'lognormal'

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
                    rtol=0.1)
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

    CAS = deepcopy(S.loc[:, ('injuries', 'casualties')])
    CAS_CDF = np.around(CAS, decimals=3)
    vals, counts = np.unique(CAS_CDF, return_counts=True)
    assert_allclose(vals, [0, 0.075, 0.15, 0.25, 0.3, 0.5, 1.])
    assert_allclose(counts / 10000.,
                    np.array([35, 1, 3.5, 2, 2.5, 7, 5]) / 56., atol=0.01,
                    rtol=0.1)

    CAS = deepcopy(S.loc[:, ('injuries', 'fatalities')])
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
    DL_input = base_input_path + "DL_input_test_2.json"
    EDP_input = base_input_path + "EDP_table_test_2.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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
    DL_input = base_input_path + "DL_input_test_3.json"
    EDP_input = base_input_path + "EDP_table_test_3.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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
    assert_allclose(COV / np.outer(sig, sig), rho_target, atol=0.15)

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

    assert P_no_RED_target == pytest.approx(P_no_RED_test, abs=0.01)
    
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
    Some information is lost when a large number of analysis results are 
    replaced with 1.0 and 100.0. The ML estimator used in the current version
    can provide a reasonable estimate of the original covariance matrix, but
    the error is not negligible. That error is not due to a mistake or bug, but
    a rather an expected product of the implemented estimator. Those errors 
    forced us to increase the tolerances in this test compared to the previous
    ones to avoid false positives. Future improvements in the tmvn_MLE 
    algorithm will hopefully allow us to eventually return to the original
    tolerance levels.
    """

    base_input_path = 'resources/'

    DL_input = base_input_path + "DL_input_test_4.json"
    EDP_input = base_input_path + "EDP_table_test_4.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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
    assert_allclose(COV / np.outer(sig, sig), rho_target, atol=0.15)

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

        assert_allclose(P_target, P_test, atol=0.05)
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

        assert_allclose(P_target, P_test, atol=0.05)
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

        assert_allclose(P_target, P_test, atol=0.05)
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

        assert_allclose(P_target, P_test, atol=0.05)
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

    assert P_no_RED_target == pytest.approx(P_no_RED_test, abs=0.02)
    
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

    DL_input = base_input_path + "DL_input_test_5.json"
    EDP_input = base_input_path + "EDP_table_test_5.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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
    rho_target = [
        [1.0, 0.8, 0.6, 0.5, 0.3, 0.3, 0.3, 0.3],
        [0.8, 1.0, 0.5, 0.6, 0.3, 0.3, 0.3, 0.3],
        [0.6, 0.5, 1.0, 0.8, 0.3, 0.3, 0.3, 0.3],
        [0.5, 0.6, 0.8, 1.0, 0.3, 0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3, 0.3, 1.0, 0.8, 0.7, 0.6],
        [0.3, 0.3, 0.3, 0.3, 0.8, 1.0, 0.6, 0.7],
        [0.3, 0.3, 0.3, 0.3, 0.7, 0.6, 1.0, 0.8],
        [0.3, 0.3, 0.3, 0.3, 0.6, 0.7, 0.8, 1.0]]
    assert_allclose(COV / np.outer(sig, sig), rho_target, atol=0.1)
    COV_target = rho_target * np.outer(sig_target, sig_target)

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

    assert COL_check['mean'].values[0] == pytest.approx(col_target, rel=0.1)

    # DMG
    DMG_check = [len(np.where(A._DMG.iloc[:, i] > 0.0)[0]) / 10000. for i in
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
    P_test = P_test / 10000.

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
    P_test = P_test / 10000.

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
    P_test = P_test / 10000.

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

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
    P_test = P_test / 10000.

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
    P_test = P_test / 10000.

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
    P_test = P_test / 10000.

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

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
    P_test = P_test / 10000.

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
    P_test = P_test / 10000.

    assert_allclose(P_target, P_test, atol=0.05)
    assert_allclose(C_target, C_test, rtol=0.001)
    assert_allclose(T_target, T_test, rtol=0.001)

    # RED TAG
    RED_check = A._DV_dict['red_tag'].describe().T
    RED_check = (RED_check['mean'] * RED_check['count'] / 10000.).values

    assert_allclose(RED_check, DMG_ref, rtol=0.10)

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
        ('red tagged?', ''), 'count'] / 10000.

    assert P_no_RED_target == pytest.approx(P_no_RED_test, abs=0.01)
    
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

    DL_input = base_input_path + "DL_input_test_6.json"
    EDP_input = base_input_path + "EDP_table_test_6.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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
    assert_allclose(sig, np.ones(8) * np.sqrt(2e-8), rtol=0.1)
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
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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

    DL_input = base_input_path + "DL_input_test_7.json"
    EDP_input = base_input_path + "EDP_table_test_7.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

    with pytest.warns(UserWarning) as e_info:
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
                    [0., 0., 0., 0., 1., 0., 0., 0.])

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

    DL_input = base_input_path + "DL_input_test_8.json"
    EDP_input = base_input_path + "EDP_table_test_8.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test_det.json"

    A = FEMA_P58_Assessment()

    with pytest.warns(UserWarning) as e_info:
        A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

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
                    S.loc[:, ('injuries', 'casualties')])
    assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                    S.loc[:, ('injuries', 'fatalities')])
    
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

    DL_input = base_input_path + "DL_input_test_8.json"
    EDP_input = base_input_path + "EDP_table_test_8.out"
    CMP_data = base_input_path
    POP_data = base_input_path + "population_test_det.json"

    for dep in ['FG', 'PG', 'DIR', 'LOC']:

        A = FEMA_P58_Assessment()

        with pytest.warns(UserWarning) as e_info:
            A.read_inputs(DL_input, EDP_input, CMP_data, POP_data,
                          verbose=False)

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
                        S.loc[:, ('injuries', 'casualties')])
        assert_allclose(A._DV_dict['injuries'][1].sum(axis=1),
                        S.loc[:, ('injuries', 'fatalities')])