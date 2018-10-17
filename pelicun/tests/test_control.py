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

    A.read_inputs(DL_input, EDP_input, CMP_data, POP_data, verbose=False)

    A.define_random_variables()

    # ------------------------------------------------------ check random variables

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
    order = np.argsort(RV_FRG.dimension_tags)
    assert_allclose(RV_FRG.theta[order], np.array([0.37, 0.5, 0.82]) * g, rtol=0.01)
    COV = deepcopy(RV_FRG.COV)
    sig = np.sqrt(np.diagonal(COV))
    assert_allclose(sig[order], np.array([0.3, 0.4, 0.5]), rtol=0.01)
    assert_allclose(COV / np.outer(sig, sig), np.ones((3, 3)), rtol=0.01)
    assert RV_FRG._distribution_kind == 'lognormal'

    # RED
    RV_RED = A._RV_dict['DV_RED']
    order = np.argsort(RV_RED.dimension_tags)
    assert_allclose(RV_RED.theta[order], np.ones(2), rtol=0.01)
    assert_allclose(RV_RED.COV, np.array([[1, 0], [0, 1]]) * (1e-4) ** 2.,
                    rtol=0.01)
    assert RV_RED._distribution_kind == 'normal'
    assert RV_RED.tr_limits_pre == None
    assert_allclose(RV_RED.tr_limits_post[0][order], np.array([0., 0.]),
                    rtol=0.01)
    assert_allclose(RV_RED.tr_limits_post[1][order], np.array([2., 4.]),
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

    # ------------------------------------------------------ check damage calculation
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

    # ------------------------------------------------------ check loss calculation

    # RED
    DV_RED = A._DV_dict['red_tag'].describe().T
    assert_allclose(DV_RED['mean'], np.array([0.341344, 0.1586555]), rtol=0.1)

    # INJ - collapse
    print(A._COL.columns)
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

    # ------------------------------------------------------ check result aggregation

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