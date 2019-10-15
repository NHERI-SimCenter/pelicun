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
This subpackage performs unit tests on the model module of pelicun.

"""
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, truncnorm

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir))

from pelicun.model import *
from pelicun.tests.test_reference_data import standard_normal_table
from pelicun.uq import RandomVariable, RandomVariableSubset

# -------------------------------------------------------------------------------
# Fragility_Function
# ------------------------------------------------------------------------------

def test_FragilityFunction_Pexc_lognormal_unit_mean_unit_std():
    """
    Given a lognormal fragility function with theta=1.0 and beta=1.0, test if
    the calculated exceedance probabilities are sufficiently accurate.
    The reference results are based on a standard normal table. This limits the
    accuracy of testing to an absolute probability difference of 1e-5.
    """
    # prepare the inputs
    EDP = np.exp(np.concatenate([-standard_normal_table[0][::-1],
                                 standard_normal_table[0]]))
    reference_P_exc = np.concatenate([0.5 - standard_normal_table[1][::-1],
                                      0.5 + standard_normal_table[1]])

    # create the fragility function
    RV = RandomVariable(ID=1, dimension_tags='A',
                        distribution_kind='lognormal',
                        theta=1.0, COV=1.0)
    fragility_function = FragilityFunction(
        EDP_limit=RandomVariableSubset(RV, 'A'))

    # calculate the exceedance probabilities
    test_P_exc = fragility_function.P_exc(EDP, DSG_ID=1)

    assert_allclose(test_P_exc, reference_P_exc, atol=1e-5)

def test_FragilityFunction_Pexc_lognormal_non_trivial_case():
    """
    Given a lognormal fragility function with theta=0.5 and beta=0.2, test if
    the calculated exceedance probabilities are sufficiently accurate.
    The reference results are based on a standard normal table. This limits the
    accuracy of testing to an absolute probability difference of 1e-5.
    """
    # prepare the inputs
    target_theta = 0.5
    target_beta = 0.2
    EDP = np.concatenate([-standard_normal_table[0][::-1],
                          standard_normal_table[0]])
    EDP = np.exp(EDP * target_beta + np.log(target_theta))
    reference_P_exc = np.concatenate([0.5 - standard_normal_table[1][::-1],
                                      0.5 + standard_normal_table[1]])

    # create the fragility function
    RV = RandomVariable(ID=1, dimension_tags='A',
                        distribution_kind='lognormal',
                        theta=target_theta, COV=target_beta ** 2.)
    fragility_function = FragilityFunction(
        EDP_limit=RandomVariableSubset(RV, 'A'))

    # calculate the exceedance probabilities
    test_P_exc = fragility_function.P_exc(EDP, DSG_ID=1)

    assert_allclose(test_P_exc, reference_P_exc, atol=1e-5)

def test_FragilityFunction_Pexc_lognormal_zero_input():
    """
    Given a zero EDP input to a lognormal fragility function, the result shall
    be 0 exceedance probability, even though zero input in log space shall
    correspond to -infinity. This slight modification makes our lives much
    easier when real inputs are fed to the fragility functions.
    """
    # create the fragility function
    RV = RandomVariable(ID=1, dimension_tags='A',
                        distribution_kind='lognormal',
                        theta=1.0, COV=1.0)
    fragility_function = FragilityFunction(
        EDP_limit=RandomVariableSubset(RV, 'A'))

    # calculate the exceedance probability
    test_P_exc = fragility_function.P_exc(0., DSG_ID=1)

    assert test_P_exc == 0.

def test_FragilityFunction_Pexc_lognormal_nonzero_scalar_input():
    """
    Given a nonzero scalar EDP input, the fragility function should return a
    nonzero scalar output.
    """
    # create the fragility function
    RV = RandomVariable(ID=1, dimension_tags='A',
                        distribution_kind='lognormal',
                        theta=1.0, COV=1.0)
    fragility_function = FragilityFunction(
        EDP_limit=RandomVariableSubset(RV, 'A'))

    # calculate the exceedance probabilities
    test_P_exc = fragility_function.P_exc(standard_normal_table[0][0], DSG_ID=1)

    assert test_P_exc == pytest.approx(standard_normal_table[1][0], abs=1e-5)

def test_FragilityFunction_Pexc_multiple_damage_states_with_correlation():
    """
    Test if the fragility function returns an appropriate list of exceedance
    probabilities for various scenarios with multiple damage states that have
    potentially correlated fragilities.

    """
    # P_exc is requested for a list of EDPs
    EDP = np.exp(np.linspace(-2., 2., num=11))

    # 3 damage state groups, perfectly correlated
    # the DSGs are unordered in the RV only to make the test more general
    dims = 3
    ref_mean = np.exp([2.0, 0., 0.5])
    ref_std = [1.5, 0.5, 1.0]
    ref_rho = np.ones((dims, dims)) * 1.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    RV = RandomVariable(ID=1, dimension_tags=['C', 'A', 'B'],
                        distribution_kind='lognormal',
                        theta=ref_mean, COV=ref_COV)

    # a single DSG fragility
    # note that A is correlated with the other RV components
    RVS = RandomVariableSubset(RV=RV, tags='A')
    test_res = FragilityFunction(EDP_limit=RVS).P_exc(EDP, 1)
    ref_res = norm.cdf(np.log(EDP),
                       loc=np.log(ref_mean[1]), scale=ref_std[1])
    assert_allclose(test_res, ref_res)

    # three DSGs in proper order, P_exc for A is requested considering all
    # three
    RVS = RandomVariableSubset(RV=RV, tags=['A', 'B', 'C'])
    test_res = FragilityFunction(EDP_limit=RVS).P_exc(EDP, 1)
    ref_res = [norm.cdf(np.log(EDP),
                        loc=np.log(ref_mean[i]), scale=ref_std[i])
               for i in range(3)]
    ref_res = np.max(np.asarray(ref_res), axis=0)
    assert_allclose(test_res, ref_res)

    # change the covariance matrix - uncorrelated fragilities
    ref_rho = np.ones((dims, dims)) * 0.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho
    RV = RandomVariable(ID=1, dimension_tags=['C', 'A', 'B'],
                        distribution_kind='lognormal',
                        theta=ref_mean, COV=ref_COV)

    # three DSGs, still interested in P_exc for A considering all three
    RVS = RandomVariableSubset(RV=RV, tags=['A', 'B', 'C'])
    test_res = FragilityFunction(EDP_limit=RVS).P_exc(EDP, 1)
    ref_res = [norm.cdf(np.log(EDP),
                        loc=np.log(ref_mean[i]), scale=ref_std[i])
               for i in range(3)]
    ref_res[1] = ref_res[1] * (1. - ref_res[0])
    ref_res[2] = ref_res[2] * (1. - np.sum(np.asarray(ref_res[:2]), axis=0))
    ref_res = np.sum(np.asarray(ref_res), axis=0)
    assert_allclose(ref_res, test_res)

def test_FragilityFunction_DSG_ID_given_EDP_general():
    """
    Test if the DSG_IDs returned by the function are appropriate using
    exceedance probabilities from the already tested P_exc function.

    """
    # 3 damage state groups, perfectly correlated
    # the DSGs are unordered in the RV only to make the test more general
    dims = 3
    ref_mean = np.exp([2.0, 0., 0.5])
    ref_std = [1.5, 0.5, 1.0]
    ref_rho = np.ones((dims, dims)) * 1.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    RV = RandomVariable(ID=1, dimension_tags=['C', 'A', 'B'],
                        distribution_kind='lognormal',
                        theta=ref_mean, COV=ref_COV)

    RVS = RandomVariableSubset(RV=RV, tags=['A', 'B', 'C'])
    FF = FragilityFunction(EDP_limit=RVS)

    # same EDP 10^5 times to allow for P_exc-based testing
    for target_EDP in [0., 0.75, 2.0]:
        # create the EDP vector
        EDP = np.ones(10000) * np.exp(target_EDP)

        # get the DSG_IDs
        DSG_ID = FF.DSG_given_EDP(EDP, force_resampling=False)

        # calculate the DSG_ID probabilities
        P_DS_test = np.histogram(DSG_ID.values,
                                 bins=np.arange(5) - 0.5, density=True)[0]

        # use the P_exc function to arrive at the reference DSG_ID probabilities
        P_exc = np.asarray(list(map(lambda x: FF.P_exc(np.exp(target_EDP), x),
                                    [0, 1, 2, 3])))
        P_DS_ref = np.concatenate([P_exc[:-1] - P_exc[1:], [P_exc[-1], ]])

        # compare
        assert_allclose(P_DS_test, P_DS_ref, atol=0.02)

    # random set of EDPs uniformly distributed over the several different
    # domains
    for a, b in [[-1., -0.9], [1., 1.1], [-1., 1.]]:
        EDP = np.exp(np.random.uniform(a, b, 100000))

        # get a DSG_ID sample for each EDP
        DSG_ID = FF.DSG_given_EDP(EDP, force_resampling=True)

        # get the test DSG_ID probabilities
        P_DS_test = \
        np.histogram(DSG_ID.values, bins=np.arange(5) - 0.5, density=True)[0]

        # get the EDP-P_exc functions - basically the fragility functions
        EDP = np.exp(np.linspace(a, b, num=100))
        P_exc_f = np.asarray(
            list(map(lambda x: FF.P_exc(EDP, x), [0, 1, 2, 3])))

        # Calculate the area enclosed by the two functions that define each DS
        # it should be the same as the P_DS from the test
        CDF = [p - np.max(P_exc_f[i + 1:], axis=0)
               for i, p in enumerate(P_exc_f[:3])]
        CDF.append(P_exc_f[-1])
        CDF = np.asarray(CDF)
        P_DS_ref = np.asarray(list(map(lambda x: np.trapz(CDF[x], np.log(EDP)),
                                       [0, 1, 2, 3]))) / (b - a)

        assert_allclose(P_DS_test, P_DS_ref, atol=0.02)

def test_FragilityFunction_DSG_given_EDP_insufficient_samples():
    """
    Test if the function raises an error message if the number of EDP values
    provided is greater than the number of available samples from the RVS.

    """
    # create a simple random variable
    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='lognormal',
                        theta=1.0, COV=1.0)

    # assign it to the fragility function
    RVS = RandomVariableSubset(RV=RV, tags=['A'])
    FF = FragilityFunction(EDP_limit=RVS)

    # sample 10 realizations
    RVS.sample_distribution(10)

    # create 100 EDP values
    EDP = np.ones(100)

    # try to get the DSG_IDs... and expect an error
    with pytest.raises(ValueError) as e_info:
        FF.DSG_given_EDP(EDP)

# ------------------------------------------------------------------------------
# Consequence_Function
# ------------------------------------------------------------------------------

def test_ConsequenceFunction_fixed_median_value():
    """
    Test if the function returns the prescribed median.
    """
    for dist in ['normal', 'lognormal']:
        RV = RandomVariable(ID=1, dimension_tags=['A'],
                            distribution_kind=dist,
                            theta=1.0, COV=1.0)

        conseq_function = ConsequenceFunction(
            DV_median=prep_constant_median_DV(1.0),
            DV_distribution=RandomVariableSubset(RV, 'A')
        )

        assert conseq_function.median() == 1.0
        assert conseq_function.median(1.0) == 1.0
        assert conseq_function.median([1., 1.]) == 1.0


def test_ConsequenceFunction_bounded_linear_median_value():
    """
    Test if the function returns an appropriate output for single quantities
    and for quantity arrays, and if it raises an error if the quantity is not
    specified for a quantity-dependent median.
    """
    test_quants = [0.5, 1.0, 1.5, 2.0, 2.5]
    ref_vals = [2.0, 2.0, 1.5, 1.0, 1.0]

    for dist in ['normal', 'lognormal']:
        RV = RandomVariable(ID=1, dimension_tags=['A'],
                            distribution_kind=dist,
                            theta=1.0, COV=1.0)
        f_median = prep_bounded_linear_median_DV(
            median_max=2.0, median_min=1.0,
            quantity_lower=1.0, quantity_upper=2.0
        )
        conseq_function = ConsequenceFunction(
            DV_median=f_median,
            DV_distribution=RandomVariableSubset(RV, 'A')
        )

        # should raise an error if the quantity is not specified
        with pytest.raises(ValueError) as e_info:
            conseq_function.median()

        # single quantities
        for quant, ref_val in zip(test_quants, ref_vals):
            assert conseq_function.median(quantity=quant) == ref_val

        # quantity array
        test_medians = conseq_function.median(quantity=test_quants)
        assert_allclose(test_medians, ref_vals, rtol=1e-10)


def test_ConsequenceFunction_sample_unit_DV():
    """
    Test if the function samples the DV distribution properly. Note that we
    have already tested the sampling algorithm in the uq module, so we will not
    do a thorough verification of the samples here, but rather check for errors
    in the inputs that would typically lead to significant mistakes in the
    results.
    """
    test_quants = [0.5, 1.0, 1.5, 2.0, 2.5]

    # create a Random Variable with 3 correlated decision variables
    dims = 3
    ref_mean = [1., 1., 0.]
    ref_std = [0.4, 0.3, 0.2]
    ref_rho = np.ones((dims, dims)) * 0.8
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    ref_mean[2] = np.exp(ref_mean[2])

    # prepare lower truncation limits at 0 for all...
    tr_lower = np.zeros(dims).tolist()
    # and an upper limit at 2 sigma for the second
    tr_upper = [np.inf, 1.6, np.inf]

    # make sure the correlations are applied post-truncation
    corr_ref = 'post'

    RV = RandomVariable(ID=1, dimension_tags=['A', 'B', 'C'],
                        distribution_kind=['normal', 'normal', 'lognormal'],
                        corr_ref=corr_ref,
                        theta=ref_mean, COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])

    # first test sampling for each decision variable
    for r_i, tag in enumerate(['A', 'B', 'C']):

        # use fixed value for 'B' and bounded linear for the other two
        if tag == 'B':
            f_median = prep_constant_median_DV(10.)
        else:
            f_median = prep_bounded_linear_median_DV(
                median_max=20.0, median_min=2.0,
                quantity_lower=1.0, quantity_upper=2.0
            )

        # create the consequence function
        conseq_function = ConsequenceFunction(
            DV_median=f_median,
            DV_distribution=RandomVariableSubset(RV, tag)
        )

        for qnt in test_quants:
            samples = conseq_function.sample_unit_DV(quantity=qnt,
                                                     sample_size=1000,
                                                     force_resampling=True)

            # transform the results to log space for 'C' to facilitate testing
            if tag == 'C':
                samples = np.log(samples)
                ref_mu = np.log(f_median(qnt))
                ref_min = np.log(max(np.nextafter(0, 1), tr_lower[r_i]))
                ref_max = np.log(max(np.nextafter(0, 1), tr_upper[r_i]))
                a = (ref_min - np.log(ref_mean[r_i])) / ref_std[r_i]
                b = (ref_max - np.log(ref_mean[r_i])) / ref_std[r_i]
                ref_max = ref_mu * b
            else:
                ref_mu = f_median(qnt)
                ref_min = tr_lower[r_i]
                a = (ref_min - ref_mean[r_i]) / ref_std[r_i]
                b = (tr_upper[r_i] - ref_mean[r_i]) / ref_std[r_i]
                ref_max = ref_mu * b

            trNorm = truncnorm(a=a, b=b, loc=ref_mu,
                               scale=ref_std[r_i] if tag == 'C'
                               else ref_std[r_i] * ref_mu)
            ref_samples = trNorm.rvs(size=1000)

            # test the means and coefficients of variation
            assert np.mean(samples) == pytest.approx(np.mean(ref_samples),
                                                     rel=0.1)
            assert np.std(samples) == pytest.approx(np.std(ref_samples),
                                                    rel=0.15)

            # test the limits
            assert np.min(samples) > ref_min
            assert np.max(samples) < ref_max

            # verify that the correlation in the random variable follows the
            # prescribed correlation matrix
            CORR_sample = RV.samples
            CORR_sample['C'] = np.log(CORR_sample['C'])
            assert_allclose(np.corrcoef(CORR_sample, rowvar=False),
                            ref_rho, rtol=0.1)

def test_ConsequenceFunction_sample_unit_DV_insufficient_samples():
    """
    Test if the function raises an error message if the number of samples
    requested is greater than the number of available samples from the RVS.

    """
    # create a simple random variable
    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='lognormal',
                        theta=1.0, COV=1.0)

    # assign it to the fragility function
    RVS = RandomVariableSubset(RV=RV, tags=['A'])
    CF = ConsequenceFunction(DV_median=prep_constant_median_DV(1.0),
                             DV_distribution = RVS)

    # sample 10 realizations
    RVS.sample_distribution(10)

    # try to get the DSG_IDs... and expect an error
    with pytest.raises(ValueError) as e_info:
        CF.sample_unit_DV(sample_size=100)

# ------------------------------------------------------------------------------
# Damage State
# ------------------------------------------------------------------------------
def test_DamageState_weight():
    """
    Test if the damage state returns the assigned weight value.
    """
    DS = DamageState(ID=1, weight=0.4)

    assert DS.weight == 0.4


def test_DamageState_description():
    """
    Test if the damage state returns the assigned description.
    """
    ref_str = 'Test description.'
    DS = DamageState(ID=1, description=ref_str)

    assert DS.description == ref_str


def test_DamageState_repair_cost_sampling():
    """
    Test if the repair cost consequence function is properly linked to the
    damage state and if it returns the requested samples.
    """

    # create a consequence function (the small standard deviation facilitates
    # the assertion of the returned samples)
    f_median = prep_bounded_linear_median_DV(
        median_max=2.0, median_min=1.0,
        quantity_lower=10.0, quantity_upper=20.0
    )

    RV=RandomVariable(ID=1, dimension_tags=['A'],
                      distribution_kind='lognormal',
                      theta=1.0, COV=1e-10)

    CF = ConsequenceFunction(DV_median=f_median,
                             DV_distribution=RandomVariableSubset(RV,'A'))

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, repair_cost_CF=CF)

    # sample the repair cost distribution
    test_vals = DS.unit_repair_cost(quantity=[5.0, 10.0, 15.0, 20.0, 25.0],
                                    sample_size=4)

    assert test_vals.size == 5

    ref_medians = np.asarray([2.0, 2.0, 1.5, 1.0, 1.0])

    assert_allclose(test_vals, ref_medians, rtol=1e-4)


def test_DamageState_reconstruction_time_sampling():
    """
    Test if the reconstruction time consequence function is properly linked to
    the damage state and if it returns the requested samples.
    """

    # create a consequence function (the small standard deviation facilitates
    # the assertion of the returned samples)
    f_median = prep_bounded_linear_median_DV(
        median_max=2.0, median_min=1.0,
        quantity_lower=10.0, quantity_upper=20.0
    )

    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='lognormal',
                        theta=1.0, COV=1e-10)

    CF = ConsequenceFunction(DV_median=f_median,
                             DV_distribution=RandomVariableSubset(RV, 'A'))

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, reconstruction_time_CF=CF)

    # sample the repair cost distribution
    test_vals = DS.unit_reconstruction_time(
        quantity=[5.0, 10.0, 15.0, 20.0, 25.0],
        sample_size=4)

    assert test_vals.size == 5

    ref_medians = np.asarray([2.0, 2.0, 1.5, 1.0, 1.0])

    assert_allclose(test_vals, ref_medians, rtol=1e-4)

def test_DamageState_red_tag_sampling():
    """
    Test if the red tag consequence function is properly linked to the damage
    state and if it returns the requested samples.
    """

    # create a consequence function
    f_median = prep_constant_median_DV(0.25)

    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='normal',
                        theta=1.0, COV=1.0 ** 2.,
                        truncation_limits=[0., 4.])

    CF = ConsequenceFunction(DV_median=f_median,
                             DV_distribution=RandomVariableSubset(RV, 'A'))

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, red_tag_CF=CF)

    # sample the repair cost distribution
    test_vals = DS.red_tag_dmg_limit(sample_size=1000)

    assert test_vals.size == 1000

    # sample the reference truncated normal distribution and use the samples for testing
    ref_samples = truncnorm.rvs(a=-1., b=3., loc=0.25, scale=0.25, size=1000)

    assert np.mean(test_vals) == pytest.approx(np.mean(ref_samples), rel=0.1)
    assert np.std(test_vals) == pytest.approx(np.std(ref_samples), rel=0.1)

    assert np.min(test_vals) > 0.
    assert np.max(test_vals) < 1.


def test_DamageState_injury_sampling():
    """
    Test if the set of injury consequence functions is properly linked to the
    damage state and if it returns the requested samples.
    """

    # create two consequence functions that are correlated
    ref_median = [0.5, 0.4]
    ref_cov = np.asarray([0.5, 0.6])
    ref_COV = np.outer(ref_cov, ref_cov)
    ref_COV[0, 1] = ref_COV[0, 1] * 0.8
    ref_COV[1, 0] = ref_COV[0, 1]
    tr_lower = np.zeros(2)
    tr_upper = 1. + ((np.ones(2) - ref_median) / ref_median)

    f_median_0 = prep_constant_median_DV(ref_median[0])
    f_median_1 = prep_constant_median_DV(ref_median[1])
    RV = RandomVariable(ID=1, dimension_tags=['A', 'B'],
                        distribution_kind='normal',
                        corr_ref='post',
                        theta=np.ones(2), COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])

    CF_0 = ConsequenceFunction(DV_median=f_median_0,
                               DV_distribution=RandomVariableSubset(RV, 'A'))
    CF_1 = ConsequenceFunction(DV_median=f_median_1,
                               DV_distribution=RandomVariableSubset(RV, 'B'))

    # create a damage state and assign the CF list to it
    DS = DamageState(ID=1, injuries_CF_set=[CF_0, CF_1])

    # sample the two types of injuries and check if the values are appropriate
    for s_i in [0, 1]:
        samples = DS.unit_injuries(severity_level=s_i, sample_size=10000)
        assert samples.size == 10000

        # sample the reference truncated normal distribution and use the
        # samples for testing
        ref_samples = truncnorm.rvs(a=(-1.) / ref_cov[s_i],
                                    b=(tr_upper[s_i] - 1.) / ref_cov[s_i],
                                    loc=ref_median[s_i],
                                    scale=ref_cov[s_i] * ref_median[s_i],
                                    size=1000)

        assert np.mean(samples) == pytest.approx(np.mean(ref_samples), rel=0.1)
        assert np.std(samples) == pytest.approx(np.std(ref_samples), abs=0.02)
        assert np.min(samples) == pytest.approx(0., abs=0.05)
        assert np.max(samples) == pytest.approx(1., abs=0.05)

    # finally, check the correlation between A and B
    test_corr = np.corrcoef(RV.samples, rowvar=False)[0, 1]
    assert  test_corr == pytest.approx(0.8, rel=0.1)

# ------------------------------------------------------------------------------
# Damage State Group
# ------------------------------------------------------------------------------
def test_DamageStateGroup_kind():
    """
    Test if the damage state group returns the assigned group type.
    """
    ref_kind = 'single'
    DSG = DamageStateGroup(ID=1, DS_set=None, DS_set_kind=ref_kind)

    assert DSG._DS_set_kind == ref_kind

# ------------------------------------------------------------------------------
# Performance Group
# ------------------------------------------------------------------------------

def test_PerformanceGroup_Pexc():
    """
    Test if the performance group returns exceedance probabilities from the
    assigned fragility function for a given damage state group appropriately.
    """
    # create the fragility function
    RV = RandomVariable(ID=1, dimension_tags=['A', 'B'],
                        distribution_kind='lognormal',
                        theta=[0.5, 0.7], COV=np.ones((2, 2)) * 0.16)
    FF = FragilityFunction(EDP_limit=RandomVariableSubset(RV, ['A', 'B']))

    # create two damage state groups
    DSG_0 = DamageStateGroup(ID=1, DS_set=None, DS_set_kind='single')
    DSG_1 = DamageStateGroup(ID=2, DS_set=None, DS_set_kind='single')

    # create a random quantity variable
    QNT = RandomVariableSubset(
        RandomVariable(
            ID=2, dimension_tags='Q_A',
            distribution_kind='normal',
            theta=100., COV=100.),
        tags='Q_A'
    )

    # create the performance group
    PG = PerformanceGroup(ID=1, location=1, quantity=QNT,
                          fragility_functions=FF,
                          DSG_set=[DSG_0, DSG_1])

    EDP = np.linspace(0.1, 0.9, 9)

    for edp in EDP:
        assert FF.P_exc(edp, DSG_ID=1) == PG.P_exc(edp, DSG_ID=1)
        assert FF.P_exc(edp, DSG_ID=2) == PG.P_exc(edp, DSG_ID=2)

    assert_allclose(PG.P_exc(EDP, DSG_ID=1), PG.P_exc(EDP, DSG_ID=1),
                    rtol=1e-10)
    assert_allclose(PG.P_exc(EDP, DSG_ID=2), PG.P_exc(EDP, DSG_ID=2),
                    rtol=1e-10)

# ------------------------------------------------------------------------------
# Fragility Group
# ------------------------------------------------------------------------------
def test_FragilityGroup_description_and_name():
    """
    Test if the fragility group returns the assigned description.
    """
    ref_desc = 'Test long description.'
    ref_name = 'Test short description.'

    # create a dummy performance group

    # create the fragility function
    RV = RandomVariable(ID=1, dimension_tags='A',
                        distribution_kind='lognormal',
                        theta=0.5, COV=0.16)
    FF = FragilityFunction(EDP_limit=RandomVariableSubset(RV, 'A'))

    # some of the inputs below do not make sense, but since the subject of the
    # test is not the performance group, they will work fine
    PG = PerformanceGroup(ID=1, location=1,
                          quantity=RandomVariableSubset(RV, 'A'),
                          fragility_functions=FF,
                          DSG_set=None)

    FG = FragilityGroup(ID=1, demand_type='PID',
                        performance_groups = [PG, ],
                        name=ref_name, description=ref_desc)

    assert FG.name == ref_name
    assert FG.description == ref_desc