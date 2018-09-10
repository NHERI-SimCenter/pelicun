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
from copy import deepcopy

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir)) 

from pelicun.tests.test_reference_data import standard_normal_table
from pelicun.tests.test_pelicun import assert_normal_distribution
from pelicun.model import *
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

def test_conseq_function_fixed_no_median_defined():
    """
    Test if the function raises an error when its median parameter is not 
    specified.
    """
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        with pytest.raises(ValueError) as e_info:
            ConsequenceFunction(median_kind='fixed',
                                distribution_kind=dist,
                                standard_deviation=1.0)


def test_conseq_function_fixed_both_std_and_cov_defined():
    """
    Test if the function raises an error when the dispersion of the quantities
    is defined in two ways simultaneously.
    """
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        with pytest.raises(ValueError) as e_info:
            ConsequenceFunction(median_kind='fixed',
                                distribution_kind=dist,
                                standard_deviation=1.0,
                                coefficient_of_variation=0.5,
                                median=1.0)


def test_conseq_function_fixed_neither_std_nor_cov_defined():
    """
    Test if the function raises an error when the dispersion of the quantities
    is undefined.
    """
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        with pytest.raises(ValueError) as e_info:
            ConsequenceFunction(median_kind='fixed',
                                distribution_kind=dist,
                                median=1.0)


def test_conseq_function_fixed_dispersion_with_cov():
    """
    Test if the function raises an error for lognormal distributions if the
    dispersion is defined through cov, and if it works well for normal 
    distributions.
    """
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        parameters = dict(median_kind='fixed',
                          distribution_kind=dist,
                          coefficient_of_variation=0.3,
                          median=1.0)
        if dist == 'lognormal':
            with pytest.raises(ValueError) as e_info:
                ConsequenceFunction(**parameters)
        else:
            CF = ConsequenceFunction(**parameters)
            assert CF.median() == 1.0


def test_conseq_function_fixed_normal_median_value():
    """
    Test if the function returns the prescribed median.
    """
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        conseq_function = ConsequenceFunction(median_kind='fixed',
                                              distribution_kind=dist,
                                              standard_deviation=1.0,
                                              median=1.0)

        assert conseq_function.median() == 1.0


def test_conseq_function_bounded_linear_incorrect_definition():
    """
    Test if the function raises an error when at least one of its median 
    function parameters is not specified.
    """

    for dist in ['normal', 'lognormal', 'truncated_normal']:
        parameters = dict(median_kind='bounded_linear', distribution_kind=dist,
                          standard_deviation=1.0,
                          quantity_lower_bound=1.,
                          quantity_upper_bound=2.,
                          median_max=2.,
                          median_min=1.)
        for missing_p in ['median_min', 'median_max',
                          'quantity_lower_bound', 'quantity_upper_bound']:
            test_p = deepcopy(parameters)
            test_p[missing_p] = None
            with pytest.raises(ValueError) as e_info:
                ConsequenceFunction(**test_p)


def test_conseq_function_bounded_linear_incorrect_median_query():
    """
    Test if the function returns an error when its median is requested without
    specifying the quantity of damaged components.
    """
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        conseq_function = ConsequenceFunction(median_kind='bounded_linear',
                                              distribution_kind=dist,
                                              standard_deviation=1.0,
                                              median_min=1.0,
                                              median_max=2.0,
                                              quantity_lower_bound=1.0,
                                              quantity_upper_bound=2.0)

        with pytest.raises(ValueError) as e_info:
            output = conseq_function.median()


def test_conseq_function_bounded_linear_median_value():
    """
    Test if the function returns an appropriate output for single quantities 
    and for quantity arrays.
    """
    test_quants = [0.5, 1.0, 1.5, 2.0, 2.5]
    ref_vals = [2.0, 2.0, 1.5, 1.0, 1.0]
    for dist in ['normal', 'lognormal', 'truncated_normal']:
        conseq_function = ConsequenceFunction(median_kind='bounded_linear',
                                              distribution_kind=dist,
                                              standard_deviation=1.0,
                                              median_min=1.0,
                                              median_max=2.0,
                                              quantity_lower_bound=1.0,
                                              quantity_upper_bound=2.0)

        # single quantities
        for quant, ref_val in zip(test_quants, ref_vals):
            assert conseq_function.median(quantity=quant) == ref_val

        # quantity array
        test_medians = conseq_function.median(quantity=test_quants)
        assert_allclose(test_medians, ref_vals, rtol=1e-10)


def test_conseq_function_min_max_values_incorrect_definition():
    """
    Test if the function raises an error when the max_value or min_value 
    parameter is specified for a non-truncated distribution type.
    """
    for dist in ['normal', 'lognormal']:
        for max_val, min_val in zip([np.inf, 10., 10.], [0.5, 0., 0.5]):
            with pytest.raises(ValueError) as e_info:
                ConsequenceFunction(median_kind='fixed',
                                    distribution_kind=dist,
                                    standard_deviation=1.0,
                                    median=1.0,
                                    max_value=max_val,
                                    min_value=min_val)


def test_conseq_function_min_max_values_correct_definition_median():
    """
    Test if the function returns a correct median when the max_value or 
    min_value parameter is specified for a truncated distribution type.
    """
    for max_val, min_val in zip([np.inf, 10., 10.], [0.5, 0., 0.5]):
        CF = ConsequenceFunction(median_kind='fixed',
                                 distribution_kind='truncated_normal',
                                 standard_deviation=1.0,
                                 median=1.0,
                                 max_value=max_val,
                                 min_value=min_val)

        assert CF.median() == 1.0


def test_conseq_function_normal_samples():
    """
    Test if the function samples the consequence distribution properly for a
    normal distribution with fixed median.
    """
    ref_median = 0.5
    ref_stdev = 0.25

    # first define the function using the standard deviation
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='normal',
                             standard_deviation=ref_stdev,
                             median=ref_median)

    # first check that the shape of the returned values is appropriate
    # Note that if the median is fixed, then providing the quantity attribute
    # shall not change the returned value.
    assert CF.unit_consequence().shape == ()
    assert CF.unit_consequence(quantity=[1.0]).shape == ()
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]]).shape == ()

    assert CF.unit_consequence(sample_size=2).shape == (2,)
    assert CF.unit_consequence(quantity=[1.0], sample_size=2).shape == (2,)
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]],
                               sample_size=2).shape == (2,)

    # then check if the distribution of the samples is appropriate
    assert assert_normal_distribution(CF.unit_consequence,
                                      ref_median, ref_stdev**2.)

    # perform the same  distribution test with a CF defined through a 
    # coefficient of variation
    ref_cov = 0.5
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='normal',
                             coefficient_of_variation=ref_cov,
                             median=ref_median)

    assert assert_normal_distribution(CF.unit_consequence,
                                      ref_median, ref_stdev**2.)

    # test if the shape of the returned values is appropriate for a consequence
    # function with bounded linear median
    CF = ConsequenceFunction(median_kind='bounded_linear',
                             distribution_kind='normal',
                             standard_deviation=ref_stdev,
                             median_min=ref_median - 1.,
                             median_max=ref_median + 1.,
                             quantity_lower_bound=1.0,
                             quantity_upper_bound=2.0)

    with pytest.raises(ValueError) as e_info:
        CF.unit_consequence()

    assert CF.unit_consequence(quantity=[1.0]).shape == (1,)
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]]).shape == (
    2, 2)

    with pytest.raises(ValueError) as e_info:
        CF.unit_consequence(sample_size=3)

    assert CF.unit_consequence(quantity=[1.0], sample_size=3).shape == (3, 1)
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]],
                               sample_size=3).shape == (3, 2, 2)


def test_conseq_function_lognormal_samples():
    """
    Test if the function samples the consequence distribution properly for a
    lognormal distribution with fixed median.
    """
    ref_median = 0.5
    ref_stdev = 0.5
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='lognormal',
                             standard_deviation=ref_stdev,
                             median=ref_median)

    # first check that the shape of the returned values is appropriate
    assert CF.unit_consequence().shape == ()
    assert CF.unit_consequence(quantity=[1.0]).shape == ()
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]]).shape == ()

    assert CF.unit_consequence(sample_size=2).shape == (2,)
    assert CF.unit_consequence(quantity=[1.0], sample_size=2).shape == (2,)
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]],
                               sample_size=2).shape == (2,)

    # then check if the distribution of the samples is appropriate
    assert assert_normal_distribution(
        lambda sample_size: np.log(
            CF.unit_consequence(sample_size=sample_size)),
        np.log(ref_median), ref_stdev**2.)


def test_conseq_function_truncated_normal_samples():
    """
    Test if the function samples the consequence distribution properly for a
    truncated normal distribution with fixed median.
    """
    ref_median = 0.5
    ref_stdev = 0.25

    # Start with testing the truncated normal distribution using a very large
    # range. This should produce a normal distribution and shall pass the tests
    # designed for that case.
    # first define the function using the standard deviation
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='truncated_normal',
                             standard_deviation=ref_stdev,
                             median=ref_median,
                             min_value=-1.e+10,
                             max_value=1.e+10
                             )

    # first check that the shape of the returned values is appropriate
    assert CF.unit_consequence().shape == ()
    assert CF.unit_consequence(quantity=[1.0]).shape == ()
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]]).shape == ()

    assert CF.unit_consequence(sample_size=2).shape == (2,)
    assert CF.unit_consequence(quantity=[1.0], sample_size=2).shape == (2,)
    assert CF.unit_consequence(quantity=[[1.0, 2.0], [1.0, 2.0]],
                               sample_size=2).shape == (2,)

    # then check if the distribution of the samples is appropriate
    assert assert_normal_distribution(CF.unit_consequence,
                                      ref_median, ref_stdev**2.)

    # perform the same assertion with a CF defined using a coefficient of 
    # variation
    ref_cov = 0.5
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='truncated_normal',
                             coefficient_of_variation=ref_cov,
                             median=ref_median,
                             min_value=-1.e+10,
                             max_value=1.e+10
                             )

    assert assert_normal_distribution(CF.unit_consequence,
                                      ref_median, ref_stdev**2.)

    # Next, test if the truncation works appropriately by creating a CF with
    # much stricter bounds...
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='truncated_normal',
                             coefficient_of_variation=ref_cov,
                             median=ref_median,
                             min_value=0.25,
                             max_value=0.75
                             )

    # and checking that the every generated sample is within those bounds
    samples = CF.unit_consequence(sample_size=10000)
    assert np.max(samples) <= 0.75
    assert np.min(samples) >= 0.25


# ------------------------------------------------------------------------------
# Damage State
# ------------------------------------------------------------------------------
def test_damage_state_weight():
    """
    Test if the damage state returns the assigned weight value.
    """
    DS = DamageState(ID=1, weight=0.4)

    assert DS.weight == 0.4


def test_damage_state_description():
    """
    Test if the damage state returns the assigned description. 
    """
    ref_str = 'Test description.'
    DS = DamageState(ID=1, description=ref_str)

    assert DS.description == ref_str


def test_damage_state_repair_cost_sampling():
    """
    Test if the repair cost consequence function is properly linked to the 
    damage state and if it returns the requested samples.
    """

    # create a consequence function (the small standard deviation facilitates
    # the assertion of the returned samples)
    CF = ConsequenceFunction(median_kind='bounded_linear',
                             distribution_kind='lognormal',
                             standard_deviation=1e-6,
                             median_min=1.0,
                             median_max=2.0,
                             quantity_lower_bound=10.0,
                             quantity_upper_bound=20.0)

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, repair_cost_CF=CF)

    # sample the repair cost distribution
    test_vals = DS.unit_repair_cost(quantity=[[10.0, 15.0, 20.0],
                                              [20.0, 25.0, 5.0]],
                                    sample_size=4)

    assert test_vals.shape == (4, 2, 3)

    ref_medians = np.asarray([[2.0, 1.5, 1.0], [1.0, 1.0, 2.0]])

    for sample in test_vals:
        assert_allclose(sample, ref_medians, rtol=1e-4)


def test_damage_state_reconstruction_time_sampling():
    """
    Test if the reconstruction time consequence function is properly linked to 
    the damage state and if it returns the requested samples.
    """

    # create a consequence function (the small standard deviation facilitates
    # the assertion of the returned samples)
    CF = ConsequenceFunction(median_kind='bounded_linear',
                             distribution_kind='lognormal',
                             standard_deviation=1e-6,
                             median_min=1.0,
                             median_max=2.0,
                             quantity_lower_bound=10.0,
                             quantity_upper_bound=20.0)

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, reconstruction_time_CF=CF)

    # sample the reconstruction time distribution
    test_vals = DS.unit_reconstruction_time(quantity=[[10.0, 15.0, 20.0],
                                                      [20.0, 25.0, 5.0]],
                                            sample_size=4)

    assert test_vals.shape == (4, 2, 3)

    ref_medians = np.asarray([[2.0, 1.5, 1.0], [1.0, 1.0, 2.0]])

    for sample in test_vals:
        assert_allclose(sample, ref_medians, rtol=1e-4)


def test_damage_state_red_tag_sampling():
    """
    Test if the red tag consequence function is properly linked to the damage 
    state and if it returns the requested samples.
    """

    # create a consequence function (the small standard deviation facilitates
    # the assertion of the returned samples)
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='truncated_normal',
                             standard_deviation=1e-6,
                             median=0.5,
                             min_value=0.0,
                             max_value=1.0)

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, red_tag_CF=CF)

    # sample the red tag distribution
    test_vals = DS.unit_red_tag(sample_size=4)

    assert test_vals.shape == (4,)

    ref_medians = np.ones(4) * 0.5

    assert_allclose(test_vals, ref_medians, rtol=1e-4)


def test_damage_state_injury_sampling():
    """
    Test if the set of injury consequence functions is properly linked to the 
    damage state and if it returns the requested samples.
    """

    # create two consequence functions (the small standard deviation facilitates
    # the assertion of the returned samples)
    CF_0 = ConsequenceFunction(median_kind='fixed',
                               distribution_kind='truncated_normal',
                               standard_deviation=1e-6,
                               median=0.5, min_value=0.0, max_value=1.0)

    CF_1 = ConsequenceFunction(median_kind='fixed',
                               distribution_kind='truncated_normal',
                               standard_deviation=1e-6,
                               median=0.1, min_value=0.0, max_value=1.0)

    # create a damage state and assign the CF to it
    DS = DamageState(ID=1, injuries_CF_set=[CF_0, CF_1])

    # first, sample the distribution of lower injury severity
    test_vals = DS.unit_injuries(severity_level=0, sample_size=4)
    assert test_vals.shape == (4,)
    ref_medians = np.ones(4) * 0.5
    assert_allclose(test_vals, ref_medians, rtol=1e-4)

    # then, sample the distribution of higher injury severity
    test_vals = DS.unit_injuries(severity_level=1, sample_size=4)
    assert test_vals.shape == (4,)
    ref_medians = np.ones(4) * 0.1
    assert_allclose(test_vals, ref_medians, rtol=1e-4)


# ------------------------------------------------------------------------------
# Damage State Group
# ------------------------------------------------------------------------------
def test_damage_state_group_description():
    """
    Test if the damage state group returns the assigned description. 
    """
    ref_str = 'Test description.'
    DSG = DamageStateGroup(ID=1, description=ref_str,
                           DS_set=None, DS_set_kind='single',
                           fragility_function=None)

    assert DSG.description == ref_str

'''
def test_damage_state_group_fragility():
    """
    Test if the damage state group returns results from the assigned fragility
    function.
    """
    FF = FragilityFunction(theta=0.5, beta=0.2)
    DSG = DamageStateGroup(ID=1, DS_set=None, DS_set_kind='single',
                           fragility_function=FF)

    EDP = np.linspace(0.1, 0.9, 9)

    for edp in EDP:
        assert FF.P_exc(edp) == DSG.P_exc(edp)

    assert_allclose(DSG.P_exc(EDP), FF.P_exc(EDP), rtol=1e-10)
'''

# ------------------------------------------------------------------------------
# Fragility Group
# ------------------------------------------------------------------------------
def test_fragility_group_description():
    """
    Test if the fragility group returns the assigned description. 
    """
    ref_str = 'Test description.'
    FG = FragilityGroup(ID=1, kind='structural', demand_type='PID',
                        description=ref_str, DSG_set=None)

    assert FG.description == ref_str