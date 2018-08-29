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
This subpackage performs unit tests on pelicun.

"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import normaltest, t, chi2
from copy import deepcopy

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir) 

from pelicun.tests.reference_data import standard_normal_table
from pelicun import *

def assert_normal_distribution(sampling_function, ref_mean, ref_stdev):
    """
    Check if the sampling function produces samples from a normal distribution 
    with mean and stdev equal to the provided reference values.
    
    The samples will never have the exact same mean and stdev, which makes 
    such assertions rather difficult. We perform three hypothesis tests: 
    
    (i) test if the distribution is normal using the K2 test by D'Agostino and 
    Pearson (1973);
    (ii) assuming a normal distribution test if its mean is the reference mean;
    (iii) assuming a normal distribution test if its standard deviation is the
    reference standard deviation.
    
    The level of significance is set at 0.05. This would often lead to false 
    negatives during unit testing (in about 15% of the cases), but with a much 
    smaller value we risk not recognizing slight, but consistent errors in 
    sampling. Therefore, instead of lowering the level of significance, we 
    adopt the following strategy: If the samples do not support our null 
    hypothesis, we draw more samples - this conceptually corresponds to 
    performing additional experiments when we experience a strange result in 
    testing. If the underlying distribution is truly not normal, drawing more 
    samples should not help. In the other false negative cases, this should 
    correct the assertion and reduce the likelihood of false negatives to a 
    sufficiently low level that makes this assertion applicable for unit 
    testing. Note that since the additional test results are influenced by the 
    previous (outlier) samples, the results of hypothesis testing are 
    conditioned on the failure of previous tests. Therefore, we have about 
    25% probability of false negatives in the additional tests. Given at most 
    9 sample draws, a false negative assertion will occur about once in every 
    500,000 tests. 
    
    Parameters
    ----------
    sampling_function: function
        Any function that takes sample_size as its only argument and provides
        that many samples of a supposedly normal distribution as a result. 
    ref_mean: float
        Mean of the reference distribution.
    ref_stdev: float
        Standard deviation of the reference distribution.

    Returns
    -------
    output: int
        Assertion result. True if the samples support the hypotheses, False 
        otherwise.
    """

    alpha = 0.05
    chances = 10
    size = 5
    samples = np.asarray([])
    for j in range(chances):
        size = size * 4
        samples = np.concatenate((samples, sampling_function(sample_size=size)))
        size = len(samples)

        # test if the distribution is normal
        __, p_k2 = normaltest(samples)
        if p_k2 > alpha:
            
            # test if the mean and stdev are appropriate
            sample_mean = np.mean(samples)
            sample_stdev = np.std(samples, ddof=1)

            df = size - 1
            mean_stat = ((sample_mean - ref_mean) / 
                         (sample_stdev / np.sqrt(size)))
            p_mean = 2 * t.cdf(-np.abs(mean_stat), df=df)
            
            std_stat = df * sample_stdev ** 2. / ref_stdev ** 2.
            std_stat_delta = np.abs(std_stat - df)
            p_std = (chi2.cdf(df - std_stat_delta, df=df) +
                     (1. - chi2.cdf(df + std_stat_delta, df=df)))
            
            if p_mean > alpha and p_std > alpha:                
                break
       
    # if the hypothesis tests failed after extending the samples several 
    # (i.e. chances) times, then the underlying distribution is probably
    # not normal
    if j==chances-1:
        return False
    else:
        return True
        #return j+1 #- if you want more information
# ------------------------------------------------------------------------------
# Random_Variable
# ------------------------------------------------------------------------------
def test_random_variable_incorrect_none_defined():
    """
    Test if the random variable object raises an error when neither raw data
    nor distributional information is provided. 
    """
    
    with pytest.raises(ValueError) as e_info:
        RandomVariable(ID=1, dimension_tags='test', )
        
def test_random_variable_incorrect_censored_data_definition():
    """
    Test if the random variable object raises an error when raw samples of 
    censored data are provided without sufficient information about the 
    censoring; and test that it does not raise an error when all parameters are 
    provided. 
    """

    # Single dimension
    parameters = dict(ID=1, dimension_tags='test', 
                      raw_data=[1,2,3,4,5,6],
                      detection_limits=[0, None],
                      censored_count = 3)                      
    for missing_p in ['detection_limits', 'censored_count']:
        test_p = deepcopy(parameters)
        test_p[missing_p] = None
        with pytest.raises(ValueError) as e_info:
            RandomVariable(**test_p)              
    assert RandomVariable(**parameters)

    # Multiple dimension
    parameters = dict(ID=1, dimension_tags='test',
                      raw_data=[[1,0], [2,1], [3,1], [4,0], [5,1], [6,0]],
                      detection_limits=[[0,None], [0,1]],
                      censored_count=5)
    for missing_p in ['detection_limits', 'censored_count']:
        test_p = deepcopy(parameters)
        test_p[missing_p] = None
        with pytest.raises(ValueError) as e_info:
            RandomVariable(**test_p)
    assert RandomVariable(**parameters)

def test_random_variable_incorrect_normal_lognormal_definition():
    """
    Test if the random variable object raises an error when a normal or a 
    lognormal distribution is defined with insufficient number of parameters,
    and test that it does not raise an error when all parameters are provided. 
    """

    median = 0.5
    beta = 0.2

    # Single dimension
    for dist in ['normal', 'lognormal']:
        parameters = dict(ID=1, dimension_tags='test', 
                          distribution_kind=dist,
                          theta=median, COV=beta**2.)
        for missing_p in ['theta', 'COV']:
            test_p = deepcopy(parameters)
            test_p[missing_p] = None
            with pytest.raises(ValueError) as e_info:
                RandomVariable(**test_p)              
        assert RandomVariable(**parameters)
        
    # Multiple dimensions
    for dist in ['normal', 'lognormal']:
        parameters = dict(ID=1, dimension_tags=['test_1', 'test_2', 'test_3'], 
                          distribution_kind=dist,
                          theta=np.ones(3) * median, 
                          COV=np.ones((3,3))*beta**2.)
        for missing_p in ['theta', 'COV']:
            test_p = deepcopy(parameters)
            test_p[missing_p] = None
            with pytest.raises(ValueError) as e_info:
                RandomVariable(**test_p)              
        assert RandomVariable(**parameters)

def test_random_variable_incorrect_truncated_normal_lognormal_definition():
    """
    Test if the random variable object raises an error when a truncated normal 
    or a truncated lognormal distribution is defined with insufficient number 
    of parameters, and test that it does not raise an error when all parameters 
    are provided. 
    """

    median = 0.5
    beta = 0.2

    # Single dimension
    for dist in ['truncated_normal', 'truncated_lognormal']:
        parameters = dict(ID=1, dimension_tags='test',
                          distribution_kind=dist,
                          theta=median, COV=beta ** 2.,
                          min_value = 0.1, max_value=0.8)
        for missing_p in [['theta',], ['COV',], ['min_value','max_value']]:
            test_p = deepcopy(parameters)
            for par in missing_p:
                test_p[par] = None
            with pytest.raises(ValueError) as e_info:
                RandomVariable(**test_p)
        for missing_p in [[],['min_value',],['max_value',]]:
            test_p = deepcopy(parameters)
            for par in missing_p:
                test_p[par] = None
            assert RandomVariable(**parameters)

    # Multiple dimensions
    for dist in ['truncated_normal', 'truncated_lognormal']:
        parameters = dict(ID=1, dimension_tags=['test_1', 'test_2', 'test_3'],
                          distribution_kind=dist,
                          theta=np.ones(3) * median,
                          COV=np.ones((3, 3)) * beta ** 2.,
                          min_value=np.ones(3) * 0.1, 
                          max_value=np.ones(3) * 0.8)
        for missing_p in [['theta',], ['COV',], ['min_value','max_value']]:
            test_p = deepcopy(parameters)
            for par in missing_p:
                test_p[par] = None
            with pytest.raises(ValueError) as e_info:
                RandomVariable(**test_p)
        for missing_p in [[],['min_value',],['max_value',]]:
            test_p = deepcopy(parameters)
            for par in missing_p:
                test_p[par] = None
            assert RandomVariable(**parameters)


def test_random_variable_incorrect_multinomial_definition():
    """
    Test if the random variable object raises an error when a multinomial
    distribution is defined with insufficient number of parameters, and test 
    that it does not raise an error when all parameters are provided. 
    """

    p_values = [0.5, 0.2, 0.1, 0.2]

    # Single dimension
    parameters = dict(ID=1, dimension_tags='test',
                      distribution_kind='multinomial',
                      p_set=p_values[0])
    for missing_p in ['p_set',]:
        test_p = deepcopy(parameters)
        test_p[missing_p] = None
        with pytest.raises(ValueError) as e_info:
            RandomVariable(**test_p)
    assert RandomVariable(**parameters)

    # Multiple dimensions
    parameters = dict(ID=1, dimension_tags='test',
                      distribution_kind='multinomial',
                      p_set=p_values)
    for missing_p in ['p_set', ]:
        test_p = deepcopy(parameters)
        test_p[missing_p] = None
        with pytest.raises(ValueError) as e_info:
            RandomVariable(**test_p)
    assert RandomVariable(**parameters)

#-------------------------------------------------------------------------------
# Fragility_Function
# ------------------------------------------------------------------------------

def test_fragility_function_lognormal_unit_mean_unit_std():
    """
    Given a lognormal fragility function with theta=1.0 and beta=1.0, test if 
    the calculated exceedance probabilities are sufficiently accurate.
    The reference results are based on a standard normal table. This limits the
    accuracy of testing to an absolute probability difference of 1e-5.
    """
    EDP = np.exp(np.concatenate([-standard_normal_table[0][::-1],
                                 standard_normal_table[0]]))
    reference_P_exc = np.concatenate([0.5-standard_normal_table[1][::-1], 
                                     0.5+standard_normal_table[1]])
    
    fragility_function = FragilityFunction(theta=1.0, beta=1.0)
    test_P_exc = fragility_function.P_exc(EDP)
        
    assert_allclose(test_P_exc, reference_P_exc, atol=1e-5)

    # if you want the details...
    # for val_ref, val_test in zip(reference_P_exc, test_P_exc):
    #    assert val_ref == pytest.approx(val_test, abs=1e-5)


def test_fragility_function_lognormal_non_trivial_case():
    """
    Given a lognormal fragility function with theta=0.5 and beta=0.2, test if 
    the calculated exceedance probabilities are sufficiently accurate.
    The reference results are based on a standard normal table. This limits the
    accuracy of testing to an absolute probability difference of 1e-5.
    """
    target_theta = 0.5
    target_beta = 0.2
    EDP = np.concatenate([-standard_normal_table[0][::-1],
                           standard_normal_table[0]])
    # modify the inputs to match the table's outputs
    EDP = np.exp(EDP * target_beta + np.log(target_theta))
    reference_P_exc = np.concatenate([0.5 - standard_normal_table[1][::-1],
                                      0.5 + standard_normal_table[1]])

    fragility_function = FragilityFunction(theta=target_theta,
                                           beta=target_beta)
    test_P_exc = fragility_function.P_exc(EDP)

    assert_allclose(test_P_exc, reference_P_exc, atol=1e-5)

    # if you want the details...
    # for val_ref, val_test in zip(reference_P_exc, test_P_exc):
    #    assert val_ref == pytest.approx(val_test, abs=1e-5)
    
def test_fragility_function_lognormal_zero_input():
    """
    Given a zero EDP input to a lognormal fragility function, the result shall
    be 0 exceedance probability, even though zero input in log space shall
    correspond to -infinity. This slight modification makes our lives much
    easier when real inputs are fed to the fragility functions.
    """
    fragility_function = FragilityFunction(theta=1.0, beta=1.0)
    
    test_P_exc = fragility_function.P_exc(0.)
    
    assert test_P_exc == 0.
    
def test_fragility_function_lognormal_nonzero_scalar_input():
    """
    Given a nonzero scalar EDP input, the fragility function should return a 
    nonzero scalar output.
    """
    fragility_function = FragilityFunction(theta=1.0, beta=1.0)
    
    test_P_exc = fragility_function.P_exc(standard_normal_table[0][0])
    
    assert test_P_exc == pytest.approx(standard_normal_table[1][0],abs=1e-5)

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
        parameters = dict(median_kind='fixed', distribution_kind=dist,
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
        for max_val, min_val in zip([np.inf, 10.,10.],[0.5, 0.,0.5]):
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
    for max_val, min_val in zip([np.inf, 10.,10.],[0.5, 0.,0.5]):        
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
    ref_median=0.5
    ref_stdev=0.25
    
    # first define the function using the standard deviation
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='normal',
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
    
    #then check if the distribution of the samples is appropriate
    assert assert_normal_distribution(CF.unit_consequence, 
                                      ref_median, ref_stdev)

    # perform the same  distribution test with a CF defined through a 
    # coefficient of variation
    ref_cov = 0.5
    CF = ConsequenceFunction(median_kind='fixed',
                             distribution_kind='normal',
                             coefficient_of_variation=ref_cov,
                             median=ref_median)
    
    assert assert_normal_distribution(CF.unit_consequence,
                                      ref_median, ref_stdev)
    
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
        lambda sample_size: np.log(CF.unit_consequence(sample_size=sample_size)),
        np.log(ref_median), ref_stdev)


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
                                      ref_median, ref_stdev)

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
                                      ref_median, ref_stdev)
    
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
    DS = DamageState(ID=1, repair_cost_CF = CF)
    
    # sample the repair cost distribution
    test_vals = DS.unit_repair_cost(quantity=[[10.0, 15.0, 20.0],
                                              [20.0, 25.0,  5.0]],
                                    sample_size=4)
    
    assert test_vals.shape == (4, 2, 3)
    
    ref_medians = np.asarray([[2.0, 1.5, 1.0],[1.0, 1.0, 2.0]])
    
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

    ref_medians = np.ones(4)*0.5

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
    ref_medians = np.ones(4)*0.5
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

def test_damage_state_group_fragility():
    """
    Test if the damage state group returns results from the assigned fragility
    function.
    """
    FF = FragilityFunction(theta=0.5, beta=0.2)
    DSG = DamageStateGroup(ID=1, DS_set=None, DS_set_kind='single',
                           fragility_function=FF)
    
    EDP = np.linspace(0.1,0.9,9)
    
    for edp in EDP:
        assert FF.P_exc(edp) == DSG.P_exc(edp)
        
    assert_allclose(DSG.P_exc(EDP), FF.P_exc(EDP), rtol=1e-10)
    
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
    
