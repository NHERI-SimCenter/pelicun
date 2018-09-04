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
This subpackage performs unit tests on the random module of pelicun.

"""
import pytest
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose
from copy import deepcopy

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir)) 

from pelicun.tests.test_reference_data import standard_normal_table
from pelicun.tests.test_pelicun import assert_normal_distribution
from pelicun.uq import *

# ------------------------------------------------------------------------------
# tmvn_rvs
# ------------------------------------------------------------------------------
def test_TMVN_sampling_alpha_error():
    """
    Test if the function raises an error when the probability density that the
    truncation limits define is not sufficiently accurate for further use.

    """
    with pytest.raises(ValueError) as e_info:
        tmvn_rvs(0.5, 0.25**2., lower=-10., upper=-9.9, size=10)
        
def test_TMVN_sampling_alpha_warning():
    """
    Test if the function shows a warning when the rejection rate is too high.

    """
    with pytest.warns(UserWarning) as e_info:
        tmvn_rvs(0.5, 0.25**2., lower=-1., upper=-0.3, size=10)
        
def test_TMVN_sampling_multiple_iterations():
    """
    Test if the function can perform multiple iterations to collect the 
    required number of samples. It is hard to force the function to do more
    than one iteration and it is designed to work using a single iteration in
    the majority of cases. However, by assigning a sufficiently narrow 
    truncated area and running the sampling multiple times, the probability of
    running into multiple iterations is above 99.9%.
    
    """
    counter = 0
    for i in range(20):
        samples = tmvn_rvs(0.5, 0.25, lower=-0.1, upper=0., size=5)
        counter += len(samples)
     
    # verify that the right number of samples was returned   
    assert counter == 20*5

def test_TMVN_sampling_non_truncated():
    """
    Test if the sampling method returns appropriate samples for a non-truncated
    univariate and multivariate normal distribution.

    """
    
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std**2.
    
    sampling_function = lambda sample_size: tmvn_rvs(ref_mean, ref_var, 
                                                     size=sample_size)
    assert assert_normal_distribution(sampling_function, ref_mean, ref_var)
    
    # bivariate case, various correlation coefficients
    rho_list = [-1., -0.5, 0., 0.5, 1.]
    for rho in rho_list:
        ref_mean = [0.5, 1.5]
        ref_std = [0.25, 1.0]
        ref_rho = np.asarray([[1.0, rho],[rho, 1.0]])
        ref_COV = np.outer(ref_std, ref_std) * ref_rho
        
        sampling_function = lambda sample_size: tmvn_rvs(ref_mean, ref_COV, 
                                                         size=sample_size)
        
        assert assert_normal_distribution(sampling_function, ref_mean, ref_COV)
        
    # multi-dimensional case
    dims = 5
    ref_mean = np.arange(dims)
    ref_std = np.arange(1,dims+1)
    ref_rho = np.ones((dims,dims))*0.3
    ref_COV = np.outer(ref_std,ref_std) * ref_rho
    np.fill_diagonal(ref_COV,ref_std**2.)
    
    sampling_function = lambda sample_size: tmvn_rvs(ref_mean, ref_COV,
                                                     size=sample_size)
    
    assert assert_normal_distribution(sampling_function, ref_mean, ref_COV)

def test_TMVN_sampling_truncated_wide_limits():
    """
    Test if the sampling method returns appropriate samples for a truncated
    univariate and multivariate normal distribution when the truncation limits
    are sufficiently wide to consider the result a normal distribution.

    """
    
    # assign a non-symmetric, but very wide set of limits for all cases

    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    lower = -1e10
    upper = 1e9

    sampling_function = lambda sample_size: tmvn_rvs(ref_mean, ref_var,
                                                     lower = lower,
                                                     upper = upper,
                                                     size=sample_size)
    assert assert_normal_distribution(sampling_function, ref_mean, ref_var)

    # bivariate case, various correlation coefficients
    rho_list = [-1., -0.5, 0., 0.5, 1.]
    lower = np.ones(2) * lower
    upper = np.ones(2) * upper
    for rho in rho_list:
        ref_mean = [0.5, 1.5]
        ref_std = [0.25, 1.0]
        ref_rho = np.asarray([[1.0, rho], [rho, 1.0]])
        ref_COV = np.outer(ref_std, ref_std) * ref_rho

        sampling_function = lambda sample_size: tmvn_rvs(ref_mean, ref_COV,
                                                         lower=lower,
                                                         upper=upper,
                                                         size=sample_size)

        assert assert_normal_distribution(sampling_function, ref_mean, ref_COV)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.3
    ref_COV = np.outer(ref_std, ref_std) * ref_rho
    np.fill_diagonal(ref_COV, ref_std ** 2.)
    lower = np.ones(dims) * lower[0]
    upper = np.ones(dims) * upper[0]
    
    sampling_function = lambda sample_size: tmvn_rvs(ref_mean, ref_COV,
                                                     lower=lower,
                                                     upper=upper,
                                                     size=sample_size)

    assert assert_normal_distribution(sampling_function, ref_mean, ref_COV)

def test_TMVN_sampling_truncated_narrow_limits():
    """
    Test if the sampling method returns appropriate samples for a truncated
    univariate and multivariate normal distribution when the truncation limits
    are narrow.

    """
    
    # here we focus on testing if the returned samples are between the 
    # pre-defined limits in all dimensions
    
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    lower = 0.1
    upper = 0.6

    samples = tmvn_rvs(ref_mean, ref_var, lower=lower, upper=upper, 
                       size=1000)
    
    sample_min = np.min(samples)
    sample_max = np.max(samples)
    
    assert sample_min > lower
    assert sample_max < upper
    
    assert sample_min == pytest.approx(lower, abs=0.01)
    assert sample_max == pytest.approx(upper, abs=0.01)

    # multi-dimensional case
    dims = 5
    ref_mean = np.arange(dims)
    ref_std = np.ones(1) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.3
    ref_COV = np.outer(ref_std, ref_std) * ref_rho
    np.fill_diagonal(ref_COV, ref_std ** 2.)
    
    lower = ref_mean - ref_std * 2.5
    upper = ref_mean + ref_std * 1.5
    
    samples = tmvn_rvs(ref_mean, ref_COV, lower=lower, upper=upper, 
                       size=1000)
    
    sample_min = np.amin(samples, axis=0)
    sample_max = np.amax(samples, axis=0)
    
    assert np.all(sample_min > lower)
    assert np.all(sample_max < upper)
    
    assert_allclose(sample_min, lower, atol=0.1)
    assert_allclose(sample_max, upper, atol=0.1)
    
# ------------------------------------------------------------------------------
# mvn_orthotope_density
# ------------------------------------------------------------------------------
def test_MVN_CDF_univariate():
    """
    Test if the MVN CDF function provides accurate results for the special 
    univariate case.
    
    """
    
    # Testing is based on the CDF results from scipy's norm function.
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    lower = np.arange(13)/5.-1.5
    upper = np.arange(13)/3.
    for l,u in zip(lower,upper):
        ref_l, ref_u = norm.cdf([l,u],loc=ref_mean, scale=ref_std)
        ref_res = ref_u-ref_l
        test_res, __ = mvn_orthotope_density(ref_mean, ref_var, 
                                             lower=l, upper=u)
        
        assert ref_res == pytest.approx(test_res)
        
    # test if the function works properly with infinite boundaries
    test_res, __ = mvn_orthotope_density(ref_mean, ref_var)
    assert test_res == 1.
        
def test_MVN_CDF_multivariate():
    """
    Test if the MVN CDF function provides accurate results for multivariate 
    cases.
    
    """
    # Testing is based on univariate results compared to estimates of MVN CDF
    # with special correlation structure
    
    # First, assume perfect correlation. Results should be identical for all
    # dimensions
    
    for dims in range(2, 6):
        ref_mean = np.arange(dims)
        ref_std = np.arange(1, dims + 1)
        ref_rho = np.ones((dims, dims)) * 1.0
        ref_COV = np.outer(ref_std, ref_std) * ref_rho
        np.fill_diagonal(ref_COV, ref_std ** 2.)

        lower = ref_mean - ref_std * 2.5
        upper = ref_mean + ref_std * 1.5

        test_res, __ = mvn_orthotope_density(ref_mean, ref_COV,
                                             lower=lower, upper=upper)

        ref_l, ref_u = norm.cdf([-2.5, 1.5], loc=0., scale=1.)
        ref_res = ref_u - ref_l
        
        assert ref_res == pytest.approx(test_res)
        
    # Second, assume independence. Results should be equal to the univariate
    # result on the power of the number of dimensions.
    for dims in range(2, 6):
        ref_mean = np.arange(dims)
        ref_std = np.arange(1, dims + 1)
        ref_rho = np.ones((dims, dims)) * 0.0
        ref_COV = np.outer(ref_std, ref_std) * ref_rho
        np.fill_diagonal(ref_COV, ref_std ** 2.)

        lower = ref_mean - ref_std * 2.5
        upper = ref_mean + ref_std * 1.5

        test_res, __ = mvn_orthotope_density(ref_mean, ref_COV,
                                             lower=lower, upper=upper)

        ref_l, ref_u = norm.cdf([-2.5, 1.5], loc=0., scale=1.)
        ref_res = ref_u - ref_l

        assert ref_res**dims == pytest.approx(test_res)

# ------------------------------------------------------------------------------
# tmvn_MLE
# ------------------------------------------------------------------------------
def test_MVN_MLE_baseline():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in the baseline case with no truncation and no 
    censoring.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.

    # generate samples
    samples = tmvn_rvs(ref_mean, ref_var,
                       size=1000)
    
    # estimate the parameters of the distribution
    mu, var = tmvn_MLE(samples)

    assert ref_mean == pytest.approx(mu, abs=0.05)
    assert ref_var == pytest.approx(var, rel=0.2)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    samples = tmvn_rvs(ref_mean, ref_COV, size=100)
    
    test_mu, test_COV = tmvn_MLE(np.transpose(samples))
    test_std = np.sqrt(test_COV.diagonal())
    test_rho = test_COV/np.outer(test_std,test_std)

    assert_allclose(test_mu, ref_mean, atol=0.15)
    assert_allclose(test_std**2., ref_std**2., rtol=0.5)
    assert_allclose(test_rho, ref_rho, atol=0.3)


def test_MVN_MLE_censored():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in cases with no truncation and censored data.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    c_lower = 0.35
    c_upper = 1.25

    # generate censored samples
    samples = tmvn_rvs(ref_mean, ref_var,
                       size=1000)

    good_ones = np.all([samples>c_lower, samples<c_upper],axis=0)
    c_samples = samples[good_ones]
    c_count = 1000 - sum(good_ones)

    # estimate the parameters of the distribution
    mu, var = tmvn_MLE(c_samples, censored_count=c_count,
                       det_lower=c_lower, det_upper=c_upper)

    assert ref_mean == pytest.approx(mu, abs=0.05)
    assert ref_var == pytest.approx(var, rel=0.2)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho
    
    c_lower = ref_mean - 0.5 * ref_std
    c_upper = ref_mean + 8.5 * ref_std
    
    c_lower[2] = -np.inf
    c_upper[0] = np.inf

    samples = tmvn_rvs(ref_mean, ref_COV, size=1000)

    good_ones = np.all([samples > c_lower, samples < c_upper], axis=0)
    good_ones = np.all(good_ones, axis=1)
    c_samples = np.transpose(samples[good_ones])
    c_count = 1000 - sum(good_ones)

    test_mu, test_COV = tmvn_MLE(c_samples,
                                 censored_count=c_count,
                                 det_lower=c_lower, det_upper=c_upper)
    test_std = np.sqrt(test_COV.diagonal())
    test_rho = test_COV / np.outer(test_std, test_std)

    assert_allclose(test_mu, ref_mean, atol=0.15)
    assert_allclose(test_std ** 2., ref_std ** 2., rtol=0.5)
    assert_allclose(test_rho, ref_rho, atol=0.4)


def test_MVN_MLE_truncated():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in cases with truncation and uncensored data.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    tr_lower = 0.35
    tr_upper = 1.25

    # generate samples of a TMVN distribution 
    # (assume the tmvn_rvs function works properly)
    samples = tmvn_rvs(ref_mean, ref_var, lower=tr_lower, upper=tr_upper,
                       size=1000)

    # estimate the parameters of the distribution
    mu, var = tmvn_MLE(samples, tr_lower=tr_lower, tr_upper=tr_upper)

    assert ref_mean == pytest.approx(mu, abs=0.05)
    assert ref_var == pytest.approx(var, rel=0.2)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    tr_lower = ref_mean - 0.5 * ref_std
    tr_upper = ref_mean + 8.5 * ref_std

    tr_lower[2] = -np.inf
    tr_upper[0] = np.inf

    samples = tmvn_rvs(ref_mean, ref_COV, 
                       lower=tr_lower, upper=tr_upper, 
                       size=500)

    test_mu, test_COV = tmvn_MLE(np.transpose(samples),
                                 tr_lower=tr_lower, tr_upper=tr_upper,)
    test_std = np.sqrt(test_COV.diagonal())
    test_rho = test_COV / np.outer(test_std, test_std)

    assert_allclose(test_mu, ref_mean, atol=0.15)
    assert_allclose(test_std ** 2., ref_std ** 2., rtol=0.5)
    assert_allclose(test_rho, ref_rho, atol=0.4)
    
def test_MVN_MLE_truncated_and_censored():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in cases with truncation and censored data.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    tr_lower = 0.35
    tr_upper = 2.5
    det_upper = 1.25
    det_lower = tr_lower

    # generate samples of a TMVN distribution 
    # (assume the tmvn_rvs function works properly)
    samples = tmvn_rvs(ref_mean, ref_var, lower=tr_lower, upper=tr_upper,
                       size=1000)
    
    # censor the samples
    good_ones = samples < det_upper
    c_samples = samples[good_ones]
    c_count = 1000 - sum(good_ones)

    # estimate the parameters of the distribution
    mu, var = tmvn_MLE(c_samples, tr_lower=tr_lower, tr_upper=tr_upper,
                       censored_count=c_count, 
                       det_lower=det_lower, det_upper=det_upper)

    assert ref_mean == pytest.approx(mu, abs=0.05)
    assert ref_var == pytest.approx(var, rel=0.2)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    tr_lower = ref_mean - 4.5 * ref_std
    tr_upper = ref_mean + 2.5 * ref_std
    tr_lower[2] = -np.inf
    tr_upper[0] = np.inf

    det_lower = ref_mean - 1. * ref_std
    det_lower[2] = -np.inf
    det_upper = tr_upper

    samples = tmvn_rvs(ref_mean, ref_COV,
                       lower=tr_lower, upper=tr_upper,
                       size=1000)

    good_ones = np.all([samples > det_lower, samples < det_upper], axis=0)
    good_ones = np.all(good_ones, axis=1)
    c_samples = np.transpose(samples[good_ones])
    c_count = 1000 - sum(good_ones)

    test_mu, test_COV = tmvn_MLE(c_samples,
                                 tr_lower=tr_lower, tr_upper=tr_upper, 
                                 censored_count = c_count, 
                                 det_lower=det_lower, det_upper=det_upper)
    test_std = np.sqrt(test_COV.diagonal())
    test_rho = test_COV / np.outer(test_std, test_std)

    assert_allclose(test_mu, ref_mean, atol=0.15)
    assert_allclose(test_std ** 2., ref_std ** 2., rtol=0.5)
    assert_allclose(test_rho, ref_rho, atol=0.4)
    
def test_MVN_MLE_small_alpha():
    """
    Assigning truncation or detection limits that correspond to very small 
    probability densities shall raise warning messages. Test if the messages
    are raised.

    """
    
    # use a univariate case
    ref_mean = 0.5
    ref_std = 0.25
    ref_var = ref_std ** 2.
    tr_lower = -1.0
    tr_upper = -0.2
    det_upper = -0.4
    det_lower = tr_lower

    # generate samples of a TMVN distribution 
    # (assume the tmvn_rvs function works properly)
    samples = tmvn_rvs(ref_mean, ref_var, lower=tr_lower, upper=tr_upper,
                       size=100)

    # censor the samples
    good_ones = np.all([samples > det_lower, samples < det_upper], axis=0)
    c_samples = samples[good_ones]
    c_count = 100 - sum(good_ones)

    # warning about truncation limits
    with pytest.warns(UserWarning) as e_info:
        tmvn_MLE(c_samples, tr_lower=tr_lower, tr_upper=tr_upper - 0.6,
                 censored_count=c_count, det_lower=det_lower, 
                 det_upper=det_upper)

    # warning about detection limits
    with pytest.warns(UserWarning) as e_info:
        tmvn_MLE(c_samples, tr_lower=tr_lower, tr_upper=tr_upper,
                 censored_count=c_count, det_lower=det_lower,
                 det_upper=det_upper - 0.6)
        
    # warning about alpha being smaller than the specified limit
    with pytest.warns(UserWarning) as e_info:
        tmvn_MLE(c_samples, tr_lower=tr_lower, tr_upper=tr_upper,
                 censored_count=c_count, det_lower=det_lower,
                 det_upper=det_upper, alpha_lim=0.2)

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
                      raw_data=[1, 2, 3, 4, 5, 6],
                      detection_limits=[0, None],
                      censored_count=3)
    for missing_p in ['detection_limits', 'censored_count']:
        test_p = deepcopy(parameters)
        test_p[missing_p] = None
        with pytest.raises(ValueError) as e_info:
            RandomVariable(**test_p)
    assert RandomVariable(**parameters)

    # Multiple dimension
    parameters = dict(ID=1, dimension_tags='test',
                      raw_data=[[1, 0], [2, 1], [3, 1], [4, 0], [5, 1], [6, 0]],
                      detection_limits=[[0, None], [0, 1]],
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
                          theta=median, COV=beta ** 2.)
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
                          COV=np.ones((3, 3)) * beta ** 2.)
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
                          min_value=0.1, max_value=0.8)
        for missing_p in [['theta', ], ['COV', ], ['min_value', 'max_value']]:
            test_p = deepcopy(parameters)
            for par in missing_p:
                test_p[par] = None
            with pytest.raises(ValueError) as e_info:
                RandomVariable(**test_p)
        for missing_p in [[], ['min_value', ], ['max_value', ]]:
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
        for missing_p in [['theta', ], ['COV', ], ['min_value', 'max_value']]:
            test_p = deepcopy(parameters)
            for par in missing_p:
                test_p[par] = None
            with pytest.raises(ValueError) as e_info:
                RandomVariable(**test_p)
        for missing_p in [[], ['min_value', ], ['max_value', ]]:
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
    for missing_p in ['p_set', ]:
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
    
def test_random_variable_theta_attribute():
    """
    Test if the random variable returns the assigned median value and if it 
    returns an error message if no median has been assigned.
    """
    theta_ref = 0.5
    COV_ref = 0.4
    
    # single dimension
    RV = RandomVariable(ID=1, dimension_tags='test', distribution_kind='normal',
                        theta=theta_ref, COV=COV_ref)

    assert RV.theta == theta_ref
    
    # multiple dimensions
    RV = RandomVariable(ID=1, dimension_tags=['test1', 'test2'], 
                        distribution_kind='normal',
                        theta=np.ones(2) * theta_ref,
                        COV=np.ones((2,2))*COV_ref)

    assert_allclose(RV.theta,np.ones(2) * theta_ref,rtol=1e-6)
    
    # no median available
    RV = RandomVariable(ID=1, dimension_tags=['test1', 'test2'], 
                        raw_data=[1,2,3])
    with pytest.raises(ValueError) as e_info:
        print(RV.theta)


def test_random_variable_COV_attribute():
    """
    Test if the random variable returns the assigned covariance matrix and if 
    it returns an error message if no COV has been assigned.
    """
    theta_ref = 0.5
    COV_ref = 0.4

    # single dimension
    RV = RandomVariable(ID=1, dimension_tags='test', distribution_kind='normal',
                        theta=theta_ref, COV=COV_ref)

    assert RV.COV == COV_ref

    # multiple dimensions
    RV = RandomVariable(ID=1, dimension_tags=['test1', 'test2'],
                        distribution_kind='normal',
                        theta=np.ones(2) * theta_ref, 
                        COV=np.ones((2, 2)) * COV_ref)

    assert_allclose(RV.COV, np.ones((2,2)) * COV_ref, rtol=1e-6)

    # no median available
    RV = RandomVariable(ID=1, dimension_tags=['test1', 'test2'],
                        raw_data=[1, 2, 3])
    with pytest.raises(ValueError) as e_info:
        print(RV.COV)
