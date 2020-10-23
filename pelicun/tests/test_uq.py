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

from pelicun.tests.test_pelicun import assert_normal_distribution
from pelicun.uq import *

# Probably unused
'''
def test_RandomVariable_incorrect_censored_data_definition():
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
    parameters['detection_limits'] = [None, 0]
    assert RandomVariable(**parameters)

    # Multiple dimensions
    parameters = dict(ID=1, dimension_tags='test',
                      raw_data=[[1, 2, 3, 4, 5, 6], [0, 1, 1, 0, 1, 0]],
                      detection_limits=[None, [None, 1]],
                      censored_count=5)
    for missing_p in ['detection_limits', 'censored_count']:
        test_p = deepcopy(parameters)
        test_p[missing_p] = None
        with pytest.raises(ValueError) as e_info:
            RandomVariable(**test_p)
    assert RandomVariable(**parameters)
    parameters['detection_limits'] = [[None, 1], None]
    assert RandomVariable(**parameters)

def test_RandomVariable_incorrect_normal_lognormal_definition():
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
                          theta=median, COV=beta ** 2.,
                          truncation_limits=[None, None])
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

def test_RandomVariable_truncation_limit_conversion():
    """
    Test if the None values in the truncation limits are properly converted
    into infinite truncation limits during initialization.

    """
    # univariate
    parameters = dict(ID=1, dimension_tags='test',
                      distribution_kind='normal',
                      theta=0.5, COV=0.25 ** 2.,
                      truncation_limits=[None, None])
    for tl, target_tl in zip([[None, 1.], [-1., None], [-1., 1.]],
                             [[-np.inf, 1.], [-1., np.inf], [-1., 1.]]):
        parameters['truncation_limits'] = tl
        RV = RandomVariable(**parameters)
        for lim, target_lim in zip(RV._tr_limits_pre, target_tl):
            assert lim == target_lim

    # multivariate
    parameters = dict(ID=1, dimension_tags='test',
                      distribution_kind='normal',
                      theta=[0.5, 0.8], COV=np.ones((3, 3)) * 0.25 ** 2.,
                      truncation_limits=[None, None])
    for tl, target_tl in zip([[None, [1., 2.]],
                              [[-1., -0.5], None],
                              [[-1., -0.5], [1., 2.]],
                              [[-1., None], [None, 2.]]],
                             [[[-np.inf, -np.inf], [1., 2.]],
                              [[-1., -0.5], [np.inf, np.inf]],
                              [[-1., -0.5], [1., 2.]],
                              [[-1., -np.inf], [np.inf, 2.]]]):
        parameters['truncation_limits'] = tl
        RV = RandomVariable(**parameters)
        for lim_list, target_lim_list in zip(RV._tr_limits_pre, target_tl):
            for lim, target_lim in zip(lim_list, target_lim_list):
                assert lim == target_lim

def test_RandomVariable_theta_attribute():
    """
    Test if the random variable returns the assigned median value and if it
    returns an error message if no median has been assigned.
    """
    theta_ref = 0.5
    COV_ref = 0.4

    # single dimension
    RV = RandomVariable(ID=1, dimension_tags='test',
                        distribution_kind='lognormal',
                        theta=theta_ref, COV=COV_ref)

    assert RV.theta == theta_ref
    assert RV.mu == np.log(theta_ref)

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

def test_RandomVariable_COV_attribute():
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

def test_RandomVariable_simple_attributes():
    """
    Test if the attributes of the RV are properly exposed to the user.

    """
    # create a random variable with censored data
    censored_count = 3
    detection_limits = [0, 4]
    dimension_tags = ['A']
    RV = RandomVariable(ID=1, dimension_tags=dimension_tags,
                        raw_data=[1, 2, 3],
                        censored_count=censored_count,
                        detection_limits=detection_limits)

    assert RV.censored_count == censored_count
    assert RV.dimension_tags == dimension_tags
    assert RV.det_lower == detection_limits[0]
    assert RV.det_upper == detection_limits[1]
    assert RV.tr_lower_pre == None
    assert RV.tr_upper_pre == None
    assert_allclose(RV.detection_limits, detection_limits)

    # create a random variable with pre-defined truncated distribution
    truncation_limits = [0, 4]
    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='normal', theta=0.5, COV=0.25,
                        truncation_limits=truncation_limits)

    assert RV.det_lower == None
    assert RV.det_upper == None
    assert RV.tr_lower_pre == truncation_limits[0]
    assert RV.tr_upper_pre == truncation_limits[1]
    assert_allclose(RV.tr_limits_pre, truncation_limits)
    assert RV.tr_lower_post == None
    assert RV.tr_upper_post == None
    assert RV.tr_limits_post == None

    # create a bivariate distribution with post-truncation correlation
    truncation_limits = [[0, None], [4, None]]
    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='normal',
                        theta=[0.5, 0.5], COV=np.ones((2, 2)), corr_ref='post',
                        truncation_limits=truncation_limits)

    assert RV.det_lower == None
    assert RV.det_upper == None
    assert RV.tr_lower_pre == None
    assert RV.tr_upper_pre == None
    assert RV.tr_limits_pre == None
    assert_allclose(RV.tr_lower_post, [0., -np.inf])
    assert_allclose(RV.tr_upper_post, [4., np.inf])
    assert_allclose(RV.tr_limits_post, [[0., -np.inf], [4., np.inf]])

    # create a bivariate distribution with mixed pre and post truncation
    # correlation
    truncation_limits = [[0., -1.], [4., 5.]]
    RV = RandomVariable(ID=1, dimension_tags=['A'],
                        distribution_kind='normal',
                        theta=[0.5, 0.5], COV=np.ones((2, 2)),
                        corr_ref=['post', 'pre'],
                        truncation_limits=truncation_limits)

    assert RV.det_lower == None
    assert RV.det_upper == None
    assert_allclose(RV.tr_lower_pre, [-np.inf, -1])
    assert_allclose(RV.tr_upper_pre, [np.inf, 5])
    assert_allclose(RV.tr_limits_pre, [[-np.inf, -1], [np.inf, 5]])
    assert_allclose(RV.tr_lower_post, [0., -np.inf])
    assert_allclose(RV.tr_upper_post, [4., np.inf])
    assert_allclose(RV.tr_limits_post, [[0., -np.inf], [4., np.inf]])
    
def test_RandomVariable_sample_distribution_pre_and_post_truncation():
    """
    Test if the truncation limits are applied appropriately for pre- and
    post-truncation correlation settings.

    """
    # two extreme cases are tested: uncorrelated and perfectly correlated
    for r_i, rho in enumerate([0., 1.]):
        # multivariate case
        dims = 3
        ref_mean = np.ones(dims) * 2.
        ref_std = np.ones(dims) * 1.00
        ref_rho = np.ones((dims, dims)) * rho
        np.fill_diagonal(ref_rho, 1.0)
        ref_COV = np.outer(ref_std, ref_std) * ref_rho

        # prepare the truncation limits
        a = [-0.25, -np.inf, -1.5]
        b = [np.inf, 1.0, 1.5]
        tr_lower = (ref_mean + ref_std * a).tolist()
        tr_upper = (ref_mean + ref_std * b).tolist()

        # three types of corr_ref settings are tested:
        # 1) every variable is pre-truncated
        # 2) every variable is post-truncated
        # 3) mixed truncation
        for c_i, corr_ref in enumerate([
            ['pre', 'pre', 'pre'],
            ['post', 'post', 'post'],
            ['post', 'pre', 'post']]):

            # variable 1 is assumed to have lognormal distribution
            tr_lower[1] = np.exp(tr_lower[1])
            tr_upper[1] = np.exp(tr_upper[1])
            ref_mean[1] = np.exp(ref_mean[1])

            RV = RandomVariable(ID=1, dimension_tags=np.arange(dims),
                                distribution_kind=['normal', 'lognormal',
                                                   'normal'],
                                theta=ref_mean, COV=ref_COV,
                                corr_ref=corr_ref,
                                truncation_limits=[tr_lower, tr_upper])
            RVS = RandomVariableSubset(RV=RV, tags=1)

            samples = RV.sample_distribution(1000)

            # make sure that the samples attribute of the RV works as intended
            assert_allclose(samples, RV.samples)
            assert_allclose(samples[1], RVS.samples)

            # then check if resampling through RVS works well
            sample_size = 100
            RVS.sample_distribution(sample_size)
            old_diff = (RVS.samples - samples[1].iloc[:sample_size]).abs().sum()
            new_diff = (
                    RVS.samples - RV.samples[1].iloc[:sample_size]).abs().sum()

            assert old_diff > 0
            assert new_diff == 0

            # transfer the samples and reference values back to log space
            samples[1] = np.log(samples[1])
            ref_mean[1] = np.log(ref_mean[1])
            tr_lower[1] = np.log(max(np.nextafter(0, 1), tr_lower[1]))
            tr_upper[1] = np.log(tr_upper[1])

            if r_i == 0:
                # Means and standard deviations in the uncorrelated case shall
                # shall be equal to those from a corresponding univariate
                # truncated normal distribution
                ref_samples = np.asarray(list(map(
                    lambda x: truncnorm.rvs(a=x[0], b=x[1], loc=x[2],
                                            scale=x[3], size=1000),
                    zip(a, b, ref_mean, ref_std))))
                ref_mean_trunc = np.mean(ref_samples, axis=1)
                ref_std_trunc = np.std(ref_samples, axis=1)
                assert_allclose(np.mean(samples, axis=0), ref_mean_trunc,
                                atol=0.2)
                assert_allclose(np.std(samples, axis=0), ref_std_trunc,
                                atol=0.15)
                # zero correlations shall not be influenced by the truncation
                assert_allclose(np.corrcoef(samples, rowvar=False),
                                ref_rho, atol=0.15)

                # also make sure that the minimum and maximum of the samples
                # are within the truncation limits
                assert np.all(np.min(samples, axis=0) > tr_lower)
                assert np.all(np.max(samples, axis=0) < tr_upper)

            elif r_i == 1:
                # results under perfect correlation depend on the corr_ref
                # setting
                if c_i == 0:
                    # The pre-truncated setting will force every variable
                    # between the narrowest of the prescribed truncation
                    # limits. Their distribution will be an identical truncated
                    # normal (because their means and variances were identical
                    # originally).
                    ref_samples = truncnorm.rvs(a=max(a), b=min(b),
                                                loc=ref_mean[0],
                                                scale=ref_std[0], size=1000)
                    ref_mean_trunc = np.mean(ref_samples)
                    ref_std_trunc = np.std(ref_samples)
                    assert_allclose(np.mean(ref_samples, axis=0),
                                    ref_mean_trunc, atol=0.2)
                    assert_allclose(np.std(ref_samples, axis=0), ref_std_trunc,
                                    atol=0.15)

                    # all samples shall be within the stringest of the limits
                    assert np.all(np.min(samples, axis=0) > max(tr_lower))
                    assert np.all(np.max(samples, axis=0) < min(tr_upper))

                    # the perfect correlation shall be properly represented
                    assert_allclose(np.corrcoef(samples, rowvar=False), ref_rho,
                                    atol=0.15)

                elif c_i == 1:
                    # The post-truncated setting will let every component
                    # respect its own limits and the marginal distributions
                    # will follow the corresponding truncated normal
                    # distribution. However, the correlations are also
                    # preserved. Due to the truncation, the relationship
                    # between components is no longer linear. This leads to a
                    # correlation coefficient below 1, but higher than 0.85
                    # in this case.
                    ref_samples = np.asarray(list(map(
                        lambda x: truncnorm.rvs(a=x[0], b=x[1], loc=x[2],
                                                scale=x[3], size=1000),
                        zip(a, b, ref_mean, ref_std))))
                    ref_mean_trunc = np.mean(ref_samples, axis=1)
                    ref_std_trunc = np.std(ref_samples, axis=1)
                    assert_allclose(np.mean(samples, axis=0), ref_mean_trunc,
                                    atol=0.2)
                    assert_allclose(np.std(samples, axis=0), ref_std_trunc,
                                    atol=0.15)
                    # zero correlations shall not be influenced by the truncation
                    assert_allclose(np.corrcoef(samples, rowvar=False),
                                    ref_rho, atol=0.15)
                    # also make sure that the minimum and maximum of the samples
                    # are within the truncation limits
                    assert np.all(np.min(samples, axis=0) > tr_lower)
                    assert np.all(np.max(samples, axis=0) < tr_upper)
                elif c_i == 2:
                    # The mixed pre- and post-truncated setting first enforces
                    # the truncation of component 2 on every other component,
                    # and then transforms component 1 and 3 from normal to
                    # their truncated normal distribution.

                    # Component 2 will have a truncated normal distribution
                    # similar to c_i==0 case:
                    ref_samples = truncnorm.rvs(a=a[1], b=b[1],
                                                loc=ref_mean[1],
                                                scale=ref_std[1],
                                                size=1000)
                    assert np.mean(ref_samples) == pytest.approx(
                        np.mean(samples[1]),
                        abs=0.2)
                    assert np.std(ref_samples) == pytest.approx(
                        np.std(samples[1]),
                        abs=0.15)
                    # its samples shall be within its own truncation limits
                    assert np.min(samples[1]) > tr_lower[1]
                    assert np.max(samples[1]) < tr_upper[1]

                    # The other two components have their distribution
                    # truncated twice
                    ppf_limits = norm.cdf([a[1], b[1]], loc=0., scale=1.)
                    for comp in [0, 2]:
                        new_limits = truncnorm.ppf(ppf_limits, a=a[comp],
                                                   b=b[comp], loc=0., scale=1.)
                        ref_samples = truncnorm.rvs(a=new_limits[0],
                                                    b=new_limits[1],
                                                    loc=ref_mean[comp],
                                                    scale=ref_std[comp],
                                                    size=1000)

                        assert np.mean(ref_samples) == pytest.approx(
                            np.mean(samples[comp]), abs=0.2)
                        assert np.std(ref_samples) == pytest.approx(
                            np.std(samples[comp]), abs=0.15)

                        # samples shall be within the new_limits
                        assert np.min(samples[comp]) > \
                               ref_mean[comp] + ref_std[comp] * new_limits[0]
                        assert np.max(samples[comp]) < \
                               ref_mean[comp] + ref_std[comp] * new_limits[1]

def test_fitting_minimum_sample_size():
    """
    Test if the max. likelihood estimator function properly checks the number
    of samples available and raises an error when insufficient samples are
    available.

    """
    dims = 8
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    tr_lower = ref_mean + ref_std * (-2.)

    samples = tmvn_rvs(ref_mean, ref_COV, lower=tr_lower, size=3)

    with pytest.warns(UserWarning) as e_info:
        tmvn_MLE(np.transpose(samples), tr_lower=tr_lower)
        
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
                       size=1000)

    # censor the samples
    good_ones = np.all([samples > det_lower, samples < det_upper], axis=0)
    c_samples = samples[good_ones]
    c_count = 1000 - sum(good_ones)
    print(c_count)

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
    print('----------------------------')
    # warning about alpha being smaller than the specified limit
    with pytest.warns(UserWarning) as e_info:
        tmvn_MLE(c_samples, tr_lower=tr_lower, tr_upper=tr_upper,
                 censored_count=c_count, det_lower=det_lower,
                 det_upper=det_upper, alpha_lim=0.2)
                 
    def test_RandomVariable_fit_distribution_simple():
    """
    Test if the distribution fitting is performed appropriately for a simple
    normal distribution.

    """

    # univariate case
    # generate raw data
    raw_data = norm.rvs(loc=0.5, scale=0.25, size=10)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, var_ref = tmvn_MLE(np.transpose(raw_data))

    #create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=['A'], raw_data=raw_data)
    mu, var = RV.fit_distribution('normal')

    # compare results
    assert mu == pytest.approx(mu_ref, rel=1e-2)
    assert var == pytest.approx(var_ref, rel=1e-2)

    # multivariate case
    # generate raw data
    dims = 6
    in_mean = np.arange(dims, dtype=np.float64)
    in_std = np.ones(dims) * 0.25
    in_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(in_rho, 1.0)
    in_COV = np.outer(in_std, in_std) * in_rho

    raw_data = multivariate_normal.rvs(mean=in_mean, cov=in_COV, size=25)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, COV_ref = tmvn_MLE(np.transpose(raw_data))

    # create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=[np.arange(dims)],
                        raw_data=np.transpose(raw_data))
    mu, COV = RV.fit_distribution('normal')

    # compare results
    assert_allclose(mu, mu_ref, rtol=1e-2, atol=1e-2)
    assert_allclose(COV, COV_ref, rtol=1e-2, atol=1e-2)

def test_RandomVariable_fit_distribution_truncated():
    """
    Test if the distribution fitting is performed appropriately for a truncated
    normal distribution.

    """
    # define the truncation limits
    tr_lower = -4.
    tr_upper = 2.5

    # univariate case
    # generate raw data using the (previously tested) sampler
    raw_data = tmvn_rvs(mu=0.5, COV=0.25, lower=tr_lower, upper=tr_upper,
                        size=10)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, var_ref = tmvn_MLE(np.transpose(raw_data),
                               tr_lower=tr_lower, tr_upper=tr_upper)

    # create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=['A'], raw_data=raw_data)
    mu, var = RV.fit_distribution('normal',
                                  truncation_limits=[tr_lower, tr_upper])

    # compare results
    assert mu == pytest.approx(mu_ref, rel=1e-2)
    assert var == pytest.approx(var_ref, rel=1e-2)

    # multivariate case
    # generate raw data
    dims = 6
    in_mean = np.arange(dims, dtype=np.float64)
    in_std = np.ones(dims) * 0.25
    in_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(in_rho, 1.0)
    in_COV = np.outer(in_std, in_std) * in_rho

    tr_lower = in_mean + in_std * tr_lower
    tr_upper = in_mean + in_std * tr_upper

    # generate raw data using the (previously tested) sampler
    raw_data = tmvn_rvs(mu=in_mean, COV=in_COV, lower=tr_lower, upper=tr_upper,
                        size=100)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, COV_ref = tmvn_MLE(np.transpose(raw_data),
                               tr_lower=tr_lower, tr_upper=tr_upper)

    # create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=[np.arange(dims)],
                        raw_data=np.transpose(raw_data))
    mu, COV = RV.fit_distribution('normal',
                                  truncation_limits=[tr_lower, tr_upper])

    # compare results
    assert_allclose(mu, mu_ref, atol=0.05)
    assert_allclose(COV, COV_ref, atol=0.05)

def test_RandomVariable_fit_distribution_truncated_and_censored():
    """
    Test if the distribution fitting is performed appropriately for a truncated
    and censored normal distribution.

    """
    # define the truncation and detection limits
    tr_lower = -4.
    tr_upper = 2.5
    det_lower = tr_lower
    det_upper = 2.0

    # univariate case
    # generate raw data using the (previously tested) sampler
    raw_data = tmvn_rvs(mu=0.5, COV=0.25, lower=tr_lower, upper=tr_upper,
                        size=100)

    # censor the samples
    good_ones = raw_data < det_upper
    c_samples = raw_data[good_ones]
    c_count = 100 - sum(good_ones)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, var_ref = tmvn_MLE(np.transpose(raw_data),
                               tr_lower=tr_lower, tr_upper=tr_upper,
                               censored_count=c_count,
                               det_lower=det_lower, det_upper=det_upper)

    # create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=['A'], raw_data=raw_data,
                        detection_limits=[det_lower, det_upper],
                        censored_count=c_count)
    mu, var = RV.fit_distribution('normal',
                                  truncation_limits=[tr_lower, tr_upper])

    # compare results
    assert mu == pytest.approx(mu_ref, rel=1e-2)
    assert var == pytest.approx(var_ref, rel=1e-2)

    # multivariate case
    # generate raw data
    dims = 3
    in_mean = np.arange(dims, dtype=np.float64)
    in_std = np.ones(dims) * 0.25
    in_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(in_rho, 1.0)
    in_COV = np.outer(in_std, in_std) * in_rho

    tr_lower = in_mean + in_std * tr_lower
    tr_upper = in_mean + in_std * tr_upper
    det_lower = in_mean + in_std * det_lower
    det_upper = in_mean + in_std * det_upper

    # generate raw data using the (previously tested) sampler
    raw_data = tmvn_rvs(mu=in_mean, COV=in_COV, lower=tr_lower, upper=tr_upper,
                        size=100)

    # censor the samples
    good_ones = np.all([raw_data > det_lower, raw_data < det_upper], axis=0)
    good_ones = np.all(good_ones, axis=1)
    c_samples = np.transpose(raw_data[good_ones])
    c_count = 100 - sum(good_ones)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, COV_ref = tmvn_MLE(np.transpose(raw_data),
                               tr_lower=tr_lower, tr_upper=tr_upper,
                               censored_count=c_count,
                               det_lower=det_lower, det_upper=det_upper)

    # create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=[np.arange(dims)],
                        raw_data=np.transpose(raw_data),
                        detection_limits=[det_lower, det_upper],
                        censored_count=c_count)
    mu, COV = RV.fit_distribution('normal',
                                  truncation_limits=[tr_lower, tr_upper])

    # compare results
    assert_allclose(mu, mu_ref, atol=0.03, rtol=0.03)
    assert_allclose(COV, COV_ref, atol=0.03, rtol=0.03)


def test_RandomVariable_fit_distribution_log_and_linear():
    """
    Test if the distribution fitting is performed appropriately when the data
    is lognormal in some dimensions and normal in others.

    """
    # generate raw data
    dims = 3
    in_mean = np.arange(dims, dtype=np.float64)
    in_std = np.ones(dims) * 0.25
    in_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(in_rho, 1.0)
    in_COV = np.outer(in_std, in_std) * in_rho

    raw_data = multivariate_normal.rvs(mean=in_mean, cov=in_COV, size=25)

    # create a lognormal distribution in the second variable
    RT = np.transpose(deepcopy(raw_data))
    RT[1] = np.exp(RT[1])
    raw_dataL = np.transpose(RT)

    # reference data is directly from the (previously tested) ML estimator
    mu_ref, COV_ref = tmvn_MLE(np.transpose(raw_data))
    mu_ref[1] = np.exp(mu_ref[1])

    # create a random variable and perform fitting
    RV = RandomVariable(ID=1, dimension_tags=[np.arange(dims)],
                        raw_data=np.transpose(raw_dataL))
    mu, COV = RV.fit_distribution(['normal', 'lognormal', 'normal'])

    # compare results
    assert_allclose(mu, mu_ref, rtol=1e-2, atol=1e-2)
    assert_allclose(COV, COV_ref, rtol=1e-2, atol=1e-2)

def test_RandomVariable_orthotope_density():
    """
    Test if the orthotope density function provides accurate estimates of the
    probability densities within several different hyperrectangles for TMVN
    distributions.

    """
    # multivariate, uncorrelated case
    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 1.00
    ref_rho = np.ones((dims, dims)) * 0.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    # prepare the truncation limits
    tr_lower = (ref_mean + ref_std * (-2.)).tolist()
    tr_upper = (ref_mean + ref_std * 2.).tolist()

    # variable 1 is assumed to have lognormal distribution
    ref_mean[1] = np.exp(ref_mean[1])
    tr_lower[1] = np.exp(tr_lower[1])
    tr_upper[1] = np.exp(tr_upper[1])

    # variable 2 is assumed to have no truncation
    tr_lower[2] = None
    tr_upper[2] = None

    RV = RandomVariable(ID=1, dimension_tags=np.arange(dims),
                        distribution_kind=['normal', 'lognormal', 'normal'],
                        theta=ref_mean, COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])

    # Test if the full (truncated) space corresponds to a density of 1.0
    assert RV.orthotope_density()[0] == pytest.approx(1.0)

    # Test if adding limits outside the truncated area influence the results
    test_alpha = RV.orthotope_density(lower=[-3., 0., None],
                                      upper=[4., 25., None])[0]
    assert test_alpha == pytest.approx(1.0)

    # Test if limiting the third variable at its mean reduces the density to
    # 0.5
    test_alpha = RV.orthotope_density(lower=[None, None, 2.])[0]
    assert test_alpha == pytest.approx(0.5)

    # Test if limiting variables 1-2 to only +-1-sigma reduces the density
    # appropriately
    test_alpha = RV.orthotope_density(lower=[-1., np.exp(0.), None],
                                      upper=[1., np.exp(2.), None])[0]
    assert test_alpha == pytest.approx(0.5115579696)

    # Now let us introduce perfect correlation
    ref_rho = np.ones((dims, dims)) * 1.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    RV = RandomVariable(ID=1, dimension_tags=np.arange(dims),
                        distribution_kind=['normal', 'lognormal', 'normal'],
                        theta=ref_mean, COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])

    # Test if the full (truncated) space corresponds to a density of 1.0
    assert RV.orthotope_density()[0] == pytest.approx(1.0)

    # Test if limiting the third variable outside 2-sigma influence the results
    test_alpha = RV.orthotope_density(upper=[None, None, 4.])[0]
    assert test_alpha == pytest.approx(1.0)

    # Test if limiting the third variable at its mean reduces the density to
    # 0.5
    test_alpha = RV.orthotope_density(lower=[None, None, 2.])[0]
    assert test_alpha == pytest.approx(0.5)

    # Test if limiting variables 1-2 to only +-1-sigma reduces the density
    # appropriately
    test_alpha = RV.orthotope_density(lower=[-1., np.exp(0.), None],
                                      upper=[1., np.exp(2.), None])[0]
    assert test_alpha == pytest.approx(0.71523280799)

    # The next test uses a mix of pre-truncation and post-truncation
    # correlations
    RV = RandomVariable(ID=1, dimension_tags=np.arange(dims),
                        distribution_kind=['normal', 'lognormal', 'normal'],
                        corr_ref=['pre', 'post', 'pre'],
                        theta=ref_mean, COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])

    # Test if the full (truncated) space corresponds to a density of 1.
    assert RV.orthotope_density()[0] == pytest.approx(1.0)

    # Test if limiting the second variable outside 2-sigma in the pre-truncated
    # distribution influences the results
    test_alpha = RV.orthotope_density(upper=[None, np.exp(3.), None])[0]
    assert test_alpha == pytest.approx(1.0)

    # Test if limiting the second variable at '2-sigma' quantiles in the
    # marginal truncated distribution influences the results.
    # It shouldn't because the perfect correlation with 2-sigma limited other
    # variables already poses such limits on variable 2. Note that these
    # quantiles are more strict than the 2-sigma 'pre' limits tested above,
    # and they would lead to sub 1.0 results if all correlations were
    # pre-truncation.
    limits = truncnorm.ppf(norm.cdf([-2., 2.]),
                           loc=np.log(ref_mean[1]), scale=ref_std[1],
                           a=-2., b=2.)
    test_alpha = RV.orthotope_density(lower=[None, np.exp(limits[0]), None],
                                      upper=[None, np.exp(limits[1]), None])[0]
    assert test_alpha == pytest.approx(1.0)

    # Test if limiting variable 2 to only +- 1-sigma quantiles in the marginal
    # truncated distribution reduces the density appropriately. These limits
    # are more strict than the 1 sigma 'pre' limits would be because they are
    # defined in the truncated distribution.
    limits = truncnorm.ppf(norm.cdf([-1., 1.]),
                           loc=np.log(ref_mean[1]), scale=ref_std[1],
                           a=-2., b=2.)
    test_alpha = RV.orthotope_density(lower=[None, np.exp(limits[0]), None],
                                      upper=[None, np.exp(limits[1]), None])[0]
    ref_alpha = truncnorm.cdf([-1., 1.], loc=0., scale=1., a=-2., b=2.)
    ref_alpha = ref_alpha[1] - ref_alpha[0]
    assert test_alpha == pytest.approx(ref_alpha)

    # test if limiting the second variable at its mean reduces the density to
    # 0.5
    test_alpha = RV.orthotope_density(lower=[None, np.exp(1.), None])[0]
    assert test_alpha == pytest.approx(0.5)

    # Finally, test if the function works well for a non-truncated MVN
    # distribution with uncorrelated variables
    ref_rho = np.ones((dims, dims)) * 0.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    RV = RandomVariable(ID=1, dimension_tags=np.arange(dims),
                        distribution_kind=['normal', 'lognormal', 'normal'],
                        theta=ref_mean, COV=ref_COV)

    # Test if the full (truncated) space corresponds to a density of 1.0
    assert RV.orthotope_density()[0] == pytest.approx(1.0)

    # Test if limiting the third variable at its mean reduces the density to
    # 0.5
    test_alpha = RV.orthotope_density(lower=[None, None, 2.])[0]
    assert test_alpha == pytest.approx(0.5)

def test_RandomVariableSubset_orthotope_density():
    """
    Test if the orthotope density function provides accurate estimates of the
    probability densities within several different hyperrectangles for TMVN
    distributions. Consider that the RVS does not have access to every
    dimension in the RV, yet the limits assigned to those dimensions can
    influence the results.

    """
    # multivariate case, uncorrelated
    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 1.00
    ref_rho = np.ones((dims, dims)) * 0.
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    # prepare the truncation limits
    tr_lower = (ref_mean + ref_std * (-2.)).tolist()
    tr_upper = (ref_mean + ref_std * 2.).tolist()

    # variable 1 is assumed to have lognormal distribution
    ref_mean[1] = np.exp(ref_mean[1])
    tr_lower[1] = np.exp(tr_lower[1])
    tr_upper[1] = np.exp(tr_upper[1])

    # variable 2 is assumed to have no truncation
    tr_lower[2] = None
    tr_upper[2] = None

    RV = RandomVariable(ID=1, dimension_tags=['A', 'B', 'C'],
                        distribution_kind=['normal', 'lognormal', 'normal'],
                        theta=ref_mean, COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])
    RVS = RandomVariableSubset(RV=RV, tags=['B', 'A'])

    # Test if the full (truncated) space corresponds to a density of 1.0
    assert RVS.orthotope_density()[0] == pytest.approx(1.)

    # Test if limiting variable B at its mean reduces the density to 0.5
    test_alpha = RVS.orthotope_density(lower=[ref_mean[1], None])[0]
    assert test_alpha == pytest.approx(0.5)

    # Do the same test for variable A and an upper limit
    test_alpha = RVS.orthotope_density(lower=[None, ref_mean[0]])[0]
    assert test_alpha == pytest.approx(0.5)

    # Now check how a correlated variable in RV affects the densities in the
    # RVS
    ref_COV[2, 0] = 1.
    ref_COV[0, 2] = 1.
    # A and B are independent, and A is perfectly correlated with C

    # truncation limits are only introduced for C
    tr_lower = [None, None, 1.]
    tr_upper = [None, None, 3.]

    RV = RandomVariable(ID=1, dimension_tags=['A', 'B', 'C'],
                        distribution_kind=['normal', 'lognormal', 'normal'],
                        theta=ref_mean, COV=ref_COV,
                        truncation_limits=[tr_lower, tr_upper])
    RVS = RandomVariableSubset(RV=RV, tags=['B', 'A'])

    # Test if the full (truncated) space corresponds to a density of 1.0
    assert RVS.orthotope_density()[0] == pytest.approx(1.)

    # Test if limiting variable A at one sigma on both sides reduces the density - it shouldn't
    test_alpha = RVS.orthotope_density(lower=[None, -1.],
                                       upper=[None, 1.])[0]
    assert test_alpha == pytest.approx(1.0)

    # Test if limiting variable B at one sigma on both sides reduces the density - it should
    test_alpha = RVS.orthotope_density(lower=[np.exp(0.), None],
                                       upper=[np.exp(2.), None])[0]
    assert test_alpha == pytest.approx(0.682689492)



'''

# ------------------------------------------------------------------------------
# ORTHOTOPE DENSITY
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
        ref_mean = np.arange(dims, dtype=np.float64)
        ref_std = np.arange(1, dims + 1, dtype=np.float64)
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
        ref_mean = np.arange(dims, dtype=np.float64)
        ref_std = np.arange(1, dims + 1, dtype=np.float64)
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
# Random_Variable objects
# ------------------------------------------------------------------------------
def test_RandomVariable_incorrect_none_defined():
    """
    Test if the random variable object raises an error when the distribution
    is not empirical and no parameters are provided.
    """

    with pytest.raises(ValueError) as e_info:
        RandomVariable(name='A', distribution='normal')

def test_RandomVariable_incorrect_multinomial_definition():
    """
    Test if the random variable object raises an error when a multinomial
    distribution is defined with incorrect parameters, and test
    that it does not raise an error when the right parameters are provided.
    """

    p_values = [0.5, 0.2, 0.1, 0.2]
    # correct parameters
    RandomVariable(name='A', distribution='multinomial',
                   theta=[0.5, 0.2, 0.1, 0.2])

    # sum(p) less than 1.0 -> should be automatically corrected
    RandomVariable(name='A', distribution='multinomial',
                   theta=[0.5, 0.2, 0.1, 0.1])

    # sum(p) more than 1.0 -> should raise an error
    with pytest.raises(ValueError) as e_info:
        RandomVariable(name='A', distribution='multinomial',
                       theta=[0.5, 0.2, 0.1, 0.3])

# ------------------------------------------------------------------------------
# SAMPLING
# ------------------------------------------------------------------------------
def test_sampling_tr_alpha_error():
    """
    Test if the function raises an error when the probability density that the
    truncation limits define is not sufficiently accurate for further use.

    """

    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[0.5, 0.25],
                                 truncation_limits = [-10.0, -9.9]))

    with pytest.raises(ValueError) as e_info:
        RV_reg.generate_samples(sample_size=10, seed=1)

def test_sampling_non_truncated():
    """
    Test if the sampling method returns appropriate samples for a non-truncated
    univariate and multivariate normal distribution.

    """

    # univariate case
    ref_mean = 0.5
    ref_std = 0.25

    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std]))

    def sampling_function(sample_size):

        RV_reg.generate_samples(sample_size=sample_size)

        return RV_reg.RV_samples['A']

    assert assert_normal_distribution(sampling_function, ref_mean, ref_std**2.)

    # bivariate case, various correlation coefficients
    rho_list = [-1., -0.5, 0., 0.5, 1.]
    for rho in rho_list:
        ref_mean = [0.5, 1.5]
        ref_std = [0.25, 1.0]
        ref_rho = np.asarray([[1.0, rho],[rho, 1.0]])
        ref_COV = np.outer(ref_std, ref_std) * ref_rho

        RV_reg = RandomVariableRegistry()

        for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
            RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                         theta=[mu, std]))

        RV_reg.add_RV_set(
            RandomVariableSet('A',
                              [RV_reg.RV[rv] for rv in range(len(ref_mean))],
                              ref_rho))

        def sampling_function(sample_size):

            RV_reg.generate_samples(sample_size=sample_size)

            return pd.DataFrame(RV_reg.RV_samples).values

        assert_normal_distribution(sampling_function, ref_mean, ref_COV)

    # multi-dimensional case
    dims = 5
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.arange(1,dims+1, dtype=np.float64)
    ref_rho = np.ones((dims,dims))*0.3
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std,ref_std) * ref_rho

    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(len(ref_mean))],
                          ref_rho))

    def sampling_function(sample_size):

        RV_reg.generate_samples(sample_size=sample_size)

        return pd.DataFrame(RV_reg.RV_samples).values

    assert_normal_distribution(sampling_function, ref_mean, ref_COV)

def test_sampling_truncated_wide_limits():
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

    # truncation both ways
    RV_reg = RandomVariableRegistry()
    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std],
                                 truncation_limits = [lower, upper]))

    def sampling_function(sample_size):
        RV_reg.generate_samples(sample_size=sample_size)

        return RV_reg.RV_samples['A']

    assert assert_normal_distribution(sampling_function, ref_mean,
                                      ref_std ** 2.)
    # upper truncation
    RV_reg = RandomVariableRegistry()
    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std],
                                 truncation_limits=[None, upper]))

    def sampling_function(sample_size):
        RV_reg.generate_samples(sample_size=sample_size)

        return RV_reg.RV_samples['A']

    assert assert_normal_distribution(sampling_function, ref_mean,
                                      ref_std ** 2.)

    # lower truncation
    RV_reg = RandomVariableRegistry()
    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std],
                                 truncation_limits=[lower, None]))

    def sampling_function(sample_size):
        RV_reg.generate_samples(sample_size=sample_size)

        return RV_reg.RV_samples['A']

    assert assert_normal_distribution(sampling_function, ref_mean,
                                      ref_std ** 2.)

    # bivariate case, various correlation coefficients
    rho_list = [-1., -0.5, 0., 0.5, 1.]
    lower = np.ones(2) * lower
    upper = np.ones(2) * upper
    for rho in rho_list:
        ref_mean = [0.5, 1.5]
        ref_std = [0.25, 1.0]
        ref_rho = np.asarray([[1.0, rho], [rho, 1.0]])
        ref_COV = np.outer(ref_std, ref_std) * ref_rho

        RV_reg = RandomVariableRegistry()

        for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
            RV_reg.add_RV(
                RandomVariable(name=i, distribution='normal',
                               theta=[mu, std],
                               truncation_limits = [lower[i], upper[i]]))

        RV_reg.add_RV_set(
            RandomVariableSet('A',
                              [RV_reg.RV[rv] for rv in range(len(ref_mean))],
                              ref_rho))

        def sampling_function(sample_size):

            RV_reg.generate_samples(sample_size=sample_size)

            return pd.DataFrame(RV_reg.RV_samples).values

        assert_normal_distribution(sampling_function, ref_mean, ref_COV)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.3
    np.fill_diagonal(ref_rho, 1.0)
    ref_COV = np.outer(ref_std, ref_std) * ref_rho

    lower = np.ones(dims) * lower[0]
    upper = np.ones(dims) * upper[0]

    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std],
                                     truncation_limits = [lower[i], upper[i]]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(len(ref_mean))],
                          ref_rho))

    def sampling_function(sample_size):

        RV_reg.generate_samples(sample_size=sample_size)

        return pd.DataFrame(RV_reg.RV_samples).values

    assert_normal_distribution(sampling_function, ref_mean, ref_COV)

def test_sampling_truncated_narrow_limits():
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

    RV_reg = RandomVariableRegistry()
    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std],
                                 truncation_limits=[lower, upper]))

    RV_reg.generate_samples(sample_size=1000)

    samples = RV_reg.RV_samples['A']

    sample_min = np.min(samples)
    sample_max = np.max(samples)

    assert sample_min > lower
    assert sample_max < upper

    assert sample_min == pytest.approx(lower, abs=0.01)
    assert sample_max == pytest.approx(upper, abs=0.01)

    # multi-dimensional case
    dims = 2
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.3
    np.fill_diagonal(ref_rho, 1.0)

    lower = ref_mean - ref_std * 2.5
    upper = ref_mean + ref_std * 1.5

    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std],
                                     truncation_limits=[lower[i], upper[i]]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(len(ref_mean))],
                          ref_rho))

    RV_reg.generate_samples(sample_size=1000)

    samples = pd.DataFrame(RV_reg.RV_samples).values

    sample_min = np.amin(samples, axis=0)
    sample_max = np.amax(samples, axis=0)

    assert np.all(sample_min > lower)
    assert np.all(sample_max < upper)

    assert_allclose(sample_min, lower, atol=0.1)
    assert_allclose(sample_max, upper, atol=0.1)

def test_RandomVariable_sample_distribution_mixed_normal():
    """
    Test if the distribution is sampled appropriately for a correlated mixture
    of normal and lognormal variables. Note that we already tested the sampling
    algorithm itself earlier, so we will not do a thorough verification of
    the samples, but rather check for errors in the inputs that would
    typically lead to significant mistakes in the results.

    """

    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 1.00
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)

    # prepare the truncation limits - note that these are wide limits and
    # barely affect the distribution
    lower = (ref_mean - ref_std * 10.).tolist()
    upper = ref_mean + ref_std * 10.

    # variable 1 is assumed to have lognormal distribution with no lower
    # truncation
    rv_mean = ref_mean.copy()
    rv_mean[1] = np.exp(rv_mean[1])
    lower[1] = None
    upper[1] = np.exp(upper[1])

    dist_list = ['normal', 'lognormal', 'normal']

    RV_reg = RandomVariableRegistry()

    for i, (mu, std, dist) in enumerate(zip(rv_mean, ref_std, dist_list)):
        RV_reg.add_RV(RandomVariable(name=i, distribution=dist,
                                     theta=[mu, std],
                                     truncation_limits=[lower[i],
                                                        upper[i]]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(len(ref_mean))],
                          ref_rho))

    RV_reg.generate_samples(sample_size=1000)

    samples = pd.DataFrame(RV_reg.RV_samples).values.T

    # make sure that retrieving samples of an individual RV works as intended
    assert_allclose(samples[1], RV_reg.RV_samples[1])

    # convert the lognormal samples to log space for the checks
    samples[1] = np.log(samples[1])

    assert_allclose(np.mean(samples, axis=1), ref_mean, atol=0.01)

    ref_COV = np.outer(ref_std, ref_std) * ref_rho
    assert_allclose(np.cov(samples), ref_COV, atol=0.10)

def test_RandomVariable_sample_distribution_multinomial():
    """
    Test if the distribution is sampled appropriately for a multinomial
    variable. Also test that getting values for an individual RV that is part
    of a correlated RV_set works appropriately."

    """
    # first test with an incomplete p_ref
    p_ref = [0.1, 0.3, 0.5]
    dims = 3

    RV_reg= RandomVariableRegistry()

    for i in range(dims):
        RV_reg.add_RV(RandomVariable(name=i, distribution='multinomial',
                                     theta=p_ref))

    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)], ref_rho))

    RV_reg.generate_samples(sample_size=10000)

    samples = pd.DataFrame(RV_reg.RV_samples).values.T

    p_ref[-1] = 1. - np.sum(p_ref[:-1])

    h_bins = np.arange(len(p_ref) + 1) - 0.5
    p_test = np.histogram(samples[1], bins=h_bins, density=True)[0]

    assert_allclose(p_test, p_ref, atol=0.05)

    # also make sure that individual RV's samples are returned appropriately
    p_test_RV = np.histogram(RV_reg.RV_samples[1], bins=h_bins, density=True)[0]
    assert_allclose(p_test_RV, p_ref, atol=0.05)

    assert_allclose(samples[1], RV_reg.RV_samples[1])

    # and the prescribed correlation is applied
    # note that the correlation between the samples is not going to be identical
    # to the correlation used to generate the underlying uniform distribution
    rho_target = [[1.0, 0.383, 0.383], [0.383, 1.0, 0.383], [0.383, 0.383, 1.0]]
    assert_allclose(np.corrcoef(samples), rho_target, atol=0.05)

    # finally, check the original sampling with the complete p_ref
    RV_reg = RandomVariableRegistry()

    for i in range(dims):
        RV_reg.add_RV(RandomVariable(name=i, distribution='multinomial',
                                     theta=p_ref))

    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)
    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)], ref_rho))

    RV_reg.generate_samples(sample_size=10000)

    samples = pd.DataFrame(RV_reg.RV_samples).values.T

    p_test = np.histogram(samples[1], bins=h_bins, density=True)[0]

    assert_allclose(p_test, p_ref, atol=0.05)

# ------------------------------------------------------------------------------
# FITTING
# ------------------------------------------------------------------------------
def test_fitting_baseline():
    """
    Test if the max. likelihood estimates of a (multivariate) normal
    distribution are sufficiently accurate in the baseline case with no
    truncation and no censoring.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25

    # generate samples
    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std]))

    RV_reg.generate_samples(sample_size=100)

    samples = RV_reg.RV_samples['A']

    # estimate the parameters of the distribution
    mu, std = fit_distribution(samples, 'normal')[0][0]

    assert ref_mean == pytest.approx(mu, abs=0.01)
    assert ref_std == pytest.approx(std, rel=0.05)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.5
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)

    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)],
                          ref_rho))

    RV_reg.generate_samples(sample_size=100)

    samples = pd.DataFrame(RV_reg.RV_samples).values.T

    # estimate the parameters of the distribution
    test_theta, test_rho = fit_distribution(samples, 'normal')

    test_mu, test_std = test_theta.T

    assert_allclose(test_mu, ref_mean, atol=0.01)
    assert_allclose(test_std, ref_std, rtol=0.1)
    assert_allclose(test_rho, ref_rho, atol=0.2)

def test_fitting_censored():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in cases with censored data.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    c_lower = 0.35
    c_upper = 1.25
    sample_count = 100

    # generate censored samples
    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std]))

    RV_reg.generate_samples(sample_size=sample_count)

    samples = RV_reg.RV_samples['A']

    # censor the samples
    good_ones = np.all([samples>c_lower, samples<c_upper],axis=0)
    c_samples = samples[good_ones]
    c_count = sample_count - sum(good_ones)

    # estimate the parameters of the distribution
    test_theta, __ = fit_distribution(c_samples, 'normal',
                                      censored_count=c_count,
                                      detection_limits=[c_lower, c_upper])

    test_mu, test_std = test_theta[0]

    assert ref_mean == pytest.approx(test_mu, abs=0.05)
    assert ref_std == pytest.approx(test_std, rel=0.05)

    # multi-dimensional case

    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)

    c_lower = ref_mean - 1.0 * ref_std
    c_upper = ref_mean + 8.5 * ref_std

    c_lower[2] = -np.inf
    c_upper[0] = np.inf

    sample_count = 1000

    # generate samples
    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)],
                          ref_rho))

    RV_reg.generate_samples(sample_size=sample_count)

    samples = pd.DataFrame(RV_reg.RV_samples).values

    # censor the samples
    good_ones = np.all([samples > c_lower, samples < c_upper], axis=0)
    good_ones = np.all(good_ones, axis=1)
    c_samples = samples[good_ones]
    c_count = sample_count - sum(good_ones)
    det_lims = np.array([c_lower, c_upper]).T

    test_theta, test_rho = fit_distribution(c_samples.T, 'normal',
                                            censored_count=c_count,
                                            detection_limits=det_lims)

    test_mu, test_std = test_theta.T

    assert_allclose(test_mu, ref_mean, atol=0.1)
    assert_allclose(test_std , ref_std, rtol=0.25)
    assert_allclose(test_rho, ref_rho, atol=0.3)

def test_fitting_truncated():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in cases with truncation and uncensored data.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    tr_lower = 0.35
    tr_upper = 1.25

    # generate samples of a TMVN distribution
    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std],
                                 truncation_limits=[tr_lower, tr_upper]))

    RV_reg.generate_samples(sample_size=100)

    samples = RV_reg.RV_samples['A']

    # estimate the parameters of the distribution
    test_theta, __ = fit_distribution(samples, 'normal',
                                      truncation_limits=[tr_lower, tr_upper])

    test_mu, test_std = test_theta[0]

    assert ref_mean == pytest.approx(test_mu, abs=0.05)
    assert ref_std == pytest.approx(test_std, rel=0.05)

    # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)

    tr_lower = ref_mean - 0.25 * ref_std
    tr_upper = ref_mean + 2.0 * ref_std

    tr_lower[2] = -np.inf
    tr_upper[0] = np.inf

    # generate samples
    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std],
                                     truncation_limits=[tr_lower[i],
                                                        tr_upper[i]]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)],
                          ref_rho))

    RV_reg.generate_samples(sample_size=1000)

    samples = pd.DataFrame(RV_reg.RV_samples).values.T

    # estimate the parameters of the distribution
    tr_lims = np.array([tr_lower, tr_upper]).T
    test_theta, test_rho = fit_distribution(samples, 'normal',
                                            truncation_limits=tr_lims)

    test_mu, test_std = test_theta.T

    assert_allclose(test_mu, ref_mean, atol=0.05)
    assert_allclose(test_std, ref_std, rtol=0.1)
    assert_allclose(test_rho, ref_rho, atol=0.1)

def test_fitting_truncated_and_censored():
    """
    Test if the max. likelihood estimates of a multivariate normal distribution
    are sufficiently accurate in cases with truncation and censored data.

    """
    # univariate case
    ref_mean = 0.5
    ref_std = 0.25
    tr_lower = 0.35
    tr_upper = 2.5
    det_upper = 1.25
    det_lower = tr_lower

    # generate samples of a TMVN distribution
    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='normal',
                                 theta=[ref_mean, ref_std],
                                 truncation_limits=[tr_lower, tr_upper]))

    RV_reg.generate_samples(sample_size=100)

    samples = RV_reg.RV_samples['A']

    # censor the samples
    good_ones = samples < det_upper
    c_samples = samples[good_ones]
    c_count = 100 - sum(good_ones)

    # estimate the parameters of the distribution
    test_theta, __ = fit_distribution(c_samples, 'normal',
                                      censored_count=c_count,
                                      detection_limits=[det_lower, det_upper],
                                      truncation_limits=[tr_lower, tr_upper])

    test_mu, test_std = test_theta[0]

    assert ref_mean == pytest.approx(test_mu, abs=0.05)
    assert ref_std == pytest.approx(test_std, rel=0.05)

    # # multi-dimensional case
    dims = 3
    ref_mean = np.arange(dims, dtype=np.float64)
    ref_std = np.ones(dims) * 0.25
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)

    tr_lower = ref_mean - 4.5 * ref_std
    tr_upper = ref_mean + 2.5 * ref_std
    tr_lower[2] = -np.inf
    tr_upper[0] = np.inf

    det_lower = ref_mean - 1. * ref_std
    det_lower[2] = -np.inf
    det_upper = tr_upper

    # generate samples
    RV_reg = RandomVariableRegistry()

    for i, (mu, std) in enumerate(zip(ref_mean, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='normal',
                                     theta=[mu, std],
                                     truncation_limits=[tr_lower[i],
                                                        tr_upper[i]]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)],
                          ref_rho))

    RV_reg.generate_samples(sample_size=1000)

    samples = pd.DataFrame(RV_reg.RV_samples).values

    # censor the samples
    good_ones = np.all([samples > det_lower, samples < det_upper], axis=0)
    good_ones = np.all(good_ones, axis=1)
    c_samples = samples[good_ones]
    c_count = 1000 - sum(good_ones)

    # estimate the parameters of the distribution
    det_lims = np.array([det_lower, det_upper]).T
    tr_lims = np.array([tr_lower, tr_upper]).T

    test_theta, test_rho = fit_distribution(
        c_samples.T, 'normal', censored_count=c_count,
        detection_limits=det_lims, truncation_limits=tr_lims)

    test_mu, test_std = test_theta.T

    assert_allclose(test_mu, ref_mean, atol=0.1)
    assert_allclose(test_std, ref_std, rtol=0.25)
    assert_allclose(test_rho, ref_rho, atol=0.3)

def test_fitting_lognormal():
    """
    Test if the max. likelihood estimates of a multivariate lognormal
    distribution are sufficiently accurate
    """
    # univariate case
    # generate raw data
    ref_median = 0.5
    ref_std = 0.25

    # generate samples
    RV_reg = RandomVariableRegistry()

    RV_reg.add_RV(RandomVariable(name='A', distribution='lognormal',
                                 theta=[ref_median, ref_std]))

    RV_reg.generate_samples(sample_size=100)

    samples = RV_reg.RV_samples['A']

    # estimate the parameters of the distribution
    median, std = fit_distribution(samples, 'lognormal')[0][0]

    assert ref_median == pytest.approx(median, abs=0.01)
    assert ref_std == pytest.approx(std, rel=0.05)

    # multivariate case
    # generate raw data
    dims = 6
    ref_median = np.exp(np.arange(dims, dtype=np.float64))
    ref_std = np.ones(dims) * 0.5
    ref_rho = np.ones((dims, dims)) * 0.5
    np.fill_diagonal(ref_rho, 1.0)

    RV_reg = RandomVariableRegistry()

    for i, (median, std) in enumerate(zip(ref_median, ref_std)):
        RV_reg.add_RV(RandomVariable(name=i, distribution='lognormal',
                                     theta=[median, std]))

    RV_reg.add_RV_set(
        RandomVariableSet('A',
                          [RV_reg.RV[rv] for rv in range(dims)],
                          ref_rho))

    RV_reg.generate_samples(sample_size=100)

    samples = pd.DataFrame(RV_reg.RV_samples).values.T

    # estimate the parameters of the distribution
    test_theta, test_rho = fit_distribution(samples, 'lognormal')

    test_median, test_std = test_theta.T

    assert_allclose(np.log(test_median), np.log(ref_median), atol=0.01)
    assert_allclose(test_std, ref_std, rtol=0.2)
    assert_allclose(test_rho, ref_rho, atol=0.2)



