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
These are unit and integration tests on the uq module of pelicun.

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
    assert_allclose(test_rho, ref_rho, atol=0.3)

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

    #print(max(abs(test_mu-ref_mean)), max(abs(test_std-ref_std)))

    assert_allclose(test_mu, ref_mean, atol=0.1)
    assert_allclose(test_std, ref_std, atol=0.05)
    assert_allclose(test_rho, ref_rho, atol=0.2)

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
    assert_allclose(test_rho, ref_rho, atol=0.3)



