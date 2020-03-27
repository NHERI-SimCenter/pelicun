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
This module defines helper methods for pelicun unit tests.

"""

import pytest
import numpy as np
from scipy.stats import normaltest, t, chi2
from scipy.stats import kde

import os, sys, inspect
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir))

def prob_approx(prob, atol_50):
    if prob > 0 and prob < 1:
        atol = 2.0 * np.sqrt(prob * (1 - prob)) * atol_50
    else:
        atol = 0.01 * atol_50

    return pytest.approx(prob, abs=atol)


def prob_allclose(reference, test, atol_50):
    for ref, tst in zip(reference, test):
        assert ref == prob_approx(tst, atol_50)

def assert_normal_distribution(sampling_function, ref_mean, ref_COV):
    """
    Check if the sampling function produces samples from a normal distribution
    with mean and COV equal to the provided reference values. The means and
    variances of a multivariate normal distribution are tested by looking at
    its marginal distributions. The correlation structure is tested by
    evaluating the likelihood of being in an orthant of the input space.

    The samples will never have the exact same mean and variance, which makes
    such assertions rather difficult. We perform three hypothesis tests:

    (i) test if the distribution is normal using the K2 test by D'Agostino and
    Pearson (1973);
    (ii) assuming a normal distribution test if its mean is the reference mean;
    (iii) assuming a normal distribution test if its standard deviation is the
    reference standard deviation.

    The level of significance is set at 0.01. This would often lead to false
    negatives during unit testing, but with a much smaller value we risk not
    recognizing slight, but consistent errors in sampling. Therefore, instead
    of lowering the level of significance, we adopt the following strategy: If
    the samples do not support our null hypothesis, we draw more samples -
    this conceptually corresponds to performing additional experiments when we
    experience a strange result in testing. If the underlying distribution is
    truly not normal, drawing more samples should not help. In the other false
    negative cases, this should correct the assertion and reduce the likelihood
    of false negatives to a sufficiently low level that makes this assertion
    applicable for unit testing. Note that since the additional test results
    are influenced by the previous (outlier) samples, the results of hypothesis
    testing are conditioned on the failure of previous tests. Therefore, we
    have non-negligible probability of false negatives in the additional tests.
    Given at most 6 sample draws, a false negative assertion becomes a
    sufficiently rare event to allow this function to work for unit testing.

    Parameters
    ----------
    sampling_function: function
        Any function that takes sample_size as its only argument and provides
        that many samples of a supposedly normal distribution as a result.
    ref_mean: float scalar or ndarray
        Mean(s) of the reference distribution.
    ref_COV: float scalar or ndarray
        Covariance matrix of the reference distribution.

    Returns
    -------
    output: int
        Assertion result. True if the samples support the hypotheses, False
        otherwise.
    """

    # get the number of dimensions
    ref_means = np.asarray(ref_mean)
    if ref_means.shape == ():
        ndim = 1
        ref_means = np.asarray([ref_means])
        ref_COV = np.asarray([ref_COV])
        ref_stds = np.sqrt(ref_COV)
    else:
        ndim = len(ref_means)
        ref_COV = np.asarray(ref_COV)
        ref_stds = np.sqrt(np.diagonal(ref_COV))

        #prepare variables for correlation checks
        lower = np.ones(ndim) * -np.inf
        upper = np.ones(ndim)
        lowinf = np.isneginf(lower)
        uppinf = np.isposinf(upper)
        infin = 2.0 * np.ones(ndim)
        np.putmask(infin, lowinf, 0)
        np.putmask(infin, uppinf, 1)
        np.putmask(infin, lowinf * uppinf, -1)
        corr = ref_COV / np.outer(ref_stds, ref_stds)
        correl = corr[np.tril_indices(ndim, -1)]
        # estimate the proportion of samples that should be in the
        # orthant below the mean + std based on the covariance matrix
        __, est_prop, __ = kde.mvn.mvndst(lower, upper, infin, correl)

    # initialize the control parameters
    alpha = 0.01
    chances = 6
    size = 5
    samples = None

    # test the means and variances
    for j in range(chances):
        size = size * 4

        if samples is not None:
            samples = np.concatenate((samples,
                                      sampling_function(sample_size=size)),
                                     axis=0)
        else:
            samples = sampling_function(sample_size=size)
        size = len(samples)

        test_result = True
        # for each dimension...
        for d in range(ndim):
            if test_result == True:
                ref_mean = ref_means[d]
                ref_std = ref_stds[d]
                if ndim > 1:
                    samples_d = np.transpose(samples)[d]
                else:
                    samples_d = samples
                # test if the distribution is normal
                __, p_k2 = normaltest(samples_d)
                if p_k2 > alpha:

                    # test if the mean and stdev are appropriate
                    sample_mean = np.mean(samples_d)
                    sample_std = np.std(samples_d, ddof=1)

                    df = size - 1
                    mean_stat = ((sample_mean - ref_mean) /
                                 (sample_std / np.sqrt(size)))
                    p_mean = 2 * t.cdf(-np.abs(mean_stat), df=df)

                    std_stat = df * sample_std ** 2. / ref_std ** 2.
                    std_stat_delta = np.abs(std_stat - df)
                    p_std = (chi2.cdf(df - std_stat_delta, df=df) +
                             (1. - chi2.cdf(df + std_stat_delta, df=df)))

                    if (p_mean < alpha) or (p_std < alpha):
                        test_result = False
                else:
                    test_result = False

        if test_result == False:
            continue

        # if the distribution still seems normal and this is a multivariate
        # case, then test the correlation structure
        else:
            if ndim > 1:
                # calculate the proportion of samples in the orthant below the
                # (sample) mean + one std
                sample_mean = np.mean(samples, axis=0)
                sample_std = np.std(samples, axis=0, ddof=1)
                orthant_limit = sample_mean + sample_std
                empirical_prop = np.sum(np.all(samples<orthant_limit, axis=1))
                empirical_prop = empirical_prop/float(size)

                if np.abs(np.log(empirical_prop/est_prop)) < 0.05:
                    # the result is accepted if the error is less than 5%
                    break

            else:
                # in a univariate case, the previous tests already confirmed
                # the normality
                break

    # if the hypothesis tests failed after extending the samples several
    # (i.e. chances) times, then the underlying distribution is probably
    # not normal
    return j != (chances-1)

