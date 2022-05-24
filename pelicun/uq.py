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
This module defines constants, classes and methods for uncertainty
quantification in pelicun.

.. rubric:: Contents

.. autosummary::

    mvn_orthotope_density
    fit_distribution_to_sample
    fit_distribution_to_percentiles

    RandomVariable
    RandomVariableSet
    RandomVariableRegistry


"""

from .base import *

from scipy.stats import uniform, norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats.mvn import mvndst
from scipy.linalg import cholesky, svd
from scipy.optimize import minimize

import warnings


def scale_distribution(scale_factor, family, theta, truncation_limits=None):
    """
    Scale parameters of a random distribution.

    Parameters
    ----------
    family: {'normal', 'lognormal', 'uniform'}
        Defines the type of probability distribution for the random variable.
    theta: float ndarray
        Set of parameters that define the cumulative distribution function of
        the variable given its distribution type. See the expected parameters
        explained in the RandomVariable class. Each parameter can be defined by
        one or more values. If a set of values are provided for one parameter,
        they define ordinates of a multilinear function that is used to get
        the parameter values given an independent variable.
    truncation_limits: float ndarray, default: None
        Defines the [a,b] truncation limits for the distribution. Use None to
        assign no limit in one direction.
    """

    if truncation_limits is not None:
        truncation_limits = truncation_limits * scale_factor

    # undefined family is considered deterministic
    if pd.isna(family):
        family = 'deterministic'

    theta_new = np.full_like(theta, np.nan)
    if family == 'normal':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1] # because we use cov instead of std

    elif family == 'lognormal':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1]  # because it is log std

    elif family == 'uniform':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1] * scale_factor

    elif family == 'deterministic':
        theta_new[0] = theta[0] * scale_factor

    return theta_new, truncation_limits

def mvn_orthotope_density(mu, COV, lower=np.nan, upper=np.nan):
    """
    Estimate the probability density within a hyperrectangle for an MVN distr.

    Use the method of Alan Genz (1992) to estimate the probability density
    of a multivariate normal distribution within an n-orthotope (i.e.,
    hyperrectangle) defined by its lower and upper bounds. Limits can be
    relaxed in any direction by assigning infinite bounds (i.e. numpy.inf).

    Parameters
    ----------
    mu: float scalar or ndarray
        Mean(s) of the non-truncated distribution.
    COV: float ndarray
        Covariance matrix of the non-truncated distribution
    lower: float vector, optional, default: None
        Lower bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from below
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. -numpy.inf) to those dimensions.
    upper: float vector, optional, default: None
        Upper bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from above
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. numpy.inf) to those dimensions.
    Returns
    -------
    alpha: float
        Estimate of the probability density within the hyperrectangle
    eps_alpha: float
        Estimate of the error in alpha.

    """

    # process the inputs and get the number of dimensions
    mu = np.atleast_1d(mu)
    COV = np.atleast_2d(COV)

    if mu.shape == ():
        mu = np.asarray([mu])
        COV = np.asarray([COV])
    else:
        COV = np.asarray(COV)

    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig, sig)

    ndim = mu.size

    if np.all(np.isnan(lower)):
        lower = -np.ones(ndim) * np.inf
    else:
        lower = np.atleast_1d(lower)

    if np.all(np.isnan(upper)):
        upper = np.ones(ndim) * np.inf
    else:
        upper = np.atleast_1d(upper)

    # replace None with np.inf
    lower[np.where(np.isnan(lower))[0]] = -np.inf
    lower = lower.astype(np.float64)
    upper[np.where(np.isnan(upper))[0]] = np.inf
    upper = upper.astype(np.float64)

    # standardize the truncation limits
    lower = (lower - mu) / sig
    upper = (upper - mu) / sig

    # prepare the flags for infinite bounds (these are needed for the mvndst
    # function)
    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)
    infin = 2.0 * np.ones(ndim)

    np.putmask(infin, lowinf, 0)
    np.putmask(infin, uppinf, 1)
    np.putmask(infin, lowinf * uppinf, -1)

    # prepare the correlation coefficients
    if ndim == 1:
        correl = 0
    else:
        correl = corr[np.tril_indices(ndim, -1)]

    # estimate the density
    eps_alpha, alpha, __ = mvndst(lower, upper, infin, correl)

    return alpha, eps_alpha


def _get_theta(params, inits, dist_list):
    """
    Returns the parameters of the target distributions.

    Uses the parameter values from the optimization algorithm (that are relative
    to the initial values) and the initial values to transform them to the
    parameters of the target distributions.

    """

    theta = np.zeros(inits.shape)

    for i, (params_i, inits_i, dist_i) in enumerate(zip(params, inits, dist_list)):

        if dist_i in ['normal', 'lognormal']:

            # Note that the standard deviation is fit in log space, hence the
            # unusual-looking transformation here
            sig = np.exp(np.log(inits_i[1]) + params_i[1])

            # The mean uses the standard transformation
            mu = inits_i[0] + params_i[0] * sig

            theta[i, 0] = mu
            theta[i, 1] = sig

    return theta


def _get_limit_probs(limits, distribution, theta):

    if distribution in ['normal', 'lognormal']:

        a, b = limits
        mu = theta[0]
        sig = theta[1]

        if np.isnan(a):
            p_a = 0.0
        else:
            p_a = norm.cdf((a - mu) / sig)

        if np.isnan(b):
            p_b = 1.0
        else:
            p_b = norm.cdf((b - mu) / sig)

    return p_a, p_b


def _get_std_samples(samples, theta, tr_limits, dist_list):

    ndims = samples.shape[0]

    std_samples = np.zeros(samples.shape)

    for i, (samples_i, theta_i, tr_lim_i, dist_i) in enumerate(
        zip(samples, theta, tr_limits, dist_list)):

        if dist_i in ['normal', 'lognormal']:

            # first transform from normal to uniform
            uni_samples = norm.cdf(samples_i, loc=theta_i[0], scale=theta_i[1])

            # consider truncation if needed
            p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
            uni_samples = (uni_samples - p_a) / (p_b - p_a)

            # then transform from uniform to standard normal
            std_samples[i] = norm.ppf(uni_samples, loc=0., scale=1.)

    return std_samples


def _get_std_corr_matrix(std_samples):

    n_dims, n_samples = std_samples.shape

    # initialize the correlation matrix estimate
    rho_hat = np.zeros((n_dims, n_dims))
    np.fill_diagonal(rho_hat, 1.0)

    # take advantage of having standard normal samples
    for dim_i in range(n_dims):
        for dim_j in np.arange(dim_i + 1, n_dims):
            rho_hat[dim_i, dim_j] = (
                    np.sum(std_samples[dim_i] * std_samples[dim_j]) / n_samples)
            rho_hat[dim_j, dim_i] = rho_hat[dim_i, dim_j]

    # make sure rho_hat is positive semidefinite
    try:
        L = cholesky(rho_hat, lower=True)  # if this works, we're good

    except:  # otherwise, we can try to fix the matrix using SVD

        try:
            U, s, V = svd(rho_hat, )
        except:
            # if this also fails, we give up
            return None

        S = np.diagflat(s)

        rho_hat = U @ S @ U.T
        np.fill_diagonal(rho_hat, 1.0)

        # check if we introduced any unreasonable values
        if ((np.max(rho_hat) > 1.0) or (np.min(rho_hat) < -1.0)):
            return None

    return rho_hat


def _mvn_scale(x, rho):

    x = np.atleast_2d(x)
    n_dims = x.shape[1]

    # create an uncorrelated covariance matrix
    rho_0 = np.zeros((n_dims, n_dims))
    np.fill_diagonal(rho_0, 1)

    a = mvn.pdf(x, mean=np.zeros(n_dims), cov=rho_0)

    b = mvn.pdf(x, mean=np.zeros(n_dims), cov=rho)

    return b / a

def _neg_log_likelihood(params, inits, bnd_lower, bnd_upper, samples,
                        dist_list, tr_limits, det_limits, censored_count,
                        enforce_bounds=False):

    # First, check if the parameters are within the pre-defined bounds
    # TODO: check if it is more efficient to use a bounded minimization algo
    if enforce_bounds:
        if ((params > bnd_lower) & (params < bnd_upper)).all(0) == False:
            # if they are not, then return a large value to discourage the
            # optimization algorithm from going in that direction
            return 1e10

    # If there is nan in params, return a large value
    if np.isnan(np.sum(params)):
        return 1e10

    params = np.reshape(params, inits.shape)
    n_dims, n_samples = samples.shape

    theta = _get_theta(params, inits, dist_list)

    likelihoods = np.zeros(samples.shape)

    # calculate the marginal likelihoods
    for i, (theta_i, samples_i, tr_lim_i, dist_i) in enumerate(
            zip(theta, samples, tr_limits, dist_list)):

        # consider truncation if needed
        p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
        tr_alpha = p_b - p_a  # this is the probability mass within the
                              # truncation limits

        # Calculate the likelihood for each available sample
        # Note that we are performing this without any transformation to be able
        # to respect truncation limits
        if dist_i in ['normal', 'lognormal']:
            likelihoods[i] = norm.pdf(
                samples_i, loc=theta_i[0], scale=theta_i[1]) / tr_alpha

    # transform every sample into standard normal space
    std_samples = _get_std_samples(samples, theta, tr_limits, dist_list)

    # if the problem is more than one dimensional, get the correlation matrix
    if n_dims > 1:
        rho_hat = _get_std_corr_matrix(std_samples)
        if rho_hat is None:
            return 1e10
    else:
        rho_hat = np.atleast_2d([1.0])

    # likelihoods related to censoring need to be handled together
    if censored_count > 0:

        det_lower = np.zeros(n_dims)
        det_upper = np.zeros(n_dims)

        for i, (theta_i, tr_lim_i, det_lim_i, dist_i) in enumerate(
            zip(theta, tr_limits, det_limits, dist_list)):

            # prepare the standardized truncation and detection limits
            p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
            p_l, p_u = _get_limit_probs(det_lim_i, dist_i, theta_i)

            # rescale detection limits to consider truncation
            p_l, p_u = [np.min([np.max([lim, p_a]), p_b]) for lim in [p_l, p_u]]
            p_l, p_u = [(lim - p_a) / (p_b - p_a) for lim in [p_l, p_u]]

            # transform limits to standard normal space
            det_lower[i], det_upper[i] = norm.ppf([p_l, p_u], loc=0., scale=1.)

        # get the likelihood of getting a non-censored sample given the
        # detection limits and the correlation matrix
        det_alpha, eps_alpha = mvn_orthotope_density(
            np.zeros(n_dims), rho_hat, det_lower, det_upper)

        # Make sure det_alpha is estimated with sufficient accuracy
        if det_alpha <= 100. * eps_alpha:
            return 1e10

        # make sure that the likelihood of censoring a sample is positive
        cen_likelihood = max(1.0 - det_alpha, np.nextafter(0, 1))

    else:
        # If the data is not censored, use 1.0 for cen_likelihood to get a
        # zero log-likelihood later. Note that although this is
        # theoretically not correct, it does not alter the solution and
        # it is numerically much more convenient than working around the
        # log of zero likelihood.
        cen_likelihood = 1.0

    # take the product of likelihoods calculated in each dimension
    try:
        scale = _mvn_scale(std_samples.T, rho_hat)
    except:
        return 1e10
    # TODO: We can almost surely replace the product of likelihoods with a call
    # to mvn()
    likelihoods = np.prod(likelihoods, axis=0) * scale

    # Zeros are a result of limited floating point precision. Replace them
    # with the smallest possible positive floating point number to
    # improve convergence.
    likelihoods = np.clip(likelihoods, a_min=np.nextafter(0, 1), a_max=None)

    # calculate the total negative log likelihood
    NLL = -(np.sum(np.log(likelihoods))  # from samples
            + censored_count * np.log(cen_likelihood))  # censoring influence

    # normalize the NLL with the sample count
    NLL = NLL / samples.size

    # print(theta[0], NLL)

    return NLL

def fit_distribution_to_sample(raw_samples, distribution,
                               truncation_limits=[np.nan, np.nan],
                               censored_count=0, detection_limits=[np.nan, np.nan],
                               multi_fit=False, alpha_lim=1e-4):
    """
    Fit a distribution to sample using maximum likelihood estimation.

    The number of dimensions of the distribution are inferred from the
    shape of the sample data. Censoring is automatically considered if the
    number of censored samples and the corresponding detection limits are
    provided. Infinite or unspecified truncation limits lead to fitting a
    non-truncated distribution in that dimension.

    Parameters
    ----------
    raw_samples: float ndarray
        Raw data that serves as the basis of estimation. The number of samples
        equals the number of columns and each row introduces a new feature. In
        other words: a list of sample lists is expected where each sample list
        is a collection of samples of one variable.
    distribution: {'normal', 'lognormal'}
        Defines the target probability distribution type. Different types of
        distributions can be mixed by providing a list rather than a single
        value. Each element of the list corresponds to one of the features in
        the raw_samples.
    truncation_limits: float ndarray, optional, default: [None, None]
        Lower and/or upper truncation limits for the specified distributions.
        A two-element vector can be used for a univariate case, while two lists
        of limits are expected in multivariate cases. If the distribution is
        non-truncated from one side in a subset of the dimensions, use either
        `None` or assign an infinite value (i.e. numpy.inf) to those dimensions.
    censored_count: int, optional, default: None
        The number of censored samples that are beyond the detection limits.
        All samples outside the detection limits are aggregated into one set.
        This works the same way in one and in multiple dimensions. Prescription
        of specific censored sample counts for sub-regions of the input space
        outside the detection limits is not supported.
    detection_limits: float ndarray, optional, default: [None, None]
        Lower and/or upper detection limits for the provided samples. A
        two-element vector can be used for a univariate case, while two lists
        of limits are expected in multivariate cases. If the data is not
        censored from one side in a subset of the dimensions, use either `None`
        or assign an infinite value (i.e. numpy.inf) to those dimensions.
    multi_fit: bool, optional, default: False
        If True, we attempt to fit a multivariate distribution to the samples.
        Otherwise, we fit each marginal univariate distribution independently
        and estimate the correlation matrix in the end based on the fitted
        marginals. Using multi_fit can be advantageous with censored data and
        if the correlation in the data is not Gaussian. It leads to
        substantially longer calculation time and does not always produce
        better results, especially when the number of dimensions is large.
    alpha_lim: float, optional, default:None
        Introduces a lower limit to the probability density within the
        n-orthotope defined by the truncation limits. Assigning a reasonable
        minimum (such as 1e-4) can be useful when the mean of the distribution
        is several standard deviations from the truncation limits and the
        sample size is small. Such cases without a limit often converge to
        distant means with inflated variances. Besides being incorrect
        estimates, those solutions only offer negligible reduction in the
        negative log likelihood, while making subsequent sampling of the
        truncated normal distribution very challenging.

    Returns
    -------
    theta: float ndarray
        Estimates of the parameters of the fitted probability distribution in
        each dimension. The following parameters are returned for the supported
        distributions:
        normal - mean, standard deviation;
        lognormal - median, log standard deviation;
    Rho: float 2D ndarray, optional
        In the multivariate case, returns the estimate of the correlation
        matrix.
    """

    samples = np.atleast_2d(raw_samples)
    tr_limits = np.atleast_2d(truncation_limits)
    det_limits = np.atleast_2d(detection_limits)
    dist_list = np.atleast_1d(distribution)
    n_dims, n_samples = samples.shape

    if (tr_limits.shape[0] == 1) and (n_dims != 1):
        tr_limits = np.tile(tr_limits[0], n_dims).reshape([n_dims, 2])

    if (det_limits.shape[0] == 1) and (n_dims != 1):
        det_limits = np.tile(det_limits[0], n_dims).reshape([n_dims, 2])

    if (dist_list.shape[0] == 1) and (n_dims != 1):
        dist_list = np.tile(dist_list[0], n_dims).reshape([n_dims, 1])

    # transpose limit arrays
    tr_limits = tr_limits.T
    det_limits = det_limits.T

    # Convert samples and limits to log space if the distribution is lognormal
    for d_i, distribution in enumerate(dist_list):

        if distribution == 'lognormal':

            samples[d_i] = np.log(samples[d_i])

            for lim in range(2):
                if not np.isnan(tr_limits[d_i][lim]):
                    tr_limits[d_i][lim] = np.log(tr_limits[d_i][lim])

            for lim in range(2):
                if not np.isnan(det_limits[d_i][lim]):
                    det_limits[d_i][lim] = np.log(det_limits[d_i][lim])

    # Define initial values of distribution parameters
    # Initialize arrays
    mu_init = np.ones(n_dims)*np.nan
    sig_init = np.ones_like(mu_init)*np.nan

    for d_i, distribution in enumerate(dist_list):

        if distribution in ['normal', 'lognormal']:
            # use the first two moments
            mu_init[d_i] = np.mean(samples[d_i])

            if n_samples == 1:
                sig_init[d_i] = 0.0
            else:
                sig_init[d_i] = np.std(samples[d_i])

    # replace zero standard dev with negligible standard dev
    sig_zero_id = np.where(sig_init == 0.0)[0]
    sig_init[sig_zero_id] = (1e-6 * np.abs(mu_init[sig_zero_id])
                             + np.nextafter(0, 1))

    # prepare a vector of initial values
    # Note: The actual optimization uses zeros as initial parameters to
    # avoid bias from different scales. These initial values are sent to
    # the likelihood function and considered in there.
    inits = np.transpose([mu_init, sig_init])

    # Define the bounds for each input (assuming standardized initials)
    # These bounds help avoid unrealistic results and improve the
    # convergence rate
    # Note that the standard deviation (2nd parameter) is fit in log space to
    # facilitate fitting it within reasonable bounds.
    bnd_lower = np.array([[-10.0, -5.0] for t in range(n_dims)])
    bnd_upper = np.array([[10.0, 5.0] for t in range(n_dims)])

    bnd_lower = bnd_lower.flatten()
    bnd_upper = bnd_upper.flatten()

    #inits_0 = np.copy(inits)

    # There is nothing to gain from a time-consuming optimization if..
    #     the number of samples is too small
    if ((n_samples < 3) or
        # there are no truncation or detection limits involved
        (np.all(np.isnan(tr_limits)) and np.all(np.isnan(det_limits)))):

        # In this case, it is typically hard to improve on the method of
        # moments estimates for the parameters of the marginal distributions
        theta = inits

    # Otherwise, we run the optimization that aims to find the parameters that
    # maximize the likelihood of observing the samples
    else:

        # First, optimize for each marginal independently
        for dim in range(n_dims):

            inits_i = inits[dim:dim + 1]

            # Censored samples are only considered in the following step, but
            # we fit a truncated distribution if there are censored samples to
            # make it easier to fit the censored distribution later.
            tr_limits_i = [np.nan, np.nan]
            for lim in range(2):
                if ((np.isnan(tr_limits[dim][lim])) and
                    (not np.isnan(det_limits[dim][lim]))):
                    tr_limits_i[lim] = det_limits[dim][lim]
                elif not np.isnan(det_limits[dim][lim]):
                    if lim == 0:
                        tr_limits_i[lim] = np.min([tr_limits[dim][lim],
                                                   det_limits[dim][lim]])
                    elif lim == 1:
                        tr_limits_i[lim] = np.max([tr_limits[dim][lim],
                                                   det_limits[dim][lim]])
                else:
                    tr_limits_i[lim] = tr_limits[dim][lim]

            out_m_i = minimize(_neg_log_likelihood,
                               np.zeros(inits[dim].size),
                               args=(inits_i,
                                     bnd_lower[dim:dim + 1],
                                     bnd_upper[dim:dim + 1],
                                     samples[dim:dim + 1],
                                     [dist_list[dim],],
                                     [tr_limits_i, ],
                                     [np.nan, np.nan],
                                     0, True,),
                               method='BFGS',
                               options=dict(maxiter=50)
                               )

            out = out_m_i.x.reshape(inits_i.shape)
            theta = _get_theta(out, inits_i, [dist_list[dim],])
            inits[dim] = theta[0]

        # Second, if multi_fit is requested or there are censored samples,
        # we attempt the multivariate fitting using the marginal results as
        # initial parameters.
        if multi_fit or (censored_count > 0):

            out_m = minimize(_neg_log_likelihood,
                             np.zeros(inits.size),
                             args=(inits, bnd_lower, bnd_upper, samples,
                                   dist_list, tr_limits, det_limits,
                                   censored_count, True,),
                             method='BFGS',
                             options=dict(maxiter=50)
                             )

            out = out_m.x.reshape(inits.shape)
            theta = _get_theta(out, inits, dist_list)

        else:
            theta = inits

    # Calculate rho in the standard normal space because we will generate new
    # samples using that type of correlation (i.e., Gaussian copula)
    std_samples = _get_std_samples(samples, theta, tr_limits, dist_list)
    rho_hat = _get_std_corr_matrix(std_samples)
    if rho_hat is None:
        # If there is not enough data to produce a valid correlation matrix
        # estimate, we assume uncorrelated demands
        rho_hat = np.zeros((n_dims, n_dims))
        np.fill_diagonal(rho_hat, 1.0)

        log_msg("\nWARNING: Demand sample size too small to reliably estimate "
                "the correlation matrix. Assuming uncorrelated demands.",
                prepend_timestamp=False, prepend_blank_space=False)


    for d_i, distribution in enumerate(dist_list):
        # Convert mean back to linear space if the distribution is lognormal
        if distribution == 'lognormal':
            theta[d_i][0] = np.exp(theta[d_i][0])
            #theta_mod = theta.T.copy()
            #theta_mod[0] = np.exp(theta_mod[0])
            #theta = theta_mod.T
        # Convert the std to cov if the distribution is normal
        elif distribution == 'normal':
            theta[d_i][1] = theta[d_i][1] / np.abs(theta[d_i][0])

    #for val in list(zip(inits_0, theta)):
    #    print(val)

    return theta, rho_hat

def _OLS_percentiles(params, values, perc, family):

    theta_0 = params[0]
    theta_1 = params[1]

    if theta_0 <= 0:
        return 1e10

    if theta_1 <= 0:
        return 1e10

    if family == 'normal':

        val_hat = norm.ppf(perc, loc=theta_0, scale=theta_1)

    elif family == 'lognormal':

        val_hat = np.exp(norm.ppf(perc, loc=np.log(theta_0), scale=theta_1))

    else:
        raise ValueError(f"Distribution family not recognized: {family}")

    return np.sum((val_hat - values) ** 2.0)

def fit_distribution_to_percentiles(values, percentiles, families):
    """
    Fit distribution to pre-defined values at a finite number of percentiles.

    Parameters
    ----------
    values: array of float
        Pre-defined values at the given percentiles. At least two values are
        expected.
    percentiles: array of float
        Percentiles where values are defined. At least two percentiles are
        expected.
    families: array of strings {'normal', 'lognormal'}
        Defines the distribution family candidates.

    Returns
    -------
    family: string
        The optimal choice of family among the provided list of families
    theta: array of float
        Parameters of the fitted distribution.
    """

    out_list = []

    percentiles = np.array(percentiles)

    median_id = np.argmin(np.abs(percentiles - 0.5))
    extreme_id = np.argmax(percentiles - 0.5)

    for family in families:

        inits = [values[median_id],]

        if family == 'normal':
            inits.append((np.abs(values[extreme_id] - inits[0]) /
                          np.abs(norm.ppf(percentiles[extreme_id],
                                          loc=0, scale=1))
                          ))

        elif family == 'lognormal':
            inits.append((np.abs(np.log(values[extreme_id]/inits[0])) /
                          np.abs(norm.ppf(percentiles[extreme_id],
                                          loc=0, scale=1))
                          ))

        out_list.append(minimize(_OLS_percentiles, inits,
                                 args=(values, percentiles, family),
                                 method='BFGS'))

    best_out_id = np.argmin([out.fun for out in out_list])

    return families[best_out_id], out_list[best_out_id].x


class RandomVariable(object):
    """
    Description

    Parameters
    ----------
    name: string
        A unique string that identifies the random variable.
    distribution: {'normal', 'lognormal', 'multinomial', 'custom', 'empirical',
        'coupled_empirical', 'uniform', 'deterministic'}, optional
        Defines the type of probability distribution for the random variable.
    theta: float scalar or ndarray, optional
        Set of parameters that define the cumulative distribution function of
        the variable given its distribution type. The following parameters are
        expected currently for the supported distribution types:
        normal - mean, standard deviation;
        lognormal - median, log standard deviation;
        uniform - a, b, the lower and upper bounds of the distribution;
        multinomial - likelihood of each unique event (the last event's
        likelihood is adjusted automatically to ensure the likelihoods sum up
        to one);
        custom - according to the custom expression provided;
        empirical and coupled_empirical - N/A;
        deterministic - the deterministic value assigned to the variable.
    truncation_limits: float ndarray, optional
        Defines the [a,b] truncation limits for the distribution. Use None to
        assign no limit in one direction.
    bounded: float ndarray, optional
        Defines the [P_a, P_b] probability bounds for the distribution. Use None
        to assign no lower or upper bound.
    custom_expr: string, optional
        Provide an expression that is a Python syntax for a custom CDF. The
        controlling variable shall be "x" and the parameters shall be "p1",
        "p2", etc.
    f_map: function, optional
        A user-defined function that is applied on the realizations before
        returning a sample.
    anchor: RandomVariable, optional
        Anchors this to another variable. If the anchor is not None, this
        variable will be perfectly correlated with its anchor. Note that
        the attributes of this variable and its anchor do not have to be
        identical.
    """

    def __init__(self, name, distribution, theta=np.nan,
                 truncation_limits=np.nan,
                 bounds=None, custom_expr=None, raw_samples=None,
                 f_map=None, anchor=None):

        self.name = name

        if pd.isna(distribution):
            distribution = 'deterministic'

        if ((distribution not in ['empirical', 'coupled_empirical']) and
            (np.all(np.isnan(theta)))):

            raise ValueError(
                f"A random variable that follows a {distribution} distribution "
                f"is characterized by a set of parameters (theta). The "
                f"parameters need to be provided when the RV is created."
            )

        if distribution == 'multinomial':
            if np.sum(theta) > 1:
                raise ValueError(
                    f"The set of p values provided for a multinomial "
                    f"distribution shall sum up to less than or equal to 1.0. "
                    f"The provided values sum up to {np.sum(theta)}. p = "
                    f"{theta} ."
                )

        # save the other parameters internally
        self._distribution = distribution
        self._theta = np.atleast_1d(theta)
        self._truncation_limits = truncation_limits
        self._bounds = bounds
        self._custom_expr = custom_expr
        self._f_map = f_map
        self._raw_samples = np.atleast_1d(raw_samples)
        self._uni_samples = None
        self._RV_set = None

        if anchor == None:
            self._anchor = self
        else:
            self._anchor = anchor

    @property
    def distribution(self):
        """
        Return the assigned probability distribution type.
        """
        return self._distribution

    @property
    def theta(self):
        """
        Return the assigned probability distribution parameters.
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        """
        Define the parameters of the distribution of the random variable
        """
        self._theta = value

    @property
    def truncation_limits(self):
        """
        Return the assigned truncation limits.
        """
        return self._truncation_limits

    @property
    def bounds(self):
        """
        Return the assigned probability bounds.
        """
        return self._bounds

    @property
    def custom_expr(self):
        """
        Return the assigned custom expression for CDF.
        """
        return self._custom_expr

    @property
    def RV_set(self):
        """
        Return the RV_set this RV is a member of
        """
        return self._RV_set

    @RV_set.setter
    def RV_set(self, value):
        """
         Assign an RV_set to this RV
        """
        self._RV_set = value

    @property
    def sample(self):
        """
        Return the empirical or generated sample.
        """
        if self._f_map is not None:

            return self._f_map(self._sample)

        else:
            return self._sample

    @property
    def sample_DF(self):
        """
        Return the empirical or generated sample in a pandas Series.
        """
        if self._f_map is not None:

            return self._sample_DF.apply(self._f_map)

        else:
            return self._sample_DF

    @sample.setter
    def sample(self, value):
        """
        Assign a sample to the random variable
        """
        self._sample = value
        self._sample_DF = pd.Series(value)

    @property
    def uni_sample(self):
        """
        Return the sample from the controlling uniform distribution.
        """
        return self._anchor._uni_samples

    @uni_sample.setter
    def uni_sample(self, value):
        """
        Assign the controlling sample to the random variable

        Parameters
        ----------
        value: float ndarray
            An array of floating point values in the [0, 1] domain.
        """
        self._uni_samples = value

    @property
    def anchor(self):
        """
        Return the anchor of the variable (if any).
        """
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        """
        Assign an anchor to the random variable
        """
        self._anchor = value

    def cdf(self, values):
        """
        Returns the cdf at the given values
        """
        result = None

        if self.distribution == 'normal':
            mu, cov = self.theta[:2]
            sig = np.abs(mu)*cov

            if np.any(~np.isnan(self.truncation_limits)):
                a, b = self.truncation_limits

                if np.isnan(a):
                    a = -np.inf
                if np.isnan(b):
                    b = np.inf

                p_a, p_b = [norm.cdf((lim - mu) / sig) for lim in [a, b]]

                # cap the values at the truncation limits
                values = np.minimum(np.maximum(values, a), b)

                # get the cdf from a non-truncated normal
                p_vals = norm.cdf(values, loc=mu, scale=sig)

                # adjust for truncation
                result = (p_vals - p_a) / (p_b - p_a)

            else:
                result = norm.cdf(values, loc=mu, scale=sig)

        elif self.distribution == 'lognormal':
            theta, beta = self.theta[:2]

            if np.any(~np.isnan(self.truncation_limits)):
                a, b = self.truncation_limits

                if np.isnan(a):
                    a = np.nextafter(0, 1)
                if np.isnan(b):
                    b = np.inf

                p_a, p_b = [norm.cdf((np.log(lim) - np.log(theta)) / beta)
                            for lim in [a, b]]

                # cap the values at the truncation limits
                values = np.minimum(np.maximum(values, a), b)

                # get the cdf from a non-truncated lognormal
                p_vals = norm.cdf(np.log(values), loc=np.log(theta), scale=beta)

                # adjust for truncation
                result = (p_vals - p_a) / (p_b - p_a)

            else:
                values = np.maximum(values, np.nextafter(0, 1))

                result = norm.cdf(np.log(values), loc=np.log(theta), scale=beta)

        elif self.distribution == 'uniform':
            a, b = self.theta[:2]

            if np.isnan(a):
                a = -np.inf
            if np.isnan(b):
                b = np.inf

            if np.any(~np.isnan(self.truncation_limits)):
                a, b = self.truncation_limits

            result = uniform.cdf(values, loc=a, scale=b-a)

        return result


    def inverse_transform(self, values=None, sample_size=None):
        """
        Uses inverse probability integral transformation on the provided values.
        """
        result = None

        if self.distribution == 'normal':

            if values is None:
                raise ValueError(
                    "Missing uniform sample for inverse transform sampling a "
                    "normal random variable.")

            else:

                mu, cov = self.theta[:2]
                sig = np.abs(mu) * cov

                if np.any(~np.isnan(self.truncation_limits)):
                    a, b = self.truncation_limits

                    if np.isnan(a):
                        a = -np.inf
                    if np.isnan(b):
                        b = np.inf

                    p_a, p_b = [norm.cdf((lim-mu)/sig) for lim in [a, b]]

                    if p_b - p_a == 0:
                        raise ValueError(
                            "The probability mass within the truncation limits is "
                            "too small and the truncated distribution cannot be "
                            "sampled with sufficiently high accuracy. This is most "
                            "probably due to incorrect truncation limits set for "
                            "the distribution."
                        )

                    result = norm.ppf(values * (p_b - p_a) + p_a,
                                            loc=mu, scale=sig)

                else:
                    result = norm.ppf(values, loc=mu, scale=sig)

        elif self.distribution == 'lognormal':

            if values is None:
                raise ValueError(
                    "Missing uniform sample for inverse transform sampling a "
                    "lognormal random variable.")

            else:

                theta, beta = self.theta[:2]

                if np.any(~np.isnan(self.truncation_limits)):
                    a, b = self.truncation_limits

                    if np.isnan(a):
                        a = np.nextafter(0, 1)
                    else:
                        a = np.maximum(np.nextafter(0, 1), a)

                    if np.isnan(b):
                        b = np.inf

                    p_a, p_b = [norm.cdf((np.log(lim) - np.log(theta)) / beta)
                                for lim in [a, b]]

                    result = np.exp(
                        norm.ppf(values * (p_b - p_a) + p_a,
                                 loc=np.log(theta), scale=beta))

                else:
                    result = np.exp(norm.ppf(values, loc=np.log(theta), scale=beta))

        elif self.distribution == 'uniform':

            if values is None:
                raise ValueError(
                    "Missing uniform sample for inverse transform sampling a "
                    "uniform random variable.")

            else:

                a, b = self.theta[:2]

                if np.isnan(a):
                    a = -np.inf
                if np.isnan(b):
                    b = np.inf

                if np.any(~np.isnan(self.truncation_limits)):
                    a, b = self.truncation_limits

                result = uniform.ppf(values, loc=a, scale=b-a)

        elif self.distribution == 'empirical':

            if values is None:
                raise ValueError(
                    "Missing uniform sample for inverse transform sampling an "
                    "empirical random variable.")

            else:

                s_ids = (values * len(self._raw_samples)).astype(int)
                result = self._raw_samples[s_ids]

        elif self.distribution == 'coupled_empirical':

            if sample_size is None:
                raise ValueError(
                    "Missing sample size information for sampling a coupled "
                    "empirical random variable.")
            else:
                raw_sample_count = len(self._raw_samples)
                new_sample = np.tile(self._raw_samples,
                                      int(sample_size/raw_sample_count)+1)
                result = new_sample[:sample_size]

        elif self.distribution == 'deterministic':

            if sample_size is None:
                raise ValueError(
                    "Missing sample size information for sampling a "
                    "deterministic random variable.")
            else:
                result = np.full(sample_size, self.theta[0])

        elif self.distribution == 'multinomial':

            if values is None:
                raise ValueError(
                    "Missing uniform sample for sampling a multinomial random "
                    "variable.")

            else:

                p_cum = np.cumsum(self.theta)[:-1]

                samples = values

                for i, p_i in enumerate(p_cum):
                    samples[samples < p_i] = 10 + i
                samples[samples <= 1.0] = 10 + len(p_cum)

                result = samples - 10

        return result

    def inverse_transform_sampling(self, sample_size=None):
        """
        Creates a sample using inverse probability integral transformation.
        """

        self.sample = self.inverse_transform(self.uni_sample, sample_size)

class RandomVariableSet(object):
    """
    Description

    Parameters
    ----------
    name: string
        A unique string that identifies the set of random variables.
    RV_list: list of RandomVariable
        Defines the random variables in the set
    Rho: float 2D ndarray
        Defines the correlation matrix that describes the correlation between
        the random variables in the set. Currently, only the Gaussian copula
        is supported.
    """

    def __init__(self, name, RV_list, Rho):

        self.name = name

        if len(RV_list) > 1:

            # put the RVs in a dictionary for more efficient access
            reorder = np.argsort([RV.name for RV in RV_list])
            self._variables = dict([(RV_list[i].name, RV_list[i]) for i in reorder])

            # reorder the entries in the correlation matrix to correspond to the
            # sorted list of RVs
            self._Rho = np.asarray(Rho[(reorder)].T[(reorder)].T)

        else: # if there is only one variable (for testing, probably)
            self._variables = dict([(rv.name, rv) for rv in RV_list])
            self._Rho = np.asarray(Rho)

        # assign this RV_set to the variables
        for __, var in self._variables.items():
            var.RV_set = self

    @property
    def RV(self):
        """
        Return the random variable(s) assigned to the set
        """
        return self._variables

    @property
    def size(self):
        """
        Return the size (i.e., number of variables in the) RV set
        """
        return len(self._variables)

    @property
    def sample(self):
        """
        Return the sample of the variables in the set
        """
        return dict([(name, rv.sample) for name, rv
                     in self._variables.items()])

    def Rho(self, var_subset=None):
        """
        Return the (subset of the) correlation matrix.
        """
        if var_subset is None:
            return self._Rho
        else:
            var_ids = [list(self._variables.keys()).index(var_i)
                       for var_i in var_subset]
            return (self._Rho[var_ids]).T[var_ids]

    def apply_correlation(self):
        """
        Apply correlation to n dimensional uniform samples.

        Currently, correlation is applied using a Gaussian copula. First, we
        try using Cholesky transformation. If the correlation matrix is not
        positive semidefinite and Cholesky fails, use SVD to apply the
        correlations while preserving as much as possible from the correlation
        matrix.
        """

        U_RV = np.array([RV.uni_sample for RV_name, RV in self.RV.items()])

        # First try doing the Cholesky transformation
        try:
            N_RV = norm.ppf(U_RV)

            L = cholesky(self._Rho, lower=True)

            NC_RV = L @ N_RV

            UC_RV = norm.cdf(NC_RV)

        except:

            # if the Cholesky doesn't work, we need to use the more
            # time-consuming but more robust approach based on SVD
            N_RV = norm.ppf(U_RV)

            U, s, __ = svd(self._Rho, )
            S = np.diagflat(np.sqrt(s))

            NC_RV = (N_RV.T @ S @ U.T).T

            UC_RV = norm.cdf(NC_RV)

        for (RV_name, RV), uc_RV in zip(self.RV.items(), UC_RV):
            RV.uni_sample = uc_RV

    def orthotope_density(self, lower=np.nan, upper=np.nan, var_subset=None):
        """
        Estimate the probability density within an orthotope for the RV set.

        Use the mvn_orthotope_density function in this module for the
        calculation. The distribution of individual RVs is not limited to the
        normal family. The provided limits are converted to the standard normal
        space that is the basis of all RVs in pelicun. Truncation limits and
        correlation (using Gaussian copula) are automatically taken into
        consideration.

        Parameters
        ----------
        lower: float ndarray, optional, default: None
            Lower bound(s) of the orthotope. A scalar value can be used for a
            univariate RV; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from below in a dimension, use
            'None' to that dimension.
        upper: float ndarray, optional, default: None
            Upper bound(s) of the orthotope. A scalar value can be used for a
            univariate RV; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from above in a dimension, use
            'None' to that dimension.
        var_subset: list of strings, optional, default: None
            If provided, allows for selecting only a subset of the variables in
            the RV_set for the density calculation.

        Returns
        -------
        alpha: float
            Estimate of the probability density within the orthotope.
        eps_alpha: float
            Estimate of the error in alpha.

        """

        if np.any(~np.isnan(lower)):
            target_shape = lower.shape
        elif np.any(~np.isnan(upper)):
            target_shape = upper.shape
        else:
            return 1.0

        lower_std = np.full(target_shape, np.nan)
        upper_std = np.full(target_shape, np.nan)

        # collect the variables involved
        if var_subset is None:
            vars = list(self._variables.keys())
        else:
            vars = var_subset

        # first, convert limits to standard normal values
        for var_i, var_name in enumerate(vars):

            var = self._variables[var_name]

            if (np.any(~np.isnan(lower))) and (~np.isnan(lower[var_i])):
                lower_std[var_i] = norm.ppf(var.cdf(lower[var_i]), loc=0, scale=1)

            if (np.any(~np.isnan(upper))) and (~np.isnan(upper[var_i])):
                upper_std[var_i] = norm.ppf(var.cdf(upper[var_i]), loc=0, scale=1)

        # then calculate the orthotope results in std normal space
        lower_std = lower_std.T
        upper_std = upper_std.T

        OD = [mvn_orthotope_density(mu=np.zeros(len(vars)),
                                    COV=self.Rho(var_subset),
                                    lower=l_i, upper=u_i)[0]
              for l_i, u_i in zip(lower_std, upper_std)]

        return np.asarray(OD)

class RandomVariableRegistry(object):
    """
    Description

    Parameters
    ----------

    """

    def __init__(self):

        self._variables = {}
        self._sets = {}

    @property
    def RV(self):
        """
        Return all random variable(s) in the registry
        """
        return self._variables

    def RVs(self, keys):
        """
        Return a subset of the random variables in the registry
        """
        return {name:self._variables[name] for name in keys}

    def add_RV(self, RV):
        """
        Add a new random variable to the registry.
        """
        self._variables.update({RV.name: RV})

    @property
    def RV_set(self):
        """
        Return the random variable set(s) in the registry.
        """
        return self._sets

    def add_RV_set(self, RV_set):
        """
        Add a new set of random variables to the registry
        """
        self._sets.update({RV_set.name: RV_set})

    @property
    def RV_sample(self):
        """
        Return the sample for every random variable in the registry
        """
        return dict([(name, rv.sample) for name, rv in self.RV.items()])


    def generate_sample(self, sample_size, method=None):
        """
        Generates samples for all variables in the registry.

        Parameters
        ----------

        sample_size: int
            The number of samples requested per variable.
        method: {'MonteCarlo', 'LHS', 'LHS_midpoint'}, optional
            The sample generation method to use. 'MonteCarlo' stands for
            conventional random sampling; 'LHS' is Latin HyperCube Sampling
            with random sample location within each bin of the hypercube;
            'LHS_midpoint' is like LHS, but the samples are assigned to the
            midpoints of the hypercube bins.
        seed: int, optional
            Random seed used for sampling.
        """
        if method is None:
            method = options.sampling_method

        # Initialize the random number generator
        rng = options.rng

        # Generate a dictionary with IDs of the free (non-anchored and
        # non-deterministic) variables
        RV_list = [RV_name for RV_name, RV in self.RV.items() if
                   ((RV.anchor == RV) or
                    (RV.distribution in ['deterministic',
                                         'coupled_empirical']))]
        RV_ID = dict([(RV_name, ID) for ID, RV_name in enumerate(RV_list)])
        RV_count = len(RV_ID)

        # Generate controlling samples from a uniform distribution for free RVs
        if 'LHS' in method:
            bin_low = np.array([rng.permutation(sample_size)
                                for i in range(RV_count)])

            if method == 'LHS_midpoint':
                U_RV = np.ones([RV_count, sample_size]) * 0.5
                U_RV = (bin_low + U_RV) / sample_size

            elif method == 'LHS':
                U_RV = rng.random(size=[RV_count, sample_size])
                U_RV = (bin_low + U_RV) / sample_size

        elif method == 'MonteCarlo':
            U_RV = rng.random(size=[RV_count, sample_size])

        # Assign the controlling samples to the RVs
        for RV_name, RV_id in RV_ID.items():
            self.RV[RV_name].uni_sample = U_RV[RV_id]

        # Apply correlations for the pre-defined RV sets
        for RV_set_name, RV_set in self.RV_set.items():
            # prepare the correlated uniform distribution for the set
            RV_set.apply_correlation()

        # Convert from uniform to the target distribution for every RV
        for RV_name, RV in self.RV.items():
            RV.inverse_transform_sampling(sample_size)
