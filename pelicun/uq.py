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
# John Vouvakis Manousakis

"""
This module defines constants, classes and methods for uncertainty
quantification in pelicun.

.. rubric:: Contents

.. autosummary::

    scale_distribution
    mvn_orthotope_density
    fit_distribution_to_sample
    fit_distribution_to_percentiles

    RandomVariable
    RandomVariableSet
    RandomVariableRegistry


"""

from abc import ABC, abstractmethod
from scipy.stats import uniform, norm
from scipy.stats import multivariate_normal as mvn
from scipy.stats._mvn import mvndst  # pylint: disable=no-name-in-module
from scipy.linalg import cholesky, svd
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def scale_distribution(scale_factor, family, theta, truncation_limits=None):
    """
    Scale parameters of a random distribution.

    Parameters
    ----------
    scale_factor: float
        Value by which to scale the parameters.
    family: {'normal', 'lognormal', 'uniform'}
        Defines the type of probability distribution for the random variable.
    theta: float ndarray of length 2
        Set of parameters that define the cumulative distribution function of
        the variable given its distribution type. See the expected parameters
        explained in the RandomVariable class. Each parameter can be defined by
        one or more values. If a set of values are provided for one parameter,
        they define ordinates of a multilinear function that is used to get
        the parameter values given an independent variable.
    truncation_limits: float ndarray of length 2, default: None
        Defines the [a,b] truncation limits for the distribution. Use None to
        assign no limit in one direction.

    Returns
    -------
    tuple
        A tuple containing the scaled parameters and truncation
        limits:
        - theta_new (float ndarray of length 2): Scaled parameters of
          the distribution.
        - truncation_limits (float ndarray of length 2 or None):
          Scaled truncation limits for the distribution, or None if no
          truncation is applied.

    Raises
    ------
    ValueError
        If the specified distribution family is unsupported.

    """

    if truncation_limits is not None:
        truncation_limits = truncation_limits * scale_factor

    # undefined family is considered deterministic
    if pd.isna(family):
        family = 'deterministic'

    theta_new = np.full_like(theta, np.nan)
    if family == 'normal':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1]  # because we use cov instead of std

    elif family == 'lognormal':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1]  # because it is log std

    elif family == 'uniform':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1] * scale_factor

    elif family == 'deterministic':
        theta_new[0] = theta[0] * scale_factor

    elif family == 'multilinear_CDF':
        theta_new[0] = theta[0] * scale_factor

    else:
        raise ValueError(f'Unsupported distribution: {family}')

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
    lower: float vector, optional, default: np.nan
        Lower bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from below
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. -numpy.inf) to those dimensions.
    upper: float vector, optional, default: np.nan
        Upper bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from above
        in a subset of the dimensions, use either `None` or assign an infinite
        value (i.e. numpy.inf) to those dimensions.

    Returns
    -------
    tuple
        alpha: float
            Estimate of the probability density within the hyperrectangle.
        eps_alpha: float
            Estimate of the error in the calculated probability density.

    """

    # process the inputs and get the number of dimensions
    mu = np.atleast_1d(mu)
    COV = np.atleast_2d(COV)

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
    eps_alpha, alpha, _ = mvndst(lower, upper, infin, correl)

    return alpha, eps_alpha


def _get_theta(params, inits, dist_list):
    """
    Returns the parameters of the target distributions.

    Uses the parameter values from the optimization algorithm (that are relative
    to the initial values) and the initial values to transform them to the
    parameters of the target distributions.

    Parameters
    ----------
    params: float ndarray, Nx2
      Numpy array containing the parameter values
    inits: float ndarray, Nx2
      Numpy array containing the initial values
    dist_list: list of str
      List of strings containing the names of the distributions.

    Returns
    -------
    Theta
      The estimated parameters.

    Raises
    ------
    ValueError
      If any of the distributions is unsupported.

    """

    theta = np.zeros(inits.shape)

    for i, (params_i, inits_i, dist_i) in enumerate(zip(params, inits, dist_list)):
        if dist_i in {'normal', 'lognormal'}:
            # Note that the standard deviation is fit in log space, hence the
            # unusual-looking transformation here
            sig = np.exp(np.log(inits_i[1]) + params_i[1])

            # The mean uses the standard transformation
            mu = inits_i[0] + params_i[0] * sig

            theta[i, 0] = mu
            theta[i, 1] = sig

        else:
            raise ValueError(f'Unsupported distribution: {dist_i}')

    return theta


def _get_limit_probs(limits, distribution, theta):
    """
    Get the CDF value at the specified limits.

    Parameters
    ----------
    limits: float ndarray
      The limits on which to return the CDF value.
    distribution: str
      The distribution to be used.
    theta: float ndarray
      The parameters of the specified distribution.

    Returns
    -------
    tuple
      The CDF values.

    Raises
    ------
    ValueError
      If any of the distributions is unsupported.

    """

    if distribution in {'normal', 'normal-stdev', 'lognormal'}:
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

    else:
        raise ValueError(f'Unsupported distribution: {distribution}')

    return p_a, p_b


def _get_std_samples(samples, theta, tr_limits, dist_list):
    """
    Transform samples to standard normal space.

    Parameters
    ----------
    samples: float ndarray DxN
      2D array of samples. Each row represents a sample.
    theta: float ndarray Dx2
      2D array of theta values that represent each dimension of the
      samples
    tr_limits: float ndarray Dx2
      2D array with rows that represent [a, b] pairs of truncation
      limits
    dist_list: str ndarray of length D
      1D array containing the names of the distributions

    Returns
    -------
    ndarray
      float DxN ndarray of the samples transformed to standard normal
      space, with each row representing a transformed sample in
      standard normal space.

    Raises
    ------
    ValueError
      If any of the distributions is unsupported.

    """

    std_samples = np.zeros(samples.shape)

    for i, (samples_i, theta_i, tr_lim_i, dist_i) in enumerate(
        zip(samples, theta, tr_limits, dist_list)
    ):
        if dist_i in {'normal', 'normal-stdev', 'lognormal'}:
            lim_low = tr_lim_i[0]
            lim_high = tr_lim_i[1]

            if (
                True in (samples_i > lim_high).tolist()
                or True in (samples_i < lim_low).tolist()
            ):
                raise ValueError(
                    'One or more sample values lie outside '
                    'of the specified truncation limits.'
                )

            # first transform from normal to uniform
            uni_samples = norm.cdf(samples_i, loc=theta_i[0], scale=theta_i[1])

            # replace 0 and 1 values with the nearest float
            uni_samples[uni_samples == 0] = np.nextafter(0, 1)
            uni_samples[uni_samples == 1] = np.nextafter(1, -1)

            # consider truncation if needed
            p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
            uni_samples = (uni_samples - p_a) / (p_b - p_a)

            # then transform from uniform to standard normal
            std_samples[i] = norm.ppf(uni_samples, loc=0.0, scale=1.0)

        else:
            raise ValueError(f'Unsupported distribution: {dist_i}')

    return std_samples


def _get_std_corr_matrix(std_samples):
    """
    Estimate the correlation matrix of the given standard normal
    samples. Ensure that the correlation matrix is positive
    semidefinite.

    Parameters
    ----------
    std_samples: float ndarray, DxN
      Array containing the standard normal samples. Each column is a
      sample. It should not contain Inf or NaN values.

    Returns
    -------
    ndarray
      Correlation matrix. float ndarray, DxD

    Raises
    ------
    ValueError
      If any of the elements of std_samples is np.inf or np.nan

    """

    if True in np.isinf(std_samples) or True in np.isnan(std_samples):
        raise ValueError('std_samples array must not contain inf or NaN values')

    n_dims, n_samples = std_samples.shape

    # initialize the correlation matrix estimate
    rho_hat = np.zeros((n_dims, n_dims))
    np.fill_diagonal(rho_hat, 1.0)

    # take advantage of having standard normal samples
    for dim_i in range(n_dims):
        for dim_j in np.arange(dim_i + 1, n_dims):
            rho_hat[dim_i, dim_j] = (
                np.sum(std_samples[dim_i] * std_samples[dim_j]) / n_samples
            )
            rho_hat[dim_j, dim_i] = rho_hat[dim_i, dim_j]

    # make sure rho_hat is positive semidefinite
    try:
        cholesky(rho_hat, lower=True)  # if this works, we're good

    # otherwise, we can try to fix the matrix using SVD
    except np.linalg.LinAlgError:
        try:
            U, s, _ = svd(
                rho_hat,
            )

        except np.linalg.LinAlgError:
            # if this also fails, we give up
            return None

        S = np.diagflat(s)

        rho_hat = U @ S @ U.T
        np.fill_diagonal(rho_hat, 1.0)

        # check if we introduced any unreasonable values
        if (np.max(rho_hat) > 1.01) or (np.min(rho_hat) < -1.01):
            return None

        # round values to 1.0 and -1.0, if needed
        if np.max(rho_hat) > 1.0:
            rho_hat /= np.max(rho_hat)

        if np.min(rho_hat) < -1.0:
            rho_hat /= np.abs(np.min(rho_hat))

    return rho_hat


def _mvn_scale(x, rho):
    """
    Scaling utility function

    Parameters
    ----------
    x: ndarray
      Input array
    rho: ndarray
      Covariance matrix

    Returns
    -------
    ndarray
      Scaled values

    """
    x = np.atleast_2d(x)
    n_dims = x.shape[1]

    # create an uncorrelated covariance matrix
    rho_0 = np.eye(n_dims, n_dims)

    a = mvn.pdf(x, mean=np.zeros(n_dims), cov=rho_0)
    a[a < 1.0e-10] = 1.0e-10

    b = mvn.pdf(x, mean=np.zeros(n_dims), cov=rho)

    return b / a


def _neg_log_likelihood(
    params,
    inits,
    bnd_lower,
    bnd_upper,
    samples,
    dist_list,
    tr_limits,
    det_limits,
    censored_count,
    enforce_bounds=False,
):
    """
    Calculate the negative log likelihood of the given data samples
    given the parameter values and distribution information.

    This function is used as an objective function in optimization
    algorithms to estimate the parameters of the distribution of the
    input data.

    Parameters
    ----------
    params : ndarray
        1D array with the parameter values to be assessed.
    inits : ndarray
        1D array with the initial estimates for the distribution
        parameters.
    bnd_lower : ndarray
        1D array with the lower bounds for the distribution
        parameters.
    bnd_upper : ndarray
        1D array with the upper bounds for the distribution
        parameters.
    samples : ndarray
        2D array with the data samples. Each column corresponds to a
        different random variable.
    dist_list : list
        List with the distribution types for each random variable.
    tr_limits : list
        List with the truncation limits for each random variable.
    det_limits : list
        List with the detection limits for each random variable.
    censored_count : int
        Number of censored samples in the data.
    enforce_bounds : bool, optional
        If True, the parameters are only considered valid if they are
        within the bounds defined by bnd_lower and bnd_upper. The
        default value is False.

    Returns
    -------
    float
        The negative log likelihood of the data given the distribution parameters.
    """

    # First, check if the parameters are within the pre-defined bounds
    # TODO: check if it is more efficient to use a bounded minimization algo
    if enforce_bounds:
        if not ((params > bnd_lower) & (params < bnd_upper)).all(0):
            # if they are not, then return a large value to discourage the
            # optimization algorithm from going in that direction
            return 1e10

    # If there is nan in params, return a large value
    if np.isnan(np.sum(params)):
        return 1e10

    params = np.reshape(params, inits.shape)
    n_dims, _ = samples.shape

    theta = _get_theta(params, inits, dist_list)

    likelihoods = np.zeros(samples.shape)

    # calculate the marginal likelihoods
    for i, (theta_i, samples_i, tr_lim_i, dist_i) in enumerate(
        zip(theta, samples, tr_limits, dist_list)
    ):
        # consider truncation if needed
        p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
        # this is the probability mass within the
        # truncation limits
        tr_alpha = p_b - p_a

        # Calculate the likelihood for each available sample
        # Note that we are performing this without any transformation to be able
        # to respect truncation limits
        if dist_i in {'normal', 'lognormal'}:
            likelihoods[i] = (
                norm.pdf(samples_i, loc=theta_i[0], scale=theta_i[1]) / tr_alpha
            )

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
            zip(theta, tr_limits, det_limits, dist_list)
        ):
            # prepare the standardized truncation and detection limits
            p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
            p_l, p_u = _get_limit_probs(det_lim_i, dist_i, theta_i)

            # rescale detection limits to consider truncation
            p_l, p_u = [np.min([np.max([lim, p_a]), p_b]) for lim in (p_l, p_u)]
            p_l, p_u = [(lim - p_a) / (p_b - p_a) for lim in (p_l, p_u)]

            # transform limits to standard normal space
            det_lower[i], det_upper[i] = norm.ppf([p_l, p_u], loc=0.0, scale=1.0)

        # get the likelihood of getting a non-censored sample given the
        # detection limits and the correlation matrix
        det_alpha, eps_alpha = mvn_orthotope_density(
            np.zeros(n_dims), rho_hat, det_lower, det_upper
        )

        # Make sure det_alpha is estimated with sufficient accuracy
        if det_alpha <= 100.0 * eps_alpha:
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
    scale = _mvn_scale(std_samples.T, rho_hat)
    # TODO: We can almost surely replace the product of likelihoods with a call
    # to mvn()
    likelihoods = np.prod(likelihoods, axis=0) * scale

    # Zeros are a result of limited floating point precision. Replace them
    # with the smallest possible positive floating point number to
    # improve convergence.
    likelihoods = np.clip(likelihoods, a_min=np.nextafter(0, 1), a_max=None)

    # calculate the total negative log likelihood
    NLL = -(
        np.sum(np.log(likelihoods))  # from samples
        + censored_count * np.log(cen_likelihood)
    )  # censoring influence

    # normalize the NLL with the sample count
    NLL = NLL / samples.size

    # print(theta[0], params, NLL)

    return NLL


def fit_distribution_to_sample(
    raw_samples,
    distribution,
    truncation_limits=(np.nan, np.nan),
    censored_count=0,
    detection_limits=(np.nan, np.nan),
    multi_fit=False,
    logger_object=None,
):
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
    logger_object:
        Logging object to be used. If no object is specified, no
        logging is performed.

    Returns
    -------
    tuple
        theta: float ndarray
            Estimates of the parameters of the fitted probability
            distribution in each dimension. The following parameters
            are returned for the supported distributions: normal -
            mean, coefficient of variation; lognormal - median, log
            standard deviation;
        Rho: float 2D ndarray, optional
            In the multivariate case, returns the estimate of the
            correlation matrix.

    Raises
    ------
    ValueError
        If NaN values are produced during standard normal space transformation

    """

    samples = np.atleast_2d(raw_samples)
    tr_limits = np.atleast_2d(truncation_limits)
    det_limits = np.atleast_2d(detection_limits)
    dist_list = np.atleast_1d(distribution)
    n_dims, n_samples = samples.shape

    if (tr_limits.shape[0] == 1) and (n_dims != 1):
        tr_limits = np.tile(tr_limits[0], (n_dims, 1))

    if (det_limits.shape[0] == 1) and (n_dims != 1):
        det_limits = np.tile(det_limits[0], (n_dims, 1))

    if (dist_list.shape[0] == 1) and (n_dims != 1):
        dist_list = np.tile(dist_list[0], n_dims)

    # Convert samples and limits to log space if the distribution is lognormal
    for d_i, distr in enumerate(dist_list):
        if distr == 'lognormal':
            samples[d_i] = np.log(samples[d_i])

            for lim in range(2):
                if not np.isnan(tr_limits[d_i][lim]):
                    tr_limits[d_i][lim] = np.log(tr_limits[d_i][lim])

            for lim in range(2):
                if not np.isnan(det_limits[d_i][lim]):
                    det_limits[d_i][lim] = np.log(det_limits[d_i][lim])

    # Define initial values of distribution parameters
    # Initialize arrays
    mu_init = np.ones(n_dims) * np.nan
    sig_init = np.ones_like(mu_init) * np.nan

    for d_i, distr in enumerate(dist_list):
        if distr in {'normal', 'normal-stdev', 'lognormal'}:
            # use the first two moments
            mu_init[d_i] = np.mean(samples[d_i])

            if n_samples == 1:
                sig_init[d_i] = 0.0
            else:
                sig_init[d_i] = np.std(samples[d_i])

    # replace zero standard dev with negligible standard dev
    sig_zero_id = np.where(sig_init == 0.0)[0]
    sig_init[sig_zero_id] = 1e-6 * np.abs(mu_init[sig_zero_id]) + np.nextafter(0, 1)

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

    # There is nothing to gain from a time-consuming optimization if..
    #     the number of samples is too small
    if (n_samples < 3) or (
        # there are no truncation or detection limits involved
        np.all(np.isnan(tr_limits))
        and np.all(np.isnan(det_limits))
    ):
        # In this case, it is typically hard to improve on the method of
        # moments estimates for the parameters of the marginal distributions
        theta = inits

    # Otherwise, we run the optimization that aims to find the parameters that
    # maximize the likelihood of observing the samples
    else:
        # First, optimize for each marginal independently
        for dim in range(n_dims):
            inits_i = inits[dim : dim + 1]

            # Censored samples are only considered in the following step, but
            # we fit a truncated distribution if there are censored samples to
            # make it easier to fit the censored distribution later.
            tr_limits_i = [np.nan, np.nan]
            for lim in range(2):
                if (np.isnan(tr_limits[dim][lim])) and (
                    not np.isnan(det_limits[dim][lim])
                ):
                    tr_limits_i[lim] = det_limits[dim][lim]
                elif not np.isnan(det_limits[dim][lim]):
                    if lim == 0:
                        tr_limits_i[lim] = np.min(
                            [tr_limits[dim][lim], det_limits[dim][lim]]
                        )
                    elif lim == 1:
                        tr_limits_i[lim] = np.max(
                            [tr_limits[dim][lim], det_limits[dim][lim]]
                        )
                else:
                    tr_limits_i[lim] = tr_limits[dim][lim]

            out_m_i = minimize(
                _neg_log_likelihood,
                np.zeros(inits[dim].size),
                args=(
                    inits_i,
                    bnd_lower[dim],
                    bnd_upper[dim],
                    samples[dim : dim + 1],
                    [
                        dist_list[dim],
                    ],
                    [
                        tr_limits_i,
                    ],
                    [np.nan, np.nan],
                    0,
                    True,
                ),
                method='BFGS',
                options={'maxiter': 50},
            )

            out = out_m_i.x.reshape(inits_i.shape)
            theta = _get_theta(
                out,
                inits_i,
                [
                    dist_list[dim],
                ],
            )
            inits[dim] = theta[0]

        # Second, if multi_fit is requested or there are censored samples,
        # we attempt the multivariate fitting using the marginal results as
        # initial parameters.
        if multi_fit or (censored_count > 0):
            bnd_lower = bnd_lower.flatten()
            bnd_upper = bnd_upper.flatten()

            out_m = minimize(
                _neg_log_likelihood,
                np.zeros(inits.size),
                args=(
                    inits,
                    bnd_lower,
                    bnd_upper,
                    samples,
                    dist_list,
                    tr_limits,
                    det_limits,
                    censored_count,
                    True,
                ),
                method='BFGS',
                options={'maxiter': 50},
            )

            out = out_m.x.reshape(inits.shape)
            theta = _get_theta(out, inits, dist_list)

        else:
            theta = inits

    # Calculate rho in the standard normal space because we will generate new
    # samples using that type of correlation (i.e., Gaussian copula)
    std_samples = _get_std_samples(samples, theta, tr_limits, dist_list)
    if True in np.isnan(std_samples) or True in np.isinf(std_samples):
        raise ValueError(
            'Something went wrong.'
            '\n'
            'Conversion to standard normal space was unsuccessful. \n'
            'The given samples might deviate '
            'substantially from the specified distribution.'
        )
    rho_hat = _get_std_corr_matrix(std_samples)
    if rho_hat is None:
        # If there is not enough data to produce a valid correlation matrix
        # estimate, we assume uncorrelated demands
        rho_hat = np.zeros((n_dims, n_dims))
        np.fill_diagonal(rho_hat, 1.0)

        if logger_object:
            logger_object.msg(
                "\nWARNING: Demand sample size too small to reliably estimate "
                "the correlation matrix. Assuming uncorrelated demands.",
                prepend_timestamp=False,
                prepend_blank_space=False,
            )
        else:
            print(
                "\nWARNING: Demand sample size too small to reliably estimate "
                "the correlation matrix. Assuming uncorrelated demands."
            )

    for d_i, distr in enumerate(dist_list):
        # Convert mean back to linear space if the distribution is lognormal
        if distr == 'lognormal':
            theta[d_i][0] = np.exp(theta[d_i][0])
            # theta_mod = theta.T.copy()
            # theta_mod[0] = np.exp(theta_mod[0])
            # theta = theta_mod.T
        # Convert the std to cov if the distribution is normal
        elif distr == 'normal':
            # replace standard deviation with coefficient of variation
            # note: this results in cov=inf if the mean is zero.
            if np.abs(theta[d_i][0]) < 1.0e-40:
                theta[d_i][1] = np.inf
            else:
                theta[d_i][1] = theta[d_i][1] / np.abs(theta[d_i][0])

    return theta, rho_hat


def _OLS_percentiles(params, values, perc, family):
    """
    Estimate percentiles using ordinary least squares (OLS).

    Parameters
    ----------
    params : tuple of floats
        The parameters of the selected distribution family.
    values : float ndarray
        The sample values for which the percentiles are requested.
    perc : float ndarray
        The requested percentile(s).
    family : str
        The distribution family to use for the percentile estimation.
        Can be either 'normal' or 'lognormal'.

    Returns
    -------
    float
        The sum of the squared errors between the estimated and actual values.

    Raises
    ------
    ValueError
        If `family` is not 'normal' or 'lognormal'.

    """

    if family == 'normal':
        theta_0 = params[0]
        theta_1 = params[1]

        if theta_1 <= 0:
            return 1e10

        val_hat = norm.ppf(perc, loc=theta_0, scale=theta_1)

    elif family == 'lognormal':
        theta_0 = params[0]
        theta_1 = params[1]

        if theta_0 <= 0:
            return 1e10

        if theta_1 <= 0:
            return 1e10

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
    tuple
        family: string
            The optimal choice of family among the provided list of
            families
        theta: array of float
            Parameters of the fitted distribution.

    """

    out_list = []

    percentiles = np.array(percentiles)

    median_id = np.argmin(np.abs(percentiles - 0.5))
    extreme_id = np.argmax(percentiles - 0.5)

    for family in families:
        inits = [
            values[median_id],
        ]

        if family == 'normal':
            inits.append(
                (
                    np.abs(values[extreme_id] - inits[0])
                    / np.abs(norm.ppf(percentiles[extreme_id], loc=0, scale=1))
                )
            )

        elif family == 'lognormal':
            inits.append(
                (
                    np.abs(np.log(values[extreme_id] / inits[0]))
                    / np.abs(norm.ppf(percentiles[extreme_id], loc=0, scale=1))
                )
            )

        out_list.append(
            minimize(
                _OLS_percentiles,
                inits,
                args=(values, percentiles, family),
                method='BFGS',
            )
        )

    best_out_id = np.argmin([out.fun for out in out_list])

    return families[best_out_id], out_list[best_out_id].x


class BaseRandomVariable(ABC):
    """
    Base abstract class for different types of random variables.

    """

    def __init__(
        self,
        name,
        f_map=None,
        anchor=None,
    ):
        """
        Initializes a RandomVariable object.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        f_map: function, optional
            A user-defined function that is applied on the realizations before
            returning a sample.
        anchor: RandomVariable, optional
            Anchors this to another variable. If the anchor is not None, this
            variable will be perfectly correlated with its anchor. Note that
            the attributes of this variable and its anchor do not have to be
            identical.

        Raises
        ------
        ValueError
            If there are issues with the specified distribution theta
            parameters.

        """

        self.name = name
        self.distribution = None
        self.f_map = f_map
        self._uni_samples = None
        self.RV_set = None
        self._sample_DF = None
        self._sample = None
        if anchor is None:
            self.anchor = self
        else:
            self.anchor = anchor

    @property
    def sample(self):
        """
        Return the empirical or generated sample.

        Returns
        -------
        ndarray
          The empirical or generated sample.

        """
        if self.f_map is not None:
            return self.f_map(self._sample)
        return self._sample

    @sample.setter
    def sample(self, value):
        """
        Assign a sample to the random variable.

        Parameters
        ----------
        value: ndarray
          Sample to assign

        """
        self._sample = value
        self._sample_DF = pd.Series(value)

    @property
    def sample_DF(self):
        """
        Return the empirical or generated sample in a pandas Series.

        Returns
        -------
        ndarray
          The empirical or generated sample in a pandas Series.

        """
        if self.f_map is not None:
            return self._sample_DF.apply(self.f_map)

        return self._sample_DF

    @property
    def uni_sample(self):
        """
        Return the sample from the controlling uniform distribution.

        Returns
        -------
        ndarray
          The sample from the controlling uniform distribution.

        """
        return self.anchor._uni_samples

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


class RandomVariable(BaseRandomVariable):
    """
    Random variable that needs `values` in `inverse_transform`
    """

    @abstractmethod
    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        """
        Instantiates a normal random variable.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        theta: 2-element float ndarray
          Set of parameters that define the Cumulative Distribution
          Function (CDF) of the variable: Mean, coefficient of
          variation.
        truncation_limits: float ndarray, optional
          Defines the np.array((a, b)) truncation limits for the
          distribution. Use np.nan to assign no limit in one direction,
          like so: np.array((a, np.nan)), or np.array((np.nan, b)).
        f_map: function, optional
            A user-defined function that is applied on the realizations before
            returning a sample.
        anchor: RandomVariable, optional
            Anchors this to another variable. If the anchor is not None, this
            variable will be perfectly correlated with its anchor. Note that
            the attributes of this variable and its anchor do not have to be
            identical.

        """
        super().__init__(
            name=name,
            f_map=f_map,
            anchor=anchor,
        )

    @abstractmethod
    def inverse_transform(self, values):
        """
        Uses inverse probability integral transformation on the
        provided values.

        """

    def inverse_transform_sampling(self):
        """
        Creates a sample using inverse probability integral
        transformation.

        Raises
        ------
        ValueError
          If there is no available uniform sample.
        """
        if self.uni_sample is None:
            raise ValueError('No available uniform sample.')
        self.sample = self.inverse_transform(self.uni_sample)


class UtilityRandomVariable(BaseRandomVariable):
    """
    Random variable that needs `sample_size` in `inverse_transform`
    """

    @abstractmethod
    def __init__(
        self,
        name,
        f_map=None,
        anchor=None,
    ):
        """
        Instantiates a normal random variable.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        f_map: function, optional
            A user-defined function that is applied on the realizations before
            returning a sample.
        anchor: RandomVariable, optional
            Anchors this to another variable. If the anchor is not None, this
            variable will be perfectly correlated with its anchor. Note that
            the attributes of this variable and its anchor do not have to be
            identical.

        """
        super().__init__(
            name=name,
            f_map=f_map,
            anchor=anchor,
        )

    @abstractmethod
    def inverse_transform(self, sample_size):
        """
        Uses inverse probability integral transformation on the
        provided values.

        """

    def inverse_transform_sampling(self, sample_size):
        """
        Creates a sample using inverse probability integral
        transformation.
        """
        self.sample = self.inverse_transform(sample_size)


class NormalRandomVariable(RandomVariable):
    """
    Normal random variable.

    """

    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'normal'
        self.theta = np.atleast_1d(theta)
        self.truncation_limits = truncation_limits

    def cdf(self, values):
        """
        Returns the Cumulative Density Function (CDF) at the specified
        values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          1D float ndarray containing CDF values

        """
        mu, cov = self.theta[:2]
        sig = np.abs(mu) * cov

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

            if np.isnan(a):
                a = -np.inf
            if np.isnan(b):
                b = np.inf

            p_a, p_b = [norm.cdf((lim - mu) / sig) for lim in (a, b)]

            # cap the values at the truncation limits
            values = np.minimum(np.maximum(values, a), b)

            # get the cdf from a non-truncated normal
            p_vals = norm.cdf(values, loc=mu, scale=sig)

            # adjust for truncation
            result = (p_vals - p_a) / (p_b - p_a)

        else:
            result = norm.cdf(values, loc=mu, scale=sig)

        return result

    def inverse_transform(self, values):
        """
        Evaluates the inverse of the Cumulative Density Function (CDF)
        for the given values. Used to generate random variable
        realizations.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the inverse CDF

        Returns
        -------
        ndarray
          Inverse CDF values

        Raises
        ------
        ValueError
          If the probability massss within the truncation limits is
          too small

        """

        mu, cov = self.theta[:2]
        sig = np.abs(mu) * cov

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

            if np.isnan(a):
                a = -np.inf
            if np.isnan(b):
                b = np.inf

            p_a, p_b = [norm.cdf((lim - mu) / sig) for lim in (a, b)]

            if p_b - p_a == 0:
                raise ValueError(
                    "The probability mass within the truncation limits is "
                    "too small and the truncated distribution cannot be "
                    "sampled with sufficiently high accuracy. This is most "
                    "probably due to incorrect truncation limits set for "
                    "the distribution."
                )

            result = norm.ppf(values * (p_b - p_a) + p_a, loc=mu, scale=sig)

        else:
            result = norm.ppf(values, loc=mu, scale=sig)

        return result


class LogNormalRandomVariable(RandomVariable):
    """
    Lognormal random variable.

    """

    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'lognormal'
        self.theta = np.atleast_1d(theta)
        self.truncation_limits = truncation_limits

    def cdf(self, values):
        """
        Returns the Cumulative Density Function (CDF) at the specified
        values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          CDF values

        """
        theta, beta = self.theta[:2]

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

            if np.isnan(a):
                a = np.nextafter(0, 1)
            if np.isnan(b):
                b = np.inf

            p_a, p_b = [
                norm.cdf((np.log(lim) - np.log(theta)) / beta) for lim in (a, b)
            ]

            # cap the values at the truncation limits
            values = np.minimum(np.maximum(values, a), b)

            # get the cdf from a non-truncated lognormal
            p_vals = norm.cdf(np.log(values), loc=np.log(theta), scale=beta)

            # adjust for truncation
            result = (p_vals - p_a) / (p_b - p_a)

        else:
            values = np.maximum(values, np.nextafter(0, 1))

            result = norm.cdf(np.log(values), loc=np.log(theta), scale=beta)

        return result

    def inverse_transform(self, values):
        """
        Evaluates the inverse of the Cumulative Density Function (CDF)
        for the given values. Used to generate random variable
        realizations.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the inverse CDF

        Returns
        -------
        ndarray
          Inverse CDF values

        """

        theta, beta = self.theta[:2]

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

            if np.isnan(a):
                a = np.nextafter(0, 1)
            else:
                a = np.maximum(np.nextafter(0, 1), a)

            if np.isnan(b):
                b = np.inf

            p_a, p_b = [
                norm.cdf((np.log(lim) - np.log(theta)) / beta) for lim in (a, b)
            ]

            result = np.exp(
                norm.ppf(values * (p_b - p_a) + p_a, loc=np.log(theta), scale=beta)
            )

        else:
            result = np.exp(norm.ppf(values, loc=np.log(theta), scale=beta))

        return result


class UniformRandomVariable(RandomVariable):
    """
    Uniform random variable.

    """

    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'uniform'
        self.theta = np.atleast_1d(theta)
        self.truncation_limits = truncation_limits

    def cdf(self, values):
        """
        Returns the Cumulative Density Function (CDF) at the specified
        values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          CDF values

        """
        a, b = self.theta[:2]

        if np.isnan(a):
            a = -np.inf
        if np.isnan(b):
            b = np.inf

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

        result = uniform.cdf(values, loc=a, scale=(b - a))

        return result

    def inverse_transform(self, values):
        """
        Evaluates the inverse of the Cumulative Density Function (CDF)
        for the given values. Used to generate random variable
        realizations.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the inverse CDF

        Returns
        -------
        ndarray
          Inverse CDF values

        """
        a, b = self.theta[:2]

        if np.isnan(a):
            a = -np.inf
        if np.isnan(b):
            b = np.inf

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

        result = uniform.ppf(values, loc=a, scale=(b - a))

        return result


class MultilinearCDFRandomVariable(RandomVariable):
    """
    Multilinear CDF random variable. This RV is defined by specifying
    the points that define its Cumulative Density Function (CDF), and
    linear interpolation between them.

    """

    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'multilinear_CDF'

        if not np.all(np.isnan(truncation_limits)):
            raise NotImplementedError(
                f'{self.distribution} RVs do not support truncation'
            )

        y_1 = theta[0, 1]
        if y_1 != 0.00:
            raise ValueError(
                "For multilinear CDF random variables, y_1 should be set to 0.00"
            )
        y_n = theta[-1, 1]
        if y_n != 1.00:
            raise ValueError(
                "For multilinear CDF random variables, y_n should be set to 1.00"
            )

        x_s = theta[:, 0]
        if not np.array_equal(np.sort(x_s), x_s):
            raise ValueError(
                "For multilinear CDF random variables, "
                "Xs should be specified in ascending order"
            )
        if np.any(np.isclose(np.diff(x_s), 0.00)):
            raise ValueError(
                "For multilinear CDF random variables, "
                "Xs should be specified in strictly ascending order"
            )

        y_s = theta[:, 1]
        if not np.array_equal(np.sort(y_s), y_s):
            raise ValueError(
                "For multilinear CDF random variables, "
                "Ys should be specified in ascending order"
            )

        if np.any(np.isclose(np.diff(y_s), 0.00)):
            raise ValueError(
                "For multilinear CDF random variables, "
                "Ys should be specified in strictly ascending order"
            )

        self.theta = np.atleast_1d(theta)

    def cdf(self, values):
        """
        Returns the Cumulative Density Function (CDF) at the specified
        values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          CDF values

        """
        x_i = [-np.inf] + [x[0] for x in self.theta] + [np.inf]
        y_i = [0.00] + [x[1] for x in self.theta] + [1.00]

        # Using Numpy's interp for linear interpolation
        result = np.interp(values, x_i, y_i, left=0.00, right=1.00)

        return result

    def inverse_transform(self, values):
        """
        Evaluates the inverse of the Cumulative Density Function (CDF)
        for the given values. Used to generate random variable
        realizations.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the inverse CDF

        Returns
        -------
        ndarray
          Inverse CDF values

        """

        x_i = [x[0] for x in self.theta]
        y_i = [x[1] for x in self.theta]

        # using Numpy's interp for the inverse CDF
        # note: by definition, y_i /has/ to include the values 0.00
        # and 1.00, and `values` have to be in the range [0.00, 1.00],
        # so there is no need to handle edge cases here (i.e.,
        # extrapolate).
        # note: swapping the roles of x_i and y_i for inverse
        # interpolation
        result = np.interp(values, y_i, x_i)

        return result


class EmpiricalRandomVariable(RandomVariable):
    """
    Empirical random variable.

    """

    def __init__(
        self,
        name,
        raw_samples,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        super().__init__(
            name=name,
            theta=raw_samples,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'empirical'
        if not np.all(np.isnan(truncation_limits)):
            raise NotImplementedError(
                f'{self.distribution} RVs do not support truncation'
            )

        self._raw_samples = np.atleast_1d(raw_samples)

    def inverse_transform(self, values):
        """
        Maps given values to their corresponding positions within the
        empirical data array, simulating an inverse transformation
        based on the empirical distribution.  This can be seen as a
        simple form of inverse CDF where values represent normalized
        positions within the empirical data set.

        Parameters
        ----------
        values: 1D float ndarray
          Normalized values between 0 and 1, representing positions
          within the empirical data distribution.

        Returns
        -------
        ndarray
          The empirical data points corresponding to the given
          normalized positions.

        """
        s_ids = (values * len(self._raw_samples)).astype(int)
        result = self._raw_samples[s_ids]
        return result


class CoupledEmpiricalRandomVariable(UtilityRandomVariable):
    """
    Coupled empirical random variable.

    """

    def __init__(
        self,
        name,
        raw_samples,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        """
        Instantiates a coupled empirical random variable.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        raw_samples: 1D float ndarray
          Samples from which to draw empirical realizations.
        truncation_limits: 2D float ndarray
          Not supported for CoupledEmpirical RVs.
          Should be np.array((np.nan, np.nan))
        f_map: function, optional
            A user-defined function that is applied on the realizations before
            returning a sample.
        anchor: RandomVariable, optional
            Anchors this to another variable. If the anchor is not None, this
            variable will be perfectly correlated with its anchor. Note that
            the attributes of this variable and its anchor do not have to be
            identical.

        Raises
        ------
        NotImplementedError
          When truncation limits are provided

        """
        super().__init__(
            name=name,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'coupled_empirical'
        if not np.all(np.isnan(truncation_limits)):
            raise NotImplementedError(
                f'{self.distribution} RVs do not support truncation'
            )

        self._raw_samples = np.atleast_1d(raw_samples)

    def inverse_transform(self, sample_size):
        """
        Generates a new sample array from the existing empirical data
        by repeating the dataset until it matches the requested sample
        size.

        Parameters
        ----------
        sample_size: int
          The desired size of the sample array to be generated. It
          dictates how many times the original dataset will be
          repeated to match or exceed this size, after which the array
          is trimmed to precisely match the requested size.

        Returns
        -------
        ndarray
          A new sample array derived from repeating the original
          dataset.

        """

        raw_sample_count = len(self._raw_samples)
        new_sample = np.tile(
            self._raw_samples, int(sample_size / raw_sample_count) + 1
        )
        result = new_sample[:sample_size]
        return result


class DeterministicRandomVariable(UtilityRandomVariable):
    """
    Deterministic random variable.

    """

    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        """
        Instantiates a deterministic random variable. This behaves
        like a RandomVariable object but represents a specific,
        deterministic value.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        theta: 1-element float ndarray
          The value.
        truncation_limits: 2D float ndarray
          Not supported for Deterministic RVs.
          Should be np.array((np.nan, np.nan))
        f_map: function, optional
            A user-defined function that is applied on the realizations before
            returning a sample.
        anchor: RandomVariable, optional
            Anchors this to another variable. If the anchor is not None, this
            variable will be perfectly correlated with its anchor. Note that
            the attributes of this variable and its anchor do not have to be
            identical.

        Raises
        ------
        NotImplementedError
          When truncation limits are provided

        """
        super().__init__(
            name=name,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'deterministic'
        if not np.all(np.isnan(truncation_limits)):
            raise NotImplementedError(
                f'{self.distribution} RVs do not support truncation'
            )

        self.theta = np.atleast_1d(theta)

    def inverse_transform(self, sample_size):
        """
        Generates samples that correspond to the value.

        Parameters
        ----------
        sample_size: int
          The desired size of the sample array to be generated.

        Returns
        -------
        ndarray
          Sample array containing the deterministic value.

        """

        result = np.full(sample_size, self.theta[0])
        return result


class MultinomialRandomVariable(RandomVariable):
    """
    Multinomial random variable.

    """

    def __init__(
        self,
        name,
        theta,
        truncation_limits=np.array((np.nan, np.nan)),
        f_map=None,
        anchor=None,
    ):
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        if not np.all(np.isnan(truncation_limits)):
            raise NotImplementedError(
                f'{self.distribution} RVs do not support truncation'
            )
        self.distribution = 'multinomial'
        if np.sum(theta) > 1.00:
            raise ValueError(
                f"The set of p values provided for a multinomial "
                f"distribution shall sum up to less than or equal to 1.0. "
                f"The provided values sum up to {np.sum(theta)}. p = "
                f"{theta} ."
            )

        self.theta = np.atleast_1d(theta)

    def inverse_transform(self, values):
        """
        Transforms continuous values into discrete events based
        on the cumulative probabilities of the multinomial
        distribution derived by `theta`.

        Parameters
        ----------
        values: 1D float ndarray
          Continuous values to be transformed into discrete events
          according to the multinomial distribution's cumulative
          probabilities.

        Returns
        -------
        ndarray
          Discrete events corresponding to the input values.

        """
        p_cum = np.cumsum(self.theta)[:-1]

        for i, p_i in enumerate(p_cum):
            values[values < p_i] = 10 + i
        values[values <= 1.0] = 10 + len(p_cum)

        result = values - 10

        return result


class RandomVariableSet:
    """
    Represents a set of random variables, each of which is described
    by its own probability distribution. The set allows the user to
    define correlations between the random variables, and provides
    methods to sample from the correlated variables and estimate
    various statistical properties of the set, such as the probability
    density within a specified range or orthotope.

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
            self._variables = {RV_list[i].name: RV_list[i] for i in reorder}

            # reorder the entries in the correlation matrix to correspond to the
            # sorted list of RVs
            self._Rho = np.asarray(Rho[(reorder)].T[(reorder)].T)

        else:  # if there is only one variable (for testing, probably)
            self._variables = {rv.name: rv for rv in RV_list}
            self._Rho = np.asarray(Rho)

        # assign this RV_set to the variables
        for _, var in self._variables.items():
            var.RV_set = self

    @property
    def RV(self):
        """
        Returns the random variable(s) assigned to the set.

        Returns
        -------
        ndarray
          The random variable(s) assigned to the set.

        """
        return self._variables

    @property
    def size(self):
        """
        Returns the size (i.e., number of variables in the) RV set.

        Returns
        -------
        ndarray
          The size (i.e., number of variables in the) RV set.

        """
        return len(self._variables)

    @property
    def sample(self):
        """
        Returns the sample of the variables in the set.

        Returns
        -------
        ndarray
          The sample of the variables in the set.

        """
        return {name: rv.sample for name, rv in self._variables.items()}

    def Rho(self, var_subset=None):
        """
        Returns the (subset of the) correlation matrix.

        Returns
        -------
        ndarray
          The (subset of the) correlation matrix.

        """
        if var_subset is None:
            return self._Rho
        var_ids = [list(self._variables.keys()).index(var_i) for var_i in var_subset]
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

        except np.linalg.LinAlgError:
            # if the Cholesky doesn't work, we need to use the more
            # time-consuming but more robust approach based on SVD
            N_RV = norm.ppf(U_RV)

            U, s, _ = svd(
                self._Rho,
            )
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
        tuple
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
            variables = list(self._variables.keys())
        else:
            variables = var_subset

        # first, convert limits to standard normal values
        for var_i, var_name in enumerate(variables):
            var = self._variables[var_name]

            if (np.any(~np.isnan(lower))) and (~np.isnan(lower[var_i])):
                lower_std[var_i] = norm.ppf(var.cdf(lower[var_i]), loc=0, scale=1)

            if (np.any(~np.isnan(upper))) and (~np.isnan(upper[var_i])):
                upper_std[var_i] = norm.ppf(var.cdf(upper[var_i]), loc=0, scale=1)

        # then calculate the orthotope results in std normal space
        lower_std = lower_std.T
        upper_std = upper_std.T

        OD = [
            mvn_orthotope_density(
                mu=np.zeros(len(variables)),
                COV=self.Rho(var_subset),
                lower=l_i,
                upper=u_i,
            )[0]
            for l_i, u_i in zip(lower_std, upper_std)
        ]

        return np.asarray(OD)


class RandomVariableRegistry:
    """
    Description

    Parameters
    ----------

    """

    def __init__(self, rng):
        """
        rng: numpy.random._generator.Generator
            Random variable generator object.
            e.g.: np.random.default_rng(seed)
        """
        self._rng = rng
        self._variables = {}
        self._sets = {}

    @property
    def RV(self):
        """
        Returns all random variable(s) in the registry.

        Returns
        -------
        dict
          all random variable(s) in the registry.

        """
        return self._variables

    def RVs(self, keys):
        """
        Returns a subset of the random variables in the registry

        Parameters
        ----------
        keys: list of str
          Keys that define the subset.

        Returns
        -------
        dict
          A subset random variable(s) in the registry.

        """
        return {name: self._variables[name] for name in keys}

    def add_RV(self, RV):
        """
        Add a new random variable to the registry.

        Raises
        ------
        ValueError
          When the RV already exists in the registry

        """
        if RV.name in self._variables:
            raise ValueError(f'RV {RV.name} already exists in the registry.')
        self._variables.update({RV.name: RV})

    @property
    def RV_set(self):
        """
        Return the random variable set(s) in the registry.

        Returns
        -------
        dict
          The random variable set(s) in the registry.

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
        Return the sample for every random variable in the registry.

        Returns
        -------
        dict
          The sample for every random variable in the registry.

        """
        return {name: rv.sample for name, rv in self.RV.items()}

    def generate_sample(self, sample_size, method):
        """
        Generates samples for all variables in the registry.

        Parameters
        ----------

        sample_size: int
            The number of samples requested per variable.
        method: str
            Can be any of: 'MonteCarlo', 'LHS', 'LHS_midpoint'
            The sample generation method to use. 'MonteCarlo' stands for
            conventional random sampling; 'LHS' is Latin HyperCube Sampling
            with random sample location within each bin of the hypercube;
            'LHS_midpoint' is like LHS, but the samples are assigned to the
            midpoints of the hypercube bins.

        Raises
        ------
        NotImplementedError
          When the RV parent class is Unknown

        """

        # Generate a dictionary with IDs of the free (non-anchored and
        # non-deterministic) variables
        RV_list = [
            RV_name
            for RV_name, RV in self.RV.items()
            if (
                (RV.anchor == RV)
                or (RV.distribution in {'deterministic', 'coupled_empirical'})
            )
        ]
        RV_ID = {RV_name: ID for ID, RV_name in enumerate(RV_list)}
        RV_count = len(RV_ID)

        # Generate controlling samples from a uniform distribution for free RVs
        if 'LHS' in method:
            bin_low = np.array(
                [self._rng.permutation(sample_size) for i in range(RV_count)]
            )

            if method == 'LHS_midpoint':
                U_RV = np.ones([RV_count, sample_size]) * 0.5
                U_RV = (bin_low + U_RV) / sample_size

            elif method == 'LHS':
                U_RV = self._rng.random(size=[RV_count, sample_size])
                U_RV = (bin_low + U_RV) / sample_size

        elif method == 'MonteCarlo':
            U_RV = self._rng.random(size=[RV_count, sample_size])

        # Assign the controlling samples to the RVs
        for RV_name, RV_id in RV_ID.items():
            self.RV[RV_name].uni_sample = U_RV[RV_id]

        # Apply correlations for the pre-defined RV sets
        for RV_set in self.RV_set.values():
            # prepare the correlated uniform distribution for the set
            RV_set.apply_correlation()

        # Convert from uniform to the target distribution for every RV
        for RV in self.RV.values():
            if RV.__class__.__mro__[1] is RandomVariable:
                # no sample size needed, since that information is
                # available in the uniform sample
                RV.inverse_transform_sampling()
            elif RV.__class__.__mro__[1] is UtilityRandomVariable:
                RV.inverse_transform_sampling(sample_size)
            else:
                raise NotImplementedError('Unknown RV parent class.')


def rv_class_map(distribution_name):
    """
    Maps convenient distribution names to their corresponding random
    variable class.

    Parameters
    ----------
    distribution_name: str
      The name of a distribution.

    Returns
    -------
    RandomVariable
      RandomVariable class.

    Raises
    ------
    ValueError
      If the given distribution name does not correspond to a
      distribution class.


    """
    if pd.isna(distribution_name):
        distribution_name = 'deterministic'
    distribution_map = {
        'normal': NormalRandomVariable,
        'lognormal': LogNormalRandomVariable,
        'uniform': UniformRandomVariable,
        'multilinear_CDF': MultilinearCDFRandomVariable,
        'empirical': EmpiricalRandomVariable,
        'coupled_empirical': CoupledEmpiricalRandomVariable,
        'deterministic': DeterministicRandomVariable,
        'multinomial': MultinomialRandomVariable,
    }
    if distribution_name not in distribution_map:
        raise ValueError(f'Unsupported distribution: {distribution_name}')
    return distribution_map[distribution_name]
