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

"""Constants, classes and methods for uncertainty quantification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import colorama
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, svd  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.stats import multivariate_normal as mvn  # type: ignore
from scipy.stats import norm, uniform, weibull_min  # type: ignore
from scipy.stats._mvn import (
    mvndst,  # type: ignore # noqa: PLC2701
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pelicun.base import Logger

colorama.init()
FIRST_POSITIVE_NUMBER = np.nextafter(0, 1)


def scale_distribution(
    scale_factor: float,
    family: str,
    theta: np.ndarray,
    truncation_limits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Scale parameters of a random distribution.

    Parameters
    ----------
    scale_factor: float
        Value by which to scale the parameters.
    family: {'normal' (or 'normal_cov'), 'normal_std', 'lognormal',
        'uniform'}
        Defines the type of probability distribution for the random
        variable.
    theta: float ndarray of length 2
        Set of parameters that define the cumulative distribution
        function of the variable given its distribution type. See the
        expected parameters explained in the RandomVariable
        class. Each parameter can be defined by one or more values. If
        a set of values are provided for one parameter, they define
        ordinates of a multilinear function that is used to get the
        parameter values given an independent variable.
    truncation_limits: float ndarray of length 2, default: None
        Defines the [a,b] truncation limits for the distribution. Use
        None to assign no limit in one direction.

    Returns
    -------
    tuple
        A tuple containing the scaled parameters and truncation
        limits:

        * theta_new (float ndarray of length 2): Scaled parameters of
          the distribution.
        * truncation_limits (float ndarray of length 2 or None):
          Scaled truncation limits for the distribution, or None if no
          truncation is applied.

    Raises
    ------
    ValueError
        If the specified distribution family is unsupported.

    """
    if truncation_limits is not None:
        truncation_limits = truncation_limits.copy()
        truncation_limits *= scale_factor

    # undefined family is considered deterministic
    if pd.isna(family):
        family = 'deterministic'

    theta_new = np.full_like(theta, np.nan)
    if family == 'normal_std':
        theta_new[0] = theta[0] * scale_factor  # mean
        theta_new[1] = theta[1] * scale_factor  # STD

    elif family in {'normal', 'normal_cov'}:
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1]  # because it is CoV

    elif family == 'lognormal':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1]  # because it is log std

    elif family == 'uniform':
        theta_new[0] = theta[0] * scale_factor
        theta_new[1] = theta[1] * scale_factor

    elif family in {'deterministic', 'multilinear_CDF'}:
        theta_new[0] = theta[0] * scale_factor

    else:
        msg = f'Unsupported distribution: {family}'
        raise ValueError(msg)

    return theta_new, truncation_limits


def mvn_orthotope_density(
    mu: float | np.ndarray,
    cov: np.ndarray,
    lower: float | np.ndarray = np.nan,
    upper: float | np.ndarray = np.nan,
) -> tuple[float, float]:
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
    cov: float ndarray
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
    cov = np.atleast_2d(cov)

    sig = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sig, sig)

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
    correl = np.array([0.0]) if ndim == 1 else corr[np.tril_indices(ndim, -1)]

    # estimate the density
    eps_alpha, alpha, _ = mvndst(lower, upper, infin, correl)

    return alpha, eps_alpha


def _get_theta(
    params: np.ndarray, inits: np.ndarray, dist_list: np.ndarray
) -> np.ndarray:
    """
    Return the parameters of the target distributions.

    Uses the parameter values from the optimization algorithm (that
    are relative to the initial values) and the initial values to
    transform them to the parameters of the target distributions.

    Parameters
    ----------
    params: float ndarray, Nx2
      Numpy array containing the parameter values.
    inits: float ndarray, Nx2
      Numpy array containing the initial values.
    dist_list: str ndarray
      Array of strings containing the names of the distributions.

    Returns
    -------
    theta: float ndarray
      The estimated parameters.

    Raises
    ------
    ValueError
      If any of the distributions is unsupported.

    """
    theta = np.zeros(inits.shape)

    for i, (params_i, inits_i, dist_i) in enumerate(zip(params, inits, dist_list)):
        if dist_i in {'normal', 'normal_std', 'lognormal'}:
            # Standard deviation is used directly for 'normal' and
            # 'lognormal'
            sig = (
                np.exp(np.log(inits_i[1]) + params_i[1])
                if dist_i == 'lognormal'
                else inits_i[1] + params_i[1]
            )

            # The mean uses the standard transformation
            mu = inits_i[0] + params_i[0]

            theta[i, 0] = mu
            theta[i, 1] = sig

        elif dist_i == 'normal_cov':
            # Note that the CoV is used for 'normal_cov'
            sig = np.exp(np.log(inits_i[1]) + params_i[1])

            # The mean uses the standard transformation
            mu = inits_i[0] + params_i[0] * sig

            theta[i, 0] = mu
            theta[i, 1] = sig

        else:
            msg = f'Unsupported distribution: {dist_i}'
            raise ValueError(msg)

    return theta


def _get_limit_probs(
    limits: np.ndarray, distribution: str, theta: np.ndarray
) -> tuple[float, float]:
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
    if distribution in {'normal', 'normal_std', 'normal_cov', 'lognormal'}:
        a, b = limits
        mu = theta[0]
        sig = theta[1] if distribution != 'normal_COV' else np.abs(mu) * theta[1]

        p_a = 0.0 if np.isnan(a) else norm.cdf((a - mu) / sig)
        p_b = 1.0 if np.isnan(b) else norm.cdf((b - mu) / sig)

    else:
        msg = f'Unsupported distribution: {distribution}'
        raise ValueError(msg)

    return p_a, p_b


def _get_std_samples(
    samples: np.ndarray,
    theta: np.ndarray,
    tr_limits: np.ndarray,
    dist_list: np.ndarray,
) -> np.ndarray:
    """
    Transform samples to standard normal space.

    Parameters
    ----------
    samples: float ndarray DxN
      2D array of samples. Each row represents a sample.
    theta: float ndarray Dx2
      2D array of theta values that represent each dimension of the
      samples.
    tr_limits: float ndarray Dx2
      2D array with rows that represent [a, b] pairs of truncation
      limits.
    dist_list: str ndarray of length D
      1D array containing the names of the distributions.

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
        if dist_i in {'normal', 'normal_std', 'normal_cov', 'lognormal'}:
            lim_low = tr_lim_i[0]
            lim_high = tr_lim_i[1]

            if (
                True in (samples_i > lim_high).tolist()
                or True in (samples_i < lim_low).tolist()
            ):
                msg = (
                    'One or more sample values lie outside '
                    'of the specified truncation limits.'
                )
                raise ValueError(msg)

            # first transform from normal to uniform
            uni_sample = norm.cdf(samples_i, loc=theta_i[0], scale=theta_i[1])

            # replace 0 and 1 values with the nearest float
            uni_sample[uni_sample == 0] = FIRST_POSITIVE_NUMBER
            uni_sample[uni_sample == 1] = np.nextafter(1, -1)

            # consider truncation if needed
            p_a, p_b = _get_limit_probs(tr_lim_i, dist_i, theta_i)
            uni_sample = (uni_sample - p_a) / (p_b - p_a)

            # then transform from uniform to standard normal
            std_samples[i] = norm.ppf(uni_sample, loc=0.0, scale=1.0)

        else:
            msg = f'Unsupported distribution: {dist_i}'
            raise ValueError(msg)

    return std_samples


def _get_std_corr_matrix(std_samples: np.ndarray) -> np.ndarray | None:
    """
    Estimate the correlation matrix.

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
        msg = 'std_samples array must not contain inf or NaN values'
        raise ValueError(msg)

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
            u_matrix, s_vector, _ = svd(
                rho_hat,
            )

        except np.linalg.LinAlgError:
            # if this also fails, we give up
            return None

        s_diag = np.diagflat(s_vector)

        rho_hat = u_matrix @ s_diag @ u_matrix.T
        np.fill_diagonal(rho_hat, 1.0)

        # check if we introduced any unreasonable values
        vmax = 1.01
        vmin = -1.01
        if (np.max(rho_hat) > vmax) or (np.min(rho_hat) < vmin):
            return None

        # round values to 1.0 and -1.0, if needed
        if np.max(rho_hat) > 1.0:
            rho_hat /= np.max(rho_hat)

        if np.min(rho_hat) < -1.0:
            rho_hat /= np.abs(np.min(rho_hat))

    return rho_hat


def _mvn_scale(x: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Scaling utility function.

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
    a[a < FIRST_POSITIVE_NUMBER] = FIRST_POSITIVE_NUMBER

    b = mvn.pdf(x, mean=np.zeros(n_dims), cov=rho)

    return b / a


def _neg_log_likelihood(  # noqa: C901
    params: np.ndarray,
    inits: np.ndarray,
    bnd_lower: np.ndarray,
    bnd_upper: np.ndarray,
    samples: np.ndarray,
    dist_list: np.ndarray,
    tr_limits: np.ndarray,
    det_limits: list[np.ndarray],
    censored_count: int,
    enforce_bounds: bool = False,  # noqa: FBT001, FBT002
) -> float:
    """
    Calculate negative log likelihood.

    Calculate the negative log likelihood of the given data samples
    given the parameter values and distribution information.

    This function is used as an objective function in optimization
    algorithms to estimate the parameters of the distribution of the
    input data.

    Parameters
    ----------
    params: ndarray
        1D array with the parameter values to be assessed.
    inits: ndarray
        1D array with the initial estimates for the distribution
        parameters.
    bnd_lower: ndarray
        1D array with the lower bounds for the distribution
        parameters.
    bnd_upper: ndarray
        1D array with the upper bounds for the distribution
        parameters.
    samples: ndarray
        2D array with the data samples. Each column corresponds to a
        different random variable.
    dist_list: str ndarray of length D
        1D array containing the names of the distributions
    tr_limits: float ndarray Dx2
        2D array with rows that represent [a, b] pairs of truncation
        limits.
    det_limits: list
        List with the detection limits for each random variable.
    censored_count: int
        Number of censored samples in the data.
    enforce_bounds: bool, optional
        If True, the parameters are only considered valid if they are
        within the bounds defined by bnd_lower and bnd_upper. The
        default value is False.

    Returns
    -------
    float
        The negative log likelihood of the data given the distribution parameters.

    """
    # First, check if the parameters are within the pre-defined bounds
    # TODO(AZ): check if it is more efficient to use a bounded
    # minimization algo
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
        if dist_i in {'normal', 'normal_std', 'normal_cov', 'lognormal'}:
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
            p_l, p_u = (np.min([np.max([lim, p_a]), p_b]) for lim in (p_l, p_u))
            p_l, p_u = ((lim - p_a) / (p_b - p_a) for lim in (p_l, p_u))

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
        cen_likelihood = max(1.0 - det_alpha, FIRST_POSITIVE_NUMBER)

    else:
        # If the data is not censored, use 1.0 for cen_likelihood to get a
        # zero log-likelihood later. Note that although this is
        # theoretically not correct, it does not alter the solution and
        # it is numerically much more convenient than working around the
        # log of zero likelihood.
        cen_likelihood = 1.0

    # take the product of likelihoods calculated in each dimension
    scale = _mvn_scale(std_samples.T, rho_hat)
    # TODO(AZ): We can almost surely replace the product of likelihoods
    # with a call to mvn()
    likelihoods = np.prod(likelihoods, axis=0) * scale

    # Zeros are a result of limited floating point precision. Replace them
    # with the smallest possible positive floating point number to
    # improve convergence.
    likelihoods = np.clip(likelihoods, a_min=FIRST_POSITIVE_NUMBER, a_max=None)

    # calculate the total negative log likelihood
    negative_log_likelihood = -(
        np.sum(np.log(likelihoods))  # from samples
        + censored_count * np.log(cen_likelihood)
    )  # censoring influence

    # print(theta[0], params, NLL)

    # normalize the NLL with the sample count
    return negative_log_likelihood / samples.size


def fit_distribution_to_sample(  # noqa: C901
    raw_sample: np.ndarray,
    distribution: str | list[str],
    truncation_limits: tuple[float, float] = (np.nan, np.nan),
    censored_count: int = 0,
    detection_limits: tuple[float, float] = (np.nan, np.nan),
    *,
    multi_fit: bool = False,
    logger_object: Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a distribution to sample using maximum likelihood estimation.

    The number of dimensions of the distribution are inferred from the
    shape of the sample data. Censoring is automatically considered if the
    number of censored samples and the corresponding detection limits are
    provided. Infinite or unspecified truncation limits lead to fitting a
    non-truncated distribution in that dimension.

    Parameters
    ----------
    raw_sample: float ndarray
        Raw data that serves as the basis of estimation. The number of samples
        equals the number of columns and each row introduces a new feature. In
        other words: a list of sample lists is expected where each sample list
        is a collection of samples of one variable.
    distribution: {'normal', 'lognormal'}
        Defines the target probability distribution type. Different types of
        distributions can be mixed by providing a list rather than a single
        value. Each element of the list corresponds to one of the features in
        the raw_sample.
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
            are returned for the supported distributions: normal,
            normal_cov - mean, coefficient of variation; normal_std -
            mean, standard deviation; lognormal - median, log standard
            deviation;
        Rho: float 2D ndarray, optional
            In the multivariate case, returns the estimate of the
            correlation matrix.

    Raises
    ------
    ValueError
        If NaN values are produced during standard normal space transformation

    """
    samples = np.atleast_2d(raw_sample)
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
        if distr in {'normal', 'normal_cov', 'normal_std', 'lognormal'}:
            # use the first two moments
            mu_init[d_i] = np.mean(samples[d_i])

            if n_samples == 1:
                sig_init[d_i] = 0.0
            else:
                sig_init[d_i] = np.std(samples[d_i])

    # replace zero standard dev with negligible standard dev
    sig_zero_id = np.where(sig_init == 0.0)[0]
    sig_init[sig_zero_id] = (
        1e-6 * np.abs(mu_init[sig_zero_id]) + FIRST_POSITIVE_NUMBER
    )

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
    min_sample_size_for_optimization = 3
    if (n_samples < min_sample_size_for_optimization) or (
        # there are no truncation or detection limits involved
        np.all(np.isnan(tr_limits)) and np.all(np.isnan(det_limits))
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
                np.array([dist_list[dim]]),
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
        msg = (
            'Something went wrong.'
            '\n'
            'Conversion to standard normal space was unsuccessful. \n'
            'The given samples might deviate '
            'substantially from the specified distribution.'
        )
        raise ValueError(msg)
    rho_hat = _get_std_corr_matrix(std_samples)
    if rho_hat is None:
        # If there is not enough data to produce a valid correlation matrix
        # estimate, we assume uncorrelated demands
        rho_hat = np.zeros((n_dims, n_dims))
        np.fill_diagonal(rho_hat, 1.0)

        if logger_object:
            logger_object.warning(
                'Demand sample size too small to reliably estimate '
                'the correlation matrix. Assuming uncorrelated demands.'
            )
        else:
            print(  # noqa: T201
                '\nWARNING: Demand sample size '
                'too small to reliably estimate '
                'the correlation matrix. Assuming '
                'uncorrelated demands.'
            )

    for d_i, distr in enumerate(dist_list):
        # Convert mean back to linear space if the distribution is lognormal
        if distr == 'lognormal':
            theta[d_i][0] = np.exp(theta[d_i][0])
            # theta_mod = theta.T.copy()
            # theta_mod[0] = np.exp(theta_mod[0])
            # theta = theta_mod.T
        # Convert the std to cov if the distribution is normal_cov
        elif distr in {'normal', 'normal_cov'}:
            # replace standard deviation with coefficient of variation
            # note: this results in cov=inf if the mean is zero.
            almost_zero = 1.0e-40
            if np.abs(theta[d_i][0]) < almost_zero:
                theta[d_i][1] = np.inf
            else:
                theta[d_i][1] /= np.abs(theta[d_i][0])

    return theta, rho_hat


def _OLS_percentiles(  # noqa: N802
    params: tuple[float, float], values: np.ndarray, perc: np.ndarray, family: str
) -> float:
    """
    Estimate percentiles using ordinary least squares (OLS).

    Parameters
    ----------
    params: tuple of floats
        The parameters of the selected distribution family.
    values: float ndarray
        The sample values for which the percentiles are requested.
    perc: float ndarray
        The requested percentile(s).
    family: str
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
        msg = f'Distribution family not recognized: {family}'
        raise ValueError(msg)

    return np.sum((val_hat - values) ** 2.0)


def fit_distribution_to_percentiles(
    values: list[float], percentiles: list[float], families: list[str]
) -> tuple[str, list[float]]:
    """
    Fit distribution to pre-defined values at a finite number of percentiles.

    Parameters
    ----------
    values: list of float
        Pre-defined values at the given percentiles. At least two values are
        expected.
    percentiles: list of float
        Percentiles where values are defined. At least two percentiles are
        expected.
    families: list of strings {'normal', 'lognormal'}
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

    percentiles_np = np.array(percentiles)

    median_id = np.argmin(np.abs(percentiles_np - 0.5))
    extreme_id = np.argmax(percentiles_np - 0.5)

    for family in families:
        inits = [
            values[median_id],
        ]

        if family == 'normal':
            inits.append(
                np.abs(values[extreme_id] - inits[0])
                / np.abs(norm.ppf(percentiles_np[extreme_id], loc=0, scale=1))
            )

        elif family == 'lognormal':
            inits.append(
                np.abs(np.log(values[extreme_id] / inits[0]))
                / np.abs(norm.ppf(percentiles_np[extreme_id], loc=0, scale=1))
            )

        out_list.append(
            minimize(
                _OLS_percentiles,
                inits,
                args=(values, percentiles_np, family),
                method='BFGS',
            )
        )

    best_out_id = np.argmin([out.fun for out in out_list])

    return families[best_out_id], out_list[best_out_id].x


class BaseRandomVariable(ABC):  # noqa: B024
    """Base abstract class for different types of random variables."""

    __slots__: list[str] = [
        'RV_set',
        '_sample',
        '_sample_DF',
        '_uni_sample',
        'anchor',
        'distribution',
        'f_map',
        'name',
    ]

    def __init__(
        self,
        name: str,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a RandomVariable object.

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
        self.name = name
        self.distribution: str | None = None
        self.f_map = f_map
        self._uni_sample: np.ndarray | None = None
        self.RV_set: RandomVariableSet | None = None
        self._sample_DF: pd.Series | None = None
        self._sample: np.ndarray | None = None
        if anchor is None:
            self.anchor = self
        else:
            self.anchor = anchor

    @property
    def sample(self) -> np.ndarray | None:
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
    def sample(self, value: np.ndarray) -> None:
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
    def sample_DF(self) -> pd.Series | None:  # noqa: N802
        """
        Return the empirical or generated sample in a pandas Series.

        Returns
        -------
        ndarray
          The empirical or generated sample in a pandas Series.

        """
        if self.f_map is not None:
            assert self._sample_DF is not None
            return self._sample_DF.apply(self.f_map)

        return self._sample_DF

    @property
    def uni_sample(self) -> np.ndarray | None:
        """
        Return the sample from the controlling uniform distribution.

        Returns
        -------
        ndarray
          The sample from the controlling uniform distribution.

        """
        if self.anchor is self:
            return self._uni_sample
        return self.anchor.uni_sample

    @uni_sample.setter
    def uni_sample(self, value: np.ndarray) -> None:
        """
        Assign the controlling sample to the random variable.

        Parameters
        ----------
        value: float ndarray
            An array of floating point values in the [0, 1] domain.

        """
        self._uni_sample = value


class RandomVariable(BaseRandomVariable):
    """Random variable that needs `values` in `inverse_transform`."""

    __slots__: list[str] = ['theta', 'truncation_limits']

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a normal random variable.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        theta: float ndarray
          Set of parameters that define the Cumulative Distribution
          Function (CDF) of the variable: E.g., mean and coefficient
          of variation. Actual parameters depend on the distribution.
          A 1D `theta` array represents constant parameters and results
          in realizations that are all from the same distribution.
          A 2D `theta` array represents variable parameters, meaning
          that each realization will be sampled from the distribution
          family that the object represents, but with the parameters
          set for that realization.
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
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))

        # For backwards compatibility, cast to a numpy array if
        # given a tuple or list.
        if isinstance(theta, (list, tuple)):
            theta = np.array(theta)
        if isinstance(truncation_limits, (list, tuple)):
            truncation_limits = np.array(truncation_limits)

        # Verify type
        if theta is not None:
            assert isinstance(
                theta, np.ndarray
            ), 'Parameter `theta` should be a numpy array.'
            assert theta.ndim in {
                1,
                2,
            }, 'Parameter `theta` can only be a 1D or 2D array.'
            theta = np.atleast_1d(theta)

        assert isinstance(
            truncation_limits, np.ndarray
        ), 'Parameter `truncation_limits` should be a numpy array.'
        assert truncation_limits.ndim in {
            1,
            2,
        }, 'Parameter `truncation_limits` can only be a 1D or 2D array.'
        # 1D corresponds to constant parameters.
        # 2D corresponds to variable parameters (different in each
        # realization).

        self.theta = theta
        self.truncation_limits = truncation_limits

        super().__init__(
            name=name,
            f_map=f_map,
            anchor=anchor,
        )

    def constant_parameters(self) -> bool:
        """
        If the RV has constant or variable parameters.

        Constant parameters are the same in each realization.

        Returns
        -------
        bool
          True if the parameters are constant, false otherwise.

        """
        if self.theta is None:
            return True
        assert self.theta.ndim in {1, 2}
        return self.theta.ndim == 1

    def _prepare_theta_and_truncation_limit_arrays(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the `theta` and `truncation_limits` arrays.

        Prepare the `theta` and `truncation_limits` arrays for use in
        calculations. This method adjusts the shape and size of the
        `theta` and `truncation_limits` attributes to ensure
        compatibility with the provided `values` array. The
        adjustments enable support for two approaches:
        * Constant parameters: The parameters remain the same
        across all realizations.
        * Variable parameters: The parameters vary across
        realizations.

        Depending on whether the random variable uses constant or
        variable parameters, the method ensures that the arrays are
        correctly sized and broadcasted as needed.

        Parameters
        ----------
        values : np.ndarray
            Array of values for which the `theta` and
            `truncation_limits` need to be prepared. The size of
            `values` determines how the attributes are adjusted.

        Returns
        -------
        tuple
            A tuple containing:
            * `theta` (np.ndarray): Adjusted array of parameters.
            * `truncation_limits` (np.ndarray): Adjusted array of
            truncation limits.

        Raises
        ------
        ValueError
            If the number of elements in `values` does not match the
            number of rows of the `theta` attribute or if the
            `truncation_limits` array is incompatible with the `theta`
            array.

        Notes
        -----
        The method ensures that `truncation_limits` are broadcasted to
        match the shape of `theta` if needed. For constant parameters,
        a single-row `theta` is expanded to a 2D array. For variable
        parameters, the number of rows in `theta` must match the size
        of `values`.
        """
        theta = self.theta
        truncation_limits = self.truncation_limits
        assert truncation_limits is not None
        if self.constant_parameters():
            theta = np.atleast_2d(theta)
        elif len(values) != theta.shape[0]:
            msg = (
                'Number of elements in `values` variable should '
                'match the number of rows of the parameter '
                'attribute `theta`.'
            )
            raise ValueError(msg)

        # Broadcast truncation limits to match shape
        truncation_limits = np.atleast_2d(truncation_limits)
        assert truncation_limits is not None

        if truncation_limits.shape != theta.shape:
            # Number of rows should match
            if truncation_limits.shape[1] != theta.shape[1]:
                msg = 'Incompatible `truncation_limits` value.'
                raise ValueError(msg)
            truncation_limits = np.tile(truncation_limits, (theta.shape[0], 1))
        return theta, truncation_limits

    @staticmethod
    def _ensure_positive_probability_difference(
        p_b: np.ndarray, p_a: np.ndarray
    ) -> None:
        """
        Ensure there is probability mass between the truncation limits.

        Parameters
        ----------
        p_b: float
          The probability of not exceeding the upper truncation limit
          based on the CDF of the random variable.
        p_a: float
          The probability of not exceeding the lower truncation limit
          based on the CDF of the random variable.

        Raises
        ------
        ValueError
          If a negative probability difference is found.

        """
        if np.any((p_b - p_a) < FIRST_POSITIVE_NUMBER):
            msg = (
                'The probability mass within the truncation limits is '
                'too small and the truncated distribution cannot be '
                'sampled with sufficiently high accuracy. This is most '
                'probably due to incorrect truncation limits set for '
                'the distribution.'
            )
            raise ValueError(msg)

    @abstractmethod
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Uses inverse probability integral transformation on the
        provided values.

        """

    def inverse_transform_sampling(self) -> None:
        """
        Create a sample with inverse transform sampling.

        Raises
        ------
        ValueError
          If there is no available uniform sample.

        """
        if self.uni_sample is None:
            msg = 'No available uniform sample.'
            raise ValueError(msg)
        self.sample = self.inverse_transform(self.uni_sample)


class UtilityRandomVariable(BaseRandomVariable):
    """Random variable that needs `sample_size` in `inverse_transform`."""

    __slots__: list[str] = []

    @abstractmethod
    def __init__(
        self,
        name: str,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a normal random variable.

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
    def inverse_transform(self, sample_size: int) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Uses inverse probability integral transformation on the
        provided values.

        """

    def inverse_transform_sampling(self, sample_size: int) -> None:
        """Create a sample with inverse transform sampling."""
        self.sample = self.inverse_transform(sample_size)


class NormalRandomVariable(RandomVariable):
    """Normal random variable."""

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """Instantiate a Normal random variable."""
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'normal'

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Return the CDF at the given values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          1D float ndarray containing CDF values

        """
        theta, truncation_limits = self._prepare_theta_and_truncation_limit_arrays(
            values
        )
        mu, sig = theta.T

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = truncation_limits.T

            # Replace NaN values
            a = np.nan_to_num(a, nan=-np.inf)
            b = np.nan_to_num(b, nan=np.inf)

            p_a, p_b = (norm.cdf((lim - mu) / sig) for lim in (a, b))
            self._ensure_positive_probability_difference(p_b, p_a)

            # cap the values at the truncation limits
            values = np.minimum(np.maximum(values, a), b)

            # get the cdf from a non-truncated normal
            p_vals = norm.cdf(values, loc=mu, scale=sig)

            # adjust for truncation
            result = (p_vals - p_a) / (p_b - p_a)

        else:
            result = norm.cdf(values, loc=mu, scale=sig)

        return result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

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
        theta, truncation_limits = self._prepare_theta_and_truncation_limit_arrays(
            values
        )
        mu, sig = theta.T

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = truncation_limits.T

            # Replace NaN values
            a = np.nan_to_num(a, nan=-np.inf)
            b = np.nan_to_num(b, nan=np.inf)

            p_a, p_b = (norm.cdf((lim - mu) / sig) for lim in (a, b))
            self._ensure_positive_probability_difference(p_b, p_a)

            result = norm.ppf(values * (p_b - p_a) + p_a, loc=mu, scale=sig)

        else:
            result = norm.ppf(values, loc=mu, scale=sig)

        return result


class Normal_STD(NormalRandomVariable):
    """
    Normal random variable with standard deviation.

    This class represents a normal random variable defined by mean and
    standard deviation.

    """

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """Instantiate a Normal_STD random variable."""
        mean, std = theta[:2]
        theta = np.array([mean, std])
        super().__init__(name, theta, truncation_limits, f_map, anchor)


class Normal_COV(NormalRandomVariable):
    """
    Normal random variable with coefficient of variation.

    This class represents a normal random variable defined by mean and
    coefficient of variation.

    """

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a Normal_COV random variable.

        Raises
        ------
        ValueError
          If the specified mean is zero.

        """
        mean, cov = theta[:2]

        almost_zero = 1e-40
        if np.abs(mean) < almost_zero:
            msg = 'The mean of Normal_COV RVs cannot be zero.'
            raise ValueError(msg)

        std = mean * cov
        theta = np.array([mean, std])
        super().__init__(name, theta, truncation_limits, f_map, anchor)


class LogNormalRandomVariable(RandomVariable):
    """Lognormal random variable."""

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """Instantiate a LogNormal random variable."""
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'lognormal'

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Return the CDF at the given values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          1D float ndarray containing CDF values

        """
        theta, truncation_limits = self._prepare_theta_and_truncation_limit_arrays(
            values
        )
        theta, beta = theta.T

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = truncation_limits.T

            # Replace NaN values
            a = np.nan_to_num(a, nan=FIRST_POSITIVE_NUMBER)
            b = np.nan_to_num(b, nan=np.inf)

            p_a, p_b = (
                norm.cdf((np.log(lim) - np.log(theta)) / beta) for lim in (a, b)
            )
            self._ensure_positive_probability_difference(p_b, p_a)

            # cap the values at the truncation limits
            values = np.minimum(np.maximum(values, a), b)

            # get the cdf from a non-truncated lognormal
            p_vals = norm.cdf(np.log(values), loc=np.log(theta), scale=beta)

            # adjust for truncation
            result = (p_vals - p_a) / (p_b - p_a)

        else:
            values = np.maximum(values, FIRST_POSITIVE_NUMBER)

            result = norm.cdf(np.log(values), loc=np.log(theta), scale=beta)

        return result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Uses inverse probability integral transformation on the
        provided values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the inverse CDF

        Returns
        -------
        ndarray
          Inverse CDF values

        """
        theta, truncation_limits = self._prepare_theta_and_truncation_limit_arrays(
            values
        )
        theta, beta = theta.T

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = truncation_limits.T

            # Replace NaN values
            a = np.nan_to_num(a, nan=FIRST_POSITIVE_NUMBER)
            a[a <= 0] = FIRST_POSITIVE_NUMBER
            b = np.nan_to_num(b, nan=np.inf)

            p_a, p_b = (
                norm.cdf((np.log(lim) - np.log(theta)) / beta) for lim in (a, b)
            )
            self._ensure_positive_probability_difference(p_b, p_a)

            result = np.exp(
                norm.ppf(values * (p_b - p_a) + p_a, loc=np.log(theta), scale=beta)
            )

        else:
            result = np.exp(norm.ppf(values, loc=np.log(theta), scale=beta))

        return result


class UniformRandomVariable(RandomVariable):
    """Uniform random variable."""

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a Uniform random variable.

        Raises
        ------
        ValueError
          If variable parameters are specified.

        """
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'uniform'

        if self.theta.ndim != 1:
            msg = (
                'Variable parameters are currently not supported for '
                'Uniform random variables.'
            )
            raise ValueError(msg)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Return the CDF at the given values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          1D float ndarray containing CDF values

        """
        a, b = self.theta[:2]

        if np.isnan(a):
            a = -np.inf
        if np.isnan(b):
            b = np.inf

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

        return uniform.cdf(values, loc=a, scale=(b - a))

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Uses inverse probability integral transformation on the
        provided values.

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

        return uniform.ppf(values, loc=a, scale=(b - a))


class WeibullRandomVariable(RandomVariable):
    """Weibull random variable."""

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a Weibull random variable.

        Raises
        ------
        ValueError
          If variable parameters are specified.

        """
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'weibull'

        if self.theta.ndim != 1:
            msg = (
                'Variable parameters are currently not supported for '
                'Weibull random variables.'
            )
            raise ValueError(msg)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Return the CDF at the given values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          1D float ndarray containing CDF values

        """
        lambda_, kappa = self.theta[:2]

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

            if np.isnan(a):
                # Weibull is not defined for negative values
                a = 0.0
            if np.isnan(b):
                b = np.inf

            p_a, p_b = (weibull_min.cdf(lim, kappa, scale=lambda_) for lim in (a, b))
            self._ensure_positive_probability_difference(p_b, p_a)

            # cap the values at the truncation limits
            values = np.minimum(np.maximum(values, a), b)

            # get the cdf from a non-truncated weibull
            p_vals = weibull_min.cdf(values, kappa, scale=lambda_)

            # adjust for truncation
            result = (p_vals - p_a) / (p_b - p_a)

        else:
            values = np.maximum(
                values, 0.0
            )  # Weibull is not defined for negative values

            result = weibull_min.cdf(values, kappa, scale=lambda_)

        return result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Uses inverse probability integral transformation on the
        provided values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the inverse CDF

        Returns
        -------
        ndarray
          Inverse CDF values

        """
        lambda_, kappa = self.theta[:2]

        if np.any(~np.isnan(self.truncation_limits)):
            a, b = self.truncation_limits

            if np.isnan(a):
                a = 0.0  # Weibull is not defined for negative values
            else:
                a = np.maximum(0.0, a)

            if np.isnan(b):
                b = np.inf

            p_a, p_b = (weibull_min.cdf(lim, kappa, scale=lambda_) for lim in (a, b))
            self._ensure_positive_probability_difference(p_b, p_a)

            result = weibull_min.ppf(
                values * (p_b - p_a) + p_a, kappa, scale=lambda_
            )

        else:
            result = weibull_min.ppf(values, kappa, scale=lambda_)

        return result


class MultilinearCDFRandomVariable(RandomVariable):
    """
    Multilinear CDF random variable.

    This RV is defined by specifying the points that define its
    Cumulative Density Function (CDF), and linear interpolation
    between them.

    """

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a MultilinearCDF random variable.

        Raises
        ------
        ValueError
            In case of incompatible input parameters.
        NotImplementedError
            If truncation limits are specified.

        """
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'multilinear_CDF'

        if not np.all(np.isnan(truncation_limits)):
            msg = f'{self.distribution} RVs do not support truncation'
            raise NotImplementedError(msg)

        y_1 = theta[0, 1]
        if y_1 != 0.00:
            msg = 'For multilinear CDF random variables, y_1 should be set to 0.00'
            raise ValueError(msg)
        y_n = theta[-1, 1]
        if y_n != 1.00:
            msg = 'For multilinear CDF random variables, y_n should be set to 1.00'
            raise ValueError(msg)

        x_s = theta[:, 0]
        if not np.array_equal(np.sort(x_s), x_s):
            msg = (
                'For multilinear CDF random variables, '
                'Xs should be specified in ascending order'
            )
            raise ValueError(msg)
        if np.any(np.isclose(np.diff(x_s), 0.00)):
            msg = (
                'For multilinear CDF random variables, '
                'Xs should be specified in strictly ascending order'
            )
            raise ValueError(msg)

        y_s = theta[:, 1]
        if not np.array_equal(np.sort(y_s), y_s):
            msg = (
                'For multilinear CDF random variables, '
                'Ys should be specified in ascending order'
            )
            raise ValueError(msg)

        if np.any(np.isclose(np.diff(y_s), 0.00)):
            msg = (
                'For multilinear CDF random variables, '
                'Ys should be specified in strictly ascending order'
            )
            raise ValueError(msg)

        required_ndim = 2
        assert self.theta.ndim == required_ndim, 'Invalid `theta` dimensions.'

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Return the CDF at the given values.

        Parameters
        ----------
        values: 1D float ndarray
          Values for which to evaluate the CDF

        Returns
        -------
        ndarray
          1D float ndarray containing CDF values

        """
        x_i = [-np.inf] + [x[0] for x in self.theta] + [np.inf]
        y_i = [0.00] + [x[1] for x in self.theta] + [1.00]

        # Using Numpy's interp for linear interpolation
        return np.interp(values, x_i, y_i, left=0.00, right=1.00)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Uses inverse probability integral transformation on the
        provided values.

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
        return np.interp(values, y_i, x_i)


class EmpiricalRandomVariable(RandomVariable):
    """Empirical random variable."""

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """Instantiate an Empirical random variable."""
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))

        theta = np.atleast_1d(theta)

        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'empirical'
        if not np.all(np.isnan(truncation_limits)):
            msg = f'{self.distribution} RVs do not support truncation'
            raise NotImplementedError(msg)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

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
        s_ids = (values * len(self.theta)).astype(int)
        return self.theta[s_ids]


class CoupledEmpiricalRandomVariable(UtilityRandomVariable):
    """Coupled empirical random variable."""

    __slots__: list[str] = ['theta']

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a coupled empirical random variable.

        Parameters
        ----------
        name: string
            A unique string that identifies the random variable.
        theta: 1D float ndarray
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
        if truncation_limits is not None:
            msg = f'{self.distribution} RVs do not support truncation'
            raise NotImplementedError(msg)

        self.theta = np.atleast_1d(theta)

    def inverse_transform(self, sample_size: int) -> np.ndarray:
        """
        Evaluate the inverse CDF.

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
        raw_sample_count = len(self.theta)
        new_sample = np.tile(self.theta, int(sample_size / raw_sample_count) + 1)
        return new_sample[:sample_size]


class DeterministicRandomVariable(UtilityRandomVariable):
    """Deterministic random variable."""

    __slots__: list[str] = ['theta']

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a deterministic random variable.

        This behaves like a RandomVariable object but represents a
        specific, deterministic value.

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
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            f_map=f_map,
            anchor=anchor,
        )
        self.distribution = 'deterministic'
        if not np.all(np.isnan(truncation_limits)):
            msg = f'{self.distribution} RVs do not support truncation'
            raise NotImplementedError(msg)

        self.theta = np.atleast_1d(theta)

    def inverse_transform(self, sample_size: int) -> np.ndarray:
        """
        Evaluate the inverse CDF.

        Parameters
        ----------
        sample_size: int
          The desired size of the sample array to be generated.

        Returns
        -------
        ndarray
          Sample array containing the deterministic value.

        """
        return np.full(sample_size, self.theta[0])


class MultinomialRandomVariable(RandomVariable):
    """Multinomial random variable."""

    __slots__: list[str] = []

    def __init__(
        self,
        name: str,
        theta: np.ndarray,
        truncation_limits: np.ndarray | None = None,
        f_map: Callable | None = None,
        anchor: BaseRandomVariable | None = None,
    ) -> None:
        """
        Instantiate a Multinomial random variable.

        Raises
        ------
        ValueError
            In case of incompatible input parameters.
        NotImplementedError
            If truncation limits are specified.

        """
        if truncation_limits is None:
            truncation_limits = np.array((np.nan, np.nan))
        super().__init__(
            name=name,
            theta=theta,
            truncation_limits=truncation_limits,
            f_map=f_map,
            anchor=anchor,
        )
        if not np.all(np.isnan(truncation_limits)):
            msg = f'{self.distribution} RVs do not support truncation'
            raise NotImplementedError(msg)
        self.distribution = 'multinomial'
        if np.sum(theta) > 1.00:
            msg = (
                f'The set of p values provided for a multinomial '
                f'distribution shall sum up to less than or equal to 1.0. '
                f'The provided values sum up to {np.sum(theta)}. p = '
                f'{theta} .'
            )
            raise ValueError(msg)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate the inverse CDF.

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

        return values - 10


class RandomVariableSet:
    """
    Random variable set.

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

    __slots__: list[str] = ['_Rho', '_variables', 'name']

    def __init__(
        self, name: str, rv_list: list[BaseRandomVariable], rho: np.ndarray
    ) -> None:
        """Instantiate a random variable set."""
        self.name = name

        if len(rv_list) > 1:
            # put the RVs in a dictionary for more efficient access
            reorder = np.argsort([RV.name for RV in rv_list])
            self._variables = {rv_list[i].name: rv_list[i] for i in reorder}

            # reorder the entries in the correlation matrix to correspond to the
            # sorted list of RVs
            self._Rho = np.asarray(rho[(reorder)].T[(reorder)].T)

        else:  # if there is only one variable (for testing, probably)
            self._variables = {rv.name: rv for rv in rv_list}
            self._Rho = np.asarray(rho)

        # assign this RV_set to the variables
        for var in self._variables.values():
            var.RV_set = self

    @property
    def RV(self) -> dict[str, RandomVariable]:  # noqa: N802
        """
        Returns the random variable(s) assigned to the set.

        Returns
        -------
        ndarray
          The random variable(s) assigned to the set.

        """
        return self._variables

    @property
    def size(self) -> int:
        """
        Returns the size (i.e., number of variables in the) RV set.

        Returns
        -------
        ndarray
          The size (i.e., number of variables in the) RV set.

        """
        return len(self._variables)

    @property
    def sample(self) -> dict[str, np.ndarray | None]:
        """
        Returns the sample of the variables in the set.

        Returns
        -------
        ndarray
          The sample of the variables in the set.

        """
        return {name: rv.sample for name, rv in self._variables.items()}

    def Rho(self, var_subset: list[str] | None = None) -> np.ndarray:  # noqa: N802
        """
        Return the (subset of the) correlation matrix.

        Returns
        -------
        ndarray
          The (subset of the) correlation matrix.

        """
        if var_subset is None:
            return self._Rho
        var_ids = [list(self._variables.keys()).index(var_i) for var_i in var_subset]
        return (self._Rho[var_ids]).T[var_ids]

    def apply_correlation(self) -> None:
        """
        Apply correlation to n dimensional uniform samples.

        Currently, correlation is applied using a Gaussian copula. First, we
        try using Cholesky transformation. If the correlation matrix is not
        positive semidefinite and Cholesky fails, use SVD to apply the
        correlations while preserving as much as possible from the correlation
        matrix.
        """
        u_rv = np.array([RV.uni_sample for RV_name, RV in self.RV.items()])

        # First try doing the Cholesky transformation
        try:
            n_rv = norm.ppf(u_rv)

            l_mat = cholesky(self._Rho, lower=True)

            nc_rv = l_mat @ n_rv

            uc_rv = norm.cdf(nc_rv)

        except np.linalg.LinAlgError:
            # if the Cholesky doesn't work, we need to use the more
            # time-consuming but more robust approach based on SVD
            n_rv = norm.ppf(u_rv)

            u_mat, s_mat, _ = svd(
                self._Rho,
            )
            s_diag = np.diagflat(np.sqrt(s_mat))

            nc_rv = (n_rv.T @ s_diag @ u_mat.T).T

            uc_rv = norm.cdf(nc_rv)

        for rv, ucrv in zip(self.RV.values(), uc_rv):
            rv.uni_sample = ucrv

    def orthotope_density(
        self,
        lower: np.ndarray | float = np.nan,
        upper: np.ndarray | float = np.nan,
        var_subset: list[str] | None = None,
    ) -> np.ndarray:
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
        lower: float ndarray, optional, default: np.nan
            Lower bound(s) of the orthotope. A scalar value can be used for a
            univariate RV; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from below in a dimension, use
            'np.nan' to that dimension.
        upper: float ndarray, optional, default: np.nan
            Upper bound(s) of the orthotope. A scalar value can be used for a
            univariate RV; a list of bounds is expected in multivariate cases.
            If the orthotope is not bounded from above in a dimension, use
            'np.nan' to that dimension.
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
        if isinstance(lower, float):
            lower = np.array([lower])
        if isinstance(upper, float):
            upper = np.array([upper])

        if np.any(~np.isnan(lower)):
            target_shape = lower.shape
        elif np.any(~np.isnan(upper)):
            target_shape = upper.shape
        else:
            return np.array([1.0])

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

        od = [
            mvn_orthotope_density(
                mu=np.zeros(len(variables)),
                cov=self.Rho(var_subset),
                lower=l_i,
                upper=u_i,
            )[0]
            for l_i, u_i in zip(lower_std, upper_std)
        ]

        return np.asarray(od)


class RandomVariableRegistry:
    """Random variable registry."""

    __slots__: list[str] = ['_rng', '_sets', '_variables']

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Instantiate a random variable registry.

        Parameters
        ----------
        rng: numpy.random._generator.Generator
            Random variable generator object.
            e.g.: np.random.default_rng(seed).

        """
        self._rng = rng
        self._variables: dict[str, BaseRandomVariable] = {}
        self._sets: dict[str, RandomVariableSet] = {}

    @property
    def RV(self) -> dict[str, BaseRandomVariable]:  # noqa: N802
        """
        Returns all random variable(s) in the registry.

        Returns
        -------
        dict
          all random variable(s) in the registry.

        """
        return self._variables

    def RVs(self, keys: list[str]) -> dict[str, BaseRandomVariable]:  # noqa: N802
        """
        Return a subset of the random variables in the registry.

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

    def add_RV(self, rv: BaseRandomVariable) -> None:  # noqa: N802
        """
        Add a new random variable to the registry.

        Raises
        ------
        ValueError
          When the RV already exists in the registry

        """
        if rv.name in self._variables:
            msg = f'RV {rv.name} already exists in the registry.'
            raise ValueError(msg)
        self._variables.update({rv.name: rv})

    @property
    def RV_set(self) -> dict[str, RandomVariableSet]:  # noqa: N802
        """
        Return the random variable set(s) in the registry.

        Returns
        -------
        dict
          The random variable set(s) in the registry.

        """
        return self._sets

    def add_RV_set(self, rv_set: RandomVariableSet) -> None:  # noqa: N802
        """Add a new set of random variables to the registry."""
        self._sets.update({rv_set.name: rv_set})

    @property
    def RV_sample(self) -> dict[str, np.ndarray | None]:  # noqa: N802
        """
        Return the sample for every random variable in the registry.

        Returns
        -------
        dict
          The sample for every random variable in the registry.

        """
        return {name: rv.sample for name, rv in self.RV.items()}

    def generate_sample(self, sample_size: int, method: str) -> None:
        """
        Generate samples for all variables in the registry.

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
        rv_list = [
            RV_name
            for RV_name, RV in self.RV.items()
            if (
                (RV.anchor == RV)
                or (RV.distribution in {'deterministic', 'coupled_empirical'})
            )
        ]
        rv_id = {RV_name: ID for ID, RV_name in enumerate(rv_list)}
        rv_count = len(rv_id)

        # Generate controlling samples from a uniform distribution for free RVs
        if 'LHS' in method:
            bin_low = np.array(
                [self._rng.permutation(sample_size) for i in range(rv_count)]
            )

            if method == 'LHS_midpoint':
                u_rv = np.ones([rv_count, sample_size]) * 0.5
                u_rv = (bin_low + u_rv) / sample_size

            elif method == 'LHS':
                u_rv = self._rng.random(size=[rv_count, sample_size])
                u_rv = (bin_low + u_rv) / sample_size

        elif method == 'MonteCarlo':
            u_rv = self._rng.random(size=[rv_count, sample_size])

        # Assign the controlling samples to the RVs
        for rv_name, rvid in rv_id.items():
            self.RV[rv_name].uni_sample = u_rv[rvid]

        # Apply correlations for the pre-defined RV sets
        for rv_set in self.RV_set.values():
            # prepare the correlated uniform distribution for the set
            rv_set.apply_correlation()

        # Convert from uniform to the target distribution for every RV
        for rv in self.RV.values():
            if isinstance(rv, UtilityRandomVariable):
                rv.inverse_transform_sampling(sample_size)
            elif isinstance(rv, RandomVariable):
                rv.inverse_transform_sampling()
            else:
                msg = 'Unknown RV parent class.'
                raise NotImplementedError(msg)


def rv_class_map(
    distribution_name: str,
) -> type[RandomVariable | UtilityRandomVariable]:
    """
    Map convenient distributions to their corresponding class.

    Parameters
    ----------
    distribution_name: str
        The name of a distribution.

    Returns
    -------
    type[RandomVariable | UtilityRandomVariable]
        The class of the corresponding random variable.

    Raises
    ------
    ValueError
        If the given distribution name does not correspond to a
        distribution class.

    """
    if pd.isna(distribution_name):
        distribution_name = 'deterministic'

    # Mapping for RandomVariable subclasses
    random_variable_map: dict[str, type[RandomVariable]] = {
        'normal': Normal_COV,
        'normal_std': Normal_STD,
        'normal_cov': Normal_COV,
        'lognormal': LogNormalRandomVariable,
        'uniform': UniformRandomVariable,
        'weibull': WeibullRandomVariable,
        'multilinear_CDF': MultilinearCDFRandomVariable,
        'empirical': EmpiricalRandomVariable,
        'multinomial': MultinomialRandomVariable,
    }

    # Mapping for UtilityRandomVariable subclasses
    utility_random_variable_map: dict[str, type[UtilityRandomVariable]] = {
        'coupled_empirical': CoupledEmpiricalRandomVariable,
        'deterministic': DeterministicRandomVariable,
    }

    if distribution_name in random_variable_map:
        return random_variable_map[distribution_name]
    if distribution_name in utility_random_variable_map:
        return utility_random_variable_map[distribution_name]
    msg = f'Unsupported distribution: {distribution_name}'
    raise ValueError(msg)
