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

    MLE_normal
    RandomVariable

"""

import warnings
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import kde
from scipy.optimize import minimize

def tmvn_rvs(mu, COV, lower=None, upper=None, size=1):
    """
    Sample a truncated MVN distribution.
    
    Truncation of the multivariate normal distribution is currently considered 
    through rejection sampling. The applicability of this method is limited by
    the amount of probability density enclosed by the hyperrectangle defined by
    the truncation limits. The lower that density is, the more samples will 
    need to be rejected which makes the method inefficient when the tails of
    the MVN shall be sampled in high-dimensional space. Such cases can be 
    handled by a Gibbs sampler, which is a planned future feature of this 
    function.
    
    Parameters
    ----------
    mu: float scalar or ndarray
        Mean(s) of the non-truncated distribution.
    COV: float ndarray
        Covariance matrix of the non-truncated distribution.
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
    size: int
        Number of samples requested.

    Returns
    -------
    samples: float ndarray
        Samples generated from the truncated distribution.

    """
    
    mu = np.asarray(mu)
    if mu.shape == ():
        mu = np.asarray([mu])
        COV = np.asarray([COV])
    
    # if there are no bounds, simply sample an MVN distribution
    if lower is None and upper is None:
        
        samples = multivariate_normal.rvs(mean=mu, cov=COV, size=size)
    
    else:        
        # first, get the rejection rate
        alpha, eps_alpha = mvn_orthotope_density(mu, COV, lower, upper)
        
        # initialize the data for sample collection
        sample_count = 0
        samples = None
        ndim = len(mu)

        # If the error in the alpha estimate is too large, then we are 
        # beyond the applicability limits of the function used for 
        # estimating alpha. Raise an error in such a case
        if alpha <= 100. * eps_alpha:  # i.e. max. error is limited at 1%
            raise ValueError(
                "The density of the joint probability distribution within the "
                "truncation limits is too small and cannot be estimated with "
                "sufficiently high accuracy. This is most probably due to "
                "incorrect limits set for the distribution."
            )
        
        # If the rejection rate is sufficiently low, perform rejection sampling
        # Note: the minimum rate is set to zero until a Gibbs sampler is 
        # implemented, but a warning message is displayed for anything below
        # 1e-3 
        if alpha < 1e-3:
            warnings.warn(UserWarning(
                "The rejection rate for sampling the prescribed truncated MVN "
                "distribution is higher than 0.999. This makes sampling with "
                "our current implementation very resource-intensive and "
                "inefficient. If you need to sample such parts of MVN "
                "distributions, please let us know and we will improve "
                "this function in the future."
            ))
        if alpha > 0.:
            while sample_count < size:
                
                # estimate the required number of samples
                req_samples = max(int(1.1*(size-sample_count)/alpha), 2)
                
                # generate the raw samples
                raw_samples = multivariate_normal.rvs(mu, COV, 
                                                      size=req_samples)
                
                # remove the samples that are outside the truncation limits
                good_ones = np.all([raw_samples>lower, raw_samples<upper],
                                   axis=0)
                if ndim > 1:
                    good_ones = np.all(good_ones, axis=1)
                
                new_samples = raw_samples[good_ones]
                
                # add the new samples to the pool of samples
                if sample_count > 0:
                    samples = np.concatenate([samples, new_samples], axis=0)
                else:
                    samples = new_samples
                
                # check the number of available samples and generate more if 
                # needed
                sample_count = len(samples)

            samples = samples[:size]
                
        #else:
        # TODO: Gibbs sampler
        
    return samples

def mvn_orthotope_density(mu, COV, lower=None, upper=None):
    """
    Estimate the probability density within a hyperrectangle for an MVN distr.
    
    Use the method of Alan Genz (1992) to estimate the probability density
    of a multivariate normal distribution within an n-orthotope (i.e. 
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
    mu = np.asarray(mu)
    if mu.shape == ():
        mu = np.asarray([mu])
        COV = np.asarray([COV])
    else:
        COV = np.asarray(COV)
    sig = np.sqrt(np.diag(COV))
    corr = COV / np.outer(sig,sig)

    ndim = len(mu)
    
    if lower is None:
        lower = -np.ones(ndim) * np.inf
    else:
        lower = np.asarray(lower)
    
    if upper is None:
        upper = np.ones(ndim) * np.inf
    else:
        upper = np.asarray(upper)
    
    # standardize the truncation limits
    lower = (lower-mu)/sig
    upper = (upper-mu)/sig
    
    # prepare the flags for infinite bounds (these are needed for the mvndst
    # function)
    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)
    infin = 2.0*np.ones(ndim)
    
    np.putmask(infin, lowinf, 0)
    np.putmask(infin, uppinf, 1)
    np.putmask(infin, lowinf*uppinf, -1)
    
    # prepare the correlation coefficients
    if ndim == 1:
        correl = 0
    else:
        correl = corr[np.tril_indices(ndim, -1)]
    
    # estimate the density
    eps_alpha, alpha, __ = kde.mvn.mvndst(lower, upper, infin, correl)
    
    return alpha, eps_alpha

def tmvn_MLE(samples,
             tr_lower=None, tr_upper=None,
             censored_count=0, det_lower=None, det_upper=None,
             alpha_lim=None):
    """
    Fit a truncated multivariate normal distribution to samples using MLE.

    The number of dimensions of the distribution function are inferred from the
    shape of the sample data. Censoring is automatically considered if the 
    number of censored samples and the corresponding detection limits are 
    provided. Infinite or unspecified truncation limits lead to fitting a 
    non-truncated normal distribution in that dimension.

    Parameters
    ----------
    samples: ndarray
        Raw data that serves as the basis of estimation. The number of samples
        equals the number of columns and each row introduces a new feature. In
        other words: a list of sample lists is expected where each sample list
        is a collection of samples of one variable.
    tr_lower: float vector, optional, default: None
        Lower bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from below
        in a subset of the dimensions, use either `None` or assign an infinite 
        value (i.e. -numpy.inf) to those dimensions.
    tr_upper: float vector, optional, default: None
        Upper bound(s) for the truncated distributions. A scalar value can be
        used for a univariate case, while a list of bounds is expected in
        multivariate cases. If the distribution is non-truncated from above
        in a subset of the dimensions, use either `None` or assign an infinite 
        value (i.e. numpy.inf) to those dimensions.    
    censored_count: int, optional, default: None
        The number of censored samples that are beyond the detection limits. 
        All samples outside the detection limits are aggregated into one set.
        This works the same way in one and in multiple dimensions. Prescription
        of specific censored sample counts for sub-regions of the input space 
        outside the detection limits is not supported.
    det_lower: float ndarray, optional, default: None
        Lower detection limit(s) for censored data. In multivariate cases the 
        limits need to be defined as a vector; a scalar value is sufficient in 
        a univariate case. If the data is not censored from below in a 
        particular dimension, assign None to that position of the ndarray.
    det_upper: float ndarray, optional, default: None
        Upper detection limit(s) for censored data. In multivariate cases the 
        limits need to be defined as a vector; a scalar value is sufficient in 
        a univariate case. If the data is not censored from above in a 
        particular dimension, assign None to that position of the ndarray.
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
    mu: float scalar or ndarray
        Mean of the fitted probability distribution. A vector of means is 
        returned in a multivariate case.
    COV: float scalar or 2D ndarray
        Covariance matrix of the fitted probability distribution. A 2D square 
        ndarray is returned in a multi-dimensional case, while a single 
        variance (not standard deviation!) value is returned in a univariate
        case.

    """
    
    # extract some basic information about the number of dimensions and the 
    # number of samples from raw data
    samples = np.asarray(samples)
    if samples.ndim == 1:
        ndims = 1
        nsamples = len(samples)
        samplesT = samples
    else:
        ndims, nsamples = samples.shape
        samplesT = np.transpose(samples)
    
    # define initial values of distribution parameters using simple estimates
    if ndims == 1:
        mu_init = np.mean(samples)
        # use biased estimate for std, because MLE will converge to that anyway
        sig_init = np.std(samples, ddof=0) 
        # prepare a vector of initial values
        inits = np.asarray([mu_init, sig_init])
    else:
        mu_init = np.mean(samples, axis=1)
        # use biased estimate, see comment above
        sig_init = np.std(samples, axis=1, ddof=0)
        rho_init = np.corrcoef(samples)
        # collect the independent values (i.e. elements above the main 
        # diagonal) from the correlation matrix in a list
        rho_init_ids = np.triu_indices(ndims, k=1)
        rho_init_list = rho_init[rho_init_ids]
        # save the ids of the elements below the main diagonal for future use
        rho_init_ids2 = rho_init_ids[::-1]
        # prepare a vector of initial values
        inits=np.concatenate([mu_init, sig_init, rho_init_list])
    
    # define the bounds for the distribution parameters
    # mu is not bounded
    mu_bounds = [(-np.inf, np.inf) for t in range(ndims)]
    # sig is bounded below at (0
    sig_bounds = [(np.nextafter(0,1), np.inf) for s in range(ndims)]
    # rho is bounded on both sides by (-1,1)
    # Note that -1.0 and 1.0 are not allowed to avoid numerical problems due to
    # a singular and/or non-positive definite covariance matrix
    if ndims > 1:
        rho_bounds = [(-1.+1e-3, 1.-1e-3) for r in range(len(rho_init_list))]
    else:
        # there is no need for rho bounds in a univariate case
        rho_bounds = []
    # create a lower and an upper bounds vector
    bounds = mu_bounds + sig_bounds + rho_bounds
    bnd_lower, bnd_upper = np.transpose(bounds)
    
    # create a convenience function that converts a vector of distribution 
    # parameters to the standard mu and COV arrays
    def _get_mu_COV(params, unbiased=False):
        """
        The unbiased flag controls if the bias in standard deviation 
        estimates shall be corrected during conversion.
        """
        if ndims == 1:
            mu, COV = params
        else:
            mu = params[:ndims]
            sig = params[ndims:2*ndims]
            if unbiased:
                sig = sig * nsamples / (nsamples-1)
            rho_list = params[2*ndims:]
            
            # reconstruct the covariance matrix
            COV = np.outer(sig, sig)
            # add correlation estimates above...
            COV[rho_init_ids] = COV[rho_init_ids] * rho_list
            # and below the main diagonal
            COV[rho_init_ids2] = COV[rho_init_ids2] * rho_list
            
        return mu, COV
    
    # create the negative log likelihood function for censored data from a 
    # truncated multivariate normal distribution
    def _neg_log_likelihood(params):
        
        # first, check if the parameters are within the pre-defined bounds
        if ((params > bnd_lower) & (params < bnd_upper)).all(0) == False:
            # if they are not, then return an infinite value to discourage the
            # optimization algorithm from going in that direction
            return np.inf
        
        # reconstruct the mu and COV arrays from the parameters
        mu, COV = _get_mu_COV(params)
        
        # calculate the probability density within the truncation limits
        if (tr_lower is not None) and (tr_upper is not None):
            alpha, eps_alpha = mvn_orthotope_density(mu, COV, 
                                                     tr_lower, tr_upper)

            # If the error in the alpha estimate is too large, then we are 
            # beyond the applicability limits of the function used for 
            # estimating alpha. Show a warning message and try to find another 
            # solution by discouraging the optimization algorithm from going in 
            # this direction.
            if alpha <= 100.*eps_alpha: #i.e. max. error is limited at 1%
                # Note: throwing an error here would be too extreme, because it
                # would stop the analysis completely, while a solution might be 
                # reached if we let the optimization algorithm converge to it.
                if msg[0] == False:
                    warnings.warn(UserWarning(
                        'The density of the joint probability distribution '
                        'within the truncation limits is too small and '
                        'cannot be estimated with sufficiently high '
                        'accuracy.'
                    ))
                    msg[0] = True
                return np.inf
            
            # If a lower limit was prescribed for alpha, it should also be 
            # enforced here
            if (alpha_lim is not None) and (alpha < alpha_lim):
                if msg[1] == False:
                    warnings.warn(UserWarning(
                        'The density of the joint probability distribution '
                        'within the truncation limits is less than the '
                        'prescribed minimum limit.'
                    ))
                    msg[1] = True
                return np.inf
        
        else:
            alpha, eps_alpha = 1., 0.
        
        # calculate the likelihood for each available sample
        likelihoods = multivariate_normal.pdf(samplesT, mean=mu, cov=COV,
                                              allow_singular=True)
        
        # Zeros are a result of limited floating point precision. Replace them 
        # with the smallest possible positive floating point number to 
        # improve convergence.
        likelihoods = np.clip(likelihoods, a_min=np.nextafter(0,1), a_max=None)
        
        # calculate the likelihoods corresponding to censored data (if any)
        if censored_count > 0:
            # calculate the probability density within the detection limits
            det_alpha, eps_alpha = mvn_orthotope_density(mu, COV, 
                                                         det_lower, det_upper)
            # Similarly to alpha above, make sure that det_alpha is estimated
            # with sufficient accuracy.
            if det_alpha <= 100.*eps_alpha:
                if msg[2] == False:
                    warnings.warn(
                        'The density of the joint probability distribution '
                        'within the detection limits is too small and '
                        'cannot be estimated with sufficiently high '
                        'accuracy.'
                    )
                    msg[2] = True
                return np. inf
            
            # calculate the likelihood of censoring a sample
            cen_likelihood = (alpha - det_alpha) / alpha
            
            # make sure that the likelihood is a positive number
            cen_likelihood = max(cen_likelihood, np.nextafter(0,1))
            
        else:
            # If the data is not censored, use 1.0 for cen_likelihood to get a
            # zero log-likelihood later. Note that although this is 
            # theoretically not correct, it does not alter the solution and 
            # it is numerically much more convenient than working around the 
            # log of zero likelihood.
            cen_likelihood = 1.
            
        # calculate the total negative log-likelihood
        NLL = -(
            np.sum(np.log(likelihoods))                 # from samples 
            - nsamples*np.log(alpha)                    # truncation influence
            + censored_count*np.log(cen_likelihood)     # censoring influence
        )
        
        # normalize the likelihoods with the sample count
        NLL = NLL/nsamples
        #print(mu, NLL)
        
        return NLL
    
    # initialize the message flags
    msg = [False, False, False]
    
    # minimize the negative log-likelihood function using the adaptive 
    # Nelder-Mead algorithm (Gao and Han, 2012)
    out = minimize(_neg_log_likelihood, inits, method='Nelder-Mead',
                   options={'maxfev': 400*ndims, 
                            'xatol': np.max(inits[:ndims])*0.1, 
                            'fatol': 5e-5 * ndims, 
                            'adaptive': True})
    #print(out.fun, out.nfev)
    
    # reconstruct the mu and COV arrays from the solutions and return them
    mu, COV = _get_mu_COV(out.x, unbiased=True)
    
    return mu, COV


class RandomVariable(object):
    """
    Characterizes a Random Variable (RV) that represents a source of 
    uncertainty in the calculation.

    The uncertainty can be described either through raw data or through a 
    pre-defined distribution function. When using raw data, provide potentially 
    correlated raw samples in an N dimensional array. If the data is left or
    right censored in any number of its dimensions, provide the list of 
    detection limits and the number of censored samples. No other information 
    is needed to define the object from raw data. Then, either resample the raw 
    data, or fit a prescribed distribution to the samples and sample from that 
    distribution later. Alternatively, one can choose to prescribe a 
    distribution type and its parameters and sample from that distribution 
    later.

    Parameters
    ----------
    ID: int
    dimension_tags: str array
        A series of strings that identify the stochastic model parameters that 
        correspond to each dimension of the random variable. When the RV is one 
        dimensional, the dim_tag is a single string. In multi-dimensional 
        cases, the order of strings shall match the order of elements provided 
        as other inputs. 
    raw_data: float scalar or ndarray, optional, default: None
        Samples of an uncertain variable. The samples can describe a 
        multi-dimensional random variable if they are arranged in a 
        multi-dimensional ndarray.
    detection_limits: float ndarray, optional, default: None
        Defines the limits for censored data. If the raw data is 
        multi-dimensional, the limits need to be defined in a 2D ndarray that
        is structured as a series of vectors with two elements: left and right
        limits. If the data is not censored in a particular direction, assign
        None to that position of the ndarray.
    censored_count: int, optional, default: None
        The number of censored samples that are beyond the detection limits. 
        All samples outside the detection limits are aggregated into one set. 
        This works the same way in one and in multiple dimensions. Prescription 
        of censored sample counts for sub-regions of the input space outside 
        the detection limits is not yet supported. If such an approach is 
        desired, the censored raw data shall be used to fit a distribution in a 
        pre-processing step and the fitted distribution can be specified for 
        this random variable.
    distribution_kind: {'normal', 'lognormal', 'truncated_normal', 'truncated_lognormal', 'multinomial'}, optional, default: None
        Defines the type of probability distribution when raw data is not
        provided, but the distribution is directly specified.
    theta: float scalar or ndarray, optional, default: None
        Median of the probability distribution. A vector of medians is expected
        in a multi-dimensional case. 
    COV: float scalar or 2D ndarray, optional, default: None
        Covariance matrix of the random variable. In a multi-dimensional case
        this parameter has to be a 2D square ndarray, and the number of its 
        rows has to be equal to the number of elements in the supplied theta 
        vector. In a one-dimensional case, a single value is expected that 
        equals the variance (not the standard deviation!) of the distribution.
        The COV for lognormal distributions is assumed to be specified in 
        logarithmic space. 
    p_set: float 1D ndarray, optional, default: None
        Probabilities of a finite set of events described by a multinomial
        distribution. The RV will have binomial distribution if only one
        element is provided in this vector. The number of events equals the 
        number of vector elements if their probabilities sum up to 1.0. If the
        sum is less than 1.0, then an additional event is assumed with the 
        remaining probability of occurrence assigned to it. The sum of 
        event probabilities shall never be more than 1.0.
    min_value: float scalar or ndarray, optional, default: None
        Lower bound(s) for the truncated distribution functions. 
        Multi-dimensional truncated distributions require a vector of bounds.
    max_value: float scalar or ndarray, optional, default: None
        Upper bound(s) for the truncated distribution functions. 
        Multi-dimensional truncated distributions require a vector of bounds.

    """

    def __init__(self, ID, dimension_tags,
                 raw_data=None, detection_limits=None, censored_count=None,
                 distribution_kind=None,
                 theta=None, COV=None, p_set=None,
                 min_value=None, max_value=None):

        self._ID = ID

        self._dimension_tags = np.asarray(dimension_tags)

        if raw_data is not None:
            raw_data = np.asarray(raw_data)
        self._raw_data = raw_data

        if detection_limits is not None:
            detection_limits = np.asarray(detection_limits)
        self._detection_limits = detection_limits

        self._censored_count = censored_count

        self._distribution_kind = distribution_kind

        if theta is not None:
            theta = np.asarray(theta)
        self._theta = theta

        if COV is not None:
            COV = np.asarray(COV)
        self._COV = COV

        if p_set is not None:
            p_set = np.asarray(p_set)
        self._p_set = p_set

        if min_value is not None:
            min_value = np.asarray(min_value)
        self._min_value = min_value

        if max_value is not None:
            max_value = np.asarray(max_value)
        self._max_value = max_value

        # perform some basic checks to make sure that the provided data will be
        # sufficient to define a random variable
        if self._raw_data is None:

            # if the RV is defined by providing distribution data...
            if self._distribution_kind is None:
                raise ValueError(
                    "Either raw samples or a distribution needs to be defined "
                    "for a random variable."
                )

            if (self._distribution_kind in ['normal', 'lognormal']
                and (self._theta is None or self._COV is None)):
                raise ValueError(
                    "Normal and lognormal distributions require theta and "
                    "COV parameters."
                )

            if (self._distribution_kind in ['truncated_normal',
                                            'truncated_lognormal']
                and (self._theta is None or self._COV is None
                     or
                     (self._min_value is None and self._max_value is None))):
                raise ValueError(
                    "Truncated normal and lognormal distributions require "
                    "theta, COV, and at least a minimum or a maximum "
                    "boundary value as parameters."
                )

            if (self._distribution_kind in ['multinomial']
                and self._p_set is None):
                raise ValueError(
                    "Multinomial distributions require a set of p values as "
                    "parameters."
                )
        else:

            # if the RV is defined through raw samples
            if ((self._detection_limits is None) !=
                (self._censored_count is None)):
                raise ValueError(
                    "Definition of censored data requires information about "
                    "the detection limits and the number of censored samples."
                )

    @property
    def theta(self):
        """
        Return the median value(s) of the probability distribution.
        """
        if self._theta is not None:
            return self._theta
        else:
            raise ValueError(
                "The median of the probability distribution of this random "
                "variable is not yet specified."
            )

    @property
    def COV(self):
        """
        Return the covariance matrix of the probability distribution.
        """
        if self._COV is not None:
            return self._COV
        else:
            raise ValueError(
                "The covariance matrix of the probability distribution of "
                "this random variable is not yet specified."
            )