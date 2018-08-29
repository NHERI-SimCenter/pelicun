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
This module defines constants, basic classes and methods for pelicun.

.. rubric:: Contents

.. autosummary::

    RandomVariable

"""

import numpy as np
from scipy.stats import norm, truncnorm

# Constants for unit conversion
kN = 1e3

MPa = 1e6
GPa = 1e9

g = 9.80665

mm = 0.001
mm2 = mm**2.

cm = 0.01
cm2 = cm**2.

km = 1000.
km2 = km**2.

inch = 0.0254
inch2 = inch**2.

ft = 12. * inch
ft2 = ft**2.

mile = 5280. * ft
mile2 = mile**2.


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
    censored_count: int scalar, optional, default: None
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