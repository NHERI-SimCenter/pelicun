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

    RandomVariable
    RandomVariableSet
    RandomVariableRegistry


"""

from .base import *

from scipy.stats import uniform, norm
from scipy.linalg import cholesky, svd

import warnings


class RandomVariable(object):
    """
    Description

    Parameters
    ----------
    name: string
        A unique string that identifies the random variable.
    distribution: {'normal', 'lognormal', 'multinomial', 'custom', 'empirical'
        }, optional
        Defines the type of probability distribution for the random variable.
    theta: float scalar or ndarray, optional
        Set of parameters that define the cumulative distribution function of
        the variable given its distribution type. The following parameters are
        expected currently for the supported distribution types: normal - mean,
        standard deviation; lognormal - median, log standard deviation; uniform
        - a, b, the lower and upper bounds of the distribution; multinomial -
        likelihood of all but one unique events (the last event's likelihood is
        set automatically to ensure the likelihoods sum up to one); custom -
        according to the custom expression provided; empirical - N/A.
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
    parent: RandomVariable, optional
        Provides a direct link to another variable. If the parent is not None,
        this variable will become an alias for the parent.

    """

    def __init__(self, name, distribution, theta=None, truncation_limits=None,
                 bounds=None, custom_expr=None, samples=None, parent=None):

        self.name = name

        # use the parent as reference if it is available
        if parent is not None:
            self.ref = parent
        else:
            self.ref = self

        # save the other parameters internally
        self._distribution = distribution
        self._theta = theta
        self._truncation_limits = truncation_limits
        self._bounds = bounds
        self._custom_expr = custom_expr
        self._samples = samples
        self._parent = parent

    @property
    def distribution(self):
        """
        Return the assigned probability distribution type.
        """
        return self.ref._distribution

    @property
    def theta(self):
        """
        Return the assigned probability distribution parameters.
        """
        return self.ref._theta

    @property
    def truncation_limits(self):
        """
        Return the assigned truncation limits.
        """
        return self.ref._truncation_limits

    @property
    def bounds(self):
        """
        Return the assigned probability bounds.
        """
        return self.ref._bounds

    @property
    def custom_expr(self):
        """
        Return the assigned custom expression for CDF.
        """
        return self.ref._custom_expr

    @property
    def samples(self):
        """
        Return the empirical or generated samples.
        """
        return self.ref._samples

    @samples.setter
    def samples(self, value):
        """
        Assign samples to the random variable
        """
        self.ref._samples = value

    @property
    def parent(self):
        """

        Return the parent of the variable (if any).
        """
        return self._parent

    def inverse_transform_sampling(self, uni_samples):
        """
        Creates samples using inverse probability integral transformation.

        Parameters
        ----------
        uni_samples: float ndarray
            An array of floating point values in the [0, 1] domain.
        """
        if self.distribution == 'normal':
            self.samples = norm.ppf(uni_samples,
                                    loc=self.theta[0], scale=self.theta[1])

        elif self.distribution == 'lognormal':
            self.samples = np.exp(norm.ppf(uni_samples,
                                           loc=np.log(self.theta[0]),
                                           scale=self.theta[1]))
        elif self.distribution == 'uniform':
            self.samples = uniform.ppf(uni_samples,
                                       loc=self.theta[0],
                                       scale=self.theta[1]-self.theta[0])

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

        # put the RVs in a dictionary for more efficient access
        reorder = np.argsort([RV.name for RV in RV_list])
        self._variables = dict([(RV_list[i].name, RV_list[i]) for i in reorder])

        # reorder the entries in the correlation matrix to correspond to the
        # sorted list of RVs
        self._Rho = Rho[(reorder)].T[(reorder)].T

    @property
    def RV(self):
        """
        Return the random variable(s) assigned to the set
        """
        return self._variables

    def apply_correlation(RV_set, U_iid_samples):
        """
        Apply correlation to n dimensional uniform samples.

        Currently, correlation is applied using a Gaussian copula. First, we
        try using Cholesky transformation. If the correlation matrix is not
        positive semidefinite and Cholesky fails, use SVD to apply the
        correlations while preserving as much as possible from the correlation
        matrix.

        Parameters
        ----------
        U_iid_samples: 2D ndarray of float
            An array of iid, uniformly distributed floating point values in the
            [0, 1] domain.

        Returns
        -------
        U_corr_samples: 2D ndarray of float
            An array of correlated, uniformly distributed values in n
            dimensions.
        """
        U_RV = U_iid_samples  # just for the sake of convenience

        # First try doing the Cholesky transformation
        try:
            N_RV = norm.ppf(U_RV)

            L = cholesky(RV_set._Rho, lower=True)

            NC_RV = L @ N_RV

            UC_RV = norm.cdf(NC_RV)

        except:

            # if the Cholesky doesn't work, we need to use the more
            # time-consuming but more robust approach based on SVD
            N_RV = norm.ppf(U_RV)

            U, s, __ = svd(RV_set._Rho, )
            S = np.diagflat(np.sqrt(s))

            NC_RV = (N_RV.T @ S @ U.T).T

            UC_RV = norm.cdf(NC_RV)

        U_corr_samples = UC_RV

        return U_corr_samples

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
        Return the random variable(s) in the registry
        """
        return self._variables

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
    def RV_samples(self):
        """
        Return the samples for every random variable in the registry
        """
        return dict([(name, rv.samples) for name,rv in self.RV.items()])


    def generate_samples(self, sample_size, method='LHS_midpoint', seed=1):
        """
        Generates samples for all variables in the registry.

        Parameters
        ----------

        sample_size: int
            The number of samples requested per variable.
        method: {'random', 'LHS', 'LHS_midpoint'}, optional
            The sample generation method to use. 'random' stands for
            conventional random sampling; 'LHS' is Latin HyperCube Sampling
            with random sample location within each bin of the hypercube;
            'LHS_midpoint' is like LHS, but the samples are assigned to the
            midpoints of the hypercube bins.
        seed: int, optional
            Random seed used for sampling.
        """
        # Initialize the random number generator
        rng = np.random.default_rng(seed)

        # Generate a dictionary with IDs
        RV_ID = dict([(RV_name, ID) for ID, RV_name in enumerate(self.RV)])
        RV_count = len(RV_ID)

        # Generate samples from a uniform distribution for every RV
        if 'LHS' in method:
            bin_low = np.array([rng.permutation(sample_size)
                                for i in range(RV_count)])

            if method == 'LHS_midpoint':
                U_RV = np.ones([RV_count, sample_size]) * 0.5
                U_RV = (bin_low + U_RV) / sample_size

            elif method == 'LHS':
                U_RV = rng.random(size=[RV_count, sample_size])
                U_RV = (bin_low + U_RV) / sample_size

        elif method == 'random':
            U_RV = rng.random(size=[RV_count, sample_size])

        # Apply correlations for the pre-defined sets
        for RV_set_name, RV_set in self.RV_set.items():
            # get the list of random variables in the set
            set_ID_list = [RV_ID[name] for name in RV_set.RV.keys()]
            # prepare the correlated uniform distribution for the set
            U_RV[set_ID_list] = RV_set.apply_correlation(U_RV[set_ID_list])

        # Convert from uniform to the target distribution for each variable
        for (RV_name, RV), u_RV in zip(self.RV.items(), U_RV):
            RV.inverse_transform_sampling(u_RV)
