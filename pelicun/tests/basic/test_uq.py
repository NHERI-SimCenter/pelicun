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
These are unit and integration tests on the uq module of pelicun.

Note: Test functions that require reading the expected test results
from a file should include a reset=False argument to enable automatic
reset from the `reset_all_test_data` function in `reset_tests.py`.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest
from scipy.stats import (
    lognorm,  # type: ignore
    norm,  # type: ignore
    weibull_min,  # type: ignore
)

from pelicun import uq
from pelicun.base import ensure_value
from pelicun.tests.util import export_pickle, import_pickle

# The tests maintain the order of definitions of the `uq.py` file.

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.


def test_scale_distribution() -> None:
    # used in all cases
    theta = np.array((-1.00, 1.00))
    trunc = np.array((-2.00, 2.00))

    # case 1:
    # normal distribution, factor of two
    theta_new, truncation_limits = uq.scale_distribution(
        2.00, 'normal', theta, trunc
    )
    assert truncation_limits is not None
    assert np.allclose(theta_new, np.array((-2.00, 1.00)))
    assert np.allclose(truncation_limits, np.array((-4.00, 4.00)))

    # case 2:
    # normal_std distribution, factor of two
    res = uq.scale_distribution(2.00, 'normal_std', theta, trunc)
    assert res[1] is not None
    assert np.allclose(res[0], np.array((-2.00, 2.00)))  # theta_new
    assert np.allclose(res[1], np.array((-4.00, 4.00)))  # truncation_limits

    # case 3:
    # normal_cov distribution, factor of two
    res = uq.scale_distribution(2.00, 'normal_cov', theta, trunc)
    assert res[1] is not None
    assert np.allclose(res[0], np.array((-2.00, 1.00)))  # theta_new
    assert np.allclose(res[1], np.array((-4.00, 4.00)))  # truncation_limits

    # case 4:
    # lognormal distribution, factor of two
    res = uq.scale_distribution(2.00, 'lognormal', theta, trunc)
    assert res[1] is not None
    assert np.allclose(res[0], np.array((-2.00, 1.00)))  # theta_new
    assert np.allclose(res[1], np.array((-4.00, 4.00)))  # truncation_limits

    # case 5:
    # uniform distribution, factor of two
    res = uq.scale_distribution(2.00, 'uniform', theta, trunc)
    assert res[1] is not None
    assert np.allclose(res[0], np.array((-2.00, 2.00)))  # theta_new
    assert np.allclose(res[1], np.array((-4.00, 4.00)))  # truncation_limits

    # case 6: unsupported distribution
    with pytest.raises(
        ValueError, match='Unsupported distribution: benktander-weibull'
    ):
        uq.scale_distribution(0.50, 'benktander-weibull', np.array((1.00, 10.00)))


def test_mvn_orthotope_density() -> None:
    # case 1:
    # zero-width slice should result in a value of zero.
    mu_val = 0.00
    cov_val = 1.00
    lower_val = -1.00
    upper_val = -1.00
    res = uq.mvn_orthotope_density(
        mu_val, np.atleast_2d([cov_val]), lower_val, upper_val
    )
    assert np.allclose(res, np.array((0.00, 2.00e-16)))

    # case 2:
    # all negative values should result in a value of 0.50
    mu_val = 0.00
    cov_val = 1.00
    lower_val = np.nan
    upper_val = 0.00
    res = uq.mvn_orthotope_density(
        mu_val, np.atleast_2d([cov_val]), lower_val, upper_val
    )
    assert np.allclose(res, np.array((0.50, 2.00e-16)))

    # case 3:
    # similar to 2, all positive values should result in 0.50
    mu_val = 0.00
    cov_val = 1.00
    lower_val = 0.00
    upper_val = np.nan
    res = uq.mvn_orthotope_density(
        mu_val, np.atleast_2d([cov_val]), lower_val, upper_val
    )
    assert np.allclose(res, np.array((0.50, 2.00e-16)))

    # case 4:
    # bivariate standard normal, hyperrectangle occupying positive
    # domain should result in 1/4
    mu_arr = np.array((0.00, 0.00))
    cov_mat = np.array(((1.00, 0.00), (0.00, 1.00)))
    lower_arr = np.array((0.00, 0.00))
    upper_arr = np.array((np.nan, np.nan))
    res = uq.mvn_orthotope_density(mu_arr, cov_mat, lower_arr, upper_arr)
    assert np.allclose(res, np.array((1.00 / 4.00, 2.00e-16)))

    # case 5:
    # bivariate normal with correlation
    mu_arr = np.array((0.00, 0.00))
    cov_arr = np.array(((1.00, 0.50), (0.50, 1.00)))
    lower_arr = np.array((0.00, 0.00))
    upper_arr = np.array((np.nan, np.nan))
    res = uq.mvn_orthotope_density(mu_arr, cov_arr, lower_arr, upper_arr)
    assert np.allclose(res, np.array((1.00 / 3.00, 2.00e-16)))

    # case 6:
    # multivariate 3-D standard normal, hyperrectangle occupying
    # positive domain should result in 1/8
    mu_arr = np.array((0.00, 0.00, 0.00))
    cov_arr = np.eye(3)
    lower_arr = np.array((0.00, 0.00, 0.00))
    upper_arr = np.array((np.nan, np.nan, np.nan))
    res = uq.mvn_orthotope_density(mu_arr, cov_arr, lower_arr, upper_arr)
    assert np.allclose(res, np.array((1.00 / 8.00, 2.00e-16)))


def test__get_theta() -> None:
    # Evaluate uq._get_theta() for some valid inputs
    res = uq._get_theta(
        np.array(((1.00, 1.00), (1.00, 0.5), (0.00, 0.3), (1.50, 0.2))),
        np.array(((0.00, 1.00), (1.00, 0.5), (0.00, 0.3), (1.00, 0.2))),
        np.array(['normal', 'lognormal', 'normal_std', 'normal_cov']),
    )

    # Check that the expected output is obtained for each distribution type
    expected_res = np.array(
        ((1.00, 2.00), (2.00, 0.82436064), (0.00, 0.60), (1.36642083, 0.24428055))
    )

    assert np.allclose(res, expected_res)

    # Check that it fails for invalid inputs
    with pytest.raises(
        ValueError, match='Unsupported distribution: not_a_distribution'
    ):
        uq._get_theta(
            np.array((1.00,)), np.array((1.00,)), np.array(['not_a_distribution'])
        )


def test__get_limit_probs() -> None:
    # verify that it works for valid inputs

    res = uq._get_limit_probs(
        np.array((0.10, 0.20)), 'normal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.4800611941616275, 0.5199388058383725)))

    res = uq._get_limit_probs(
        np.array((np.nan, 0.20)), 'normal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.0, 0.5199388058383725)))

    res = uq._get_limit_probs(
        np.array((0.10, np.nan)), 'normal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.4800611941616275, 1.0)))

    res = uq._get_limit_probs(
        np.array((np.nan, np.nan)), 'normal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.0, 1.0)))

    res = uq._get_limit_probs(
        np.array((0.10, 0.20)), 'lognormal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.4800611941616275, 0.5199388058383725)))

    res = uq._get_limit_probs(
        np.array((np.nan, 0.20)), 'lognormal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.0, 0.5199388058383725)))

    res = uq._get_limit_probs(
        np.array((0.10, np.nan)), 'lognormal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.4800611941616275, 1.0)))

    res = uq._get_limit_probs(
        np.array((np.nan, np.nan)), 'lognormal', np.array((0.15, 1.00))
    )
    assert np.allclose(res, np.array((0.0, 1.0)))

    # verify that it fails for invalid inputs

    with pytest.raises(
        ValueError, match='Unsupported distribution: not_a_distribution'
    ):
        uq._get_limit_probs(
            np.array((1.00,)),
            'not_a_distribution',
            np.array((1.00,)),
        )


def test__get_std_samples() -> None:
    # test that it works with valid inputs

    # case 1:
    # univariate samples
    samples = np.array(((1.00, 2.00, 3.00),))
    theta = np.array(((0.00, 1.0),))
    tr_limits = np.array(((np.nan, np.nan),))
    dist_list = np.array(('normal',))
    res = uq._get_std_samples(samples, theta, tr_limits, dist_list)
    assert np.allclose(res, np.array((1.00, 2.00, 3.00)))

    # case 2:
    # multivariate samples
    samples = np.array(
        (
            (0.657965, 1.128253, 1.044239, 1.599209),
            (1.396495, 1.435923, 2.055659, 1.416298),
            (1.948161, 1.576571, 1.469571, 1.190853),
        )
    )
    theta = np.array(
        (
            (1.00, 0.20),
            (1.50, 0.6),
            (1.30, 2.0),
        )
    )
    tr_limits = np.array(
        (
            (np.nan, np.nan),
            (1.10, np.nan),
            (np.nan, 2.80),
        )
    )
    dist_list = np.array(('normal', 'lognormal', 'normal'))
    res = uq._get_std_samples(samples, theta, tr_limits, dist_list)
    assert np.allclose(
        res,
        np.array(
            (
                (-1.710175, 0.641265, 0.221195, 2.996045),
                (-0.70791883, -0.60009227, 0.7158206, -0.65293631),
                (0.88090031, 0.57580461, 0.49642554, 0.30123205),
            )
        ),
    )

    # test that it fails for invalid inputs
    with pytest.raises(
        ValueError, match='Unsupported distribution: some_unsupported_distribution'
    ):
        uq._get_std_samples(
            np.array(((1.00, 2.00, 3.00),)),
            np.array(((0.00, 1.0),)),
            np.array(((np.nan, np.nan),)),
            np.array(('some_unsupported_distribution',)),
        )


def test__get_std_corr_matrix() -> None:
    # test that it works with valid inputs

    # case 1:
    std_samples = np.array(((1.00,),))
    res = uq._get_std_corr_matrix(std_samples)
    assert res is not None
    assert np.allclose(res, np.array(((1.00,),)))

    # case 2:
    std_samples = np.array(((1.00, 0.00), (0.00, 1.00)))
    res = uq._get_std_corr_matrix(std_samples)
    assert res is not None
    assert np.allclose(res, np.array(((1.00, 0.00), (0.00, 1.00))))

    # case 3:
    std_samples = np.array(((1.00, 0.00), (0.00, -1.00)))
    res = uq._get_std_corr_matrix(std_samples)
    assert res is not None
    assert np.allclose(res, np.array(((1.00, 0.00), (0.00, 1.00))))

    # case 4:
    std_samples = np.array(((1.00, 1.00), (1.00, 1.00)))
    res = uq._get_std_corr_matrix(std_samples)
    assert res is not None
    assert np.allclose(res, np.array(((1.00, 1.00), (1.00, 1.00))))

    # case 5:
    std_samples = np.array(((1.00, 1e50), (-1.00, -1.00)))
    res = uq._get_std_corr_matrix(std_samples)
    assert res is not None
    assert np.allclose(res, np.array(((1.00, 0.00), (0.00, 1.00))))

    # test that it fails for invalid inputs

    for bad_item in (np.nan, np.inf, -np.inf):
        x = np.array(((1.00, bad_item), (-1.00, -1.00)))
        with pytest.raises(
            ValueError, match='std_samples array must not contain inf or NaN values'
        ):
            uq._get_std_corr_matrix(x)


def test__mvn_scale() -> None:
    # case 1:
    np.random.seed(40)
    sample = np.random.normal(0.00, 1.00, size=(2, 5)).T
    rho = np.array(((1.00, 0.00), (0.00, 1.00)))
    res = uq._mvn_scale(sample, rho)
    assert np.allclose(res, np.array((1.0, 1.0, 1.0, 1.0, 1.0)))

    # case 2:
    np.random.seed(40)
    sample = np.random.normal(1.0e10, 1.00, size=(2, 5)).T
    rho = np.array(((1.00, 0.00), (0.00, 1.00)))
    res = uq._mvn_scale(sample, rho)
    assert np.allclose(res, np.array((0.0, 0.0, 0.0, 0.0, 0.0)))


def test__neg_log_likelihood() -> None:
    # Parameters not within the pre-defined bounds should yield a
    # large value to discourage the optimization algorithm from going
    # in that direction.
    res = uq._neg_log_likelihood(
        params=np.array((1e8, 0.20)),
        inits=np.array((1.00, 0.20)),
        bnd_lower=np.array((0.00, 0.00)),
        bnd_upper=np.array((20.00, 1.00)),
        samples=np.array(
            (
                (0.90, 0.10),
                (1.10, 0.30),
            ),
        ),
        dist_list=np.array(('normal', 'normal')),
        tr_limits=np.array((np.nan, np.nan)),
        det_limits=[np.array((np.nan, np.nan))],
        censored_count=0,
        enforce_bounds=True,
    )

    assert res == 1e10

    # if there is nan in the parameters, the function should return a large value.
    res = uq._neg_log_likelihood(
        np.array((np.nan, 0.20)),
        np.array((1.00, 0.20)),
        np.atleast_1d((0.00,)),
        np.atleast_1d((20.00,)),
        np.array(
            (
                (0.90, 0.10),
                (1.10, 0.30),
            ),
        ),
        np.array(('normal', 'normal')),
        np.array((-np.inf, np.inf)),
        [np.array((np.nan, np.nan))],
        0,
        enforce_bounds=False,
    )

    assert res == 1e10


def test_fit_distribution_to_sample_univariate() -> None:
    # a single value in the sample
    sample_vec = np.array((1.00,))
    res = uq.fit_distribution_to_sample(sample_vec, 'normal')
    assert np.isclose(res[0][0, 0], 1.00)
    assert np.isclose(res[0][0, 1], 1e-06)
    assert np.isclose(res[1][0, 0], 1.00)

    # baseline case
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape(
        (1, -1)
    )
    res = uq.fit_distribution_to_sample(sample_vec, 'normal')
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], np.inf)
    assert np.isclose(res[1][0, 0], 1.00)
    res = uq.fit_distribution_to_sample(sample_vec, 'normal_std')
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], 2.0)
    assert np.isclose(res[1][0, 0], 1.00)
    res = uq.fit_distribution_to_sample(sample_vec, 'normal_cov')
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], np.inf)
    assert np.isclose(res[1][0, 0], 1.00)

    # baseline case where the cov=mu/sigma is defined
    sample_vec += 10.00
    res = uq.fit_distribution_to_sample(sample_vec, 'normal_cov')
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], np.std(sample_vec) / np.mean(sample_vec))
    assert np.isclose(res[1][0, 0], 1.00)
    res = uq.fit_distribution_to_sample(sample_vec, 'normal_std')
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], np.std(sample_vec))
    assert np.isclose(res[1][0, 0], 1.00)
    res = uq.fit_distribution_to_sample(sample_vec, 'normal')
    assert np.isclose(res[0][0, 0], np.mean(sample_vec))
    assert np.isclose(res[0][0, 1], np.std(sample_vec) / np.mean(sample_vec))
    assert np.isclose(res[1][0, 0], 1.00)

    # lognormal
    log_sample_vec = np.log(sample_vec)
    res = uq.fit_distribution_to_sample(log_sample_vec, 'lognormal')
    assert np.isclose(np.log(res[0][0, 0]), np.mean(log_sample_vec))
    assert np.isclose(res[0][0, 1], np.std(log_sample_vec))
    assert np.isclose(res[1][0, 0], 1.00)

    # censored data, lower and upper
    np.random.seed(40)
    c_lower = 1.00 - 2.00 * 0.20
    c_upper = 1.00 + 2.00 * 0.20
    sample_vec = np.array(
        (
            1.19001858,
            0.94546098,
            1.17789766,
            1.20168158,
            0.91329968,
            0.92214045,
            0.83480078,
            0.75774220,
            1.12245935,
            1.11947970,
            0.84877398,
            0.98338148,
            0.68880282,
            1.20237202,
            0.94543761,
            1.26858046,
            1.14934510,
            1.21250879,
            0.89558603,
            0.90804330,
        )
    )
    usable_sample_idx = np.all([sample_vec > c_lower, sample_vec < c_upper], axis=0)
    usable_sample = sample_vec[usable_sample_idx].reshape((1, -1))
    c_count = len(sample_vec) - len(usable_sample)
    usable_sample = usable_sample.reshape((1, -1))
    res_a = uq.fit_distribution_to_sample(
        usable_sample,
        'normal_cov',
        censored_count=c_count,
        detection_limits=(c_lower, c_upper),
    )
    compare_a = (
        np.array((1.13825975, 0.46686491)),
        np.array(
            ((1.00,)),
        ),
    )
    assert np.allclose(res_a[0], compare_a[0])
    assert np.allclose(res_a[1], compare_a[1])

    res_a = uq.fit_distribution_to_sample(
        usable_sample,
        'normal_std',
        censored_count=c_count,
        detection_limits=(c_lower, c_upper),
    )
    compare_a = (
        np.array((1.13825975, 0.53141375)),
        np.array(
            ((1.00,)),
        ),
    )
    assert np.allclose(res_a[0], compare_a[0])
    assert np.allclose(res_a[1], compare_a[1])

    # censored data, only lower
    np.random.seed(40)
    c_lower = -1.50
    c_upper = np.inf
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00))
    usable_sample_idx = np.all([sample_vec > c_lower, sample_vec < c_upper], axis=0)
    usable_sample = sample_vec[usable_sample_idx].reshape((1, -1))
    c_count = len(sample_vec) - len(usable_sample)
    usable_sample = usable_sample.reshape((1, -1))
    res_b = uq.fit_distribution_to_sample(
        usable_sample,
        'normal_cov',
        censored_count=c_count,
        detection_limits=(c_lower, c_upper),
    )
    compare_b = (
        np.array((-1.68598848, 1.75096914)),
        np.array(
            ((1.00,)),
        ),
    )
    assert np.allclose(res_b[0], compare_b[0])
    assert np.allclose(res_b[1], compare_b[1])

    # censored data, only upper
    np.random.seed(40)
    c_lower = -np.inf
    c_upper = 1.50
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00))
    usable_sample_idx = np.all([sample_vec > c_lower, sample_vec < c_upper], axis=0)
    usable_sample = sample_vec[usable_sample_idx].reshape((1, -1))
    c_count = len(sample_vec) - len(usable_sample)
    usable_sample = usable_sample.reshape((1, -1))
    res_c = uq.fit_distribution_to_sample(
        usable_sample,
        'normal_cov',
        censored_count=c_count,
        detection_limits=(c_lower, c_upper),
    )
    compare_c = (
        np.array((1.68598845, 1.75096921)),
        np.array(
            ((1.00,)),
        ),
    )
    assert np.allclose(res_c[0], compare_c[0])
    assert np.allclose(res_c[1], compare_c[1])

    # symmetry check
    assert np.isclose(res_b[0][0, 0], -res_c[0][0, 0])
    assert np.isclose(res_b[0][0, 1], res_c[0][0, 1])

    # truncated data, lower and upper, expect failure
    t_lower = -1.50
    t_upper = 1.50
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape(
        (1, -1)
    )
    with pytest.raises(
        ValueError,
        match='One or more sample values lie outside of the specified truncation limits.',
    ):
        res = uq.fit_distribution_to_sample(
            sample_vec, 'normal_cov', truncation_limits=(t_lower, t_upper)
        )

    # truncated data, only lower, expect failure
    t_lower = -1.50
    t_upper = np.inf
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape(
        (1, -1)
    )
    with pytest.raises(
        ValueError,
        match=(
            'One or more sample values lie '
            'outside of the specified truncation limits.'
        ),
    ):
        res = uq.fit_distribution_to_sample(
            sample_vec, 'normal_cov', truncation_limits=(t_lower, t_upper)
        )

    # truncated data, only upper, expect failure
    t_lower = -np.inf
    t_upper = 1.50
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape(
        (1, -1)
    )
    with pytest.raises(
        ValueError,
        match=(
            'One or more sample values lie '
            'outside of the specified truncation limits.'
        ),
    ):
        res = uq.fit_distribution_to_sample(
            sample_vec, 'normal_cov', truncation_limits=(t_lower, t_upper)
        )

    # truncated data, lower and upper
    np.random.seed(40)
    t_lower = -0.50
    t_upper = +4.50
    sample_vec = np.array((0.00, 1.00, 2.00, 3.00, 4.00)).reshape((1, -1))
    res_a = uq.fit_distribution_to_sample(
        sample_vec, 'normal_cov', truncation_limits=(t_lower, t_upper)
    )
    compare_a = (
        np.array((1.99999973, 2.2639968)),
        np.array(
            ((1.00,)),
        ),
    )
    assert np.allclose(res_a[0], compare_a[0])
    assert np.allclose(res_a[1], compare_a[1])

    # truncated data, only lower
    np.random.seed(40)
    t_lower = -4.50
    t_upper = np.inf
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape(
        (1, -1)
    )
    res_b = uq.fit_distribution_to_sample(
        sample_vec, 'normal_cov', truncation_limits=(t_lower, t_upper)
    )
    compare_b = (np.array((-0.09587816, 21.95601487)), np.array((1.00,)))
    assert np.allclose(res_b[0], compare_b[0])
    assert np.allclose(res_b[1], compare_b[1])

    # truncated data, only upper
    np.random.seed(40)
    t_lower = -np.inf
    t_upper = 4.50
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00)).reshape(
        (1, -1)
    )
    res_c = uq.fit_distribution_to_sample(
        sample_vec, 'normal_cov', truncation_limits=(t_lower, t_upper)
    )
    compare_c = (
        np.array((0.09587811, 21.95602574)),
        np.array(
            ((1.00,)),
        ),
    )
    assert np.allclose(res_c[0], compare_c[0])
    assert np.allclose(res_c[1], compare_c[1])

    # symmetry check
    assert np.isclose(res_b[0][0, 0], -res_c[0][0, 0])
    assert np.isclose(res_b[0][0, 1], res_c[0][0, 1])


def test_fit_distribution_to_sample_multivariate() -> None:
    # uncorrelated, normal
    np.random.seed(40)
    sample = np.random.multivariate_normal(
        (1.00, 1.00), np.array(((1.00, 0.00), (0.00, 1.00))), size=10000
    ).T
    np.random.seed(40)
    # note: distribution can be specified once, implying that it is
    # the same for all random variables.
    res = uq.fit_distribution_to_sample(sample, ['normal_cov'])
    compare = (
        np.array(((0.9909858, 1.01732669), (0.99994493, 0.99588164))),
        np.array(((1.00, 0.0092258), (0.0092258, 1.00))),
    )
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # correlated, normal
    np.random.seed(40)
    sample = np.random.multivariate_normal(
        (1.00, 1.00), np.array(((1.00, 0.70), (0.70, 1.00))), size=10000
    ).T
    np.random.seed(40)
    res = uq.fit_distribution_to_sample(sample, ['normal_cov', 'normal_cov'])
    compare = (
        np.array(((1.00833201, 1.0012552), (1.00828936, 0.99477853))),
        np.array(((1.00, 0.70623679), (0.70623679, 1.00))),
    )
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # correlated, normal, truncated and with detection limits
    np.random.seed(40)
    sample = np.random.multivariate_normal(
        (1.00, 1.00), np.array(((1.00, 0.70), (0.70, 1.00))), size=10000
    ).T
    np.random.seed(40)
    res = uq.fit_distribution_to_sample(
        sample,
        ['normal_cov', 'normal_cov'],
        truncation_limits=(-5.00, 6.00),
        detection_limits=(0.20, 1.80),
    )
    compare = (
        np.array(((1.00833201, 1.0012552), (1.00828936, 0.99477853))),
        np.array(((1.00, 0.70624434), (0.70624434, 1.00))),
    )
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # samples that contain duplicate rows
    np.random.seed(40)
    sample = np.full(
        (2, 10),
        123.00,
    )
    np.random.seed(40)
    res = uq.fit_distribution_to_sample(sample, ['normal_cov', 'normal_cov'])
    compare = (
        np.array(((123.00, 1.0e-6), (123.00, 1.0e-6))),
        np.array(((1.00, 0.00), (0.00, 1.00))),
    )
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # lognormal with detection limits
    np.random.seed(40)
    sample = np.log(
        np.random.multivariate_normal(
            (100.00, 100.00), np.array(((1e-2, 1e-8), (1e-8, 1e-2))), size=10000
        ).T
    )
    np.random.seed(40)
    res = uq.fit_distribution_to_sample(
        sample, ['lognormal', 'lognormal'], detection_limits=(1e-8, 5.00)
    )
    compare = (
        np.array(((4.60517598e00, 2.18581908e-04), (4.60517592e00, 2.16575944e-04))),
        np.array(((1.00, 0.01229486), (0.01229486, 1.00))),
    )
    assert np.allclose(res[0], compare[0])
    assert np.allclose(res[1], compare[1])

    # only a single row: requires different syntax (see univariate)
    # and thus fails.
    np.random.seed(40)
    sample = np.full(
        (1, 10),
        math.pi,
    )
    np.random.seed(40)
    with pytest.raises(IndexError):
        res = uq.fit_distribution_to_sample(sample, ['normal_cov', 'normal_cov'])

    # extreme examples:
    # for these we just ensure that the function works without
    # producing any error messages.

    # 1) noisy input data, normal fit
    sample = np.random.multivariate_normal(
        (np.log(20.00), np.log(20.00)),
        np.array(((0.10, 0.05), (0.05, 0.10))),
        size=10000,
    ).T
    sample = np.exp(sample)
    sample += np.random.uniform(-10.00, 10.00, size=sample.shape)
    res = uq.fit_distribution_to_sample(sample, ['normal_cov', 'normal_cov'])
    for res_i in res:
        assert not np.any(np.isinf(res_i))
        assert not np.any(np.isnan(res_i))

    # 2) very noisy input data, normal fit
    sample = np.random.uniform(-10.00, 10.00, size=sample.shape)
    res = uq.fit_distribution_to_sample(sample, ['normal_cov', 'normal_cov'])
    for res_i in res:
        assert not np.any(np.isinf(res_i))
        assert not np.any(np.isnan(res_i))

    # 3) very noisy input data, lognormal fit
    sample = np.random.uniform(10.00, 100.00, size=sample.shape)
    res = uq.fit_distribution_to_sample(sample, ['lognormal', 'lognormal'])
    for res_i in res:
        assert not np.any(np.isinf(res_i))
        assert not np.any(np.isnan(res_i))

    # 4) data that deviate substantially from normal
    # https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    # Thanks @ Boris Verkhovskiy & Mike
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sample = np.concatenate(
            (np.random.normal(0.00, 1.00, size=100000), np.array((np.inf,)))
        )
        with pytest.raises(
            ValueError, match='Conversion to standard normal space was unsuccessful'
        ):
            uq.fit_distribution_to_sample(sample, ['normal_cov'])


def test_fit_distribution_to_percentiles() -> None:
    # normal, mean of 20 and standard deviation of 10
    percentiles = np.linspace(0.01, 0.99, num=10000).tolist()
    values = norm.ppf(percentiles, loc=20, scale=10).tolist()
    res = uq.fit_distribution_to_percentiles(
        values, percentiles, ['normal', 'lognormal']
    )
    assert res[0] == 'normal'
    assert np.allclose(res[1], np.array((20.00, 10.00)))

    # lognormal, median of 20 and beta of 0.4
    ln_values = lognorm.ppf(percentiles, s=0.40, scale=20.00).tolist()
    res = uq.fit_distribution_to_percentiles(
        ln_values, percentiles, ['normal', 'lognormal']
    )
    assert res[0] == 'lognormal'
    assert np.allclose(res[1], np.array((20.0, 0.40)))

    # unrecognized distribution family
    percentiles = np.linspace(0.01, 0.99, num=10000).tolist()
    values = norm.ppf(percentiles, loc=20, scale=10).tolist()
    with pytest.raises(
        ValueError, match='Distribution family not recognized: birnbaum-saunders'
    ):
        uq.fit_distribution_to_percentiles(
            values, percentiles, ['lognormal', 'birnbaum-saunders']
        )


def test__OLS_percentiles() -> None:
    # normal: negative standard deviation
    params = (2.50, -0.10)
    perc = np.linspace(1e-2, 1.00 - 1e-2, num=5)
    values = norm.ppf(perc, loc=20, scale=10)
    family = 'normal'
    res = uq._OLS_percentiles(params, values, perc, family)
    assert res == 10000000000.0

    # lognormal: negative median
    params = (-1.00, 0.40)
    perc = np.linspace(1e-2, 1.00 - 1e-2, num=5)
    values = lognorm.ppf(perc, s=0.40, scale=20.00)
    family = 'lognormal'
    res = uq._OLS_percentiles(params, values, perc, family)
    assert res == 10000000000.0


#  __  __      _   _               _
# |  \/  | ___| |_| |__   ___   __| |___
# | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
# | |  | |  __/ |_| | | | (_) | (_| \__ \
# |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
#
# The following tests verify the methods of the objects of the module.


def test_NormalRandomVariable() -> None:
    rv = uq.NormalRandomVariable('rv_name', theta=np.array((0.00, 1.00)))
    assert rv.name == 'rv_name'
    assert rv.theta is not None
    np.testing.assert_allclose(rv.theta, np.array((0.00, 1.00)))
    assert np.all(np.isnan(rv.truncation_limits))
    assert rv.RV_set is None
    assert rv.sample_DF is None
    # confirm that creating an attribute on the fly is not allowed
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore


def test_Normal_STD() -> None:
    rv = uq.Normal_STD('rv_name', theta=np.array((0.00, 1.00)))
    assert rv.name == 'rv_name'
    assert rv.theta is not None
    np.testing.assert_allclose(rv.theta, np.array((0.00, 1.00)))
    assert np.all(np.isnan(rv.truncation_limits))
    assert rv.RV_set is None
    assert rv.sample_DF is None
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore


def test_Normal_COV() -> None:
    with pytest.raises(
        ValueError, match='The mean of Normal_COV RVs cannot be zero.'
    ):
        rv = uq.Normal_COV('rv_name', theta=np.array((0.00, 1.00)))
    rv = uq.Normal_COV('rv_name', theta=np.array((2.00, 1.00)))
    assert rv.name == 'rv_name'
    assert rv.theta is not None
    np.testing.assert_allclose(rv.theta, np.array((2.00, 2.00)))
    assert np.all(np.isnan(rv.truncation_limits))
    assert rv.RV_set is None
    assert rv.sample_DF is None
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore


def test_NormalRandomVariable_cdf() -> None:
    # test CDF method
    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 1.0)),
        truncation_limits=np.array((0.00, np.nan)),
    )

    # evaluate CDF at different points
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    assert np.allclose(cdf, (0.0, 0.0, 0.1781461, 0.40571329, 0.81142658), rtol=1e-5)

    # repeat without truncation limits
    rv = uq.NormalRandomVariable('test_rv', theta=np.array((1.0, 1.0)))

    # evaluate CDF at different points
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    assert np.allclose(
        cdf, (0.02275013, 0.15865525, 0.30853754, 0.5, 0.84134475), rtol=1e-5
    )


def test_NormalRandomVariable_variable_theta_cdf() -> None:
    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array(((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))),
        truncation_limits=np.array((0.00, np.nan)),
    )

    # evaluate CDF at different points
    x = np.array((-1.0, 0.5, 2.0))
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    expected_cdf = (0.0, 0.1781461, 0.48836013)

    assert np.allclose(cdf, expected_cdf, rtol=1e-5)

    # repeat without truncation limits
    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array(((0.0, 1.0), (1.0, 1.0), (2.0, 1.0))),
    )

    x = np.array((-1.0, 0.5, 2.0))
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    expected_cdf = (0.15865525, 0.30853754, 0.5)
    assert np.allclose(cdf, expected_cdf, rtol=1e-5)


def test_Normal_STD_cdf() -> None:
    rv = uq.Normal_STD(
        'test_rv',
        theta=np.array((1.0, 1.0)),
        truncation_limits=np.array((0.00, np.nan)),
    )

    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(cdf, (0.0, 0.0, 0.1781461, 0.40571329, 0.81142658), rtol=1e-5)


def test_Normal_COV_cdf() -> None:
    rv = uq.Normal_COV(
        'test_rv',
        theta=np.array((1.0, 1.0)),
        truncation_limits=np.array((0.00, np.nan)),
    )

    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(cdf, (0.0, 0.0, 0.1781461, 0.40571329, 0.81142658), rtol=1e-5)


def test_NormalRandomVariable_inverse_transform() -> None:
    samples = np.array((0.10, 0.20, 0.30))

    rv = uq.NormalRandomVariable('test_rv', theta=np.array((1.0, 0.5)))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = rv.sample
    assert inverse_transform is not None
    assert np.allclose(
        inverse_transform, np.array((0.35922422, 0.57918938, 0.73779974)), rtol=1e-5
    )

    rv = uq.NormalRandomVariable('test_rv', theta=np.array((1.0, 0.5)))
    with pytest.raises(ValueError, match='No available uniform sample.'):
        rv.inverse_transform_sampling()

    # with truncation limits

    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 0.5)),
        truncation_limits=np.array((np.nan, 1.20)),
    )
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.24508018, 0.43936, 0.57313359)), rtol=1e-5
    )

    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 0.5)),
        truncation_limits=np.array((0.80, np.nan)),
    )
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.8863824, 0.96947866, 1.0517347)), rtol=1e-5
    )

    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 0.5)),
        truncation_limits=np.array((0.80, 1.20)),
    )
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.84155378, 0.88203946, 0.92176503)), rtol=1e-5
    )

    #
    # edge cases
    #

    # normal with problematic truncation limits
    rv = uq.NormalRandomVariable(
        'test_rv', theta=np.array((1.0, 0.5)), truncation_limits=np.array((1e8, 2e8))
    )
    rv.uni_sample = samples
    with pytest.raises(
        ValueError,
        match=(
            'The probability mass within the truncation '
            'limits is too small and the truncated '
            'distribution cannot be sampled with '
            'sufficiently high accuracy. This is most probably '
            'due to incorrect truncation limits set '
            'for the distribution.'
        ),
    ):
        rv.inverse_transform_sampling()


def test_NormalRandomVariable_variable_theta_inverse_transform() -> None:
    rv = uq.NormalRandomVariable(
        'test_rv',
        theta=np.array(
            (
                (0.0, 1.0),
                (1.0, 1.0),
                (2.0, 1.0),
                (1.0, 1.0),
                (1.0, 1.0),
                (1.0, 1.0),
                (1.0, 1.0),
            )
        ),
    )
    rv.uni_sample = np.array((0.5, 0.5, 0.5, 0.0, 1.0, 0.20, 0.80))
    rv.inverse_transform_sampling()
    inverse_transform = rv.sample
    expected_result = np.array(
        (0.0, 1.0, 2.0, -np.inf, np.inf, 0.15837877, 1.84162123)
    )
    assert inverse_transform is not None
    assert np.allclose(inverse_transform, expected_result, rtol=1e-5)


def test_Normal_STD_inverse_transform() -> None:
    samples = np.array((0.10, 0.20, 0.30))
    rv = uq.Normal_STD('test_rv', theta=np.array((1.0, 0.5)))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.35922422, 0.57918938, 0.73779974)), rtol=1e-5
    )


def test_Normal_COV_inverse_transform() -> None:
    samples = np.array((0.10, 0.20, 0.30))
    rv = uq.Normal_COV('test_rv', theta=np.array((1.0, 0.5)))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.35922422, 0.57918938, 0.73779974)), rtol=1e-5
    )


def test_LogNormalRandomVariable_cdf() -> None:
    # lower truncation
    rv = uq.LogNormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 1.0)),
        truncation_limits=np.array((0.10, np.nan)),
    )
    # confirm that creating an attribute on the fly is not allowed
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(
        cdf, (0.0, 0.0, 0.23597085, 0.49461712, 0.75326339), rtol=1e-5
    )

    # upper truncation
    rv = uq.LogNormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 1.0)),
        truncation_limits=np.array((np.nan, 5.00)),
    )
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(
        cdf, (0.00, 0.00, 0.25797755, 0.52840734, 0.79883714), rtol=1e-5
    )

    # no truncation
    rv = uq.LogNormalRandomVariable('test_rv', theta=np.array((1.0, 1.0)))
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(cdf, (0.0, 0.0, 0.2441086, 0.5, 0.7558914), rtol=1e-5)


def test_LogNormalRandomVariable_variable_theta_cdf() -> None:
    rv = uq.LogNormalRandomVariable(
        'test_rv',
        theta=np.array(((0.1, 1.0), (1.0, 1.0), (2.0, 1.0))),
        truncation_limits=np.array((0.20, 3.0)),
    )

    # evaluate CDF at different points
    x = np.array((0.01, 0.5, 2.0))
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    expected_cdf = (0.0, 0.23491926, 0.75659125)

    assert np.allclose(cdf, expected_cdf, rtol=1e-5)

    # repeat without truncation limits
    rv = uq.LogNormalRandomVariable(
        'test_rv',
        theta=np.array(((0.01, 1.0), (1.0, 1.0), (2.0, 1.0))),
    )

    x = np.array((-1.0, 0.5, 2.0))
    cdf = rv.cdf(x)

    # assert that CDF values are correct
    expected_cdf = (0.0, 0.2441086, 0.5)
    assert np.allclose(cdf, expected_cdf, rtol=1e-5)


def test_LogNormalRandomVariable_inverse_transform() -> None:
    samples = np.array((0.10, 0.20, 0.30))
    rv = uq.LogNormalRandomVariable('test_rv', theta=np.array((1.0, 0.5)))

    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)

    assert np.allclose(
        inverse_transform, np.array((0.52688352, 0.65651442, 0.76935694)), rtol=1e-5
    )

    #
    # lognormal with truncation limits
    #

    rv = uq.LogNormalRandomVariable(
        'test_rv',
        theta=np.array((1.0, 0.5)),
        truncation_limits=np.array((0.50, np.nan)),
    )
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.62614292, 0.73192471, 0.83365823)), rtol=1e-5
    )

    #
    # edge cases
    #

    # lognormal without values to sample from
    rv = uq.LogNormalRandomVariable('test_rv', theta=np.array((1.0, 0.5)))
    with pytest.raises(ValueError, match='No available uniform sample.'):
        rv.inverse_transform_sampling()


def test_LogNormalRandomVariable_variable_theta_inverse_transform() -> None:
    rv = uq.LogNormalRandomVariable(
        'test_rv',
        theta=np.array(
            (
                (0.10, 1.0),
                (1.0, 1.0),
                (2.0, 1.0),
                (1.0, 1.0),
                (1.0, 1.0),
                (1.0, 1.0),
                (1.0, 1.0),
            )
        ),
    )
    rv.uni_sample = np.array((0.5, 0.5, 0.5, 0.0, 1.0, 0.20, 0.80))
    rv.inverse_transform_sampling()
    inverse_transform = rv.sample
    expected_result = np.array((0.1, 1.0, 2.0, 0.0, np.inf, 0.43101119, 2.32012539))
    assert inverse_transform is not None
    assert np.allclose(inverse_transform, expected_result, rtol=1e-5)


def test_UniformRandomVariable_cdf() -> None:
    # uniform, both theta values
    rv = uq.UniformRandomVariable('test_rv', theta=np.array((0.0, 1.0)))
    # confirm that creating an attribute on the fly is not allowed
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(cdf, (0.0, 0.0, 0.5, 1.0, 1.0), rtol=1e-5)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # uniform, only upper theta value ( -inf implied )
        rv = uq.UniformRandomVariable('test_rv', theta=np.array((np.nan, 100.00)))
        x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
        cdf = rv.cdf(x)
        assert np.all(np.isnan(cdf))

    # uniform, only lower theta value ( +inf implied )
    rv = uq.UniformRandomVariable('test_rv', theta=np.array((0.00, np.nan)))
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(cdf, (0.0, 0.0, 0.0, 0.0, 0.0), rtol=1e-5)

    # uniform, with truncation limits
    rv = uq.UniformRandomVariable(
        'test_rv',
        theta=np.array((0.0, 10.0)),
        truncation_limits=np.array((0.00, 1.00)),
    )
    x = np.array((-1.0, 0.0, 0.5, 1.0, 2.0))
    cdf = rv.cdf(x)
    assert np.allclose(cdf, (0.0, 0.0, 0.5, 1.0, 1.0), rtol=1e-5)


def test_UniformRandomVariable_inverse_transform() -> None:
    rv = uq.UniformRandomVariable('test_rv', theta=np.array((0.0, 1.0)))
    samples = np.array((0.10, 0.20, 0.30))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(inverse_transform, samples, rtol=1e-5)

    #
    # uniform with unspecified bounds
    #

    rv = uq.UniformRandomVariable('test_rv', theta=np.array((np.nan, 1.0)))
    samples = np.array((0.10, 0.20, 0.30))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.all(np.isnan(inverse_transform))

    rv = uq.UniformRandomVariable('test_rv', theta=np.array((0.00, np.nan)))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.all(np.isinf(inverse_transform))

    rv = uq.UniformRandomVariable(
        'test_rv',
        theta=np.array((0.00, 1.00)),
        truncation_limits=np.array((0.20, 0.80)),
    )
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(inverse_transform, np.array((0.26, 0.32, 0.38)), rtol=1e-5)

    # sample as a pandas series, with a log() map
    rv.f_map = np.log
    assert ensure_value(rv.sample_DF).to_dict() == {
        0: -1.3470736479666092,
        1: -1.1394342831883646,
        2: -0.9675840262617056,
    }

    #
    # edge cases
    #

    # uniform without values to sample from
    rv = uq.UniformRandomVariable('test_rv', theta=np.array((0.0, 1.0)))
    with pytest.raises(ValueError, match='No available uniform sample.'):
        rv.inverse_transform_sampling()


def test_WeibullRandomVariable() -> None:
    rv = uq.WeibullRandomVariable('rv_name', theta=np.array((1.5, 2.0)))
    assert rv.name == 'rv_name'
    assert rv.theta is not None
    np.testing.assert_allclose(rv.theta, np.array((1.5, 2.0)))
    assert np.all(np.isnan(rv.truncation_limits))
    assert rv.RV_set is None
    assert rv.sample_DF is None
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore


def test_WeibullRandomVariable_cdf() -> None:
    rv = uq.WeibullRandomVariable(
        'test_rv',
        theta=np.array((1.5, 2.0)),
        truncation_limits=np.array((0.5, 2.5)),
    )

    x = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    cdf = rv.cdf(x)

    expected_cdf = np.array([0.0, 0.0, 0.30463584, 0.63286108, 0.87169261, 1.0, 1.0])
    assert np.allclose(cdf, expected_cdf, rtol=1e-5)

    rv = uq.WeibullRandomVariable('test_rv', theta=np.array((1.5, 2.0)))
    cdf = rv.cdf(x)
    expected_cdf_no_trunc = weibull_min.cdf(x, 2.0, scale=1.5)
    assert np.allclose(cdf, expected_cdf_no_trunc, rtol=1e-5)


def test_WeibullRandomVariable_inverse_transform() -> None:
    samples = np.array((0.10, 0.20, 0.30))

    rv = uq.WeibullRandomVariable('test_rv', theta=np.array((1.5, 2.0)))
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    expected_samples = weibull_min.ppf(samples, 2.0, scale=1.5)
    assert np.allclose(inverse_transform, expected_samples, rtol=1e-5)

    rv = uq.WeibullRandomVariable(
        'test_rv', theta=np.array((1.5, 2.0)), truncation_limits=np.array((0.5, 2.5))
    )
    rv.uni_sample = samples
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    truncated_samples = weibull_min.ppf(
        samples
        * (
            weibull_min.cdf(2.5, 2.0, scale=1.5)
            - weibull_min.cdf(0.5, 2.0, scale=1.5)
        )
        + weibull_min.cdf(0.5, 2.0, scale=1.5),
        2.0,
        scale=1.5,
    )
    assert np.allclose(inverse_transform, truncated_samples, rtol=1e-5)


def test_MultinomialRandomVariable() -> None:
    # multinomial with invalid p values provided in the theta vector
    with pytest.raises(
        ValueError,
        match=re.escape(
            'The set of p values provided for a multinomial '
            'distribution shall sum up to less than or equal to 1.0. '
            'The provided values sum up to 43.0. '
            'p = [ 0.2  0.7  0.1 42. ] .'
        ),
    ):
        uq.MultinomialRandomVariable(
            'rv_invalid', np.array((0.20, 0.70, 0.10, 42.00))
        )


def test_MultilinearCDFRandomVariable() -> None:
    # multilinear CDF: cases that should fail

    x_values = (0.00, 1.00, 2.00, 3.00, 4.00)
    y_values = (100.00, 0.20, 0.20, 0.80, 1.00)
    values = np.column_stack((x_values, y_values))
    with pytest.raises(
        ValueError,
        match='For multilinear CDF random variables, y_1 should be set to 0.00',
    ):
        uq.MultilinearCDFRandomVariable('test_rv', theta=values)

    x_values = (0.00, 1.00, 2.00, 3.00, 4.00)
    y_values = (0.00, 0.20, 0.20, 0.80, 0.80)
    values = np.column_stack((x_values, y_values))
    with pytest.raises(
        ValueError,
        match='For multilinear CDF random variables, y_n should be set to 1.00',
    ):
        uq.MultilinearCDFRandomVariable('test_rv', theta=values)

    x_values = (0.00, 3.00, 1.00, 2.00, 4.00)
    y_values = (0.00, 0.25, 0.50, 0.75, 1.00)
    values = np.column_stack((x_values, y_values))
    with pytest.raises(
        ValueError,
        match='For multilinear CDF random variables, Xs should be specified in ascending order',
    ):
        uq.MultilinearCDFRandomVariable('test_rv', theta=values)

    x_values = (0.00, 1.00, 2.00, 3.00, 4.00)
    y_values = (0.00, 0.75, 0.50, 0.25, 1.00)
    values = np.column_stack((x_values, y_values))
    with pytest.raises(
        ValueError,
        match='For multilinear CDF random variables, Ys should be specified in ascending order',
    ):
        uq.MultilinearCDFRandomVariable('test_rv', theta=values)

    x_values = (0.00, 1.00, 2.00, 3.00, 4.00)
    y_values = (0.00, 0.50, 0.50, 0.50, 1.00)
    values = np.column_stack((x_values, y_values))
    with pytest.raises(
        ValueError,
        match=(
            'For multilinear CDF random variables, '
            'Ys should be specified in strictly ascending order'
        ),
    ):
        uq.MultilinearCDFRandomVariable('test_rv', theta=values)

    x_values = (0.00, 2.00, 2.00, 3.00, 4.00)
    y_values = (0.00, 0.20, 0.40, 0.50, 1.00)
    values = np.column_stack((x_values, y_values))
    with pytest.raises(
        ValueError,
        match=(
            'For multilinear CDF random variables, '
            'Xs should be specified in strictly ascending order'
        ),
    ):
        uq.MultilinearCDFRandomVariable('test_rv', theta=values)


def test_MultilinearCDFRandomVariable_cdf() -> None:
    x_values = (0.00, 1.00, 2.00, 3.00, 4.00)
    y_values = (0.00, 0.20, 0.30, 0.80, 1.00)
    values = np.column_stack((x_values, y_values))
    rv = uq.MultilinearCDFRandomVariable('test_rv', theta=values)
    # confirm that creating an attribute on the fly is not allowed
    with pytest.raises(AttributeError):
        rv.xyz = 123  # type: ignore
    x = np.array(
        (-100.00, 0.00, 0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 100.00)
    )
    cdf = rv.cdf(x)

    assert np.allclose(
        cdf,
        (0.00, 0.00, 0.10, 0.20, 0.25, 0.30, 0.55, 0.80, 0.90, 1.00, 1.0),
        rtol=1e-5,
    )


def test_MultilinearCDFRandomVariable_inverse_transform() -> None:
    x_values = (0.00, 1.00, 2.00, 3.00, 4.00)
    y_values = (0.00, 0.20, 0.30, 0.80, 1.00)
    values = np.column_stack((x_values, y_values))
    rv = uq.MultilinearCDFRandomVariable('test_rv', theta=values)

    rv.uni_sample = np.array((0.00, 0.1, 0.2, 0.5, 0.8, 0.9, 1.00))
    rv.inverse_transform_sampling()
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform,
        np.array((0.00, 0.50, 1.00, 2.40, 3.00, 3.50, 4.00)),
        rtol=1e-5,
    )


def test_EmpiricalRandomVariable_inverse_transform() -> None:
    samples = np.array((0.10, 0.20, 0.30))

    rv_empirical = uq.EmpiricalRandomVariable(
        'test_rv_empirical', theta=np.array((1.00, 2.00, 3.00, 4.00))
    )
    # confirm that creating an attribute on the fly is not allowed
    with pytest.raises(AttributeError):
        rv_empirical.xyz = 123  # type: ignore

    samples = np.array((0.10, 0.50, 0.90))

    rv_empirical.uni_sample = samples
    rv_empirical.inverse_transform_sampling()
    inverse_transform = ensure_value(rv_empirical.sample)

    assert np.allclose(inverse_transform, np.array((1.00, 3.00, 4.00)), rtol=1e-5)

    rv_coupled = uq.CoupledEmpiricalRandomVariable(
        'test_rv_coupled',
        theta=np.array((1.00, 2.00, 3.00, 4.00)),
    )
    rv_coupled.inverse_transform_sampling(sample_size=6)
    inverse_transform = ensure_value(rv_coupled.sample)

    assert np.allclose(
        inverse_transform, np.array((1.00, 2.00, 3.00, 4.00, 1.00, 2.00)), rtol=1e-5
    )


def test_DeterministicRandomVariable_inverse_transform() -> None:
    rv = uq.DeterministicRandomVariable('test_rv', theta=np.array((0.00,)))
    rv.inverse_transform_sampling(4)
    inverse_transform = ensure_value(rv.sample)
    assert np.allclose(
        inverse_transform, np.array((0.00, 0.00, 0.00, 0.00)), rtol=1e-5
    )


def test_RandomVariable_Set() -> None:
    # a set of two random variables
    rv_1 = uq.NormalRandomVariable('rv1', theta=np.array((1.0, 1.0)))
    rv_2 = uq.NormalRandomVariable('rv2', theta=np.array((1.0, 1.0)))
    rv_set = uq.RandomVariableSet(
        'test_set', [rv_1, rv_2], np.array(((1.0, 0.50), (0.50, 1.0)))
    )

    # size of the set
    assert rv_set.size == 2

    # a set with only one random variable
    rv_1 = uq.NormalRandomVariable('rv1', theta=np.array((1.0, 1.0)))
    rv_set = uq.RandomVariableSet('test_set', [rv_1], np.array(((1.0, 0.50),)))


def test_RandomVariable_perfect_correlation() -> None:
    # Test that the `.anchor` attribute is propagated correctly
    rv_1 = uq.NormalRandomVariable('rv1', theta=np.array((1.0, 1.0)))
    rv_2 = uq.NormalRandomVariable('rv2', theta=np.array((1.0, 1.0)), anchor=rv_1)
    rv_1.uni_sample = np.random.random(size=10)
    assert np.all(rv_2.uni_sample == rv_1.uni_sample)

    rv_1 = uq.NormalRandomVariable('rv1', theta=np.array((1.0, 1.0)))
    rv_2 = uq.NormalRandomVariable('rv2', theta=np.array((1.0, 1.0)))
    rv_1.uni_sample = np.random.random(size=10)
    assert rv_2.uni_sample is None


def test_RandomVariable_Set_apply_correlation(*, reset: bool = False) -> None:
    data_dir = (
        'pelicun/tests/basic/data/uq/test_random_variable_set_apply_correlation'
    )

    # correlated, uniform
    np.random.seed(40)
    rv_1 = uq.UniformRandomVariable(name='rv1', theta=np.array((-5.0, 5.0)))
    rv_2 = uq.UniformRandomVariable(name='rv2', theta=np.array((-5.0, 5.0)))

    rv_1.uni_sample = np.random.random(size=100)
    rv_2.uni_sample = np.random.random(size=100)

    rvs = uq.RandomVariableSet(
        name='test_set', rv_list=[rv_1, rv_2], rho=np.array(((1.0, 0.5), (0.5, 1.0)))
    )
    rvs.apply_correlation()

    for i, rv in enumerate((rv_1, rv_2)):
        res = ensure_value(rv.uni_sample)
        file_incr = i + 1
        filename = f'{data_dir}/test_{file_incr}.pcl'
        if reset:
            export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)

    # we also test .sample here

    rv_1.inverse_transform_sampling()
    rv_2.inverse_transform_sampling()
    rvset_sample = rvs.sample
    assert set(rvset_sample.keys()) == {'rv1', 'rv2'}
    vals = list(rvset_sample.values())
    assert np.all(vals[0] == rv_1.sample)
    assert np.all(vals[1] == rv_2.sample)


def test_RandomVariable_Set_apply_correlation_special() -> None:
    # This function tests the apply_correlation method of the
    # RandomVariableSet class when given special input conditions.
    # The first test checks that the method works when given a non
    # positive semidefinite correlation matrix.
    # The second test checks that the method works when given a non full
    # rank matrix.

    # inputs that cause `apply_correlation` to use the SVD

    # note: The inputs passed to this function may not be valid
    # correlation matrices, but they are suitable for causing the svd
    # to be utilized for testing purposes.

    # non positive semidefinite correlation matrix
    rho = np.array(((1.00, 0.50), (0.50, -1.00)))
    rv_1 = uq.NormalRandomVariable('rv1', theta=np.array((5.0, 0.1)))
    rv_2 = uq.NormalRandomVariable('rv2', theta=np.array((5.0, 0.1)))
    rv_1.uni_sample = np.random.random(size=100)
    rv_2.uni_sample = np.random.random(size=100)
    rv_set = uq.RandomVariableSet('rv_set', [rv_1, rv_2], rho)
    rv_set.apply_correlation()

    # non full rank matrix
    rho = np.array(((0.00, 0.00), (0.0, 0.0)))
    rv_1 = uq.NormalRandomVariable('rv1', theta=np.array((5.0, 0.1)))
    rv_2 = uq.NormalRandomVariable('rv2', theta=np.array((5.0, 0.1)))
    rv_1.uni_sample = np.random.random(size=100)
    rv_2.uni_sample = np.random.random(size=100)
    rv_set = uq.RandomVariableSet('rv_set', [rv_1, rv_2], rho)
    rv_set.apply_correlation()
    np.linalg.svd(
        rho,
    )


def test_RandomVariable_Set_orthotope_density(*, reset: bool = False) -> None:
    data_dir = (
        'pelicun/tests/basic/data/uq/test_random_variable_set_orthotope_density'
    )

    # create some random variables
    rv_1 = uq.Normal_COV(
        'rv1', theta=np.array((5.0, 0.1)), truncation_limits=np.array((np.nan, 10.0))
    )
    rv_2 = uq.LogNormalRandomVariable('rv2', theta=np.array((10.0, 0.2)))
    rv_3 = uq.UniformRandomVariable('rv3', theta=np.array((13.0, 17.0)))
    rv_4 = uq.UniformRandomVariable('rv4', theta=np.array((0.0, 1.0)))
    rv_5 = uq.UniformRandomVariable('rv5', theta=np.array((0.0, 1.0)))

    # create a random variable set
    rv_set = uq.RandomVariableSet(
        'rv_set', [rv_1, rv_2, rv_3, rv_4, rv_5], np.identity(5)
    )

    # define test cases
    test_cases = (
        # lower bounds, upper bounds, var_subset
        (
            np.array([4.0, 9.0, 14.0, np.nan]),
            np.array([6.0, 11.0, 16.0, 0.80]),
            ['rv1', 'rv2', 'rv3', 'rv4'],
        ),
        (
            np.array([4.0, 9.0, 14.0, np.nan, 0.20]),
            np.array([6.0, 11.0, 16.0, 0.80, 0.40]),
            None,
        ),
        (
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([6.0, 11.0, 16.0, 0.80, 0.40]),
            None,
        ),
        (
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            None,
        ),
    )

    # loop over test cases
    for i, (lower, upper, var_subset) in enumerate(test_cases):
        # evaluate the density of the orthotope
        res = rv_set.orthotope_density(lower, upper, var_subset=var_subset)
        # check that the density is equal to the expected value
        # construct a filepath for the results
        filename = f'{data_dir}/test_{i + 1}.pcl'
        # overwrite results if needed
        if reset:
            export_pickle(filename, res)
        # retrieve expected results
        compare = import_pickle(filename)
        # verify equality
        assert np.allclose(res, compare)


def test_RandomVariableRegistry_generate_sample(*, reset: bool = False) -> None:
    data_dir = (
        'pelicun/tests/basic/data/uq/test_RandomVariableRegistry_generate_sample'
    )
    file_incr = 0

    for method in ('LHS_midpoint', 'LHS', 'MonteCarlo'):
        #
        # Random variable registry with a single random variable
        #

        # create the registry
        rng = np.random.default_rng(0)
        rv_registry_single = uq.RandomVariableRegistry(rng)
        # create the random variable and add it to the registry
        rv = uq.NormalRandomVariable('x', theta=np.array((1.0, 1.0)))
        rv_registry_single.add_RV(rv)

        # Generate a sample
        sample_size = 1000
        rv_registry_single.generate_sample(sample_size, method)

        res = ensure_value(rv_registry_single.RV_sample['x'])
        assert len(res) == sample_size

        file_incr += 1
        filename = f'{data_dir}/test_{file_incr}.pcl'
        if reset:
            export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)

        # unfortunately, tests of stochastic outputs like like these fail
        # on rare occasions.
        # assert np.isclose(np.mean(res), 1.0, atol=1e-2)
        # assert np.isclose(np.std(res), 1.0, atol=1e-2)

        #
        # Random variable registry with multiple random variables
        #

        # create a random variable registry and add some random variables to it
        rng = np.random.default_rng(4)
        rv_registry = uq.RandomVariableRegistry(rng)
        rv_1 = uq.Normal_COV('rv1', theta=np.array((5.0, 0.1)))
        rv_2 = uq.LogNormalRandomVariable('rv2', theta=np.array((10.0, 0.2)))
        rv_3 = uq.UniformRandomVariable('rv3', theta=np.array((13.0, 17.0)))
        rv_registry.add_RV(rv_1)
        rv_registry.add_RV(rv_2)
        rv_registry.add_RV(rv_3)
        with pytest.raises(
            ValueError, match='RV rv3 already exists in the registry.'
        ):
            rv_registry.add_RV(rv_3)

        # create a random variable set and add it to the registry
        rv_set = uq.RandomVariableSet(
            'rv_set', [rv_1, rv_2, rv_3], np.identity(3) + np.full((3, 3), 0.20)
        )
        rv_registry.add_RV_set(rv_set)

        # add some more random variables that are not part of the set
        rv_4 = uq.Normal_COV('rv4', theta=np.array((14.0, 0.30)))
        rv_5 = uq.Normal_COV('rv5', theta=np.array((15.0, 0.50)))
        rv_registry.add_RV(rv_4)
        rv_registry.add_RV(rv_5)

        rv_registry.generate_sample(10, method=method)

        # verify that all samples have been generated as expected
        for rv_name in (f'rv{i + 1}' for i in range(5)):
            res = ensure_value(rv_registry.RV_sample[rv_name])
            file_incr += 1
            filename = f'{data_dir}/test_{file_incr}.pcl'
            if reset:
                export_pickle(filename, res)
            compare = import_pickle(filename)
            assert np.allclose(res, compare)

        # obtain multiple RVs from the registry
        rv_dictionary = rv_registry.RVs(['rv1', 'rv2'])
        assert 'rv1' in rv_dictionary
        assert 'rv2' in rv_dictionary
        assert 'rv3' not in rv_dictionary


def test_rv_class_map() -> None:
    rv_class = uq.rv_class_map('normal_std')
    assert rv_class.__name__ == 'Normal_STD'

    with pytest.raises(
        ValueError, match=re.escape('Unsupported distribution: <unsupported>')
    ):
        uq.rv_class_map('<unsupported>')


if __name__ == '__main__':
    pass
