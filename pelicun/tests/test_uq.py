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
import pickle
import itertools
import os
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
from pelicun import uq

RNG = np.random.default_rng(40)


def export_pickle(filepath, obj, makedirs=True):
    """
    Auxiliary function to export a pickle object.
    Parameters
    ----------
    filepath: str
      The path of the file to be exported,
      including any subdirectories.
    obj: object
      The object to be pickled
    makedirs: bool
      If True, then the directories preceding the filename
      will be created if they do not exist.
    """
    dirname = os.path.dirname(filepath)
    if makedirs:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def import_pickle(filepath):
    """
    Auxiliary function to import a pickle object.
    Parameters
    ----------
    filepath: str
      The path of the file to be imported.

    Returns
    -------
    The pickled object.

    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def test_scale_distribution(reset=False):
    data_dir = 'tests/data/uq/test_scale_distribution'
    args_iter = itertools.product(
        (2.00,),
        ('normal', 'lognormal', 'uniform'),
        (np.array((-1.00, 1.00)),),
        (np.array((-2.00, 2.00)),)
    )
    args_list = list(args_iter)
    args_list.append(
        (2.00, 'uniform', np.array((-1.00, 1.00)), np.array((-2.00, 2.00)))
    )
    for file_incr, arg in enumerate(args_list):
        factor, distr, theta, trunc = arg
        res = uq.scale_distribution(factor, distr, theta, trunc)
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res[0], compare[0])
        assert np.allclose(res[1], compare[1])


def test_mvn_orthotope_density(reset=False):
    data_dir = 'tests/data/uq/test_mvn_orthotope_density'
    mu_vals = (
        0.00,
        0.00,
        0.00,
        np.array((0.00, 0.00)),
        np.array((0.00, 0.00)),
    )
    cov_vals = (
        1.00,
        1.00,
        1.00,
        np.array(
            ((1.00, 0.00),
             (0.00, 1.00))
        ),
        np.array(
            ((1.00, 0.50),
             (0.50, 1.00))
        ),
    )
    lower_vals = (
        -1.00,
        np.nan,
        +0.00,
        np.array((0.00, 0.00)),
        np.array((0.00, 0.00))
    )
    upper_vals = (
        -1.00,
        +0.00,
        np.nan,
        np.array((np.nan, np.nan)),
        np.array((np.nan, np.nan))
    )
    file_incr = 0
    for args in zip(
        mu_vals, cov_vals, lower_vals, upper_vals):
        file_incr += 1
        res = uq.mvn_orthotope_density(*args)
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res[0], compare[0])
        assert np.allclose(res[1], compare[1])


def test__get_theta(reset=False):
    data_dir = 'tests/data/uq/test__get_theta'
    res = uq._get_theta(
        np.array(
            (
                (1.00, 1.00),
                (1.00, 0.5)
            )
        ),
        np.array(
            (
                (0.00, 1.00),
                (1.00, 0.5)
            )
        ),
        ['normal', 'lognormal']
    )
    filename = f'{data_dir}/test_1.pcl'
    if reset: export_pickle(filename, res)
    compare = import_pickle(filename)
    assert np.allclose(res, compare)
    with pytest.raises(ValueError):
        uq._get_theta(
            np.array((1.00,)), np.array((1.00,)),
            'not_a_distribution')


def test__get_limit_probs(reset=False):
    data_dir = 'tests/data/uq/test__get_limit_probs'
    args_iter = itertools.product(
        (
            np.array((0.10, 0.20)),
            np.array((np.nan, 0.20)),
            np.array((0.10, np.nan)),
            np.array((np.nan, np.nan))
        ),
        ('normal', 'lognormal'),
        (np.array((0.15, 1.0)),)
    )
    for file_incr, args in enumerate(args_iter):
        res = uq._get_limit_probs(*args)
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res[0], compare[0])
        assert np.allclose(res[1], compare[1])
    with pytest.raises(ValueError):
        uq._get_limit_probs(
            np.array((1.00,)),
            'not_a_distribution',
            np.array((1.00,)),
        )


def test__get_std_samples(reset=False):
    data_dir = 'tests/data/uq/test__get_std_samples'
    samples_list = [
        np.array((
            (1.00, 2.00, 3.00),
        )),
        np.array((
            (0.657965, 1.128253, 1.044239, 1.599209),
            (1.396495, 1.435923, 2.055659, 1.416298),
            (1.948161, 1.576571, 1.469571, 1.190853)
        )),
    ]
    theta_list = [
        np.array((
            (0.00, 1.0),
        )),
        np.array((
            (1.00, 0.20),
            (1.50, 0.6),
            (1.30, 2.0),
        )),
    ]
    tr_limits_list = [
        np.array((
            (np.nan, np.nan),
        )),
        np.array((
            (np.nan, np.nan),
            (1.10, np.nan),
            (np.nan, 2.80),
        ))
    ]
    dist_list_list = [
        np.array(('normal',)),
        np.array(('normal', 'lognormal', 'normal')),
    ]
    for file_incr, args in enumerate(zip(
            samples_list, theta_list, tr_limits_list, dist_list_list
    )):
        res = uq._get_std_samples(*args)
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)
    with pytest.raises(ValueError):
        uq._get_std_samples(
            np.array((
                (1.00, 2.00, 3.00),
            )),
            np.array((
                (0.00, 1.0),
            )),
            np.array((
                (np.nan, np.nan),
            )),
            np.array(('some_unsupported_distribution',)),
        )


def test__get_std_corr_matrix(reset=False):
    data_dir = 'tests/data/uq/test__get_std_corr_matrix'
    std_samples_list = [
        np.array((
            (1.00,),
        )),
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        )),
        np.array((
            (1.00, 0.00),
            (0.00, -1.00)
        )),
        np.array((
            (1.00, 1.00),
            (1.00, 1.00)
        )),
        np.array((
            (1.00, 1e50),
            (-1.00, -1.00)
        )),
    ]
    for file_incr, std_samples in enumerate(std_samples_list):
        res = uq._get_std_corr_matrix(std_samples)
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)
    for bad_item in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            x = np.array((
                (1.00, bad_item),
                (-1.00, -1.00)
            ))
            uq._get_std_corr_matrix(x)


def test__mvn_scale(reset=False):
    data_dir = 'tests/data/uq/test__mvn_scale'
    np.random.seed(40)
    sample_list = [
        np.random.normal(0.00, 1.00, size=(2, 5)).T,
        np.random.normal(1.0e10, 1.00, size=(2, 5)).T
    ]
    rho_list = [
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        )),
        np.array((
            (1.00, 0.00),
            (0.00, 1.00)
        ))
    ]
    for file_incr, args in enumerate(zip(sample_list, rho_list)):
        res = uq._mvn_scale(*args)
        filename = f'{data_dir}/test_{file_incr+1}.pcl'
        if reset: export_pickle(filename, res)
        compare = import_pickle(filename)
        assert np.allclose(res, compare)



def test_fit_distribution_to_sample_univariate(reset=False):
    data_dir = 'tests/data/uq/test_fit_distribution_to_sample_univariate'
    np.random.seed(40)

    # baseline case
    sample_vec = np.array((-3.00, -2.00, -1.00, 0.00, 1.00, 2.00, 3.00))
    res = uq.fit_distribution_to_sample(
        sample_vec,
        'normal'
    )
    assert np.isclose(res[0, 0], np.mean(sample_vec))
    assert np.isclose(res[0, 1], np.std(sample_vec))

    # # censored data  # we have issues here
    # c_lower = -1.50
    # c_upper = 1.50
    # usable_sample_idx = np.all([sample_vec>c_lower, sample_vec<c_upper], axis=0)
    # usable_sample = sample_vec[usable_sample_idx]
    # c_count = len(sample_vec) - len(usable_sample)
    # uq.fit_distribution_to_sample(
    #     usable_sample, 'normal',
    #     censored_count=c_count,
    #     detection_limits=[c_lower, c_upper])
    
    


if __name__ == '__main__':
    pass
