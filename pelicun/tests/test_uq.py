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
import pandas as pd
from scipy.stats import norm
from numpy.testing import assert_allclose
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


if __name__ == '__main__':
    pass
