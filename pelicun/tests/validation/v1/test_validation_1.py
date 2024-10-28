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

"""
Validation test for the probability of each damage state of a
component.

"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

from pelicun import assessment, file_io


def test_validation_ds_probabilities() -> None:
    sample_size = 1000000

    asmnt = assessment.Assessment({'PrintLog': False, 'Seed': 42})

    #
    # Demands
    #

    demands = pd.DataFrame(
        {
            'Theta_0': [0.015],
            'Theta_1': [0.60],
            'Family': ['lognormal'],
            'Units': ['rad'],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ('PID', '1', '1'),
            ],
        ),
    )

    # load the demand model
    asmnt.demand.load_model({'marginals': demands})

    # generate samples
    asmnt.demand.generate_sample({'SampleSize': sample_size})

    #
    # Asset
    #

    # specify number of stories
    asmnt.stories = 1

    # load component definitions
    cmp_marginals = pd.read_csv(
        'pelicun/tests/validation/v1/data/CMP_marginals.csv', index_col=0
    )
    cmp_marginals['Blocks'] = cmp_marginals['Blocks']
    asmnt.asset.load_cmp_model({'marginals': cmp_marginals})

    # generate sample
    asmnt.asset.generate_cmp_sample(sample_size)

    #
    # Damage
    #

    damage_db = file_io.load_data(
        'pelicun/tests/validation/v1/data/damage_db.csv',
        reindex=False,
        unit_conversion_factors=asmnt.unit_conversion_factors,
    )
    assert isinstance(damage_db, pd.DataFrame)

    cmp_set = set(asmnt.asset.list_unique_component_ids())

    # load the models into pelicun
    asmnt.damage.load_model_parameters([damage_db], cmp_set)

    # calculate damages
    asmnt.damage.calculate()

    probs = asmnt.damage.ds_model.probabilities()

    #
    # Analytical calculation of the probability of each damage state
    #

    demand_median = 0.015
    demand_beta = 0.60
    capacity_1_median = 0.015
    capacity_2_median = 0.02
    capacity_beta = 0.50

    # If Y is LogNormal(delta, beta), then X = Log(Y) is Normal(mu, sigma)
    # with mu = log(delta) and sigma = beta
    demand_mean = np.log(demand_median)
    capacity_1_mean = np.log(capacity_1_median)
    capacity_2_mean = np.log(capacity_2_median)
    demand_std = demand_beta
    capacity_std = capacity_beta

    p0 = 1.00 - norm.cdf(
        (demand_mean - capacity_1_mean) / np.sqrt(demand_std**2 + capacity_std**2)
    )
    p1 = norm.cdf(
        (demand_mean - capacity_1_mean) / np.sqrt(demand_std**2 + capacity_std**2)
    ) - norm.cdf(
        (demand_mean - capacity_2_mean) / np.sqrt(demand_std**2 + capacity_std**2)
    )
    p2 = norm.cdf(
        (demand_mean - capacity_2_mean) / np.sqrt(demand_std**2 + capacity_std**2)
    )

    assert np.allclose(probs.iloc[0, 0], p0, atol=1e-2)  # type: ignore
    assert np.allclose(probs.iloc[0, 1], p1, atol=1e-2)  # type: ignore
    assert np.allclose(probs.iloc[0, 2], p2, atol=1e-2)  # type: ignore

    #
    # Also test load/save sample
    #

    assert asmnt.damage.ds_model.sample is not None
    asmnt.damage.ds_model.sample = asmnt.damage.ds_model.sample.iloc[0:100, :]
    # (we reduce the number of realizations to conserve resources)
    before = asmnt.damage.ds_model.sample.copy()
    temp_dir = tempfile.mkdtemp()
    asmnt.damage.save_sample(f'{temp_dir}/mdl.csv')
    asmnt.damage.load_sample(f'{temp_dir}/mdl.csv')
    pd.testing.assert_frame_equal(before, asmnt.damage.ds_model.sample)
