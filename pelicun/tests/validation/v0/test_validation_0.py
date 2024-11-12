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
Validation test on loss functions.

In this example, a single loss function is defined as a 1:1 mapping of
the input EDP.  This means that the resulting loss distribution will
be the same as the EDP distribution, allowing us to test and confirm
that this is what happens.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pelicun import assessment, file_io


def test_validation_loss_function() -> None:
    sample_size = 100000

    # initialize a pelicun assessment
    asmnt = assessment.Assessment({'PrintLog': False, 'Seed': 42})

    #
    # Demands
    #

    demands = pd.DataFrame(
        {
            'Theta_0': [0.50],
            'Theta_1': [0.90],
            'Family': ['lognormal'],
            'Units': ['mps2'],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ('PFA', '0', '1'),
            ],
        ),
    )

    asmnt.demand.load_model({'marginals': demands})

    asmnt.demand.generate_sample({'SampleSize': sample_size})

    #
    # Asset
    #

    asmnt.stories = 1

    cmp_marginals = pd.read_csv(
        'pelicun/tests/validation/v0/data/CMP_marginals.csv', index_col=0
    )
    cmp_marginals['Blocks'] = cmp_marginals['Blocks']
    asmnt.asset.load_cmp_model({'marginals': cmp_marginals})

    asmnt.asset.generate_cmp_sample(sample_size)

    #
    # Damage
    #

    # nothing to do here.

    #
    # Losses
    #

    asmnt.loss.decision_variables = ('Cost',)

    loss_map = pd.DataFrame(['cmp.A'], columns=['Repair'], index=['cmp.A'])
    asmnt.loss.add_loss_map(loss_map)

    loss_functions = file_io.load_data(
        'pelicun/tests/validation/v0/data/loss_functions.csv',
        reindex=False,
        unit_conversion_factors=asmnt.unit_conversion_factors,
    )
    assert isinstance(loss_functions, pd.DataFrame)
    asmnt.loss.load_model_parameters([loss_functions])
    asmnt.loss.calculate()

    loss, _ = asmnt.loss.aggregate_losses(future=True)
    assert isinstance(loss, pd.DataFrame)

    loss_vals = loss['repair_cost'].to_numpy()

    # sample median should be close to 0.05
    assert np.allclose(np.median(loss_vals), 0.05, atol=1e-2)
    # dispersion should be close to 0.9
    assert np.allclose(np.log(loss_vals).std(), 0.90, atol=1e-2)

    # TODO(JVM): also test save/load sample
    # asmnt.loss.save_sample('/tmp/sample.csv')
    # asmnt.loss.load_sample('/tmp/sample.csv')
