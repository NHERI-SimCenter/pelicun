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
Validation test on loss functions.

In this example, the loss function is a 1:1 mapping of the input EDP.
This means that the resulting loss distribution will be the same as
the EDP input distribution.

"""

import numpy as np
import pandas as pd
import pelicun
from pelicun import assessment


idx = pd.IndexSlice
sample_size = 100000

# initialize a pelicun assessment
assessment = assessment.Assessment({"PrintLog": False, "Seed": 42})

#
# Demands
#

demands = pd.DataFrame(
    {
        'Theta_0': [0.50],
        'Theta_1': [0.90],
        'Family': ['lognormal'],
        'Units': ['g'],
    },
    index=pd.MultiIndex.from_tuples(
        [
            ('PFA', '0', '1'),
        ],
    ),
)

# load the demand model
assessment.demand.load_model({'marginals': demands})

# generate samples
assessment.demand.generate_sample({"SampleSize": sample_size})

#
# Asset
#

# specify number of stories
assessment.stories = 1

# load component definitions
cmp_marginals = pd.read_csv(
    'pelicun/tests/validation/0/data/CMP_marginals.csv', index_col=0
)
cmp_marginals['Blocks'] = cmp_marginals['Blocks']
assessment.asset.load_cmp_model({'marginals': cmp_marginals})

# generate sample
assessment.asset.generate_cmp_sample(sample_size)

#
# Damage
#

# load the models into pelicun
# assessment.damage.load_model_parameters(
#     [
#         additional_damage_db,
#     ],
#     assessment.asset.list_unique_component_ids(as_set=True),
# )

# assessment.damage.calculate()

#
# Losses
#

loss_functions = pelicun.file_io.load_data(
    'pelicun/tests/validation/0/data/loss_functions.csv',
    reindex=False,
    unit_conversion_factors=assessment.unit_conversion_factors,
)


# create the loss map
loss_map = pd.DataFrame(['cmp.A'], columns=['Repair'], index=['cmp.A'])

# load the loss model

assessment.loss.decision_variables = ('Cost',)

assessment.loss.add_loss_map(loss_map)

assessment.loss.load_model_parameters(
    [
        loss_functions,
    ]
)

# perform the calculation
assessment.loss.calculate()

# get the aggregate losses
loss = assessment.loss.aggregate_losses()['repair_cost'].values

# sample median should be close to 0.50
assert np.allclose(np.median(loss), 0.50, atol=1e-2)

# dispersion should be close to 0.9
assert np.allclose(np.log(loss).std(), 0.90, atol=1e-2)


# # TODO also test save/load sample
# assessment.loss.save_sample('/tmp/sample.csv')
# assessment.loss.load_sample('/tmp/sample.csv')
