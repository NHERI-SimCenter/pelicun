# %% [markdown]
"""# Example 1: Simple loss function."""

# %% [markdown]
"""
In this example, a single loss function is defined as a 1:1 mapping of the input EDP.
This means that the resulting loss distribution will be the same as the EDP distribution, allowing us to test and confirm that this is what happens.
"""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from pelicun import assessment, file_io

# %%
sample_size = 100000

# %%
# initialize a pelicun assessment
asmnt = assessment.Assessment({'PrintLog': False, 'Seed': 42})

# %%

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

cmp_marginals = pd.read_csv('example_1/CMP_marginals.csv', index_col=0)
cmp_marginals['Blocks'] = cmp_marginals['Blocks']
asmnt.asset.load_cmp_model({'marginals': cmp_marginals})

asmnt.asset.generate_cmp_sample(sample_size)

# %%
#
# Damage
#

# nothing to do here.

# %%
#
# Losses
#

asmnt.loss.decision_variables = ('Cost',)

loss_map = pd.DataFrame(['cmp.A'], columns=['Repair'], index=['cmp.A'])
asmnt.loss.add_loss_map(loss_map)

loss_functions = file_io.load_data(
    'example_1/loss_functions.csv',
    reindex=False,
    unit_conversion_factors=asmnt.unit_conversion_factors,
)
# %% nbsphinx="hidden"
assert isinstance(loss_functions, pd.DataFrame)
# %%
asmnt.loss.load_model_parameters([loss_functions])
asmnt.loss.calculate()

loss, _ = asmnt.loss.aggregate_losses(future=True)
# %% nbsphinx="hidden"
assert isinstance(loss, pd.DataFrame)

loss_vals = loss['repair_cost'].to_numpy()

# %% nbsphinx="hidden"
# sample median should be close to 0.05
assert np.allclose(np.median(loss_vals), 0.05, atol=1e-2)
# dispersion should be close to 0.9
assert np.allclose(np.log(loss_vals).std(), 0.90, atol=1e-2)
