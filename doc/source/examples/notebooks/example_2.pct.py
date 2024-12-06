# %% [markdown]
r"""
# Example 2: Damage state validation.

Validation test for the probability of each damage state of a
component.

Here we test whether we get the correct damage state probabilities for
a single component with two damage states.
For such a component, assuming the EDP demand and the fragility curve
capacities are all lognormal, there is a closed-form solution for the
probability of each damage state.
We utilize those equations to ensure that the probabilities obtained
from our Monte-Carlo sample are in line with our expectations.

If $\mathrm{Y} \sim \textrm{LogNormal}(\delta, \beta)$,
then  $\mathrm{X} = \log(\mathrm{Y}) \sim \textrm{Normal}(\mu, \sigma)$ with
$\mu = \log(\delta)$ and $\sigma = \beta$.

$$
\begin{align*}
\mathrm{P}(\mathrm{DS}=0) &= 1 - \Phi\left(\frac{\log(\delta_D) - \log(\delta_{C1})}{\sqrt{\beta_{D}^2 + \beta_{C1}^2}}\right), \\
\mathrm{P}(\mathrm{DS}=1) &= \Phi\left(\frac{\log(\delta_D) - \log(\delta_{C1})}{\sqrt{\beta_D^2 + \beta_{C1}^2}}\right) - \Phi\left(\frac{\log(\delta_{D}) - \log(\delta_{C2})}{\sqrt{\beta_D^2 + \beta_{C2}^2}}\right), \\
\mathrm{P}(\mathrm{DS}=2) &= \Phi\left(\frac{\log(\delta_D) - \log(\delta_{C2})}{\sqrt{\beta_D^2 + \beta_{C2}^2}}\right), \\
\end{align*}
$$
where $\Phi$ is the cumulative distribution function of the standard normal distribution,
$\delta_{C1}$, $\delta_{C2}$, $\beta_{C1}$, $\beta_{C2}$ are the medians and dispersions of the
fragility curve capacities, and $\delta_{D}$, $\beta_{D}$ is
the median and dispersion of the EDP demand.

The equations inherently assume that the capacity RVs for the damage
states are perfectly correlated, which is the case for sequential
damage states.

"""

# %%
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

from pelicun import assessment, file_io

# %%
sample_size = 1000000

asmnt = assessment.Assessment({'PrintLog': False, 'Seed': 42})

# %%
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

# %%
#
# Asset
#

# specify number of stories
asmnt.stories = 1

# load component definitions
cmp_marginals = pd.read_csv('example_2/CMP_marginals.csv', index_col=0)
cmp_marginals['Blocks'] = cmp_marginals['Blocks']
asmnt.asset.load_cmp_model({'marginals': cmp_marginals})

# generate sample
asmnt.asset.generate_cmp_sample(sample_size)

# %%
#
# Damage
#

damage_db = file_io.load_data(
    'example_2/damage_db.csv',
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

# %%
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

# %% nbsphinx="hidden"
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
