# %% [markdown]
"""
# Residual drift inference

This example demonstrates a pelicun workflow using
residual drift inference. Two methods are available:

### FEMA P-58

The FEMA P-58 method implements the model in Appendix C of FEMA P-58
Volume 1 (Second Edition).

### Trilinear Weibull

The trilinear Weibull approach is based on research by Vouvakis
Manousakis, Zsarnoczay and Konstantinidis. The model captures the
peak-drift to residual drift relationship using a Weibull distribution
conditioned on peak drift, with conditional parameters defined by a
trilinear function. It is calibrated to available nonlinear
response-history analysis results and is intended to more faithfully
reproduce the observed residual drift patterns.

The example:
1. loads a demand sample from nonlinear time history analysis results of multiple hazard levels,
2. extracts the results for a single hazard level, assuming the goal is to perform an intensity-based evaluation
3. infers residual drift,
4. and visualizes the fitted RID model and the inferred RID sample.
"""

# %%
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pelicun
from pelicun import assessment, file_io
from pelicun.pelicun_warnings import PelicunWarning

# %% [markdown]
"""
## Configuration
"""

# %%
# rid_method = 'trilinear_weibull'
rid_method = 'FEMA P58'

target_hazard_level = '7'

# %% [markdown]
"""
## Initialize assessment
"""

# %%
temp_dir = tempfile.mkdtemp()
asmnt = assessment.Assessment(
    {'PrintLog': True, 'Seed': 415, 'LogFile': f'{temp_dir}/log_file.txt'}
)

# %% [markdown]
"""
## Load demand sample and keep one hazard level

The file contains PID and RID results across all hazard levels.
For the pelicun assessment, we load only one hazard level and remove
the empirical RID columns so that RID can be inferred inside pelicun.
"""

# %%
demand_sample = pd.read_csv(
    'example_4/demand_sample_smrf_9_ii_all_hz.csv',
    header=[0, 1, 2],
    index_col=[0, 1],
)
demand_sample.columns = demand_sample.columns.set_names(['type', 'loc', 'dir'])
demand_sample.index = demand_sample.index.set_names(['hz', 'gm'])
display(demand_sample)

# %%
mask = (
    demand_sample.index.get_level_values('hz')
    .astype(str)
    .isin(['Units', target_hazard_level])
)
demand_sample_hz = demand_sample.loc[mask, :].copy()

empirical_rid = demand_sample_hz['RID'].copy()
demand_sample_hz = demand_sample_hz.drop(columns='RID', level='type')

units = demand_sample_hz.loc[
    demand_sample_hz.index.get_level_values('hz').astype(str) == 'Units'
].copy()
data = demand_sample_hz.loc[
    demand_sample_hz.index.get_level_values('hz').astype(str) == target_hazard_level
].copy()

units.index = ['Units']
data.index = range(len(data))

demand_sample_hz = pd.concat([units, data], axis=0)

asmnt.demand.load_sample(demand_sample_hz)

# %% [markdown]
"""
## Prepare training data for the trilinear Weibull model

The trilinear RID model is fit using structural analysis results from
all hazard levels. The fitted model is then used to infer RID for the
current demand sample.
"""

# %%
training_sample = demand_sample
training_sample = training_sample.loc[
    training_sample.index.get_level_values('hz').astype(str) != 'Units'
].copy()
training_sample.index = pd.MultiIndex.from_arrays(
    [
        training_sample.index.get_level_values('hz').astype(str),
        training_sample.index.get_level_values('gm').astype(str),
    ],
    names=['hz', 'gm'],
)
training_data = (
    training_sample.stack(['loc', 'dir'])
    .loc[:, ['PID', 'RID']]
    .reorder_levels(['hz', 'gm', 'loc', 'dir'])
    .sort_index()
    .astype(float)
)
display(training_data)

# %% [markdown]
"""
## Configure residual drift inference

For `FEMA P58`, this is not required.
For `trilinear_weibull`, the following is an example configuration.

- `approach` can either be `story` or `max_max`. For `story`, a residual drift model is calibrated to the results of each separate `loc`. For `max_max`, a single model is used, simulating the maximum observed RID across stories given the maximum observed PID across stories, regardless of the story they occur.
- `fit_directions_together` can be set to `True` for structures where the PID-RID relationship is likely to be the same for both principal directions (e.g. same structural system and similar stiffness in the two directions). This increases the sample size available for model fitting and can lead to more robust results.
- Accounting for RID correlations across stories is only needed if `approach=story`. The correlation source can be either `model`, in which case correlation structure is inferred from the training sample, or a dictionary containing a reference correlation model (leaving the default values provided here is recommended). The latter case needs the `z/H` value corresponding to each `loc`, where `z` is the elevaiton and `H` is the total height.
- The `model` field can contain an already fitted residual drift model from a previously defined demand model, to avoid repeated fitting to the same data when multiple intensity-based assessments are performed.
"""

# %%
residual_drift_config = {
    'method': rid_method,
    'params': {
        'yield_drift': 0.01,
    },
    'training_data': training_data,
    'force_fit': False,
    'model': None,
    'model_parameters': {
        'approach': 'max_max',  # 'story' or 'max_max'
        'fit_directions_together': True,
        'correlation_source': 'model',
        # 'correlation_source': {
        #     'c1': 10.0,
        #     'c2': 2.0,
        #     'z_over_h': {
        #         '1': 0.126,
        #         '2': 0.235,
        #         '3': 0.345,
        #         '4': 0.454,
        #         '5': 0.563,
        #         '6': 0.672,
        #         '7': 0.782,
        #         '8': 0.891,
        #         '9': 1.000,
        #     },
        # },
        'censoring_limit': 0.005,
    },
}

# %% [markdown]
"""
## Generate demands
"""
config = {
    'ALL': {'DistributionFamily': 'lognormal'},
    'PID': {
        'DistributionFamily': 'lognormal',
        'TruncateUpper': '0.06',
    },
}
asmnt.demand.calibrate_model(config)
asmnt.demand.generate_sample({'SampleSize': 1000})


# %% [markdown]
"""
## Infer residual drift
"""

# %%
if residual_drift_config['method'] in {'FEMA P58', 'FEMA P-58'}:
    asmnt.demand.estimate_RID_and_adjust_sample(
        params=residual_drift_config['params'],
        method=residual_drift_config['method'],
    )
elif residual_drift_config['method'] == 'trilinear_weibull':
    asmnt.demand.infer_residual_drift(residual_drift_config)
else:
    raise ValueError(
        f'Unknown residual drift method: {residual_drift_config["method"]}'
    )

demand_sample_with_rid = asmnt.demand.save_sample()

# %%
display(demand_sample_with_rid)

# %% [markdown]
"""
## Diagnostic plots for the fitted trilinear RID model
"""

# %%
if rid_method == 'trilinear_weibull':
    rid_model = asmnt.demand.residual_drift_model
    assert rid_model is not None

    fig_diag, axs_diag = rid_model.diagnostic_plots(
        figsize=(4.0, 3.0),
        max_cols=3,
        rolling=True,
        training=True,
        model=True,
        show_parameters=True,
        parameter_format='.3g',
    )
    plt.show()

# %% [markdown]
"""
With residual drifts inferred, the rest of the evaluation can proceed
with the asset, damage, and loss calculations.
"""
