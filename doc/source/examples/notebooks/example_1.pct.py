# %% [markdown]
# # Example 1

# %% [markdown]
"""
## Introduction

This example focuses on the seismic performance assessment of a steel
moment frame structure using the FEMA P-58 method. We look at demand,
damage, and loss modeling in detail and highlight the inputs required
by Pelicun, the some of the settings researchers might want to
experiment with and the outputs provided by such a high-resolution
calculation.

This example is based on an example notebook for an earlier version of
pelicun, hosted on
[DesignSafe](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3411v5?version=5).

"""

# %%
# Imports
import pprint
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from pelicun.assessment import Assessment, DLCalculationAssessment
from pelicun.base import convert_to_MultiIndex

idx = pd.IndexSlice
pd.options.display.max_rows = 30

# %% [markdown]
"""
## Initialize Assessment

When creating a Pelicun Assessment, you can provide a number of
settings to control the the analysis. The following options are
currently available:

- **Verbose** If True, provides more detailed messages about the
  calculations. Default: False.

- **Seed** Providing a seed makes probabilistic calculations
  reproducible. Default: No seed.

- **PrintLog** If True, prints the messages on the screen as well as
  in the log file. Default: False.

- **LogFile** Allows printing the log to a specific file under a path
  provided here as a string. By default, the log is printed to the
  pelicun_log.txt file.

- **LogShowMS** If True, Pelicun provides more detailed time
  information by showing times up to microsecond
  precision. Default: False, meaning times are provided with
  second precision.

- **SamplingMethod** Three methods are available: {'MonteCarlo',
  'LHS', 'LHS_midpoint'}; Default: LHS_midpoint
    * 'MonteCarlo' stands for conventional random sampling;
    * 'LHS' is Latin HyperCube Sampling with random sample location
       within each chosen bin of the hypercube;
    * 'LHS_midpoint' is like LHS, but the samples are assigned to
       the midpoints of the hypercube bins.

- **DemandOffset** Expects a dictionary with
  {demand_type:offset_value} key-value pairs. demand_type could be
  'PFA' or 'PIH' for example. The offset values are applied to the
  location values when Performance Group locations are parsed to
  demands that control the damage or losses. Default: {'PFA':-1,
  'PFV':-1}, meaning floor accelerations and velocities are pulled
  from the bottom slab associated with the given floor. For
  example, floor 2 would get accelerations from location 1, which
  is the first slab above ground.

- **NonDirectionalMultipliers** Expects a dictionary with
  {demand_type:scale_factor} key-value pairs. demand_type could be
  'PFA' or 'PIH' for example; use 'ALL' to define a scale factor
  for all demands at once. The scale factor considers that for
  components with non-directional behavior the maximum of demands
  is typically larger than the ones available in two orthogonal
  directions. Default: {'ALL': 1.2}, based on FEMA P-58.

- **RepairCostAndTimeCorrelation** Specifies the correlation
  coefficient between the repair cost and repair time of
  individual component blocks. Default: 0.0, meaning uncorrelated
  behavior. Use 1.0 to get perfect correlation or anything between
  0-1 to get partial correlation. Values in the -1 - 0 range are
  also valid to consider negative correlation between cost and
  time.

- **EconomiesOfScale** Controls how the damages are aggregated when
  the economies of scale are calculated. Expects the following
  dictionary: {'AcrossFloors': bool, 'AcrossDamageStates': bool}
  where bool is either True or False. Default: {'AcrossFloors':
  True, 'AcrossDamageStates': False}

  * 'AcrossFloors' if True, aggregates damages across floors to get
    the quantity of damage. If False, it uses damaged quantities and
    evaluates economies of scale independently for each floor.

  * 'AcrossDamageStates' if True, aggregates damages across damage
    states to get the quantity of damage. If False, it uses damaged
    quantities and evaluates economies of scale independently for each
    damage state.

We use the default values for this analysis and only ask for a seed
to make the results repeatable and ask to print the log file to show
outputs within this Jupyter notebook.
"""

# %%
# initialize a pelicun Assessment
assessment = Assessment({'PrintLog': True, 'Seed': 415})

# %% [markdown]
"""
## Demands

### Load demand distribution data

Demand distribution data was extracted from the FEMA P-58 background
documentation referenced in the Introduction. The nonlinear analysis
results from Figures 1-14 &ndash; 1-21 provide the 10th percentile,
median, and 90th percentile of EDPs in two directions on each floor
at each intensity level. We fit a lognormal distributions to those
data and collected the parameters of those distribution in the
demand_data.csv file.

Note that these results do not match the (non-directional) EDP
parameters in Table 1-35 &ndash; 1-42 in the report, so those must
have been processed in another way. The corresponding methodology is
not provided in the report; we are not using the results from those
tables in this example.
"""

# %%
raw_demands = pd.read_csv('example_1/demand_data.csv', index_col=0)
raw_demands

# %% [markdown]
"""
**Pelicun uses SimCenter's naming convention for demands:**

- The first number represents the event_ID. This can be used to
  differentiate between multiple stripes of an analysis, or multiple
  consecutive events in a main-shock - aftershock sequence, for
  example. Currently, Pelicun does not use the first number
  internally, but we plan to utilize it in the future.

- The type of the demand identifies the EDP or IM. The following
  options are available:
  * 'Story Drift Ratio' :              'PID',
  * 'Peak Interstory Drift Ratio':     'PID',
  * 'Roof Drift Ratio' :               'PRD',
  * 'Peak Roof Drift Ratio' :          'PRD',
  * 'Damageable Wall Drift' :          'DWD',
  * 'Racking Drift Ratio' :            'RDR',
  * 'Peak Floor Acceleration' :        'PFA',
  * 'Peak Floor Velocity' :            'PFV',
  * 'Peak Gust Wind Speed' :           'PWS',
  * 'Peak Inundation Height' :         'PIH',
  * 'Peak Ground Acceleration' :       'PGA',
  * 'Peak Ground Velocity' :           'PGV',
  * 'Spectral Acceleration' :          'SA',
  * 'Spectral Velocity' :              'SV',
  * 'Spectral Displacement' :          'SD',
  * 'Peak Spectral Acceleration' :     'SA',
  * 'Peak Spectral Velocity' :         'SV',
  * 'Peak Spectral Displacement' :     'SD',
  * 'Permanent Ground Deformation' :   'PGD',
  * 'Mega Drift Ratio' :               'PMD',
  * 'Residual Drift Ratio' :           'RID',
  * 'Residual Interstory Drift Ratio': 'RID'

- The third part is an integer the defines the location where the
  demand was recorded. In buildings, locations are typically floors,
  but in other assets, locations could reference any other part of the
  structure. Other pelicun examples show how location can also
  identify individual buildings in a regional analysis.

- The last part is an integer the defines the direction of the
  demand. Typically 1 stands for horizontal X and 2 for horizontal Y,
  but any other numbering convention can be used. Direction does not
  have to be used strictly to identify directions. It can be
  considered a generic second-level location identifier that
  differentiates demands and Performance Groups within a location.

The location and direction numbers need to be in line with the
component definitions presented later.

**MultiIndex and SimpleIndex in Pelicun**:

Pelicun uses a hierarchical indexing for rows and columns to organize
data efficiently internally. It provides methods to convert simple
indexes to hierarchical ones (so-called MultiIndex in Python's pandas
package). These methods require simple indexes follow some basic
formatting conventions:

- information at different levels is separated by a dash character: '-'

- no dash character is used in the labels themselves

- spaces are allowed, but are not preserved

The index of the DataFrame above shows how the simple index labels
look like and the DataFrame below shows how they are converted to a
hierarchical MultiIndex.
"""

# %%
# convert index to MultiIndex to make it easier to slice the data
raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
raw_demands.index.names = ['stripe', 'type', 'loc', 'dir']
raw_demands.tail(30)

# %% [markdown]
"""
### Prepare demand input for pelicun

Pelicun offers several options to obtain a desired demand sample:
1. provide the sample directly;
2. provide a distribution (i.e., marginals, and optional correlation
   matrix) and sample it;
3. provide a small set of demand values, fit a distribution, and
   sample that distribution to a large enough sample for performance
   assessment.

In this example, we are going to use the demand information from the
FEMA P-58 background documentation to provide a marginal of each
demand type and sample it (i.e., option 2 from the list). Then, we
will extract the sample from pelicun, extend it with additional demand
types and load it back into Pelicun (i.e., option 1 from the list)

**Scenarios**

Currently, Pelicun performs calculations for one scenario at a
time. Hence, we need to select the stripe we wish to investigate from
the eight available stripes that were used in the multi-stripe
analysis.

**Units**

Pelicun allows users to choose from various units for the all inputs,
including demands. Internally, Pelicun uses Standard International
units, but we support typical units used in the United States as
well. Let us know if a unit you desire to use is not supported - you
will see an error message in this case - we are happy to extend the
list of supported units.
"""

# %%
# we'll use stripe 3 for this example
stripe = '3'
stripe_demands = raw_demands.loc[stripe, :]

# units - - - - - - - - - - - - - - - - - - - - - - - -
stripe_demands.insert(0, 'Units', '')

# PFA is in "g" in this example, while PID is "rad"
stripe_demands.loc['PFA', 'Units'] = 'g'
stripe_demands.loc['PID', 'Units'] = 'rad'

# distribution family  - - - - - - - - - - - - - - - - -
stripe_demands.insert(1, 'Family', '')

# we assume lognormal distribution for all demand marginals
stripe_demands['Family'] = 'lognormal'

# distribution parameters  - - - - - - - - - - - - - - -
# pelicun uses generic parameter names to handle various distributions within the same data structure
# we need to rename the parameter columns as follows:
# median -> theta_0
# log_std -> theta_1
stripe_demands = stripe_demands.rename(
    columns={'median': 'Theta_0', 'log_std': 'Theta_1'}
)

stripe_demands

# %% [markdown]
# Let's plot the demand data to perform a sanity check before the
# analysis

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        '<b>Peak Interstory Drift ratio</b><br> ',
        '<b>Peak Floor Acceleration</b><br> ',
    ),
    shared_yaxes=True,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
)

for demand_i, demand_type in enumerate(['PID', 'PFA']):
    if demand_type == 'PID':
        offset = -0.5
    else:
        offset = 0.0

    for d_i, (dir_, d_color) in enumerate(zip([1, 2], ['blue', 'red'])):
        result_name = f'{demand_type} dir {dir_}'

        params = stripe_demands.loc[
            idx[demand_type, :, str(dir_)], ['Theta_0', 'Theta_1']
        ]
        params.index = params.index.get_level_values(1).astype(float)

        # plot +- 2 log std
        for mul, m_dash in zip([1, 2], ['dash', 'dot']):
            if mul == 1:
                continue

            for sign in [-1, 1]:
                fig.add_trace(
                    go.Scatter(
                        x=np.exp(
                            np.log(params['Theta_0'].values)
                            + params['Theta_1'].to_numpy() * sign * mul
                        ),
                        y=params.index + offset,
                        hovertext=result_name + ' median +/- 2logstd',
                        name=result_name + ' median +/- 2logstd',
                        mode='lines+markers',
                        line={'color': d_color, 'dash': m_dash, 'width': 0.5},
                        marker={'size': 4 / mul},
                        showlegend=False,
                    ),
                    row=1,
                    col=demand_i + 1,
                )

        # plot the medians
        fig.add_trace(
            go.Scatter(
                x=params['Theta_0'].values,
                y=params.index + offset,
                hovertext=result_name + ' median',
                name=result_name + ' median',
                mode='lines+markers',
                line={'color': d_color, 'width': 1.0},
                marker={'size': 8},
                showlegend=False,
            ),
            row=1,
            col=demand_i + 1,
        )

        if d_i == 0:
            shared_ax_props = {
                'showgrid': True,
                'linecolor': 'black',
                'gridwidth': 0.05,
                'gridcolor': 'rgb(192,192,192)',
            }

            if demand_type == 'PID':
                fig.update_xaxes(
                    title_text='drift ratio',
                    range=[0, 0.05],
                    row=1,
                    col=demand_i + 1,
                    **shared_ax_props,
                )

            elif demand_type == 'PFA':
                fig.update_xaxes(
                    title_text='acceleration [g]',
                    range=[0, 1.0],
                    row=1,
                    col=demand_i + 1,
                    **shared_ax_props,
                )

            if demand_i == 0:
                fig.update_yaxes(
                    title_text='story',
                    range=[0, 4],
                    row=1,
                    col=demand_i + 1,
                    **shared_ax_props,
                )
            else:
                fig.update_yaxes(
                    range=[0, 4], row=1, col=demand_i + 1, **shared_ax_props
                )

fig.update_layout(
    title=f'intensity level {stripe} ~ 475 yr return period',
    height=500,
    width=900,
    plot_bgcolor='white',
)

fig.show()

# %% [markdown]
"""
### Sample the demand distribution

The scripts below load the demand marginal information to Pelicun and
ask it to generate a sample with the provided number of
realizations. We do not have correlation information from the
background documentation, but it is generally better (i.e.,
conservative from a damage, loss, and risk point of view) to assume
perfect correlation in such cases than to assume independence. Hence,
we prepare a correlation matrix that represents perfect correlation
and feed it to Pelicun with the marginal parameters.

After generating the sample, we extract it and print the first few
realizations below.
"""

# %%
# prepare a correlation matrix that represents perfect correlation
ndims = stripe_demands.shape[0]
demand_types = stripe_demands.index

perfect_corr = pd.DataFrame(
    np.ones((ndims, ndims)), columns=demand_types, index=demand_types
)

# load the demand model
assessment.demand.load_model(
    {'marginals': stripe_demands, 'correlation': perfect_corr}
)

# %%
# choose a sample size for the analysis
sample_size = 10000

# generate demand sample
assessment.demand.generate_sample({'SampleSize': sample_size})

# extract the generated sample

# Note that calling the save_sample() method is better than directly
# pulling the sample attribute from the demand object because the
# save_sample method converts demand units back to the ones you
# specified when loading in the demands.

demand_sample = assessment.demand.save_sample()

demand_sample.head()

# %% [markdown]
r"""
### Extend the sample

The damage and loss models we use later in this example need residual
drift and spectral acceleration [Sa(T=1.13s)] information for each
realizations. The residual drifts are used to consider irreparable
damage to the building; the spectral accelerations are used to
evaluate the likelihood of collapse.

**Residual drifts**

Residual drifts could come from nonlinear analysis, but they are often
not available or not robust enough. Pelicun provides a convenience
method to convert PID to RID and we use that function in this
example. Currently, the method implements the procedure recommended in
FEMA P-58, but it is designed to support multiple approaches for
inferring RID from available demand information.

The FEMA P-58 RID calculation is based on the yield drift ratio. There
are conflicting data in FEMA P-58 on the yield drift ratio that should
be applied for this building:

* According to Vol 2 4.7.3, $\Delta_y = 0.0035$ , but this value leads
  to excessive irreparable drift likelihood that does not match the
  results in the background documentation.

* According to Vol 1 Table C-2, $\Delta_y = 0.0075$ , which leads to
  results that are more in line with those in the background
  documentation.

We use the second option below. Note that we get a different set of
residual drift estimates for every floor of the building.

**Spectral acceleration**

The Sa(T) can be easily added as a new column to the demand
sample. Note that Sa values are identical across all realizations
because we are running the analysis for one stripe that has a
particular Sa(T) assigned to it. We assign the Sa values to direction
1 and we will make sure to have the collapse fragility defined as a
directional component (see Damage/Fragility data) to avoid scaling
these spectral accelerations with the nondirectional scale factor.

The list below provides Sa values for each stripe from the analysis -
the values are from the background documentation referenced in the
Introduction.
"""

# %%
# get residual drift estimates
delta_y = 0.0075
PID = demand_sample['PID']

RID = assessment.demand.estimate_RID(PID, {'yield_drift': delta_y})

# and join them with the demand_sample
demand_sample_ext = pd.concat([demand_sample, RID], axis=1)

# add spectral acceleration
Sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756]
demand_sample_ext['SA_1.13', 0, 1] = Sa_vals[int(stripe) - 1]

demand_sample_ext.describe().T

# %% [markdown]
"""
The plot below illustrates that the relationship between a PID and RID
variable is not multivariate lognormal. This underlines the importance
of generating the sample for such additional demand types
realization-by-realization rather than adding a marginal RID to the
initial set and asking Pelicun to sample RIDs from a multivariate
lognormal distribution.

You can use the plot below to display the joint distribution of any
two demand variables
"""

# %%
# plot two demands from the sample

demands = ['PID-1-1', 'RID-1-1']

fig = go.Figure()

demand_file = 'response.csv'
output_path = 'doc/source/examples/notebooks/example_1/output'
coupled_edp = True
realizations = '100'
auto_script_path = 'PelicunDefault/Hazus_Earthquake_IM.py'
detailed_results = False
output_format = None
custom_model_dir = None
color_warnings = False

shared_ax_props = {
    'showgrid': True,
    'linecolor': 'black',
    'gridwidth': 0.05,
    'gridcolor': 'rgb(192,192,192)',
    'type': 'log',
}

if 'PFA' in demands[0]:
    fig.update_xaxes(
        title_text=f'acceleration [g]<br>{demands[0]}',
        range=np.log10([0.001, 1.5]),
        **shared_ax_props,
    )

else:
    fig.update_xaxes(
        title_text=f'drift ratio<br>{demands[0]}',
        range=np.log10([0.001, 0.1]),
        **shared_ax_props,
    )

if 'PFA' in demands[1]:
    fig.update_yaxes(
        title_text=f'{demands[1]}<br>acceleration [g]',
        range=np.log10([0.0001, 1.5]),
        **shared_ax_props,
    )

else:
    fig.update_yaxes(
        title_text=f'{demands[1]}<br>drift ratio',
        range=np.log10([0.0001, 0.1]),
        **shared_ax_props,
    )


fig.update_layout(title='demand sample', height=600, width=650, plot_bgcolor='white')

fig.show()

# %% [markdown]
"""
### Load Demand Samples

The script below adds unit information to the sample data and loads it
to Pelicun.

Note that you could skip the first part of this demand calculation and
prepare a demand sample entirely by yourself. That allows you to
consider any kind of distributions and any kind of correlation
structure between the demands. As long as you have the final list of
realizations formatted according to the conventions explained above,
you should be able to load it directly to Pelicun.
"""

# %%
# add units to the data
demand_sample_ext.T.insert(0, 'Units', '')

# PFA and SA are in "g" in this example, while PID and RID are "rad"
demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
demand_sample_ext.loc['Units', ['PID', 'RID']] = 'rad'

display(demand_sample_ext)

assessment.demand.load_sample(demand_sample_ext)

# %% [markdown]
# This concludes the Demand section. The demand sample is ready, we
# can move on to damage calculation

# %% [markdown]
"""
## Damage

Damage simulation requires an asset model, fragility data, and a
damage process that describes dependencies between damages in the
system. We will look at each of these in detail below.

### Define asset model

The asset model assigns components to the building and defines where
they are and how much of each component is at each location.

The asset model can consider uncertainties in the types of components
assigned and in their quantities. This example does not introduce
those uncertainties for the sake of brevity, but they are discussed in
other examples. For this example, the component types and their
quantities are identical in all realizations.

Given this deterministic approach, we can take advantage of a
convenience method in Pelicun for defining the asset model. We can
prepare a table (see the printed data below) where each row identifies
a component and assigns some quantity of it to a set of locations and
directions. Such a table can be prepared in Excel or in a text editor
and saved in a CSV file - like we did in this example, see
CMP_marginals.csv - or it could be prepared as part of this
script. Storing these models in a CSV file facilitates sharing the
basic inputs of an analysis with other researchers.

The tabular information is organized as follows:

* Each row in the table can assign component quantities (Theta_0) to
  one or more Performance Groups (PG). A PG is a group of components
  at a given floor (location) and direction that is affected by the
  same demand (EDP or IM) values.

* The quantity defined under Theta_0 is assigned to each location and
  direction listed. For example, the first row in the table below
  assigns 2.0 of B.10.41.001a to the third and fourth floors in
  directions 1 and 2. That is, it creates 4 Performance Groups, each
  with 2 of these components in it.

* Zero ("0") is reserved for "Not Applicable" use cases in the
  location and direction column. As a location, it represents
  components with a general effect that cannot be linked to a
  particular floor (e.g., collapse). In directions, it is used to
  identify non-directional components.

* The index in this example refers to the component ID in FEMA P58,
  but it can be any arbitrary string that has a corresponding entry in
  the applied fragility database (see the Fragility data section below
  for more information).

* Blocks are the number of independent units within a Performance
  Group. By default (i.e., when the provided value is missing or NaN),
  each PG is assumed to have one block which means that all of the
  components assigned to it will have the same behavior. FEMA P-58
  identifies unit sizes for its components. We used these sizes to
  determine the number of independent blocks for each PG. See, for
  example, B.20.22.031 that has a 30 ft2 unit size in FEMA P-58. We
  used a large number of blocks to capture that each of those curtain
  wall elements can get damaged independently of the others.

* Component quantities (Theta_0) can be provided in any units
  compatible with the component type. (e.g., ft2, inch2, m2 are all
  valid)

* The last three components use custom fragilities that are not part
  of the component database in FEMA P-58. We use these to consider
  irreparable damage and collapse probability. We will define the
  corresponding fragility and consequence functions in later sections
  of this example.

* The Comment column is not used by Pelicun, any text is acceptable
  there.

"""
# %%
# load the component configuration
cmp_marginals = pd.read_csv('example_1/CMP_marginals.csv', index_col=0)

display(cmp_marginals.head(15))
print('...')
cmp_marginals.tail(10)

# %%
# to make the convenience keywords work in the model, we need to
# specify the number of stories
assessment.stories = 4

# now load the model into Pelicun
assessment.asset.load_cmp_model({'marginals': cmp_marginals})

# %% [markdown]
"""
Note that we could have assigned uncertain component quantities by
adding a "Family" and "Theta_1", "Theta_2" columns to describe their
distribution. Additional "TruncateLower" and "TruncateUpper" columns
allow for bounded component quantity distributions that is especially
useful when the distribution family is supported below zero values.

Our input in this example describes a deterministic configuration
resulting in the fairly simple table shown below.
"""

# %%
# let's take a look at the generated marginal parameters
assessment.asset.cmp_marginal_params.loc['B.10.41.002a', :]

# %% [markdown]
"""
### Sample asset distribution

In this example, the quantities are identical for every
realization. We still need to generate a component quantity sample
because the calculations in Pelicun expect an array of component
quantity realizations. The sample size for the component quantities is
automatically inferred from the demand sample. If such a sample is not
available, you need to provide a sample size as the first argument of
the generate_cmp_sample method.

The table below shows the statistics for each Performance Group's
quantities. Notice the zero standard deviation and that the minimum
and maximum values are identical - this confirms that the quantities
are deterministic.

We could edit this sample and load the edited version back to Pelicun
like we did for the Demands earlier.
"""

# %%
# Generate the component quantity sample
assessment.asset.generate_cmp_sample()

# get the component quantity sample - again, use the save function to
# convert units
cmp_sample = assessment.asset.save_cmp_sample()

cmp_sample.describe()

# %% [markdown]
"""
### Define component fragilities

Pelicun comes with fragility data, including the FEMA P-58 component
fragility functions. We will start with taking a look at those data
first.

Pelicun uses the following terminology for fragility data:

- Each Component has a number of pre-defined Limit States (LS) that
  are triggered when a controlling Demand exceeds the Capacity of the
  component.

- The type of controlling Demand can be any of the demand types
  supported by the tool - see the list of types in the Demands section
  of this example.

- Units of the controlling Demand can be chosen freely, as long as
  they are compatible with the demand type (e.g., g, mps2, ftps2 are
  all acceptable for accelerations, but inch and m are not)

- The controlling Demand can be Offset in terms of location (e.g.,
  ceilings use acceleration from the floor slab above the floor) by
  providing a non-zero integer in the Offset column.

- The Capacity of a component can be either deterministic or
  probabilistic. A deterministic capacity only requires the assignment
  of Theta_0 to the limit state. A probabilistic capacity is described
  by a Fragility function. Fragility functions use Theta_0 as well as
  the Family and Theta_1 (i.e., the second parameter) to define a
  distribution function for the random capacity variable.

- When a Limit State is triggered, the Component can end up in one or
  more Damage States. DamageStateWeights are used to assign more than
  one mutually exclusive Damage States to a Limit State. Using more
  than one Damage States allows us to recognize multiple possible
  damages and assign unique consequences to each damage in the loss
  modeling step.

- The Incomplete flag identifies components that require additional
  information from the user. More than a quarter of the components in
  FEMA P-58 have incomplete fragility definitions. If the user does
  not provide the missing information, Pelicun provides a warning
  message and skips Incomplete components in the analysis.

The SimCenter is working on a web-based damage and loss library that
will provide a convenient overview of the available fragility and
consequence data. Until then, the get_default_data method allows you
to pull any of the default fragility datasets from Pelicun and
review/edit/reload the data.
"""

# %%
# review the damage model - in this example: fragility functions
P58_data = assessment.get_default_data('damage_DB_FEMA_P58_2nd')

display(P58_data.head(3))

print(P58_data['Incomplete'].sum(), ' incomplete component fragility definitions')

# %% [markdown]
"""
Let's focus on the incomplete column and check which of the components
we want to use have incomplete damage models. We do this by filtering
the component database and only keeping those components that are part
of our asset model and have incomplete definitions.
"""

# %%
# note that we drop the last three components here (excessiveRID, irreparable, and collapse)
# because they are not part of P58
cmp_list = cmp_marginals.index.unique().to_numpy()[:-3]

P58_data_for_this_assessment = P58_data.loc[cmp_list, :].sort_values(
    'Incomplete', ascending=False
)

additional_fragility_db = P58_data_for_this_assessment.loc[
    P58_data_for_this_assessment['Incomplete'] == 1
].sort_index()

additional_fragility_db

# %% [markdown]
"""
The component database bundled with Pelicun includes a CSV file and a
JSON file for each dataset. The CSV file contains the data required to
perform the calculations; the JSON file provides additional metadata
for each component. The get_default_metadata method in Pelicun
provides convenient access to this metadata. Below we demonstrate how
to pull in the data on the first incomplete component. The metadata in
this example are directly from FEMA P-58.
"""
# %%
P58_metadata = assessment.get_default_metadata('damage_DB_FEMA_P58_2nd')

pprint.pprint(P58_metadata['D.20.22.013a'])

# %% [markdown]
"""
We need to add the missing information to the incomplete components.

Note that the numbers below are just reasonable placeholders. This
step would require substantial work from the engineer to review these
components and assign the missing values. Such work is out of the
scope of this example.

The table below shows the completed fragility information.
"""

# %%
# D2022.013a, 023a, 023b - Heating, hot water piping and bracing
# dispersion values are missing, we use 0.5
additional_fragility_db.loc[
    ['D.20.22.013a', 'D.20.22.023a', 'D.20.22.023b'],
    [('LS1', 'Theta_1'), ('LS2', 'Theta_1')],
] = 0.5

# D2031.013b - Sanitary Waste piping
# dispersion values are missing, we use 0.5
additional_fragility_db.loc['D.20.31.013b', ('LS1', 'Theta_1')] = 0.5

# D2061.013b - Steam piping
# dispersion values are missing, we use 0.5
additional_fragility_db.loc['D.20.61.013b', ('LS1', 'Theta_1')] = 0.5

# D3031.013i - Chiller
# use a placeholder of 1.5|0.5
additional_fragility_db.loc['D.30.31.013i', ('LS1', 'Theta_0')] = 1.5  # g
additional_fragility_db.loc['D.30.31.013i', ('LS1', 'Theta_1')] = 0.5

# D3031.023i - Cooling Tower
# use a placeholder of 1.5|0.5
additional_fragility_db.loc['D.30.31.023i', ('LS1', 'Theta_0')] = 1.5  # g
additional_fragility_db.loc['D.30.31.023i', ('LS1', 'Theta_1')] = 0.5

# D3052.013i - Air Handling Unit
# use a placeholder of 1.5|0.5
additional_fragility_db.loc['D.30.52.013i', ('LS1', 'Theta_0')] = 1.5  # g
additional_fragility_db.loc['D.30.52.013i', ('LS1', 'Theta_1')] = 0.5

# We can set the incomplete flag to 0 for these components
additional_fragility_db['Incomplete'] = 0

additional_fragility_db

# %% [markdown]
"""
Now we need to add three new components:

* **excessiveRID** is used to monitor residual drifts on every floor
    in every direction and check if they exceed the capacity assigned
    to irreparable damage.

* **irreparable** is a global limit state that is triggered by having
    at least one excessive RID and leads to the replacement of the
    building. This triggering requires one component to affect another
    and it is handled in the Damage Process section below. For its
    individual damage evaluation, this component uses a deterministic,
    placeholder capacity that is sufficiently high so that it will
    never get triggered by the controlling demand.

* **collapse** represents the global collapse limit state that is
    modeled with a collapse fragility function and uses spectral
    acceleration at the dominant vibration period as the
    demand. Multiple collapse modes could be considered by assigning a
    set of Damage State weights to the collapse component.

The script in this cell creates the table shown below. We could also
create such information in a CSV file and load it to the notebook.
"""

# %%

# irreparable damage
# this is based on the default values in P58
additional_fragility_db.loc[
    'excessiveRID',
    [
        ('Demand', 'Directional'),
        ('Demand', 'Offset'),
        ('Demand', 'Type'),
        ('Demand', 'Unit'),
    ],
] = [1, 0, 'Residual Interstory Drift Ratio', 'rad']

additional_fragility_db.loc[
    'excessiveRID', [('LS1', 'Family'), ('LS1', 'Theta_0'), ('LS1', 'Theta_1')]
] = ['lognormal', 0.01, 0.3]

additional_fragility_db.loc[
    'irreparable',
    [
        ('Demand', 'Directional'),
        ('Demand', 'Offset'),
        ('Demand', 'Type'),
        ('Demand', 'Unit'),
    ],
] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']


# a very high capacity is assigned to avoid damage from demands
additional_fragility_db.loc['irreparable', ('LS1', 'Theta_0')] = 1e10

# collapse
# capacity is assigned based on the example in the FEMA P58 background documentation
additional_fragility_db.loc[
    'collapse',
    [
        ('Demand', 'Directional'),
        ('Demand', 'Offset'),
        ('Demand', 'Type'),
        ('Demand', 'Unit'),
    ],
] = [1, 0, 'Peak Spectral Acceleration|1.13', 'g']


additional_fragility_db.loc[
    'collapse', [('LS1', 'Family'), ('LS1', 'Theta_0'), ('LS1', 'Theta_1')]
] = ['lognormal', 1.35, 0.5]

# We set the incomplete flag to 0 for the additional components
additional_fragility_db['Incomplete'] = 0

additional_fragility_db.tail(3)

# %% [markdown]
"""
### Load component fragility data

Now that we have the fragility data completed and available for all
components in the asset model, we can load the data to the damage
model in Pelicun.

When providing custom data, you can directly provide a DataFrame like
we do in this example (additional_fragility_db), or you can provide a
path to a CSV file that is structured like the table we prepared
above.

Default databases are loaded using the keyword "PelicunDefault" in the
path and then providing the name of the database. The PelicunDefault
keyword is automatically replaced with the path to the default
component data directory.

Note that there are identical components in the listed sources. The
additional_fragility_db contains the additional global components
(e.g., collapse) and the ones that are incomplete in FEMA P-58. The
latter ones are also listed in the default FEMA P-58 database. Such
conflicts are resolved by preserving the first occurrence of every
component. Hence, always start with the custom data when listing
sources and add default databases in the end.
"""

# %%
cmp_set = set(assessment.asset.list_unique_component_ids())
assessment.damage.load_model_parameters(
    [
        additional_fragility_db,  # This is the extra fragility data we've just created
        'PelicunDefault/damage_DB_FEMA_P58_2nd.csv',  # and this is a table with the default P58 data
    ],
    cmp_set,
)

# %% [markdown]
"""
### Damage Process

Damage processes are a powerful new feature in Pelicun 3. They are
used to connect damages of different components in the performance
model and they can be used to create complex cascading damage models.

The default FEMA P-58 damage process is fairly simple. The process
below can be interpreted as follows:

* If Damage State 1 (DS1) of the collapse component is triggered
  (i.e., the building collapsed), then damage for all other components
  should be cleared from the results. This considers that component
  damages (and their consequences) in FEMA P-58 are conditioned on no
  collapse.

* If Damage State 1 (DS1) of any of the excessiveRID components is
  triggered (i.e., the residual drifts are larger than the prescribed
  capacity on at least one floor), then the irreparable component
  should be set to DS1.

"""

# %%
# FEMA P58 uses the following process:
dmg_process = {
    '1_collapse': {'DS1': 'ALL_NA'},
    '2_excessiveRID': {'DS1': 'irreparable_DS1'},
}

# %% [markdown]
"""
### Damage calculation

Damage calculation in Pelicun requires

- a pre-assigned set of component fragilities;

- a pre-assigned sample of component quantities;

- a pre-assigned sample of demands;

- and an (optional) damage process

The sample size for the damage calculation is automatically inferred
from the demand sample size.

**Expected Runtime & Best Practices**

The output below shows the total number of Performance Groups (121)
and Component Blocks (1736). The number of component blocks is a good
proxy for the size of the problem. Damage calculation is the most
demanding part of the performance assessment workflow. The runtime for
damage calculations in Pelicun scales approximately linearly with the
number of component blocks above 500 blocks and somewhat better than
linearly with the sample size above 10000 samples. Below 10000 sample
size and 500 blocks, the overhead takes a substantial part of the
approximately few second calculation time. Below 1000 sample size and
100 blocks, these variables have little effect on the runtime.

Pelicun can handle failry large problems but it is ideal to make sure
both the intermediate data and the results fit in the RAM of the
system. Internal calculations are automatically disaggregated to
1000-block batches at a time to avoid memory-related issues. This
might still be too large of a batch if the number of samples is more
than 10,000. You can manually adjust the batch size using the
block_batch_size argument in the calculate method below. We recommend
using only 100-block batches when running a sample size of
100,000. Even larger sample sizes coupled with a complex model
probably benefit from running in batches across the sample. Contact
the SimCenter if you are interested in such large problems and we are
happy to provide support.

Results are stored at a Performance Group (rather than Component
Block) resolution to allow users to run larger problems. The size of
the output data is proportional to the number of Performance Groups x
number of active Damage States per PG x sample size. Modern computers
with 64-bit memory addressing and 4+ GB of RAM should be able to
handle problems with up to 10,000 performance groups and a sample size
of 10,000. This limit shall be sufficient for even the most complex
and high resolution models of a single building - note in the next
cell that the size of the results from this calculation (121 PG x
10,000 realizations) is just 30 MB.
"""

# %%
# Now we can run the calculation
assessment.damage.calculate(
    dmg_process=dmg_process
)  # , block_batch_size=100) #- for large calculations

# %% [markdown]
"""
### Damage estimates

Below, we extract the damage sample from Pelicun and show a few
example plots to illustrate how rich information this data provides
about the damages in the building
"""

# %%
damage_sample = assessment.damage.save_sample()

print('Size of damage results: ', sys.getsizeof(damage_sample) / 1024 / 1024, 'MB')

# %% [markdown]
"""
**Damage statistics of a component type**

The table printed below shows the mean, standard deviation, minimum,
10th, 50th, and 90th percentile, and maximum quantity of the given
component in each damage state across various locations and directions
in the building.
"""

# %%
component = 'B.20.22.031'
damage_sample.describe([0.1, 0.5, 0.9]).T.loc[component, :].head(30)

# %%
dmg_plot = (
    damage_sample.loc[:, component].groupby(level=['loc', 'ds'], axis=1).sum().T
)

px.bar(
    x=dmg_plot.index.get_level_values(1),
    y=dmg_plot.mean(axis=1),
    color=dmg_plot.index.get_level_values(0),
    barmode='group',
    labels={'x': 'Damage State', 'y': 'Component Quantity [ft2]', 'color': 'Floor'},
    title=f'Mean Quantities of component {component} in each Damage State',
    height=500,
)

# %%
dmg_plot = (
    damage_sample.loc[:, component]
    .loc[:, idx[:, :, :, '2']]
    .groupby(level=['loc', 'ds'], axis=1)
    .sum()
    / damage_sample.loc[:, component].groupby(level=['loc', 'ds'], axis=1).sum()
).T

fifty_percent = 0.50
px.bar(
    x=dmg_plot.index.get_level_values(0),
    y=(dmg_plot > fifty_percent).mean(axis=1),
    color=dmg_plot.index.get_level_values(1),
    barmode='group',
    labels={'x': 'Floor', 'y': 'Probability', 'color': 'Direction'},
    title=f'Probability of having more than 50% of component {component} in DS2',
    height=500,
)

# %%
dmg_plot = (
    damage_sample.loc[:, component]
    .loc[:, idx[:, :, :, '2']]
    .groupby(level=[0], axis=1)
    .sum()
    / damage_sample.loc[:, component].groupby(level=[0], axis=1).sum()
).T

px.scatter(
    x=dmg_plot.loc['1'],
    y=dmg_plot.loc['2'],
    color=dmg_plot.loc['3'],
    opacity=0.1,
    color_continuous_scale=px.colors.diverging.Portland,
    marginal_x='histogram',
    marginal_y='histogram',
    labels={
        'x': 'Proportion in DS2 in Floor 1',
        'y': 'Proportion in DS2 in Floor 2',
        'color': 'Proportion in<br>DS2 in Floor 3',
    },
    title=f'Correlation between component {component} damages across three floors',
    height=600,
    width=750,
)

# %%
print(
    'Probability of collapse: ',
    1.0 - damage_sample['collapse', '0', '1', '0', '0'].mean(),
)
print(
    'Probability of irreparable damage: ',
    damage_sample['irreparable', '0', '1', '0', '1'].mean(),
)

# %% [markdown]
"""
## Losses - repair consequences

Loss simulation is an umbrella term that can include the simulation of
various types of consequences. In this example we focus on repair cost
and repair time consequences. Pelicun provides a flexible framework
that can be expanded with any arbitrary decision variable. Let us know
if you need a particular decision variable for your work that would be
good to support in Pelicun.

Losses can be either based on consequence functions controlled by the
quantity of damages, or based on loss functions controlled by demand
intensity. Pelicun supports both approaches and they can be mixed
within the same analysis; in this example we use consequence functions
following the FEMA P-58 methodology.

Loss simulation requires a demand/damage sample, consequence/loss
function data, and a mapping that links the demand/damage components
to the consequence/loss functions. The damage sample in this example
is already available from the previous section. We will show below how
to prepare the mapping matrix and how to load the consequence
functions.

### Consequence mapping to damages

Consequences are decoupled from damages in pelicun to enforce and
encourgae a modular approach to performance assessment.

The map that we prepare below describes which type of damage leads to
which type of consequence. With FEMA P-58 this is quite
straightforward because the IDs of the fragility and consequence data
are identical - note that we would have the option to link different
ones though. Also, several fragilities in P58 have identical
consequences and the approach in Pelicun will allow us to remove such
redundancy in future datasets.  We plan to introduce a database that
is a more concise and streamlined version of the one provided in FEMA
P58 and encourage researchers to extend it by providing data to the
incomplete components.

The mapping is defined by a table (see the example below). Each row
has a demand/damage ID and a list of consequence IDs, one for each
type of decision variable. Here, we are looking at building repair
consequences only, hence, there is only one column with consequence
IDs. The IDs of FEMA P-58 consequence functions are identical to the
name of the components they are assigned to. Damage sample IDs in the
index of the table are preceded by 'DMG', while demand sample IDs
would be preceded by 'DEM'.

Notice that besides the typical FEMA P-58 IDs, the table also includes
'DMG-collapse' and 'DMG-irreparable' to capture the consequences of
those events. Both irreparable damage and collapse lead to the
replacement of the building. Consequently, we can use the same
consequence model (called 'replacement') for both types of damages. We
will define what the replacement consequence is in the next section.
"""

# %%
# let us prepare the map based on the component list

# we need to prepend 'DMG-' to the component names to tell pelicun to look for the damage of these components
drivers = cmp_marginals.index.unique().to_list()
drivers = drivers[:-3] + drivers[-2:]

# we are looking at repair consequences in this example
# the components in P58 have consequence models under the same name
loss_models = cmp_marginals.index.unique().tolist()[:-3]

# We will define the replacement consequence in the following cell.
loss_models += ['replacement'] * 2

# Assemble the DataFrame with the mapping information
# The column name identifies the type of the consequence model.
loss_map = pd.DataFrame(loss_models, columns=['Repair'], index=drivers)

loss_map

# %% [markdown]
"""
### Define component consequence data

Pelicun comes with consequence data, including the FEMA P-58 component
consequence functions. We will start with taking a look at those data
first.

Pelicun uses the following terminology for consequence data:

- Each Component has a number of pre-defined Damage States (DS)

- The quantity of each Component in each DS in various locations and
  direction in the building is provided as a damage sample.

- The index of the consequence data table can be hierarchical and list
  several consequence types that belong to the same group. For
  example, the repair consequences here include 'Cost' and 'Time';
  injury consequences include injuries of various severity. Each row
  in the table corresponds to a combination of a component and a
  consequence type.

- Consequences in each damage state can be:

    * Deterministic: use only the 'Theta_1' column

    * Probabilistic: provide information on the 'Family', 'Theta_0'
      and 'Theta_1' to describe the distribution family and its two
      parameters.

- The first parameter of the distribution (Theta_0) can be either a
  scalar or a function of the quantity of damage. This applies to both
  deterministic and probabilistic cases. When Theta_0 is a function of
  the quantity of damage, two series of numbers are expected,
  separated by a '|' character. The two series are used to construct a
  multilinear function - the first set of numbers are the Theta_0
  values, the second set are the corresponding quantities. The
  functions are assumed to be constant below the minimum and above the
  maximum quantities.

- The LongLeadTime column is currently informational only - it does
  not affect the calculation.

- The DV-Unit column (see the right side of the table below) defines
  the unit of the outputs for each consequence function - i.e., the
  unit of the Theta_0 values.

- The Quantity-Unit column defines the unit of the damage/demand
  quantity. This allows mixing fragility and consequence functions
  that use different units - as long as the units are compatible,
  Pelicun takes care of the conversions automatically.

- The Incomplete column is 1 if some of the data is missing from a
  row.

The SimCenter is working on a web-based damage and loss library that
will provide a convenient overview of the available fragility and
consequence data. Until then, the get_default_data method allows you
to pull any of the default consequence datasets from Pelicun and
review/edit/reload the data.

After pulling the data, first, we need to check if the repair
consequence functions for the components in this building are complete
in FEMA P-58. 27 components in FEMA P-58 only have damage models and
do not have repair consequence models at all. All of the other models
are complete. As you can see from the message below, this example only
includes components with complete consequence information.
"""

# %%
# load the consequence models
P58_data = assessment.get_default_data('loss_repair_DB_FEMA_P58_2nd')

# get the consequences used by this assessment
P58_data_for_this_assessment = P58_data.loc[loss_map['Repair'].to_numpy()[:-2], :]

print(
    P58_data_for_this_assessment['Incomplete'].sum(),
    ' components have incomplete consequence models assigned.',
)

display(P58_data_for_this_assessment.head(30))

# %% [markdown]
r"""
**Adding custom consequence functions**

Now we need to define the replacement consequence for the collapse and
irreparable damage cases.

The FEMA P-58 background documentation provides the \$21.6 million as
replacement cost and 400 days as replacement time. The second edition
of FEMA P-58 introduced worker-days as the unit of replacement time;
hence, we need a replacement time in worker-days. We show two options
below to estimate that value:

- We can use the assumption of 0.001 worker/ft2 from FEMA P-58
  multiplied by the floor area of the building to get the average
  number of workers on a typical day. The total number of worker-days
  is the product of the 400 days of construction and this average
  number of workers. Using the plan area of the building for this
  calculation assumes that one floor is being worked on at a time -
  this provides a lower bound of the number of workers: 21600 x 0.001
  = 21.6. The upper bound of workers is determined by using the gross
  area for the calculation: 86400 x 0.001 = 86.4. Consequently, the
  replacement time will be between 8,640 and 34,560 worker-days.

- The other approach is taking the replacement cost, assuming a ratio
  that is spent on labor (0.3-0.5 is a reasonable estimate) and
  dividing that labor cost with the daily cost of a worker (FEMA P-58
  estimates \$680 in 2011 USD for the SF Bay Area which we will apply
  to this site in Los Angeles). This calculation yields 9,529 - 15,882
  worker-days depending on the labor ratio chosen.

Given the above estimates, we use 12,500 worker-days for this example.

Note that

- We efficiently use the same consequence for the collapse and
  irreparable damages

- We could consider uncertainty in the replacement cost/time with this
  approach. We are not going to do that now for the sake of simplicity

"""

# %%
# initialize the dataframe
additional_consequences = pd.DataFrame(
    columns=pd.MultiIndex.from_tuples(
        [
            ('Incomplete', ''),
            ('Quantity', 'Unit'),
            ('DV', 'Unit'),
            ('DS1', 'Theta_0'),
        ]
    ),
    index=pd.MultiIndex.from_tuples(
        [('replacement', 'Cost'), ('replacement', 'Time')]
    ),
)

# add the data about replacement cost and time
additional_consequences.loc['replacement', 'Cost'] = [
    0,
    '1 EA',
    'USD_2011',
    21600000,
]
additional_consequences.loc['replacement', 'Time'] = [
    0,
    '1 EA',
    'worker_day',
    12500,
]

additional_consequences

# %% [markdown]
"""
### Load component consequence data

Now that we have the consequence data completed and available for all
components in the damage sample, we can load the data to the loss
model in Pelicun.

When providing custom data, you can directly provide a DataFrame like
we do in this example (additional_consequences), or you can provide a
path to a CSV file that is structured like the table we prepared
above.

Default databases are loaded using the keyword "PelicunDefault" in the
path and then providing the name of the database. The PelicunDefault
keyword is automatically replaced with the path to the default
component data directory.

If there were identical components in the listed sources, Pelicun
always preserves the first occurrence of a component. Hence, always
start with the custom data when listing sources and add default
databases in the end.
"""

# %%
# Load the loss model to pelicun
assessment.loss.decision_variables = ('Cost', 'Time', 'Energy', 'Carbon')
assessment.loss.add_loss_map(loss_map)
assessment.loss.load_model_parameters(
    [additional_consequences, 'PelicunDefault/loss_repair_DB_FEMA_P58_2nd.csv'],
)

# %% [markdown]
"""
### Loss calculation

Loss calculation in Pelicun requires

- a pre-assigned set of component consequence functions;

- a pre-assigned sample of demands and/or damages;

- and a loss mapping matrix

The sample size for the loss calculation is automatically inferred
from the demand/damage sample size.
"""

# %%
# and run the calculations
assessment.bldg_repair.calculate()

# %% [markdown]
"""
### Loss estimates

**Repair cost of individual components and groups of components**

Below, we extract the loss sample from Pelicun and show a few example
plots to illustrate how rich information this data provides about the
repair consequences in the building
"""

# %%
loss_sample = assessment.bldg_repair.sample

print(
    'Size of repair cost & time results: ',
    sys.getsizeof(loss_sample) / 1024 / 1024,
    'MB',
)

# %%
loss_sample['Cost']['B.20.22.031'].groupby(level=[0, 2, 3], axis=1).sum().describe(
    [0.1, 0.5, 0.9]
).T

# %%
loss_plot = (
    loss_sample.groupby(level=['dv', 'dmg'], axis=1).sum()['Cost'].iloc[:, :-2]
)

# we add 100 to the loss values to avoid having issues with zeros when creating a log plot
loss_plot += 100

px.box(
    y=np.tile(loss_plot.columns, loss_plot.shape[0]),
    x=loss_plot.to_numpy().flatten(),
    color=[c[0] for c in loss_plot.columns] * loss_plot.shape[0],
    orientation='h',
    labels={
        'x': 'Aggregate repair cost [2011 USD]',
        'y': 'Component ID',
        'color': 'Component Group',
    },
    title='Range of repair cost realizations by component type',
    log_x=True,
    height=1500,
)

# %%
loss_plot = (
    loss_sample['Cost']
    .groupby('loc', axis=1)
    .sum()
    .describe([0.1, 0.5, 0.9])
    .iloc[:, 1:]
)

roof_level = 5
fig = px.pie(
    values=loss_plot.loc['mean'],
    names=[
        f'floor {c}' if int(c) < roof_level else 'roof' for c in loss_plot.columns
    ],
    title='Contribution of each floor to the average non-collapse repair costs',
    height=500,
    hole=0.4,
)

fig.update_traces(textinfo='percent+label')

# %%
loss_plot = loss_sample['Cost'].groupby(level=[1], axis=1).sum()

loss_plot['repairable'] = loss_plot.iloc[:, :-2].sum(axis=1)
loss_plot = loss_plot.iloc[:, -3:]

px.bar(
    x=loss_plot.columns,
    y=loss_plot.describe().loc['mean'],
    labels={'x': 'Damage scenario', 'y': 'Average repair cost'},
    title='Contribution to average losses from the three possible damage scenarios',
    height=400,
)

# %% [markdown]
"""
**Aggregate losses**

Aggregating losses for repair costs is straightforward, but repair
times are less trivial. Pelicun adopts the method from FEMA P-58 and
provides two bounding values for aggregate repair times:

- **parallel** assumes that repairs are conducted in parallel across
    locations. In each location, repairs are assumed to be
    sequential. This translates to aggregating component repair times
    by location and choosing the longest resulting aggregate value
    across locations.

- **sequential** assumes repairs are performed sequentially across
    locations and within each location. This translates to aggregating
    component repair times across the entire building.

The parallel option is considered a lower bound and the sequential is
an upper bound of the real repair time. Pelicun automatically
calculates both options for all (i.e., not only FEMA P-58) analyses.

"""

# %%
agg_df = assessment.bldg_repair.aggregate_losses()

agg_df.describe([0.1, 0.5, 0.9])

# %%
# filter only the repairable cases
fixed_replacement_cost = 2e7
agg_df_plot = agg_df.loc[agg_df['repair_cost'] < fixed_replacement_cost]

px.scatter(
    x=agg_df_plot['repair_time', 'sequential'],
    y=agg_df_plot['repair_time', 'parallel'],
    opacity=0.1,
    marginal_x='histogram',
    marginal_y='histogram',
    labels={
        'x': 'Sequential repair time [worker-days]',
        'y': 'Parallel repair time [worker-days]',
    },
    title='Two bounds of repair time conditioned on repairable damage',
    height=750,
    width=750,
)
