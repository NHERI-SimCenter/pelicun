"""
This test verifies backwards compatibility with the following:

    DesignSafe PRJ-3411 > Example01_FEMA_P58_Introduction

There are some changes made to the code and input files, so the output
is not the same with what the original code would produce. We only
want to confirm that executing this code does not raise an error.

"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pelicun.warnings import PelicunWarning
from pelicun.base import convert_to_MultiIndex
from pelicun.assessment import Assessment


def test_compatibility_DesignSafe_PRJ_3411_Example01():

    sample_size = 10000
    raw_demands = pd.read_csv(
        'pelicun/tests/compatibility/PRJ-3411v5/demand_data.csv', index_col=0
    )
    raw_demands = convert_to_MultiIndex(raw_demands, axis=0)
    raw_demands.index.names = ['stripe', 'type', 'loc', 'dir']
    stripe = '3'
    stripe_demands = raw_demands.loc[stripe, :]
    stripe_demands.insert(0, 'Units', "")
    stripe_demands.loc['PFA', 'Units'] = 'g'
    stripe_demands.loc['PID', 'Units'] = 'rad'
    stripe_demands.insert(1, 'Family', "")
    stripe_demands['Family'] = 'lognormal'
    stripe_demands.rename(columns={'median': 'Theta_0'}, inplace=True)
    stripe_demands.rename(columns={'log_std': 'Theta_1'}, inplace=True)
    ndims = stripe_demands.shape[0]
    demand_types = stripe_demands.index
    perfect_CORR = pd.DataFrame(
        np.ones((ndims, ndims)), columns=demand_types, index=demand_types
    )

    PAL = Assessment({"PrintLog": True, "Seed": 415})
    PAL.demand.load_model({'marginals': stripe_demands, 'correlation': perfect_CORR})
    PAL.demand.generate_sample({"SampleSize": sample_size})

    demand_sample = PAL.demand.save_sample()

    delta_y = 0.0075
    PID = demand_sample['PID']
    RID = PAL.demand.estimate_RID(PID, {'yield_drift': delta_y})
    demand_sample_ext = pd.concat([demand_sample, RID], axis=1)
    Sa_vals = [0.158, 0.387, 0.615, 0.843, 1.071, 1.299, 1.528, 1.756]
    demand_sample_ext[('SA_1.13', 0, 1)] = Sa_vals[int(stripe) - 1]
    demand_sample_ext.T.insert(0, 'Units', "")
    demand_sample_ext.loc['Units', ['PFA', 'SA_1.13']] = 'g'
    demand_sample_ext.loc['Units', ['PID', 'RID']] = 'rad'

    PAL.demand.load_sample(demand_sample_ext)

    cmp_marginals = pd.read_csv(
        'pelicun/tests/compatibility/PRJ-3411v5/CMP_marginals.csv', index_col=0
    )

    PAL.stories = 4
    PAL.asset.load_cmp_model({'marginals': cmp_marginals})
    PAL.asset.generate_cmp_sample()

    cmp_sample = PAL.asset.save_cmp_sample()
    assert cmp_sample is not None

    with pytest.warns(PelicunWarning):
        P58_data = PAL.get_default_data('fragility_DB_FEMA_P58_2nd')

    cmp_list = cmp_marginals.index.unique().values[:-3]
    P58_data_for_this_assessment = P58_data.loc[cmp_list, :].sort_values(
        'Incomplete', ascending=False
    )
    additional_fragility_db = P58_data_for_this_assessment.sort_index()

    P58_metadata = PAL.get_default_metadata('fragility_DB_FEMA_P58_2nd')
    assert P58_metadata is not None

    additional_fragility_db.loc[
        ['D.20.22.013a', 'D.20.22.023a', 'D.20.22.023b'],
        [('LS1', 'Theta_1'), ('LS2', 'Theta_1')],
    ] = 0.5
    additional_fragility_db.loc['D.20.31.013b', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.20.61.013b', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.30.31.013i', ('LS1', 'Theta_0')] = 1.5  # g
    additional_fragility_db.loc['D.30.31.013i', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.30.31.023i', ('LS1', 'Theta_0')] = 1.5  # g
    additional_fragility_db.loc['D.30.31.023i', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db.loc['D.30.52.013i', ('LS1', 'Theta_0')] = 1.5  # g
    additional_fragility_db.loc['D.30.52.013i', ('LS1', 'Theta_1')] = 0.5
    additional_fragility_db['Incomplete'] = 0

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
    additional_fragility_db.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
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
    additional_fragility_db['Incomplete'] = 0

    with pytest.warns(PelicunWarning):
        PAL.damage.load_damage_model(
            [additional_fragility_db, 'PelicunDefault/fragility_DB_FEMA_P58_2nd.csv']
        )

    # FEMA P58 uses the following process:
    dmg_process = {
        "1_collapse": {"DS1": "ALL_NA"},
        "2_excessiveRID": {"DS1": "irreparable_DS1"},
    }
    PAL.damage.calculate(dmg_process=dmg_process)

    damage_sample = PAL.damage.save_sample()
    assert damage_sample is not None

    drivers = [f'DMG-{cmp}' for cmp in cmp_marginals.index.unique()]
    drivers = drivers[:-3] + drivers[-2:]

    loss_models = cmp_marginals.index.unique().tolist()[:-3]
    loss_models += ['replacement'] * 2
    loss_map = pd.DataFrame(loss_models, columns=['BldgRepair'], index=drivers)

    with pytest.warns(PelicunWarning):
        P58_data = PAL.get_default_data('bldg_repair_DB_FEMA_P58_2nd')

    P58_data_for_this_assessment = P58_data.loc[
        loss_map['BldgRepair'].values[:-2], :
    ]

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
    additional_consequences.loc[('replacement', 'Cost')] = [
        0,
        '1 EA',
        'USD_2011',
        21600000,
    ]
    additional_consequences.loc[('replacement', 'Time')] = [
        0,
        '1 EA',
        'worker_day',
        12500,
    ]

    with pytest.warns(PelicunWarning):
        PAL.bldg_repair.load_model(
            [
                additional_consequences,
                "PelicunDefault/bldg_repair_DB_FEMA_P58_2nd.csv",
            ],
            loss_map,
        )

    PAL.bldg_repair.calculate()

    loss_sample = PAL.bldg_repair.sample
    assert loss_sample is not None

    with pytest.warns(PelicunWarning):
        agg_DF = PAL.bldg_repair.aggregate_losses()
    assert agg_DF is not None
