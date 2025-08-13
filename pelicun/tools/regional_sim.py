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

"""Temporary solution that provides regional simulation capability to Pelicun."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from pelicun.assessment import Assessment
from pelicun.auto import auto_populate
from pelicun.file_io import substitute_default_path
from pelicun.tools.NNR import NNR


def unique_list(x: pd.Series) -> str:
    """
    Return unique values in a pandas Series as a comma-separated string.

    Parameters
    ----------
    x : pd.Series
        pandas Series containing values to extract unique elements from

    Returns
    -------
    str
        Comma-separated string of unique values, or single value if only one unique value exists

    """
    vals = x.unique().astype(str)

    if len(vals) > 1:
        return f'{",".join(vals)}'

    return f'{vals[0]}'


def format_elapsed_time(start_time: float) -> str:
    """
    Format elapsed time from a start timestamp to current time as hh:mm:ss.

    Parameters
    ----------
    start_time : float
        Start time as a float timestamp (from time.time())

    Returns
    -------
    str
        Formatted elapsed time string in hh:mm:ss format

    """
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> contextlib.Generator[None, None, None]:
    """
    Context manager to patch joblib to report progress into a tqdm progress bar.

    This function temporarily replaces joblib's BatchCompletionCallBack to update
    the provided tqdm progress bar with batch completion information during
    parallel processing.

    Parameters
    ----------
    tqdm_object : tqdm
        tqdm progress bar object to update with progress information

    Yields
    ------
    None
        Context manager yields control to the calling code

    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):  # noqa: ANN204, ANN002, ANN003
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def process_buildings_chunk(
    bldg_df_chunk: pd.DataFrame,
    grid_points: pd.DataFrame,
    grid_data: pd.DataFrame,
    sample_size_demand: int,
    sample_size_damage: int,
    dl_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process a chunk of buildings through the complete regional simulation pipeline.

    This function performs event-to-building mapping, building-to-archetype mapping,
    damage calculation, and loss calculation for a subset of buildings. It uses
    nearest neighbor regression to map grid-based intensity measures to building
    locations, then applies Pelicun assessment methods to calculate damage and losses.

    Parameters
    ----------
    bldg_df_chunk : pd.DataFrame
        DataFrame containing building inventory data for this chunk,
        must include Longitude, Latitude, and building characteristics
    grid_points : pd.DataFrame
        DataFrame with grid point coordinates (Longitude, Latitude)
    grid_data : pd.DataFrame
        DataFrame with intensity measure data for each grid point
    sample_size_demand : int
        Number of demand realizations available
    sample_size_damage : int
        Number of damage realizations to generate
    dl_method : str
        Damage and loss methodology

    Returns
    -------
    tuple
        demand_sample : pd.DataFrame
            DataFrame with demand sample results (intensity measures)
        damage_df : pd.DataFrame
            DataFrame with damage state results for each component
        repair_costs : pd.DataFrame
            DataFrame with repair cost estimates
        repair_times : pd.DataFrame
            DataFrame with repair time estimates

    """
    # 3 Event-to-Building Mapping
    # Map event IMs to building centroids using the nearest neighbor method

    X = grid_points[['Longitude', 'Latitude']].to_numpy()  # noqa: N806
    Z = grid_data.T.to_numpy()  # noqa: N806

    X_t = bldg_df_chunk[['Longitude', 'Latitude']].to_numpy()  # noqa: N806

    Z_hat = NNR(X_t, X, Z, sample_size=-1, n_neighbors=8, weight='distance2')  # noqa: N806

    # Prepare IM information as demands in the Pelicun format

    # TODO: allow for more flexible columns  # noqa: TD002
    demand_sample = pd.DataFrame(
        np.vstack([Z_hat.T, np.full((1, Z_hat.shape[0]), 'g')]),
        columns=[f'PGA-{loc}-1' for loc in bldg_df_chunk.index],
        index=[*list(range(sample_size_demand)), 'Units'],
    )

    # 4 Building-to-Archetype Mapping
    # Use Pelicun and the built-in mapping for Hazus IM-based damage models to map each building in the inventory to an archetype

    auto_script_path = substitute_default_path(
        [f'PelicunDefault/{dl_method}/pelicun_config.py']
    )[0]

    CMP_list = []  # noqa: N806

    for bldg_id, row_data in bldg_df_chunk.iloc[:].iterrows():
        _, CMP = auto_populate(  # noqa: N806
            {
                'GeneralInformation': dict(row_data),
                'assetType': 'Buildings',
                'Applications': {
                    'DL': {
                        'ApplicationData': {
                            'coupled_EDP': True,
                            'lifeline_facility': True,
                            'ground_failure': False,
                        }
                    }
                },
            },
            auto_script_path,
        )

        CMP['Location'] = bldg_id

        CMP_list.append(CMP)

    cmp_marginals = pd.concat(CMP_list)

    cmp_marginals_raw = cmp_marginals.copy()

    cmp_marginals = cmp_marginals_raw.groupby(cmp_marginals_raw.index).agg(
        {
            'Units': unique_list,
            'Location': unique_list,
            'Direction': unique_list,
            'Theta_0': unique_list,
        }
    )

    # 5 Calculate Damage
    # Calculate damage with a deterministic inventory realization

    # initialize assessment object
    PAL = Assessment(  # noqa: N806
        {
            'LogFile': 'pelicun_log.txt',
            'Verbose': True,
            'NonDirectionalMultipliers': {'ALL': 1.0},
        }
    )

    # load demands
    PAL.demand.load_sample(demand_sample)

    PAL.demand.calibrate_model({'ALL': {'DistributionFamily': 'empirical'}})

    PAL.demand.generate_sample(
        {'SampleSize': sample_size_damage, 'PreserveRawOrder': True}
    )

    # load component assignment
    PAL.asset.load_cmp_model({'marginals': cmp_marginals})

    PAL.asset.generate_cmp_sample()

    # get the path to the built-in fragility functions
    component_db_path = substitute_default_path(
        [f'PelicunDefault/{dl_method}/fragility.csv']
    )[0]

    # import the required fragility functions
    cmp_set = PAL.asset.list_unique_component_ids()

    PAL.damage.load_model_parameters(
        [
            component_db_path,
        ],
        cmp_set,
    )

    # run the damage calculation
    PAL.damage.calculate()

    # retrieve damage information
    damage_sample, damage_units = PAL.damage.save_sample(save_units=True)

    damage_units = damage_units.to_frame().T

    # aggregate across uid
    # this is trivial since we don't have multiple identical components at the same location
    damage_units = damage_units.groupby(
        level=['cmp', 'loc', 'dir', 'ds'], axis=1
    ).first()

    damage_groupby_uid = damage_sample.groupby(
        level=['cmp', 'loc', 'dir', 'ds'], axis=1
    )
    damage_sample = damage_groupby_uid.sum().mask(
        damage_groupby_uid.count() == 0, np.nan
    )

    # aggregate across dir
    # also trivial, all results are in dir 1

    damage_groupby = damage_sample.groupby(level=['cmp', 'loc', 'ds'], axis=1)

    damage_units = damage_units.groupby(level=['cmp', 'loc', 'ds'], axis=1).first()
    grp_damage = damage_groupby.sum().mask(damage_groupby.count() == 0, np.nan)

    # replace non-zero values with 1
    # this is probably not making any meaningful changes since we have 1 ea quantity of each component
    # Honestly, I am not quite sure why we need this step...
    grp_damage = grp_damage.mask(grp_damage.astype(np.float64).to_numpy() > 0, 1)

    # get the corresponding DS for each column
    ds_list = grp_damage.columns.get_level_values('ds').astype(int)

    # replace ones with the corresponding DS in each cell
    grp_damage = grp_damage.mul(ds_list, axis=1)

    # aggregate across damage state indices
    damage_groupby_2 = grp_damage.groupby(level=['cmp', 'loc'], axis=1)

    # choose the max value
    # i.e., the governing DS for each comp-loc pair
    # Note that in each realization, each component will have only one non-zero damage state result,
    # so this is not picking the max damage state, but rather picking the realized damage state
    grp_damage = damage_groupby_2.max().mask(damage_groupby_2.count() == 0, np.nan)

    # aggregate units to the same format
    # assume identical units across locations for each comp
    damage_units = damage_units.groupby(level=['cmp', 'loc'], axis=1).first()

    grp_damage = grp_damage.astype(int).T.reorder_levels(['loc', 'cmp'])

    damage_df = grp_damage.copy()

    # assuming there's only one component in each building, we can drop the component info
    damage_df = damage_df.groupby(level='loc').sum()
    damage_df.index = damage_df.index.astype(int)
    damage_df = damage_df.sort_index()

    # 6 Calculate Losses

    # get the path to the built-in consequence functions
    consequence_db_path = substitute_default_path(
        [f'PelicunDefault/{dl_method}/consequence_repair.csv']
    )[0]

    # Hazus consequence functions depend only on occupancy class
    # we need to list building IDs for each occupancy type first
    loss_groups = bldg_df_chunk[['StructureType', 'OccupancyClass']].copy()
    loss_groups['IDs'] = loss_groups.index

    loss_groups = loss_groups.groupby(['OccupancyClass']).agg({'IDs': unique_list})

    # create a lookup table to find which fragility IDs are at which location
    dmg_sample = PAL.damage.save_sample()
    cmp_loc = dmg_sample.groupby(level=['cmp', 'loc'], axis=1).first()

    cmp_lookup = pd.Series(
        cmp_loc.columns.get_level_values('cmp'),
        index=cmp_loc.columns.get_level_values('loc').astype(int),
    ).sort_index()

    # Loss calculation is a bit complicated now. I need to add a new feature to Pelicun to make it work as smoothly as the damage does.
    # What I do below is brute force, but it works, and still takes only a few minutes to run.
    # I'll enhance Pelicun in the coming weeks to do more sophisticated loss mapping and be able to handle the following calculations faster without the for loop

    # we'll collect the results in this dict
    loss_results = {'Cost': [], 'Time': []}

    # start by extracting the full damage sample and preserving it
    full_dmg_sample = PAL.damage.save_sample()

    for occ_type, raw_building_ids in loss_groups.iloc[:].iterrows():
        # convert the building id list to a numpy array of ints
        building_ids = np.array(raw_building_ids.to_numpy()[0].split(',')).astype(
            int
        )

        # and make sure those IDs are in the component lookup table
        building_ids = [
            building_id
            for building_id in building_ids
            if building_id in cmp_lookup.index
        ]

        # load only the subset of the damage sample that is needed for the calculation
        idx = pd.IndexSlice
        dmg_subset = full_dmg_sample.loc[
            :, idx[:, np.array(building_ids, dtype=str), :, :, :]
        ]
        PAL.damage.load_sample(dmg_subset)

        # create a loss map that maps every fragility ID to a consequence for the given occupancy type
        loss_cmp = f'LF.{occ_type}'

        drivers = []
        loss_models = []

        dmg_cmps = dmg_subset.columns.get_level_values('cmp').unique()

        for cmp_id in dmg_cmps.unique():
            drivers.append(f'{cmp_id}')
            loss_models.append(loss_cmp)

        loss_map = pd.DataFrame(loss_models, columns=['Repair'], index=drivers)

        decision_variables = ['Cost', 'Time']

        PAL.loss.decision_variables = decision_variables
        PAL.loss.add_loss_map(loss_map, loss_map_policy=None)
        PAL.loss.load_model_parameters(
            [
                consequence_db_path,
            ]
        )

        PAL.loss.calculate()

        # - - - - - -

        repair_sample, repair_units = PAL.loss.save_sample(save_units=True)

        # aggregate across uid
        # this is trivial since we don't have multiple identical components at the same location
        repair_units = repair_units.groupby(
            level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir']
        ).first()

        repair_groupby_uid = repair_sample.groupby(
            level=['dv', 'loss', 'dmg', 'ds', 'loc', 'dir'], axis=1
        )

        repair_sample = repair_groupby_uid.sum().mask(
            repair_groupby_uid.count() == 0, np.nan
        )

        # now aggregate across loss, dmg, damage state, and direction
        # all of those are only one value per location, so they do not provide additional information
        repair_groupby = repair_sample.groupby(level=['dv', 'loc'], axis=1)

        repair_units = repair_units.groupby(level=['dv', 'loc']).first()

        grp_repair = repair_groupby.sum().mask(repair_groupby.count() == 0, np.nan)

        # - - - -

        # append the results to the main dict
        for DV_type in ['Cost', 'Time']:  # noqa: N806
            grp_repair_dv = grp_repair[DV_type].copy()
            grp_repair_dv.columns = grp_repair_dv.columns.astype(int)

            # we did not preserve the units for now; we can add them here if needed

            loss_results[DV_type].append(grp_repair_dv.loc[:, building_ids])

    repair_costs = pd.concat(loss_results['Cost'], axis=1).sort_index(axis=1).T
    repair_times = pd.concat(loss_results['Time'], axis=1).sort_index(axis=1).T

    # finish by putting the full damage sample back to the assessment object
    PAL.damage.load_sample(full_dmg_sample)

    return demand_sample, damage_df, repair_costs, repair_times


def process_and_save_chunk(
    i: int,
    chunk: pd.DataFrame,
    temp_dir: str,
    grid_points: pd.DataFrame,
    grid_data: pd.DataFrame,
    sample_size_demand: int,
    sample_size_damage: int,
    dl_method: str,
) -> None:
    """
    Process a single chunk of buildings and save results to temporary compressed CSV files.

    This function serves as a wrapper around process_buildings_chunk that handles
    the file I/O operations for parallel processing. It processes a chunk of buildings
    through the complete simulation pipeline and saves the results to temporary files
    for later aggregation.

    Parameters
    ----------
    i : int
        Chunk index number used for naming output files
    chunk : pd.DataFrame
        DataFrame containing building inventory data for this specific chunk
    temp_dir : str
        Path to temporary directory where results will be saved
    grid_points : pd.DataFrame
        DataFrame with grid point coordinates (Longitude, Latitude)
    grid_data : pd.DataFrame
        DataFrame with intensity measure data for each grid point
    sample_size_demand : int
        Number of demand realizations available
    sample_size_damage : int
        Number of damage realizations to generate
    dl_method : str
        Damage and loss methodology

    """
    # Process buildings through steps 3-6
    demand_sample_chunk, damage_df_chunk, repair_costs_chunk, repair_times_chunk = (
        process_buildings_chunk(
            chunk,
            grid_points,
            grid_data,
            sample_size_demand,
            sample_size_damage,
            dl_method,
        )
    )

    # Save each result DataFrame to compressed CSV files
    demand_sample_chunk.to_csv(f'{temp_dir}/demand_part_{i}.csv', compression='zip')
    damage_df_chunk.to_csv(f'{temp_dir}/damage_part_{i}.csv', compression='zip')
    repair_costs_chunk.to_csv(
        f'{temp_dir}/repair_costs_part_{i}.csv', compression='zip'
    )
    repair_times_chunk.to_csv(
        f'{temp_dir}/repair_times_part_{i}.csv', compression='zip'
    )


def regional_sim(config_file: str, num_cores: int | None = None) -> None:
    """
    Perform a regional-scale disaster impact simulation.

    This function orchestrates the complete regional simulation workflow including:
    1. Loading earthquake event data from gridded intensity measure files
    2. Loading building inventory data
    3. Mapping event intensity measures to building locations using nearest neighbor regression
    4. Mapping buildings to damage/loss archetypes using Pelicun auto-population
    5. Calculating damage states for all buildings
    6. Calculating repair costs and times
    7. Aggregating and saving results to CSV files

    The simulation is performed in parallel chunks to handle large building inventories
    efficiently. Results are saved as compressed CSV files for demand samples, damage
    states, repair costs, and repair times.

    Parameters
    ----------
    config_file : str
        Path to JSON configuration file containing simulation parameters,
        file paths, and analysis settings (inputRWHALE.json from SimCenter's R2D Tool)
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. If None,
        uses all available cores minus one

    Notes
    -----
    Output Files:
        - demand_sample.csv: Intensity measure realizations for all buildings
        - damage_sample.csv: Damage state realizations for all building components
        - repair_cost_sample.csv: Repair cost estimates for all buildings
        - repair_time_sample.csv: Repair time estimates for all buildings

    """
    batch_size = 1000

    # Initialize start time for timestamp tracking
    start_time = time.time()

    with Path(config_file).open(encoding='utf-8') as f:
        config = json.load(f)

    sample_size_demand = config['Applications']['RegionalMapping']['Buildings'][
        'ApplicationData'
    ]['samples']
    sample_size_damage = config['Applications']['DL']['Buildings'][
        'ApplicationData'
    ]['Realizations']

    # 1 Earthquake Event
    # Load gridded event IM information from a standard SimCenter EventGrid file and the corresponding site files.

    event_data_folder = config['RegionalEvent']['eventFilePath']
    event_grid_path = f"{event_data_folder}/{config['RegionalEvent']['eventFile']}"

    grid_points = pd.read_csv(event_grid_path)

    grid_point_data_array = []

    for grid_point_file in tqdm(
        grid_points['GP_file'],
        desc=f'[{format_elapsed_time(start_time)}] 1 Earthquake Event - Loading grid point data',
    ):
        grid_point_data = pd.read_csv(
            f'{event_data_folder}/{grid_point_file}', nrows=sample_size_demand
        )

        grid_point_data_array.append(grid_point_data)

    grid_data = pd.concat(grid_point_data_array, axis=1, keys=grid_points.index)

    # 2 Building Inventory
    # Load probabilistic building inventory from a CSV file
    print(f'[{format_elapsed_time(start_time)}] 2 Building Inventory')  # noqa: T201

    bldg_data_folder = config['Applications']['Assets']['Buildings'][
        'ApplicationData'
    ]['pathToSource']
    bldg_data_path = f"{bldg_data_folder}/{config['Applications']['Assets']['Buildings']['ApplicationData']['assetSourceFile']}"

    bldg_df = pd.read_csv(bldg_data_path, index_col=0)
    original_index = bldg_df.index
    bldg_df = bldg_df.sort_values(by='OccupancyClass')

    # bldg_df = bldg_df.iloc[:batch_size*5]

    # Get DL method for processing
    dl_method = config['Applications']['DL']['Buildings']['ApplicationData'][
        'DL_Method'
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Chunk the DataFrame into a list of smaller DataFrames
        bldg_chunks = [
            bldg_df.iloc[i : i + batch_size]
            for i in range(0, len(bldg_df), batch_size)
        ]

        # Determine the number of CPU cores to use
        if num_cores:
            n_jobs = num_cores
        else:
            n_jobs = max(1, os.cpu_count() - 1)

        # Process chunks in parallel with a proper progress bar
        with (
            tqdm(total=len(bldg_chunks), desc='Processing building chunks') as pbar,
            tqdm_joblib(pbar),
        ):
            Parallel(n_jobs=n_jobs)(
                delayed(process_and_save_chunk)(
                    i,
                    chunk,
                    temp_dir,
                    grid_points,
                    grid_data,
                    sample_size_demand,
                    sample_size_damage,
                    dl_method,
                )
                for i, chunk in enumerate(bldg_chunks)
            )

        # Read and combine all temporary files
        demand_list = []
        damage_list = []
        costs_list = []
        times_list = []

        for filename in os.listdir(temp_dir):
            filepath = Path(temp_dir) / filename
            if filename.startswith('demand_part_'):
                demand_list.append(
                    pd.read_csv(filepath, index_col=0, compression='zip')
                )
            elif filename.startswith('damage_part_'):
                damage_list.append(
                    pd.read_csv(filepath, index_col=0, compression='zip')
                )
            elif filename.startswith('repair_costs_part_'):
                costs_list.append(
                    pd.read_csv(filepath, index_col=0, compression='zip')
                )
            elif filename.startswith('repair_times_part_'):
                times_list.append(
                    pd.read_csv(filepath, index_col=0, compression='zip')
                )

        # Concatenate all parts into final DataFrames
        demand_sample = pd.concat(demand_list, axis=1)
        damage_df = pd.concat(damage_list, axis=0)
        repair_costs = pd.concat(costs_list, axis=0)
        repair_times = pd.concat(times_list, axis=0)

        # Restore original building order for all result files
        demand_sample = demand_sample.reindex(
            columns=[f'PGA-{loc}-1' for loc in original_index]
        )
        damage_df = damage_df.reindex(original_index)
        repair_costs = repair_costs.reindex(original_index)
        repair_times = repair_times.reindex(original_index)

        # 7 Save results
        print(f'[{format_elapsed_time(start_time)}] 7 Save results')  # noqa: T201

        demand_sample.to_csv('demand.csv')
        damage_df.to_csv('damage.csv')
        repair_costs.to_csv('repair_costs.csv')
        repair_times.to_csv('repair_times.csv')
