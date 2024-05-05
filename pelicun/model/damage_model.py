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
This file defines the DamageModel object and its methods.

.. rubric:: Contents

.. autosummary::

    DamageModel

"""

from typing import Callable
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pelicun.model.pelicun_model import PelicunModel
from pelicun import base
from pelicun import uq
from pelicun import file_io


idx = base.idx


class DamageModel(PelicunModel):
    """
    Manages damage information used in assessments.

    """

    def __init__(self, assessment):
        super().__init__(assessment)

        self.ds_model: DamageModel_DS = DamageModel_DS(assessment)
        self.missing_components: list[str] = []

    @property
    def damage_models(self):
        return (self.ds_model,)

    def load_model_parameters(self, data_paths):
        """
        Load damage model parameters.

        Parameters
        ----------
        data_paths: list of (string | DataFrame)
            List of paths to data or files with damage model
            information. Default XY datasets can be accessed as
            PelicunDefault/XY. Order matters. Parameters defined in
            prior elements in the list take precedence over the same
            parameters in subsequent data paths. I.e., place the
            Default datasets in the back.

        """

        self.log_div()
        self.log_msg('Loading damage model...', prepend_timestamp=False)

        # replace default flag with default data path
        data_paths = file_io.substitute_default_path(data_paths)

        #
        # load damage parameter data into the models
        #

        for data_path in data_paths:
            data = file_io.load_data(
                data_path, None, orientation=1, reindex=False, log=self._asmnt.log
            )
            # determine if the damage model parameters are for damage
            # states
            if _is_for_ds_model(data):
                self.ds_model._load_model_parameters(data)
            else:
                raise ValueError(f'Invalid damage model parameters: {data_path}')

        self.log_msg('Damage model parameters loaded successfully.', prepend_timestamp=False)

        #
        # remove items
        #

        self.log_msg('Removing unused damage model parameters.', prepend_timestamp=False)
        # get a list of unique component IDs
        cmp_set = self._asmnt.asset.list_unique_component_ids(as_set=True)

        for damage_model in self.damage_models:
            # drop unused damage parameter definitions
            damage_model._drop_unused_damage_parameters(cmp_set)
            # remove components with incomplete damage parameters
            damage_model._remove_incomplete_components()

        #
        # convert units
        #

        self.log_msg('Converting damage model parameter units.', prepend_timestamp=False)
        for damage_model in self.damage_models:
            damage_model._convert_damage_parameter_units()

        #
        # verify damage parameter availability
        #

        self.log_msg(
            'Checking damage model parameter '
            'availability for all components in the asset model.', prepend_timestamp=False
        )
        missing_components = self._ensure_damage_parameter_availability(cmp_set)

        self.missing_components = missing_components

    def calculate(
        self,
        sample_size=None,
        dmg_process=None,
        block_batch_size=1000,
        scaling_specification=None,
    ):
        """
        Calculate the damage of each component block.

        """
        self.log_div()
        self.log_msg('Calculating damages...')

        if not sample_size:
            # todo: Deprecation warning
            sample_size = self._asmnt.demand.sample.shape[0]

        # obtain damage states for applicable components
        ds_sample = self.ds_model._obtain_ds_sample(
            sample_size=sample_size,
            block_batch_size=block_batch_size,
            scaling_specification=scaling_specification,
            missing_components=self.missing_components,
        )

        # Apply the prescribed damage process, if any
        if dmg_process is not None:
            self.log_msg("Applying damage processes.")

            # Sort the damage processes tasks
            dmg_process = {key: dmg_process[key] for key in sorted(dmg_process)}

            # Perform damage tasks in the sorted order
            for task in dmg_process.items():
                self.ds_model._perform_dmg_task(task, ds_sample)

            self.log_msg(
                "Damage processes successfully applied.", prepend_timestamp=False
            )

        qnt_sample = self.ds_model._prepare_dmg_quantities(ds_sample, dropzero=False)

        # If requested, extend the quantity table with all possible DSs
        if self._asmnt.options.list_all_ds:
            qnt_sample = self._complete_ds_cols(qnt_sample)

        self.ds_model.sample = qnt_sample

        self.log_msg('Damage calculation completed.', prepend_timestamp=False)

    def _get_component_id_set(self):
        """
        Get a set of components for which damage parameters are
        available.
        """
        cmp_list = []
        if self.ds_model.damage_params is not None:
            cmp_list.extend(self.ds_model.damage_params.index.to_list())
        return set(cmp_list)

    def _ensure_damage_parameter_availability(self, cmp_list):
        """
        Makes sure that all components have damage parameters.

        Returns
        -------
        list
            List of component IDs with missing damage parameters.

        """

        available_components = self._get_component_id_set()

        missing_components = [
            component
            for component in cmp_list
            if component not in available_components
        ]

        if missing_components:
            self.log_msg(
                f"\n"
                f"WARNING: The damage model does not provide "
                f"damage information for the following component(s) "
                f"in the asset model: {missing_components}."
                f"\n",
                prepend_timestamp=False,
            )

        return missing_components


def _is_for_ds_model(data):
    """
    Determines if the specified damage model parameters are for
    components modeled with discrete Damage States (DS).
    """
    return 'LS1' in data.columns.get_level_values(0)


class DamageModel_Base(PelicunModel):
    """
    Base class for damage models

    """

    def __init__(self, assessment):
        super().__init__(assessment)

        self.damage_params = None
        self.sample = None

    def _load_model_parameters(self, data):
        """
        Load model parameters from a dataframe, extending those
        already available. Parameters already defined take precedence,
        i.e. redefinitions of parameters are ignored.

        Parameters
        ----------
        data: DataFrame
            Data with damage model information.

        """

        if self.damage_params is not None:
            data = pd.concat((self.damage_params, data), axis=0)

        # drop redefinitions of components
        data = data.groupby(data.index).first()

        # TODO: load defaults for Demand-Offset and Demand-Directional

        self.damage_params = data

    def _convert_damage_parameter_units(self):
        """
        Converts previously loaded damage parameters to base units.

        """
        if self.damage_params is None:
            return

        units = self.damage_params[('Demand', 'Unit')]
        for LS_i in self.damage_params.columns.unique(level=0):
            if LS_i.startswith('LS'):
                self.damage_params.loc[:, LS_i] = self._convert_marginal_params(
                    self.damage_params.loc[:, LS_i].copy(), units
                ).values

    def _remove_incomplete_components(self):
        """
        Removes components that have incomplete damage model
        definitions from the damage model parameters.

        """
        if self.damage_params is None:
            return

        if ('Incomplete', '') not in self.damage_params.columns:
            return

        cmp_incomplete_idx = self.damage_params.loc[
            self.damage_params[('Incomplete', '')] == 1
        ].index

        self.damage_params.drop(cmp_incomplete_idx, inplace=True)

        if len(cmp_incomplete_idx) > 0:
            self.log_msg(
                f"\n"
                f"WARNING: Damage model information is incomplete for "
                f"the following component(s) "
                f"{cmp_incomplete_idx.to_list()}. They "
                f"were removed from the analysis."
                f"\n",
                prepend_timestamp=False,
            )

    def _drop_unused_damage_parameters(self, cmp_list):
        """
        Removes damage parameter definitions for component IDs not
        present in the given list.

        Parameters
        ----------
        cmp_list: list
            List of component IDs to be preserved in the damage
            parameters.
        """

        if self.damage_params is None:
            return
        cmp_mask = self.damage_params.index.isin(cmp_list, level=0)
        self.damage_params = self.damage_params.iloc[cmp_mask, :]

    def _get_pg_batches(self, block_batch_size, missing_components):
        """
        Group performance groups into batches for efficient damage
        assessment.

        The method takes as input the block_batch_size, which
        specifies the maximum number of blocks per batch. The method
        first checks if performance groups have been defined in the
        cmp_marginal_params dataframe, and if so, it uses the 'Blocks'
        column as the performance group information. If performance
        groups have not been defined in cmp_marginal_params, the
        method uses the cmp_sample dataframe to define the performance
        groups, with each performance group having a single block.

        The method then checks if the performance groups are available
        in the damage parameters dataframe, and removes any
        performance groups that are not found in the damage
        parameters. The method then groups the performance groups
        based on the locations and directions of the components, and
        calculates the cumulative sum of the blocks for each
        group. The method then divides the performance groups into
        batches of size specified by block_batch_size and assigns a
        batch number to each group. Finally, the method groups the
        performance groups by batch number, component, location, and
        direction, and returns a dataframe that shows the number of
        blocks for each batch.

        Parameters
        ----------
        block_batch_size: int
            Maximum number of components in each batch.
        missing_components: list[str]
            Set of component IDs for which damage parameters are
            unavailable. These components are ignored.

        Returns
        -------
        DataFrame
            A DataFrame indexed by batch number, component identifier,
            location, direction, and unique ID, with a column
            indicating the number of blocks assigned to each
            batch. This dataframe facilitates the management and
            execution of damage assessment tasks by grouping
            components into manageable batches based on the specified
            block batch size.

        Raises
        ------
        Warning
            Logs a warning if any performance groups do not have
            corresponding damage model information and are therefore
            excluded from the analysis.
        """

        # Get the marginal parameters for the components from the
        # asset model
        cmp_marginals = self._asmnt.asset.cmp_marginal_params
        cmp_sample = self._asmnt.asset.cmp_sample

        # Initialize the batch dataframe
        pg_batch = None

        # Instantiate `pg_batch`
        if 'Blocks' in cmp_marginals.columns:
            # If a `Blocks` column is available, use `cmp_marginals`
            pg_batch = cmp_marginals['Blocks'].to_frame().astype('int64')
        else:
            # Otherwise asume 1.00 for the number of blocks and
            # initialize `pg_batch` using the columns of `cmp_sample`.
            pg_batch = pd.DataFrame(
                np.ones(cmp_sample.shape[1]),
                index=cmp_sample.columns,
                columns=['Blocks'],
                dtype='int64',
            )

        # A warning has already been issued for components with
        # missing damage parameters (in
        # `DamageModel._ensure_damage_parameter_availability`).
        pg_batch.drop(pd.Index(missing_components), inplace=True)

        # It is safe to simply disregard components that are not
        # present in the `damage_params` of *this* model, and let them
        # be handled by another damage model.
        available_components = self.damage_params.index.unique().to_list()
        pg_batch = pg_batch.loc[pd.IndexSlice[available_components, :, :, :], :]

        # Sum up the number of blocks for each performance group
        pg_batch = pg_batch.groupby(['loc', 'dir', 'cmp', 'uid']).sum()
        pg_batch.sort_index(axis=0, inplace=True)

        # Calculate cumulative sum of blocks
        pg_batch['CBlocks'] = np.cumsum(pg_batch['Blocks'].values.astype(int))
        pg_batch['Batch'] = 0

        # Group the performance groups into batches
        for batch_i in range(1, pg_batch.shape[0] + 1):
            # Find the mask for blocks that are less than the batch
            # size and greater than 0
            batch_mask = np.all(
                np.array(
                    [
                        pg_batch['CBlocks'] <= block_batch_size,
                        pg_batch['CBlocks'] > 0,
                    ]
                ),
                axis=0,
            )

            if np.sum(batch_mask) < 1:
                batch_mask = np.full(batch_mask.shape, False)
                batch_mask[np.where(pg_batch['CBlocks'] > 0)[0][0]] = True

            pg_batch.loc[batch_mask, 'Batch'] = batch_i

            # Decrement the cumulative block count by the max count in
            # the current batch
            pg_batch['CBlocks'] -= pg_batch.loc[
                pg_batch['Batch'] == batch_i, 'CBlocks'
            ].max()

            # If the maximum cumulative block count is 0, exit the
            # loop
            if pg_batch['CBlocks'].max() == 0:
                break

        # Group the performance groups by batch, component, location,
        # and direction, and keep only the number of blocks for each
        # group
        pg_batch = (
            pg_batch.groupby(['Batch', 'cmp', 'loc', 'dir', 'uid'])
            .sum()
            .loc[:, 'Blocks']
            .to_frame()
        )

        return pg_batch

    def _get_required_demand_type(self, PGB):
        """
        Returns the id of the demand needed to calculate damage to a
        component. We assume that a damage model sample is available.

        This method returns the demand type and its properties
        required to calculate the damage to a component. The
        properties include whether the demand is directional, the
        offset, and the type of the demand. The method takes as input
        a dataframe PGB that contains information about the component
        groups in the asset. For each component group PG in the PGB
        dataframe, the method retrieves the relevant damage parameters
        from the damage_params dataframe and parses the demand type
        into its properties. If the demand type has a subtype, the
        method splits it and adds the subtype to the demand type to
        form the EDP (engineering demand parameter) type. The method
        also considers the default offset for the demand type, if it
        is specified in the options attribute of the assessment, and
        adds the offset to the EDP. If the demand is directional, the
        direction is added to the EDP. The method collects all the
        unique EDPs for each component group and returns them as a
        dictionary where each key is an EDP and its value is a list of
        component groups that require that EDP.

        Parameters
        ----------
        `PGB`: pd.DataFrame
            A pandas DataFrame with the block information for
            each component

        Returns
        -------
        dict
            A dictionary of EDP requirements, where each key is the EDP
            string (e.g., "Peak Ground Acceleration-0-0"), and the
            corresponding value is a list of tuples (component_id,
            location, direction)

        """

        # Assign the damage_params attribute to a local variable `DP`
        DP = self.damage_params

        # Check if verbose logging is enabled in `self._asmnt.log`
        if self._asmnt.log.verbose:
            # If verbose logging is enabled, log a message indicating
            # that we are collecting demand information
            self.log_msg(
                'Collecting required demand information...', prepend_timestamp=True
            )

        # Initialize an empty dictionary to store the unique EDP
        # requirements
        EDP_req = {}

        # Iterate over the index of the `PGB` DataFrame
        for PG in PGB.index:
            # Get the component name from the first element of the
            # `PG` tuple
            cmp = PG[0]

            # Get the directional, offset, and demand_type parameters
            # from the `DP` DataFrame
            directional, offset, demand_type = DP.loc[
                cmp,
                [
                    ('Demand', 'Directional'),
                    ('Demand', 'Offset'),
                    ('Demand', 'Type'),
                ],
            ]

            # Parse the demand type

            # Check if there is a subtype included in the demand_type
            # string
            if '|' in demand_type:
                # If there is a subtype, split the demand_type string
                # on the '|' character
                demand_type, subtype = demand_type.split('|')
                # Convert the demand type to the corresponding EDP
                # type using `base.EDP_to_demand_type`
                demand_type = base.EDP_to_demand_type[demand_type]
                # Concatenate the demand type and subtype to form the
                # EDP type
                EDP_type = f'{demand_type}_{subtype}'
            else:
                # If there is no subtype, convert the demand type to
                # the corresponding EDP type using
                # `base.EDP_to_demand_type`
                demand_type = base.EDP_to_demand_type[demand_type]
                # Assign the EDP type to be equal to the demand type
                EDP_type = demand_type

            # Consider the default offset, if needed
            if demand_type in self._asmnt.options.demand_offset.keys():
                # If the demand type has a default offset in
                # `self._asmnt.options.demand_offset`, add the offset
                # to the default offset
                offset = int(offset + self._asmnt.options.demand_offset[demand_type])
            else:
                # If the demand type does not have a default offset in
                # `self._asmnt.options.demand_offset`, convert the
                # offset to an integer
                offset = int(offset)

            # Determine the direction
            if directional:
                # If the demand is directional, use the third element
                # of the `PG` tuple as the direction
                direction = PG[2]
            else:
                # If the demand is not directional, use '0' as the
                # direction
                direction = '0'

            # Concatenate the EDP type, offset, and direction to form
            # the EDP key
            EDP = f"{EDP_type}-{str(int(PG[1]) + offset)}-{direction}"

            # If the EDP key is not already in the `EDP_req`
            # dictionary, add it and initialize it with an empty list
            if EDP not in EDP_req:
                EDP_req.update({EDP: []})

            # Add the current PG (performance group) to the list of
            # PGs associated with the current EDP key
            EDP_req[EDP].append(PG)

        # Return the unique EDP requirements
        return EDP_req

    def _assemble_required_demand_data(self, EDP_req):
        """
        Assembles demand data for damage state determination.

        The method takes the maximum of all available directions for
        non-directional demand, scaling it using the non-directional
        multiplier specified in self._asmnt.options, and returning the
        result as a dictionary with keys in the format of
        '<demand_type>-<location>-<direction>' and values as arrays of
        demand values. If demand data is not found, logs a warning
        message and skips the corresponding damages calculation.

        Parameters
        ----------
        EDP_req : dict
            A dictionary of unique EDP requirements

        Returns
        -------
        demand_dict : dict
            A dictionary of assembled demand data for calculation

        Raises
        ------
        KeyError
            If demand data for a given EDP cannot be found

        """

        if self._asmnt.log.verbose:
            self.log_msg(
                'Assembling demand data for calculation...', prepend_timestamp=True
            )

        demand_source = self._asmnt.demand.sample

        demand_dict = {}

        for EDP in EDP_req.keys():
            EDP = EDP.split('-')

            # if non-directional demand is requested...
            if EDP[2] == '0':
                # assume that the demand at the given location is available
                try:
                    # take the maximum of all available directions and scale it
                    # using the nondirectional multiplier specified in the
                    # self._asmnt.options (the default value is 1.2)
                    demand = (
                        demand_source.loc[:, (EDP[0], EDP[1])].max(axis=1).values
                    )
                    demand = demand * self._asmnt.options.nondir_multi(EDP[0])

                except KeyError:
                    demand = None

            else:
                demand = demand_source[(EDP[0], EDP[1], EDP[2])].values

            if demand is None:
                self.log_msg(
                    f'\nWARNING: Cannot find demand data for {EDP}. The '
                    'corresponding damages cannot be calculated.',
                    prepend_timestamp=False,
                )
            else:
                demand_dict.update({f'{EDP[0]}-{EDP[1]}-{EDP[2]}': demand})

        return demand_dict


class DamageModel_DS(DamageModel_Base):
    """
    Damage model for components that have discrete Damage States (DS).

    """

    def __init__(self, assessment):
        super().__init__(assessment)

    def save_sample(self, filepath=None, save_units=False):
        """
        Saves the damage sample data to a CSV file or returns it
        directly with an option to include units.

        This function handles saving the sample data of damage
        assessments to a specified file path or, if no path is
        provided, returns the data as a DataFrame. The function can
        optionally include a row for unit information when returning
        data.

        Parameters
        ----------
        filepath : str, optional
            The path to the file where the damage sample should be
            saved. If not provided, the sample is not saved to disk
            but returned.
        save_units : bool, default: False
            Indicates whether to include a row with unit information
            in the returned DataFrame. This parameter is ignored if a
            file path is provided.

        Returns
        -------
        None or tuple
            If `filepath` is provided, the function returns None after
            saving the data.
            If no `filepath` is specified, returns:
            - DataFrame containing the damage sample.
            - Optionally, a Series containing the units for each
              column if `save_units` is True.
        """
        self.log_div()
        self.log_msg('Saving damage sample...')

        cmp_units = self._asmnt.asset.cmp_units
        qnt_units = pd.Series(
            index=self.sample.columns, name='Units', dtype='object'
        )
        for cmp in cmp_units.index:
            qnt_units.loc[cmp] = cmp_units.loc[cmp]

        res = file_io.save_to_csv(
            self.sample,
            filepath,
            units=qnt_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )

        if filepath is not None:
            self.log_msg(
                'Damage sample successfully saved.', prepend_timestamp=False
            )
            return None

        # else:
        units = res.loc["Units"]
        res.drop("Units", inplace=True)
        res.index = res.index.astype('int64')

        if save_units:
            return res.astype(float), units

        return res.astype(float)

    def load_sample(self, filepath):
        """
        Load damage state sample data.

        """
        self.log_div()
        self.log_msg('Loading damage sample...')

        self.sample = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors, log=self._asmnt.log
        )

        # set the names of the columns
        self.sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'ds']

        self.log_msg('Damage sample successfully loaded.', prepend_timestamp=False)

    def _obtain_ds_sample(
        self,
        sample_size,
        block_batch_size,
        scaling_specification,
        missing_components,
    ):
        """
        Obtain the damage state of each performance group in the
        model.
        """

        # Break up damage calculation and perform it by performance group.
        # Compared to the simultaneous calculation of all PGs, this approach
        # reduces demands on memory and increases the load on CPU. This leads
        # to a more balanced workload on most machines for typical problems.
        # It also allows for a straightforward extension with parallel
        # computing.

        # get the list of performance groups
        self.log_msg(
            f'Number of Performance Groups in Asset Model:'
            f' {self._asmnt.asset.cmp_sample.shape[1]}',
            prepend_timestamp=False,
        )

        pg_batch = self._get_pg_batches(block_batch_size, missing_components)
        batches = pg_batch.index.get_level_values(0).unique()

        self.log_msg(
            f'Number of Component Blocks: {pg_batch["Blocks"].sum()}',
            prepend_timestamp=False,
        )

        self.log_msg(
            f"{len(batches)} batches of Performance Groups prepared "
            "for damage assessment",
            prepend_timestamp=False,
        )

        # for PG_i in self._asmnt.asset.cmp_sample.columns:
        ds_samples = []
        for PGB_i in batches:

            performance_group = pg_batch.loc[PGB_i]

            self.log_msg(
                f"Calculating damage states for PG batch {PGB_i} with "
                f"{int(performance_group['Blocks'].sum())} blocks"
            )

            # Generate an array with component capacities for each block and
            # generate a second array that assigns a specific damage state to
            # each component limit state. The latter is primarily needed to
            # handle limit states with multiple, mutually exclusive DS options
            capacity_sample, lsds_sample = self._generate_dmg_sample(
                sample_size, performance_group, scaling_specification
            )

            # Get the required demand types for the analysis
            EDP_req = self._get_required_demand_type(performance_group)

            # Create the demand vector
            demand_dict = self._assemble_required_demand_data(EDP_req)

            # Evaluate the Damage State of each Component Block
            ds_sample = self._evaluate_damage_state(
                demand_dict, EDP_req, capacity_sample, lsds_sample
            )

            ds_samples.append(ds_sample)

        ds_sample = pd.concat(ds_samples, axis=1)

        self.log_msg("Damage state calculation successful.", prepend_timestamp=False)

        return ds_sample

    def _handle_operation(self, initial_value, operation, other_value):
        """
        This method is used in `_create_dmg_RVs` to apply capacity
        adjustment operations whenever required. It is defined as a
        safer alternative to directly using `eval`.

        Parameters
        ----------
        initial_value: float
          Value before operation
        operation: str
          Any of +, -, *, /
        other_value: float
          Value used to apply the operation

        Returns
        -------
        float
          The result of the operation

        Raises
        ------
        ValueError
            If the operation is invalid.

        """
        if operation == '+':
            return initial_value + other_value
        if operation == '-':
            return initial_value - other_value
        if operation == '*':
            return initial_value * other_value
        if operation == '/':
            return initial_value / other_value
        raise ValueError(f'Invalid operation: {operation}')

    def _create_dmg_RVs(self, PGB, scaling_specification=None):
        """
        Creates random variables required later for the damage calculation.

        The method initializes two random variable registries,
        capacity_RV_reg and lsds_RV_reg, and loops through each
        performance group in the input performance group batch (PGB)
        dataframe. For each performance group, it retrieves the
        component sample and blocks and checks if the limit state is
        defined for the component. If the limit state is defined, the
        method gets the list of limit states and the parameters for
        each limit state. The method assigns correlation between limit
        state random variables, adds the limit state random variables
        to the capacity_RV_reg registry, and adds LSDS assignments to
        the lsds_RV_reg registry. After looping through all
        performance groups, the method returns the two registries.

        Parameters
        ----------
        PGB : DataFrame
            A DataFrame that groups performance groups into batches
            for efficient damage assessment.
        scaling_specification: dict, optional
            A dictionary defining the shift in median.
            Example: {'CMP-1-1': '*1.2', 'CMP-1-2': '/1.4'}
            The keys are individual components that should be present
            in the `capacity_sample`.  The values should be strings
            containing an operation followed by the value formatted as
            a float.  The operation can be '+' for addition, '-' for
            subtraction, '*' for multiplication, and '/' for division.

        Returns
        -------
        tuple
            A tuple containing two RandomVariableRegistry instances:
            one for the capacity random variables and one for the LSDS
            assignments.

        Raises
        ------
        ValueError
            Raises an error if the scaling specification is invalid or
            if the input DataFrame does not meet the expected format.
            Also, raises errors if there are any issues with the types
            or formats of the data in the input DataFrame.
        """

        def assign_lsds(ds_weights, ds_id, lsds_RV_reg, lsds_rv_tag):
            """
            Assigns limit states to damage states using random
            variables, updating the provided random variable registry.
            This function either creates a deterministic random
            variable for a single damage state or a multinomial random
            variable for multiple damage states.

            Parameters
            ----------
            ds_weights : str or None
                A string representing the weights of different damage
                states associated with a limit state, separated by
                '|'.  If None, indicates that there is only one damage
                state associated with the limit state.
            ds_id : int
                The starting index for damage state IDs. This ID helps
                in mapping damage states to limit states.
            lsds_RV_reg : RandomVariableRegistry
                The registry where the newly created random variables
                (for mapping limit states to damage states) will be
                added.
            lsds_rv_tag : str
                A unique identifier for the random variable being
                created, typically including identifiers for
                component, location, direction, and limit state.

            Returns
            -------
            int
                The updated damage state ID, incremented based on the
                number of damage states handled in this call.

            Notes
            -----
            This function supports detailed control over the mapping
            from limit states to damage states within a stochastic
            framework, enhancing the flexibility and accuracy of
            probabilistic damage assessments. It dynamically adjusts
            to the number of damage states specified and applies a
            mapping function to correctly assign state IDs.
            """
            # If the limit state has a single damage state assigned
            # to it, we don't need random sampling
            if pd.isnull(ds_weights):
                ds_id += 1

                lsds_RV_reg.add_RV(
                    uq.DeterministicRandomVariable(
                        name=lsds_rv_tag,
                        theta=ds_id,
                    )
                )

            # Otherwise, we create a multinomial random variable
            else:
                # parse the DS weights
                ds_weights = np.array(
                    ds_weights.replace(" ", "").split('|'), dtype=float
                )

                def map_ds(values, offset=int(ds_id + 1)):
                    """
                    Maps an array of damage state indices to their
                    corresponding actual state IDs by applying an
                    offset.

                    Parameters
                    ----------
                    values : array-like
                        An array of indices representing damage
                        states. These indices are typically sequential
                        integers starting from zero.
                    offset : int
                        The value to be added to each element in
                        `values` to obtain the actual damage state
                        IDs.

                    Returns
                    -------
                    array
                        An array where each original index from
                        `values` has been incremented by `offset` to
                        reflect its actual damage state ID.
                    """
                    return values + offset

                lsds_RV_reg.add_RV(
                    uq.MultinomialRandomVariable(
                        name=lsds_rv_tag,
                        theta=ds_weights,
                        f_map=map_ds,
                    )
                )

                ds_id += len(ds_weights)

            return ds_id

        if self._asmnt.log.verbose:
            self.log_msg('Generating capacity variables ...', prepend_timestamp=True)

        # initialize the registry
        capacity_RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)
        lsds_RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        # capacity adjustment:
        # ensure the scaling_specification is a dictionary
        if not scaling_specification:
            scaling_specification = {}
        else:
            # if there are contents, ensure they are valid.
            # See docstring for an example of what is expected.
            parsed_scaling_specification = {}
            # validate contents
            for key, value in scaling_specification.items():
                css = 'capacity adjustment specification'
                if not isinstance(value, str):
                    raise ValueError(
                        f'Invalud entry in {css}: {value}. It has to be a string. '
                        f'See docstring of DamageModel._create_dmg_RVs.'
                    )
                capacity_adjustment_operation = value[0]
                number = value[1::]
                if capacity_adjustment_operation not in ('+', '-', '*', '/'):
                    raise ValueError(
                        f'Invalid operation in {css}: '
                        f'{capacity_adjustment_operation}'
                    )
                fnumber = base.float_or_None(number)
                if fnumber is None:
                    raise ValueError(f'Invalid number in {css}: {number}')
                parsed_scaling_specification[key] = (
                    capacity_adjustment_operation,
                    fnumber,
                )
                scaling_specification = parsed_scaling_specification

        # get the component sample and blocks from the asset model
        for PG in PGB.index:
            # determine demand capacity adjustment operation, if required
            cmp_loc_dir = '-'.join(PG[0:3])
            capacity_adjustment_operation = scaling_specification.get(
                cmp_loc_dir, None
            )

            cmp_id = PG[0]
            blocks = PGB.loc[PG, 'Blocks']

            # Calculate the block weights
            blocks = np.full(int(blocks), 1.0 / blocks)

            # initialize the damaged quantity sample variable
            assert self.damage_params is not None
            if cmp_id in self.damage_params.index:
                frg_params = self.damage_params.loc[cmp_id, :]

                # get the list of limit states
                limit_states = []

                for val in frg_params.index.get_level_values(0).unique():
                    if 'LS' in val:
                        limit_states.append(val[2:])

                ds_id = 0

                frg_rv_set_tags = [[] for b in blocks]
                anchor_RVs = []

                for ls_id in limit_states:
                    frg_params_LS = frg_params[f'LS{ls_id}']

                    theta_0 = frg_params_LS.get('Theta_0', np.nan)
                    family = frg_params_LS.get('Family', 'deterministic')
                    ds_weights = frg_params_LS.get('DamageStateWeights', np.nan)

                    # check if the limit state is defined for the component
                    if pd.isna(theta_0):
                        continue

                    theta = [
                        frg_params_LS.get(f"Theta_{t_i}", np.nan) for t_i in range(3)
                    ]

                    if capacity_adjustment_operation:
                        if family in {'normal', 'lognormal'}:
                            theta[0] = self._handle_operation(
                                theta[0],
                                capacity_adjustment_operation[0],
                                capacity_adjustment_operation[1],
                            )
                        else:
                            self.log_msg(
                                f'\nWARNING: Capacity adjustment is only supported '
                                f'for `normal` or `lognormal` distributions. '
                                f'Ignoring: {cmp_loc_dir}, which is {family}',
                                prepend_timestamp=False,
                            )

                    tr_lims = [
                        frg_params_LS.get(f"Truncate{side}", np.nan)
                        for side in ("Lower", "Upper")
                    ]

                    for block_i, _ in enumerate(blocks):
                        frg_rv_tag = (
                            'FRG-'
                            f'{PG[0]}-'  # cmp_id
                            f'{PG[1]}-'  # loc
                            f'{PG[2]}-'  # dir
                            f'{PG[3]}-'  # uid
                            f'{block_i+1}-'  # block
                            f'{ls_id}'
                        )

                        # Assign correlation between limit state random
                        # variables
                        # Note that we assume perfectly correlated limit
                        # state random variables here. This approach is in
                        # line with how mainstream PBE calculations are
                        # performed. Assigning more sophisticated
                        # correlations between limit state RVs is possible,
                        # if needed. Please let us know through the
                        # SimCenter Message Board if you are interested in
                        # such a feature.
                        # Anchor all other limit state random variables to
                        # the first one to consider the perfect correlation
                        # between capacities in each LS
                        if ls_id == limit_states[0]:
                            anchor = None
                        else:
                            anchor = anchor_RVs[block_i]

                        # parse theta values for multilinear_CDF
                        if family == 'multilinear_CDF':
                            theta = np.column_stack(
                                (
                                    np.array(
                                        theta[0].split('|')[0].split(','),
                                        dtype=float,
                                    ),
                                    np.array(
                                        theta[0].split('|')[1].split(','),
                                        dtype=float,
                                    ),
                                )
                            )

                        RV = uq.rv_class_map(family)(
                            name=frg_rv_tag,
                            theta=theta,
                            truncation_limits=tr_lims,
                            anchor=anchor,
                        )

                        capacity_RV_reg.add_RV(RV)

                        # add the RV to the set of correlated variables
                        frg_rv_set_tags[block_i].append(frg_rv_tag)

                        if ls_id == limit_states[0]:
                            anchor_RVs.append(RV)

                        # Now add the LS->DS assignments
                        lsds_rv_tag = (
                            'LSDS-'
                            f'{PG[0]}-'  # cmp_id
                            f'{PG[1]}-'  # loc
                            f'{PG[2]}-'  # dir
                            f'{PG[3]}-'  # uid
                            f'{block_i+1}-'  # block
                            f'{ls_id}'
                        )

                        ds_id_next = assign_lsds(
                            ds_weights, ds_id, lsds_RV_reg, lsds_rv_tag
                        )

                    ds_id = ds_id_next

        if self._asmnt.log.verbose:
            rv_count = len(lsds_RV_reg.RV)
            self.log_msg(
                f"2x{rv_count} random variables created.", prepend_timestamp=False
            )

        return capacity_RV_reg, lsds_RV_reg

    def _generate_dmg_sample(self, sample_size, PGB, scaling_specification=None):
        """
        This method generates a damage sample by creating random
        variables (RVs) for capacities and limit-state-damage-states
        (lsds), and then sampling from these RVs. The sample size and
        performance group batches (PGB) are specified as inputs. The
        method returns the capacity sample and the lsds sample.

        Parameters
        ----------
        sample_size : int
            The number of realizations to generate.
        PGB : DataFrame
            A DataFrame that groups performance groups into batches
            for efficient damage assessment.
        scaling_specification: dict, optional
            A dictionary defining the shift in median.
            Example: {'CMP-1-1': '*1.2', 'CMP-1-2': '/1.4'}
            The keys are individual components that should be present
            in the `capacity_sample`.  The values should be strings
            containing an operation followed by the value formatted as
            a float.  The operation can be '+' for addition, '-' for
            subtraction, '*' for multiplication, and '/' for division.

        Returns
        -------
        capacity_sample : DataFrame
            A DataFrame that represents the capacity sample.
        lsds_sample : DataFrame
            A DataFrame that represents the .

        Raises
        ------
        ValueError
            If the damage parameters have not been specified.

        """

        # Check if damage model parameters have been specified
        if self.damage_params is None:
            raise ValueError(
                'Damage model parameters have not been specified. '
                'Load parameters from the default damage model '
                'databases or provide your own damage model '
                'definitions before generating a sample.'
            )

        # Create capacity and LSD RVs for each performance group
        capacity_RVs, lsds_RVs = self._create_dmg_RVs(PGB, scaling_specification)

        if self._asmnt.log.verbose:
            self.log_msg('Sampling capacities...', prepend_timestamp=True)

        # Generate samples for capacity RVs
        capacity_RVs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        # Generate samples for LSD RVs
        lsds_RVs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        if self._asmnt.log.verbose:
            self.log_msg("Raw samples are available", prepend_timestamp=True)

        # get the capacity and lsds samples
        capacity_sample = (
            pd.DataFrame(capacity_RVs.RV_sample)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        capacity_sample = base.convert_to_MultiIndex(capacity_sample, axis=1)['FRG']
        capacity_sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ls']

        lsds_sample = (
            pd.DataFrame(lsds_RVs.RV_sample)
            .sort_index(axis=0)
            .sort_index(axis=1)
            .astype(int)
        )
        lsds_sample = base.convert_to_MultiIndex(lsds_sample, axis=1)['LSDS']
        lsds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ls']

        if self._asmnt.log.verbose:
            self.log_msg(
                f"Successfully generated {sample_size} realizations.",
                prepend_timestamp=True,
            )

        return capacity_sample, lsds_sample

    def _evaluate_damage_state(
        self, demand_dict, EDP_req, capacity_sample, lsds_sample
    ):
        """
        Use the demand and LS capacity sample to evaluate damage states

        Parameters
        ----------
        demand_dict: dict
            Dictionary containing the demand of each demand type.
        EDP_req: dict
            Dictionary containing the EDPs assigned to each demand
            type.
        capacity_sample: DataFrame
            Provides a sample of the capacity.
        lsds_sample: DataFrame
            Provides the mapping between limit states and damage
            states.

        Returns
        -------
        DataFrame
            Assigns a Damage State to each component block in the
            asset model.
        """

        if self._asmnt.log.verbose:
            self.log_msg('Evaluating damage states...', prepend_timestamp=True)

        # Create an empty dataframe with columns and index taken from
        # the input capacity sample
        dmg_eval = pd.DataFrame(
            columns=capacity_sample.columns, index=capacity_sample.index
        )

        # Initialize an empty list to store demand data
        demand_df = []

        # For each demand type in the demand dictionary
        for demand_name, demand_vals in demand_dict.items():
            # Get the list of PGs assigned to this demand type
            PG_list = EDP_req[demand_name]

            # Create a list of columns for the demand data
            # corresponding to each PG in the PG_list
            PG_cols = pd.concat(
                [dmg_eval.loc[:1, PG_i] for PG_i in PG_list], axis=1, keys=PG_list
            ).columns
            PG_cols.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ls']
            # Create a dataframe with demand values repeated for the
            # number of PGs and assign the columns as PG_cols
            demand_df.append(
                pd.concat(
                    [pd.Series(demand_vals)] * len(PG_cols), axis=1, keys=PG_cols
                )
            )

        # Concatenate all demand dataframes into a single dataframe
        demand_df = pd.concat(demand_df, axis=1)
        # Sort the columns of the demand dataframe
        demand_df.sort_index(axis=1, inplace=True)

        # Evaluate the damage exceedance by subtracting demand from
        # capacity and checking if the result is less than zero
        dmg_eval = (capacity_sample - demand_df) < 0

        # Remove any columns with NaN values from the damage
        # exceedance dataframe
        dmg_eval.dropna(axis=1, inplace=True)

        # initialize the DataFrames that store the damage states and
        # quantities
        ds_sample = pd.DataFrame(
            0,  # fill value
            columns=capacity_sample.columns.droplevel('ls').unique(),
            index=capacity_sample.index,
            dtype='int32',
        )

        # get a list of limit state ids among all components in the damage model
        ls_list = dmg_eval.columns.get_level_values(5).unique()

        # for each consecutive limit state...
        for LS_id in ls_list:
            # get all cmp - loc - dir - block where this limit state occurs
            dmg_e_ls = dmg_eval.loc[:, idx[:, :, :, :, :, LS_id]].dropna(axis=1)

            # Get the damage states corresponding to this limit state in each
            # block
            # Note that limit states with a set of mutually exclusive damage
            # states options have their damage state picked here.
            lsds = lsds_sample.loc[:, dmg_e_ls.columns]

            # Drop the limit state level from the columns to make the damage
            # exceedance DataFrame compatible with the other DataFrames in the
            # following steps
            dmg_e_ls.columns = dmg_e_ls.columns.droplevel(5)

            # Same thing for the lsds DataFrame
            lsds.columns = dmg_e_ls.columns

            # Update the damage state in the result with the values from the
            # lsds DF if the limit state was exceeded according to the
            # dmg_e_ls DF.
            # This one-liner updates the given Limit State exceedance in the
            # entire damage model. If subsequent Limit States are also exceeded,
            # those cells in the result matrix will get overwritten by higher
            # damage states.
            ds_sample.loc[:, dmg_e_ls.columns] = ds_sample.loc[
                :, dmg_e_ls.columns
            ].mask(dmg_e_ls, lsds)

        return ds_sample

    def _prepare_dmg_quantities(self, damage_state_sample, dropzero=True):
        """
        Combine component quantity and damage state information in one
        DataFrame.

        This method assumes that a component quantity sample is
        available in the asset model and a damage state sample is
        available in the damage model.

        Parameters
        ----------
        damage_state_sample: DataFrame
            A DataFrame that assigns a damage state to each component
            block in the asset model.
        dropzero: bool, optional, default: True
            If True, the quantity of non-damaged components is not
            saved.

        Returns
        -------
        DataFrame
            A DataFrame that combines the component quantity and
            damage state information.

        """

        # pylint: disable=missing-return-doc
        if self._asmnt.log.verbose:
            self.log_msg('Calculating damage quantities...', prepend_timestamp=True)

        # Retrieve the component quantity information and component
        # marginal parameters from the asset model

        # ('cmp', 'loc', 'dir', 'uid') -> component quantity series
        component_quantities = self._asmnt.asset.cmp_sample.to_dict('series')
        component_marginal_parameters = self._asmnt.asset.cmp_marginal_params

        if (component_marginal_parameters is not None) and (
            'Blocks' in component_marginal_parameters.columns
        ):
            # if this information is available, use it

            # ('cmp', 'loc', 'dir', 'uid) -> number of blocks
            num_blocks = component_marginal_parameters['Blocks'].to_dict()

            def get_num_blocks(key):
                # pylint: disable=missing-return-type-doc
                return float(num_blocks[key])

        else:
            # otherwise assume 1 block regardless of
            # ('cmp', 'loc', 'dir', 'uid) key
            def get_num_blocks(_):
                # pylint: disable=missing-return-type-doc
                return 1.00

        # ('cmp', 'loc', 'dir', 'uid', 'block') -> damage state series
        damage_state_sample_dict = damage_state_sample.to_dict('series')

        dmg_qnt_series_collection = {}
        for key, damage_state_series in damage_state_sample_dict.items():
            component, location, direction, uid, block = key
            damage_state_set = set(damage_state_series.values)
            for ds in damage_state_set:
                if ds == -1:
                    continue
                if dropzero and ds == 0:
                    continue
                dmg_qnt_vals = np.where(
                    damage_state_series.values == ds,
                    component_quantities[component, location, direction, uid].values
                    / get_num_blocks((component, location, direction, uid)),
                    0.00,
                )
                if -1 in damage_state_set:
                    dmg_qnt_vals = np.where(
                        damage_state_series.values != -1, dmg_qnt_vals, np.nan
                    )
                dmg_qnt_series = pd.Series(dmg_qnt_vals)
                dmg_qnt_series_collection[
                    (component, location, direction, uid, block, str(ds))
                ] = dmg_qnt_series

        damage_quantities = pd.concat(
            dmg_qnt_series_collection.values(),
            axis=1,
            keys=dmg_qnt_series_collection.keys(),
        )
        damage_quantities.columns.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ds']

        # sum up block quantities
        damage_quantities = damage_quantities.groupby(
            level=['cmp', 'loc', 'dir', 'uid', 'ds'], axis=1
        ).sum()

        return damage_quantities

    def _perform_dmg_task(self, task, ds_sample):
        """
        Perform a task from a damage process.

        The method performs a task from a damage process on a given
        damage state sample. The events of the task are triggered by a
        damage state occurrence. The method assigns target
        component(s) into the target damage state based on the damage
        state of the source component. If the target event is "NA",
        the method removes damage state information from the target
        components.

        Parameters
        ----------
        task : list
            A list representing a task from the damage process. The
            list contains two elements:
            - The first element is a string representing the source
              component, e.g., `'1_CMP_A'`. The number in the beginning
              is used to order the tasks and is not considered here.
            - The second element is a dictionary representing the
              events triggered by the damage state of the source
              component. The keys of the dictionary are strings that
              represent the damage state of the source component,
              e.g., `'DS1'`. The values are lists of strings
              representing the target component(s) and event(s), e.g.,
              `['CMP_B.DS1', 'CMP_C.DS1']`. They could also be a
              single element instead of a list.
              Examples of a task:
                ['1_CMP.A', {'DS1': ['CMP.B_DS1', 'CMP.C_DS2']}]
                ['1_CMP.A', {'DS1': 'CMP.B_DS1', 'DS2': 'CMP.B_DS2'}]
                ['1_CMP.A-LOC', {'DS1': 'CMP.B_DS1'}]
        ds_sample : pandas DataFrame
            A DataFrame representing the damage state of the
            components. It is modified in place to represent the
            damage states of the components after the task has been
            performed.

        Raises
        ------
        ValueError
            Raises an error if the source or target event descriptions
            do not follow expected formats.
        """

        if self._asmnt.log.verbose:
            self.log_msg(f'Applying task {task}...', prepend_timestamp=True)

        # parse task
        source_cmp = task[0].split('_')[1]  # source component
        events = task[1]  # prescribed events

        # check for the `-LOC` suffix. If this is the case, we need to
        # match locations.
        if source_cmp.endswith('-LOC'):
            source_cmp = source_cmp.replace('-LOC', '')
            match_locations = True
        else:
            match_locations = False

        # check if the source component exists in the damage state
        # dataframe
        if source_cmp not in ds_sample.columns.get_level_values('cmp'):
            self.log_msg(
                f"WARNING: Source component {source_cmp} in the prescribed "
                "damage process not found among components in the damage "
                "sample. The corresponding part of the damage process is "
                "skipped.",
                prepend_timestamp=False,
            )
            return

        # execute the events pres prescribed in the damage task
        for source_event, target_infos in events.items():

            # events can only be triggered by damage state occurrence
            if not source_event.startswith('DS'):
                raise ValueError(
                    f"Unable to parse source event in damage "
                    f"process: {source_event}"
                )
            # get the ID of the damage state that triggers the event
            ds_source = int(source_event[2:])

            # turn the target_infos into a list if it is a single
            # argument, for consistency
            if not isinstance(target_infos, list):
                target_infos = [target_infos]

            for target_info in target_infos:

                # get the target component and event type
                target_cmp, target_event = target_info.split('_')

                if (target_cmp != 'ALL') and (
                    target_cmp not in ds_sample.columns.get_level_values('cmp')
                ):
                    self.log_msg(
                        f"WARNING: Target component {target_cmp} in the prescribed "
                        "damage process not found among components in the damage "
                        "sample. The corresponding part of the damage process is "
                        "skipped.",
                        prepend_timestamp=False,
                    )
                    continue

                # trigger a damage state
                if target_event.startswith('DS'):

                    # get the ID of the damage state to switch the target
                    # components to
                    ds_target = int(target_event[2:])

                # clear damage state information
                elif target_event == 'NA':
                    ds_target = -1
                    # -1 stands for nan (ints don'ts support nan)

                else:
                    raise ValueError(
                        f"Unable to parse target event in damage "
                        f"process: {target_event}"
                    )

                if match_locations:
                    self._perform_dmg_event_loc(
                        ds_sample, source_cmp, ds_source, target_cmp, ds_target
                    )

                else:
                    self._perform_dmg_event(
                        ds_sample, source_cmp, ds_source, target_cmp, ds_target
                    )

        if self._asmnt.log.verbose:
            self.log_msg(
                'Damage process task successfully applied.', prepend_timestamp=False
            )

    def _perform_dmg_event(
        self, ds_sample, source_cmp, ds_source, target_cmp, ds_target
    ):
        """
        Perform a damage event.
        See `_perform_dmg_task`.

        """

        # affected rows
        row_selection = np.where(
            # for many instances of source_cmp, we
            # consider the highest damage state
            ds_sample[source_cmp].max(axis=1).values
            == ds_source
        )[0]
        # affected columns
        if target_cmp == 'ALL':
            column_selection = np.where(
                ds_sample.columns.get_level_values('cmp') != source_cmp
            )[0]
        else:
            column_selection = np.where(
                ds_sample.columns.get_level_values('cmp') == target_cmp
            )[0]
        ds_sample.iloc[row_selection, column_selection] = ds_target

    def _perform_dmg_event_loc(
        self, ds_sample, source_cmp, ds_source, target_cmp, ds_target
    ):
        """
        Perform a damage event matching locations.
        See `_perform_dmg_task`.

        """

        # get locations of source component
        source_locs = set(ds_sample[source_cmp].columns.get_level_values('loc'))
        for loc in source_locs:
            # apply damage task matching locations
            row_selection = np.where(
                # for many instances of source_cmp, we
                # consider the highest damage state
                ds_sample[source_cmp, loc].max(axis=1).values
                == ds_source
            )[0]

            # affected columns
            if target_cmp == 'ALL':
                column_selection = np.where(
                    np.logical_and(
                        ds_sample.columns.get_level_values('cmp') != source_cmp,
                        ds_sample.columns.get_level_values('loc') == loc,
                    )
                )[0]
            else:
                column_selection = np.where(
                    np.logical_and(
                        ds_sample.columns.get_level_values('cmp') == target_cmp,
                        ds_sample.columns.get_level_values('loc') == loc,
                    )
                )[0]
            ds_sample.iloc[row_selection, column_selection] = ds_target

    def _complete_ds_cols(self, dmg_sample):
        """
        Completes the damage sample dataframe with all possible damage
        states for each component.

        Parameters
        ----------
        dmg_sample : DataFrame
            A DataFrame containing the damage state information for
            each component block in the asset model. The columns are
            MultiIndexed with levels corresponding to component
            information ('cmp', 'loc', 'dir', 'uid') and the damage
            state ('ds').

        Returns
        -------
        DataFrame
            A DataFrame similar to `dmg_sample` but with additional
            columns for missing damage states for each component,
            ensuring that all possible damage states are
            represented. The new columns are filled with zeros,
            indicating no occurrence of those damage states in the
            sample.

        Notes
        -----
        - The method assumes that the damage model parameters
          (`self.damage_params`) are available and contain the
          necessary information to determine the total number of
          damage states for each component.

        """
        # get a shortcut for the damage model parameters
        DP = self.damage_params

        # Get the header for the results that we can use to identify
        # cmp-loc-dir-uid sets
        dmg_header = (
            dmg_sample.groupby(level=[0, 1, 2, 3], axis=1).first().iloc[:2, :]
        )

        # get the number of possible limit states
        ls_list = [col for col in DP.columns.unique(level=0) if 'LS' in col]

        # initialize the result dataframe
        res = pd.DataFrame()

        # walk through all components that have damage parameters provided
        for cmp_id in DP.index:
            # get the component-specific parameters
            cmp_data = DP.loc[cmp_id]

            # and initialize the damage state counter
            ds_count = 0

            # walk through all limit states for the component
            for ls in ls_list:
                # check if the given limit state is defined
                if not pd.isna(cmp_data[(ls, 'Theta_0')]):
                    # check if there is only one damage state
                    if pd.isna(cmp_data[(ls, 'DamageStateWeights')]):
                        ds_count += 1

                    else:
                        # or if there are more than one, how many
                        ds_count += len(
                            cmp_data[(ls, 'DamageStateWeights')].split('|')
                        )

            # get the list of valid cmp-loc-dir-uid sets
            cmp_header = dmg_header.loc[
                :,
                [
                    cmp_id,
                ],
            ]

            # Create a dataframe where they are repeated ds_count times in the
            # columns. The keys put the DS id in the first level of the
            # multiindexed column
            cmp_headers = pd.concat(
                [cmp_header for ds_i in range(ds_count + 1)],
                keys=[str(r) for r in range(0, ds_count + 1)],
                axis=1,
            )
            cmp_headers.columns.names = ['ds', *cmp_headers.columns.names[1::]]

            # add these new columns to the result dataframe
            res = pd.concat([res, cmp_headers], axis=1)

        # Fill the result dataframe with zeros and reorder its columns to have
        # the damage states at the lowest like - matching the dmg_sample input
        res = pd.DataFrame(
            0.0,
            columns=res.columns.reorder_levels([1, 2, 3, 4, 0]),
            index=dmg_sample.index,
        )

        # replace zeros wherever the dmg_sample has results
        res.loc[:, dmg_sample.columns.to_list()] = dmg_sample

        return res


def stringterpolation(
    arguments: str,
) -> tuple[Callable[np.array, np.array]]:
    """
    Turns a string of specially formatted arguments into a multilinear
    interpolating funciton.

    Parameters
    ----------
    arguments: str
        String of arguments containing Y values and X values,
        separated by a pipe symbol (`|`). Individual values are
        separated by commas (`,`). Example:
        arguments = 'y1,y2,y3|x1,x2,x3'

    Returns
    -------
    Callable
        A callable interpolating function

    """
    split = arguments.split('|')
    x_vals = split[0].split(',')
    y_vals = split[1].split(',')
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    return interp1d(x=x, y=y, kind='linear')
