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


"""DamageModel object and methods."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pelicun import base, file_io, uq
from pelicun.model.demand_model import (
    _assemble_required_demand_data,
    _get_required_demand_type,
    _verify_edps_available,
)
from pelicun.model.pelicun_model import PelicunModel

if TYPE_CHECKING:
    from pelicun.assessment import AssessmentBase
    from pelicun.uq import RandomVariableRegistry

idx = base.idx


class DamageModel(PelicunModel):
    """Manages damage information used in assessments."""

    __slots__ = ['ds_model', 'missing_components']

    def __init__(self, assessment: AssessmentBase) -> None:
        """
        Initialize a Damage model.

        Parameters
        ----------
        assessment: AssessmentBase
            The parent assessment object.

        """
        super().__init__(assessment)

        self.ds_model: DamageModel_DS = DamageModel_DS(assessment)
        self.missing_components: list[str] = []

    @property
    def _damage_models(self) -> tuple[DamageModel_DS]:
        """
        Points to the damage model objects included in DamageModel.

        Returns
        -------
        tuple
            A tuple containing the damage models.

        """
        return (self.ds_model,)

    def load_damage_model(
        self, data_paths: list[str | pd.DataFrame], *, warn_missing: bool = False
    ) -> None:
        """<backwards compatibility>."""
        self.log.warning(
            '`load_damage_model` is deprecated and will be '
            'dropped in future versions of pelicun. '
            'Please use `load_model_parameters` instead, '
            'like so: \n`cmp_set = set({your_assessment_obj}.'
            'asset.'
            'list_unique_component_ids())`, '
            'and then \n`{your_assessment_obj}.damage.'
            'load_model_parameters(data_paths, cmp_set)`.'
        )
        cmp_set = set(self._asmnt.asset.list_unique_component_ids())
        self.load_model_parameters(data_paths, cmp_set, warn_missing=warn_missing)

    @property
    def sample(self) -> pd.DataFrame:
        """
        <backwards compatibility>.

        Returns
        -------
        pd.DataFrame
            The damage sample of the `ds_model`.

        """
        self.log.warning(
            '`{damage model}.sample` is deprecated and will be '
            'dropped in future versions of pelicun. '
            'Please use `{damage model}.ds_model.sample` instead. '
            'Now returning `{damage model}.ds_model.sample`.'
        )
        assert self.ds_model.sample is not None
        return self.ds_model.sample

    def load_model_parameters(
        self,
        data_paths: list[str | pd.DataFrame],
        cmp_set: set[str],
        *,
        warn_missing: bool = True,
    ) -> None:
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
        cmp_set: set
            Set of component IDs that are present in the asset model.
            Damage parameters in the input files for components
            outside of that set are omitted for performance.
        warn_missing: bool
            Whether to check if there are components in the asset model
            that do not have specified damage parameters. Should be
            set to True if all components in the asset model are
            damage state-driven, or if only a damage estimation is
            performed, without a subsequent loss estimation.

        Raises
        ------
        ValueError
            If the method can't parse the damage parameters in the
            specified paths.

        """
        self.log.div()
        self.log.msg('Loading damage model...', prepend_timestamp=False)

        # replace default flag with default data path
        data_paths = file_io.substitute_default_path(data_paths, log=self.log)

        #
        # load damage parameter data into the models
        #

        for data_path in data_paths:
            data = file_io.load_data(
                data_path, None, orientation=1, reindex=False, log=self._asmnt.log
            )
            # determine if the damage model parameters are for damage
            # states
            assert isinstance(data, pd.DataFrame)
            if _is_for_ds_model(data):
                self.ds_model.load_model_parameters(data)
            else:
                msg = f'Invalid damage model parameters: {data_path}'
                raise ValueError(msg)

        self.log.msg(
            'Damage model parameters loaded successfully.', prepend_timestamp=False
        )

        #
        # remove items
        #

        self.log.msg(
            'Removing unused damage model parameters.', prepend_timestamp=False
        )

        for damage_model in self._damage_models:
            # drop unused damage parameter definitions
            damage_model.drop_unused_damage_parameters(cmp_set)
            # remove components with incomplete damage parameters
            damage_model.remove_incomplete_components()

        #
        # convert units
        #

        self.log.msg(
            'Converting damage model parameter units.', prepend_timestamp=False
        )
        for damage_model in self._damage_models:
            damage_model.convert_damage_parameter_units()

        #
        # verify damage parameter availability
        #

        self.log.msg(
            'Checking damage model parameter '
            'availability for all components in the asset model.',
            prepend_timestamp=False,
        )
        missing_components = self._ensure_damage_parameter_availability(
            cmp_set, warn_missing=warn_missing
        )

        self.missing_components = missing_components

    def calculate(
        self,
        dmg_process: dict | None = None,
        block_batch_size: int = 1000,
        scaling_specification: dict | None = None,
    ) -> None:
        """
        Calculate the damage of each component block.

        Parameters
        ----------
        dmg_process: dict, optional
            Allows simulating damage processes, where damage to some
            component can alter the damage state of other components.
        block_batch_size: int
            Maximum number of components in each batch.
        scaling_specification: dict, optional
            A dictionary defining the shift in median.
            Example: {'CMP-1-1': '*1.2', 'CMP-1-2': '/1.4'}
            The keys are individual components that should be present
            in the `capacity_sample`.  The values should be strings
            containing an operation followed by the value formatted as
            a float.  The operation can be '+' for addition, '-' for
            subtraction, '*' for multiplication, and '/' for division.

        """
        self.log.div()
        self.log.msg('Calculating damages...')

        assert self._asmnt.asset.cmp_sample is not None
        assert self._asmnt.asset.cmp_marginal_params is not None
        self.log.msg(
            f'Number of Performance Groups in Asset Model:'
            f' {self._asmnt.asset.cmp_sample.shape[1]}',
            prepend_timestamp=False,
        )

        # Instantiate `component_blocks`
        if 'Blocks' in self._asmnt.asset.cmp_marginal_params.columns:
            # If a `Blocks` column is available, use `cmp_marginals`
            component_blocks = (
                self._asmnt.asset.cmp_marginal_params['Blocks']
                .to_frame()
                .astype('int64')
            )
        else:
            # Otherwise assume 1.00 for the number of blocks and
            # initialize `component_blocks` using the columns of `cmp_sample`.
            component_blocks = pd.DataFrame(
                np.ones(self._asmnt.asset.cmp_sample.shape[1]),
                index=self._asmnt.asset.cmp_sample.columns,
                columns=['Blocks'],
                dtype='int64',
            )

        # obtain damage states for applicable components
        assert self._asmnt.demand.sample is not None
        self.ds_model.obtain_ds_sample(
            demand_sample=self._asmnt.demand.sample,
            component_blocks=component_blocks,
            block_batch_size=block_batch_size,
            scaling_specification=scaling_specification,
            missing_components=self.missing_components,
            nondirectional_multipliers=self._asmnt.options.nondir_multi_dict,
        )

        # Apply the prescribed damage process, if any
        if dmg_process is not None:
            self.log.msg('Applying damage processes.')

            # Sort the damage processes tasks
            dmg_process = {key: dmg_process[key] for key in sorted(dmg_process)}

            # Perform damage tasks in the sorted order
            for task in dmg_process.items():
                self.ds_model.perform_dmg_task(task)

            self.log.msg(
                'Damage processes successfully applied.', prepend_timestamp=False
            )

        qnt_sample = self.ds_model.prepare_dmg_quantities(
            self._asmnt.asset.cmp_sample,
            self._asmnt.asset.cmp_marginal_params,
            dropzero=False,
        )

        # If requested, extend the quantity table with all possible DSs
        if self._asmnt.options.list_all_ds:
            qnt_sample = self.ds_model.complete_ds_cols(qnt_sample)

        self.ds_model.sample = qnt_sample

        self.log.msg('Damage calculation completed.', prepend_timestamp=False)

    def save_sample(
        self, filepath: str | None = None, *, save_units: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series] | None:
        """
        Save or return the damage sample data.

        Saves the damage sample data to a CSV file or returns it
        directly with an option to include units.

        This function handles saving the sample data of damage
        assessments to a specified file path or, if no path is
        provided, returns the data as a DataFrame. The function can
        optionally include a row for unit information when returning
        data.

        Parameters
        ----------
        filepath: str, optional
            The path to the file where the damage sample should be
            saved. If not provided, the sample is not saved to disk
            but returned.
        save_units: bool, default: False
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
        self.log.div()
        self.log.msg('Saving damage sample...')

        if self.ds_model.sample is None:
            return None

        cmp_units = self._asmnt.asset.cmp_units
        qnt_units = pd.Series(
            index=self.ds_model.sample.columns, name='Units', dtype='object'
        )
        assert cmp_units is not None
        for cmp in cmp_units.index:
            qnt_units.loc[cmp] = cmp_units.loc[cmp]

        res = file_io.save_to_csv(
            self.ds_model.sample,
            Path(filepath) if filepath is not None else None,
            units=qnt_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )

        if filepath is not None:
            self.log.msg(
                'Damage sample successfully saved.', prepend_timestamp=False
            )
            return None

        # else:
        assert isinstance(res, pd.DataFrame)
        units = res.loc['Units']
        assert isinstance(units, pd.Series)
        res = res.drop('Units')
        res.index = res.index.astype('int64')
        res = res.astype(float)
        assert isinstance(res, pd.DataFrame)

        if save_units:
            return res, units

        return res

    def load_sample(self, filepath: str) -> None:
        """Load damage state sample data."""
        self.log.div()
        self.log.msg('Loading damage sample...')

        data = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors, log=self._asmnt.log
        )
        assert isinstance(data, pd.DataFrame)
        self.ds_model.sample = data

        # set the names of the columns
        self.ds_model.sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'ds']

        self.log.msg('Damage sample successfully loaded.', prepend_timestamp=False)

    def _ensure_damage_parameter_availability(
        self, cmp_set: set[str], *, warn_missing: bool
    ) -> list[str]:
        """
        Make sure that all components have damage parameters.

        Parameters
        ----------
        cmp_set: list
            List of component IDs in the asset model.
        warn_missing: bool
            Whether to issue a warning if missing components are found.

        Returns
        -------
        list
            List of component IDs with missing damage parameters.

        """
        available_components = self._get_component_id_set()

        missing_components = [
            component
            for component in cmp_set
            if component not in available_components
        ]

        if missing_components and warn_missing:
            self.log.warning(
                f'The damage model does not provide '
                f'damage information for the following component(s) '
                f'in the asset model: {missing_components}.'
            )

        return missing_components

    def _get_component_id_set(self) -> set[str]:
        """
        Get a set of components with available damage parameters.

        Returns
        -------
        set
          Set of components with available damage parameters.

        """
        cmp_list = []
        if self.ds_model.damage_params is not None:
            cmp_list.extend(self.ds_model.damage_params.index.to_list())
        return set(cmp_list)


class DamageModel_Base(PelicunModel):
    """Base class for damage models."""

    __slots__ = ['damage_params', 'sample']

    def __init__(self, assessment: AssessmentBase) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        assessment: AssessmentBase
            Parent assessment object.

        """
        super().__init__(assessment)

        self.damage_params: pd.DataFrame | None = None
        self.sample: pd.DataFrame | None = None

    def load_model_parameters(self, data: pd.DataFrame) -> None:
        """
        Load model parameters from a DataFrame.

        Loads model parameters from a DataFrame, extending those
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
        data = data.groupby(data.index).first(skipna=False)

        # TODO(AZ): load defaults for Demand-Offset and Demand-Directional
        self.damage_params = data

    def convert_damage_parameter_units(self) -> None:
        """Convert previously loaded damage parameters to base units."""
        if self.damage_params is None:
            return

        units = self.damage_params['Demand', 'Unit']
        self.damage_params = self.damage_params.drop(('Demand', 'Unit'), axis=1)
        for ls_i in self.damage_params.columns.unique(level=0):
            if ls_i.startswith('LS'):
                params = self.damage_params.loc[:, ls_i].copy()
                assert isinstance(params, pd.DataFrame)
                self.damage_params.loc[:, ls_i] = self._convert_marginal_params(
                    params, units
                ).to_numpy()

    def remove_incomplete_components(self) -> None:
        """
        Remove components with incompelte damage parameter info.

        Removes components that have incomplete damage model
        definitions from the damage model parameters.

        """
        if self.damage_params is None:
            return

        if ('Incomplete', '') not in self.damage_params.columns:
            return

        cmp_incomplete_idx = self.damage_params.loc[
            self.damage_params['Incomplete', ''] == 1
        ].index

        self.damage_params = self.damage_params.drop(cmp_incomplete_idx)

    def drop_unused_damage_parameters(self, cmp_set: set[str]) -> None:
        """
        Remove info for non existent components.

        Removes damage parameter definitions for component IDs not
        present in the given list.

        Parameters
        ----------
        cmp_set: set
            Set of component IDs to be preserved in the damage
            parameters.

        """
        if self.damage_params is None:
            return
        cmp_mask = self.damage_params.index.isin(cmp_set, level=0)

        self.damage_params = self.damage_params.iloc[cmp_mask, :]

    def _get_pg_batches(
        self,
        component_blocks: pd.DataFrame,
        block_batch_size: int,
        missing_components: list[str],
    ) -> pd.DataFrame:
        """
        Group performance groups into batches for efficiency.

        The method takes as input the block_batch_size, which
        specifies the maximum number of blocks per batch. The method
        first checks if performance groups have been defined in the
        cmp_marginal_params DataFrame, and if so, it uses the 'Blocks'
        column as the performance group information. If performance
        groups have not been defined in cmp_marginal_params, the
        method uses the cmp_sample DataFrame to define the performance
        groups, with each performance group having a single block.

        The method then checks if the performance groups are available
        in the damage parameters DataFrame, and removes any
        performance groups that are not found in the damage
        parameters. The method then groups the performance groups
        based on the locations and directions of the components, and
        calculates the cumulative sum of the blocks for each
        group. The method then divides the performance groups into
        batches of size specified by block_batch_size and assigns a
        batch number to each group. Finally, the method groups the
        performance groups by batch number, component, location, and
        direction, and returns a DataFrame that shows the number of
        blocks for each batch.

        Parameters
        ----------
        component_blocks: pd.DataFrame
            DataFrame containing a single column, `Blocks`, which lists
            the number of blocks for each (`cmp`-`loc`-`dir`-`uid`).
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
            batch. This DataFrame facilitates the management and
            execution of damage assessment tasks by grouping
            components into manageable batches based on the specified
            block batch size.

        """
        # A warning has already been issued for components with
        # missing damage parameters (in
        # `DamageModel._ensure_damage_parameter_availability`).
        component_blocks = component_blocks.drop(pd.Index(missing_components))

        # It is safe to simply disregard components that are not
        # present in the `damage_params` of *this* model, and let them
        # be handled by another damage model.
        assert self.damage_params is not None
        available_components = self.damage_params.index.unique().to_list()
        component_blocks = component_blocks.loc[
            pd.IndexSlice[available_components, :, :, :], :
        ]

        # Sum up the number of blocks for each performance group
        component_blocks = component_blocks.groupby(
            ['loc', 'dir', 'cmp', 'uid']
        ).sum()
        component_blocks = component_blocks.sort_index(axis=0)

        # Calculate cumulative sum of blocks
        component_blocks['CBlocks'] = np.cumsum(
            component_blocks['Blocks'].to_numpy().astype(int)
        )
        component_blocks['Batch'] = 0

        # Group the performance groups into batches
        for batch_i in range(1, component_blocks.shape[0] + 1):
            # Find the mask for blocks that are less than the batch
            # size and greater than 0
            batch_mask = np.all(
                np.array(
                    [
                        component_blocks['CBlocks'] <= block_batch_size,
                        component_blocks['CBlocks'] > 0,
                    ]
                ),
                axis=0,
            )

            if np.sum(batch_mask) < 1:
                batch_mask = np.full(batch_mask.shape, fill_value=False)
                batch_mask[np.where(component_blocks['CBlocks'] > 0)[0][0]] = True

            component_blocks.loc[batch_mask, 'Batch'] = batch_i

            # Decrement the cumulative block count by the max count in
            # the current batch
            component_blocks['CBlocks'] -= component_blocks.loc[
                component_blocks['Batch'] == batch_i, 'CBlocks'
            ].max()

            # If the maximum cumulative block count is 0, exit the
            # loop
            if component_blocks['CBlocks'].max() == 0:
                break

        # Group the performance groups by batch, component, location,
        # and direction, and keep only the number of blocks for each
        # group
        component_blocks = (
            component_blocks.groupby(['Batch', 'cmp', 'loc', 'dir', 'uid'])
            .sum()
            .loc[:, 'Blocks']
            .to_frame()
        )
        return component_blocks.sort_index(
            level=['Batch', 'cmp', 'loc', 'dir', 'uid']
        )


class DamageModel_DS(DamageModel_Base):
    """Damage model for components that have discrete Damage States (DS)."""

    __slots__ = ['ds_sample']

    def __init__(self, assessment: AssessmentBase) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        assessment: AssessmentBase
            Parent assessment object.

        """
        super().__init__(assessment)
        self.ds_sample: pd.DataFrame | None = None

    def probabilities(self) -> pd.DataFrame:
        """
        Return the probability of each observed damage state.

        Returns
        -------
        pd.DataFrame
            DataFrame with the probability of each damage state for
            each component block.

        """
        sample = self.ds_sample
        assert sample is not None

        probabilities = {}

        for col in sample.columns:
            values = sample[col]
            # skip NA cases that are the result of damage processes
            values = values[values != -1]
            if len(values) == 0:
                # can't determine without a sample
                probabilities[col] = np.nan
            else:
                vcounts = values.value_counts() / len(values)
                probabilities[col] = vcounts  # type: ignore

        return (
            pd.DataFrame(probabilities)
            .T.rename_axis(
                index=['cmp', 'loc', 'dir', 'uid', 'block'], columns='Damage State'
            )
            .sort_index(axis=1)
            .sort_index(axis=0)
        )

    def obtain_ds_sample(
        self,
        demand_sample: pd.DataFrame,
        component_blocks: pd.DataFrame,
        block_batch_size: int,
        scaling_specification: dict | None,
        missing_components: list[str],
        nondirectional_multipliers: dict[str, float],
    ) -> None:
        """Obtain the damage state of each performance group."""
        # Break up damage calculation and perform it by performance group.
        # Compared to the simultaneous calculation of all PGs, this approach
        # reduces demands on memory and increases the load on CPU. This leads
        # to a more balanced workload on most machines for typical problems.
        # It also allows for a straightforward extension with parallel
        # computing.

        sample_size = len(demand_sample)

        component_blocks = self._get_pg_batches(
            component_blocks, block_batch_size, missing_components
        )
        batches = component_blocks.index.get_level_values(0).unique()

        self.log.msg(
            f'Number of Component Blocks: {component_blocks["Blocks"].sum()}',
            prepend_timestamp=False,
        )

        self.log.msg(
            f'{len(batches)} batches of Performance Groups prepared '
            'for damage assessment',
            prepend_timestamp=False,
        )

        # for PG_i in self._asmnt.asset.cmp_sample.columns:
        ds_samples = []
        for pgb_i in batches:
            performance_group = component_blocks.loc[pgb_i]

            self.log.msg(
                f"Calculating damage states for PG batch {pgb_i} with "
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
            if self._asmnt.log.verbose:
                self.log.msg(
                    'Collecting required demand information...',
                    prepend_timestamp=True,
                )
            demand_offset = self._asmnt.options.demand_offset
            assert self.damage_params is not None
            required_edps = _get_required_demand_type(
                self.damage_params, performance_group, demand_offset
            )

            available_edps = (
                pd.DataFrame(index=demand_sample.columns)
                .reset_index()
                .groupby(['type', 'loc'])['dir']
                .agg(lambda x: list(set(x)))
                .to_dict()
            )

            # Raise an error if demand sample is missing necessary entries.
            _verify_edps_available(available_edps, set(required_edps.keys()))

            # Create the demand vector
            if self._asmnt.log.verbose:
                self.log.msg(
                    'Assembling demand data for calculation...',
                    prepend_timestamp=True,
                )
            demand_dict = _assemble_required_demand_data(
                set(required_edps.keys()), nondirectional_multipliers, demand_sample
            )

            # Evaluate the Damage State of each Component Block
            ds_sample = self._evaluate_damage_state(
                demand_dict, required_edps, capacity_sample, lsds_sample
            )

            ds_samples.append(ds_sample)

        self.ds_sample = pd.concat(ds_samples, axis=1)

        self.log.msg('Damage state calculation successful.', prepend_timestamp=False)

    def _handle_operation(  # noqa: PLR6301
        self, initial_value: float, operation: str, other_value: float
    ) -> float:
        """
        Handle a capacity adjustment operation.

        This method is used in `_create_dmg_RVs` to apply capacity
        adjustment operations whenever required. It is defined as a
        safer alternative to directly using `eval`.

        Parameters
        ----------
        initial_value: float
            Value before operation
        operation: str
            Any of `+`, `-`, `*`, `/`
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
        msg = f'Invalid operation: `{operation}`'
        raise ValueError(msg)

    def _handle_operation_list(
        self, initial_value: float, operations: list[tuple[str, float]]
    ) -> np.ndarray:
        """
        Apply one or more operations to an initial value and return the results.

        Parameters.
        ----------
        initial_value : float
            The initial value to which the operations will be applied.
        operations : list of tuple
            A list of operations where each operation is represented as a tuple.
            The first element of the tuple is a string representing the operation
            type, and the second element is a float representing the value to be
            used in the operation.

        Returns
        -------
        np.ndarray
            An array of results after applying each operation to the initial value.
        """
        if len(operations) == 1:
            return np.array(
                [
                    self._handle_operation(
                        initial_value, operations[0][0], operations[0][1]
                    )
                ]
            )
        new_values = [
            self._handle_operation(initial_value, operation[0], operation[1])
            for operation in operations
        ]
        return np.array(new_values)

    def _generate_dmg_sample(
        self,
        sample_size: int,
        pgb: pd.DataFrame,
        scaling_specification: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate the damage sample.

        Generates a damage sample by creating random variables (RVs)
        for capacities and limit-state-damage-states (lsds), and then
        sampling from these RVs. The sample size and performance group
        batches (PGB) are specified as inputs. The method returns the
        capacity sample and the lsds sample.

        Parameters
        ----------
        sample_size: int
            The number of realizations to generate.
        pgb: DataFrame
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
        capacity_sample: DataFrame
            A DataFrame that represents the capacity sample.
        lsds_sample: DataFrame
            A DataFrame that represents the .

        Raises
        ------
        ValueError
            If the damage parameters have not been specified.

        """
        # Check if damage model parameters have been specified
        if self.damage_params is None:
            msg = (
                'Damage model parameters have not been specified. '
                'Load parameters from the default damage model '
                'databases or provide your own damage model '
                'definitions before generating a sample.'
            )
            raise ValueError(msg)

        # Create capacity and LSD RVs for each performance group
        capacity_rvs, lsds_rvs = self._create_dmg_RVs(pgb, scaling_specification)

        if self._asmnt.log.verbose:
            self.log.msg('Sampling capacities...', prepend_timestamp=True)

        # Generate samples for capacity RVs
        assert self._asmnt.options.sampling_method is not None
        capacity_rvs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        # Generate samples for LSD RVs
        lsds_rvs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        if self._asmnt.log.verbose:
            self.log.msg('Raw samples are available', prepend_timestamp=True)

        # get the capacity and lsds samples
        capacity_sample = (
            pd.DataFrame(capacity_rvs.RV_sample)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        capacity_sample_mi = base.convert_to_MultiIndex(capacity_sample, axis=1)[
            'FRG'
        ]
        assert isinstance(capacity_sample_mi, pd.DataFrame)
        capacity_sample = capacity_sample_mi
        capacity_sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ls']

        lsds_sample = (
            pd.DataFrame(lsds_rvs.RV_sample)
            .sort_index(axis=0)
            .sort_index(axis=1)
            .astype(int)
        )
        lsds_sample_mi = base.convert_to_MultiIndex(lsds_sample, axis=1)['LSDS']
        assert isinstance(lsds_sample_mi, pd.DataFrame)
        lsds_sample = lsds_sample_mi
        lsds_sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ls']

        if self._asmnt.log.verbose:
            self.log.msg(
                f'Successfully generated {sample_size} realizations.',
                prepend_timestamp=True,
            )

        return capacity_sample, lsds_sample

    def _evaluate_damage_state(
        self,
        demand_dict: dict[str, np.ndarray],
        required_edps: dict[str, list[tuple]],
        capacity_sample: pd.DataFrame,
        lsds_sample: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Use the demand and LS capacity sample to evaluate damage states.

        Parameters
        ----------
        demand_dict: dict
            Dictionary containing the demand of each demand type.
        required_edps: dict
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
            self.log.msg('Evaluating damage states...', prepend_timestamp=True)

        # Create an empty DataFrame with columns and index taken from
        # the input capacity sample
        dmg_eval = pd.DataFrame(
            columns=capacity_sample.columns, index=capacity_sample.index
        )

        # Initialize an empty list to store demand data
        demand_df = []

        # For each demand type in the demand dictionary
        for demand_name, demand_vals in demand_dict.items():
            # Get the list of PGs assigned to this demand type
            pg_list = required_edps[demand_name]

            # Create a list of columns for the demand data
            # corresponding to each PG in the PG_list
            pg_cols = pd.concat(
                [dmg_eval.loc[:1, PG_i] for PG_i in pg_list],  # type: ignore
                axis=1,
                keys=pg_list,
            ).columns
            pg_cols.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ls']
            # Create a DataFrame with demand values repeated for the
            # number of PGs and assign the columns as PG_cols
            demand_df.append(
                pd.concat(
                    [pd.Series(demand_vals)] * len(pg_cols), axis=1, keys=pg_cols
                )
            )

        # Concatenate all demand DataFrames into a single DataFrame
        demand_df_concat = pd.concat(demand_df, axis=1)
        # Sort the columns of the demand DataFrame
        demand_df_concat = demand_df_concat.sort_index(axis=1)

        # Evaluate the damage exceedance by subtracting demand from
        # capacity and checking if the result is less than zero
        dmg_eval = (capacity_sample - demand_df_concat) < 0

        # Remove any columns with NaN values from the damage
        # exceedance DataFrame
        dmg_eval = dmg_eval.dropna(axis=1)

        # initialize the DataFrames that store the damage states and
        # quantities
        ds_sample = pd.DataFrame(
            0,  # fill value
            columns=capacity_sample.columns.droplevel('ls').unique(),
            index=capacity_sample.index,
            dtype='int64',
        )

        # get a list of limit state ids among all components in the damage model
        ls_list = dmg_eval.columns.get_level_values(5).unique()

        # for each consecutive limit state...
        for ls_id in ls_list:
            # get all cmp - loc - dir - block where this limit state occurs
            dmg_e_ls = dmg_eval.loc[
                :,  # type: ignore
                idx[:, :, :, :, :, ls_id],
            ].dropna(axis=1)

            # Get the damage states corresponding to this limit state in each
            # block
            # Note that limit states with a set of mutually exclusive damage
            # states options have their damage state picked here.
            lsds = lsds_sample.loc[:, dmg_e_ls.columns]  # type: ignore

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
                :, dmg_e_ls.columns  # type: ignore
            ].mask(dmg_e_ls, lsds)

        return ds_sample

    def _create_dmg_RVs(  # noqa: N802, C901
        self, pgb: pd.DataFrame, scaling_specification: dict | None = None
    ) -> tuple[uq.RandomVariableRegistry, uq.RandomVariableRegistry]:
        """
        Create random variables for the damage calculation.

        The method initializes two random variable registries,
        capacity_RV_reg and lsds_RV_reg, and loops through each
        performance group in the input performance group batch (PGB)
        DataFrame. For each performance group, it retrieves the
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
        pgb: DataFrame
            A DataFrame that groups performance groups into batches
            for efficient damage assessment.
        scaling_specification: dict, optional
                A dictionary defining the shift in median.
                Example: {'CMP-1-1': {'LS1':['*1.2'. '*0.8'], 'LS2':'*1.2'},
                'CMP-1-2': {'ALL':'/1.4'}} The first level keys are individual
                components that should be present in the `capacity_sample`. The
                second level key is the limit state to apply the scaling to. The
                values should be strings or list of strings. The strings should
                contain an operation followed by the value formatted as a float.
                The operation can be '+' for addition, '-' for subtraction, '*'
                for multiplication, and '/' for division. If different operations
                are required for different realizations, a list of strings can
                be provided. When 'ALL' is used as the key, the operation will
                be applied to all limit states.

        Returns
        -------
        tuple
            A tuple containing two RandomVariableRegistry instances:
            one for the capacity random variables and one for the LSDS
            assignments.

        """

        def assign_lsds(
            ds_weights: str | None,
            ds_id: int,
            lsds_rv_reg: RandomVariableRegistry,
            lsds_rv_tag: str,
        ) -> int:
            """
            Assign limit states to damage states.

            Assigns limit states to damage states using random
            variables, updating the provided random variable registry.
            This function either creates a deterministic random
            variable for a single damage state or a multinomial random
            variable for multiple damage states.

            Parameters
            ----------
            ds_weights: str or None
                A string representing the weights of different damage
                states associated with a limit state, separated by
                '|'.  If None, indicates that there is only one damage
                state associated with the limit state.
            ds_id: int
                The starting index for damage state IDs. This ID helps
                in mapping damage states to limit states.
            lsds_rv_reg: RandomVariableRegistry
                The registry where the newly created random variables
                (for mapping limit states to damage states) will be
                added.
            lsds_rv_tag: str
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
            if pd.isna(ds_weights):
                ds_id += 1
                lsds_rv_reg.add_RV(
                    uq.DeterministicRandomVariable(
                        name=lsds_rv_tag,
                        theta=np.array((ds_id,)),
                    )
                )

            # Otherwise, we create a multinomial random variable
            else:
                assert isinstance(ds_weights, str)
                # parse the DS weights
                ds_weights_np = np.array(
                    ds_weights.replace(' ', '').split('|'), dtype=float
                )

                def map_ds(values: np.ndarray, offset: int) -> np.ndarray:
                    """
                    Map DS indices to damage states.

                    Maps an array of damage state indices to their
                    corresponding actual state IDs by applying an
                    offset.

                    Parameters
                    ----------
                    values: array-like
                        An array of indices representing damage
                        states. These indices are typically sequential
                        integers starting from zero.
                    offset: int
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

                lsds_rv_reg.add_RV(
                    uq.MultinomialRandomVariable(
                        name=lsds_rv_tag,
                        theta=ds_weights_np,
                        f_map=partial(map_ds, offset=ds_id + 1),
                    )
                )

                ds_id += len(ds_weights_np)

            return ds_id

        def parse_scaling_specification(scaling_specification: dict) -> dict:  # noqa: C901
            """
            Parse and validate the scaling specification, used in the '_create_dmg_RVs' method.

            Parameters
            ----------
            scaling_specification: dict, optional
                A dictionary defining the shift in median.
                Example: {'CMP-1-1': {'LS1':['*1.2'. '*0.8'], 'LS2':'*1.2'},
                'CMP-1-2': {'ALL':'/1.4'}} The first level keys are individual
                components that should be present in the `capacity_sample`. The
                second level key is the limit state to apply the scaling to. The
                values should be strings or list of strings. The strings should
                containing an operation followed by the value formatted as
                a float.  The operation can be '+' for addition, '-' for
                subtraction, '*' for multiplication, and '/' for division. If
                different operations are required for different realizations, a
                list of strings can be provided. When 'ALL' is used as the key,
                the operation will be applied to all limit states.

            Returns
            -------
            dict
                The parsed and validated scaling specification.

            Raises
            ------
            ValueError
                If the scaling specification is invalid.
            TypeError
                If the type of an entry is invalid.
            """
            # if there are contents, ensure they are valid.
            # See docstring for an example of what is expected.
            parsed_scaling_specification: defaultdict = defaultdict(dict)
            # validate contents
            for key, value in scaling_specification.items():
                # loop through limit states
                if 'ALL' in value:
                    if len(value) > 1:
                        msg = (
                            f'Invalid entry in scaling_specification: '
                            f"{value}. No other entries are allowed for a component when 'ALL' is used."
                        )
                        raise ValueError(msg)
                for limit_state_id, specifics in value.items():
                    if not (
                        limit_state_id.startswith('LS') or limit_state_id == 'ALL'
                    ):
                        msg = (
                            f'Invalid entry in scaling_specification: {limit_state_id}. '
                            f"It has to start with 'LS' or be 'ALL'. "
                            f'See docstring of DamageModel._create_dmg_RVs.'
                        )
                        raise ValueError(msg)
                    css = 'capacity adjustment specification'
                    if not isinstance(specifics, list):
                        specifics_list = [specifics]
                    else:
                        specifics_list = specifics
                    for spec in specifics_list:
                        if not isinstance(spec, str):
                            msg = (
                                f'Invalud entry in {css}: {spec}.'
                                f'The specified scaling operation has to be a string.'
                                f'See docstring of DamageModel._create_dmg_RVs.'
                            )
                            raise TypeError(msg)
                        capacity_adjustment_operation = spec[0]
                        number = spec[1::]
                        if capacity_adjustment_operation not in {'+', '-', '*', '/'}:
                            msg = f'Invalid operation in {css}: '
                            raise ValueError(msg, f'{capacity_adjustment_operation}')
                        fnumber = base.float_or_None(number)
                        if fnumber is None:
                            msg = f'Invalid number in {css}: {number}'
                            raise ValueError(msg)
                        if limit_state_id not in parsed_scaling_specification[key]:
                            parsed_scaling_specification[key][limit_state_id] = []
                        parsed_scaling_specification[key][limit_state_id].append(
                            (capacity_adjustment_operation, fnumber)
                        )
            return parsed_scaling_specification

        if self._asmnt.log.verbose:
            self.log.msg('Generating capacity variables ...', prepend_timestamp=True)

        # initialize the registry
        capacity_rv_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)
        lsds_rv_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        # capacity adjustment:
        # ensure the scaling_specification is a dictionary
        if not scaling_specification:
            scaling_specification = {}
        else:
            scaling_specification = parse_scaling_specification(
                scaling_specification
            )

        # get the component sample and blocks from the asset model
        for pg in pgb.index:  # noqa: PLR1702
            # determine demand capacity adjustment operation, if required
            cmp_loc_dir = '-'.join(pg[0:3])
            capacity_adjustment_operation = scaling_specification.get(  # type: ignore
                cmp_loc_dir,
            )

            cmp_id = pg[0]
            blocks = pgb.loc[pg, 'Blocks']

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
                        limit_states.append(val[2:])  # noqa: PERF401

                ds_id = 0

                frg_rv_set_tags: list = [[] for b in blocks]
                anchor_rvs: list = []

                for ls_id in limit_states:
                    frg_params_ls = frg_params[f'LS{ls_id}']

                    theta_0 = frg_params_ls.get('Theta_0', np.nan)
                    family = frg_params_ls.get('Family', 'deterministic')

                    # if `family` is defined but is `None`, we
                    # consider it to be `deterministic`
                    if not family:
                        family = 'deterministic'
                    ds_weights = frg_params_ls.get('DamageStateWeights', None)

                    # check if the limit state is defined for the component
                    if pd.isna(theta_0):
                        continue

                    theta = np.array(
                        [
                            value
                            for t_i in range(3)
                            if (value := frg_params_ls.get(f'Theta_{t_i}', None))
                            is not None
                        ]
                    )

                    if capacity_adjustment_operation:
                        if family in {'normal', 'lognormal', 'deterministic'}:
                            # Only scale the median value if ls_id is defined in capacity_adjustment_operation
                            # Otherwise, use the original value
                            new_theta_0 = None
                            if 'ALL' in capacity_adjustment_operation:
                                new_theta_0 = self._handle_operation_list(
                                    theta[0],
                                    capacity_adjustment_operation['ALL'],
                                )
                            elif f'LS{ls_id}' in capacity_adjustment_operation:
                                new_theta_0 = self._handle_operation_list(
                                    theta[0],
                                    capacity_adjustment_operation[f'LS{ls_id}'],
                                )
                            if new_theta_0 is not None:
                                if new_theta_0.size == 1:
                                    theta[0] = new_theta_0[0]
                                else:
                                    # Repeat the theta values new_theta_0.size times along axis 0
                                    # and 1 time along axis 1
                                    theta = np.tile(theta, (new_theta_0.size, 1))
                                    theta[:, 0] = new_theta_0
                        else:
                            self.log.warning(
                                f'Capacity adjustment is only supported '
                                f'for `normal` or `lognormal` distributions. '
                                f'Ignoring: `{cmp_loc_dir}`, which is `{family}`'
                            )

                    tr_lims = np.array(
                        [
                            frg_params_ls.get(f'Truncate{side}', np.nan)
                            for side in ('Lower', 'Upper')
                        ]
                    )

                    for block_i, _ in enumerate(blocks):
                        frg_rv_tag = (
                            'FRG-'
                            f'{pg[0]}-'  # cmp_id
                            f'{pg[1]}-'  # loc
                            f'{pg[2]}-'  # dir
                            f'{pg[3]}-'  # uid
                            f'{block_i + 1}-'  # block
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
                            anchor = anchor_rvs[block_i]

                        # parse theta values for multilinear_CDF
                        if family == 'multilinear_CDF':
                            theta = np.column_stack(  # type: ignore
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

                        rv = uq.rv_class_map(family)(  # type: ignore
                            name=frg_rv_tag,
                            theta=theta,
                            truncation_limits=tr_lims,
                            anchor=anchor,
                        )

                        capacity_rv_reg.add_RV(rv)  # type: ignore

                        # add the RV to the set of correlated variables
                        frg_rv_set_tags[block_i].append(frg_rv_tag)

                        if ls_id == limit_states[0]:
                            anchor_rvs.append(rv)

                        # Now add the LS->DS assignments
                        lsds_rv_tag = (
                            'LSDS-'
                            f'{pg[0]}-'  # cmp_id
                            f'{pg[1]}-'  # loc
                            f'{pg[2]}-'  # dir
                            f'{pg[3]}-'  # uid
                            f'{block_i + 1}-'  # block
                            f'{ls_id}'
                        )

                        ds_id_next = assign_lsds(
                            ds_weights, ds_id, lsds_rv_reg, lsds_rv_tag
                        )

                    ds_id = ds_id_next

        if self._asmnt.log.verbose:
            rv_count = len(lsds_rv_reg.RV)
            self.log.msg(
                f'2x{rv_count} random variables created.', prepend_timestamp=False
            )

        return capacity_rv_reg, lsds_rv_reg

    def prepare_dmg_quantities(
        self,
        component_sample: pd.DataFrame,
        component_marginal_parameters: pd.DataFrame | None,
        *,
        dropzero: bool = True,
    ) -> pd.DataFrame:
        """
        Combine component quantity and damage state information.

        This method assumes that a component quantity sample is
        available in the asset model and a damage state sample is
        available in the damage model.

        Parameters
        ----------
        component_sample: pd.DataFrame
            Component quantity sample from the AssetModel.
        component_marginal_parameters: pd.DataFrame
            Component marginal parameters from the AssetModel.
        dropzero: bool, optional, default: True
            If True, the quantity of non-damaged components is not
            saved.

        Returns
        -------
        DataFrame
            A DataFrame that combines the component quantity and
            damage state information.

        """
        # ('cmp', 'loc', 'dir', 'uid') -> component quantity series
        component_quantities = component_sample.to_dict('series')

        if self._asmnt.log.verbose:
            self.log.msg('Calculating damage quantities...', prepend_timestamp=True)

        # Retrieve the component quantity information and component
        # marginal parameters from the asset model

        if (component_marginal_parameters is not None) and (
            'Blocks' in component_marginal_parameters.columns
        ):
            # if this information is available, use it

            # ('cmp', 'loc', 'dir', 'uid) -> number of blocks
            num_blocks = component_marginal_parameters['Blocks'].to_dict()

            def get_num_blocks(key: object) -> float:
                return float(num_blocks[key])

        else:
            # otherwise assume 1 block regardless of
            # ('cmp', 'loc', 'dir', 'uid) key
            def get_num_blocks(key: object) -> float:  # noqa: ARG001
                return 1.00

        # ('cmp', 'loc', 'dir', 'uid', 'block') -> damage state series
        assert self.ds_sample is not None
        damage_state_sample_dict = self.ds_sample.to_dict('series')

        dmg_qnt_series_collection = {}
        for key, damage_state_series in damage_state_sample_dict.items():
            component: str
            location: str
            direction: str
            uid: str
            block: str
            component, location, direction, uid, block = key  # type: ignore
            damage_state_set = set(damage_state_series.values)
            for ds in damage_state_set:
                if ds == -1:
                    continue
                if dropzero and ds == 0:
                    continue
                dmg_qnt_vals = np.where(
                    damage_state_series.to_numpy() == ds,
                    component_quantities[
                        component, location, direction, uid
                    ].to_numpy()
                    / get_num_blocks((component, location, direction, uid)),
                    0.00,
                )
                if -1 in damage_state_set:
                    dmg_qnt_vals = np.where(
                        damage_state_series.to_numpy() != -1, dmg_qnt_vals, np.nan
                    )
                dmg_qnt_series = pd.Series(dmg_qnt_vals)
                dmg_qnt_series_collection[
                    component, location, direction, uid, block, str(ds)
                ] = dmg_qnt_series

        damage_quantities = pd.concat(
            dmg_qnt_series_collection.values(),
            axis=1,
            keys=dmg_qnt_series_collection.keys(),
        )
        damage_quantities.columns.names = ['cmp', 'loc', 'dir', 'uid', 'block', 'ds']

        # min_count=1 is specified so that the sum cross all NaNs will
        # result in NaN instead of zero.
        # https://stackoverflow.com/questions/33448003/sum-across-all-nans-in-pandas-returns-zero
        return damage_quantities.groupby(  # type: ignore
            level=['cmp', 'loc', 'dir', 'uid', 'ds'], axis=1
        ).sum(min_count=1)

    def perform_dmg_task(self, task: tuple) -> None:  # noqa: C901
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
        task: list
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

        Raises
        ------
        ValueError
            Raises an error if the source or target event descriptions
            do not follow expected formats.

        """
        if self._asmnt.log.verbose:
            self.log.msg(f'Applying task {task}...', prepend_timestamp=True)

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
        # DataFrame
        assert self.ds_sample is not None
        if source_cmp not in self.ds_sample.columns.get_level_values('cmp'):
            self.log.warning(
                f'Source component `{source_cmp}` in the prescribed '
                'damage process not found among components in the damage '
                'sample. The corresponding part of the damage process is '
                'skipped.'
            )
            return

        # execute the events prescribed in the damage task
        for source_event, target_infos in events.items():
            # events can only be triggered by damage state occurrence
            if not source_event.startswith('DS'):
                msg = (
                    f'Unable to parse source event in damage '
                    f'process: `{source_event}`'
                )
                raise ValueError(msg)
            # get the ID of the damage state that triggers the event
            ds_source = int(source_event[2:])

            # turn the target_infos into a list if it is a single
            # argument, for consistency
            if not isinstance(target_infos, list):
                target_infos = [target_infos]  # noqa: PLW2901

            for target_info in target_infos:
                # get the target component and event type
                target_cmp, target_event = target_info.split('_')

                if (target_cmp != 'ALL') and (
                    target_cmp not in self.ds_sample.columns.get_level_values('cmp')
                ):
                    self.log.warning(
                        f'Target component `{target_cmp}` in the prescribed '
                        'damage process not found among components in the damage '
                        'sample. The corresponding part of the damage process is '
                        'skipped.'
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
                    msg = (
                        f'Unable to parse target event in damage '
                        f'process: `{target_event}`'
                    )
                    raise ValueError(msg)

                if match_locations:
                    self._perform_dmg_event_loc(
                        source_cmp, ds_source, target_cmp, ds_target
                    )

                else:
                    self._perform_dmg_event(
                        source_cmp, ds_source, target_cmp, ds_target
                    )

        if self._asmnt.log.verbose:
            self.log.msg(
                'Damage process task successfully applied.', prepend_timestamp=False
            )

    def _perform_dmg_event(
        self, source_cmp: str, ds_source: int, target_cmp: str, ds_target: int
    ) -> None:
        """
        Perform a damage event.

        See `_perform_dmg_task`.
        """
        # affected rows
        assert self.ds_sample is not None
        row_selection = np.where(
            # for many instances of source_cmp, we
            # consider the highest damage state
            self.ds_sample[source_cmp].max(axis=1).to_numpy()  # type: ignore
            == ds_source
        )[0]
        # affected columns
        if target_cmp == 'ALL':
            column_selection = np.where(
                self.ds_sample.columns.get_level_values('cmp') != source_cmp
            )[0]
        else:
            column_selection = np.where(
                self.ds_sample.columns.get_level_values('cmp') == target_cmp
            )[0]
        self.ds_sample.iloc[row_selection, column_selection] = ds_target

    def _perform_dmg_event_loc(
        self, source_cmp: str, ds_source: int, target_cmp: str, ds_target: int
    ) -> None:
        """
        Perform a damage event matching locations.

        Parameters
        ----------
        source_cmp: str
            Source component, e.g., `'1_CMP_A'`. The number in the beginning
            is used to order the tasks and is not considered here.
        ds_source: int
            Source damage state.
        target_cmp: str
            Target component, e.g., `'CMP_B'`. The components that
            will be affected when `source_cmp` gets to `ds_source`.
        ds_target: int
            Target damage state, e.g., `'DS_1'`. The damage state that
            is assigned to `target_cmp` when `source_cmp` gets to
            `ds_source`.

        """
        # get locations of source component
        assert self.ds_sample is not None
        source_locs = set(self.ds_sample[source_cmp].columns.get_level_values('loc'))
        for loc in source_locs:
            # apply damage task matching locations
            row_selection = np.where(
                # for many instances of source_cmp, we
                # consider the highest damage state
                self.ds_sample[source_cmp, loc].max(axis=1).to_numpy() == ds_source
            )[0]

            # affected columns
            if target_cmp == 'ALL':
                column_selection = np.where(
                    np.logical_and(
                        self.ds_sample.columns.get_level_values('cmp') != source_cmp,
                        self.ds_sample.columns.get_level_values('loc') == loc,
                    )
                )[0]
            else:
                column_selection = np.where(
                    np.logical_and(
                        self.ds_sample.columns.get_level_values('cmp') == target_cmp,
                        self.ds_sample.columns.get_level_values('loc') == loc,
                    )
                )[0]
            self.ds_sample.iloc[row_selection, column_selection] = ds_target

    def complete_ds_cols(self, dmg_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Complete damage state columns.

        Completes the damage sample DataFrame with all possible damage
        states for each component.

        Parameters
        ----------
        dmg_sample: DataFrame
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
        dp = self.damage_params
        assert dp is not None

        # Get the header for the results that we can use to identify
        # cmp-loc-dir-uid sets
        dmg_header = (
            dmg_sample.groupby(  # type: ignore
                level=[0, 1, 2, 3],
                axis=1,
            )
            .first()
            .iloc[:2, :]
        )
        damaged_components = set(dmg_header.columns.get_level_values('cmp'))

        # get the number of possible limit states
        ls_list = [col for col in dp.columns.unique(level=0) if 'LS' in col]

        # initialize the result DataFrame
        res = pd.DataFrame()

        # TODO(JVM): For the code below, store the number of damage states
        # for each component ID as an attribute of the ds_model when
        # loading the parameters, and then directly access them here
        # much faster instead of parsing the parameters again.

        # walk through all components that have damage parameters provided
        for cmp_id in dp.index:
            # get the component-specific parameters
            cmp_data = dp.loc[cmp_id]

            # and initialize the damage state counter
            ds_count = 0

            # walk through all limit states for the component
            for ls in ls_list:
                # check if the given limit state is defined
                if not pd.isna(cmp_data[ls, 'Theta_0']):
                    # check if there is only one damage state
                    if pd.isna(cmp_data[ls, 'DamageStateWeights']):
                        ds_count += 1

                    else:
                        # or if there are more than one, how many
                        ds_count += len(
                            cmp_data[ls, 'DamageStateWeights'].split('|')
                        )

            # get the list of valid cmp-loc-dir-uid sets
            if cmp_id not in damaged_components:
                continue
            cmp_header = dmg_header.loc[:, [cmp_id]]

            # Create a DataFrame where they are repeated ds_count times in the
            # columns. The keys put the DS id in the first level of the
            # multiindexed column
            cmp_headers = pd.concat(
                [cmp_header for ds_i in range(ds_count + 1)],
                keys=[str(r) for r in range(ds_count + 1)],
                axis=1,
            )
            cmp_headers.columns.names = ['ds', *cmp_headers.columns.names[1::]]

            # add these new columns to the result DataFrame
            res = pd.concat([res, cmp_headers], axis=1)

        # Fill the result DataFrame with zeros and reorder its columns to have
        # the damage states at the lowest like - matching the dmg_sample input
        res = pd.DataFrame(
            0.0,
            columns=res.columns.reorder_levels([1, 2, 3, 4, 0]),  # type: ignore
            index=dmg_sample.index,
        )

        # replace zeros wherever the dmg_sample has results
        res.loc[:, dmg_sample.columns.to_list()] = dmg_sample

        return res


def _is_for_ds_model(data: pd.DataFrame) -> bool:
    """
    Check if data are for `ds_model`.

    Determines if the specified damage model parameters are for
    components modeled with discrete Damage States (DS).

    Parameters
    ----------
    data: pd.DataFrame
        The data to check.

    Returns
    -------
    bool
        If the data are for `ds_model`.

    """
    return 'LS1' in data.columns.get_level_values(0)
