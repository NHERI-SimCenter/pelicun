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


"""Loss model objects and associated methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from pelicun import base, file_io, uq
from pelicun.model.demand_model import (
    _assemble_required_demand_data,
    _get_required_demand_type,
    _verify_edps_available,
)
from pelicun.model.pelicun_model import PelicunModel
from pelicun.pelicun_warnings import InconsistentUnitsError

if TYPE_CHECKING:
    from collections.abc import Callable

    from pelicun.assessment import AssessmentBase

idx = base.idx


class LossModel(PelicunModel):
    """
    Manages loss information used in assessments.

    Contains a loss model for components with Damage States (DS) and
    one for components with Loss Functions (LF).

    """

    __slots__ = ['ds_model', 'dv_units', 'lf_model']

    def __init__(
        self,
        assessment: AssessmentBase,
        decision_variables: tuple[str, ...] = ('Cost', 'Time'),
        dv_units: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize LossModel objects.

        Parameters
        ----------
        assessment: pelicun.AssessmentBase
            Parent assessment
        decision_variables: tuple
            Defines the decision variables to be included in the loss
            calculations. Defaults to those supported, but fewer can be
            used if desired. When fewer are used, the loss parameters for
            those not used will not be required.

        """
        super().__init__(assessment)

        self.ds_model: RepairModel_DS = RepairModel_DS(assessment)
        self.lf_model: RepairModel_LF = RepairModel_LF(assessment)
        self._loss_map = None
        self.decision_variables = decision_variables
        self.dv_units = dv_units

    @property
    def sample(self) -> pd.DataFrame | None:
        """
        Combines the samples of the ds_model and lf_model sub-models.

        Returns
        -------
        pd.DataFrame
            The combined loss sample.

        """
        # Handle `None` cases

        if self.ds_model.sample is None and self.lf_model.sample is None:
            return None

        if self.ds_model.sample is None:
            return self.lf_model.sample

        if self.lf_model.sample is None:
            return self.ds_model.sample

        # If both are not None, combine

        ds_model_levels = self.ds_model.sample.columns.names

        # add a `ds` level to the lf_model sample
        new_index = self.lf_model.sample.columns.to_frame(index=False)
        # add
        new_index['ds'] = 'N/A'
        # reorder
        new_index = new_index[ds_model_levels]
        new_multiindex = pd.MultiIndex.from_frame(new_index)
        self.lf_model.sample.columns = new_multiindex

        return pd.concat((self.ds_model.sample, self.lf_model.sample), axis=1)

    @property
    def loss_params(self) -> pd.DataFrame | None:
        """
        Combines the loss params of the ds_model and lf_model sub-models.

        Returns
        -------
        pd.DataFrame
            The combined loss parameter table

        """
        # Handle `None` cases

        if self.ds_model.loss_params is None and self.lf_model.loss_params is None:
            return None

        if self.ds_model.loss_params is None:
            return self.lf_model.loss_params

        if self.lf_model.loss_params is None:
            return self.ds_model.loss_params

        # If both are not None, combine

        return pd.concat((self.ds_model.loss_params, self.lf_model.loss_params))

    @property
    def decision_variables(self) -> tuple[str, ...]:
        """
        Retrieve the decision variables.

        Returns
        -------
        tuple
            Decision variables.

        """
        # pick the object from one of the models
        # it's the same for the other(s).
        return self.ds_model.decision_variables

    @decision_variables.setter
    def decision_variables(self, decision_variables: tuple[str, ...]) -> None:
        """
        Set the decision variables.

        Supported: {`Cost`, `Time`, `Energy`, `Carbon`}.
        Could also be any other string, as long as the provided loss
        parameters contain that decision variable.

        """
        # assign the same DVs to the included loss models.
        for model in self._loss_models:
            model.decision_variables = decision_variables

    def add_loss_map(
        self,
        loss_map_path: str | pd.DataFrame | None = None,
        loss_map_policy: str | None = None,
    ) -> None:
        """
        Add a loss map to the loss model.

        A loss map defines what loss parameter definition should be
        used for each component ID in the asset model.

        Parameters
        ----------
        loss_map_path: str or pd.DataFrame or None
            Path to a csv file or DataFrame object that maps
            components IDs to their loss parameter definitions.
        loss_map_policy: str or None
            If None, does not modify the loss map.
            If set to `fill`, each component ID that is present in
            the asset model but not in the loss map is mapped to
            itself, but `excessiveRID` is excluded.
            If set to `fill_all`, each component ID that is present in
            the asset model but not in the loss map is mapped to
            itself without exceptions.


        Raises
        ------
        ValueError
            If both arguments are None.

        """
        self.log.msg('Loading loss map...')

        # If no loss map is provided and no default is requested,
        # there is no loss map and we can't proceed.
        if loss_map_path is None and loss_map_policy is None:
            msg = 'Please provide a loss map and/or a loss map extension policy.'
            raise ValueError(msg)

        # get a list of unique component IDs
        cmp_set = set(self._asmnt.asset.list_unique_component_ids())

        if loss_map_path is not None:
            self.log.msg('Loss map is provided.', prepend_timestamp=False)
            # Read the loss map into a variable
            loss_map = file_io.load_data(
                loss_map_path,
                None,
                orientation=1,
                reindex=False,
                log=self._asmnt.log,
            )
            assert isinstance(loss_map, pd.DataFrame)
            # <backwards compatibility>
            if np.any(['DMG' in x for x in loss_map.index]):  # type: ignore
                self.log.warning(
                    'The `DMG-` flag in the loss_map index is deprecated '
                    'and no longer necessary. '
                    'Please do not prepend `DMG-` before the component '
                    'names in the loss map.'
                )
                loss_map.index = pd.Index([x[1] for x in loss_map.index])

        else:
            self.log.msg('Using default loss map.', prepend_timestamp=False)
            # Instantiate an empty loss map.
            loss_map = pd.DataFrame({'Repair': pd.Series(dtype='object')})
            loss_map.index = loss_map.index.astype('object')

        if loss_map_policy in {'fill', 'fill_all'}:
            # Populate missing rows with cmp_id -> cmp_id
            for component in cmp_set:
                if component not in loss_map.index:
                    if loss_map_policy == 'fill' and component == 'excessiveRID':
                        continue
                    loss_map.loc[component, :] = component

        elif loss_map_policy is None:
            # Don't do anything.
            pass

        # TODO(AZ): add other loss map policies.
        else:
            msg = f'Unknown loss map policy: `{loss_map_policy}`.'
            raise ValueError(msg)

        # Assign the loss map to the available loss models
        self._loss_map = loss_map

        self.log.msg('Loss map loaded successfully.', prepend_timestamp=True)

    def load_model(
        self,
        data_paths: list[str | pd.DataFrame],
        loss_map: str | pd.DataFrame,
        decision_variables: tuple[str, ...] | None = None,
    ) -> None:
        """<backwards compatibility>."""
        self.log.warning(
            '`load_model` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `load_model_parameters` instead.'
        )
        self.add_loss_map(loss_map)
        self.load_model_parameters(data_paths, decision_variables)

    def load_model_parameters(
        self,
        data_paths: list[str | pd.DataFrame],
        decision_variables: tuple[str, ...] | None = None,
    ) -> None:
        """
        Load loss model parameters.

        Parameters
        ----------
        data_paths: list of (string | DataFrame)
            List of paths to data or files with loss model
            information. Default XY datasets can be accessed as
            PelicunDefault/XY. Order matters. Parameters defined in
            prior elements in the list take precedence over the same
            parameters in subsequent data paths. I.e., place the
            Default datasets in the back.
        decision_variables: tuple
            Defines the decision variables to be included in the loss
            calculations. Defaults to those supported, but fewer can be
            used if desired. When fewer are used, the loss parameters for
            those not used will not be required.

        """
        if decision_variables is not None:
            # <backwards compatibility>
            self.decision_variables = decision_variables
            self.log.warning(
                'The `decision_variables` argument has been removed. '
                'Please set your desired decision variables like so: '
                '{assessment object}.loss.decision_variables '
                "= ('dv1', 'dv2', ...) before calling "
                '{assessment object}.add_loss_map().'
            )

        self.log.div()
        self.log.msg('Loading loss parameters...')

        # replace `PelicunDefault/` flag with default data path
        data_paths = file_io.substitute_default_path(data_paths, log=self.log)

        #
        # load loss parameter data into the models
        #

        for data_path in data_paths:
            self._load_from_data_path(data_path)

        self.log.msg(
            'Loss model parameters loaded successfully.', prepend_timestamp=False
        )

        #
        # remove items
        #

        self.log.msg(
            'Removing unused loss model parameters.', prepend_timestamp=False
        )

        assert self._loss_map is not None
        for loss_model in self._loss_models:
            # drop unused loss parameter definitions
            loss_model.drop_unused_loss_parameters(self._loss_map)
            # remove components with incomplete loss parameters
            loss_model.remove_incomplete_components()

        # drop unused damage state columns
        self.ds_model.drop_unused_damage_states()

        #
        # obtain DV units
        #
        dv_units: dict = {}
        if self.ds_model.loss_params is not None:
            dv_units.update(
                self.ds_model.loss_params['DV', 'Unit']
                .groupby(level=[1])
                .first()
                .to_dict()
            )
        if self.lf_model.loss_params is not None:
            dv_units.update(
                self.lf_model.loss_params['DV', 'Unit']
                .groupby(level=[1])
                .first()
                .to_dict()
            )
        self.dv_units = dv_units

        #
        # convert units
        #

        self.log.msg(
            'Converting loss model parameter units.', prepend_timestamp=False
        )
        for loss_model in self._loss_models:
            loss_model.convert_loss_parameter_units()

        #
        # verify loss parameter availability
        #

        self.log.msg(
            'Checking loss model parameter '
            'availability for all components in the asset model.',
            prepend_timestamp=False,
        )
        self._ensure_loss_parameter_availability()

    def _load_from_data_path(self, data_path: str | pd.DataFrame) -> None:
        if 'bldg_repair_DB' in data_path:
            data_path = data_path.replace('bldg_repair_DB', 'loss_repair_DB')
            self.log.warning(
                '`bldg_repair_DB` is deprecated and will '
                'be dropped in future versions of pelicun. '
                'Please use `loss_repair_DB` instead.'
            )
        data = file_io.load_data(
            data_path, None, orientation=1, reindex=False, log=self._asmnt.log
        )
        assert isinstance(data, pd.DataFrame)

        # Check for unit consistency
        data.index.names = ['cmp', 'dv']
        units_isolated = data.reset_index()[[('dv', ''), ('DV', 'Unit')]]
        units_isolated.columns = pd.Index(['dv', 'Units'])
        units_isolated_grp = units_isolated.groupby('dv')['Units']
        unit_counts = units_isolated_grp.nunique()
        more_than_one = unit_counts[unit_counts > 1]
        if not more_than_one.empty:
            raise InconsistentUnitsError

        # determine if the loss model parameters are for damage
        # states or loss functions
        if _is_for_ds_model(data):
            self.ds_model.load_model_parameters(data)
        elif _is_for_lf_model(data):
            self.lf_model.load_model_parameters(data)
        else:
            msg = f'Invalid loss model parameters: {data_path}'
            raise ValueError(msg)

    def calculate(self) -> None:
        """
        Calculate the loss of each component block.

        Note: This method simply calculates the loss of each component
        block without any special treatment to `replacement`
        consequences. This can be done at a later step with the
        `aggregate_losses` method.

        Raises
        ------
        ValueError
            If the size of the demand sample and the damage sample
            don't match.

        """
        self.log.div()
        self.log.msg('Calculating losses...')

        # Get the damaged quantities in each damage state for each
        # component of interest.
        demand = self._asmnt.demand.sample
        assert demand is not None
        demand_offset = self._asmnt.options.demand_offset
        assert demand_offset is not None
        nondirectional_multipliers = self._asmnt.options.nondir_multi_dict
        assert nondirectional_multipliers is not None
        assert self._asmnt.asset.cmp_sample is not None
        cmp_sample = self._asmnt.asset.cmp_sample.to_dict('series')
        cmp_marginal_params = self._asmnt.asset.cmp_marginal_params
        assert cmp_marginal_params is not None
        if self._asmnt.damage.ds_model.sample is not None:
            # TODO(JVM): FIND A WAY to avoid making a copy of this.
            dmg_quantities = self._asmnt.damage.ds_model.sample.copy()
            if len(demand) != len(dmg_quantities):
                msg = (
                    f'The demand sample contains {len(demand)} realizations, '
                    f'but the damage sample contains {len(dmg_quantities)}. '
                    f'Loss calculation cannot proceed when '
                    f'these numbers are different. '
                )
                raise ValueError(msg)
            self.ds_model.calculate(dmg_quantities)

        self.lf_model.calculate(
            demand,
            cmp_sample,
            cmp_marginal_params,
            demand_offset,
            nondirectional_multipliers,
        )

        self.log.msg('Loss calculation successful.')

    def consequence_scaling(self, scaling_specification: str) -> None:
        """
        Apply scale factors to losses.

        Applies scale factors to the loss sample according to the
        given scaling specification. The scaling specification should
        be a path to a CSV file. It should contain a `Decision
        Variable` column with a specified decision variable in each
        row. Other optional columns are `Component`, `Location`,
        `Direction`. Each row acts as an independent scaling
        operation, with the scale factor defined in the `Scale Factor`
        column, which is required. If any value is missing in the
        optional columns, it is assumed that the scale factor should
        be applied to all entries of the loss sample where the other
        column values match. For example, if the specification has a
        single row with `Decision Variable` set to 'Cost', `Scale
        Factor` set to 2.0, and no other columns, this will double the
        'Cost' DV. If instead `Location` was also set to `1`, it would
        double the Cost of all components that have that location. The
        columns `Location` and `Direction` can contain ranges, like
        this: `1--3` means `1`, `2`, and `3`. If a range is used in
        both `Location` and `Direction`, the factor of that row will
        be applied once to all combinations.

        Parameters
        ----------
        scaling_specification: str
            Path to a CSV file containing the scaling specification.

        Raises
        ------
        ValueError
            If required columns are missing or contain NaNs.

        """
        # Specify expected dtypes from the start.
        dtypes = {
            'Decision Variable': 'str',
            'Component': 'str',
            'Location': 'str',
            'Direction': 'str',
            'Scale Factor': 'float64',
        }

        scaling_specification_df = pd.read_csv(scaling_specification, dtype=dtypes)

        if (
            'Decision Variable' not in scaling_specification_df.columns
            or scaling_specification_df['Decision Variable'].isna().any()
        ):
            msg = (
                'The `Decision Variable` column is missing '
                'from the scaling specification or contains NaN values.'
            )
            raise ValueError(msg)
        if (
            'Scale Factor' not in scaling_specification_df.columns
            or scaling_specification_df['Scale Factor'].isna().any()
        ):
            msg = (
                'The `Scale Factor` column is missing '
                'from the scaling specification or contains NaN values.'
            )
            raise ValueError(msg)

        # Add missing optional columns with NaN values
        optional_cols = ['Component', 'Location', 'Direction']
        for col in optional_cols:
            if col not in scaling_specification_df.columns:
                scaling_specification_df[col] = np.nan

        # Rename the columns to the internally used values
        name_map = {
            'Decision Variable': 'dv',
            'Component': 'dmg',
            'Location': 'loc',
            'Direction': 'dir',
            'Scale Factor': 'scaling',
        }
        scaling_specification_df = scaling_specification_df.rename(columns=name_map)

        # Expand ranges in 'loc' and 'dir'
        def _expand_range(col):  # noqa: ANN001, ANN202
            if pd.isna(col):
                return [col]
            if '--' in col:
                start, end = (int(x) for x in col.split('--'))
                return [str(x) for x in range(start, end + 1)]
            return [col]

        # Generate all combinations of loc and dir ranges
        expanded_df = scaling_specification_df.apply(  # type: ignore
            lambda row: pd.DataFrame(
                list(product(_expand_range(row['loc']), _expand_range(row['dir']))),
                columns=['loc', 'dir'],
            ).assign(dv=row['dv'], dmg=row['dmg'], scaling=row['scaling']),
            axis=1,
        )

        expanded_df = pd.concat(expanded_df.values)

        # Now, for each unique combination in expanded_df, apply
        # consequence scaling
        for _, row in expanded_df.iterrows():
            scaling_conditions = {
                k: row[k] for k in ('dv', 'dmg', 'loc', 'dir') if not pd.isna(row[k])
            }
            self._apply_consequence_scaling(
                scaling_conditions, row['scaling'], raise_missing=False
            )

    def _apply_consequence_scaling(
        self,
        scaling_conditions: dict,
        scale_factor: float,
        *,
        raise_missing: bool = True,
    ) -> None:
        """
        Apply a scale factor to selected loss sample columns.

        The scaling conditions are passed as a dictionary mapping
        level names with their required value for the condition to be
        met. It has to contain `dv` as one of its keys, defining the
        decision variable where the factors should be applied. Other
        valid levels include:
        - `dmg`: containing a source component name,
        - `loc`: containing a location,
        - `dir`: containing a direction,
        - `uid`: containing a Unique Component ID (UID).

        If any of the keys is missing, it is assumed that the scaling
        factor should be applied to all relevant consequences (those
        matching the remaining values of the hierarchical index).

        Parameters
        ----------
        scaling_conditions: dict
            A dictionary mapping level names with a single value. Only the
            rows where the index levels have the provided values will be
            affected. The dictionary can be empty, in which case all rows
            will be affected, or contain only some levels and values, in
            which case only the matching rows will be affected.
        scale_factor: float
            Scale factor to use.
        raise_missing: bool
            Raise an error if no rows are matching the given conditions.

        Raises
        ------
        ValueError
            If the scaling_conditions dictionary does not contain a
            `dv` key.

        """
        # make sure we won't apply the same factor to all DVs at once,
        # highly unlikely anyone would actually want to do this.
        if 'dv' not in scaling_conditions:
            msg = (
                'The index of the `scaling_conditions` dictionary '
                'should contain a level named `dv` listing the '
                'relevant decision variable.'
            )
            raise ValueError(msg)

        for model in self._loss_models:
            # check if it's empty
            if model.sample is None:
                continue

            # ensure the levels exist (but don't check if specified
            # values exist yet)
            for name in scaling_conditions:
                if name not in model.sample.columns.names:
                    msg = (
                        f'`scaling_conditions` contains an unknown level: `{name}`.'
                    )
                    raise ValueError(msg)

            # apply scale factors
            base.multiply_factor_multiple_levels(
                model.sample,
                scaling_conditions,
                scale_factor,
                axis=1,
                raise_missing=raise_missing,
            )

    def save_sample(
        self, filepath: str | None = None, *, save_units: bool = False
    ) -> None | pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Save a combined sample of the `ds_model` and `lf_model`.

        This method handles the storage of a sample of loss estimates,
        which can either be saved directly to a file or returned as a
        DataFrame for further manipulation. When saving to a file,
        additional information such as unit conversion factors and
        column units can be included. If the data is not being saved
        to a file, the method can return the DataFrame with or without
        units as specified.

        Parameters
        ----------
        filepath: str, optional
            The path to the file where the loss sample should be
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
            * DataFrame containing the loss sample.
            * Optionally, a Series containing the units for each
            column if `save_units` is True.

        """
        self.log.div()
        if filepath is not None:
            self.log.msg('Saving loss sample...')

        assert self.sample is not None
        assert self.loss_params is not None

        cmp_units = self.loss_params['DV', 'Unit'].sort_index()
        dv_units = pd.Series(
            index=self.sample.columns, name='Units', dtype='object'
        ).sort_index()

        valid_dv_types = dv_units.index.unique(level=0)
        valid_cmp_ids = dv_units.index.unique(level=1)

        for cmp_id, dv_type in cmp_units.index:
            if (dv_type in valid_dv_types) and (cmp_id in valid_cmp_ids):
                dv_units.loc[dv_type, cmp_id] = cmp_units.loc[cmp_id, dv_type]

        res = file_io.save_to_csv(
            self.sample,
            Path(filepath) if filepath is not None else None,
            units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )
        if filepath is not None:
            self.log.msg('Loss sample successfully saved.', prepend_timestamp=False)
            return None

        assert isinstance(res, pd.DataFrame)

        units = res.loc['Units']
        res = res.drop('Units')
        res = res.astype(float)
        assert isinstance(res, pd.DataFrame)
        assert isinstance(units, pd.Series)

        if save_units:
            return res, units

        return res

    def load_sample(self, filepath: str | pd.DataFrame) -> None:
        """
        <backwards compatibility>.

        Saves the sample of the `ds_model`.

        """
        self.log.warning(
            '`{loss model}.load_sample` is deprecated and will raise '
            'in future versions of pelicun. Please use '
            '{loss model}.ds_model.load_sample instead.'
        )
        dv_units = self.ds_model.load_sample(filepath=filepath)
        self.dv_units = dv_units

    def aggregate_losses(  # noqa: C901
        self,
        replacement_configuration: (
            tuple[uq.RandomVariableRegistry, dict[str, float]] | None
        ) = None,
        loss_combination: dict | None = None,
        *,
        future: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate the losses produced by each component.

        Parameters
        ----------
        replacement_configuration: Tuple, optional
            Tuple containing a RandomVariableRegistry and a
            dictionary. The RandomVariableRegistry is defining
            building replacement consequence RVs for the active
            decision variables. The dictionary defines exceedance
            thresholds. If the aggregated value for a decision
            variable (conditioned on no replacement) exceeds the
            threshold, then replacement is triggered. This can happen
            for multiple decision variables at the same
            realization. The consequence keyword `replacement` is
            reserved to represent exclusive triggering of the
            replacement consequences, and other consequences are
            ignored for those realizations where replacement is
            triggered. When assigned to None, then `replacement` is
            still treated as an exclusive consequence (other
            consequences are set to zero when replacement is nonzero)
            but it is not being additionally triggered by the
            exceedance of any thresholds. The aggregated loss sample
            contains an additional column with information on whether
            replacement was already present or triggered by a
            threshold exceedance for each realization.
        loss_combination: dict, optional
            Dictionary defining how losses for specific components
            should be aggregated for a given decision variable. It has
            the following structure: {`dv`: {(`c1`, `c2`): `arr`,
            ...}, ...}, where `dv` is some decision variable, (`c1`,
            `c2`) is a tuple defining a component pair, `arr` is a NxN
            numpy array defining a combination table, and `...` means
            that more key-value pairs with the same schema can exist
            in the dictionaries.  The loss sample is expected to
            contain columns that include both `c1` and `c2` listed as
            the component. The combination is applied to all pairs of
            columns where the components are `c1` and `c2`, and all of
            the rest of the multiindex levels match (`loc`, `dir`,
            `uid`). This means, for example, that when combining wind
            and flood losses, the asset model should contain both a
            wind and a flood component defined at the same
            location-direction.  `arr` can also be an M-dimensional
            numpy array where each dimension has length N (NxNx...xN).
            This structure allows for the loss combination of M
            components.  In this case the (`c1`, `c2`) tuple should
            contain M elements instead of two.
        future: bool, optional
            Defaults to False. When set to True, it enables the
            updated return type.

        Notes
        -----
        Regardless of the value of the arguments, this method does not
        alter the state of the loss model, i.e., it does not modify
        the values of the `.sample` attributes.

        Returns
        -------
        dataframe or tuple
            Dataframe with the aggregated loss of each realization,
            and another boolean dataframe with information on which DV
            thresholds were exceeded in each realization, triggering
            replacement. If no thresholds are specified it only
            contains False values. The second dataframe is only
            returned with `future` set to True.

        """
        # TODO(JVM): When we start working on the documentation,
        # simplify the docstring above and point the relevant detailed
        # section in the documentation.

        # validate input
        if replacement_configuration is not None:
            self._validate_input_replacement_thresholds(replacement_configuration)
        # validate loss_combination input
        if loss_combination is not None:
            self._validate_input_loss_combination(loss_combination)

        #
        # operate on copes of the loss samples to avoid altering them.
        #

        if self.ds_model.sample is not None:
            ds_sample = self.ds_model.sample.copy()
        else:
            ds_sample = None
        if self.lf_model.sample is not None:
            lf_sample = self.lf_model.sample.copy()
        else:
            lf_sample = None

        def _construct_columns() -> list[str]:
            columns = [
                f'repair_{x.lower()}' for x in self.decision_variables if x != 'Time'
            ]
            # Note: The `Time` DV gets special treatment.
            # create the summary DF
            if 'Time' in self.decision_variables:
                columns.extend(('repair_time-sequential', 'repair_time-parallel'))
            return columns

        if ds_sample is None and lf_sample is None:
            self.log.msg('There are no losses.')
            return pd.DataFrame(0.00, index=[0], columns=_construct_columns())

        #
        #  handle `replacement`, regardless of whether
        # `replacement_thresholds` is empty. (if `replacement`
        # occurs, we ignore the losses from other componnets)
        #

        if ds_sample is not None:
            self._make_replacement_exclusive(ds_sample, lf_sample)

        #
        # combine samples
        #

        # levels to preserve (this aggregates `ds` for the ds_model)
        column_levels = ['dv', 'loss', 'dmg', 'loc', 'dir', 'uid']
        combined_sample = self.sample
        sample = (
            combined_sample.groupby(level=column_levels, axis=1)  # type: ignore
            .sum()
            .sort_index(axis=1)
        )

        #
        # perform loss combinations (special non-additive
        # aggregations, e.g., Wind + Flood)
        #

        if loss_combination is not None:
            sample = self._apply_loss_combinations(loss_combination, sample)

        #
        # consider replacement threshold values
        #

        sample, exceedance_bool_df = self._apply_replacement_thresholds(
            sample, replacement_configuration
        )

        # Sum-up component losses
        df_agg = self._aggregate_sample(sample, _construct_columns())

        if not future:
            self.log.warning(
                '`aggregate_losses` has been expanded to support the '
                'consideration of the exceedance of loss threshold '
                'values leading to asset replacement '
                '(like excessive repair costs). The new implementation '
                'returns a tuple where the first element is the '
                'aggregated losses and the second contains information '
                'on which decision variables triggered replacement '
                'considering the specified replacement trhesholds. '
                'To obtain the new output and silence this warning, '
                'please specify `future=True` as an argument to this method.'
            )
            return df_agg

        return df_agg, exceedance_bool_df

    def _validate_input_loss_combination(self, loss_combination: dict) -> None:
        for dv, combinations in loss_combination.items():
            if dv not in self.decision_variables:
                msg = (
                    f'`loss_combination` contains the key '
                    f'`{dv}` which is not found in the active '
                    f'decision variables. These are: '
                    f'{self.decision_variables}.'
                )
                raise ValueError(msg)
            for components, array in combinations.items():
                if not isinstance(components, tuple):
                    msg = (
                        f'Invalid type for components in loss combination '
                        f'for `{dv}`: {type(components)}. It should be a tuple.'
                    )
                    raise TypeError(msg)
                if not all(isinstance(c, str) for c in components):
                    msg = (
                        f'All elements of the components tuple in loss '
                        f'combination for `{dv}` should be strings.'
                    )
                    raise ValueError(msg)
                if not isinstance(array, np.ndarray):
                    msg = (
                        f'Invalid type for array in loss combination '
                        f'for `{dv}`: {type(array)}. It should be a numpy array.'
                    )
                    raise TypeError(msg)

    def _validate_input_replacement_thresholds(
        self,
        replacement_configuration: tuple[
            uq.RandomVariableRegistry, dict[str, float]
        ],
    ) -> None:
        replacement_consequence_rv_reg, replacement_ratios = (
            replacement_configuration
        )
        if not isinstance(replacement_consequence_rv_reg, uq.RandomVariableRegistry):
            msg = (
                f'Invalid type for replacement consequence RV registry: '
                f'{type(replacement_consequence_rv_reg)}. It should be '
                f'uq.RandomVariableRegistry.'
            )
            raise TypeError(msg)
        for key in replacement_consequence_rv_reg.RV:
            if key not in self.decision_variables:
                msg = (
                    f'`replacement_consequence_RV_reg` contains the key '
                    f'`{key}` which is not found in the active '
                    f'decision variables. These are: '
                    f'{self.decision_variables}.'
                )
                if self.query_error_setup(
                    'Loss/ReplacementThreshold/RaiseOnUnknownKeys'
                ):
                    raise ValueError(msg)
                self.log.warning(msg)

        for key in replacement_ratios:
            if key not in self.decision_variables:
                msg = (
                    f'`replacement_ratios` contains the key '
                    f'`{key}` which is not found in the active '
                    f'decision variables. These are: '
                    f'{self.decision_variables}.'
                )
                if self.query_error_setup(
                    'Loss/ReplacementThreshold/RaiseOnUnknownKeys'
                ):
                    raise ValueError(msg)
                self.log.warning(msg)
        # The replacement_consequence_RV_reg should contain an RV for
        # all active DVs, regardless of whether there is a replacement
        # threshold for that DV, because when replacememnt is
        # triggered, we need to assign a consequence for all DVs.
        for key in self.decision_variables:
            if key not in replacement_consequence_rv_reg.RV:
                msg = f'Missing replacement consequence RV ' f'for `{key}`.'
                raise ValueError(msg)

    def _apply_loss_combinations(
        self, loss_combination: dict, sample: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform loss combinations of specified components.

        This function deconstructs the loss combination arrays,
        identifies the combinable components, and applies the
        specified loss combinations to the sample data. The
        transformed sample, including the combined columns, is
        returned as a new DataFrame.

        Parameters
        ----------
        loss_combination: dict
            A dictionary containing the loss combination
            information. The structure is nested dictionaries where
            the outer keys are decision variables, inner keys are
            components to combine, and the values are array objects
            representing the combination data.

        sample: pandas.DataFrame
            The input DataFrame containing the sample data. The
            columns are assumed to be a MultiIndex with at least the
            levels (decision_variable, loss_id, component_id,
            location, direction, uid).

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with the combined loss data.

        """
        # deconstruct combination arrays to extract the input domains
        loss_combination_converted = self._deconstruct_loss_combination_arrays(
            loss_combination
        )

        # initialize variables

        # sample as dictionary for fast column retrieval
        dsample = {col: sample[col] for col in sample.columns}

        # Transformed sample (includes the combined columns), as
        # dictionary. Will be turned into a dataframe in the end.
        # This avoids manipulating the original sample dataframe which
        # would be slow.
        dcsample: dict = {}

        # add columns to the new sample dictionary.
        # those that should be combined
        self._loss_combination_add_combinable(
            dsample, loss_combination_converted, dcsample
        )
        # and the remaining
        for col, val in dsample.items():
            dcsample[col] = val  # noqa: PERF403

        # turn into a dataframe
        return pd.DataFrame(dcsample).rename_axis(columns=sample.columns.names)

    def _loss_combination_add_combinable(
        self, dsample: dict, loss_combination_converted: dict, dcsample: dict
    ) -> None:
        """
        Add combinable loss data.

        This function identifies groups of `loc`-`dir`-`uid` that can
        be combined for each decision variable and computes the
        combined loss using interpolation functions. It modifies the
        given datasets `dsample` and `dcsample` in-place, removing
        combinable columns from dsample and adding the combined losses
        to dcsample.

        Parameters
        ----------
        dsample: dict
            A dictionary representing the loss sample data, where keys
            are tuples of the form (decision_variable, loss_id,
            component_id, location, direction, uid) and values are the
            corresponding data arrays.

        loss_combination_converted: dict
            A dictionary containing loss combination data. The
            structure is nested dictionaries where the outer keys are
            decision variables, inner keys are components to combine,
            and the values are tuples of combination parameters
            (domains and reference values).

        dcsample: dict
            A dictionary to store the combined loss data, where keys
            are tuples of the form (decision_variable, 'combination',
            combined_component_string, location, direction, uid) and
            values are the combined loss data arrays.

        """
        dmg_to_loss = self._map_component_ids_to_loss_ids(dsample)

        # identify all `loc`-`dir`-`uid`s that can be grouped for each
        # decision variable.
        potential_groups = self._identify_potential_groups(dsample)

        # cache already defined interpolation functions. This obviates
        # the need to define all of them and we can just define them
        # on the spot when needed, and reuse them if available.
        interpolation_function_cache: dict = {}

        for (
            decision_variable,
            combination_data,
        ) in loss_combination_converted.items():
            for (
                components_to_combine,
                combination_parameters,
            ) in combination_data.items():
                # determine if the components to combine are part of
                # an available group
                target_group: dict | None = None
                for available_group in potential_groups[decision_variable]:
                    # check if `components_to_combine` is a subset of
                    # that available group
                    if frozenset(components_to_combine) <= available_group:
                        target_group = available_group
                        break
                assert target_group is not None
                # construct relevant loss sample columns
                for loc_dir_uid in potential_groups[decision_variable][target_group]:
                    cols = [
                        (decision_variable, dmg_to_loss[x], x, *loc_dir_uid)
                        for x in target_group
                    ]
                    values = np.column_stack([dsample[col] for col in cols])
                    # define/get interpolation function
                    if (
                        interpolation_function_cache.get(components_to_combine)
                        is not None
                    ):
                        interp_func = interpolation_function_cache.get(
                            components_to_combine
                        )
                    else:
                        domains, reference_values = combination_parameters
                        interp_func = RegularGridInterpolator(
                            domains, reference_values
                        )
                    assert interp_func is not None
                    combined_loss = interp_func(values)
                    combined_loss_col = (
                        decision_variable,
                        'combination',
                        '(' + ', '.join(components_to_combine) + ')',
                        *loc_dir_uid,
                    )
                    dcsample[combined_loss_col] = combined_loss
                    for col in cols:
                        dsample.pop(col)

    def _identify_potential_groups(self, dsample: dict) -> dict:  # noqa: PLR6301
        """
        Identify potential groups of `loc`-`dir`-`uid` for each DV.

        This function identifies all combinations of `loc`-`dir`-`uid`
        that can be grouped for each decision variable based on the
        provided data sample.

        Parameters
        ----------
        dsample: iterable
            An iterable where each containing tuple contains
            information about the components and their attributes. The
            expected format of each tuple is (decision_variable,
            loss_id, component_id, location, direction, uid).

        Returns
        -------
        dict
            A dictionary where keys are decision variables and values
            are nested dictionaries. The nested dictionaries map
            frozensets of component IDs to lists of (location,
            direction, uid) tuples.

        """
        grouped: defaultdict = defaultdict(defaultdict(list).copy)
        for col in dsample:
            c_dv, _, c_dmg, c_loc, c_dir, c_uid = col
            grouped[c_dv][c_loc, c_dir, c_uid].append(c_dmg)
        # invert so that we have component sets mapped to
        # `loc`-`dir`-`uid`s.
        inverted: defaultdict = defaultdict(defaultdict(list).copy)
        for c_dv in grouped:
            for loc_dir_uid, component_set in grouped[c_dv].items():
                inverted[c_dv][frozenset(component_set)].append(loc_dir_uid)
        return inverted

    def _map_component_ids_to_loss_ids(self, dsample: dict) -> dict:  # noqa: PLR6301
        """
        Map component IDs to loss IDs.

        This function maps components to losses based on the loss
        sample's columns. It assumes that multiple component IDs can
        have the same loss ID, but the same component ID cannot have
        multiple loss IDs.

        Parameters
        ----------
        dsample: tuple dictionary keys
            Each tuple contains information about the components and
            corresponding losses.

        Returns
        -------
        dict
            A dictionary where keys are component IDs and values are
            loss IDs.

        """
        dmg_to_loss = {}
        for col in dsample:
            c_loss = col[1]
            c_dmg = col[2]
            dmg_to_loss[c_dmg] = c_loss
        return dmg_to_loss

    def _deconstruct_loss_combination_arrays(self, loss_combination: dict) -> dict:  # noqa: PLR6301
        """
        Deconstruct loss combination arrays.

        This function converts a nested dictionary of loss combination
        arrays into a format suitable for further processing. It
        extracts the array values and the index information from each
        array.

        Parameters
        ----------
        loss_combination: dict
           A dictionary where keys are decision variables and values
           are another dictionary. The inner dictionary has keys as
           components to combine and values as numpy array
           objects representing the combination data.

        Returns
        -------
        dict
            A dictionary with the same structure as the input
            `loss_combination`.  For each decision variable and
            component combination, the array is replaced with a
            tuple containing the combination domain and the combination
            array itself.

        """
        loss_combination_converted: dict = {}
        for decision_variable, combination_data in loss_combination.items():
            loss_combination_converted[decision_variable] = {}
            for (
                components_to_combine,
                combination_array,
            ) in combination_data.items():
                combination_index = (
                    combination_array[:, 0],
                    combination_array[0, :],
                )
                loss_combination_converted[decision_variable][
                    components_to_combine
                ] = (
                    combination_index,
                    combination_array,
                )
        return loss_combination_converted

    def _aggregate_sample(self, sample: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Sum up component losses.

        Returns
        -------
        pd.DataFrame
            Dataframe with the aggregated losses.

        """
        df_agg = pd.DataFrame(index=sample.index, columns=columns)
        # group results by DV type and location
        aggregated = sample.groupby(
            level=['dv', 'loc'],
            axis=1,  # type: ignore
        ).sum()

        for decision_variable in self.decision_variables:
            # Time
            if (
                decision_variable == 'Time'
                and 'Time' in aggregated.columns.get_level_values('dv')
            ):
                df_agg['repair_time-sequential'] = aggregated['Time'].sum(axis=1)

                df_agg['repair_time-parallel'] = aggregated['Time'].max(axis=1)
            elif (
                decision_variable == 'Time'
                and 'Time' not in aggregated.columns.get_level_values('dv')
            ):
                df_agg = df_agg.drop(
                    ['repair_time-parallel', 'repair_time-sequential'], axis=1
                )
            # All other
            elif decision_variable in aggregated.columns.get_level_values('dv'):
                df_agg[f'repair_{decision_variable.lower()}'] = aggregated[
                    decision_variable
                ].sum(axis=1)
            else:
                df_agg = df_agg.drop(f'repair_{decision_variable.lower()}', axis=1)

        # Convert units ..
        column_measures = [
            x.replace('repair_', '')
            .replace('-sequential', '')
            .replace('-parallel', '')
            for x in df_agg.columns.get_level_values(0)
        ]
        assert self.dv_units is not None
        column_units = [self.dv_units[x.title()] for x in column_measures]
        dv_units = pd.Series(column_units, index=df_agg.columns, name='Units')
        res = file_io.save_to_csv(
            df_agg,
            None,
            units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=False,
            log=self._asmnt.log,
        )
        assert isinstance(res, pd.DataFrame)
        df_agg = res
        df_agg = df_agg.drop('Units')
        df_agg = df_agg.astype(float)

        df_agg_mi = base.convert_to_MultiIndex(df_agg, axis=1)
        assert isinstance(df_agg_mi, pd.DataFrame)
        df_agg = df_agg_mi
        df_agg = df_agg.sort_index(axis=1)
        df_agg = df_agg.reset_index(drop=True)
        assert isinstance(df_agg, pd.DataFrame)
        return df_agg

    def _apply_replacement_thresholds(  # noqa: PLR6301
        self,
        sample: pd.DataFrame,
        replacement_configuration: (
            tuple[uq.RandomVariableRegistry, dict[str, float]] | None
        ),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # If there is no `replacement_configuration`, simply return.
        if replacement_configuration is None:
            # `exceedance_bool_df` is empty in this case.
            exceedance_bool_df = pd.DataFrame(index=sample.index, dtype=bool)
            return sample, exceedance_bool_df

        replacement_consequence_rv_reg, replacement_ratios = (
            replacement_configuration
        )
        exceedance_bool_df = pd.DataFrame(  # type: ignore
            data=False,
            index=sample.index,
            columns=replacement_consequence_rv_reg.RV.keys(),
            dtype=bool,
        )

        # Sample replacement consequences from the registry
        replacement_consequence_rv_reg.generate_sample(len(sample), 'MonteCarlo')

        sample_dvs = replacement_consequence_rv_reg.RV.keys()
        for sample_dv in sample_dvs:
            sub_sample = sample.loc[:, sample_dv]
            if 'replacement' in sub_sample.columns.get_level_values('loss'):
                # If `replacement` already exists as a consequence,
                # determine the realizations where it is non-zero.
                no_replacement_mask = (
                    ~(sub_sample['replacement'] > 0.00).any(axis=1).to_numpy()
                )
                no_replacement_columns = (
                    sub_sample.columns.get_level_values('loss') != 'replacement'
                )
            else:
                # Otherwise there is no row where we already have replacement
                no_replacement_mask = np.full(len(sub_sample), fill_value=True)
                no_replacement_columns = np.full(
                    len(sub_sample.columns), fill_value=True
                )
            # Get the sum to compare with the thresholds
            consequence_sum_given_no_replacement = sub_sample.iloc[  # type: ignore
                no_replacement_mask, no_replacement_columns
            ].sum(axis=1)
            if not consequence_sum_given_no_replacement.empty:
                consequence_values = replacement_consequence_rv_reg.RV[
                    sample_dv
                ].sample
                assert consequence_values is not None
                exceedance_mask = (
                    consequence_sum_given_no_replacement
                    > consequence_values[no_replacement_mask]
                    * replacement_ratios[sample_dv]
                )
                # Fill the remaining rows with False
                if len(exceedance_mask) < len(sub_sample):
                    exceedance_mask = exceedance_mask.reindex(
                        sub_sample.index, fill_value=False
                    )
            else:
                exceedance_mask = pd.Series(
                    np.full(len(sub_sample), fill_value=False),
                    index=sub_sample.index,
                )

            # Monitor triggering of replacement
            exceedance_bool_df[sample_dv] = exceedance_mask

        # Assign replacement consequences where the threshold has been
        # exceeded.
        exceedance_realizations = exceedance_bool_df.any(axis=1)
        # Assign replacement consequences: needs to include all DVs
        for other_dv in replacement_consequence_rv_reg.RV:
            col = (
                other_dv,
                'replacement',
                'threshold_exceedance',
                '0',
                '1',
                '0',
            )
            # If it doesn't exist, initialize and set to 0.00
            if col not in sample:
                sample[col] = 0.00
                sample = sample.sort_index(axis=1)
            # Assign replacement consequences
            other_sample = replacement_consequence_rv_reg.RV[other_dv].sample
            assert other_sample is not None
            sample.loc[exceedance_realizations, col] = other_sample[
                exceedance_realizations
            ]
        # Remove all other realized consequences from the realizations
        # where the threshold was exceeded.
        sample.loc[
            exceedance_realizations,
            sample.columns.get_level_values('dmg') != 'threshold_exceedance',
        ] = 0.00

        return sample, exceedance_bool_df

    def _make_replacement_exclusive(  # noqa: PLR6301
        self, ds_sample: pd.DataFrame, lf_sample: pd.DataFrame | None
    ) -> None:
        """
        Make the replacement consequence exclusive.

        If `replacement` columns exist in `ds_sample`, this method
        treats all nonzero loss values driven by `replacement` as
        exclusive and zeroes-out the loss values of all other columns
        for the applicable rows.
        """
        # rows where replacement is non-zero
        replacement_rows: list = []

        # columns that correspond to the replacement consequence
        replacement_columns = (
            ds_sample.columns.get_level_values('loss') == 'replacement'
        )
        rows_df = ds_sample.iloc[:, replacement_columns]

        if not rows_df.empty:
            replacement_rows = (
                np.argwhere(np.any(rows_df.to_numpy() > 0.0, axis=1))
                .reshape(-1)
                .tolist()
            )
        ds_sample.iloc[replacement_rows, ~replacement_columns] = 0.00  # type: ignore
        if lf_sample is not None:
            lf_sample.iloc[replacement_rows, :] = 0.00

    @property
    def _loss_models(self) -> tuple[RepairModel_DS, RepairModel_LF]:
        return (self.ds_model, self.lf_model)

    @property
    def _loss_map(self) -> pd.DataFrame | None:
        """
        Returns the loss map.

        Returns
        -------
        pd.DataFrame
            The loss map.

        """
        # Retrieve the DataFrame from one of the included loss models.
        # We use a single loss map for all.
        return self.ds_model.loss_map

    @_loss_map.setter
    def _loss_map(self, loss_map: pd.DataFrame) -> None:
        """
        Set the loss map.

        Parameters
        ----------
        loss_map: pd.DataFrame
            The loss map.

        """
        # Add the DataFrame to the included loss models.
        # We use a single loss map for all.
        for model in self._loss_models:
            model.loss_map = loss_map

    @property
    def _missing(self) -> set[tuple[str, str]]:
        """
        Returns the missing components.

        Returns
        -------
        set
            Set containing tuples identifying missing loss parameter
            definitions.

        """
        return self.ds_model.missing

    @_missing.setter
    def _missing(self, missing: set[tuple[str, str]]) -> None:
        """
        Assign missing parameter definitions to the loss models.

        Parameters
        ----------
        missing: set
            Set containing tuples identifying missing loss parameter
            definitions.

        """
        for model in self._loss_models:
            model.missing = missing

    def _ensure_loss_parameter_availability(self) -> None:
        """Make sure that all components have loss parameters."""
        #
        # Repair Models (currently the only type supported)
        #

        required = []
        assert self._loss_map is not None
        for dv in self.decision_variables:
            required.extend(
                [(component, dv) for component in self._loss_map['Repair']]
            )
        missing_set = set(required)

        for model in (self.ds_model, self.lf_model):
            missing_set -= model.get_available()

        if missing_set:
            self.log.warning(
                f'The loss model does not provide '
                f'loss information for the following component(s) '
                f'in the asset model: {sorted(missing_set)}.'
            )

        self._missing = missing_set


class RepairModel_Base(PelicunModel):
    """Base class for loss models."""

    __slots__ = [
        'consequence',
        'decision_variables',
        'loss_map',
        'loss_params',
        'missing',
        'sample',
    ]

    def __init__(self, assessment: AssessmentBase) -> None:
        """
        Initialize RepairModel_Base objects.

        Parameters
        ----------
        assessment: pelicun.AssessmentBase
            Parent assessment

        """
        super().__init__(assessment)

        self.loss_params: pd.DataFrame | None = None
        self.sample: pd.DataFrame | None = None
        self.consequence = 'Repair'
        self.decision_variables: tuple[str, ...] = ()
        self.loss_map: pd.DataFrame | None = None
        self.missing: set = set()

    def load_model_parameters(self, data: pd.DataFrame) -> None:
        """
        Load model parameters from a DataFrame.

        Extends those already available. Parameters already defined
        take precedence, i.e. redefinitions of parameters are ignored.

        Parameters
        ----------
        data: DataFrame
            Data with loss model information.

        """
        data.index.names = ['Loss Driver', 'Decision Variable']

        if self.loss_params is not None:
            data = pd.concat((self.loss_params, data), axis=0)

        # drop redefinitions of components
        data = (
            data.groupby(level=[0, 1]).first().transform(lambda x: x.fillna(np.nan))
        )
        # note: .groupby introduces None entries. We replace them with
        # NaN for consistency.

        self.loss_params = data

    def drop_unused_loss_parameters(self, loss_map: pd.DataFrame) -> None:
        """
        Remove loss parameter definitions.

        Applicable to component IDs not present in the loss map.

        Parameters
        ----------
        loss_map: str or pd.DataFrame or None
            Path to a csv file or DataFrame object that maps
            components IDs to their loss parameter definitions.
            Components in the asset model that are omitted from the
            provided loss map are mapped to the same ID.


        """
        if self.loss_params is None:
            return

        # <backwards compatibility>
        if 'BldgRepair' in loss_map.columns:
            loss_map['Repair'] = loss_map['BldgRepair']
            loss_map = loss_map.drop('BldgRepair', axis=1)
            self.log.warning(
                '`BldgRepair` as a loss map column name is '
                'deprecated and will be dropped in '
                'future versions of pelicun. Please use `Repair` instead.'
            )

        # get a list of unique component IDs
        cmp_set = set(loss_map['Repair'].unique())

        cmp_mask = self.loss_params.index.get_level_values(0).isin(cmp_set, level=0)
        self.loss_params = self.loss_params.iloc[cmp_mask, :]

    def remove_incomplete_components(self) -> None:
        """
        Remove incomplete components.

        Removes components that have incomplete loss model
        definitions from the loss model parameters.

        """
        if self.loss_params is None:
            return

        if ('Incomplete', '') not in self.loss_params.columns:
            return

        cmp_incomplete_idx = self.loss_params.loc[
            self.loss_params['Incomplete', ''] == 1
        ].index

        self.loss_params = self.loss_params.drop(cmp_incomplete_idx)

        if len(cmp_incomplete_idx) > 0:
            self.log.msg(
                f'\n'
                f'WARNING: Loss model information is incomplete for '
                f'the following component(s) '
                f'{cmp_incomplete_idx.to_list()}. They '
                f'were removed from the analysis.'
                f'\n',
                prepend_timestamp=False,
            )

    def get_available(self) -> set:
        """
        Get a set of components with available loss parameters.

        Returns
        -------
        set
            Set of components with available loss parameters.
        """
        if self.loss_params is not None:
            cmp_list = self.loss_params.index.to_list()
            return set(cmp_list)
        return set()

    @abstractmethod
    def convert_loss_parameter_units(self) -> None:
        """Convert previously loaded loss parameters to base units."""


class RepairModel_DS(RepairModel_Base):
    """Repair consequences for components with damage states."""

    __slots__ = ['RV_reg']

    def save_sample(
        self, filepath: str | None = None, *, save_units: bool = False
    ) -> None | pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Save or return the loss sample.

        This method handles the storage of a sample of loss estimates,
        which can either be saved directly to a file or returned as a
        DataFrame for further manipulation. When saving to a file,
        additional information such as unit conversion factors and
        column units can be included. If the data is not being saved
        to a file, the method can return the DataFrame with or without
        units as specified.

        Parameters
        ----------
        filepath: str, optional
            The path to the file where the loss sample should be
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
            * DataFrame containing the loss sample.
            * Optionally, a Series containing the units for each
            column if `save_units` is True.

        """
        self.log.div()
        if filepath is not None:
            self.log.msg('Saving loss sample...')

        assert self.sample is not None
        assert self.loss_params is not None
        cmp_units = self.loss_params['DV', 'Unit'].sort_index()
        dv_units = pd.Series(
            index=self.sample.columns, name='Units', dtype='object'
        ).sort_index()

        valid_dv_types = dv_units.index.unique(level=0)
        valid_cmp_ids = dv_units.index.unique(level=1)

        for cmp_id, dv_type in cmp_units.index:
            if (dv_type in valid_dv_types) and (cmp_id in valid_cmp_ids):
                dv_units.loc[dv_type, cmp_id] = cmp_units.loc[cmp_id, dv_type]

        res = file_io.save_to_csv(
            self.sample,
            Path(filepath) if filepath is not None else None,
            units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )
        if filepath is not None:
            self.log.msg('Loss sample successfully saved.', prepend_timestamp=False)
            return None

        assert isinstance(res, pd.DataFrame)

        units = res.loc['Units']
        res = res.drop('Units')
        res = res.astype(float)
        assert isinstance(res, pd.DataFrame)
        assert isinstance(units, pd.Series)

        if save_units:
            return res, units

        return res

    def load_sample(self, filepath: str | pd.DataFrame) -> dict[str, str]:
        """
        Load loss sample data.

        Parameters
        ----------
        filepath: str
          Path to an existing sample stored in a file, or dataframe
          containing the existing sample.

        Returns
        -------
        dict[str, str]
          Dictionary mapping each decision variable to its assigned
          unit.

        Raises
        ------
        ValueError
          If the columns have an invalid number of levels.

        """
        names = ['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'uid']
        self.log.div()
        self.log.msg('Loading loss sample...')

        sample, units = file_io.load_data(
            filepath,
            self._asmnt.unit_conversion_factors,
            log=self._asmnt.log,
            return_units=True,
        )
        assert isinstance(sample, pd.DataFrame)
        assert isinstance(units, pd.Series)
        units.index.names = names
        # Obtain the DV units
        # Note: we don't need to check for consistency (all rows
        # having the same unit) since the units are extracted from a
        # single row in the CSV, affecting all subsequent rows.
        units_isolated = (
            units.reset_index()[['dv', 'Units']]
            .set_index('dv')
            .groupby('dv')['Units']
        )
        dv_units = units_isolated.first().to_dict()

        # check if `uid` level was provided
        num_levels = len(sample.columns.names)
        num_levels_without_uid = 6
        num_levels_with_uid = num_levels_without_uid + 1
        if num_levels == num_levels_without_uid:
            sample.columns.names = names[:-1]
            sample = base.dedupe_index(sample.T).T
        elif num_levels == num_levels_with_uid:
            sample.columns.names = names
        else:
            msg = (
                f'Invalid loss sample: Column MultiIndex '
                f'has an unexpected length: {num_levels}'
            )
            raise ValueError(msg)

        self.sample = sample

        self.log.msg('Loss sample successfully loaded.', prepend_timestamp=False)

        return dv_units

    def calculate(self, dmg_quantities: pd.DataFrame) -> None:  # noqa: C901
        """
        Calculate damage consequences.

        Parameters
        ----------
        dmg_quantities: DataFrame
            A table with the quantity of damage experienced in each
            damage state of each performance group at each location
            and direction. You can use the prepare_dmg_quantities
            method in the DamageModel to get such a DF.

        """
        assert self.loss_map is not None

        sample_size = len(dmg_quantities)

        # If everything is undamaged there are no losses
        if set(dmg_quantities.columns.get_level_values('ds')) == {'0'}:
            self.sample = None
            self.log.msg(
                'There is no damage---DV sample is set to None.',
                prepend_timestamp=False,
            )
            return

        # calculate the quantities for economies of scale
        self.log.msg('\nAggregating damage quantities...', prepend_timestamp=False)

        if self._asmnt.options.eco_scale['AcrossFloors']:
            if self._asmnt.options.eco_scale['AcrossDamageStates']:
                eco_levels = [0]
                eco_columns = ['cmp']

            else:
                eco_levels = [0, 4]
                eco_columns = ['cmp', 'ds']

        elif self._asmnt.options.eco_scale['AcrossDamageStates']:
            eco_levels = [0, 1]
            eco_columns = ['cmp', 'loc']

        else:
            eco_levels = [0, 1, 4]
            eco_columns = ['cmp', 'loc', 'ds']

        eco_group = dmg_quantities.groupby(level=eco_levels, axis=1)  # type: ignore
        eco_qnt = eco_group.sum().mask(eco_group.count() == 0, np.nan)
        assert eco_qnt.columns.names == eco_columns

        self.log.msg(
            'Successfully aggregated damage quantities.', prepend_timestamp=False
        )

        # apply the median functions, if needed, to get median consequences for
        # each realization
        self.log.msg(
            '\nCalculating the median repair consequences...',
            prepend_timestamp=False,
        )

        medians = self._calc_median_consequence(eco_qnt)

        self.log.msg(
            'Successfully determined median repair consequences.',
            prepend_timestamp=False,
        )

        # combine the median consequences with the samples of deviation from the
        # median to get the consequence realizations.
        self.log.msg(
            '\nConsidering deviations from the median values to obtain '
            'random DV sample...'
        )

        self.log.msg(
            'Preparing random variables for repair consequences...',
            prepend_timestamp=False,
        )
        self.RV_reg = self._create_DV_RVs(dmg_quantities.columns)  # type: ignore

        if self.RV_reg is not None:
            assert self._asmnt.options.sampling_method is not None
            self.RV_reg.generate_sample(
                sample_size=sample_size, method=self._asmnt.options.sampling_method
            )

            std_sample = base.convert_to_MultiIndex(
                pd.DataFrame(self.RV_reg.RV_sample), axis=1
            )
            std_sample.columns.names = ['dv', 'cmp', 'ds', 'loc', 'dir', 'uid']
            std_sample = std_sample.sort_index(axis=1)

        else:
            std_sample = None

        self.log.msg(
            f'\nSuccessfully generated {sample_size} realizations of '
            'deviation from the median consequences.',
            prepend_timestamp=False,
        )

        res_list = []
        key_list: list[tuple[Any, ...]] = []

        dmg_quantities.columns = dmg_quantities.columns.reorder_levels(  # type: ignore
            ['cmp', 'ds', 'loc', 'dir', 'uid']
        )
        dmg_quantities = dmg_quantities.sort_index(axis=1)

        if std_sample is not None:
            std_dvs = std_sample.columns.unique(level=0)
        else:
            std_dvs = []

        for decision_variable in self.decision_variables:  # noqa: PLR1702
            if decision_variable in std_dvs:
                assert isinstance(std_sample, pd.DataFrame)
                prob_cmp_list = std_sample[decision_variable].columns.unique(level=0)
            else:
                prob_cmp_list = []

            cmp_list: list[tuple[Any, ...]] = []

            if decision_variable not in medians:
                continue
            for component in medians[decision_variable].columns.unique(level=0):
                # check if there is damage in the component
                consequence = self.loss_map.loc[component, 'Repair']

                if component not in dmg_quantities.columns.get_level_values('cmp'):
                    continue

                ds_list = []

                for ds in (
                    medians[decision_variable]
                    .loc[:, component]
                    .columns.unique(level=0)
                ):
                    loc_list = []

                    for loc_id, loc in enumerate(
                        dmg_quantities.loc[
                            :, (component, ds)  # type: ignore
                        ].columns.unique(level=0)
                    ):
                        if (
                            self._asmnt.options.eco_scale['AcrossFloors'] is True
                        ) and (loc_id > 0):
                            break

                        if self._asmnt.options.eco_scale['AcrossFloors'] is True:
                            median_i = medians[decision_variable].loc[
                                :, (component, ds)
                            ]
                            dmg_i = dmg_quantities.loc[
                                :, (component, ds)  # type: ignore
                            ]

                            if component in prob_cmp_list:
                                assert std_sample is not None
                                std_i = std_sample.loc[
                                    :,
                                    (
                                        decision_variable,
                                        component,
                                        ds,
                                    ),  # type: ignore
                                ]
                            else:
                                std_i = None

                        else:
                            median_i = medians[decision_variable].loc[
                                :, (component, ds, loc)
                            ]
                            dmg_i = dmg_quantities.loc[
                                :, (component, ds, loc)  # type: ignore
                            ]

                            if component in prob_cmp_list:
                                assert std_sample is not None
                                std_i = std_sample.loc[
                                    :,
                                    (
                                        decision_variable,
                                        component,
                                        ds,
                                        loc,
                                    ),  # type: ignore
                                ]
                            else:
                                std_i = None

                        if std_i is not None:
                            res_list.append(dmg_i.mul(median_i, axis=0) * std_i)
                        else:
                            res_list.append(dmg_i.mul(median_i, axis=0))

                        loc_list.append(loc)

                    if self._asmnt.options.eco_scale['AcrossFloors'] is True:
                        ds_list += [
                            ds,
                        ]
                    else:
                        ds_list += [(ds, loc) for loc in loc_list]

                if self._asmnt.options.eco_scale['AcrossFloors'] is True:
                    cmp_list += [(consequence, component, ds) for ds in ds_list]
                else:
                    cmp_list += [
                        (consequence, component, ds, loc) for ds, loc in ds_list
                    ]

            if self._asmnt.options.eco_scale['AcrossFloors'] is True:
                key_list += [
                    (decision_variable, loss_cmp_i, dmg_cmp_i, ds)
                    for loss_cmp_i, dmg_cmp_i, ds in cmp_list
                ]
            else:
                key_list += [
                    (decision_variable, loss_cmp_i, dmg_cmp_i, ds, loc)
                    for loss_cmp_i, dmg_cmp_i, ds, loc in cmp_list
                ]

        lvl_names = ['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'uid']
        dv_sample = pd.concat(res_list, axis=1, keys=key_list, names=lvl_names)

        dv_sample = dv_sample.fillna(0).convert_dtypes()

        self.log.msg('Successfully obtained DV sample.', prepend_timestamp=False)
        self.sample = dv_sample

    def convert_loss_parameter_units(self) -> None:
        """Convert previously loaded loss parameters to base units."""
        if self.loss_params is None:
            return
        units = self.loss_params['DV', 'Unit']
        arg_units = self.loss_params['Quantity', 'Unit']
        for column in self.loss_params.columns.unique(level=0):
            if not column.startswith('DS'):
                continue
            params = self.loss_params.loc[:, column].copy()
            assert isinstance(params, pd.DataFrame)
            self.loss_params.loc[:, column] = self._convert_marginal_params(
                params, units, arg_units
            ).to_numpy()

    def drop_unused_damage_states(self) -> None:
        """
        Remove unused columns.

        Remove columns from the loss model parameters corresponding
        to unused damage states.

        """
        if self.loss_params is None:
            return
        first_level = self.loss_params.columns.get_level_values(0).unique().to_list()
        ds_list = [x for x in first_level if x.startswith('DS')]
        ds_to_drop = []
        for damage_state in ds_list:
            if (
                np.all(
                    pd.isna(
                        self.loss_params.loc[
                            :,  # type: ignore
                            idx[damage_state, :],
                        ].values
                    )
                )
                # Note: When this evaluates to True, when you add `is
                # True` on the right it suddenly evaluates to
                # False. We need to figure out why this is happening,
                # but the way it's written now does what we want in
                # each case.
            ):
                ds_to_drop.append(damage_state)  # noqa: PERF401

        self.loss_params = self.loss_params.drop(columns=ds_to_drop, level=0)

    def _create_DV_RVs(  # noqa: N802, C901
        self, cases: pd.MultiIndex
    ) -> uq.RandomVariableRegistry | None:
        """
        Prepare the random variables.

        Prepare the random variables associated with decision
        variables, such as repair cost and time.

        Parameters
        ----------
        cases: MultiIndex
            Index with cmp-loc-uid-dir-ds descriptions that identify
            the RVs we need for the simulation.

        Returns
        -------
        RandomVariableRegistry or None
            A RandomVariableRegistry containing all the generated
            random variables necessary for the simulation. If no
            random variables are generated (due to missing parameters
            or conditions), returns None.

        """
        # Convert the MultiIndex to a DataFrame
        case_df = pd.DataFrame(index=cases).reset_index()
        # maps `cmp` to array of damage states
        damage_states = case_df.groupby('cmp')['ds'].unique().to_dict()
        # maps `cmp`-`ds` to tuple of `loc`-`dir`-`uid` tuples
        loc_dir_uids = (
            case_df.groupby(['cmp', 'ds'])[['loc', 'dir', 'uid']]
            .apply(
                lambda x: tuple(  # type: ignore
                    zip(
                        x['loc'],
                        x['dir'],
                        x['uid'],
                    )
                )
            )
            .to_dict()
        )
        damaged_components = set(cases.get_level_values('cmp'))

        rv_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        rv_count = 0

        # for each component in the loss map
        assert isinstance(self.loss_map, pd.DataFrame)
        for component, consequence in self.loss_map['Repair'].items():
            # if that component does not have realized damage states,
            # skip it (e.g., this can happen when there is only
            # `collapse`).
            if component not in damaged_components:
                continue

            # for each DV
            for decision_variable in self.decision_variables:
                # If loss parameters are missing for that consequence,
                # don't estimate losses for it. A warning has already
                # been issued for what is missing.
                if (consequence, decision_variable) in self.missing:
                    continue

                # If loss parameters are missing for that consequence,
                # for this particular loss model, they will exist in
                # the other(s).
                assert self.loss_params is not None
                if (consequence, decision_variable) not in self.loss_params.index:
                    continue

                # load the corresponding parameters
                parameters = (
                    self.loss_params.loc[(consequence, decision_variable), :]
                    .dropna()
                    .to_dict()
                )

                for ds in damage_states[component]:
                    if ds == '0':
                        continue

                    ds_family = parameters.get((f'DS{ds}', 'Family'))
                    ds_theta = [
                        theta
                        for t_i in range(3)
                        if (theta := parameters.get((f'DS{ds}', f'Theta_{t_i}')))
                        is not None
                    ]
                    # If there is no RV family we don't need an RV
                    if ds_family is None:
                        continue

                    # If the first parameter is controlled by a function, we use
                    # 1.0 in its place and will scale the results in a later
                    # step
                    if isinstance(ds_theta[0], str) and '|' in ds_theta[0]:
                        ds_theta[0] = 1.0

                    loc_dir_uid = loc_dir_uids[component, ds]

                    for loc, direction, uid in loc_dir_uid:
                        # assign RVs
                        rv_reg.add_RV(
                            uq.rv_class_map(ds_family)(  # type: ignore
                                name=(
                                    f'{decision_variable}-{component}-'
                                    f'{ds}-{loc}-{direction}-{uid}'
                                ),
                                theta=np.array(ds_theta),
                                truncation_limits=np.array([0.0, np.nan]),
                            )
                        )
                        rv_count += 1

        # assign Time-Cost correlation whenever applicable
        rho = self._asmnt.options.rho_cost_time
        if rho != 0.0:
            for rv_tag in rv_reg.RV:
                if not rv_tag.startswith('Cost'):
                    continue
                component = rv_tag.split('-')[1]
                ds = rv_tag.split('-')[2]
                loc = rv_tag.split('-')[3]
                direction = rv_tag.split('-')[4]
                uid = rv_tag.split('-')[5]
                time_rv_tag = rv_tag.replace('Cost', 'Time')
                if time_rv_tag in rv_reg.RV:
                    rv_reg.add_RV_set(
                        uq.RandomVariableSet(
                            f'DV-{component}-{ds}-{loc}-{direction}-{uid}_set',
                            list(rv_reg.RVs([rv_tag, time_rv_tag]).values()),
                            np.array([[1.0, rho], [rho, 1.0]]),
                        )
                    )

        self.log.msg(
            f'\n{rv_count} random variables created.', prepend_timestamp=False
        )

        if rv_count > 0:
            return rv_reg
        return None

    def _calc_median_consequence(self, eco_qnt: pd.DataFrame) -> dict:  # noqa: C901
        """
        Calculate median reiapr consequences.

        Calculate the median repair consequences for each loss
        component based on its quantity realizations and the
        associated loss parameters.

        Parameters
        ----------
        eco_qnt: DataFrame
            A DataFrame containing quantity realizations for various
            components and damage states, appropriately grouped to
            account for economies of scale.

        decision_variables: list
            Defines the decision variables to be included in the loss
            calculations. Defaults to those supported, but fewer can be
            used if desired. When fewer are used, the loss parameters for
            those not used will not be required.

        Returns
        -------
        dict
            A dictionary where keys are the types of decision
            variables (DV) like 'COST' or 'TIME', and values are
            DataFrames containing the median consequences for each
            component and damage state. These DataFrames are structured
            with MultiIndex columns that may include 'cmp' (component),
            'ds' (damage state), and potentially 'loc' (location),
            depending on the way economies of scale are handled.

        Raises
        ------
        ValueError
            If any loss driver types or distribution types are not
            recognized, or if the parameters are incomplete or
            unsupported.

        """
        medians = {}

        for decision_variable in self.decision_variables:
            cmp_list = []
            median_list = []

            assert self.loss_map is not None
            for loss_cmp_id, loss_cmp_name in self.loss_map['Repair'].items():
                if (loss_cmp_name, decision_variable) in self.missing:
                    continue

                if loss_cmp_id not in eco_qnt.columns.get_level_values(0).unique():
                    continue

                ds_list = []
                sub_medians = []

                assert self.loss_params is not None
                for ds in self.loss_params.columns.get_level_values(0).unique():
                    if not ds.startswith('DS'):
                        continue

                    ds_id = ds[2:]

                    if ds_id == '0':
                        continue

                    loss_params_ds = self.loss_params.loc[
                        (loss_cmp_name, decision_variable), ds
                    ]

                    # check if theta_0 is defined
                    theta_0 = loss_params_ds.get('Theta_0', np.nan)

                    if pd.isna(theta_0):
                        continue

                    # check if the distribution type is supported
                    family = loss_params_ds.get('Family', np.nan)

                    if (not pd.isna(family)) and (
                        family not in {'normal', 'lognormal', 'deterministic'}
                    ):
                        msg = (
                            f'Loss Distribution of type {family} ' f'not supported.'
                        )
                        raise ValueError(msg)

                    # If theta_0 is a scalar
                    try:
                        theta_0 = float(theta_0)

                        if pd.isna(loss_params_ds.get('Family', np.nan)):
                            # if theta_0 is constant, then use it directly
                            f_median = _prep_constant_median_DV(theta_0)

                        else:
                            # otherwise use a constant 1.0 as the median
                            # The random variable will be generated as a
                            # variation from this 1.0 and added in a later step.
                            f_median = _prep_constant_median_DV(1.0)

                    except ValueError:
                        # otherwise, use the multilinear function
                        all_vals = np.array(
                            [val.split(',') for val in theta_0.split('|')],
                            dtype=float,
                        )
                        medns = all_vals[0]
                        qnts = all_vals[1]
                        f_median = _prep_bounded_multilinear_median_DV(medns, qnts)

                    # get the corresponding aggregate damage quantities
                    # to consider economies of scale
                    if 'ds' in eco_qnt.columns.names:
                        avail_ds = eco_qnt.loc[
                            :,  # type: ignore
                            loss_cmp_id,
                        ].columns.unique(level=0)

                        if ds_id not in avail_ds:
                            continue

                        eco_qnt_i = eco_qnt.loc[
                            :,  # type: ignore
                            (loss_cmp_id, ds_id),
                        ].copy()

                    else:
                        eco_qnt_i = eco_qnt.loc[
                            :,  # type: ignore
                            loss_cmp_id,
                        ].copy()

                    if isinstance(eco_qnt_i, pd.Series):
                        eco_qnt_i = eco_qnt_i.to_frame()
                        eco_qnt_i.columns = ['X']
                        eco_qnt_i.columns.name = 'del'

                    # generate the median values for each realization
                    eco_qnt_i.loc[:, :] = f_median(eco_qnt_i.values)  # type: ignore

                    sub_medians.append(eco_qnt_i)
                    ds_list.append(ds_id)

                if len(ds_list) > 0:
                    # combine medians across damage states into one DF
                    median_list.append(pd.concat(sub_medians, axis=1, keys=ds_list))
                    cmp_list.append(loss_cmp_id)

            if len(cmp_list) > 0:
                # combine medians across components into one DF
                result = pd.concat(median_list, axis=1, keys=cmp_list)

                # remove the extra column header level
                if 'del' in result.columns.names:
                    result.columns = result.columns.droplevel('del')

                # name the remaining column header levels
                if eco_qnt.columns.names == ['cmp', 'ds', 'loc']:
                    result.columns.names = ['cmp', 'ds', 'loc']
                else:
                    result.columns.names = ['cmp', 'ds']

                # save the results to the returned dictionary
                medians.update({decision_variable: result})

        return medians


class RepairModel_LF(RepairModel_Base):
    """Repair consequences for components with loss functions."""

    __slots__ = []

    def calculate(
        self,
        demand_sample: pd.DataFrame,
        cmp_sample: dict,
        cmp_marginal_params: pd.DataFrame,
        demand_offset: dict,
        nondirectional_multipliers: dict,
    ) -> None:
        """
        Calculate repair consequences.

        Parameters
        ----------
        demand_sample: pd.DataFrame
            The sample of the demand model to be used for the inputs
            of the loss functions.
        cmp_sample: dict
            Dict mapping each `cmp`-`loc`-`dir`-`uid` to the component
            quantity realizations in the asset model in the form of
            pd.Series objects.
        cmp_marginal_params: pd.DataFrame
            Dataframe containing component marginal distribution
            parameters.
        demand_offset: dict
            Dictionary specifying the demand offset.
        nondirectional_multipliers: dict
            Dictionary specifying the non directional multipliers used
            to combine the directional demands.


        """
        if self.loss_params is None:
            return

        assert self.loss_map is not None
        loss_map = self.loss_map['Repair'].to_dict()
        sample_size = len(demand_sample)

        # TODO(JVM): this can be taken out and simply passed as blocks in
        # the arguments, and cast to a dict in here. Index can be
        # obtained from there.
        index = [
            x
            for x in cmp_marginal_params.index.get_level_values(0)
            if loss_map.get(x) in self.loss_params.index
        ]
        # If `Blocks` information is unspecified add one block per
        # component.
        if 'Blocks' not in cmp_marginal_params.columns:
            cmp_marginal_params['Blocks'] = 1
        blocks = cmp_marginal_params.loc[index, 'Blocks'].to_dict()

        performance_group_dict = {}
        for (component, location, direction, uid), num_blocks in blocks.items():
            for decision_variable in self.decision_variables:
                if (component, decision_variable) in self.missing:
                    continue
                performance_group_dict[
                    (component, decision_variable), location, direction, uid
                ] = num_blocks

        if not performance_group_dict:
            self.log.msg(
                'No loss function-driven components---LF sample is set to None.',
                prepend_timestamp=False,
            )
            return

        performance_group = pd.DataFrame(  # type: ignore
            performance_group_dict.values(),
            index=performance_group_dict.keys(),
            columns=['Blocks'],
        )
        performance_group.index.names = ['cmp', 'loc', 'dir', 'uid']

        required_edps = base.invert_mapping(
            _get_required_demand_type(
                self.loss_params, performance_group, demand_offset
            )
        )

        available_edps = (
            pd.DataFrame(index=demand_sample.columns)
            .reset_index()
            .groupby(['type', 'loc'])['dir']
            .agg(lambda x: list(set(x)))
            .to_dict()
        )

        # Raise an error if demand sample is missing necessary entries.
        _verify_edps_available(available_edps, set(required_edps.values()))

        demand_dict = _assemble_required_demand_data(
            set(required_edps.values()),
            nondirectional_multipliers,
            demand_sample,
        )

        self.log.msg(
            '\nCalculating the median repair consequences...',
            prepend_timestamp=False,
        )

        medians = self._calc_median_consequence(
            performance_group, loss_map, required_edps, demand_dict, cmp_sample
        )

        self.log.msg(
            'Successfully determined median repair consequences.',
            prepend_timestamp=False,
        )

        self.log.msg(
            '\nConsidering deviations from the median values to obtain '
            'random DV sample...'
        )

        self.log.msg(
            'Preparing random variables for repair cost and time...',
            prepend_timestamp=False,
        )

        rv_reg = self._create_DV_RVs(medians.columns)  # type: ignore
        if rv_reg is not None:
            assert self._asmnt.options.sampling_method is not None
            rv_reg.generate_sample(
                sample_size=sample_size, method=self._asmnt.options.sampling_method
            )

            std_sample = base.convert_to_MultiIndex(
                pd.DataFrame(rv_reg.RV_sample), axis=1
            )
            std_sample.columns.names = [
                'dv',
                'loss',
                'dmg',
                'loc',
                'dir',
                'uid',
                'block',
            ]
            std_sample = std_sample.sort_index(axis=1)
            sample = (medians * std_sample).combine_first(medians)

        else:
            sample = medians

        self.log.msg(
            f'\nSuccessfully generated {sample_size} realizations of '
            'deviation from the median consequences.',
            prepend_timestamp=False,
        )

        # sum up the block losses
        sample = sample.groupby(  # type: ignore
            by=['dv', 'loss', 'dmg', 'loc', 'dir', 'uid'], axis=1
        ).sum()

        self.log.msg('Successfully obtained DV sample.', prepend_timestamp=False)
        self.sample = sample

        return

    def convert_loss_parameter_units(self) -> None:
        """Convert previously loaded loss parameters to base units."""
        if self.loss_params is None:
            return
        units = self.loss_params['DV', 'Unit']
        arg_units = self.loss_params['Demand', 'Unit']
        column = 'LossFunction'
        params = self.loss_params[column].copy()
        assert isinstance(params, pd.DataFrame)
        self.loss_params.loc[:, column] = self._convert_marginal_params(
            params,
            units,
            arg_units,
            divide_units=False,
        ).to_numpy()
        return

    def _calc_median_consequence(
        self,
        performance_group: pd.DataFrame,
        loss_map: dict,
        required_edps: dict,
        demand_dict: dict,
        cmp_sample: dict,
    ) -> pd.DataFrame:
        """
        Calculate median repair consequences.

        Calculates the median repair consequences for each loss
        function-driven component based on its quantity realizations
        and the associated loss parameters.

        Parameters
        ----------
        performance_group: pd.DataFrame
            Dataframe with a single column `Blocks` containing an
            integer for the number of blocks of each
            (`cmp`-`decision_variable`)-`loc`-`dir`-`uid`.
        loss_map: dict
            Dictionary mpping component IDs, `cmp`, to their repair
            consequences.
        required_edps: dict
            Dictionary mapping (`cmp`-`realization`)-`loc`-`dir`-`uid`
            (each entry of the `performance_group`'s index) with the
            EDP name (e.g., `PFA-1-1`) that should be used as its loss
            function input.
        demand_dict: dict
            Dictionary mapping each EDP name to the values in the
            demand sample in the form of numpy arrays.
        cmp_sample: dict
            Dict mapping each `cmp`-`loc`-`dir`-`uid` to the component
            quantity realizations in the asset model in the form of
            pd.Series objects.


        Returns
        -------
        pd.DataFrame
            Dataframe with medial loss for loss-function driven
            components.

        Raises
        ------
        ValueError
            If loss function interpolation fails.

        """
        medians_dict = {}

        # for each component in the asset model
        component: str
        decision_variable: str
        location: str
        direction: str
        uid: str
        blocks: int
        for (  # type: ignore
            (component, decision_variable),
            location,
            direction,
            uid,
        ), blocks in performance_group['Blocks'].items():
            consequence = loss_map[component]
            edp = required_edps[
                (consequence, decision_variable), location, direction, uid
            ]
            edp_values = demand_dict[edp]
            assert self.loss_params is not None
            loss_function_str = self.loss_params.loc[
                (component, decision_variable), ('LossFunction', 'Theta_0')
            ]
            assert isinstance(loss_function_str, str)
            try:
                median_loss = base.stringterpolation(loss_function_str)(edp_values)
            except ValueError as exc:
                msg = (
                    f'Loss function interpolation for consequence '
                    f'`{consequence}-{decision_variable}` has failed. '
                    f'Ensure a sufficient interpolation domain  '
                    f'for the X values (those after the `|` symbol)  '
                    f'and verify the X-value and Y-value lengths match.'
                )
                raise ValueError(msg) from exc
            for block in range(blocks):
                medians_dict[
                    decision_variable,
                    consequence,
                    component,
                    location,
                    direction,
                    uid,
                    str(block),
                ] = (
                    median_loss
                    * cmp_sample[component, location, direction, uid].to_numpy()
                    / float(blocks)
                )

        medians = pd.DataFrame(medians_dict)
        medians.columns.names = ['dv', 'loss', 'dmg', 'loc', 'dir', 'uid', 'block']
        return medians.sort_index(axis=1)

    def _create_DV_RVs(  # noqa: N802
        self, cases: pd.MultiIndex
    ) -> uq.RandomVariableRegistry | None:
        """
        Prepare the decision variable random variables.

        Prepares the random variables associated with decision
        variables, such as repair cost and time.

        Parameters
        ----------
        cases: MultiIndex
            Index with `dv`-`loss`-`dmg`-`loc`-`dir`-`uid`
            entries that identify the RVs we need for the
            simulation (columns of the `medians` dataframe).

        Returns
        -------
        RandomVariableRegistry or None
            A RandomVariableRegistry containing all the generated
            random variables necessary for the simulation. If no
            random variables are generated (due to missing parameters
            or conditions), returns None.

        """
        rv_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        rv_count = 0

        # for each component in the loss map
        for (
            decision_variable,
            consequence,
            component,
            location,
            direction,
            uid,
            block,
        ) in cases:
            # load the corresponding parameters
            assert self.loss_params is not None
            parameters = self.loss_params.loc[(consequence, decision_variable), :]

            if ('LossFunction', 'Family') not in parameters:
                # Everything is deterministic, no need to create RVs.
                continue
            family = parameters.loc['LossFunction', 'Family']
            theta = [
                theta_value
                for t_i in range(3)
                if (theta_value := parameters.get(('LossFunction', f'Theta_{t_i}')))
                is not None
            ]
            # If there is no RV family we don't need an RV
            if pd.isna(family):
                continue

            # Since the first parameter is controlled by a function,
            # we use 1.0 in its place and will scale the results in a
            # later step.
            theta[0] = 1.0  # type: ignore

            # assign RVs
            rv_reg.add_RV(
                uq.rv_class_map(family)(  # type: ignore
                    name=(
                        f'{decision_variable}-{consequence}-'
                        f'{component}-{location}-{direction}-{uid}-{block}'
                    ),
                    theta=np.array(theta),
                    truncation_limits=np.array([0.0, np.nan]),
                )
            )
            rv_count += 1

        # assign Time-Cost correlation whenever applicable
        rho = self._asmnt.options.rho_cost_time
        if rho != 0.0:
            for rv_tag in rv_reg.RV:
                if not rv_tag.startswith('Cost'):
                    continue
                split = rv_tag.split('-')
                consequence = split[1]
                component = split[2]
                location = split[3]
                direction = split[4]
                uid = split[5]
                block = split[6]
                time_rv_tag = rv_tag.replace('Cost', 'Time')
                if time_rv_tag in rv_reg.RV:
                    rv_reg.add_RV_set(
                        uq.RandomVariableSet(
                            (
                                f'DV-{consequence}-{component}-'
                                f'{location}-{direction}-{uid}-{block}_set'
                            ),
                            list(rv_reg.RVs([rv_tag, time_rv_tag]).values()),
                            np.array([[1.0, rho], [rho, 1.0]]),
                        )
                    )

        self.log.msg(
            f'\n{rv_count} random variables created.', prepend_timestamp=False
        )

        if rv_count > 0:
            return rv_reg
        return None


def _prep_constant_median_DV(median: float) -> Callable:  # noqa: N802
    """
    Return a constant median Decision Variable (DV) function.

    Parameters
    ----------
    median: float
        The median DV for a consequence function with fixed median.

    Returns
    -------
    callable
        A function that returns the constant median DV for all component
        quantities.

    """

    def f(*args):  # noqa: ANN002, ANN202, ARG001
        return median

    return f


def _prep_bounded_multilinear_median_DV(  # noqa: N802
    medians: np.ndarray, quantities: np.ndarray
) -> Callable:
    """
    Return a bounded multilinear median Decision Variable (DV) function.

    The median DV equals the min and max values when the quantity is
    outside of the prescribed quantity bounds. When the quantity is within the
    bounds, the returned median is calculated by linear interpolation.

    Parameters
    ----------
    medians: ndarray
        Series of values that define the y coordinates of the multilinear DV
        function.
    quantities: ndarray
        Series of values that define the component quantities corresponding to
        the series of medians and serving as the x coordinates of the
        multilinear DV function.

    Returns
    -------
    callable
        A function that returns the median DV given the quantity of damaged
        components.

    """

    def f(quantity):  # noqa: ANN001, ANN202
        if quantity is None:
            msg = (
                'A bounded linear median Decision Variable function called '
                'without specifying the quantity of damaged components'
            )
            raise ValueError(msg)

        q_array = np.asarray(quantity, dtype=np.float64)

        # calculate the median consequence given the quantity of damaged
        # components
        return np.interp(q_array, quantities, medians)

    return f


def _is_for_lf_model(data: pd.DataFrame) -> bool:
    """
    Determine if the data are for the lf_model.

    Parameters
    ----------
    data: pd.DataFrame
        Data to be checked.

    Returns
    -------
    bool
        Whether the data are for the lf_model.

    """
    return 'LossFunction' in data.columns.get_level_values(0)


def _is_for_ds_model(data: pd.DataFrame) -> bool:
    """
    Determine if the data are for the ds_model.

    Parameters
    ----------
    data: pd.DataFrame
        Data to be checked.

    Returns
    -------
    bool
        Whether the data are for the ds_model.

    """
    return 'DS1' in data.columns.get_level_values(0)
