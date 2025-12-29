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


"""Classes and methods that control the performance assessment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pelicun import base, file_io, model, uq
from pelicun.__init__ import __version__ as pelicun_version  # type: ignore
from pelicun.base import EDP_to_demand_type, get

if TYPE_CHECKING:
    from pelicun.base import Logger

default_damage_processes = {
    'FEMA P-58': {
        '1_excessive.coll.DEM': {'DS1': 'collapse_DS1'},
        '2_collapse': {'DS1': 'ALL_NA'},
        '3_excessiveRID': {'DS1': 'irreparable_DS1'},
        '4_irreparable': {'DS1': 'ALL_NA'},
        '5_irreparable': {'DS1': 'collapse_DS0'},
    },
    # TODO(AZ): expand with ground failure logic
    'Hazus Earthquake - Lifeline Facilities': {
        '1_LF': {'DS5': 'collapse_DS1'},
        '2_collapse': {'DS1': 'ALL_NA'},
    },
    'Hazus Earthquake - Buildings': {
        '1_STR': {'DS5': 'collapse_DS1'},
        '2_excessive.coll.DEM': {'DS1': 'collapse_DS1'},
        '3_collapse': {'DS1': 'ALL_NA'},
        '4_excessiveRID': {'DS1': 'irreparable_DS1'},
        '5_irreparable': {'DS1': 'ALL_NA'},
        '6_irreparable': {'DS1': 'collapse_DS0'},
    },
    'Hazus Hurricane': {},
}


class AssessmentBase:
    """
    Base class for Assessment objects.

    Assessment objects manage the models, data, and calculations in pelicun.

    """

    __slots__: list[str] = [
        'asset',
        'damage',
        'demand',
        'log',
        'loss',
        'options',
        'stories',
        'unit_conversion_factors',
    ]

    def __init__(self, config_options: dict[str, Any] | None = None) -> None:
        """
        Initialize an Assessment object.

        Parameters
        ----------
        config_options:
            User-specified configuration dictionary.

        """
        self.stories: int | None = None
        self.options = base.Options(config_options, self)
        self.unit_conversion_factors: dict = base.parse_units(
            self.options.units_file
        )

        self.log: Logger = self.options.log
        self.log.msg(
            f'pelicun {pelicun_version} | \n',
            prepend_timestamp=False,
            prepend_blank_space=False,
        )
        self.log.print_system_info()
        self.log.div()
        self.log.msg('Assessment Started')

        self.demand: model.DemandModel = model.DemandModel(self)
        self.asset: model.AssetModel = model.AssetModel(self)
        self.damage: model.DamageModel = model.DamageModel(self)
        self.loss: model.LossModel = model.LossModel(self)

    @property
    def bldg_repair(self) -> model.LossModel:
        """
        Exists for <backwards compatibility>.

        Returns
        -------
        model.LossModel
            The loss model.

        """
        self.log.warning(
            '`.bldg_repair` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `.loss` instead.'
        )

        return self.loss

    @property
    def repair(self) -> model.LossModel:
        """
        Exists for <backwards compatibility>.

        Returns
        -------
        RepairModel_DS
            The damage state-driven component loss model.

        """
        self.log.warning(
            '`.repair` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `.loss` instead.'
        )
        return self.loss

    def get_default_data(
        self, method_name: str, model_type: str | None = None
    ) -> pd.DataFrame:
        """
        Load a default data file.

        Loads a default data file by name and returns it. This method
        is specifically designed to access predefined CSV files from a
        structured directory path related to the SimCenter fragility
        library.

        Parameters
        ----------
        method_name: str
            The name of the method to be used. This name is used to look
            up the full path to the model files in the Damage and Loss Model
            Library.
        model_type: str
            The type of model requested. Currently, the following types
            are supported: 'fragility', 'consequence_repair',
            'loss_repair'

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the data loaded from the
            model CSV file.

        """
        # <backwards compatibility>
        if model_type is None:
            # Legacy inputs will have a filename provided instead of a
            # method name
            data_path = file_io.substitute_default_path(
                [f'PelicunDefault/{method_name}.csv'], log=self.log
            )[0]

        else:
            data_path = file_io.substitute_default_path(
                [f'PelicunDefault/{method_name}/{model_type}.csv'], log=self.log
            )[0]
        assert isinstance(data_path, str)

        data = file_io.load_data(
            data_path, None, orientation=1, reindex=False, log=self.log
        )

        assert isinstance(data, pd.DataFrame)
        return data

    def get_default_metadata(
        self, method_name: str, model_type: str | None = None
    ) -> dict:
        """
        Load a default metadata file and pass it to the user.

        Parameters
        ----------
        method_name: string
            The name of the method to be used. This name is used to look
            up the full path to the model files in the Damage and Loss Model
            Library.
        model_type: str
            The type of model requested. Currently, the following types
            are supported: 'fragility', 'consequence_repair',
            'loss_repair'

        Returns
        -------
        dict
            Default metadata describing the models available for the
            specified method.

        """
        # <backwards compatibility>
        if model_type is None:
            # Legacy inputs will have a filename provided instead of a
            # method name
            data_path = file_io.substitute_default_path(
                [f'PelicunDefault/{method_name}.json'], log=self.log
            )[0]

        else:
            data_path = file_io.substitute_default_path(
                [f'PelicunDefault/{method_name}/{model_type}.json'], log=self.log
            )[0]
        assert isinstance(data_path, str)

        with Path(data_path).open(encoding='utf-8') as f:
            data = json.load(f)

        return data  # noqa: RET504

    def calc_unit_scale_factor(self, unit: str) -> float:
        """
        Determine unit scale factor.

        Determines the scale factor from input unit to the
        corresponding base unit.

        Parameters
        ----------
        unit: str
            Either a unit name, or a quantity and a unit name
            separated by a space.
            For example: 'ft' or '100 ft'.

        Returns
        -------
        float
            Scale factor that convert values from unit to base unit

        Raises
        ------
        KeyError
            When an invalid unit is specified

        """
        unit_lst = unit.strip().split(' ')

        # check if there is a quantity specified; if yes, parse it
        if len(unit_lst) > 1:
            unit_count_str, unit_name = unit_lst
            unit_count = float(unit_count_str)

        else:
            unit_count = 1
            unit_name = unit_lst[0]

        try:
            scale_factor = unit_count * self.unit_conversion_factors[unit_name]

        except KeyError as exc:
            msg = f'Specified unit not recognized: {unit_count} {unit_name}'
            raise KeyError(msg) from exc

        return scale_factor

    def scale_factor(self, unit: str | None) -> float:
        """
        Get scale factor of given unit.

        Returns the scale factor of a given unit. If the unit is
        unknown it raises an error. If the unit is None it returns
        1.00.

        Parameters
        ----------
        unit: str
            A unit name.

        Returns
        -------
        float
            Scale factor

        Raises
        ------
        ValueError
            If the unit is unknown.

        """
        if unit is not None:
            if unit in self.unit_conversion_factors:
                scale_factor = self.unit_conversion_factors[unit]

            else:
                msg = f'Unknown unit: {unit}'
                raise ValueError(msg)
        else:
            scale_factor = 1.0

        return scale_factor


class Assessment(AssessmentBase):
    """
    Assessment class.

    Has methods implementing a Scenario-Based assessment.

    """

    __slots__: list[str] = []

    def calculate_damage(
        self,
        num_stories: int,
        demand_config: dict,
        demand_data_source: str | dict,
        cmp_data_source: str | dict[str, pd.DataFrame],
        damage_data_paths: list[str | pd.DataFrame],
        dmg_process: dict | None = None,
        scaling_specification: dict | None = None,
        residual_drift_configuration: dict | None = None,
        collapse_fragility_configuration: dict | None = None,
        block_batch_size: int = 1000,
    ) -> None:
        """
        Calculate damage.

        Parameters
        ----------
        num_stories: int
            Number of stories of the asset. Applicable to buildings.
        demand_config: dict
            A dictionary containing configuration options for the
            sample generation. Key options include:
            * 'SampleSize': The number of samples to generate.
            * 'PreserveRawOrder': Boolean indicating whether to
            preserve the order of the raw data. Defaults to False.
            * 'DemandCloning': Specifies if and how demand cloning
            should be applied. Can be a boolean or a detailed
            configuration.
        demand_data_source: string or dict
            If string, the demand_data_source is a file prefix
            (<prefix> in the following description) that identifies
            the following files: <prefix>_marginals.csv,
            <prefix>_empirical.csv, <prefix>_correlation.csv. If dict,
            the demand data source is a dictionary with the following
            optional keys: 'marginals', 'empirical', and
            'correlation'. The value under each key shall be a
            DataFrame.
        cmp_data_source: str or dict
            The source from where to load the component model data. If
            it's a string, it should be the prefix for three files:
            one for marginal distributions (`<prefix>_marginals.csv`),
            one for empirical data (`<prefix>_empirical.csv`), and one
            for correlation data (`<prefix>_correlation.csv`). If it's
            a dictionary, it should have keys 'marginals',
            'empirical', and 'correlation', with each key associated
            with a DataFrame containing the corresponding data.
        damage_data_paths: list of (string | DataFrame)
            List of paths to data or files with damage model
            information. Default XY datasets can be accessed as
            PelicunDefault/XY. Order matters. Parameters defined in
            prior elements in the list take precedence over the same
            parameters in subsequent data paths. I.e., place the
            Default datasets in the back.
        dmg_process: dict, optional
            Allows simulating damage processes, where damage to some
            component can alter the damage state of other components.
        scaling_specification: dict, optional
            A dictionary defining the shift in median.
            Example: {'CMP-1-1': '*1.2', 'CMP-1-2': '/1.4'}
            The keys are individual components that should be present
            in the `capacity_sample`.  The values should be strings
            containing an operation followed by the value formatted as
            a float.  The operation can be '+' for addition, '-' for
            subtraction, '*' for multiplication, and '/' for division.
        residual_drift_configuration: dict
            Dictionary containing the following keys-values:
            - params: dict
            A dictionary containing parameters required for the
            estimation method, such as 'yield_drift', which is the
            drift at which yielding is expected to occur.
            - method: str, optional
            The method used to estimate the RID values. Currently,
            only 'FEMA P58' is implemented. Defaults to 'FEMA P58'.
        collapse_fragility_configuration: dict
            Dictionary containing the following keys-values:
            - label: str
            Label to use to extend the MultiIndex of the demand
            sample.
            - value: float
            Values to add to the rows of the additional column.
            - unit: str
            Unit that corresponds to the additional column.
            - location: str, optional
            Optional location, defaults to `0`.
            - direction: str, optional
            Optional direction, defaults to `1`.
        block_batch_size: int
            Maximum number of components in each batch.

        """
        # TODO(JVM): when we build the API docs, ensure the above is
        # properly rendered.

        self.demand.load_model(demand_data_source)
        self.demand.generate_sample(demand_config)

        if residual_drift_configuration:
            self.demand.estimate_RID_and_adjust_sample(
                residual_drift_configuration['parameters'],
                residual_drift_configuration['method'],
            )

        if collapse_fragility_configuration:
            self.demand.expand_sample(
                collapse_fragility_configuration['label'],
                collapse_fragility_configuration['value'],
                collapse_fragility_configuration['unit'],
            )

        self.stories = num_stories
        self.asset.load_cmp_model(cmp_data_source)
        self.asset.generate_cmp_sample()

        self.damage.load_model_parameters(
            damage_data_paths, set(self.asset.list_unique_component_ids())
        )
        self.damage.calculate(dmg_process, block_batch_size, scaling_specification)

    def calculate_loss(
        self,
        decision_variables: tuple[str, ...],
        loss_model_data_paths: list[str | pd.DataFrame],
        loss_map_path: str | pd.DataFrame | None = None,
        loss_map_policy: str | None = None,
    ) -> None:
        """
        Calculate loss.

        Parameters
        ----------
        decision_variables: tuple
            Defines the decision variables to be included in the loss
            calculations. Defaults to those supported, but fewer can be
            used if desired. When fewer are used, the loss parameters for
            those not used will not be required.
        loss_model_data_paths: list of (string | DataFrame)
            List of paths to data or files with loss model
            information. Default XY datasets can be accessed as
            PelicunDefault/XY. Order matters. Parameters defined in
            prior elements in the list take precedence over the same
            parameters in subsequent data paths. I.e., place the
            Default datasets in the back.
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

        """
        self.loss.decision_variables = decision_variables
        self.loss.add_loss_map(loss_map_path, loss_map_policy)
        self.loss.load_model_parameters(loss_model_data_paths)
        self.loss.calculate()

    def aggregate_loss(
        self,
        replacement_configuration: (
            tuple[uq.RandomVariableRegistry, dict[str, float]] | None
        ) = None,
        loss_combination: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate losses.

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

        Notes
        -----
        Regardless of the value of the arguments, this method does not
        alter the state of the loss model, i.e., it does not modify
        the values of the `.sample` attributes.

        Returns
        -------
        tuple
            Dataframe with the aggregated loss of each realization,
            and another boolean dataframe with information on which DV
            thresholds were exceeded in each realization, triggering
            replacement. If no thresholds are specified it only
            contains False values.

        """
        output = self.loss.aggregate_losses(
            replacement_configuration, loss_combination, future=True
        )
        assert isinstance(output, tuple)
        return output


class DLCalculationAssessment(AssessmentBase):
    """Base class for the assessment objects used in `DL_calculation.py`."""

    __slots__: list[str] = []

    def calculate_demand(  # noqa: C901
        self,
        demand_path: Path,
        collapse_limits: dict[str, float] | None,
        length_unit: str | None,
        demand_calibration: dict | None,
        sample_size: int,
        demand_cloning: dict | None,
        residual_drift_inference: dict | None,
        *,
        coupled_demands: bool,
    ) -> None:
        """
        Calculate demands.

        Parameters
        ----------
        demand_path: str
            Path to the demand data file.
        collapse_limits: dict[str, float] or None
            Optional dictionary with demand types and their respective
            collapse limits.
        length_unit : str, optional
            Unit of length to be used to add units to the demand data
            if needed.
        demand_calibration: dict or None
            Calibration data for the demand model.
        sample_size: int
            Number of realizations.
        coupled_demands: bool
            Whether to preserve the raw order of the demands.
        demand_cloning: dict or None
            Demand cloning configuration.
        residual_drift_inference: dict or None
            Information for residual drift inference.

        Raises
        ------
        ValueError
            When an unknown residual drift method is specified.

        """
        idx = pd.IndexSlice
        raw_demands = pd.read_csv(demand_path, index_col=0)

        # remove excessive demands that are considered collapses, if needed
        if collapse_limits:
            raw_demands_m = base.convert_to_MultiIndex(raw_demands, axis=1)
            assert isinstance(raw_demands_m, pd.DataFrame)
            raw_demands = raw_demands_m

            if 'Units' in raw_demands.index:
                raw_units = raw_demands.loc['Units', :]
                raw_demands = raw_demands.drop('Units', axis=0).astype(float)

            else:
                raw_units = None

            dem_to_drop = np.full(raw_demands.shape[0], fill_value=False)

            for dem_type, limit in collapse_limits.items():
                assert isinstance(dem_type, str)
                assert isinstance(limit, (str, float))
                nlevels_with_event_id = 4
                if raw_demands.columns.nlevels == nlevels_with_event_id:
                    dem_to_drop += raw_demands.loc[
                        :,  # type: ignore
                        idx[:, dem_type, :, :],
                    ].max(axis=1) > float(limit)

                else:
                    dem_to_drop += raw_demands.loc[
                        :,  # type: ignore
                        idx[dem_type, :, :],
                    ].max(axis=1) > float(limit)

            raw_demands = raw_demands.loc[~dem_to_drop, :]

            if isinstance(raw_units, pd.Series):
                raw_demands = pd.concat(
                    [raw_demands, raw_units.to_frame().T], axis=0
                )

            self.log.msg(
                f'{np.sum(dem_to_drop)} realizations removed from the demand '
                f'input because they exceed the collapse limit. The remaining '
                f'sample size: {raw_demands.shape[0]}'
            )

        # add units to the demand data if needed
        if 'Units' not in raw_demands.index:
            if length_unit is None:
                msg = 'A length unit is required to infer demand units.'
                raise ValueError(msg)
            demands = _add_units(raw_demands, length_unit)

        else:
            demands = raw_demands

        # load the available demand sample
        self.demand.load_sample(demands)

        # get the calibration information
        if demand_calibration:
            # then use it to calibrate the demand model
            self.demand.calibrate_model(demand_calibration)

        else:
            # if no calibration is requested,
            # set all demands to use empirical distribution
            self.demand.calibrate_model({'ALL': {'DistributionFamily': 'empirical'}})

        # and generate a new demand sample
        self.demand.generate_sample(
            {
                'SampleSize': sample_size,
                'PreserveRawOrder': coupled_demands,
                'DemandCloning': demand_cloning,
            }
        )

        # get the generated demand sample
        demand_sample_tuple = self.demand.save_sample(save_units=True)
        assert demand_sample_tuple is not None
        demand_sample, demand_units = demand_sample_tuple
        assert isinstance(demand_sample, pd.DataFrame)
        assert isinstance(demand_units, pd.Series)

        demand_sample = pd.concat([demand_sample, demand_units.to_frame().T])

        # get residual drift estimates, if needed
        if residual_drift_inference:
            # `method` is guaranteed to exist because it is confirmed when
            # parsing the configuration file.
            rid_inference_method = residual_drift_inference.pop('method')

            if rid_inference_method == 'FEMA P-58':
                rid_list: list[pd.DataFrame] = []
                pid = demand_sample['PID'].copy()
                pid = pid.drop('Units')
                pid = pid.astype(float)

                for direction, delta_yield in residual_drift_inference.items():
                    pids = pid.loc[:, idx[:, direction]]  # type: ignore
                    assert isinstance(pids, pd.DataFrame)
                    rid = self.demand.estimate_RID(
                        pids,
                        {'yield_drift': float(delta_yield)},
                    )

                    rid_list.append(rid)

                rid = pd.concat(rid_list, axis=1)
                rid_units = pd.Series(
                    ['unitless'] * rid.shape[1],
                    index=rid.columns,
                    name='Units',
                )
                rid_sample = pd.concat([rid, rid_units.to_frame().T])
                demand_sample = pd.concat([demand_sample, rid_sample], axis=1)

            else:
                msg = (
                    f'Unknown residual drift inference method: '
                    f'`{rid_inference_method}`.'
                )
                raise ValueError(msg)

        # add a constant one demand
        demand_sample['ONE', '0', '1'] = np.ones(demand_sample.shape[0])
        demand_sample.loc['Units', ('ONE', '0', '1')] = 'unitless'

        self.demand.load_sample(base.convert_to_SimpleIndex(demand_sample, axis=1))

    def calculate_asset(
        self,
        num_stories: int,
        component_assignment_file: str | None,
        collapse_fragility_demand_type: str | None,
        component_sample_file: str | None,
        *,
        add_irreparable_damage_columns: bool,
    ) -> None:
        """
        Generate the asset model sample.

        Parameters
        ----------
        num_stories: int
            Number of stories.
        component_assignment_file: str or None
            Path to a component assignment file.
        collapse_fragility_demand_type: str or None
            Optional demand type for the collapse fragility.
        add_irreparable_damage_columns: bool
            Whether to add columns for irreparable damage.
        component_sample_file: str or None
            Optional path to an existing component sample file.

        Raises
        ------
        ValueError
            With invalid combinations of arguments.

        """
        # retrieve the demand sample
        demand_sample = self.demand.save_sample()
        assert isinstance(demand_sample, pd.DataFrame)

        # set the number of stories
        if num_stories:
            self.stories = num_stories

        # We either accept a `component_assignment_file` or a
        # `component_sample_file`, not both.
        if (
            component_assignment_file is not None
            and component_sample_file is not None
        ):
            msg = (
                'Both `component_assignment_file` and '
                '`component_sample_file` are provided. '
                'Please provide only one.'
            )
            raise ValueError(msg)

        # load a component model and generate a sample
        if component_assignment_file is not None:
            cmp_marginals = pd.read_csv(
                component_assignment_file,
                index_col=0,
                encoding_errors='replace',
            )

            dem_types = demand_sample.columns.unique(level=0)

            # add component(s) to support collapse calculation
            if collapse_fragility_demand_type is not None:
                if not collapse_fragility_demand_type.startswith('SA'):
                    # we need story-specific collapse assessment
                    # (otherwise we have a global demand and evaluate
                    # collapse directly, so this code should be skipped)

                    if collapse_fragility_demand_type in dem_types:
                        # excessive coll_DEM is added on every floor
                        # to detect large RIDs
                        cmp_marginals.loc['excessive.coll.DEM', 'Units'] = 'ea'

                        locs = demand_sample[
                            collapse_fragility_demand_type  # type: ignore
                        ].columns.unique(level=0)
                        cmp_marginals.loc['excessive.coll.DEM', 'Location'] = (
                            ','.join(locs)
                        )

                        dirs = demand_sample[
                            collapse_fragility_demand_type  # type: ignore
                        ].columns.unique(level=1)
                        cmp_marginals.loc['excessive.coll.DEM', 'Direction'] = (
                            ','.join(dirs)
                        )

                        cmp_marginals.loc['excessive.coll.DEM', 'Theta_0'] = 1.0

                    else:
                        self.log.msg(
                            f'WARNING: No {collapse_fragility_demand_type} '
                            f'among available demands. Collapse cannot '
                            f'be evaluated.'
                        )

            # always add a component to support basic collapse calculation
            cmp_marginals.loc['collapse', 'Units'] = 'ea'
            cmp_marginals.loc['collapse', 'Location'] = 0
            cmp_marginals.loc['collapse', 'Direction'] = 1
            cmp_marginals.loc['collapse', 'Theta_0'] = 1.0

            # add components to support irreparable damage calculation
            if add_irreparable_damage_columns:
                if 'RID' in dem_types:
                    # excessive RID is added on every floor to detect large RIDs
                    cmp_marginals.loc['excessiveRID', 'Units'] = 'ea'

                    locs = demand_sample['RID'].columns.unique(level=0)
                    cmp_marginals.loc['excessiveRID', 'Location'] = ','.join(locs)

                    dirs = demand_sample['RID'].columns.unique(level=1)
                    cmp_marginals.loc['excessiveRID', 'Direction'] = ','.join(dirs)

                    cmp_marginals.loc['excessiveRID', 'Theta_0'] = 1.0

                    # irreparable is a global component to recognize is any of the
                    # excessive RIDs were triggered
                    cmp_marginals.loc['irreparable', 'Units'] = 'ea'
                    cmp_marginals.loc['irreparable', 'Location'] = 0
                    cmp_marginals.loc['irreparable', 'Direction'] = 1
                    cmp_marginals.loc['irreparable', 'Theta_0'] = 1.0

                else:
                    self.log.msg(
                        'WARNING: No residual interstory drift ratio among '
                        'available demands. Irreparable damage cannot be '
                        'evaluated.'
                    )

            # load component model
            self.asset.load_cmp_model({'marginals': cmp_marginals})

            # generate component quantity sample
            self.asset.generate_cmp_sample()

        # if requested, load the quantity sample from a file
        if component_sample_file is not None:
            self.asset.load_cmp_sample(component_sample_file)

    def calculate_damage(  # noqa: C901
        self,
        length_unit: str | None,
        component_database: str,
        component_database_path: str | None = None,
        collapse_fragility: dict | None = None,
        irreparable_damage: dict | None = None,
        damage_process_approach: str | None = None,
        damage_process_file_path: str | None = None,
        custom_model_dir: str | None = None,
        scaling_specification: dict | None = None,
        *,
        is_for_water_network_assessment: bool = False,
    ) -> None:
        """
        Calculate damage.

        Parameters
        ----------
        length_unit : str, optional
            Unit of length to be used to add units to the demand data
            if needed.
        component_database: str
            Name of the component database.
        component_database_path: str or None
            Optional path to a component database file.
        collapse_fragility: dict or None
            Collapse fragility information.
        irreparable_damage: dict or None
            Information for irreparable damage.
        damage_process_approach: str or None
            Approach for the damage process.
        damage_process_file_path: str or None
            Optional path to a damage process file.
        custom_model_dir: str or None
            Optional directory for custom models.
        scaling_specification: dict, optional
            A dictionary defining the shift in median.
            Example: {'CMP-1-1': '*1.2', 'CMP-1-2': '/1.4'}
            The keys are individual components that should be present
            in the `capacity_sample`.  The values should be strings
            containing an operation followed by the value formatted as
            a float.  The operation can be '+' for addition, '-' for
            subtraction, '*' for multiplication, and '/' for division.
        is_for_water_network_assessment: bool
            Whether the assessment is for a water network.

        Raises
        ------
        ValueError
            With invalid combinations of arguments.

        """
        # load the fragility information
        component_db = []

        if not pd.isna(component_database):
            for method_name in [
                cdb.strip() for cdb in component_database.split(',')
            ]:
                if method_name == 'None':
                    continue

                # <backwards compatibility>
                if method_name.endswith(('csv', 'CSV')):
                    component_db_path = file_io.substitute_default_path(
                        [f'PelicunDefault/{method_name}'], log=self.log
                    )[0]
                else:
                    component_db_path = file_io.substitute_default_path(
                        [f'PelicunDefault/{method_name}/fragility.csv'], log=self.log
                    )[0]
                assert isinstance(component_db_path, str)

                if Path(component_db_path).is_file():
                    component_db.append(component_db_path)

        if component_database_path is not None:
            if 'CustomDLDataFolder' in component_database_path:
                if custom_model_dir is None:
                    msg = (
                        '`custom_model_dir` needs to be specified '
                        'when `component_database_path` includes CustomDLDataFolder.'
                    )
                    raise ValueError(msg)

                component_database_path = component_database_path.replace(
                    'CustomDLDataFolder', custom_model_dir
                )

            component_db += [component_database_path]

        component_db = component_db[::-1]

        # prepare additional fragility data

        # get the database header from the default P58 db
        p58_data = self.get_default_data('FEMA P-58', 'fragility')

        adf = pd.DataFrame(columns=p58_data.columns)

        if collapse_fragility:
            assert self.asset.cmp_marginal_params is not None

            if (
                'excessive.coll.DEM'
                in self.asset.cmp_marginal_params.index.get_level_values('cmp')
            ):
                # if there is story-specific evaluation
                coll_cmp_name = 'excessive.coll.DEM'
            else:
                # otherwise, for global collapse evaluation
                coll_cmp_name = 'collapse'

            adf.loc[coll_cmp_name, ('Demand', 'Directional')] = 1
            adf.loc[coll_cmp_name, ('Demand', 'Offset')] = 0

            coll_dem = collapse_fragility['DemandType']

            if '_' in coll_dem:
                coll_dem, coll_dem_spec = coll_dem.split('_')
            else:
                coll_dem_spec = None

            coll_dem_name = None
            for demand_name, demand_short in EDP_to_demand_type.items():
                if demand_short == coll_dem:
                    coll_dem_name = demand_name
                    break

            if coll_dem_name is None:
                msg = (
                    'A valid demand type acronym was not provided in'
                    'the configuration file. Please ensure the'
                    "'DemandType' field in the collapse fragility"
                    'section contains one of the recognized acronyms'
                    "(e.g., 'SA', 'PFA', 'PGA'). Refer to the"
                    "configuration file's 'collapse_fragility'"
                    'section.'
                )
                raise ValueError(msg)

            if coll_dem_spec is None:
                adf.loc[coll_cmp_name, ('Demand', 'Type')] = coll_dem_name

            else:
                adf.loc[coll_cmp_name, ('Demand', 'Type')] = (
                    f'{coll_dem_name}|{coll_dem_spec}'
                )

            if length_unit is None:
                msg = 'A length unit is required.'
                raise ValueError(msg)
            coll_dem_unit = _add_units(
                pd.DataFrame(
                    columns=[
                        f'{coll_dem}-1-1',
                    ]
                ),
                length_unit,
            ).iloc[0, 0]

            adf.loc[coll_cmp_name, ('Demand', 'Unit')] = coll_dem_unit
            adf.loc[coll_cmp_name, ('LS1', 'Family')] = collapse_fragility[
                'CapacityDistribution'
            ]
            adf.loc[coll_cmp_name, ('LS1', 'Theta_0')] = collapse_fragility[
                'CapacityMedian'
            ]
            adf.loc[coll_cmp_name, ('LS1', 'Theta_1')] = collapse_fragility[
                'Theta_1'
            ]
            adf.loc[coll_cmp_name, 'Incomplete'] = 0

            if coll_cmp_name != 'collapse':
                # for story-specific evaluation, we need to add a placeholder
                # fragility that will never trigger, but helps us aggregate
                # results in the end
                adf.loc['collapse', ('Demand', 'Directional')] = 1
                adf.loc['collapse', ('Demand', 'Offset')] = 0
                adf.loc['collapse', ('Demand', 'Type')] = 'One'
                adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
                adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
                adf.loc['collapse', 'Incomplete'] = 0

        elif not is_for_water_network_assessment:
            # add a placeholder collapse fragility that will never trigger
            # collapse, but allow damage processes to work with collapse

            adf.loc['collapse', ('Demand', 'Directional')] = 1
            adf.loc['collapse', ('Demand', 'Offset')] = 0
            adf.loc['collapse', ('Demand', 'Type')] = 'One'
            adf.loc['collapse', ('Demand', 'Unit')] = 'unitless'
            adf.loc['collapse', ('LS1', 'Theta_0')] = 1e10
            adf.loc['collapse', 'Incomplete'] = 0

        if irreparable_damage:
            # add excessive RID fragility according to settings provided in the
            # input file
            adf.loc['excessiveRID', ('Demand', 'Directional')] = 1
            adf.loc['excessiveRID', ('Demand', 'Offset')] = 0
            adf.loc['excessiveRID', ('Demand', 'Type')] = (
                'Residual Interstory Drift Ratio'
            )

            adf.loc['excessiveRID', ('Demand', 'Unit')] = 'unitless'
            adf.loc['excessiveRID', ('LS1', 'Theta_0')] = irreparable_damage[
                'DriftCapacityMedian'
            ]
            adf.loc['excessiveRID', ('LS1', 'Family')] = 'lognormal'
            adf.loc['excessiveRID', ('LS1', 'Theta_1')] = irreparable_damage[
                'DriftCapacityLogStd'
            ]

            adf.loc['excessiveRID', 'Incomplete'] = 0

            # add a placeholder irreparable fragility that will never trigger
            # damage, but allow damage processes to aggregate excessiveRID here
            adf.loc['irreparable', ('Demand', 'Directional')] = 1
            adf.loc['irreparable', ('Demand', 'Offset')] = 0
            adf.loc['irreparable', ('Demand', 'Type')] = 'One'
            adf.loc['irreparable', ('Demand', 'Unit')] = 'unitless'
            adf.loc['irreparable', ('LS1', 'Theta_0')] = 1e10
            adf.loc['irreparable', 'Incomplete'] = 0

        # TODO(AZ): we can improve this by creating a water
        # network-specific assessment class
        if is_for_water_network_assessment:
            # add a placeholder aggregate fragility that will never trigger
            # damage, but allow damage processes to aggregate the
            # various pipeline damages
            adf.loc['aggregate', ('Demand', 'Directional')] = 1
            adf.loc['aggregate', ('Demand', 'Offset')] = 0
            adf.loc['aggregate', ('Demand', 'Type')] = 'Peak Ground Velocity'
            adf.loc['aggregate', ('Demand', 'Unit')] = 'mps'
            adf.loc['aggregate', ('LS1', 'Theta_0')] = 1e10
            adf.loc['aggregate', ('LS2', 'Theta_0')] = 1e10
            adf.loc['aggregate', 'Incomplete'] = 0

        self.damage.load_model_parameters(
            [*component_db, adf],
            set(self.asset.list_unique_component_ids()),
        )

        # load the damage process if needed
        dmg_process = None
        if damage_process_approach is not None:  # noqa: PLR1702
            if damage_process_approach in default_damage_processes:
                dmg_process = default_damage_processes[damage_process_approach]

                # For Hazus Earthquake, we need to specify the component ids
                if damage_process_approach in {
                    'Hazus Earthquake',
                    'Hazus Earthquake - Buildings',
                    'Hazus Earthquake - Lifeline Facilities',
                }:
                    cmp_sample = self.asset.save_cmp_sample()
                    assert isinstance(cmp_sample, pd.DataFrame)

                    cmp_list = cmp_sample.columns.unique(level=0)

                    cmp_map = {'STR': '', 'LF': '', 'NSA': ''}

                    for cmp in cmp_list:
                        for cmp_type in cmp_map:
                            if cmp_type + '.' in cmp:
                                cmp_map[cmp_type] = cmp

                    new_dmg_process = dmg_process.copy()
                    for source_cmp, action in dmg_process.items():
                        # first, look at the source component id
                        new_source = None
                        for cmp_type, cmp_id in cmp_map.items():
                            if (cmp_type in source_cmp) and (cmp_id != ''):  # noqa: PLC1901
                                new_source = source_cmp.replace(cmp_type, cmp_id)
                                break

                        if new_source is not None:
                            new_dmg_process[new_source] = action
                            del new_dmg_process[source_cmp]
                        else:
                            new_source = source_cmp

                        # then, look at the target component ids
                        for ds_i, target_vals in action.items():
                            if isinstance(target_vals, str):
                                for cmp_type, cmp_id in cmp_map.items():
                                    if (cmp_type in target_vals) and (cmp_id != ''):  # noqa: PLC1901
                                        target_vals = target_vals.replace(  # noqa: PLW2901
                                            cmp_type, cmp_id
                                        )

                                new_target_vals = target_vals

                            else:
                                # we assume that target_vals is a list of str
                                new_target_vals = []

                                for target_val in target_vals:
                                    for cmp_type, cmp_id in cmp_map.items():
                                        if (cmp_type in target_val) and (
                                            cmp_id != ''  # noqa: PLC1901
                                        ):
                                            target_val = target_val.replace(  # noqa: PLW2901
                                                cmp_type, cmp_id
                                            )

                                    new_target_vals.append(target_val)

                            new_dmg_process[new_source][ds_i] = new_target_vals

                    dmg_process = new_dmg_process

                # Remove components not present in the asset model
                # from the source components of the damage process.
                asset_components = set(self.asset.list_unique_component_ids())
                filtered_dmg_process = {}
                for key in dmg_process:
                    component = key.split('_')[1]
                    if component in asset_components:
                        filtered_dmg_process[key] = dmg_process[key]
                dmg_process = filtered_dmg_process

            elif damage_process_approach == 'User Defined':
                if damage_process_file_path is None:
                    msg = (
                        'When `damage_process_approach` is set to '
                        '`User Defined`, a `damage_process_file_path` '
                        'needs to be provided.'
                    )
                    raise ValueError(msg)

                # load the damage process from a file
                with Path(damage_process_file_path).open(encoding='utf-8') as f:
                    dmg_process = json.load(f)

            elif damage_process_approach == 'None':
                # no damage process applied for the calculation
                dmg_process = None

            else:
                self.log.msg(
                    f'Prescribed Damage Process not recognized: '
                    f'`{damage_process_approach}`.'
                )

        # calculate damages
        self.damage.calculate(
            dmg_process=dmg_process,
            scaling_specification=scaling_specification,
        )

    def calculate_loss(
        self,
        loss_map_approach: str,
        occupancy_type: str,
        consequence_database: str,
        consequence_database_path: str | None = None,
        custom_model_dir: str | None = None,
        damage_process_approach: str = 'User Defined',
        replacement_cost_parameters: dict[str, float | str] | None = None,
        replacement_time_parameters: dict[str, float | str] | None = None,
        replacement_carbon_parameters: dict[str, float | str] | None = None,
        replacement_energy_parameters: dict[str, float | str] | None = None,
        loss_map_path: str | None = None,
        decision_variables: tuple[str, ...] | None = None,
        replacement_configuration: (
            tuple[uq.RandomVariableRegistry, dict[str, float]] | None
        ) = None,
        loss_combination_method: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate losses.

        Parameters
        ----------
        loss_map_approach: str
            Approach for the loss map generation. Can be either
            `User Defined` or `Automatic`.
        occupancy_type: str
            Occupancy type.
        consequence_database: str
            Name of the consequence database.
        consequence_database_path: str or None
            Optional path to a consequence database file.
        custom_model_dir: str or None
            Optional directory for custom models.
        damage_process_approach: str
            Damage process approach. Defaults to `User Defined`.
        replacement_cost_parameters: dict or None
            Parameters for replacement cost.
        replacement_time_parameters: dict or None
            Parameters for replacement time.
        replacement_carbon_parameters: dict or None
            Parameters for replacement carbon.
        replacement_energy_parameters: dict or None
            Parameters for replacement energy.
        loss_map_path: str or None
            Optional path to a loss map file.
        decision_variables: tuple[str] or None
            Optional decision variables for the assessment.
        replacement_configuration: tuple or None
            Loss thresholds of replacement consequences.
        loss_combination_method: str, optional
            String defining the method to use for combining losses for
            components that represent different demands.

        Returns
        -------
        tuple
            Dataframe with the aggregated loss of each realization,
            and another boolean dataframe with information on which DV
            thresholds were exceeded in each realization, triggering
            replacement. If no thresholds are specified it only
            contains False values.

        Raises
        ------
        ValueError
            When an invalid loss map approach is specified.

        """
        conseq_df, consequence_db = self.load_consequence_info(
            consequence_database,
            consequence_database_path,
            custom_model_dir,
        )

        # remove duplicates from conseq_df
        conseq_df = conseq_df.loc[conseq_df.index.unique(), :]

        # add the replacement consequence to the data
        adf = pd.DataFrame(
            columns=conseq_df.columns,
            index=pd.MultiIndex.from_tuples(
                [
                    ('replacement', 'Cost'),
                    ('replacement', 'Time'),
                    ('replacement', 'Carbon'),
                    ('replacement', 'Energy'),
                ]
            ),
        )

        _loss__add_replacement_cost(
            adf,
            damage_process_approach,
            unit=get(replacement_cost_parameters, 'Unit'),
            median=get(replacement_cost_parameters, 'Median'),
            distribution=get(replacement_cost_parameters, 'Distribution'),
            theta_1=get(replacement_cost_parameters, 'Theta_1'),
        )

        _loss__add_replacement_time(
            adf,
            damage_process_approach,
            conseq_df,
            occupancy_type=occupancy_type,
            unit=get(replacement_time_parameters, 'Unit'),
            median=get(replacement_time_parameters, 'Median'),
            distribution=get(replacement_time_parameters, 'Distribution'),
            theta_1=get(replacement_time_parameters, 'Theta_1'),
        )

        _loss__add_replacement_carbon(
            adf,
            damage_process_approach,
            unit=get(replacement_carbon_parameters, 'Unit'),
            median=get(replacement_carbon_parameters, 'Median'),
            distribution=get(replacement_carbon_parameters, 'Distribution'),
            theta_1=get(replacement_carbon_parameters, 'Theta_1'),
        )

        _loss__add_replacement_energy(
            adf,
            damage_process_approach,
            unit=get(replacement_energy_parameters, 'Unit'),
            median=get(replacement_energy_parameters, 'Median'),
            distribution=get(replacement_energy_parameters, 'Distribution'),
            theta_1=get(replacement_energy_parameters, 'Theta_1'),
        )

        # prepare the loss map
        loss_map = None
        if loss_map_approach == 'Automatic':
            # get the damage sample
            loss_map = _loss__map_auto(
                self, conseq_df, damage_process_approach, occupancy_type
            )

        elif loss_map_approach == 'User Defined':
            assert custom_model_dir is not None
            loss_map = _loss__map_user(custom_model_dir, loss_map_path)

        else:
            msg = f'Invalid MapApproach value: `{loss_map_approach}`.'
            raise ValueError(msg)

        # prepare additional loss map entries, if needed
        if 'DMG-collapse' not in loss_map.index:
            loss_map.loc['collapse', 'Repair'] = 'replacement'
            loss_map.loc['irreparable', 'Repair'] = 'replacement'

        if decision_variables:
            self.loss.decision_variables = decision_variables

        self.loss.add_loss_map(loss_map, loss_map_policy=None)
        self.loss.load_model_parameters([*consequence_db, adf])

        self.loss.calculate()

        if loss_combination_method is None:
            loss_combination = None

        elif loss_combination_method == 'Hazus Hurricane':
            # assemble the combination dict for wind and storm surge
            # open the base combination matrix
            file_path = file_io.substitute_default_path(
                ['PelicunDefault/Hazus Hurricane Wind/combine_wind_flood.csv'],
                log=self.log,
            )[0]
            assert isinstance(file_path, str)
            combination_array = pd.read_csv(
                file_path,
                index_col=None,
                header=None,
            ).to_numpy()

            # get the component names
            # assume that the first and second component in the loss map
            # are the wind and flood components, respectively
            wind_comp, flood_comp = loss_map.index.to_numpy()[[0, 1]]

            loss_combination = {
                'Cost': {
                    (wind_comp, flood_comp): combination_array,
                },
            }

        else:
            msg = f'Invalid loss combination method: `{loss_combination_method}`.'
            raise ValueError(msg)

        df_agg, exceedance_bool_df = self.loss.aggregate_losses(
            replacement_configuration, loss_combination, future=True
        )
        assert isinstance(df_agg, pd.DataFrame)
        assert isinstance(exceedance_bool_df, pd.DataFrame)
        return df_agg, exceedance_bool_df

    def load_consequence_info(
        self,
        consequence_database: str,
        consequence_database_path: str | None = None,
        custom_model_dir: str | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Load consequence information for the assessment.

        Parameters
        ----------
        consequence_database: str
            Name of the consequence database.
        consequence_database_path: str or None
            Optional path to a consequence database file.
        custom_model_dir: str or None
            Optional directory for custom models.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            A tuple containing:
            - A DataFrame with the consequence data.
            - A list of paths to the consequence databases used.

        Raises
        ------
        ValueError
            With invalid combinations of arguments.

        """
        # load the consequence information
        consequence_db = []
        conseq_df = pd.DataFrame()

        if not pd.isna(consequence_database):
            for method_name in [
                cdb.strip() for cdb in consequence_database.split(',')
            ]:
                if method_name == 'None':
                    continue

                # <backwards compatibility>
                if method_name.endswith(('csv', 'CSV')):
                    consequence_db_path = file_io.substitute_default_path(
                        [f'PelicunDefault/{method_name}'], log=self.log
                    )[0]
                else:
                    consequence_db_path = file_io.substitute_default_path(
                        [f'PelicunDefault/{method_name}/consequence_repair.csv'],
                        log=self.log,
                    )[0]
                assert isinstance(consequence_db_path, str)

                if Path(consequence_db_path).is_file():
                    consequence_db.append(consequence_db_path)

                    conseq_df = pd.concat(
                        [
                            conseq_df,
                            self.get_default_data(method_name, 'consequence_repair'),
                        ]
                    )
                    assert isinstance(conseq_df, pd.DataFrame)

                else:
                    # try loading loss functions instead
                    loss_db_path = file_io.substitute_default_path(
                        [f'PelicunDefault/{method_name}/loss_repair.csv'],
                        log=self.log,
                    )[0]

                    if Path(loss_db_path).is_file():
                        consequence_db.append(loss_db_path)

                        conseq_df = pd.concat(
                            [
                                conseq_df,
                                self.get_default_data(method_name, 'loss_repair'),
                            ]
                        )
                        assert isinstance(conseq_df, pd.DataFrame)

        if consequence_database_path is not None:
            if 'CustomDLDataFolder' in consequence_database_path:
                if custom_model_dir is None:
                    msg = (
                        'When `consequence_database_path` includes CustomDLDataFolder, '
                        '`custom_model_dir` needs to be specified as well.'
                    )
                    raise ValueError(msg)

                consequence_database_path = consequence_database_path.replace(
                    'CustomDLDataFolder', custom_model_dir
                )

            consequence_db += [consequence_database_path]

            extra_conseq_df = file_io.load_data(
                consequence_database_path,
                unit_conversion_factors=None,
                orientation=1,
                reindex=False,
                log=self.log,
            )
            assert isinstance(extra_conseq_df, pd.DataFrame)

            conseq_df = pd.concat([conseq_df, extra_conseq_df])
            assert isinstance(conseq_df, pd.DataFrame)

        consequence_db = consequence_db[::-1]

        return conseq_df, consequence_db


def _add_units(raw_demands: pd.DataFrame, length_unit: str) -> pd.DataFrame:
    """
    Add units to demand columns in a DataFrame.

    Parameters
    ----------
    raw_demands: pd.DataFrame
        The raw demand data to which units will be added.
    length_unit: str
        The unit of length to be used (e.g., 'in' for inches).

    Returns
    -------
    pd.DataFrame
        The DataFrame with units added to the appropriate demand columns.

    """
    demands = raw_demands.T

    demands.insert(0, 'Units', np.nan)

    if length_unit == 'in':
        length_unit = 'inch'

    demands = pd.DataFrame(
        base.convert_to_MultiIndex(demands, axis=0).sort_index(axis=0).T
    )

    nlevels_with_event_id = 4
    dem_level = 1 if demands.columns.nlevels == nlevels_with_event_id else 0

    # drop demands with no EDP type identified
    demands = demands.drop(
        demands.columns[demands.columns.get_level_values(dem_level) == ''],
        axis=1,
    )

    # assign units
    demand_cols = demands.columns.get_level_values(dem_level).to_list()

    # remove additional info from demand names
    demand_cols = [d.split('_')[0] for d in demand_cols]

    # acceleration
    acc_edps = ['PFA', 'PGA', 'SA']
    edp_mask = np.isin(demand_cols, acc_edps)

    if np.any(edp_mask):
        demands.iloc[0, edp_mask] = length_unit + 'ps2'  # type: ignore

    # speed
    speed_edps = ['PFV', 'PWS', 'PGV', 'SV']
    edp_mask = np.isin(demand_cols, speed_edps)

    if np.any(edp_mask):
        demands.iloc[0, edp_mask] = length_unit + 'ps'  # type: ignore

    # displacement
    disp_edps = ['PFD', 'PIH', 'SD', 'PGD']
    edp_mask = np.isin(demand_cols, disp_edps)

    if np.any(edp_mask):
        demands.iloc[0, edp_mask] = length_unit  # type: ignore

    # drift ratio
    rot_edps = ['PID', 'PRD', 'DWD', 'RDR', 'PMD', 'RID']
    edp_mask = np.isin(demand_cols, rot_edps)

    if np.any(edp_mask):
        demands.iloc[0, edp_mask] = 'unitless'  # type: ignore

    # convert back to simple header and return the DF
    return base.convert_to_SimpleIndex(demands, axis=1)


def _loss__add_replacement_energy(
    adf: pd.DataFrame,
    dl_method: str,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
) -> None:
    """
    Add replacement energy information.

    Parameters
    ----------
    adf : pandas.DataFrame
        Dataframe containing loss information.
    DL_method : str
        Supported methods are 'FEMA P-58'.
    unit : str, optional
        Unit for the energy value (e.g., 'MJ'). Defaults to None.
    median : float, optional
        Median replacement energy. If provided, it defines the base
        replacement energy value. Defaults to None.
    distribution : str, optional
        Distribution family to model uncertainty around the median
        energy (e.g., 'lognormal'). Required if `median` is
        provided. Defaults to None.
    theta_1 : float, optional
        Distribution parameter (e.g., standard deviation). Required if
        `distribution` is provided. Defaults to None.

    Notes
    -----
    If `median` is not provided, a default value is assigned based on
    the `DL_method`. For 'FEMA P-58', the default replacement energy
    value is 0 MJ.  For other methods, this consequence is removed
    from the dataframe entirely.
    """
    ren = ('replacement', 'Energy')
    if median is not None:
        # TODO(JVM): in this case we need unit (add config parser check)

        adf.loc[ren, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[ren, ('DV', 'Unit')] = unit
        adf.loc[ren, ('DS1', 'Theta_0')] = median

        if distribution is not None:
            # TODO(JVM): in this case we need theta_1 (add config parser check)

            adf.loc[ren, ('DS1', 'Family')] = distribution
            adf.loc[ren, ('DS1', 'Theta_1')] = theta_1
    elif dl_method == 'FEMA P-58':
        adf.loc[ren, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[ren, ('DV', 'Unit')] = 'MJ'
        adf.loc[ren, ('DS1', 'Theta_0')] = 0

    else:
        # for everything else, remove this consequence
        adf = adf.drop(ren)


def _loss__add_replacement_carbon(
    adf: pd.DataFrame,
    damage_process_approach: str,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
) -> None:
    """
    Add replacement carbon emission information.

    Parameters
    ----------
    adf : pandas.DataFrame
        Dataframe containing loss information.
    damage_process_approach : str
        Supported approaches include 'FEMA P-58'.
    unit : str, optional
        Unit for the carbon emission value (e.g., 'kg'). Defaults to
        None.
    median : float, optional
        Median replacement carbon emissions. If provided, it defines
        the base replacement carbon value. Defaults to None.
    distribution : str, optional
        Distribution family to model uncertainty around the median
        carbon emissions (e.g., 'lognormal'). Required if `median` is
        provided.  Defaults to None.
    theta_1 : float, optional
        Distribution parameter (e.g., standard deviation). Required if
        `distribution` is provided. Defaults to None.

    Notes
    -----
    If `median` is not provided, a default value is assigned based on
    the `damage_process_approach`. For 'FEMA P-58', the default
    replacement carbon emissions value is 0 kg. For other approaches,
    this consequence is removed from the dataframe entirely.
    """
    rcarb = ('replacement', 'Carbon')
    if median is not None:
        # TODO(JVM): in this case we need unit (add config parser check)

        adf.loc[rcarb, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rcarb, ('DV', 'Unit')] = unit
        adf.loc[rcarb, ('DS1', 'Theta_0')] = median

        if distribution is not None:
            # TODO(JVM): in this case we need theta_1 (add config parser check)

            adf.loc[rcarb, ('DS1', 'Family')] = distribution
            adf.loc[rcarb, ('DS1', 'Theta_1')] = theta_1
    elif damage_process_approach == 'FEMA P-58':
        adf.loc[rcarb, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rcarb, ('DV', 'Unit')] = 'kg'
        adf.loc[rcarb, ('DS1', 'Theta_0')] = 0

    else:
        # for everything else, remove this consequence
        adf = adf.drop(rcarb)


def _loss__add_replacement_time(
    adf: pd.DataFrame,
    damage_process_approach: str,
    conseq_df: pd.DataFrame,
    occupancy_type: str | None = None,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
) -> None:
    """
    Add replacement time information.

    Parameters
    ----------
    adf : pandas.DataFrame
        Dataframe containing loss information.
    damage_process_approach : str
        Supported approaches are 'FEMA P-58', 'Hazus Earthquake -
        Buildings'.
    conseq_df : pandas.DataFrame
        Dataframe containing consequence data for different damage
        states.
    occupancy_type : str, optional
        Type of occupancy, used to look up replacement time in the
        consequence dataframe for Hazus Earthquake approach. Defaults
        to None.
    unit : str, optional
        Unit for the replacement time (e.g., 'day, 'worker_day').
        Defaults to None.
    median : float, optional
        Median replacement time or loss ratio. If provided, it defines
        the base replacement time. Defaults to None.
    distribution : str, optional
        Distribution family to model uncertainty around the median
        time (e.g., 'lognormal'). Required if `median` is
        provided. Defaults to None.
    theta_1 : float, optional
        Distribution parameter (e.g., standard deviation). Required if
        `distribution` is provided. Defaults to None.

    Notes
    -----
    If `median` is not provided, a default value is assigned based on
    the `damage_process_approach`. For 'FEMA P-58', the default
    replacement time is 0 worker_days. For 'Hazus Earthquake -
    Buildings', the replacement time is fetched from `conseq_df` for
    the provided `occupancy_type` and corresponds to the total loss
    (damage state 5, DS5). In other cases, a placeholder value of 1 is
    used.

    """
    rt = ('replacement', 'Time')
    if median is not None:
        # TODO(JVM): in this case we need unit (add config parser check)

        adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rt, ('DV', 'Unit')] = unit
        adf.loc[rt, ('DS1', 'Theta_0')] = median

        if distribution is not None:
            # TODO(JVM): in this case we need theta_1 (add config parser check)

            adf.loc[rt, ('DS1', 'Family')] = distribution
            adf.loc[rt, ('DS1', 'Theta_1')] = theta_1
    elif damage_process_approach == 'FEMA P-58':
        adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rt, ('DV', 'Unit')] = 'worker_day'
        adf.loc[rt, ('DS1', 'Theta_0')] = 0

    # for Hazus EQ, use 1.0 as a loss_ratio
    elif damage_process_approach == 'Hazus Earthquake - Buildings':
        adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rt, ('DV', 'Unit')] = 'day'

        # load the replacement time that corresponds to total loss
        adf.loc[rt, ('DS1', 'Theta_0')] = conseq_df.loc[
            (f'STR.{occupancy_type}', 'Time'), ('DS5', 'Theta_0')
        ]

    elif damage_process_approach == 'Hazus Earthquake - Lifeline Facilities':
        adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rt, ('DV', 'Unit')] = 'day'

        # load the replacement time that corresponds to total loss
        adf.loc[rt, ('DS1', 'Theta_0')] = conseq_df.loc[
            (f'LF.{occupancy_type}', 'Time'), ('DS5', 'Theta_0')
        ]

    # otherwise, use 1 (and expect to have it defined by the user)
    else:
        adf.loc[rt, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rt, ('DV', 'Unit')] = 'loss_ratio'
        adf.loc[rt, ('DS1', 'Theta_0')] = 1


def _loss__add_replacement_cost(
    adf: pd.DataFrame,
    dl_method: str,
    unit: str | None = None,
    median: float | None = None,
    distribution: str | None = None,
    theta_1: float | None = None,
) -> None:
    """
    Add replacement cost information.

    Parameters
    ----------
    adf : pandas.DataFrame
        Dataframe containing loss information.
    DL_method : str
        Supported methods are 'FEMA P-58', 'Hazus Earthquake', and
        'Hazus Hurricane'.
    unit : str, optional
        Unit for the replacement cost (e.g., 'USD_2011',
        'loss_ratio'). Defaults to None.
    median : float, optional
        Median replacement cost or loss ratio. If provided, it defines
        the base replacement cost. Defaults to None.
    distribution : str, optional
        Distribution family to model uncertainty around the median
        cost (e.g., 'lognormal'). Required if `median` is
        provided. Defaults to None.
    theta_1 : float, optional
        Distribution parameter (e.g., standard deviation). Required if
        `distribution` is provided. Defaults to None.
    """
    rc = ('replacement', 'Cost')
    if median is not None:
        # TODO(JVM): in this case we need unit (add config parser check)

        adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rc, ('DV', 'Unit')] = unit
        adf.loc[rc, ('DS1', 'Theta_0')] = median

        if distribution is not None:
            # TODO(JVM): in this case we need theta_1 (add config parser check)

            adf.loc[rc, ('DS1', 'Family')] = distribution
            adf.loc[rc, ('DS1', 'Theta_1')] = theta_1

    elif dl_method == 'FEMA P-58':
        adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rc, ('DV', 'Unit')] = 'USD_2011'
        adf.loc[rc, ('DS1', 'Theta_0')] = 0

    # for Hazus EQ and HU, use 1.0 as a loss_ratio
    elif dl_method in {'Hazus Earthquake', 'Hazus Hurricane'}:
        adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'

        # store the replacement cost that corresponds to total loss
        adf.loc[rc, ('DS1', 'Theta_0')] = 1.00

    # otherwise, use 1 (and expect to have it defined by the user)
    else:
        adf.loc[rc, ('Quantity', 'Unit')] = '1 EA'
        adf.loc[rc, ('DV', 'Unit')] = 'loss_ratio'
        adf.loc[rc, ('DS1', 'Theta_0')] = 1


def _loss__map_user(
    custom_model_dir: str, loss_map_path: str | None = None
) -> pd.DataFrame:
    """
    Load a user-defined loss map from a specified path.

    Parameters
    ----------
    custom_model_dir : str
        Directory containing custom models.
    loss_map_path : str, optional
        Path to the loss map file. The path can include a placeholder
        'CustomDLDataFolder' that will be replaced by
        `custom_model_dir`.  If not provided, raises a ValueError.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the loss map information.

    Raises
    ------
    ValueError
        If `loss_map_path` is not provided.

    """
    if loss_map_path is not None:
        loss_map_path = loss_map_path.replace('CustomDLDataFolder', custom_model_dir)

    else:
        msg = 'Missing loss map path.'
        raise ValueError(msg)

    return pd.read_csv(loss_map_path, index_col=0)


def _loss__map_auto(
    assessment: DLCalculationAssessment,
    conseq_df: pd.DataFrame,
    dl_method: str,
    occupancy_type: str | None = None,
) -> pd.DataFrame:
    """
    Automatically generate a loss map.

    Automatically generate a loss map based on the damage sample and
    the consequence database.

    Parameters
    ----------
    assessment : AssessmentBase
        The assessment object containing the damage model and sample.
    conseq_df : pandas.DataFrame
        DataFrame containing consequence data for different damage
        states.
    DL_method : str
        Damage loss method, which defines how the loss map is
        generated.  Supported methods are 'FEMA P-58', 'Hazus
        Earthquake', 'Hazus Hurricane', and 'Hazus Earthquake
        Transportation'.
    occupancy_type : str, optional
        Occupancy type, used to map damage components to the correct
        loss models in Hazus Earthquake methods. Defaults to None.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the automatically generated loss map,
        where the index corresponds to the damage components and the
        values indicate the associated loss models.

    Notes
    -----
    - For 'FEMA P-58' and 'Hazus Hurricane', the method assumes that
      fragility and consequence data have matching component IDs.
    - For 'Hazus Earthquake' and 'Hazus Earthquake Transportation',
      the method assumes that consequence archetypes are only
      differentiated by occupancy type.

    """
    identical_damage_and_conseqence_ids = True
    if dl_method.startswith('Hazus Earthquake'):
        identical_damage_and_conseqence_ids = False

    # get the component sample
    asset_sample = assessment.asset.save_cmp_sample()
    assert isinstance(asset_sample, pd.DataFrame)

    # get the damage sample
    # TODO(AZ): check why the damage sample was needed here
    # dmg_sample = assessment.damage.save_sample()
    # assert isinstance(dmg_sample, pd.DataFrame)

    # create a mapping for all components that are also in
    # the prescribed consequence database
    asset_cmps = asset_sample.columns.unique(level='cmp')
    loss_cmps = conseq_df.index.unique(level=0)

    drivers = []
    loss_models = []

    if identical_damage_and_conseqence_ids:
        # with these methods, we assume fragility and consequence data
        # have the same IDs

        for asset_cmp in asset_cmps:
            if asset_cmp == 'collapse':
                continue

            if asset_cmp in loss_cmps:
                drivers.append(asset_cmp)
                loss_models.append(asset_cmp)

    else:
        # Currently, we only get here with Hazus Earthquake
        # with Hazus Earthquake we assume that consequence
        # archetypes are only differentiated by occupancy type
        for asset_cmp in asset_cmps:
            if asset_cmp == 'collapse':
                continue

            cmp_class = asset_cmp.split('.')[0]
            if occupancy_type is not None:
                loss_cmp = f'{cmp_class}.{occupancy_type}'
            else:
                loss_cmp = cmp_class

            if loss_cmp in loss_cmps:
                drivers.append(asset_cmp)
                loss_models.append(loss_cmp)

    return pd.DataFrame(loss_models, columns=['Repair'], index=drivers)


class TimeBasedAssessment:
    """Time-based assessment."""
