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
This module has classes and methods that control the performance assessment.

.. rubric:: Contents

.. autosummary::

    Assessment

"""

from __future__ import annotations
from typing import Any
import json
import pandas as pd
from pelicun import base
from pelicun import uq
from pelicun import file_io
from pelicun import model
from pelicun.__init__ import __version__ as pelicun_version  # type: ignore


class Assessment:
    """
    Assessment objects manage the models, data, and calculations in pelicun.

    Parameters
    ----------
    demand: DemandModel
        ...
    asset: AssetModel
        ...
    damage: DamageModel
        ...
    repair: RepairModel
        ...
    stories: int
        Number of stories.
    options: Options
        Options object.
    """

    __slots__ = [
        'stories',
        'options',
        'unit_conversion_factors',
        'log',
        'demand',
        'asset',
        'damage',
        'loss',
    ]

    def __init__(self, config_options: dict[str, Any] | None = None):
        """
        Initializes an Assessment object.

        Parameters
        ----------
        config_options (Optional[dict]):
            User-specified configuration dictionary.
        """
        self.stories = None
        self.options = base.Options(config_options, self)
        self.unit_conversion_factors = base.parse_units(self.options.units_file)

        self.log = self.options.log
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
    def bldg_repair(self):
        """
        <backwards compatibility>

        Returns
        -------
        model.LossModel
            The loss model.

        """
        self.log.warn(
            '`.bldg_repair` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `.loss` instead.'
        )

        return self.loss

    @property
    def repair(self):
        """
        <backwards compatibility>

        Returns
        -------
        RepairModel_DS
            The damage state-driven component loss model.

        """
        self.log.warn(
            '`.repair` is deprecated and will be dropped in '
            'future versions of pelicun. '
            'Please use `.loss` instead.'
        )
        return self.loss

    def get_default_data(self, data_name: str) -> pd.DataFrame:
        """
        Loads a default data file by name and returns it. This method
        is specifically designed to access predefined CSV files from a
        structured directory path related to the SimCenter fragility
        library.

        Parameters
        ----------
        data_name : str
            The name of the CSV file to be loaded, without the '.csv'
            extension. This name is used to construct the full path to
            the file.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the data loaded from the
            specified CSV file.
        """

        # <backwards compatibility>
        if 'fragility_DB' in data_name:
            data_name = data_name.replace('fragility_DB', 'damage_DB')
            self.log.warn(
                '`fragility_DB` is deprecated and will be dropped in '
                'future versions of pelicun. '
                'Please use `damage_DB` instead.'
            )
        if 'bldg_repair_DB' in data_name:
            data_name = data_name.replace('bldg_repair_DB', 'loss_repair_DB')
            self.log.warn(
                '`bldg_repair_DB` is deprecated and will be dropped in '
                'future versions of pelicun. '
                'Please use `loss_repair_DB` instead.'
            )

        data_path = f'{base.pelicun_path}/resources/SimCenterDBDL/{data_name}.csv'

        return file_io.load_data(
            data_path, None, orientation=1, reindex=False, log=self.log
        )

    def get_default_metadata(self, data_name: str) -> dict:
        """
        Load a default metadata file and pass it to the user.

        Parameters
        ----------
        data_name: string
            Name of the json file to be loaded

        Returns
        -------
        dict
            Default metadata

        """

        # <backwards compatibility>
        if 'fragility_DB' in data_name:
            data_name = data_name.replace('fragility_DB', 'damage_DB')
            self.log.warn(
                '`fragility_DB` is deprecated and will be dropped in '
                'future versions of pelicun. Please use `damage_DB` instead.'
            )
        data_path = f'{base.pelicun_path}/resources/SimCenterDBDL/{data_name}.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def calc_unit_scale_factor(self, unit: str) -> float:
        """
        Determines the scale factor from input unit to the
        corresponding base unit

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
            unit_count, unit_name = unit_lst
            unit_count = float(unit_count)

        else:
            unit_count = 1
            unit_name = unit_lst[0]

        try:
            scale_factor = unit_count * self.unit_conversion_factors[unit_name]

        except KeyError as exc:
            raise KeyError(
                f"Specified unit not recognized: {unit_count} {unit_name}"
            ) from exc

        return scale_factor

    def scale_factor(self, unit: str | None) -> float:
        """
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
                raise ValueError(f"Unknown unit: {unit}")
        else:
            scale_factor = 1.0

        return scale_factor

    def calculate_damage(
        self,
        num_stories: int,
        demand_config: dict,
        demand_data_source: str | dict,
        cmp_data_source: str | dict[str, pd.DataFrame],
        damage_data_paths: list[str | pd.DataFrame],
        dmg_process: dict | None = None,
        scaling_specification: dict | None = None,
        yield_drift_configuration: dict | None = None,
        collapse_fragility_configuration: dict | None = None,
        block_batch_size: int = 1000,
    ) -> None:
        """
        Calculates damage.

        Paraemters
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
        cmp_data_source : str or dict
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
        yield_drift_configuration: dict
            Dictionary containing the following keys-values:
            - params : dict
            A dictionary containing parameters required for the
            estimation method, such as 'yield_drift', which is the
            drift at which yielding is expected to occur.
            - method : str, optional
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
        # TODO: when we build the API docs, ensure the above is
        # properly rendered.

        self.demand.load_model(demand_data_source)
        self.demand.generate_sample(demand_config)

        if yield_drift_configuration:
            self.demand.estimate_RID_and_adjust_sample(
                yield_drift_configuration['parameters'],
                yield_drift_configuration['method'],
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
            damage_data_paths, self.asset.list_unique_component_ids(as_set=True)
        )
        self.damage.calculate(dmg_process, block_batch_size, scaling_specification)

    def calculate_loss(
        self,
        decision_variables: tuple[str, ...],
        loss_model_data_paths: list[str | pd.DataFrame],
        loss_map_path: str | pd.DataFrame | None = None,
        loss_map_policy: str | None = None,
    ):
        """
        Calculates loss.

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
    ):
        """
        Aggregates losses.

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
            for multuple decision variables at the same
            realization. The consequence keyword `replacement` is
            reserved to represent exclusive triggering of the
            replacement consequences, and other consequences are
            ignored for those realizations where replacement is
            triggered. When assigned to None, then `replacement` is
            still treated as an exclusive consequence (other
            consequences are set to zero when replacement is nonzero)
            but it is not being additinally triggered by the
            exceedance of any thresholds. The aggregated loss sample
            conains an additional column with information on whether
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

        Note
        ----
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

        Raises
        ------
        ValueError
            When inputs are invalid.

        """

        return self.loss.aggregate_losses(
            replacement_configuration, loss_combination, future=True
        )


class ComponentLevelAssessment(Assessment):
    """
    Detailed risk assessment with component-level resolution.

    """


class SubAssemblyLevelAssessment(Assessment):
    """
    Risk assessment with subassembly-level resolution.

    """


class PortfolioLevelAssessment(Assessment):
    """
    High-level assessment of a portfolio of assets and smaller
    resolution.

    """

    def calculate_damage(
        self,
        num_stories: int,
        demand_config: dict,
        demand_data_source: str | dict,
        cmp_data_source: str | dict[str, pd.DataFrame],
        damage_data_paths: list[str | pd.DataFrame],
        dmg_process: dict | None = None,
        scaling_specification: dict | None = None,
        block_batch_size: int = 1000,
    ) -> None:
        """
        Calculates damage.

        Paraemters
        ----------
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
        cmp_data_source : str or dict
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
        block_batch_size: int
            Maximum number of components in each batch.

        """
        # TODO: when we build the API docs, ensure the above is
        # properly rendered.

        self.demand.load_model(demand_data_source)
        self.demand.generate_sample(demand_config)
        self.asset.load_cmp_model(cmp_data_source)
        self.asset.generate_cmp_sample()

        self.damage.load_model_parameters(
            damage_data_paths, self.asset.list_unique_component_ids(as_set=True)
        )
        self.damage.calculate(dmg_process, block_batch_size, scaling_specification)


class TimeBasedAssessment:
    """
    Time-based assessment.

    """


def test():
    """
    This code is temporary, and will eventually be turned into a unit
    test.

    """

    # pylint: disable=import-outside-toplevel
    import tempfile
    import numpy as np

    # variable setup

    temp_dir = tempfile.mkdtemp()
    config = {"PrintLog": True, "Seed": 415, "LogFile": f'{temp_dir}/log_file.txt'}

    sample_size = 10000

    demand_data = file_io.load_data(
        'pelicun/tests/validation/2/data/demand_data.csv',
        unit_conversion_factors=None,
        reindex=False,
    )
    ndims = len(demand_data)
    perfect_correlation = pd.DataFrame(
        np.ones((ndims, ndims)), columns=demand_data.index, index=demand_data.index
    )
    demand_data_dct = {'marginals': demand_data, 'correlation': perfect_correlation}

    num_stories = 1

    cmp_marginals = pd.read_csv(
        'pelicun/tests/validation/2/data/CMP_marginals.csv', index_col=0
    )
    cmp_marginals['Blocks'] = cmp_marginals['Blocks']

    cmp_model_input = {'marginals': cmp_marginals}

    damage_model_parameters = [
        'pelicun/tests/validation/2/data/additional_damage_db.csv',
        'PelicunDefault/damage_DB_FEMA_P58_2nd.csv',
    ]

    dmg_process = {
        "1_collapse": {"DS1": "ALL_NA"},
        "2_excessiveRID": {"DS1": "irreparable_DS1"},
    }

    loss_model_parameters = [
        'pelicun/tests/validation/2/data/additional_consequences.csv',
        'pelicun/tests/validation/2/data/additional_loss_functions.csv',
        "PelicunDefault/loss_repair_DB_FEMA_P58_2nd.csv",
    ]

    loss_map = pd.DataFrame(
        ['replacement', 'replacement'],
        columns=['Repair'],
        index=['collapse', 'irreparable'],
    )

    yield_drift_configuration = {
        'method': 'FEMA P-58',
        'parameters': {'yield_drift': 0.01},
    }

    collapse_fragility_configuration = {
        'label': 'SA_1.13',
        'value': 1.50,
        'unit': 'g',
    }

    decision_variables = ('Cost', 'Time')
    loss_map_policy = 'fill'

    demand_config = {"SampleSize": sample_size}

    asmnt = Assessment(config)
    asmnt.calculate_damage(
        num_stories,
        demand_config,
        demand_data_dct,
        cmp_model_input,
        damage_model_parameters,
        dmg_process,
        None,
        yield_drift_configuration,
        collapse_fragility_configuration,
    )
    asmnt.calculate_loss(
        decision_variables,
        loss_model_parameters,
        loss_map,
        loss_map_policy,
    )
    x1, x2 = asmnt.aggregate_loss()
    print(x1)
    print(x2)

    # TODO
    # - [X] expose all method arguments
    # - [X] write docstrings, copy parts from the other methods
    # - [X] copy the methods to the child classes and redefine them
    # - see how we can utilize these new objects in DL_calculation
    # - add a test of each assessment type
    # - implement TimeBasedAssessment
    # - test TimeBasedAssessment
