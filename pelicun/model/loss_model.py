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
This file defines Loss model objects and their methods.

.. rubric:: Contents

.. autosummary::

    LossModel

"""

import warnings
import numpy as np
import pandas as pd
from pelicun.model.pelicun_model import PelicunModel
from pelicun import base
from pelicun import uq
from pelicun import file_io


idx = base.idx


class LossModel(PelicunModel):
    """
    Manages loss information used in assessments.

    Groups a loss model for components with Damage States (DS) and
    another for components with Loss Functions (LF).

    """

    __slots__ = ['ds_model', 'lf_model', '_loss_map' 'decision_variables']

    def __init__(
        self, assessment, decision_variables=('Carbon', 'Cost', 'Energy', 'Time')
    ):
        """
        Initializes LossModel objects.

        Parameters
        ----------
        assessment: pelicun.Assessment
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

    @property
    def decision_variables(self):
        """
        Retrieves the decision variables to be used in the loss
        calculations.

        Returns
        -------
        tuple
            Decision variables.
        """
        # pick the object from one of the models
        # it's the same for the other(s).
        return self.ds_model.decision_variables

    @decision_variables.setter
    def decision_variables(self, decision_variables):
        """
        Sets the decision variables to be used in the loss
        calculations.

        Supported: {`Cost`, `Time`, `Energy`, `Carbon`}.
        Could also be any other string, as long as the provided loss
        parameters contain that decision variable.

        """
        # assign the same DVs to the included loss models.
        for model in self._loss_models:
            model.decision_variables = decision_variables

    def add_loss_map(self, loss_map_path=None, loss_map_policy=None):
        """
        Add a loss map to the loss model. A loss map defines what loss
        parameter definition should be used for each component ID in
        the asset model.

        Parameters
        ----------
        loss_map_path: str or pd.DataFrame or None
            Path to a csv file or DataFrame object that maps
            components IDs to their loss parameter definitions.
        loss_map_policy: str or None
            If None, does not modify the loss map.
            If set to `fill`, each component ID that is present in the
            asset model but not in the loss map is mapped to itself.

        Raises
        ------
        ValueError
            If both arguments are None.

        """

        self.log_msg('Loading loss map...')

        # If no loss map is provided and no default is requested,
        # there is no loss map and we can't proceed.
        if loss_map_path is None and loss_map_policy is None:
            raise ValueError(
                'Please provide a loss map and/or a loss map extension policy.'
            )

        # get a list of unique component IDs
        cmp_set = self._asmnt.asset.list_unique_component_ids(as_set=True)

        if loss_map_path is not None:
            self.log_msg('Loss map is provided.', prepend_timestamp=False)
            # Read the loss map into a variable
            loss_map = file_io.load_data(
                loss_map_path,
                None,
                orientation=1,
                reindex=False,
                log=self._asmnt.log,
            )
        else:
            self.log_msg('Using default loss map.', prepend_timestamp=False)
            # Instantiate an empty loss map.
            loss_map = pd.DataFrame({'Repair': pd.Series(dtype='object')})
            loss_map.index = loss_map.index.astype('object')

        if loss_map_policy == 'fill':
            # Populate missing rows with cmp_id -> cmp_id
            for component in cmp_set:
                if component not in loss_map.index:
                    loss_map.loc[component, :] = component

        elif loss_map_policy is None:
            # Don't do anything.
            pass

        # TODO: add other loss map policies.
        else:
            raise ValueError(f'Unknown loss map policy: `{loss_map_policy}`.')

        # Assign the loss map to the available loss models
        self._loss_map = loss_map

        self.log_msg('Loss map loaded successfully.', prepend_timestamp=True)

    def load_model_parameters(self, data_paths):
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

        Raises
        ------
        ValueError
            If the method can't parse the loss parameters in the
            specified paths.

        """
        self.log_div()
        self.log_msg('Loading loss parameters...')

        # replace `PelicunDefault/` flag with default data path
        data_paths = file_io.substitute_default_path(data_paths)

        #
        # load loss parameter data into the models
        #

        for data_path in data_paths:
            data = file_io.load_data(
                data_path, None, orientation=1, reindex=False, log=self._asmnt.log
            )
            # determine if the loss model parameters are for damage
            # states or loss functions
            if _is_for_ds_model(data):
                self.ds_model._load_model_parameters(data)
            elif _is_for_lf_model(data):
                self.lf_model._load_model_parameters(data)
            else:
                raise ValueError(f'Invalid loss model parameters: {data_path}')

        self.log_msg(
            'Loss model parameters loaded successfully.', prepend_timestamp=False
        )

        #
        # remove items
        #

        self.log_msg(
            'Removing unused loss model parameters.', prepend_timestamp=False
        )

        for loss_model in self._loss_models:
            # drop unused loss parameter definitions
            loss_model._drop_unused_loss_parameters(self._loss_map)
            # remove components with incomplete loss parameters
            loss_model._remove_incomplete_components()

        # drop unused damage state columns
        self.ds_model._drop_unused_damage_states()

        #
        # convert units
        #

        self.log_msg(
            'Converting loss model parameter units.', prepend_timestamp=False
        )
        for loss_model in self._loss_models:
            loss_model._convert_loss_parameter_units()

        #
        # verify loss parameter availability
        #

        self.log_msg(
            'Checking loss model parameter '
            'availability for all components in the asset model.',
            prepend_timestamp=False,
        )
        missing = self._ensure_loss_parameter_availability()

        self._missing = missing

    def calculate(self, sample_size=None):
        """
        Calculate the loss of each component block.

        """
        self.log_div()
        self.log_msg('Calculating losses...')

        if not sample_size:
            warnings.warn(
                'Using default sample size is deprecated and will '
                'be removed in future versions. '
                'Please provide the `sample_size` explicitly.',
                DeprecationWarning,
            )
            sample_size = self._asmnt.demand.sample.shape[0]

        # Get the damaged quantities in each damage state for each
        # component of interest.
        # TODO: FIND A WAY to avoid making a copy of this.
        dmg_quantities = self._asmnt.damage.ds_model.sample.copy()
        self.ds_model._calculate(dmg_quantities, sample_size)

        self.lf_model._calculate(sample_size)

        self.log_msg("Loss calculation successful.")

    def aggregate_losses(self):
        """
        Aggregates repair consequences across components.

        Returns
        -------
        DataFrame
            A DataFrame containing aggregated repair
            consequences. Columns include:
            - 'repair_cost': Total repair costs across all components.
            - 'repair_time-parallel': Minimum possible repair time
              assuming repairs are conducted in parallel.
            - 'repair_time-sequential': Maximum possible repair time
              assuming sequential repairs.
            - 'repair_carbon': Total carbon emissions associated with
              repairs.
            - 'repair_energy': Total energy usage associated with
              repairs.
            Each of these columns is summed or calculated based on the
            repair data available.

        """

        self.log_div()
        self.log_msg("Aggregating repair consequences...")

        ds_model_losses = self.ds_model._aggregate_losses()

        self.log_msg("Repair consequences successfully aggregated.")

        return ds_model_losses

    @property
    def _loss_models(self):
        return (self.ds_model, self.lf_model)

    @property
    def _loss_map(self):
        """
        Returns
        -------
        pd.DataFrame
            The loss map.

        """
        # Retrieve the dataframe from one of the included loss models.
        # We use a single loss map for all.
        return self.ds_model._loss_map

    @_loss_map.setter
    def _loss_map(self, loss_map):
        """
        Sets the loss map.

        Parameters
        ----------
        loss_map: pd.DataFrame
            The loss map.

        """
        # Add the dataframe to the included loss models.
        # We use a single loss map for all.
        for model in self._loss_models:
            model._loss_map = loss_map

    @property
    def _missing(self):
        """
        Returns
        -------
        set
            Set containing tuples identifying missing loss parameter
            definitions.

        """
        return self.ds_model._missing

    @_missing.setter
    def _missing(self, missing):
        """
        Assigns missing parameter definitions to the loss models.

        Parameters
        ----------
        missing: set
            Set containing tuples identifying missing loss parameter
            definitions.

        """
        for model in self._loss_models:
            model._missing = missing

    def _ensure_loss_parameter_availability(self):
        """
        Makes sure that all components have loss parameters.

        Returns
        -------
        list
            List of component IDs with missing loss parameters.

        """

        #
        # Repair Models (currently the only type supported)
        #

        required = []
        for dv in self.decision_variables:
            required.extend(
                [(component, dv) for component in self._loss_map['Repair']]
            )
        missing_set = set(required)

        for model in (self.ds_model, self.lf_model):
            missing_set = missing_set - model._get_available()

        if missing_set:
            self.log_msg(
                f"\n"
                f"WARNING: The loss model does not provide "
                f"loss information for the following component(s) "
                f"in the asset model: {sorted(list(missing_set))}."
                f"\n",
                prepend_timestamp=False,
            )

        return missing_set


class RepairModel_Base(PelicunModel):
    """
    Base class for loss models

    """

    def __init__(self, assessment):
        """
        Initializes RepairModel_Base objects.

        Parameters
        ----------
        assessment: pelicun.Assessment
            Parent assessment

        """
        super().__init__(assessment)

        self.loss_params = None
        self.sample = None
        self.consequence = 'Repair'

    def _load_model_parameters(self, data):
        """
        Load model parameters from a dataframe, extending those
        already available. Parameters already defined take precedence,
        i.e. redefinitions of parameters are ignored.

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

    def _drop_unused_loss_parameters(self, loss_map):
        """
        Removes loss parameter definitions for component IDs not
        present in the loss map.

        Parameters
        ----------
        loss_map_path: str or pd.DataFrame or None
            Path to a csv file or DataFrame object that maps
            components IDs to their loss parameter definitions.
            Components in the asset model that are omitted from the
            provided loss map are mapped to the same ID.


        """

        if self.loss_params is None:
            return

        # get a list of unique component IDs
        cmp_set = set(loss_map['Repair'].unique())

        cmp_mask = self.loss_params.index.get_level_values(0).isin(cmp_set, level=0)
        self.loss_params = self.loss_params.iloc[cmp_mask, :]

    def _remove_incomplete_components(self):
        """
        Removes components that have incomplete loss model
        definitions from the loss model parameters.

        """
        if self.loss_params is None:
            return

        if ('Incomplete', '') not in self.loss_params.columns:
            return

        cmp_incomplete_idx = self.loss_params.loc[
            self.loss_params[('Incomplete', '')] == 1
        ].index

        self.loss_params.drop(cmp_incomplete_idx, inplace=True)

        if len(cmp_incomplete_idx) > 0:
            self.log_msg(
                f"\n"
                f"WARNING: Loss model information is incomplete for "
                f"the following component(s) "
                f"{cmp_incomplete_idx.to_list()}. They "
                f"were removed from the analysis."
                f"\n",
                prepend_timestamp=False,
            )

    def _get_available(self):
        """
        Get a set of components for which loss parameters are
        available.
        """
        if self.loss_params is not None:
            cmp_list = self.loss_params.index.to_list()
            return set(cmp_list)
        return set()


class RepairModel_DS(RepairModel_Base):
    """
    Manages repair consequences driven by components that are modeled
    with discrete Damage States (DS)

    """

    def save_sample(self, filepath=None, save_units=False):
        """
        Saves the loss sample to a CSV file or returns it as a
        DataFrame with optional units.

        This method handles the storage of a sample of loss estimates,
        which can either be saved directly to a file or returned as a
        DataFrame for further manipulation. When saving to a file,
        additional information such as unit conversion factors and
        column units can be included. If the data is not being saved
        to a file, the method can return the DataFrame with or without
        units as specified.

        Parameters
        ----------
        filepath : str, optional
            The path to the file where the loss sample should be
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
            - DataFrame containing the loss sample.
            - Optionally, a Series containing the units for each
              column if `save_units` is True.

        Raises
        ------
        IOError
            Raises an IOError if there is an issue saving the file to
            the specified `filepath`.
        """
        self.log_div()
        if filepath is not None:
            self.log_msg('Saving loss sample...')

        cmp_units = self.loss_params[('DV', 'Unit')]
        dv_units = pd.Series(index=self.sample.columns, name='Units', dtype='object')

        valid_dv_types = dv_units.index.unique(level=0)
        valid_cmp_ids = dv_units.index.unique(level=1)

        for cmp_id, dv_type in cmp_units.index:
            if (dv_type in valid_dv_types) and (cmp_id in valid_cmp_ids):
                dv_units.loc[(dv_type, cmp_id)] = cmp_units.at[(cmp_id, dv_type)]

        res = file_io.save_to_csv(
            self.sample,
            filepath,
            units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )

        if filepath is not None:
            self.log_msg('Loss sample successfully saved.', prepend_timestamp=False)
            return None

        # else:
        units = res.loc["Units"]
        res.drop("Units", inplace=True)

        if save_units:
            return res.astype(float), units

        return res.astype(float)

    def load_sample(self, filepath):
        """
        Load loss sample data.

        """
        self.log_div()
        self.log_msg('Loading loss sample...')

        self.sample = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors, log=self._asmnt.log
        )

        self.log_msg('Loss sample successfully loaded.', prepend_timestamp=False)

    def _calculate(self, dmg_quantities, sample_size):
        """
        Calculate the consequences of each damage state-driven
        component block damage in the asset.

        Parameters
        ----------
        dmg_quantities: DataFrame
            A table with the quantity of damage experienced in each
            damage state of each performance group at each location
            and direction. You can use the prepare_dmg_quantities
            method in the DamageModel to get such a DF.
        sample_size: integer
            The number of realizations to generate.

        Raises
        ------
        ValueError
            When any Loss Driver is not recognized.

        """

        # If everything is undamaged there are no losses
        if set(dmg_quantities.columns.get_level_values('ds')) == {'0'}:
            self.sample = None
            self.log_msg(
                "There is no damage---DV sample is set to None.",
                prepend_timestamp=False,
            )
            return

        # calculate the quantities for economies of scale
        self.log_msg("\nAggregating damage quantities...", prepend_timestamp=False)

        if self._asmnt.options.eco_scale["AcrossFloors"]:
            if self._asmnt.options.eco_scale["AcrossDamageStates"]:
                eco_levels = [
                    0,
                ]
                eco_columns = [
                    'cmp',
                ]

            else:
                eco_levels = [0, 4]
                eco_columns = ['cmp', 'ds']

        elif self._asmnt.options.eco_scale["AcrossDamageStates"]:
            eco_levels = [0, 1]
            eco_columns = ['cmp', 'loc']

        else:
            eco_levels = [0, 1, 4]
            eco_columns = ['cmp', 'loc', 'ds']

        eco_group = dmg_quantities.groupby(level=eco_levels, axis=1)
        eco_qnt = eco_group.sum().mask(eco_group.count() == 0, np.nan)
        assert eco_qnt.columns.names == eco_columns

        self.log_msg(
            "Successfully aggregated damage quantities.", prepend_timestamp=False
        )

        # apply the median functions, if needed, to get median consequences for
        # each realization
        self.log_msg(
            "\nCalculating the median repair consequences...",
            prepend_timestamp=False,
        )

        medians = self._calc_median_consequence(eco_qnt)

        self.log_msg(
            "Successfully determined median repair consequences.",
            prepend_timestamp=False,
        )

        # combine the median consequences with the samples of deviation from the
        # median to get the consequence realizations.
        self.log_msg(
            "\nConsidering deviations from the median values to obtain "
            "random DV sample..."
        )

        self.log_msg(
            "Preparing random variables for repair cost and time...",
            prepend_timestamp=False,
        )
        RV_reg = self._create_DV_RVs(dmg_quantities.columns)

        if RV_reg is not None:
            RV_reg.generate_sample(
                sample_size=sample_size, method=self._asmnt.options.sampling_method
            )

            std_sample = base.convert_to_MultiIndex(
                pd.DataFrame(RV_reg.RV_sample), axis=1
            )
            std_sample.columns.names = ['dv', 'cmp', 'ds', 'loc', 'dir', 'uid']
            std_sample.sort_index(axis=1, inplace=True)

        else:
            std_sample = None

        self.log_msg(
            f"\nSuccessfully generated {sample_size} realizations of "
            "deviation from the median consequences.",
            prepend_timestamp=False,
        )

        res_list = []
        key_list = []

        dmg_quantities.columns = dmg_quantities.columns.reorder_levels(
            ['cmp', 'ds', 'loc', 'dir', 'uid']
        )
        dmg_quantities.sort_index(axis=1, inplace=True)

        if std_sample is not None:
            std_dvs = std_sample.columns.unique(level=0)
        else:
            std_dvs = []

        for decision_variable in self.decision_variables:
            if decision_variable in std_dvs:
                prob_cmp_list = std_sample[decision_variable].columns.unique(level=0)
            else:
                prob_cmp_list = []

            cmp_list = []

            if decision_variable not in medians:
                continue
            for component in medians[decision_variable].columns.unique(level=0):
                # check if there is damage in the component
                consequence = self._loss_map.at[component, 'Repair']

                if not (component in dmg_quantities.columns.get_level_values('cmp')):
                    continue

                ds_list = []

                for ds in (
                    medians[decision_variable]
                    .loc[:, component]
                    .columns.unique(level=0)
                ):
                    loc_list = []

                    for loc_id, loc in enumerate(
                        dmg_quantities.loc[:, (component, ds)].columns.unique(
                            level=0
                        )
                    ):
                        if (
                            self._asmnt.options.eco_scale["AcrossFloors"] is True
                        ) and (loc_id > 0):
                            break

                        if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                            median_i = medians[decision_variable].loc[
                                :, (component, ds)
                            ]
                            dmg_i = dmg_quantities.loc[:, (component, ds)]

                            if component in prob_cmp_list:
                                std_i = std_sample.loc[
                                    :, (decision_variable, component, ds)
                                ]
                            else:
                                std_i = None

                        else:
                            median_i = medians[decision_variable].loc[
                                :, (component, ds, loc)
                            ]
                            dmg_i = dmg_quantities.loc[:, (component, ds, loc)]

                            if component in prob_cmp_list:
                                std_i = std_sample.loc[
                                    :, (decision_variable, component, ds, loc)
                                ]
                            else:
                                std_i = None

                        if std_i is not None:
                            res_list.append(dmg_i.mul(median_i, axis=0) * std_i)
                        else:
                            res_list.append(dmg_i.mul(median_i, axis=0))

                        loc_list.append(loc)

                    if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                        ds_list += [
                            ds,
                        ]
                    else:
                        ds_list += [(ds, loc) for loc in loc_list]

                if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                    cmp_list += [(consequence, component, ds) for ds in ds_list]
                else:
                    cmp_list += [
                        (consequence, component, ds, loc) for ds, loc in ds_list
                    ]

            if self._asmnt.options.eco_scale["AcrossFloors"] is True:
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
        DV_sample = pd.concat(res_list, axis=1, keys=key_list, names=lvl_names)

        DV_sample = DV_sample.fillna(0).convert_dtypes()

        # Get the flags for replacement consequence trigger
        DV_sum = DV_sample.groupby(level=[1], axis=1).sum()
        if 'replacement' in DV_sum.columns:
            # When the 'replacement' consequence is triggered, all
            # local repair consequences are discarded. Note that
            # global consequences are assigned to location '0'.

            id_replacement = DV_sum['replacement'] > 0

            # get the list of non-zero locations
            locs = DV_sample.columns.get_level_values('loc').unique().values
            locs = locs[locs != '0']

            DV_sample.loc[id_replacement, idx[:, :, :, :, locs]] = 0.0

        self.sample = DV_sample

        self.log_msg("Successfully obtained DV sample.", prepend_timestamp=False)

    def _aggregate_losses(self):

        # group results by DV type and location
        aggregated = self.sample.groupby(level=[0, 4], axis=1).sum()

        # Note: The `Time` DV receives special treatment.

        # create the summary DF
        columns = [
            f'repair_{x.lower()}' for x in self.decision_variables if x != 'Time'
        ]
        if 'Time' in self.decision_variables:
            columns.extend(('repair_time-sequential', 'repair_time-parallel'))
        df_agg = pd.DataFrame(index=self.sample.index, columns=columns)

        for decision_variable in self.decision_variables:

            # Time ..
            if decision_variable == 'Time':
                if 'Time' in aggregated.columns:
                    df_agg['repair_time-sequential'] = aggregated['Time'].sum(axis=1)

                    df_agg['repair_time-parallel'] = aggregated['Time'].max(axis=1)
                else:
                    df_agg = df_agg.drop(
                        ['repair_time-parallel', 'repair_time-sequential'], axis=1
                    )
            # All other ..
            else:
                if decision_variable in aggregated.columns:
                    df_agg[f'repair_{decision_variable.lower()}'] = aggregated[
                        decision_variable
                    ].sum(axis=1)
                else:
                    df_agg = df_agg.drop(
                        f'repair_{decision_variable.lower()}', axis=1
                    )

        # TODO: implement conversion of loss values to any desired
        # unit other than the base loss units

        df_agg = base.convert_to_MultiIndex(df_agg, axis=1)
        df_agg.sort_index(axis=1, inplace=True)

        return df_agg

    def _convert_loss_parameter_units(self):
        """
        Converts previously loaded loss parameters to base units.

        """
        units = self.loss_params[('DV', 'Unit')]
        arg_units = self.loss_params[('Quantity', 'Unit')]
        for column in self.loss_params.columns.unique(level=0):
            if not column.startswith('DS'):
                continue
            self.loss_params.loc[:, column] = self._convert_marginal_params(
                self.loss_params.loc[:, column].copy(), units, arg_units
            ).values

    def _drop_unused_damage_states(self):
        """
        Removes columns from the loss model parameters corresponding
        to unused damage states.

        """

        first_level = self.loss_params.columns.get_level_values(0).unique().to_list()
        ds_list = [x for x in first_level if x.startswith('DS')]
        ds_to_drop = []
        for damage_state in ds_list:
            if (
                np.all(pd.isna(self.loss_params.loc[:, idx[damage_state, :]].values))
                # Note: When this evaluates to True, when you add `is
                # True` on the right it suddenly evaluates to
                # False. We need to figure out why this is happening,
                # but the way it's written now does what we want in
                # each case.
            ):
                ds_to_drop.append(damage_state)

        self.loss_params.drop(columns=ds_to_drop, level=0, inplace=True)

    def _create_DV_RVs(self, case_list):
        """
        Prepare the random variables associated with decision
        variables, such as repair cost and time.

        Parameters
        ----------
        case_list: MultiIndex
            Index with cmp-loc-dir-ds descriptions that identify the
            RVs we need for the simulation.

        Returns
        -------
        RandomVariableRegistry or None
            A RandomVariableRegistry containing all the generated
            random variables necessary for the simulation. If no
            random variables are generated (due to missing parameters
            or conditions), returns None.

        Raises
        ------
        ValueError
            If an unrecognized loss driver type is encountered,
            indicating a configuration or data input error.

        """

        # Convert the MultiIndex to a DataFrame
        case_df = pd.DataFrame(index=case_list).reset_index()
        # maps `cmp` to array of damage states
        damage_states = case_df.groupby('cmp')['ds'].unique().to_dict()
        # maps `cmp`-`ds` to tuple of `loc`-`dir`-`uid` tuples
        loc_dir_uids = (
            case_df.groupby(['cmp', 'ds'])
            .apply(lambda x: tuple(zip(x['loc'], x['dir'], x['uid'])))
            .to_dict()
        )

        RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        rv_count = 0

        # for each component in the loss map
        for component, consequence in self._loss_map['Repair'].items():

            # for each DV
            for decision_variable in self.decision_variables:

                # If loss parameters are missing for that consequence,
                # don't estimate losses for it. A warning has already
                # been issued for what is missing.
                if (consequence, decision_variable) in self._missing:
                    continue

                # If loss parameters are missing for that consequence,
                # for this particular loss model, they will exist in
                # the other(s).
                if (consequence, decision_variable) not in self.loss_params.index:
                    continue

                # load the corresponding parameters
                parameters = self.loss_params.loc[
                    (consequence, decision_variable), :
                ]

                for ds in damage_states[component]:
                    if ds == '0':
                        continue

                    ds_family = parameters.at[(f'DS{ds}', 'Family')]
                    ds_theta = [
                        parameters.get((f'DS{ds}', f'Theta_{t_i}'), np.nan)
                        for t_i in range(3)
                    ]

                    # If there is no RV family we don't need an RV
                    if pd.isna(ds_family):
                        continue

                    # If the first parameter is controlled by a function, we use
                    # 1.0 in its place and will scale the results in a later
                    # step
                    if isinstance(ds_theta[0], str) and '|' in ds_theta[0]:
                        ds_theta[0] = 1.0

                    loc_dir_uid = loc_dir_uids[(component, ds)]

                    for loc, direction, uid in loc_dir_uid:
                        # assign RVs
                        RV_reg.add_RV(
                            uq.rv_class_map(ds_family)(
                                name=(
                                    f'{decision_variable}-{component}-'
                                    f'{ds}-{loc}-{direction}-{uid}'
                                ),
                                theta=ds_theta,
                                truncation_limits=[0.0, np.nan],
                            )
                        )
                        rv_count += 1

        # assign Time-Cost correlation whenever applicable
        for component, consequence in self._loss_map['Repair'].items():
            for decision_variable in self.decision_variables:
                if (consequence, decision_variable) in self._missing:
                    continue
                if (consequence, decision_variable) not in self.loss_params.index:
                    continue
                for ds in damage_states[component]:
                    if ds == '0':
                        continue
                    loc_dir_uid = loc_dir_uids[(component, ds)]
                    for loc, direction, uid in loc_dir_uid:
                        cost_rv_tag = (
                            f'Cost-{consequence}-{ds}-{loc}-{direction}-{uid}'
                        )
                        time_rv_tag = (
                            f'Time-{consequence}-{ds}-{loc}-{direction}-{uid}'
                        )
                        set_flag = f'DV-{component}-{ds}-{loc}-{direction}-{uid}_set'
                        if (
                            cost_rv_tag in RV_reg.RV
                            and time_rv_tag in RV_reg.RV
                            and (self._asmnt.options.rho_cost_time != 0.0)
                        ):
                            rho = self._asmnt.options.rho_cost_time
                            RV_reg.add_RV_set(
                                uq.RandomVariableSet(
                                    set_flag,
                                    list(
                                        RV_reg.RVs(
                                            [cost_rv_tag, time_rv_tag]
                                        ).values()
                                    ),
                                    np.array([[1.0, rho], [rho, 1.0]]),
                                )
                            )

        self.log_msg(
            f"\n{rv_count} random variables created.", prepend_timestamp=False
        )

        if rv_count > 0:
            return RV_reg
        return None

    def _calc_median_consequence(self, eco_qnt):
        """
        Calculates the median repair consequences for each loss
        component based on their quantities and the associated loss
        parameters.

        Parameters
        ----------
        eco_qnt: DataFrame
            A DataFrame containing quantity realizations for various
            components and damage states, appropriately grouped to
            account for economies of scale.

        decision_variables: tuple
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

            for loss_cmp_id, loss_cmp_name in self._loss_map['Repair'].items():

                if (loss_cmp_name, decision_variable) in self._missing:
                    continue

                if loss_cmp_id not in eco_qnt.columns.get_level_values(0).unique():
                    continue

                ds_list = []
                sub_medians = []

                for ds in self.loss_params.columns.get_level_values(0).unique():
                    if not ds.startswith('DS'):
                        continue

                    ds_id = ds[2:]

                    if ds_id == '0':
                        continue

                    loss_params_DS = self.loss_params.loc[
                        (loss_cmp_name, decision_variable), ds
                    ]

                    # check if theta_0 is defined
                    theta_0 = loss_params_DS.get('Theta_0', np.nan)

                    if pd.isna(theta_0):
                        continue

                    # check if the distribution type is supported
                    family = loss_params_DS.get('Family', np.nan)

                    if (not pd.isna(family)) and (
                        family not in ['normal', 'lognormal', 'deterministic']
                    ):
                        raise ValueError(
                            f"Loss Distribution of type {family} " f"not supported."
                        )

                    # If theta_0 is a scalar
                    try:
                        theta_0 = float(theta_0)

                        if pd.isna(loss_params_DS.get('Family', np.nan)):
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
                        avail_ds = eco_qnt.loc[:, loss_cmp_id].columns.unique(
                            level=0
                        )

                        if ds_id not in avail_ds:
                            continue

                        eco_qnt_i = eco_qnt.loc[:, (loss_cmp_id, ds_id)].copy()

                    else:
                        eco_qnt_i = eco_qnt.loc[:, loss_cmp_id].copy()

                    if isinstance(eco_qnt_i, pd.Series):
                        eco_qnt_i = eco_qnt_i.to_frame()
                        eco_qnt_i.columns = ['X']
                        eco_qnt_i.columns.name = 'del'

                    # generate the median values for each realization
                    eco_qnt_i.loc[:, :] = f_median(eco_qnt_i.values)

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
                if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                    result.columns.names = ['cmp', 'ds']

                else:
                    result.columns.names = ['cmp', 'ds', 'loc']

                # save the results to the returned dictionary
                medians.update({decision_variable: result})

        return medians


class RepairModel_LF(RepairModel_Base):
    """
    Manages repair consequences driven by components that are modeled
    with Loss Functions (LF)

    """

    def _calculate(self, sample_size):
        pass

    def _convert_loss_parameter_units(self):
        """
        Converts previously loaded loss parameters to base units.

        """
        units = self.loss_params[('DV', 'Unit')]
        arg_units = self.loss_params[('Demand', 'Unit')]
        column = 'LossFunction'
        self.loss_params.loc[:, column] = self._convert_marginal_params(
            self.loss_params[column].copy(), units, arg_units, divide_units=False
        ).values

def _prep_constant_median_DV(median):
    """
    Returns a constant median Decision Variable (DV) function.

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

    def f(*args):
        # pylint: disable=unused-argument
        # pylint: disable=missing-return-doc
        # pylint: disable=missing-return-type-doc
        return median

    return f


def _prep_bounded_multilinear_median_DV(medians, quantities):
    """
    Returns a bounded multilinear median Decision Variable (DV) function.

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

    def f(quantity):
        # pylint: disable=missing-return-doc
        # pylint: disable=missing-return-type-doc
        if quantity is None:
            raise ValueError(
                'A bounded linear median Decision Variable function called '
                'without specifying the quantity of damaged components'
            )

        q_array = np.asarray(quantity, dtype=np.float64)

        # calculate the median consequence given the quantity of damaged
        # components
        output = np.interp(q_array, quantities, medians)

        return output

    return f


def _is_for_lf_model(data):
    """
    Determines if the specified loss model parameters are for
    components modeled with Loss Functions (LF).
    """
    return 'LossFunction' in data.columns.get_level_values(0)


def _is_for_ds_model(data):
    """
    Determines if the specified loss model parameters are for
    components modeled with discrete Damage States (DS).
    """
    return 'DS1' in data.columns.get_level_values(0)
