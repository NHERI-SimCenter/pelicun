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

    prep_constant_median_DV
    prep_bounded_multilinear_median_DV

    LossModel
    RepairModel

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
    Parent object for loss models.

    All loss assessment methods should be children of this class.

    Parameters
    ----------

    """

    def __init__(self, assessment):
        super().__init__(assessment)


class RepairModel(LossModel):
    """
    Manages building repair consequence assessments.

    Parameters
    ----------

    """

    def __init__(self, assessment):
        super().__init__(assessment)

        self.sample = None
        self.loss_map = None
        self.loss_params = None
        self.loss_type = 'Repair'

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
        Load damage sample data.

        """
        self.log_div()
        self.log_msg('Loading loss sample...')

        self.sample = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors, log=self._asmnt.log
        )

        self.log_msg('Loss sample successfully loaded.', prepend_timestamp=False)

    def load_model(self, data_paths, mapping_path, decision_variables=None):
        """
        Load the list of prescribed consequence models and their parameters

        Parameters
        ----------
        data_paths: list of string or DataFrame
            List of paths to data files with consequence model
            parameters.  Default XY datasets can be accessed as
            PelicunDefault/XY.  The list can also contain DataFrame
            objects, in which case that data is used directly.
        mapping_path: string
            Path to a csv file that maps drivers (i.e., damage or edp data) to
            loss models.
        decision_variables: list of string, optional
            List of decision variables to include in the analysis. If None,
            all variables provided in the consequence models are included. When
            a list is provided, only variables in the list will be included.
        """

        self.log_div()
        self.log_msg(f'Loading loss map for {self.loss_type}...')

        loss_map = file_io.load_data(
            mapping_path, None, orientation=1, reindex=False, log=self._asmnt.log
        )

        loss_map['Driver'] = loss_map.index.values
        loss_map['Consequence'] = loss_map[self.loss_type]
        loss_map.index = np.arange(loss_map.shape[0])
        loss_map = loss_map.loc[:, ['Driver', 'Consequence']]
        loss_map.dropna(inplace=True)

        self.loss_map = loss_map

        self.log_msg("Loss map successfully parsed.", prepend_timestamp=False)

        self.log_div()
        self.log_msg(f'Loading loss parameters for {self.loss_type}...')

        # replace default flag with default data path
        data_paths = file_io.substitute_default_path(data_paths)

        data_list = []
        # load the data files one by one
        for data_path in data_paths:
            data = file_io.load_data(
                data_path, None, orientation=1, reindex=False, log=self._asmnt.log
            )

            data_list.append(data)

        loss_params = pd.concat(data_list, axis=0)

        # drop redefinitions of components
        loss_params = (
            loss_params.groupby(level=[0, 1])
            .first()
            .transform(lambda x: x.fillna(np.nan))
        )
        # note: .groupby introduces None entries. We replace them with
        # NaN for consistency.

        # keep only the relevant data
        loss_cmp = np.unique(self.loss_map['Consequence'].values)

        available_cmp = loss_params.index.unique(level=0)
        missing_cmp = []
        for cmp in loss_cmp:
            if cmp not in available_cmp:
                missing_cmp.append(cmp)

        if len(missing_cmp) > 0:
            self.log_msg(
                "\nWARNING: The loss model does not provide "
                "consequence information for the following component(s) "
                f"in the loss map: {missing_cmp}. They are removed from "
                "further analysis\n",
                prepend_timestamp=False,
            )

        self.loss_map = self.loss_map.loc[~loss_map['Consequence'].isin(missing_cmp)]
        loss_cmp = np.unique(self.loss_map['Consequence'].values)

        loss_params = loss_params.loc[idx[loss_cmp, :], :]

        # drop unused damage states
        DS_list = loss_params.columns.get_level_values(0).unique()
        DS_to_drop = []
        for DS in DS_list:
            if np.all(pd.isna(loss_params.loc[:, idx[DS, :]].values)) is True:
                DS_to_drop.append(DS)

        loss_params.drop(columns=DS_to_drop, level=0, inplace=True)

        # convert values to internal base units
        for DS in loss_params.columns.unique(level=0):
            if DS.startswith('DS'):
                loss_params.loc[:, DS] = self.convert_marginal_params(
                    loss_params.loc[:, DS].copy(),
                    loss_params[('DV', 'Unit')],
                    loss_params[('Quantity', 'Unit')],
                ).values

        # check for components with incomplete loss information
        cmp_incomplete_list = loss_params.loc[
            loss_params[('Incomplete', '')] == 1
        ].index

        if len(cmp_incomplete_list) > 0:
            loss_params.drop(cmp_incomplete_list, inplace=True)

            self.log_msg(
                "\n"
                "WARNING: Loss information is incomplete for the "
                f"following component(s) {cmp_incomplete_list}. "
                "They were removed from the analysis."
                "\n",
                prepend_timestamp=False,
            )

        # filter decision variables, if needed
        if decision_variables is not None:
            loss_params = loss_params.reorder_levels([1, 0])

            available_DVs = loss_params.index.unique(level=0)
            filtered_DVs = []

            for DV_i in decision_variables:
                if DV_i in available_DVs:
                    filtered_DVs.append(DV_i)

            loss_params = loss_params.loc[filtered_DVs, :].reorder_levels([1, 0])

        self.loss_params = loss_params.sort_index(axis=1)

        self.log_msg("Loss parameters successfully parsed.", prepend_timestamp=False)

    def calculate(self, sample_size=None):
        """
        Calculate the consequences of each component block damage in
        the asset.

        """
        if not sample_size:
            sample_size = self._asmnt.demand.sample.shape[0]
            warnings.warn(
                'Using default sample size is deprecated and will '
                'be removed in future versions. '
                'Please provide the `sample_size` explicitly.',
                DeprecationWarning,
            )

        self.log_div()
        self.log_msg("Calculating losses...")

        # First, get the damaged quantities in each damage state for
        # each component of interest.
        dmg_q = self._asmnt.damage.ds_model.sample.copy()

        # Now sample random Decision Variables
        # Note that this method is DV-specific and needs to be
        # implemented in every child of the LossModel independently.
        self._generate_DV_sample(dmg_q, sample_size)

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

        DV = self.sample

        # group results by DV type and location
        DVG = DV.groupby(level=[0, 4], axis=1).sum()

        # create the summary DF
        df_agg = pd.DataFrame(
            index=DV.index,
            columns=[
                'repair_cost',
                'repair_time-parallel',
                'repair_time-sequential',
                'repair_carbon',
                'repair_energy',
            ],
        )

        if 'Cost' in DVG.columns:
            df_agg['repair_cost'] = DVG['Cost'].sum(axis=1)
        else:
            df_agg = df_agg.drop('repair_cost', axis=1)

        if 'Time' in DVG.columns:
            df_agg['repair_time-sequential'] = DVG['Time'].sum(axis=1)

            df_agg['repair_time-parallel'] = DVG['Time'].max(axis=1)
        else:
            df_agg = df_agg.drop(
                ['repair_time-parallel', 'repair_time-sequential'], axis=1
            )

        if 'Carbon' in DVG.columns:
            df_agg['repair_carbon'] = DVG['Carbon'].sum(axis=1)
        else:
            df_agg = df_agg.drop('repair_carbon', axis=1)

        if 'Energy' in DVG.columns:
            df_agg['repair_energy'] = DVG['Energy'].sum(axis=1)
        else:
            df_agg = df_agg.drop('repair_energy', axis=1)

        # convert units

        cmp_units = (
            self.loss_params[('DV', 'Unit')]
            .groupby(
                level=[
                    1,
                ]
            )
            .agg(lambda x: x.value_counts().index[0])
        )

        dv_units = pd.Series(index=df_agg.columns, name='Units', dtype='object')

        if 'Cost' in DVG.columns:
            dv_units['repair_cost'] = cmp_units['Cost']

        if 'Time' in DVG.columns:
            dv_units['repair_time-parallel'] = cmp_units['Time']
            dv_units['repair_time-sequential'] = cmp_units['Time']

        if 'Carbon' in DVG.columns:
            dv_units['repair_carbon'] = cmp_units['Carbon']

        if 'Energy' in DVG.columns:
            dv_units['repair_energy'] = cmp_units['Energy']

        df_agg = file_io.save_to_csv(
            df_agg,
            None,
            units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=False,
            log=self._asmnt.log,
        )

        df_agg.drop("Units", inplace=True)

        # convert header

        df_agg = base.convert_to_MultiIndex(df_agg, axis=1)

        self.log_msg("Repair consequences successfully aggregated.")

        return df_agg.astype(float)

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

        RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)
        LP = self.loss_params

        # make ds the second level in the MultiIndex
        case_DF = pd.DataFrame(
            index=case_list.reorder_levels([0, 4, 1, 2, 3]),
            columns=[
                0,
            ],
        )
        case_DF.sort_index(axis=0, inplace=True)
        driver_cmps = case_list.get_level_values(0).unique()

        rv_count = 0

        # for each loss component
        for loss_cmp_id in self.loss_map.index.values:
            # load the corresponding parameters
            driver_type, driver_cmp_id = self.loss_map.loc[loss_cmp_id, 'Driver']
            conseq_cmp_id = self.loss_map.loc[loss_cmp_id, 'Consequence']

            # currently, we only support DMG-based loss calculations
            # but this will be extended in the very near future
            if driver_type != 'DMG':
                raise ValueError(
                    f"Loss Driver type not recognized: " f"{driver_type}"
                )

            # load the parameters
            # TODO: remove specific DV_type references and make the code below
            # generate parameters for any DV_types provided
            if (conseq_cmp_id, 'Cost') in LP.index:
                cost_params = LP.loc[(conseq_cmp_id, 'Cost'), :]
            else:
                cost_params = None

            if (conseq_cmp_id, 'Time') in LP.index:
                time_params = LP.loc[(conseq_cmp_id, 'Time'), :]
            else:
                time_params = None

            if (conseq_cmp_id, 'Carbon') in LP.index:
                carbon_params = LP.loc[(conseq_cmp_id, 'Carbon'), :]
            else:
                carbon_params = None

            if (conseq_cmp_id, 'Energy') in LP.index:
                energy_params = LP.loc[(conseq_cmp_id, 'Energy'), :]
            else:
                energy_params = None

            if driver_cmp_id not in driver_cmps:
                continue

            for ds in case_DF.loc[driver_cmp_id, :].index.unique(level=0):
                if ds == '0':
                    continue

                if cost_params is not None:
                    cost_params_DS = cost_params[f'DS{ds}']

                    cost_family = cost_params_DS.get('Family', np.nan)
                    cost_theta = [
                        cost_params_DS.get(f"Theta_{t_i}", np.nan)
                        for t_i in range(3)
                    ]

                    # If the first parameter is controlled by a function, we use
                    # 1.0 in its place and will scale the results in a later
                    # step
                    if '|' in str(cost_theta[0]):
                        # if isinstance(cost_theta[0], str):
                        cost_theta[0] = 1.0

                else:
                    cost_family = np.nan

                if time_params is not None:
                    time_params_DS = time_params[f'DS{ds}']

                    time_family = time_params_DS.get('Family', np.nan)
                    time_theta = [
                        time_params_DS.get(f"Theta_{t_i}", np.nan)
                        for t_i in range(3)
                    ]

                    # If the first parameter is controlled by a function, we use
                    # 1.0 in its place and will scale the results in a later
                    # step
                    if '|' in str(time_theta[0]):
                        # if isinstance(time_theta[0], str):
                        time_theta[0] = 1.0

                else:
                    time_family = np.nan

                if carbon_params is not None:
                    carbon_params_DS = carbon_params[f'DS{ds}']

                    carbon_family = carbon_params_DS.get('Family', np.nan)
                    carbon_theta = [
                        carbon_params_DS.get(f"Theta_{t_i}", np.nan)
                        for t_i in range(3)
                    ]

                    # If the first parameter is controlled by a function, we use
                    # 1.0 in its place and will scale the results in a later
                    # step
                    if '|' in str(carbon_theta[0]):
                        # if isinstance(carbon_theta[0], str):
                        carbon_theta[0] = 1.0

                else:
                    carbon_family = np.nan

                if energy_params is not None:
                    energy_params_DS = energy_params[f'DS{ds}']

                    energy_family = energy_params_DS.get('Family', np.nan)
                    energy_theta = [
                        energy_params_DS.get(f"Theta_{t_i}", np.nan)
                        for t_i in range(3)
                    ]

                    # If the first parameter is controlled by a function, we use
                    # 1.0 in its place and will scale the results in a later
                    # step
                    if '|' in str(energy_theta[0]):
                        # if isinstance(energy_theta[0], str):
                        energy_theta[0] = 1.0

                else:
                    energy_family = np.nan

                # If neither of the DV_types has a stochastic model assigned,
                # we do not need random variables for this DS
                if (
                    (pd.isna(cost_family))
                    and (pd.isna(time_family))
                    and (pd.isna(carbon_family))
                    and (pd.isna(energy_family))
                ):
                    continue

                # Otherwise, load the loc-dir cases
                loc_dir_uid = case_DF.loc[(driver_cmp_id, ds)].index.values

                for loc, direction, uid in loc_dir_uid:
                    # assign cost RV
                    if pd.isna(cost_family) is False:
                        cost_rv_tag = (
                            f'Cost-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(
                            uq.rv_class_map(cost_family)(
                                name=cost_rv_tag,
                                theta=cost_theta,
                                truncation_limits=[0.0, np.nan],
                            )
                        )
                        rv_count += 1

                    # assign time RV
                    if pd.isna(time_family) is False:
                        time_rv_tag = (
                            f'Time-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(
                            uq.rv_class_map(time_family)(
                                name=time_rv_tag,
                                theta=time_theta,
                                truncation_limits=[0.0, np.nan],
                            )
                        )
                        rv_count += 1

                    # assign time RV
                    if pd.isna(carbon_family) is False:
                        carbon_rv_tag = (
                            f'Carbon-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(
                            uq.rv_class_map(carbon_family)(
                                name=carbon_rv_tag,
                                theta=carbon_theta,
                                truncation_limits=[0.0, np.nan],
                            )
                        )
                        rv_count += 1

                    # assign time RV
                    if pd.isna(energy_family) is False:
                        energy_rv_tag = (
                            f'Energy-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(
                            uq.rv_class_map(energy_family)(
                                name=energy_rv_tag,
                                theta=energy_theta,
                                truncation_limits=[0.0, np.nan],
                            )
                        )
                        rv_count += 1

                    # assign correlation between RVs across DV_types
                    # TODO: add more DV_types and handle cases with only a
                    # subset of them being defined
                    if (
                        (pd.isna(cost_family) is False)
                        and (pd.isna(time_family) is False)
                        and (self._asmnt.options.rho_cost_time != 0.0)
                    ):
                        rho = self._asmnt.options.rho_cost_time

                        RV_reg.add_RV_set(
                            uq.RandomVariableSet(
                                f'DV-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}_set',
                                list(
                                    RV_reg.RVs([cost_rv_tag, time_rv_tag]).values()
                                ),
                                np.array([[1.0, rho], [rho, 1.0]]),
                            )
                        )

        self.log_msg(
            f"\n{rv_count} random variables created.", prepend_timestamp=False
        )

        if rv_count > 0:
            return RV_reg
        # else:
        return None

    def _calc_median_consequence(self, eco_qnt):
        """
        Calculates the median repair consequences for each loss
        component based on their quantities and the associated loss
        parameters.

        This function evaluates the median consequences for different
        types of decision variables (DV), such as repair costs or
        repair time, based on the provided loss parameters. It
        utilizes the eco_qnt DataFrame, which contains economic
        quantity realizations for various damage states and
        components, to compute the consequences.

        Parameters
        ----------
        eco_qnt : DataFrame
            A DataFrame containing economic quantity realizations for
            various components and damage states, indexed or
            structured to align with the loss parameters.

        Returns
        -------
        dict
            A dictionary where keys are the types of decision variables
            (DV) like 'COST' or 'TIME', and values are DataFrames
            containing the median consequences for each component and
            damage state. These DataFrames are structured with
            MultiIndex columns that may include 'cmp' (component),
            'ds' (damage state), and potentially 'loc' (location),
            depending on assessment options.

        Raises
        ------
        ValueError
            If any loss driver types or distribution types are not
            recognized, or if the parameters are incomplete or
            unsupported.
        """

        medians = {}

        DV_types = self.loss_params.index.unique(level=1)

        # for DV_type, DV_type_scase in zip(['COST', 'TIME'], ['Cost', 'Time']):
        for DV_type in DV_types:
            cmp_list = []
            median_list = []

            for loss_cmp_id in self.loss_map.index:
                driver_type, driver_cmp = self.loss_map.loc[loss_cmp_id, 'Driver']
                loss_cmp_name = self.loss_map.loc[loss_cmp_id, 'Consequence']

                # check if the given DV type is available as an output for the
                # selected component
                if (loss_cmp_name, DV_type) not in self.loss_params.index:
                    continue

                if driver_type != 'DMG':
                    raise ValueError(
                        f"Loss Driver type not recognized: " f"{driver_type}"
                    )

                if driver_cmp not in eco_qnt.columns.get_level_values(0).unique():
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
                        (loss_cmp_name, DV_type), ds
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
                            f_median = prep_constant_median_DV(theta_0)

                        else:
                            # otherwise use a constant 1.0 as the median
                            # The random variable will be generated as a
                            # variation from this 1.0 and added in a later step.
                            f_median = prep_constant_median_DV(1.0)

                    except ValueError:
                        # otherwise, use the multilinear function
                        all_vals = np.array(
                            [val.split(',') for val in theta_0.split('|')],
                            dtype=float,
                        )
                        medns = all_vals[0]
                        qnts = all_vals[1]
                        f_median = prep_bounded_multilinear_median_DV(medns, qnts)

                    # get the corresponding aggregate damage quantities
                    # to consider economies of scale
                    if 'ds' in eco_qnt.columns.names:
                        avail_ds = eco_qnt.loc[:, driver_cmp].columns.unique(level=0)

                        if ds_id not in avail_ds:
                            continue

                        eco_qnt_i = eco_qnt.loc[:, (driver_cmp, ds_id)].copy()

                    else:
                        eco_qnt_i = eco_qnt.loc[:, driver_cmp].copy()

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
                medians.update({DV_type: result})

        return medians

    def _generate_DV_sample(self, dmg_quantities, sample_size):
        """
        Generate a sample of repair costs and times.

        Parameters
        ----------
        dmg_quantities: DataFrame
            A table with the quantity of damage experienced in each damage state
            of each performance group at each location and direction. You can use
            the prepare_dmg_quantities method in the DamageModel to get such a
            DF.
        sample_size: integer
            The number of realizations to generate.

        Raises
        ------
        ValueError
            When any Loss Driver is not recognized.
        """

        # calculate the quantities for economies of scale
        self.log_msg("\nAggregating damage quantities...", prepend_timestamp=False)

        # If everything is undamaged there are no losses
        if set(dmg_quantities.columns.get_level_values('ds')) == {'0'}:
            self.sample = None
            self.log_msg(
                "There is no damage---DV sample is set to None.",
                prepend_timestamp=False,
            )
            return

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
            ).sort_index(axis=1)
            std_sample.columns.names = ['dv', 'cmp', 'ds', 'loc', 'dir', 'uid']

            # convert column names to int
            std_idx = std_sample.columns.levels

            std_sample.columns = std_sample.columns.set_levels(
                [
                    std_idx[0],
                    std_idx[1].astype(int),
                    std_idx[2],
                    std_idx[3],
                    std_idx[4],
                    std_idx[5],
                ]
            )

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
            [0, 4, 1, 2, 3]
        )
        dmg_quantities.sort_index(axis=1, inplace=True)

        DV_types = self.loss_params.index.unique(level=1)

        if isinstance(std_sample, pd.DataFrame):
            std_DV_types = std_sample.columns.unique(level=0)
        else:
            std_DV_types = []

        # for DV_type, _ in zip(['COST', 'TIME'], ['Cost', 'Time']):
        for DV_type in DV_types:
            if DV_type in std_DV_types:
                prob_cmp_list = std_sample[DV_type].columns.unique(level=0)
            else:
                prob_cmp_list = []

            cmp_list = []

            if DV_type not in medians:
                continue

            for cmp_i in medians[DV_type].columns.unique(level=0):
                # check if there is damage in the component
                driver_type, dmg_cmp_i = self.loss_map.loc[cmp_i, 'Driver']
                loss_cmp_i = self.loss_map.loc[cmp_i, 'Consequence']

                if driver_type != 'DMG':
                    raise ValueError(
                        f"Loss Driver type not " f"recognized: {driver_type}"
                    )

                if not (dmg_cmp_i in dmg_quantities.columns.unique(level=0)):
                    continue

                ds_list = []

                for ds in medians[DV_type].loc[:, cmp_i].columns.unique(level=0):
                    loc_list = []

                    for loc_id, loc in enumerate(
                        dmg_quantities.loc[:, (dmg_cmp_i, ds)].columns.unique(
                            level=0
                        )
                    ):
                        if (
                            self._asmnt.options.eco_scale["AcrossFloors"] is True
                        ) and (loc_id > 0):
                            break

                        if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                            median_i = medians[DV_type].loc[:, (cmp_i, ds)]
                            dmg_i = dmg_quantities.loc[:, (dmg_cmp_i, ds)]

                            if cmp_i in prob_cmp_list:
                                std_i = std_sample.loc[:, (DV_type, cmp_i, ds)]
                            else:
                                std_i = None

                        else:
                            median_i = medians[DV_type].loc[:, (cmp_i, ds, loc)]
                            dmg_i = dmg_quantities.loc[:, (dmg_cmp_i, ds, loc)]

                            if cmp_i in prob_cmp_list:
                                std_i = std_sample.loc[:, (DV_type, cmp_i, ds, loc)]
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
                    cmp_list += [(loss_cmp_i, dmg_cmp_i, ds) for ds in ds_list]
                else:
                    cmp_list += [
                        (loss_cmp_i, dmg_cmp_i, ds, loc) for ds, loc in ds_list
                    ]

            if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                key_list += [
                    (DV_type, loss_cmp_i, dmg_cmp_i, ds)
                    for loss_cmp_i, dmg_cmp_i, ds in cmp_list
                ]
            else:
                key_list += [
                    (DV_type, loss_cmp_i, dmg_cmp_i, ds, loc)
                    for loss_cmp_i, dmg_cmp_i, ds, loc in cmp_list
                ]

        lvl_names = ['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'uid']
        DV_sample = pd.concat(res_list, axis=1, keys=key_list, names=lvl_names)

        DV_sample = DV_sample.fillna(0).convert_dtypes()
        DV_sample.columns.names = lvl_names

        # Get the flags for replacement consequence trigger
        DV_sum = DV_sample.groupby(
            level=[
                1,
            ],
            axis=1,
        ).sum()
        if 'replacement' in DV_sum.columns:
            # When the 'replacement' consequence is triggered, all
            # local repair consequences are discarded. Note that
            # global consequences are assigned to location '0'.

            id_replacement = DV_sum['replacement'] > 0

            # get the list of non-zero locations
            locs = DV_sample.columns.get_level_values(4).unique().values

            locs = locs[locs != '0']

            DV_sample.loc[id_replacement, idx[:, :, :, :, locs]] = 0.0

        self.sample = DV_sample

        self.log_msg("Successfully obtained DV sample.", prepend_timestamp=False)


def prep_constant_median_DV(median):
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


def prep_bounded_multilinear_median_DV(medians, quantities):
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
