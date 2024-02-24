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
This module has classes and methods that define and access the model used for
loss assessment.

.. rubric:: Contents

.. autosummary::

    prep_constant_median_DV
    prep_bounded_multilinear_median_DV

    DemandModel
    AssetModel
    DamageModel
    LossModel
    BldgRepairModel

"""

from itertools import product
from copy import deepcopy
import numpy as np
import pandas as pd
from . import base
from . import uq
from . import file_io


idx = base.idx


class PelicunModel:
    """
    Generic model class to manage methods shared between all models in Pelicun.

    """

    def __init__(self, assessment):

        # link the PelicunModel object to its Assessment object
        self._asmnt = assessment

        # link logging methods as attributes enabling more
        # concise syntax
        self.log_msg = self._asmnt.log.msg
        self.log_div = self._asmnt.log.div

    def convert_marginal_params(self, marginal_params, units, arg_units=None):
        """
        Converts the parameters of marginal distributions in a model to SI units.

        Parameters
        ----------
        marginal_params: DataFrame
            Each row corresponds to a marginal distribution with Theta
            parameters and TruncateLower, TruncateUpper truncation limits
            identified in separate columns.
        units: Series
            Identifies the input units of each marginal. The index shall be
            identical to the index of the marginal_params argument. The values
            are strings that correspond to the units listed in base.py.
        arg_units: Series
            Identifies the size of a reference entity for the marginal
            parameters. For example, when the parameters refer to a component
            repair cost, the reference size is the component block size the
            repair cost corresponds to. When the parameters refer to a capacity,
            demand, or component quantity, the reference size can be omitted
            and the default value will ensure that the corresponding scaling is
            skipped. This Series provides the units of the reference entities
            for each component. Use '1 EA' if you want to skip such scaling for
            select components but provide arg units for others.

        Returns
        -------
        marginal_params: DataFrame
            Same structure as the input DataFrame but with values scaled to
            represent internal Standard International units.

        """
        assert np.all(marginal_params.index == units.index)
        if arg_units is not None:
            assert np.all(
                marginal_params.index == arg_units.index)

        # preserve the columns in the input marginal_params
        original_cols = marginal_params.columns

        # add extra columns if they are not available in the marginals
        for col_name in ('Family',
                         'Theta_0', 'Theta_1', 'Theta_2',
                         'TruncateLower', 'TruncateUpper'):
            if col_name not in marginal_params.columns:

                marginal_params[col_name] = np.nan

        # get a list of unique units
        unique_units = units.unique()

        # for each unit
        for unit_name in unique_units:

            # get the scale factor for converting from the source unit
            unit_factor = self._asmnt.calc_unit_scale_factor(unit_name)

            # get the variables that use the given unit
            unit_ids = marginal_params.loc[units == unit_name].index

            # for each variable
            for row_id in unit_ids:

                # pull the parameters of the marginal distribution
                family = marginal_params.at[row_id, 'Family']

                if family == 'empirical':
                    continue

                # load the theta values
                theta = marginal_params.loc[
                    row_id, ['Theta_0', 'Theta_1', 'Theta_2']].values

                # for each theta
                args = []
                for t_i, theta_i in enumerate(theta):

                    # if theta_i evaluates to NaN, it is considered undefined
                    if pd.isna(theta_i):
                        args.append([])
                        continue

                    try:
                        # if theta is a scalar, just store it
                        theta[t_i] = float(theta_i)
                        args.append([])

                    except ValueError:

                        # otherwise, we assume it is a string using SimCenter
                        # array notation to identify coordinates of a
                        # multilinear function
                        values = [val.split(',') for val in theta_i.split('|')]

                        # the first set of values defines the ordinates that
                        # need to be passed to the distribution scaling method
                        theta[t_i] = np.array(values[0], dtype=float)

                        # the second set of values defines the abscissae that
                        # we will use after the distribution scaling
                        args.append(np.array(values[1], dtype=float))

                # load the truncation limits
                tr_limits = marginal_params.loc[
                    row_id, ['TruncateLower', 'TruncateUpper']]

                arg_unit_factor = 1.0

                # check if there is a need to scale due to argument units
                if not (arg_units is None):

                    # get the argument unit for the given marginal
                    arg_unit = arg_units.get(row_id)

                    if arg_unit != '1 EA':

                        # get the scale factor
                        arg_unit_factor = self._asmnt.calc_unit_scale_factor(
                            arg_unit
                        )

                        # scale arguments, if needed
                        for a_i, arg in enumerate(args):

                            if isinstance(arg, np.ndarray):
                                args[a_i] = arg * arg_unit_factor

                # convert the distribution parameters to SI
                theta, tr_limits = uq.scale_distribution(
                    unit_factor / arg_unit_factor, family, theta, tr_limits)

                # convert multilinear function parameters back into strings
                for a_i, arg in enumerate(args):

                    if len(arg) > 0:

                        theta[a_i] = '|'.join(
                            [','.join([f'{val:g}' for val in vals])
                             for vals in (theta[a_i], args[a_i])])

                # and update the values in the DF
                marginal_params.loc[
                    row_id, ['Theta_0', 'Theta_1', 'Theta_2']] = theta

                marginal_params.loc[
                    row_id, ['TruncateLower', 'TruncateUpper']] = tr_limits

        # remove the added columns
        marginal_params = marginal_params[original_cols]

        return marginal_params


class DemandModel(PelicunModel):
    """
    Manages demand information used in assessments.

    Parameters
    ----------
    marginal_params: DataFrame
        Available after the model has been calibrated or calibration data has
        been imported. Defines the marginal distribution of each demand
        variable.
    correlation: DataFrame
        Available after the model has been calibrated or calibration data has
        been imported. Defines the correlation between the demand variables in
        standard normal space. That is, the variables are sampled in standard
        normal space and then transformed into the space of their respective
        distributions and the correlation matrix corresponds to the space where
        they are sampled.
    empirical_data: DataFrame
        Available after the model has been calibrated or calibration data has
        been imported. It provides an empirical dataset for the demand
        variables that are modeled with an empirical distribution.
    sample: DataFrame
        Available after a sample has been generated. Demand variables are
        listed in columns and each row provides an independent realization of
        the joint demand distribution.
    units: Series
        Available after any demand data has been loaded. The index identifies
        the demand variables and the values provide the unit for each variable.

    """

    def __init__(self, assessment):

        super().__init__(assessment)

        self.marginal_params = None
        self.correlation = None
        self.empirical_data = None
        self.units = None

        self._RVs = None
        self.sample = None

    def save_sample(self, filepath=None, save_units=False):
        """
        Save demand sample to a csv file or return it in a DataFrame

        """

        self.log_div()
        if filepath is not None:
            self.log_msg('Saving demand sample...')

        res = file_io.save_to_csv(
            self.sample, filepath, units=self.units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log)

        if filepath is not None:
            self.log_msg('Demand sample successfully saved.',
                         prepend_timestamp=False)
            return None

        # else:
        units = res.loc["Units"]
        res.drop("Units", inplace=True)

        if save_units:
            return res.astype(float), units

        # else:
        return res.astype(float)

    def load_sample(self, filepath):
        """
        Load demand sample data and parse it.

        Besides parsing the sample, the method also reads and saves the units
        specified for each demand variable. If no units are specified, Standard
        Units are assumed.

        Parameters
        ----------
        filepath: string or DataFrame
            Location of the file with the demand sample.

        """

        def parse_header(raw_header):

            old_MI = raw_header

            # The first number (event_ID) in the demand labels is optional and
            # currently not used. We remove it if it was in the raw data.
            if old_MI.nlevels == 4:

                if self._asmnt.log.verbose:
                    self.log_msg('Removing event_ID from header...',
                                 prepend_timestamp=False)

                new_column_index_array = np.array(
                    [old_MI.get_level_values(i) for i in range(1, 4)])

            else:
                new_column_index_array = np.array(
                    [old_MI.get_level_values(i) for i in range(3)])

            # Remove whitespace to avoid ambiguity

            if self._asmnt.log.verbose:
                self.log_msg('Removing whitespace from header...',
                             prepend_timestamp=False)

            wspace_remove = np.vectorize(lambda name: str(name).replace(' ', ''))

            new_column_index = wspace_remove(new_column_index_array)

            # Creating new, cleaned-up header

            new_MI = pd.MultiIndex.from_arrays(
                new_column_index, names=['type', 'loc', 'dir'])

            return new_MI

        self.log_div()
        self.log_msg('Loading demand data...')

        demand_data, units = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors,
            return_units=True, log=self._asmnt.log)

        parsed_data = demand_data.copy()

        # start with cleaning up the header

        parsed_data.columns = parse_header(parsed_data.columns)

        # Remove errors, if needed
        if 'ERROR' in parsed_data.columns.get_level_values(0):

            self.log_msg('Removing errors from the raw data...',
                         prepend_timestamp=False)

            error_list = parsed_data.loc[:, idx['ERROR', :, :]].values.astype(bool)

            parsed_data = parsed_data.loc[~error_list, :].copy()
            parsed_data.drop('ERROR', level=0, axis=1, inplace=True)

            self.log_msg("\nBased on the values in the ERROR column, "
                         f"{np.sum(error_list)} demand samples were removed.\n",
                         prepend_timestamp=False)

        self.sample = parsed_data

        self.log_msg('Demand data successfully parsed.', prepend_timestamp=False)

        # parse the index for the units
        units.index = parse_header(units.index)

        self.units = units

        self.log_msg('Demand units successfully parsed.', prepend_timestamp=False)

    def estimate_RID(self, demands, params, method='FEMA P58'):
        """
        Estimate residual drift realizations based on other demands

        Parameters
        ----------
        demands: DataFrame
            Sample of demands required for the method to estimate the RID values
        params: dict
            Parameters required for the method to estimate the RID values
        method: {'FEMA P58'}, default: 'FEMA P58'
            Method to use for the estimation - currently, only one is available.
        """

        if method == 'FEMA P58':

            # method is described in FEMA P-58 Volume 1 Section 5.4 & Appendix C

            # the provided demands shall be PID values at various loc-dir pairs
            PID = demands

            # there's only one parameter needed: the yield drift
            yield_drift = params['yield_drift']

            # three subdomains of demands are identified
            small = PID < yield_drift
            medium = PID < 4 * yield_drift
            large = PID >= 4 * yield_drift

            # convert PID to RID in each subdomain
            RID = PID.copy()
            RID[large] = PID[large] - 3 * yield_drift
            RID[medium] = 0.3 * (PID[medium] - yield_drift)
            RID[small] = 0.

            # add extra uncertainty to nonzero values
            rng = self._asmnt.options.rng
            eps = rng.normal(scale=0.2, size=RID.shape)
            RID[RID > 0] = np.exp(np.log(RID[RID > 0]) + eps)

            # finally, make sure the RID values are never larger than the PIDs
            RID = pd.DataFrame(
                np.minimum(PID.values, RID.values),
                columns=pd.DataFrame(
                    1, index=['RID', ],
                    columns=PID.columns).stack(level=[0, 1]).index,
                index=PID.index)

        else:
            RID = None

        # return the generated drift realizations
        return RID

    def calibrate_model(self, config):
        """
        Calibrate a demand model to describe the raw demand data

        The raw data shall be parsed first to ensure that it follows the
        schema expected by this method. The calibration settings define the
        characteristics of the multivariate distribution that is fit to the
        raw data.

        Parameters
        ----------
        config: dict
            A dictionary, typically read from a json file, that specifies the
            distribution family, truncation and censoring limits, and other
            settings for the calibration.

        """

        def parse_settings(settings, demand_type):

            def parse_str_to_float(in_str, context_string):

                try:
                    out_float = float(in_str)

                except ValueError:

                    self.log_msg(f"WARNING: Could not parse {in_str} provided as "
                                 f"{context_string}. Using NaN instead.",
                                 prepend_timestamp=False)

                    out_float = np.nan

                return out_float

            active_d_types = (
                demand_sample.columns.get_level_values('type').unique())

            if demand_type == 'ALL':
                cols = tuple(active_d_types)

            else:
                cols_lst = []

                for d_type in active_d_types:
                    if d_type.split('_')[0] == demand_type:
                        cols_lst.append(d_type)

                cols = tuple(cols_lst)

            # load the distribution family
            cal_df.loc[idx[cols, :, :], 'Family'] = settings['DistributionFamily']

            # load limits
            for lim in ('CensorLower', 'CensorUpper',
                        'TruncateLower', 'TruncateUpper'):

                if lim in settings.keys():
                    val = parse_str_to_float(settings[lim], lim)
                    if not pd.isna(val):
                        cal_df.loc[idx[cols, :, :], lim] = val

            # scale the censor and truncation limits, if needed
            scale_factor = self._asmnt.scale_factor(settings.get('Unit', None))

            rows_to_scale = ['CensorLower', 'CensorUpper',
                             'TruncateLower', 'TruncateUpper']
            cal_df.loc[idx[cols, :, :], rows_to_scale] *= scale_factor

            # load the prescribed additional uncertainty
            if 'AddUncertainty' in settings.keys():

                sig_increase = parse_str_to_float(settings['AddUncertainty'],
                                                  'AddUncertainty')

                # scale the sig value if the target distribution family is normal
                if settings['DistributionFamily'] == 'normal':
                    sig_increase *= scale_factor

                cal_df.loc[idx[cols, :, :], 'SigIncrease'] = sig_increase

        def get_filter_mask(lower_lims, upper_lims):

            demands_of_interest = demand_sample.iloc[:, pd.notna(upper_lims)]
            limits_of_interest = upper_lims[pd.notna(upper_lims)]
            upper_mask = np.all(demands_of_interest < limits_of_interest,
                                axis=1)

            demands_of_interest = demand_sample.iloc[:, pd.notna(lower_lims)]
            limits_of_interest = lower_lims[pd.notna(lower_lims)]
            lower_mask = np.all(demands_of_interest > limits_of_interest,
                                axis=1)

            return np.all([lower_mask, upper_mask], axis=0)

        self.log_div()
        self.log_msg('Calibrating demand model...')

        demand_sample = self.sample

        # initialize a DataFrame that contains calibration information
        cal_df = pd.DataFrame(
            columns=['Family',
                     'CensorLower', 'CensorUpper',
                     'TruncateLower', 'TruncateUpper',
                     'SigIncrease', 'Theta_0', 'Theta_1'],
            index=demand_sample.columns,
            dtype=float
        )

        cal_df['Family'] = cal_df['Family'].astype(str)

        # start by assigning the default option ('ALL') to every demand column
        parse_settings(config['ALL'], 'ALL')

        # then parse the additional settings and make the necessary adjustments
        for demand_type in config.keys():
            if demand_type != 'ALL':
                parse_settings(config[demand_type], demand_type)

        if self._asmnt.log.verbose:
            self.log_msg(
                "\nCalibration settings successfully parsed:\n" + str(cal_df),
                prepend_timestamp=False)
        else:
            self.log_msg(
                "\nCalibration settings successfully parsed:\n",
                prepend_timestamp=False)

        # save the settings
        model_params = cal_df.copy()

        # Remove the samples outside of censoring limits
        # Currently, non-empirical demands are assumed to have some level of
        # correlation, hence, a censored value in any demand triggers the
        # removal of the entire sample from the population.
        upper_lims = cal_df.loc[:, 'CensorUpper'].values
        lower_lims = cal_df.loc[:, 'CensorLower'].values

        if ~np.all(pd.isna(np.array([upper_lims, lower_lims]))):

            censor_mask = get_filter_mask(lower_lims, upper_lims)
            censored_count = np.sum(~censor_mask)

            demand_sample = demand_sample.loc[censor_mask, :]

            self.log_msg("\nBased on the provided censoring limits, "
                         f"{censored_count} samples were censored.",
                         prepend_timestamp=False)
        else:
            censored_count = 0

        # Check if there is any sample outside of truncation limits
        # If yes, that suggests an error either in the samples or the
        # configuration. We handle such errors gracefully: the analysis is not
        # terminated, but we show an error in the log file.
        upper_lims = cal_df.loc[:, 'TruncateUpper'].values
        lower_lims = cal_df.loc[:, 'TruncateLower'].values

        if ~np.all(pd.isna(np.array([upper_lims, lower_lims]))):

            truncate_mask = get_filter_mask(lower_lims, upper_lims)
            truncated_count = np.sum(~truncate_mask)

            if truncated_count > 0:

                demand_sample = demand_sample.loc[truncate_mask, :]

                self.log_msg("\nBased on the provided truncation limits, "
                             f"{truncated_count} samples were removed before demand "
                             "calibration.",
                             prepend_timestamp=False)

        # Separate and save the demands that are kept empirical -> i.e., no
        # fitting. Currently, empirical demands are decoupled from those that
        # have a distribution fit to their samples. The correlation between
        # empirical and other demands is not preserved in the demand model.
        empirical_edps = []
        for edp in cal_df.index:
            if cal_df.loc[edp, 'Family'] == 'empirical':
                empirical_edps.append(edp)

        self.empirical_data = demand_sample.loc[:, empirical_edps].copy()

        # remove the empirical demands from the samples used for calibration
        demand_sample = demand_sample.drop(empirical_edps, axis=1)

        # and the calibration settings
        cal_df = cal_df.drop(empirical_edps, axis=0)

        if self._asmnt.log.verbose:
            self.log_msg(f"\nDemand data used for calibration:\n{demand_sample}",
                         prepend_timestamp=False)

        # fit the joint distribution
        self.log_msg("\nFitting the prescribed joint demand distribution...",
                     prepend_timestamp=False)

        demand_theta, demand_rho = uq.fit_distribution_to_sample(
            raw_samples=demand_sample.values.T,
            distribution=cal_df.loc[:, 'Family'].values,
            censored_count=censored_count,
            detection_limits=cal_df.loc[
                :, ['CensorLower', 'CensorUpper']].values,
            truncation_limits=cal_df.loc[
                :, ['TruncateLower', 'TruncateUpper']].values,
            multi_fit=False,
            logger_object=self._asmnt.log
        )
        # fit the joint distribution
        self.log_msg("\nCalibration successful, processing results...",
                     prepend_timestamp=False)

        # save the calibration results
        model_params.loc[cal_df.index, ['Theta_0', 'Theta_1']] = demand_theta

        # increase the variance of the marginal distributions, if needed
        if ~np.all(pd.isna(model_params.loc[:, 'SigIncrease'].values)):

            self.log_msg("\nIncreasing demand variance...",
                         prepend_timestamp=False)

            sig_inc = np.nan_to_num(model_params.loc[:, 'SigIncrease'].values)
            sig_0 = model_params.loc[:, 'Theta_1'].values

            model_params.loc[:, 'Theta_1'] = (
                np.sqrt(sig_0 ** 2. + sig_inc ** 2.))

        # remove unneeded fields from model_params
        for col in ('SigIncrease', 'CensorLower', 'CensorUpper'):
            model_params = model_params.drop(col, axis=1)

        # reorder the remaining fields for clarity
        model_params = model_params[[
            'Family', 'Theta_0', 'Theta_1', 'TruncateLower', 'TruncateUpper']]

        self.marginal_params = model_params

        self.log_msg("\nCalibrated demand model marginal distributions:\n"
                     + str(model_params),
                     prepend_timestamp=False)

        # save the correlation matrix
        self.correlation = pd.DataFrame(demand_rho,
                                        columns=cal_df.index,
                                        index=cal_df.index)

        self.log_msg("\nCalibrated demand model correlation matrix:\n"
                     + str(self.correlation),
                     prepend_timestamp=False)

    def save_model(self, file_prefix):
        """
        Save parameters of the demand model to a set of csv files

        """

        self.log_div()
        self.log_msg('Saving demand model...')

        # save the correlation and empirical data
        file_io.save_to_csv(self.correlation, file_prefix + '_correlation.csv')
        file_io.save_to_csv(
            self.empirical_data,
            file_prefix + '_empirical.csv',
            units=self.units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            log=self._asmnt.log,
        )

        # the log standard deviations in the marginal parameters need to be
        # scaled up before feeding to the saving method where they will be
        # scaled back down and end up being saved unscaled to the target file

        marginal_params = self.marginal_params.copy()

        log_rows = marginal_params['Family'] == 'lognormal'
        log_demands = marginal_params.loc[log_rows, :]

        for label in log_demands.index:

            if label in self.units.index:

                unit_factor = self._asmnt.calc_unit_scale_factor(self.units[label])

                marginal_params.loc[label, 'Theta_1'] *= unit_factor

        file_io.save_to_csv(
            marginal_params,
            file_prefix + '_marginals.csv',
            units=self.units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            orientation=1,
            log=self._asmnt.log,
        )

        self.log_msg('Demand model successfully saved.', prepend_timestamp=False)

    def load_model(self, data_source):
        """
        Load the model that describes demands on the asset.

        Parameters
        ----------
        data_source: string or dict
            If string, the data_source is a file prefix (<prefix> in the
            following description) that identifies the following files:
            <prefix>_marginals.csv,  <prefix>_empirical.csv,
            <prefix>_correlation.csv. If dict, the data source is a dictionary
            with the following optional keys: 'marginals', 'empirical', and
            'correlation'. The value under each key shall be a DataFrame.
        """

        self.log_div()
        self.log_msg('Loading demand model...')

        # prepare the marginal data source variable to load the data
        if isinstance(data_source, dict):
            marginal_data_source = data_source.get('marginals')
            empirical_data_source = data_source.get('empirical', None)
            correlation_data_source = data_source.get('correlation', None)
        else:
            marginal_data_source = data_source + '_marginals.csv'
            empirical_data_source = data_source + '_empirical.csv'
            correlation_data_source = data_source + '_correlation.csv'

        if empirical_data_source is not None:
            self.empirical_data = file_io.load_data(
                empirical_data_source,
                self._asmnt.unit_conversion_factors,
                log=self._asmnt.log,
            )
            if not self.empirical_data.empty:
                self.empirical_data.columns.set_names(
                    ['type', 'loc', 'dir'], inplace=True
                )
            else:
                self.empirical_data = None
        else:
            self.empirical_data = None

        if correlation_data_source is not None:
            self.correlation = file_io.load_data(
                correlation_data_source,
                self._asmnt.unit_conversion_factors,
                reindex=False, log=self._asmnt.log)
            self.correlation.index.set_names(['type', 'loc', 'dir'], inplace=True)
            self.correlation.columns.set_names(['type', 'loc', 'dir'], inplace=True)
        else:
            self.correlation = None

        # the log standard deviations in the marginal parameters need to be
        # adjusted after getting the data from the loading method where they
        # were scaled according to the units of the corresponding variable

        # Note that a data source without marginal information is not valid
        marginal_params, units = file_io.load_data(
            marginal_data_source,
            None,
            orientation=1,
            reindex=False,
            return_units=True,
            log=self._asmnt.log,
        )
        marginal_params.index.set_names(['type', 'loc', 'dir'], inplace=True)

        marginal_params = self.convert_marginal_params(marginal_params.copy(),
                                                       units)

        self.marginal_params = marginal_params
        self.units = units

        self.log_msg('Demand model successfully loaded.', prepend_timestamp=False)

    def _create_RVs(self, preserve_order=False):
        """
        Create a random variable registry for the joint distribution of demands.

        """

        # initialize the registry
        RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        # add a random variable for each demand variable
        for rv_params in self.marginal_params.itertuples():

            edp = rv_params.Index
            rv_tag = f'EDP-{edp[0]}-{edp[1]}-{edp[2]}'
            family = getattr(rv_params, "Family", np.nan)

            if family == 'empirical':

                if preserve_order:
                    dist_family = 'coupled_empirical'
                else:
                    dist_family = 'empirical'

                # empirical RVs need the data points
                RV_reg.add_RV(uq.RandomVariable(
                    name=rv_tag,
                    distribution=dist_family,
                    raw_samples=self.empirical_data.loc[:, edp].values
                ))

            else:

                # all other RVs need parameters of their distributions
                RV_reg.add_RV(uq.RandomVariable(
                    name=rv_tag,
                    distribution=family,
                    theta=[getattr(rv_params, f"Theta_{t_i}", np.nan)
                           for t_i in range(3)],
                    truncation_limits=[
                        getattr(rv_params, f"Truncate{side}", np.nan)
                        for side in ("Lower", "Upper")],


                ))

        self.log_msg(f"\n{self.marginal_params.shape[0]} random variables created.",
                     prepend_timestamp=False)

        # add an RV set to consider the correlation between demands, if needed
        if self.correlation is not None:
            rv_set_tags = [f'EDP-{edp[0]}-{edp[1]}-{edp[2]}'
                           for edp in self.correlation.index.values]

            RV_reg.add_RV_set(uq.RandomVariableSet(
                'EDP_set', list(RV_reg.RVs(rv_set_tags).values()),
                self.correlation.values))

            self.log_msg(
                f"\nCorrelations between {len(rv_set_tags)} random variables "
                "successfully defined.",
                prepend_timestamp=False)

        self._RVs = RV_reg

    def propagate_demands(self, demand_propagation):
        """
        Propagates demands. This means copying over columns of the
        original demand sample and assigning given names to them. The
        columns to be copied over and the names to assign to the
        copies are defined as the keys and values of the
        `demand_propagation` dictionary, respectively.
        The method modifies `sample` inplace.

        Parameters
        ----------
        demand_propagation : dict
            Keys correspond to the columns of the original sample to
            be copied over and the values correspond to the intended
            names for the copies. Caution: It's possible to define a
            dictionary with duplicate keys, and Python will just keep
            the last entry without warning. Users need to be careful
            enough to avoid duplicate keys, because we can't validate
            them.
            E.g.: x = {'1': 1.00, '1': 2.00} results in x={'1': 2.00}.

        Raises
        ------
        ValueError
            In multiple instances of invalid demand_propagation entries.

        """

        # it's impossible to have duplicate keys, because
        # demand_propagation is a dictionary.
        new_columns_list = demand_propagation.values()
        # The following prevents duplicate entries in the values
        # corresponding to a single propagated demand (1), but
        # also the same column being specified as the propagated
        # entry of multiple demands (2).
        # e.g.
        # (1): {'PGV-0-1': ['PGV-1-1', 'PGV-1-1', ...]}
        # (2): {'PGV-0-1': ['PGV-1-1', ...], 'PGV-0-2': ['PGV-1-1', ...]}
        flat_list = []
        for new_columns in new_columns_list:
            flat_list.extend(new_columns)
        if len(set(flat_list)) != len(flat_list):
            raise ValueError(
                'Duplicate entries in demand propagation '
                'configuration.'
            )

        # turn the config entries to tuples
        def turn_to_tuples(demand_propagation):
            demand_propagation_tuples = {}
            for key, values in demand_propagation.items():
                demand_propagation_tuples[tuple(key.split('-'))] = [
                    tuple(x.split('-')) for x in values
                ]
            return demand_propagation_tuples

        demand_propagation = turn_to_tuples(demand_propagation)

        # The demand propagation confuguration should not include
        # columns that are not present in the orignal sample.
        warn_columns = []
        for column in demand_propagation:
            if column not in self.sample.columns:
                warn_columns.append(column)
        if warn_columns:
            warn_columns = ['-'.join(x) for x in warn_columns]
            self.log_msg(
                "\nWARNING: The demand propagation configuration lists "
                "columns that are not present in the original demand sample's "
                f"columns: {warn_columns}.\n",
                prepend_timestamp=False,
            )

        # we iterate over the existing columns of the sample and try
        # to locate columns that need to be copied as required by the
        # demand propagation configuration.  If a column does not need
        # to be propagated it is left as is.  Otherwise, we keep track
        # of its initial index location (in `column_index`) and the
        # number of times it needs to be replicated, along with the
        # new names of its copies (in `column_values`).
        column_index = []
        column_values = []
        for i, column in enumerate(self.sample.columns):
            if column not in demand_propagation:
                column_index.append(i)
                column_values.append(column)
            else:
                new_column_values = demand_propagation[column]
                column_index.extend([i] * len(new_column_values))
                column_values.extend(new_column_values)
        # copy the columns
        self.sample = self.sample.iloc[:, column_index]
        # update the column index
        self.sample.columns = pd.MultiIndex.from_tuples(column_values)

    def generate_sample(self, config):
        """
        Generates an RV sample with the specified configuration.
        """

        if self.marginal_params is None:
            raise ValueError('Model parameters have not been specified. Either'
                             'load parameters from a file or calibrate the '
                             'model using raw demand data.')

        self.log_div()
        self.log_msg('Generating sample from demand variables...')

        self._create_RVs(
            preserve_order=config.get('PreserveRawOrder', False))

        sample_size = config['SampleSize']
        self._RVs.generate_sample(
            sample_size=sample_size,
            method=self._asmnt.options.sampling_method)

        # replace the potentially existing raw sample with the generated one
        assert self._RVs is not None
        assert self._RVs.RV_sample is not None
        sample = pd.DataFrame(self._RVs.RV_sample)
        sample.sort_index(axis=0, inplace=True)
        sample.sort_index(axis=1, inplace=True)

        sample = base.convert_to_MultiIndex(sample, axis=1)['EDP']

        sample.columns.names = ['type', 'loc', 'dir']
        self.sample = sample

        if config.get('DemandPropagation', False):
            self.propagate_demands(config['DemandPropagation'])

        self.log_msg(f"\nSuccessfully generated {sample_size} realizations.",
                     prepend_timestamp=False)


class AssetModel(PelicunModel):
    """
    Manages asset information used in assessments.

    Parameters
    ----------

    """

    def __init__(self, assessment):

        super().__init__(assessment)

        self.cmp_marginal_params = None
        self.cmp_units = None

        self._cmp_RVs = None
        self._cmp_sample = None

    @property
    def cmp_sample(self):
        """
        Assigns the _cmp_sample attribute if it is None and returns
        the component sample.
        """

        if self._cmp_sample is None:

            cmp_sample = pd.DataFrame(self._cmp_RVs.RV_sample)
            cmp_sample.sort_index(axis=0, inplace=True)
            cmp_sample.sort_index(axis=1, inplace=True)

            cmp_sample = base.convert_to_MultiIndex(cmp_sample, axis=1)['CMP']

            cmp_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

            self._cmp_sample = cmp_sample

        else:
            cmp_sample = self._cmp_sample

        return cmp_sample

    def save_cmp_sample(self, filepath=None, save_units=False):
        """
        Save component quantity sample to a csv file

        """

        self.log_div()
        if filepath is not None:
            self.log_msg('Saving asset components sample...')

        # prepare a units array
        sample = self.cmp_sample

        units = pd.Series(name='Units', index=sample.columns, dtype=object)

        for cmp_id, unit_name in self.cmp_units.items():
            units.loc[cmp_id, :] = unit_name

        res = file_io.save_to_csv(
            sample, filepath, units=units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log)

        if filepath is not None:
            self.log_msg('Asset components sample successfully saved.',
                         prepend_timestamp=False)
            return None
        # else:
        units = res.loc["Units"]
        res.drop("Units", inplace=True)

        if save_units:
            return res.astype(float), units

        return res.astype(float)

    def load_cmp_sample(self, filepath):
        """
        Load component quantity sample from a csv file

        """

        self.log_div()
        self.log_msg('Loading asset components sample...')

        sample, units = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors,
            return_units=True, log=self._asmnt.log)

        sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        self._cmp_sample = sample

        self.cmp_units = units.groupby(level=0).first()

        self.log_msg('Asset components sample successfully loaded.',
                     prepend_timestamp=False)

    def load_cmp_model(self, data_source):
        """
        Load the model that describes component quantities in the asset.

        Parameters
        ----------
        data_source: string or dict
            If string, the data_source is a file prefix (<prefix> in the
            following description) that identifies the following files:
            <prefix>_marginals.csv,  <prefix>_empirical.csv,
            <prefix>_correlation.csv. If dict, the data source is a dictionary
            with the following optional keys: 'marginals', 'empirical', and
            'correlation'. The value under each key shall be a DataFrame.
        """

        def get_locations(loc_str):

            try:
                res = str(int(loc_str))
                return np.array([res, ])

            except ValueError as exc:

                stories = self._asmnt.stories

                if "--" in loc_str:
                    s_low, s_high = loc_str.split('--')
                    s_low = get_locations(s_low)
                    s_high = get_locations(s_high)
                    return np.arange(int(s_low[0]), int(s_high[0]) + 1).astype(str)

                if "," in loc_str:
                    return np.array(loc_str.split(','), dtype=int).astype(str)

                if loc_str == "all":
                    return np.arange(1, stories + 1).astype(str)

                if loc_str == "top":
                    return np.array([stories, ]).astype(str)

                if loc_str == "roof":
                    return np.array([stories + 1, ]).astype(str)

                raise ValueError(f"Cannot parse location string: "
                                 f"{loc_str}") from exc

        def get_directions(dir_str):

            if pd.isnull(dir_str):
                return np.ones(1).astype(str)

            # else:
            try:
                res = str(int(dir_str))
                return np.array([res, ])

            except ValueError as exc:

                if "," in dir_str:
                    return np.array(dir_str.split(','), dtype=int).astype(str)

                if "--" in dir_str:
                    d_low, d_high = dir_str.split('--')
                    d_low = get_directions(d_low)
                    d_high = get_directions(d_high)
                    return np.arange(
                        int(d_low[0]), int(d_high[0]) + 1).astype(str)

                # else:
                raise ValueError(f"Cannot parse direction string: "
                                 f"{dir_str}") from exc

        def get_attribute(attribute_str, dtype=float, default=np.nan):

            if pd.isnull(attribute_str):
                return default

            # else:

            try:

                res = dtype(attribute_str)
                return res

            except ValueError as exc:

                if "," in attribute_str:
                    # a list of weights
                    w = np.array(attribute_str.split(','), dtype=float)

                    # return a normalized vector
                    return w / np.sum(w)

                # else:
                raise ValueError(f"Cannot parse Blocks string: "
                                 f"{attribute_str}") from exc

        self.log_div()
        self.log_msg('Loading component model...')

        # Currently, we assume independent component distributions are defined
        # throughout the building. Correlations may be added afterward or this
        # method can be extended to read correlation matrices too if needed.

        # prepare the marginal data source variable to load the data
        if isinstance(data_source, dict):
            marginal_data_source = data_source['marginals']
        else:
            marginal_data_source = data_source + '_marginals.csv'

        marginal_params, units = file_io.load_data(
            marginal_data_source,
            None,
            orientation=1,
            reindex=False,
            return_units=True,
            log=self._asmnt.log,
        )

        # group units by cmp id to avoid redundant entries
        self.cmp_units = units.copy().groupby(level=0).first()

        marginal_params = pd.concat([marginal_params, units], axis=1)

        cmp_marginal_param_dct = {
            'Family': [], 'Theta_0': [], 'Theta_1': [], 'Theta_2': [],
            'TruncateLower': [], 'TruncateUpper': [], 'Blocks': [],
            'Units': []
        }
        index_list = []
        for row in marginal_params.itertuples():
            locs = get_locations(row.Location)
            dirs = get_directions(row.Direction)
            indices = list(product((row.Index, ), locs, dirs))
            num_vals = len(indices)
            for col, cmp_marginal_param in cmp_marginal_param_dct.items():
                if col == 'Blocks':
                    cmp_marginal_param.extend(
                        [
                            get_attribute(
                                getattr(row, 'Blocks', np.nan),
                                dtype=int,
                                default=1.0,
                            )
                        ]
                        * num_vals
                    )
                elif col == 'Units':
                    cmp_marginal_param.extend(
                        [self.cmp_units[row.Index]] * num_vals
                    )
                elif col == 'Family':
                    cmp_marginal_param.extend(
                        [getattr(row, col, np.nan)] * num_vals
                    )
                else:
                    cmp_marginal_param.extend(
                        [get_attribute(getattr(row, col, np.nan))] * num_vals
                    )
            index_list.extend(indices)
        index = pd.MultiIndex.from_tuples(index_list, names=['cmp', 'loc', 'dir'])
        dtypes = {
            'Family': object, 'Theta_0': float, 'Theta_1': float,
            'Theta_2': float, 'TruncateLower': float,
            'TruncateUpper': float, 'Blocks': int, 'Units': object
        }
        cmp_marginal_param_series = []
        for col, cmp_marginal_param in cmp_marginal_param_dct.items():
            cmp_marginal_param_series.append(
                pd.Series(
                    cmp_marginal_param,
                    dtype=dtypes[col], name=col, index=index))

        cmp_marginal_params = pd.concat(
            cmp_marginal_param_series, axis=1
        )

        assert not cmp_marginal_params['Theta_0'].isnull().values.any()

        cmp_marginal_params.dropna(axis=1, how='all', inplace=True)

        self.log_msg("Model parameters successfully parsed. "
                     f"{cmp_marginal_params.shape[0]} performance groups identified",
                     prepend_timestamp=False)

        # Now we can take care of converting the values to base units
        self.log_msg("Converting model parameters to internal units...",
                     prepend_timestamp=False)

        # ensure that the index has unique entries by introducing an
        # internal component uid
        base.dedupe_index(cmp_marginal_params)

        cmp_marginal_params = self.convert_marginal_params(
            cmp_marginal_params, cmp_marginal_params['Units']
        )

        self.cmp_marginal_params = cmp_marginal_params.drop('Units', axis=1)

        self.log_msg("Model parameters successfully loaded.",
                     prepend_timestamp=False)

        self.log_msg("\nComponent model marginal distributions:\n"
                     + str(cmp_marginal_params),
                     prepend_timestamp=False)

        # the empirical data and correlation files can be added later, if needed

    def _create_cmp_RVs(self):
        """
        Defines the RVs used for sampling component quantities.
        """

        # initialize the registry
        RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        # add a random variable for each component quantity variable
        for rv_params in self.cmp_marginal_params.itertuples():

            cmp = rv_params.Index

            # create a random variable and add it to the registry
            RV_reg.add_RV(uq.RandomVariable(
                name=f'CMP-{cmp[0]}-{cmp[1]}-{cmp[2]}-{cmp[3]}',
                distribution=getattr(rv_params, "Family", np.nan),
                theta=[getattr(rv_params, f"Theta_{t_i}", np.nan)
                       for t_i in range(3)],
                truncation_limits=[getattr(rv_params, f"Truncate{side}", np.nan)
                                   for side in ("Lower", "Upper")],
            ))

        self.log_msg(f"\n{self.cmp_marginal_params.shape[0]} "
                     "random variables created.",
                     prepend_timestamp=False)

        self._cmp_RVs = RV_reg

    def generate_cmp_sample(self, sample_size=None):
        """
        Generates component quantity realizations.  If a sample_size
        is not specified, the sample size found in the demand model is
        used.
        """

        if self.cmp_marginal_params is None:
            raise ValueError('Model parameters have not been specified. Load'
                             'parameters from a file before generating a '
                             'sample.')

        self.log_div()
        self.log_msg('Generating sample from component quantity variables...')

        if sample_size is None:
            if self._asmnt.demand.sample is None:
                raise ValueError(
                    'Sample size was not specified, '
                    'and it cannot be determined from '
                    'the demand model.')
            sample_size = self._asmnt.demand.sample.shape[0]

        self._create_cmp_RVs()

        self._cmp_RVs.generate_sample(
            sample_size=sample_size,
            method=self._asmnt.options.sampling_method)

        # replace the potentially existing sample with the generated one
        self._cmp_sample = None

        self.log_msg(f"\nSuccessfully generated {sample_size} realizations.",
                     prepend_timestamp=False)


class DamageModel(PelicunModel):
    """
    Manages damage information used in assessments.

    This class contains the following methods:

    - save_sample()
    - load_sample()
    - load_damage_model()
    - calculate()
        - _get_pg_batches()
        - _generate_dmg_sample()
            - _create_dmg_rvs()
        - _get_required_demand_type()
        - _assemble_required_demand_data()
        - _evaluate_damage_state()
        - _prepare_dmg_quantities()
        - _perform_dmg_task()
        - _apply_dmg_funcitons()

    Parameters
    ----------

    """

    def __init__(self, assessment):

        super().__init__(assessment)

        self.damage_params = None
        self.sample = None

    def save_sample(self, filepath=None, save_units=False):
        """
        Save damage sample to a csv file

        """
        self.log_div()
        self.log_msg('Saving damage sample...')

        cmp_units = self._asmnt.asset.cmp_units
        qnt_units = pd.Series(index=self.sample.columns, name='Units',
                              dtype='object')
        for cmp in cmp_units.index:
            qnt_units.loc[cmp] = cmp_units.loc[cmp]

        res = file_io.save_to_csv(
            self.sample, filepath,
            units=qnt_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log)

        if filepath is not None:
            self.log_msg('Damage sample successfully saved.',
                         prepend_timestamp=False)
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
            filepath, self._asmnt.unit_conversion_factors,
            log=self._asmnt.log)

        # set the names of the columns
        self.sample.columns.names = ['cmp', 'loc', 'dir', 'uid', 'ds']

        self.log_msg('Damage sample successfully loaded.',
                     prepend_timestamp=False)

    def load_damage_model(self, data_paths):
        """
        Load limit state damage model parameters and damage state assignments

        Parameters
        ----------
        data_paths: list of string
            List of paths to data files with damage model information. Default
            XY datasets can be accessed as PelicunDefault/XY.
        """

        self.log_div()
        self.log_msg('Loading damage model...')

        # replace default flag with default data path
        for d_i, data_path in enumerate(data_paths):

            if 'PelicunDefault/' in data_path:
                data_paths[d_i] = data_path.replace(
                    'PelicunDefault/',
                    f'{base.pelicun_path}/resources/SimCenterDBDL/',
                )

        data_list = []
        # load the data files one by one
        for data_path in data_paths:

            data = file_io.load_data(
                data_path, None, orientation=1, reindex=False, log=self._asmnt.log
            )

            data_list.append(data)

        damage_params = pd.concat(data_list, axis=0)

        # drop redefinitions of components
        damage_params = damage_params.groupby(damage_params.index).first()

        # get the component types defined in the asset model
        cmp_labels = self._asmnt.asset.cmp_sample.columns

        # only keep the damage model parameters for the components in the model
        cmp_unique = cmp_labels.unique(level=0)
        cmp_mask = damage_params.index.isin(cmp_unique, level=0)

        damage_params = damage_params.loc[cmp_mask, :]

        if np.sum(cmp_mask) != len(cmp_unique):

            cmp_list = cmp_unique[
                np.isin(cmp_unique, damage_params.index.values,
                        invert=True)].to_list()

            self.log_msg("\nWARNING: The damage model does not provide "
                         "vulnerability information for the following component(s) "
                         f"in the asset model: {cmp_list}.\n",
                         prepend_timestamp=False)

        # TODO: load defaults for Demand-Offset and Demand-Directional

        # Now convert model parameters to base units
        for LS_i in damage_params.columns.unique(level=0):
            if LS_i.startswith('LS'):

                damage_params.loc[:, LS_i] = self.convert_marginal_params(
                    damage_params.loc[:, LS_i].copy(),
                    damage_params[('Demand', 'Unit')],
                ).values

        # check for components with incomplete damage model information
        cmp_incomplete_list = damage_params.loc[
            damage_params[('Incomplete', '')] == 1].index

        damage_params.drop(cmp_incomplete_list, inplace=True)

        if len(cmp_incomplete_list) > 0:
            self.log_msg(f"\nWARNING: Damage model information is incomplete for "
                         f"the following component(s) {cmp_incomplete_list}. They "
                         f"were removed from the analysis.\n",
                         prepend_timestamp=False)

        self.damage_params = damage_params

        self.log_msg("Damage model parameters successfully parsed.",
                     prepend_timestamp=False)

    def _handle_operation(self, initial_value, operation, other_value):
        """
        This method is used in `_create_dmg_RVs` to apply capacity
        scaling operations whenever required. It is defined as a safer
        alternative to directly using `eval`.

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
        result: float
          The result of the operation

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
        performance group in the input performance group block (PGB)
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

        """

        def assign_lsds(ds_weights, ds_id, lsds_RV_reg, lsds_rv_tag):
            """
            Prepare random variables to handle mutually exclusive damage states.

            """

            # If the limit state has a single damage state assigned
            # to it, we don't need random sampling
            if pd.isnull(ds_weights):

                ds_id += 1

                lsds_RV_reg.add_RV(
                    uq.RandomVariable(
                        name=lsds_rv_tag,
                        distribution='deterministic',
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
                    return values + offset

                lsds_RV_reg.add_RV(
                    uq.RandomVariable(
                        name=lsds_rv_tag,
                        distribution='multinomial',
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

        # capacity scaling:
        # ensure the scaling_specification is a dictionary
        if not scaling_specification:
            scaling_specification = {}
        else:
            # if there are contents, ensure they are valid.
            # See docstring for an example of what is expected.
            parsed_scaling_specification = {}
            # validate contents
            for key, value in scaling_specification.items():
                css = 'capacity scaling specification'
                if not isinstance(value, str):
                    raise ValueError(
                        f'Invalud entry in {css}: {value}. It has to be a string. '
                        f'See docstring of DamageModel._create_dmg_RVs.'
                    )
                capacity_scaling_operation = value[0]
                number = value[1::]
                if capacity_scaling_operation not in ('+', '-', '*', '/'):
                    raise ValueError(
                        f'Invalid operation in {css}: {capacity_scaling_operation}'
                    )
                fnumber = base.float_or_None(number)
                if fnumber is None:
                    raise ValueError(f'Invalid number in {css}: {number}')
                parsed_scaling_specification[key] = (
                    capacity_scaling_operation,
                    fnumber,
                )
                scaling_specification = parsed_scaling_specification

        # get the component sample and blocks from the asset model
        for PG in PGB.index:

            # determine demand capacity scaling operation, if required
            cmp_loc_dir = '-'.join(PG[0:3])
            capacity_scaling_operation = scaling_specification.get(cmp_loc_dir, None)

            cmp_id = PG[0]
            blocks = PGB.loc[PG, 'Blocks']

            # if the number of blocks is provided, calculate the weights
            if np.atleast_1d(blocks).shape[0] == 1:
                blocks = np.full(int(blocks), 1.0 / blocks)
            # otherwise, assume that the list contains the weights

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
                    family = frg_params_LS.get('Family', np.nan)
                    ds_weights = frg_params_LS.get('DamageStateWeights', np.nan)

                    # check if the limit state is defined for the component
                    if pd.isna(theta_0):
                        continue

                    theta = [
                        frg_params_LS.get(f"Theta_{t_i}", np.nan) for t_i in range(3)
                    ]

                    if capacity_scaling_operation:
                        if family in {'normal', 'lognormal'}:
                            theta[0] = self._handle_operation(
                                theta[0],
                                capacity_scaling_operation[0],
                                capacity_scaling_operation[1],
                            )
                        else:
                            self.log_msg(
                                f'\nWARNING: Capacity scaling is only supported '
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

                        RV = uq.RandomVariable(
                            name=frg_rv_tag,
                            distribution=family,
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
            raise ValueError('Damage model parameters have not been specified. '
                             'Load parameters from the default damage model '
                             'databases or provide your own damage model '
                             'definitions before generating a sample.')

        # Create capacity and LSD RVs for each performance group
        capacity_RVs, lsds_RVs = self._create_dmg_RVs(PGB, scaling_specification)

        if self._asmnt.log.verbose:
            self.log_msg('Sampling capacities...',
                         prepend_timestamp=True)

        # Generate samples for capacity RVs
        capacity_RVs.generate_sample(
            sample_size=sample_size,
            method=self._asmnt.options.sampling_method)

        # Generate samples for LSD RVs
        lsds_RVs.generate_sample(
            sample_size=sample_size,
            method=self._asmnt.options.sampling_method)

        if self._asmnt.log.verbose:
            self.log_msg("Raw samples are available",
                         prepend_timestamp=True)

        # get the capacity and lsds samples
        capacity_sample = pd.DataFrame(
            capacity_RVs.RV_sample).sort_index(
                axis=0).sort_index(axis=1)
        capacity_sample = base.convert_to_MultiIndex(
            capacity_sample, axis=1)['FRG']
        capacity_sample.columns.names = [
            'cmp', 'loc', 'dir', 'uid', 'block', 'ls']

        lsds_sample = pd.DataFrame(
            lsds_RVs.RV_sample).sort_index(
                axis=0).sort_index(axis=1).astype(int)
        lsds_sample = base.convert_to_MultiIndex(
            lsds_sample, axis=1)['LSDS']
        lsds_sample.columns.names = [
            'cmp', 'loc', 'dir', 'uid', 'block', 'ls']

        if self._asmnt.log.verbose:
            self.log_msg(
                f"Successfully generated {sample_size} realizations.",
                prepend_timestamp=True)

        return capacity_sample, lsds_sample

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
        EDP_req: dict
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
            self.log_msg('Collecting required demand information...',
                         prepend_timestamp=True)

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
                cmp, [('Demand', 'Directional'),
                      ('Demand', 'Offset'),
                      ('Demand', 'Type')]]

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
            self.log_msg('Assembling demand data for calculation...',
                         prepend_timestamp=True)

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
                    demand = demand_source.loc[
                        :, (EDP[0], EDP[1])].max(axis=1).values
                    demand = demand * self._asmnt.options.nondir_multi(EDP[0])

                except KeyError:

                    demand = None

            else:
                demand = demand_source[(EDP[0], EDP[1], EDP[2])].values

            if demand is None:

                self.log_msg(f'\nWARNING: Cannot find demand data for {EDP}. The '
                             'corresponding damages cannot be calculated.',
                             prepend_timestamp=False)
            else:
                demand_dict.update({f'{EDP[0]}-{EDP[1]}-{EDP[2]}': demand})

        return demand_dict

    def _evaluate_damage_state(
            self, demand_dict, EDP_req, capacity_sample, lsds_sample):
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
        dmg_sample: DataFrame
            Assigns a Damage State to each component block in the
            asset model.
        """

        # Log a message indicating that damage states are being
        # evaluated

        if self._asmnt.log.verbose:
            self.log_msg('Evaluating damage states...', prepend_timestamp=True)

        # Create an empty dataframe with columns and index taken from
        # the input capacity sample
        dmg_eval = pd.DataFrame(columns=capacity_sample.columns,
                                index=capacity_sample.index)

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
            demand_df.append(pd.concat([pd.Series(demand_vals)] * len(PG_cols),
                                       axis=1, keys=PG_cols))

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
        ds_sample = capacity_sample.groupby(level=[0, 1, 2, 3, 4], axis=1).first()
        ds_sample.loc[:, :] = np.zeros(ds_sample.shape, dtype=int)

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
            ds_sample.loc[:, dmg_e_ls.columns] = (
                ds_sample.loc[:, dmg_e_ls.columns].mask(dmg_e_ls, lsds))

        return ds_sample

    def _prepare_dmg_quantities(self, PGB, ds_sample, dropzero=True):
        """
        Combine component quantity and damage state information in one
        DataFrame.

        This method assumes that a component quantity sample is
        available in the asset model and a damage state sample is
        available in the damage model.

        Parameters
        ----------
        PGB: DataFrame
            A DataFrame that contains the Block identifier for each
            component.
        ds_sample: DataFrame
            A DataFrame that assigns a damage state to each component
            block in the asset model.
        dropzero: bool, optional, default: True
            If True, the quantity of non-damaged components is not
            saved.

        Returns
        -------
        res_df: DataFrame
            A DataFrame that combines the component quantity and
            damage state information.

        Raises
        ------
        ValueError
            If the number of blocks is not provided or if the list of
            weights does not contain the same number of elements as
            the number of blocks.

        """

        # Log a message indicating that the calculation of damage
        # quantities is starting
        if self._asmnt.log.verbose:
            self.log_msg('Calculating damage quantities...',
                         prepend_timestamp=True)

        # Store the damage state sample as a local variable
        dmg_ds = ds_sample

        # Retrieve the component quantity information from the asset
        # model
        cmp_qnt = self._asmnt.asset.cmp_sample  # .values
        # Retrieve the component marginal parameters from the asset
        # model
        cmp_params = self._asmnt.asset.cmp_marginal_params

        # Combine the component quantity information for the columns
        # in the damage state sample
        dmg_qnt = pd.concat(
            [cmp_qnt[PG[:4]] for PG in dmg_ds.columns],
            axis=1, keys=dmg_ds.columns)

        # Initialize a list to store the block weights
        block_weights = []

        # For each component in the list of PG blocks
        for PG in PGB.index:

            # Set the number of blocks to 1, unless specified
            # otherwise in the component marginal parameters
            blocks = 1
            if cmp_params is not None:
                if 'Blocks' in cmp_params.columns:

                    blocks = cmp_params.loc[PG, 'Blocks']

            # If the number of blocks is specified, calculate the
            # weights as the reciprocal of the number of blocks
            if np.atleast_1d(blocks).shape[0] == 1:
                blocks_array = np.full(int(blocks), 1. / blocks)

            # Otherwise, assume that the list contains the weights
            block_weights += blocks_array.tolist()

        # Broadcast the block weights to match the shape of the damage
        # quantity DataFrame
        block_weights = np.broadcast_to(
            block_weights,
            (dmg_qnt.shape[0], len(block_weights)))

        # Multiply the damage quantities by the block weights
        dmg_qnt *= block_weights

        # Get the unique damage states from the damage state sample
        # Note that these might be fewer than all possible Damage
        # States
        ds_list = np.unique(dmg_ds.values)
        # Filter out any NaN values from the list of damage states
        ds_list = ds_list[pd.notna(ds_list)].astype(int)

        # If the dropzero option is True, remove the zero damage state
        # from the list of damage states
        if dropzero:

            ds_list = ds_list[ds_list != 0]

        # Only proceed with the calculation if there is at least one
        # damage state in the list
        if len(ds_list) > 0:

            # Create a list of DataFrames, where each DataFrame stores
            # the damage quantities for a specific damage state
            res_list = [pd.DataFrame(
                np.where(dmg_ds == ds_i, dmg_qnt, 0),
                columns=dmg_ds.columns,
                index=dmg_ds.index
            ) for ds_i in ds_list]

            # Combine the damage quantity DataFrames into a single
            # DataFrame
            res_df = pd.concat(
                res_list, axis=1,
                keys=[f'{ds_i:g}' for ds_i in ds_list])
            res_df.columns.names = ['ds', *res_df.columns.names[1::]]
            # remove the block level from the columns
            res_df.columns = res_df.columns.reorder_levels([1, 2, 3, 4, 0, 5])
            res_df = res_df.groupby(level=[0, 1, 2, 3, 4], axis=1).sum()

            # The damage states with no damaged quantities are dropped
            # Note that some of these are not even valid DSs at the given PG
            res_df = res_df.iloc[:, np.where(res_df.sum(axis=0) != 0)[0]]

        return res_df

    def _perform_dmg_task(self, task, qnt_sample):
        """
        Perform a task from a damage process.

        The method performs a task from a damage process on a given
        quantity sample. The method first checks if the source
        component specified in the task exists among the available
        components in the quantity sample. If the source component is
        not found, a warning message is logged and the method returns
        the original quantity sample unchanged.  Otherwise, the method
        executes the events specified in the task. The events can be
        triggered by a limit state exceedance or a damage state
        occurrence. If the event is triggered by a damage state, the
        method moves all quantities of the target component(s) into
        the target damage state in pre-selected realizations. If the
        target event is "NA", the method removes quantity information
        from the target components in the pre-selected
        realizations. After executing the events, the method returns
        the updated quantity sample.

        Parameters
        ----------
        task : list
            A list representing a task from the damage process. The
            list contains two elements:
            - The first element is a string representing the source
              component, e.g., `'CMP_A'`.
            - The second element is a dictionary representing the
              events triggered by the damage state of the source
              component. The keys of the dictionary are strings that
              represent the damage state of the source component,
              e.g., `'DS1'`. The values are lists of strings
              representing the target component(s) and event(s), e.g.,
              `['CMP_B', 'CMP_C']`.
        qnt_sample : pandas DataFrame
            A DataFrame representing the quantities of the components
            in the damage sample. It is modified in place to represent
            the quantities of the components in the damage sample
            after the task has been performed.

        Raises
        ------
        ValueError
            If the source component is not found among the components
            in the damage sample
        ValueError
            If the source event is not a limit state (LS) or damage
            state (DS)
        ValueError
            If the target event is not a limit state (LS), damage
            state (DS), or not available (NA)
        ValueError
            If the target event is a limit state (LS)

        """

        if self._asmnt.log.verbose:
            self.log_msg('Applying task...',
                         prepend_timestamp=True)

        # get the list of available components
        cmp_list = qnt_sample.columns.get_level_values(0).unique().tolist()

        # get the component quantities
        cmp_qnt = self._asmnt.asset.cmp_sample

        # get the source component
        source_cmp = task[0].split('_')[1]

        # check if it exists among the available ones
        if source_cmp not in cmp_list:

            self.log_msg(
                f"WARNING: Source component {source_cmp} in the prescribed "
                "damage process not found among components in the damage "
                "sample. The corresponding part of the damage process is "
                "skipped.", prepend_timestamp=False)

            return

        # get the damage quantities for the source component
        source_cmp_df = qnt_sample.loc[:, source_cmp]

        # execute the prescribed events
        for source_event, target_infos in task[1].items():

            # events triggered by limit state exceedance
            if source_event.startswith('LS'):

                # ls_i = int(source_event[2:])
                # TODO: implement source LS support
                raise ValueError('LS not supported yet.')

            # events triggered by damage state occurrence
            if source_event.startswith('DS'):

                # get the ID of the damage state that triggers the event
                ds_list = [source_event[2:], ]

                # if we are only looking for a single DS
                if len(ds_list) == 1:

                    ds_target = ds_list[0]

                    # get the realizations with non-zero quantity of the target DS
                    source_ds_vals = source_cmp_df.groupby(
                        level=[3], axis=1).max()

                    if ds_target in source_ds_vals.columns:
                        source_ds_vals = source_ds_vals[ds_target]
                        source_mask = source_cmp_df.loc[source_ds_vals > 0.0].index
                    else:
                        # if tge source_cmp is not in ds_target in any of the
                        # realizations, the prescribed event is not triggered
                        continue

                else:
                    pass  # TODO: implement multiple DS support

            else:
                raise ValueError(f"Unable to parse source event in damage "
                                 f"process: {source_event}")

            # get the information about the events
            target_infos = np.atleast_1d(target_infos)

            # for each event
            for target_info in target_infos:

                # get the target component and event type
                target_cmp, target_event = target_info.split('_')

                # ALL means all, but the source component
                if target_cmp == 'ALL':

                    # copy the list of available components
                    target_cmp = deepcopy(cmp_list)

                    # remove the source component
                    if source_cmp in target_cmp:
                        target_cmp.remove(source_cmp)

                # otherwise we target a specific component
                elif target_cmp in cmp_list:
                    target_cmp = [target_cmp, ]

                # trigger a limit state
                if target_event.startswith('LS'):

                    # ls_i = int(target_event[2:])
                    # TODO: implement target LS support
                    raise ValueError('LS not supported yet.')

                # trigger a damage state
                if target_event.startswith('DS'):

                    # get the target damage state ID
                    ds_i = target_event[2:]

                    # move all quantities of the target component(s) into the
                    # target damage state in the pre-selected realizations
                    qnt_sample.loc[source_mask, target_cmp] = 0.0

                    for target_cmp_i in target_cmp:
                        locs = cmp_qnt[target_cmp_i].columns.get_level_values(0)
                        dirs = cmp_qnt[target_cmp_i].columns.get_level_values(1)
                        uids = cmp_qnt[target_cmp_i].columns.get_level_values(2)
                        for loc, direction, uid in zip(locs, dirs, uids):
                            # because we cannot be certain that ds_i had been
                            # triggered earlier, we have to add this damage
                            # state manually for each PG of each component, if needed
                            if ds_i not in qnt_sample[
                                    (target_cmp_i, loc, direction, uid)].columns:
                                qnt_sample[
                                    (target_cmp_i, loc, direction, uid, ds_i)] = 0.0

                            qnt_sample.loc[
                                source_mask,
                                (target_cmp_i, loc, direction, uid, ds_i)] = (
                                cmp_qnt.loc[
                                    source_mask,
                                    (target_cmp_i, loc, direction, uid)].values)

                # clear all damage information
                elif target_event == 'NA':

                    # remove quantity information from the target components
                    # in the pre-selected realizations
                    qnt_sample.loc[source_mask, target_cmp] = np.nan

                else:
                    raise ValueError(f"Unable to parse target event in damage "
                                     f"process: {target_event}")

        if self._asmnt.log.verbose:
            self.log_msg('Damage process task successfully applied.',
                         prepend_timestamp=False)

    def _get_pg_batches(self, block_batch_size):
        """
        Group performance groups into batches for efficient damage assessment.

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

        """

        # Get the marginal parameters for the components from the
        # asset model
        cmp_marginals = self._asmnt.asset.cmp_marginal_params

        # Initialize the batch dataframe
        pg_batch = None

        # If marginal parameters are available, use the 'Blocks'
        # column to initialize the batch dataframe
        if cmp_marginals is not None:

            # Check if the "Blocks" column exists in the component
            # marginal parameters
            if 'Blocks' in cmp_marginals.columns:
                pg_batch = cmp_marginals['Blocks'].to_frame()

        # If the "Blocks" column doesn't exist, create a new dataframe
        # with "Blocks" column filled with ones, using the component
        # sample as the index.
        if pg_batch is None:
            cmp_sample = self._asmnt.asset.cmp_sample
            pg_batch = pd.DataFrame(np.ones(cmp_sample.shape[1]),
                                    index=cmp_sample.columns,
                                    columns=['Blocks'])

        # Check if the damage model information exists for each
        # performance group If not, remove the performance group from
        # the analysis and log a warning message.
        first_time = True
        for pg_i in pg_batch.index:

            if np.any(np.isin(pg_i, self.damage_params.index)):

                blocks_i = pg_batch.loc[pg_i, 'Blocks']

                # If the "Blocks" column contains a list of block
                # weights, get the number of blocks from the shape of
                # the list.
                if np.atleast_1d(blocks_i).shape[0] != 1:
                    blocks_i = np.atleast_1d(blocks_i).shape[0]

                pg_batch.loc[pg_i, 'Blocks'] = blocks_i

            else:
                pg_batch.drop(pg_i, inplace=True)

                if first_time:
                    self.log_msg("\nWARNING: Damage model information is "
                                 "incomplete for some of the performance groups "
                                 "and they had to be removed from the analysis:",
                                 prepend_timestamp=False)

                    first_time = False

                self.log_msg(f"{pg_i}", prepend_timestamp=False)

        # Convert the data types of the dataframe to be efficient
        pg_batch = pg_batch.convert_dtypes()

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
                np.array([pg_batch['CBlocks'] <= block_batch_size,
                          pg_batch['CBlocks'] > 0]),
                axis=0)

            if np.sum(batch_mask) < 1:
                batch_mask = np.full(batch_mask.shape, False)
                batch_mask[np.where(pg_batch['CBlocks'] > 0)[0][0]] = True

            pg_batch.loc[batch_mask, 'Batch'] = batch_i

            # Decrement the cumulative block count by the max count in
            # the current batch
            pg_batch['CBlocks'] -= pg_batch.loc[
                pg_batch['Batch'] == batch_i, 'CBlocks'].max()

            # If the maximum cumulative block count is 0, exit the
            # loop
            if pg_batch['CBlocks'].max() == 0:
                break

        # Group the performance groups by batch, component, location,
        # and direction, and keep only the number of blocks for each
        # group
        pg_batch = pg_batch.groupby(
            ['Batch', 'cmp', 'loc', 'dir', 'uid']).sum().loc[:, 'Blocks'].to_frame()

        return pg_batch

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
                            cmp_data[(ls, 'DamageStateWeights')].split('|'))

            # get the list of valid cmp-loc-dir-uid sets
            cmp_header = dmg_header.loc[:, [cmp_id, ]]

            # Create a dataframe where they are repeated ds_count times in the
            # columns. The keys put the DS id in the first level of the
            # multiindexed column
            cmp_headers = pd.concat(
                [cmp_header for ds_i in range(ds_count + 1)],
                keys=[str(r) for r in range(0, ds_count + 1)],
                axis=1)
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

    def calculate(
        self, dmg_process=None, block_batch_size=1000, scaling_specification=None
    ):
        """
        Calculate the damage state of each component block in the asset.

        """

        self.log_div()
        self.log_msg('Calculating damages...')

        sample_size = self._asmnt.demand.sample.shape[0]

        # Break up damage calculation and perform it by performance group.
        # Compared to the simultaneous calculation of all PGs, this approach
        # reduces demands on memory and increases the load on CPU. This leads
        # to a more balanced workload on most machines for typical problems.
        # It also allows for a straightforward extension with parallel
        # computing.

        # get the list of performance groups
        qnt_samples = []

        self.log_msg(f'Number of Performance Groups in Asset Model:'
                     f' {self._asmnt.asset.cmp_sample.shape[1]}',
                     prepend_timestamp=False)

        pg_batch = self._get_pg_batches(block_batch_size)
        batches = pg_batch.index.get_level_values(0).unique()

        self.log_msg(f'Number of Component Blocks: {pg_batch["Blocks"].sum()}',
                     prepend_timestamp=False)

        self.log_msg(f"{len(batches)} batches of Performance Groups prepared "
                     "for damage assessment",
                     prepend_timestamp=False)

        # for PG_i in self._asmnt.asset.cmp_sample.columns:
        for PGB_i in batches:

            PGB = pg_batch.loc[PGB_i]

            self.log_msg(f"Calculating damage for PG batch {PGB_i} with "
                         f"{int(PGB['Blocks'].sum())} blocks")

            # Generate an array with component capacities for each block and
            # generate a second array that assigns a specific damage state to
            # each component limit state. The latter is primarily needed to
            # handle limit states with multiple, mutually exclusive DS options
            capacity_sample, lsds_sample = self._generate_dmg_sample(
                sample_size, PGB, scaling_specification)

            # Get the required demand types for the analysis
            EDP_req = self._get_required_demand_type(PGB)

            # Create the demand vector
            demand_dict = self._assemble_required_demand_data(EDP_req)

            # Evaluate the Damage State of each Component Block
            ds_sample = self._evaluate_damage_state(
                demand_dict, EDP_req,
                capacity_sample, lsds_sample)
            qnt_sample = self._prepare_dmg_quantities(PGB, ds_sample, dropzero=False)

            qnt_samples.append(qnt_sample)

        qnt_sample = pd.concat(qnt_samples, axis=1)

        # Create a comprehensive table with all possible DSs to have a robust
        # input for the damage processes evaluation below
        qnt_sample = self._complete_ds_cols(qnt_sample)
        qnt_sample.sort_index(axis=1, inplace=True)

        self.log_msg("Raw damage calculation successful.",
                     prepend_timestamp=False)

        # Apply the prescribed damage process, if any
        if dmg_process is not None:
            self.log_msg("Applying damage processes...")

            # sort the processes
            dmg_process = {key: dmg_process[key] for key in sorted(dmg_process)}

            for task in dmg_process.items():

                self._perform_dmg_task(task, qnt_sample)

            self.log_msg("Damage processes successfully applied.",
                         prepend_timestamp=False)

        # If requested, remove columns with no damage from the sample
        if self._asmnt.options.list_all_ds is False:
            qnt_sample = qnt_sample.iloc[:, np.where(qnt_sample.sum(axis=0) != 0)[0]]

        self.sample = qnt_sample

        self.log_msg('Damage calculation successfully completed.')


class LossModel(PelicunModel):
    """
    Parent object for loss models.

    All loss assessment methods should be children of this class.

    Parameters
    ----------

    """

    def __init__(self, assessment):

        super().__init__(assessment)

        self._sample = None
        self.loss_map = None
        self.loss_params = None
        self.loss_type = 'Generic'

    @property
    def sample(self):
        """
        sample property
        """
        return self._sample

    def save_sample(self, filepath=None, save_units=False):
        """
        Save loss sample to a csv file

        """
        self.log_div()
        if filepath is not None:
            self.log_msg('Saving loss sample...')

        cmp_units = self.loss_params[('DV', 'Unit')]
        dv_units = pd.Series(index=self.sample.columns, name='Units',
                             dtype='object')

        for cmp_id, dv_type in cmp_units.index:
            dv_units.loc[(dv_type, cmp_id)] = cmp_units.at[(cmp_id, dv_type)]

        res = file_io.save_to_csv(
            self.sample, filepath, units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log)

        if filepath is not None:
            self.log_msg('Loss sample successfully saved.',
                         prepend_timestamp=False)
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

        self._sample = file_io.load_data(
            filepath, self._asmnt.unit_conversion_factors, log=self._asmnt.log)

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
        for d_i, data_path in enumerate(data_paths):

            if 'PelicunDefault/' in data_path:
                data_paths[d_i] = data_path.replace(
                    'PelicunDefault/',
                    f'{base.pelicun_path}/resources/SimCenterDBDL/')

        data_list = []
        # load the data files one by one
        for data_path in data_paths:
            data = file_io.load_data(
                data_path, None, orientation=1, reindex=False, log=self._asmnt.log
            )

            data_list.append(data)

        loss_params = pd.concat(data_list, axis=0)

        # drop redefinitions of components
        loss_params = loss_params.groupby(
            level=[0, 1]).first().transform(lambda x: x.fillna(np.nan))
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
            self.log_msg("\nWARNING: The loss model does not provide "
                         "consequence information for the following component(s) "
                         f"in the loss map: {missing_cmp}. They are removed from "
                         "further analysis\n",
                         prepend_timestamp=False)

        self.loss_map = self.loss_map.loc[
            ~loss_map['Consequence'].isin(missing_cmp)]
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
                    loss_params[('Quantity', 'Unit')]
                ).values

        # check for components with incomplete loss information
        cmp_incomplete_list = loss_params.loc[
            loss_params[('Incomplete', '')] == 1].index

        if len(cmp_incomplete_list) > 0:
            loss_params.drop(cmp_incomplete_list, inplace=True)

            self.log_msg(
                "\n"
                "WARNING: Loss information is incomplete for the "
                f"following component(s) {cmp_incomplete_list}. "
                "They were removed from the analysis."
                "\n",
                prepend_timestamp=False)

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

        self.log_msg("Loss parameters successfully parsed.",
                     prepend_timestamp=False)

    def aggregate_losses(self):
        """
        This is placeholder method.

        The method of aggregating the Decision Variable sample is specific to
        each DV and needs to be implemented in every child of the LossModel
        independently.
        """
        raise NotImplementedError

    def _generate_DV_sample(self, dmg_quantities, sample_size):
        """
        This is placeholder method.

        The method of sampling decision variables in Decision
        Variable-specific and needs to be implemented in every child
        of the LossModel independently.
        """
        raise NotImplementedError

    def calculate(self):
        """
        Calculate the consequences of each component block damage in
        the asset.

        """

        self.log_div()
        self.log_msg("Calculating losses...")

        drivers = [d for d, _ in self.loss_map['Driver']]

        if 'DMG' in drivers:
            sample_size = self._asmnt.damage.sample.shape[0]
        elif 'DEM' in drivers:
            sample_size = self._asmnt.demand.sample.shape[0]
        else:
            raise ValueError(
                'Invalid loss drivers. Check the specified loss map.')

        # First, get the damaged quantities in each damage state for
        # each component of interest.
        dmg_q = self._asmnt.damage.sample.copy()

        # Now sample random Decision Variables
        # Note that this method is DV-specific and needs to be
        # implemented in every child of the LossModel independently.
        self._generate_DV_sample(dmg_q, sample_size)

        self.log_msg("Loss calculation successful.")


class BldgRepairModel(LossModel):
    """
    Manages building repair consequence assessments.

    Parameters
    ----------

    """

    def __init__(self, assessment):

        super().__init__(assessment)

        self.loss_type = 'BldgRepair'

    # def load_model(self, data_paths, mapping_path):

    #     super().load_model(data_paths, mapping_path)

    # def calculate(self):

    #     super().calculate()

    def _create_DV_RVs(self, case_list):
        """
        Prepare the random variables used for repair cost and time simulation.

        Parameters
        ----------
        case_list: MultiIndex
            Index with cmp-loc-dir-ds descriptions that identify the RVs
            we need for the simulation.

        Raises
        ------
        ValueError
            When any Loss Driver is not recognized.
        """

        RV_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)
        LP = self.loss_params

        # make ds the second level in the MultiIndex
        case_DF = pd.DataFrame(
            index=case_list.reorder_levels([0, 4, 1, 2, 3]), columns=[0, ])
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
                raise ValueError(f"Loss Driver type not recognized: "
                                 f"{driver_type}")

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
                    cost_theta = [cost_params_DS.get(f"Theta_{t_i}", np.nan)
                                  for t_i in range(3)]

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
                    time_theta = [time_params_DS.get(f"Theta_{t_i}", np.nan)
                                  for t_i in range(3)]

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
                            uq.RandomVariable(
                                name=cost_rv_tag,
                                distribution=cost_family,
                                theta=cost_theta,
                                truncation_limits=[0., np.nan]
                            )
                        )
                        rv_count += 1

                    # assign time RV
                    if pd.isna(time_family) is False:
                        time_rv_tag = (
                            f'Time-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(uq.RandomVariable(
                            name=time_rv_tag,
                            distribution=time_family,
                            theta=time_theta,
                            truncation_limits=[0., np.nan]
                        ))
                        rv_count += 1

                    # assign time RV
                    if pd.isna(carbon_family) is False:
                        carbon_rv_tag = (
                            f'Carbon-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(uq.RandomVariable(
                            name=carbon_rv_tag,
                            distribution=carbon_family,
                            theta=carbon_theta,
                            truncation_limits=[0., np.nan]
                        ))
                        rv_count += 1

                    # assign time RV
                    if pd.isna(energy_family) is False:
                        energy_rv_tag = (
                            f'Energy-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}'
                        )

                        RV_reg.add_RV(uq.RandomVariable(
                            name=energy_rv_tag,
                            distribution=energy_family,
                            theta=energy_theta,
                            truncation_limits=[0., np.nan]
                        ))
                        rv_count += 1

                    # assign correlation between RVs across DV_types
                    # TODO: add more DV_types and handle cases with only a
                    # subset of them being defined
                    if ((pd.isna(cost_family) is False) and (
                            pd.isna(time_family) is False) and (
                                self._asmnt.options.rho_cost_time != 0.0)):

                        rho = self._asmnt.options.rho_cost_time

                        RV_reg.add_RV_set(uq.RandomVariableSet(
                            f'DV-{loss_cmp_id}-{ds}-{loc}-{direction}-{uid}_set',
                            list(RV_reg.RVs([cost_rv_tag, time_rv_tag]).values()),
                            np.array([[1.0, rho], [rho, 1.0]])))

        self.log_msg(f"\n{rv_count} random variables created.",
                     prepend_timestamp=False)

        if rv_count > 0:
            return RV_reg
        # else:
        return None

    def _calc_median_consequence(self, eco_qnt):
        """
        Calculate the median repair consequence for each loss component.

        """

        medians = {}

        DV_types = self.loss_params.index.unique(level=1)

        # for DV_type, DV_type_scase in zip(['COST', 'TIME'], ['Cost', 'Time']):
        for DV_type in DV_types:

            cmp_list = []
            median_list = []

            for loss_cmp_id in self.loss_map.index:

                driver_type, driver_cmp = self.loss_map.loc[
                    loss_cmp_id, 'Driver']
                loss_cmp_name = self.loss_map.loc[loss_cmp_id, 'Consequence']

                # check if the given DV type is available as an output for the
                # selected component
                if (loss_cmp_name, DV_type) not in self.loss_params.index:
                    continue

                if driver_type != 'DMG':
                    raise ValueError(f"Loss Driver type not recognized: "
                                     f"{driver_type}")

                if driver_cmp not in eco_qnt.columns.get_level_values(
                        0).unique():
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
                        (loss_cmp_name, DV_type),
                        ds]

                    # check if theta_0 is defined
                    theta_0 = loss_params_DS.get('Theta_0', np.nan)

                    if pd.isna(theta_0):
                        continue

                    # check if the distribution type is supported
                    family = loss_params_DS.get('Family', np.nan)

                    if ((not pd.isna(family)) and (
                        family not in [
                            'normal', 'lognormal', 'deterministic'])):
                        raise ValueError(f"Loss Distribution of type {family} "
                                         f"not supported.")

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
                            dtype=float)
                        medns = all_vals[0]
                        qnts = all_vals[1]
                        f_median = prep_bounded_multilinear_median_DV(
                            medns, qnts)

                    # get the corresponding aggregate damage quantities
                    # to consider economies of scale
                    if 'ds' in eco_qnt.columns.names:

                        avail_ds = (
                            eco_qnt.loc[:, driver_cmp].columns.unique(level=0))

                        if (ds_id not in avail_ds):
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
                    median_list.append(pd.concat(sub_medians, axis=1,
                                                 keys=ds_list))
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

    def aggregate_losses(self):
        """
        Aggregates repair consequences across components.

        Repair costs are simply summed up for each realization while repair
        times are aggregated to provide lower and upper limits of the total
        repair time using the assumption of parallel and sequential repair of
        floors, respectively. Repairs within each floor are assumed to occur
        sequentially.
        """

        self.log_div()
        self.log_msg("Aggregating repair consequences...")

        DV = self.sample

        # group results by DV type and location
        DVG = DV.groupby(level=[0, 4], axis=1).sum()

        # create the summary DF
        df_agg = pd.DataFrame(index=DV.index,
                              columns=['repair_cost',
                                       'repair_time-parallel',
                                       'repair_time-sequential',
                                       'repair_carbon',
                                       'repair_energy'])

        if 'Cost' in DVG.columns:
            df_agg['repair_cost'] = DVG['Cost'].sum(axis=1)
        else:
            df_agg = df_agg.drop('repair_cost', axis=1)

        if 'Time' in DVG.columns:
            df_agg['repair_time-sequential'] = DVG['Time'].sum(axis=1)

            df_agg['repair_time-parallel'] = DVG['Time'].max(axis=1)
        else:
            df_agg = df_agg.drop(['repair_time-parallel',
                                  'repair_time-sequential'],
                                 axis=1)

        if 'Carbon' in DVG.columns:
            df_agg['repair_carbon'] = DVG['Carbon'].sum(axis=1)
        else:
            df_agg = df_agg.drop('repair_carbon', axis=1)

        if 'Energy' in DVG.columns:
            df_agg['repair_energy'] = DVG['Energy'].sum(axis=1)
        else:
            df_agg = df_agg.drop('repair_energy', axis=1)

        # convert units

        cmp_units = self.loss_params[('DV', 'Unit')].groupby(level=[1, ]).agg(
            lambda x: x.value_counts().index[0])

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
            df_agg, None, units=dv_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=False,
            log=self._asmnt.log)

        df_agg.drop("Units", inplace=True)

        # convert header

        df_agg = base.convert_to_MultiIndex(df_agg, axis=1)

        self.log_msg("Repair consequences successfully aggregated.")

        return df_agg.astype(float)

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
        self.log_msg("\nAggregating damage quantities...",
                     prepend_timestamp=False)

        if self._asmnt.options.eco_scale["AcrossFloors"]:

            if self._asmnt.options.eco_scale["AcrossDamageStates"]:

                eco_levels = [0, ]
                eco_columns = ['cmp', ]

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

        self.log_msg("Successfully aggregated damage quantities.",
                     prepend_timestamp=False)

        # apply the median functions, if needed, to get median consequences for
        # each realization
        self.log_msg("\nCalculating the median repair consequences...",
                     prepend_timestamp=False)

        medians = self._calc_median_consequence(eco_qnt)

        self.log_msg("Successfully determined median repair consequences.",
                     prepend_timestamp=False)

        # combine the median consequences with the samples of deviation from the
        # median to get the consequence realizations.
        self.log_msg("\nConsidering deviations from the median values to obtain "
                     "random DV sample...")

        self.log_msg("Preparing random variables for repair cost and time...",
                     prepend_timestamp=False)
        RV_reg = self._create_DV_RVs(dmg_quantities.columns)

        if RV_reg is not None:
            RV_reg.generate_sample(
                sample_size=sample_size, method=self._asmnt.options.sampling_method)

            std_sample = base.convert_to_MultiIndex(
                pd.DataFrame(RV_reg.RV_sample), axis=1).sort_index(axis=1)
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

        self.log_msg(f"\nSuccessfully generated {sample_size} realizations of "
                     "deviation from the median consequences.",
                     prepend_timestamp=False)

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
                    raise ValueError(f"Loss Driver type not "
                                     f"recognized: {driver_type}")

                if not (dmg_cmp_i
                        in dmg_quantities.columns.unique(level=0)):
                    continue

                ds_list = []

                for ds in medians[DV_type].loc[:, cmp_i].columns.unique(level=0):

                    loc_list = []

                    for loc_id, loc in enumerate(
                            dmg_quantities.loc[
                                :, (dmg_cmp_i, ds)].columns.unique(level=0)):

                        if ((self._asmnt.options.eco_scale[
                                "AcrossFloors"] is True) and (
                                loc_id > 0)):
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
                        ds_list += [ds, ]
                    else:
                        ds_list += [(ds, loc) for loc in loc_list]

                if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                    cmp_list += [(loss_cmp_i, dmg_cmp_i, ds) for ds in ds_list]
                else:
                    cmp_list += [
                        (loss_cmp_i, dmg_cmp_i, ds, loc) for ds, loc in ds_list]

            if self._asmnt.options.eco_scale["AcrossFloors"] is True:
                key_list += [(DV_type, loss_cmp_i, dmg_cmp_i, ds)
                             for loss_cmp_i, dmg_cmp_i, ds in cmp_list]
            else:
                key_list += [(DV_type, loss_cmp_i, dmg_cmp_i, ds, loc)
                             for loss_cmp_i, dmg_cmp_i, ds, loc in cmp_list]

        lvl_names = ['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'uid']
        DV_sample = pd.concat(res_list, axis=1, keys=key_list,
                              names=lvl_names)

        DV_sample = DV_sample.fillna(0).convert_dtypes()
        DV_sample.columns.names = lvl_names

        # Get the flags for replacement consequence trigger
        DV_sum = DV_sample.groupby(level=[1, ], axis=1).sum()
        if 'replacement' in DV_sum.columns:

            # When the 'replacement' consequence is triggered, all
            # local repair consequences are discarded. Note that
            # global consequences are assigned to location '0'.

            id_replacement = DV_sum['replacement'] > 0

            # get the list of non-zero locations
            locs = DV_sample.columns.get_level_values(4).unique().values

            locs = locs[locs != '0']

            DV_sample.loc[id_replacement, idx[:, :, :, :, locs]] = 0.0

        self._sample = DV_sample

        self.log_msg("Successfully obtained DV sample.",
                     prepend_timestamp=False)


def prep_constant_median_DV(median):
    """
    Returns a constant median Decision Variable (DV) function.

    Parameters
    ----------
    median: float
        The median DV for a consequence function with fixed median.

    Returns
    -------
    f: callable
        A function that returns the constant median DV for all component
        quantities.
    """
    def f(*args):
        # pylint: disable=unused-argument
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
    f: callable
        A function that returns the median DV given the quantity of damaged
        components.
    """
    def f(quantity):
        if quantity is None:
            raise ValueError(
                'A bounded linear median Decision Variable function called '
                'without specifying the quantity of damaged components')

        q_array = np.asarray(quantity, dtype=np.float64)

        # calculate the median consequence given the quantity of damaged
        # components
        output = np.interp(q_array, quantities, medians)

        return output

    return f
