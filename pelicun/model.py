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

"""
This module has classes and methods that define and access the model used for
loss assessment.

.. rubric:: Contents

.. autosummary::

    prep_constant_median_DV
    prep_bounded_linear_median_DV
    prep_bounded_multilinear_median_DV

    FragilityFunction
    ConsequenceFunction
    DamageState
    DamageStateGroup
    PerformanceGroup
    FragilityGroup

    DemandModel

"""

from .base import log_msg, log_div
from . import base
from . import uq
from . import file_io

from copy import deepcopy
import numpy as np
import pandas as pd

class DemandModel(object):
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

    def __init__(self):

        self.marginal_params = None
        self.correlation = None
        self.empirical_data = None
        self.units = None

        self._RVs = None
        self._sample = None

    @property
    def sample(self):

        if self._sample is None:

            sample = pd.DataFrame(self._RVs.RV_sample)

            sample = base.convert_to_MultiIndex(sample, axis=1)['EDP']

            self._sample = sample

        else:
            sample = self._sample

        return sample

    def save_sample(self, filepath):
        """
        Save demand sample to a csv file

        """

        log_div()
        log_msg(f'Saving demand sample...')

        file_io.save_to_csv(self.sample, filepath, units=self.units)

        log_msg(f'Demand sample successfully saved.', prepend_timestamp=False)

    def load_sample(self, filepath):
        """
        Load demand sample data and parse it.

        Besides parsing the sample, the method also reads and saves the units
        specified for each demand variable. If no units are specified, Standard
        Units are assumed.

        Parameters
        ----------
        filepath: string
            Location of the file with the demand sample.

        """

        def parse_header(raw_header):

            old_MI = raw_header

            # The first number (event_ID) in the demand labels is optional and
            # currently not used. We remove it if it was in the raw data.
            if old_MI.nlevels == 4:

                if base.options.verbose:
                    log_msg(f'Removing event_ID from header...',
                            prepend_timestamp=False)

                new_column_index = np.array(
                    [old_MI.get_level_values(i) for i in range(1, 4)])

            else:
                new_column_index = np.array(
                    [old_MI.get_level_values(i) for i in range(3)])

            # Remove whitespace to avoid ambiguity

            if base.options.verbose:
                log_msg(f'Removing whitespace from header...',
                        prepend_timestamp=False)

            wspace_remove = np.vectorize(lambda name: name.replace(' ', ''))

            new_column_index = wspace_remove(new_column_index)

            # Creating new, cleaned-up header

            new_MI = pd.MultiIndex.from_arrays(
                new_column_index, names=['type', 'loc', 'dir'])

            return new_MI

        log_div()
        log_msg(f'Loading demand data...')

        demand_data, units = file_io.load_from_csv(filepath, return_units=True)

        parsed_data = demand_data.copy()

        # start with cleaning up the header

        parsed_data.columns = parse_header(parsed_data.columns)

        # Remove errors, if needed
        if 'ERROR' in parsed_data.columns.get_level_values(0):

            log_msg(f'Removing errors from the raw data...',
                    prepend_timestamp=False)

            error_list = parsed_data.loc[:,base.idx['ERROR',:,:]].values.astype(bool)

            parsed_data = parsed_data.loc[~error_list, :].copy()
            parsed_data.drop('ERROR', level=0, axis=1, inplace=True)

            log_msg(f"\nBased on the values in the ERROR column, "
                    f"{np.sum(error_list)} demand samples were removed.\n",
                    prepend_timestamp=False)

        self._sample = parsed_data

        log_msg(f'Demand data successfully parsed.', prepend_timestamp=False)

        # parse the index for the units
        units.index = parse_header(units.index)

        self.units = units

        log_msg(f'Demand units successfully parsed.', prepend_timestamp=False)

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

            active_d_types = (
                demand_sample.columns.get_level_values('type').unique())

            if demand_type == 'ALL':
                cols = tuple(active_d_types)

            else:
                cols = []

                for d_type in active_d_types:
                    if d_type.split('_')[0] == demand_type:
                        cols.append(d_type)

                cols = tuple(cols)

            # load the distribution family
            cal_df.loc[base.idx[cols,:,:], 'family'] = settings['DistributionFamily']

            # load the censor limits
            if 'CensorAt' in settings.keys():
                censor_lower, censor_upper = settings['CensorAt']
                cal_df.loc[base.idx[cols,:,:], 'censor_lower'] = censor_lower
                cal_df.loc[base.idx[cols,:,:], 'censor_upper'] = censor_upper

            # load the truncation limits
            if 'TruncateAt' in settings.keys():
                truncate_lower, truncate_upper = settings['TruncateAt']
                cal_df.loc[base.idx[cols,:,:], 'truncate_lower'] = truncate_lower
                cal_df.loc[base.idx[cols,:,:], 'truncate_upper'] = truncate_upper

            # scale the censor and truncation limits, if needed
            scale_factor = base.options.scale_factor(settings.get('Unit', None))

            rows_to_scale = ['censor_lower', 'censor_upper',
                             'truncate_lower', 'truncate_upper']
            cal_df.loc[base.idx[cols,:,:], rows_to_scale] *= scale_factor

            # load the prescribed additional uncertainty
            if 'AddUncertainty' in settings.keys():

                sig_increase = settings['AddUncertainty']

                # scale the sig value if the target distribution family is normal
                if settings['DistributionFamily'] == 'normal':
                    sig_increase *= scale_factor

                cal_df.loc[base.idx[cols,:,:], 'sig_increase'] = sig_increase

        def get_filter_mask(lower_lims, upper_lims):

            demands_of_interest = demand_sample.iloc[:, ~np.isnan(upper_lims)]
            limits_of_interest = upper_lims[~np.isnan(upper_lims)]
            upper_mask = np.all(demands_of_interest < limits_of_interest,
                                axis=1)

            demands_of_interest = demand_sample.iloc[:, ~np.isnan(lower_lims)]
            limits_of_interest = lower_lims[~np.isnan(lower_lims)]
            lower_mask = np.all(demands_of_interest > limits_of_interest,
                                axis=1)

            return np.all([lower_mask, upper_mask], axis=0)

        log_div()
        log_msg('Calibrating demand model...')

        demand_sample = self.sample

        # initialize a DataFrame that contains calibration information
        cal_df = pd.DataFrame(
            columns=['family',
                     'censor_lower', 'censor_upper',
                     'truncate_lower', 'truncate_upper',
                     'sig_increase', 'theta_0', 'theta_1'],
            index=demand_sample.columns,
            dtype=float
            )

        cal_df['family'] = cal_df['family'].astype(str)

        # start by assigning the default option ('ALL') to every demand column
        parse_settings(config['ALL'], 'ALL')

        # then parse the additional settings and make the necessary adjustments
        for demand_type in config.keys():
            if demand_type != 'ALL':
                parse_settings(config[demand_type], demand_type)

        if base.options.verbose:
            log_msg(f"\nCalibration settings successfully parsed:\n"+str(cal_df),
                    prepend_timestamp=False)
        else:
            log_msg(
                f"\nCalibration settings successfully parsed:\n",
                prepend_timestamp=False)

        # save the settings
        model_params = cal_df.copy()

        # Remove the samples outside of censoring limits
        # Currently, non-empirical demands are assumed to have some level of
        # correlation, hence, a censored value in any demand triggers the
        # removal of the entire sample from the population.
        upper_lims = cal_df.loc[:, 'censor_upper'].values
        lower_lims = cal_df.loc[:, 'censor_lower'].values

        if ~np.all(np.isnan(np.array([upper_lims, lower_lims]))):

            censor_mask = get_filter_mask(lower_lims, upper_lims)
            censored_count = np.sum(~censor_mask)

            demand_sample = demand_sample.loc[censor_mask, :]

            log_msg(f"\nBased on the provided censoring limits, "
                    f"{censored_count} samples were censored.",
                    prepend_timestamp=False)
        else:
            censored_count = 0

        # Check if there is any sample outside of truncation limits
        # If yes, that suggests an error either in the samples or the
        # configuration. We handle such errors gracefully: the analysis is not
        # terminated, but we show an error in the log file.
        upper_lims = cal_df.loc[:, 'truncate_upper'].values
        lower_lims = cal_df.loc[:, 'truncate_lower'].values

        if ~np.all(np.isnan(np.array([upper_lims, lower_lims]))):

            truncate_mask = get_filter_mask(lower_lims, upper_lims)
            truncated_count = np.sum(~truncate_mask)

            if truncated_count > 0:

                demand_sample = demand_sample.loc[truncate_mask, :]

                log_msg(f"\nBased on the provided truncation limits, "
                        f"{truncated_count} samples were removed before demand "
                        f"calibration.",
                        prepend_timestamp=False)

        # Separate and save the demands that are kept empirical -> i.e., no
        # fitting. Currently, empirical demands are decoupled from those that
        # have a distribution fit to their samples. The correlation between
        # empirical and other demands is not preserved in the demand model.
        empirical_edps = []
        for edp in cal_df.index:
            if cal_df.loc[edp, 'family'] == 'empirical':
                empirical_edps.append(edp)

        self.empirical_data = demand_sample.loc[:, empirical_edps].copy()

        # remove the empirical demands from the samples used for calibration
        demand_sample = demand_sample.drop(empirical_edps, 1)

        # and the calibration settings
        cal_df = cal_df.drop(empirical_edps, 0)

        if base.options.verbose:
            log_msg(f"\nDemand data used for calibration:\n"+str(demand_sample),
                    prepend_timestamp=False)

        # fit the joint distribution
        log_msg(f"\nFitting the prescribed joint demand distribution...",
                prepend_timestamp=False)

        demand_theta, demand_rho = uq.fit_distribution_to_sample(
            raw_samples = demand_sample.values.T,
            distribution = cal_df.loc[:, 'family'].values,
            censored_count = censored_count,
            detection_limits = cal_df.loc[:,
                               ['censor_lower', 'censor_upper']].values.T,
            truncation_limits = cal_df.loc[:,
                                ['truncate_lower', 'truncate_upper']].values.T,
            multi_fit=False
        )

        # fit the joint distribution
        log_msg(f"\nCalibration successful, processing results...",
                prepend_timestamp=False)

        # save the calibration results
        model_params.loc[cal_df.index, ['theta_0','theta_1']] = demand_theta

        # increase the variance of the marginal distributions, if needed
        if ~np.all(np.isnan(model_params.loc[:, 'sig_increase'].values)):

            log_msg(f"\nIncreasing demand variance...",
                    prepend_timestamp=False)

            sig_inc = np.nan_to_num(model_params.loc[:, 'sig_increase'].values)
            sig_0 = model_params.loc[:, 'theta_1'].values

            model_params.loc[:, 'theta_1'] = (
                np.sqrt(sig_0 ** 2. + sig_inc ** 2.))

        # remove unneeded fields from model_params
        for col in ['sig_increase', 'censor_lower', 'censor_upper']:
            model_params = model_params.drop(col, 1)

        # reorder the remaining fields for clarity
        model_params = model_params[[
            'family','theta_0','theta_1','truncate_lower','truncate_upper']]

        self.marginal_params = model_params

        log_msg(f"\nCalibrated demand model marginal distributions:\n" +
                str(model_params),
                prepend_timestamp=False)

        # save the correlation matrix
        self.correlation = pd.DataFrame(demand_rho,
                                      columns = cal_df.index,
                                      index = cal_df.index)

        log_msg(f"\nCalibrated demand model correlation matrix:\n" +
                str(self.correlation),
                prepend_timestamp=False)

    def save_model(self, file_prefix):
        """
        Save parameters of the demand model to a set of csv files

        """

        log_div()
        log_msg(f'Saving demand model...')

        # save the correlation and empirical data
        file_io.save_to_csv(self.correlation, file_prefix + '_correlation.csv')
        file_io.save_to_csv(self.empirical_data, file_prefix + '_empirical.csv',
                    units=self.units)

        # the log standard deviations in the marginal parameters need to be
        # scaled up before feeding to the saving method where they will be
        # scaled back down and end up being saved unscaled to the target file

        marginal_params = self.marginal_params.copy()

        log_rows = marginal_params['family']=='lognormal'
        log_demands = marginal_params.loc[log_rows,:]

        for label in log_demands.index:

            if label in self.units.index:

                unit_factor = base.UC[self.units[label]]

                marginal_params.loc[label, 'theta_1'] *= unit_factor

        file_io.save_to_csv(marginal_params, file_prefix+'_marginals.csv',
                    units=self.units, orientation=1)

        log_msg(f'Demand model successfully saved.', prepend_timestamp=False)

    def load_model(self, file_prefix):

        log_div()
        log_msg(f'Loading demand model...')

        self.empirical_data = file_io.load_from_csv(file_prefix+'_empirical.csv')
        self.empirical_data.columns.set_names(['type', 'loc', 'dir'],
                                            inplace=True)

        self.correlation = file_io.load_from_csv(file_prefix + '_correlation.csv',
                                         reindex=False)
        self.correlation.index.set_names(['type', 'loc', 'dir'], inplace=True)
        self.correlation.columns.set_names(['type', 'loc', 'dir'], inplace=True)

        # the log standard deviations in the marginal parameters need to be
        # adjusted after getting the data from the loading method where they
        # were scaled according to the units of the corresponding variable
        marginal_params, units = file_io.load_from_csv(file_prefix + '_marginals.csv',
                                               orientation=1, reindex=False,
                                               return_units=True)
        marginal_params.index.set_names(['type', 'loc', 'dir'],inplace=True)

        log_rows = marginal_params.loc[:, 'family'] == 'lognormal'
        log_demands = marginal_params.loc[log_rows,:].index.values

        for label in log_demands:

            if label in units.index:

                unit_factor = base.UC[units[label]]

                marginal_params.loc[label, 'theta_1'] /= unit_factor

        self.units = units
        self.marginal_params = marginal_params

        log_msg(f'Demand model successfully loaded.', prepend_timestamp=False)

    def _create_RVs(self, preserve_order=False):
        """
        Create a random variable registry for the joint distribution of demands.

        """

        # initialize the registry
        RV_reg = uq.RandomVariableRegistry()

        # add a random variable for each demand variable
        for rv_params in self.marginal_params.itertuples():

            edp = rv_params.Index
            rv_tag = f'EDP-{edp[0]}-{edp[1]}-{edp[2]}'

            if rv_params.family == 'empirical':

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
                    distribution=rv_params.family,
                    theta=[rv_params.theta_0, rv_params.theta_1],
                    truncation_limits=[rv_params.truncate_lower,
                                       rv_params.truncate_upper],


                ))

        log_msg(f"\n{self.marginal_params.shape[0]} random variables created.",
                prepend_timestamp=False)

        # add an RV set to consider the correlation between demands
        rv_set_tags = [f'EDP-{edp[0]}-{edp[1]}-{edp[2]}'
                       for edp in self.correlation.index.values]

        RV_reg.add_RV_set(uq.RandomVariableSet(
            'EDP_set', list(RV_reg.RVs(rv_set_tags).values()),
            self.correlation.values))

        log_msg(f"\nCorrelations between {len(rv_set_tags)} random variables "
                f"successfully defined.",
                prepend_timestamp=False)

        self._RVs = RV_reg

    def generate_sample(self, config):

        if self.marginal_params is None:
            raise ValueError('Model parameters have not been specified. Either'
                             'load parameters from a file or calibrate the '
                             'model using raw demand data.')

        log_div()
        log_msg(f'Generating sample from demand variables...')

        self._create_RVs(
            preserve_order=config.get('PreserveRawOrder', False))

        sample_size = config['SampleSize']
        self._RVs.generate_sample(sample_size=sample_size)

        # replace the potentially existing raw sample with the generated one
        self._sample = None

        log_msg(f"\nSuccessfully generated {sample_size} realizations.",
                prepend_timestamp=False)

class AssetModel(object):
    """
    Manages asset information used in assessments.

    Parameters
    ----------

    """

    def __init__(self, assessment):

        self._asmnt = assessment

        self.cmp_marginal_params = None
        self.cmp_units = None

        self._cmp_RVs = None
        self._cmp_sample = None

    @property
    def cmp_sample(self):

        if self._cmp_sample is None:

            cmp_sample = pd.DataFrame(self._cmp_RVs.RV_sample)

            cmp_sample = base.convert_to_MultiIndex(cmp_sample, axis=1)['CMP']

            self._cmp_sample = cmp_sample

        else:
            cmp_sample = self._cmp_sample

        return cmp_sample

    def save_cmp_sample(self, filepath):
        """
        Save component quantity sample to a csv file

        """

        log_div()
        log_msg(f'Saving asset components sample...')

        # prepare a units array
        sample = self.cmp_sample

        units = pd.Series(name='units', index=sample.columns)

        for cmp_id, unit_name in self.cmp_units.items():
            units.loc[cmp_id, :] = unit_name

        file_io.save_to_csv(sample, filepath, units=units)

        log_msg(f'Asset components sample successfully saved.',
                prepend_timestamp=False)

    def load_cmp_sample(self, filepath):
        """
        Load component quantity sample from a csv file

        """

        log_div()
        log_msg(f'Loading asset components sample...')

        sample, units = file_io.load_from_csv(filepath, return_units=True)

        self._cmp_sample = sample

        self.cmp_units = units.groupby(level=0).first()

        log_msg(f'Asset components sample successfully loaded.',
                prepend_timestamp=False)

    def load_cmp_model(self, file_prefix):
        """
        Load the model that describes component quantities in the building.

        """

        def get_locations(loc_str):

            try:
                res = int(loc_str)
                return np.array([res, ])

            except:
                stories = self._asmnt.stories

                if "-" in loc_str:
                    s_low, s_high = loc_str.split('-')
                    s_low = get_locations(s_low)
                    s_high = get_locations(s_high)
                    return np.arange(s_low[0], s_high[0] + 1)

                elif "," in loc_str:
                    return np.array(loc_str.split(','), dtype=int)

                elif loc_str == "all":
                    return np.arange(1, stories + 1)

                elif loc_str == "top":
                    return np.array([stories, ])

                elif loc_str == "roof":
                    return np.array([stories, ])

                else:
                    raise ValueError(f"Cannot parse location string: "
                                     f"{loc_str}")

        def get_directions(dir_str):

            if pd.isnull(dir_str):
                return np.ones(1)

            else:

                try:
                    res = int(dir_str)
                    return np.array([res, ])

                except:

                    if "," in dir_str:
                        return np.array(dir_str.split(','), dtype=int)

                    elif "-" in dir_str:
                        d_low, d_high = dir_str.split('-')
                        d_low = get_directions(d_low)
                        d_high = get_directions(d_high)
                        return np.arange(d_low[0], d_high[0] + 1)

                    else:
                        raise ValueError(f"Cannot parse direction string: "
                                         f"{dir_str}")

        def get_blocks(block_str):

            if pd.isnull(block_str):
                return np.ones(1) * np.nan

            else:

                try:
                    res = float(block_str)
                    return np.array([res, ])

                except:

                    if "," in block_str:
                        return np.array(block_str.split(','), dtype=int)

                    else:
                        raise ValueError(f"Cannot parse theta_0 string: "
                                         f"{block_str}")

        # Currently, we assume independent component distributions are defined
        # throughout the building. Correlations may be added afterward or this
        # method can be extended to read correlation matrices too if needed.
        marginal_params, units = file_io.load_from_csv(
            file_prefix + '_marginals.csv',
            orientation=1,
            reindex=False,
            return_units=True,
            convert=[])

        self.cmp_units = units.copy()

        marginal_params = pd.concat([marginal_params, units], axis=1)

        # First, we need to expand the table to have unique component blocks in
        # each row

        log_msg(f"\nParsing model file to characterize each component block",
                prepend_timestamp=False)

        # Create a multiindex that identifies individual component blocks
        MI_list = []
        for row in marginal_params.itertuples():
            locs = get_locations(row.location)
            dirs = get_directions(row.direction)
            blocks = range(1, len(get_blocks(row.theta_0)) + 1)

            MI_list.append(pd.MultiIndex.from_product(
                [[row.Index, ], locs, dirs, blocks],
                names=['cmp', 'loc', 'dir', 'block']))

        MI = MI_list[0].append(MI_list[1:])

        # Create a DataFrame that will hold marginal params for component blocks
        marginal_cols = ['units', 'family', 'theta_0', 'theta_1',
                         'truncate_lower', 'truncate_upper']
        cmp_marginal_params = pd.DataFrame(
            columns=marginal_cols,
            index=MI,
            dtype=float
        )
        cmp_marginal_params[['units', 'family']] = (
            cmp_marginal_params[['units', 'family']].astype(object))

        # Fill the DataFrame with information on component quantity variables

        for row in marginal_params.itertuples():

            locs = get_locations(row.location)
            dirs = get_directions(row.direction)
            theta_0s = get_blocks(row.theta_0)
            theta_1s = get_blocks(row.theta_1)
            trnc_ls = get_blocks(row.truncate_lower)
            trnc_us = get_blocks(row.truncate_upper)

            # parse the distribution characteristics
            if len(theta_1s) != len(theta_0s):

                if len(theta_1s) == 1:
                    theta_1s = theta_1s[0] * np.ones(theta_0s.shape)

                else:
                    raise ValueError(f"Unable to parse theta_1 string: "
                                     f"{row.theta_1}")

            if len(trnc_ls) != len(theta_0s):

                if len(trnc_ls) == 1:
                    trnc_ls = trnc_ls[0] * np.ones(theta_0s.shape)

                else:
                    raise ValueError(f"Unable to parse theta_1 string: "
                                     f"{row.truncate_lower}")

            if len(trnc_us) != len(theta_0s):

                if len(trnc_us) == 1:
                    trnc_us = trnc_us[0] * np.ones(theta_0s.shape)

                else:
                    raise ValueError(f"Unable to parse theta_1 string: "
                                     f"{row.truncate_upper}")

            blocks = range(1, len(theta_0s) + 1)

            for block in blocks:
                MI = pd.MultiIndex.from_product(
                    [[row.Index, ], locs, dirs, [block, ]],
                    names=['cmp', 'loc', 'dir', 'block'])

                cmp_marginal_params.loc[MI, marginal_cols] = [
                    row.units, row.family, theta_0s[block - 1],
                    theta_1s[block - 1],
                    trnc_ls[block - 1], trnc_us[block - 1]]

        log_msg(f"Model parameters successfully parsed. "
                f"{cmp_marginal_params.shape[0]} component blocks identified",
                prepend_timestamp=False)

        # Now we can take care of converting the values to SI units
        log_msg(f"Converting model parameters to internal units...",
                prepend_timestamp=False)

        unique_units = cmp_marginal_params['units'].unique()

        for unit_name in unique_units:

            try:
                unit_factor = base.UC[unit_name]

            except:
                raise ValueError(f"Specified unit name not recognized: "
                                 f"{unit_name}")

            unit_ids = cmp_marginal_params.loc[
                cmp_marginal_params['units'] == unit_name].index

            cmp_marginal_params.loc[
                unit_ids,
                ['theta_0', 'truncate_lower', 'truncate_upper']] *= unit_factor

            sigma_ids = cmp_marginal_params.loc[unit_ids].loc[
                cmp_marginal_params.loc[unit_ids, 'family'] == 'normal'].index

            cmp_marginal_params.loc[sigma_ids, 'theta_1'] *= unit_factor

        self.cmp_marginal_params = cmp_marginal_params

        log_msg(f"Model parameters successfully loaded.",
                prepend_timestamp=False)

        log_msg(f"\nComponent model marginal distributions:\n" +
                str(cmp_marginal_params),
                prepend_timestamp=False)

        # the empirical data and correlation files can be added later, if needed

    def _create_cmp_RVs(self):

        # initialize the registry
        RV_reg = uq.RandomVariableRegistry()

        # add a random variable for each component quantity variable
        for rv_params in self.cmp_marginal_params.itertuples():

            cmp = rv_params.Index
            rv_tag = f'CMP-{cmp[0]}-{cmp[1]}-{cmp[2]}-{cmp[3]}'

            if pd.isnull(rv_params.family):

                # we use an empirical RV to generate deterministic values
                RV_reg.add_RV(uq.RandomVariable(
                    name=rv_tag,
                    distribution='empirical',
                    raw_samples=np.ones(10000) * rv_params.theta_0
                ))

            else:

                # all other RVs need parameters of their distributions
                RV_reg.add_RV(uq.RandomVariable(
                    name=rv_tag,
                    distribution=rv_params.family,
                    theta=[rv_params.theta_0, rv_params.theta_1],
                    truncation_limits=[rv_params.truncate_lower,
                                       rv_params.truncate_upper],

                ))

        log_msg(f"\n{self.cmp_marginal_params.shape[0]} "
                f"random variables created.",
                prepend_timestamp=False)

        self._cmp_RVs = RV_reg

    def generate_cmp_sample(self, sample_size):

        if self.cmp_marginal_params is None:
            raise ValueError('Model parameters have not been specified. Load'
                             'parameters from a file before generating a '
                             'sample.')

        log_div()
        log_msg(f'Generating sample from component quantity variables...')

        self._create_cmp_RVs()

        self._cmp_RVs.generate_sample(sample_size=sample_size)

        # replace the potentially existing sample with the generated one
        self._cmp_sample = None

        log_msg(f"\nSuccessfully generated {sample_size} realizations.",
                prepend_timestamp=False)

class DamageModel(object):
    """
    Manages damage information used in assessments.

    Parameters
    ----------

    """

    def __init__(self, assessment):

        self._asmnt = assessment

        self.fragility_params = None

        self._frg_RVs = None
        self._frg_sample = None
        self._lsds_RVs = None
        self._lsds_sample = None

        self._sample = None

    @property
    def frg_sample(self):

        if self._frg_sample is None:

            frg_sample = pd.DataFrame(self._frg_RVs.RV_sample)

            frg_sample = base.convert_to_MultiIndex(frg_sample, axis=1)['FRG']

            self._frg_sample = frg_sample

        else:
            frg_sample = self._frg_sample

        return frg_sample

    @property
    def lsds_sample(self):

        if self._lsds_sample is None:

            lsds_sample = pd.DataFrame(self._lsds_RVs.RV_sample)

            lsds_sample = base.convert_to_MultiIndex(lsds_sample, axis=1)['LSDS']

            lsds_sample = lsds_sample.astype(int)

            self._lsds_sample = lsds_sample

        else:
            lsds_sample = self._lsds_sample

        return lsds_sample

    @property
    def sample(self):

        return self._sample

    def save_sample(self, filepath):
        """
        Save damage sample to a csv file

        """
        log_div()
        log_msg(f'Saving damage sample...')

        file_io.save_to_csv(self.sample, filepath)

        log_msg(f'Damage sample successfully saved.', prepend_timestamp=False)

    def load_sample(self, filepath):
        """
        Load damage sample data.

        """
        log_div()
        log_msg(f'Loading damage sample...')

        self._sample = file_io.load_from_csv(filepath)

        log_msg(f'Damage sample successfully loaded.', prepend_timestamp=False)

    def load_fragility_model(self, data_paths):
        """
        Load limit state fragility functions and damage state assignments

        Parameters
        ----------
        data_paths: list of string
            List of paths to data files with fragility information. Default
            XY datasets can be accessed as PelicunDefault/XY.
        """

        log_div()
        log_msg(f'Loading fragility model...')

        # replace default flag with default data path
        for d_i, data_path in enumerate(data_paths):

            if 'PelicunDefault/' in data_path:
                data_paths[d_i] = data_path.replace('PelicunDefault/',
                                                   str(base.pelicun_path)+
                                                    '/resources/')

        data_list = []
        # load the data files one by one
        for data_path in data_paths:

            data = file_io.load_from_csv(
                data_path,
                orientation=1,
                reindex=False,
                convert=[]
            )

            data_list.append(data)

        fragility_params = pd.concat(data_list, axis=0)

        # drop redefinitions of components
        fragility_params = fragility_params.groupby(fragility_params.index).first()

        # get the component types defined in the asset model
        cmp_labels = self._asmnt.asset.cmp_sample.columns

        # only keep the fragility parameters for the components in the model
        cmp_unique = cmp_labels.unique(level=0)
        fragility_params = fragility_params.loc[cmp_unique, :]

        # now convert the units - where needed
        unique_units = fragility_params[('Demand', 'Unit')].unique()

        for unit_name in unique_units:

            try:
                unit_factor = base.UC[unit_name]

            except:
                raise ValueError(f"Specified unit name not recognized: "
                                 f"{unit_name}")

            unit_ids = fragility_params.loc[
                fragility_params[('Demand', 'Unit')] == unit_name].index

            for LS_i in fragility_params.columns.unique(level=0):

                if 'LS' in LS_i:

                    # theta_0 needs to be scaled for both all families
                    fragility_params.loc[
                        unit_ids, (LS_i, 'Theta_0')] *= unit_factor

                    # theta_1 needs to be scaled for normal
                    sigma_ids = fragility_params.loc[unit_ids].loc[
                        fragility_params.loc[unit_ids, (LS_i, 'Family')] == 'normal'].index

                    # theta_1 needs to be scaled for uniform
                    sigma_ids = fragility_params.loc[unit_ids].loc[
                        fragility_params.loc[
                            unit_ids, (LS_i, 'Family')] == 'uniform'].index

                    fragility_params.loc[
                        sigma_ids, (LS_i, 'Theta_1')] *= unit_factor

        # check for components with incomplete fragility information
        cmp_incomplete_list = fragility_params.loc[
            fragility_params[('Incomplete','')]==1].index

        fragility_params.drop(cmp_incomplete_list, inplace=True)

        log_msg(f"\nWARNING: Fragility information is incomplete for the "
                f"following component(s) {cmp_incomplete_list}. They were "
                f"removed from the analysis.\n",
                prepend_timestamp=False)

        self.fragility_params = fragility_params

        log_msg(f"Fragility parameters successfully parsed.",
                prepend_timestamp=False)

    def _create_frg_RVs(self):

        # initialize the registry
        frg_RV_reg = uq.RandomVariableRegistry()
        lsds_RV_reg = uq.RandomVariableRegistry()

        rv_count = 0

        # get the component types defined in the asset model
        cmp_labels = self._asmnt.asset.cmp_sample.columns

        for label in cmp_labels:

            cmp_id = label[0]

            if cmp_id in self.fragility_params.index:

                frg_params = self.fragility_params.loc[cmp_id,:]

                limit_states = []
                [limit_states.append(val[2:]) if 'LS' in val else val
                for val in frg_params.index.get_level_values(0).unique()]

                ds_id = 0

                frg_rv_set_tags = []
                for ls_id in limit_states:

                    theta_0 = frg_params.loc[(f'LS{ls_id}','Theta_0')]

                    # check if the limit state is defined for the component
                    if ~np.isnan(theta_0):

                        frg_rv_tag = f'FRG-{label[0]}-{label[1]}-{label[2]}-{label[3]}-{ls_id}'
                        lsds_rv_tag = f'LSDS-{label[0]}-{label[1]}-{label[2]}-{label[3]}-{ls_id}'

                        family, theta_1, ds_weights = frg_params.loc[
                            [(f'LS{ls_id}','Family'),
                             (f'LS{ls_id}','Theta_1'),
                             (f'LS{ls_id}','DamageStateWeights')]]

                        # Start with the limit state capacities...
                        # if the limit state is deterministic, we use an
                        # empirical RV
                        if pd.isnull(family):

                            frg_RV_reg.add_RV(uq.RandomVariable(
                                name=frg_rv_tag,
                                distribution='empirical',
                                raw_samples=np.ones(10000) * theta_0
                            ))

                        else:

                            # all other RVs have parameters of their distributions
                            frg_RV_reg.add_RV(uq.RandomVariable(
                                name=frg_rv_tag,
                                distribution=family,
                                theta=[theta_0, theta_1],
                            ))

                        # add the RV to the set of correlated variables
                        frg_rv_set_tags.append(frg_rv_tag)

                        # Now add the LS->DS assignments
                        # if the limit state has a single damage state assigned
                        # to it, we don't need random sampling
                        if pd.isnull(ds_weights):

                            ds_id += 1

                            lsds_RV_reg.add_RV(uq.RandomVariable(
                                name=lsds_rv_tag,
                                distribution='empirical',
                                raw_samples=np.ones(10000) * ds_id
                            ))

                        else:

                            # parse the DS weights
                            ds_weights = np.array(
                                ds_weights.replace(" ", "").split('|'),
                                dtype=float)

                            def map_ds(values, offset=int(ds_id+1)):
                                return values+offset

                            lsds_RV_reg.add_RV(uq.RandomVariable(
                                name=lsds_rv_tag,
                                distribution='multinomial',
                                theta=ds_weights,
                                f_map=map_ds
                            ))

                            ds_id += len(ds_weights)

                        rv_count += 1

                # Assign correlation between limit state random variables
                # Note that we assume perfectly correlated limit state random
                # variables here. This approach is in line with how mainstream
                # PBE calculations are performed.
                # Assigning more sophisticated correlations between limit state
                # RVs is possible, if needed. Please let us know through the
                # SimCenter Message Board if you are interested in such a
                # feature.
                frg_RV_reg.add_RV_set(uq.RandomVariableSet(
                    f'FRG-{label[0]}-{label[1]}-{label[2]}-{label[3]}_set',
                    list(frg_RV_reg.RVs(frg_rv_set_tags).values()),
                    np.ones((len(frg_rv_set_tags),len(frg_rv_set_tags)))))

        log_msg(f"\n2x{rv_count} random variables created.",
                prepend_timestamp=False)

        self._frg_RVs = frg_RV_reg
        self._lsds_RVs = lsds_RV_reg

    def _generate_frg_sample(self, sample_size):

        if self.fragility_params is None:
            raise ValueError('Fragility parameters have not been specified. '
                             'Load parameters from the default fragility '
                             'databases or provide your own fragility '
                             'definitions before generating a sample.')

        log_msg(f'Generating sample from fragility variables...',
                prepend_timestamp=False)

        self._create_frg_RVs()

        self._frg_RVs.generate_sample(sample_size=sample_size)

        self._lsds_RVs.generate_sample(sample_size=sample_size)

        # replace the potentially existing raw sample with the generated one
        self._frg_sample = None
        self._lsds_sample = None

        log_msg(f"\nSuccessfully generated {sample_size} realizations.",
                prepend_timestamp=False)

    def get_required_demand_types(self):
        """
        Returns a list of demands needed to calculate damage to components

        Note that we assume that a fragility sample is available.

        """

        log_msg(f'Collecting required demand information...',
                prepend_timestamp=False)

        EDP_req = pd.Series(
            index=self.frg_sample.groupby(level=[0,1,2], axis=1).first().columns)

        # get the list of  active components
        cmp_list = EDP_req.index.get_level_values(0).unique()

        # for each component
        for cmp in cmp_list:

            # get the parameters from the fragility db
            directional, offset, demand_type = self.fragility_params.loc[
                cmp, [('Demand', 'Directional'),
                      ('Demand', 'Offset'),
                      ('Demand', 'Type')]]

            if directional:

                # respect both the location and direction for directional
                # components
                locations, directions = [
                    EDP_req[cmp].groupby(level=[0, 1]).first(
                    ).index.get_level_values(lvl).values.astype(int)
                    for lvl in range(2)]

            else:

                # only store the location and use 0 as direction for the
                # non-directional components
                locations = EDP_req[cmp].index.get_level_values(0).unique(
                ).values.astype(int)
                directions = np.zeros(locations.shape, dtype=int)

            # parse the demand type

            # first check if there is a subtype included
            if '|' in demand_type:
                demand_type, subtype = demand_type.split('|')
                demand_type = base.EDP_to_demand_type[demand_type]
                EDP_type = f'{demand_type}_{subtype}'
            else:
                demand_type = base.EDP_to_demand_type[demand_type]
                EDP_type = demand_type

            # consider the default offset, if needed
            if demand_type in base.options.demand_offset.keys():

                offset = int(offset + base.options.demand_offset[demand_type])

            else:
                offset = int(offset)

            # add the required EDPs to the list
            EDP_req[cmp] = [f'{EDP_type}-{loc+offset}-{dir}'
                            for loc, dir in zip(locations, directions)]

        # return the unique EDP requirements
        return EDP_req

    def _assemble_required_demand_data(self, EDP_req):

        log_msg(f'Assembling demand data for calculation...',
                prepend_timestamp=False)

        demands = pd.DataFrame(columns=EDP_req, index=self.frg_sample.index)
        demands = base.convert_to_MultiIndex(demands, axis=1)

        demand_source = self._asmnt.demand.sample

        # And fill it with data
        for col in demands.columns:

            # if non-directional demand is requested...
            if col[2] == '0':

                # check if the demand at the given location is available
                available_demands = demand_source.groupby(
                    level=[0, 1], axis=1).first().columns

                if col[:2] in available_demands:
                    # take the maximum of all available directions and scale it
                    # using the nondirectional multiplier specified in the
                    # options (the default value is 1.2)
                    demands[col] = demand_source.loc[:, (col[0], col[1])].max(
                        axis=1) * base.options.nondir_multi(col[0])

            elif col in demand_source.columns:

                demands[col] = demand_source[col]

        # Report missing demand data
        for col in demands.columns[np.all(demands.isna(), axis=0)]:
            log_msg(f'\nWARNING: Cannot find demand data for {col}. The '
                    f'corresponding damages cannot be calculated.',
                    prepend_timestamp=False)

        demands.dropna(axis=1, how='all', inplace=True)

        return demands

    def _evaluate_damage(self, CMP_to_EDP, demands):
        """
        Use the provided demands and the LS capacity sample the evaluate damage

        Parameters
        ----------
        CMP_to_EDP: Series
            Identifies the EDP assigned to each component
        demands: DataFrame
            Provides a sample of the demands required (and available) for the
            damage assessment.

        Returns
        -------
        dmg_sample: DataFrame
            Assigns a Damage State to each component block in the asset model.
        """
        dmg_eval = pd.DataFrame(columns=self.frg_sample.columns,
                                index=self.frg_sample.index)

        # for each available demand
        for demand in demands.columns:

            # get the corresponding component - loc - dir
            cmp_affected = CMP_to_EDP[
                CMP_to_EDP == '-'.join(demand)].index.values

            full_cmp_list = []

            # assemble a list of cmp - loc - dir - block - ls
            for cmp in cmp_affected:
                vals = dmg_eval.loc[:, cmp].columns.values

                full_cmp_list += [cmp + val for val in vals]

            # evaluate ls exceedance for each case
            dmg_eval.loc[:, full_cmp_list] = (
                    self.frg_sample.loc[:, full_cmp_list].sub(
                        demands[demand], axis=0) < 0)

        # drop the columns that do not have valid results
        dmg_eval.dropna(how='all', axis=1, inplace=True)

        # initialize the DataFrame that stores the damage states
        cmp_sample = self._asmnt.asset.cmp_sample
        dmg_sample = pd.DataFrame(np.zeros(cmp_sample.shape),
                                  columns=cmp_sample.columns,
                                  index=cmp_sample.index, dtype=int)

        # get a list of limit state ids among all components in the damage model
        ls_list = dmg_eval.columns.get_level_values(4).unique()

        # for each consecutive limit state...
        for LS_id in ls_list:
            # get all cmp - loc - dir - block where this limit state occurs
            dmg_e_ls = dmg_eval.loc[:, base.idx[:, :, :, :, LS_id]].dropna(axis=1)

            # Get the damage states corresponding to this limit state in each
            # block
            # Note that limit states with a set of mutually exclusive damage
            # states options have their damage state picked here.
            lsds = self.lsds_sample.loc[:, dmg_e_ls.columns]

            # Drop the limit state level from the columns to make the damage
            # exceedance DataFrame compatible with the other DataFrames in the
            # following steps
            dmg_e_ls.columns = dmg_e_ls.columns.droplevel(4)

            # Same thing for the lsds DataFrame
            lsds.columns = dmg_e_ls.columns

            # Update the damage state in the result with the values from the
            # lsds DF if the limit state was exceeded according to the
            # dmg_e_ls DF.
            # This one-liner updates the given Limit State exceedance in the
            # entire damage model. If subsequent Limit States are also exceeded,
            # those cells in the result matrix will get overwritten by higher
            # damage states.
            dmg_sample.loc[:, dmg_e_ls.columns] = (
                dmg_sample.loc[:, dmg_e_ls.columns].mask(dmg_e_ls, lsds))

        return dmg_sample


    def _perform_dmg_task(self, task):
        """
        Perform a task from a damage process.

        """
        cmp_list = self.sample.columns.get_level_values(0).unique().tolist()

        # get the source component
        source_cmp = task[0].split('_')[1]

        # check if it exists
        if source_cmp not in cmp_list:
            raise ValueError(f"source component not found among components in "
                             f"the damage sample: {source_cmp}")

        source_cmp_df = self.sample.loc[:,source_cmp]

        for source_event, target_infos in task[1].items():

            if source_event.startswith('LS'):

                ls_i = int(source_event[2:])
                # TODO: implement source LS support

            elif source_event.startswith('DS'):

                ds_list = [int(source_event[2:]),]

            else:
                raise ValueError(f"Unable to parse source event in damage "
                                 f"process: {source_event}")

            if len(ds_list) == 1:

                source_mask = source_cmp_df.loc[source_cmp_df.values == ds_list[0]].index

            else:
                pass # TODO: implement multiple DS support

            target_infos = np.atleast_1d(target_infos)

            for target_info in target_infos:

                target_cmp, target_event = target_info.split('_')

                if target_cmp == 'ALL':

                    target_cmp = deepcopy(cmp_list)

                    if source_cmp in target_cmp:
                        target_cmp.remove(source_cmp)

                if target_event.startswith('LS'):

                    ls_i = int(target_event[2:])

                    # TODO: implement target LS support

                elif target_event.startswith('DS'):

                    ds_i = int(target_event[2:])

                elif target_event == 'NA':

                    ds_i = None

                else:
                    raise ValueError(f"Unable to parse target event in damage "
                                     f"process: {target_event}")

                if ds_i is None:

                    self._sample.loc[source_mask, target_cmp] = np.nan

                else:
                    self._sample.loc[source_mask, target_cmp] = ds_i


    def calculate(self, sample_size, dmg_process=None):
        """
        Calculate the damage state of each component block in the asset.

        """

        log_div()
        log_msg(f'Calculating damages...')

        # Generate an array with component capacities for each block and
        # generate a second array that assigns a specific damage state to
        # each component limit state. The latter is primarily needed to handle
        # limit states with multiple, mutually exclusive DS options
        self._generate_frg_sample(sample_size)

        # Get the required demand types for the analysis
        EDP_req = self.get_required_demand_types()

        # Create the table of demands
        demands = self._assemble_required_demand_data(EDP_req.unique())

        # Evaluate the Damage State of each Component Block
        dmg_sample = self._evaluate_damage(EDP_req, demands)

        self._sample = dmg_sample

        # Finally, apply the damage prescribed damage process, if any
        if dmg_process is not None:

            for task in dmg_process.items():

                self._perform_dmg_task(task)

        log_msg(f'Damage calculation successfully completed.')

    def prepare_dmg_quantities(self, cmp_list='ALL', dropzero=True,
                               dropempty=True):
        """
        Combine component quantity and damage state information in one DF.

        This method assumes that a component quantity sample is available in
        the asset model and a damage state sample is available here in the
        damage model.

        Parameters
        ----------
        cmp_list: list of strings, optional, default: "ALL"
            The method will return damage results for these components. Choosing
            "ALL" will return damage results for all available components.
        dropzero: bool, optional, default: True
            If True, the quantity of non-damaged components is not saved.
        dropempty: bool, optional, default: True

        """

        dmg = self.sample
        cmp = self._asmnt.asset.cmp_sample

        cmp_list = np.atleast_1d(cmp_list)

        # load the list of all components, if needed
        if cmp_list[0] == 'ALL':
            cmp_list = dmg.columns.get_level_values(0).unique()

        res = []
        cmp_included = []
        # perform the combination for each requested component
        for cmp_id in cmp_list:

            # get the corresponding parts of the quantity and damage matrices
            cmp_i = cmp.loc[:, cmp_id]
            dmg_i = dmg.loc[:, cmp_id]

            # get the realized Damage States
            # Note that these might be much fewer than all possible Damage
            # States
            ds_list = np.unique(dmg_i.values)
            ds_list = np.array(ds_list[~np.isnan(ds_list)], dtype=int)

            # If requested, drop the zero damage case
            if dropzero:
                ds_list = ds_list[ds_list != 0]

            # only perform this if there is at least one DS we are interested in
            if len(ds_list) > 0:

                # initialize the shell for the result DF
                dmg_q = pd.DataFrame(columns=dmg_i.columns, index=dmg.index)

                # collect damaged quantities in each DS and add it to res
                res.append(pd.concat(
                    [dmg_q.mask(dmg_i == ds_i, cmp_i) for ds_i in ds_list],
                     axis=1, keys=[f'{ds_i:g}' for ds_i in ds_list]))

                # keep track of the components that have damaged quantities
                cmp_included.append(cmp_id)

        # assemble the final result and make sure the component keys are
        # included in the column header
        res = pd.concat(res, axis=1, keys=cmp_included)

        # If requested, the blocks with no damage are dropped
        if dropempty:
            res.dropna(how='all', axis=1, inplace=True)

        return res

class LossModel(object):
    """
    Parent object for loss models.

    All loss assessment methods should be children of this class.

    Parameters
    ----------

    """

    def __init__(self, assessment):

        self._asmnt = assessment

        self._sample = None

        self.loss_type = 'Generic'

    @property
    def sample(self):

        return self._sample

    def save_sample(self, filepath):
        """
        Save loss sample to a csv file

        """
        log_div()
        log_msg(f'Saving loss sample...')

        file_io.save_to_csv(self.sample, filepath)

        log_msg(f'Loss sample successfully saved.', prepend_timestamp=False)

    def load_sample(self, filepath):
        """
        Load damage sample data.

        """
        log_div()
        log_msg(f'Loading loss sample...')

        self._sample = file_io.load_from_csv(filepath)

        log_msg(f'Loss sample successfully loaded.', prepend_timestamp=False)

    def load_model(self, data_paths, mapping_path):
        """
        Load the list of prescribed consequence models and their parameters

        Parameters
        ----------
        data_paths: list of string
            List of paths to data files with consequence model parameters.
            Default XY datasets can be accessed as PelicunDefault/XY.
        mapping_path: string
            Path to a csv file that maps drivers (i.e., damage or edp data) to
            loss models.
        """

        log_div()
        log_msg(f'Loading loss map for {self.loss_type}...')

        loss_map = file_io.load_from_csv(mapping_path, orientation=1,
                             reindex=False, convert=[])

        loss_map['Driver'] = loss_map.index.values
        loss_map['Consequence'] = loss_map[self.loss_type]
        loss_map.index = np.arange(loss_map.shape[0])
        loss_map = loss_map.loc[:, ['Driver', 'Consequence']]
        loss_map.dropna(inplace=True)

        self.loss_map = loss_map

        log_msg(f"Loss map successfully parsed.", prepend_timestamp=False)

        log_div()
        log_msg(f'Loading loss parameters for {self.loss_type}...')

        # replace default flag with default data path
        for d_i, data_path in enumerate(data_paths):

            if 'PelicunDefault/' in data_path:
                data_paths[d_i] = data_path.replace('PelicunDefault/',
                                                    str(base.pelicun_path) +
                                                    '/resources/')

        data_list = []
        # load the data files one by one
        for data_path in data_paths:
            data = file_io.load_from_csv(
                data_path,
                orientation=1,
                reindex=False,
                convert=[]
            )

            data_list.append(data)

        loss_params = pd.concat(data_list, axis=0)

        # drop redefinitions of components
        loss_params = loss_params.groupby(level=[0,1]).first()

        # keep only the relevant data
        loss_cmp = np.unique(self.loss_map['Consequence'].values)
        loss_params = loss_params.loc[base.idx[loss_cmp, :],:]

        # drop unused damage states
        DS_list = loss_params.columns.get_level_values(0).unique()
        DS_to_drop = []
        for DS in DS_list:
            if np.all(pd.isna(loss_params.loc[:,base.idx[DS,:]].values)) == True:
                DS_to_drop.append(DS)

        loss_params.drop(columns=DS_to_drop, level=0, inplace=True)

        # convert values to internal SI units
        units = loss_params[('Quantity', 'Unit')]

        DS_list = loss_params.columns.get_level_values(0).unique()

        for DS in DS_list:
            if DS.startswith('DS'):

                # median values
                thetas = loss_params.loc[:, (DS, 'Theta_0')]
                loss_params.loc[:, (DS, 'Theta_0')] = [
                    base.convert_unit(theta, unit)
                    for theta, unit in list(zip(thetas, units))]

                # cov and beta for normal and lognormal dists. do not need to
                # be scaled
                families = loss_params.loc[:, (DS, 'Family')]
                for family in families:
                    if ((pd.isna(family)==True) or
                        (family in ['normal', 'lognormal'])):
                        pass
                    else:
                        raise ValueError(f"Unexpected distribution family in "
                                         f"loss model: {family}")

        # check for components with incomplete loss information
        cmp_incomplete_list = loss_params.loc[
            loss_params[('Incomplete', '')] == 1].index

        if len(cmp_incomplete_list) > 0:
            loss_params.drop(cmp_incomplete_list, inplace=True)

            log_msg(f"\nWARNING: Loss information is incomplete for the "
                    f"following component(s) {cmp_incomplete_list}. They were "
                    f"removed from the analysis.\n",
                    prepend_timestamp=False)

        self.loss_params = loss_params

        log_msg(f"Loss parameters successfully parsed.",
                prepend_timestamp=False)

    def aggregate_losses(self):
        """
        This is placeholder method.

        The method of aggregating the Decision Variable sample is specific to
        each DV and needs to be implemented in every child of the LossModel
        independently.
        """
        pass

    def _generate_DV_sample(self, dmg_quantities, sample_size):
        """
        This is placeholder method.

        The method of sampling decision variables in Decision Variable-specific
        and needs to be implemented in every child of the LossModel
        independently.
        """
        pass

    def calculate(self, sample_size):
        """
        Calculate the repair cost and time of each component block in the asset.

        """

        log_div()
        log_msg(f"Calculating losses...")

        # First, get the damaged quantities in each damage state for each block
        # of each component of interest.
        # Note that we are using the damages that drive the losses and not the
        # names of the loss components directly.
        log_msg(f"Preparing damaged quantities...")
        cmp_list = np.unique([val for driver_type, val
                              in self.loss_map['Driver'].values])
        dmg_q = self._asmnt.damage.prepare_dmg_quantities(cmp_list = cmp_list)

        # Now sample random Decision Variables
        # Note that this method is DV-specific and needs to be implemented in
        # every child of the LossModel independently.
        self._generate_DV_sample(dmg_q, sample_size)

        log_msg(f"Loss calculation successful.")

class BldgRepairModel(LossModel):
    """
    Manages building repair consequence assessments.

    Parameters
    ----------

    """

    def __init__(self, assessment):
        super(BldgRepairModel, self).__init__(assessment)

        self.loss_type = 'BldgRepair'

    def load_model(self, data_paths, mapping_path):

        super(BldgRepairModel, self).load_model(data_paths, mapping_path)

    def calculate(self, sample_size):

        super(BldgRepairModel, self).calculate(sample_size)

    def _create_DV_RVs(self, case_list):
        """
        Prepare the random variables used for repair cost and time simulation.

        Parameters
        ----------
        case_list: MultiIndex
            Index with cmp-ds-loc-dir-block descriptions that identify the RVs
            we need for the simulation.
        """

        RV_reg = uq.RandomVariableRegistry()
        LP = self.loss_params

        case_DF = pd.DataFrame(index=case_list, columns=[0,])
        driver_cmps = case_list.get_level_values(0).unique()

        rv_count = 0

        for loss_cmp_id in self.loss_map.index.values:

            # load the corresponding parameters
            driver_type, driver_cmp_id = self.loss_map.loc[loss_cmp_id, 'Driver']

            if driver_type != 'DMG':
                raise ValueError(f"Loss Driver type not recognized: "
                                 f"{driver_type}")

            # load the parameters
            if (driver_cmp_id, 'Cost') in LP.index:
                cost_params = LP.loc[(driver_cmp_id, 'Cost'), :]
            else:
                cost_params = None

            if (driver_cmp_id, 'Time') in LP.index:
                time_params = LP.loc[(driver_cmp_id, 'Time'), :]
            else:
                time_params = None

            if not driver_cmp_id in driver_cmps:
                continue

            for ds in case_DF.loc[
                      driver_cmp_id,:].index.get_level_values(0).unique():

                if cost_params is not None:
                    cost_family, cost_theta_1 = cost_params.loc[
                        [(f'DS{ds}', 'Family'), (f'DS{ds}','Theta_1')]]

                if time_params is not None:
                    time_family, time_theta_1 = time_params.loc[
                        [(f'DS{ds}', 'Family'), (f'DS{ds}', 'Theta_1')]]

                if ((pd.isna(cost_family)==True) and
                    (pd.isna(time_family)==True)):
                    continue

                # load the loc-dir-block cases
                loc_dir_block = case_DF.loc[(driver_cmp_id, ds)].index.values

                for loc, dir, block in loc_dir_block:

                    if pd.isna(cost_family)==False:

                        cost_rv_tag = f'COST-{loss_cmp_id}-{ds}-{loc}-{dir}-{block}'

                        RV_reg.add_RV(uq.RandomVariable(
                            name=cost_rv_tag,
                            distribution = cost_family,
                            theta = [1.0, cost_theta_1]
                        ))
                        rv_count += 1

                    if pd.isna(time_family) == False:
                        time_rv_tag = f'TIME-{loss_cmp_id}-{ds}-{loc}-{dir}-{block}'

                        RV_reg.add_RV(uq.RandomVariable(
                            name=time_rv_tag,
                            distribution=time_family,
                            theta=[1.0, time_theta_1]
                        ))
                        rv_count += 1

                    if ((pd.isna(cost_family) == False) and
                        (pd.isna(time_family) == False) and
                        (base.options.rho_cost_time != 0.0)):

                        rho = base.options.rho_cost_time

                        RV_reg.add_RV_set(uq.RandomVariableSet(
                            f'DV-{loss_cmp_id}-{ds}-{loc}-{dir}-{block}_set',
                            list(RV_reg.RVs([cost_rv_tag, time_rv_tag]).values()),
                            np.array([[1.0, rho],[rho, 1.0]])))

        log_msg(f"\n{rv_count} random variables created.",
                prepend_timestamp=False)

        return RV_reg

    def _calc_median_consequence(self, eco_qnt):
        """
        Calculate the median repair consequence for each loss component.

        """

        medians = {}

        for DV_type, DV_type_scase in zip(['COST', 'TIME'], ['Cost', 'Time']):

            cmp_list = []
            median_list = []

            for loss_cmp_id in self.loss_map.index:

                driver_type, driver_cmp = self.loss_map.loc[
                    loss_cmp_id, 'Driver']
                loss_cmp_name = self.loss_map.loc[loss_cmp_id, 'Consequence']

                if driver_type != 'DMG':
                    raise ValueError(f"Loss Driver type not recognized: "
                                     f"{driver_type}")

                if not driver_cmp in eco_qnt.columns.get_level_values(
                        0).unique():
                    continue

                ds_list = []
                sub_medians = []

                for ds in self.loss_params.columns.get_level_values(0).unique():

                    if not ds.startswith('DS'):
                        continue

                    ds_id = ds[2:]

                    theta_0 = self.loss_params.loc[
                        (loss_cmp_name, DV_type_scase),
                        (ds, 'Theta_0')]

                    if pd.isna(theta_0):
                        continue

                    try:
                        theta_0 = float(theta_0)
                        f_median = prep_constant_median_DV(theta_0)

                    except:
                        theta_0 = np.array(
                            [val.split(',') for val in theta_0.split('|')],
                            dtype=float)
                        f_median = prep_bounded_multilinear_median_DV(
                            theta_0[0], theta_0[1])

                    if 'ds' in eco_qnt.columns.names:

                        avail_ds = eco_qnt.loc[:,
                                   driver_cmp].columns.get_level_values(
                            0).unique()

                        if (not ds_id in avail_ds):
                            continue

                        eco_qnt_i = eco_qnt.loc[:, (driver_cmp, ds_id)].copy()

                    else:
                        eco_qnt_i = eco_qnt.loc[:, driver_cmp].copy()

                    if isinstance(eco_qnt_i, pd.Series):
                        eco_qnt_i = eco_qnt_i.to_frame()
                        eco_qnt_i.columns = ['X']
                        eco_qnt_i.columns.name = 'del'

                    eco_qnt_i.loc[:, :] = f_median(eco_qnt_i.values)

                    sub_medians.append(eco_qnt_i)
                    ds_list.append(ds_id)

                if len(ds_list) > 0:
                    median_list.append(pd.concat(sub_medians, axis=1,
                                                 keys=ds_list))
                    cmp_list.append(loss_cmp_id)

            if len(cmp_list) > 0:

                result = pd.concat(median_list, axis=1, keys=cmp_list)

                if 'del' in result.columns.names:
                    result.columns = result.columns.droplevel('del')

                if base.options.eco_scale["AcrossFloors"] == True:
                    result.columns.names = ['cmp', 'ds']

                else:
                    result.columns.names = ['cmp', 'ds', 'loc']

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

        log_div()
        log_msg(f"Aggregating repair consequences...")

        DV = self.sample

        # group results by DV type and location
        DVG = DV.groupby(level=[0, 4], axis=1).sum()

        # create the summary DF
        df_agg = pd.DataFrame(index=DV.index,
                              columns=['repair_cost',
                                       'repair_time-parallel',
                                       'repair_time-sequential'])

        df_agg['repair_cost'] = DVG['COST'].sum(axis=1)
        df_agg['repair_time-sequential'] = DVG['TIME'].sum(axis=1)

        df_agg['repair_time-parallel'] = DVG['TIME'].max(axis=1)

        df_agg = base.convert_to_MultiIndex(df_agg, axis=1)

        log_msg(f"Repair consequences successfully aggregated.")

        return df_agg


    def _generate_DV_sample(self, dmg_quantities, sample_size):
        """
        Generate a sample of repair costs and times.

        Parameters
        ----------
        dmg_quantitites: DataFrame
            A table with the quantity of damage experienced in each damage state
            of each component block at each location and direction. You can use
            the prepare_dmg_quantities method in the DamageModel to get such a
            DF.
        sample_size: integer
            The number of realizations to generate.

        """

        log_msg(f"Preparing random variables for repair cost and time...")
        RV_reg = self._create_DV_RVs(dmg_quantities.columns)

        RV_reg.generate_sample(sample_size=sample_size)

        std_sample = base.convert_to_MultiIndex(pd.DataFrame(RV_reg.RV_sample),
                                           axis=1).sort_index(axis=1)
        std_sample.columns.names = ['dv', 'cmp', 'ds', 'loc', 'dir', 'block']

        log_msg(f"\nSuccessfully generated {sample_size} realizations of "
                f"deviation from the median consequences.",
                prepend_timestamp=False)

        # calculate the quantities for economies of scale
        log_msg(f"\nCalculating the quantity of damage...",
                prepend_timestamp=False)

        if base.options.eco_scale["AcrossFloors"]==True:

            if base.options.eco_scale["AcrossDamageStates"] == True:

                eco_qnt = dmg_quantities.groupby(level=[0,], axis=1).sum()
                eco_qnt.columns.names = ['cmp',]

            else:

                eco_qnt = dmg_quantities.groupby(level=[0,1], axis=1).sum()
                eco_qnt.columns.names = ['cmp', 'ds']

        else:

            if base.options.eco_scale["AcrossDamageStates"] == True:

                eco_qnt = dmg_quantities.groupby(level=[0, 2], axis=1).sum()
                eco_qnt.columns.names = ['cmp', 'loc']

            else:

                eco_qnt = dmg_quantities.groupby(level=[0, 1, 2], axis=1).sum()
                eco_qnt.columns.names = ['cmp', 'ds', 'loc']

        log_msg(f"Successfully aggregated damage quantities.",
                prepend_timestamp=False)

        # apply the median functions, if needed, to get median consequences for
        # each realization
        log_msg(f"\nCalculating the median repair consequences...",
                prepend_timestamp=False)

        medians = self._calc_median_consequence(eco_qnt)

        log_msg(f"Successfully determined median repair consequences.",
                prepend_timestamp=False)

        # combine the median consequences with the samples of deviation from the
        # median to get the consequence realizations.
        log_msg(f"\nConsidering deviations from the median values to obtain "
                f"random DV sample...",
                prepend_timestamp=False)

        res_list = []
        key_list = []
        prob_cmp_list = std_sample.columns.get_level_values(1).unique()

        for DV_type, DV_type_scase in zip(['COST', 'TIME'],['Cost','Time']):

            cmp_list = []

            for cmp_i in medians[DV_type].columns.get_level_values(0).unique():

                # check if there is damage in the component
                driver_type, dmg_cmp_i = self.loss_map.loc[cmp_i, 'Driver']
                loss_cmp_i = self.loss_map.loc[cmp_i, 'Consequence']

                if driver_type != 'DMG':
                    raise ValueError(f"Loss Driver type not "
                                     f"recognized: {driver_type}")

                if not (dmg_cmp_i
                        in dmg_quantities.columns.get_level_values(0).unique()):
                    continue

                ds_list = []

                for ds in medians[DV_type].loc[:, cmp_i].columns.get_level_values(0).unique():

                    loc_list = []

                    for loc_id, loc in enumerate(
                            dmg_quantities.loc[:, (dmg_cmp_i, ds)].columns.get_level_values(0).unique()):

                        if ((base.options.eco_scale["AcrossFloors"] == True) and
                            (loc_id > 0)):
                            break

                        if base.options.eco_scale["AcrossFloors"] == True:
                            median_i = medians[DV_type].loc[:,(cmp_i, ds)]
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

                    if base.options.eco_scale["AcrossFloors"] == True:
                        ds_list += [ds, ]
                    else:
                        ds_list+=[(ds, loc) for loc in loc_list]

                if base.options.eco_scale["AcrossFloors"] == True:
                    cmp_list += [(loss_cmp_i, dmg_cmp_i, ds) for ds in ds_list]
                else:
                    cmp_list+=[(loss_cmp_i, dmg_cmp_i, ds, loc) for ds, loc in ds_list]

            if base.options.eco_scale["AcrossFloors"] == True:
                key_list += [(DV_type, loss_cmp_i, dmg_cmp_i, ds)
                             for loss_cmp_i, dmg_cmp_i, ds in cmp_list]
            else:
                key_list+=[(DV_type, loss_cmp_i, dmg_cmp_i, ds, loc)
                           for loss_cmp_i, dmg_cmp_i, ds, loc in cmp_list]

        lvl_names = ['dv', 'loss', 'dmg', 'ds', 'loc', 'dir', 'block']
        DV_sample = pd.concat(res_list, axis=1, keys=key_list,
                              names = lvl_names)

        DV_sample = DV_sample.fillna(0).convert_dtypes()
        DV_sample.columns.names = lvl_names

        # When the 'replacement' consequence is triggered, all local repair
        # consequences are discarded. Note that global consequences are assigned
        # to location '0'.

        # Get the flags for replacement consequence trigger
        id_replacement = DV_sample.groupby(level=[1, ],
                                           axis=1).sum()['replacement'] > 0

        # get the list of non-zero locations
        locs = DV_sample.columns.get_level_values(4).unique().values
        locs = locs[locs != '0']

        DV_sample.loc[id_replacement, base.idx[:, :, :, :, locs]] = 0.0

        self._sample = DV_sample

        log_msg(f"Successfully obtained DV sample.",
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
    def f(quantity):
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
