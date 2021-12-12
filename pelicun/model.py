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

from .base import *
from .uq import *
from .file_io import save_to_csv, load_from_csv

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

            sample = convert_to_MultiIndex(sample, axis=1)['EDP']

            self._sample = sample

        else:
            sample = self._sample

        return sample

    def save_sample(self, filepath):
        """
        Save demand sample to a csv file

        """

        save_to_csv(self.sample, filepath, units=self.units)

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

                if options.verbose:
                    log_msg(f'Removing event_ID from header...',
                            prepend_timestamp=False)

                new_column_index = np.array(
                    [old_MI.get_level_values(i) for i in range(1, 4)])

            else:
                new_column_index = np.array(
                    [old_MI.get_level_values(i) for i in range(3)])

            # Remove whitespace to avoid ambiguity

            if options.verbose:
                log_msg(f'Removing whitespace from header...',
                        prepend_timestamp=False)

            wspace_remove = np.vectorize(lambda name: name.replace(' ', ''))

            new_column_index = wspace_remove(new_column_index)

            # Creating new, cleaned-up header

            new_MI = pd.MultiIndex.from_arrays(
                new_column_index, names=['type', 'loc', 'dir'])

            return new_MI



        demand_data, units = load_from_csv(filepath, return_units=True)

        log_div()
        log_msg(f'Loading demand data...')

        parsed_data = demand_data.copy()

        # start with cleaning up the header

        parsed_data.columns = parse_header(parsed_data.columns)

        # Remove errors, if needed
        if 'ERROR' in parsed_data.columns.get_level_values(0):

            log_msg(f'Removing errors from the raw data...',
                    prepend_timestamp=False)

            error_list = parsed_data.loc[:,idx['ERROR',:,:]].values.astype(bool)

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
            cal_df.loc[idx[cols,:,:], 'family'] = settings['DistributionFamily']

            # load the censor limits
            if 'CensorAt' in settings.keys():
                censor_lower, censor_upper = settings['CensorAt']
                cal_df.loc[idx[cols,:,:], 'censor_lower'] = censor_lower
                cal_df.loc[idx[cols,:,:], 'censor_upper'] = censor_upper

            # load the truncation limits
            if 'TruncateAt' in settings.keys():
                truncate_lower, truncate_upper = settings['TruncateAt']
                cal_df.loc[idx[cols,:,:], 'truncate_lower'] = truncate_lower
                cal_df.loc[idx[cols,:,:], 'truncate_upper'] = truncate_upper

            # scale the censor and truncation limits, if needed
            scale_factor = options.scale_factor(settings.get('Unit', None))

            rows_to_scale = ['censor_lower', 'censor_upper',
                             'truncate_lower', 'truncate_upper']
            cal_df.loc[idx[cols,:,:], rows_to_scale] *= scale_factor

            # load the prescribed additional uncertainty
            if 'AddUncertainty' in settings.keys():

                sig_increase = settings['AddUncertainty']

                # scale the sig value if the target distribution family is normal
                if settings['DistributionFamily'] == 'normal':
                    sig_increase *= scale_factor

                cal_df.loc[idx[cols,:,:], 'sig_increase'] = sig_increase

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

        if options.verbose:
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

        if options.verbose:
            log_msg(f"\nDemand data used for calibration:\n"+str(demand_sample),
                    prepend_timestamp=False)

        # fit the joint distribution
        log_msg(f"\nFitting the prescribed joint demand distribution...",
                prepend_timestamp=False)

        demand_theta, demand_rho = fit_distribution(
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

        # save the correlation and empirical data
        save_to_csv(self.correlation, file_prefix + '_correlation.csv')
        save_to_csv(self.empirical_data, file_prefix + '_empirical.csv',
                    units=self.units)

        # the log standard deviations in the marginal parameters need to be
        # scaled up before feeding to the saving method where they will be
        # scaled back down and end up being saved unscaled to the target file

        marginal_params = self.marginal_params.copy()

        log_rows = marginal_params['family']=='lognormal'
        log_demands = marginal_params.loc[log_rows,:]

        for label in log_demands.index:

            if label in self.units.index:

                unit_factor = globals()[self.units[label]]

                marginal_params.loc[label, 'theta_1'] *= unit_factor

        save_to_csv(marginal_params, file_prefix+'_marginals.csv',
                    units=self.units, orientation=1)

    def load_model(self, file_prefix):

        self.empirical_data = load_from_csv(file_prefix+'_empirical.csv')
        self.empirical_data.columns.set_names(['type', 'loc', 'dir'],
                                            inplace=True)

        self.correlation = load_from_csv(file_prefix + '_correlation.csv',
                                         reindex=False)
        self.correlation.index.set_names(['type', 'loc', 'dir'], inplace=True)
        self.correlation.columns.set_names(['type', 'loc', 'dir'], inplace=True)

        # the log standard deviations in the marginal parameters need to be
        # adjusted after getting the data from the loading method where they
        # were scaled according to the units of the corresponding variable
        marginal_params, units = load_from_csv(file_prefix + '_marginals.csv',
                                               orientation=1, reindex=False,
                                               return_units=True)
        marginal_params.index.set_names(['type', 'loc', 'dir'],inplace=True)

        log_rows = marginal_params.loc[:, 'family'] == 'lognormal'
        log_demands = marginal_params.loc[log_rows,:].index.values

        for label in log_demands:

            if label in units.index:

                unit_factor = globals()[units[label]]

                marginal_params.loc[label, 'theta_1'] /= unit_factor

        self.units = units
        self.marginal_params = marginal_params

    def _create_RVs(self, preserve_order=False):
        """
        Create a random variable registry for the joint distribution of demands.

        """

        # initialize the registry
        RV_reg = RandomVariableRegistry()

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
                RV_reg.add_RV(RandomVariable(
                    name=rv_tag,
                    distribution=dist_family,
                    raw_samples=self.empirical_data.loc[:, edp].values
                ))

            else:

                # all other RVs need parameters of their distributions
                RV_reg.add_RV(RandomVariable(
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

        RV_reg.add_RV_set(RandomVariableSet(
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

            cmp_sample = convert_to_MultiIndex(cmp_sample, axis=1)['CMP']

            self._cmp_sample = cmp_sample

        else:
            cmp_sample = self._cmp_sample

        return cmp_sample

    def save_cmp_sample(self, filepath):
        """
        Save component quantity sample to a csv file

        """

        # prepare a units array
        sample = self.cmp_sample

        units = pd.Series(name='units', index=sample.columns)

        for cmp_id, unit_name in self.cmp_units.items():
            units.loc[cmp_id, :] = unit_name

        save_to_csv(sample, filepath, units=units)

    def load_cmp_sample(self, filepath):
        """
        Load component quantity sample from a csv file

        """

        sample, units = load_from_csv(filepath, return_units=True)

        self._cmp_sample = sample

        self.cmp_units = units.groupby(level=0).first()

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
        marginal_params, units = load_from_csv(
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
                unit_factor = globals()[unit_name]

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
        RV_reg = RandomVariableRegistry()

        # add a random variable for each component quantity variable
        for rv_params in self.cmp_marginal_params.itertuples():

            cmp = rv_params.Index
            rv_tag = f'CMP-{cmp[0]}-{cmp[1]}-{cmp[2]}-{cmp[3]}'

            if pd.isnull(rv_params.family):

                # we use an empirical RV to generate deterministic values
                RV_reg.add_RV(RandomVariable(
                    name=rv_tag,
                    distribution='empirical',
                    raw_samples=np.ones(10000) * rv_params.theta_0
                ))

            else:

                # all other RVs need parameters of their distributions
                RV_reg.add_RV(RandomVariable(
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

    @property
    def frg_sample(self):

        if self._frg_sample is None:

            frg_sample = pd.DataFrame(self._frg_RVs.RV_sample)

            frg_sample = convert_to_MultiIndex(frg_sample, axis=1)['FRG']

            self._frg_sample = frg_sample

        else:
            frg_sample = self._frg_sample

        return frg_sample

    @property
    def lsds_sample(self):

        if self._lsds_sample is None:

            lsds_sample = pd.DataFrame(self._lsds_RVs.RV_sample)

            lsds_sample = convert_to_MultiIndex(lsds_sample, axis=1)['LSDS']

            lsds_sample = lsds_sample.astype(int)

            self._lsds_sample = lsds_sample

        else:
            lsds_sample = self._lsds_sample

        return lsds_sample

    def load_fragility_model(self, data_paths):
        """
        Load limit state fragility functions and damage state assignments

        Parameters
        ----------
        data_paths: list of string
            List of paths to data files with fragility information. Default
            datasets can be accessed as PelicunDefault/XY.
        """

        # replace default flag with default data path
        for d_i, data_path in enumerate(data_paths):

            if 'PelicunDefault/' in data_path:
                data_paths[d_i] = data_path.replace('PelicunDefault/',
                                                   str(pelicun_path)+'/resources/')

        data_list = []
        # load the data files one by one
        for data_path in data_paths:

            data = load_from_csv(
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
                unit_factor = globals()[unit_name]

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
        frg_RV_reg = RandomVariableRegistry()
        lsds_RV_reg = RandomVariableRegistry()

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

                            frg_RV_reg.add_RV(RandomVariable(
                                name=frg_rv_tag,
                                distribution='empirical',
                                raw_samples=np.ones(10000) * theta_0
                            ))

                        else:

                            # all other RVs have parameters of their distributions
                            frg_RV_reg.add_RV(RandomVariable(
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

                            lsds_RV_reg.add_RV(RandomVariable(
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

                            lsds_RV_reg.add_RV(RandomVariable(
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
                frg_RV_reg.add_RV_set(RandomVariableSet(
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

        log_div()
        log_msg(f'Generating sample from fragility variables...')

        self._create_frg_RVs()

        self._frg_RVs.generate_sample(sample_size=sample_size)

        self._lsds_RVs.generate_sample(sample_size=sample_size)

        # replace the potentially existing raw sample with the generated one
        self._frg_sample = None
        self._lsds_sample = None

        log_msg(f"\nSuccessfully generated {sample_size} realizations.",
                prepend_timestamp=False)

    def calculate(self, sample_size):
        """
        Calculate the damage state of each component block in the asset.

        """

        # Generate an array with component capacities for each block and
        # generate a second array that assigns a specific damage state to
        # each component limit state. The latter is primarily needed to handle
        # limit states with multiple, mutually exclusive DS options
        self._generate_frg_sample(sample_size)

        # Use the above arrays to evaluate the damage state of each block



class FragilityFunction(object):
    """
    Describes the relationship between asset response and damage.

    Asset response is characterized by a Demand value that represents an
    engineering demand parameter (EDP). Only a scalar EDP is supported
    currently. The damage is characterized by a set of DamageStateGroup (DSG)
    objects. For each DSG, the corresponding EDP limit (i.e. the EDP at which
    the asset is assumed to experience damage described by the DSG) is
    considered uncertain; hence, it is described by a random variable. The
    random variables that describe EDP limits for the set of DSGs are not
    necessarily independent.

    We assume that the EDP limit will be approximated by a probability
    distribution for each DSG and these variables together form a multivariate
    distribution. Following common practice, the correlation between
    variables is assumed perfect by default, but the framework allows the
    users to explore other, more realistic options.

    Parameters
    ----------
    EDP_limit: list of RandomVariable
        A list of correlated random variables where each variable corresponds
        to an EDP limit that triggers a damage state. The number of
        list elements shall be equal to the number of DSGs handled by the
        Fragility Function (FF) and they shall be in ascending order of damage
        severity.

    """

    def __init__(self, EDP_limit):
        self._EDP_limit = EDP_limit

        self._EDP_tags = [EDP_lim_i.name for EDP_lim_i in EDP_limit]

    def P_exc(self, EDP, DSG_ID):
        """
        Return the probability of damage state exceedance.

        Calculate the probability of exceeding the damage corresponding to the
        DSG identified by the DSG_ID conditioned on a particular EDP value.

        Parameters
        ----------
        EDP: float scalar or ndarray
            Single EDP or numpy array of EDP values.
        DSG_ID: int
            Identifies the conditioning DSG. The DSG numbering is 1-based,
            because zero typically corresponds to the undamaged state.

        Returns
        -------
        P_exc: float scalar or ndarray
            DSG exceedance probability at the given EDP point(s).
        """

        EDP = np.asarray(EDP, dtype=np.float64)
        nvals = EDP.size

        # The exceedance probability corresponding to no damage is 1.
        # Although this is trivial, returning 1 for DSG_ID=0 makes using P_exc
        # more convenient.
        if DSG_ID == 0:
            P_exc = np.ones(EDP.size)
        else:
            # prepare the limits for the density calculation
            ndims = len(self._EDP_limit)

            limit_list = np.full((ndims, nvals), -np.inf, dtype=np.float64)
            limit_list[DSG_ID - 1:] = EDP
            limit_list[:DSG_ID - 1] = None

            P_exc = 1.0 - self._EDP_limit[0].RV_set.orthotope_density(
                lower=limit_list, var_subset = self._EDP_tags)

        # if EDP was a scalar, make sure that the result is also a scalar
        if EDP.size == 1:
            return P_exc[0]
        else:
            return P_exc

    def DSG_given_EDP(self, EDP, force_resampling=False):
        """
        Given an EDP, get a damage level based on the fragility function.

        The damage is evaluated by sampling the joint distribution of
        fragilities corresponding to all possible damage levels and checking
        which damage level the given EDP falls into. This approach allows for
        efficient damage state evaluation for a large number of EDP
        realizations.

        Parameters
        ----------
        EDP: float scalar or ndarray or Series
            Single EDP, or numpy array or pandas Series of EDP values.
        force_resampling: bool, optional, default: False
            If True, the probability distribution is resampled before
            evaluating the damage for each EDP. This is not recommended if the
            fragility functions are correlated with other sources of
            uncertainty because those variables will also be resampled in this
            case. If False, which is the default approach, we assume that
            the random variable has already been sampled and the number of
            samples greater or equal to the number of EDP values.

        Returns
        -------
        DSG_ID: Series
            Identifies the damage that corresponds to the given EDP. A DSG_ID
            of 0 means no damage.

        """
        # get the number of samples needed
        nsamples = EDP.size

        # if there are no samples or resampling is forced, then sample the
        # distribution first
        # TODO: force_resampling is probably not needed
        # if force_resampling or (self._EDP_limit.samples is None):
        #     self._EDP_limit.sample_distribution(sample_size=nsamples)
        #
        # # if the number of samples is not sufficiently large, raise an error
        # if self._EDP_limit.samples.shape[0] < nsamples:
        #     raise ValueError(
        #         'Damage evaluation requires at least as many samples of the '
        #         'joint distribution defined by the fragility functions as '
        #         'many EDP values are provided to the DSG_given_EDP function. '
        #         'You might want to consider setting force_resampling to True '
        #         'or sampling the distribution before calling the DSG_given_EDP '
        #         'function.')

        #samples = pd.DataFrame(self._EDP_limit.samples)
        samples = pd.DataFrame(dict([(lim_i.name, lim_i.sample)
                                     for lim_i in self._EDP_limit]))

        if type(EDP) not in [pd.Series, pd.DataFrame]:
            EDP = pd.Series(EDP, name='EDP')

        #nstates = samples.columns.values.size
        nstates = samples.shape[1]

        samples = samples.loc[EDP.index,:]

        # sort columns
        sample_cols = samples.columns
        samples = samples[sample_cols[np.argsort(sample_cols)]]

        # check for EDP exceedance
        EXC = samples.sub(EDP, axis=0) < 0.

        DSG_ID = pd.Series(np.zeros(len(samples.index)), name='DSG_ID',
                           index=samples.index, dtype=np.int)

        for s in range(nstates):
            DSG_ID[EXC.iloc[:,s]] = s + 1

        return DSG_ID

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

def prep_bounded_linear_median_DV(median_max, median_min, quantity_lower,
                                  quantity_upper):
    """
    Returns a bounded linear median Decision Variable (DV) function.

    The median DV equals the min and max values when the quantity is
    outside of the prescribed quantity bounds. When the quantity is within the
    bounds, the returned median is calculated by a linear function with a
    negative slope between max and min values.

    Parameters
    ----------
    median_max: float, optional
    median_min: float, optional
        Minimum and maximum limits that define the bounded_linear median DV
        function.
    quantity_lower: float, optional
    quantity_upper: float, optional
        Lower and upper bounds of component quantity that define the
        bounded_linear median DV function.

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
        output = np.interp(q_array,
                           [quantity_lower, quantity_upper],
                           [median_max, median_min])

        return output

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

class ConsequenceFunction(object):
    """
    Describes the relationship between damage and a decision variable.

    Indicates the distribution of a quantified Decision Variable (DV)
    conditioned on a component, an element, or the system reaching a given
    damage state (DS). DV can be reconstruction cost, repair time, casualties,
    injuries, etc. Its distribution might depend on the quantity of damaged
    components.

    Parameters
    ----------

    DV_median: callable
        Describes the median DV as an f(quantity) function of the total
        quantity of damaged components. Use the prep_constant_median_DV, and
        prep_bounded_linear_median_DV helper functions to conveniently
        prescribe the typical FEMA P-58 functions.
    DV_distribution: RandomVariable
        A random variable that characterizes the uncertainty in the DV. The
        distribution shall be normalized by the median DV (i.e. the RV is
        expected to have a unit median). Truncation can be used to
        prescribe lower and upper limits for the DV, such as the (0,1) domain
        needed for red tag evaluation.

    """

    def __init__(self, DV_median, DV_distribution):

        self._DV_median = DV_median
        self._DV_distribution = DV_distribution

    def median(self, quantity=None):
        """
        Return the value of the median DV.

        The median DV corresponds to the component damage state (DS). If the
        damage consequence depends on the quantity of damaged components, the
        total quantity of damaged components shall be specified through the
        quantity parameter.

        Parameters
        ----------
        quantity: float scalar or ndarray, optional
            Total quantity of damaged components that determines the magnitude
            of median DV. Not needed for consequence functions with a fixed
            median DV.

        Returns
        -------
        median: float scalar or ndarray
            A single scalar for fixed median; a scalar or an array depending on
            the shape of the quantity parameter for bounded_linear median.

        """
        return self._DV_median(quantity)

    def sample_unit_DV(self, quantity=None, sample_size=1,
                       force_resampling=False):
        """
        Sample the decision variable quantity per component unit.

        The Unit Decision Variable (UDV) corresponds to the component Damage
        State (DS). It shall be multiplied by the quantity of damaged
        components to get the total DV that corresponds to the quantity of the
        damaged components in the asset. If the DV depends on the total
        quantity of damaged components, that value shall be specified through
        the quantity parameter.

        Parameters
        ----------
        quantity: float scalar, ndarray or Series, optional, default: None
            Total quantity of damaged components that determines the magnitude
            of median DV. Not needed for consequence functions with a fixed
            median DV.
        sample_size: int, optional, default: 1
            Number of samples drawn from the DV distribution. The default value
            yields one sample. If quantity is an array with more than one
            element, the sample_size parameter is ignored.
        force_resampling: bool, optional, default: False
            If True, the DV distribution (and the corresponding RV if there
            are correlations) is resampled even if there are samples already
            available. This is not recommended if the DV distribution is
            correlated with other sources of uncertainty because those
            variables will also be resampled in this case. If False, which is
            the default approach, we assume that the random variable has
            already been sampled and the number of samples is greater or equal
            to the number of samples requested.

        Returns
        -------
        unit_DV: float scalar or ndarray
            Unit DV samples.

        """
        # get the median DV conditioned on the provided quantities
        median = self.median(quantity=np.asarray(quantity))

        # if the distribution is None, then there is no uncertainty in the DV
        # and the median values are the samples
        if self._DV_distribution is None:
            return median

        else:
            # if there are more than one quantities defined, infer the number of
            # samples from the number of quantities
            if quantity is not None:
                if type(quantity) not in [pd.Series, pd.DataFrame]:
                    quantity = pd.Series(quantity, name='QNT')

                if quantity.size > 1:
                    sample_size = quantity.size
                elif sample_size > 1:
                    quantity = pd.Series(np.ones(sample_size) * quantity.values,
                                         name='QNT')

            # if there are no samples or resampling is forced, then sample the
            # distribution first
            # TODO: force_resampling is probably not needed
            # if (force_resampling or
            #     (self._DV_distribution.samples is None)):
            #     self._DV_distribution.sample_distribution(sample_size=sample_size)

            # # if the number of samples is not sufficiently large, raise an error
            # if self._DV_distribution.samples.shape[0] < sample_size:
            #     raise ValueError(
            #         'Consequence evaluation requires at least as many samples of '
            #         'the Decision Variable distribution as many samples are '
            #         'requested or as many quantity values are provided to the '
            #         'sample_unit_DV function. You might want to consider setting '
            #         'force_resampling to True or sampling the distribution before '
            #         'calling the sample_unit_DV function.')

            # get the samples
            if quantity is not None:
                samples = pd.Series(self._DV_distribution.sample).loc[quantity.index]
            else:
                samples = pd.Series(self._DV_distribution.sample).iloc[:sample_size]
            samples = samples * median

            return samples


class DamageState(object):
    """
    Characterizes one type of damage that corresponds to a particular DSG.

    The occurrence of damage is evaluated at the DSG. The DS describes one of
    the possibly several types of damages that belong to the same DSG and the
    consequences of such damage.

    Parameters
    ----------

    ID:int
    weight: float, optional, default: 1.0
        Describes the probability of DS occurrence, conditioned on the damage
        being in the DSG linked to this DS. This information is only used for
        DSGs with multiple DS corresponding to them. The weights of the set of
        DS shall sum up to 1.0 if they are mutually exclusive. When the set of
        DS occur simultaneously, the sum of weights typically exceeds 1.0.
    description: str, optional
        Provides a short description of the damage state.
    affected_area: float, optional, default: 0.
        Defines the area over which life safety hazards from this DS exist.
    repair_cost_CF: ConsequenceFunction, optional
        A consequence function that describes the cost necessary to restore the
        component to its pre-disaster condition.
    reconstruction_time_CF: ConsequenceFunction, optional
        A consequence function that describes the time, necessary to repair the
        damaged component to its pre-disaster condition.
    injuries_CF_set: ConsequenceFunction array, optional
        A set of consequence functions; each describes the number of people
        expected to experience injury of a particular severity when the
        component is in this DS. Any number of injury-levels can be considered.
    red_tag_CF: ConsequenceFunction, optional
        A consequence function that describes the proportion of components
        (within a Performance Group) that needs to be damaged to trigger an
        unsafe placard (i.e. red tag) for the building during post-disaster
        inspection.

    """

    def __init__(self, ID, weight=1.0, description='',
                 repair_cost_CF=None, reconstruction_time_CF=None,
                 injuries_CF_set=None, affected_area=0., red_tag_CF=None):
        self._ID = int(ID)
        self._weight = float(weight)
        self._description = description
        self._repair_cost_CF = repair_cost_CF
        self._reconstruction_time_CF = reconstruction_time_CF
        self._injuries_CF_set = injuries_CF_set
        self._affected_area = affected_area
        self._red_tag_CF = red_tag_CF

    @property
    def description(self):
        """
        Return the damage description.
        """
        return self._description

    @property
    def weight(self):
        """
        Return the weight of DS among the set of damage states in the DSG.
        """
        return self._weight

    def unit_repair_cost(self, quantity=None, sample_size=1, **kwargs):
        """
        Sample the repair cost distribution and return the unit repair costs.

        The unit repair costs shall be multiplied by the quantity of damaged
        components to get the total repair costs for the components in this DS.

        Parameters
        ----------
        quantity: float scalar, ndarray or Series, optional, default: None
            Total quantity of damaged components that determines the median
            repair cost. Not used for repair cost models with fixed median.
        sample_size: int, optional, default: 1
            Number of samples drawn from the repair cost distribution. The
            default value yields one sample.

        Returns
        -------
        unit_repair_cost: float scalar or ndarray
            Unit repair cost samples.

        """
        output = None
        if self._repair_cost_CF is not None:
            output = self._repair_cost_CF.sample_unit_DV(quantity=quantity,
                                                         sample_size=sample_size,
                                                         **kwargs)

        return output

    def unit_reconstruction_time(self, quantity=None, sample_size=1,
                                 **kwargs):
        """
        Sample the reconstruction time distribution and return the unit
        reconstruction times.

        The unit reconstruction times shall be multiplied by the quantity of
        damaged components to get the total reconstruction time for the
        components in this DS.

        Parameters
        ----------
        quantity: float scalar, ndarray or Series, optional, default: None
            Total quantity of damaged components that determines the magnitude
            of median reconstruction time. Not used for reconstruction time
            models with fixed median.
        sample_size: int, optional, default: 1
            Number of samples drawn from the reconstruction time distribution.
            The default value yields one sample.

        Returns
        -------
        unit_reconstruction_time: float scalar or ndarray
            Unit reconstruction time samples.

        """
        output = None
        if self._reconstruction_time_CF is not None:
            output = self._reconstruction_time_CF.sample_unit_DV(
                quantity=quantity,
                sample_size=sample_size, **kwargs)

        return output

    def red_tag_dmg_limit(self, sample_size=1, **kwargs):
        """
        Sample the red tag consequence function and return the proportion of
        components that needs to be damaged to trigger a red tag.

        The red tag consequence function is assumed to have a fixed median
        value that does not depend on the quantity of damaged components.

        Parameters
        ----------
        sample_size: int, optional, default: 1
            Number of samples drawn from the red tag consequence distribution.
            The default value yields one sample.

        Returns
        -------
        red_tag_trigger: float scalar or ndarray
            Samples of damaged component proportions that trigger a red tag.

        """
        output = None
        if self._red_tag_CF is not None:
            output = self._red_tag_CF.sample_unit_DV(sample_size=sample_size,
                                                     **kwargs)

        return output

    def unit_injuries(self, severity_level=0, sample_size=1, **kwargs):
        """
        Sample the injury consequence function that corresponds to the
        specified level of severity and return the injuries per component unit.

        The injury consequence function is assumed to have a fixed median
        value that does not depend on the quantity of damaged components (i.e.
        the number of injuries per component unit does not change with the
        quantity of components.)

        Parameters
        ----------
        severity_level: int, optional, default: 1
            Identifies which injury consequence to sample. The indexing of
            severity levels is zero-based.
        sample_size: int, optional, default: 1
            Number of samples drawn from the injury consequence distribution.
            The default value yields one sample.

        Returns
        -------
        unit_injuries: float scalar or ndarray
            Unit injury samples.

        """


        output = None
        if len(self._injuries_CF_set) > severity_level:
            CF = self._injuries_CF_set[severity_level]
            if CF is not None:
                output = CF.sample_unit_DV(sample_size=sample_size, **kwargs)

        return output


class DamageStateGroup(object):
    """
    A set of similar component damages that are controlled by the same EDP.

    Damages are described in detail by the set of Damage State objects.
    Damages in a DSG are assumed to occur at the same EDP magnitude. A Damage
    State Group (DSG) might have only a single DS in the simplest case.

    Parameters
    ----------
    ID: int
    DS_set: DamageState array
    DS_set_kind: {'single', 'mutually_exclusive', 'simultaneous'}
        Specifies the relationship among the DS in the set. When only one DS is
        defined, use the 'single' option to improve calculation efficiency.
        When multiple DS are present, the 'mutually_exclusive' option assumes
        that the occurrence of one DS precludes the occurrence of another DS.
        In such a case, the weights of the DS in the set shall sum up to 1.0.
        In a 'simultaneous' case the DS are independent and unrelated. Hence,
        they can occur at the same time and at least one of them has to occur.
    """

    def __init__(self, ID, DS_set, DS_set_kind):
        self._ID = ID
        self._DS_set = DS_set
        self._DS_set_kind = DS_set_kind

class PerformanceGroup(object):
    """
    A group of similar components that experience the same demands.

    FEMA P-58: Performance Groups (PGs) are a sub-categorization of fragility
    groups. A performance group is a subset of fragility group components that
    are subjected to the same demands (e.g. story drift, floor acceleration,
    etc.).

    In buildings, most performance groups shall be organized by story level.
    There is no need to separate performance groups by direction, because the
    direction of components within a group can be specified during definition,
    and it will be taken into consideration in the analysis.

    Parameters
    ----------
    ID: int
    location: int
        Identifies the location of the components that belong to the PG. In a
        building, location shall typically refer to the story of the building.
        The location assigned to each PG shall be in agreement with the
        locations assigned to the Demand objects.
    quantity: RandomVariable
        Specifies the quantity of components that belong to this PG.
        Uncertainty in component quantities is considered by assigning a
        random variable to this property.
    fragility_functions: FragilityFunction list
        Each fragility function describes the probability that the damage in
        a subset of components will meet or exceed the damages described by
        each damage state group in the DSG_set. Each is a multi-dimensional
        function if there is more than one DSG. The number of functions shall
        match the number of subsets defined by the `csg_weights` parameter.
    DSG_set: DamageStateGroup array
        A set of sequential Damage State Groups that describe the plausible set
        of damage states of the components in the FG.
    csg_weights: float ndarray, optional, default: [1.0]
        Identifies subgroups of components within a PG, each of which have
        perfectly correlated behavior. Correlation between the damage and
        consequences among subgroups is controlled by the `correlation`
        parameter of the FragilityGroup that the PG belongs to. Note that if
        the components are assumed to have perfectly correlated behavior at the
        PG level, assigning several subgroups to the PG is unnecessary. This
        input shall be a list of weights that are applied to the quantity
        of components to define the amount of components in each subgroup. The
        sum of assigned weights shall be 1.0.
    directions: int ndarray, optional, default: [0]
        Identifies the direction of each subgroup of components within the PG.
        The number of directions shall be identical to the number of
        csg_weights assigned. In buildings, directions typically correspond to
        the orientation of components in plane. Hence, using 0 or 1 to identify
        'X' or 'Y' is recommended. These directions shall be in agreement with
        the directions assigned to Demand objects.
    """

    def __init__(self, ID, location, quantity, fragility_functions, DSG_set,
                 csg_weights=[1.0], direction=0):
        self._ID = ID
        self._location = location
        self._quantity = quantity
        if type(fragility_functions) == FragilityFunction:
            self._FF_set = [fragility_functions,]
        else:
            self._FF_set = fragility_functions
        self._DSG_set = DSG_set
        self._csg_weights = csg_weights
        self._direction = direction

    def P_exc(self, EDP, DSG_ID):
        """
        This is a convenience function that provides a shortcut to
        fragility_function.P_exc(). It calculates the exceedance probability
        of a given DSG conditioned on the provided EDP value(s). The fragility
        functions assigned to the first subset are used for this calculation
        because P_exc shall be identical among subsets.

        Parameters
        ----------
        EDP: float scalar or ndarray
            Single EDP or numpy array of EDP values.
        DSG_ID: int
            Identifies the DSG of interest.

        Returns
        -------
        P_exc: float scalar or ndarray
            Exceedance probability of the given DSG at the EDP point(s).
        """
        return self._FF_set[0].P_exc(EDP, DSG_ID)


class FragilityGroup(object):
    """
    Groups a set of similar components from a loss-assessment perspective.

    Characterizes a set of structural or non-structural components that have
    similar construction characteristics, similar potential modes of damage,
    similar probability of incurring those modes of damage, and similar
    potential consequences resulting from their damage.

    Parameters
    ----------
    ID: int
    demand_type: {'PID', 'PFA', 'PSD', 'PSA', 'ePGA', 'PGD'}
        The type of Engineering Demand Parameter (EDP) that controls the damage
        of the components in the FG. See Demand for acronym descriptions.
    performance_groups: PerformanceGroup array
        A list of performance groups that contain the components characterized
        by the FG.
    directional: bool, optional, default: True
        Determines whether the components in the FG are sensitive to the
        directionality of the EDP.
    correlation: bool, optional, default: True
        Determines whether the components within a Performance Group (PG) will
        have correlated or uncorrelated damage. Correlated damage means that
        all components will have the same damage state. In the uncorrelated
        case, each component in the performance group will have its damage
        state evaluated independently. Correlated damage reduces the required
        computational effort for the calculation. Incorrect correlation
        modeling will only slightly affect the mean estimates, but might
        significantly change the dispersion of results.
    demand_location_offset: int, optional, default: 0
        Indicates if the location for the demand shall be different from the
        location of the components. Damage to components of the ceiling, for
        example, is controlled by demands on the floor above the one that the
        components belong to. This can be indicated by setting the
        demand_location_offset to 1 for such an FG.
    incomplete: bool, optional, default: False
        Indicates that the FG information is not complete and corresponding
        results shall be treated with caution.
    name: str, optional, default: ''
        Provides a short description of the fragility group.
    description: str, optional, default: ''
        Provides a detailed description of the fragility group.
    """

    def __init__(self, ID, demand_type, performance_groups,
                 directional=True, correlation=True, demand_location_offset=0,
                 incomplete=False, name='', description='', unit="ea"):
        self._ID = ID
        self._demand_type = demand_type
        self._performance_groups = performance_groups
        self._directional = directional
        self._correlation = correlation
        self._demand_location_offset = demand_location_offset
        self._incomplete = incomplete
        self._name = name
        self._description = description
        self._unit = unit

    @property
    def description(self):
        """
        Return the fragility group description.
        """
        return self._description

    @property
    def name(self):
        """
        Return the name of the fragility group.

        """
        return self._name