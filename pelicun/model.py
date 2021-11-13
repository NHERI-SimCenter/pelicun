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
from .uq import fit_distribution

class DemandModel(object):
    """
    Handles the demand information used by the assessments.

    Parameters
    ----------
    demand_data: DataFrame
        Each column corresponds to a demand type - location - direction and each
        row to a sample.
    error_list: ndarray of bool
        Identifies if there was an error in each simulation that yielded the
        samples in demand_data.
    stripe_list: ndarray of float
        Identifies the stripe (by a scalar number) that the demand belongs to.
        This information is used when evaluating multiple stripes for a
        time-based assessment.

    """

    def __init__(self, raw_data, units):

        self._raw_data = raw_data
        self._units = units

        #initialize flags
        self.error_list = np.zeros(raw_data.shape[0], dtype=bool)
        self.stripe_list = np.ones(raw_data.shape[0])

        self.parse_demands()

        self.convert_units()

    def parse_demands(self):
        """
        Create a DataFrame that holds all relevant demand information.

        """

        # Initialize the list of demand names
        demand_names = []
        demand_ids = []

        for demand_id, name in enumerate(self._raw_data.columns.values):

            # Remove all whitespace to avoid ambiguity
            name = name.replace(' ', '')

            if name not in ['ERROR', 'STRIPE']:

                # Split by the '-' character
                info = name.split('-')

                # The first number (event_ID) is optional and currently not used
                if len(info)==4:
                    info = info[1:]

                if len(info)!=3:
                    raise ValueError(f"Demand name {name} does not follow the "
                                     f"naming convention used in pelicun.")

                demand_names.append(info)
                demand_ids.append(demand_id)

            elif name == 'ERROR':

                self.error_list = (
                    self._raw_data.iloc[:, demand_id].astype(bool))

            elif name == 'STRIPE':

                self.stripe_list = (
                    self._raw_data.iloc[:, demand_id].astype(float))

        # Prepare a MultiIndex for the columns
        demand_names = np.transpose(demand_names)

        demand_types = np.unique(demand_names[0])
        demand_locations = np.unique(demand_names[1])
        demand_directions = np.unique(demand_names[2])

        MI = pd.MultiIndex.from_product(
            [demand_types, demand_locations, demand_directions],
            names = ['type', 'loc', 'dir'])

        # Initialize the demand DF
        demand_data = pd.DataFrame(columns=MI, index=self._raw_data.index,
                                   dtype=float)

        # Store the raw data in the demand DataFrame
        for demand_id, info in zip(demand_ids, demand_names.T):

            demand_data.loc[:,(info[0], info[1], info[2])] = (
                self._raw_data.iloc[:,demand_id].astype(float))

        # Remove the empty columns
        demand_data.dropna(axis=1, how='all', inplace=True)

        self.demand_data = demand_data

    def _get_unit_conversion_scale_factor(self, demand_type):
        """
        Return the scale factor for a particular demand type given the units
        provided by the user.

        """

        # the short demand is the acronym without the specific details that
        # come after the _ character in the name (e.g., SA for SA_1.00)
        short_demand = demand_type.split('_')[0]

        # get the target unit for this demand type
        target_unit = self._units.get(short_demand, 'missing')

        # throw an error if there is no target unit specified
        if target_unit == 'missing':
            raise ValueError(f"No units defined for {demand_type}")

        # scale the values if the unit is not None (e.g. None makes sense
        # for drifts, for example
        if target_unit != None:

            # scale factors are defined in the base module
            # everything is scaled to Standard Units
            scale_factor = globals()[target_unit]

        else:

            scale_factor = 1.0

        return scale_factor


    def convert_units(self):
        """
        Scale the demand values according to the prescribed units

        """

        # get a list of demand types in the model
        demand_type_list = self.demand_data.columns.get_level_values('type').values

        # for each demand type
        for demand_type in set(demand_type_list):

            scale_factor = self._get_unit_conversion_scale_factor(demand_type)

            if scale_factor != 1.0:

                # get the columns in the demand DF that correspond to this
                # demand type and scale the values in those columns
                self.demand_data.loc[:, idx[demand_type, :, :]] *= scale_factor

    def calibrate(self, calibration_settings, remove_errors=True):
        """
        Find the parameters of a probability distribution that describes demands

        """

        def parse_settings(settings, demand_type):

            active_d_types = (
                demand_samples.columns.get_level_values('type').unique())

            if demand_type == 'ALL':
                cols = tuple(active_d_types)

                # the default scale factor is 1.0
                scale_factor = 1.0

            else:
                cols = []

                for d_type in active_d_types:
                    if d_type.split('_')[0] == demand_type:
                        cols.append(d_type)

                cols = tuple(cols)

                # When the demand type is provided we can obtain a demand type
                # specific scale factor
                scale_factor = self._get_unit_conversion_scale_factor(demand_type)

            # load the distribution family
            cal_df.loc['family', idx[cols,:,:]] = settings['DistributionFamily']

            # load the censor limits
            if 'CensorAt' in settings.keys():
                censor_lower, censor_upper = settings['CensorAt']
                cal_df.loc['censor_lower', idx[cols,:,:]] = censor_lower
                cal_df.loc['censor_upper', idx[cols,:,:]] = censor_upper

            # load the truncation limits
            if 'TruncateAt' in settings.keys():
                truncate_lower, truncate_upper = settings['TruncateAt']
                cal_df.loc['truncate_lower', idx[cols,:,:]] = truncate_lower
                cal_df.loc['truncate_upper', idx[cols,:,:]] = truncate_upper

            # scale the censor and truncation limits
            rows_to_scale = ['censor_lower', 'censor_upper',
                             'truncate_lower', 'truncate_upper']
            cal_df.loc[rows_to_scale, idx[cols,:,:]] *= scale_factor

            # load the prescribed additional uncertainty
            if 'AddUncertainty' in settings.keys():

                sig_increase = settings['AddUncertainty']

                # scale the sig value if the target distribution family is normal
                if settings['DistributionFamily'] == 'normal':
                    sig_increase *= scale_factor

                cal_df.loc['sig_increase', idx[cols,:,:]] = sig_increase

        def get_filter_mask(lower_lims, upper_lims):

            demands_of_interest = demand_samples.iloc[:, ~np.isnan(upper_lims)]
            limits_of_interest = upper_lims[~np.isnan(upper_lims)]
            upper_mask = np.all(demands_of_interest < limits_of_interest,
                                axis=1)

            demands_of_interest = demand_samples.iloc[:, ~np.isnan(lower_lims)]
            limits_of_interest = lower_lims[~np.isnan(lower_lims)]
            lower_mask = np.all(demands_of_interest > limits_of_interest,
                                axis=1)

            return np.all([lower_mask, upper_mask], axis=0)

        # start by removing results from erroneous simulations (if needed)
        if remove_errors:
            demand_samples = self.demand_data.loc[~self.error_list,:].copy()
        else:
            demand_samples = self.demand_data.copy()

        errors_removed = np.sum(self.error_list)
        log_msg(f"\nBased on the values in the ERROR column, "
                f"{errors_removed} samples were removed.",
                prepend_timestamp=False)

        # initialize a DataFrame that contains calibration information
        cal_df = pd.DataFrame(
            columns=demand_samples.columns,
            index = ['family',
                     'censor_lower','censor_upper',
                     'truncate_lower','truncate_upper',
                     'sig_increase', 'theta_0', 'theta_1'])

        # start by assigning the default option ('ALL') to every demand column
        parse_settings(calibration_settings['ALL'], 'ALL')

        # then parse the additional settings and make the necessary adjustments
        for demand_type in calibration_settings.keys():
            if demand_type != 'ALL':
                parse_settings(calibration_settings[demand_type], demand_type)

        if options.verbose:
            log_msg(f"\nCalibration settings:\n"+str(cal_df),
                    prepend_timestamp=False)

        # save the settings
        self.model_params = cal_df.copy()

        # Remove those demands that are kept empirical -> i.e., no fitting
        # Currently, empirical demands are decoupled from those that have a
        # distribution fit to their samples. The correlation between empirical
        # and other demands is not preserved in the demand model.
        for col in cal_df.columns:
            if cal_df.loc['family', col] == 'empirical':
                demand_samples.drop(col, 1, inplace=True)
                cal_df.drop(col, 1, inplace=True)

        # Remove the samples outside of censoring limits
        # Currently, non-empirical demands are assumed to have some level of
        # correlation, hence, a censored value in any demand triggers the
        # removal of the entire sample from the population.
        upper_lims = cal_df.loc['censor_upper', :].values.astype(float)
        lower_lims = cal_df.loc['censor_lower', :].values.astype(float)

        censor_mask = get_filter_mask(lower_lims, upper_lims)
        censored_count = np.sum(~censor_mask)

        demand_samples = demand_samples.loc[censor_mask, :]

        log_msg(f"\nBased on the provided censoring limits, "
                f"{censored_count} samples were censored.",
                prepend_timestamp=False)

        # Check if there is any sample outside of truncation limits
        # If yes, that suggest an error either in the samples or the
        # configuration. We handle such errors gracefully: the analysis is not
        # terminated, but we show an error in the log file.
        upper_lims = cal_df.loc['truncate_upper', :].values.astype(float)
        lower_lims = cal_df.loc['truncate_lower', :].values.astype(float)

        truncate_mask = get_filter_mask(lower_lims, upper_lims)
        truncated_count = np.sum(~truncate_mask)

        if truncated_count > 0:

            demand_samples = demand_samples.loc[truncate_mask, :]

            log_msg(f"\nBased on the provided truncation limits, "
                    f"{truncated_count} samples were removed before demand "
                    f"calibration.",
                    prepend_timestamp=False)

        if options.verbose:
            log_msg(f"\nDemand data used for calibration:\n"+str(demand_samples),
                    prepend_timestamp=False)

        # fit the joint distribution
        demand_theta, demand_rho = fit_distribution(
            raw_samples = demand_samples.values.T,
            distribution = cal_df.loc['family',:].values,
            censored_count = censored_count,
            detection_limits = np.array(
                [cal_df.loc['censor_lower',:].values,
                 cal_df.loc['censor_upper',:].values], dtype=float),
            truncation_limits = np.array(
                [cal_df.loc['truncate_lower',:].values,
                 cal_df.loc['truncate_upper',:].values], dtype=float),
            multi_fit=False
        )

        # save the calibration results
        self.model_params.loc[['theta_0','theta_1'], cal_df.columns] = (
            demand_theta.T
        )

        # increase the variance of the marginal distributions, if needed
        sig_inc = np.nan_to_num(
            self.model_params.loc['sig_increase', :].values.astype(float))
        sig_0 = self.model_params.loc['theta_1', :].values.astype(float)

        self.model_params.loc['theta_1', :] = (
            np.sqrt(sig_0 ** 2. + sig_inc ** 2.))

        log_msg(f"\nDemand model parameters:\n"+str(self.model_params),
                prepend_timestamp=False)

        # save the correlation matrix
        self.model_rho = pd.DataFrame(demand_rho,
                                      columns = cal_df.columns,
                                      index = cal_df.columns)

        log_msg(f"\nDemand model correlation matrix:\n" +
                str(self.model_rho),
                prepend_timestamp=False)


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
        samples = pd.DataFrame(dict([(lim_i.name, lim_i.samples)
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
                samples = pd.Series(self._DV_distribution.samples).loc[quantity.index]
            else:
                samples = pd.Series(self._DV_distribution.samples).iloc[:sample_size]
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