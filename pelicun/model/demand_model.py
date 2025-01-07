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


"""DemandModel object and associated methods."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numexpr as ne
import numpy as np
import pandas as pd

from pelicun import base, file_io, uq
from pelicun.model.pelicun_model import PelicunModel

if TYPE_CHECKING:
    from pelicun.assessment import AssessmentBase

idx = base.idx


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
    calibrated: bool
        Signifies whether the DemandModel object has been calibrated.

    """

    __slots__ = [
        '_RVs',
        'calibrated',
        'correlation',
        'empirical_data',
        'marginal_params',
        'sample',
        'user_units',
    ]

    def __init__(self, assessment: AssessmentBase) -> None:
        """
        Instantiate a DemandModel.

        Parameters
        ----------
        assessment: Assessment
            Parent assessment object.

        """
        super().__init__(assessment)

        self.marginal_params: pd.DataFrame | None = None
        self.correlation: pd.DataFrame | None = None
        self.empirical_data: pd.DataFrame | None = None
        self.user_units: pd.Series | None = None
        self.calibrated = False

        self._RVs: uq.RandomVariableRegistry | None = None
        self.sample: pd.DataFrame | None = None

    @overload
    def save_sample(
        self, filepath: None = None, *, save_units: bool = False
    ) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame: ...

    @overload
    def save_sample(self, filepath: str, *, save_units: bool = False) -> None: ...

    def save_sample(
        self, filepath: str | None = None, *, save_units: bool = False
    ) -> None | tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        """
        Save demand sample to a csv file or return it in a DataFrame.

        Returns
        -------
        None or tuple
            If `filepath` is specified, the function saves the demand
            sample to a CSV file and returns None.
            If `filepath` is not specified, it returns the DataFrame
            containing the demand sample.
            If `save_units` is True, it returns a tuple of the
            DataFrame and a Series containing the units.

        """
        self.log.div()
        if filepath is not None:
            self.log.msg('Saving demand sample...')

        assert self.sample is not None
        res = file_io.save_to_csv(
            self.sample,
            Path(filepath) if filepath is not None else None,
            units=self.user_units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )
        if filepath is not None:
            self.log.msg(
                'Demand sample successfully saved.', prepend_timestamp=False
            )
            return None

        # else:
        assert isinstance(res, pd.DataFrame)

        units = res.loc['Units']
        res = res.drop('Units')
        assert isinstance(units, pd.Series)

        if save_units:
            return res.astype(float), units

        # else:
        return res.astype(float)

    def load_sample(self, filepath: str | pd.DataFrame) -> None:
        """
        Load demand sample data and parse it.

        Besides parsing the sample, the method also reads and saves
        the units specified for each demand variable. If no units are
        specified, base units are assumed.

        Parameters
        ----------
        filepath: string or DataFrame
            Location of the file with the demand sample.

        """

        def parse_header(raw_header: pd.Index[str]) -> pd.Index[str]:
            """
            Parse and clean header.

            Parses and cleans the header of a demand DataFrame from
            raw multi-level index to a standardized format.

            This function adjusts the raw header of a DataFrame,
            removing optional event IDs and whitespace, and
            standardizing the format to facilitate further
            processing. It is designed to handle headers with either
            three or four levels, where the first level (event_ID) is
            optional and not used in further analysis.  Note: This
            will soon change, and that first level will be enforced
            instead of removed.

            Parameters
            ----------
            raw_header: pd.MultiIndex
                The original multi-level index (header) of the
                DataFrame, which may contain an optional event_ID and
                might have excess whitespace in the labels.

            Returns
            -------
            pd.MultiIndex
                A new MultiIndex for the DataFrame's columns that is
                cleaned of any unwanted characters or levels. This
                index has three levels: 'type', 'loc', and 'dir',
                representing the type of demand, location, and
                direction, respectively.

            """
            old_mi = raw_header

            # The first number (event_ID) in the demand labels is optional and
            # currently not used. We remove it if it was in the raw data.
            num_levels_with_event_id = 4
            if old_mi.nlevels == num_levels_with_event_id:
                if self._asmnt.log.verbose:
                    self.log.msg(
                        'Removing event_ID from header...', prepend_timestamp=False
                    )

                new_column_index_array = np.array(
                    [old_mi.get_level_values(i) for i in range(1, 4)]
                )

            else:
                new_column_index_array = np.array(
                    [old_mi.get_level_values(i) for i in range(3)]
                )

            # Remove whitespace to avoid ambiguity

            if self._asmnt.log.verbose:
                self.log.msg(
                    'Removing whitespace from header...', prepend_timestamp=False
                )

            wspace_remove = np.vectorize(lambda name: str(name).replace(' ', ''))

            new_column_index = wspace_remove(new_column_index_array)

            # Creating new, cleaned-up header

            return pd.MultiIndex.from_arrays(
                new_column_index, names=['type', 'loc', 'dir']
            )

        self.log.div()
        self.log.msg('Loading demand data...')

        demand_data, units = file_io.load_data(
            filepath,
            self._asmnt.unit_conversion_factors,
            return_units=True,
            log=self._asmnt.log,
        )
        assert isinstance(demand_data, pd.DataFrame)
        assert isinstance(units, pd.Series)

        parsed_data = demand_data.copy()

        # start with cleaning up the header

        parsed_data.columns = parse_header(parsed_data.columns)

        # Remove errors, if needed
        if 'ERROR' in parsed_data.columns.get_level_values(0):
            self.log.msg(
                'Removing errors from the raw data...', prepend_timestamp=False
            )

            error_list = (
                parsed_data.loc[  # type: ignore
                    :,  # type: ignore
                    idx['ERROR', :, :],  # type: ignore
                ]
                .to_numpy()
                .astype(  # type: ignore
                    bool  # type: ignore
                )
            )  # type: ignore

            parsed_data = parsed_data.loc[~error_list, :].copy()
            parsed_data = parsed_data.drop('ERROR', level=0, axis=1)

            self.log.msg(
                '\nBased on the values in the ERROR column, '
                f'{np.sum(error_list)} demand samples were removed.\n',
                prepend_timestamp=False,
            )

        self.sample = parsed_data

        self.log.msg('Demand data successfully parsed.', prepend_timestamp=False)

        # parse the index for the units
        units.index = parse_header(units.index)

        self.user_units = units

        self.log.msg('Demand units successfully parsed.', prepend_timestamp=False)

    def estimate_RID(  # noqa: N802
        self,
        demands: pd.DataFrame | pd.Series,
        params: dict,
        method: str = 'FEMA P58',
    ) -> pd.DataFrame:
        """
        Estimate residual inter-story drift (RID).

        Estimates residual inter-story drift (RID) realizations based
        on peak inter-story drift (PID) and other demand parameters
        using specified methods.

        This method calculates RID based on the peak inter-story drift
        provided in the demands DataFrame and parameters such as yield
        drift specified in the params dictionary. The calculation
        adheres to the FEMA P-58 methodology, which includes
        conditions for different ranges of drift.

        Parameters
        ----------
        demands: DataFrame
            A DataFrame containing samples of demands, specifically
            peak inter-story drift (PID) values for various
            location-direction pairs required for the estimation
            method.
        params: dict
            A dictionary containing parameters required for the
            estimation method, such as 'yield_drift', which is the
            drift at which yielding is expected to occur.
        method: str, optional
            The method used to estimate the RID values. Currently,
            only 'FEMA P58' is implemented. Defaults to 'FEMA P58'.

        Returns
        -------
        DataFrame
            A DataFrame containing the estimated residual inter-story
            drift (RID) realizations, indexed and structured similarly
            to the input demands DataFrame.

        Raises
        ------
        ValueError
            Raises a ValueError if an unrecognized method is provided
            or required parameters are missing in the `params`
            dictionary.

        Notes
        -----
        The FEMA P-58 estimation approach divides the drift into three
        domains, with different transformation rules for
        each. Additional stochastic variation is introduced to nonzero
        RID values to model the inherent uncertainty. The method
        ensures that the RID values do not exceed the corresponding
        PID values.

        """
        if method in {'FEMA P58', 'FEMA P-58'}:
            # method is described in FEMA P-58 Volume 1 Section 5.4 &
            # Appendix C

            # the provided demands shall be PID values at various
            # loc-dir pairs
            pid = demands

            # there's only one parameter needed: the yield drift
            yield_drift = params['yield_drift']

            # three subdomains of demands are identified
            small = yield_drift > pid
            medium = 4 * yield_drift > pid
            large = 4 * yield_drift <= pid

            # convert PID to RID in each subdomain
            rid = pid.copy()
            rid[large] = pid[large] - 3 * yield_drift
            rid[medium] = 0.3 * (pid[medium] - yield_drift)
            rid[small] = 0.0

            # add extra uncertainty to nonzero values
            rng = self._asmnt.options.rng
            eps = rng.normal(scale=0.2, size=rid.shape)
            rid[rid > 0] = np.exp(np.log(rid[rid > 0]) + eps)  # type: ignore

            # finally, make sure the RID values are never larger than
            # the PIDs
            rid = pd.DataFrame(
                np.minimum(pid.values, rid.values),  # type: ignore
                columns=pd.DataFrame(  # noqa: PD013
                    1,
                    index=['RID'],
                    columns=pid.columns,
                )
                .stack(level=[0, 1])
                .index,
                index=pid.index,
            )

        else:
            msg = f'Invalid method: `{method}`.'
            raise ValueError(msg)

        return rid

    def estimate_RID_and_adjust_sample(  # noqa: N802
        self, params: dict, method: str = 'FEMA P58'
    ) -> None:
        """
        Estimate residual inter-story drift (RID) and modifies sample.

        Uses `self.estimate_RID` and adjusts the demand sample.
        See the docstring of the `estimate_RID` method for details.

        Parameters
        ----------
        params: dict
            A dictionary containing parameters required for the
            estimation method, such as 'yield_drift', which is the
            drift at which yielding is expected to occur.
        method: str, optional
            The method used to estimate the RID values. Currently,
            only 'FEMA P58' is implemented. Defaults to 'FEMA P58'.

        Raises
        ------
        ValueError
            If the method is called before a sample is generated.

        """
        if self.sample is None:
            msg = 'Demand model does not have a sample yet.'
            raise ValueError(msg)

        sample_tuple = self.save_sample(save_units=True)
        assert isinstance(sample_tuple, tuple)
        demand_sample, demand_units = sample_tuple
        assert isinstance(demand_sample, pd.DataFrame)
        assert isinstance(demand_units, pd.Series)
        pid = demand_sample['PID']
        rid = self.estimate_RID(pid, params, method)
        rid_units = pd.Series('unitless', index=rid.columns)
        demand_sample_ext = pd.concat([demand_sample, rid], axis=1)
        units_ext = pd.concat([demand_units, rid_units])
        demand_sample_ext.loc['Units', :] = units_ext
        self.load_sample(demand_sample_ext)

    def expand_sample(
        self,
        label: str,
        value: float | np.ndarray,
        unit: str,
        location: str = '0',
        direction: str = '1',
    ) -> None:
        """
        Add an extra column to the demand sample.

        The column contains repeated instances of `value`, is accessed
        via the multi-index (`label`-`location`-`direction`), and has
        units of `unit`.

        Parameters
        ----------
        label: str
            Label to use to extend the MultiIndex of the demand sample.
        value: float | np.ndarray
            Values to add to the rows of the additional column.
        unit: str
            Unit that corresponds to the additional column.
        location: str, optional
            Optional location, defaults to `0`.
        direction: str, optional
            Optional direction, defaults to `1`.

        Raises
        ------
        ValueError
            If the method is called before a sample is generated.
        ValueError
            If `value` is a numpy array of incorrect shape.

        """
        if self.sample is None:
            msg = 'Demand model does not have a sample yet.'
            raise ValueError(msg)
        sample_tuple = self.save_sample(save_units=True)
        assert isinstance(sample_tuple, tuple)
        demand_sample, demand_units = sample_tuple
        assert isinstance(demand_sample, pd.DataFrame)
        assert isinstance(demand_units, pd.Series)
        if isinstance(value, np.ndarray) and len(value) != len(demand_sample):
            msg = 'Incompatible array length.'
            raise ValueError(msg)
        demand_sample[label, location, direction] = value
        demand_units[label, location, direction] = unit
        demand_sample.loc['Units', :] = demand_units
        self.load_sample(demand_sample)

    def calibrate_model(self, config: dict) -> None:  # noqa: C901
        """
        Calibrate a demand model to describe the raw demand data.

        The raw data shall be parsed first to ensure that it follows the
        schema expected by this method. The calibration settings define the
        characteristics of the multivariate distribution that is fit to the
        raw data.

        Parameters
        ----------
        config: dict
            A dictionary, typically read from a JSON file, that specifies the
            distribution family, truncation and censoring limits, and other
            settings for the calibration.

        """
        if self.calibrated:
            self.log.warning('DemandModel has been previously calibrated.')

        def parse_settings(  # noqa: C901
            cal_df: pd.DataFrame, settings: dict, demand_type: str
        ) -> None:
            def parse_str_to_float(in_str: str, context_string: str) -> float:
                try:
                    out_float = (
                        np.nan if base.check_if_str_is_na(in_str) else float(in_str)
                    )

                except ValueError:
                    self.log.warning(
                        f'Could not parse {in_str} provided as '
                        f'{context_string}. Using NaN instead.'
                    )

                    out_float = np.nan

                return out_float

            demand_sample = self.save_sample()
            assert isinstance(demand_sample, pd.DataFrame)
            active_d_types = demand_sample.columns.get_level_values('type').unique()

            if demand_type == 'ALL':
                cols = tuple(active_d_types)

            else:
                cols_lst = []

                for d_type in active_d_types:
                    if d_type.split('_')[0] == demand_type:
                        cols_lst.append(d_type)  # noqa: PERF401

                cols = tuple(cols_lst)

            # load the distribution family
            cal_df.loc[list(cols), 'Family'] = settings['DistributionFamily']

            # load limits
            for lim in (
                'CensorLower',
                'CensorUpper',
                'TruncateLower',
                'TruncateUpper',
            ):
                if lim in settings:
                    val = parse_str_to_float(settings[lim], lim)
                    if not pd.isna(val):
                        cal_df.loc[list(cols), lim] = val

            # scale the censor and truncation limits, if needed
            scale_factor = self._asmnt.scale_factor(settings.get('Unit'))

            rows_to_scale = [
                'CensorLower',
                'CensorUpper',
                'TruncateLower',
                'TruncateUpper',
            ]
            cal_df.loc[idx[cols, :, :], rows_to_scale] *= scale_factor  # type: ignore

            # load the prescribed additional uncertainty
            if 'AddUncertainty' in settings:
                sig_increase = parse_str_to_float(
                    settings['AddUncertainty'], 'AddUncertainty'
                )

                # scale the sig value if the target distribution family is normal
                if settings['DistributionFamily'] == 'normal':
                    sig_increase *= scale_factor

                cal_df.loc[list(cols), 'SigIncrease'] = sig_increase

        def get_filter_mask(
            demand_sample: pd.DataFrame,
            lower_lims: np.ndarray,
            upper_lims: np.ndarray,
        ) -> bool:
            demands_of_interest = demand_sample.iloc[:, pd.notna(upper_lims)]
            limits_of_interest = upper_lims[pd.notna(upper_lims)]
            upper_mask = np.all(demands_of_interest < limits_of_interest, axis=1)

            demands_of_interest = demand_sample.iloc[:, pd.notna(lower_lims)]
            limits_of_interest = lower_lims[pd.notna(lower_lims)]
            lower_mask = np.all(demands_of_interest > limits_of_interest, axis=1)

            return np.all([lower_mask, upper_mask], axis=0)

        self.log.div()
        self.log.msg('Calibrating demand model...')

        demand_sample = self.sample
        assert isinstance(demand_sample, pd.DataFrame)

        # initialize a DataFrame that contains calibration information
        cal_df = pd.DataFrame(
            columns=[
                'Family',
                'CensorLower',
                'CensorUpper',
                'TruncateLower',
                'TruncateUpper',
                'SigIncrease',
                'Theta_0',
                'Theta_1',
            ],
            index=demand_sample.columns,
            dtype=float,
        )

        cal_df['Family'] = cal_df['Family'].astype(str)

        # start by assigning the default option ('ALL') to every demand column
        parse_settings(cal_df, config['ALL'], demand_type='ALL')

        # then parse the additional settings and make the necessary adjustments
        for demand_type in config:  # noqa: PLC0206
            if demand_type != 'ALL':
                parse_settings(cal_df, config[demand_type], demand_type=demand_type)

        if self._asmnt.log.verbose:
            self.log.msg(
                '\nCalibration settings successfully parsed:\n' + str(cal_df),
                prepend_timestamp=False,
            )
        else:
            self.log.msg(
                '\nCalibration settings successfully parsed:\n',
                prepend_timestamp=False,
            )

        # save the settings
        model_params = cal_df.copy()

        # Remove the samples outside of censoring limits
        # Currently, non-empirical demands are assumed to have some level of
        # correlation, hence, a censored value in any demand triggers the
        # removal of the entire sample from the population.
        upper_lims = cal_df.loc[:, 'CensorUpper'].to_numpy()
        lower_lims = cal_df.loc[:, 'CensorLower'].to_numpy()

        assert isinstance(demand_sample, pd.DataFrame)
        if ~np.all(pd.isna(np.array([upper_lims, lower_lims]))):
            censor_mask = get_filter_mask(demand_sample, lower_lims, upper_lims)
            censored_count = np.sum(~censor_mask)

            demand_sample = pd.DataFrame(demand_sample.loc[censor_mask, :])

            self.log.msg(
                '\nBased on the provided censoring limits, '
                f'{censored_count} samples were censored.',
                prepend_timestamp=False,
            )
        else:
            censored_count = 0

        # Check if there is any sample outside of truncation limits
        # If yes, that suggests an error either in the samples or the
        # configuration. We handle such errors gracefully: the analysis is not
        # terminated, but we show an error in the log file.
        upper_lims = cal_df.loc[:, 'TruncateUpper'].to_numpy()
        lower_lims = cal_df.loc[:, 'TruncateLower'].to_numpy()

        assert isinstance(demand_sample, pd.DataFrame)
        if ~np.all(pd.isna(np.array([upper_lims, lower_lims]))):
            truncate_mask = get_filter_mask(demand_sample, lower_lims, upper_lims)
            truncated_count = np.sum(~truncate_mask)

            if truncated_count > 0:
                demand_sample = pd.DataFrame(demand_sample.loc[truncate_mask, :])

                self.log.msg(
                    '\nBased on the provided truncation limits, '
                    f'{truncated_count} samples were removed before demand '
                    'calibration.',
                    prepend_timestamp=False,
                )

        # Separate and save the demands that are kept empirical -> i.e., no
        # fitting. Currently, empirical demands are decoupled from those that
        # have a distribution fit to their samples. The correlation between
        # empirical and other demands is not preserved in the demand model.
        empirical_edps = []
        for edp in cal_df.index:
            if cal_df.loc[edp, 'Family'] == 'empirical':
                empirical_edps.append(edp)  # noqa: PERF401

        assert isinstance(demand_sample, pd.DataFrame)
        if empirical_edps:
            self.empirical_data = demand_sample.loc[:, empirical_edps].copy()

        # remove the empirical demands from the samples used for calibration
        demand_sample = demand_sample.drop(empirical_edps, axis=1)

        # and the calibration settings
        cal_df = cal_df.drop(empirical_edps, axis=0)

        if self._asmnt.log.verbose:
            self.log.msg(
                f'\nDemand data used for calibration:\n{demand_sample}',
                prepend_timestamp=False,
            )

        # fit the joint distribution
        self.log.msg(
            '\nFitting the prescribed joint demand distribution...',
            prepend_timestamp=False,
        )

        demand_theta, demand_rho = uq.fit_distribution_to_sample(
            raw_sample=demand_sample.to_numpy().T,
            distribution=cal_df.loc[:, 'Family'].values,  # type: ignore
            censored_count=censored_count,
            detection_limits=cal_df.loc[  # type: ignore
                :,
                ['CensorLower', 'CensorUpper'],
            ].values,
            truncation_limits=cal_df.loc[  # type: ignore
                :, ['TruncateLower', 'TruncateUpper']
            ].values,
            multi_fit=False,
            logger_object=self._asmnt.log,
        )
        # fit the joint distribution
        self.log.msg(
            '\nCalibration successful, processing results...',
            prepend_timestamp=False,
        )

        # save the calibration results
        model_params.loc[cal_df.index, ['Theta_0', 'Theta_1']] = demand_theta

        # increase the variance of the marginal distributions, if needed
        if ~np.all(pd.isna(model_params.loc[:, 'SigIncrease'].values)):
            self.log.msg('\nIncreasing demand variance...', prepend_timestamp=False)

            sig_inc = np.nan_to_num(
                model_params.loc[:, 'SigIncrease'].values,  # type: ignore
            )
            sig_0 = model_params.loc[:, 'Theta_1'].to_numpy()

            model_params.loc[:, 'Theta_1'] = np.sqrt(
                sig_0**2.0 + sig_inc**2.0,  # type: ignore
            )

        # remove unneeded fields from model_params
        for col in ('SigIncrease', 'CensorLower', 'CensorUpper'):
            model_params = model_params.drop(col, axis=1)

        # reorder the remaining fields for clarity
        model_params = model_params[
            ['Family', 'Theta_0', 'Theta_1', 'TruncateLower', 'TruncateUpper']
        ]

        self.marginal_params = model_params

        self.log.msg(
            '\nCalibrated demand model marginal distributions:\n'
            + str(model_params),
            prepend_timestamp=False,
        )

        # save the correlation matrix
        self.correlation = pd.DataFrame(
            demand_rho, columns=cal_df.index, index=cal_df.index
        )

        self.log.msg(
            '\nCalibrated demand model correlation matrix:\n'
            + str(self.correlation),
            prepend_timestamp=False,
        )

        self.calibrated = True

    def save_model(self, file_prefix: str) -> None:
        """Save parameters of the demand model to a set of csv files."""
        self.log.div()
        self.log.msg('Saving demand model...')

        # save the correlation and empirical data
        file_io.save_to_csv(self.correlation, Path(file_prefix + '_correlation.csv'))
        if self.empirical_data is not None:
            file_io.save_to_csv(
                self.empirical_data,
                Path(file_prefix + '_empirical.csv'),
                units=self.user_units,
                unit_conversion_factors=self._asmnt.unit_conversion_factors,
                log=self._asmnt.log,
            )

        # Converting the marginal parameters requires special
        # treatment, so we can't rely on file_io's universal unit
        # conversion functionality. We do it manually here instead.
        assert isinstance(self.marginal_params, pd.DataFrame)
        assert isinstance(self.user_units, pd.Series)
        marginal_params_user_units = self._convert_marginal_params(
            self.marginal_params.copy(), self.user_units, inverse_conversion=True
        )
        marginal_params_user_units['Units'] = self.user_units

        file_io.save_to_csv(
            marginal_params_user_units,
            Path(file_prefix + '_marginals.csv'),
            orientation=1,
            log=self._asmnt.log,
        )

        self.log.msg('Demand model successfully saved.', prepend_timestamp=False)

    def load_model(self, data_source: str | dict) -> None:
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

        Raises
        ------
        TypeError
            If the data source type is invalid.

        """
        self.log.div()
        self.log.msg('Loading demand model...')

        # prepare the marginal data source variable to load the data
        if isinstance(data_source, dict):
            marginal_data_source = data_source.get('marginals')
            assert isinstance(marginal_data_source, pd.DataFrame)
            empirical_data_source = data_source.get('empirical', None)
            correlation_data_source = data_source.get('correlation', None)
        elif isinstance(data_source, str):
            marginal_data_source = data_source + '_marginals.csv'
            empirical_data_source = data_source + '_empirical.csv'
            correlation_data_source = data_source + '_correlation.csv'
        else:
            msg = f'Invalid data_source type: {type(data_source)}.'
            raise TypeError(msg)

        if empirical_data_source is not None:
            if (
                isinstance(empirical_data_source, str)
                and Path(empirical_data_source).exists()
            ):
                empirical_data = file_io.load_data(
                    empirical_data_source,
                    self._asmnt.unit_conversion_factors,
                    log=self._asmnt.log,
                )
                assert isinstance(empirical_data, pd.DataFrame)
                self.empirical_data = empirical_data
                self.empirical_data.columns.names = ['type', 'loc', 'dir']
        else:
            self.empirical_data = None

        if correlation_data_source is not None:
            correlation = file_io.load_data(
                correlation_data_source,
                self._asmnt.unit_conversion_factors,
                reindex=False,
                log=self._asmnt.log,
            )
            assert isinstance(correlation, pd.DataFrame)
            self.correlation = correlation
            self.correlation.index = self.correlation.index.set_names(
                ['type', 'loc', 'dir']
            )
            self.correlation.columns = self.correlation.columns.set_names(
                ['type', 'loc', 'dir']
            )
        else:
            self.correlation = None

        # Note that a data source without marginal information is not valid
        marginal_params, units = file_io.load_data(
            marginal_data_source,
            None,
            orientation=1,
            reindex=False,
            return_units=True,
            log=self._asmnt.log,
        )
        assert isinstance(marginal_params, pd.DataFrame)
        assert isinstance(units, pd.Series)

        marginal_params.index = marginal_params.index.set_names(
            ['type', 'loc', 'dir']
        )

        marginal_params = self._convert_marginal_params(
            marginal_params.copy(), units
        )

        self.marginal_params = marginal_params
        self.user_units = units

        self.log.msg('Demand model successfully loaded.', prepend_timestamp=False)

    def _create_RVs(self, *, preserve_order: bool = False) -> None:  # noqa: N802
        """Create a random variable registry for the joint distribution of demands."""
        assert self.marginal_params is not None

        # initialize the registry
        rv_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        # add a random variable for each demand variable
        for rv_params in self.marginal_params.itertuples():
            edp = rv_params.Index
            rv_tag = f'EDP-{edp[0]}-{edp[1]}-{edp[2]}'  # type: ignore
            family = getattr(rv_params, 'Family', 'deterministic')

            if family == 'empirical':
                dist_family = 'coupled_empirical' if preserve_order else 'empirical'

                # empirical RVs need the data points
                rv_reg.add_RV(
                    uq.rv_class_map(dist_family)(
                        name=rv_tag,
                        theta=self.empirical_data.loc[  # type: ignore
                            :,  # type: ignore
                            edp,
                        ].values,
                    )
                )

            else:
                # all other RVs need parameters of their distributions
                rv_reg.add_RV(
                    uq.rv_class_map(family)(  # type: ignore
                        name=rv_tag,
                        theta=np.array(
                            [
                                value
                                for t_i in range(3)
                                if (
                                    value := getattr(rv_params, f'Theta_{t_i}', None)
                                )
                                is not None
                            ]
                        ),
                        truncation_limits=np.array(
                            [
                                getattr(rv_params, f'Truncate{side}', np.nan)
                                for side in ('Lower', 'Upper')
                            ]
                        ),
                    )
                )

        self.log.msg(
            f'\n{self.marginal_params.shape[0]} random variables created.',
            prepend_timestamp=False,
        )

        # add an RV set to consider the correlation between demands, if needed
        if self.correlation is not None:
            rv_set_tags = [
                f'EDP-{edp[0]}-{edp[1]}-{edp[2]}'
                for edp in self.correlation.index.to_numpy()
            ]

            rv_reg.add_RV_set(
                uq.RandomVariableSet(
                    'EDP_set',
                    list(rv_reg.RVs(rv_set_tags).values()),
                    self.correlation.values,
                )
            )

            self.log.msg(
                f'\nCorrelations between {len(rv_set_tags)} random variables '
                'successfully defined.',
                prepend_timestamp=False,
            )

        self._RVs = rv_reg

    def clone_demands(self, demand_cloning: dict) -> None:
        """
        Clone demands.

        Copies over columns of the original demand sample and
        assigns given names to them. The columns to be copied over
        and the names to be assigned to the copies are defined as the keys
        and values of the `demand_cloning` dictionary.
        The method modifies `sample` inplace.

        Parameters
        ----------
        demand_cloning: dict
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
            In multiple instances of invalid demand_cloning entries.

        """
        # it's impossible to have duplicate keys, because
        # demand_cloning is a dictionary.
        new_columns_list = demand_cloning.values()
        # The following prevents duplicate entries in the values
        # corresponding to a single cloned demand (1), but
        # also the same column being specified as the cloned
        # entry of multiple demands (2).
        # e.g.
        # (1): {'PGV-0-1': ['PGV-1-1', 'PGV-1-1', ...]}
        # (2): {'PGV-0-1': ['PGV-1-1', ...], 'PGV-0-2': ['PGV-1-1', ...]}
        flat_list = []
        for new_columns in new_columns_list:
            flat_list.extend(new_columns)
        if len(set(flat_list)) != len(flat_list):
            msg = 'Duplicate entries in demand cloning configuration.'
            raise ValueError(msg)

        # turn the config entries to tuples
        def turn_to_tuples(demand_cloning: dict) -> dict:
            demand_cloning_tuples = {}
            for key, values in demand_cloning.items():
                demand_cloning_tuples[tuple(key.split('-'))] = [
                    tuple(x.split('-')) for x in values
                ]
            return demand_cloning_tuples

        demand_cloning = turn_to_tuples(demand_cloning)

        # The demand cloning configuration should not include
        # columns that are not present in the original sample.
        warn_columns = []
        assert self.sample is not None
        for column in demand_cloning:
            if column not in self.sample.columns:
                warn_columns.append(column)  # noqa: PERF401
        if warn_columns:
            warn_columns = ['-'.join(x) for x in warn_columns]
            self.log.warning(
                'The demand cloning configuration lists '
                "columns that are not present in the original demand sample's "
                f'columns: {warn_columns}.'
            )

        # we iterate over the existing columns of the sample and try
        # to locate columns that need to be copied as required by the
        # demand cloning configuration.  If a column does not need
        # to be cloned it is left as is.  Otherwise, we keep track
        # of its initial index location (in `column_index`) and the
        # number of times it needs to be replicated, along with the
        # new names of its copies (in `column_values`).
        column_index = []
        column_values = []
        for i, column in enumerate(self.sample.columns):
            if column not in demand_cloning:
                column_index.append(i)
                column_values.append(column)
            else:
                new_column_values = demand_cloning[column]
                column_index.extend([i] * len(new_column_values))
                column_values.extend(new_column_values)
        # copy the columns
        self.sample = self.sample.iloc[:, column_index]
        # update the column index
        self.sample.columns = pd.MultiIndex.from_tuples(column_values)
        # update units
        self.user_units = self.user_units.iloc[column_index]  # type: ignore
        assert self.user_units is not None
        self.user_units.index = self.sample.columns

    def generate_sample(self, config: dict) -> None:
        """
        Generate the demand sample.

        Generates a sample of random variables (RVs) based on the
        specified configuration for demand modeling.

        This method utilizes the current settings for marginal
        distribution parameters to generate a sample of demand
        variables. The configuration can specify details such as the
        sample size, whether to preserve the order of raw data, and
        whether to apply demand cloning. The generated sample is
        stored internally and can be used for subsequent analysis.

        Parameters
        ----------
        config: dict
            A dictionary containing configuration options for the
            sample generation. Key options include:
            * 'SampleSize': The number of samples to generate.
            * 'PreserveRawOrder': Boolean indicating whether to
            preserve the order of the raw data. Defaults to False.
            * 'DemandCloning': Specifies if and how demand cloning
            should be applied. Can be a boolean or a detailed
            configuration.

        Raises
        ------
        ValueError
            If model parameters are not loaded or specified before
            attempting to generate a sample.

        Notes
        -----
        The function is responsible for the creation of random
        variables based on the distribution parameters specified in
        `marginal_params`.  It ensures that the sample is properly
        indexed and sorted according to type, location, and
        direction. It also handles the configuration of demand cloning
        if specified, which is a method to artificially augment the
        variability in the sample based on existing realizations.

        Examples
        --------
        >>> config = {
                'SampleSize': 1000,
                'PreserveRawOrder': True,
                'DemandCloning': False
            }
        >>> model.generate_sample(config)
        # This will generate 1000 realizations of demand variables
        # with the specified configuration.

        """
        if self.marginal_params is None:
            msg = (
                'Model parameters have not been specified. Either '
                'load parameters from a file or calibrate the '
                'model using raw demand data.'
            )
            raise ValueError(msg)

        self.log.div()
        self.log.msg('Generating sample from demand variables...')

        self._create_RVs(preserve_order=config.get('PreserveRawOrder', False))

        assert self._RVs is not None
        assert self._asmnt.options.sampling_method is not None
        sample_size = config['SampleSize']
        self._RVs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        # replace the potentially existing raw sample with the generated one
        assert self._RVs is not None
        assert self._RVs.RV_sample is not None
        sample = pd.DataFrame(self._RVs.RV_sample)
        sample = sample.sort_index(axis=0)
        sample = sample.sort_index(axis=1)

        sample_mi = base.convert_to_MultiIndex(sample, axis=1)['EDP']
        assert isinstance(sample_mi, pd.DataFrame)
        sample = sample_mi

        sample.columns.names = ['type', 'loc', 'dir']
        self.sample = sample

        if config.get('DemandCloning', False):
            self.clone_demands(config['DemandCloning'])

        self.log.msg(
            f'\nSuccessfully generated {sample_size} realizations.',
            prepend_timestamp=False,
        )


def _get_required_demand_type(
    model_parameters: pd.DataFrame,
    pgb: pd.DataFrame,
    demand_offset: dict | None = None,
) -> dict:
    """
    Get the required demand type for the components.

    Returns the demand type and its properties required to calculate
    the the damage or loss of a component. The properties include
    whether the demand is directional, the offset, and the type of the
    demand. The method takes as input a dataframe `PGB` that contains
    information about the component groups in the asset. For each
    performance group PG in the PGB dataframe, the method retrieves
    the relevant parameters from the model_params dataframe and parses
    the demand type into its properties. If the demand type has a
    subtype, the method splits it and adds the subtype to the demand
    type to form the EDP type. The method also considers the default
    offset for the demand type, if it is specified in the options
    attribute of the assessment, and adds the offset to the EDP. If
    the demand is directional, the direction is added to the EDP. The
    method collects all the unique EDPs for each component group and
    returns them as a dictionary where each key is an EDP and its
    value is a list of component groups that require that EDP.

    Parameters
    ----------
    model_parameters: pd.DataFrame
        Model parameters. Damage model parameters, or
        loss-function loss model parameters.
    pgb: pd.DataFrame
        A pandas DataFrame with the block information for
        each component
    demand_offset: dict, optional
        Specifies an additional location offset for specific
        demand types. Example:
        {'PFA': -1, 'PFV': +2}.

    Returns
    -------
    dict
        A dictionary of EDP requirements, where each key is the EDP
        string (e.g., "PGA-0-1"), and the
        corresponding value is a list of tuples (component_id,
        location, direction)

    Raises
    ------
    ValueError
        When a negative value is used for `loc`. Currently not
        supported.

    """
    model_parameters = model_parameters.sort_index(axis=1)

    # Assign default demand_offset to empty dict.
    if not demand_offset:
        demand_offset = {}

    required_edps = defaultdict(list)

    for pg in pgb.index:
        cmp = pg[0]

        # Utility Demand: if there is an `Expression`, then load the
        # rest of the demand types.
        expression = model_parameters.loc[cmp, :].get(('Demand', 'Expression'))
        if expression is not None:
            # get the number of variables in the expression using
            # the numexpr library
            parsed_expr = ne.NumExpr(_clean_up_expression(expression))
            num_terms = len(parsed_expr.input_names)
            demand_parameters_list = []
            for i in range(num_terms):
                if i == 0:
                    index_lvl0 = 'Demand'
                else:
                    index_lvl0 = f'Demand{i + 1}'
                demand_parameters_list.append(
                    (
                        model_parameters.loc[cmp, (index_lvl0, 'Type')],
                        model_parameters.loc[cmp, (index_lvl0, 'Offset')],
                        model_parameters.loc[cmp, (index_lvl0, 'Directional')],
                    )
                )
        else:
            demand_parameters_list = [
                (
                    model_parameters.loc[cmp, ('Demand', 'Type')],
                    model_parameters.loc[cmp, ('Demand', 'Offset')],
                    model_parameters.loc[cmp, ('Demand', 'Directional')],
                )
            ]

        # Parse the demand type

        edps = []
        for demand_parameters in demand_parameters_list:
            demand_type = demand_parameters[0]
            offset = demand_parameters[1]
            directional = demand_parameters[2]

            assert isinstance(demand_type, str)

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
                edp_type = f'{demand_type}_{subtype}'
            else:
                # If there is no subtype, convert the demand type to
                # the corresponding EDP type using
                # `base.EDP_to_demand_type`
                demand_type = base.EDP_to_demand_type[demand_type]
                # Assign the EDP type to be equal to the demand type
                edp_type = demand_type

            # Consider the default offset, if needed
            if demand_type in demand_offset:
                # If the demand type has a default offset in
                # `demand_offset`, add the offset
                # to the default offset
                offset = int(offset + demand_offset[demand_type])  # type: ignore
            else:
                # If the demand type does not have a default offset in
                # `demand_offset`, convert the
                # offset to an integer
                offset = int(offset)  # type: ignore

            # Determine the direction
            direction = pg[2] if directional else '0'

            # Concatenate the EDP type, offset, and direction to form
            # the EDP key
            edp = f'{edp_type}-{int(pg[1]) + offset!s}-{direction}'

            if int(pg[1]) + offset < 0:
                msg = (
                    f'Negative location encountered for component '
                    f'(cmp, loc, dir, uid)=`{pg}`. Would require `{edp}`. '
                    f'Please update the location of the component.'
                )
                raise ValueError(msg)

            edps.append(edp)

        edps_t = tuple(edps)  # makes it hashable

        # Add the current PG (performance group) to the list of
        # PGs associated with the current EDP key
        required_edps[edps_t, expression].append(pg)

    # Return the required EDPs
    return required_edps


def _assemble_required_demand_data(
    required_edps: set, nondirectional_multipliers: dict, demand_sample: pd.DataFrame
) -> dict:
    """
    Assembles demand data for damage state determination.

    The method takes the maximum of all available directions for
    non-directional demand, scaling it using the non-directional
    multiplier specified in self._asmnt.options, and returning the
    result as a dictionary with keys in the format of
    '<demand_type>-<location>-<direction>' and values as arrays of
    demand values.

    Parameters
    ----------
    required_edps: set
        Set of required EDPs. The elements in the set should be
        tuples. For each, the first element should be a tuple
        containing EDPs in the `type`-`loc`-`dir` format. The second
        should be an expression defining how the EDPs in the tuple
        should be combined when it contains more than a single EDP.
    nondirectional_multipliers: dict
        Nondirectional components are sensitive to demands coming
        in any direction. Results are typically available in two
        orthogonal directions. FEMA P-58 suggests using the
        formula `max(dir_1, dir_2) * 1.2` to estimate the demand
        for such components. This parameter allows modifying the
        1.2 multiplier with a user-specified value. The change can
        be applied to "ALL" EDPs, or for specific EDPs, such as
        "PFA", "PFV", etc. Examples:
        #. {'PFA': 1.2, 'PID': 1.00}
        #. {'ALL': 1.0}
    demand_sample: pd.DataFrame
        Dataframe containing the demand sample, realizations of EDPs
        (or/and IMs) that are used for damage and loss calculations.

    Returns
    -------
    demand_dict: dict
        A dictionary of assembled demand data for calculation

    Raises
    ------
    ValueError
        If demand data for a given EDP cannot be found

    """
    demand_dict = {}

    for edps, expression in required_edps:
        edp_values = {}

        for edp in edps:
            edp_type, location, direction = edp.split('-')

            if direction == '0':
                # non-directional
                demand = (
                    demand_sample.loc[
                        :,  # type: ignore
                        (edp_type, location),
                    ]
                    .max(axis=1)
                    .to_numpy()
                )

                if edp_type in nondirectional_multipliers:
                    multiplier = nondirectional_multipliers[edp_type]

                elif 'ALL' in nondirectional_multipliers:
                    multiplier = nondirectional_multipliers['ALL']

                else:
                    msg = (
                        f'Peak orthogonal EDP multiplier '
                        f'for non-directional demand '
                        f'calculation of `{edp_type}` not specified.'
                    )
                    raise ValueError(msg)

                demand *= multiplier

            else:
                # directional
                demand = demand_sample[edp_type, location, direction].to_numpy()

            edp_values[edp] = demand

        # evaluate expression
        if expression is not None:
            # build a dict of values
            value_dict = {}
            for i, edp_value in enumerate(edp_values.values()):
                value_dict[f'X{i + 1}'] = edp_value
            demand = ne.evaluate(
                _clean_up_expression(expression), local_dict=value_dict
            )
        demand_dict[edps, expression] = demand

    return demand_dict


def _clean_up_expression(expression: str) -> str:
    """
    Clean up a mathematical expression in a string.

    Cleans up the given mathematical expression by ensuring it
    contains only allowed characters and replaces the caret (^)
    exponentiation operator with the double asterisk (**) operator.

    Parameters
    ----------
    expression: str
        The mathematical expression to clean up.

    Returns
    -------
    str
        The cleaned-up mathematical expression.

    Raises
    ------
    ValueError
        If the expression contains invalid characters.

    Examples
    --------
    >>> _clean_up_expression('3 + 5 * 2')
    '3 + 5 * 2'
    >>> _clean_up_expression('2^3')
    '2**3'
    >>> _clean_up_expression('2 ** 3')
    '2 ** 3'
    >>> _clean_up_expression(
    ...     "[o.fork() for (o,i) in "
    ...     "[(__import__('os'), __import__('itertools'))] "
    ...     "for x in i.repeat(0)]"
    ... )
    Traceback (most recent call last): ...

    """
    allowed_chars = re.compile(r'^[0-9a-zA-Z\^\+\-\*/\(\)\s]*$')
    if not bool(allowed_chars.match(expression)):
        msg = f'Invalid expression: {expression}'
        raise ValueError(msg)
    # replace exponantiation with `^` with the native `**` in case `^`
    # was used. But please use `**`..
    return expression.replace('^', '**')


def _verify_edps_available(available_edps: dict, required: set) -> None:
    """
    Verify EDP availability.

    Verifies that the required EDPs are available and raises
    appropriate errors otherwise.

    Parameters
    ----------
    available_edps: dict
        Dictionary mapping (`edp_type`-`cmp`-`dir`) to list of `loc`
        where values are available.
    required: set
        Set of required EDPs, expressed as
        `edp_type`-`loc`-`dir`. Direction `0` has special meaning: It
        is used for directional demands.

    Raises
    ------
    ValueError
        When the verification fails.

    """
    # Verify that the required EDPs are available in the
    # demand sample
    for edps, _ in required:
        for edp in edps:
            edp_type, location, direction = edp.split('-')
            if (edp_type, location) not in available_edps:
                msg = (
                    f'Unable to locate `{edp_type}` at location '
                    f'{location} in demand sample.'
                )
                raise ValueError(msg)
            # if non-directional demand is requested, ensure there
            # are entries (all directions accepted)
            num_entries = len(available_edps[edp_type, location])
            if edp[2] == '0' and num_entries == 0:
                msg = (
                    f'Unable to locate any `{edp_type}` '
                    f'at location {location} and direction {direction}.'
                )
                raise ValueError(msg)
            if edp[2] != '0' and num_entries == 0:
                msg = f'Unable to locate `{edp_type}-{location}-{direction}`.'
                raise ValueError(msg)
