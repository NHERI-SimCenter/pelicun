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
This file defines the AssetModel object and its methods.

.. rubric:: Contents

.. autosummary::

    AssetModel

"""

from itertools import product
import numpy as np
import pandas as pd
from .pelicun_model import PelicunModel
from .. import base
from .. import uq
from .. import file_io


idx = base.idx


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
            sample,
            filepath,
            units=units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )

        if filepath is not None:
            self.log_msg(
                'Asset components sample successfully saved.',
                prepend_timestamp=False,
            )
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
            filepath,
            self._asmnt.unit_conversion_factors,
            return_units=True,
            log=self._asmnt.log,
        )

        sample.columns.names = ['cmp', 'loc', 'dir', 'uid']

        self._cmp_sample = sample

        self.cmp_units = units.groupby(level=0).first()

        self.log_msg(
            'Asset components sample successfully loaded.', prepend_timestamp=False
        )

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
                return np.array(
                    [
                        res,
                    ]
                )

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
                    return np.array(
                        [
                            stories,
                        ]
                    ).astype(str)

                if loc_str == "roof":
                    return np.array(
                        [
                            stories + 1,
                        ]
                    ).astype(str)

                raise ValueError(
                    f"Cannot parse location string: " f"{loc_str}"
                ) from exc

        def get_directions(dir_str):
            if pd.isnull(dir_str):
                return np.ones(1).astype(str)

            # else:
            try:
                res = str(int(dir_str))
                return np.array(
                    [
                        res,
                    ]
                )

            except ValueError as exc:
                if "," in dir_str:
                    return np.array(dir_str.split(','), dtype=int).astype(str)

                if "--" in dir_str:
                    d_low, d_high = dir_str.split('--')
                    d_low = get_directions(d_low)
                    d_high = get_directions(d_high)
                    return np.arange(int(d_low[0]), int(d_high[0]) + 1).astype(str)

                # else:
                raise ValueError(
                    f"Cannot parse direction string: " f"{dir_str}"
                ) from exc

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
                raise ValueError(
                    f"Cannot parse Blocks string: {attribute_str}"
                ) from exc

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
            'Family': [],
            'Theta_0': [],
            'Theta_1': [],
            'Theta_2': [],
            'TruncateLower': [],
            'TruncateUpper': [],
            'Blocks': [],
            'Units': [],
        }
        index_list = []
        for row in marginal_params.itertuples():
            locs = get_locations(row.Location)
            dirs = get_directions(row.Direction)
            indices = list(product((row.Index,), locs, dirs))
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
                    cmp_marginal_param.extend([self.cmp_units[row.Index]] * num_vals)
                elif col == 'Family':
                    cmp_marginal_param.extend([getattr(row, col, np.nan)] * num_vals)
                else:
                    cmp_marginal_param.extend(
                        [get_attribute(getattr(row, col, np.nan))] * num_vals
                    )
            index_list.extend(indices)
        index = pd.MultiIndex.from_tuples(index_list, names=['cmp', 'loc', 'dir'])
        dtypes = {
            'Family': object,
            'Theta_0': float,
            'Theta_1': float,
            'Theta_2': float,
            'TruncateLower': float,
            'TruncateUpper': float,
            'Blocks': int,
            'Units': object,
        }
        cmp_marginal_param_series = []
        for col, cmp_marginal_param in cmp_marginal_param_dct.items():
            cmp_marginal_param_series.append(
                pd.Series(
                    cmp_marginal_param, dtype=dtypes[col], name=col, index=index
                )
            )

        cmp_marginal_params = pd.concat(cmp_marginal_param_series, axis=1)

        assert not cmp_marginal_params['Theta_0'].isnull().values.any()

        cmp_marginal_params.dropna(axis=1, how='all', inplace=True)

        self.log_msg(
            "Model parameters successfully parsed. "
            f"{cmp_marginal_params.shape[0]} performance groups identified",
            prepend_timestamp=False,
        )

        # Now we can take care of converting the values to base units
        self.log_msg(
            "Converting model parameters to internal units...",
            prepend_timestamp=False,
        )

        # ensure that the index has unique entries by introducing an
        # internal component uid
        base.dedupe_index(cmp_marginal_params)

        cmp_marginal_params = self.convert_marginal_params(
            cmp_marginal_params, cmp_marginal_params['Units']
        )

        self.cmp_marginal_params = cmp_marginal_params.drop('Units', axis=1)

        self.log_msg(
            "Model parameters successfully loaded.", prepend_timestamp=False
        )

        self.log_msg(
            "\nComponent model marginal distributions:\n" + str(cmp_marginal_params),
            prepend_timestamp=False,
        )

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
            RV_reg.add_RV(
                uq.RandomVariable(
                    name=f'CMP-{cmp[0]}-{cmp[1]}-{cmp[2]}-{cmp[3]}',
                    distribution=getattr(rv_params, "Family", np.nan),
                    theta=[
                        getattr(rv_params, f"Theta_{t_i}", np.nan)
                        for t_i in range(3)
                    ],
                    truncation_limits=[
                        getattr(rv_params, f"Truncate{side}", np.nan)
                        for side in ("Lower", "Upper")
                    ],
                )
            )

        self.log_msg(
            f"\n{self.cmp_marginal_params.shape[0]} random variables created.",
            prepend_timestamp=False,
        )

        self._cmp_RVs = RV_reg

    def generate_cmp_sample(self, sample_size=None):
        """
        Generates component quantity realizations.  If a sample_size
        is not specified, the sample size found in the demand model is
        used.
        """

        if self.cmp_marginal_params is None:
            raise ValueError(
                'Model parameters have not been specified. Load'
                'parameters from a file before generating a '
                'sample.'
            )

        self.log_div()
        self.log_msg('Generating sample from component quantity variables...')

        if sample_size is None:
            if self._asmnt.demand.sample is None:
                raise ValueError(
                    'Sample size was not specified, '
                    'and it cannot be determined from '
                    'the demand model.'
                )
            sample_size = self._asmnt.demand.sample.shape[0]

        self._create_cmp_RVs()

        self._cmp_RVs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        # replace the potentially existing sample with the generated one
        self._cmp_sample = None

        self.log_msg(
            f"\nSuccessfully generated {sample_size} realizations.",
            prepend_timestamp=False,
        )
