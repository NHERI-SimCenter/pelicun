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


"""AssetModel object and methods."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from pelicun import base, file_io, uq
from pelicun.model.pelicun_model import PelicunModel

if TYPE_CHECKING:
    from pelicun.assessment import AssessmentBase

idx = base.idx


class AssetModel(PelicunModel):
    """Asset information used in assessments."""

    __slots__ = ['_cmp_RVs', 'cmp_marginal_params', 'cmp_sample', 'cmp_units']

    def __init__(self, assessment: AssessmentBase) -> None:
        """
        Initialize an Asset model.

        Parameters
        ----------
        assessment: AssessmentBase
            Parent assessment object.

        """
        super().__init__(assessment)

        self.cmp_marginal_params: pd.DataFrame | None = None
        self.cmp_units: pd.Series | None = None
        self.cmp_sample: pd.DataFrame | None = None

        self._cmp_RVs: uq.RandomVariableRegistry | None = None

    def save_cmp_sample(
        self, filepath: str | None = None, *, save_units: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series] | None:
        """
        Save or retrieve component quantity sample.

        Saves the component quantity sample to a CSV file or returns
        it as a DataFrame with optional units.  This method handles
        the storage of a sample of component quantities, which can
        either be saved directly to a file or returned as a DataFrame
        for further manipulation. When saving to a file, additional
        information such as unit conversion factors and column units
        can be included. If the data is not being saved to a file, the
        method can return the DataFrame with or without units as
        specified.

        Parameters
        ----------
        filepath: str, optional
            The path to the file where the component quantity sample
            should be saved. If not provided, the sample is not saved
            to disk but returned.
        save_units: bool, default: False
            Indicates whether to include a row with unit information
            in the returned DataFrame. This parameter is ignored if a
            file path is provided.

        Returns
        -------
        None or tuple
            If `filepath` is provided, the function returns None after
            saving the data.
            If no `filepath` is specified, returns:
            * DataFrame containing the component quantity sample.
            * Optionally, a Series containing the units for each
            column if `save_units` is True.

        Notes
        -----
        The function utilizes internal logging to notify the start and
        completion of the saving process. It adjusts index types and
        handles unit conversions based on assessment configurations.

        """
        self.log.div()
        if filepath is not None:
            self.log.msg('Saving asset components sample...')

        # prepare a units array
        sample = self.cmp_sample
        assert isinstance(sample, pd.DataFrame)

        units = pd.Series(name='Units', index=sample.columns, dtype=object)

        assert self.cmp_units is not None
        for cmp_id, unit_name in self.cmp_units.items():
            units.loc[cmp_id, :] = unit_name  # type: ignore

        res = file_io.save_to_csv(
            sample,
            Path(filepath) if filepath is not None else None,
            units=units,
            unit_conversion_factors=self._asmnt.unit_conversion_factors,
            use_simpleindex=(filepath is not None),
            log=self._asmnt.log,
        )
        if filepath is not None:
            self.log.msg(
                'Asset components sample successfully saved.',
                prepend_timestamp=False,
            )
            return None
        # else:

        assert isinstance(res, pd.DataFrame)

        units_part = res.loc['Units']
        assert isinstance(units_part, pd.Series)
        units = units_part

        res = res.drop('Units')

        if save_units:
            return res.astype(float), units

        return res.astype(float)

    def load_cmp_sample(self, filepath: str) -> None:
        """
        Load a component quantity sample from a specified CSV file.

        This method reads a CSV file that contains component quantity
        samples, setting up the necessary DataFrame structures within
        the application. It also handles unit conversions using
        predefined conversion factors and captures the units of each
        component quantity from the CSV file.

        Parameters
        ----------
        filepath: str
            The path to the CSV file from which to load the component
            quantity sample.

        Raises
        ------
        ValueError
          If the columns have an invalid number of levels.

        Notes
        -----
        Upon successful loading, the method sets the component sample
        and units as internal attributes of the class, making them
        readily accessible for further operations. It also sets
        appropriate column names for the DataFrame to match expected
        indexing levels such as component ('cmp'), location ('loc'),
        direction ('dir'), and unique identifier ('uid').

        Examples
        --------
        Assuming the filepath to the component sample CSV is known and
        accessible:

        >>> model.load_cmp_sample('path/to/component_sample.csv')
        # This will load the component quantity sample into the model
        # from the specified file.

        """
        self.log.div()
        self.log.msg('Loading asset components sample...')

        sample, units = file_io.load_data(
            filepath,
            self._asmnt.unit_conversion_factors,
            return_units=True,
            log=self._asmnt.log,
        )
        assert isinstance(sample, pd.DataFrame)
        assert isinstance(units, pd.Series)

        # Check if a `uid` level was passed
        num_levels = len(sample.columns.names)
        num_levels_without_uid = 3
        num_levels_with_uid = num_levels_without_uid + 1
        if num_levels == num_levels_without_uid:
            # No `uid`, add one.
            sample.columns.names = ['cmp', 'loc', 'dir']
            sample = base.dedupe_index(sample.T).T
        elif num_levels == num_levels_with_uid:
            sample.columns.names = ['cmp', 'loc', 'dir', 'uid']
        else:
            msg = (
                f'Invalid component sample: Column MultiIndex '
                f'has an unexpected length: {num_levels}'
            )
            raise ValueError(msg)

        self.cmp_sample = sample

        self.cmp_units = units.groupby(level=0).first()

        # Add marginal parameters with Blocks information (later calls
        # rely on that attribute being defined)
        # Obviously we can't trace back the distributions and their
        # parameters, those columns are left undefined.
        cmp_marginal_params = pd.DataFrame(
            self.cmp_sample.columns.to_list(), columns=self.cmp_sample.columns.names
        ).astype(str)
        cmp_marginal_params['Blocks'] = 1
        cmp_marginal_params = cmp_marginal_params.set_index(
            ['cmp', 'loc', 'dir', 'uid']
        )
        self.cmp_marginal_params = cmp_marginal_params

        self.log.msg(
            'Asset components sample successfully loaded.', prepend_timestamp=False
        )

    def load_cmp_model(self, data_source: str | dict[str, pd.DataFrame]) -> None:
        """
        Load the asset model from a specified data source.

        This function is responsible for loading data related to the
        component model of an asset. It supports loading from multiple
        types of data sources. If the data source is a string, it is
        treated as a prefix to filenames that contain the necessary
        data. If it is a dictionary, it directly contains the data as
        DataFrames.

        Parameters
        ----------
        data_source: str or dict
            The source from where to load the component model data. If
            it's a string, it should be the prefix for three files:
            one for marginal distributions (`<prefix>_marginals.csv`),
            one for empirical data (`<prefix>_empirical.csv`), and one
            for correlation data (`<prefix>_correlation.csv`). If it's
            a dictionary, it should have keys 'marginals',
            'empirical', and 'correlation', with each key associated
            with a DataFrame containing the corresponding data.

        Notes
        -----
        The function utilizes helper functions to handle complex
        location strings that can describe single locations, ranges,
        lists, or special keywords like 'all', 'top', and
        'roof'. These are used to specify which parts of the asset the
        data pertains to, especially useful in buildings with multiple
        stories or sections.

        Examples
        --------
        To load data using file prefixes:

        >>> model.load_cmp_model('path/to/data_prefix')

        To load data using a dictionary of DataFrames:

        >>> data_dict = {
            'marginals': df_marginals,
            'empirical': df_empirical,
            'correlation': df_correlation
        }
        >>> model.load_cmp_model(data_dict)

        """
        self.log.div()
        self.log.msg('Loading component model...')

        # Currently, we assume independent component distributions are defined
        # throughout the building. Correlations may be added afterward or this
        # method can be extended to read correlation matrices too if needed.

        # prepare the marginal data source variable to load the data
        if isinstance(data_source, dict):
            marginal_data_source: pd.DataFrame | str = data_source['marginals']
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
        assert isinstance(marginal_params, pd.DataFrame)
        assert isinstance(units, pd.Series)

        # group units by cmp id to avoid redundant entries
        self.cmp_units = units.copy().groupby(level=0).first()

        marginal_params = pd.concat([marginal_params, units], axis=1)

        cmp_marginal_param_dct: dict[str, list[Any]] = {
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
            locs = self._get_locations(str(row.Location))
            dirs = self._get_directions(str(row.Direction))
            indices = list(product((row.Index,), locs, dirs))
            num_vals = len(indices)
            for col, cmp_marginal_param in cmp_marginal_param_dct.items():
                if col == 'Blocks':
                    cmp_marginal_param.extend(
                        [
                            int(row.Blocks)  # type: ignore
                            if ('Blocks' in dir(row) and not pd.isna(row.Blocks))
                            else 1,
                        ]
                        * num_vals
                    )
                elif col == 'Units':
                    cmp_marginal_param.extend([self.cmp_units[row.Index]] * num_vals)
                elif col == 'Family':
                    cmp_marginal_param.extend([getattr(row, col, np.nan)] * num_vals)
                else:
                    cmp_marginal_param.extend(
                        [str(getattr(row, col, np.nan))] * num_vals
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

        assert not (
            cmp_marginal_params['Theta_0'].isna().to_numpy().any()  # type: ignore
        )

        cmp_marginal_params = cmp_marginal_params.dropna(axis=1, how='all')

        self.log.msg(
            'Model parameters successfully parsed. '
            f'{cmp_marginal_params.shape[0]} performance groups identified',
            prepend_timestamp=False,
        )

        # Now we can take care of converting the values to base units
        self.log.msg(
            'Converting model parameters to internal units...',
            prepend_timestamp=False,
        )

        # ensure that the index has unique entries by introducing an
        # internal component uid
        cmp_marginal_params = base.dedupe_index(cmp_marginal_params)

        cmp_marginal_params = self._convert_marginal_params(
            cmp_marginal_params, cmp_marginal_params['Units']
        )

        self.cmp_marginal_params = cmp_marginal_params.drop('Units', axis=1)

        self.log.msg(
            'Model parameters successfully loaded.', prepend_timestamp=False
        )

        self.log.msg(
            '\nComponent model marginal distributions:\n' + str(cmp_marginal_params),
            prepend_timestamp=False,
        )

        # the empirical data and correlation files can be added later, if needed

    def list_unique_component_ids(self) -> list[str]:
        """
        Obtain unique component IDs.

        Returns
        -------
        list | set
            Unique components in the asset model.

        """
        assert self.cmp_marginal_params is not None
        return self.cmp_marginal_params.index.unique(level=0).to_list()

    def generate_cmp_sample(self, sample_size: int | None = None) -> None:
        """
        Generate a component sample.

        Generates a sample of component quantity realizations based on
        predefined model parameters and optionally specified sample
        size.  If no sample size is provided, the function attempts to
        use the sample size from an associated demand model.

        Parameters
        ----------
        sample_size: int, optional
            The number of realizations to generate. If not specified,
            the sample size is taken from the demand model associated
            with the assessment.

        Raises
        ------
        ValueError
            If the model parameters are not loaded before sample
            generation, or if neither sample size is specified nor can
            be determined from the demand model.

        """
        if self.cmp_marginal_params is None:
            msg = (
                'Model parameters have not been specified. Load '
                'parameters from a file before generating a '
                'sample.'
            )
            raise ValueError(msg)

        self.log.div()
        self.log.msg('Generating sample from component quantity variables...')

        if sample_size is None:
            if self._asmnt.demand.sample is None:
                msg = (
                    'Sample size was not specified, '
                    'and it cannot be determined from '
                    'the demand model.'
                )
                raise ValueError(msg)
            sample_size = self._asmnt.demand.sample.shape[0]

        self._create_cmp_RVs()

        assert self._cmp_RVs is not None
        assert self._asmnt.options.sampling_method is not None
        self._cmp_RVs.generate_sample(
            sample_size=sample_size, method=self._asmnt.options.sampling_method
        )

        cmp_sample = pd.DataFrame(self._cmp_RVs.RV_sample)
        cmp_sample = cmp_sample.sort_index(axis=0)
        cmp_sample = cmp_sample.sort_index(axis=1)
        cmp_sample_mi = base.convert_to_MultiIndex(cmp_sample, axis=1)['CMP']
        assert isinstance(cmp_sample_mi, pd.DataFrame)
        cmp_sample = cmp_sample_mi
        cmp_sample.columns.names = ['cmp', 'loc', 'dir', 'uid']
        self.cmp_sample = cmp_sample

        self.log.msg(
            f'\nSuccessfully generated {sample_size} realizations.',
            prepend_timestamp=False,
        )

    def _create_cmp_RVs(self) -> None:  # noqa: N802
        """Define the RVs used for sampling component quantities."""
        # initialize the registry
        rv_reg = uq.RandomVariableRegistry(self._asmnt.options.rng)

        # add a random variable for each component quantity variable
        assert self.cmp_marginal_params is not None
        for rv_params in self.cmp_marginal_params.itertuples():
            cmp = rv_params.Index

            # create a random variable and add it to the registry
            family = getattr(rv_params, 'Family', 'deterministic')
            rv_reg.add_RV(
                uq.rv_class_map(family)(
                    name=f'CMP-{cmp[0]}-{cmp[1]}-{cmp[2]}-{cmp[3]}',  # type: ignore
                    theta=np.array(
                        [
                            value
                            for t_i in range(3)
                            if (value := getattr(rv_params, f'Theta_{t_i}', None))
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
            f'\n{self.cmp_marginal_params.shape[0]} random variables created.',
            prepend_timestamp=False,
        )

        self._cmp_RVs = rv_reg
