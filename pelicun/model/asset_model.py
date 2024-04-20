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
        A property that gets or creates a DataFrame representing the
        component sample for the current assessment.

        If the component sample has not been previously set or
        generated, this property will generate it by retrieving
        samples from the component random variables (_cmp_RVs),
        sorting the indexes, and converting the DataFrame to use a
        MultiIndex. The component sample is structured to include
        information on component ('cmp'), location ('loc'), direction
        ('dir'), and unique identifier ('uid').

        Returns
        -------
        DataFrame
            A DataFrame containing the component samples, indexed and
            sorted appropriately. The columns are multi-indexed to
            represent various dimensions of the component data.

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
        Saves the component quantity sample to a CSV file or returns
        it as a DataFrame with optional units.

        This method handles the storage of a sample of component
        quantities, which can either be saved directly to a file or
        returned as a DataFrame for further manipulation. When saving
        to a file, additional information such as unit conversion
        factors and column units can be included. If the data is not
        being saved to a file, the method can return the DataFrame
        with or without units as specified.

        Parameters
        ----------
        filepath : str, optional
            The path to the file where the component quantity sample
            should be saved. If not provided, the sample is not saved
            to disk but returned.
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
            - DataFrame containing the component quantity sample.
            - Optionally, a Series containing the units for each
              column if `save_units` is True.

        Raises
        ------
        IOError
            Raises an IOError if there is an issue saving the file to
            the specified `filepath`.

        Notes
        -----
        The function utilizes internal logging to notify the start and
        completion of the saving process. It adjusts index types and
        handles unit conversions based on assessment configurations.
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
        Loads a component quantity sample from a specified CSV file
        into the system.

        This method reads a CSV file that contains component quantity
        samples, setting up the necessary DataFrame structures within
        the application. It also handles unit conversions using
        predefined conversion factors and captures the units of each
        component quantity from the CSV file.

        Parameters
        ----------
        filepath : str
            The path to the CSV file from which to load the component
            quantity sample.

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
        Loads the model describing component quantities in an asset
        from specified data sources.

        This function is responsible for loading data related to the
        component model of an asset. It supports loading from multiple
        types of data sources. If the data source is a string, it is
        treated as a prefix to filenames that contain the necessary
        data. If it is a dictionary, it directly contains the data as
        DataFrames.

        Parameters
        ----------
        data_source : str or dict
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
        def get_locations(loc_str):
            """
            Parses a location string to determine specific sections of
            an asset to be processed.

            This function interprets various string formats to output
            a list of strings representing sections or parts of the
            asset.  It can handle single numbers, ranges (e.g.,
            '3--7'), lists separated by commas (e.g., '1,2,5'), and
            special keywords like 'all', 'top', or 'roof'.

            Parameters
            ----------
            loc_str : str
                A string that describes the location or range of
                sections in the asset.  It can be a single number, a
                range, a comma-separated list, 'all', 'top', or
                'roof'.

            Returns
            -------
            numpy.ndarray
                An array of strings, each representing a section
                number. These sections are processed based on the
                input string, which can denote specific sections,
                ranges of sections, or special keywords.

            Raises
            ------
            ValueError
                If the location string cannot be parsed into any
                recognized format, a ValueError is raised with a
                message indicating the problematic string.

            Examples
            --------
            Given an asset with multiple sections:

            >>> get_locations('5')
            array(['5'])

            >>> get_locations('3--7')
            array(['3', '4', '5', '6', '7'])

            >>> get_locations('1,2,5')
            array(['1', '2', '5'])

            >>> get_locations('all')
            array(['1', '2', '3', ..., '10'])

            >>> get_locations('top')
            array(['10'])

            >>> get_locations('roof')
            array(['11'])
            """
            try:
                res = str(int(loc_str))
                return np.array([res])

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
            """
            Parses a direction string to determine specific
            orientations or directions applicable within an asset.

            This function processes direction descriptions to output
            an array of strings, each representing a specific
            direction.  It can handle single numbers, ranges (e.g.,
            '1--3'), lists separated by commas (e.g., '1,2,5'), and
            null values that default to '1'.

            Parameters
            ----------
            dir_str : str or None
                A string that describes the direction or range of
                directions in the asset. It can be a single number, a
                range, a comma-separated list, or it can be null,
                which defaults to representing a single default
                direction ('1').

            Returns
            -------
            numpy.ndarray
                An array of strings, each representing a
                direction. These directions are processed based on the
                input string, which can denote specific directions,
                ranges of directions, or a list.

            Raises
            ------
            ValueError
                If the direction string cannot be parsed into any
                recognized format, a ValueError is raised with a
                message indicating the problematic string.

            Examples
            --------
            Given an asset with multiple potential orientations:

            >>> get_directions(None)
            array(['1'])

            >>> get_directions('2')
            array(['2'])

            >>> get_directions('1--3')
            array(['1', '2', '3'])

            >>> get_directions('1,2,5')
            array(['1', '2', '5'])
            """
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
            # pylint: disable=missing-return-doc
            # pylint: disable=missing-return-type-doc
            if pd.isnull(attribute_str):
                return default
            return dtype(attribute_str)

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
            family = getattr(rv_params, "Family", 'deterministic')
            RV_reg.add_RV(
                uq.rv_class_map(family)(
                    name=f'CMP-{cmp[0]}-{cmp[1]}-{cmp[2]}-{cmp[3]}',
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
