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
This module has classes and methods that control the performance assessment.

.. rubric:: Contents

.. autosummary::

    Assessment

"""

from __future__ import annotations
import json
from pelicun import base
from pelicun import file_io
from pelicun import model
from pelicun.__init__ import __version__ as pelicun_version


class Assessment:
    """
    Assessment objects manage the models, data, and calculations in pelicun.

    Parameters
    ----------
    demand: DemandModel
        ...
    asset: AssetModel
        ...
    damage: DamageModel
        ...
    repair: RepairModel
        ...
    stories: int
        Number of stories.
    options: Options
        Options object.
    """

    __slots__ = [
        'stories',
        'options',
        'unit_conversion_factors',
        'log',
        'demand',
        'asset',
        'damage',
        'loss',
    ]

    def __init__(self, config_options=None):
        """
        Initializes an Assessment object.

        Parameters
        ----------
        config_options (Optional[dict]):
            User-specified configuration dictionary.
        """
        self.stories = None
        self.options = base.Options(config_options, self)
        self.unit_conversion_factors = base.parse_units(self.options.units_file)

        self.log = self.options.log
        self.log.msg(
            f'pelicun {pelicun_version} | \n',
            prepend_timestamp=False,
            prepend_blank_space=False,
        )
        self.log.print_system_info()
        self.log.div()
        self.log.msg('Assessment Started')

        self.demand: model.DemandModel = model.DemandModel(self)
        self.asset: model.AssetModel = model.AssetModel(self)
        self.damage: model.DamageModel = model.DamageModel(self)
        self.loss: model.LossModel = model.LossModel(self)

    def get_default_data(self, data_name):
        """
        Loads a default data file by name and returns it. This method
        is specifically designed to access predefined CSV files from a
        structured directory path related to the SimCenter fragility
        library.

        Parameters
        ----------
        data_name : str
            The name of the CSV file to be loaded, without the '.csv'
            extension. This name is used to construct the full path to
            the file.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the data loaded from the
            specified CSV file.
        """
        data_path = f'{base.pelicun_path}/resources/SimCenterDBDL/{data_name}.csv'

        return file_io.load_data(
            data_path, None, orientation=1, reindex=False, log=self.log
        )

    def get_default_metadata(self, data_name):
        """
        Load a default metadata file and pass it to the user.

        Parameters
        ----------
        data_name: string
            Name of the json file to be loaded

        Returns
        -------
        dict
            Default metadata

        """

        data_path = f'{base.pelicun_path}/resources/SimCenterDBDL/{data_name}.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def calc_unit_scale_factor(self, unit):
        """
        Determines the scale factor from input unit to the
        corresponding base unit

        Parameters
        ----------
        unit: str
            Either a unit name, or a quantity and a unit name
            separated by a space.
            For example: 'ft' or '100 ft'.

        Returns
        -------
        float
            Scale factor that convert values from unit to base unit

        Raises
        ------
        KeyError
            When an invalid unit is specified
        """

        unit_lst = unit.strip().split(' ')

        # check if there is a quantity specified; if yes, parse it
        if len(unit_lst) > 1:
            unit_count, unit_name = unit_lst
            unit_count = float(unit_count)

        else:
            unit_count = 1
            unit_name = unit_lst[0]

        try:
            scale_factor = unit_count * self.unit_conversion_factors[unit_name]

        except KeyError as exc:
            raise KeyError(
                f"Specified unit not recognized: {unit_count} {unit_name}"
            ) from exc

        return scale_factor

    def scale_factor(self, unit):
        """
        Returns the scale factor of a given unit. If the unit is
        unknown it raises an error. If the unit is None it returns
        1.00.

        Parameters
        ----------
        unit: str
            A unit name.

        Returns
        -------
        float
            Scale factor

        Raises
        ------
        ValueError
            If the unit is unknown.

        """

        if unit is not None:

            if unit in self.unit_conversion_factors:
                scale_factor = self.unit_conversion_factors[unit]

            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:
            scale_factor = 1.0

        return scale_factor
