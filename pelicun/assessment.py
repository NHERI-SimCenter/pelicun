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
This module has classes and methods that control the performance assessment.

.. rubric:: Contents

.. autosummary::

    Assessment

"""

import json
from datetime import datetime
from . import base
from . import file_io
from . import model
from .__init__ import __version__ as pelicun_version


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
    bldg_repair: BldgRepairModel
        ...
    stories: int
        Number of stories.
    options: Options
        Options object.
    """

    def __init__(self, config_options=None):


        self.stories = None
        self.options = base.Options(self)

        file_io.load_default_options(self.options)

        self.options.set_options(file_io.merge_default_config(
            config_options, self.options))
        self.unit_conversion_factors = file_io.parse_units(
            self.options.units_file)

        self.log_msg(f'pelicun {pelicun_version} | \n',
                     prepend_timestamp=False, prepend_blank_space=False)

        base.print_system_info(self)

        self.log_div()
        self.log_msg('Assessement Started')

    @property
    def demand(self):
        """
        Return a DemandModel object that manages the demand information.

        """
        # pylint: disable = access-member-before-definition

        if hasattr(self, '_demand'):
            return self._demand

        self._demand = model.DemandModel(self)
        return self.demand

    @property
    def asset(self):
        """
        Return an AssetModel object that manages the asset information.

        """
        # pylint: disable = access-member-before-definition

        if hasattr(self, '_asset'):
            return self._asset

        self._asset = model.AssetModel(self)
        return self.asset

    @property
    def damage(self):
        """
        Return an DamageModel object that manages the damage information.

        """
        # pylint: disable = access-member-before-definition

        if hasattr(self, '_damage'):
            return self._damage

        self._damage = model.DamageModel(self)
        return self.damage

    @property
    def bldg_repair(self):
        """
        Return an BldgRepairModel object that manages the repair information.

        """
        # pylint: disable = access-member-before-definition

        if hasattr(self, '_bldg_repair'):
            return self._bldg_repair

        self._bldg_repair = model.BldgRepairModel(self)
        return self.bldg_repair

    def get_default_data(self, data_name):
        """
        Load a default data file and pass it to the user.

        Parameters
        ----------
        data_name: string
            Name of the csv file to be loaded

        """

        data_path = str(base.pelicun_path)+'/resources/'+data_name+'.csv'

        return file_io.load_data(
            data_path, self.unit_conversion_factors,
            orientation=1, reindex=False, convert=[])

    def get_default_metadata(self, data_name):
        """
        Load a default metadata file and pass it to the user.

        Parameters
        ----------
        data_name: string
            Name of the json file to be loaded

        """

        data_path = str(base.pelicun_path) + '/resources/' + data_name + '.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def log_div(self, prepend_timestamp=False):
        """
        Print a divider line to the log file

        """

        if prepend_timestamp:
            msg = self.options.log_div

        else:
            msg = '-' * 80

        self.log_msg(msg, prepend_timestamp=prepend_timestamp)


    def log_msg(self, msg='', prepend_timestamp=True, prepend_blank_space=True):
        """
        Print a message to the screen with the current time as prefix

        The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

        Parameters
        ----------
        msg: string
           Message to print.

        """

        # pylint: disable = consider-using-f-string
        msg_lines = msg.split('\n')

        for msg_i, msg_line in enumerate(msg_lines):

            if (prepend_timestamp and (msg_i == 0)):
                formatted_msg = '{} {}'.format(
                    datetime.now().strftime(self.options.log_time_format), msg_line)
            elif prepend_timestamp:
                formatted_msg = self.options.log_pref + msg_line
            elif prepend_blank_space:
                formatted_msg = self.options.log_pref + msg_line
            else:
                formatted_msg = msg_line

            if self.options.print_log:
                print(formatted_msg)

            if self.options.log_file is not None:
                with open(self.options.log_file, 'a', encoding='utf-8') as f:
                    f.write('\n'+formatted_msg)


    def calc_unit_scale_factor(self, unit):
        """
        Determines the scale factor from input unit to the corresponding SI unit

        Parameters
        ----------
        unit: str
            Either a unit name, or a quantity and a unit name separated by a space.
            For example: 'ft' or '100 ft'.

        Returns
        -------
        scale_factor: float
            Scale factor that convert values from unit to SI unit

        Raises
        ------
        KeyError:
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
            raise KeyError(f"Specified unit not recognized: "
                           f"{unit_count} {unit_name}") from exc

        return scale_factor

