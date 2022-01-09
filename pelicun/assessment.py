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

from .base import *
from .file_io import *
from .model import *

class Assessment(object):
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
    """

    def __init__(self, config_options=None):

        load_default_options()

        if config_options is not None:
            set_options(config_options)

        log_msg(f'pelicun {pelicun_version} | \n',
                prepend_timestamp=False, prepend_blank_space=False)

        print_system_info()

        log_div()
        log_msg('Assessement Started')

        self.stories = None

    @property
    def demand(self):
        """
        Return a DemandModel object that manages the demand information.

        """

        if hasattr(self, '_demand'):
            return self._demand

        else:
            self._demand = DemandModel()
            return self.demand

    @property
    def asset(self):
        """
        Return an AssetModel object that manages the asset information.

        """

        if hasattr(self, '_asset'):
            return self._asset

        else:
            self._asset = AssetModel(self)
            return self.asset

    @property
    def damage(self):
        """
        Return an DamageModel object that manages the damage information.

        """

        if hasattr(self, '_damage'):
            return self._damage

        else:
            self._damage = DamageModel(self)
            return self.damage

    @property
    def bldg_repair(self):
        """
        Return an BldgRepairModel object that manages the repair information.

        """

        if hasattr(self, '_bldg_repair'):
            return self._bldg_repair

        else:
            self._bldg_repair = BldgRepairModel(self)
            return self.bldg_repair