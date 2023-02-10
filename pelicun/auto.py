# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Leland Stanford Junior University
# Copyright (c) 2023 The Regents of the University of California
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
This module has classes and methods that auto-populate DL models.

.. rubric:: Contents

.. autosummary::

    auto_populate

"""

import sys
import importlib
import json
from pathlib import Path

from . import base

def auto_populate(config, auto_script_path, **kwargs):
    """
    Automatically prepares the DL configuration for a Pelicun calculation.

    Parameters
    ----------
    config: dict
        Configuration dictionary with a GeneralInformation key that holds
        another dictionary with attributes of the asset of interest.
    auto_script_path: string
        Path pointing to a python script with the auto-population rules. 
        Built-in scripts can be referenced using the PelicunDefault/XY format
        where XY is the name of the script.
    """

    # try to get the AIM attributes
    AIM = config.get('GeneralInformation', None)
    if AIM == None:
        raise ValueError(
            "No Asset Information provided for the auto-population routine."
        )

    # replace default keyword with actual path in auto_script location
    if 'PelicunDefault/' in auto_script_path:
        auto_script_path = auto_script_path.replace(
            'PelicunDefault/', f'{base.pelicun_path}/resources/auto/')

    # load the auto population module
    ASP = Path(auto_script_path).resolve()
    sys.path.insert(0, str(ASP.parent)+'/')
    auto_script = importlib.__import__(ASP.name[:-3], globals(), locals(), [], 0)
    auto_populate_ext = auto_script.auto_populate

    # generate the DL input data
    AIM_ap, DL_ap, CMP = auto_populate_ext(AIM = AIM)

    # assemble the extended config
    config['GeneralInformation'].update(AIM_ap)
    config.update({'DL': DL_ap})

    # return the extended config data and the component quantities
    return config, CMP

