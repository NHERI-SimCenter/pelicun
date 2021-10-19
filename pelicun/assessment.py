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

class Assessment(object):
    """
    A high-level class that collects features common to all supported assessment
    methods. This class is only rarely called directly.
    """

    def __init__(self, log_file=True):

        # initialize the log file
        if log_file:
            set_log_file('pelicun_log.txt')

        #initialize the assessment type
        self._assessment_type = None

        log_div()
        log_msg('Assessement Started')

    def read_configuration(self, path_config):
        """
        Read and process the configuration file for pelicun.

        Parameters
        ----------
        path_config: string
            Path to the json file that contains the configuration for the
            assessment to run.

        """

        log_div()
        log_msg('Reading the configuration file...')
        if options.verbose:
            log_msg(f'{options.log_pref}file path: {path_config}\n ',
                    prepend_timestamp=False)

        self.config = read_config_file(path_config,
                                       assessment_type=self._assessment_type)

        log_msg(prepend_timestamp=False)
        log_msg('Configuration file successfully parsed.')

class DemandAssessment(Assessment):
    """
    An Assessment class for characterizing the demands acting on an asset.
    """

    def __init__(self, log_file=True):
        super(DemandAssessment, self).__init__(log_file)

        log_msg(f'{options.log_pref}type: Demand Assessment\n',
                prepend_timestamp=False)

        self._assessment_type = 'demand'


class DamageAssessment(Assessment):
    """
    An Assessment class for characterizing the damage done to an asset.
    """

    def __init__(self, log_file=True):
        super(DamageAssessment, self).__init__(log_file)

        log_msg('type: Damage Assessment')

        self._assessment_type = 'damage'

class LossAssessment(Assessment):
    """
    An Assessment class for characterizing the losses experienced by an asset
    and its inhabitants or users.
    """

    def __init__(self, log_file=True):
        super(LossAssessment, self).__init__(log_file)

        log_msg('type: Loss Assessment')

        self._assessment_type = 'loss'

