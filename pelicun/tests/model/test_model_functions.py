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
These are unit and integration tests on the model module's functions.
"""


import pytest
import numpy as np
from pelicun import model

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=arguments-renamed


class TestModelFunctions:
    def _test_prep_constant_median_DV(self):
        median = 10.00
        constant_median_DV = model.loss_model.prep_constant_median_DV(median)
        assert constant_median_DV() == median
        values = (1.0, 2.0, 3.0, 4.0, 5.0)
        for value in values:
            assert constant_median_DV(value) == 10.00

    def _test_prep_bounded_multilinear_median_DV(self):
        medians = np.array((1.00, 2.00, 3.00, 4.00, 5.00))
        quantities = np.array((0.00, 1.00, 2.00, 3.00, 4.00))
        f = model.loss_model.prep_bounded_multilinear_median_DV(medians, quantities)

        result = f(2.5)
        expected = 3.5
        assert result == expected

        result = f(0.00)
        expected = 1.00
        assert result == expected

        result = f(4.00)
        expected = 5.0
        assert result == expected

        result = f(-1.00)
        expected = 1.00
        assert result == expected

        result = f(5.00)
        expected = 5.00
        assert result == expected

        result_list = f([2.5, 3.5])
        expected_list = [3.5, 4.5]
        assert np.allclose(result_list, expected_list)

        with pytest.raises(ValueError):
            f(None)
