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

"""These are unit and integration tests on the assessment module of pelicun."""

from __future__ import annotations

import pytest

from pelicun import assessment


def create_assessment_obj(config: dict | None = None) -> assessment.Assessment:
    return assessment.Assessment(config) if config else assessment.Assessment({})


def test_Assessment_init() -> None:
    asmt = create_assessment_obj()
    # confirm attributes
    for attribute in (
        'asset',
        'calc_unit_scale_factor',
        'damage',
        'demand',
        'get_default_data',
        'get_default_metadata',
        'log',
        'loss',
        'options',
        'scale_factor',
        'stories',
        'unit_conversion_factors',
    ):
        assert hasattr(asmt, attribute)
    # confirm that creating an attribute on the fly is not allowed
    with pytest.raises(AttributeError):
        asmt.my_attribute = 2  # type: ignore


def test_assessment_get_default_metadata() -> None:
    asmt = create_assessment_obj()

    method_names = (
        # test for backwards compatibility
        'damage_DB_FEMA_P58_2nd',
        'damage_DB_Hazus_EQ_bldg',
        'damage_DB_Hazus_EQ_trnsp',
        'loss_repair_DB_FEMA_P58_2nd',
        'loss_repair_DB_Hazus_EQ_bldg',
        'loss_repair_DB_Hazus_EQ_trnsp',
        # current valid values
        'Hazus Earthquake - Buildings',
        'Hazus Earthquake - Stories',
        'Hazus Earthquake - Transportation',
        'Hazus Hurricane Wind - Buildings',
    )

    for method_name in method_names:
        for model_type in ['fragility', 'consequence_repair']:
            if method_name.startswith(('damage', 'loss')):
                model_type = None  # noqa: PLW2901

            # here we just test that we can load the data file, without
            # checking the contents.
            asmt.get_default_data(method_name, model_type)
            asmt.get_default_metadata(method_name, model_type)


def test_assessment_calc_unit_scale_factor() -> None:
    # default unit file
    asmt = create_assessment_obj()

    # without specifying a quantity
    assert asmt.calc_unit_scale_factor('m') == 1.00
    assert asmt.calc_unit_scale_factor('in') == 0.0254

    # with quantity
    assert asmt.calc_unit_scale_factor('2.00 m') == 2.00
    assert asmt.calc_unit_scale_factor('2 in') == 2.00 * 0.0254

    # when a custom unit file is specified, changing the base units
    asmt = create_assessment_obj(
        {
            'UnitsFile': (
                'pelicun/tests/basic/data/assessment/'
                'test_assessment_calc_unit_scale_factor/'
                'custom_units.json'
            )
        }
    )

    assert asmt.calc_unit_scale_factor('in') == 1.00
    assert asmt.calc_unit_scale_factor('m') == 39.3701

    # exceptions

    # unrecognized unit
    with pytest.raises(KeyError):
        asmt.calc_unit_scale_factor('smoot')
        # 1 smoot was 67 inches in 1958.


def test_assessment_scale_factor() -> None:
    # default unit file
    asmt = create_assessment_obj()
    assert asmt.scale_factor('m') == 1.00
    assert asmt.scale_factor('in') == 0.0254

    # when a custom unit file is specified, changing the base units
    asmt = create_assessment_obj(
        {
            'UnitsFile': (
                'pelicun/tests/basic/data/assessment/'
                'test_assessment_calc_unit_scale_factor/'
                'custom_units.json'
            )
        }
    )

    assert asmt.scale_factor('in') == 1.00
    assert asmt.scale_factor('m') == 39.3701

    # exceptions
    with pytest.raises(ValueError, match='Unknown unit: helen'):
        asmt.scale_factor('helen')
