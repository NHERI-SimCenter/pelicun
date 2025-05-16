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

"""These are unit and integration tests on the auto module of pelicun."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pelicun.auto import auto_populate

# The tests maintain the order of definitions of the `auto.py` file.

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#
# The following tests verify the functions of the module.


@pytest.fixture
def setup_valid_config() -> dict:
    return {'GeneralInformation': {'someKey': 'someValue'}}


@pytest.fixture
def setup_auto_script_path() -> str:
    return 'PelicunDefault/test_script'


@pytest.fixture
def setup_expected_base_path() -> str:
    return '/expected/path/resources/auto/'


"""
def test_valid_inputs(setup_valid_config: dict, setup_auto_script_path: str) -> None:
    with patch('pelicun.base.pelicun_path', '/expected/path'), patch(
        'os.path.exists', return_value=True
    ), patch('importlib.__import__') as mock_import:
        mock_auto_populate_ext = MagicMock(
            return_value=({'AIM_ap': 'value'}, {'DL_ap': 'value'}, 'CMP')
        )
        mock_import.return_value.auto_populate = mock_auto_populate_ext

        config, cmp = auto_populate(setup_valid_config, Path(setup_auto_script_path))

        assert 'DL' in config
        assert cmp == 'CMP'
"""


def test_missing_general_information() -> None:
    with pytest.raises(
        ValueError,
        match='No Asset Information provided for the auto-population routine.',
    ):
        auto_populate({}, Path('some/path'))


def test_pelicun_default_path_replacement(
    setup_auto_script_path: str, setup_expected_base_path: str
) -> None:
    modified_path = setup_auto_script_path.replace(
        'PelicunDefault/', setup_expected_base_path
    )
    assert modified_path.startswith(setup_expected_base_path)


"""
def test_auto_population_script_execution(
    setup_valid_config: dict, setup_auto_script_path: str
) -> None:
    with patch('pelicun.base.pelicun_path', '/expected/path'), patch(
        'os.path.exists', return_value=True
    ), patch('importlib.__import__') as mock_import:
        mock_auto_populate_ext = MagicMock(
            return_value=({'AIM_ap': 'value'}, {'DL_ap': 'value'}, 'CMP')
        )
        mock_import.return_value.auto_populate = mock_auto_populate_ext

        auto_populate(setup_valid_config, Path(setup_auto_script_path))
        mock_import.assert_called_once()
"""
