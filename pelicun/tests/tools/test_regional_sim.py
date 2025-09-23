#
# Copyright (c) 2025 Leland Stanford Junior University
# Copyright (c) 2025 The Regents of the University of California
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

"""These are tests for the regional_sim module of pelicun."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from pelicun.tools import regional_sim


def test_regional_sim_earthquake_run(
    setup_earthquake_test_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Integration test for the regional_sim module using earthquake data.
    """

    # Change current working directory to the temporary directory
    # This ensures the simulation reads inputs and writes outputs in the correct location
    temp_dir = setup_earthquake_test_data
    monkeypatch.chdir(temp_dir)

    # Construct the path to the config file
    config_file_path = temp_dir / 'test_config.json'

    # Run the regional simulation
    regional_sim.regional_sim(config_file_path, num_cores=1)

    # Verify that the expected output files were created
    expected_files = [
        'demand.csv',
        'damage.csv',
        'repair_costs.csv',
        'repair_times.csv',
    ]
    for filename in expected_files:
        assert (
            temp_dir / filename
        ).is_file(), f'Expected output file {filename} was not created'

    # Verify that the damage.csv file contains exactly 5 rows (one for each building)
    damage_df = pd.read_csv(temp_dir / 'damage.csv')
    assert (
        len(damage_df) == 5
    ), f'Expected 5 rows in damage.csv, got {len(damage_df)}'


@pytest.mark.parametrize(
    ('filter_str', 'expected_count'),
    [
        ('1,2', 2),  # Test a simple list
        ('2-4', 3),  # Test a simple range
        ('1, 3-4', 3),  # Test a mixed filter
    ],
)
def test_regional_sim_filter_success(
    filter_str: str,
    expected_count: int,
    setup_earthquake_test_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test successful building filtering with various filter formats.
    """
    # Use the test fixture that provides the 5-building inventory
    temp_dir = setup_earthquake_test_data
    monkeypatch.chdir(temp_dir)

    # Load the default config and add the filter
    config_path = temp_dir / 'test_config.json'
    with Path(config_path).open(encoding='utf-8') as f:
        config = json.load(f)

    config['Applications']['Assets']['Buildings']['ApplicationData']['filter'] = (
        filter_str
    )

    with Path(config_path).open(mode='w', encoding='utf-8') as f:
        json.dump(config, f)

    # Run the simulation
    regional_sim.regional_sim(config_file=str(config_path), num_cores=1)

    # Check that the output has the correctly filtered number of buildings
    damage_df = pd.read_csv(temp_dir / 'damage.csv', index_col=0)
    assert len(damage_df) == expected_count


@pytest.mark.parametrize(
    ('filter_str', 'expected_error'),
    [
        # Test invalid syntax (invalid range)
        ('1, 5-3, 8', "Invalid range in filter"),
        # Test invalid syntax (non-numeric value)
        ('1, 3-5, 8a', "Invalid part '8a' in filter string."),
        # Test with a building ID that does not exist
        ('1, 999999', 'building IDs from the filter were not found'),
    ],
)
def test_regional_sim_filter_failures(
    filter_str: str,
    expected_error: str,
    setup_earthquake_test_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test that invalid filter strings raise a ValueError.
    """
    temp_dir = setup_earthquake_test_data
    monkeypatch.chdir(temp_dir)

    # Load the default config and add the filter
    config_path = temp_dir / 'test_config.json'
    with Path(config_path).open(encoding='utf-8') as f:
        config = json.load(f)

    config['Applications']['Assets']['Buildings']['ApplicationData']['filter'] = (
        filter_str
    )

    with Path(config_path).open(mode='w', encoding='utf-8') as f:
        json.dump(config, f)

    # Expect a ValueError to be raised that matches the expected text
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        regional_sim.regional_sim(config_file=str(config_path), num_cores=1)
