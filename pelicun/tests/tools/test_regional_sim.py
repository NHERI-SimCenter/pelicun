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

import os
from typing import TYPE_CHECKING

import pandas as pd

from pelicun.tools import regional_sim

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_regional_sim_earthquake_run(
    setup_earthquake_test_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Integration test for the regional_sim module using earthquake data.

    Tests that the regional_sim function correctly processes earthquake data
    and generates the expected output files.

    Args:
        setup_earthquake_test_data: Pytest fixture that creates and yields
            a temporary directory with test data files.
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

    # Verify that the damage.csv file contains exactly 2 rows (one for each building)
    damage_df = pd.read_csv(temp_dir / 'damage.csv')
    assert (
        len(damage_df) == 2
    ), f'Expected 2 rows in damage.csv, got {len(damage_df)}'
