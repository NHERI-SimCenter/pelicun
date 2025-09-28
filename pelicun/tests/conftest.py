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

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pandas as pd
import pytest

# This is an explicit list of tests that handle their own network mocking
# and should be excluded from the global fixture.
EXCLUDED_TESTS = {
    'test_check_dlml_data_with_missing_data',
    'test_check_dlml_data_with_existing_data_update_available',
    'test_check_dlml_data_download_failure',
    'test_check_dlml_data_permission_error',
    'test_check_dlml_data_version_check_failure',
    'test_logging_configuration',
    'test_warning_system_integration',
}


@pytest.fixture(autouse=True)
def mock_dlml_data_check(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """
    Mocks the DLML data check for the entire test session, UNLESS
    the test is in the specific exclusion list.

    This provides a high-performance, fail-safe "no network" policy
    while allowing specific integration tests to run their own logic.
    """
    # If the current test's name is in our exclusion list, do not
    # apply the global mock.
    if request.node.name in EXCLUDED_TESTS:
        yield
    else:
        # For all other tests, apply the global mock.
        target_function_path = 'pelicun.tools.dlml.check_dlml_data'
        with patch(target_function_path) as mocked_check_func:
            mocked_check_func.return_value = None
            yield


def _setup_common_test_data() -> Path:
    """
    Create a temporary directory with common test data for regional simulations.

    This helper function sets up common test data used by multiple test fixtures:
    1. Creates a temporary directory
    2. Creates the test_event_grid.csv with 2 grid points

    Returns:
        Path object to the temporary directory
    """
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create test_event_grid.csv
    grid_data = pd.DataFrame(
        {
            'GP_file': ['GP_1.csv', 'GP_2.csv'],
            'Latitude': [37.8720, 37.8690],
            'Longitude': [-122.2730, -122.2690],
        }
    )
    grid_data.to_csv(temp_dir / 'test_event_grid.csv', index=False)

    return temp_dir


@pytest.fixture(scope='function')  # noqa: PT003
def setup_earthquake_test_data() -> Generator[Path, None, None]:
    """
    Create a temporary directory with mock data for earthquake tests.

    This fixture:
    1. Uses the common helper to create the temporary directory and common files
    2. Adds earthquake-specific building inventory
    3. Adds earthquake-specific configuration and data files
    4. Yields the path to the temporary directory
    5. Cleans up after the test by removing the directory and all its contents

    Yields:
        Generator yielding the Path object to the temporary directory
    """
    # Set up common test data
    temp_dir = _setup_common_test_data()

    try:
        # Create earthquake-specific test_bldg_inventory.csv
        bldg_data = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'Latitude': [37.8716, 37.8700, 37.8684, 37.8668, 37.8652],
                'Longitude': [-122.2727, -122.2700, -122.2683, -122.2666, -122.2649],
                'HeightClass': [
                    'Low-Rise',
                    'Low-Rise',
                    'Mid-Rise',
                    'Low-Rise',
                    'Low-Rise',
                ],
                'DesignLevel': [
                    'Pre-Code',
                    'Pre-Code',
                    'Pre-Code',
                    'High-Code',
                    'Pre-Code',
                ],
                'PlanArea': [7752, 3600, 28000, 14709, 3040],
                'NumberOfStories': [2, 3, 4, 1, 3],
                'YearBuilt': [1906, 1931, 1906, 1983, 1912],
                'ReplacementCost': [1022488.8, 6843570.3, 3693200, 1912170, 418000],
                'StructureType': ['C1', 'W2', 'RM2', 'S1', 'S1'],
                'OccupancyClass': ['COM4', 'RES3', 'COM4', 'RES3', 'RES3'],
            }
        )
        bldg_data.to_csv(temp_dir / 'test_bldg_inventory.csv', index=False)

        # Create test_config.json for earthquake scenario
        config_data = {
            'Applications': {
                'RegionalMapping': {
                    'Buildings': {'ApplicationData': {'neighbors': 2, 'samples': 10}}
                },
                'DL': {
                    'Buildings': {
                        'ApplicationData': {
                            'Realizations': 10,
                            'DL_Method': 'Hazus Earthquake - Buildings',
                        }
                    }
                },
                'Assets': {
                    'Buildings': {
                        'ApplicationData': {
                            'assetSourceFile': 'test_bldg_inventory.csv',
                            'pathToSource': '.',
                        }
                    }
                },
            },
            'RegionalEvent': {
                'eventFile': 'test_event_grid.csv',
                'eventFilePath': '.',
                'units': {'PGA': 'g'},
            },
        }

        with (temp_dir / 'test_config.json').open('w') as f:
            json.dump(config_data, f, indent=2)

        # Create GP_1.csv for earthquake scenario
        gp1_data = pd.DataFrame(
            {'PGA': [0.5, 0.51, 0.49, 0.52, 0.48, 0.5, 0.51, 0.49, 0.52, 0.48]}
        )
        gp1_data.to_csv(temp_dir / 'GP_1.csv', index=False)

        # Create GP_2.csv for earthquake scenario
        gp2_data = pd.DataFrame(
            {'PGA': [0.6, 0.61, 0.59, 0.62, 0.58, 0.6, 0.61, 0.59, 0.62, 0.58]}
        )
        gp2_data.to_csv(temp_dir / 'GP_2.csv', index=False)

        yield temp_dir

    finally:
        # Clean up by removing the temporary directory and all its contents
        shutil.rmtree(temp_dir)


@pytest.fixture(scope='function')  # noqa: PT003
def setup_hurricane_test_data() -> Generator[Path, None, None]:
    """
    Create a temporary directory with mock data for hurricane tests.

    This fixture:
    1. Uses the common helper to create the temporary directory and common files
    2. Adds hurricane-specific building inventory with features like RoofShape and Shutters
    3. Adds hurricane-specific configuration and data files
    4. Yields the path to the temporary directory
    5. Cleans up after the test by removing the directory and all its contents

    Yields:
        Generator yielding the Path object to the temporary directory
    """
    # Set up common test data
    temp_dir = _setup_common_test_data()

    try:
        # Create hurricane-specific test_bldg_inventory.csv
        hurricane_inventory_content = """id,Latitude,Longitude,BuildingType,StructureType,NumberOfStories,Height,PlanArea,RoofShape,RoofQuality,RoofSystem,RoofCover,SecondaryWaterResistance,RoofDeckAttachment,RoofToWallConnection,NumberOfUnits,JoistSpacing,Garage,Shutters,WindowArea,TieDowns,MasonryReinforcing,WindDebrisClass,LandCover,OccupancyClass
1,37.8716,-122.2727,Wood,Multi-Unit Housing,1,15,744.6,Hip,,Truss,,0,8d,Toe-nail,,,,0,,,,,Light Trees,RES3A
2,37.8700,-122.2700,Wood,Single Family Housing,1,12.97,250.2,Gable,,Truss,,0,8d,Toe-nail,,,No,1,,,,,Suburban,RES1
3,37.8684,-122.2683,Wood,Single Family Housing,1,16.28,170.4,Gable,,Truss,,0,8d,Toe-nail,,,No,0,,,,,Suburban,RES1
4,37.8668,-122.2666,Wood,Single Family Housing,2,34.58,141.9,Gable,,Truss,,0,8d,Toe-nail,,,Weak,0,,,,,Suburban,RES1
5,37.8652,-122.2649,Wood,Single Family Housing,1,14.22,269.8,Gable,,Truss,,0,8d,Toe-nail,,,Weak,0,,,,,Suburban,RES1"""

        with (temp_dir / 'test_bldg_inventory.csv').open('w') as f:
            f.write(hurricane_inventory_content)

        # Create test_config.json for hurricane scenario
        config_data = {
            'Applications': {
                'RegionalMapping': {
                    'Buildings': {'ApplicationData': {'neighbors': 2, 'samples': 10}}
                },
                'DL': {
                    'Buildings': {
                        'ApplicationData': {
                            'Realizations': 10,
                            'DL_Method': 'Hazus Hurricane Wind',
                        }
                    }
                },
                'Assets': {
                    'Buildings': {
                        'ApplicationData': {
                            'assetSourceFile': 'test_bldg_inventory.csv',
                            'pathToSource': '.',
                        }
                    }
                },
            },
            'RegionalEvent': {
                'eventFile': 'test_event_grid.csv',
                'eventFilePath': '.',
                'units': {'PWS': 'mph'},
            },
        }

        with (temp_dir / 'test_config.json').open('w') as f:
            json.dump(config_data, f, indent=2)

        # Create GP_1.csv for hurricane scenario
        gp1_data = pd.DataFrame(
            {'PWS': [100, 102, 98, 104, 96, 100, 102, 98, 104, 96]}
        )
        gp1_data.to_csv(temp_dir / 'GP_1.csv', index=False)

        # Create GP_2.csv for hurricane scenario
        gp2_data = pd.DataFrame(
            {'PWS': [110, 112, 108, 114, 106, 110, 112, 108, 114, 106]}
        )
        gp2_data.to_csv(temp_dir / 'GP_2.csv', index=False)

        yield temp_dir

    finally:
        # Clean up by removing the temporary directory and all its contents
        shutil.rmtree(temp_dir)
