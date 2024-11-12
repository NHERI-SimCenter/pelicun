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

"""DL Calculation Example 9."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from pelicun.pelicun_warnings import PelicunWarning
from pelicun.tools.DL_calculation import run_pelicun


@pytest.fixture
def obtain_temp_dir() -> Generator:
    # get the path of this file
    this_file = __file__

    initial_dir = Path.cwd()
    this_dir = str(Path(this_file).parent)

    temp_dir = tempfile.mkdtemp()

    yield this_dir, temp_dir

    # go back to the right directory, otherwise any tests that follow
    # could have issues.
    os.chdir(initial_dir)


def test_dl_calculation_9(obtain_temp_dir: tuple[str, str]) -> None:
    this_dir, temp_dir = obtain_temp_dir

    # Copy all input files to a temporary directory.
    # All outputs will also go there.
    # This approach is more robust to changes in the output files over
    # time.
    ruleset_files = [
        path.resolve()
        for path in Path('pelicun/tests/dl_calculation/rulesets').glob(
            '*Rulesets.py'
        )
    ]

    dl_models_dir = Path(f'{this_dir}/CustomDLModels').resolve()
    os.chdir(this_dir)
    temp_dir = tempfile.mkdtemp()
    # copy input files
    for file_name in ('3500-AIM.json', 'response.csv', 'custom_pop.py'):
        shutil.copy(f'{this_dir}/{file_name}', f'{temp_dir}/{file_name}')
    # copy ruleset files
    for file_path in ruleset_files:
        shutil.copy(str(file_path), f'{temp_dir}/{file_path.name}')
    # copy the custom models
    shutil.copytree(str(dl_models_dir), f'{temp_dir}/{dl_models_dir.name}')
    # change directory to there
    os.chdir(temp_dir)

    # run
    run_pelicun(
        demand_file='response.csv',
        config_path='3500-AIM.json',
        output_path=None,
        coupled_edp=True,
        realizations=100,
        auto_script_path='custom_pop.py',
        detailed_results=False,
        output_format=None,
        custom_model_dir='./CustomDLModels',
    )

    # now remove the ruleset files and auto script
    for file_path in ruleset_files:
        Path(f'{temp_dir}/{file_path.name}').unlink()
    Path('custom_pop.py').unlink()

    #
    # Test files
    #

    # Ensure the number of files is as expected
    num_files = sum(1 for entry in Path(temp_dir).iterdir() if entry.is_file())
    assert num_files == 19

    # Verify their names
    files = {
        '3500-AIM.json',
        '3500-AIM_ap.json',
        'CMP_QNT.csv',
        'CMP_sample.json',
        'DEM_sample.json',
        'DL_summary.csv',
        'DL_summary.json',
        'DL_summary_stats.csv',
        'DL_summary_stats.json',
        'DMG_grp.json',
        'DMG_grp_stats.json',
        'DV_repair_agg.json',
        'DV_repair_agg_stats.json',
        'DV_repair_grp.json',
        'DV_repair_sample.json',
        'DV_repair_stats.json',
        'pelicun_log.txt',
        'pelicun_log_warnings.txt',
        'response.csv',
    }

    for file in files:
        assert Path(f'{temp_dir}/{file}').is_file()

    #
    # Check the values: TODO
    #
