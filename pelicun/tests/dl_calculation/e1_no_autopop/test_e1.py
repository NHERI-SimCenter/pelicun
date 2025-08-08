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

"""DL Calculation Example 1."""

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


def test_dl_calculation_1(obtain_temp_dir: str) -> None:
    this_dir: str
    temp_dir: str

    this_dir, temp_dir = obtain_temp_dir  # type: ignore

    # Copy all input files to a temporary directory.
    # All outputs will also go there.
    # This approach is more robust to changes in the output files over
    # time.

    os.chdir(this_dir)
    temp_dir = tempfile.mkdtemp()
    # copy input files
    for file_name in ('8000-AIM.json', 'response.csv', 'CMP_QNT.csv'):
        shutil.copy(f'{this_dir}/{file_name}', f'{temp_dir}/{file_name}')

    # change directory to there
    os.chdir(temp_dir)

    # run
    run_pelicun(
        demand_file='response.csv',
        config_path='8000-AIM.json',
        output_path=None,
        coupled_edp=True,
        realizations=100,
        auto_script_path=None,
        detailed_results=False,
        output_format=None,
        custom_model_dir=None,
    )

    #
    # Test files
    #

    # Get all files in the directory
    all_files = [entry.name for entry in Path(temp_dir).iterdir() if entry.is_file()]
    # print(f"Files in directory: {all_files}")

    # Ensure the number of files is as expected
    num_files = len(all_files)
    assert num_files == 18

    # Verify their names
    files = {
        '8000-AIM.json',
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
