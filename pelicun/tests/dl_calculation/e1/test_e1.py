"""
DL Calculation Example 1

"""

import tempfile
import os
import shutil
from pathlib import Path
import pytest
import pandas as pd
from pelicun.warnings import PelicunWarning
from pelicun.tools.DL_calculation import run_pelicun


def test_dl_calculation_1():

    # get the path of this file
    this_file = __file__

    initial_dir = os.getcwd()
    this_dir = str(Path(this_file).parent)

    # Copy all input files to a temporary directory.
    # All outputs will also go there.
    # This approach is more robust to changes in the output files over
    # time.

    os.chdir(this_dir)
    temp_dir = tempfile.mkdtemp()
    # copy input files
    for file_name in ('8000-AIM.json', 'response.csv'):
        shutil.copy(f'{this_dir}/{file_name}', f'{temp_dir}/{file_name}')

    # change directory to there
    os.chdir(temp_dir)

    # run
    with pytest.warns(PelicunWarning):
        return_int = run_pelicun(
            demand_file='response.csv',
            config_path='8000-AIM.json',
            output_path=None,
            coupled_EDP=True,
            realizations='10000',
            auto_script_path='PelicunDefault/Hazus_Earthquake_IM.py',
            detailed_results=False,
            regional=True,
            output_format=None,
            custom_model_dir=None,
            color_warnings=False,
        )

    # Python is written in C after all.
    assert return_int == 0

    #
    # Test files
    #

    # Ensure the number of files is as expected
    num_files = sum(1 for entry in Path(temp_dir).iterdir() if entry.is_file())
    assert num_files == 19

    # Verify their names
    files = {
        '8000-AIM.json',
        '8000-AIM_ap.json',
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
    # Check the values in DL_summary.csv
    #

    # We test that the mean/std of the result matches within a 50%
    # relative margin.

    dl_summary = pd.read_csv(f'{temp_dir}/DL_summary.csv')
    mean = dl_summary.mean()
    std = dl_summary.std()
    df = pd.concat((mean, std), axis=1, keys=['mu', 'sigma'])
    df.drop('#', inplace=True)

    expected = pd.DataFrame(
        {
            'mu': [0.408, 96.0, 96.0, 0.033, 0.00],
            'sigma': [0.378, 87.80, 87.80, 0.18, 0.00],
        },
        index=df.index,
    )

    pd.testing.assert_frame_equal(df, expected, rtol=0.5)

    # go back to the right directory, otherwise any tests that follow
    # could have issues.
    os.chdir(initial_dir)
