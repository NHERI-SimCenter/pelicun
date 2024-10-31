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

"""These are unit and integration tests on the base module of pelicun."""

from __future__ import annotations

import argparse
import io
import platform
import re
import subprocess  # noqa: S404
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pelicun import base
from pelicun.base import ensure_value

# The tests maintain the order of definitions of the `base.py` file.


def test_options_init() -> None:
    temp_dir = tempfile.mkdtemp()

    # Create a sample user_config_options dictionary
    user_config_options = {
        'Verbose': False,
        'Seed': None,
        'LogShowMS': False,
        'LogFile': f'{temp_dir}/test_log_file',
        'PrintLog': False,
        'DemandOffset': {'PFA': -1, 'PFV': -1},
        'Sampling': {
            'SamplingMethod': 'MonteCarlo',
            'SampleSize': 1000,
            'PreserveRawOrder': False,
        },
        'SamplingMethod': 'MonteCarlo',
        'NonDirectionalMultipliers': {'ALL': 1.2},
        'EconomiesOfScale': {'AcrossFloors': True, 'AcrossDamageStates': True},
        'RepairCostAndTimeCorrelation': 0.7,
    }

    # Create an Options object using the user_config_options
    # dictionary
    options = base.Options(user_config_options)

    # Check that the Options object was created successfully
    assert options is not None

    # Check that the values of the Options object attributes match the
    # values in the user_config_options dictionary
    assert options.sampling_method == 'MonteCarlo'
    assert options.units_file is None
    assert options.demand_offset == {'PFA': -1, 'PFV': -1}
    assert options.nondir_multi_dict == {'ALL': 1.2}
    assert options.rho_cost_time == 0.7
    assert options.eco_scale == {'AcrossFloors': True, 'AcrossDamageStates': True}

    # Check that the Logger object attribute of the Options object is
    # initialized with the correct parameters
    assert options.log.verbose is False
    assert options.log.log_show_ms is False
    assert Path(ensure_value(options.log.log_file)).name == 'test_log_file'
    assert options.log.print_log is False

    # test seed property and setter
    options.seed = 42
    assert options.seed == 42

    # test rng
    assert isinstance(options.rng, np.random._generator.Generator)


def test_nondir_multi() -> None:
    options = base.Options({'NonDirectionalMultipliers': {'PFA': 1.5, 'PFV': 1.00}})
    assert options.nondir_multi_dict == {'PFA': 1.5, 'PFV': 1.0, 'ALL': 1.2}


def test_logger_init() -> None:
    # Test that the Logger object is initialized with the correct
    # attributes based on the input configuration

    temp_dir = tempfile.mkdtemp()

    log_config = {
        'verbose': True,
        'log_show_ms': False,
        'log_file': f'{temp_dir}/log.txt',
        'print_log': True,
    }
    log = base.Logger(**log_config)  # type: ignore
    assert log.verbose is True
    assert log.log_show_ms is False
    assert Path(ensure_value(log.log_file)).name == 'log.txt'
    assert log.print_log is True

    # test exceptions
    log_config = {
        'verbose': True,
        'log_show_ms': False,
        'log_file': '/',
        'print_log': True,
    }
    with pytest.raises((IsADirectoryError, FileExistsError, FileNotFoundError)):
        log = base.Logger(**log_config)  # type: ignore


def test_logger_msg() -> None:
    temp_dir = tempfile.mkdtemp()

    # Test that the msg method prints the correct message to the
    # console and log file
    log_config = {
        'verbose': True,
        'log_show_ms': True,
        'log_file': f'{temp_dir}/log.txt',
        'print_log': True,
    }
    log = base.Logger(**log_config)  # type: ignore
    # Check that the message is printed to the console
    with io.StringIO() as buf, redirect_stdout(buf):
        log.msg('This is a message')
        output = buf.getvalue()
    assert 'This is a message' in output
    # Check that the message is written to the log file
    with Path(f'{temp_dir}/log.txt').open(encoding='utf-8') as f:
        assert 'This is a message' in f.read()

    # Check if timestamp is printed
    with io.StringIO() as buf, redirect_stdout(buf):
        log.msg(
            ('This is a message\nSecond line'),
            prepend_timestamp=True,
        )
        output = buf.getvalue()
        pattern = r'(\d{2}:\d{2}:\d{2})'
        assert re.search(pattern, output) is not None


def test_logger_div() -> None:
    temp_dir = tempfile.mkdtemp()

    # We test the divider with and without the timestamp
    prepend_timestamp_args = (True, False)
    patterns = (
        r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]:[0-9][0-9][0-9][0-9][0-9][0-9]\s-+',
        r'\s+-+',
    )
    for case, pattern_str in zip(prepend_timestamp_args, patterns):
        pattern = re.compile(pattern_str)
        # Test that the div method adds a divider as intended
        log_config = {
            'verbose': True,
            'log_show_ms': True,
            'log_file': f'{temp_dir}/log.txt',
            'print_log': True,
        }
        log = base.Logger(**log_config)  # type: ignore

        # check console output
        with io.StringIO() as buf, redirect_stdout(buf):
            log.div(prepend_timestamp=case)
            output = buf.getvalue()
        assert pattern.match(output)
        # check log file
        with Path(f'{temp_dir}/log.txt').open(encoding='utf-8') as f:
            # simply check that it is not empty
            assert f.read()


@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason='Skipping test on Windows due to path handling issues.',
)
def test_logger_exception() -> None:
    # Create a temporary directory for log files
    temp_dir = tempfile.mkdtemp()

    # Create a sample Python script that will raise an exception
    test_script = Path(temp_dir) / 'test_script.py'
    test_script_content = f"""
from pathlib import Path
from pelicun.base import Logger

log_file_A = Path("{temp_dir}") / 'log_A.txt'
log_file_B = Path("{temp_dir}") / 'log_B.txt'

log_A = Logger(
    log_file=log_file_A,
    verbose=True,
    log_show_ms=True,
    print_log=True,
)
log_B = Logger(
    log_file=log_file_B,
    verbose=True,
    log_show_ms=True,
    print_log=True,
)

raise ValueError('Test exception in subprocess')
"""

    # Write the test script to the file
    test_script.write_text(test_script_content)

    # Use subprocess to run the script
    process = subprocess.run(  # noqa: S603
        ['python', str(test_script)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )

    # Check that the process exited with an error
    assert process.returncode == 1

    # Check the stdout/stderr for the expected output
    assert 'Test exception in subprocess' in process.stdout

    # Check that the exception was logged in the log file
    log_files = (
        Path(temp_dir) / 'log_A_warnings.txt',
        Path(temp_dir) / 'log_B_warnings.txt',
    )
    for log_file in log_files:
        assert log_file.exists(), 'Log file was not created'
        log_content = log_file.read_text()
        assert 'Test exception in subprocess' in log_content
        assert 'Traceback' in log_content
        assert 'ValueError' in log_content


def test_split_file_name() -> None:
    file_path = 'example.file.name.txt'
    name, extension = base.split_file_name(file_path)
    assert name == 'example.file.name'
    assert extension == '.txt'

    file_path = 'example'
    name, extension = base.split_file_name(file_path)
    assert name == 'example'
    assert extension == ''  # noqa: PLC1901


def test_print_system_info() -> None:
    temp_dir = tempfile.mkdtemp()

    # create a logger object
    log_config = {
        'verbose': True,
        'log_show_ms': True,
        'log_file': f'{temp_dir}/log.txt',
        'print_log': True,
    }
    log = base.Logger(**log_config)  # type: ignore

    # run print_system_info and get the console output
    with io.StringIO() as buf, redirect_stdout(buf):
        log.print_system_info()
        output = buf.getvalue()

    # verify the contents of the output
    assert 'System Information:\n' in output


def test_update_vals() -> None:
    primary = {'b': {'c': 4, 'd': 5}, 'g': 7}
    update = {'a': 1, 'b': {'c': 3, 'd': 5}, 'f': 6}
    base.update_vals(update, primary, 'update', 'primary')
    assert primary == {'b': {'c': 4, 'd': 5}, 'g': 7}  # unchanged
    assert update == {'a': 1, 'b': {'c': 3, 'd': 5}, 'f': 6, 'g': 7}  # updated
    # note: key 'g' created, 'f' left there, 'c', 'd' updated, as intended

    primary = {'a': {'b': {'c': 4}}}
    update = {'a': {'b': {'c': 3}}}
    base.update_vals(update, primary, 'update', 'primary')
    assert primary == {'a': {'b': {'c': 4}}}  # unchanged
    assert update == {'a': {'b': {'c': 3}}}  # updated

    primary = {'a': {'b': 4}}
    update = {'a': {'b': {'c': 3}}}
    with pytest.raises(ValueError, match='should not map to a dictionary'):
        base.update_vals(update, primary, 'update', 'primary')

    primary = {'a': {'b': 3}}
    update = {'a': 1, 'b': 2}
    with pytest.raises(ValueError, match='should map to a dictionary'):
        base.update_vals(update, primary, 'update', 'primary')


def test_merge_default_config() -> None:
    # Test merging an empty user config with the default config
    user_config: dict[str, object] | None = {}
    merged_config = base.merge_default_config(user_config)
    assert merged_config == base.load_default_options()

    user_config = None  # same as {}
    merged_config = base.merge_default_config(user_config)
    assert merged_config == base.load_default_options()

    # Test merging a user config with a single option set
    user_config = {'Verbose': True}
    merged_config = base.merge_default_config(user_config)
    assert merged_config == {**base.load_default_options(), **user_config}

    # Test merging a user config with multiple options set
    user_config = {'Verbose': True, 'Seed': 12345}
    merged_config = base.merge_default_config(user_config)
    assert merged_config == {**base.load_default_options(), **user_config}

    # Test merging a user config with a nested option set
    user_config = {'NonDirectionalMultipliers': {'PFA': 1.5}}
    merged_config = base.merge_default_config(user_config)
    assert merged_config == {**base.load_default_options(), **user_config}

    # Test merging a user config with a nested option set and a top-level option set
    user_config = {'Verbose': True, 'NonDirectionalMultipliers': {'PFA': 1.5}}
    merged_config = base.merge_default_config(user_config)
    assert merged_config == {**base.load_default_options(), **user_config}


def test_convert_dtypes() -> None:
    # All columns able to be converted

    # Input DataFrame
    df_input = pd.DataFrame({'a': ['1', '2', '3'], 'b': ['4.0', '5.5', '6.75']})

    # Expected DataFrame
    df_expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.5, 6.75]}).astype(
        {'a': 'int64', 'b': 'float64'}
    )

    # Convert data types
    df_result = base.convert_dtypes(df_input)

    pd.testing.assert_frame_equal(
        df_result, df_expected, check_index_type=False, check_column_type=False
    )

    # No columns that can be converted

    df_input = pd.DataFrame(
        {'a': ['foo', 'bar', 'baz'], 'b': ['2021-01-01', '2021-01-02', '2021-01-03']}
    )
    df_expected = df_input.copy()
    df_result = base.convert_dtypes(df_input)
    pd.testing.assert_frame_equal(
        df_result, df_expected, check_index_type=False, check_column_type=False
    )

    # Columns with mixed types

    df_input = pd.DataFrame(
        {
            'a': ['1', '2', 'three'],
            'b': ['4.0', '5.5', 'six'],
            'c': ['7', 'eight', '9'],
        }
    )
    df_result = base.convert_dtypes(df_input)
    pd.testing.assert_frame_equal(
        df_result, df_input, check_index_type=False, check_column_type=False
    )

    # None values present

    df_input = pd.DataFrame({'a': [None, '2', '3'], 'b': ['4.0', None, '6.75']})
    df_expected = pd.DataFrame({'a': [np.nan, 2, 3], 'b': [4.0, np.nan, 6.75]})
    df_result = base.convert_dtypes(df_input)
    pd.testing.assert_frame_equal(
        df_result,
        df_expected,
        check_dtype=False,
        check_index_type=False,
        check_column_type=False,
    )

    # Empty DataFrame

    df_input = pd.DataFrame({})
    df_expected = pd.DataFrame({})
    df_result = base.convert_dtypes(df_input)
    pd.testing.assert_frame_equal(
        df_result, df_expected, check_index_type=False, check_column_type=False
    )


def test_convert_to_SimpleIndex() -> None:
    # Test conversion of a multiindex to a simple index following the
    # SimCenter dash convention
    index = pd.MultiIndex.from_tuples((('a', 'b'), ('c', 'd')))
    data = pd.DataFrame([[1, 2], [3, 4]], index=index)
    data.index.names = ['name_1', 'name_2']
    data_simple = base.convert_to_SimpleIndex(data, axis=0)
    assert data_simple.index.tolist() == ['a-b', 'c-d']
    assert data_simple.index.name == '-'.join(data.index.names)

    # Test inplace modification
    df_inplace = data.copy()
    base.convert_to_SimpleIndex(df_inplace, axis=0, inplace=True)
    assert df_inplace.index.tolist() == ['a-b', 'c-d']
    assert df_inplace.index.name == '-'.join(data.index.names)

    # Test conversion of columns
    index = pd.MultiIndex.from_tuples((('a', 'b'), ('c', 'd')))
    data = pd.DataFrame([[1, 2], [3, 4]], columns=index)
    data.columns.names = ['name_1', 'name_2']
    data_simple = base.convert_to_SimpleIndex(data, axis=1)
    assert data_simple.columns.tolist() == ['a-b', 'c-d']
    assert data_simple.columns.name == '-'.join(data.columns.names)

    # Test inplace modification
    df_inplace = data.copy()
    base.convert_to_SimpleIndex(df_inplace, axis=1, inplace=True)
    assert df_inplace.columns.tolist() == ['a-b', 'c-d']
    assert df_inplace.columns.name == '-'.join(data.columns.names)

    # Test invalid axis parameter
    with pytest.raises(ValueError, match='Invalid axis parameter: 2'):
        base.convert_to_SimpleIndex(data, axis=2)


def test_convert_to_MultiIndex() -> None:
    # Test a case where the index needs to be converted to a MultiIndex
    data = pd.DataFrame({'A': (1, 2, 3), 'B': (4, 5, 6)})
    data.index = pd.Index(['A-1', 'B-1', 'C-1'])
    data_converted = base.convert_to_MultiIndex(data, axis=0, inplace=False)
    expected_index = pd.MultiIndex.from_arrays((('A', 'B', 'C'), ('1', '1', '1')))
    assert data_converted.index.equals(expected_index)
    # original data should not have changed
    assert data.index.equals(pd.Index(('A-1', 'B-1', 'C-1')))

    # Test a case where the index is already a MultiIndex
    data_converted = pd.DataFrame(
        base.convert_to_MultiIndex(data_converted, axis=0, inplace=False)
    )
    assert data_converted.index.equals(expected_index)

    # Test a case where the columns need to be converted to a MultiIndex
    data = pd.DataFrame({'A-1': (1, 2, 3), 'B-1': (4, 5, 6)})
    data_converted = base.convert_to_MultiIndex(data, axis=1, inplace=False)
    expected_columns = pd.MultiIndex.from_arrays((('A', 'B'), ('1', '1')))
    assert data_converted.columns.equals(expected_columns)
    # original data should not have changed
    assert data.columns.equals(pd.Index(('A-1', 'B-1')))

    # Test a case where the columns are already a MultiIndex
    data_converted = pd.DataFrame(
        base.convert_to_MultiIndex(data_converted, axis=1, inplace=False)
    )
    assert data_converted.columns.equals(expected_columns)

    # Test an invalid axis parameter
    with pytest.raises(ValueError, match='Invalid axis parameter: 2'):
        base.convert_to_MultiIndex(data_converted, axis=2, inplace=False)

    # inplace=True
    data = pd.DataFrame({'A': (1, 2, 3), 'B': (4, 5, 6)})
    data.index = pd.Index(['A-1', 'B-1', 'C-1'])
    base.convert_to_MultiIndex(data, axis=0, inplace=True)
    expected_index = pd.MultiIndex.from_arrays((('A', 'B', 'C'), ('1', '1', '1')))
    assert data.index.equals(expected_index)


def test_show_matrix() -> None:
    # Test with a simple 2D array
    arr = np.array(((1, 2, 3), (4, 5, 6)))
    base.show_matrix(arr)

    # Test with a DataFrame
    data = pd.DataFrame(((1, 2, 3), (4, 5, 6)), columns=('a', 'b', 'c'))
    base.show_matrix(data)

    # Test with use_describe=True
    base.show_matrix(arr, use_describe=True)


def test_multiply_factor_multiple_levels() -> None:
    # Original DataFrame definition
    data = pd.DataFrame(
        np.full((5, 3), 1.00),
        index=pd.MultiIndex.from_tuples(
            [
                ('A', 'X', 'K'),
                ('A', 'X', 'L'),
                ('A', 'Y', 'M'),
                ('B', 'X', 'K'),
                ('B', 'Y', 'M'),
            ],
            names=['lv1', 'lv2', 'lv3'],
        ),
        columns=['col1', 'col2', 'col3'],
    )

    # Test 1: Basic multiplication on rows
    result_df = pd.DataFrame(
        np.array(
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        index=pd.MultiIndex.from_tuples(
            [
                ('A', 'X', 'K'),
                ('A', 'X', 'L'),
                ('A', 'Y', 'M'),
                ('B', 'X', 'K'),
                ('B', 'Y', 'M'),
            ],
            names=['lv1', 'lv2', 'lv3'],
        ),
        columns=['col1', 'col2', 'col3'],
    )
    test_df = data.copy()
    base.multiply_factor_multiple_levels(test_df, {'lv1': 'A', 'lv2': 'X'}, 2)
    pd.testing.assert_frame_equal(
        test_df,
        result_df,
    )

    # Test 2: Multiplication on all rows
    result_df_all = pd.DataFrame(
        np.full((5, 3), 3.00),
        index=pd.MultiIndex.from_tuples(
            [
                ('A', 'X', 'K'),
                ('A', 'X', 'L'),
                ('A', 'Y', 'M'),
                ('B', 'X', 'K'),
                ('B', 'Y', 'M'),
            ],
            names=['lv1', 'lv2', 'lv3'],
        ),
        columns=['col1', 'col2', 'col3'],
    )
    test_df = data.copy()
    base.multiply_factor_multiple_levels(test_df, {}, 3)
    pd.testing.assert_frame_equal(test_df, result_df_all)

    # Original DataFrame definition for columns test
    df_columns = pd.DataFrame(
        np.ones((3, 5)),
        index=['row1', 'row2', 'row3'],
        columns=pd.MultiIndex.from_tuples(
            [
                ('A', 'X', 'K'),
                ('A', 'X', 'L'),
                ('A', 'Y', 'M'),
                ('B', 'X', 'K'),
                ('B', 'Y', 'M'),
            ],
            names=['lv1', 'lv2', 'lv3'],
        ),
    )

    # Test 3: Multiplication on columns
    result_df_columns = pd.DataFrame(
        np.array(
            [
                [2.0, 2.0, 1.0, 2.0, 1.0],
                [2.0, 2.0, 1.0, 2.0, 1.0],
                [2.0, 2.0, 1.0, 2.0, 1.0],
            ]
        ),
        index=['row1', 'row2', 'row3'],
        columns=pd.MultiIndex.from_tuples(
            [
                ('A', 'X', 'K'),
                ('A', 'X', 'L'),
                ('A', 'Y', 'M'),
                ('B', 'X', 'K'),
                ('B', 'Y', 'M'),
            ],
            names=['lv1', 'lv2', 'lv3'],
        ),
    )
    test_df = df_columns.copy()
    base.multiply_factor_multiple_levels(test_df, {'lv2': 'X'}, 2, axis=1)
    pd.testing.assert_frame_equal(
        test_df,
        result_df_columns,
    )

    # Test 4: Multiplication with no matching conditions
    with pytest.raises(
        ValueError, match="No rows found matching the conditions: `{'lv1': 'C'}`"
    ):
        base.multiply_factor_multiple_levels(data.copy(), {'lv1': 'C'}, 2)

    # Test 5: Invalid axis
    with pytest.raises(ValueError, match='Invalid axis: `2`'):
        base.multiply_factor_multiple_levels(data.copy(), {'lv1': 'A'}, 2, axis=2)

    # Test 6: Empty conditions affecting all rows
    result_df_empty = pd.DataFrame(
        np.full((5, 3), 4.00),
        index=pd.MultiIndex.from_tuples(
            [
                ('A', 'X', 'K'),
                ('A', 'X', 'L'),
                ('A', 'Y', 'M'),
                ('B', 'X', 'K'),
                ('B', 'Y', 'M'),
            ],
            names=['lv1', 'lv2', 'lv3'],
        ),
        columns=['col1', 'col2', 'col3'],
    )
    testing_df = data.copy()
    base.multiply_factor_multiple_levels(testing_df, {}, 4)
    pd.testing.assert_frame_equal(testing_df, result_df_empty)


def test_describe() -> None:
    expected_idx: pd.Index = pd.Index(
        (
            'count',
            'mean',
            'std',
            'log_std',
            'min',
            '0.1%',
            '2.3%',
            '10%',
            '15.9%',
            '50%',
            '84.1%',
            '90%',
            '97.7%',
            '99.9%',
            'max',
        ),
        dtype='object',
    )

    # case 1:
    # passing a DataFrame

    data = pd.DataFrame(
        ((1.00, 2.00, 3.00), (4.00, 5.00, 6.00)), columns=['A', 'B', 'C']
    )
    desc = base.describe(data)
    assert np.all(desc.index == expected_idx)
    assert np.all(desc.columns == pd.Index(('A', 'B', 'C'), dtype='object'))

    # case 2:
    # passing a series

    sr = pd.Series((1.00, 2.00, 3.00), name='A')
    desc = base.describe(sr)
    assert np.all(desc.index == expected_idx)
    assert np.all(desc.columns == pd.Index(('A',), dtype='object'))

    # case 3:
    # passing a 2D numpy array

    desc = base.describe(np.array(((1.00, 2.00, 3.00), (4.00, 5.00, 6.00))))
    assert np.all(desc.index == expected_idx)
    assert np.all(desc.columns == pd.Index((0, 1, 2), dtype='object'))

    # case 4:
    # passing a 1D numpy array

    desc = base.describe(np.array((1.00, 2.00, 3.00)))
    assert np.all(desc.index == expected_idx)
    assert np.all(desc.columns == pd.Index((0,), dtype='object'))


def test_str2bool() -> None:
    assert base.str2bool('True') is True
    assert base.str2bool('False') is False
    assert base.str2bool('yes') is True
    assert base.str2bool('no') is False
    assert base.str2bool('t') is True
    assert base.str2bool('f') is False
    assert base.str2bool('1') is True
    assert base.str2bool('0') is False
    assert base.str2bool(v=True) is True
    assert base.str2bool(v=False) is False
    with pytest.raises(argparse.ArgumentTypeError):
        base.str2bool('In most cases, it depends..')


def test_float_or_None() -> None:
    # Test with a string that can be converted to a float
    assert base.float_or_None('123.00') == 123.00

    # Test with a string that represents an integer
    assert base.float_or_None('42') == 42.0

    # Test with a string that represents a negative number
    assert base.float_or_None('-123.00') == -123.00

    # Test with a string that can't be converted to a float
    assert base.float_or_None('hello') is None

    # Test with an empty string
    assert base.float_or_None('') is None


def test_int_or_None() -> None:
    # Test the case when the string can be converted to int
    assert base.int_or_None('123') == 123
    assert base.int_or_None('-456') == -456
    assert base.int_or_None('0') == 0
    assert base.int_or_None('+789') == 789

    # Test the case when the string cannot be converted to int
    assert base.int_or_None('abc') is None
    assert base.int_or_None('123a') is None
    assert base.int_or_None(' ') is None
    assert base.int_or_None('') is None


def test_with_parsed_str_na_values() -> None:
    data = pd.DataFrame(
        {
            'A': [1.00, 2.00, 'N/A', 4.00, 5.00],
            'B': ['foo', 'bar', 'NA', 'baz', 'qux'],
            'C': [1, 2, 3, 4, 5],
        }
    )

    res = base.with_parsed_str_na_values(data)
    pd.testing.assert_frame_equal(
        res,
        pd.DataFrame(
            {
                'A': [1.00, 2.00, np.nan, 4.00, 5.00],
                'B': ['foo', 'bar', np.nan, 'baz', 'qux'],
                'C': [1, 2, 3, 4, 5],
            }
        ),
    )


def test_run_input_specs() -> None:
    assert Path(base.pelicun_path).name == 'pelicun'


def test_dedupe_index() -> None:
    tuples = [('A', '1'), ('A', '1'), ('B', '2'), ('B', '3')]
    index = pd.MultiIndex.from_tuples(tuples, names=['L1', 'L2'])
    data = np.full((4, 1), 0.00)
    data_pd = pd.DataFrame(data, index=index)
    data_pd = base.dedupe_index(data_pd)
    assert data_pd.to_dict() == {
        0: {
            ('A', '1', '0'): 0.0,
            ('A', '1', '1'): 0.0,
            ('B', '2', '0'): 0.0,
            ('B', '3', '0'): 0.0,
        }
    }


def test_dict_raise_on_duplicates() -> None:
    res = base.dict_raise_on_duplicates([('A', '1'), ('B', '2')])
    assert res == {'A': '1', 'B': '2'}
    with pytest.raises(ValueError, match='duplicate key: A'):
        base.dict_raise_on_duplicates([('A', '1'), ('A', '2')])


def test_parse_units() -> None:
    # Test the default units are parsed correctly
    units = base.parse_units()
    assert isinstance(units, dict)
    expect = {
        'sec': 1.0,
        'minute': 60.0,
        'hour': 3600.0,
        'day': 86400.0,
        'm': 1.0,
        'mm': 0.001,
        'cm': 0.01,
        'km': 1000.0,
        'in': 0.0254,
        'inch': 0.0254,
        'ft': 0.3048,
        'mile': 1609.344,
        'm2': 1.0,
        'mm2': 1e-06,
        'cm2': 0.0001,
        'km2': 1000000.0,
        'in2': 0.00064516,
        'inch2': 0.00064516,
        'ft2': 0.09290304,
        'mile2': 2589988.110336,
        'm3': 1.0,
        'in3': 1.6387064e-05,
        'inch3': 1.6387064e-05,
        'ft3': 0.028316846592,
        'cmps': 0.01,
        'mps': 1.0,
        'mph': 0.44704,
        'inps': 0.0254,
        'inchps': 0.0254,
        'ftps': 0.3048,
        'mps2': 1.0,
        'inps2': 0.0254,
        'inchps2': 0.0254,
        'ftps2': 0.3048,
        'g': 9.80665,
        'kg': 1.0,
        'ton': 1000.0,
        'lb': 0.453592,
        'N': 1.0,
        'kN': 1000.0,
        'lbf': 4.4482179868,
        'kip': 4448.2179868,
        'kips': 4448.2179868,
        'Pa': 1.0,
        'kPa': 1000.0,
        'MPa': 1000000.0,
        'GPa': 1000000000.0,
        'psi': 6894.751669043338,
        'ksi': 6894751.669043338,
        'Mpsi': 6894751669.043338,
        'A': 1.0,
        'V': 1.0,
        'kV': 1000.0,
        'ea': 1.0,
        'unitless': 1.0,
        'rad': 1.0,
        'C': 1.0,
        'USD_2011': 1.0,
        'USD': 1.0,
        'loss_ratio': 1.0,
        'worker_day': 1.0,
        'EA': 1.0,
        'SF': 0.09290304,
        'LF': 0.3048,
        'TN': 1000.0,
        'AP': 1.0,
        'CF': 0.0004719474432,
        'KV': 1000.0,
        'J': 1.0,
        'MJ': 1000000.0,
        'test_two': 2.00,
        'test_three': 3.00,
    }
    for thing, value in units.items():
        assert thing in expect
        assert value == expect[thing]

    # Test that additional units are parsed correctly
    additional_units_file = (
        'pelicun/tests/basic/data/base/test_parse_units/additional_units_a.json'
    )
    units = base.parse_units(additional_units_file)
    assert isinstance(units, dict)
    assert 'year' in units
    assert units['year'] == 1.00

    # Test that an exception is raised if the additional units file is not found
    with pytest.raises(FileNotFoundError):
        units = base.parse_units('invalid/file/path.json')

    # Test that an exception is raised if the additional units file is
    # not a valid JSON file
    invalid_json_file = 'pelicun/tests/basic/data/base/test_parse_units/invalid.json'
    with pytest.raises(
        ValueError,
        match='not a valid JSON file.',
    ):
        units = base.parse_units(invalid_json_file)

    # Test that an exception is raised if a unit is defined twice in
    # the additional units file
    duplicate_units_file = (
        'pelicun/tests/basic/data/base/test_parse_units/duplicate2.json'
    )
    with pytest.raises(
        ValueError,
        match='sec defined twice',
    ):
        units = base.parse_units(duplicate_units_file)

    # Test that an exception is raised if a unit conversion factor is not a float
    invalid_units_file = (
        'pelicun/tests/basic/data/base/test_parse_units/not_float.json'
    )
    with pytest.raises(TypeError):
        units = base.parse_units(invalid_units_file)

    # Test that we get an error if some first-level key does not point
    # to a dictionary
    invalid_units_file = (
        'pelicun/tests/basic/data/base/test_parse_units/not_dict.json'
    )
    with pytest.raises(
        (ValueError, TypeError),
        match="contains first-level keys that don't point to a dictionary",
    ):
        units = base.parse_units(invalid_units_file)


def test_unit_conversion() -> None:
    # Test scalar conversion from feet to meters
    assert base.convert_units(1.00, 'ft', 'm') == 0.3048

    # Test list conversion from feet to meters
    feet_values_list = [1.0, 2.0, 3.0]
    meter_values_list = [0.3048, 0.6096, 0.9144]
    np.testing.assert_array_almost_equal(
        base.convert_units(feet_values_list, 'ft', 'm'), meter_values_list
    )

    # Test numpy array conversion from feet to meters
    feet_values_array = np.array([1.0, 2.0, 3.0])
    meter_values_array = np.array([0.3048, 0.6096, 0.9144])
    np.testing.assert_array_almost_equal(
        base.convert_units(feet_values_array, 'ft', 'm'), meter_values_array
    )

    # Test conversion with explicit category
    assert base.convert_units(1.00, 'ft', 'm', category='length') == 0.3048

    # Test error handling for invalid input type
    with pytest.raises(TypeError) as excinfo:
        base.convert_units('one', 'ft', 'm')  # type: ignore
    assert str(excinfo.value) == 'Invalid input type for `values`'

    # Test error handling for unknown unit
    with pytest.raises(ValueError, match='Unknown unit `xyz`'):
        base.convert_units(1.00, 'xyz', 'm')

    # Test error handling for mismatched category
    with pytest.raises(ValueError, match='Unknown unit: `ft`'):
        base.convert_units(1.00, 'ft', 'm', category='volume')

    # Test error handling unknown category
    with pytest.raises(ValueError, match='Unknown category: `unknown_category`'):
        base.convert_units(1.00, 'ft', 'm', category='unknown_category')

    # Test error handling different categories
    with pytest.raises(
        ValueError,
        match='`lb` is a `mass` unit, but `m` is not specified in that category.',
    ):
        base.convert_units(1.00, 'lb', 'm')


def test_stringterpolation() -> None:
    func = base.stringterpolation('1,2,3|4,5,6')
    x_new = np.array([4, 4.5, 5])
    expected = np.array([1, 1.5, 2])
    np.testing.assert_array_almost_equal(func(x_new), expected)


def test_invert_mapping() -> None:
    original_dict = {'a': [1, 2], 'b': [3]}
    expected = {1: 'a', 2: 'a', 3: 'b'}
    assert base.invert_mapping(original_dict) == expected

    # with duplicates, raises an error
    original_dict = {'a': [1, 2], 'b': [2]}
    with pytest.raises(
        ValueError, match='Cannot invert mapping with duplicate values.'
    ):
        base.invert_mapping(original_dict)
