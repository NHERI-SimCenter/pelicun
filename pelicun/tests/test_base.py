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
These are unit and integration tests on the base module of pelicun.
"""

import os
import io
from contextlib import redirect_stdout
import re
import argparse
import pytest
import pandas as pd
import numpy as np
from pelicun import base

# pylint: disable=missing-function-docstring

# The tests maintain the order of definitions of the `base.py` file.


def test_options_init():
    # Create a sample user_config_options dictionary
    user_config_options = {
        "Verbose": False,
        "Seed": None,
        "LogShowMS": False,
        "LogFile": 'test_log_file',
        "PrintLog": False,
        "DemandOffset": {"PFA": -1, "PFV": -1},
        "SamplingMethod": "MonteCarlo",
        "NonDirectionalMultipliers": {"ALL": 1.2},
        "EconomiesOfScale": {"AcrossFloors": True, "AcrossDamageStates": True},
        "RepairCostAndTimeCorrelation": 0.7,
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
    assert options.eco_scale == {"AcrossFloors": True, "AcrossDamageStates": True}

    # Check that the Logger object attribute of the Options object is
    # initialized with the correct parameters
    assert options.log.verbose is False
    assert options.log.show_warnings is False
    assert options.log.log_show_ms is False
    assert os.path.basename(options.log.log_file) == 'test_log_file'
    assert options.log.print_log is False

    # remove the log file that was created
    os.remove('test_log_file')

    # test seed property and setter
    options.seed = 42
    assert options.seed == 42


def test_nondir_multi():
    # Tests that the nondir_multi method of the Options class returns
    # the correct value for the specified EDP type. Tests that the
    # method uses the value associated with the 'ALL' key if the EDP
    # type is not present in the nondir_multi_dict attribute. Tests
    # that a ValueError is raised if the 'ALL' key is not present in the
    # nondir_multi_dict attribute.

    # Create an instance of the Options class with default values for all options,
    # except for the nondir_multi_dict attribute
    options = base.Options({'NonDirectionalMultipliers': {'PFA': 1.5, 'PFV': 1.00}})

    # Call the nondir_multi method with the specific EDP type as the argument
    assert options.nondir_multi('PFA') == 1.5
    assert options.nondir_multi('PFV') == 1.00

    # the 'ALL' key is automatically assigned to 1.2, even if the user
    # does not specify it
    assert 'ALL' in options.nondir_multi_dict
    assert options.nondir_multi('ALL') == 1.2

    # When an EDP type is not present in the nondir_multi_dict, the
    # value associated with 'ALL' is used.
    assert options.nondir_multi('spread love') == 1.2

    # We get an error if the 'ALL' key is not present, but this would
    # be unexpected.
    options.nondir_multi_dict.pop('ALL')  # 'ALL' is gone now
    # the following will cause a ValueError
    with pytest.raises(ValueError):
        options.nondir_multi('Sa(T*)')


def test_logger_init():
    # Test that the Logger object is initialized with the correct
    # attributes based on the input configuration
    log_config = {
        'verbose': True,
        'show_warnings': True,
        'log_show_ms': False,
        'log_file': 'log.txt',
        'print_log': True,
    }
    log = base.Logger(**log_config)
    assert log.verbose is True
    assert log.show_warnings is True
    assert log.log_show_ms is False
    assert os.path.basename(log.log_file) == 'log.txt'
    assert log.print_log is True
    os.remove('log.txt')

    # test exceptions
    log_config = {
        'verbose': True,
        'show_warnings': True,
        'log_show_ms': False,
        'log_file': '/',
        'print_log': True,
    }
    with pytest.raises((IsADirectoryError, FileExistsError, FileNotFoundError)):
        log = base.Logger(**log_config)


def test_logger_msg():
    # Test that the msg method prints the correct message to the
    # console and log file
    log_config = {
        'verbose': True,
        'show_warnings': True,
        'log_show_ms': True,
        'log_file': 'log.txt',
        'print_log': True,
    }
    log = base.Logger(**log_config)
    # Check that the message is printed to the console
    with io.StringIO() as buf, redirect_stdout(buf):
        log.msg('This is a message')
        output = buf.getvalue()
    assert 'This is a message' in output
    # Check that the message is written to the log file
    with open('log.txt', 'r', encoding='utf-8') as f:
        assert 'This is a message' in f.read()
    os.remove('log.txt')


def test_logger_div():
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
            'show_warnings': True,
            'log_show_ms': True,
            'log_file': 'log.txt',
            'print_log': True,
        }
        log = base.Logger(**log_config)

        # check console output
        with io.StringIO() as buf, redirect_stdout(buf):
            log.div(prepend_timestamp=case)
            output = buf.getvalue()
        assert pattern.match(output)
        # check log file
        with open('log.txt', 'r', encoding='utf-8') as f:
            # simply check that it is not empty
            assert f.read()

        # remove the created log file
        os.remove('log.txt')


def test_print_system_info():
    # create a logger object
    log_config = {
        'verbose': True,
        'show_warnings': True,
        'log_show_ms': True,
        'log_file': 'log.txt',
        'print_log': True,
    }
    log = base.Logger(**log_config)

    # run print_system_info and get the console output
    with io.StringIO() as buf, redirect_stdout(buf):
        log.print_system_info()
        output = buf.getvalue()

    # verify the contents of the output
    assert 'System Information:\n' in output

    # remove the created log file
    os.remove('log.txt')


def test_update_vals():
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
    with pytest.raises(ValueError):
        base.update_vals(update, primary, 'update', 'primary')

    primary = {'a': {'b': 3}}
    update = {'a': 1, 'b': 2}
    with pytest.raises(ValueError):
        base.update_vals(update, primary, 'update', 'primary')


def test_merge_default_config():
    # Test merging an empty user config with the default config
    user_config = {}
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


def test_convert_dtypes():
    # All columns able to be converted

    # Input DataFrame
    df_input = pd.DataFrame({'a': ['1', '2', '3'], 'b': ['4.0', '5.5', '6.75']})

    # Expected DataFrame
    df_expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.5, 6.75]}).astype(
        {'a': int, 'b': float}
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

    # Empty dataframe

    df_input = pd.DataFrame({})
    df_expected = pd.DataFrame({})
    df_result = base.convert_dtypes(df_input)
    pd.testing.assert_frame_equal(
        df_result, df_expected, check_index_type=False, check_column_type=False
    )


def test_convert_to_SimpleIndex():
    # Test conversion of a multiindex to a simple index following the
    # SimCenter dash convention
    index = pd.MultiIndex.from_tuples((('a', 'b'), ('c', 'd')))
    df = pd.DataFrame([[1, 2], [3, 4]], index=index)
    df.index.names = ['name_1', 'name_2']
    df_simple = base.convert_to_SimpleIndex(df, axis=0)
    assert df_simple.index.tolist() == ['a-b', 'c-d']
    assert df_simple.index.name == '-'.join(df.index.names)

    # Test inplace modification
    df_inplace = df.copy()
    base.convert_to_SimpleIndex(df_inplace, axis=0, inplace=True)
    assert df_inplace.index.tolist() == ['a-b', 'c-d']
    assert df_inplace.index.name == '-'.join(df.index.names)

    # Test conversion of columns
    index = pd.MultiIndex.from_tuples((('a', 'b'), ('c', 'd')))
    df = pd.DataFrame([[1, 2], [3, 4]], columns=index)
    df.columns.names = ['name_1', 'name_2']
    df_simple = base.convert_to_SimpleIndex(df, axis=1)
    assert df_simple.columns.tolist() == ['a-b', 'c-d']
    assert df_simple.columns.name == '-'.join(df.columns.names)

    # Test inplace modification
    df_inplace = df.copy()
    base.convert_to_SimpleIndex(df_inplace, axis=1, inplace=True)
    assert df_inplace.columns.tolist() == ['a-b', 'c-d']
    assert df_inplace.columns.name == '-'.join(df.columns.names)

    # Test invalid axis parameter
    with pytest.raises(ValueError):
        base.convert_to_SimpleIndex(df, axis=2)


def test_convert_to_MultiIndex():
    # Test a case where the index needs to be converted to a MultiIndex
    data = pd.DataFrame({'A': (1, 2, 3), 'B': (4, 5, 6)})
    data.index = ('A-1', 'B-1', 'C-1')
    data_converted = base.convert_to_MultiIndex(data, axis=0, inplace=False)
    expected_index = pd.MultiIndex.from_arrays((('A', 'B', 'C'), ('1', '1', '1')))
    assert data_converted.index.equals(expected_index)
    # original data should not have changed
    assert data.index.equals(pd.Index(('A-1', 'B-1', 'C-1')))

    # Test a case where the index is already a MultiIndex
    data_converted = base.convert_to_MultiIndex(
        data_converted, axis=0, inplace=False
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
    data_converted = base.convert_to_MultiIndex(
        data_converted, axis=1, inplace=False
    )
    assert data_converted.columns.equals(expected_columns)

    # Test an invalid axis parameter
    with pytest.raises(ValueError):
        base.convert_to_MultiIndex(data_converted, axis=2, inplace=False)

    # inplace=True
    data = pd.DataFrame({'A': (1, 2, 3), 'B': (4, 5, 6)})
    data.index = ('A-1', 'B-1', 'C-1')
    base.convert_to_MultiIndex(data, axis=0, inplace=True)
    expected_index = pd.MultiIndex.from_arrays((('A', 'B', 'C'), ('1', '1', '1')))
    assert data.index.equals(expected_index)


def test_show_matrix():
    # Test with a simple 2D array
    arr = ((1, 2, 3), (4, 5, 6))
    base.show_matrix(arr)
    assert True  # if no AssertionError is thrown, then the test passes

    # Test with a DataFrame
    df = pd.DataFrame(((1, 2, 3), (4, 5, 6)), columns=('a', 'b', 'c'))
    base.show_matrix(df)
    assert True  # if no AssertionError is thrown, then the test passes

    # Test with use_describe=True
    base.show_matrix(arr, use_describe=True)
    assert True  # if no AssertionError is thrown, then the test passes


def test__warning(capsys):
    msg = 'This is a test.'
    category = 'undefined'
    base._warning(msg, category, '{path to a file}', '{line number}')
    captured = capsys.readouterr()
    assert (
        captured.out
        == 'WARNING in {path to a file} at line {line number}\nThis is a test.\n\n'
    )
    base._warning(msg, category, 'some\\file', '{line number}')
    captured = capsys.readouterr()
    assert (
        captured.out
        == 'WARNING in some/file at line {line number}\nThis is a test.\n\n'
    )
    base._warning(msg, category, 'some/file', '{line number}')
    captured = capsys.readouterr()
    assert (
        captured.out
        == 'WARNING in some/file at line {line number}\nThis is a test.\n\n'
    )


def test_describe():
    expected_idx = pd.Index(
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
    # passing a dataframe

    df = pd.DataFrame(
        ((1.00, 2.00, 3.00), (4.00, 5.00, 6.00)), columns=['A', 'B', 'C']
    )
    desc = base.describe(df)
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


def test_str2bool():
    assert base.str2bool('True') is True
    assert base.str2bool('False') is False
    assert base.str2bool('yes') is True
    assert base.str2bool('no') is False
    assert base.str2bool('t') is True
    assert base.str2bool('f') is False
    assert base.str2bool('1') is True
    assert base.str2bool('0') is False
    assert base.str2bool(True) is True
    assert base.str2bool(False) is False
    with pytest.raises(argparse.ArgumentTypeError):
        base.str2bool('In most cases, it depends..')


def test_float_or_None():
    # Test with a string that can be converted to a float
    assert base.float_or_None('3.14') == 3.14

    # Test with a string that represents an integer
    assert base.float_or_None('42') == 42.0

    # Test with a string that represents a negative number
    assert base.float_or_None('-3.14') == -3.14

    # Test with a string that can't be converted to a float
    assert base.float_or_None('hello') is None

    # Test with an empty string
    assert base.float_or_None('') is None


def test_int_or_None():
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


def test_process_loc():
    # Test when string can be converted to an int
    assert base.process_loc('5', 10) == [
        5,
    ]

    # Test when string is in the form 'low-high'
    assert base.process_loc('2-5', 10) == [2, 3, 4, 5]

    # Test when string is 'all'
    assert base.process_loc('all', 10) == list(range(1, 11))

    # Test when string is 'top'
    assert base.process_loc('top', 10) == [
        10,
    ]

    # Test when string is 'roof'
    assert base.process_loc('roof', 10) == [
        10,
    ]

    # Test when string cannot be converted to an int or recognized
    assert base.process_loc('abc', 10) is None


def test_run_input_specs():
    assert os.path.basename(base.pelicun_path) == 'pelicun'
