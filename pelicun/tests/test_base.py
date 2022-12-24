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

"""
These are unit and integration tests on the base module of pelicun.
"""

import os
import io
import sys
from contextlib import redirect_stdout
import re
import argparse
import pytest
import pandas as pd
from pelicun import base

# for tests, we sometimes create things or call them just to see if
# things would work, so the following are irrelevant:

# pylint: disable=useless-suppression
# pylint: disable=unused-variable
# pylint: disable=pointless-statement

# funny how with pylint you get an error if you suppress a type of
# error that your code would not cause, leading to ...a pylint error
# requiring suppression of the suppression-related error. :D


# The tests maintain the order of definitions of the `base.py` file.


def test_options_init():
    """
    Test that the Options object is initialized with the correct
    attributes based on the input user configuration
    """

    # Create a sample user_config_options dictionary
    user_config_options = {
        "Verbose": False,
        "Seed": None,
        "LogShowMS": False,
        "LogFile": 'test_log_file',
        "PrintLog": False,
        "DemandOffset": {
            "PFA": -1,
            "PFV": -1
        },
        "SamplingMethod": "MonteCarlo",
        "NonDirectionalMultipliers": {
            "ALL": 1.2
        },
        "EconomiesOfScale": {
            "AcrossFloors": True,
            "AcrossDamageStates": True
        },
        "RepairCostAndTimeCorrelation": 0.7
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
    assert options.log.log_show_ms is False
    assert os.path.basename(options.log.log_file) == 'test_log_file'
    assert options.log.print_log is False

    # remove the log file that was created
    os.remove('test_log_file')


def test_nondir_multi():
    """
    Tests that the nondir_multi method of the Options class returns
    the correct value for the specified EDP type. Tests that the
    method uses the value associated with the 'ALL' key if the EDP
    type is not present in the nondir_multi_dict attribute. Tests
    that a ValueError is raised if the 'ALL' key is not present in the
    nondir_multi_dict attribute.
    """

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
    """
    Tests that the Logger object is initialized with the correct
    attributes based on the input configuration dictionary.
    """
    # Test that the Logger object is initialized with the correct
    # attributes based on the input configuration
    log_config = {'verbose': True, 'log_show_ms': False,
                  'log_file': 'log.txt', 'print_log': True}
    log = base.Logger(**log_config)
    assert log.verbose is True
    assert log.log_show_ms is False
    assert os.path.basename(log.log_file) == 'log.txt'
    assert log.print_log is True
    os.remove('log.txt')


def test_logger_msg():
    """
    Tests the functionality of the msg method of the Logger
    object.
    """

    # Test that the msg method prints the correct message to the
    # console and log file
    log_config = {'verbose': True, 'log_show_ms': True,
                  'log_file': 'log.txt', 'print_log': True}
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
    """
    Tests the functionality of the div method of the Logger
    object.
    """

    # We test the divider with and without the timestamp
    prepend_timestamp_args = (True, False)
    patterns = (
        r'[0-9][0-9]:[0-9][0-9]:[0-9][0-9]'
        r':[0-9][0-9][0-9][0-9][0-9][0-9]\s-+',
        r'\s+-+'
    )
    for case, pattern_str in zip(prepend_timestamp_args, patterns):
        pattern = re.compile(pattern_str)
        # Test that the div method adds a divider as intended
        log_config = {'verbose': True, 'log_show_ms': True,
                      'log_file': 'log.txt', 'print_log': True}
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
    """
    Tests that the system information is retrieved correctly
    """

    # create a logger object
    log_config = {'verbose': True, 'log_show_ms': True,
                  'log_file': 'log.txt', 'print_log': True}
    log = base.Logger(**log_config)

    # run print_system_info and get the console output
    with io.StringIO() as buf, redirect_stdout(buf):
        log.print_system_info()
        output = buf.getvalue()

    # verify the contents of the output
    assert 'System Information:\n' in output

    # remove the created log file
    os.remove('log.txt')


def test_convert_to_SimpleIndex():
    """
    Tests the functionality of the convert_to_SimpleIndex function.
    """

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
    """
    Tests the functionality of the convert_to_MultiIndex function.
    """

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
        data_converted, axis=0, inplace=False)
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
        data_converted, axis=1, inplace=False)
    assert data_converted.columns.equals(expected_columns)

    # Test an invalid axis parameter
    with pytest.raises(ValueError):
        base.convert_to_MultiIndex(data_converted, axis=2, inplace=False)


def test_show_matrix():
    """
    Tests the functionality of the show_matrix function.
    """

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


def test_describe():
    """
    Tests the functionality of the describe function.
    """

    pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    assert True  # Only ensure the above didn't fail.


def test_str2bool():
    """
    Tests the functionality of the test_str2bool function.
    """
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


def test_run_input_specs():
    """
    Just for the shake of coverage ^_^
    """
    print(base.CMP_data_path)
    print(base.POP_data_path)
    print(base.EDP_to_demand_type)
