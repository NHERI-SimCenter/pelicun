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
These are unit and integration tests on the file_io module of pelicun.
"""

import tempfile
import os
import pytest
import numpy as np
import pandas as pd
from pelicun import file_io

# for tests, we sometimes create things or call them just to see if
# things would work, so the following are irrelevant:

# pylint: disable=useless-suppression
# pylint: disable=unused-variable
# pylint: disable=pointless-statement


# The tests maintain the order of definitions of the `file_io.py` file.

def test_float_or_None():
    """
    Tests the functionality of the float_or_None function.
    """

    # Test with a string that can be converted to a float
    assert file_io.float_or_None('3.14') == 3.14

    # Test with a string that represents an integer
    assert file_io.float_or_None('42') == 42.0

    # Test with a string that represents a negative number
    assert file_io.float_or_None('-3.14') == -3.14

    # Test with a string that can't be converted to a float
    assert file_io.float_or_None('hello') is None

    # Test with an empty string
    assert file_io.float_or_None('') is None


def test_int_or_None():
    """
    Tests the functionality of the int_or_None function.
    """

    # Test the case when the string can be converted to int
    assert file_io.int_or_None('123') == 123
    assert file_io.int_or_None('-456') == -456
    assert file_io.int_or_None('0') == 0
    assert file_io.int_or_None('+789') == 789

    # Test the case when the string cannot be converted to int
    assert file_io.int_or_None('abc') is None
    assert file_io.int_or_None('123a') is None
    assert file_io.int_or_None(' ') is None
    assert file_io.int_or_None('') is None


def test_process_loc():
    """
    Tests the functionality of the process_loc function.
    """

    # Test when string can be converted to an int
    assert file_io.process_loc('5', 10) == [5, ]

    # Test when string is in the form 'low-high'
    assert file_io.process_loc('2-5', 10) == [2, 3, 4, 5]

    # Test when string is 'all'
    assert file_io.process_loc('all', 10) == list(range(1, 11))

    # Test when string is 'top'
    assert file_io.process_loc('top', 10) == [10, ]

    # Test when string is 'roof'
    assert file_io.process_loc('roof', 10) == [10, ]

    # Test when string cannot be converted to an int or recognized
    assert file_io.process_loc('abc', 10) is None


def test_update_vals():
    """
    Tests the functionality of the update_vals function.
    """

    primary = {'b': {'c': 4, 'd': 5}, 'g': 7}
    update = {'a': 1, 'b': {'c': 3, 'd': 5}, 'f': 6}
    file_io.update_vals(update, primary, 'update', 'primary')
    assert primary == {'b': {'c': 4, 'd': 5}, 'g': 7}

    primary = {'a': {'b': {'c': 4}}}
    update = {'a': {'b': {'c': 3}}}
    file_io.update_vals(update, primary, 'update', 'primary')
    assert primary == {'a': {'b': {'c': 4}}}

    primary = {'a': {'b': 4}}
    update = {'a': {'b': {'c': 3}}}
    with pytest.raises(ValueError):
        file_io.update_vals(update, primary, 'update', 'primary')

    primary = {'a': {'b': 3}}
    update = {'a': 1, 'b': 2}
    with pytest.raises(ValueError):
        file_io.update_vals(update, primary, 'update', 'primary')


def test_merge_default_config():
    """
    Tests the functionality of the merge_default_config function.
    """

    # Test merging an empty user config with the default config
    user_config = {}
    merged_config = file_io.merge_default_config(user_config)
    assert merged_config == file_io.load_default_options()

    # Test merging a user config with a single option set
    user_config = {'Verbose': True}
    merged_config = file_io.merge_default_config(user_config)
    assert merged_config == {**file_io.load_default_options(), **user_config}

    # Test merging a user config with multiple options set
    user_config = {'Verbose': True, 'Seed': 12345}
    merged_config = file_io.merge_default_config(user_config)
    assert merged_config == {**file_io.load_default_options(), **user_config}

    # Test merging a user config with a nested option set
    user_config = {'NonDirectionalMultipliers': {'PFA': 1.5}}
    merged_config = file_io.merge_default_config(user_config)
    assert merged_config == {**file_io.load_default_options(), **user_config}

    # Test merging a user config with a nested option set and a top-level option set
    user_config = {'Verbose': True, 'NonDirectionalMultipliers': {'PFA': 1.5}}
    merged_config = file_io.merge_default_config(user_config)
    assert merged_config == {**file_io.load_default_options(), **user_config}


def test_parse_units():
    """
    Tests the functionality of the parse_units function.
    """

    # Test the default units are parsed correctly
    units = file_io.parse_units()
    assert isinstance(units, dict)
    expect = {
        'sec': 1.0, 'minute': 60.0, 'hour': 3600.0,
        'day': 86400.0, 'm': 1.0, 'mm': 0.001,
        'cm': 0.01, 'km': 1000.0, 'in': 0.0254,
        'inch': 0.0254, 'ft': 0.3048, 'mile': 1609.344,
        'm2': 1.0, 'mm2': 1e-06, 'cm2': 0.0001,
        'km2': 1000000.0, 'in2': 0.00064516, 'inch2': 0.00064516,
        'ft2': 0.09290304, 'mile2': 2589988.110336, 'm3': 1.0,
        'in3': 1.6387064e-05, 'inch3': 1.6387064e-05, 'ft3': 0.028316846592,
        'cmps': 0.01, 'mps': 1.0, 'mph': 0.44704,
        'inps': 0.0254, 'inchps': 0.0254, 'ftps': 0.3048,
        'mps2': 1.0, 'inps2': 0.0254, 'inchps2': 0.0254,
        'ftps2': 0.3048, 'g': 9.80665, 'kg': 1.0,
        'ton': 1000.0, 'lb': 0.453592, 'N': 1.0,
        'kN': 1000.0, 'lbf': 4.4482179868, 'kip': 4448.2179868,
        'kips': 4448.2179868, 'Pa': 1.0, 'kPa': 1000.0,
        'MPa': 1000000.0, 'GPa': 1000000000.0, 'psi': 6894.751669043338,
        'ksi': 6894751.669043338, 'Mpsi': 6894751669.043338, 'A': 1.0,
        'V': 1.0, 'kV': 1000.0, 'ea': 1.0,
        'rad': 1.0, 'C': 1.0, 'USD_2011': 1.0,
        'USD': 1.0, 'loss_ratio': 1.0, 'worker_day': 1.0,
        'EA': 1.0, 'SF': 0.09290304, 'LF': 0.3048,
        'TN': 1000.0, 'AP': 1.0, 'CF': 0.0004719474432,
        'KV': 1000.0
    }
    for thing in units:
        assert thing in expect
        assert units[thing] == expect[thing]

    # Test that additional units are parsed correctly
    additional_units_file = \
        'tests/data/file_io/test_parse_units/additional_units_a.json'
    units = file_io.parse_units(additional_units_file)
    assert isinstance(units, dict)
    assert 'year' in units
    assert units['year'] == 1.00

    # Test that an exception is raised if the additional units file is not found
    with pytest.raises(FileNotFoundError):
        units = file_io.parse_units('invalid/file/path.json')

    # Test that an exception is raised if the additional units file is
    # not a valid JSON file
    invalid_json_file = 'tests/data/file_io/test_parse_units/invalid.json'
    with pytest.raises(Exception):
        units = file_io.parse_units(invalid_json_file)

    # Test that an exception is raised if a unit is defined twice in
    # the additional units file
    duplicate_units_file = 'tests/data/file_io/test_parse_units/duplicate.json'
    with pytest.raises(ValueError):
        units = file_io.parse_units(duplicate_units_file)

    # Test that an exception is raised if a unit conversion factor is not a float
    invalid_units_file = 'tests/data/file_io/test_parse_units/not_float.json'
    with pytest.raises(TypeError):
        units = file_io.parse_units(invalid_units_file)


def test_save_to_csv():
    """
    Tests the functionality of the save_to_csv function.
    """

    # Test saving with orientation 0
    data = pd.DataFrame(
        {"A": [1e-3, 2e-3, 3e-3],
         "B": [4e-3, 5e-3, 6e-3]})
    units = pd.Series(
        ["meters", "meters"], index=["A", "B"])
    unit_conversion_factors = {"meters": 0.001}

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'foo.csv')
        file_io.save_to_csv(
            data, filepath,
            units, unit_conversion_factors, orientation=0)
        assert os.path.isfile(filepath)
        # Check that the file contains the expected data
        with open(filepath, 'r', encoding='utf-8') as f:
            contents = f.read()
            assert contents == (
                ',A,B\n0,meters,meters\n0,1.0,4.0'
                '\n1,2.0,5.0\n2,3.0,6.0\n')

    # Test saving with orientation 1
    data = pd.DataFrame(
        {"A": [1e-3, 2e-3, 3e-3],
         "B": [4e-3, 5e-3, 6e-3]})
    units = pd.Series(
        ["meters", "meters"], index=["A", "B"])
    unit_conversion_factors = {"meters": 0.001}

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'bar.csv')
        file_io.save_to_csv(
            data, filepath,
            units, unit_conversion_factors, orientation=1)
        assert os.path.isfile(filepath)
        # Check that the file contains the expected data
        with open(filepath, 'r', encoding='utf-8') as f:
            contents = f.read()
            assert contents == (
                ',0,A,B\n0,,0.001,0.004\n1,,0.002,'
                '0.005\n2,,0.003,0.006\n')


def test_load_data():
    """
    Tests the functionality of the load_data function.
    """

    # test loading data with orientation 0

    filepath = os.path.join(
        'tests', 'data', 'file_io',
        'test_load_data', 'units.csv')
    unit_conversion_factors = {"inps2": 0.0254, "rad": 1.00}

    data = file_io.load_data(filepath, unit_conversion_factors)
    assert np.array_equal(data.index.values, np.array(range(6)))
    assert data.shape == (6, 19)
    assert isinstance(data.columns, pd.core.indexes.multi.MultiIndex)
    assert data.columns.nlevels == 4

    _, units = file_io.load_data(
        filepath, unit_conversion_factors, return_units=True)

    for item in unit_conversion_factors:
        assert item in units.unique()

    filepath = os.path.join(
        'tests', 'data', 'file_io',
        'test_load_data', 'no_units.csv')
    data_nounits = file_io.load_data(filepath, {})
    assert isinstance(data_nounits, pd.DataFrame)

    # test loading data with orientation 1
    filepath = os.path.join(
        'tests', 'data', 'file_io',
        'test_load_data', 'orient_1.csv')
    data = file_io.load_data(
        filepath, unit_conversion_factors,
        orientation=1, reindex=False)
    assert isinstance(data.index, pd.core.indexes.multi.MultiIndex)
    assert data.shape == (10, 2)
    assert data.index.nlevels == 4

    # try with reindexing
    data = file_io.load_data(
        filepath, unit_conversion_factors,
        orientation=1, reindex=True)
    assert np.array_equal(data.index.values, np.array(range(10)))


if __name__ == '__main__':
    pass
