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

import pickle
import itertools
import os
import re
import inspect
import pytest
import numpy as np
from scipy.stats import norm
from pelicun import file_io

# for tests, we sometimes create things or call them just to see if
# things would work, so the following are irrelevant:

# pylint: disable=useless-suppression
# pylint: disable=unused-variable
# pylint: disable=pointless-statement


# The tests maintain the order of definitions of the `file_io.py` file.

def test_float_or_None():
    # Test with a string that can be converted to a float
    assert file_io.float_or_None('3.14') == 3.14

    # Test with a string that represents an integer
    assert file_io.float_or_None('42') == 42.0

    # Test with a string that represents a negative number
    assert file_io.float_or_None('-3.14') == -3.14

    # Test with a string that can't be converted to a float
    assert file_io.float_or_None('hello') == None

    # Test with an empty string
    assert file_io.float_or_None('') == None


def test_int_or_None():

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


if __name__ == '__main__':
    pass
