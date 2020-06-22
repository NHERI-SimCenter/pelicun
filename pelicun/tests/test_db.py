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
This subpackage performs unit tests on the file_io module of pelicun.

"""

import pytest

import os, sys, inspect
from pathlib import Path
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir))

from pelicun.db import *

# -----------------------------------------------------------------------------
# convert P58 data to JSON
# -----------------------------------------------------------------------------

def test_convert_P58_data_to_json():
    """
    Test if the damage and loss data from the FEMA P58 project is properly
    converted into the SimCenter JSON format using xml and xlsx files avialable
    from ATC (the test uses a subset of the files).
    """

    data_dir = Path('resources/io testing/P58 converter/source/').resolve()
    ref_dir  = Path('resources/io testing/P58 converter/ref/').resolve()
    test_dir = Path('test/').resolve()

    try:
        # convert the files in the data folder
        os.mkdir(test_dir)
        convert_P58_data_to_json(data_dir, test_dir)

        # collect the prepared reference files
        ref_files = sorted(os.listdir(ref_dir))

        # collect the converted files
        test_files = sorted(os.listdir(test_dir / 'DL json'))

        # compare the reference files to the converted ones
        for test, ref in zip(test_files, ref_files):
            with open((test_dir / 'DL json') / test,'r') as f_test:
                with open(ref_dir / ref,'r') as f_ref:
                    #print(test, ref)
                    assert json.load(f_test) == json.load(f_ref)

    finally:
        #pass
        shutil.rmtree(test_dir)

# -----------------------------------------------------------------------------
# create HAZUS JSON files
# -----------------------------------------------------------------------------

def test_create_HAZUS_json_files():
    """
    Test if the damage and loss data from HAZUS is properly converted into the
    SimCenter JSON format the prepared raw HAZUS JSON files (the test uses a
    subset of the HAZUS input data).
    """

    data_dir = Path('resources/io testing/HAZUS creator/source/').resolve()
    ref_dir  = Path('resources/io testing/HAZUS creator/ref/').resolve()
    test_dir = Path('test/').resolve()

    print(test_dir)

    try:
        # convert the files in the data folder
        os.mkdir(test_dir)
        create_HAZUS_EQ_json_files(data_dir, test_dir)

        # collect the prepared reference files
        ref_files = sorted(os.listdir(ref_dir / 'DL json'))

        # collect the converted files
        test_files = sorted(os.listdir(test_dir / 'DL json/'))

        # compare the reference files to the converted ones
        for test, ref in zip(test_files, ref_files):
            with open((test_dir / 'DL json') / test, 'r') as f_test:
                with open((ref_dir / 'DL json') /  ref, 'r') as f_ref:
                    #print(test, ref)
                    assert json.load(f_test) == json.load(f_ref)

        # compare the population files
        with open(test_dir / 'population.json', 'r') as f_test:
            with open(ref_dir / 'population.json', 'r') as f_ref:
                #print(test, ref)
                assert json.load(f_test) == json.load(f_ref)

    finally:
        #pass
        shutil.rmtree(test_dir)