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

import os, sys, inspect, shutil
current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,os.path.dirname(parent_dir))

from pelicun.file_io import *

# -----------------------------------------------------------------------------
# read_SimCenter_DL_input
# -----------------------------------------------------------------------------

def test_read_SimCenter_DL_input_minimum_input():
    """
    Test if the minimum input is read without producing any errors and also
    check if some warnings are shown that draw attention to the lack of
    potentially important information in the input file.
    """

    # load the reference results
    with open('resources/io testing/ref/ref_DL_input_min.json') as f:
        ref_DL = json.load(f)

    # read the input file and check for at least one warning
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                          'test_DL_input_min.json',
                                          verbose=False)

    # make sure the paths under data sources point to the right locations
    assert test_DL['data_sources']['path_CMP_data'] == \
           pelicun_path + '/resources/FEMA P58 first edition/DL json/'
    test_DL.pop('data_sources', None)

    # check if the returned dictionary is appropriate
    for key in set(list(ref_DL.keys())+list(test_DL.keys())):
        assert ref_DL[key] == test_DL[key]


def test_read_SimCenter_DL_input_full_input():
    """
    Test if the full input (i.e. all possible fields populated and all supported
    decision variables turned on) is read without producing any errors or
    warnings.
    """

    # load the reference results
    with open('resources/io testing/ref/ref_DL_input_full.json') as f:
        ref_DL = json.load(f)

    # read the input file
    test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                      'test_DL_input_full.json', verbose=False)

    # make sure the paths under data sources point to the right locations
    assert test_DL['data_sources']['path_CMP_data'] == \
           pelicun_path + '/resources/FEMA P58 first edition/DL json/'
    assert test_DL['data_sources']['path_POP_data'] == \
           pelicun_path + '/resources/FEMA P58 first edition/population.json'
    test_DL.pop('data_sources', None)

    # check if the returned dictionary is appropriate
    for key in set(list(ref_DL.keys()) + list(test_DL.keys())):
        assert ref_DL[key] == test_DL[key]

def test_read_SimCenter_DL_input_non_standard_units():
    """
    Test if the inputs are properly converted when non-standard units are used.
    """

    # load the reference results
    with open('resources/io testing/ref/ref_DL_input_ns_units.json') as f:
        ref_DL = json.load(f)

    # read the input file and check for at least one warning
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                          'test_DL_input_ns_units.json',
                                          verbose=False)

    # make sure the path under data sources points to the right location
    assert test_DL['data_sources']['path_CMP_data'] == \
           pelicun_path + '/resources/FEMA P58 first edition/DL json/'
    test_DL.pop('data_sources', None)

    # check if the returned dictionary is appropriate
    for key in set(list(ref_DL.keys()) + list(test_DL.keys())):
        assert ref_DL[key] == test_DL[key]

def test_read_SimCenter_DL_input_unknown_unit():
    """
    Test if a warning is shown if the input file contains an unknown unit type.
    """

    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                          'test_DL_input_unknown_unit.json',
                                          verbose=False)

def test_read_SimCenter_DL_input_injuries_only():
    """
    Test if the inputs are read properly if the user is only interested in
    calculating injuries.
    """

    # read the input file and check for at least one error because the plan
    # area is not specified in the file
    with pytest.raises(ValueError) as e_info:
        test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                          'test_DL_input_injuries_only.json',
                                          verbose=False)

    # now test if warnings are shown if the plan area is in the file, but the
    # population is only specified for the first two stories

    # load the reference results
    with open('resources/io testing/ref/'
                         'ref_DL_input_injuries_missing_pop.json') as f:
        ref_DL = json.load(f)

    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/io testing/test/test_DL_input_injuries_missing_pop.json',
            verbose=False)

    # remove the data_sources entry (it has already been tested)
    test_DL.pop('data_sources', None)

    # check if the returned dictionary is appropriate
    for key in set(list(ref_DL.keys()) + list(test_DL.keys())):
        assert ref_DL[key] == test_DL[key]

    # now test if an error is shown if other pieces of data are missing

    # load the reference results
    with open('resources/io testing/ref/'
                         'ref_DL_input_injuries_missing_data.json') as f:
        ref_DL = json.load(f)

    with pytest.raises(ValueError) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/io testing/test/test_DL_input_injuries_missing_data.json',
            verbose=False)


def test_read_SimCenter_DL_input_unknown_component_unit():
    """
    Test if an error is shown if the input file contains an unknown unit type
    for one of the components.
    """

    with pytest.raises(ValueError) as e_info:
        test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                          'test_DL_input_unknown_comp_unit.json',
                                          verbose=False)

def test_read_SimCenter_DL_input_no_realizations():
    """
    Test if an error is shown if the input file contains no information about
    the number of realizations to run.
    """

    with pytest.raises(ValueError) as e_info:
        test_DL = read_SimCenter_DL_input('resources/io testing/test/'
                                          'test_DL_input_no_realizations.json',
                                          verbose=False)

# -----------------------------------------------------------------------------
# read_SimCenter_EDP_input
# -----------------------------------------------------------------------------

def test_read_SimCenter_EDP_input():
    """
    Test if the function can read the provided set of EDPs from a file and
    return them and the corresponding additional information in a properly
    structured format.
    """
    # load the reference results
    with open('resources/io testing/ref/ref_EDP_input.json') as f:
        ref_EDP = json.load(f)

    # read the input file
    test_EDP = read_SimCenter_EDP_input(
        'resources/io testing/test/test_EDP_input.out',
        EDP_kinds=('PID', 'PFA', 'RD', 'PRD'),
        units = dict(PID=1., PFA=9.81, RD=1., PRD=0.2),
        verbose=False)

    # check if the returned dictionary is appropriate
    assert ref_EDP == test_EDP

# -----------------------------------------------------------------------------
# read_population_distribution
# -----------------------------------------------------------------------------

def test_read_population_distribution():
    """
    Test if the population distribution is read according to the specified
    occupancy type and if the peak population is properly converted from
    1/1000*ft2 to 1/m2.
    """

    # load the reference results
    with open('resources/io testing/ref/ref_POP_data.json') as f:
        ref_POP = json.load(f)

    # read the input file
    test_POP = read_population_distribution(
        'resources/io testing/test/test_POP_data.json',
        occupancy='Commercial Office', verbose=False)

    # check if the returned dictionary is appropriate
    assert ref_POP == test_POP

# -----------------------------------------------------------------------------
# read_component_DL_data
# -----------------------------------------------------------------------------

def test_read_component_DL_data():
    """
    Test if the component data is properly read from the resource xml files.
    Use a series of tests to see if certain component features trigger proper
    warnings or errors.
    """

    # basic case with a typical component
    comp_info = {
        "B1071.011": {
            "locations": [2, 2, 3, 3],
            "directions": [1, 2, 1, 2],
            "quantities": [50.0, 50.0, 50.0, 50.0],
            "csg_weights" : [
                [1.0,],
                [1.0,],
                [1.0,],
                [1.0,]],
            "cov"         : ["0.1", "0.1", "0.1", "0.1"],
            "distribution": ["normal", "normal", "normal", "normal"],
            "unit"        : "ft2"
        }
    }

    # read the component data
    test_CMP = read_component_DL_data(
        '../resources/FEMA P58 first edition/DL json/',
        comp_info)

    # load the reference results
    with open('resources/io testing/ref/ref_CMP_B1071.011.json',
              'r') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    for key in set(list(ref_CMP.keys())+list(test_CMP.keys())):
        assert ref_CMP[key] == test_CMP[key]

    # acceleration-sensitive component with injuries
    comp_info = {
        "C3032.001a": {
            "locations"   : [1, 1],
            "directions"  : [1, 2],
            "quantities"  : [125.0, 125.0],
            "csg_weights" : [
                [0.2, 0.8],
                [0.4, 0.6],
            ],
            "cov"         : ["1.0", "1.0"],
            "distribution": ["normal", "normal"],
            "unit"        : "ft2"
        }
    }

    # read the component data
    test_CMP = read_component_DL_data(
        '../resources/FEMA P58 first edition/DL json/',
        comp_info)

    # load the reference results
    with open('resources/io testing/ref/ref_CMP_C3032.001a.json',
              'r') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    for key in set(list(ref_CMP.keys()) + list(test_CMP.keys())):
        assert ref_CMP[key] == test_CMP[key]

    # component with simultaneous damage states
    comp_info = {
        "D1014.011": {
            "locations"   : [1, 1],
            "directions"  : [1, 2],
            "quantities" : [3.0, 7.0],
            "csg_weights": [
                [2./3., 1./3.],
                [1./7., 6./7.],
            ],
            "cov"         : ["1.0", "1.0"],
            "distribution": ["normal", "normal"],
            "unit"        : "ea"
        }
    }

    # read the component data
    test_CMP = read_component_DL_data(
        '../resources/FEMA P58 first edition/DL json/',
        comp_info)

    # load the reference results
    with open('resources/io testing/ref/ref_CMP_D1014.011.json',
              'r') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    assert ref_CMP["D1014.011"] == test_CMP["D1014.011"]

    # component with mutually exclusive damage states
    comp_info = {
        "B1035.051": {
            "locations"   : [1, 1],
            "directions"  : [1, 2],
            "quantities" : [3.0, 7.0],
            "csg_weights": [
                [2./3., 1./3.],
                [1./7., 6./7.],
            ],
            "cov"         : ["1.0", "1.0"],
            "distribution": ["normal", "normal"],
            "unit"        : "ea"
        }
    }

    # read the component data
    test_CMP = read_component_DL_data(
        '../resources/FEMA P58 first edition/DL json/',
        comp_info)

    # load the reference results
    with open('resources/io testing/ref/ref_CMP_B1035.051.json',
              'r') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    assert test_CMP == ref_CMP

    # an incomplete component shall not get parsed and shall produce a warning
    comp_info = {
        "E2022.023": {
            "locations"   : [1, 1],
            "directions"  : [1, 2],
            "quantities" : [3.0, 7.0],
            "csg_weights": [
                [2./3., 1./3.],
                [1./7., 6./7.],
            ],
            "cov"         : ["1.0", "1.0"],
            "distribution": ["normal", "normal"],
            "unit"        : "ea"
        }
    }

    # read the component data
    with pytest.warns(UserWarning) as e_info:
        test_CMP = read_component_DL_data(
            '../resources/FEMA P58 first edition/DL json/',
            comp_info)

    assert test_CMP == {}

    # a component with unknown EDP shall not get parsed and shall produce a warning
    comp_info = {
        "B1042.001a": {
            "locations"   : [1, 1],
            "directions"  : [1, 2],
            "quantities" : [3.0, 7.0],
            "csg_weights": [
                [2./3., 1./3.],
                [1./7., 6./7.],
            ],
            "cov"         : ["1.0", "1.0"],
            "distribution": ["normal", "normal"],
            "unit"        : "ea"
        }
    }

    # read the component data
    with pytest.warns(UserWarning) as e_info:
        test_CMP = read_component_DL_data(
            '../resources/FEMA P58 first edition/DL json/',
            comp_info)

    assert test_CMP == {}

# -----------------------------------------------------------------------------
# convert P58 data to JSON
# -----------------------------------------------------------------------------

def test_convert_P58_data_to_json():
    """
    Test if the damage and loss data from the FEMA P58 project is properly
    converted into the SimCenter JSON format using xml and xlsx files avialable
    from ATC (the test uses a subset of the files).
    """

    data_dir = 'resources/io testing/P58 converter/source/'
    ref_dir = 'resources/io testing/P58 converter/ref/'
    test_dir = 'test/'

    try:
        # convert the files in the data folder
        os.mkdir(test_dir)
        convert_P58_data_to_json(data_dir, test_dir)

        # collect the prepared reference files
        ref_files = sorted(os.listdir(ref_dir))

        # collect the converted files
        test_files = sorted(os.listdir(test_dir))

        # compare the reference files to the converted ones
        for test, ref in zip(test_files, ref_files):
            with open(os.path.join(test_dir,test),'r') as f_test:
                with open(os.path.join(ref_dir,ref),'r') as f_ref:
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

    data_dir = 'resources/io testing/HAZUS creator/source/'
    ref_dir = 'resources/io testing/HAZUS creator/ref/'
    test_dir = 'test/'

    try:
        # convert the files in the data folder
        os.mkdir(test_dir)
        os.mkdir(test_dir+'DL json/')
        create_HAZUS_EQ_json_files(data_dir, test_dir)

        # collect the prepared reference files
        ref_files = sorted(os.listdir(ref_dir+'DL json/'))

        # collect the converted files
        test_files = sorted(os.listdir(test_dir+'DL json/'))

        # compare the reference files to the converted ones
        for test, ref in zip(test_files, ref_files):
            with open(os.path.join(test_dir + 'DL json/', test), 'r') as f_test:
                with open(os.path.join(ref_dir + 'DL json/', ref), 'r') as f_ref:
                    #print(test, ref)
                    assert json.load(f_test) == json.load(f_ref)

        # compare the population files
        with open(os.path.join(test_dir, 'population.json'), 'r') as f_test:
            with open(os.path.join(ref_dir, 'population.json'), 'r') as f_ref:
                #print(test, ref)
                assert json.load(f_test) == json.load(f_ref)

    finally:
        #pass
        shutil.rmtree(test_dir)

# -----------------------------------------------------------------------------
# write_SimCenter_DL_output
# -----------------------------------------------------------------------------

