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
    with open('resources/ref_DL_input_min.json') as f:
        ref_DL = json.load(f)
    
    # read the input file and check for at least one warning
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input('resources/test_DL_input_min.json', 
                                        verbose=False)
        
    # check if the returned dictionary is appropriate
    assert ref_DL == test_DL


def test_read_SimCenter_DL_input_full_input():
    """
    Test if the full input (i.e. all possible fields populated and all supported 
    decision variables turned on) is read without producing any errors or 
    warnings.
    """

    # load the reference results
    with open('resources/ref_DL_input_full.json') as f:
        ref_DL = json.load(f)

    # read the input file
    test_DL = read_SimCenter_DL_input('resources/test_DL_input_full.json',
                                      verbose=False)

    # check if the returned dictionary is appropriate
    assert ref_DL == test_DL
    
def test_read_SimCenter_DL_input_non_standard_units():
    """
    Test if the inputs are properly converted when non-standard units are used.
    """

    # load the reference results
    with open('resources/ref_DL_input_ns_units.json') as f:
        ref_DL = json.load(f)

    # read the input file and check for at least one warning
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/test_DL_input_ns_units.json', verbose=False)

    # check if the returned dictionary is appropriate
    assert ref_DL == test_DL
    
def test_read_SimCenter_DL_input_unknown_unit():
    """
    Test if a warning is shown if the input file contains an unknown unit type.
    """
    
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/test_DL_input_unknown_unit.json', verbose=False)
        
def test_read_SimCenter_DL_input_injuries_only():
    """
    Test if the inputs are read properly if the user is only interested in 
    calculating injuries.
    """

    # load the reference results
    with open('resources/ref_DL_input_injuries_only.json') as f:
        ref_DL = json.load(f)

    # read the input file and check for at least one warning because the plan
    # area is not specified in the file 
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/test_DL_input_injuries_only.json', verbose=False)

    # check if the returned dictionary is appropriate
    assert ref_DL == test_DL
    
    # now test if warnings are shown if the plan area is in the file, but 
    # other pieces of data are missing
    
    # load the reference results
    with open('resources/ref_DL_input_injuries_missing_data.json') as f:
        ref_DL = json.load(f)
    
    with pytest.warns(UserWarning) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/test_DL_input_injuries_missing_data.json', 
            verbose=False)

    # check if the returned dictionary is appropriate
    assert ref_DL == test_DL


def test_read_SimCenter_DL_input_unknown_component_unit():
    """
    Test if an error is shown if the input file contains an unknown unit type
    for one of the components.
    """

    with pytest.raises(ValueError) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/test_DL_input_unknown_comp_unit.json',
            verbose=False)
        
def test_read_SimCenter_DL_input_no_realizations():
    """
    Test if an error is shown if the input file contains no information about 
    the number of realizations to run.
    """

    with pytest.raises(ValueError) as e_info:
        test_DL = read_SimCenter_DL_input(
            'resources/test_DL_input_no_realizations.json', verbose=False)
              
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
    with open('resources/ref_EDP_input.json') as f:
        ref_EDP = json.load(f)

    # read the input file 
    test_EDP = read_SimCenter_EDP_input(
        'resources/test_EDP_input.csv',
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
    with open('resources/ref_POP_data.json') as f:
        ref_POP = json.load(f)

    # read the input file 
    test_POP = read_population_distribution(
        'resources/test_POP_data.json',
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
        "B1071.011" : {
            "quantities"  : [1.0, 1.0],
            "csg_weights" : [0.5, 0.5],
            "dirs"        : [0, 1],
            "kind"        : "structural",
            "distribution": "normal",
            "cov"         : 0.1,
            "unit"        : [100.0, "SF"],
            "locations"   : [2, 3]
        },
    }
        
    # read the component data
    test_CMP = read_component_DL_data(
        '../../resources/component DL/FEMA P58 first edition/', comp_info)

    # load the reference results
    with open('resources/ref_CMP_B1071.011.json') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    assert test_CMP == ref_CMP

    # acceleration-sensitive component with injuries
    comp_info = {
        "E2022.023": {
            "quantities"  : [1.0],
            "csg_weights" : [0.1, 0.2, 0.3, 0.4],
            "dirs"        : [0, 1, 1, 0],
            "kind"        : "non-structural",
            "distribution": "normal",
            "cov"         : 1.0,
            "unit"        : [1.0, "ea"],
            "locations"   : [1]
        },
    }

    # read the component data
    with pytest.warns(UserWarning) as e_info:
        test_CMP = read_component_DL_data(
            '../../resources/component DL/FEMA P58 first edition/', 
            comp_info)

    # load the reference results
    with open('resources/ref_CMP_E2022.023.json') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    assert test_CMP == ref_CMP

    # component with simultaneous damage states
    comp_info = {
        "D1014.011": {
            "quantities"  : [1.0],
            "csg_weights" : [0.2, 0.1, 0.1, 0.6],
            "dirs"        : [0, 0, 1, 1],
            "kind"        : "non-structural",
            "distribution": "normal",
            "cov"         : 1.0,
            "unit"        : [1.0, "ea"],
            "locations"   : [1]
        },
    }

    # read the component data
    test_CMP = read_component_DL_data(
        '../../resources/component DL/FEMA P58 first edition/',
        comp_info)

    # load the reference results
    with open('resources/ref_CMP_D1014.011.json') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    assert test_CMP == ref_CMP

    # component with mutually exclusive damage states
    comp_info = {
        "B1035.051": {
            "quantities"  : [1.0],
            "csg_weights" : [0.2, 0.1, 0.1, 0.6],
            "dirs"        : [0, 0, 1, 1],
            "kind"        : "structural",
            "distribution": "normal",
            "cov"         : 1.0,
            "unit"        : [1.0, "ea"],
            "locations"   : [1]
        },
    }

    # read the component data
    test_CMP = read_component_DL_data(
        '../../resources/component DL/FEMA P58 first edition/',
        comp_info)

    # load the reference results
    with open('resources/ref_CMP_B1035.051.json') as f:
        ref_CMP = json.load(f)

    # check if the returned dictionary is appropriate
    assert test_CMP == ref_CMP
    
# -----------------------------------------------------------------------------
# write_SimCenter_DL_output
# -----------------------------------------------------------------------------

