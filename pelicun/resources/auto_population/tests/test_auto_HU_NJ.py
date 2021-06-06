# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Kuanshi Zhong

"""
These are unit and integration tests on the auto_HU_NJ module.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import json
import os
import inspect
# Importing auto_HU_NJ module
from auto_HU_NJ import *
# Current directory
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Test input directory
base_input_path = 'resources'

def test_parse_BIM():
    """
    Testing the parse_BIM function.
    """
    # Testing the ruleset for Hurricane-Prone Region (HPR)
    res = []
    ref = [1, 0, 1, 0]
    for i in range(4):
        BIM_dir = os.path.join(cur_dir, base_input_path, 'BIM_Data',
                               'parse_BIM_test_' + str(i+1) + '.json')
        with open(BIM_dir) as f:
            BIM_input = json.load(f)
        BIM_output = parse_BIM(BIM_input['GI'])
        res.append(int(BIM_output['HPR']))
    # Check
    assert_allclose(res, ref, atol=1e-5)

    # Testing the ruleset for Wind Borne Debris (WBD)
    res = []
    ref = [0, 0, 0, 0, 1, 1, 1, 1]
    for i in range(8):
        BIM_dir = os.path.join(cur_dir, base_input_path, 'BIM_Data',
                               'parse_BIM_test_' + str(i+1) + '.json')
        with open(BIM_dir) as f:
            BIM_input = json.load(f)
        BIM_output = parse_BIM(BIM_input['GI'])
        res.append(int(BIM_output['WBD']))
    # Check
    assert_allclose(res, ref, atol=1e-5)

    # Testing the ruleset for terrain
    res = []
    ref = [3, 15, 35, 70, 3, 15, 35, 70]
    for i in range(8):
        BIM_dir = os.path.join(cur_dir, base_input_path, 'BIM_Data',
                               'parse_BIM_test_' + str(i+1) + '.json')
        with open(BIM_dir) as f:
            BIM_input = json.load(f)
        BIM_output = parse_BIM(BIM_input['GI'])
        res.append(int(BIM_output['terrain']))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_building_class():
    """
    Testing the building class function.
    """
    # Testing the ruleset for classifying Hazus building class
    res = []
    ref_class = ['WSF', 'WMUH', 'SERB', 'SECB', 'SPMB', 'CERB', 'CECB', 'MSF',
                 'MERB', 'MECB', 'MLRI', 'MMUH', 'MLRM']
    ref = np.ones(13)
    for i in range(13):
        data_dir = os.path.join(cur_dir, base_input_path, 'BuildingClass_Data',
                               'building_class_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = building_class(tmp)
        print(data_output)
        res.append(int(data_output == ref_class[i]))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_WSF_config():
    """
    Testing the WSF_config function.
    """
    res = []
    ref_class = ['WSF2_gab_0_8d_tnail_no',
                 'WSF2_gab_1_8d_tnail_no',
                 'WSF2_hip_1_8d_tnail_no',
                 'WSF2_hip_0_8d_tnail_no',
                 '8s_strap_no',
                 '8s_strap_no',
                 '8s_tnail_no',
                 '8s_strap_sup',
                 '8d_strap_std',
                 '8d_tnail_wkd',
                 'WSF1']
    ref = np.ones(11)
    for i in range(11):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'wsf_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = WSF_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_WMUH_config():
    """
    Testing the WMUH_config function.
    """
    res = []
    ref_class = ['WMUH2_flt_spm_god_null',
                 'WMUH2_flt_spm_god_null',
                 'WMUH2_flt_spm_god_null',
                 'WMUH2_gab_null_null_1',
                 'WMUH2_hip_null_null_1',
                 'WMUH2_gab_null_null_0',
                 'WMUH2_hip_null_null_0',
                 'WMUH2_flt_spm_por_null',
                 'WMUH2_flt_bur_por_null',
                 'WMUH2_flt_spm_god_null_8s',
                 'WMUH2_flt_spm_god_null_8d',
                 'WMUH2_flt_spm_god_null_8s',
                 'WMUH2_flt_spm_god_null_8d',
                 'strap',
                 'tnail',
                 'tnail',
                 'tnail_1',
                 'WMUH3']
    ref = np.ones(18)
    for i in range(18):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'wmuh_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = WMUH_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_MSF_config():
    """
    Testing the MSF_config function.
    """
    res = []
    ref_class = ['nav_1',
                 'nav_0',
                 '8s',
                 '8d',
                 '8s',
                 '8d',
                 'MSF2']
    ref = np.ones(7)
    for i in range(7):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'msf_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = MSF_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_MMUH_config():
    """
    Testing the MMUH_config function.
    """
    res = []
    ref_class = ['flt_1_spm_god',
                 'flt_1_spm_por',
                 'flt_1_bur_por',
                 '8s_strap',
                 '8d_strap',
                 '8d_tnail',
                 '8s_strap',
                 '8d_tnail',
                 'MMUH3']
    ref = np.ones(9)
    for i in range(9):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'mmuh_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = MMUH_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_MLRM_config():
    """
    Testing the MLRM_config function.
    """
    res = []
    ref_class = ['spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 '6d_god',
                 '6d_por',
                 'std',
                 'sup',
                 'A_1_sup',
                 'sgl',
                 'mlt']
    ref = np.ones(12)
    for i in range(12):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'mlrm_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = MLRM_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_MLRI_config():
    """
    Testing the MLRI_config function.
    """
    res = []
    ref_class = ['sup',
                 'std',
                 'god',
                 'por',
                 'god',
                 'por']
    ref = np.ones(6)
    for i in range(6):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'mlri_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = MLRI_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_MERB_config():
    """
    Testing the MERB_config function.
    """
    res = []
    ref_class = ['bur',
                 'spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 'std',
                 'sup',
                 'low',
                 'med',
                 'hig',
                 'MERBL',
                 'MERBM',
                 'MERBH']
    ref = np.ones(14)
    for i in range(14):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'merb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = MERB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_MECB_config():
    """
    Testing the MECB_config function.
    """
    res = []
    ref_class = ['bur',
                 'spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 'std',
                 'sup',
                 'low',
                 'med',
                 'hig',
                 'MECBL',
                 'MECBM',
                 'MECBH']
    ref = np.ones(14)
    for i in range(14):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'mecb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = MECB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_CECB_config():
    """
    Testing the CECB_config function.
    """
    res = []
    ref_class = ['bur',
                 'spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 'low',
                 'med',
                 'hig',
                 'CECBL',
                 'CECBM',
                 'CECBH']
    ref = np.ones(12)
    for i in range(12):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'cecb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = CECB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_CERB_config():
    """
    Testing the CERB_config function.
    """
    res = []
    ref_class = ['bur',
                 'spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 'low',
                 'med',
                 'hig',
                 'CERBL',
                 'CERBM',
                 'CERBH']
    ref = np.ones(12)
    for i in range(12):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'cerb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = CERB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_SPMB_config():
    """
    Testing the SPMB_config function.
    """
    res = []
    ref_class = ['god',
                 'por',
                 'std',
                 'sup',
                 'SPMBS',
                 'SPMBM',
                 'SPMBL']
    ref = np.ones(7)
    for i in range(7):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'spmb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = SPMB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_SECB_config():
    """
    Testing the SECB_config function.
    """
    res = []
    ref_class = ['bur',
                 'spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 'std',
                 'sup',
                 'low',
                 'med',
                 'hig',
                 'SECBL',
                 'SECBM',
                 'SECBH']
    ref = np.ones(14)
    for i in range(14):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'secb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = SECB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_SERB_config():
    """
    Testing the SERB_config function.
    """
    res = []
    ref_class = ['bur',
                 'spm',
                 'bur',
                 'C',
                 'D',
                 'A',
                 'std',
                 'sup',
                 'low',
                 'med',
                 'hig',
                 'SERBL',
                 'SERBM',
                 'SERBH']
    ref = np.ones(14)
    for i in range(14):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'secb_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = SERB_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_FL_config():
    """
    Testing the FL_config function.
    """
    res = []
    ref_class = ['raz',
                 'cvz',
                 'sl',
                 'bn',
                 'bw']
    ref = np.ones(5)
    for i in range(5):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'fl_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        data_output = FL_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)


def test_Assm_config():
    """
    Testing the Assm_config function.
    """
    res = []
    ref_class = ['raz',
                 'caz',
                 'cvz',
                 '1',
                 '0']
    ref = np.ones(5)
    for i in range(5):
        data_dir = os.path.join(cur_dir, base_input_path, 'Config_Data',
                               'assm_test_' + str(i+1) + '.json')
        with open(data_dir) as f:
            data_input = json.load(f)
        tmp = parse_BIM(data_input['GI'])
        tmp2, data_output = Assm_config(tmp)
        print(data_output)
        res.append(int(ref_class[i] in data_output))
    # Check
    assert_allclose(res, ref, atol=1e-5)
