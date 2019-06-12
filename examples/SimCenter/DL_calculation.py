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

# imports for Python 2.X support
from __future__ import division, print_function
import os, sys
import warnings
if sys.version.startswith('2'):
    range=xrange
    string_types = basestring
else:
    string_types = str

import sys, os, json, ntpath, posixpath, argparse
import numpy as np

import pelicun
from pelicun.control import FEMA_P58_Assessment, HAZUS_Assessment
from pelicun.file_io import write_SimCenter_DL_output

def replace_FG_IDs_with_FG_names(assessment, df):
	FG_list = sorted(assessment._FG_dict.keys())
	new_col_names = dict(
		(fg_id, fg_name) for (fg_id, fg_name) in 
		zip(np.arange(1, len(FG_list) + 1), FG_list))

	return df.rename(columns=new_col_names)

def run_pelicun(DL_input_path, EDP_input_path, output_path=None):

	DL_input_path = os.path.abspath(DL_input_path)
	EDP_input_path = os.path.abspath(EDP_input_path)
	
	# If the output dir was not specified, results are saved in the directory of
	# the input file.
	if output_path is None:
		output_path = ntpath.dirname(DL_input_path)

	# delete output files from previous runs (if needed)
	files_to_delete = [
		'DL_summary.csv',
        'DL_summary_stats.csv',
        'DMG.csv',
        'DMG_agg.csv',
        'DV_red_tag.csv',
        'DV_red_tag_agg.csv',
        'DV_rec_cost.csv',
        'DV_rec_cost_agg.csv',
        'DV_rec_time.csv',
        'DV_rec_time_agg.csv',
        'DV_injuries_0.csv',
        'DV_injuries_0_agg.csv',
        'DV_injuries_1.csv',
        'DV_injuries_1_agg.csv',
	]
	for file_name in files_to_delete:
		try:
			os.remove(posixpath.join(output_path, file_name))
		except:
			pass

	# read the type of assessment from the DL input file
	with open(DL_input_path, 'r') as f:
		DL_input = json.load(f)

	DL_method = DL_input['LossModel']['DLMethod']

	if DL_method == 'FEMA P58':
		A = FEMA_P58_Assessment()
	elif DL_method == 'HAZUS MH':
		A = HAZUS_Assessment()

	A.read_inputs(DL_input_path, EDP_input_path, verbose=False)

	A.define_random_variables()

	A.define_loss_model()

	A.calculate_damage()

	A.calculate_losses()

	A.aggregate_results()

	try:
		write_SimCenter_DL_output(
			posixpath.join(output_path, 'DL_summary.csv'), A._SUMMARY, 
			index_name='#Num', collapse_columns=True)

		write_SimCenter_DL_output(
			posixpath.join(output_path, 'DL_summary_stats.csv'), A._SUMMARY, 
			index_name='attribute', collapse_columns=True,  stats_only=True)

		EDPs = sorted(A._EDP_dict.keys())
		write_SimCenter_DL_output(
			posixpath.join(output_path, 'EDP.csv'), 
			A._EDP_dict[EDPs[0]]._RV.samples, 
			index_name='#Num', collapse_columns=False)
		
		DMG_mod = replace_FG_IDs_with_FG_names(A, A._DMG)
		write_SimCenter_DL_output(
			posixpath.join(output_path, 'DMG.csv'), DMG_mod,
			index_name='#Num', collapse_columns=False)

		write_SimCenter_DL_output(
			posixpath.join(output_path, 'DMG_agg.csv'), 
			DMG_mod.T.groupby(level=0).aggregate(np.sum).T,
			index_name='#Num', collapse_columns=False)

		DV_mods, DV_names = [], []
		for key in A._DV_dict.keys():
			if key != 'injuries':
				DV_mods.append(replace_FG_IDs_with_FG_names(A, A._DV_dict[key]))
				DV_names.append('DV_{}'.format(key))
			else:
				for i in range(2 if DL_method == 'FEMA P58' else 4):
					DV_mods.append(replace_FG_IDs_with_FG_names(A, A._DV_dict[key][i]))
					DV_names.append('DV_{}_{}'.format(key, i))

		for DV_mod, DV_name in zip(DV_mods, DV_names):
			write_SimCenter_DL_output(
			posixpath.join(output_path, DV_name+'.csv'), DV_mod, 
			index_name='#Num', collapse_columns=False)

			write_SimCenter_DL_output(
			posixpath.join(output_path, DV_name+'_agg.csv'), 
			DV_mod.T.groupby(level=0).aggregate(np.sum).T,
			index_name='#Num', collapse_columns=False)

	except:
		print("ERROR when trying to create DL output files.")

	return 0

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--filename_DL_input')
	parser.add_argument('--filename_EDP_input')
	parser.add_argument('--filename_DM_output')
	parser.add_argument('--filename_DV_output')
	parser.add_argument('--dirname_output')
	args = parser.parse_args()

	print(args.dirname_output)
	sys.exit(run_pelicun(
		args.filename_DL_input, args.filename_EDP_input,
		args.dirname_output
		))