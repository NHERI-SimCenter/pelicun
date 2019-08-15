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

import json, ntpath, posixpath, argparse
import numpy as np
import pandas as pd

idx = pd.IndexSlice

import pelicun
from pelicun.control import FEMA_P58_Assessment, HAZUS_Assessment
from pelicun.file_io import write_SimCenter_DL_output

def replace_FG_IDs_with_FG_names(assessment, df):
	FG_list = sorted(assessment._FG_dict.keys())
	new_col_names = dict(
		(fg_id, fg_name) for (fg_id, fg_name) in 
		zip(np.arange(1, len(FG_list) + 1), FG_list))

	return df.rename(columns=new_col_names)

ap_DesignLevel = {
	1950: 'Pre-Code',
	1970: 'Low-Code',
	1990: 'Moderate-Code',
	2100: 'High-Code'
}

ap_Occupancy = {
	'Residential': "RES1",
	'Retail': "COM1"
}

convert_design_level = {
        'High-Code'    : 'HC',
        'Moderate-Code': 'MC',
        'Low-Code'     : 'LC',
        'Pre-Code'     : 'PC'
    }

convert_dv_name = {
	'DV_rec_cost': 'Reconstruction Cost',
	'DV_rec_time': 'Reconstruction Time',
	'DV_injuries_0': 'Injuries lvl. 1',
	'DV_injuries_1': 'Injuries lvl. 2',
	'DV_injuries_2': 'Injuries lvl. 3',
	'DV_injuries_3': 'Injuries lvl. 4',
	'DV_red_tag': 'Red Tag ',
}

def auto_populate(DL_input_path):

	with open(DL_input_path, 'r') as f:
		DL_input = json.load(f)

	bt = DL_input['GI']['structType']
	ot = ap_Occupancy[DL_input['GI']['occupancy']]

	loss_dict = {
		'DLMethod': 'HAZUS MH',
		'BuildingDamage': {
			'ReplacementCost': DL_input['GI']['replacementCost'],
			'ReplacementTime': DL_input['GI']['replacementTime'],
			'StructureType': bt,
		},
		'UncertaintyQuantification': {
			'Realizations': "10000"
		},
		'Inhabitants': {
			'PeakPopulation': "1",
			'OccupancyType': ot
		},
		'Components': []
	}

	year_built = DL_input['GI']['yearBuilt']
	for year in sorted(ap_DesignLevel.keys()):
		if year_built <= year:
			loss_dict['BuildingDamage'].update(
				{'DesignLevel': ap_DesignLevel[year]})
			break
	dl = convert_design_level[loss_dict['BuildingDamage']['DesignLevel']]

	components = [
		{'ID': 'S-{}-{}-{}'.format(bt, dl ,ot), 'structural': True},
		{'ID': 'NSA-{}-{}'.format(dl ,ot),      'structural': False},
		{'ID': 'NSD-{}'.format(ot),             'structural': False}
	]

	loss_dict['Components'] = components

	DL_input.update({'LossModel':loss_dict})

	DL_ap_path = DL_input_path[:-5]+'_ap.json'

	with open(DL_ap_path, 'w') as f:
		json.dump(DL_input, f, indent = 2)

	return DL_input, DL_ap_path

def write_DM_output(DM_file_path, DMG_df):

	# Start with the probability of being in a particular damage state.
	# Here, the damage state of the building (asset) is defined as the highest 
	# damage state among the building components/component groups. This works 
	# well for a HAZUS assessment, but something more sophisticated is needed
	# for a FEMA P58 assessment.

	# Determine the probability of DS exceedance by collecting the DS from all 
	# components and assigning ones to all lower damage states.
	DMG_agg = DMG_df.T.groupby('DS').sum().T
	DMG_agg[DMG_agg > 0.0] = DMG_agg[DMG_agg > 0.0] / DMG_agg[DMG_agg > 0.0]
	
	cols = DMG_agg.columns
	for i in range(len(cols)):
	    filter = np.where(DMG_agg.iloc[:,i].values > 0.0)[0]
	    DMG_agg.iloc[filter,idx[0:i]] = 1.0

	# The P(DS=ds) probability is determined by subtracting consecutive DS 
	# exceedance probabilites. This will not work well for a FEMA P58 assessment
	# with Damage State Groups that include multiple Damage States.	    
	DMG_agg_mean = DMG_agg.describe().loc['mean',:]
	DS_0 = 1.0 - DMG_agg_mean['1-1']
	for i in range(len(DMG_agg_mean.index)-1):
	    DMG_agg_mean.iloc[i] = DMG_agg_mean.iloc[i] - DMG_agg_mean.iloc[i+1]
	    
	# Add the probability of no damage for convenience.
	DMG_agg_mean['0'] = DS_0
	DMG_agg_mean = DMG_agg_mean.sort_index()

	# Save the results in the output json file
	DM = {'aggregate': {}}

	for id in DMG_agg_mean.index:
	    DM['aggregate'].update({str(id): DMG_agg_mean[id]})

	# Now determine the probability of being in a damage state for individual 
	# components / component assemblies...
	DMG_mean = DMG_df.describe().loc['mean',:]

	# and save the results in the output json file.
	for FG in sorted(DMG_mean.index.get_level_values('FG').unique()):
	    DM.update({str(FG):{}})

	    for PG in sorted(
	    	DMG_mean.loc[idx[FG],:].index.get_level_values('PG').unique()):
	        DM[str(FG)].update({str(PG):{}})
	        
	        for DS in sorted(
	        	DMG_mean.loc[idx[FG, PG],:].index.get_level_values('DS').unique()):
	            DM[str(FG)][str(PG)].update({str(DS): DMG_mean.loc[(FG,PG,DS)]})            

	with open(DM_file_path, 'w') as f:
		json.dump(DM, f, indent = 2)

def write_DV_output(DV_file_path, DV_df, DV_name):

	DV_name = convert_dv_name[DV_name]

	try:
		with open(DV_file_path, 'r') as f:
			DV = json.load(f)
	except:
		DV = {}

	DV.update({DV_name: {}})

	DV_i = DV[DV_name]

	try:		
		DV_tot = DV_df.sum(axis=1).describe([0.1,0.5,0.9]).drop('count')
		DV_i.update({'total':{}})
		for stat in DV_tot.index:
			DV_i['total'].update({stat: DV_tot.loc[stat]})

		DV_stats = DV_df.describe([0.1,0.5,0.9]).drop('count')
		for FG in sorted(DV_stats.columns.get_level_values('FG').unique()):
		    DV_i.update({str(FG):{}})

		    for PG in sorted(
		    	DV_stats.loc[:,idx[FG]].columns.get_level_values('PG').unique()):
		        DV_i[str(FG)].update({str(PG):{}})
		        
		        for DS in sorted(
		        	DV_stats.loc[:,idx[FG, PG]].columns.get_level_values('DS').unique()):
		            DV_i[str(FG)][str(PG)].update({str(DS): {}}) 
		            DV_stats_i = DV_stats.loc[:,(FG,PG,DS)]
		            for stat in DV_stats_i.index:
		            	DV_i[str(FG)][str(PG)][str(DS)].update({
		            		stat: DV_stats_i.loc[stat]})
	except:
		pass

	with open(DV_file_path, 'w') as f:
		json.dump(DV, f, indent = 2)

def run_pelicun(DL_input_path, EDP_input_path, EVENT_input_path=None, 
	output_path=None, DM_file = 'DM.json', DV_file = 'DV.json'):

	DL_input_path = os.path.abspath(DL_input_path)
	EDP_input_path = os.path.abspath(EDP_input_path)

	# If the output dir was not specified, results are saved in the directory of
	# the input file.
	if output_path is None:
		output_path = ntpath.dirname(DL_input_path)

	# delete output files from previous runs
	files = os.listdir(output_path)
	for filename in files:
		if (filename[-3:] == 'csv') and (
			('DL_summary' in filename) or
			('DMG' in filename) or 
			('DV_' in filename) or
			('EDP' in filename)
			):
			try:
				os.remove(posixpath.join(output_path, filename))
			except:
				pass

	"""
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
	"""
	
	# If the event file is specified, we expect a multi-stripe analysis...
	if EVENT_input_path is not None:
		EVENT_input_path = os.path.abspath(EVENT_input_path)

		# Collect stripe and rate information for every event
		with open(EVENT_input_path, 'r') as f:
			event_list = json.load(f)['Events'][0]['Events']

		df_event = pd.DataFrame(columns=['name', 'stripe', 'rate'], 
								index=np.arange(len(event_list)))

		for evt_i, event in enumerate(event_list):
			df_event.iloc[evt_i] = [event['name'], event['stripe'], event['rate']]

		# Create a separate EDP input for each stripe
		EDP_input_full = pd.read_csv(EDP_input_path, sep='\s+', header=0, 
									 index_col=0)

		EDP_input_full.to_csv(EDP_input_path[:-4]+'_1.out', sep=' ')

		stripes = df_event['stripe'].unique()
		EDP_files = []
		for stripe in stripes:
			events = df_event[df_event['stripe']==stripe]['name'].values

			EDP_input = EDP_input_full[EDP_input_full['MultipleEvent'].isin(events)]

			EDP_files.append(EDP_input_path[:-4]+'_{}.out'.format(stripe))

			EDP_input.to_csv(EDP_files[-1], sep=' ')
	else:
		stripes = [1]
		EDP_files = [EDP_input_path]	
	
	# read the type of assessment from the DL input file
	with open(DL_input_path, 'r') as f:
		DL_input = json.load(f)

	# check if the DL input file has information about the loss model
	if 'LossModel' in DL_input:
		pass
	else:
		# if the loss model is not defined, give a warning
		print('WARNING No loss model defined in the BIM file. Trying to auto-populate.')

		# and try to auto-populate the loss model using the BIM information
		DL_input, DL_input_path = auto_populate(DL_input_path)

	DL_method = DL_input['LossModel']['DLMethod']

	# run the analysis and save results separately for each stripe
	#print(stripes, EDP_files)

	for s_i, stripe in enumerate(stripes):

		stripe_str = '' if len(stripes) == 1 else str(stripe)+'_'

		if DL_method == 'FEMA P58':
			A = FEMA_P58_Assessment()
		elif DL_method == 'HAZUS MH':
			A = HAZUS_Assessment()

		A.read_inputs(DL_input_path, EDP_files[s_i], verbose=False)

		A.define_random_variables()

		A.define_loss_model()

		A.calculate_damage()

		A.calculate_losses()

		A.aggregate_results()

		try:
		#if True:
			write_SimCenter_DL_output(
				posixpath.join(output_path, 
				'{}DL_summary.csv'.format(stripe_str)), A._SUMMARY, 
				index_name='#Num', collapse_columns=True)

			write_SimCenter_DL_output(
				posixpath.join(output_path, 
				'{}DL_summary_stats.csv'.format(stripe_str)), A._SUMMARY, 
				index_name='attribute', collapse_columns=True,  stats_only=True)

			EDPs = sorted(A._EDP_dict.keys())
			write_SimCenter_DL_output(
				posixpath.join(output_path, 
				'{}EDP.csv'.format(stripe_str)), A._EDP_dict[EDPs[0]]._RV.samples, 
				index_name='#Num', collapse_columns=False)
			
			DMG_mod = replace_FG_IDs_with_FG_names(A, A._DMG)
			write_SimCenter_DL_output(
				posixpath.join(output_path, 
				'{}DMG.csv'.format(stripe_str)), DMG_mod,
				index_name='#Num', collapse_columns=False)

			# create the DM.json file
			if DL_method == 'HAZUS MH':
				write_DM_output(posixpath.join(output_path, stripe_str+DM_file), 
					DMG_mod)

			write_SimCenter_DL_output(
				posixpath.join(output_path, 
				'{}DMG_agg.csv'.format(stripe_str)), 
				DMG_mod.T.groupby(level=0).aggregate(np.sum).T,
				index_name='#Num', collapse_columns=False)

			DV_mods, DV_names = [], []
			for key in A._DV_dict.keys():
				if key != 'injuries':
					DV_mods.append(replace_FG_IDs_with_FG_names(A, A._DV_dict[key]))
					DV_names.append('{}DV_{}'.format(stripe_str, key))
				else:
					for i in range(2 if DL_method == 'FEMA P58' else 4):
						DV_mods.append(replace_FG_IDs_with_FG_names(A, A._DV_dict[key][i]))
						DV_names.append('{}DV_{}_{}'.format(stripe_str, key, i))

			for DV_mod, DV_name in zip(DV_mods, DV_names):
				write_SimCenter_DL_output(
				posixpath.join(output_path, DV_name+'.csv'), DV_mod, 
				index_name='#Num', collapse_columns=False)

				if DL_method == 'HAZUS MH':
					write_DV_output(posixpath.join(output_path, stripe_str+DV_file), 
						DV_mod, DV_name)

				write_SimCenter_DL_output(
				posixpath.join(output_path, DV_name+'_agg.csv'), 
				DV_mod.T.groupby(level=0).aggregate(np.sum).T,
				index_name='#Num', collapse_columns=False)

		except:
			print("ERROR when trying to create DL output files.")

	return 0

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--filenameDL')
	parser.add_argument('--filenameEDP')
	parser.add_argument('--filenameEVENT', default = None)
	parser.add_argument('--filenameDM', default = 'DM.json')
	parser.add_argument('--filenameDV', default = 'DV.json')
	parser.add_argument('--dirnameOutput')
	args = parser.parse_args()

	#print(args.dirnameOutput)
	sys.exit(run_pelicun(
		args.filenameDL, args.filenameEDP, args.filenameDL,
		args.dirnameOutput, args.filenameDM, args.filenameDV))