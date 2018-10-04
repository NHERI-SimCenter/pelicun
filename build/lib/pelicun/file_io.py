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
This module has classes and methods that handle file input and output.

.. rubric:: Contents

.. autosummary::

    read_SimCenter_DL_input
    read_SimCenter_EDP_input
    read_P58_population_distribution
    read_P58_component_data
    write_SimCenter_DL_output

"""

import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
from distutils.util import strtobool
import pprint

from .base import *

pp = pprint.PrettyPrinter(indent=4)

def _classify(conversion_dict, source_class):
    """
    Converts a class in the source framework to a class in the target one. 

    Parameters
    ----------
    class_dict: dictionary
        Explain...
    source_class: string
        Explain...

    Returns
    -------
    target_class: string
        Explain...

    """
    for cls in conversion_dict.keys():
        if cls != '__type__':
            if (source_class == cls or
                source_class in conversion_dict[cls]):
                return cls

    print('ERROR: Could not convert',
          '{} using {} conversion.'.format(source_class,
                                           conversion_dict['__type__']))

    return

def read_SimCenter_DL_input(input_path, verbose=False):
    
    with open(input_path, 'r') as f:
        jd = json.load(f)

    # get the list of random variables
    randoms = dict((rv['name'], rv) for rv in jd['randomVariables'])

    # get the data required for DL
    data = dict([(label, dict()) for label in [
        'general', 'units', 'components', 'collapse_modes',
        'decision_variables', 'dependencies'
    ]])

    # general information
    GI = jd['GeneralInformation']
    for target_att, source_att, f_conv in [
        ['plan_area', 'planArea', float],
        ['stories', 'stories', int],
        ['building_type', 'type', str],
        ['story_height', 'height', float],
        ['year_built', 'year', int],
    ]:
        data['general'].update({target_att: f_conv(GI[source_att])})

    # units
    [data['units'].update({key: value}) for key, value in GI['units'].items()]

    LM = jd['LossModel']

    # components
    for comp in LM['Components']:
        comp_data = {
            'quantities'  : 
                np.array([float(qnt) for qnt in comp['quantity'].split(',')]),
            'csg_weights' : 
                np.array([float(wgt) for wgt in comp['weights'].split(',')]),
            'dirs'        : 
                np.array([int(dir_) for dir_ in comp['directions'].split(',')]),
            'kind'        : 'structural' if comp['structural'] else 'non-structural',
            'distribution': comp['distribution'],
            'cov'         : float(comp['cov']),
            'unit'        : [float(comp['unit_size']), comp['unit_type']],
        }
        comp_data.update({
            'locations':(np.where(comp_data['quantities'] > 0.)[0]+1)
        })
        # remove the zeros from the quantities
        nonzero = comp_data['quantities'] > 0.
        comp_data['quantities'] = comp_data['quantities'][nonzero]
        if comp_data['quantities'].shape == ():
            comp_data['quantities'] = np.array([comp_data['quantities']])
        data['components'].update({comp['ID']: comp_data})

    # collapse modes
    for coll_mode in LM['CollapseModes']:
        cm_data = {
            'w'            : float(coll_mode['w']),
            'injuries'     : [float(inj) for inj in
                              coll_mode['injuries'].split(',')],
            'affected_area': [float(cfar) for cfar in
                              coll_mode['affected_area'].split(',')],
        }
        data['collapse_modes'].update({coll_mode['name']: cm_data})

    def float_or_None(text):
        return float(text) if text != '' else None

    # other general info
    data['general'].update({
        'collapse_limits'       :
            dict([(key, float_or_None(value)) for key, value in
                  LM['BuildingDamage']['CollapseLimits'].items()]),

        'irrepairable_res_drift':
            dict([(key, float_or_None(value)) for key, value in
                  LM['BuildingDamage']['IrrepairableResidualDrift'].items()]),

        'detection_limits'      :
            dict([(key, float_or_None(value)) for key, value in
                  LM['BuildingResponse']['DetectionLimits'].items()]),

        'yield_drift'           : float_or_None(
            LM['BuildingResponse']['YieldDriftRatio']),

        'added_uncertainty'     : {
            'beta_gm': float_or_None(
                LM['UncertaintyQuantification']['AdditionalUncertainty'][
                    'GroundMotion']),
            'beta_m' : float_or_None(
                LM['UncertaintyQuantification']['AdditionalUncertainty'][
                    'Modeling']),
        },

        'realizations'          : int(LM['UncertaintyQuantification'][
            'Realizations']),
        
        'replacement_cost'      : float_or_None(
            LM['BuildingDamage']['ReplacementCost']),
        'replacement_time'      : float_or_None(
            LM['BuildingDamage']['ReplacementTime']),
        
        'occupancy_type'        : LM['Inhabitants']['OccupancyType'],
        'population'            : [float(pop) for pop in
                                   LM['Inhabitants']['PeakPopulation'].split(',')], 
    })

    # decision variables of interest
    DV = LM['DecisionVariables']
    for target_att, source_att in [
        ['injuries', 'Injuries'],
        ['rec_cost', 'ReconstructionCost'],
        ['rec_time', 'ReconstructionTime'],
        ['red_tag', 'RedTag'],
    ]:
        data['decision_variables'].update({target_att: bool(DV[source_att])})

    # dependencies
    dependency_to_acronym = {
        'btw. Fragility Groups'  : 'FG',
        'btw. Performance Groups': 'PG',
        'btw. Floors'            : 'LOC',
        'btw. Directions'        : 'DIR',
        'btw. Damage States'     : 'DS',
        'Independent'            : 'IND',
        'per ATC recommendation' : 'ATC,'
    }
    DEP = LM['LossModelDependencies']
    for target_att, source_att in [
        ['quantities', 'Quantities'],
        ['fragilities', 'Fragilities'],
        ['injuries', 'Injuries'],
        ['rec_costs', 'ReconstructionCosts'],
        ['rec_times', 'ReconstructionTimes'],
        ['red_tags', 'RedTagProbabilities'],
    ]:
        data['dependencies'].update({
            target_att:dependency_to_acronym[DEP[source_att]]})
        
    data['dependencies'].update({
        'cost_and_time': bool(DEP['CostAndTime']),
        'injury_lvls'  : bool(DEP['Injuries'])
    })

    if verbose: pp.pprint(data)
    
    return data

def read_SimCenter_EDP_input(input_path, verbose=False):
    
    # initialize the data container
    data = {}

    # read the dakota table output
    EDP_raw = pd.read_csv(input_path, sep='\s+', header=0,
                          index_col='%eval_id')
    EDP_raw.index = EDP_raw.index - 1

    # store the EDP data
    for column in EDP_raw.columns:
        for kind in ['PFA', 'PID']:
            if kind in column:

                if kind not in data.keys():
                    data.update({kind: []})

                info = column.split('-')
                data[kind].append(dict(
                    raw_data=EDP_raw[column].values,
                    location=info[2],
                    direction=info[3],
                    scenario_id=info[0]
                ))

    if verbose: pp.pprint(data)

    return data

def read_P58_population_distribution(path_POP, occupancy, verbose=False):
    with open(path_POP, 'r') as f:
        jd = json.load(f)

    data = jd[occupancy]

    # convert peak population to persons/m2
    data['peak'] = data['peak'] / (1000. * ft2)

    if verbose:
        pp.pprint(data)

    return data


def read_P58_component_data(path_CMP, comp_info, verbose=False):
    def parse_DS_xml(DS_xml):

        CFG = DS_xml.find('ConsequenceGroup')
        CFG_C = CFG.find('CostConsequence')
        CFG_T = CFG.find('TimeConsequence')

        if DS_xml.find('Percent') is not None:
            weight = float(DS_xml.find('Percent').text)
        else:
            weight = 1.0

        data = dict(
            description=DS_xml.find('Description').text,
            weight=weight,
            repair_cost=dict(
                median_max=float(CFG_C.find('MaxAmount').text),
                median_min=float(CFG_C.find('MinAmount').text),
                quantity_lower=float(CFG_C.find('LowerQuantity').text),
                quantity_upper=float(CFG_C.find('UpperQuantity').text),
                distribution_kind=curve_type[CFG_C.find('CurveType').text],
                cov=float(CFG_C.find('Uncertainty').text),
            ),
            repair_time=dict(
                median_max=float(CFG_T.find('MaxAmount').text),
                median_min=float(CFG_T.find('MinAmount').text),
                quantity_lower=float(CFG_T.find('LowerQuantity').text),
                quantity_upper=float(CFG_T.find('UpperQuantity').text),
                distribution_kind=curve_type[CFG_T.find('CurveType').text],
                cov=float(CFG_T.find('Uncertainty').text),
            ),
            red_tag=dict(
                theta=float(CFG.find('RedTagMedian').text),
                cov=float(CFG.find('RedTagBeta').text),
            ),
            injuries=dict(
                theta=[float(CFG.find('AffectedInjuryRate').text),
                       float(CFG.find('AffectedDeathRate').text)],
                cov=[float(CFG.find('AffectedInjuryRateBeta').text),
                     float(CFG.find('AffectedDeathRateBeta').text)]
            )
        )

        # if there is a chance of injuries, load the affected floor area
        if data['injuries']['theta'][0] > 0.:
            CFG_A = CFG.find('AffectedFloorArea').find('Area')
            unit = CFG_A.find('Unit').text
            if unit == 'Square Foot':
                affected_area = float(CFG_A.find('Value').text) * SF
            else:
                print('WARNING: unknown unit for affected floor area: ',
                      '{}'.format(unit))
                affected_area = 0.
            data.update({'affected_area': affected_area})
        else:
            data.update({'affected_area': 0.})

        # convert the units to standard ones
        data['repair_cost']['quantity_lower'] *= c_data['unit']
        data['repair_cost']['quantity_upper'] *= c_data['unit']
        data['repair_time']['quantity_lower'] *= c_data['unit']
        data['repair_time']['quantity_upper'] *= c_data['unit']

        data['repair_cost']['median_min'] /= c_data['unit']
        data['repair_cost']['median_max'] /= c_data['unit']
        data['repair_time']['median_min'] /= c_data['unit']
        data['repair_time']['median_max'] /= c_data['unit']

        return data

    data = dict([(c_id, dict([(key, None) for key in [
        'ID',
        'name',
        'description',
        'kind',
        'demand_type',
        'directional',
        'correlation',
        'offset',
        'incomplete',
        'locations',
        'quantities',
        'csg_weights',
        'dir_weights',
        'directions',
        'distribution_kind',
        'cov',
        'unit',
        'DSG_set',
    ]])) for c_id in comp_info.keys()])

    # for each component
    for c_id, c_data in data.items():

        # first, get the parameters from the BIM component info
        ci_data = comp_info[c_id]
        c_data['kind'] = ci_data['kind']
        c_data['unit'] = ci_data['unit'][0] * globals()[ci_data['unit'][1]]
        c_data['quantities'] = np.asarray(ci_data['quantities']) * c_data[
            'unit']
        c_data['distribution_kind'] = ci_data['distribution']
        c_data['csg_weights'] = np.asarray(ci_data['csg_weights'])
        c_data['directions'] = np.asarray(ci_data['dirs'], dtype=np.int)
        c_data['locations'] = np.asarray(ci_data['locations'], dtype=np.int)
        c_data['cov'] = ci_data['cov']

        # calculate the quantity weights in each direction
        dirs = c_data['directions']
        u_dirs = np.unique(dirs)
        weights = c_data['csg_weights']
        c_data['dir_weights'] = [sum(weights[np.where(dirs == d_i)]) 
                                 for d_i in u_dirs]

        # parse the xml file
        # TODO: replace the xml with a json
        tree = ET.parse(path_CMP + c_id + '.xml')
        root = tree.getroot()

        c_data['ID'] = root.find('ID').text
        c_data['name'] = root.find('Name').text
        c_data['description'] = root.find('Description').text
        c_data['offset'] = int(strtobool(
            root.find('UseEDPValueOfFloorAbove').text))
        c_data['correlation'] = strtobool(root.find('Correlation').text)
        c_data['directional'] = strtobool(root.find('Directional').text)
        c_data['incomplete'] = strtobool(root.find('Incomplete').text)

        if c_data['incomplete']:
            print('WARNING: fragility information for',
                  '{} is incomplete'.format(c_id))

        EDP = root.find('EDPType')
        EDP_type = EDP.find('TypeName').text
        demand_factor = 1.0
        if EDP_type == 'Story Drift Ratio':
            demand_type = 'PID'
        elif EDP_type == 'Acceleration':
            demand_type = 'PFA'
            demand_factor = g
        elif EDP_type == 'Effective Drift':
            demand_type = 'PID'
            print('WARNING: Unable to handle {} EDP type,',
                  'using Story Drift Ratio instead'.format(EDP_type))
        else:
            demand_type = None
            print('WARNING: Unknown EDP type: {}'.format(EDP_type))
        c_data['demand_type'] = demand_type

        # dictionary to convert xml to internal
        curve_type = {'LogNormal': 'lognormal',
                      'Normal'   : 'normal'}
        DS_set_kind = {'MutuallyExclusive' : 'mutually exclusive',
                       'mutually exclusive': 'mutually exclusive',
                       'simultaneous'      : 'simultaneous',
                       'Simultaneous'      : 'simultaneous',
                       'single'            : 'single'}

        # load the damage state group information        
        c_data['DSG_set'] = dict()
        DSGs = root.find('DamageStates').findall('DamageState')
        for DSG_i, DSG_xml in enumerate(DSGs):
            DSG_data = dict(
                theta=float(DSG_xml.find('Median').text) * demand_factor,
                sig=float(DSG_xml.find('Beta').text),
                description=DSG_xml.find('Description').text,
            )

            if DSG_xml.find('DamageStates') is not None:
                # the group contains multiple damage states                
                DS_xml = DSG_xml.find('DamageStates')
                DSG_data.update({'DS_set_kind':
                                     DS_set_kind[
                                         DS_xml.find('DSGroupType').text]})
                DSG_data.update(dict(DS_set=dict()))

                for DS_i, DS_xml in enumerate(
                    DS_xml.findall('DamageState')):
                    DS = parse_DS_xml(DS_xml)
                    DSG_data['DS_set'].update({'DS-' + str(DS_i + 1): DS})

            else:
                # the group contains only one damage state
                DSG_data.update({'DS_set_kind': 'single'})
                DSG_data.update(dict(DS_set=dict()))

                DS = parse_DS_xml(DSG_xml)
                DSG_data['DS_set'].update({'DS-1': DS})

            c_data['DSG_set'].update({'DSG-' + str(DSG_i + 1): DSG_data})

    if verbose:
        for c_id, c_data in data.items():
            print(c_id)
            pp.pprint(c_data)

    return data

def write_SimCenter_DL_output(output_path, output_df):
    
    output_df.to_csv(output_path)