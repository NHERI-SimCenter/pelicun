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
    read_population_distribution
    read_component_DL_data
    write_SimCenter_DL_output

"""

import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
from distutils.util import strtobool
import pprint
from copy import deepcopy

import warnings

from .base import *

pp = pprint.PrettyPrinter(indent=4)

def read_SimCenter_DL_input(input_path, verbose=False):
    """
    Read the damage and loss input information from a json file.
    
    The SimCenter in the function name refers to having specific fields 
    available in the file. Such a file is automatically prepared by the 
    SimCenter PBE Application. The accepted input fields are explained in 
    detail in the Input section of the documentation.
    
    Parameters
    ----------
    input_path: string
        Location of the DL input json file.
    verbose: boolean
        If True, the function echoes the information read from the file. This
        can be useful to ensure that the information in the file is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with all the damage and loss data.

    """
    
    with open(input_path, 'r') as f:
        jd = json.load(f)

    # get the data required for DL
    data = dict([(label, dict()) for label in [
        'general', 'units', 'components', 'collapse_modes',
        'decision_variables', 'dependencies'
    ]])

    LM = jd['LossModel']

    # decision variables of interest
    if 'DecisionVariables' in LM.keys():        
        DV = LM['DecisionVariables']
        for target_att, source_att in [
            ['injuries', 'Injuries'],
            ['rec_cost', 'ReconstructionCost'],
            ['rec_time', 'ReconstructionTime'],
            ['red_tag', 'RedTag'],
        ]:
            data['decision_variables'].update({target_att: bool(DV[source_att])})
    else:
        warnings.warn(UserWarning(
            "No decision variables specified in the input file. Assuming that "
            "all decision variables shall be calculated."))
        data['decision_variables'].update({
            'injuries': True,
            'rec_cost': True,
            'rec_time': True,
            'red_tag': True
        })
        
    DV = data['decision_variables']

    # general information
    if 'GeneralInformation' in jd:
        GI = jd['GeneralInformation']
    else:
        GI = None

    # units
    if (GI is not None) and ('units' in GI.keys()):
        for key, value in GI['units'].items():
            if value == 'in':
                value = 'inch'
            if value in globals().keys():
                data['units'].update({key: globals()[value]})
            else:
                warnings.warn(UserWarning(
                    "Unknown {} unit: {}".format(key, value)
                ))
        if 'length' in data['units'].keys():
            data['units'].update({
                'area': data['units']['length']**2.,
                'volume': data['units']['length']**3.                
            })
            if 'acceleration' not in data['units'].keys():
                data['units'].update({
                    'acceleration': 1.0 }) 
            #            data['units']['length'] / data['units']['time'] ** 2.})
    else:
        warnings.warn(UserWarning(
            "No units were specified in the input file. Standard units are "
            "assumed."))
        data['units'].update({
            'force': 1.,
            'length': 1.,
            'area': 1.,
            'volume': 1.,
            'acceleration': 1.,
        })

    # other attributes
    for target_att, source_att, f_conv, unit_kind, dv_req in [
        ['plan_area', 'planArea', float, 'area', 'injuries'],
        ['stories', 'stories', int, '', ''],
        # The following lines are commented out for now, because we do not use
        # these pieces of data anyway.
        #['building_type', 'type', str, ''],
        #['height', 'height', float, 'length'],
        #['year_built', 'year', int, ''],
    ]:
        if (GI is not None) and (source_att in GI.keys()):
            if unit_kind is not '':
                f_unit = data['units'][unit_kind]
            else:
                f_unit = 1
            att_value = f_conv(GI[source_att]) * f_unit
            data['general'].update({target_att: att_value})
        else:
            if (dv_req!='') and DV[dv_req]:
                warnings.warn(UserWarning(
                    "{} is not in the DL input file.".format(source_att)))
        
    # components
    # Having components defined is not necessary, but if a component is defined
    # then all of its attributes need to be specified.
    if 'Components' in LM.keys():
        for comp in LM['Components']:
            comp_data = {
                'quantities'  : 
                    np.array([float(qnt) for qnt in comp['quantity'].split(',')]),
                'csg_weights' : 
                    [float(wgt) for wgt in comp['weights'].split(',')],
                'dirs'        : 
                    [int(dir_) for dir_ in comp['directions'].split(',')],
                'kind'        : 'structural' if comp['structural'] else 'non-structural',
                'distribution': comp['distribution'],
                'cov'         : float(comp['cov']),
                'unit'        : [float(comp['unit_size']), comp['unit_type']],
            }

            # sort the dirs and their weights to have better structured matrices 
            # later
            dir_order = np.argsort(comp_data['dirs'])
            comp_data['dirs'] = [comp_data['dirs'][d_i] for d_i in dir_order]
            comp_data['csg_weights'] = [comp_data['csg_weights'][d_i] for d_i in dir_order]
            
            # get the location(s) of components based on non-zero quantities
            comp_data.update({
                'locations':(np.where(comp_data['quantities'] > 0.)[0]+1).tolist()
            })
            # remove the zeros from the quantities
            nonzero = comp_data['quantities'] > 0.
            comp_data['quantities'] = comp_data['quantities'][nonzero].tolist()
                
            # scale the quantities according to the specified unit
            unit_kind = comp_data['unit'][1]
            if unit_kind not in globals().keys():
                raise ValueError(
                    "Unknown unit for component {}: {}".format(
                        comp['ID'], unit_kind))
            
            # store the component data
            data['components'].update({comp['ID']: comp_data})
    else:
        warnings.warn(UserWarning(
            "No components were defined in the input file."))

    # collapse modes
    # Having collapse modes defined is not necessary, but if a collapse mode is
    # defined, then all of its attributes need to be specified.
    if 'CollapseModes' in LM.keys():
        for coll_mode in LM['CollapseModes']:
            cm_data = {
                'w'            : float(coll_mode['w']),
                'injuries'     : [float(inj) for inj in
                                  coll_mode['injuries'].split(',')],
                'affected_area': [float(cfar) for cfar in
                                  coll_mode['affected_area'].split(',')],
            }
            data['collapse_modes'].update({coll_mode['name']: cm_data})
    else:
        warnings.warn(UserWarning(
            "No collapse modes were defined in the input file."))

    # the number of realizations has to be specified in the file    
    if (('UncertaintyQuantification' in LM) and 
        ('Realizations' in LM['UncertaintyQuantification'])):
            data['general'].update({
                'realizations': int(LM['UncertaintyQuantification'][
                                        'Realizations'])})
    else:
        raise ValueError(
            "Number of realizations is not specified in the input file.")

    # this is a convenience function for converting strings to float or None
    def float_or_None(text):
        return float(text) if text != '' else None
    
    EDP_units = dict(
        # PID is not here because it is unitless
        PFA = 'acceleration'
    )

    # other general info
    if 'BuildingDamage' in LM.keys():
        if 'CollapseLimits' in LM['BuildingDamage'].keys():
            data['general'].update({
                'collapse_limits':
                    dict([(key, float_or_None(value)) for key, value in
                          LM['BuildingDamage']['CollapseLimits'].items()])})            
            # scale the limits by the units
            DGCL = data['general']['collapse_limits']
            for EDP_kind, value in DGCL.items():
                if (EDP_kind in EDP_units.keys()) and (value is not None):
                    f_EDP = data['units'][EDP_units[EDP_kind]]
                    DGCL[EDP_kind] = DGCL[EDP_kind] * f_EDP
        else:
            warnings.warn(UserWarning(
                "Collapse EDP limits were not defined in the input file. "
                "No EDP limits are assumed."))        
        # make sure that PID and PFA collapse limits are identified
        if 'collapse_limits' not in data['general'].keys():
            data['general'].update({'collapse_limits':{}})
        for key in ['PID', 'PFA']:
            if key not in data['general']['collapse_limits'].keys():
                data['general']['collapse_limits'].update({key: None})
            
            
        if 'IrrepairableResidualDrift' in LM['BuildingDamage'].keys():
            data['general'].update({
                'irrepairable_res_drift':
                    dict([(key, float_or_None(value)) for key, value in
                          LM['BuildingDamage'][
                              'IrrepairableResidualDrift'].items()])})
        elif DV['rec_cost'] or DV['rec_time']:
            warnings.warn(UserWarning(
                "Residual drift limits corresponding to irrepairable "
                "damage were not defined in the input file. We assume that "
                "damage is repairable regardless of the residual drift."))
            
        if 'ReplacementCost' in LM['BuildingDamage'].keys():
            data['general'].update({
                'replacement_cost': float_or_None(
                    LM['BuildingDamage']['ReplacementCost'])})
        elif DV['rec_cost']:
            warnings.warn(UserWarning(
                "Building replacement cost was not defined in the input file."))
    
        if 'ReplacementTime' in LM['BuildingDamage'].keys():
            data['general'].update({
                'replacement_time': float_or_None(
                    LM['BuildingDamage']['ReplacementTime'])})
        elif DV['rec_time']:
            warnings.warn(UserWarning(
                "Building replacement time was not defined in the input file."))
    else:
        warnings.warn(UserWarning(
            "Building damage characteristics were not defined in the input "
            "file"))
    
    if 'BuildingResponse' in LM.keys():
        if 'DetectionLimits' in LM['BuildingResponse'].keys():
            data['general'].update({
                'detection_limits':
                    dict([(key, float_or_None(value)) for key, value in
                          LM['BuildingResponse']['DetectionLimits'].items()])})
            DGDL = data['general']['detection_limits']
            # scale the limits by the units            
            for EDP_kind, value in DGDL.items():
                if (EDP_kind in EDP_units.keys()) and (value is not None):
                    f_EDP = data['units'][EDP_units[EDP_kind]]
                    DGDL[EDP_kind] = DGDL[EDP_kind] * f_EDP
        else:
            warnings.warn(UserWarning(
                "EDP detection limits were not defined in the input file. "
                "Assuming no detection limits."))
        if 'detection_limits' not in data['general'].keys():
            data['general'].update({'detection_limits':{}})
        for key in ['PID', 'PFA']:
            if key not in data['general']['detection_limits'].keys():
                data['general']['detection_limits'].update({key: None})
        
        if 'YieldDriftRatio' in LM['BuildingResponse'].keys():
            data['general'].update({
                'yield_drift': float_or_None(
                    LM['BuildingResponse']['YieldDriftRatio'])})
        elif DV['rec_cost'] or DV['rec_time']:
            warnings.warn(UserWarning(
                "Yield drift ratio was not defined in the input file. "
                "Assuming a yield drift ratio of 0.01 radian."))
            data['general'].update({'yield_drift': 0.01})
            
    else:
        warnings.warn(UserWarning(
            "Building response characteristics were not defined in the input "
            "file. Assuming no detection limits and a yield drift ratio of "
            "0.01 radian."))
        data['general'].update({
            'detection_limits': dict([(key, None) for key in ['PFA', 'PID']]),
            'yield_drift': 0.01
        })
        
        
    if 'AdditionalUncertainty' in LM['UncertaintyQuantification'].keys():
        data['general'].update({
            'added_uncertainty': {
                'beta_gm': float_or_None(
                    LM['UncertaintyQuantification']['AdditionalUncertainty'][
                        'GroundMotion']),
                'beta_m' : float_or_None(
                    LM['UncertaintyQuantification']['AdditionalUncertainty'][
                        'Modeling'])}})
    else:
        warnings.warn(UserWarning(
            "No additional uncertainties were defined in the input file. "
            "Assuming that EDPs already include all ground motion and modeling "
            "uncertainty."))
        data['general'].update({
            'added_uncertainty': {
                'beta_gm': 0.0001,
                'beta_m': 0.0001
            }
        })
    
    if 'Inhabitants' in LM.keys():
        if 'OccupancyType' in LM['Inhabitants'].keys():
            data['general'].update({
                'occupancy_type': LM['Inhabitants']['OccupancyType']})
        elif DV['injuries']:
            warnings.warn(UserWarning(
                "Occupancy type was not defined in the input file."))
    
        if 'PeakPopulation' in LM['Inhabitants'].keys():
            peak_pop = [float(pop) for pop in
                               LM['Inhabitants']['PeakPopulation'].split(',')]
            
            # If the number of stories is specified and the population list
            # does not provide values for every story...
            # If only one value is provided, then it is assumed at every story.
            # Otherwise, the values are assumed to correspond to the bottom
            # stories and the upper ones are filled with zeros. 
            # A warning message is displayed in this case.
            if 'stories' in data['general'].keys():
                stories = data['general']['stories']
                pop_in = len(peak_pop)
                for s in range(pop_in, stories):
                    if pop_in == 1:
                        peak_pop.append(peak_pop[0])
                    else:
                        peak_pop.append(0)
                
                if pop_in > 1 and pop_in != stories:
                    warnings.warn(UserWarning(
                        "Peak population was specified to some, but not all "
                        "stories. The remaining stories are assumed to have "
                        "zero population."
                    ))
                
            data['general'].update({'population': peak_pop})
        elif DV['injuries']:
            warnings.warn(UserWarning(
                "Peak population was not defined in the input file."))
    elif DV['injuries']:
        warnings.warn(UserWarning(
            "Information about inhabitants was not defined in the input file."))

    # dependencies
    # We assume 'Independent' for all unspecified fields except for the 
    # fragilities where 'per ATC recommendation' is the default setting.
    dependency_to_acronym = {
        'btw. Fragility Groups'  : 'FG',
        'btw. Performance Groups': 'PG',
        'btw. Floors'            : 'LOC',
        'btw. Directions'        : 'DIR',
        'btw. Damage States'     : 'DS',
        'Independent'            : 'IND',
        'per ATC recommendation' : 'ATC',
    }
    
    if 'LossModelDependencies' in LM:
        DEP = LM['LossModelDependencies']
    else:
        DEP = None
        
    for target_att, source_att, dv_req in [
        ['quantities', 'Quantities', ''],
        ['fragilities', 'Fragilities', ''],
        ['injuries', 'Injuries', 'injuries'],
        ['rec_costs', 'ReconstructionCosts', 'rec_cost'],
        ['rec_times', 'ReconstructionTimes', 'rec_time'],
        ['red_tags', 'RedTagProbabilities', 'red_tag'],
    ]:
        if (DEP is not None) and (source_att in DEP.keys()):
            data['dependencies'].update({
                target_att:dependency_to_acronym[DEP[source_att]]})
        elif dv_req == '' or DV[dv_req]:
            if target_att is not 'fragilities':
                data['dependencies'].update({target_att: 'IND'})
            else:
                data['dependencies'].update({target_att: 'ATC'})
                        
            warnings.warn(UserWarning(
                "Correlation between {} was not ".format(source_att)+
                "defined in the input file. Using default values."))
                
    if (DEP is not None) and ('CostAndTime' in DEP.keys()):
        data['dependencies'].update({'cost_and_time': bool(DEP['CostAndTime'])})
    elif DV['rec_cost'] or DV['rec_time']:
        data['dependencies'].update({'cost_and_time': False})
        warnings.warn(UserWarning(
            "Correlation between reconstruction cost and time was not defined "
            "in the input file. Using default values."))

    if (DEP is not None) and ('InjurySeverities' in DEP.keys()):
        data['dependencies'].update({'injury_lvls': 
                                         bool(DEP['InjurySeverities'])})
    elif DV['injuries']:
        data['dependencies'].update({'injury_lvls': False})
        warnings.warn(UserWarning(
            "Correlation between injury levels was not defined in the input "
            "file. Using default values."))

    if verbose: pp.pprint(data)
    
    return data

def read_SimCenter_EDP_input(input_path, EDP_kinds=('PID','PFA'), 
                             units = dict(PID=1., PFA=1.),
                             verbose=False):
    """
    Read the EDP input information from a text file with a tabular structure.
    
    The SimCenter in the function name refers to having specific columns 
    available in the file. Currently, the expected formatting follows the
    output formatting of Dakota that is applied for the dakotaTab.out. When
    using pelicun with the PBE Application, such a dakotaTab.out is 
    automatically generated. The Input section of the documentation provides
    more information about the expected formatting of the EDP input file.
    
    Parameters
    ----------
    input_path: string
        Location of the EDP input file.
    EDP_kinds: tuple of strings, default: ('PID', 'PFA')
        Collection of the kinds of EDPs in the input file. The default pair of
        'PID' and 'PFA' can be replaced or extended by any other EDPs. 
    units: dict, default: {'PID':1., 'PFA':1}
        Defines the unit conversion that shall be applied to the EDP values.
    verbose: boolean
        If True, the function echoes the information read from the file. This
        can be useful to ensure that the information in the file is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with all the EDP data.
    """
    
    # initialize the data container
    data = {}

    # read the collection of EDP inputs...
    # the read_csv method in pandas is sufficiently versatile to handle the
    # tabular format of dakota
    EDP_raw = pd.read_csv(input_path, sep='\s+', header=0,
                          index_col=0)
    # set the index to be zero-based
    EDP_raw.index = EDP_raw.index - 1

    # search the header for EDP information
    for column in EDP_raw.columns:
        for kind in EDP_kinds:
            if kind in column:

                if kind not in data.keys():
                    data.update({kind: []})

                # extract info about the location, direction, and scenario
                info = column.split('-')
                
                # get the scale factor to perform unit conversion
                f_unit = units[kind]
                
                # store the data
                data[kind].append(dict(
                    raw_data=(EDP_raw[column].values * f_unit).tolist(),
                    location=info[2],
                    direction=info[3],
                    scenario_id=info[0]
                ))

    if verbose: pp.pprint(data)

    return data

def read_population_distribution(path_POP, occupancy, verbose=False):
    """
    Read the population distribution from an external json file.
    
    The population distribution is expected in a format used in FEMA P58, but
    the list of occupancy categories can be extended beyond those available
    in that document. The population distributions for the occupancy categories
    from FEMA P58 are provided with pelicun in the population.json in the 
    resources folder.
    
    Parameters
    ----------
    path_POP: string
        Location of the population distribution json file.
    occupancy: string
        Identifies the occupancy category. There must be a matching category in 
        the population distribution json file. 
    verbose: boolean
        If True, the function echoes the information read from the file. This
        can be useful to ensure that the information in the file is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with the population distribution data.
    """
    with open(path_POP, 'r') as f:
        jd = json.load(f)

    data = jd[occupancy]

    # convert peak population to persons/m2
    data['peak'] = data['peak'] / (1000. * ft2)

    if verbose: # pragma: no cover
        pp.pprint(data)

    return data


def read_component_DL_data(path_CMP, comp_info, verbose=False):
    """
    Read the damage and loss data for the components of the asset.

    DL data for each component is assumed to be stored in a JSON file following 
    the DL file format specified by SimCenter. The name of the file is the ID 
    (key) of the component in the comp_info dictionary. Besides the filename, 
    the comp_info dictionary is also used to get other pieces of data about the 
    component that is not available in the JSON files. Therefore, the following 
    attributes need to be provided in the comp_info: ['quantities', 
    'csg_weights', 'dirs', 'kind', 'distribution', 'cov', 'unit', 'locations'] 
    Further information about these attributes is available in the Input 
    section of the documentation.

    Parameters
    ----------
    path_CMP: string
        Location of the folder that contains the component data in JSON files.
    comp_info: dict
        Dictionary with additional information about the components.
    verbose: boolean
        If True, the function echoes the information read from the files. This
        can be useful to ensure that the information in the files is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with damage and loss data for each component.

    """

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
    s_cmp_keys = sorted(data.keys())
    for c_id in s_cmp_keys:
        c_data = data[c_id]

        # parse the json file
        with open(os.path.join(path_CMP, c_id + '.json'), 'r') as f:
            DL_data = json.load(f)

        DL_GI = DL_data['GeneralInformation']
        DL_EDP = DL_data['EDP']
        DL_DSG = DL_data['DSGroups']

        # First, check if the DL data is complete. Incomplete data can lead to 
        # all kinds of problems, so in such a case we display a warning and do 
        # not use the component. This can be relaxed later if someone creates a 
        # method to replace unknown values with reasonable estimates.
        c_data['incomplete'] = int(DL_GI['Incomplete'])
        if c_data['incomplete']:
            # show warning            
            del data[c_id]
            warnings.warn(UserWarning(
                'Fragility information for {} is incomplete. The component '
                'cannot be used for loss assessment.'.format(c_id)))
            continue

        # Get the parameters from the BIM component info
        ci_data = comp_info[c_id]
        c_data['kind'] = ci_data['kind']
        c_data['unit'] = ci_data['unit'][0] * globals()[ci_data['unit'][1]]
        c_data['quantities'] = (np.asarray(ci_data['quantities']) * c_data[
            'unit']).tolist()
        c_data['distribution_kind'] = ci_data['distribution']
        c_data['csg_weights'] = ci_data['csg_weights']
        c_data['directions'] = ci_data['dirs']
        c_data['locations'] = ci_data['locations']
        c_data['cov'] = ci_data['cov']

        # calculate the quantity weights in each direction
        dirs = np.asarray(c_data['directions'], dtype=np.int)
        u_dirs = np.unique(dirs)
        weights = np.asarray(c_data['csg_weights'])
        c_data['dir_weights'] = [sum(weights[np.where(dirs == d_i)])
                                 for d_i in u_dirs]

        c_data['ID'] = c_id
        c_data['name'] = DL_data['Name']
        c_data['description'] = DL_GI['Description']
        c_data['offset'] = DL_EDP['Offset']
        c_data['correlation'] = int(DL_data['Correlated'])
        c_data['directional'] = int(DL_data['Directional'])

        EDP_type = DL_EDP['Type']
        demand_factor = 1.0
        if EDP_type == 'Story Drift Ratio':
            demand_type = 'PID'
        elif EDP_type == 'Peak Floor Acceleration':
            demand_type = 'PFA'
            demand_factor = g
        elif EDP_type in [
            'Peak Floor Velocity',
            'Link Rotation Angle',
            'Link Beam Chord Rotation']:
            demand_type = None
            warnings.warn(UserWarning(
                'Component {} requires {} as EDP, which is not yet '
                'implemented.'.format(c_data['ID'], EDP_type)))
        else: # pragma: no cover
            demand_type = None
            warnings.warn(UserWarning(
                'Unexpected EDP type: {}'.format(EDP_type)))
        if demand_type is None:
            del data[c_id]
            continue
        c_data['demand_type'] = demand_type

        # dictionary to convert DL data to internal representation
        curve_type = {'LogNormal': 'lognormal',
                      'Normal'   : 'normal'}
        DS_set_kind = {'MutuallyExclusive' : 'mutually exclusive',
                       'mutually exclusive': 'mutually exclusive',
                       'simultaneous'      : 'simultaneous',
                       'Simultaneous'      : 'simultaneous',
                       'single'            : 'single',
                       'Single'            : 'single'}

        # load the damage state group information
        c_data['DSG_set'] = dict()
        for DSG_id, DSG_i in enumerate(DL_DSG):
            DSG_data = dict(
                theta=DSG_i['MedianEDP'] * demand_factor,
                sig=DSG_i['Beta'],
                DS_set_kind=DS_set_kind[DSG_i['DSGroupType']],
                # distribution_kind = curve_type[DSG_i['CurveType']],
                DS_set={}
            )

            for DS_id, DS_i in enumerate(DSG_i['DamageStates']):
                DS_data = {'description': DS_i['Description'],
                           'weight'     : DS_i['Weight']}

                DS_C = DS_i['Consequences']
                if 'ReconstructionCost' in DS_C.keys():
                    DS_CC = DS_C['ReconstructionCost']
                    DS_data.update({'repair_cost': {
                        'median_max'       : DS_CC['Amount'][0],
                        'median_min'       : DS_CC['Amount'][1],
                        'quantity_lower'   : DS_CC['Quantity'][0],
                        'quantity_upper'   : DS_CC['Quantity'][1],
                        'distribution_kind': curve_type[DS_CC['CurveType']],
                        'cov'              : DS_CC['Beta'],
                    }})

                    # convert the units to standard ones
                    DS_data['repair_cost']['quantity_lower'] *= c_data['unit']
                    DS_data['repair_cost']['quantity_upper'] *= c_data['unit']
                    DS_data['repair_cost']['median_min'] /= c_data['unit']
                    DS_data['repair_cost']['median_max'] /= c_data['unit']

                if 'ReconstructionTime' in DS_C.keys():
                    DS_CT = DS_C['ReconstructionTime']
                    DS_data.update({'repair_time': {
                        'median_max'       : DS_CT['Amount'][0],
                        'median_min'       : DS_CT['Amount'][1],
                        'quantity_lower'   : DS_CT['Quantity'][0],
                        'quantity_upper'   : DS_CT['Quantity'][1],
                        'distribution_kind': curve_type[DS_CT['CurveType']],
                        'cov'              : DS_CT['Beta'],
                    }})

                    # convert the units to standard ones
                    DS_data['repair_time']['quantity_lower'] *= c_data['unit']
                    DS_data['repair_time']['quantity_upper'] *= c_data['unit']
                    DS_data['repair_time']['median_min'] /= c_data['unit']
                    DS_data['repair_time']['median_max'] /= c_data['unit']

                if 'RedTag' in DS_C.keys():
                    DS_CR = DS_C['RedTag']
                    DS_data.update({'red_tag': {
                        'theta': DS_CR['Amount'],
                        # 'distribution_kind': curve_type[DS_CR['CurveType']],
                        'cov'  : DS_CR['Beta'],
                    }})

                if 'Injuries' in DS_C.keys():
                    DS_CI = DS_C['Injuries']
                    DS_data.update({'injuries': {
                        'theta': [I_i['Amount'] for I_i in DS_CI],
                        # 'distribution_kind': curve_type[DS_CR['CurveType']],
                        'cov'  : [I_i['Beta'] for I_i in DS_CI],
                    }})

                    # if there is a chance of injuries, load the affected floor area
                    affected_area, unit = DS_i['AffectedArea']
                    if unit == 'SF':
                        affected_area = affected_area * SF
                    else: # pragma: no cover
                        warnings.warn(UserWarning(
                            'Unknown unit for affected floor area: {}'.format(
                                unit)))
                        affected_area = 0.
                    DS_data.update({'affected_area': affected_area})

                    # convert the units to standard ones
                    DS_data['affected_area'] /= c_data['unit']

                DSG_data['DS_set'].update({'DS-' + str(DS_id + 1): DS_data})

            c_data['DSG_set'].update({'DSG-' + str(DSG_id + 1): DSG_data})

    if verbose: # pragma: no cover
        for c_id, c_data in data.items():
            print(c_id)
            pp.pprint(c_data)

    return data

def convert_P58_data_to_json(data_dir, target_dir):
    """
    Create JSON data files from publicly available P58 data.
    
    FEMA P58 damage and loss information is publicly available in an Excel 
    spreadsheet and also in a series of XML files as part of the PACT tool. 
    Those files are copied to the resources folder in the pelicun repo. Here
    we collect the available information on Fragility Groups from those files
    and save the damage and loss data in the common SimCenter JSON format.
    
    A large part of the Fragility Groups in FEMA P58 do not have complete
    damage and loss information available. These FGs are clearly marked with
    an incomplete flag in the JSON file and the 'Undefined' value highlights
    the missing pieces of information. 
    
    Parameters
    ----------
    data_dir: string
        Path to the folder with the FEMA P58 Excel file and a 'DL xml' 
        subfolder in it that contains the XML files.
    target_dir: string
        Path to the folder where the JSON files shall be saved.

    """
    
    convert_unit = {
        'Unit less': 'ea',
        'Radians'  : 'rad',
        'g'        : 'g',
        'meter/sec': 'mps'
    }

    convert_DSG_type = {
        'MutEx': 'MutuallyExclusive',
        'Simul': 'Simultaneous'
    }

    def decode_DS_Hierarchy(DSH):

        if 'Seq' == DSH[:3]:
            DSH = DSH[4:-1]

        DS_setup = []

        while len(DSH) > 0:
            if DSH[:2] == 'DS':
                DS_setup.append(DSH[:3])
                DSH = DSH[4:]
            elif DSH[:5] in ['MutEx', 'Simul']:
                closing_pos = DSH.find(')')
                subDSH = DSH[:closing_pos + 1]
                DSH = DSH[closing_pos + 2:]

                DS_setup.append([subDSH[:5]] + subDSH[6:-1].split(','))

        return DS_setup

    def parse_DS_xml(DS_xml):
        CFG = DS_xml.find('ConsequenceGroup')
        CFG_C = CFG.find('CostConsequence')
        CFG_T = CFG.find('TimeConsequence')

        repair_cost = dict(
            Amount=[float(CFG_C.find('MaxAmount').text),
                    float(CFG_C.find('MinAmount').text)],
            Quantity=[float(CFG_C.find('LowerQuantity').text),
                      float(CFG_C.find('UpperQuantity').text)],
            CurveType=CFG_C.find('CurveType').text,
            Beta=float(CFG_C.find('Uncertainty').text),
            Bounds=[0, 'None']
        )
        if repair_cost['Amount'] == [0.0, 0.0]:
            repair_cost['Amount'] = 'Undefined'

        repair_time = dict(
            Amount=[float(CFG_T.find('MaxAmount').text),
                    float(CFG_T.find('MinAmount').text)],
            Quantity=[float(CFG_T.find('LowerQuantity').text),
                      float(CFG_T.find('UpperQuantity').text)],
            CurveType=CFG_T.find('CurveType').text,
            Beta=float(CFG_T.find('Uncertainty').text),
            Bounds=[0, 'None']
        )
        if repair_time['Amount'] == [0.0, 0.0]:
            repair_time['Amount'] = 'Undefined'

        return repair_cost, repair_time

    def is_float(s):
        try:
            if type(s) == str and s[-1] == '%':
                s_f = float(s[:-1]) / 100.
            else:
                s_f = float(s)
            if np.isnan(s_f):
                return False
            else:
                return True
        except ValueError:
            return False

    src_df = pd.read_excel(
        os.path.join(data_dir, 'FEMAP58ED1_fragility_data.xlsx'))
    ID_list = src_df['NISTIR Classification']

    XML_list = [f for f in os.listdir(data_dir) if f.endswith('.xml')]

    incomplete_count = 0

    for filename in XML_list:

        comp_ID = filename[:-4]

        try:
        #if True:
            tree = ET.parse(os.path.join(data_dir, comp_ID + '.xml'))
            root = tree.getroot()

            # correct for the error in the numbering of RC beams
            if (comp_ID[:5] == 'B1051') and (
                comp_ID[-1] not in str([1, 2, 3, 4])):
                comp_ID = 'B1042' + comp_ID[5:]

            row = src_df.loc[np.where(ID_list == comp_ID)[0][0], :]

            json_output = {}
            incomplete = False

            json_output.update({'Name': row['Component Name']})

            QU = row['Fragility Unit of Measure']
            QU = QU.split(' ')
            if is_float(QU[1]):
                json_output.update({'QuantityUnit': [int(QU[1]), QU[0]]})
            else:
                json_output.update({'QuantityUnit': [0, 'Undefined']})
                incomplete = True

            json_output.update({'Directional': row['Directional?'] in ['YES']})
            json_output.update({'Correlated': row['Correlated?'] in ['YES']})

            json_output.update({
                'EDP': {
                    'Type'  : row['Demand Parameter (value):'],
                    'Unit'  : [1,
                               convert_unit[row['Demand Parameter (unit):']]],
                    'Offset': int(
                        row['Demand Location (use floor above? Yes/No)'] in [
                            'Yes'])
                }
            })

            json_output.update({
                'GeneralInformation': {
                    'ID'         : row['NISTIR Classification'],
                    'Description': row['Component Description'],
                    'Author'     : row['Author'],
                    'Official'   : root.find('Official').text in ['True',
                                                                  'true'],
                    'DateCreated': root.find('DateCreated').text,
                    'Approved'   : root.find('Approved').text in ['True',
                                                                  'true'],
                    'Incomplete' : root.find('Incomplete').text in ['True',
                                                                    'true'],
                    'Notes'      : row['Comments / Notes']
                }
            })
            for key in json_output['GeneralInformation'].keys():
                if json_output['GeneralInformation'][key] is np.nan:
                    json_output['GeneralInformation'][key] = 'Undefined'

            json_output.update({
                'Ratings': {
                    'DataQuality'  : row['Data Quality'],
                    'DataRelevance': row['Data Relevance'],
                    'Documentation': row['Documentation Quality'],
                    'Rationality'  : row['Rationality'],
                }
            })
            for key in json_output['Ratings'].keys():
                if json_output['Ratings'][key] is np.nan:
                    json_output['Ratings'][key] = 'Undefined'

            DSH = decode_DS_Hierarchy(row['DS Hierarchy'])

            json_output.update({'DSGroups': []})

            for DSG in DSH:
                if DSG[0] in ['MutEx', 'Simul']:
                    mu = row['DS {}, Median Demand'.format(DSG[1][-1])]
                    beta = row[
                        'DS {}, Total Dispersion (Beta)'.format(DSG[1][-1])]
                    if is_float(mu) and is_float(beta):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu),
                            'Beta'        : float(beta),
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : convert_DSG_type[DSG[0]],
                            'DamageStates': DSG[1:]
                        })
                    else:
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu) if is_float(
                                mu) else 'Undefined',
                            'Beta'        : float(beta) if is_float(
                                beta) else 'Undefined',
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : convert_DSG_type[DSG[0]],
                            'DamageStates': DSG[1:]
                        })
                        incomplete = True
                else:
                    mu = row['DS {}, Median Demand'.format(DSG[-1])]
                    beta = row['DS {}, Total Dispersion (Beta)'.format(DSG[-1])]
                    if is_float(mu) and is_float(beta):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu),
                            'Beta'        : float(beta),
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [DSG],
                        })
                    else:
                        json_output['DSGroups'].append({
                            'MedianEDP'   : float(mu) if is_float(
                                mu) else 'Undefined',
                            'Beta'        : float(beta) if is_float(
                                beta) else 'Undefined',
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [DSG],
                        })
                        incomplete = True

            need_INJ = False
            need_RT = False
            for DSG in json_output['DSGroups']:
                DS_list = DSG['DamageStates']
                DSG['DamageStates'] = []
                for DS in DS_list:
                    DSG['DamageStates'].append({
                        'Weight'        : 
                            float(row['DS {}, Probability'.format(DS[-1])]),
                        'LongLeadTime'  : 
                            row['DS {}, Long Lead Time'.format(DS[-1])] in [
                                'YES'],
                        'Consequences'  : {},
                        'Description'   : 
                            row['DS {}, Description'.format(DS[-1])],
                        'RepairMeasures': 
                            row['DS {}, Repair Description'.format(DS[-1])],
                    })

                    IMG = row['DS{}, Illustrations'.format(DS[-1])]
                    if IMG not in ['none', np.nan]:
                        DSG['DamageStates'][-1].update({'DamageImageName': IMG})

                    AA = row['DS {} - Casualty Affected Area'.format(DS[-1])]
                    if (isinstance(AA, string_types) and (is_float(AA.split(' ')[0]))):
                        AA = AA.split(' ')
                        DSG['DamageStates'][-1].update(
                            {'AffectedArea': [int(AA[0]), AA[1]]})
                        need_INJ = True
                    else:
                        DSG['DamageStates'][-1].update(
                            {'AffectedArea': [0, 'SF']})

                    DSG['DamageStates'][-1]['Consequences'].update(
                        {'Injuries': [{}, {}]})

                    INJ0 = DSG[
                        'DamageStates'][-1]['Consequences']['Injuries'][0]
                    INJ_mu = row[
                        'DS {} Serious Injury Rate - Median'.format(DS[-1])]
                    INJ_beta = row[
                        'DS {} Serious Injury Rate - Dispersion'.format(DS[-1])]
                    if is_float(INJ_mu) and is_float(INJ_beta):
                        INJ0.update({
                            'Amount'   : float(INJ_mu),
                            'Beta'     : float(INJ_beta),
                            'CurveType': 'Normal',
                            'Bounds'   : [0., 1.]
                        })                        
                        
                        if INJ_mu != 0.0:
                            need_INJ = True
                            if DSG['DamageStates'][-1]['AffectedArea'][0] == 0:
                                incomplete = True
                    else:
                        INJ0.update({'Amount'   : 
                                         float(INJ_mu) if is_float(INJ_mu) 
                                         else 'Undefined',
                                     'Beta'     : 
                                         float(INJ_beta) if is_float(INJ_beta) 
                                         else 'Undefined',
                                     'CurveType': 'Normal'})
                        if ((INJ0['Amount'] == 'Undefined') or 
                            (INJ0['Beta'] == 'Undefined')):
                            incomplete = True

                    INJ1 = DSG[
                        'DamageStates'][-1]['Consequences']['Injuries'][1]
                    INJ_mu = row['DS {} Loss of Life Rate - Median'.format(DS[-1])]
                    INJ_beta = row['DS {} Loss of Life Rate - Dispersion'.format(DS[-1])]
                    if is_float(INJ_mu) and is_float(INJ_beta):
                        INJ1.update({
                            'Amount'   : float(INJ_mu),
                            'Beta'     : float(INJ_beta),
                            'CurveType': 'Normal',
                            'Bounds'   : [0., 1.]
                        })                    
                        if INJ_mu != 0.0:
                            need_INJ = True
                            if DSG['DamageStates'][-1]['AffectedArea'][0] == 0:
                                incomplete = True
                    else:
                        INJ1.update({'Amount'   : 
                                         float(INJ_mu) if is_float(INJ_mu) 
                                         else 'Undefined',
                                     'Beta'     : 
                                         float(INJ_beta) if is_float(INJ_beta) 
                                         else 'Undefined',
                                     'CurveType': 'Normal', 
                                     'Bounds': [0., 1.]})
                        if ((INJ1['Amount'] == 'Undefined') or 
                            (INJ1['Beta'] == 'Undefined')):
                            incomplete = True

                    DSG['DamageStates'][-1]['Consequences'].update({'RedTag': {}})
                    RT = DSG['DamageStates'][-1]['Consequences']['RedTag']

                    RT_mu = row['DS {}, Unsafe Placard Damage Median'.format(DS[-1])]
                    RT_beta = row['DS {}, Unsafe Placard Damage Dispersion'.format(DS[-1])]
                    if is_float(RT_mu) and is_float(RT_beta):
                        RT.update({
                            'Amount'   : float(RT_mu),
                            'Beta'     : float(RT_beta),
                            'CurveType': 'Normal',
                            'Bounds'   : [0., 1.]
                        })
                        if RT['Amount'] != 0.0:
                            need_RT = True
                    else:
                        RT.update({'Amount'   : 
                                       float(RT_mu[:-1]) if is_float(RT_mu) 
                                       else 'Undefined',
                                   'Beta'     : 
                                       float(RT_beta[:-1]) if is_float(RT_beta) 
                                       else 'Undefined',
                                   'CurveType': 'Normal', 
                                   'Bounds': [0., 1.]})
                        if ((RT['Amount'] == 'Undefined') or 
                            (RT['Beta'] == 'Undefined')):
                            incomplete = True

            # remove the unused fields
            if not need_INJ:
                for DSG in json_output['DSGroups']:
                    for DS in DSG['DamageStates']:
                        del DS['AffectedArea']
                        del DS['Consequences']['Injuries']

            if not need_RT:
                for DSG in json_output['DSGroups']:
                    for DS in DSG['DamageStates']:
                        del DS['Consequences']['RedTag']

            # collect the repair cost and time consequences from the XML file
            DSG_list = root.find('DamageStates').findall('DamageState')
            for DSG_i, DSG_xml in enumerate(DSG_list):

                if DSG_xml.find('DamageStates') is not None:
                    DS_list = (DSG_xml.find('DamageStates')).findall('DamageState')
                    for DS_i, DS_xml in enumerate(DS_list):
                        r_cost, r_time = parse_DS_xml(DS_xml)
                        CONSEQ = json_output['DSGroups'][DSG_i][
                            'DamageStates'][DS_i]['Consequences']
                        CONSEQ.update({
                            'ReconstructionCost': r_cost,
                            'ReconstructionTime': r_time
                        })
                        if ((r_cost['Amount'] == 'Undefined') or 
                            (r_time['Amount'] == 'Undefined')):
                            incomplete = True

                else:
                    r_cost, r_time = parse_DS_xml(DSG_xml)
                    CONSEQ = json_output['DSGroups'][DSG_i][
                        'DamageStates'][0]['Consequences']
                    CONSEQ.update({
                        'ReconstructionCost': r_cost,
                        'ReconstructionTime': r_time
                    })
                    if ((r_cost['Amount'] == 'Undefined') or 
                        (r_time['Amount'] == 'Undefined')):
                        incomplete = True

            if incomplete:
                json_output['GeneralInformation']['Incomplete'] = True
                incomplete_count += 1

            with open(os.path.join(target_dir, comp_ID + '.json'),'w') as f:
                json.dump(json_output, f)

        except:
            warnings.warn(UserWarning(
                'Error converting data for component {}'.format(comp_ID)))
            
def create_HAZUS_json_files(data_dir, target_dir):
    """
    Create JSON data files from publicly available HAZUS data.

    HAZUS damage and loss information is publicly available in the technical
    manuals. The relevant tables have been converted into a JSON input file
    (hazus_data_eq.json) that is stored in the 'resources/HAZUS MH 2.1' folder
    in the pelicun repo. Here we read that file (or a file of similar format)
    and produce damage and loss data for Fragility Groups in the common 
    SimCenter JSON format.
    
    HAZUS handles damage and losses at the assembly level differentiating only
    structural and two types of non-structural component assemblies. In this 
    implementation we consider each of those assemblies a Fragility Group
    and describe their damage and its consequences in a FEMA P58-like framework
    but using the data from the HAZUS Technical Manual.  

    Parameters
    ----------
    data_dir: string
        Path to the folder with the hazus_data_eq JSON file.
    target_dir: string
        Path to the folder where the results shall be saved. The population 
        distribution file will be saved here, the DL JSON files will be saved
        to a 'DL json' subfolder.

    """

    convert_design_level = {
        'High_code'    : 'HC',
        'Moderate_code': 'MC',
        'Low_code'     : 'LC',
        'Pre_code'     : 'PC'
    }

    convert_DS_description = {
        'DS1': 'Slight',
        'DS2': 'Moderate',
        'DS3': 'Extensive',
        'DS4': 'Complete',
        'DS5': 'Collapse',
    }

    # open the raw HAZUS data
    with open(os.path.join(data_dir, 'hazus_data_eq.json'), 'r') as f:
        raw_data = json.load(f)

    design_levels = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'].keys())
    building_types = list(
        raw_data['Structural_Fragility_Groups']['EDP_limits'][
            'Pre_code'].keys())
    occupancy_types = list(raw_data['Structural_Fragility_Groups'][
                               'Reconstruction_cost'].keys())

    S_data = raw_data['Structural_Fragility_Groups']
    NSA_data = raw_data[
        'NonStructural_Acceleration_Sensitive_Fragility_Groups']
    NSD_data = raw_data['NonStructural_Drift_Sensitive_Fragility_Groups']

    for ot in occupancy_types:

        # first, structural fragility groups
        for dl in design_levels:
            for bt in building_types:
                if bt in S_data['EDP_limits'][dl].keys():

                    json_output = {}

                    dl_id = 'S-{}-{}-{}'.format(bt,
                                                convert_design_level[dl],
                                                ot)

                    # this might get replaced by a more descriptive name in the future
                    json_output.update({'Name': dl_id})

                    # General info
                    json_output.update({
                        'GeneralInformation': {
                            'ID'         : dl_id,
                            'Description': dl_id,
                        # this might get replaced by more details in the future
                            # other fields can be added here if needed
                        }
                    })

                    # EDP info
                    json_output.update({
                        'EDP': {
                            'Type': 'Story Drift Ratio',
                            'Unit': [1, 'rad']
                        }
                    })

                    # Damage State info
                    json_output.update({'DSGroups': []})
                    EDP_lim = S_data['EDP_limits'][dl][bt]

                    for dsg_i in range(4):
                        json_output['DSGroups'].append({
                            'MedianEDP'   : EDP_lim[dsg_i],
                            'Beta'        : S_data['Fragility_beta'][dl],
                            'CurveType'   : 'LogNormal',
                            'DSGroupType' : 'Single',
                            'DamageStates': [{
                                'Weight'      : 1.0,
                                'Consequences': {},
                                'Description' : 'DS{}'.format(dsg_i + 1),
                            }]
                        })
                        # the last DSG is different
                        if dsg_i == 3:
                            json_output['DSGroups'][-1][
                                'DSGroupType'] = 'MutuallyExclusive'
                            DS5_w = S_data['P_collapse'][bt]
                            json_output['DSGroups'][-1][
                                'DamageStates'].append({
                                'Weight'      : DS5_w,
                                'Consequences': {},
                                'Description' : 'DS5'
                            })
                            json_output['DSGroups'][-1]['DamageStates'][0][
                                'Weight'] = 1.0 - DS5_w

                    for dsg_i, DSG in enumerate(json_output['DSGroups']):
                        for DS in DSG['DamageStates']:
                            DS_id = DS['Description']
                            DS['Consequences'] = {
                                # injury rates are provided in percentages of the population
                                'Injuries'          : [
                                    {'Amount': val / 100.} for val in
                                    S_data['Injury_rates'][DS_id][bt]],
                                # reconstruction cost is provided in percentages of replacement cost
                                'ReconstructionCost': {"Amount": S_data[
                                                                     'Reconstruction_cost'][
                                                                     ot][
                                                                     dsg_i] / 100.},
                                'ReconstructionTime': {"Amount": S_data[
                                    'Reconstruction_time'][ot][dsg_i]}
                            }
                            DS['Description'] = convert_DS_description[
                                DS['Description']]

                    with open(os.path.join(target_dir + 'DL json/',
                                           dl_id + '.json'), 'w') as f:
                        json.dump(json_output, f)

            # second, nonstructural acceleration sensitive fragility groups
            json_output = {}

            dl_id = 'NSA-{}-{}'.format(convert_design_level[dl], ot)

            # this might get replaced by a more descriptive name in the future
            json_output.update({'Name': dl_id})

            # General info
            json_output.update({
                'GeneralInformation': {
                    'ID'         : dl_id,
                    'Description': dl_id,
                # this might get replaced by more details in the future
                    # other fields can be added here if needed
                }
            })

            # EDP info
            json_output.update({
                'EDP': {
                    'Type': 'Peak Floor Acceleration',
                    'Unit': [1, 'g']
                }
            })

            # Damage State info
            json_output.update({'DSGroups': []})

            for dsg_i in range(4):
                json_output['DSGroups'].append({
                    'MedianEDP'   : NSA_data['EDP_limits'][dl][dsg_i],
                    'Beta'        : NSA_data['Fragility_beta'],
                    'CurveType'   : 'LogNormal',
                    'DSGroupType' : 'Single',
                    'DamageStates': [{
                        'Weight'      : 1.0,
                        'Consequences': {
                            # reconstruction cost is provided in percentages of replacement cost
                            'Reconstruction_cost': {'Amount': NSA_data[
                                                                  'Reconstruction_cost'][
                                                                  ot][
                                                                  dsg_i] / 100.}
                        },
                        'Description' : convert_DS_description[
                            'DS{}'.format(dsg_i + 1)]
                    }]
                })

            with open(
                os.path.join(target_dir + 'DL json/', dl_id + '.json'),
                'w') as f:
                json.dump(json_output, f)

                # third, nonstructural drift sensitive fragility groups
        json_output = {}

        dl_id = 'NSD-{}'.format(ot)

        # this might get replaced by a more descriptive name in the future
        json_output.update({'Name': dl_id})

        # General info
        json_output.update({
            'GeneralInformation': {
                'ID'         : dl_id,
                'Description': dl_id,
            # this might get replaced by more details in the future
                # other fields can be added here if needed
            }
        })

        # EDP info
        json_output.update({
            'EDP': {
                'Type': 'Story Drift Ratio',
                'Unit': [1, 'rad']
            }
        })

        # Damage State info
        json_output.update({'DSGroups': []})

        for dsg_i in range(4):
            json_output['DSGroups'].append({
                'MedianEDP'   : NSD_data['EDP_limits'][dsg_i],
                'Beta'        : NSD_data['Fragility_beta'],
                'CurveType'   : 'LogNormal',
                'DSGroupType' : 'Single',
                'DamageStates': [{
                    'Weight'      : 1.0,
                    'Consequences': {
                        # reconstruction cost is provided in percentages of replacement cost
                        'Reconstruction_cost': {
                            'Amount': NSD_data['Reconstruction_cost'][ot][
                                          dsg_i] / 100.}
                    },
                    'Description' : convert_DS_description[
                        'DS{}'.format(dsg_i + 1)]
                }]
            })

        with open(os.path.join(target_dir + 'DL json/', dl_id + '.json'),
                  'w') as f:
            json.dump(json_output, f)

    # finally, prepare the population distribution data
    PD_data = raw_data['Population_Distribution']

    pop_output = {}
    for ot in PD_data.keys():
        night_ids = raw_data['Parts_of_day']['Nighttime']
        day_ids = raw_data['Parts_of_day']['Daytime']
        commute_ids = raw_data['Parts_of_day']['Commute']

        daily_pop = np.ones(24)
        daily_pop[night_ids] = PD_data[ot][0]
        daily_pop[day_ids] = PD_data[ot][1]
        daily_pop[commute_ids] = PD_data[ot][2]
        daily_pop = list(daily_pop)

        # HAZUS does not introduce monthly and weekend/weekday variation
        pop_output.update({ot: {
            "weekday": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            },
            "weekend": {
                "daily"  : daily_pop,
                "monthly": list(np.ones(12))
            }
        }})

    with open(os.path.join(target_dir, 'population.json'), 'w') as f:
        json.dump(pop_output, f)

def write_SimCenter_DL_output(output_path, output_df, index_name='#Num', 
                              collapse_columns = True, stats_only=False):
    
    output_df = deepcopy(output_df)
    
    # if the summary flag is set, then not all realizations are returned, but
    # only the first two moments and the empirical CDF through 100 percentiles
    if stats_only:
        output_df = output_df.describe(np.arange(1, 100)/100.)
        
    # the name of the index column is replaced with the provided value
    output_df.index.name = index_name


    # multiple levels of indices are collapsed into a single level if needed
    # TODO: check for the number of levels and prepare a smarter collapse method
    if collapse_columns:
        output_df.columns = [('{}/{}'.format(s0, s1)).replace(' ', '_')
                     for s0, s1 in zip(output_df.columns.get_level_values(0),
                                       output_df.columns.get_level_values(1))]
        
    # write the results in a csv file
    # TODO: provide other file formats
    output_df.to_csv(output_path)