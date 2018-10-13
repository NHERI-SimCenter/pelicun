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
    # We assume that these are always specified in the file.
    DV = LM['DecisionVariables']
    for target_att, source_att in [
        ['injuries', 'Injuries'],
        ['rec_cost', 'ReconstructionCost'],
        ['rec_time', 'ReconstructionTime'],
        ['red_tag', 'RedTag'],
    ]:
        data['decision_variables'].update({target_att: bool(DV[source_att])})
    DV = data['decision_variables']

    # general information
    if 'GeneralInformation' in jd:
        GI = jd['GeneralInformation']
    else:
        GI = None

    # units
    if (GI is not None) and ('units' in GI.keys()):
        for key, value in GI['units'].items():
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
                    'acceleration': 
                        data['units']['length'] / data['units']['time'] ** 2.})
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
        # The following lines are commented out for now, because we do not use
        # these pieces of data anyway.
        #['stories', 'stories', int, ''],
        #['building_type', 'type', str, ''],
        #['height', 'height', float, 'length'],
        #['year_built', 'year', int, ''],
    ]:
        if (GI is not None) and (source_att in GI.keys()):
            #if unit_kind is not '':
            #    f_unit = data['units'][unit_kind]
            #else:
            #    f_unit = 1
            f_unit = data['units'][unit_kind]
            att_value = f_conv(GI[source_att]) * f_unit
            data['general'].update({target_att: att_value})
        else:
            if DV[dv_req]:
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
                "Infinite EDP limits are assumed."))
            
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
            # scale the limits by the units
            DGDL = data['general']['detection_limits']
            for EDP_kind, value in DGDL.items():
                if (EDP_kind in EDP_units.keys()) and (value is not None):
                    f_EDP = data['units'][EDP_units[EDP_kind]]
                    DGDL[EDP_kind] = DGDL[EDP_kind] * f_EDP
        else:
            warnings.warn(UserWarning(
                "EDP detection limits were not defined in the input file."))
        
        if 'YieldDriftRatio' in LM['BuildingResponse'].keys():
            data['general'].update({
                'yield_drift': float_or_None(
                    LM['BuildingResponse']['YieldDriftRatio'])})
        elif DV['rec_cost'] or DV['rec_time']:
            warnings.warn(UserWarning(
                "Yield drift ratio was not defined in the input file."))
            
    else:
        warnings.warn(UserWarning(
            "Building response characteristics were not defined in the input "
            "file."))
        
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
            "No additional uncertainties were defined in the input file."))
    
    if 'Inhabitants' in LM.keys():
        if 'OccupancyType' in LM['Inhabitants'].keys():
            data['general'].update({
                'occupancy_type': LM['Inhabitants']['OccupancyType']})
        elif DV['injuries']:
            warnings.warn(UserWarning(
                "Occupancy type was not defined in the input file."))
    
        if 'PeakPopulation' in LM['Inhabitants'].keys():
            data['general'].update({
                'population': [float(pop) for pop in
                               LM['Inhabitants']['PeakPopulation'].split(',')]})
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
        'per ATC recommendation' : 'ATC,'
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
                          index_col='%eval_id')
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

    if verbose:
        pp.pprint(data)

    return data

def read_component_DL_data(path_CMP, comp_info, verbose=False):
    """
    Read the damage and loss data for the components of the asset.
    
    DL data for each component is assumed to be stored in an xml file. The name
    of the file is the ID (key) of the component in the comp_info dictionary.
    Besides the filename, the comp_info dictionary is also used to get other
    pieces of data about the component that is not available in the xml files.
    Therefore, the following attributes need to be provided in the comp_info:
    ['quantities', 'csg_weights', 'dirs', 'kind', 'distribution', 'cov', 
    'unit', 'locations'] Further information about these attributes is 
    available in the Input section of the documentation.
    
    Parameters
    ----------
    path_CMP: string
        Location of the folder that contains the component data in xml files.
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
                warnings.warn(UserWarning(
                    'Unknown unit for affected floor area: {}'.format(unit)))
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
            warnings.warn(UserWarning(
                'Fragility information for {} is incomplete'.format(c_id)))

        EDP = root.find('EDPType')
        EDP_type = EDP.find('TypeName').text
        demand_factor = 1.0
        if EDP_type == 'Story Drift Ratio':
            demand_type = 'PID'
        elif EDP_type == 'Acceleration':
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
        else:
            demand_type = None
            warnings.warn(UserWarning(
                'Unexpected EDP type: {}'.format(EDP_type)))
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

def write_SimCenter_DL_output(output_path, output_df, index_name='#Num', 
                              collapse_columns = True, stats_only=False):
    """
    
    Parameters
    ----------
    output_path
    output_df
    index_name
    collapse_columns

    Returns
    -------

    """
    
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