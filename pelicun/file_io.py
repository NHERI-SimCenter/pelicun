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
    read_SimCenter_UQ_input
    read_P58_population_distribution
    read_P58_component_data
    write_SimCenter_DL_output

"""

import numpy as np
import json
import xml.etree.ElementTree as ET
from distutils.util import strtobool

from .base import *

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
    
    Occupancy_to_P58 = dict([
        ('__type__', 'occupancy to P58'),
        ('Commercial Office', ['office', ]),
        ('Elementary School', ['education',
                               'school']),
        ('Middle School', []),
        ('High School', []),
        ('Healthcare', ['healthcare', ]),
        ('Hospitality', ['hospitality',
                         'hotel']),
        ('Multi-Unit Residential', ['residence',
                                    'Residential']),
        ('Retail', ['retail', ]),
        ('Warehouse', ['warehouse',
                       'industrial']),
        ('Research Laboratories', ['research', ]),
    ])
    
    with open(input_path, 'r') as f:
        jd = json.load(f)

    # replace random values with their mean
    # we will load actual realizations of those values in a different method
    randoms = dict((rv['name'], rv['mean']) for rv in jd['RandomVariables'])
    jd = jd['GI']
    for attrib in jd.keys():
        if attrib in randoms:
            jd[attrib] = randoms[attrib]

    # load the other attributes
    # note that we assume that everything is provided in standard units
    data = dict(
        name             = jd['name'],
        area             = float(jd['area']) * m2,
        stories          = int(jd['numStory']),
        year_built       = int(jd['yearBuilt']),
        str_type         = jd['structType'],
        occupancy        = _classify(Occupancy_to_P58, jd['occupancy']),
        height           = float(jd['height']) * m,
        replacement_cost = float(jd['replacementCost']),
        replacement_time = float(jd['replacementTime']),
    )
    if jd['population'] == 'auto':
        data.update(dict(population = 'auto'))
    else:
        data.update(dict(population = np.asarray(jd['population'])))

    # print the parsed data to the screen if requested
    if verbose:
        for attribute, value in data.items():
            print(attribute, ':', value)
        print('-' * 75)

    return data

def read_SimCenter_EDP_input(input_path, verbose=False):
    
    Demand_to_Acronym = dict([
        ('__type__', 'demand to acronym'),
        ('PFA', ['max_abs_acceleration', ]),
        ('PID', ['max_drift', ]),
        ('RD', ['residual_disp', ]),
    ])
    
    # initialize the dictionary of EDP data
    data = {}

    with open(input_path, 'r') as f:
        jd = json.load(f)['EngineeringDemandParameters']

    events = []
    for i, event_data in enumerate(jd):
        # to make sure every IM level has a unique and informative ID
        events.append('{}_{}'.format(i + 1, event_data['name']))

    for i, edp in enumerate(jd[0]['responses']):
        kind = _classify(Demand_to_Acronym, edp['type'])
        if kind not in data.keys():
            data.update({kind: []})

        scale_factor = 1.0  # we assume that EDPs are provided in standard units

        raw_data = dict([
            (event,
             np.array(jd[e]['responses'][i]['scalar_data'],
                      dtype=np.float64) * scale_factor
             )
            for e, event in enumerate(events)
        ])

        data[kind].append(dict(
            cline=int(edp['cline']),
            floor=int(edp['floor1' if kind == 'PID' else 'floor']),
            raw_data=raw_data,
            floor2=int(edp['floor2']) if kind == 'PID' else None,
        ))

    # print the parsed data to the screen if requested
    if verbose:
        for kind, EDP_list in data.items():
            print(kind)
            for EDP_attributes in EDP_list:
                for attribute, value in EDP_attributes.items():
                    if attribute is not 'raw_data':
                        print('\t', attribute, value)
                    else:
                        print('\t', attribute)
                        for event, edp_data in value.items():
                            print('\t\t', event,
                                  '| {} samples: '.format(len(edp_data)),
                                  '[{:.4f}, ..., {:.4f}]'.format(min(edp_data),
                                                                 max(edp_data)))
                print()
            print('-' * 75)

    return data


def read_P58_population_distribution(path_POP, occupancy, verbose=False):
    with open(path_POP, 'r') as f:
        jd = json.load(f)

    data = jd[occupancy]

    # convert peak population to persons/m2
    data['peak'] = data['peak'] / (1000. * ft2)

    if verbose:
        for attribute, value_set in data.items():
            if type(value_set) is not dict:
                print(attribute, ':', value_set)
            else:
                print(attribute)
                for sub_attribute, value in value_set.items():
                    if len(value) < 13:
                        print('\t', sub_attribute, ':', value)
                    else:
                        print('\t', sub_attribute, ':', value[:12], '...')
        print('-' * 75)

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
        'proportions',
        'directions',
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
        c_data['proportions'] = ci_data['props']
        c_data['directions'] = ci_data['dirs']
        c_data['locations'] = ci_data['locations']
        c_data['cov'] = ci_data['cov']

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
            for att, vals in c_data.items():
                if type(vals) is not dict:
                    print(att, ':', vals)
                else:
                    print(att)
                    for att2, vals2 in vals.items():
                        if type(vals2) is not dict:
                            print('  ', att2, ':', vals2)
                        else:
                            print('  ', att2)
                            for att3, vals3 in vals2.items():
                                if type(vals3) is not dict:
                                    print('    ', att3, ':', vals3)
                                else:
                                    print('    ', att3)
                                    for att4, vals4 in vals3.items():
                                        if type(vals4) is not dict:
                                            print('      ', att4, ':', vals4)
                                        else:
                                            print('      ', att4)
                                            for att5, vals5 in vals4.items():
                                                if type(vals5) is not dict:
                                                    print('        ',
                                                          att5, ':', vals5)
                                                else:
                                                    print('        ', att5)
                                                    for att6, vals6 in vals5.items():
                                                        if type(
                                                            vals6) is not dict:
                                                            print('          ',
                                                                  att6, ':',
                                                                  vals6)
                                                        else:
                                                            print('          ',
                                                                  att6)
            print('-' * 75)

    return data

def write_SimCenter_DL_output(output_path):
    pass