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
# Pouria Kourehpaz
# Kuanshi Zhong

"""
This module has classes and methods that handle file input and output.

.. rubric:: Contents

.. autosummary::

    read_SimCenter_DL_input
    read_SimCenter_EDP_input
    read_population_distribution
    read_component_DL_data
    write_SimCenter_DL_output
    write_SimCenter_EDP_output
    write_SimCenter_DM_output
    write_SimCenter_DV_output

"""

from .base import *
from pathlib import Path
from .db import convert_Series_to_dict

import json, posixpath

from time import sleep


import warnings

convert_dv_name = {
    'DV_rec_cost': 'Reconstruction Cost',
    'DV_rec_time': 'Reconstruction Time',
    'DV_injuries_0': 'Injuries lvl. 1',
    'DV_injuries_1': 'Injuries lvl. 2',
    'DV_injuries_2': 'Injuries lvl. 3',
    'DV_injuries_3': 'Injuries lvl. 4',
    'DV_red_tag': 'Red Tag ',
}

dependency_to_acronym = {
        'btw. Fragility Groups'  : 'FG',
        'btw. Performance Groups': 'PG',
        'btw. Floors'            : 'LOC',
        'btw. Directions'        : 'DIR',
        'btw. Component Groups'  : 'CSG',
        'btw. Damage States'     : 'DS',
        'Independent'            : 'IND',
        'per ATC recommendation' : 'ATC',
    }

HAZUS_occ_converter = {
        'RES' : 'Residential',
        'COM' : 'Commercial',
        'REL' : 'Commercial',
        'EDU' : 'Educational',
        'IND' : 'Industrial',
        'AGR' : 'Industrial'
    }

# this is a convenience function for converting strings to float or None
def float_or_None(string):
    try:
        res = float(string)
        return res
    except:
        return None

def int_or_None(string):
    try:
        res = int(string)
        return res
    except:
        return None

def process_loc(string, stories):
    try:
        res = int(string)
        return [res, ]
    except:
        if "-" in string:
            s_low, s_high = string.split('-')
            s_low = process_loc(s_low, stories)
            s_high = process_loc(s_high, stories)
            return list(range(s_low[0], s_high[0]+1))
        elif string == "all":
            return list(range(1, stories+1))
        elif string == "top":
            return [stories,]
        elif string == "roof":
            return [stories,]
        else:
            return None

def get_required_resources(input_path, assessment_type):
    """
    List the data files required to perform an assessment.

    It extracts the information from the config file about the methods and
    functional data required for the analysis and provides a list of paths to
    the files that would be used.
    This method is helpful in an HPC context to copy the required resources to
    the local node from the shared file storage.

    Parameters
    ----------
    input_path: string
        Location of the DL input json file.
    assessment_type: {'P58', 'HAZUS_EQ', 'HAZUS_HU'}
        Identifies the default databases based on the type of assessment.

    Returns
    -------
    resources: list of strings
        A list of paths to the required resource files.
    """

    resources = {}

    AT = assessment_type

    with open(input_path, 'r') as f:
        jd = json.load(f)

    DL_input = jd['DamageAndLoss']

    loss = DL_input.get('LossModel', None)
    if loss is not None:
        inhabitants = loss.get('Inhabitants', None)
        dec_vars    = loss.get('DecisionVariables', None)

        if dec_vars is not None:
            injuries = bool(dec_vars.get('Injuries', False))
    else:
        inhabitants = None
        dec_vars = None
        injuries = False

    # check if the user specified custom data sources
    path_CMP_data = DL_input.get("ComponentDataFolder", "")

    if path_CMP_data == "":
        # Use the P58 path as default
        path_CMP_data = pelicun_path + CMP_data_path[AT]

    resources.update({'component': path_CMP_data})

    # HAZUS combination of flood and wind losses
    if ((AT == 'HAZUS_HU') and (DL_input.get('Combinations', None) is not None)):
        path_combination_data = pelicun_path + CMP_data_path['HAZUS_MISC']
        resources.update({'combination': path_combination_data})

    # The population data is only needed if we are interested in injuries
    if inhabitants is not None:
        path_POP_data = inhabitants.get("PopulationDataFile", "")
    else:
        path_POP_data = ""

    if ((injuries) and (path_POP_data == "")):
        path_POP_data = pelicun_path + POP_data_path[AT]
        resources.update({'population': path_POP_data})

    return resources

def read_SimCenter_DL_input(input_path, assessment_type='P58', verbose=False):
    """
    Read the damage and loss input information from a json file.

    The SimCenter in the function name refers to having specific fields
    available in the file. Such a file is automatically prepared by the
    SimCenter PBE Application, but it can also be easily manipulated or created
    manually. The accepted input fields are explained in detail in the Input
    section of the documentation.

    Parameters
    ----------
    input_path: string
        Location of the DL input json file.
    assessment_type: {'P58', 'HAZUS_EQ', 'HAZUS_HU'}
        Tailors the warnings and verifications towards the type of assessment.
        default: 'P58'.
    verbose: boolean
        If True, the function echoes the information read from the file. This
        can be useful to ensure that the information in the file is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with all the damage and loss data.

    """

    AT = assessment_type

    log_msg('\t\tOpening the configuration file...')
    with open(input_path, 'r') as f:
        jd = json.load(f)

    # get the data required for DL
    data = dict([(label, dict()) for label in [
        'general', 'units', 'unit_names', 'components', 'collapse_modes',
        'decision_variables', 'dependencies', 'data_sources', 'damage_logic'
    ]])

    # create a few internal variables for convenience
    DL_input = jd['DamageAndLoss']

    log_msg('\t\tLoading the Models:')
    log_msg('\t\t\tResponse Model')
    response = DL_input.get('ResponseModel',None)
    if response is not None:
        res_description = response.get('ResponseDescription', None)
        det_lims = response.get('DetectionLimits', None)
        uncertainty = response.get('AdditionalUncertainty', None)

    else:
        res_description = None
        det_lims = None
        uncertainty = None

    log_msg('\t\t\tDamage Model')
    damage = DL_input.get('DamageModel',None)
    if damage is not None:
        irrep_res_drift = damage.get('IrreparableResidualDrift', None)
        coll_prob = damage.get('CollapseProbability', None)
        coll_lims = damage.get('CollapseLimits', None)
        design_lvl = damage.get('DesignLevel', None)
        struct_type = damage.get('StructureType', None)

    else:
        irrep_res_drift = None
        coll_prob = None
        coll_lims = None
        design_lvl = None
        struct_type = None

    log_msg('\t\t\tLoss Model')
    loss = DL_input.get('LossModel', None)
    if loss is not None:
        repl_cost = loss.get('ReplacementCost', None)
        repl_time = loss.get('ReplacementTime', None)
        dec_vars = loss.get('DecisionVariables', None)
        inhabitants = loss.get('Inhabitants', None)

    else:
        repl_cost = None
        repl_time = None
        dec_vars = None
        inhabitants = None

    log_msg('\t\t\tPerformance Model')
    components = DL_input.get('Components', None)

    depends = DL_input.get('Dependencies', None)
    coll_modes = DL_input.get('CollapseModes', None)

    dam_logic = DL_input.get('DamageLogic', None)

    # decision variables of interest
    if dec_vars is not None:
        for target_att, source_att in [ ['injuries', 'Injuries'],
                                        ['rec_cost', 'ReconstructionCost'],
                                        ['rec_time', 'ReconstructionTime'],
                                        ['red_tag', 'RedTag'], ]:
            val = bool(dec_vars.get(source_att, False))
            data['decision_variables'].update({target_att: val})
    else:
        show_warning("No decision variables specified in the input file.")
        log_msg("Assuming that only reconstruction cost and time needs to be calculated.")
        data['decision_variables'].update({ 'injuries': False,
                                            'rec_cost': True,
                                            'rec_time': True,})
        # red tag is only used by P58 now
        if AT == 'P58':
            data['decision_variables'].update({'red_tag': False})

    dec_vars = data['decision_variables']

    # data sources
    # default data locations
    default_data_name = {
        'P58'     : 'FEMA P58 second edition',
        'HAZUS_EQ': 'HAZUS MH 2.1 earthquake',
        'HAZUS_HU': 'HAZUS MH 2.1 hurricane'
    }

    # check if the user specified custom data sources
    path_CMP_data = DL_input.get("ComponentDataFolder", "")

    if path_CMP_data == "":
        # Use the P58 path as default
        path_CMP_data = pelicun_path + CMP_data_path[AT]

    data['data_sources'].update({'path_CMP_data': path_CMP_data})

    # HAZUS combination of flood and wind losses
    if AT == 'HAZUS_HU':
        log_msg('\t\t\tCombinations')
        comb = DL_input.get('Combinations', None)
        path_combination_data = pelicun_path
        if comb is not None:
            if AT == 'HAZUS_HU':
                path_combination_data += CMP_data_path['HAZUS_MISC']
        data['data_sources'].update({'path_combination_data': path_combination_data})
        data['loss_combination'] = comb

    # The population data is only needed if we are interested in injuries
    if inhabitants is not None:
        path_POP_data = inhabitants.get("PopulationDataFile", "")
    else:
        path_POP_data = ""

    if data['decision_variables']['injuries']:
        if path_POP_data == "":
            path_POP_data = pelicun_path + POP_data_path[AT]

        data['data_sources'].update({'path_POP_data': path_POP_data})

    # general information
    GI = jd.get("GeneralInformation", None)
    data['GI'] = GI

    # units
    if (GI is not None) and ('units' in GI.keys()):
        for key, value in GI['units'].items():
            if value == 'in':
                value = 'inch'
            if value in globals().keys():
                data['unit_names'].update({key: value})
            else:
                show_warning("Unknown {} unit: {}".format(key, value))

        if 'length' in data['unit_names'].keys():
            if 'area' not in data['unit_names']:
                data['unit_names'].update({
                    'area': data['unit_names']['length']+'2'})

            if 'volume' not in data['unit_names']:
                data['unit_names'].update({
                    'volume': data['unit_names']['length']+'3'})

            if 'speed' not in data['unit_names'].keys():
                data['unit_names'].update({
                    'speed': data['unit_names']['length']+'ps'})

            if 'acceleration' not in data['unit_names'].keys():
                data['unit_names'].update({
                    #'acceleration': 1.0 })
                    'acceleration': data['unit_names']['length']+'ps2'})
    else:
        show_warning("No units were specified in the input file.")
        data['unit_names'].update(default_units)

    for unit_type, unit_name in data['unit_names'].items():
        data['units'].update({unit_type: globals()[unit_name]})

    # other attributes that can be used by a P58 assessment
    if AT == 'P58':
        for target_att, source_att, f_conv, unit_kind, dv_req in [
            ['plan_area', 'PlanArea', float, 'area', 'injuries'],
            ['stories', 'NumberOfStories', int, '', 'all'],
        ]:
            if (GI is not None) and (source_att in GI.keys()):
                if unit_kind != '':
                    f_unit = data['units'][unit_kind]
                else:
                    f_unit = 1
                att_value = f_conv(GI[source_att]) * f_unit
                data['general'].update({target_att: att_value})
            else:
                if (dv_req != '') and ((dv_req == 'all') or dec_vars[dv_req]):
                    raise ValueError(
                        "{} has to be specified in the DL input file to "
                        "estimate {} decision variable(s).".format(source_att,
                                                                   dv_req))
    elif AT.startswith('HAZUS'):
        data['general'].update({'stories': int(GI['NumberOfStories'])})

    # is this a coupled assessment?
    if res_description is not None:
        data['general'].update({'coupled_assessment':
                            res_description.get('CoupledAssessment', False)})
    else:
        data['general'].update({'coupled_assessment': False})

    # Performance Model
    # Having components defined is not necessary, but if a component is defined
    # then all of its attributes need to be specified. Note that the required
    # set of attributes depends on the type of assessment.
    if components is not None:
        for fg_id, frag_group in components.items():
            if AT == 'P58':
                # TODO: allow location and direction inputs with '-' in them
                comp_data = {
                    'locations'   : [],
                    'directions'  : [],
                    'quantities'  : [],
                    'unit'        : [],
                    'distribution': [],
                    'cov'         : [],
                    'csg_weights':  [],
                }

                for comp in frag_group:
                    locs = []
                    for loc_ in comp['location'].split(','):
                        for l in process_loc(loc_, data['general']['stories']):
                            locs.append(l)
                    locs.sort()

                    dirs = sorted([int_or_None(dir_)
                                   for dir_ in comp['direction'].split(',')])
                    qnts = [float(qnt)
                            for qnt in comp['median_quantity'].split(',')]
                    csg_weights = (qnts / np.sum(qnts)).tolist()
                    qnts = np.sum(qnts)

                    pg_count = len(locs) * len(dirs)

                    comp_data['locations'] = (comp_data['locations'] +
                                               [l for l in locs for d in dirs])
                    comp_data['directions'] = (comp_data['directions'] +
                                              dirs * len(locs))

                    unit = comp['unit']
                    if unit not in globals().keys():
                        raise ValueError(
                            "Unknown unit for component {}: {}".format(fg_id,
                                                                       unit))
                    for i in range(pg_count):
                        comp_data['quantities'].append(qnts)
                        comp_data['csg_weights'].append(csg_weights)
                        comp_data['unit'].append(unit)
                        comp_data['distribution'].append(comp.get('distribution', 'N/A'))
                        comp_data['cov'].append(comp.get('cov', None))

                sorted_ids = np.argsort(comp_data['locations'])
                for key in ['locations', 'directions', 'quantities',
                            'csg_weights', 'distribution', 'cov']:
                    comp_data[key] = [comp_data[key][s_id] for s_id in sorted_ids]

                if len(set(comp_data['unit'])) != 1:
                    raise ValueError(
                        "Multiple types of units specified for fragility group "
                        "{}. Make sure that every component group in a "
                        "fragility group is defined using the same "
                        "unit.".format(fg_id))
                comp_data['unit'] = comp_data['unit'][0]

                # aggregate PGs that are in the same loc & dir
                PG_loc_dir_list = list(zip(comp_data['locations'], comp_data['directions']))
                to_aggregate = set([x for x in PG_loc_dir_list if PG_loc_dir_list.count(x) > 1])
                for combo in to_aggregate:
                    PG_loc_dir_list = list(zip(comp_data['locations'], comp_data['directions']))
                    combo_ids = [i for i,e in enumerate(PG_loc_dir_list) if e==combo]

                    c_base = combo_ids[0]
                    comp_data['csg_weights'][c_base] = (np.array(comp_data['csg_weights'][c_base]) * comp_data['quantities'][c_base]).tolist()
                    for ci in combo_ids[1:]:
                        comp_data['quantities'][c_base] += comp_data['quantities'][ci]
                        comp_data['csg_weights'][c_base] += (np.array(comp_data['csg_weights'][ci]) * comp_data['quantities'][ci]).tolist()
                    comp_data['csg_weights'][c_base] = (np.array(comp_data['csg_weights'][c_base]) / comp_data['quantities'][c_base]).tolist()

                    for ci in combo_ids[1:][::-1]:
                        for key in ['locations', 'directions', 'quantities',
                        'csg_weights', 'distribution', 'cov']:
                            del comp_data[key][ci]

            elif AT.startswith('HAZUS'):
                comp_data = {
                    'locations'   : [],
                    'directions'  : [],
                    'quantities'  : [],
                    'unit'        : [],
                    'distribution': [],
                    'cov'         : [],
                    'csg_weights':  [],
                }

                for comp in frag_group:
                    if 'location' in comp:
                        locs = []
                        for loc_ in comp['location'].split(','):
                            for l in process_loc(loc_, data['general']['stories']):
                                locs.append(l)
                        locs.sort()
                    else:
                        locs = [1,]

                    if 'direction' in comp:
                        dirs = sorted([int_or_None(dir_)
                                       for dir_ in comp['direction'].split(',')])
                    else:
                        dirs = [1, ]

                    if 'median_quantity' in comp:
                        qnts = [float(qnt)
                                for qnt in comp['median_quantity'].split(',')]
                        csg_weights = (qnts / np.sum(qnts)).tolist()
                        qnts = np.sum(qnts)

                    pg_count = len(locs) * len(dirs)

                    comp_data['locations'] = (comp_data['locations'] +
                                               [l for l in locs for d in dirs])
                    comp_data['directions'] = (comp_data['directions'] +
                                              dirs * len(locs))

                    if 'unit' in comp:
                        unit = comp['unit']
                        if unit not in globals().keys():
                            raise ValueError(
                                "Unknown unit for component {}: {}".format(fg_id,
                                                                           unit))
                    else:
                        unit = 'ea'

                    for i in range(pg_count):
                        comp_data['quantities'].append(qnts)
                        comp_data['csg_weights'].append(csg_weights)
                        comp_data['unit'].append(unit)
                        comp_data['distribution'].append(comp.get('distribution', 'N/A'))
                        comp_data['cov'].append(comp.get('cov', None))

                sorted_ids = np.argsort(comp_data['locations'])
                for key in ['locations', 'directions', 'quantities',
                            'csg_weights', 'distribution', 'cov']:
                    comp_data[key] = [comp_data[key][s_id] for s_id in sorted_ids]

                if len(set(comp_data['unit'])) != 1:
                    print(comp_data['unit'])
                    raise ValueError(
                        "Multiple types of units specified for fragility group "
                        "{}. Make sure that every component group in a "
                        "fragility group is defined using the same "
                        "unit.".format(fg_id))
                comp_data['unit'] = comp_data['unit'][0]

            # store the component data
            data['components'].update({fg_id: comp_data})
    else:
        show_warning("No components were defined in the input file.")

    # collapse modes
    if AT == 'P58':
        # Having collapse modes defined is not necessary, but if a collapse mode
        # is defined, then all of its attributes need to be specified.
        if coll_modes is not None:
            for coll_mode in coll_modes:
                cm_data = {
                    'w'            : float(coll_mode['weight']),
                    'injuries'     : [float(inj) for inj in
                                      coll_mode['injuries'].split(',')],
                    'affected_area': [float(cfar) for cfar in
                                      coll_mode['affected_area'].split(',')],
                }
                if len(cm_data['affected_area']) == 1:
                    cm_data['affected_area'] = (np.ones(data['general']['stories'])*cm_data['affected_area']).tolist()
                if len(cm_data['injuries']) == 1:
                    cm_data['injuries'] = (np.ones(data['general']['stories'])*cm_data['injuries']).tolist()
                data['collapse_modes'].update({coll_mode['name']: cm_data})
        else:
            show_warning("No collapse modes were defined in the input file.")

    # the number of realizations has to be specified in the file
    if res_description is not None:
        realizations = res_description.get("Realizations", None)
        if realizations is not None:
            data['general'].update({'realizations': int(realizations)})
    else:
        raise ValueError(
            "Number of realizations is not specified in the input file.")

    if AT in ['P58', 'HAZUS_EQ']:
        EDP_keys = ['PID', 'PRD', 'PFA',
                    'PGV', 'RID', 'PMD',
                    'PGA', 'SA', 'SV', 'SD',
                    'RDR','DWD']
    elif AT in ['HAZUS_HU']:
        EDP_keys = ['PWS', 'PIH']
    elif AT in ['HAZUS_FL']:
        EDP_keys = ['PIH']

    # response model info ------------------------------------------------------
    if response is None:
        show_warning("Response model characteristics were not defined in the input "
            "file")

    # detection limits
    if ((response is not None) and (det_lims is not None)):
        data['general'].update({
            'detection_limits':
                dict([(key, float_or_None(value)) for key, value in
                      det_lims.items()])})
        DGDL = data['general']['detection_limits']
        # scale the limits by the units
        for EDP_kind, value in DGDL.items():
            if (EDP_kind in EDP_units.keys()) and (value is not None):
                f_EDP = data['units'][EDP_units[EDP_kind]]
                DGDL[EDP_kind] = DGDL[EDP_kind] * f_EDP
    else:
        data['general'].update({'detection_limits':{}})

    # make sure that detection limits are initialized
    for key in EDP_keys:
        if key not in data['general']['detection_limits'].keys():
            data['general']['detection_limits'].update({key: None})

    # response description
    if ((response is not None) and (res_description is not None)):
        #TODO: move the collapse-related data to another field
        data['general'].update({'response': {
            'EDP_distribution': res_description.get('EDP_Distribution',
                                                    'lognormal'),
            'EDP_dist_basis':   res_description.get('BasisOfEDP_Distribution',
                                                    'all results')}})
    else:
        data['general'].update({'response': {
            'EDP_distribution': 'lognormal',
            'EDP_dist_basis'  : 'all results'}})

    # additional uncertainty
    if ((response is not None) and (uncertainty is not None)):
        data['general'].update({
            'added_uncertainty': {
                'beta_gm': float_or_None(uncertainty['GroundMotion']),
                'beta_m' : float_or_None(uncertainty['Modeling'])}})
    else:
        data['general'].update({
            'added_uncertainty': {
                'beta_gm': None,
                'beta_m': None
            }
        })

    # damage model info --------------------------------------------------------
    if damage is None:
        if AT == 'P58':
            show_warning("Damage model characteristics were not defined in the "
                "input file")
        elif AT.startswith('HAZUS'):
            pass

    # P58-specific things
    if AT == 'P58':
        # EDP limits for collapse
        if ((damage is not None) and (coll_lims is not None)):
            # load the limits
            data['general'].update({
                'collapse_limits':
                    dict([(key, float_or_None(value)) for key, value
                          in coll_lims.items()])})

            # scale the limits according to their units
            DGCL = data['general']['collapse_limits']
            for EDP_kind, value in DGCL.items():
                if (EDP_kind in EDP_units.keys()) and (value is not None):
                    f_EDP = data['units'][EDP_units[EDP_kind]]
                    DGCL[EDP_kind] = DGCL[EDP_kind] * f_EDP
        else:
            data['general'].update({'collapse_limits': {}})

        # make sure that collapse limits are initialized
        for key in EDP_keys:
            if key not in data['general']['collapse_limits'].keys():
                data['general']['collapse_limits'].update({key: None})

        # irreparable drift
        if ((damage is not None) and (irrep_res_drift is not None)):
            data['general'].update({
                'irreparable_res_drift':
                    dict([(key, float_or_None(value)) for key, value in
                          irrep_res_drift.items()])})
            # TODO: move this in the irreparable part of general
            yield_drift = irrep_res_drift.get("YieldDriftRatio", None)
            if yield_drift is not None:
                data['general'].update({
                    'yield_drift': float_or_None(yield_drift)})
            elif ((data['decision_variables']['rec_cost']) or
                  (data['decision_variables']['rec_time'])):
                data['general'].update({'yield_drift': 0.01})

        elif ((data['decision_variables']['rec_cost']) or
              (data['decision_variables']['rec_time'])):
            pass
            #TODO: show this warning in the log file instead

            # show_warning(
            #     "Residual drift limits corresponding to irreparable "
            #     "damage were not defined in the input file. We assume that "
            #     "damage is repairable regardless of the residual drift.")
            # we might need to have a default yield drift here

        # collapse probability
        if 'response' not in data['general'].keys():
            data['general'].update({'response': {}})
        if ((damage is not None) and (coll_prob is not None)):
            data['general']['response'].update({
                'coll_prob'   : coll_prob.get('Value', 'estimated'),
                'CP_est_basis': coll_prob.get('BasisOfEstimate', 'raw EDP')})
            if data['general']['response']['coll_prob'] != 'estimated':
                data['general']['response']['coll_prob'] = \
                    float_or_None(data['general']['response']['coll_prob'])
        else:
            data['general']['response'].update({
                'coll_prob'       : 'estimated',
                'CP_est_basis'    : 'raw EDP'})

    # loss model info ----------------------------------------------------------
    if loss is None:
        show_warning("Loss model characteristics were not defined in the input file")

    # replacement cost
    if ((loss is not None) and (repl_cost is not None)):
        data['general'].update({
            'replacement_cost': float_or_None(repl_cost)})
    elif data['decision_variables']['rec_cost']:
        if AT == 'P58':
            show_warning("Building replacement cost was not defined in the "
                "input file.")
        elif AT.startswith('HAZUS'):
            raise ValueError(
                "Building replacement cost was not defined in the input "
                "file.")

    # replacement time
    if ((loss is not None) and (repl_time is not None)):
        data['general'].update({
            'replacement_time': float_or_None(repl_time)})
    elif data['decision_variables']['rec_time']:
        if AT == 'P58':
            show_warning("Building replacement cost was not defined in the "
                "input file.")
        elif AT.startswith('HAZUS'):
            raise ValueError(
                "Building replacement cost was not defined in the input "
                "file.")

    # inhabitants
    if data['decision_variables']['injuries']:
        if ((loss is not None) and (inhabitants is not None)):

            # occupancy type
            occupancy = inhabitants.get("OccupancyType", None)
            if occupancy is not None:
                data['general'].update({'occupancy_type': occupancy})
            else:
                raise ValueError("Occupancy type was not defined in the input "
                                 "file.")

            # event time
            event_time = inhabitants.get("EventTime", None)
            data['general'].update({'event_time': event_time})

            # peak population
            peak_pop = inhabitants.get("PeakPopulation", None)
            if peak_pop is not None:
                peak_pop = [float_or_None(pop) for pop in peak_pop.split(',')]

                # If the number of stories is specified...
                if 'stories' in data['general'].keys():
                    stories = data['general']['stories']
                    pop_in = len(peak_pop)

                    # If only one value is provided then we assume that it
                    # corresponds to the whole building
                    if pop_in == 1:
                        peak_pop = (np.ones(stories)*peak_pop[0]/stories).tolist()

                    # If more than one value is provided then we assume they
                    # define population for every story
                    else:
                        # If the population list does not provide values
                        # for every story the values are assumed to correspond
                        # to the lower stories and the upper ones are filled
                        # with zeros
                        for s in range(pop_in, stories):
                            peak_pop.append(0)

                        if pop_in > 1 and pop_in != stories:
                            show_warning(
                                "Peak population was specified to some, but not all "
                                "stories. The remaining stories are assumed to have "
                                "zero population."
                            )

                data['general'].update({'population': peak_pop})
            else:
                raise ValueError(
                    "Peak population was not defined in the input file.")
        else:
            raise ValueError(
                "Information about inhabitants was not defined in the input "
                "file.")

    # dependencies -------------------------------------------------------------

    # set defaults
    # We assume 'Independent' for all unspecified fields except for the
    # fragilities where 'per ATC recommendation' is the default setting.

    for target_att, source_att, dv_req in [
        ['quantities', 'Quantities', ''],
        ['fragilities', 'Fragilities', ''],
        ['injuries', 'Injuries', 'injuries'],
        ['rec_costs', 'ReconstructionCosts', 'rec_cost'],
        ['rec_times', 'ReconstructionTimes', 'rec_time'],
        ['red_tags', 'RedTagProbabilities', 'red_tag'],]:

        if ((depends is not None) and (source_att in depends.keys())):
            data['dependencies'].update({
                target_att:dependency_to_acronym[depends[source_att]]})
        #elif dv_req == '' or data['decision_variables'][dv_req]:
        else:
            if target_att != 'fragilities':
                data['dependencies'].update({target_att: 'IND'})
            else:
                data['dependencies'].update({target_att: 'ATC'})

            log_msg("\t\t\t\t\t"+
                "Correlation between {} was not ".format(source_att)+
                "defined in the input file. Using default values.")

    if ((depends is not None) and ('CostAndTime' in depends.keys())):
        data['dependencies'].update({
            'cost_and_time': bool(depends['CostAndTime'])})
    elif ((data['decision_variables']['rec_cost']) or
          (data['decision_variables']['rec_time'])):
        data['dependencies'].update({'cost_and_time': False})
        log_msg("\t\t\t\t\t"+
            "Correlation between reconstruction cost and time was not "
            "defined in the input file. Using default values.")

    if ((depends is not None) and ('InjurySeverities' in depends.keys())):
        data['dependencies'].update({
            'injury_lvls': bool(depends['InjurySeverities'])})
    elif data['decision_variables']['injuries']:
        data['dependencies'].update({'injury_lvls': False})
        log_msg("\t\t\t\t\t"+
            "Correlation between injury levels was not defined in the "
            "input file. Using default values.")

    # damage logic info --------------------------------------------------------

    data['damage_logic'] = dam_logic

    if verbose: pp.pprint(data)

    return data

def read_SimCenter_EDP_input(input_path,
                             #EDP_kinds=('PID', 'PFA'),
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
    log_msg('\t\tOpening the input file...')
    # If the file name ends with csv, we assume a standard csv file
    if input_path.endswith('csv'):
        EDP_raw = pd.read_csv(input_path, header=0, index_col=0)

    # otherwise, we assume that a dakota file is provided...
    else:
        # the read_csv method in pandas is sufficiently versatile to handle the
        # tabular format of dakota
        EDP_raw = pd.read_csv(input_path, sep=r'\s+', header=0, index_col=0)

    # set the index to be zero-based
    if EDP_raw.index[0] == 1:
        EDP_raw.index = EDP_raw.index - 1

    # search the header for EDP information
    for column in EDP_raw.columns:

        # extract info about the location, direction, EDP_kind and scenario
        info = column.split('-')

        if len(info) != 4:
            continue

        kind = info[1].replace(' ','')

        #for kind in EDP_kinds:
        #    if kind in column:

        if kind not in data.keys():
            data.update({kind: []})

        # get the scale factor to perform unit conversion
        f_unit = units[kind.split('_')[0]]

        # store the data
        data[kind].append(dict(
            raw_data=(EDP_raw[column].values * f_unit).tolist(),
            location=info[2],
            direction=info[3],
            scenario_id=info[0]
        ))

    if verbose: pp.pprint(data)

    return data

def read_population_distribution(path_POP, occupancy, assessment_type='P58',
    verbose=False):
    """
    Read the population distribution from an external json file.

    The population distribution is expected in a format used in FEMA P58, but
    the list of occupancy categories can be modified and/or extended beyond
    those available in that document. The population distributions for the
    occupancy categories from FEMA P58 and HAZUS MH are provided with pelicun
    in the population.json files in the corresponding folder under resources.

    Note: Population distributions in HAZUS do not have a 1:1 mapping to the
    occupancy types provided in the Technical Manual. We expect inputs to
    follow the naming convention in the HAZUS Technical Manual and convert
    those to the broader categories here automatically. During conversion, the
    following assumptions are made about the occupancy classes: i) RES classes
    are best described as Residential; ii) COM and REL as Commercial; iii) EDU
    as Educational; iv) IND and AGR as Industrial; v) Hotels do not have a
    matching occupancy class.

    Parameters
    ----------
    path_POP: string
        Location of the population distribution json file.
    occupancy: string
        Identifies the occupancy category.
    assessment_type: {'P58', 'HAZUS_EQ'}
        Tailors the warnings and verifications towards the type of assessment.
        default: 'P58'.
    verbose: boolean
        If True, the function echoes the information read from the file. This
        can be useful to ensure that the information in the file is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with the population distribution data.
    """

    AT = assessment_type

    # Convert the HAZUS occupancy classes to the categories used to define
    # population distribution.
    if AT == 'HAZUS_EQ':
        occupancy = HAZUS_occ_converter.get(occupancy[:3], None)

        if occupancy is None:
            warnings.warn(UserWarning(
                'Unknown, probably invalid, occupancy class for HAZUS '
                'assessment: {}. When defining population distribution, '
                'assuming RES1 instead.'.format(occupancy)))
            occupancy = 'Residential'

    # Load the population data

    # If a json file is provided:
    if path_POP.endswith('json'):
        with open(path_POP, 'r') as f:
            jd = json.load(f)

        data = jd[occupancy]

    # else if an HDF5 file is provided
    elif path_POP.endswith('hdf'):

        store = pd.HDFStore(path_POP)
        store.open()
        pop_table = store.select('pop', where=f'index in {[occupancy, ]}')
        store.close()

        if pop_table is not None:
            data = convert_Series_to_dict(pop_table.loc[occupancy, :])
        else:
            raise IOError("Couldn't read the HDF file for POP data.")

    # convert peak population to persons/m2
    if 'peak' in data.keys():
        data['peak'] = data['peak'] / (1000. * ft2)

    if verbose: # pragma: no cover
        pp.pprint(data)

    return data


def read_combination_DL_data(path_combination_data, comp_info, assessment_type='HAZUS_HU',
    verbose=False):
    """
    Read the combination rules for hurricane damage and loss for the components of the asset.

    Parameters
    ----------
    path_combination_data: string
        Location of the folder that contains the combination rules.
    comp_info: list
        List of data names that contains the comibnation rules.
    assessment_type: {'HAZUS_HU'}
        Tailors the warnings and verifications towards the type of assessment.
        default: 'HAZUS_HU'.
    verbose: boolean
        If True, the function echoes the information read from the files. This
        can be useful to ensure that the information in the files is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with damage and loss data for each component.

    """

    AT = assessment_type

    comb_data_dict = {}
    if os.path.isdir(path_combination_data):
        tmp_dir = Path(path_combination_data).resolve()
        for c_id in comp_info:
            with open(tmp_dir / f'{c_id}.json', 'r') as f:
                comb_data_dict.update({c_id: json.load(f)})
    ## TODO: hdf type
    elif path_combination_data.endswith('hdf'):
        for c_id in comp_info:

            store = pd.HDFStore(path_combination_data)
            store.open()
            comb_data_table = store['HAZUS Subassembly Loss Ratio']
            store.close()

            if comb_data_table is not None:
                comb_data_dict.update(
                    {c_id: {'LossRatio': comb_data_table[c_id].tolist()}})
            else:
                raise IOError("Couldn't read the HDF file for combination data.")

    return comb_data_dict


def read_component_DL_data(path_CMP, comp_info, assessment_type='P58', avail_edp=None,
    verbose=False):
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
    assessment_type: {'P58', 'HAZUS_EQ', 'HAZUS_HU'}
        Tailors the warnings and verifications towards the type of assessment.
        default: 'P58'.
    avail_edp: list
        EDP name string list. default: None
    verbose: boolean
        If True, the function echoes the information read from the files. This
        can be useful to ensure that the information in the files is properly
        read by the method.

    Returns
    -------
    data: dict
        A dictionary with damage and loss data for each component.

    """

    AT = assessment_type

    data = dict([(c_id, dict([(key, None) for key in [
        'ID',
        'name',
        'description',
        #'kind',
        'demand_type',
        'directional',
        'correlation',
        'offset',
        'incomplete',
        'locations',
        'quantities',
        'csg_weights',
        #'dir_weights',
        'directions',
        'distribution_kind',
        'cov',
        'unit',
        'DSG_set',
    ]])) for c_id in comp_info.keys()])

    s_cmp_keys = sorted(data.keys())
    DL_data_dict = {}

    # If the path_CMP is a folder we assume it contains a set of json files
    if os.path.isdir(path_CMP):

        CMP_dir = Path(path_CMP).resolve()

        for c_id in s_cmp_keys:
            with open(CMP_dir / f'{c_id}.json', 'r') as f:
                DL_data_dict.update({c_id: json.load(f)})

    # else if an HDF5 file is provided we assume it contains the DL data
    elif path_CMP.endswith('hdf'):
        # hurricane
        if AT == 'HAZUS_HU':
            for c_id in s_cmp_keys:
                if c_id.startswith('fl'):
                    path_CMP_m = path_CMP.replace('.hdf','_FL.hdf') # flood DL
                else:
                    path_CMP_m = path_CMP.replace('.hdf','_HU.hdf') # wind DL

                store = pd.HDFStore(path_CMP_m)
                store.open()
                CMP_table = store.select('data', where=f'index in {c_id}')
                store.close()

                if CMP_table is not None:
                    DL_data_dict.update(
                        {c_id: convert_Series_to_dict(CMP_table.loc[c_id, :])})
                else:
                    raise IOError("Couldn't read the HDF file for DL data.")
        else:

            store = pd.HDFStore(path_CMP)
            store.open()
            CMP_table = store.select('data', where=f'index in {s_cmp_keys}')
            store.close()

            if CMP_table is not None:
                for c_id in s_cmp_keys:
                    DL_data_dict.update(
                        {c_id: convert_Series_to_dict(CMP_table.loc[c_id, :])})
            else:
                raise IOError("Couldn't read the HDF file for DL data.")

    else:
        raise ValueError(
            "Component data source not recognized. Please provide "
            "either a folder with DL json files or an HDF5 table.")

    # for each component
    for c_id in s_cmp_keys:
        c_data = data[c_id]

        DL_data = DL_data_dict[c_id]

        DL_GI = DL_data['GeneralInformation']
        DL_EDP = DL_data['EDP']
        DL_DSG = DL_data['DSGroups']

        # First, check if the DL data is complete. Incomplete data can lead to
        # all kinds of problems, so in such a case we display a warning and do
        # not use the component. This can be relaxed later if someone creates a
        # method to replace unknown values with reasonable estimates.
        if 'Incomplete' in DL_GI.keys():
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
        c_data['locations'] = ci_data['locations']
        c_data['directions'] = ci_data['directions']

        c_data['unit'] = globals()[ci_data['unit']]
        c_data['quantities'] = [(np.asarray(qnt) * c_data['unit']).tolist()
                                for qnt in ci_data['quantities']]
        c_data['csg_weights'] = ci_data['csg_weights']

        c_data['distribution_kind'] = ci_data['distribution']
        c_data['cov'] = [float_or_None(cov) for cov in ci_data['cov']]

        c_data['ID'] = c_id
        c_data['name'] = DL_data['Name']
        c_data['description'] = DL_GI['Description']
        c_data['offset'] =int(DL_EDP.get('Offset', 0))
        c_data['correlation'] = int(DL_data.get('Correlated', False))
        c_data['directional'] = int(DL_data.get('Directional', False))

        EDP_type = DL_EDP['Type']
        if DL_EDP['Unit'][1] == 'in':
            DL_EDP['Unit'][1] = 'inch'
        demand_factor = globals()[DL_EDP['Unit'][1]] * DL_EDP['Unit'][0]

        demand_type = EDP_to_demand_type.get(EDP_type, None)

        if demand_type is None:
            if EDP_type in ['Link Rotation Angle',
                            'Link Beam Chord Rotation']:

                warnings.warn(UserWarning(
                    'Component {} requires {} as EDP, which is not yet '
                    'implemented.'.format(c_data['ID'], EDP_type)))

            else:  # pragma: no cover
                warnings.warn(UserWarning(
                    f'Unexpected EDP type in component {c_id}: {EDP_type}'))

            del data[c_id]
            continue

        else:
            c_data['demand_type'] = demand_type

        if demand_type in EDP_offset_adjustment.keys():
            c_data['offset'] = c_data['offset'] + EDP_offset_adjustment[demand_type]

        # check if the DL_EDP is available in the _EDP_in
        # if not: delete the relavant data and print warning info
        if (avail_edp is not None) and (demand_type not in avail_edp):
            del data[c_id]
            warnings.warn(UserWarning('{} as EDP is not available in the input. '
               'This may occur in using IMasEDP with missing IM field(s). '
               'Note: the corresponding fragility group {} is neglected while the simulation will continue.'.format(demand_type, c_data['ID'])))
            continue

        # dictionary to convert DL data to internal representation
        curve_type = {'LogNormal': 'lognormal',
                      'Normal'   : 'normal',
                      'N/A'      : None}
        DS_set_kind = {'MutuallyExclusive' : 'mutually exclusive',
                       'mutually exclusive': 'mutually exclusive',
                       'simultaneous'      : 'simultaneous',
                       'Simultaneous'      : 'simultaneous',
                       'single'            : 'single',
                       'Single'            : 'single'}

        # load the damage state group information
        c_data['DSG_set'] = dict()
        QNT_unit = DL_data.get('QuantityUnit', [1, 'ea'])
        data_unit = QNT_unit[0] * globals()[QNT_unit[1]]
        for DSG_id, DSG_i in enumerate(DL_DSG):
            DSG_data = dict(
                theta=float(DSG_i['MedianEDP']) * demand_factor,
                sig=float(DSG_i['Beta']),
                DS_set_kind=DS_set_kind[DSG_i['DSGroupType']],
                distribution_kind = curve_type[DSG_i['CurveType']],
                DS_set={}
            )
            # sig needs to be scaled for normal distributions
            if DSG_data['distribution_kind'] == 'normal':
                DSG_data['sig'] = DSG_data['sig'] * demand_factor

            for DS_id, DS_i in enumerate(DSG_i['DamageStates']):
                DS_data = {'description': DS_i['Description'],
                           'weight'     : DS_i['Weight']}

                DS_C = DS_i.get('Consequences', None)
                if DS_C is not None:
                    if 'ReconstructionCost' in DS_C.keys():
                        DS_CC = DS_C['ReconstructionCost']
                        if isinstance(DS_CC['Amount'], list):
                            DS_data.update({'repair_cost': {
                                'medians'          : np.array([float(a) for a in DS_CC['Amount']]),
                                'quantities'       : np.array(DS_CC['Quantity']),
                                'distribution_kind': curve_type[DS_CC.get('CurveType','N/A')],
                                'cov'              : DS_CC.get('Beta',None),
                            }})

                            # convert the quantity units to standard ones
                            DS_data['repair_cost']['quantities'] *= data_unit
                            DS_data['repair_cost']['quantities'] = DS_data['repair_cost']['quantities'].tolist()
                        else:
                            DS_data.update({'repair_cost': {
                                'medians': np.array([float(DS_CC['Amount']),]),
                                'distribution_kind': curve_type[DS_CC.get('CurveType','N/A')],
                                'cov'              : DS_CC.get('Beta',None),
                            }})

                        # convert the median units to standard ones
                        DS_data['repair_cost']['medians'] /= data_unit
                        DS_data['repair_cost']['medians'] = DS_data['repair_cost']['medians'].tolist()

                    if 'ReconstructionTime' in DS_C.keys():
                        DS_CT = DS_C['ReconstructionTime']
                        if isinstance(DS_CT['Amount'], list):
                            DS_data.update({'repair_time': {
                                'medians'          : np.array([float(a) for a in DS_CT['Amount']]),
                                'quantities'       : np.array(DS_CT['Quantity']),
                                'distribution_kind': curve_type[DS_CT.get('CurveType','N/A')],
                                'cov'              : DS_CT.get('Beta',None),
                            }})

                            # convert the quantity units to standard ones
                            DS_data['repair_time']['quantities'] *= data_unit
                            DS_data['repair_time']['quantities'] = DS_data['repair_time']['quantities'].tolist()
                        else:
                            DS_data.update({'repair_time': {
                                'medians': np.array([float(DS_CT['Amount']),]),
                                'distribution_kind': curve_type[DS_CT.get('CurveType','N/A')],
                                'cov'              : DS_CT.get('Beta',None),
                            }})

                        # convert the median units to standard ones
                        DS_data['repair_time']['medians'] /= data_unit
                        DS_data['repair_time']['medians'] = DS_data['repair_time']['medians'].tolist()

                    if 'RedTag' in DS_C.keys():
                        DS_CR = DS_C['RedTag']
                        DS_data.update({'red_tag': {
                            'theta': DS_CR['Amount'],
                            # 'distribution_kind': curve_type[DS_CR['CurveType']],
                            'cov'  : DS_CR['Beta'],
                        }})

                    if 'Injuries' in DS_C.keys():
                        DS_CI = DS_C['Injuries']
                        if DS_CI[0].get('Beta') is not None:
                            DS_data.update({'injuries': {
                                'theta': [float(I_i['Amount']) for I_i in DS_CI],
                                # 'distribution_kind': curve_type[DS_CR['CurveType']],
                                'cov'  : [I_i['Beta'] for I_i in DS_CI],
                            }})
                        else:
                            DS_data.update({
                                'injuries': [I_i['Amount'] for I_i in DS_CI]})

                        # if there is a chance of injuries, load the affected floor area
                        affected_area, unit = DS_i.get('AffectedArea',
                                                       [0.0, 'SF'])
                        if unit == 'SF':
                            affected_area = affected_area * SF
                        else: # pragma: no cover
                            warnings.warn(UserWarning(
                                'Unknown unit for affected floor area: {}'.format(
                                    unit)))
                            affected_area = 0.
                        DS_data.update({'affected_area': affected_area})

                        # convert the units to standard ones
                        DS_data['affected_area'] /= data_unit

                DSG_data['DS_set'].update({'DS-' + str(DS_id + 1): DS_data})

            c_data['DSG_set'].update({'DSG-' + str(DSG_id + 1): DSG_data})

    if verbose: # pragma: no cover
        for c_id, c_data in data.items():
            print(c_id)
            pp.pprint(c_data)

    return data

def write_SimCenter_DL_output(output_dir, output_filename, output_df, index_name='#Num',
                              collapse_columns = True, stats_only=False):

    # if the summary flag is set, then not all realizations are returned, but
    # only the first two moments and the empirical CDF through 100 percentiles
    if stats_only:
        #output_df = output_df.describe(np.arange(1, 100)/100.)
        #output_df = output_df.describe([0.1,0.5,0.9])
        if len(output_df.columns) > 0:
            output_df = describe(output_df)
        else:
            output_df = describe(np.zeros(len(output_df.index)))
    else:
        output_df = output_df.copy()

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
    log_msg('\t\t\tSaving file {}'.format(output_filename))
    file_path = posixpath.join(output_dir, output_filename)
    output_df.to_csv(file_path)
    # TODO: this requires pandas 1.0+ > wait until next release
    #with open(file_path[:-3]+'zip', 'w') as f:
    #    output_df.to_csv(f, compression=dict(mehtod='zip', archive_name=output_filename))

def write_SimCenter_BIM_output(output_dir, BIM_filename, BIM_dict):

    #flatten the dictionary
    BIM_flat_dict = {}
    for key, item in BIM_dict.items():
        if isinstance(item, dict):
            for sub_key, sub_item in item.items():
                BIM_flat_dict.update({f'{key}_{sub_key}': sub_item})
        else:
            BIM_flat_dict.update({key: [item,]})

    # create the output DF
    #BIM_flat_dict.update({"index": [0,]})
    for header_to_remove in ['geometry', 'Footprint']:
        try:
            BIM_flat_dict.pop(header_to_remove)
        except:
            pass

    df_res = pd.DataFrame.from_dict(BIM_flat_dict)

    df_res.dropna(axis=1, how='all', inplace=True)

    df_res.to_csv('BIM.csv')

def write_SimCenter_EDP_output(output_dir, EDP_filename, EDP_df):

    # initialize the output DF
    col_info = np.transpose([col.split('-')[1:] for col in EDP_df.columns])

    EDP_types = np.unique(col_info[0])
    EDP_locs = np.unique(col_info[1])
    EDP_dirs = np.unique(col_info[2])

    MI = pd.MultiIndex.from_product(
        [EDP_types, EDP_locs, EDP_dirs, ['median', 'beta']],
        names=['type', 'loc', 'dir', 'stat'])

    df_res = pd.DataFrame(columns=MI, index=[0, ])
    if ('PID', '0') in df_res.columns:
        del df_res[('PID', '0')]

    # store the EDP statistics in the output DF
    for col in np.transpose(col_info):
        df_res.loc[0, (col[0], col[1], col[2], 'median')] = EDP_df[
            '1-{}-{}-{}'.format(col[0], col[1], col[2])].median()
        if np.min(EDP_df['1-{}-{}-{}'.format(col[0], col[1], col[2])]) <= 0:
            # negative EDP values are also possible, so switching to normal
            # distribution (e.g., inundation height PIH can be negative)
            df_res.loc[0, (col[0], col[1], col[2], 'beta')] = \
                EDP_df['1-{}-{}-{}'.format(col[0], col[1], col[2])].std()
        else:
            # assume lognormal distribution for this kind of EDP
            df_res.loc[0, (col[0], col[1], col[2], 'beta')] = np.log(
                EDP_df['1-{}-{}-{}'.format(col[0], col[1], col[2])]).std()

    df_res.dropna(axis=1, how='all', inplace=True)

    df_res = df_res.astype(float) #.round(4)

    # save the output
    df_res.to_csv('EDP.csv')

def write_SimCenter_DM_output(output_dir, DM_filename, SUMMARY_df, DMG_df):

    # first, get the collapses from the SUMMARY_df
    df_res_c = pd.DataFrame([0,],
        columns=pd.MultiIndex.from_tuples([('probability',' '),]),
        index=[0, ])
    df_res_c['probability'] = SUMMARY_df[('collapses', 'collapsed')].mean()

    # aggregate the damage data along Performance Groups
    DMG_agg = DMG_df.groupby(level=['FG', 'DSG_DS'], axis=1).sum()

    comp_types = []
    FG_list = [c for c in DMG_agg.columns.get_level_values('FG').unique()]
    for comp_type in ['S', 'NS', 'NSA', 'NSD']:
        if np.sum([fg.startswith(comp_type) for fg in FG_list]) > 0:
            comp_types.append(comp_type)
    if np.sum([np.any([fg.startswith(comp_type) for comp_type in comp_types])
                       for fg in FG_list]) != len(FG_list):
        comp_types.append('other')

    # second, get the damage state likelihoods
    df_res_l = pd.DataFrame(
        columns=pd.MultiIndex.from_product([comp_types,
                                            ['0', '1_1', '2_1', '3_1', '4_1', '4_2']],
                                           names=['comp_type', 'DSG_DS']),
        index=[0, ])

    # third, get the damage quantities conditioned on damage state
    # we do not do that for now
    # df_res_q = pd.DataFrame(
    #     columns=pd.MultiIndex.from_product([comp_types,
    #                                         ['1_1', '2_1', '3_1', '4_1', '4_2']],
    #                                        names=['comp_type', 'DSG_DS']),
    #     index=[0, ])

    for comp_type in ['NSA', 'NSD', 'NS', 'other']:
        if comp_type in comp_types:
            del df_res_l[(comp_type, '4_2')]
            # del df_res_q[(comp_type, '4_2')]

    # for each type of component...
    for comp_type in comp_types:

        # select the corresponding subset of columns
        if comp_type == 'other':
            type_cols = [fg for fg in FG_list
                         if np.all([~fg.startswith(comp_type) for comp_type in comp_types])]
        else:
            type_cols = [c for c in DMG_agg.columns.get_level_values('FG').unique()
                         if c.startswith(comp_type)]

        df_sel = DMG_agg.loc[:, type_cols].groupby(level='DSG_DS',axis=1).sum()
        df_sel = df_sel / len(type_cols)

        # calculate the probability of DSG exceedance
        df_sel[df_sel > 0.0] = df_sel[df_sel > 0.0] / df_sel[df_sel > 0.0]

        cols = df_sel.columns
        for i in range(len(cols)):
            filter = np.where(df_sel.iloc[:, i].values > 0.0)[0]
            df_sel.iloc[filter, idx[0:i]] = 1.0

        df_sel_exc = pd.Series(np.mean(df_sel.values, axis=0),
                               index=df_sel.columns)

        DS_0 = 1.0 - df_sel_exc['1_1']
        for i in range(len(df_sel_exc.index) - 1):
            df_sel_exc.iloc[i] = df_sel_exc.iloc[i] - df_sel_exc.iloc[i + 1]

        # Add the probability of no damage for convenience.
        df_sel_exc.loc['0'] = DS_0
        df_sel_exc = df_sel_exc.sort_index()

        # store the results in the output DF
        for dsg_i in df_sel_exc.index:
            if df_sel_exc[dsg_i] > 0.0:
                df_res_l.loc[:, idx[comp_type, dsg_i]] = df_sel_exc[dsg_i]

        # get the quantity of components in the highest damage state
        # skip this part for now to reduce file size
        if False:
            df_init = DMG_agg.loc[:, type_cols].groupby(level='DSG_DS', axis=1).sum()
            df_init = (df_init / len(type_cols)).round(2)

            df_sel = df_sel.sum(axis=1)

            for lvl, lvl_label in zip([1.0, 2.0, 3.0, 4.0, 5.0],
                                      ['1_1', '2_1', '3_1', '4_1', '4_2']):

                df_cond = df_init[df_sel == lvl]

                if df_cond.size > 0:
                    unique_vals, unique_counts = np.unique(
                        df_cond[lvl_label].values, return_counts=True)
                    unique_counts = np.around(unique_counts / df_cond.shape[0],
                                              decimals=4)
                    sorter = np.argsort(unique_counts)[::-1][:4]
                    DQ = list(zip(unique_vals[sorter], unique_counts[sorter]))

                    # store the damaged quantities in the output df
                    df_res_q.loc[:,idx[comp_type, lvl_label]] = str(DQ)

    # join the output dataframes
    #df_res = pd.concat([df_res_c, df_res_l, df_res_q], axis=1,
    #    keys=['Collapse','DS likelihood','Damaged Quantities'])
    df_res = pd.concat([df_res_c, df_res_l], axis=1, keys=['Collapse','DS likelihood'])

    # save the output
    with open(posixpath.join(output_dir, DM_filename), 'w') as f:
        df_res.to_csv(f)

def write_SimCenter_DM_output_hu(output_dir, DM_filename, SUMMARY_df, DMG_df):

    # first, get the collapses from the SUMMARY_df
    df_res_c = pd.DataFrame([0,],
        columns=pd.MultiIndex.from_tuples([('probability',' '),]),
        index=[0, ])
    df_res_c['probability'] = SUMMARY_df[('collapses', 'collapsed')].mean()

    # aggregate the damage data along Performance Groups
    DMG_agg = DMG_df.groupby(level=['FG', 'DSG_DS'], axis=1).sum()

    comp_types = []
    FG_list = [c for c in DMG_agg.columns.get_level_values('FG').unique()]

    for fg_i, fg_id in enumerate(FG_list):
        cur_DMG = DMG_agg.groupby(level=['FG'], axis=1).get_group(fg_id)

        for comp_type in ['CECB', 'CERB', 'MECB', 'MERB', 'MH', 'MLRI', 'MLRM',
                          'MMUH', 'MSF', 'SECB', 'SERB', 'SPMB', 'WMUH', 'WSF']:
            if fg_id.startswith(comp_type) > 0:
                comp_types.append('Wind')
        for comp_type in ['fl']:
            if fg_id.startswith(comp_type) > 0:
                comp_types.append('Flood')

        # second, get the damage state likelihoods
        tmp_colname = list(cur_DMG.columns.get_level_values('DSG_DS'))
        tmp_colname.insert(0,'0')
        df_res_l = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[comp_types[fg_i]], tmp_colname],
                                               names=['comp_type', 'DSG_DS']),
            index=[0, ])

        # third, get the damage quantities conditioned on damage state
        tmp_colname = list(cur_DMG.columns.get_level_values('DSG_DS'))
        tmp_colname.append('4_2')

        df_res_q = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[comp_types[fg_i]], tmp_colname],
                                               names=['comp_type', 'DSG_DS']),
            index=[0, ])

        type_cols = fg_id
        df_sel = cur_DMG.loc[:, type_cols].groupby(level='DSG_DS',axis=1).sum()
        df_sel = df_sel / len(type_cols)

        # calculate the probability of DSG exceedance
        df_sel[df_sel > 0.0] = df_sel[df_sel > 0.0] / df_sel[df_sel > 0.0]

        cols = df_sel.columns
        for i in range(len(cols)):
            filter = np.where(df_sel.iloc[:, i].values > 0.0)[0]
            df_sel.iloc[filter, idx[0:i]] = 1.0

        df_sel_exc = pd.Series(np.mean(df_sel.values, axis=0),
                               index=df_sel.columns)

        DS_0 = 1.0 - df_sel_exc['1_1']
        for i in range(len(df_sel_exc.index) - 1):
            df_sel_exc.iloc[i] = df_sel_exc.iloc[i] - df_sel_exc.iloc[i + 1]

        # Add the probability of no damage for convenience.
        df_sel_exc.loc['0'] = DS_0
        df_sel_exc = df_sel_exc.sort_index()

        # store the results in the output DF
        df_res_l.loc[:, idx[comp_types[fg_i], :]] = df_sel_exc.values

        # get the quantity of components in the highest damage state
        # skip this part for now to reduce file size
        if False:
            df_init = cur_DMG.loc[:, type_cols].groupby(level='DSG_DS', axis=1).sum()
            df_init = (df_init / len(type_cols)).round(2)

            df_sel = df_sel.sum(axis=1)

            for lvl, lvl_label in zip([1.0, 2.0, 3.0, 4.0, 5.0],
                                      ['1_1', '2_1', '3_1', '4_1', '4_2']):

                df_cond = df_init[df_sel == lvl]

                if df_cond.size > 0:
                    unique_vals, unique_counts = np.unique(
                        df_cond[lvl_label].values, return_counts=True)
                    unique_counts = np.around(unique_counts / df_cond.shape[0],
                                              decimals=4)
                    sorter = np.argsort(unique_counts)[::-1][:4]
                    DQ = list(zip(unique_vals[sorter], unique_counts[sorter]))

                    # store the damaged quantities in the output df
                    df_res_q.loc[:,idx[comp_type, lvl_label]] = str(DQ)

        # join the output dataframes
        if fg_i == 0:
            df_res = df_res_l
        else:
            df_res = pd.concat([df_res, df_res_l], axis = 1, keys=['DS likelihood','DS likelihood'])

    # join the output dataframes
    if len(FG_list) == 1:
        # wind-only or flood-only results
        df_res = pd.concat([df_res_c, df_res_l], axis=1, keys=['Collapse','DS likelihood'])
    else:
        df_res['Collapse','probability',' '] = df_res_c['probability']
    # save the output
    with open(posixpath.join(output_dir, DM_filename), 'w') as f:
        df_res.to_csv(f)

def write_SimCenter_DM_output_old(output_dir, DM_filename, DMG_df):

    # Start with the probability of being in a particular damage state.
    # Here, the damage state of the building (asset) is defined as the highest
    # damage state among the building components/component groups. This works
    # well for a HAZUS assessment, but something more sophisticated is needed
    # for a FEMA P58 assessment.

    # Determine the probability of DS exceedance by collecting the DS from all
    # components and assigning ones to all lower damage states.
    DMG_agg = DMG_df.T.groupby('DSG_DS').sum().T
    DMG_agg[DMG_agg > 0.0] = DMG_agg[DMG_agg > 0.0] / DMG_agg[DMG_agg > 0.0]

    cols = DMG_agg.columns
    for i in range(len(cols)):
        filter = np.where(DMG_agg.iloc[:,i].values > 0.0)[0]
        DMG_agg.iloc[filter,idx[0:i]] = 1.0

    # The P(DS=ds) probability is determined by subtracting consecutive DS
    # exceedance probabilites. This will not work well for a FEMA P58 assessment
    # with Damage State Groups that include multiple Damage States.
    #DMG_agg_mean = DMG_agg.describe().loc['mean',:]
    DMG_agg_mean = pd.Series(np.mean(DMG_agg.values, axis=0), index=DMG_agg.columns)

    DS_0 = 1.0 - DMG_agg_mean['1_1']
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
    #DMG_mean = DMG_df.describe().loc['mean',:]
    DMG_mean = pd.Series(np.mean(DMG_df.values, axis=0), index=DMG_df.columns)

    # and save the results in the output json file.
    for FG in sorted(DMG_mean.index.get_level_values('FG').unique()):
        DM.update({str(FG):{}})

        for PG in sorted(
            DMG_mean.loc[idx[FG],:].index.get_level_values('PG').unique()):
            DM[str(FG)].update({str(PG):{}})

            for DS in sorted(
                DMG_mean.loc[idx[FG],:].loc[idx[:,PG],:].index.get_level_values('DSG_DS').unique()):
                DM[str(FG)][str(PG)].update({str(DS): DMG_mean.loc[(FG,PG,DS)]})

    log_msg('\t\t\tSaving file {}'.format(DM_filename))
    with open(posixpath.join(output_dir, DM_filename), 'w') as f:
        json.dump(DM, f, indent = 2)

def write_SimCenter_DV_output(output_dir, DV_filename, GI, SUMMARY_df, DV_dict):

    DV_cost = None
    DV_time = None
    DV_inj = [None,]*4

    for DV_name, DV_mod in DV_dict.items():
        if 'rec_cost' in DV_name:
            DV_cost = DV_mod
        elif 'rec_time' in DV_name:
            DV_time = DV_mod
        elif 'injuries_1' in DV_name:
            DV_inj[0] = DV_mod
        elif 'injuries_2' in DV_name:
            DV_inj[1] = DV_mod
        elif 'injuries_3' in DV_name:
            DV_inj[2] = DV_mod
        elif 'injuries_4' in DV_name:
            DV_inj[3] = DV_mod

    DVs = SUMMARY_df.columns.get_level_values(1)

    if DV_cost is not None:

        comp_types = []
        FG_list = [c for c in DV_cost.columns.get_level_values('FG').unique()]
        for comp_type in ['S', 'NS', 'NSA', 'NSD']:
            if np.sum([fg.startswith(comp_type) for fg in FG_list]) > 0:
                if any([fg.startswith('SE') or fg.startswith('SP') for fg in FG_list]):
                    # additional name check for HAZUS building class tags
                    continue
                else:
                    comp_types.append(comp_type)

        # Hurricane comp_types:
        for comp_type in ['CECB', 'CERB', 'MECB', 'MERB', 'MH', 'MLRI', 'MLRM',
                          'MMUH', 'MSF', 'SECB', 'SERB', 'SPMB', 'WMUH', 'WSF']:
            if np.sum([fg.startswith(comp_type) for fg in FG_list]) > 0:
                comp_types.append('Wind')
        for comp_type in ['fl']:
            if np.sum([fg.startswith(comp_type) for fg in FG_list]) > 0:
                comp_types.append('Flood')

        repl_cost = GI['replacement_cost']

        headers = [['Repair Cost',],
                   ['aggregate',],
                   [' ',],
                   ['mean','std','10%','median','90%']]

        MI = pd.MultiIndex.from_product(headers,
                                        names=['DV', 'comp_type', 'DSG_DS', 'stat'])

        df_res_Cagg = pd.DataFrame(columns=MI, index=[0, ])
        df_res_Cagg.fillna(0, inplace=True)

        headers = [['Repair Impractical',],
                   ['probability',],
                   [' ',],
                   [' ',]]

        MI = pd.MultiIndex.from_product(headers,
                                        names=['DV', 'comp_type', 'DSG_DS', 'stat'])

        df_res_Cimp = pd.DataFrame(columns=MI, index=[0, ])
        df_res_Cimp[('Repair Impractical', 'probability')] = SUMMARY_df[('reconstruction', 'cost impractical')].mean()
        df_res_Cimp = df_res_Cimp.astype(float)

        headers = [['Repair Cost',],
                   comp_types,
                   ['aggregate','1_1', '2_1', '3_1', '4_1', '4_2'],
                   ['mean',]]

        MI = pd.MultiIndex.from_product(headers,
                                        names=['DV', 'comp_type', 'DSG_DS', 'stat'])

        df_res_C = pd.DataFrame(columns=MI, index=[0, ])

        for comp_type in ['NSA', 'NSD', 'NS']:
            if comp_type in comp_types:
                del df_res_C[('Repair Cost', comp_type, '4_2')]

        # Hurricane comp_types
        for comp_type in ['Wind', 'Flood']:
            if comp_type in comp_types:
                del df_res_C[('Repair Cost', comp_type, '4_2')]

    if DV_time is not None:

        repl_time = GI['replacement_time']

        headers = [['Repair Time',],
                   [' ',],
                   ['aggregate',],
                   ['mean','std','10%','median','90%']]

        MI = pd.MultiIndex.from_product(headers,
                                        names=['DV', 'comp_type', 'DSG_DS', 'stat'])

        df_res_Tagg = pd.DataFrame(columns=MI, index=[0, ])
        df_res_Tagg.fillna(0, inplace=True)

    if DV_inj[0] is not None:

        lvls = []
        [lvls.append(f'sev{i+1}') for i in range(4) if DV_inj[i] is not None]

        headers = [['Injuries',],
                   lvls,
                   ['aggregate',],
                   ['mean','std','10%','median','90%']]

        MI = pd.MultiIndex.from_product(headers,
                                        names=['DV', 'comp_type', 'DSG_DS', 'stat'])

        df_res_Iagg = pd.DataFrame(columns=MI, index=[0, ])
        df_res_Iagg.fillna(0, inplace=True)

    dfs_to_join = []

    # start with the disaggregated costs...
    if DV_cost is not None:
        for type_ID in comp_types:

            DV_res = DV_cost.groupby(level=['FG', 'DSG_DS'], axis=1).sum()

            type_cols = [c for c in DV_res.columns.get_level_values('FG').unique() if c.startswith(type_ID)]

            # Hurricane comp_types:
            if type_ID == 'Wind':
                for k in  ['CECB', 'CERB', 'MECB', 'MERB', 'MH', 'MLRI', 'MLRM',
                           'MMUH', 'MSF', 'SECB', 'SERB', 'SPMB', 'WMUH', 'WSF']:
                    try:
                        type_cols = [c for c in DV_res.columns.get_level_values('FG').unique() if c.startswith(k)]
                        if type_cols:
                            break
                    except:
                        print('Cannot find fragility ID for the wind loss type.')
            elif type_ID == 'Flood':
                type_cols = [c for c in DV_res.columns.get_level_values('FG').unique() \
                             if c.startswith('fl')]

            df_cost = DV_res.loc[:, type_cols].groupby(level='DSG_DS',axis=1).sum()

            # create a df with 1s at cells with damage and identify the governing DS
            df_sel = df_cost.copy()
            df_sel[df_sel>0.0] = df_sel[df_sel>0.0] / df_sel[df_sel>0.0]

            cols = df_sel.columns
            for i in range(len(cols)):
                filter = np.where(df_sel.iloc[:,i].values > 0.0)[0]
                df_sel.iloc[filter,idx[0:i]] = 1.0
            df_sel = df_sel.sum(axis=1)

            if type_ID == 'S':
                ds_list = ['1_1', '2_1', '3_1', '4_1', '4_2']
            else:
                ds_list = ['1_1', '2_1', '3_1', '4_1']

            # Hurricane ds_list
            if type_ID == 'Wind':
                ds_list = ['1_1', '2_1', '3_1', '4_1']
            elif type_ID == 'Flood':
                # flood ds (Hazus does not have ds, ds here just indicates PIH)
                ds_list = ['10_1', '11_1', '12_1', '13_1', '14_1', '15_1', '16_1',
                           '17_1', '18_1', '19_1', '1_1', '2_1', '3_1', '4_1',
                           '5_1', '6_1', '7_1', '8_1', '9_1']

            # store the results in the output DF
            df_cost = df_cost.sum(axis=1)
            df_cost.loc[:] = np.minimum(df_cost.values, repl_cost)
            mean_costs = [df_cost.loc[df_sel == dsg_i+1].mean() for dsg_i, dsg in enumerate(ds_list)]

            #df_res_C.loc[:, idx['Repair Cost', type_ID, ds_list, 'mean']] = mean_costs
            # Looping to fill the mean_costs (for various damage state numbers)
            for cur_ds in ds_list:
                df_res_C.loc[:, idx['Repair Cost', type_ID, cur_ds, 'mean']] = mean_costs[ds_list.index(cur_ds)]
            df_res_C.loc[:, idx['Repair Cost', type_ID, 'aggregate', 'mean']] = df_cost.mean()

            df_res_C = df_res_C.astype(float) #.round(0)

        # now store the aggregate results for cost
        DV_res = describe(SUMMARY_df[('reconstruction','cost')])

        df_res_Cagg.loc[:, idx['Repair Cost', 'aggregate', ' ', ['mean', 'std','10%','median','90%']]] = DV_res[['mean', 'std','10%','50%','90%']].values

        df_res_Cagg = df_res_Cagg.astype(float) #.round(0)
        dfs_to_join = dfs_to_join + [df_res_Cagg, df_res_Cimp, df_res_C]

    if DV_time is not None:
        DV_res = describe(SUMMARY_df[('reconstruction','time')])

        df_res_Tagg.loc[:, idx['Repair Time', ' ', 'aggregate', ['mean', 'std','10%','median','90%']]] = DV_res[['mean', 'std','10%','50%','90%']].values

        df_res_Tagg = df_res_Tagg.astype(float) #.round(1)
        dfs_to_join.append(df_res_Tagg)

    if DV_inj[0] is not None:
        for i in range(4):
            if DV_inj[i] is not None:
                DV_res = describe(SUMMARY_df[('injuries',f'sev{i+1}')])

                df_res_Iagg.loc[:, idx['Injuries', f'sev{i+1}', 'aggregate', ['mean', 'std','10%','median','90%']]] = DV_res[['mean', 'std','10%','50%','90%']].values

                df_res_Iagg = df_res_Iagg.astype(float) #.round(6)

        dfs_to_join.append(df_res_Iagg)

    df_res = pd.concat(dfs_to_join,axis=1)

    # save the output
    with open(posixpath.join(output_dir, DV_filename), 'w') as f:
        df_res.to_csv(f)

def write_SimCenter_DV_output_old(output_dir, DV_filename, DV_df, DV_name):

    DV_name = convert_dv_name[DV_name]

    DV_file_path = posixpath.join(output_dir, DV_filename)

    try:
        with open(DV_file_path, 'r') as f:
            DV = json.load(f)
    except:
        DV = {}

    DV.update({DV_name: {}})

    DV_i = DV[DV_name]

    try:
    #if True:
        #DV_tot = DV_df.sum(axis=1).describe([0.1,0.5,0.9]).drop('count')
        DV_tot = describe(np.sum(DV_df.values, axis=1))
        DV_i.update({'total':{}})
        for stat in DV_tot.index:
            DV_i['total'].update({stat: DV_tot.loc[stat]})

        #DV_stats = DV_df.describe([0.1,0.5,0.9]).drop('count')
        DV_stats = describe(DV_df)
        for FG in sorted(DV_stats.columns.get_level_values('FG').unique()):
            DV_i.update({str(FG):{}})

            for PG in sorted(
                DV_stats.loc[:,idx[FG]].columns.get_level_values('PG').unique()):
                DV_i[str(FG)].update({str(PG):{}})

                for DS in sorted(
                    DV_stats.loc[:,idx[FG, PG]].columns.get_level_values('DSG_DS').unique()):
                    DV_i[str(FG)][str(PG)].update({str(DS): {}})
                    DV_stats_i = DV_stats.loc[:,(FG,PG,DS)]
                    for stat in DV_stats_i.index:
                        DV_i[str(FG)][str(PG)][str(DS)].update({
                            stat: DV_stats_i.loc[stat]})
    except:
        pass

    log_msg('\t\t\tSaving file {}'.format(DV_filename))
    with open(DV_file_path, 'w') as f:
        json.dump(DV, f, indent = 2)
