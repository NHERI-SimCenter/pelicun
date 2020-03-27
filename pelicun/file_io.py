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
    convert_P58_data_to_json
    create_HAZUS_EQ_json_files
    create_HAZUS_HU_json_files
    write_SimCenter_DL_output
    write_SimCenter_DM_output
    write_SimCenter_DV_output

"""

from .base import *

import json, csv, posixpath
import xml.etree.ElementTree as ET
from distutils.util import strtobool
from copy import deepcopy

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

    log_msg('\t\tOpening the json file...')
    with open(input_path, 'r') as f:
        jd = json.load(f)

    # get the data required for DL
    data = dict([(label, dict()) for label in [
        'general', 'units', 'unit_names', 'components', 'collapse_modes',
        'decision_variables', 'dependencies', 'data_sources',
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
        irrep_res_drift = damage.get('IrrepairableResidualDrift', None)
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
        'P58'     : 'FEMA P58 first edition',
        'HAZUS_EQ': 'HAZUS MH 2.1 earthquake',
        'HAZUS_HU': 'HAZUS MH 2.1 hurricane'
    }

    # check if the user specified custom data sources
    path_CMP_data = DL_input.get("ComponentDataFolder", "")

    if path_CMP_data == "":
        path_CMP_data = pelicun_path
        if AT == 'P58':
            path_CMP_data += '/resources/FEMA P58 first edition/DL json/'
        elif AT == 'HAZUS_EQ':
            path_CMP_data += '/resources/HAZUS MH 2.1 earthquake/DL json/'
        elif AT == 'HAZUS_HU':
            path_CMP_data += '/resources/HAZUS MH 2.1 hurricane/DL json/'
    data['data_sources'].update({'path_CMP_data': path_CMP_data})

    # The population data is only needed if we are interested in injuries
    if inhabitants is not None:
        path_POP_data = inhabitants.get("PopulationDataFile", "")
    else:
        path_POP_data = ""

    if data['decision_variables']['injuries']:
        if path_POP_data == "":
            path_POP_data = pelicun_path
            if AT == 'P58':
                path_POP_data += '/resources/FEMA P58 first edition/population.json'
            elif AT == 'HAZUS_EQ':
                path_POP_data += '/resources/HAZUS MH 2.1 earthquake/population.json'
        data['data_sources'].update({'path_POP_data': path_POP_data})

    # general information
    GI = jd.get("GeneralInformation", None)
    if GI is None:
        GI = jd.get("GI", None)

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
        data['unit_names'].update({
            'force':        'N',
            'length':       'm',
            'area':         'm2',
            'volume':       'm3',
            'speed':        'mps',
            'acceleration': 'mps2',
        })

    for unit_type, unit_name in data['unit_names'].items():
        data['units'].update({unit_type: globals()[unit_name]})

    # other attributes that can be used by a P58 assessment
    if AT == 'P58':
        for target_att, source_att, f_conv, unit_kind, dv_req in [
            ['plan_area', 'planArea', float, 'area', 'injuries'],
            ['stories', 'stories', int, '', 'all'],
            # The following lines are commented out for now, because we do not
            # use these pieces of data anyway.
            #['building_type', 'type', str, ''],
            #['height', 'height', float, 'length'],
            #['year_built', 'year', int, ''],
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
        data['general'].update({'stories': int(GI['stories'])})

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
                    comp_data['csg_weights'][c_base] = list(np.array(comp_data['csg_weights'][c_base]) * comp_data['quantities'][c_base])
                    for ci in combo_ids[1:]:
                        comp_data['quantities'][c_base] += comp_data['quantities'][ci]
                        comp_data['csg_weights'][c_base] += list(np.array(comp_data['csg_weights'][ci]) * comp_data['quantities'][ci])
                    comp_data['csg_weights'][c_base] = list(np.array(comp_data['csg_weights'][c_base]) / comp_data['quantities'][c_base])

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

                # some basic pre-processing
                # sort the dirs and their weights to have better structured
                # matrices later
                #dir_order = np.argsort(comp_data['directions'])
                #comp_data['directions'] = [comp_data['directions'][d_i] for d_i
                #                     in dir_order]

                # get the location(s) of components based on non-zero quantities
                #comp_data.update({
                #    'locations': (np.where(comp_data['quantities'] > 0.)[
                #                      0] + 1).tolist()
                #})
                # remove the zeros from the quantities
                #nonzero = comp_data['quantities'] > 0.
                #comp_data['quantities'] = comp_data['quantities'][
                #    nonzero].tolist()

                # scale the quantities according to the specified unit

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

    EDP_units = dict(
        # PID, RID, and MID are not here because they are unitless
        PFA = 'acceleration',
        PWS = 'speed'
    )
    if AT in ['P58', 'HAZUS_EQ']:
        EDP_keys = ['PID', 'PFA', 'PGV', 'RID', 'PMD']
    elif AT in ['HAZUS_HU']:
        EDP_keys = ['PWS', ]

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

        # irrepairable drift
        if ((damage is not None) and (irrep_res_drift is not None)):
            data['general'].update({
                'irrepairable_res_drift':
                    dict([(key, float_or_None(value)) for key, value in
                          irrep_res_drift.items()])})
            # TODO: move this in the irrepairable part of general
            yield_drift = irrep_res_drift.get("YieldDriftRatio", None)
            if yield_drift is not None:
                data['general'].update({
                    'yield_drift': float_or_None(yield_drift)})
            elif ((data['decision_variables']['rec_cost']) or
                  (data['decision_variables']['rec_time'])):
                data['general'].update({'yield_drift': 0.01})

        elif ((data['decision_variables']['rec_cost']) or
              (data['decision_variables']['rec_time'])):
            show_warning(
                "Residual drift limits corresponding to irrepairable "
                "damage were not defined in the input file. We assume that "
                "damage is repairable regardless of the residual drift.")
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

            # peak population
            peak_pop = inhabitants.get("PeakPopulation", None)
            if peak_pop is not None:
                peak_pop = [float_or_None(pop) for pop in peak_pop.split(',')]

                # If the number of stories is specified...
                if 'stories' in data['general'].keys():
                    stories = data['general']['stories']
                    pop_in = len(peak_pop)

                    # and the population list does not provide values
                    # for every story:
                    for s in range(pop_in, stories):
                        # If only one value is provided, then it is assumed to
                        # be the population on every story.
                        if pop_in == 1:
                            peak_pop.append(peak_pop[0])

                        # Otherwise, the values are assumed to correspond to
                        # the bottom stories and the upper ones are filled with
                        # zeros. A warning message is displayed in this case.
                        else:
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
        elif dv_req == '' or data['decision_variables'][dv_req]:
            if target_att != 'fragilities':
                data['dependencies'].update({target_att: 'IND'})
            else:
                data['dependencies'].update({target_att: 'ATC'})

            show_warning(
                "Correlation between {} was not ".format(source_att)+
                "defined in the input file. Using default values.")

    if ((depends is not None) and ('CostAndTime' in depends.keys())):
        data['dependencies'].update({
            'cost_and_time': bool(depends['CostAndTime'])})
    elif ((data['decision_variables']['rec_cost']) or
          (data['decision_variables']['rec_time'])):
        data['dependencies'].update({'cost_and_time': False})
        show_warning(
            "Correlation between reconstruction cost and time was not "
            "defined in the input file. Using default values.")

    if ((depends is not None) and ('InjurySeverities' in depends.keys())):
        data['dependencies'].update({
            'injury_lvls': bool(depends['InjurySeverities'])})
    elif data['decision_variables']['injuries']:
        data['dependencies'].update({'injury_lvls': False})
        show_warning(
            "Correlation between injury levels was not defined in the "
            "input file. Using default values.")

    if verbose: pp.pprint(data)

    return data

def read_SimCenter_EDP_input(input_path, EDP_kinds=('PID', 'PFA'),
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
    log_msg('\t\tOpening the input file...')
    # If the file name ends with csv, we assume a standard csv file
    if input_path[-3:] == 'csv':
        EDP_raw = pd.read_csv(input_path, header=0, index_col=0)

    # otherwise, we assume that a dakota file is provided...
    else:
        # the read_csv method in pandas is sufficiently versatile to handle the
        # tabular format of dakota
        EDP_raw = pd.read_csv(input_path, sep=r'\s+', header=0, index_col=0)
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
    with open(path_POP, 'r') as f:
        jd = json.load(f)

    AT = assessment_type

    # Convert the HAZUS occupancy classes to the broader categories used for
    # population distribution.
    if AT == 'HAZUS_EQ':
        base_occupancy = occupancy
        if base_occupancy[:3] == "RES":
            occupancy = "Residential"
        elif base_occupancy[:3] in ["COM", "REL"]:
            occupancy = "Commercial"
        elif base_occupancy[:3] == "EDU":
            occupancy = "Educational"
        elif base_occupancy[:3] in ["IND", "AGR"]:
            occupancy = "Industrial"
        else:
            warnings.warn(UserWarning(
                'Unknown, probably invalid, occupancy class for HAZUS '
                'assessment: {}. When defining population distribution, '
                'assuming RES1 instead.'.format(base_occupancy)))
            occupancy = 'Residential'

    data = jd[occupancy]

    # convert peak population to persons/m2
    if 'peak' in data.keys():
        data['peak'] = data['peak'] / (1000. * ft2)

    if verbose: # pragma: no cover
        pp.pprint(data)

    return data


def read_component_DL_data(path_CMP, comp_info, assessment_type='P58',
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

        # replace N/A distribution with normal and negligible cov
        c_data['cov'] = [0.0001 if dk == 'N/A' else cov
                         for cov,dk in list(zip(c_data['cov'],
                                                c_data['distribution_kind']))]
        c_data['distribution_kind'] = ['normal' if dk == 'N/A' else dk
                                       for dk in c_data['distribution_kind']]
        c_data['cov'] = [0.0001 if cov==None else cov
                         for cov in c_data['cov']]

        #c_data['kind'] = ci_data['kind']
        #c_data['unit'] = ci_data['unit'][0] * globals()[ci_data['unit'][1]]
        #c_data['quantities'] = (np.asarray(ci_data['quantities']) * c_data[
        #    'unit']).tolist()

        # calculate the quantity weights in each direction
        #dirs = np.asarray(c_data['directions'], dtype=np.int)
        #u_dirs = np.unique(dirs)
        #weights = np.asarray(c_data['csg_weights'])
        #c_data['dir_weights'] = [sum(weights[np.where(dirs == d_i)])
        #                         for d_i in u_dirs]

        c_data['ID'] = c_id
        c_data['name'] = DL_data['Name']
        c_data['description'] = DL_GI['Description']
        c_data['offset'] =DL_EDP.get('Offset', 0)
        c_data['correlation'] = int(DL_data.get('Correlated', False))
        c_data['directional'] = int(DL_data.get('Directional', False))

        EDP_type = DL_EDP['Type']
        demand_factor = 1.0
        if EDP_type == 'Story Drift Ratio':
            demand_type = 'PID'
        elif EDP_type == 'Peak Floor Acceleration':
            demand_type = 'PFA'
            demand_factor = g
            # PFA corresponds to the top of the given story. The ground floor
            # has an idex of 0. When damage of acceleration-sensitive components
            # is controlled by the acceleration of the bottom of the story, the
            # corresponding PFA location needs to be reduced by 1. Since FEMA
            # P58 assumes that PFA corresponds to the bottom of the given story
            # by default, we need to subtract 1 from the location values in a
            # FEMA P58 assessment. Rather than changing the locations themselves,
            # we assign an offset so that the results still get collected at the
            # appropriate story.
            if AT == 'P58':
                c_data['offset'] = c_data['offset'] - 1
        elif EDP_type == 'Peak Gust Wind Speed':
            demand_type = 'PWS'
            demand_factor = mph
        elif EDP_type == 'Peak Ground Velocity':
            demand_type = 'PGV'
            demand_factor = cmps
        elif EDP_type == 'Mega Drift Ratio':
            demand_type = 'PMD'
        elif EDP_type == 'Residual Drift Ratio':
            demand_type = 'RID'
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
        QNT_unit = DL_data.get('QuantityUnit', [1, 'ea'])
        data_unit = QNT_unit[0] * globals()[QNT_unit[1]]
        for DSG_id, DSG_i in enumerate(DL_DSG):
            DSG_data = dict(
                theta=DSG_i['MedianEDP'] * demand_factor,
                sig=DSG_i['Beta'],
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

                DS_C = DS_i['Consequences']
                if 'ReconstructionCost' in DS_C.keys():
                    DS_CC = DS_C['ReconstructionCost']
                    if isinstance(DS_CC['Amount'], list):
                        DS_data.update({'repair_cost': {
                            'median_max'       : DS_CC['Amount'][0],
                            'median_min'       : DS_CC['Amount'][1],
                            'quantity_lower'   : DS_CC['Quantity'][0],
                            'quantity_upper'   : DS_CC['Quantity'][1],
                            'distribution_kind': curve_type[DS_CC['CurveType']],
                            'cov'              : DS_CC['Beta'],
                        }})

                        # convert the units to standard ones
                        DS_data['repair_cost']['quantity_lower'] *= data_unit
                        DS_data['repair_cost']['quantity_upper'] *= data_unit
                        DS_data['repair_cost']['median_min'] /= data_unit
                        DS_data['repair_cost']['median_max'] /= data_unit
                    else:
                        DS_data.update({'repair_cost': DS_CC['Amount']})

                if 'ReconstructionTime' in DS_C.keys():
                    DS_CT = DS_C['ReconstructionTime']
                    if isinstance(DS_CT['Amount'], list):
                        DS_data.update({'repair_time': {
                            'median_max'       : DS_CT['Amount'][0],
                            'median_min'       : DS_CT['Amount'][1],
                            'quantity_lower'   : DS_CT['Quantity'][0],
                            'quantity_upper'   : DS_CT['Quantity'][1],
                            'distribution_kind': curve_type[DS_CT['CurveType']],
                            'cov'              : DS_CT['Beta'],
                        }})

                        # convert the units to standard ones
                        DS_data['repair_time']['quantity_lower'] *= data_unit
                        DS_data['repair_time']['quantity_upper'] *= data_unit
                        DS_data['repair_time']['median_min'] /= data_unit
                        DS_data['repair_time']['median_max'] /= data_unit
                    else:
                        DS_data.update({'repair_time': DS_CT['Amount']})

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
                            'theta': [I_i['Amount'] for I_i in DS_CI],
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
        os.path.join(data_dir, 'PACT_fragility_data.xlsx'))
    ID_list = src_df['NISTIR Classification']

    XML_list = [f for f in os.listdir(data_dir+'DL xml/') if f.endswith('.xml')]

    incomplete_count = 0

    for filename in XML_list:

        comp_ID = filename[:-4]

        #try:
        if True:
            tree = ET.parse(os.path.join(data_dir+'DL xml/', comp_ID + '.xml'))
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
                if QU[0] in ['TN', 'AP', 'CF', 'KV']:
                    QU[0] = 'ea'
                    QU[1] = 1
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

                    # avoid having NaN as repair measures
                    repair_measures = row['DS {}, Repair Description'.format(DS[-1])]
                    if not isinstance(repair_measures, str):
                        repair_measures = ""

                    DSG['DamageStates'].append({
                        'Weight'        :
                            float(row['DS {}, Probability'.format(DS[-1])]),
                        'LongLeadTime'  :
                            row['DS {}, Long Lead Time'.format(DS[-1])] in [
                                'YES'],
                        'Consequences'  : {},
                        'Description'   :
                            row['DS {}, Description'.format(DS[-1])],
                        'RepairMeasures': repair_measures
                    })

                    IMG = row['DS{}, Illustrations'.format(DS[-1])]
                    if IMG not in ['none', np.nan]:
                        DSG['DamageStates'][-1].update({'DamageImageName': IMG})

                    AA = row['DS {} - Casualty Affected Area'.format(DS[-1])]
                    if (isinstance(AA, str) and (is_float(AA.split(' ')[0]))):
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
                json.dump(json_output, f, indent=2)

        #except:
        #    warnings.warn(UserWarning(
        #        'Error converting data for component {}'.format(comp_ID)))

def create_HAZUS_EQ_json_files(data_dir, target_dir):
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
        raw_data['Structural_Fragility_Groups']['P_collapse'].keys())
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
                        'Directional': True,
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
                        json.dump(json_output, f, indent=2)

            # second, nonstructural acceleration sensitive fragility groups
            json_output = {}

            dl_id = 'NSA-{}-{}'.format(convert_design_level[dl], ot)

            # this might get replaced by a more descriptive name in the future
            json_output.update({'Name': dl_id})

            # General info
            json_output.update({
                'Directional': False,
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
                            'ReconstructionCost': {'Amount': NSA_data[
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
                json.dump(json_output, f, indent=2)

                # third, nonstructural drift sensitive fragility groups
        json_output = {}

        dl_id = 'NSD-{}'.format(ot)

        # this might get replaced by a more descriptive name in the future
        json_output.update({'Name': dl_id})

        # General info
        json_output.update({
            'Directional': True,
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
                        'ReconstructionCost': {
                            'Amount': NSD_data['Reconstruction_cost'][ot][
                                          dsg_i] / 100.}
                    },
                    'Description' : convert_DS_description[
                        'DS{}'.format(dsg_i + 1)]
                }]
            })

        with open(os.path.join(target_dir + 'DL json/', dl_id + '.json'),
                  'w') as f:
            json.dump(json_output, f, indent=2)

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
        json.dump(pop_output, f, indent=2)

def create_HAZUS_HU_json_files(data_dir, target_dir):
    """
    Create JSON data files from publicly available HAZUS data.

    HAZUS damage and loss information is publicly available in the technical
    manuals and the HAZUS software tool. The relevant data have been collected
    in a series of Excel files (e.g., hu_Wood.xlsx) that are stored in the
    'resources/HAZUS MH 2.1 hurricane' folder in the pelicun repo. Here we read
    that file (or a file of similar format) and produce damage and loss data
    for Fragility Groups in the common SimCenter JSON format.

    The HAZUS hurricane methodology handles damage and losses at the assembly
    level. In this implementation each building is represented by one Fragility
    Group that describes the damage states and their consequences in a FEMA
    P58-like framework but using the data from the HAZUS Technical Manual.

    Note: HAZUS calculates lossess independently of damage using peak wind gust
    speed as a controlling variable. We fitted a model to the curves in HAZUS
    that assigns losses to each damage state and determines losses as a function
    of building damage. Results shall be in good agreement with those of HAZUS
    for the majority of building configurations. Exceptions and more details
    are provided in the ... section of the documentation.

    Parameters
    ----------
    data_dir: string
        Path to the folder with the hazus_data_eq JSON file.
    target_dir: string
        Path to the folder where the results shall be saved. The population
        distribution file will be saved here, the DL JSON files will be saved
        to a 'DL json' subfolder.

    """

    # open the raw HAZUS data
    df_wood = pd.read_excel(os.path.join(data_dir, 'hu_Wood.xlsx'))

    # some general formatting to make file name generation easier
    df_wood['shutters'] = df_wood['shutters'].astype(int)
    df_wood['terr_rough'] = (df_wood['terr_rough'] * 100.).astype(int)

    convert_building_type = {
        'WSF1' : 'Wood Single-Family Homes 1 story',
        'WSF2' : 'Wood Single-Family Homes 2+ stories',
        'WMUH1': 'Wood Multi-Unit or Hotel or Motel 1 story',
        'WMUH2': 'Wood Multi-Unit or Hotel or Motel 2 stories',
        'WMUH3': 'Wood Multi-Unit or Hotel or Motel 3+ stories',
    }

    convert_bldg_char_names = {
        'roof_shape'     : 'Roof Shape',
        'sec_water_res'  : 'Secondary Water Resistance',
        'roof_deck_attch': 'Roof Deck Attachment',
        'roof_wall_conn' : 'Roof-Wall Connection',
        'garage'         : 'Garage',
        'shutters'       : 'Shutters',
        'roof_cover'     : 'Roof Cover Type',
        'roof_quality'   : 'Roof Cover Quality',
        'terr_rough'     : 'Terrain',
    }

    convert_bldg_chars = {
        1      : True,
        0      : False,

        'gab'  : 'gable',
        'hip'  : 'hip',
        'flt'  : 'flat',

        '6d'   : '6d @ 6"/12"',
        '8d'   : '8d @ 6"/12"',
        '6s'   : '6d/8d mix @ 6"/6"',
        '8s'   : '8D @ 6"/6"',

        'tnail': 'Toe-nail',
        'strap': 'Strap',

        'no'   : 'None',
        'wkd'  : 'Weak',
        'std'  : 'Standard',
        'sup'  : 'SFBC 1994',

        'bur'  : 'BUR',
        'spm'  : 'SPM',

        'god'  : 'Good',
        'por'  : 'Poor',

        3      : 'Open',
        15     : 'Light Suburban',
        35     : 'Suburban',
        70     : 'Light Trees',
        100    : 'Trees',
    }

    convert_dist = {
        'normal'   : 'Normal',
        'lognormal': 'LogNormal',
    }

    convert_ds = {
        1: 'Minor',
        2: 'Moderate',
        3: 'Severe',
        4: 'Destruction',
    }

    for index, row in df_wood.iterrows():
        #print(index, end=' ')

        json_output = {}

        # define the name of the building damage and loss configuration
        bldg_type = row["bldg_type"]

        if bldg_type[:3] == "WSF":
            cols_of_interest = ["bldg_type", "roof_shape", "sec_water_res",
                                "roof_deck_attch", "roof_wall_conn", "garage",
                                "shutters", "terr_rough"]
        elif bldg_type[:4] == "WMUH":
            cols_of_interest = ["bldg_type", "roof_shape", "roof_cover",
                                "roof_quality", "sec_water_res",
                                "roof_deck_attch", "roof_wall_conn", "shutters",
                                "terr_rough"]

        bldg_chars = row[cols_of_interest]

        if np.isnan(bldg_chars["sec_water_res"]):
            bldg_chars["sec_water_res"] = 'null'
        else:
            bldg_chars["sec_water_res"] = int(bldg_chars["sec_water_res"])

        if bldg_type[:4] == "WMUH":
            if (not isinstance(bldg_chars["roof_cover"],str)
                and np.isnan(bldg_chars["roof_cover"])):
                bldg_chars["roof_cover"] = 'null'
            if (not isinstance(bldg_chars["roof_quality"], str)
                and np.isnan(bldg_chars["roof_quality"])):
                bldg_chars["roof_quality"] = 'null'

        dl_id = "_".join(bldg_chars.astype(str))

        json_output.update({'Name': dl_id})

        # general information
        json_output.update({
            'GeneralInformation': {
                'ID'           : index,
                'Description'  : dl_id,
                'Building type': convert_building_type[bldg_type],
            }
        })
        for col in cols_of_interest:
            if (col != 'bldg_type') and (bldg_chars[col] != 'null'):
                json_output['GeneralInformation'].update({
                    convert_bldg_char_names[col]: convert_bldg_chars[
                        bldg_chars[col]]
                })

        # EDP info
        json_output.update({
            'EDP': {
                'Type': 'Peak Gust Wind Speed',
                'Unit': [1, 'mph']
            }
        })

        # Damage States
        json_output.update({'DSGroups': []})

        for dsg_i in range(1, 5):
            json_output['DSGroups'].append({
                'MedianEDP'   : row['DS{}_mu'.format(dsg_i)],
                'Beta'        : row['DS{}_sig'.format(dsg_i)],
                'CurveType'   : convert_dist[row['DS{}_dist'.format(dsg_i)]],
                'DSGroupType' : 'Single',
                'DamageStates': [{
                    'Weight'      : 1.0,
                    'Consequences': {
                        'ReconstructionCost': {
                            'Amount': row[
                                'L{}'.format(dsg_i)] if dsg_i < 4 else 1.0
                        }
                    },
                    'Description' : convert_ds[dsg_i]
                }]
            })

        with open(os.path.join(target_dir + '/DL json/', dl_id + '.json'),
                  'w') as f:
            json.dump(json_output, f, indent=2)

def write_SimCenter_DL_output(output_dir, output_filename, output_df, index_name='#Num',
                              collapse_columns = True, stats_only=False):

    # if the summary flag is set, then not all realizations are returned, but
    # only the first two moments and the empirical CDF through 100 percentiles
    if stats_only:
        #output_df = output_df.describe(np.arange(1, 100)/100.)
        #output_df = output_df.describe([0.1,0.5,0.9])
        output_df = describe(output_df)
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

def write_SimCenter_DM_output(output_dir, DM_filename, DMG_df):

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

    with open(posixpath.join(output_dir, DM_filename), 'w') as f:
        json.dump(DM, f, indent = 2)

def write_SimCenter_DV_output(output_dir, DV_filename, DV_df, DV_name):

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

    with open(DV_file_path, 'w') as f:
        json.dump(DV, f, indent = 2)