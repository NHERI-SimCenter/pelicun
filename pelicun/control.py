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
This module has classes and methods that control the loss assessment.

.. rubric:: Contents

.. autosummary::

       

"""

from .base import *
from .uq import *
from .model import *
from .file_io import *

class Assessment(object):
    """
    Description
    
    """
    
    def __init__(self, realizations):
        
        # assessment settings
        self._realizations = realizations 
        
        # initialize the basic data containers      
        # inputs
        self._AIM_in = None
        self._EDP_in = None
        self._POP_in = None
        self._FG_in = None
        
        # random variables and loss model
        self._RV_dict = None # dictionary to store random variables
        self._EDP_dict = None
        self._FG_dict = None
        
        # results
        self._TIME = None
        self._POP = None
        self._COL = None
        self._ID_dict = None
        self._DMG = None
        self._DV_dict = None
        self._SUMMARY = None
        
    def read_inputs(self, path_DL_input, path_EDP_input):
        
        # read SimCenter inputs -----------------------------------------------
        # BIM file
        self._AIM_in = read_SimCenter_DL_input(path_DL_input, verbose=False)

        # EDP file
        self._EDP_in = read_SimCenter_EDP_input(path_EDP_input, verbose=False)
    
    def define_random_variables(self):
        pass
    
    def define_loss_model(self):
        pass
    
    def calculate_damage(self):
        pass
    
    def calculate_losses(self):
        pass
    
    def write_outputs(self):
        pass
    
class FEMA_P58_Assessment(Assessment):
    """
    Description
    """
    def __init__(self, realizations,
                 beta_m, beta_gm,
                 f_secondary_EDP = 0.7, inj_lvls = 2):
        super().__init__(realizations)
        
        # assessment settings
        self._beta_m = beta_m
        self._beta_gm = beta_gm
        
        # constants for the FEMA-P58 methodology
        self._f_secondary_EDP = f_secondary_EDP
        self._inj_lvls = inj_lvls
        
    @property
    def beta_additional(self):
        return np.sqrt(self._beta_m ** 2. + self._beta_gm ** 2.)

    def read_inputs(self, path_DL_input, path_EDP_input, 
                    path_CMP_data=None, path_POP_data=None):
        super().read_inputs(path_DL_input, path_EDP_input)
        
        # check if the component data path is provided by the user
        if path_CMP_data is None:
            raise ValueError(
                "You need to specify the path to the component data files."
            )

        # assume that the asset is a building
        # TODO: If we want to apply FEMA-P58 to non-building assets, several parts of this methodology need to be extended.
        BIM = self._AIM_in

        # read component and population data ----------------------------------
        # components
        self._FG_in = read_P58_component_data(path_CMP_data,
                                              BIM['components'], verbose=False)
        # TODO: BIM['components'] is component info now
        
        if path_POP_data is not None:
            # population
            POP = read_P58_population_distribution(path_POP_data, 
                                                   BIM['occupancy'], 
                                                   verbose=False)
    
            # if the peak population is specified in the BIM, then use that
            # if the population is set to automatic, then calculate it based on the 
            # gross building area
            if ((BIM['population'].size == 1) 
                and (BIM['population'] == 'auto')):
                POP['peak'] = [POP['peak'] * BIM['area'] 
                               for s in range(BIM['stories'])]
            else:
                POP['peak'] = BIM['population']          
            self._POP_in = POP

    def define_random_variables(self):
        super().define_random_variables()

        # create the random variables -----------------------------------------
        rho_dict = self._AIM_in['correlations']

        # quantities 100
        self._RV_dict.update({'QNT': 
                            self._create_RV_quantities(rho_dict['quantity'])})

        # fragilities 300
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            self._RV_dict.update({'FR-' + c_name: 
                            self._create_RV_fragilities(c_id, comp, 
                                                        rho_dict['fragility'])})

        # consequences 400
        self._RV_dict.update({'DV_RED': 
                            self._create_RV_red_tags(rho_dict['red_tag'])})
        self._RV_dict.update({'DV_REP': 
                            self._create_RV_repairs(rho_dict['repair'])})
        self._RV_dict.update({'DV_INJ': 
                            self._create_RV_injuries(rho_dict['injury'])})

        # demands 200
        self._RV_dict.update({'EDP': self._create_RV_demands(
            self.beta_additional)})

        # sample the random variables -----------------------------------------
        for r_i, rv in self._RV_dict.items():
            rv.sample_distribution(self._realizations)

    def define_loss_model(self):
        super().define_loss_model()

        # fragility groups
        self._FG_dict = self._create_fragility_groups()

        # demands
        self._EDP_dict = dict(
            [(tag, RandomVariableSubset(self._RV_dict['EDP'],tags=tag))
             for tag in self._RV_dict['EDP']._dimension_tags])
        
    def calculate_damage(self):
        super().calculate_damage()

        # event time - month, weekday, and hour realizations
        self._TIME = self._sample_event_time()
        
        # get the population conditioned on event time
        self.POP = self._get_population()

        # collapses
        COL, collapsed_IDs = self._calc_collapses()
        self._ID_dict.update({'collapse':collapsed_IDs})

        # select the non-collapse cases for further analyses
        non_collapsed_IDs = self.POP[
            ~self.POP.index.isin(collapsed_IDs)].index.values.astype(int)
        self._ID_dict.update({'non-collapse': non_collapsed_IDs})

        # damage in non-collapses
        self._DMG = self._calc_damage()

    def calculate_losses(self):
        super().calculate_losses()

        # red tag probability
        DV_RED = self._calc_red_tag()

        # irrepairable cases
        irrepairable_IDs = self._calc_irrepairable()

        # collect the IDs of repairable realizations
        P_NC = self._POP.loc[self._ID_dict['non-collapsed']]
        repairable_IDs = P_NC[
            ~P_NC.index.isin(irrepairable_IDs)].index.values.astype(int)

        self._ID_dict.update({'repairable': repairable_IDs})
        self._ID_dict.update({'irrepairable': irrepairable_IDs})

        # reconstruction cost and time for repairable cases
        DV_COST, DV_TIME = self._calc_repair_cost_and_time()
        
        # injuries due to collapse
        # ------------------------------------------------------------------------------
        COL_INJ = self._calc_collapse_injuries()

        # injuries in non-collapsed cases

        DV_INJ_dict = self._calc_non_collapse_injuries()
        
        # store results
        self.COL = pd.concat([self._COL, COL_INJ], axis=1)

        self._DV_dict.update({
            'red tag': DV_RED,
            'repair cost': DV_COST,
            'repair time': DV_TIME,
            'injuries': DV_INJ_dict
        })
        
    def aggregate_results(self):

        MI_raw = [
            ('event time', 'month'),
            ('event time', 'weekday?'),
            ('event time', 'hour'),
            ('inhabitants', ''),
            ('collapses', 'collapsed?'),
            ('collapses', 'mode'),
            ('red tagged?', ''),
            ('reconstruction', 'irrepairable?'),
            ('reconstruction', 'cost impractical?'),
            ('reconstruction', 'cost'),
            ('reconstruction', 'time impractical?'),
            ('reconstruction', 'time-sequential'),
            ('reconstruction', 'time-parallel'),
            ('injuries', 'casualties'),
            ('injuries', 'fatalities'),
        ]
        
        ncID = self._ID_dict['non-collapse']
        colID = self._ID_dict['collapse']
        repID = self._ID_dict['repairable']
        irID = self._ID_dict['irrepairable']

        MI = pd.MultiIndex.from_tuples(MI_raw)

        SUMMARY = pd.DataFrame(np.empty((self._realizations, len(MI))),
                               columns=MI)
        SUMMARY[:] = np.NaN

        # event time
        for prop in ['month', 'weekday?', 'hour']:
            offset = 0
            if prop == 'month':
                offset = 1
            SUMMARY.loc[:, ('event time', prop)] = self._TIME.loc[:, prop] + offset

        # inhabitants
        SUMMARY.loc[:, ('inhabitants', '')] = self._POP.sum(axis=1)

        # collapses
        SUMMARY.loc[:, ('collapses', 'collapsed?')] = self._COL.iloc[:, 0]
        SUMMARY.loc[colID, ('collapses', 'mode')] = self._COL.loc[:, 'CM']

        # red tag
        SUMMARY.loc[ncID, ('red tagged?', '')] = \
            self._DV_dict['red tag'].max(axis=1)

        # reconstruction cost
        SUMMARY.loc[ncID, ('reconstruction', 'cost')] = \
            self._DV_dict['repair cost'].sum(axis=1)

        repl_cost = self._AIM_in['replacement_cost']
        SUMMARY.loc[colID, ('reconstruction', 'cost')] = repl_cost

        SUMMARY.loc[ncID, ('reconstruction', 'irrepairable?')] = 0
        SUMMARY.loc[irID,
                    ('reconstruction', 'irrepairable?')] = 1
        SUMMARY.loc[irID, ('reconstruction', 'cost')] = repl_cost

        repair_impractical_IDs = SUMMARY.loc[
            SUMMARY.loc[:, ('reconstruction', 'cost')] > repl_cost].index
        SUMMARY.loc[repID, ('reconstruction', 'cost impractical?')] = 0
        SUMMARY.loc[repair_impractical_IDs,
                    ('reconstruction', 'cost impractical?')] = 1
        SUMMARY.loc[
            repair_impractical_IDs, ('reconstruction', 'cost')] = repl_cost

        # reconstruction time
        SUMMARY.loc[ncID, ('reconstruction', 'time-sequential')] = \
            self._DV_dict['repair time'].sum(axis=1)
        SUMMARY.loc[ncID, ('reconstruction', 'time-parallel')] = \
            self._DV_dict['repair time'].max(axis=1)

        rep_time = self._AIM_in['replacement_time']

        for t_label in ['time-sequential', 'time-parallel']:
            SUMMARY.loc[colID, ('reconstruction', t_label)] = rep_time
            SUMMARY.loc[irID, ('reconstruction', t_label)] = rep_time

        repair_impractical_IDs = \
            SUMMARY.loc[SUMMARY.loc[:, ('reconstruction',
                                        'time-parallel')] > rep_time].index
        SUMMARY.loc[repID, ('reconstruction', 'time impractical?')] = 0
        SUMMARY.loc[repair_impractical_IDs,('reconstruction', 
                                            'time impractical?')] = 1
        SUMMARY.loc[repair_impractical_IDs, ('reconstruction', 
                                             'time-parallel')] = rep_time

        # injuries
        SUMMARY.loc[colID, ('injuries', 'casualties')] = \
            self._COL.loc[:, 'INJ-0']
        SUMMARY.loc[colID, ('injuries', 'fatalities')] = \
            self._COL.loc[:, 'INJ-1']

        SUMMARY.loc[ncID, ('injuries', 'casualties')] = \
            self._DV_dict['injuries'][0].sum(axis=1)
        SUMMARY.loc[ncID, ('injuries', 'fatalities')] = \
            self._DV_dict['injuries'][1].sum(axis=1)
        
        self._SUMMARY = SUMMARY

    def write_outputs(self):
        super().write_outputs()

    def _create_RV_quantities(self, rho_qnt):

        q_theta, q_sig, q_tag, q_dist = [np.array([]) for i in range(4)]

        # collect the parameters for each quantity dimension
        for c_id, comp in self._FG_in.items():
            q_theta = np.append(q_theta, comp['quantities'])
            if comp['distribution_kind'] == 'normal':
                q_sig = np.append(q_sig, (
                    comp['cov'] * np.asarray(comp['quantities'])).tolist())
            else:
                q_sig = np.append(q_sig, (
                    np.ones(len(comp['locations'])) * comp['cov']).tolist())
            q_tag = np.append(q_tag, [c_id + '-QNT-' + str(s_i) for s_i in
                                      comp['locations']])
            q_dist = np.append(q_dist, [comp['distribution_kind'] for s_i in
                                        comp['locations']])

        dims = len(q_theta)
        rho = np.ones((dims, dims)) * rho_qnt
        np.fill_diagonal(rho, 1.0)
        q_COV = np.outer(q_sig, q_sig) * rho

        # add lower limits to ensure only positive quantities
        # zero is probably too low, and it might make sense to introduce upper 
        # limits as well
        tr_lower = [0. for d in range(dims)]
        tr_upper = [None for d in range(dims)]
        # to avoid truncations affecting other dimensions when rho_QNT is large, 
        # assign a post-truncation correlation structure
        corr_ref = 'post'

        # create a random variable for component quantities in performance groups
        quantity_RV = RandomVariable(ID=100,
                                     dimension_tags=q_tag,
                                     distribution_kind=q_dist,
                                     theta=q_theta,
                                     COV=q_COV,
                                     truncation_limits=[tr_lower, tr_upper],
                                     corr_ref=corr_ref)

        return quantity_RV

    def _create_RV_fragilities(self, c_id, comp, rho_fr):

        # prepare the basic multivariate distribution data for one component subgroup considering all damage states
        d_theta, d_sig, d_tag = [np.array([]) for i in range(3)]

        for d_id, DSG in comp['DSG_set'].items():
            d_theta = np.append(d_theta, DSG['theta'])
            d_sig = np.append(d_sig, DSG['sig'])
            d_tag = np.append(d_tag, comp['ID'] + '-' + str(d_id))
        dims = len(d_theta)

        # get the total number of random variables for this fragility group
        rv_count = len(comp['locations']) * len(comp['proportions']) * dims

        # create the (empty) input arrays for the RV
        c_theta = np.zeros(rv_count)
        c_tag = np.empty(rv_count, dtype=object)
        c_sig = np.zeros(rv_count)
        c_rho = np.ones(
            (rv_count, rv_count)) * rho_fr['PG']  # set correlations between PGs

        pos_id = 0
        PG_size = len(comp['proportions']) * dims
        for l_id in comp['locations']:
            # for each performance group (i.e. location)
            # correlation between subgroups is controlled by the component data
            if rho_fr['CSG'] is not None:
                rho = rho_fr['CSG']
            else:
                rho = comp['correlation']
            c_rho[pos_id:pos_id + PG_size, pos_id:pos_id + PG_size] = rho
            
            for p_id, __ in enumerate(comp['proportions']):
                # for each component-subgroup
                c_theta[pos_id:pos_id + dims] = d_theta
                c_sig[pos_id:pos_id + dims] = d_sig
                c_tag[pos_id:pos_id + dims] = [
                    t + '-LOC-{}-CSG-{}'.format(l_id, p_id) for t in d_tag]
                c_rho[pos_id:pos_id + dims, 
                      pos_id:pos_id + dims] = rho_fr['DSG']
                pos_id += dims

        # create the covariance matrix
        np.fill_diagonal(c_rho, 1.0)
        c_COV = np.outer(c_sig, c_sig) * c_rho

        fragility_RV = RandomVariable(ID=300 + c_id,
                                      dimension_tags=c_tag,
                                      distribution_kind='lognormal',
                                      theta=c_theta,
                                      COV=c_COV)

        return fragility_RV

    def _create_RV_red_tags(self, rho_rt):

        # get the total number of red tag decision variables
        rv_count = 0
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            for dsg_i, DSG in comp['DSG_set'].items():
                for ds_i, DS in DSG['DS_set'].items():
                    if DS['red_tag']['theta'] > 0:
                        rv_count += len(comp['locations'])

        # create the (empty) input arrays for the RV
        f_theta = np.zeros(rv_count)
        f_tag = np.empty(rv_count, dtype=object)
        f_sig = np.zeros(rv_count)
        f_rho = np.ones(
            (rv_count, rv_count)) * rho_rt['CMP']  # set correlations between FGs

        f_pos_id = 0
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):

            d_theta, d_sig, d_tag = [np.array([]) for i in range(3)]

            for dsg_i, DSG in comp['DSG_set'].items():
                for ds_i, DS in DSG['DS_set'].items():
                    theta = DS['red_tag']['theta']
                    if theta > 0:
                        d_theta = np.append(d_theta, theta)
                        d_sig = np.append(d_sig, DS['red_tag']['cov'])
                        d_tag = np.append(
                            d_tag,
                            comp['ID'] + '-' + str(dsg_i) + '-' + str(ds_i))
            dims = len(d_theta)

            # get the total number of random variables for this fragility group
            PG_size = len(comp['locations']) * dims

            # create the (empty) input arrays for the RV
            c_theta = np.zeros(PG_size)
            c_tag = np.empty(PG_size, dtype=object)
            c_sig = np.zeros(PG_size)
            c_rho = np.ones(
                (PG_size, PG_size)) * rho_rt['PG'] # set correlations between PGs

            pos_id = 0
            for l_id in comp['locations']:
                # for each performance group (i.e. location)
                c_theta[pos_id:pos_id + dims] = d_theta
                c_sig[pos_id:pos_id + dims] = d_sig
                c_tag[pos_id:pos_id + dims] = [
                    t + '-LOC-{}'.format(l_id) for t in d_tag]
                c_rho[pos_id:pos_id + dims, pos_id:pos_id + dims] = rho_rt['DS']
                pos_id += dims

            # now append the performance group data to the global arrays
            f_end_id = f_pos_id + len(c_theta)
            f_theta[f_pos_id:f_end_id] = c_theta
            f_sig[f_pos_id:f_end_id] = c_sig
            f_tag[f_pos_id:f_end_id] = c_tag
            f_rho[f_pos_id:f_end_id, f_pos_id:f_end_id] = c_rho
            f_pos_id = f_end_id

        # create the global covariance matrix
        np.fill_diagonal(f_rho, 1.0)
        f_COV = np.outer(f_sig, f_sig) * f_rho

        tr_upper = 1. + (1. - f_theta) / f_theta

        red_tag_RV = RandomVariable(ID=400,
                                    dimension_tags=f_tag,
                                    distribution_kind='normal',
                                    theta=np.ones(len(f_theta)),
                                    COV=f_COV,
                                    corr_ref='post',
                                    truncation_limits=[np.zeros(len(f_theta)),
                                                       tr_upper])

        return red_tag_RV

    def _create_RV_repairs(self, rho_rep):

        # get the total number of repair cost and time decision variables
        rv_count = 0
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            for dsg_i, DSG in comp['DSG_set'].items():
                for ds_i, DS in DSG['DS_set'].items():
                    rv_count += 2 * len(comp['locations'])

        # create the (empty) input arrays for the RV
        f_tag = np.empty(rv_count, dtype=object)
        f_dkind = np.empty(rv_count, dtype=object)
        f_sig = np.zeros(rv_count)
        f_rho = np.ones(
            (rv_count, rv_count)) * rho_rep['CMP'] # set correlations between CMPs

        c_dims = []
        f_pos_id_c = 0
        f_pos_id_t = rv_count // 2
        # set the correlation between repair cost and repair time
        z = np.zeros((f_pos_id_t, f_pos_id_t))
        np.fill_diagonal(z, rho_rep['CT'])
        f_rho[:f_pos_id_t, f_pos_id_t:] = z
        f_rho[f_pos_id_t:, :f_pos_id_t] = z

        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):

            (dc_sig, dc_tag, dc_dkind,
             dt_sig, dt_tag, dt_dkind) = [np.array([]) for i in range(6)]

            for dsg_i, DSG in comp['DSG_set'].items():
                for ds_i, DS in DSG['DS_set'].items():
                    dc_sig = np.append(dc_sig, DS['repair_cost']['cov'])
                    dt_sig = np.append(dt_sig, DS['repair_time']['cov'])
                    dc_dkind = np.append(dc_dkind,
                                         DS['repair_cost']['distribution_kind'])
                    dt_dkind = np.append(dt_dkind,
                                         DS['repair_time']['distribution_kind'])
                    dc_tag = np.append(dc_tag,
                                       comp['ID'] + '-' + str(
                                           dsg_i) + '-' + str(
                                           ds_i) + '-C')
                    dt_tag = np.append(dt_tag, dc_tag[-1][:-1] + 'T')
            dims = len(dc_sig)

            # get the total number of random variables for this fragility group
            PG_size = len(comp['locations']) * dims

            # create the (empty) input arrays for the RV
            c_tag = np.empty(PG_size, dtype=object)
            c_dkind = np.empty(PG_size, dtype=object)
            c_sig = np.zeros(PG_size)
            c_rho = np.ones(
                (PG_size, PG_size)) * rho_rep['PG']  # set correlations between PGs

            t_tag = np.empty(PG_size, dtype=object)
            t_dkind = np.empty(PG_size, dtype=object)
            t_sig = np.zeros(PG_size)
            t_rho = np.ones(
                (PG_size, PG_size)) * rho_rep['PG'] # set correlations between PGs

            pos_id = 0
            for l_id in comp['locations']:
                # for each performance group (i.e. location)
                c_sig[pos_id:pos_id + dims] = dc_sig
                c_tag[pos_id:pos_id + dims] = [
                    t + '-LOC-{}'.format(l_id) for t in dc_tag]
                c_rho[pos_id:pos_id + dims, pos_id:pos_id + dims] = rho_rep['DS']
                c_dkind[pos_id:pos_id + dims] = dc_dkind

                t_sig[pos_id:pos_id + dims] = dt_sig
                t_tag[pos_id:pos_id + dims] = [
                    t + '-LOC-{}'.format(l_id) for t in dt_tag]
                t_rho[pos_id:pos_id + dims, pos_id:pos_id + dims] = rho_rep['DS']
                t_dkind[pos_id:pos_id + dims] = dt_dkind
                pos_id += dims

            # now add the performance group data to the global arrays
            f_end_id = f_pos_id_c + len(c_sig)
            f_sig[f_pos_id_c:f_end_id] = c_sig
            f_tag[f_pos_id_c:f_end_id] = c_tag
            f_dkind[f_pos_id_c:f_end_id] = c_dkind
            f_rho[f_pos_id_c:f_end_id, f_pos_id_c:f_end_id] = c_rho
            f_pos_id_c = f_end_id

            f_end_id = f_pos_id_t + len(t_sig)
            f_sig[f_pos_id_t:f_end_id] = t_sig
            f_tag[f_pos_id_t:f_end_id] = t_tag
            f_dkind[f_pos_id_t:f_end_id] = t_dkind
            f_rho[f_pos_id_t:f_end_id, f_pos_id_t:f_end_id] = t_rho
            f_pos_id_t = f_end_id

        # create the global covariance matrix
        np.fill_diagonal(f_rho, 1.0)
        f_COV = np.outer(f_sig, f_sig) * f_rho

        repair_RV = RandomVariable(ID=401,
                                   dimension_tags=f_tag,
                                   distribution_kind=f_dkind,
                                   theta=np.ones(len(f_sig)),
                                   COV=f_COV,
                                   corr_ref='post',
                                   truncation_limits=[np.zeros(len(f_sig)),
                                                      None])

        return repair_RV

    def _create_RV_injuries(self, rho_inj):

        inj_lvls = self._inj_lvls
        
        # get the total number of injury-related decision variables
        rv_count = 0
        comp_list = []
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            for dsg_i, DSG in comp['DSG_set'].items():
                for ds_i, DS in DSG['DS_set'].items():
                    thetas = DS['injuries']['theta']
                    if np.sum(thetas) > 0.:
                        rv_count += len(thetas) * len(comp['locations'])
                        comp_list.append(c_name)

        # create the (empty) input arrays for the RV
        f_theta = np.zeros(rv_count)
        f_tag = np.empty(rv_count, dtype=object)
        f_sig = np.zeros(rv_count)
        f_rho = np.ones(
            (rv_count, rv_count)) * rho_inj['CMP'] # set correlations between CMPs

        f_pos_id = 0
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            if c_name in comp_list:

                d_theta, d_sig, d_tag = [np.array([]) for i in range(3)]

                for dsg_i, DSG in comp['DSG_set'].items():
                    for ds_i, DS in DSG['DS_set'].items():
                        theta_list = DS['injuries']['theta']
                        cov_list = DS['injuries']['cov']
                        inj_lvls = len(cov_list)
                        for inj_id in range(len(cov_list)):
                            d_theta = np.append(d_theta, theta_list[inj_id])
                            d_sig = np.append(d_sig, cov_list[inj_id])
                            d_tag = np.append(d_tag,
                                              comp['ID'] + '-' + str(
                                                  dsg_i) + '-' + str(
                                                  ds_i) + '-' + str(inj_id))
                dims = len(d_sig)

                # get the total number of random variables for this fragility group
                PG_size = len(comp['locations']) * dims

                # create the (empty) input arrays for the RV
                c_tag = np.empty(PG_size, dtype=object)
                c_theta = np.zeros(PG_size)
                c_sig = np.zeros(PG_size)
                c_rho = np.ones(
                    (PG_size,
                     PG_size)) * rho_inj['PG'] # set correlations between PGs

                pos_id = 0

                for l_id in comp['locations']:
                    # for each performance group (i.e. location)
                    c_theta[pos_id:pos_id + dims] = d_theta
                    c_sig[pos_id:pos_id + dims] = d_sig
                    c_tag[pos_id:pos_id + dims] = [
                        t + '-LOC-{}'.format(l_id) for t in d_tag]
                    c_rho[pos_id:pos_id + dims,
                    pos_id:pos_id + dims] = rho_inj['DS']
                    pos_id += dims

                # finally, add the correlation between injury levels
                for i in range(PG_size // inj_lvls):
                    c_rho[i * inj_lvls:(i + 1) * inj_lvls,
                    i * inj_lvls:(i + 1) * inj_lvls] = rho_inj['LVL']

                # now add the performance group data to the global arrays
                f_end_id = f_pos_id + len(c_sig)
                f_theta[f_pos_id:f_end_id] = c_theta
                f_sig[f_pos_id:f_end_id] = c_sig
                f_tag[f_pos_id:f_end_id] = c_tag
                f_rho[f_pos_id:f_end_id, f_pos_id:f_end_id] = c_rho
                f_pos_id = f_end_id

        # create the global covariance matrix
        np.fill_diagonal(f_rho, 1.0)
        f_COV = np.outer(f_sig, f_sig) * f_rho

        # remove invalid entries
        z_list = np.where(f_theta == 0.)[0]
        f_COV = np.delete(np.delete(f_COV, z_list, axis=0), z_list, axis=1)
        f_tag = f_tag[f_theta > 0.]
        f_sig = f_sig[f_theta > 0.]
        f_theta = f_theta[f_theta > 0.]

        tr_upper = 1. + (1. - f_theta) / f_theta

        injury_RV = RandomVariable(ID=402,
                                   dimension_tags=f_tag,
                                   distribution_kind='normal',
                                   theta=np.ones(len(f_sig)),
                                   COV=f_COV,
                                   corr_ref='post',
                                   truncation_limits=[np.zeros(len(f_sig)),
                                                      tr_upper])

        return injury_RV

    def _create_RV_demands(self, beta_added):

        # unlike other random variables, the demand RV is based on raw data
        # first, collect the raw values from the EDP dict
        demand_data = []
        d_tags = []
        detection_limits = []
        for d_id, d_list in self._EDP_in.items():
            for i in range(len(d_list)):
                d_rd = d_list[i]['raw_data']
                events = list(d_rd.keys())
                demand_data.append(d_rd[events[0]])
                d_tags.append(str(d_id) + '-LOC-' + str(d_list[i]['floor']))
                detection_limits.append(
                    [0., self._AIM_in['EDP_detection_limits'][d_id]])

        detection_limits = np.transpose(np.asarray(detection_limits))
        demand_data = np.transpose(np.asarray(demand_data))

        # get the number of censored samples
        EDP_filter = np.all([np.all(demand_data > detection_limits[0], axis=1),
                             np.all(demand_data < detection_limits[1], axis=1)],
                            axis=0)
        censored_count = len(EDP_filter) - sum(EDP_filter)
        demand_data = demand_data[EDP_filter]
        demand_data = np.transpose(demand_data)

        # create the random variable
        demand_RV = RandomVariable(ID=200, dimension_tags=d_tags,
                                   raw_data=demand_data,
                                   detection_limits=detection_limits,
                                   censored_count=censored_count
                                   )

        # fit a multivariate lognormal distribution to the censored raw data
        demand_RV.fit_distribution('lognormal')

        # if we want to add other sources of uncertainty, we will need to redefine
        # the random variable
        if beta_added > 0.:
            # get the covariance matrix with added uncertainty
            COV_orig = demand_RV.COV
            sig_orig = np.sqrt(np.diagonal(COV_orig))
            rho_orig = COV_orig / np.outer(sig_orig, sig_orig)
            sig_mod = np.sqrt(sig_orig ** 2. + beta_added ** 2.)
            COV_mod = np.outer(sig_mod, sig_mod) * rho_orig

            # redefine the random variable
            demand_RV = RandomVariable(ID=200,
                                       dimension_tags=demand_RV.dimension_tags,
                                       distribution_kind='lognormal',
                                       theta=demand_RV.theta,
                                       COV=COV_mod)

        return demand_RV

    def _create_fragility_groups(self):

        RVd = self._RV_dict

        # create a list for the fragility groups
        FG_dict = dict()

        for c_id, comp in self._FG_in.items():

            FG_ID = len(FG_dict.keys())

            # create a list for the performance groups
            performance_groups = []

            # one group for each of the stories prescribed by the user
            PG_locations = comp['locations']
            for loc in PG_locations:
                PG_ID = 1000 + FG_ID * 10 + loc

                # get the quantity
                QNT = RandomVariableSubset(RVd['QNT'],
                                           tags=[c_id + '-QNT-' + str(loc), ])

                # create the damage objects
                # consequences do not need to be calculated on a subgroup basis

                # create a list for the damage state groups and their tags
                DSG_list = []
                d_tags = []
                for dsg_i, (DSG_ID, DSG) in enumerate(comp['DSG_set'].items()):
                    d_tags.append(c_id + '-' + DSG_ID)

                    # create a list for the damage states
                    DS_set = []

                    for ds_i, (DS_ID, DS) in enumerate(DSG['DS_set'].items()):

                        # create the consequence functions
                        data = DS['repair_cost']
                        f_median = prep_bounded_linear_median_DV(
                            **{k: data.get(k, None) for k in
                               ('median_max', 'median_min',
                                'quantity_lower', 'quantity_upper')})
                        cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + '-C' + \
                                 '-LOC-{}'.format(loc)
                        CF_RV = RandomVariableSubset(RVd['DV_REP'],
                                                     tags=cf_tag)
                        CF_cost = ConsequenceFunction(DV_median=f_median,
                                                      DV_distribution=CF_RV)

                        data = DS['repair_time']
                        f_median = prep_bounded_linear_median_DV(
                            **{k: data.get(k, None) for k in
                               ('median_max', 'median_min', 'quantity_lower',
                                'quantity_upper')})
                        cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + '-T' + \
                                 '-LOC-{}'.format(loc)
                        CF_RV = RandomVariableSubset(RVd['DV_REP'],
                                                     tags=cf_tag)
                        CF_time = ConsequenceFunction(DV_median=f_median,
                                                      DV_distribution=CF_RV)

                        data = DS['red_tag']
                        if data['theta'] > 0:
                            f_median = prep_constant_median_DV(data['theta'])
                            cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + \
                                     '-LOC-{}'.format(loc)
                            CF_RV = RandomVariableSubset(RVd['DV_RED'],
                                                         tags=cf_tag)
                            CF_red_tag = ConsequenceFunction(DV_median=f_median,
                                                             DV_distribution=CF_RV)
                        else:
                            CF_red_tag = None

                        data = DS['injuries']
                        CF_inj_set = []
                        for inj_i, theta in enumerate(data['theta']):
                            if theta > 0.:
                                f_median = prep_constant_median_DV(theta)
                                cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + \
                                         '-{}-LOC-{}'.format(inj_i, loc)
                                CF_RV = RandomVariableSubset(RVd['DV_INJ'],
                                                             tags=cf_tag)
                                CF_inj_set.append(ConsequenceFunction(
                                    DV_median=f_median,
                                    DV_distribution=CF_RV))
                            else:
                                CF_inj_set.append(None)

                        # add the DS to the list
                        DS_set.append(DamageState(ID=ds_i + 1,
                                                  description=DS['description'],
                                                  weight=DS['weight'],
                                                  affected_area=DS[
                                                      'affected_area'],
                                                  repair_cost_CF=CF_cost,
                                                  reconstruction_time_CF=CF_time,
                                                  red_tag_CF=CF_red_tag,
                                                  injuries_CF_set=CF_inj_set
                                                  ))

                    # add the DSG to the list
                    DSG_list.append(DamageStateGroup(ID=dsg_i + 1,
                                                     DS_set=DS_set,
                                                     DS_set_kind=DSG[
                                                         'DS_set_kind'],
                                                     description=DSG[
                                                         'description']
                                                     ))

                # create the fragility functions
                FF_set = []
                for p_id, prop in enumerate(comp['proportions']):
                    # assign the appropriate random variable to the fragility 
                    # function
                    ff_tags = [t + '-LOC-{}-CSG-{}'.format(loc, p_id)
                               for t in d_tags]
                    EDP_limit = RandomVariableSubset(RVd['FR-' + c_id],
                                                     tags=ff_tags)
                    FF_set.append(FragilityFunction(EDP_limit))

                # create the performance group
                PG = PerformanceGroup(ID=PG_ID,
                                      location=loc,
                                      quantity=QNT,
                                      fragility_functions=FF_set,
                                      DSG_set=DSG_list,
                                      proportions=comp['proportions'],
                                      directions=comp['directions']
                                      )
                performance_groups.append(PG)

            # create the fragility group
            FG = FragilityGroup(ID=FG_ID,
                                kind=comp['kind'],
                                demand_type=comp['demand_type'],
                                performance_groups=performance_groups,
                                directional=comp['directional'],
                                correlation=comp['correlation'],
                                demand_location_offset=comp['offset'],
                                incomplete=comp['incomplete'],
                                name=str(FG_ID) + ' - ' + comp['ID'],
                                description=comp['description']
                                )

            FG_dict.update({comp['ID']:FG})

        return FG_dict

    def _sample_event_time(self):
        
        sample_count = self._realizations
        
        # month - uniform distribution over [0,11]
        month = np.random.randint(0, 12, size=sample_count)

        # weekday - binomial with p=5/7
        weekday = np.random.binomial(1, 5. / 7., size=sample_count)

        # hour - uniform distribution over [0,23]
        hour = np.random.randint(0, 24, size=sample_count)

        data = pd.DataFrame(data={'month'   : month,
                                  'weekday?': weekday,
                                  'hour'    : hour},
                            dtype=int)

        return data

    def _get_population(self):
        """
        Use the population characteristics to generate random population samples.

        Returns
        -------
        P: DataFrame
            Explain...
        """
        POPin = self._POP_in
        TIME = self._TIME
        
        POP = pd.DataFrame(
            np.ones((len(TIME.index), len(POPin['peak']))) * POPin['peak'],
            columns=['LOC' + str(loc + 1)
                     for loc in range(len(POPin['peak']))])

        weekdays = TIME[TIME['weekday?'] == 1].index
        weekends = TIME[~TIME.index.isin(weekdays)].index

        for col in POP.columns.values:
            POP.loc[weekdays, col] = (
                POP.loc[weekdays, col] *
                np.array(POPin['weekday']['daily'])[
                    TIME.loc[weekdays, 'hour'].values.astype(int)] *
                np.array(POPin['weekday']['monthly'])[
                    TIME.loc[weekdays, 'month'].values.astype(int)])

            POP.loc[weekends, col] = (
                POP.loc[weekends, col] *
                np.array(POPin['weekday']['daily'])[
                    TIME.loc[weekends, 'hour'].values.astype(int)] *
                np.array(POPin['weekday']['monthly'])[
                    TIME.loc[weekends, 'month'].values.astype(int)])

        return POP

    def _calc_collapses(self):

        # filter the collapsed cases based on the demand samples
        collapsed_IDs = np.array([])
        for demand_ID, demand in self._EDP_dict.items():
            coll_df = pd.DataFrame()
            kind = demand_ID[:3]
            collapse_limit = self._AIM_in['EDP_collapse_limits'][kind]
            if collapse_limit is not None:
                EDP_samples = demand.samples
                coll_df = EDP_samples[EDP_samples > collapse_limit]
            collapsed_IDs = np.concatenate(
                (collapsed_IDs, coll_df.index.values))

        # get a list of IDs of the collapsed cases
        collapsed_IDs = np.unique(collapsed_IDs).astype(int)

        COL = pd.DataFrame(np.zeros(self._realizations), columns=['COL', ])
        COL.loc[collapsed_IDs, 'COL'] = 1

        return COL, collapsed_IDs

    def _calc_damage(self):
        
        ncID = self._ID_dict['non-collapse']
        NC_samples = len(ncID)
        DMG = pd.DataFrame()

        for fg_id, FG in self._FG_dict.items():

            PG_set = FG._performance_groups

            DS_list = []
            for DSG in PG_set[0]._DSG_set:
                for DS in DSG._DS_set:
                    DS_list.append(str(DSG._ID) + '-' + str(DS._ID))
            d_count = len(DS_list)

            MI = pd.MultiIndex.from_product([[FG._ID, ],
                                             [pg._ID for pg in PG_set],
                                             DS_list],
                                            names=['FG', 'PG', 'DS'])

            FG_damages = pd.DataFrame(np.zeros((NC_samples, len(MI))),
                                      columns=MI,
                                      index=ncID)

            for pg_i, PG in enumerate(PG_set):

                PG_ID = PG._ID
                PG_qnt = PG._quantity.samples.loc[ncID]

                # get the corresponding demands
                demand_ID = FG._demand_type + '-LOC-' + str(PG._location)
                EDP_samples = self._EDP_dict[demand_ID].samples.loc[
                    ncID]

                prps = PG._proportions
                dirs = PG._directions

                for csg_i, (csg_w, csg_dir) in enumerate(zip(prps, dirs)):
                    if csg_dir == 0:
                        DSG_df = PG._FF_set[csg_i].DSG_given_EDP(EDP_samples)
                    else:
                        DSG_df = PG._FF_set[csg_i].DSG_given_EDP(
                            EDP_samples * self._f_secondary_EDP)

                    for DSG in PG._DSG_set:
                        in_this_DSG = DSG_df[DSG_df.values == DSG._ID].index
                        if DSG._DS_set_kind == 'single':
                            DS = DSG._DS_set[0]
                            DS_tag = str(DSG._ID) + '-' + str(DS._ID)
                            FG_damages.loc[in_this_DSG,
                                           (FG._ID, PG_ID, DS_tag)] += csg_w
                        elif DSG._DS_set_kind == 'mutually exclusive':
                            DS_weights = [DS._weight for DS in DSG._DS_set]
                            DS_RV = RandomVariable(
                                ID=-1, dimension_tags=['me_DS', ],
                                distribution_kind='multinomial',
                                p_set=DS_weights)
                            DS_df = DS_RV.sample_distribution(
                                len(in_this_DSG)) + 1
                            for DS in DSG._DS_set:
                                DS_tag = str(DSG._ID) + '-' + str(DS._ID)
                                in_this_DS = DS_df[DS_df.values == DS._ID].index
                                FG_damages.loc[in_this_DSG[in_this_DS],
                                               (FG._ID, PG_ID, DS_tag)] += csg_w
                        else:
                            # TODO: simultaneous
                            print(DSG._DS_set_kind)

                FG_damages.iloc[:,
                pg_i * d_count:(pg_i + 1) * d_count] = FG_damages.mul(
                    PG_qnt.iloc[:, 0], axis=0)

            DMG = pd.concat((DMG, FG_damages), axis=1)

        DMG.index = ncID

        return DMG

    def _calc_red_tag(self):
        idx = pd.IndexSlice

        ncID = self._ID_dict['non-collapse']
        NC_samples = len(ncID)
        DV_RED = pd.DataFrame()

        for fg_id, FG in self._FG_dict.items():

            PG_set = FG._performance_groups

            DS_list = self._DMG.loc[:, idx[FG._ID, PG_set[0]._ID, :]].columns
            DS_list = DS_list.levels[2][DS_list.labels[2]].values

            MI = pd.MultiIndex.from_product([[FG._ID, ],
                                             [pg._ID for pg in PG_set],
                                             DS_list],
                                            names=['FG', 'PG', 'DS'])

            FG_RED = pd.DataFrame(np.zeros((NC_samples, len(MI))),
                                  columns=MI,
                                  index=ncID)

            for pg_i, PG in enumerate(PG_set):

                PG_ID = PG._ID
                PG_qnt = PG._quantity.samples.loc[ncID]

                PG_DMG = self._DMG.loc[:, idx[FG._ID, PG_ID, :]].div(
                    PG_qnt.iloc[:, 0],
                    axis=0)

                for d_i, d_tag in enumerate(DS_list):
                    dsg_i = int(d_tag[0]) - 1
                    ds_i = int(d_tag[-1]) - 1

                    DS = PG._DSG_set[dsg_i]._DS_set[ds_i]

                    if DS._red_tag_CF is not None:
                        RED_samples = DS.red_tag_dmg_limit(
                            sample_size=NC_samples)

                        is_red = PG_DMG.loc[:, (FG._ID, PG_ID, d_tag)].sub(
                            RED_samples, axis=0)
                        FG_RED.loc[:, (FG._ID, PG_ID, d_tag)] = (
                            is_red > 0.).astype(int)
                    else:
                        FG_RED.drop(labels=[(FG._ID, PG_ID, d_tag), ], axis=1,
                                    inplace=True)

            if FG_RED.size > 0:
                DV_RED = pd.concat((DV_RED, FG_RED), axis=1)

        return DV_RED

    def _calc_irrepairable(self):

        ncID = self._ID_dict['non-collapse']
        NC_samples = len(ncID)

        # determine which realizations lead to irrepairable damage
        # get the max residual drifts
        RED_max = None
        PID_max = None
        for demand_ID, demand in self._EDP_dict.items():
            kind = demand_ID[:3]
            if kind == 'RED':
                r_max = demand.samples.loc[ncID].values
                if RED_max is None:
                    RED_max = r_max
                else:
                    RED_max = np.max((RED_max, r_max), axis=0)
            elif kind == 'PID':
                d_max = demand.samples.loc[ncID].values
                if PID_max is None:
                    PID_max = d_max
                else:
                    PID_max = np.max((PID_max, d_max), axis=0)

        if RED_max is None:
            # we need to estimate residual drifts based on peak drifts
            RED_max = np.zeros(NC_samples)

            # based on Appendix C in FEMA P-58
            delta_y = self._AIM_in['yield_drift']
            small = PID_max < delta_y
            medium = PID_max < 4 * delta_y
            large = PID_max >= 4 * delta_y

            RED_max[large] = PID_max[large] - 3 * delta_y
            RED_max[medium] = 0.3 * (PID_max[medium] - delta_y)
            RED_max[small] = 0.

        # get the probabilities of irrepairability
        irrep_frag = self._AIM_in['irrepairable_fragility']
        RV_irrep = RandomVariable(ID=-1, dimension_tags=['RED_irrep', ],
                                  distribution_kind='lognormal',
                                  theta=irrep_frag['theta'],
                                  COV=irrep_frag['sig'] ** 2.
                                  )
        RED_irrep = RV_irrep.sample_distribution(NC_samples)['RED_irrep'].values

        # determine if the realizations are repairable
        irrepairable = RED_max > RED_irrep
        irrepairable_IDs = ncID[np.where(irrepairable)[0]]

        return irrepairable_IDs

    def _calc_repair_cost_and_time(self):

        idx = pd.IndexSlice

        DMG_by_FG_and_DS = self._DMG.groupby(level=[0, 2], axis=1).sum()

        repID = self._ID_dict['repairable']
        REP_samples = len(repID)
        DV_COST = pd.DataFrame(np.zeros((REP_samples, len(self._DMG.columns))),
                               columns=self._DMG.columns, index=repID)
        DV_TIME = deepcopy(DV_COST)

        for fg_id, FG in self._FG_dict.items():

            PG_set = FG._performance_groups

            DS_list = self._DMG.loc[:, idx[FG._ID, PG_set[0]._ID, :]].columns
            DS_list = DS_list.levels[2][DS_list.labels[2]].values

            for pg_i, PG in enumerate(PG_set):

                PG_ID = PG._ID

                for d_i, d_tag in enumerate(DS_list):
                    dsg_i = int(d_tag[0]) - 1
                    ds_i = int(d_tag[-1]) - 1

                    DS = PG._DSG_set[dsg_i]._DS_set[ds_i]

                    TOT_qnt = DMG_by_FG_and_DS.loc[
                        repID, (FG._ID, d_tag)]
                    PG_qnt = self._DMG.loc[repID, 
                                           (FG._ID, PG_ID, d_tag)]

                    # repair cost
                    COST_samples = DS.unit_repair_cost(quantity=TOT_qnt)
                    DV_COST.loc[:,
                    (FG._ID, PG_ID, d_tag)] = COST_samples * PG_qnt

                    # repair time
                    TIME_samples = DS.unit_reconstruction_time(quantity=TOT_qnt)
                    DV_TIME.loc[:,
                    (FG._ID, PG_ID, d_tag)] = TIME_samples * PG_qnt

        return DV_COST, DV_TIME

    def _calc_collapse_injuries(self):

        # calculate casualties and injuries for the collapsed cases
        # generate collapse modes
        colID = self._ID_dict['collapse']
        C_samples = len(colID)

        inj_lvls = self._inj_lvls
        coll_modes = self._AIM_in['collapse_modes']
        P_keys = [cmk for cmk in coll_modes.keys()]
        P_modes = [coll_modes[k]['w'] for k in P_keys]

        # create the DataFrame that collects the decision variables
        COL_INJ = pd.DataFrame(np.zeros((C_samples, inj_lvls + 1)),
                               columns=('CM', *['INJ-{}'.format(i) for i in
                                                range(inj_lvls)]),
                               index=colID)

        CM_RV = RandomVariable(ID=-1, dimension_tags=['CM', ],
                               distribution_kind='multinomial',
                               p_set=P_modes)
        COL_INJ['CM'] = CM_RV.sample_distribution(C_samples).values

        # get the popoulation values corresponding to the collapsed cases
        P_sel = self._POP.loc[colID]

        # calculate the exposure of the popoulation
        for cm_i, cmk in enumerate(P_keys):
            mode_IDs = COL_INJ[COL_INJ['CM'] == cm_i].index
            CFAR = coll_modes[cmk]['CFAR']
            INJ = coll_modes[cmk]['injuries']
            for loc_i in range(len(CFAR)):
                loc_label = 'LOC{}'.format(loc_i + 1)
                if loc_label in P_sel.columns:
                    for inj_i in range(inj_lvls):
                        INJ_i = P_sel.loc[mode_IDs, loc_label] * CFAR[loc_i] * \
                                INJ[inj_i]
                        COL_INJ.loc[mode_IDs, 'INJ-{}'.format(inj_i)] += INJ_i

        return COL_INJ

    def _calc_non_collapse_injuries(self):

        idx = pd.IndexSlice

        ncID = self._ID_dict['non-collapse']
        NC_samples = len(ncID)
        DV_INJ_dict = dict([(i, pd.DataFrame(np.zeros((NC_samples,
                                                       len(self._DMG.columns))),
                                             columns=self._DMG.columns,
                                             index=ncID))
                            for i in range(self._inj_lvls)])

        for fg_id, FG in self._FG_dict.items():

            PG_set = FG._performance_groups

            DS_list = self._DMG.loc[:, idx[FG._ID, PG_set[0]._ID, :]].columns
            DS_list = DS_list.levels[2][DS_list.labels[2]].values

            for pg_i, PG in enumerate(PG_set):

                PG_ID = PG._ID

                for d_i, d_tag in enumerate(DS_list):
                    dsg_i = int(d_tag[0]) - 1
                    ds_i = int(d_tag[-1]) - 1

                    DS = PG._DSG_set[dsg_i]._DS_set[ds_i]

                    if DS._affected_area > 0.:
                        P_affected = (self._POP.loc[ncID]
                                      * DS._affected_area / self._AIM_in['area'])

                        QNT = self._DMG.loc[:, (FG._ID, PG_ID, d_tag)]

                        # estimate injuries
                        for i in range(self._inj_lvls):
                            INJ_samples = DS.unit_injuries(severity_level=i,
                                                           sample_size=NC_samples)
                            if INJ_samples is not None:
                                INJ_samples.index = ncID
                                P_aff_i = P_affected.loc[:,
                                          'LOC{}'.format(PG._location)]
                                INJ_i = INJ_samples * P_aff_i * QNT
                                DV_INJ_dict[i].loc[:,
                                (FG._ID, PG_ID, d_tag)] = INJ_i

                                # remove the useless columns from DV_INJ
        for i in range(self._inj_lvls):
            DV_INJ = DV_INJ_dict[i]
            DV_INJ_dict[i] = DV_INJ.loc[:, (DV_INJ != 0.0).any(axis=0)]

        return DV_INJ_dict
