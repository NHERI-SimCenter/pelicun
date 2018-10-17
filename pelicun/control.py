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

    Assessment
    FEMA_P58_Assessment

"""

from .base import *
from .uq import *
from .model import *
from .file_io import *

class Assessment(object):
    """
    A high-level class that collects features common to all supported loss
    assessment methods. This class will only rarely be called directly when
    using pelicun.
    """
    
    def __init__(self):
        
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
        
    def read_inputs(self, path_DL_input, path_EDP_input, verbose=False):
        """
        Read and process the input files to describe the loss assessment task.
        
        Parameters
        ----------
        path_DL_input: string
            Location of the Damage and Loss input file. The file is expected to
            be a JSON with data stored in a standard format described in detail
            in the Input section of the documentation.
        path_EDP_input: string
            Location of the EDP input file. The file is expected to follow the 
            output formatting of Dakota. The Input section of the documentation
            provides more information about the expected formatting.
        verbose: boolean, default: False
            If True, the method echoes the information read from the files.
            This can be useful to ensure that the information in the file is 
            properly read by the method.
        """
        
        # read SimCenter inputs -----------------------------------------------
        # BIM file
        self._AIM_in = read_SimCenter_DL_input(path_DL_input, verbose=verbose)

        # EDP file
        self._EDP_in = read_SimCenter_EDP_input(path_EDP_input, 
                                                verbose=verbose)
    
    def define_random_variables(self):
        """
        Define the random variables used for loss assessment.

        """
        pass
    
    def define_loss_model(self):
        """
        Create the stochastic loss model based on the inputs provided earlier.

        """
        pass
    
    def calculate_damage(self):
        """
        Characterize the damage experienced in each random event realization.

        """
        self._ID_dict = {}
    
    def calculate_losses(self):
        """
        Characterize the consequences of damage in each random event realization.

        """
        self._DV_dict = {}
    
    def write_outputs(self):
        """
        Export the results.

        """
        pass
    
class FEMA_P58_Assessment(Assessment):
    """
    An Assessment class that implements the loss assessment method in FEMA P58.
    """
    def __init__(self, inj_lvls = 2):
        super(FEMA_P58_Assessment, self).__init__()
        
        # constants for the FEMA-P58 methodology
        self._inj_lvls = inj_lvls
        
    @property
    def beta_additional(self):
        """
        Calculate the total additional uncertainty for post processing.
        
        The total additional uncertainty is the squared root of sum of squared
        uncertainties corresponding to ground motion and modeling.
        
        Returns
        -------
        beta_tot: float
            The total uncertainty (logarithmic EDP standard deviation) to add
            to the EDP distribution.
        """
        
        AU = self._AIM_in['general']['added_uncertainty']
        
        beta_tot = 0.
        if AU['beta_m'] is not None:
            beta_tot += AU['beta_m']**2.
        if AU['beta_gm'] is not None:
            beta_tot += AU['beta_gm']**2.
            
        # if no uncertainty is assigned, we consider a minimum of 10^-4
        if beta_tot == 0:
            beta_tot = 1e-4
        else:
            beta_tot = np.sqrt(beta_tot)
        
        return beta_tot

    def read_inputs(self, path_DL_input, path_EDP_input, 
                    path_CMP_data=None, path_POP_data=None, verbose=False):
        """
        Read and process the input files to describe the loss assessment task.
        
        Parameters
        ----------
        path_DL_input: string
            Location of the Damage and Loss input file. The file is expected to
            be a JSON with data stored in a standard format described in detail
            in the Input section of the documentation.
        path_EDP_input: string
            Location of the EDP input file. The file is expected to follow the 
            output formatting of Dakota. The Input section of the documentation
            provides more information about the expected formatting.
        path_CMP_data: string, default: None
            Location of the folder with component damage and loss data files. 
            The default None value triggers the use of the FEMA P58 first
            edition data from pelicun/resources/component DL/FEMA P58 first 
            edition/.
        path_POP_data: string, default: None
            Location of the JSON file that describes the temporal distribution
            of the population per the FEMA P58 method. The default None value
            triggers the use of the FEMA P58 first edition distribution from
            pelicun/resources/population.json. 
        verbose: boolean, default: False
            If True, the method echoes the information read from the files.
            This can be useful to ensure that the information in the file is 
            properly read by the method.

        """
        
        super(FEMA_P58_Assessment, self).read_inputs(path_DL_input, path_EDP_input, verbose)
        
        # check if the component data path is provided by the user
        if path_CMP_data is None:
            warnings.warn(UserWarning(
                "The component database is not specified; using the default "
                "FEMA P58 first edition data."
            ))
            path_CMP_data = '../../resources/component DL/FEMA P58 first edition/'

        if path_POP_data is None:
            warnings.warn(UserWarning(
                "The population distribution is not specified; using the default "
                "FEMA P58 first edition data."
            ))
            path_POP_data = '../../resources/population.json'

        # assume that the asset is a building
        # TODO: If we want to apply FEMA-P58 to non-building assets, several parts of this methodology need to be extended.
        BIM = self._AIM_in

        # read component and population data ----------------------------------
        # components
        self._FG_in = read_component_DL_data(path_CMP_data, BIM['components'],
                                             verbose=verbose)
        
        # population
        POP = read_population_distribution(
            path_POP_data, 
            BIM['general']['occupancy_type'], 
            verbose=verbose)

        POP['peak'] = BIM['general']['population']          
        self._POP_in = POP              

    def define_random_variables(self):
        """
        Define the random variables used for loss assessment.
        
        Following the FEMA P58 methodology, the groups of parameters below are 
        considered random. Simple correlation structures within each group can
        be specified through the DL input file. The random decision variables
        are only created and used later if those particular decision variables
        are requested in the input file.
        
        1. Demand (EDP) distribution
        
        Describe the uncertainty in the demands. Unlike other random variables,
        the EDPs are characterized by the EDP input data provided earlier. All
        EDPs are handled in one multivariate lognormal distribution. If more 
        than one sample is provided, the distribution is fit to the EDP data.
        Otherwise, the provided data point is assumed to be the median value
        and the additional uncertainty prescribed describes the dispersion. See
        _create_RV_demands() for more details.
        
        2. Component quantities
        
        Describe the uncertainty in the quantity of components in each 
        Performance Group. All Fragility Groups are handled in the same 
        multivariate distribution. Consequently, correlation between various 
        groups of component quantities can be specified. See 
        _create_RV_quantities() for details. 
        
        3. Fragility EDP limits
        
        Describe the uncertainty in the EDP limit that corresponds to 
        exceedance of each Damage State. EDP limits are grouped by Fragility
        Groups. Consequently, correlation between fragility limits are 
        currently are limited within Fragility Groups. See 
        _create_RV_fragilities() for details.
        
        4. Reconstruction cost and time
        
        Describe the uncertainty in the cost and duration of reconstruction of 
        each component conditioned on the damage state of the component. All
        Fragility Groups are handled in the same multivariate distribution.
        Consequently, correlation between various groups of component 
        reconstruction time and cost estimates can be specified. See 
        _create_RV_repairs() for details.
        
        5. Damaged component proportions that trigger a red tag
        
        Describe the uncertainty in the amount of damaged components needed to
        trigger a red tag for the building. All Fragility Groups are handled in
        the same multivariate distribution. Consequently, correlation between
        various groups of component proportion limits can be specified. See
        _create_RV_red_tags() for details.
        
        6. Injuries
        
        Describe the uncertainty in the proportion of people in the affected
        area getting injuries exceeding a certain level of severity. FEMA P58
        uses two severity levels: injury and fatality. Both levels for all
        Fragility Groups are handled in the same multivariate distribution. 
        Consequently, correlation between various groups of component injury
        expectations can be specified. See _create_RV_injuries() for details. 
        """
        super(FEMA_P58_Assessment, self).define_random_variables()

        # create the random variables -----------------------------------------
        DEP = self._AIM_in['dependencies']
        
        self._RV_dict = {}

        # quantities 100
        self._RV_dict.update({'QNT': 
                            self._create_RV_quantities(DEP['quantities'])})
        
        # fragilities 300
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            self._RV_dict.update({
                'FR-' + c_name: 
                    self._create_RV_fragilities(c_id, comp, 
                                                DEP['fragilities'])})
            
        # consequences 400
        DVs = self._AIM_in['decision_variables']
        
        if DVs['red_tag']:
            self._RV_dict.update({'DV_RED': 
                                self._create_RV_red_tags(DEP['red_tags'])})
        if DVs['rec_time'] or DVs['rec_cost']:
            self._RV_dict.update({'DV_REP': 
                                self._create_RV_repairs(
                                    DEP['rec_costs'], 
                                    DEP['rec_times'],
                                    DEP['cost_and_time'])})
        if DVs['injuries']:
            self._RV_dict.update({'DV_INJ': 
                                self._create_RV_injuries(
                                    DEP['injuries'],
                                    DEP['injury_lvls'])})

        # demands 200
        self._RV_dict.update({'EDP': self._create_RV_demands(
            self.beta_additional)})

        # sample the random variables -----------------------------------------
        for r_i, rv in self._RV_dict.items():
            rv.sample_distribution(self._AIM_in['general']['realizations'])

    def define_loss_model(self):
        """
        Create the stochastic loss model based on the inputs provided earlier.
        
        Following the FEMA P58 methodology, the components specified in the 
        Damage and Loss input file are used to create Fragility Groups. Each
        Fragility Group corresponds to a component that might be present in 
        the building at several locations. See _create_fragility_groups() for
        more details about the creation of Fragility Groups.

        """
        super(FEMA_P58_Assessment, self).define_loss_model()
        
        # fragility groups
        self._FG_dict = self._create_fragility_groups()

        # demands
        self._EDP_dict = dict(
            [(tag, RandomVariableSubset(self._RV_dict['EDP'],tags=tag))
             for tag in self._RV_dict['EDP']._dimension_tags])
        
    def calculate_damage(self):
        """
        Characterize the damage experienced in each random event realization.
        
        First, the time of the event (month, weekday/weekend, hour) is randomly
        generated for each realization. Given the event time, the number of 
        people present at each floor of the building is calculated.
        
        Second, the realizations that led to collapse are filtered. See 
        _calc_collapses() for more details on collapse estimation.
        
        Finally, the realizations that did not lead to building collapse are 
        further investigated and the quantities of components in each damage
        state are estimated. See _calc_damage() for more details on damage 
        estimation.

        """
        super(FEMA_P58_Assessment, self).calculate_damage()

        # event time - month, weekday, and hour realizations
        self._TIME = self._sample_event_time()
        
        # get the population conditioned on event time
        self._POP = self._get_population()
        
        # collapses
        self._COL, collapsed_IDs = self._calc_collapses()
        self._ID_dict.update({'collapse':collapsed_IDs})

        # select the non-collapse cases for further analyses
        non_collapsed_IDs = self._POP[
            ~self._POP.index.isin(collapsed_IDs)].index.values.astype(int)
        self._ID_dict.update({'non-collapse': non_collapsed_IDs})
        
        # damage in non-collapses
        self._DMG = self._calc_damage()

    def calculate_losses(self):
        """
        Characterize the consequences of damage in each random event realization.
        
        For the sake of efficiency, only the decision variables requested in 
        the input file are estimated. The following consequences are handled by 
        this method:
        
        Reconstruction time and cost
        Estimate the irrepairable cases based on residual drift magnitude and
        the provided irrepairable drift limits. Realizations that led to 
        irrepairable damage or collapse are assigned the replacement cost and
        time of the building when reconstruction cost and time is estimated. 
        Repairable cases get a cost and time estimate for each Damage State in 
        each Performance Group. For more information about estimating 
        irrepairability see _calc_irrepairable() and reconstruction cost and
        time see _calc_repair_cost_and_time() methods.
        
        Injuries
        Collapse-induced injuries are based on the collapse modes and 
        corresponding injury characterization. Injuries conditioned on no 
        collapse are based on the affected area and the probability of 
        injuries of various severity specified in the component data file. For
        more information about estimating injuries conditioned on collapse and
        no collapse, see _calc_collapse_injuries() and 
        _calc_non_collapse_injuries, respecitvely.
        
        Red Tag
        The probability of getting an unsafe placard or red tag is a function
        of the amount of damage experienced in various Damage States for each
        Performance Group. The damage limits that trigger an unsafe placard are
        specified in the component data file. For more information on 
        assigning red tags to realizations see the _calc_red_tag() method.
        
        """
        super(FEMA_P58_Assessment, self).calculate_losses()
        DVs = self._AIM_in['decision_variables']
        
        # red tag probability
        if DVs['red_tag']:
            DV_RED = self._calc_red_tag()
            
            self._DV_dict.update({'red_tag': DV_RED})

        # reconstruction cost and time
        if DVs['rec_cost'] or DVs['rec_time']:
            # irrepairable cases
            irrepairable_IDs = self._calc_irrepairable()
    
            # collect the IDs of repairable realizations
            P_NC = self._POP.loc[self._ID_dict['non-collapse']]
            repairable_IDs = P_NC[
                ~P_NC.index.isin(irrepairable_IDs)].index.values.astype(int)
    
            self._ID_dict.update({'repairable': repairable_IDs})
            self._ID_dict.update({'irrepairable': irrepairable_IDs})
    
            # reconstruction cost and time for repairable cases
            DV_COST, DV_TIME = self._calc_repair_cost_and_time()

            self._DV_dict.update({
                'rec_cost': DV_COST,
                'rec_time': DV_TIME,
            })
            
        # injuries due to collapse
        if DVs['injuries']:
            COL_INJ = self._calc_collapse_injuries()
    
            # injuries in non-collapsed cases
            DV_INJ_dict = self._calc_non_collapse_injuries()
            
            # store results
            if COL_INJ is not None:
                self._COL = pd.concat([self._COL, COL_INJ], axis=1)

            self._DV_dict.update({'injuries': DV_INJ_dict})
        
    def aggregate_results(self):
        """
        
        Returns
        -------

        """

        DVs = self._AIM_in['decision_variables']

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
        if DVs['rec_cost'] or DVs['rec_time']:
            repID = self._ID_dict['repairable']
            irID = self._ID_dict['irrepairable']

        MI = pd.MultiIndex.from_tuples(MI_raw)

        SUMMARY = pd.DataFrame(np.empty((
            self._AIM_in['general']['realizations'], 
            len(MI))), columns=MI)
        SUMMARY[:] = np.NaN

        # event time
        for prop in ['month', 'weekday?', 'hour']:
            offset = 0
            if prop == 'month':
                offset = 1
            SUMMARY.loc[:, ('event time', prop)] = \
                self._TIME.loc[:, prop] + offset

        # inhabitants
        SUMMARY.loc[:, ('inhabitants', '')] = self._POP.sum(axis=1)

        # collapses
        SUMMARY.loc[:, ('collapses', 'collapsed?')] = self._COL.iloc[:, 0]        

        # red tag
        if DVs['red_tag']:
            SUMMARY.loc[ncID, ('red tagged?', '')] = \
                self._DV_dict['red_tag'].max(axis=1)

        # reconstruction cost
        if DVs['rec_cost']:
            SUMMARY.loc[ncID, ('reconstruction', 'cost')] = \
                self._DV_dict['rec_cost'].sum(axis=1)
    
            repl_cost = self._AIM_in['general']['replacement_cost']
            SUMMARY.loc[colID, ('reconstruction', 'cost')] = repl_cost

        if DVs['rec_cost'] or DVs['rec_time']:
            SUMMARY.loc[ncID, ('reconstruction', 'irrepairable?')] = 0
            SUMMARY.loc[irID, ('reconstruction', 'irrepairable?')] = 1

        if DVs['rec_cost']:
            SUMMARY.loc[irID, ('reconstruction', 'cost')] = repl_cost

        
            repair_impractical_IDs = SUMMARY.loc[
                SUMMARY.loc[:, ('reconstruction', 'cost')] > repl_cost].index
            SUMMARY.loc[repID, ('reconstruction', 'cost impractical?')] = 0
            SUMMARY.loc[repair_impractical_IDs,
                        ('reconstruction', 'cost impractical?')] = 1
            SUMMARY.loc[
                repair_impractical_IDs, ('reconstruction', 'cost')] = repl_cost

        # reconstruction time
        if DVs['rec_time']:
            SUMMARY.loc[ncID, ('reconstruction', 'time-sequential')] = \
                self._DV_dict['rec_time'].sum(axis=1)
            SUMMARY.loc[ncID, ('reconstruction', 'time-parallel')] = \
                self._DV_dict['rec_time'].max(axis=1)

            rep_time = self._AIM_in['general']['replacement_time']

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
        if DVs['injuries']:
            if 'CM' in self._COL.columns:
                SUMMARY.loc[colID, ('collapses', 'mode')] = self._COL.loc[:, 'CM']
            
                SUMMARY.loc[colID, ('injuries', 'casualties')] = \
                    self._COL.loc[:, 'INJ-0']
                SUMMARY.loc[colID, ('injuries', 'fatalities')] = \
                    self._COL.loc[:, 'INJ-1']
    
            SUMMARY.loc[ncID, ('injuries', 'casualties')] = \
                self._DV_dict['injuries'][0].sum(axis=1)
            SUMMARY.loc[ncID, ('injuries', 'fatalities')] = \
                self._DV_dict['injuries'][1].sum(axis=1)
        
        self._SUMMARY = SUMMARY.dropna(axis=1,how='all')

    def write_outputs(self):
        """
        
        Returns
        -------

        """
        super(FEMA_P58_Assessment, self).write_outputs()

    def _create_correlation_matrix(self, rho_target, c_target=-1, 
                                   include_CSG=False, 
                                   include_DSG=False, include_DS=False):
        """
        
        Parameters
        ----------
        rho_target
        c_target
        include_CSG
        include_DSG
        include_DS

        Returns
        -------

        """

        # set the correlation structure
        rho_FG, rho_PG, rho_LOC, rho_DIR, rho_CSG, rho_DS = np.zeros(6)

        if rho_target in ['FG', 'PG', 'DIR', 'LOC', 'CSG', 'ATC', 'DS']:
            rho_DS = 1.0
        if rho_target in ['FG', 'PG', 'DIR', 'LOC', 'CSG']:
            rho_CSG = 1.0
        if rho_target in ['FG', 'PG', 'DIR']:
            rho_DIR = 1.0
        if rho_target in ['FG', 'PG', 'LOC']:
            rho_LOC = 1.0
        if rho_target in ['FG', 'PG']:
            rho_PG = 1.0
        if rho_target == 'FG':
            rho_FG = 1.0

        L_D_list = []
        dims = []
        DS_list = []
        ATC_rho = []
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):
            if ((c_target == -1) or (c_id == c_target)):
                c_L_D_list = []
                c_DS_list = []
                ATC_rho.append(comp['correlation'])

                if include_DSG:
                    DS_count = 0
                    for dsg_i, DSG in comp['DSG_set'].items():
                        if include_DS:
                            DS_count += len(DSG['DS_set'])
                        else:
                            DS_count += 1
                else:
                    DS_count = 1

                for loc in comp['locations']:
                    if include_CSG:
                        u_dirs = comp['directions']
                    else:
                        u_dirs = np.unique(comp['directions'])
                    c_L_D_list.append([])
                    for dir_ in u_dirs:
                        c_DS_list.append(DS_count)
                        for ds_i in range(DS_count):
                            c_L_D_list[-1].append(dir_)

                c_dims = sum([len(loc) for loc in c_L_D_list])
                dims.append(c_dims)
                L_D_list.append(c_L_D_list)
                DS_list.append(c_DS_list)

        rho = np.ones((sum(dims), sum(dims))) * rho_FG

        f_pos_id = 0
        for c_id, (c_L_D_list, c_dims, c_DS_list) in enumerate(
            zip(L_D_list, dims, DS_list)):
            c_rho = np.ones((c_dims, c_dims)) * rho_PG

            # dependencies btw directions
            if rho_DIR != 0:
                c_pos_id = 0
                for loc_D_list in c_L_D_list:
                    l_dim = len(loc_D_list)
                    c_rho[c_pos_id:c_pos_id + l_dim,
                    c_pos_id:c_pos_id + l_dim] = rho_DIR
                    c_pos_id = c_pos_id + l_dim

            # dependencies btw locations
            if rho_LOC != 0:
                flat_dirs = []
                [[flat_dirs.append(dir_i) for dir_i in dirs] for dirs in
                 c_L_D_list]
                flat_dirs = np.array(flat_dirs)
                for u_dir in np.unique(flat_dirs):
                    dir_ids = np.where(flat_dirs == u_dir)[0]
                    for i in dir_ids:
                        for j in dir_ids:
                            c_rho[i, j] = rho_LOC

            if ((rho_CSG != 0) or (rho_target == 'ATC')):
                c_pos_id = 0
                if rho_target == 'ATC':
                    rho_to_use = float(ATC_rho[c_id])
                else:
                    rho_to_use = rho_CSG
                for loc_D_list in c_L_D_list:
                    flat_dirs = np.array(loc_D_list)
                    for u_dir in np.unique(flat_dirs):
                        dir_ids = np.where(flat_dirs == u_dir)[0]
                        for i in dir_ids:
                            for j in dir_ids:
                                c_rho[c_pos_id + i, c_pos_id + j] = rho_to_use
                    c_pos_id = c_pos_id + len(loc_D_list)

            if rho_DS != 0:
                c_pos_id = 0
                for l_dim in c_DS_list:
                    c_rho[c_pos_id:c_pos_id + l_dim,
                          c_pos_id:c_pos_id + l_dim] = rho_DS
                    c_pos_id = c_pos_id + l_dim

            rho[f_pos_id:f_pos_id + c_dims,
                f_pos_id:f_pos_id + c_dims] = c_rho
            f_pos_id = f_pos_id + c_dims

        np.fill_diagonal(rho, 1.0)

        return rho

    def _create_RV_quantities(self, rho_qnt):
        """
        
        Parameters
        ----------
        rho_qnt

        Returns
        -------

        """

        q_theta, q_sig, q_tag, q_dist = [np.array([]) for i in range(4)]

        # collect the parameters for each quantity dimension
        for c_id, comp in self._FG_in.items():
            u_dirs = np.unique(comp['directions'])

            dir_weights = comp['dir_weights']
            theta_list = []
            [[theta_list.append(qnt * dw) for dw in dir_weights]
             for qnt in comp['quantities']]
            q_theta = np.append(q_theta, theta_list)

            if comp['distribution_kind'] == 'normal':
                q_sig = np.append(q_sig, (
                    comp['cov'] * np.asarray(theta_list)).tolist())
            else:
                q_sig = np.append(q_sig, (
                    np.ones(len(theta_list)) * comp['cov']).tolist())

            q_tag = np.append(q_tag,
                              [[c_id + '-QNT-' + str(s_i) + '-' + str(d_i)
                                for d_i in u_dirs]
                               for s_i in comp['locations']])
            q_dist = np.append(q_dist,
                               [[comp['distribution_kind'] for d_i in u_dirs]
                                for s_i in comp['locations']])

        dims = len(q_theta)
        rho = self._create_correlation_matrix(rho_qnt)
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
        """
        
        Parameters
        ----------
        c_id
        comp
        rho_fr

        Returns
        -------

        """

        # prepare the basic multivariate distribution data for one component subgroup considering all damage states
        d_theta, d_sig, d_tag = [np.array([]) for i in range(3)]

        for d_id, DSG in comp['DSG_set'].items():
            d_theta = np.append(d_theta, DSG['theta'])
            d_sig = np.append(d_sig, DSG['sig'])
            d_tag = np.append(d_tag, comp['ID'] + '-' + str(d_id))
        dims = len(d_theta)

        # get the total number of random variables for this fragility group
        rv_count = len(comp['locations']) * len(comp['directions']) * dims

        # create the (empty) input arrays for the RV
        c_theta = np.zeros(rv_count)
        c_tag = np.empty(rv_count, dtype=object)
        c_sig = np.zeros(rv_count)

        pos_id = 0
        for l_id in comp['locations']:
            # for each location-direction pair)
            for d_id, __ in enumerate(comp['directions']):
                # for each component-subgroup
                c_theta[pos_id:pos_id + dims] = d_theta
                c_sig[pos_id:pos_id + dims] = d_sig
                c_tag[pos_id:pos_id + dims] = [
                    t + '-LOC-{}-CSG-{}'.format(l_id, d_id) for t in d_tag]
                pos_id += dims

        # create the covariance matrix
        c_rho = self._create_correlation_matrix(rho_fr, c_target=c_id, 
                                                include_DSG=True,
                                                include_CSG=True)
        c_COV = np.outer(c_sig, c_sig) * c_rho

        fragility_RV = RandomVariable(ID=300 + c_id,
                                      dimension_tags=c_tag,
                                      distribution_kind='lognormal',
                                      theta=c_theta,
                                      COV=c_COV)

        return fragility_RV

    def _create_RV_red_tags(self, rho_target):

        f_theta, f_sig, f_tag = [np.array([]) for i in range(3)]
        for c_id, (c_name, comp) in enumerate(self._FG_in.items()):

            d_theta, d_sig, d_tag = [np.array([]) for i in range(3)]

            for dsg_i, DSG in comp['DSG_set'].items():
                for ds_i, DS in DSG['DS_set'].items():
                    theta = DS['red_tag']['theta']
                    d_theta = np.append(d_theta, theta)
                    d_sig = np.append(d_sig, DS['red_tag']['cov'])
                    d_tag = np.append(d_tag,
                                      comp['ID'] + '-' + str(dsg_i) + '-' + str(
                                          ds_i))

            for loc in comp['locations']:
                for dir_ in np.unique(comp['directions']):
                    f_theta = np.append(f_theta, d_theta)
                    f_sig = np.append(f_sig, d_sig)
                    f_tag = np.append(f_tag,
                                      [t + '-LOC-{}-DIR-{}'.format(loc, dir_)
                                       for t in d_tag])

        rho = self._create_correlation_matrix(rho_target, c_target=-1, 
                                              include_DSG=True,
                                              include_DS=True)

        # remove the unnecessary fields
        to_remove = np.where(f_theta == 0)[0]
        rho = np.delete(rho, to_remove, axis=0)
        rho = np.delete(rho, to_remove, axis=1)

        f_theta, f_sig, f_tag = [np.delete(f_vals, to_remove)
                                 for f_vals in [f_theta, f_sig, f_tag]]

        f_COV = np.outer(f_sig, f_sig) * rho

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

    def _create_RV_repairs(self, rho_cost, rho_time, rho_cNt):

        # prepare the cost and time parts of the data separately
        ct_sig, ct_tag, ct_dkind = [np.array([]) for i in range(3)]
        for rho_target, name in zip([rho_cost, rho_time], ['cost', 'time']):
            
            f_sig, f_tag, f_dkind = [np.array([]) for i in range(3)]
            for c_id, (c_name, comp) in enumerate(self._FG_in.items()):

                d_sig, d_tag, d_dkind = [np.array([]) for i in range(3)]

                for dsg_i, DSG in comp['DSG_set'].items():
                    for ds_i, DS in DSG['DS_set'].items():
                        d_sig = np.append(d_sig,
                                          DS['repair_{}'.format(name)]['cov'])
                        d_dkind = np.append(d_dkind,
                                            DS['repair_{}'.format(name)][
                                                'distribution_kind'])
                        d_tag = np.append(d_tag,
                                          comp['ID'] + '-' + str(
                                              dsg_i) + '-' + str(
                                              ds_i) + '-{}'.format(name))

                for loc in comp['locations']:
                    for dir_ in np.unique(comp['directions']):
                        f_sig = np.append(f_sig, d_sig)
                        f_dkind = np.append(f_dkind, d_dkind)
                        f_tag = np.append(f_tag,
                                          [t + '-LOC-{}-DIR-{}'.format(loc,
                                                                       dir_)
                                           for t in d_tag])
            ct_sig = np.append(ct_sig, f_sig)
            ct_tag = np.append(ct_tag, f_tag)
            ct_dkind = np.append(ct_dkind, f_dkind)

        rho_c = self._create_correlation_matrix(rho_cost, c_target=-1,
                                          include_DSG=True,
                                          include_DS=True)
        rho_t = self._create_correlation_matrix(rho_time, c_target=-1,
                                          include_DSG=True,
                                          include_DS=True)

        dims = len(ct_tag)
        ct_rho = np.zeros((dims, dims))

        dims = dims // 2
        ct_rho[:dims, :dims] = rho_c
        ct_rho[dims:, dims:] = rho_t

        # if correlation between cost and time is considered, add that to the matrix
        if rho_cNt:
            rho_ct = np.zeros((dims, dims))
            np.fill_diagonal(rho_ct, 1.)
            ct_rho[:dims, dims:] = rho_ct
            ct_rho[dims:, :dims] = rho_ct

        ct_COV = np.outer(ct_sig, ct_sig) * ct_rho

        repair_RV = RandomVariable(ID=401,
                                   dimension_tags=ct_tag,
                                   distribution_kind=ct_dkind,
                                   theta=np.ones(len(ct_sig)),
                                   COV=ct_COV,
                                   corr_ref='post',
                                   truncation_limits=[np.zeros(len(ct_sig)),
                                                      None])

        return repair_RV

    def _create_RV_injuries(self, rho_target, rho_lvls):

        inj_lvls = self._inj_lvls

        # prepare the cost and time parts of the data separately
        full_theta, full_sig, full_tag = [np.array([]) for i in range(3)]
        for i_lvl in range(inj_lvls):

            f_theta, f_sig, f_tag = [np.array([]) for i in range(3)]
            for c_id, (c_name, comp) in enumerate(self._FG_in.items()):

                d_theta, d_sig, d_tag = [np.array([]) for i in range(3)]

                for dsg_i, DSG in comp['DSG_set'].items():
                    for ds_i, DS in DSG['DS_set'].items():
                        d_theta = np.append(d_theta, DS['injuries']['theta'][i_lvl])
                        d_sig = np.append(d_sig, DS['injuries']['cov'][i_lvl])
                        d_tag = np.append(d_tag,
                                          comp['ID'] + '-' + str(
                                              dsg_i) + '-' + str(
                                              ds_i) + '-{}'.format(i_lvl))

                for loc in comp['locations']:
                    for dir_ in np.unique(comp['directions']):
                        f_theta = np.append(f_theta, d_theta)
                        f_sig = np.append(f_sig, d_sig)
                        f_tag = np.append(f_tag,
                                          [t + '-LOC-{}-DIR-{}'.format(loc,
                                                                       dir_)
                                           for t in d_tag])

            full_theta = np.append(full_theta, f_theta)
            full_sig = np.append(full_sig, f_sig)
            full_tag = np.append(full_tag, f_tag)

        dims = len(full_tag)
        full_rho = np.zeros((dims, dims))
        dims = dims // inj_lvls

        # if correlation between different levels of severity is considered, add that to the matrix
        if rho_lvls:
            rho_i = self._create_correlation_matrix(rho_target, c_target=-1,
                                              include_DSG=True,
                                              include_DS=True)            
            for i in range(inj_lvls):
                for j in range(inj_lvls):
                    full_rho[i * dims:(i + 1) * dims,
                    j * dims:(j + 1) * dims] = rho_i

        # and now add the values around the main diagonal
        for i in range(inj_lvls):
            rho_i = self._create_correlation_matrix(rho_target, c_target=-1,
                                              include_DSG=True,
                                              include_DS=True)
            full_rho[i * dims:(i + 1) * dims, i * dims:(i + 1) * dims] = rho_i

            # finally, remove the unnecessary lines
        to_remove = np.where(full_theta == 0)[0]
        full_rho = np.delete(full_rho, to_remove, axis=0)
        full_rho = np.delete(full_rho, to_remove, axis=1)

        full_theta, full_sig, full_tag = [np.delete(f_vals, to_remove)
                                          for f_vals in
                                          [full_theta, full_sig, full_tag]]

        full_COV = np.outer(full_sig, full_sig) * full_rho

        tr_upper = 1. + (1. - full_theta) / full_theta

        injury_RV = RandomVariable(ID=402,
                                   dimension_tags=full_tag,
                                   distribution_kind='normal',
                                   theta=np.ones(len(full_sig)),
                                   COV=full_COV,
                                   corr_ref='post',
                                   truncation_limits=[np.zeros(len(full_sig)),
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
                demand_data.append(d_list[i]['raw_data'])
                d_tags.append(str(d_id) + 
                              '-LOC-' + str(d_list[i]['location']) + 
                              '-DIR-' + str(d_list[i]['direction']))
                det_lim = self._AIM_in['general']['detection_limits'][d_id]
                if det_lim is None:
                    det_lim = np.inf
                detection_limits.append([0., det_lim])

        detection_limits = np.transpose(np.asarray(detection_limits))
        demand_data = np.transpose(np.asarray(demand_data))
        
        # if more than one sample is provided
        if demand_data.shape[0] > 1:
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
        else:
            # Since we only have one data point, the best we can do is assume
            # it is the median of the multivariate distribution. The dispersion
            # is assumed to be negligible.
            dim = len(demand_data[0])
            if dim > 1:
                sig = np.ones(dim)*1e-4
                rho = np.zeros((dim,dim))
                np.fill_diagonal(rho, 1.0)
                COV = np.outer(sig,sig) * rho
            else:
                COV = np.asarray((1e-4)**2.)
                
            demand_RV = RandomVariable(ID=200, dimension_tags=d_tags,
                                       distribution_kind='lognormal',
                                       theta=demand_data[0],
                                       COV=COV)

        # if we want to add other sources of uncertainty, we will need to 
        # redefine the random variable
        if beta_added > 0.:
            # get the covariance matrix with added uncertainty
            COV_orig = demand_RV.COV
            if COV_orig.shape is not ():
                sig_orig = np.sqrt(np.diagonal(COV_orig))
                rho_orig = COV_orig / np.outer(sig_orig, sig_orig)
                sig_mod = np.sqrt(sig_orig ** 2. + beta_added ** 2.)
                COV_mod = np.outer(sig_mod, sig_mod) * rho_orig
            else:
                COV_mod = np.sqrt(COV_orig**2. + beta_added**2.)

            # redefine the random variable
            demand_RV = RandomVariable(ID=200,
                                       dimension_tags=demand_RV.dimension_tags,
                                       distribution_kind='lognormal',
                                       theta=demand_RV.theta,
                                       COV=COV_mod)

        return demand_RV

    def _create_fragility_groups(self):

        RVd = self._RV_dict
        DVs = self._AIM_in['decision_variables']

        # create a list for the fragility groups
        FG_dict = dict()

        for c_id, comp in self._FG_in.items():

            FG_ID = len(FG_dict.keys())

            # create a list for the performance groups
            performance_groups = []

            # one group for each of the stories prescribed by the user
            PG_locations = comp['locations']
            PG_directions = np.unique(comp['directions'])
            for loc in PG_locations:
                for dir_ in PG_directions:
                    PG_ID = 1000 + dir_*100 + FG_ID * 10 + loc
    
                    # get the quantity
                    QNT = RandomVariableSubset(
                        RVd['QNT'],
                        tags=[c_id + '-QNT-' + str(loc) + '-' + str(dir_), ])
    
                    # create the damage objects
                    # consequences are calculated on a performance group level
    
                    # create a list for the damage state groups and their tags
                    DSG_list = []
                    d_tags = []
                    for dsg_i, (DSG_ID, DSG) in enumerate(comp['DSG_set'].items()):
                        d_tags.append(c_id + '-' + DSG_ID)
    
                        # create a list for the damage states
                        DS_set = []
    
                        for ds_i, (DS_ID, DS) in enumerate(DSG['DS_set'].items()):
    
                            # create the consequence functions
                            if DVs['rec_cost']:
                                data = DS['repair_cost']
                                f_median = prep_bounded_linear_median_DV(
                                    **{k: data.get(k, None) for k in
                                       ('median_max', 'median_min',
                                        'quantity_lower', 'quantity_upper')})
                                cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + \
                                         '-cost' + \
                                         '-LOC-{}-DIR-{}'.format(loc, dir_)
                                CF_RV = RandomVariableSubset(RVd['DV_REP'],
                                                             tags=cf_tag)
                                CF_cost = ConsequenceFunction(DV_median=f_median,
                                                              DV_distribution=CF_RV)
                            else:
                                CF_cost = None
    
                            if DVs['rec_time']:
                                data = DS['repair_time']
                                f_median = prep_bounded_linear_median_DV(
                                    **{k: data.get(k, None) for k in
                                       ('median_max', 'median_min', 'quantity_lower',
                                        'quantity_upper')})
                                cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + \
                                         '-time' + \
                                         '-LOC-{}-DIR-{}'.format(loc, dir_)
                                CF_RV = RandomVariableSubset(RVd['DV_REP'],
                                                             tags=cf_tag)
                                CF_time = ConsequenceFunction(DV_median=f_median,
                                                              DV_distribution=CF_RV)
                            else:
                                CF_time = None
    
                            if DVs['red_tag']:
                                data = DS['red_tag']
                                if data['theta'] > 0:
                                    f_median = prep_constant_median_DV(data['theta'])
                                    cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + \
                                             '-LOC-{}-DIR-{}'.format(loc, dir_)
                                    CF_RV = RandomVariableSubset(RVd['DV_RED'],
                                                                 tags=cf_tag)
                                    CF_red_tag = ConsequenceFunction(DV_median=f_median,
                                                                     DV_distribution=CF_RV)
                                else:
                                    CF_red_tag = None
                            else:
                                CF_red_tag = None
    
                            if DVs['injuries']:
                                data = DS['injuries']
                                CF_inj_set = []
                                for inj_i, theta in enumerate(data['theta']):
                                    if theta > 0.:
                                        f_median = prep_constant_median_DV(theta)
                                        cf_tag = c_id + '-' + DSG_ID + '-' + DS_ID + \
                                                 '-{}-LOC-{}-DIR-{}'.format(inj_i, loc, dir_)
                                        CF_RV = RandomVariableSubset(RVd['DV_INJ'],
                                                                     tags=cf_tag)
                                        CF_inj_set.append(ConsequenceFunction(
                                            DV_median=f_median,
                                            DV_distribution=CF_RV))
                                    else:
                                        CF_inj_set.append(None)
                            else:
                                CF_inj_set = [None,]
    
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
                    CSG_this = np.where(comp['directions']==dir_)[0]
                    PG_weights = np.asarray(comp['csg_weights'])[CSG_this]
                    # normalize the weights
                    PG_weights /= sum(PG_weights)
                    for csg_id in CSG_this:
                        # assign the appropriate random variable to the fragility 
                        # function
                        ff_tags = [t + '-LOC-{}-CSG-{}'.format(loc, csg_id)
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
                                          csg_weights=PG_weights,
                                          direction=dir_
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
        
        sample_count = self._AIM_in['general']['realizations']
        
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
                np.array(POPin['weekend']['daily'])[
                    TIME.loc[weekends, 'hour'].values.astype(int)] *
                np.array(POPin['weekend']['monthly'])[
                    TIME.loc[weekends, 'month'].values.astype(int)])

        return POP

    def _calc_collapses(self):

        # filter the collapsed cases based on the demand samples
        collapsed_IDs = np.array([])
        for demand_ID, demand in self._EDP_dict.items():
            coll_df = pd.DataFrame()
            kind = demand_ID[:3]
            collapse_limit = self._AIM_in['general']['collapse_limits'][kind]
            if collapse_limit is not None:
                EDP_samples = demand.samples
                coll_df = EDP_samples[EDP_samples > collapse_limit]
            collapsed_IDs = np.concatenate(
                (collapsed_IDs, coll_df.index.values))

        # get a list of IDs of the collapsed cases
        collapsed_IDs = np.unique(collapsed_IDs).astype(int)

        COL = pd.DataFrame(np.zeros(self._AIM_in['general']['realizations']), 
                           columns=['COL', ])
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
                demand_ID = (FG._demand_type + 
                             '-LOC-' + str(PG._location) +
                             '-DIR-' + str(PG._direction+1))
                if demand_ID in self._EDP_dict.keys():
                    EDP_samples = self._EDP_dict[demand_ID].samples.loc[ncID]
                else:
                    # If the required demand is not available, then we are most
                    # likely analyzing a 3D structure using results from a 2D 
                    # simulation. The best thing we can do in that particular
                    # case is to use the EDP from the 0 direction for all other
                    # directions.
                    demand_ID = (FG._demand_type + 
                                 '-LOC-' + str(PG._location) + '-DIR-1')
                    EDP_samples = self._EDP_dict[demand_ID].samples.loc[ncID]
                    
                csg_w_list = PG._csg_weights

                for csg_i, csg_w in enumerate(csg_w_list): 
                    DSG_df = PG._FF_set[csg_i].DSG_given_EDP(EDP_samples)
                    
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

                FG_damages.iloc[:,pg_i * d_count:(pg_i + 1) * d_count] = \
                    FG_damages.mul(PG_qnt.iloc[:, 0], axis=0)

            DMG = pd.concat((DMG, FG_damages), axis=1)

        DMG.index = ncID
        
        # sort the columns to enable index slicing later
        DMG = DMG.sort_index(axis=1, ascending=True)

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
                        RED_samples.index = ncID

                        is_red = PG_DMG.loc[:, (FG._ID, PG_ID, d_tag)].sub(
                            RED_samples, axis=0)
                        
                        FG_RED.loc[:, (FG._ID, PG_ID, d_tag)] = (
                            is_red > 0.).astype(int)
                    else:
                        FG_RED.drop(labels=[(FG._ID, PG_ID, d_tag), ], axis=1,
                                    inplace=True)

            if FG_RED.size > 0:
                DV_RED = pd.concat((DV_RED, FG_RED), axis=1)

        # sort the columns to enable index slicing later
        DV_RED = DV_RED.sort_index(axis=1, ascending=True)

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

        if (RED_max is None) and (PID_max is not None):
            # we need to estimate residual drifts based on peak drifts
            RED_max = np.zeros(NC_samples)

            # based on Appendix C in FEMA P-58
            delta_y = self._AIM_in['general']['yield_drift']
            small = PID_max < delta_y
            medium = PID_max < 4 * delta_y
            large = PID_max >= 4 * delta_y

            RED_max[large] = PID_max[large] - 3 * delta_y
            RED_max[medium] = 0.3 * (PID_max[medium] - delta_y)
            RED_max[small] = 0.
        else:
            # If no drift data is available, then we cannot provide an estimate
            # of irrepairability. We assume that all non-collapse realizations 
            # are repairable in this case.
            return np.array([])
            
        # get the probabilities of irrepairability
        irrep_frag = self._AIM_in['general']['irrepairable_res_drift']
        RV_irrep = RandomVariable(ID=-1, dimension_tags=['RED_irrep', ],
                                  distribution_kind='lognormal',
                                  theta=irrep_frag['Median'],
                                  COV=irrep_frag['Sig'] ** 2.
                                  )
        RED_irrep = RV_irrep.sample_distribution(NC_samples)['RED_irrep'].values

        # determine if the realizations are repairable
        irrepairable = RED_max > RED_irrep
        irrepairable_IDs = ncID[np.where(irrepairable)[0]]

        return irrepairable_IDs

    def _calc_repair_cost_and_time(self):

        idx = pd.IndexSlice
        DVs = self._AIM_in['decision_variables']

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

                    TOT_qnt = DMG_by_FG_and_DS.loc[repID, (FG._ID, d_tag)]
                    PG_qnt = self._DMG.loc[repID, 
                                           (FG._ID, PG_ID, d_tag)]

                    # repair cost
                    if DVs['rec_cost']:
                        COST_samples = DS.unit_repair_cost(quantity=TOT_qnt)
                        DV_COST.loc[:,
                        (FG._ID, PG_ID, d_tag)] = COST_samples * PG_qnt
                    
                    if DVs['rec_time']:
                        # repair time
                        TIME_samples = DS.unit_reconstruction_time(quantity=TOT_qnt)
                        DV_TIME.loc[:,
                        (FG._ID, PG_ID, d_tag)] = TIME_samples * PG_qnt

        # sort the columns to enable index slicing later
        if DVs['rec_cost']:
            DV_COST = DV_COST.sort_index(axis=1, ascending=True)
        else:
            DV_COST = None
        if DVs['rec_time']:
            DV_TIME = DV_TIME.sort_index(axis=1, ascending=True)
        else:
            DV_TIME = None

        return DV_COST, DV_TIME

    def _calc_collapse_injuries(self):

        inj_lvls = self._inj_lvls

        # calculate casualties and injuries for the collapsed cases
        # generate collapse modes
        colID = self._ID_dict['collapse']
        C_samples = len(colID)
        
        if C_samples > 0:
            
            inj_lvls = self._inj_lvls
            coll_modes = self._AIM_in['collapse_modes']
            P_keys = [cmk for cmk in coll_modes.keys()]
            P_modes = [coll_modes[k]['w'] for k in P_keys]
    
            # create the DataFrame that collects the decision variables
            COL_INJ = pd.DataFrame(np.zeros((C_samples, inj_lvls + 1)),
                                   columns=[['CM',] + 
                                            ['INJ-{}'.format(i) for i in
                                             range(inj_lvls)]],
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
                CFAR = coll_modes[cmk]['affected_area']
                INJ = coll_modes[cmk]['injuries']
                for loc_i in range(len(CFAR)):
                    loc_label = 'LOC{}'.format(loc_i + 1)
                    if loc_label in P_sel.columns:
                        for inj_i in range(inj_lvls):
                            INJ_i = P_sel.loc[mode_IDs, loc_label] * CFAR[loc_i] * \
                                    INJ[inj_i]
                            COL_INJ.loc[mode_IDs, 'INJ-{}'.format(inj_i)] = (
                                COL_INJ.loc[mode_IDs, 'INJ-{}'.format(inj_i)] 
                                + INJ_i)

            return COL_INJ
        
        else:
            return None

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
                                      * DS._affected_area / 
                                      self._AIM_in['general']['plan_area'])

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

        # sort the columns to enable index slicing later
        for i in range(self._inj_lvls):
            DV_INJ_dict[i] = DV_INJ_dict[i].sort_index(axis=1, ascending=True)

        return DV_INJ_dict
