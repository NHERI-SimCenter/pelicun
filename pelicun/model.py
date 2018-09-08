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
This module has classes and methods that define and access the model used for
loss assessment.

.. rubric:: Contents

.. autosummary::

    FragilityFunction
    ConsequenceFunction
    DamageState
    DamageStateGroup
    FragilityGroup

"""

import numpy as np
from scipy.stats import norm, truncnorm
import pandas as pd
from .uq import RandomVariableSubset

class FragilityFunction(object):
    """
    Describes the relationship between asset response and damage.
    
    Asset response is characterized by a Demand value that represents an 
    engineering demand parameter (EDP). Only a scalar EDP is supported 
    currently. The damage is characterized by a set of DamageStateGroup (DSG)
    objects. For each DSG, the corresponding EDP limit (i.e. the EDP at which
    the asset is assumed to experience damage described by the DSG) is 
    considered uncertain; hence, it is described by a random variable. The 
    random variables that describe EDP limits for the set of DSGs are not 
    independent. 
    
    We assume that the EDP limit will be approximated by a normal or lognormal 
    distribution for each DSG and these variables together form a multivariate
    normal distribution. Following common practice, the correlation between
    variables is assumed perfect by default, but the framework allows the
    users to explore other, more realistic options. 

    Parameters
    ----------
    RVS: RandomVariableSubset
        A multidimensional random variable that might be defined as a subset
        of a bigger correlated group of variables or a complete set of 
        variables created only for this Fragility Function (FF). The number of 
        dimensions shall be equal to the number of DSGs handled by the
        FF.
        
    """

    def __init__(self, RVS):
        self._RVS = RVS

    def P_exc(self, EDP, DSG_ID):
        """
        Return the probability of damage exceedance.
        
        Calculate the probability of exceeding the damage corresponding to the 
        DSG identified by the DSG_ID conditioned on a particular EDP value.

        Parameters
        ----------
        EDP: float scalar or ndarray
            Single EDP or numpy array of EDP values.
        DSG_ID: int
            Identifies the conditioning DSG. The DSG numbering is 1-based, 
            because zero typically corresponds to the undamaged state.

        Returns
        -------
        P_exc: float scalar or ndarray
            DSG exceedance probability at the given EDP point(s).
        """
        
        EDP = np.asarray(EDP, dtype=np.float64)
        nvals = EDP.size

        # The exceedance probability corresponding to no damage is 1.
        # Although this is trivial, returning 1 for DSG_ID=0 makes using P_exc
        # more convenient.
        if DSG_ID == 0:
            P_exc = np.ones(EDP.size)
        else:
            # prepare the limits for the density calculation
            ndims = np.asarray(self._RVS.tags).size
            
            limit_list = np.full((ndims, nvals), None)
            limit_list[DSG_ID - 1:] = EDP
            limit_list = np.transpose(limit_list)
    
            # get the pointer for the orthotope density function to save time
            RVS_od = self._RVS.orthotope_density
            P_exc = 1. - np.asarray([RVS_od(lower=limit)[0] 
                                     for limit in limit_list])

        # if EDP was a scalar, make sure that the result is also a scalar
        if EDP.size == 1:
            return P_exc[0]
        else:
            return P_exc
        
    def DSG_given_EDP(self, EDP, force_resampling=False):
        """
        Given an EDP, get a damage level based on the fragility function.
        
        The damage is evaluated by sampling the joint distribution of 
        fragilities corresponding to all possible damage levels and checking
        which damage level the given EDP falls into. This approach allows for
        efficient damage state evaluation for a large number of EDP 
        realizations. 
        
        Parameters
        ----------
        EDP: float scalar or ndarray or Series
            Single EDP, or numpy array or pandas Series of EDP values.
        force_resampling: bool, optional, default: False
            If True, the probability distribution is resampled before 
            evaluating the damage for each EDP. This is not recommended if the
            fragility functions are correlated with other sources of 
            uncertainty because those variables will also be resampled in this
            case. If False, which is the default approach, we assume that
            the random variable has already been sampled and the number of 
            samples greater or equal to the number of EDP values.

        Returns
        -------
        DSG_ID: Series
            Identifies the damage that corresponds to the given EDP. A DSG_ID
            of 0 means no damage. 

        """
        # get the number of samples needed
        nsamples = EDP.size

        # if there are no samples or resampling is forced, then sample the 
        # distribution first
        if force_resampling or (self._RVS.samples is None):
            self._RVS.sample_distribution(sample_size=nsamples)

        # if the number of samples is not sufficiently large, raise an error
        if self._RVS.samples.shape[0] < nsamples:
            raise ValueError(
                'Damage evaluation requires at least as many samples of the '
                'joint distribution defined by the fragility functions as '
                'many EDP values are provided to the DSG_given_EDP function. '
                'You might want to consider setting force_resampling to True '
                'or sampling the distribution before calling the DSG_given_EDP '
                'function.')

        samples = self._RVS.samples
        EDP = pd.Series(EDP, name='EDP')

        nstates = samples.columns.values.size

        EXC = samples.sub(EDP, axis=0) < 0.

        DSG_ID = pd.Series(np.zeros(len(samples.index)), name='DSG_ID', 
                        dtype=np.int)

        for s in range(nstates):
            DSG_ID[EXC.iloc[:,s]] = s + 1

        return DSG_ID
        
class ConsequenceFunction(object):
    """
    Indicates the distribution of quantified consequences of a component, an
    element, or the system reaching a given damage state (DS). Consequences 
    can be reconstruction cost, repair time, casualties, injuries, etc. Their
    distribution might depend on the quantity of damaged components. 

    Parameters
    ----------

    median_kind: {'fixed','bounded_linear'}
        The fixed option corresponds to a consequence distribution with a fixed 
        median value; in the bounded_linear case the median is a linear 
        function of the component quantity within maximum and minimum bounds
        and a negative slope.
    distribution_kind: {'normal', 'lognormal','truncated_normal'}
        The probability distribution of the consequence quantities. The 
        'truncated_normal' shall be used when a lower or upper limit is 
        desired. An example of such a case is the evaluation of red tag 
        probability.
    standard_deviation: float, optional
        The standard deviation or the logarithmic standard deviation of the 
        distribution of consequence quantities.
    coefficient_of_variation: float, optional
        The coefficient of variation of the distribution of consequence 
        quantities. This option is only available for normal distributions.
    median: float, optional
        The median consequence value for cases with fixed median.
    median_max: float, optional
    median_min: float, optional
        Minimum and maximum consequence limits that define the bounded_linear 
        median consequence function.
    quantity_lower_bound: float, optional
    quantity_upper_bound: float, optional
        Lower and upper bounds of component quantity that define the 
        bounded_linear median consequence function.
    min_value: float, optional
        Lower bound of consequence values for the truncated_normal distribution
        function
    max_value: float, optional
        Upper bound of consequence values for the truncated_normal
        distribution function.
    """

    def __init__(self, median_kind, distribution_kind,
                 standard_deviation=None, coefficient_of_variation=None,
                 median=None, median_max=None, median_min=None,
                 quantity_lower_bound=None, quantity_upper_bound=None,
                 min_value=-np.inf, max_value=np.inf):

        self._median_kind = median_kind
        self._distribution_kind = distribution_kind
        self._std = standard_deviation
        self._cov = coefficient_of_variation
        self._median = median
        self._median_max = median_max
        self._median_min = median_min
        self._quantity_lower_bound = quantity_lower_bound
        self._quantity_upper_bound = quantity_upper_bound
        self._max_value = max_value
        self._min_value = min_value

        # perform some basic checks
        if self._median_kind == 'fixed' and self._median is None:
            raise ValueError(
                "The value of 'median' needs to be specified for a consequence "
                "function with 'fixed' median."
            )

        if (self._median_kind == 'bounded_linear'
            and (
                self._median_max is None
                or self._median_min is None
                or self._quantity_lower_bound is None
                or self._quantity_upper_bound is None
            )
        ):
            raise ValueError(
                "The values of 'median_max', 'median_min', "
                "'quantity_lower_bound', and 'quantity_upper_bound' needs to be"
                "specified for a consequence function with 'bounded_linear'"
                "median."
            )

        if (self._distribution_kind != 'truncated_normal'
            and (
                self._max_value != np.inf
                or self._min_value != -np.inf
            )
        ):
            raise ValueError(
                "The 'max_value' and 'min_value' parameters are only allowed "
                "for consequence functions with a truncated normal "
                "distribution."
            )

        if (self._std is None and self._cov is None):
            raise ValueError(
                "Either the standard deviation or the coefficient of variation"
                "of the consequence distribution needs to be specified."
            )

        if (self._std is not None and self._cov is not None):
            raise ValueError(
                "Only one of the (standard deviation, coefficient of variation)"
                "pair can be specified to avoid contradicting values."
            )

        if (self._distribution_kind == 'lognormal' and self._cov is not None):
            raise ValueError(
                "The dispersion of lognormally distributed consequence "
                "quantities needs to be specified using their standard "
                "deviation."
            )

    def median(self, quantity=None):
        """
        Return the median consequence value.

        The median consequence corresponds to the component damage state (DS). 
        If the consequence depends on the quantity of damaged components, the 
        total quantity shall be specified through the quantity parameter.

        Parameters
        ----------
        quantity: float scalar or ndarray, optional
            Total quantity of damaged components that determines the magnitude 
            of median consequence. Not needed for consequence functions with
            a fixed median.

        Returns
        -------
        median: float scalar or ndarray
            A single scalar for fixed median; a scalar or an array depending on
            the shape of the quantity parameter for bounded_linear median.

        """
        if self._median_kind == 'fixed':
            return self._median

        elif self._median_kind == 'bounded_linear':

            if quantity is None:
                raise ValueError(
                    'Consequence function with bounded linear median called '
                    'without specifying the quantity of damaged components')

            q_array = np.asarray(quantity, dtype=np.float64)

            # calculate the median consequence given the quantity of damaged
            # components
            output = np.interp(
                q_array,
                [self._quantity_lower_bound, self._quantity_upper_bound],
                [self._median_max, self._median_min])

            return output

    def unit_consequence(self, quantity=None, sample_size=None):
        """
        Sample the consequence distribution and return a unit consequence.

        The unit consequence corresponds to the component damage state (DS). 
        It shall be multiplied by the quantity of damaged components. If the 
        consequence depends on the quantity of damaged components, the total
        quantity shall be specified through the quantity parameter.

        Parameters
        ----------
        quantity: float scalar or ndarray, optional, default: None
            Total quantity of damaged components that determines the magnitude 
            of median consequence. Not needed for consequence functions with
            a fixed median.
        sample_size: int, optional, default: None
            Number of samples drawn from the consequence distribution. The
            default value yields one sample.

        Returns
        -------
        unit_consequence: float scalar or ndarray
            Unit consequence samples.

        """
        median = self.median(quantity=quantity)

        if type(median) is np.ndarray:
            if sample_size is not None:
                sample_size = (sample_size, *median.shape)
            else:
                sample_size = median.shape
            # print(sample_size)

        if self._std is not None:
            stdev = self._std
        else:
            stdev = self._cov * median

        if self._distribution_kind == 'lognormal':
            median = np.log(median)

        output = truncnorm.rvs(a=(self._min_value - median) / stdev,
                               b=(self._max_value - median) / stdev,
                               loc=median,
                               scale=stdev,
                               size=sample_size,
                               )

        if self._distribution_kind == 'lognormal':
            return np.exp(output)
        else:
            return output


class DamageState(object):
    """
    Characterizes one type of damage that corresponds to a particular Damage 
    State Group (DSG). The occurrence of damage is evaluated at the DSG. The DS 
    describes one of the possibly several types of damages that belong to the 
    same DSG. 

    Parameters
    ----------

    ID:int
    weight: float, optional, default: 1.0        
        Describes the probability of DS occurrence, conditioned on the damage
        being in the DSG linked to this DS. This information is only used for
        DSGs with multiple DS corresponding to them. The weights of the set of 
        DS shall sum up to 1.0 if they are mutually exclusive. When the set of
        DS occur simultaneously, the sum of weights typically exceeds 1.0.
    description: str, optional
        Provides a short description of the damage state.
    repair_cost_CF: ConsequenceFunction, optional
        A consequence function that describes the cost necessary to restore the
        component to its pre-disaster condition.    
    reconstruction_time_CF: ConsequenceFunction, optional
        A consequence function that describes the time, necessary to repair the 
        damaged component to its pre-disaster condition.
    injuries_CF_set: ConsequenceFunction array, optional
        A set of consequence functions; each describes the number of people 
        expected to experience injury of a particular severity when the 
        component is in this DS. Any number of injury-levels can be considered. 
    red_tag_CF: ConsequenceFunction, optional
        A consequence function that describes the proportion of components 
        (within a Performance Group) that needs to be damaged to trigger an 
        unsafe placard (i.e. red tag) for the building during post-disaster 
        inspection. 

    """

    def __init__(self, ID, weight=1.0, description='',
                 repair_cost_CF=None, reconstruction_time_CF=None,
                 injuries_CF_set=None, red_tag_CF=None):
        self._ID = int(ID)
        self._weight = float(weight)
        self._description = description
        self._repair_cost_CF = repair_cost_CF
        self._reconstruction_time_CF = reconstruction_time_CF
        self._injuries_CF_set = injuries_CF_set
        self._red_tag_CF = red_tag_CF

    @property
    def description(self):
        """
        Return the damage description.
        """
        return self._description

    @property
    def weight(self):
        """
        Return the weight of DS among the set of damage states in the DSG.
        """
        return self._weight

    def unit_repair_cost(self, quantity=None, sample_size=None):
        """
        Sample the repair cost distribution and return the unit repair costs.

        The unit repair costs shall be multiplied by the quantity of damaged 
        components to get the total repair costs for the components in this DS.       

        Parameters
        ----------
        quantity: float scalar or ndarray, optional, default: None
            Total quantity of damaged components that determines the magnitude
            of median repair cost. Not used for repair cost models with fixed
            median.
        sample_size: int, optional, default: None
            Number of samples drawn from the repair cost distribution. The 
            default value yields one sample.

        Returns
        -------
        unit_repair_cost: float scalar or ndarray
            Unit repair cost samples.

        """
        output = self._repair_cost_CF.unit_consequence(quantity=quantity,
                                                       sample_size=sample_size)

        return output

    def unit_reconstruction_time(self, quantity=None, sample_size=None):
        """
        Sample the reconstruction time distribution and return the unit 
        reconstruction times.

        The unit reconstruction times shall be multiplied by the quantity of 
        damaged components to get the total reconstruction time for the 
        components in this DS.       

        Parameters
        ----------
        quantity: float scalar or ndarray, optional, default: None
            Total quantity of damaged components that determines the magnitude
            of median reconstruction time. Not used for reconstruction time 
            models with fixed median.
        sample_size: int, optional, default: None
            Number of samples drawn from the reconstruction time distribution. 
            The default value yields one sample.

        Returns
        -------
        unit_reconstruction_time: float scalar or ndarray
            Unit reconstruction time samples.

        """
        output = self._reconstruction_time_CF.unit_consequence(
            quantity=quantity,
            sample_size=sample_size)

        return output

    def unit_red_tag(self, sample_size=None):
        """
        Sample the red tag consequence function and return the proportion of 
        components that needs to be damaged to trigger a red tag.

        The red tag consequence function is assumed to have a fixed median 
        value that does not depend on the quantity of damaged components.     

        Parameters
        ----------
        sample_size: int, optional, default: None
            Number of samples drawn from the red tag consequence distribution. 
            The default value yields one sample.

        Returns
        -------
        red_tag_trigger: float scalar or ndarray
            Samples of damaged component proportions that trigger a red tag.

        """
        output = self._red_tag_CF.unit_consequence(sample_size=sample_size)

        return output

    def unit_injuries(self, severity_level=0, sample_size=None):
        """
        Sample the injury consequence function that corresponds to the 
        specified level of severity and return the injuries per component unit.

        The injury consequence function is assumed to have a fixed median 
        value that does not depend on the quantity of damaged components (i.e.
        the number of injuries per component unit does not change with the 
        quantity of components.)     

        Parameters
        ----------
        severity_level: int, optional, default: 1
            Identifies which injury consequence to sample. The indexing of 
            severity levels is zero-based.       
        sample_size: int, optional, default: None
            Number of samples drawn from the injury consequence distribution. 
            The default value yields one sample.

        Returns
        -------
        unit_injuries: float scalar or ndarray
            Unit injury samples.

        """
        CF = self._injuries_CF_set[severity_level]
        output = CF.unit_consequence(sample_size=sample_size)

        return output


class DamageStateGroup(object):
    """
    Collects component damages that are controlled by the same kind of EDP and 
    occur at the same EDP magnitude. The damages are characterized by Damage
    States (DS). A Damage State Group (DSG) might have only a single DS in the
    simplest case.

    Parameters
    ----------
    ID: int
    DS_set: DamageState array
    DS_set_kind: {'single', 'mutually_exclusive', 'simultaneous'}
        Specifies the relationship among the DS in the set. When only one DS is
        defined, use the 'single' option to improve calculation efficiency. 
        When multiple DS are present, the 'mutually_exclusive' option assumes
        that the occurrence of one DS precludes the occurrence of another DS.
        In such a case, the weights of the DS in the set shall sum up to 1.0.
        In a 'simultaneous' case the DS are independent and unrelated. Hence,
        they can occur at the same time and at least one of them has to occur.
    fragility_function: FragilityFunction
        This fragility function describes the probability that the damage in 
        the component will meet or exceed the damages described by the DS_set. 
    description: str, optional, default: ''
        Provides a short description of the damage state group.
    """

    def __init__(self, ID, DS_set, DS_set_kind, fragility_function,
                 description=''):
        self._ID = ID
        self._DS_set = DS_set
        self._DS_set_kind = DS_set_kind
        self._fragility_function = fragility_function
        self._description = description

    @property
    def description(self):
        """
        Return the damage state group description.
        """
        return self._description

    def P_exc(self, EDP):
        """
        This is a convenience function that provides a shortcut to 
        fragility_function.P_exc(). It calculates the DSG exceedance 
        probability given a particular EDP value.

        Parameters
        ----------
        EDP: float scalar or ndarray
            Single EDP or numpy array of EDP values.

        Returns
        -------
        P_exc: float scalar or ndarray
            DSG exceedance probability at the given EDP point(s).
        """
        return self._fragility_function.P_exc(EDP)


class FragilityGroup(object):
    """
    Characterizes a set of structural or non-structural components that have
    similar construction characteristics, similar potential modes of damage,
    similar probability of incurring those modes of damage, and similar 
    potential consequences resulting from their damage.

    Parameters
    ----------
    ID: int
    kind: {'structural','non-structural'}
        Defines the type of components in the Fragility Group (FG).
    demand_type: {'PID', 'PFA', 'PSD', 'PSA', 'ePGA', 'PGD'}
        The type of Engineering Demand Parameter (EDP) that controls the damage 
        of the components in the FG. See Demand for acronym descriptions.
    DSG_set: DamageStateGroup array
        A set of sequential Damage State Groups that describe the plausible set
        of damage states of the components in the FG.
    directional: bool, optional, default: True
        Determines whether the components in the FG are sensitive to the 
        directionality of the EDP.   
    correlation: bool, optional, default: True
        Determines whether the components within a Performance Group (PG) will
        have correlated or uncorrelated damage. Correlated damage means that 
        all components will have the same damage state. In the uncorrelated 
        case, each component in the performance group will have its damage
        state evaluated independently. Correlated damage reduces the required 
        computational effort for the calculation. Incorrect correlation 
        modeling will only slightly affect the mean estimates, but might 
        significantly change their dispersion. 
    demand_location_offset: int, optional, default: 0
        Indicates if the location for the demand shall be different from the 
        location of the components. Damage to components of the ceiling, for 
        example, is controlled by demands on the floor above the one that the
        components belong to. This can be indicated by setting the 
        demand_location_offset to 1 for such an FG.
    incomplete: bool, optional, default: False
        Indicates that the FG information is not complete and corresponding
        results shall be treated with caution.
    description: str, optional, default: ''
        Provides a short description of the fragility group.
    """

    def __init__(self, ID, kind, demand_type, DSG_set,
                 directional=True, correlation=True, demand_location_offset=0,
                 incomplete=False, description=''):
        self._ID = ID
        self._kind = kind
        self._demand_type = demand_type
        self._DSG_set = DSG_set
        self._directional = directional
        self._correlation = correlation
        self._demand_location_offset = demand_location_offset
        self._incomplete = incomplete
        self._description = description

    @property
    def description(self):
        """
        Return the fragility group description.
        """
        return self._description