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
    PerformanceGroup
    FragilityGroup

    prep_constant_median_DV
    prep_bounded_linear_median_DV

"""

import numpy as np
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
    EDP_limit: RandomVariableSubset
        A multidimensional random variable that might be defined as a subset
        of a bigger correlated group of variables or a complete set of
        variables created only for this Fragility Function (FF). The number of
        dimensions shall be equal to the number of DSGs handled by the
        FF.

    """

    def __init__(self, EDP_limit):
        self._EDP_limit = EDP_limit

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
            ndims = np.asarray(self._EDP_limit.tags).size

            limit_list = np.full((ndims, nvals), None)
            limit_list[DSG_ID - 1:] = EDP
            limit_list = np.transpose(limit_list)

            # get the pointer for the orthotope density function to save time
            RVS_od = self._EDP_limit.orthotope_density
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
        if force_resampling or (self._EDP_limit.samples is None):
            self._EDP_limit.sample_distribution(sample_size=nsamples)

        # if the number of samples is not sufficiently large, raise an error
        if self._EDP_limit.samples.shape[0] < nsamples:
            raise ValueError(
                'Damage evaluation requires at least as many samples of the '
                'joint distribution defined by the fragility functions as '
                'many EDP values are provided to the DSG_given_EDP function. '
                'You might want to consider setting force_resampling to True '
                'or sampling the distribution before calling the DSG_given_EDP '
                'function.')

        samples = self._EDP_limit.samples

        if type(EDP) not in [pd.Series, pd.DataFrame]:
            EDP = pd.Series(EDP, name='EDP')

        nstates = samples.columns.values.size

        samples = samples.loc[EDP.index,:]

        sample_cols = samples.columns
        col_order = np.argsort(sample_cols)
        ordered_cols = sample_cols[col_order]
        samples = samples[ordered_cols]

        EXC = samples.sub(EDP, axis=0) < 0.

        DSG_ID = pd.Series(np.zeros(len(samples.index)), name='DSG_ID',
                           index=samples.index, dtype=np.int)

        for s in range(nstates):
            DSG_ID[EXC.iloc[:,s]] = s + 1

        return DSG_ID

def prep_constant_median_DV(median):
    """
    Returns a constant median Decision Variable (DV) function.

    Parameters
    ----------
    median: float
        The median DV for a consequence function with fixed median.

    Returns
    -------
    f: callable
        A function that returns the constant median DV for all component
        quantities.
    """
    def f(quantity):
        return median

    return f

def prep_bounded_linear_median_DV(median_max, median_min, quantity_lower,
                                  quantity_upper):
    """
    Returns a bounded linear median Decision Variable (DV) function.

    The median DV equals the min and max values when the quantity is
    outside of the prescribed quantity bounds. When the quantity is within the
    bounds, the returned median is calculated by a linear function with a
    negative slope between max and min values.

    Parameters
    ----------
    median_max: float, optional
    median_min: float, optional
        Minimum and maximum limits that define the bounded_linear median DV
        function.
    quantity_lower: float, optional
    quantity_upper: float, optional
        Lower and upper bounds of component quantity that define the
        bounded_linear median DV function.

    Returns
    -------
    f: callable
        A function that returns the median DV given the quantity of damaged
        components.
    """
    def f(quantity):
        if quantity is None:
            raise ValueError(
                'A bounded linear median Decision Variable function called '
                'without specifying the quantity of damaged components')

        q_array = np.asarray(quantity, dtype=np.float64)

        # calculate the median consequence given the quantity of damaged
        # components
        output = np.interp(q_array,
                           [quantity_lower, quantity_upper],
                           [median_max, median_min])

        return output

    return f

class ConsequenceFunction(object):
    """
    Describes the relationship between damage and a decision variable.

    Indicates the distribution of a quantified Decision Variable (DV)
    conditioned on a component, an element, or the system reaching a given
    damage state (DS). DV can be reconstruction cost, repair time, casualties,
    injuries, etc. Its distribution might depend on the quantity of damaged
    components.

    Parameters
    ----------

    DV_median: callable
        Describes the median DV as an f(quantity) function of the total
        quantity of damaged components. Use the prep_constant_median_DV, and
        prep_bounded_linear_median_DV helper functions to conveniently
        prescribe the typical FEMA P-58 functions.
    DV_distribution: RandomVariableSubset
        A one-dimensional random variable (or a one-dimensional subset of a
        multi-dimensional random variable) that characterizes the uncertainty
        in the DV. The distribution shall be normalized by the median DV (i.e.
        the RVS is expected to have a unit median). Truncation can be used to
        prescribe lower and upper limits for the DV, such as the (0,1) domain
        needed for red tag evaluation.

    """

    def __init__(self, DV_median, DV_distribution):

        self._DV_median = DV_median
        self._DV_distribution = DV_distribution

    def median(self, quantity=None):
        """
        Return the value of the median DV.

        The median DV corresponds to the component damage state (DS). If the
        damage consequence depends on the quantity of damaged components, the
        total quantity of damaged components shall be specified through the
        quantity parameter.

        Parameters
        ----------
        quantity: float scalar or ndarray, optional
            Total quantity of damaged components that determines the magnitude
            of median DV. Not needed for consequence functions with a fixed
            median DV.

        Returns
        -------
        median: float scalar or ndarray
            A single scalar for fixed median; a scalar or an array depending on
            the shape of the quantity parameter for bounded_linear median.

        """
        return self._DV_median(quantity)

    def sample_unit_DV(self, quantity=None, sample_size=1,
                       force_resampling=False):
        """
        Sample the decision variable quantity per component unit.

        The Unit Decision Variable (UDV) corresponds to the component Damage
        State (DS). It shall be multiplied by the quantity of damaged
        components to get the total DV that corresponds to the quantity of the
        damaged components in the asset. If the DV depends on the total
        quantity of damaged components, that value shall be specified through
        the quantity parameter.

        Parameters
        ----------
        quantity: float scalar, ndarray or Series, optional, default: None
            Total quantity of damaged components that determines the magnitude
            of median DV. Not needed for consequence functions with a fixed
            median DV.
        sample_size: int, optional, default: 1
            Number of samples drawn from the DV distribution. The default value
            yields one sample. If quantity is an array with more than one
            element, the sample_size parameter is ignored.
        force_resampling: bool, optional, default: False
            If True, the DV distribution (and the corresponding RV if there
            are correlations) is resampled even if there are samples already
            available. This is not recommended if the DV distribution is
            correlated with other sources of uncertainty because those
            variables will also be resampled in this case. If False, which is
            the default approach, we assume that the random variable has
            already been sampled and the number of samples is greater or equal
            to the number of samples requested.

        Returns
        -------
        unit_DV: float scalar or ndarray
            Unit DV samples.

        """
        # get the median DV conditioned on the provided quantities
        median = self.median(quantity=np.asarray(quantity))

        # if the distribution is None, then there is no uncertainty in the DV
        # and the median values are the samples
        if self._DV_distribution is None:
            return median

        else:
            # if there are more than one quantities defined, infer the number of
            # samples from the number of quantities
            if quantity is not None:
                if type(quantity) not in [pd.Series, pd.DataFrame]:
                    quantity = pd.Series(quantity, name='QNT')

                if quantity.size > 1:
                    sample_size = quantity.size
                elif sample_size > 1:
                    quantity = pd.Series(np.ones(sample_size) * quantity.values,
                                         name='QNT')

            # if there are no samples or resampling is forced, then sample the
            # distribution first
            if (force_resampling or
                (self._DV_distribution.samples is None)):
                self._DV_distribution.sample_distribution(sample_size=sample_size)

            # if the number of samples is not sufficiently large, raise an error
            if self._DV_distribution.samples.shape[0] < sample_size:
                raise ValueError(
                    'Consequence evaluation requires at least as many samples of '
                    'the Decision Variable distribution as many samples are '
                    'requested or as many quantity values are provided to the '
                    'sample_unit_DV function. You might want to consider setting '
                    'force_resampling to True or sampling the distribution before '
                    'calling the sample_unit_DV function.')

            # get the samples
            if quantity is not None:
                samples = self._DV_distribution.samples.loc[quantity.index]
            else:
                samples = self._DV_distribution.samples.iloc[:sample_size]
            samples = samples * median

            return samples


class DamageState(object):
    """
    Characterizes one type of damage that corresponds to a particular DSG.

    The occurrence of damage is evaluated at the DSG. The DS describes one of
    the possibly several types of damages that belong to the same DSG and the
    consequences of such damage.

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
    affected_area: float, optional, default: 0.
        Defines the area over which life safety hazards from this DS exist.
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
                 injuries_CF_set=None, affected_area=0., red_tag_CF=None):
        self._ID = int(ID)
        self._weight = float(weight)
        self._description = description
        self._repair_cost_CF = repair_cost_CF
        self._reconstruction_time_CF = reconstruction_time_CF
        self._injuries_CF_set = injuries_CF_set
        self._affected_area = affected_area
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

    def unit_repair_cost(self, quantity=None, sample_size=1, **kwargs):
        """
        Sample the repair cost distribution and return the unit repair costs.

        The unit repair costs shall be multiplied by the quantity of damaged
        components to get the total repair costs for the components in this DS.

        Parameters
        ----------
        quantity: float scalar, ndarray or Series, optional, default: None
            Total quantity of damaged components that determines the median
            repair cost. Not used for repair cost models with fixed median.
        sample_size: int, optional, default: 1
            Number of samples drawn from the repair cost distribution. The
            default value yields one sample.

        Returns
        -------
        unit_repair_cost: float scalar or ndarray
            Unit repair cost samples.

        """
        output = None
        if self._repair_cost_CF is not None:
            output = self._repair_cost_CF.sample_unit_DV(quantity=quantity,
                                                         sample_size=sample_size,
                                                         **kwargs)

        return output

    def unit_reconstruction_time(self, quantity=None, sample_size=1,
                                 **kwargs):
        """
        Sample the reconstruction time distribution and return the unit
        reconstruction times.

        The unit reconstruction times shall be multiplied by the quantity of
        damaged components to get the total reconstruction time for the
        components in this DS.

        Parameters
        ----------
        quantity: float scalar, ndarray or Series, optional, default: None
            Total quantity of damaged components that determines the magnitude
            of median reconstruction time. Not used for reconstruction time
            models with fixed median.
        sample_size: int, optional, default: 1
            Number of samples drawn from the reconstruction time distribution.
            The default value yields one sample.

        Returns
        -------
        unit_reconstruction_time: float scalar or ndarray
            Unit reconstruction time samples.

        """
        output = None
        if self._reconstruction_time_CF is not None:
            output = self._reconstruction_time_CF.sample_unit_DV(
                quantity=quantity,
                sample_size=sample_size, **kwargs)

        return output

    def red_tag_dmg_limit(self, sample_size=1, **kwargs):
        """
        Sample the red tag consequence function and return the proportion of
        components that needs to be damaged to trigger a red tag.

        The red tag consequence function is assumed to have a fixed median
        value that does not depend on the quantity of damaged components.

        Parameters
        ----------
        sample_size: int, optional, default: 1
            Number of samples drawn from the red tag consequence distribution.
            The default value yields one sample.

        Returns
        -------
        red_tag_trigger: float scalar or ndarray
            Samples of damaged component proportions that trigger a red tag.

        """
        output = None
        if self._red_tag_CF is not None:
            output = self._red_tag_CF.sample_unit_DV(sample_size=sample_size,
                                                     **kwargs)

        return output

    def unit_injuries(self, severity_level=0, sample_size=1, **kwargs):
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
        sample_size: int, optional, default: 1
            Number of samples drawn from the injury consequence distribution.
            The default value yields one sample.

        Returns
        -------
        unit_injuries: float scalar or ndarray
            Unit injury samples.

        """


        output = None
        if len(self._injuries_CF_set) > severity_level:
            CF = self._injuries_CF_set[severity_level]
            if CF is not None:
                output = CF.sample_unit_DV(sample_size=sample_size, **kwargs)

        return output


class DamageStateGroup(object):
    """
    A set of similar component damages that are controlled by the same EDP.

    Damages are described in detail by the set of Damage State objects.
    Damages in a DSG are assumed to occur at the same EDP magnitude. A Damage
    State Group (DSG) might have only a single DS in the simplest case.

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
    """

    def __init__(self, ID, DS_set, DS_set_kind):
        self._ID = ID
        self._DS_set = DS_set
        self._DS_set_kind = DS_set_kind

class PerformanceGroup(object):
    """
    A group of similar components that experience the same demands.

    FEMA P-58: Performance Groups (PGs) are a sub-categorization of fragility
    groups. A performance group is a subset of fragility group components that
    are subjected to the same demands (e.g. story drift, floor acceleration,
    etc.).

    In buildings, most performance groups shall be organized by story level.
    There is no need to separate performance groups by direction, because the
    direction of components within a group can be specified during definition,
    and it will be taken into consideration in the analysis.

    Parameters
    ----------
    ID: int
    location: int
        Identifies the location of the components that belong to the PG. In a
        building, location shall typically refer to the story of the building.
        The location assigned to each PG shall be in agreement with the
        locations assigned to the Demand objects.
    quantity: RandomVariableSubset
        Specifies the quantity of components that belong to this PG.
        Uncertainty in component quantities is considered by assigning a
        random variable to this property.
    fragility_functions: FragilityFunction list
        Each fragility function describes the probability that the damage in
        a subset of components will meet or exceed the damages described by
        each damage state group in the DSG_set. Each is a multi-dimensional
        function if there is more than one DSG. The number of functions shall
        match the number of subsets defined by the `csg_weights` parameter.
    DSG_set: DamageStateGroup array
        A set of sequential Damage State Groups that describe the plausible set
        of damage states of the components in the FG.
    csg_weights: float ndarray, optional, default: [1.0]
        Identifies subgroups of components within a PG, each of which have
        perfectly correlated behavior. Correlation between the damage and
        consequences among subgroups is controlled by the `correlation`
        parameter of the FragilityGroup that the PG belongs to. Note that if
        the components are assumed to have perfectly correlated behavior at the
        PG level, assigning several subgroups to the PG is unnecessary. This
        input shall be a list of weights that are applied to the quantity
        of components to define the amount of components in each subgroup. The
        sum of assigned weights shall be 1.0.
    directions: int ndarray, optional, default: [0]
        Identifies the direction of each subgroup of components within the PG.
        The number of directions shall be identical to the number of
        csg_weights assigned. In buildings, directions typically correspond to
        the orientation of components in plane. Hence, using 0 or 1 to identify
        'X' or 'Y' is recommended. These directions shall be in agreement with
        the directions assigned to Demand objects.
    """

    def __init__(self, ID, location, quantity, fragility_functions, DSG_set,
                 csg_weights=[1.0], direction=0):
        self._ID = ID
        self._location = location
        self._quantity = quantity
        if type(fragility_functions) == FragilityFunction:
            self._FF_set = [fragility_functions,]
        else:
            self._FF_set = fragility_functions
        self._DSG_set = DSG_set
        self._csg_weights = csg_weights
        self._direction = direction

    def P_exc(self, EDP, DSG_ID):
        """
        This is a convenience function that provides a shortcut to
        fragility_function.P_exc(). It calculates the exceedance probability
        of a given DSG conditioned on the provided EDP value(s). The fragility
        functions assigned to the first subset are used for this calculation
        because P_exc shall be identical among subsets.

        Parameters
        ----------
        EDP: float scalar or ndarray
            Single EDP or numpy array of EDP values.
        DSG_ID: int
            Identifies the DSG of interest.

        Returns
        -------
        P_exc: float scalar or ndarray
            Exceedance probability of the given DSG at the EDP point(s).
        """
        return self._FF_set[0].P_exc(EDP, DSG_ID)


class FragilityGroup(object):
    """
    Groups a set of similar components from a loss-assessment perspective.

    Characterizes a set of structural or non-structural components that have
    similar construction characteristics, similar potential modes of damage,
    similar probability of incurring those modes of damage, and similar
    potential consequences resulting from their damage.

    Parameters
    ----------
    ID: int
    demand_type: {'PID', 'PFA', 'PSD', 'PSA', 'ePGA', 'PGD'}
        The type of Engineering Demand Parameter (EDP) that controls the damage
        of the components in the FG. See Demand for acronym descriptions.
    performance_groups: PerformanceGroup array
        A list of performance groups that contain the components characterized
        by the FG.
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
        significantly change the dispersion of results.
    demand_location_offset: int, optional, default: 0
        Indicates if the location for the demand shall be different from the
        location of the components. Damage to components of the ceiling, for
        example, is controlled by demands on the floor above the one that the
        components belong to. This can be indicated by setting the
        demand_location_offset to 1 for such an FG.
    incomplete: bool, optional, default: False
        Indicates that the FG information is not complete and corresponding
        results shall be treated with caution.
    name: str, optional, default: ''
        Provides a short description of the fragility group.
    description: str, optional, default: ''
        Provides a detailed description of the fragility group.
    """

    def __init__(self, ID, demand_type, performance_groups,
                 directional=True, correlation=True, demand_location_offset=0,
                 incomplete=False, name='', description='', unit="ea"):
        self._ID = ID
        self._demand_type = demand_type
        self._performance_groups = performance_groups
        self._directional = directional
        self._correlation = correlation
        self._demand_location_offset = demand_location_offset
        self._incomplete = incomplete
        self._name = name
        self._description = description
        self._unit = unit

    @property
    def description(self):
        """
        Return the fragility group description.
        """
        return self._description

    @property
    def name(self):
        """
        Return the name of the fragility group.

        """
        return self._name