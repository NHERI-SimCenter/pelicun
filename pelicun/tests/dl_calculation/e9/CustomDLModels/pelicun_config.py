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
# Stevan Gavrilovic
# Adam Zsarnoczay
# Example 9 Tsunami, Seaside

import pandas as pd

# noqa: INP001


def auto_populate(aim: dict):  # noqa: ANN201
    """
    Populates the DL model for tsunami example using custom fragility functions

    Assumptions:
    * Everything relevant to auto-population is provided in the
    Building Information Model (AIM).
    * The information expected in the AIM file is described in the
    parse_AIM method.

    Parameters
    ----------
    aim: dictionary
        Contains the information that is available about the asset and will be
        used to auto-populate the damage and loss model.

    Returns
    -------
    gi_ap: dictionary
        Contains the extended AIM data.
    dl_ap: dictionary
        Contains the auto-populated loss model.
    """

    # parse the AIM data
    # print(aim) # Look in the AIM.json file to see what you can access here

    # extract the General Information
    gi = aim.get('GeneralInformation')

    # gi_ap is the 'extended AIM data - this case no extended AIM data
    gi_ap = gi.copy()

    # Get the number of Stories - note the column heading needs to be exactly
    # 'NumberOfStories'.
    nstories = gi_ap.get('NumberOfStories')
    if nstories is None:
        print('NumberOfStories attribute missing from AIM file.')  # noqa: T201
        return None, None, None

    # Get the fragility tag according to some building attribute; the
    # NumberOfStories in this case. The fragility tag needs to be unique, i.e.,
    # one tag for each fragility group. The fragility tag has to match the file
    # name of the json file in the 'ComponentDataFolder' (without the .json
    # suffix)

    if nstories == 1:
        fragility_function_tag = 'building.1'
    elif nstories == 2:
        fragility_function_tag = 'building.2'
    elif nstories >= 3:
        fragility_function_tag = 'building.3andAbove'
    else:
        print(f'Invalid number of storeys provided: {nstories}')  # noqa: T201

    # prepare the component assignment
    comp = pd.DataFrame(
        {f'{fragility_function_tag}': ['ea', 1, 1, 1, 'N/A']},
        index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
    ).T

    # Populate the dl_ap
    dl_ap = {
        'Asset': {
            'ComponentAssignmentFile': 'CMP_QNT.csv',
            'ComponentDatabase': 'None',
            'ComponentDatabasePath': 'CustomDLDataFolder/damage_Tsunami.csv',
        },
        'Damage': {'DamageProcess': 'None'},
        'Demands': {},
        'Losses': {
            'Repair': {
                'ConsequenceDatabase': 'None',
                'ConsequenceDatabasePath': (
                    'CustomDLDataFolder/loss_repair_Tsunami.csv'
                ),
                'MapApproach': 'User Defined',
                'MapFilePath': 'CustomDLDataFolder/loss_map.csv',
                'DecisionVariables': {
                    'Cost': True,
                    'Carbon': False,
                    'Energy': False,
                    'Time': False,
                },
            }
        },
    }

    return gi_ap, dl_ap, comp
