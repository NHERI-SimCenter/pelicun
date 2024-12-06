# -*- coding: utf-8 -*-

# Contributors:
# Stevan Gavrilovic
# Adam Zsarnoczay
# Example 9 Tsunami, Seaside

import pandas as pd


def auto_populate(aim):
    """
    Populates the DL model for tsunami example using custom fragility functions

    Assumptions:
    * Everything relevant to auto-population is provided in the
    Buiding Information Model (AIM).
    * The information expected in the AIM file is described in the
    parse_AIM method.

    Parameters
    ----------
    aim: dictionary
        Contains the information that is available about the asset and will be
        used to auto-populate the damage and loss model.

    Returns
    -------
    GI_ap: dictionary
        Contains the extended AIM data.
    DL_ap: dictionary
        Contains the auto-populated loss model.
    """

    # parse the AIM data
    # print(aim) # Look in the AIM.json file to see what you can access here

    # extract the General Information
    GI = aim.get('GeneralInformation', None)

    # GI_ap is the 'extended AIM data - this case no extended AIM data
    GI_ap = GI.copy()

    # Get the number of Stories - note the column heading needs to be exactly
    # 'NumberOfStories'.
    nstories = GI_ap.get('NumberOfStories', None)
    if nstories is None:
        print("NumberOfStories attribute missing from AIM file.")
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
        print(f"Invalid number of storeys provided: {nstories}")

    # prepare the component assignment
    CMP = pd.DataFrame(
        {f'{fragility_function_tag}': ['ea', 1, 1, 1, 'N/A']},
        index=['Units', 'Location', 'Direction', 'Theta_0', 'Family'],
    ).T

    # Populate the DL_ap
    DL_ap = {
        "Asset": {
            "ComponentAssignmentFile": "CMP_QNT.csv",
            "ComponentDatabase": "None",
            "ComponentDatabasePath": "CustomDLDataFolder/damage_Tsunami.csv",
        },
        "Damage": {"DamageProcess": "None"},
        "Demands": {},
        "Losses": {
            "Repair": {
                "ConsequenceDatabase": "None",
                "ConsequenceDatabasePath": (
                    "CustomDLDataFolder/loss_repair_Tsunami.csv"
                ),
                "MapApproach": "User Defined",
                "MapFilePath": "CustomDLDataFolder/loss_map.csv",
                "DecisionVariables": {
                    "Cost": True,
                    "Carbon": False,
                    "Energy": False,
                    "Time": False,
                },
            }
        },
    }

    return GI_ap, DL_ap, CMP
