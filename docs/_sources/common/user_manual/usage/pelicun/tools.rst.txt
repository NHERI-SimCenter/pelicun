.. _lbl-tools:

****************
Standalone Tools
****************

Standalon python scripts are provided with pelicun that are built using methods from the library and facilitate using pelicun for those who prefer to work in an environment other than Python. These tools are listed below.

==============
DL_calculation
==============

`DL_calculation.py` uses elements of the pelicun library to assemble a damage and loss calculation workflow, perform the calculations and save the results. It can be run from the command line as follows::

    python DL_calculation.py --filenameDL <DL_file_path> --filenameEDP <EDP_file_path> <additional args>

where `<DL_file_path>` should be replaced with a string pointing to the damage and loss configuration file and `<EDP_file_path>` should be replaced with a string pointing to the EDP input file. (see :numref:`lbl-inputs` for details on those files)

Additional optional settings through `<additional args>` allow researchers to customize their calculations. All of these settings need to be provided by adding `--argument_name argument_value` to the command above, where `argument_name` is the name shown in bold below and `argument_value` is the provided setting for that argument.

The following arguments are available:

    :Realizations:
        Overrides the number of realizations set in the configuration file.

    :dirnameOutput:
        Specifies the location where the output files will be saved at the end of the calculation. The default location is the folder where the command is executed.

    :event_time:
        Setting it to "False" turns off the consideration of event time and the change in the population in the building over the course of a day. When the event time is turned off, pelicun calculates with the peak population for every realization. The default value is "True".

    :detailed_results:
        When set to "False", the `DL_calculation` script only prints the main results of the calculation that provide an overview of the EDPs, DMs, and DVs. Detailed results allow researchers to review intermediate calculation outputs and disaggregate the damage and loss measures to individual components. The default value is "True".

    :coupled_EDP:
        Overrides the coupled_EDP setting in the configuration file. The default value is "False".

    :log_file:
        When set to "False", no log file is produced. The default value is "True".

========
exportDB
========

This tool faciliates exporting the contents of HDF5 Damage and Loss databases to JSON files that can be easily edited by researchers.

The script can be run from the command line as follows::

    python export_DB.py --DL_DB_path <DB_file_path> --target_dir <target_dir_path>

where `<DB_file_path>` should be replaced with a string pointing to the HDF5 file tha contains the damage and loss database, and `<target_dir_path>` should be replaced with a string pointing to the directory where you want to save the JSON files.
