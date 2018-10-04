import sys

sys.path.append('C:/Adam/pelicun')

import numpy as np

import pelicun
from pelicun.control import FEMA_P58_Assessment
from pelicun.file_io import write_SimCenter_DL_output

# location of DL input file
DL_input_path = "c:/Adam/Workdir/tmp.SimCenter/templatedir/dakota.json"
EDP_input_path = "c:/Adam/Workdir/tmp.SimCenter/dakotaTab.out"
CMP_data_path = 'C:/Adam/Dropbox/Workdir/SimCenter/LOSS data/fragilities/'
POP_data_path = 'C:/Adam/Dropbox/Workdir/SimCenter/LOSS data/population.json'

A = FEMA_P58_Assessment()
A.read_inputs(DL_input_path, EDP_input_path, 
	CMP_data_path, POP_data_path, verbose=False)

A.define_random_variables()

A.define_loss_model()

A.calculate_damage()

A.calculate_losses()

A.aggregate_results()

S = A._SUMMARY

S.columns = [('{}/{}'.format(s0, s1)).replace(' ', '_') 
             for s0, s1 in zip(S.columns.get_level_values(0),
                               S.columns.get_level_values(1))]

write_SimCenter_DL_output('DL_summary.csv', S)
write_SimCenter_DL_output('DL_summary_stats.csv', 
                          S.describe(np.arange(1, 100)/100.))