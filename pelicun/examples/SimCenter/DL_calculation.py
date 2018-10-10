import sys
import numpy as np

import pelicun
from pelicun.control import FEMA_P58_Assessment
from pelicun.file_io import write_SimCenter_DL_output

def run_pelicun(DL_input_path, EDP_input_path, CMP_data_path, POP_data_path):

	A = FEMA_P58_Assessment()
	A.read_inputs(DL_input_path, EDP_input_path, 
		CMP_data_path, POP_data_path, verbose=False)

	A.define_random_variables()

	A.define_loss_model()

	A.calculate_damage()

	A.calculate_losses()

	A.aggregate_results()

	S = A._SUMMARY

	S.index.name = '#Num'

	S.columns = [('{}/{}'.format(s0, s1)).replace(' ', '_') 
	             for s0, s1 in zip(S.columns.get_level_values(0),
	                               S.columns.get_level_values(1))]

	write_SimCenter_DL_output('DL_summary.csv', S)

	S_stat = S.describe(np.arange(1, 100)/100.)

	S_stat.index.name = 'attribute'

	write_SimCenter_DL_output('DL_summary_stats.csv', S_stat)

	return 0

if __name__ == '__main__':

	sys.exit(run_pelicun(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))