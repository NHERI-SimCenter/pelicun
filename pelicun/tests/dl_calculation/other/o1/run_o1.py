"""
Runs pelicun assessments using DL_calculation.py via subprocess.
"""

import sys
from pelicun.tools.DL_calculation import main

sys.argv = ['pelicun', '-c', 'config_file.json', '--dirnameOutput', 'output']

main()
