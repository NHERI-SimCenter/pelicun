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

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

from pelicun.db import convert_Series_to_dict


def export_DB(data_path, target_dir):
    data_path = Path(data_path).resolve()
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(exist_ok=True)

    # start with the data

    target_dir_data = target_dir / 'data'
    target_dir_data.mkdir(exist_ok=True)

    DB_df = pd.read_hdf(data_path, 'data')

    for row_id, row in DB_df.iterrows():

        row_dict = convert_Series_to_dict(row)

        with open(target_dir_data / f'{row_id}.json', 'w', encoding='utf-8') as f:
            json.dump(row_dict, f, indent=2)

    # add population if it exists

    try:

        DB_df = pd.read_hdf(data_path, 'pop')

        pop_dict = {}

        for row_id, row in DB_df.iterrows():

            pop_dict.update({row_id: convert_Series_to_dict(row)})

        with open(target_dir / 'population.json', 'w', encoding='utf-8') as f:
            json.dump(pop_dict, f, indent=2)

    except (ValueError, NotImplementedError, FileNotFoundError):
        pass


if __name__ == '__main__':

    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--DL_DB_path')
    parser.add_argument('--target_dir')

    args_namespace = parser.parse_args(args)

    export_DB(args_namespace.DL_DB_path, args_namespace.target_dir)
