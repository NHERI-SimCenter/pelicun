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

import time

import numpy as np
import pandas as pd


def benchmark():
    # Create a large DataFrame
    df = pd.DataFrame(np.random.rand(1000000, 10), columns=list('ABCDEFGHIJ'))

    # Measure time for df.to_dict(orient='list')
    start_time = time.time()
    df.to_dict(orient='list')
    end_time = time.time()
    print(f'Time taken with to_dict(orient="list"): {end_time - start_time} seconds')

    # Measure time for dictionary comprehension
    start_time = time.time()
    {col: df[col].tolist() for col in df.columns}
    end_time = time.time()
    print(
        f'Time taken with dictionary comprehension: {end_time - start_time} seconds'
    )

    # Measure time for dictionary comprehension without to list
    start_time = time.time()
    {col: df[col] for col in df.columns}
    end_time = time.time()
    print(
        f'Time taken with dictionary comprehension '
        f'without to_list: {end_time - start_time} seconds'
    )

    # Measure time for .values
    start_time = time.time()
    df.values
    end_time = time.time()
    print(f'Time taken with .values: {end_time - start_time} seconds')

    # Measure time for using df.to_numpy()
    start_time = time.time()
    data_array = df.to_numpy()
    {col: data_array[:, i].tolist() for i, col in enumerate(df.columns)}
    end_time = time.time()
    print(f'Time taken with df.to_numpy(): {end_time - start_time} seconds')

    # Measure time for using df.to_dict()
    start_time = time.time()
    df.to_dict()
    end_time = time.time()
    print(f'Time taken with df.to_dict(): {end_time - start_time} seconds')


if __name__ == '__main__':
    benchmark()
