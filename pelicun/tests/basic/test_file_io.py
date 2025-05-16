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
# John Vouvakis Manousakis

"""These are unit and integration tests on the file_io module of pelicun."""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pelicun import base, file_io
from pelicun.pelicun_warnings import PelicunWarning

# The tests maintain the order of definitions of the `file_io.py` file.


def test_save_to_csv() -> None:
    # Test saving with orientation 0
    data = pd.DataFrame({'A': [1e-3, 2e-3, 3e-3], 'B': [4e-3, 5e-3, 6e-3]})
    units = pd.Series(['meters', 'meters'], index=['A', 'B'])
    unit_conversion_factors = {'meters': 0.001}

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'foo.csv'
        file_io.save_to_csv(
            data, filepath, units, unit_conversion_factors, orientation=0
        )
        assert Path(filepath).is_file()
        # Check that the file contains the expected data
        with Path(filepath).open(encoding='utf-8') as f:
            contents = f.read()
            assert contents == (
                ',A,B\n0,meters,meters\n0,1.0,4.0' '\n1,2.0,5.0\n2,3.0,6.0\n'
            )

    # Test saving with orientation 1
    data = pd.DataFrame({'A': [1e-3, 2e-3, 3e-3], 'B': [4e-3, 5e-3, 6e-3]})
    units = pd.Series(['meters', 'meters'], index=['A', 'B'])
    unit_conversion_factors = {'meters': 0.001}

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'bar.csv'
        file_io.save_to_csv(
            data, filepath, units, unit_conversion_factors, orientation=1
        )
        assert Path(filepath).is_file()
        # Check that the file contains the expected data
        with Path(filepath).open(encoding='utf-8') as f:
            contents = f.read()
            assert contents == (
                ',0,A,B\n0,,0.001,0.004\n1,,0.002,' '0.005\n2,,0.003,0.006\n'
            )

    #
    # edge cases
    #

    data = pd.DataFrame({'A': [1e-3, 2e-3, 3e-3], 'B': [4e-3, 5e-3, 6e-3]})
    units = pd.Series(['meters', 'meters'], index=['A', 'B'])

    # units given, without unit conversion factors
    filepath = Path(tmpdir) / 'foo.csv'
    with pytest.raises(
        ValueError,
        match='When `units` is not None, `unit_conversion_factors` must be provided.',
    ), tempfile.TemporaryDirectory() as tmpdir:
        file_io.save_to_csv(
            data, filepath, units, unit_conversion_factors=None, orientation=0
        )

    unit_conversion_factors = {'meters': 0.001}

    # not csv extension
    filepath = Path(tmpdir) / 'foo.xyz'
    with pytest.raises(
        ValueError,
        match=('Please use the `.csv` file extension. Received file name is '),
    ), tempfile.TemporaryDirectory() as tmpdir:
        file_io.save_to_csv(
            data, filepath, units, unit_conversion_factors, orientation=0
        )

    # no data, log a complaint
    mylogger = base.Logger(
        log_file=None, verbose=True, log_show_ms=False, print_log=True
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / 'foo.csv'
        with pytest.warns(PelicunWarning) as record:
            file_io.save_to_csv(
                None,
                filepath,
                units,
                unit_conversion_factors,
                orientation=0,
                log=mylogger,
            )
    assert 'Data was empty, no file saved.' in str(record.list[0].message)


@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason='Skipping test on Windows due to path handling issues.',
)
def test_substitute_default_path() -> None:
    input_paths: list[str | pd.DataFrame] = [
        'PelicunDefault/FEMA P-58/fragility.csv',
        '/data/file2.txt',
    ]
    result_paths = file_io.substitute_default_path(input_paths)
    assert (
        'seismic/building/component/FEMA P-58 2nd Edition/fragility.csv'
    ) in result_paths[0]
    assert result_paths[1] == '/data/file2.txt'


def test_load_data() -> None:
    # test loading data with orientation 0

    filepath = 'pelicun/tests/basic/data/file_io/test_load_data/units.csv'
    unit_conversion_factors = {'inps2': 0.0254, 'rad': 1.00}

    data = file_io.load_data(filepath, unit_conversion_factors)
    assert np.array_equal(data.index.values, np.array(range(6)))  # type: ignore
    assert data.shape == (6, 19)  # type: ignore
    assert isinstance(data.columns, pd.core.indexes.multi.MultiIndex)  # type: ignore
    assert data.columns.nlevels == 4  # type: ignore

    _, units = file_io.load_data(
        filepath, unit_conversion_factors, return_units=True
    )

    for item in unit_conversion_factors:
        assert item in units.unique()  # type: ignore

    filepath = 'pelicun/tests/basic/data/file_io/test_load_data/no_units.csv'
    data_nounits = file_io.load_data(filepath, {})
    assert isinstance(data_nounits, pd.DataFrame)

    # test loading data with orientation 1
    filepath = 'pelicun/tests/basic/data/file_io/test_load_data/orient_1.csv'
    data = file_io.load_data(
        filepath, unit_conversion_factors, orientation=1, reindex=False
    )
    assert isinstance(data.index, pd.core.indexes.multi.MultiIndex)
    assert data.shape == (10, 2)
    assert data.index.nlevels == 4

    # with convert=None
    filepath = 'pelicun/tests/basic/data/file_io/test_load_data/orient_1_units.csv'
    unit_conversion_factors = {'g': 1.00, 'rad': 1.00}
    data = file_io.load_data(
        filepath, unit_conversion_factors, orientation=1, reindex=False
    )
    assert isinstance(data.index, pd.core.indexes.multi.MultiIndex)
    assert data.shape == (10, 3)
    assert data.index.nlevels == 4

    # try with reindexing
    data = file_io.load_data(
        filepath, unit_conversion_factors, orientation=1, reindex=True
    )
    assert np.array_equal(data.index.values, np.array(range(10)))  # type: ignore

    #
    # edge cases
    #

    # exception: not an existing file
    with pytest.raises(FileNotFoundError):
        file_io.load_from_file('/')
    # exception: not a .csv file
    with pytest.raises(
        ValueError,
        match='Unexpected file type received when trying to load from csv',
    ):
        file_io.load_from_file('pelicun/base.py')


if __name__ == '__main__':
    pass
