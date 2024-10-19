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

"""
A utility script for detecting duplicated blocks of lines across
Python test files.
"""

from __future__ import annotations

from pathlib import Path

from glob2 import glob  # type: ignore


def main(file: str) -> None:
    """
    Identifies and displays repeated consecutive line blocks within a
    file, including their line numbers.

    Parameters
    ----------
    file: str
        Path to the file to be checked for duplicates.

    """
    # file = 'tests/test_uq.py'
    group = 15  # find repeated blocks this many lines

    with Path(file).open(encoding='utf-8') as f:
        contents = f.readlines()
    num_lines = len(contents)
    for i in range(0, num_lines, group):
        glines = contents[i : i + group]
        for j in range(i + 1, num_lines):
            jlines = contents[j : j + group]
            if glines == jlines:
                print(f'{i, j}: ')  # noqa: T201
                for k in range(group):
                    print(f'    {jlines[k]}', end='')  # noqa: T201
                print()  # noqa: T201


def all_test_files() -> None:
    """
    Searches for all Python test files in the 'tests' directory and
    runs the main function to find and print repeated line blocks in each file.
    """
    test_files = glob('tests/*.py')
    for file in test_files:
        print()  # noqa: T201
        print(file)  # noqa: T201
        print()  # noqa: T201
        main(file)


if __name__ == '__main__':
    all_test_files()
