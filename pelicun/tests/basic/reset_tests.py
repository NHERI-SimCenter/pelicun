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

"""This file is used to reset all expected test result data."""

from __future__ import annotations

import ast
import importlib
import os
import re
from pathlib import Path


def reset_all_test_data(*, restore: bool = True, purge: bool = False) -> None:  # noqa: C901
    """
    Update the expected result pickle files with new results, accepting
    the values obtained by executing the code as correct from now on.

    CAUTION: This function should never be used if tests are
    failing. Its only purpose is to aid the development of more tests
    and keeping things tidy. If tests are failing, the specific tests
    need to be investigated, and after rectifying the cause, new
    expected test result values should be created at an individual
    basis.

    Note: This function assumes that the interpreter's current
    directory is the package root directory (`pelicun`). The code
    assumes that the test data directory exists.
    Data deletion only involves `.pcl` files that begin with `test_` and
    reside in /pelicun/tests/basic/data.

    Parameters
    ----------
    restore: bool
      Whether to re-generate the test result data
    purge: bool
      Whether to remove the test result data before re-generating the
      new values.

    Raises
    ------
    ValueError
      If the test directory is not found.

    OSError
      If the code is ran from the wrong directory. This code is only
      meant to be executed with the interpreter running in the
      `pelicun` directory. Dangerous things may happen otherwise.

    """
    cwd = Path.cwd()
    if cwd != 'pelicun':
        msg = (
            'Wrong directory. '
            'See the docstring of `reset_all_test_data`. Aborting'
        )
        raise OSError(msg)

    # where the test result data are stored
    testdir = Path('tests') / 'data'
    if not testdir.exists():
        msg = 'pelicun/tests/basic/data directory not found.'
        raise ValueError(msg)

    # clean up existing test result data
    # only remove .pcl files that start with `test_`
    pattern = re.compile(r'^test_.\.pcl')
    if purge:
        for root, _, files in os.walk('.'):
            for filename in files:
                if pattern.match(filename):
                    (Path(root) / filename).unlink()

    # generate new data
    if restore:
        # get a list of all existing test files and iterate
        test_files = list(Path('tests').glob('*test*.py'))
        for test_file in test_files:
            # open the file and statically parse the code looking for functions
            with Path(test_file).open(encoding='utf-8') as file:
                node = ast.parse(file.read())
            functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            # iterate over the functions looking for test_ functions
            # with `reset` as an argument
            for function in functions:
                if function.name.startswith('test_'):
                    # list the arguments of the function
                    arguments = [a.arg for a in function.args.args]
                    if 'reset' in arguments:
                        # we want to import it and run it with reset=True
                        # construct a module name, like 'tests.test_uq'
                        module_name = 'tests.' + Path(test_file).name.replace(
                            '.py', ''
                        )
                        # import the module
                        module = importlib.import_module(module_name)
                        # get the function
                        func = getattr(module, function.name)
                        # run it to reset its expected test output data
                        func(reset=True)
