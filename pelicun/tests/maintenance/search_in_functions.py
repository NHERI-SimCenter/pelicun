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

"""Code inspection methods/functions."""

from __future__ import annotations  # noqa: I001
from pathlib import Path

import ast


def visit_FunctionDef(
    node: ast.FunctionDef,
    filename: str,
    search_string: str,
    functions_with_string: list[str],
) -> None:
    """
    Visit a function definition node and check if it contains the
    search string.

    Parameters
    ----------
    node: ast.FunctionDef
        The AST node representing the function definition.
    filename: str
        The path to the Python file to be searched.
    search_string: str
        The string to search for within the function bodies.
    functions_with_string: list[str]
        The list to append function names that contain the search
        string.

    """
    with Path(filename).open(encoding='utf-8') as f:
        contents = f.read()

    function_code = ast.get_source_segment(contents, node)
    assert function_code is not None

    if search_string in function_code:
        functions_with_string.append(node.name)


def find_functions_with_string(filename: str, search_string: str) -> list[str]:
    """
    Finds functions in a Python file that contain a specific string in
    their body.

    Parameters
    ----------
    filename: str
        The path to the Python file to be searched.
    search_string: str
        The string to search for within the function bodies.

    Returns
    -------
    list[str]
        A list of function names that contain the search string in
        their bodies.

    """
    with Path(filename).open(encoding='utf-8') as file:
        contents = file.read()
    tree = ast.parse(contents, filename=filename)

    functions_with_string: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            visit_FunctionDef(node, filename, search_string, functions_with_string)

    return functions_with_string
