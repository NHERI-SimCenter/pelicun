"""
Code inspection methods/functions.
"""

from __future__ import annotations
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
    node : ast.FunctionDef
        The AST node representing the function definition.
    filename : str
        The path to the Python file to be searched.
    search_string : str
        The string to search for within the function bodies.
    functions_with_string : list[str]
        The list to append function names that contain the search
        string.

    """
    with open(filename, 'r', encoding='utf-8') as f:
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
    filename : str
        The path to the Python file to be searched.
    search_string : str
        The string to search for within the function bodies.

    Returns
    -------
    list[str]
        A list of function names that contain the search string in
        their bodies.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        contents = file.read()
    tree = ast.parse(contents, filename=filename)

    functions_with_string: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            visit_FunctionDef(node, filename, search_string, functions_with_string)

    return functions_with_string
