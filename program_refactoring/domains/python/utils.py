import ast
import re
from typing import Set, Dict, List

from program_refactoring.headers import PYTHON_HEADER


def get_function_calls(func_def: ast.FunctionDef) -> List[str]:
    """Extract all function calls from a function definition."""
    calls = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    return calls


def build_call_graph(source_code: str) -> Dict[str, List[str]]:
    """Build a call graph from source code."""
    tree = ast.parse(source_code)
    call_graph = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            call_graph[node.name] = get_function_calls(node)

    return call_graph


def get_transitive_closure(
    call_graph: Dict[str, List[str]], start_func: str
) -> Set[str]:
    """Get the transitive closure of functions called from start_func."""
    visited = set()
    to_visit = [start_func]

    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue

        visited.add(current)

        # Add all functions called by current function
        if current in call_graph:
            for called_func in call_graph[current]:
                if called_func not in visited:
                    to_visit.append(called_func)

    # Remove the start function itself from the closure
    visited.discard(start_func)
    return visited


def get_func_names(program: str, codebank: str) -> set[str]:
    """Get the transitive closure of all functions called in main"""
    combined_source = codebank + "\n" + program
    call_graph = build_call_graph(combined_source)
    closure = get_transitive_closure(call_graph, "main")
    return closure


def wrong_get_func_names(program: str, codebank: str) -> list[str]:
    """Get all function calls that are not part of the header"""
    import pdb

    pdb.set_trace()
    parsed = ast.parse(program)
    # get all expressions
    func_names = []
    header_func_names = ["print"]

    for node in ast.walk(parsed):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            try:
                func_names.append(node.value.func.id)
            except AttributeError:
                pass
        elif isinstance(node, ast.Call):
            try:
                func_names.append(node.func.id)
            except AttributeError:
                try:
                    func_names.append(node.func.value.func.id)
                except AttributeError:
                    try:
                        func_names.append(node.func.value.id)
                    except AttributeError:
                        pass

        else:
            pass
        # ignore all imported functions
        if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
            for name in node.names:
                if name.asname is None:
                    header_func_names.append(name.name)
                else:
                    header_func_names.append(name.asname)

    # get header names
    header = ast.parse(PYTHON_HEADER)
    for node in ast.walk(header):
        if isinstance(node, ast.FunctionDef):
            header_func_names.append(node.name)

    # func_names = list(set(func_names) - set(header_func_names))
    func_names = list(set(func_names))

    return func_names


def clean_import(program):
    return re.sub("from .*\.codebank import \*", "", program).strip()
