# ci/ci_local.py
"""
Resolves user input (file / node / function) to graph keys.
Used by output_builder.py to find changed functions.
"""

from pathlib import Path
from ci.graph_builder import load_graph, update_modified


def resolve_input(
    files:     list[str] = None,
    nodes:     list[str] = None,
    functions: list[str] = None,
    graph:     dict      = None,
) -> set[str]:
    """
    Convert user input to a set of graph node keys.

    --file      re-parse file, find changed methods via hash diff
    --node      all methods of that class
    --function  exact key(s) e.g. KlarfReaderNode.execute
    """
    if graph is None:
        graph = load_graph()

    changed_keys = set()

    # exact function keys
    if functions:
        for fn in functions:
            if fn in graph["nodes"]:
                changed_keys.add(fn)
                print(f"  function : {fn}")
            else:
                print(f"  ⚠ not in graph: {fn} — adding anyway")
                changed_keys.add(fn)

    # all methods of a node class
    if nodes:
        for node_name in nodes:
            matched = [
                key for key, n in graph["nodes"].items()
                if n.get("class") == node_name and n.get("method")
            ]
            if matched:
                changed_keys.update(matched)
                print(f"  node     : {node_name} → {len(matched)} methods")
            else:
                print(f"  ⚠ node not found in graph: {node_name}")

    # by file — detect changed functions via AST hash diff
    if files:
        for filepath in files:
            print(f"  file     : {filepath}")
            if not Path(filepath).exists():
                print(f"  ⚠ file not found: {filepath}")
                continue
            changed = update_modified(filepath)
            changed_keys.update(changed)
            print(f"    {len(changed)} changed function(s) detected")

    return changed_keys