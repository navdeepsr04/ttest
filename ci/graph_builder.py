# ci/graph_builder.py
"""
Builds and maintains the code graph from parsed node metadata.

Graphs stored in data/code_graph.json:
  nodes             — every class.method as a node with metadata
  call_edges        — A calls B
  inheritance_edges — A inherits from B
  settings_edges    — Node A uses Settings B

PR update operations:
  update_modified(file)  — re-parse changed file, update edges
  update_added(file)     — parse new file, add to graph
  update_deleted(file)   — remove all nodes/edges for deleted file
"""

import json
import time
from pathlib import Path

from ci.ast_parser import parse_file, parse_folder


GRAPH_PATH  = Path("data/code_graph.json")
NODES_PATH  = Path("data/parsed_nodes.json")
MAX_DEPTH   = 2   # max depth for impact traversal


# ── empty graph template ──────────────────────────────────────────────────────

def empty_graph() -> dict:
    return {
        "built_at":          "",
        "nodes":             {},
        "call_edges":        [],
        "inheritance_edges": [],
        "settings_edges":    [],
    }


# ── node key ──────────────────────────────────────────────────────────────────

def node_key(class_name: str, method_name: str = None) -> str:
    """
    Unique key for a graph node.
    Class level  : "KlarfReaderNode"
    Method level : "KlarfReaderNode.execute"
    """
    if method_name:
        return f"{class_name}.{method_name}"
    return class_name


# ── build from parsed files ───────────────────────────────────────────────────

def build_from_parsed(parsed_files: list[dict]) -> dict:
    """
    Build complete graph from list of parsed file results.
    Each item is the output of ast_parser.parse_file().
    """
    graph = empty_graph()
    graph["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    for parsed in parsed_files:
        filepath = parsed["file"]
        _add_file_to_graph(graph, parsed, filepath)

    return graph


def _add_file_to_graph(graph: dict, parsed: dict, filepath: str):
    """Add all classes and edges from one parsed file into the graph."""

    for cls in parsed["classes"]:
        cls_name = cls["name"]
        cls_type = cls["type"]

        # ── class-level node ──────────────────────────────────────────────
        key = node_key(cls_name)
        graph["nodes"][key] = {
            "file":       filepath,
            "class":      cls_name,
            "method":     None,
            "type":       cls_type,
            "bases":      cls["bases"],
            "ast_hash":   cls["ast_hash"],
            "lineno":     cls.get("lineno", 0),
        }

        # ── inheritance edges ─────────────────────────────────────────────
        for base in cls["bases"]:
            short_base = base.split(".")[-1]
            graph["inheritance_edges"].append({
                "child":  cls_name,
                "parent": short_base,
            })

        # ── settings edges ────────────────────────────────────────────────
        # if this is a node class, find which settings class it references
        if cls_type == "node":
            for method in cls["methods"]:
                for call in method.get("calls", []):
                    # heuristic: if a call matches a known settings class name
                    # we detect this more accurately in _resolve_settings below
                    pass

        # ── method-level nodes and call edges ─────────────────────────────
        for method in cls["methods"]:
            m_name = method["name"]
            m_key  = node_key(cls_name, m_name)

            graph["nodes"][m_key] = {
                "file":       filepath,
                "class":      cls_name,
                "method":     m_name,
                "type":       cls_type,
                "params":     method["params"],
                "returns":    method["returns"],
                "docstring":  method["docstring"],
                "ast_hash":   method["ast_hash"],
                "is_private": method["is_private"],
                "is_static":  method["is_static"],
                "lineno":     method["lineno"],
            }

            # call edges — this method calls other methods
            for called in method.get("calls", []):
                # try to resolve to full key within same class first
                full_called = node_key(cls_name, called)
                graph["call_edges"].append({
                    "caller": m_key,
                    "callee": full_called,
                    "raw_callee": called,
                })

    # ── settings edges ────────────────────────────────────────────────────
    # after all classes loaded, link node classes to their settings classes
    _resolve_settings_edges(graph, parsed)


def _resolve_settings_edges(graph: dict, parsed: dict):
    """
    Detect which Settings class each Node class uses.
    Strategy: if NodeXxx and XxxSettings exist in the same file
    or share a name prefix, create a settings edge.
    """
    cls_names = [c["name"] for c in parsed["classes"]]
    node_classes     = [c for c in parsed["classes"] if c["type"] == "node"]
    settings_classes = [c for c in parsed["classes"] if c["type"] == "settings"]

    for node_cls in node_classes:
        for settings_cls in settings_classes:
            # name-based heuristic:
            # KlarfReaderNode ↔ KlarfReaderSettings
            # TSNENode ↔ TSNESettings
            node_prefix     = node_cls["name"].replace("Node", "").lower()
            settings_prefix = settings_cls["name"].replace("Settings", "").lower()

            if node_prefix == settings_prefix or \
               settings_cls["name"] in node_cls["name"] or \
               node_cls["name"].replace("Node", "") in settings_cls["name"]:

                graph["settings_edges"].append({
                    "node":     node_cls["name"],
                    "settings": settings_cls["name"],
                    "file":     parsed["file"],
                })


# ── build full graph ──────────────────────────────────────────────────────────

def build(
    nodes_folder: str  = None,
    parsed_path:  str  = str(NODES_PATH),
    output_path:  str  = str(GRAPH_PATH),
    force_rebuild: bool = False,
) -> dict:
    """
    Build the complete code graph.

    If nodes_folder provided — re-parse everything from source.
    Else — load from parsed_nodes.json.
    """
    # check if already built
    out = Path(output_path)
    if out.exists() and not force_rebuild and not nodes_folder:
        print(f"  Graph already exists at {output_path}")
        print(f"  Use force_rebuild=True to rebuild.")
        return json.loads(out.read_text(encoding="utf-8"))

    # parse from folder or load existing
    if nodes_folder:
        print(f"  Parsing folder: {nodes_folder}")
        parsed_files = parse_folder(nodes_folder)
        # save parsed for reuse
        NODES_PATH.write_text(
            json.dumps(parsed_files, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  Parsed {len(parsed_files)} files")
    else:
        p = Path(parsed_path)
        if not p.exists():
            print(f"  {parsed_path} not found. Provide nodes_folder to parse.")
            raise SystemExit(1)
        parsed_files = json.loads(p.read_text(encoding="utf-8"))
        print(f"  Loaded {len(parsed_files)} parsed files from {parsed_path}")

    # build graph
    print(f"  Building graph...")
    graph = build_from_parsed(parsed_files)

    # save
    out.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    _print_graph_stats(graph, output_path)
    return graph


# ── PR update operations ──────────────────────────────────────────────────────

def load_graph(path: str = str(GRAPH_PATH)) -> dict:
    p = Path(path)
    if not p.exists():
        return empty_graph()
    return json.loads(p.read_text(encoding="utf-8"))


def save_graph(graph: dict, path: str = str(GRAPH_PATH)):
    Path(path).write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def update_modified(filepath: str) -> dict:
    """
    PR case: existing file was modified.
    Re-parse file, update graph nodes and edges.
    Returns set of changed method keys.
    """
    graph   = load_graph()
    changed = set()

    # get old hashes for this file
    old_hashes = {
        key: node["ast_hash"]
        for key, node in graph["nodes"].items()
        if node["file"] == filepath
    }

    # remove old nodes and edges for this file
    graph = _remove_file_from_graph(graph, filepath)

    # re-parse
    parsed = parse_file(filepath)
    if parsed:
        _add_file_to_graph(graph, parsed, filepath)
        _resolve_settings_edges(graph, parsed)

        # find what changed by comparing hashes
        for key, node in graph["nodes"].items():
            if node["file"] != filepath:
                continue
            old_hash = old_hashes.get(key)
            new_hash = node["ast_hash"]
            if old_hash != new_hash:
                changed.add(key)
                if old_hash is None:
                    print(f"    NEW   : {key}")
                else:
                    print(f"    CHANGED: {key}")
    else:
        changed = set(old_hashes.keys())

    graph["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_graph(graph)
    return changed


def update_added(filepath: str) -> dict:
    """
    PR case: new file was added.
    Parse and add to graph.
    Returns set of new method keys.
    """
    graph  = load_graph()
    new_keys = set()

    parsed = parse_file(filepath)
    if parsed:
        _add_file_to_graph(graph, parsed, filepath)
        _resolve_settings_edges(graph, parsed)

        for key, node in graph["nodes"].items():
            if node["file"] == filepath:
                new_keys.add(key)
                print(f"    ADDED : {key}")

    graph["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_graph(graph)
    return new_keys


def update_deleted(filepath: str) -> set:
    """
    PR case: file was deleted.
    Remove all nodes and edges for this file.
    Returns set of removed method keys.
    """
    graph = load_graph()

    removed = {
        key for key, node in graph["nodes"].items()
        if node["file"] == filepath
    }

    for key in removed:
        print(f"    REMOVED: {key}")

    graph = _remove_file_from_graph(graph, filepath)
    graph["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_graph(graph)
    return removed


def _remove_file_from_graph(graph: dict, filepath: str) -> dict:
    """Remove all nodes and edges belonging to a file."""
    # remove nodes
    graph["nodes"] = {
        k: v for k, v in graph["nodes"].items()
        if v["file"] != filepath
    }

    # remove call edges where caller belongs to this file
    # (callee may be in another file — keep those)
    file_classes = set()
    for key, node in graph["nodes"].items():
        if node["file"] == filepath:
            file_classes.add(node["class"])

    graph["call_edges"] = [
        e for e in graph["call_edges"]
        if not e["caller"].split(".")[0] in file_classes
    ]

    graph["inheritance_edges"] = [
        e for e in graph["inheritance_edges"]
        if e["child"] not in file_classes
    ]

    graph["settings_edges"] = [
        e for e in graph["settings_edges"]
        if e["file"] != filepath
    ]

    return graph


# ── impact analysis ───────────────────────────────────────────────────────────

def get_dirty_set(
    changed_keys: set[str],
    graph:        dict = None,
    max_depth:    int  = MAX_DEPTH,
) -> list[dict]:
    """
    Given a set of directly changed function keys,
    find all affected functions by walking the graph.

    Returns list of dicts:
    {
      "key":    "KlarfReaderNode.execute",
      "depth":  0,
      "reason": "directly modified",
      "priority": "high"
    }
    """
    if graph is None:
        graph = load_graph()

    # build reverse call map: callee → list of callers
    reverse_calls = {}
    for edge in graph["call_edges"]:
        callee = edge["callee"]
        caller = edge["caller"]
        reverse_calls.setdefault(callee, set()).add(caller)

    # build forward call map: caller → list of callees
    forward_calls = {}
    for edge in graph["call_edges"]:
        caller = edge["caller"]
        callee = edge["callee"]
        forward_calls.setdefault(caller, set()).add(callee)

    # build subclass map: parent → list of children
    subclasses = {}
    for edge in graph["inheritance_edges"]:
        parent = edge["parent"]
        child  = edge["child"]
        subclasses.setdefault(parent, set()).add(child)

    # build settings map: settings → list of nodes using it
    settings_users = {}
    for edge in graph["settings_edges"]:
        s = edge["settings"]
        n = edge["node"]
        settings_users.setdefault(s, set()).add(n)

    dirty   = {}   # key → {depth, reason, priority}
    visited = set()

    def add_dirty(key, depth, reason):
        if key not in dirty or dirty[key]["depth"] > depth:
            priority = "high" if depth <= 1 else "medium"
            dirty[key] = {
                "key":      key,
                "depth":    depth,
                "reason":   reason,
                "priority": priority,
            }

    def walk(key, depth, reason):
        if depth > max_depth or key in visited:
            return
        visited.add(key)
        add_dirty(key, depth, reason)

        if depth >= max_depth:
            return

        # 1. callees — functions this one calls
        for callee in forward_calls.get(key, set()):
            if callee in graph["nodes"]:
                walk(callee, depth + 1,
                     f"called by {key} (depth {depth+1})")

        # 2. callers — functions that call this one
        for caller in reverse_calls.get(key, set()):
            if caller in graph["nodes"]:
                walk(caller, depth + 1,
                     f"calls {key} (depth {depth+1})")

        # 3. subclasses — if a parent class method changed
        cls_name = key.split(".")[0]
        method   = key.split(".")[1] if "." in key else None
        for sub in subclasses.get(cls_name, set()):
            sub_key = node_key(sub, method) if method else sub
            if sub_key in graph["nodes"]:
                walk(sub_key, depth + 1,
                     f"subclass of {cls_name} (depth {depth+1})")

        # 4. settings users — if a settings class changed
        if graph["nodes"].get(key, {}).get("type") == "settings":
            for user_node in settings_users.get(cls_name, set()):
                user_key = node_key(user_node, "execute")
                if user_key in graph["nodes"]:
                    walk(user_key, depth + 1,
                         f"uses settings {cls_name} (depth {depth+1})")

    # seed with directly changed keys
    for key in changed_keys:
        walk(key, depth=0, reason="directly modified")

    # sort by depth then key
    result = sorted(dirty.values(), key=lambda x: (x["depth"], x["key"]))
    return result


# ── stats ─────────────────────────────────────────────────────────────────────

def _print_graph_stats(graph: dict, path: str):
    nodes    = len(graph["nodes"])
    methods  = sum(1 for n in graph["nodes"].values() if n["method"])
    classes  = sum(1 for n in graph["nodes"].values() if not n["method"])
    calls    = len(graph["call_edges"])
    inherits = len(graph["inheritance_edges"])
    settings = len(graph["settings_edges"])

    print(f"\n{'='*50}")
    print(f"  Code graph built")
    print(f"{'='*50}")
    print(f"  Total nodes       : {nodes}")
    print(f"    Classes         : {classes}")
    print(f"    Methods         : {methods}")
    print(f"  Call edges        : {calls}")
    print(f"  Inheritance edges : {inherits}")
    print(f"  Settings edges    : {settings}")
    print(f"  Saved → {path}")
    print(f"{'='*50}\n")


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    args     = sys.argv[1:]
    folder   = next((a for a in args if Path(a).is_dir()), None)
    rebuild  = "--rebuild" in args
    test_impact = "--impact" in args

    print(f"\n{'='*50}")
    print(f"  Graph Builder")
    print(f"{'='*50}\n")

    # build graph
    graph = build(
        nodes_folder  = folder,
        force_rebuild = rebuild,
    )

    # test impact analysis
    if test_impact:
        print(f"\nImpact analysis test:")
        print(f"  Simulating change to: KlarfReaderNode.execute\n")

        dirty = get_dirty_set(
            {"KlarfReaderNode.execute"},
            graph = graph,
        )

        print(f"  Dirty set ({len(dirty)} functions):\n")
        for item in dirty:
            print(f"  depth={item['depth']} [{item['priority']:6}] "
                  f"{item['key']}")
            print(f"           reason: {item['reason']}")