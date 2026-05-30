# ci/output_builder.py
"""
Fetches test cases for a dirty set and writes output.json.

Input  : dirty set from graph_builder (list of affected functions)
Output : output.json with test cases grouped by function
"""

import json
from pathlib import Path

from ci.graph_builder       import load_graph, get_dirty_set
from ci.context_builder     import build_query
from knowledge_base.tc_store import search, get_by_node


def fetch_test_cases_for_dirty_set(
    dirty_set:   list[dict],
    graph:       dict = None,
    use_search:  bool = True,   # True = semantic search, False = get all by node
    top_k:       int  = 8,
    min_score:   float = 0.30,
) -> dict:
    """
    For each function in the dirty set, fetch relevant test cases.

    Returns:
    {
      "KlarfReaderNode.execute": {
        "depth":      0,
        "reason":     "directly modified",
        "priority":   "high",
        "test_cases": [ {...}, {...} ]
      },
      ...
    }
    """
    if graph is None:
        graph = load_graph()

    results  = {}
    seen_tcs = set()   # avoid duplicating same test case across functions

    for item in dirty_set:
        key      = item["key"]
        depth    = item["depth"]
        node_cls = key.split(".")[0]
        method   = key.split(".")[1] if "." in key else None

        # skip class-level and dunder nodes
        if not method or method.startswith("__"):
            continue

        print(f"  {key:<50} depth={depth}", end=" ... ", flush=True)

        if use_search:
            # semantic search using function summary as query
            query = build_query(key, graph)
            tcs   = search(
                query     = query,
                node      = node_cls,
                top_k     = top_k,
                min_score = min_score,
            )
        else:
            # get all test cases for this node
            tcs = get_by_node(node_cls)

        # deduplicate across functions
        new_tcs = []
        for tc in tcs:
            tc_key = tc.get("id", "") + tc.get("_doc_id", "")
            if tc_key not in seen_tcs:
                seen_tcs.add(tc_key)
                new_tcs.append(tc)

        if not new_tcs:
            print("no test cases found")
            continue

        results[key] = {
            "depth":      depth,
            "reason":     item["reason"],
            "priority":   item["priority"],
            "node":       node_cls,
            "test_cases": new_tcs,
        }

        print(f"{len(new_tcs)} test case(s)")

    return results


def write_output(
    results:     dict,
    output_path: str = "data/output.json",
) -> dict:
    """
    Write final output.json and print summary.
    """
    # flatten for summary stats
    all_tcs     = []
    by_category = {}
    by_node     = {}

    for key, entry in results.items():
        tcs  = entry["test_cases"]
        node = entry["node"]
        all_tcs.extend(tcs)

        by_node[node] = by_node.get(node, 0) + len(tcs)

        for tc in tcs:
            cat = tc.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1

    output = {
        "summary": {
            "total_functions_affected": len(results),
            "total_test_cases":         len(all_tcs),
            "by_category":              by_category,
            "by_node":                  by_node,
        },
        "functions": results,
    }

    Path(output_path).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\n{'='*55}")
    print(f"  Output written → {output_path}")
    print(f"{'='*55}")
    print(f"  Functions affected : {len(results)}")
    print(f"  Total test cases   : {len(all_tcs)}")
    print(f"\n  By category:")
    for k, v in sorted(by_category.items()):
        print(f"    {k:20}: {v}")
    print(f"\n  By node:")
    for k, v in sorted(by_node.items()):
        print(f"    {k:30}: {v}")

    return output


def run(
    files:     list[str] = None,
    nodes:     list[str] = None,
    functions: list[str] = None,
    max_depth: int       = 2,
    output:    str       = "data/output.json",
):
    from ci.graph_builder import update_modified
    from ci.ci_local      import resolve_input

    print(f"\n{'='*55}")
    print(f"  Output Builder")
    print(f"{'='*55}\n")

    graph = load_graph()
    if not graph["nodes"]:
        print("  Graph empty. Run graph_builder first.")
        raise SystemExit(1)

    # resolve input to changed keys
    print("  Resolving input...")
    changed_keys = resolve_input(
        files=files, nodes=nodes, functions=functions, graph=graph
    )

    if not changed_keys:
        print("  No matching functions found.")
        raise SystemExit(0)

    # get dirty set
    print(f"\n  Impact analysis (max_depth={max_depth})...")
    dirty = get_dirty_set(changed_keys, graph=graph, max_depth=max_depth)

    print(f"\n  Dirty set ({len(dirty)}):")
    for item in dirty:
        print(f"    depth={item['depth']} [{item['priority']:6}] {item['key']}")

    # fetch test cases
    print(f"\n  Fetching test cases from store...")
    results = fetch_test_cases_for_dirty_set(dirty, graph=graph)

    # write output
    output_data = write_output(results, output)

    return output_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file",     nargs="+")
    parser.add_argument("--node",     nargs="+")
    parser.add_argument("--function", nargs="+")
    parser.add_argument("--depth",    type=int, default=2)
    parser.add_argument("--output",   default="data/output.json")
    args = parser.parse_args()

    if not any([args.file, args.node, args.function]):
        parser.print_help()
        raise SystemExit(1)

    run(
        files     = args.file,
        nodes     = args.node,
        functions = args.function,
        max_depth = args.depth,
        output    = args.output,
    )