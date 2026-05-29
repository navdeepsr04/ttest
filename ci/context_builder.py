# ci/context_builder.py
"""
For each function in the dirty set:
  1. Build a text summary from graph metadata
  2. Embed it
  3. Search vector store for matching requirements
  4. Return context packages ready for test generation

Output per function:
{
  "function_key":  "KlarfReaderNode.execute",
  "file":          "nodes/klarf_reader/klarf_reader.py",
  "summary":       "class KlarfReaderNode... def execute...",
  "requirements":  [
    {
      "id":    "FR-008",
      "score": 0.821,
      "what":  "node reads KLARF file from configured path",
      ...
    }
  ]
}
"""

import json
from pathlib import Path

from knowledge_base.vector_store import search
from knowledge_base.embedder     import embed_texts
from ci.graph_builder            import load_graph


# ── minimum score to include a requirement ────────────────────────────────────
MIN_SCORE   = 0.35
TOP_K       = 5


# ── build query text from graph node ─────────────────────────────────────────

def build_query(
    function_key: str,
    graph:        dict,
) -> str:
    """
    Build a search query string from graph metadata for one function.
    This is what gets embedded and searched against requirements.
    """
    node = graph["nodes"].get(function_key)
    if not node:
        return function_key

    parts = []
    cls   = node.get("class", "")
    method = node.get("method", "")

    # class + method signature
    bases = node.get("bases", [])
    if bases:
        parts.append(f"class {cls}({', '.join(bases)})")
    else:
        parts.append(f"class {cls}")

    if method:
        params  = ", ".join(node.get("params", []))
        returns = node.get("returns", "")
        parts.append(f"def {method}({params}) -> {returns}")

        doc = node.get("docstring", "")
        if doc:
            parts.append(doc)

    # find related settings class from settings_edges
    settings_edges = graph.get("settings_edges", [])
    for edge in settings_edges:
        if edge["node"] == cls:
            settings_cls = edge["settings"]
            parts.append(f"uses settings: {settings_cls}")

            # find settings fields from graph nodes
            settings_node = graph["nodes"].get(settings_cls)
            if settings_node:
                parts.append(f"settings type: {settings_cls}")
            break

    # calls
    call_edges = graph.get("call_edges", [])
    callees = [
        e["raw_callee"] for e in call_edges
        if e["caller"] == function_key
    ]
    if callees:
        parts.append(f"calls: {', '.join(callees[:5])}")

    # inheritance context
    inh_edges = graph.get("inheritance_edges", [])
    for edge in inh_edges:
        if edge["child"] == cls:
            parts.append(f"inherits from: {edge['parent']}")
            break

    return "\n".join(parts)


# ── search requirements for one function ─────────────────────────────────────

def find_requirements(
    function_key: str,
    graph:        dict,
    top_k:        int   = TOP_K,
    min_score:    float = MIN_SCORE,
) -> dict:
    """
    Find relevant requirements for one function from the dirty set.

    Returns context package dict.
    """
    node = graph["nodes"].get(function_key, {})

    query = build_query(function_key, graph)

    # search vector store
    hits = search(
        query_text = query,
        top_k      = top_k,
        min_score  = min_score,
    )

    return {
        "function_key": function_key,
        "file":         node.get("file", ""),
        "class":        node.get("class", ""),
        "method":       node.get("method", ""),
        "summary":      query,
        "requirements": hits,
        "ast_hash":     node.get("ast_hash", ""),
    }


# ── build full context for dirty set ─────────────────────────────────────────

def build_context(
    dirty_set:  list[dict],
    graph:      dict = None,
    min_score:  float = MIN_SCORE,
    top_k:      int   = TOP_K,
) -> list[dict]:
    """
    Build context packages for all functions in the dirty set.

    Skips:
      - private helpers with no matching requirements
      - depth > 1 functions that share requirements with depth 0

    Returns list of context packages, deduplicated by requirement.
    """
    if graph is None:
        graph = load_graph()

    print(f"\n  Building context for {len(dirty_set)} dirty functions...\n")

    results           = []
    seen_req_ids      = set()   # track which requirements already covered

    for item in dirty_set:
        key      = item["key"]
        depth    = item["depth"]
        reason   = item["reason"]
        priority = item["priority"]

        node = graph["nodes"].get(key, {})
        method = node.get("method", "")

        print(f"  {key:<50} depth={depth}", end=" ... ", flush=True)

        # skip class-level nodes (no method) — we test methods not classes
        if not method:
            print("skip (class node)")
            continue

        # skip dunder methods
        if method.startswith("__"):
            print("skip (dunder)")
            continue

        context = find_requirements(
            key,
            graph,
            top_k     = top_k,
            min_score = min_score,
        )

        reqs = context["requirements"]

        if not reqs:
            print("no requirements matched")
            continue

        # filter out requirements already covered by higher-priority functions
        new_reqs = [r for r in reqs if r["id"] not in seen_req_ids]

        if not new_reqs and depth > 0:
            print(f"skip (all {len(reqs)} reqs already covered)")
            continue

        # for depth > 1 — only include requirements not seen at depth 0/1
        if depth > 1:
            reqs = new_reqs

        # mark requirements as seen
        for r in reqs:
            seen_req_ids.add(r["id"])

        context["requirements"] = reqs
        context["depth"]        = depth
        context["reason"]       = reason
        context["priority"]     = priority

        results.append(context)
        print(
            f"matched {len(reqs)} requirement(s) "
            f"[scores: {', '.join(format(r['score'], '.2f') for r in reqs[:3])}]"
        )

    print(f"\n  Context built:")
    print(f"    Functions with matched requirements : {len(results)}")
    print(f"    Unique requirements covered         : {len(seen_req_ids)}")

    return results


# ── save context ──────────────────────────────────────────────────────────────

def save_context(
    context:    list[dict],
    path:       str = "data/test_context.json",
):
    Path(path).write_text(
        json.dumps(context, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  Saved → {path}")


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ci.graph_builder import get_dirty_set

    # default: simulate a change to KlarfReaderNode.execute
    changed_key = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "KlarfReaderNode.execute"
    )

    print(f"\n{'='*55}")
    print(f"  Context Builder")
    print(f"  Simulating change: {changed_key}")
    print(f"{'='*55}")

    graph = load_graph()

    if not graph["nodes"]:
        print("  Graph is empty. Run graph_builder.py first.")
        raise SystemExit(1)

    # get dirty set
    dirty = get_dirty_set({changed_key}, graph=graph)

    print(f"\n  Dirty set: {len(dirty)} functions")
    for item in dirty:
        print(f"    depth={item['depth']} {item['key']}")

    # build context
    context = build_context(dirty, graph=graph)

    # save
    save_context(context)

    # print summary
    print(f"\n{'='*55}")
    print(f"  Results")
    print(f"{'='*55}\n")

    for ctx in context:
        print(f"  {ctx['function_key']}")
        print(f"  depth={ctx['depth']}  reason={ctx['reason']}")
        if ctx["requirements"]:
            for r in ctx["requirements"]:
                print(f"    [{r['score']:.3f}] {r['id']:10} {r['title'][:50]}")
        print()