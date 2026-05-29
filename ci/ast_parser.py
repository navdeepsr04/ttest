# ci/ast_parser.py
"""
Parses a single .py file from the KAI codebase.
Extracts structured metadata — no source code stored.

Output per file:
{
  "file":     "nodes/klarf_reader/klarf_reader.py",
  "classes": [
    {
      "name":       "KlarfReaderSettings",
      "type":       "settings",          # settings | node | other
      "bases":      ["NodeSetting"],
      "fields":     [{"name": "input_file_path", "type": "str", "default": ""}],
      "validators": ["validate_klarf_path"],
      "properties": ["klarf_path", "generate_tables"],
      "methods":    [],
      "ast_hash":   "a3f9c2d1"
    },
    {
      "name":       "KlarfReaderNode",
      "type":       "node",
      "bases":      ["Node"],
      "fields":     [],
      "validators": [],
      "properties": [],
      "methods": [
        {
          "name":       "execute",
          "params":     ["node_setting: dict[str, Any]"],
          "returns":    "None",
          "docstring":  "Execute the node to read a KLARF file...",
          "calls":      ["_download_from_remote", "_add_table_to_repo"],
          "ast_hash":   "b7e4a1f2",
          "is_private": false,
          "is_static":  false,
          "lineno":     98
        }
      ],
      "ast_hash": "c9d2e3f4"
    }
  ]
}
"""

import ast
import os
import hashlib
import json
from pathlib import Path


# ── class type detection ──────────────────────────────────────────────────────

# base classes that identify Settings classes
SETTINGS_BASES = {"NodeSetting", "BaseModel", "BaseSettings"}

# base classes that identify Node classes
NODE_BASES = {"Node"}

# methods we always want to capture fully
KEY_METHODS = {"execute", "run", "process", "validate", "__init__"}


# ── AST hash ──────────────────────────────────────────────────────────────────

def hash_node(node: ast.AST) -> str:
    """
    Produce a short hash of an AST node.
    Used to detect changes between PR runs.
    Same code → same hash. Any change → different hash.
    """
    try:
        source = ast.dump(node, annotate_fields=True)
        return hashlib.md5(source.encode()).hexdigest()[:8]
    except Exception:
        return "00000000"


# ── annotation to string ──────────────────────────────────────────────────────

def annotation_to_str(annotation) -> str:
    """Convert an AST annotation node to a readable string."""
    if annotation is None:
        return "Any"
    try:
        return ast.unparse(annotation)
    except Exception:
        return "Any"


# ── extract function calls ────────────────────────────────────────────────────

def extract_calls(func_node: ast.FunctionDef) -> list[str]:
    """
    Find all function/method calls inside a function body.
    Returns only self.xxx() calls — direct method calls within the class.
    """
    calls = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            # self.method_name()
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"):
                calls.append(node.func.attr)
            # ClassName.method() — static calls
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    # deduplicate, preserve order
    seen = set()
    unique = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ── extract method ────────────────────────────────────────────────────────────

def extract_method(node: ast.FunctionDef) -> dict:
    """Extract metadata from a single method/function."""

    # parameters (skip 'self')
    params = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        ann = annotation_to_str(arg.annotation)
        params.append(f"{arg.arg}: {ann}" if ann != "Any" else arg.arg)

    # defaults for params
    defaults = [ast.unparse(d) for d in node.args.defaults]
    if defaults:
        # align defaults to the end of params list
        offset = len(params) - len(defaults)
        for i, default in enumerate(defaults):
            idx = offset + i
            if 0 <= idx < len(params):
                params[idx] = f"{params[idx]} = {default}"

    # return type
    returns = annotation_to_str(node.returns)

    # docstring — first line only for compactness
    docstring = ""
    if (node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)):
        full_doc = node.body[0].value.value.strip()
        docstring = full_doc.split("\n")[0].strip()

    # decorators
    decorators = []
    for dec in node.decorator_list:
        try:
            decorators.append(ast.unparse(dec))
        except Exception:
            pass

    is_static   = "staticmethod" in decorators
    is_property = "property" in decorators
    is_validator = any("validator" in d.lower() for d in decorators)
    is_private  = node.name.startswith("_") and not node.name.startswith("__")
    is_dunder   = node.name.startswith("__") and node.name.endswith("__")

    return {
        "name":         node.name,
        "params":       params,
        "returns":      returns,
        "docstring":    docstring,
        "calls":        extract_calls(node),
        "decorators":   decorators,
        "is_static":    is_static,
        "is_property":  is_property,
        "is_validator": is_validator,
        "is_private":   is_private,
        "is_dunder":    is_dunder,
        "ast_hash":     hash_node(node),
        "lineno":       node.lineno,
    }


# ── extract class field ───────────────────────────────────────────────────────

def extract_fields(class_node: ast.ClassDef) -> list[dict]:
    """
    Extract Pydantic-style field declarations from a class body.
    Handles:   field_name: type = default
    """
    fields = []

    for node in class_node.body:
        # annotated assignment: field: type = default
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name    = node.target.id
            type_   = annotation_to_str(node.annotation)
            default = ast.unparse(node.value) if node.value else None

            # skip dunder and private class vars
            if name.startswith("__"):
                continue

            fields.append({
                "name":    name,
                "type":    type_,
                "default": default,
            })

    return fields


# ── detect class type ─────────────────────────────────────────────────────────

def detect_class_type(class_node: ast.ClassDef) -> str:
    """
    Classify a class as: settings | node | other
    Based on base class names.
    """
    bases = []
    for base in class_node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            pass

    for base in bases:
        # strip module prefix: kai.sdk.Node → Node
        short = base.split(".")[-1]
        if short in SETTINGS_BASES:
            return "settings"
        if short in NODE_BASES:
            return "node"

    return "other"


# ── extract class ─────────────────────────────────────────────────────────────

def extract_class(class_node: ast.ClassDef) -> dict:
    """Extract full metadata from a class definition."""

    # base classes
    bases = []
    for base in class_node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            pass

    class_type = detect_class_type(class_node)
    fields     = extract_fields(class_node) if class_type == "settings" else []
    methods    = []
    validators = []
    properties = []

    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        method = extract_method(node)

        if method["is_validator"]:
            validators.append(method["name"])
        elif method["is_property"]:
            properties.append(method["name"])
        elif not method["is_dunder"] or node.name in KEY_METHODS:
            methods.append(method)

    return {
        "name":       class_node.name,
        "type":       class_type,
        "bases":      bases,
        "fields":     fields,
        "validators": validators,
        "properties": properties,
        "methods":    methods,
        "ast_hash":   hash_node(class_node),
        "lineno":     class_node.lineno,
    }


# ── parse file ────────────────────────────────────────────────────────────────

def parse_file(filepath: str) -> dict | None:
    """
    Parse a single .py file and return structured metadata.

    Returns None if file cannot be parsed or has no relevant classes.
    """
    path = Path(filepath)

    if not path.exists():
        return None

    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  ⚠ Parse error in {filepath}: {e}")
        return None

    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            extracted = extract_class(node)
            # only keep settings, node, or classes with execute/run methods
            if extracted["type"] in ("settings", "node"):
                classes.append(extracted)
            elif any(m["name"] in KEY_METHODS for m in extracted["methods"]):
                classes.append(extracted)

    if not classes:
        return None

    # file-level hash — hash of all class hashes combined
    combined = "".join(c["ast_hash"] for c in classes)
    file_hash = hashlib.md5(combined.encode()).hexdigest()[:8]

    return {
        "file":      str(path),
        "file_hash": file_hash,
        "classes":   classes,
    }


# ── build summary string (for vector search) ─────────────────────────────────

def build_summary(parsed: dict) -> str:
    """
    Build a compact text summary of a parsed file.
    This is what gets embedded and searched against requirements.
    """
    parts = []

    for cls in parsed["classes"]:
        cls_line = f"class {cls['name']}({', '.join(cls['bases'])})"
        parts.append(cls_line)

        # settings fields
        if cls["fields"]:
            field_strs = [
                f"{f['name']}: {f['type']}"
                + (f" = {f['default']}" if f["default"] else "")
                for f in cls["fields"]
            ]
            parts.append(f"  fields: {', '.join(field_strs)}")

        # validation rules
        if cls["validators"]:
            parts.append(f"  validators: {', '.join(cls['validators'])}")

        # properties
        if cls["properties"]:
            parts.append(f"  properties: {', '.join(cls['properties'])}")

        # methods — include docstring for execute
        for m in cls["methods"]:
            if m["name"] == "__init__":
                continue
            param_str  = ", ".join(m["params"])
            method_line = f"  def {m['name']}({param_str}) -> {m['returns']}"
            if m["docstring"]:
                method_line += f"  # {m['docstring'][:80]}"
            parts.append(method_line)

            if m["calls"]:
                parts.append(f"    calls: {', '.join(m['calls'][:5])}")

    return "\n".join(parts)

# ci/ast_parser.py  — add after build_summary()

# folders and files to skip inside nodes/
SKIP_FOLDERS = {"__pycache__", "poc", "distributed_computing", "temp", ".windsurf"}
SKIP_FILES   = {"__init__.py", "conftest.py"}


def parse_folder(folder_path: str) -> list[dict]:
    """
    Recursively parse all .py files in a folder.
    Skips irrelevant files and folders.
    Returns list of parsed file results.
    """
    results = []
    folder  = Path(folder_path)

    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return results

    for root, dirs, files in os.walk(folder):
        # skip unwanted folders in-place
        dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS]

        for filename in files:
            if filename in SKIP_FILES:
                continue
            if not filename.endswith(".py"):
                continue

            filepath = str(Path(root) / filename)
            parsed   = parse_file(filepath)

            if parsed:
                results.append(parsed)

    return results


def parse_folder_summary(results: list[dict]) -> None:
    """Print summary of folder parse results."""
    total_classes = sum(len(r["classes"]) for r in results)
    nodes         = sum(1 for r in results
                        for c in r["classes"] if c["type"] == "node")
    settings      = sum(1 for r in results
                        for c in r["classes"] if c["type"] == "settings")

    print(f"\n{'='*50}")
    print(f"  Folder parse summary")
    print(f"{'='*50}")
    print(f"  Files parsed    : {len(results)}")
    print(f"  Total classes   : {total_classes}")
    print(f"    Node classes  : {nodes}")
    print(f"    Settings      : {settings}")
    print(f"{'='*50}\n")

    for r in results:
        classes_str = ", ".join(
            f"{c['name']}({c['type']})" for c in r["classes"]
        )
        rel_path = Path(r["file"]).name
        print(f"  {rel_path:<40} {classes_str}")


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  single file : python -m ci.ast_parser <file.py>")
        print("  folder      : python -m ci.ast_parser <folder/>")
        raise SystemExit(1)

    target = sys.argv[1]

    if Path(target).is_dir():
        print(f"Scanning folder: {target}\n")
        results = parse_folder(target)
        parse_folder_summary(results)

        # save to data/parsed_nodes.json
        out = Path("data/parsed_nodes.json")
        out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\nSaved → {out}")

    else:
        result = parse_file(target)
        if result is None:
            print("No relevant classes found.")
            raise SystemExit(0)
        print(json.dumps(result, indent=2))