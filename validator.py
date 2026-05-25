# validator.py
"""
Validates LLM-generated requirement objects.

Three levels of checks:
  1. Schema validation   — correct fields, correct types
  2. Sanity checks       — meaningful content, no empty critical fields
  3. Tautology detection — weak assertions in test_hints
"""

import re

# ── constants ─────────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"functional", "non_functional", "ui", "api", "validation"}
VALID_PRIORITIES = {"high", "medium", "low"}

REQUIRED_FIELDS  = [
    "category", "title", "module", "priority",
    "description", "inputs", "expected_outputs",
    "error_cases", "business_rules", "test_hints"
]

# default values used when a field is missing
FIELD_DEFAULTS = {
    "category":        "functional",
    "title":           "Untitled Requirement",
    "module":          "Unknown",
    "priority":        "medium",
    "description":     "",
    "inputs":          [],
    "expected_outputs":[],
    "error_cases":     [],
    "business_rules":  [],
    "test_hints":      [],
}

# patterns that indicate a weak / tautological test hint
WEAK_HINT_PATTERNS = [
    r"assert True",
    r"assert False",
    r"assert .+ == .+\1",          # assert x == x
    r"assert len\(.+\) >= 0",      # always true
    r"assert .+ is not None",      # too generic
    r"assert response$",           # no actual value checked
]


# ── individual field validators ───────────────────────────────────────────────

def validate_string_field(value, field_name: str) -> tuple[str, list[str]]:
    """Ensure a field is a non-empty string."""
    warnings = []
    if not isinstance(value, str):
        warnings.append(f"'{field_name}' should be a string, got {type(value).__name__} — converted")
        value = str(value) if value is not None else ""
    if not value.strip():
        warnings.append(f"'{field_name}' is empty")
    return value, warnings


def validate_list_field(value, field_name: str) -> tuple[list, list[str]]:
    """Ensure a field is a list."""
    warnings = []
    if not isinstance(value, list):
        warnings.append(f"'{field_name}' should be a list, got {type(value).__name__} — wrapped")
        value = [value] if value is not None else []
    return value, warnings


def validate_category(value: str) -> tuple[str, list[str]]:
    """Ensure category is one of the known values."""
    warnings = []
    if not isinstance(value, str):
        return "functional", [f"category is not a string — defaulted to 'functional'"]
    val = value.lower().strip()
    if val not in VALID_CATEGORIES:
        warnings.append(
            f"unknown category '{val}' — defaulted to 'functional'. "
            f"Valid: {VALID_CATEGORIES}"
        )
        val = "functional"
    return val, warnings


def validate_priority(value: str) -> tuple[str, list[str]]:
    """Ensure priority is high / medium / low."""
    warnings = []
    if not isinstance(value, str):
        return "medium", [f"priority is not a string — defaulted to 'medium'"]
    val = value.lower().strip()
    if val not in VALID_PRIORITIES:
        warnings.append(
            f"unknown priority '{val}' — defaulted to 'medium'. "
            f"Valid: {VALID_PRIORITIES}"
        )
        val = "medium"
    return val, warnings


def validate_inputs(inputs: list) -> tuple[list, list[str]]:
    """Validate each input parameter object."""
    warnings = []
    cleaned  = []

    for i, inp in enumerate(inputs):
        if not isinstance(inp, dict):
            warnings.append(f"inputs[{i}] is not a dict — skipped")
            continue

        clean = {}

        # name
        clean["name"] = inp.get("name") or f"param_{i}"
        if not inp.get("name"):
            warnings.append(f"inputs[{i}] missing 'name' — defaulted to '{clean['name']}'")

        # type
        valid_types = {"int", "float", "string", "list", "bool", "dict"}
        raw_type    = str(inp.get("type", "string")).lower()
        clean["type"] = raw_type if raw_type in valid_types else "string"
        if raw_type not in valid_types:
            warnings.append(f"inputs[{i}] unknown type '{raw_type}' — defaulted to 'string'")

        # required
        req = inp.get("required", True)
        clean["required"] = bool(req) if isinstance(req, (bool, int)) else True

        # default — can be null
        clean["default"] = inp.get("default", None)

        # validation — can be null
        clean["validation"] = inp.get("validation", None)

        cleaned.append(clean)

    return cleaned, warnings


def validate_error_cases(error_cases: list) -> tuple[list, list[str]]:
    """Validate each error case object."""
    warnings = []
    cleaned  = []

    for i, ec in enumerate(error_cases):
        if not isinstance(ec, dict):
            warnings.append(f"error_cases[{i}] is not a dict — skipped")
            continue

        clean = {}
        clean["trigger"] = ec.get("trigger") or "Unknown trigger"

        # expected_status should be an int like 400, 422, 401
        status = ec.get("expected_status", 400)
        try:
            clean["expected_status"] = int(status)
        except (TypeError, ValueError):
            clean["expected_status"] = 400
            warnings.append(f"error_cases[{i}] invalid status '{status}' — defaulted to 400")

        clean["expected_message"] = ec.get("expected_message", None)

        cleaned.append(clean)

    return cleaned, warnings


def detect_weak_hints(test_hints: list) -> list[str]:
    """Return any hints that look tautological or too vague."""
    weak = []
    for hint in test_hints:
        if not isinstance(hint, str):
            continue
        for pattern in WEAK_HINT_PATTERNS:
            if re.search(pattern, hint, re.IGNORECASE):
                weak.append(hint)
                break
        # also flag very short hints — likely uninformative
        if len(hint.strip()) < 10:
            weak.append(hint)
    return list(set(weak))


# ── sanity checks ─────────────────────────────────────────────────────────────

def sanity_check(req: dict) -> list[str]:
    """
    Higher-level checks on the requirement as a whole.
    Returns list of warning strings.
    """
    warnings = []

    # description should be substantial
    desc = req.get("description", "")
    if len(desc.strip()) < 30:
        warnings.append(
            "description is very short — LLM may not have extracted enough detail"
        )

    # functional and validation requirements should have error_cases
    cat = req.get("category", "")
    if cat in ("functional", "validation") and not req.get("error_cases"):
        warnings.append(
            "no error_cases defined for a functional/validation requirement — "
            "consider adding failure scenarios"
        )

    # api requirements should have an endpoint
    if cat == "api" and not req.get("endpoint"):
        warnings.append(
            "API requirement has no endpoint defined — "
            "add the HTTP method and path"
        )

    # should have at least one test hint
    if not req.get("test_hints"):
        warnings.append(
            "no test_hints — add boundary values and edge cases manually"
        )

    # check for weak hints
    weak = detect_weak_hints(req.get("test_hints", []))
    if weak:
        warnings.append(
            f"weak/tautological test_hints detected: {weak}"
        )

    return warnings


# ── main validate function ────────────────────────────────────────────────────

def validate_requirement(req: dict) -> dict:
    """
    Validate and clean a single LLM-generated requirement.

    Returns the cleaned requirement with two extra fields:
      _warnings      : list of issues found (non-fatal)
      _needs_review  : True if human should check this before test generation
    """
    all_warnings = []

    # ── step 1: fill missing fields with defaults ──────────────────────────
    for field in REQUIRED_FIELDS:
        if field not in req or req[field] is None:
            req[field] = FIELD_DEFAULTS[field]
            all_warnings.append(f"missing field '{field}' — set to default")

    # ── step 2: validate individual fields ────────────────────────────────
    req["category"],  w = validate_category(req["category"])
    all_warnings.extend(w)

    req["priority"],  w = validate_priority(req["priority"])
    all_warnings.extend(w)

    req["title"],     w = validate_string_field(req["title"],       "title")
    all_warnings.extend(w)

    req["module"],    w = validate_string_field(req["module"],      "module")
    all_warnings.extend(w)

    req["description"], w = validate_string_field(req["description"], "description")
    all_warnings.extend(w)

    req["inputs"],    w = validate_inputs(req.get("inputs", []))
    all_warnings.extend(w)

    req["error_cases"], w = validate_error_cases(req.get("error_cases", []))
    all_warnings.extend(w)

    req["business_rules"], w = validate_list_field(req["business_rules"], "business_rules")
    all_warnings.extend(w)

    req["test_hints"], w = validate_list_field(req["test_hints"], "test_hints")
    all_warnings.extend(w)

    # ── step 3: sanity checks ──────────────────────────────────────────────
    sanity_warnings = sanity_check(req)
    all_warnings.extend(sanity_warnings)

    # ── step 4: mark for review if any warnings ────────────────────────────
    req["_warnings"]     = all_warnings
    req["_needs_review"] = len(all_warnings) > 0

    return req


def validate_all(requirements: list[dict]) -> tuple[list[dict], dict]:
    """
    Validate a list of requirements.

    Returns:
        cleaned     : list of validated requirement dicts
        stats       : summary counts
    """
    cleaned       = []
    total_warnings = 0
    needs_review  = 0

    for req in requirements:
        validated = validate_requirement(req)
        cleaned.append(validated)
        if validated["_warnings"]:
            total_warnings += len(validated["_warnings"])
        if validated["_needs_review"]:
            needs_review += 1

    stats = {
        "total":          len(cleaned),
        "needs_review":   needs_review,
        "clean":          len(cleaned) - needs_review,
        "total_warnings": total_warnings,
    }

    return cleaned, stats


def print_validation_summary(requirements: list[dict], stats: dict):
    print(f"\n{'='*55}")
    print(f"  Validation summary")
    print(f"{'='*55}")
    print(f"  Total requirements : {stats['total']}")
    print(f"  Clean              : {stats['clean']}")
    print(f"  Needs review       : {stats['needs_review']}")
    print(f"  Total warnings     : {stats['total_warnings']}")
    print(f"{'='*55}")

    if stats["needs_review"] > 0:
        print(f"\n  Requirements needing review:\n")
        for req in requirements:
            if req.get("_needs_review"):
                print(f"  [{req.get('id','?')}] {req.get('title','')}")
                for w in req["_warnings"]:
                    print(f"         ⚠  {w}")
                print()


# ── run standalone to validate existing requirements.json ────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path

    path = Path("data/requirements.json")
    if not path.exists():
        print("data/requirements.json not found — run extractor.py first")
        raise SystemExit(1)

    requirements = json.loads(path.read_text(encoding="utf-8"))
    print(f"Loaded {len(requirements)} requirements")

    cleaned, stats = validate_all(requirements)
    print_validation_summary(cleaned, stats)

    # save cleaned version back
    path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nSaved cleaned requirements → {path}")