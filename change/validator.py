# validator.py
import json
from pathlib import Path

VALID_TYPES = {"happy_path", "error_path", "boundary", "output_check"}

REQUIRED    = ["id", "what", "input", "expect", "type"]


def fix_test_case(tc: dict, index: int) -> tuple[dict, list[str]]:
    """
    Fix obvious issues in a test case.
    Returns (fixed_tc, list_of_warnings).
    """
    warnings = []

    # id
    if not tc.get("id"):
        tc["id"] = f"TC-{index:03d}"

    # node default
    if not tc.get("node"):
        tc["node"]   = None
        tc["method"] = "execute"

    # what
    if not tc.get("what"):
        warnings.append("missing 'what'")
        tc["what"] = "unnamed test"

    if len(tc.get("what", "")) > 120:
        tc["what"] = tc["what"][:120]
        warnings.append("'what' truncated to 120 chars")

    # input
    if not isinstance(tc.get("input"), dict):
        tc["input"] = {}
        warnings.append("invalid 'input' — reset to {}")

    # expect
    if not isinstance(tc.get("expect"), dict):
        tc["expect"] = {"success": None, "error": None, "message": None}
        warnings.append("invalid 'expect' — reset")
    else:
        tc["expect"].setdefault("success", None)
        tc["expect"].setdefault("error",   None)
        tc["expect"].setdefault("message", None)

    # expect must have at least one of success/error
    exp = tc["expect"]
    if not exp.get("success") and not exp.get("error"):
        warnings.append("'expect' has neither success nor error defined")

    # type
    if tc.get("type") not in VALID_TYPES:
        tc["type"] = "happy_path"
        warnings.append(f"invalid type — defaulted to 'happy_path'")

    return tc, warnings


def validate_all(
    test_cases:  list[dict],
    strict:      bool = False,  # if True, remove cases with warnings
) -> tuple[list[dict], dict]:

    fixed   = []
    removed = []

    for i, tc in enumerate(test_cases, 1):
        tc, warnings = fix_test_case(tc, i)

        if warnings and strict:
            removed.append({"id": tc["id"], "warnings": warnings})
            continue

        if warnings:
            tc["_warnings"] = warnings

        fixed.append(tc)

    stats = {
        "total":   len(test_cases),
        "valid":   len(fixed),
        "removed": len(removed),
    }

    return fixed, stats


def print_summary(test_cases: list[dict], stats: dict):
    print(f"\n{'='*50}")
    print(f"  Validation")
    print(f"{'='*50}")
    print(f"  Total   : {stats['total']}")
    print(f"  Valid   : {stats['valid']}")
    print(f"  Removed : {stats['removed']}")

    warned = [tc for tc in test_cases if tc.get("_warnings")]
    if warned:
        print(f"\n  Cases with warnings ({len(warned)}):")
        for tc in warned[:5]:
            print(f"    [{tc['id']}] {tc['what'][:50]}")
            for w in tc["_warnings"]:
                print(f"           ⚠ {w}")


if __name__ == "__main__":
    path = Path("data/test_cases.json")
    if not path.exists():
        print("Run extractor.py first.")
        raise SystemExit(1)

    test_cases = json.loads(path.read_text(encoding="utf-8"))
    fixed, stats = validate_all(test_cases)
    print_summary(fixed, stats)

    path.write_text(
        json.dumps(fixed, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n  Saved → {path}")