# test_generator.py
"""
Generates pytest test files from validated requirements.

Guards applied to every generated file:
  Guard 1 — AST syntax check
  Guard 2 — pytest --collect-only check
  Guard 3 — weak assertion detection

Output folders:
  generated_tests/   clean files, all guards passed
  review_needed/     generated but has weak assertions
  failed/            broken syntax or collection failure
"""

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


# ── folders ───────────────────────────────────────────────────────────────────

GENERATED_DIR   = Path("generated_tests")
REVIEW_DIR      = Path("review_needed")
FAILED_DIR      = Path("failed")

for d in [GENERATED_DIR, REVIEW_DIR, FAILED_DIR]:
    d.mkdir(exist_ok=True)


# ── test generation prompt ────────────────────────────────────────────────────

TEST_SYSTEM_PROMPT = """
You are a senior Python test engineer writing pytest tests for a
FastAPI backend of an analytical software platform for chip
manufacturing defect analysis.

Given a structured requirement object, generate a complete pytest
test file.

STRICT RULES:
  1. Use httpx.AsyncClient for all HTTP calls.
  2. Use @pytest.mark.asyncio on every async test function.
  3. Import the FastAPI app as:
       from app.main import app
  4. Use this fixture at the top of every file:
       @pytest.fixture
       async def client():
           async with AsyncClient(app=app, base_url="http://test") as c:
               yield c
  5. Test the happy path AND every error_case in the requirement.
  6. Use @pytest.mark.parametrize for multiple similar inputs.
  7. Mock external dependencies (DB, file I/O) with pytest-mock.
  8. Every test function name: test_<what_it_tests>
  9. Every test function has a one-line docstring.
  10. Assertions must check SPECIFIC values, not just existence:
        WRONG : assert response.status_code
        RIGHT : assert response.status_code == 200
        WRONG : assert "tsne_x" in response.json()
        RIGHT : assert isinstance(response.json()["tsne_x"], list)
  11. Return ONLY raw Python code. No explanation.
      No markdown fences. No ```python. Just the code.
"""


def build_test_prompt(req: dict) -> str:
    return f"""Generate pytest tests for this requirement:

{json.dumps(req, indent=2)}

Remember:
- One test per error_case listed above
- Use the test_hints to add boundary/edge case tests
- All assertions must check specific values, not just existence
"""


# ── guards ────────────────────────────────────────────────────────────────────

def guard_syntax(code: str) -> tuple[bool, str]:
    """
    Guard 1 — AST parse check.
    Returns (passed, error_message).
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


def guard_collection(filepath: Path) -> tuple[bool, str]:
    """
    Guard 2 — pytest --collect-only check.
    Runs pytest in collection mode — no tests are executed.
    Returns (passed, error_output).
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(filepath),
         "--collect-only", "-q", "--no-header"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return False, result.stdout + result.stderr
    return True, ""


# patterns for weak assertions
WEAK_PATTERNS = [
    (r"assert True\b",              "assert True — always passes"),
    (r"assert False\b",             "assert False — always fails"),
    (r"assert len\(.+\)\s*>=\s*0",  "len >= 0 — always true"),
    (r"assert .+\s*is not None\b",  "is not None — too generic"),
    (r"assert response\.status_code[^=!<>]",
                                    "status_code not compared to a value"),
    (r"assert response\s*$",        "asserting response object itself"),
]

def guard_weak_assertions(code: str) -> list[str]:
    """
    Guard 3 — detect weak / tautological assertions.
    Returns list of problem descriptions found.
    """
    problems = []
    for pattern, description in WEAK_PATTERNS:
        if re.search(pattern, code):
            problems.append(description)
    return problems


# ── retry logic ───────────────────────────────────────────────────────────────

def call_llm_with_retry(
    messages: list[dict],
    max_retries: int = 3,
    timeout: int = 45,
) -> str | None:
    """
    Call GPT-4o with retry + exponential backoff.
    Returns the response text, or None if all retries fail.
    """
    wait_times = [0, 10, 20]   # seconds before each attempt

    for attempt in range(max_retries):
        wait = wait_times[attempt]
        if wait > 0:
            print(f"    retrying in {wait}s (attempt {attempt+1}/{max_retries})...",
                  end=" ", flush=True)
            time.sleep(wait)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=messages,
                timeout=timeout,
            )
            return response.choices[0].message.content

        except Exception as e:
            error = str(e)

            if "rate_limit" in error.lower():
                print(f"rate limit hit —", end=" ", flush=True)
                continue

            if "timeout" in error.lower():
                print(f"timeout —", end=" ", flush=True)
                continue

            # unexpected error — stop retrying
            print(f"unexpected error: {e}")
            return None

    return None


def fix_syntax(code: str, error_msg: str) -> str | None:
    """
    Ask GPT-4o to fix a syntax error.
    One attempt only.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a Python expert. Fix the syntax error in the "
                       "code below. Return ONLY the corrected code, no explanation."
        },
        {
            "role": "user",
            "content": f"Syntax error: {error_msg}\n\nCode:\n{code}"
        }
    ]
    return call_llm_with_retry(messages, max_retries=1)


# ── clean generated code ──────────────────────────────────────────────────────

def clean_code(raw: str) -> str:
    """Remove markdown fences GPT sometimes wraps code in."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",           "", raw)
    return raw.strip()


# ── generate one test file ────────────────────────────────────────────────────

def generate_test_file(req: dict) -> dict:
    """
    Generate a pytest file for one requirement.

    Returns a result dict:
    {
        "id":       requirement id,
        "status":   "generated" | "review_needed" | "failed",
        "filepath": path where file was saved,
        "reason":   why it was flagged / failed (if applicable)
    }
    """
    req_id   = req.get("id", "REQ-000")
    title    = req.get("title", "untitled")

    # make a safe filename
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:40].strip("_")
    filename = f"test_{req_id.lower().replace('-', '_')}_{slug}.py"

    print(f"  [{req_id}] {title[:50]}", end=" ... ", flush=True)

    # ── call LLM ──────────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": TEST_SYSTEM_PROMPT},
        {"role": "user",   "content": build_test_prompt(req)},
    ]

    raw = call_llm_with_retry(messages)

    if raw is None:
        print("FAILED (no response after retries)")
        filepath = FAILED_DIR / filename
        filepath.write_text(
            f"# FAILED TO GENERATE\n# Requirement: {req_id}\n# {title}\n",
            encoding="utf-8"
        )
        return {
            "id": req_id, "status": "failed",
            "filepath": str(filepath),
            "reason": "no response from LLM after retries"
        }

    code = clean_code(raw)

    # ── Guard 1: syntax check ──────────────────────────────────────────────
    syntax_ok, syntax_error = guard_syntax(code)

    if not syntax_ok:
        print(f"syntax error — attempting fix ...", end=" ", flush=True)
        fixed = fix_syntax(code, syntax_error)

        if fixed:
            code = clean_code(fixed)
            syntax_ok, syntax_error = guard_syntax(code)

        if not syntax_ok:
            print("FAILED (syntax)")
            filepath = FAILED_DIR / filename
            filepath.write_text(
                f"# SYNTAX ERROR — needs manual fix\n"
                f"# {syntax_error}\n\n{code}",
                encoding="utf-8"
            )
            return {
                "id": req_id, "status": "failed",
                "filepath": str(filepath),
                "reason": f"syntax error: {syntax_error}"
            }

    # ── Guard 2: pytest collection check ──────────────────────────────────
    # write to a temp location first
    temp_path = FAILED_DIR / f"_temp_{filename}"
    temp_path.write_text(code, encoding="utf-8")

    collection_ok, collection_error = guard_collection(temp_path)
    temp_path.unlink(missing_ok=True)   # delete temp file

    if not collection_ok:
        print("FAILED (collection)")
        filepath = FAILED_DIR / filename
        filepath.write_text(
            f"# COLLECTION ERROR — needs manual fix\n"
            f"# {collection_error[:300]}\n\n{code}",
            encoding="utf-8"
        )
        return {
            "id": req_id, "status": "failed",
            "filepath": str(filepath),
            "reason": f"pytest collection failed: {collection_error[:200]}"
        }

    # ── Guard 3: weak assertion check ─────────────────────────────────────
    weak = guard_weak_assertions(code)

    if weak:
        print(f"REVIEW (weak assertions: {len(weak)})")
        filepath = REVIEW_DIR / filename
        warning_comment = "\n".join(f"# ⚠  {w}" for w in weak)
        filepath.write_text(
            f"# REVIEW NEEDED — weak assertions detected\n"
            f"{warning_comment}\n\n{code}",
            encoding="utf-8"
        )
        return {
            "id": req_id, "status": "review_needed",
            "filepath": str(filepath),
            "reason": f"weak assertions: {weak}"
        }

    # ── all guards passed ──────────────────────────────────────────────────
    print("OK")
    filepath = GENERATED_DIR / filename
    filepath.write_text(code, encoding="utf-8")
    return {
        "id": req_id, "status": "generated",
        "filepath": str(filepath),
        "reason": None
    }


# ── conftest.py ───────────────────────────────────────────────────────────────

CONFTEST = '''# conftest.py
# Shared fixtures for all generated tests.
# Update app import path to match your project.

import pytest
from httpx import AsyncClient
from app.main import app          # ← update this import


@pytest.fixture
async def client():
    """Async test client for FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c


@pytest.fixture
def sample_numeric_data():
    """Small numeric dataset for t-SNE / PCA tests."""
    return {
        "columns": ["xsize", "ysize", "dist"],
        "rows": [
            [1.2, 3.4, 5.6],
            [2.1, 4.3, 6.5],
            [3.0, 5.0, 7.0],
        ]
    }


@pytest.fixture
def large_numeric_data():
    """Larger dataset (500 rows) for performance / perplexity tests."""
    import random
    random.seed(42)
    return {
        "columns": ["xsize", "ysize", "dist"],
        "rows": [
            [random.uniform(0, 100) for _ in range(3)]
            for _ in range(500)
        ]
    }
'''

def write_conftest():
    path = GENERATED_DIR / "conftest.py"
    if not path.exists():
        path.write_text(CONFTEST, encoding="utf-8")
        print(f"  Created → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_generation(
    requirements_path: str = "data/requirements_clean.json",
    report_path:       str = "data/generation_report.json",
):
    if not Path(requirements_path).exists():
        print(f"{requirements_path} not found.")
        print("Run extractor.py first, then validator.py.")
        raise SystemExit(1)

    requirements = json.loads(
        Path(requirements_path).read_text(encoding="utf-8")
    )

    print(f"\n{'='*55}")
    print(f"  Test Generation — {len(requirements)} requirements")
    print(f"  Model: gpt-4o  |  Guards: syntax, collect, assertions")
    print(f"{'='*55}\n")

    write_conftest()
    print()

    results = []
    for req in requirements:
        result = generate_test_file(req)
        results.append(result)

    # ── summary ───────────────────────────────────────────────────────────
    generated    = [r for r in results if r["status"] == "generated"]
    review       = [r for r in results if r["status"] == "review_needed"]
    failed       = [r for r in results if r["status"] == "failed"]

    print(f"\n{'='*55}")
    print(f"  Generation complete")
    print(f"{'='*55}")
    print(f"  Generated     : {len(generated):>3}  → generated_tests/")
    print(f"  Review needed : {len(review):>3}  → review_needed/")
    print(f"  Failed        : {len(failed):>3}  → failed/")
    print(f"  Total         : {len(results):>3}")

    if failed:
        print(f"\n  Failed files (fix manually):")
        for r in failed:
            print(f"    [{r['id']}] {r['reason']}")

    if review:
        print(f"\n  Review needed (check assertions):")
        for r in review:
            print(f"    [{r['id']}] {r['reason']}")

    # ── save report ───────────────────────────────────────────────────────
    report = {
        "total":          len(results),
        "generated":      len(generated),
        "review_needed":  len(review),
        "failed":         len(failed),
        "results":        results,
    }
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n  Report → {report_path}")
    print(f"\n  Run tests:")
    print(f"    pytest generated_tests/ -v\n")


if __name__ == "__main__":
    run_generation()



# uv add pytest-asyncio